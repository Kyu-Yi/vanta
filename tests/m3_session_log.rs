use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::Duration;

use tempfile::TempDir;
use vanta::core::{EcrEirSnapshot, EvidenceSource, FailureClass, TestIntent, TestResult};
use vanta::session::{PatchAttemptId, SessionId, SessionRecord, TransitionRow, ecr_eir_from_rows, SCHEMA_VERSION};

/// These tests each spawn a `vanta_m3` process that boots sandboxes and runs
/// real pytest. Run concurrently they saturate the machine, and the short
/// in-process test timeout starts tripping on trivial baseline runs — which
/// turns a real behaviour into a flaky failure. They therefore take a
/// process-wide lock and run one at a time; a poisoned lock (a prior test
/// panicked) still lets the rest run, so one real failure is not multiplied.
static SERIAL_RUN: Mutex<()> = Mutex::new(());

fn serial_run() -> std::sync::MutexGuard<'static, ()> {
    SERIAL_RUN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Block until the client closes `stream`, then bump `disconnects`.
///
/// [VANTA-WHY] `Ok(0)` is a clean FIN: the client hung up, which is exactly the
/// cancel the idle timeout is supposed to send. A read error (RST) means the
/// same thing. `WouldBlock` is not a disconnect — the accepted socket may be
/// non-blocking — so it polls until a deadline rather than being mistaken for
/// one.
fn wait_for_disconnect(stream: &mut std::net::TcpStream, disconnects: &AtomicUsize) {
    let mut scratch = [0_u8; 256];
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match stream.read(&mut scratch) {
            Ok(0) => {
                disconnects.fetch_add(1, Ordering::SeqCst);
                return;
            }
            Ok(_) => continue,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => {
                disconnects.fetch_add(1, Ordering::SeqCst);
                return;
            }
        }
    }
}

/// How the stub behaves on a chat-completion request.
///
/// [VANTA-WHY] `stream: true` means a liveness bug only shows up over TIME, so
/// the stub must be able to model time: answer promptly (Normal), emit one token
/// and then go silent (`Stall`), or refuse the call outright (`Fail`).
#[derive(Clone, Copy)]
enum StubBehaviour {
    /// Answer every chat call with a complete SSE stream (the old JSON answer,
    /// re-framed as events). Used by every pre-existing test.
    Normal,
    /// Emit one `content` event and then send nothing more, holding the socket
    /// open. The client must idle-time-out and close the connection; the stub
    /// records that disconnection.
    Stall,
    /// Answer every chat call with HTTP 500, so the client's retry loop burns
    /// its attempts and the run ends as an infrastructure failure.
    Fail,
    /// Answer every chat call with HTTP 401 whose body ECHOES the bearer token
    /// the client sent — the shape a real provider uses when it rejects a key
    /// ("Incorrect API key provided: sk-…"). This is the harshest case for the
    /// redaction rule: the credential arrives inside a body that Vanta is about
    /// to print, so a rule that only knew about variable NAMES would leak it.
    EchoKeyUnauthorized,
}

struct StubLlmServer {
    base_url: String,
    request_count: Arc<AtomicUsize>,
    request_kinds: Arc<Mutex<Vec<String>>>,
    request_bodies: Arc<Mutex<Vec<String>>>,
    /// Number of stalled chat connections the client closed on us. It is how a
    /// test proves the cancel reached the SERVER (not just the client's future).
    disconnects: Arc<AtomicUsize>,
    /// When set, a BATCHED intent-audit request is answered with a JSON answer
    /// CUT OFF mid-string and a tail frame carrying `finish_reason: "length"` —
    /// the shape a server publishes when the generation hit `max_tokens`. Set
    /// after `start` (the answer depends on the request, which does not exist
    /// yet at construction time) and read per request, so it can be flipped
    /// between rounds. Per-test audit requests are answered normally, which is
    /// what lets a test prove the per-test fallback RECOVERS the round (§14.5).
    truncate_batch_audit: Arc<AtomicBool>,
    /// Request kinds still to be answered with a reply CUT OFF at the output cap
    /// (`finish_reason: "length"`), consumed one per matching request.
    ///
    /// [VANTA-WHY] §14.5 — a ONE-SHOT cut-off is what lets a test tell "the
    /// caller retried and recovered" from "the caller gave up after one try":
    /// the first matching request is answered with a reply no parser can read,
    /// and the retry is answered normally. The `truncate_batch_audit` flag above
    /// cannot express this — it truncates EVERY batch call, which is right for
    /// the temp-0.0 auditor (whose retries are deterministic) and wrong for a
    /// call site that is supposed to recover on the retry.
    truncate_next: Arc<Mutex<Vec<String>>>,
    /// Request kinds still to be answered with a 429 "the provider is busy"
    /// refusal, consumed one per matching request (§14.30).
    ///
    /// [VANTA-WHY] §14.30 — a ONE-SHOT refusal is what lets a test tell "the
    /// caller re-sent the SAME request and the model answered once" from "the
    /// caller counted a throttle as a model attempt and the model answered
    /// twice". The body carries `Resource Unavailable`, which is the shape the
    /// hosted API uses when a `flex` tier has no capacity, so the run records the
    /// refusal under `resource_unavailable` rather than the generic rate limit.
    busy_next: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl StubLlmServer {
    fn start(attacker_response_json: &str) -> Self {
        Self::start_scripted(None, attacker_response_json, None, None, None, None, None)
    }

    fn start_with_creator_and_attacker(
        creator_response_json: &str,
        attacker_response_json: &str,
    ) -> Self {
        Self::start_scripted(
            Some(creator_response_json),
            attacker_response_json,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// As `start_with_creator_and_attacker`, but `creator_responses_json` is a
    /// JSON ARRAY of Creator answers served one per Creator call.
    ///
    /// [VANTA-WHY] A repair-mode regression needs the Creator to get it WRONG
    /// once and RIGHT afterwards — that is the whole point of feeding the failure
    /// back — and a stub that answers every call identically cannot express it.
    /// The last element repeats if the loop asks for more, so the caller only
    /// scripts the calls it cares about.
    fn start_with_creator_script(
        creator_responses_json: &str,
        attacker_response_json: &str,
    ) -> Self {
        Self::start_scripted(
            Some(creator_responses_json),
            attacker_response_json,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// As `start`, but the patch-blind intent audit answers with a scripted
    /// verdict instead of the default "the issue asks for this behaviour".
    fn start_with_intent_audit(attacker_response_json: &str, audit_response_json: &str) -> Self {
        Self::start_scripted(
            None,
            attacker_response_json,
            None,
            None,
            None,
            None,
            Some(audit_response_json),
        )
    }

    /// As `start_with_intent_audit`, but the Attacker also has a scripted answer
    /// for a revision request — a demoted test that no longer passes the
    /// baseline goes back to the Attacker, so both scripts are needed at once.
    fn start_with_intent_audit_and_revision(
        attacker_response_json: &str,
        revise_response_json: &str,
        audit_response_json: &str,
    ) -> Self {
        Self::start_scripted(
            None,
            attacker_response_json,
            Some(revise_response_json),
            None,
            None,
            None,
            Some(audit_response_json),
        )
    }

    /// As `start`, but the Attacker's simplify call — the one the Unknown path
    /// makes — is answered with a scripted body instead of the first element of
    /// the generation script. A test that wants the diagnosed behaviour of a
    /// particular substitute test needs to say what that substitute is.
    fn start_with_simplify(attacker_response_json: &str, simplify_response_json: &str) -> Self {
        Self::start_scripted(
            None,
            attacker_response_json,
            None,
            Some(simplify_response_json),
            None,
            None,
            None,
        )
    }

    /// `on_request` runs with the 0-based index of each request, before the
    /// response is written. It lets a test change the world between rounds —
    /// e.g. replace the --patch file so the next attempt verifies a different
    /// candidate.
    fn start_with_hook(
        attacker_response_json: &str,
        on_request: Option<Box<dyn Fn(usize) + Send>>,
    ) -> Self {
        Self::start_scripted(None, attacker_response_json, None, None, on_request, None, None)
    }

    /// Drop the TCP connection (no HTTP response) for every request whose index
    /// is >= `abort_at`. The client sees a transport error and the harness dies
    /// mid-run — a stand-in for a `HarnessFailure` abort, which happens before
    /// the end-of-run session log is written.
    fn start_aborting(attacker_response_json: &str, abort_at: usize) -> Self {
        Self::start_scripted(
            None,
            attacker_response_json,
            None,
            None,
            None,
            Some(abort_at),
            None,
        )
    }

    fn start_scripted(
        creator_response_json: Option<&str>,
        attacker_response_json: &str,
        revise_response_json: Option<&str>,
        simplify_response_json: Option<&str>,
        on_request: Option<Box<dyn Fn(usize) + Send>>,
        abort_at: Option<usize>,
        intent_audit_response_json: Option<&str>,
    ) -> Self {
        Self::start_scripted_impl(
            creator_response_json,
            attacker_response_json,
            revise_response_json,
            simplify_response_json,
            on_request,
            abort_at,
            intent_audit_response_json,
            StubBehaviour::Normal,
        )
    }

    /// As `start_scripted`, but with an explicit chat-completion behaviour and a
    /// Creator answer, for the streaming/liveness tests.
    fn start_streaming(
        creator_response_json: Option<&str>,
        attacker_response_json: &str,
        behaviour: StubBehaviour,
    ) -> Self {
        Self::start_scripted_impl(
            creator_response_json,
            attacker_response_json,
            None,
            None,
            None,
            None,
            None,
            behaviour,
        )
    }

    fn start_scripted_impl(
        creator_response_json: Option<&str>,
        attacker_response_json: &str,
        revise_response_json: Option<&str>,
        simplify_response_json: Option<&str>,
        on_request: Option<Box<dyn Fn(usize) + Send>>,
        abort_at: Option<usize>,
        intent_audit_response_json: Option<&str>,
        behaviour: StubBehaviour,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://127.0.0.1:{}/v1", addr.port());
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_kinds = Arc::new(Mutex::new(Vec::new()));
        let request_bodies = Arc::new(Mutex::new(Vec::new()));
        let disconnects = Arc::new(AtomicUsize::new(0));
        let truncate_batch_audit = Arc::new(AtomicBool::new(false));
        let truncate_next = Arc::new(Mutex::new(Vec::new()));
        let busy_next = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let request_count_clone = Arc::clone(&request_count);
        let request_kinds_clone = Arc::clone(&request_kinds);
        let request_bodies_clone = Arc::clone(&request_bodies);
        let disconnects_clone = Arc::clone(&disconnects);
        let truncate_batch_audit_clone = Arc::clone(&truncate_batch_audit);
        let truncate_next_clone = Arc::clone(&truncate_next);
        let busy_next_clone = Arc::clone(&busy_next);
        let stop_clone = Arc::clone(&stop);
        // The Attacker has two call sites with different response shapes: a
        // generation call expects `{"tests":[...]}` (AttackerResponse), while a
        // revision/simplify call expects a SINGLE `{test_name,...}` object
        // (GeneratedTest). The stub serves the array to the former and its first
        // element to the latter. Without this, a test that returns to the
        // Attacker for revision would exhaust the client's internal retries on a
        // body it can never parse — and the run would look identical to one where
        // revision never happens.
        let attacker_value: serde_json::Value =
            serde_json::from_str(attacker_response_json).unwrap();
        let revise_value: Option<serde_json::Value> =
            revise_response_json.map(|json| serde_json::from_str(json).unwrap());
        let simplify_value: Option<serde_json::Value> =
            simplify_response_json.map(|json| serde_json::from_str(json).unwrap());
        let creator_value: Option<serde_json::Value> =
            creator_response_json.map(|json| serde_json::from_str(json).unwrap());
        let intent_audit_value: Option<serde_json::Value> =
            intent_audit_response_json.map(|json| serde_json::from_str(json).unwrap());

        let join = thread::spawn(move || {
            // How many Creator calls have been answered, so a scripted array of
            // answers can be served in order.
            let mut creator_calls = 0_usize;
            while !stop_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _addr)) => {
                        let mut request = Vec::new();
                        let mut buf = [0_u8; 4096];
                        let mut header_end = None;
                        let mut content_length = 0_usize;

                        loop {
                            match stream.read(&mut buf) {
                                Ok(0) => break,
                                Ok(n) => {
                                    request.extend_from_slice(&buf[..n]);
                                    if header_end.is_none() {
                                        if let Some(pos) =
                                            request.windows(4).position(|w| w == b"\r\n\r\n")
                                        {
                                            header_end = Some(pos + 4);
                                            let headers =
                                                String::from_utf8_lossy(&request[..pos + 4]);
                                            for line in headers.lines() {
                                                if let Some(value) =
                                                    line.strip_prefix("Content-Length:").or_else(
                                                        || line.strip_prefix("content-length:"),
                                                    )
                                                {
                                                    content_length =
                                                        value.trim().parse::<usize>().unwrap_or(0);
                                                }
                                            }
                                        }
                                    }
                                    if let Some(end) = header_end {
                                        if request.len() >= end + content_length {
                                            break;
                                        }
                                    }
                                }
                                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                                    thread::sleep(Duration::from_millis(10));
                                }
                                Err(_) => break,
                            }
                        }

                        // [VANTA-WHY] `/props` and `/tokenize` are PROBES, not
                        // chat completions: the client asks the server for its
                        // context window and (rarely) for an exact token count.
                        // They are answered here and deliberately NOT counted, so
                        // every pre-existing request-count assertion, and the
                        // `abort_at` index that drops the Nth CHAT call, keep
                        // meaning exactly what they meant before streaming.
                        let head_end = header_end.unwrap_or(request.len());
                        let head = String::from_utf8_lossy(&request[..head_end]).to_string();
                        let request_line = head.lines().next().unwrap_or("").to_string();
                        let path = request_line
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("")
                            .to_string();
                        if path.ends_with("/props") {
                            let body = "{\"default_generation_settings\":{\"n_ctx\":131072},\"n_ctx\":131072}";
                            let response = format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }
                        if path.ends_with("/tokenize") {
                            let body = "{\"count\":1}";
                            let response = format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }

                        let index = request_count_clone.fetch_add(1, Ordering::SeqCst);
                        if let Some(hook) = &on_request {
                            hook(index);
                        }

                        if abort_at.map_or(false, |at| index >= at) {
                            let _ = stream.shutdown(std::net::Shutdown::Both);
                            continue;
                        }

                        // Pick the response shape the caller can actually parse.
                        // The generation system prompt is the only one containing
                        // "write adversarial tests"; revision and simplify calls
                        // want a single test object.
                        let body = String::from_utf8_lossy(&request[header_end.unwrap_or(0)..]);
                        let request_kind = if body.contains("You are Vanta Creator") {
                            "creator"
                        } else if body.contains("You are Vanta Intent Auditor") {
                            "intent_audit"
                        } else if body.contains("write adversarial tests") {
                            "generate"
                        } else if body.contains("failed BATE") {
                            "revise"
                        } else if body.contains("UNKNOWN result") {
                            "simplify"
                        } else {
                            "other"
                        };
                        request_kinds_clone
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(request_kind.to_string());
                        request_bodies_clone
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(body.to_string());
                        // [VANTA-WHY] §14.30 — a ONE-SHOT "the provider is busy"
                        // refusal, consumed per matching request. 429 with a
                        // `Resource Unavailable` body is the documented answer when
                        // a `flex` tier has no capacity, and it is answered here —
                        // AFTER the kind is recorded, BEFORE any stream — because
                        // the refusal happens before the model is reached: the
                        // request never became an answer, which is exactly the
                        // distinction the caller must keep.
                        let busy_now = {
                            let mut queued = busy_next_clone
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            match queued.iter().position(|kind| kind == request_kind) {
                                Some(position) => {
                                    queued.remove(position);
                                    true
                                }
                                None => false,
                            }
                        };
                        if busy_now {
                            let refusal = serde_json::json!({
                                "error": {
                                    "message": "Resource Unavailable: the requested service tier has no capacity right now.",
                                    "type": "server_error",
                                    "code": "resource_unavailable",
                                }
                            })
                            .to_string();
                            let response = format!(
                                "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\nretry-after: 1\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                refusal.len(),
                                refusal
                            );
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }
                        let default_single_test = || {
                            attacker_value
                                .get("tests")
                                .and_then(|tests| tests.get(0))
                                .cloned()
                                .unwrap_or_else(|| attacker_value.clone())
                        };
                        let content = if request_kind == "creator" {
                            // [VANTA-WHY] A creator answer may be a SCRIPT: a JSON
                            // array whose elements are served one per Creator call
                            // (the last repeating). A repair-mode regression needs
                            // the first candidate to be wrong and a later one to be
                            // right, which a single fixed answer cannot express.
                            let served = match creator_value.as_ref() {
                                Some(serde_json::Value::Array(items)) if !items.is_empty() => {
                                    let index = creator_calls.min(items.len() - 1);
                                    items[index].clone()
                                }
                                Some(value) => value.clone(),
                                None => serde_json::json!({"patch_text":"def add(a, b):\n    return a + b\n","explanation":"noop","files_touched":["calc.py"],"confidence":0.1}),
                            };
                            creator_calls += 1;
                            served
                        } else if request_kind == "generate" {
                            attacker_value.clone()
                        } else if request_kind == "intent_audit" {
                            // Default keeps existing non-preservation labels: the
                            // audit says the issue asks for the behaviour, so a
                            // test that bypassed the old deterministic rule still
                            // survives its label. A test that wants a demotion
                            // scripts `start_with_intent_audit`.
                            intent_audit_value.clone().unwrap_or_else(|| {
                                serde_json::json!({
                                    "issue_asks_for_behaviour": true,
                                    "reason": "stub default: the issue asks for this behaviour",
                                })
                            })
                        } else if request_kind == "revise" {
                            revise_value.clone().unwrap_or_else(default_single_test)
                        } else if request_kind == "simplify" {
                            simplify_value.clone().unwrap_or_else(default_single_test)
                        } else {
                            default_single_test()
                        };
                        // [VANTA-WHY] Every chat completion is now answered as
                        // `text/event-stream`, because the client asks with
                        // `stream: true` and reassembles the events. The frames
                        // carry exactly the payload the old JSON envelope did —
                        // the answer in `choices[0].delta.content` — so a parser
                        // on top of `post_chat` cannot tell the difference.
                        //
                        // [VANTA-WHY] §14.5 — a batched intent audit can be
                        // answered with a reply CUT OFF at the cap. The batch
                        // request is the one carrying "TESTS TO AUDIT"; a
                        // per-test request carries "TEST UNDER AUDIT" and is
                        // never truncated, so the fallback has something to
                        // succeed against.
                        // [VANTA-WHY] §14.5 — a ONE-SHOT cut-off: the first
                        // request of each queued kind is answered with a reply
                        // that never closes, and every later request of that
                        // kind is answered normally. `truncate_next_calls` is
                        // the only way to script it — the parse fails, and
                        // whether the caller comes back is the thing under test.
                        let cut_off_batch = request_kind == "intent_audit"
                            && body.contains("TESTS TO AUDIT")
                            && truncate_batch_audit_clone.load(Ordering::SeqCst);
                        let cut_off_once = {
                            let mut queued = truncate_next_clone
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            match queued.iter().position(|kind| kind == request_kind) {
                                Some(position) => {
                                    queued.remove(position);
                                    true
                                }
                                None => false,
                            }
                        };
                        let (content_text, finish_reason) = if cut_off_batch {
                            (
                                TRUNCATED_BATCH_AUDIT_ANSWER.to_string(),
                                Some("length"),
                            )
                        } else if cut_off_once {
                            (TRUNCATED_REPLY.to_string(), Some("length"))
                        } else {
                            (content.to_string(), None)
                        };

                        if matches!(behaviour, StubBehaviour::Fail) {
                            let response = "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }

                        if matches!(behaviour, StubBehaviour::EchoKeyUnauthorized) {
                            // The token is read out of the request HEADERS and put
                            // into the body. Vanta must not print it: the message it
                            // builds from this response reaches the session log and
                            // stderr.
                            let token = head
                                .lines()
                                .find_map(|line| {
                                    line.strip_prefix("Authorization: Bearer ")
                                        .or_else(|| line.strip_prefix("authorization: Bearer "))
                                })
                                .unwrap_or("<no authorization header>")
                                .trim()
                                .to_string();
                            let body = serde_json::json!({
                                "error": {
                                    "message": format!(
                                        "Incorrect API key provided: {token}. You can find \
                                         your API key at https://platform.openai.com/account/api-keys"
                                    ),
                                    "type": "invalid_request_error",
                                    "code": "invalid_api_key",
                                }
                            })
                            .to_string();
                            let response = format!(
                                "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }

                        let headers = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
                        let token_frame = format!(
                            "data: {}\n\n",
                            serde_json::json!({
                                "choices": [{"delta": {"content": content_text}}]
                            })
                        );
                        let _ = stream.write_all(headers.as_bytes());
                        let _ = stream.write_all(token_frame.as_bytes());
                        let _ = stream.flush();

                        if matches!(behaviour, StubBehaviour::Stall) {
                            // One token, then silence. The client's idle timeout
                            // must fire and close the socket; we sit here until it
                            // does and record that we saw the disconnect — proof
                            // the cancel reached the SERVER, not merely the
                            // client's own future.
                            wait_for_disconnect(&mut stream, &disconnects_clone);
                            continue;
                        }

                        // [VANTA-WHY] §14.5 — `finish_reason` rides on the same
                        // tail frame a real server publishes it on, and only for
                        // a truncated reply: the field is absent otherwise, so a
                        // test that forgets to truncate cannot pass by accident.
                        let mut choice = serde_json::json!({"delta": {}});
                        if let Some(reason) = finish_reason {
                            choice["finish_reason"] = serde_json::json!(reason);
                        }
                        let tail = format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            serde_json::json!({
                                "choices": [choice],
                                // [VANTA-WHY] 777/123/900, not 1/1/2. A token
                                // count a local estimate could plausibly have
                                // produced would let a test that reads the
                                // estimate pass for a test that reads the
                                // server; these numbers cannot come from
                                // anywhere but this frame. `completion_tokens`
                                // is the field a streamed response omits unless
                                // `stream_options.include_usage` was asked for.
                                //
                                // [VANTA-WHY] §14.29 — the NESTED details are
                                // published here too, and are the shape the
                                // hosted providers use: `cached_tokens` lives
                                // under `prompt_tokens_details` and
                                // `reasoning_tokens` under
                                // `completion_tokens_details`, not at the top
                                // level. 314 and 41 rather than round numbers,
                                // and both strictly inside their parents (314 ≤
                                // 777, 41 ≤ 123), because the arithmetic that
                                // prices a call subtracts the cached part from
                                // the prompt and must never be able to overbill.
                                // A stub that only ever published the flat
                                // counts would let a parser that never looked
                                // into the nested objects pass.
                                "usage": {
                                    "prompt_tokens": 777,
                                    "completion_tokens": 123,
                                    "total_tokens": 900,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 314
                                    },
                                    "completion_tokens_details": {
                                        "reasoning_tokens": 41
                                    }
                                }
                            })
                        );
                        let _ = stream.write_all(tail.as_bytes());
                        let _ = stream.flush();
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            base_url,
            request_count,
            request_kinds,
            request_bodies,
            disconnects,
            truncate_batch_audit,
            truncate_next,
            busy_next,
            stop,
            join: Some(join),
        }
    }

    /// Answer every BATCHED intent-audit request with a reply CUT OFF at the
    /// output cap (`finish_reason: "length"`), leaving per-test audit requests
    /// alone (§14.5). Called after `start*`.
    fn truncate_batch_audit(&self) {
        self.truncate_batch_audit.store(true, Ordering::SeqCst);
    }

    /// Cut off the next request of each named kind (stub kind labels: `creator`,
    /// `generate`, `revise`, `simplify`, `intent_audit`), one cut-off per entry,
    /// and answer every later request normally (§14.5). Called after `start*`.
    fn truncate_next_calls(&self, kinds: &[&str]) {
        let mut queued = self
            .truncate_next
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        queued.clear();
        queued.extend(kinds.iter().map(|kind| kind.to_string()));
    }

    /// Refuse the next request of each named kind with a 429 `Resource
    /// Unavailable` (stub kind labels: `creator`, `generate`, `revise`,
    /// `simplify`, `intent_audit`), one refusal per entry, and answer every later
    /// request normally (§14.30). Called after `start*`.
    fn refuse_next_calls(&self, kinds: &[&str]) {
        let mut queued = self
            .busy_next
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        queued.clear();
        queued.extend(kinds.iter().map(|kind| kind.to_string()));
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }

    /// How many stalled chat connections the client has closed on us.
    fn disconnects(&self) -> usize {
        self.disconnects.load(Ordering::SeqCst)
    }

    fn request_kinds(&self) -> Vec<String> {
        self.request_kinds
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn request_bodies(&self) -> Vec<String> {
        self.request_bodies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl Drop for StubLlmServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(
            self.base_url
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap(),
        );
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Semantic-but-wrong patch: valid Python, wrong arithmetic. The patched tree
/// imports fine, so the tests must surface as Failed/Logic.
const WRONG_BEHAVIOUR_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n";

/// [VANTA-WHY] Trailing binary operator => SyntaxError. The patched tree cannot
/// be imported for a genuinely STATIC reason, so ruff/mypy must flag it and the
/// result must stay Static. This is the mirror of the import bug: it proves the
/// Infrastructure reroute only catches harness failures, and never swallows a
/// real syntax error the model actually introduced.
const SYNTAX_ERROR_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a + b +\n";

/// Behaviour-PRESERVING patch: it changes calc.py (so `apply_patch` reports the
/// file as changed and the run is a real review of a real candidate) without
/// touching what `add` returns. In review mode a candidate that PASSES round k
/// is attacked again in round k+1, so a multi-round run needs a candidate that
/// survives every round — this is that candidate.
const PRESERVING_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n+    \"\"\"Return the sum of two numbers.\"\"\"\n     return a + b\n";

/// The tree `PRESERVING_PATCH` produces from `write_test_repo`'s `calc.py`.
const PRESERVING_PATCHED_SOURCE: &str =
    "def add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n    return a + b\n";

const THREE_SIGNAL_TESTS: &str = r#"{"tests":[
        {
            "test_name":"test_bug_regression_signal",
            "test_source":"from calc import add\n\ndef test_bug_regression_signal():\n    assert add(2, 3) == 6\n",
            "intent":"BugReproduction",
            "reason":"Fails on the original baseline, so BATE admits it.",
            "issue_quote":"The issue expects a bug-reproduction probe asserting add(2, 3) == 6."
        },
        {
            "test_name":"test_feature_acceptance_signal",
            "test_source":"from calc import add\n\ndef test_feature_acceptance_signal():\n    assert add(2, 3) == 7\n",
            "intent":"FeatureAcceptance",
            "reason":"Also fails on the original baseline, so BATE admits it.",
            "issue_quote":"The issue expects a feature-acceptance probe asserting add(2, 3) == 7."
        },
        {
            "test_name":"test_behavior_preservation_signal",
            "test_source":"from calc import add\n\ndef test_behavior_preservation_signal():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original baseline and fails on the bad patch.",
            "issue_quote":null
        }
    ]}"#;

/// The single-test form of the stub's keep-the-label answer.
const SINGLE_AUDIT_KEEP_ANSWER: &str =
    r#"{"issue_asks_for_behaviour":true,"reason":"stub: keep the label"}"#;

/// [VANTA-WHY] §14.5 — the BATCHED form of the same answer: one entry per
/// numbered test, which is what makes it a transport change and not a decision
/// change. The batched auditor must reach the same verdicts for
/// `THREE_SIGNAL_TESTS` as three single-test calls carrying one entry each.
const THREE_SIGNAL_BATCH_AUDIT_ANSWER: &str = r#"{"tests":[
        {"index":1,"issue_asks_for_behaviour":true,"reason":"stub: keep the label"},
        {"index":2,"issue_asks_for_behaviour":true,"reason":"stub: keep the label"},
        {"index":3,"issue_asks_for_behaviour":true,"reason":"stub: keep the label"}
    ]}"#;

/// Three BehaviourPreservation probes that all PASS on the original tree.
///
/// [VANTA-WHY] §14.5 — demoting a probe the audit could not answer is a no-op for
/// these, so a run whose audit fails has an arithmetic that is the audit retries
/// and nothing else: no BATE revision, no relabel gate. That is what makes the
/// call counts below a measurement of the transport rather than of the fixture.
const THREE_PRESERVATION_TESTS: &str = r#"{"tests":[
        {
            "test_name":"test_preserve_add_pair",
            "test_source":"from calc import add\n\ndef test_preserve_add_pair():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"The original tree returns 5 for this input."
        },
        {
            "test_name":"test_preserve_add_pair_reversed",
            "test_source":"from calc import add\n\ndef test_preserve_add_pair_reversed():\n    assert add(3, 2) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"The original tree returns 5 for this input too."
        },
        {
            "test_name":"test_preserve_add_identity",
            "test_source":"from calc import add\n\ndef test_preserve_add_identity():\n    assert add(0, 0) == 0\n",
            "intent":"BehaviorPreservation",
            "reason":"Adding zero is unchanged behaviour."
        }
    ]}"#;

/// A batched answer that answers ONE of the round's three tests. An unknown,
/// duplicated or missing index is a parse failure of the WHOLE response (§14.5),
/// so this can never be read as "the other two answered false".
const SHELL_BATCH_AUDIT_ANSWER: &str =
    r#"{"tests":[{"index":1,"issue_asks_for_behaviour":true,"reason":"only one of three"}]}"#;

/// A batched answer CUT OFF mid-string, which is what a server's reply looks
/// like when the generation hit `max_tokens`: the JSON never closes, so no
/// parser can read it, and the tail frame says `finish_reason: "length"`.
///
/// [VANTA-WHY] §14.5 — the truncation is a fact about the REQUEST (N answers in
/// one capped generation), not about the round, so re-sending the same batch
/// cannot clear it. The per-test path can: it asks the same questions one at a
/// time, each well inside the single call's own output allowance.
const TRUNCATED_BATCH_AUDIT_ANSWER: &str =
    r#"{"tests":[{"index":1,"issue_asks_for_behaviour":true,"reason":"stub: this answer was cut off at the cap"#;

/// A reply CUT OFF mid-string in the shape the caller expected, for a ONE-SHOT
/// cut-off on any call kind (§14.5).
///
/// [VANTA-WHY] It has to be unparseable for the kind it is served to — an
/// unterminated JSON string is — so the retry is reached through the ordinary
/// parse-failure path rather than a special one, and the tail frame's
/// `finish_reason: "length"` is what tells the retry loop the answer was CUT
/// OFF rather than wrong. That distinction is the whole point: a cut-off answer
/// is retried only where the call site samples above temperature 0.0.
const TRUNCATED_REPLY: &str =
    r#"{"tests":[{"test_name":"test_cut_off_at_the_output_cap","test_source":"from calc import add\n\ndef test_cut_off"#;

/// A single BugReproduction test whose baseline CONTRADICTS its intent:
/// `calc.add(2, 3)` is 5 on the original tree, so the asserted `== 5` passes,
/// while a BugReproduction test is only admissible if it FAILS there. The stub
/// serves this array to the generation call and its first (still
/// intent-contradicting) element to each revision, so every revision disagrees
/// with the baseline too — the only way BATE can stop is to scrap after its
/// revision budget is spent.
const CONTRADICTORY_INTENT_TEST: &str = r#"{"tests":[
        {
            "test_name":"test_intent_contradiction",
            "test_source":"from calc import add\n\ndef test_intent_contradiction():\n    assert add(2, 3) == 5\n",
            "intent":"BugReproduction",
            "reason":"Declares a bug that the original tree does not have.",
            "issue_quote":"The issue expects a bug-reproduction probe asserting add(2, 3) == 6."
        }
    ]}"#;

/// The oracle loophole, in the exact shape the real fixture produced: the test
/// asserts the PATCH's own arithmetic (`add(2, 3) == -1`) and legitimises it by
/// quoting a sentence that IS verbatim in the issue — a sentence which in fact
/// forbids the change. A presence-only quote check admits this; the quote must
/// also name the asserted input.
const ISSUE_SCOPE_LOOPHOLE_TEST: &str = r#"{"tests":[
        {
            "test_name":"test_patch_written_feature",
            "test_source":"from calc import add\n\ndef test_patch_written_feature():\n    assert add(2, 3) == -1\n",
            "intent":"FeatureAcceptance",
            "reason":"Reads the patched arithmetic back as if the issue had asked for it.",
            "issue_quote":"Please fix that boundary bug without changing the existing behavior at the other thresholds."
        }
    ]}"#;

const ISSUE_SCOPE_REVISED_PRESERVATION_TEST: &str = r#"{
    "test_name":"test_patch_written_feature",
    "test_source":"from calc import add\n\ndef test_patch_written_feature():\n    assert add(2, 3) == 5\n",
    "intent":"BehaviorPreservation",
    "reason":"The issue never asked for subtraction, so this behavior must be preserved.",
    "issue_quote":null
}"#;

/// Patch that makes `calc.add(2, 3)` return 6, so a probe can tell the patched
/// tree from the original one (where it returns 5).
const HANGING_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a + b + 1\n";

/// A BehaviourPreservation probe that PASSES on the original tree (add(2, 3) is
/// 5 there) and HANGS on the patched one (add(2, 3) is 6). BATE therefore admits
/// it — which is the point: it is the PATCHED run, not the baseline, that the
/// backend has to kill, so the timeout signal path is what is under test.
const HANGING_PROBE: &str = r#"{"tests":[
        {
            "test_name":"test_hangs_on_patched_tree",
            "test_source":"import time\nfrom calc import add\n\ndef test_hangs_on_patched_tree():\n    if add(2, 3) == 6:\n        time.sleep(3600)\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original tree and hangs on the patched one."
        }
    ]}"#;

const SINGLE_BEHAVIOR_TEST: &str = r#"{"tests":[
        {
            "test_name":"test_behavior_preservation_only",
            "test_source":"from calc import add\n\ndef test_behavior_preservation_only():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original tree and fails on the wrong patch."
        }
    ]}"#;

/// ONE `BugReproduction` probe, and nothing else in the round.
///
/// [VANTA-WHY] §14.2 — this is the shape whose failures NO rule covers: `(b)`
/// needs the issue to name the raised exception and `TEST_ISSUE_TEXT` names none,
/// `(c)` needs two distinct tests, and `(d)` covers preservation/acceptance
/// failures only. It is also PySnooper/3's dev-fix shape: the only evidence
/// against the patch is a test nothing corroborates, which must reach a human
/// instead of being reported as a rejection.
const SINGLE_BUG_REPRODUCTION_TEST: &str = r#"{"tests":[
        {
            "test_name":"test_bug_reproduction_only",
            "test_source":"from calc import add\n\ndef test_bug_reproduction_only():\n    assert add(2, 3) == 6\n",
            "intent":"BugReproduction",
            "reason":"Fails on the original baseline, so BATE admits it.",
            "issue_quote":"The issue expects a bug-reproduction probe asserting add(2, 3) == 6."
        }
    ]}"#;

/// A round of two tests: a reproduction probe that pins a string the issue never
/// states, and a preservation probe that passes on the original tree.
///
/// [VANTA-WHY] §14.4 — one answer is served to both tests, so this is what proves
/// the mirror: the third question may scrap the REPRODUCTION probe, and it may
/// never scrap the preservation probe, whatever the answer says.
const UNGROUNDED_AND_PRESERVING_TESTS: &str = r#"{"tests":[
        {
            "test_name":"test_bug_reproduction_pins_invented_text",
            "test_source":"from calc import add\n\ndef test_bug_reproduction_pins_invented_text():\n    assert add(2, 3) == 6\n    assert \"Starting var\" in str(add(2, 3))\n",
            "intent":"BugReproduction",
            "reason":"Fails on the original baseline, so BATE admits it.",
            "issue_quote":"The issue expects a bug-reproduction probe asserting add(2, 3) == 6."
        },
        {
            "test_name":"test_preserve_add_pair",
            "test_source":"from calc import add\n\ndef test_preserve_add_pair():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original baseline."
        }
    ]}"#;

/// A BehaviourPreservation probe asserting the ORIGINAL result at an input the
/// issue requires to CHANGE. This is exactly the probe the boundary rule tells
/// the Attacker to write — correct when the issue preserves that boundary, wrong
/// when the issue moves it. If the intent check did not ask the complementary
/// question, this probe would be admitted as preservation, fail on a CORRECT
/// patch, and reject it.
const PRESERVATION_PROBE_CONTRADICTING_THE_ISSUE: &str = r#"{"tests":[
        {
            "test_name":"test_preserve_old_boundary",
            "test_source":"from calc import add\n\ndef test_preserve_old_boundary():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Preserves the old value at an input the issue asks to change."
        }
    ]}"#;

/// A BehaviourPreservation probe that in fact asserts the NEW behaviour — the
/// value the issue asks for — so it FAILS on the original tree.
///
/// [VANTA-WHY] Item 6: this is the B v2 shape. The label is wrong, not the probe.
/// A preservation probe that passes on the original proves nothing about the
/// patch, so scrapping it discards the one test that could catch a broken patch
/// (a broken patch is exactly one that FAILS this assertion). The intent audit
/// must relabel it to FeatureAcceptance, whose baseline rule then demands it fail
/// on the original — which it does — before it is free to pass on the patched
/// tree.
const PRESERVATION_PROBE_ASSERTING_THE_NEW_BEHAVIOUR: &str = r#"{"tests":[
        {
            "test_name":"test_new_behaviour_probe",
            "test_source":"from calc import add\n\ndef test_new_behaviour_probe():\n    assert add(2, 3) == 6\n",
            "intent":"BehaviorPreservation",
            "reason":"Asserts the value the issue asks for, so it is an acceptance probe wearing a preservation label."
        }
    ]}"#;

/// A patch that implements the behaviour the promoted probe asserts: `add(2, 3)`
/// becomes 6, the value the issue asks for. It is a CORRECT patch, so a run that
/// wrongly scrapped the probe would report a false rejection.
const NEW_BEHAVIOUR_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a + b + 1\n";

/// A Creator patch that APPLIES cleanly but leaves the tree unimportable.
///
/// [VANTA-WHY] It has to be a real unified diff. The Creator's patch goes
/// through the repair-mode cascade, and a *bare* source blob with a syntax error
/// is now refused by the syntax guard (Fix 1): the original parsed and the result
/// does not, so the apply is reported as failed and the run stops with
/// PatchApplicationFailed before the Diagnostician ever sees it. A git-applied
/// diff is trusted by design — a syntax error inside a valid hunk is the model's
/// fault, not Vanta's — so this is the shape that produces the Static finding
/// whose structured payload the second Creator call must carry.
const CREATOR_SYNTAX_ERROR_RESPONSE: &str = r#"{
    "patch_text":"diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a + b +\n",
    "explanation":"introduce a broken patch for feedback regression coverage",
    "files_touched":["calc.py"],
    "confidence":0.2
}"#;

/// Original tree for the native-suite gate: an extra function the generated
/// tests never touch, so the patch can break it invisibly to them.
const NATIVE_REPO_SOURCE: &str =
    "def add(a, b):\n    return a + b\n\n\ndef scale(x, factor):\n    return x * factor\n";

/// The repository's OWN test. It passes on the original tree; the generated
/// stub tests only ever assert on `add`, so nothing they produce will notice
/// when the patch breaks `scale`.
const NATIVE_REPO_TEST: &str =
    "from calc import scale\n\n\ndef test_scale_doubles():\n    assert scale(3, 2) == 6\n";

/// Leaves `add` alone and breaks `scale`, so only the repo's own suite can see it.
const NATIVE_BREAKING_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -5,2 +5,2 @@\n def scale(x, factor):\n-    return x * factor\n+    return x * factor + 1\n";


fn metric_row_from_observed_outcome(
    to_result: TestResult,
    to_failure_class: FailureClass,
) -> TransitionRow {
    TransitionRow {
        session_id: SessionId::new("metric-test"),
        schema_version: Some(SCHEMA_VERSION),
        from_patch_attempt_id: Some(PatchAttemptId::new("p0")),
        to_patch_attempt_id: Some(PatchAttemptId::new("p1")),
        from_attempt: 0,
        to_attempt: 1,
        test_id: "hash-timeout".into(),
        test_name: "test_hangs_on_patched_tree".into(),
        shared_set_id: "set-1".into(),
        from_result: TestResult::Passed,
        to_result,
        from_failure_class: None,
        to_failure_class: Some(to_failure_class),
        evidence_source: vanta::core::EvidenceSource::AttackerGenerated,
        test_intent: TestIntent::BehaviorPreservation,
        censored: false,
        apply_failed: false,
        gap: false,
        in_s0: true,
    }
}

/// The issue text every stub run is given. It carries one quotable sentence per
/// intent the stub fixtures declare, plus a verbatim sentence that names NO
/// input.
const TEST_ISSUE_TEXT: &str = concat!(
    "Verify a candidate patch for calc.add.\n",
    "The issue expects a bug-reproduction probe asserting add(2, 3) == 6.\n",
    "The issue expects a feature-acceptance probe asserting add(2, 3) == 7.\n",
    "Existing behavior add(2, 3) == 5 must be preserved unless the issue says otherwise.\n",
    // [VANTA-WHY] A verbatim, quotable issue sentence that names NO input.
    // It is the shape the real fixture produced: the Attacker quoted a
    // genuine line ("without changing the existing behavior at the other
    // thresholds") that forbids the very change its test encoded, and a
    // presence-only quote check admitted it.
    "Please fix that boundary bug without changing the existing behavior at the other thresholds.\n",
);

fn write_test_repo(root: &Path, patch_text: &str) -> (PathBuf, PathBuf, PathBuf) {
    write_test_repo_with(root, "def add(a, b):\n    return a + b\n", patch_text, None)
}

/// As `write_test_repo`, but the original source and an optional file under
/// `tests/` are caller-supplied. The repo's own test needs both: a test that
/// passes on the original tree only exists if the original tree defines what it
/// calls, and a patch that breaks it must not be one the generated tests cover.
fn write_test_repo_with(
    root: &Path,
    source: &str,
    patch_text: &str,
    native_test: Option<&str>,
) -> (PathBuf, PathBuf, PathBuf) {
    let repo_dir = root.join("repo");
    fs::create_dir_all(&repo_dir).unwrap();
    fs::write(repo_dir.join("calc.py"), source).unwrap();
    let issue_path = repo_dir.join("issue.md");
    fs::write(&issue_path, TEST_ISSUE_TEXT).unwrap();

    if let Some(test) = native_test {
        let tests_dir = repo_dir.join("tests");
        fs::create_dir_all(&tests_dir).unwrap();
        fs::write(tests_dir.join("test_core.py"), test).unwrap();
    }

    let patch_path = root.join("candidate.patch");
    fs::write(&patch_path, patch_text).unwrap();

    (repo_dir, issue_path, patch_path)
}

/// Path of the JSONL session log, `vanta_sessions/{session_id}_session.jsonl`.
/// Exactly one must exist: an aborted run leaves the same single file, just with
/// fewer records.
fn find_session_log(dir: &Path) -> PathBuf {
    let sessions_dir = dir.join("vanta_sessions");
    let mut logs: Vec<PathBuf> = fs::read_dir(&sessions_dir)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .ends_with("_session.jsonl")
        })
        .collect();
    logs.sort();
    assert_eq!(logs.len(), 1, "expected exactly one session log");
    logs.remove(0)
}

/// Read the JSONL session log, one `SessionRecord` per line. The first record is
/// always `session.start`; a truncated final line is a hard failure, since the
/// sink flushes each record complete.
fn read_session_records(path: &Path) -> Vec<SessionRecord> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<SessionRecord>(line).unwrap())
        .collect()
}

fn find_transitions(dir: &Path) -> PathBuf {
    let sessions_dir = dir.join("vanta_sessions");
    let mut files: Vec<PathBuf> = fs::read_dir(&sessions_dir)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .ends_with("_transitions.jsonl")
        })
        .collect();
    files.sort();
    assert_eq!(
        files.len(),
        1,
        "the raw transition dataset must be written once per run"
    );
    files.remove(0)
}

/// Read the JSONL row store, one `TransitionRow` per line. A truncated final
/// line is a hard failure: the durability contract is that a flushed row is
/// complete, so a partial line means the sink or its flush is broken.
fn read_transition_rows(path: &Path) -> Vec<TransitionRow> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<TransitionRow>(line).unwrap())
        .collect()
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn host_python() -> String {
    let output = Command::new("which").arg("python3").output().unwrap();
    assert!(
        output.status.success(),
        "python3 must be available for the integration test"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

/// A sentinel API key for every stub run. It is deliberately distinctive so the
/// test below can assert it never reaches the session log: `LlmConfig`'s
/// `Display` redacts the key, but `Display` does NOT apply to serde, so the log
/// must be proven clean by construction, not assumed.
const TEST_API_KEY: &str = "sk-SENTINELmustNEVERreachtheLOG";

/// Run vanta_m3 in patch mode against a stub LLM and return the exit status.
///
/// `VANTA_PYTHON` is pinned to the host interpreter so the static-analysis pass
/// (ruff + mypy) runs against an interpreter that actually has them installed —
/// the `/usr/bin/python3` on PATH does not.
fn run_patch_mode(
    workspace: &Path,
    patch_path: &Path,
    server: &StubLlmServer,
) -> std::process::ExitStatus {
    run_patch_mode_with(workspace, patch_path, server, 3, &[])
}

/// As `run_patch_mode`, but with an explicit attempt budget and extra env vars.
///
/// [VANTA-WHY] A regression test for one specific behaviour wants ONE attempt
/// (so any extra LLM call can only have come from inside that behaviour) and
/// sometimes a non-default setting (e.g. a short test timeout). Pinning the rest
/// of the environment keeps the run deterministic.
fn run_patch_mode_with(
    workspace: &Path,
    patch_path: &Path,
    server: &StubLlmServer,
    max_attempts: u32,
    extra_env: &[(&str, &str)],
) -> std::process::ExitStatus {
    run_patch_mode_with_output(workspace, patch_path, server, max_attempts, extra_env).status
}

fn run_patch_mode_with_output(
    workspace: &Path,
    patch_path: &Path,
    server: &StubLlmServer,
    max_attempts: u32,
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    let (repo_dir, issue_path) = (
        workspace.join("repo"),
        workspace.join("repo").join("issue.md"),
    );
    let python = host_python();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vanta_m3"));
    cmd.current_dir(workspace)
        .arg("--repo")
        .arg(&repo_dir)
        .arg("--issue")
        .arg(&issue_path)
        .arg("--patch")
        .arg(patch_path)
        .arg("--max-attempts")
        .arg(max_attempts.to_string())
        .arg("--test-cmd")
        .arg(format!("{python} -m pytest"))
        .env(
            "VANTA_LLM_BASE_URL",
            format!("{}/v1", server.base_url.trim_end_matches("/v1")),
        )
        .env("VANTA_LLM_MODEL", "stub-model")
        .env("VANTA_API_KEY", TEST_API_KEY)
        .env("VANTA_LLM_IDLE_TIMEOUT_SECS", "5")
        .env("VANTA_PYTHON", python)
        .env("PATH", "/usr/bin:/bin")
        .env("VANTA_SANDBOX_MEMORY_MB", "0");
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.output().unwrap()
}

fn run_creator_mode_with_output(
    workspace: &Path,
    server: &StubLlmServer,
    max_attempts: u32,
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    let repo_dir = workspace.join("repo");
    let issue_path = workspace.join("repo").join("issue.md");
    let python = host_python();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vanta_m3"));
    cmd.current_dir(workspace)
        .arg("--repo")
        .arg(&repo_dir)
        .arg("--issue")
        .arg(&issue_path)
        .arg("--max-attempts")
        .arg(max_attempts.to_string())
        .arg("--test-cmd")
        .arg(format!("{python} -m pytest"))
        .env(
            "VANTA_LLM_BASE_URL",
            format!("{}/v1", server.base_url.trim_end_matches("/v1")),
        )
        .env("VANTA_LLM_MODEL", "stub-model")
        .env("VANTA_API_KEY", TEST_API_KEY)
        .env("VANTA_LLM_IDLE_TIMEOUT_SECS", "5")
        .env("VANTA_PYTHON", python)
        .env("PATH", "/usr/bin:/bin")
        .env("VANTA_SANDBOX_MEMORY_MB", "0");
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.output().unwrap()
}

#[test]
fn patch_mode_three_attempt_run_writes_multi_record_session_log() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // [VANTA-WHY] In review mode `--max-attempts` is ATTACK DEPTH, and a round
    // that finds a Logic failure ends the run at once. So a run that spans three
    // rounds needs a candidate that SURVIVES every round — a behaviour-preserving
    // patch and a preservation-only test set — not a bad patch that would be
    // rejected on round 1.
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);

    assert!(
        status.success(),
        "a candidate that survives the whole attack depth is the approval"
    );
    assert_eq!(
        server.request_count(),
        4,
        "one Attacker call per review round (3), plus the round-1 patch-blind intent \
         audit of the preservation probe; rounds 2 and 3 generate the same probe, so \
         the dedup drops it before there is anything left to audit"
    );

    let records = read_session_records(&find_session_log(workspace.path()));

    // [VANTA-WHY] The session log is appended as records are produced, and its
    // FIRST record is `session.start` — the run's attribution, on disk before
    // attempt 1. Without it a crash would leave transition rows with no record
    // of which model produced them.
    assert_eq!(
        records.first().map(|r| r.phase.as_str()),
        Some("session.start"),
        "the run identity must be the first record on disk"
    );
    let start = &records[0];
    assert!(
        start.summary.contains("model=stub-model"),
        "session.start must name the model that produced the run; got {}",
        start.summary
    );
    assert!(
        start.summary.contains("requested_backend=") && start.summary.contains("actual_backend="),
        "session.start must record both the requested and actual sandbox backend; got {}",
        start.summary
    );
    // A commit hash alone would name a commit that does not reproduce a run made
    // on a dirty tree, so the clean/dirty flag travels with it.
    assert!(
        start.summary.contains("vanta_commit=") && start.summary.contains("vanta_dirty="),
        "session.start must record the Vanta revision AND whether the tree was dirty; got {}",
        start.summary
    );
    assert!(
        start.summary.contains("tested_commit="),
        "session.start must record the full tested commit the PR path will branch from; got {}",
        start.summary
    );
    assert!(
        start.summary.contains("target_repo_dirty="),
        "session.start must record whether the target repo working tree was dirty; got {}",
        start.summary
    );
    assert!(
        start.summary.contains("baseline_test_timeout_secs=")
            && start.summary.contains("patched_test_timeout_secs="),
        "session.start must record the effective baseline and patched-side test timeouts; got {}",
        start.summary
    );
    assert!(
        start.summary.contains("deps_freeze="),
        "session.start must record the dependency checkpoint's installed package versions; got {}",
        start.summary
    );
    // The key was set for this run (`VANTA_API_KEY`), and the run-identity line
    // must say only that a key was present — never the key itself.
    assert!(
        start.summary.contains("api_key=set"),
        "session.start must record that a key was present; got {}",
        start.summary
    );
    assert!(
        !start.summary.contains(TEST_API_KEY),
        "the API key must never reach the session log"
    );
    // The whole record file, not just the summary, must be free of the key: this
    // is the artifact that gets published alongside the paper.
    let raw_log = fs::read_to_string(find_session_log(workspace.path())).unwrap();
    assert!(
        !raw_log.contains(TEST_API_KEY) && !raw_log.contains("sk-SENTINEL"),
        "the serialised session log must contain no key field carrying the secret"
    );
    assert!(
        start.patch_attempt_id.is_none(),
        "a run-level record has no patch attempt; it must be null, not invented"
    );

    // The raw transition dataset is a JSONL row store, written once per run.
    // --patch mode still compares review rounds, so it records rows here; the
    // from == to property that marks them as attack depth is asserted by its
    // own test.
    let rows = read_transition_rows(&find_transitions(workspace.path()));
    assert!(
        !rows.is_empty(),
        "--patch mode compares review rounds, so the row store must not be empty"
    );

    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let attempt1_manifest = read_json(&debug_dir.join("attempt_1_patched_tree_manifest.json"));
    assert_eq!(
        attempt1_manifest["paths"].as_array().unwrap(),
        &vec![serde_json::Value::String("calc.py".into())]
    );
    assert_eq!(
        fs::read_to_string(debug_dir.join("attempt_1_patched_tree").join("calc.py")).unwrap(),
        PRESERVING_PATCHED_SOURCE
    );

    // The structural checks above prove the tree is well-formed. These prove the
    // patched tests actually RAN against it: a candidate that survives review
    // must produce Passed assertion results, not Error. Error here would mean the
    // harness failed to import the tree — the exact bug that previously hid
    // behind Static rows. (The failing-result side of that pair is pinned by
    // `review_mode_rejects_a_bad_candidate_on_round_one`.)
    let mut patched_result_kinds: Vec<String> = Vec::new();
    for entry in fs::read_dir(&debug_dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("attempt_") && name.contains("_diagnostic_") {
            let diag = read_json(&path);
            patched_result_kinds.push(diag["result"].as_str().unwrap_or_default().to_string());
        }
    }
    assert!(
        patched_result_kinds.iter().all(|r| r == "Passed"),
        "the behaviour-preserving candidate must yield Passed patched results — an Error means the \
         harness failed to produce a runnable tree; got {patched_result_kinds:?}"
    );

    let attempt_start_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase == "attempt.start")
        .collect();
    assert_eq!(attempt_start_rows.len(), 3);

    let attempt_summary_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase == "attempt.summary")
        .collect();
    assert_eq!(attempt_summary_rows.len(), 3);

    let attacker_summary_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase == "attacker.summary")
        .collect();
    assert_eq!(attacker_summary_rows.len(), 3);

    let bate_summary_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase == "bate.summary")
        .collect();
    assert_eq!(bate_summary_rows.len(), 3);

    let final_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase == "M3_repair_loop")
        .collect();
    assert_eq!(final_rows.len(), 1);

    let patched_result_rows: Vec<_> = records
        .iter()
        .filter(|r| r.phase.starts_with("patched.result["))
        .collect();
    assert_eq!(
        patched_result_rows.len(),
        3,
        "one shared probe per round: rounds 2 and 3 generate a duplicate and drop it, so \
         the shared set never grows and each round runs exactly one test"
    );

    let mut per_iteration_counts: Vec<usize> = attempt_start_rows
        .iter()
        .map(|attempt| {
            let iteration_id = attempt.testing_iteration_id.as_ref().unwrap().as_str();
            patched_result_rows
                .iter()
                .filter(|row| {
                    row.testing_iteration_id.as_ref().map(|id| id.as_str()) == Some(iteration_id)
                })
                .count()
        })
        .collect();
    per_iteration_counts.sort_unstable();
    assert_eq!(
        per_iteration_counts,
        vec![1, 1, 1],
        "the stub repeats the same test every round; from round 2 the Attacker's copy is \
         a duplicate of the shared probe and is dropped, so each round runs exactly the \
         one shared probe"
    );

    let unique_patch_ids: std::collections::HashSet<&str> = attempt_start_rows
        .iter()
        .map(|row| row.patch_attempt_id.as_ref().unwrap().as_str())
        .collect();
    assert_eq!(
        unique_patch_ids.len(),
        1,
        "same patch should keep one stable patch_attempt_id"
    );

    // Per round: attempt.start, attacker.dedup, attacker.summary,
    // native_suite.summary, one patched.result (the shared probe), and
    // attempt.summary — 7, plus the round-1 intent.check row for the freshly
    // generated probe. Plus the run-level session.start, run.summary,
    // context.select (the source selection's provenance, §1.6.2 item 4) and
    // M3_repair_loop records.
    //
    // [VANTA-WHY] Counted over the ATTEMPT-STRUCTURE rows only. Token accounting
    // writes one `call.usage` row per chat call, and how many calls a round makes
    // depends on retries and BATE revisions — pinning the total here would make a
    // test about the repair loop's record shape fail for a reason outside it.
    // Those rows have their own test
    // (`token_accounting_records_the_servers_counts_per_call_and_a_run_total`).
    let structure_rows: Vec<_> = records.iter().filter(|r| r.phase != "call.usage").collect();
    assert_eq!(structure_rows.len(), 26);
    assert!(
        records.iter().any(|r| r.phase == "call.usage"),
        "this run called the model, so it must record token usage"
    );

    // [VANTA-WHY] §1.6.2 item 4 — the source selection is part of the run's
    // identity: WHICH files a role was shown, WHY each was chosen, and how many
    // bytes/tokens each cost. A run whose context changed silently would
    // otherwise look like the same experiment as the one before it.
    let context_rows: Vec<_> = records.iter().filter(|r| r.phase == "context.select").collect();
    assert_eq!(
        context_rows.len(),
        1,
        "the run-level source selection must be recorded exactly once"
    );
    let selection = &context_rows[0].summary;
    assert!(
        selection.starts_with("mode=") && selection.contains(":role="),
        "the provenance line must name the mode and the roles it was selected for; got {selection}"
    );
    assert!(
        selection.contains(":selected=[") && selection.contains(":pruned=["),
        "the provenance line must list the selected files AND the pruned candidates; got {selection}"
    );
    // The session log is a one-record-per-line, space-delimited `key=value`
    // store: a value carrying a space or a newline would split the row.
    assert!(
        !selection.contains('\n') && !selection.contains('\r') && !selection.contains(' '),
        "the provenance line must survive the space-separated log format: {selection}"
    );
    assert!(
        context_rows[0].patch_attempt_id.is_none(),
        "a run-level record has no patch attempt; it must be null, not invented"
    );
    // The same provenance must travel with the run summary, so a paper table
    // read from `run.summary` alone can see it.
    let run_line = run_summary(workspace.path()).summary;
    let summary_selection = summary_value(&run_line, "context_selection");
    assert_eq!(
        summary_selection, *selection,
        "run.summary must carry the same provenance the context.select row recorded"
    );
}

/// [VANTA-WHY] A21 follow-up item 3 — the dedup arm of the terminal fallback.
///
/// The fallback tells a late round that a repeat is an ACCEPTABLE answer BECAUSE
/// duplicates are discarded automatically. This pins the other half of that
/// promise: a round whose three Attacker tests are ALL duplicates of the shared
/// set ends NORMALLY. Every duplicate is dropped before it is audited, the round
/// still runs the shared set, and the run advances to the approval — no audit
/// call is made for a round with nothing to audit, and nothing is an error.
///
/// The fixture repeats the SAME three preservation probes every round, so from
/// round 2 the shared set IS those three and every generated test is a duplicate.
#[test]
fn a_round_of_all_duplicates_is_dropped_and_the_run_advances() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(THREE_PRESERVATION_TESTS);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);

    assert!(
        status.success(),
        "a round that only repeats the shared set is not an error: the candidate still passes"
    );

    // One generate call per round (3). The intent audit runs ONLY for round 1's
    // freshly generated probes: rounds 2 and 3 have nothing left after the dedup,
    // so there is nothing to audit and no further call may be made.
    assert_eq!(
        server.request_kinds(),
        vec![
            "generate",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "generate",
            "generate"
        ],
        "rounds 2 and 3 drop all three probes, so they must make no audit call"
    );

    let records = read_session_records(&find_session_log(workspace.path()));

    let dedup_rows: Vec<&SessionRecord> = records
        .iter()
        .filter(|r| r.phase == "attacker.dedup")
        .collect();
    assert_eq!(dedup_rows.len(), 3, "one dedup row per round");

    // Round 1 has no shared set yet, so it drops nothing.
    assert!(
        dedup_rows[0].summary.contains("shared_set_size=0")
            && dedup_rows[0].summary.contains("generated=3")
            && dedup_rows[0].summary.contains("dropped_duplicates=0")
            && dedup_rows[0].summary.contains("remaining=3"),
        "round 1 has no shared set yet, so it drops nothing; got {}",
        dedup_rows[0].summary
    );
    // Rounds 2 and 3 repeat the whole shared set: all three dropped, none left.
    for row in &dedup_rows[1..] {
        assert!(
            row.summary.contains("shared_set_size=3")
                && row.summary.contains("generated=3")
                && row.summary.contains("dropped_duplicates=3")
                && row.summary.contains("remaining=0"),
            "a fully repeated round must drop all three and keep none; got {}",
            row.summary
        );
    }

    // Nothing to audit means no audit: the three intent.check records all belong
    // to round 1, the round that actually generated the probes.
    let audit_rows: Vec<&SessionRecord> = records
        .iter()
        .filter(|r| r.phase.starts_with("intent.check["))
        .collect();
    assert_eq!(
        audit_rows.len(),
        3,
        "exactly round 1's three probes are audited; a dropped round is never audited"
    );
    assert!(
        audit_rows
            .iter()
            .all(|r| r.summary.contains("source=attacker.generate")),
        "every audit must be a generate-side audit, not a BATE revision"
    );
    let round_one_iteration = dedup_rows[0].testing_iteration_id.as_ref();
    assert!(
        audit_rows
            .iter()
            .all(|r| r.testing_iteration_id.as_ref() == round_one_iteration),
        "every audit record must belong to the round that generated its probes"
    );

    // The run advances: three rounds ran, and the candidate is approved at the
    // end of the attack depth rather than erroring out on a duplicate-only round.
    let attempt_summaries: Vec<&SessionRecord> = records
        .iter()
        .filter(|r| r.phase == "attempt.summary")
        .collect();
    assert_eq!(attempt_summaries.len(), 3, "all three rounds ran");
    assert!(
        attempt_summaries
            .last()
            .unwrap()
            .summary
            .contains("decision=AllTestsPassed"),
        "the candidate must be approved at the end of the attack depth; got {}",
        attempt_summaries.last().unwrap().summary
    );
    assert!(
        !records.iter().any(|r| r.phase == "run.infra_failure"),
        "a duplicate-only round is normal, not an infrastructure failure"
    );
    assert!(
        run_summary(workspace.path())
            .summary
            .contains("final_decision=AllTestsPassed"),
        "the run's terminal decision must be the approval"
    );
}

/// The mirror of the import bug.
///
/// The Infrastructure reroute exists so a harness that fails to build a tree is
/// never blamed on the model. That reroute only fires when ruff AND mypy are
/// clean. This test pins the other side of that rule: when the patch itself
/// introduces a real syntax error, ruff/mypy must see it and the result must stay
/// Static. Without this, "fix the false positive" and "hide every real syntax
/// error" are indistinguishable.
#[test]
fn syntax_error_patch_is_classified_static_not_infrastructure() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), SYNTAX_ERROR_PATCH);
    let server = StubLlmServer::start(THREE_SIGNAL_TESTS);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(
        !status.success(),
        "a patch that breaks syntax must never be approved"
    );

    let records = read_session_records(&find_session_log(workspace.path()));

    let patched_rows: Vec<&str> = records
        .iter()
        .filter(|r| r.phase.starts_with("patched.result["))
        .map(|r| r.summary.as_str())
        .collect();
    assert!(
        !patched_rows.is_empty(),
        "the patched run must produce per-test result rows"
    );
    assert!(
        patched_rows
            .iter()
            .any(|s| s.contains("failure_class=Static")),
        "a real syntax error introduced by the patch must stay Static; got {patched_rows:#?}"
    );
    assert!(
        !patched_rows
            .iter()
            .any(|s| s.contains("failure_class=Infrastructure")),
        "a real syntax error must never be excused as harness Infrastructure; got {patched_rows:#?}"
    );

    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final M3_repair_loop row must exist");
    assert!(
        final_row.summary.contains("Rejected"),
        "a syntax error the patch introduces is a Static finding in a file the \
         patch touches, so review mode must REJECT the candidate rather than keep \
         attacking a tree it has already found broken; got {}",
        final_row.summary
    );
}

/// In --patch (review) mode the candidate never changes, so every comparison is
/// attack depth against the SAME code, not a patch transition. That fact has to
/// be readable off the row itself: with only a destination id these rows would
/// be indistinguishable from genuine repair transitions, and an analysis that
/// groups "patch transitions" would silently absorb review rounds. Both ends are
/// stamped, so the exclusion is the single condition from == to.
#[test]
fn review_rounds_record_equal_patch_ids_on_every_row() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // [VANTA-WHY] In review mode a round that finds a Logic failure ends the run
    // (a rejection is terminal), so only a candidate that SURVIVES every round
    // reaches rounds 2 and 3 — and only a run that reaches them can record the
    // 1->2 and 2->3 comparisons this test is about. A behaviour-preserving
    // candidate plus a preservation-only probe is that candidate.
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(
        status.success(),
        "a candidate that survives the whole attack depth is the approval"
    );
    assert_eq!(
        server.request_count(),
        4,
        "three review rounds must all run (plus the round-1 intent audit)"
    );

    let rows = read_transition_rows(&find_transitions(workspace.path()));
    assert!(
        !rows.is_empty(),
        "three review rounds must record the 1->2 and 2->3 comparisons"
    );
    // Attack-depth rows: both ends observed and EQUAL. A null end is not attack
    // depth — it is a hole, and must not be counted as one of these rows.
    let offenders: Vec<(u32, u32, String, String)> = rows
        .iter()
        .filter(|r| {
            !(r.from_patch_attempt_id.is_some() && r.from_patch_attempt_id == r.to_patch_attempt_id)
        })
        .map(|r| {
            (
                r.from_attempt,
                r.to_attempt,
                format!("{:?}", r.from_patch_attempt_id),
                format!("{:?}", r.to_patch_attempt_id),
            )
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "--patch mode never changes the candidate, so every row must be attack depth (both ids observed and equal); offenders: {offenders:?}"
    );
}

/// The complement of the review-round test: when the candidate DOES change
/// between rounds, the row must say so, or the patch-transition analysis
/// (`from != to`) would drop real repair transitions. The candidate is swapped
/// after the first round so attempt 1 runs patch A and attempt 2 runs patch B.
#[test]
fn changing_the_patch_between_rounds_records_a_patch_transition() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // Patch A must PRESERVE behaviour so round 1 passes and the review continues
    // to round 2 — a rejection on round 1 is terminal and would leave no
    // comparison to inspect.
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);

    // Fired on the first request — i.e. during attempt 1, after it has already
    // read patch A. Attempt 2 then reads patch B, so 1->2 is a real transition.
    let replacement = patch_path.clone();
    let hook = move |index: usize| {
        if index == 0 {
            fs::write(&replacement, SYNTAX_ERROR_PATCH).unwrap();
        }
    };
    let server = StubLlmServer::start_with_hook(SINGLE_BEHAVIOR_TEST, Some(Box::new(hook)));

    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(!status.success(), "neither candidate should be approved");

    let rows = read_transition_rows(&find_transitions(workspace.path()));
    // The corrected patch-transition rule: a row counts only when BOTH ids are
    // observed and differ. A null end disqualifies the row — it is a hole, not a
    // transition.
    let transitions: Vec<(u32, u32, String, String)> = rows
        .iter()
        .filter(|r| {
            r.from_patch_attempt_id.is_some()
                && r.to_patch_attempt_id.is_some()
                && r.from_patch_attempt_id != r.to_patch_attempt_id
        })
        .map(|r| {
            (
                r.from_attempt,
                r.to_attempt,
                format!("{:?}", r.from_patch_attempt_id),
                format!("{:?}", r.to_patch_attempt_id),
            )
        })
        .collect();
    assert!(
        !transitions.is_empty(),
        "a candidate change between rounds must surface as at least one patch transition (both ids observed and !=); got {} row(s) total",
        rows.len()
    );
}

/// Two durability contracts are exercised here. Rows are flushed as each
/// transition is computed, and the session log is appended as each record is
/// produced — `session.start` first. A run that dies mid-way (a `HarnessFailure`
/// is terminal by design, and the sandbox has already filled the disk once) must
/// still leave: (a) every row it produced, and (b) enough of the session log to
/// say WHICH model produced those rows. Here the harness is killed at attempt 3's
/// attacker call, so the 1->2 rows are all that exist.
#[test]
fn transitions_survive_an_abort_after_attempt_two() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // [VANTA-WHY] Reaching attempt 2 needs a candidate that passes attempt 1:
    // a review-mode rejection on round 1 is terminal and would abort on its own
    // terms, not at the request this test drops. A preserving candidate plus a
    // preservation-only probe gets the run as far as attempt 3.
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);

    // [VANTA-WHY] Round 1 now spends TWO requests (generate, then the patch-blind
    // intent audit of the probe), so attempt 3's generate call is request 3, not
    // 2. Dropping that one aborts the run after attempt 2 has completed — which
    // is what leaves the 1->2 rows on disk.
    let server = StubLlmServer::start_aborting(SINGLE_BEHAVIOR_TEST, 3);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(!status.success(), "an aborted run must not report success");

    // The session log is appended as records are produced, so the run's
    // attribution survives the abort. `session.start` was written before attempt
    // 1; without an append-as-you-go log the abort would have destroyed it,
    // leaving rows saying WHAT changed with no record of WHICH model produced
    // them.
    let records = read_session_records(&find_session_log(workspace.path()));
    assert_eq!(
        records.first().map(|r| r.phase.as_str()),
        Some("session.start"),
        "the run identity must be on disk before attempt 1 and survive the abort"
    );
    let start = &records[0];
    assert!(
        start.summary.contains("model=stub-model"),
        "the surviving log must name the model that produced the rows; got {}",
        start.summary
    );
    assert!(
        records.iter().any(|r| r.phase == "attempt.start"),
        "attempts 1 and 2 ran before the abort, so their start records must be on disk too"
    );
    // [VANTA-WHY] A dropped LLM call no longer kills the process. It ends the run
    // cleanly as `InfraFailure`, so the end-of-run records NOW EXIST — the
    // attribution and the on-disk rows are what must survive, and the decision
    // must say the HARNESS failed, never that the candidate was approved.
    let decision_record = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("an LLM call that failed must end the run with a recorded decision");
    assert!(
        decision_record.summary.contains("final_decision=InfraFailure"),
        "an aborted LLM call is an infrastructure failure, not an approval; got {}",
        decision_record.summary
    );

    // read_transition_rows parses line by line, so a clean read proves the rows
    // on disk are complete, not a truncated blob.
    let rows = read_transition_rows(&find_transitions(workspace.path()));
    assert!(
        rows.iter()
            .any(|r| r.from_attempt == 0 && r.to_attempt == 1),
        "the attempt 1->2 rows must survive an abort at attempt 3; got {:?}",
        rows.iter()
            .map(|r| (r.from_attempt, r.to_attempt))
            .collect::<Vec<_>>()
    );
    assert!(
        rows.iter().all(|r| r.to_attempt < 2),
        "attempt 3 never completed, so no row may claim to include it; got {:?}",
        rows.iter()
            .map(|r| (r.from_attempt, r.to_attempt))
            .collect::<Vec<_>>()
    );
}

/// Regression guard for BATE Layer 2 — the Attacker revision path.
///
/// [VANTA-WHY] A test whose declared intent contradicts its baseline is not
/// simply thrown away. BATE sends it back to the Attacker with the mismatch
/// reason (up to 3 times) and only scraps it once every revision still
/// disagrees. That behaviour was silently missing from M3 for a while and the
/// suite stayed green, because nothing exercised it — this test is what makes
/// its absence loud.
///
/// The stub answers every request with the SAME intent-contradicting body, so
/// the run can only exceed one LLM call by actually going back to the Attacker.
/// A single attempt keeps the arithmetic honest: extra requests can only come
/// from inside BATE.
#[test]
fn a_test_whose_baseline_contradicts_its_intent_is_revised_before_being_scrapped() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // [VANTA-WHY] Fix 6 runs a relabel gate before each revision: a
    // BugReproduction test that keeps PASSING its baseline is asked whether the
    // issue requires the asserted behaviour to change. The stub's DEFAULT audit
    // answer does not carry `issue_requires_change` at all — which is the case
    // the gate must fail CLOSED on: an omitted field is UNKNOWN, so the test
    // takes the ordinary revision path rather than being relabelled.
    let server = StubLlmServer::start(CONTRADICTORY_INTENT_TEST);

    let status = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !status.success(),
        "a run whose only test is scrapped must not report success"
    );

    // [VANTA-WHY] Each non-preservation test is audited by the patch-blind
    // intent check: once for the initial batch, once per relabel gate, and once
    // per revision. With one BugReproduction test that is 1 generation + 1 batch
    // audit + 3 x (relabel gate + revision + revised-test audit) + the final
    // relabel gate that ends the budget = 12 LLM calls. Scrapping on the first
    // mismatch — the regressed behaviour — would leave this at 2 (the generation
    // and its audit).
    assert_eq!(
        server.request_count(),
        12,
        "a test whose baseline contradicts its intent must be sent back to the \
         Attacker for revision (up to 3 times) before being scrapped; \
         expected 1 generation + 1 batch audit + 3 x (relabel gate + revision + \
         revised audit) + the final relabel gate = 12 LLM calls"
    );
    assert_eq!(
        server.request_kinds(),
        vec![
            "generate",
            "intent_audit",
            "intent_audit",
            "revise",
            "intent_audit",
            "intent_audit",
            "revise",
            "intent_audit",
            "intent_audit",
            "revise",
            "intent_audit",
            "intent_audit",
        ],
        "the regression guard must prove the exact call sequence — one generate, \
         then a patch-blind audit of it, then three genuine revision/audit pairs \
         (each gated by the Fix-6 relabel question) — so internal parse retries \
         cannot masquerade as a working revision path"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    assert!(
        records
            .iter()
            .any(|r| r.phase == "bate.scrap[test_intent_contradiction]"),
        "a test that still disagrees with its intent after every revision must be \
         scrapped, and the scrap must be recorded; got {:?}",
        records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
    );
    let bate_summary = records
        .iter()
        .find(|r| r.phase == "bate.summary")
        .expect("the BATE summary row must exist");
    assert!(
        bate_summary.summary.contains("discard_rate_pct=100.0"),
        "a run that scraps its only test must record a 100%% discard rate; got {}",
        bate_summary.summary
    );
    // [VANTA-WHY] §14.14 — the round's Attacker remediation budget must be on the
    // record, not just enforced: the cap AND how much of it this round spent. The
    // three revisions above are exactly the three this test just proved happened,
    // so `revise_calls_used=3` is the receipt for them — a budget that was never
    // spent, or one written down as the cap rather than the usage, would read
    // differently here.
    assert!(
        bate_summary.summary.contains("revise_budget=4"),
        "the BATE summary must record the attempt's revise budget; got {}",
        bate_summary.summary
    );
    assert!(
        bate_summary.summary.contains("revise_calls_used=3"),
        "the BATE summary must record the three revisions this round spent; got {}",
        bate_summary.summary
    );

    // [VANTA-WHY] §14.17 — the revise request must carry what the baseline
    // ACTUALLY did, not just BATE's category. The stub's test passes on the
    // original tree, so the honest account is "PASSED", and telling the model
    // that is the difference between revising the test and guessing why it came
    // back wrong: the overnight sanic session rewrote one test 45 times because
    // "expects baseline=Failed but got Error" was all it ever saw.
    let revise_bodies: Vec<String> = server
        .request_kinds()
        .into_iter()
        .zip(server.request_bodies())
        .filter(|(kind, _)| kind == "revise")
        .map(|(_, body)| body)
        .collect();
    assert_eq!(
        revise_bodies.len(),
        3,
        "the three revisions this test proves happened are the three bodies to inspect"
    );
    for body in revise_bodies {
        assert!(
            body.contains("WHAT THE BASELINE ACTUALLY DID"),
            "the revise prompt must carry the run's own output, not only the mismatch \
             category: {body}"
        );
        assert!(
            body.contains("verdict: PASSED"),
            "the baseline passed on the original tree, and that fact is exactly what \
             the model has to be told: {body}"
        );
    }
}

/// Regression guard for the oracle loophole: a patch must not be able to write
/// its own acceptance test.
///
/// [VANTA-WHY] The Attacker writes tests FROM the patch, so it can always
/// relabel a patch-introduced behaviour change as FeatureAcceptance — which BATE
/// permits to fail on the baseline — and thereby approve itself. The real
/// fixture matrix did exactly this: it labelled `shipping_band(500) -> "light"`
/// as FeatureAcceptance and quoted a genuine issue sentence that *forbade* the
/// change. A presence-only quote check admitted it and the bad patch was
/// approved twice, on both binaries.
///
/// The fix is a PATCH-BLIND audit: a separate LLM call sees only the issue, the
/// original source and the test — never the patch — and answers whether the
/// issue actually asks for the asserted behaviour. The stub reproduces the
/// loophole's exact shape: the quote IS verbatim in the issue (so any
/// presence-only check would have admitted it), yet the audit answers that the
/// issue does not ask for the change. The test is demoted to
/// BehaviorPreservation, fails the original baseline as a mismatch, goes back to
/// the Attacker once, and the revised probe rejects the bad patch as Logic.
#[test]
fn patch_written_feature_acceptance_without_valid_issue_quote_is_relabelled_then_rejected() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // The audit is what demotes: it answers "the issue does NOT ask for this".
    let server = StubLlmServer::start_with_intent_audit_and_revision(
        ISSUE_SCOPE_LOOPHOLE_TEST,
        ISSUE_SCOPE_REVISED_PRESERVATION_TEST,
        r#"{"issue_asks_for_behaviour":false,"reason":"the issue forbids changing the other thresholds"}"#,
    );

    let status = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !status.success(),
        "a bad patch must be rejected once its patch-written acceptance test is relabelled honestly"
    );
    // One generation, one patch-blind audit, one revision, and one audit OF that
    // revision: the audit runs for EVERY test, BehaviourPreservation included, so
    // a preservation probe cannot slip past it just by carrying the safe label.
    assert_eq!(
        server.request_kinds(),
        vec!["generate", "intent_audit", "revise", "intent_audit"],
        "the invalid feature-acceptance test should be audited (and demoted), fail BATE as preservation, go back once for revision, and have that revision audited too"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    // The patch-blind audit must have run and decided, and it must have decided
    // AGAINST the label while the quote evidence still looked admissible — the
    // whole point being that the substring result is logged, not enforced.
    let check = records
        .iter()
        .find(|r| r.phase == "intent.check[test_patch_written_feature]")
        .unwrap_or_else(|| {
            panic!(
                "the session log must record the patch-blind intent verdict; got phases {:?}",
                records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
            )
        });
    assert!(
        check.summary.contains("issue_asks_for_behaviour=false"),
        "the audit must decide against the declared intent; got {}",
        check.summary
    );
    assert!(
        check.summary.contains("quote_present=true")
            && check.summary.contains("quote_in_issue=true"),
        "the quote WAS verbatim in the issue, so a presence-only check would have kept \
         FeatureAcceptance; the demotion must come from the audit; got {}",
        check.summary
    );

    let relabel = records
        .iter()
        .find(|r| r.phase == "attacker.relabel[test_patch_written_feature]")
        .unwrap_or_else(|| {
            panic!(
                "the session log must record the relabel event; got phases {:?}",
                records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
            )
        });
    assert!(
        relabel.summary.contains("from=FeatureAcceptance")
            && relabel.summary.contains("to=BehaviorPreservation"),
        "got {}",
        relabel.summary
    );
    assert!(
        relabel.summary.contains("the issue does not ask for"),
        "the relabel must carry the audit's reason; got {}",
        relabel.summary
    );
    assert!(
        records
            .iter()
            .any(|r| r.phase == "patched.result[test_patch_written_feature]"
                && r.summary.contains("outcome=Counterexample")
                && r.summary.contains("failure_class=Logic")),
        "the revised preservation probe must reject the bad patch as a Logic failure; got {:?}",
        records
            .iter()
            .filter(|r| r.phase == "patched.result[test_patch_written_feature]")
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
    );
}

/// The intent audit must never see the patch.
///
/// [VANTA-WHY] The whole reason the audit exists is that it is asked from OUTSIDE
/// the patch: it can only judge whether the ISSUE asks for a behaviour if it does
/// not read the patch's own answer back as the requirement. This asserts it on
/// the wire — the bytes the stub server received for the audit request must carry
/// the ORIGINAL source (positive control) and must NOT carry the patch diff or the
/// patched source. The audit request body is what a future refactor would leak
/// into first, silently re-opening the loop the auditor was added to close.
#[test]
fn intent_check_request_never_contains_the_patch() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // The loophole test declares FeatureAcceptance, so exactly one audit runs. The
    // audit keeps the label here — this test is about the request body, not the
    // outcome — so the run does not take the revision path.
    let server = StubLlmServer::start_with_intent_audit(
        ISSUE_SCOPE_LOOPHOLE_TEST,
        r#"{"issue_asks_for_behaviour":true,"reason":"stub: keep the label"}"#,
    );

    let _ = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);

    let kinds = server.request_kinds();
    let bodies = server.request_bodies();
    let audits: Vec<(usize, &str)> = kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| kind.as_str() == "intent_audit")
        .map(|(index, _)| (index, bodies[index].as_str()))
        .collect();
    assert_eq!(
        audits.len(),
        1,
        "the single FeatureAcceptance test must be audited exactly once; got kinds {kinds:?}"
    );
    for (index, body) in audits {
        assert!(
            !body.contains("diff --git") && !body.contains("+++"),
            "intent audit request {index} must not carry the patch diff; got {body}"
        );
        assert!(
            !body.contains("return a - b"),
            "intent audit request {index} must not carry the patched source; got {body}"
        );
        assert!(
            body.contains("return a + b"),
            "the audit must still see the ORIGINAL source, so the assertions above cannot \
             pass on an empty body; got {body}"
        );
    }
}

/// The "except where the issue requires the change" clause must not live in
/// prompt wording alone.
///
/// [VANTA-WHY] The boundary rule tells the Attacker to write a
/// BehaviourPreservation probe at the OLD boundary value — except where the issue
/// requires that behaviour to change. A prompt cannot enforce "except": an
/// Attacker that reads it too loosely emits a probe asserting the OLD value at an
/// input the issue moves, and that probe then fails on a CORRECT patch and
/// rejects it. The intent check closes the hole by asking the complementary
/// question ("does the issue require this behaviour to change?") and scrapping
/// the probe. This test pins that path: the probe is dropped, the drop is logged
/// as a verdict for the probe, and nothing runs against the patched tree.
#[test]
fn a_preservation_probe_the_issue_requires_to_change_is_scrapped_by_the_intent_check() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // The audit answers the SECOND question: the issue requires the behaviour at
    // this input to change, so a preservation probe there contradicts it.
    let server = StubLlmServer::start_with_intent_audit(
        PRESERVATION_PROBE_CONTRADICTING_THE_ISSUE,
        r#"{"issue_asks_for_behaviour":false,"issue_requires_change":true,"reason":"the issue asks for a different result at this input"}"#,
    );

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "with its only probe scrapped there is nothing left to verify, so the run cannot approve"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let check = records
        .iter()
        .find(|r| r.phase == "intent.check[test_preserve_old_boundary]")
        .unwrap_or_else(|| {
            panic!(
                "the session log must carry a verdict for the preservation probe too, not just \
                 for non-preservation tests; got phases {:?}",
                records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
            )
        });
    assert!(
        check.summary.contains("scrapped=true"),
        "a preservation probe the issue requires to change must be scrapped, and the scrap \
         must be logged as a verdict; got {}",
        check.summary
    );
    assert!(
        check.summary.contains("issue_requires_change=true"),
        "the log must show WHICH question scrapped it, so a reader can tell a scrap from a \
         demotion; got {}",
        check.summary
    );

    let summary = records
        .iter()
        .find(|r| r.phase == "attacker.summary")
        .expect("the Attacker summary row must exist");
    assert!(
        summary.summary.contains("intent_scrapped=1") && summary.summary.contains("valid_tests=0"),
        "the scrapped probe must be counted and must not reach verification; got {}",
        summary.summary
    );
    assert!(
        !records
            .iter()
            .any(|r| r.phase.starts_with("patched.result[")),
        "a scrapped probe must never run against the patched tree; got {:?}",
        records
            .iter()
            .filter(|r| r.phase.starts_with("patched.result["))
            .map(|r| r.phase.as_str())
            .collect::<Vec<_>>()
    );
}

/// Item 6 — a preservation probe the issue ASKS FOR and REQUIRES to change is
/// RELABELLED to FeatureAcceptance, not scrapped.
///
/// [VANTA-WHY] The complementary case to the scrap test above. When the issue
/// asks for the asserted behaviour AND requires it to change at the asserted
/// input, the preservation label is simply wrong: the probe is an acceptance
/// test for the new behaviour, and scrapping it discards the one test that would
/// catch a broken patch. Promoting it is safe because BATE's FeatureAcceptance
/// rule then demands the probe FAIL on the original — it must prove the new
/// behaviour is ABSENT before the patch is even run — so the relabel can only
/// make the test harder to admit, never easier. This pins the whole path: the
/// promotion is logged, the probe is admitted, and it runs against the patched
/// tree.
#[test]
fn a_preservation_probe_the_issue_asks_for_is_promoted_not_scrapped() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), NEW_BEHAVIOUR_PATCH);
    // The audit answers BOTH questions: the issue asks for the behaviour and
    // requires it to change, which is what makes the error an acceptance probe.
    let server = StubLlmServer::start_with_intent_audit(
        PRESERVATION_PROBE_ASSERTING_THE_NEW_BEHAVIOUR,
        r#"{"issue_asks_for_behaviour":true,"issue_requires_change":true,"reason":"the issue asks for the new value at this input"}"#,
    );

    let status = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        status.success(),
        "a probe that proves the new behaviour is absent on the original and present on the \
         patched tree must not sink a correct patch"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    // The promotion must be logged with BOTH endpoints, so a reader can tell a
    // promotion from a demotion without re-deriving the verdict.
    let relabel = records
        .iter()
        .find(|r| r.phase == "attacker.relabel[test_new_behaviour_probe]")
        .unwrap_or_else(|| {
            panic!(
                "the session log must record the promotion; got phases {:?}",
                records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
            )
        });
    assert!(
        relabel.summary.contains("from=BehaviorPreservation")
            && relabel.summary.contains("to=FeatureAcceptance"),
        "the promotion must name both endpoints; got {}",
        relabel.summary
    );
    assert!(
        relabel.summary.contains("the issue asks for the asserted behaviour"),
        "the relabel must carry the audit's reason; got {}",
        relabel.summary
    );

    // It must be admitted, not scrapped, and the promotion counted as a relabel.
    let summary = records
        .iter()
        .find(|r| r.phase == "attacker.summary")
        .expect("the Attacker summary row must exist");
    assert!(
        summary.summary.contains("intent_scrapped=0")
            && summary.summary.contains("relabels=1")
            && summary.summary.contains("valid_tests=1"),
        "a promoted probe must survive the audit and be counted as a relabel; got {}",
        summary.summary
    );

    // And it must actually run against the patched tree and pass.
    //
    // [VANTA-WHY] A passing generated test is recorded with the BATE outcome
    // `VerifiedWithinBounds` (driver.rs maps `TestResult::Passed` to it), not
    // with the word "Passed": the row carries a `VerificationOutcome`, and the
    // promoted probe must also be stamped `intent=FeatureAcceptance` so a reader
    // can see the relabel reached the run.
    assert!(
        records
            .iter()
            .any(|r| r.phase == "patched.result[test_new_behaviour_probe]"
                && r.summary.contains("outcome=VerifiedWithinBounds")
                && r.summary.contains("intent=FeatureAcceptance")
                && r.summary.contains("failure_class=None")),
        "the promoted probe must run against the patched tree and pass; got {:?}",
        records
            .iter()
            .filter(|r| r.phase.starts_with("patched.result["))
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
    );
}

/// [VANTA-WHY] §14.5 — the batched audit is a TRANSPORT change. The same round,
/// the same three answers, must produce the same per-test decisions and records
/// whether they arrived in one call or three; the only observable difference
/// allowed is the number of calls.
#[test]
fn the_batched_audit_is_one_call_per_round_with_the_same_decisions() {
    let _serial = serial_run();

    // One run of `THREE_SIGNAL_TESTS` against the bad patch, with the audit
    // answered in the given shape. Returns the call kinds, the audit bodies, the
    // per-test decisions the run logged and the terminal decision.
    let run = |audit_answer: &str, extra_env: &'static [(&'static str, &'static str)]| {
        let workspace = TempDir::new().unwrap();
        let (_repo_dir, _issue_path, patch_path) =
            write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
        let server = StubLlmServer::start_with_intent_audit(THREE_SIGNAL_TESTS, audit_answer);
        let _ = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, extra_env);

        let records = read_session_records(&find_session_log(workspace.path()));
        let decisions: Vec<(String, String)> = records
            .iter()
            .filter(|r| r.phase.starts_with("intent.check["))
            .map(|r| (r.phase.clone(), r.summary.clone()))
            .collect();
        let terminal = records
            .iter()
            .find(|r| r.phase == "M3_repair_loop")
            .map(|r| r.summary.clone())
            .unwrap_or_default();
        let batch_flag = summary_value(&run_summary(workspace.path()).summary, "batch_audit");
        (
            server.request_kinds(),
            server.request_bodies(),
            decisions,
            terminal,
            batch_flag,
            workspace,
        )
    };

    let (kinds_off, _bodies_off, decisions_off, terminal_off, flag_off, _off_workspace) =
        run(SINGLE_AUDIT_KEEP_ANSWER, &[]);
    let (kinds_on, bodies_on, decisions_on, terminal_on, flag_on, _on_workspace) = run(
        THREE_SIGNAL_BATCH_AUDIT_ANSWER,
        &[("VANTA_BATCH_AUDIT", "on")],
    );

    assert_eq!(
        kinds_off,
        vec!["generate", "intent_audit", "intent_audit", "intent_audit"],
        "with the flag off each test is audited in its own call — the v0.2 schedule"
    );
    assert_eq!(
        kinds_on,
        vec!["generate", "intent_audit"],
        "with the flag on the round's three tests are audited in ONE call"
    );
    assert_eq!(flag_off, "off");
    assert_eq!(flag_on, "on");

    // The one batched call must carry the WHOLE round, each test numbered — that
    // numbering is what binds an answer to a test.
    let batch_body = &bodies_on[1];
    for numbered in [
        "TEST 1 — test_bug_regression_signal",
        "TEST 2 — test_feature_acceptance_signal",
        "TEST 3 — test_behavior_preservation_signal",
    ] {
        assert!(
            batch_body.contains(numbered),
            "the batched audit request must carry every numbered test; missing {numbered:?} \
             from {batch_body}"
        );
    }

    assert_eq!(
        decisions_off.len(),
        3,
        "every test gets a verdict, preservation included; got {decisions_off:?}"
    );
    assert_eq!(
        decisions_off, decisions_on,
        "the same three answers must produce the same verdicts and the same records \
         whichever transport carried them"
    );
    assert_eq!(
        terminal_off, terminal_on,
        "a transport change may not move the run's decision"
    );
    assert!(
        terminal_on.contains("Rejected"),
        "three failing generated tests are still a rejection; got {terminal_on}"
    );
}

/// [VANTA-WHY] §14.5 — an answer that does not cover the round is a parse failure
/// of the whole response, so it is retried like any other malformed structured
/// output; and when it finally fails, every test falls back to EXACTLY the
/// single-call failure shape. Both halves are asserted against the flag-off run
/// of the same fixture, whose calls fail the same way at the same place.
///
/// This is also the other half of the truncation test's contract: an answer that
/// was NOT cut off keeps its retries, and the session log tells the two events
/// apart by name — `malformed_retried` here against `truncated_fallback` there.
#[test]
fn a_batch_the_auditor_cannot_answer_falls_back_to_the_single_call_failure_shape() {
    let _serial = serial_run();

    let run = |extra_env: &'static [(&'static str, &'static str)]| {
        let workspace = TempDir::new().unwrap();
        let (_repo_dir, _issue_path, patch_path) =
            write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
        // The SAME unparseable answer in both transports: one entry for a
        // three-test round.
        let server =
            StubLlmServer::start_with_intent_audit(THREE_PRESERVATION_TESTS, SHELL_BATCH_AUDIT_ANSWER);
        let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, extra_env);

        let records = read_session_records(&find_session_log(workspace.path()));
        let decisions: Vec<String> = records
            .iter()
            .filter(|r| r.phase.starts_with("intent.check["))
            .map(|r| r.summary.clone())
            .collect();
        let transports: Vec<String> = records
            .iter()
            .filter(|r| r.phase == "intent.audit_batch")
            .map(|r| r.summary.clone())
            .collect();
        (
            server.request_kinds(),
            server.request_bodies(),
            decisions,
            transports,
            output,
        )
    };

    let (kinds_off, bodies_off, decisions_off, transports_off, output_off) = run(&[]);
    let (kinds_on, bodies_on, decisions_on, transports_on, output_on) =
        run(&[("VANTA_BATCH_AUDIT", "on")]);

    // Off: three tests, each call retried the full budget = 1 + 3x3.
    assert_eq!(
        kinds_off,
        vec![
            "generate",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
            "intent_audit",
        ],
        "with the flag off an unparseable audit is retried per test"
    );
    // On: ONE logical call, retried three times. The retries are the point: this
    // answer was NOT cut off, so the model can answer it differently next time,
    // and only a cut-off answer skips the budget (see the truncation test).
    assert_eq!(
        kinds_on,
        vec!["generate", "intent_audit", "intent_audit", "intent_audit"],
        "an incomplete batch must be retried as ONE call, not split into three"
    );
    // …and the session log names the event, since the call count no longer has to
    // carry that distinction on its own.
    assert!(
        transports_off.is_empty(),
        "with the flag off no batched call is made; got {transports_off:?}"
    );
    assert_eq!(
        transports_on,
        vec!["source=attacker.generate transport=malformed_retried tests=3"],
        "the record must say the batch was MALFORMED and retried, not cut off"
    );
    for body in &bodies_on[1..] {
        assert!(
            body.contains("TEST 3 — test_preserve_add_identity"),
            "every retry must carry the whole round, not the tests answered so far; got {body}"
        );
    }
    assert!(
        bodies_off[1..].iter().all(|body| !body.contains("TEST 2 —")),
        "with the flag off a request carries exactly one test"
    );

    assert_eq!(decisions_off.len(), 3);
    assert_eq!(decisions_on.len(), 3);
    for (summary_off, summary_on) in decisions_off.iter().zip(&decisions_on) {
        // The DECISION is what must not depend on the transport: the same fields,
        // the same values. The reason quotes the parse error, which necessarily
        // names a different failure, so it is checked for the shared marker only.
        for field in [
            "issue_asks_for_behaviour=false",
            "issue_requires_change=unknown",
            "assertions_grounded=unknown",
            "scrapped=false",
        ] {
            assert!(
                summary_off.contains(field) && summary_on.contains(field),
                "a failed audit must take the same safe shape in both transports: {field} \
                 is missing from {summary_off:?} or {summary_on:?}"
            );
        }
        assert!(
            summary_off.contains("intent audit unavailable")
                && summary_on.contains("intent audit unavailable"),
            "the record must say the audit was unavailable; got {summary_off:?} and {summary_on:?}"
        );
    }

    // A failed auditor demotes the round; it may not end the run as an
    // infrastructure failure, and it may not approve the bad patch.
    for (status, summary) in [
        (output_off.status, &decisions_off[0]),
        (output_on.status, &decisions_on[0]),
    ] {
        assert!(!status.success(), "the bad patch must not be approved: {summary}");
    }
}

/// [VANTA-WHY] §14.5 — a batched answer the server CUT OFF at `max_tokens`
/// (`finish_reason: "length"`) is the one batch failure the per-test path FIXES:
/// the truncation is caused by asking for N answers in one capped generation, so
/// one test per call is answered easily. The round must therefore keep the
/// verdicts the flag-off run gets, and no record may say the audit was
/// unavailable. The truncation is forced on the BATCH request only, so what the
/// test measures is the fallback rather than an unanswerable round (the shape
/// `a_batch_the_auditor_cannot_answer…` covers).
///
/// [VANTA-WHY] §14.5 item 2 — the fallback is reached after TWO batch calls, not
/// one. A cut-off answer is never parsed and never re-sent unchanged: the batch
/// is asked once more with the short `answer directly` instruction appended, and
/// when THAT is cut off as well the call gives up with its ordinary failure value
/// (here: the `truncated_fallback` transport, i.e. the per-test path). The two
/// batch bodies are asserted to differ for exactly this reason — a pipeline that
/// re-sent the identical request would still fall back, and would cost one more
/// full-length generation to do it.
#[test]
fn a_batch_cut_off_at_the_cap_falls_back_to_per_test_audits_with_the_same_decisions() {
    let _serial = serial_run();

    let run = |extra_env: &'static [(&'static str, &'static str)], truncate: bool| {
        let workspace = TempDir::new().unwrap();
        let (_repo_dir, _issue_path, patch_path) =
            write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
        let server = StubLlmServer::start_with_intent_audit(
            THREE_PRESERVATION_TESTS,
            SINGLE_AUDIT_KEEP_ANSWER,
        );
        if truncate {
            server.truncate_batch_audit();
        }
        let _ = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, extra_env);

        let records = read_session_records(&find_session_log(workspace.path()));
        let decisions: Vec<(String, String)> = records
            .iter()
            .filter(|r| r.phase.starts_with("intent.check["))
            .map(|r| (r.phase.clone(), r.summary.clone()))
            .collect();
        let transports: Vec<String> = records
            .iter()
            .filter(|r| r.phase == "intent.audit_batch")
            .map(|r| r.summary.clone())
            .collect();
        (
            server.request_kinds(),
            server.request_bodies(),
            decisions,
            transports,
            run_summary(workspace.path()).summary,
            workspace,
        )
    };

    let (kinds_off, _bodies_off, decisions_off, transports_off, summary_off, _off_workspace) =
        run(&[], false);
    let (kinds_on, bodies_on, decisions_on, transports_on, summary_on, _on_workspace) =
        run(&[("VANTA_BATCH_AUDIT", "on")], true);

    assert_eq!(
        kinds_off,
        vec!["generate", "intent_audit", "intent_audit", "intent_audit"],
        "with the flag off each test is audited in its own call — the v0.2 schedule"
    );
    assert_eq!(
        kinds_on,
        vec![
            "generate",
            // the batch: cut off at the cap, so it is NEVER parsed and never
            // re-sent unchanged — the same round is asked again with a short
            // `answer directly` instruction appended (§14.5 item 2)
            "intent_audit",
            // and that re-ask is cut off too, which is the batch saying it cannot
            // answer inside its cap: the call gives up rather than spending the
            // rest of the retry budget on full-length generations that stop at the
            // same prefix
            "intent_audit",
            // then the fallback: one call per test, which is answered
            "intent_audit",
            "intent_audit",
            "intent_audit",
        ],
        "a batch cut off at the cap must be re-asked ONCE with the brief instruction and then fall \
         back to one audit call PER TEST — not spend the retry budget on the identical request, \
         and not demote the whole round"
    );

    // Which call was which is the point: the two first audit calls carried the
    // whole round (the request, then the brief re-ask), the fallback carried one
    // test each.
    assert!(
        bodies_on[1].contains("TESTS TO AUDIT"),
        "the second audit call is the batch; got {}",
        bodies_on[1]
    );
    assert!(
        bodies_on[2].contains("TESTS TO AUDIT")
            && bodies_on[2].contains("CUT OFF at the output limit"),
        "the batch's re-ask must be the SAME question with the brief instruction appended, never \
         the identical request; got {}",
        bodies_on[2]
    );
    assert!(
        !bodies_on[1].contains("CUT OFF at the output limit"),
        "the first batch request must be the pinned one, with no nudge; got {}",
        bodies_on[1]
    );
    for body in &bodies_on[3..] {
        assert!(
            body.contains("TEST UNDER AUDIT") && !body.contains("TESTS TO AUDIT"),
            "the fallback audits one test per call; got {body}"
        );
    }

    assert_eq!(decisions_off.len(), 3);
    assert_eq!(
        decisions_off, decisions_on,
        "the per-test fallback must recover the same verdicts and records as the flag-off run"
    );
    for (phase, summary) in &decisions_on {
        assert!(
            !summary.contains("intent audit unavailable"),
            "every test was answered by the fallback, so no record may say the audit was \
             unavailable; got {phase}: {summary:?}"
        );
    }

    // The session log names the event, because the call pattern can no longer:
    // no record with the flag off (there was no batched call), and exactly one
    // `truncated_fallback` with it on.
    assert!(
        transports_off.is_empty(),
        "with the flag off no batched call is made, so the log must not claim one; got \
         {transports_off:?}"
    );
    assert_eq!(
        transports_on,
        vec!["source=attacker.generate transport=truncated_fallback tests=3"],
        "the record must say the batch was CUT OFF and fell back per test"
    );

    // The output cap is a recorded property of the run: one test's allowance per
    // test in the round, under the ceiling.
    assert!(summary_off.contains("batch_audit=off"), "got {summary_off}");
    assert!(summary_on.contains("batch_audit=on"), "got {summary_on}");
    assert!(
        summary_on.contains("batch_audit_tokens_per_test=8192"),
        "the per-test allowance must be recorded; got {summary_on}"
    );
    assert!(
        summary_on.contains("batch_audit_max_tokens=24576"),
        "three tests × one test's allowance is 24576, under the ceiling; got {summary_on}"
    );
}

/// [VANTA-WHY] §14.5 — a CUT-OFF answer at a call site that samples ABOVE
/// temperature 0.0 must be RE-ASKED, not failed fast.
///
/// A model can spend its whole output cap on reasoning and return no content at
/// all — Ornith does exactly this on the Attacker — and at temperature 0.2 the
/// next attempt is a different sample, which is precisely the remedy a retry is
/// for. A fail-fast scoped to "any truncation" turns one model run-away into a
/// dead run: `InfraFailure`, which in the BugsInPy harness BLOCKS the wave gate
/// (§14.7). Hence the scope: re-ask once, never the identical request, and give up
/// only if the re-ask is cut off too.
///
/// [VANTA-WHY] The harness pins ONE seed per run, and the same seed with the
/// same prompt can reproduce the identical cut-off prefix, so each retry
/// advances the seed by its attempt index. The record has to say which seed each
/// attempt ACTUALLY sent — `VANTA_LLM_SEED` alone no longer answers that — or a
/// run with a retry cannot be reproduced after the fact.
///
/// [VANTA-WHY] The retry is also a DIFFERENT QUESTION: the brief `answer
/// directly` instruction is appended (§14.5 item 2), because on a call site whose
/// temperature is 0.0 the seed offset cannot change the answer and the nudge is
/// the only thing that can.
///
/// [VANTA-WHY] ONE round (`--max-attempts 1`), because the retry under test
/// happens inside a round. In patch mode the Governor treats the cap as attack
/// DEPTH and keeps attacking a candidate that PASSES — a clean round is a
/// sample, not a finding (`governor.rs`) — so at depth 3 this run continues past
/// the round the assertion is about. Every later round re-asks the same stub for
/// the same probe, the dedup drops it as already shared, and the round makes one
/// generation call and NO audit (an empty test list audits nothing). That is 5
/// calls — `generate, generate, intent_audit` then `generate` twice — so the
/// exact-sequence assertion below would be measuring the Governor's depth policy
/// rather than the retry. The `truncated_calls_attacker=1` assertion is what keeps
/// that honest from the other side: a second cut-off anywhere in the run, in any
/// round, would read 2 and fail.
#[test]
fn a_cut_off_attacker_answer_is_re_asked_and_recovers_on_the_next_attempt() {
    let _serial = serial_run();

    let run = |extra_env: &'static [(&'static str, &'static str)]| {
        let workspace = TempDir::new().unwrap();
        let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
        let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
        // The FIRST generation is cut off at the cap; the retry is answered
        // normally, so "retried and recovered" is distinguishable from "gave up".
        server.truncate_next_calls(&["generate"]);
        let output =
            run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, extra_env);

        let records = read_session_records(&find_session_log(workspace.path()));
        // One `call.usage` row per CHAT CALL, written in call order, so the first
        // two rows are this generation's two attempts — which is what makes the
        // seed each attempt sent checkable.
        let usage: Vec<(String, String, String)> = records
            .iter()
            .filter(|r| r.phase == "call.usage")
            .map(|r| {
                (
                    summary_value(&r.summary, "site"),
                    summary_value(&r.summary, "seed"),
                    summary_value(&r.summary, "truncated"),
                )
            })
            .collect();
        (
            output,
            server.request_kinds(),
            server.request_bodies(),
            usage,
            run_summary(workspace.path()).summary,
        )
    };

    let (output_plain, kinds_plain, bodies_plain, usage_plain, summary_plain) = run(&[]);
    let (output_seeded, kinds_seeded, bodies_seeded, usage_seeded, summary_seeded) =
        run(&[("VANTA_LLM_SEED", "4242")]);

    // The regression guard: the run RECOVERS instead of dying. Before the
    // retry was restored this ended as an infrastructure failure (exit 2).
    for (label, output) in [("unseeded", &output_plain), ("seeded", &output_seeded)] {
        assert_eq!(
            output.status.code(),
            Some(0),
            "a cut-off Attacker answer above temperature 0.0 must be retried, and the run must \
             still reach the reference approval ({label}); stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    // One cut-off, one retry: the second generation is answered, so the run
    // makes exactly two generation calls rather than spending its budget.
    let expected_kinds = vec!["generate", "generate", "intent_audit"];
    assert_eq!(
        kinds_plain, expected_kinds,
        "a cut-off generation must be retried exactly once here, and the recovered answer must \
         carry the round on to its audit"
    );
    assert_eq!(
        kinds_seeded, expected_kinds,
        "the pinned seed must not change how many attempts the retry costs"
    );

    // Unseeded: no attempt may add a seed key — the byte-identity rule is not
    // suspended for retries.
    for body in &bodies_plain {
        assert!(
            !body.contains("\"seed\""),
            "an unset VANTA_LLM_SEED must add NO seed key, retry or not; got: {body}"
        );
    }

    // Seeded: attempt 1 sends the pinned seed, attempt 2 sends pinned + 1.
    assert!(
        bodies_seeded[0].contains("\"seed\":4242") && bodies_seeded[1].contains("\"seed\":4243"),
        "the retry must advance the seed by its attempt index, or the same seed reproduces the \
         same cut-off prefix; got attempt 1: {} attempt 2: {}",
        bodies_seeded[0],
        bodies_seeded[1]
    );
    for body in &bodies_seeded[2..] {
        assert!(
            body.contains("\"seed\":4242"),
            "a call that never retried must send the pinned seed unchanged; got: {body}"
        );
    }

    // §14.5 item 2 — the re-ask is a DIFFERENT question: a cut-off answer is
    // never re-sent unchanged, because the identical request on a pinned
    // temperature reaches the identical cut-off point. It is the first generate
    // body that must be free of the nudge, and the second that must carry it.
    for (label, bodies) in [("unseeded", &bodies_plain), ("seeded", &bodies_seeded)] {
        assert!(
            !bodies[0].contains("CUT OFF at the output limit"),
            "the first attempt is the pinned request, with no nudge ({label}); got {}",
            bodies[0]
        );
        assert!(
            bodies[1].contains("CUT OFF at the output limit"),
            "the re-ask must carry the brief `answer directly` instruction ({label}); got {}",
            bodies[1]
        );
    }

    // The record of what each attempt actually sent.
    let generate_rows = |usage: &[(String, String, String)]| -> Vec<(String, String)> {
        usage
            .iter()
            .filter(|(site, _, _)| site == "attacker_generate")
            .map(|(_, seed, truncated)| (seed.clone(), truncated.clone()))
            .collect()
    };
    assert_eq!(
        generate_rows(&usage_plain),
        vec![
            ("unset".to_string(), "true".to_string()),
            ("unset".to_string(), "false".to_string()),
        ],
        "the session log must show the first attempt cut off with no seed and the second answered"
    );
    assert_eq!(
        generate_rows(&usage_seeded),
        vec![
            ("4242".to_string(), "true".to_string()),
            ("4243".to_string(), "false".to_string()),
        ],
        "the session log must record the seed EACH attempt sent, not the configured one"
    );

    // §14.5's reportable number: how often the model ran away, per role.
    for summary in [&summary_plain, &summary_seeded] {
        assert!(
            summary.contains("truncated_calls_attacker=1"),
            "the run-away must be counted against the Attacker's role; got {summary}"
        );
        assert!(
            summary.contains("truncated_calls_creator=0")
                && summary.contains("truncated_calls_intent_audit=0"),
            "only the Attacker ran away in this run; got {summary}"
        );
    }
}

/// [VANTA-WHY] §14.5 item 2 — a CUT-OFF answer is never re-sent unchanged, at
/// ANY call site. The per-test intent auditor is pinned at temperature 0.0, so
/// the identical request reaches the identical prefix; the retry therefore
/// carries the short `answer directly` instruction, which is a DIFFERENT
/// question and the only kind that can come back different on a greedy sampler.
/// One truncation, one re-ask, and the round keeps the verdict it would have had.
///
/// [VANTA-WHY] ONE round (`--max-attempts 1`), because the retry under test
/// happens inside a round; at greater depth the Governor keeps attacking and the
/// later rounds audit again, which would make the call count measure the depth
/// policy instead of the retry.
#[test]
fn a_cut_off_per_test_audit_answer_is_re_asked_once_with_the_brief_instruction() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    // The one per-test audit of this single-test round is cut off; no other kind
    // is touched, so the count below measures the retry loop and nothing else.
    server.truncate_next_calls(&["intent_audit"]);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);

    let kinds = server.request_kinds();
    assert_eq!(
        kinds.iter().filter(|kind| *kind == "intent_audit").count(),
        2,
        "a cut-off answer must be re-asked exactly ONCE — not re-sent unchanged, and not repeated \
         until the retry budget is gone; kinds={kinds:?}"
    );

    let bodies = server.request_bodies();
    let audits: Vec<&String> = bodies
        .iter()
        .filter(|body| body.contains("TEST UNDER AUDIT"))
        .collect();
    assert!(
        !audits[0].contains("CUT OFF at the output limit"),
        "the first audit request must be the pinned one, with no nudge; got {}",
        audits[0]
    );
    assert!(
        audits[1].contains("CUT OFF at the output limit"),
        "the re-ask must carry the brief instruction, so it is never the identical request; got {}",
        audits[1]
    );

    assert_eq!(
        output.status.code(),
        Some(0),
        "the recovered audit must leave the run at its ordinary verdict; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let summary = run_summary(workspace.path()).summary;
    assert!(
        summary.contains("truncated_calls_intent_audit=1"),
        "the run-away must be counted against the auditor's role; got {summary}"
    );
    assert!(
        summary.contains("truncated_twice_calls=0"),
        "one truncation is not a double truncation; got {summary}"
    );
    assert!(
        summary.contains("truncated_calls_attacker=0") && summary.contains("truncated_calls_creator=0"),
        "only the auditor ran away in this run; got {summary}"
    );
}

/// [VANTA-WHY] §14.5 item 2 — cut off TWICE, so the call gives up. The remedy for
/// "this site cannot answer inside its cap" is a smaller question, not another
/// full-length generation, so the pipeline stops after the brief re-ask and takes
/// the conservative failure value the step ALREADY has. For the per-test auditor
/// that value is its ordinary unavailable verdict — `issue_asks_for_behaviour =
/// false`, the two `Option` fields UNKNOWN — which can demote or keep a test but
/// can never manufacture a rejection, and it must not end the run.
#[test]
fn a_per_test_audit_cut_off_twice_takes_its_ordinary_unavailable_verdict() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    // BOTH requests of this one audit are cut off, so the call is given up on
    // after the brief re-ask rather than retried a third time.
    server.truncate_next_calls(&["intent_audit", "intent_audit"]);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);

    let kinds = server.request_kinds();
    assert_eq!(
        kinds.iter().filter(|kind| *kind == "intent_audit").count(),
        2,
        "cut off twice must end the call after the brief re-ask — no third full-length \
         generation; kinds={kinds:?}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let checks: Vec<&String> = records
        .iter()
        .filter(|r| r.phase.starts_with("intent.check["))
        .map(|r| &r.summary)
        .collect();
    assert!(
        checks.iter().any(|s| s.contains("intent audit unavailable")),
        "the auditor's ordinary conservative value must be what the double cut-off yields; got \
         {checks:?}"
    );

    assert_eq!(
        output.status.code(),
        Some(0),
        "an unavailable auditor must leave the run at its ordinary verdict rather than ending it; \
         stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let summary = run_summary(workspace.path()).summary;
    assert!(
        summary.contains("truncated_calls_intent_audit=2"),
        "both cut-off requests are counted; got {summary}"
    );
    assert!(
        summary.contains("truncated_twice_calls=1"),
        "the run must report that ONE call was cut off twice; got {summary}"
    );
}

/// [VANTA-WHY] §14.5 — the seed offsetting exists for RETRIES only, so a run
/// with the §14 flags OFF and nothing truncated must be inert: attempt 1 of
/// every call sends the pinned seed and nothing else. That is what keeps such a
/// run byte-comparable with the v0.2 baseline the pilot is measured against, and
/// what makes `truncated_calls_*` a proof that no offset ever applied.
///
/// §14.23 changed what the pinned seed means *between* rounds and deliberately
/// left this test's point untouched *inside* one. The offset is now per ROUND —
/// round `r` sends `VANTA_LLM_SEED + (r - 1)`, because the re-asked prompt is a
/// fresh draw rather than a replay — while a retry inside a round remains the
/// only thing that may move the seed further, and nothing was truncated here. So
/// the assertion is per round instead of per run: every call in round `r` carries
/// `4242 + (r - 1)`, all calls within that round carry the same value, and no
/// call carries an offset past its own round. The rounds are read off the paired
/// `call.usage` rows rather than inferred from a body's position, because the row
/// is what records which round sent it.
#[test]
fn a_flag_off_run_without_truncation_never_moves_the_seed() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_LLM_SEED", "4242")],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "the reference run must be the approval; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let bodies = server.request_bodies();
    assert!(!bodies.is_empty(), "the run must have called the LLM");

    let rounds_and_seeds: Vec<(u32, String)> =
        read_session_records(&find_session_log(workspace.path()))
            .iter()
            .filter(|record| record.phase == "call.usage")
            .map(|record| {
                (
                    summary_value(&record.summary, "round").parse().unwrap(),
                    summary_value(&record.summary, "seed"),
                )
            })
            .collect();
    assert_eq!(
        rounds_and_seeds.len(),
        bodies.len(),
        "one `call.usage` row per chat body"
    );

    // `round` filled in call order, so a round seen twice must have sent the same
    // seed both times — which is exactly the "never moves inside a round" half.
    let mut per_round: Vec<(u32, String)> = Vec::new();
    for (body, (round, seed)) in bodies.iter().zip(rounds_and_seeds.iter()) {
        let expected = format!("{}", 4242_u64 + u64::from(round.saturating_sub(1)));
        assert_eq!(
            seed, &expected,
            "round {round} must send the pinned seed advanced by its own index and nothing \
             more (§14.23): with nothing truncated, no retry offset may apply"
        );
        assert!(
            body.contains(&format!("\"seed\":{expected}")),
            "the round {round} body must carry the seed its own row records ({expected}); \
             got: {body}"
        );
        match per_round.iter().find(|(seen, _)| seen == round) {
            Some((_, first)) => assert_eq!(
                first, &expected,
                "round {round} sent two different seeds — a per-CALL offset where §14.23 \
                 sanctions only a per-round one"
            ),
            None => per_round.push((*round, expected)),
        }
    }
    // One round cannot tell a per-round offset from a fixed seed, so the guard has
    // to have seen more than one or it asserts nothing.
    assert!(
        per_round.len() > 1,
        "this run must span more than one round to mean anything; saw rounds {:?}",
        per_round.iter().map(|(round, _)| *round).collect::<Vec<_>>()
    );

    let summary = run_summary(workspace.path()).summary;
    for flag in ["corroborate=off", "grounded_audit=off", "batch_audit=off"] {
        assert!(
            summary.contains(flag),
            "this guard is about the flag-off configuration; got {summary}"
        );
    }
    for key in [
        "truncated_calls_creator=0",
        "truncated_calls_attacker=0",
        "truncated_calls_intent_audit=0",
    ] {
        assert!(
            summary.contains(key),
            "no call was cut off, so `{key}` must be recorded as zero; got {summary}"
        );
    }
}

/// The repo's OWN test suite is a gate the generated tests cannot substitute for.
///
/// [VANTA-WHY] Generated tests only cover what the Attacker sampled. A patch can
/// break a function no generated test touches and still win every round — on a
/// real repository that is a bigger hole than any single missed boundary value.
/// The gate runs the repo's own suite on the ORIGINAL and on the PATCHED tree: a
/// test that passes on the original and fails on the patch is a regression, and
/// the patch is rejected. It is recorded as its own `native_suite` section — not
/// part of the shared set, never counted in EIR/ECR.
#[test]
fn a_native_suite_regression_rejects_the_patch_even_when_generated_tests_pass() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // The generated probe only asserts on `add`; the patch only breaks `scale`.
    // So the ONLY thing that can reject this patch is the repository's own test.
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        NATIVE_BREAKING_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let status = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !status.success(),
        "a patch that breaks the repository's own test must be rejected"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("regressions=1"),
        "the repo's own suite must report the one broken test; got {}",
        summary.summary
    );
    assert!(
        records.iter().any(|r| r.phase.starts_with("native_suite.test[")
            && r.summary.contains("regression=true")),
        "the broken repo test must be recorded as a regression with its own row; got {:?}",
        records
            .iter()
            .filter(|r| r.phase.starts_with("native_suite.test["))
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
    );

    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("Rejected"),
        "a native-suite regression is a rejection, not a budget outcome; got {}",
        final_row.summary
    );
}

/// Regression guard for the timeout signal path.
///
/// [VANTA-WHY] A test killed by the backend timeout has to be recognised AS a
/// timeout: the flag must travel from the execution result into the
/// classification (`ExecutionResult.timed_out` →
/// `TestExecutionDetails.timed_out` → Infrastructure) and trigger the Spec §7
/// Phase-2 remediation — reset the sandbox and rerun once. For a while M3's
/// Diagnostician never received the flag, so a hung test fell through to
/// Unknown, was pushed down the Unknown → simplify → quarantine chain, and the
/// reset-and-rerun never ran. An Unknown asks the Attacker to simplify the probe,
/// which is a second LLM call — the observable that makes the regression loud.
///
/// The probe hangs ONLY on the patched tree, so BATE admits it as a
/// BehaviourPreservation test and it is the PATCHED run that times out.
#[test]
fn a_timed_out_test_is_infrastructure_and_is_rerun_once_in_a_fresh_sandbox() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo(workspace.path(), HANGING_PATCH);
    let server = StubLlmServer::start(HANGING_PROBE);

    // A short cap so the hang is killed promptly instead of after the 120s
    // default. This cap applies only to the PATCHED-side execution; BATE's
    // baseline stays on its normal timeout so machine load cannot move the
    // failure earlier in the workflow.
    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_PATCHED_TEST_TIMEOUT_SECS", "5")],
    );
    let status = output.status;
    assert!(
        !status.success(),
        "a patch whose test hangs must not be approved"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("patched-side test timeout differs from baseline timeout"),
        "a run that uses the test-only timeout seam must warn about the mismatch; got {stderr:?}"
    );

    // Exactly two requests: the Attacker generation and the patch-blind intent
    // audit of the probe. Classifying the timeout as Unknown would have called
    // the Attacker a THIRD time to simplify the probe.
    assert_eq!(
        server.request_kinds(),
        vec!["generate", "intent_audit"],
        "a timed-out test must classify as Infrastructure, not fall through to \
         Unknown — the Unknown chain asks the Attacker to simplify the probe, \
         which would show up as a further LLM call"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let diagnostic =
        read_json(&debug_dir.join("attempt_1_diagnostic_test_hangs_on_patched_tree.json"));

    assert!(
        diagnostic["failure_class"]
            .as_str()
            .unwrap_or_default()
            .contains("Infrastructure"),
        "a killed test must be classified Infrastructure; got {}",
        diagnostic["failure_class"]
    );
    let reason = diagnostic["reason"].as_str().unwrap_or_default();
    assert!(
        reason.contains("after sandbox reset + rerun"),
        "Infrastructure must run the Phase-2 remediation — reset the sandbox and \
         rerun once — and the diagnostic must record that it did; got {reason:?}"
    );
    let payload = diagnostic["structured_payload"]
        .as_object()
        .expect("Infrastructure rerun must write a structured payload");
    assert_eq!(
        payload.get("fresh_sandbox_spawned"),
        Some(&serde_json::Value::Bool(true)),
        "the rerun must happen in a freshly spawned sandbox; got {payload:?}"
    );
    assert_ne!(
        payload
            .get("initial_sandbox_root")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        payload
            .get("rerun_sandbox_root")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "the rerun must use a different sandbox root than the original failing run; got {payload:?}"
    );

    let snap: EcrEirSnapshot = ecr_eir_from_rows(&[metric_row_from_observed_outcome(
        TestResult::Error,
        FailureClass::Infrastructure,
    )]);
    assert_eq!(snap.ecr_count, 0);
    assert_eq!(snap.eir_count, 0);
    assert_eq!(
        snap.complete, 0,
        "an Infrastructure timeout must be excluded from the metric denominator too"
    );
    assert_eq!(
        snap.excluded_non_logic, 1,
        "the exclusion must be visible so the run reads as evidence-starved, not as a silent zero"
    );
}

#[test]
fn an_unknown_failure_is_rerun_then_simplified_then_quarantined() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_TEST_FORCE_UNKNOWN", "1")],
    );
    assert!(
        !output.status.success(),
        "a quarantined Unknown must not approve the patch"
    );
    assert_eq!(
        server.request_kinds(),
        vec!["generate", "intent_audit", "simplify"],
        "the Unknown path must rerun locally and ask the Attacker to simplify once, \
         after the patch-blind audit of the generated probe"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let diagnostic =
        read_json(&debug_dir.join("attempt_1_diagnostic_test_behavior_preservation_only.json"));
    assert!(
        diagnostic["failure_class"]
            .as_str()
            .unwrap_or_default()
            .contains("Unknown"),
        "the forced-Unknown path must stay Unknown through quarantine; got {}",
        diagnostic["failure_class"]
    );
    assert!(
        diagnostic["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("QUARANTINED"),
        "persistent Unknowns must quarantine, not count as a model bug; got {:?}",
        diagnostic["reason"]
    );
    let payload = diagnostic["structured_payload"]
        .as_object()
        .expect("quarantined Unknowns must write a structured payload");
    assert_eq!(
        payload.get("action"),
        Some(&serde_json::Value::String("quarantine".into()))
    );
}

/// [VANTA-WHY] A10 (§14.10) — the Unknown path read ANY passing simplified test
/// as proof that the Unknown was resolved, and nothing checked that the
/// simplification still asserted anything. A body reduced to `add(2, 3)` keeps
/// the same inputs, drops the assertion, and therefore PASSES on a patched tree
/// the original probe failed on — so a real failure was erased by a test that
/// had stopped testing. The pass may retire the failure only when the same input
/// and expected value are still asserted; otherwise the simplified test is
/// SCRAPPED and the Unknown stands.
#[test]
fn a_simplified_test_that_dropped_its_assertion_does_not_erase_the_failure() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // The generated probe asserts `add(2, 3) == 5`, which passes on the original
    // tree and fails on this patch (`a - b`). The simplification keeps the call
    // and the inputs but drops the assertion, so it passes on the patched tree.
    const SIMPLIFIED_WITHOUT_ASSERTION: &str = r#"{
        "test_name":"test_behavior_preservation_only",
        "test_source":"from calc import add\n\ndef test_behavior_preservation_only():\n    add(2, 3)\n",
        "intent":"BehaviorPreservation",
        "reason":"simplified: reduced the body to the call under test",
        "issue_quote":null
    }"#;
    let server =
        StubLlmServer::start_with_simplify(SINGLE_BEHAVIOR_TEST, SIMPLIFIED_WITHOUT_ASSERTION);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_TEST_FORCE_UNKNOWN", "1")],
    );
    assert!(
        !output.status.success(),
        "a failure that was not erased must not approve the patch"
    );
    assert_eq!(
        server.request_kinds(),
        vec!["generate", "intent_audit", "simplify"],
        "the Unknown path still asks for exactly one simplification"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let diagnostic =
        read_json(&debug_dir.join("attempt_1_diagnostic_test_behavior_preservation_only.json"));
    assert!(
        diagnostic["failure_class"]
            .as_str()
            .unwrap_or_default()
            .contains("Unknown"),
        "a simplification that no longer asserts anything must not resolve the Unknown; got {}",
        diagnostic["failure_class"]
    );
    let reason = diagnostic["reason"].as_str().unwrap_or_default();
    assert!(
        reason.contains("QUARANTINED"),
        "the failure must still be quarantined, not erased; got {reason:?}"
    );
    assert!(
        reason.contains("dropped the assertion"),
        "the record must name WHY the passing simplification was refused; got {reason:?}"
    );
    let payload = diagnostic["structured_payload"]
        .as_object()
        .expect("quarantined Unknowns must write a structured payload");
    assert_eq!(
        payload.get("action"),
        Some(&serde_json::Value::String("quarantine".into()))
    );
}

#[test]
fn a_network_failure_is_classified_infrastructure_not_logic() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_TEST_FORCE_NETWORK_DOWN", "1")],
    );
    assert!(
        !output.status.success(),
        "a network-down run must not approve the patch"
    );
    assert_eq!(
        server.request_kinds(),
        vec!["generate", "intent_audit"],
        "network-down classification must not fall through to Unknown/simplify"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let diagnostic =
        read_json(&debug_dir.join("attempt_1_diagnostic_test_behavior_preservation_only.json"));
    assert!(
        diagnostic["failure_class"]
            .as_str()
            .unwrap_or_default()
            .contains("Infrastructure"),
        "network failure must classify as Infrastructure; got {}",
        diagnostic["failure_class"]
    );
    assert_eq!(
        diagnostic["structured_payload"]["signal"],
        serde_json::Value::String("network_down".into())
    );
}

#[test]
fn structured_diagnostic_payload_reaches_creator_in_repair_mode() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start_with_creator_and_attacker(
        CREATOR_SYNTAX_ERROR_RESPONSE,
        SINGLE_BEHAVIOR_TEST,
    );

    let output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);
    assert!(
        !output.status.success(),
        "the stub creator keeps returning a bad patch, so the run should exhaust its attempts"
    );

    let creator_bodies: Vec<String> = server
        .request_kinds()
        .into_iter()
        .zip(server.request_bodies())
        .filter_map(
            |(kind, body)| {
                if kind == "creator" { Some(body) } else { None }
            },
        )
        .collect();
    assert_eq!(
        creator_bodies.len(),
        2,
        "two repair attempts must produce two Creator calls"
    );
    let second_creator_body = &creator_bodies[1];
    let second_creator_body_lower = second_creator_body.to_lowercase();
    assert!(
        second_creator_body.contains("structured_payload=")
            && second_creator_body.contains("new_static_errors"),
        "repair-mode feedback must include the structured diagnostic payload for the Creator; got {second_creator_body}"
    );
    assert!(
        second_creator_body.contains("failure_class=Static"),
        "repair-mode feedback must still include the class label alongside the payload; got {second_creator_body}"
    );
    assert!(
        second_creator_body.contains("test_behavior_preservation_only"),
        "repair-mode feedback must name the failing test so the Creator sees which behaviour broke; got {second_creator_body}"
    );
    assert!(
        second_creator_body.contains("ruff_stderr")
            || second_creator_body.contains("mypy_stderr"),
        "repair-mode feedback must include the raw static-analysis excerpt field; got {second_creator_body}"
    );
    assert!(
        second_creator_body_lower.contains("invalid syntax")
            || second_creator_body.contains("Expected an expression")
            || second_creator_body.contains("return a + b +"),
        "repair-mode feedback must carry a concrete error excerpt, not just a class label; got {second_creator_body}"
    );
}

/// A Creator answer that is NOT a diff: a bare source blob whose body cannot
/// parse.
///
/// [VANTA-WHY] The cascade's full-source strategy picks it up, the syntax guard
/// refuses the result, and the apply is reported `InvalidPython` — the CONTENT
/// failure. A diff no strategy can place is the FORMAT failure. Both must reach
/// the Creator as feedback, and the two messages differ on purpose.
const BARE_BROKEN_SOURCE: &str = "def add(a, b):\n    return a + b +\n";

/// A Creator patch that does not apply must NOT end a repair-mode run.
///
/// [VANTA-WHY] The repair loop's whole premise is that the Creator gets one bad
/// answer and fixes it from the feedback. Ending the run on the first
/// unappliable diff makes every malformed answer cost a whole benchmark
/// instance instead of one attempt, which drags the repair rate down for a
/// reason that has nothing to do with the model's ability to repair. Only review
/// mode stops — its candidate was written outside the run, so there is no
/// Creator to tell.
#[test]
fn a_creator_patch_that_does_not_apply_is_fed_back_to_the_creator_not_fatal() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let script = serde_json::json!([
        {
            "patch_text": BARE_BROKEN_SOURCE,
            "explanation": "emit source that cannot parse",
            "files_touched": ["calc.py"],
            "confidence": 0.1
        },
        {
            "patch_text": WRONG_BEHAVIOUR_PATCH,
            "explanation": "correct the arithmetic",
            "files_touched": ["calc.py"],
            "confidence": 0.8
        }
    ])
    .to_string();
    let server = StubLlmServer::start_with_creator_script(&script, SINGLE_BEHAVIOR_TEST);

    let _output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);

    let creator_bodies: Vec<String> = server
        .request_kinds()
        .into_iter()
        .zip(server.request_bodies())
        .filter_map(|(kind, body)| if kind == "creator" { Some(body) } else { None })
        .collect();
    assert_eq!(
        creator_bodies.len(),
        2,
        "a failed apply must spend the attempt and go BACK to the Creator rather than \
         stop the run; got {:?}",
        server.request_kinds()
    );
    let second_creator_body = &creator_bodies[1];
    assert!(
        second_creator_body.contains("patch result is not valid Python")
            || second_creator_body.contains("patch did not apply"),
        "the second Creator call must carry the apply failure as feedback; got {second_creator_body}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let failed_applies: Vec<&str> = records
        .iter()
        .filter(|r| r.phase == "patch.apply_failed")
        .map(|r| r.summary.as_str())
        .collect();
    assert_eq!(
        failed_applies.len(),
        1,
        "the failed apply must be recorded exactly once in the session log; got {failed_applies:?}"
    );
    assert!(
        !records
            .iter()
            .any(|r| r.phase == "M3_repair_loop" && r.summary.contains("PatchApplicationFailed")),
        "repair mode must not end at PatchApplicationFailed; got {:?}",
        records
            .iter()
            .filter(|r| r.phase == "M3_repair_loop")
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
    );

    let evidence = read_json(
        &workspace
            .path()
            .join("vanta_sessions")
            .join(format!("{}_evidence.json", records[0].session_id.as_str())),
    );
    assert_eq!(
        evidence["apply_failed_attempts"].as_u64(),
        Some(1),
        "a failed apply is counted per run so the analysis can report how many holes \
         were Vanta's own apply path; got {evidence}"
    );

    // The failed attempt never produced a tree, so it may be recorded as a HOLE
    // but never as an observed transition. A first attempt has nothing to be
    // compared against, so here the honest record is "no row at all".
    let transition_path = find_transitions(workspace.path());
    let rows = read_transition_rows(&transition_path);
    assert!(
        rows.iter().all(|row| !row.apply_failed),
        "an attempt that never ran may not be stamped onto an observed row; got {rows:?}"
    );
}

/// The repository's own suite with FOUR tests, two of which import `scale`.
const NATIVE_REPO_TESTS_FOUR: &str = "from calc import add, scale\n\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n\n\ndef test_add_pair():\n    assert add(1, 2) == 3\n\n\ndef test_scale_pair():\n    assert scale(2, 3) == 6\n\n\ndef test_scale_other():\n    assert scale(3, 4) == 12\n";

/// Removes `scale` and nothing else.
///
/// [VANTA-WHY] `add` is untouched, so every generated test still passes: the
/// ONLY signal left is the repo's own suite, which now cannot even import. pytest
/// stops at collection, so the patched side reports 0 tests.
const DELETE_SCALE_PATCH: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,6 +1,2 @@\n def add(a, b):\n     return a + b\n-\n-\n-def scale(x, factor):\n-    return x * factor\n";

/// 4 tests → 0 tests must read as a REJECTION, not as a clean bill of health.
///
/// [VANTA-WHY] This is the heretic run's scariest false-clean: a patch that stops
/// pytest at collection collects ZERO tests, and a regression walk over the
/// PATCHED side finds nothing wrong in an empty list — so "4 tests → 0 tests"
/// printed "0 regression(s)". The walk is over the ORIGINAL side now and "gone"
/// is a regression, so the collapsed suite rejects. The generated probe passes
/// here on purpose: the repo's own suite is the only thing that can catch this.
#[test]
fn a_native_suite_that_collapses_from_four_tests_to_zero_rejects_the_patch() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        DELETE_SCALE_PATCH,
        Some(NATIVE_REPO_TESTS_FOUR),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a patch that stops the repo's own suite from collecting must be rejected"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("original_tests=4") && summary.summary.contains("patched_tests=0"),
        "the native suite must be reported as 4 → 0, not as a clean run; got {}",
        summary.summary
    );
    assert!(
        summary.summary.contains("regressions=4"),
        "all four vanished tests are regressions, not zero; got {}",
        summary.summary
    );

    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("Rejected"),
        "a collapsed native suite is a rejection, not a budget outcome; got {}",
        final_row.summary
    );
}

// ============================================================================
// Test integrity — the native suite must grade the patch with the ORIGINAL tests
// ============================================================================
//
// [VANTA-WHY] Session 888b6931 approved a patch that always passed
// `max_response_length` to `get_responses`, even when the setting was unset —
// exactly the behaviour the issue forbade. It got through because the Creator
// ALSO edited the repository's own test (`FakeContext.get_responses`) to accept
// the new kwarg, and the native suite ran the Creator's edited file. Run on the
// original test file that test fails with a TypeError. A patch must never
// control the tests that grade it, so the native run restores every original
// test file, drops patch-added ones, restores deleted ones, and reports the
// pytest-config change instead of trusting it.

/// A patch that breaks `scale` AND rewrites the repository's own test to accept
/// the broken arithmetic — the 888b6931 shape. On the ORIGINAL test file the
/// patched source fails (`scale(3, 2)` is 7, the test wants 6).
const WEAKEN_THE_CATCHING_TEST_PATCH: &str = concat!(
    "diff --git a/calc.py b/calc.py\n",
    "--- a/calc.py\n",
    "+++ b/calc.py\n",
    "@@ -5,2 +5,2 @@\n",
    " def scale(x, factor):\n",
    "-    return x * factor\n",
    "+    return x * factor + 1\n",
    "diff --git a/tests/test_core.py b/tests/test_core.py\n",
    "--- a/tests/test_core.py\n",
    "+++ b/tests/test_core.py\n",
    "@@ -1,5 +1,5 @@\n",
    " from calc import scale\n",
    " \n",
    " \n",
    " def test_scale_doubles():\n",
    "-    assert scale(3, 2) == 6\n",
    "+    assert scale(3, 2) == 7\n",
);

/// The same break, plus a root `conftest.py` that SKIPS the catching test — the
/// cheapest way for a patch to neutralise the suite that grades it.
const BREAK_AND_NEUTRALISE_WITH_CONFTEST_PATCH: &str = concat!(
    "diff --git a/calc.py b/calc.py\n",
    "--- a/calc.py\n",
    "+++ b/calc.py\n",
    "@@ -5,2 +5,2 @@\n",
    " def scale(x, factor):\n",
    "-    return x * factor\n",
    "+    return x * factor + 1\n",
    "diff --git a/conftest.py b/conftest.py\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/conftest.py\n",
    "@@ -0,0 +1,7 @@\n",
    "+import pytest\n",
    "+\n",
    "+\n",
    "+def pytest_collection_modifyitems(config, items):\n",
    "+    for item in items:\n",
    "+        if \"test_scale_doubles\" in item.nodeid:\n",
    "+            item.add_marker(pytest.mark.skip(reason=\"neutralised by the patch\"))\n",
);

/// The same break, plus a DELETION of the catching test file.
const BREAK_AND_DELETE_THE_TEST_PATCH: &str = concat!(
    "diff --git a/calc.py b/calc.py\n",
    "--- a/calc.py\n",
    "+++ b/calc.py\n",
    "@@ -5,2 +5,2 @@\n",
    " def scale(x, factor):\n",
    "-    return x * factor\n",
    "+    return x * factor + 1\n",
    "diff --git a/tests/test_core.py b/tests/test_core.py\n",
    "deleted file mode 100644\n",
    "--- a/tests/test_core.py\n",
    "+++ /dev/null\n",
    "@@ -1,5 +0,0 @@\n",
    "-from calc import scale\n",
    "-\n",
    "-\n",
    "-def test_scale_doubles():\n",
    "-    assert scale(3, 2) == 6\n",
);

/// A patch that adds ONLY a new test file and touches no source.
const ADD_ONLY_A_NEW_TEST_FILE_PATCH: &str = concat!(
    "diff --git a/tests/test_new_probe.py b/tests/test_new_probe.py\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/tests/test_new_probe.py\n",
    "@@ -0,0 +1,5 @@\n",
    "+from calc import scale\n",
    "+\n",
    "+\n",
    "+def test_scale_new_probe():\n",
    "+    assert scale(3, 2) == 6\n",
);

/// The repo scanner only picks up source extensions, so a pytest-config file is
/// never in the original file map; the patch therefore ADDS the config section.
/// What matters is that the change is detected from the diff and recorded.
const EDIT_PYTEST_CONFIG_PATCH: &str = concat!(
    "diff --git a/pyproject.toml b/pyproject.toml\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/pyproject.toml\n",
    "@@ -0,0 +1,2 @@\n",
    "+[tool.pytest.ini_options]\n",
    "+markers = [\"slow: marks slow tests\", \"flaky: marks flaky tests\"]\n",
);

/// `WRONG_BEHAVIOUR_PATCH` plus a root `conftest.py` that REWRITES every test
/// outcome to `passed`.
///
/// [VANTA-WHY] The strongest form of "let the candidate grade itself": it does
/// not deselect the probe, it makes a failing probe REPORT as passing. Measured
/// against pytest 9, a generated probe that fails on the broken tree reports
/// `passed` while this conftest is present, and `failed` once Vanta's own root
/// `conftest.py` is written over it. It is the vector the generated-test run had
/// and the repo's own suite did not.
const BREAK_ADD_AND_FORCE_PASS_WITH_CONFTEST_PATCH: &str = concat!(
    "diff --git a/calc.py b/calc.py\n",
    "--- a/calc.py\n",
    "+++ b/calc.py\n",
    "@@ -1,5 +1,5 @@\n",
    " def add(a, b):\n",
    "-    return a + b\n",
    "+    return a - b\n",
    " \n",
    " \n",
    " def scale(x, factor):\n",
    "diff --git a/conftest.py b/conftest.py\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/conftest.py\n",
    "@@ -0,0 +1,8 @@\n",
    "+import pytest\n",
    "+\n",
    "+\n",
    "+@pytest.hookimpl(hookwrapper=True)\n",
    "+def pytest_runtest_makereport(item, call):\n",
    "+    outcome = yield\n",
    "+    report = outcome.get_result()\n",
    "+    report.outcome = \"passed\"\n",
);

/// A patch that touches NO source and adds only a pytest config that DESELECTS
/// the repo's own catching test.
///
/// [VANTA-WHY] `modified_test_config` is warn-only, so `addopts` is free for a
/// patch to rewrite; this one makes `-k` drop `test_scale_doubles` from the
/// patched run entirely. Measured against pytest 9: `1 deselected`, and the JSON
/// report carries no entry for the test at all. Because the patch breaks nothing,
/// the vanished test is the ONLY signal — a walk that did not treat absence as a
/// regression would approve this candidate with zero findings.
const DESELECT_THE_CATCHING_TEST_WITH_ADDOPS_PATCH: &str = concat!(
    "diff --git a/pyproject.toml b/pyproject.toml\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/pyproject.toml\n",
    "@@ -0,0 +1,2 @@\n",
    "+[tool.pytest.ini_options]\n",
    "+addopts = \"-k 'not test_scale_doubles'\"\n",
);

/// A patch that touches NO source and SILENCES the repo's own test through a
/// pytest plugin loaded from `addopts`.
///
/// [VANTA-WHY] This is the `skipped` outcome, not deselection: the test is still
/// collected and executed, and reports `skipped`. `native_outcome` maps only
/// passed/failed/error, so `skipped` becomes `None` and the test drops out of the
/// patched report — a different route into the same `None` branch. Measured
/// against pytest 9: the JSON report carries
/// `("tests/test_core.py::test_scale_doubles", "skipped")`.
const SILENCE_THE_CATCHING_TEST_WITH_A_SKIP_PLUGIN_PATCH: &str = concat!(
    "diff --git a/pyproject.toml b/pyproject.toml\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/pyproject.toml\n",
    "@@ -0,0 +1,2 @@\n",
    "+[tool.pytest.ini_options]\n",
    "+addopts = \"-p skip_plugin\"\n",
    "diff --git a/skip_plugin.py b/skip_plugin.py\n",
    "new file mode 100644\n",
    "--- /dev/null\n",
    "+++ b/skip_plugin.py\n",
    "@@ -0,0 +1,7 @@\n",
    "+import pytest\n",
    "+\n",
    "+\n",
    "+def pytest_collection_modifyitems(config, items):\n",
    "+    for item in items:\n",
    "+        if \"test_scale_doubles\" in item.nodeid:\n",
    "+            item.add_marker(pytest.mark.skip(reason=\"neutralised by the patch\"))\n",
);

/// The evidence file path for the single session a toy run writes.
fn evidence_for(workspace: &Path, records: &[SessionRecord]) -> serde_json::Value {
    read_json(
        &workspace
            .join("vanta_sessions")
            .join(format!("{}_evidence.json", records[0].session_id.as_str())),
    )
}

/// A patch that breaks the behaviour AND rewrites the very test that catches it
/// must still be graded by the ORIGINAL test.
///
/// [VANTA-WHY] This is 888b6931 exactly. The Creator's edit made the test pass
/// on its own broken tree, the native suite ran the edited file, and a bad patch
/// was approved. The native run must restore the original test file, find the
/// regression, reject, and say in the rejection that the patch itself modified
/// the test the rejection rests on.
#[test]
fn a_patch_that_weakens_the_test_it_breaks_is_graded_by_the_original_test() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        WEAKEN_THE_CATCHING_TEST_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a patch that rewrote the test it breaks must still be rejected"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("regressions=1") && summary.summary.contains("patched_tests=1"),
        "the native run must grade the patch with the ORIGINAL test — one test collected, one \
         regression; got {}",
        summary.summary
    );
    assert!(
        records.iter().any(|r| r.phase
            == "native_suite.test[tests/test_core.py::test_scale_doubles]"
            && r.summary.contains("regression=true")),
        "the rewritten test must be recorded as a regression under its original name; got {:?}",
        records
            .iter()
            .filter(|r| r.phase.starts_with("native_suite.test["))
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Test integrity: native_test_source=original")
            && stdout.contains("modified_existing_tests=[tests/test_core.py]"),
        "the run output must report that the native suite ran the original tests and which \
         existing test the patch rewrote; got {stdout}"
    );
    assert!(
        stdout.contains("REJECTED") && stdout.contains("the patch also modified tests/test_core.py"),
        "the review-mode rejection must name the regressing test and say the patch itself \
         modified it; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(evidence["native_test_source"], serde_json::json!("original"));
    assert_eq!(
        evidence["modified_existing_tests"],
        serde_json::json!(["tests/test_core.py"])
    );
    assert_eq!(evidence["removed_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["added_test_files"], serde_json::json!([]));
    assert_eq!(evidence["modified_test_config"], serde_json::json!([]));
}

/// A root `conftest.py` that skips the catching test must not reach the native
/// run: the added conftest is excluded, so the original test still executes and
/// the regression is still found.
///
/// [VANTA-WHY] `patched_tests=1` is the assertion that matters. Had the added
/// conftest been collected, the test would have been SKIPPED — pytest reports
/// `skipped` and the native parser drops it, so the patched side would have
/// collected zero tests and the regression would have been found for the wrong
/// reason (absence, not failure). One collected test that FAILS is the original
/// test doing the grading.
#[test]
fn a_patch_added_conftest_cannot_skip_the_regressing_test() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        BREAK_AND_NEUTRALISE_WITH_CONFTEST_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a conftest that skips the catching test must not save the patch"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("patched_tests=1") && summary.summary.contains("regressions=1"),
        "the added conftest must be excluded, so the original test still runs and fails; got {}",
        summary.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("added_test_files=[conftest.py]"),
        "the added conftest must be reported as info, not silently dropped; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(
        evidence["added_test_files"],
        serde_json::json!(["conftest.py"])
    );
    assert_eq!(evidence["modified_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["removed_existing_tests"], serde_json::json!([]));
}

/// Deleting the catching test file must not silence the regression: the file is
/// read off the diff (`+++ /dev/null` is invisible in the resulting file map)
/// and restored for the native run.
#[test]
fn a_patch_that_deletes_the_test_file_has_it_restored_for_the_native_run() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        BREAK_AND_DELETE_THE_TEST_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "deleting the catching test must not turn a broken patch into an approval"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("patched_tests=1") && summary.summary.contains("regressions=1"),
        "the deleted test must be restored, so it runs on the patched side and fails; got {}",
        summary.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("removed_existing_tests=[tests/test_core.py]"),
        "the run output must report the restored test file; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(
        evidence["removed_existing_tests"],
        serde_json::json!(["tests/test_core.py"])
    );
    assert_eq!(evidence["modified_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["added_test_files"], serde_json::json!([]));
}

/// A patch that only ADDS a new test file changes neither the source nor the
/// baseline: it is reported as info and never counts toward approval.
#[test]
fn a_patch_added_test_file_is_information_not_baseline() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        ADD_ONLY_A_NEW_TEST_FILE_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        output.status.success(),
        "adding a test file must not change the verdict; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("patched_tests=1") && summary.summary.contains("regressions=0"),
        "the new test file is not part of the native baseline, so the patched side still \
         collects only the repo's own test; got {}",
        summary.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("added_test_files=[tests/test_new_probe.py]"),
        "the added test file must be reported in the run output; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(
        evidence["added_test_files"],
        serde_json::json!(["tests/test_new_probe.py"])
    );
    assert_eq!(evidence["modified_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["removed_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["modified_test_config"], serde_json::json!([]));
}

/// A patch that carries a `[tool.pytest.ini_options]` change is recorded with
/// its diff instead of being trusted — the section is not restored (pyproject
/// also carries non-test content), so the change is surfaced for an audit.
#[test]
fn a_patch_that_edits_pytest_config_is_recorded_with_its_diff() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        EDIT_PYTEST_CONFIG_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("modified_test_config=[pyproject.toml]")
            && stdout.contains("pytest config 'pyproject.toml' changed by the patch"),
        "a pytest-config edit must be warned about and named in the run output; got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let evidence = evidence_for(workspace.path(), &records);
    let config = evidence["modified_test_config"]
        .as_array()
        .expect("modified_test_config must be an array");
    assert_eq!(
        config.len(),
        1,
        "exactly one pytest-config change must be recorded; got {config:?}"
    );
    assert_eq!(config[0]["path"], serde_json::json!("pyproject.toml"));
    let diff = config[0]["diff"].as_str().unwrap_or_default();
    assert!(
        diff.contains("tool.pytest.ini_options") && diff.contains("flaky: marks flaky tests"),
        "the recorded config change must carry its diff; got {diff:?}"
    );
    assert_eq!(evidence["modified_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["removed_existing_tests"], serde_json::json!([]));
    assert_eq!(evidence["added_test_files"], serde_json::json!([]));
}

/// In repair mode the Creator is TOLD that its edits to an existing test were
/// discarded, and given the native regression with its failure output — so it
/// does not "fix" the same test again on the next attempt.
#[test]
fn repair_mode_tells_the_creator_its_test_edits_were_discarded() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        WEAKEN_THE_CATCHING_TEST_PATCH,
        Some(NATIVE_REPO_TEST),
    );

    // The Creator's multi-file answer is the SAME bad patch every attempt: the
    // feedback is what is under test, not the Creator's ability to fix it.
    let creator_patch = concat!(
        "===FILE: calc.py===\n",
        "diff --git a/calc.py b/calc.py\n",
        "--- a/calc.py\n",
        "+++ b/calc.py\n",
        "@@ -5,2 +5,2 @@\n",
        " def scale(x, factor):\n",
        "-    return x * factor\n",
        "+    return x * factor + 1\n",
        "===FILE: tests/test_core.py===\n",
        "diff --git a/tests/test_core.py b/tests/test_core.py\n",
        "--- a/tests/test_core.py\n",
        "+++ b/tests/test_core.py\n",
        "@@ -1,5 +1,5 @@\n",
        " from calc import scale\n",
        " \n",
        " \n",
        " def test_scale_doubles():\n",
        "-    assert scale(3, 2) == 6\n",
        "+    assert scale(3, 2) == 7\n",
    );
    let script = serde_json::json!([
        {
            "patch_text": creator_patch,
            "explanation": "make scale add one and relax the repo's own test",
            "files_touched": ["calc.py", "tests/test_core.py"],
            "confidence": 0.3
        }
    ])
    .to_string();
    let server = StubLlmServer::start_with_creator_script(&script, SINGLE_BEHAVIOR_TEST);

    let output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);
    assert!(
        !output.status.success(),
        "the stub creator never repairs the patch, so the run must exhaust its budget"
    );

    let creator_bodies: Vec<String> = server
        .request_kinds()
        .into_iter()
        .zip(server.request_bodies())
        .filter_map(|(kind, body)| if kind == "creator" { Some(body) } else { None })
        .collect();
    assert_eq!(
        creator_bodies.len(),
        2,
        "a rejected attempt must be fed back to the Creator as a second call"
    );
    let feedback = &creator_bodies[1];
    assert!(
        feedback.contains("You modified or deleted existing test file(s). Those edits were \
                           DISCARDED")
            || feedback.contains("Those edits were DISCARDED"),
        "the Creator must be told its test edits were discarded; got {feedback}"
    );
    assert!(
        feedback.contains("modified (discarded): tests/test_core.py"),
        "the discarded file must be named; got {feedback}"
    );
    assert!(
        feedback.contains("tests/test_core.py::test_scale_doubles"),
        "the native regression must be named; got {feedback}"
    );
    assert!(
        feedback.contains("failure:"),
        "the native regression must carry its failure output; got {feedback}"
    );
    assert!(
        feedback.contains(
            "Do not modify or delete existing tests. If you need a new test, put it in a NEW \
             test file."
        ),
        "the Creator must be told the rule, not just the consequence; got {feedback}"
    );
}

// ============================================================================
// Gap 1 — the generated tests must be graded on the SAME sanitized tree
// ============================================================================
//
// [VANTA-WHY] The repo's own suite already ran the patch's SOURCE with the
// ORIGINAL tests. The generated tests did not: they ran on the patch's own tree,
// `conftest.py` and all, so a candidate could reach the run that judges it. A
// root `conftest.py` carrying a `pytest_runtest_makereport` hook that rewrites
// every outcome to `passed` is the sharpest version — it does not even need to
// know the probe's name. The patched-side run now uses the same graded tree the
// native suite uses, so the two can never disagree about which tests ran.

/// A patch-added root `conftest.py` that forces every outcome to `passed` must
/// NOT save the patch from the generated probe.
///
/// [VANTA-WHY] Measured against pytest 9: on the patch's own tree this conftest
/// makes the probe report `passed` even though `add(2, 3)` is `-1`. If the
/// generated tests ran there, this candidate would be approved with nothing
/// failing. The graded tree drops the added conftest, so the probe still fails
/// and the run is rejected on the generated test.
#[test]
fn a_patch_added_conftest_cannot_force_the_generated_test_to_pass() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        BREAK_ADD_AND_FORCE_PASS_WITH_CONFTEST_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a conftest that rewrites every outcome to 'passed' must not save a patch the generated \
         probe catches; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let result_row = records
        .iter()
        .find(|r| r.phase == "patched.result[test_behavior_preservation_only]")
        .expect("the generated test must have a patched-side result row");
    assert!(
        result_row.summary.contains("outcome=Counterexample"),
        "the probe must still be recorded as failing on the graded tree, not as verified on the \
         patch's own; got {}",
        result_row.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("added_test_files=[conftest.py]"),
        "the patch-added conftest must be surfaced as an added test file, never silently \
         honoured; got {stdout}"
    );
    assert!(
        stdout.contains("✖ REJECTED")
            && stdout.contains("failing test(s): test_behavior_preservation_only"),
        "the rejection must name the generated test that failed; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(
        evidence["added_test_files"],
        serde_json::json!(["conftest.py"])
    );
    assert_eq!(evidence["modified_existing_tests"], serde_json::json!([]));
}

// ============================================================================
// Gap 2 — a SILENCED native test is a regression, not a clean round
// ============================================================================
//
// [VANTA-WHY] `modified_test_config` is warn-only: a patch that rewrites
// `[tool.pytest.ini_options]` is reported, never itself rejected. So the only
// defence against a silenced test is that a silenced test is a MISSING test, and
// a missing test is a regression. `native_regressions` walks the ORIGINAL side
// and counts a test that no longer reports `passed` OR is absent — which is
// exactly how pass -> skipped / xfail / deselected is caught, since
// `native_outcome` drops anything that is not passed/failed/error. Both patches
// below touch no source at all, so the vanished test is the entire signal.

/// Deselecting the repo's own test through `addopts` is a regression.
#[test]
fn a_patch_that_deselects_a_native_test_through_addopts_is_a_regression() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        DESELECT_THE_CATCHING_TEST_WITH_ADDOPS_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a patch that deselects the repo's own test must be rejected; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("original_tests=1")
            && summary.summary.contains("patched_tests=0")
            && summary.summary.contains("regressions=1"),
        "the deselected test must read as gone, not as passing; got {}",
        summary.summary
    );

    let row = records
        .iter()
        .find(|r| r.phase == "native_suite.test[tests/test_core.py::test_scale_doubles]")
        .expect("the original test must be recorded even when the patch deselects it");
    assert!(
        row.summary.contains("original=Passed")
            && row.summary.contains("patched=None")
            && row.summary.contains("regression=true"),
        "pass -> deselected is recorded as `patched=None` with regression=true; got {}",
        row.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("modified_test_config=[pyproject.toml]"),
        "the pytest-config change must still be reported, even though it never rejects on its \
         own; got {stdout}"
    );
    assert!(
        stdout.contains("✖ REJECTED")
            && stdout.contains("native regression: tests/test_core.py::test_scale_doubles"),
        "the rejection must rest on the vanished native test; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    let regressions = evidence["native_regressions"]
        .as_array()
        .expect("evidence.json must record native regressions under a clear key");
    assert!(
        regressions.iter().any(|r| r["attempt"] == serde_json::json!(1)
            && r["regressions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str().unwrap_or("").contains("test_scale_doubles"))),
        "attempt 1's native regression must name the deselected test; got {regressions:?}"
    );
}

/// Making the repo's own test SKIP is a regression.
#[test]
fn a_patch_that_skips_a_native_test_is_a_regression() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        SILENCE_THE_CATCHING_TEST_WITH_A_SKIP_PLUGIN_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !output.status.success(),
        "a patch that skips the repo's own test must be rejected; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "native_suite.summary")
        .expect("the native-suite gate must log its own summary row");
    assert!(
        summary.summary.contains("original_tests=1")
            && summary.summary.contains("patched_tests=0")
            && summary.summary.contains("regressions=1"),
        "a skipped test is neither a pass nor a failure, so it must read as gone; got {}",
        summary.summary
    );

    let row = records
        .iter()
        .find(|r| r.phase == "native_suite.test[tests/test_core.py::test_scale_doubles]")
        .expect("the original test must be recorded even when the patch skips it");
    assert!(
        row.summary.contains("original=Passed")
            && row.summary.contains("patched=None")
            && row.summary.contains("regression=true"),
        "`skipped` is dropped by the native parser, so pass -> skipped is `patched=None` with \
         regression=true; got {}",
        row.summary
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("✖ REJECTED")
            && stdout.contains("native regression: tests/test_core.py::test_scale_doubles"),
        "the rejection must rest on the skipped native test; got {stdout}"
    );

    let evidence = evidence_for(workspace.path(), &records);
    assert_eq!(
        evidence["added_test_files"],
        serde_json::json!([]),
        "the silencing plugin is not a test file, so it must not be reported as an added test"
    );
    let regressions = evidence["native_regressions"]
        .as_array()
        .expect("evidence.json must record native regressions under a clear key");
    assert!(
        regressions.iter().any(|r| r["attempt"] == serde_json::json!(1)
            && r["regressions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str().unwrap_or("").contains("test_scale_doubles"))),
        "attempt 1's native regression must name the skipped test; got {regressions:?}"
    );
}

/// ECR/EIR pass — session 4d55123d, reproduced end to end.
///
/// [VANTA-WHY] In that run the ONE real correction was to a repository-native
/// test the previous attempt had broken, and it was invisible: the native suite
/// wrote only session records, so the transitions file held three
/// AttackerGenerated `Passed -> Passed` rows, ECR read 0/0, and the FINAL line
/// still claimed there had been no transition. A native regression is evidence
/// about the candidate, so it must produce a row like any other.
#[test]
fn a_native_regression_fixed_on_the_next_attempt_counts_as_a_correction() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        NATIVE_BREAKING_PATCH,
        Some(NATIVE_REPO_TEST),
    );

    // Attempt 1 breaks `scale`, which the repo's own test asserts and the
    // generated probe never touches. Attempt 2 stops touching `scale` (it only
    // adds a docstring to `add`), so the native test passes again.
    let script = serde_json::json!([
        {
            "patch_text": NATIVE_BREAKING_PATCH,
            "explanation": "break scale by adding one",
            "files_touched": ["calc.py"],
            "confidence": 0.3
        },
        {
            "patch_text": PRESERVING_PATCH,
            "explanation": "leave scale alone and document add",
            "files_touched": ["calc.py"],
            "confidence": 0.8
        }
    ])
    .to_string();
    let server = StubLlmServer::start_with_creator_script(&script, SINGLE_BEHAVIOR_TEST);

    let output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);
    assert!(
        output.status.success(),
        "the second attempt repairs the regression, so the run must be approved; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    // The native regression went through the SAME failure path to the Creator as
    // any generated failure, and it is counted there.
    assert!(
        stdout.contains(
            "Feeding 1 failure(s) back to Creator for retry (0 generated, 1 native, 0 static)"
        ),
        "the native regression must be counted as a failure fed back to the Creator; got {stdout}"
    );

    let rows = read_transition_rows(&find_transitions(workspace.path()));
    let native_rows: Vec<&TransitionRow> = rows
        .iter()
        .filter(|r| r.evidence_source == EvidenceSource::Native)
        .collect();
    assert_eq!(
        native_rows.len(),
        1,
        "one native test ran on both attempts, so exactly one Native row must exist; got {:?}",
        rows.iter()
            .map(|r| (&r.evidence_source, &r.from_result, &r.to_result))
            .collect::<Vec<_>>()
    );
    let native = native_rows[0];
    assert_eq!(
        native.shared_set_id, "native",
        "a native row is its own question set, not the Attacker's"
    );
    assert!(
        native.test_id.contains("test_scale_doubles"),
        "the row's test_id must be the native nodeid; got {}",
        native.test_id
    );
    assert_eq!(native.from_result, TestResult::Failed);
    assert_eq!(native.to_result, TestResult::Passed);

    let snap = ecr_eir_from_rows(&rows);
    assert_eq!(
        snap.native.ecr_count, 1,
        "ECR(Native) must count the correction the run actually made"
    );
    assert_eq!(
        snap.native.ecr_denominator, 1,
        "ECR is conditional: the one native test failing at N is its denominator"
    );
    assert_eq!(
        snap.native.eir_denominator, 0,
        "no native test was passing at N, so EIR(Native) has no denominator"
    );
    assert_eq!(
        snap.native.complete, 1,
        "the Native row count is the churn number, separate from the rate denominators"
    );
    assert_eq!(
        snap.ecr_count, 1,
        "the correction must appear in the combined ECR, not only per source"
    );
    assert_eq!(snap.ecr_denominator, 1);
    assert_eq!(
        snap.complete, 2,
        "the Attacker row (Passed->Passed) and the Native row are both countable"
    );

    // The FINAL line must not claim there was no transition: rows exist.
    assert!(
        !stdout.contains("no N→N+1 transition"),
        "the run wrote transition rows, so the FINAL line must not claim none; got {stdout}"
    );
    assert!(
        stdout.contains(
            "attempts_run=2 fixed_at_attempt_1=false — ECR=1/1 (100.00%) EIR=0/1 (0.00%) \
             over 2 transition row(s)"
        ),
        "combined ECR is conditional: the one row failing at N was fixed, so 1/1 — the \
         Attacker bystander belongs to EIR's denominator and must not dilute ECR; got {stdout}"
    );
    assert!(
        stdout.contains("Native[ECR=1/1 (100.00%) EIR=n/a]"),
        "the per-source line must attribute the correction to the Native suite, with \
         no invented EIR denominator; got {stdout}"
    );

    // Evidence file records each attempt's native regressions under a clear key.
    let records = read_session_records(&find_session_log(workspace.path()));
    let evidence = evidence_for(workspace.path(), &records);
    let regressions = evidence["native_regressions"]
        .as_array()
        .expect("evidence.json must record native regressions under a clear key");
    let attempt_one = regressions
        .iter()
        .find(|r| r["attempt"] == serde_json::json!(1))
        .expect("attempt 1 must have a native-regression entry");
    assert!(
        attempt_one["regressions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str().unwrap_or("").contains("test_scale_doubles")),
        "attempt 1's native regression must be recorded; got {attempt_one}"
    );
}

/// The mirror image: attempt 2 breaks a native test that PASSED at attempt 1, so
/// `EIR(Native)` must count the introduction.
#[test]
fn a_native_test_broken_on_the_next_attempt_counts_as_an_introduced_error() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        WRONG_BEHAVIOUR_PATCH,
        Some(NATIVE_REPO_TEST),
    );

    // Attempt 1 leaves `scale` alone (the native test passes) and gets `add`
    // wrong. Attempt 2 leaves `add` alone (the generated probe passes) but breaks
    // `scale`, which was passing the attempt before.
    let script = serde_json::json!([
        {
            "patch_text": WRONG_BEHAVIOUR_PATCH,
            "explanation": "break add instead of scale",
            "files_touched": ["calc.py"],
            "confidence": 0.4
        },
        {
            "patch_text": NATIVE_BREAKING_PATCH,
            "explanation": "break scale, which passed on the previous attempt",
            "files_touched": ["calc.py"],
            "confidence": 0.2
        }
    ])
    .to_string();
    let server = StubLlmServer::start_with_creator_script(&script, SINGLE_BEHAVIOR_TEST);

    let output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);
    assert!(
        !output.status.success(),
        "the Creator never reaches a clean patch, so the budget must be exhausted"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    let rows = read_transition_rows(&find_transitions(workspace.path()));
    let native_rows: Vec<&TransitionRow> = rows
        .iter()
        .filter(|r| r.evidence_source == EvidenceSource::Native)
        .collect();
    assert_eq!(
        native_rows.len(),
        1,
        "one native test ran on both attempts; got {:?}",
        rows.iter()
            .map(|r| (&r.evidence_source, &r.from_result, &r.to_result))
            .collect::<Vec<_>>()
    );
    assert_eq!(native_rows[0].from_result, TestResult::Passed);
    assert_eq!(native_rows[0].to_result, TestResult::Failed);

    let snap = ecr_eir_from_rows(&rows);
    assert_eq!(
        snap.native.eir_count, 1,
        "EIR(Native) must count the newly broken repo test"
    );
    assert_eq!(
        snap.native.eir_denominator, 1,
        "EIR is conditional: the one native test PASSING at N is its denominator"
    );
    assert_eq!(
        snap.native.ecr_denominator, 0,
        "nothing was failing at N under the Native source, so ECR(Native) is n/a"
    );
    assert_eq!(snap.native.complete, 1);
    assert_eq!(
        snap.eir_count, 1,
        "the break must appear in the combined EIR, not only per source"
    );
    assert!(
        stdout.contains("Native[ECR=n/a EIR=1/1 (100.00%)]"),
        "the per-source line must attribute the break to the Native suite, and ECR must \
         read n/a rather than 0.00%; got {stdout}"
    );
}

/// A run whose every transition is excluded (here: attempt 1 never applied its
/// patch, so the only transition is a censored hole) has a ZERO denominator, and
/// a zero denominator must print `n/a`, never `0.00%`.
///
/// [VANTA-WHY] "nothing countable" and "nothing corrected" are opposite claims,
/// and `0.00%` states the second. The FINAL line here also exercises the other
/// half of the fix: it may say "no N→N+1 transition" ONLY when there are no
/// transition rows at all — which is exactly this run.
#[test]
fn a_zero_denominator_prints_not_applicable_never_a_zero_percent() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let _ = write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);

    // Attempt 1 emits source that cannot parse, so nothing was observed for it
    // and its transition is a censored hole. Attempt 2 applies but fails the
    // generated probe — still nothing countable, because the `from` side never
    // produced an observation.
    let script = serde_json::json!([
        {
            "patch_text": BARE_BROKEN_SOURCE,
            "explanation": "emit source that cannot parse",
            "files_touched": ["calc.py"],
            "confidence": 0.1
        },
        {
            "patch_text": WRONG_BEHAVIOUR_PATCH,
            "explanation": "correct only the arithmetic, still wrong",
            "files_touched": ["calc.py"],
            "confidence": 0.8
        }
    ])
    .to_string();
    let server = StubLlmServer::start_with_creator_script(&script, SINGLE_BEHAVIOR_TEST);

    let output = run_creator_mode_with_output(workspace.path(), &server, 2, &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("ECR=n/a EIR=n/a"),
        "a zero denominator must render n/a on the combined metric; got {stdout}"
    );
    assert!(
        stdout.contains(
            "per source: AttackerGenerated[ECR=n/a EIR=n/a] Native[ECR=n/a EIR=n/a] Static[ECR=n/a EIR=n/a]"
        ),
        "every source with an empty denominator must render n/a; got {stdout}"
    );
    assert!(
        !stdout.contains("0.00%"),
        "a zero denominator must never be printed as a rate; got {stdout}"
    );
    assert!(
        stdout.contains("a repair with no N→N+1 transition is outside ECR/EIR by construction"),
        "with no transition rows at all, the FINAL line may say so; got {stdout}"
    );
}

// ============================================================================
// LLM call robustness: streaming liveness, cancel, InfraFailure
// ============================================================================
//
// [VANTA-WHY] The issue2 repair runs died because a healthy 600s generation was
// killed by a TOTAL deadline; llama-server kept generating the orphan, the retry
// opened a second slot against the same KV cache, and the run crashed with no
// final decision. These tests pin the three behaviours that replace that: a
// stalled stream is cancelled at the SOCKET (not merely abandoned), a prompt the
// server can never accept is refused once instead of three times, and an
// unrecoverable LLM failure ends the run as a clean `InfraFailure`.
//
// The "slow but steady stream completes past 600s of simulated time" case lives
// in `src/llm/mod.rs` (`_streaming_tests`): it needs a PAUSED clock, so 700
// simulated seconds cost no real time — impossible against a real subprocess.

/// A stalled stream fires the idle timeout, and the SERVER sees the disconnect.
#[test]
fn a_stalled_stream_is_cancelled_and_the_stub_sees_the_disconnect() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start_streaming(None, SINGLE_BEHAVIOR_TEST, StubBehaviour::Stall);

    // A one-second idle timeout makes the test quick while still exercising the
    // real watchdog. The 3600s backstop is far away, so the IDLE limit is what
    // must fire — and one second is well under 600s, so this proves the timeout
    // is driven by SILENCE, not by total elapsed time.
    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_LLM_IDLE_TIMEOUT_SECS", "1")],
    );

    assert_eq!(
        output.status.code(),
        Some(2),
        "an idle timeout must end the run with the distinct InfraFailure exit code; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        server.disconnects() >= 1,
        "the client must CLOSE the stalled connection so the server cancels the task; \
         the stub saw {} disconnect(s)",
        server.disconnects()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("INFRASTRUCTURE FAILURE"),
        "the run must say why it ended; got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let infra = records
        .iter()
        .find(|r| r.phase == "run.infra_failure")
        .expect("a run that gave up on the LLM must record why");
    assert!(
        infra.summary.contains("stalled"),
        "the recorded error must name the idle timeout; got {}",
        infra.summary
    );
}

/// [VANTA-WHY] §14.10 f — the run identity is recorded TWICE: the hand-bumped
/// version LABEL, and a content HASH of the system prompts the run sends. The
/// label was forgotten twice (the A3 follow-up `3588728` and the A10 follow-up
/// `d4ea314` reworded prompts after the `a2617b3` bump and left it alone), so the
/// hash is the field a later log comparison can trust. That is only true if the
/// recorded value really is the hash of the text sent, so this pins BOTH rows
/// that carry it against a recomputation from the accessors the call sites use —
/// never against the recorded string itself, which would pass however stale it
/// had become.
#[test]
fn the_recorded_prompt_hash_is_the_hash_of_the_prompts_the_run_sends() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start_streaming(None, SINGLE_BEHAVIOR_TEST, StubBehaviour::Normal);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 3, &[]);
    assert_ne!(
        output.status.code(),
        Some(2),
        "the run must reach a verdict before its identity can be read; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let expected = format!(
        "attacker={};intent_check={};creator={};diagnostician={}",
        vanta::core::prompt_content_hash(&vanta::agents::attacker::attacker_system_prompts()),
        vanta::core::prompt_content_hash(&vanta::agents::attacker::intent_auditor_system_prompts()),
        vanta::core::prompt_content_hash(&[vanta::agents::creator::creator_system_prompt()]),
        vanta::core::prompt_content_hash(&[]),
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let start = records
        .first()
        .expect("the first record on disk is session.start");
    assert_eq!(start.phase, "session.start");
    assert!(
        start.summary.contains(&format!("prompt_hash={expected}")),
        "session.start must record the hash of the prompts this run sends; got {}",
        start.summary
    );

    // The same value in the terminal row: a session's debug artifacts outlive the
    // shell that produced them, so the run's prompt identity has to be readable
    // from the summary alone.
    let summary = run_summary(workspace.path());
    assert!(
        summary.summary.contains(&format!("prompt_hash={expected}")),
        "run.summary must record the same prompt hash; got {}",
        summary.summary
    );
}

/// A prompt that cannot fit the window is refused ONCE, without retrying.
#[test]
fn the_context_precheck_fails_once_without_retrying() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start_streaming(None, SINGLE_BEHAVIOR_TEST, StubBehaviour::Normal);

    // A one-token window makes EVERY prompt exceed the budget. That is decided
    // by arithmetic before a byte leaves the process, so it cannot succeed on a
    // retry.
    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_LLM_CONTEXT_TOKENS", "1"), ("VANTA_LLM_RETRY_ATTEMPTS", "3")],
    );

    assert_eq!(
        output.status.code(),
        Some(2),
        "a refused prompt is an infrastructure failure, not a verdict; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let hits = combined
        .matches("does not fit the model's context window")
        .count();
    assert_eq!(
        hits, 1,
        "a deterministic pre-flight failure must be surfaced ONCE, not once per attempt; \
         got {hits} in:\n{combined}"
    );
    // Nothing was ever POSTed as a chat completion: the guard refuses first.
    assert!(
        !server
            .request_kinds()
            .iter()
            .any(|kind| kind == "generate" || kind == "creator"),
        "the pre-flight guard must refuse BEFORE sending a chat completion; kinds={:?}",
        server.request_kinds()
    );
}

/// An LLM failure that survives its retries ends the run as an `InfraFailure`.
#[test]
fn an_llm_failure_after_retries_writes_an_infrafailure_run_summary() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start_streaming(None, SINGLE_BEHAVIOR_TEST, StubBehaviour::Fail);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_LLM_RETRY_ATTEMPTS", "2")],
    );

    assert_eq!(
        output.status.code(),
        Some(2),
        "an unrecoverable LLM failure must exit with the distinct InfraFailure code; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        server.request_count() >= 2,
        "the call must actually have been retried before giving up; got {}",
        server.request_count()
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "run.summary")
        .expect("the run must write a summary row even when the LLM never answered");
    assert!(
        summary.summary.contains("final_decision=InfraFailure"),
        "the summary must carry the verdict; got {}",
        summary.summary
    );
    assert!(
        records.iter().any(|r| r.phase == "run.infra_failure"),
        "the error text must be recorded beside the verdict"
    );

    // Evidence and transitions must be flushed even on this path: a reader has
    // to be able to tell an infrastructure failure from a candidate verdict.
    let evidence = evidence_for(workspace.path(), &records);
    assert!(
        evidence.is_object(),
        "evidence.json must exist after an InfraFailure run"
    );
    assert!(find_transitions(workspace.path()).exists());
}

// ============================================================================
// BugsInPy harness — §11 step 5: env switches + per-call provenance
// ============================================================================
//
// The harness drives the SAME binary the existing suite exercises, so these
// tests are the guard that the three new env switches change nothing about the
// verdict machinery and that the provenance block records what really happened.

/// The stub's `request_kinds` labels (detected by body substring) mapped onto the
/// call-site names in `vanta::llm::SAMPLING_SCHEDULE`.
const SITE_FOR_KIND: &[(&str, &str)] = &[
    ("creator", "creator"),
    ("generate", "attacker_generate"),
    ("revise", "attacker_revise"),
    ("simplify", "attacker_simplify"),
    ("intent_audit", "intent_audit"),
];

/// The one `run.summary` record a terminal run must write.
fn run_summary(workspace: &Path) -> vanta::session::SessionRecord {
    read_session_records(&find_session_log(workspace))
        .into_iter()
        .find(|r| r.phase == "run.summary")
        .expect("every terminal run must write a run.summary row")
}

/// Read one `key=value` pair out of the free-form `run.summary` line.
fn summary_value(summary: &str, key: &str) -> String {
    let needle = format!("{key}=");
    let start = summary
        .find(&needle)
        .unwrap_or_else(|| panic!("run.summary has no `{key}`: {summary}"))
        + needle.len();
    let rest = &summary[start..];
    let end = rest
        .find(|c: char| c == ' ' || c == '\n' || c == '\r')
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

/// The request body as JSON with any `seed` key removed — a body from a seeded
/// run must be otherwise identical to the same body from an unseeded one.
fn body_without_seed(body: &str) -> serde_json::Value {
    let mut value: serde_json::Value = serde_json::from_str(body)
        .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
    if let Some(obj) = value.as_object_mut() {
        obj.remove("seed");
    }
    value
}

/// Assert that the `run.summary` schedule is the one the request bodies carried,
/// and return the call sites that were actually cross-checked.
///
/// [VANTA-WHY] §4.3: the harness PINS `temperature`/`max_tokens` per call site,
/// which is why one run samples at several temperatures and why a single
/// run-level pair would be a fabricated claim. The record is the SCHEDULE, so
/// every pinned site must appear even if this run never reached it; and wherever
/// a body WAS sent, its numbers must equal the recorded ones — otherwise the
/// provenance block describes a run that did not happen.
fn assert_schedule_matches_bodies(
    workspace: &Path,
    server: &StubLlmServer,
    preset: vanta::llm::SamplingPreset,
) -> Vec<String> {
    let summary = run_summary(workspace).summary;

    // [VANTA-WHY] §14.11: the record is the EFFECTIVE schedule for the active
    // preset, and `sampling_pins_text` is the SAME table `with_sampling` builds
    // the request bodies from — so a site that forgot a pin, or a preset that
    // stopped pinning one, surfaces here as a missing key rather than as a
    // silently different sampling.
    assert_eq!(
        summary_value(&summary, "sampling_preset"),
        preset.name(),
        "run.summary must name the active sampling preset (§14.11)"
    );
    for (site, ..) in vanta::llm::SAMPLING_SCHEDULE {
        for (key, value) in vanta::llm::sampling_pins_text(preset, site) {
            assert_eq!(
                summary_value(&summary, &format!("{key}_{site}")),
                value,
                "run.summary must record the pinned `{key}` for `{site}`"
            );
        }
    }

    let kinds = server.request_kinds();
    let bodies = server.request_bodies();
    assert_eq!(kinds.len(), bodies.len(), "one kind per recorded body");

    let mut checked: Vec<String> = Vec::new();
    for (kind, body) in kinds.iter().zip(bodies.iter()) {
        let site = SITE_FOR_KIND
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, site)| *site)
            .unwrap_or_else(|| panic!("stub saw an unmapped request kind `{kind}`"));
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        // [VANTA-WHY] Token accounting reads the server's own counts, and a
        // STREAMED response only carries a `usage` block when the request asks
        // for one — this key is the ask. Without it every recorded cost falls
        // back to the CHARACTER-derived estimate, which over-counts by ~1.5x and
        // cannot see a completion count at all, so a paper table built from it
        // would be wrong in a way nothing marked. Asserted on EVERY body: one
        // call site that forgot to ask is a hole in the run total.
        assert_eq!(
            value
                .get("stream_options")
                .and_then(|v| v.get("include_usage"))
                .and_then(|v| v.as_bool()),
            Some(true),
            "the `{kind}` body must ask for the usage block (`stream_options.include_usage`): {body}"
        );
        // Every pin the preset gives this site must be BOTH recorded and
        // actually carried by the body — the recorded schedule is the effective
        // one, so a missing body key is a call site that forgot `with_sampling`.
        for (key, expected) in vanta::llm::sampling_pins_text(preset, site) {
            let actual = value
                .get(key)
                .unwrap_or_else(|| panic!("the `{kind}` body carries no `{key}`: {body}"));
            let actual_text = if let Some(n) = actual.as_u64() {
                n.to_string()
            } else {
                actual.as_f64().map(|v| format!("{v:?}")).unwrap_or_else(|| {
                    panic!("the `{kind}` body's `{key}` is not a number: {body}")
                })
            };
            assert_eq!(
                actual_text, expected,
                "the `{kind}` body must carry the `{key}` the preset pins for `{site}`"
            );
        }
        checked.push(site.to_string());
    }
    assert!(
        !checked.is_empty(),
        "no chat body was sent, so the schedule was never cross-checked"
    );
    checked
}

/// `VANTA_LLM_SEED` must be invisible when unset and present in every chat body
/// when set, offset per review round.
///
/// [VANTA-WHY] An unset seed has to leave the request body byte-for-byte what it
/// was before the field existed: `"seed": null` would change what every server
/// sees, and `"seed": 0` would move the sampling baseline every existing
/// measurement was taken under. A test cannot run the pre-seed binary, so the
/// guard is twofold — no key at all when unset, exactly that key when set, and
/// the two bodies equal once the key is removed.
///
/// [VANTA-WHY] §14.23 — with ONE pinned seed every review round sampled
/// identically, so round 2 was a deterministic duplicate of round 1. The seed is
/// now offset by the round index, which makes this the test that has to see MORE
/// than one round: three rounds, so round 1 is the pinned seed unchanged (the
/// byte-identity guarantee) and rounds 2 and 3 are pinned + 1 and pinned + 2. The
/// `call.usage` rows are read rather than assumed, so each body is held to the
/// seed its own row records and the row's `seed_mechanism` has to agree.
#[test]
fn unset_seed_leaves_request_bodies_unseeded_and_the_verdict_unchanged() {
    let _serial = serial_run();

    // The `call.usage` rows, in call order, so they line up with the recorded
    // bodies one-for-one: (round, seed, seed_mechanism).
    let usage_rows = |workspace: &Path| -> Vec<(u32, String, String)> {
        read_session_records(&find_session_log(workspace))
            .iter()
            .filter(|r| r.phase == "call.usage")
            .map(|r| {
                (
                    summary_value(&r.summary, "round").parse().unwrap(),
                    summary_value(&r.summary, "seed"),
                    summary_value(&r.summary, "seed_mechanism"),
                )
            })
            .collect()
    };

    let plain = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(plain.path(), PRESERVING_PATCH);
    let plain_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let plain_output =
        run_patch_mode_with_output(plain.path(), &patch_path, &plain_server, 3, &[]);
    assert_eq!(
        plain_output.status.code(),
        Some(0),
        "the reference run must be the approval; stdout:\n{}",
        String::from_utf8_lossy(&plain_output.stdout)
    );
    let plain_bodies = plain_server.request_bodies();
    assert!(!plain_bodies.is_empty(), "the run must have called the LLM");
    for body in &plain_bodies {
        assert!(
            !body.contains("\"seed\""),
            "an unset VANTA_LLM_SEED must add NO seed key to the body; got: {body}"
        );
    }
    // No round may invent a seed, and the mechanism has to say none was sent.
    let plain_usage = usage_rows(plain.path());
    assert_eq!(
        plain_usage.len(),
        plain_bodies.len(),
        "one `call.usage` row per chat body"
    );
    for (round, seed, mechanism) in &plain_usage {
        assert_eq!(
            (seed.as_str(), mechanism.as_str()),
            ("unset", "unset"),
            "round {round} of an unseeded run must record no seed and no mechanism"
        );
    }
    assert_eq!(
        summary_value(&run_summary(plain.path()).summary, "seed_policy"),
        "per_round_offset",
        "a post-§14.23 run must name the seed policy the rounds ran under"
    );

    let seeded = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(seeded.path(), PRESERVING_PATCH);
    let seeded_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let seeded_output = run_patch_mode_with_output(
        seeded.path(),
        &patch_path,
        &seeded_server,
        3,
        &[("VANTA_LLM_SEED", "4242")],
    );
    assert_eq!(
        seeded_output.status.code(),
        Some(0),
        "the seed is a sampling hint, not a verdict input, so the verdict must be the same; \
         stdout:\n{}",
        String::from_utf8_lossy(&seeded_output.stdout)
    );
    let seeded_bodies = seeded_server.request_bodies();
    assert_eq!(
        seeded_bodies.len(),
        plain_bodies.len(),
        "the seed must not change how many calls the run makes"
    );
    let seeded_usage = usage_rows(seeded.path());
    assert_eq!(
        seeded_usage.len(),
        seeded_bodies.len(),
        "one `call.usage` row per chat body"
    );
    // The offset is only exercised if the run really reached later rounds.
    assert!(
        seeded_usage.iter().any(|(round, ..)| *round == 2)
            && seeded_usage.iter().any(|(round, ..)| *round == 3),
        "the run must span rounds 1..3 for the offset to be measured; got {:?}",
        seeded_usage.iter().map(|(round, ..)| *round).collect::<Vec<_>>()
    );
    assert_eq!(seeded_usage[0].0, 1, "the first call is in round 1");
    // Round 1 sends the pinned seed UNCHANGED; round `r` sends pinned + `r - 1`.
    for (round, seed, mechanism) in &seeded_usage {
        let expected = (4242 + u64::from(round - 1)).to_string();
        assert_eq!(
            seed, &expected,
            "round {round} must send the pinned seed plus the round offset"
        );
        assert_eq!(
            mechanism, "llamacpp:seed",
            "a seeded call must record the mechanism that carried the seed"
        );
    }
    for (body, (round, seed, _)) in seeded_bodies.iter().zip(seeded_usage.iter()) {
        assert!(
            body.contains(&format!("\"seed\":{seed}")),
            "the round {round} body must carry the seed its row records ({seed}); got: {body}"
        );
    }
    assert_eq!(
        summary_value(&run_summary(seeded.path()).summary, "seed_policy"),
        "per_round_offset",
        "the seeded run must name the same seed policy"
    );
    // The ONLY difference the new field may introduce is the key itself.
    for (unseeded, seeded) in plain_bodies.iter().zip(seeded_bodies.iter()) {
        assert_eq!(
            body_without_seed(seeded),
            body_without_seed(unseeded),
            "with the seed key removed, a seeded body must be byte-identical to the unseeded one"
        );
    }
}

/// §14.26: on the hosted backend, the sampler keys a llama.cpp body carries are
/// DROPPED before the request leaves, the output cap is RENAMED to the spelling a
/// reasoning model accepts, and every other field the body asked for still goes.
///
/// [VANTA-WHY] The two runs below differ in ONE variable — `VANTA_LLM_BACKEND`
/// (plus the model id it names) — so the local run is the reference the hosted run
/// is measured against, field by field. The requirement has two halves and this
/// test is deliberately both: a hosted body must lose exactly the documented
/// fields (item 3), and the LOCAL body must still carry every one of them, because
/// "the local path stays byte-identical" is only true while nothing filters it.
/// The dropped-name list is spelled out here rather than read from the capability
/// table on purpose: a table that drifted from the API docs would otherwise keep
/// passing against itself. The per-call rows must agree with the bodies, too —
/// `requested_fields` admitting a key the `sent_fields` no longer has is the whole
/// point of recording both.
///
/// §14.28 adds a second question this test has to answer, because the effort is
/// the one field whose correct VALUE is also per-model: a level a family does not
/// list must stop the run before it starts, and the same level on a family that
/// does list it must go out on the wire. Both halves run below, so neither the
/// refusal nor the acceptance can pass for the wrong reason.
#[test]
fn the_hosted_backend_sends_only_the_fields_the_model_accepts() {
    let _serial = serial_run();

    // The sampler keys `VANTA_SAMPLING=coding` pins at EVERY site (§14.11). The
    // reference run needs the fullest sampler set the harness ever sends, because a
    // body that never carried `top_k` could not prove a drop.
    const SAMPLER_KEYS: &[&str] = &[
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
    ];
    // The full documented drop list for these models, per the API reference: the
    // four sampler values plus the two the harness can also send (`frequency_penalty`
    // under a preset, `reasoning_budget_tokens` at the three budget sites) plus the
    // logprob pair. The cap is not dropped — it is renamed — so it is listed apart.
    //
    // §14.28 adds `reasoning_effort` to this list: the call sites' hard-coded `low`
    // is a llama.cpp fact, so with `VANTA_REASONING_EFFORT` unset — which is what
    // this run is — the hosted body must not carry the field at all.
    const DROPPED_KEYS: &[&str] = &[
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "reasoning_budget_tokens",
        "reasoning_effort",
    ];
    const RENAME: &str = "max_tokens->max_completion_tokens";

    // (1) The local reference run.
    let local = TempDir::new().unwrap();
    let (_repo, _issue, local_patch) = write_test_repo(local.path(), PRESERVING_PATCH);
    let local_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let local_output = run_patch_mode_with_output(
        local.path(),
        &local_patch,
        &local_server,
        3,
        &[("VANTA_SAMPLING", "coding")],
    );
    assert_eq!(
        local_output.status.code(),
        Some(0),
        "the reference run must be the approval; stdout:\n{}",
        String::from_utf8_lossy(&local_output.stdout)
    );
    let local_bodies = local_server.request_bodies();
    assert!(
        !local_bodies.is_empty(),
        "the reference run must have called the LLM"
    );
    for body in &local_bodies {
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        let object = value.as_object().expect("a chat body is a JSON object");
        for key in SAMPLER_KEYS {
            assert!(
                object.contains_key(*key),
                "the LOCAL `coding` body must still carry `{key}` — the local path is \
                 not allowed to filter anything: {body}"
            );
        }
        assert!(
            object.contains_key("max_tokens"),
            "the local body must keep the llama.cpp cap spelling: {body}"
        );
        assert!(
            !object.contains_key("max_completion_tokens"),
            "the llama.cpp body must never hear the hosted cap spelling: {body}"
        );
    }
    // §14.28's host-side drop is only a fact about something that was there: the
    // reference run must carry the call sites' own effort somewhere, or "the
    // hosted body does not" would be true of a run that never had one either.
    assert!(
        local_bodies
            .iter()
            .any(|body| body.contains("reasoning_effort")),
        "the LOCAL run must carry the sites' hard-coded effort, or §14.28's drop \
         proves nothing: {local_bodies:?}"
    );
    // The local run's rule is "no filter", and it says so rather than staying
    // silent — that is what makes a local summary self-describing.
    let local_summary = run_summary(local.path()).summary;
    assert_eq!(
        summary_value(&local_summary, "request_shape_backend"),
        "llamacpp"
    );
    assert_eq!(
        summary_value(&local_summary, "request_shape_not_sent"),
        "none",
        "the local run drops nothing: {local_summary}"
    );
    assert_eq!(summary_value(&local_summary, "request_shape_renamed"), "none");

    // (2) The hosted run: same repo, same patch, same stub — only the backend differs.
    let hosted = TempDir::new().unwrap();
    let (_repo, _issue, hosted_patch) = write_test_repo(hosted.path(), PRESERVING_PATCH);
    let hosted_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let hosted_output = run_patch_mode_with_output(
        hosted.path(),
        &hosted_patch,
        &hosted_server,
        3,
        &[
            ("VANTA_SAMPLING", "coding"),
            ("VANTA_LLM_BACKEND", "openai"),
            ("VANTA_LLM_MODEL", "gpt-6-luna"),
            ("OPENAI_API_KEY", "sk-test-not-a-real-key"),
            ("VANTA_LLM_CONTEXT_TOKENS", "131072"),
        ],
    );
    assert_eq!(
        hosted_output.status.code(),
        Some(0),
        "shaping the request must not change the verdict — only the dialect; stdout:\n{}",
        String::from_utf8_lossy(&hosted_output.stdout)
    );
    let hosted_bodies = hosted_server.request_bodies();
    assert_eq!(
        hosted_bodies.len(),
        local_bodies.len(),
        "shaping must not change how many calls the run makes"
    );

    for (local_body, hosted_body) in local_bodies.iter().zip(hosted_bodies.iter()) {
        let local_value: serde_json::Value = serde_json::from_str(local_body).unwrap();
        let hosted_value: serde_json::Value = serde_json::from_str(hosted_body).unwrap();
        let local_object = local_value.as_object().unwrap();
        let hosted_object = hosted_value.as_object().unwrap();

        for key in DROPPED_KEYS {
            assert!(
                !hosted_object.contains_key(*key),
                "the hosted body must not send `{key}`: {hosted_body}"
            );
        }
        assert!(
            hosted_object.contains_key("max_completion_tokens"),
            "the hosted body must carry the cap under the name the model needs: {hosted_body}"
        );
        assert!(
            !hosted_object.contains_key("max_tokens"),
            "the legacy cap spelling is the one these models reject: {hosted_body}"
        );

        // The strongest form of "nothing else changed": rebuild the expected hosted
        // key set by applying exactly the documented rule to the local body, and
        // demand equality. A filter that ate a field nobody listed fails here, and
        // so does one that forgot to rename the cap.
        let mut expected: Vec<String> = local_object
            .keys()
            .filter(|key| !DROPPED_KEYS.contains(&key.as_str()))
            .map(|key| {
                if key == "max_tokens" {
                    "max_completion_tokens".to_string()
                } else {
                    key.clone()
                }
            })
            .collect();
        expected.sort();
        let mut actual: Vec<String> = hosted_object.keys().cloned().collect();
        actual.sort();
        assert_eq!(
            actual, expected,
            "the hosted body must differ from the local one by the documented rule and \
             nothing else; local={local_body} hosted={hosted_body}"
        );
    }

    // (3) Both sides of the question are on the per-call record.
    let usage_rows: Vec<String> = read_session_records(&find_session_log(hosted.path()))
        .into_iter()
        .filter(|record| record.phase == "call.usage")
        .map(|record| record.summary)
        .collect();
    assert_eq!(
        usage_rows.len(),
        hosted_bodies.len(),
        "one `call.usage` row per chat body"
    );
    let split = |list: &str| -> Vec<String> { list.split(',').map(str::to_string).collect() };
    for (index, row) in usage_rows.iter().enumerate() {
        let requested = split(&summary_value(row, "requested_fields"));
        let sent = split(&summary_value(row, "sent_fields"));
        let dropped = split(&summary_value(row, "dropped_fields"));
        for key in SAMPLER_KEYS {
            assert!(
                requested.contains(&key.to_string()),
                "call {index}: the row must admit the body REQUESTED `{key}`: {row}"
            );
            assert!(
                !sent.contains(&key.to_string()),
                "call {index}: the row must not claim `{key}` was sent: {row}"
            );
            assert!(
                dropped.contains(&key.to_string()),
                "call {index}: the row must name `{key}` as dropped: {row}"
            );
        }
        assert!(
            sent.contains(&"max_completion_tokens".to_string()),
            "call {index}: the record must show the renamed cap as sent: {row}"
        );
        assert!(
            !sent.contains(&"max_tokens".to_string()),
            "call {index}: the record must not show the legacy cap as sent: {row}"
        );
        assert_eq!(
            summary_value(row, "renamed_fields"),
            RENAME,
            "call {index}: the rename must be recorded: {row}"
        );
    }

    // (4) The run-level rule explains those rows, and names every dropped key.
    let hosted_summary = run_summary(hosted.path()).summary;
    assert_eq!(
        summary_value(&hosted_summary, "request_shape_backend"),
        "openai"
    );
    assert_eq!(
        summary_value(&hosted_summary, "request_shape_family"),
        "gpt-6-luna",
        "the rule must name the family it was applied for: {hosted_summary}"
    );
    assert_eq!(
        summary_value(&hosted_summary, "request_shape_renamed"),
        RENAME
    );
    let not_sent = split(&summary_value(&hosted_summary, "request_shape_not_sent"));
    for key in DROPPED_KEYS {
        assert!(
            not_sent.contains(&key.to_string()),
            "run.summary must name `{key}` as not sent: {hosted_summary}"
        );
    }

    // (5) §14.28 — the effort is checked against the MODEL, not against a fixed
    //     list. `gpt-6.1-sol`'s page has no `none` and `gpt-6-luna`'s does, so the
    //     SAME switch value has to be refused on one and run on the other. Both
    //     halves are asserted, because a check that refused `none` for every model
    //     would pass the refusal on its own.
    let hosted_arm: &[(&str, &str)] = &[
        ("VANTA_LLM_BACKEND", "openai"),
        ("VANTA_LLM_CONTEXT_TOKENS", "131072"),
        ("OPENAI_API_KEY", "sk-test-not-a-real-key"),
        ("VANTA_REASONING_EFFORT", "none"),
    ];
    let mut sol_env: Vec<(&str, &str)> = hosted_arm.to_vec();
    sol_env.push(("VANTA_LLM_MODEL", "gpt-6.1-sol"));
    let refused = TempDir::new().unwrap();
    let (_repo, _issue, refused_patch) = write_test_repo(refused.path(), PRESERVING_PATCH);
    let refused_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let refused_output =
        run_patch_mode_with_output(refused.path(), &refused_patch, &refused_server, 1, &sol_env);
    assert_eq!(
        refused_output.status.code(),
        Some(1),
        "an effort the run's model does not accept must fail the run; stdout:\n{}",
        String::from_utf8_lossy(&refused_output.stdout)
    );
    let refused_stderr = String::from_utf8_lossy(&refused_output.stderr);
    assert!(
        refused_stderr.contains("CONFIG_ERROR") && refused_stderr.contains("none"),
        "the refusal must name the level it refused; stderr:\n{refused_stderr}"
    );
    assert!(
        refused_stderr.contains("gpt-6.1-sol"),
        "the refusal must name the MODEL that does not accept the level — the ladder \
         is per-family, so the reader has to be able to see which family said no; \
         stderr:\n{refused_stderr}"
    );
    assert!(
        refused_server.request_bodies().is_empty(),
        "a refused config must fail BEFORE the first call, or the run has already \
         spent an hour generating"
    );

    // The other half: the same `none` on the family whose page documents it.
    let mut luna_env: Vec<(&str, &str)> = hosted_arm.to_vec();
    luna_env.push(("VANTA_LLM_MODEL", "gpt-6-luna"));
    let accepted = TempDir::new().unwrap();
    let (_repo, _issue, accepted_patch) = write_test_repo(accepted.path(), PRESERVING_PATCH);
    let accepted_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let accepted_output = run_patch_mode_with_output(
        accepted.path(),
        &accepted_patch,
        &accepted_server,
        3,
        &luna_env,
    );
    assert_eq!(
        accepted_output.status.code(),
        Some(0),
        "the same value on the family that documents it must run; stdout:\n{}",
        String::from_utf8_lossy(&accepted_output.stdout)
    );
    assert!(
        accepted_server.request_bodies().iter().any(|body| {
            serde_json::from_str::<serde_json::Value>(body)
                .ok()
                .and_then(|value| value.get("reasoning_effort").cloned())
                == Some(serde_json::json!("none"))
        }),
        "a run that asked for `none` must SEND it to the model that accepts it: {:?}",
        accepted_server.request_bodies()
    );
    let accepted_summary = run_summary(accepted.path()).summary;
    assert_eq!(
        summary_value(&accepted_summary, "reasoning_effort"),
        "none",
        "{accepted_summary}"
    );
    assert_eq!(
        summary_value(&accepted_summary, "reasoning_effort_mechanism"),
        "openai:reasoning_effort",
        "an effort that went out is the hosted mechanism, not a no-op: {accepted_summary}"
    );
    // The run-level rule must not claim the field was withheld: this run sent it.
    // That is the other direction of §14.28's one predicate, and the reason it is
    // shared with the shaper rather than re-derived here.
    let accepted_not_sent = summary_value(&accepted_summary, "request_shape_not_sent");
    assert!(
        !split(&accepted_not_sent)
            .iter()
            .any(|key| key == "reasoning_effort"),
        "the run-level rule and the bodies must agree: {accepted_summary}"
    );
    let accepted_rows: Vec<String> = read_session_records(&find_session_log(accepted.path()))
        .into_iter()
        .filter(|record| record.phase == "call.usage")
        .map(|record| record.summary)
        .collect();
    assert!(
        !accepted_rows.is_empty(),
        "the accepted run made no calls, so it proves nothing about the effort"
    );
    for row in &accepted_rows {
        assert!(
            split(&summary_value(row, "sent_fields")).contains(&"reasoning_effort".to_string()),
            "a body that carries the effort must say so: {row}"
        );
    }
}

/// §4.2 `baseline`: `VANTA_ATTACKER=off` skips the Attacker phase entirely, so
/// the verdict comes from the repo's own suite and static analysis — and not one
/// Attacker LLM call is made.
#[test]
fn attacker_off_makes_zero_attacker_calls_and_grades_on_native_and_static() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // A real native suite, so the approval rests on a test that actually ran
    // rather than on a vacuously-empty one.
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        PRESERVING_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    // A stub that WOULD answer an Attacker call: if one is made, this test sees it.
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_ATTACKER", "off")],
    );

    assert!(
        output.status.success(),
        "a behaviour-preserving patch with a passing native suite and clean static \
         analysis is the native+static baseline approval; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        server.request_count(),
        0,
        "VANTA_ATTACKER=off must skip the Attacker phase — zero LLM calls; kinds={:?}",
        server.request_kinds()
    );

    let summary = run_summary(workspace.path());
    assert!(
        summary.summary.contains("config=baseline"),
        "the config is derived from the switches, so Attacker-off must record `baseline`; got {}",
        summary.summary
    );
}

/// §4.2 third configuration: `VANTA_PLANNER=on` is a PLACEHOLDER. It must end as
/// an explicit `InfraFailure` instead of a fabricated result, and it must not do
/// any work — no LLM call, no verdict about a candidate.
#[test]
fn planner_on_ends_as_an_explicit_infrastructure_failure() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_PLANNER", "on")],
    );

    assert_eq!(
        output.status.code(),
        Some(2),
        "the planner placeholder must reuse the InfraFailure exit code; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        server.request_count(),
        0,
        "the placeholder must stop before any work; kinds={:?}",
        server.request_kinds()
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let summary = records
        .iter()
        .find(|r| r.phase == "run.summary")
        .expect("even the placeholder must write a run.summary row");
    assert!(
        summary.summary.contains("config=vanta_planner"),
        "got {}",
        summary.summary
    );
    assert!(
        summary.summary.contains("final_decision=InfraFailure"),
        "got {}",
        summary.summary
    );
    assert_eq!(
        summary.benchmark_instance.as_deref(),
        Some("vanta_planner"),
        "the manifest groups by benchmark_instance, so it must name the config"
    );
    assert_eq!(
        summary.benchmark_label.as_deref(),
        Some("infra_failure"),
        "an InfraFailure is the one verdict a run can label itself (§5.3)"
    );
    assert!(
        records.iter().any(|r| r.phase == "run.infra_failure"),
        "the reason must be recorded beside the verdict"
    );
}

/// §4.3 (corrected): the harness PINS `temperature`/`max_tokens` per call site,
/// so the provenance record is a per-agent SCHEDULE, not a run-level scalar. The
/// recorded schedule must be the one the bodies actually carried.
#[test]
fn the_recorded_sampling_schedule_matches_the_request_bodies() {
    let _serial = serial_run();

    // (1) Review mode: no Creator call, but the Attacker's generate and the
    //     patch-blind intent audit both fire.
    let review = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(review.path(), PRESERVING_PATCH);
    let review_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let status = run_patch_mode(review.path(), &patch_path, &review_server);
    assert!(status.success(), "the reference run must reach a verdict");
    let checked =
        assert_schedule_matches_bodies(review.path(), &review_server, vanta::llm::SamplingPreset::Pinned);
    for required in ["attacker_generate", "intent_audit"] {
        assert!(
            checked.iter().any(|site| site == required),
            "review mode must exercise `{required}`; checked={checked:?}"
        );
    }

    // (2) Repair mode: the Creator call runs too, so its site is covered as well —
    //     the schedule table names five sites and this is what keeps the Creator
    //     one from being a value nothing ever checks.
    let repair = TempDir::new().unwrap();
    let _ = write_test_repo(repair.path(), PRESERVING_PATCH);
    let creator_answer = serde_json::json!({
        "patch_text": WRONG_BEHAVIOUR_PATCH,
        "explanation": "stub candidate for the sampling-schedule test",
        "files_touched": ["calc.py"],
        "confidence": 0.2
    })
    .to_string();
    let repair_server =
        StubLlmServer::start_with_creator_and_attacker(&creator_answer, THREE_SIGNAL_TESTS);
    let _ = run_creator_mode_with_output(repair.path(), &repair_server, 1, &[]);
    let checked =
        assert_schedule_matches_bodies(repair.path(), &repair_server, vanta::llm::SamplingPreset::Pinned);
    for required in ["creator", "attacker_generate", "intent_audit"] {
        assert!(
            checked.iter().any(|site| site == required),
            "repair mode must exercise `{required}`; checked={checked:?}"
        );
    }
}

/// §14.11: `VANTA_SAMPLING=coding` moves EVERY call site — the auditor included —
/// to the coding sampler, and leaves the `pinned` run's bodies exactly as they
/// were.
///
/// [VANTA-WHY] The preset is a sampling choice, not a verdict input, so a `coding`
/// run must reach the SAME verdict an otherwise identical `pinned` run reaches.
/// The auditor used to be excluded from the preset (`site != "intent_audit"`),
/// which left it judging a 0.6-sampled Attacker's tests at 0.0 — the one site
/// whose output is a judgement on the run's noisiest generation. It is now covered
/// like the rest, while `max_tokens` is NOT part of the preset: the auditor keeps
/// its 8192 cap, so a cut-off audit at 0.6 takes the retry path (temperature
/// above 0.0) instead of the temperature-0.0 fail-fast. Both halves are asserted
/// here, against the recorded schedule AND the bodies the server received.
#[test]
fn the_coding_preset_covers_every_site_including_the_auditor() {
    let _serial = serial_run();

    // (1) A `pinned` run, to pin what the auditor must NOT do: 0.0 and none of the
    //     coding keys.
    let plain = TempDir::new().unwrap();
    let (_repo, _issue, plain_patch) = write_test_repo(plain.path(), PRESERVING_PATCH);
    let plain_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let plain_output = run_patch_mode_with_output(plain.path(), &plain_patch, &plain_server, 3, &[]);
    assert_eq!(
        plain_output.status.code(),
        Some(0),
        "the reference run must be the approval; stdout:\n{}",
        String::from_utf8_lossy(&plain_output.stdout)
    );
    let plain_summary = run_summary(plain.path()).summary;
    assert_eq!(
        summary_value(&plain_summary, "temperature_intent_audit"),
        "0.0"
    );
    assert!(
        !plain_summary.contains("top_p_intent_audit"),
        "a pinned run must not record a coding pin for the auditor: {plain_summary}"
    );
    let mut pinned_audits = 0usize;
    for (kind, body) in plain_server
        .request_kinds()
        .iter()
        .zip(plain_server.request_bodies().iter())
    {
        if kind != "intent_audit" {
            continue;
        }
        pinned_audits += 1;
        assert!(
            body.contains("\"temperature\":0.0"),
            "the pinned auditor body must sample at 0.0: {body}"
        );
        for key in ["top_p", "top_k", "min_p", "presence_penalty"] {
            assert!(
                !body.contains(&format!("\"{key}\"")),
                "the pinned auditor body must carry no coding `{key}` pin: {body}"
            );
        }
    }
    assert!(
        pinned_audits > 0,
        "the pinned run must have audited something, or (1) proves nothing"
    );

    // (2) The coding run.
    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        3,
        &[("VANTA_SAMPLING", "coding")],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "the preset is a sampling choice, not a verdict input, so an approving run \
         stays the approval; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(
        summary_value(&summary, "sampling_preset"),
        "coding",
        "run.summary must name the active preset: {summary}"
    );

    // Every site, the auditor included, takes the coding sampler.
    for site in [
        "creator",
        "attacker_generate",
        "attacker_revise",
        "attacker_simplify",
        "intent_audit",
    ] {
        assert_eq!(summary_value(&summary, &format!("temperature_{site}")), "0.6");
        assert_eq!(summary_value(&summary, &format!("top_p_{site}")), "0.95");
        assert_eq!(summary_value(&summary, &format!("top_k_{site}")), "20");
        assert_eq!(summary_value(&summary, &format!("min_p_{site}")), "0.0");
        assert_eq!(
            summary_value(&summary, &format!("presence_penalty_{site}")),
            "0.0"
        );
    }
    // The preset moves samplers, not caps: the auditor keeps its 8192.
    assert_eq!(summary_value(&summary, "max_tokens_intent_audit"), "8192");
    assert_eq!(summary_value(&summary, "max_tokens_attacker_generate"), "32768");

    // The recorded schedule must be the one the bodies carried — the auditor's
    // body is the one this test exists for.
    let checked =
        assert_schedule_matches_bodies(workspace.path(), &server, vanta::llm::SamplingPreset::Coding);
    for required in ["attacker_generate", "intent_audit"] {
        assert!(
            checked.iter().any(|site| site == required),
            "the coding run must exercise `{required}`; checked={checked:?}"
        );
    }
    let mut coding_audits = 0usize;
    for (kind, body) in server
        .request_kinds()
        .iter()
        .zip(server.request_bodies().iter())
    {
        if kind != "intent_audit" {
            continue;
        }
        coding_audits += 1;
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        assert_eq!(value.get("temperature").and_then(|v| v.as_f64()), Some(0.6));
        assert_eq!(value.get("top_p").and_then(|v| v.as_f64()), Some(0.95));
        assert_eq!(value.get("top_k").and_then(|v| v.as_u64()), Some(20));
        assert_eq!(value.get("min_p").and_then(|v| v.as_f64()), Some(0.0));
        assert_eq!(
            value.get("max_tokens").and_then(|v| v.as_u64()),
            Some(8192),
            "the coding preset must not move the auditor's cap: {body}"
        );
    }
    assert!(
        coding_audits > 0,
        "the coding run must have audited something, or the auditor's coding pins \
         were never sent"
    );
}

/// §14.16 — `VANTA_PRESENCE_PENALTY` reaches every body, and zero reaches none.
///
/// [VANTA-WHY] The overnight batch's run-aways were the SAME answer rewritten
/// 45-87 times until `max_tokens` cut the stream, and `presence_penalty` is the
/// sampler parameter that penalises tokens already emitted. Whether it helps is an
/// experiment — one arm at 0.0 and one at 1.0 — and an experiment whose arms are
/// not distinguishable afterwards is not one. Zero must therefore change NOTHING
/// in the request bodies (or the control is not a control) while still being
/// RECORDED, and a non-zero value must reach every site's body. Both halves are
/// asserted here, against the bodies the server actually received.
#[test]
fn the_presence_penalty_override_reaches_every_body_and_zero_reaches_none() {
    let _serial = serial_run();

    // (1) The control: asked for zero. Bodies byte-identical, value on the record.
    let control = TempDir::new().unwrap();
    let (_repo, _issue, control_patch) = write_test_repo(control.path(), PRESERVING_PATCH);
    let control_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let control_output = run_patch_mode_with_output(
        control.path(),
        &control_patch,
        &control_server,
        1,
        &[("VANTA_PRESENCE_PENALTY", "0")],
    );
    assert_eq!(
        control_output.status.code(),
        Some(0),
        "the override is a sampling choice and must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&control_output.stdout)
    );
    let control_bodies = control_server.request_bodies();
    assert!(
        !control_bodies.is_empty(),
        "the control run must have made calls, or it proves nothing about them"
    );
    for body in &control_bodies {
        assert!(
            !body.contains("\"presence_penalty\""),
            "a zero penalty must leave the body byte-identical to the default: {body}"
        );
    }
    let control_summary = run_summary(control.path()).summary;
    assert_eq!(
        summary_value(&control_summary, "presence_penalty"),
        "0.0",
        "zero and unset send the same bodies, so the ONLY way a batch table can \
         tell the two arms apart is this field; got {control_summary}"
    );

    // (2) The arm: asked for 1.0. Every site's body carries it, and the record
    //     matches the bodies site by site (`assert_schedule_matches_bodies` is the
    //     same guard the preset uses, so the override cannot drift from the pins).
    let arm = TempDir::new().unwrap();
    let (_repo, _issue, arm_patch) = write_test_repo(arm.path(), PRESERVING_PATCH);
    let arm_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let arm_output = run_patch_mode_with_output(
        arm.path(),
        &arm_patch,
        &arm_server,
        1,
        &[("VANTA_PRESENCE_PENALTY", "1.0")],
    );
    assert_eq!(
        arm_output.status.code(),
        Some(0),
        "a penalised run of the same approving candidate is still the approval; \
         stdout:\n{}",
        String::from_utf8_lossy(&arm_output.stdout)
    );
    for body in arm_server.request_bodies() {
        let value: serde_json::Value = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        assert_eq!(
            value.get("presence_penalty").and_then(|v| v.as_f64()),
            Some(1.0),
            "every call site must carry the override, the auditor included: {body}"
        );
    }
    let arm_summary = run_summary(arm.path()).summary;
    assert_eq!(summary_value(&arm_summary, "presence_penalty"), "1.0");
    for site in [
        "creator",
        "attacker_generate",
        "attacker_revise",
        "attacker_simplify",
        "intent_audit",
    ] {
        assert_eq!(
            summary_value(&arm_summary, &format!("presence_penalty_{site}")),
            "1.0",
            "the record is the schedule, so every site must be named even when this \
             run never reached it: {arm_summary}"
        );
    }
    assert_schedule_matches_bodies(arm.path(), &arm_server, vanta::llm::SamplingPreset::Pinned);
}

/// §14.20 — `VANTA_MAX_TOKENS_<SITE>` replaces one site's `max_tokens` slot.
///
/// [VANTA-WHY] A revised test that spends its whole 32,768-token cap re-checking
/// its own formatting is a run-away §14.14's call budget cannot see, because it is
/// ONE call. The switch is an ablation arm, so it has to be true in three places
/// at once: each body carries the cap the record names for ITS OWN site, the sites
/// the switch does not name keep the schedule, and `run.summary` names what was
/// ASKED FOR — because a reader comparing batch directories needs the switch, not
/// a number that happens to differ. A malformed value must fail the run before it
/// spends an hour generating, which is asserted on the exit code and the absence
/// of any call.
#[test]
fn the_output_cap_override_replaces_one_sites_slot_and_only_that_site() {
    let _serial = serial_run();

    // (1) The control: nothing asked for. Every body keeps the schedule's cap and
    //     the record says so.
    let control = TempDir::new().unwrap();
    let (_repo, _issue, control_patch) = write_test_repo(control.path(), PRESERVING_PATCH);
    let control_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let control_output =
        run_patch_mode_with_output(control.path(), &control_patch, &control_server, 1, &[]);
    assert_eq!(
        control_output.status.code(),
        Some(0),
        "the switch is a sampling choice and must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&control_output.stdout)
    );
    let control_summary = run_summary(control.path()).summary;
    assert_eq!(
        summary_value(&control_summary, "max_tokens_overrides"),
        "unset",
        "an unset switch is not an empty pin; the record must say nobody asked: {control_summary}"
    );
    assert_schedule_matches_bodies(
        control.path(),
        &control_server,
        vanta::llm::SamplingPreset::Pinned,
    );

    // (2) The arm: two sites capped. Each capped body carries 12288 and exactly
    //     one `max_tokens` key, the other three sites keep the schedule, and the
    //     run-level field names the switch in schedule order.
    let arm = TempDir::new().unwrap();
    let (_repo, _issue, arm_patch) = write_test_repo(arm.path(), PRESERVING_PATCH);
    let arm_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let arm_output = run_patch_mode_with_output(
        arm.path(),
        &arm_patch,
        &arm_server,
        1,
        &[
            ("VANTA_MAX_TOKENS_ATTACKER_REVISE", "12288"),
            ("VANTA_MAX_TOKENS_ATTACKER_SIMPLIFY", "12288"),
        ],
    );
    assert_eq!(
        arm_output.status.code(),
        Some(0),
        "a lower output cap on two remediation sites must leave the verdict alone; \
         stdout:\n{}",
        String::from_utf8_lossy(&arm_output.stdout)
    );
    let arm_bodies = arm_server.request_bodies();
    let arm_kinds = arm_server.request_kinds();
    assert_eq!(arm_kinds.len(), arm_bodies.len(), "one kind per recorded body");
    assert!(
        !arm_bodies.is_empty(),
        "the arm must have made calls, or it proves nothing about them"
    );
    // [VANTA-WHY] §14.20 — the per-body check maps kind -> site -> the cap the
    // RECORD names for that site, rather than against `sampling_pins_text`, whose
    // ground truth is the PARENT's pin table and therefore cannot see an override
    // the child read from the environment. `assert_schedule_matches_bodies` is
    // deliberately not used on the arm for that reason: it would compare the
    // child's 12288 against the parent's 32768 and report a broken override.
    let arm_summary = run_summary(arm.path()).summary;
    let expected_cap = |site: &str| -> u64 {
        summary_value(&arm_summary, &format!("max_tokens_{site}"))
            .parse()
            .expect("every site's effective cap is recorded")
    };
    for (kind, body) in arm_kinds.iter().zip(arm_bodies.iter()) {
        let site = SITE_FOR_KIND
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, site)| *site)
            .unwrap_or_else(|| panic!("stub saw an unmapped request kind `{kind}`"));
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        assert_eq!(
            value.get("max_tokens").and_then(|v| v.as_u64()),
            Some(expected_cap(site)),
            "the `{kind}` body must carry the cap the record names for `{site}`: {body}"
        );
    }
    assert_eq!(
        summary_value(&arm_summary, "max_tokens_overrides"),
        "attacker_revise=12288,attacker_simplify=12288",
        "the run-level field is the switch itself, in schedule order; got {arm_summary}"
    );
    assert_eq!(
        summary_value(&arm_summary, "max_tokens_attacker_revise"),
        "12288"
    );
    assert_eq!(
        summary_value(&arm_summary, "max_tokens_attacker_simplify"),
        "12288"
    );
    for (site, scheduled) in [
        ("creator", "32768"),
        ("attacker_generate", "32768"),
        ("intent_audit", "8192"),
    ] {
        assert_eq!(
            summary_value(&arm_summary, &format!("max_tokens_{site}")),
            scheduled,
            "a site the switch does not name must keep the schedule: {arm_summary}"
        );
    }

    // (3) Malformed and zero values are refused before a single call is made.
    for (name, value) in [
        ("VANTA_MAX_TOKENS_ATTACKER_REVISE", "12k"),
        ("VANTA_MAX_TOKENS_ATTACKER_REVISE", "0"),
    ] {
        let bad = TempDir::new().unwrap();
        let (_repo, _issue, bad_patch) = write_test_repo(bad.path(), PRESERVING_PATCH);
        let bad_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
        let bad_output =
            run_patch_mode_with_output(bad.path(), &bad_patch, &bad_server, 1, &[(name, value)]);
        assert_eq!(
            bad_output.status.code(),
            Some(1),
            "{name}={value} must fail the run; stdout:\n{}",
            String::from_utf8_lossy(&bad_output.stdout)
        );
        let stderr = String::from_utf8_lossy(&bad_output.stderr);
        assert!(
            stderr.contains("CONFIG_ERROR") && stderr.contains(name),
            "{name}={value} must name the offending switch; stderr:\n{stderr}"
        );
        assert!(
            bad_server.request_bodies().is_empty(),
            "a refused config must fail BEFORE the first call, or the run has already \
             spent an hour generating"
        );
    }
}

/// §14.21 — the reasoning controls reach the Attacker's calls and nothing else.
///
/// [VANTA-WHY] §14.13 measured a model that worked the answer out and then spent
/// its whole output cap re-checking it. §14.20 caps the ANSWER; this caps the
/// THINKING, on the three Attacker sites whose reasoning the run-away was in.
/// The knob has to be true in three places at once: the allow-listed bodies carry
/// the control, the sites outside the allow-list do not, and `run.summary` names
/// both the switch and the mechanism that implemented it — the mechanism is what
/// makes the same arm comparable with a hosted backend later. A malformed value,
/// and a zero, must fail the run before it spends an hour generating, which is
/// asserted on the exit code and the absence of any call. The control run is held
/// to `assert_schedule_matches_bodies`, so the claim that an unset switch changes
/// nothing is checked against the same ground truth every other sampling test uses.
#[test]
fn the_reasoning_budget_reaches_the_attacker_calls_and_leaves_a_default_run_alone() {
    let _serial = serial_run();

    // The budget the arm asks for — a plausible ceiling for one thinking block,
    // nowhere near the model's own 32,768-token output cap — and the same number
    // as `run.summary` renders it.
    const ARM_BUDGET_TOKENS: u64 = 12000;
    const ARM_BUDGET_TEXT: &str = "12000";

    // (1) The control: nothing asked for. No body carries a reasoning budget, the
    //     record reads the two mechanisms the transport has always used, and the
    //     effective effort is the `low` the six call sites have always sent.
    let control = TempDir::new().unwrap();
    let (_repo, _issue, control_patch) = write_test_repo(control.path(), PRESERVING_PATCH);
    let control_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let control_output =
        run_patch_mode_with_output(control.path(), &control_patch, &control_server, 1, &[]);
    assert_eq!(
        control_output.status.code(),
        Some(0),
        "the switch is a reasoning choice and must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&control_output.stdout)
    );
    let control_summary = run_summary(control.path()).summary;
    for (key, expected) in [
        ("reasoning_budget", "unset"),
        ("reasoning_budget_mechanism", "unset"),
        // The EFFECTIVE level, not the switch: `unset` here would describe a run
        // that sent no effort at all, and every run sends `low`.
        ("reasoning_effort", "low"),
        ("reasoning_effort_mechanism", "llamacpp:noop"),
    ] {
        assert_eq!(
            summary_value(&control_summary, key),
            expected,
            "an unset reasoning switch must record `{key}={expected}`: {control_summary}"
        );
    }
    for body in control_server.request_bodies() {
        let value: serde_json::Value = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        assert_eq!(
            value.get("reasoning_budget_tokens"),
            None,
            "an unset reasoning budget must leave every body untouched: {body}"
        );
    }
    assert_schedule_matches_bodies(
        control.path(),
        &control_server,
        vanta::llm::SamplingPreset::Pinned,
    );

    // (2) The arm: a budget and an effort. The three allow-listed sites carry the
    //     budget; the audit and the Creator do not. Every body that carries an
    //     effort level carries the one that was asked for, and no body is handed
    //     one it did not have.
    let arm = TempDir::new().unwrap();
    let (_repo, _issue, arm_patch) = write_test_repo(arm.path(), PRESERVING_PATCH);
    let arm_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let arm_output = run_patch_mode_with_output(
        arm.path(),
        &arm_patch,
        &arm_server,
        1,
        &[
            ("VANTA_REASONING_BUDGET", "12000"),
            ("VANTA_REASONING_EFFORT", "high"),
        ],
    );
    assert_eq!(
        arm_output.status.code(),
        Some(0),
        "a reasoning budget must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&arm_output.stdout)
    );
    let arm_bodies = arm_server.request_bodies();
    let arm_kinds = arm_server.request_kinds();
    assert_eq!(arm_kinds.len(), arm_bodies.len(), "one kind per recorded body");
    let mut budgeted_sites: Vec<String> = Vec::new();
    let mut effort_sites: Vec<String> = Vec::new();
    for (kind, body) in arm_kinds.iter().zip(arm_bodies.iter()) {
        let site = SITE_FOR_KIND
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, site)| *site)
            .unwrap_or_else(|| panic!("stub saw an unmapped request kind `{kind}`"));
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        if vanta::llm::REASONING_BUDGET_SITES.contains(&site) {
            assert_eq!(
                value.get("reasoning_budget_tokens").and_then(|v| v.as_u64()),
                Some(ARM_BUDGET_TOKENS),
                "the `{kind}` body is an allow-listed site and must carry the budget: {body}"
            );
            budgeted_sites.push(site.to_string());
        } else {
            assert_eq!(
                value.get("reasoning_budget_tokens"),
                None,
                "the `{kind}` body (`{site}`) is outside the allow-list and must not carry it: {body}"
            );
        }
        match value.get("reasoning_effort") {
            // A site that asked for an effort gets the one the run asked for…
            Some(level) => {
                assert_eq!(
                    level, "high",
                    "the `{kind}` body must carry the effort the run asked for: {body}"
                );
                effort_sites.push(site.to_string());
            }
            // …and a site that never asked for one is not handed one.
            None => {}
        }
    }
    assert!(
        !budgeted_sites.is_empty(),
        "the arm reached no allow-listed site, so it proves nothing about the budget"
    );
    assert!(
        !effort_sites.is_empty(),
        "the arm reached no site that carries an effort, so it proves nothing about the level"
    );
    let arm_summary = run_summary(arm.path()).summary;
    for (key, expected) in [
        ("reasoning_budget", ARM_BUDGET_TEXT),
        (
            "reasoning_budget_mechanism",
            "llamacpp:reasoning_budget_tokens",
        ),
        ("reasoning_effort", "high"),
        ("reasoning_effort_mechanism", "llamacpp:noop"),
    ] {
        assert_eq!(
            summary_value(&arm_summary, key),
            expected,
            "the arm's record must name `{key}={expected}`: {arm_summary}"
        );
    }

    // (3) A malformed value, a zero budget and an unknown level are each refused
    //     before a single call is made.
    for (name, value) in [
        ("VANTA_REASONING_BUDGET", "12k"),
        ("VANTA_REASONING_BUDGET", "0"),
        ("VANTA_REASONING_EFFORT", "bogus"),
    ] {
        let bad = TempDir::new().unwrap();
        let (_repo, _issue, bad_patch) = write_test_repo(bad.path(), PRESERVING_PATCH);
        let bad_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
        let bad_output =
            run_patch_mode_with_output(bad.path(), &bad_patch, &bad_server, 1, &[(name, value)]);
        assert_eq!(
            bad_output.status.code(),
            Some(1),
            "{name}={value} must fail the run; stdout:\n{}",
            String::from_utf8_lossy(&bad_output.stdout)
        );
        let stderr = String::from_utf8_lossy(&bad_output.stderr);
        assert!(
            stderr.contains("CONFIG_ERROR") && stderr.contains(name),
            "{name}={value} must name the offending switch; stderr:\n{stderr}"
        );
        assert!(
            bad_server.request_bodies().is_empty(),
            "a refused config must fail BEFORE the first call, or the run has already \
             spent an hour generating"
        );
    }
}

/// §14.22 — `VANTA_ATTACKER_CONCISE` adds one sentence, to the generate prompt only.
///
/// [VANTA-WHY] §14.20 caps the ANSWER and §14.21 caps the THINKING; this is the one
/// item that tries to remove the reason the model produced so much at all, so its
/// claim is narrow and is checked narrowly. Three things have to hold at once: the
/// sentence is on the generate prompt's USER message, no other call site carries it,
/// and `run.summary` names which of the two prompts the run sent — that last field is
/// how an ablation row is assigned to an arm after the fact. The SYSTEM message is
/// asserted to be free of the sentence because the §14.10-f `attacker=` hash covers
/// exactly the three system prompts; a suffix that leaked there would change the hash
/// while the switch still reported itself. The control run is held to
/// `assert_schedule_matches_bodies`, so "an unset switch changes nothing" is checked
/// against the same ground truth every other sampling test uses.
#[test]
fn the_concise_switch_appends_the_sentence_to_the_generate_prompt_only() {
    let _serial = serial_run();

    // The text the arm appends, read from the crate rather than copied: the unit
    // test in `agents::attacker` pins the literal, so this test can compare the
    // bodies against the one accessor that produces the sent text.
    let suffix = vanta::agents::attacker::attacker_concise_suffix();

    // The decoded user message of a recorded body. Decoded, not raw: the suffix
    // begins with two newlines, which a raw JSON body carries as the four
    // characters `\n\n`, so a substring test on the raw text would silently pass.
    let user_message = |body: &str| -> String {
        let value: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
        value["messages"]
            .as_array()
            .unwrap_or_else(|| panic!("body must carry a messages array: {body}"))
            .iter()
            .find(|m| m["role"] == "user")
            .and_then(|m| m["content"].as_str())
            .unwrap_or_else(|| panic!("body must carry a user message string: {body}"))
            .to_string()
    };

    // (1) The control: the switch is off, which is every run before §14.22.
    let control = TempDir::new().unwrap();
    let (_repo, _issue, control_patch) = write_test_repo(control.path(), PRESERVING_PATCH);
    let control_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let control_output =
        run_patch_mode_with_output(control.path(), &control_patch, &control_server, 1, &[]);
    assert_eq!(
        control_output.status.code(),
        Some(0),
        "the switch is a prompt choice and must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&control_output.stdout)
    );
    assert_eq!(
        summary_value(&run_summary(control.path()).summary, "attacker_concise"),
        "off",
        "an unset switch must record `attacker_concise=off`, the default prompt"
    );
    let control_kinds = control_server.request_kinds();
    for (kind, body) in control_kinds.iter().zip(control_server.request_bodies()) {
        assert!(
            !user_message(&body).contains(suffix),
            "the default run's `{kind}` USER message must not carry the §14.22 sentence: {body}"
        );
    }
    assert_schedule_matches_bodies(
        control.path(),
        &control_server,
        vanta::llm::SamplingPreset::Pinned,
    );

    // (2) The arm: the sentence rides the generate prompt's USER message and no
    //     other call site's, and the SYSTEM text the §14.10-f hash covers is
    //     untouched because the sentence never reaches it.
    let arm = TempDir::new().unwrap();
    let (_repo, _issue, arm_patch) = write_test_repo(arm.path(), PRESERVING_PATCH);
    let arm_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let arm_output = run_patch_mode_with_output(
        arm.path(),
        &arm_patch,
        &arm_server,
        1,
        &[("VANTA_ATTACKER_CONCISE", "1")],
    );
    assert_eq!(
        arm_output.status.code(),
        Some(0),
        "the switch is a prompt choice and must not move the verdict; stdout:\n{}",
        String::from_utf8_lossy(&arm_output.stdout)
    );
    assert_eq!(
        summary_value(&run_summary(arm.path()).summary, "attacker_concise"),
        "on",
        "the arm must record `attacker_concise=on`, which is how a row is assigned to it"
    );
    let arm_kinds = arm_server.request_kinds();
    let arm_bodies = arm_server.request_bodies();
    assert_eq!(
        arm_kinds, control_kinds,
        "the switch is a prompt choice and must not change which calls are made"
    );
    assert_eq!(arm_kinds.len(), arm_bodies.len(), "one kind per recorded body");
    let mut generate_bodies = 0usize;
    for (kind, body) in arm_kinds.iter().zip(arm_bodies.iter()) {
        let user = user_message(body);
        if kind == "generate" {
            generate_bodies += 1;
            assert!(
                user.ends_with(suffix),
                "the `{kind}` USER message must end with the §14.22 sentence: {body}"
            );
            let value: serde_json::Value = serde_json::from_str(body)
                .unwrap_or_else(|e| panic!("request body is not JSON: {e}\n{body}"));
            let system = value["messages"]
                .as_array()
                .and_then(|ms| ms.iter().find(|m| m["role"] == "system"))
                .and_then(|m| m["content"].as_str())
                .unwrap_or_else(|| panic!("generate body must carry a system message: {body}"));
            assert!(
                !system.contains(suffix),
                "the §14.22 sentence must ride the USER message, not the hashed SYSTEM \
                 prompt, or the attacker= hash stops naming the prompts sent: {body}"
            );
        } else {
            assert!(
                !user.contains(suffix),
                "only the generate prompt may carry the sentence; the `{kind}` body must \
                 not, or the arm is not the narrow ablation it claims to be: {body}"
            );
        }
    }
    assert!(
        generate_bodies > 0,
        "the arm reached no generate call, so it proves nothing about the sentence"
    );
}

/// §4.3 token accounting: the numbers a cost table is built from are the
/// SERVER's, recorded per call, and `run.summary` totals exactly those.
///
/// [VANTA-WHY] Before this, the only prompt figure anywhere was
/// `conservative_token_estimate`, derived from CHARACTERS and deliberately high,
/// and NO completion count existed at all — so any cost number was inflated by
/// an unknown factor with nothing marking it. The stub advertises 777/123/900,
/// which no local estimate can produce (the estimate cannot see completions at
/// all), so the assertions below distinguish a measurement from the fallback by
/// the values themselves rather than by trusting the label.
#[test]
fn token_accounting_records_the_servers_counts_per_call_and_a_run_total() {
    let _serial = serial_run();

    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch_path) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(status.success(), "the reference run must reach a verdict");

    let records = read_session_records(&find_session_log(workspace.path()));
    let usages: Vec<&SessionRecord> = records
        .iter()
        .filter(|r| r.phase == "call.usage")
        .collect();
    assert!(
        !usages.is_empty(),
        "a run that called the model must write one `call.usage` row per call"
    );

    for usage in &usages {
        assert_eq!(
            summary_value(&usage.summary, "prompt_tokens"),
            "777",
            "the prompt count must be the server's, not the character estimate: {}",
            usage.summary
        );
        assert_eq!(
            summary_value(&usage.summary, "completion_tokens"),
            "123",
            "a completion count exists only because the request asked for usage: {}",
            usage.summary
        );
        assert_eq!(
            summary_value(&usage.summary, "total_tokens"),
            "900",
            "total must come from the same usage block: {}",
            usage.summary
        );
        assert_eq!(
            summary_value(&usage.summary, "source"),
            "server-usage",
            "the row must say the numbers were measured, not estimated: {}",
            usage.summary
        );
        let site = summary_value(&usage.summary, "site");
        assert!(
            vanta::llm::SAMPLING_SCHEDULE
                .iter()
                .any(|(s, _, _)| *s == site),
            "`{site}` is not a call site in SAMPLING_SCHEDULE"
        );
    }

    // §14.12 — the round, the retry index, the server's own `finish_reason` and
    // the wall time travel with EVERY call. The runaway diagnosis was made from a
    // log that had the calls and none of these four: the round had to be
    // reconstructed, the retries were invisible, and a 5-10 minute call left no
    // time anywhere but a debug line nothing kept.
    for usage in &usages {
        assert!(
            summary_value(&usage.summary, "round").parse::<u32>().unwrap_or(0) >= 1,
            "every call in a run belongs to a round the driver set: {}",
            usage.summary
        );
        assert_eq!(
            summary_value(&usage.summary, "attempt"),
            "1",
            "this run retries nothing, so every call is attempt 1: {}",
            usage.summary
        );
        assert_eq!(
            summary_value(&usage.summary, "finish_reason"),
            "unset",
            "the stub publishes no finish_reason for a complete answer, and a \
             plausible-looking `stop` would be a claim the server never made: {}",
            usage.summary
        );
        assert!(
            summary_value(&usage.summary, "wall_ms").parse::<u64>().is_ok(),
            "the wall time must be recorded as whole milliseconds: {}",
            usage.summary
        );
    }

    // `seq` is the only ordering key the rows carry — timestamps are wall-clock
    // and two calls can share a second — so it must be dense and increasing, or
    // the order of calls within a run is unrecoverable.
    let seqs: Vec<u64> = usages
        .iter()
        .map(|u| {
            summary_value(&u.summary, "seq")
                .parse::<u64>()
                .expect("seq must be a number")
        })
        .collect();
    assert_eq!(
        seqs,
        (0..seqs.len() as u64).collect::<Vec<_>>(),
        "seq must start at 0 and increment by one call"
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(
        summary_value(&summary, "llm_calls"),
        usages.len().to_string(),
        "the run total must count every recorded call"
    );
    assert_eq!(
        summary_value(&summary, "llm_calls_measured"),
        usages.len().to_string(),
        "every call received a usage block, so every call counts as measured"
    );
    assert_eq!(
        summary_value(&summary, "llm_prompt_tokens"),
        (777 * usages.len()).to_string(),
        "the run total must sum the server's prompt counts"
    );
    assert_eq!(
        summary_value(&summary, "llm_completion_tokens"),
        (123 * usages.len()).to_string(),
        "the run total must sum the server's completion counts"
    );

    // §14.12 — the same facts at the two granularities the runaway report needs:
    // a run total, and every CALL SITE (the Attacker's three included, which the
    // single `attacker` role merges by construction).
    assert_eq!(
        summary_value(&summary, "truncated_calls"),
        "0",
        "nothing was cut off in this run: {summary}"
    );
    assert_eq!(
        summary_value(&summary, "completion_tokens_total"),
        (123 * usages.len()).to_string(),
        "the requested run-level completion total must equal the per-call sum: {summary}"
    );

    let mut per_site_calls = 0_u64;
    for (site, _, _) in vanta::llm::SAMPLING_SCHEDULE {
        let calls: u64 = summary_value(&summary, &format!("llm_calls_site_{site}"))
            .parse()
            .unwrap_or_else(|e| panic!("`llm_calls_site_{site}` must be a count ({e}): {summary}"));
        // A site this run never reached is recorded as a ZERO, not omitted: a
        // breakdown that silently drops a site makes "it never ran" and "the
        // record forgot it" the same reading.
        assert_eq!(
            summary_value(&summary, &format!("truncated_calls_site_{site}")),
            "0",
            "nothing was cut off: {summary}"
        );
        // `unset` is the honest value for a site whose completions were never
        // measured (an `estimate`-sourced row has no completion count at all).
        let completions = summary_value(&summary, &format!("completion_tokens_site_{site}"));
        assert!(
            completions == "unset" || completions.parse::<u64>().is_ok(),
            "`completion_tokens_site_{site}` must be a count or `unset`: {summary}"
        );
        per_site_calls += calls;
    }
    assert_eq!(
        per_site_calls,
        usages.len() as u64,
        "the per-site breakdown must account for every call the run made, or a site \
         is missing from the report: {summary}"
    );
}

/// §14.29 — the four usage facts a cost table cannot derive from the flat counts,
/// per call and totalled on `run.summary`.
///
/// [VANTA-WHY] What is asserted here is: how much of the prompt the provider served
/// from its cache, how much of the completion was reasoning, what the call cost, and
/// — the part that makes a total trustworthy — which price row produced the number.
/// The stub publishes the NESTED shape the hosted providers use
/// (`prompt_tokens_details.cached_tokens` / `completion_tokens_details.reasoning_tokens`),
/// so a parser that only read the top-level counts answers `unset` here and fails.
///
/// The same run is made twice, on the backend that bills and the one that does not,
/// because two different claims have to stay apart: "the details were recorded" (both)
/// and "the call was priced" (only the hosted one). A local run is `not_billed` with
/// `cost_usd=unset` — never `0.000000`, which is the different claim that the call WAS
/// priced and came to nothing.
#[test]
fn usage_details_and_cost_are_recorded_per_call_and_totalled() {
    let _serial = serial_run();

    // (1) Local: the details are recorded, the call is not billed.
    let local = TempDir::new().unwrap();
    let (_repo, _issue, local_patch) = write_test_repo(local.path(), PRESERVING_PATCH);
    let local_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let local_status = run_patch_mode(local.path(), &local_patch, &local_server);
    assert!(local_status.success(), "the reference run must reach a verdict");

    let local_records = read_session_records(&find_session_log(local.path()));
    let local_usages: Vec<&SessionRecord> = local_records
        .iter()
        .filter(|record| record.phase == "call.usage")
        .collect();
    let calls = local_usages.len();
    assert!(calls > 0, "a run that called the model must write `call.usage` rows");
    for row in &local_usages {
        assert_eq!(
            summary_value(&row.summary, "cached_tokens"),
            "314",
            "the nested `prompt_tokens_details.cached_tokens` must be recorded: {}",
            row.summary
        );
        assert_eq!(
            summary_value(&row.summary, "reasoning_tokens"),
            "41",
            "the nested `completion_tokens_details.reasoning_tokens` must be recorded: {}",
            row.summary
        );
        assert_eq!(
            summary_value(&row.summary, "cost_usd"),
            "unset",
            "a call nobody billed has no cost — `unset`, not `0.000000`: {}",
            row.summary
        );
        assert_eq!(
            summary_value(&row.summary, "cost_basis"),
            "not_billed",
            "the row must say WHY there is no cost: {}",
            row.summary
        );
    }
    let local_summary = run_summary(local.path()).summary;
    assert_eq!(
        summary_value(&local_summary, "llm_cached_tokens"),
        (314 * calls).to_string(),
        "the run total must sum the server's cached counts: {local_summary}"
    );
    assert_eq!(
        summary_value(&local_summary, "llm_reasoning_tokens"),
        (41 * calls).to_string(),
        "the run total must sum the server's reasoning counts: {local_summary}"
    );
    assert_eq!(summary_value(&local_summary, "cost_usd"), "unset");
    assert_eq!(summary_value(&local_summary, "cost_basis"), "not_billed");

    // (2) Hosted: same stub, same numbers — now priced.
    let hosted = TempDir::new().unwrap();
    let (_repo, _issue, hosted_patch) = write_test_repo(hosted.path(), PRESERVING_PATCH);
    let hosted_server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    let hosted_output = run_patch_mode_with_output(
        hosted.path(),
        &hosted_patch,
        &hosted_server,
        3,
        &[
            ("VANTA_LLM_BACKEND", "openai"),
            ("VANTA_LLM_MODEL", "gpt-6-luna"),
            ("OPENAI_API_KEY", "sk-test-not-a-real-key"),
            ("VANTA_LLM_CONTEXT_TOKENS", "131072"),
        ],
    );
    assert_eq!(
        hosted_output.status.code(),
        Some(0),
        "pricing a call must not change the verdict — only what is recorded; stdout:\n{}",
        String::from_utf8_lossy(&hosted_output.stdout)
    );

    let hosted_usages: Vec<String> = read_session_records(&find_session_log(hosted.path()))
        .into_iter()
        .filter(|record| record.phase == "call.usage")
        .map(|record| record.summary)
        .collect();
    assert_eq!(
        hosted_usages.len(),
        calls,
        "the backend must not change how many calls the run makes"
    );
    for row in &hosted_usages {
        assert_eq!(summary_value(row, "cached_tokens"), "314");
        assert_eq!(summary_value(row, "reasoning_tokens"), "41");
        // 777 prompt of which 314 were cached, 123 out, at Luna's standard rates:
        //   463 uncached × $0.10 + 314 cached × $0.01 + 123 out × $0.50, per Mtok
        //   = $0.00011094, which rounds to six decimals as below.
        // The literal is the whole point of the assertion: billing the cached 314 at
        // the FULL input rate would give `0.000139`, so a price table that ignored
        // the cache cannot produce this string.
        assert_eq!(
            summary_value(row, "cost_usd"),
            "0.000111",
            "the cached part of the prompt must be billed at the cached rate: {row}"
        );
        assert_eq!(
            summary_value(row, "cost_basis"),
            "gpt-6-luna/standard/2026-10-26",
            "the row must name the price row that produced the number: {row}"
        );
    }
    let hosted_summary = run_summary(hosted.path()).summary;
    assert_eq!(
        summary_value(&hosted_summary, "llm_cached_tokens"),
        (314 * hosted_usages.len()).to_string()
    );
    assert_eq!(
        summary_value(&hosted_summary, "llm_reasoning_tokens"),
        (41 * hosted_usages.len()).to_string()
    );
    assert_eq!(
        summary_value(&hosted_summary, "cost_usd"),
        format!("{:.6}", 0.00011094 * hosted_usages.len() as f64),
        "the run total must sum the calls it priced: {hosted_summary}"
    );
    assert_eq!(
        summary_value(&hosted_summary, "cost_basis"),
        "gpt-6-luna/standard/2026-10-26"
    );
}

/// §14.30 — a provider that refuses a request because it is busy is re-asked the
/// SAME request, and that refusal is NOT charged to the model's attempt budget.
///
/// [VANTA-WHY] What is asserted here is the whole reason the provider got its own
/// ladder: the run reaches the same verdict, the `creator` call succeeds on its
/// FIRST model attempt (`attempt=1`) while recording `provider_retries=1`, and the
/// run total names the refusal's class. A design that folded a 429 into `attempt`
/// would show `attempt=2` on a call the model answered correctly the first time it
/// was asked — a throttled run disguised as a model that needed two tries, which
/// is the one reading this test exists to make impossible. The stub's body says
/// `Resource Unavailable` (the `flex` capacity refusal), so the class asserted is
/// `resource_unavailable` and a caller that reported everything as a rate limit
/// fails here.
#[test]
fn a_busy_provider_is_re_asked_without_spending_a_model_attempt() {
    let _serial = serial_run();

    let workspace = TempDir::new().unwrap();
    let (_repo, _issue, patch) = write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);
    // The FIRST Attacker generation request is refused; the re-send is answered
    // normally. (Patch mode never calls the Creator, so the generation is the
    // run's first chat call.)
    server.refuse_next_calls(&["generate"]);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch,
        &server,
        3,
        &[
            ("VANTA_LLM_BACKEND", "openai"),
            ("VANTA_LLM_MODEL", "gpt-6-luna"),
            ("OPENAI_API_KEY", "sk-test-not-a-real-key"),
            ("VANTA_LLM_CONTEXT_TOKENS", "131072"),
        ],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "a throttle the caller rode out must not change the verdict; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let kinds = server.request_kinds();
    let generate_requests = kinds.iter().filter(|kind| kind.as_str() == "generate").count();
    let records = read_session_records(&find_session_log(workspace.path()));
    let attacker_rows: Vec<&SessionRecord> = records
        .iter()
        .filter(|record| {
            record.phase == "call.usage"
                && summary_value(&record.summary, "site") == "attacker_generate"
        })
        .collect();
    assert!(
        !attacker_rows.is_empty(),
        "the run must have generated tests; request kinds: {kinds:?}"
    );
    // The refusal was seen as a REQUEST — the stub answered nothing to it — so the
    // generation appears one time MORE often than the model was asked about it.
    assert_eq!(
        generate_requests,
        attacker_rows.len() + 1,
        "the refused request must be re-sent and nothing else added; request kinds: {kinds:?}"
    );

    let throttled = attacker_rows
        .iter()
        .filter(|record| summary_value(&record.summary, "provider_retries") == "1")
        .count();
    assert_eq!(
        throttled, 1,
        "exactly one call must record the provider's refusal; request kinds: {kinds:?}"
    );
    for record in &attacker_rows {
        assert_eq!(
            summary_value(&record.summary, "attempt"),
            "1",
            "the throttle must not be charged to the model's attempt budget: {}",
            record.summary
        );
    }

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(
        summary_value(&summary, "provider_retries"),
        "1",
        "the run total must count the provider's refusal: {summary}"
    );
    assert_eq!(
        summary_value(&summary, "provider_retries_unavailable"),
        "1",
        "the `Resource Unavailable` body is the flex class, not the generic rate limit: {summary}"
    );
    assert_eq!(
        summary_value(&summary, "provider_retries_rate_limit"),
        "0",
        "a capacity refusal must not be reported as a rate limit: {summary}"
    );
}

// ============================================================================
// §14.8 — corroboration, grounded audits and needs-review, END TO END
// ============================================================================
//
// [VANTA-WHY] `corroborate.rs` and `attacker.rs` unit-test the RULES. What only
// the binary can prove is that the switch reaches the driver, that a downgraded
// rejection keeps the loop alive instead of stopping it, that the terminal
// decision and the process exit code follow (§14.3), and that `run.summary`
// names the rule that decided and the weak failures that did not — the record a
// human needs in order to know WHAT to review. With the flag off none of it may
// be reachable: the v0.2 verdict is unchanged, which is the compat half of the
// contract.

/// §14.2 rule (d): the ONLY evidence against the candidate is a
/// `BehaviorPreservation` probe pinning behaviour the issue says to preserve, so
/// the issue itself authorises the rejection. This is the toy fixture's bad-patch
/// shape, and it is the guard that corroboration did not go soft.
#[test]
fn a_validated_preservation_failure_is_corroborated_by_rule_d() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // One preservation probe: it passes on the original tree and fails on the bad
    // patch, and it is the whole of the evidence against the patch.
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_CORROBORATE", "on")],
    );

    assert_eq!(
        output.status.code(),
        Some(1),
        "a corroborated rejection is a rejection, not a needs-review: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("CORROBORATED by rule (d)"),
        "the console must name the rule that decided; got {stdout}"
    );
    assert!(
        stdout.contains("test_behavior_preservation_only@1"),
        "the corroboration line must name the evidence that decided; got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("final_decision=Rejected"),
        "got {}",
        final_row.summary
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(summary_value(&summary, "corroborate"), "on");
    assert_eq!(summary_value(&summary, "deciding_rule"), "d");
    assert_eq!(
        summary_value(&summary, "deciding_evidence"),
        "[test_behavior_preservation_only@1]"
    );
    assert_eq!(
        summary_value(&summary, "needs_review_weak"),
        "[]",
        "a corroborated rejection has nothing weak to review; got {summary}"
    );
}

/// §14.3/§14.8: failures NO rule covers must reach a human, and "a human should
/// look" has to be visible on the process boundary (exit code 3) and in
/// `run.summary` — a harness that read it as a rejection would count an
/// uncorroborated failure as a false reject, which is the mislabel this change
/// exists to remove.
#[test]
fn a_lone_uncorroborated_failure_ends_as_needs_review_with_exit_code_3() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // ONE BugReproduction probe, which fails on the patched tree as Logic. Nothing
    // corroborates it: `TEST_ISSUE_TEXT` names no exception class, so rule (b)
    // cannot fire; a single test can never satisfy rule (c); and (d)/(e) do not
    // apply. This is PySnooper/3's dev-fix shape, the false reject §14 removed.
    let server = StubLlmServer::start(SINGLE_BUG_REPRODUCTION_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_CORROBORATE", "on")],
    );

    assert_eq!(
        output.status.code(),
        Some(3),
        "an uncorroborated failure is neither an approval nor a rejection: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("UNCORROBORATED failure(s)"),
        "the console must say the failure was uncorroborated; got {stdout}"
    );
    assert!(
        !stdout.contains("CORROBORATED by rule"),
        "no rule may fire on this evidence, so nothing may be reported as \
         corroborated; got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("final_decision=NeedsReview"),
        "the budget ran out on an uncorroborated failure, so the honest terminal \
         decision is NeedsReview, not Rejected or MaxAttemptsReached; got {}",
        final_row.summary
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(summary_value(&summary, "corroborate"), "on");
    assert_eq!(summary_value(&summary, "deciding_rule"), "none");
    assert_eq!(summary_value(&summary, "deciding_evidence"), "[]");
    assert_eq!(
        summary_value(&summary, "needs_review_weak"),
        "[test_bug_reproduction_only=Logic/BugReproduction@1]",
        "the summary must name the weak failures, or `NeedsReview` tells the human \
         nothing to look at; got {summary}"
    );

    // §14.3 — NeedsReview attaches the candidate exactly as a rejection does, so
    // the patch the reviewer has to look at is on disk.
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    assert!(
        debug_dir.join("final.patch").exists(),
        "a NeedsReview must leave the candidate patch for the reviewer; missing under {}",
        debug_dir.display()
    );
}

/// §14.15 — a run that exhausts its OWN clock ends as a recorded verdict.
///
/// [VANTA-WHY] The overnight batch lost sanic/1 to the harness's 3 h SIGKILL: the
/// process died between the terminal decision and the write of `run.summary`, so
/// the row was `no run.summary` — no verdict, no cost, no prompt provenance. This
/// run arms `VANTA_DEADLINE_SECS=0`, which makes the clock already past when the
/// attempt loop begins, so Vanta must refuse to start any call, stop BETWEEN
/// calls, and end as `NeedsReview` (exit 3) with `stop_reason=budget_exhausted` on
/// disk. The distinction the row has to carry is the whole point of the item:
/// `timed_out` says the run was killed before it could say anything, and this says
/// the run had something to say and said it.
#[test]
fn a_run_that_exhausts_its_own_clock_ends_as_needs_review_and_says_why() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_DEADLINE_SECS", "0")],
    );

    assert_eq!(
        output.status.code(),
        Some(3),
        "a run out of clock is a NeedsReview — not an InfraFailure (2, the machine \
         broke) and not a rejection (1, the candidate is wrong): {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        server.request_count(),
        0,
        "the deadline was already past when the loop began, so NO call may have been \
         made: refusing to start is the entire point of checking before every call"
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(
        summary_value(&summary, "stop_reason"),
        "budget_exhausted",
        "run.summary must say the run stopped itself; got {summary}"
    );
    assert_eq!(
        summary_value(&summary, "deadline_secs"),
        "0",
        "the summary must record the deadline it was measured against; got {summary}"
    );
    assert!(
        summary.contains("final_decision=NeedsReview"),
        "the deadline is a fact about the run, not the candidate, so the verdict is \
         NeedsReview; got {summary}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let budget_row = records
        .iter()
        .find(|r| r.phase == "run.budget_exhausted")
        .unwrap_or_else(|| {
            panic!(
                "the run must explain why its own clock ended it, in its own row — \
                 `NeedsReview` plus an empty `needs_review_weak` is indistinguishable \
                 from an ordinary uncorroborated failure; phases: {:?}",
                records.iter().map(|r| r.phase.as_str()).collect::<Vec<_>>()
            )
        });
    assert!(
        budget_row.summary.contains("stop_reason=budget_exhausted")
            && budget_row.summary.contains("final_decision=NeedsReview"),
        "the row must carry both the verdict and the reason; got {}",
        budget_row.summary
    );
}

/// §14.2/§14.8 byte-compat: with the flag off the SAME fixture takes v0.2's path
/// untouched — the rejection stands, nothing weak is recorded and no rule is
/// named, because no rule ran.
#[test]
fn with_the_corroboration_flag_off_an_uncorroborated_failure_is_still_a_rejection() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start(SINGLE_BUG_REPRODUCTION_TEST);

    let output = run_patch_mode_with_output(workspace.path(), &patch_path, &server, 1, &[]);

    assert_eq!(
        output.status.code(),
        Some(1),
        "with the switch off there is no needs-review path at all: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("final_decision=Rejected"),
        "the v0.2 verdict is a rejection; got {}",
        final_row.summary
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(summary_value(&summary, "corroborate"), "off");
    assert_eq!(summary_value(&summary, "deciding_rule"), "none");
    assert_eq!(summary_value(&summary, "needs_review_weak"), "[]");
}

/// §14.2 rule (a) is deterministic — the repository's OWN suite cannot be an
/// LLM's bad test — so it must stand with corroboration on, and it must win over
/// every weaker signal.
#[test]
fn a_native_suite_regression_is_corroborated_by_rule_a() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    // The generated probe only asserts on `add`; the patch only breaks `scale`.
    // So the repository's own test is the only evidence against this patch.
    let (_repo_dir, _issue_path, patch_path) = write_test_repo_with(
        workspace.path(),
        NATIVE_REPO_SOURCE,
        NATIVE_BREAKING_PATCH,
        Some(NATIVE_REPO_TEST),
    );
    let server = StubLlmServer::start(SINGLE_BEHAVIOR_TEST);

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_CORROBORATE", "on")],
    );

    assert_eq!(
        output.status.code(),
        Some(1),
        "a regression in the project's own suite must still be rejected: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("CORROBORATED by rule (a)"),
        "got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("final_decision=Rejected"),
        "got {}",
        final_row.summary
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(summary_value(&summary, "deciding_rule"), "a");
    assert!(
        summary_value(&summary, "deciding_evidence").contains("test_scale_doubles"),
        "the rule's evidence must name the broken repo test; got {summary}"
    );
    assert_eq!(summary_value(&summary, "needs_review_weak"), "[]");
}

/// §14.4 end to end: an ungrounded answer scraps the `BugReproduction` probe it
/// was asked about, is logged with the auditor's own reason on the console (the
/// line that makes over-scrapping visible while the dev split is still in front
/// of you), and NEVER scraps the `BehaviorPreservation` probe that received the
/// very same answer.
#[test]
fn the_grounded_audit_scraps_an_ungrounded_reproduction_probe_and_never_a_preservation_one() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    // ONE answer, served to BOTH tests: the second record is what makes this the
    // mirror rather than just a scrap test.
    let server = StubLlmServer::start_with_intent_audit(
        UNGROUNDED_AND_PRESERVING_TESTS,
        r#"{"issue_asks_for_behaviour":true,"assertions_grounded":false,"reason":"stub: pins a value the issue never states"}"#,
    );

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_GROUNDED_AUDIT", "on")],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("UNGROUNDED: test_bug_reproduction_pins_invented_text scrapped"),
        "every ungrounded scrap must be visible on the console; got {stdout}"
    );
    assert!(
        stdout.contains("pins a value the issue never states"),
        "the console line must carry the auditor's reason, not a generic one; got {stdout}"
    );

    let records = read_session_records(&find_session_log(workspace.path()));

    let reproduction = records
        .iter()
        .find(|r| r.phase == "intent.check[test_bug_reproduction_pins_invented_text]")
        .expect("the reproduction probe must have an intent.check row");
    assert!(
        reproduction.summary.contains("assertions_grounded=false")
            && reproduction.summary.contains("scrapped=true")
            && reproduction.summary.contains("source=attacker.generate"),
        "the reproduction probe must be scrapped on the ungrounded answer; got {}",
        reproduction.summary
    );

    let preservation = records
        .iter()
        .find(|r| r.phase == "intent.check[test_preserve_add_pair]")
        .expect("the preservation probe must be audited too, not skipped by its label");
    assert!(
        preservation.summary.contains("assertions_grounded=false"),
        "the same answer must reach the preservation probe, or the mirror below is \
         vacuous; got {}",
        preservation.summary
    );
    assert!(
        preservation.summary.contains("scrapped=false"),
        "an ungrounded answer may never scrap a BehaviorPreservation probe; got {}",
        preservation.summary
    );

    let attacker_summary = records
        .iter()
        .find(|r| r.phase == "attacker.summary")
        .expect("the Attacker summary row must exist");
    assert!(
        attacker_summary.summary.contains("intent_scrapped=1")
            && attacker_summary.summary.contains("valid_tests=1"),
        "exactly one test is scrapped and the round keeps the other; got {}",
        attacker_summary.summary
    );
}

/// §14.8's positive guard: with all three flags on, a correct patch is still
/// APPROVED. Corroboration exists to remove false rejects, so the failure it
/// must not introduce is a false `NeedsReview` — an approval turned into
/// "a human should look" would be the same wrong label, one step safer.
#[test]
fn the_three_flags_on_do_not_turn_an_approval_into_a_needs_review() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    // The batched transport needs a batched answer; the single-test shape would
    // fail the whole batch and take the unavailable-audit path instead.
    let server = StubLlmServer::start_with_intent_audit(
        SINGLE_BEHAVIOR_TEST,
        r#"{"tests":[{"index":1,"issue_asks_for_behaviour":true,"assertions_grounded":true,"reason":"stub: keep the label"}]}"#,
    );

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[
            ("VANTA_CORROBORATE", "on"),
            ("VANTA_GROUNDED_AUDIT", "on"),
            ("VANTA_BATCH_AUDIT", "on"),
        ],
    );

    assert_eq!(
        output.status.code(),
        Some(0),
        "a correct patch must still be approved with every switch on: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let final_row = records
        .iter()
        .find(|r| r.phase == "M3_repair_loop")
        .expect("a final decision row must exist");
    assert!(
        final_row.summary.contains("final_decision=AllTestsPassed"),
        "got {}",
        final_row.summary
    );

    let summary = run_summary(workspace.path()).summary;
    assert_eq!(summary_value(&summary, "corroborate"), "on");
    assert_eq!(summary_value(&summary, "grounded_audit"), "on");
    assert_eq!(summary_value(&summary, "batch_audit"), "on");
    assert_eq!(summary_value(&summary, "deciding_rule"), "none");
    assert_eq!(
        summary_value(&summary, "needs_review_weak"),
        "[]",
        "an approval carries no weak failures; got {summary}"
    );
}

// ============================================================================
// Hosted OpenAI backend — §14.25: the API key reaches no file and no stream
// ============================================================================

/// Every regular file under `root`, depth-first and sorted, so a failure names a
/// deterministic path and the sweep is reproducible.
///
/// [VANTA-WHY] The redaction tests below must sweep the WHOLE tree rather than a
/// list of files written today. A list is a promise to update it whenever a new
/// artifact appears, and the failure of that promise is silent: a run would
/// start leaking through the file nobody remembered to add. Walking the tree
/// makes "written by the run" the criterion, not "known to the test".
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Whether `needle` occurs anywhere in `haystack`.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

/// The API key must not reach any file a run writes, nor stdout, nor stderr —
/// even when the provider ECHOES it back inside a rejected request's body.
///
/// [VANTA-WHY] A provider is free to quote the credential it just rejected back
/// at the client ("Incorrect API key provided: sk-…"), and Vanta's own error
/// path prints the provider's body to make the failure legible. So the leak this
/// test hunts is not a careless `println!` of a variable — it is a *legitimate*
/// message that happens to contain the secret, reaching the session log, the
/// evidence bundle and the terminal. Redacting the variable NAME would not catch
/// it; only redacting the VALUE does. The stub does the echoing for real (it
/// reads the bearer token out of the request headers), so the run exercises the
/// whole path rather than a hand-built error string.
#[test]
fn the_api_key_never_reaches_a_file_stream_or_record_when_the_provider_echoes_it() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), PRESERVING_PATCH);
    let server = StubLlmServer::start_streaming(
        None,
        SINGLE_BEHAVIOR_TEST,
        StubBehaviour::EchoKeyUnauthorized,
    );

    let output = run_patch_mode_with_output(
        workspace.path(),
        &patch_path,
        &server,
        1,
        // One attempt per call: the rejection is deterministic, so a retry adds
        // nothing but time, and the run must reach its recorded failure fast.
        &[("VANTA_LLM_RETRY_ATTEMPTS", "1")],
    );

    assert_eq!(
        output.status.code(),
        Some(2),
        "a rejected key is an infrastructure failure, not a verdict; stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        server.request_count() >= 1,
        "the run must actually have made a chat call for the echo path to be \
         exercised; got {}",
        server.request_count()
    );

    let key = TEST_API_KEY.as_bytes();

    // (1) the two streams the run owns.
    for (name, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        assert!(
            !contains_bytes(bytes, key),
            "the API key reached {name}:\n{}",
            String::from_utf8_lossy(bytes)
        );
    }

    // (2) every file the run wrote, whatever its name. The workspace is the
    // run's CWD, so this covers the session log, the transitions store, the
    // evidence bundle and any sandbox scratch files alike.
    let files = files_under(workspace.path());
    assert!(
        !files.is_empty(),
        "the sweep must not be vacuous — a run that wrote nothing would pass it \
         for the wrong reason"
    );
    for path in &files {
        let bytes = fs::read(path).unwrap();
        assert!(
            !contains_bytes(&bytes, key),
            "the API key reached {}:\n{}",
            path.display(),
            String::from_utf8_lossy(&bytes)
        );
    }

    // (3) the positive control. Without it the test would also pass if the echo
    // path never ran — the assertion would be checking a message that was never
    // built. The recorded failure must show the MARKER in the key's place and
    // still name the status, so a reader can see the rejection happened.
    let records = read_session_records(&find_session_log(workspace.path()));
    let infra = records
        .iter()
        .find(|r| r.phase == "run.infra_failure")
        .expect("a rejected key must be recorded as a run.infra_failure row");
    assert!(
        infra.summary.contains("401"),
        "the recorded failure must name the status the provider returned; got {}",
        infra.summary
    );
    assert!(
        infra.summary.contains("<redacted>"),
        "the echoed credential must have been replaced by the marker, proving \
         the message was built and redacted rather than never built; got {}",
        infra.summary
    );
}
