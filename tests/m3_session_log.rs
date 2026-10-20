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
}

struct StubLlmServer {
    base_url: String,
    request_count: Arc<AtomicUsize>,
    request_kinds: Arc<Mutex<Vec<String>>>,
    request_bodies: Arc<Mutex<Vec<String>>>,
    /// Number of stalled chat connections the client closed on us. It is how a
    /// test proves the cancel reached the SERVER (not just the client's future).
    disconnects: Arc<AtomicUsize>,
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
        let stop = Arc::new(AtomicBool::new(false));

        let request_count_clone = Arc::clone(&request_count);
        let request_kinds_clone = Arc::clone(&request_kinds);
        let request_bodies_clone = Arc::clone(&request_bodies);
        let disconnects_clone = Arc::clone(&disconnects);
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
                        let content_text = content.to_string();

                        if matches!(behaviour, StubBehaviour::Fail) {
                            let response = "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
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

                        let tail = format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            serde_json::json!({
                                "choices": [{"delta": {}}],
                                // [VANTA-WHY] 777/123/900, not 1/1/2. A token
                                // count a local estimate could plausibly have
                                // produced would let a test that reads the
                                // estimate pass for a test that reads the
                                // server; these numbers cannot come from
                                // anywhere but this frame. `completion_tokens`
                                // is the field a streamed response omits unless
                                // `stream_options.include_usage` was asked for.
                                "usage": {
                                    "prompt_tokens": 777,
                                    "completion_tokens": 123,
                                    "total_tokens": 900
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
            stop,
            join: Some(join),
        }
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
    // generated probe. Plus the run-level session.start, run.summary and
    // M3_repair_loop records.
    //
    // [VANTA-WHY] Counted over the ATTEMPT-STRUCTURE rows only. Token accounting
    // writes one `call.usage` row per chat call, and how many calls a round makes
    // depends on retries and BATE revisions — pinning the total here would make a
    // test about the repair loop's record shape fail for a reason outside it.
    // Those rows have their own test
    // (`token_accounting_records_the_servers_counts_per_call_and_a_run_total`).
    let structure_rows: Vec<_> = records.iter().filter(|r| r.phase != "call.usage").collect();
    assert_eq!(structure_rows.len(), 25);
    assert!(
        records.iter().any(|r| r.phase == "call.usage"),
        "this run called the model, so it must record token usage"
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
fn assert_schedule_matches_bodies(workspace: &Path, server: &StubLlmServer) -> Vec<String> {
    let summary = run_summary(workspace).summary;

    for (site, temperature, max_tokens) in vanta::llm::SAMPLING_SCHEDULE {
        assert_eq!(
            summary_value(&summary, &format!("temperature_{site}")),
            format!("{temperature:?}"),
            "run.summary must record the pinned temperature for `{site}`"
        );
        assert_eq!(
            summary_value(&summary, &format!("max_tokens_{site}")),
            max_tokens.to_string(),
            "run.summary must record the pinned max_tokens for `{site}`"
        );
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
        let body_temperature = value
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| panic!("the `{kind}` body carries no temperature: {body}"));
        let body_max_tokens = value
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("the `{kind}` body carries no max_tokens: {body}"));
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
        assert_eq!(
            summary_value(&summary, &format!("temperature_{site}")),
            format!("{body_temperature:?}"),
            "the recorded schedule disagrees with the `{kind}` body actually sent"
        );
        assert_eq!(
            summary_value(&summary, &format!("max_tokens_{site}")),
            body_max_tokens.to_string(),
            "the recorded schedule disagrees with the `{kind}` body actually sent"
        );
        checked.push(site.to_string());
    }
    assert!(
        !checked.is_empty(),
        "no chat body was sent, so the schedule was never cross-checked"
    );
    checked
}

/// `VANTA_LLM_SEED` must be invisible when unset and present in every chat body
/// when set.
///
/// [VANTA-WHY] An unset seed has to leave the request body byte-for-byte what it
/// was before the field existed: `"seed": null` would change what every server
/// sees, and `"seed": 0` would move the sampling baseline every existing
/// measurement was taken under. A test cannot run the pre-seed binary, so the
/// guard is twofold — no key at all when unset, exactly that key when set, and
/// the two bodies equal once the key is removed.
#[test]
fn unset_seed_leaves_request_bodies_unseeded_and_the_verdict_unchanged() {
    let _serial = serial_run();

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
    for body in &seeded_bodies {
        assert!(
            body.contains("\"seed\":4242"),
            "a set VANTA_LLM_SEED must reach every chat body; got: {body}"
        );
    }
    // The ONLY difference the new field may introduce is the key itself.
    for (unseeded, seeded) in plain_bodies.iter().zip(seeded_bodies.iter()) {
        assert_eq!(
            body_without_seed(seeded),
            body_without_seed(unseeded),
            "with the seed key removed, a seeded body must be byte-identical to the unseeded one"
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
    let checked = assert_schedule_matches_bodies(review.path(), &review_server);
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
    let checked = assert_schedule_matches_bodies(repair.path(), &repair_server);
    for required in ["creator", "attacker_generate", "intent_audit"] {
        assert!(
            checked.iter().any(|site| site == required),
            "repair mode must exercise `{required}`; checked={checked:?}"
        );
    }
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
}

