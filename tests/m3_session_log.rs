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
use vanta::core::{EcrEirSnapshot, FailureClass, TestIntent, TestResult};
use vanta::session::{PatchAttemptId, SessionId, SessionRecord, TransitionRow, ecr_eir_from_rows};

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

struct StubLlmServer {
    base_url: String,
    request_count: Arc<AtomicUsize>,
    request_kinds: Arc<Mutex<Vec<String>>>,
    request_bodies: Arc<Mutex<Vec<String>>>,
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://127.0.0.1:{}/v1", addr.port());
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_kinds = Arc::new(Mutex::new(Vec::new()));
        let request_bodies = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let request_count_clone = Arc::clone(&request_count);
        let request_kinds_clone = Arc::clone(&request_kinds);
        let request_bodies_clone = Arc::clone(&request_bodies);
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
            while !stop_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _addr)) => {
                        let index = request_count_clone.fetch_add(1, Ordering::SeqCst);
                        if let Some(hook) = &on_request {
                            hook(index);
                        }
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
                            creator_value
                                .clone()
                                .unwrap_or_else(|| serde_json::json!({"patch_text":"def add(a, b):\n    return a + b\n","explanation":"noop","files_touched":["calc.py"],"confidence":0.1}))
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
                        // The assistant `content` field is a JSON STRING holding
                        // the model's answer, so the serialised value is quoted
                        // once more before it is embedded in the envelope.
                        let content_text = content.to_string();
                        let response_body = format!(
                            "{{\"choices\":[{{\"message\":{{\"content\":{},\"reasoning_content\":\"\"}}}}],\"usage\":{{\"total_tokens\":1}}}}",
                            serde_json::to_string(&content_text).unwrap()
                        );

                        let response = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            response_body.len(),
                            response_body
                        );
                        let _ = stream.write_all(response.as_bytes());
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
            stop,
            join: Some(join),
        }
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
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

const CREATOR_SYNTAX_ERROR_RESPONSE: &str = r#"{
    "patch_text":"def add(a, b):\n    return a + b +\n",
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
        .env("VANTA_LLM_TIMEOUT_SECS", "5")
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
        .env("VANTA_LLM_TIMEOUT_SECS", "5")
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
    assert_eq!(records.len(), 25);
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
        final_row.summary.contains("MaxAttemptsReached"),
        "a real syntax error must not trip the harness-collapse ladder; got {}",
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
    assert!(
        !records.iter().any(|r| r.phase == "M3_repair_loop"),
        "the run aborted before end-of-run, so no final-decision record may exist"
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
    let server = StubLlmServer::start(CONTRADICTORY_INTENT_TEST);

    let status = run_patch_mode_with(workspace.path(), &patch_path, &server, 1, &[]);
    assert!(
        !status.success(),
        "a run whose only test is scrapped must not report success"
    );

    // [VANTA-WHY] Each non-preservation test is audited by the patch-blind
    // intent check: once for the initial batch and once per revision. With one
    // BugReproduction test that is 1 generation + 1 audit + 3 (revise + audit)
    // = 8 LLM calls. Scrapping on the first mismatch — the regressed behaviour —
    // would leave this at 2 (the generation and its audit).
    assert_eq!(
        server.request_count(),
        8,
        "a test whose baseline contradicts its intent must be sent back to the \
         Attacker for revision (up to 3 times) before being scrapped; \
         expected 1 generation + 1 batch audit + 3 (revision + revised audit) = 8 LLM calls"
    );
    assert_eq!(
        server.request_kinds(),
        vec![
            "generate",
            "intent_audit",
            "revise",
            "intent_audit",
            "revise",
            "intent_audit",
            "revise",
            "intent_audit",
        ],
        "the regression guard must prove the exact call sequence — one generate, \
         then a patch-blind audit of it, then three genuine revision/audit pairs — \
         so internal parse retries cannot masquerade as a working revision path"
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
