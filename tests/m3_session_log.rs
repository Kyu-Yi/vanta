use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use tempfile::TempDir;
use vanta::session::{SessionRecord, TransitionRow};

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
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl StubLlmServer {
    fn start(attacker_response_json: &str) -> Self {
        Self::start_full(attacker_response_json, None, None)
    }

    /// `on_request` runs with the 0-based index of each request, before the
    /// response is written. It lets a test change the world between rounds —
    /// e.g. replace the --patch file so the next attempt verifies a different
    /// candidate.
    fn start_with_hook(
        attacker_response_json: &str,
        on_request: Option<Box<dyn Fn(usize) + Send>>,
    ) -> Self {
        Self::start_full(attacker_response_json, on_request, None)
    }

    /// Drop the TCP connection (no HTTP response) for every request whose index
    /// is >= `abort_at`. The client sees a transport error and the harness dies
    /// mid-run — a stand-in for a `HarnessFailure` abort, which happens before
    /// the end-of-run session log is written.
    fn start_aborting(attacker_response_json: &str, abort_at: usize) -> Self {
        Self::start_full(attacker_response_json, None, Some(abort_at))
    }

    fn start_full(
        attacker_response_json: &str,
        on_request: Option<Box<dyn Fn(usize) + Send>>,
        abort_at: Option<usize>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://127.0.0.1:{}/v1", addr.port());
        let request_count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let request_count_clone = Arc::clone(&request_count);
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
                                                if let Some(value) = line
                                                    .strip_prefix("Content-Length:")
                                                    .or_else(|| line.strip_prefix("content-length:"))
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
                        let content = if body.contains("write adversarial tests") {
                            attacker_value.clone()
                        } else {
                            attacker_value
                                .get("tests")
                                .and_then(|tests| tests.get(0))
                                .cloned()
                                .unwrap_or_else(|| attacker_value.clone())
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
            stop,
            join: Some(join),
        }
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }
}

impl Drop for StubLlmServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(self.base_url.trim_start_matches("http://").split('/').next().unwrap());
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

const THREE_SIGNAL_TESTS: &str = r#"{"tests":[
        {
            "test_name":"test_bug_regression_signal",
            "test_source":"from calc import add\n\ndef test_bug_regression_signal():\n    assert add(2, 3) == 6\n",
            "intent":"BugReproduction",
            "reason":"Fails on the original baseline, so BATE admits it."
        },
        {
            "test_name":"test_feature_acceptance_signal",
            "test_source":"from calc import add\n\ndef test_feature_acceptance_signal():\n    assert add(2, 3) == 7\n",
            "intent":"FeatureAcceptance",
            "reason":"Also fails on the original baseline, so BATE admits it."
        },
        {
            "test_name":"test_behavior_preservation_signal",
            "test_source":"from calc import add\n\ndef test_behavior_preservation_signal():\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original baseline and fails on the bad patch."
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
            "reason":"Declares a bug that the original tree does not have."
        }
    ]}"#;

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
            "test_source":"import time\nfrom calc import add\n\ndef test_hangs_on_patched_tree():\n    if add(2, 3) == 6:\n        while True:\n            time.sleep(0.5)\n    assert add(2, 3) == 5\n",
            "intent":"BehaviorPreservation",
            "reason":"Passes on the original tree and hangs on the patched one."
        }
    ]}"#;

fn write_test_repo(root: &Path, patch_text: &str) -> (PathBuf, PathBuf, PathBuf) {
    let repo_dir = root.join("repo");
    fs::create_dir_all(&repo_dir).unwrap();
    fs::write(
        repo_dir.join("calc.py"),
        "def add(a, b):\n    return a + b\n",
    )
    .unwrap();
    let issue_path = repo_dir.join("issue.md");
    fs::write(&issue_path, "Verify a candidate patch for calc.add.\n").unwrap();

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
    assert!(output.status.success(), "python3 must be available for the integration test");
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
fn run_patch_mode(workspace: &Path, patch_path: &Path, server: &StubLlmServer) -> std::process::ExitStatus {
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
    let (repo_dir, issue_path) = (workspace.join("repo"), workspace.join("repo").join("issue.md"));
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
        .env("VANTA_LLM_BASE_URL", format!("{}/v1", server.base_url.trim_end_matches("/v1")))
        .env("VANTA_LLM_MODEL", "stub-model")
        .env("VANTA_API_KEY", TEST_API_KEY)
        .env("VANTA_LLM_TIMEOUT_SECS", "5")
        .env("VANTA_PYTHON", python)
        .env("PATH", "/usr/bin:/bin")
        .env("VANTA_SANDBOX_MEMORY_MB", "0");
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.status().unwrap()
}

#[test]
fn patch_mode_three_attempt_run_writes_multi_record_session_log() {
    let _serial = serial_run();
    let workspace = TempDir::new().unwrap();
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start(THREE_SIGNAL_TESTS);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);

    assert!(!status.success(), "the deliberately bad patch should not be approved");
    assert_eq!(server.request_count(), 3, "expected one attacker call per attempt");

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
        start.summary.contains("backend="),
        "session.start must record the sandbox backend; got {}",
        start.summary
    );
    // A commit hash alone would name a commit that does not reproduce a run made
    // on a dirty tree, so the clean/dirty flag travels with it.
    assert!(
        start.summary.contains("vanta_commit=") && start.summary.contains("vanta_dirty="),
        "session.start must record the Vanta revision AND whether the tree was dirty; got {}",
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
        "def add(a, b):\n    return a - b\n"
    );

    // The structural checks above prove the tree is well-formed. These prove the
    // patched tests actually RAN against it: a wrong patch must produce Failed
    // assertion results, not Error. Error here would mean the harness failed to
    // import the tree — the exact bug that previously hid behind Static rows.
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
        patched_result_kinds.iter().any(|r| r == "Failed"),
        "the deliberately wrong stub patch must yield Failed patched results; got {patched_result_kinds:?}"
    );
    assert!(
        !patched_result_kinds.iter().any(|r| r == "Error"),
        "no patched test may Error — Error means the harness failed to produce a runnable tree; got {patched_result_kinds:?}"
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
    assert_eq!(patched_result_rows.len(), 15);

    let mut per_iteration_counts: Vec<usize> = attempt_start_rows
        .iter()
        .map(|attempt| {
            let iteration_id = attempt.testing_iteration_id.as_ref().unwrap().as_str();
            patched_result_rows
                .iter()
                .filter(|row| {
                    row.testing_iteration_id
                        .as_ref()
                        .map(|id| id.as_str())
                        == Some(iteration_id)
                })
                .count()
        })
        .collect();
    per_iteration_counts.sort_unstable();
    assert_eq!(
        per_iteration_counts,
        vec![3, 6, 6],
        "the stub repeats the same three test names, so the shared set stabilizes at three unique tests"
    );

    let unique_patch_ids: std::collections::HashSet<&str> = attempt_start_rows
        .iter()
        .map(|row| row.patch_attempt_id.as_ref().unwrap().as_str())
        .collect();
    assert_eq!(unique_patch_ids.len(), 1, "same patch should keep one stable patch_attempt_id");

    // 28 attempt records + the `session.start` run-identity record.
    assert_eq!(records.len(), 29);
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
    assert!(!status.success(), "a patch that breaks syntax must never be approved");

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
        patched_rows.iter().any(|s| s.contains("failure_class=Static")),
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
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);
    let server = StubLlmServer::start(THREE_SIGNAL_TESTS);

    let status = run_patch_mode(workspace.path(), &patch_path, &server);
    assert!(!status.success(), "the deliberately bad patch should not be approved");
    assert_eq!(server.request_count(), 3, "three review rounds must all run");

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
            !(r.from_patch_attempt_id.is_some()
                && r.from_patch_attempt_id == r.to_patch_attempt_id)
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
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);

    // Fired on the first request — i.e. during attempt 1, after it has already
    // read patch A. Attempt 2 then reads patch B, so 1->2 is a real transition.
    let replacement = patch_path.clone();
    let hook = move |index: usize| {
        if index == 0 {
            fs::write(&replacement, SYNTAX_ERROR_PATCH).unwrap();
        }
    };
    let server = StubLlmServer::start_with_hook(THREE_SIGNAL_TESTS, Some(Box::new(hook)));

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
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), WRONG_BEHAVIOUR_PATCH);

    // Requests 0 and 1 serve attempts 1 and 2; request 2 is attempt 3's attacker
    // call. Dropping that connection aborts the run before attempt 3 completes.
    let server = StubLlmServer::start_aborting(THREE_SIGNAL_TESTS, 2);

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
        rows.iter().any(|r| r.from_attempt == 0 && r.to_attempt == 1),
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

    // 1 generation + 3 revisions. Scrapping on the first mismatch — the
    // regressed behaviour — would leave this at 1.
    assert_eq!(
        server.request_count(),
        4,
        "a test whose baseline contradicts its intent must be sent back to the \
         Attacker for revision (up to 3 times) before being scrapped; \
         expected 1 generation + 3 revisions = 4 LLM calls"
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
    let (_repo_dir, _issue_path, patch_path) =
        write_test_repo(workspace.path(), HANGING_PATCH);
    let server = StubLlmServer::start(HANGING_PROBE);

    // A short cap so the hang is killed promptly instead of after the 120s
    // default. This is the ONLY way the timeout path gets exercised for real.
    let status = run_patch_mode_with(
        workspace.path(),
        &patch_path,
        &server,
        1,
        &[("VANTA_TEST_TIMEOUT_SECS", "3")],
    );
    assert!(
        !status.success(),
        "a patch whose test hangs must not be approved"
    );

    // Exactly one request: the Attacker generation. Classifying the timeout as
    // Unknown would have called the Attacker again to simplify the probe.
    assert_eq!(
        server.request_count(),
        1,
        "a timed-out test must classify as Infrastructure, not fall through to \
         Unknown — the Unknown chain asks the Attacker to simplify the probe, \
         which would show up as a second LLM call"
    );

    let records = read_session_records(&find_session_log(workspace.path()));
    let debug_dir = workspace
        .path()
        .join("vanta_sessions")
        .join(records[0].session_id.as_str())
        .join("debug");
    let diagnostic = read_json(&debug_dir.join("attempt_1_diagnostic_test_hangs_on_patched_tree.json"));

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
}
