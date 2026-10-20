// ============================================================================
// HARNESS — the sandbox test-running layer (Spec §7 Phase 1/2 execution)
// ============================================================================
//
// [VANTA-WHY] M2 and M3 each grew their own copy of "spawn a sandbox, write the
// repo + tests, run pytest, parse the report". The two copies drifted (different
// report parsing, different static-analysis streams). This module is the single
// definition of HOW Vanta executes tests; the Creator/Attacker/BATE/Governor
// logic is separate and calls in here. Every caller — the BATE baseline, the
// patched run, the Diagnostician's rerun/simplify probes — goes through these
// functions, so a fix to test execution lands once.
// ============================================================================

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use futures::future::join_all;

use crate::agents::attacker::GeneratedTest;
use crate::agents::creator::SourceFile;
use crate::core::TestResult;
use crate::pipeline::candidate::TestCommand;
use crate::sandbox::{Backend, ExecutionResult};

// ============================================================================
// TestExecutionDetails — a single test's outcome plus the raw streams
// ============================================================================
//
// [VANTA-WHY] The Diagnostician needs more than pass/fail: it classifies a
// failure from the timeout flag, whether a report was produced, and the captured
// stdout/stderr. Carrying the streams alongside the result is what lets a
// "pytest never wrote report.json" (Infrastructure) be told apart from a real
// assertion failure (Logic).
#[derive(Debug, Clone, PartialEq)]
pub struct TestExecutionDetails {
    pub result: TestResult,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub report_path_exists: bool,
    pub sandbox_root: Option<String>,
}

impl TestExecutionDetails {
    pub fn infrastructure_error() -> Self {
        Self {
            result: TestResult::Error,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            timed_out: false,
            report_path_exists: false,
            sandbox_root: None,
        }
    }

    /// Wrap a bare `TestResult` from a rerun (`run_single_test`, which returns
    /// only pass/fail) as details the Diagnostician can classify.
    ///
    /// [VANTA-WHY] A rerun DID produce a verdict, so `report_path_exists` is
    /// true: the "pytest never wrote report.json" Infrastructure signal belongs
    /// to the original run, not to a probe that already returned a result.
    pub fn from_rerun(result: TestResult) -> Self {
        Self {
            result,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            timed_out: false,
            report_path_exists: true,
            sandbox_root: None,
        }
    }
}

// ============================================================================
// run_tests_parallel_detailed — one sandbox per test, in parallel
// ============================================================================
pub async fn run_tests_parallel_detailed(
    source: &str,
    tests: &[&GeneratedTest],
    test_cmd: &TestCommand,
) -> Result<HashMap<String, TestExecutionDetails>, Box<dyn std::error::Error + Send + Sync>> {
    let mut handles = Vec::new();
    for test in tests {
        let source = source.to_string();
        let test = (*test).clone();
        let cmd_template = test_cmd.clone();
        handles.push(tokio::spawn(async move {
            let name = test.test_name.clone();
            let sb = match Backend::spawn().await {
                Ok(b) => b,
                Err(_) => return (name, TestExecutionDetails::infrastructure_error()),
            };
            if write_source_files(sb.root(), &source).is_err() {
                sb.destroy();
                return (name, TestExecutionDetails::infrastructure_error());
            }
            let _ = write_conftest(sb.root());
            let _ = fs::write(sb.root().join("test_run.py"), &test.test_source);
            let cmd = build_single_test_cmd(&cmd_template, "test_run.py");
            let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
            let sandbox_root = sb.root().display().to_string();
            let exec = sb
                .execute(&cmd_refs, Path::new(""), patched_test_timeout_secs())
                .await;
            let result = match exec {
                Ok(e) => parse_pytest_report_detailed(
                    sb.root().join("report.json"),
                    &e,
                    Some(sandbox_root),
                    "test_run.py",
                ),
                Err(_) => TestExecutionDetails::infrastructure_error(),
            };
            sb.destroy();
            (name, result)
        }));
    }

    let mut outcomes = HashMap::new();
    for res in join_all(handles).await {
        if let Ok((name, result)) = res {
            outcomes.insert(name, result);
        }
    }
    Ok(outcomes)
}

/// Run multiple tests in parallel and return only their pass/fail results.
pub async fn run_tests_parallel(
    source: &str,
    tests: &[&GeneratedTest],
    test_cmd: &TestCommand,
) -> Result<HashMap<String, TestResult>, Box<dyn std::error::Error + Send + Sync>> {
    let detailed = run_tests_parallel_detailed(source, tests, test_cmd).await?;
    Ok(detailed
        .into_iter()
        .map(|(name, details)| (name, details.result))
        .collect())
}

// ============================================================================
// run_single_test — run ONE test against a source tree, return its TestResult
// ============================================================================
//
// [VANTA-WHY] Used by the BATE baseline (and its flaky pre-check) and by the
// Diagnostician's rerun/simplify probes. `run_single_test` returns only a
// pass/fail, so a timeout here collapses to `TestResult::Error` — the detailed
// path (which keeps the `timed_out` flag) is `run_tests_parallel_detailed`.
pub async fn run_single_test(
    source_concat: &str,
    test_source: &str,
    test_cmd: &TestCommand,
) -> TestResult {
    run_single_test_with_timeout(source_concat, test_source, test_cmd, test_timeout_secs())
        .await
        .result
}

/// Run one test and keep the execution details for rerun diagnostics.
pub async fn run_single_test_with_timeout(
    source_concat: &str,
    test_source: &str,
    test_cmd: &TestCommand,
    timeout_secs: u64,
) -> TestExecutionDetails {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return TestExecutionDetails::infrastructure_error(),
    };
    let _ = write_source_files(sb.root(), source_concat);
    let _ = write_conftest(sb.root());
    let _ = fs::write(sb.root().join("test_bate.py"), test_source);
    let cmd = build_single_test_cmd(test_cmd, "test_bate.py");
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let sandbox_root = sb.root().display().to_string();
    let exec = sb.execute(&cmd_refs, Path::new(""), timeout_secs).await;
    let result = match exec {
        Ok(e) => parse_pytest_report_detailed(
            sb.root().join("report.json"),
            &e,
            Some(sandbox_root),
            "test_bate.py",
        ),
        Err(_) => TestExecutionDetails::infrastructure_error(),
    };
    sb.destroy();
    result
}

// ============================================================================
// Native suite — the repo's OWN tests, run as a gate
// ============================================================================
//
// [VANTA-WHY] Every other function here runs ONE generated test file. The
// repo's own suite is a different question: it does not depend on what the
// Attacker happened to sample, so a patch that breaks an existing repo test
// fails here even when no generated test covers that input. It is NOT part of
// the shared set and never contributes to EIR/ECR — it is a separate
// `native_suite` section — because it is not a transition measurement.
//
// [VANTA-WHY] Per-test IDs, not one aggregate verdict: "pass on original, fail
// on patch" is a regression; "already failing on original" is not (the patch
// did not break it). A single pass/fail bit cannot express that difference.
pub struct NativeSuiteReport {
    /// nodeid → outcome. Empty when the suite could not be run at all.
    pub tests: Vec<(String, TestResult)>,
    /// nodeid → the test's own failure output, for a regression report.
    ///
    /// [VANTA-WHY] A regression the Creator is told about has to carry the
    /// output that shows WHY, exactly as a generated-test failure does. Without
    /// it the repair feedback for a native regression reduces to "a test broke"
    /// — the model cannot act on that. Only failing/erroring tests appear here.
    pub failure_outputs: HashMap<String, String>,
    /// False when no report was produced (missing pytest/plugin, timeout,
    /// crash). A gate that could not run must never be read as "no regressions".
    pub ran: bool,
    /// The test command's exit status, when it ran at all.
    pub exit_code: Option<i32>,
    /// The last few non-empty stderr lines, for a run-failure report.
    pub stderr_tail: String,
}

impl NativeSuiteReport {
    /// The suite never started: the backend could not spawn, or the source tree
    /// could not be written into the sandbox.
    fn not_run() -> Self {
        Self {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: false,
            exit_code: None,
            stderr_tail: String::new(),
        }
    }

    /// The suite started but produced no usable report (timeout, crash, bad
    /// JSON, or a collection error). Carries the exit code and stderr tail so a
    /// caller can REPORT why the tree could not be measured instead of reading
    /// the emptiness as "the tree has no tests".
    fn failed_run(exit_code: Option<i32>, stderr_tail: String) -> Self {
        Self {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: false,
            exit_code,
            stderr_tail,
        }
    }

    /// The recorded failure output for a nodeid, collapsed to one line.
    pub fn failure_output(&self, nodeid: &str) -> Option<&str> {
        self.failure_outputs.get(nodeid).map(String::as_str)
    }

    /// Did this tree produce no usable test set — because the suite never ran,
    /// or because it ran and collected ZERO tests?
    ///
    /// [VANTA-WHY] Fix 2 of the rerun fixes. `lookup_native` returns `None` both
    /// when a tree's suite ran and genuinely lacks a nodeid (item 11's absence,
    /// which discriminates) and when the suite produced nothing at all (a broken
    /// command, which measures nothing). The two are only distinguishable by
    /// asking whether the tree collected anything, so that question is a method.
    pub fn collected_nothing(&self) -> bool {
        !self.ran || self.tests.is_empty()
    }
}

/// The last few non-empty lines of a tool's stderr, for a run-failure note.
///
/// [VANTA-WHY] A failed test command explains itself at the END of stderr —
/// pytest prints the collection error and its traceback last. The whole stream
/// can be tens of KB, so the tail is what makes a `tree_run_failed` note
/// actionable without burying the reader.
pub fn stderr_tail(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect();
    let tail = lines[lines.len().saturating_sub(3)..].join(" | ");
    if tail.chars().count() <= 300 {
        return tail;
    }
    let mut out: String = tail.chars().take(299).collect();
    out.push('…');
    out
}

/// Run the repository's own test suite against a source tree.
///
/// The source tree includes the repo's `tests/`, so pytest discovers them from
/// the sandbox root; no generated test file is written.
pub async fn run_native_suite(source: &str, test_cmd: &TestCommand) -> NativeSuiteReport {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return NativeSuiteReport::not_run(),
    };
    if write_source_files(sb.root(), source).is_err() {
        sb.destroy();
        return NativeSuiteReport::not_run();
    }
    let _ = write_conftest(sb.root());

    // No test-file argument: pytest collects the repo's own tests from rootdir.
    let mut cmd = test_cmd.as_cmd_vec();
    cmd.push("--json-report".to_string());
    cmd.push("--json-report-file=report.json".to_string());
    cmd.push("-q".to_string());
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let exec = sb
        .execute(&cmd_refs, Path::new(""), test_timeout_secs())
        .await;
    let report = match exec {
        Ok(e) => parse_native_report(&sb.root().join("report.json"), &e),
        Err(_) => NativeSuiteReport::not_run(),
    };
    sb.destroy();
    report
}

/// Parse a pytest JSON report into per-test outcomes.
fn parse_native_report(report_path: &Path, exec: &ExecutionResult) -> NativeSuiteReport {
    let tail = stderr_tail(&exec.stderr);
    if exec.timed_out {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    }
    let Ok(text) = fs::read_to_string(report_path) else {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    };
    let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) else {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    };
    let mut tests = Vec::new();
    let mut failure_outputs: HashMap<String, String> = HashMap::new();
    if let Some(arr) = report["tests"].as_array() {
        for t in arr {
            let nodeid = t["nodeid"].as_str().unwrap_or_default().to_string();
            let Some(result) = native_outcome(t["outcome"].as_str().unwrap_or_default()) else {
                continue;
            };
            if nodeid.is_empty() {
                continue;
            }
            if result != TestResult::Passed {
                // [VANTA-WHY] pytest attaches the traceback to whichever phase
                // produced it: `call` for an assertion, `setup`/`teardown` for a
                // fixture or import error. Reading all three is what makes a
                // collection-time failure explain itself instead of arriving as
                // a bare nodeid.
                for phase in ["call", "setup", "teardown"] {
                    let repr = t[phase]["longrepr"]
                        .as_str()
                        .or_else(|| t[phase]["crash"]["message"].as_str());
                    if let Some(repr) = repr {
                        failure_outputs.insert(nodeid.clone(), one_line(repr));
                        break;
                    }
                }
            }
            tests.push((nodeid, result));
        }
    }
    NativeSuiteReport {
        tests,
        failure_outputs,
        ran: true,
        exit_code: exec.exit_code,
        stderr_tail: tail,
    }
}

/// Collapse a multi-line failure report to its last meaningful line, capped.
///
/// [VANTA-WHY] A pytest `longrepr` is the whole assertion block; the actionable
/// line (the `E   TypeError: ...`) is the last one. The feedback prompt and the
/// console both want that line, not ten lines of file context.
fn one_line(text: &str) -> String {
    let last = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .next_back()
        .unwrap_or("")
        .to_string();
    if last.chars().count() <= 300 {
        return last;
    }
    let mut out: String = last.chars().take(299).collect();
    out.push('…');
    out
}

/// Map one pytest-json-report `outcome` string to a `TestResult`.
///
/// [VANTA-WHY] Item 9 — pytest reports a `unittest.TestCase.subTest` item with
/// the outcome `"subtests passed"`, NOT `"passed"`. This parser used to match
/// only `"passed"/"failed"/"error"` and drop everything else, so every subTest
/// item silently vanished: the heretic suite has three subTest tests, and Vanta
/// counted 4 native tests on master where pytest ran 5, and 12 on the C patch
/// where pytest ran 15. The undercount is not merely cosmetic — a subTest item
/// that PASSED on the original was absent from the original side, and
/// `native_regressions` walks the ORIGINAL side, so a patch that broke such a
/// test could never be reported as a regression. A failing subtest already
/// arrives as `"failed"`; only the passing case needed mapping.
fn native_outcome(outcome: &str) -> Option<TestResult> {
    match outcome {
        "passed" | "subtests passed" => Some(TestResult::Passed),
        // "failed" = assertion failure (a failing subtest included); "error" =
        // exception/setup error.
        "failed" | "error" => Some(TestResult::Failed),
        // A skipped test is neither a pass nor a failure signal, and an
        // unrecognised outcome is not a licence to guess.
        _ => None,
    }
}

/// Nodeids that PASS on the original and did not pass on the patch — the repo's
/// own suite reporting a regression. Tests already failing on the original are
/// excluded: the patch did not break them.
///
/// [VANTA-WHY] This used to iterate the PATCHED side only and look for
/// `patched == Failed && original == Passed`. A patch that introduces a syntax
/// error stops pytest at collection, so the patched side collected ZERO tests —
/// and a filter over an empty list reports zero regressions. The heretic run
/// went 4 tests → 0 tests and printed "0 regression(s)": the scariest possible
/// clean bill of health for a tree that does not import.
///
/// So the walk is over the ORIGINAL side now, and "gone" is a regression too. A
/// test that passed on the original and is MISSING from the patched report
/// (collection error, crashed interpreter, deleted test file) is not evidence
/// that the behaviour survived.
///
/// [VANTA-WHY] "Gone" is also how pass -> SKIPPED / XFAIL / DESELECTED is
/// caught, and that is deliberate rather than incidental. `native_outcome` maps
/// only `passed`/`subtests passed` and `failed`/`error`; a `skipped` (or an
/// unknown future) outcome becomes `None` and is dropped from the patched
/// report entirely, so the test lands in the `None` branch below and is counted
/// as a regression. The same holds for a test pytest never ran at all because a
/// config change deselected it (`-k`, `--deselect`, `--ignore`, a narrowed
/// `python_files`): it is absent, so it is a regression.
///
/// This is the ONLY thing that catches those cases. `modified_test_config` is
/// warn-only by design — a patch that rewrites `[tool.pytest.ini_options]` is
/// reported, never itself rejected — so a patch CAN silence a test through
/// `addopts` or a plugin, and the only defence is that a silenced test is a
/// missing test and a missing test is a regression. Adding an explicit
/// `Some(Skipped)` arm here would be redundant: the outcome is discarded before
/// this function is ever called.
pub fn native_regressions(
    original: &[(String, TestResult)],
    patched: &[(String, TestResult)],
) -> Vec<String> {
    let patched_map: HashMap<&str, &TestResult> =
        patched.iter().map(|(n, r)| (n.as_str(), r)).collect();

    let mut regressions: Vec<String> = Vec::new();
    for (name, result) in original {
        if *result != TestResult::Passed {
            continue;
        }
        match patched_map.get(name.as_str()) {
            Some(TestResult::Passed) => {}
            // Still present and no longer passing: a plain regression.
            Some(_) => regressions.push(name.clone()),
            // Absent from the patched report: the suite collapsed before (or
            // during) this test. Also a regression.
            None => regressions.push(name.clone()),
        }
    }
    regressions
}

/// Did the patched side collect FEWER tests than the original ran?
///
/// [VANTA-WHY] Kept separate from `native_regressions` so the console can say
/// "the suite collapsed" rather than just listing names: 4 → 0 is a different
/// failure from 4 → 3, even though both are regressions.
pub fn native_suite_collapsed(
    original: &NativeSuiteReport,
    patched: &NativeSuiteReport,
) -> bool {
    original.ran && !original.tests.is_empty() && patched.tests.len() < original.tests.len()
}

// ============================================================================
// The native tree — the patched source graded by the ORIGINAL tests
// ============================================================================
//
// [VANTA-WHY] A patch must never control the tests that grade it. The vulnerable
// shape, seen in repair session 888b6931: the Creator's fix made
// `keyword_rate.py` always pass `max_response_length` to `get_responses`, which
// the issue forbids when the setting is unset — and the SAME patch edited
// `tests/test_keyword_rate.py`'s `FakeContext.get_responses` to swallow the new
// kwarg. The native suite ran the patched tree, so it collected the Creator's
// edited test, the weakened test passed, Vanta reported `regressions=0`, and the
// Gatekeeper approved a patch the repo's own test rejects with a `TypeError`.
//
// The native suite therefore runs the ORIGINAL test files. Every file pytest
// would collect or load — `test_*.py`, `*_test.py`, anything under a collected
// test directory, and every `conftest.py` — is pinned back to the contents the
// original repo had, on top of the patched source. A patch-added test file is
// not part of the baseline either, so it is left out of the native run
// entirely; the baseline is the repo's ORIGINAL suite and nothing else.

/// Is this a file the repo's own test suite owns — collected by pytest, or
/// loaded by it as a conftest?
///
/// [VANTA-WHY] `conftest.py` is its own case because pytest loads it even though
/// the name matches no collection pattern, and a conftest is the cheapest place
/// to neutralize a test (`collect_ignore`, a monkeypatched fixture, a
/// `pytest_collection_modifyitems` that deselects). Anything under a `test`/
/// `tests` directory counts too: a helper module there changes what the tests do
/// without being a test file itself.
pub fn is_native_test_path(path: &str) -> bool {
    let path_ref = Path::new(path);
    let file_name = path_ref
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if file_name == "conftest.py" {
        return true;
    }
    if !file_name.ends_with(".py") {
        return false;
    }
    if file_name.starts_with("test_") || file_name.ends_with("_test.py") {
        return true;
    }
    path_ref.parent().is_some_and(|dir| {
        dir.components().any(|component| {
            matches!(component, std::path::Component::Normal(name)
                if name == "test" || name == "tests")
        })
    })
}

/// The native-suite view of one patch attempt.
pub struct NativeTree {
    /// The tree the native (patched-side) run is given.
    pub files: HashMap<String, String>,
    /// Existing test files the patch rewrote; their edits were discarded.
    pub modified_existing_tests: Vec<String>,
    /// Existing test files the patch deleted; they were restored.
    pub removed_existing_tests: Vec<String>,
    /// Test files the patch added. Not part of the baseline; reported as info.
    pub added_test_files: Vec<String>,
}

/// Build the patched tree the native suite runs: patched source, ORIGINAL tests.
///
/// `patch_deleted_tests` comes from the patch TEXT, not from `patched_files`:
/// the apply cascade leaves a deleted file at its original contents (the strict
/// path skips a `/dev/null` destination outright), so a deletion is invisible in
/// the file map and can only be read off the diff itself.
pub fn native_tree_with_original_tests(
    original_files: &HashMap<String, String>,
    patched_files: &HashMap<String, String>,
    patch_deleted_tests: &[String],
) -> NativeTree {
    let mut files: HashMap<String, String> = HashMap::new();
    let mut modified_existing_tests: Vec<String> = Vec::new();
    let mut added_test_files: Vec<String> = Vec::new();

    let mut paths: Vec<&String> = patched_files.keys().collect();
    paths.sort();
    for path in paths {
        let contents = &patched_files[path];
        if !is_native_test_path(path) {
            files.insert(path.clone(), contents.clone());
            continue;
        }
        match original_files.get(path) {
            Some(original) => {
                if original != contents {
                    modified_existing_tests.push(path.clone());
                }
                files.insert(path.clone(), original.clone());
            }
            // A test file the patch ADDED — including a new `conftest.py`. It is
            // not part of the original baseline, so it is excluded rather than
            // restored, and it can never be the reason an original test "passes"
            // or disappears.
            None => added_test_files.push(path.clone()),
        }
    }

    // A test file the patch deleted: restore the original so the test still
    // runs. Without this a patch could silence a test simply by removing it.
    let mut removed_existing_tests: Vec<String> = Vec::new();
    for path in patch_deleted_tests {
        if !is_native_test_path(path) {
            continue;
        }
        let Some(original) = original_files.get(path) else {
            continue;
        };
        removed_existing_tests.push(path.clone());
        files.insert(path.clone(), original.clone());
    }
    removed_existing_tests.sort();
    removed_existing_tests.dedup();

    NativeTree {
        files,
        modified_existing_tests,
        removed_existing_tests,
        added_test_files,
    }
}

// ============================================================================
// Source-tree helpers — write the concatenated source into a sandbox
// ============================================================================

/// Parse a source string concatenated with `# FILE: <path>` headers into
/// (path, contents) pairs.
pub fn parse_source_files(
    source_concat: &str,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
    let mut files = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_content_start = 0usize;
    let mut offset = 0usize;

    for segment in source_concat.split_inclusive('\n') {
        let line = segment.strip_suffix('\n').unwrap_or(segment);
        if let Some(rest) = line.strip_prefix("# FILE: ") {
            if let Some(path) = current_path.take() {
                let content_end = offset.saturating_sub(2);
                let content = source_concat
                    .get(current_content_start..content_end)
                    .ok_or("invalid concatenated source boundaries")?;
                files.push((path, content.to_string()));
            }
            current_path = Some(rest.trim().to_string());
            current_content_start = offset + segment.len();
        }
        offset += segment.len();
    }

    if let Some(path) = current_path.take() {
        let content = source_concat
            .get(current_content_start..)
            .ok_or("invalid final source boundaries")?;
        files.push((path, content.to_string()));
    }

    if files.is_empty() {
        return Err("no # FILE: entries found in concatenated source".into());
    }

    Ok(files)
}

/// Write source files (concatenated with `# FILE:` headers) into a sandbox root.
pub fn write_source_files(
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for (path, content) in parse_source_files(source_concat)? {
        let out = root.join(&path);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, content)?;
    }
    Ok(())
}

/// Write a forensics snapshot of the source tree plus a manifest.
pub fn write_tree_snapshot(
    snapshot_root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if snapshot_root.exists() {
        fs::remove_dir_all(snapshot_root)?;
    }
    fs::create_dir_all(snapshot_root)?;

    let files = parse_source_files(source_concat)?;
    let mut paths: Vec<String> = files.iter().map(|(path, _)| path.clone()).collect();
    paths.sort();

    let file_count = files.len();
    for (path, content) in &files {
        let out = snapshot_root.join(path);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, content)?;
    }

    let manifest = serde_json::json!({
        "paths": paths,
        "file_count": file_count,
    });
    let manifest_name = format!(
        "{}_manifest.json",
        snapshot_root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("snapshot_root missing file name")?
    );
    let manifest_path = snapshot_root
        .parent()
        .ok_or("snapshot_root missing parent")?
        .join(manifest_name);
    fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

/// Write a conftest.py at the sandbox root that puts `src/` and `.` on sys.path.
/// [VANTA-WHY] Repos with a `src/` layout need `src/` on sys.path for
/// `import <pkg>` to work. pytest auto-loads conftest.py from the rootdir.
pub fn write_conftest(root: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // [VANTA-WHY] Use a raw string literal (r#"..."#) so the Python indentation
    // is preserved exactly. Line-continuation style stripped leading whitespace
    // and caused IndentationError → every test returned Error (pytest exit 4,
    // no report.json).
    let conftest = r#"import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, 'src'), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
"#;
    fs::write(root.join("conftest.py"), conftest)?;
    Ok(())
}

// ============================================================================
// pytest report parsing
// ============================================================================

/// Parse pytest's JSON report into the outcome of a specific generated test
/// file.
///
/// [VANTA-WHY] An "error" outcome (ImportError/AttributeError during the test
/// body) means the test did NOT pass — equivalent to "failed". `TestResult::Error`
/// is reserved for INFRASTRUCTURE failures (timeout, missing report, crash),
/// which now also covers "the report does not contain the generated test".
pub fn parse_pytest_report(
    report_path: PathBuf,
    exec: &ExecutionResult,
    generated_test_file: &str,
) -> TestResult {
    parse_pytest_report_detailed(report_path, exec, None, generated_test_file).result
}

pub fn parse_pytest_report_detailed(
    report_path: PathBuf,
    exec: &ExecutionResult,
    sandbox_root: Option<String>,
    generated_test_file: &str,
) -> TestExecutionDetails {
    if exec.timed_out {
        return TestExecutionDetails {
            result: TestResult::Error,
            stdout: exec.stdout.clone(),
            stderr: exec.stderr.clone(),
            exit_code: exec.exit_code,
            timed_out: true,
            report_path_exists: report_path.exists(),
            sandbox_root,
        };
    }
    if report_path.exists() {
        if let Ok(text) = fs::read_to_string(&report_path) {
            if let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(result) = generated_test_outcome(&report, generated_test_file) {
                    return TestExecutionDetails {
                        result,
                        stdout: exec.stdout.clone(),
                        stderr: exec.stderr.clone(),
                        exit_code: exec.exit_code,
                        timed_out: false,
                        report_path_exists: true,
                        sandbox_root,
                    };
                }
            }
        }
    }
    // [VANTA-WHY] Fail closed. The report exists but the GENERATED test's nodeid
    // is not in it (or there is no usable report at all): the generated test was
    // never graded, so `report_path_exists == false` — "this tree did not reach a
    // verdict" — and the result is `Error`, never `Passed`. `tests.first()` used
    // to stand here, and the first entry was a passing repository test.
    TestExecutionDetails {
        result: TestResult::Error,
        stdout: exec.stdout.clone(),
        stderr: exec.stderr.clone(),
        exit_code: exec.exit_code,
        timed_out: false,
        report_path_exists: false,
        sandbox_root,
    }
}

/// The outcome of the GENERATED test file, selected by the file part of its
/// nodeid.
///
/// [VANTA-WHY] The grading bug. A `--test-cmd` that named a test path (e.g.
/// `python -m pytest tests`) made pytest collect the repository's own suite
/// before the generated file, and this reader took `report["tests"].first()` as
/// the generated test's result. That first entry was always one of the repo's
/// tests — which passes — so every BugReproduction test was recorded as "passed
/// on the buggy original", scrapped by BATE, and no candidate could ever be
/// rejected for failing to reproduce the bug. The generated file is written into
/// the sandbox as one fixed name (`test_run.py` / `test_bate.py`), so its nodeid
/// is identified by that FILE. A nodeid that is absent from the report yields
/// `None` — the caller fails closed rather than reading someone else's result.
fn generated_test_outcome(
    report: &serde_json::Value,
    generated_test_file: &str,
) -> Option<TestResult> {
    let wanted = nodeid_file(generated_test_file);
    for t in report["tests"].as_array()? {
        let nodeid = t["nodeid"].as_str().unwrap_or_default();
        if nodeid_file(nodeid) == wanted {
            let result = match t["outcome"].as_str().unwrap_or_default() {
                "passed" | "subtests passed" => TestResult::Passed,
                // "failed" = assertion failure; "error" = exception during the
                // test body. Both mean the generated test did not pass.
                "failed" | "error" => TestResult::Failed,
                // An outcome we do not recognise is not a licence to pass.
                _ => TestResult::Error,
            };
            return Some(result);
        }
    }
    None
}

/// The file part of a pytest nodeid, normalised so `./a.py` and `a.py` match.
fn nodeid_file(nodeid: &str) -> String {
    nodeid
        .split("::")
        .next()
        .unwrap_or(nodeid)
        .trim()
        .trim_start_matches("./")
        .replace('\\', "/")
}

// ============================================================================
// Static analysis (ruff + mypy) — SET differences, not COUNT differences
// ============================================================================
//
// [VANTA-WHY] The old check compared COUNTS: `patch_ruff > base_ruff`. A count
// is not a set. If the patch fixed two pre-existing ruff errors and introduced
// two new ones, the count is identical and the run reports "no new static
// errors" for a patch that just added two. Worse, ruff and mypy both number
// their diagnostics by LINE, so a patch that shifts every line down by one
// changes every diagnostic's position without changing a single finding, and the
// raw text comparison would call all of them new.
//
// The fix is to compare the findings as a SET keyed on (file, code, message) —
// deliberately WITHOUT line numbers — and to hand the Diagnostician only the
// findings the patch actually introduced.

/// One static-analysis finding.
///
/// [VANTA-WHY] The `(file, code, message)` triple is the IDENTITY used by the
/// set diff, so the line number is excluded from `PartialEq`/`Hash`: the same
/// diagnostic moving down the file after a patch is not a NEW error, and
/// counting it as one would reject a candidate that merely changed its own line
/// count. The line is still CARRIED whenever the tool reported one — the
/// evidence record and the console need the exact site, the set diff must not.
#[derive(Debug, Clone)]
pub struct StaticDiagnostic {
    /// Repo-relative path the tool reported.
    pub file: String,
    /// 1-based line the tool reported, when it reported one.
    pub line: Option<u32>,
    /// ruff rule (e.g. `F821`) or mypy error code (e.g. `return-value`).
    pub code: String,
    /// The human-readable finding text, without the location.
    pub message: String,
}

// [VANTA-WHY] Identity is (file, code, message) — see the struct doc. Deriving
// PartialEq/Hash would fold the line in and break the set diff, so they are
// written by hand and deliberately ignore `line`.
impl PartialEq for StaticDiagnostic {
    fn eq(&self, other: &Self) -> bool {
        self.file == other.file && self.code == other.code && self.message == other.message
    }
}
impl Eq for StaticDiagnostic {}
impl std::hash::Hash for StaticDiagnostic {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.file.hash(state);
        self.code.hash(state);
        self.message.hash(state);
    }
}

impl StaticDiagnostic {
    /// Is this finding a fact about the ANALYSIS ENVIRONMENT rather than the
    /// code?
    ///
    /// [VANTA-WHY] Item 10. mypy's `import-not-found` / `import-untyped` mean
    /// exactly "I cannot see this module from here" — an environment fact, not a
    /// defect in the candidate. Session 40e84874 rejected C because the generated
    /// `tests/test_auto_rank.py` imports `torch` and the sandbox had no
    /// dependencies installed, so `torch` was reported as `import-not-found`.
    /// These findings are still RECORDED (a reader must be able to see them) but
    /// must never block a review round: blocking on one claims the candidate is
    /// broken when it merely imports a library the analysis environment lacks.
    pub fn is_environment_finding(&self) -> bool {
        matches!(self.code.as_str(), "import-not-found" | "import-untyped")
    }

    /// One line for the Diagnostician's payload and the session log, with the
    /// reported line number when the tool gave one.
    pub fn render(&self) -> String {
        let location = match self.line {
            Some(line) => format!("{}:{}", self.file, line),
            None => self.file.clone(),
        };
        if self.code.is_empty() {
            format!("{}: {}", location, self.message)
        } else {
            format!("{}: [{}] {}", location, self.code, self.message)
        }
    }
}

/// What one static run found, per tool, plus the raw streams.
#[derive(Debug, Clone, Default)]
pub struct StaticRun {
    pub ruff: Vec<StaticDiagnostic>,
    pub mypy: Vec<StaticDiagnostic>,
    pub ruff_raw: String,
    pub mypy_raw: String,
}

/// Repo config files that change what ruff / mypy / pytest report.
///
/// [VANTA-WHY] Item 10: the sandbox only carried `src/` and `tests/`, so ruff
/// ran with its DEFAULT ruleset and mypy with no config at all. A repo's own
/// config is part of how that repo defines "correct" — without it the static
/// step reports the wrong findings and says nothing about the candidate. These
/// are the standard discovery names of the three tools, plus pytest's, so a
/// root `conftest.py` that adjusts `sys.path` is honoured too.
pub const STATIC_CONFIG_FILES: &[&str] = &[
    "pyproject.toml",
    "setup.cfg",
    "mypy.ini",
    ".mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "pytest.ini",
    "tox.ini",
    "conftest.py",
];

/// The ruff invocation, minus the interpreter.
///
/// [VANTA-WHY] Item 10: ruff's DEFAULT output is the multi-line "full" report
/// (`F821 Undefined name ...` then ` --> path.py:1:5` on its own line), which
/// `parse_ruff_line` cannot read — so every ruff run so far silently parsed to
/// ZERO findings, whether the tree had one finding or forty. `concise` is the
/// one-line `path.py:1:5: F821 message` form the parser expects, so the flag is
/// part of the contract, not a preference.
pub const RUFF_ARGS: &[&str] = &["-m", "ruff", "check", "--output-format=concise", "."];

/// Copy the repo's config files from `repo_root` into the sandbox `dest`.
///
/// [VANTA-WHY] Only files that actually exist are copied: a repo without a
/// `ruff.toml` must not gain an empty one, because an empty config is not the
/// same as "no config" — it would change ruff's behaviour rather than reproduce
/// it. Returns the names copied, so a caller can log them.
pub fn copy_static_configs(repo_root: &Path, dest: &Path) -> Vec<String> {
    let mut copied = Vec::new();
    for name in STATIC_CONFIG_FILES {
        let src = repo_root.join(name);
        if src.is_file() && fs::copy(&src, dest.join(name)).is_ok() {
            copied.push((*name).to_string());
        }
    }
    copied
}

/// The Python interpreter a test command actually runs.
///
/// [VANTA-WHY] Item 10: the static tools must run under the SAME interpreter as
/// the tests — only that interpreter has the repo's dependencies installed, so
/// only it can resolve `torch`/`pydantic` instead of reporting them as
/// `import-not-found`, and only under it does mypy see real types rather than
/// treating every module as `Any`. The real heretic command is
/// `env PYTHONPATH=src /path/.venv/bin/python -m pytest -q`, where the
/// interpreter hides behind an `env` wrapper and an env assignment; a plain
/// `python3 -m pytest` names it directly. Both must resolve to the venv python,
/// never to the host's bare `python3`. Falls back to `VANTA_PYTHON` (then
/// `python3`) when the command names no interpreter at all.
pub fn interpreter_for(test_cmd: &TestCommand) -> String {
    let mut tokens = vec![test_cmd.program.as_str()];
    tokens.extend(test_cmd.args.iter().map(String::as_str));
    for token in tokens {
        // `PYTHONPATH=src` is an env assignment and `-m` is a flag; neither is
        // the interpreter.
        if token.contains('=') || token.starts_with('-') {
            continue;
        }
        let name = Path::new(token)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if name.starts_with("python") {
            return token.to_string();
        }
    }
    python_exe()
}

/// Can `python` import `module`?
///
/// [VANTA-WHY] A cheap HOST-side probe (`import <module>` only resolves an
/// already-installed package — no untrusted code runs). It is what lets the
/// static step pick a runner that actually has the tool installed, instead of
/// assuming the project interpreter and the tool interpreter are the same
/// process.
fn has_module(python: &str, module: &str) -> bool {
    std::process::Command::new(python)
        .args(["-c", &format!("import {module}")])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The interpreter to RUN a static tool with.
///
/// [VANTA-WHY] A tool must run where the tool is installed, and mypy must still
/// SEE the project's dependencies — and those are not always one interpreter.
/// heretic's venv has `ruff` but no `mypy` (its dev type checker is `ty`), while
/// Vanta's own `python3` has mypy and ruff but none of heretic's dependencies.
/// So the runner is the PROJECT interpreter when it can import the tool, and
/// Vanta's own (`VANTA_PYTHON` / `python3`) otherwise; mypy is separately
/// pointed at the project interpreter with `--python-executable`. Hard-coding
/// either one makes the step fail: the bare host python reports every dependency
/// as `import-not-found` (the 40e84874 false reject), and the bare venv cannot
/// run mypy at all and so reports NOTHING.
fn static_tool_runner(project_python: &str, module: &str) -> String {
    if has_module(project_python, module) {
        project_python.to_string()
    } else {
        python_exe()
    }
}

/// Run ruff and mypy over a source tree; parse each tool's findings into a set.
///
/// [VANTA-WHY] `test_cmd` and `config_root` exist so the static step is run the
/// way the repo is actually checked (item 10): the tools run under an
/// interpreter that has them, mypy inspects the interpreter the TESTS run under
/// (the one with the dependencies), and the repo's own config files are copied
/// in beside `src/` and `tests/`. Before this, the step ran the host's `python3`
/// against a tree with no config and no dependencies — noisy (one
/// `import-not-found` per third-party import) and blind (a missing module types
/// as `Any`, hiding real errors).
pub async fn run_static(
    source: &str,
    test_cmd: &TestCommand,
    config_root: Option<&Path>,
) -> StaticRun {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return StaticRun::default(),
    };
    let _ = write_source_files(sb.root(), source);
    if let Some(root) = config_root {
        copy_static_configs(root, sb.root());
    }

    // [VANTA-WHY] The interpreter the TESTS run under — the only one with the
    // repo's dependencies. mypy is told to inspect it (`--python-executable`) so
    // a third-party import resolves to the installed package rather than
    // collapsing to `Any` (item 10).
    let project_py = interpreter_for(test_cmd);

    let ruff_py = static_tool_runner(&project_py, "ruff");
    let mut ruff_argv = vec![ruff_py.as_str()];
    ruff_argv.extend_from_slice(RUFF_ARGS);
    let (ruff, ruff_raw) = match sb.execute(&ruff_argv, Path::new(""), 30).await {
        Ok(r) => {
            let raw = format!("{}\n{}", r.stdout, r.stderr);
            let findings = raw
                .lines()
                .filter_map(parse_ruff_line)
                .collect::<Vec<_>>();
            (findings, raw)
        }
        Err(_) => (Vec::new(), String::new()),
    };

    let mypy_py = static_tool_runner(&project_py, "mypy");
    let (mypy, mypy_raw) = match sb
        .execute(
            &[
                mypy_py.as_str(),
                "-m",
                "mypy",
                "--python-executable",
                project_py.as_str(),
                ".",
            ],
            Path::new(""),
            60,
        )
        .await
    {
        Ok(r) => {
            let raw = format!("{}\n{}", r.stdout, r.stderr);
            let findings = raw
                .lines()
                .filter_map(parse_mypy_line)
                .collect::<Vec<_>>();
            (findings, raw)
        }
        Err(_) => (Vec::new(), String::new()),
    };

    sb.destroy();
    StaticRun {
        ruff,
        mypy,
        ruff_raw,
        mypy_raw,
    }
}

/// Findings present in `patched` and absent from `baseline`.
///
/// [VANTA-WHY] Set difference, not a count comparison — a patch that fixes two
/// findings and adds two must report two NEW findings, not zero.
pub fn new_diagnostics<'a>(
    baseline: &[StaticDiagnostic],
    patched: &'a [StaticDiagnostic],
) -> Vec<&'a StaticDiagnostic> {
    let known: std::collections::HashSet<&StaticDiagnostic> = baseline.iter().collect();
    patched.iter().filter(|d| !known.contains(d)).collect()
}

/// Is this ruff finding an ERROR — code that cannot run or cannot resolve — or a
/// style preference?
///
/// [VANTA-WHY] A new finding in a file the patch touches may reject a review
/// round, but ruff's default ruleset is mostly STYLE (B, BLE, UP, SIM, TRY, ...).
/// Rejecting on those reports a candidate as broken when it merely has a nit,
/// which is a false finding about the model and burns an attack round. Only rules
/// that mean "this code is wrong" may block:
///   - `F`  — Pyflakes: undefined names, unreachable code, broken redefinitions;
///   - `E9` — ruff's syntax errors (`E999` unparsable).
/// Everything else is still REPORTED (the Creator sees it, the session log keeps
/// it); it just does not reject.
///
/// [VANTA-WHY] mypy needs no equivalent: its parser only accepts `: error:`
/// lines, so every finding it produces is already error-class.
pub fn ruff_is_error_class(code: &str) -> bool {
    code.starts_with('F') || code.starts_with("E9")
}

/// Is `file` a path the patch touches?
///
/// [VANTA-WHY] ruff reports paths relative to the repo root and a patch names
/// them as its headers do (`a/src/pkg/mod.py`), so neither plain equality nor a
/// suffix match alone is right. Both are tried. One definition, shared by the
/// static gate and the Diagnostician's "was the fault in a file the patch
/// touched?" probe — two copies would drift exactly as M2/M3's harnesses did.
pub fn path_touched(file: &str, touched: &[String]) -> bool {
    let norm = file.trim_start_matches("./");
    touched.iter().any(|p| {
        let p = p.trim_start_matches("./");
        p == norm || norm.ends_with(p) || p.ends_with(norm)
    })
}

/// Render findings as one newline-separated block (the Diagnostician payload).
pub fn render_diagnostics(findings: &[&StaticDiagnostic]) -> String {
    findings
        .iter()
        .map(|d| d.render())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `path.py:12:5: F821 Undefined name ...` — the code is the leading token of
/// the text after the location.
fn parse_ruff_line(line: &str) -> Option<StaticDiagnostic> {
    let parts: Vec<&str> = line.splitn(4, ':').collect();
    if parts.len() < 4 {
        return None;
    }
    let file = parts[0].trim();
    if !file.ends_with(".py") || !is_line_number(parts[1]) || !is_line_number(parts[2]) {
        return None;
    }
    let rest = parts[3].trim();
    let (code, message) = match rest.split_once(char::is_whitespace) {
        Some((c, m)) => (c.to_string(), m.trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    Some(StaticDiagnostic {
        file: file.to_string(),
        line: parts[1].trim().parse::<u32>().ok(),
        code,
        message,
    })
}

/// `path.py:12: error: message  [code]` (mypy reports the line, sometimes a
/// column, then `error:`).
fn parse_mypy_line(line: &str) -> Option<StaticDiagnostic> {
    let (head, tail) = line.split_once(": error:")?;
    let mut head_parts = head.split(':');
    let file = head_parts.next()?.trim();
    if !file.ends_with(".py") {
        return None;
    }
    // mypy prints `path.py:LINE: error: …` (a column only under
    // --show-column-numbers, in which case the first number is still LINE).
    let line_no = head_parts.next().and_then(|n| n.trim().parse::<u32>().ok());
    let tail = tail.trim();
    // The bracketed error code, when present, is the final `[...]` group.
    let (message, code) = match tail.rfind('[') {
        Some(i) if tail.ends_with(']') => (
            tail[..i].trim().to_string(),
            tail[i + 1..tail.len() - 1].trim().to_string(),
        ),
        _ => (tail.to_string(), String::new()),
    };
    Some(StaticDiagnostic {
        file: file.to_string(),
        line: line_no,
        code,
        message,
    })
}

fn is_line_number(token: &str) -> bool {
    !token.trim().is_empty() && token.trim().parse::<u32>().is_ok()
}

// ============================================================================
// Command construction
// ============================================================================

/// Options that consume the NEXT argument as their value.
///
/// [VANTA-WHY] Needed to tell a positional TEST PATH from the value of a flag.
/// `python -m pytest tests` and `pytest -m slow` look identical to a reader that
/// does not know the option set: in the first the trailing token is a path to be
/// REPLACED, in the second it is a marker expression to KEEP. Getting it wrong
/// in either direction is expensive — dropping an option's value breaks the
/// command, keeping a test path re-collects the repository's own suite (the bug
/// `build_single_test_cmd` exists to fix). Only this table can separate them.
const VALUE_TAKING_OPTIONS: &[&str] = &[
    "-k", "-m", "-p", "-c", "-n", "-o", "-W", "-r",
    "--rootdir", "--ignore", "--ignore-glob", "--deselect", "--maxfail",
    "--tb", "--timeout", "--junitxml", "--junit-xml", "--basetemp", "--dist",
    "--durations", "--capture", "--override-ini", "--import-mode",
    "--log-level", "--log-file", "--log-cli-level", "--json-report-file",
];

/// Build the command that runs **only** the generated test file.
///
/// [VANTA-WHY] The user's test command still supplies the interpreter, the flags
/// and (through the backend) the cwd — but its test SELECTION is replaced, not
/// extended. Appending the generated file to a command that already named a path
/// (`python -m pytest tests`) made pytest collect the repository's own suite
/// first, and the report reader then took the first result in the report as the
/// generated test's. Every BugReproduction test was therefore graded against a
/// passing repository test: it "passed on the buggy original", so BATE scrapped
/// it, so no candidate could ever be rejected for failing to reproduce the bug.
/// The two halves — this and `generated_test_outcome` — are fixed together.
///
/// [VANTA-WHY] Every non-flag argument is dropped, because in a pytest
/// invocation a positional argument has exactly one meaning: a test selection.
/// The tempting alternative — drop the positionals that name a path *in the
/// tree* — needs the tree, and the only tree available here is the source
/// concat, which is capped at 64 KB per file and 256 KB in total. The
/// repository's own test file is regularly the thing that falls outside those
/// caps (black's `tests/test_black.py` is ~200 KB), so a membership test would
/// miss, `tests` would be kept, and the repository's suite would be
/// re-collected: the bug this function exists to fix comes straight back, and
/// this time only for the largest repositories. The rule's cost is the opposite
/// direction — an option missing from `VALUE_TAKING_OPTIONS` loses its value —
/// and that is the benign failure: a reporting flag falls back to its default,
/// instead of a test being graded against the wrong result with nobody noticing.
pub fn build_single_test_cmd(test_cmd: &TestCommand, test_file: &str) -> Vec<String> {
    let mut cmd = vec![test_cmd.program.clone()];
    let args = &test_cmd.args;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            // Everything after `--` is positional by definition, and positional
            // arguments to pytest ARE test selections. The marker itself only
            // exists to introduce them, so it goes too.
            break;
        }
        if VALUE_TAKING_OPTIONS.contains(&arg.as_str()) {
            cmd.push(arg.clone());
            if let Some(value) = args.get(i + 1) {
                cmd.push(value.clone());
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if arg.starts_with('-') {
            cmd.push(arg.clone());
        }
        // Anything else is a positional test selection: replaced, not extended,
        // by the generated file pushed below.
        i += 1;
    }

    cmd.push(test_file.to_string());
    cmd.push("--json-report".to_string());
    cmd.push("--json-report-file=report.json".to_string());
    cmd.push("-q".to_string());
    cmd
}

/// Python interpreter to use for test execution.
/// [VANTA-WHY] Some repos ship their own venv with heavy deps. Set
/// VANTA_PYTHON to that venv's python. Defaults to `python3` from PATH.
pub fn python_exe() -> String {
    std::env::var("VANTA_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Wall-clock cap for a single test run, in seconds.
///
/// [VANTA-WHY] 120s by default: importing torch/transformers can take 10-20s, so
/// a tight cap would misreport a slow import as a timeout. `VANTA_TEST_TIMEOUT_SECS`
/// lowers it so a test can drive a REAL timeout — a hanging test actually killed
/// by the backend — without waiting two minutes. The timeout signal path
/// (`ExecutionResult.timed_out` → `TestExecutionDetails.timed_out` →
/// Infrastructure → reset-and-rerun) can only be exercised end-to-end when the
/// timeout really fires.
pub fn test_timeout_secs() -> u64 {
    std::env::var("VANTA_TEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(120)
}

/// Timeout for PATCHED-side execution. Tests may lower only this cap so the
/// real timeout path is exercised without making BATE's baseline load-sensitive.
pub fn patched_test_timeout_secs() -> u64 {
    std::env::var("VANTA_PATCHED_TEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(test_timeout_secs)
}

/// Detect the Python import path for a repo (Spec §7 "Import paths" row).
///
/// Rules:
///   - `src/<pkg>/...` → `<pkg>` (PEP 517 src-layout; conftest adds `src/`).
///   - `<pkg>/…` with `__init__.py` → `<pkg>` (flat layout with a package dir).
///   - files at repo root → None (import modules directly).
pub fn detect_import_path(source_files: &[SourceFile], _repo_root: &Path) -> Option<String> {
    for sf in source_files {
        if let Some(rest) = sf.path.strip_prefix("src/") {
            if let Some(pkg) = rest.split('/').next() {
                if !pkg.is_empty() {
                    return Some(pkg.to_string());
                }
            }
        }
    }
    for sf in source_files {
        let parts: Vec<&str> = sf.path.split('/').collect();
        if parts.len() >= 2 && parts[1] == "__init__.py" {
            return Some(parts[0].to_string());
        }
    }
    None
}

// ============================================================================
// Dependency self-healing
// ============================================================================

/// Probe the sandbox for Python modules pytest cannot import.
///
/// [VANTA-WHY] Manifest detection tells us what the repo DECLARED; tests can
/// import more. We ask pytest to COLLECT the generated tests inside the VM —
/// collection imports them, so a missing third-party import surfaces as
/// "No module named 'X'".
pub async fn probe_missing_modules(
    original_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
) -> Vec<String> {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if write_source_files(sb.root(), original_source).is_err() {
        sb.destroy();
        return Vec::new();
    }
    let _ = write_conftest(sb.root());
    for (i, t) in tests.iter().enumerate() {
        let _ = fs::write(sb.root().join(format!("test_probe_{i}.py")), &t.test_source);
    }
    let mut cmd = test_cmd.as_cmd_vec();
    cmd.push("--collect-only".to_string());
    cmd.push("-q".to_string());
    let refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let output = sb.execute(&refs, Path::new(""), 120).await;
    sb.destroy();
    match output {
        Ok(e) => {
            let mut combined = e.stdout;
            combined.push('\n');
            combined.push_str(&e.stderr);
            crate::sandbox::deps::parse_missing_modules(&combined)
        }
        Err(_) => Vec::new(),
    }
}

/// Self-healing dependency resolution.
///
/// [VANTA-WHY] Probes the tests for undeclared imports, installs them into a
/// checkpoint DERIVED from the current one, and switches the sandbox to it.
/// Bounded to 3 rounds. Skipped when auto-deps is disabled or shuru is absent.
pub async fn heal_dependencies(
    original_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
    import_hint: Option<&str>,
) {
    if std::env::var("VANTA_SHURU_AUTO_DEPS").as_deref() == Ok("0") {
        return;
    }
    if !crate::sandbox::shuru_available() {
        return;
    }

    let mut parent =
        std::env::var("VANTA_SHURU_CHECKPOINT").unwrap_or_else(|_| "vanta-python".into());
    let mut parent_disk: u64 = std::env::var("VANTA_SHURU_DISK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::sandbox::deps::BASE_DISK_MB);

    for _ in 0..3 {
        let missing: Vec<String> = probe_missing_modules(original_source, tests, test_cmd)
            .await
            .into_iter()
            // A missing *repo* submodule (e.g. "heretic.foo") is a bad test,
            // not a missing dependency — never try to pip install the repo.
            .filter(|m| import_hint != Some(m.as_str()))
            .collect();
        if missing.is_empty() {
            return;
        }

        println!(
            "  ⟳ VM deps: undeclared imports detected: {}",
            missing.join(", ")
        );

        match crate::sandbox::shuru::ShuruBackend::ensure_derived_checkpoint(
            &parent,
            parent_disk,
            &missing,
        ) {
            Ok(Some((name, disk))) => {
                println!("  ⟳ VM deps: derived checkpoint {name} (disk {disk} MB)");
                parent = name;
                parent_disk = disk;
                // SAFETY: single-threaded CLI; no other thread reads env concurrently.
                unsafe {
                    std::env::set_var("VANTA_SHURU_CHECKPOINT", &parent);
                    std::env::set_var("VANTA_SHURU_DISK_MB", parent_disk.to_string());
                }
            }
            Ok(None) => return,
            Err(e) => {
                eprintln!("  ⚠ VM deps self-heal failed: {e}");
                return;
            }
        }
    }
    eprintln!("  ⚠ VM deps self-heal hit the round limit — some imports may stay missing");
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::{
        NativeSuiteReport, RUFF_ARGS, StaticDiagnostic, build_single_test_cmd,
        copy_static_configs, interpreter_for, is_native_test_path, native_outcome,
        native_regressions, native_suite_collapsed, native_tree_with_original_tests,
        new_diagnostics, parse_mypy_line, parse_native_report, parse_pytest_report_detailed,
        parse_ruff_line, ruff_is_error_class, run_single_test, run_tests_parallel,
        static_tool_runner, write_source_files,
    };
    use crate::agents::attacker::GeneratedTest;
    use crate::core::{TestIntent, TestResult};
    use std::collections::HashMap;
    use crate::pipeline::candidate::TestCommand;
    use crate::sandbox::ExecutionResult;
    use std::fs;

    fn generated_test(name: &str, assertion: &str) -> GeneratedTest {
        GeneratedTest {
            test_name: name.to_string(),
            test_source: format!("from calc import add\n\ndef {name}():\n    {assertion}\n"),
            intent: TestIntent::BehaviorPreservation,
            reason: "exercise the same repo code through both harness paths".to_string(),
            issue_quote: None,
            relabeled: false,
        }
    }

    #[test]
    fn write_source_files_preserves_nested_tree_structure_and_contents() {
        let source = concat!(
            "# FILE: pkg/__init__.py\n",
            "from .core import add\n",
            "\n\n",
            "# FILE: pkg/core.py\n",
            "def add(a, b):\n",
            "    return a + b\n",
            "\n",
        );
        let dir = tempfile::tempdir().unwrap();

        write_source_files(dir.path(), source).unwrap();

        let mut paths: Vec<String> = fs::read_dir(dir.path().join("pkg"))
            .unwrap()
            .map(|entry| format!("pkg/{}", entry.unwrap().file_name().to_string_lossy()))
            .collect();
        paths.sort();

        assert_eq!(paths, vec!["pkg/__init__.py", "pkg/core.py"]);
        assert_eq!(
            fs::read_to_string(dir.path().join("pkg/__init__.py")).unwrap(),
            "from .core import add\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("pkg/core.py")).unwrap(),
            "def add(a, b):\n    return a + b\n\n"
        );
    }

    #[tokio::test]
    async fn baseline_and_patched_paths_match_on_identical_source() {
        let source = "# FILE: calc.py\ndef add(a, b):\n    return a + b\n";
        let test_cmd = TestCommand::default();
        let tests = vec![
            (
                "test_add_passes",
                generated_test("test_add_passes", "assert add(2, 3) == 5"),
                TestResult::Passed,
            ),
            (
                "test_add_fails",
                generated_test("test_add_fails", "assert add(2, 3) == 6"),
                TestResult::Failed,
            ),
        ];

        for (name, test, expected) in tests {
            let baseline = run_single_test(source, &test.test_source, &test_cmd).await;
            let refs = vec![&test];
            let patched = run_tests_parallel(source, &refs, &test_cmd).await.unwrap();
            let patched_result = patched.get(&test.test_name).cloned();

            assert_eq!(
                baseline, expected,
                "{name} should produce the expected baseline result"
            );
            assert_eq!(
                patched_result,
                Some(expected.clone()),
                "{name} should match between BATE and patched-run paths when the source is identical"
            );
        }
    }

    #[tokio::test]
    async fn semantic_patch_reports_assertion_failures_not_static_errors() {
        let patched_source = "# FILE: calc.py\ndef add(a, b):\n    return a - b\n";
        let test_cmd = TestCommand::default();
        let test = generated_test("test_behavior_preservation", "assert add(2, 3) == 5");
        let refs = vec![&test];
        let patched = run_tests_parallel(patched_source, &refs, &test_cmd)
            .await
            .unwrap();

        assert_eq!(
            patched.get(&test.test_name),
            Some(&TestResult::Failed),
            "a semantic behavior change should stay an assertion failure, not become Error/Static"
        );
    }

    // ---- native-suite gate ----------------------------------------------------
    //
    // [VANTA-WHY] The heretic run went 4 tests → 0 tests and printed
    // "0 regression(s)": a tree that does not import collected no tests, and a
    // filter over an empty list finds nothing wrong with it. The walk is over the
    // ORIGINAL side, and a test that passed there and is ABSENT from the patched
    // report is a regression — that is the case these pin.

    fn native_report(tests: &[(&str, TestResult)]) -> NativeSuiteReport {
        NativeSuiteReport {
            tests: tests
                .iter()
                .map(|(name, result)| ((*name).to_string(), result.clone()))
                .collect(),
            failure_outputs: HashMap::new(),
            ran: true,
            exit_code: Some(0),
            stderr_tail: String::new(),
        }
    }

    #[test]
    fn a_suite_that_collapsed_to_zero_tests_is_four_regressions() {
        let original = native_report(&[
            ("test_a", TestResult::Passed),
            ("test_b", TestResult::Passed),
            ("test_c", TestResult::Passed),
            ("test_d", TestResult::Passed),
        ]);
        let patched = native_report(&[]);

        assert_eq!(
            native_regressions(&original.tests, &patched.tests).len(),
            4,
            "4 tests → 0 tests is four regressions, not a clean bill of health"
        );
        assert!(
            native_suite_collapsed(&original, &patched),
            "the console must be able to say the suite COLLAPSED, not just list names"
        );
    }

    #[test]
    fn a_suite_that_kept_its_tests_has_not_collapsed() {
        let original = native_report(&[("test_a", TestResult::Passed)]);
        let patched = native_report(&[("test_a", TestResult::Passed)]);

        assert!(native_regressions(&original.tests, &patched.tests).is_empty());
        assert!(!native_suite_collapsed(&original, &patched));
    }

    #[test]
    fn a_suite_that_never_ran_is_not_read_as_collapsed() {
        let original = NativeSuiteReport::not_run();
        let patched = native_report(&[]);

        assert!(
            !native_suite_collapsed(&original, &patched),
            "a gate that could not run must not be reported as a collapse — that would \
             invent a finding out of a missing measurement"
        );
    }

    // ---- the native tree is graded by the ORIGINAL tests ---------------------
    //
    // [VANTA-WHY] The 888b6931 hole: the native run collected the patch's OWN
    // edited test file, so the weakened test passed and the Gatekeeper approved a
    // patch the original test rejects. These pin that the patch's edits to
    // existing tests never reach the native tree.

    #[test]
    fn a_collected_test_file_is_recognised_by_every_name_pytest_uses() {
        for path in [
            "test_core.py",
            "tests/test_core.py",
            "pkg/tests/test_deep.py",
            "core_test.py",
            "tests/helpers.py",
            "conftest.py",
            "tests/conftest.py",
        ] {
            assert!(is_native_test_path(path), "{path} is owned by the test suite");
        }
        for path in ["calc.py", "src/heretic/plugin.py", "latest.py"] {
            assert!(
                !is_native_test_path(path),
                "{path} is not a test file and must not be pinned back"
            );
        }
    }

    #[test]
    fn a_patch_that_weakens_an_existing_test_is_graded_by_the_original_test() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x * 2\n".to_string()),
            (
                "tests/test_core.py".to_string(),
                "def test_scale():\n    assert scale(3) == 6\n".to_string(),
            ),
        ]);
        let mut patched = original.clone();
        patched.insert(
            "calc.py".to_string(),
            "def scale(x):\n    return x * 2 + 1\n".to_string(),
        );
        patched.insert(
            "tests/test_core.py".to_string(),
            "def test_scale():\n    assert True\n".to_string(),
        );

        let tree = native_tree_with_original_tests(&original, &patched, &[]);

        assert_eq!(
            tree.files["tests/test_core.py"],
            original["tests/test_core.py"],
            "the native run must grade the patch with the ORIGINAL test, not the weakened one"
        );
        assert_eq!(
            tree.files["calc.py"], patched["calc.py"],
            "the patched source must be what the native run executes"
        );
        assert_eq!(tree.modified_existing_tests, vec!["tests/test_core.py"]);
        assert!(tree.added_test_files.is_empty());
        assert!(tree.removed_existing_tests.is_empty());
    }

    #[test]
    fn a_patch_added_conftest_and_test_file_never_reach_the_native_tree() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x * 2\n".to_string(),
        )]);
        let mut patched = original.clone();
        patched.insert(
            "conftest.py".to_string(),
            "collect_ignore = ['tests/test_core.py']\n".to_string(),
        );
        patched.insert(
            "tests/test_new.py".to_string(),
            "def test_new():\n    assert True\n".to_string(),
        );

        let tree = native_tree_with_original_tests(&original, &patched, &[]);

        assert!(
            !tree.files.contains_key("conftest.py"),
            "a patch-added conftest.py can deselect the very tests that grade the patch"
        );
        assert!(
            !tree.files.contains_key("tests/test_new.py"),
            "a test file the patch added is not part of the original baseline"
        );
        assert_eq!(
            tree.added_test_files,
            vec!["conftest.py".to_string(), "tests/test_new.py".to_string()],
            "added test files are reported as info"
        );
    }

    #[test]
    fn a_deleted_test_file_is_restored_for_the_native_run() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x * 2\n".to_string()),
            (
                "tests/test_core.py".to_string(),
                "def test_scale():\n    assert scale(3) == 6\n".to_string(),
            ),
        ]);
        // The cascade leaves a deleted file at its original contents, so the
        // deletion is only visible in the diff.
        let patched = original.clone();
        let deleted = vec!["tests/test_core.py".to_string()];

        let tree = native_tree_with_original_tests(&original, &patched, &deleted);

        assert_eq!(
            tree.files["tests/test_core.py"], original["tests/test_core.py"],
            "a deleted test must still run against the patched source"
        );
        assert_eq!(tree.removed_existing_tests, vec!["tests/test_core.py"]);
    }

    // ---- subTest parsing (item 9) --------------------------------------------
    //
    // [VANTA-WHY] pytest-json-report reports a `unittest.TestCase.subTest` item
    // with outcome `"subtests passed"`. The old parser matched only
    // `"passed"/"failed"/"error"` and silently dropped everything else, so the
    // heretic suite's three subTest tests vanished: Vanta counted 4 where pytest
    // ran 5 on master, and 12 where pytest ran 15 on the C patch. The undercount
    // was not cosmetic — the dropped item was missing from the ORIGINAL side
    // `native_regressions` walks, so a patch that broke it could never be
    // reported. These pin that the item is counted.

    #[test]
    fn a_passing_subtest_item_is_counted_as_passed() {
        assert_eq!(
            native_outcome("subtests passed"),
            Some(TestResult::Passed),
            "pytest reports a passing subTest item as \"subtests passed\", not \"passed\""
        );
        assert_eq!(native_outcome("failed"), Some(TestResult::Failed));
        assert_eq!(native_outcome("error"), Some(TestResult::Failed));
        assert_eq!(
            native_outcome("skipped"),
            None,
            "a skip is neither a pass nor a failure and must not be counted as either"
        );
    }

    #[test]
    fn parse_native_report_no_longer_drops_subtest_items() {
        let dir = tempfile::tempdir().unwrap();
        let report_path = dir.path().join("report.json");
        fs::write(
            &report_path,
            r#"{"tests":[
                {"nodeid":"tests/test_config.py::C::test_plain","outcome":"passed"},
                {"nodeid":"tests/test_config.py::C::test_subtests","outcome":"subtests passed"},
                {"nodeid":"tests/test_config.py::C::test_broken","outcome":"failed"},
                {"nodeid":"tests/test_config.py::C::test_skipped","outcome":"skipped"}
            ]}"#,
        )
        .unwrap();
        let exec = ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(1),
            wall_time_ms: 0,
            timed_out: false,
        };

        let report = parse_native_report(&report_path, &exec);

        assert!(report.ran, "a readable report means the suite ran");
        assert_eq!(
            report.tests.len(),
            3,
            "the subTest item must be counted; a skipped test must not"
        );
        assert_eq!(
            report
                .tests
                .iter()
                .find(|(name, _)| name.ends_with("test_subtests"))
                .map(|(_, result)| result.clone()),
            Some(TestResult::Passed),
            "the passing subTest item must survive parsing as a Passed test"
        );
    }

    // ---- static findings: error-class only ------------------------------------

    #[test]
    fn only_error_class_ruff_codes_may_block_a_review_round() {
        assert!(ruff_is_error_class("F821"), "undefined name is code that cannot run");
        assert!(ruff_is_error_class("E999"), "ruff's own syntax error");
        assert!(!ruff_is_error_class("E401"), "import hygiene is a style nit");
        assert!(!ruff_is_error_class("B008"), "flake8-bugbear style");
        assert!(!ruff_is_error_class("BLE001"), "broad-except style");
        assert!(!ruff_is_error_class("UP032"), "pyupgrade style");
        assert!(!ruff_is_error_class("SIM108"), "simplify style");
        assert!(!ruff_is_error_class("TRY003"), "tryceratops style");
    }

    /// A new static finding must carry its exact site for the evidence record,
    /// yet that site must NOT be part of the finding's identity — otherwise a
    /// patch that merely shifts a line would read as introducing a fresh error.
    #[test]
    fn a_new_static_finding_keeps_its_site_but_not_as_its_identity() {
        let ruff = parse_ruff_line("pkg/core.py:12:5: F821 Undefined name `x`").unwrap();
        assert_eq!(ruff.line, Some(12), "the reported line must be retained");
        assert_eq!(ruff.file, "pkg/core.py");
        assert_eq!(ruff.code, "F821");

        let mypy = parse_mypy_line("pkg/core.py:88: error: bad return  [return-value]").unwrap();
        assert_eq!(mypy.line, Some(88), "mypy's line must be retained");
        assert_eq!(mypy.code, "return-value");

        let baseline = vec![ruff.clone()];
        let moved = StaticDiagnostic {
            line: Some(40),
            ..ruff.clone()
        };
        assert!(
            new_diagnostics(&baseline, &[moved]).is_empty(),
            "the same finding moved down the file is not a NEW finding"
        );

        let different = StaticDiagnostic {
            code: "F841".to_string(),
            ..ruff.clone()
        };
        assert_eq!(
            new_diagnostics(&baseline, &[different]).len(),
            1,
            "a genuinely different finding in the same file is still new"
        );
    }

    /// The ruff invocation must ask for the concise report.
    ///
    /// [VANTA-WHY] Item 10 follow-up: ruff's default "full" report is
    /// multi-line, so `parse_ruff_line` read ZERO findings from every run no
    /// matter what the tree contained. The flag is what makes the parser's
    /// contract hold, so its absence is a silent blindness bug, not a style.
    #[test]
    fn ruff_is_asked_for_the_report_the_parser_can_read() {
        assert!(
            RUFF_ARGS.contains(&"--output-format=concise"),
            "without concise output every ruff finding parses to nothing"
        );

        let concise = "pkg/core.py:3:5: F821 Undefined name `x`";
        assert_eq!(
            parse_ruff_line(concise).map(|d| d.code),
            Some("F821".to_string()),
            "the concise line is exactly the parser's input"
        );

        let full = "F821 Undefined name `x`\n  --> pkg/core.py:3:5\n   |\n 3 | x = x\n";
        assert!(
            full.lines().filter_map(parse_ruff_line).next().is_none(),
            "the default full report carries no parsable one-line finding"
        );
    }

    // ---- item 10: the static step must run like the repo does ---------------

    /// The static tools must run under the interpreter the TESTS run under —
    /// that is the only one with the repo's dependencies. Session 40e84874 was
    /// rejected because mypy ran under the host's `python3` and reported
    /// `import torch` as `import-not-found`.
    #[test]
    fn the_static_interpreter_follows_the_test_command() {
        let heretic = TestCommand::from_shell(
            "env PYTHONPATH=src /Users/kyu-yi/Documents/Trae/default/heretic/.venv/bin/python -m pytest -q",
        );
        assert_eq!(
            interpreter_for(&heretic),
            "/Users/kyu-yi/Documents/Trae/default/heretic/.venv/bin/python",
            "the venv python hides behind `env` and a `PYTHONPATH=src` assignment, \
             not behind the host's bare python3"
        );

        assert_eq!(
            interpreter_for(&TestCommand::from_shell("python3 -m pytest")),
            "python3"
        );

        // A command that names no interpreter at all must fall back rather than
        // pick a token out of thin air (`pytest` is not an interpreter).
        assert_eq!(
            interpreter_for(&TestCommand::from_shell("pytest -q")),
            super::python_exe()
        );
    }

    /// The tool must RUN where it is installed, which is not always where the
    /// project lives: heretic's venv has `ruff` but no `mypy`.
    #[test]
    fn a_static_tool_runs_where_it_is_installed() {
        let host = super::python_exe();
        // Every CPython can import `sys`, so a real interpreter is used as-is.
        assert_eq!(static_tool_runner(&host, "sys"), host.clone());
        // An interpreter that cannot import the tool must not be used to run it,
        // or mypy would fail with `No module named mypy` and report NOTHING.
        assert_eq!(
            static_tool_runner("/nonexistent/venv/bin/python", "mypy"),
            host
        );
    }

    /// Only the config files the repo actually has may be copied — inventing an
    /// empty `ruff.toml` would CHANGE ruff's behaviour instead of reproducing it.
    #[test]
    fn only_the_configs_that_exist_reach_the_sandbox() {
        let repo = tempfile::tempdir().unwrap();
        let sandbox = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("pyproject.toml"),
            "[tool.ruff.lint]\nignore = [\"B023\"]\n",
        )
        .unwrap();
        fs::write(repo.path().join("ruff.toml"), "line-length = 100\n").unwrap();

        let mut copied = copy_static_configs(repo.path(), sandbox.path());
        copied.sort();

        assert_eq!(
            copied,
            vec!["pyproject.toml".to_string(), "ruff.toml".to_string()]
        );
        assert_eq!(
            fs::read_to_string(sandbox.path().join("pyproject.toml")).unwrap(),
            "[tool.ruff.lint]\nignore = [\"B023\"]\n",
            "the repo's own ruff config must reach the sandbox verbatim"
        );
        assert!(
            !sandbox.path().join("mypy.ini").exists(),
            "a config the repo does not have must not be invented"
        );
    }

    /// `import-not-found` / `import-untyped` are facts about the analysis
    /// environment, not defects in the candidate.
    #[test]
    fn a_missing_dependency_is_an_environment_finding() {
        let line = "tests/test_auto_rank.py:5: error: Cannot find implementation or library \
                    stub for module named \"torch\"  [import-not-found]";
        let missing = parse_mypy_line(line).expect("a mypy error line parses");
        assert_eq!(missing.code, "import-not-found");
        assert_eq!(missing.file, "tests/test_auto_rank.py");
        assert_eq!(missing.line, Some(5));
        assert!(missing.is_environment_finding());

        let untyped = StaticDiagnostic {
            file: "src/heretic/data.py".to_string(),
            line: Some(1),
            code: "import-untyped".to_string(),
            message: "Library stubs not installed for \"requests\"".to_string(),
        };
        assert!(untyped.is_environment_finding());

        let real = StaticDiagnostic {
            file: "src/heretic/modifiers/abliteration.py".to_string(),
            line: Some(440),
            code: "assignment".to_string(),
            message: "Incompatible types in assignment".to_string(),
        };
        assert!(
            !real.is_environment_finding(),
            "a genuine type error is a finding about the code"
        );
    }

    // ---- the generated test is graded by ITS OWN nodeid -----------------------
    //
    // [VANTA-WHY] The grading bug. A `--test-cmd` that named a test path
    // (`python -m pytest tests`) made the one-test run collect the repository's
    // own suite FIRST, and the reader took `report["tests"].first()` as the
    // generated test's result. That first entry was a repo test — which passes —
    // so every BugReproduction test was recorded as "passed on the buggy
    // original", BATE scrapped it, and no candidate could ever be rejected for
    // failing to reproduce the bug. These pin both halves of the fix: the run
    // REPLACES the selection with the generated file, and the reader selects only
    // that file's result and fails closed when it is absent.

    fn buggy_exec() -> ExecutionResult {
        ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(1),
            wall_time_ms: 0,
            timed_out: false,
        }
    }

    fn written_report(tests_json: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        fs::write(&path, format!("{{\"tests\":[{tests_json}]}}")).unwrap();
        (dir, path)
    }

    #[test]
    fn a_report_where_repo_tests_come_first_grades_the_generated_test() {
        let (_dir, path) = written_report(concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"tests/test_core.py::test_b","outcome":"passed"},"#,
            r#"{"nodeid":"test_bate.py::test_reproduces_bug","outcome":"failed"}"#
        ));

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");

        assert_eq!(
            details.result,
            TestResult::Failed,
            "the generated test FAILED on the buggy original; the passing repo test that \
             happened to come first must not be read as its result"
        );
        assert!(
            details.report_path_exists,
            "a graded test reached a verdict, even though it failed"
        );
    }

    #[test]
    fn a_report_missing_the_generated_nodeid_fails_closed() {
        let (_dir, path) = written_report(concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"tests/test_core.py::test_b","outcome":"passed"}"#
        ));

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");

        assert_eq!(
            details.result,
            TestResult::Error,
            "a report without the generated nodeid is a harness error, never a Passed"
        );
        assert!(
            !details.report_path_exists,
            "the generated test was never graded, so this tree did not reach a verdict"
        );
    }

    #[test]
    fn a_generated_file_named_with_a_leading_dot_slash_still_matches() {
        let (_dir, path) =
            written_report(r#"{"nodeid":"./test_run.py::test_x","outcome":"passed"}"#);

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_run.py");

        assert_eq!(
            details.result,
            TestResult::Passed,
            "rootdir-relative nodeids may carry a `./` prefix and must still match"
        );
    }

    #[test]
    fn a_test_cmd_that_names_a_path_has_it_replaced_not_extended() {
        let cmd = build_single_test_cmd(
            &TestCommand::from_shell("python3 -m pytest tests -q"),
            "test_bate.py",
        );

        assert!(
            !cmd.iter().any(|arg| arg == "tests"),
            "the repository's own test path must be REPLACED, not extended: {cmd:?}"
        );
        assert_eq!(
            cmd,
            vec![
                "python3",
                "-m",
                "pytest",
                "-q",
                "test_bate.py",
                "--json-report",
                "--json-report-file=report.json",
                "-q",
            ],
            "the interpreter and flags are kept; only the test selection is replaced"
        );
    }

    #[test]
    fn a_value_taking_flag_keeps_its_value_and_is_not_read_as_a_path() {
        let cmd = build_single_test_cmd(
            &TestCommand::from_shell("python3 -m pytest -k test_add -m slow tests"),
            "test_bate.py",
        );

        assert!(
            cmd.windows(2).any(|w| w == ["-k", "test_add"]),
            "`-k`'s value is a marker expression, not a path to drop: {cmd:?}"
        );
        assert!(
            cmd.windows(2).any(|w| w == ["-m", "slow"]),
            "`-m`'s value is a marker expression, not a path to drop: {cmd:?}"
        );
        assert!(
            !cmd.iter().any(|arg| arg == "tests"),
            "the trailing path is still a selection and must be replaced: {cmd:?}"
        );
        assert!(cmd.iter().any(|arg| arg == "test_bate.py"), "{cmd:?}");
    }

    #[test]
    fn a_test_cmd_without_a_path_keeps_its_flags_and_gains_the_generated_file() {
        let cmd = build_single_test_cmd(&TestCommand::default(), "test_run.py");

        assert_eq!(
            cmd,
            vec![
                "python3",
                "-m",
                "pytest",
                "test_run.py",
                "--json-report",
                "--json-report-file=report.json",
                "-q",
            ],
            "with no selection in the command the behaviour is unchanged, minus the append"
        );
    }
}
