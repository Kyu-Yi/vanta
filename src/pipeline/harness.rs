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
            let cmd = build_test_cmd(&cmd_template, "test_run.py");
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
    let cmd = build_test_cmd(test_cmd, "test_bate.py");
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let sandbox_root = sb.root().display().to_string();
    let exec = sb.execute(&cmd_refs, Path::new(""), timeout_secs).await;
    let result = match exec {
        Ok(e) => {
            parse_pytest_report_detailed(sb.root().join("report.json"), &e, Some(sandbox_root))
        }
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
    /// False when no report was produced (missing pytest/plugin, timeout,
    /// crash). A gate that could not run must never be read as "no regressions".
    pub ran: bool,
}

/// Run the repository's own test suite against a source tree.
///
/// The source tree includes the repo's `tests/`, so pytest discovers them from
/// the sandbox root; no generated test file is written.
pub async fn run_native_suite(source: &str, test_cmd: &TestCommand) -> NativeSuiteReport {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => {
            return NativeSuiteReport {
                tests: Vec::new(),
                ran: false,
            }
        }
    };
    if write_source_files(sb.root(), source).is_err() {
        sb.destroy();
        return NativeSuiteReport {
            tests: Vec::new(),
            ran: false,
        };
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
        Err(_) => NativeSuiteReport {
            tests: Vec::new(),
            ran: false,
        },
    };
    sb.destroy();
    report
}

/// Parse a pytest JSON report into per-test outcomes.
fn parse_native_report(report_path: &Path, exec: &ExecutionResult) -> NativeSuiteReport {
    if exec.timed_out {
        return NativeSuiteReport {
            tests: Vec::new(),
            ran: false,
        };
    }
    let Ok(text) = fs::read_to_string(report_path) else {
        return NativeSuiteReport {
            tests: Vec::new(),
            ran: false,
        };
    };
    let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) else {
        return NativeSuiteReport {
            tests: Vec::new(),
            ran: false,
        };
    };
    let mut tests = Vec::new();
    if let Some(arr) = report["tests"].as_array() {
        for t in arr {
            let nodeid = t["nodeid"].as_str().unwrap_or_default().to_string();
            let result = match t["outcome"].as_str() {
                Some("passed") => TestResult::Passed,
                // "failed" = assertion failure; "error" = exception/setup error.
                Some("failed") | Some("error") => TestResult::Failed,
                // A skipped test is neither a pass nor a failure signal.
                _ => continue,
            };
            if !nodeid.is_empty() {
                tests.push((nodeid, result));
            }
        }
    }
    NativeSuiteReport { tests, ran: true }
}

/// Nodeids that PASS on the original and FAIL on the patch — the repo's own
/// suite reporting a regression. Tests already failing on the original are
/// excluded: the patch did not break them.
pub fn native_regressions(
    original: &[(String, TestResult)],
    patched: &[(String, TestResult)],
) -> Vec<String> {
    let orig: HashMap<&str, &TestResult> =
        original.iter().map(|(n, r)| (n.as_str(), r)).collect();
    patched
        .iter()
        .filter(|(n, r)| {
            *r == TestResult::Failed && orig.get(n.as_str()).copied() == Some(&TestResult::Passed)
        })
        .map(|(n, _)| n.clone())
        .collect()
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

/// Parse pytest's JSON report into a single TestResult.
///
/// [VANTA-WHY] An "error" outcome (ImportError/AttributeError during the test
/// body) means the test did NOT pass — equivalent to "failed". `TestResult::Error`
/// is reserved for INFRASTRUCTURE failures (timeout, missing report, crash).
pub fn parse_pytest_report(report_path: PathBuf, exec: &ExecutionResult) -> TestResult {
    parse_pytest_report_detailed(report_path, exec, None).result
}

pub fn parse_pytest_report_detailed(
    report_path: PathBuf,
    exec: &ExecutionResult,
    sandbox_root: Option<String>,
) -> TestExecutionDetails {
    if exec.timed_out {
        return TestExecutionDetails {
            result: TestResult::Error,
            stdout: exec.stdout.clone(),
            stderr: exec.stderr.clone(),
            timed_out: true,
            report_path_exists: report_path.exists(),
            sandbox_root,
        };
    }
    if report_path.exists() {
        if let Ok(text) = fs::read_to_string(&report_path) {
            if let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(tests) = report["tests"].as_array() {
                    if let Some(t) = tests.first() {
                        let result = match t["outcome"].as_str() {
                            Some("passed") => TestResult::Passed,
                            // "failed" = assertion failure; "error" = exception
                            // during test body. Both mean the test didn't pass.
                            Some("failed") | Some("error") => TestResult::Failed,
                            _ => TestResult::Error,
                        };
                        return TestExecutionDetails {
                            result,
                            stdout: exec.stdout.clone(),
                            stderr: exec.stderr.clone(),
                            timed_out: false,
                            report_path_exists: true,
                            sandbox_root,
                        };
                    }
                }
            }
        }
    }
    TestExecutionDetails {
        result: TestResult::Error,
        stdout: exec.stdout.clone(),
        stderr: exec.stderr.clone(),
        timed_out: false,
        report_path_exists: false,
        sandbox_root,
    }
}

// ============================================================================
// Static analysis (ruff + mypy)
// ============================================================================

/// Count diagnostic lines for `needle` across BOTH streams.
///
/// [VANTA-WHY] ruff and mypy write diagnostics to STDOUT, never stderr.
/// Counting stderr alone made every static run look clean, silently disabling
/// the Static class. Scan both streams.
pub fn count_diagnostics(stdout: &str, stderr: &str, needle: &str) -> usize {
    stdout
        .lines()
        .chain(stderr.lines())
        .filter(|line| line.contains(needle))
        .count()
}

/// Run static analysis on both original and patched; return new-error flags.
pub async fn run_static_diff(original: &str, patched: &str) -> (bool, bool) {
    let (orig_ruff, _, orig_mypy, _) = run_static_raw(original).await;
    let (patch_ruff, _, patch_mypy, _) = run_static_raw(patched).await;
    (patch_ruff > orig_ruff, patch_mypy > orig_mypy)
}

pub async fn run_static_raw(source: &str) -> (usize, String, usize, String) {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return (0, String::new(), 0, String::new()),
    };
    let _ = write_source_files(sb.root(), source);

    let py = python_exe();
    let ruff = sb
        .execute(
            &[py.as_str(), "-m", "ruff", "check", "."],
            Path::new(""),
            30,
        )
        .await;
    let (ruff_count, ruff_err) = match ruff {
        Ok(r) => (
            count_diagnostics(&r.stdout, &r.stderr, ".py:"),
            format!("{}\n{}", r.stdout, r.stderr),
        ),
        Err(_) => (0, String::new()),
    };

    let mypy = sb
        .execute(&[py.as_str(), "-m", "mypy", "."], Path::new(""), 60)
        .await;
    let (mypy_count, mypy_err) = match mypy {
        Ok(r) => (
            count_diagnostics(&r.stdout, &r.stderr, "error:"),
            format!("{}\n{}", r.stdout, r.stderr),
        ),
        Err(_) => (0, String::new()),
    };

    sb.destroy();
    (ruff_count, ruff_err, mypy_count, mypy_err)
}

// ============================================================================
// Command construction
// ============================================================================

/// Build the sandbox command vector from the user's TestCommand + a test file.
/// [VANTA-WHY] The user supplies the base runner (e.g. `pytest` or
/// `python -m pytest`); we append the generated test file + json-report flags.
pub fn build_test_cmd(test_cmd: &TestCommand, test_file: &str) -> Vec<String> {
    let mut cmd = test_cmd.as_cmd_vec();
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
    use super::{run_single_test, run_tests_parallel, write_source_files};
    use crate::agents::attacker::GeneratedTest;
    use crate::core::{TestIntent, TestResult};
    use crate::pipeline::candidate::TestCommand;
    use std::fs;

    fn generated_test(name: &str, assertion: &str) -> GeneratedTest {
        GeneratedTest {
            test_name: name.to_string(),
            test_source: format!("from calc import add\n\ndef {name}():\n    {assertion}\n"),
            intent: TestIntent::BehaviorPreservation,
            reason: "exercise the same repo code through both harness paths".to_string(),
            issue_quote: None,
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
}
