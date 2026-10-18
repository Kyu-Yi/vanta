// ============================================================================
// DIAGNOSTICIAN — 4-way FailureClass triage (Spec §7 Phase 2)
// ============================================================================
//
// [VANTA-WHY] When a test FAILS on the patched repo, the Diagnostician must
// classify WHY it failed so the right repair strategy is used:
//
//   Infrastructure → sandbox/network/timeout (retry, not a code bug)
//   Static         → new lint/type errors introduced by the patch
//   Logic          → runs but wrong behavior (the patch has a logic bug)
//   Unknown        → can't classify (default before analysis)
//
// Diagnostic signal: ruff + mypy diff vs baseline. If original had 0 ruff
// errors and patched has new ones → Static. If patch adds type errors → Static.
// If pytest dies during import/collect without new static signals, treat that as
// Infrastructure so the harness does not blame the model for a broken tree.
// If tests run but assert fails → Logic.
// ============================================================================

use crate::agents::attacker::{GeneratedTest, attacker_simplify_test};
use crate::config::LlmConfig;
use crate::core::{FailureClass, TestResult};
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::harness::{self, TestExecutionDetails};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, warn};

// ============================================================================
// DIAGNOSTICIAN_PROMPT_VERSION — the run-identity tag for the Diagnostician
// ============================================================================
//
// [VANTA-WHY] The Diagnostician is rule-based today (no prompt text of its own),
// but its Unknown path calls the Attacker's simplify prompt. `session.start`
// lists every agent prompt so that a future Diagnostician-side prompt change is
// visible in the log rather than silently counted as "no change"; that change
// must bump this tag.
pub const DIAGNOSTICIAN_PROMPT_VERSION: &str = "1";

/// Cap on the pytest output saved with a failing test (the fix-list's ~4 KB).
/// A full traceback for one test is far below this; the cap only bounds a
/// pathological case (e.g. a dump of a huge fixture) so evidence.json stays
/// readable.
pub const FAILURE_OUTPUT_LIMIT: usize = 4096;

/// First `limit` bytes of `s`, cut on a char boundary, with a marker when cut.
pub fn truncate_output(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let mut end = limit;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n… [truncated: {} of {} bytes shown]",
        &s[..end],
        end,
        s.len()
    )
}

/// Locate a pytest failure's innermost frame.
///
/// [VANTA-WHY] pytest ends a traceback with a bare `path.py:LINE: ExceptionType`
/// line — the frame where the exception was actually raised. That triple is what
/// lets a later step tell a failure in the CODE UNDER TEST ("the patch has a
/// bug") from one raised in the TEST's own objects ("the test is broken"):
/// `'FakeTensor' object has no attribute 'shape'` is the test's fake, not the
/// patch. Parsing it from the saved output keeps the classification auditable.
///
/// Returns `(file, line, exception_type)`, each `None` when the output has no
/// recognisable frame (e.g. the test never got that far).
pub fn parse_failure_site(output: &str) -> (Option<String>, Option<u32>, Option<String>) {
    for raw in output.lines().rev() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, ':');
        let (file, lineno, rest) = match (parts.next(), parts.next(), parts.next()) {
            (Some(f), Some(l), Some(r)) => (f.trim(), l.trim(), r.trim()),
            _ => continue,
        };
        if !file.ends_with(".py") {
            continue;
        }
        let Ok(line_no) = lineno.parse::<u32>() else {
            continue;
        };
        // Exception types are Capitalised; this also rejects the `in helper`
        // frames pytest interleaves into a traceback.
        let exc = rest
            .split(|c: char| c == ':' || c.is_whitespace())
            .next()
            .unwrap_or("");
        let capitalised = exc.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        if !capitalised {
            continue;
        }
        return (Some(file.to_string()), Some(line_no), Some(exc.to_string()));
    }
    (None, None, None)
}

/// The signature of a failure: `(exception_type, file, line)` from its output.
///
/// [VANTA-WHY] Two runs are "the same failure" when they raise the same
/// exception at the same site. Comparing signatures — not merely "both trees
/// failed" — is what lets a patch-insensitive failure be told apart from a
/// coincidence: two DIFFERENT bugs that happen to both fail must stay Logic.
fn failure_signature(exec: &TestExecutionDetails) -> Option<(String, Option<String>, Option<u32>)> {
    let output = format!("{}\n{}", exec.stdout, exec.stderr);
    let (file, line, exception_type) = parse_failure_site(&output);
    exception_type.map(|e| (e, file, line))
}

/// Does the same test fail the SAME way on the unpatched tree?
///
/// [VANTA-WHY] Item 2 of the heretic fix list: a failure that reproduces
/// identically on the original tree carries no information about the patch, so
/// it must not reject. `false` whenever either side has no parseable signature
/// — an unreadable failure must not be silently reclassified as a test defect.
fn is_patch_insensitive(patched: &TestExecutionDetails, original: &TestExecutionDetails) -> bool {
    match (failure_signature(patched), failure_signature(original)) {
        (Some(p), Some(o)) => p == o,
        _ => false,
    }
}

// ============================================================================
// DiagnosticResult — output of Diagnostician
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub failure_class: Option<FailureClass>,
    pub reason: String,
    /// Optional structured payload (e.g. ruff diff, mypy errors).
    pub structured_payload: Option<serde_json::Value>,
}

// ============================================================================
// Diagnostician inputs — what we feed the 4-way classifier
// ============================================================================
#[derive(Debug, Clone, Default)]
pub struct DiagnosticInputs {
    /// Did the test runner exit with a non-zero code due to import/syntax errors?
    pub import_or_syntax_error: bool,
    /// ruff stderr diff (patched vs original) — does patched introduce new lint errors?
    pub ruff_new_errors: bool,
    /// mypy stderr diff — does patched introduce new type errors?
    pub mypy_new_errors: bool,
    /// Wall time hit the timeout?
    pub timed_out: bool,
    /// Test ran to completion but assertion failed?
    pub assertion_failed: bool,
    /// Network connectivity check failed (network killed mid-run)?
    /// [VANTA-WHY] M2 EXIT TEST: "Kill the network mid-run and it classifies as
    /// Infrastructure, not a model error." If network_alive() returns false,
    /// any failure is Infrastructure regardless of other signals.
    pub network_down: bool,
    /// Raw ruff stderr from patched run (for structured_payload).
    pub ruff_stderr: String,
    /// Raw mypy stderr from patched run (for structured_payload).
    pub mypy_stderr: String,
    /// Raw pytest stdout from the failing test run.
    pub pytest_stdout: String,
    /// Raw pytest stderr from the failing test run.
    pub pytest_stderr: String,
    /// Whether pytest failed before producing report.json.
    pub pytest_report_missing: bool,
    /// The failing test's own source, saved with the evidence so a reader can
    /// see WHAT failed without hunting for the GeneratedTest artifact.
    pub test_source: String,
}

// ============================================================================
// diagnose — the 4-way classifier (Spec §7 Phase 2)
// ============================================================================
pub fn diagnose(inputs: &DiagnosticInputs) -> DiagnosticResult {
    debug!(?inputs, "Diagnostician: classifying failure");

    // Priority order: Infrastructure > Static > Logic > Unknown
    // [VANTA-WHY] Infrastructure failures are environmental — retrying might
    // fix them. Static failures are the patch breaking the build. Logic is the
    // patch having wrong behavior. Unknown is the fallback.

    // Network-down takes highest priority — if the network is killed mid-run,
    // the failure is environmental, not a code bug. (M2 EXIT TEST requirement.)
    if inputs.network_down {
        info!("Diagnostician → Infrastructure (network down)");
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Network connectivity failed mid-run — infrastructure issue".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "network_down",
            })),
        };
    }

    if inputs.timed_out {
        info!("Diagnostician → Infrastructure (timeout)");
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Test timed out — likely infrastructure/resource issue".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "timeout",
            })),
        };
    }

    if inputs.ruff_new_errors || inputs.mypy_new_errors {
        info!("Diagnostician → Static (new lint/type errors)");
        // [VANTA-WHY] Populate structured_payload with the actual ruff/mypy
        // stderr so M3's Creator can see exactly what static errors the patch
        // introduced. This is the "structured JSON diagnostic payload back to
        // Creator" requirement from Spec §7 Phase 2.
        let mut payload = serde_json::json!({
            "signal": "new_static_errors",
            "ruff_new_errors": inputs.ruff_new_errors,
            "mypy_new_errors": inputs.mypy_new_errors,
        });
        if inputs.ruff_new_errors && !inputs.ruff_stderr.is_empty() {
            payload["ruff_stderr"] = serde_json::Value::String(inputs.ruff_stderr.clone());
        }
        if inputs.mypy_new_errors && !inputs.mypy_stderr.is_empty() {
            payload["mypy_stderr"] = serde_json::Value::String(inputs.mypy_stderr.clone());
        }
        return DiagnosticResult {
            failure_class: Some(FailureClass::Static),
            reason: "Patch introduces new ruff/mypy errors".into(),
            structured_payload: Some(payload),
        };
    }

    if inputs.import_or_syntax_error {
        info!(
            "Diagnostician → Infrastructure (pytest import/collect error without new static diagnostics)"
        );
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Pytest failed before executing the test, but no new ruff/mypy diagnostics were introduced".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "pytest_import_or_collect_error",
                "pytest_stdout": inputs.pytest_stdout,
                "pytest_stderr": inputs.pytest_stderr,
                "pytest_report_missing": inputs.pytest_report_missing,
            })),
        };
    }

    if inputs.assertion_failed {
        info!("Diagnostician → Logic (assertion failed)");
        // [VANTA-WHY] This payload used to be `{"signal":"assertion_failed"}` and
        // nothing more: no test source, no pytest output, no traceback. A reader
        // of evidence.json then could not tell WHY the patch was blamed — the
        // exact defect the heretic runs exposed. Save the test's own source, the
        // failure output (capped), and the innermost frame so the claim "the
        // patch has a logic bug" is auditable from the record alone.
        let full_output = format!("{}\n{}", inputs.pytest_stdout, inputs.pytest_stderr);
        let (frame_file, frame_line, exception_type) = parse_failure_site(&full_output);
        let payload = serde_json::json!({
            "signal": "assertion_failed",
            "test_source": inputs.test_source,
            "failure_output": truncate_output(&full_output, FAILURE_OUTPUT_LIMIT),
            "exception_type": exception_type,
            "frame_file": frame_file,
            "frame_line": frame_line,
        });
        return DiagnosticResult {
            failure_class: Some(FailureClass::Logic),
            reason: "Test ran but assertion failed — patch has a logic bug".into(),
            structured_payload: Some(payload),
        };
    }

    info!("Diagnostician → Unknown");
    DiagnosticResult {
        failure_class: Some(FailureClass::Unknown),
        reason: "Could not classify failure with available signals".into(),
        structured_payload: None,
    }
}

// ============================================================================
// diagnose_failure — classify ONE failing test WITH the Spec §7 Phase-2
// remediation workflow (reset-and-rerun for Infrastructure, simplify for Unknown)
// ============================================================================
//
// [VANTA-WHY] `diagnose` above is the pure classifier. This is the workflow that
// ACTS on the class, which the M3 loop had dropped:
//   Infrastructure → reset the sandbox and rerun once. A single transient fault
//     is discovered and cleared here rather than being recorded against the model.
//   Unknown        → rerun in a fresh sandbox, then ask the Attacker to simplify
//     the test, then rerun the simplified one. Only if it is STILL Unknown is the
//     Attacker-generated test QUARANTINED (excluded from EIR/ECR, reason logged)
//     rather than silently counted as a model error.
//   Static / Logic → no remediation here (M3 feeds them back to the Creator).
pub async fn diagnose_failure(
    test: &GeneratedTest,
    execution: &TestExecutionDetails,
    ruff_new_errors: bool,
    mypy_new_errors: bool,
    ruff_stderr: &str,
    mypy_stderr: &str,
    network_down: bool,
    patched_source: &str,
    cfg: &LlmConfig,
    original_source: &str,
    test_cmd: &TestCommand,
) -> DiagnosticResult {
    let force_unknown = std::env::var("VANTA_TEST_FORCE_UNKNOWN").ok().as_deref() == Some("1");
    // [VANTA-WHY] The FIRST classification sees the real execution: whether the
    // run timed out, whether pytest produced report.json, and the actual streams.
    // Throwing those away (as an earlier cut did) downgrades a timeout — a
    // genuine Infrastructure failure — to an unclassifiable Unknown, and an
    // Unknown triggers an Attacker call it never needed.
    let classify = |exec: &TestExecutionDetails| -> DiagnosticResult {
        if force_unknown {
            return DiagnosticResult {
                failure_class: Some(FailureClass::Unknown),
                reason: "Forced Unknown classification for regression testing".into(),
                structured_payload: Some(json!({
                    "signal": "forced_unknown",
                })),
            };
        }
        diagnose(&DiagnosticInputs {
            import_or_syntax_error: exec.result == TestResult::Error,
            ruff_new_errors,
            mypy_new_errors,
            timed_out: exec.timed_out,
            assertion_failed: exec.result == TestResult::Failed,
            network_down,
            ruff_stderr: ruff_stderr.to_string(),
            mypy_stderr: mypy_stderr.to_string(),
            pytest_stdout: exec.stdout.clone(),
            pytest_stderr: exec.stderr.clone(),
            pytest_report_missing: !exec.report_path_exists,
            test_source: test.test_source.clone(),
        })
    };

    let enrich_with_sandbox_roots =
        |mut diag: DiagnosticResult, initial_root: Option<String>, rerun_root: Option<String>| {
            let mut payload = diag.structured_payload.take().unwrap_or_else(|| json!({}));
            if let Some(obj) = payload.as_object_mut() {
                if let Some(root) = initial_root.clone() {
                    obj.insert(
                        "initial_sandbox_root".into(),
                        serde_json::Value::String(root),
                    );
                }
                if let Some(root) = rerun_root.clone() {
                    obj.insert("rerun_sandbox_root".into(), serde_json::Value::String(root));
                }
                obj.insert(
                    "fresh_sandbox_spawned".into(),
                    serde_json::Value::Bool(
                        initial_root.is_some()
                            && rerun_root.is_some()
                            && initial_root != rerun_root,
                    ),
                );
            }
            diag.structured_payload = Some(payload);
            diag
        };

    let diag = classify(execution);

    // [VANTA-WHY] Item 2 of the heretic fix list. A failure that reproduces
    // IDENTICALLY on the ORIGINAL tree says nothing about the patch. B v2's
    // generated test built `HereticSettings(model=...)`, whose CLI settings
    // source parses pytest's own argv and raises `SystemExit` — on the unpatched
    // tree exactly as on the patch. Blaming the patch there is a false finding
    // about the model. When the failing test raises the same exception at the
    // same file:line on the original tree, the TEST is defective, not the
    // candidate: classify it TestDefect, never Logic.
    if diag.failure_class == Some(FailureClass::Logic) {
        if let Some(patched_sig) = failure_signature(execution) {
            let original = harness::run_single_test_with_timeout(
                original_source,
                &test.test_source,
                test_cmd,
                harness::patched_test_timeout_secs(),
            )
            .await;
            if is_patch_insensitive(execution, &original) {
                let (exception, file, line) = &patched_sig;
                let site = format!(
                    "{}:{}",
                    file.clone().unwrap_or_else(|| "?".into()),
                    line.map(|l| l.to_string()).unwrap_or_else(|| "?".into()),
                );
                tracing::info!(
                    test = %test.test_name,
                    "failure reproduces on the ORIGINAL tree → TestDefect (patch-insensitive)"
                );
                return enrich_with_sandbox_roots(
                    DiagnosticResult {
                        failure_class: Some(FailureClass::TestDefect),
                        reason: format!(
                            "Test fails identically on the unpatched tree ({exception} at {site}) — \
                             patch-insensitive, so the failure is a property of the test, not the \
                             patch. Sent back to the Attacker for revision."
                        ),
                        structured_payload: Some(json!({
                            "signal": "patch_insensitive_failure",
                            "exception_type": exception,
                            "frame_file": file,
                            "frame_line": line,
                            "original_sandbox_root": original.sandbox_root.clone(),
                        })),
                    },
                    execution.sandbox_root.clone(),
                    original.sandbox_root.clone(),
                );
            }
        }
    }

    match diag.failure_class {
        Some(FailureClass::Infrastructure) => {
            // Spec §7 Phase 2: "Infrastructure: reset backend and rerun."
            tracing::info!(test = %test.test_name, "Infrastructure → resetting sandbox and rerunning");
            let rerun = harness::run_single_test_with_timeout(
                patched_source,
                &test.test_source,
                test_cmd,
                harness::patched_test_timeout_secs(),
            )
            .await;
            match rerun.result {
                TestResult::Passed => {
                    // Flaky infra — the test actually passes on rerun.
                    enrich_with_sandbox_roots(
                        DiagnosticResult {
                            failure_class: None,
                            reason: "Infrastructure failure resolved on rerun".into(),
                            structured_payload: None,
                        },
                        execution.sandbox_root.clone(),
                        rerun.sandbox_root.clone(),
                    )
                }
                _ => {
                    let mut re = classify(&rerun);
                    re.reason = format!("{} (after sandbox reset + rerun)", re.reason);
                    enrich_with_sandbox_roots(
                        re,
                        execution.sandbox_root.clone(),
                        rerun.sandbox_root.clone(),
                    )
                }
            }
        }
        Some(FailureClass::Unknown) => {
            // Spec §7 Phase 2 Unknown handling.
            tracing::info!(test = %test.test_name, "Unknown → rerunning in fresh sandbox");

            // Step 1: fresh-sandbox rerun.
            let rerun = harness::run_single_test_with_timeout(
                patched_source,
                &test.test_source,
                test_cmd,
                harness::patched_test_timeout_secs(),
            )
            .await;
            if rerun.result == TestResult::Passed {
                return enrich_with_sandbox_roots(
                    DiagnosticResult {
                        failure_class: None,
                        reason: "Unknown resolved on fresh-sandbox rerun".into(),
                        structured_payload: None,
                    },
                    execution.sandbox_root.clone(),
                    rerun.sandbox_root.clone(),
                );
            }
            let rerun_diag = classify(&rerun);
            if !matches!(rerun_diag.failure_class, Some(FailureClass::Unknown)) {
                return enrich_with_sandbox_roots(
                    rerun_diag,
                    execution.sandbox_root.clone(),
                    rerun.sandbox_root.clone(),
                );
            }

            // Step 2: ask the Attacker to simplify the test.
            tracing::info!(test = %test.test_name, "Unknown → asking Attacker to simplify");
            match attacker_simplify_test(cfg, original_source, test).await {
                Ok(simplified) => {
                    // Step 3: run the simplified test.
                    let simp_result = harness::run_single_test_with_timeout(
                        patched_source,
                        &simplified.test_source,
                        test_cmd,
                        harness::patched_test_timeout_secs(),
                    )
                    .await;
                    if simp_result.result == TestResult::Passed {
                        return enrich_with_sandbox_roots(
                            DiagnosticResult {
                                failure_class: None,
                                reason: "Unknown resolved after Attacker simplification".into(),
                                structured_payload: None,
                            },
                            execution.sandbox_root.clone(),
                            simp_result.sandbox_root.clone(),
                        );
                    }
                    let simp_diag = classify(&simp_result);
                    if !matches!(simp_diag.failure_class, Some(FailureClass::Unknown)) {
                        return enrich_with_sandbox_roots(
                            simp_diag,
                            execution.sandbox_root.clone(),
                            simp_result.sandbox_root.clone(),
                        );
                    }
                }
                Err(e) => {
                    warn!("attacker_simplify_test failed: {e}");
                }
            }

            // Step 4: still Unknown → quarantine (Attacker-generated tests).
            // Repository/benchmark tests would escalate to human review instead.
            DiagnosticResult {
                failure_class: Some(FailureClass::Unknown),
                reason: "Unknown persisted after rerun + simplification → QUARANTINED (Attacker-generated test). For repo/benchmark tests this would escalate to human review.".into(),
                structured_payload: Some(serde_json::json!({
                    "action": "quarantine",
                    "evidence_source": "AttackerGenerated",
                    "initial_sandbox_root": execution.sandbox_root,
                    "note": "ExistingRepositoryTest and BenchmarkOracle tests escalate to human review instead of quarantine",
                })),
            }
        }
        // Static and Logic: no remediation in this phase (M3 feeds back to Creator).
        _ => diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_is_infrastructure() {
        let inputs = DiagnosticInputs {
            timed_out: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn import_or_syntax_error_without_new_static_signals_is_infrastructure() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn ruff_new_errors_is_static() {
        let inputs = DiagnosticInputs {
            ruff_new_errors: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Static));
    }

    #[test]
    fn assertion_failed_is_logic() {
        let inputs = DiagnosticInputs {
            assertion_failed: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Logic));
    }

    /// Item 1 of the heretic fix list: the assertion payload used to be a bare
    /// `{"signal":"assertion_failed"}` — no test source, no output, no site. The
    /// evidence must now let a reader audit WHY the patch was blamed.
    #[test]
    fn an_assertion_failure_saves_the_test_source_and_the_failure_site() {
        let inputs = DiagnosticInputs {
            assertion_failed: true,
            test_source: "def test_x():\n    assert f() == 5\n".to_string(),
            pytest_stdout: concat!(
                "============================= test session starts ==============================\n",
                "collected 1 item\n\n",
                "test_x.py::test_x\n",
                "\n",
                "    def test_x():\n",
                ">       assert f() == 5\n",
                "E       assert 6 == 5\n",
                "\n",
                "test_x.py:2: AssertionError\n",
            )
            .to_string(),
            ..Default::default()
        };

        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Logic));
        let payload = result
            .structured_payload
            .expect("the failing test's evidence must be saved, not just its class");
        assert_eq!(payload["signal"], json!("assertion_failed"));
        assert_eq!(payload["exception_type"], json!("AssertionError"));
        assert_eq!(payload["frame_file"], json!("test_x.py"));
        assert_eq!(payload["frame_line"], json!(2));
        assert!(
            payload["test_source"]
                .as_str()
                .unwrap()
                .contains("def test_x"),
            "the failing test's own source must ride with the evidence"
        );
        assert!(
            payload["failure_output"]
                .as_str()
                .unwrap()
                .contains("assert 6 == 5"),
            "pytest's failure output must ride with the evidence"
        );
    }

    #[test]
    fn a_huge_failure_output_is_truncated_with_a_marker() {
        let big = "x".repeat(FAILURE_OUTPUT_LIMIT * 3);
        let out = truncate_output(&big, FAILURE_OUTPUT_LIMIT);
        assert!(out.len() < big.len());
        assert!(out.contains("truncated"), "a cut must be visible: {out}");
    }

    #[test]
    fn the_innermost_frame_of_an_error_in_the_test_object_is_located() {
        // The 56eb4fd3 shape: the fake the TEST defined lacks an attribute.
        let output = "test_x.py:9: AttributeError: 'FakeTensor' object has no attribute 'shape'\n";
        let (file, line, exc) = parse_failure_site(output);
        assert_eq!(file.as_deref(), Some("test_x.py"));
        assert_eq!(line, Some(9));
        assert_eq!(exc.as_deref(), Some("AttributeError"));
    }

    fn exec_at(result: TestResult, output: &str) -> TestExecutionDetails {
        TestExecutionDetails {
            result,
            stdout: output.to_string(),
            stderr: String::new(),
            timed_out: false,
            report_path_exists: true,
            sandbox_root: Some("sandbox".to_string()),
        }
    }

    /// Item 2 of the heretic fix list: a failure that reproduces identically on
    /// the unpatched tree says nothing about the patch (B v2's HereticSettings
    /// raised SystemExit on both trees). Same exception at the same site →
    /// patch-insensitive; anything else → not.
    #[test]
    fn a_failure_that_reproduces_on_the_original_tree_is_patch_insensitive() {
        let patched = exec_at(TestResult::Failed, "test_x.py:9: SystemExit\n");
        let identical = patched.clone();
        assert!(
            is_patch_insensitive(&patched, &identical),
            "same exception at the same site on both trees is the test's own defect"
        );

        let other_site = exec_at(TestResult::Failed, "test_x.py:42: SystemExit\n");
        assert!(
            !is_patch_insensitive(&patched, &other_site),
            "same exception at a DIFFERENT site is not the same failure"
        );

        let other_exception = exec_at(TestResult::Failed, "test_x.py:9: AssertionError\n");
        assert!(
            !is_patch_insensitive(&patched, &other_exception),
            "a different exception type is a different failure"
        );

        let original_passed = exec_at(TestResult::Passed, "");
        assert!(
            !is_patch_insensitive(&patched, &original_passed),
            "a test that passes on the original is evidence about the patch"
        );
    }

    #[test]
    fn no_signals_is_unknown() {
        let inputs = DiagnosticInputs::default();
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Unknown));
    }

    #[test]
    fn network_down_is_infrastructure() {
        // M2 EXIT TEST: "Kill the network mid-run and it classifies as
        // Infrastructure, not a model error."
        let inputs = DiagnosticInputs {
            network_down: true,
            assertion_failed: true, // would otherwise be Logic
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn import_error_without_new_static_signals_is_not_static() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            pytest_stderr: "ModuleNotFoundError: No module named 'calc'".into(),
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
        assert_ne!(result.failure_class, Some(FailureClass::Static));
    }

    #[test]
    fn infrastructure_payload_captures_pytest_stderr() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            pytest_stderr: "ImportError while importing test module".into(),
            pytest_stdout: "collected 0 items / 1 error".into(),
            ..Default::default()
        };
        let result = diagnose(&inputs);
        let payload = result
            .structured_payload
            .expect("payload should be present");
        assert_eq!(payload["signal"], "pytest_import_or_collect_error");
        assert_eq!(
            payload["pytest_stderr"],
            "ImportError while importing test module"
        );
        assert_eq!(payload["pytest_stdout"], "collected 0 items / 1 error");
    }
}
