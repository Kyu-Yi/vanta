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
        return DiagnosticResult {
            failure_class: Some(FailureClass::Logic),
            reason: "Test ran but assertion failed — patch has a logic bug".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "assertion_failed",
            })),
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
