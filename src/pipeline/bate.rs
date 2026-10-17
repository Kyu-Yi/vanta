// ============================================================================
// BATE — Baseline-Aware Test Evaluation (Spec §7 Phase 1)
// ============================================================================
//
// [VANTA-WHY] BATE's job: BEFORE running tests against the patched repo,
// validate that each Attacker-generated test actually has the expected result
// on the ORIGINAL (unpatched) repo. This is the "test is a real test" check.
//
// Spec §7 Phase 1 intent→baseline mapping:
//   BugReproduction      → baseline must FAIL (test catches the bug)
//   FeatureAcceptance    → baseline must FAIL (feature doesn't exist yet)
//   BehaviorPreservation → baseline must PASS  (existing behavior unchanged)
//
// If a test's actual baseline result differs from its expected baseline →
// BateOutcome::Mismatch → test is scrapped (can't trust it).
// Max 3 retries per test before scrapping.
// ============================================================================

use crate::agents::attacker::{
    GeneratedTest, IntentRelabel, attacker_revise_test, relabel_test_to_issue_scope,
};
use crate::config::LlmConfig;
use crate::core::{TestIntent, TestResult};
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::harness;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ============================================================================
// BateOutcome — result of validating a test against the baseline
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BateOutcome {
    /// Test's actual baseline result matches the expected baseline for its intent.
    Valid,
    /// Test's actual baseline result does NOT match expected. Scrapped.
    Mismatch(String),
}

// ============================================================================
// Expected baseline result per intent (Spec §7 Phase 1)
// ============================================================================
pub fn expected_baseline(intent: &TestIntent) -> TestResult {
    match intent {
        // BugReproduction test MUST fail on the original buggy code.
        TestIntent::BugReproduction => TestResult::Failed,
        // FeatureAcceptance test MUST fail on original (feature doesn't exist).
        TestIntent::FeatureAcceptance => TestResult::Failed,
        // BehaviorPreservation test MUST pass on original (existing behavior).
        TestIntent::BehaviorPreservation => TestResult::Passed,
    }
}

// ============================================================================
// validate_test_against_baseline — the core BATE check (single attempt)
// ============================================================================
pub fn validate_test_against_baseline(
    test_name: &str,
    intent: &TestIntent,
    actual_baseline: TestResult,
) -> BateOutcome {
    let expected = expected_baseline(intent);
    debug!(
        test = test_name,
        ?intent,
        ?expected,
        ?actual_baseline,
        "BATE validating test against baseline"
    );

    if expected == actual_baseline {
        info!(test = test_name, "BATE: test VALID — baseline matches intent");
        BateOutcome::Valid
    } else {
        let reason = format!(
            "BATE mismatch: test '{test_name}' intent={intent:?} expects baseline={expected:?} \
             but got {actual_baseline:?}."
        );
        warn!(test = test_name, expected = ?expected, actual = ?actual_baseline, "BATE: test MISMATCH");
        BateOutcome::Mismatch(reason)
    }
}

// ============================================================================
// validate_with_retry — retry a test up to max_attempts before scrapping
// ============================================================================
//
// [VANTA-WHY] Spec §7 Phase 1: "max 3 retries per test before scrapping".
// Tests can be flaky (timing, ordering, resource contention). If a test
// mismatches on attempt 1 but matches on attempt 2 or 3, it's still Valid.
// Only when ALL attempts mismatch do we scrap the test.
//
// [SYNTAX] Async function taking an async closure. Each call to run_fn is
// awaited, allowing the test runner to be async (e.g. sandbox.execute).
pub async fn validate_with_retry<F, Fut>(
    test_name: &str,
    intent: &TestIntent,
    max_attempts: u32,
    mut run_fn: F,
) -> BateOutcome
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = TestResult>,
{
    let mut last_mismatch: Option<String> = None;

    for attempt in 1..=max_attempts {
        debug!(test = test_name, attempt, max_attempts, "BATE retry loop: running baseline test");
        let actual = run_fn().await;
        match validate_test_against_baseline(test_name, intent, actual) {
            BateOutcome::Valid => {
                info!(test = test_name, attempt, "BATE: test VALID on attempt {attempt}");
                return BateOutcome::Valid;
            }
            BateOutcome::Mismatch(reason) => {
                warn!(test = test_name, attempt, "BATE: mismatch on attempt {attempt}, retrying");
                last_mismatch = Some(reason);
            }
        }
    }

    // All attempts mismatched → scrap the test.
    let reason = format!(
        "{} (scrapped after {} attempts)",
        last_mismatch.unwrap_or_else(|| "unknown mismatch".into()),
        max_attempts
    );
    warn!(test = test_name, max_attempts, "BATE: test SCRAPPED after {max_attempts} mismatches");
    BateOutcome::Mismatch(reason)
}

// ============================================================================
// BateReport — summary of validating a batch of tests
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BateReport {
    pub valid_tests: Vec<String>,
    pub scrapped_tests: Vec<(String, String)>, // (test_name, reason)
}

impl BateReport {
    pub fn all_valid(&self) -> bool {
        self.scrapped_tests.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "BATE report: {} valid, {} scrapped",
            self.valid_tests.len(),
            self.scrapped_tests.len()
        )
    }
}

// ============================================================================
// BateRun — the outcome of a full BATE phase (validate + revise)
// ============================================================================
//
// [VANTA-WHY] BATE has TWO retry layers (Spec §7):
//   Layer 1 (validate_with_retry): re-run the SAME test up to 3× in a fresh
//     sandbox to rule out flakiness.
//   Layer 2 (attacker_revise_test): if the baseline still disagrees with the
//     test's declared intent, send the test back to the Attacker with the
//     mismatch reason. Max 3 revisions.
//   Only after BOTH layers exhaust is the test scrapped.
//
// The caller must run `valid_tests` (not the originals) against the patched
// tree: a test that survived BATE after revision is a DIFFERENT test body than
// the one the Attacker first produced, and running the original would verify a
// candidate against a probe that was never validated.
pub struct BateRun {
    pub report: BateReport,
    pub valid_tests: Vec<GeneratedTest>,
    pub relabel_events: Vec<IntentRelabel>,
}

/// Run the BATE phase over a batch of Attacker tests against the ORIGINAL code.
///
/// Returns the report (for discard-rate logging) and the surviving tests using
/// their possibly-revised source.
pub async fn run_bate(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
) -> BateRun {
    let mut valid_tests: Vec<GeneratedTest> = Vec::new();
    let mut scrapped_tests: Vec<(String, String)> = Vec::new();
    let mut relabel_events: Vec<IntentRelabel> = Vec::new();

    for test in tests {
        // [VANTA-WHY] Spec §7 "Flaky tests" row: run the baseline twice; a test
        // that flips PASS↔FAIL between the two runs would silently corrupt
        // EIR/ECR, so it is scrapped as flaky before it can join the shared set.
        let r1 = harness::run_single_test(original_source, &test.test_source, test_cmd).await;
        let r2 = harness::run_single_test(original_source, &test.test_source, test_cmd).await;
        if r1 != r2 {
            scrapped_tests.push((
                test.test_name.clone(),
                format!("FLAKY: baseline disagreed with itself (run1={r1:?}, run2={r2:?})"),
            ));
            continue;
        }

        let mut current_test = test.clone();
        let mut final_outcome: Option<BateOutcome> = None;

        // Layer 2: up to 3 Attacker revisions when the baseline mismatches intent.
        for revision in 0..=3 {
            let src = original_source.to_string();
            let test_source = current_test.test_source.clone();
            let cmd_template = test_cmd.clone();
            let outcome = validate_with_retry(
                &current_test.test_name,
                &current_test.intent,
                3, // Layer 1: re-run in a fresh sandbox up to 3× for flakiness
                || {
                    let src = src.clone();
                    let test_source = test_source.clone();
                    let cmd_template = cmd_template.clone();
                    async move { harness::run_single_test(&src, &test_source, &cmd_template).await }
                },
            )
            .await;

            match outcome {
                BateOutcome::Valid => {
                    final_outcome = Some(BateOutcome::Valid);
                    break;
                }
                BateOutcome::Mismatch(reason) => {
                    if revision < 3 {
                        info!(
                            test = %current_test.test_name,
                            revision = revision + 1,
                            "BATE mismatch → sending test back to Attacker for revision"
                        );
                        match attacker_revise_test(
                            cfg,
                            issue_text,
                            original_source,
                            &current_test,
                            &reason,
                        )
                        .await
                        {
                            Ok(revised) => {
                                let (revised, relabel) =
                                    relabel_test_to_issue_scope(&revised, issue_text);
                                if let Some(relabel) = relabel {
                                    relabel_events.push(relabel);
                                }
                                current_test = revised;
                            }
                            Err(e) => {
                                warn!("attacker_revise_test failed: {e} — scrapping test");
                                final_outcome = Some(BateOutcome::Mismatch(reason));
                                break;
                            }
                        }
                    } else {
                        final_outcome = Some(BateOutcome::Mismatch(reason));
                    }
                }
            }
        }

        match final_outcome {
            Some(BateOutcome::Valid) => valid_tests.push(current_test),
            Some(BateOutcome::Mismatch(reason)) => {
                scrapped_tests.push((current_test.test_name.clone(), reason))
            }
            None => scrapped_tests.push((
                current_test.test_name.clone(),
                "unknown BATE outcome".into(),
            )),
        }
    }

    BateRun {
        report: BateReport {
            valid_tests: valid_tests.iter().map(|t| t.test_name.clone()).collect(),
            scrapped_tests,
        },
        valid_tests,
        relabel_events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bug_reproduction_requires_failing_baseline() {
        // A bug-reproduction test that passes on the original code is useless.
        let outcome = validate_test_against_baseline(
            "huge_cart",
            &TestIntent::BugReproduction,
            TestResult::Passed,
        );
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
    }

    #[test]
    fn bug_reproduction_with_failing_baseline_is_valid() {
        let outcome = validate_test_against_baseline(
            "huge_cart",
            &TestIntent::BugReproduction,
            TestResult::Failed,
        );
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[test]
    fn behavior_preservation_requires_passing_baseline() {
        let outcome = validate_test_against_baseline(
            "empty_cart",
            &TestIntent::BehaviorPreservation,
            TestResult::Passed,
        );
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[test]
    fn feature_acceptance_requires_failing_baseline() {
        let outcome = validate_test_against_baseline(
            "new_feature",
            &TestIntent::FeatureAcceptance,
            TestResult::Failed,
        );
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[tokio::test]
    async fn retry_loop_scraps_after_all_mismatches() {
        // Test that ALWAYS returns Passed for a BugReproduction intent.
        // After 3 attempts it should be scrapped.
        let outcome = validate_with_retry(
            "flaky_fail",
            &TestIntent::BugReproduction,
            3,
            || async { TestResult::Passed },
        )
        .await;
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
    }

    #[tokio::test]
    async fn retry_loop_passes_on_later_attempt() {
        // Test that fails twice then passes. Should be Valid (not scrapped).
        use std::cell::RefCell;
        let attempts = RefCell::new(0);
        let outcome = validate_with_retry(
            "recovers",
            &TestIntent::BugReproduction,
            3,
            || async {
                let mut a = attempts.borrow_mut();
                *a += 1;
                if *a < 3 {
                    TestResult::Passed // mismatch on attempts 1, 2
                } else {
                    TestResult::Failed // matches on attempt 3
                }
            },
        )
        .await;
        assert_eq!(outcome, BateOutcome::Valid);
        assert_eq!(*attempts.borrow(), 3);
    }
}
