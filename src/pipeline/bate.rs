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
    GeneratedTest, IntentCheckRecord, IntentRelabel, attacker_revise_test, audit_issue_requires_change,
    audit_test_intent,
};
use crate::config::LlmConfig;
use crate::core::{TestIntent, TestResult};
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::harness;
use crate::pipeline::{REVISE_BUDGET_EXHAUSTED, ReviseBudget};
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
//
// [VANTA-WHY] Returns the LAST observed baseline alongside the outcome. BATE's
// relabel-before-revise rule (Fix 6) needs to know that the baseline PASSED — a
// BugReproduction that passes reproduced nothing — and that fact is only
// observable here, where the baseline actually ran. Reconstructing it from a
// `Mismatch` would confuse "passed" with "errored", and an errored baseline must
// not be relabelled into a preservation probe.
pub async fn validate_with_retry<F, Fut>(
    test_name: &str,
    intent: &TestIntent,
    max_attempts: u32,
    mut run_fn: F,
) -> (BateOutcome, TestResult)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = TestResult>,
{
    let mut last_mismatch: Option<String> = None;
    let mut last_actual = TestResult::Error;

    for attempt in 1..=max_attempts {
        debug!(test = test_name, attempt, max_attempts, "BATE retry loop: running baseline test");
        let actual = run_fn().await;
        last_actual = actual.clone();
        match validate_test_against_baseline(test_name, intent, actual) {
            BateOutcome::Valid => {
                info!(test = test_name, attempt, "BATE: test VALID on attempt {attempt}");
                return (BateOutcome::Valid, last_actual);
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
    (BateOutcome::Mismatch(reason), last_actual)
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
    /// One record per revised test the reviver produced, so a run's log shows the
    /// patch-blind intent audit ran on revised tests too — including a
    /// preservation revision the audit scrapped for contradicting the issue.
    pub intent_checks: Vec<IntentCheckRecord>,
}

/// What BATE should do with a BugReproduction test whose baseline mismatched.
///
/// [VANTA-WHY] Fix 6 — relabel before revise. A BugReproduction test that PASSES
/// on the original reproduced nothing: it asserts behaviour the original already
/// has. Sending it straight to revision asks the Attacker to "fix" a test that is
/// not broken. When the issue requires that behaviour to STAY, the test is a
/// legitimate preservation probe that was mislabelled, so it is relabelled and
/// admitted UNCHANGED. Only when the issue requires the behaviour to change (or
/// the auditor cannot say) does the ordinary revision path run.
///
/// [VANTA-WHY] The relabel only ever goes BugReproduction → BehaviorPreservation,
/// never the other way. A preservation test whose baseline FAILED must not
/// become a BugReproduction: that direction would launder a broken probe into
/// one that is allowed to fail on the original, which is the loophole.
///
/// [VANTA-WHY] `issue_requires_change` is an `Option` and the relabel fires ONLY
/// on an explicit `Some(false)`. A local model dropping the field must not be
/// able to promote a test: absent means UNKNOWN, and unknown fails closed to the
/// ordinary revision path (`Revise`). Unavailable auditors arrive here as `None`
/// for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MislabeledReproductionOutcome {
    /// The baseline passed and the issue explicitly requires that behaviour
    /// preserved: relabel to BehaviorPreservation and admit the test unchanged.
    RelabelToPreservation,
    /// Anything else — including an unknown answer: hand the test to the
    /// Attacker for revision as before.
    Revise,
}

pub fn mislabeled_reproduction_decision(
    intent: &TestIntent,
    baseline: TestResult,
    issue_requires_change: Option<bool>,
) -> MislabeledReproductionOutcome {
    if *intent == TestIntent::BugReproduction
        && baseline == TestResult::Passed
        && issue_requires_change == Some(false)
    {
        MislabeledReproductionOutcome::RelabelToPreservation
    } else {
        MislabeledReproductionOutcome::Revise
    }
}

/// Run the BATE phase over a batch of Attacker tests against the ORIGINAL code.
///
/// Returns the report (for discard-rate logging) and the surviving tests using
/// their possibly-revised source.
///
/// [VANTA-WHY] `grounded_audit` is §14.4's flag, threaded to the two intent
/// audits this phase performs (the relabel question and the audit of a revised
/// test). It only selects the auditor's prompt variant; with it `false` both calls
/// are byte-identical to v0.2's.
///
/// [VANTA-WHY] §14.14 — `revise_budget` is the attempt's SHARED ceiling on
/// Attacker remediation calls. It is borrowed rather than local so the
/// Diagnostician's Unknown-path simplification draws from the same counter:
/// "revise+simplify" is one budget per attempt, not two.
pub async fn run_bate(
    cfg: &LlmConfig,
    issue_text: &str,
    original_execution_source: &str,
    original_prompt_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
    grounded_audit: bool,
    revise_budget: &mut ReviseBudget,
) -> BateRun {
    let mut valid_tests: Vec<GeneratedTest> = Vec::new();
    let mut scrapped_tests: Vec<(String, String)> = Vec::new();
    let mut relabel_events: Vec<IntentRelabel> = Vec::new();
    let mut intent_checks: Vec<IntentCheckRecord> = Vec::new();

    for test in tests {
        // [VANTA-WHY] Spec §7 "Flaky tests" row: run the baseline twice; a test
        // that flips PASS↔FAIL between the two runs would silently corrupt
        // EIR/ECR, so it is scrapped as flaky before it can join the shared set.
        let r1 =
            harness::run_single_test(original_execution_source, &test.test_source, test_cmd).await;
        let r2 =
            harness::run_single_test(original_execution_source, &test.test_source, test_cmd).await;
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
            let src = original_execution_source.to_string();
            let test_source = current_test.test_source.clone();
            let cmd_template = test_cmd.clone();
            let (outcome, baseline_result) = validate_with_retry(
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
                    // [VANTA-WHY] Fix 6 — relabel before revise. A test labelled
                    // BugReproduction whose baseline PASSED is not a reproduction
                    // at all: it asserts behaviour the original already has.
                    // Before asking the Attacker to "fix" it, ask whether the
                    // issue requires that behaviour to stay. If it does, the
                    // test is a legitimate — merely mislabelled — preservation
                    // probe, so it is relabelled and admitted UNCHANGED. If the
                    // issue requires the behaviour to change, the ordinary
                    // revision path runs as before.
                    if current_test.intent == TestIntent::BugReproduction
                        && baseline_result == TestResult::Passed
                    {
                        let (requires_change, record) = audit_issue_requires_change(
                            cfg,
                            issue_text,
                            original_prompt_source,
                            &current_test,
                            grounded_audit,
                        )
                        .await;
                        intent_checks.push(record);
                        if requires_change.is_none() {
                            // [VANTA-WHY] Fail closed. A local model that drops
                            // the field must not be able to promote a test: an
                            // absent answer is UNKNOWN, and unknown never
                            // relabels. Take the ordinary revision path.
                            info!(
                                test = %current_test.test_name,
                                "audit omitted issue_requires_change, not relabeling"
                            );
                        }
                        if mislabeled_reproduction_decision(
                            &current_test.intent,
                            baseline_result.clone(),
                            requires_change,
                        ) == MislabeledReproductionOutcome::RelabelToPreservation
                        {
                            let mut relabeled = current_test.clone();
                            relabeled.intent = TestIntent::BehaviorPreservation;
                            relabeled.issue_quote = None;
                            // [VANTA-WHY] Item 6 — record that this test has
                            // spent its one relabel, so a later verdict can never
                            // flip it back.
                            relabeled.relabeled = true;
                            info!(
                                test = %current_test.test_name,
                                "relabeled BugReproduction→BehaviorPreservation (baseline passed, issue requires preservation)"
                            );
                            relabel_events.push(IntentRelabel {
                                test_name: relabeled.test_name.clone(),
                                from_intent: TestIntent::BugReproduction,
                                to_intent: TestIntent::BehaviorPreservation,
                                reason: "relabeled BugReproduction→BehaviorPreservation \
                                         (baseline passed, issue requires preservation)"
                                    .to_string(),
                            });
                            current_test = relabeled;
                            final_outcome = Some(BateOutcome::Valid);
                            break;
                        }
                    }
                    if revision < 3 {
                        // [VANTA-WHY] §14.14 — the attempt's SHARED revise+simplify
                        // budget. The per-test cap (`revision < 3`) bounds one
                        // test; this bounds the whole round, and a revision that
                        // returns a RENAMED test cannot escape it, because the
                        // counter belongs to the round and is not reset when the
                        // name changes. Spending it BEFORE the call is what makes
                        // "at most 4 revisions" true rather than "at most 4 that
                        // returned". When it is spent the test is scrapped and the
                        // round continues with the tests it has.
                        if !revise_budget.try_use() {
                            info!(
                                test = %current_test.test_name,
                                "BATE: the attempt's revise budget is spent — scrapping"
                            );
                            final_outcome = Some(BateOutcome::Mismatch(
                                REVISE_BUDGET_EXHAUSTED.to_string(),
                            ));
                            break;
                        }
                        info!(
                            test = %current_test.test_name,
                            revision = revision + 1,
                            "BATE mismatch → sending test back to Attacker for revision"
                        );
                        match attacker_revise_test(
                            cfg,
                            issue_text,
                            original_prompt_source,
                            &current_test,
                            &reason,
                        )
                        .await
                        {
                            Ok(revised) => {
                                // [VANTA-WHY] Item 6 — a revision is the same test
                                // under a new body, so it inherits the relabel
                                // ceiling: the test has already spent its one
                                // relabel, and a verdict that would relabel it
                                // again flips it back, which the audit scraps.
                                let mut revised = revised;
                                revised.relabeled = current_test.relabeled;
                                // [VANTA-WHY] The reviver can change what a test
                                // declares, so the revised test is audited too: a
                                // revision that claims FeatureAcceptance for a
                                // behaviour the issue never asked for must be
                                // demoted before it re-enters the baseline check.
                                let (audited, relabel, record) =
                                    audit_test_intent(
                                        cfg,
                                        issue_text,
                                        original_prompt_source,
                                        &revised,
                                        grounded_audit,
                                    )
                                        .await;
                                if let Some(relabel) = relabel {
                                    relabel_events.push(relabel);
                                }
                                // [VANTA-WHY] A preservation revision that asserts
                                // behaviour the ISSUE requires to change
                                // contradicts the issue. Revising again cannot
                                // help — the issue is the fixed point — so the
                                // test is scrapped, with the audit reason as the
                                // scrap reason.
                                let contradictory = record
                                    .as_ref()
                                    .filter(|r| r.scrapped)
                                    .map(|r| r.reason.clone());
                                if let Some(record) = record {
                                    intent_checks.push(record);
                                }
                                if let Some(reason) = contradictory {
                                    info!(
                                        test = %current_test.test_name,
                                        "BATE: revision contradicts the issue — scrapping"
                                    );
                                    final_outcome = Some(BateOutcome::Mismatch(format!(
                                        "revision contradicts the issue: {reason}"
                                    )));
                                    break;
                                }
                                match audited {
                                    Some(revised) => current_test = revised,
                                    None => {
                                        final_outcome = Some(BateOutcome::Mismatch(
                                            "intent audit scrapped the revised test".into(),
                                        ));
                                        break;
                                    }
                                }
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
        intent_checks,
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
        let (outcome, last) = validate_with_retry(
            "flaky_fail",
            &TestIntent::BugReproduction,
            3,
            || async { TestResult::Passed },
        )
        .await;
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
        assert_eq!(
            last,
            TestResult::Passed,
            "the observed baseline must ride along so BATE can tell 'passed' from 'errored'"
        );
    }

    #[tokio::test]
    async fn retry_loop_passes_on_later_attempt() {
        // Test that fails twice then passes. Should be Valid (not scrapped).
        use std::cell::RefCell;
        let attempts = RefCell::new(0);
        let (outcome, _last) = validate_with_retry(
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

    // ---- Fix 6: relabel before revise -----------------------------------------
    //
    // [VANTA-WHY] The decision is a pure function so the rule can be pinned
    // without an LLM: a BugReproduction whose baseline PASSED is only a
    // legitimate probe when the issue requires that behaviour to stay.

    #[test]
    fn a_preservation_style_test_mislabelled_bug_reproduction_is_relabelled_not_revised() {
        assert_eq!(
            mislabeled_reproduction_decision(
                &TestIntent::BugReproduction,
                TestResult::Passed,
                Some(false),
            ),
            MislabeledReproductionOutcome::RelabelToPreservation,
            "baseline passed and the issue explicitly requires preservation: keep the test, fix the label"
        );
    }

    #[test]
    fn a_bug_reproduction_the_issue_wants_changed_is_still_revised() {
        assert_eq!(
            mislabeled_reproduction_decision(
                &TestIntent::BugReproduction,
                TestResult::Passed,
                Some(true),
            ),
            MislabeledReproductionOutcome::Revise,
            "the issue requires the behaviour to change, so this probe is not a preservation test"
        );
    }

    #[test]
    fn an_errored_baseline_is_not_relabelled() {
        assert_eq!(
            mislabeled_reproduction_decision(
                &TestIntent::BugReproduction,
                TestResult::Error,
                Some(false),
            ),
            MislabeledReproductionOutcome::Revise,
            "an error is not a pass — it is not evidence the behaviour already holds"
        );
    }

    #[test]
    fn the_relabel_never_runs_the_other_way() {
        // [VANTA-WHY] The loophole direction. A preservation test whose baseline
        // FAILED must never be laundered into a BugReproduction, which is allowed
        // to fail on the original.
        assert_eq!(
            mislabeled_reproduction_decision(
                &TestIntent::BehaviorPreservation,
                TestResult::Failed,
                Some(false),
            ),
            MislabeledReproductionOutcome::Revise,
            "a failing preservation test must not become a BugReproduction"
        );
    }

    #[test]
    fn no_relabel_on_missing_issue_requires_change() {
        // [VANTA-WHY] Fail closed. The model omitting the field is exactly the
        // case that used to promote a test: `#[serde(default)]` filled in
        // `false`, which is the "preserve" answer. Absent must mean UNKNOWN, and
        // unknown must take the ordinary revision path — never relabel.
        assert_eq!(
            mislabeled_reproduction_decision(
                &TestIntent::BugReproduction,
                TestResult::Passed,
                None,
            ),
            MislabeledReproductionOutcome::Revise,
            "a dropped field is unknown, not an explicit preservation answer"
        );
    }
}
