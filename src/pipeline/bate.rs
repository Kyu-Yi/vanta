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
use crate::pipeline::harness::TestExecutionDetails;
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
/// [VANTA-WHY] §14.17 — the run returns the whole [`TestExecutionDetails`], not
/// just the `TestResult`. BATE's revise step is the one place a test is handed
/// back to the model with an account of what went wrong, and the account used to
/// be the CLASSIFIED string alone ("expects baseline=Failed but got Error"). A
/// model shown only the category cannot act on it: the overnight sanic session
/// shows a test rewritten 45 times while the model guessed what "ERROR" meant,
/// because the exception, its message and the traceback were thrown away here.
/// Carrying the details keeps one runner for both the verdict and the evidence.
pub async fn validate_with_retry<F, Fut>(
    test_name: &str,
    intent: &TestIntent,
    max_attempts: u32,
    mut run_fn: F,
) -> (BateOutcome, TestExecutionDetails)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = TestExecutionDetails>,
{
    let mut last_mismatch: Option<String> = None;
    // Only ever read when `max_attempts` is 0, which no caller passes.
    let mut last_details = TestExecutionDetails::from_rerun(TestResult::Error);

    for attempt in 1..=max_attempts {
        debug!(test = test_name, attempt, max_attempts, "BATE retry loop: running baseline test");
        let actual = run_fn().await;
        let result = actual.result.clone();
        last_details = actual;
        match validate_test_against_baseline(test_name, intent, result) {
            BateOutcome::Valid => {
                info!(test = test_name, attempt, "BATE: test VALID on attempt {attempt}");
                return (BateOutcome::Valid, last_details);
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
    (BateOutcome::Mismatch(reason), last_details)
}

// ============================================================================
// baseline_evidence — what the baseline ACTUALLY did, for the revise prompt
// ============================================================================
//
// [VANTA-WHY] §14.17 — BATE's mismatch reason is a CATEGORY ("got Error"); the
// revise prompt needs the EVIDENCE. The overnight sanic session shows the model
// guessing, over and over, why a test it wrote came back Error when the answer
// was in the run's output all along: the exception, its message, and the frames
// pytest printed. This renders exactly that, and nothing else — the model is
// being told why its test failed, not given the repository's whole log.
//
// [VANTA-WHY] BOUNDED, for the same reason §14.13 is: a generated test can print
// tens of thousands of lines (one sanic test did), and the remedy for "the model
// cannot see the error" must not become "the model is buried in it". The tail is
// the right end to keep: pytest prints the innermost frame last.

/// The most lines a revise prompt's baseline evidence may carry, footer included.
pub const REVISE_EVIDENCE_MAX_LINES: usize = 40;

/// One compact, bounded account of what the baseline run actually did: the
/// verdict in BATE's own terms, the exception line, and the tail of the last
/// traceback. Used verbatim as the revise prompt's evidence block.
pub fn baseline_evidence(details: &TestExecutionDetails) -> String {
    let combined = format!("{}\n{}", details.stdout, details.stderr);
    let lines: Vec<&str> = combined.lines().map(str::trim_end).collect();

    // The LAST traceback is the failure the run ended on — an earlier one may be
    // a captured log line or a warning's stack.
    let start = lines
        .iter()
        .rposition(|line| line.trim_start().starts_with("Traceback (most recent call last)"))
        .unwrap_or(0);
    let tail: Vec<&str> = lines[start..]
        .iter()
        .copied()
        .filter(|line| !line.trim().is_empty())
        .collect();

    let mut out: Vec<String> = vec![format!("verdict: {}", verdict_label(details))];
    if let Some(exception) = exception_line(&tail) {
        out.push(format!("exception: {exception}"));
    }
    if !tail.is_empty() {
        out.push("run output (last lines, most recent last):".to_string());
        // The header lines count against the cap, so a 40-line slice is 40 lines.
        let budget = REVISE_EVIDENCE_MAX_LINES.saturating_sub(out.len());
        let from = tail.len().saturating_sub(budget);
        out.extend(tail[from..].iter().map(|line| (*line).to_string()));
    }
    out.join("\n")
}

/// BATE's verdict in words a reader can act on, with the distinction it cares
/// about: ERROR (the test never reached its assertions) is not FAILED (an
/// assertion ran and did not hold).
fn verdict_label(details: &TestExecutionDetails) -> &'static str {
    match details.result {
        TestResult::Passed => "PASSED (the test ran and every assertion held)",
        TestResult::Failed => "FAILED (an assertion in the test did not hold)",
        TestResult::Error if details.timed_out => {
            "ERROR (the test timed out and never produced a verdict)"
        }
        TestResult::Error if !details.report_path_exists => {
            "ERROR (the test was never graded: it failed to import or collect)"
        }
        TestResult::Error => {
            "ERROR (the test raised an exception before its assertions completed)"
        }
    }
}

/// The exception line from a failure's tail, e.g. `AttributeError: 'NoneType'
/// object has no attribute 'items'`.
///
/// [VANTA-WHY] The TYPE comes from `diagnostician::parse_failure_site`, the same
/// locator the Diagnostician classifies with, so "what exception was that" has
/// one answer in this codebase. A plain `assert` prints no type in its `E` lines,
/// so the line is preferred by type and only then by pytest's own `E` marker.
fn exception_line(tail: &[&str]) -> Option<String> {
    let output = tail.join("\n");
    let (_, _, exception_type) = crate::pipeline::diagnostician::parse_failure_site(&output);
    if let Some(exception_type) = exception_type {
        if let Some(line) = tail.iter().rev().find(|line| line.contains(&exception_type)) {
            return Some(line.trim().to_string());
        }
    }
    tail.iter().rev().find_map(|line| {
        line.trim_start()
            .strip_prefix("E ")
            .map(|rest| rest.trim().to_string())
    })
}

// ============================================================================
// harness_error — a baseline run that died before pytest graded the test (§14.19)
// ============================================================================
//
// [VANTA-WHY] §14.19 — some failures are the ENVIRONMENT, not the test, and
// revising the test cannot fix them. The overnight sanic session is the worked
// example: `pytest_sanic` is a `pytest11` entry point, so pytest imported the
// project under test while loading plugins, that import reached a module the
// (then truncated) tree did not have, and pytest aborted in `_preparse` with a
// plain traceback and exit 1. No `report.json` was ever written, so the
// generated test was never graded — it came back `Error`, never `Failed` — and
// BATE scrapped it, then handed it back for revision. The Attacker was asked to
// fix a broken environment from inside a test file, twice per test, at 32k
// tokens a call, and burned hours.
//
// [VANTA-WHY] The discriminator is two facts together, and both are needed:
//   1. `report_path_exists == false` — pytest reached no verdict for the
//      generated test. Combined with `result == Error` and `!timed_out`, this
//      is "never graded": a timeout DID run the test, and a real exception
//      inside the test leaves a report with its nodeid in it.
//   2. The output does NOT name the generated test file. A collection error
//      that names it is the test's own fault — a syntax error, a bad import in
//      the test — and the Attacker CAN fix that, so it must still be revised.
//      A plugin load or a `conftest.py` import error names other files, and the
//      Attacker can do nothing about it.
//
// [VANTA-WHY] Read from the LAST of `validate_with_retry`'s attempts, not the
// first: that function re-runs three times precisely so a one-off spawn failure
// is absorbed. A transient VM fault must not abort the attempt, and a genuine
// broken environment reproduces across all three.
//
// [VANTA-WHY] Why the output text is quoted into the record: the guard exists to
// turn this class into one legible row, and "harness_error" alone would send the
// reader back to the log to learn WHICH plugin or module failed.

/// A baseline run that errored before pytest could grade the generated test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessError {
    /// The test whose baseline run surfaced it. Every test in the batch would
    /// fail the same way, which is why BATE stops here rather than continuing.
    pub test_name: String,
    /// The tail of what pytest printed, bounded like §14.17's revise evidence.
    pub evidence: String,
}

/// §14.19 — `Some(evidence)` when this baseline run never reached a verdict for
/// the generated test and the failure is not the test's own doing.
pub fn harness_error(details: &TestExecutionDetails) -> Option<String> {
    if details.result != TestResult::Error || details.timed_out || details.report_path_exists {
        return None;
    }
    let output = format!("{}\n{}", details.stdout, details.stderr);
    if output.contains(harness::BATE_TEST_FILE) {
        // The generated test file is named in the failure, so its own collection
        // or import is what broke: the Attacker's problem, not the environment's.
        return None;
    }
    // Reuse §14.17's renderer for the log tail, so the record carries the same
    // "verdict / exception / last lines" shape the revise prompt would have
    // seen, and is bounded by the same cap.
    Some(baseline_evidence(details))
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
    /// [VANTA-WHY] §14.19 — `Some` when a baseline run died before pytest graded
    /// the generated test, so this batch stopped here. It is NOT a scrap reason:
    /// the tests were never judged, so reporting them as scrapped would blame the
    /// Attacker for a broken environment and inflate the discard rate with a
    /// non-test. The caller stops the attempt instead of revising.
    pub harness_error: Option<HarnessError>,
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
            let (outcome, baseline) = validate_with_retry(
                &current_test.test_name,
                &current_test.intent,
                3, // Layer 1: re-run in a fresh sandbox up to 3× for flakiness
                || {
                    let src = src.clone();
                    let test_source = test_source.clone();
                    let cmd_template = cmd_template.clone();
                    // [VANTA-WHY] §14.17 — the DETAILED runner, not
                    // `run_single_test`: the verdict is the same (the plain
                    // wrapper is this call's `.result`), but the revise step
                    // below needs the output the verdict was read from.
                    async move {
                        harness::run_single_test_with_timeout(
                            &src,
                            &test_source,
                            &cmd_template,
                            harness::test_timeout_secs(),
                        )
                        .await
                    }
                },
            )
            .await;
            let baseline_result = baseline.result.clone();

            // [VANTA-WHY] §14.19 — the environment, not the test. Checked before
            // the relabel and revise paths, because both of those hand the test
            // back to the Attacker, and the Attacker cannot fix a plugin load or
            // a `conftest.py` that fails to import. Every remaining test in the
            // batch would fail identically (same interpreter, same plugins, same
            // tree), so BATE stops for this attempt instead of spending a
            // revision budget on each of them.
            if let Some(evidence) = harness_error(&baseline) {
                warn!(
                    test = %current_test.test_name,
                    "BATE: baseline never reached a verdict and the failure is not the test's \
                     — stopping BATE for this attempt"
                );
                return BateRun {
                    report: BateReport {
                        valid_tests: valid_tests.iter().map(|t| t.test_name.clone()).collect(),
                        scrapped_tests,
                    },
                    valid_tests,
                    relabel_events,
                    intent_checks,
                    harness_error: Some(HarnessError {
                        test_name: current_test.test_name.clone(),
                        evidence,
                    }),
                };
            }

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
                            &baseline_evidence(&baseline),
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
        harness_error: None,
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
            || async { TestExecutionDetails::from_rerun(TestResult::Passed) },
        )
        .await;
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
        assert_eq!(
            last.result,
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
                    TestExecutionDetails::from_rerun(TestResult::Passed) // mismatch on 1, 2
                } else {
                    TestExecutionDetails::from_rerun(TestResult::Failed) // matches on 3
                }
            },
        )
        .await;
        assert_eq!(outcome, BateOutcome::Valid);
        assert_eq!(*attempts.borrow(), 3);
    }

    // ---- §14.17: what the baseline ACTUALLY did, for the revise prompt --------
    //
    // [VANTA-WHY] The revise prompt used to carry only BATE's CATEGORY ("got
    // Error") and the model guessed at the cause for 45 rewrites. These pin the
    // evidence block: it must name the exception and the frames, and it must stay
    // small enough that a runaway test cannot bury the message.

    /// Details as the detailed runner would report them: a verdict plus the
    /// streams it was read from.
    fn details(result: TestResult, stdout: &str, stderr: &str) -> TestExecutionDetails {
        TestExecutionDetails {
            result,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            exit_code: Some(1),
            timed_out: false,
            report_path_exists: true,
            sandbox_root: None,
        }
    }

    #[test]
    fn the_evidence_carries_the_exception_and_the_frames_the_model_used_to_guess_at() {
        // The shape pytest prints for a test that raised: the location line, the
        // traceback, and the short-summary line that repeats type and message.
        let stdout = "=================== FAILURES ====================\n\
                      ____________________ test_reverse_order ____________________\n\n\
                      \x20   def test_reverse_order():\n\
                      >       cart = make_cart(None)\n\n\
                      tests/test_bate.py:4: \n\
                      _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n\n\
                      value = None\n\n\
                      \x20   def make_cart(value):\n\
                      >       return value.items()\n\
                      E       AttributeError: 'NoneType' object has no attribute 'items'\n\n\
                      pkg/cart.py:88: AttributeError\n\
                      =================== short test summary info ====================\n\
                      FAILED test_bate.py::test_reverse_order - AttributeError: 'NoneType' \
                      object has no attribute 'items'\n\
                      ====================== 1 failed in 0.04s ======================\n";
        let evidence = baseline_evidence(&details(TestResult::Error, stdout, ""));

        assert!(
            evidence.contains("verdict: ERROR (the test raised an exception"),
            "an ERROR must not be reported as a FAILED assertion: {evidence}"
        );
        assert!(
            evidence.contains(
                "exception: FAILED test_bate.py::test_reverse_order - AttributeError: \
                 'NoneType' object has no attribute 'items'"
            ),
            "the type AND the message must both be in front of the model: {evidence}"
        );
        assert!(
            evidence.contains(">       return value.items()"),
            "the innermost frame is what says WHICH call raised: {evidence}"
        );
    }

    #[test]
    fn a_passing_baseline_is_reported_as_passed_and_an_assertion_as_failed() {
        assert!(
            baseline_evidence(&details(TestResult::Passed, "", "")).starts_with("verdict: PASSED"),
            "a preservation probe's passing baseline must read as a pass"
        );
        assert!(
            baseline_evidence(&details(TestResult::Failed, "", "")).starts_with("verdict: FAILED"),
            "a failing assertion is FAILED, which is not the same claim as ERROR"
        );
    }

    #[test]
    fn an_ungraded_error_is_told_apart_from_an_exception_the_test_raised() {
        // `report_path_exists == false` means the generated test never got a
        // verdict at all (import/collect failure) — the distinction that used to
        // reach the model as the single word "Error".
        let mut ungraded = details(TestResult::Error, "", "");
        ungraded.report_path_exists = false;
        let evidence = baseline_evidence(&ungraded);
        assert!(
            evidence.contains("never graded"),
            "a test that was never graded must not be blamed on an exception: {evidence}"
        );
    }

    // ---- §14.19: the environment vs the test ---------------------------------
    //
    // [VANTA-WHY] The recorded sanic failure, verbatim in shape: a `pytest11`
    // plugin imports the project under test, the import reaches a module the tree
    // does not have, and pytest aborts in `_preparse` — before collection, before
    // any report. `test_bate.py` is nowhere in it.

    const PLUGIN_LOAD_FAILURE: &str = "\
Traceback (most recent call last):
  File \"/opt/venv38/lib/python3.8/site-packages/_pytest/config/__init__.py\", line 890, in _preparse
    self.pluginmanager.load_setuptools_entrypoints(\"pytest11\")
  File \"/opt/venv38/lib/python3.8/site-packages/pluggy/manager.py\", line 299, in load_setuptools_entrypoints
    plugin = ep.load()
  File \"/opt/venv38/lib/python3.8/site-packages/pytest_sanic/plugin.py\", line 6, in <module>
    from .utils import TestServer, TestClient
  File \"/workspace/sanic/asgi.py\", line 26, in <module>
    from sanic.websocket import WebSocketConnection
ModuleNotFoundError: No module named 'sanic.websocket'
";

    #[test]
    fn a_plugin_that_fails_to_load_is_the_environment_not_the_test() {
        // `report_path_exists == false` + the output naming no generated test.
        let mut ungraded = details(TestResult::Error, "", PLUGIN_LOAD_FAILURE);
        ungraded.report_path_exists = false;

        let evidence =
            harness_error(&ungraded).expect("a plugin load failure is not the Attacker's to fix");
        assert!(
            evidence.contains("never graded"),
            "the record must say the test was never judged: {evidence}"
        );
        assert!(
            evidence.contains("ModuleNotFoundError"),
            "the record must name what actually broke, or the reader is back in the log: {evidence}"
        );
    }

    #[test]
    fn a_collection_error_that_names_the_generated_test_is_still_the_tests_own() {
        // The Attacker CAN fix a syntax error in its own file, so this must stay
        // on the revise path even though nothing was graded.
        let mut ungraded = details(
            TestResult::Error,
            "",
            "ERROR collecting test_bate.py\n\
             test_bate.py:2: in <module>\n\
             import sanic\n\
             E   SyntaxError: invalid syntax\n",
        );
        ungraded.report_path_exists = false;

        assert!(
            harness_error(&ungraded).is_none(),
            "a failure that names the generated test is revisable, not a harness error"
        );
    }

    #[test]
    fn nothing_else_is_a_harness_error() {
        // A graded verdict, however negative, is evidence about the test.
        assert!(harness_error(&details(TestResult::Passed, "", "")).is_none());
        assert!(harness_error(&details(TestResult::Failed, "assert 1 == 2", "")).is_none());
        let mut graded_error = details(TestResult::Error, "Traceback...\nValueError: x", "");
        graded_error.report_path_exists = true;
        assert!(
            harness_error(&graded_error).is_none(),
            "an exception inside the test leaves a report for its nodeid: that is the test's own"
        );
        // A timeout DID run the test — it is slow, not ungraded.
        let mut timed_out = details(TestResult::Error, "", "hung");
        timed_out.report_path_exists = false;
        timed_out.timed_out = true;
        assert!(
            harness_error(&timed_out).is_none(),
            "a timeout is a verdict about the test's runtime, not a broken environment"
        );
    }

    #[test]
    fn a_runaway_tests_output_is_cut_to_the_forty_lines_of_evidence() {
        let mut stdout = String::from("Traceback (most recent call last):\n");
        for i in 0..5_000 {
            stdout.push_str(&format!("  File \"pkg/big.py\", line {i}, in f\n    step({i})\n"));
        }
        stdout.push_str("AssertionError: boom\n");
        let evidence = baseline_evidence(&details(TestResult::Failed, &stdout, ""));

        assert!(
            evidence.lines().count() <= REVISE_EVIDENCE_MAX_LINES,
            "the block claims to be bounded: {} lines",
            evidence.lines().count()
        );
        assert!(
            evidence.ends_with("AssertionError: boom"),
            "the cap must keep the tail — the innermost failure is the last thing printed: \
             {evidence}"
        );
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
