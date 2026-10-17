// ============================================================================
// CHECKPOINT D — TestIntent + GeneratedTest + BATE Baseline Mismatch
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | TestIntent enum           → Spec §5: every Attacker test declares a
//     |                           →   PURPOSE (bug repro / feature / behavior
//     |                           →   preservation). BATE CANNOT WORK without it.
//  ---+---------------------------+-------------------------------------------
//   2 | GeneratedTest struct      → The unit of adversarial testing: name +
//     |                           →   source code + intent + human-readable
//     |                           →   reason. Attacker returns Vec<GeneratedTest>.
//  ---+---------------------------+-------------------------------------------
//   3 | BATE mismatch rule        → Spec §7 Phase 1: a BugReproduction test
//     | (intent → expected        →   that PASSES on original code is USELESS
//     |   baseline)               →   (it didn't reproduce the bug). Scrapped.
//  ---+---------------------------+-------------------------------------------
//   4 | validate_test_against_    → Pure function: intent + actual_baseline →
//     |   baseline()              →   BateOutcome::Valid or Mismatch(reason).
//     |                           →   No I/O, fully unit-testable.
//  ---+---------------------------+-------------------------------------------
//   5 | #[test] assertions        → The same logic runs as a binary AND as
//     |                           →   unit tests. cargo test catches regressions
//     |                           →   in the BATE classification table.
// ============================================================================

use vanta::core::{TestIntent, TestResult};
use vanta::pipeline::bate::{validate_test_against_baseline, BateOutcome, BateReport};

// ----------------------------------------------------------------------------
// Demo: simulate two Attacker-generated tests and run BATE on each.
// ----------------------------------------------------------------------------
// [VANTA-WHY] This is the CORE of Spec §7 Phase 1. Before we spend compute
// running tests against the PATCHED repo, we validate that each test is
// trustworthy by checking its behavior on the ORIGINAL (unpatched) baseline.
// A test that doesn't behave as its intent promises is scrapped — feeding
// untrustworthy tests into the repair loop pollutes EIR/ECR statistics.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHECKPOINT D — TestIntent + BATE Baseline Validation        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: BugReproduction test that FAILS on original (correct — proves bug)
    let test_a_name = "huge_cart_overflows";
    let test_a_intent = TestIntent::BugReproduction;
    let test_a_baseline = TestResult::Failed; // original code overflows → test fails
    let outcome_a = validate_test_against_baseline(test_a_name, &test_a_intent, test_a_baseline.clone());
    println!("Test A: {test_a_name} (intent=BugReproduction, baseline={:?}) → {:?}",
        test_a_baseline, outcome_a);

    // Test 2: BugReproduction test that PASSES on original (WRONG — scrapped)
    // [PITFALL] This is the classic BATE failure mode: the Attacker wrote a
    // "bug reproduction" test that doesn't actually reproduce the bug. If we
    // didn't scrap it, a passing result on the patched code would be meaningless.
    let test_b_name = "trivial_addition";
    let test_b_intent = TestIntent::BugReproduction;
    let test_b_baseline = TestResult::Passed; // original handles it fine → test passes
    let outcome_b = validate_test_against_baseline(test_b_name, &test_b_intent, test_b_baseline.clone());
    println!("Test B: {test_b_name} (intent=BugReproduction, baseline={:?}) → {:?}",
        test_b_baseline, outcome_b);

    // Test 3: BehaviorPreservation test that PASSES on original (correct)
    let test_c_name = "empty_cart_zero";
    let test_c_intent = TestIntent::BehaviorPreservation;
    let test_c_baseline = TestResult::Passed;
    let outcome_c = validate_test_against_baseline(test_c_name, &test_c_intent, test_c_baseline.clone());
    println!("Test C: {test_c_name} (intent=BehaviorPreservation, baseline={:?}) → {:?}",
        test_c_baseline, outcome_c);

    // Build a BateReport summarizing the batch
    let mut report = BateReport {
        valid_tests: Vec::new(),
        scrapped_tests: Vec::new(),
    };
    for (name, _intent, _baseline, outcome) in [
        (test_a_name, test_a_intent, test_a_baseline, outcome_a.clone()),
        (test_b_name, test_b_intent, test_b_baseline, outcome_b.clone()),
        (test_c_name, test_c_intent, test_c_baseline, outcome_c.clone()),
    ] {
        match outcome {
            BateOutcome::Valid => report.valid_tests.push(name.to_string()),
            BateOutcome::Mismatch(reason) => report.scrapped_tests.push((name.to_string(), reason)),
        }
    }

    println!("\n{}", report.summary());
    println!("  Valid:    {:?}", report.valid_tests);
    println!("  Scrapped: {:?}", report.scrapped_tests.iter().map(|(n, _)| n).collect::<Vec<_>>());

    assert!(report.valid_tests.contains(&test_a_name.to_string()));
    assert!(report.scrapped_tests.iter().any(|(n, _)| n == test_b_name));
    assert!(report.valid_tests.contains(&test_c_name.to_string()));

    println!("\n✔ CHECKPOINT D PASSED — BATE mismatch classification is correct.");
    Ok(())
}

// ============================================================================
// UNIT TESTS — run via `cargo test --bin checkpoint_d`
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bug_reproduction_passing_baseline_is_mismatch() {
        // A bug-reproduction test that passes on original code is USELESS.
        let outcome = validate_test_against_baseline(
            "trivial", &TestIntent::BugReproduction, TestResult::Passed);
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
    }

    #[test]
    fn bug_reproduction_failing_baseline_is_valid() {
        let outcome = validate_test_against_baseline(
            "overflow", &TestIntent::BugReproduction, TestResult::Failed);
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[test]
    fn behavior_preservation_passing_baseline_is_valid() {
        let outcome = validate_test_against_baseline(
            "empty", &TestIntent::BehaviorPreservation, TestResult::Passed);
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[test]
    fn feature_acceptance_failing_baseline_is_valid() {
        let outcome = validate_test_against_baseline(
            "new_feat", &TestIntent::FeatureAcceptance, TestResult::Failed);
        assert_eq!(outcome, BateOutcome::Valid);
    }

    #[test]
    fn behavior_preservation_failing_baseline_is_mismatch() {
        // A behavior-preservation test that fails on original means the
        // behavior wasn't preserved BEFORE the patch — the test is wrong.
        let outcome = validate_test_against_baseline(
            "broken", &TestIntent::BehaviorPreservation, TestResult::Failed);
        assert!(matches!(outcome, BateOutcome::Mismatch(_)));
    }
}

// ============================================================================
// RECIPES — copy-paste templates for future milestones
// ============================================================================
//
// ----------------------------------------------------------------------------
// RECIPE D1 — "Declare a test's intent (Spec §5 mandatory field)"
//
//   use vanta::core::TestIntent;
//   let intent = TestIntent::BugReproduction;  // or FeatureAcceptance / BehaviorPreservation
//
//   → USE IN: Attacker agent output, GeneratedTest struct, BATE validation.
//     WITHOUT intent, BATE cannot determine expected baseline behavior.
//
// ----------------------------------------------------------------------------
// RECIPE D2 — "Validate one test against its baseline (BATE core)"
//
//   use vanta::core::{TestIntent, TestResult};
//   use vanta::pipeline::bate::validate_test_against_baseline;
//   let outcome = validate_test_against_baseline("test_name", TestIntent::BugReproduction, TestResult::Failed);
//   match outcome { BateOutcome::Valid => run_it, BateOutcome::Mismatch(r) => scrap(r) }
//
//   → USE IN: M2 Phase 1 loop over every Attacker-generated test.
//
// ----------------------------------------------------------------------------
// RECIPE D3 — "Build a BateReport from a batch of outcomes"
//
//   let mut report = BateReport { valid_tests: vec![], scrapped_tests: vec![] };
//   for (name, outcome) in outcomes { match outcome { Valid => report.valid_tests.push(name), Mismatch(r) => report.scrapped_tests.push((name, r)) } }
//
//   → USE IN: M2 after running all baseline tests, before patched execution.
//
// ----------------------------------------------------------------------------
// RECIPE D4 — "Intent → expected baseline lookup table"
//
//   use vanta::pipeline::bate::expected_baseline;
//   let expected = expected_baseline(&TestIntent::BugReproduction);  // TestResult::Failed
//
//   → USE IN: Debugging why BATE scrapped a test (compare expected vs actual).
//
// ----------------------------------------------------------------------------
// RECIPE D5 — "Inline #[test] for BATE invariants (regression safety net)"
//
//   #[test] fn bug_repro_must_fail_baseline() {
//       assert!(matches!(validate_test_against_baseline("x", TestIntent::BugReproduction, TestResult::Passed), BateOutcome::Mismatch(_)));
//   }
//
//   → USE IN: Every module that changes classification logic. The Spec §7
//     table is the contract; tests lock it down so a refactor can't flip it.
//
// ============================================================================
