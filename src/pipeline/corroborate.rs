// ============================================================================
// CORROBORATION — §14.2: when a failing test is evidence about the PATCH
// ============================================================================
//
// [VANTA-WHY] No prompt can guarantee an LLM writes a correct test. A rejection
// is a claim about the PATCH, so it needs evidence that survives the possibility
// that the test itself is wrong. Before this, a single Attacker-authored test
// that asserted output text the issue never mentions rejected a developer fix
// that three earlier rounds had approved (PySnooper/3, §14). A single
// UNCORROBORATED failure must stop being a rejection and become "a human should
// look at this" — that is `GovernorDecision::NeedsReview`.
//
// Every rule here is a PURE function of evidence the run already has:
// `TestEvidence`'s test_name / failure_class / intent, the Diagnostician's
// `structured_payload`, the native-regression list, the blocking static
// findings, and the issue text. No LLM, no I/O — so it is unit-testable exactly
// like `route_intent`, and the decision it drives can be re-derived from the
// artifacts of a finished run.
//
// The rules (first one that fires, in this order, decides; `run.summary` records
// its letter in `deciding_rule`):
//
//   (a) native regression      the project's OWN suite, not a Vanta-authored test
//   (b) symptom match          a BugReproduction failure whose exception the ISSUE names
//   (c) two distinct tests     >=2 distinct validated tests failing as Logic
//   (d) validated BP/FA         a preservation/acceptance failure the issue does not override
//   (e) static in touched file  deterministic tool output, not model-authored
// ============================================================================

use crate::core::{FailureClass, TestIntent};

/// Which rule (§14.2) corroborated a rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorroborationRule {
    /// (a) A repo-native test passed on the original tree and fails on the patch.
    NativeRegression,
    /// (b) A BATE-validated `BugReproduction` failure whose exception type is one
    /// the ISSUE text itself names.
    SymptomMatch,
    /// (c) Two or more distinct BATE-validated generated tests failing as Logic.
    TwoDistinctTests,
    /// (d) A BATE-validated `BehaviorPreservation`/`FeatureAcceptance` failure the
    /// auditor did not answer `issue_requires_change = true` for.
    ValidatedPreservationFailure,
    /// (e) A blocking static finding in a file the patch touched.
    StaticInTouchedFile,
}

impl CorroborationRule {
    /// The letter §14.2 names the rule by — what `run.summary` records.
    pub fn code(self) -> &'static str {
        match self {
            CorroborationRule::NativeRegression => "a",
            CorroborationRule::SymptomMatch => "b",
            CorroborationRule::TwoDistinctTests => "c",
            CorroborationRule::ValidatedPreservationFailure => "d",
            CorroborationRule::StaticInTouchedFile => "e",
        }
    }
}

/// One failing generated test, plus the two facts about it that are not on the
/// test itself: whether BATE holds a baseline for it, and what the intent
/// auditor answered about the issue (rule (d)'s downgrade).
#[derive(Debug, Clone, PartialEq)]
pub struct FailureEvidence {
    pub test_name: String,
    /// 1-based round the failure was observed in. Recorded so a rejection can
    /// name the round, and so the distinctness key of rule (c) is an
    /// observation and not just a name.
    pub attempt: u32,
    pub intent: TestIntent,
    pub failure_class: Option<FailureClass>,
    /// The exception the failure actually raised, read from the Diagnostician's
    /// `structured_payload["exception_type"]`. `None` when the payload carried
    /// none — which can never satisfy rule (b), and is the honest direction: an
    /// unreadable failure cannot be matched against the issue's symptom.
    pub exception_type: Option<String>,
    /// BATE holds a baseline for this test (it ran it on the ORIGINAL tree and
    /// the result matched the intent). A failure without a baseline is not
    /// evidence about anything, so no rule may use it.
    pub bate_validated: bool,
    /// The auditor's answer to "does the issue require the behaviour at THIS
    /// test's input to change?". `Some(true)` downgrades the failure: a test
    /// pinning behaviour the issue says must change is broken by a correct fix,
    /// so its failure is not evidence against the patch.
    pub issue_requires_change: Option<bool>,
}

/// Everything the rules read, for one round of one candidate.
pub struct CorroborationInput<'a> {
    /// This round's generated-test failures. Includes the carried-over shared
    /// set (re-run every round in review mode), so a rule (c) second test can
    /// come from an earlier round's batch without any accumulation here.
    pub failures: &'a [FailureEvidence],
    pub native_regressions: &'a [String],
    /// Blocking (error-class, touched-file) static findings.
    pub new_static_in_touched_files: &'a [String],
    /// The issue text as the run received it. Rule (b)'s symptoms are read from
    /// HERE and nowhere else — a symptom Vanta inferred is not the issue's.
    pub issue_text: &'a str,
}

/// What the rules concluded, and the evidence for each half of it.
#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    /// The first rule that fired, or `None` when the failures are uncorroborated
    /// — the decision is then `NeedsReview`, not a rejection.
    pub rule: Option<CorroborationRule>,
    /// `<test_name>@<attempt>`, the evidence that fired `rule`.
    pub evidence: Vec<String>,
    /// The failures no rule covers, as `<test_name>=<Class>/<Intent>@<attempt>`.
    /// Reported for a `NeedsReview` so a reader sees WHICH weak failures caused
    /// it, not just that one did.
    pub weak: Vec<String>,
}

/// The exception-class tokens an issue text names.
///
/// [VANTA-WHY] Mechanically identical to the harness's own expression
/// (`scripts/bugsinpy/issue.py`: `\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b`),
/// implemented by hand because a token run of word characters IS what that
/// pattern matches. Same expression, same tokens: the issue Vanta receives has
/// its behavioural line built from the harness's `failure_kinds`, so an issue
/// whose symptom is `NameError` says `NameError` here too.
///
/// An issue that names no such token returns an empty list, and rule (b) then
/// cannot fire. That is the honest direction: an issue with no named symptom
/// cannot corroborate an exception.
pub fn issue_symptom_tokens(issue_text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for token in issue_text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if is_symptom_token(token) && !found.iter().any(|t| t == token) {
            found.push(token.to_string());
        }
    }
    found.sort();
    found
}

/// `\b[A-Z][A-Za-z0-9_]*(Error|Exception|Warning)\b` for one word-character run.
///
/// [VANTA-WHY] The suffix must START at index >= 1, because the pattern spends
/// its first `[A-Z]` before it: `Error` on its own does NOT match (only `rror`
/// would be left for the suffix), while `OSError` does. Being one character
/// looser here would let a bare `Error` in the issue text corroborate an
/// exception the issue never named.
fn is_symptom_token(token: &str) -> bool {
    if !token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return false;
    }
    match token.chars().next() {
        Some(first) if first.is_ascii_uppercase() => {}
        _ => return false,
    }
    ["Error", "Exception", "Warning"]
        .iter()
        .any(|suffix| token.len() > suffix.len() && token.ends_with(suffix))
}

/// Does this failure name the issue's own symptom?
fn matches_issue_symptom(failure: &FailureEvidence, symptoms: &[String]) -> bool {
    match &failure.exception_type {
        Some(exception) => symptoms.iter().any(|s| s == exception),
        None => false,
    }
}

/// Rule (b)'s applicability: a validated BugReproduction failure that says the
/// issue's own symptom.
fn is_symptom_match(failure: &FailureEvidence, symptoms: &[String]) -> bool {
    failure.bate_validated
        && failure.intent == TestIntent::BugReproduction
        && failure.failure_class == Some(FailureClass::Logic)
        && matches_issue_symptom(failure, symptoms)
}

/// Rule (d)'s applicability: a validated preservation/acceptance failure the
/// auditor did not answer `issue_requires_change = true` for.
///
/// [VANTA-WHY] The downgrade is the whole point. A `BehaviorPreservation` probe
/// whose input the issue requires to CHANGE is broken by a correct fix, so its
/// failure is evidence FOR the patch, not against it. The audit already answers
/// that question for every test, so the rule reads the answer instead of
/// re-deciding it. A `BehaviorPreservation` test with `Some(true)` cannot
/// normally reach a round at all — `route_intent` scraps or promotes it — but
/// rule (d) does not rely on that: it reads the recorded answer.
fn is_validated_preservation_failure(failure: &FailureEvidence) -> bool {
    failure.bate_validated
        && matches!(
            failure.intent,
            TestIntent::BehaviorPreservation | TestIntent::FeatureAcceptance
        )
        && failure.failure_class == Some(FailureClass::Logic)
        && failure.issue_requires_change != Some(true)
}

/// The DISTINCT test names among the Logic failures.
///
/// [VANTA-WHY] Rule (c) counts tests, not observations. The shared set's
/// identity is the test body, and the same test is re-run every round in review
/// mode — so a single bogus test failing in rounds 1 and 2 is ONE test failing
/// twice, not two independent observations. Counting observations would let
/// exactly the one-wrong-test case this module exists for manufacture its own
/// corroboration. The two tests may come from different rounds and may carry
/// different intents; they must be different tests.
fn distinct_test_names(logic_failures: &[&FailureEvidence]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for failure in logic_failures {
        if !names.contains(&failure.test_name) {
            names.push(failure.test_name.clone());
        }
    }
    names
}

/// Apply §14.2 to one round's evidence.
pub fn assess(input: &CorroborationInput<'_>) -> Assessment {
    let symptoms = issue_symptom_tokens(input.issue_text);
    let mut logic_failures: Vec<&FailureEvidence> = input
        .failures
        .iter()
        .filter(|f| f.failure_class == Some(FailureClass::Logic))
        .collect();
    logic_failures.sort_by(|a, b| a.test_name.cmp(&b.test_name).then(a.attempt.cmp(&b.attempt)));

    // (a) — the project's own suite. Nothing about it is model-authored.
    if !input.native_regressions.is_empty() {
        return Assessment {
            rule: Some(CorroborationRule::NativeRegression),
            evidence: input
                .native_regressions
                .iter()
                .map(|nodeid| format!("{nodeid}@native"))
                .collect(),
            weak: Vec::new(),
        };
    }

    // (b) — the issue's own symptom, reproduced.
    for failure in &logic_failures {
        if is_symptom_match(failure, &symptoms) {
            return Assessment {
                rule: Some(CorroborationRule::SymptomMatch),
                evidence: vec![render(failure)],
                weak: Vec::new(),
            };
        }
    }

    // (c) — two independent tests, so one wrong test cannot manufacture it.
    let distinct = distinct_test_names(&logic_failures);
    if distinct.len() >= 2 {
        return Assessment {
            rule: Some(CorroborationRule::TwoDistinctTests),
            evidence: distinct
                .iter()
                .map(|name| {
                    let f = logic_failures
                        .iter()
                        .find(|f| &f.test_name == name)
                        .expect("name came from this list");
                    render(f)
                })
                .collect(),
            weak: Vec::new(),
        };
    }

    // (d) — a validated preservation/acceptance failure the issue does not
    // override.
    for failure in &logic_failures {
        if is_validated_preservation_failure(failure) {
            return Assessment {
                rule: Some(CorroborationRule::ValidatedPreservationFailure),
                evidence: vec![render(failure)],
                weak: Vec::new(),
            };
        }
    }

    // (e) — deterministic tool output in a file the patch touched.
    if !input.new_static_in_touched_files.is_empty() {
        return Assessment {
            rule: Some(CorroborationRule::StaticInTouchedFile),
            evidence: input.new_static_in_touched_files.to_vec(),
            weak: Vec::new(),
        };
    }

    // No rule fired. Every Logic failure is weak by definition, and the caller
    // turns "there is at least one" into `NeedsReview`.
    Assessment {
        rule: None,
        evidence: Vec::new(),
        weak: logic_failures.iter().map(|f| render_weak(f)).collect(),
    }
}

/// `<test_name>@<attempt>` — the `deciding_evidence` form.
fn render(failure: &FailureEvidence) -> String {
    format!("{}@{}", failure.test_name, failure.attempt)
}

/// `<test_name>=<Class>/<Intent>@<attempt>` — the `needs_review_weak` form.
fn render_weak(failure: &FailureEvidence) -> String {
    format!(
        "{}={}/{}@{}",
        failure.test_name,
        class_text(failure.failure_class.as_ref()),
        intent_text(&failure.intent),
        failure.attempt,
    )
}

fn class_text(class: Option<&FailureClass>) -> String {
    match class {
        Some(c) => format!("{c:?}"),
        None => "None".to_string(),
    }
}

fn intent_text(intent: &TestIntent) -> String {
    format!("{intent:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(
        name: &str,
        intent: TestIntent,
        exception: Option<&str>,
        requires_change: Option<bool>,
    ) -> FailureEvidence {
        FailureEvidence {
            test_name: name.to_string(),
            attempt: 4,
            intent,
            failure_class: Some(FailureClass::Logic),
            exception_type: exception.map(|e| e.to_string()),
            bate_validated: true,
            issue_requires_change: requires_change,
        }
    }

    fn assess_with<'a>(
        failures: &'a [FailureEvidence],
        natives: &'a [String],
        statics: &'a [String],
        issue_text: &'a str,
    ) -> Assessment {
        assess(&CorroborationInput {
            failures,
            native_regressions: natives,
            new_static_in_touched_files: statics,
            issue_text,
        })
    }

    const PYSNOOPER_ISSUE: &str = "Observed behaviour: the project's own test suite fails on the \
        reported scenario (NameError). Expected behaviour: it should not.";

    // ---- the symptom-token expression matches the harness's ----

    #[test]
    fn symptom_tokens_are_the_exception_classes_the_issue_names() {
        assert_eq!(issue_symptom_tokens(PYSNOOPER_ISSUE), vec!["NameError"]);
        assert_eq!(
            issue_symptom_tokens("it raises a ValueError and a KeyError."),
            vec!["KeyError".to_string(), "ValueError".to_string()]
        );
        // Not a token: lowercase, no suffix, or a suffix followed by more word
        // characters (`\b` after the suffix must hold).
        assert!(issue_symptom_tokens("nameerror, errors, ValueErrors, Error").is_empty());
    }

    #[test]
    fn an_issue_with_no_named_symptom_yields_no_tokens() {
        assert!(issue_symptom_tokens("the docs are out of date").is_empty());
        assert!(issue_symptom_tokens("").is_empty());
    }

    // ---- (a) ----

    #[test]
    fn rule_a_fires_on_a_native_regression() {
        let natives = vec!["tests/test_x.py::test_y".to_string()];
        let a = assess_with(&[], &natives, &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, Some(CorroborationRule::NativeRegression));
        assert_eq!(a.evidence, vec!["tests/test_x.py::test_y@native"]);
    }

    // ---- (b) ----

    #[test]
    fn rule_b_fires_on_a_bug_reproduction_naming_the_issues_symptom() {
        let failures = vec![failure(
            "test_snoop_with_string_path",
            TestIntent::BugReproduction,
            Some("NameError"),
            Some(true),
        )];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, Some(CorroborationRule::SymptomMatch));
        assert_eq!(a.evidence, vec!["test_snoop_with_string_path@4"]);
        assert!(a.weak.is_empty());
    }

    #[test]
    fn rule_b_does_not_fire_on_a_different_exception() {
        // The PySnooper/3 dev-fix shape: the issue names NameError, the failing
        // test raised AssertionError.
        let failures = vec![failure(
            "test_snoop_file_output_contains_starting_var",
            TestIntent::BugReproduction,
            Some("AssertionError"),
            Some(true),
        )];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, None);
        assert_eq!(
            a.weak,
            vec!["test_snoop_file_output_contains_starting_var=Logic/BugReproduction@4"]
        );
    }

    #[test]
    fn rule_b_cannot_fire_when_the_issue_names_no_symptom() {
        let failures = vec![failure(
            "t",
            TestIntent::BugReproduction,
            Some("NameError"),
            None,
        )];
        let a = assess_with(&failures, &[], &[], "the docs are out of date");
        assert_eq!(a.rule, None);
    }

    #[test]
    fn rule_b_requires_a_validated_test() {
        let mut f = failure("t", TestIntent::BugReproduction, Some("NameError"), None);
        f.bate_validated = false;
        let failures = vec![f];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, None);
    }

    #[test]
    fn rule_b_only_covers_bug_reproduction() {
        let failures = vec![failure(
            "t",
            TestIntent::BehaviorPreservation,
            Some("NameError"),
            None,
        )];
        // (d) covers it instead — the rule that fires is not (b).
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, Some(CorroborationRule::ValidatedPreservationFailure));
    }

    // ---- (c) ----

    #[test]
    fn rule_c_fires_on_two_distinct_tests() {
        let failures = vec![
            failure("t_one", TestIntent::BugReproduction, Some("KeyError"), None),
            failure("t_two", TestIntent::BehaviorPreservation, Some("KeyError"), None),
        ];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, Some(CorroborationRule::TwoDistinctTests));
        assert_eq!(a.evidence, vec!["t_one@4", "t_two@4"]);
    }

    #[test]
    fn rule_c_is_not_satisfied_by_one_test_failing_twice() {
        // The same test re-run in two rounds is ONE test. Counting observations
        // would let a single wrong test corroborate itself.
        let mut first = failure("t_same", TestIntent::BugReproduction, Some("KeyError"), None);
        first.attempt = 1;
        let mut second = failure("t_same", TestIntent::BugReproduction, Some("KeyError"), None);
        second.attempt = 2;
        let failures = vec![first, second];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, None);
        assert_eq!(a.weak.len(), 2);
    }

    // ---- (d) and its downgrade ----

    #[test]
    fn rule_d_fires_on_a_validated_preservation_failure() {
        let failures = vec![failure(
            "test_shipping_band_500_stays_standard",
            TestIntent::BehaviorPreservation,
            Some("AssertionError"),
            Some(false),
        )];
        let a = assess_with(&failures, &[], &[], "an issue that says nothing");
        assert_eq!(
            a.rule,
            Some(CorroborationRule::ValidatedPreservationFailure)
        );
    }

    #[test]
    fn rule_d_downgrades_when_the_issue_requires_that_behaviour_to_change() {
        let failures = vec![failure(
            "test_shipping_band_500_stays_standard",
            TestIntent::BehaviorPreservation,
            None,
            Some(true),
        )];
        let a = assess_with(&failures, &[], &[], "an issue that says nothing");
        assert_eq!(a.rule, None);
        assert_eq!(
            a.weak,
            vec!["test_shipping_band_500_stays_standard=Logic/BehaviorPreservation@4"]
        );
    }

    #[test]
    fn rule_d_covers_feature_acceptance_too() {
        let failures = vec![failure(
            "t_fa",
            TestIntent::FeatureAcceptance,
            None,
            None,
        )];
        let a = assess_with(&failures, &[], &[], "an issue that says nothing");
        assert_eq!(
            a.rule,
            Some(CorroborationRule::ValidatedPreservationFailure)
        );
    }

    #[test]
    fn an_unanswered_requires_change_does_not_downgrade() {
        // `None` is UNKNOWN, and only an explicit `true` downgrades. Absent must
        // not be read as "the issue requires this to stay" either way round.
        let failures = vec![failure("t_bp", TestIntent::BehaviorPreservation, None, None)];
        let a = assess_with(&failures, &[], &[], "an issue that says nothing");
        assert_eq!(
            a.rule,
            Some(CorroborationRule::ValidatedPreservationFailure)
        );
    }

    // ---- (e) ----

    #[test]
    fn rule_e_fires_on_a_new_static_finding_in_a_touched_file() {
        let statics = vec!["mypy: src/x.py:12: error: ...".to_string()];
        let a = assess_with(&[], &[], &statics, "an issue that says nothing");
        assert_eq!(a.rule, Some(CorroborationRule::StaticInTouchedFile));
        assert_eq!(a.evidence, statics);
    }

    // ---- ordering ----

    #[test]
    fn the_first_rule_in_order_decides() {
        let natives = vec!["tests/test_x.py::test_y".to_string()];
        let statics = vec!["mypy: src/x.py:12: error: ...".to_string()];
        let failures = vec![failure(
            "t_bp",
            TestIntent::BehaviorPreservation,
            None,
            Some(false),
        )];
        // (a) before (d) before (e).
        let a = assess_with(&failures, &natives, &statics, "an issue that says nothing");
        assert_eq!(a.rule, Some(CorroborationRule::NativeRegression));
        let a = assess_with(&failures, &[], &statics, "an issue that says nothing");
        assert_eq!(
            a.rule,
            Some(CorroborationRule::ValidatedPreservationFailure)
        );
        let a = assess_with(&[], &[], &statics, "an issue that says nothing");
        assert_eq!(a.rule, Some(CorroborationRule::StaticInTouchedFile));
    }

    #[test]
    fn rule_codes_are_the_letters_the_doc_names() {
        assert_eq!(CorroborationRule::NativeRegression.code(), "a");
        assert_eq!(CorroborationRule::SymptomMatch.code(), "b");
        assert_eq!(CorroborationRule::TwoDistinctTests.code(), "c");
        assert_eq!(CorroborationRule::ValidatedPreservationFailure.code(), "d");
        assert_eq!(CorroborationRule::StaticInTouchedFile.code(), "e");
    }

    // ---- the honest default ----

    #[test]
    fn nothing_at_all_is_not_a_rejection() {
        let a = assess_with(&[], &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, None);
        assert!(a.weak.is_empty());
    }

    #[test]
    fn a_non_logic_failure_is_never_evidence() {
        // Infrastructure is not a finding about the candidate (unchanged rule).
        let mut f = failure("t", TestIntent::BugReproduction, Some("NameError"), None);
        f.failure_class = Some(FailureClass::Infrastructure);
        let failures = vec![f];
        let a = assess_with(&failures, &[], &[], PYSNOOPER_ISSUE);
        assert_eq!(a.rule, None);
        assert!(a.weak.is_empty());
    }
}
