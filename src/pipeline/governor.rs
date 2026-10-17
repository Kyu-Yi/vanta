// ============================================================================
// GOVERNOR — Fixed-N repair loop governor with shared test set (Spec §8)
// ============================================================================
//
// [VANTA-WHY] Spec §8: "V1 = Fixed-N retry, default 4, every transition logged."
// The Governor is the STOPPING CONDITION for the repair loop. V1 uses a fixed
// attempt cap (N=4) because EIR/ECR rates can't be computed without history.
// Every transition is logged so M4's live EIR/ECR governor can be computed from
// the same data without rerunning.
//
// The FIXED SHARED TEST SET protocol (Spec §8):
//   - Comparing attempt N vs N+1 requires running BOTH on the SAME saved tests.
//   - Newly generated tests run immediately and can reject a patch on the spot,
//     but they are recorded separately and only join the shared set for the
//     FOLLOWING comparison.
//   - Incomplete transitions (one attempt didn't finish the shared set) are
//     marked CENSORED and excluded from EIR/ECR estimation.
//
// ECR = fail -> pass (new patch corrected an error)
// EIR = pass -> fail (new patch introduced an error)
// ============================================================================

use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use crate::core::{
    normalized_test_body_hash, shared_set_fingerprint, AttemptTransition, EvidenceSource,
    FailureClass, GovernorDecision, TestIntent, TestResult,
};

// [VANTA-WHY] Default max attempts = 4 (Spec §8 Governor by version table).
// V1 generates history with fixed-N; V2 replaces with live EIR/ECR condition.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 4;

// ============================================================================
// GovernorMode — what `max_attempts` MEANS for this run
// ============================================================================
//
// [VANTA-WHY] The same number means two different things depending on whether
// the candidate can change. In REPAIR mode attempt N+1 is a NEW patch, so an
// attempt that passes is finished business and the loop stops at once; the cap
// only bounds how many patches the Creator may burn. In REVIEW mode (--patch)
// the candidate is FIXED, so attempt N+1 is a NEW batch of attacks against the
// same code — the cap is ATTACK DEPTH, not a patch budget. Stopping on the first
// passing round there throws away the depth that was asked for, which is exactly
// how a bad patch survived: round 1's batch happened to sample inputs the patch
// did not touch, and the second round that would have sampled the changed input
// was never run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernorMode {
    /// Creator-driven repair: stop the moment every shared-set test passes.
    Repair,
    /// `--patch` review: keep attacking a fixed candidate for `max_attempts`
    /// rounds; stop early only on a rejection.
    Review,
}

// ============================================================================
// AttemptVerdict — what the patched run showed about the candidate
// ============================================================================
//
// [VANTA-WHY] `all_shared_passed: bool` was not enough to state the stop rule.
// "Not everything passed" lumps together "the candidate is wrong" (a Logic
// failure, or a repo-native regression) and "we could not get clean evidence"
// (Infrastructure / Unknown). Only the first is a rejection; the second must not
// be reported as one, because an environment fault is not a finding about the
// model. The three-way verdict keeps that distinction out of the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptVerdict {
    /// Every test passed.
    Passed,
    /// The candidate demonstrably fails: at least one Logic-class failure, or a
    /// repo-native test that passed on the original and fails on the candidate.
    Rejected,
    /// Failures exist but none is attributable to the candidate (Infrastructure,
    /// Unknown or Static only). Not enough evidence to reject, and nothing
    /// passed. The loop keeps going if it has budget.
    Inconclusive,
}

/// One observation of a test inside the fixed shared set.
///
/// [VANTA-WHY] `test_source` is carried rather than a precomputed hash so the
/// identity of an observation can never drift from its content. A caller that
/// passed a stale hash would silently compare unequal things; here the hash is
/// always derived from the source actually recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedObservation {
    pub test_name: String,
    pub test_source: String,
    pub result: TestResult,
    /// None for a pass. Some(..) for any failure — the class is what decides
    /// whether this observation is allowed to move EIR/ECR.
    pub failure_class: Option<FailureClass>,
    /// [VANTA-WHY] Provenance travels WITH the observation, so a transition row
    /// can record who wrote the test without the Governor having to look the
    /// test up again (and without the lookup silently failing when a probe only
    /// exists on one side of the comparison).
    pub evidence_source: EvidenceSource,
    pub test_intent: TestIntent,
}

impl SharedObservation {
    pub fn test_hash(&self) -> String {
        normalized_test_body_hash(&self.test_source)
    }
}

pub struct Governor {
    max_attempts: u32,
    current_attempt: u32,
    // [VANTA-WHY] shared_results: normalised test-body hash -> observation,
    // ACCUMULATED across attempts. This is the "fixed shared test set" — the
    // same tests are run on every attempt so transitions are comparable. Keyed
    // by content rather than name, because a name is not an identity: reusing
    // one for a different body silently swaps evidence underneath the
    // comparison. A probe is added once and never evicted.
    shared_results: HashMap<String, SharedObservation>,
    // [VANTA-WHY] previous_shared: the shared set from the PREVIOUS attempt. We
    // compare current vs previous to compute EIR/ECR transitions.
    previous_shared: Option<HashMap<String, SharedObservation>>,
    // [VANTA-WHY] name -> hash, purely to DETECT name reuse across differing
    // bodies. Never used for matching.
    name_hashes: HashMap<String, String>,
    name_collisions: usize,
    // [VANTA-WHY] s0_hashes: S^(0), the shared set as it stood on the FIRST
    // attempt. Frozen on the first record_shared_results and never updated, so
    // the S^(0)-restricted analysis in the paper can be reconstructed from the
    // persisted rows alone.
    s0_hashes: HashSet<String>,
    // [VANTA-WHY] All transitions recorded so far. Logged to session for M4.
    transitions: Vec<AttemptTransition>,
}

impl Governor {
    pub fn new(max_attempts: u32) -> Self {
        Governor {
            max_attempts: max_attempts.max(1),
            current_attempt: 0,
            shared_results: HashMap::new(),
            previous_shared: None,
            name_hashes: HashMap::new(),
            name_collisions: 0,
            s0_hashes: HashSet::new(),
            transitions: Vec::new(),
        }
    }

    pub fn current_attempt(&self) -> u32 {
        self.current_attempt
    }

    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Number of distinct probes currently in the shared set.
    pub fn shared_probe_count(&self) -> usize {
        self.shared_results.len()
    }

    /// How many times a name was reused for a DIFFERENT body. Non-zero means the
    /// Attacker is recycling names, which is exactly the case that used to
    /// silently overwrite evidence.
    pub fn name_collisions(&self) -> usize {
        self.name_collisions
    }

    /// Record the shared-set observations for the current attempt.
    /// Computes transitions vs the previous attempt and stores them.
    pub fn record_shared_results(&mut self, observations: Vec<SharedObservation>) {
        let mut current: HashMap<String, SharedObservation> = HashMap::new();
        for obs in observations {
            let hash = obs.test_hash();
            self.record_name_binding(&obs.test_name, &hash);
            match current.get(&hash) {
                // Same probe observed twice in one attempt (two names, one body).
                // Keep the pessimistic result so a probe that failed anywhere is
                // not recorded as a pass.
                Some(existing) if existing.result != TestResult::Passed => {}
                _ => {
                    current.insert(hash, obs);
                }
            }
        }

        // [VANTA-WHY] Freeze S^(0) on the first attempt. Everything observed
        // then IS the initial shared set; later attempts only add to it. Taking
        // it here (not from the accumulator) is deliberate: fold_new_tests runs
        // after record, so the accumulator at this instant is exactly the
        // first-attempt set.
        if self.previous_shared.is_none() {
            self.s0_hashes = current.keys().cloned().collect();
        }

        // Compute transitions vs previous attempt.
        if let Some(prev) = &self.previous_shared {
            // [VANTA-WHY] Every row from this call was compared on the SAME
            // question set, so they share one id. Computing it from `prev` (not
            // from the accumulator) is deliberate: `prev` is the set that
            // actually ran last attempt, which is what the comparison is on.
            let shared_set_id = shared_set_fingerprint(prev.keys());
            for (hash, prev_obs) in prev {
                let curr = current.get(hash);
                let complete = curr.is_some();
                // Provenance is a property of the probe; prefer the current
                // attempt's copy (the Attacker may have restated intent) and fall
                // back to the previous one when the probe only ran last time.
                let provenance = curr.unwrap_or(prev_obs);
                self.transitions.push(AttemptTransition {
                    from_attempt: self.current_attempt.saturating_sub(1),
                    to_attempt: self.current_attempt,
                    shared_set_id: shared_set_id.clone(),
                    test_id: hash.clone(),
                    test_name: curr
                        .map(|o| o.test_name.clone())
                        .unwrap_or_else(|| prev_obs.test_name.clone()),
                    from_result: prev_obs.result.clone(),
                    to_result: curr.map(|o| o.result.clone()).unwrap_or(TestResult::Error),
                    from_failure_class: prev_obs.failure_class.clone(),
                    to_failure_class: curr.and_then(|o| o.failure_class.clone()),
                    complete,
                    in_s0: self.s0_hashes.contains(hash),
                    evidence_source: provenance.evidence_source.clone(),
                    test_intent: provenance.test_intent.clone(),
                });
            }
        }

        // [VANTA-WHY] previous_shared must hold the results JUST recorded so the
        // next record_shared_results() can compare against them. Using
        // mem::take(shared_results) here is a BUG: on the first call
        // shared_results is empty, so previous_shared would be set to an empty
        // map and no transitions would ever be computed.
        //
        // [VANTA-WHY] shared_results ACCUMULATES rather than being replaced. It
        // is the fixed shared test set — Spec §8 says the same tests run on
        // every attempt, and the runner's own corpus only ever grows. Replacing
        // it would evict a probe the moment one attempt didn't observe it,
        // which is exactly how a renamed/re-bodied test used to vanish. A probe
        // stays known; if it wasn't run this attempt its transition is CENSORED
        // (handled above), not silently deleted from the set.
        self.previous_shared = Some(current.clone());
        for (hash, obs) in current {
            self.shared_results.entry(hash).or_insert(obs);
        }
    }

    /// Add NEW tests to the shared set (they join on the NEXT comparison).
    /// Spec §8: "Newly generated tests ... recorded separately, then folded
    /// into the next comparison."
    pub fn fold_new_tests(&mut self, new_tests: &[SharedObservation]) {
        for obs in new_tests {
            let hash = obs.test_hash();
            self.record_name_binding(&obs.test_name, &hash);
            self.shared_results
                .entry(hash.clone())
                .or_insert_with(|| obs.clone());
            // [VANTA-WHY] Register the probe on the COMPARISON side too, not
            // just in the accumulator. Spec §8: a new test "runs immediately"
            // on the attempt that generated it and joins the shared set "for
            // the following comparison" — so the result observed here IS the
            // `from` side of its first transition. Registering it only in the
            // accumulator (which is never compared) silently cost one
            // comparison of evidence per generated test: the probe's first
            // transition row was never produced. `or_insert` because a probe
            // already observed in the shared set must keep its own record.
            if let Some(prev) = self.previous_shared.as_mut() {
                prev.entry(hash).or_insert_with(|| obs.clone());
            }
        }
    }

    /// Track name -> hash so reuse of a name for a different body is loud
    /// instead of silent. Both probes are kept; neither is dropped.
    fn record_name_binding(&mut self, name: &str, hash: &str) {
        match self.name_hashes.get(name) {
            Some(existing) if existing != hash => {
                self.name_collisions += 1;
                warn!(
                    test_name = name,
                    previous_hash = existing.as_str(),
                    new_hash = hash,
                    "Governor: test name reused for a DIFFERENT body — tracked as separate \
                     probes (never merged, never dropped). Rename tests to keep reports readable."
                );
            }
            Some(_) => {}
            None => {
                self.name_hashes.insert(name.to_string(), hash.to_string());
            }
        }
    }

    /// Decide whether to continue the loop.
    ///
    /// [VANTA-WHY] The stop rule is MODE-DEPENDENT, and getting that wrong is how
    /// a bad patch survived review. `max_attempts` bounds two different things:
    ///
    ///   Repair  — a budget of PATCHES. Attempt N is a brand-new candidate, so a
    ///             clean pass ends the run immediately; there is nothing left to
    ///             repair and burning more attempts would only invite churn.
    ///   Review  — a budget of ATTACKS against ONE fixed candidate. A clean pass
    ///             on round 1 is not a finding: the round's batch is a sample, and
    ///             an untouched input is exactly what the sampler missed. The
    ///             correct behaviour is to keep attacking to the requested depth
    ///             and stop early ONLY on a rejection.
    ///
    /// `verdict` is three-way (see `AttemptVerdict`) so an environment fault is
    /// never reported as a finding about the candidate.
    pub fn decide(
        &mut self,
        mode: GovernorMode,
        verdict: AttemptVerdict,
        patch_changed: bool,
    ) -> GovernorDecision {
        if !patch_changed {
            warn!("Governor: patch did not change the file — stopping (PatchApplicationFailed)");
            return GovernorDecision::PatchApplicationFailed;
        }

        match mode {
            GovernorMode::Repair => {
                if verdict == AttemptVerdict::Passed {
                    info!(attempt = self.current_attempt,
                        "Governor: all shared tests passed — stopping (AllTestsPassed)");
                    return GovernorDecision::AllTestsPassed;
                }
            }
            GovernorMode::Review => {
                if verdict == AttemptVerdict::Rejected {
                    info!(attempt = self.current_attempt,
                        "Governor: candidate demonstrably failed — stopping (Rejected)");
                    return GovernorDecision::Rejected;
                }
            }
        }

        self.current_attempt += 1;
        if self.current_attempt >= self.max_attempts {
            // [VANTA-WHY] In review mode, spending the whole attack depth without
            // a rejection IS the approval — the candidate survived every round we
            // were willing to run. Reporting MaxAttemptsReached here would say
            // "gave up", which is the opposite of what happened and is exactly the
            // distinction the `Rejected` variant exists to preserve.
            if mode == GovernorMode::Review && verdict == AttemptVerdict::Passed {
                info!(attempt = self.current_attempt, depth = self.max_attempts,
                    "Governor: attack depth exhausted with no rejection — candidate passed \
                     all rounds (AllTestsPassed)");
                return GovernorDecision::AllTestsPassed;
            }
            info!(attempt = self.current_attempt, max = self.max_attempts,
                "Governor: reached max attempts — stopping (MaxAttemptsReached)");
            return GovernorDecision::MaxAttemptsReached;
        }

        debug!(attempt = self.current_attempt, max = self.max_attempts, ?mode,
            "Governor: continuing loop");
        GovernorDecision::Continue
    }

    pub fn transitions(&self) -> &[AttemptTransition] {
        &self.transitions
    }
}

impl Default for Governor {
    fn default() -> Self {
        Governor::new(DEFAULT_MAX_ATTEMPTS)
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn obs(name: &str, source: &str, result: TestResult, class: Option<FailureClass>) -> SharedObservation {
        SharedObservation {
            test_name: name.to_string(),
            test_source: source.to_string(),
            result,
            failure_class: class,
            evidence_source: EvidenceSource::AttackerGenerated,
            test_intent: TestIntent::BugReproduction,
        }
    }

    fn failed(name: &str, source: &str, class: FailureClass) -> SharedObservation {
        obs(name, source, TestResult::Failed, Some(class))
    }

    fn passed(name: &str, source: &str) -> SharedObservation {
        obs(name, source, TestResult::Passed, None)
    }

    /// Advance one REPAIR attempt whose evidence was neither a clean pass nor a
    /// demonstrated failure (the plain "keep going" case the transition tests need).
    fn advance_repair(g: &mut Governor) {
        assert_eq!(
            g.decide(GovernorMode::Repair, AttemptVerdict::Inconclusive, true),
            GovernorDecision::Continue,
            "an attempt that is not a clean pass must keep the repair loop going"
        );
    }

    #[test]
    fn test_stop_on_all_passed() {
        let mut g = Governor::new(4);
        g.record_shared_results(vec![passed("t1", "def test_t1():\n    assert True\n")]);
        let decision = g.decide(GovernorMode::Repair, AttemptVerdict::Passed, true);
        assert_eq!(decision, GovernorDecision::AllTestsPassed);
    }

    #[test]
    fn test_stop_on_patch_failed() {
        let mut g = Governor::new(4);
        let decision = g.decide(GovernorMode::Repair, AttemptVerdict::Inconclusive, false);
        assert_eq!(decision, GovernorDecision::PatchApplicationFailed);
    }

    #[test]
    fn test_stop_on_max_attempts() {
        let mut g = Governor::new(2);
        let src = "def test_t1():\n    assert False\n";
        g.record_shared_results(vec![failed("t1", src, FailureClass::Logic)]);
        assert_eq!(
            g.decide(GovernorMode::Repair, AttemptVerdict::Inconclusive, true),
            GovernorDecision::Continue
        );
        g.record_shared_results(vec![failed("t1", src, FailureClass::Logic)]);
        assert_eq!(
            g.decide(GovernorMode::Repair, AttemptVerdict::Inconclusive, true),
            GovernorDecision::MaxAttemptsReached
        );
    }

    // ---- review mode: `max_attempts` is ATTACK DEPTH ------------------------
    //
    // [VANTA-WHY] These two tests pin the exact defect that let a bad patch
    // through: a clean round 1 used to end the run, so the second batch of
    // attacks — the one that would have sampled the input the patch changed —
    // never ran.

    #[test]
    fn review_mode_runs_every_round_on_a_passing_candidate() {
        let mut g = Governor::new(3);
        let src = "def test_t1():\n    assert shipping_band(1000) == \"standard\"\n";
        // Three clean rounds. The first two must NOT stop: each is one batch of
        // attacks, and a batch is a sample, not a verdict.
        for round in 1..=2 {
            g.record_shared_results(vec![passed("t1", src)]);
            assert_eq!(
                g.decide(GovernorMode::Review, AttemptVerdict::Passed, true),
                GovernorDecision::Continue,
                "review round {round} passed, but the requested attack depth is 3 — \
                 a clean sample is not an approval"
            );
        }
        // The last round spends the depth and is the approval.
        g.record_shared_results(vec![passed("t1", src)]);
        assert_eq!(
            g.decide(GovernorMode::Review, AttemptVerdict::Passed, true),
            GovernorDecision::AllTestsPassed,
            "surviving the whole attack depth without a rejection is the approval"
        );
    }

    #[test]
    fn review_mode_stops_early_on_a_rejection() {
        let mut g = Governor::new(5);
        let src = "def test_t1():\n    assert shipping_band(500) == \"standard\"\n";
        g.record_shared_results(vec![passed("t1", src)]);
        assert_eq!(
            g.decide(GovernorMode::Review, AttemptVerdict::Passed, true),
            GovernorDecision::Continue
        );
        // Round 2 finds the regression. Depth is 5, but a rejection is terminal.
        g.record_shared_results(vec![failed("t1", src, FailureClass::Logic)]);
        assert_eq!(
            g.decide(GovernorMode::Review, AttemptVerdict::Rejected, true),
            GovernorDecision::Rejected,
            "a demonstrated failure ends the review at once — there is nothing to repair"
        );
    }

    #[test]
    fn review_mode_reports_an_unresolvable_round_as_max_attempts_not_a_rejection() {
        let mut g = Governor::new(2);
        let src = "def test_t1():\n    assert shipping_band(500) == \"standard\"\n";
        // [VANTA-WHY] Infrastructure/Static-only failures are not findings about
        // the candidate. Reviewing such a round to the point of exhaustion must
        // read "gave up", never "found a regression".
        g.record_shared_results(vec![failed("t1", src, FailureClass::Static)]);
        assert_eq!(
            g.decide(GovernorMode::Review, AttemptVerdict::Inconclusive, true),
            GovernorDecision::Continue
        );
        g.record_shared_results(vec![failed("t1", src, FailureClass::Static)]);
        assert_eq!(
            g.decide(GovernorMode::Review, AttemptVerdict::Inconclusive, true),
            GovernorDecision::MaxAttemptsReached
        );
    }

    /// The Governor's job is to RECORD transitions; the counting rule itself
    /// lives once in `session::ecr_eir_from_rows`. So this asserts the recorded
    /// directions, not the aggregate.
    #[test]
    fn transitions_record_both_directions() {
        let mut g = Governor::new(4);
        let t1 = "def test_t1():\n    assert add(2, 3) == 6\n";
        let t2 = "def test_t2():\n    assert add(2, 3) == 5\n";

        g.record_shared_results(vec![
            failed("t1", t1, FailureClass::Logic),
            passed("t2", t2),
        ]);
        advance_repair(&mut g);

        // attempt 1: t1 passed (a correction), t2 failed (an introduction)
        g.record_shared_results(vec![
            passed("t1", t1),
            failed("t2", t2, FailureClass::Logic),
        ]);

        let transitions = g.transitions();
        assert_eq!(transitions.len(), 2);
        let t1_row = transitions.iter().find(|t| t.test_name == "t1").unwrap();
        assert!(matches!(
            (&t1_row.from_result, &t1_row.to_result),
            (TestResult::Failed, TestResult::Passed)
        ));
        assert!(t1_row.complete);
        let t2_row = transitions.iter().find(|t| t.test_name == "t2").unwrap();
        assert!(matches!(
            (&t2_row.from_result, &t2_row.to_result),
            (TestResult::Passed, TestResult::Failed)
        ));
        assert_eq!(
            t2_row.to_failure_class,
            Some(FailureClass::Logic),
            "the recorded transition must carry the failure class the row will be judged on"
        );
        assert!(t2_row.complete);
    }

    #[test]
    fn an_unobserved_probe_makes_the_transition_incomplete() {
        let mut g = Governor::new(4);
        let t1 = "def test_t1():\n    assert add(2, 3) == 6\n";
        let t2 = "def test_t2():\n    assert add(2, 3) == 5\n";

        g.record_shared_results(vec![
            failed("t1", t1, FailureClass::Logic),
            passed("t2", t2),
        ]);
        advance_repair(&mut g);

        // attempt 1: only t1 ran (t2 not observed this attempt)
        g.record_shared_results(vec![passed("t1", t1)]);

        let t2_row = g
            .transitions()
            .iter()
            .find(|t| t.test_name == "t2")
            .expect("the unobserved probe must still produce a transition row");
        assert!(
            !t2_row.complete,
            "a probe missing from one attempt is a censored hole, not a complete row"
        );
    }

    // ---- body-hash identity ------------------------------------------------

    #[test]
    fn same_probe_under_two_names_is_one_entry() {
        let mut g = Governor::new(4);
        g.record_shared_results(vec![
            failed("test_probe", "def test_probe():\n    assert add(2, 3) == 6\n", FailureClass::Logic),
            failed("test_probe_v2", "def test_probe_v2():\n    assert add(2, 3) == 6\n", FailureClass::Logic),
        ]);
        assert_eq!(
            g.shared_probe_count(),
            1,
            "two names for one body are one probe, not two"
        );
        assert_eq!(g.name_collisions(), 0, "different names are not a collision");
    }

    #[test]
    fn name_reuse_for_a_different_body_is_flagged_not_dropped() {
        let mut g = Governor::new(4);
        g.record_shared_results(vec![failed(
            "test_edge_case",
            "def test_edge_case():\n    assert add(2, 3) == 6\n",
            FailureClass::Logic,
        )]);
        g.record_shared_results(vec![failed(
            "test_edge_case",
            "def test_edge_case():\n    assert add(9, 9) == 18\n",
            FailureClass::Logic,
        )]);
        assert_eq!(
            g.shared_probe_count(),
            2,
            "the second body must be kept as its own probe, never merged into the first"
        );
        assert_eq!(
            g.name_collisions(),
            1,
            "reusing a name for a different body must be counted, not silent"
        );
    }

    #[test]
    fn duplicate_probe_within_one_attempt_keeps_the_pessimistic_result() {
        let mut g = Governor::new(4);
        g.record_shared_results(vec![
            passed("test_a", "def test_a():\n    assert add(2, 3) == 5\n"),
            failed("test_b", "def test_b():\n    assert add(2, 3) == 5\n", FailureClass::Logic),
        ]);
        assert_eq!(g.shared_probe_count(), 1);
        assert!(
            g.transitions().is_empty(),
            "there is no previous attempt yet, so no transition exists to count"
        );
    }

    // ---- the counting rule itself ------------------------------------------
    //
    // [VANTA-WHY] The Logic-class gate — that a non-Logic failure must not move
    // EIR/ECR — is part of the ONE counting definition,
    // `session::ecr_eir_from_rows`, and is tested there on persisted rows. It is
    // deliberately not re-tested here: a second copy of the rule in this file is
    // exactly the drift the single function exists to prevent.

    // ---- S^(0) membership and provenance on the persisted rows -------------

    #[test]
    fn first_attempt_probes_are_marked_s0() {
        let mut g = Governor::new(4);
        let src = "def test_t1():\n    assert add(2, 3) == 5\n";
        g.record_shared_results(vec![passed("t1", src)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![failed("t1", src, FailureClass::Logic)]);

        let t = &g.transitions()[0];
        assert!(t.in_s0, "a probe present on the first attempt is in S^(0)");
    }

    #[test]
    fn probe_folded_in_later_is_not_in_s0() {
        let mut g = Governor::new(4);
        let first = "def test_first():\n    assert add(2, 3) == 5\n";
        let later = "def test_later():\n    assert add(1, 1) == 2\n";

        g.record_shared_results(vec![passed("first", first)]);
        // A test generated on the first attempt joins the shared set for the
        // NEXT comparison — but it was never part of S^(0).
        g.fold_new_tests(&[passed("later", later)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![
            failed("first", first, FailureClass::Logic),
            passed("later", later),
        ]);

        let later_rows: Vec<_> = g.transitions().iter().filter(|t| t.test_name == "later").collect();
        assert_eq!(later_rows.len(), 1, "the folded probe must still produce a transition");
        assert!(
            !later_rows[0].in_s0,
            "a probe folded in after attempt 0 is NOT in S^(0)"
        );
        let first_rows: Vec<_> = g.transitions().iter().filter(|t| t.test_name == "first").collect();
        assert!(first_rows[0].in_s0);
    }

    #[test]
    fn transition_carries_provenance_of_the_probe() {
        let mut g = Governor::new(4);
        let src = "def test_t1():\n    assert add(2, 3) == 5\n";

        let oracle_probe = |result, class| {
            let mut o = obs("t1", src, result, class);
            o.evidence_source = EvidenceSource::BenchmarkOracle;
            o.test_intent = TestIntent::BehaviorPreservation;
            o
        };

        g.record_shared_results(vec![oracle_probe(TestResult::Passed, None)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![oracle_probe(TestResult::Failed, Some(FailureClass::Logic))]);

        let t = &g.transitions()[0];
        assert_eq!(
            t.evidence_source,
            EvidenceSource::BenchmarkOracle,
            "provenance must reach the row, so oracle-only re-analysis is possible"
        );
        assert_eq!(t.test_intent, TestIntent::BehaviorPreservation);
    }

    #[test]
    fn censored_row_keeps_the_provenance_of_the_side_that_did_run() {
        let mut g = Governor::new(4);
        let src = "def test_t1():\n    assert add(2, 3) == 5\n";
        let mut oracle = passed("t1", src);
        oracle.evidence_source = EvidenceSource::BenchmarkOracle;

        g.record_shared_results(vec![oracle]);
        advance_repair(&mut g);
        // The next attempt does not observe the probe at all.
        g.record_shared_results(vec![]);

        let t = &g.transitions()[0];
        assert!(!t.complete, "an unobserved probe is a censored transition, not a deleted row");
        assert_eq!(
            t.evidence_source,
            EvidenceSource::BenchmarkOracle,
            "a censored row still records who wrote the probe"
        );
    }

    // ---- shared-set identity on the persisted rows -------------------------

    #[test]
    fn shared_set_fingerprint_is_order_independent() {
        let a = "aaa".to_string();
        let b = "bbb".to_string();
        assert_eq!(
            shared_set_fingerprint(vec![&a, &b]),
            shared_set_fingerprint(vec![&b, &a]),
            "the same probe set must fingerprint the same regardless of order"
        );
    }

    #[test]
    fn rows_of_one_comparison_share_a_shared_set_id() {
        let mut g = Governor::new(4);
        let a = "def test_a():\n    assert add(2, 3) == 5\n";
        let b = "def test_b():\n    assert add(4, 4) == 8\n";

        g.record_shared_results(vec![passed("a", a), passed("b", b)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![
            failed("a", a, FailureClass::Logic),
            passed("b", b),
        ]);

        let rows = g.transitions();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].shared_set_id, rows[1].shared_set_id,
            "rows from one comparison were asked the same questions"
        );

        let first_id = rows[0].shared_set_id.clone();
        advance_repair(&mut g);
        g.record_shared_results(vec![
            failed("a", a, FailureClass::Logic),
            passed("b", b),
        ]);
        assert_eq!(
            g.transitions().last().unwrap().shared_set_id,
            first_id,
            "the same probe set must fingerprint identically across attempts"
        );
    }

    #[test]
    fn shared_set_id_changes_when_the_question_set_grows() {
        let mut g = Governor::new(4);
        let a = "def test_a():\n    assert add(2, 3) == 5\n";
        let c = "def test_c():\n    assert add(0, 0) == 0\n";

        g.record_shared_results(vec![passed("a", a)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![failed("a", a, FailureClass::Logic)]);
        let before = g.transitions().last().unwrap().shared_set_id.clone();

        // A test generated this attempt joins the set for the NEXT comparison,
        // so the next comparison is on a strictly larger question set.
        g.fold_new_tests(&[passed("c", c)]);
        advance_repair(&mut g);
        g.record_shared_results(vec![
            failed("a", a, FailureClass::Logic),
            passed("c", c),
        ]);

        let after = g.transitions().last().unwrap().shared_set_id.clone();
        assert_ne!(
            before, after,
            "a grown question set is a different comparison and must be marked as one"
        );
    }
}
