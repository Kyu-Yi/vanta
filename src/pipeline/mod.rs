// ============================================================================
// PIPELINE MODULE — BATE (Phase 1) + Diagnostician (Phase 2) +
//                   patch application + Governor (Phase 3) + Gatekeeper
// ============================================================================
pub mod bate;
pub mod candidate;
pub mod corroborate;
pub mod diagnostician;
pub mod driver;
pub mod harness;
pub mod patch;
pub mod governor;
pub mod gatekeeper;
pub mod rerun;

/// The per-attempt ceiling on the Attacker's remediation calls — revise (BATE)
/// and simplify (the Diagnostician's Unknown path) together (§14.14).
///
/// [VANTA-WHY] The per-test cap of 3 revisions bounds ONE test. Nothing bounded
/// their product across a round: three tests each allowed three revisions is
/// nine Attacker calls, and worse, a revision is free to return a renamed test
/// (`run_bate` does not check the name), which then enters as a brand-new test
/// with a fresh 3-revision budget. That is the treadmill the overnight run
/// measured. This constant is the cap the round cannot escape by renaming,
/// because it is counted per ATTEMPT and shared by every test in it; the
/// wall-clock deadline (§14.15) is the outer bound.
pub const REVISE_BUDGET_PER_ATTEMPT: u32 = 4;

/// The scrap reason a test gets when the attempt's remediation budget is spent
/// (§14.14).
pub const REVISE_BUDGET_EXHAUSTED: &str = "revise_budget_exhausted";

/// The `stop_reason` a run records when its own wall-clock deadline ended it
/// (§14.15).
///
/// [VANTA-WHY] The harness reports `timed_out` when IT killed the process — a
/// row with no `run.summary`, because the kill lands mid-work. This token is the
/// other case: Vanta noticed its own deadline, stopped, and wrote the verdict
/// down. A batch table that reads both must not average them together, so the
/// two are recorded under different names rather than one `stopped=true`.
pub const BUDGET_EXHAUSTED: &str = "budget_exhausted";

/// One attempt's shared Attacker remediation budget (§14.14).
///
/// [VANTA-WHY] Held by the attempt loop and borrowed by BATE and the
/// Diagnostician, so "revise+simplify" is ONE counter with one ceiling. BATE
/// spends a unit before each `attacker_revise_test`; the Diagnostician spends
/// one before its Unknown-path `attacker_simplify_test`. When it is spent, BATE
/// scraps the test as `revise_budget_exhausted` and the round continues with the
/// tests it has, and the Diagnostician leaves the test's Unknown standing
/// (exactly as a failed simplification would), so the outcome is never stronger
/// than the un-remediated one.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReviseBudget {
    used: u32,
}

impl ReviseBudget {
    pub fn new() -> Self {
        Self { used: 0 }
    }

    /// Spend one unit. Returns `false` when this attempt's budget is spent.
    pub fn try_use(&mut self) -> bool {
        if self.used < REVISE_BUDGET_PER_ATTEMPT {
            self.used += 1;
            true
        } else {
            false
        }
    }

    /// How many remediation calls this attempt has made.
    pub fn used(&self) -> u32 {
        self.used
    }
}

/// The prompt versions in force for this build, as one `session.start`-ready
/// string.
///
/// [VANTA-WHY] This is the single definition of the run's prompt provenance.
/// It names EVERY agent prompt, not just the Attacker's: the Attacker, the
/// patch-blind intent auditor, the Creator and the Diagnostician each change
/// what a run can find or accept. If only the Attacker were tagged, a later
/// edit to the intent-check or Creator prompt would be logged as "no change"
/// and two different experiments would look identical. The `session.start`
/// record and the PR body both read from here so the two can never drift.
pub fn prompt_versions() -> String {
    format!(
        "attacker={};intent_check={};creator={};diagnostician={}",
        crate::agents::attacker::ATTACKER_PROMPT_VERSION,
        crate::agents::attacker::INTENT_AUDITOR_PROMPT_VERSION,
        crate::agents::creator::CREATOR_PROMPT_VERSION,
        crate::pipeline::diagnostician::DIAGNOSTICIAN_PROMPT_VERSION,
    )
}

/// A short CONTENT hash of each role's system prompt text, in the same
/// `role=hash` shape as [`prompt_versions`].
///
/// [VANTA-WHY] §14.10 f — the version labels above are bumped by hand, and the
/// hand can forget: `3588728` (the A3 follow-up) and `d4ea314` (the A10
/// follow-up) both changed prompt wording after the `a2617b3` bump without
/// bumping, so two different prompts sat behind one tag and a log comparison
/// across the change would have said "no change". This string is DERIVED from the
/// text each role sends, so a wording change cannot be invisible: it moves the
/// hash whether or not anyone remembers the label. The two records are
/// complementary — the label says which revision a human MEANT to ship, the hash
/// says which text actually went out.
///
/// Every system prompt a role can send is covered, not just the one a given run
/// happens to select: the Attacker's three (generate / revise / simplify), the
/// auditor's four (the flag combinations of the grounded and batched variants),
/// and the Creator's one. A variant's wording change shows in the hash of any run,
/// which is what makes the record comparable across runs under different flags.
///
/// The Diagnostician is rule-based and sends no prompt of its own (its Unknown
/// path sends the Attacker's simplify prompt, already covered above), so its slot
/// is the literal `not_applicable` rather than a hash. Hashing the empty prompt
/// set would produce `e3b0c44298fc`, the SHA-256 of the empty string — a value
/// indistinguishable from a prompt that was truncated or mistyped, which is the
/// opposite of what the record is for. Naming the reason keeps the key's fixed
/// shape in every log AND keeps the value from reading as a real prompt.
pub fn prompt_hashes() -> String {
    let attacker = crate::core::prompt_content_hash(
        &crate::agents::attacker::attacker_system_prompts(),
    );
    let intent_check = crate::core::prompt_content_hash(
        &crate::agents::attacker::intent_auditor_system_prompts(),
    );
    let creator = crate::core::prompt_content_hash(&[crate::agents::creator::CREATOR_SYSTEM_PROMPT]);
    let diagnostician = "not_applicable";

    format!(
        "attacker={attacker};intent_check={intent_check};creator={creator};diagnostician={diagnostician}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [VANTA-WHY] §14.10 f — the string the two log rows record must be the
    /// hash of the prompts the call sites SEND, not of a private copy: a hash
    /// taken over text the model never receives would log "no change" for a
    /// wording change, which is the exact failure the record exists to prevent
    /// (`3588728` and `d4ea314` reworded prompts after the `a2617b3` bump).
    ///
    /// The roles, their order and the shape are pinned too, because the record is
    /// compared across runs and sessions: a role that silently dropped out of the
    /// string, or a hash of the wrong length, would make two different prompts
    /// compare equal by accident.
    #[test]
    fn the_recorded_hashes_are_the_hashes_of_the_prompts_the_call_sites_send() {
        let recorded = prompt_hashes();
        let fields: Vec<(&str, &str)> = recorded
            .split(';')
            .map(|field| {
                field
                    .split_once('=')
                    .expect("every field is `role=hash`")
            })
            .collect();

        assert_eq!(
            fields.iter().map(|(role, _)| *role).collect::<Vec<_>>(),
            vec!["attacker", "intent_check", "creator", "diagnostician"],
            "the hash must name every role `prompt_versions` names, in the same order"
        );

        let expected = [
            crate::core::prompt_content_hash(&crate::agents::attacker::attacker_system_prompts()),
            crate::core::prompt_content_hash(
                &crate::agents::attacker::intent_auditor_system_prompts(),
            ),
            crate::core::prompt_content_hash(&[crate::agents::creator::creator_system_prompt()]),
            // The Diagnostician is rule-based and sends no prompt of its own, so
            // its slot names the reason instead of hashing an empty set — which
            // would be `e3b0c44298fc`, the SHA-256 of "", indistinguishable from a
            // real prompt that was truncated.
            "not_applicable".to_string(),
        ];

        for ((role, hash), expected) in fields.iter().zip(expected) {
            assert_eq!(
                *hash, expected,
                "`{role}`'s recorded hash must be the hash of the prompts that role sends"
            );
            if *role == "diagnostician" {
                assert_eq!(
                    *hash, "not_applicable",
                    "the rule-based role has no prompt to hash, so it is named not_applicable"
                );
                continue;
            }
            assert!(
                hash.len() == 12
                    && hash
                        .chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "`{role}`'s hash must be 12 lowercase hex characters; got {hash:?}"
            );
        }
    }

    /// [VANTA-WHY] §14.14 — the budget's whole contract in one place: it hands
    /// out exactly `REVISE_BUDGET_PER_ATTEMPT` units, it reports how many it has
    /// handed out, and a ceiling of zero would hand out none. The revision
    /// treadmill the overnight run measured was unbounded, so "exactly N, then
    /// no" is the property the fix rests on; a total that could drift with the
    /// number of tests is the bug, not a detail.
    #[test]
    fn the_revise_budget_hands_out_exactly_the_cap_and_then_nothing() {
        let mut budget = ReviseBudget::new();
        assert_eq!(budget.used(), 0);
        for spent in 1..=REVISE_BUDGET_PER_ATTEMPT {
            assert!(
                budget.try_use(),
                "unit {spent} of {REVISE_BUDGET_PER_ATTEMPT} must be granted"
            );
            assert_eq!(budget.used(), spent);
        }
        assert!(
            !budget.try_use(),
            "the budget must refuse past its cap, not extend it"
        );
        assert_eq!(
            budget.used(),
            REVISE_BUDGET_PER_ATTEMPT,
            "a refused request must not be counted as spent"
        );
    }
}
