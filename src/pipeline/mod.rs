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
pub mod safe_path;
pub mod summary_escape;
pub mod suite_cache;

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

/// A short CONTENT hash of each role's prompt-template text, in the same
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
/// Every SYSTEM and USER template a role can send is covered, not just the one a
/// given run happens to select: the Attacker's three system prompts (generate /
/// revise / simplify) and its three user bodies, the auditor's four system
/// variants (the flag combinations of grounded and batched) and its seven user
/// bodies and flag-selected closing questions, and the Creator's one system
/// prompt and three user blocks. A variant's wording change shows in the hash of
/// any run, which is what makes the record comparable across runs under different
/// flags.
///
/// [VANTA-WHY] M3 item 19 (fix 4) — the record used to hash the SYSTEM prompts
/// only. Item 16 reordered the Attacker's generate USER prompt and NO recorded
/// hash moved, so the frozen configuration could not detect a prompt change:
/// `config_hash` described a configuration whose prompt text was no longer the
/// text the binary sent. The user templates are covered now, so an edit to any
/// template moves its role's hash, and
/// `the_committed_prompts_block_is_the_hash_of_the_templates_this_code_sends`
/// makes the committed `[prompts]` block — and therefore `config_hash` — follow.
///
/// The Diagnostician is rule-based and sends no prompt of its own (its Unknown
/// path sends the Attacker's simplify prompt, already covered above), so its slot
/// is the literal `not_applicable` rather than a hash. Hashing the empty prompt
/// set would produce `e3b0c44298fc`, the SHA-256 of the empty string — a value
/// indistinguishable from a prompt that was truncated or mistyped, which is the
/// opposite of what the record is for. Naming the reason keeps the key's fixed
/// shape in every log AND keeps the value from reading as a real prompt.
pub fn prompt_hashes() -> String {
    let attacker = crate::core::prompt_content_hash(&role_templates(
        &crate::agents::attacker::attacker_system_prompts(),
        &crate::agents::attacker::attacker_user_prompts(),
    ));
    let intent_check = crate::core::prompt_content_hash(&role_templates(
        &crate::agents::attacker::intent_auditor_system_prompts(),
        &crate::agents::attacker::intent_auditor_user_prompts(),
    ));
    let creator = crate::core::prompt_content_hash(&role_templates(
        &[crate::agents::creator::creator_system_prompt()],
        &crate::agents::creator::creator_user_prompts(),
    ));
    let diagnostician = "not_applicable";

    format!(
        "attacker={attacker};intent_check={intent_check};creator={creator};diagnostician={diagnostician}"
    )
}

/// `system then user` — the template list one role's hash is taken over.
///
/// [VANTA-WHY] M3 item 19 (fix 4) — the order is fixed and meaningful: a role's
/// system templates first, then its user templates. `prompt_content_hash` joins
/// the parts with U+001F, so a change to any template moves the role's hash and
/// text moved across the system/user boundary cannot be invisible.
///
/// `pub(crate)` for C4: the judge's own hash (`agents::judge::judge_prompt_hash`)
/// is taken over its system and user templates in this same order, and the judge
/// must not have a second, differently-ordered notion of "this role's prompt
/// text" than the arm it is compared with.
pub(crate) fn role_templates(system: &[&'static str], user: &[&'static str]) -> Vec<&'static str> {
    system.iter().copied().chain(user.iter().copied()).collect()
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
            crate::core::prompt_content_hash(&role_templates(
                &crate::agents::attacker::attacker_system_prompts(),
                &crate::agents::attacker::attacker_user_prompts(),
            )),
            crate::core::prompt_content_hash(&role_templates(
                &crate::agents::attacker::intent_auditor_system_prompts(),
                &crate::agents::attacker::intent_auditor_user_prompts(),
            )),
            crate::core::prompt_content_hash(&role_templates(
                &[crate::agents::creator::creator_system_prompt()],
                &crate::agents::creator::creator_user_prompts(),
            )),
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

    /// [VANTA-WHY] M3 item 19 (fix 4) — the user templates count too. Item 16
    /// reordered a USER prompt and no recorded hash moved, so a hash over the
    /// system prompts alone could not see a prompt change. This pins the fix's two
    /// halves per role: a role's user templates are INSIDE its hash (hashing the
    /// system half alone gives a different value), and editing ANY one template —
    /// system or user — moves that role's hash.
    #[test]
    fn every_prompt_template_is_inside_its_roles_hash() {
        let roles: [(&str, Vec<&'static str>, Vec<&'static str>); 3] = [
            (
                "attacker",
                crate::agents::attacker::attacker_system_prompts().to_vec(),
                crate::agents::attacker::attacker_user_prompts().to_vec(),
            ),
            (
                "intent_check",
                crate::agents::attacker::intent_auditor_system_prompts().to_vec(),
                crate::agents::attacker::intent_auditor_user_prompts().to_vec(),
            ),
            (
                "creator",
                vec![crate::agents::creator::creator_system_prompt()],
                crate::agents::creator::creator_user_prompts().to_vec(),
            ),
        ];

        for (role, system, user) in roles {
            assert!(!system.is_empty(), "{role} must expose its system templates");
            assert!(!user.is_empty(), "{role} must expose its user templates");
            let all = role_templates(&system, &user);
            for (i, template) in all.iter().enumerate() {
                assert!(
                    !template.is_empty(),
                    "{role}'s template {i} is empty; an empty template hashes to the \
                     e3b0c44298fc value this record exists to keep out"
                );
            }
            assert_ne!(
                crate::core::prompt_content_hash(&system),
                crate::core::prompt_content_hash(&all),
                "{role}'s recorded hash must cover its user templates, not only its system ones"
            );
            for i in 0..all.len() {
                let edited: Vec<String> = all
                    .iter()
                    .enumerate()
                    .map(|(j, t)| if j == i { format!("{t} ") } else { t.to_string() })
                    .collect();
                let edited_refs: Vec<&str> = edited.iter().map(String::as_str).collect();
                assert_ne!(
                    crate::core::prompt_content_hash(&all),
                    crate::core::prompt_content_hash(&edited_refs),
                    "{role}'s template {i} is not inside the hash: editing it changed nothing"
                );
            }
        }
    }

    /// [VANTA-WHY] M3 item 19 (fix 4) — the chain that makes a prompt edit move
    /// `config_hash`. `config_hash` is the SHA-256 of `configs/v1-m3.1.toml`, and
    /// the `[prompts]` block records this code's own hashes. Edit a template and
    /// this test fails; the only fix is to update `[prompts]`, which changes the
    /// file's bytes and therefore every row's `config_hash`. Without this link the
    /// block could silently describe text the binary no longer sends — exactly the
    /// gap item 16 fell into.
    #[test]
    fn the_committed_prompts_block_is_the_hash_of_the_templates_this_code_sends() {
        let raw = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/configs/v1-m3.1.toml"));
        let doc: toml::Value = toml::from_str(raw).expect("configs/v1-m3.1.toml must parse as TOML");
        let recorded = doc
            .get("prompts")
            .and_then(|value| value.as_table())
            .expect("configs/v1-m3.1.toml must carry a [prompts] table");

        let expected: [(&str, String); 5] = [
            (
                "attacker",
                crate::core::prompt_content_hash(&role_templates(
                    &crate::agents::attacker::attacker_system_prompts(),
                    &crate::agents::attacker::attacker_user_prompts(),
                )),
            ),
            (
                "intent_check",
                crate::core::prompt_content_hash(&role_templates(
                    &crate::agents::attacker::intent_auditor_system_prompts(),
                    &crate::agents::attacker::intent_auditor_user_prompts(),
                )),
            ),
            (
                "creator",
                crate::core::prompt_content_hash(&role_templates(
                    &[crate::agents::creator::creator_system_prompt()],
                    &crate::agents::creator::creator_user_prompts(),
                )),
            ),
            ("diagnostician", "not_applicable".to_string()),
            // [VANTA-WHY] C4 — the judge's prompt is hashed here too, though it is
            // NOT in `prompt_hashes()`: it is sent only on a judge arm, so the
            // every-row string must not name it, but `config_hash` must still be
            // able to see a reworded judge prompt. Its value comes from the one
            // accessor the driver records on the row, so the two cannot describe
            // different prompts.
            ("judge", crate::agents::judge::judge_prompt_hash()),
        ];

        assert_eq!(
            recorded.len(),
            expected.len(),
            "[prompts] must name exactly the prompt records the code has — `prompt_hashes()`'s \
             roles plus the arm-gated judge"
        );
        let mismatches: Vec<String> = expected
            .iter()
            .filter_map(|(role, want)| {
                let got = recorded.get(*role).and_then(|value| value.as_str());
                (got != Some(want.as_str())).then(|| format!("{role}: file={got:?} code={want}"))
            })
            .collect();
        assert!(
            mismatches.is_empty(),
            "configs/v1-m3.1.toml [prompts] must be the hashes this code sends, or `config_hash` \
             describes other text; update the file (which moves config_hash) or revert the \
             template edit. Mismatches: {mismatches:#?}"
        );
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
