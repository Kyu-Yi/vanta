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
/// The Diagnostician is rule-based and has no system prompt of its own (its
/// Unknown path sends the Attacker's simplify prompt, already covered above), so
/// its hash is the hash of the empty prompt set. It is recorded rather than
/// omitted so the four keys keep a fixed shape in every log.
pub fn prompt_hashes() -> String {
    let attacker = crate::core::prompt_content_hash(
        &crate::agents::attacker::attacker_system_prompts(),
    );
    let intent_check = crate::core::prompt_content_hash(
        &crate::agents::attacker::intent_auditor_system_prompts(),
    );
    let creator = crate::core::prompt_content_hash(&[crate::agents::creator::CREATOR_SYSTEM_PROMPT]);
    let diagnostician = crate::core::prompt_content_hash(&[]);

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
            // The Diagnostician is rule-based and sends no system prompt of its
            // own, so its slot is the hash of the empty set rather than absent.
            crate::core::prompt_content_hash(&[]),
        ];

        for ((role, hash), expected) in fields.iter().zip(expected) {
            assert_eq!(
                *hash, expected,
                "`{role}`'s recorded hash must be the hash of the prompts that role sends"
            );
            assert!(
                hash.len() == 12
                    && hash
                        .chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "`{role}`'s hash must be 12 lowercase hex characters; got {hash:?}"
            );
        }
    }
}
