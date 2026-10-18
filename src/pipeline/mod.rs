// ============================================================================
// PIPELINE MODULE — BATE (Phase 1) + Diagnostician (Phase 2) +
//                   patch application + Governor (Phase 3) + Gatekeeper
// ============================================================================
pub mod bate;
pub mod candidate;
pub mod diagnostician;
pub mod driver;
pub mod harness;
pub mod patch;
pub mod governor;
pub mod gatekeeper;

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
