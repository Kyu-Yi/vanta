// ============================================================================
// AGENTS MODULE — Creator (writes patches) + Attacker (writes tests)
// ============================================================================
//
// Spec §4 Architecture: "two adversarial LLM agents — a Creator that writes
// patches and an Attacker that writes tests designed to break them."
//
// This module defines the TYPED request/response shapes for both agents.
// The actual LLM call logic (post_chat, extract_first_valid_json,
// structured_call_with_retry) lives in crate::llm (promoted from M0 bootstrap).
// ============================================================================

pub mod attacker;
pub mod creator;
// [VANTA-WHY] M3 item 7 — the judge arm's one-call verdict. Declared beside the
// two adversarial agents because it is the third "role that reads the issue and
// the code"; its prompt version and content hash are recorded on judge rows only.
pub mod judge;
