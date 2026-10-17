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
