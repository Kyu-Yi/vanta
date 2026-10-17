# VANTA M0 SPIKE RESULT — GO/NO-GO Verdict
**Date:** 2026-09-09
**Model:** Qwen3.5-4B-UD-Q4_K_XL (single shard ~3GB GGUF, loaded into llama-server Metal)
**Provider/client:** llama.cpp llama-server OpenAI-compat endpoint + reqwest (Rust HTTP client) — rig-core 0.42.0 was TRIALLED and then REJECTED via kill switch (see [bootstrap/mod.rs](file:///Users/kyu-yi/vanta/src/bootstrap/mod.rs) inline comments)
**Verdict:** ✅ **GO** — proceed to M1 (Proof-of-Concept Governor)

---

## Executive summary
The M0 spike successfully demonstrates all 5 acceptance criteria required by the Vanta Master Plan 2 §M0 EXIT TEST, running 100% locally on a single M-series Mac with no frontier API key. All three milestones (unstructured chat, typed structured JSON, and a 2-agent chained pipeline) completed end-to-end against a locally-served GGUF model.

---

## Evidence (4 bullets, reproducible)
Run these exact commands from the repo root `/Users/kyu-yi/vanta` to reproduce:

1. **[M1] Unstructured completion works against llama-server:**
   - Command: `cargo run --bin vanta`
   - Observed: M1 block prints "✔ M1 OK" with a 100-char preview of the model's explanation of `Result<T, E>`. HTTP 200, ~17s wall-clock, 2044 tokens total (reasoning_content ~7.5k chars).

2. **[M2] Structured CreatorResponse typed JSON parses + malformed case handled without panic:**
   - Command: `cargo run --bin vanta` (same run, consecutive milestone)
   - Observed: M2 block prints "✔ M2 OK — files_touched = [\"src/discount.rs\"], confidence = 0.95" with a typed `CreatorResponse { patch_text, explanation, files_touched, confidence }` serde-typed parse. Builtin deliberate malformed-JSON case is caught as `Result::Err` with line/col info printed (no panic).

3. **[M3] 2-agent Creator→Attacker chain feeds typed output through a second model call:**
   - Command: `cargo run --bin vanta` (same run)
   - Observed: Creator returns typed patch → `patch_text` (66 chars) and `files_touched` are **extracted via serde_json::Value** and string-interpolated into a second HTTP request for Attacker → Attacker returns typed `AttackerResponse { tests: [AttackerTestCase] }` of exactly 2 tests (huge_cart, empty_cart) with `expect_pass=true` and inline Rust `#[test]` source code. All intermediate errors propagate up via `?`.

4. **Robustness pipeline handles small-model pathological output (prose wrappers, non-JSON brace pairs, missing content field):**
   - Command: unit tests + real run combined
   - Observed: 13/13 lib tests pass, including `extract_first_valid_json` multi-candidate scan + parse-attempts pattern, `extract_json_candidate` 4-stage cleaning pipeline, and `LlmConfig` defaults/traits. Real run scans 6+ non-JSON brace-balanced candidates in the reasoning block before finding and correctly parsing the real JSON content candidate that deserializes cleanly.

---

## Architecture decisions recorded here (for M1 carry-forward)

| Decision | Rationale | Future revisit trigger |
|----------|-----------|------------------------|
| **reqwest chosen over rig-core 0.42.0** (kill switch NO-GO applied at step 11) | Rig adds 1-2 extra layers of provider opinionation without yet exposing OpenAI-compat custom paths ergonomically enough for pedagogical M0 learning; reqwest code maps 1:1 onto the curl smoke test, making each header/body/field visible. | When M1 introduces 2+ LLM providers (frontier + local) and unified provider traits become worth the abstraction tax. |
| **`extract_first_valid_json` = brace-scan ALL candidates then first-parse-success wins** | Direct cure for Qwen3.5-4B writing prose/schema-examples/rust-snippets wrapped in `{…}` BEFORE the real JSON answer. Cheaper than 2nd-trip retry, survives every small-model output we hit. | When graduating to a provider that offers real grammar-constrained sampling (Outlines/llama.cpp grammars, Anthropic prompt caching with strict schemas, OpenAI response_format strict mode). |
| **Local defaults: `127.0.0.1:8080`, no API key, `Qwen3.5-4B` model alias** | Zero-setup reproducibility for M0. Spec explicitly requires local-model-first development. | Swapping for remote URL is trivial: `export VANTA_LLM_BASE_URL VANTA_LLM_MODEL VANTA_API_KEY VANTA_LLM_TIMEOUT_SECS`. |
| **Fusion policy: ALWAYS concatenate `content + reasoning_content` with section markers** | Qwen3.5 variants emit answers split across both fields (sometimes reasoning-only, sometimes content-only, sometimes split). Concat + bracket-robust scanner handles every split without needing per-call case analysis. | For frontier providers where `content` alone reliably holds final answer, this can be gated behind a config flag to save scan cost. |
| **Generous `max_tokens` on structured calls (4500–5000)** | UD "thing-heavy" reasoning variant writes 3000–4000 tokens of chain-of-thought in `reasoning_content` BEFORE emitting 100–600 tokens of actual JSON. If total budget < reasoning + content, `content` is EMPTY and parse fails 100% of the time. | Swap to a non-UD model or turn reasoning-level down. |

---

## Signature (M0 done — move to M1)
```
SPIKE_OWNER: Vanta spike binary src/bootstrap/mod.rs
DATE:        2026-09-09
VERDICT:     GO
REPRO_CMD:   cd /Users/kyu-yi/vanta && cargo run --bin vanta
```
