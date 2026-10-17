# M0 Completion Implementation Plan

## Repository Research

**Current project state (as of 2026-09-09):**
- Skeleton project only. `Cargo.toml` has no dependencies.
- `src/main.rs` thin entry point, calls `bootstrap::run()`
- `src/bootstrap/mod.rs` is a placeholder that prints "test"
- `src/config/mod.rs` is completely empty
- `src/core/mod.rs` has 3 enums with `#[derive(Debug)]` only: `FailureClass`, `TestIntent`, `VerificationOutcome` — correctly shaped per Spec §5/§6

**Environment confirmed — local models available (smallest first for testing):**
| Model | GGUF Location | Size (approx in RAM) | Use in M0 |
|---|---|---|---|
| **Qwen3.5-4B UD Q4_K_XL** (PRIMARY) | `models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176.../Qwen3.5-4B-UD-Q4_K_XL.gguf` | ~3-4 GB | **All spike milestones 1-3.** Fast, single shard, instant iteration. Swappable later via env var. |
| Qwen3-Coder-Next Q6_K (4 shards) | `models--Qwen--Qwen3-Coder-Next-GGUF/.../` | ~5+ GB | Backup. Try if 4B fails structured JSON consistently. |
| Qwen3.8-27B / 3.6-35B / Ornith-35B | Various | 20-40+ GB | **SKIP for M0.** Overkill for learning the Rust→LLM call pattern. Wasteful during 100+ compile-retry cycles. |
| llama-server binary | `/Users/kyu-yi/llama.cpp/build/bin/llama-server` | - | OpenAI-compatible HTTP endpoint on localhost:8080. Metal GPU-accelerated. |

This aligns **exactly** with Spec §1 property table: *"Model strategy: Local models (Ollama/llama.cpp) for development; frontier APIs for the tier experiments"*

**Milestone M0 per Master Plan 2 covers:**
1. **M0 STUDY phases 0.3–0.11** — Ownership through Rig, each validated by a build checkpoint
2. **M0 BUILD — the Rig spike (3-day hard stop with kill switch)** — 3 milestones: typed model call → structured JSON response → two agents chained
3. **M0 EXIT TEST** — "You can write a Rust program from scratch that calls [the model], gets typed JSON back, feeds it into a second call, and handles a malformed response without panicking. And the GO/NO-GO verdict is written down."

**Non-negotiable scope boundary (per Master Plan):**
- ❌ NO real Creator/Attacker Vanta prompts (that's M1)
- ❌ NO execution/sandbox/BATE (that's M2)
- ❌ NO repair loop/Gatekeeper/PR (that's M3+)

**NON-NEGOTIABLE COMMENT QUALITY (the user's #1 learning priority):**
Every checkpoint binary and every non-trivial module function must have super-intensive comments using this standardized taxonomy:

```
// ============================================================================
// BLOCK HEADER: Name of what this section teaches
// ============================================================================

// [SYNTAX]    — What the Rust syntax token is doing, dictionary-definition clear
// [VANTA-WHY] — Why Vanta specifically needs this pattern (spec reference when applicable)
// [PITFALL]   — Common mistake beginners make, what symptom it causes, how to spot
// [RECIPE]    — Copy-pastable pattern template you can reuse in any future code
// [EXAMPLE]   — A tiny unrelated code snippet showing the concept in isolation
// [TRACE]     — Walk-through of a concrete execution path (for async/Result/loops)
```

Plus, each checkpoint binary MUST have:
1. **TOP FILE BLOCK (≈40 lines)** — "What you'll learn from this binary" checklist, linking each Rust concept to its Vanta M1+ usage.
2. **BOTTOM FILE BLOCK (≈20 lines)** — "5 reusable recipes" that distill the binary's patterns into templates you can copy later.
3. **Inline line comments on EVERY non-trivial line of code.** No line with a `?`, `&`, `match`, `for`, `Vec::new()`, `derive` attribute, or type annotation goes unexplained.

---

## Files and Modules

| File | Expected Change |
|---|---|
| `Cargo.toml` | Add base dependencies: `serde` (+derive), `serde_json`, `tokio` (+full). Then either `rig-core` (exact `=` pin) or `reqwest` (+json) based on spike. |
| `src/core/mod.rs` | Add `#[derive(Serialize, Deserialize, Clone)]` + `#[serde(rename_all = "PascalCase")]` (or whatever tag style) demo comment to existing 3 enums. Add `EvidenceSource` enum (6 variants per Spec §8). Add `TestEvidence` struct. **Each derive attribute gets a [SYNTAX] and [VANTA-WHY] comment.** |
| `src/config/mod.rs` | **NEW CONTENT:** `LlmConfig` struct (base_url, model_id, api_key: Option, timeout). `LlmConfig::from_env() -> Result<Self, String>` reading `VANTA_LLM_BASE_URL`, `VANTA_LLM_MODEL`, `VANTA_API_KEY`. Defaults: `http://localhost:8080/v1` + 4B model + None api_key. Line-level comments on every field explaining why it exists. |
| `src/main.rs` | Convert to `#[tokio::main] async fn main()` to support async bootstrap from day one. Commented conversion showing why. |
| `src/bootstrap/mod.rs` | Rewrite as async spike entry point. Contains 3 spike milestones as separate functions: `spike_m1_single_call().await`, `spike_m2_structured_json().await`, `spike_m3_two_agent_chain().await`. Each returns `Result<(), Box<dyn Error>>`. Each milestone function has a 5-line header comment of what it proves. |
| `src/bin/checkpoint_a.rs` | **NEW — learning binary (highest comment density).** Vec<TestEvidence>, match triage, Result empty-Vec error. 40-line header, 5-recipe footer, line-by-line [SYNTAX]/[VANTA-WHY]/[PITFALL] comments. |
| `src/bin/checkpoint_b.rs` | **NEW — learning binary.** CreatorResponse serde roundtrip. Same comment density. Header + footer. Demonstrates why typed structs > raw JSON. |
| `src/bin/checkpoint_c.rs` | **NEW — learning binary.** tokio async + sequential vs `tokio::join!` concurrency with `Instant` timing. Same header/footer comment structure. Concrete timing output proves the runtime scheduler works. |
| `SPIKE_RESULT.md` | **NEW at repo root.** GO/NO-GO verdict + date + 3 milestone pass/fail evidence. Per Master Plan "Written verdict committed to the repo". |

---

## Implementation Steps (dependency-ordered)

### Phase A: Checkpoints A/B/C — the STUDY block with full-comment binaries

**Step 1.** Add base dependencies only (NO rig or reqwest yet — save those for spike phase):
```bash
cargo add serde --features derive
cargo add serde_json
cargo add tokio --features full
```
Run `cargo check` — must pass before proceeding.

**Step 2.** Convert `src/main.rs` to async:
```rust
// [SYNTAX] #[tokio::main] is an attribute macro that expands this async main
// into a synchronous main() that creates a Tokio runtime, blocks_on the async
// body, and returns the result. Without this, `async fn main()` would not
// compile because the binary entry point must be non-async (OS requirement).
//
// [VANTA-WHY] Vanta orchestrates N concurrent things: Creator LLM call,
// Attacker LLM call, N sandbox executions, verifier backends. All of them
// use async HTTP or process APIs. If we defer adding async until M2, we'd
// have to rewrite the entire module tree. Do it once now and never touch it again.
//
// [PITFALL] Forgetting to add features=["full"] on tokio. Without it tokio::time::sleep
// doesn't exist and you get a cryptic compile error. Always enable full during dev.
#[tokio::main]
async fn main() {
    // … same body, but bootstrap::run can now be async
}
```
Also convert bootstrap `run()` to `pub async fn run()` with a TODO await. Compiles = step done.

**Step 3.** Write `src/bin/checkpoint_a.rs`. Must satisfy ALL of:

**HEADER BLOCK (40 lines minimum):**
```
// ============================================================================
// CHECKPOINT A — Ownership, Collections, Enums, Pattern Matching, Result
// ============================================================================
//
// What you'll learn from this binary (Rust concept → where Vanta reuses it):
//
//  1. Vec<T> collections        → M2: storing 100+ generated tests and evidence
//  2. &T borrowing vs owned T   → Everywhere: passing evidence without clones
//  3. match on enums (exhaust)  → Diagnostician 4-way triage, Gatekeeper verdicts
//  4. Option<T>                 → failure_class is None for passing tests
//  5. Result<T,E> + ? operator  → Every orchestration function (no panics)
//  6. main() -> Result          → Errors bubble up as pretty messages, not panics
//  7. Counter variables         → EIR/ECR counting in Governor (M3/M4)
//
// You will reuse EVERY line pattern below in M1–M5. Type it out and think about it.
// ============================================================================
```

**ACTUAL CODE (fully commented):**
Every line with a non-keyword token gets a comment. Examples of what must be commented:
- `use vanta::core::*;` → [SYNTAX] + [PITFALL] (glob use OK in binaries; bad in libs)
- `#[derive(Debug)]` on TestEvidence → [SYNTAX] (what derive generates at compile time) + [VANTA-WHY] (we println!("{:?}", evidence) during debugging)
- `Vec::new()` → [SYNTAX] (empty heap-allocated growable buffer) + [RECIPE] (pattern: create, push, iterate)
- `evidence.push(...)` → [VANTA-WHY] (Attacker writes N tests; this is how we accumulate them M2)
- `for ev in evidence.iter()` → [SYNTAX] iter() vs into_iter() vs iter_mut() + [PITFALL] (if you use into_iter you move and can't print after)
- `match &ev.outcome { ... }` → [SYNTAX] (matching behind a reference with `&`) + [VANTA-WHY] (Diagnostician Phase 2 triage is exactly this, 6→4 classification)
- Counter increments with `verified_within_bounds_count += 1;` → [VANTA-WHY] Governor EIR/ECR = exactly this pattern (M4)
- `return Err("No evidence to triage! empty vec!".to_string());` → [SYNTAX] to_string() converts str literal to owned String for the Err variant
- `triage_summary(&evidence)?;` → [SYNTAX] `?` expands to `match val { Ok(v)=>v, Err(e)=>return Err(e.into()) }` + [VANTA-WHY] orchestration chains 10+ Results in a row without nested match (M2-M3)
- `triage_summary(&[])?;` → [TRACE] Walk through what happens: empty check fires → Err returns → `?` in main returns Err → Rust runtime prints the Err message and exits non-zero WITHOUT a stacktrace. That's the behavior being tested.

**FOOTER BLOCK (20 lines minimum):**
```
// ============================================================================
// 5 REUSABLE RECIPES FROM CHECKPOINT A
// ============================================================================
//
// RECIPE A1 — "Process a collection and count things, error if empty"
//   fn summarize(items: &[Thing]) -> Result<Counts, String> {
//       if items.is_empty() { return Err("empty!".into()); }
//       let mut counts = Counts::default();
//       for item in items { /* match + increment */ }
//       Ok(counts)
//   }
//   → USE THIS in: Diagnostician failure counts, Attacker discard metrics (M2)
//
// RECIPE A2 — "Exhaustive match on N outcomes, grouping some arms"
//   match outcome {
//       A | B => handle_together(),
//       C     => handle_c(),
//       _     => unreachable!(), // or a default
//   }
//   → USE THIS in: VerificationOutcome → Gatekeeper policy (M3)
//
// RECIPE A3 — "Borrowed slices (&[T]) as function input parameters"
//   → Always accept &[T] in functions. Caller can pass &vec or &array or a slice
//   → No clone/copy needed. This is 90% of Rust fn args for collections.
//
// RECIPE A4 — "main() returns Result, propagated via ?"
//   fn main() -> Result<(), String> { ... thing()?; ... Ok(()) }
//   → Use in EVERY single binary you ever write. No unwrap() in main.
//
// RECIPE A5 — "Option<T> for fields that may not exist"
//   struct Record { value: Option<i32> }
//   if let Some(v) = record.value { ... }  or match record.value { ... }
//   → USE THIS in: TestEvidence.failure_class (passing tests = None)
// ============================================================================
```

Compile & run. Expected output: ~15 lines of per-test triage + 4-line count table + 1 line Err string from empty call. **NO RUST_BACKTRACE anywhere in output.**

**Step 4.** Update `src/core/mod.rs`:
- Add `#[derive(Serialize, Deserialize, Clone)]` to all three enums.
- Each derive line gets its own comment:
  ```rust
  // [SYNTAX] #[derive(Debug)] generates a fmt::Debug impl. It lets you write
  // println!("{:?}", enum_variant) instead of manually implementing display.
  // [VANTA-WHY] Every evidence record in the JSON logs (M5 raw data) is Debug-printable
  // during development when we need to quickly inspect values in a terminal.
  #[derive(Debug)]
  // [SYNTAX] #[derive(Serialize)] generates a serde::Serialize impl at compile time.
  // It knows how to turn this enum into a JSON string via serde_json::to_string().
  // [VANTA-WHY] Session logs (Spec §4 institutional memory, M1 session data model)
  // are flat JSON incident files. Every enum stored on disk must be serializable.
  #[derive(Serialize)]
  // [SYNTAX] Deserialize is the reverse: turns JSON string back into enum value.
  // [VANTA-WHY] Re-running a previous session from disk (regression on a Vanta run)
  // requires loading saved evidence back into typed Rust structs — not parsing raw strings.
  #[derive(Deserialize)]
  // [SYNTAX] Clone generates .clone() — a deep copy of the enum value into new memory.
  // [VANTA-WHY] The same TestEvidence is referenced from both the live pipeline and
  // the governor's fixed shared set. Cloning avoids borrow checker fights; for ~64-byte
  // structs the cost is irrelevant vs correctness.
  #[derive(Clone)]
  pub enum FailureClass { ... }
  ```
- Add `EvidenceSource` enum (6 variants, same derives, commented). Add `TestEvidence` struct (5 fields: test_name, outcome, failure_class, intent, evidence_source, all derives). **Each field gets a [VANTA-WHY] comment linking to Spec §8 provenance rules.**
- Now go back to checkpoint_a.rs, DELETE its local TestEvidence definition, and replace with `use vanta::core::TestEvidence;`. Recompile & run. It must still work identically. This proves you promoted a throwaway type to a real reusable library type correctly.

**Step 5.** Write `src/bin/checkpoint_b.rs` with identical HEADER/FOOTER block structure. Header must list the 5 things it teaches (Serialize impls generated, Deserialize symmetric, typed vs Value::Object, from_str Result error handling, why schema-based JSON beats freeform). Same line-level [SYNTAX]/[VANTA-WHY]/[PITFALL] comment density. Code shape:
- Define CreatorResponse struct locally first (to show the pattern), then note the TODO-promote-to-core note.
- `serde_json::to_string_pretty(&resp).unwrap()` in a context where we KNOW it won't fail (no unrepresentable types) + comment why unwrap() is safe here but never on user/LLM input.
- Roundtrip: assert_eq!(deserialized.patch_text, original.patch_text)
- **Malformed test:** create a json str with a missing comma → `from_str::<CreatorResponse>(&bad_json)` → `match` arm prints the Serde error line-and-column info. Do NOT let panics happen. `Result` all the way.
- Footer must have 5 RECIPES including "RECIPE B3 — Always deserialize into a typed struct, never HashMap<String, Value>. The compiler is your schema validator."

**Step 6.** Run checkpoint_b. Confirm: prints pretty JSON, says "Roundtrip OK" on valid, prints structured Serde error with line/col on malformed case. No panics anywhere.

**Step 7.** Write `src/bin/checkpoint_c.rs`. Same structure. The key learning artifact is the **timing output**. Use `std::time::Instant` visibly:
```rust
// [RECIPE] Measure any async block's duration:
//   let t0 = Instant::now();
//   do_thing().await;
//   println!("took {}ms", t0.elapsed().as_millis());
// → You will use this in M4 for per-patch-attempt timing in the EIR/ECR tables.
```

Show BOTH:
1. **Sequential:** `.await` two calls back-to-back → duration ~= sum (~1000ms for two 500ms sleeps)
2. **Concurrent:** `tokio::join!(a, b)` → duration ~= max (~500ms for two 500ms tasks running on the same runtime concurrently)

The user must SEE the numeric difference printed. That's what makes async "click" — it's not theoretical. You can see the runtime scheduler interleave both sleeps onto one thread.

Footer recipes include "RECIPE C1 — Sequential vs concurrent pattern: use join!() when tasks are independent". And "RECIPE C2 — tokio::main + async fn pattern — the ONLY binary entry point you need from here on."

**Step 8.** Run checkpoint_c. Confirm times match expectations (sequential ≈ 1000ms, concurrent ≈ 500ms). Variance up to 200ms is fine for OS scheduling noise.

---

### Phase B: LlmConfig + Rig spike (with kill switch to reqwest)

**Step 9.** Implement `src/config/mod.rs` with full-commented `LlmConfig`:

```rust
// ============================================================================
// LlmConfig — Model provider abstraction (the first trait-adjacent pattern)
// ============================================================================
//
// Core idea: The rest of Vanta NEVER hardcodes "localhost:8080" or a model ID.
// Everything reads from this struct. That means when we switch from 4B local
// to Claude Opus for the paper experiments (M5), we change ENV VARS only.
// Zero code edits. That's the [VANTA-WHY] of a config struct.
// ============================================================================

use std::env; // [SYNTAX] std::env::var reads a shell environment variable by name.

#[derive(Debug, Clone)]
// [SYNTAX] Debug so we can println!("{:?}", config) during development without
// leaking the api_key in logs. Oh wait — Debug prints all fields by default!
// [PITFALL] If you derive Debug on a struct with secrets, anyone who sets
// RUST_LOG=debug gets your key in logs. Fix: either manually implement Debug
// to redact the key, OR wrap api_key in a SecretString newtype that redacts Debug.
// For M0 we'll accept the risk since api_key is None for local. TODO for M1.
pub struct LlmConfig {
    // [VANTA-WHY] llama-server is OpenAI-compatible. So is Anthropic's newer API.
    // By parameterizing the base URL, one Rust code path hits both providers
    // and any future provider (Ollama, Fireworks, Together, OpenRouter...).
    pub base_url: String,

    // [VANTA-WHY] Each experiment in M5 uses a DIFFERENT model. Hardcoding an ID
    // means you'd have to diff source code to reproduce an experiment run. The
    // paper needs reproducibility: provider ID, model release version, run date,
    // prompt locked. Spec §8 Planned evaluation says it explicitly.
    pub model_id: String,

    // [SYNTAX] Option<String> = either Some(owned_string) or None.
    // [VANTA-WHY] Local llama-server needs NO key. Frontier APIs (Anthropic etc.)
    // DO need a key. Option expresses "this field may legitimately be absent"
    // without inventing a sentinel empty string that could be misused.
    pub api_key: Option<String>,

    // [VANTA-WHY] Timeouts are an Infrastructure failure class (Spec §5 Phase 2).
    // We need a timeout because M2 sandboxes kill on timeout. The LLM layer is
    // the same: a hanging request is classified Infrastructure, not Logic.
    // If you don't set one, reqwest defaults to infinite = benchmark corruption.
    pub request_timeout_secs: u64,
}
```

Then `impl LlmConfig { pub fn from_env() -> Result<Self, String> { ... } }` with comments on env::var pattern, ok() conversion to Option for api_key, default values, etc.

**Step 10.** Start llama-server in a separate terminal tab BEFORE writing Rust spike code. Verifies the model actually loads:
```bash
/Users/kyu-yi/llama.cpp/build/bin/llama-server \
  --model "/Users/kyu-yi/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf" \
  --ctx-size 4096 \
  --port 8080
```
Wait for `"HTTP server listening"` log line. Keep open. Quick test with curl in another tab:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"say hello in 3 words"}]}'
```
Returns valid JSON with assistant content = proceed. If curl fails, stop. Fix the server before any Rust spike work.

**Step 11.** **Spike Milestone 1 — Rig first, pinned exact version.**
- Check latest Rig version: `cargo search rig-core`. Then: `cargo add rig-core --vers "=X.Y.Z"`. (The `=` is NON-NEGOTIABLE for pre-1.0 per Master Plan spike dependency rules.)
- Rewrite `bootstrap/mod.rs` into async:
  - Imports LlmConfig.
  - `spike_m1_single_call(config: &LlmConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>>`.
  - Box<dyn Error + Send + Sync> = "any error type that's thread-safe and printable". Comment why this is the idiomatic `main` error type.
  - Use Rig's OpenAI-compatible provider constructor. Set base_url to config.base_url. Set api_key if present.
  - Send one message: "Say hello in exactly 3 words with no punctuation."
  - Print the raw response string.
- Run `cargo check` first (fast) then `cargo run`.

**NO-GO KILL SWITCH CRITERIA (exact from Master Plan M0 BUILD §THE KILL SWITCH):**
- ❌ Examples in Rig docs don't compile as-written
- ❌ API surface doesn't match docs (method renamed, args missing)
- ❌ Structured output path won't work (trait bound we can't satisfy, missing schema parameter)
- ❌ Provider-specific errors we can't debug past after 30 mins of trying

IF ANY KILL SWITCH TRIGGERS:
1. Remove rig-core from Cargo.toml: `cargo rm rig-core`
2. Run: `cargo add reqwest --features json`
3. Implement spike with raw HTTP POST to `/v1/chat/completions` using `reqwest::Client`. JSON body built with `serde_json::json!({...})` macro. This ALWAYS works against any OpenAI-compatible endpoint.
4. Write the exact failure symptom into SPIKE_RESULT.md evidence bullets.
5. This is still a successful spike. We answered the question cheaply.

**Step 12.** Spike Milestone 2 — Structured output as Result.
- Define/promote `CreatorResponse` struct with `#[derive(Debug, Serialize, Deserialize, Clone)]`:
  - `patch_text: String`, `explanation: String`, `files_touched: Vec<String>`, `confidence: f32`
- Prompt:
  ```
  System: You are a code patching AI. Return ONLY valid JSON matching this schema:
  {"patch_text": string, "explanation": string, "files_touched": [string], "confidence": float 0-1}
  Do not include markdown fences. Do not write text outside the JSON object.
  
  User: The bug is that add(2,3) returns 4. Fix it in src/math.py.
  ```
- Add `response_format: {type: "json_object"}` to the request body (llama-server supports this).
- Receive the raw string. **Parse it:** `serde_json::from_str::<CreatorResponse>(&raw_text)`. This is a `Result`.
  - If Ok(typed): print each field separately via `println!("Patch: {}", typed.patch_text)` — proves we're accessing the typed STRUCT fields, not JSON["patch_text"].
  - If Err(e): print `⚠ Creator response parse error: {e}` — ERROR, not panic. Return Ok(()) from the function normally after reporting. This is the handling; the exit test explicitly requires this.
- **Deliberate malformed test case:** separately parse a deliberately broken JSON string through the same function to prove error handling works even when model "never fails" in practice.

**Step 13.** Spike Milestone 3 — Two-agent chain with error propagation.
- Define `AttackerTestSummary`: `name: String, intent: String` (using String for intent in spike; M1 upgrades to the enum).
- Define `AttackerResponse`: `tests: Vec<AttackerTestSummary>`.
- Chain:
  ```
  CreatorResponse (from M2, typed)
    → feed patch_text + explanation as user message to Attacker call
    → Attacker prompt: "Given this patch, write 2 adversarial tests. Return JSON: {\"tests\": [{\"name\": string, \"intent\": string}]}"
    → Parse Attacker output into AttackerResponse.
    → Print Creator patch and BOTH Attacker tests clearly labelled.
  ```
- Use `?` to propagate any error upward. A single `?` at any stage short-circuits and reports the exact failure to the user.
- Run end to end.

**Step 14.** Write `SPIKE_RESULT.md` at repo root:
```markdown
# SPIKE RESULT — Rig vs reqwest (M0)
Date: 2026-09-09
Default test model: Qwen3.5-4B-UD-Q4_K_XL (local llama-server http://localhost:8080)
Rig version tested: X.Y.Z  (or N/A if reqwest was chosen immediately)

Verdict: [RIG GO | RIG NO-GO, switching to reqwest]

Evidence:
- [concrete bullet 1 re: whether Rig compiled and docs matched API]
- Spike milestone 1 (single typed call through selected provider): PASS / FAIL reason
- Spike milestone 2 (structured JSON → CreatorResponse typed struct, malformed handled via Result): PASS / FAIL reason
- Spike milestone 3 (Creator → Attacker two-agent chain, error propagation via ?): PASS / FAIL reason
```

---

### Phase C: Exit test confirmation

**Step 15.** Run the M0 EXIT TEST checklist in order:
1. `cargo run --bin checkpoint_a 2>&1 | grep -E "(BugReproduction|count|No evidence)"` — outputs without backtrace → ✓
2. `cargo run --bin checkpoint_b 2>&1 | grep -E "(Roundtrip|Malformed handled)"` → ✓
3. `cargo run --bin checkpoint_c 2>&1 | grep -E "(Sequential|Concurrent)"` → times show concurrent faster → ✓
4. `cargo run 2>&1` — bootstrap runs all 3 spike milestones → Creator patch typed, Attacker tests typed, malformed handled → ✓
5. `ls -la /Users/kyu-yi/vanta/SPIKE_RESULT.md` → file exists with verdict → ✓
6. `cargo check --all-targets 2>&1 | tail -5` — **0 warnings** (allow only upstream dep warnings, nothing from our code) → ✓

All 6 items = M0 complete. You have typed LLM calls + error handling + Rust fundamentals in your fingers. M1 starts next session.

---

## Dependencies and Considerations

1. **Small model = fast feedback loops.** Qwen3.5-4B UD Q4_K_XL loads in <10 seconds on M5 Max Metal. If a compile cycle takes 5s and model inference takes <3s, you can iterate 60 times per hour. With 27B/35B models that drops to ~5/hour. **Small model for development; big models only for M5 benchmark runs.** This is explicitly in the spec.

2. **Swappability via env vars is the whole config's reason for existing.** VANTA_LLM_BASE_URL and VANTA_LLM_MODEL are the ONLY two things you change to run on a frontier API later. When you start M5 experiment sweeps you write a shell loop that exports those vars. No recompilation per model.

3. **Structured output = schema in prompt + json_object flag + Result parse.** Even with both, small local models can return bad JSON occasionally. The spike does NOT fail if Qwen4B occasionally returns an unparseable string. The Rust code correctly handling it via Result IS the pass criterion. M0 exit test says explicitly: "handles a malformed response without panicking". A correct Err path is a passing test.

4. **Edition 2024 is fine.** Cargo.toml already has edition = "2024". If it causes any crate compatibility issue, flip to 2021. It doesn't affect any M0 code.

5. **No junk drawer files.** Per user profile: no utils.rs, no helpers.rs. Checkpoints go in `src/bin/` (each a standalone learning artifact). Real types go in `core::`. Config goes in `config::`. Spike goes in `bootstrap::`. This is the module layout from day one.

6. **Every checkpoint binary is a standalone compilation unit.** This is why `src/bin/*.rs` exists in Cargo. You can `cargo test --bin checkpoint_a` later, or even `cargo run --example ...` but bin is simpler.

---

## Validation

After each step:
| Step | Command | Pass Condition |
|---|---|---|
| 1 | `cargo check` | No errors, no warnings |
| 3 | `cargo run --bin checkpoint_a 2>&1` | Per-test triage + counts. Last line is Err string. NO "thread 'main' panicked" substring anywhere |
| 5 | `cargo run --bin checkpoint_b 2>&1` | Pretty JSON printed. "Roundtrip OK". Then malformed case prints "Malformed handled: ..." with Serde error containing `line X column Y` |
| 7 | `cargo run --bin checkpoint_c 2>&1` | Sequential: ~900-1100ms. Concurrent: ~400-600ms. Concurrent < Sequential numerically |
| 10 | curl command above | JSON with valid assistant content field returned in <5s |
| 11 | `cargo run` + milestone 1 call | Model replies in <5s with "Hello there friend" (or similar 3 words). No compile errors. |
| 12 | Milestone 2 test | CreatorResponse.patch_text printed as typed field access (not JSON["..."]). Malformed test case: prints Serde error, program exits normally (exit 0). |
| 13 | Milestone 3 chain | Prints Creator patch, then prints AttackerResponse with Vec of 2 tests (each name + intent). Whole pipeline works via ? error propagation. |
| 14 | SPIKE_RESULT.md | Present. Contains today's date. Contains explicit GO/NO-GO verdict string. Contains 4 evidence bullets. |

---

## Risks and Handling

| Risk | Likelihood | Impact | Handling |
|---|---|---|---|
| Rig's OpenAI provider has no documented base_url override or the override doesn't actually hit localhost:8080 correctly | Medium | Milestone 1 blocks | Apply kill switch immediately. reqwest + serde_json::json!() body is 15 lines of straightforward code. It has never failed to hit an OpenAI-compatible endpoint in my experience. Document the specific Rig limitation in SPIKE_RESULT.md with the compile error. |
| Qwen3.5-4B ignores system prompt and returns markdown-fenced JSON ```json ... ``` instead of raw JSON, breaking from_str 50% of the time | High | Milestone 2 hits Err on every run | Mitigation in order: (1) Add a 3-line string preprocessing step: strip ```json\n prefix, strip ``` suffix, trim(). This is an Attacker/Creator output normalization pattern you'll use in M1 anyway. Do it M0 and comment heavily. (2) Prompt says "IMPORTANT: Do NOT use markdown code fences. Do NOT write any explanatory text. Return ONLY the raw JSON object starting with {". (3) Even with these, accept that some runs hit Err — that's exactly why Result handling exists, and it's what the exit test asks us to prove. You pass by handling the Err correctly, not by making the model perfect. |
| llama-server loads model but Metal isn't actually used (slow inference) → 4B takes 30s per call → iteration dies | Low | 10x slowdown | Check llama-server startup log for Metal backend init line. If not present, recompile llama.cpp with `-DLLAMA_METAL=ON` CMake flag. Not Vanta code's problem. If truly stuck, checkpoint binaries still validate 100% of M0 STUDY without any running LLM. The 3 spike milestones are the only LLM-dependent parts. Proceed with mocked LLM responses (a `#[cfg(test)]` or a feature flag that returns hardcoded JSON strings) for the spike code compile correctness check, then do a single real LLM run at the end. |
| "Box<dyn Error>" trait object has a lifetime/compile error in the spike functions | Medium | Milestone 1-2 stuck | The full signature is `Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>`. Always add Send + Sync, and 'static if the compiler complains. The `anyhow` crate is the normal solution, but M0 avoids adding more crates. Write the full signature once in a type alias: `type AppResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;` and use `AppResult<()>` everywhere. |
| `main()` returning Result formats the error ugly under edition 2024 | Very low | Cosmetic | If needed, add a manual match in main: `match real_main() { Ok(()) => {}, Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); } }` |
