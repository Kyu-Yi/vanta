# M1 + M2 Implementation Plan — The Adversarial Core & Ground Truth

## Repository Research

**Current state (M0 complete, validated):**
- Mixed lib+bin crate. `src/lib.rs` declares `pub mod bootstrap; pub mod config; pub mod core;`.
- `core/mod.rs` already has ALL 4 domain enums (`FailureClass`, `TestIntent`, `VerificationOutcome`, `EvidenceSource`) + `TestEvidence` struct, each with 5 derives + heavy pedagogical comments. These are the permanent semantic backbone.
- `config/mod.rs` has `LlmConfig { base_url, model_id, api_key: Option<String>, request_timeout_secs }` with `from_env()`, Display (redacts key), `chat_completions_url()`. Default model is now lowercase `qwen3.5-4b` to match trae-locals ini.
- `bootstrap/mod.rs` contains the M0 spike: `CreatorResponse`, `AttackerResponse`, `AttackerTestCase` types (defined locally — must be PROMOTED to `agents/` modules), `run()` → m1→m2→m3 chain, plus the production-grade helpers: `build_reqwest_client`, `post_chat` (fusion: content+reasoning_content), `extract_first_valid_json` (brace-balanced multi-candidate scanner + serde try-parse each), `structured_call_with_retry<T>` (N-attempt loop + repropmt tails).
- `bin/checkpoint_a/b/c.rs` — learning binaries with 40-line HEADER + 5-recipe FOOTER pattern. This pattern MUST be replicated for checkpoint D (M1) and checkpoint E (M2).
- No `.py` files exist yet. M1/M2 exit tests require a sample Python bug file.

**Local model stack (confirmed working):** trae-locals router on `127.0.0.1:8080/v1`, default model `qwen3.5-4b` (case-corrected). 6 models available via `VANTA_LLM_MODEL` env var swap.

**M1 scope (per Master Plan §M1):**
- Creator agent (issue+source → patch), Attacker agent (patch → intent-labelled tests)
- Session data model with IDs on every record, attempt log to disk as JSON
- One-shot pipeline: issue → Creator → Attacker → log
- CHECKPOINT D: TestIntent + GeneratedTest + BugReproduction/passing-baseline mismatch assertion
- EXIT TEST: one command against a real Python bug → patch + intent-labelled tests in log file. Nothing executed.
- DO NOT build: running tests, retry loop (that's M2), sandbox, EIR/ECR.

**M2 scope (per Master Plan §M2):**
- `SandboxBackend` trait (spawn/execute/destroy) + subprocess.rs (fast path) + shuru.rs (macOS microVM)
- Phase 1 BATE: run each generated test on UNMODIFIED code, compare to declared intent; mismatch → retry max 3 → scrap + log reason; record baseline; log discard rate
- Phase 2 four-way Diagnostician: `FailureClass` (Infrastructure/Static/Logic/Unknown). Static = ruff+mypy BEFORE execution, differenced against baseline. Structured JSON payload back to Creator.
- Evidence provenance (already in core: EvidenceSource, TestEvidence)
- CHECKPOINT E: copy dir → temp → pytest --json-report → 30s timeout → parse JSON → print per-test pass/fail
- EXIT TEST: feed patch → every test baseline-validated, invalid repaired/scrapped, valid run against patched, failures tagged. Kill network mid-run → classified as Infrastructure (not model error).
- DO NOT build: Linux/Firecracker, formal verification, Graphiti, live EIR/ECR.

**Teaching constraints (non-negotiable, from user):**
- Same comment taxonomy: `[SYNTAX]`, `[VANTA-WHY]`, `[PITFALL]`, `[RECIPE]`, `[EXAMPLE]`, `[TRACE]`.
- Every new checkpoint binary (D, E) MUST have the 40-line HEADER (learning outcomes) + 20-line FOOTER (5 copy-paste RECIPE blocks).
- Explain where everything connects (cross-reference file:line links, "this is used by X in module Y").

---

## Files and Modules

### New files (M1)
| File | Purpose |
|------|---------|
| `src/logging/mod.rs` | `init_logging()` — reads `VANTA_LOG_LEVEL`, installs `tracing_subscriber::fmt` subscriber. Called by every binary's `main()`. |
| `src/llm/mod.rs` | Promoted LLM helpers from bootstrap: `post_chat`, `build_reqwest_client`, `extract_first_valid_json`, `extract_json_candidate`, `strip_markdown_fences`, `structured_call_with_retry<T>`. Shared by Creator, Attacker, and M0 bootstrap spike. |
| `src/agents/mod.rs` | `pub mod creator; pub mod attacker;` — re-export shared agent types |
| `src/agents/creator.rs` | `CreatorRequest { issue_text, source_files, prior_feedback: Option<String> }`, `CreatorResponse { patch_text, explanation, files_touched, confidence }`, `creator_generate_patch()` — Verify-First system prompt. Reuses `post_chat` + `structured_call_with_retry` from bootstrap (promoted to shared). |
| `src/agents/attacker.rs` | `AttackerRequest { patch_text, original_source, issue_text }`, `GeneratedTest { test_name, test_source, intent: TestIntent, reason }`, `AttackerResponse { tests: Vec<GeneratedTest> }`, `attacker_generate_tests()`. Reuses `TestIntent` from core. |
| `src/session/mod.rs` | `SessionId`, `PatchAttemptId`, `TestingIterationId`, `TestId` (newtype wrappers around String/Uuid). `SessionLog` struct holding Vec of attempt records. `write_session_log()` → JSON to `vanta_sessions/{session_id}.json`. |
| `src/bin/checkpoint_d.rs` | M1 CHECKPOINT D learning binary. 40-line header + 5-recipe footer. Asserts: a `BugReproduction` test whose baseline result is PASS is classified as a BATE mismatch. |
| `src/bin/vanta_m1.rs` | M1 one-shot pipeline binary. Reads a Python file + issue description, runs Creator → Attacker, writes session log to disk. |
| `examples/buggy_python/cart.py` | Sample Python file with a real integer-overflow bug (M1/M2 exit test target). |
| `examples/buggy_python/issue.md` | Sample issue description. |

### New files (M2)
| File | Purpose |
|------|---------|
| `src/sandbox/mod.rs` | `SandboxBackend` trait: `spawn() -> Result<Self>`, `execute(&self, cmd: &[&str], cwd: &Path) -> Result<ExecutionResult>`, `destroy(self)`. `ExecutionResult { stdout, stderr, exit_code, wall_time_ms, timed_out: bool }`. `SandboxKind` enum (Subprocess / Shuru). |
| `src/sandbox/subprocess.rs` | `SubprocessBackend` — copies target repo to temp dir, runs command with `tokio::process::Command`, enforces timeout via `tokio::time::timeout`, kills process group on timeout. Network disabled via env (documented as best-effort on macOS). |
| `src/sandbox/shuru.rs` | `ShuruBackend` — spawns Shuru microVM via CLI, copies repo in, executes, destroys. |
| `src/pipeline/mod.rs` | `pub mod bate; pub mod diagnostician;` |
| `src/pipeline/bate.rs` | Phase 1 BATE: `validate_test_against_baseline(test, baseline_result) -> BateOutcome { Valid, Mismatch(Reason) }`. Intent→expected-baseline table. Retry loop max 3, then scrap + log reason. `BaselineResult { passed: bool, raw_output: String }`. |
| `src/pipeline/diagnostician.rs` | Phase 2 Diagnostician: `DiagnosticResult { failure_class: Option<FailureClass>, reason: String, structured_payload: Value }`. Classification logic: exit codes/signals/timeouts → Infrastructure; ruff/mypy failures (new vs baseline) → Static; assertion failure → Logic; unclassifiable → Unknown. |
| `src/bin/checkpoint_e.rs` | M2 CHECKPOINT E learning binary. 40-line header + 5-recipe footer. Copies a dir to temp, runs `pytest --json-report` with 30s timeout, parses JSON, prints per-test pass/fail. |
| `src/bin/vanta_m2.rs` | M2 full pipeline binary: Creator → Attacker → BATE baseline validation → Diagnostician on patched tree → session log. |

### Modified files
| File | Change |
|------|--------|
| `src/lib.rs` | Add `pub mod logging; pub mod llm; pub mod agents; pub mod session; pub mod sandbox; pub mod pipeline;`. Keep existing mods. |
| `src/bootstrap/mod.rs` | Remove local `CreatorResponse`/`AttackerResponse`/`AttackerTestCase` definitions. Replace with `use crate::agents::creator::CreatorResponse;` etc. (promote types to shared). Keep all M0 spike logic + helpers intact. |
| `src/config/mod.rs` | Add `retry_attempts: u32` field (default 3) to `LlmConfig`. Used by `structured_call_with_retry`. |
| `Cargo.toml` | Add `tempfile`, `uuid` (v4, no_std), `thiserror` (clean error enums in M2), `tracing` + `tracing-subscriber` (structured logging across all phases), `tokio` process already via full. |

---

## Implementation Steps (dependency-ordered)

### Phase A — M1 Foundation (types + session model)

**Step 0.** Logging infrastructure: add `tracing` + `tracing-subscriber` to Cargo.toml. Create `src/logging/mod.rs` with `init_logging()` that reads `VANTA_LOG_LEVEL` (default `info`) and installs a `tracing_subscriber::fmt` subscriber with timestamps + target. Every binary calls `vanta::logging::init_logging()` at the top of `main()`. Run `cargo check`.

**Step 1.** Add M2-needed deps now (tempfile, uuid, thiserror) so they're available throughout. Run `cargo check`.

**Step 2.** Create `src/session/mod.rs`: ID newtypes + `SessionLog` + `write_session_log()`. Include `[SYNTAX]` comments on newtype pattern, `[VANTA-WHY]` on provenance (Spec §8).

**Step 3.** Promote agent types: create `src/agents/mod.rs`, `src/agents/creator.rs` (CreatorRequest, CreatorResponse, creator_generate_patch with Verify-First prompt), `src/agents/attacker.rs` (AttackerRequest, GeneratedTest with `intent: TestIntent`, AttackerResponse, attacker_generate_tests). Move the `post_chat`, `extract_first_valid_json`, `extract_json_candidate`, `structured_call_with_retry` helpers OUT of bootstrap into a new shared location `src/llm/mod.rs` (so both agents and bootstrap reuse them). Update bootstrap to `use crate::llm::*`.

**Step 4.** Update `src/lib.rs` to declare the new modules. Update `bootstrap/mod.rs` imports. Run `cargo check --all-targets` → 0 warnings.

**Step 5.** CHECKPOINT D: `src/bin/checkpoint_d.rs` — TestIntent enum usage + GeneratedTest struct + `#[test]` asserting BugReproduction test with passing baseline = BATE mismatch. 40-line header + 5-recipe footer. Run `cargo run --bin checkpoint_d`.

**Step 6.** M1 one-shot binary: `src/bin/vanta_m1.rs` — read `examples/buggy_python/cart.py` + `issue.md` → Creator → Attacker → `write_session_log`. Run `cargo run --bin vanta_m1`. Verify: patch_text present, tests have declared intents, log file written to `vanta_sessions/`.

**M1 EXIT TEST:** `cargo run --bin vanta_m1` against `examples/buggy_python/cart.py` produces a patch + intent-labelled tests in a JSON log. No test execution happens.

### Phase B — M2 Sandbox Layer

**Step 7.** `src/sandbox/mod.rs` — `SandboxBackend` trait + `ExecutionResult` + `SandboxKind`. Heavy `[VANTA-WHY]` on the subprocess-is-NOT-sandbox distinction (Spec §7).

**Step 8.** `src/sandbox/subprocess.rs` — `SubprocessBackend` impl: copy repo to tempfile, `tokio::process::Command` with process-group handling, `tokio::time::timeout` enforcement, kill on timeout, capture stdout/stderr/exit code. `[TRACE]` walk-through of timeout future vs actual process death.

**Step 9.** `src/sandbox/shuru.rs` — `ShuruBackend` impl: shell out to `shuru` CLI to spawn microVM, copy files via guest agent, execute, destroy.

**Step 10.** CHECKPOINT E: `src/bin/checkpoint_e.rs` — copy a sample dir to temp, run `pytest --json-report --json-report-file=report.json` with 30s timeout, parse `report.json`, print per-test `{name: pass/fail}`. 40-line header + 5-recipe footer. Run `cargo run --bin checkpoint_e`.

### Phase C — M2 BATE (Phase 1) + Diagnostician (Phase 2)

**Step 11.** `src/pipeline/bate.rs` — `BateOutcome` enum, `validate_test_against_baseline()`, intent→expected-baseline table (BugReproduction→FAIL on original, BehaviorPreservation→PASS on original, FeatureAcceptance→FAIL-or-unsupported). Retry max 3 → scrap + log reason. Reuses `TestIntent` from core.

**Step 12.** `src/pipeline/diagnostician.rs` — `DiagnosticResult`, classify failure into 4 `FailureClass` variants. Static analysis: subprocess run of `ruff check` + `mypy` on BOTH unmodified and patched trees, diff the diagnostics, only NEW ones count as Static. Timeout/signal/non-zero-exit-without-structured-output → Infrastructure. Structured assertion failure → Logic. Anything else → Unknown (escalate to Attacker simplify, quarantine if persistent).

**Step 13.** M2 full binary: `src/bin/vanta_m2.rs` — chain: Creator → Attacker → for each test: BATE baseline validate on unmodified tree (subprocess backend) → if valid, run on patched tree → Diagnostician triage → write full session log with `FailureClass` on each evidence record.

**Step 14.** M2 EXIT TEST: `cargo run --bin vanta_m2` against `examples/buggy_python/`. Every generated test baseline-validated, invalid scrapped, valid run against patched, failures tagged Infrastructure/Static/Logic/Unknown. Network-kill test: simulate by running with a command that tries network access → classified Infrastructure.

### Phase D — Integration + Cleanup

**Step 15.** Full `cargo check --all-targets` → 0 warnings. `cargo test --lib` → all green. Run checkpoint_a through e → all produce correct output. Run vanta_m1 and vanta_m2 end-to-end.

---

## Dependencies and Considerations

- **LLM helper promotion:** `post_chat`, `extract_first_valid_json`, `extract_json_candidate`, `structured_call_with_retry`, `build_reqwest_client` currently live in `bootstrap/mod.rs`. They MUST move to `src/llm/mod.rs` so Creator/Attacker agents in `agents/` can use them without circular deps. Bootstrap's M0 spike then `use crate::llm::*`. This is a clean move — no logic change, just relocation.
- **TestIntent reuse:** M2 BATE needs `TestIntent` (already in core). M1 GeneratedTest carries `intent: TestIntent`. No duplication.
- **FailureClass reuse:** M2 Diagnostician outputs `FailureClass` (already in core). No duplication.
- **EvidenceSource + TestEvidence reuse:** Already in core. M2 writes `TestEvidence` records into session log.
- **Subprocess backend caveat:** On macOS, true network isolation requires the microVM (Shuru). The subprocess backend documents network isolation as best-effort (env var hints + no proxy). Spec §7 is explicit: subprocess is NOT a sandbox. We follow the fast path (subprocess works today) + Shuru (real boundary).
- **Shuru availability:** The user confirmed Shuru is available. We shell out to the `shuru` CLI binary. If not in PATH, `ShuruBackend::spawn()` returns a clear Infrastructure error.
- **pytest-json-report:** The target Python env must have `pytest` and `pytest-json-report` installed. Checkpoint E handles missing tools gracefully (Infrastructure-class error, not panic).
- **UUID for IDs:** `uuid` crate v4 feature, no_std. Session IDs are deterministic-ish (timestamp + random) for reproducibility.
- **thiserror for M2 errors:** The Diagnostician and Sandbox layers need typed error variants. `thiserror` is the standard lightweight derive-macro crate. NOT used in M0 (Box<dyn Error> was fine for spike); M2 needs structured error matching for classification.

---

## Logging & Debugging Strategy (EXTENSIVE — user requirement)

Every module emits structured logs via the `tracing` crate. A global `tracing_subscriber` is initialized in each binary entry point. Log level is controlled by `VANTA_LOG_LEVEL` env var (default `info`; set to `debug` or `trace` for deep diagnostics).

### Log levels and what they capture
| Level | What it captures |
|-------|------------------|
| `error` | Unrecoverable failures (HTTP 5xx, sandbox spawn failure, parse exhaustion after all retries) |
| `warn` | Retries, test scraps after max attempts, Unknown-class failures, Shuru backend unavailable (fallback to subprocess) |
| `info` | Phase transitions (Creator start/done, Attacker start/done, BATE valid/scrap, Diagnostician classification), session ID, patch attempt ID, final summary |
| `debug` | Per-call inputs/outputs: prompt length, raw model response length, extracted JSON snippet, test name + intent + baseline result, sandbox command + exit code + wall time |
| `trace` | Full raw model output, full sandbox stdout/stderr, full JSON diagnostic payload, every BATE table lookup, every Diagnostician classification intermediate |

### Per-section logging requirements
| Section | Required log events |
|---------|---------------------|
| **agents/creator.rs** | `info!` "creator start" with session_id + issue length + source_files count; `debug!` system prompt length + user prompt length; `info!` "creator done" with patch_text length + files_touched + confidence; on retry `warn!` "creator retry attempt N" with last error; `trace!` full raw model output |
| **agents/attacker.rs** | `info!` "attacker start" with patch_text length; `debug!` per-test: name + intent + source length; `info!` "attacker done" with test count; `warn!` on retry; `trace!` full raw output |
| **session/mod.rs** | `info!` "session created" with all 4 IDs; `info!` "session log written" with path + record count; `debug!` per-record fields as they're appended |
| **sandbox/subprocess.rs** | `info!` "sandbox spawn" with backend kind + temp dir; `debug!` "execute" with cmd + cwd + timeout; `info!` "execute done" with exit_code + wall_time_ms + timed_out flag; `warn!` "timeout — killing process group"; `trace!` full stdout/stderr |
| **sandbox/shuru.rs** | `info!` "shuru vm spawn" with image; `warn!` "shuru unavailable, falling back to subprocess"; `debug!` guest exec command + exit code |
| **pipeline/bate.rs** | `info!` "BATE validate" with test_name + intent; `debug!` "baseline result" passed:bool; `info!` "BATE outcome" Valid/Mismatch with reason; `warn!` "scrapping test after N retries" with reason; `trace!` intent→expected-baseline table lookup |
| **pipeline/diagnostician.rs** | `info!` "diagnose start" with test_name; `debug!` static analysis diff (new diagnostics count); `info!` "diagnose done" with FailureClass + reason; `warn!` "Unknown → quarantine" or "Unknown → escalate to Attacker simplify"; `trace!` full structured JSON payload back to Creator |

### Debug dump files (post-hoc forensics)
In addition to structured logs, every phase writes raw artifacts to disk under `vanta_sessions/{session_id}/debug/`:
- `creator_raw_response.txt` — full content + reasoning_content concatenation
- `creator_extracted_json.json` — the cleaned JSON candidate that parsed
- `attacker_raw_response.txt` — same for attacker
- `attacker_extracted_json.json` — same
- `bate_baseline_{test_name}.txt` — raw baseline execution stdout/stderr
- `diagnostic_{test_name}.json` — full Diagnostician structured payload

These are NOT the session log (which is structured JSON). They are forensics for when something goes wrong and you need to see exactly what the model/sandbox produced.

### `VANTA_LOG_LEVEL` env var
Set `VANTA_LOG_LEVEL=trace` to see everything. Default `info` is clean enough for normal runs. The session log JSON always contains the full structured record regardless of log level (logs are for humans, session log is for M5 experiments).

---

## Validation

1. `cargo check --all-targets 2>&1 | grep -v "warning:" | tail -5` → only "Finished", no errors.
2. `cargo test --lib` → all tests pass (existing 13 + new session/agent tests).
3. `cargo run --bin checkpoint_d` → asserts BugReproduction+passing-baseline = mismatch.
4. `cargo run --bin checkpoint_e` → copies dir, runs pytest, parses JSON, prints pass/fail table.
5. `cargo run --bin vanta_m1` → patch + intent-labelled tests written to `vanta_sessions/*.json`. No execution.
6. `cargo run --bin vanta_m2` → full pipeline: BATE validates, invalid scrapped, valid run on patched, failures tagged with FailureClass.
7. Verify `vanta_sessions/` log files are valid JSON with `session_id`, `patch_attempt_id`, `test_id` on every record.
8. Spot-check a Logic-class failure: a test that runs and assertion-fails on patched code → `failure_class: "Logic"`.
9. Spot-check an Infrastructure-class failure: a test command that times out or segfaults → `failure_class: "Infrastructure"`.
10. **Logging verification:** Run with `VANTA_LOG_LEVEL=trace cargo run --bin vanta_m2` → confirm every phase emits `info!` (phase transitions), `debug!` (inputs/outputs), and `trace!` (raw dumps) level events. Confirm `vanta_sessions/{session_id}/debug/` contains `creator_raw_response.txt`, `attacker_raw_response.txt`, `bate_baseline_*.txt`, and `diagnostic_*.json` forensics files.

---

## Risks

| Risk | Handling |
|------|----------|
| Promoting LLM helpers breaks M0 spike | Validate `cargo run --bin vanta` (M0) still passes immediately after the move. If broken, the move has a bug — helpers must be byte-identical. |
| Shuru CLI not in PATH or behaves differently than expected | `ShuruBackend::spawn()` returns Infrastructure error with clear message. Subprocess backend is the fallback and always works. Shuru is additive, not blocking. |
| Small model (qwen3.5-4b) produces low-quality patches/tests for the sample Python bug | The M1/M2 exit tests accept ANY valid patch + ANY intent-labelled tests (quality is M3's concern). The 3-attempt retry loop + extract_first_valid_json ensures structured output. If quality is poor, swap to `VANTA_LLM_MODEL=ornith-1.0-35b` for better output — same code, env var flip. |
| pytest-json-report not installed in user's Python env | Checkpoint E and vanta_m2 detect `pytest: command not found` → Infrastructure error, not panic. Document the `pip install pytest pytest-json-report` requirement in the binary's error message. |
| Scope creep into M3 (repair loop, Gatekeeper, PR) | Hard boundary: M2 runs the shared driver with a one-attempt budget, so it stops at Diagnostician output + session log. At N=1 the Governor halts after the first attempt and no PR is opened, so M2 needs no Governor-free special case — it is the same loop, configured differently. What stays M3-only is the multi-attempt repair loop and the GitHub PR. |
| Memory-limit regression coverage | Deferred. The small pure-Python end-to-end target will not exercise sandbox memory pressure, so a dedicated driver-level regression for memory-limit handling comes after the clean end-to-end run rather than being rushed into this slice. |
| Network isolation on subprocess is impossible on macOS | Honest documentation (Spec §7). The subprocess backend is the fast path for trusted benchmark repos. Shuru is the real boundary. We never claim subprocess = sandbox. |
