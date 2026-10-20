# Vanta

**Autonomous adversarial verification for AI-generated code.**

> AI coding tools are getting better at producing code.
> The hard part is proving it doesn't break anything.

[![Status](https://img.shields.io/badge/status-v0.2%20research%20prototype-blue)]()
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs)
[![Sandbox](https://img.shields.io/badge/sandbox-macOS%20(Shuru)-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-in%20preparation-purple)]()

---

## The problem

Every mainstream coding agent optimises for one question:

> *Can I make the failing test pass?*

Vanta answers a stricter one:

> *Did this change fix the issue **without breaking anything else**?*

A patch that fixes the reported bug while silently regressing three unrelated code paths is often worse than no patch at all, and an agent that can edit the tests it is graded by can make almost anything look green. Vanta exists to close that gap: it attacks every patch with new tests, grades it with tests the patch cannot tamper with, and only approves when every independent signal agrees.

## How it works

Two adversarial LLM agents, several independent non-LLM sources of truth. The patch under verification can come from Vanta's own Creator agent, an external coding agent, or a human, so Vanta is a verification layer first and a repair loop second.

```
                Target codebase + bug report
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │       RUST ORCHESTRATION PLANE        │
        │                                        │
        │   Creator Agent ──── writes patch      │
        │   (or patch arrives from an external   │
        │    agent / human developer)            │
        │        │                               │
        │        ▼                               │
        │   Attacker Agent ─── writes tests,     │
        │        │            each with a        │
        │        │            declared INTENT    │
        │        ▼                               │
        │   PHASE 1 · BATE                       │
        │   baseline result on UNMODIFIED code   │
        │   must MATCH each test's intent        │
        │        │                               │
        │        ▼                               │
        │   SANDBOXED RUN (microVM)              │
        │   original repo tests + Attacker tests │
        │   + static checks → 4-way triage       │
        │        │                               │
        │        ▼                               │
        │   PHASE 3 · GOVERNOR                   │
        │   attempt cap today; EIR/ECR           │
        │   stability rule planned               │
        │        │                               │
        │        ▼                               │
        │   GATEKEEPER ── all pass → open PR     │
        │              └─ any fail → retry       │
        └──────────────────────────────────────┘
                           │
              ┌ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ┐
                  planned for V2:
              │   formal verifiers,     │
                  institutional memory
              └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

| Component | Role | Trusts the LLM? | Status |
|---|---|---|---|
| **Creator** | Reads issue + source, writes a minimal patch, retries on structured feedback | (is the LLM) | ✅ shipped |
| **Attacker** | Writes adversarial regression tests tied to the issue, not the patch | (is the LLM) | ✅ shipped |
| **BATE** | Checks every generated test against the original code before it may judge anything | No, deterministic | ✅ shipped |
| **Diagnostician** | Triages every failure: infrastructure / static / logic / unknown | No, deterministic | ✅ shipped |
| **Governor** | Decides when to stop; logs every attempt-to-attempt transition | No, arithmetic | ✅ fixed cap · 🔜 live EIR/ECR rule |
| **Gatekeeper** | Approves, and opens a PR, only when every required signal passes | No, deterministic | ✅ shipped |
| **Verifier backends** | Formal/symbolic checks behind one trait | No, mathematical | 🔜 V2 |

## What makes Vanta different

**1 · Adversarial by construction.** A dedicated Attacker agent actively tries to break every patch, writing new tests each round based on the issue, instead of passively checking a fixed suite. Its tests are derived from the issue text rather than the patch, so a wrong patch can never steer the tests that judge it, and it is prompted to probe exact boundary values. In review mode every extra round is a fresh batch of attacks against the same patch, because one round can easily sample inputs the patch never touches.

**2 · Intent-labelled, baseline-validated tests (BATE).** Every Attacker test declares its intent: `bug_reproduction` (should FAIL on the original code, since that failure proves the bug was reproduced), `feature_acceptance`, or `behavior_preservation` (should PASS on the original code). BATE runs each test against the *unmodified* codebase and checks the result **matches the declared intent**. Mismatches are returned for revision (max 3 retries) then scrapped, and the discard rate is logged as an Attacker-reliability metric. This filters malformed adversarial tests, pins every baseline, and excludes pre-existing failures from regression counting, so a patch is only ever penalised for damage *it* caused.

**3 · A grader the patch can't tamper with.** The repository's own tests always run in their *original* form, restored onto the patched source: modified test files are reverted, deleted tests are restored, and any `conftest.py` or test config the patch added is removed. A test that used to pass and is now skipped, xfailed, deselected or missing counts as a regression, exactly like a failure. What the patch tried to change is still recorded as evidence, so tampering is visible rather than silently undone. Attacker tests run on the same graded snapshot, so every stage judges the same code.

**4 · Four-way failure triage.** Not every red test means the model failed:

| Class | Catches | Counted as a model error? |
|---|---|---|
| Infrastructure | Sandbox/timeout/OOM/runner failures, caught by process-level signals (timeouts, exit codes, memory limits, health checks) | **No**, sandbox reset & rerun |
| Static | Pre-execution errors (syntax, types, imports), caught by per-language static analysis | Tracked separately |
| Logic | Genuine logic failures: test ran, assertion failed | **Yes**, the only class that counts |
| Unknown | Unclassifiable execution failures | **No**: rerun → capture logs → quarantine (generated tests only; repository tests instead become *human review required*) |

A model endpoint that stalls or disappears ends the run as `InfraFailure` with its own exit code, never as a rejection. Logic-class diagnostics travel back to the Creator as structured JSON payloads; a token-optimised [GCF](https://www.gcformat.com/) variant is a planned ablation.

**5 · Provenance on every signal, and reproducible rejections.** Every result records where it came from (repository test, Attacker-generated, static check, and later verifier or benchmark oracle), so generated-test evidence is never mistaken for ground truth. A rejected session keeps its graded snapshot, and its rejecting signals can be re-run later under the exact sandbox backend that produced them, so "the Attacker caught it" is something you can reproduce, not just a log line.

**6 · Formal verification as a second, independent verdict (V2).** One Rust `Verifier` trait over language-specific backends, each returning a structured outcome: `VerifiedWithinBounds` / `Counterexample` / `Inconclusive` / `Unsupported` / `Timeout` / `ToolFailure`. The outcome type exists today; the backends do not yet. On real code *inconclusive* is a first-class result, and "verified within bounds" is an honest claim where "mathematically proven" would not be. One rule is fixed in advance: a property is only as independent as its source, so a property written by an LLM is logged as generated evidence, and only properties from a human-written specification count as ground truth.

| Target language | Backend | Status |
|---|---|---|
| Python | CrossHair (uses Z3 internally) | 🔜 V2 · Python-only by necessity: it hooks the Python runtime itself |
| Rust | Kani | 🔜 V2 |
| Pure properties | Z3 direct | 🔜 optional backend |
| Java | OpenJML | future |
| C / C++ | CBMC | future |
| JS / TS | none | no viable backend; adversarial suite only |

**7 · Language-agnostic.** Vanta itself is 100% Rust. The code it verifies can be anything: per-language test runners, linters, and verifiers are dispatched as subprocesses inside the sandbox, the way a build script calls `git`.

## The research angle

Vanta's planned stopping rule implements **Liu & Meng, *"Self-Correction as Feedback Control"* ([arXiv:2604.22273](https://arxiv.org/abs/2604.22273))**, which models LLM self-correction as a two-state Markov chain and derives a measurable stability threshold:

```
 continue only while    ECR / EIR  >  Acc / (1 − Acc)

 ECR = of the tests that FAILED at attempt N, the share that PASS at N+1   (fixing a broken thing)
 EIR = of the tests that PASSED at attempt N, the share that FAIL at N+1   (breaking a working thing)
```

Vanta already computes both rates from its attempt logs, per evidence source (native, static, Attacker-generated) and combined, with `n/a` reported when a denominator is zero. The Governor still stops on a fixed attempt cap; turning the condition into a live stopping rule is V2 work, and only worth doing if the evaluation shows it helps.

The original paper validates the threshold on reasoning benchmarks under *intrinsic* self-correction and names its limitations as future work. Vanta is built to test whether it survives the move:

- **Domain transfer.** Vanta's loop is driven by *external* test feedback on code repair, graded across an entire evolving test suite rather than one answer string. Whether the threshold still means anything there is the core open question, not an assumption.
- **Non-stationarity.** Because the Attacker generates harder tests each round, error rates are non-stationary *by construction*. Vanta measures transitions on a **fixed shared test set** between attempts: new tests can reject a patch immediately, but only join the set for the following comparison, so adversarial escalation never contaminates the error definition.
- **Clean inputs.** Only genuine, patch-caused, logic-level changes on baseline-validated tests enter the calculation. Review-mode rounds, censored transitions, defective tests and non-logic failures are excluded.

**Evaluation:** BugsInPy (primary, Python, harness in progress) · Defects4J (cross-language check, Java) · SWE-bench-Lite with a regression-rate metric layered on top.
**Paper:** in preparation, targeting arXiv (cs.SE) → ICSE / FSE or co-located workshop, alongside a KAIST MSc thesis.

## Evaluation design

The BugsInPy harness (`scripts/bugsinpy_run.py`, full design in [`plans/bugsinpy.md`](plans/bugsinpy.md)) turns real historical bugs into a fair test of what the Attacker adds:

- **A hidden oracle.** Vanta sees the buggy source plus the fixed version's tests, *minus* the tests the real fix makes pass. Those hidden fail-to-pass tests are the oracle, and a patch is labelled good only if they pass and nothing else in the suite regresses.
- **A paired baseline.** Every bug runs under Vanta and under a baseline with the Attacker switched off, first in repair mode, then in review mode where both configurations judge the **identical** candidates: the developer's real fix, a no-op, and up to two failed patches pooled from the repair runs.
- **Faithful environments.** Each bug's dependencies are pinned to versions released before its commit, every suite runs inside Shuru, the project is never installed into site-packages (so tests always import the patched tree), and every run records full provenance.
- **Sanity gates.** Waves stop if the developer's fix grades as bad, the no-op grades as good, or any review row is missing its pair.

## Sandboxing

Untrusted code (the target repository and every LLM-written test) executes only inside a sandbox, behind one Rust trait. The orchestrator never knows which platform it's on, and the backend that actually ran is recorded in every session, so a run can't quietly downgrade from a microVM to a plain process.

| Platform | Backend | Tech | Status |
|---|---|---|---|
| macOS | `shuru.rs` | [Shuru](https://github.com/superhq-ai/shuru) / Apple Virtualization.framework | ✅ primary backend |
| any | `subprocess.rs` | process limits only, **no isolation** | ✅ development on trusted code only |
| Linux | `zeroboot.rs` | Firecracker (KVM), sub-ms CoW forking | 🔜 V2 production target |
| Windows | `wsl.rs` | WSL2 (Hyper-V) | 🔜 weakest isolation; built last |

## Institutional memory (V2)

The Attacker is far more dangerous when it remembers how this codebase has broken before.

- **Step 1:** flat JSON incident files. Simple lookup, proves the concept.
- **Step 2:** [Graphiti](https://github.com/getzep/graphiti) bi-temporal knowledge graph. Tracks *event time* and *ingestion time* independently, preserving the causal chains (`Incident → Caused_By → PR → Violates → Policy`) that a similarity-ranked store collapses.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | Rust (tokio) |
| LLM client | OpenAI-compatible client on `reqwest`, streaming with idle timeouts and cancellation · local `llama-server` or any hosted OpenAI-compatible API |
| Sandboxing | Shuru (macOS) · Firecracker (Linux, V2) · WSL2 (Windows, V2) |
| Test execution | `pytest --json-report` (Python) · `cargo-nextest` (Rust targets) · per-language equivalents |
| Failure triage | process-level signals (timeouts, exit codes, memory limits, health checks) · `ruff`/`mypy`/`clippy`/`tsc` |
| Diagnostics wire format | structured JSON · [GCF](https://www.gcformat.com/) as V2 ablation |
| PRs | `octocrab` (GitHub API) |
| Verification | `Verifier` trait: CrossHair / Kani / Z3-direct (V2) |
| Memory | JSON → [Graphiti](https://github.com/getzep/graphiti) (V2) |

## Quickstart

**You need:** Rust (stable), macOS with [Shuru](https://github.com/superhq-ai/shuru) for sandboxed runs, and an OpenAI-compatible model endpoint. Vanta is developed against a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` running a 27B-class Qwen model with a 131k context; smaller models struggle with structured output. Run the server with **one slot** (`parallel = 1`), since parallel slots share one KV cache and long agent prompts will overflow it.

```bash
git clone https://github.com/Kyu-Yi/vanta.git
cd vanta
cargo build --release

export VANTA_LLM_BASE_URL=http://localhost:8080/v1
export VANTA_LLM_MODEL=<your-model>
export VANTA_LLM_CONTEXT_TOKENS=131072

# review an existing patch (rounds = attack depth)
./target/release/vanta_m3 --repo fixtures/toy_calc --issue fixtures/toy_calc/issue.md --patch fixtures/toy_calc/bad.patch -n 3

# let the Creator repair it (attempts = patch budget, default 4)
./target/release/vanta_m3 --repo fixtures/toy_calc --issue fixtures/toy_calc/issue.md
```

Each run writes `vanta_sessions/<session-id>/`: a JSONL event log ending in a `run.summary` (decision, backend that ran, prompt versions, sampling, build commit), the attempt-to-attempt transition rows ECR/EIR are computed from, the shared test set, and per-attempt tree snapshots. Exit codes: `0` approved · `1` rejected / max attempts / patch didn't apply / harness failure · `2` infrastructure failure.

<details>
<summary><b>CLI options</b></summary>

| Option | Meaning |
|---|---|
| `--repo, -r PATH` | Repository to verify or repair (required) |
| `--issue, -i PATH` | Issue or bug description |
| `--patch, -p PATH` | Review an existing patch instead of generating one |
| `--max-attempts, -n N` | Attack depth in review mode, patch budget in repair mode (default 4) |
| `--test-cmd CMD` | Test command (default `python3 -m pytest`) |
| `--original-tree PATH` | Clean checkout of the base commit, for two-tree runs |
| `--model, -m NAME` | Override `VANTA_LLM_MODEL` |
| `--open-pr` | Open a draft PR after Gatekeeper approval |
| `--rerun-signals` | Re-run a recorded session's rejecting signals under its original backend (see `--help`) |

</details>

<details>
<summary><b>Environment variables</b></summary>

| Variable | Default | Meaning |
|---|---|---|
| `VANTA_LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible endpoint |
| `VANTA_LLM_MODEL` | built-in | Model identifier |
| `VANTA_API_KEY` | unset | Bearer token, if the endpoint needs one |
| `VANTA_LLM_CONTEXT_TOKENS` | `65536` | Context used to budget prompts; set it to your server's real context |
| `VANTA_LLM_IDLE_TIMEOUT_SECS` | `180` | Abandon a stream that sends nothing for this long |
| `VANTA_LLM_MAX_SECS` | `3600` | Hard limit on one LLM call |
| `VANTA_LLM_RETRY_ATTEMPTS` | built-in | Retries for a failed call |
| `VANTA_LLM_SEED` | unset | Sampling seed; unset sends no seed, so requests stay byte-identical |
| `VANTA_ATTACKER` | `on` | `off` gives the no-Attacker baseline used in evaluation |
| `VANTA_PLANNER` | `off` | Reserved for the V2 planner; `on` currently ends the run as `InfraFailure` |
| `VANTA_SANDBOX_BACKEND` | `auto` | `auto`, `shuru` or `subprocess` |
| `VANTA_SHURU_CHECKPOINT` | `vanta-python` | Checkpoint test VMs boot from |
| `VANTA_SHURU_DISK_MB` | built-in | VM disk size |
| `VANTA_SHURU_AUTO_DEPS` | on | Build a dependency checkpoint for the target repo automatically |
| `VANTA_TEST_TIMEOUT_SECS` | built-in | Timeout for one test run |
| `VANTA_GITHUB_TOKEN` | unset | Token for `--open-pr` |
| `VANTA_LOG_LEVEL` | `info` | Log verbosity |

</details>

**Pull requests.** With `--open-pr`, Vanta opens a PR only after Gatekeeper approval, as a **draft** by default. It authenticates only with `VANTA_GITHUB_TOKEN` (use a fine-grained token scoped to the one target repo), never falls back to `gh` or `git` credentials, and refuses to run from a dirty target repo. The PR body carries the session ID, tested commit and an evidence summary.

## Roadmap

**Definition of done for V1:** *given a real Python repo and a bug description, Vanta produces a PR containing a patch plus Attacker-generated tests that pass baseline validation, with the whole loop running unattended.* ✅ Met.

### Milestone 0 — Foundations
- [x] Architecture & technical specification
- [x] Research framing validated against Liu & Meng (2026)
- [x] Dev environment (Rust toolchain, cargo-nextest, gh)
- [x] LLM client spike: Rig trialled, plain `reqwest` chosen (see FAQ)

### Milestone 1 — The adversarial core
- [x] Orchestration skeleton
- [x] Creator agent: issue in, candidate patch out
- [x] Attacker agent: intent-labelled adversarial tests out, tied to the issue
- [x] Structured Creator ⇄ Attacker loop with attempt logging

### Milestone 2 — Ground truth
- [x] Sandbox trait (`mod.rs`) + `shuru.rs` macOS backend
- [x] Phase 1: intent-aware BATE baseline validation with 3-retry test repair
- [x] Phase 2: four-way Diagnostician (process-level signals, per-language static analysis, Unknown quarantine flow)
- [x] Structured JSON diagnostic payloads + session/attempt/iteration/test IDs on every result

### Milestone 3 — V1 ship 🚢
- [x] Fixed-N Governor + deterministic Gatekeeper
- [x] GitHub PR integration
- [x] CLI: review (`--patch`) and repair modes
- [x] End-to-end unattended run on a real repository ← **V1 acceptance test**
  - Exit test evidence: [vanta-pr-test#6](https://github.com/Kyu-Yi/vanta-pr-test/pull/6)
  - Session: `7e6d812a-6d8d-49ea-8445-c4fc916e94f5`

### Milestone 3.5 — v0.2 hardening
- [x] Tamper-resistant native suite: original tests restored, patch-added test config removed, silenced tests counted as regressions
- [x] Attacker tests graded on the same tree as the native suite
- [x] Conditional EIR/ECR per evidence source over attempt logs (fixed-shared-set protocol)
- [x] Re-running rejecting signals under the recorded sandbox backend
- [x] Streaming LLM calls with idle timeout, cancellation and `InfraFailure` as its own outcome
- [x] Session metrics against ground-truth labels (`scripts/metrics.py`)
- [x] Real-world development case study (6 review/repair cases), tagged `v0.2`

### Milestone 4 — V2: proof, memory & scale
- [ ] Planner + shared board coordinating several Attackers on a VM pool without overlap
- [ ] Live Governor (stability condition) + Verify-First intervention
- [ ] Verifier trait + backends: CrossHair (Python), Kani (Rust), Z3 direct
- [ ] `zeroboot.rs`: Firecracker CoW forking (Linux, via CI)
- [ ] GCF vs JSON payload ablation
- [ ] JSON → Graphiti memory migration

### Milestone 5 — Research
- [ ] BugsInPy evaluation harness ← **in progress**
- [ ] Defects4J / SWE-bench-Lite evaluations
- [ ] EIR/ECR experiments across model tiers
- [ ] Paper: results section from real data → arXiv (cs.SE)
- [ ] `wsl.rs` Windows backend

## Status

**v0.2 tagged; BugsInPy evaluation in progress.** Vanta runs unattended on real repositories in both review and repair mode, verifies candidates in a microVM against a grader they can't edit, and opens a draft PR when the Gatekeeper approves. V1 acceptance evidence: [vanta-pr-test#6](https://github.com/Kyu-Yi/vanta-pr-test/pull/6) (Shuru backend, repair mode, native suite of `455` tests with `0` regressions and `2` pre-existing failures fixed).

v0.2 was hardened against a six-case review/repair study on a real-world open-source project: **4/6 correct verdicts, 0 false rejections, 2 false approvals**. Both misses were the same gap, an issue requirement no test exercised: one needed a check that a function is still *called exactly as before*, the other needed real GPU hardware to expose a device-placement bug. That study was used to find and fix weaknesses in Vanta (including a patch weakening its own native test), so it is a development set, and its numbers are not a benchmark result.

## Known limitations

- **Thin evidence so far.** One acceptance run, one development study and toy fixtures; nothing yet shows Vanta beats a simple baseline. That's what the BugsInPy harness is for.
- **Untested requirements can still pass.** Vanta can only reject what some test observes. If no native or generated test exercises a clause of the issue, a patch that ignores it can be approved, which is exactly how both development-study misses happened.
- **The stopping theory may not transfer** from intrinsic self-correction to externally-graded code repair. That's the research question, not a settled design.
- The Attacker still sometimes misses exact boundary values. On the toy boundary fixture the boundary rule lifted the bad-patch catch rate to `5/6`, but it remains prompt-dependent; toy-fixture numbers are development results, not evidence.
- Repair mode stops on the first passing attempt, so a clean first-attempt fix produces no N→N+1 transition and sits outside EIR/ECR by construction.
- Local 27B models are slow (tens of minutes per repair run) and need a large context and a single-slot server.
- Sandboxed runs are macOS-only until the Firecracker backend lands.

## FAQ

**Why Rust for an AI tool?**
The hard part of Vanta isn't calling a model, it's orchestrating untrusted processes reliably: booting sandboxes, enforcing timeouts, cleaning up after crashes, and never losing evidence. Rust's type safety, explicit ownership and dependable process control fit that well, and one trait serves every sandbox backend.

**Why plain `reqwest` instead of an agent framework like Rig?**
Rig was trialled in the M0 spike and dropped. Vanta's agents are prompt-in, structured-JSON-out, so a framework adds little, while owning the client keeps every request body visible and controlled: streaming with idle timeouts and cancellation, exact sampling parameters recorded per call, and byte-identical requests when no seed is set. That matters for reproducibility and for fair comparisons across models. Any OpenAI-compatible endpoint works, local or hosted. A framework may earn its place later if the agents start calling tools.

**Then why does CrossHair need Python?**
Because the *artifact being verified* is Python, and CrossHair symbolically executes it inside the Python interpreter itself. It's a system dependency called as a subprocess, not Python code inside Vanta.

**Isn't this just AI code review?**
Review tools comment on diffs. Vanta *executes* the patch in hardware isolation, *attacks* it with generated regression tests, grades it with tests it can't tamper with, and records exactly why it approved or rejected. Then it opens the PR itself.

**Can the Attacker write bad tests?**
Yes, it's an LLM. That's what Phase 1 exists for: every test declares its intent, and its result on the original codebase must *match that intent* before it's allowed to judge anything, with a 3-retry-then-scrap rule and the discard rate logged as an Attacker-reliability metric. BATE is a filter, not semantic proof. Every generated test carries provenance so it's never mistaken for ground truth.

**Can a patch cheat by editing the tests?**
It can try, and it will be recorded, but it won't help. The native suite always runs the original tests, and silencing a test counts as breaking it. This rule exists because a real patch did exactly that during development.

## Repository layout

```
src/agents/      Creator and Attacker
src/pipeline/    driver, BATE, harness, patch application, Diagnostician, Governor, Gatekeeper
src/sandbox/     Sandbox trait + Shuru and subprocess backends
src/llm/         OpenAI-compatible streaming client
src/core/        shared types: results, intents, evidence sources, decisions
src/pr/          GitHub PR integration
src/bin/         vanta_m3 (the CLI)
scripts/         metrics + BugsInPy harness and its tests
fixtures/        small repos with planted bugs and good/bad patches
plans/           design docs, including the BugsInPy evaluation plan
```

## Contributing

Vanta is in early single-maintainer development, but issues, ideas, and discussion are welcome, especially from anyone who has fought LLM coding agents in production.

## Citation

If you build on the stability-governor design, please cite the underlying framework:

```bibtex
@article{liu2026selfcorrection,
  title   = {Self-Correction as Feedback Control: Error Dynamics,
             Stability Thresholds, and Prompt Interventions in LLMs},
  author  = {Liu, Aofan and Meng, Jingxiang},
  journal = {arXiv preprint arXiv:2604.22273},
  year    = {2026}
}
```

A citation entry for the Vanta paper will be added on arXiv publication.

## License

[MIT](LICENSE)

---

<p align="center">
Built by <a href="https://github.com/Kyu-Yi">Nathan Lawson</a> · KAIST MSc AI/CS · Daejeon
</p>
