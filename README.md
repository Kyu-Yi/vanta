# Vanta

**Autonomous adversarial verification for AI-generated code.**

> AI coding tools are getting better at producing code.
> The hard part is proving it doesn't break anything.

[![Status](https://img.shields.io/badge/status-in%20development-blue)]()
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-in%20preparation-purple)]()

---

## The problem

Roughly **55.8% of AI-generated code artifacts contain identifiable vulnerabilities**, and telling the model to "write secure code" improves this by about 4 percentage points. The failure is architectural, not instructional. You cannot prompt your way out of it.

Every mainstream coding agent optimises for one question:

> *Can I make the failing test pass?*

Vanta answers a stricter one:

> *Did this change fix the issue **without breaking anything else**, violating a historical constraint, or failing a mathematical proof of correctness?*

A patch that fixes the reported bug while silently regressing three unrelated routes is often worse than no patch at all. Vanta exists to close that gap.

## How it works

Two adversarial LLM agents, multiple independent non-LLM sources of truth. The patch under verification can come from Vanta's own Creator agent, an external coding agent, or a human. Vanta is a verification layer, not just a repair loop. A patch ships only when it passes every **required** verdict source; non-binary verification outcomes (`Inconclusive` / `Unsupported` / `Timeout`) block or pass according to a per-repository verification policy (**required** for critical paths, **best-effort** by default, **disabled** for unsupported languages).

```
                Target codebase + bug report
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │     RUST ORCHESTRATION PLANE (Rig)    │
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
        │   ┌────┴─────────────────┐             │
        │   ▼                      ▼             │
        │  VERIFIER BACKENDS   SANDBOX           │
        │  CrossHair/Kani/Z3   microVM exec +    │
        │  (verified-in-bounds 4-way failure     │
        │   or counterexample) triage            │
        │   └────┬─────────────────┘             │
        │        ▼                               │
        │   PHASE 3 · GOVERNOR                   │
        │   EIR/ECR stability check:             │
        │   stop when correction turns harmful   │
        │        │                               │
        │        ▼                               │
        │   GATEKEEPER ── all pass → open PR     │
        │              └─ any fail → retry       │
        └──────────────────────────────────────┘
                           │
                           ▼
              Institutional memory
              (JSON → Graphiti bi-temporal graph)
```

| Component | Role | Trusts the LLM? |
|---|---|---|
| **Creator** | Reads issue + source + history, writes a minimal patch | (is the LLM) |
| **Attacker** | Writes new adversarial regression tests targeting the patch's weak points | (is the LLM) |
| **Diagnostician** | Triages every failure: infrastructure / static / logic / unknown | No, deterministic |
| **Verifier backends** | Language-specific formal/symbolic systems behind one trait; return structured outcomes | No, mathematical |
| **Governor** | Computes EIR/ECR live; halts when correction turns net-harmful | No, arithmetic |
| **Gatekeeper** | Opens a PR only when every signal passes | No, deterministic |

## What makes Vanta different

**1 · Adversarial by construction.** A dedicated Attacker agent actively tries to break every patch, evolving new tests each round based on what the patch actually does, instead of passively checking a fixed suite.

**2 · Intent-labelled, baseline-validated tests (BATE).** Every Attacker-generated test declares its intent: `bug_reproduction` (should FAIL on the original code, since that failure proves the bug was reproduced), `feature_acceptance`, or `behavior_preservation` (should PASS on the original code). BATE runs each test against the *unmodified* codebase and checks the result **matches the declared intent**. Mismatches are returned for revision (max 3 retries) then scrapped. This filters malformed adversarial tests, pins every baseline, and excludes pre-existing failures from regression counting, so a patch is only ever penalised for damage *it* caused.

**3 · Four-way failure triage.** Not every red test means the model failed:

| Class | Catches | Counted as a model error? |
|---|---|---|
| Infrastructure | Sandbox/timeout/OOM/runner failures, caught by process-level signals (timeouts, exit codes, memory limits, health checks) | **No**, sandbox reset & rerun |
| Static | Pre-execution errors (syntax, types, imports), caught by per-language static analysis | Tracked separately |
| Logic | Genuine logic failures: test ran, assertion failed | **Yes**, the only class that counts |
| Unknown | Unclassifiable execution failures | **No**: rerun → capture logs → quarantine (generated tests only; repository/benchmark tests instead become *human review required*) |

Logic-class diagnostics travel back to the patch source as structured JSON payloads. A token-optimised [GCF](https://www.gcformat.com/) variant is a planned V2 ablation experiment.

**4 · Formal verification as a second, independent verdict.** One Rust `Verifier` trait over language-specific verification backends. These are full verification systems, not thin translators (several use Z3 internally, and a direct Z3 backend handles pure mathematical properties). Every backend returns a structured outcome: `VerifiedWithinBounds` / `Counterexample` / `Inconclusive` / `Unsupported` / `Timeout` / `ToolFailure`. On real code, *inconclusive* is a first-class result, and "verified within bounds" is an honest claim where "mathematically proven" would not be. A patch can pass *every test* and still be rejected on a concrete counter-example (*"if amount=150 and charge=100, this code allows a refund larger than the charge"*).

| Target language | Backend | Status |
|---|---|---|
| Python | CrossHair (uses Z3 internally) | ✅ production. Python-only by necessity: it hooks the Python runtime itself |
| Rust | Kani | ✅ production (AWS-maintained) |
| Pure properties | Z3 direct | optional backend |
| Java | OpenJML | future |
| C / C++ | CBMC | future |
| JS / TS | none | no viable backend; verification track skipped, adversarial suite only |

**5 · A path toward principled stopping.** V1 uses a fixed attempt cap while recording the full transition history needed to estimate EIR/ECR. V2 replaces the cap with the live stability governor (below), so the loop halts on a measured condition rather than `for i in 0..4`.

**6 · Language-agnostic.** Vanta itself is 100% Rust. The code it verifies can be anything: per-language test runners, linters, and translators are dispatched as subprocesses, the way a build script calls `git`.

## The research angle

Vanta's stopping rule implements **Liu & Meng, *"Self-Correction as Feedback Control"* ([arXiv:2604.22273](https://arxiv.org/abs/2604.22273))**, which models LLM self-correction as a two-state Markov chain and derives a measurable stability threshold:

```
 continue only while    ECR / EIR  >  Acc / (1 − Acc)

 EIR = P(correct → incorrect)   breaking a working thing
 ECR = P(incorrect → correct)   fixing a broken thing
```

The original paper validates this on reasoning benchmarks under *intrinsic* self-correction and explicitly names its limitations as future work. Vanta is built to address them:

- **Domain transfer.** Vanta investigates how EIR/ECR can be defined and measured on adversarially-verified *code repair*, where correctness is graded across an entire evolving test suite rather than one answer string.
- **Non-stationarity.** Because the Attacker generates harder tests each round, the error rates are non-stationary *by construction*. Vanta applies the paper's per-transition condition live rather than its steady-state results, and measures EIR/ECR transitions on a **fixed shared test set** between patch attempts (newly generated tests are recorded separately, then join the set), so adversarial escalation never contaminates the error definition.
- **Evidence provenance.** Every signal is logged with its source (repository test, benchmark oracle, Attacker-generated, verifier), so "verified within bounds against a generated property" is never conflated with "proven against a human-specified requirement".

The three-phase pipeline is what makes those measurements defensible: only genuine, patch-caused, logic-level failures, on baseline-validated tests, net of pre-existing failures, ever enter the EIR/ECR calculation.

**Planned evaluation:** BugsInPy (primary, Python) · Defects4J (cross-language check, Java) · SWE-bench-Lite with a novel regression-rate metric layered on top.
**Paper:** in preparation, targeting arXiv (cs.SE) → ICSE / FSE or co-located workshop, alongside a KAIST MSc thesis.

## Sandboxing

Untrusted AI-generated code executes only inside a hardware-isolated microVM, behind one Rust trait. The orchestrator never knows which platform it's on:

| Platform | Backend | Tech | Notes |
|---|---|---|---|
| macOS | `shuru.rs` | [Shuru](https://github.com/superhq-ai/shuru) / Apple Virtualization.framework | **primary dev backend, built first** |
| Linux | `zeroboot.rs` | Firecracker (KVM) | sub-ms CoW forking; production target. **Built in V2**, V1 ships macOS-only |
| Windows | `wsl.rs` | WSL2 (Hyper-V) | weakest isolation; built last |

## Institutional memory

The Attacker is far more dangerous when it remembers how this codebase has broken before.

- **V1:** flat JSON incident files. Simple lookup, proves the concept.
- **V2:** [Graphiti](https://github.com/getzep/graphiti) bi-temporal knowledge graph. Tracks *event time* and *ingestion time* independently, preserving the causal chains (`Incident → Caused_By → PR → Violates → Policy`) that a similarity-ranked store collapses.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | Rust + [Rig](https://github.com/0xPlaygrounds/rig) |
| Sandboxing | Shuru (macOS) · Firecracker (Linux) · WSL2 (Windows) |
| Test execution | `cargo-nextest` (Rust targets, structured JSON) · `pytest --json-report` (Python) · per-language equivalents |
| Failure triage | process-level signals (timeouts, exit codes, memory limits, health checks) · `ruff`/`mypy`/`clippy`/`tsc` · `parking_lot` inside Vanta's own orchestrator |
| Diagnostics wire format | structured JSON (V1) · [GCF](https://www.gcformat.com/) as V2 ablation |
| Verification | `Verifier` trait: CrossHair / Kani / Z3-direct (OpenJML, CBMC future) |
| Memory | JSON → [Graphiti](https://github.com/getzep/graphiti) |

## Roadmap

**Definition of done for V1:** *given a real Python repo and a bug description, Vanta produces a PR containing a patch plus Attacker-generated tests that pass baseline validation, with the whole loop running unattended.*

### Milestone 0 — Foundations
- [x] Architecture & technical specification
- [x] Research framing validated against Liu & Meng (2026)
- [x] Dev environment (Rust toolchain, cargo-nextest, gh)
- [ ] Rig spike: call an LLM API, parse structured output, chain two agent calls

### Milestone 1 — The adversarial core
- [ ] Orchestration skeleton (Rig)
- [ ] Creator agent: issue in, candidate patch out
- [ ] Attacker agent: patch in, intent-labelled adversarial tests out
- [ ] Structured Creator ⇄ Attacker loop with attempt logging

### Milestone 2 — Ground truth
- [ ] Sandbox trait (`mod.rs`) + `shuru.rs` macOS backend
- [ ] Phase 1: intent-aware BATE baseline validation with 3-retry test repair
- [ ] Phase 2: four-way Diagnostician (process-level signals, per-language static analysis, Unknown quarantine flow)
- [ ] Structured JSON diagnostic payloads + session/attempt/iteration/test IDs on every result

### Milestone 3 — V1 ship 🚢
- [ ] Fixed-N Governor + deterministic Gatekeeper
- [ ] GitHub PR integration
- [ ] CLI: `vanta run --repo ./target --issue ./bug.md`
- [ ] End-to-end unattended run on a real repository ← **V1 acceptance test**

### Milestone 4 — V2: proof & memory
- [ ] `zeroboot.rs`: Firecracker CoW forking (Linux, via CI)
- [ ] Verifier trait + backends: CrossHair (Python), Kani (Rust), Z3 direct
- [ ] EIR/ECR instrumentation over attempt logs (fixed-shared-set protocol)
- [ ] GCF vs JSON payload ablation
- [ ] Live Governor (stability condition) + Verify-First intervention
- [ ] JSON → Graphiti memory migration

### Milestone 5 — Research
- [ ] BugsInPy / Defects4J evaluation harness
- [ ] EIR/ECR experiments across model tiers
- [ ] Paper: results section from real data → arXiv (cs.SE)
- [ ] `wsl.rs` Windows backend

## Status

**Design complete · implementation in progress.** This repository currently contains the specification and research framing; code lands milestone by milestone per the roadmap above. Watch the repo or follow along. This is being built in public.

## FAQ

**Why Rust for an AI tool?**
Rust provides strong type safety, explicit resource ownership, and dependable process control, which is a natural fit for sandbox orchestration where one trait serves three platform backends. It also gives the project a deliberate systems-engineering focus. [Rig](https://github.com/0xPlaygrounds/rig) handles the LLM orchestration natively, so no Python is needed for the agents.

**Then why does CrossHair need Python?**
Because the *artifact being verified* is Python, and CrossHair symbolically executes it inside the Python interpreter itself. It's a system dependency called as a subprocess, not Python code inside Vanta. Verifying a Rust patch uses Kani and involves no Python at all.

**Isn't this just AI code review?**
Review tools comment on diffs. Vanta *executes* the patch in hardware isolation, *attacks* it with generated regression tests, *checks* supported properties through formal and symbolic verification backends, and *decides when to stop trying* on a published stability condition. Then it opens the PR itself.

**Can the Attacker write bad tests?**
Yes, it's an LLM. That's what Phase 1 exists for: every test declares its intent, and its result on the original codebase must *match that intent* (a bug-reproduction test is supposed to fail there, since that's proof the bug was reproduced) before it's allowed to judge anything, with a 3-retry-then-scrap rule and the discard rate logged as an Attacker-reliability metric. BATE is a filter, not semantic proof. Persistent failure patterns are flagged for provenance-aware review, because persistence alone may mean the test is wrong, the patches are wrong, or the bug is genuinely hard. Every generated test carries provenance so it's never mistaken for ground truth.

## Contributing

Vanta is in early single-maintainer development, but issues, ideas, and discussion are welcome, especially from anyone who has fought LLM coding agents in production. Once the V1 core lands, `good-first-issue` labels will follow.

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
Built by <a href="https://github.com/Kyu-Yi">Nathan Lawson</a> · KAIST MSc AI/CS · Seoul
</p>
