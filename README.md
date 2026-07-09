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

Roughly **55.8% of AI-generated code artifacts contain identifiable vulnerabilities**, and telling the model to "write secure code" improves this by only ~4 percentage points. The failure is architectural, not instructional — you cannot prompt your way out of it.

Every mainstream coding agent optimises for one question:

> *Can I make the failing test pass?*

Vanta answers a stricter one:

> *Did this change fix the issue **without breaking anything else**, violating a historical constraint, or failing a mathematical proof of correctness?*

A patch that fixes the reported bug while silently regressing three unrelated routes is, in production, often worse than no patch at all. Vanta exists to close that gap.

## How it works

Two adversarial LLM agents, three independent non-LLM sources of truth. A patch ships only when it survives all of them.

```
                Target codebase + bug report
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │     RUST ORCHESTRATION PLANE (Rig)    │
        │                                        │
        │   Creator Agent ──── writes patch      │
        │        │                               │
        │        ▼                               │
        │   Attacker Agent ─── writes tests      │
        │        │            that try to        │
        │        │            break the patch    │
        │        ▼                               │
        │   PHASE 1 · BATE                       │
        │   every new test runs against the      │
        │   UNMODIFIED codebase first            │
        │        │                               │
        │   ┌────┴─────────────────┐             │
        │   ▼                      ▼             │
        │  FORMAL PROOF        SANDBOX           │
        │  translator → Z3     microVM exec +    │
        │  (proof or           3-tier failure    │
        │   counter-example)   triage            │
        │   └────┬─────────────────┘             │
        │        ▼                               │
        │   PHASE 3 · GOVERNOR                   │
        │   EIR/ECR stability check —            │
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
| **Creator** | Reads issue + source + history, writes a minimal patch | — (is the LLM) |
| **Attacker** | Writes new adversarial regression tests targeting the patch's weak points | — (is the LLM) |
| **Diagnostician** | Triages every failure: infra noise / static error / real logic failure | No — deterministic |
| **Formal proof track** | Translates the patch to logic; Z3 proves the invariant or returns a counter-example | No — mathematical |
| **Governor** | Computes EIR/ECR live; halts when correction turns net-harmful | No — arithmetic |
| **Gatekeeper** | Opens a PR only when every signal passes | No — deterministic |

## What makes Vanta different

**1 · Adversarial by construction.** A dedicated Attacker agent actively tries to break every patch — evolving new tests each round based on what the patch actually does — instead of passively checking a fixed suite.

**2 · Baseline-validated tests (BATE).** Every Attacker-generated test runs against the *unmodified* codebase first. Tests that fail there are returned for revision (max 3 retries) then scrapped. This filters malformed adversarial tests **and** excludes pre-existing failures from regression counting — a patch is only ever penalised for damage *it* caused.

**3 · Three-tier failure triage.** Not every red test means the model failed:

| Tier | Catches | Counted as a model error? |
|---|---|---|
| 1 | Infra noise — timeouts, OOM, deadlocks (trapped via `parking_lot` deadlock detection) | **No** — sandbox reset & retried |
| 2 | Pre-execution errors — syntax/type/imports, caught by per-language static analysis | Tracked separately |
| 3 | Genuine logic failures — test ran, assertion failed | **Yes** — the only class that counts |

Tier-3 diagnostics travel back to the Creator as token-optimised [GCF](https://www.gcformat.com/) payloads rather than JSON.

**4 · Formal proof as a second, independent verdict.** One solver — **Z3** — behind per-language translator front-ends. A patch can pass *every test* and still be rejected on a concrete counter-example (*"if amount=150 and charge=100, this code allows a refund larger than the charge"*).

| Target language | Translator | Status |
|---|---|---|
| Python | CrossHair | ✅ production (Python-only by necessity — it hooks the Python runtime itself) |
| Rust | Kani | ✅ production (AWS-maintained) |
| Java | OpenJML | mature |
| C / C++ | CBMC | very mature |
| JS / TS | — | no viable translator; formal track skipped, adversarial suite only |

**5 · A principled stopping rule, not an iteration cap.** The repair loop halts on a published stability condition (below), not `for i in 0..4`.

**6 · Language-agnostic.** Vanta itself is 100% Rust. The code it verifies can be anything — per-language test runners, linters, and translators are dispatched as subprocesses, the way a build script calls `git`.

## The research angle

Vanta's stopping rule implements **Liu & Meng, *"Self-Correction as Feedback Control"* ([arXiv:2604.22273](https://arxiv.org/abs/2604.22273))**, which models LLM self-correction as a two-state Markov chain and derives a measurable stability threshold:

```
 continue only while    ECR / EIR  >  Acc / (1 − Acc)

 EIR = P(correct → incorrect)   breaking a working thing
 ECR = P(incorrect → correct)   fixing a broken thing
```

The original paper validates this on reasoning benchmarks under *intrinsic* self-correction and explicitly names two limitations as future work. Vanta is built to address both:

- **Domain transfer** — first system to measure EIR/ECR on adversarially-verified *code repair*, where correctness is graded across an entire evolving test suite rather than one answer string.
- **Non-stationarity** — because the Attacker generates harder tests each round, the error rates are non-stationary *by construction*; Vanta applies the paper's per-iteration equilibrium condition live rather than its steady-state results.

The three-phase pipeline is what makes those measurements defensible: only genuine, patch-caused, logic-level failures — on baseline-validated tests, net of pre-existing failures — ever enter the EIR/ECR calculation.

**Planned evaluation:** BugsInPy (primary, Python) · Defects4J (cross-language check, Java) · SWE-bench-Lite with a novel regression-rate metric layered on top.
**Paper:** in preparation, targeting arXiv (cs.SE) → ICSE / FSE or co-located workshop, alongside a KAIST MSc thesis.

## Sandboxing

Untrusted AI-generated code executes only inside a hardware-isolated microVM, behind one Rust trait — the orchestrator never knows which platform it's on:

| Platform | Backend | Tech | Notes |
|---|---|---|---|
| macOS | `shuru.rs` | [Shuru](https://github.com/superhq-ai/shuru) / Apple Virtualization.framework | **primary dev backend — built first** |
| Linux | `zeroboot.rs` | Firecracker (KVM) | sub-ms CoW forking; production target |
| Windows | `wsl.rs` | WSL2 (Hyper-V) | weakest isolation; built last |

## Institutional memory

The Attacker is far more dangerous when it remembers how this codebase has broken before.

- **V1:** flat JSON incident files — simple lookup, proves the concept.
- **V2:** [Graphiti](https://github.com/getzep/graphiti) bi-temporal knowledge graph — tracks *event time* and *ingestion time* independently, preserving the causal chains (`Incident → Caused_By → PR → Violates → Policy`) that a similarity-ranked store collapses.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | Rust + [Rig](https://github.com/0xPlaygrounds/rig) |
| Sandboxing | Shuru (macOS) · Firecracker (Linux) · WSL2 (Windows) |
| Test execution | `cargo-nextest` (Rust targets, structured JSON) · `pytest --json-report` (Python) · per-language equivalents |
| Failure triage | kernel exit codes · `parking_lot` deadlock detection · `ruff`/`mypy`/`clippy`/`tsc` |
| Diagnostics wire format | [GCF](https://www.gcformat.com/) (Rust crate) |
| Formal verification | Z3 (via `z3` crate) + CrossHair / Kani / OpenJML / CBMC |
| Memory | JSON → [Graphiti](https://github.com/getzep/graphiti) |

## Roadmap

**Definition of done for V1:** *given a real Python repo and a bug description, Vanta produces a PR containing a patch plus Attacker-generated tests that pass baseline validation — with the whole loop running unattended.*

### Milestone 0 — Foundations
- [x] Architecture & technical specification
- [x] Research framing validated against Liu & Meng (2026)
- [x] Dev environment (Rust toolchain, cargo-nextest, gh)
- [ ] Rig spike: call an LLM API, parse structured output, chain two agent calls

### Milestone 1 — The adversarial core
- [ ] Orchestration skeleton (Rig)
- [ ] Creator agent — issue in, candidate patch out
- [ ] Attacker agent — patch in, adversarial tests out
- [ ] Structured Creator ⇄ Attacker loop with attempt logging

### Milestone 2 — Ground truth
- [ ] Sandbox trait (`mod.rs`) + `shuru.rs` macOS backend
- [ ] Phase 1 — BATE baseline validation with 3-retry test repair
- [ ] Phase 2 — three-tier Diagnostician (`parking_lot`, per-language static analysis)
- [ ] GCF Tier-3 diagnostic payloads

### Milestone 3 — V1 ship 🚢
- [ ] Fixed-N Governor + deterministic Gatekeeper
- [ ] GitHub PR integration
- [ ] CLI: `vanta run --repo ./target --issue ./bug.md`
- [ ] End-to-end unattended run on a real repository ← **V1 acceptance test**

### Milestone 4 — V2: proof & memory
- [ ] `zeroboot.rs` — Firecracker CoW forking (Linux, via CI)
- [ ] Formal-proof track: Z3 + CrossHair (Python), Kani (Rust)
- [ ] EIR/ECR instrumentation over attempt logs
- [ ] Live Governor (stability condition) + Verify-First intervention
- [ ] JSON → Graphiti memory migration

### Milestone 5 — Research
- [ ] BugsInPy / Defects4J evaluation harness
- [ ] EIR/ECR experiments across model tiers
- [ ] Paper: results section from real data → arXiv (cs.SE)
- [ ] `wsl.rs` Windows backend

## Status

**Design complete · implementation in progress.** This repository currently contains the specification and research framing; code lands milestone by milestone per the roadmap above. Watch the repo or follow along — this is being built in public.

## FAQ

**Why Rust for an AI tool?**
The sandbox layer needs direct, GC-free control over microVM APIs, and one trait serving three platform backends is what systems languages are for. [Rig](https://github.com/0xPlaygrounds/rig) handles the LLM orchestration natively — no Python needed for the agents.

**Then why does CrossHair need Python?**
Because the *artifact being verified* is Python, and CrossHair symbolically executes it inside the Python interpreter itself. It's a system dependency called as a subprocess — not Python code inside Vanta. Verifying a Rust patch uses Kani and involves no Python at all.

**Isn't this just AI code review?**
Review tools comment on diffs. Vanta *executes* the patch in hardware isolation, *attacks* it with generated regression tests, *proves* invariants about it mathematically, and *decides when to stop trying* on a published stability condition — then opens the PR itself.

**Can the Attacker write bad tests?**
Yes — it's an LLM. That's exactly what Phase 1 exists for: every test must pass against the known-good original codebase before it's allowed to judge anything, with a 3-retry-then-scrap rule and the discard rate logged as an Attacker-reliability metric.

## Contributing

Vanta is in early single-maintainer development, but issues, ideas, and discussion are welcome — especially from anyone who has fought LLM coding agents in production. Once the V1 core lands, `good-first-issue`s will follow.

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
