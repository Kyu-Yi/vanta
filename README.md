# Vanta

**Adversarial verification for AI-generated code patches.**

[![Status](https://img.shields.io/badge/status-research%20prototype-blue)]()
[![Version](https://img.shields.io/badge/version-v0.2-informational)]()
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs)
[![Sandbox](https://img.shields.io/badge/sandbox-macOS%20(Shuru)-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Most coding agents are built to answer one question: *can I make the failing test pass?* Vanta answers a stricter one: **did this change fix the issue without breaking anything else?** It does that by actively trying to break a patch with generated tests, running the repository's own tests in a form the patch cannot tamper with, and only approving when every piece of evidence agrees. The patch can come from Vanta's own Creator agent, from another coding agent, or from a person, so Vanta works both as a repair loop and as an independent reviewer.

Vanta is a research prototype written in Rust, developed for a KAIST MSc thesis and an accompanying paper. It runs end to end on real Python repositories today, but it has not yet been evaluated at scale, and this README tries to be precise about which parts are built, which are experimental, and which are still plans.

---

## Contents

- [Status at a glance](#status-at-a-glance)
- [How it works](#how-it-works)
- [Two modes: review and repair](#two-modes-review-and-repair)
- [Why the verdict can be trusted](#why-the-verdict-can-be-trusted)
- [Measuring repair: ECR and EIR](#measuring-repair-ecr-and-eir)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Opening pull requests](#opening-pull-requests)
- [Security model](#security-model)
- [Evaluation](#evaluation)
- [Limitations and open questions](#limitations-and-open-questions)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Citation](#citation)

---

## Status at a glance

| | |
|---|---|
| **Current version** | v0.2 |
| **Works today** | Review and repair modes on Python repositories, sandboxed execution on macOS, intent-checked adversarial tests, tamper-resistant native test gate, failure triage, ECR/EIR reporting, draft PRs |
| **In progress** | A BugsInPy benchmark harness that compares Vanta against a no-Attacker baseline on real historical bugs |
| **Not built yet** | Formal verifier backends, a live ECR/EIR stopping rule, a planner for multiple Attackers, institutional memory, Linux and Windows sandboxes |
| **Evidence** | One end-to-end acceptance run on a real repository and one development case study. There are no benchmark results yet, so there is no evidence either way on whether Vanta beats a simpler baseline. |

Status labels used throughout: **Shipped** means it is in v0.2 and exercised by tests or real runs, **Experimental** means the code exists but has not been validated, and **Planned** means it has not been built.

---

## How it works

A run takes a repository and an issue description, and either produces a patch (repair mode) or receives one (review mode). That patch then goes through a fixed sequence of stages, and only a patch that survives all of them is approved.

```
   repository + issue text
            │
            ▼
   ┌──────────────────┐   repair mode: writes the patch
   │     Creator      │   review mode: skipped, the patch is given
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   writes new tests aimed at the patch,
   │     Attacker     │   each with a declared intent
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   runs each new test on the ORIGINAL code and
   │       BATE       │   keeps it only if the result matches its intent
   └────────┬─────────┘
            ▼
   ┌──────────────────────────────────────────────┐
   │  Patched run inside the sandbox               │
   │   · the repository's original tests (native)  │
   │   · the surviving Attacker tests              │
   │   · static checks                             │
   └────────┬─────────────────────────────────────┘
            ▼
   ┌──────────────────┐   sorts every failure into infrastructure,
   │  Diagnostician   │   static, logic or unknown
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   decides whether to stop or try again,
   │     Governor     │   and records every attempt-to-attempt transition
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   approves only when every required signal passes;
   │    Gatekeeper    │   optionally opens a draft PR
   └──────────────────┘
```

**Creator** (Shipped). An LLM agent that reads the issue and the relevant source and writes a minimal patch. In repair mode it receives structured feedback after a failed attempt and tries again, up to the attempt limit. Its prompt explicitly forbids weakening or deleting existing tests, and the native test gate described below enforces that rule rather than trusting it.

**Attacker** (Shipped). A second LLM agent whose only job is to break the patch. It writes new regression tests, and each test declares an *intent*: `bug_reproduction` (it should fail on the original code, because that failure is what proves the bug was reproduced), `feature_acceptance`, or `behavior_preservation` (it should pass on the original code). The Attacker's tests are tied to the issue text rather than to the patch, so a wrong patch cannot steer the tests that judge it, and the Attacker is prompted to probe exact boundary values.

**BATE, baseline-validated tests** (Shipped). Every Attacker test is run against the *unmodified* code first, and its result must match the intent it declared. A test that does not match is sent back for revision up to three times and then discarded, and the discard rate is logged as a measure of how reliable the Attacker is. This filters out malformed tests and pins a baseline for every test, so a patch is only ever penalised for failures it caused. BATE is a filter, not a proof that a test is semantically right.

**Native suite and static checks** (Shipped). The repository's own tests run against the patched source, together with the surviving Attacker tests and per-language static analysis. How the native tests are protected from the patch is covered in [Why the verdict can be trusted](#why-the-verdict-can-be-trusted).

**Diagnostician** (Shipped). Not every red test means the model got something wrong, so each failure is classified before it counts:

| Class | What it catches | Counts against the patch? |
|---|---|---|
| Infrastructure | Sandbox crashes, timeouts, out-of-memory, runner failures, detected from process-level signals | No. The run is reset and retried. |
| Static | Errors before execution: syntax, types, imports | Tracked separately |
| Logic | The test ran and an assertion failed | **Yes.** This is the only class that counts as a model error. |
| Unknown | Execution failures that cannot be classified | No. Generated tests are rerun, logged and quarantined; repository tests are flagged for human review instead. |

**Governor** (Shipped as a fixed attempt cap; the stability rule is Planned). The Governor decides whether the loop continues. Today that decision is a fixed cap on attempts, with one meaning per mode (see the next section). It also records every transition between consecutive attempts on a fixed shared test set, which is the data the planned stability rule needs. ECR and EIR are computed and reported from that data today, but they do not yet make stop decisions.

**Gatekeeper** (Shipped). A deterministic final check. It approves only when every required signal passes, and it is the only component allowed to open a pull request.

---

## Two modes: review and repair

**Review mode** checks a patch that already exists, whether it came from a person or another agent. The Creator is skipped, and the patch is held fixed while the Attacker attacks it.

```bash
./target/release/vanta_m3 --repo ./target-repo --issue ./issue.md --patch ./candidate.diff --max-attempts 3
```

In review mode, `--max-attempts` is **attack depth**: each round is a fresh batch of Attacker tests against the same patch, and the run stops early only if the patch is rejected. A single round can happen to sample inputs the patch never touches, which is exactly how a bad patch slips through, so depth matters.

**Repair mode** lets the Creator write the patch and retry on feedback.

```bash
./target/release/vanta_m3 --repo ./target-repo --issue ./issue.md --max-attempts 4
```

In repair mode, `--max-attempts` is a **patch budget** (default 4). The loop stops as soon as an attempt passes every check, and otherwise feeds the logic-class failures back to the Creator as structured JSON for the next attempt.

---

## Why the verdict can be trusted

A verification tool is only as good as its resistance to being fooled, including by the patch it is checking. Vanta's grading rules exist because each one closed a real hole found during development.

**The patch cannot edit its own grader.** The native suite always runs the repository's *original* test files, restored onto the patched source. Test files the patch modified are reverted, tests it deleted are restored, and any `conftest.py` or test configuration it added is removed before grading. What the patch tried to change is still recorded as evidence (`modified_existing_tests`, `removed_existing_tests`, `added_test_files`, `modified_test_config`), so tampering is visible rather than silently undone.

**Silencing a test counts as breaking it.** A test that passed on the original code and is now skipped, marked as an expected failure, deselected or missing is counted as a regression, exactly like a test that now fails.

**Every test runs against the same tree.** Attacker tests run on the same graded snapshot the native suite saw, and BATE runs on the untouched original, so no stage judges a different version of the code from the one that was approved.

**Pre-existing failures are not the patch's fault.** BATE pins the baseline result of every test, and the native suite compares against the original run, so a test that was already failing before the patch never counts as a regression.

**Infrastructure noise is never a model error.** The Diagnostician keeps sandbox and timeout failures out of the logic class, and an LLM endpoint that times out or goes away ends the run as `InfraFailure` with its own exit code rather than as a rejection.

**Every signal carries its provenance.** Each result records where it came from (repository test, Attacker-generated test, static check, and later verifier or benchmark oracle), so evidence from a generated test is never mistaken for ground truth.

**Rejections can be re-checked.** A rejected session's graded snapshot is kept, and the rejecting signals can be re-run later under the backend that originally produced them. That turns "the Attacker caught it" from a claim into something you can reproduce.

---

## Measuring repair: ECR and EIR

Repair loops can make things worse: a later attempt can fix one test while breaking another. Vanta measures that directly with two rates computed over consecutive attempts on a fixed shared test set:

- **ECR, error correction rate**: of the tests that *failed* at attempt N, the fraction that *pass* at attempt N+1.
- **EIR, error introduction rate**: of the tests that *passed* at attempt N, the fraction that *fail* at attempt N+1.

Both are conditional rates, so their denominators are only the tests that could have changed in that direction, and a rate with a zero denominator is reported as `n/a` rather than as zero. They are computed separately for each evidence source (native, static, Attacker-generated) and combined. Transitions from review mode, censored transitions (an attempt that did not finish the shared set or whose patch failed to apply), tests that BATE identified as defective, and non-logic failures are all excluded, so only genuine, patch-caused logic changes enter the calculation. Newly generated tests can reject a patch immediately, but they only join the shared set for the *next* comparison, so a harder round of attacks never contaminates the rates.

The motivation comes from Liu and Meng, [*Self-Correction as Feedback Control*](https://arxiv.org/abs/2604.22273), which models LLM self-correction as a two-state Markov chain and derives a stability condition: correction is worth continuing only while `ECR / EIR > Acc / (1 − Acc)`. Vanta records everything needed to apply that condition, but its Governor still stops on a fixed cap. Whether the condition carries over is an open research question rather than an assumption, because the paper derives it for *intrinsic* self-correction on reasoning benchmarks, while Vanta's loop is driven by *external* test feedback against a test set that grows harder over time. The evaluation is designed to answer that question.

One consequence worth knowing: repair mode stops at the first attempt that passes, so a clean first-attempt fix produces no transition and contributes nothing to ECR or EIR.

---

## Quickstart

### Requirements

- **Rust** (stable) and Cargo.
- **macOS with [Shuru](https://github.com/superhq-ai/shuru)** for sandboxed runs. Shuru runs code in microVMs through Apple's Virtualization framework. On other platforms only the unsandboxed subprocess backend is available, which is meant for development on trusted code (see [Security model](#security-model)).
- **An OpenAI-compatible LLM endpoint.** Vanta is developed against a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` running a 27B-class Qwen model with a 131k context. Smaller models tend to fail at producing valid structured output and patches. Run the server with a single slot (`parallel = 1`), because concurrent slots share one KV cache and long agent prompts will overflow it.
- **Python 3 and pytest** inside the sandbox image for Python targets.

### Build and run

```bash
git clone https://github.com/Kyu-Yi/vanta.git
cd vanta
cargo build --release

export VANTA_LLM_BASE_URL=http://localhost:8080/v1
export VANTA_LLM_MODEL=<your-model-name>
export VANTA_LLM_CONTEXT_TOKENS=131072

./target/release/vanta_m3 --help
./target/release/vanta_m3 --repo fixtures/toy_calc --issue fixtures/toy_calc/issue.md --patch fixtures/toy_calc/bad.patch
```

`fixtures/toy_calc` is a small repository with a planted bug and a good and a bad patch, useful for checking that a setup works. See its own README for the expected results.

### CLI options

| Option | Meaning |
|---|---|
| `--repo, -r PATH` | Repository to verify or repair (required) |
| `--issue, -i PATH` | Issue or bug description |
| `--patch, -p PATH` | Review an existing patch instead of generating one |
| `--max-attempts, -n N` | Attack depth in review mode, patch budget in repair mode (default 4) |
| `--test-cmd CMD` | Test command (default `python3 -m pytest`) |
| `--original-tree PATH` | A clean checkout of the base commit, for two-tree runs |
| `--model, -m NAME` | Override `VANTA_LLM_MODEL` |
| `--open-pr` | Open a draft PR after Gatekeeper approval |
| `--rerun-signals` | Re-run a recorded session's rejecting signals (see `--help` for usage) |

---

## Configuration

Vanta is configured through environment variables. Unset variables use the defaults below.

| Variable | Default | Meaning |
|---|---|---|
| `VANTA_LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible endpoint |
| `VANTA_LLM_MODEL` | built-in default | Model identifier sent with each request |
| `VANTA_API_KEY` | unset | Bearer token, if the endpoint needs one |
| `VANTA_LLM_CONTEXT_TOKENS` | `65536` | Context size used to budget prompts. Set it to your server's real context; Vanta also reads `n_ctx` from llama-server's `/props` when available. |
| `VANTA_LLM_IDLE_TIMEOUT_SECS` | `180` | Give up on a streaming response that sends nothing for this long |
| `VANTA_LLM_MAX_SECS` | `3600` | Hard limit on a single LLM call |
| `VANTA_LLM_RETRY_ATTEMPTS` | built-in default | Retries for a failed LLM call |
| `VANTA_LLM_SEED` | unset | Sampling seed. Unset sends no seed, so requests are byte-identical to an unseeded run. |
| `VANTA_ATTACKER` | `on` | `off` disables the Attacker, which gives the no-Attacker baseline used in evaluation |
| `VANTA_PLANNER` | `off` | Reserved for the planned multi-Attacker planner. `on` currently ends the run as `InfraFailure`. |
| `VANTA_SANDBOX_BACKEND` | `auto` | `auto`, `shuru` or `subprocess` |
| `VANTA_SHURU_CHECKPOINT` | `vanta-python` | Shuru checkpoint to boot test VMs from |
| `VANTA_SHURU_DISK_MB` | built-in default | Disk size for Shuru VMs |
| `VANTA_SHURU_AUTO_DEPS` | on | Build a dependency checkpoint for the target repository automatically |
| `VANTA_TEST_TIMEOUT_SECS` | built-in default | Timeout for a test run |
| `VANTA_GITHUB_TOKEN` | unset | Token for `--open-pr` (see below) |
| `VANTA_LOG_LEVEL` | `info` | Log verbosity |

---

## Outputs

Each run writes a session directory under `vanta_sessions/<session-id>/`:

- a JSON Lines session log with one event per stage, every result tagged with session, attempt, iteration and test IDs, ending in a `run.summary` event that records the final decision, the backend that actually ran, prompt versions, sampling settings and the build commit;
- `<session-id>_transitions.jsonl`, the attempt-to-attempt rows that ECR and EIR are computed from;
- `shared_tests.json`, the fixed shared test set;
- `debug/`, the patched and graded tree snapshots for each attempt.

The final decision is one of:

| Decision | Meaning | Exit code |
|---|---|---|
| `AllTestsPassed` | Approved by the Gatekeeper | 0 |
| `Rejected` | The patch demonstrably fails a logic-class check or regresses a native test | 1 |
| `MaxAttemptsReached` | Repair mode used its whole budget without an approved patch | 1 |
| `PatchApplicationFailed` | The patch did not apply | 1 |
| `HarnessFailure` | The test harness itself could not produce usable evidence | 1 |
| `InfraFailure` | The LLM endpoint or another dependency failed, so nothing was learned about the patch | 2 |

`scripts/metrics.py` summarises sessions against a labels file into true and false approvals and rejections, unresolved runs and infrastructure failures, and reports ECR and EIR.

---

## Opening pull requests

With `--open-pr`, Vanta opens a pull request only after the Gatekeeper approves, and opens it as a draft by default. It authenticates only with `VANTA_GITHUB_TOKEN`, which should be a fine-grained token scoped to the one repository you intend to target, and it never falls back to `gh` or `git` credentials. It refuses to open a PR from a target repository with uncommitted changes, and the PR body records the session ID, the tested commit and a summary of the evidence. An example from the v1 acceptance run is [vanta-pr-test#6](https://github.com/Kyu-Yi/vanta-pr-test/pull/6).

---

## Security model

Vanta executes code it did not write and cannot trust: the target repository's own code and tests, and tests written by an LLM. All of it runs inside a sandbox behind a single Rust `Sandbox` trait, so the orchestration code does not know or care which backend is underneath.

| Backend | Platform | Isolation | Status |
|---|---|---|---|
| Shuru | macOS | microVM (Apple Virtualization framework) | Shipped, the default for real runs |
| Subprocess | any | process limits only, **no isolation** | Shipped, for development on trusted code |
| Firecracker | Linux | microVM (KVM) | Planned |
| WSL2 | Windows | Hyper-V | Planned |

The backend that actually ran is recorded in every session, and re-running a session's signals uses that same backend, so a run can never be quietly downgraded from a microVM to a plain subprocess.

---

## Evaluation

**What exists.** The v1 acceptance test was an unattended repair run on a real Python repository, which produced an approved patch with zero regressions across a 455-test native suite and opened [vanta-pr-test#6](https://github.com/Kyu-Yi/vanta-pr-test/pull/6). In that run the repository's own suite already covered the planted bug, so the native gate carried most of the evidence. Separately, Vanta was developed against a real-world case study of six review and repair cases on a local fork of an open-source project, where it reached the correct verdict in four cases, with no false rejections and two false approvals. Both false approvals came from a clause of the issue that no test exercised: one needed a test that a function is still called exactly as before, the other needed real GPU hardware to show a device-placement bug. That case study was used to find and fix weaknesses in Vanta itself, including the test-tampering hole described above, so it is a development set and its numbers are not a measurement.

**What is being built.** The measurement comes from a [BugsInPy](https://github.com/soarsmu/BugsInPy) harness (`scripts/bugsinpy_run.py`, design in [`plans/bugsinpy.md`](plans/bugsinpy.md)), currently **in progress**. For each historical bug, Vanta sees the buggy source together with the fixed version's tests, minus the tests that the real fix makes pass. Those hidden fail-to-pass tests become an oracle Vanta never sees, and a patch is labelled good only if the hidden tests pass and no other test in the suite regresses. Each bug is run under two configurations, Vanta and a baseline with the Attacker turned off, in repair mode first and then in review mode, where both configurations review the identical set of candidates: the developer's real fix, a no-op patch, and up to two failed patches pooled from the repair runs. Because both configurations judge exactly the same patches, the comparison isolates what the Attacker adds. The harness pins each bug's dependencies to versions released before the bug's commit, runs every suite inside Shuru, records full provenance, and stops between waves if a sanity gate fails, for example if the developer's fix is graded as bad or the no-op is graded as good.

**What comes after.** Defects4J as a cross-language check, SWE-bench-Lite with a regression-rate metric, and comparisons of ECR and EIR across model tiers.

---

## Limitations and open questions

- **The evidence is thin.** One acceptance run, one development case study and toy fixtures. Toy catch rates (for example 5 of 6 bad patches caught on the boundary fixture) are development results, not evidence, and nothing yet shows that Vanta beats a simple baseline.
- **Untested requirements can still pass.** Vanta can only reject what some test can observe. If no test, native or generated, exercises a clause of the issue, a patch that ignores that clause can be approved. Both false approvals in the case study were this kind.
- **The stopping theory may not transfer.** See [Measuring repair](#measuring-repair-ecr-and-eir). This is the main research question, not a settled design.
- **Generated tests are only as good as the Attacker.** BATE filters malformed tests but cannot prove a test checks the right thing, and the boundary-value behaviour is still prompt-dependent.
- **Local models are slow.** A single repair run on a local 27B model can take tens of minutes, and long prompts need a large context window and a single-slot server.
- **macOS only, for now.** Sandboxed runs need Shuru, and the Linux backend is planned.
- **Single maintainer.** The scope is large for one person, and the roadmap is ordered so that evidence comes before new features.

---

## Roadmap

**Shipped (v0.2)**
- [x] Creator and Attacker agents with intent-labelled tests tied to the issue
- [x] BATE baseline validation with revise-then-discard
- [x] Four-way Diagnostician with quarantine for unknown failures
- [x] Tamper-resistant native test gate, with silenced tests counted as regressions
- [x] Review mode with attack depth, and repair mode with feedback
- [x] Fixed-cap Governor with transition logging on a fixed shared test set
- [x] Conditional ECR and EIR per evidence source, and session metrics
- [x] Re-running rejecting signals under the recorded backend
- [x] Streaming LLM calls with idle timeouts, and `InfraFailure` as a separate outcome
- [x] Shuru sandbox backend and draft PR integration

**In progress**
- [ ] BugsInPy harness: paired baseline-versus-Vanta runs with hidden-test oracles
- [ ] Wilson confidence intervals for pooled ECR and EIR

**Planned**
- [ ] Live ECR/EIR stopping rule in the Governor
- [ ] A planner and shared board that coordinate several Attackers on a pool of VMs without overlapping work
- [ ] Formal verifier backends behind a `Verifier` trait (CrossHair for Python, Kani for Rust), whose outcomes are already defined as `VerifiedWithinBounds`, `Counterexample`, `Inconclusive`, `Unsupported`, `Timeout` and `ToolFailure`. Where the properties come from is still open, and it matters: properties written by an LLM would not be an independent check, so a verifier verdict will carry the provenance of its property, and only properties from a human-written specification will count as ground truth.
- [ ] Institutional memory of how a codebase has broken before, starting with JSON incident files and moving to a [Graphiti](https://github.com/getzep/graphiti) bi-temporal graph
- [ ] Firecracker (Linux) and WSL2 (Windows) sandbox backends
- [ ] Defects4J and SWE-bench-Lite evaluations, and comparisons across model tiers
- [ ] The paper, from real results, on arXiv (cs.SE)

---

## Repository layout

```
src/
  agents/        Creator and Attacker
  pipeline/      driver, BATE, harness, patch application, Diagnostician, Governor, Gatekeeper
  sandbox/       Sandbox trait with the Shuru and subprocess backends
  llm/           OpenAI-compatible client with streaming and timeouts
  config/        environment-variable configuration
  core/          shared types: test results, intents, evidence sources, decisions
  session/       session logs and transition records
  pr/            GitHub pull request integration
  bin/vanta_m3.rs  the CLI
scripts/         metrics and the BugsInPy harness, with its tests
fixtures/        small repositories with planted bugs and good and bad patches
plans/           design documents, including the BugsInPy evaluation plan
tests/           Rust integration tests
```

`vanta_sessions/` and `runs/` hold run output and are not committed.

---

## Why Rust?

The hard part of Vanta is not calling an LLM, it is orchestrating untrusted processes reliably: booting sandboxes, enforcing timeouts, cleaning up after crashes and never losing evidence. Rust's type system and explicit ownership make that kind of process control dependable, and the one `Sandbox` trait over several platform backends is a natural fit for it. The code being verified can be in any language, since test runners and analysers are called as subprocesses.

---

## Contributing

Vanta is in early, single-maintainer development, but issues, ideas and discussion are welcome, especially from anyone who has watched a coding agent confidently break something in production.

## Citation

If you build on the stopping-rule design, please cite the framework it is based on:

```bibtex
@article{liu2026selfcorrection,
  title   = {Self-Correction as Feedback Control: Error Dynamics,
             Stability Thresholds, and Prompt Interventions in LLMs},
  author  = {Liu, Aofan and Meng, Jingxiang},
  journal = {arXiv preprint arXiv:2604.22273},
  year    = {2026}
}
```

A citation for the Vanta paper will be added once it is on arXiv.

## License

[MIT](LICENSE)

---

<p align="center">
Built by <a href="https://github.com/Kyu-Yi">Nathan Lawson</a> · KAIST MSc AI/CS · Daejeon
</p>
