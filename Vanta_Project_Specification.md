# Vanta — Complete Project Specification

**Revision 4 · Current State · compiled 27 August 2026**

Autonomous adversarial verification for AI-generated code.

Nathan Owen Lawson · KAIST MSc AI/CS · github.com/Kyu-Yi/vanta

> *AI coding tools are getting better at producing code. The hard part is proving it doesn't break anything.*

---

## 0. Document Status

This is the single current-state reference for Vanta. Where any earlier note conflicts with this document, this document wins. Revision 4 folds in the execution-layer/security-boundary distinction, static-analysis baseline differencing, process lifecycle handling, provider-native structured output, and the local-model-first development decision.

Superseded framings — do not reintroduce

| Dropped idea | Current reality |
|---|---|
| V1 in Python/LangGraph, later rewritten to Rust | Rust from day one. There was never a Python orchestration phase. |
| Python/FastAPI is Vanta’s scope | Language-agnostic. Python is the first target, not the boundary. |
| “checkout-core” is part of the architecture | A throwaway demo target only, built after V1 works. |
| ContextWeaver / Wink / “Phoenix” frameworks | Dropped — could not be verified as real. |
| V1 and V2 are two codebases | One architecture; V1/V2 track component sophistication, not a rewrite. |
| Z3 is the universal solver behind every language | Wrong. Backends are complete verification systems with their own machinery; several use Z3 internally. |
| A subprocess backend is a sandbox | Wrong and unsafe. A subprocess is a controlled executor for trusted code only; the microVM is the security boundary. |
| GCF is the V1 diagnostic wire format | JSON in V1. GCF is a V2 ablation experiment. |

> Confirmed-real building blocks Rust, Rig, Shuru, Firecracker, Z3, CrossHair, Kani, Graphiti, cargo-nextest, Schemars, Ruff, mypy, BugsInPy and Defects4J are all real, current, and independently verifiable. Every architectural claim in this document rests on one of them or on the cited literature.

## 1. Executive Summary

Vanta verifies AI-generated code changes before they are allowed to merge. Mainstream coding agents optimise for one question: did the failing test pass? Vanta answers a stricter one: did this change fix the issue without breaking anything else, violating a historical constraint, or failing a check of its stated properties?

It pairs two adversarial LLM agents — a Creator that writes patches and an Attacker that writes tests designed to break them — with independent, non-LLM sources of truth: executed tests in a controlled backend, formal and symbolic verification backends, and a stability governor grounded in a published Markov model of self-correction. A patch ships only when it passes every required verdict source.

The patch under verification does not have to come from Vanta’s own Creator. It can arrive from an external coding agent or a human developer. Vanta is a verification layer, not merely a repair loop.

| Property | Value |
|---|---|
| What it is | Adversarial verification and repair agent for AI-generated code |
| Language of Vanta itself | Rust (orchestration via the Rig framework, pending spike verdict) |
| Languages it can verify | Any — per-language tooling dispatched as subprocesses |
| Core novelty | Adversarial test generation + formal verification + Markov-governed stopping, as one measured loop |
| Research contribution | Domain transfer of EIR/ECR stability analysis to adversarially verified code repair |
| Primary dev machine | 16-inch MacBook Pro M5 Max, 128GB — macOS-first |
| Model strategy | Local models (Ollama/llama.cpp) for development; frontier APIs for the tier experiments |
| Status | Design complete; implementation beginning August 2026 |

## 2. The Problem

Roughly 55.8% of AI-generated code artifacts contain identifiable vulnerabilities, and instructing the model to “write secure code” improves this by only about 4 percentage points. The failure is architectural, not instructional. You cannot prompt your way out of it. A second reviewer is required, and that reviewer should not itself be a single unchecked model.

Two questions sit at the heart of code repair. Almost every tool answers only the first:

“Can I make the failing test pass?”  — what SWE-bench and most coding agents measure.

“Did I break anything that already worked?”  — what production engineers care about, and what almost nothing measures.

A patch that fixes the reported bug while silently regressing three unrelated routes is often worse than no patch at all: the new failure is unexpected, untraced, and may reach users before anyone notices. Independent evidence supports this concern directly — PatchDiff (ICSE 2026) found that roughly 29.6% of patches counted as successful on SWE-bench diverge behaviourally from the correct fix, and 82.7% of those divergences are invisible to the existing test suites.

## 3. Design Principles

| Principle | What it means in practice |
|---|---|
| Language-agnostic | Code under repair may be any language; Vanta’s own code is Rust and calls per-language tools as subprocesses. |
| Adversarial by construction | A dedicated Attacker agent actively tries to break every patch, rather than passively checking a fixed suite. |
| Never trust the LLM for verdicts | Every accept/reject decision rests on deterministic evidence: test results, verifier outcomes, exit codes. |
| Measure only what the patch caused | Infrastructure noise, pre-existing failures, and pre-existing static diagnostics are excluded before anything counts as a model error. |
| Stop on a measured condition | The repair loop halts on a stability condition, not an arbitrary iteration cap (V2; V1 uses a fixed cap while gathering the data). |
| Isolation is a claim you must earn | Only the microVM backend is a security boundary. A subprocess executor is for trusted repositories and is documented as such. |
| Provenance on every signal | Each piece of evidence records where it came from and what produced it, because sources are not equally trustworthy. |

## 4. System Architecture

Vanta is a Rust orchestration plane coordinating two LLM agents and three independent verdict sources. Orchestration, backend control, memory, and the governor are native Rust. Other languages appear only as subprocesses: test runners, static analysers, and verification backends.

> Issue or feature request + patch attempt      (from Vanta's Creator, another agent, or a human)             |             v   [ Attacker Agent ] -- tests, each with a DECLARED INTENT             |             v   PHASE 1  BATE  baseline validation      run each test on the UNMODIFIED tree; result must      MATCH its declared intent             |      +------+---------------------------+      |                                  |      v                                  v   VERIFIER BACKENDS               EXECUTION BACKEND   CrossHair / Kani / Z3           subprocess or microVM   structured outcome              PHASE 2 four-way triage      |                                  |      +------+---------------------------+             |             v   PHASE 3  GOVERNOR   stability check on the            fixed shared test set             |             v   [ GATEKEEPER ]  every REQUIRED signal passes -> PR                   any required failure -> feedback, retry             |             v   Institutional memory  (JSON -> Graphiti)

| Component | Role | Trusts the LLM? |
|---|---|---|
| Creator agent | Reads issue, source, and history; writes a minimal patch. Optional — patches may arrive from elsewhere. | n/a (is the LLM) |
| Attacker agent | Given a patch, writes adversarial tests with declared intent targeting its weak points. | n/a (is the LLM) |
| Execution backend | Runs tests in a controlled environment; microVM when isolation is required. | No — deterministic |
| Diagnostician | Triages every failure into infrastructure, static, logic, or unknown. | No — deterministic |
| Verifier backends | Language-specific formal/symbolic systems behind one trait; return structured outcomes. | No — mathematical |
| Governor | Computes EIR/ECR on the fixed shared set; halts when correction turns net-harmful. | No — arithmetic |
| Gatekeeper | Opens a PR only when every required signal passes under repository policy. | No — deterministic |

## 5. The Three-Phase Error System

Not every failing test means the same thing. Treating them identically wastes compute and, more importantly, contaminates the EIR/ECR measurements at the centre of the research contribution. Only genuine, patch-caused, logic-level failures may ever count as a model error.

### Test intents — the prerequisite for everything else

A test cannot be judged against the original code without knowing what it is for. The Attacker receives the issue, the patch, and the surrounding code, so it returns each test with a declared intent and an expected baseline.

| Intent | Purpose | Expected on ORIGINAL | Expected on PATCHED |
|---|---|---|---|
| bug_reproduction | Proves the reported defect exists | FAIL | PASS |
| feature_acceptance | Checks a requested new capability | FAIL / unsupported | PASS |
| behavior_preservation | Checks unrelated working behaviour survived | PASS | PASS |

> Why the old rule was wrong The previous rule — ‘every Attacker test must pass on the unmodified code’ — holds only for behaviour-preservation tests. A bug-reproduction test SHOULD fail on the original: that failure is evidence the test successfully reproduced the reported defect, not evidence the test is bad. On fail-before/pass-after benchmarks like BugsInPy, this distinction is the whole game.

A planned refinement separates intent from expected baseline as two fields, because two feature-acceptance tests can legitimately fail differently on the original tree (a clean assertion failure versus a route that does not exist), while a syntax error in the generated test is malformed regardless of what baseline was expected.

### Phase 1 — BATE, baseline-aware test evaluation

Every generated test runs against the unmodified tree first, and its result is compared to what its declared intent predicts. A match validates the test and records the baseline; a mismatch sends it back for revision.

> Attacker generates test  (with declared intent)    -> run against ORIGINAL tree         baseline MATCHES intent              -> test is valid, baseline recorded -> Phase 2         baseline CONTRADICTS intent              -> return to Attacker to revise                 retry, max 3 attempts                 still mismatched after 3 -> SCRAP,                         log reason, continue

BATE does not prove a generated test is semantically correct. A badly written test can still accidentally behave as its intent predicts. BATE is a filter that confirms behaviour matches declared purpose and pins the original-code baseline, so that later regression counting only ever charges a patch for damage it caused. Scrapped tests are logged with reasons as an Attacker-reliability metric for the research write-up.

### Phase 2 — execution and the four-way Diagnostician

Validated tests run against the patched tree in the selected execution backend. Failures are triaged into four classes; the fourth exists because pretending every failure is classifiable corrupts the measurements.

| Class | Catches | Mechanism | Counts as model error? |
|---|---|---|---|
| Infrastructure | Backend spawn/transport failure, abnormal signals, timeouts, OOM, health-check failure | Structured result inspection, not exit code alone; reset backend and rerun | No — excluded |
| Static | Syntax errors, type errors, import failures, newly introduced lint diagnostics | Syntax compilation plus repository-configured linters, run before execution, differenced against baseline | Tracked separately |
| Logic | Test ran, code executed, assertion failed | Structured diagnostic payload returned to the patch source | Yes — the only class |
| Unknown | Unclassifiable execution failures | Rerun in fresh backend, capture logs, simplify, quarantine if persistent | No — excluded, logged |

### Two rules that are easy to get wrong

A non-zero exit code is not by itself infrastructure failure. pytest exits non-zero on ordinary test failures. Classification must inspect the structured result, not the exit status alone.

Static analysis must be differenced against the baseline. Run the same static commands on both the unmodified and patched trees; only newly introduced diagnostics are patch-caused. Pre-existing diagnostics are baseline evidence. An unconfigured optional tool is skipped, a tool that fails to launch is Infrastructure, and an ambiguous result is Unknown — never a patch error.

Use the repository’s own tooling. A repository may configure Ruff, mypy, both, or neither. Discover the configuration rather than imposing a house style; imposed tools generate diagnostics that say nothing about the patch.

Unknown handling

> Unknown failure    -> rerun in a fresh backend instance         disappears -> transient infrastructure noise         repeats   -> capture logs, exit status, stdout/stderr,                      runner metadata                   -> ask Attacker to simplify the test, rerun                   -> still unknown: QUARANTINE the generated test,                      exclude from EIR/ECR, log the reason

Repository tests and benchmark oracle tests are never quarantined. An unknown failure in ground-truth tests escalates to ‘verification inconclusive’ or ‘human review required’, because discarding inconvenient ground truth is how a measurement pipeline quietly starts lying.

### Phase 3 — the Governor

The stopping condition for the loop, detailed in Section 8. It also handles the case of a subtly wrong test that survived Phase 1: if patches keep failing the same valid test with no EIR/ECR improvement, the Governor halts and raises a possible test-or-specification conflict for review. Persistent failure may mean the test is wrong, the patches are wrong, the requirement is misunderstood, or the bug is genuinely hard. Persistence alone identifies none of these.

> Why this matters for the paper, not just the product Only Phase-2 Logic outcomes, on Phase-1-validated tests, net of the recorded baseline, on the fixed shared test set, ever feed the EIR/ECR calculation. Without this pipeline the reported rates would be contaminated by flaky infrastructure, malformed tests, pre-existing defects, and unclassifiable failures — and a careful reviewer would reject the paper on exactly that hole.

## 6. Formal and Symbolic Verification

A patch faces two separate checks that can each reject it independently. The execution track asks whether it passes the evolving test suite. The verification track asks whether it is sound against stated properties, running on the patch source in parallel. A patch can pass every test and still be rejected here.

### One interface, per-language backends

CrossHair, Kani, OpenJML, and CBMC are not ‘translators to Z3’. Each is a complete verification system with its own rules, machinery, and limitations. Several use Z3 internally, and a direct Z3 backend exists for properties already expressed in logical form. Vanta does not depend on those internals: it asks each backend one question through a single trait and receives a structured answer.

> trait Verifier {     fn verify(&self, request: VerificationRequest)         -> VerificationOutcome; }   enum VerificationOutcome {     VerifiedWithinBounds,   // no violation within checked limits     Counterexample(Input),  // concrete input that violates it     Inconclusive,           // tried, could not decide     Unsupported,            // backend cannot handle this code     Timeout,                // exceeded the verification budget     ToolFailure,            // the verifier itself misbehaved }

| Target | Backend | Notes | Stage |
|---|---|---|---|
| Python | CrossHair | Symbolic execution inside the Python interpreter; uses Z3 internally; unavoidable Python subprocess | V2 — first backend |
| Rust | Kani | Model checker, AWS-maintained; compiles through CBMC machinery | V2 |
| Pure properties | Z3 direct | Checks properties already encoded in SMT-LIB or Vanta’s internal logical form; cannot verify arbitrary source without a language model in front | V2 optional |
| Java | OpenJML | Mature | Future |
| C / C++ | CBMC | Very mature bounded model checker | Future |
| JS / TS | — | No viable backend; verification track skipped, adversarial suite only | Skipped |

### Reading the outcomes honestly

VerifiedWithinBounds is not universal proof. A bounded checker may explore loops only to a set depth. The outcome means ‘no violation found within the assumptions and limits checked’, and Vanta records those bounds.

Counterexample is strong rejection evidence. ‘charge=100, refund=100.01 — refund exceeds charge’ rejects the patch outright, regardless of test outcomes.

Inconclusive, Unsupported, and Timeout are first-class results, not errors. Real code produces them constantly. They report as ‘not verified’, never as ‘verified’.

### Verification policy — what non-binary outcomes do to approval

A six-valued outcome enum means ‘all pass’ is not a decision rule. Each repository, or path within it, sets a policy defining how the Gatekeeper treats each outcome.

> enum VerificationPolicy {     Required,    // payment / auth / critical paths     BestEffort,  // ordinary application code (V1 default)     Disabled,    // unsupported languages }

| Outcome | Required | BestEffort |
|---|---|---|
| VerifiedWithinBounds | Continue | Continue |
| Counterexample | Reject | Reject |
| Inconclusive | Block / human review | Report and continue |
| Unsupported | Block / human review | Report and continue |
| Timeout | Retry, then block/review | Retry, then continue unverified |
| ToolFailure | Infrastructure retry | Infrastructure retry |

### Verifier backends run in isolation too

The verification track does not execute the normal test suite, but its backends still import, compile, or symbolically execute target code. CrossHair loads Python modules, which can trigger module-level code, file access, network access, and native extensions. Verifier backends therefore run in their own restricted environment — microVM, container, or similarly constrained worker — rather than on the host. Formal verification is not automatically safe host-side execution.

> Known limitation — stateful endpoints Symbolic execution works cleanly on pure functions. Real endpoints call databases, caches, and external services that cannot be symbolically executed. Planned mitigation (V2+): a mocking layer that replaces stateful calls with symbolic mocks bounded by interface contracts — ‘find_order returns a valid Order or NotFound’ — so the backend reasons about the interface rather than the database. Execution tests exercise real behaviour; verification reasons over simplified contracts. Two layers, two different problems.

## 7. Execution Backends and Institutional Memory

### The execution layer, and what ‘isolation’ actually means

A single Rust trait exposes spawn, execute, and destroy. The orchestrator never knows which backend it is on. But the backends differ in a way that matters more than portability: only some of them are security boundaries.

| Backend | Technology | Security boundary? | Stage |
|---|---|---|---|
| subprocess.rs | Plain OS process in a temp copy of the tree | NO — trusted repositories only | V1 fast path |
| shuru.rs | Apple Virtualization.framework microVM via Shuru | Yes — the real boundary on macOS | V1 primary |
| firecracker.rs | Firecracker microVM on KVM | Yes | V2, Linux via CI/VPS |
| wsl.rs | WSL2 / Hyper-V | Weaker than a dedicated hypervisor call | Later |

> The distinction that must never be blurred A subprocess shares the host kernel, filesystem, and network. It is a controlled executor, useful for running trusted benchmark repositories quickly, and it is the fast path to a working V1. It is not a sandbox and must never be described as one. Untrusted or model-generated code that could plausibly do damage runs in the microVM backend, or it does not run.

### Process lifecycle requirements, all backends

Kill and reap. On timeout, terminate the child and collect its exit status. A timeout future that returns is not the same as a process that stopped.

Handle descendants deliberately. pytest spawns children. Killing one PID may leave orphans running, corrupting later runs and producing exactly the phantom infrastructure failures that pollute measurements. Process groups exist for this.

Network disabled by default. Enabling it requires an explicit backend policy and host allowlist.

Time limits everywhere; hard memory limits belong to the microVM. Any subprocess-level memory limit is platform-specific and best-effort, and is documented as such.

Fresh state per run. Each execution starts from a clean copy or mount, with deterministic cleanup afterwards.

### Institutional memory

The Attacker is considerably more dangerous when it remembers how this codebase has broken before. Memory feeds historical failure modes into adversarial test generation.

| Version | Implementation | What it buys |
|---|---|---|
| V1 | Flat JSON incident files | Simple lookup; enough to show historical context sharpens adversarial targeting. |
| V2 | Graphiti bi-temporal knowledge graph | Tracks event time and ingestion time independently, preserving causal chains a similarity-ranked store collapses. |

> Incident INC-2024-017    +-- Caused_By  -> PR #312           +-- Violates -> Policy: mobile-schema-stability                  +-- Affects -> checkout.py, schemas.py

Bi-temporal memory suits engineering history because that history is causal: a March incident causing an April policy change should shape how the Attacker tests a June patch. It also supports point-in-time queries during an active session — what was known at attempt 2 that was not known at attempt 1.

### One driver, thin binaries

There is exactly one repair loop, in one driver module. The M2 and M3 binaries are thin configuration over it: M2 is the same loop with a one-attempt budget, M3 with a larger one. Nothing about a stage is conditional on which binary is running. This is deliberate — a second loop, or a flag that disables a stage in one binary only, is how the two paths drift apart and a stage silently stops running in one of them. Configure the loop; never fork it.

## 8. The Stability Governor and Research Contribution

The stopping rule is grounded in a 2026 paper: Liu & Meng, arXiv:2604.22273, which models iterative LLM self-correction as a two-state Markov chain over {Correct, Incorrect}, parameterised by an Error Introduction Rate (EIR) and an Error Correction Rate (ECR).

### The stability condition

> ECR        Acc        ----- >  ---------         EIR      1 - Acc     EIR = P(correct -> incorrect)   breaking a working thing   ECR = P(incorrect -> correct)   fixing a broken thing   Acc = current accuracy

The original paper validates this on reasoning benchmarks (GSM8K, MATH, StrategyQA) under intrinsic self-correction, finds a sharp near-zero EIR boundary around 0.5% separating beneficial from harmful correction, and shows a Verify-First prompt intervention driving one model’s EIR from 2% to 0%, converting a 6.2 point accuracy loss into a 0.2 point gain. Notably, capability does not predict loop safety: GPT-5 and Claude Opus 4.6 have similar baseline accuracy but a tenfold difference in EIR, and opposite outcomes.

What Vanta contributes

The paper names three limitations as future work. Vanta is built to address all three.

Domain transfer. Vanta investigates how EIR/ECR can be defined and measured on adversarially verified code repair, where correctness is graded across an entire evolving test suite rather than one answer string.

Non-stationarity. Because the Attacker generates harder tests each round, rates are non-stationary by construction. Vanta applies the per-transition condition live rather than the paper’s steady-state results.

External feedback. The paper evaluates intrinsic self-correction with no oracle. Vanta’s tracebacks, counterexamples, and verifier outcomes are exactly the exogenous signal the authors suggest could shift the balance.

> Honest scope of the contribution This is a domain transfer and empirical validation, not a new theorem. The mathematics is the original paper’s. The contribution is applying its per-transition condition as a live governor in a setting the authors could not reach, with a measurement pipeline that makes the numbers defensible, plus a verification signal that has no analogue in the self-correction literature.

### Measurement protocol — the fixed shared test set

EIR and ECR describe transitions between patch attempts, not between testing-depth iterations. Comparing two attempts requires asking both the same questions.

> Patch attempt 1 on shared set:  T1 pass  T2 FAIL  T3 pass  T4 pass Patch attempt 2 on shared set:  T1 pass  T2 pass  T3 FAIL  T4 pass     T2  fail -> pass   the new patch CORRECTED an error  (ECR)   T3  pass -> fail   the new patch INTRODUCED an error (EIR)

Without a fixed shared set, an attempt judged on twelve tests looks worse than one judged on four purely because it was asked more questions. Newly generated tests still run immediately and can reject a patch on the spot, but they are recorded separately and only join the shared set for the following comparison.

Censored transitions

If a failure is discovered mid-iteration, no further adversarial tests are generated for that attempt, but the fixed shared set is allowed to complete so the transition is fully observed. If execution cannot complete, the transition is marked incomplete and excluded from complete-set EIR/ECR estimation. A partially observed transition is never treated as complete: recording that only a subset ran identifies a hole, it does not fill one.

### Transition rows — the raw dataset

EIR/ECR are **derived from rows, never instead of them**. Every transition is persisted as a row so that the aggregate is recomputable after the fact; a bug in the counting code then costs a re-analysis, not a re-run of the model. Each row is keyed by the test's *body hash* — not its name — because a name is not an identity: the model reusing `test_edge_case` for a different body would otherwise swap evidence under a live row, and renaming one probe would look like new coverage.

Rows are written to `vanta_sessions/{session_id}_transitions.jsonl`, one row per line, appended and flushed as each transition is computed — never one JSON document written at the end. A run can end early by design (a `HarnessFailure` aborts mid-run) or by accident (the sandbox has already filled the disk once); a single end-of-run write would lose every row from the session, including the good attempts that ran before the failure. An append-and-flush sink keeps the dataset equal to the work already done, and a file truncated mid-write still parses line by line up to its last complete row.

| Field | Meaning |
|---|---|
| `session_id` | The run. Joins across sessions. |
| `from_patch_attempt_id`, `to_patch_attempt_id` | Deterministic fingerprints of the two candidates compared. Both ends are recorded because they are not always different: in `--patch` (review) mode the candidate never changes, so a comparison between review rounds is attack depth against one candidate, not a repair transition. With only the destination id those rows are indistinguishable from genuine repair transitions; with both, the patch-transition analysis is the single filter `from_patch_attempt_id != to_patch_attempt_id`. **Both ends are nullable.** When an attempt is skipped — censored, a `HarnessFailure`, or a patch that did not apply — there is no real id for that end, and the row records `null`. A neighbouring attempt's id is never filled in: that would be a guess written to disk as a fact, and it would corrupt the analysis in both directions (a differing guess reads as a repair transition, a matching guess as a review round). A null is itself filterable and means *exclude this*. The analysis rule is therefore **a row counts only when both ids are non-null and differ**. |
| `from_attempt`, `to_attempt` | The two attempts compared. |
| `test_id` | Normalised test-body hash. The stable join key across attempts. |
| `test_name` | Readability only; never used to match. |
| `shared_set_id` | Identity of the question set the comparison ran on. Changes as the Attacker grows the set, which is the non-stationarity the restricted analysis controls for. |
| `from_result`, `to_result` | Outcome on the older and newer attempt. |
| `from_failure_class`, `to_failure_class` | Both sides' classes. EIR is judged on the new failure's class, ECR on the old one's; a fail→fail row needs both. |
| `evidence_source`, `test_intent` | Provenance, so the oracle-only robustness check can re-select rows without a re-run. |
| `censored` | True when one side was not observed. A censored row is a recorded hole, not a deleted row. A null `from_patch_attempt_id` or `to_patch_attempt_id` forces `censored = true`: an unobserved end and a complete transition are never the same row. |
| `in_s0` | Membership of S⁽⁰⁾, the shared set as it stood on the first attempt. Required by the restricted analysis and not reconstructable later. |

Only `Logic`-class failures move EIR/ECR; non-logic rows are excluded at *calculation* time and the count of exclusions is reported, so an evidence-starved run reads as thin rather than as precise.

The whole counting rule — both ids non-null and differing, and every failing side `Logic` — has exactly **one** implementation. The live console and the analysis layer both call it, over the same persisted rows, so the number printed during a run is the number the paper reports. Two definitions of one metric drift apart, and this project has already paid for that pattern twice (two pipelines, two specs); a second EIR is exactly the divergence a single function and one test against it prevent.

The session log carries the run-level facts — provider, model ID, model release version, prompt version, Vanta commit **with a working-tree `dirty` flag**, backend, benchmark instance — that make a run attributable, and they live on session records, not on transition rows. The commit is recorded with its `dirty` state because during development most runs happen on a tree with uncommitted changes: a bare HEAD hash would name a commit that does not contain the edits the binary actually ran, so the run would claim a provenance it cannot reproduce. A `dirty` run is identifiable on disk, and by rule is not one that feeds the paper; a run whose `dirty` state cannot be determined is recorded as dirty, so an unverified provenance never masquerades as a clean one. It is therefore written the same way the transitions are: `vanta_sessions/{session_id}_session.jsonl`, one record per line, appended and flushed as each record is produced. A `session.start` record is appended at launch, **before attempt 1**, carrying the run identity; the rest of the session follows. A run that aborts mid-way (a `HarnessFailure` is a designed outcome, not an accident) then leaves transition rows *and* the configuration that produced them on disk. Without this, a crash yields transitions saying what changed with no record of the model that changed it — an unattributed run is unusable for a study whose central comparison is across model tiers.

### Evidence provenance

Sources are not equally trustworthy, and one provenance field is not enough. The source of a property and the engine that checked it are different facts.

> enum EvidenceSource {     ExistingRepositoryTest, BenchmarkOracle, AttackerGenerated,     HumanProvided, FormalVerifier, IssueDescription, }   struct TestEvidence {     test_source: EvidenceSource,   // who wrote the test     intent: TestIntent,     execution_source: TestRunner,  // what executed it     result: TestOutcome, }   struct VerificationEvidence {     property_source: EvidenceSource,     // who supplied the property     outcome: VerificationOutcome,     outcome_source: VerificationBackend, // which engine produced it }

This matters concretely. A solver may correctly prove that a patch satisfies ‘every refund is strictly less than the original charge’ while the real policy permits a refund equal to the charge. The solver did its job; the property was wrong. The log must therefore read ‘verified within bounds, property Attacker-generated’, never ‘mathematically proven correct’.

### Governor by version

| Version | Governor | Why |
|---|---|---|
| V1 | Fixed-N retry, default 4, every transition logged | Rates cannot be computed without a history to measure. V1 generates that history. |
| V2 | Live EIR/ECR on the shared set, plus Verify-First from the first correction attempt | Once logs exist, the cap is replaced by the real condition. Verify-First is evaluated as an intervention: the reasoning-domain result suggests it suppresses error introduction without harming correction, but whether that transfers to code repair is established empirically, not assumed. |

### Planned evaluation

BugsInPy is the primary and sufficient benchmark for the first paper. Defects4J changes both language and toolchain, and SWE-bench adds substantial environment and evaluation complexity; neither begins until the BugsInPy pipeline works end to end. Experiments run across four model tiers — frontier, mid, cheap API, and local — reporting fix rate, regression rate, EIR, ECR, governor firing rate, Attacker discard rate, and static failure rate.

Reproducibility requires recording exact provider, model ID, model release version, run date, prompts, attempt budget, sampling settings, dependency lockfile, and benchmark commit. Tier labels alone are not reproducible. The bug subset and inclusion rules are frozen before any model comparison begins.

> If results are negative or null Report them honestly and reframe as a measurement-methodology contribution. ‘External deterministic feedback dissolves the stability problem’ is a real finding, as is ‘even with perfect feedback, code repair still needs adversarial governance’. Do not force a positive claim. The experiment you cannot lose is the one you actually run.

## 9. Technology Stack

| Layer | Technology | Stage |
|---|---|---|
| Orchestration | Rust + Rig framework (pending spike verdict; reqwest fallback) | V1 |
| Development models | Local via Ollama / llama.cpp, OpenAI-compatible endpoint | V1 |
| Experiment models | Frontier and mid-tier APIs, plus local tiers | V5 |
| Structured output | Provider-native JSON schema where supported; Schemars for schema generation | V1 |
| Execution (macOS) | Shuru / Apple Virtualization.framework microVM | V1 primary |
| Execution (fast path) | Plain subprocess, trusted repositories only | V1 |
| Execution (Linux) | Firecracker on KVM, via CI or VPS | V2 |
| Execution (Windows) | WSL2 | Later |
| Test running | pytest with JSON report; cargo-nextest for Rust targets | V1 |
| Static analysis | Repository-configured Ruff / mypy, plus syntax compilation | V1 |
| Diagnostics format | Structured JSON | V1 |
| Diagnostics ablation | GCF, Graph Compact Format | V2, optional |
| Verification | Verifier trait: CrossHair, Kani, Z3 direct | V2 |
| Memory | Flat JSON, then Graphiti bi-temporal graph | V1 / V2 |
| Governor | Fixed-N, then live EIR/ECR with Verify-First | V1 / V2 |
| Integration | octocrab for GitHub PRs; clap for the CLI | V1 |

## 10. V1 and V2

V1 and V2 are not two codebases. They are the same architecture with components at different levels of sophistication. The orchestration plane is built once and never rewritten.

| Component | V1 | V2 |
|---|---|---|
| Orchestration | Creator/Attacker loop | Same, no rewrite |
| Execution, macOS | subprocess fast path, then Shuru | Same |
| Execution, Linux | — | firecracker.rs with CoW forking |
| Verification | None | Verifier trait: CrossHair, Kani, Z3 |
| Phase 1 BATE | Intent-matched baseline, 3-retry | Same |
| Phase 2 triage | Four-way, baseline-differenced static | Same, refined |
| Diagnostics | Structured JSON | Plus GCF ablation |
| Memory | Flat JSON | Graphiti bi-temporal |
| Governor | Fixed-N | Live EIR/ECR + Verify-First |

Definition of done for V1

> Given a real Python repository and a bug description, Vanta produces a pull request containing a patch plus Attacker-generated tests that pass baseline validation, with the whole loop running unattended.

## 11. Known Limitations and Risks

### Architectural limitations, documented rather than solved

| Limitation | Impact | Response |
|---|---|---|
| Subprocess backend is not isolation | Cannot safely run untrusted code | Documented explicitly; microVM used whenever isolation is claimed |
| Entropy duplication in CoW forks | Concurrent forks share CSPRNG state | Irrelevant to current scope; revisit only if Vanta enters security auditing |
| Stateful endpoints resist symbolic execution | Backends cannot reason through a real database | Interface-contract mocking layer (V2+) |
| No network inside execution backends | No mid-run dependency installation | Dependencies pre-baked into the base image |
| Attacker can write subtly wrong tests | A test may match its intent yet encode the wrong requirement | Persistent failures raised for provenance-aware review; discard rate logged |
| Static tools may not see the repo's own environment | Baseline differencing cancels most of it — an unresolved module misreports identically on both trees and the diagnostics cancel. It does not cancel a patch that adds an import of an uninstalled package: the diagnostic is genuinely new, is classed `Static`, and is really an environment gap | Documented as a threat; static passes must run inside the repo's own environment (V2) |
| EIR threshold may not transfer | Governor may halt too early or never fire | Verify-First and threshold behaviour measured, not assumed |

### Execution risks

> The single biggest risk Learning Rust from scratch while depending on a pre-1.0 framework and an experimental sandbox, on a December timeline, alongside a full course load. The most likely failure mode is that V1 slips and V2 never ships. The agreed mitigation is explicit: if the schedule slips by September, prioritise a finished, smaller V1 with real EIR/ECR numbers and the paper over a half-built V2. A complete V1 with real measurements beats an impressive but broken V2 on every axis that matters — hiring signal, paper viability, and demo credibility.

### Competitive reality

The space is more crowded than it first appears. Adversarial test generation for AI code is a funded product category: Qodo ships a regression-testing agent, and Greptile shipped TREX in June 2026, which writes and runs targeted tests per pull request in an isolated sandbox. Research prior art includes SWT-Bench, AutoCodeSherpa, and PatchDiff, and control-theoretic stopping for agent loops appeared in 2026 through CyberCorrect and POMDP/Bayesian-control papers.

Vanta’s genuine differentiation is the specific combination — adversarial generation, verification backends, and Markov-governed stopping as one measured loop — together with the measurement-validity pipeline. No competitor closes the repair loop under a stability governor, and none publishes the error dynamics. The paper, not the tool alone, is what establishes that the combination is novel.

## 12. Glossary

| Term | Meaning |
|---|---|
| Creator agent | The LLM that writes candidate patches. Optional; patches may come from elsewhere. |
| Attacker agent | The LLM that writes adversarial tests, each with a declared intent. |
| BATE | Baseline-Aware Test Evaluation. Running each new test on the unmodified tree and checking the result matches its declared intent. |
| Test intent | Why a test exists: bug_reproduction, feature_acceptance, or behavior_preservation. |
| Diagnostician | Deterministic triage classifying failures as Infrastructure, Static, Logic, or Unknown. |
| Execution backend | Where tests run. subprocess (trusted code) or microVM (security boundary). |
| Verifier | Trait over language-specific verification backends returning structured outcomes. |
| VerifiedWithinBounds | No violation found within the assumptions and limits the backend checked. Not universal proof. |
| Governor | The EIR/ECR stability check deciding when to stop the repair loop. |
| Gatekeeper | Final deterministic gate; opens a PR only when every required signal passes. |
| EIR / ECR | Error Introduction Rate and Error Correction Rate: how often the model breaks or fixes things. |
| Patch attempt | One version of a proposed fix. The unit EIR/ECR transitions are measured on. |
| Testing iteration | A depth level of adversarial testing within a single patch attempt. Not the same as a patch transition. |
| Fixed shared test set | The saved tests both patch attempts are judged on, so transitions are comparable. |
| Censored transition | A transition where the shared set could not complete; excluded from estimation. |
| Evidence provenance | Recording who produced a signal and what engine evaluated it. |
| Shuru | macOS microVM backend built on Apple Virtualization.framework. |
| Graphiti | Bi-temporal knowledge-graph library used for V2 institutional memory. |
| Rig | Rust framework for LLM agent orchestration; subject to a GO/NO-GO spike. |
| Non-stationary | Error rates that change each round because the Attacker’s tests keep getting harder. |

Vanta complete project specification · Revision 4 · supersedes all prior architecture notes · github.com/Kyu-Yi/vanta
