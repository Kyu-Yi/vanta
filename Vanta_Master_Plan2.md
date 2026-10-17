# Vanta Master Plan: Study + Build, M0 to M5

One document. Everything you need to learn and everything you need to build, in the order you need it.

How to use this:
- work top to bottom, one milestone at a time
- each milestone has a **STUDY** block and a **BUILD** block; study first, build second, both inside the same milestone
- do not start a milestone until the one above passes its exit test
- if you are ever unsure what to do next, find the first unchecked box

Companion documents:
- `Vanta_Master_Specification.docx` is the architecture source of truth
- `Vanta_V3_V4_Roadmap.docx` is post-December, ignore until V1 ships

The rule that overrides everything:
> If it is not in the current milestone, it does not get built. Scope creep is the most likely cause of failure.

The rule that overrides *that*:
> Every study block ends with code you typed. Reading without typing is not progress.

---

## Timeline and honest expectations

| Milestone | Delivers | Target | Conditions |
| --- | --- | --- | --- |
| M0 | Rust literacy + Rig GO/NO-GO | August 2026 | 3 weeks, no classes |
| M1 | Creator and Attacker agents talking | September 2026 | 4 courses start Aug 31 |
| M2 | Execution, BATE, Diagnostician | Sep-Oct 2026 | 4 courses |
| M3 | **V1 SHIPS** | Oct-Nov 2026 | 4 courses |
| M4 | V2 components | Nov-Dec 2026 | optional, cut first |
| M5 | Experiments + paper | Nov-Dec 2026 | non-negotiable |

**Realistic probability of V1 by end of October: 40-50%. By end of November: about 70%.**
The two months of M1-M3 give you roughly 180 working hours against 150-210 hours of work, with no slack. That is why the cut order at the bottom of this document exists, and why M2 has a documented fast path.

**What actually has to be true by December:** V1 shipped, EIR/ECR instrumentation working, paper started. V2 is a cherry on top. If you chase V2 and V1 slips, you arrive at applications with nothing finished. Never let that happen.

---

# M0 — Foundations

**What it is:** learning enough Rust to write Vanta, and deciding whether Rig is the right foundation.
**Target:** August 2026, three weeks, no classes.

## M0 STUDY

### Phase 0.1 — Anthropic API key
The entire spike calls Claude. Without a key, nothing runs.

- create a key at https://console.anthropic.com/settings/keys
- `export ANTHROPIC_API_KEY="sk-ant-..."` in `~/.zshrc`
- read it in Rust with `std::env::var("ANTHROPIC_API_KEY")`
- never commit it, never paste it into a `.rs` file even temporarily

Done when: `echo $ANTHROPIC_API_KEY` prints your key in a fresh terminal.

### Phase 0.2 — rustlings, running alongside all reading below
Reading alone does not teach Rust; the compiler does.

- https://github.com/rust-lang/rustlings
- `cargo install rustlings`, then `rustlings init`, then `rustlings`
- do each topic's exercises immediately after that topic's reading

Done when: ownership, structs, enums, collections, error handling, and modules exercise sets are complete.

### Phase 0.3 — Ownership and Borrowing
Why: you will pass data across modules constantly; this concept affects everything else.

- https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
- https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
- https://doc.rust-lang.org/book/ch04-03-slices.html

Done when: you can explain `String` vs `&str`, know when to borrow instead of clone, and understand `&` and `&mut`.

### Phase 0.4 — Structs and Methods
Why: Creator and Attacker need typed request and response models.

- https://doc.rust-lang.org/book/ch05-01-defining-structs.html
- https://doc.rust-lang.org/book/ch05-03-method-syntax.html

Done when: you can write a struct for agent input and add an `impl` block with helper methods.

### Phase 0.5 — Enums and Pattern Matching
Why: `match` is how every verdict, policy, and outcome gets handled.

- https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html
- https://doc.rust-lang.org/book/ch06-02-match.html
- https://doc.rust-lang.org/book/ch06-03-if-let.html

Done when: you can explain `struct` vs `enum` and `match` on a `VerificationOutcome`.

### Phase 0.6 — Common Collections
Why: attempt logs, test lists, and payloads are all collections.

- https://doc.rust-lang.org/book/ch08-01-vectors.html
- https://doc.rust-lang.org/book/ch08-02-strings.html
- https://doc.rust-lang.org/book/ch08-03-hash-maps.html

Done when: you can build and iterate a `Vec` of structs, and you know why `String` indexing does not work like Python.

### Phase 0.7 — Modules and Paths
Why: the project grows fast and needs clean boundaries early.

- https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html
- https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html
- https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html
- https://doc.rust-lang.org/book/ch07-05-separating-modules-into-different-files.html

Done when: you understand `mod`, `pub`, `use`, `crate::`, and why `main.rs` stays thin.

### Phase 0.8 — Error handling with `Result`
Why: API calls, env vars, parsing, and subprocesses all fail. Orchestration code must not panic casually.

- https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html
- https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html

Done when: you can read `Result<T, E>` signatures and use `?`.

> **BUILD CHECKPOINT A.** A throwaway binary: define a fake `VerificationOutcome` enum and `TestEvidence` struct, put several in a `Vec`, loop and `match` each one, print a triage summary. Add one function returning `Result` that errors on an empty `Vec`, handled with `?`. Compiles, runs, written without copying a full example.

### Phase 0.9 — Serde
Why: typed LLM input and output shapes, rather than raw strings forever.

- https://serde.rs/derive.html
- optional: https://serde.rs/json.html and https://serde.rs/enum-representations.html

Done when: you understand `Serialize` and `Deserialize` and can say why typed payloads beat raw strings.

> **BUILD CHECKPOINT B.** Define a fake `CreatorResponse` (patch text, explanation, confidence). Serialize to JSON, print it. Deserialize hand-written JSON back into the struct. Then break the JSON deliberately and watch the error come through `Result` instead of a panic.

### Phase 0.10 — Async and Tokio
Why: Rig calls are async; you need working fluency, not mastery.

- https://doc.rust-lang.org/book/ch17-00-async-await.html
- https://rust-lang.github.io/async-book/01_getting_started/04_async_await_primer.html
- https://tokio.rs/tokio/tutorial
- https://rust-lang.github.io/async-book/part-guide/async-await.html

Done when: you understand `async fn`, `.await`, why a runtime is needed, and are not confused by `#[tokio::main] async fn main()`.

> **BUILD CHECKPOINT C.** A `#[tokio::main]` binary with one `async fn` that sleeps a second and returns a `String`. Await it twice, print both. Then say in one sentence what the runtime did.

### Phase 0.11 — Rig
Why: get the mental model before coding against it.

- https://github.com/0xPlaygrounds/rig
- https://docs.rig.rs
- https://docs.rs/rig-core/latest/rig_core/

Focus on the minimal call path: client setup, provider setup, one agent, one prompt, one response. Then the structured-output examples.

Done when: you can describe the smallest possible call flow, and how you want Creator and Attacker outputs represented in Rust.

## M0 BUILD — the spike (3 days, hard stop)

Dependency rules before writing anything:
- pin Rig exactly (pre-1.0): `rig-core = "=0.x.y"`, whatever version dependi shows today. The `=` matters.
- `tokio = { version = "1", features = ["full"] }`
- `serde = { version = "1", features = ["derive"] }` and `serde_json = "1"`
- do not upgrade any of these mid-spike

- [ ] Add `orchestrator/` module, keep `main.rs` thin
- [ ] `CreatorRequest` and `CreatorResponse` structs
- [ ] Read `ANTHROPIC_API_KEY` from env, fail with a clear message if missing
- [ ] **Spike milestone 1:** one typed Claude call through Rig, response printed
- [ ] **Spike milestone 2:** model returns JSON matching `CreatorResponse`, deserialized; malformed output surfaces as a handled `Result`, not a panic
- [ ] **Spike milestone 3:** two agents chained, Creator output feeding an Attacker-shaped second call
- [ ] Written verdict committed to the repo

### THE KILL SWITCH
Write the start date down at your first `cargo add`.

- the Rig portion gets **3 working days**
- NO-GO symptoms: examples that do not compile, docs that do not match the API, structured output that will not work, provider errors you cannot get past
- if NO-GO, switch to raw HTTP without guilt:
  - `reqwest = { version = "0.12", features = ["json"] }`
  - https://docs.rs/reqwest/latest/reqwest/
  - https://docs.claude.com/en/api/messages
- a spike that returns NO-GO is a successful spike. It answered the question cheaply.
- week-6-you does not get to renegotiate this timebox

Commit the verdict: `SPIKE RESULT: Rig GO` or `SPIKE RESULT: Rig NO-GO, switching to reqwest`.

## M0 EXIT TEST
> You can write a Rust program from scratch that calls Claude, gets typed JSON back, feeds it into a second call, and handles a malformed response without panicking. And the GO/NO-GO verdict is written down.

## M0 DO NOT BUILD YET
Anything resembling the real Creator or Attacker. Retries, config systems, sandboxes, verification.

---

# M1 — The Adversarial Core

**What it is:** the two agents that make Vanta adversarial, talking to each other properly. No execution yet. Patch in, intent-labelled tests out, everything logged.
**Target:** September 2026.

## M1 STUDY

### Traits and generics
Why: the sandbox trait and `Verifier` trait are Vanta's load-bearing abstractions, and M2 needs them.

- https://doc.rust-lang.org/book/ch10-01-syntax.html
- https://doc.rust-lang.org/book/ch10-02-traits.html

Done when: you can define a trait with one method, implement it for two structs, and explain `impl Trait` vs `dyn Trait` at surface level.

### Testing in Rust
Why: you are building a testing tool. Your own code needs tests.

- https://doc.rust-lang.org/book/ch11-01-writing-tests.html
- https://nexte.st/docs/running/

Done when: you can write a `#[test]` and run the suite with `cargo nextest run`.

### Prompt engineering for structured agents
Why: Creator and Attacker quality is mostly prompt quality.

- https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
- https://docs.claude.com/en/docs/build-with-claude/structured-outputs

Done when: you can write a system prompt that reliably returns valid JSON matching a schema.

### Re-read before building
- Spec section 4 (architecture) and section 5 (test intents, the three-phase system)

> **BUILD CHECKPOINT D.** Write a `TestIntent` enum and `GeneratedTest` struct, then a `#[test]` asserting that a `BugReproduction` test with a passing baseline is classified as a mismatch. You are writing the type system before the logic.

## M1 BUILD

- [ ] `orchestrator/` module structure, thin `main.rs`
- [ ] Simple config struct: model choice, API key loading, retry limits (a struct, not a config system)
- [ ] **Creator agent**: issue description + source file in, patch out
  - [ ] `CreatorRequest` (issue text, file contents, optional prior failure feedback)
  - [ ] `CreatorResponse` (patch text, explanation, files touched)
  - [ ] Verify-First prompting in the system prompt from day one
- [ ] **Attacker agent**: patch in, adversarial tests with declared intent out
  - [ ] `AttackerRequest` (patch, original source, issue text)
  - [ ] `GeneratedTest` (test code, name, `intent`, reason)
  - [ ] `TestIntent` enum: `BugReproduction`, `FeatureAcceptance`, `BehaviorPreservation`
- [ ] Session data model, because everything downstream depends on it
  - [ ] `session_id`, `patch_attempt_id`, `testing_iteration_id`, `test_id` on every record
  - [ ] Attempt log written to disk as JSON — this becomes M5's raw data
- [ ] One-shot pipeline: issue in, Creator patch, Attacker tests, all logged

## M1 EXIT TEST
> One command against a real Python file with a real bug produces a patch and a set of intent-labelled tests in a log file. Nothing has executed yet, and that is correct.

## M1 DO NOT BUILD YET
Running the tests. The retry loop. Sandboxing. Anything about EIR/ECR.

---

# M2 — Ground Truth

**What it is:** the part that makes Vanta more than a prompt chain. Real execution, isolated, with honest triage of what failures mean.
**Target:** September to October 2026. **This is the biggest milestone. Budget accordingly.**

> **FAST PATH, if October matters to you:** build the `SandboxBackend` trait with a plain subprocess backend first (run pytest in a temp directory with timeouts). Ship V1 on that. Slot Shuru in as a second backend during November. The architecture does not change, only the backend does. This is the single highest-leverage schedule decision available to you.

## M2 STUDY

### Process handling and timeouts
- https://doc.rust-lang.org/std/process/struct.Command.html
- https://docs.rs/tokio/latest/tokio/process/
- https://docs.rs/tokio/latest/tokio/time/fn.timeout.html

Done when: you can spawn a process, capture stdout/stderr/exit code, and kill it on a timeout.

### Structured test output
- pytest JSON report: https://pypi.org/project/pytest-json-report/
- cargo-nextest machine output (for Rust targets later): https://nexte.st/docs/machine-readable/

Done when: you can parse a pytest run into per-test results rather than scraping text.

### Sandboxing with Shuru (only when you reach the microVM backend)
- https://github.com/superhq-ai/shuru

Done when: you can start a microVM, run one command inside it, and destroy it from the CLI by hand, before writing any Rust around it.

### Filesystem and temp directories
- https://docs.rs/tempfile/latest/tempfile/
- https://doc.rust-lang.org/std/fs/

### Re-read before building
- Spec section 5 in full (BATE, four-way triage, Unknown handling) and section 7 (sandbox trait)

> **BUILD CHECKPOINT E.** A standalone binary that copies a directory to a temp location, runs `pytest --json-report` inside it with a 30-second timeout, parses the JSON, and prints per-test pass/fail. This is M2's engine in miniature.

## M2 BUILD

**Sandbox layer**
- [ ] `SandboxBackend` trait: `spawn`, `execute`, `destroy`
- [ ] `subprocess.rs` — plain subprocess backend (the fast path; do this first)
- [ ] `shuru.rs` — macOS microVM backend via Shuru
- [ ] Copy target repo in, run a command, capture stdout/stderr/exit code, destroy
- [ ] Timeouts and memory limits enforced at process level

**Phase 1: BATE**
- [ ] Run every generated test against the **unmodified** codebase first
- [ ] Compare the result to the declared `intent`:
  - `BugReproduction` must FAIL on original
  - `BehaviorPreservation` must PASS on original
  - `FeatureAcceptance` may fail or be unsupported
- [ ] Mismatch returns to the Attacker for revision, max 3 retries, then scrap and log the reason
- [ ] Record every baseline result; this is the reference point for all regression counting
- [ ] Log Attacker discard rate as a reliability metric for M5

**Phase 2: four-way Diagnostician**
- [ ] `FailureClass` enum: `Infrastructure`, `Static`, `Logic`, `Unknown`
- [ ] Infrastructure: exit codes, signals, timeouts, OOM, health-check failures. Reset sandbox and rerun. Never a model error.
- [ ] Static: run `ruff` and `mypy` **before** execution
- [ ] Logic: test ran, assertion failed. The only class that counts as a model error.
- [ ] Unknown: rerun in a fresh sandbox, capture logs, return to the Attacker to simplify, quarantine if it persists. Repository and benchmark tests are never quarantined; they escalate to "human review required".
- [ ] Structured JSON diagnostic payload back to Creator (not GCF; that is a V2 ablation)

**Evidence provenance**
- [ ] `EvidenceSource` enum: `ExistingRepositoryTest`, `BenchmarkOracle`, `AttackerGenerated`, `HumanProvided`, `FormalVerifier`, `IssueDescription`
- [ ] `TestEvidence` struct carrying test_source, intent, execution_source, result

## M2 EXIT TEST
> Feed in a patch. Every generated test gets baseline-validated against its intent, invalid ones get repaired or scrapped, valid ones run against the patched code in isolation, and every failure comes back tagged Infrastructure / Static / Logic / Unknown with a reason. Kill the network mid-run and it classifies as Infrastructure, not as a model error.

## M2 DO NOT BUILD YET
Linux/Firecracker backend. Formal verification. Graphiti. Live EIR/ECR.

---

# M3 — V1 Ships

**What it is:** closing the loop and pointing it at a real repository, unattended. The milestone that changes your CV, your README, and your odds.
**Target:** October to November 2026.

**Definition of done, verbatim from the README:**
> Given a real Python repo and a bug description, Vanta produces a PR containing a patch plus Attacker-generated tests that pass baseline validation, with the whole loop running unattended.

## M3 STUDY

### GitHub API from Rust
- https://docs.rs/octocrab/latest/octocrab/
- https://docs.github.com/en/rest/pulls/pulls
- https://docs.rs/git2/latest/git2/ (only if you need local git operations)

Done when: you can create a branch, commit a file, and open a PR from a Rust program against a scratch repo.

### CLI building
- https://docs.rs/clap/latest/clap/
- https://rust-cli.github.io/book/index.html

Done when: `vanta run --repo X --issue Y` parses cleanly with `--help` output you would not be embarrassed by.

### Re-read before building
- Spec section 8 (governor, fixed shared test set) and the verification-policy table in section 6

> **BUILD CHECKPOINT F.** Open a pull request on a throwaway repo from a Rust binary. Do this before wiring it into Vanta, so that when the loop finally works you are not debugging GitHub auth at the same time.

## M3 BUILD

- [ ] **Repair loop**: failure payload, Creator retry, re-run the shared test set, repeat
- [ ] **Fixed-N Governor** (N=4 default): stop after N attempts, log every transition
  - [ ] Fixed shared test set protocol: comparing attempt N to N+1 runs both on the same saved tests
  - [ ] New Attacker tests recorded separately, then folded into the next comparison
  - [ ] Incomplete transitions marked censored, never treated as complete
- [ ] **Gatekeeper**: deterministic approve/reject on required signals only
  - [ ] `VerificationPolicy` enum stubbed (`Required` / `BestEffort` / `Disabled`), default BestEffort
- [ ] **GitHub PR integration**: branch, commit patch and accepted tests, PR body summarising evidence
- [ ] **CLI**: `vanta run --repo ./target --issue ./bug.md`
- [ ] End-to-end unattended run on a real repository
- [ ] README milestone boxes ticked, status line updated
- [ ] Tag release `v0.1.0`

## M3 EXIT TEST
> One command. Walk away. Come back to a pull request on GitHub containing a patch and tests, opened by Vanta, with no human intervention in between.

## M3 SHIP CHECKLIST — do these before touching M4
- [ ] CV: swap "In Development" for shipped, add repo link
- [ ] LinkedIn: headline, Featured section, first post
- [ ] Build-in-public post on X with a demo GIF
- [ ] Approach a KAIST professor about thesis supervision, opening with V1 running
- [ ] Consider Show HN, Tuesday or Wednesday, US morning

Shipping without telling anyone wastes the milestone.

---

# M4 — V2 Components

**What it is:** the sophistication layer. Same architecture, better components. Everything here is optional relative to M5, and this is what gets cut when time runs out.
**Target:** November to December 2026.

## M4 STUDY

### Formal verification
- https://crosshair.readthedocs.io/
- https://docs.rs/z3/latest/z3/
- https://model-checking.github.io/kani/

Done when: you can run CrossHair by hand on a small Python function and read its counterexample output.

### The stability framework, re-read properly
- Liu and Meng, arXiv:2604.22273, section III before implementing the governor

Done when: you can derive `ECR/EIR > Acc/(1-Acc)` on paper and say what each term means in Vanta's setting.

### Memory and sandboxing (only if you reach them)
- https://github.com/getzep/graphiti
- https://github.com/firecracker-microvm/firecracker

## M4 BUILD — strict priority order, stop whenever M5 needs the time

**1. EIR/ECR instrumentation — non-negotiable, M5 depends on it**
- [ ] Compute EIR and ECR from attempt logs on the fixed shared set
- [ ] Per-transition table: which tests went pass to fail (EIR) and fail to pass (ECR)
- [ ] Exclude Infrastructure, Static, Unknown, and pre-existing failures
- [ ] Export runs to CSV

**2. Live Governor**
- [ ] Replace fixed-N with `ECR/EIR > Acc/(1-Acc)`
- [ ] Verify-First evaluated as an intervention, a hypothesis rather than an assumed benefit
- [ ] Persistent single-test failure raises "possible test or specification conflict, review required"

**3. Formal verification track**
- [ ] `Verifier` trait with `VerificationOutcome`: `VerifiedWithinBounds`, `Counterexample`, `Inconclusive`, `Unsupported`, `Timeout`, `ToolFailure`
- [ ] CrossHair backend for Python via subprocess
- [ ] Verifier runs in its own restricted environment, not on the host
- [ ] Wire `VerificationPolicy` into the Gatekeeper properly

**4. Institutional memory**
- [ ] Flat JSON incident store feeding historical failure modes into Attacker prompts
- [ ] Graphiti migration only if time genuinely allows

**5. Linux sandbox**
- [ ] `zeroboot.rs` Firecracker backend via CI or a cheap Linux VPS, since macOS cannot run KVM

**6. Optional ablation**
- [ ] GCF vs JSON diagnostic payloads

## M4 EXIT TEST
> Vanta reports live EIR/ECR for a run and halts on the stability condition rather than a counter. At minimum item 1 works, because M5 cannot happen without it.

---

# M5 — Research

**What it is:** the highest-value milestone in the project. What makes Vanta a research contribution rather than a tool, and what your December applications are built around.
**Target:** November to December 2026.

## M5 STUDY

### The benchmarks
- https://github.com/soarsmu/BugsInPy
- https://github.com/rjust/defects4j
- https://www.swebench.com/

Done when: you can check out one BugsInPy bug by hand, build its environment, and run its test suite. Do this for a single bug before automating anything; historical Python versions and dependencies are the nastiest part of this milestone.

### Related work you must cite
- Liu and Meng 2026, arXiv:2604.22273
- CyberCorrect, arXiv:2605.17305
- SWT-Bench, AutoCodeSherpa, PatchDiff
- the 2026 POMDP and Bayesian-control agent papers

### Submitting
- https://info.arxiv.org/help/submit/index.html

## M5 BUILD AND WRITE

- [ ] **BugsInPy harness**
  - [ ] Per-bug environment setup, automated
  - [ ] Batch run across a bug subset, results to CSV
- [ ] **Experiments across model tiers**
  - [ ] Frontier (Opus/Fable-class), mid (Sonnet-class), cheap (Haiku/4o-mini-class), local (Qwen-coder/DeepSeek-coder)
  - [ ] Per tier: fix rate, **regression rate**, EIR, ECR, governor firing rate, Attacker discard rate, static failure rate
- [ ] **Verify-First ablation**, measured rather than assumed
- [ ] Optional: Defects4J subset for cross-language generalisation
- [ ] Optional: SWE-bench-Lite with the regression-rate metric layered on
- [ ] Fill paper section V with real numbers, no placeholders
- [ ] Frame the contribution honestly: domain transfer plus measurement validity, not a new theorem
- [ ] Post to arXiv (cs.SE)
- [ ] Email the preprint personally to 5-10 cited researchers, including Liu and Meng
- [ ] Add the arXiv link to CV, LinkedIn, README, and the repo citation block

## M5 EXIT TEST
> A preprint on arXiv containing a results table you generated yourself, showing how EIR and ECR behave in adversarially verified code repair across at least three model tiers.

## If results are negative or null
Report them honestly and reframe as a measurement-methodology contribution. "External deterministic feedback dissolves the stability problem" is a real finding. Do not force a positive claim. The experiment you cannot lose is the one you actually run.

---

## Cut order when time runs short

Cut in this order:
1. GCF ablation
2. Graphiti migration (stay on flat JSON)
3. Defects4J and SWE-bench comparisons (BugsInPy alone is enough)
4. Firecracker/Linux backend (macOS or even plain subprocess is fine for a paper)
5. Formal verification track (the adversarial plus governor story stands alone)

Never cut:
- M3 shipping V1
- M4 item 1, EIR/ECR instrumentation
- M5 core experiments and the preprint

## Note-taking headings

Keep running notes under these, across the whole project:

- Ownership and borrowing
- Struct vs enum
- Collections
- Modules, paths, `pub`, `use`
- `Result` and error handling
- Serde
- Async and Tokio
- Traits
- Rig or reqwest observations
- What fought me today
- Questions I still have
