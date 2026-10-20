# BugsInPy harness — design

**Status:** design, revised after review. No harness code written, nothing committed.
**Branch:** `bugsinpy`
**Goal:** one command runs a committed list of BugsInPy bugs through several configurations and
produces ground-truth-labelled sessions. No hand-typed runs.

```
python3 scripts/bugsinpy_run.py prepare --split plans/bugsinpy_split.json
python3 scripts/bugsinpy_run.py run     --split plans/bugsinpy_split.json
python3 scripts/bugsinpy_run.py report  --split plans/bugsinpy_split.json
```

**Revision note.** This version folds in the review. The three *pilot-breakers* it fixes:

1. **Test files.** The bug-revealing test normally does **not** exist on the buggy commit — the fix
   adds it, and often updates existing tests too. Grading against the buggy commit's own tests makes
   `HIDDEN` come out empty and turns a correct fix into a phantom "native regression". Fixed by the
   standard **SWE-bench split** (§1.2–§1.4).
2. **Old Pythons do not build on the base image.** 3.6/3.7/3.8 all need OpenSSL 1.1.1; Debian trixie
   ships OpenSSL 3, so a naive build yields an interpreter with no working `ssl` and `pip` cannot
   install anything (§3.2).
3. **Context size.** Vanta puts the whole source concat in the prompt; `youtube-dl` is megabytes and
   would die at the runtime context precheck after a checkpoint was already baked. `prepare` now
   estimates and **rejects** over-budget bugs up front (§1.6).

---

## 0. What is being measured, and why

Vanta currently has **no external ground truth**. On `heretic` we can only say "the Governor
rejected" or "the Attacker found a signal" — never "the patch was actually wrong". BugsInPy gives
us bugs where:

- a **known-good** repair exists (the upstream fix commit), and
- a **known-bad** state exists (the buggy commit), and
- a **machine-checkable oracle** exists (the bug-revealing test passes only after the fix).

That converts every existing Vanta metric into a *labelled* one: approval precision, false
approvals, false rejects, and ECR/EIR with a known answer key. The harness's job is to produce
those labels, not to change how Vanta decides anything.

**Hard constraints carried from the request (restated so the doc is self-contained):**

| # | Constraint |
|---|---|
| C1 | New files where possible. The existing pipeline is touched only where the request forces it. |
| C2 | Do **not** change the pipeline's verdict logic or the ECR/EIR definitions. |
| C3 | **Nothing** third-party runs outside Shuru — including `prepare`'s suite runs. BugsInPy source *and* LLM-written tests are untrusted. |
| C4 | No commit until the design is reviewed. |

---

## 1. Loader

### 1.1 What BugsInPy gives us per bug

```
projects/<project>/bug.info          # key=value, quoted strings
projects/<project>/bugs/<N>/bug_patch.txt      # the developer fix, as a git diff
projects/<project>/bugs/<N>/requirements.txt   # extra deps for THIS bug (often empty)
projects/<project>/bugs/<N>/run_test.sh        # the bug-revealing test invocation
projects/<project>/project.info      # github_url=, status=, cause=
```

`bug.info` fields we depend on (all observed live):

```
python_version="3.8.1"
buggy_commit_id="6a1b2c..."          # NOTE: some files write `fixed_commit_id ="..."` with a space
fixed_commit_id ="9d8e7f..."         #   before `=`. The parser must tolerate whitespace around `=`.
test_file="tests/test_chinese.py"    # the file containing the bug-revealing test
pythonpath="tqdm/build/lib/"         # OPTIONAL — present on tqdm/cookiecutter, absent on PySnooper
```

The `name = "value"` whitespace quirk is real (`youtube-dl/1` has it). Parser rule: split on the
first `=`, `strip()` both sides, strip one layer of `"`.

### 1.2 Trees — the SWE-bench split

BugsInPy's own checkout copies the **fixed** version's test file into the buggy tree, because the
bug-revealing test is normally *added* by the fix. We do the same, but with the standard SWE-bench
split so that the grade is fair:

| Tree | Content | Seen by |
|---|---|---|
| `src_buggy/` | plain `git checkout <buggy_commit_id>` — buggy source, **buggy** tests, no fix | prepare only (pre-existing-failure baseline) |
| `src_fixed/` | plain `git checkout <fixed_commit_id>` — fixed source, **fixed** tests | prepare only (gate, `PASS_TO_PASS`) |
| **`vanta_tree/`** | buggy source **+ the fixed commit's test files**, minus `FAIL_TO_PASS` | **Vanta (`--repo`)** |
| `vanta_clean/` | byte-identical pristine copy of `vanta_tree/` | Vanta (`--original-tree`) — two-tree mode in [driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L332-L350) |
| `oracle_tree/` | buggy source **+ the fixed commit's test files**, hidden tests **intact** | the oracle (§5), and prepare's known-good check |

Rationale, one line each:

- **Why the fixed version's test files?** The fix frequently updates existing tests to match the new
  behaviour. Grading a correct fix against the *old* versions of those tests reports a native
  regression that is not the patch's fault — a false reject manufactured by the harness. Vanta's
  native suite must therefore be the **post-fix** test suite.
- **Why remove `FAIL_TO_PASS`?** Those are the oracle. Leaving them in would hand Vanta the answer.
- **Why keep `src_buggy/` too?** It is the only thing that tells us which failures are *pre-existing*
  (§1.3 step 6), which is what lets the fixed-tree gate be relaxed instead of dropped.

The checkout is done by the harness (plain `git`), **not** by `bugsinpy-checkout`, so we control the
working directory and avoid BugsInPy's venv assumptions. We pin the BugsInPy clone's own commit SHA
into the split, so the metadata we froze is reproducible.

### 1.3 Hidden oracle: `FAIL_TO_PASS` / `PASS_TO_PASS`

Do **not** trust `test_file` / `run_test.sh` as the definition. Derive it empirically:

1. Run the whole suite on `vanta_tree/` **before stripping** — i.e. buggy source + fixed tests →
   failure set `F_before`.
2. Run the whole suite on `src_fixed/` → failure set `F_after`.
3. **Every suite is run twice.** Any nodeid whose outcome differs between the two runs is flagged
   **flaky**, excluded from both sets, and recorded in `flaky_nodeids` in the prepared manifest. A
   flaky nodeid must never silently become the oracle.
4. `FAIL_TO_PASS = F_before \ F_after` (nodeids) — fails on buggy source + fixed tests, passes on
   fixed. **This is the hidden oracle.**
5. `PASS_TO_PASS` = everything that passes on `src_fixed/` — this is Vanta's native suite.
6. Sanity gates (prepare fails loudly if any trips):
   - `FAIL_TO_PASS` is non-empty → otherwise the bug is not reproduced at all;
   - every nodeid in `FAIL_TO_PASS` is inside `test_file` (or is what `run_test.sh` names) →
     otherwise our suite runner is not running what BugsInPy intends;
   - **`F_after \ F_buggy_pre = ∅`**, where `F_buggy_pre` = failures on the *plain* `src_buggy/`
     tree (buggy source + buggy tests). Reading: **the fixed tree has no failure that the buggy tree
     does not also have.** This replaces the previous "`F_fixed` must be empty" gate, which the
     review correctly called too strict. Pre-existing failures are **recorded**
     (`preexisting_failures`) and never charged to a patch. Note this is a *relaxation, not a
     removal*: a failure that is fixed-only is still a hard stop, because it means the environment is
     wrong.
   - `vanta_tree/`'s suite is green again after stripping (§1.5).
7. **Strip**: remove the `FAIL_TO_PASS` nodeids from `vanta_tree/` and `vanta_clean/`. If a test
   *file* is entirely hidden, delete the file; otherwise delete only the named functions
   (**AST-level, not regex**). Record `hidden_removed` and `hidden_files_deleted` in the prepared
   manifest.

### 1.4 What Vanta sees

`vanta_tree/` = buggy source + the **fixed** test suite minus the `FAIL_TO_PASS` tests: no fix, no
oracle. The `Repo` is loaded by [driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L152)
from `--repo` and the issue text from `--issue` — **no pipeline change needed** for either.

### 1.5 Consequence we must state explicitly (not a bug, a property)

After stripping, `vanta_tree/`'s native suite is green **by construction** — it is the SWE-bench
`PASS_TO_PASS` set, which passes on the fixed tree and therefore also passes on the buggy source with
those tests. Vanta's repair loop does not require a failing test to start — its first act is a
Creator call ([driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L474-L525)) — but with a
green suite the Creator has **no failing signal**, so in `baseline` (no Attacker) it is a one-shot
repair from the issue text with no iterative feedback, and its approvals are unverified by
construction. That is exactly the base rate we want to measure against `vanta`.

The prepared manifest records `native_green_on_vanta_tree: true/false` per bug so this is auditable
rather than assumed. If it is *not* green, the harness records the residual failure set and measures
`suite_regressions` relative to **that**, not relative to "zero failures".

The SWE-bench split also removes the trap the review named: because the native suite is the post-fix
suite, a correct fix can no longer be flagged as a native regression simply for matching the fix's
new behaviour.

### 1.6 Context-budget precheck — `prepare` rejects, it does not warn

Vanta puts the **whole source concat** into the prompt. `heretic` was 250 KB; `youtube-dl` is
megabytes (hundreds of extractor files). Without a precheck those bugs die inside `vanta_m3`'s own
context guard — *after* a checkpoint has been baked and a wave has been scheduled.

1. Build the source concat for `vanta_tree/` with Vanta's own file-selection and framing, then
   estimate tokens, and calibrate that estimate against the server (the repo's existing local
   estimate runs ~1.5× high versus `usage.prompt_tokens`; the harness records the calibration factor).
2. **Reject** a bug whose estimate — plus the issue text, plus the feedback and completion budget we
   cannot shrink — exceeds the window with **margin**. The margin is a documented constant (default:
   **25 % of `n_ctx` left free**), not a magic number.
3. `prepare` writes **every candidate's** size, not only the rejects, to two files:
   `runs/bugsinpy/<batch>/context_sizes.md` for reading and `context_report.json` for tooling. The
   JSON carries, per bug: `estimated_prompt_tokens`, `prompt_tokens_real` (**null** until a session
   has actually paid for the prompt, with `prompt_tokens_real_source` naming the session that filled
   it in), `budget_tokens`, `verdict`, `source_bytes` and `file_count`, plus the reasons for a
   rejection. `prepare` prints the same numbers as a table on stdout.
4. **`plans/bugsinpy_split.json` is an input and is never written by a stage.** Measurements live in
   the batch directory (`context_report.json`, `prepare.json`); the split only changes through a
   deliberate edit that is reviewed. A stage that rewrites the split turns a measurement into a
   decision while looking like it only recorded something, and the difference between "the plan says
   this" and "this run observed that" is what the two files exist to keep apart. `prepare` hashes the
   file before and after and fails if it moved; a test asserts no production module contains
   `set_derived(` or `.save(`.
5. **Calibrate before substituting.** The estimate is `ceil(source_bytes / 4)` plus a frame and runs
   high (measured ~1.5× on the heretic sessions: 84.5k estimated vs 55.3k real), so a swap chosen
   from the estimate alone is uncalibrated. Wave 1's real `usage.prompt_tokens` is the calibration;
   waves 2–3 are re-measured against it and any proposed swap shows **both** the estimate and the
   observed number.
6. **Narrowing files to fit is v2 work**, explicitly out of scope. A rejected bug is replaced from
   the reserve list (§8); it is not patched up.

---

## 2. Issue text that never leaks the fix or the hidden test

This is the highest-risk part of the design: a leaky issue text turns the whole pilot into
"copy the fix from the prompt".

### 2.1 Options considered

| # | Option | Leak risk | Cost | Fidelity |
|---|---|---|---|---|
| **A** | Fix-commit subject + body, **diff and code blocks stripped**, plus a deliberate behavioural paraphrase; frozen into the split | Low, *if* gated | Low (automated + review) | Medium |
| **B** | Curator writes one paragraph per bug by hand, stored in the split | Lowest | 10 × ~10 min manual | Highest |
| **C** | LLM paraphrases `bug_patch.txt` **with the diff withheld** (it sees only the file paths touched + the failing behaviour) | Medium — the model can guess from paths | Low | Medium |
| **D** | Derive from BugsInPy's `<project>-fail.txt` / `verify.sh` | — | — | **Not viable.** Those are *test-name lists* and a runner, not issue text. |
| **E** | Give the Creator the hidden test's traceback (real BugsInPy-style report) | **Total** — the assertion states the expected value | Low | Highest, and useless |

Option E is what most repair benchmarks do and it is precisely what the request forbids.

### 2.2 Recommendation: **A, with a mandatory leak gate**

Pipeline of `prepare`:

1. `git show -s --format=%B <fixed_commit_id>` → subject + body.
2. **Strip diff and code only**: any `diff --git` / `@@` / `+` / `-` diff line, any fenced or
   indented code block, any inline `code span`. **Nothing else.**
   *Corrected after review:* the earlier draft also stripped "any line containing a file path or a
   symbol found in `bug_patch.txt`". That rule deletes the name of the function being fixed — a name
   that **is present in the buggy tree** and that the Creator needs in order to work at all. It would
   have made many issues unsolvable. Symbol-level leakage is the **leak gate**'s job (step 4), which
   is strictly better because it fires only on tokens that exist **nowhere** in the sanitized tree.
3. Generate the behavioural sentence with the local model, prompted only with (a) the stripped
   message and (b) the *removed* tests' **failure kind** — e.g. "an assertion on the parsed output
   fails" — never the assertion's expression, expected value, or nodeid.
4. **Leak gate** (hard, blocks `run`). Rule: reject or strip any token that appears **only in the fix
   diff or the hidden test** — a token present in (`bug_patch.txt` added/removed lines ∪ `HIDDEN`
   test source) whose occurrence count in `vanta_tree/` is **zero**. That is the precise meaning of
   "this came from the answer": a name that exists nowhere in the code the Creator is allowed to
   read.
   - Covered token classes: function/method names, test names, identifiers, string literals, numeric
     literals introduced by the fix, file paths, and any substring of a `FAIL_TO_PASS` nodeid or of a
     hidden-test assertion line.
   - `vanta_tree/`'s token set is collected once per bug, so the gate is a set lookup, not a scan.
   - Normalisation: case-fold, `_` ≡ `-`, strip `self.` prefixes and quotes.
   - **Stripping is logged**: every removed token goes to
     `runs/bugsinpy/<batch>/leak_gate/<bug>.json` as
     `{token, class, reason, occurrences_in_vanta_tree, action: "rejected"|"stripped"}`.
   - The gate **fails closed**: a *rejected* text stops `prepare`; a *stripped* text continues with
     the edit recorded and surfaced in the review dump below.
   - This is a **structural assertion** — it fails when the evidence for a clean text is absent,
     rather than depending on a reviewer noticing something.
5. Freeze the accepted text into `plans/bugsinpy_split.json` with its `sha256`. `run` recomputes the
   hash of the issue file it hands to Vanta and refuses to start on a mismatch.
6. **Pilot review dump.** `prepare` writes every generated issue text — pre-gate, post-gate, plus the
   removed-token list — to `runs/bugsinpy/<batch>/issue_texts/`, concatenated into a single
   `issue_texts_all.md`. `run` refuses to start until that file exists and its hash is recorded in
   the manifest.
   - **`NEEDS DECISION` section.** Any bug whose stripped commit message yields no actionable
     behavioural statement (a one-word subject like "fix bug", a message that was entirely a code
     block, a message the leak gate emptied) is listed at the **top** of `issue_texts_all.md` under a
     `NEEDS DECISION` heading, so you can supply a replacement text or drop the bug. The harness does
     not guess and does not silently ship a useless prompt.

A human read-through of the 10 frozen texts is a **one-time ~15 min** review before the split is
committed — that is the audit, and it costs nothing at run time.

### 2.3 Honest limitation

Option A's text is *derived from* the fix commit, so it is not independent of the fix. It cannot be
"real bug report" evidence in a paper. It is, however, the only option that is (a) automatable over
a committed list and (b) leak-gated. Reported as such.

---

## 3. Environments in Shuru

### 3.1 How our checkpoints actually work (investigated, live)

From [shuru.rs](file:///Users/kyu-yi/vanta/src/sandbox/shuru.rs) and a live `shuru checkpoint list`:

- `vanta-python` — the base, 1.5 GB.
- `vanta-deps-<fnv1a-hex>` — per-repo dependency layers, built by
  [`ensure_repo_checkpoint`](file:///Users/kyu-yi/vanta/src/sandbox/shuru.rs#L280) from
  [`deps::DependencyPlan`](file:///Users/kyu-yi/vanta/src/sandbox/deps.rs).
- Checkpoints are **APFS copy-on-write clones**, immutable, and **stackable** (`--from`).
- A checkpoint build is the only step needing `--allow-net`; a `shuru run --from <cp>` boots with
  **no network device at all**. This is what satisfies C3: deps are baked once, runs are offline.
- `shuru run --mount H:G` requires `H` to be **inside the CWD** (the
  [`TempDir::new_in(current_dir())`](file:///Users/kyu-yi/vanta/src/sandbox/shuru.rs) workaround
  exists for this reason). The harness must therefore keep all bug trees under the Vanta working
  directory.
- Disk size must be ≥ the base image's; `shuru checkpoint create … --disk-size N`.

### 3.2 BLOCKER 1 — the old Pythons do not build on the base image

**The base checkpoint is Debian GNU/Linux 13 (trixie) with Python 3.13.5 only** (verified by booting
`vanta-python`: `PRETTY_NAME="Debian GNU/Linux 13 (trixie)"`, `Python 3.13.5`, `/usr/bin/python3.13`;
no other interpreter, no `pyenv`).

BugsInPy demands **3.6.9 / 3.7.0 / 3.8.1** for the bugs inspected. Two problems compound:

- The interpreters do not exist and must be built (network + `build-essential` + `zlib1g-dev` +
  `libffi-dev` at bake time).
- **Debian trixie ships OpenSSL 3, and CPython 3.6/3.7/3.8 require OpenSSL 1.1.1.** A naive build
  succeeds at the interpreter and fails at `_ssl`: you get a Python that runs, has no `ssl` module,
  and therefore cannot `pip install` anything. The checkpoint bake dies at its first dependency, and
  it dies on *every* wave — including wave 1.
- Even with a working interpreter, several of these projects (2018–2020 era) will not *install* on
  3.13, so the version is not optional.

**Design — per version:**

- **Python 3.8 → standalone build.** `uv python install 3.8` (python-build-standalone) ships its own
  OpenSSL and needs no compiler. Cheapest and most robust; this is why py3.8 goes first (§8).
- **Python 3.6 / 3.7 → OpenSSL 1.1.1 inside the checkpoint.** Two options, pick at bake time and
  record which in the checkpoint notes:
  1. **Build OpenSSL 1.1.1 from source** into `/opt/openssl-1.1.1`, then build CPython with
     `--with-openssl=/opt/openssl-1.1.1` (plus `LDFLAGS=-Wl,-rpath,...`). Self-contained, one distro,
     recommended.
  2. **An older Debian base** (`bullseye`, which still ships 1.1.1) via `debootstrap` inside the
     microVM, building the interpreter in that rootfs. More moving parts, but avoids compiling
     OpenSSL.

```
vanta-bugsinpy-py3.6   ← standalone or source-built, with working ssl
vanta-bugsinpy-py3.7
vanta-bugsinpy-py3.8
```

The layer name is keyed on the **minor** version. `bug.info` records patch levels (`3.8.1` on
PySnooper, `3.8.3` on black); keying on those would bake two identical interpreters under two names.
All bugs on one minor share a layer, and `pytest`/`pytest-json-report` pins must therefore agree
within a minor version — `bake.pins_for_python` fails the bake if they do not, because a shared layer
cannot carry two pins.

then a thin per-bug layer on top:

```
shuru checkpoint create vanta-bi-<project>-<N> --from vanta-bugsinpy-py3.<v> --allow-net \
  -- sh -c 'set -e; pip install -r bug_requirements.txt; pip install <setup.sh deps>; \
            pip freeze > /opt/vanta-deps-freeze.txt; touch /opt/vanta-deps-ok'
```

This mirrors the existing `vanta-deps-<fingerprint>` + `/opt/vanta-deps-ok` marker + freeze-file
pattern exactly, so nothing new has to be invented for verification.

**What the per-bug layer installs, and what it deliberately does not.** BugsInPy's own install step
is `setup.sh` (`python setup.py install`, plus extra `pip install`s), and `requirements.txt` is
frequently empty — 0 bytes on all three PySnooper bugs, while `python_toolbox` and `decorator` exist
only in `setup.sh`. So the layer takes its specs from **both**, reading the `pip install …` lines of
`setup.sh` (measured, not guessed) and the whole of `requirements.txt`. The project's own install
line is dropped, for two reasons:

1. **A baked copy would be graded instead of the tree.** Every suite runs `python -m pytest` with its
   cwd set to the tree root, so `import <package>` resolves to the mounted tree — the copy the run is
   actually being graded on. Installing the project into the layer's `site-packages` puts a frozen
   copy in front of nothing useful and risks a run that reports `hidden_pass` while the patch it was
   meant to test never took effect.
2. **An editable install cannot work.** `pip install -e .` bakes the *bake-time* path, and each run
   gets its own tree path; the layer would point at a directory that no longer exists.

Projects that genuinely need building (`pythonpath="<pkg>/build/lib/"` on tqdm and cookiecutter) are
built **per tree, at prepare time, inside Shuru** — the same place, and for the same reason, that
their suites run.

**Mandatory bake-time gate (part of the wave-1 gate, §8.1).** For each version checkpoint, assert
*inside the checkpoint*:

```
python -c "import ssl, sys; print(sys.version, ssl.OPENSSL_VERSION)"   # must not fail
python -c "import ssl; ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)"        # functional, not a string
python -m pip install "pytest==<pin>" "pytest-json-report==<pin>"      # must reach PyPI and install
python -m pytest --version                                             # pinned version, must import
python -c "import pytest_jsonreport"                                   # the plugin, not just the wheel
```

The `ssl` check is functional rather than a version-string match on purpose: an interpreter whose
`_ssl` failed to build can still report a version, and the failure that matters — `pip` cannot
install anything — only shows up when the module is *used*. Standalone builds report OpenSSL 3.x and
are fine; a source-built 3.6/3.7 reports 1.1.1x. Both satisfy the gate, so the gate asserts the
property (works) rather than the version (which would fail the very build §3.2 recommends).

**Version pins.** `pytest` and `pytest-json-report` are pinned to versions that support each
interpreter — 3.6 caps at pytest 6.x, 3.7 at 7.4.x, 3.8 at 8.0.x — written into the checkpoint build
and recorded per bug in the split (`pytest_version`, `json_report_version`) so a report can be read
back years later.

### 3.3 BLOCKER 2 — dependency detection under-provisions

[`deps::detect()`](file:///Users/kyu-yi/vanta/src/sandbox/deps.rs) reads `requirements.txt` and
`pyproject.toml`. Observed reality:

- `youtube-dl/1/requirements.txt` is **0 bytes**; youtube-dl declares deps in `setup.py`.
- tqdm / cookiecutter use `pythonpath="<pkg>/build/lib/"` — the project must be **built**
  (`python setup.py build`) before its own tests can import it. BugsInPy's `bugsinpy-compile` does
  exactly this.

So `deps::detect` alone will under-provision. Two mitigations, both already present in our code:

1. **Prepare-time install**: the per-bug checkpoint build runs the project's own
   `setup.py` / `pip install -e .` and its `requirements.txt` **verbatim**, so the authoritative dep
   list comes from the project, not from our detector.
2. **Self-heal**: [`parse_missing_modules`](file:///Users/kyu-yi/vanta/src/sandbox/deps.rs) +
   [`ensure_derived_checkpoint`](file:///Users/kyu-yi/vanta/src/sandbox/shuru.rs) is the fallback if
   a run still hits `ModuleNotFoundError`.

**Blocker to accept up front:** `--allow-net` is required while building every checkpoint. If the
sandbox has no network at bake time, `prepare` cannot succeed. `network_alive()` already exists to
probe this; `prepare` must fail fast with the reason rather than half-build.

### 3.4 BLOCKER 3 — pytest vs unittest, and the report format

Vanta's harness parses **pytest `--json-report`** output
([`parse_pytest_report`](file:///Users/kyu-yi/vanta/src/pipeline/harness.rs#L739)). BugsInPy's older
projects run `python -m unittest` (e.g. `youtube-dl/1/run_test.sh` →
`python -m unittest -q test.test_utils.TestUtil.test_match_str`).

**Design:** normalise every bug to a pytest command, since pytest collects `unittest.TestCase`
subclasses. `prepare` validates collectability with `pytest --collect-only -q` inside the checkpoint
and **rejects the bug from the split** if it does not collect. The resolved `test_cmd`, plus the
pinned `pytest` / `pytest-json-report` versions (§3.2), are stored per bug in the split.
`run_test.sh` is still read, but used as the *cross-check* on which nodeid is the hidden test
(§1.3), not as the executor.

---

## 4. Configurations

Three configs, expressed as a small struct the runner passes to `vanta_m3`. Requested behaviour:

| Config | Attacker | Creator feedback | Status |
|---|---|---|---|
| `baseline` | **off** | native + static only | needs a new switch (below) |
| `vanta` | on | native + static + Attacker-generated | current pipeline, unchanged |
| `vanta_planner` | on | + planner | **placeholder — not implemented** |

### 4.1 Gaps found in our own code

1. **No Attacker-skip.** `driver.rs` has no flag or env to skip the Attacker. Also no way to
   restrict the Creator's feedback to native/static only — the feedback string is built inline.
2. **No seed plumbing anywhere.** [`LlmConfig`](file:///Users/kyu-yi/vanta/src/config/mod.rs#L62-L63)
   has `base_url, model_id, api_key, idle_timeout_secs, max_call_secs, retry_attempts,
   context_window_tokens` — **no `seed`**. "Seeds as a parameter" currently has nowhere to go.
3. **`vanta_m3` has no `--seed`, `--no-attacker`, or `--benchmark-instance` flag.** Full flag list:
   `--repo`, `--issue`, `--max-attempts`, `--model`, `--open-pr`, `--patch`, `--test-cmd`,
   `--original-tree`, `--rerun-signals`, `--help`.
4. **`final.patch` is only written on approval.** See §5.1 — this is a prerequisite fix.
5. **`benchmark_instance` / `benchmark_label` exist on `SessionRecord`**
   ([session/mod.rs](file:///Users/kyu-yi/vanta/src/session/mod.rs#L123)) but `driver.rs` writes
   `benchmark_instance: None, benchmark_label: None` at **every** construction site (~18 of them),
   including the `run.summary` record. So the M5 labelling hooks are present and unused — we do not
   need to add a field, we need to *populate* one at the `run.summary` site.

### 4.2 Proposed minimal-change mechanism

All three switches are **env-gated and additive**, so C2 holds (no verdict logic changes; the
defaults reproduce today's behaviour byte-for-byte):

| Env | Default | Effect |
|---|---|---|
| `VANTA_ATTACKER=on\|off` | `on` | `off` skips the Attacker phase and restricts Creator feedback to native/static |
| `VANTA_PLANNER=on\|off` | `off` | **placeholder**; `on` returns a clear "not implemented" `InfraFailure` so the config is addressable but cannot silently pass |
| `VANTA_LLM_SEED=<u64>` | unset | adds `seed` to the chat completion body; recorded in the manifest |

`vanta_planner` deliberately **exits as `InfraFailure`** rather than producing a fake result. A
config that "passes" because it does nothing is exactly the failure mode this project exists to
catch.

### 4.3 Sampling — corrected: the harness PINS `temperature` / `max_tokens` per call

**Corrected after review, twice.** The first draft's manifest example showed
`temperature: 0.2, top_p: 0.95, max_tokens: 4096`. The second draft over-corrected to "the harness
sets no `temperature` / `top_p` / `max_tokens` at all, and the record shows the server's effective
values." **That was factually wrong about this codebase** and is retracted here: the pipeline has
always pinned `temperature` and `max_tokens` **per call site**.

**What the code actually does.** Every M3 agent passes its own sampling through `extra_fields`, and
[post_chat](file:///Users/kyu-yi/vanta/src/llm/mod.rs#L686-L691) merges those **after** the body
defaults — so the agent's value WINS. `post_chat`'s own `max_tokens: 16384` / `temperature: 0.2` are
the M0 spike's values and are dead for every M3 caller (live only for `bootstrap`). The real schedule:

| Call site | `temperature` | `max_tokens` |
|---|---|---|
| `creator` | 0.1 | 32768 |
| `attacker_generate` | 0.2 | 32768 |
| `attacker_revise` | 0.3 | 32768 |
| `attacker_simplify` | 0.3 | 32768 |
| `intent_audit` | 0.0 | 2048 |

These pins are **part of v0.2's behaviour** — the sampling the baseline was collected under. They are
therefore **not configurable**, no env var may override them, and the harness must never *add* a cap
of its own. `top_p` is the one parameter **no call site pins**, so every request inherits whatever the
server is configured with and the server's answer is the only truthful source for it.

**Decided:**

1. **Seed: implemented** — `VANTA_LLM_SEED`, `LlmConfig` gains `seed: Option<u64>`, added to the
   chat-completion body **only when set** (unset adds no key at all — not `null`, not `0`).
2. **No new sampling knobs.** `LlmConfig` does not gain `temperature` / `top_p` / `max_tokens`.
3. **Record the real per-call schedule**, not a run-level scalar. The pinned table lives in
   `llm::SAMPLING_SCHEDULE` and is written to `run.summary` as one pair per site
   (`temperature_creator=0.1 max_tokens_creator=32768 … max_tokens_intent_audit=2048`), with the
   **server's** `top_p` from `/props` beside them (`top_p=…`, or `unset` when the server publishes
   nothing). A single run-level `temperature` would be a fabricated claim — one review run samples at
   0.0, 0.2 *and* 0.3. Recording the server's own `temperature` / `n_predict` would be worse: numbers
   the model never sampled with.
   **Keys are underscored, not dotted** (`temperature_creator`, not `temperature.creator`):
   `rerun.rs::summary_field` and `scripts/metrics.py::summary_field` both accept a key only when it
   matches `[a-z_][a-z0-9_]*`, so a dot would not terminate the value and *both* consumers would
   silently read the wrong string.
4. **A test asserts the record is real.**
   `tests/m3_session_log.rs::the_recorded_sampling_schedule_matches_the_request_bodies` compares the
   `run.summary` schedule against the `temperature` / `max_tokens` actually present in the bodies the
   stub server received.

**Honest limitation:** llama.cpp determinism under a fixed seed is **not guaranteed** — the prompt
cache, continuous batching, and non-deterministic float reduction order across threads all perturb
sampling. The seed is stored as a *reproducibility hint and a repeat key*, and the doc must **never
claim bit-exact reruns**.

### 4.4 Provenance (decided)

An env var alone is not provenance: a session directory copied out of the batch must be
self-describing. Every run records **config, mode, seed, model id, the per-call sampling schedule
plus the server's `top_p`, pinned pytest versions, and the Vanta git commit** in two places:

1. **`run.summary`** — the existing free-form `summary` string gains lowercase keys
   (`config=… mode=… seed=… model=… top_p=… commit=… dirty=…` plus one
   `temperature_<site>=…` / `max_tokens_<site>=…` pair per call site in §4.3), satisfying the repo's
   lowercase-only summary rule, **and** the two already-present-but-always-`None` fields are
   populated: `benchmark_instance = <config>` and `benchmark_label = <oracle label>`. This is one
   construction site (the `run.summary` record), not a struct change — `SessionRecord` keeps its
   shape ([session/mod.rs](file:///Users/kyu-yi/vanta/src/session/mod.rs#L123)).
   `benchmark_label` is the **oracle** label (§5.3), and the oracle runs from the harness *after*
   `vanta_m3` exits, so a run can settle only the one label its own verdict already fixes:
   `InfraFailure` → `infra_failure`. Every other row needs `patch_good` and stays `None` for the
   manifest to fill — guessing from the verdict would write `true_approval` for a patch the oracle
   may well call a `false_approval`.
2. **The batch manifest** (§7.3) — the same fields, structured.

`metrics.py` **groups by all of them** (§9.1), so any table can be cut by config × seed × model
without re-running anything. `commit` / `dirty` come from the existing build-time
`VANTA_BUILD_COMMIT` / `VANTA_BUILD_DIRTY` provenance, not from a runtime `git` call.

[VANTA-WHY] The `top_p` probe is the **only** HTTP call in the run that carries a deadline (5 s). It
fires *after* the verdict, so an unreachable or wedged `/props` must not be able to stop the session
from writing its own `run.summary` — the row that records the verdict. On timeout the key is recorded
as `unset`, which is the truth.

---

## 5. Oracle

### 5.1 Prerequisite — build `final.patch` for **every** terminal verdict

`final.patch` is the diff the oracle consumes, and **it does not exist where the oracle needs it**:
[driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L1811-L1841) calls `write_final_patch`
**only inside the `GatekeeperVerdict::Approved` branch**. Meanwhile `final_patched_files` is
populated on `AllTestsPassed`, `Rejected`, *and* `MaxAttemptsReached`
([driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L1634-L1653)). So a `Rejected` or
`MaxAttemptsReached` run — precisely the runs with the most ground-truth value — leaves **no patch
on disk at all**.

**Fix first, before any oracle work:**

- Move the `write_final_patch` call out of the Approved-only branch so it runs for **every terminal
  decision** where `final_patched_files` is `Some`. Destination stays
  `<session>/debug/final.patch`.
- `final.patch` is the diff of **non-test and test hunks** against the original tree — a faithful
  record of what the Creator (or the reviewed candidate) changed. Keeping the test hunks in the
  artifact matters for audit; the oracle filters them out when applying (§5.2).
- The Approved-only print / PR block is unchanged, so behaviour on today's path is byte-identical.

### 5.2 Per run, after `vanta_m3` exits

1. Fresh copy of `oracle_tree/` (buggy source + **fixed** tests, hidden tests intact) inside Shuru.
2. Apply **only the non-test hunks** of `final.patch` — `git apply` with the test-file hunks filtered
   out by path, **no fuzzy fallback** (matching the existing strict-application rule for review mode).
   Record `apply_ok`.
   - **The test files are never taken from the patch.** The oracle always uses the fixed version's
     test files as-is, so a Creator patch that edits the test file cannot tamper with the hidden
     test — it is simply not applied there.
   - `apply_ok = false` ⇒ outcome `unobservable`, recorded as such and never counted as a failure.
     This mirrors the M5 censoring rule (partially observed transitions are censored with null IDs)
     rather than inventing a new convention.
3. Run the **hidden tests** (`FAIL_TO_PASS`) → `hidden_pass` (bool).
4. Run the **native suite** (`PASS_TO_PASS`) → failure set `F_patched`.
   `suite_regressions = F_patched \ F_buggy_baseline` (set difference against the prepare-time
   baseline from §1.3/§1.5, so pre-existing failures are not charged to the patch).
5. `patch_good = hidden_pass AND suite_regressions == 0`.
6. Write `oracle.json` per session:
   `{apply_ok, hidden_pass, hidden_nodeids, suite_regressions, suite_regressions_count, patch_good,
   oracle_secs}`.

### 5.3 Labels — updated for `patch_good`

The label is a pure function of (Vanta verdict, oracle). **A patch is good only if it passes the
hidden test *and* regresses nothing:**

```
patch_good  ≡  hidden_pass  ∧  suite_regressions == 0
```

| Vanta verdict | `patch_good` | Label | Note |
|---|---|---|---|
| `AllTestsPassed` (approved) | `true` | `true_approval` | |
| `AllTestsPassed` (approved) | `false` | **`false_approval`** | hidden fail **or** broke other tests |
| `Rejected` | `true` | **`false_reject`** | a real fix, refused |
| `Rejected` | `false` | `true_reject` | correctly refused |
| `MaxAttemptsReached` | `true` | `unresolved_but_fixed` | |
| `MaxAttemptsReached` | `false` | `unresolved` | |
| any, `apply_ok = false` | — | `unobservable` | censored, excluded from rates |
| `InfraFailure` | — | `infra_failure` | not a verdict; metrics bucket only |

*Corrected after review:* previously a patch that passes the hidden test but breaks other tests was
labelled a `false_reject` when Vanta refused it — i.e. Vanta was penalised for being right. Under
`patch_good` that case is a **`true_reject`**, and the mirrored case (Vanta approves a patch that
passes the hidden test but regresses the suite) is a **`false_approval`**. The `suite_regressions`
column is therefore a first-class part of the label, not a footnote.

### 5.4 Review-mode candidate sets

Review mode is the sharper instrument, because both candidates have a known label:

- **known-good**: the developer fix. Apply the **non-test hunks** of
  `projects/<p>/bugs/<N>/bug_patch.txt` to `oracle_tree/` → `hidden_pass = true` by construction,
  asserted at prepare. Then run `vanta_m3 --patch <fix.patch>` against `vanta_tree/`. Any `Rejected`
  verdict here is a **false reject** — Vanta refused a patch that the ground truth says is correct.
- **known-bad**: the **no-op / whitespace-only** floor case, plus up to **2** failed Creator patches
  pooled from *both* configs' repair runs. A pooled patch is a reviewable known-bad candidate when the
  oracle says it **applied cleanly and is not `patch_good`** — `apply_ok = true` and
  (`hidden_pass = false` or `suite_regressions > 0`). `InfraFailure` runs and patches with
  `apply_ok = false` are **skipped**: a patch that never applied is not a patch, so reviewing it would
  measure nothing. If a bug produced no failed patches, its set is just {dev fix, no-op}. Then
  `--patch` against `vanta_tree/`. Any `AllTestsPassed` here is a **false approval**. Candidates are
  **deduplicated by their *normalized* diff** — index lines, `a/`/`b/` prefixes and trailing
  whitespace stripped, so two patches that differ only in those are one candidate, run once — and each
  records its source (`dev` / `noop` / `repair:<config>:<session>`) and its oracle label in the
  manifest — see §7.1 and §7.3.

Both candidate kinds are applied with their **test hunks dropped**, same rule as the oracle, so a
candidate cannot smuggle a change into the test file.

These feed `metrics.py`'s existing `--labels` mechanism (`{session_id: {"known_bad": bool}}`) — the
harness generates this file from the `candidate.oracle` label recorded in the manifest (§7.3), so no
oracle is re-run to score a review row. Note the existing project rule: a review-mode `Rejected`
counts as a
true catch **only if it discriminates** (the signal must rerun as `Failed→Passed`); otherwise it is
labelled `spurious_reject`. The harness must therefore also record the rerun-signal outcome, which
means retaining the graded tree or its reconstruction (§6).

---

## 6. Disk

### 6.1 The conflict

[`write_tree_snapshot`](file:///Users/kyu-yi/vanta/src/pipeline/harness.rs#L670-L710) writes a
**full copy of every file** to `debug/attempt_N_{patched,graded}_tree/`, plus a sibling
`<name>_manifest.json` holding only `{"paths": [...], "file_count": N}`. On a repo the size of
`black`, ×attempts×runs×configs, this dominates the disk budget. The request asks for **a manifest
plus a diff against the base, not a full copy**.

### 6.2 Constraint we cannot ignore

`--rerun-signals` reads the graded tree back as a directory
([rerun.rs](file:///Users/kyu-yi/vanta/src/pipeline/rerun.rs#L1014-L1024)):
`debug/attempt_N_graded_tree` → `BadTreeSource::GradedSnapshot`. Replacing the directory with a
diff **breaks `--rerun-signals`** for any session it was used on, and §5.4 needs rerun signals for
the discriminating-check. So this is not a free change.

### 6.3 Decided: harness-side compaction, with a rerun-safety rule

The harness is the only consumer that cares about pilot-scale disk. So:

1. Leave `write_tree_snapshot` and `rerun.rs` **untouched** (C2). The driver keeps writing full
   snapshots, exactly as today.
2. **Compaction must not break `--rerun-signals`.** The rule, decided: keep the full graded snapshot
   for any session with a **rejecting signal** until its rerun has been written — *or* make the rerun
   rebuild the tree from manifest + diff. Concretely, a session is compactable only when one of these
   holds:
   - it has no rejecting signal (nothing to rerun), **or**
   - the rerun has already run and its result is recorded in the session (the
     `signals=… discriminating=… undecidable=…` line), **or**
   - the rerun rebuilt the tree from base + diff and produced an identical result.
   The middle case is the "keep it" branch; the last is the "rebuild" branch — the rebuild is what
   lets us compact aggressively, and it is also how equivalence gets proven.
3. After the oracle, the harness compacts an eligible session: write
   `attempt_N_graded_tree.diff` (unified diff vs. `vanta_clean/`, pinned by commit SHA) +
   `attempt_N_graded_tree_manifest.json` (`{path: blake3}`), then delete
   `attempt_N_{patched,graded}_tree/`.
4. **Test, either way** (decided): a harness test asserts that (a) a session with a rejecting signal
   is *not* compacted before its rerun, and (b) a compacted session replays to a byte-identical tree
   from base + diff — i.e. `read_tree_concat`
   ([rerun.rs](file:///Users/kyu-yi/vanta/src/pipeline/rerun.rs#L1239)) over the reconstructed tree
   equals the recorded concat. Fails when the evidence for replayability is absent, rather than
   trusting the implementation.

Every pilot bug's base tree is small and pinned by commit SHA, so "base + diff" is complete and far
smaller than a full copy. The alternative — an env-gated `VANTA_SNAPSHOT_MODE=diff` inside the
driver — is cleaner but is a pipeline change, and this design prefers new files.

---

## 7. Batch runner

### 7.1 Interface

```
python3 scripts/bugsinpy_run.py prepare --split plans/bugsinpy_split.json   # checkout, split, strip, gate, bake
python3 scripts/bugsinpy_run.py run     --split plans/bugsinpy_split.json   # the one command
python3 scripts/bugsinpy_run.py report  --split plans/bugsinpy_split.json   # metrics per config
```

`run` = prepare-or-resume → for each wave → dispatch the four job kinds in the order below → oracle →
manifest append → compaction.

**Ordering (decided).** Repair passes run **first**, review passes **last**, so the failed Creator
patches a review needs already exist when the first review starts (§5.4). Within a wave:

1. **baseline repair** — one job per (bug, seed).
2. **vanta repair** — one job per (bug, seed).
3. **baseline review** (`--patch`, `VANTA_ATTACKER=off`) — **no LLM calls at all**, so it is the fast
   plumbing check: it proves the candidate set, the strict `--patch` apply, the oracle and the manifest
   work before a token is spent on the vanta review. It also establishes the plumbing-level floor: the
   developer fix is `Approved`, and a known-bad patch passes the sanitized suite (degenerate, but it is
   the baseline's honest ceiling).
4. **vanta review** (`--patch`) — the expensive LLM attack on the same candidate set.

**Review candidate set (decided, symmetric).** Once both repair passes for a bug are done, the runner
pools the failed Creator patches from *both* configs' repair runs. A pooled patch is a candidate when
the oracle says it **applied cleanly and is not `patch_good`** — `apply_ok = true` and
(`hidden_pass = false` or `suite_regressions > 0`); `InfraFailure` runs and `apply_ok = false` patches
are **skipped**, because a patch that never applied cannot be reviewed. The set is:

- the **developer fix** (known-good by construction, §5.4), and
- the **no-op / whitespace floor** (always present), and
- **up to 2** of the pooled failed patches, after dedup.

If a bug produced no failed patches, its set is just {dev fix, no-op}; the manifest records that.
**Baseline and vanta review exactly the same set at the same seed**, so every review row is a **paired
comparison** — the only thing that differs between the two rows is the config.

*Dedup key.* Two failed patches are the same candidate when their **normalized** diffs are equal: the
diff text with the `index` lines, the `a/` / `b/` prefixes and trailing whitespace stripped. A patch
that differs only in an index line or trailing space is one candidate, run once. When more than 2
distinct failed patches survive dedup, the runner keeps the **2 with the smallest `patch_sha256`** — a
deterministic, content-based choice, so the candidate set is reproducible from the manifest alone.

Each candidate's **source** and **oracle label** are recorded in the manifest — `dev`, `noop`, or
`repair:<config>:<session>`, plus `{apply_ok, hidden_pass, suite_regressions, patch_good}` (§7.3) — so
`metrics.py` can score review rows without re-running the oracle.

Wave 1 is a **go/no-go gate** (§8.2): if the checkpoint build, the oracle or the leak gate
misbehaves there, `run` stops and reports instead of building wave 2.

### 7.2 Split file — `plans/bugsinpy_split.json`

Committed **before any run**; `plans/` is not gitignored (verified), so this tracks as intended.
It is both the plan and the frozen record of the experimental design, and it is **hand-authored**:
no stage rewrites it (§1.6.4), so every value in it is a decision that was reviewed. Measurements a
run makes — sizes, verdicts, `FAIL_TO_PASS`, flakiness — live under `runs/bugsinpy/<batch>/` and only
enter this file through a deliberate edit:

```json
{
  "schema_version": 1,
  "created_utc": "2026-10-19T00:00:00Z",
  "bugsinpy_commit": "<sha>",
  "configs": ["baseline", "vanta"],
  "seeds": [0],
  "review_mode": true,
  "sampling": {
    "policy": "pinned_per_call_plus_server_top_p",
    "note": "temperature/max_tokens come from the pipeline's per-call pins (§4.3); top_p is the one value read from the server's /props. The harness adds no run-level value."
  },
  "bugs": [
    {
      "project": "PySnooper", "bug_id": 1,
      "python_version": "3.8.1",
      "buggy_commit": "…", "fixed_commit": "…",
      "test_file": "tests/test_chinese.py",
      "pythonpath": null,
      "needs_build": false,
      "native_test_paths": ["tests"],
      "test_cmd": ["python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--json-report", "--json-report-file={report}"],
      "pytest_version": "8.0.2", "json_report_version": "1.5.0",
      "checkpoint": "vanta-bi-PySnooper-1",
      "fail_to_pass": ["tests/test_chinese.py::test_chinese"],
      "pass_to_pass_count": 12,
      "flaky_nodeids": [],
      "preexisting_failures": [],
      "prompt_tokens_estimate": 41000,
      "context_ok": true,
      "issue_text": "…frozen, leak-gated…",
      "issue_text_sha256": "…"
    }
  ]
}
```

`vanta_planner` is listed in the doc and supported by the env switch but **not** in `configs` until
it exists. `seeds` is `[0]` for the pilot; seeds are a later, paper-only sweep (§8.3).

The `fail_to_pass` / `prompt_tokens_estimate` / `context_ok` / `issue_text` fields above are shown in
their *frozen* state: they are what a reviewed edit writes back into this file once the batch artifacts
exist. `prepare` itself never writes it — it writes `runs/bugsinpy/<batch>/prepare.json` and
`context_report.json` — so the committed file never contains a value no one approved.

### 7.3 Manifest — one record per (bug, config, seed)

`runs/bugsinpy/<batch_id>/manifest.jsonl` — append-only, one JSON object per line:

```json
{
  "schema_version": 1,
  "batch_id": "…", "wave": 1, "bug": "PySnooper/1",
  "config": "vanta", "mode": "repair", "seed": 0,
  "session_id": "…", "vanta_commit": "<build commit>", "vanta_dirty": false,
  "model": "…",
  "sampling": {
    "source": "pinned_per_call_plus_server_top_p",
    "pinned": [
      {"site": "creator",           "temperature": 0.1, "max_tokens": 32768},
      {"site": "attacker_generate", "temperature": 0.2, "max_tokens": 32768},
      {"site": "attacker_revise",   "temperature": 0.3, "max_tokens": 32768},
      {"site": "attacker_simplify", "temperature": 0.3, "max_tokens": 32768},
      {"site": "intent_audit",      "temperature": 0.0, "max_tokens": 2048}
    ],
    "server_top_p": null,
    "server_top_p_note": "unset — /props did not publish it"
  },
  "checkpoint": "vanta-bi-PySnooper-1",
  "python_version": "3.8.20", "python_version_pinned": "3.8.1",
  "pytest_version": "8.0.2", "json_report_version": "1.5.0",
  "candidate": {
    "kind": "dev_fix", "source": "dev",
    "patch_sha256": "…", "normalized_sha256": "…",
    "oracle": {"apply_ok": true, "hidden_pass": true, "suite_regressions": 0,
               "suite_regressions_count": 0, "patch_good": true}
  },
  "verdict": "Rejected", "exit_code": 1,
  "oracle": {"apply_ok": true, "hidden_pass": true, "suite_regressions": 0,
             "suite_regressions_count": 0, "patch_good": true},
  "label": "false_reject",
  "timings": {"prepare_s": 0, "run_s": 0, "oracle_s": 0, "total_s": 0},
  "artifacts": {"session_dir": "…", "final_patch": "…", "graded_tree_diff": "…",
                "issue_sha256": "…", "compact_state": "full|compacted"}
}
```

Note `sampling`: the harness chooses **no** run-level value. `pinned` is the per-call schedule the
pipeline itself enforces (§4.3), so a run's parameters are several rather than one, and
`server_top_p` is the one parameter no call site pins, read from `/props` (`null` when the server
publishes nothing). The manifest never *implies* the harness picked them.

Note `python_version`: that is the interpreter the checkpoint **actually** contains, not the version
BugsInPy's `bug.info` pins. The two differ (§12.3) and both are recorded, so the gap is auditable
rather than hidden by writing the actual number into a field that claims to be the pinned one.

`config` / `mode` / `seed` / `model` / `sampling` / `vanta_commit` are the provenance axes §4.4 and
§9.1 group by. `candidate` is `null` for **repair** runs. For review runs it names the patch under
test: `kind` is `dev_fix` / `noop` / `failed_patch`, `source` is `dev` / `noop` /
`repair:<config>:<session>` (which repair run produced it), and `patch_sha256` /
`normalized_sha256` are its identity — `normalized_sha256` is the dedup key (§7.1). The nested
`candidate.oracle` is the **label computed when that patch was produced**: it lets `metrics.py` score
every review row against the true label without re-running the oracle (§5.4).

**Known-bad dedup (decided):** pooled failed patches are deduplicated by their **normalized diff**
(`normalized_sha256`), not by raw bytes — two patches that differ only in an index line, an `a/`/`b/`
prefix or trailing whitespace are one candidate, run once. When more than 2 distinct patches survive,
the 2 with the smallest `patch_sha256` are kept. Every candidate records `source`, so a known-bad
result is traceable back to the repair run that generated it.

Append-only + a separate `state.json` (`{key: "done"|"running"|"failed"}`) makes **resume** trivial:
on start, replay `state.json`; at most one entry may be `running`, and it is re-run (Vanta runs are
idempotent per session id — the harness gives each attempt a fresh session dir and only marks done
after the oracle record is flushed).

### 7.4 Ordering and concurrency

**Strictly sequential, one LLM slot.** No parallel bugs: a single local `llama-server` is the
bottleneck, and interleaving would contaminate the per-run timing column (which is a deliverable).
The runner is a plain loop with a `finally`-style flush, so a Ctrl-C or a crash leaves the manifest
consistent. On reboot, `run` resumes from `state.json`.

`vanta_m3` is invoked as a subprocess with a controlled environment: pinned
`VANTA_SANDBOX_BACKEND=shuru`, `VANTA_SHURU_CHECKPOINT`, `VANTA_PYTHON`, `VANTA_LLM_*`, plus the new
`VANTA_ATTACKER` / `VANTA_LLM_SEED`. Its exit code is interpreted with the request-H contract
(0 approved / 2 InfraFailure / 1 other negative) — and the runner reads the session summary for the
real verdict rather than trusting the code alone.

---

## 8. Pilot set

Chosen for light, pure-Python, low-dependency projects, staged into waves by Python version, and
**filtered by the §1.6 context precheck**. Bug counts and Python versions verified live.

| Project | bugs | Python versions seen | chosen | wave | context (design estimate) |
|---|---|---|---|---|---|
| PySnooper | 3 | 3.8.1 | 1, 2, 3 | 1 | tiny, ~40 k tok |
| thefuck | 20 | 3.7.0 | 1, 2, 3 | 2 | small, **borderline** — measure |
| httpie | 5 | 3.7.3 | 1, 2, 3 | 2 | small |
| cookiecutter | 4 | 3.6.9 | 1 | 3 | small, needs `setup.py build` |
| **youtube-dl** | ≥1 | 3.7.0 | ~~1~~ | — | **megabytes — expected context reject** |
| black | 20 | (verify) | reserve | — | medium, slow to collect |
| tqdm | 9 | 3.6.9 | reserve | — | small, needs `setup.py build` |

**The `youtube-dl` decision (review point).** youtube-dl is hundreds of extractor files — the source
concat is orders of magnitude past any window we run. Rather than silently dropping it, it stays in
the split with `context_ok: false` so the precheck's verdict is **auditable in the committed record**,
and it consumes **zero runs**. Its slot in the 10-bug pilot is taken by a reserve. The
`context_sizes.md` dump (§1.6) reports sizes for **all** candidates, reserves included, so the
choice is data, not assertion.

Reserves, in order: **black 1–2** (first choice — 20 bugs, so the reserve is not exhausted by one
bad environment), **tqdm 1**, then further PySnooper/httpie bugs.

Waves (decided) — ordering by Python version, cheapest environment first, because the interpreter
build is the only step that can invalidate a whole batch:

| Wave | Python | Bugs | Seeds | Purpose |
|---|---|---|---|---|
| 1 | 3.8.1 | PySnooper 1, 2, 3 | `[0]` | **go/no-go gate** — checkpoint, `import ssl`+pip gate, oracle, leak gate, runner |
| 2 | 3.7.0 / 3.7.3 | thefuck 1–3, httpie 1–3 | `[0]` | the pilot proper |
| 3 | 3.6.9 | cookiecutter 1 | `[0]` | the last pilot slot; the two-repo `setup.py build` case |

Pilot = waves 1 + 2 + 3 = **10 bugs**.

Per-bug environment notes that shape the estimates:

| Bug | py | Checkpoint | Notes |
|---|---|---|---|
| PySnooper/1..3 | 3.8.1 | `vanta-bugsinpy-py3.8` | clean, no `pythonpath`, no build step — cheapest, standalone Python |
| thefuck/1..3 | 3.7.0 | `vanta-bugsinpy-py3.7` | pytest-native suite, no build step; **context borderline** |
| httpie/1..3 | 3.7.3 | `vanta-bugsinpy-py3.7` | pytest-native, `requests`-like deps |
| cookiecutter/1 | 3.6.9 | `vanta-bugsinpy-py3.6` | **`pythonpath=cookiecutter/build/lib/` → needs `setup.py build`** |

### 8.1 Per-bug time estimate

Anchor: a full repair run on `heretic` took **2040 s** (≈34 min) to reach `AllTestsPassed`. These
bugs are far smaller (patches of ~10–100 lines, suites of seconds), and the prompt is dominated by
the source concat, which is small here — so calls get *faster*, not slower. Working estimate per run
type:

| Run type | LLM calls | Estimate |
|---|---|---|
| `baseline` repair (no Attacker, ~1–2 attempts) | 1–2 | 4–12 min |
| `vanta` repair (Attacker + more test rounds, ~2–4 attempts) | 3–8 | 10–30 min |
| review, one candidate (`baseline` = 0 calls, `vanta` = 3–6) | 0 / 3–6 | 1–3 min / 10–25 min |
| oracle (filtered patch + hidden run + suite ×2) | 0 | 2–5 min |

Runs per bug = `|seeds| · (2 + 2·|candidates|)`, where `|candidates|` — the per-bug review set (§7.1) —
is **2 to 4** (dev fix + no-op, plus up to 2 pooled failed patches). Both configs review the same set,
so the reviews contribute `2·|candidates|` per seed. `|candidates|` is not known until the repair runs
finish, so the counts below are ranges: at `seeds=[0]` a bug takes **6–10 runs**, and the reviews alone
are 4–8 of them.

| Wave | Bugs | Seeds | Runs | Checkpoint bake | Estimated wall clock |
|---|---|---|---|---|---|
| 1 (gate) | 3 | `[0]` | 18–30 | 20–40 min (py3.8, standalone) | 3–6 h |
| 2 (pilot) | 6 | `[0]` | 36–60 | 30–70 min (py3.7 + OpenSSL 1.1.1) | 6–12 h |
| 3 (last slot) | 1 | `[0]` | 6–10 | 30–70 min (py3.6 + OpenSSL 1.1.1) | 1–2 h |
| **total** | **10** | `[0]` | **60–100** | ~1.5–3 h | **~12–22 h → 2 overnight batches** |

Runs are strictly sequential (one LLM slot), so wall clock is the **sum**, not the max. At
`seeds=[0]` the whole pilot fits comfortably; the previous `seeds=[0,1,2]` plan (98 runs in wave 2
alone) did not fit and is dropped (§8.3). Adding `vanta_planner` later multiplies the vanta rows by
~1.5.

**Wave 1 is the go/no-go gate (decided).** If the checkpoint build, the `import ssl` + pip assertion,
the oracle, or the leak gate misbehaves on wave 1, `run` **stops and reports** before any wave-2 work
is scheduled. Wave 1 is deliberately cheap — 18–30 runs, `seeds=[0]`, one interpreter — so a bad
assumption costs hours, not days.

**Calibration step (mandatory).** Before wave 2, replace this table with measured wave-1 timings.
The estimates above are not data.

### 8.2 Wave-1 gate checklist

Explicitly, wave 1 must demonstrate all of:

1. `vanta-bugsinpy-py3.8` passes `import ssl`, a real `pip install`, and `pytest --version` (§3.2).
2. The per-bug layers bake and `pytest --collect-only -q` succeeds inside them.
3. `prepare` derives a non-empty `FAIL_TO_PASS` for all three PySnooper bugs, with no flaky escapes.
4. The leak gate produces a clean text for all three, and `issue_texts_all.md` reads as actionable.
5. The oracle returns `patch_good = true` for the developer fix and `false` for the no-op patch, and
   the pooled candidate set for a bug is reproducible from the manifest alone.
6. Baseline review (Attacker off) drives all its runs to completion with a consistent manifest, and
   the paired vanta review rows exist for the same candidate set and seed.

### 8.3 Seeds

**Pilot uses `seeds=[0]`.** One seed, one sample per (bug, config). The class pilot is about whether
the harness produces *correct labels at all*, not about variance. Seeds are a later, paper-only sweep
(`[0,1,2,3,4]`) once the pipeline is known to work and once the write-up needs intervals on the
model's stochasticity. Stated plainly because it bounds every claim the pilot can make: a pilot
`false_approval` is one sample, not a rate.

---

## 9. Metrics

`scripts/metrics.py` already has `classify()` → `Verdict(.bucket, .signals_total, .signals_deciding,
.signals_undecidable, .warnings, .schema_label)`, the `--labels` oracle mechanism, and the
`infra_failure` bucket from request H. What the pilot additionally needs:

### 9.1 Pooled, per config

Everything below is computed **grouped by provenance** (decided): `config` × `mode` × `seed` ×
`model` × sampling params × `vanta_commit`, read from the batch manifest and each run's
`run.summary`. No comparison is ever made across provenance groups — that is the point of §4.4.

| Metric | Definition |
|---|---|
| `resolve_rate` | `hidden_pass = true` ÷ runs, per config |
| `patch_good_rate` | `patch_good = true` ÷ runs, per config |
| `approval_precision` | `true_approval` ÷ (`true_approval` + `false_approval`) |
| `false_approvals` | count + list of `(bug, seed, reason: hidden_fail \| suite_regression)` |
| `false_rejects` | count + list, from review mode's known-good set |
| `spurious_reject_rate` | rejected-known-good that did **not** discriminate (existing project rule) |
| `ECR` / `EIR` | pooled over all transition rows of the config, plus a **Wilson 95 % interval** |
| `infra_failure_rate` | `InfraFailure` ÷ runs — the pilot's environment health gauge |
| `unobservable_rate` | `apply_ok = false` ÷ runs — the oracle's own failure gauge |

### 9.2 Wilson interval

Implemented locally (no scipy — keep the dependency surface flat):

```
centre = (p̂ + z²/2n) / (1 + z²/n)
half   = z·√( p̂(1−p̂)/n + z²/4n² ) / (1 + z²/n)
CI     = centre ± half        z = 1.96
```

`n` = the metric's own denominator (for ECR: rows Failed at N; for EIR: rows Passed at N) — the same
conditional denominators the project already mandates, **not** redefined (C2).

### 9.3 Caveat to report, not to hide

Pooling across heterogeneous bugs is a **cluster** problem: rows within a bug are not independent,
so the Wilson interval understates uncertainty. The doc reports pooled + per-bug, and labels the
pooled interval as naive. A cluster-robust alternative (per-bug proportion → interval on the mean)
is noted as future work, not implemented in the pilot. At `seeds=[0]` the intervals are also
dominated by run-to-run model variance, which §8.3 explicitly defers.

### 9.4 Where the code goes

**Decided:** an **additive section** in `scripts/metrics.py` — new functions, `--pool-by-config`,
grouping by the §4.4 provenance axes, and an `oracle` block that consumes the harness manifest.
Existing `classify()` semantics are untouched, so existing tests and existing reports are unaffected.

---

## 10. Change inventory

| Change | File | Size | C2 risk |
|---|---|---|---|
| **`final.patch` written for every terminal verdict** (§5.1) | [driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L1811-L1841) | small (move one call) | none (Approved path byte-identical) |
| `VANTA_ATTACKER=off` skip + Creator feedback restriction | [driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs) | small, env-gated | none (default = today) |
| `VANTA_PLANNER` placeholder → `InfraFailure` | driver.rs | tiny | none |
| `seed` on `LlmConfig`, added to the request body | [config/mod.rs](file:///Users/kyu-yi/vanta/src/config/mod.rs), [llm/mod.rs](file:///Users/kyu-yi/vanta/src/llm/mod.rs) | small | none (unset = today) |
| `/props` extended to report effective sampler defaults (§4.3) | [llm/mod.rs](file:///Users/kyu-yi/vanta/src/llm/mod.rs#L266-L299) | small, read-only | none |
| provenance keys in `run.summary` + populate `benchmark_instance` / `benchmark_label` | driver.rs (one construction site) | small | none |
| `.gitignore` line for `runs/bugsinpy/` | [.gitignore](file:///Users/kyu-yi/vanta/.gitignore) | 1 line | none |
| everything else | new files under `scripts/`, `plans/` | — | none |

Nothing here alters `GovernorDecision`, `attempt_verdict`, the Gatekeeper, or the ECR/EIR math.

New files:

```
plans/bugsinpy.md                  ← this document
plans/bugsinpy_split.json          ← hand-authored list (per-bug test_cmd, layer, issue text once frozen)
scripts/bugsinpy_run.py            ← loader + split + runner + oracle + leak gate + precheck + resume
scripts/metrics.py                 ← additive only: pooled rates, Wilson, provenance grouping
runs/bugsinpy/<batch>/             ← gitignored (one new .gitignore line)
    manifest.jsonl, state.json, context_report.json, context_sizes.md, issue_texts/,
    issue_texts_all.md, leak_gate/, oracle/, <session>/attempt_N_graded_tree.diff + _manifest.json
```

---

## 11. Build order

Reordered after review: `final.patch` is now **step 1**, because the oracle is untestable without it.

1. **`final.patch` for every terminal verdict** (§5.1). Small, testable in isolation: run a
   deliberately-rejected patch and assert `debug/final.patch` exists and applies cleanly.
2. Split file + loader + `prepare`: checkout, **SWE-bench split**, two-run flaky-flagged
   `FAIL_TO_PASS` / `PASS_TO_PASS` derivation, AST strip, relaxed fixed-tree gate, **context
   precheck** (§1.6) with `context_sizes.md`, leak gate, collectability gate, and the
   `issue_texts_all.md` review dump with its `NEEDS DECISION` section. No LLM, no Vanta.
3. **py3.8 checkpoint only**, standalone Python, with the `import ssl` + `pip install` + `pytest`
   assertions, plus the PySnooper per-bug layers.
4. Oracle standalone (test-hunk filter, hidden run, suite diff ×2) — fully testable on known-good
   **and** known-bad with no Vanta run at all. This is what lets every review candidate carry its
   oracle label, so the review passes can be scored without re-running the oracle.
5. `VANTA_ATTACKER` / `VANTA_PLANNER` / `VANTA_LLM_SEED` plumbing, the per-call sampling schedule
   plus the server-`top_p` probe (§4.3), and the §4.4 provenance keys in `run.summary`. Tests: with
   the env unset the request bodies are byte-identical (no `seed` key) and the existing suite gives
   the same verdicts; `VANTA_ATTACKER=off` makes zero Attacker LLM calls without changing a verdict;
   `VANTA_PLANNER=on` exits 2 with `final_decision=InfraFailure`; and the recorded schedule equals the
   `temperature` / `max_tokens` the request bodies actually carried.
6. Runner + manifest + resume: repair passes first, then both review passes over a **symmetric,
   deduplicated** candidate set (§7.1). `.gitignore` line.
7. Compaction (diff + manifest) with the §6.3 rerun-safety test.
8. **Wave 1 gate run** (§8.2) → report it, and only then bake py3.7 and py3.6 and schedule waves 2–3.
9. Metrics additions + Wilson; calibration numbers written back into §8.1.

---

## 12. Decisions and remaining unknowns

### 12.1 Decided

| # | Question | Decision |
|---|---|---|
| D1 | Issue text | Derived from the stripped fix-commit message + an LLM behavioural paraphrase, with a **hard leak gate** (§2.2). Step 2 strips **diff and code blocks only** (no symbol rule). The gate rejects or strips any token that exists only in the fix diff or the hidden test, logs every removal, and flags unactionable texts under `NEEDS DECISION` in `issue_texts_all.md` before any run. |
| D2 | Python environments | One checkpoint **per version** + thin per-bug layers, using the existing `vanta-deps-<fingerprint>` marker/freeze pattern. **py3.8 via standalone Python; py3.6/3.7 build OpenSSL 1.1.1 into the checkpoint** (or an older Debian base). `import ssl` + a real `pip install` + `pytest --version` are part of the wave-1 gate. **pytest / pytest-json-report pinned per version.** |
| D3 | `baseline` | Env-gated `VANTA_ATTACKER=off` in `driver.rs`; skips the Attacker and restricts Creator feedback to native/static. Defaults to today's behaviour. |
| D4 | Seeds / sampling | `VANTA_LLM_SEED` implemented and stored (the `seed` key is added to the body **only** when set; unset leaves it byte-identical). The pipeline's own **per-call** `temperature` / `max_tokens` pins (§4.3) are kept exactly as they are and recorded per site; `top_p` is the only server-sourced value, read from `/props`. No new sampling knobs, and the harness adds no cap of its own. Determinism under a seed is not claimed. |
| D5 | Provenance | Config, mode, seed, model id, the per-call sampling schedule + server `top_p`, pinned pytest versions and Vanta commit recorded in **both** `run.summary` (lowercase keys + the existing `benchmark_instance` / `benchmark_label` fields) and the batch manifest. `metrics.py` groups by all of them. |
| D6 | Disk | Harness-side compaction to diff + manifest. **Rerun-safety rule:** a session with a rejecting signal keeps its full graded snapshot until its rerun is written, or the rerun rebuilds from base + diff. A test covers both branches. |
| D7 | Metrics location | Additive section in `scripts/metrics.py`. |
| D8 | Test trees | **SWE-bench split** (§1.2–§1.3): Vanta's tree = buggy source + **fixed** test files − `FAIL_TO_PASS`. `PASS_TO_PASS` is the native suite. Fixed-tree gate relaxed to "no fixed-only failures"; pre-existing failures recorded; every suite run twice to flag flaky nodeids. |
| D9 | Oracle & labels | `final.patch` built for **every** terminal verdict (§5.1). The oracle applies **only non-test hunks**; test files are always the fixed version's. `patch_good = hidden_pass ∧ suite_regressions == 0`, and the label table keys off `patch_good` (§5.3). |
| D10 | Waves & budget | Wave 1 = PySnooper (py3.8, `[0]`) as a **go/no-go gate**; wave 2 = thefuck + httpie (py3.7, `[0]`); wave 3 = cookiecutter (py3.6, `[0]`). 10 bugs, 60–100 runs, ~12–22 h. Budgets in §8.1. |
| D11 | Context precheck | `prepare` estimates each bug's source-concat tokens and **rejects** over-budget bugs with a documented 25 % margin, writing `context_sizes.md` **and** `context_report.json` for **all** candidates and printing the table. The split is an input and is never rewritten by a stage. Substitutions wait for wave 1's real `usage.prompt_tokens` — the estimate runs ~1.5× high. Narrowing files to fit is v2. |
| D12 | Review coverage | Review mode on **every** bug over a **symmetric** per-bug candidate set — developer fix (`dev`, known-good) + no-op floor (`noop`) + up to 2 pooled failed Creator patches (`repair:<config>:<session>`, known-bad). `InfraFailure` / `apply_ok = false` skipped; deduplicated by **normalized diff**; each candidate records its source and oracle label. Baseline and vanta review the same set at the same seed (paired). Applied with test hunks dropped. |
| D13 | Run order | baseline repair → vanta repair → baseline review (no LLM) → vanta review. Repair first, so failed patches exist before any review; reviews last, so both configs see the same set. |
| D14 | Artifacts | `runs/bugsinpy/<batch>/`, with a `.gitignore` line so the path stays readable. |

### 12.2 Still open — defaults chosen, flag if you disagree

1. **Review runs re-roll per seed** (the Attacker is stochastic), which is why runs/bug is
   `|seeds| · (2 + 2·|candidates|)`. At `seeds=[0]` this is moot; it matters again when the paper
   sweep arrives. The **candidate count** itself is data-dependent (2–4, §7.1) and is fixed by the
   repair runs, not chosen per seed.
2. **`vanta_planner` placeholder returns `InfraFailure`** rather than a silent pass, so the config is
   addressable but cannot produce a fake result.
3. **black is the first reserve**, not a pilot bug. If a wave-2 bug dies at prepare, black/1 takes its
   slot; black has 20 bugs so one bad environment does not exhaust the reserve.
4. **youtube-dl stays in the split with `context_ok: false`** rather than being deleted, so its
   rejection is auditable in the committed record.
5. **`jsonschema`-style strictness on the split file**: unknown keys are an error, not ignored — the
   split is the experiment's protocol.

### 12.3 Known limitations to state in the write-up

- The issue text is *derived from* the fix commit, so it is not independent of the fix (§2.3). It
  cannot be presented as a "real bug report".
- **Memorisation risk.** These BugsInPy bugs are **public since 2020**, and their fixes, commit
  messages and bug-revealing tests are all on GitHub. A model trained after 2020 may have memorised
  the fix outright, which inflates resolve rates in a way our harness cannot detect. This bounds any
  claim of the form "Vanta repaired bug X": the pilot can measure *labelling correctness* and
  *rejection behaviour*, and cannot separate repair from recall.
- Sanitizing `vanta_tree/` makes the native suite green by construction, so `baseline` is a one-shot
  repair with no iterative signal — its approvals are unverified by construction. That is the base
  rate `vanta` is measured against, not a defect of the harness (§1.5).
- **Interpreter version divergence (measured).** BugsInPy pins `python_version="3.8.1"` in each
  `bug.info`, but the py3.8 checkpoint actually built is **3.8.20** — the last 3.8 release that still
  builds against the base image's OpenSSL. Every environment-dependent result therefore runs on a
  different *patch release* than the record's own `bug.info` names. Both numbers are recorded side by
  side in the manifest (`python_version` = actual, `python_version_pinned` = BugsInPy's), so the gap
  is auditable rather than papered over by writing the actual number into a field that claims to be
  the pinned one. Nothing in the pilot is sensitive to 3.8.1-versus-3.8.20 (all three projects are
  pure Python at these versions), but the write-up must not claim the pinned interpreter.
- `PASS_TO_PASS` comes from the fixed tree, so a bug whose fix deleted tests has a smaller native
  suite than the buggy commit had. That is the SWE-bench convention, and it is the fair one, but it
  means "native suite" is not a constant per project.
- Pooled Wilson intervals across heterogeneous bugs are **naive** (clustering); per-bug numbers are
  reported alongside (§9.3).
- Seeds do not guarantee reproducibility (§4.3), and the pilot runs a single seed (§8.3), so every
  pilot rate is one sample per (bug, config).
