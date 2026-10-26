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
| **`vanta_tree/`** | buggy source **+ the fixed commit's test files**, minus `FAIL_TO_PASS` and minus the undeclared flippers | **Vanta (`--repo`)** |
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
4. Split the flipping set `F_before \ F_after` by **what BugsInPy declares for this bug** — the file
   `bug.info`'s `test_file` names, plus any nodeid `run_test.sh` names, and nothing else:
   - `FAIL_TO_PASS` = the flipping nodeids that *are* declared. **This is the hidden oracle**, and
     `patch_good` is judged only on it.
   - `f2p_undeclared` = the flipping nodeids that are *not* declared. They are recorded with a
     `f2p_undeclared_reason`, excluded from the oracle **and** from `PASS_TO_PASS`, and hidden from
     `vanta_tree/` as well (step 7) so the Creator never chases a test the benchmark does not ask
     for. Measured cause on PySnooper/2: the fix commit adds
     `tests/test_utils/test_ensure_tuple.py` for a helper it also adds, so that test flips because
     the helper is absent on the buggy source — not because of the bug. A flipping test is a
     *candidate* oracle, not an oracle.
5. `PASS_TO_PASS` = everything that passes on `src_fixed/` — this is Vanta's native suite.
6. Sanity gates (prepare fails loudly if any trips):
   - the **declared intersection is non-empty** → otherwise there is no oracle to grade against at
     all, even though tests flip (PySnooper/1's shape). An empty `FAIL_TO_PASS` after the split is a
     gate failure, not a silent pass;
   - **`F_after \ F_buggy_pre = ∅`**, where `F_buggy_pre` = failures on the *plain* `src_buggy/`
     tree (buggy source + buggy tests). Reading: **the fixed tree has no failure that the buggy tree
     does not also have.** This replaces the previous "`F_fixed` must be empty" gate, which the
     review correctly called too strict. Pre-existing failures are **recorded**
     (`preexisting_failures`) and never charged to a patch. Note this is a *relaxation, not a
     removal*: a failure that is fixed-only is still a hard stop, because it means the environment is
     wrong.
   - `vanta_tree/`'s suite is green again after stripping (§1.5).
7. **Strip**: remove `FAIL_TO_PASS` **and** `f2p_undeclared` from `vanta_tree/` and `vanta_clean/`.
   If a test *file* is entirely hidden, delete the file; otherwise delete only the named functions
   (**AST-level, not regex**). Record `hidden_files_deleted` and `partially_hidden_functions` in the
   prepared manifest, alongside the `f2p_undeclared` list and its reason. The leak checks (§2.2) run
   against this stripped tree, so removing the extra tests must keep them passing.

### 1.4 What Vanta sees

`vanta_tree/` = buggy source + the **fixed** test suite minus the `FAIL_TO_PASS` tests and minus the
undeclared flippers: no fix, no oracle. The `Repo` is loaded by [driver.rs](file:///Users/kyu-yi/vanta/src/pipeline/driver.rs#L152)
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
   rejection. Its top level carries the batch's `n_ctx` / `n_ctx_source` (the window the gate was
   sized against — §1.6.1), and `prepare` prints the same numbers as a table on stdout.
   **A bug with no measurement prints its reason beside the verdict, in the table and in the JSON,
   and both are never empty.** `not measured` is `context is None`, which every early return in
   `prepare_one` produces, and each of those has already recorded why in `gate_failures` — a refused
   frozen report, a missing `github_url`, a bug with no report to send, a metadata mismatch. The
   first dev prepare printed the verdict for all eleven bugs and the cause of none of them beside a
   row, so `fastapi/1`'s refusal — one of five, in the same words as the other four — was visible
   only in the summary's bullet list, three tables above the row, and the row itself read as if the
   bug had never been reached. The reason column is not decoration: "failed a gate" and "never got
   to" license different responses, and a table that cannot tell them apart makes a batch look empty
   for the wrong reason. A row with no recorded failure prints `NO_MEASUREMENT_REASON`, which says
   the reason was lost in `prepare` itself and names that as the defect — an empty `reasons` list in
   `context_report.json` is the same hole, so it is filled the same way.
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
6. **~~Narrowing files to fit is v2 work~~ — superseded by §1.6.2.** This point said a bug whose
   source concat overran the window was replaced from the reserve list, never patched up. The
   measurement that forced the reversal: `prepare --measure-only --reserves measure` rejected **10
   of 11** dev/reserve bugs as `source_truncated_by_cap`, because Vanta was showing the Creator the
   *first* ~262 KB of an 11.8 MB repo (matplotlib) and the bug's own file was usually past that
   point. Replacing ten bugs is not a pilot; the prompt has to become the *relevant* files. §1.6.2
   is that change.

#### 1.6.1 The gate may only size against a window the run will use

The precheck is only worth running if it measures the prompt the *run* will send. The run resolves
its window the way `llm::context_window_tokens` does — `VANTA_LLM_CONTEXT_TOKENS`, else the server's
`/props`, else **65536** — and that fallback is a wrong answer that looks like a right one:

> The first full `prepare` ran with `VANTA_LLM_CONTEXT_TOKENS` unset in the shell. The gate sized
> against the 65536 fallback and gave a **16128**-token budget instead of **65280**, then rejected
> `PySnooper/1` and `/2` as `source_truncated_by_cap` — for a prompt no run would ever have sent.

So `prepare` **refuses** to run the context gate unless the window comes from an explicit source, in
this order:

| source | how |
|---|---|
| `VANTA_LLM_CONTEXT_TOKENS` | the variable the run itself reads first — wins |
| `--n-ctx TOKENS` | a per-invocation flag, **checked against** the other two rather than overriding them |
| the server's `/props` | `default_generation_settings.n_ctx` or top-level `n_ctx`, base URL `VANTA_LLM_BASE_URL` |

If none is available, `prepare` fails before the first checkout with a `ContextError` naming all
three ways out. `--n-ctx` is deliberately *not* an override: a run never reads the flag, so a flag
that disagreed with the variable or with `/props` would size the gate against a window the run will
not use — the same footgun in a different coat. A disagreement is an error, not a preference.

The resolved window and its source are printed above the summary table, recorded at the top level of
`prepare.json` (`n_ctx`, `n_ctx_source`) and of `context_report.json`, and re-read by `run`, which
**warns loudly on stderr** when its own window differs from the batch's — before `--plan` prints, so
the cheap moment to notice is the moment it is noticed. The two halves are the rule: `prepare`
refuses to start without an explicit source, `run` refuses to stay silent about a drift.

#### 1.6.2 Selection replaces truncation — show the relevant files, not a capped dump

The old prompt was built by `repo::collect_source_files`, which walked the repo in `fs::read_dir`
order and applied **256 KB total / 64 KB per file during the walk**, `continue`-ing on any file that
would exceed the cap. The surviving set was therefore "the files the filesystem happened to hand us
first", and the bug's own file was frequently not in it. `matplotlib-1` (11.8 MB) got the first
~262 KB; only `PySnooper-3` (20 KB) fit whole. This is the cap the 10-of-11 rejection names.

The reversal is to **select**, not to truncate. One Rust module, `crate::context`
([src/context/mod.rs](file:///Users/kyu-yi/vanta/src/context/mod.rs)), owns both halves so there is a
single implementation for the run and for the gate (item 5 shells the gate out to it):

1. **Walk the whole tree, deterministically, with no cap.** Directory entries are sorted at every
   level and the candidate list is sorted by path, so two runs over the same tree — on any
   filesystem — produce the same order. `.git`, build output and vendored/generated trees are pruned
   by name (`SKIP_DIRS`); symlinks are never followed; the walk records a **skip ledger** of every
   pruned path and why (`pruned_dir:<name>`, `symlink`, `unreadable`). "We skipped 3000 files" is not
   checkable; "we pruned `build/`, `node_modules/`, `.git/` and these 4 symlinks" is.

2. **The caps apply only to the selected set, never before selection.** A file chosen by the ranking
   goes in **in full** where it fits the role's token budget; a file over **64 KB** goes in as an
   **outline** (signatures + docstrings, parsed line-by-line — nothing is executed) rather than being
   skipped, so a big touched file is never silently dropped. A file the budget leaves no room for in
   full is priced down the ladder to an outline, and then to its **path alone** (§1.6.9); only a file
   that does not fit even as a path is dropped, and it is recorded in `dropped_for_budget`.

3. **The budget is a token budget, computed the same way in the run and the gate**
   (`context::budget_tokens`): `n_ctx − OUTPUT_RESERVE_TOKENS(32768) − CHAT_OVERHEAD_TOKENS(256) −
   25 % of n_ctx`. The first two terms are exactly what `llm::context_budget_exceeded` subtracts, so
   the budget's edge sits on the run-time guard's edge; the margin is the harness's addition (§1.6
   point 2). At `n_ctx=131072` this is **65280** tok ≈ 255 KiB.

4. **Fill greedily by score.** Each selected file carries its reason(s); the score is the sum of the
   reason weights, ties broken by path (so the order is total and stable). The prompt is the file
   tree (paths only, repair mode) followed by each file in score order, each in the richest of the three
   forms — full, outline, path — the remaining budget still allows (§1.6.9).

**Review mode (item 1).** With a patch under review, the selection is, in full: the patch's own
touched files, the modules they import at module level (one hop), and the visible test files that
import a touched module or name one of its modules. The patch is *the thing under review*, so reading
it is not a peek at the answer; nothing else is an input.

**Repair mode (item 2).** With no patch, files are ranked from the **issue text only**: explicit
paths or module names, traceback frames, and identifiers the issue mentions that a file *defines* —
found by parsing, never by running code — plus one hop of import neighbours for the top files, and the
repo's file tree so the model sees what exists.

Two rules keep a plain English word in an issue from boosting a file's rank. **A name is counted only
when it is code-shaped:** it contains `_`, a digit or an inner capital (CamelCase), or the reporter
writes it as code — in backticks, as `name(`, or as the name side of `obj.name`. So `a`, `in`, `fig`,
`release`, `version` and `report` are not names, while `register_named_middleware`, `is_string_dtype`
and `BigQueryTarget` are. **A definition is any `def`/`class` at any depth** — a method counts, not only
a module-level function — **or a column-0 binding** at the top of a module; a binding nested inside a
function is not a definition a reporter could be naming. A GitHub blob URL into the project's own
repository (`github.com/<org>/<repo>/blob/<sha>/<path>#L<n>`) is a **path** mention of `<path>`: the
reporter named that file, the line anchor is stripped and the sha is never read.

**Three refinements the dev-2 ranking needed.** The first ranking put `pandas/__init__.py` first for
`pandas/1`, and a dozen generic `pandas/core/arrays/*.py` above `pandas/core/dtypes/common.py` — the file
that defines the one symbol the issue backticked. Three rules close that, and they are amendments to the
rules above rather than new signals:

- **A dotless module name is not a path signal.** `python_module_path("pandas/__init__.py")` is `pandas`,
  and every pandas report writes "the latest version of pandas" in ordinary prose, so matching a module
  with no `.` in it made the package's own `__init__.py` a `path` match at weight 400 — the top of the
  ranking — for an issue that never named it. `path_match_rule` now requires the module to be **dotted**:
  `sanic.app` is a path the reporter had to spell out, `pandas` is the project's name.
- **A definition outranks a mention, and the match's worth is scaled by the name's rarity.** A file is
  reached by `Identifier` only when it *defines* a name the issue names; a mention earns nothing from that
  signal, so the string merely appearing in a file can never be worth more than the file the name lives
  in. What the match is worth is then weighted by document frequency — the IDF the retrieval literature
  uses, `ln(1 + N/df)`, with `N` the tree's file count and `df` the count that *define* the name — mapped
  onto the band `[IDENTIFIER_WEIGHT_FLOOR 120, IDENTIFIER_WEIGHT 200]`: a name exactly one file defines is
  evidence and sits at the top, a name a large share of the tree defines stands near the floor. The
  population counted is definitions, not mentions, so a file's score is a property of the tree the run is
  given: a hidden test defines no symbol the issue names, which is why
  `the_selection_is_unchanged_when_a_hidden_test_is_added_or_removed` still holds. The floor sits above
  `ImportNeighbour` (100) on purpose, so a definition is never displaced by reachability.
- **A fenced code block is not a form that writes a name as code.** The reader used to count backticks as
  a toggle, so the three backticks of a ```python fence flipped the state and every token in the block
  became `backtick`: `pandas/1`'s reproduction binds `dtype`, a dozen array modules define a `dtype`
  method, and they ranked above the file the issue named. `issue_identifiers` now reads the written forms
  by **adjacency** — a token touching a backtick, a token followed by `(`, the name side of `obj.name` —
  which is both the rule stated above and the reader the paper's flag uses (`issue.py`'s
  `_written_as_code`). So `fig` inside `matplotlib/1`'s fenced block is a local and stays out, and the
  fence's own `dtype` is plain English.

**The fallback for an issue that names no definition (item 3).** The three signals above all read
*names*: a path, a frame, a symbol a file declares. Some reports name none of those — `fastapi/1`'s
report names `response_model_exclude_defaults`, which is a **parameter**, so no candidate declares it and
the selection came out empty. A fourth, lowest signal closes that: **BM25 over the code-shaped tokens of
the issue text against the file contents**, which is SWE-bench's own retrieval baseline. The query is
`issue_identifiers`' keys, the same reader every other signal uses; each file's contents are counted as
term frequencies over identifier-shaped tokens; the score is the standard BM25 sum, `k1 = 1.2`,
`b = 0.75`, with `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`. It is recorded as the reason **`bm25`**, rule
`bm25`, and its **score is recorded with it** — quantized onto `[1, BM25_WEIGHT 40]` by the run's own
maximum, so the best match tops the band and the rest are ordered by ratio below it.

Three properties make it a fallback rather than a signal:

- **It is marked only on a file no other signal reached.** A file the issue's own path, frame or
  definition reached keeps exactly the score that signal gave it; `bm25` is never folded into it. So the
  fallback cannot re-price or displace a file the report named — it orders the files nothing named
  (`fastapi/1`) and nothing else.
- **Its band tops out below every other reason's**, at 40, below `Test` (80). A file the text reached can
  therefore never outrank one a real signal reached, whatever the corpus.
- **Its corpus is the project's source with the tests left out.** A test is the harness's to strip — the
  F2P test *is* the answer — and a document the harness removes must not re-price the documents that
  stay; the same rule the identifier's document frequency follows. The frequencies are read from the tree
  the run holds, so the fallback never learns anything the run was not given.

It is **not** §1.6.3's deleted fallback. That one filled the tree in walk order — a listing with the
alphabet on top — and the gate rightly refuses an ungrounded selection. This one is reached from the
issue text's own tokens against the tree's contents, so the file it finds is grounded in what the
reporter wrote, which is what `grounds_content` asks. It carries a body only where §1.6.8 allows one: a
`.md` or an example reached by text alone travels as its path in the listing.

**Never hidden information (item 3).** `F2P`/hidden tests, the fix commit and its diff, and
`bug.info` are not inputs and cannot be reached from the selection. The harness keeps them out of the
tree as today; a test proves selection is unchanged when the hidden tests are removed or added.

**Provenance (item 4).** Every selected file, its score, its reason(s) and the bytes/tokens shown go
into the session log and `run.summary`; the role prompts' versions are bumped.

**The gate (item 5).** `prepare`'s context gate checks the **selected** context fits the budget (not
that the whole repo does) by shelling out to `vanta_m3 --select-context --json` — the same binary and
the same `crate::context` the run uses — and `context.py`'s mirror of the old walk is removed, so
there is one implementation. `localization_hit` (whether the fix commit's touched files were among
those shown) is computed **after** a run, never during, and recorded per run.

Landed in **item 1** — review mode: `crate::context`, the budget block, the review prompt seam, and
the sorted walks in `repo::collect_source_files` / `candidate::collect_tree`. It supersedes the
`fs::read_dir`-ordered capped walk as it stood at `a29b9c8`, the parent of that commit. (The item's
own hash is reported with the commit, not cited here: a hash written into the commit it names cannot
survive that commit's own creation.)

Landed in **item 2** — repair mode, on top of item 1 at `51339e8`: `select_repair_context` and the
issue-text parsers that feed it (`issue_tokens`, `issue_traceback_files`, `issue_identifiers`,
`module_level_symbols`), the one hop of import neighbours for the seed files, the repo file tree in
the prompt, and the driver seams `select_repair_prompt` / `creator_source_files`. The Creator now
receives the *selected* files plus a `REPOSITORY FILE TREE (paths only)` listing entry instead of
`repo.source_files`, and the Attacker, the auditor, BATE and the Diagnostician receive the same
selection through `ContextSelection::render` — the seam that used to be called
`review_prompt_concat` is `prompt_source_concat` because it is no longer review-only. Weights are
ordered so a file the *issue* names always outranks a neighbour of one: `PatchTouched` 1000 > `Path`
400 > `Traceback` 300 > `Identifier` 200 > `ImportNeighbour` 100 > `Test` 80.

Landed in **item 3** — on top of item 2 at `bab6f97`. The landing is a test, because the harness's
half of the rule is what it already did: `prepare` / `trees::strip_hidden_tests` removes the hidden
(F2P) tests from `vanta_tree` and `vanta_clean` before any run sees a tree, and the fix commit's
diff and `bug.info` are never written into a tree at all. The localizer's half is structural — its
inputs are the tree, the issue text, the patch under review and the budget, and `FAIL_TO_PASS`, the
fix commit, the fix diff and `bug.info` are not parameters of `select_review_context` or
`select_repair_context` and cannot be reached from either. What was missing was a test that the
*output* cannot move either:
`context::tests::the_selection_is_unchanged_when_a_hidden_test_is_added_or_removed` walks a real tree
(with the tempdir and the real `walk_candidates`), adds a F2P-shaped hidden test that imports the
touched module — the strongest reason a test could ever be shown, and the test asserts it *would*
be shown, so the scenario is not vacuous — and removes it again, then asserts the entries shown for
the visible tree are equal in path, score, reason, bytes, tokens, order and body across all three
walks, in both repair and review mode. A hidden test that could reorder, displace or re-cut a
visible file would make the selection a channel for the answer.

Landed in **item 4** — provenance, on top of item 3 at `eb811d2`. `ContextSelection::provenance_line`
renders one **space-free, newline-free** `key=value` line — the session log is a one-record-per-line,
space-delimited store, so a value that carried either would split a row — naming the mode, the role(s),
the budget, the file and tree counts, the dropped-for-budget count, the total bytes/tokens, then every
selected file as `path@score[reasons]shown:bytesB/tok` (reasons joined by `+`, `shown` is `full` or
`outline`) and the prune ledger as `path:reason` for everything `walk_candidates` skipped and why. The
driver records it once per run on a new run-level `context.select` row **and** appends it as
`context_selection=` on `run.summary`, so a paper table read from `run.summary` alone can see which
files a run was shown. All four role prompt versions are bumped (`CREATOR_PROMPT_VERSION` `3`,
`ATTACKER_PROMPT_VERSION` `10.localized-source-context`, `INTENT_AUDITOR_PROMPT_VERSION`
`6.localized-source-context`, `DIAGNOSTICIAN_PROMPT_VERSION` `2`): the system text did not change, but
what each role is *shown* did, and two runs whose roles read different code are different experiments.
Pinned by `context::tests::the_provenance_line_is_one_space_free_line_of_every_selected_file` (the
builder) and the `context.select` / `context_selection=` assertions in
`patch_mode_three_attempt_run_writes_multi_record_session_log` (the write to disk).

Landed in **item 5** — the gate measures the run's selection, on top of item 4 at `bd633bd`. The gate
used to *re-derive* what the run would send: `context.py` carried a Python mirror of the walk
(`SOURCE_EXTENSIONS`, `SKIP_DIRS`, `MAX_FILE_BYTES`, `MAX_TOTAL_SOURCE_BYTES`), of the token estimate
(`BYTES_PER_TOKEN_ESTIMATE`, `estimate_tokens`, `ConcatStats`) and of the prompt frame
(`PROMPT_FRAME_TOKENS`), and the only thing holding the two implementations together was a test
asserting they agreed. A gate that measures a *different* prompt from the one the run sends is worse
than no gate: it can pass a bug the run then refuses, or refuse one it would have accepted, and the
reason it prints describes neither. The fix is that there is now one implementation and the gate
calls it:

- **Rust.** `context::select_for_role(tree, issue, patch, role, budget)` is the single entry point
  (`walk_candidates` → `select_review_context` / `select_repair_context` → the greedy budget fill), and
  a new client subcommand on the same binary exposes it: `vanta_m3 --select-context --json --repo
  TREE --issue ISSUE [--patch PATCH] --select-role ROLE --select-budget N` prints a
  `SelectContextReport` (the mode, the selected files with score/reasons/bytes/tokens/shown, the tree
  and prune counts, `dropped_for_budget`, the issue and prompt-frame token counts, the estimate and
  `fits`). Its surface is *exactly* a real run's inputs and nothing else — there is no flag for
  `FAIL_TO_PASS`, a hidden test, the fix commit, the fix diff or `bug.info`, and
  `vanta_m3::tests::the_select_context_parser_has_no_flag_for_the_answer` walks the declared flag set
  (`SELECT_CONTEXT_FLAGS`) so a future flag cannot add one by accident. `parse_args` was split into a
  testable `parse_args_from(&[String])`, and the subcommand runs before the banner so it writes JSON
  and nothing else.

- **Python.** `context.py`'s mirror is **deleted**; `select_context()` shells out to that binary,
  writes the issue text to a temp file and parses the JSON, and `precheck` now needs either an
  already-computed `selection=` or the `vanta_bin=` to shell out to — there is deliberately **no
  fallback** (`ContextError: … it has no walk of its own to fall back on`), so a caller that forgot
  to thread the binary fails loudly instead of silently measuring something else. The rejection
  reasons are the selection's, not the repo's: `no_prompt_budget`, `prompt_exceeds_budget`, and the
  new `selection_dropped_for_budget` (the ranking chose a file the budget could not take even as an
  outline — the one case where the run is shown *less* than the ranking decided).
  `TestContextConstants` pins the plumbing: `ctx.SELECT_ROLE` must equal the Rust
  `REPAIR_SOURCE_ROLES`, and the mirror's names must not come back (`not hasattr(ctx, …)`).

- **`prepare` refuses a stale binary.** Because the gate is now the run's own code, `prepare` resolves
  `vanta_bin`, calls the existing `runner.stale_binary_warning`, and **raises** when the binary was
  built from an older commit than `HEAD` — a warning would be too late, since the number it is about
  to write down *is* the gate's own output and no later `run`-time warning could undo it. Pinned by
  `TestStaleBinary.test_prepare_refuses_to_size_the_gate_against_a_stale_binary`.

- **`localization_hit` is computed after a run, from evidence.** `runner.localization_hit(prep,
  context_selection)` takes the fix patch's touched **source** paths (`fix_touched_paths` drops the
  test hunks first — a test path could never be shown, and counting one as a miss would report every
  bug as unlocalized) and asks whether any of them is among the paths in the `selected=[…]` list of
  the run's recorded `context_selection=` line (`selected_paths`). It is `None`, not `False`, when the
  question cannot be answered — no `context_selection` (a batch from before item 4) or a fix that
  touches only tests — so "no evidence" never pools with "the selection missed". It is written per run
  in `build_record` and pooled in the digest as `hit` / `miss` / `unknown` per label. Pinned by
  `TestLocalizationHit`.

The harness suite goes from 917 to **931** tests, all passing; `cargo test --lib` is 338 (the two
microVM-booting tests skipped on the host, as before) and `cargo test --bin vanta_m3` is 5.

**Expected effect on the re-run.** The 10-of-11 `source_truncated_by_cap` rejections were a
*whole-repo* measurement, and item 5 changes the gate to measure the selection, so the number to
watch is not "how much of the repo is in the prompt" but "was the bug's own file selected". The
selection now asks three separate questions — does the issue name the file's path or module, does it
carry a traceback frame into the file, does it name a `def`/`class` the file defines — and answers
*none* of them with hidden information, so a bug whose issue text says nothing about where the code
is will still fall back to the tree plus whatever its one-hop neighbours reach. The per-bug
predictions, with each bug's top-ranked files and the reason each was picked, are recorded in
`runs/bugsinpy/measure-dev-1/selection_expectation.md` (item 6, below).

Landed in **item 6** — the ranking measured, on top of item 5 at `4cb560b`. Items 1 and 2 had no test of
their own; five are added to `context::tests`:

- `the_ranking_orders_by_reason_weight_and_breaks_ties_by_path` — the ranking pinned as a **unit**, on
  hand-built candidates: a file scores the **sum** of the reasons it was picked for (a file named by path,
  named again by a traceback frame and defining the symbol the issue names scores 900 = 400+300+200), a
  stronger reason is never displaced by a weaker one, and a tie is broken by path (`pkg/alpha.py` 200 =
  `pkg/beta.py` 200, ordered by path; `pkg/helper.py` 100 as the only neighbour; `pkg/unrelated.py`
  absent because no reason reaches it). The order is total and stable.
- `the_budget_fill_shows_small_files_whole_and_big_files_as_an_outline` — the fill is what makes the cap
  safe: a small file goes in `Full`; a file over `MAX_FILE_BYTES` goes in as an `Outline` (smaller than
  the source, keeping `def entry():`) rather than being skipped, and `dropped_for_budget` stays empty.
- `the_budget_fill_reports_a_file_it_had_to_drop_instead_of_hiding_it` — the one case where the run is
  shown *less* than the ranking chose (a budget the file tree has already spent) names the dropped file
  in `dropped_for_budget`, which is what the gate's `selection_dropped_for_budget` verdict reads.
- `the_buggy_file_is_found_from_a_traceback_in_the_issue` — the **integration** test: a real tree is
  walked through `select_for_role` (the `--select-context` entry point), the bug's own file is selected
  from a traceback frame in the issue with reasons `[Path, Traceback]`, the repo file tree travels
  alongside it, and `render()` carries both the `# REPOSITORY FILES (paths only)` listing and the
  `# FILE: …` body.
- `review_selection_follows_a_real_patch_to_its_imports_and_tests` — the **review-mode** integration
  test, also through `select_for_role` so the patch-splitting seam is exercised: the patch's touched file
  is `Full` with reason `[PatchTouched]`, its one-hop `from .helper import helper` arrives as an
  `ImportNeighbour`, the visible test that imports the touched module arrives as `Test`, and files the
  patch does not reach stay out.

The **per-bug expectation** is [`selection_expectation.py`](file:///Users/kyu-yi/vanta/runs/bugsinpy/scratch/selection_expectation.py)
(a hand-run script, imported by nothing) →
[`selection_expectation.md`](file:///Users/kyu-yi/vanta/runs/bugsinpy/measure-dev-1/selection_expectation.md).
It shells out to the real `target/debug/vanta_m3 --select-context --json` through `ctx.select_context`, so
every row is the run's own ranking and not a mirror; the per-bug JSON and the issue text each row measured
are under `runs/bugsinpy/scratch/selection_expectation/`. Its one approximation is the issue text: under
`--measure-only` there is no frozen text (the gate returns before `build_issue`), so each bug's text is
derived with the real `issue` module from its fix commit message — a **superset** of the frozen text, so
each selection is an **upper bound** (a real run can show fewer files, never more). `PySnooper/3` also has
a frozen text on disk; measuring both isolates the approximation from the ranking, and both are empty, so
the empty selection is the *text's* property, not the derivation's.

**The result, stated plainly.** All 11 selections fit the 163584-token budget and **nothing is dropped for
the cap**, so the `source_truncated_by_cap` rejection that took 10 of 11 out of the pilot is gone — that was
the change's purpose and it holds. But only **2 of 11** bugs localize (`pandas/1` and `fastapi/2`, each
showing 1 of the files the fix touched) and **7 of 11 selections are empty**. The mechanism is not the
limit; the issue text is. BugsInPy's issue text is the *fix commit message*, and for most of these bugs
that is a sentence of English prose — `fix calling method on wrong object` (luigi/1), `FIX: do not let
no-op monkey patches to renderer leak out` (matplotlib/1) — naming no path, no traceback frame and no
module-level symbol, so the ranking has nothing to rank and the prompt is the file tree alone. §1.6.2's
prediction that such a bug "will still fall back to the tree plus whatever its one-hop neighbours reach" is
too optimistic: with **zero** seeds there is no neighbour hop, so the fallback is the bare tree. A
diagnostic on `PySnooper/3` confirms the other direction — given the frozen text plus one sentence naming
the touched path, the selector shows `pysnooper/pysnooper.py` @400 [path] and its four @100
[import_neighbour] files, exactly the behaviour `the_buggy_file_is_found_from_a_traceback_in_the_issue`
pins. The finding is recorded honestly: the cap fix is necessary and done, and it is not sufficient,
because the issue text the harness supplies (a fix-commit message) is what a repair run is left to
localize from.

#### 1.6.3 The fallback a seed-less selection needed — **superseded by §1.6.6**

> **Superseded.** The fallback below was implemented, measured, and removed. It was the wrong answer:
> a seed-less selection is not a context, and filling it with the tree's own files in walk order produced
> a prompt that was a listing with the top of the alphabet pasted in — 11 of 11 dev bugs, every file at
> score 10 under one reason, `localization_hit 0 / 11`. §1.6.6 records what replaced it (nothing: the
> selection stays empty and the gate refuses it). The reasoning below is kept because the *problem* it
> names is real and still governs the refusal.

Measured on the second measure-only run (`runs/bugsinpy/measure-dev-2/context_report.json`), all 11 dev
bugs passed the gate with `file_count 0` and `source_bytes` **exactly** equal to `tree_bytes`
(`PySnooper/3`: tree 9 paths / 204 B, source 204 B). That equality is the arithmetic of an empty
`files` list — `ContextSelection::total_bytes` is `selected_bytes + tree_bytes`, so with nothing
selected the "context" is the tree listing and its size is the listing's size. The gate was therefore
passing every bug for a reason that had nothing to do with a context: **a listing of paths always
fits**. Item 1's diagnosis, run with exactly a real run's inputs (`--repo TREE --issue ISSUE
--select-role … --select-budget 163584`, `n_ctx = 262144`), separates the two possible causes:

| bug | case | mode | files | sel B | tree B | issue tok |
|---|---|---|---|---|---|---|
| `PySnooper/3` | no `--issue` | repair | 0 | 204 | 204 | 0 |
| `PySnooper/3` | the built issue text | repair | 0 | 204 | 204 | 38 |
| `matplotlib/1` | no `--issue` | repair | 0 | 45118 | 45118 | 0 |
| `matplotlib/1` | the built issue text | repair | 0 | 45118 | 45118 | 54 |
| `PySnooper/3` | an issue naming `pysnooper/pysnooper.py` | repair | 5 | 11053 | 204 | — |
| `PySnooper/3` | `bug_patch.txt` as `--patch` | review | 5 | 10849 | 0 | — |
| `matplotlib/1` | `bug_patch.txt` as `--patch` | review | 5 | 213510 | 0 | — |
| `pandas/1` | `bug_patch.txt` as `--patch` | review | 47 | 654090 | 0 | — |

Two answers, both from the run's own selector. **The issue text does reach it** in measure-only mode —
`issue_tokens` is 38 and 54, not 0, because a measure-only prepare still shells out to
`vanta_m3 --select-context` (with `""` for the pre-strip tree, then the real text for the final tree),
so the empty selection is not a measure-only artifact. **And a real run selects no source either**: the
authoritative gate (`prepare.py:1220`, with `prep.issue.text`) produces the same zero files. The code
path is `precheck` → `select_context` (shell-out) → `select_for_role` → `select_repair_context` →
`fill`: `select_repair_context` marks `Path`/`Traceback`/`Identifier` on candidates the issue text
names, and with **no seed** the one-hop `ImportNeighbour` loop has nothing to expand from, so `fill`
receives an empty `reasons` map and selects nothing. `selected_bytes == tree_bytes` is the arithmetic
of that emptiness, not a measurement of a context.

(The selector is not broken: given the issue *plus* one sentence naming the touched path, or given the
patch under review, it ranks and includes real source — `pysnooper/pysnooper.py` @700
[path+traceback] plus its neighbours; `matplotlib/1`'s three patch-touched files, one of them 63 KB as
an outline; `pandas/1`'s touched file plus 46 neighbours and tests. §1.6.2's ranking works, and it is
what a repair run does *not* get to use, because BugsInPy's issue text is a fix-commit message.)

The fix is at the one place every mode shares — `fill` — and it is a fallback, not a filler:

- **`SelectionReason::Tree`, weight 10.** A ranking that reached **no** file at all falls back to the
  tree's own files, in walk order (the walk sorts the candidate list by path, so the order is as
  deterministic as any other ranking), under a reason of its own. It is the weakest weight on the
  ladder (`PatchTouched` 1000 > `Path` 400 > `Traceback` 300 > `Identifier` 200 > `ImportNeighbour`
  100 > `Test` 80 > **`Tree` 10**) because it says only "this file is in the repository", and it is
  *named* rather than anonymous so a report can still tell a file the issue named from a file that
  merely exists. It fires in **both** modes, because both go through `fill` — a review run whose patch
  touches nothing selectable has the same problem a repair run has.
- **It is a fallback, not a filler.** When the ranking reached a file, nothing is added: a bug whose
  issue names one path still shows that path and its neighbours, not the tree as well. Pinned by
  `context::tests::a_ranking_that_reaches_no_file_falls_back_to_the_trees_own_source`, which asserts
  both directions plus the two properties below.
- **A file the fallback could not fit is not a *drop*.** `dropped_for_budget` means "the ranking chose
  this and the budget refused it" — the one case where a run is shown less than its ranking decided,
  and the case the gate's `selection_dropped_for_budget` reads. The fallback does not choose anything
  (the walk does), and the paths it could not fit are already in the prompt as the tree listing, so
  recording them would turn "the repository is bigger than the window", which is the normal case, into
  a rejection of every large project. A file over `MAX_FILE_BYTES` still goes in as an outline as
  before.

#### 1.6.4 The report must name the files, not only count them

`context_report.json` recorded the size of the selection and never its contents: `file_count`,
`source_bytes`, `tree_bytes`, `tree_count`, `dropped_for_budget`, `reasons`. A reader could therefore
not tell a selection of nine files from a selection of none, because both were described by the same
arithmetic — `selected_bytes` is `sum(files.bytes) + tree_bytes`, so a listing and a real context differ
only by whether `source_bytes` is *larger* than `tree_bytes`, and nothing in the report said a word
about which files produced the difference. That is how the second measure-only run was read as 11 clean
passes.

Each bug record now carries, alongside the counts it already had:

- **`select_role`** — the role the context was *selected* for (`SELECT_ROLE`, the union of the repair
  roles), which is what the run's own `context.select` row names. The record's `role` field stays the
  split role (`pilot`/`reserve`); the two are different questions and a report that conflated them could
  not be lined up with a run.
- **`selected_files`** — one entry per file, in the order the prompt puts them, each with `path`,
  `rank`, `score`, `reasons`, `rules`, `bytes`, `tokens` and `shown`. `reasons` is the ranked vocabulary
  and is what makes the ranking auditable: `patch-touched` (1000) > `path` (400) > `traceback` (300) >
  `identifier` (200) > `import-neighbour` (100) > `test` (80). There is no seventh entry: §1.6.6 removed
  the `tree` reason, so every reason in the vocabulary is a signal and a `reasons` list that is empty is
  impossible. `shown` is `full`, `outline` or `path` (§1.6.9), so a file that contributed only its
  signatures — or only its path — is not counted as if its body were in the prompt. `rank` is the file's
  1-based place in score order, restating the array's order for a report read on its own; like `rules`
  (§1.6.9), it is recorded for readers and nothing acts on it.
- **`rules`** — per signal, the rule that matched it, keyed by the signal's own name in `reasons`:
  `{"path": "exact"}` or `{"identifier": "camel", "path": "blob-url"}`. A reason alone says *which*
  input named the file; it is not enough to audit one, because `path` (400) covers five rules —
  `hunk`, `exact`, `basename`, `module` and `blob-url` — and a file the issue names outright is a
  different finding from one that matched a bare basename in another directory. The rules are the same
  ones §1.6.2 item 2 sets out: `identifier` records `snake`/`digit`/`camel` for shape and
  `backtick`/`call`/`attribute` for the written form, `import-neighbour` records `import`, `test`
  records `covering-test`, and `patch-touched` records `hunk`. Nothing reads `rules`: no weight and no
  gate depends on it, so a report written before the selector recorded it stays readable and the audit
  (§1.6.5) prints the signal without its rule rather than refusing a selection it can already judge.
- **`files_bytes` / `files_tokens`** — what the file content cost, *separately* from `tree_bytes` /
  `tree_count`, which are the listing. `source_bytes` is kept as the selector's own total
  (`files_bytes + tree_bytes`) for continuity with earlier reports, so the identity
  `files_bytes + tree_bytes == source_bytes` is checkable in the file itself. With these two fields a
  listing is visible as what it is: `files_bytes == 0` and `source_bytes == tree_bytes`.
- **`prompt_tokens_estimate`** — the selection's own total plus the issue text and a fixed frame, from
  `context::prompt_tokens_estimate`, and the only number `prompt_exceeds_budget` acts on. It is a
  function rather than an expression in the binary because the selector's total *already* carries the
  listing (`selected_tokens` is `files + tree`, §1.6.4), and the expression added `tree_tokens` a second
  time: one extra listing per selection, on the corpus's larger trees on the order of ten thousand
  tokens, which is a large part of the margin that decides an over-budget verdict.

The `not_measured` record carries the same keys with nulls (and an empty `selected_files`), so a reader
never has to branch on the schema to ask which files a bug was shown.

#### 1.6.5 The ranking audit — offline, dev split only, and never an input

§1.6.2 item 5 defines `localization_hit`: were the files the fix commit touched among those the run was
shown? The run-time answer is computed per row from the `context_selection=` provenance the driver
wrote. The same question can be asked one stage earlier and **offline**, and it should be, because the
selection is written down by then:

    python3 scripts/bugsinpy_run.py localize --batch runs/bugsinpy/<batch>

reads `context_report.json`'s `selected_files` (§1.6.4) and each bug's `bug_patch.txt` through the
harness's own reader, and prints one row per dev bug:

```
bug           files selected  top 3 files                hit
--------------------------------------------------------------
PySnooper/3   9               pysnooper/pysnooper.py …   yes
```

`top 3 files` carries each file's reasons *and* the rule that matched each one — `pkg/core.py[path:exact]`,
`sanic/app.py[path:blob-url]` — because the reason alone is not enough to audit the ranking: `path`
covers five rules and `identifier` six, so `[path:exact]` and `[path:basename]` are different findings
that read the same without the rule beside them. It is the `rules` map of §1.6.4 printed per signal. A
report whose selector did not yet record rules prints the bare reason — `pkg/older.py[path]` — never a
placeholder, so an older batch is still readable.

Three properties are the point of the command, not decoration:

- **It never runs the selector.** The audit is a *view* of a selection that was already made, so it can
  not become a term the ranking optimises — the failure mode being that a metric computed *inside*
  selection turns the dev split into training data for the ranker. Nothing in `prepare` calls it; it
  reads files, prints a table, and exits.
- **The dev split only, and it refuses anything else.** Ranking quality is a tuning number, and a tuning
  number produced over the evaluation split is how a held-out set stops being held out. `cmd_localize`
  compares the resolved `--split` against `plans/bugsinpy_dev_split.json` and raises `SplitError` for any
  other path, so `--split plans/bugsinpy_split.json` is refused rather than quietly obeyed. The
  evaluation split's ranking is reported once, at the end, from its runs.
- **One definition of "touched".** `fix_touched_paths` now delegates to `fix_touched_paths_from_patch`,
  which the audit calls directly, for the same reason §1.6.1 gave for shelling out to the real selector:
  two implementations of the same question is how the run-time metric and the audit come to disagree
  while both look authoritative. Both drop the fix patch's test hunks first — the hidden tests are
  stripped from the tree before any run sees it, so a test path is never a file a selection could show.

`hit` is three-valued, and the third value is the one that keeps the number honest. `paths_hit` returns
`None` — printed `—`, pooled as `unknown` — for a bug whose report row is `not_measured` (no selection
was made, so there is nothing to judge) and for a fix whose patch touches only test files (there is no
source path the selection could have shown). Pooling either as a miss would report a miss *rate* over
rows that never had a chance to hit.

The first reading is the honest one for the evidence that existed at the time. Run against the second
measure-only batch — whose report predates §1.6.3 — the command printed `files selected 0`, `—` and
`hit no` for all 11 dev bugs, with the footer

```
localization_hit: hit 0 / miss 11 / unknown 0 over 11 dev bug(s) — computed offline from the selection
already recorded, never fed back into it (§1.6.5)
```

That is 11 misses for the same reason §1.6.3 existed: each of those selections contained no file at all,
so none of them could have shown the fix's file. The failure that read as 11 clean passes in the report
reads as a 0-of-11 hit rate here — which is what the audit is for. After a measure-only rerun on the
fixed selector the same command reports the real selections.

The columns above are the ones the audit printed when this section was written. §1.6.11 widens the row to
ask a sharper question — was the fix's file shown **in full**, and what rank did the selection give it —
so `files selected` is now `files in full`, `touched (rank)` and `refused` are new columns, and `hit`
means "shown in full" rather than "among those shown". The three properties above are unchanged; only the
question and the row are.

#### 1.6.6 The tree is a listing, not a reason

The third measure-only run (`runs/bugsinpy/measure-dev-2/context_report.json`, overwritten after §1.6.3)
selects real file *content* at last, and it is still not localization. Every selected file in all 11 bugs
scores **10** with `reasons ["tree"]` — no `path`, no `traceback`, no `identifier`, no
`import-neighbour`, no `test`, anywhere — and the files arrive in path order: `fastapi/1` starts at
`docs/en/docs/js/chat.js`, `matplotlib/1` at `doc/conf.py`, `pandas/1` at `asv_bench/benchmarks/...`.
That is the old capped repository dump with a bigger cap:

| bug | files | src B | tree paths | budget | est tok | verdict |
|---|---|---|---|---|---|---|
| `fastapi/1` | 381 | 653752 | 505 | 163584 | 172157 | REJECT |
| `matplotlib/1` | 285 | 653890 | 1114 | 163584 | 177860 | REJECT |
| `pandas/1` | 99 | 654208 | 1125 | 163584 | 177364 | REJECT |
| `sanic/1` | 116 | 628597 | 116 | 163584 | 160942 | ok (by 2642 tok) |
| `PySnooper/3` | 9 | 21162 | 9 | 163584 | 8345 | ok |
| *(the other six)* | 89–371 | ~654000 | 268–1114 | 163584 | 168–172000 | REJECT |

Nine of eleven exceed the budget, and `dropped_for_budget` is empty for **all** eleven — §1.6.3's third
bullet suppressed the drop ledger for fallback files, so the report could not even say that a budget had
run out.

**The rule, which replaces §1.6.3's fallback: `tree` is never a reason to include content.** The file
tree travels as paths and nothing else; content comes only from scored signals. There is no
`SelectionReason::Tree` any more, `fill` has no fallback, and a ranking that reaches no file returns no
file. Everything the fallback did was the wrong shape: it invented a reason ("this path exists"), it made
the prompt alphabetical at the top, and it spent the whole budget on it, because a fill that is told to
fill will fill.

The three consequences, all of them the point:

- **A seed-less selection stays empty.** `files == []`, so `total_bytes == tree_bytes` exactly — the same
  signature §1.6.3 read, now with the honest meaning: the prompt is the listing and the issue text.
- **The gate refuses it** (§1.6.10): a role whose selection has no grounded file is refused by name, not
  sized. This is what makes the failure visible instead of tidy.
- **A drop is a drop again.** `dropped_for_budget` records every file the ranking chose and the budget
  could not take, unconditionally, so "the budget ran out" is legible in the report.

What this does **not** fix is the underlying problem §1.6.2 item 6 named: BugsInPy's issue text is the
fix commit message, and for most of these bugs it names no path, no frame and no symbol. Removing the
fallback does not create localization out of a text that carries none — it stops pretending there is
some. §1.6.7 prints exactly what the selector extracts from those texts.

#### 1.6.7 What the reading extracts, and the top of what it reaches

§1.6.6 refused the fallback and left one question open: a thin selection has two causes — the text names
nothing, or it names something no candidate defines — and the report carried only the ranking, so a
reader could not tell them apart. `signals.md` is the answer, and this section is what it contains.

**The extraction is recorded beside the selection.** `--select-context --json` carries `signals`, built
from the selector's own readers — so the report cannot describe a reading the ranking did not make:

- **`paths`** — `issue_path_tokens`: the tokens of `issue_tokens` that carry a `/` or a source extension.
  [VANTA-WHY] `issue_tokens` is every word the `path` and `traceback` signals match against, and it is
  mostly prose — `matplotlib/1`'s text contains `figure`, `the` and `version`, and matching those against
  a path is what the *rules* are for, not what the *extraction* is. This reader is the narrower set, so
  the table can print what the issue named as a file rather than hundreds of words that happened to reach
  a basename. It is reporting only: nothing ranks with it, and the signals still match against the whole
  of `issue_tokens`.
- **`traceback_files`** — the file paths a traceback inside the issue names (`issue_traceback_files`),
  the input the `traceback` signal's `frame` rule reads.
- **`identifiers`** — every name the issue writes *as code*, each with the rule that made it one
  (`issue_identifiers`: `snake`, `camel`, `attribute`, `call`, `backtick`, `digit`). The rule travels
  with the name because the name alone does not say why it qualified.

`prepare` writes the block into `context_report.json` as `issue_signals` (§1.6.4) and renders it into
`signals.md`: per bug the signals in weight order, above the top `SIGNALS_TOP_N = 10` files with rank,
score, form, path and `reason:rule`.

[VANTA-WHY] **Nothing ranks with it.** `signals` is a view of a selection that was already made — the
same property, for the same reason, that §1.6.5's audit is offline — so it cannot become a term the
ranking optimises, and it is never fed back. It is read tolerantly like `rules` and `rank`: a
`context_report.json` written before the field existed reads as `None`, never as an empty extraction,
because "the text named nothing" and "no reading was recorded" are different findings.

**What the six dev texts extract** (n_ctx 262144, budget 163584 tok, `runs/bugsinpy/measure-dev-5`):

| bug | gated text | `path` (400) | `traceback` (300) | `identifier` (120–200) |
|---|---|---|---|---|
| `fastapi/1` | 169 B, 43 tok | `github.com/…/pull/1166` | — none — | 3: `Documented` (camel), `These` (camel), `com` (attribute) |
| `luigi/1` | *not measured* — excluded before the precheck, `issue_source=none` (§2.4) | | | |
| `matplotlib/1` | 1994 B, 499 tok | `tmp/image.png`, two `user-images…png`, `x_size/100`, `y_size/100` | — none — | 48 |
| `pandas/1` | 2127 B, 532 tok | — none — | — none — | 30 |
| `sanic/1` | 2746 B, 687 tok | the `blob` URL of `sanic/app.py`, the issue URL, the PR URL | — none — | 49 |
| `PySnooper/3` | 234 B, 59 tok | the `blob` URL of `pysnooper/pysnooper.py` | — none — | 8 |

**And the top ten of each selection**, as `signals.md` prints it. `why` is `reason:rule` per §1.6.4; a
bare reason means the file's rules were not recorded, never a placeholder.

`fastapi/1` — 11 files, estimate 17066 of 163584 tok, `bm25` alone, nothing dropped:

| rank | score | form | file | why |
|---|---|---|---|---|
| 1 | 40 | full | `docs_src/advanced_middleware/tutorial002.py` | `bm25:bm25` |
| 2 | 39 | full | `docs_src/cors/tutorial001.py` | `bm25:bm25` |
| 3 | 35 | full | `docs_src/custom_response/tutorial006.py` | `bm25:bm25` |
| 4 | 27 | full | `docs_src/extending_openapi/tutorial001.py` | `bm25:bm25` |
| 5 | 27 | full | `docs_src/security/tutorial002.py` | `bm25:bm25` |
| 6 | 25 | full | `docs_src/security/tutorial003.py` | `bm25:bm25` |
| 7 | 22 | full | `fastapi/openapi/docs.py` | `bm25:bm25` |
| 8 | 20 | full | `scripts/gitter_releases_bot.py` | `bm25:bm25` |
| 9 | 17 | full | `docs_src/security/tutorial005.py` | `bm25:bm25` |
| 10 | 12 | full | `docs_src/security/tutorial004.py` | `bm25:bm25` |

`matplotlib/1` — 32 files, estimate 163575 of 163584 tok, **165 dropped**, `identifier` alone:

| rank | score | form | file | why |
|---|---|---|---|---|
| 1 | 200 | full | `lib/matplotlib/__init__.py` | `identifier:snake` |
| 2 | 200 | outline | `lib/matplotlib/axes/_base.py` | `identifier:snake` |
| 3 | 200 | outline | `lib/matplotlib/figure.py` | `identifier:snake` |
| 4 | 200 | full | `setupext.py` | `identifier:camel` |
| 5 | 191 | outline | `lib/mpl_toolkits/mplot3d/axes3d.py` | `identifier:snake` |
| 6 | 186 | full | `lib/matplotlib/artist.py` | `identifier:snake` |
| 7 | 186 | outline | `lib/matplotlib/axes/_axes.py` | `identifier:camel` |
| 8 | 186 | full | `lib/matplotlib/backend_managers.py` | `identifier:attribute` |
| 9 | 186 | full | `lib/matplotlib/backend_tools.py` | `identifier:attribute` |
| 10 | 186 | outline | `lib/matplotlib/pyplot.py` | `identifier:attribute` |

`pandas/1` — 167 files, estimate 163560 of 163584 tok, **81 dropped**, `identifier` then `import-neighbour`:

| rank | score | form | file | why |
|---|---|---|---|---|
| 1 | 200 | full | `pandas/core/dtypes/common.py` | `identifier:snake` |
| 2 | 200 | outline | `pandas/core/series.py` | `identifier:camel` |
| 3 | 200 | full | `pandas/io/s3.py` | `identifier:digit` |
| 4 | 200 | full | `pandas/tests/io/test_gbq.py` | `identifier:snake` |
| 5 | 200 | full | `pandas/util/_print_versions.py` | `identifier:snake` |
| 6 | 191 | full | `pandas/io/formats/style.py` | `identifier:digit` |
| 7 | 191 | full | `pandas/tests/io/formats/test_style.py` | `identifier:digit` |
| 8 | 100 | full | `pandas/__init__.py` | `import-neighbour:import` |
| 9 | 100 | full | `pandas/_config/__init__.py` | `import-neighbour:import` |
| 10 | 100 | full | `pandas/_libs/__init__.py` | `import-neighbour:import` |

`sanic/1` — 29 files, estimate 70098 of 163584 tok, nothing dropped:

| rank | score | form | file | why |
|---|---|---|---|---|
| 1 | 600 | full | `sanic/app.py` | `path:blob-url` + `identifier:camel` |
| 2 | 600 | full | `sanic/response.py` | `path:module` + `identifier:call` |
| 3 | 200 | full | `sanic/blueprints.py` | `identifier:camel` |
| 4 | 187 | full | `sanic/handlers.py` | `identifier:attribute` |
| 5 | 187 | full | `sanic/worker.py` | `identifier:attribute` |
| 6 | 179 | full | `sanic/blueprint_group.py` | `identifier:attribute` |
| 7–10 | 100 | full | `sanic/__init__.py`, `sanic/asgi.py`, `sanic/compat.py`, `sanic/config.py` | `import-neighbour:import` |

`PySnooper/3` — 6 files, estimate 6106 of 163584 tok, nothing dropped:

| rank | score | form | file | why |
|---|---|---|---|---|
| 1 | 400 | full | `pysnooper/pysnooper.py` | `path:blob-url` |
| 2–5 | 100 | full | `pysnooper/__init__.py`, `pysnooper/pycompat.py`, `pysnooper/tracer.py`, `pysnooper/utils.py` | `import-neighbour:import` |
| 6 | 40 | full | `setup.py` | `bm25:bm25` |

**Which tree these numbers are from.** Both are the batch's own: `--measure-only` records the *early*
precheck, which `prepare` runs against `oracle_tree` to avoid three suite runs on a bug that cannot fit
anyway, and a full `prepare` replaces it with `vanta_tree` — the same source with the hidden tests
stripped (§1.6.9), so never larger. A `signals.md` written by a full prepare is therefore the smaller of
the two readings.

**What the tables show.** Read as a set, five findings:

- **Item 2's fix is visible and the ranking is right where the text is informative.** `pandas/1`'s `#1`
  is `pandas/core/dtypes/common.py`, the file that *defines* the backticked `is_string_dtype`;
  `pandas/__init__.py` fell from `#1` to `#8`, behind seven scored files, and arrives at all only as an
  import neighbour. `sanic/1`'s `#1` is `sanic/app.py` at 600 (`path:blob-url` + `identifier:camel`),
  `PySnooper/3`'s `#1` is `pysnooper/pysnooper.py` at 400, both `#1` by two independent signals.
- **`fastapi/1` is the section's first named case, and it is worse than "names no definition": its gated
  text names nothing at all.** The frozen text is 169 B and its whole body is a PR link plus two
  `[redacted]` markers — the leak gate's own record strips two names,
  `response_model_exclude_none` and `response_model_exclude_defaults`, at 71 and 70 occurrences in the
  answer and **0** in the tree Vanta may read — so the only names the extraction finds are
  `Documented`, `These` and `com`. The whole selection is therefore `bm25`'s (§1.6.2
  item 3), and it reaches FastAPI's *tutorial scripts* — this is the case BM25 exists for, and it is the
  case where a retrieval baseline is the only signal a benchmark's own text can support.
- **Two of the five selections are refused, and not for size.** `matplotlib/1` (165) and `pandas/1` (81)
  are refused as `selection_dropped_for_budget`: `est ≤ budget` holds for both — item 1's overshoot is
  gone, 167098 → 163560 on `pandas/1` — but the ranking chose more files than the budget takes even as
  paths, and §1.6.9 makes a drop a drop rather than a silent truncation. The refusal is the gate saying
  "the run would be shown a smaller context than the ranking decided on", which is true and is the point
  of the ladder; it is also the honest reason those two bugs are still not runnable, and it is the
  first thing a reader of this table should see.
- **The `camel` rule fires on English prose.** `matplotlib/1`'s 48 names include `Actual`, `However`,
  `Operating`, `Produced`; `sanic/1`'s 49 include `Additional`, `Describe`, `See`, `There`. That is not a
  bug in the extraction (a capitalized word is camel-shaped by the only rule that can be applied without
  a symbol table) — it is exactly why §1.6.2 item 2c weights identifier matches by rarity across the
  repo: `However` matches a file that contains it and scores with it, and combinations against
  `add_axes`/`set_visible` are what put `lib/matplotlib/*` on top.
- **A URL is path-shaped.** Every `path` reading above except `pandas/1`'s (none) is a URL, because a URL
  carries `/`. That is the extraction being deliberately looser than the matching: the `path` signal
  applies five rules to those tokens, and `blob-url` is the one that reads a GitHub blob link as the file
  it points at. The `rule` column is what reconciles the two readings, which is why §1.6.4 records it and
  why `signals.md` prints it beside the reason.
- **`docs_src` is paratext this repo's list does not have.** `fastapi/1`'s BM25 top ten is nine
  `docs_src/**` paths and one `scripts/**` path, all in **full** — `PARATEXT_DIRECTORIES` (§1.6.8) holds
  `docs`, `doc`, `examples`, `benchmarks` and their kin, and `docs_src/` is neither, so §1.6.8's
  extension-and-paratext test lets a tutorial's source through as a body. Worth stating where it is
  visible rather than inferred: the path-only rule is only as good as that list, and `docs_src` is one
  entry a real repository disagrees with it about.

#### 1.6.8 What a file's body may be

§1.6.6 removed the reason that meant "this file is in the tree". What it left is still too broad: a file
a *signal* reached was shown as code whatever it was. The signals are textual — a path the issue names,
a traceback frame, a symbol a file defines — and they fire on any file the walk lists, so a Python
repo's own `docs/en/docs/js/chat.js`, its `asv_bench` harness, or a `.ts` example could be shown in full
to a role whose bug is in a `.py` file. A body is the expensive half of the prompt, and for a file that
is not the project's code it can only spend budget on text that cannot hold the bug.

Two questions had been answered by one list, and they are not the same question:

- **what the walk can see** — `SOURCE_EXTENSIONS`. This decides which paths exist as candidates at all.
  It is a fact about the tree, and it is what §1.6.2 item 1's walk applies.
- **what a body may be** — this section. A file the walk listed, and a signal reached, may still
  contribute only its path.

**The rule: a file's body is carried only when**

1. **its extension is the project's own source** — `.py`, `.pyx`, `.pxd`, `.pyi` — **and it is not the
   project's paratext**; **or**
2. **the run's own inputs name that file** — `Path` (the issue names its path, basename or module),
   `Traceback` (a frame of a traceback inside the issue names it), or `PatchTouched` (the patch under
   review touches it).

Everything else a signal reaches is a path, never a body. Being in the tree is not a signal (§1.6.6),
and neither is being *reachable* from something the issue named: an import neighbour, and a covering
test of a file that is not project source, are reachable rather than named, and §1.6.6's listing is
already where their paths belong.

**The paratext carve-out (item 4).** The dev-2 run exposed a gap the extension rule alone leaves open:
`sanic/1`'s issue names `add_task`, and `examples/add_task_sanic.py` *declares* it, so the example was
shown in full at rank 4. A `.py` example passes the extension test — it is Python the project wrote —
but it is not the code a bug is fixed in: it is a demonstrator, and a demonstrator is a name's
*mention*, never its *place*. The same holds for the documentation (`docs/conf.py` is Python and is
never the bug) and for an `asv_bench` harness. These directories are the project's **paratext**:

- `example`, `examples`; `doc`, `docs`; `benchmark`, `benchmarks`, `asv_bench`.

A file under one of them is a **path only**, even when its extension is project source, unless a
**path** signal — `Path`, `Traceback` or `PatchTouched` — names it. An `Identifier` or a `bm25` match is
not enough; the escape is the same one every non-source file has, and it is deliberate that a *name*
does not open it, exactly as §1.6.8's last paragraph refuses a body to a `.js` class the issue names.
Only **directories** are tested, and only the *directory* components — the file's own name is dropped
first — so a `pkg/docs.py` module is source and a `docs/conf.py` is not. The predicate lives beside the
extension rule as `is_paratext`, and `carries_body` asks both: `names_the_file(reasons) ||
(is_project_source(path) && !is_paratext(path))`.

`.pyx`, `.pxd` and `.pyi` count as project source alongside `.py`. pandas and spacy have real bugs in
Cython files — `pandas/_libs/algos.pyx` is where this corpus's `.pyx` bugs live — and a `.pxd` is the
declaration half of the same code. `.pyi` is Python the project wrote, and `module_level_symbols`
already parses it with the Python rules; the walk did not list it, which made that branch dead on every
walked tree. All three join `SOURCE_EXTENSIONS`, so the walk can *see* them and this rule can then give
them a body. A Cython `def`/`cpdef` is parsed as the symbol it is, so an issue naming a Cython function
reaches its file by `Identifier` as well.

`PatchTouched` wins whatever the extension. In review mode the patch is the subject, so a hunk in a file
the walk lists is shown even when the extension is not project source; §1.6.2's review rule ("the
patch's own touched files") is what the review selection is *for*. A touched file whose extension the
walk does not list at all (a `.rst` changelog, a `.yaml` CI file) is not a candidate and contributes no
content: the walk's extension set decides what can be reached, and a review reads the patch itself
rather than the whole of every file it touches.

**Where it lives.** The predicate is a method on the reason vocabulary —
`SelectionReason::names_the_file`, true for `Path`, `Traceback` and `PatchTouched` — so "named by the
run's inputs" has one definition, sitting next to the weights and next to `grounds_content`, the gate's
own predicate (§1.6.10), which answers a different question: whether the selection as a whole is
grounded, not whether one file's body may be shown. `fill` consults it once per ranked file, before the
caps. A file that fails it is not selected and is **not** recorded as a budget drop: the budget did not
refuse it, and `dropped_for_budget` naming it would report the wrong cause.

The one reachable case this rule refuses is `Identifier` on a file that is not project source — a `.js`
class the issue names, say. That refusal is deliberate: a symbol is a name, not a file, and the
extension rule is the thing that keeps a repo's documentation site out of a Python bug's prompt. A
JavaScript bug would need `.js` in the project-source set; that is a corpus question, not a selector
tweak.

#### 1.6.9 The fill's ladder — whole, outline, path, and never over budget

§1.6.8 decided *whether* a reached file may carry a body at all. What it did not decide is how much of
it, and the fill had only two answers — whole or outline — with a file that fit neither silently
dropped. That is wrong in the direction that reads least obviously: a ranking that reached a monolith
first could spend what remained on one file's outline and drop every file below it, and the report said
only `dropped_for_budget: [three paths]` — "the budget ran out", with no way to see that the ranking
wanted those files and that a listing is all they got.

The fill is now a ladder of three forms, richest first:

- **full** — the whole source.
- **outline** — signatures and docstrings: §1.6.8's `outline_body`, the module docstring and imports,
  then every `def`/`class` signature with its docstring.
- **path** — the `# FILE:` header and one marker line, and no source at all.

Files are walked in score order and each takes the richest form the budget still allows. The three are
*ordered* — `ShownMode` derives `Ord` from that order — and the fill carries one `form`: the richest form
still available to the files below the current one. `form` only ever descends, and only the budget moves
it:

- a file that fits whole is shown whole, and `form` is untouched;
- a file whose whole body does not fit is priced again as an outline, and from there **every file below
  it takes an outline at best**;
- a file whose outline does not fit is priced again as a path, and from there every file below it takes a
  path at best;
- a file that does not fit even as its path is **dropped**, and named in `dropped_for_budget`.

So the run is shown a *prefix* of the ranking's forms, never a patchwork: one monotone boundary, which is
what makes "the model was shown this file" a property of the ranking rather than a per-file accident.

**The budget is the whole prompt's, so the fill reserves it before pricing the first file.** The number
`--select-budget` hands the fill is the whole prompt's, so `fill` subtracts everything that is *not* a
selected file from it first: the file-tree listing, the issue text, and `PROMPT_FRAME_TOKENS`. The issue
text and the frame are exactly what `prompt_tokens_estimate` adds back on top of the selection
(§1.6.4), so reserving them inside the fill — not merely computing them afterwards — is what makes "a
selection can never exceed the budget" true rather than nearly true. Before this the fill reserved the
listing alone, and a selection that spent all of what it was handed overshot the whole-prompt budget by
exactly `issue_tokens + PROMPT_FRAME_TOKENS`: on pandas/1 that was 167,098 estimated against a 163,584
budget, an over-budget refusal produced by the fill's own arithmetic rather than by the selection being
too large. The reservation is priced with the same `estimate_tokens` the gate's `issue_tokens` uses
(§1.6.4), so the fill's boundary and the gate's `prompt_exceeds_budget` sit on one number.

**The per-file cap is not the budget, and must not move the ladder.** A file over `MAX_FILE_BYTES`
(64 KB) is denied its *source* by the cap — that is §1.6.2 item 6's rule, and it still holds — so it goes
to an outline *on its own*, at whatever rank it sits. It must not push the files below it down: the
cascade is driven by what budget *remains*, and one oversized file the budget could still afford an
outline for is that file's own business. So the cap is applied as that file's own ceiling — the lower of
the form the ladder would allow and the cap's answer — and `form` is left exactly where it was. A small
helper ranked below a monolith still goes in whole; a small helper ranked below a file the budget could
not take in full does not, because that denial *was* the budget's.

**What is recorded.** Every selected file carries `rank`, `score`, `reasons`, `rules`, `bytes`, `tokens`
and `shown` (§1.6.4), and `shown` is which of the three forms it got. `rank` is `index + 1` — a second
statement of the array's order, recorded because the report is read on its own by analyses that sort the
array their own way. Recording the form is what lets a later audit say a bug was *shown* a file and mean
its source rather than its path.

**The path form is a body, and it costs.** A `# FILE:` header with nothing under it reads as an *empty*
file, and a model that believes a module is empty reasons about the bug as if its code did not exist. So
the path form is not header-only: it is the header and one marker line saying the budget ran out before
this file's source could be shown. It is the cheapest form, not a free one, and it is priced like any
other — a file whose header and marker do not fit is a drop, in `dropped_for_budget`, not a silent
omission.

**A path is not content.** §1.6.10's gate asks whether the selection has any of the project's content
grounded in the run's inputs, and a file travelling as a path has none. `ShownMode::carries_content` is
false only for `Path`, and `grounded` consults it: a selection spent entirely on the listing and
path-only headers is refused as `no_issue_grounded_selection`, the same refusal as a listing with no
files. The path-only file is still a *selected* file — it is what the ranking decided, and it is
recorded — it is simply not a body.

**Where it lives.** `ShownMode` and `carries_content` sit with the record; the ladder is the body of
`fill`; `path_only_body` is its own function for the same reason `outline_body` is — the marker is the
whole reason the form costs anything and it should be findable. Nothing outside `fill` branches on the
form except `grounded`, so a fourth form would be a change to the ladder and to `carries_content`, not to
a scattered set of call sites.

#### 1.6.10 The gate refuses by name, and the selector owns the name

The refusal §1.6.6 left the gate was free text — `no source selected: the selection carries no file at
all — only the repository file tree (…)` — found by a reader looking for the phrase. It also described
the wrong thing. What the gate is refusing is a selection whose content is not *grounded in the run's
own inputs*, which is a claim about the reasons a file carries, not about a list being empty.

- The reason is **`no_issue_grounded_selection`**. It is a name a reader can filter on, and both the
  §1.6.6 table and §1.6.11's audit do.
- The verdict is the **selector's own answer**, not the harness deriving one. `SelectContextReport`
  carries `grounded` — `SelectionReason::grounds_content`, the predicate on the reason vocabulary,
  applied to the files the selection carries: true when at least one selected file's content is grounded
  in the issue text or the patch under review.
- `precheck` refuses on `not selection.grounded`, so a reason that does not ground content is refused
  **without the harness learning its name** — the harness asks a question instead of holding a copy of
  the answer. That is the same reason `context.py` shells out to the selector rather than walking a
  tree (§1.6.2 item 5), and the same reason the predicate sits on the enum rather than in `precheck`.
- `Selection.from_dict` **requires** the field. A binary that does not print it is a gate that cannot
  judge, so the module fails loudly rather than reading a missing `grounded` as `false` — which would
  refuse every bug the moment the field ever dropped off the wire format.

Every reason grounds content today, so on this vocabulary `grounded` coincides with "the selection
carries a file". The predicate is what makes that a coincidence of the vocabulary rather than the
definition of the gate: a reason that names a file without grounding its body would be refused, and the
harness would not need to change to refuse it.

The over-budget half is `prompt_exceeds_budget`, acting on `prompt_tokens_estimate` and nothing else —
now a number that counts the listing once (§1.6.4), which is what makes it the same comparison the run's
own guard makes.

#### 1.6.11 The audit asks its second question — shown, or shown **in full**

§1.6.5's audit asks whether the fix's own files were *among those shown*. Shown is three forms
(§1.6.1) — full, outline, path — and a file the prompt prints as a bare tree entry, or cuts to its
signatures, is a **pointer** to the fix and not the fix. An audit that counts that as a hit reports the
ranking's *reach* as though it were the prompt's *content*, which is the one number §1.6.5 exists to
keep honest. §1.6.11 widens the row and sharpens the question.

    python3 scripts/bugsinpy_run.py localize --split plans/bugsinpy_dev_split.json \
        --batch runs/bugsinpy/<batch>

The same command §1.6.5 defined, and the same three properties: no VM, no LLM and no selector; the dev
split only, refused with `SplitError` for any other path; and never read by `prepare`, so the number
stays a view of the ranking rather than a term it optimises. Three things a row says change:

- **`files in full` replaces `files selected`**, written `in full / selected`. The count the hit is
  asked over, beside the count the ranking chose — a run that selected thirty files and showed two of
  them whole is a different finding from one that selected two, and folding them into one number is how
  a thin prompt reads as a full one.
- **`touched (rank)` is a new column**: every path the fix commit changed, each with the place the
  selection gave it and the form it was shown in.
- **`refused` is a new column**: the gate's own name for a row that was not a clean selection
  (§1.6.10), read from the report's `reasons` as the identifier-shaped token before the first colon. A
  refused selection and a ranking that reached nothing produce the same `no`, and the column is where
  the audit stops reading them as one finding — which is exactly the name §1.6.10 said a reader would be
  able to filter on.
- **`hit` is `yes` only when a touched file was shown in full.** The set `paths_hit` is handed is now
  `runner.shown_full(entries)` — the selection's `shown == "full"` rows — in place of every selected
  path. `paths_hit` itself is unchanged and reused: what moved is *which* set it is asked about, not a
  second definition of "touched", so §1.6.5's third property (one implementation for the run-time
  metric and the audit) still holds.

`touched (rank)` is **three-valued**, read the way `context.rank_of` reads the field, because its states
are three different findings and only one of them is a ranking failure:

| cell | means |
|---|---|
| `pandas/core/dtypes/common.py #1 (full)` | the ranker put the fix's file first, and the fill afforded it whole |
| `lib/matplotlib/figure.py #3 (outline)` | reached, ranked, and cut to a signature — a near miss the `hit` column does not count |
| `pkg/core.py #?` | in the selection, but the report recorded no rank (written before §1.6.9) — the tolerant mark, never a number read off the array's order |
| `fastapi/applications.py #—` | the ranking reached nothing for the file the fix changed |

Read over the dev split with `measure-dev-5` — the same report §1.6.7 reads, whose rows are the early
precheck's selection (see the note there):

| bug | files in full | top 3 files | touched (rank) | hit | refused |
|---|---|---|---|---|---|
| `fastapi/1` | 11/11 | `docs_src/advanced_middleware/tutorial002.py[bm25:bm25]`, `docs_src/cors/tutorial001.py[bm25:bm25]`, `docs_src/custom_response/tutorial006.py[bm25:bm25]` | `fastapi/applications.py #—`, `fastapi/encoders.py #—`, `fastapi/openapi/utils.py #—`, `fastapi/routing.py #—` | no | — |
| `luigi/1` | 0/0 | — | `luigi/server.py #—` | — | `issue_source=none` |
| `matplotlib/1` | 6/32 | `lib/matplotlib/__init__.py[identifier:snake]`, `lib/matplotlib/axes/_base.py[identifier:snake]`, `lib/matplotlib/figure.py[identifier:snake]` | `lib/matplotlib/backend_bases.py #17 (path)`, `lib/matplotlib/figure.py #3 (outline)`, `lib/matplotlib/tight_layout.py #28 (path)` | no | `selection_dropped_for_budget` |
| `pandas/1` | 24/167 | `pandas/core/dtypes/common.py[identifier:snake]`, `pandas/core/series.py[identifier:camel]`, `pandas/io/s3.py[identifier:digit]` | `pandas/core/dtypes/common.py #1 (full)` | yes | `selection_dropped_for_budget` |
| `sanic/1` | 29/29 | `sanic/app.py[path:blob-url+identifier:camel]`, `sanic/response.py[path:module+identifier:call]`, `sanic/blueprints.py[identifier:camel]` | `sanic/app.py #1 (full)` | yes | — |
| `scrapy/1` | 0/0 | — | `scrapy/spidermiddlewares/offsite.py #—` | — | — |
| `spacy/4` | 0/0 | — | `spacy/cli/converters/conllu2json.py #—` | — | — |
| `fastapi/2` | 0/0 | — | `fastapi/routing.py #—` | — | — |
| `luigi/2` | 0/0 | — | `luigi/contrib/beam_dataflow.py #—` | — | — |
| `matplotlib/2` | 0/0 | — | `lib/matplotlib/axes/_axes.py #—` | — | — |
| `PySnooper/3` | 6/6 | `pysnooper/pysnooper.py[path:blob-url]`, `pysnooper/__init__.py[import_neighbour]`, `pysnooper/pycompat.py[import_neighbour]` | `pysnooper/pysnooper.py #1 (full)` | yes | — |

and the footer, verbatim:

```
localization_hit: hit 3 / miss 2 / unknown 6 over 11 dev bug(s) — a hit is a file the fix touched
shown **in full** (not merely selected, nor outlined), computed offline from the selection already
recorded and never fed back into it (§1.6.11)
```

**What the rows show.**

- **The item-2 fix is visible as a *rank*, not only as a first place.** `pandas/1`'s
  `pandas/core/dtypes/common.py` — the file that *defines* `is_string_dtype`, the name the issue
  backticks — is `#1 (full)`, and the row is a `hit`. §1.6.2 made the defining file outrank the files
  that merely mention the name; the audit now prints the rank that says so, and the generic
  `pandas/core/arrays/*.py` files and `pandas/__init__.py` (the top-level-package-as-a-path case §1.6.2
  item 2a removed) are no longer the top of the row.
- **`sanic/1` and `PySnooper/3` hit at rank 1, by different rules.** `sanic/app.py #1 (full)` on
  `path:blob-url + identifier:camel`; `pysnooper/pysnooper.py #1 (full)` on `path:blob-url`. These are
  the two bugs the dev prepare was already right about, and here is why: both were reached by a signal
  that names the file's own module, not by a common word.
- **`fastapi/1` is a full prompt and a total miss, on a selection nothing refused.** `11/11` shown in
  full, every one of the fix's four files `#—`, and `refused` empty — so the `no` is the ranking's own,
  not a gate's. Eight of its eleven full files are `docs_src/**` tutorials and two more are `scripts/**`
  (`gitter_releases_bot.py`, `docs.py`), the BM25 fallback (§1.6.2 item 3) filling the prompt from prose
  since `docs_src` is not a directory `PARATEXT_DIRECTORIES` lists (§1.6.8). A row that read only
  `files selected 11` would call this a healthy selection; the `touched` column is what says the prompt
  held none of the fix.
- **`matplotlib/1` is a reached-but-not-afforded miss, and it is the refused case beside `fastapi/1`'s
  clean one.** The fix's files *were* ranked — `figure.py` at **#3**, `backend_bases.py` at **#17**,
  `tight_layout.py` at **#28** — and all three were shown as an outline or a path. Under §1.6.5's
  "among those shown" they would read as three hits for a prompt that carried none of their bodies. The
  row's `selection_dropped_for_budget` says why: the same budget §1.6.7 measured (165 files dropped at
  est 163575 against 163584). The ranking found the fix and the fill could not pay for it — §1.6.9's
  finding read from the other end — and the `refused` column is what keeps that from reading as the
  ranker's fault.
- **`pandas/1` is a hit *and* a refusal**, and the two columns have to be read together: `#1 (full)` is
  the ranking scoring a clean hit, and `selection_dropped_for_budget` is the fill dropping 81 of the 167
  files it chose after the fix's own file was already in. A refusal is not a miss; a hit is not an
  approval.
- **Six rows are `0/0 … #— … —`, and they are `unknown` rather than misses.** `luigi/1` is
  `issue_source=none` (§2.4) and the five reserve bugs are not measured by a measure-only run (no frozen
  text), and their `refused` cell is `—` where the recorded reason is a *sentence* rather than a name
  (`frozen issue report: …`) — the audit prints the honest unknown mark instead of a first word
  pretending to be a name. They cannot be judged, so they are not counted: the number is a 3-of-11 hit
  rate, not a 3-of-11 with six rows read as misses.
- **The audit reads the report it finds, whichever prepare wrote it.** `measure-dev-5` is a
  measure-only batch, so its record is the *early precheck's* selection against `oracle_tree` (§1.6.7's
  note). A full prepare replaces that record with the post-strip `vanta_tree` reading, which is never
  larger and so never a better hit rate; the audit prints whatever the report holds and never re-runs a
  selector to improve it.

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

### 2.4 Superseded: the text is the project's own issue report

**This section replaces the recommendation in §2.2.** Option A above is withdrawn, and the reason is
the one §2.3 could not fix.

#### 2.4.1 Why the fix commit's message cannot be the text

The message of the fix commit is written by the person who fixed the bug, *after* they fixed it. On
the development split it names the fix location outright for **7 of 11** bugs — `is_string_dtype`,
`allowed_domains`, `conllu2json`, `beam_dataflow`, `named_response_middlware`,
`response_model_exclude_defaults`, `monkey patches to renderer`. Those are not incidental mentions;
they are the symbol the Creator is being asked to find.

The leak gate (§2.2 step 4) cannot take them back, and this is structural rather than a tuning
problem. The gate strips a token only when it appears in the fix diff **or** the hidden test **and
nowhere in the tree the Creator may read**: `tree_tokens.get(token, 0) == 0`. A symbol the developer
fixed is, by definition, *present in the buggy tree* — that is what "the bug is in this function"
means. So the gate's rule ("this came from the answer") does not fire on "this came from the
answer's *author*". Raising the gate's strictness does not help; the condition it tests is false for
exactly the tokens that matter.

The text the model should read is the **report the bug was filed against** — what a real repair run
gets, and what a reader of the paper would expect the prompt to be. So it becomes the source.

#### 2.4.2 Resolution — host side, in `prepare`, cached

`issue_source.py` (`referenced_numbers`, `resolve`, `GitHubClient`) implements a one-off, cached
step:

1. Read the fix commit's message and pull the numbers it references, most explicit phrasing first:
   cross-repo `owner/repo#N`; `close[sd]/fix(es/ed)/resolve[sd] #N`; `Merge pull request #N`;
   `Issue N` / `issues #N`; and the squash-merge `(#N)`. Order is the point — for `thefuck/3`,
   `Fix #869` is the report and `(#917)` is the pull request it came through, and the closing
   keyword is what the committer meant.
2. Ask the GitHub API whether that number is an issue or a pull request.
3. For a pull request, follow it to the issue it closes: its body's own closing keywords first, then
   the `cross-referenced` events on its timeline. **The pull request's body and diff are never used
   as issue text** — a pull request *is* the fix, so its prose is the answer's prose. The body is
   *read* only to find a number, and from the payload already in hand: for a pull request
   `/issues/N` and `/pulls/N` carry the same body, so the second request would buy nothing but rate
   limit.
4. A cross-repo reference is recognised but is not a candidate for this bug: the text has to be this
   project's own report, so `oh-my-fish/plugin-thefuck#11` is reported and set aside, and is masked
   out of the same-repo scan so it cannot be re-read as `thefuck#11`.

**No `gh`, no `git` fallback.** `gh` is a second implementation of the same API behind a binary that
may not be installed and whose output format is not the API's; `git` cannot answer the question at
all, and every `git`-shaped shortcut would be a way for the issue text to come from somewhere other
than the cached, hashed API response. One implementation, one record. A test asserts this by
grepping the module's own source for `gh` and for `subprocess`.

**Rate limit and token.** Unauthenticated by default: 60 requests per hour, which the on-disk cache
and the backoff make sufficient — a bug costs at most a handful of requests, once, ever. On
`403`/`429` the client waits out `Retry-After` or the `X-RateLimit-Reset` and retries **once**; a
wait beyond the ceiling (3600 s) is an error that names `VANTA_GITHUB_TOKEN` rather than hanging.
`VANTA_GITHUB_TOKEN`, if set, is used read-only for public repositories and is never written to the
cache, to a log, or into a cache key.

#### 2.4.3 The frozen artifact — identical by construction

`prepare` writes `plans/bugsinpy_issues/<project>_<bug>.md` and **commits it**. It carries the
title and body — the issue text proper — and, separately, the issue's comments **up to the fix
commit's date and never anything after it**. The cut is the point: the fix commit's date is the
last moment at which a comment says what the reporter knew; a comment posted after the fix can
only be about the fix.

**The comments are hints and the title and body are the issue.** This is the SWE-bench
convention and it is the right one on the merits: `problem_statement` is title + body, and a
comment is a *hint* (`hints_text`), which SWE-bench does not show by default. The reason is
concrete rather than conventional. On a real report a maintainer often arrives before the fix and
names the cause — `matplotlib/1`'s comments name `backend_agg.py#L411-L421` and describe the
defect in the maintainer's own words — and that is the answer, in the prompt, in prose the token
gate cannot take back because every symbol in it is present in the buggy tree. So the comments are
frozen, gated, and kept in their own section, and **the default setting never shows them**; only
the explicit with-hints ablation does (§2.4.4).

The file has five sections behind one header, in this order:

- `<!-- gate -->` — the gate record as JSON: the removals, their classes, and the ungated prose for
  **both** the content and the hints (§2.4.4). Legible only next to what was removed, which is why
  the raw sections exist.
- `<!-- raw -->` — the report's title and body exactly as the API returned them, ungated.
- `<!-- content -->` — the **gated** title and body, and the only section the default setting reads.
- `<!-- raw-hints -->` — the comments exactly as the API returned them, ungated.
- `<!-- hints -->` — the **gated** comments. Frozen and hashed like everything else, and read by
  nothing unless the with-hints ablation asks for it.

The header carries `issue` (URL), `resolved` (`direct` / `via PR #N` / `none`), `fetched` (UTC),
`fix-commit-date`, `comments-included`, then the gate's own provenance — `gate-tree-sha256`,
`gate-answer-sha256`, `gate-removals` — and the pairs the artifact is *for*: `raw-sha256` /
`content-sha256` for the issue text and `raw-hints-sha256` / `hints-sha256` for the comments. The
two `gate-*-sha256` lines make the freeze auditable: they are hashes of the gate's **inputs**, so a
reader can tell which tree and which answer tokens produced the removals without re-deriving them.
A report with no comments has empty `raw-hints` / `hints` sections and carries the hashes of the
empty string; `comments-included` is `0`, and a file that says `0` while carrying a hints section
is refused.

`prepare`, the measure-only precheck and the run all read this file's `<!-- content -->` section
rather than re-deriving a text. Identity is therefore **by construction**, not by convention or by
re-running: there is one text, and every stage reads the same bytes. `load_frozen` verifies both
`content-sha256` and `raw-sha256`, so a file edited by hand — in the prompt text or in the record of
what was removed — is refused rather than silently shipped. A test asserts that measure-only and a
full `prepare` produce byte-identical issue text.

**One canonical function, applied to a section on both sides, and it is the text Vanta is given.**
`_canonical` is that form: line endings folded to `\n`, the blank lines the framing adds at a
section's edges dropped, and exactly one trailing `\n` on a non-empty body. Both hashes are taken
over it (`content-sha256 = sha256(_canonical(gated))`, and likewise for `raw-sha256`), and the
string `FrozenIssue.content` hands out **is** that form — so "what is hashed" and "what the prompt
holds" are one object rather than two that have to be kept in step. The edge rule removes newlines
and nothing else: a `strip()` would also eat a trailing space, which is a byte of the text, and a
hand edit that only moved whitespace would then verify.

**The fold is the rule, not an accident of who called which reader.** The API returns an issue body
carrying whatever `\r\n` the reporter's editor used. The first dev freeze hashed that text as it
stood and then wrote it, while the reader got it back through a universal-newline `read_text`, so
every line had already lost its `\r` before hashing; written and read forms therefore disagreed on
every report containing a carriage return — five of the six dev bugs, the pandas report naming
`2820bbb3…` in the header while the reader computed `0f9eb3c9…`. Both sides now fold first, and the
file on disk is the LF-only form that was hashed, so the artifact is one git, an editor and a diff
all read the same way. This is a rule about the **file**: it folds line endings and it never
touches anything else. It is not the whitespace *inside* the text the gate removes tokens from —
that whitespace is part of what the text *is*, and the gate leaves it exactly as the reporter
wrote it (§2.4.4).

**`write_frozen` verifies before it writes, through the reader's own parser.** It renders, runs
`parse_frozen` over the exact text it is about to store, and only then opens the file
(`newline=""`, no translation in either direction). An artifact its own reader would refuse thus
cannot be created, and nothing is left behind when the two disagree — which matters because the
first freeze's refusal surfaced a stage later, as a gate failure on a file that looked fine on
disk. Three tests pin the round trip: write-and-read verifies; one content byte changed (a trailing
space, the case a naive rule absorbs) fails the content hash; a header or a gate byte changed
leaves the content and its hash alone, because those parts carry their own provenance.

**A mismatched artifact is refused and left exactly as it is, and `--refreeze` is the one way
past that.** A file whose hashes disagree with its own header arises two ways — a hand edit, or a
writer that has since been fixed — and in both cases it is not a text to run: it is a file that
cannot say what it contains. So nothing overwrites it, not even a full `prepare`; the refusal is a
gate failure naming the file, the hashes and the remedy, and the bug leaves the batch with that
reason on it. The rewrite is `prepare --refreeze PROJECT/BUG`: it names the bug, re-fetches the
report (the client's cache answers, so it may touch no network at all), re-runs the gate, and
writes the new artifact with `- refrozen: <UTC>` in its header when it replaced a file that was
already there — the date is what lets a reviewer of the diff read a decision with a moment on it
rather than an unexplained move of `content-sha256`, and a first freeze carries no such line.
`--refreeze` selects only the bugs it names and prepares them **in full**, because the re-fetch
re-runs the gate and the gate's inputs are the bug's trees; `--refreeze` together with
`--measure-only` is refused outright rather than half-honoured.

The markers are explicit lines rather than blank ones because a `none` report's content is empty,
and a reader that split on "the first blank line after the header" could not tell that from a
truncated file.

**Only the content section is sent, and a test says so over the gate's own inputs.** The gate's
inputs are answer-side — `GateRecord.files` is the hidden test file the answer tokens were counted
from, and the removals are the tokens that occurred in the fix diff or those files. A reader that
took "the file's text" rather than its content section would put both in the prompt, and nothing in
any metric afterwards would show it. So the assertion is not "the text looks clean" but "every path
the gate counted answer tokens from is present in the committed file (the record is honest) and
absent from `content` (the prompt is clean)", plus the header's own lines and the four other markers
being absent from `content`. Read back, `FrozenIssue.content` is the *only* string `prepare` hands
the selector and the run (`build_issue` sets `prep.issue.text` from it), so the test's subject and
the prompt's are the same bytes.

**The same claim is made about the hints, and it is made in the code rather than in a flag.**
"Never shown by default" is not "shown only if a boolean is false": under the default setting
`issue_text_for` returns `FrozenIssue.content` **before it looks at the hints at all**, so no copy
of the comments exists anywhere on the path to a prompt. A test pins that by monkeypatching the
module's own `load_or_none` to fail and asserting the default call still succeeds — a flag that
merely gated the read would pass a weaker test and could be flipped by a caller. `show_hints` is
recorded in the run provenance (`Job.show_hints`, the row's `show_hints`, `run --show-hints`) and
is part of the job key, so the ablation's rows can never be confused with the main arm's on a
resume.

**The gate removes tokens and never reformats, and a test says so over a code block.** The report
is the reporter's own text: a fenced block or an indented traceback in it is Python they pasted,
and its indentation is a byte of the text. So the gated section is the raw section with
`[redacted]` in place of the removed tokens and **nothing else changed** — no whitespace collapse,
no re-wrapping, no trimming of runs of spaces — apart from the one line-end fold of §2.4.3
(`\r` out, and a trailing `\n` normalised at the section's edge). The addendum's test freezes a
body containing `    print('1')` inside a fence, reads the artifact back, and asserts the section
is byte-identical; a second test does the same for an indented non-fenced body. A gate that
collapsed runs of spaces would turn that line into `print('1')`, which is not Python, and the
freeze would be destroying the evidence the bug is about.

#### 2.4.4 The leak gate, now applied to the issue text — once, at freeze time

The gate of §2.2 step 4 is kept, unchanged in rule, and now runs over the issue text instead of the
commit message: strip any token that occurs in the fix diff or the hidden tests and **nowhere** in
the buggy tree, and record every removal. Applied to a real issue report the rule is finally doing
what it was written for — the reporter has not seen the fix, so a token they use that occurs nowhere
in the buggy tree is genuinely suspicious, while a symbol that *is* in the tree is not removed,
which is correct.

**It runs twice per freeze, on the issue text and on the comments, and only the issue text gets a
verdict.** `gate_text` is the substitution alone; `leak_gate` is `gate_text` plus the actionability
check and runs on the title and body; `gate_hints` is `gate_text` and runs on the comments. The
hints are gated for the same reason the text is — if the ablation shows them, they must not carry
the answer's tokens — but they are **not** asked whether they are still actionable, because nothing
sends them by default: a comment section the gate shredded is not a reason to drop a bug from the
main evaluation, and refusing a bug for the state of a section the main evaluation never reads
would be a rule about the wrong arm. The gate record consequently holds an optional `hints` block
that reports removals and ungated prose and carries no `ok`/`reason`, and it is `None` for a report
with no comments rather than an "empty and fine" object, so "there were no comments" and "the
comments were gated and came out clean" stay distinguishable in the record.

**It runs once, when the report is frozen, and its inputs come from `git` alone.** This is what
lets measure-only and a full run agree. Measure-only has no `vanta_tree` — that tree is cut from the
oracle after the strip, and the strip needs `FAIL_TO_PASS`, which measure-only does not derive — so a
gate that needed the run's own oracle could not run in both modes, and the two would ship different
prompts. The freeze-time inputs are:

- **tree tokens** — counted over `src_buggy`, the buggy commit's own source. No test run, no oracle.
- **answer tokens** — the fix diff (`bug_patch.txt`) plus the test files `bug.info` declares, read at
  `src_fixed`. The hidden test sources and `FAIL_TO_PASS` nodeids are *not* needed: the file the
  hidden tests live in is enough to catch a token that came from the answer.

`freeze_gate` records the two token→count maps' hashes in the header (above), so the inputs are
pinned even though they are not stored.

**If a full `prepare` later derives a different `FAIL_TO_PASS` set, that is flagged, not re-gated.**
`check_gate_drift` compares the derived F2P files against the files the freeze gated with and appends
a refusal naming any file outside that set, telling the reader to delete the artifact and re-freeze
it against the oracle the batch derived. Silently re-gating would produce a second text for the same
bug — exactly the drift §2.4.3 exists to prevent.

**Reporter code blocks and tracebacks are kept, byte for byte.** They are legitimate report content
and are exactly what §1.6.7's traceback and path signals are for — a real bug report that contains a
traceback is the case localization is supposed to handle, not a case to sanitise away. Keeping them
is a claim about bytes as well as about policy: the gate substitutes tokens and changes nothing
else, so an indented block arrives in the prompt exactly as the reporter pasted it (§2.4.3, and the
freeze-time test that asserts it).

**The hints split is a provenance fact, not only a prompt fact.** `show_hints` is written into every
run row and is part of the job key, so a with-hints run is a different arm with its own rows rather
than the same rows with a different prompt — a resume cannot pair a hinted row with a plain one, and
the paper's ablation is a set of rows it can identify. It is off for the main evaluation, and the
selector's and the run's only source for the hints is the committed artifact's `<!-- hints -->`
section: no stage re-derives them, so the ablation cannot show a comment the artifact does not hold.

#### 2.4.5 What is dropped

- The commit-message body is dropped from the issue text **entirely**.
- When a real issue exists, the hidden-test behavioural sentence (§2.2 step 3) is **dropped** too:
  the report's own words are the behaviour.
- If no issue can be resolved, the bug is marked `issue_source=none` and **excluded from the main
  evaluation** — the count is reported — rather than falling back to the commit message. The
  fallback was the leak; removing it does not create localization out of a text that carries none.
  The exclusion is mechanical, not a judgement call: `prepare_one` returns before the precheck, so
  nothing measures a prompt no run will send, and the bug's own record carries the reason. It is
  **not** a `NEEDS DECISION` — there is nothing for a human to decide — so the summary reports it as
  `excluded` with the count of such bugs beside the batch's other verdicts, and
  `issue_texts_all.md` lists them in their own section rather than among the texts awaiting review.
  This is the one place where "the bug has no issue report" and "this report failed the gate" must
  stay distinguishable: both leave an empty `text` in the record.

#### 2.4.6 Offline, for the paper only

The old commit-message text is still recorded per bug as `issue_text_commit`, **never shown to
Vanta**. It exists so that the write-up can report how much the commit message alone would have
inflated `localization_hit` — the counterfactual that justifies this section.

- It is the fix commit's message after `strip_diff_and_code`, and **not** the old assembled text:
  the behavioural sentence §2.2 appends came from the hidden tests, not from the message, and the
  claim being measured is about the message.
- `build_issue` records it in **both** modes, from the same clone the freeze reads, so a
  measure-only run and a full prepare record the same value — it is a property of the bug, not of
  the stage. It is read with the same `fix_commit_message` the freeze already calls, so a commit
  the clone cannot show fails the bug exactly as it would have while freezing.
- Recorded even for an `issue_source=none` bug, because those are exactly the bugs whose only
  possible text was this message.
- Written to the record (`BugPrep` → `prepare.json` and `derive_verdict` → the split's
  `issue_text_commit`), and read by **nothing** that builds a prompt: the prompt's text is
  `issue.text`, the frozen report's gated section.

**What the visible text names.** A second per-bug reporting fact is recorded beside it: the repo
paths, and the names the buggy tree defines, that the text a run sends itself names
(`issue_text_names_paths`, `issue_text_names_symbols`). `sanic/1`'s report names `sanic/app.py`
and `register_named_middleware` — the visible text has already done the localisation the harness
is about to ask the Creator for, and `localization_hit` cannot see that, because it scores where
the patch went and not what the prompt said. The write-up needs the rate.

- Computed over `issue.text` — the frozen report's gated section, title and body, post-gate — and
  **never** over the comments: the default setting does not send those, so a comment naming the
  file would be a fact about a prompt nobody sends (§2.4.4).
- The tree is the buggy source the gate already reads (`src_buggy`), so the flag cannot disagree
  with the text about which tree is meant. A path counts when the tree contains a file for it; a
  bare basename (`app.py`) does not, because a basename is not a repo path.
- A **GitHub blob URL into the project's own repo** counts as the path it names:
  `github.com/…/blob/<sha>/sanic/app.py#L656` is recorded as `sanic/app.py`. The line anchor is
  dropped with the rest of the URL, and the ref — a sha — is *stripped*, never resolved, never
  fetched, so the flag stays offline and cannot depend on the ref existing. Both `PySnooper/3` and
  `sanic/1` name the fix's own file this way, and a reporter who pasted the link named that file
  as surely as one who typed the path.
- A name counts when a source file of the tree **defines** it — `def`/`class` at **any** depth, a
  method included, or a binding at column 0 — over `.py`, `.pyi`, `.pyx` and `.pxd` alike, since
  `.pyx` is where pandas' and spaCy's bugs sometimes are. Depth is not a filter: `sanic/app.py`
  declares `register_named_middleware` as an indented `def` inside a class, so a module-level-only
  scan answered "the tree does not define it" — which is false, and it lost the sharpest instance
  of the very fact the flag exists to record. Bindings stay module-level, because a name bound at
  column 0 is public to the module while a local variable inside someone's function is not a name
  of the tree in any sense the paper reports.
- **The name must be written as code.** It counts only if it is code-shaped — an underscore, a
  digit, or CamelCase (`register_named_middleware`, `is_string_dtype`, `jinja2`, `BigQueryTarget`)
  — or the reporter wrote it in one of the code forms: `` `backticks` ``, `name(`, or `obj.name`
  (the name *after* the dot: in `fig.savefig(` it is `savefig` that counts, and `fig` is the
  reporter's own variable). Plain English in ordinary prose never counts, so `a`, `in`, `release`,
  `version` and `report` cannot appear: a report saying "the 3.2.0 release notes" has named
  nothing, and a flag the write-up cites must not claim it did. A fenced code block is **not** one
  of the code forms — a reproduction script is made of the reporter's own local names (`fig`,
  `data`, `ax`), which are exactly what a project's test files bind at module level too, so
  reading a block as "everything inside counts" would put back the noise this bullet removes.
- Both lists hold the tokens **as the text writes them** — a blob URL is recorded as the repo path
  it names, which is the thing that was named — so the paper can quote the report's own words, and
  both are sorted.
- Recorded in **both** modes, like `issue_text_commit`, and read by **nothing** that builds a
  prompt. It is never used in selection: a bug whose report names half the tree is selected, sized
  and graded exactly like one whose report names nothing.
- Surfaced in the hand-review artifact (`issue_texts/<slug>.md`), so the reviewer sees the claim
  beside the text it was computed from.

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

**The py3.8 layer carries the system packages the reviewed sdist table names.** `bake.SYSTEM_PACKAGES`
is no longer a hand-written list: it is *computed* from `bake.SDIST_SYSTEM_PACKAGES`, the reviewed map
from a package that compiles from its sdist to the Debian trixie packages that build needs
(`cffi` → `libffi-dev`, `lxml` → `libxml2-dev` + `libxslt1-dev`, `scipy` → `gfortran` +
`libopenblas-dev` + `liblapack-dev`, `uvloop` → `make` + `autoconf` + `automake` + `libtool`, and so
on), unioned with the five the layer started with — `g++`, `pkg-config`, `libfreetype-dev`,
`libpng-dev`, `zlib1g-dev`. **One table, two readers**, and that is the point: the preflight's
unreviewed-build check looks a pin up in the same map, so an entry added for a preflight `gap` is
simultaneously in the install list the next bake runs. An **empty** entry is a review result and not a
missing one — it says "this package compiles, against nothing but libc and the compiler the layer
already carries" — and it *clears* the `gap`; only a package with no entry at all is reported. Every
entry was decided by reading the **pinned sdist's own `setup.py`/`setup.cfg`** (from the cached
archive, never executing it) for `libraries=`, `library_dirs`, a pkg-config / `find_library` /
`ctypes.util` call, or an include of a header the archive does not carry, and the finding is recorded
as the comment on the entry, so a reader argues with the evidence rather than with the verdict. The
measures that review settled (`coverage`, `multidict`, `yarl`, `websockets`, `tornado`, `urwid`,
`zope-interface`, `regex`, `sqlalchemy`, `psutil`, `ruamel-yaml-clib`, `httptools`, `ujson`, `brotli`,
`zstandard`, `asyncpg`, `numpy`, `cython`, `distlib`, `thrift` → toolchain only; `pyyaml` → `libyaml-dev`;
`peewee` → `libsqlite3-dev`; `matplotlib` → `libfreetype-dev` + `libpng-dev` + `zlib1g-dev` +
`pkg-config`) are in the map itself, each with its evidence. The matplotlib entry is the case that shows
the two readers are one design and not two lists: its four packages are already in
`_PY38_BASE_PACKAGES`, so the interpreter's install list is unchanged — but a `pandas/1` pin of
`matplotlib==3.2.1` still reaches the preflight's `gap`, and this entry is what clears it, from
`setupext.py`'s own `pkg_config_setup_extension` calls (`'freetype2'`, `'libpng'`, `deplib('z')`) and
`get_pkg_config`, which is the `pkg-config` it needs to find them.

The interpreter recipe installs the union (§3.2's
build-toolchain rule for the *system* half: the pinned `setuptools`/`wheel`/`Cython` are the Python
half), and a package that is not in dpkg's own `installed` state afterwards fails **that layer's** bake.
The check is on the status field and not on `dpkg-query -W`'s exit code, because a package dpkg knows
but has not installed still prints a version. Verified at the layer, not at the bug, on purpose: the
failure belongs to the environment, and left alone it surfaces mid-bake on the first bug that compiles
anything — as `g++: not found`, i.e. as that bug's problem, on every wave. Their versions are recorded
*inside* the layer, at `/opt/vanta-system-packages.txt`: the system half of the layer's provenance,
written by the same script that does the install, beside the pip half `/opt/vanta-deps-freeze.txt` every
layer already carries.

**The compiler a source build runs, and the era's warning semantics.** The 3.8 interpreter is a
`python-build-standalone` build, so its `sysconfig` records the compiler of the machine it was built on,
and an old-style `distutils` build prefers that recorded string to anything the caller assumed. What it
recorded is a fact about the layer rather than a guess: `bake._sysconfig_probe` prints it first in every
toolchain layer's recipe, and the baked `vanta-bi-fastapi-1` / `vanta-bi-PySnooper-3` logs carry
`recorded CC: cc`, `recorded CXX: c++`, `recorded LDSHARED: cc -shared -Wl,--exclude-libs,ALL`,
`recorded LDCXXSHARED: c++ -shared` — bare driver names that *do* resolve in this image. That is the
correction to this section's earlier reading of them as "a path that does not exist in this image", and to
its attribution of `matplotlib/1`'s `numpy==1.18.5` `error: [Errno 2]` to the compiler *invocation*: that
traceback is the VM's missing `/dev/shm` (the passage at the end of this section), not the compiler.
`bake.COMPILER_ENV` still exports `distutils`' documented override in every layer whose minor has a
toolchain (`bake.build_env_for`): `CC` and `CXX`, which name the two wrappers `bake.VANTA_CC_PATH` /
`bake.VANTA_CXX_PATH` rather than a driver — the passages further down are the reason each of them is a
file rather than a name — together with CPython's own Linux link lines
`LDSHARED='gcc -pthread -shared'` / `LDCXXSHARED='g++ -pthread -shared'` — so the layer builds and links
with the toolchain `SYSTEM_PACKAGES` installs rather than with whichever driver `cc` currently points at,
and so the `CFLAGS` below reach the **link** line as well as the compile line (`customize_compiler`
appends an environment `CFLAGS` to both). The recipe asserts each variable reached the layer's shell,
because an `export` is not the same fact as the variable the build reads. The interpreter layer then
builds a throwaway C **and** C++ extension before its marker, so a broken toolchain fails the base layer
instead of twenty minutes into numpy.

The same environment carries `CFLAGS`, which is the second half of that failure: the compiler was right
and its *defaults* are not. GCC 14 (the base image's) promotes six C diagnostics from warning to **error**
— `incompatible-pointer-types`, `implicit-function-declaration`, `int-conversion`, `implicit-int`,
`return-mismatch`, `declaration-missing-parameter-type` — and these are 2020-era sdists written for the
GCC of their day, where all six were warnings that scrolled past. Measured on `lxml` (an incompatible
pointer assignment) and `ruamel.yaml.clib` (a handler signature mismatch). The fix is `-Wno-error=` for
each of the six, not `-Wno-`: the diagnostics stay **visible** and only their promotion to failure is
undone, so what is restored is the era's *warning* semantics rather than silence.

The same variable carries the other thing GCC 14 took from the same sources: the standard header old C++
code no longer gets pulled in for it. `bake._ERA_STDINT_INCLUDE` is `('-include', 'stdint.h')`, and it is a
member of `CFLAGS` rather than a variable of its own for the measured reason `CXXFLAGS` cannot be used:
distutils' `customize_compiler` reads `CC`, `CXX` and `CFLAGS` from the environment — appending `CFLAGS`
to its `cflags` *and* to its link line — and never reads `CXXFLAGS` at all, so the fix cannot be spelled
the way the compiler's error invites. `stdint.h` and not the `cstdint` the error names, because this
variable reaches the **C** compiles too and `cstdint` is a C++-only header that is not on `gcc`'s C
include path, so `-include cstdint` would turn every C file in the tree into
`fatal error: cstdint: No such file or directory`. **That flag does not supply the C++ name, and the bake that
would have shown it was never run.** The paragraph above went on to say that under `g++` this is
libstdc++'s `c_compatibility/stdint.h`, declaring `namespace std { using ::uint16_t; … }` for
`pybind11==2.5.0`'s `std::uint16_t nargs;`. On this toolchain that is **false**: `stdint.h` is glibc's,
the *C* standard header, with no `namespace std` block and no `__cplusplus`-guarded one, and C++'s
`std::uint16_t` is declared by libstdc++'s `cstdint`, a *different* header that nothing on `attr.h`'s
include chain reaches. Measured on `pandas/1`'s own dependency layer
(`runs/bugsinpy/bake-logs/vanta-bi-pandas-1.log`): the `.cxx` compile line carried the flag —

```
  g++ -Wno-unused-result … -D_GNU_SOURCE -include stdint.h -fPIC -DPOCKETFFT_PTHREADS … \
      -c scipy/fft/_pocketfft/pypocketfft.cxx
```

— and the compile died at the same place, `pybind11/attr.h:185`, on `std::uint16_t`, with the five
consequences of it that follow. `_ERA_STDINT_INCLUDE` is therefore **retained in `CFLAGS` as it stands and
is *not* the C++ fix**: it is inert for the C trees beside the failure (`stdint.h` *is* the C standard
header, and the `-D_GNU_SOURCE` on the same line is what keeps its early arrival from costing `Dl_info`),
and taking it out would change the C command line of every layer that already builds — a change this fix is
not entitled to make. It is recorded here as outstanding residue rather than mistaken for the mechanism.

**How far this variable reaches a C++ compile is narrower than it reads, and that is exactly why the fix
had to leave it.** The measurement stands: the environment `CFLAGS` rides the command line that compiles
`pypocketfft.cxx` only because `numpy.distutils` keeps one flag list and swaps only the driver word — that
log shows the `.cxx` line reading `C compiler: g++ -Wno-unused-result … -fPIC`. A plain
`distutils`/`setuptools` C++ `Extension` is a different path and does *not* see it: the stdlib
implementation compiles C++ with `compiler_so` — the *C* line, whose flags came from the recorded cflags and
which reads no `CXXFLAGS` either — and the modern `setuptools._distutils` fork seeds its C++ flags from the
**recorded** `cflags` *before* it applies an environment `CFLAGS`, so only `CXXFLAGS` would move them. The
repository's own `vanta_smoke_cpp.cpp` compile in the interpreter layer's log carries none of this variable
for exactly that reason. Read together with the paragraph above, the conclusion is not "carry the header for
`numpy.distutils`' sake" but the opposite: a **C++-only** flag has no environment variable that delivers it
— an environment `CFLAGS` would reach the C files too, and `CXXFLAGS` is read by nothing — so the one
position that is only ever a C++ compile is the driver word itself.

**A C++-only flag is delivered by the driver word, and each driver word is a file its layer writes
itself.** `bake.VANTA_CXX_PATH` is `/opt/vanta-cxx`, and `bake.COMPILER_ENV["CXX"]` names it instead of
`g++`; the wrapper it names (`bake.VANTA_CXX_SOURCE`) scans its arguments and, for a C++ source
(`.cc`/`.cpp`/`.cxx`/`.c++`/`.C`/`.CC`/`.CPP`/`.CXX`), runs `exec g++ -include cstdint "$@"`, and otherwise
`exec g++ "$@"` — so a C compile sees the flag nowhere and a C++ compile sees it *first on the line*, before
`attr.h`. Anything that is not a compile passes through unchanged (`g++ --version`, which `distutils`
probes), and `bake.LDCXXSHARED` stays `g++ -pthread -shared` because the link step is not a compile and has
no pre-include to give. The file is **written by each layer's own recipe** (`bake.build_env_lines`, between
the `export`s and its assert) and not exported from a parent, for the reason `bake._ERA_GNU_SOURCE` records:
a checkpoint is a filesystem and an `export` does not survive into a layer baked `--from` another, so a
wrapper `export`ed by the interpreter layer would be a dangling path in every dependency layer. The recipe
`chmod +x`es it and then asserts it is executable, so a layer that failed to write it fails at bake time
rather than silently compiling C++ without the header it needs. It is one of **two** such files: a single
helper writes them (`bake._wrapper_lines`), once per driver word that is a path, each guarded on its own
variable and with the heredoc delimiter derived from that variable's name (`VANTA_CXX_WRAPPER`,
`VANTA_CC_WRAPPER`) — so neither can be written with the other's delimiter and a mapping that names one of
the two writes one file, not two.

**The second driver word carries the one flag the first cannot reach, because the extension that needs it
declares no C++ language at all.** `pandas/1`'s next failure after scipy is matplotlib 3.2.1's
`src/ft2font.cpp` against the base image's **FreeType 2.13.3**. Measured in
`runs/bugsinpy/bake-logs/vanta-bi-pandas-1.log`, the compile ended

```
  src/ft2font.cpp:230:29: error: invalid conversion from 'unsigned char*' to 'char*' [-fpermissive]
  src/ft2font.cpp:346:29: error: invalid conversion from 'unsigned char*' to 'char*' [-fpermissive]
  error: command 'gcc' failed with exit status 1
  ERROR: Failed building wheel for matplotlib
```

and the `gcc` that word names is the point: the failing command is a `gcc … -c src/ft2font.cpp`, not a
`g++`. `setupext.FT2Font().get_extension()` builds `Extension('matplotlib.ft2font', sources)` with no
`language='c++'` anywhere in `setupext.py`, so `distutils` never consults `CXX` for it and runs
`compiler_so` — `CC` plus `CCSHARED` — instead; `bake.VANTA_CXX_SOURCE` is therefore on the wrong word for
this file, whatever flag it carries. The diagnostic is the era mismatch again, one layer out: FreeType
changed the field's signedness in 2.13.3 (`ftimage.h:353` reads `unsigned char* tags;` here, and the
release's own `CHANGES` entry says the `FT_Outline` fields moved from signed to unsigned), while the 2020
source declares its own view of that struct as `char *tags;` (`ft2font.cpp:208`, `:326`) and fills it from
the outline (`tags = outline.tags + first;`, `:230`, `:346`). **The flag is `-fpermissive`, and it is the
only spelling that exists**: GCC prints its own remedy in the brackets, and `-Wno-error=permissive` is not a
member of the `-W` family at all — measured on this layer, `cc1plus: error: '-Wno-error=permissive': no
option '-Wpermissive'; did you mean '-fpermissive'?` — while the `incompatible-pointer-types` demotion of
`_ERA_WARNING_FLAGS` is a *C* diagnostic and this is a C++ frontend error in every standard. So
`bake.VANTA_CC_PATH` is `/opt/vanta-cc` and `bake.COMPILER_ENV["CC"]` names it instead of `gcc`, with
`bake.VANTA_CC_SOURCE` the same argument scan over the same list, running `exec gcc -fpermissive "$@"` for a
C++ translation unit and `exec gcc "$@"` otherwise.

**The evidence for that position is four measurements, and three of them are rejections.** On the layer
(`runs/bugsinpy/probe-v3.log`): **A**, the control, is `CC=gcc` with today's `CFLAGS` and reproduces the two
errors above verbatim; **B**, `-fpermissive` appended to the environment `CFLAGS` with a pure logging
passthrough on `CC`, *does* put the flag on the `ft2font.cpp` argv — the flag works, and the objection is
the position, because `customize_compiler` appends that variable to `ldshared` and it reaches every *C*
compile in the layer beside this one; **C**, a pure C compile carrying `-fpermissive`, prints only FreeType's
own `#pragma message: Compiling with FreeType version 2.13.3.` and exits `rc=0` — inert on C, which is what
makes the scan the difference between the flag and the flag on the right file; and **D**, this wrapper on
`CC`, carries `-fpermissive` on all **four** C++ translation units of that extension and on **none** of the
argv that compiled the extension's one C source, `src/checkdep_freetype2.c` (`FT2Font`'s five sources are
four `cxx` and one `c`, and every one of them is compiled by this one word — which is why the scan and not
the word does the scoping). The same shape is asserted at unit level rather than only measured once:
`scripts/test_bugsinpy.py` executes the wrapper against a stub `gcc` and asserts the flag for
`src/ft2font.cpp` and its absence for `src/checkdep_freetype2.c` and for `--version`. The wrapper is not
covered by the interpreter layer's smoke check, because the file it would need is a C++ source compiled by
the *C* word — a `.cpp` in an `Extension` with no `language='c++'` — and the smoke's C++ extension is
deliberately the other case.

**The header costs a glibc feature-test macro, and the macro is a member of the same variable.** Measured
on the bake that carried `-include stdint.h` and nothing else: `matplotlib/1`, `pandas/1` **and** `sanic/1`
all died inside `numpy`'s own sources, at

```
  numpy/core/src/multiarray/temp_elide.c:158:9: error: unknown type name 'Dl_info'
```

followed by `ERROR: Failed building wheel for numpy`. `sanic/1` had baked clean before that change, so this
is a regression rather than a fourth bug. The mechanism: `-include` is processed as if the file were the
translation unit's *first line*, so glibc's `<stdint.h>` reaches `<bits/libc-header-start.h>` and
`<features.h>` before the source's own feature-test macros are seen, `__USE_GNU` is never set, and
`Dl_info`/`dladdr` from `<dlfcn.h>` — both behind `__USE_GNU` — are never declared; GCC 14 makes the
unknown *type* a hard error while the `dladdr` call is only an implicit-declaration warning. The answer is
`bake._ERA_GNU_SOURCE`, which is `('-D_GNU_SOURCE',)` and sits in the same `CFLAGS` string immediately
before the include: GCC applies every command-line `-D` and `-U` before it reads any input, `-include`
files included, so the feature-test macro is in force when the pre-included header arrives. The macro is not
optional even though the header it protects turns out not to be the C++ fix (above): the header is retained
on every C command line, and *its* early arrival is what costs the macro, so dropping the macro and keeping
the header is a measured failure in `matplotlib/1`, `pandas/1` and `sanic/1` alike.

The interpreter layer's smoke check covers the C half of the variable, the two compilers, **and the C++
header's new route**. One C file compiles one of the six demotions (a `double *`
assigned to an `int *`), which is the receipt that an environment `CFLAGS` reached `distutils` at all; one
C++ file *builds and imports*, which is the receipt that `CXX` names a compiler able to build a Python C++
extension — the `matplotlib`/`pandas` shape. The C++ source deliberately does **not** `#include <cstdint>`
and does use `std::uint16_t`, so its compiling is the receipt that the header arrived from the driver word
and not from the file — the version of it that included `<cstdint>` itself proved nothing about the wrapper,
and the version before that left the name to `-include stdint.h` and was red on every layer forever.

Which *build* compiles that file is what makes it a receipt for one wrapper and not the other. The smoke
runs `python setup.py build_ext` in the interpreter layer, whose setuptools is `uv venv --seed`'s (75.3.4) —
the `setuptools._distutils` fork, whose `unixccompiler._compile` chooses its word per source
(`if self.detect_language(src) == 'c++'` → `compiler_so_cxx`, read in the released wheel) and whose
`customize_compiler` builds that word from the environment's `CXX`. The 2020 trees are built by the pins
their own layers install (`setuptools==46.4.0`), which have no `setuptools._distutils` at all: on Python 3.8
that is the *stdlib* `distutils`, whose `_compile` sends **every** source through `compiler_so` — the `CC`
word — whatever its extension, which is why the measured `ft2font.cpp` line begins `gcc`. So the smoke
measures `bake.VANTA_CXX_PATH` and cannot measure `bake.VANTA_CC_PATH`: the position the second wrapper
fills is a `.cpp` that only the stdlib driver reads. What the smoke does **not** claim is the C++ reach of
`CFLAGS`, which no C++ compile has (above). Each smoke
step is its own shell statement rather than an `&&` chain, which is load-bearing rather than style: measured,
an AND-OR list swallows the failure of every member but its last under `set -e`
(`sh -c 'set -eux; false && true; echo CONTINUED'` prints `CONTINUED` and exits 0), so the first version's
`cd … && build_ext && python -c …` chain let the C++ compile's hard error scroll past into `pip freeze` and
`touch /opt/vanta-deps-ok` and the layer was reported `[done]` with a broken toolchain inside it. The
de-chained smoke fails the layer it is in, which is what it exists to do.

The whole mapping is recorded inside the layer at `/opt/vanta-build-env.txt`, together with `gcc --version`:
the variables say what was asked for and the version says which compiler read them, and the flags are
written for GCC 14's defaults — a layer whose `gcc` is a different major is a layer where they mean
something else.

**The other compiler this toolchain installs has the same problem, and the same answer.** `SYSTEM_PACKAGES`
unions `SDIST_SYSTEM_PACKAGES`, whose `scipy`/`scikit-learn`/`statsmodels` entries put `gfortran` in the
interpreter layer, so Fortran sources are built in these layers too and run into the same era mismatch
one decade later. Measured on `pandas/1`, whose layer builds `scipy==1.4.1` from its sdist: its ARPACK
sources call `dvout` twice, with the same arity and a different rank in one position —

```
  scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaitr.f:658:35:
    376 |             call dvout (logfil, 1, rnorm, ndigit,
        |                                   2
    658 |             call dvout (logfil, 2, xtemp, ndigit,
        |                                   1
  Error: Rank mismatch between actual argument at (1) and actual argument at (2) (scalar and rank-1)
```

— and gfortran refused it, at `:658` and again at `:729`, after which the build stopped on
`error: Command "/usr/bin/gfortran -Wall -g -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
… dnaitr.f -o dnaitr.o" failed with exit status 1` and `ERROR: Failed building wheel for scipy`
(`ERROR: Could not build wheels for scipy, which is required to install pyproject.toml-based projects`).
It is the same story as `CFLAGS`: GCC 10 promoted Fortran's *argument-mismatch* family from warning to
error, and every release before it had printed this as a `Warning:`. Nothing in the log was killed, out of
memory, or missing a BLAS library — a grep for `Killed`, `MemoryError` and `BLAS` in that layer's log is
empty and the compile ended in an `Error:`, not a signal — so this is the flags path and not the
resource-limit one. `bake._ERA_FORTRAN_FLAGS` is therefore `('-fallow-argument-mismatch',)`, sitting
beside `_ERA_WARNING_FLAGS` for the same reason its counterpart does: the flag demotes the diagnostic
rather than silencing it, and the mismatch it is documented for is *between different calls*, which is
exactly the `dvout` pair. It goes in `FFLAGS` and not `F77FLAGS`/`F90FLAGS` because these trees build
Fortran through `numpy.distutils`, whose `FCompiler` names `FFLAGS` as the environment variable of its
**generic** flag list (`flag_vars.flags = ('self.get_flags', 'FFLAGS', 'fflags', …)` in
`numpy/distutils/fcompiler/__init__.py`), so one variable reaches the fixed-form `f77` source that failed
here and the free-form ones beside it; the per-flavour variables are in the same table and cover only
their own form. `COMPILER_ENV` carries it like the rest, so it is asserted into the layer's shell and
recorded in `/opt/vanta-build-env.txt` by the same two hooks that handle `CFLAGS` — `_build_env_assert`
iterates the mapping and `build_env_lines` joins its keys into the recorded pattern — with no separate
mechanism. The interpreter layer's smoke check gains a second file for it: `_TOOLCHAIN_SMOKE_F_SOURCE` is
fixed-form Fortran calling one implicit external procedure twice, once with a scalar and once with a
rank-1 array — the `dnaitr.f` shape in miniature, with the callee deliberately left undefined so gfortran
has no interface for it, as it has none for `dvout` — and `toolchain_smoke_lines` compiles it with
`gfortran $FFLAGS -ffixed-form -c …`, the form and the variable the failing command used. A layer where
`FFLAGS` never reached the compiler fails that compile with the same `Error:` rather than reproducing it
inside scipy.

**A package the platform cannot build says so, rather than arriving as a build error.**
`bake.UNBUILDABLE_ON_PLATFORM` is the third answer the preflight can give about an sdist-only pin and it
is a review, not a gap: `pyarrow` needs the Arrow C++ toolchain and `llvmlite`/`numba` a specific LLVM
release, so no entry in the system-package table could make those builds work on this image. Reported as
`UNBUILDABLE <pin> — <reason>` and recorded as such, because the alternative is discovering it as one
bug's compile error (§3.2's unresolvable-pin rule then decides whether that bug bakes or is flagged).

**The table is keyed by name *and* release, because one package can have two impossible builds.**
Each value is a tuple of `bake.PlatformLimit(reason, bound)` clauses read in order, and the first whose
`bound` covers the pin's version is the reason reported; an empty `bound` means every release, which is
how the three seeded entries (`pyarrow`, `llvmlite`, `numba`) read. `cryptography` is the case that
forced the shape rather than a nice-to-have: `2.9.2` — the release the split's own pins name — builds
its C bindings against OpenSSL's headers and needs the **1.1** ones, which this trixie image does not
carry (its OpenSSL is 3.x), while every release from `35.0` builds against 3.x but needs a **Rust**
toolchain the image does not carry either. Those are two reasons for one name and only the version says
which one a pin runs into, so the pin-versioned entry carries both and the name left
`SDIST_SYSTEM_PACKAGES` — an entry there is consulted by *name* alone and could not have been narrowed to
one of them. It was `libssl-dev` there for as long as the review stood for the 2.9.2 build, so that
package leaves the interpreter layer's install list with it
(`bake._system_packages_for_py38`), which is a fact about that layer and not about the table: nothing
else the split builds links OpenSSL. A pin whose version **no** clause covers falls through to the
unreviewed-`gap` answer instead of inheriting a limit stated for a different release — the reason to
report is "nobody has judged this version", not a reason borrowed from its neighbour.

Dropping `cryptography` by this rule is a drop like any other, so the dependents had to be checked
rather than assumed, and the check is what makes the drop safe: `pyOpenSSL==19.1.0` (which the split
pins) declares exactly one unconditional requirement on it — `cryptography (>=2.8)`, satisfied by the
`2.9.2` the layer pins — and `service-identity==18.1.0`'s is a **bare** `cryptography`. Neither
declares a floor above `2.9.2` and neither is what pulls a modern release in: with the pin gone, the
legacy resolver has no requirement left that names a version of `cryptography`, so it installs nothing
for it, and §3.2's audit decides per bug whether the two dependents drop with it (nothing in `luigi/1`'s
or `sanic/1`'s graded-test import closure imports `cryptography`, `OpenSSL`, `pyopenssl` or
`service_identity` — measured — so they drop by the same rule). `numpy` is deliberately **not** here: a
missing header is a fixable fault of the table, and `numpy==1.18.5`'s sdist is what this image builds
(the measured `[Errno 2]` is a toolchain invocation and not a platform limit).

**One environment for every py3.8 bug is the point, and a compiler does not change a pure-Python bug's
tests.** The layer is shared, so a compiler added for `matplotlib/1` is present for `luigi/1` as well.
That is safe because `needs_build` (§3.2) decides per bug whether anything is compiled at all: a bug
that compiles nothing never invokes the toolchain, so its collected tests, its verdicts and its report
are the same as on a layer without one. The measured set that does compile is dev `matplotlib/1`,
`matplotlib/2`, `pandas/1`, `spacy/4` and eval `cookiecutter/1`, `tqdm/1` — and the last two are
**3.6**, whose layer this leaves alone.

**Rebuild set.** The list is keyed on the minor, so only `vanta-bugsinpy-py3.8`'s recipe changes. The
next bake therefore rebuilds that layer and, because a rebuilt interpreter layer makes everything
stacked on it stale (`bake.plan_layers`), every layer beneath it: the whole dev split (11 bugs, all
3.8) and, in the evaluation split, `PySnooper/3`, `black/1` and `black/2`. `vanta-bugsinpy-py3.6` and
`vanta-bugsinpy-py3.7` are untouched — for a minor absent from `SYSTEM_PACKAGES` the recipe is
unchanged line for line — so `cookiecutter/1`, `tqdm/1`, `thefuck/1`–`3`, `httpie/1`–`3` and
`youtube-dl/1` are not rebuilt. The 3.6 layer is the one that will need the same decision before its
two compiler-using bugs can be baked.

then a thin per-bug layer on top:

```
shuru checkpoint create vanta-bi-<project>-<N> --from vanta-bugsinpy-py3.<v> --allow-net \
  -- sh -c 'set -e; pip install -r /tmp/req.txt; \
            pip freeze > /opt/vanta-deps-freeze.txt; touch /opt/vanta-deps-ok'
```

This mirrors the existing `vanta-deps-<fingerprint>` + `/opt/vanta-deps-ok` marker + freeze-file
pattern exactly, so nothing new has to be invented for verification.

**Preflight — what every pin installs as, decided before the first layer is built.** The VM is
`linux-aarch64` and most of these 2020-era pins have no `aarch64` wheel, so they are compiled from
their sdist and every missing system library behind that build costs a whole bake cycle to discover:
the measured `cffi` failure was `c/_cffi_backend.c:15:10: fatal error: ffi.h: No such file or
directory`, and it was found by *baking*, which is exactly what the preflight exists to prevent.
`bake.preflight_layers` therefore classifies every pin of every selected layer against PyPI's own
per-release file list (`bake.pypi_release_files`, reading the release payload's `urls` key — the same
URL the date fetcher reads `info` from, and the half it deliberately ignored until now) **before**
`layer_recipes` is called, and writes `runs/bugsinpy/deps/<checkpoint>.preflight.json` beside the
provenance, with `--preflight-only` to do that and stop:

- **(a) `wheel`** — a wheel that installs on `cp38 linux-aarch64`: `py3`/`py2.py3`/`py3N` (N ≤ 8) with
  `none`/`any`, or a `cp38`/`cp38m` wheel, or an `abi3` wheel from `cp30` to `cp38`, each with a
  `manylinux*`/`musllinux*`/`linux` `aarch64` platform tag. A wheel wins over the sdist of the same
  release, because that is what pip picks.
- **(b) `sdist`** — no usable wheel and an sdist: pip will compile it. The sdist's own file list is
  read (`bake.sdist_file_list`, capped at `SDIST_MAX_BYTES` and streamed so a truncated read still
  yields what it contained) and the members whose suffix says "compiled" are recorded in the pin's
  `compiled_sources`. That count is a **raw archive fact and nothing more**: it is a suffix match over
  the whole tar, so it sees `distlib` 0.3.0's `PC/launcher.c` (the Windows launcher) and `cffi`'s
  `demo/*.c` — files no Linux build compiles. It stays in the `.preflight.json` as the fact it is; it
  is never the reason a build is said to need a library and no line is printed from it. What a build
  needs is the reviewed table's answer (§3.2's table), which is the one authority for that question,
  and no path heuristic over the file list is invented to second-guess it.
  PyPI's classifiers cannot answer it either and were measured: only **7 of 51** packages carry a
  `Programming Language :: C` classifier, and 24 of the 61 class (b) pins do not — `asyncpg`, `cffi`,
  `cryptography`, `numpy`, `regex`, `sqlalchemy`, `yarl` and the rest are silent about it.
- **(c) `unavailable`** — nothing pip can install here: the release does not exist on PyPI
  (`luigi/1`'s `requests-async==0.5.0`) or the release has no Linux distribution at all
  (`sanic/1`'s `pywin32==227`: twelve wheels, every one `win_amd64`, no sdist). What the bake then
  does with such a pin — and with a class (b) pin the reviewed table cannot build — is decided by the
  unresolvable-pin rule below.

The tag has to be *read*, and as a **compressed set**: PEP 425 makes a dot in a tag field the set of
tags the wheel also claims, so `six-1.15.0-py2.py3-none-any.whl` is `py3-none-any` as well as
`py2-none-any` — read as one literal tag, or with every alternative required at once, `six`'s own
wheel comes back unavailable (which is how that was caught). `py2-none-any` alone really is not
installable on this interpreter and falls through to the sdist. Measured over both splits' provenance:
**232 distinct pins → 170 wheel, 61 sdist, 1 unavailable**, cached per run by `(name, version)` for the
file list and by filename for the sdist listing (`numpy==1.18.5` alone is in six of the eleven dev-split
bugs), the same shape as the per-run `releases` mapping `resolve_bug_deps` fetches dates with. Two
measured corrections are in the code for the same reason: an sdist's `tests/test_data/*.cc` is a renamed
corpus and not C++ (`protego 0.1.16`), so those directories are excluded from the compiled count; and
Fortran (`.f`, `.f77`, `.f90`, `.f95`) **is** counted although nothing in either split compiles any,
because the sdists that do are exactly the ones whose read is skipped for size — `scipy` and numpy's
BLAS half — and "compiles nothing" would be the wrong answer for the two packages most likely to need
`gfortran`. It is recorded, not enforced: the bake still installs what the recipe says, and the record
is what makes the claim checkable afterwards.

**Second reading: every pin's own declarations, against the layer's other pins.** The classification
above says what pip will *do* with a pin; this says whether the layer's pins can hold each other at all,
and it is the check that would have caught `sanic/1` from PyPI instead of from `ResolutionImpossible` one
build cycle later. `bake.metadata_conflicts` reads each pin's `requires_dist` — through the run's one
`bake.cached_release_metadata`, the same reader the date rule narrows itself with, so "what does this
release declare" has one answer in the file — and compares each declaration against the *other* pins: a
pin that requires a name this layer also pins, in a version the requirement excludes, is a conflict
(§3.2's cases), recorded in `<checkpoint>.preflight.json` under `conflicts` beside the pins. A
requirement carrying a marker is skipped — `foo>=1; extra == "test"` is only installed when the extra is
asked for — so a conditional requirement neither narrows nor conflicts, deliberately: a version chosen on
the word of a condition that may not apply is a choice nothing can check.

**A missing `requires_dist` is never read as "requires nothing", and that is what the fallback chain is
for.** Measured: PyPI's version JSON answers `requires_dist: null` for about **60%** of the pins in these
splits — `matplotlib==3.2.1` (which declares five requirements), `sqlalchemy==1.3.17`, `numpy==1.18.4`,
`six==1.17.0`, `pytz`, `tornado`, and every `httpcore`/`hpack`/`hstspreload`/`kiwisolver`/`pillow`/
`pyparsing` pin among them — and PyPI **never** returns `[]` there, so the field alone cannot distinguish
"declares nothing" from "we were not told". Read as `requires_dist or ()` it silently reports ~60% of the
pins as declaring nothing, which is the blank answer this check exists to avoid. `bake.declared_requires`
therefore tries, in order:

1. the version JSON's `requires_dist` — including an explicit `[]`, which *is* the answer "this release
   requires nothing";
2. the **PEP 658 metadata sidecar** (`<file-url>.metadata`), for any file whose release entry carries a
   `core-metadata` digest — measured to exist for `matplotlib 3.2.1` (13 of its 14 files; the sdist has
   none), `sqlalchemy 1.3.17`, `numpy 1.18.4` and `beautifulsoup4 4.9.1`, and to be absent for
   `httpcore 0.3.0`, which is sdist-only;
3. the **wheel's own `METADATA` member** (`*.dist-info/METADATA`), downloaded and read as a zip, for a
   release whose sidecar is not published but whose wheel is;
4. the **sdist's own `install_requires`**, read from inside the archive without executing any of it, for
   a release that publishes no wheel and no sidecar at all — the `ast`-only `_setup_kwargs` reader of
   `setup.py` follows `install_requires=['a>=1','b']` literals, and follows `install_requires` written as
   a **file read** — `open('requirements.txt').read().split('\n')` — by opening the file it names and
   parsing its `requirements_refs`. Measured, and the reason the reading exists: `fastparquet==0.4.0`'s
   `setup.py:50` is `install_requires = open('requirements.txt').read().strip().split('\n')`, and the
   `requirements.txt` inside its sdist is `pandas>=0.19`, `numba>=0.28`, `numpy>=1.11`,
   `thrift>=0.11.0`, `six`. Nothing in the JSON, sidecar or wheel had said so, so `thrift` was not a pin
   this layer's date rule narrowed; pip resolved it against PyPI at install time, took the newest
   `thrift` (`0.25.0`), and its `setup.py` opened with `from setuptools.errors import ...`, which the
   layer's pinned `setuptools==46.4.0` does not carry — `ImportError: cannot import name 'CompileError'
   from 'setuptools.errors'`, the measured failure this reading turns into a date-rule pin
   (`thrift==0.13.0`, newest ≤ `2020-05-21`). It is a *declaration* being read, not code being run: the
   two are not interchangeable, and `_file_literals` filters to names with a `.` in the last path
   segment, so a name that is not a file inside the same archive is followed nowhere;

and a pin that reaches none of the **four** is **unknown**, not empty. `setup.py` is deliberately **not**
executed to find out: reading a pin's declarations must not mean running the pin's code. The first three
steps are not interchangeable, and the measurement that says so is `pytest-json-report`: every release has
`requires_dist: null` in its JSON *and* a wheel with a sidecar, and 0.1/0.2's sidecar declares **no**
`Requires-Dist` at all while 0.3+ declares `pytest>=3.3.2` — so the chain is what proves `httpie/1`'s
plugin stays at `0.2`, and a reader that stopped at the null would have had to guess. The fourth step is
held to the same rule — a computed `install_requires` whose file the archive does not carry, or whose read
raises, answers `None` and stays unknown; it never answers `()` — and it is behind `install_requires_of`,
an injection point named apart from `preflight_pin`'s existing `sdist_of` (which is `sdist_file_list`), so
a caller can see which of the two archive readings it is passing.

**The fourth reading widened `pandas/1`'s system-package review by one entry.** `thrift==0.13.0` is now a
date-rule pin of that layer and an unreviewed sdist build would be a `gap`, so
`bake.SDIST_SYSTEM_PACKAGES` gained `"thrift": ()` — an **empty** tuple, the same shape as the
pure-Python entries: `thrift-0.13.0.tar.gz` builds `Extension('thrift.protocol.fastbinary',
sources=['src/ext/{module,types,binary,compact}.cpp'], include_dirs=['src'])`, and the measured reads say
those four `.cpp` files and every header they include (`ext/{types,binary,compact,protocol}.h`,
`ext/endian.h`, `ext/protocol.tcc`) are **in the sdist** and `include_dirs=['src']` names the archive
itself, so the only includes are those, `<Python.h>`, `<limits>` and `<stdint.h>` — there is no
`libraries=`, `library_dirs=`, `pkg-config` or `ctypes.util` call at all. It is an empty tuple and not a
package list because the build needs no system package; and it does not break the pure-Python fallback
either — `setup.py`'s `except BuildFailed: run_setup(False)` installs the extension's Python shim alone if
the compile fails, so the extension is an optimisation for this layer, not a requirement.

**Unknowns are listed by name and do not block the bake; the layer's own `pip check` is what covers
them.** A pin nothing could be read for is recorded as `bake.CASE_UNKNOWN` and printed under its own
`UNKNOWN` heading — `httpcore==0.3.0` is the measured example, sdist-only with no sidecar, whose
`h11==0.8.*` therefore reaches neither the preflight's case (b) nor the date rule's narrowing and is
found by the layer's own `pip check` after the build. Calling it refused would fail a layer over a fetch
that did not happen, so it stays outside the verdict: `bake.preflight_verdict` counts only case (a)
conflicts, `gap`s and flagged bugs. Two places read a release and have to live with an unread one —
`bake._narrowing_requirements` and `bake.newest_accepting_pytest` — and neither may read "could not be
read" as "accepts anything": the first narrows nothing and the second skips the release.

**The terminal prints one line per layer and each finding once, and ends on a verdict.**
`bake.preflight_lines` prints exactly **one** line per layer —
`[<checkpoint>] <key> (<target>): N pin(s), a wheel, b sdist-only, c unavailable, d gap, e unbuildable,
f dropped` — so a bug with nothing to report still appears (a table with the quiet bugs missing cannot
say whether they were checked), and a bug's pure-Python sdists are **not** listed: a noticed
disposition is a decision nobody has to take, and those live in `<checkpoint>.preflight.json` alone.
Then each **finding** is printed once for the *whole split* rather than once per bug that carries it —
a deduplicated `unreviewed sdist build(s) (N)`, `unbuildable pin(s) (N)`, `pin(s) nothing can
install (N)` block, and then §3.2's conflict cases as **three** more — `case (a)` first (`REFUSED`; it is
the only one that stops a layer, so a reader who reads one section reads that one), then `case (b)`
(`TOLERATED`), then the pins nobody could read (`UNKNOWN`) — each line `<LABEL> <pin> — <bug keys>` —
because the pins the split shares are why a reader is there (numpy alone is in six of the eleven
dev-split bugs) and one decision printed six times is six times the reading. One block is not a finding
and is printed for that reason: `narrowed pin(s) (N)` — `NARROWED <pin> — <why> (<bug keys>)`, the
date-rule choices a pin in the same layer moved (`bake.pin_narrowings`, both directions of case (a)),
which count in no verdict because the decision is already taken and which a reader cannot see in the pin
list itself. The `<why>` is the **claim** itself — the declaring release and the clause it wrote, as in
`NARROWED cython==0.29.14 — numpy==1.18.4 declares >=0.29.14` — printed **once per distinct claim** and
with the bug keys deduplicated (`bake._narrowing_section` keyed both maps), because the inputs are
per-claim and the same pin has one reason for one move; and a claim that wrote **no** clause is not a
reason at all, dropped by the reader that writes it (`bake.narrow_build_pins`) and skipped again by
`bake.pin_narrowings`. What `luigi/1` printed before this — `NARROWED setuptools==42.0.0 — pip==21.3.1
declares` with an empty specifier, the wrong declarer, and seventeen copies of one bug key — was neither
the fact nor the count. A pin the unresolvable-pin
rule dropped is not repeated as a finding; it appears once in the drops block below, with the audit that
settled it — and a conflict with a **dropped** pin on either side is not printed either, because the
layer will not install that pin and no pair is left to conflict. A `--preflight-only` run also
leaves out the three **per-layer build lines** — `[<checkpoint>] installs:`, `[<checkpoint>] build
pins:`, `[<checkpoint>] provenance:` — because they are the build's own log and that run is not
building; they are written into `runs/bugsinpy/deps/<checkpoint>.json` either way, so the record is
unchanged and only the terminal is quieter, which is the point of a run whose product is the findings
below. A real bake still prints them, one per layer. `--preflight-only` then ends
on a single line — `bake.preflight_verdict` — `ready to bake` when there are 0 gaps, 0 **case (a)**
conflicts and 0 flags, otherwise `not ready to bake: N unreviewed sdist build(s) across M bug(s), C pin
conflict(s) across B bug(s), none of them §3.2 case (b), F flagged bug(s)`; the counts are in it because
"not ready" without them says nothing a reader can act on. Case (b) is not counted — it is the
benchmark's environment, tolerated on purpose, and a verdict that called it "not ready" would be asking
for a decision §3.2 already took — and neither is a case (?) reading, which is a fetch that did not
happen rather than a conflict.

**What the check says about the two layers whose build logs were lost.** Both were read at HEAD by a
`--preflight-only` run, so both halves are **measured** — each `*.preflight.json` under
`runs/bugsinpy/deps/` now carries the `conflicts` and `narrowings` keys as well as the classification —
and every number below is that file's own `counts`:

- **`luigi/1` — `ready to bake`.** 198 pins (151 wheel / 47 sdist-only), 0 unavailable, 0 gap, 0
  unbuildable, 8 early. 198 is its 192 `installs` plus its six build pins (`pip`, `setuptools`, `wheel`,
  `cython`, `pytest-runner`, `setuptools-scm`), none of which it also installs — the table is the union of
  the two sets, and here it happens to add up without an overlap. 4 pins are dropped, each with the audit that settled it: `pywin32==227` and
  `requests-async==0.5.0`, which nothing pip can install on this image, and `cryptography==2.8`/`==2.9.2`,
  the unbuildable pair, which reach the list through §3.2's one rule for a pin nothing can install —
  every one of them with "nothing in either tree imports it" as its audit. One conflict, **case (b)**:
  `beautifulsoup4==4.9.1` declares `soupsieve<2.0` against `soupsieve==2.0.1`, and both sides are
  BugsInPy's own exact pins — this layer has 154 of them — so it prints as `TOLERATED` and does not enter
  the verdict. Its other 26 readings are case (`?`) (`blinker==1.4`, `distlib==0.3.0`, `httpcore==0.3.0`,
  ...), which is a fetch that did not happen rather than a conflict. **Six pins are narrowed**, 19 readings
  in all, and both of §3.2's directions are among them: three build pins are narrowed *by* the layer's own
  pins (`setuptools==42.0.0` by eleven of them, `cython==0.29.14` by `numpy==1.18.4`,
  `setuptools-scm==3.4.1` by three), and three dependency pins are the **refusal triple** — the record
  names the release each moved off and the pin that refused it: `botocore==1.12.149` (refused by
  `urllib3==1.25.9`), `boto3==1.9.149` (refused by `botocore==1.10.84`) and *both* `moto==1.3.8` and
  `moto==1.3.7` (refused by `boto3==1.7.84` and `botocore==1.10.84`). Its `httpcore==0.3.0` sits next to
  `h11==0.8.1` and is consistent — the same `httpcore` pin as `sanic/1` against a *different* `h11`, which
  is why the two layers' readings are compared rather than assumed to agree.
- **`matplotlib/1` — `ready to bake`.** 33 pins (31 wheel / 2 sdist-only), 0 unavailable, 0 gap, 0
  unbuildable, 0 dropped, **0 conflicts**, 0 narrowings, 3 early. 33 is its 30 `installs` plus its five
  build pins (`pip`, `setuptools`, `wheel`, `cython`, `numpy`), two of which — `cython` and `numpy` — it
  also installs. The two sdist-only pins are both cleared: `numpy==1.18.5` by the toolchain alone,
  `pillow==7.1.2` by the reviewed `libjpeg-dev`
  entry. Its three early pins are the build requirements `numpy==1.18.5` declares (`setuptools`, `wheel`,
  `Cython>=0.29.14`), answered by `setuptools==47.1.1`, `wheel==0.34.2` and `cython==0.29.19`. **No pin
  in this layer is BugsInPy's own** — every one is the project's own declaration at one of the two
  commits — so nothing here can be case (b), which needs both sides to be the benchmark's; a conflict
  would be case (a) and refused, and there is none. Three of its pins move for reasons §3.2 now states:
  the pre-release rule takes `pyparsing` off `3.0.0a1` to `2.4.7`, `jinja2` off `3.0.0a1` to `2.11.2` and
  `markupsafe` off `2.0.0a1` to `1.1.1` (which is also what left the second sdist and the one gap in that
  layer's earlier reading), while `cython` and `numpy` are pinned to the dependency set's own pins rather
  than to a date-rule pick.
- `pip==21.3.1` — the capability pin, a `py3-none-any` wheel — is in `build_pins` for **both** layers, and
  is the pin that answers `multidict==4.7.6`'s `pip>=18` build requirement on `luigi/1`; it needs no
  release dates, because §3.2 pins it by capability and never reads its listing.
- The unknowns are the caveat rather than a finding: PyPI's `requires_dist` is `null` for 26 of
  `luigi/1`'s 198 pins and for **none** of `matplotlib/1`'s 33, and the chain resolves a null pin whenever
  its release publishes a sidecar or a wheel — so what stays unknown is the sdist-only half of the first,
  measured shape `httpcore==0.3.0` (also `wrapt==1.11.1`, `sqlalchemy==1.3.3`, `avro==1.8.2`, ...). They
  are printed by name under `UNKNOWN` and do not change the verdict.

**One rule for a pin nothing here can install, and it is the import closure that decides.** The
preflight produces three answers a *bake* cannot act on: a release PyPI does not have at all
(`luigi/1`'s `requests-async==0.5.0`), a release with no distribution this interpreter can install
(`sanic/1`'s `pywin32==227`, twelve wheels and every one `win_amd64`, no sdist), and a pin
`bake.UNBUILDABLE_ON_PLATFORM` reviews as impossible on this image (`pandas/1`'s `pyarrow`). All three
are the same situation — the pin is in the bug's own files and cannot become a fact about the layer —
so they get one rule rather than three — and the rule is about the **graded tests' import closure**,
not about "is this module mentioned somewhere". `bake.decide_unresolvable` drops the pin with a record
when

* every import site in the tree is **optional** — inside a `try`/`except ImportError`, behind
  `pytest.importorskip(...)`, reached through `import_optional_dependency("name")` by string, inside a
  `TYPE_CHECKING` block (`type-checking only`), or inside a function or method body
  (`lazy: inside <qualname>`) — each recorded on the site as `ImportSite.optional` by
  `_collect_imports`, so the audit prints the reason it did not count — **or**
* no site at all is in the **import closure** of the bug's failing test files — the split's declared
  `test_file`, the same declaration `prepare._declared_by_bugsinpy` filters `FAIL_TO_PASS` by — and
  that closure;

and **flags** the bug only when the failing tests or their closure import it **unconditionally**. The
closure is walked by `bake.import_closure` over `bake.module_import_graph`, rooted at the declared test
file; `bake.import_auditor` reads both commits' trees, because the graded tree is the buggy source
carrying the fix's tests. A file that is not in the graph makes the audit conservative — the whole
graph is treated as the closure.

**Two structures that read like imports are not imports the module makes.** An import inside
`if TYPE_CHECKING:` — the bare name, `typing.TYPE_CHECKING`, an alias (`from typing import TYPE_CHECKING
as TC`), or the `if False:` that says the same thing without the import from `typing` — never executes:
`TYPE_CHECKING` is `False` at run time, so the block is read by a type checker, the distribution can be
missing, and the module still imports. That is the whole point of writing it that way. It is recorded as
**`type-checking only`**. An import inside a function or method body runs when *that function* is
called, not when the module is imported, so importing the suite does not need the distribution either;
it is recorded as **`lazy: inside <qualname>`**, dotted so a method reads `Dtype._arrow` and a nested
function `outer.inner`. `bake._never_runs` matches the guard in its **exact** form — the name, the
`typing.TYPE_CHECKING` attribute, the literal `False` — because `if not TYPE_CHECKING:` and
`if x or TYPE_CHECKING:` *do* run, and a rule with a false positive drops a pin a graded test needs,
which is the failure mode this rule exists to avoid. A **module-level unguarded import is unchanged**: it
is still an unconditional site, and still what makes the pin required.

**Both reasons are the audit's, and neither one touches the install set.** `bake._scope_only_reason`
marks exactly these two as reasons that answer the *audit's* question — "may a pin nothing here can
install be left out of the layer?" — and never the closure's question — "which packages does a layer
carry?". `bake.test_import_closure` installs them like any other import, exactly as it did before the
reasons existed; they are read only by `bake.tree_import_sites` → `bake.import_auditor` →
`bake.decide_unresolvable`, and only for a pin that is class (c) unavailable or
`UNBUILDABLE_ON_PLATFORM`. The distinction is not cosmetic, and it is **measurably larger than the three
the tests ask for by name**: read as closure reasons they took `pandas/1`'s merged pin list from the **41**
names it is at `39063e5` to the **22** at `4615f27` — nineteen names gone, printed as **24 pins** once the
two plugin pins are added, because `bs4`/`sklearn`/`psycopg2` are only the visible edge. The nineteen are
`beautifulsoup4`, `botocore`, `cftime`, `dask`, `fastparquet`, `ipython`, `jedi`, `numba`, `numexpr`,
`openpyxl`, `psycopg2-binary`, `pymysql`, `s3fs`, `scikit-learn`, `scipy`, `sqlalchemy`, `statsmodels`,
`xarray` and `xlrd`; `llvmlite` is in neither commit's refs at all, because nothing in either tree imports
it. The fix restores the list name for name, and the regression test asserts that **whole** list — the
merged list and not the closure, since it is the merged list the preflight counts — so the only difference
left is §3.2's drop of the two pins this image cannot build, `numba` and `pyarrow`, each with its audit
(`41` refs → `39` installs). The bug is exactly this: the tests that ask for a missing package *skip*, and
no failure set can see a skip (measured below). What the audit reads on `pandas/1` starts at
`pandas/core/arrays/boolean.py` at `3fd150c`:

    if TYPE_CHECKING:
        import pyarrow  # noqa: F401

so the block is one of `pyarrow`'s recorded sites — beside `pandas/core/arrays/_arrow_utils.py:5`, the
only *unconditional* one, and the in-method sites below — and it is why `pyarrow` may be dropped from a
layer this image cannot build it into. The install set carries the pin until that drop removes it.

The drop is a prediction like any other, and it is safe for the same reason: `prepare`'s gate — the buggy
tree must fail and the fixed tree must pass — is the measurement, and a bug whose gate fails names the
pins the layer went without (`prepare.dropped_pin_note`, at every gate exit). **Three of `pandas/1`'s
names are on the test side and are installed anyway**, and that measurement is what corrected the scope:
`import bs4` inside `test_bs4_version_fails` (`pandas/tests/io/test_html.py`), `from sklearn import svm,
datasets` inside `test_scikit_learn` (`pandas/tests/test_downstream.py`), and `import psycopg2` inside
`TestToDatetime.test_to_datetime_tz_psycopg2` (`pandas/tests/tools/test_to_datetime.py`) are `lazy:
inside <qualname>` sites, and those reasons are the *audit's*. Read as closure reasons — which is what
the reasons were first wired into — they took `beautifulsoup4`, `scikit-learn` and `psycopg2-binary` out
of the layer, and the tests that ask for them then **skipped**. Two of the three are
`@td.skip_if_no(...)`-decorated, which is what hid the loss: a skip is not a failure, so nothing in the
gate moved and the layer that had lost three packages looked exactly like the layer that had them. The
reason is still recorded on each site and still read by the audit; the closure installs the three.

**A skipped graded test is a gate failure; a skip anywhere else is a number.** `suite.FAILURE_OUTCOMES`
is `{"failed", "error"}`, so a graded test that *skips* on the fixed commit is in neither `F_before` nor
`F_after`: the flip `F_before \ F_after` reads it as one the fix makes pass, `FAIL_TO_PASS` declares it,
and `oracle.score` — reading the same report — counts it as *not* passing. A hidden test that can never
pass fails every candidate for a reason that is not the bug, and reconciling the two readings after the
batch is baked is too late — the bug is in the run and its oracle is unsatisfiable. So
`prepare.graded_skip_failure` checks every `FAIL_TO_PASS` nodeid against `suite.SuitePair.stable_outcomes`
on **both** the buggy commit (`src_buggy`) and the fixed commit (`src_fixed`) and appends a gate failure
naming the nodeid and the commit; `oracle_tree` is deliberately not consulted, because it is the buggy
source carrying the fix's tests, where every declared F2P is by construction a stable failure. A graded
test skips almost always because the layer is missing something it needs (`@td.skip_if_no`,
`pytest.importorskip`, `pytest.mark.skipif`), so the reason is one the `dropped_pin_note` clause extends
with the pins the bake dropped — the skip gate is how the `pandas/1` loss would have been caught at
prepare time rather than looked at as a pass. **Never treated as a pass.** The non-graded native suite is
*not* gated — a skip in it is a legitimate outcome — but `prepare` records it per commit as
`BugPrep.native_skips_buggy` / `BugPrep.native_skips_fixed` (each derived by `prepare.skip_count`, both
in the derived `split.DERIVED_BUG_KEYS` and in the record's round-trip), because it is the only signal
that a layer lost a dependency: no failure appears in any failure set, so the run that dropped three
packages and the run that had them look identical except for the skip counts.

The distinction is measured, and it is where the rule was wrong before. **`pandas/1` is not flagged, it
drops.** `import pyarrow` does appear in **18** `.py` files at the fix's commit, but the only
*unconditional* site is `pandas/core/arrays/_arrow_utils.py:5`, every importer of `_arrow_utils` is a
lazy in-method import (`integer.py:113`, `interval.py:1102`, `period.py:301`, `dtypes.py:1024`,
`io/parquet.py:86`), and `pandas/1`'s failing test is `pandas/tests/dtypes/test_dtypes.py` — nowhere
near parquet. Counting mentions made a pin required that pandas works without; the closure is what the
suite actually reaches. `numba`/`llvmlite` follow the same rule on the same reasoning, as does every
other pin the preflight reports as uninstallable.

The condition is not decoration. A dropped pin makes the layer *shorter than the bug's own files
declare*, and if the graded tests really import it, the suite then fails at `import` — a native
regression invented by the harness and attributed to the patch under test, which is the one failure
mode §1.2/§1.3 exist to avoid. Dropping a pin nothing the tests reach imports is the opposite:
`luigi/1` and `sanic/1` lose a Windows pipe wrapper and a version of `requests-async` that was never
uploaded. Measured by `git grep` over `*.py` at both commits: **zero** import sites for
`requests_async`, `win32` and `pydivert` in either repo, so those two pins drop with a record and both
bugs still bake. `pydivert` is not even a drop candidate — it ships a `py3-none-any` wheel, i.e. class
(a).

The drop is a **prediction**, and `prepare`'s gate is the measurement that can refute it. So a bug that
had a pin dropped names the drops in **every** reason its gate produces — `prepare.dropped_pin_note`,
applied at each gate exit because `check_deps` and `strip_and_baseline` record their own reasons deep
inside themselves. If the buggy source then fails to import a module the bake took out, the failure says
which pins were dropped and that the drop is why, so the reader can tell a broken bug from a wrong
prediction — the two call for opposite responses, fix the bug or put the pin back.

The drop has to run on both sides of the date rule, which is why `bake.resolve_deps` applies it in two
passes rather than inside `bake.resolve_pins`. `luigi/1`'s `requests-async==0.5.0` is a BugsInPy
reference pin (`requirements.txt`/`setup.sh`), so the date rule refuses the whole bug on it and the pin
must be judged **before**; `pandas/1`'s `pyarrow` comes from the test side's import closure with no
version attached, so only the *resolved* pin (0.17.1) can be judged at all, and that is **after**.
`resolve_pins` itself stays pure and keeps raising on a pin it cannot date — the passes around it are
what turn "cannot resolve" into either a record or a flag.

The decision itself (`bake.decide_unresolvable`) raises `BakeError` when something the graded tests
reach imports the pin, so flagging needs no new control flow: `resolve_bug_deps` already catches that
and files the bug with its reason, which names the sites the audit found — each with its file, line and
commit. **Nothing is silent.** A dropped pin is written to `runs/bugsinpy/deps/<checkpoint>.json` under
`dropped` — a key that is always present, `[]` when nothing was dropped, each entry
`{name, spec, pin, sources, outcome, reason, imports}`, where `imports` is the audit
(`{required_by, reason, sites}`: the sites found and *why each counts as optional*, or why the closure
never reaches one). A dropped pin is **not** printed per bug either: the same pin is usually dropped
from several bugs, so `bake.preflight_lines` prints the whole split's drops once, as
`<pin> (<outcome>) — <reason>` followed by the bug key and the audit's reason. The JSON and that block
are the same records and neither summarises the other: the block is what a reader watching the bake
sees, and the JSON is what survives it — including the per-site audit, which the block does not repeat.

**A transitive pick is a pin like any other, and the same rule judges it.** The drop above is applied
by `bake.drop_unresolvable_deps`, and that function is called **twice**: once inside `resolve_deps`
over the layer's *declared* pins, and once by `resolve_bug_deps` over the pins the **closure walk**
picked, which do not exist until `resolve_deps` has already returned. That second call is the whole
of the fix, because it is the half the rule was measured on and the half it was missing: `luigi/1` and
`sanic/1` both install `cryptography==2.8` because `moto==1.3.6`, `pyopenssl==19.1.0` and
`service-identity==18.1.0` declare it, `bake.UNBUILDABLE_ON_PLATFORM` reviews exactly that release as
impossible on this image, and each layer's *own* `cryptography` pin (2.9.2) **was** dropped for that
reason — while the transitive 2.8 went into the layer. A pick nothing here can install is the same
finding whichever half of the layer produced it: it is dropped with the same record when nothing the
graded tests reach imports it, and the bug is **flagged** by the same `bake.decide_unresolvable` when
something does. To keep the two halves one rule rather than two, `bake.resolve_bug_deps` binds the
rule's two callables **once per bug** — `reason_of` (the date rule's own release mapping, the run's
shared classifier, the layer's minor version) and `audit` (`bake.import_auditor` over both commits'
trees) — and hands the *same* two to `resolve_deps` and to the pass after `bake.transitive_pins`, so
neither half can drift. The record lands in the one `dropped` list either way, and the preflight's drop
block reads the two together. A pick has no `sources` — no file in either tree named it, which is what
made it a pick — so its record falls back to `required_by`, the releases whose `requires_dist`
declared it, which is what makes the entry name a distribution *and* a reason it is in the layer at
all.

**Rebuild semantics — `--force` means replace, and a rebuilt base invalidates its children.** A layer
is deleted *before* it is created, never merely created over (shuru refuses a duplicate name with
`checkpoint '<name>' already exists, delete it first`; `checkpoint_create` treats any non-zero result
as "the name must not survive", so a colliding create destroys the layer it collided with — which is
exactly how a `--force` rebake once named `vanta-bi-PySnooper-1` and then deleted it). `bake.plan_layers`
decides the whole run before touching anything, in this order:

1. **delete** every selected layer that exists and is being rebuilt, **children before their base**;
2. **delete** the *stale children* of any rebuilt base — the per-bug layers stacked on an interpreter
   layer rebuilt in this run, whatever minor version they belong to, **even if nobody selected them**.
   A layer baked from a base that no longer exists is not the layer the selection asked for, so it is
   removed and listed with its reason rather than left for `prepare` to run a suite in;
3. **build** base before child (`layer_recipes` order).

A layer outside the selection is never touched except as a stale child in step 2.

**Staleness is existence, not content: a recipe edit is invisible until the layer is replaced.**
`bake.plan_layers` decides from `layer_exists(name)` alone — a layer baked from an older recipe is
"already baked" and is skipped — and nothing else can see the edit, because the recipe is read only by
the bake that writes it. So a change to a recipe requires `--force` for every layer it touches, and the
two levels are told apart by which function the edit lands in. `bake.python_recipe` is the interpreter
layers', so the `FFLAGS` export/assertion and the Fortran smoke file above are an interpreter-layer
change: `vanta-bugsinpy-py3.8` was baked before it, and every bug layer stacks on that layer, so
`--force` is what the run wants as a whole — without it the plan deletes the interpreter's children as
stale (step 2) and rebuilds none of them, which is a run that ends with no usable layer at all.
`bake.requirements_recipe` is the per-bug layers', so §3.2's `--no-deps` pass and the autoload-disabled
`pytest --version` require `--force` for every `vanta-bi-<project>-<N>`.

Two edits of the current wave land in the interpreter half for the same reason and with the same cost.
`COMPILER_ENV["CC"]` now names `bake.VANTA_CC_PATH` instead of `gcc` (§3.2, the FreeType 2.13.3 conversion
on matplotlib's `ft2font.cpp`) and `build_env_lines` writes a second wrapper file beside the first, so both
the exported `CC` line and the recipe tail below it change; `COMPILER_ENV["FFLAGS"]` was added to the same
mapping a commit earlier, which is the same kind of change. The wave's third edit to that region does not:
`SDIST_SYSTEM_PACKAGES["thrift"] = ()` adds an **empty** tuple, so `_system_packages_for_py38()` is
byte-identical and the interpreter recipe with it. The next bake is therefore `--force` for every layer in
the wave — the interpreter layer and each per-bug layer, whether or not that layer has a checkpoint already
— because every per-bug layer is `--from` a base whose recipe has moved, and a layer left in place would be
one whose build log no longer describes its base.

**One failed per-bug layer does not stop the others; a failed interpreter layer stops everything above
it.** The per-bug layers are independent by construction — each is baked `--from` the interpreter layer
and none is baked from another — so a bug that cannot resolve its dependencies says nothing about the
next one, and stopping at the first turns an eleven-bug run into eleven runs to learn eleven facts.
Every per-bug failure is therefore collected and reported **together at the end**, once the rest have
been built. The interpreter layer is the one exception: every layer in the run is stacked on it, so a
failure there would fail each child too, and each of those names would burn a bake cycle to learn the
same thing. The run stops there, and the layers after it are labelled `[not started]`.

**The summary labels what happened, and the log carries the layer's own provenance.** A failed build's
checkpoint is deleted and the deletion *verified*, because `shuru checkpoint create` saves a checkpoint
*before* reporting the inner command failed — so the one-line-per-layer summary can say that nothing was
left behind and mean it, naming the survivors instead when a delete does not take (a name that survived
is a name `prepare` would accept, which is why it is named rather than the claim being softened). Each
layer is labelled from what actually happened: `[done]` only for a layer that was built and whose marker
was verified, `[FAILED]`, `[not started]` for one a failed interpreter layer stopped the run short of,
`[deleted]` for a stale child removed, `[skipped]` for one this run did not touch. `[done]` was
previously printed for every layer the *plan* said to build, including the four after a failure that
were never attempted — a list of names a reader, and `prepare`, would take for layers that exist.

**One line per failed layer, read from the layer's own log, and every layer's whole log on disk.** The
nine-bug run's error printed the **tail** of the last failure and nothing else, so two of the three
failures were represented by their layer names alone — and what that tail ended in was
`shuru: checkpoint 'vanta-bi-…' saved`, because `shuru checkpoint create` saves the checkpoint *before*
reporting that the inner command failed (the delete below is written around the same fact). The tail of a
build log is therefore not a description of its failure, and "the last meaningful line" read off it named
the layer rather than the reason, on all three. Three rules follow.

`bake.last_error_line` is a **two-pass** reader over a layer's merged output, and the lines both passes
skip are dropped once, in one list: shuru's own output (`bake.SHURU_LOG_PREFIX`, `shuru:`) — never the
failure — blank lines, and `set -x` traces (`+ …`), which are the recipe and not its result. The first
pass takes the last line starting with one of `bake.ERROR_LINE_PREFIXES` — `ERROR` (pip's
`ERROR: ResolutionImpossible`, `ERROR: Cannot install …`), `error:` (a compiler's
`error: command 'gcc' failed`), `fatal error` (a C compiler's fatal diagnostic) — because that is the one
line of a two-hundred-line log that states the reason rather than the progress. The second pass, for a
build that says nothing that way, takes the last line long enough to be a message rather than a fragment:
`failed (exit 1): i` is what a bare tail produced, and a one-character fragment is not an error message.
A **prefix** test and only that, because it is what a reader of the log can act on without guessing which
of two hundred lines matters; a compiler that prefixes its fatal diagnostic with the file it was reading
(`numpy/core/setup_common.py:9:10: fatal error: …`, which gcc does) is not matched by the first pass and
falls through to the second — a loss of precision and not of information, since the line is still printed
and is still the line that says the most.

`bake._failure_report` prints that line for **every** failed layer — `<layer>: exit <N>|timed out —
<line>` — and not only for the last one, for the same reason `run_bake_actions` collects the failures
instead of stopping at the first: one line each is what makes three failures three facts rather than one.

Each attempted layer's **whole** build output is written to `runs/bugsinpy/bake-logs/<layer>.log`
(`bake.write_bake_log`), with the path printed beside the layer's line and the directory named at the
start of the run. The directory is emptied first (`bake.clear_bake_logs`), **overwritten per bake**, so a
layer this run did not touch cannot leave a file that reads as this run's. That is what makes the
one-line error enough: the line says what happened and the file holds the two hundred lines above it,
which is where the reason usually is — pip's resolution error is followed by shuru's own
`checkpoint … saved`, and a compiler's `fatal error` is followed by the recipe's remaining commands. The
directory is the harness's own scratch output and never a source of truth (the layer's checkpoint and its
deps JSON are), which is why replacing it per bake loses nothing.

A built layer also echoes its own `/opt/vanta-build-env.txt` — the `CC`/`CXX`/`LDSHARED`/`LDCXXSHARED`
the build actually exported, not the ones the recipe intended — and the interpreter layer's toolchain
smoke result into the bake log, so the log of a run is the provenance of the layers it built. The echo is
produced *inside* the build between two sentinel lines (`vanta build env (…)` / `vanta build env end`)
and re-printed by `bake.run_bake_actions`, rather than read back with a probe that boots each layer a
third time: the interpreter layer already boots twice to verify its marker, and a boot per layer would be
paid twelve times for a file the build itself just wrote.

`bake` exits non-zero on any failure, and the commands are chained with `&&` so `prepare` never runs
against a broken set.

**One existence check, shared.** `bake.layer_exists` (→ `shuru.checkpoint list`, matching the first
token of each line) is the only function that answers "does this layer exist". `prepare`'s preflight
calls it too, rather than holding its own import binding, so the two stages cannot disagree about a
name — only about a name that moved between them, which `plan_layers` above is what prevents.

**Where the specs come from — every source, read at both commits, unioned per bug, resolved at bake
time.** BugsInPy's own install step is `setup.sh` (`python setup.py install`, plus extra `pip install`s),
and `requirements.txt` is frequently empty — 0 bytes on all three PySnooper bugs, while `python_toolbox`
and `decorator` exist only in `setup.sh` at bug 3. One source alone under-provisions, so
`bake.resolve_bug_deps` unions five sources:

1. the BugsInPy `requirements.txt` for the bug;
2. the `pip install …` lines of the BugsInPy `setup.sh` (parsed, measured — never guessed);
3. the project's own declared **test** requirements (`test_requirements.txt`, `tests_require`, a
   `tests` extra in `setup.py`, or a test extra in `pyproject.toml`);
4. the project's own runtime dependencies, read **statically** from `setup.py` (`install_requires`,
   an AST walk, plus the file it names — PySnooper/1–2 read `requirements.in`, PySnooper/3
   `requirements.txt`) **or** from `pyproject.toml` (PEP 621 / flit / poetry — see below);
5. the test side's **import closure** — the third-party top-level modules the test-side files
   themselves import (`bake.test_import_closure`).

**A `setup(**kwargs)` splat is resolved from the AST, never by running it.** Source 4 reads the
`setup()` call's keyword arguments, and a project may pass them as a splat of a module-level dict
rather than inline. `sanic/1` (`e7001b00`) is the measured case: its `setup.py` ends with
`setup(**setup_kwargs)`, and `setup_kwargs` a few lines above is a module-level dict literal carrying
`name`, `packages`, `install_requires`, `tests_require`, `extras_require` and `cmdclass`, with some
keys assigned by subscript afterwards (`setup_kwargs["extras_require"] = …`). Read inline-only, that
bug's `install_requires` is invisible and `import_name` cannot be read at all — the bake refused it
with "declares neither `name=` nor a `packages=[…]` literal" over a file that declares both. The
reader therefore dereferences a splatted name to its module-level assignment, takes the dict
literal's **string keys in source order**, overlays the name's subscript assignments, and reads the
same keys (`name`, `packages`, `install_requires`, `tests_require`, a `tests` extra) from the result.
A splat whose value is not a dict literal — a call, an import, a comprehension — contributes nothing
and the commit keeps today's refusal: the reader will not execute `setup.py` to find out what
`**kwargs` holds, because that is third-party code outside Shuru (C3). This is source 4, and it
applies to every wave.

**A bug the metadata cannot name may carry a reviewed `import_package` — last, and only last.** One
bug's `import_package` names the module its import-path check loads, as an **optional** declared key
(`DECLARED_BUG_KEYS` stays the *required* set, so no other bug's entry changes), and the decision is
recorded in that bug's `reviewed_changes`. It is used only when the metadata itself cannot name the
import package — `import_name_of`/`import_name_of_pyproject` raising, after the splat resolution
above has had its turn — so it is a fallback and never a preference: metadata that names the package
always wins, and the fallback is noted in the bug's provenance when it fires. It names the import
path only; the bug's dependencies still come from the five sources above. This applies to every
wave, not only the dev split.

**`setup.py` or `pyproject.toml` — whichever the commit actually carries.** Packaging moved, and a
bug's dependencies are wherever its own metadata put them, so sources 3 and 4 are read from
`setup.py` *or* `pyproject.toml` at each commit, never assumed from one file. `fastapi/1`
(`766157bf`) is the measured case: it is a **flit** project with no `setup.py` at all, so reading
`setup.py` alone refused the bug at `bake` before `prepare` ever ran. The reader (`bake.pyproject_refs`)
covers three spellings, exactly as written, parsed — never executed:

- **PEP 621** `[project] dependencies` and `[project.optional-dependencies]` (the standard);
- **flit** `[tool.flit.metadata] requires` and `[tool.flit.metadata.requires-extra]` (fastapi's shape:
  `requires = ["starlette ==0.13.2", "pydantic >=0.32.2,<2.0.0"]`, `requires-extra.test`, `.doc`,
  `.dev`, `.all`);
- **poetry** `[tool.poetry.dependencies]`, when present (`python` itself is skipped — it names the
  interpreter, not a distribution). Poetry's constraint syntax is not PEP 440 — `^2.22` is a caret
  range, `~1.4` a tilde range, `1.4.*` a wildcard — and translating a caret range would mean
  reimplementing poetry's resolver here, so only the spellings that can be read *exactly* are taken
  (a bare version, whose poetry meaning is `==`; a PEP 440 specifier written as such; `*` as no
  constraint) and the rest are **recorded and skipped**, never guessed at.

A **test extra** — a key named `test`, `testing`, `dev`, or `all` — counts as a declared test
requirement the same way `setup.py`'s `tests` extra does, and **every extra that was taken is
recorded** (`DepRef.source` names the table *and* the extra, e.g.
`project.optional-dependencies['test'] (pyproject.toml)` or
`tool.flit.metadata.requires-extra['test'] (pyproject.toml)`), so the
record says which table the requirement came from rather than only that it was declared. The
resulting refs go through the same rules as `setup.py`: stamped with the commit they were read from,
read at **both** commits, unioned, and each held to its own commit's date. `import_name` is read the
same way — from `[project].name`, `[tool.flit.metadata].module`/`name`, or `[tool.poetry].name`, with
the same dashes→underscores canonicalisation — because a flit project has no `setup.py` to name the
package. A commit that has **neither** `setup.py` nor a readable `pyproject.toml` keeps today's
refusal, per bug; an unparseable `pyproject.toml` is a refusal too, never a silent empty read.

**A refusal is one bug's, never the batch's.** `bake` and `prepare` both read the metadata per bug, so
a bug whose dependencies cannot be read is recorded as **flagged** with the reason, the rest of the
split carries on, and the flagged bugs are printed as a list at the end of the run — the same shape
`prepare` already uses for a bug whose `prepare_one` raised (`prepare.json`'s `breaks`). That list is
printed by **every** `bake` run and not only by the ones that finish: the layer build raises on its
first failure, so a run that dies at `matplotlib/1` never reached the old end-of-command print, and a
reader of that run could not tell a flagged bug from a bug that never got there. When nothing was
flagged the line says `flagged bugs (0)` — the same statement about the run, not a line to leave out.
The one bug
that stops everything is one the *harness* cannot proceed on at all (a missing checkpoint, a moved
BugsInPy clone), and that is exit code 2. A flagged-unreadable bug is **not** either of the two §14.7
gates — the §1.6 context gate and the §8.2 dev-fix self-check — so it never silently promotes a
reserve: the flagged list is printed **separately** from the gate results, and promoting a reserve to
replace a flagged bug is a decision taken by hand (amend the split, §8), not by the harness. This is
also measured: `fastapi/1`'s refusal stopped a whole dev-split bake before a single other bug was
resolved.

Sources 3 and 4 are read at **both commits**, and source 5 at the **fixed** one. That is the graded
tree (§1.2) rather than a preference: the source under test is the buggy commit's and the test side is
the fix's, so a requirement the fix added is a requirement of the tree a run is graded on. Measured:
PySnooper/2's fix vendors `tests/mini_toolbox/pathlib.py`, which does `import six` at module scope —
`six` is named by no requirements file, no `setup.sh` and no `extras_require` at either commit, and
`python-toolbox`, whose sources that helper was copied from, declares no dependencies of its own, so the
union of sources 1–4 is a **no-op** for that bug. The tree under test imports `six`, so the layer has to
have it.

Read the other way, though, the fix is not authoritative either: a requirement the fix *drops* was
still there for the buggy code. So each reference carries the commit it was read from (`DepRef.commit`)
and §3.2's date rule below is applied **per declaration** — a fix-only requirement is held to the
*fix's* date, not back-dated to the buggy commit's. BugsInPy's own two files are the exception: they are
one file pair per bug describing the benchmark, so they are left unstamped and keep the buggy commit's
date as their cutoff.

Source 5's scope, since "the test side's imports" can be read several ways:

- **Only the tree under test's test side** — `conftest.py`, `test_*`/`*_test.py`, and every `.py` under
  a `test`/`tests` directory (`bake.test_side_paths` + `trees.is_test_path`). That is where a vendored
  helper like `tests/mini_toolbox/` lives.
- **Parsed with `ast`, never imported or executed.** Nothing third-party runs on the host (C3). A file
  that does not parse as Python 3 is recorded and skipped: the target interpreter could not collect it
  either, so its imports cannot be imports the suite makes.
- **Third-party top-level modules only.** The interpreter's stdlib, the project's own package, the
  test side's own local modules, and **every module the tree itself ships** are excluded.
  `bake.PY38_STDLIB` is a committed literal because 3.8 predates `sys.stdlib_module_names`, so the
  harness's own Python cannot be asked what is standard.
- **A name the tree ships is never a distribution, wherever in the tree it lives.** The test side is
  not the only thing that can put a name on `sys.path`: fastapi's own `scripts/test.sh` exports
  `PYTHONPATH=./docs_src`, so
  `tests/test_tutorial/test_additional_responses/test_tutorial001.py` opens with
  `from additional_responses.tutorial001 import app` — a directory of the repository's *own* tutorial
  sources, not a package. Measured: the test-side-only exclusion set missed it and the bake died
  asking PyPI for `additional-responses` (404), which is the *consequence* and not the cause. The rule
  is therefore **the directory name of any directory that contains a `.py` file**, at any depth —
  both pytest's rootdir insertion and the project's own runner put directories on `sys.path`
  (`bake.tree_provided_modules`) — **and the stem of a `.py` file that sits directly in a directory
  that is not a package** (that directory has no `__init__.py`), because pytest puts that directory on
  `sys.path` too: matplotlib's `lib/pylab.py` imports as `pylab` for
  `lib/matplotlib/tests/test_basic.py` (measured: the closure otherwise emits a distribution named
  `pylab`, which PyPI does not publish — a 404 — and the bug stops), and pandas'
  `scripts/validate_docstrings.py` as `validate_docstrings` for
  `scripts/tests/test_validate_docstrings.py`. A stem *inside* a package is still deliberately **not**
  counted: `fastapi/requests.py` is importable as `fastapi.requests` and never as a top-level
  `requests`, so counting those would silently drop the real distribution of that name. This applies
  to every wave, not only the dev split.
- **An optional import is recorded, never installed — except the two reasons that are the audit's.** An
  import inside `try/except ImportError` or behind a version check (`sys.version_info`, `PY2`, `six.PY3`)
  is a fallback: it goes in `optional_imports` and not into `installs`, because installing a
  Python-2-only fallback on 3.8 would fail the bake over a branch that can never be taken. An import
  inside a `TYPE_CHECKING` block (`type-checking only`) or inside a function body
  (`lazy: inside <qualname>`) is **not** a reason to leave an installable package out of a layer: those
  two are read only by the audit of a pin nothing here can install (`bake._scope_only_reason`), and
  `bake.test_import_closure` installs them like any other import — measured above as the three
  `pandas/1` names that left the layer and made their tests skip.
- **An ambiguous module name stops the bake.** `import yaml` is PyYAML and `import bs4` is
  beautifulsoup4; no canonicalisation recovers that. The closure names the module and refuses, and the
  fix is a reviewed `module_distributions` entry in the split — a decision recorded in the protocol,
  never a guess in the harness. The table is keyed by the **import spelling as written**, so a dotted
  import needs its full key (`google.protobuf` → `protobuf`, since `google` alone is a distribution of
  its own on PyPI and would otherwise be installed in protobuf's place); a name the table does not key
  falls back to its **top-level word** (`import foo.bar` is provided by distribution `foo`, never
  `foo-bar`). One name is deliberately *not* in the table: a module whose canonicalised spelling
  cannot be a PEP 508 name at all — `_csv` canonicalises to `-csv` — is the interpreter's own, not the
  layer's (measured: pandas/1's `from _csv import Error` reached the merge as `-csv` and stopped that
  bug over a name nobody can publish), so the closure **notes** it and installs nothing. `psycopg2` is
  the case the refusal exists for — two distributions both provide it (`psycopg2` and
  `psycopg2-binary`, same module), so the harness will not choose and a module-level site of it would
  keep the bug flagged until a reviewer records the pick. The pick is therefore the reviewer's, written
  into the table with its reason and its rule in `reviewed_changes` rather than into the harness:
  `psycopg2` → `psycopg2-binary`, because the bug's tests only need the module to import (the SQL tests
  skip without a database server, so no connection is exercised) while the source distribution builds
  against libpq and needs a `pg_config` the sandbox does not have. On `pandas/1` both walks read that
  entry — the closure installs `psycopg2-binary`, and the audit reads the same mapping when a pin of that
  name has to be judged — because the tree's only site, `import psycopg2` in
  `pandas/tests/tools/test_to_datetime.py`, sits inside `TestToDatetime.test_to_datetime_tz_psycopg2` and
  carries a `lazy` reason, which is the audit's and never the closure's. Nothing about it is otherwise
  special: a mapped distribution that *is* installed is pinned by the same date rule as every other
  dependency, and a pin with no wheel for the sandbox's platform fails the bake as an ordinary build
  failure for that bug — flagged, not worked around, and never answered by falling back to the source
  distribution. This applies to every
  wave: the table is carried by *both* split files, so a name mapped for the dev split is mapped for
  the evaluation run too.

Nothing is taken from anywhere else, and no package is ever chosen by hand. The union is **merged to
one entry per package**, and every claim in that group must agree on one version, or `bake.DepConflict`
fails that bug and names every claim rather than picking one:

- an exact `==` pin **narrows** the bounds around it. If the pin satisfies every other claim, the
  requirement *is* the pin — `attrs==19.3.0` (BugsInPy `requirements.txt`) beside `attrs>=17.4.0`
  (`install_requires`) is one requirement, because the reference pin is a legitimate narrowing of the
  project's range. `==1.2.*` is **not** a pin: it matches a range of releases, so it is a bound like
  any other and never the claim a merge settles on.
- a pin that **violates** another source's bound is a conflict — **unless the pin is BugsInPy's own**.
  Two different `==` pins, and two different bounds, are real conflicts and fail loudly: their
  intersection is a version no source wrote, and not being able to reproduce the environment is the
  failure this check exists to surface. But when the only unsatisfiable claim is a *project*
  declaration against BugsInPy's own exact pin — luigi/1 pins `tornado==5.0` in BugsInPy's
  `requirements.txt` while the project's `install_requires` says `tornado>=4.0,<5` — the **pin wins**,
  by the reasoning of the date-rule exception below: it is the environment BugsInPy's own `run_test.sh`
  was validated fail-before/pass-after in, and the §8.2 dev-fix self-check re-verifies that claim on
  that same bug afterwards. The pin is installed as written, and the claim it beat is **recorded** —
  `overridden`, as `{spec, source}` pairs, next to the pin's own source in the provenance — so an
  override is never silent and the record can be read against `setup.py` instead of looking like a
  harness that never read it. A *project* pin that violates a bound gets no such licence: the project's
  own declarations contradict each other and nothing in the harness can say which one the buggy tree
  was built with. This applies to every wave, not only the dev split.

**Pinning.** Every third-party dep is pinned to the newest release published on or before **its own
declaration's** commit date — the buggy commit's date for a reference the buggy commit declared, or for
anything BugsInPy's own files carry; the fixed commit's date for a reference only the fix declares and
for every closure entry (`bake.resolve_pins`, the only PyPI contact, at bake time while network is
available). The cutoff is per **declaration**, not per bug: a requirement the fix added was never in the
buggy tree, so the buggy date would refuse the very release the fix's own environment had — and,
read the other way, a requirement the buggy tree already declared is stamped with the buggy date and so
cannot be upgraded past it. Two exceptions:

- A package the harness already provides (`pytest`, `pytest-json-report`) is left at the harness
  version and recorded as such **when the bug only bounds it** — a range names no release for the layer
  to install, so `pytest>=4.6.5` (spacy/4) leaves the interpreter layer's pinned pair standing. An exact
  `==` pin is the next exception instead, and it wins.
- An exact `==` pin **written in BugsInPy's own two per-bug files** (`requirements.txt`, `setup.sh`) is
  the benchmark's **reference environment**: it is installed as written and exempt from the date rule,
  because that pin is what `run_test.sh` was validated against. Whatever its release date, it is
  recorded in the provenance's `pins_after_commit_date` list (with the release date and the rule that
  chose it), so "this pin is newer than the bug" is a recorded fact rather than an invisible one. A
  bug whose `requirements.txt` is a corpus-era **pip freeze** (black/22's is: 71 pins, 65 of them
  released after its 2018-03-29 commit) therefore resolves instead of refusing — see the split's
  `reviewed_changes` for what then has to be checked about the sandbox's ability to build it. This
  exception applies to `pytest` like any other name: an exact `==` pin on `pytest` in those two files
  wins over the harness's version, for the same reason and with the same record (see the next
  paragraph). A `pytest` pin from the *project's* own files (`tests_require`, `install_requires`) gets
  no such licence: it is not the environment `run_test.sh` was validated in, so it cannot decide the
  harness's own version — a project pin that agrees with the harness version is simply that version,
  and one that does not is a decision the harness refuses to make rather than a silent downgrade of the
  plugin's host.

**Two pins of one layer whose own declarations contradict each other: two cases, and only one of them is
tolerated.** The date rule above is per *declaration*, and two declarations can land on releases that
cannot live together — `sanic/1` is the measured case, and it is why the distinction exists. Its
`requirements.txt` is a pip freeze from 2020 carrying `h11==0.9.0` (line 40) and `httpcore==0.3.0`
(line 44), and `httpcore==0.3.0`'s own metadata declares `h11==0.8.*`: two of BugsInPy's own exact pins,
jointly unsatisfiable, in one layer. The environment they were frozen from installed both anyway — pip
before 20.3 had no resolver that could refuse, it installed each requirement in turn and printed "pip's
dependency resolver does not currently take into account all the packages that are installed" — while
today's pip answers `ResolutionImpossible` instead, so the layer never gets built and the benchmark loses
the environment rather than reproducing it. Which case a conflict is decides what happens, and
`bake.PinConflict.case` carries that decision:

- **(b) both sides are BugsInPy's own exact pins** (`bake.CASE_BOTH_OWN_PINS`) — the benchmark's own
  reference environment, and **tolerated**: recorded, printed under its own `TOLERATED` heading, and not
  a failure. It is *reproduced* rather than refused, by installing with a pip that still has the old
  behaviour (below).
- **(a) anything else** (`bake.CASE_HARNESS_CHOICE`) — at least one side is a version *this harness*
  chose: the date rule, or pip's own pick of a transitive dependency. A layer holding both is a layer
  whose environment is not the benchmark's, so this one **fails the layer**, and the requirement is that
  the date rule be *narrowed* until the conflict does not arise rather than that it be tolerated — in
  either direction, "a declaration in this layer on the name being chosen" or "the chosen release's own
  declarations on a version this layer cannot move" (both below).
- **(?) neither** (`bake.CASE_UNKNOWN`) — not a conflict at all, but the absence of the reading: no
  document could be reached for one of the pins. Recorded so that "checked and clean" and "could not be
  checked" stay different answers. See the preflight's fallback chain below.

**The narrowing is recorded wherever it changed the answer.** When an exact pin in the same layer
declares a requirement on a name the date rule is choosing freely, the answer becomes the newest release
that satisfies the declaration **and** the cutoff, and `ResolvedDep.narrowed_by` carries each
`(pin, specifier)` pair that did it — a departure from "newest before the date", recorded the same way an
`overridden` claim is, so a reader can tell a narrowed choice from an ordinary one and from what it was
narrowed by. Only an **exact** pin can narrow this way, and the reason is not stylistic: only an exact
pin's own version is settled by the requirement itself, so only its `requires_dist` can be read *before*
the date rule runs. A pin whose declarations cannot be read narrows nothing and is passed over rather
than guessed at — what it asks for is unknown, and choosing a version for an unread requirement is a
choice nothing can check. `httpcore==0.3.0` is exactly this case (its declarations are in neither PyPI's
JSON, a wheel nor a sidecar — measured; the requirement lives in its **sdist**, which pip reads at
resolution), so its `h11==0.8.*` narrows nothing before the build and is caught by the layer's own
`pip check` after it.

**The same case, read the other way round — and it is what `luigi/1` measured.** The rule above fixes
the version of a name *something in the layer declares a requirement on*; the mirror of it is a release
whose **own** declarations refuse a version the layer cannot move — BugsInPy's own `==` pins and the
interpreter layer's pair (`bake._settled_versions`: `bake.reference_pin`'s exact pins over
`bake.HARNESS_PACKAGES`, with the bug's own pin winning over the pair, exactly as it does above). `moto`
is declared in `luigi/1` as a bare name, so nothing narrows *its* version and the date rule takes
`1.3.8` (2019-04-23, the newest on or before the 2019-05-15 commit) — but `1.3.8` declares
`idna (<2.9,>=2.5)` while the same layer installs BugsInPy's own `idna==2.9` as written, so the layer
that gets built is a contradiction its own `pip check` would report as case (a) and the gate would refuse.
Neither version can move, so the **choice** does: the answer is the newest release on or before the date
that the layer's own pins can hold. `ResolvedDep.refused` carries `(rejected, held, declared)`, so the
record names both the release that was passed over and the pin that refused it.
`ResolvedDep.pin_rule` renders both directions in one sentence, and `bake.preflight_lines` prints every
moved pin under its own `NARROWED` heading **before** the build — `bake.pin_narrowings` reads it off the
provenance's own `narrowed_by`/`refused`, so the record and the printout cannot disagree. That block is
not a finding and counts in no verdict: a pin that was moved is a decision already taken, and it is
printed because a pin which is *not* simply "the newest release on or before the date" is the one thing
the pin list cannot show (`botocore==1.10.84` beside `urllib3==1.25.9` looks arbitrary until the reason is
next to it). Two boundaries on the walk, both inherited from the direction above: a candidate whose
declarations cannot be read is **taken**, not refused — an unreadable release is not evidence of a
conflict, and the layer's own `pip check` is the backstop — and a name for which *every* dated candidate is
refused is a `BakeError`, because that is a decision for the split rather than something to install and
discover. Conflicts between two of BugsInPy's own exact pins are untouched by any of this and stay case
(b).

**A refusal can name *another pick of this same pass*, so the pass is a fixpoint and not one walk — and the
clause it is read from is read only when its marker holds here.** Both halves are `luigi/1`, and the second
is what the three declaration readers below now do. Every `botocore` release that bounds `urllib3` at all
carries the bound under a `python_version` marker: `1.12.149` (2019-03-20) declares
`urllib3>=1.20,<1.25; python_version >= "3.4"`, which holds in every layer this harness bakes, against
BugsInPy's own `urllib3==1.25.9` (line 144 of this bug's `requirements.txt`). That pin **cannot move** — a
reference pin is installed as written, and it wins over everything the date rule would pick — so
`urllib3==1.24.3`, which *does* satisfy the bound, is not installable here either: the resolver gives, not
the benchmark's pin. Read as the install pip will make, `1.12.149` is refused and `botocore` falls to
**`1.10.84`** (2018-08-23), the newest release that declares no `urllib3` bound at all. That is not the end,
because the refusal it produces names this layer's *own* pick rather than a BugsInPy pin: `boto3`'s newest
dated release, `1.9.149`, declares `botocore>=1.12.149,<1.13.0`, which refuses the `botocore==1.10.84` the
same pass just chose, so `boto3` falls to **`1.7.84`** (2018-08-23, declaring `botocore>=1.10.84,<1.11.0`);
and `moto`'s `1.3.7` declares `botocore>=1.12.13`, so `moto` falls to **`1.3.6`** (2018-09-22), whose own
`boto3 (<1.8,>=1.6.16)` and `botocore (<1.11,>=1.9.16)` the answers above satisfy. Three picks, one answer,
and it is not the answer of any single pass: the claims name each other's picks, so whichever ref is read
first gets the wrong last word. `bake.resolve_pins` is therefore the **fixpoint** of `bake._pin_each_ref` —
the pass is repeated with its own non-harness picks folded into the `settled` set, `settled` *replaced*
each round rather than accumulated, so a name whose pick moved is held at the release it moved to and not at
both. Only the **last** round's records are returned, because its `refused`/`narrowed_by` describe the
version the layer actually installs and an intermediate round's would name a rejection the final answer no
longer stands on. This is the same shape `bake.transitive_pins`' re-choose queue already has, and the answer
does not depend on the order the refs were read in. A round whose picks the pass has returned before is a
**cycle** and raises a `BakeError` naming the names that move between rounds, rather than a choice for the
date rule to make: §3.2's case (a) has no reading for a layer whose own pins cannot agree, and installing
the contradiction is exactly what that layer's `pip check` would then report. **The refusal record is
bounded at one entry per pin held against** (`refused.setdefault(held, …)`, not `append`), and the
newest-first walk makes the kept entry the *newest* release that pin refuses: without that bound a single
pass over `luigi/1` produces 160 distinct refusals for `boto3` alone and `ResolvedDep.pin_rule` becomes a
160-clause sentence, where the same fact about the layer — "this pin cannot hold that release" — would be
repeated once per release. The bound keeps `pin_rule`'s one-sentence rendering, `bake.pin_narrowings` and
`bake._narrowing_section` readable, and `ResolvedDep.as_dict`'s shape unchanged.

**A conflict is tolerated only when the layer's own `pip check` says so.** The preflight *predicts* from
the pins' declarations; `pip check` is what reads what actually got installed, and the layer's reading is
the gate. The recipe runs `python -m pip check > /opt/vanta-pip-check.txt 2>&1 || true` after the last
install and before the freeze, and echoes it into the build log on marked lines (a checkpoint is a
filesystem, so nothing on the host can read that file back without booting the layer). The host judges
those lines by the same rule (`bake.pip_check_conflicts`) against the same predicate: `own` is
`bake.own_pin_names`, read off each pin's own `pin_rule` and not re-derived from the declarations, so
"BugsInPy's own pin" means one thing in the record and in the layer and a conflict cannot be tolerated in
one and refused in the other. A **case (b)** conflict is logged and tolerated; a **case (a)** one becomes
the layer's failure through `bake.pip_check_failure`, reported through the same path a build failure
takes so the conflict names itself where the reader is already looking. `pip check`'s own exit status is
**not** the verdict — it exits non-zero for a conflict the benchmark deliberately reproduces — which is
why the reading is taken (`|| true`) and judged on the host. Every conflict, tolerated or not, is handed
to `bake.record_pip_check` and written back into the layer's `runs/bugsinpy/deps/<checkpoint>.json` under
`pip_check`, so that file carries what pip actually said and not only what PyPI's metadata predicted; the
key is in `bake.deps_provenance` from the start, which keeps "nothing to report" (`[]`) and "this file
predates the reading" (absent) distinguishable.

**A pin's own `requires_dist` is pinned too, by the same date rule — or the layer is not the one its
pin list describes.** Measured, and it is what `fastapi/1` failed on: `exceptiongroup==1.3.1` (2025) in a
layer whose own pins hold `typing-extensions==3.7.4.2`, reported by the layer's `pip check` as a
contradiction **no pin in it had declared**. Everything above pins what a *file* declared; pip also
resolves what a *release* declares, at the newest release the day allows, and the date rule never saw
that answer. So the preflight walks each pin's `requires_dist` (the same sidecar/wheel metadata chain as
everywhere else — an sdist is still never executed) to the full transitive closure, with
`bake.marker_applies` evaluating each marker against the **layer's** platform and never the host's
(`packaging.markers.default_environment` answers for the Mac running the bake and would silently drop
every `sys_platform`-conditional requirement). A name the layer already pins is *satisfied* and stops the
walk; a name nothing pins is pinned by the date rule and the walk continues into *that* release's
declarations. Each such pin carries `pin_rule: "transitive, date rule"` (`bake.TRANSITIVE_PIN_RULE`) and
`ResolvedDep.required_by` — the pins whose releases declared it — so "this name is here because
`pytest==5.4.3` requires it" is in the record rather than inferred. A name for which no release satisfies
the claims raises a `BakeError`: falling back to a newer release is exactly what pip does on its own, and
that fallback is the failure this rule exists to refuse. Two boundaries: a release whose declarations
cannot be read describes nothing and is not evidence it requires nothing (the layer's own `pip check` is
the backstop), and the layer's own settled pins still get the last word (`bake._chosen_release`), so a
transitive pick cannot install a release one of them refuses.

**The date rule picks a *release*, and a pre-release is not one.** `pip` skips pre-releases and dev
releases unless the requirement itself names one — the one rule of pip's the date rule carries over,
because everything else here is deliberately *not* pip's. PyPI's listing is full of them and they sort
above the release that was current, so a pool built as "everything on or before the date" gets one:
measured on `matplotlib/1`, `pyparsing>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6` (`install_requires (setup.py)
line 283`, the 2020-06-09 commit) took `3.0.0a1` — uploaded 2020-04-05, the same day as the `2.4.7` pip
would have installed, and above it — and the picks that release's own declarations produced followed it
in: `sphinx==3.1.0` declares `Jinja2>=2.3` and the layer took `jinja2==3.0.0a1`, whose metadata then
declared the `markupsafe==2.0.0a1`. Three pins in one layer, each a version the requirement never asked
for. `bake._names_a_pre_release` is the carve-out, read off the requirement's own clauses (a `>=3.0.0a1`
bound names one; a wildcard `==1.0.*` does not, and that direction only ever *denies*), and
`bake._date_rule_candidate` is the rule. The carve-out is not decoration: `black==19.10b0` is BugsInPy's
own pin in `fastapi/1`, `luigi/1` and `sanic/1`, it *is* a pre-release, and it is installed as written
(`bake.reference_pin`) — a rule that dropped every pre-release would make three of the eleven bugs
unbakeable.

The rule is applied where a version is *picked*, and that is three call sites, all of them the date
rule: `resolve_pins`' own pool — which is the list `_chosen_release` chooses from *and* the list the
narrowing filter below filters, so a narrowing claim can only ever take candidates away and can never
re-admit a pre-release — `bake._transitive_pin`'s pool, and `bake.resolve_build_pins`' pool. Narrowing
is not a fourth site but it is a fourth *reader*: a narrowing claim is a requirement on the name like
any other, so its clauses are read by the same carve-out as the declaration's, because
`sphinx==3.1.0` declaring `pyparsing>=3.0.0a1` is what makes a pre-release installable at all and
reading the carve-out off the declaration alone would refuse a layer pip builds. Measured across the
six layers, **one bug changes**: `matplotlib/1` moves `pyparsing` 3.0.0a1 → **2.4.7**, `jinja2` 3.0.0a1
→ **2.11.2**, `markupsafe` 2.0.0a1 → **1.1.1** (and that last one's `required_by` follows,
`jinja2==3.0.0a1` → `jinja2==2.11.2`). Every other `pyparsing`/`jinja2`/`markupsafe` in the run is a
`requirements.txt` reference pin already at 2.4.7/2.11.2/1.1.1, and no layer's `build_pins` or
`early_pins` holds a pre-release — none of `setuptools`, `wheel`, `cython` has one anywhere near these
commits — so no build pin and no early install moves.

**The project under test is never a transitive pin, and neither is anything else the tree ships.**
`pandas/1`'s closure walk pinned `pandas==1.0.3` because `statsmodels==0.11.1` and `xarray==0.15.1` each
declare `pandas` — a released pandas installed **over the very tree the bug is graded on**, so every
patch would have been graded against the wrong source while the recipe looked correct. The exemption is
the one the import closure already makes (`bake.tree_provided_modules`): the bug's own distribution name
and every top-level module the tree ships are names the tree provides, whoever asks for them.
`bake.transitive_pins` takes those names as `tree_satisfied` and holds them in **both** its walks — the
interpreter layer's provision walk neither descends through one nor reports it as a name the base layer
hands the layer (there is a third place it comes from), and the layer's own closure walk never pins one.
Neither is silent: each is a `bake.TreeSatisfied` record naming the releases that asked for it and how
they asked, carried in the deps JSON under `tree_satisfied` beside `dropped` and `orphans` (always
present, empty when the tree answered nothing), because "the requirement was satisfied" has to be a
statement a reader can check rather than a name that quietly went missing. The names are read once, in
`bake.bug_sources`, and carried on `DepSources.tree_modules`, so the test side's import closure and this
walk cannot disagree about what the tree ships.

**Leaving the name out of the pin list is not enough on its own, and the first bake of `pandas/1` is the
measurement that says so.** The exemption removes `pandas` from `installs`; it does not remove the
requirement `statsmodels==0.11.1`, `xarray==0.15.1` and `fastparquet==0.4.0` each wrote for it, and a name
that is in no pin list is in no `-c` constraints file either (`bake.constraint_lines` writes the layer's
own pins and nothing else), so pip resolved it itself. The layer took `pandas 2.0.3` — a 2023 wheel whose
own `requires_dist` refuses the layer's `numpy==1.18.4` and `python-dateutil==2.8.1`, which is exactly what
the layer's own `pip check` then reported (`pandas 2.0.3 has requirement numpy>=1.20.3; python_version <
"3.10", but you have numpy 1.18.4`) — beside a `tzdata 2026.5` that is no pin's dependency either, so the
exemption did not hold end to end: the released pandas the walk refused to pin was installed anyway, by
pip, one step later. The remedy is the pass §3.2 already has for the *other* rule that leaves a name out
of `installs`: the pins that declare an absent name are installed `--no-deps` in their own block, after
the main pass and before the orphan removal (`bake.no_deps_dependents` — one function for both rules,
because it is one behaviour over one list of names, and the caller passes the union of the dropped names
and the tree's). Nothing is lost by that: everything those pins require is already installed, since the
closure walk pinned it, except the absent name, which is precisely what may not be installed.

The absence that leaves is a case of its own rather than a dropped dependency, because a reader acts
differently on the two — a dropped name is one this image could not build, a tree-provided one is one the
layer must not install: `bake.CASE_TREE_PROVIDED` (`"t"`), whose tolerated line reads
`case (t) <pin> declares <name>: the tree under test provides it, by §3.2's audit — tolerated`. It is read
**by name** off the provenance's `tree_satisfied` and reaches the gate through
`bake.run_bake_actions(tree_provided=…)` from `scripts/bugsinpy_run.py`, beside the `dropped` map that has
been there since the drop rule; a missing requirement that is neither is still case (a) and still fails
the layer. Nothing is whitelisted: the declaring pins are moved out of the resolving pass, pip is stopped
from resolving the name at all, and the absence that remains is attributed to the rule that caused it. The
freeze gate needed no parameter of its own for this: a tree-provided name is in neither the layer's pins
nor its provision, so a leaked release of it fails there on the rule that already fails an orphan —
`pandas/1`'s own line was `VANTA-PIP-FREEZE pandas==2.0.3`, and the gate that refused the layer first was
the own-pin gate above it.

Measured across the six layers, **exactly one layer moves**: `pandas/1`, whose `pandas==1.0.3` pick became
no pin, no install and one `--no-deps` block; `fastapi/1`, `luigi/1`, `matplotlib/1`, `sanic/1` and
`PySnooper/1` have neither the project's own name nor a tree-shipped module spelling in their transitive
lists, so their `tree_satisfied` is `[]` and their recipes are unchanged.

**A requirement whose marker is false for *this* layer is not a requirement, and every reader that
sees one says so.** A marker is a conditional, and reading it as decoration installs what pip would
not: measured, the preflight printed `EARLY scikit-learn==0.23.1's build needs
numpy==1.13.3; python_version=='3.6' and platform_system!='AIX' and platform_python_implementation ==
'CPython' → numpy==1.18.4` for `pandas/1` — a `py3.6`-only clause in a `py3.8` layer, read as if it
applied. `bake.marker_applies(requirement, *, python_minor, extra="")` is the single answer, and it
evaluates `packaging.markers` against the **layer's** environment and never the host's, which is the
same distinction the closure walk already draws for the same reason:
`packaging.markers.default_environment()` answers for the Mac running the bake — `sys_platform` is
`darwin` there — and would drop every `sys_platform`-conditional requirement the layer really installs.
Every variable this layer has is supplied (`python_version` `3.8`, `python_full_version` `3.8.0`,
`implementation_name` `cpython`, `os_name` `posix`, `sys_platform` `linux`, `platform_machine`
`aarch64`, `platform_python_implementation` `CPython`, `platform_system` `Linux`), so what the layer
does not carry is a variable *no* layer carries: `dependency_groups`, which PEP 735 added and a bug
layer is not a project with groups, parses and fails to evaluate, and that is read as *not applying*
— pip cannot evaluate it either, and the freeze gate (:func:`unpinned_freeze`) is what names the
distribution if pip installed it anyway. A **misspelt** variable is not that branch, and the difference
is a measurement rather than a hope: on `packaging 26.3`, `Requirement('foo; python_versoin == "3.8"')`
raises `InvalidRequirement` at *parse*, so `bake._requirement` raises and the ref is kept as
unparseable instead of being read as a conditional that does not hold.

**Three readers of a *release's* declarations skipped the marker instead, and that is what `luigi/1`
measured.** `bake._chosen_release` (does one of the layer's settled pins refuse this candidate?),
`bake._narrowing_requirements` (does a pin in this layer narrow a name the date rule is choosing?) and
`bake.metadata_conflicts` (does this release contradict another pin of the layer?) each read a release's
`requires_dist`, and each used to `continue` on `requirement.marker is not None` — on the reasoning that a
conditional requirement is not something to move a settled pin's, or to refuse a layer's, on, the
conservative direction being to leave it unread and let the layer's own `pip check` report it. The question
those three are asking is what pip will put in the environment, and there a marker *is* the question, because
pip skips what does not apply: a clause is read as an install only when it holds here. Measured, the
omission cost `luigi/1` its whole layer — every `botocore` release that bounds `urllib3` carries the bound
under a `python_version` marker (`1.12.149`'s is `python_version >= "3.4"`, true in every layer this harness
bakes), so all three readers were blind to the `urllib3<1.25` against BugsInPy's own `urllib3==1.25.9`, and
the bake reached `pip check` with the contradiction the preflight had not predicted. All three now call
`marker_applies` with the layer's minor and read only the clauses that hold. The mirror check that this is
narrow: `pandas/1`'s early pass once named `scikit-learn==0.23.1`'s build dependency
`numpy==1.13.3; python_version == "3.6"`, which is false for every layer this harness bakes — a reader that
read it as an install would pin a name the build never asks for.

The **collected refs** are filtered once as well, and that is a third place rather than the same one: it
happens before any pin is chosen, over the refs themselves instead of over a release's metadata, and nothing
had to be threaded through the parsers for it because `bake.DepRef.spec` is the requirement verbatim and
still carries its marker. `bake.marker_skips(refs, *, python_minor)` returns the refs to keep and a `bake.SkippedLine` for each one
dropped, and it is called where the refs come together: twice in `bake.bug_sources` (the
deps refs and the build refs, covering `install_requires`, `requires_dist`, `setup_requires` and
`[build-system].requires` alike) and once per pinned release inside `bake.early_install_pins`. `extra`
is empty by default — `requires_dist`'s `; extra == "test"` entries and the whole `extras_require`
surface are not installed — and set from the requirement's **own** marker clause
(`bake._EXTRA_CLAUSE_RE`), so a declaration read out of `extras_require["tests"]` that says
`extra == "tests"` is judged on the rest of its marker rather than dropped for saying it. The record is
a `SkippedLine(source, line, text, reason)`, its `reason` naming the marker that dropped it, and
`bake._SOURCE_LINE_RE` recovers the file and line number from the reader-written source
(`requirements.txt line 3` → `requirements.txt`, `3`; a source that names no line is recorded as line
`0`). Nothing about it is printed: the drops fold into `DepSources.skipped` and land in the layer's
`runs/bugsinpy/deps/<checkpoint>.json` under **`skipped`** through `bake.deps_provenance`, and into
`bake.PreflightBug.skipped` — the key `skipped` of the preflight JSON, beside `early_pins` — because a
`--preflight-only` run *is* the measurement and has to leave it somewhere the reader can check rather
than on the terminal, which is for the findings a reader has to decide on.

Measured across the six layers, this pre-pass costs **one bug's `skipped` list and nothing else**. `pandas/1`'s two
sdist-only pins whose build wants `numpy` — `scipy==1.4.1` and `scikit-learn==0.23.1` — each write a
family of `numpy` clauses into their `build-system.requires`, selected by `python_version` and
`platform_system` (and, for `scikit-learn`, also by `platform_python_implementation`), and exactly one
clause of the family holds in this layer: `numpy==1.17.3; python_version>='3.8' and
platform_system!='AIX'`. The other **thirteen** are dropped and recorded; the two `EARLY` lines that
used to name a false clause now name the one that holds — `scipy==1.4.1` used to print its
`python_version=='3.5'` clause and now prints the `>= '3.8'` one, and `scikit-learn==0.23.1` its
`'3.6' and platform_python_implementation == 'CPython'` one, likewise. The clause **count** does not
move — `pandas/1` still installs `numpy==1.18.4` early and still counts `18 early` — and the other five
layers record an empty `skipped` list, so this change alone leaves the verdict `ready to bake` with no bug
flagged. (The `skipped` list is what this pre-pass moves; the *pins* the three release-readers move are
`luigi/1`'s, measured above.) On disk the reading is one key:
`runs/bugsinpy/deps/vanta-bi-pandas-1.preflight.json` carries `"skipped": [ … thirteen entries … ]`,
each naming the clause verbatim and the marker that dropped it — and nothing about them reaches the
terminal, which is the point of recording a measurement a reader can check rather than a line they have
to notice. The two checks that the rule is narrow rather than sweeping: `luigi/1`'s and `sanic/1`'s
`cffi>=1.8,!=1.11.3; platform_python_implementation != 'PyPy'` is **true** for CPython and is kept (the
measured difference between judging the layer's platform and the host's — `platform_python_implementation`
answers `CPython` on both, but a `sys_platform` clause is the case where the two disagree and the host
would win if its environment were read), and the dev split holds exactly **two** marker-bearing tree
declarations in total — both `pandas/1`'s `numpy==1.15.4; python_version>='3.7' and
platform_system!='AIX'`, kept — with none at all among the releases the layers' own pins name. The
`EARLY` line above is also the narrower reading of that block's promise: the preflight now names what
the build will *want*, not the first clause the reader happened to write.

**Every install in a layer reads the layer's one pin list, so no step can resolve a version no pin of
it names.** Measured on `fastapi/1`, and it is the freeze gate that caught it. The layer's pin list holds
`requests==2.23.0` — the date rule's answer for the release the tree's `requirements.txt` names — but the
layer's **first** install, the dated build tool `flit==2.3.0` that `pyproject.toml`'s
`[build-system].requires` names, ran with no `-c`: pip resolved flit's own `requires_dist` against the
PyPI of the day, and the log shows plain, unversioned `Collecting` lines going in
(`runs/bugsinpy/bake-logs/vanta-bi-fastapi-1.log` 46–63: `requests-2.32.4`, `charset_normalizer-3.5.2`,
`idna-3.15`, `urllib3-2.2.3`, `certifi-2026.10.23`, `docutils-0.20.1`, `flit_core-4.1.0`). The dependency
install that followed **downgraded** the names the layer's pins carry — requests to 2.23.0, idna to 2.9,
urllib3 to 1.25.9, certifi to 2020.4.5.2, docutils to 0.16, flit-core to 2.3.0 — because pip does not
uninstall what it replaces, and `charset-normalizer==3.5.2`, a dependency of requests 2.32.4 that no pin
of this layer names, was left in the freeze (`VANTA-PIP-FREEZE charset-normalizer==3.5.2`, line 440); the
bake failed with `ERROR: this layer installed distributions its pin list does not name` naming
`charset-normalizer==3.5.2` (lines 608–609). The fix is one file and four flags: `bake.constraint_lines`
writes the layer's whole pin list — build pins, early pins and the resolved dependency pins, each pin once
and each as an exact `==` (`bake.pinned_parts`) — to `/tmp/constraints.txt` (`bake.CONSTRAINTS_FILE`,
heredoc `bake.CONSTRAINTS_MARKER`) above the first install, and all **four** installs in
`bake.requirements_recipe` carry `-c /tmp/constraints.txt` — the toolchain, the declared build
requirements, the early pass and the dependency list — so no step in a layer can resolve a version the
layer has not already decided. A recipe with no pins writes no file and passes no `-c`
(`bake.constraint_lines(()) == []`), so a layer with nothing to install keeps the recipe it had. The
freeze gate is unchanged: it is what *read* this, and it still fails a layer whose freeze names a
distribution no pin of it names.

The file can only be complete because the **build pins and early pins are pins of the walk like any
other**, which is the other half of the same fix. `bake.resolve_bug_deps` seeds the closure walk with
`own_pins = pinned_parts(*build_specs, *installs_of(resolved))` — the build pins first — and
`bake.transitive_pins` starts at every one of them, so a build pin's own `requires_dist` is pinned by the
same date rule as anything else. `fastapi/1` is the measurement: `flit==2.3.0` is a build pin, and its
declarations put `flit-core==2.3.0` and `chardet==3.0.4` in the layer's `installs`
(`runs/bugsinpy/deps/vanta-bi-fastapi-1.json`, and `chardet` is requests 2.23.0's own detector — the
release the pin list holds). That is why the constraint file is an answer rather than a gesture: the
toolchain step installs `flit==2.3.0` against `requests==2.23.0`, whose metadata requires `chardet` and
not `charset-normalizer`, so the leftover the gate refused cannot be created. The early pins need no
separate handling for the same reason — they are a subset of the layer's pins by construction
(`bake.early_install_pins` returns `satisfied_by` values taken from `pinned_parts`), so their releases'
declarations are walked at the same time as the pins they belong to, and they are passed to
`bake.constraint_lines` explicitly so that the file says what it holds rather than depending on the
subset.

Measured across the six layers, this changes **one bug's toolchain step and no dependency install**:
`fastapi/1` is the only layer with a build pin whose `requires_dist` was unpinned at install time
(`flit==2.3.0`; `matplotlib/1`, `pandas/1`, `luigi/1`, `sanic/1` and `PySnooper/3` install either the
toolchain alone or build pins whose requirements are already in their pin lists). Every other layer's
installs are byte-identical commands with `-c /tmp/constraints.txt` added and the same pins behind it, so
the preflight's per-layer counts — pins, early, dropped, skipped — do not move and the verdict stays
`ready to bake`. The property is tested on the recipe rather than on a bake
(`scripts/test_bugsinpy.py::TestBake::test_one_constraints_file_holds_the_whole_layer_and_every_install_carries_it`),
including the negative: no pins, no file, no `-c`.

**The window is the requiring pin's own release, not the bug's date.** `bake.pin_cutoff` is the later of
a pin's declaration date and the date its release landed. For a date-rule pin those agree. For a BugsInPy
reference pin they do not, and the difference is load-bearing: `pytest==5.4.3` (2020-04-09) installed as
written under a 2019-04-22 commit declares `pluggy (<1.0,>=0.12)`, whose oldest release is 2019-05-27 —
held to the commit date the walk has nothing to pick and refuses the layer, i.e. it would flag the bug
for a dependency the reference environment itself satisfies. A pin's requirements cannot pre-date the
release that declares them, which is what the later date says. A transitive pick's cutoff is the newest
of its requirers' cutoffs, the rule `bake.resolve_build_pins` already uses for the same reason.

**What the layer inherits is the mirror of the closure, and it is the other half of the record.** The
interpreter layer installs `pytest` and `pytest-json-report` and everything those releases require, and
pip does not uninstall what it replaces: a layer re-pinning `pytest==5.4.3` over 8.0.2 leaves 8.0.2's own
dependencies (`exceptiongroup`) in the environment. The provision walk therefore descends the closure of
**every** harness pin *at the versions the interpreter layer has* — through names this layer re-pins, on
purpose — and reports as `provisioned` the names that remain after removing every name this layer decides
for itself. Each harness pin is a **root** of that walk rather than a node in it: a path that reaches one
from another root stops there, because `pytest-json-report` requires `pytest` and counting 8.0.2's
declarations as the *plugin's* would attribute `exceptiongroup` to a harness pin this layer keeps — the one
misreading that hides the orphan rule below. `bake.deps_provenance` writes the names out, and they are the
exemption the freeze gate reads below. What this layer's own pins reach is pinned **regardless** of the
inheritance, in the other direction: `pytest 5.4.3`'s `pluggy (<1.0,>=0.12)` is pinned by pytest's own
cutoff even though the interpreter layer holds a `pluggy` of its own, because a walk that stopped at a
provisioned name would install pytest 5.4.3 over a pluggy it refuses. The layer's pins win; the leftovers
are what is exempt. The set is a *prediction* of the interpreter layer's environment, used for its
**names** — the one thing it cannot see is a name that appears or disappears with a release.

**What the layer *replaces*, it removes: the interpreter layer's residue is not an exemption
(`bake.ORPHAN_RULE`).** Measured, and it is the rest of what `fastapi/1` failed on. The
`exceptiongroup==1.3.1` its `pip check` refused was in the layer because the interpreter layer's
`pytest 8.0.2` declares `exceptiongroup >=1.0.0rc8 ; python_version < "3.11"`, while this layer installs
`pytest==5.4.3`, a release that declares nothing of the kind — so the name is not something the interpreter
layer *provides this layer with*: it is leftover of a release this layer swapped out, and the layer may not
keep it. `bake.transitive_pins` therefore answers a third value: the inherited names **every** path of
which starts at a harness pin this layer replaced (`settled` says which those are — a harness pin whose
entry the layer's own pins overwrote). Each is a `bake.OrphanedPin` carrying `name`, the harness pin that
brought it in (`declared_by`), that release's own declaration of it (`declared`) and the pin this layer
installs over it (`replaced_by`): "why it was there" is the half a reader cannot recover from the layer. A
name a **kept** harness pin also reaches is that pin's own dependency and stays —
`pytest-json-report`'s `pytest-metadata` is inherited, not removed, and uninstalling it would break the
plugin the oracle is read from. The recipe uninstalls each orphan by name (`bake.orphan_removal_lines`)
**after** every install — an install can pull one back in — and **before** `pip check`, so the check reads
the environment the layer actually ends up with; each removal is echoed into the build log on
`VANTA-ORPHAN`-marked lines, because the freeze block below cannot say *why* a name the interpreter layer
installed is absent from it. The step carries no `|| true`: pip exits 0 for a name that is not installed,
so anything else fails the layer. No orphans is the ordinary case and the recipe is unchanged — a layer
whose bug does not re-pin a harness package removes nothing. Every removal is written into the deps JSON
under `orphans` with its reason, and the names are deliberately **not** in `provisioned`.

**A name this layer installs nothing for is not left to pip's judgement: the pin that declares it is
installed `--no-deps` (`bake.no_deps_dependents`).** Measured, and it is what `sanic/1` failed on. Its
freeze held `cryptography 47.0.0`. A drop removes a name from the **pin list**; it does not remove the
requirement another pin wrote for that name, so pip satisfies it at install time — and at the newest
release of the day, because `-c` cannot stop what it is not told about: the constraints file is built from
the pin list (`bake.constraint_pins`), a dropped name is in no pin list, and pip therefore never reads an
opinion about it. `pyopenssl==19.1.0` declares `cryptography (>=2.8)` and `service-identity==18.1.0`
declares it bare; the date rule's `cryptography==2.8` and its transitive pick `2.9.2` both have no wheel
this image can build and both are dropped; pip resolves the name for itself and installs `47.0.0`, a
release years newer than every pin in the layer. An exemption that *keeps* such a name is an
accommodation of the leak, so the leak is closed instead: the pins whose **own** release declares such a
name (`bake.no_deps_dependents`, read from that release's `requires_dist`, the same metadata every other
rule reads) are moved out of the main
pass and installed in their own pass with `--no-deps` (`requirements_recipe`'s `/tmp/no-deps.txt`, written
after the main install and before the orphan removal). The function takes one list of absent names because
the drop rule is not the only rule that leaves a name out of `installs`: the tree under test's own name
does too, by the paragraph above, and it is the same leak with the same remedy — the tree paragraph carries
that measurement (`pandas 2.0.3`) and the case it is tolerated under. Nothing is lost by that: everything
those pins require is already installed by the main pass — the closure walk pinned it — except the absent
name itself, which is precisely the one thing that may not be installed. The pass carries the same `-c`, the
same `--no-build-isolation` and the same legacy resolver as every other install; it is the same command
with one flag more, not a second mechanism. `pip check` then reads the absence this leaves
(`pyopenssl 19.1.0 requires cryptography, which is not installed`) and the own-pin gate tolerates exactly
that reading, **by name only and per rule**: the requirement the line names has to canonicalize to a name
this layer dropped (`bake.CASE_DROPPED_DEPENDENCY`) or one the tree under test provides
(`bake.CASE_TREE_PROVIDED`), so a missing requirement that is neither is still §3.2's case (a) and still
fails the layer (`bake._required_name`, and the `tree_provided` map `scripts/bugsinpy_run.py` builds from
the provenance's `tree_satisfied`). A reading with no held version is an *absence* rather than a conflict
between two pins, so case (b) cannot apply to it at all. The freeze gate below
fails hard if a name from **either** rule appears in it at any version, whatever the pin list says: the
pass not having taken is the only way for that to happen.

**The orphan rule is the whole rule, and a dropped name is not an input to it.** An earlier revision of
this section excluded a dropped name's own pip-resolved closure from the orphan set
(`transitive_pins(..., installable_dropped=…)`, `ce963f6`). That was the same accommodation read from the
other end — it made a name pip had resolved years late a permanent member of the layer rather than one pip
never gets to install — and it is reverted. `bake.transitive_pins` takes no such input, and
`bake.ORPHAN_RULE` reads as it always did: an orphan is a name the base layer brought that no pin of this
layer — direct, transitive, build or early — requires. With the `--no-deps` pass above, `sanic/1`'s
`typing-extensions` is no longer special: nothing in that layer's pin list requires it (its `pytest==5.2.1`
era requires none of it), so it is an orphan and it goes, and the `pip check` line it used to appear in is
now the dropped `cryptography` itself. The same revert is why `luigi/1` — which pins `pyopenssl==19.1.0`
and `moto==1.3.6` too, and drops the same `cryptography` — loses `cryptography` from its freeze without
losing the `typing-extensions==3.7.4.2` its own pin list does name.

**Every distribution the layer ends up holding has to be one its pin list names or the interpreter layer
supplies.** The closure walk above is a prediction; the layer's own `pip freeze` is the reading, and it
is the only place a name pip resolved for itself is written down, because
`--use-deprecated=legacy-resolver` installs whatever it is given and *succeeds*. The recipe writes
`pip freeze > /opt/vanta-deps-freeze.txt` after the last install and echoes it into the build log on
marked lines (the same reason `pip check` is echoed: a checkpoint is a filesystem, so nothing on the host
can read that file back without booting the layer). The host takes the **diff** — `bake.unpinned_freeze`,
comparing by canonical name and not by version, since a pinned name installed at some other version is
the different question the `easy_install` gate answers — exempting the pins of this layer and the
`provisioned` names above, and a name in neither is a layer whose environment its provenance does not
describe: it fails through `bake.freeze_failure`, reported through the same path a build failure takes,
naming the distribution. The exemption is applied with **no tolerance for orphans**: an orphan removed
above is in `provisioned` never and in the layer never, so the gate is a second reading of the removal —
a name the layer failed to uninstall is in neither set and fails here rather than passing. It runs after
the own-pin gate, because the two answer different questions — that one asks whether the pins can *hold*
together, this one whether the pin list *describes* the layer — and `fastapi/1` is exactly the shape where
the freeze gate would have passed the contradiction the `pip check` gate refused: the name is not in the
pin list, so before the orphan rule it was exempt as an inherited one, and now it is removed from the
layer and stated in the record instead.

**A dropped name is a hard failure here, at any version.** The diff above exempts what the pin list names
and what the interpreter layer supplies; a dropped name is in neither, so it would already fail — but only
while it is *absent* from the layer's own `dropped` record, and the judgements above depend on the exact
reading of that absence, so the gate names the case instead of inferring it: `bake.unpinned_freeze` takes
the layer's dropped names and reports a reading for one of them **whatever** the pin list or the provision
says (`dropped=…`), and `bugsinpy_run.py` reads those names out of each layer's provenance and passes them
to every gate that judges the layer. Measured, this is the half of `sanic/1` that would have caught the
leak on its own: `cryptography 47.0.0` in the freeze, with `cryptography` dropped and in no pin list, is a
layer whose `--no-deps` pass did not take. Any *other* name the freeze holds and the pin list does not
describe still fails by the paragraph above, so the two readings together are one gate.

**The bug's own `pytest` pin wins, and the report plugin follows it.** Measured, and the reason this
rule exists: all four dev-pilot bugs flagged on the same thing — fastapi/1 and luigi/1 pin
`pytest==5.4.3`, pandas/1 `5.4.2`, sanic/1 `5.2.1`, in BugsInPy's own `requirements.txt` — while the
harness installed `pytest==8.0.2` plus `pytest-json-report`. The bug's pin is the reference environment,
so it is what the layer carries; and then the plugin has to follow it, because `pytest-json-report` 0.3
and later declare `pytest>=3.3.2` (1.5.0 declares `pytest>=3.8.0`) and simply cannot live next to
pytest 5.2.1. Three things follow, and each is recorded rather than assumed:

1. **The pin lands on the bug's own layer, not on the interpreter layer.** Every bug's layer is
   `vanta-bi-<project>-<N>`, stacked on `vanta-bugsinpy-py<X.Y>`; the bug's `pytest==5.4.3` is one more
   install line in that layer's recipe. The interpreter layer stays one pair per Python minor
   (`bake.pins_for_python`, unchanged, still refusing two bugs that disagree about one interpreter), so
   **no split entry changes** — an entry's `pytest_version`/`json_report_version` continue to declare
   the *interpreter* layer's pair. What the bug's own layer actually carries is in its provenance, and
   `bake.layer_pin` reads it back: a run's row names the pytest the suite ran under, not the one the
   split's entry happens to match.
2. **The plugin is chosen from PyPI's per-release metadata, not guessed.** For a bug with its own
   `pytest` pin, `bake.plugin_pins` walks `pytest-json-report`'s releases newest-first and takes the
   first whose declared `Requires-Dist` on `pytest` accepts that version, skipping a release whose
   `Requires-Python` excludes the layer's interpreter — and pins `pytest-metadata` only when the chosen
   plugin declares it, at the newest release that both its own requirement and the interpreter's
   `Requires-Python` allow. Measured outcome: pytest 5.x on 3.8 → `pytest-json-report==1.5.0` +
   `pytest-metadata==2.0.4` (3.1.1 declares `pytest>=7.0.0`); tqdm/1's pytest 5.4.3 on 3.6 →
   `1.5.0` + `1.11.0` (2.0.4 is `>=3.7`); httpie/1's pytest 3.2.1 → `pytest-json-report==0.2` and no
   metadata, 0.3 being the first release with a pytest floor. `pytest-json-report` and
   `pytest-metadata` are therefore in the fetch set only for a bug that has its own pin; a bug with
   none needs neither. The choice is recorded next to the pin as `chosen_for_pytest` and its own
   `pin_rule` ("the newest release whose declared pytest requirement accepts pytest==<v>"), so a reader
   can tell the plugin's pin from a date pin. A bug with **no** `pytest` pin of its own keeps the
   interpreter layer's pair — `8.0.2`/`1.5.0` on 3.8 — and needs no plugin resolution at all.
3. **Every reader of the report handles both schemas, through one helper per language.** Which schema
   the plugin writes is a function of the version this rule picks, and 0.2 is exactly what httpie/1
   lands on: 0.1/0.2 write a **flat list**, 0.3 and later write `{"tests": [...]}`. Both carry `nodeid`,
   `outcome` and the `call`/`setup`/`teardown` blocks, so one entry reader knows both and every reader
   calls it — Python's `suite.report_tests` (feeding `parse_report`, `prepare.strip_and_baseline`'s
   collected-nodeid set, and `issue.failure_kinds_from_report`) and Rust's `harness.report_tests`
   (feeding the native-suite reader, the generated-test reader, and `bin/checkpoint_e.rs`). A reader
   that looked only under `["tests"]` did not fail loudly on that bug: it saw **zero** tests, so the
   native suite came back `ran: true` with an empty suite, every generated test graded `Error`
   (InfraFailure at evaluation time), the stripper hid nothing, and the issue text named no exception
   class. For the same reason a top level that is **neither** schema is an error (Python) and a failed
   run (Rust), never "no tests": silently empty is how a real report turns into a wrong answer. One
   limitation of 0.2 that no reader can repair: it builds its list from the terminal reporter's
   `passed`/`failed` stats, so a test that *errors* is absent from the file rather than recorded as
   `"error"` — an absent nodeid is excluded from the stable comparison, which is the right treatment of
   a test the report says nothing about.

A layer whose `pytest` changes has to be **rebuilt**, because the interpreter layer and everything
stacked on it were baked against the old pair; the checkpoints are the previous bake's, and
`pins_for_python` cannot notice that a *bug's* pin moved. PySnooper/3 is the one bug in both splits
with no `pytest` pin of its own, so it is unaffected: it keeps the interpreter layer's `8.0.2`/`1.5.0`,
its mirrored entry in the evaluation split stays valid, and nothing about its layers changes.

`pins_after_commit_date` therefore has **two** expected reasons, and `pin_rule` distinguishes them: a
BugsInPy reference pin installed as written, and a requirement only the fix declared, held to the fix's
own date. A pin *inside* the window carries the same fields — `released_at`, `pin_rule`, and the
`commit` its declaration came from — so that question can be asked of it too, instead of only of a late
pin.

Attribution is **per reference**, not per package (`DepRef.pin_source`): a pin the *project* wrote
still obeys the date rule even when BugsInPy declared the same package as a range, and vice versa. The
resolved set, each dep's source (file + line), the pin, the pin rule, the release date, and the commit
its declaration came from, the optional imports the closure found, and the
skipped `setup.sh` lines all go into `runs/bugsinpy/deps/<checkpoint>.json` and, as derived data, into
the split's `deps` key. That file's `installs` list is **exactly** what the layer's recipe passes to
pip (`bake.installs_of`, the one function both the recipe and the record use): a harness package the
interpreter layer already provides appears in `dependencies` with `provisioned_by: "interpreter layer"`
and is **not** in `installs`, because the bug's own layer never re-pins it — but a harness package the
bug's own files pin exactly **is** in `installs`, on top of the interpreter layer, because that is what
the recipe does. `installs` is the difference between the two, and a record that disagrees with the
recipe is worse than no record.

**What the layer deliberately does not install.** The project's own install line is dropped, for two
reasons:

1. **A baked copy would be graded instead of the tree.** Every suite runs `python -m pytest` with its
   cwd set to the tree root, so `import <package>` resolves to the mounted tree — the copy the run is
   actually being graded on. Installing the project into the layer's `site-packages` puts a frozen
   copy in front of nothing useful and risks a run that reports `hidden_pass` while the patch it was
   meant to test never took effect.
2. **An editable install cannot work.** `pip install -e .` bakes the *bake-time* path, and each run
   gets its own tree path; the layer would point at a directory that no longer exists.

**`needs_build` is read from the tree, not judged.** A build, when a tree needs one, happens **per
tree, at prepare time, inside Shuru** — the same place, and for the same reason, that the suites run.
Which trees those are is the split's `needs_build` flag, and the flag has to be derived from the tree
it is about rather than hand-set: the dev split shipped `needs_build: false` for matplotlib/1 and
pandas/1, whose tests import a package the tree itself has to compile (`pandas/tests/dtypes/test_dtypes.py`
reaches `pandas._libs`; matplotlib's tests reach `matplotlib._path`), so the first thing that would
have happened is the dependency gate's import probe failing with "does not import from the tree" —
which reads like a missing dependency and is not one, the exact confusion §3.3 exists to prevent, in
the other direction. The rule is `bake.needs_build(files, setup_py, pythonpath=…)` and it has two
independent triggers, each a property of the tree rather than a judgement about it:

1. **BugsInPy says the import lives in a build directory.** It sets `pythonpath` *inside* the tree
   (`tqdm/build/lib/`, `cookiecutter/build/lib/`), so the package its own `run_test.sh` imports only
   exists after the tree is built. This settles it whatever the tree ships.
2. **The tree compiles extensions of its own.** `setup.py` declares `ext_modules` **and** the tree
   carries sources to compile (`bake.COMPILED_SOURCE_SUFFIXES`: `.pyx`, `.c`, `.cpp`, `.cc`, `.cxx`).
   Both halves are load-bearing, and each excludes a measured shape: **black** declares `ext_modules`
   and has nothing to compile — its extension is the opt-in `mypyc` build of its own Python, off
   unless `BLACK_USE_MYPYC` is set, so its tests import pure Python; **tornado** ships
   `tornado/speedups.c` and declares no `ext_modules`, because since 5.0 that file is a stub kept for
   old compiled installs and is not built. Suffixes and never directories, because matplotlib's
   extension sources sit in the vendored `extern/` (63 files) and in `src/` (17) alike and both are
   compiled into the package; `.pxd` is deliberately absent, a Cython *declaration* not being
   compiled.

Read from the **buggy commit**, because that is the tree the suite runs against (`oracle_tree` is the
buggy source with the fixed commit's test files). The reason comes back with the answer in both
directions, so a boolean that disagrees with its tree can say which half of the rule it disagrees
with: `dev_split_author.py` derives the flag from the clone (and cross-checks the mirrored
PySnooper/3 entry against the evaluation split's value rather than trusting it), and
`prepare.check_build_declared` compares split and tree **both ways** before any tree is materialised,
flagging rather than raising (§3.2) so one mis-declared bug does not stop the rest of the batch. The
measured dev split is `matplotlib/1, matplotlib/2, pandas/1, spacy/4`; the measured evaluation split
is `cookiecutter/1, tqdm/1`.

**Build-time tools are pinned by date too.** A build-time tool is as much a part of "the environment
this tree was built in" as a dependency is, so the date rule above binds it as well — measured, not
hypothesised: on matplotlib/1 the pinned `numpy==1.18.5` has no wheel for the sandbox
(linux/aarch64, py3.8), so pip built it **from source**, in an *isolated* build environment, and that
environment supplies *today's* setuptools. Setuptools has not carried `distutils.msvccompiler` for
years, so numpy's own `numpy/distutils/mingw32ccompiler.py` died with `ModuleNotFoundError` — a
failure with nothing to do with matplotlib or the bug. Every wave therefore resolves and installs, per
layer and before anything else, a **build pin** set:

- **The toolchain, for every bug**: `setuptools`, `wheel`, `cython`, each pinned to the newest release
  on or before the **bug's** commit date. These are the tools pip's isolation would otherwise supply
  at their current versions, and one of them is the package that just failed.
- **Every name the project declares as a build requirement**, held to the declaring commit's date like
  any other reference: `setup_requires` in `setup.py` and `[build-system].requires` in
  `pyproject.toml` (`bake.build_refs`, `bake.resolve_build_pins`). This is what turns "and numpy when
  it is a build requirement" into a per-project answer — pandas declares Cython and numpy to compile
  `pandas/_libs`, matplotlib's `setup.py` says `setup_requires=["numpy>=1.15"]`, and a project that
  declares nothing gets exactly the three. These are **not** dependencies and never enter the
  dependency set; they are read at the same two commits and stamped the same way, because a build
  requirement only the fix declares is held to the fix's date for the same reason a dependency is.
- **A name the dependency set already pins follows that pin** (`dependency_pin` in the record):
  pandas needs one numpy to run and to build, and installing two versions of it in one layer would
  make the layer describe neither. A harness package (`pytest`, `pytest-json-report`) is never a
  build pin.
- A build tool with **no release on or before its cutoff** flags the bug rather than being guessed at:
  a tree cannot have been built with a tool that did not exist yet.
- **`pip` itself, pinned by capability and not by date** (`bake.PIP_PIN_VERSION`, emitted first of all):
  the layer's environment is installed by a pip that still carries
  `--use-deprecated=legacy-resolver`, and that is what makes §3.2's case (b) a reproduction of the
  benchmark's environment rather than a refusal of it. That is a property of the *tool* and not of the
  bug's date — `sanic/1`'s commit date is **2020-05-12**, *before* the flag existed, so the date rule
  would name a pip with no flag at all — so no release is fetched for it, `released_at` is empty, and
  `BuildPin.pin_rule` reports the capability rather than a date. The version is chosen by reading pip's
  own tagged sources: the option is in `pip/_internal/cli/cmdoptions.py` at 20.3, 20.3.4, 21.0, 21.3,
  **21.3.1**, 23.1.2, 23.3.2, 24.0, 25.3, 26.1 and 26.2.2, and absent at 19.3.1, 20.0.2, 20.1 and 20.2.4
  — the opt-out arrived with the resolver that made it necessary. **21.3.1** is the newest of them that
  installs on the split's oldest interpreter (22.0 moved pip's floor to 3.7, 24.1 to 3.8 and 25.0 to 3.9,
  and 3.6.9 is in the split), and the newest rather than the oldest for a reason: the pins are 2020-era,
  but the legacy resolver still installs *today's* newest release of everything the files left unpinned,
  and a newer pip reads more of what those wheels declare.

**Every install after the toolchain carries the flag, and the toolchain install is the one exception.**
`pip` goes into the **first** pip call — the toolchain one, `setuptools`/`wheel`/`cython` plus `pip`
itself — because that call is the first thing that runs and the pinned pip has to be in place before any
install that wants the flag. It is also the call that keeps pip's own isolation, and it is the install
that *puts this pip there*, so it runs under the interpreter layer's seeded pip and there is nothing yet
for the flag to change. Every call after it — the project's declared build requirements and the
dependency set — carries `--use-deprecated=legacy-resolver`. One rule rather than a flag per bug: a
layer's environment *was* built by a pip of this era, so every install in it that resolves this bug's
pins reproduces those semantics. The tolerance is not open-ended: the flag is only what lets the layer
install, and *which* conflicts it installed is read afterwards by `pip check`.

The order is what makes the pins mean anything, and `--no-build-isolation` is what makes the order
apply. The pins go into the layer in **two** pip calls, and the split between them is the fix for a
*second* measured failure of the first implementation. With the pins in one call, that call had to keep
pip's own isolation — the toolchain is the one install that cannot be made to depend on itself — and
that put matplotlib's `setup_requires=["numpy>=1.15"]`, a **build pin**, on the *isolated* side. The log
printed `build pins: setuptools==47.1.1 …` and pip built `numpy==1.18.5` in
`/tmp/pip-build-env-*/overlay` with *today's* setuptools: straight back into the failure the pins exist
to prevent, one line above the assertion that said the pin had landed. So:

1. **the toolchain alone**, in one pip call, keeping pip's own isolation — `setuptools`, `wheel`,
   `cython` are what the isolation would otherwise supply, so this is the one install that cannot be
   made to depend on itself;
2. **everything else, with `--no-build-isolation`**: the project's declared build requirements first
   (a name that is one of the three is claimed by call 1 whichever list it arrived in), then — as a
   separate call, before the dependency set — the dependency pins another pin's build needs
   (`bake.early_install_pins`, see the build-requirement block below), then the dependency set. The
   flag is pip's supported way to say "build against what is already in this environment"; with it, a
   missing build requirement fails the line instead of quietly isolating, so the flag either applies
   or the run stops.

Their versions are asserted once, with `importlib.metadata`, after **both** calls — a pin that silently
did not land is worse than no pin: the build would fall back to the interpreter layer's seeded
setuptools and look identical in the log. And every non-isolated call is followed by a check that pip
left **no** `/tmp/pip-build-env-*` directory behind: that path is the isolated build environment made
visible, pip removes it on a normal exit, so its presence means an isolated build ran anyway (failed or
aborted) and the layer must not be recorded as built. The check is a `for` loop over the glob, not
`compgen -G`, because the recipe runs under the image's `/bin/sh` (dash).

The pins go into the layer's provenance under its own **`build_pins`** key — next to `installs`, not
inside `dependencies` — because it answers a different question: a reader asking which Cython compiled
`pandas/_libs` is not asking what pandas imports. Each entry carries the name, the pin, what asked for
it (the toolchain, or the `setup_requires`/`[build-system].requires` line that declared it), the
`pin_rule`, and the release date. `bake` prints them per layer, and a layer with none records an empty
list rather than a missing key.

**A *dependency's* build requirement is read from that dependency's sdist, never from its metadata.**
`bake.build_refs` reads the declarations in the **bug's own tree**, which is the right answer for the
project's own `setup.py`/`pyproject.toml`. A dependency makes the same declaration in its own source
release, and none of the four places the pin metadata is read from carries it: measured on
`cryptography==2.9.2`, `info.requires_dist`, the PEP 658 sidecar, the wheel's `dist-info/METADATA` and the
sdist's own `install_requires` each answer with *runtime* requirements only, and its `cffi` build
requirement lives solely in the sdist's `pyproject.toml`. `bake.pypi_build_requirements` therefore opens
the release's sdist — the same
cached archive `bake.sdist_file_list` already reads, capped and streamed — and reads its **top-level**
`<name>-<version>/pyproject.toml` and `.../setup.py` (`bake._archive_texts` over
`SDIST_METADATA_MEMBERS`, tar then zip, and nothing the release vendored under a subdirectory), through
the *same* two readers the tree's declaration goes through: `[build-system].requires` through the TOML
table reader and `setup_requires` through the splat-aware AST reader. The archive is never executed. A
release pip installs from a wheel declares nothing this can read and is left alone, and a release whose
metadata cannot be read is not a failed layer — it is "nothing known", which the guard below still
covers. One reading per `(name, version)` per run (`bake.cached_build_requirements`), because the same
release is pinned in more than one bug's layer.

**A build requirement another pin satisfies is installed before the build wants it; a fetch that gets
past that fails the layer.** With `--no-build-isolation`, a `setup_requires`/`[build-system].requires`
nothing satisfies is resolved by setuptools itself, through `easy_install`, which reaches PyPI and
installs *today's* release: measured on `fastapi/1`, the layer pinned `cffi==1.14.0` and baked with
`cffi 2.1.1` beside it, and `Installing cffi-2.1.1` was the only line that gave it away. Two measures
answer it, one before the build and one after. **Before** — `bake.early_install_pins` takes each
sdist-only pin's read build requirements and, where **another pin in the same layer satisfies one**,
installs that pin in an **earlier pass**: the recipe's third non-isolated call, written to
`/tmp/early-req.txt` and run before `/tmp/req.txt`, so the build finds the pinned version already in the
environment. `EarlyInstall.satisfied_by` records which of the layer's pins it was, and a requirement no
pin satisfies is deliberately **not** invented *here*: this pass only **orders** a version the layer
already holds, and naming a version for a name nothing pins is a different question with its own answer
(`bake.uncovered_build_requirements` and `bake.build_requirement_pins`, the date rule below — item 2's
fix for `luigi/1`). **After** — *every* built
layer's output is read for the two lines only `easy_install` writes (`bake.easy_install_fetches`:
`Searching for ` and `Best match: `; the second names the release it settled on, and pip writes neither),
and a layer where one appears fails through `bake.easy_install_failure`, which names the fetched
package. That guard is kept apart from §3.2's own-pin gate because it answers a different question —
not a conflict between pins but a package that entered from **outside** the pin list, which no pin list
can be checked against — so it runs first and for **every** built layer, own pins or not, and a fetch is
not tolerated the way a case (b) conflict is: the layer that results is not the one its provenance
describes, and left alone it bakes green. The preflight computes the same `early_install_pins` for each
layer and prints the ordering decision (`EARLY <pin>'s build needs <requirement> → <pin>`, counted as
`N early` on the line), so what the bake will install early is on screen before the first byte is
installed; a pin the preflight dropped is skipped there, because the bake will not install it either,
and a requirement whose marker is false in this layer is not named at all (§3.2), so the line shows
what the build will want rather than the first clause the declaration happened to write.
Each finding is recorded in the provenance under **`early_pins`**, beside `build_pins`, with the pin,
the requirement, its source, and the pin that satisfies it. Measured on the two existing bake logs:
neither carries an `easy_install` line at all, and `bcrypt==3.1.7` shows no `Preparing metadata` step —
it installs from an `aarch64` `abi3` wheel and is never a source build — so the fetch this guards
against is a `cryptography==2.9.2`-shaped sdist-only pin and not every such pin.

**A build requirement the layer's pin does not satisfy moves the pin, and the move prints as
NARROWED.** The block above installs what a build wants *when a pin of the layer already satisfies
it*; the other case was left unanswered, and it is what `luigi/1` measured. The date rule pins the
toolchain's `cython==0.29.7` (2019-04-14, the newest on or before the 2019-05-15 commit), and the
same layer builds `numpy==1.18.4` — built here rather than installed from a wheel, because there is
no `aarch64` `cp38` wheel for it — whose own source reads the Cython it runs under and stops the
layer with `RuntimeError: Building NumPy requires Cython >= 0.29.14`
(`runs/bugsinpy/bake-logs/vanta-bi-luigi-1.log` 292; the pip-side reading is 315–318,
`raise RuntimeError("Running cythonize failed!")` then `ERROR: Command errored out with exit status
1: … prepare_metadata_for_build_wheel`). The pin was the reason, so the pin is what moves: **the
build requirement wins**, and `bake.narrow_build_pins` moves the tool to the **oldest release that
satisfies it** — oldest and not newest, because the date rule's purpose is to stay as near the bug's
era as the requirement allows, and `>=0.29.14` is a floor the build needs rather than a version it
wants: `cython==0.29.14` (2019-11-03), not `0.29.16`. **Every** claim on the name bounds
the move and not only the claims the held pin fails: the release the pin moves *to* has to satisfy
all of them, so the choice is made against `SpecifierSet(",".join(sorted(…)))` of them all — a
layer where numpy wants `>=0.29.14` and scipy wants `<0.29.16` holds `cython==0.29.14` and the
record names both claims. The move itself is made only when one of them does not hold as the pin
stands; a name every claim already satisfies is left exactly where the date rule put it.

`narrow_build_pins` reads the layer's **whole pin list** (`bake.pinned_parts` of the toolchain and
the installs), because the pin doing the building is any of them — numpy here is a dependency and not
a tool. Only a pin the date rule alone chose can move: a pin copying the dependency set's own answer
(`bake.BuildPin.dependency_pin`) and the capability pin (`bake.PIP_TOOL`) are passed over, or the
layer would install the toolchain's version and then replace it with the dependency's, compiling
against one release and importing another. The move is recorded on the pin itself,
`bake.BuildPin.narrowed_by`, in the same `(pin, specifier)` shape `bake.ResolvedDep.narrowed_by`
already uses for the dependency-side case above (`narrowed_by: [{"pin", "requires"}]`) — and that is
what puts it where it belongs: `bake.pin_narrowings` reads **both halves** of the provenance
(`build_pins` and `dependencies`), so a moved toolchain pin is a `NARROWED` line under
`bake._narrowing_section` and can never be an `EARLY` mapping.
`bake.resolve_bug_deps` runs the move immediately after `resolve_build_pins` and before
`build_installs` is read for the closure walk, so the walk, the constraint file and the recipe's
`build-tools.txt` all see the moved pin rather than the one the date rule first named. Two
boundaries, both `BakeError`s that flag the bug: a claimed name whose release dates were never
fetched (nothing to choose from), and a name no release satisfies — the layer would be installed and
then fail its own build, and refusing that before the build runs is the harness's job. The case
`pandas/1` shows is the one it deliberately does **not** move: `scipy==1.4.1` and
`scikit-learn==0.23.1` each claim `numpy==1.17.3; python_version>='3.8' and platform_system!='AIX'`
while that layer holds `numpy==1.18.4`, but numpy is pinned by the dependency set
(`"dependency_pin": "numpy==1.18.4"` in `runs/bugsinpy/deps/vanta-bi-pandas-1.json`) and so is not
movable — the build proceeds against the held release, which is what `--no-build-isolation` does
anyway.

**Every EARLY mapping is checked against its specifier, because the same function is the
preflight's.** What the measurement exposed was not only the missing move: the layer's provenance
recorded `numpy==1.18.4`'s `Cython>=0.29.14` as *satisfied by* `cython==0.29.7`
(`runs/bugsinpy/deps/vanta-bi-luigi-1.json`, `early_pins`) — a satisfaction the build itself
refuses. `bake.early_install_pins` matched on the requirement's **name**; it now asks
`bake._specifier_accepts` whether the held pin satisfies the clause, so a name the layer pins too
low is not an ordering finding at all — it is the pin that has to move above — while a requirement
with no clause (`setuptools`/`wheel` in `sanic/1`'s `[build-system].requires`, and numpy's own build
needs of the same two on `luigi/1`) is still satisfied by whatever the layer holds. An unparseable
held version is not a satisfaction: a clause that cannot be read against it is a question the build
answers, and answering yes here is how the false mapping was recorded. The check lives in the
function and not in a caller, which is what makes "the preflight checks every EARLY mapping against
its specifier" a property rather than a hope — `bake.preflight_layers` calls this same
`early_install_pins`, so an unbuildable mapping is one the preflight cannot print.
`bake._specifier_accepts` reads the clause through `bake._specifier_of`, i.e. through
`packaging.Requirement`, so a clause's marker is parsed off rather than string-matched.

Measured across the six layers, this moves **one pin** and drops **three mappings from the
`EARLY` list**. `luigi/1`'s `cython==0.29.7` becomes `cython==0.29.14`, and the layer's record of
the move is the narrowing that prints as
`NARROWED cython==0.29.14 — numpy==1.18.4 declares >=0.29.14 (luigi/1)`; the moved pin keeps the
toolchain's source and takes the date of the release it moved to, so `build-tools.txt`, the
constraint file and the layer's `pip freeze` all name 0.29.14, and the `numpy==1.18.4` → `Cython` 
mapping comes back into the `EARLY` list — satisfied now by `cython==0.29.14` rather than by a pin
that refuses it. The other two drops are `pandas/1`'s, whose early count falls from **18 to 16**:
its two `numpy==1.17.3` claims name a release the layer does not hold, and nothing moves in their
place because numpy is not movable there (above), so the requirement is left to the build — which
is what `--no-build-isolation` does with it anyway.

The preflight is read off each layer's **recorded** pins (`bake.layer_pins`), so what it prints
before a re-bake is the drop and not yet the move: on the six records on disk today `luigi/1` shows
`5 early` instead of 6 and `pandas/1` `16` instead of 18, with no `NARROWED` line anywhere, because
the pin that moves does so while the layer is *being* baked and the record is written then. After
the bake, that layer's record carries `cython==0.29.14` with its `narrowed_by`, and the preflight
reads the move back out of it — which is the property the whole block rests on, the record and the
printout being one thing rather than two. Every other layer's pin list, early count and verdict is
unchanged, and the verdict stays `ready to bake`. The rules are tested on the decisions rather than
on a bake: `scripts/test_bugsinpy.py::TestNarrowBuildPins` (eight cases — the oldest-release
choice, the intersection, a satisfied pin left alone, a clause-less requirement, a marker that does
not apply, a pin the dependency set supplied, and both `BakeError` boundaries) and
`TestEarlyInstallPins` (an unsatisfied clause records no mapping, a clause-less one still does, and
the preflight prints no `EARLY` line for it), plus
`test_a_moved_build_pin_is_read_too_and_prints_as_narrowed` for the printout and
`TestMarkerSkips::test_the_preflight_records_the_drops_in_the_json_and_not_on_the_terminal`, which
now reads the same measured `pandas/1` layer through both rules — the marker rule keeping the
`>= '3.8'` clause and dropping the `3.5` one, and this rule reading the kept clause against the
release the layer holds and recording no ordering from it.

**Mandatory bake-time gate (part of the wave-1 gate, §8.1).** For each version checkpoint, assert
*inside the checkpoint*:

```
python -c "import ssl, sys; print(sys.version, ssl.OPENSSL_VERSION)"   # must not fail
python -c "import ssl; ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)"        # functional, not a string
dpkg-query -W -f='${Package} ${Version} ${db:Status-Status}\n' <pkg>    # each one `installed`, 3.8
python -m pip install "pytest==<pin>" "pytest-json-report==<pin>"      # must reach PyPI and install
python -m pytest --version                                             # pinned version, must import
python -c "import pytest_jsonreport"                                   # the plugin, not just the wheel
```

The `ssl` check is functional rather than a version-string match on purpose: an interpreter whose
`_ssl` failed to build can still report a version, and the failure that matters — `pip` cannot
install anything — only shows up when the module is *used*. Standalone builds report OpenSSL 3.x and
are fine; a source-built 3.6/3.7 reports 1.1.1x. Both satisfy the gate, so the gate asserts the
property (works) rather than the version (which would fail the very build §3.2 recommends).

**A *bug* layer runs that same `pytest --version` with plugin autoload off, and the tree's own
importability is not what it is measuring.** Measured, and it is the other half of what `sanic/1`
failed on: after a clean install — every pin present, `pip check` reading one tolerated line — the
layer's own gate died `ModuleNotFoundError: No module named 'sanic'`, because the interpreter layer
carries *every* project's pytest plugins (that is what the per-version checkpoint is for), so a bare
`pytest --version` loads each plugin's entry point, and `pytest_sanic` imports `sanic` at startup. The
project under test is not in the layer by design — it lives in the tree, which is mounted at test
time — so loading its plugin is not a property this gate can hold. It runs as
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --version` (`requirements_recipe`), which asserts
exactly what the gate was for: pytest itself imports and runs under this interpreter. The interpreter
layer's own check keeps the bare form — it is where the plugins are installed, and it has the pytest
they were installed for — and neither form claims anything about a tree. The real suite is
unaffected, and that was checked in the code rather than assumed: `prepare` runs its runs through
`suite.py:run_suite_twice`, which issues `python -m pytest` with `guest_cwd=guest_tree_root(tree_rel)`
and an env of `PYTHONDONTWRITEBYTECODE`/`PYTHONUNBUFFERED` and **no `PYTHONPATH`** — the tree reaches
`sys.path` through `python -m`'s CWD prepend, which `suite.py` states as a deliberate property
(`import_module_path`'s probe reuses the same invocation for that reason) — and Vanta's runner does the
same: `harness.rs` executes `TestCommand::as_cmd_vec()` (`python3 -m pytest`) through
`sandbox.execute(…, Path::new(""), …)`, and the Shuru backend resolves an empty cwd to `/workspace`,
the mount of the sandbox root the tree was written to (`shuru.rs`). Both paths therefore run with the
working directory at the tree root, so the tree *is* importable when the entry-point plugins load, and
no fix is needed there — the gate was the only place running pytest outside the tree.

**Version pins.** `pytest` and `pytest-json-report` are pinned to versions that support each
interpreter — 3.6 caps at pytest 6.x, 3.7 at 7.4.x, 3.8 at 8.0.x — written into the checkpoint build
and recorded per bug in the split (`pytest_version`, `json_report_version`) so a report can be read
back years later.

Those two fields are the **interpreter** layer's pair, and the split is the protocol for that layer
alone: one pair per Python minor, `bake.pins_for_python` refuses a split where two bugs of one minor
disagree, and nothing about them changed when the bug's own `pytest` pin was given precedence above. A
bug's own layer stacks *on top* of it, so what a bug's layer actually carries is a property of that
bug, not of the split: an exact `==` pin on `pytest` in BugsInPy's own `requirements.txt`/`setup.sh`
wins, and the `pytest-json-report` (and `pytest-metadata`) release is then chosen from PyPI's
per-release metadata to accept it. That choice, and the pytest it was chosen for, are in the layer's
`runs/bugsinpy/deps/<checkpoint>.json` under `dependencies` — each plugin pin carrying
`chosen_for_pytest` and its own `pin_rule` — and `bake.layer_pin` reads the pair back out of that file,
so a run's row and a report years later name the versions the suite really ran under. The split files
therefore needed **no edit** for this rule: `bugsinpy_dev_split.json` and `bugsinpy_split.json` still
carry each entry's interpreter-layer pair, and the per-bug reality lives in the provenance beside the
layer it describes.

**A Shuru microVM boots with no `/dev/shm`, so every boot mounts one before it runs anything at all.** A
mount does not survive into a checkpoint, so "the layer has `/dev/shm`" is not a property that could be
baked once and read back later; it is re-established at each boot instead. Measured, and it is why the
`numpy==1.18.5` build above read as a compiler problem: CPython's `multiprocessing` creates its
semaphores with `sem_open` in that directory (`multiprocessing/synchronize.py` →
`_multiprocessing.SemLock`), so the first `multiprocessing.Lock()` in such a VM raises
`FileNotFoundError: [Errno 2] No such file or directory` — **with no filename**, because the errno comes
from the kernel rather than from a failed open of a named path, and `multiprocessing.pool`'s own queues
(`SimpleQueue`) take a `Lock` too, which is how numpy's parallel compile reached it under
`distutils`' `CCompiler_compile`. On the harness layer `vanta-bugsinpy-py3.8`: `ls: cannot access
'/dev/shm': No such file or directory`; `multiprocessing.Lock()` → `[Errno 2]`; and after
`mkdir -p /dev/shm && mount -t tmpfs -o rw,nosuid,nodev tmpfs /dev/shm` the lock is created. It is not
`AR`/`ARFLAGS`: both were probed in that layer and work (`AR=ar`, `create_static_lib` fine).

**One guard, in the two languages this harness runs untrusted code from.** `bugsinpy.shuru.DEV_SHM_GUARD`
(Python) and `src/sandbox/shuru.rs`'s `DEV_SHM_GUARD` (Rust) are one rule with two spellings — a second
*rule* would drift. The guard is idempotent (`grep -q '^tmpfs /dev/shm ' /proc/mounts` first, so the boot
that *created* a layer does not remount under itself), it **stops the boot loudly** (`exit 86`) when
neither the directory nor the mount can be had, and it is wired into every entry point: `checkpoint_create`
(whose recipe therefore also runs it), `run`, the interpreter and per-bug recipes (`bake.python_recipe`,
`bake.requirements_recipe`), and the Rust backend's `execute` — which is the path `pipeline/harness.rs`
uses for every sandboxed command, so the same guard precedes the native suite, the generated tests,
`ruff` and the checkpoint-E probe. The checkpoint-marker probe reads `exit 86` as a fact about the
*machine* rather than a missing marker, so a host that cannot mount a tmpfs fails the boot instead of
deleting a layer that is fine — the same class of loss the two-attempt loop exists to prevent, arriving
through another door.

**The mount is a claim; the smoke is the reading.** The guard follows the mount with
`multiprocessing.Lock()` and a 2-worker `Pool.map` (`shuru.MP_SMOKE_CODE`, the same string
`bake.toolchain_smoke_lines` runs in the interpreter layer, so the layer and the boot make one check and
not two), failing with `exit 87` if either raises. A VM with no interpreter skips it rather than failing —
the bare base image, before `create_python_checkpoint` installs one.

**Recorded, so a layer says whether it can make a semaphore at all.** `bake.build_env_lines` exports
`VANTA_DEV_SHM=tmpfs mounted` into the shell the build runs in, asserts it is set *before* writing the
record (`env | grep` omits an unset variable, so a layer whose boot never ran the check would otherwise
write a file that says nothing either way), and the variable then appears in `/opt/vanta-build-env.txt`
and in the layer's provenance beside `CC`/`CXX`/`LDSHARED`/`CC_VERSION`. `shuru.DEV_SHM_MOUNTED` is the
one spelling of the value. Without this, a bug whose tests use a `Pool`, a `Queue` or a `Lock` fails
inside the VM as a *test* failure that names no file — and §3.2's own gate cannot see the difference:
`bugsinpy.suite.SKIPPED_OUTCOME` is excluded from both `F_before` and `F_after`, so a suite that reacts to
a missing `/dev/shm` by **skipping** (tqdm's `tests/conftest.py` does exactly that, in the tree this repo
already has) is invisible there as well.

**No earlier result was affected.** Every `deps/*.json`, every `bake-logs/*`, the `prepare-wave1-*`
reports and run logs, and the suite logs were searched for `SemLock`, `/dev/shm` and a `multiprocessing`
`FileNotFoundError`. The only hit is source text — `sanic`'s own `tests/test_multiprocessing.py` inside a
recorded import closure — and no run record contains such a failure or a skip attributed to one; the wave-1
runs completed all six rows. No earlier suite result is invalidated: the recorded runs never reached a
test that opens a semaphore, which is what makes this a fix in front of the failure rather than after it.

**The build VM is sized for a build, not for a suite.** A bake boots the VM with 2 CPUs and 2048 MB
because `checkpoint_create` passed no `--cpus`/`--memory` and inherited shuru's defaults — enough to
run a suite, not enough to compile `scipy`. Measured: `pandas/1`'s `scipy==1.4.1` build died with
`g++: fatal error: Killed signal terminated program cc1plus` on `scipy/sparse/sparsetools/*.cxx` — the
kernel's OOM killer, not a compiler diagnostic — in a VM that logged
`shuru: booting VM (2cpus, 2048MB RAM, 16384MB disk)`. `bake` now asks for
`shuru.DEFAULT_CPUS` (8) and `shuru.DEFAULT_MEMORY_MB` (8192) on **every** layer's `checkpoint create`
(`--cpus` / `--memory <MB>`, the real flag names), overridable with `bugsinpy_run.py bake --cpus` /
`--memory`. The number of parallel compile jobs is capped to match the RAM — `NPY_NUM_BUILD_JOBS`
(default `min(cpus, memory_mb // 2048)` = 4 at 8 CPU/8 GB), which numpy's `ccompiler._cpu_count` reads
in place of `os.cpu_count()` — so a wide machine cannot re-open the same OOM through parallelism.

**Both numbers are VM sizing, not recipe content, so no layer's recipe changes.** `--cpus`/`--memory`
are `checkpoint create` flags (a sibling of `--disk-size`, itself already outside the recipe), and the
`NPY_NUM_BUILD_JOBS` export plus its `vanta bake VM:` echo ride the **boot prelude** — the `sh -c` line
`checkpoint_create` runs before the recipe, exactly where `DEV_SHM_GUARD` already sits. Neither edits
the interpreter or per-bug recipe strings. The already-baked layers (`base`, `fastapi-1`,
`matplotlib-1`, `sanic-1`, `PySnooper-3`) therefore **stay valid** and need no rebuild: their contents
are unchanged, and the sizing is re-established at each boot rather than baked into the filesystem, the
same property `/dev/shm` has. The sizing is *recorded* so a layer can be read back later: the
`vanta bake VM:` line in the build log and `deps_provenance`'s `build_vm` block (`cpus`, `memory_mb`,
`build_jobs`) in each `deps/*.json`, because from an install list alone a build that ran out of memory
is indistinguishable from one that built cleanly.

**A build requirement *no* pin covers is date-pinned and installed before the build — the third rule
(§3.2, item 2, on top of item 1's `2f8cbb3`).** The two blocks above install what a build wants *when a
pin of the layer satisfies it* and move a pin a requirement refuses; between them they still say nothing
about a name the layer holds **no version** of, and that third case is what `luigi/1` measured.
`ujson==2.0.3` builds here from its sdist, its `setup.py` declares `setup_requires` naming
`setuptools_scm` (`ujson==2.0.3 setup_requires (setup.py) line 24`), and no pin held it, so with
`--no-build-isolation` setuptools resolved it itself: `setuptools/dist.py:740 finalize_options` loaded
the entry point of `setuptools_scm-10.3.4` out of `ujson/.eggs/` — *today's* release, fetched by
`fetch_build_eggs` while the metadata step ran — and that release's
`setuptools_scm/_integration/version_inference.py:18` does `from setuptools import sic as
setuptools_sic`, while the layer's own `setuptools==42.0.0` has no `sic` (quoted from the log's
traceback, `runs/bugsinpy/bake-logs/vanta-bi-luigi-1.log` 585–603; read off the wheels: `sic` is absent
in `setuptools` 42.0.0, 45.3.0 and 46.0.0 and present from **46.1.0**, as `class sic(str)` in
`setuptools/__init__.py`). The metadata step died with `ImportError: cannot import name 'sic' from
'setuptools'` and `checkpoint create` saved the half-built layer anyway, which is why the failure read as
a saved checkpoint plus one error line.

`bake.uncovered_build_requirements` names such a requirement before the build runs: it reads each pin pip
will actually **build** here (the same `bake.SDIST_ONLY` filter `early_install_pins` uses, and for the
same reason — a release pip installs from a wheel never runs its own `setup.py`) and reports every build
requirement whose canonical name no pin of the layer holds, whose declaring release is not itself, and
which is not a harness name (`bake.HARNESS_PACKAGES`) — recording the declarer and the line it wrote,
once per `(required, declarer, source)`, with §3.2's marker rule applied to the **declaring** release's
own field. `bake.build_requirement_pins` then dates it the way every other unpinned pick is dated: **one
pin per name** (a layer holds one version of a name, and two of its pins can declare the same tool), the
newest release on or before the bug's commit date that satisfies **every** declaration's range, recorded
with `pin_rule = "build requirement, date rule"` and a `declared_by` naming the release whose build asked
— and a declaration nothing at the bug's date satisfies is a `BakeError` that **flags** the bug, like
every other date-rule refusal, because that is exactly the layer that would fall back on `easy_install`.
The pins ride the existing `build_pins` list, so the constraint file, `build-tools.txt`, the recipe's
early pass and the preflight table all pick them up with no new channel; `bake.resolve_bug_deps` runs the
reading and the dating **before** `narrow_build_pins`, so a name that is both covered by nothing and
refused by a held pin is answered by both rules over one pin list. The `easy_install` fetch guard above is
unchanged and stays as the backstop.

**The `NARROWED` line was printing the wrong declarer, an empty clause and one bug key seventeen times.**
The preflight's special-cause reading of that layer was wrong in the print and not in the pin:
`setuptools==42.0.0` is the date rule's own answer for `luigi/1`'s commit date, and the one claim that
moved it is `zipp==3.1.0 declares >=42` — not `pip==21.3.1` (which declares nothing on that name) and not
an empty clause. `bake.narrow_build_pins` now drops a claim that wrote **no** clause before it is stored
(`text = _specifier_text(ref); if not text: continue`), `bake.pin_narrowings` skips a stored claim with an
empty `requires`, and `bake._narrowing_section` keys both its maps, so each distinct claim and each bug
key print once: `NARROWED <pin> — <declarer> declares <clause> (<keys>)`. An empty clause never bounded or
moved a pin, so the guard that drops it **cannot change any version** — the change is on the terminal
only.

**Measured over the six bug layers on disk, the rule dates a pin into four of them.** Re-applying the
reading to each layer's recorded pin list and per-pin `kind` (via the harness's own sdist reader, so the
answer is the code's and not a hand count):

| layer | new build pin(s) | declarer(s) | commit date |
|---|---|---|---|
| `luigi/1` | `pytest-runner==4.4`; `setuptools-scm` **missed** (`2874f27`) | `idna-ssl==1.1.0`; `ujson==2.0.3`, but only as a *transitive* pick | 2019-05-09 |
| `pandas/1` | `pybind11==2.5.0`, `pytest-runner==5.2` | `scipy==1.4.1`, `fastparquet==0.4.0` | 2020-05-21 |
| `fastapi/1` | `setuptools-scm==3.5.0` | `ujson==3.0.0` | 2020-04-05 |
| `sanic/1` | `setuptools-scm==3.5.0`, `pytest-runner==5.2` | `ujson==2.0.3`, `idna-ssl==1.1.0` | 2020-05-12 |
| `matplotlib/1` | — | — | — |
| `PySnooper/3` | — | — | — |

**The `luigi/1` row above was wrong, and the bake is what showed it.** The `setuptools-scm==3.2.0` figure
came from an offline re-reading that was handed the layer's **full post-walk** pin list, and at that point
the code did not have one: `bake.resolve_bug_deps` ran the reading over the pins the layer held *before*
its syntax closure walk, and in `luigi/1` the declarer `ujson==2.0.3` is a **transitive** pick —
`sanic==20.3.0` declares it and the bug's own tree never names it — so its `setup_requires` was never
read. The baked evidence is in `runs/bugsinpy/deps/vanta-bi-luigi-1.json`: `pytest-runner==4.4` is there
(`build requirement, date rule`, declared by `idna-ssl==1.1.0`) and `setuptools-scm` is not mentioned at
all, and that layer died on the same `ImportError: cannot import name 'sic'` the rule was written for.
`sanic/1` is the control that hid it: the same `ujson==2.0.3` is *declared* in that bug's own tree, the
pre-walk reading saw it, and that layer built under `setuptools-scm==3.5.0` — so the rule was right and its
*reader* was scoped to the wrong list. `2874f27` narrows that scope (the landing note below); after it
`luigi/1` carries whatever release the next bake dates for 2019-05-09, not this row's 3.2.0.

**The next bake is `--force`, and this time because the last one proved the rule needs it twice over.**
`1737bf6` added a member to `COMPILER_ENV`, and that mapping is written into the **interpreter** layer's
recipe (`export CFLAGS=…`), so `vanta-bugsinpy-py3.8`'s recipe changes and every per-bug layer is `--from` a
base that no longer matches it. `plan_layers` skips a layer that is already present unless `--force` and does
not invalidate on a recipe change, so `--force` for the whole run is the answer — the same shape item 3's
`FFLAGS` change had. **The bake that ran `--force` over `1737bf6` is what turned that from a rule into a
measured need**: the member it added was the header alone, three layers died on `Dl_info` (above), and
`plan_layers` rebuilding nothing would have hidden it behind the stale checkpoints. The luigi half alone
would have needed only `vanta-bi-luigi-1` rebuilt, because the two layers the two-rule reading was measured
on are evidence that it works rather than victims of it: `fastapi-1` and `sanic-1` already carry the pins it
produces. That is also why the retry is not `--only luigi/1 --only pandas/1`: those two would be rebuilt over
an interpreter layer whose `CFLAGS` no longer matches their build logs.

**Two defects the same bake exposed, fixed together — the header's cost and the smoke that could not see
it.** Both are recorded here rather than in the paragraph above because they are corrections of *that*
commit's two halves, and both are measured rather than reasoned:

*The header needs a feature-test macro beside it.* `bake._ERA_GNU_SOURCE` is `('-D_GNU_SOURCE',)`, a member of
`CFLAGS` immediately before `_ERA_STDINT_INCLUDE`. `-include stdint.h` alone lost glibc's `__USE_GNU` on every
**C** tree in the layer — `matplotlib/1`, `pandas/1` and `sanic/1` all at
`numpy/core/src/multiarray/temp_elide.c:158:9: error: unknown type name 'Dl_info'` — because `-include` is
read as the translation unit's first line, before the source's own feature-test macros can set it (the
mechanism is in §3.2 above). GCC applies command-line `-D` before any input, `-include` files included, so the
macro belongs on the command line and not in a source. The signature this leaves a reader: the *same* layer
that built clean before the header now fails, which is a regression and not a fourth bug.

*The smoke check was inert, and unsatisfiable.* `bake.toolchain_smoke_lines` wrote its three checks as one
`cd DIR && setup.py build_ext && python -c …` AND-OR list, and measured, an AND-OR list does not trip `set -e`
on a non-final member (`sh -c 'set -eux; false && true; echo CONTINUED'` prints `CONTINUED`, `exit=0`), so the
C++ compile's hard error let `checkpoint create` run on to `pip freeze` and `touch /opt/vanta-deps-ok` — the
layer was saved and reported `[done]` with a broken toolchain inside it. The same C++ file could never have
compiled anyway: it left `std::uint16_t` to `-include stdint.h`, and a plain `distutils`/`setuptools` C++
`Extension` on that layer seeds its C++ flags from the **recorded** `cflags` and never reads the environment
`CFLAGS` (§3.2), so the check was red on every layer by construction. Each step is now its own statement
(de-chaining both `&&` groups), and the C++ file has since been brought back to what it should have been: it
uses `std::uint16_t` and does **not** include `<cstdint>` itself, so a successful build-and-import is the
receipt that the header arrived from the driver word (`bake.VANTA_CXX_PATH`, §3.2) and not from the file —
the version that included `<cstdint>` proved nothing about the wrapper.
`bake._TOOLCHAIN_SMOKE_CPP_SOURCE`'s comment and the §3.2 smoke paragraph state what it checks and what it
deliberately does not.

**`luigi/1`'s error had moved, and the fix is a reviewed mapping — `mysql.connector`.** With the toolchain
fixed and de-chained, `luigi/1` no longer died on `ImportError: cannot import name 'sic'` (the build pins are
correct: the log's `pip freeze` shows `setuptools==42.0.0`, `setuptools-scm==3.4.1`, `pytest-runner==4.4`); it
died one layer down, on a *dependency*. The import closure at `test/_mysqldb_test.py line 20` imports
`mysql.connector`, and the fallback resolved it to the top-level word `mysql` — a PyPI distribution of its
own, `mysql` 0.0.2 (2018), a 'virtual package' whose sdist declares `mysqlclient` — so the layer tried to
build `mysqlclient` and stopped on
`Exception: Can not find valid pkg-config name. Specify MYSQLCLIENT_CFLAGS and MYSQLCLIENT_LDFLAGS env vars
manually`, the sandbox having no MySQL client library. The module is `mysql.connector`, not `mysql`; the
reviewed table is keyed by the **import spelling** precisely so the dotted name is matched before its
top-level one (`google.protobuf` is the precedent). Both splits now map
`"mysql.connector": "mysql-connector-python"` (line 36 of each), with the matching `reviewed_changes` entry
recorded on `2026-10-24`. Under §3.2's date rule the newest release of `mysql-connector-python` on or before
the declaring commit's date that PyPI still carries is **8.0.5** (2017-11-20; 8.0.6–8.0.22 are absent from
the index), which publishes `mysql_connector_python-8.0.5-py2.py3-none-any.whl` — pure Python, nothing to
build, no system library. The harness change is only the two split files: the reviewed table is data, and
nothing in `bake.py` learns a project name.

**Tests.** `scripts/test_bugsinpy.py::TestUncoveredBuildRequirements` (eight cases: the declarer is kept,
a name the layer pins is left to the two rules around it, a build requiring itself, a harness name, only a
pin pip will build is asked about, an unreadable release, a marker that does not apply, and two declarers
of one tool) and `TestBuildRequirementPins` (the newest release on or before the date, the declared range
bounding it, both `BakeError` boundaries, one pin satisfying every declaration, and the `pin_rule`/
`declared_by` provenance), plus `TestABuildRequirementNoPinCoversReachesTheLayer` for the wiring —
fetched, dated, recorded as a *build* pin and ordered early — and `TestPinConflicts`' two print cases
(`test_a_narrowing_claim_with_no_clause_is_not_made_into_a_reason`,
`test_a_repeated_reason_and_key_print_once`) and `TestNarrowBuildPins`' clause-less case. The canonical
name is asserted throughout: the reader stores `canonicalize_name(ref.name)`, so `setuptools_scm` becomes
`setuptools-scm` in the pin and only the recorded declarer keeps the raw spelling. This change adds
`test_the_c_flags_define_the_macro_the_pre_included_header_costs` (the macro is in `CFLAGS` and precedes the
include), `test_the_smoke_checks_are_separate_statements` (no line of the smoke is an `&&` chain, `cd` stands
alone), the rewritten `test_the_smoke_check_builds_and_imports_the_cpp_half`, and the `mysql.connector` row in
`test_the_reviewed_module_table_is_pinned_in_both_splits` and its entry in
`test_real_split_reviewed_changes_carry_the_rule_and_its_measurements`. Full run: `962 passed`.

### 3.3 BLOCKER 2 — dependency detection under-provisions

[`deps::detect()`](file:///Users/kyu-yi/vanta/src/sandbox/deps.rs) reads `requirements.txt` and
`pyproject.toml`. Observed reality:

- `youtube-dl/1/requirements.txt` is **0 bytes**; youtube-dl declares deps in `setup.py`.
- tqdm / cookiecutter use `pythonpath="<pkg>/build/lib/"` — the project must be **built**
  (`python setup.py build`) before its own tests can import it. BugsInPy's `bugsinpy-compile` does
  exactly this.

So `deps::detect` alone will under-provision. The mitigation is the five-source union in §3.2, built
by `bake.resolve_bug_deps`, with two rules that were learned the hard way:

1. **Read `setup.py`, never execute it.** At the buggy commit, `setup.py` is how these projects
   declare their deps (`install_requires`, `tests_require`, a `tests` extra, or a file they read from
   `install_requires`), so it is source 4 in §3.2 — parsed with `ast`, never run on the host. Running
   it would mean executing third-party code outside Shuru (C3). Reading it statically includes
   following a `setup(**kwargs)` splat back to its module-level dict literal (§3.2, the sanic/1
   case); and a bug whose metadata even that cannot name may carry a reviewed `import_package` in its
   split entry, used last and only for the import-path check. If a future project's deps genuinely
   cannot be read statically, that bug is flagged, not improvised around.
2. **Install the deps, never the project.** `python setup.py install` is deliberately *not* run: it
   writes the buggy package into `site-packages`, where it would shadow the patched tree under test
   and make every patch grade the same (§3.2). The skipped line is recorded in the layer's provenance
   with the reason, so "the dep was not installed" is never indistinguishable from "the dep is not
   needed".

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
and **rejects the bug from the split** if it does not collect. The resolved `test_cmd` is stored per bug
in the split, alongside the *interpreter* layer's pinned `pytest` / `pytest-json-report`; the pair the
bug's own layer actually carries is a property of that layer and lives in its provenance (§3.2), which
is also where the report format is decided — a bug whose own pin caps the plugin at 0.2 writes the
older schema, and **every** reader reads both, Python and Rust alike (§3.2 point 3).
`run_test.sh` is still read, but used as the *cross-check* on which nodeid is the hidden test
(§1.3), not as the executor.

**Prepare-time dependency gate.** Collectability is not only about pytest: a bug whose layer is
missing a declared dep collects *zero* tests, and that failure is indistinguishable from a broken
oracle. PySnooper/3 failed exactly this way — `tests/test_pysnooper.py` imports `python_toolbox`,
which no BugsInPy source installs. So `prepare` runs two checks in the bug's layer, **before** oracle
derivation, and reports them as dependency failures, never as oracle failures:

| Check | Failure message |
| --- | --- |
| `pytest --collect-only -q` over the oracle tree's test files collects >0 with no collection errors, using the suite's `test_cmd` | `deps: collection failed` |
| the project package imports **from the tree under test** — `python -c "import <name>; print(<name>.__file__)"`, then assert that path is inside the tree, not `site-packages` | `deps: package imports from site-packages, not the tree` |

The import check reuses **exactly** the suite invocation — same interpreter (`test_cmd[0]`), same
cwd (the tree root), same env, and no `PYTHONPATH` — because that is the only import path the suites
actually have. The module name comes from the project's own `setup.py` (`name=`, or its first
non-`tests` package), never from a guess. A tree that passes both checks is one whose suite can
import what it tests.

#### 3.4.1 The tree is built before either check (item 5)

Both checks above were run on a tree that had never been compiled, and on `matplotlib/1` that is the
whole failure: `lib/matplotlib/tests/conftest.py` died with

```
ImportError: cannot import name 'ft2font' from partially initialized module 'matplotlib'
```

so `--collect-only` collected nothing and the gate reported `deps: collection failed` — a missing
**extension** described as a missing **dependency**, which is the confusion §3.3 exists to prevent,
in the other direction. A tree is a `git` export of the bug's commit; a project whose package has C
extensions is not importable until its own `setup.py` has run, and the export never runs it.

**The step.** After the layer is loaded and **before** the dependency gate, `prepare` builds each of
the bug's trees in place, inside the layer's VM
([`prepare.build_trees_in_place`](file:///Users/kyu-yi/vanta/scripts/bugsinpy/prepare.py),
[`suite.build_tree_in_place`](file:///Users/kyu-yi/vanta/scripts/bugsinpy/suite.py)):

| | |
| --- | --- |
| **Which trees** | `oracle_tree`, `src_buggy`, `src_fixed` — the three independent exports. `vanta_tree` / `vanta_clean` are `copy_tree` copies of `oracle_tree`, taken **after** this step in `strip_and_baseline`, so the compiled objects travel with the copy and are not built a second time. The record names `vanta_tree` anyway, so "was Vanta's own tree importable?" has an answer and not an absence. |
| **When** | only in a `full` prepare, after `load_deps` (the layer is where the toolchain and the compiler environment live) and before `check_deps` (the gate that imports the tree). |
| **The rule** | `bake.needs_build_at` — read *again* here rather than threaded out of `check_build_declared`, because one function answers "does this need a build" (§3.2) and a second reading could disagree with the one the split's flag was validated against. |
| **The command** | `python setup.py build_ext --inplace` — or `python setup.py build` when BugsInPy's `pythonpath` points *into* a build directory (`tqdm/build/lib/`), where only a full build puts the package the suite imports ([`prepare.setup_args_for`](file:///Users/kyu-yi/vanta/scripts/bugsinpy/prepare.py)). Both triggers of §3.2's rule have a command. |
| **The environment** | `bake.build_env_for(minor)` — `COMPILER_ENV`, passed **per command** and not assumed: a checkpoint is a filesystem, and an export does not survive into a layer baked `--from` another (`bake.build_env_lines`), so without it the build runs the compiler the interpreter *recorded*, which is the machine it was built on. |
| **Recorded** | `BugPrep.builds: {tree → "built: <command>" \| "not built: <why>"}`, written to both `prepare.json` and the split's derived block (a `DERIVED_BUG_KEYS` member, beside `deps`). A tree with no entry would mean the stage was skipped rather than that the tree needed nothing. |
| **On failure** | its own gate reason — `build failed in <tree>: \`python setup.py …\` exited N; <error lines>` — and the bug stops there. It is never reported as the collection it would have caused. A build that runs long is `build timed out after <n>s`, with a timeout longer than the suite's own (3600 s vs 1800 s) so the two most different failures never share a sentence. |

`pandas/1` is the same rule (`needs_build: true`; `ext_modules` and Cython sources of its own), and
so is anything else whose tests import a package the tree has to compile. `build/` is in the Rust
walker's `SKIP_DIRS` and the compiled objects (`.so`) are not a `PROJECT_SOURCE_EXTENSIONS` suffix,
so building in place does not change the file set the §1.6 gate measures — the two prechecks and
this step cannot drift because of it.

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
3. Run the **hidden tests** (the declared `FAIL_TO_PASS` derived in §1.3) → `hidden_pass` (bool).
   Tests that flip but are not declared for the bug are not in this set, so they cannot be graded.
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
   `attempt_N_<kind>_tree.diff` (unified diff vs. `vanta_clean/`, pinned by commit SHA) +
   `attempt_N_<kind>_tree.hashes.json` (`{path: sha256}`, plus the tree's `concat_sha256`), then delete
   `attempt_N_{patched,graded}_tree/`. **Both** snapshots are diffed, not just the graded one: the
   graded tree is what the rerun reads and the patched tree is its documented fallback, so deleting
   either without its diff would destroy evidence the rule never described. Three deliberate
   deviations from the sketch above: `sha256` rather than `blake3`, because `split.py`'s
   `sha256_text` / `sha256_file` are the harness's only hash and a second algorithm is a second thing
   to get right; `_tree.hashes.json` rather than `_manifest.json`, because Vanta already writes
   `attempt_N_<kind>_tree_manifest.json` and `rerun.rs`'s tree reader **skips** `*_manifest.json`, so
   reusing that name would either overwrite Vanta's own forensics or be invisible to the reader that
   matters; and `concat_sha256`, because the concatenation is the unit the test below compares, which
   is what lets the replay be checked against an artifact on disk rather than against a tree the test
   still happened to hold in memory.
4. **Test** (decided): a harness test asserts that (a) a session with a rejecting signal
   is *not* compacted before its rerun, and (b) a compacted session replays to a byte-identical tree
   from base + diff — i.e. `read_tree_concat`
   ([rerun.rs](file:///Users/kyu-yi/vanta/src/pipeline/rerun.rs#L1239)) over the reconstructed tree
   equals the recorded concat. Fails when the evidence for replayability is absent, rather than
   trusting the implementation. The test reads the diff and the hashes **back off disk** and rebuilds
   the tree with `git apply`, so a compaction that deleted the only copy without leaving a replayable
   artifact fails the test instead of surfacing later as a fabricated `spurious_reject`.

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
enter this file through a deliberate edit. (`needs_build` is the one value the authoring step derives
rather than a human setting it: it is read from the buggy commit's own tree by `bake.needs_build`, per
§3.2, and `prepare` re-checks the written file against that tree both ways — so what is reviewed here
is the rule, and the value is its output.)

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
      "issue_text_sha256": "…",
      "deps": {
        "schema_version": 1,
        "checkpoint": "vanta-bi-PySnooper-3",
        "buggy_commit": "6e3d797…", "commit_date": "2019-04-22",
        "fixed_commit": "15555ed…", "fixed_commit_date": "2019-04-22",
        "import_name": "pysnooper",
        "harness_pins": {"pytest": "8.0.2", "pytest-json-report": "1.5.0"},
        "installs": ["decorator==4.4.0", "future==0.17.1", "six==1.12.0", "python-toolbox==0.9.3"],
        "dependencies": [{"name": "decorator", "spec": "decorator>=4.3.0", "sources": ["install_requires (requirements.txt) line 1"], "pin": "decorator==4.4.0", "provisioned_by": "pinned at bake time", "pin_rule": "the newest release on or before the buggy commit's date", "released_at": "2019-03-16", "commit": "6e3d797…", "commit_date": "2019-04-22"}],
        "pins_after_commit_date": [],
        "optional_imports": [],
        "skipped": [{"source": "setup.sh", "line": 1, "text": "python setup.py install", "reason": "…"}],
        "notes": ["…"]
      }
    }
  ]
}
```

`deps` is the derived half of §3.2: it mirrors the per-bug layer's `runs/bugsinpy/deps/<checkpoint>.json`
verbatim, so a graded run can be traced back to the exact pins that were installed and to the
`setup.sh` lines that were deliberately skipped.

`vanta_planner` is listed in the doc and supported by the env switch but **not** in `configs` until
it exists. `seeds` is `[0]` for the pilot; seeds are a later, paper-only sweep (§8.3).

The `fail_to_pass` / `prompt_tokens_estimate` / `context_ok` / `issue_text` fields above are shown in
their *frozen* state: they are what a reviewed edit writes back into this file once the batch artifacts
exist. `prepare` itself never writes it — it writes `runs/bugsinpy/<batch>/prepare.json` and
`context_report.json` — so the committed file never contains a value no one approved.

#### 7.2.1 `prepare.json` ⇄ `load_prepared` — one record contract, not two copies

`prepare` writes `runs/bugsinpy/<batch>/prepare.json` and `run` reads it back with
`runner.load_prepared`. Those two halves were hand-written in different modules and **drifted**: the
first smoke run died with

```
TypeError: IssueText.__init__() missing 3 required positional arguments:
'leak', 'failure_kinds', and 'needs_decision'
```

because `IssueText` had grown three fields and the runner's reader was still building the old shape —
and a `TypeError` names neither the field nor the batch, which is what a reader needs in order to fix
it. The same audit found a second, quieter drift: `prepare` wrote `context` and `deps` and the reader
restored **neither**, while every other field was read with a `.get(name, default)` fallback that hid
the gap. `BugPrep.runnable` requires `context`, so a loaded batch could never pass `select_preps` and
the run would have failed with "these bugs did not prepare as runners" about bugs that had prepared
perfectly.

The rule is now structural (`scripts/bugsinpy/records.py`):

- Every dataclass `prepare.json` stores — `BugPrep`, `IssueText`, `LeakResult`, `LeakRemoval`,
  `ContextVerdict`, `ConcatStats` — owns exactly one `to_dict` / `from_dict` pair, and **both sides
  call it**: `prepare`'s writer and the runner's loader are the same code, not two copies.
- `field_of` / `strings_of` read a stored field or raise `RecordError` naming **the field and the
  record** (for `prepare.json`, the bug and the batch directory). A missing field is `"missing field
  'leak'"` in `IssueText in PySnooper/3 of batch runs/bugsinpy/<batch>`, not a constructor
  `TypeError`. `strings_of` additionally checks the value is a list, so a bare string is a loud error
  rather than a silent list-of-characters.
- `prepare_manifest()` is the only writer of the file's envelope, so the round-trip test builds the
  payload with the writer the stage uses instead of a second copy of the format.
- `prepare.json`'s top level also records `n_ctx` / `n_ctx_source` (§1.6.1); `run` compares them with
  its own window and warns on a mismatch.

The round-trip test builds a `BugPrep` the way `prepare` does — a real `IssueText` from
`build_issue_text`, with `leak`, `failure_kinds` and `needs_decision` set, a measured `ContextVerdict`
and real `deps` — serialises it with `prepare_manifest`, loads it with `load_prepared` and asserts
every field comes back equal. No VM, no clone.

### 7.3 Manifest — one record per (bug, config, seed)

`runs/bugsinpy/<batch_id>/manifest.jsonl` — append-only, one JSON object per line:

```json
{
  "schema_version": 1,
  "batch_id": "…", "wave": 1, "bug": "PySnooper/2",
  "config": "vanta", "mode": "repair", "seed": 0,
  "session_id": "…",
  "harness_commit": "<git HEAD of this repo when the batch started>", "harness_dirty": false,
  "vanta_build_commit": "<VANTA_BUILD_COMMIT of the binary that ran>", "vanta_build_dirty": false,
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
                "issue_sha256": "…", "compact_state": "full|compacted|none",
                "compact_reason": "…"}
}
```

Note `sampling`: the harness chooses **no** run-level value. `pinned` is the per-call schedule the
pipeline itself enforces (§4.3), so a run's parameters are several rather than one, and
`server_top_p` is the one parameter no call site pins, read from `/props` (`null` when the server
publishes nothing). The manifest never *implies* the harness picked them.

Note `python_version`: that is the interpreter the checkpoint **actually** contains, not the version
BugsInPy's `bug.info` pins. The two differ (§12.3) and both are recorded, so the gap is auditable
rather than hidden by writing the actual number into a field that claims to be the pinned one.

Note `pytest_version` / `json_report_version`: the pair the layer **actually** carries, read out of
that checkpoint's provenance with `bake.layer_pin` (§3.2), not copied from the split's entry. The
example above is PySnooper/2, which has no `pytest` pin of its own, so its layer carries the interpreter
layer's pair and the two agree; a bug that pins `pytest` exactly in BugsInPy's own files records that
version here, together with the `pytest-json-report` release chosen to accept it. `null` when the
checkpoint has no provenance to read at all.

Note `artifacts.compact_state`: `full` (snapshots still on disk), `compacted` (replaced by a diff +
hashes), or `none` (the session wrote no snapshot at all — an `InfraFailure`, or a patch that never
applied). `none` is a third value rather than a reuse of `full` because "the snapshots are still
here" and "there never were any" are different facts; `compact_reason` records which §6.3 branch
decided it. Compaction runs **before** the row is written, so the row states what actually happened
rather than what was planned.

Note the **two commits**: `harness_commit` is this repo's HEAD (`git rev-parse --short HEAD`, taken
once per batch when `run` starts, with `harness_dirty` from `git status --porcelain`), while
`vanta_build_commit` is what `build.rs` baked into the binary at cargo build time and what
`run.summary` reports — the same value §4.4 attributes to `VANTA_BUILD_COMMIT`. They are different
trees and routinely differ: the harness changes between the build and the batch. One column cannot
serve both, and treating the build commit as "the commit the batch ran at" is what made a digest
claim `44f58ab` for a batch whose HEAD was `b2efa49`. At run start, `run` diffs `src/`,
`Cargo.toml`, `Cargo.lock` and `build.rs` between `vanta_build_commit` and HEAD and prints
`WARNING: …` to stderr when they differ or the build was dirty — that, not the digest, is what says
the binary is stale. A change under `scripts/` alone is a harness change and is not a stale binary.

`config` / `mode` / `seed` / `model` / `sampling` / `harness_commit` / `vanta_build_commit` are the
provenance axes §4.4 and §9.1 group by. `candidate` is `null` for **repair** runs. For review runs it names the patch under
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
| PySnooper | 3 | 3.8.1 | 3 | 1 | tiny, ~40 k tok |
| thefuck | 20 | 3.7.0 | 1, 2, 3 | 2 | small, **borderline** — measure |
| httpie | 5 | 3.7.3 | 1, 2, 3 | 2 | small |
| cookiecutter | 4 | 3.6.9 | 1 | 3 | small, needs `setup.py build` |
| **youtube-dl** | ≥1 | 3.7.0 | ~~1~~ | — | **megabytes — expected context reject** |
| black | 20 | (verify) | reserve | — | medium, slow to collect |
| tqdm | 9 | 3.6.9 | reserve | — | small, needs `setup.py build` |

**The `youtube-dl` decision (review point).** youtube-dl is hundreds of extractor files — the source
concat is orders of magnitude past any window we run. Rather than silently dropping it, it stays in
the split with `context_ok: false` so the precheck's verdict is **auditable in the committed record**,
and it consumes **zero runs**. Its slot in the pilot is taken by a reserve. The
`context_sizes.md` dump (§1.6) reports sizes for **all** candidates, reserves included, so the
choice is data, not assertion.

Reserves, in order: **black 1–2** (first choice — 20 bugs, so the reserve is not exhausted by one
bad environment), **tqdm 1**, then further PySnooper/httpie bugs.

Waves (decided) — ordering by Python version, cheapest environment first, because the interpreter
build is the only step that can invalidate a whole batch:

| Wave | Python | Bugs | Seeds | Purpose |
|---|---|---|---|---|
| 1 | 3.8.1 | PySnooper 3 | `[0]` | **go/no-go gate** — checkpoint, `import ssl`+pip gate, oracle, leak gate, runner |
| 2 | 3.7.0 / 3.7.3 | thefuck 1–3, httpie 1–3 | `[0]` | the pilot proper |
| 3 | 3.6.9 | cookiecutter 1 | `[0]` | the last pilot slot; the two-repo `setup.py build` case |

Pilot = waves 1 + 2 + 3 = **8 bugs** (`PySnooper/1` and `PySnooper/2` are excluded —
PySnooper/1 is a Python-2-only encoding bug that passes on both commits under 3.8, and PySnooper/2's
BugsInPy `bug_patch.txt` is an incomplete developer fix whose oracle no patch can pass; see the
split's `reviewed_changes`).

Per-bug environment notes that shape the estimates:

| Bug | py | Checkpoint | Notes |
|---|---|---|---|
| PySnooper/3 | 3.8.1 | `vanta-bugsinpy-py3.8` | clean, no `pythonpath`, no build step — cheapest, standalone Python |
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
| 1 (gate) | 1 | `[0]` | 6–10 | 20–40 min (py3.8, standalone) | 3–6 h |
| 2 (pilot) | 6 | `[0]` | 36–60 | 30–70 min (py3.7 + OpenSSL 1.1.1) | 6–12 h |
| 3 (last slot) | 1 | `[0]` | 6–10 | 30–70 min (py3.6 + OpenSSL 1.1.1) | 1–2 h |
| **total** | **8** | `[0]` | **48–80** | ~1.5–3 h | **~12–22 h → 2 overnight batches** |

Runs are strictly sequential (one LLM slot), so wall clock is the **sum**, not the max. At
`seeds=[0]` the whole pilot fits comfortably; the previous `seeds=[0,1,2]` plan (98 runs in wave 2
alone) did not fit and is dropped (§8.3). Adding `vanta_planner` later multiplies the vanta rows by
~1.5.

**Wave 1 is the go/no-go gate (decided).** If the checkpoint build, the `import ssl` + pip assertion,
the oracle, or the leak gate misbehaves on wave 1, `run` **stops and reports** before any wave-2 work
is scheduled. Wave 1 is deliberately cheap — 6–10 runs, `seeds=[0]`, one interpreter — so a bad
assumption costs hours, not days. It is now a single bug (PySnooper/3): the wave's other original bug
was excluded rather than replaced.

**Calibration step (mandatory).** Before wave 2, replace this table with measured wave-1 timings.
The estimates above are not data.

### 8.2 Wave-1 gate checklist

Explicitly, wave 1 must demonstrate all of:

1. `vanta-bugsinpy-py3.8` passes `import ssl`, a real `pip install`, and `pytest --version` (§3.2).
2. The per-bug layers bake and `pytest --collect-only -q` succeeds inside them.
3. `prepare` derives a non-empty `FAIL_TO_PASS` for PySnooper/3 (the declared intersection of §1.3),
   with no flaky escapes. PySnooper/2 used to cover the `f2p_undeclared` case here, and it is now
   excluded — its BugsInPy `bug_patch.txt` is an incomplete developer fix whose oracle no patch can
   pass, so it was removed from the split before its slot was spent (see `reviewed_changes`).
4. The leak gate produces a clean text for it, and `issue_texts_all.md` reads as actionable.
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

### 9.4 Where the code goes (landed)

**Decided and implemented:** an **additive section** in [metrics.py](file:///Users/kyu-yi/vanta/scripts/metrics.py)
— new functions, `--pool-by-config`, grouping by the §4.4 provenance axes, and an `oracle` block that
consumes the harness manifest. Existing `classify()` semantics are untouched, so existing tests and
existing reports are unaffected. The section is self-contained and never re-derives a bucket: it reads
`Verdict.bucket` and `Verdict.metrics`, so the two halves cannot drift.

The shape, and the rule each piece is load-bearing for:

- **`provenance_of(summary)` / `session_provenance(data)`** — the §4.4 axes read from the `run.summary`
  line *itself*, never from an env var or the directory a session sits in, so a session copied out of
  the batch stays self-describing. An axis the line does not carry reads `unset`, which is a different
  value from any real one and therefore a different group.
- **`provenance_key(axes)`** — the group identity: the named axes in `PROVENANCE_KEYS` order, then the
  sampling keys **sorted**. The schedule is variable-length (one pair per §4.3 call site), so a fixed
  order is what makes two groups comparable and a group's identity independent of the order the line
  happened to be written in. A per-site `top_p_<site>` is provenance too — two runs sampling different
  call sites at different `top_p` are two experiments.
- **`wilson_interval(count, denominator)` / `wilson_text`** — §9.2's formula, local, `z = 1.96` a
  parameter so the arithmetic can be checked against a hand-computed interval. `None` (printed `n/a`)
  when the denominator is `0` — never `(0.0, 0.0)`, because "no row was in this denominator" and
  "every row failed" are opposite claims.
- **`PooledMetrics.rates()`** — §9.1's ratios as `(count, denominator)` pairs, each with the metric's
  **own conditional denominator** (approval_precision over the approved rows, spurious_reject_rate over
  the known-bad rejections, ECR/EIR over their transition rows). One definition of a denominator, so
  the printed rate and the tested rate cannot disagree.
- **`oracle_by_session` / `OracleFacts` / `load_manifest`** — the oracle facts come from the manifest's
  `oracle` block, keyed by session. A session with no manifest row is `n/a`, never a failed patch:
  `hidden_pass ÷ runs` would report an absent fact as a failure.
- **`render_pooled`** — one block per group, each naming its own axes, so no number can be read without
  the experiment it came from. Header carries the §9.3 naive-interval caveat.
- **CLI** — `--pool-by-config` (off by default: the existing report is byte-identical without it) and
  `--manifest`, which is loud when the path is not a file rather than silently printing `n/a`.
  [bugsinpy_run.py](file:///Users/kyu-yi/vanta/scripts/bugsinpy_run.py) `cmd_report` passes both,
  pointing `--manifest` at the batch's own `manifest.jsonl`.

Tests: [test_metrics.py](file:///Users/kyu-yi/vanta/scripts/test_metrics.py) gained a section covering
each of the above — including that an unset axis does not pool with a recorded one, that a different
seed or schedule is a different group, that a group without manifest rows reports `n/a` and not zero,
and that a false approval with no manifest row prints `?`.

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
    issue_texts_all.md, leak_gate/, oracle/, <session>/attempt_N_{patched,graded}_tree.diff +
    _tree.hashes.json
```

---

## 11. Build order

Reordered after review: `final.patch` is now **step 1**, because the oracle is untestable without it.

1. **`final.patch` for every terminal verdict** (§5.1). Small, testable in isolation: run a
   deliberately-rejected patch and assert `debug/final.patch` exists and applies cleanly.
2. Split file + loader + `prepare`: checkout, **SWE-bench split**, two-run flaky-flagged
   `FAIL_TO_PASS` / `PASS_TO_PASS` derivation, AST strip, relaxed fixed-tree gate, **context
   precheck** (§1.6) with `context_sizes.md`, leak gate, collectability gate, the **dependency gate**
   (§3.4: collection + import-from-the-tree, reported before oracle derivation), the **developer-fix
   self-check** (§8.2 item 5 made enforceable: grade the bug's own `bug_patch.txt` — test hunks
   dropped, exactly the bytes §7.1 grades as `dev` — against the oracle this stage just derived, and
   **refuse the bug** unless it is `patch_good`. It runs after the strip, so the regression comparison
   uses the same post-strip baseline `runner.grade_candidate` leaves implicit, and before the issue
   text and the authoritative context measurement, so a refused bug is never measured as a candidate.
   [VANTA-WHY] PySnooper/2 shipped a whole wave with `bug_patch.txt` grading `patch_good = false`
   because it touches only `pysnooper/tracer.py` while the fix commit also adds `pycompat.PY2` and
   `utils.get_shortish_repr(..., custom_repr=)` — the patched tree cannot import the package, so *no*
   patch of any kind could be graded good and every row measured the oracle. The symptom was
   indistinguishable from Vanta being conservative, which is why the check has to be a refusal and not
   a warning.) and the
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
| D8 | Test trees | **SWE-bench split** (§1.2–§1.3): Vanta's tree = buggy source + **fixed** test files − the declared `FAIL_TO_PASS` − the undeclared flippers (`f2p_undeclared`, recorded with a reason). `PASS_TO_PASS` is the native suite. Fixed-tree gate relaxed to "no fixed-only failures"; pre-existing failures recorded; every suite run twice to flag flaky nodeids. |
| D9 | Oracle & labels | `final.patch` built for **every** terminal verdict (§5.1). The oracle applies **only non-test hunks**; test files are always the fixed version's. `patch_good = hidden_pass ∧ suite_regressions == 0`, and the label table keys off `patch_good` (§5.3). |
| D10 | Waves & budget | Wave 1 = PySnooper/3 (py3.8, `[0]`) as a **go/no-go gate**; wave 2 = thefuck + httpie (py3.7, `[0]`); wave 3 = cookiecutter (py3.6, `[0]`). 8 bugs, 48–80 runs, ~12–22 h. Budgets in §8.1. |
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
- **Harness defects were found while gating wave 1 and were fixed before the reported runs; every
  reported result uses the fixed harness.** One was in how a candidate's reproduction tests are
  graded (the BATE path); the other two were in the Diagnostician's failure-location step, which read
  pytest's warnings summary as the failure site and demoted any failure whose innermost frame was not
  in a file the patch touched to a test defect — so the BugReproduction counterexamples that argued
  for rejection were discarded, and PySnooper/3's no-op was approved. A second defect in the same step
  compounded it: the patch-insensitivity probe demoted any failure that reproduced *identically* on
  the original tree, which is what a no-op patch produces by construction, so no no-op could be
  rejected even once its failure was located correctly. The probe is now gated on BATE's baseline — a
  validated test is never demoted, since for a `BugReproduction` test an identical failure on both
  trees is the evidence that the patch did not fix the bug — and BATE's baseline is what makes that
  boundary checkable rather than a matter of taste. Repairing a scorer mid-study is legitimate
  provided the before/after is disclosed and the affected rows are re-run on the fixed harness, which
  is what the wave-1 re-run and §11 step 9 are for; the rows a broken grader produced are not
  reported. None of the fixes touches the verdict rule, ECR/EIR or sampling. Recording the graders as
  defects is also the honest reading of the gate: finding them is what the gate is for (§8.2), and
  all three were found by running on BugsInPy rather than by inspection.
- **A false reject of PySnooper/3's developer fix (measured on the pre-fix run; to be re-measured on
  the fixed harness).** In the first wave-1 run the Attacker's round-4 reproduction test asserted
  `'Starting var' in <snoop file>` for a **zero-argument** function, but `tracer.py:170` emits
  `'Starting var:.. '` only at a `call` event **for variables already in scope**, so that string is
  never written on any tree. The assertion cannot distinguish a fixed tree from a buggy one. On the
  *fixed* harness this is a plain `AssertionError` in the generated test's own file, which is a
  verdict on the patch (§12.3's failure-location rule), so Vanta will **reject the upstream fix** —
  a false reject. It is recorded rather than tuned away: the Attacker and BATE were not modified for
  it. A false reject caused by the model's own test is an EIR-type error, not evidence about the
  fix, and it is reported as such, with the `tracer.py:170` evidence. The principled fix is a
  BATE-side gate (§13), to be developed on heretic/toy — never on a BugsInPy row, which would be
  tuning on the benchmark being measured.

---

## 13. Future work (not in this pilot)

- **Batch the intent audits** — now scheduled behind `VANTA_BATCH_AUDIT` (§14.5). In the first full
  wave-1 run `intent_audit` was **60 of 88** LLM calls (the rest: `creator` 6, `attacker_generate`
  19, `attacker_revise` 3), so auditing each generated test on its own dominates the cost of a
  round. Auditing a round's intents in one call — one prompt carrying every generated test and its
  claimed intent, one decision returned per test — keeps the per-test decision while cutting the
  call count. It stays off by default because it changes the auditor's call schedule, and the
  schedule is part of the run identity (§4.3): a re-batched auditor is a different run, not a
  faster one.
- **Real issue text from the upstream issue a bug cites.** Vanta's issue text is *derived from the fix
  commit* (§2.3), and for PySnooper/2 the derived text was `Support single tuple to , fix #144` plus a
  single generic sentence — too thin for the Attacker to write even one reproduction test, which is
  why that bug's no-op was approved with no BugReproduction evidence at all. (`fix #144` names a real
  GitHub issue.) Fetching the cited issue's own body — the issue, not the pull request that closed it
  — and passing it through the same leak gate (§2.2, rule A) would give Vanta independently written,
  realistic bug reports, which is the SWE-bench convention. It is future work rather than part of this
  pilot because it changes what the model is given: the pilot's issue text has to stay the derived one
  for its numbers to be comparable to the §8.2 gate.
- **A BATE gate on the reproduction test's symptom.** BATE checks that a `BugReproduction` test
  *fails* on the original tree, which accepts a test that fails for an unrelated reason — PySnooper/3's
  round-4 test asserted a string `tracer.py:170` never emits, so it failed everywhere and still
  matched the expected baseline (§12.3). Requiring a reproduction test to fail on the original tree
  **with the symptom the issue names** (the exception type / message the issue reports) would refuse
  that test instead of letting it reject a good fix. This is a change to BATE's gate, so it changes
  what the pilot measures and it is future work; it is to be developed and validated on heretic and
  the toy fixtures — **not** on a BugsInPy row, since tuning the pipeline on the benchmark it is
  measured against would invalidate the measurement.

---

## 14. Corroborated rejection, grounded assertions, batched audits, and the development split

§12.3 recorded that the first wave-1 run false-rejected PySnooper/3's developer fix. After the two
grader defects of §12.3 were fixed, wave 1 was re-run on the fixed harness
(`runs/bugsinpy/prepare-wave1-4/run_digest.md`), and the rows that matter are now:

```
baseline  review  false_approval=1 true_approval=1
vanta     review  false_reject=1  true_reject=1
```

The no-op is rejected for the right reason — its `BugReproduction` test fails with the `NameError`
the issue names — and the developer fix is **still** rejected, by one Attacker-authored test that
asserts `'Starting var' in <snoop file>`, a string `tracer.py:170` never emits on any tree, so the
assertion cannot distinguish a fixed tree from a buggy one. That is the case this section exists
for. No prompt can guarantee an LLM writes a correct test; what can be made robust is the verdict
built on top of it.

The idea is **corroboration**: a rejection is a claim about the *patch*, so it needs evidence that
survives the possibility that the test itself is wrong. A single uncorroborated failure stops being
a rejection and becomes "a human should look at this".

### 14.1 Shape of the change

Three env flags, **all default off**, so every run reported above — and every evaluation wave —
keeps v0.2's behaviour until they are flipped:

| Flag | Stage | Effect |
|---|---|---|
| `VANTA_CORROBORATE=on` | review mode only | a rejection requires a rule from §14.2; otherwise a failing-but-uncorroborated candidate ends in the new `NeedsReview` (§14.3) |
| `VANTA_GROUNDED_AUDIT=on` | intent auditor | the auditor also asks whether a test's assertions are grounded in the issue text (§14.4) |
| `VANTA_BATCH_AUDIT=on` | intent auditor | one audit call per round instead of one call per test (§14.5) |

Nothing in ECR/EIR, the Gatekeeper, the sampling schedule (§4.3) or the exit-code meanings of 0/1/2
changes. The three flags are recorded in `run.summary` (§14.3) because a flagged run is a different
run and the rows have to be attributable.

### 14.2 What corroborates a rejection

In **review** mode a candidate is `Rejected` only if at least one rule holds. Every rule is a pure
function of evidence the run already has (`TestEvidence`: `test_name`, `outcome`, `failure_class`,
`intent`, the Diagnostician's `structured_payload`), so it is unit-testable without an LLM, next to
`route_intent`:

| Rule | Evidence | Why it is strong |
|---|---|---|
| **(a)** native-suite regression | the native suite fails on the patched tree where it passed on the original | the project's own tests, not a Vanta-authored one |
| **(b)** symptom match | a BATE-validated `BugReproduction` failure on the patched tree whose `exception_type` is one the **issue text names** | the issue's own symptom, reproduced; the failure discriminates *and* says the right thing |
| **(c)** two distinct corroborating tests | ≥2 BATE-validated generated tests that fail as `Logic` on the patched tree, distinct by (attempt, test_name), counting different rounds and different intents | one wrong test cannot manufacture two independent ones |
| **(d)** a validated `BugReproduction`/`FeatureAcceptance` failure | one BATE-validated BR/FA test failing as `Logic`, **provided the auditor did not answer `issue_requires_change = true` for it** | a BR/FA test passes on the original and fails on the patch, so the patch is what moved it — the opposite of a wrong test |
| **(e)** a new static finding | ruff/mypy findings in a file the patch touched (review mode only) | deterministic tool output, not model-authored — see the deviation note below |

**Rule (b) is mechanical, not semantic.** The issue's named symptoms are the exception-class tokens
in the issue text, extracted with the harness's own expression
(`\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b`, §2.2's `failure_kinds_from_report`), and
compared for equality with the failing evidence's `exception_type`. Vanta's issue text already
carries them: the behavioural line is built from the failure kind, so PySnooper/3's text says
`(NameError)`. If the issue names no such token, rule (b) **cannot** fire — an issue with no named
symptom cannot corroborate an exception, which is the honest direction.

**The (d) caveat is a downgrade, not a second gate.** A `BehaviorPreservation` probe that asserts
the behaviour the issue says must *change* is broken by a correct fix, so its failure is not
evidence against the patch. The intent auditor already answers that question
(`issue_requires_change`), so rule (d) reads the auditor's answer: `true` downgrades that failure to
weak. This is the toy fixture's shape — its bad patch breaks exactly one probe,
`test_shipping_band_500_stays_standard`, whose audit says
`{"issue_asks_for_behaviour": false, "issue_requires_change": false, "reason": "…the original
already returns 'standard' for 500, so no change is required at this input."}` — so (d) fires there
and the toy still rejects bad.

**A weak failure** is any `Logic` failure no rule covers: a validated BP/FA failure the auditor
flagged `issue_requires_change = true`, or any other non-infrastructure failure. `Infrastructure`
is never a finding (unchanged), so an all-infrastructure round stays `Inconclusive`.

**Deviation to confirm:** rule (e) is not in the enumeration this section was written from (which
named a/b/c/d). It is kept because corroboration is about *model-authored* evidence — a ruff/mypy
finding in a line the patch touched is deterministic — and dropping it would turn today's
static-only rejections into `NeedsReview`, i.e. quietly loosen a signal nobody asked to loosen.
Say the word and static findings move into the weak bucket instead.

### 14.3 `NeedsReview`

New `GovernorDecision::NeedsReview`, reachable only in review mode with `VANTA_CORROBORATE=on`:

- it is **not** an approval. The Gatekeeper still approves only on `AllTestsPassed`, so nothing in
  this path can ever produce an approval. `metrics.py`'s approval buckets cannot be inflated by it;
- it is a **terminal** decision, taken when the attempt budget is exhausted, no rule fired, and at
  least one weak failure exists. While attempts remain the run keeps attacking (so rule (c)'s
  "different rounds" is reachable rather than tautological);
- it gets **exit code 3** in `vanta_m3` (`0` approved, `1` any other negative verdict, `2`
  infrastructure failure). A harness driving the binary must be able to tell "a human should look"
  from "no";
- `run.summary` gains, in the existing key=value form:

```
corroborate=on|off grounded_audit=on|off batch_audit=on|off
deciding_rule=<a|b|c|d|e|none> deciding_evidence=<test_name>@<attempt>[,<test_name>@<attempt>…]
needs_review_weak=<test_name>=<FailureClass>/<Intent>[@<attempt>][,…]
```

  `deciding_rule` is the first rule that fired in a/b/c/d/e order and `deciding_evidence` the
  evidence that fired it, so a rejected row says *why*; `needs_review_weak` lists the weak failures
  that triggered a `NeedsReview`. Both are empty/`none` when the flag is off, which keeps every
  pre-existing row byte-comparable.

**Repair mode is unchanged, deliberately.** Should the same rule apply to feedback? In repair the
corroboration rule belongs on the *message*, not on the decision: telling the Creator "your patch is
wrong because a single uncorroborated test failed" is the repair-mode version of the same mistake,
and the honest fix would be to qualify that feedback. It is not implemented here — repair's
feedback text is the Attacker's prompt surface, and this section is not allowed to touch it. It is
recorded as the next step if repair ever false-rejects a good patch on one bad test.

### 14.4 `VANTA_GROUNDED_AUDIT` — the auditor's third question

The auditor already reads the issue, the original source and one test, patch-blind. With the flag on
it answers one more question, and a `BugReproduction` test whose assertions are **not** grounded in
the issue text is scrapped exactly as a BATE mismatch is: dropped from the round, logged with
`scrapped = true` and its reason, and — when no test survives — the Attacker is asked to revise.
This is the change that would have stopped PySnooper/3's `'Starting var' in content` test at the
audit instead of letting it reject the fix.

The change is a *variant* of the existing prompt, not a replacement: with the flag off the system
prompt, the user prompt and the request body are byte-identical to today, because prompt identity is
part of run identity (`intent_audit` is one of the §4.3 call sites).

System prompt, appended after the existing rules (flag on):

```
3. "assertions_grounded": does this test assert ONLY behaviour the ISSUE describes, at the input the ISSUE names — or does it also pin a SPECIFIC VALUE the issue never states?

Rules for question 3:
- Grounded: an assertion that follows from the issue's own words about the input it names ("it should not raise", "the file is written", "the boundary value returns the standard band").
- Ungrounded: an assertion that pins wording, an exact output string, an exact message, a format or a number the issue never mentions ("the output CONTAINS 'Starting var'", "the repr is exactly ...", "the log line begins with a timestamp").
- The issue is the only source of requirements. "It should not raise NameError" is grounded; "it should print exactly this string" is grounded only if the issue prints that string.
- A test may assert more than the issue asks about ONLY when the extra assertion is a direct consequence of what the issue requires. Invented specifics are ungrounded.
- Answer this for every test you are given, whatever its declared intent.
```

The JSON contract becomes:

```
{"issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "assertions_grounded": true|false, "reason": "one short sentence"}
```

and the typed answer gains `assertions_grounded: Option<bool>` (`None` = the flag was off, or the
model omitted it). The user prompt's closing question and the re-prompt line are extended the same
way. Routing: `assertions_grounded == Some(false)` on a `BugReproduction` test maps to a new
`IntentRouting::ScrapUngrounded`, checked **before** the existing arms; every other intent keeps its
current routing, and a missing field can never scrap anything. Scrapping a test can only ever remove
a rejection signal, never create one.

**Over-scrapping is the risk, so every ungrounded scrap is logged with the auditor's own reason** —
the `intent.check[…]` record carries `assertions_grounded=` next to the other two answers and the
full reason, and the driver prints one `⚖ UNGROUNDED: <test> scrapped — <reason>` line per scrap at
both audit call sites (`attacker.generate`, `attacker.revise`). The failure mode being guarded is a
legitimate reproduction test that happens to name a specific the issue states in prose: it is
scrapped, the bug then has no probe of its own, and without a per-scrap line on the console that
would only be visible by grepping a batch log thousands of lines long. Whether the flag over-scraps
is therefore a measurement taken on the dev split (§14.7), not an assumption.

### 14.5 `VANTA_BATCH_AUDIT` — one audit call per round

`intent_audit` was **60 of 88** LLM calls in the first wave-1 run. With the flag on, a round's tests
are audited in **one** call: one prompt carrying the issue, the original source and the round's
tests, each numbered, whose structured answer carries one entry per test with the same per-test
schema as §14.4:

```
{"tests": [
  {"index": 1, "issue_asks_for_behaviour": true|false, "issue_requires_change": true|false,
   "assertions_grounded": true|false, "reason": "one short sentence"},
  …
]}
```

Contract:

- `index` is the 1-based number the prompt printed for the test. Every index must appear exactly
  once; an unknown index, a duplicate or a missing one is a **parse failure of the whole batch
  response** (retried like any other malformed structured output);
- if the call finally fails, every test in the round falls back to the single-call path's failure
  behaviour — `issue_asks_for_behaviour = false`, `issue_requires_change = None`, audit record with
  the error in its reason. So a broken auditor still cannot manufacture a rejection;
- the per-test decision, the relabel, the scrap and the `IntentCheckRecord` are computed by the same
  `route_intent` and the same record construction as the unbatched path; only the transport changes.
  A test asserts that the batched and unbatched paths produce the same per-test decisions and the
  same record shape on the same input;
- round order is preserved (the records come out in the order the tests went in), so the session log
  is a permutation-free view of the unbatched one;
- the batch's answer is **one entry per test**, so its output cap is **2048 tokens per test in the
  round** (`batch_audit_tokens_per_test`), clamped to a **16384** ceiling — half the 32768-token
  window Vanta's local models are served at (§9), and eight tests' worth of the per-test allowance.
  The 2048 the single-test call pins is *one entry's* worth and truncates a multi-test round
  mid-object, which is a lost round for no reason. The cap is part of a run's identity alongside the
  schedule, so both numbers are recorded in `run.summary`: `batch_audit_tokens_per_test=` and
  `batch_audit_max_tokens=` (the largest cap any round of the run was given; `unset` when the flag is
  off);
- a reply the server itself reports as **cut off at the cap** (`finish_reason: "length"`) falls back
  to **one audit call per test for that round**, after **one** batch call. The truncation is a fact
  about the *request* — N answers in one capped generation — and `max_tokens` is pinned per call site,
  so the same batch is cut off at the same point every time: the retries are deterministic, and on
  local hardware each one is a full-length generation (up to the 16384 ceiling) spent to reach the
  identical prefix. So a cut-off answer is **not retried at all** — it fails the call immediately —
  and the per-test path takes over on the first attempt, asking the same questions one at a time,
  each well inside its own 2048. The round keeps its verdicts instead of losing every one of them to
  "audit unavailable";
- **that fail-fast is scoped to calls pinned at `temperature = 0.0`, which is the auditor and no one
  else** — batched and per-test, both 0.0. The argument above is an argument about a *deterministic*
  call: same request, same seed, same cut-off. It does not hold for the creator (0.1) or the
  attacker's three sites (0.2, 0.3), where a generator that spent its whole 32768-token window on
  reasoning and emitted no content is a **sampling accident**, not a property of the request. So
  above 0.0 a cut-off answer **keeps its retries within the existing budget**; failing the call on
  the first attempt instead turned a re-runnable run into an `InfraFailure` row, which the wave gate
  counts as a failure. The retry loop reads the call's pinned temperature out of its own extras —
  where the schedule put it — and a caller that sends no `temperature` key at all (the bootstrap) is
  read as 0.2, i.e. not deterministic, and keeps its retries too;
- **on those calls a retry also moves the seed.** The harness pins `seeds: [0]` (§13), and a pinned
  seed with an identical prompt reproduces the identical prefix, so at 0.2 the retry would be spent
  re-reaching the same cut-off. Each attempt therefore sends `seed + (attempt - 1)`: attempt 1 sends
  the configured seed byte-for-byte, so a one-attempt run is unchanged, and attempt 2 asks a question
  the server can answer differently while the run stays reproducible. The seed an attempt actually
  sent is recorded — every `call.usage` row carries `seed=<value|unset>`;
- **truncations are counted per role in `run.summary`**: `truncated_calls_creator=`,
  `truncated_calls_attacker=` and `truncated_calls_intent_audit=` (with the same fact on each
  `call.usage` row as `truncated=`), because "how often a model runs away on the attacker" is a
  number the paper reports for the dev split (§14.7), not an observation to reconstruct from a log;
- a **malformed** answer that was *not* cut off keeps its retries: the model can answer that one
  differently, which is what the retry budget is for. Because a cut-off batch now makes one call and
  a malformed one makes the whole budget, the call count no longer says which happened, so the
  session log says it outright: one `intent.audit_batch` record per round, written **only** when a
  batched call was made, with `transport=` one of `batched`, `truncated_fallback`,
  `malformed_retried`, `unavailable` (the batched call never produced an answer at all — a transport
  error, or the deterministic context gate);
- the flag is **off** by default precisely because the call schedule is part of a run's identity
  (§13), so a re-batched auditor is a different run, not a faster one.

### 14.6 Metrics and labels for `NeedsReview`

The harness has to be able to score the new outcome without inventing an approval or a rejection:

- `runner.py`'s `label_for`: `NeedsReview` with `apply_ok = false` → `unobservable` (unchanged
  rule), otherwise `needs_review_on_good` when the oracle says the candidate was `patch_good`, and
  `needs_review_on_bad` otherwise. The oracle is derived in `prepare`, so a review row always has
  its label;
- `runner.py`'s `wave_gate_failures` must **not** count a `NeedsReview` row as a failure: it is a
  refusal to decide, not a wrong decision, and a wave gate that failed on it would stop a wave for
  behaving exactly as designed;
- `metrics.py` gains two buckets, `needs_review_on_good` and `needs_review_on_bad`, plus
  `needs_review_unlabeled` for the case where no label was supplied, so the outcome is never folded
  into an approval, a rejection, `unresolved` or `undetermined`. The wave digest prints them next to
  the approval/rejection counts.

### 14.7 The development split

Items 1–3 are developed on bugs that are **not** in any evaluation wave, so no evaluation number is
produced by a pipeline tuned on the bug it is measuring. The selection rule, recorded here and in
`plans/bugsinpy_dev_split.json` **before** any Vanta result is looked at:

1. **Pool.** BugsInPy bugs whose `python_version` is 3.8 — the version the py3.8 checkpoint actually
   is — in projects that appear in **no** wave of `plans/bugsinpy_split.json`, i.e. not
   `PySnooper`, `thefuck`, `httpie`, `cookiecutter`, `black`, `tqdm`, `youtube-dl`. Measured now:
   `fastapi` 16, `luigi` 33, `matplotlib` 30, `pandas` 169, `sanic` 5, `scrapy` 40, `spacy` 1 — 294 bugs.
2. **Order — round-robin across projects, not down one project.** The projects are taken in the
   clone's own order: `runs/bugsinpy/bugsinpy/projects/` in directory order, keeping only those with
   a py3.8 bug, which gives `fastapi`, `luigi`, `matplotlib`, `pandas`, `sanic`, `scrapy`, `spacy`
   (`ansible`, `keras` and `tornado` have no py3.8 bug; `PySnooper` and `black` are evaluation
   projects). **One pass takes each project's next py3.8 bug in ascending id** —
   `fastapi/1`, `luigi/1`, `matplotlib/1`, `pandas/1`, `sanic/1`, `scrapy/1`, `spacy/4` — and passes
   repeat until the list is full. Flat directory order was rejected on inspection: it makes the
   first ten candidates ten `fastapi` bugs, so the dev set would have been a single project and every
   item would have been developed against one codebase's idioms.
3. **Take the first bug in each project that passes both gates, until 5.** The gates are the existing
   ones: (i) the §1.6 context gate and (ii) the §8.2 dev-fix self-check (`bug_patch.txt`, test hunks
   dropped, grading `patch_good` against the oracle `prepare` just derived). Neither is loosened and
   no cap is raised. **If a project has no bug that passes, it is skipped and the round-robin
   continues** with the next project in order.
4. **If fewer than 5 pass** after the authored list is exhausted, the list extends mechanically to the
   next candidates in the same round-robin order.

The file is `plans/bugsinpy_dev_split.json`: the same closed schema, a separate file, so the
evaluation split is not rewritten (its `reviewed_changes` carries this rule and the per-candidate
gate measurements, exactly as §8 records a substitution). The first **10** candidates in round-robin
order are authored as entries — the first five as `pilot`, the rest as `reserve`. The pilots are one
bug per project, so the authored pass already satisfies "one per project" before any gate is run; if a
pilot fails a gate, the next reserve in file order is promoted (change its `role`, re-run `prepare`,
§8), and that reserve is another project's first bug, so the dev set stays five distinct projects.
The tenth entry buys the fallback depth: authoring 10 rather than exactly 5 is the one judgement here,
because the gates can only be evaluated by running them and a shorter list risks a second prepare
round. Each entry's `needs_build` is not authored but derived from the buggy commit's own tree by
`bake.needs_build` (§3.2) as the file is written — measured, that flags `matplotlib/1`, `matplotlib/2`,
`pandas/1` and `spacy/4`, whose tests import a package the tree has to compile — and `prepare` re-checks
the written file against the tree both ways.

`PySnooper/3` also joins the dev set, labelled **"seen during development"**: items 1 and 2 were
designed from its false reject, so it is development data now. It is entered in the dev split as an
extra `pilot` carrying that label, and it is never reported as an evaluation result — the evaluation
split records the exclusion in its own `reviewed_changes`, so a reader of either file sees the same
decision. The paper reports it as a development case.

### 14.8 Tests

One per rule and per new parse, plus the regression that matters most:

- rules (a), (b), (c), (d) and (e) each fire on their own evidence — including (b) *not* firing when
  the issue names no symptom, and (b) not firing on a mismatch
  (`NameError` in the issue vs `AssertionError` in the evidence, which is the PySnooper/3 shape);
- the (d) downgrade: a validated BP failure with `issue_requires_change = true` is weak, and the same
  failure with `false` is strong (the toy's two shapes);
- `NeedsReview` is terminal, is not an approval, and carries exit code 3; with the flag off none of
  the above can be reached;
- symptom extraction and matching from issue text; `run.summary`'s `deciding_rule` /
  `needs_review_weak` fields;
- the grounded-audit parse (`assertions_grounded` present, absent, and false) and the
  `ScrapUngrounded` routing for a **`BugReproduction`** test, including that a missing field never
  scraps and that the ungrounded answer is decided **before** the relabel arms (a demote-shaped
  verdict must still come out `ScrapUngrounded`);
- the mirror: an ungrounded answer must never scrap a `BehaviorPreservation` or a
  `FeatureAcceptance` test — no combination of the other two answers may reach `ScrapUngrounded` —
  and their routings are otherwise unchanged;
- a grounded prompt keeps every shared rule of the ungrounded one, and the flag-off prompt does not
  mention the third question at all;
- the batched-audit schema: batched vs unbatched give the same per-test decisions and records; a
  missing/duplicate index fails the batch; a failed batch falls back to the unbatched failure shape;
- the batched audit's **output cap** — one test's allowance per test in the round, clamped to the
  ceiling, recorded in `run.summary` — and the **truncation fallback**: a batch the server reports as
  cut off at that cap (`finish_reason: "length"`) is re-audited one test per call after **one** batch
  call, not after the retry budget, and that round's verdicts and records are the unbatched run's,
  with no record saying the audit was unavailable. Its mirror is the malformed batch, which is
  asserted to **still** spend the whole retry budget, and each of the two is asserted to write its own
  `transport=` value in the `intent.audit_batch` record;
- the fail-fast's **scope**, on the make-or-break case: an **attacker** answer the server reports as
  cut off is **retried** and the round succeeds on attempt 2, run twice — once with `VANTA_LLM_SEED`
  unset (no `seed` key in either body) and once with it set, where attempt 1 sends the pinned seed and
  attempt 2 sends `seed + 1`, the `call.usage` rows for `attacker_generate` record the seed each
  attempt actually sent (and `truncated=true` then `false`), and `run.summary` reads
  `truncated_calls_attacker=1`. Its mirror is the **per-test auditor**, whose cut-off answer still
  fails the call on the first attempt — one `intent_audit` request, not the whole budget — with
  `truncated_calls_intent_audit=1`. And with nothing truncated and the flags off, every request body
  stays byte-identical: the pinned seed on every call, `seed + 1` on none, all three counters 0;
- **the Attacker generate prompt no longer forbids the test that catches a bad patch** (§14.10): the
  both-trees preservation wording is gone, and every other rule the prompt carried is still in it —
  the same test that pins the hygiene rules (ONE INTENT PER TEST, TEST FIXTURES, PRESERVATION
  SPECIFICITY, the fake-attribute rule) must keep passing;
- **`VANTA_SAMPLING`** (§14.11): a `coding` run sends the coding sampler pins on the Creator, the
  three Attacker generative sites **and the intent auditor** (which keeps its 2048 cap), a default run's
  bodies are unchanged, and `run.summary` records the preset and the effective per-role values — the
  recorded schedule is the effective one, which the request-body comparison above already enforces;
- `metrics.py`'s two new buckets and `label_for`'s new branch, and that the wave gate ignores a
  `NeedsReview` row;
- `plans/bugsinpy_dev_split.json` loads under `load_split`; the only bug it shares with the
  evaluation split is `PySnooper/3`, and that entry says "seen during development" — every other dev
  project is in no evaluation wave;
- **the toy fixture, with all three flags on, still rejects the bad patch and approves the good one**
  (bad via rule (d), 6/6; good 3/3), which is the end-to-end guard that the corroboration rule did
  not go soft.

### 14.9 Commands

Build, test and the two comparisons (the user runs everything that boots Shuru or the GPU):

```bash
cargo build --release
cargo test --lib -- --skip harness
zsh fixtures/toy_calc/run_matrix.sh all --repeat 3            # flags OFF
VANTA_CORROBORATE=on VANTA_GROUNDED_AUDIT=on VANTA_BATCH_AUDIT=on \
  zsh fixtures/toy_calc/run_matrix.sh all --repeat 3          # flags ON

python3 scripts/bugsinpy_run.py bake    --split plans/bugsinpy_dev_split.json --pythons 3.8 --yes
python3 scripts/bugsinpy_run.py prepare --split plans/bugsinpy_dev_split.json
python3 scripts/bugsinpy_run.py run     --split plans/bugsinpy_dev_split.json --batch runs/bugsinpy/<dev-batch>
```

with `prepare` + `run` repeated once with the three flags in the environment of the `vanta_m3`
child. The evaluation waves are not touched: no `plans/bugsinpy_split.json` entry, no wave-1 batch
and no evaluation number changes.

### 14.10 The Attacker prompt's preservation contradiction

**The runaway had a cause, and the cause was the prompt.** Ornith-1.0's Attacker truncations (§15)
were read as a sampling-pin problem until the raw reply of the capped call was read:
`vanta_sessions/<id>/debug/AttackerResponse_round_1_attempt_1_raw_response.txt` is a **repetition
loop** — the same paragraphs emitted on the order of seventy times — not a model that had run out of
things to say. The loop is a **contradiction inside `ATTACKER_SYSTEM_PROMPT`**
(`src/agents/attacker.rs`), and the model's own reply quotes both halves of it:

- the intent rule said a `BehaviorPreservation` test *"MUST PASS on BOTH original and patched code"*;
- the BOUNDARY-PROBING RULE orders a `BehaviorPreservation` probe at the exact boundary the patch
  changes.

On a **bad** patch the two cannot both hold: the preservation probe at the boundary the patch broke
is precisely the test that catches it, and it fails on the patched code by definition. Read
literally, the prompt forbids writing the test that catches a bad patch — so the model, unable to
satisfy both rules, repeated itself until the 32768-token cap.

**The prompt is wrong, not the model.** BATE checks a preservation test on the **ORIGINAL** tree
only (`pipeline/bate.rs` — a `BehaviorPreservation` test is expected to pass on the baseline), and
corroboration rule (d) treats a validated preservation failure on the **PATCHED** tree as evidence
of a regression. "MUST PASS on BOTH" was never Vanta's semantics.

**Fix — always on, a bug fix, not a flag.** The generate prompt's preservation rule now reads: a
`BehaviorPreservation` test asserts behaviour the issue does **not** ask to change; it MUST pass on
the ORIGINAL code (BATE checks this); it is EXPECTED to pass on a correct patch, but if it fails on
the PATCHED code that failure **is** the finding; never drop a boundary probe because you expect the
patch to break it. The revise prompt already said "must PASS on original code" and the simplify
prompt does not repeat the rule. `ATTACKER_PROMPT_VERSION` is bumped so a session states which side
of the fix it ran.

**Correction (prompt audit, 2026-10-21).** An earlier revision of this section claimed that "the
contradiction was confined to the generate prompt" and that "every other rule is kept". Both claims
are wrong, and the run logs say so. What the audit found:

- **The "MUST" shape was not confined to the preservation rule.** The same requirement-shaped wording
  carried the umbrella bullet ("each test MUST produce a DIFFERENT result") and the BugReproduction
  and FeatureAcceptance bullets ("MUST FAIL on ORIGINAL and PASS on PATCHED"). A BugReproduction
  probe that reproduces the issue's own symptom — the probe §9's corroboration rule needs — fails on
  the patched code by definition, so the same reading that jammed the preservation rule jams these.
  `A2` generalised `9a8356a`'s qualifier to them.
- **It was not confined to the generate prompt.** The revise prompt licensed a relabel instead of a
  repair (`A4`); the simplify prompt never said the assertion had to survive, and the Diagnostician
  then read *any* passing simplified test as "Unknown resolved" (`A10`); the auditor's four prompts
  each carried an ambiguity the models quoted back at it (`A7`–`A9`). One rule was missing entirely
  (`A11`: nothing warned that the baseline is measured more than once) and two lines were another
  project's (`A17`: `PYTHONPATH=src` and the `heretic.system` import example).
- **Rules were not all kept.** `A13` and `A14` rewrote two rules, and `A18` deleted one outright (the
  reasoning word count).

#### 14.10.1 The audit — one row per finding

Method: every rule in every system prompt was listed; each was checked against every other rule under
each case it can meet, and against what the code actually does; and each rule's key phrase was traced
with `git log -S`. `A15` was **rejected** — allowing zero tests on a no-op would let the model declare
"no behaviour change" and skip testing the bug, and the no-op true reject is the core result. `A16` is
**parked**. "introduced" and "fixed" are the `git log -S` hits for the rule's own phrase, so the
history is reproducible from the references.

| # | rule / where it lives | the case where it breaks | introduced | fixed |
|---|---|---|---|---|
| A2 | `ATTACKER_SYSTEM_PROMPT`, the "each test MUST produce a DIFFERENT result" umbrella and the BR/FA "MUST FAIL on ORIGINAL and PASS on PATCHED" bullets | a bad patch: the BR probe that reproduces the issue's symptom fails on the patched tree, which the bullet forbids | `832d60a` | `9a8356a` + `0457ac3` |
| A3 | generate prompt: "Write exactly 3 tests. Label each one honestly"; follow-up: the "THIS order of priority" line | no stated order, so the three slots can be spent on duplicates, and nothing says what was left uncovered; and the three priority items read as exclusive (a slot re-covering item (1) as item (3)) with no default for a leftover slot, so the model hunts for a third thing to test | `832d60a` | `182aafa` + `3588728` |
| A4 | `ATTACKER_REVISE_SYSTEM_PROMPT`: "If you cannot make the test match its intent, change the intent to match the test's actual behavior" | a test whose input or expected value is wrong: the label moves to whatever the test does, with no requirement that the issue support it | `832d60a` | `d75964f` |
| A5 | the `issue_quote` MUSTs: "that sentence MUST name the same input the test asserts", and the label being allowed "ONLY if the quoted issue sentence explicitly asks for that behaviour" | any probe whose input the issue does not spell out; and a quote lifted from a "must not change" clause — "…without changing the existing behavior at the other thresholds." — used as the licence for the change it forbids | `1c44e64`, `9f31ebe` | `fe1d976` |
| A6 | generate prompt: "Use assertions (assert X == Y) so failures are clean, not crashes" | an issue that names its own symptom (a `NameError`, an exact message): an assertion on a neighbouring value raises a bare `AssertionError`, so corroboration rule (b) matches nothing | `832d60a` | `29a53bb` |
| A7 | the two grounded auditor prompts: the closing "answer for every test, whatever its intent" | it names no question and no test, so a two-test batch reads as "answer question 3 for both" | — | `47e52e8` |
| A8 | all four auditor prompts: the heading "Answer two questions about the input the test asserts" | the grounded variant asks three questions and the batch variant asks about each test, so the heading promises a question the auditor may not ask | `28d4367`, `b715e1d` | `f79380e` |
| A9 | both batch prompts: the rule closing on "the input THIS test asserts" | a batch of N tests: "THIS test" has no referent, so the issue-wide reading demotes probes it should keep | `28d4367`, `b715e1d`, `5f8af2f` | `0a590a0` |
| A10 | the simplify prompt ("Use literal values. Keep the same intent.") and the Diagnostician's `Passed` branch in `pipeline/diagnostician.rs` | a simplification that deletes the failing assertion (`assert add(2, 3) == 5` → `add(2, 3)`) passes on the patched tree, and the Unknown is recorded as resolved with no check that the test still asserts anything | `832d60a` | `450fb1c` + `d4ea314` |
| A11 | generate prompt: no rule at all about the baseline being measured more than once | a clock/seed/network/ordering-dependent test, written in good faith, is scrapped for flakiness with no rule it had broken | (missing) | `5da097c` |
| A12 | generate prompt: the INTENT-vs-API rule, which stopped at the FeatureAcceptance redirect | a probe using a symbol the patch ADDS when the issue asks for no new behaviour either: every remaining label is dishonest and the audit scraps it anyway | `832d60a` | `5399e95` |
| A13 | generate prompt: "targets the EXACT behaviour … must NOT pin the INTERNAL calls of a function the issue says to change" | read as an absolute ban on pinning internals, so a correct patch that legitimately calls an internal with the same arguments is failed | `d646d40`, `2eb5c36` | `8fb8270` |
| A14 | generate prompt: the IMPORT RULES and INTENT-vs-API rules, written for a whole-repo read | review mode's narrowed context: "the ORIGINAL source" is an excerpt, so a symbol absent from it is read as evidence that the patch adds it | `832d60a` | `7e6a067` |
| A17 | generate prompt: "Tests run with PYTHONPATH=src so `from heretic.modifier import X` works", and the `import heretic.system as sysmod` example | every non-Heretic target: BugsInPy sets no PYTHONPATH, has no `src/` layout and is not heretic, so the prompt describes an environment this harness never builds | `832d60a` | `b05e7e8` |
| A18 | generate prompt, line 1: "Keep reasoning SHORT (under 100 words)" | a thinking model: its reasoning is emitted before the answer, against a budget the prompt does not set (Ornith reasoned ~30k tokens), so the rule can only be read as a limit on the answer | `832d60a` | `d8d6356` |
| A19a | the four auditor prompts: question 1 on a `BehaviorPreservation` test — nothing said whether a "without changing X" clause counts as the issue ASKING for the original result at an input it does not ask to change | every preservation test (2 of the 3 in most rounds): a 17-word question re-argued instead of answered — "does the ISSUE ask for the result that test asserts, AT THE INPUT THE TEST ASSERTS?" 17×, "Preserving existing behaviour is always a valid answer" 23×, ") FORBIDS the change; it never authorises it" 18× | `28d4367` | `961e42f` |
| A19b | the two grounded auditor prompts: question 3 (assertion grounding) is asked for every intent, though `route_intent` reads the answer only for a `BugReproduction` test | every `BehaviorPreservation`/`FeatureAcceptance` test: grounding reasoning the pipeline discards (A7 — only `BugReproduction + Some(false)` scraps) | `47e52e8` | `8e09513` |
| A19c | all four auditor prompts: no rule scoped a "must not change" / "without changing X" clause | the issue's clause quoted 168× (pre-fix) / 57× (post-fix) while the model enumerates which inputs are "thresholds" (is 0 one? 999?), and "at the other thresholds" 180× | `28d4367` | `d75a60c` |
| A19d | `INTENT_AUDIT_BATCH_TOKENS_PER_TEST` / `INTENT_AUDIT_BATCH_TOKENS_CEILING` (2048 / 16384) and the auditor's schedule cap | an audit that legitimately exceeds the cap: 21 truncations across 8 of 12 runs, the largest ending mid-reasoning ("Let me reconsider grounded once more…") with no JSON | `f10858d` | `698933e` |
| A19e | `write_forensics_to`: the type name in the filename is not sanitised | the batched audit writes `IntentVerdict>_intent_audit_batch_round_1_attempt_1_raw_response.txt` — the `>` is `Vec<IntentVerdict>`'s closing bracket, and shell-hostile | `5f8af2f` | `60e2c08` |
| A19f | prompt provenance: the version label had no content hash beside it | `3588728` (A3 follow-up) and `d4ea314` (A10 follow-up) reworded prompts after the `a2617b3` bump without bumping, so one tag covers two prompt texts and the log cannot tell them apart | `a2617b3` | `6c337eb` |
| A21 | generate prompt: the A3 leftover-slot default — "an input where the ORIGINAL source returns a DIFFERENT result than at the inputs you have already tested" — has no terminal case; **and** the round>=2 user-prompt section said `DO NOT REPEAT ANY OF THESE` while the new terminal fallback says `A repeat of an existing test is an ACCEPTABLE answer here` — a rule pair pulling opposite ways in exactly the late-round state that looped | a late round where every meaningful input is already in the shared set: round 3 on the toy looped **45×** on `"Let me pick the three most valuable tests"` / `"Both are shared"` / `"-1 is shared"`, hunting for a novel input that does not exist, instead of spending the slots and finishing | `3588728` | `70fdc50` + `745cafb` + `2bd4c50` + `5526621` |

**Evidence — the same contradictions, quoted by the models, across models, binaries and attempts.**
A grep over `~/vanta-runs`, `runs/bugsinpy` and `vanta_sessions` for the rules' own words returns:

| phrase the model quoted back | hits |
|---|---|
| `violates the intent rule` | 83 |
| `violates the "MUST PASS on BOTH" rule` | 72 |
| `instructions are poorly written` | 73 |
| `instructions are contradictory` | 71 |
| `violates the BehaviorPreservation rule` | 15 |
| loop-drift variants in other languages (`contradictoire`, `contradictorio`, `contradictoriu`, …) | ~100 |

At least 19 distinct raw-reply files contain them (the listing was cut at 20). They span runs at
11:19, 11:31, 12:19, 13:17, 13:53, 15:36, 16:17 and 17:06 on 2026-10-21 — the model screen included —
across the `vanta_m2` and `vanta_m3` binaries, in first attempts, retries and round 2. So this is not
one model misreading one prompt: it is the shape of the rules.

**A3 follow-up — the two loops the priority line left open (`3588728`).** After the audit commits, an
Ornith-1.5 run (default sampling, toy, 3 runs, **all three rejected**, **zero contradiction phrases** —
so the loops below are not the A2 contradiction; they are what the priority line does when it is obeyed)
was read back from its five `AttackerResponse_*_raw_response.txt` files (49.8 KB–118.5 KB each). Two
loops:

| loop | evidence (raw reply) | count |
|---|---|---|
| priority overlap, read as exclusive | `"(1) the exact comparison the patch changes, at its boundary value"` repeated in one reply | 41 |
| | `"(3) the code path the issue asks you to change"` repeated in one reply | 39 |
| | `The code path the issue asks to change is the 1000 boundary. This is the same as test 1.` said outright, then the model keeps looking for a third test | 38 |
| third-slot hunting — filler candidates after the two tests that matter (1000 BugReproduction, 500 BehaviorPreservation) | `1001` / `0` / `2000` / `999` / `-1` mentioned as candidate probes | 23 / 28 / 19 / 19 / 15 |
| | the issue's `"without changing the existing behavior at the other thresholds."` quoted while looking for thresholds that do not exist | 180 |
| | `"Hmm, I keep coming back to the same thing"` / second-guessing openers (`Actually`, `Hmm`, `Wait`, …) | 423 / 288 / 160 |

The reading: on the toy, priority items (1) and (3) are the SAME test — the changed
comparison is the code path the issue asks to change — so a line that lists them separately invites a
slot to be spent twice on one test, and once that happens the third slot has nothing left to target and
the model scavenges fillers. The fix makes the items explicitly non-exclusive (counted once, against
the higher item), gives the leftover slot a first-found BehaviorPreservation default, and makes the plan
final. The toy fixture's issue text is **not** touched — it is reference data.

**A21 follow-up — the leftover-slot default has no terminal case (`70fdc50`).** The A3 leftover-slot
default names an input "where the ORIGINAL source returns a DIFFERENT result than at the inputs you have
already tested". It has no answer once every meaningful input is already in the shared set, which is
exactly the state a late round reaches: round 3 on the toy (the shared set already carrying the boundary
and the band from rounds 1 and 2) looped **45×** on the same three phrases —

| the loop | evidence (raw reply) | count |
|---|---|---|
| hunting for a novel input that does not exist | `"Let me pick the three most valuable tests"` / `"Both are shared"` / `"-1 is shared"`, re-argued instead of choosing | 45 |

The reading: the default is a *search* ("returns a DIFFERENT result than at the inputs you have already
tested") with no terminating branch, so when the search space is exhausted the model keeps searching
rather than spending its slots. The fix adds a **terminal fallback** to the generate prompt — when every
input the defaults point to is already covered, spend each remaining slot on **another input inside a
band that is already covered** (a midpoint of the band, or the value on the other side of a boundary
already probed), chosen on the spot without weighing alternatives. A repeat of an existing test is an
**acceptable** answer there: the driver drops duplicates before BATE, so a round of three repeats still
carries its three tests, while a round that keeps hunting loses all of them. Still exactly 3 tests; the
toy fixture's issue text is **not** touched. The first follow-up commit lands the tag
`ATTACKER_PROMPT_VERSION = 8.shared-band-terminal-fallback`.

**A21 follow-up — the fallback's two loose ends (`745cafb`, `2bd4c50`, `5526621`).** Adding the
fallback exposed a rule pair that pulled opposite ways and left the plan unbounded:

- **The clash (`745cafb`).** The round>=2 user-prompt section said `DO NOT REPEAT ANY OF THESE` about
  the shared set, while the new terminal fallback said `A repeat of an existing test is an ACCEPTABLE
  answer here`. In exactly the late-round state the fallback exists for, the two sentences disagree:
  one forbids the repeat, the other permits it. The section is now **one rule** — *aim for a probe
  that is not already in the shared set; when every useful input is already covered, use the TERMINAL
  FALLBACK, where a repeat is acceptable because duplicates are discarded automatically*. The
  `a duplicate does not count as an attack` information is kept (it is what makes the repeat
  harmless). `the_round_two_section_points_at_the_terminal_fallback_instead_of_forbidding_repeats`
  pins that no **unqualified** `DO NOT REPEAT` survives beside the fallback.
- **The unbounded plan (`2bd4c50`).** The 45× loop was also a *re-listing* loop: the model re-picked
  and re-ranked the same three candidates. The generate prompt now carries one `ONE PASS:` line near
  the FINAL line — list candidate tests at most once, do not re-list/re-rank/re-check a candidate you
  have already considered, and once you have three, write them.
  `the_prompt_limits_the_plan_to_one_pass` pins it.
- **The dedup is a no-op, not a stall (`5526621`).** The fallback's licence to repeat rests on the
  driver discarding duplicates before BATE. `a_round_of_all_duplicates_is_dropped_and_the_run_advances`
  (review mode, three rounds) pins the other half: a round whose three tests are all duplicates of the
  shared set drops all three (`dropped_duplicates=3`, `remaining=0`), makes **no** intent-audit call,
  and the run still advances to the approval — nothing is an error and the round still runs the shared
  set.

Both prompt edits land under **one** tag, `ATTACKER_PROMPT_VERSION = 9.one-pass-plan-and-qualified-repeat`,
and the run records the content hash beside it:
`prompt_hash=attacker=be8be62cb889;intent_check=bbed28c47b1c;creator=53883e3b1c81;diagnostician=e3b0c44298fc`
(SHA-256 of each role's system-prompt text joined with `U+001F`, first 12 hex — `pipeline::prompt_hashes`).
**Disclosure:** `attacker`'s hash covers the three system prompts only, so `745cafb`'s rewording of the
round>=2 **user**-prompt section does not move it; that change is carried by the version tag alone. Only
`2bd4c50`'s `ONE PASS` line, which is in the system prompt, moves `be8be62cb889`.

**Auditor-truncation follow-up — the batch that changes no verdict (`961e42f`, `8e09513`, `d75a60c`,
`698933e`, `60e2c08`, `6c337eb`).** The post-fix coding run on the toy (Ornith-1.5, 12 runs, **every
verdict correct, zero Attacker truncations**) left one symptom: **the intent auditor was cut off at its
cap 21 times across 8 of the 12 runs** (batched at `6144 = 2048 × 3`, single-test at `2048`). The
largest batched reply ends mid-reasoning — `"Let me reconsider grounded once more…"` then
`===END_REASONING===` and **no JSON at all** — which is the whole point: this is not a contradiction
loop (a grep of the 67 `IntentVerdict` raw replies finds **no contradiction phrases**), it is a
deliberation that runs past the cap and never reaches the answer. The budget goes to two questions that
do not change the outcome, for every `BehaviorPreservation` test (2 of the 3 in most rounds):

| what the auditor re-chews | evidence (67 `IntentVerdict` replies) | count |
|---|---|---|
| Q1 on a preservation test — is `"without changing the existing behaviour at the other thresholds"` the issue ASKING for the original result at input 0/999/1002? | the clause quoted | 168 (pre-fix) + 57 (post-fix) |
| | `"does the ISSUE ask for the result that test asserts, AT THE INPUT THE TEST ASSERTS?"` | 17 |
| | `"Preserving existing behaviour is always a valid answer"` / `") FORBIDS the change; it never authorises it and never requires one."` — the same doubt re-checked against the two rules | 23 + 18 |
| Q3 on a preservation test — `assertions_grounded` (`"Is 'light' at 0 a specific value the issue never states?"`), which `route_intent` reads only for `BugReproduction` (A7) | stated in the tail, then discarded by the pipeline | — |
| second-guessing openers | `Actually` / `Hmm` | 225 / 204 |

The six fixes, one commit each — (a)–(c) and (f) wording/provenance only, (d)–(e) code:

- **(a)** `961e42f` — Q1 gets a **decided answer** for a preservation input the issue does not ask to
  change: `issue_asks_for_behaviour = false`. The value is chosen to keep `route_intent` byte-for-byte:
  it matches the rule already in the prompt (a "must not change" clause *forbids* a change; it never
  *authorises* one) and matches `audit_unavailable_verdict`, and because with `issue_requires_change`
  mis-answered `true` an `asks = false` **scraps** where `asks = true` **promotes** a relabel, `false`
  is the answer that cannot manufacture a decision.
- **(b)** `8e09513` — question 3 is asked **only for a `BugReproduction` test**; any other intent writes
  `null`, and a missing/null answer parses to `None`, which never scraps
  (`an_absent_or_positive_grounded_answer_never_scraps`).
- **(c)** `d75a60c` — one line in all four auditor prompts: a "must not change" / "without changing X"
  clause covers **every** input whose original result the issue does not ask to change, and the model
  must not enumerate or debate which inputs are "thresholds".
- **(d)** `698933e` — `INTENT_AUDIT_BATCH_TOKENS_PER_TEST` 2048 → **8192** and
  `INTENT_AUDIT_BATCH_TOKENS_CEILING` 16384 → **32768**, so a long audit finishes; the single-test
  transport inherits the same 8192 through the schedule, both caps stay recorded in `run.summary`, and
  the truncation-retry behaviour is unchanged.
- **(e)** `60e2c08` — `write_forensics_to` sanitises the type name too, so the batched reply is written
  as `IntentVerdict__…` instead of `IntentVerdict>_…`.
- **(f)** `6c337eb` — `session.start` and `run.summary` record `prompt_hash=` beside `prompt_version=`
  (below), and the Attacker and auditor labels are bumped once at the end.

**No decision moves.** (a)–(c) are wording-only, and
`the_wording_fixes_never_move_a_routing_decision` walks the whole grid of intents, answers and relabel
states to show each fix leaves `route_intent`'s decision identical to the answers it replaces;
`the_prompt_scopes_a_must_not_change_clause_to_every_unchanged_input` pins the (c) clause in all four
prompts; `forensics_type_name_is_sanitized` pins the (e) filename. The toy fixture's issue text is
**not** touched.

**Before/after check.** The same grep on runs made after `a2617b3` is the check that the audit moved
them: `attacker=6.expected-on-a-correct-patch` and `intent_check=4.audit-questions-renamed` in
`run.summary` separate the two sets. Runs after this batch carry
`attacker=7.priority-overlap-and-simplify-assertion` and `intent_check=5.preservation-q1-and-grounded-scope`
and, crucially, a `prompt_hash=` that differs whenever the wording does. The A21 work then bumps the
Attacker tag twice more — `8.shared-band-terminal-fallback` for the terminal fallback, then
`9.one-pass-plan-and-qualified-repeat` for the round>=2 reconciliation and the `ONE PASS` line — so a
run after A21 is separable from a run after this batch by both the tag and the hash
(`attacker=be8be62cb889`).

**Evidence and disclosure.** The finding was made on the **toy dev fixture** (§14.7), by reading the
session debug artifact rather than by inspecting an evaluation result, and it is disclosed here as a
fix: runs made before it ran a different Attacker prompt, which is why the prompt version is now
recorded in the summary (below).

**Prompt provenance in `run.summary`.** Every terminal run now writes `prompt_version=` — the same
`attacker=…;intent_check=…;creator=…;diagnostician=…` string `session.start` carries
(`pipeline::prompt_versions`) — so a run's prompt identity is readable from the summary row alone and
a run before the fix is distinguishable from one after it. The auditor's grounded/batched variants
share the auditor's tag by construction (they are variants of one ask), and the Attacker's
revise/simplify prompts are Attacker-family; the tags this audit moved are the Attacker's and the
auditor's.

**A version label is only as good as the bump discipline, so the row also carries a hash (`6c337eb`).**
That discipline slipped twice: `3588728` (A3 follow-up) and `d4ea314` (A10 follow-up) reworded prompts
after the `a2617b3` bump and left the label alone, so rows between those commits carry one tag
(`6.expected-on-a-correct-patch`) for two different prompt texts. Both `session.start` and
`run.summary` now also write `prompt_hash=` — the first 12 hex of the SHA-256 of each role's exact
system-prompt text (`pipeline::prompt_hashes`), joined with `U+001F` so role boundaries are part of the
input. Each role's ingredients are the prompts it can actually send: the Attacker's three
(`attacker_system_prompts`), the auditor's four (`intent_auditor_system_prompts` — one per
grounded/batched combination), the Creator's one, and the hash of an empty list for the Diagnostician.
The hash cannot drift from the wire: `attacker.rs::attacker_prompt_hash_follows_the_call_sites` pins
the Attacker slots against the call-site lines read out of the file itself, and
`m3_session_log.rs::the_recorded_prompt_hash_is_the_hash_of_the_prompts_the_run_sends` recomputes both
rows from the accessors rather than from the recorded string. The two labels this batch reworded are
bumped once, at the end: `ATTACKER_PROMPT_VERSION = 7.priority-overlap-and-simplify-assertion` (the
late bump for `3588728`/`d4ea314`) and
`INTENT_AUDITOR_PROMPT_VERSION = 5.preservation-q1-and-grounded-scope`.

### 14.11 `VANTA_SAMPLING` — the sampling preset

`VANTA_SAMPLING=pinned|coding`, default **`pinned`** — which is today's schedule (§4.3),
byte-identical: Creator 0.1, Attacker generate 0.2, revise/simplify 0.3, intent audit 0.0, the same
`max_tokens` at every site, and no other sampler key pinned, so every request body is what it was
before the flag existed.

`coding` replaces the sampler pins of **every** site with temperature **0.6**, top_p **0.95**, top_k
**20**, min_p **0**, presence_penalty **0** — the Creator, the Attacker's generate/revise/simplify
sites, **and the intent auditor**. The auditor was left out of the first cut, on the argument that it
is a judgement rather than a generation and that a noisier auditor is not a fix for a noisy Attacker;
that left the audit pinned at 0.0 while the tests it audits were sampled at 0.6, so the preset did not
describe the run it was recorded for. `max_tokens` is not part of the preset, so the auditor keeps its
**8192** cap under both presets (§14.10 A19d raised it from 2048 after the cap cut 21 audits off across
a 12-run toy batch). Seeds are unchanged (§14.5 — the configured seed on attempt 1,
`seed + (attempt - 1)` on retries above 0.0), and because the truncation rules read each call's own
temperature, a cut-off audit at 0.6 takes the retry path with the seed advanced instead of failing
fast the way a pinned one does.
`m3_session_log.rs::the_coding_preset_covers_every_site_including_the_auditor` asserts both halves:
under `coding` every site including the auditor carries 0.6/0.95/20/0.0/0.0 with the auditor's 8192
cap, and under `pinned` the auditor bodies still carry temperature 0.0 and none of the extra sampler
keys.

The first cut of the auditor change (`ee9fef2`) reached the **recorded** schedule but not the wire: it
moved `intent_audit`'s row into the coding table and wrote `temperature_intent_audit = 0.6` into
`run.summary`, while the two functions that build the audit's request body still hard-coded
`"temperature": 0.0`. `the_coding_preset_covers_every_site_including_the_auditor` caught the
disagreement — the body read back at 0.0 against the schedule's 0.6. Follow-up `413bdd7` routes both
auditor builders (`attacker_check_intent`, the single-test audit reached from `audit_test_intent`,
BATE's `audit_issue_requires_change`, and the per-test fallback after a truncated batch;
`attacker_check_intent_batch`, the batched round audit) through the same `with_sampling(_, "intent_audit")`
call the generative sites use, so the grounded and ungrounded variants are one code path and the
batch's scaled `max_tokens` is written over the schedule's cap after the pins are applied.

Both the preset and the per-role values it produced are recorded in `run.summary`:
`sampling_preset=` plus `temperature_<site>` / `max_tokens_<site>`, and, under `coding`, the extra
sampler keys per site. The recorded schedule is the **effective** one — the same table the
request bodies carry — and `m3_session_log.rs::the_recorded_sampling_schedule_matches_the_request_bodies`
compares the two.

**Chosen on the dev split and frozen before evaluation.** The preset is part of a run's identity
(§13), so every row of a comparison must run under one preset. The preset is chosen on the dev split
(§14.7) and then **frozen** for the evaluation waves, and each baseline row is run under the **same**
preset as the Vanta rows — a baseline at `pinned` measured against a Vanta at `coding` would
attribute the sampling change to the pipeline.

---

### 14.12 Per-call LLM telemetry, and the per-site breakdown

The overnight `paper-off` batch at `d258ce4` ran away and **could not be diagnosed from its own
evidence**. 119 calls completed; 18 stopped at exactly `max_tokens = 32768` and another 32 ran
10k–21k tokens, at 5–10 minutes each; `sanic/1`'s vanta repair was killed by the harness at exactly
3 h with `no run.summary`. The session log held every call — and nothing that could tell the reads
apart: **no round**, so "four retries of one generation" and "four separate generations" were the
same four rows; **no `finish_reason`**, so a call cut off at the cap was indistinguishable from one
that stopped by itself; **no wall time**, so the 5–10 minutes per long call existed only in a
`debug!` line nothing kept. The INFO log could not substitute: it recorded no token counts. The
diagnosis had to be rebuilt by hand from the llama-server log.

This section is the evidence half of the fix. Every fact named above is now recorded where a reader
of the batch already looks.

**Per call, in the `call.usage` row of `{session_id}_session.jsonl`**, alongside the fields §14.5
already put there (`seq`, `site`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `source`,
`seed`, `truncated`):

- **`round`** — the pipeline round, 1-based, from the process-global the driver sets once per
  attempt (`set_forensics_round(attempt + 1)`), which is the *same* index the forensics filenames
  carry. Read at the call rather than threaded through every agent signature: the round is a
  property of the run, not of the call, and an agent signature changed for bookkeeping is an agent
  signature that can change a verdict. `0` means no round was ever set (the M0 spike).
- **`attempt`** — which retry of `structured_call_with_retry` produced the row, 1-based. Two rows
  with the same `site` and `round` and different `attempt` **are** the retry loop; that shape
  cannot be read off `seq` alone, because the sites interleave.
- **`finish_reason`** — the server's own word, verbatim (`stop`, `length`, …), or `unset` when it
  published none. `length` is the **fact**; `truncated` is that fact interpreted. Keeping both
  means a call that stopped for a reason nobody anticipated stays visible instead of being binned
  as "not truncated", which is all a boolean on its own can do.
- **`wall_ms`** — request sent to stream closed, measured **once**, around the same `Instant` the
  `debug!` line already used, so the record and the log can never disagree about the same call.

**Per run, in `run.summary`:** `llm_calls`, `truncated_calls`, `completion_tokens_total`, and for
every call site `llm_calls_site_<site>` / `truncated_calls_site_<site>` /
`completion_tokens_site_<site>`. The **role** keys (`truncated_calls_creator`,
`truncated_calls_attacker`, `truncated_calls_intent_audit`) stay exactly as they were and the site
breakdown is **additive**: roles answer "which *agent* ran away", sites answer "which of that
agent's calls did". The Attacker alone owns three sites, and a round whose revisions all run away
is a different problem from one whose first generation did — a distinction the role split collapses
by construction. Sites are reported in `SAMPLING_SCHEDULE` order and include the ones a run never
reached, as a zero rather than a missing key, because a breakdown that silently omits a site makes
"it never ran" indistinguishable from "the record forgot it"; a site outside the schedule (the M0
spike names its own) is appended, for the same reason `truncation_counts` warns about one. The
per-site completion sum **excludes rows whose `source` is `estimate`**, exactly as `token_totals`
does — a per-site number that mixes a character-derived guess into a measurement is worse than an
absent one — and `truncated_calls` is counted off the rows rather than summed from the three roles,
so an off-schedule site cannot make the total disagree with the calls it summarises.

**Per run, in the manifest row:** a new `llm` block carrying those totals, the role split
(`truncated_calls_by_role`) and a `by_site` map. The harness **reads the summary back** rather than
recomputing anything: the pipeline's numbers are the record of what it did, and a harness that
derived its own would be a second opinion nobody asked for.

`tests/m3_session_log.rs::token_accounting_records_the_servers_counts_per_call_and_a_run_total`
asserts the per-call half (`round >= 1`, `attempt == 1`, `finish_reason == unset` on a complete
reply, `wall_ms` present) and the per-run half (`truncated_calls == 0`,
`completion_tokens_total` equal to the per-call total, and the per-site call counts summing to the
call count), and `scripts/test_bugsinpy.py::TestLlmBlock` asserts the manifest block, including
that a site the run never reached reports `completion_tokens: None` rather than `0`.

### 14.13 A cut-off reply is never read, and never re-asked unchanged

§14.5 already treated a `finish_reason: "length"` answer as a fact about the reply and retried above
temperature 0.0. This section is the other half: what the pipeline is allowed to *do* with a reply the
server says it cut off — because the overnight run showed the old behaviour was still wrong in two
ways.

**A cut-off reply was parsed as if it were complete.** `structured_call_with_retry` handed the raw
string to the parser regardless of `finish_reason`, so a reply that stopped mid-JSON was scored as a
malformed answer and retried on *that* reading. On sanic every long reply ended at
`===END_REASONING===` with empty CONTENT — the whole `max_tokens` window spent on a reasoning stream,
none of it on the answer — so the parser was being asked to read a prefix. A cut-off reply is now
**never** parsed: `finish_reason == "length"` short-circuits before `parse_into_typed`. The forensics
`raw_response.txt` is still written first, so the raw reply is on disk exactly as before.

**A retry re-sent the identical request.** At the pinned seed (§13) an unchanged request reproduces
the unchanged prefix, so the retry was spent re-reaching the same cut-off. The request that follows a
cut-off answer now carries a **trailing user turn** — `TRUNCATION_NUDGE` — asking the model to answer
directly, keep its reasoning to at most three short sentences, and emit the JSON immediately instead
of exploring alternatives. It is **additive**: the existing messages are untouched, so attempt 1 of
every call — the only attempt a run that never truncates ever makes — is byte-identical to v0.2, and
the pinned-preset tests stay green. Appending rather than rewriting is what makes that guarantee
possible; rewriting the system prompt would have made a nudge a change to every call.

**A call cut off twice gives up instead of looping.** If the brief re-ask is *also* cut off, the call
stops immediately with the site's own conservative failure value — no third request — and the run
counts it. `run.summary` gains `truncated_twice_calls=` (the count of calls cut off on both the
request and the re-ask) and the manifest's `llm` block gains `truncated_twice`, read back from the
summary like every other §14.12 number. The failure text carries the marker `truncated_twice` so a
reader of the session log can tell a call that never produced an answer from one that produced a
wrong one.

**This supersedes the temperature-0.0 fail-fast.** §14.5 failed a cut-off auditor immediately because
an identical retry reproduces an identical prefix — an argument about a *deterministic* call. The
re-ask is no longer identical: it carries the nudge, so at 0.0 it can now produce a different answer,
and one re-ask is worth making for every site. The auditor therefore makes one batched call, then one
nudged batched call, then falls back to per test exactly as §14.5 already specified — the batched
transport record `truncated_fallback` still names the round where the batch was cut off, and a
per-test answer cut off twice takes its ordinary "audit unavailable" verdict rather than a third
request.

**A double-truncated Attacker generation is not an `InfraFailure`.** It used to map to `InfraFailure`
(exit 2), which turned one run-away into a dead run *and* blocked the wave gate (§14.7). With no test
from that call the round simply contributes nothing: the attempt either retries with an empty round
(the Governor bounds the attempts) or runs the shared set alone, and a session row `attacker.truncated`
records `generated_tests=0` with the reason. This is never a stronger approval than the native suite
alone would give, so C2 is untouched.

Tests: `tests/m3_session_log.rs::a_batch_cut_off_at_the_cap_falls_back_to_per_test_audits_with_the_same_decisions`
now expects **two** batch calls (the pinned one, then the nudged one) before the per-test fallback,
and asserts the nudge is absent from the first body and present in the second;
`a_cut_off_attacker_answer_is_re_asked_and_recovers_on_the_next_attempt` asserts the same pair of
bodies at the Attacker site; `a_cut_off_per_test_audit_answer_is_re_asked_once_with_the_brief_instruction`
and `a_per_test_audit_cut_off_twice_takes_its_ordinary_unavailable_verdict` assert the one re-ask and
the give-up, with `truncated_calls_intent_audit` / `truncated_twice_calls` read back from
`run.summary`. `scripts/test_bugsinpy.py::TestLlmBlock` asserts the manifest's `truncated_twice`.

### 14.14 One attempt's revise+simplify calls share a budget

§14.13 stops a *single* call from running away. This section bounds how many such calls a round is
allowed to make at all, because the overnight run's cost was not one long call — it was many.

**The per-test cap bounded one test and nothing bounded their product.** BATE sends a test back to the
Attacker at most three times (`for revision in 0..=3`), which is a bound on *that test*. A round with
three tests had nothing stopping it from spending nine Attacker calls, and the loop was worse than
that: a revision returns a `GeneratedTest` and `run_bate` does **not** check the name it comes back
with, so a revision that renames its test enters as a **brand-new test with a fresh three-revision
budget**. That is the treadmill the batch measured — generate, get scrapped or audited, revise, and
the renamed result gets its own turns — and on sanic, where every mismatch is `actual=Error` and BATE
can never validate (`valid=0 scrapped=3 discard_rate_pct=100.0`), not one of those calls could have
ended the loop by succeeding.

**A budget counted per attempt, shared by every test in it.** `REVISE_BUDGET_PER_ATTEMPT = 4`. One
`ReviseBudget` is created by the attempt loop and borrowed by BATE (`attacker_revise_test`) and by the
Diagnostician's Unknown path (`attacker_simplify_test`), so revise and simplify draw on **one** counter
with **one** ceiling. It is counted per attempt rather than per test precisely because renaming cannot
escape it: the floor is the attempt, not the test's name. A unit is spent **before** the call is made,
not after it returns, so "at most four" is true of calls *made* rather than calls that answered — a
call that fails, truncates or comes back renamed has still been paid for.

**When the budget is spent the test is scrapped, and the round keeps its other tests.** BATE sets the
outstanding outcome to `BateOutcome::Mismatch(REVISE_BUDGET_EXHAUSTED)` —
`revise_budget_exhausted` — and `break`s out of that test's revision loop; the round continues with
whatever tests it has, and if none survive it falls back to the native suite exactly as it already
does. The Diagnostician's Unknown path has no such reason to record, so a spent budget there is simply
an `Err` from the simplification, which is the outcome that path already has when simplification fails:
the Unknown stands and the test is quarantined. Neither path can produce a *stronger* outcome than the
un-remediated one, so C2 is untouched.

**The per-test cap of 3 stays.** It is still the tighter bound on any single test and still the
behaviour `a_test_whose_baseline_contradicts_its_intent_is_revised_before_being_scrapped` guards; the
budget is the bound on the round, and it is what makes an *unbounded* number of tests safe (the fourth
revision anywhere in the attempt is the attempt's last). The wall-clock deadline (§14.15) remains the
outer bound: the budget caps calls, the deadline caps time, and a call that takes ten minutes is not
made better by a call count.

**Recorded, not inferred.** `bate.summary` and `attempt.summary` each carry `revise_budget=` (the cap)
and `revise_calls_used=` (what this BATE round / this attempt actually spent); `run.summary` carries
`revise_budget_per_attempt=` and `revise_calls_used=` summed over the run; the manifest's `llm` block
reads both back, like every other §14.12 number, rather than recomputing them.

Tests: `src/pipeline/mod.rs::the_revise_budget_hands_out_exactly_the_cap_and_then_nothing` asserts the
counter grants exactly the cap, refuses the next request, and does not count the refusal as spent;
`scripts/test_bugsinpy.py::TestLlmBlock::test_the_revise_budget_is_reported_beside_the_call_totals`
asserts the manifest keys, and `test_a_summary_from_before_this_record_claims_nothing` that a summary
written before this record reports `None` rather than a fabricated `0`.

---

### 14.15 Vanta's own wall clock — `budget_exhausted`, not `timed_out`

§14.13 and §14.14 bound what a run is allowed to *spend*; this section bounds how long it may spend it.
The overnight batch's sanic/1 row is the reason: `RUN_TIMEOUT_SECS` (3 h, §8.1's backstop) fired while
the run was inside an Attacker revise call, the harness `SIGKILL`ed the process mid-call, and the artifact
on disk was `no run.summary … (exit None)`. Both `run.summary` and every `call.usage` row are written in
the terminal block, *after* the last call returns, so a process killed inside a call leaves a row with no
verdict, no cost and no prompt provenance — nothing to grade and nothing to explain. The remedy cannot be
a decision Vanta makes *after* being killed. It has to be a decision it makes *before*.

**The deadline is armed in the LLM layer, and checked before every call.** All five structured call sites
(`creator`, `attacker_generate`, `attacker_revise`, `attacker_simplify`, `intent_audit`) converge on
`structured_call_with_retry`, so one check at the top of its attempt loop is a check before **every** call
— including the calls made inside BATE and the Diagnostician, whose loops have no channel back to the
driver. That placement is the point: the run-aways were *inside* those phases, and a check the driver
could only make between phases would have arrived an hour late. A refused call returns an error carrying
`llm::DEADLINE_EXCEEDED_MARKER` (`deadline_exceeded`) instead of making an HTTP request. It is a marker
and not a sentence because the driver has to tell it apart from a transport error, and the two are
opposite claims: a transport error is `InfraFailure` (exit 2, "the machine broke"), a deadline is
"the run ran out of clock".

**A call already in flight is not aborted.** The check is before a call, never during one, so the
deadline is a bound on *starting* work plus one call. That is deliberate — killing a streaming completion
in progress is how a run loses the answer it was about to record — and it is why the harness's own limit
must be strictly larger than Vanta's. The two clocks are set to differ by exactly that one call:
`runner.py` sets `VANTA_DEADLINE_SECS = RUN_TIMEOUT_SECS - DEADLINE_MARGIN_SECS`, with
`DEADLINE_MARGIN_SECS = 10 * 60`, the length of the longest single call the batch measured on sanic's
50–65k-token prompts, plus room for the terminal write.

**The driver checks the same clock, and ends the run as a verdict.** `deadline_past()` is also read at
the top of the attempt loop, because a phase whose calls were all refused would otherwise start another
attempt — another Attacker round, another BATE pass, another native suite run — before anything noticed.
The call-level check turns "the calls stopped" into "the calls failed"; this one turns it into "the run
stopped". The decision is `NeedsReview` (exit 3), **not** a rejection: the deadline is a fact about the
run, and nothing has been shown to be wrong with the candidate. Nothing about the approval logic, the
Governor or the Gatekeeper is touched — the deadline only chooses *when* to stop, and the existing
`NeedsReview` path (§14) supplies the rest.

**`stop_reason=budget_exhausted` is a different word from `timed_out`, on purpose.** The harness's
`timed_out` means Vanta was killed before it could decide anything and usually leaves no `run.summary` to
read; `budget_exhausted` means Vanta noticed its own clock, stopped between calls and wrote the reason
down. A table that pooled them would read a verdict out of a corpse — one row is a decision that exists
and the other is a decision that does not — so they are never two spellings of one "stopped" outcome.
The run writes its own phase row, `run.budget_exhausted`, carrying `final_decision=NeedsReview` and
`stop_reason=budget_exhausted`; an ordinary `NeedsReview` names the weak failures that triggered it and
this one has none to name, so without its own row it would be indistinguishable from an uncorroborated
failure. `run.summary` gains `deadline_secs=` and `stop_reason=`, and the manifest reads both back
(`stop_reason`, `deadline_secs`) rather than inferring them from `final_decision`.

**Unset means no clock, byte for byte.** `set_deadline_secs(None)` leaves the deadline unarmed and
`deadline_past()` returns `false` (`map_or(false, …)`), so every run made before this section behaves
exactly as it did: no check refuses anything. `VANTA_DEADLINE_SECS` is therefore a switch the batch
*always* sets and the library *never* requires.

Tests: `src/llm/mod.rs::_deadline_tests::an_unarmed_deadline_never_passes_and_an_armed_one_passes_when_it_is_due`
asserts the unarmed default never ends a run, that an hour out is not yet due, that a deadline set at
`now` is already due (the `>=`, not `>`), and that disarming is total;
`tests/m3_session_log.rs::a_run_that_exhausts_its_own_clock_ends_as_needs_review_and_says_why` arms
`VANTA_DEADLINE_SECS=0`, asserts exit 3, asserts the stub server saw **zero** calls (refusing to start is
the whole point), reads `stop_reason` / `deadline_secs` / `final_decision` back out of `run.summary`, and
requires the `run.budget_exhausted` row — the last with a panic that lists the phases it did find, so an
absent row fails loudly rather than passing on a `None` comparison;
`scripts/test_bugsinpy.py::TestVantaEnv::test_the_deadline_is_set_short_of_the_timeout_that_kills_the_process`
asserts the runner's margin is real (`0 < deadline < RUN_TIMEOUT_SECS`), and
`TestRunJob::test_a_row_says_whether_the_run_stopped_itself_or_was_killed` asserts the manifest carries
the reason beside `timed_out` — and that a summary from before this record reads as "the run never
stopped itself" rather than a fabricated value.

---

### 14.16 `VANTA_PRESENCE_PENALTY` — and why zero is not unset

§14.13 stops a cut-off reply from being read and §14.14 stops a round from buying unlimited
remediations, but neither addresses the *reason* the sanic calls ran away: the classification was (i),
the same reasoning repeated. Nine of the thirteen `GeneratedTest_*_raw_response.txt` files were the same
test rewritten 45–87 times, all ending at `===END_REASONING===` with empty CONTENT. A reply that repeats
itself is not a reply that is wrong — it is a sampler left to its own devices on a prompt it has already
finished. `presence_penalty` is the parameter that penalises tokens already present in the output, so it
is the one knob pointed directly at that failure mode.

**It is a switch, not a conclusion.** Whether a penalty stops the loops is an empirical question and the
code cannot answer it — only a batch can. `VANTA_PRESENCE_PENALTY` exists so one arm of the batch can be
run at `0` and another at `1.0` with everything else identical, and so the two arms stay differentiable
afterwards. It is deliberately **not** wired into any preset: `VANTA_SAMPLING` chooses a whole sampler
(temperature *and* the nucleus together), while this changes one parameter and leaves every other pin —
including the preset's own — where it was.

**It is an override, applied last.** `sampling_pin_table` builds the preset's pins and then replaces or
inserts `presence_penalty`, so the key is the one thing that wins over both presets. Under `pinned` it
inserts a key that was never there; under `coding` it **replaces** the preset's own `0.0` rather than
sitting beside it, because two `presence_penalty` keys in one JSON object is the later one silently
winning, and a recorded schedule that named both would describe a body nobody sent. It reaches every
site, the auditor included — the repetition happens in the generative roles and the auditor reads their
output, so an override that skipped a site would be a different experiment.

**Zero inserts nothing, and that is the whole design.** `set_presence_penalty(Some(0.0))` and an unset
switch both leave every request body with no `presence_penalty` key at all, so the control arm of the
ablation is byte-identical to a default v0.2 run — which is what makes it a control — and
`the_coding_preset_covers_every_site_including_the_auditor`'s "a pinned body carries no `presence_penalty`"
still holds. The refusal to insert `0.0` is not tidiness: inserting it would change every body byte for a
run that asked for the server's own behaviour, and the byte-identity of a default run is a §4.2 property
the whole comparison rests on.

**Which is exactly why the value is recorded at the RUN level too.** Zero and unset produce identical
bodies, so the pinned-site schedule cannot tell the two arms apart; `run.summary` therefore carries
`presence_penalty=` — `unset` when nobody set the switch, `0.0` when someone asked for zero, the value
when someone asked for one. That is the same argument §14.15 makes about `deadline_secs`: record what was
**asked for**, because what was **sent** cannot distinguish the arms. When the value is non-zero the
per-site keys (`presence_penalty_creator=`, …) appear as well, through the same pin table the bodies are
built from, so the record and the bodies cannot drift; the manifest's `sampling` block reads both
(`sampling.presence_penalty`, and `presence_penalty` on each entry of `sampling.pinned`).

**A malformed value fails the run before it spends an hour.** `env_float` errors on anything that is not
a number and on non-UTF-8, exactly as `env_secs` does, because `VANTA_PRESENCE_PENALTY=high` read as "no
override" would run an arm the operator did not ask for and compare it against the wrong baseline. An
empty value is `None` — what a shell expanding an unset variable hands over — and is not a parse error.

Tests: `src/llm/mod.rs::_presence_penalty_tests::zero_and_unset_pin_nothing_and_a_non_zero_pins_every_site`
asserts the unset and zero cases insert no key (for text and for a real body), that `1.0` reaches every
site in `SAMPLING_SCHEDULE`, that the coding preset keeps exactly one `presence_penalty` pin and that it
is the override's, and that the preset's other keys are unmoved;
`tests/m3_session_log.rs::the_presence_penalty_override_reaches_every_body_and_zero_reaches_none` runs
both arms end to end against the stub server — the control's bodies contain no `presence_penalty` (with a
non-empty-body guard, so the assertion cannot pass vacuously) while its `run.summary` reads `0.0`, and the
armed run's every body carries `1.0` — then re-uses `assert_schedule_matches_bodies` so the override is
held to the same body-versus-record guard as the presets;
`scripts/test_bugsinpy.py::TestSamplingBlock::test_the_presence_penalty_arm_is_read_from_the_record_not_guessed`
asserts the manifest reads `0.0` from the record and leaves the pinned sites clean, and
`test_an_unset_presence_penalty_is_recorded_as_unset_not_as_zero` that `unset` comes back as `None` rather
than a fabricated `0.0`.

---

### 14.17 The revise prompt gets the run's own output, not just BATE's category

The follow-up scan of the sanic session (`e08ff93a`) found the single most expensive misunderstanding in
the batch: nine of thirteen generated tests were **repetition loops**, and the worst of them
(`reverse_order`, round 2 attempt 3) shows the model rewriting the same test over and over while it
*guesses* why "BATE says ERROR". It was guessing because that is all it was given. `validate_with_retry`
returned a `TestResult` and BATE's revise path passed on the CLASSIFIED string alone —
`BATE mismatch: test 'x' intent=BugReproduction expects baseline=Failed but got Error.` The exception
type, its message, and the traceback were read by the runner and then dropped on the floor: the model
cannot act on a category.

**The verdict and the evidence are now the same object.** Layer 1 of BATE runs through
`harness::run_single_test_with_timeout`, which is *already* what `run_single_test` calls — the plain
wrapper is literally that call's `.result` — so carrying [`TestExecutionDetails`] out of the retry loop
adds no run and changes no verdict. `validate_with_retry` therefore returns the details, and the revise
call passes `baseline_evidence(&details)` into the prompt as `WHAT THE BASELINE ACTUALLY DID`. One runner
serves both the verdict and the account of it, so the account cannot describe a run that did not happen.

**ERROR is not FAIL, and the block says which.** `TestResult` collapses two different failures into
`Error`: a test that raised (import, collection, an exception mid-body) and one that was *never graded at
all* (`report_path_exists == false`). Those want different revisions, so the evidence names them
separately — and a baseline that simply PASSED says so, because "the test already holds on the original"
is the whole of BATE's Finding-6 mismatch and the model has been guessing at it too. `timed_out` is named
as its own case rather than being reported as a raised exception.

**The exception's TYPE comes from the Diagnostician's locator.** `diagnostician::parse_failure_site` is
already this codebase's answer to "what exception was that", and reusing it here keeps one definition
instead of two — the evidence quotes the last line naming that type (pytest's short-summary line, which
carries type *and* message), falling back to pytest's own `E`-prefixed line for a bare `assert`, whose
output never names `AssertionError` in the traceback at all.

**Bounded, like §14.13 and for the same reason.** One sanic test printed tens of thousands of lines, so
the remedy for "the model cannot see the error" must not become "the model is buried in it":
`REVISE_EVIDENCE_MAX_LINES = 40`, and the cap keeps the TAIL, because pytest prints the innermost frame —
the frame that says which call raised — last. The header lines count against the cap, so a 40-line block
is 40 lines.

**The prompt says to read it.** The revise system prompt now states that the message carries the real run
and that a revision must be driven by it rather than by a hunch: the sanic loop is what "rewrite it again
and hope" costs. This changes the text the `attacker=` prompt-content hash covers, which is the point of
recording that hash — the batch tables for runs before and after this commit are distinguishable.

Tests: `src/pipeline/bate.rs::the_evidence_carries_the_exception_and_the_frames_the_model_used_to_guess_at`
feeds a real pytest transcript and asserts the ERROR verdict, the type-and-message line and the innermost
frame all reach the block; `a_passing_baseline_is_reported_as_passed_and_an_assertion_as_failed`,
`an_ungraded_error_is_told_apart_from_an_exception_the_test_raised` and
`a_runaway_tests_output_is_cut_to_the_forty_lines_of_evidence` (5,000 frames, asserting both the cap and
that the tail survives) pin the three distinctions;
`tests/m3_session_log.rs::a_test_whose_baseline_contradicts_its_intent_is_revised_before_being_scrapped`
now also inspects the three recorded `revise` bodies and requires each to carry the run's output and the
truthful `PASSED` verdict its stub actually produces.

---

### 14.18 The tree the tests run in is the repository, not the prompt's budget

Every sanic test in the batches has been thrown away — all of them, on both days, and for a reason that
had nothing to do with any model. The smoke run `runs/bugsinpy/smoke-sanic-1` (`a147e652`, HEAD `ff3def9`)
is where it became readable, and §14.17 is what made it readable: the revise prompt carried the run's own
output, so the sanic revise request showed `verdict: ERROR (the test was never graded: it failed to
import or collect)` over a traceback that never mentions the generated test. The Attacker responded by
spending 32,768 tokens twice trying to repair an environment error from inside a test, which is exactly
what a model does when it is asked to fix something it cannot see the cause of.

**What BATE recorded.** The chain is `pytest.py:101` → `_pytest/config/__init__.py:890`
`self.pluginmanager.load_setuptools_entrypoints("pytest11")` (inside `_preparse`) → `pluggy/manager.py:299
plugin = ep.load()` → `pytest_sanic/plugin.py:6 from .utils import TestServer, TestClient` →
`pytest_sanic/utils.py:6 from sanic.server import serve, HttpProtocol` → `/workspace/sanic/__init__.py:2`
→ `/workspace/sanic/app.py:18` → `/workspace/sanic/asgi.py:26 from sanic.websocket import
WebSocketConnection` → `ModuleNotFoundError: No module named 'sanic.websocket'`. The entry point is
loaded before anything is collected, so pytest aborted in `_preparse`, wrote **no** `report.json`, and
every generated test came back `Error` — never `Failed`. BATE saw the same `expected=Failed actual=Error`
three times (`Error` is not a baseline result a BugReproduction test can be revised into), scrapped the
test, and sent it for a revision that could only make it worse. The `TypeError ... collections.deque` text
in those two debug files is the **model's own speculation**; `deque + deque` is fine on Python 3.5+ and that
exception never occurred.

**The hypothesis was that we invoke pytest differently from the native suite. We do not.** Four probes in
the same checkpoint (`vanta-bi-sanic-1`), mounting a tree at `/workspace`:

| probe | tree | invocation | result |
| --- | --- | --- | --- |
| P1 | the capped subset | BATE's form (`python -m pytest -p no:cacheprovider test_run.py …`) | the recorded traceback, EXIT=1 |
| P2 | the capped subset | the native form (`python -m pytest tests …`) | the **identical** traceback, EXIT=1 |
| P3 | the full tree | BATE's form | `1 passed`, EXIT=0 |
| P4 | the full tree | the native form | `1 failed, 634 passed, 2 error in 34.43s`, 637 tests in the report |

Same cwd (`/workspace`), same `rootdir`, same inifile, same plugin set in both forms — because both run
through the same transport. **The difference is the tree, not the invocation**, so "make BATE's match the
native suite's" is satisfied by making *both* run in a complete tree, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD`
is not used: the native suite does not use it either, and disabling a plugin the repo's own layer is
configured to load would be a second fiction rather than a fix.

**Why the tree was not the repo.** `driver.rs` built the tree every test runs in — BATE's baseline and
every revision, the native suite, the graded patched tree, the dependency probe, the static pass — from
`repo.source_files` / `candidate::original_files()`, which is a deterministic walk stopped at 64 KB per
file and **256 KB in total** and blind to any extension it does not call source. Its own log line says what
that produced: `discovered source files file_count=57 total_bytes=262142`. Fifty-seven files *is* the
first 262,142 bytes of sanic — it drops `sanic/websocket.py`, `sanic/worker.py` and **all of `tests/`**
(the same run's context selection, which uses the cap-free walk, listed 116 candidate files and printed
`sanic/websocket.py` in the prompt). So the prompt was shown a file the filesystem did not have.

**There were two silent failures, not one.** The second is worse than the first: the native suite ran in
the same truncated tree, died in the same `_preparse`, and produced no report — which the pipeline read as
`Native: original ran=false (0 tests), patched ran=false (0 tests), 0 regression(s)`. `AllTestsPassed` on
zero tests is not a pass of the repo's suite, it is the absence of one, and the run was APPROVED
(`label: true_approval`) on that basis while the hidden test happened to pass anyway. Two independent
things had to be wrong for sanic to look like a model problem: the graded tests were `Error` and the suite
that would have contradicted the approval never ran.

**The fix separates the two budgets.** They were never the same question. What the *prompt* may spend is a
selection (§1.6.8 and the ranking); what the *tests* may see is a filesystem, and a filesystem cannot be a
prefix — an import either resolves or it does not. So:

* `repo::complete_tree_files(root)` is the **cap-free** walk (`context::walk_candidates`, the same
  deterministic walk the context selection already uses — one implementation, two consumers). It is what
  the driver now builds the execution tree from, and the run prints `Execution tree: N files from <root>`
  so the number is checkable in the log instead of inferred.
* `repo::capped_source_files(root)` keeps the loader's original capped walk for the callers that are
  genuinely prompt-shaped: `Repo::load`'s Creator fallback and the import-path probe.
* The prompt-side fallback for an empty selection is `capped_dump_concat`, byte-for-byte the dump it has
  always been. **No prompt grows because the execution tree did.**

Nothing about the verdict machinery changes: the approval/Governor/Gatekeeper logic, the `false_approval`
labels, §14.13's truncation policy and §14.14's revise cap are untouched.

**The residual limit is stated, not hidden.** The sandbox tree is materialised from a `# FILE: <path>`
concat, which can only carry UTF-8 text, so a binary fixture (`tests/static/python.png`) still cannot
reach the VM. That is a pre-existing property of the transport and it does not corrupt the native gate:
original and patched are the same tree apart from the patch, so an error caused by an absent fixture is
present on both sides and cancels in the regression diff. What the caps did *not* have was any such
symmetry with reality — they removed a module the import chain needs, which no comparison can cancel.

Tests: `src/repo/mod.rs::test_execution_tree_is_not_the_capped_prompt_dump` builds a repo large enough to
spend the 256 KB budget and asserts both halves — the prompt dump omits the file, the execution tree
carries it whole — which is the sanic/1 shape in one fixture;
`test_execution_tree_reaches_directories_the_cap_front_never_reaches` spends the budget from a directory
that sorts first and asserts `sanic/websocket.py` survives, so a future cap change cannot pass the byte
assertion while still pruning a directory the import chain needs.

---

### 14.19 A broken environment is not a bad test: BATE stops instead of revising

§14.18 removed the reason sanic's baselines were `Error`, and it is worth being explicit that it removed
only one reason. The *class* it exposed is general: a generated test can come back `Error` without ever
having been graded, and when that happens the Attacker is asked to repair something it did not break. The
overnight sanic session spent two 32,768-token revise calls per test rewriting a file to fix a plugin
import error — hours of model time producing no evidence, because the loop had no way to represent "the
environment is broken".

**What "never graded" means here, and how it is told apart.** `TestExecutionDetails` already carries the
distinction `§14.17` renders: `report_path_exists == false` means pytest reached no verdict for the
generated test — there is no usable `report.json`, or the test's nodeid is not in it. Combined with
`result == Error` and `!timed_out`, that is "the test was never run to a verdict". Two of those three
facts are essential and one is a trap:

* A **timeout** sets `Error` and can leave no report, but it *did* run the test: it is a fact about the
  test's runtime, and the Attacker can address it by writing a faster test. Excluded.
* An **exception raised inside the test** also sets `Error`, but pytest writes a report whose nodeid is
  the test's, so `report_path_exists` is `true`. That is the test's own fault. Excluded.
* What remains is the pre-collection family: a `pytest11` plugin that fails to load, a `conftest.py` that
  fails to import, a collection error. Those share one more property that separates the environment from
  the test: **the output does not name the generated test file.** A collection error that names
  `test_bate.py` is a syntax error or a bad import *in the test* — the Attacker's to fix, and it must
  still be revised. A plugin load fails in `pytest_sanic/plugin.py` and never mentions `test_bate.py`.

So `bate::harness_error` is `Some(evidence)` exactly when all four hold: `Error`, no timeout, no report
for the generated test, and the output silent about `test_bate.py`. The filename is a named constant
(`harness::BATE_TEST_FILE`) because the write and the question have to agree — if they drifted, a test
whose own collection failed would be read as a broken environment and never revised. The evidence is
rendered by `§14.17`'s `baseline_evidence`, so the record carries the same `verdict / exception /
last lines` shape under the same 40-line cap.

**Read from the last of the three baseline reruns, not the first.** `validate_with_retry` exists to absorb
flakiness, and a single transient VM fault must not abort an attempt. A genuine broken environment
reproduces across all three fresh VMs — the same reproducibility test `collapse_decision` already applies
to the patched side — so the classification is made on the details `validate_with_retry` returned, after
it has finished retrying.

**The response is to stop, not to revise.** On `Some`, `run_bate` returns immediately with a
`BateRun::harness_error` and does not touch the relabel or revise paths: every remaining test in the batch
would fail identically (same interpreter, same plugins, same tree), so continuing would spend a revision
budget per test to learn the same thing. The test is deliberately **not** added to `scrapped_tests` —
reporting it as scrapped would blame the Attacker for the environment and inflate the discard rate with a
non-test.

The driver turns that into the existing terminal verdict `HarnessFailure` ("the setup, not the patch",
which does not retry) and records it in three places:

* a `bate.harness_error[<test>]` session row carrying the full evidence;
* `run.summary`'s `harness_error=<test>` — `unset` on every run in which BATE reached verdicts, including
  every run before this change. The value is sanitized to a single token because `run.summary` is parsed
  as space-free `key=value` pairs (`SUMMARY_PAIR = (\S+)`) and a raw name containing a space would be
  silently cut at the first one;
* the manifest's `harness_error` field, beside `verdict: "HarnessFailure"`.

The result is one legible row — `verdict=HarnessFailure`, `harness_error=<the test that surfaced it>`,
`label=unobservable` (a `HarnessFailure` produces no patch, so the oracle has nothing to grade) — where
before there was a trail of revisions and no record of why. Nothing about the approval/Governor/Gatekeeper
logic or the `false_approval` labels changes; `HarnessFailure` is the verdict that already meant this.

Tests: `bate::tests::a_plugin_that_fails_to_load_is_the_environment_not_the_test` uses the recorded sanic
traceback verbatim in shape and asserts the evidence says "never graded" *and* names
`ModuleNotFoundError`; `a_collection_error_that_names_the_generated_test_is_still_the_tests_own` pins the
boundary the other way, so a syntax error in the Attacker's own file keeps its revision;
`nothing_else_is_a_harness_error` pins a passing baseline, a failing assertion, a graded exception and a
timeout as not-harness-errors, so the guard can only fire on the never-graded family.

---

### 14.20 `VANTA_MAX_TOKENS_<SITE>` — capping one call without capping the run

§14.14 caps how **many** revise and simplify calls one attempt may make; it cannot see how **long** one
of them runs. The PySnooper/3 smoke's round-2 `attacker_generate` spent its whole 32,768-token cap without
answering, and the sanic smoke's revise #1 did the same twice; a single call that spends the entire output
budget re-checking its own formatting is a run-away the call budget cannot represent, because it is one
call. The two knobs answer the same question — "is the Attacker stuck?" — from two sides, and a batch
needs to measure the second one.

**It is an override of the pin table, applied last.** `sampling_pin_table` builds the preset's pins and
then **replaces** the `max_tokens` slot, exactly as §14.16 replaces `presence_penalty`. It never appends:
two `max_tokens` keys in one JSON object is the later one silently winning, and a recorded schedule that
named both would describe a cap nobody sent. Because the slot it replaces is the one the schedule put
there, the switch **replaces the existing `max_tokens` slot** rather than adding a second source of truth
— and because the pin table is the same table `with_sampling` fills bodies from and `sampling_pins_text`
renders records from, the number in `run.summary` cannot drift from the number the body carried.

**Every site is addressable, including the first generation.** The variable name is *derived* from the
site's own name (`llm::max_tokens_env_var`), so the set of caps a run can ask for and the set of sites the
pin table knows cannot drift: `attacker_revise` is `VANTA_MAX_TOKENS_ATTACKER_REVISE`, `attacker_generate`
is `VANTA_MAX_TOKENS_ATTACKER_GENERATE`, and there is no spelling for a site the table would panic on.
The user's brief names `attacker_revise`/`attacker_simplify` as the motivation, but the round-2
`attacker_generate` in the smoke run is the same failure, so the switch is deliberately not revise-only.

**Nothing asked for is byte-identical to before.** An empty override list leaves `sampling_pin_table`
untouched, so every body and every record of a run without the switch is what it was — the §4.2 property
the whole comparison rests on. `set_max_tokens_overrides(Vec::new())` is the default and every run before
this change.

**Recorded at two levels, and the two say different things.** The per-site `max_tokens_<site>` keys in
`run.summary` come from the pin table itself, so they carry the **effective** cap every body was sent
under (the override already reached them); the run-level `max_tokens_overrides=` field carries what was
**asked for**, as `site=cap` pairs in `SAMPLING_SCHEDULE` order, or `unset`. A reader comparing two batch
directories needs the switch, not a number that happens to differ — and rendering in the schedule's own
order rather than the order the environment happened to arrive in keeps two arms of the same ablation
identical. The value holds no space, so the parser in `rerun.rs`/`metrics.py` reads it whole: it ends a
value only at a space that begins a lowercase `key=`, and `attacker_revise=12288,attacker_simplify=12288`
has none. The manifest's `sampling` block reads both (`sampling.max_tokens_overrides`, and `max_tokens` on
each entry of `sampling.pinned`).

**A malformed value fails the run before it spends an hour.** `env_max_tokens_overrides` errors on
anything that is not a number of tokens and on non-UTF-8, for `env_secs`'s reason: `VANTA_MAX_TOKENS_
ATTACKER_REVISE=12k` read as "no override" would run an arm the operator did not ask for and compare it
against the wrong baseline. **Zero is rejected rather than honoured**: a cap of zero is a site that can
emit nothing, which is a misconfiguration and never a measurement — the operator who wants the schedule's
value unsets the var. An empty value is skipped, not an error, so a shell expanding an unset variable
(`VANTA_X=`) does not fail the run.

Tests: `src/llm/mod.rs::_max_tokens_override_tests::the_output_cap_override_replaces_one_slot_and_leaves
_the_rest_on_the_schedule` asserts an unset switch leaves every site's body and record on the schedule and
renders `unset`, that a two-site arm reaches both `pinned` and `coding` bodies with exactly one
`max_tokens` key, that the three sites it does not name are unmoved, and that `attacker_generate` is
addressable — it is **one** test, like §14.16's, because the override is a process-global and parallel
tests that each set and clear it race;
`tests/m3_session_log.rs::the_output_cap_override_replaces_one_sites_slot_and_only_that_site` runs the
control and the arm end to end against the stub server — the control's record reads `unset` and is held to
`assert_schedule_matches_bodies`, the arm's record names the switch in schedule order, its unnamed sites
keep the schedule, and each body is held to **the cap the record names for its own site** (`kind` →
`SITE_FOR_KIND` → `max_tokens_<site>`, rather than `sampling_pins_text`, whose ground truth is the
parent's table and so cannot see an override the child read from the environment) — then asserts a
non-number and a zero both fail with `CONFIG_ERROR` naming the switch **before any call is made**;
`scripts/test_bugsinpy.py::TestSamplingBlock::test_the_output_cap_arm_is_read_from_the_record_per_site`
asserts the manifest reads the asked-for caps as a site→int map and that an unset switch comes back as
`None`, not a fabricated empty map.

---

### 14.21 `VANTA_REASONING_BUDGET` / `VANTA_REASONING_EFFORT` — one reasoning knob, every backend

§14.20 caps how much a call may **emit**; it does not touch how much the model **thinks** before it starts
emitting. `ornith-1.5-35b` is a hybrid-reasoning Qwen3-style model: it opens a thinking block, reasons
inside it, closes it, and only then writes the answer. §14.13 measured the shape of the problem on the
PySnooper/3 smoke — the model worked the answer out, wrote that it had ("my test can't run… Contradiction"),
and then spent the same 32k cap a second time re-checking it — and §14.13's retry catches the cut-off but
cannot stop the model choosing to spend the budget on re-checking in the first place. §14.20 lets an
operator shorten the *answer* budget; it does not let them shorten the *thinking*.

**The interface is provider-neutral; only the mechanism is local.** The user's requirement is explicit:
the same pipeline will later run against Anthropic and OpenAI, where reasoning is hidden or cannot be
continued, so every knob must mean the same thing on every backend. `VANTA_REASONING_BUDGET=<tokens>`
and `VANTA_REASONING_EFFORT=<none|low|medium|high>`, each optional, are therefore *levels and budgets*,
not transport fields: the four effort names are llama.cpp's own **and** OpenAI's `reasoning.effort`
levels, so an arm that says `high` means the same thing on either transport. Each backend maps the pair to
its own native control, and the run **records which mechanism actually implemented it** — that record is
what makes two arms comparable across backends rather than merely similarly named.

**The budget, and what it does *not* bound.** On llama.cpp the budget becomes one request field,
`reasoning_budget_tokens` (mechanism `llamacpp:reasoning_budget_tokens`), applied by
`llm::apply_reasoning_control` at the tail of `with_sampling` — the same place §14.20's output cap is
applied, and for the same reason: a value the record names must be a value a body carried. It is **per
thinking block, not per completion.** The server re-arms the budget with a fresh `remaining` on every new
start tag (`common_reasoning_budget_accept`'s `REASONING_BUDGET_DONE` arm), so a model that closes its
block and opens another gets the budget again; measured through the router at `127.0.0.1:8080`, a request
with no budget produced **11,976** characters of reasoning and `finish_reason=length` at the 4,096-token
completion cap in 45.4 s, while the same request with `reasoning_budget_tokens: 64` produced **219**
characters and `finish_reason=stop` in 10.8 s — a 54.7× difference — and `reasoning_budget_tokens: 0` ended
the block immediately (0 characters). This knob therefore **removes waste inside a call; it does not bound
a call.** The output cap plus §14.13's nudged retry remains the bound, exactly as before, and nothing here
changes either.

**Why not force the block closed.** The server *can* shorten a block that is still `COUNTING`
(`POST /v1/chat/completions/control` with `action: reasoning_end`, which needs `reasoning_control: true`
on the original request), and a prefilled open `<think>` arms the budget because the server feeds all
prompt tokens through the same sampler. Both are deliberately **not used**: neither can port to a hosted
API, where reasoning is hidden and the answer cannot be pre-filled while extended thinking is on
(OpenAI does not return reasoning tokens; Anthropic does not allow pre-filling the answer with thinking
enabled). A design that depended on continuing a partial answer would be a design that only runs here.

**The site allow-list is the Attacker's three calls, and only those.** `llm::REASONING_BUDGET_SITES` is
`attacker_generate`, `attacker_revise`, `attacker_simplify` — the sites whose reasoning is the run-away
§14.13 measured. The intent audit is pinned at temperature 0.0 and holds a measurement the design keeps
fixed; the Creator is outside it too, because a patch reasoned about for fewer tokens is a different
experiment, not the same one made cheaper. One variable (`VANTA_REASONING_BUDGET`), not one per site:
the budget is a property of the arm, and the allow-list — not an operator's spelling — decides where it
reaches.

**The effort level replaces, it does not invent.** `ornith-1.5-35b`'s chat template declares
`supports_reasoning_effort: false`, so the six call sites' hard-coded `"reasoning_effort": "low"` has been
a no-op since v0.2. `apply_reasoning_control` therefore replaces an **existing** `reasoning_effort` key
and never adds one to a site that had none, and the recorded mechanism is `llamacpp:noop` — the honest
statement that this transport ignores the level. The default is unchanged and byte-identical: `low` is
what the sites already send, which is why `run.summary`'s `reasoning_effort=` records the **effective**
level and defaults to `low` rather than reading `unset` (that would describe a run that sent no effort at
all, which no run does). Setting the switch to `low` is therefore indistinguishable from leaving it unset,
which is the truth: the two send byte-identical bodies and implement them the same way.

**Recorded at two levels, twice — switch and mechanism.** `run.summary` carries `reasoning_budget=`
(the tokens asked for, or `unset`), `reasoning_budget_mechanism=` (`llamacpp:reasoning_budget_tokens`, or
`unset`), `reasoning_effort=` (the effective level) and `reasoning_effort_mechanism=` (`llamacpp:noop`).
No value contains a space, so all four survive the `rerun.rs`/`metrics.py` value-terminator rule whole.
The manifest's `sampling.reasoning` block reads them back as `{budget, budget_mechanism, effort,
effort_mechanism}`, mapping `unset` to `None` so a reader cannot mistake "no switch" for "a switch with no
value".

**A malformed or zero value fails the run before it spends an hour.** `env_reasoning_budget` errors on
anything that is not a number of tokens, and `ReasoningEffort::parse` errors on anything that is not one
of the four levels; both fail for `env_secs`'s reason — an ablation arm that ran at a level or budget
nobody asked for is not the arm the operator compared against, and silently falling back would make the
comparison meaningless. `VANTA_REASONING_BUDGET=0` is **rejected**, not honoured: zero is not "off"
(unset is off), it is a model told to reason for no tokens at all, which measures nothing and silently
changes what the arm means. An empty value is skipped, not an error, so a shell expanding an unset
variable (`VANTA_REASONING_BUDGET=`) does not fail the run. `VANTA_REASONING_BUDGET` is unbounded above
only by `u32`; the budget is enforced per block so an over-large value is simply inert, not a failure.

**Nothing asked for is byte-identical to before.** With both switches unset, `apply_reasoning_control`
touches nothing, so every request body and every record is what it was — the §4.2 property the whole
comparison rests on. `reasoning_budget_message` is deliberately left unused: it is llama.cpp's
continuation-hint field, and a hint is a mechanism-specific nudge that would change the bodies of an
otherwise-default run.

Tests: `src/llm/mod.rs::_reasoning_control_tests::the_reasoning_switches_reach_the_three_attacker_sites
_and_nothing_else` asserts an unset pair leaves every site's body untouched and renders
`unset`/`llamacpp:noop`, that a budget reaches exactly the three allow-listed sites with the effort key
replaced only where a site already had one, that a site outside the allow-list is unmoved, and that the
four recorded fields name both the switch and the mechanism — it is **one** test, like §14.16's, because
the controls are process-globals and parallel tests that each set and clear them race;
`tests/m3_session_log.rs::the_reasoning_budget_reaches_the_attacker_calls_and_leaves_a_default_run_alone`
runs the control and the armed attempt end to end against the stub server — the control's record reads
`unset`/`low`/`llamacpp:noop` and is held to `assert_schedule_matches_bodies` while no body carries a
budget, the armed attempt's every allow-listed `attacker_*` body carries `reasoning_budget_tokens` and the
`high` it was asked for while no body outside the allow-list carries a budget and no body is handed an
effort it did not have — then asserts a non-number, a zero and an unknown effort level each fail with
`CONFIG_ERROR` naming the switch **before any call is made**;
`scripts/test_bugsinpy.py::TestSamplingBlock::test_the_reasoning_arm_is_read_from_the_record` asserts the
manifest reads the switch and the mechanism back as a block and that an unset pair comes back as
`{budget: None, budget_mechanism: None, effort: "low", effort_mechanism: "llamacpp:noop"}` — not a
fabricated empty block, and not `effort: None`, which no real run emits.

---

### 14.22 `VANTA_ATTACKER_CONCISE` — one sentence, on the first request only

Optional, **off by default**. When `VANTA_ATTACKER_CONCISE=1` the Attacker's *first* request — the
generation call, not the BATE-revision or the Unknown path — carries one extra sentence appended to its
USER message:

> Once each test is written and checked once, output the JSON immediately; do not re-verify formatting.

[VANTA-WHY] §14.13's measured run-away is a model that worked the answer out and then spent the rest of
its output cap re-checking it. §14.20 caps the ANSWER and §14.21 caps the THINKING; this is the cheapest
attempt at the third cause, that the prompt never told the model one check is enough. It is deliberately
the generate prompt and not `attacker_revise`/`attacker_simplify`: those prompts already answer a specific
BATE result, and a bare "do not re-check" there would tell a revision not to read the evidence §14.17
just handed it.

**The sentence rides the USER message, not `ATTACKER_SYSTEM_PROMPT`.** The §14.10-f `attacker=` hash is
taken over exactly the three system prompts, so appending to a system prompt would change the recorded
hash for an arm that only adds a sentence — two different sent texts sharing one hash. Keeping it on the
user message leaves the hash naming the system prompts the run actually sent; the arm is still
identifiable because `run.summary` records the switch and the manifest carries it, which is what an
ablation row needs.

The switch is read with `env_flag` (`on|1|true|yes` / `off|0|false|no`; anything else is a
`CONFIG_ERROR`, because `VANTA_ATTACKER_CONCISE=onf` quietly meaning "on" would mislabel the arm). It is a
process-global `AtomicBool` in `agents::attacker` (`set_attacker_concise`/`attacker_concise`), and the
suffix is exposed as `attacker_concise_suffix()` so a test pins the text that is SENT rather than a copy
of it. `run.summary` gains `attacker_concise=on|off` beside the other rule switches (`flag_text`, §14.3),
and the manifest's `attacker_concise` carries the pipeline's own spelling, so a row that predates the
field is distinguishable from one that ran with it off.

**Nothing here changes when the switch is unset.** The suffix is empty, so every default run's first
request body is byte-for-byte what it was before this section existed, and the safety net is untouched:
the output cap plus §14.13's nudged retry remains the bound, and the sentence is an instruction, not a
mechanism.

Tests: `src/agents/attacker.rs::tests::the_concise_suffix_is_the_sentence_and_is_not_in_a_system_prompt`
pins the exact sentence and asserts it is **not** a substring of any prompt the `attacker=` hash covers,
which is what keeps the sentence on the user message;
`tests/m3_session_log.rs::the_concise_switch_appends_the_sentence_to_the_generate_prompt_only` runs the
control and the armed attempt against the stub server — the control records `attacker_concise=off` and is
held to `assert_schedule_matches_bodies` while no decoded USER message carries the sentence, the armed
attempt's every `generate` body's USER message ends with the sentence and its SYSTEM message is free of
it while no other call site's USER message carries it, and the switch changes neither the verdict nor
which calls are made;
`scripts/test_bugsinpy.py::TestRunJob::test_a_row_says_which_attacker_prompt_it_sent` asserts the manifest
reads `on`/`off` back and that a row predating the field comes back as `None` rather than a default.

---

### 14.23 Per-round seeds — the review rounds are independent draws

The review loop re-asks the Attacker the same prompt once per round. With **one** pinned
`VANTA_LLM_SEED` every round sampled identically: round 2 was a deterministic duplicate of round 1, not a
second, independent attack, so "the Attacker found nothing in round 2" was not evidence that a fresh
look found nothing — it was the same look replayed. This is a correctness fix, not an ablation arm, so it
is **default-on and has no switch**.

**Round `r` sends `VANTA_LLM_SEED + (r - 1)`.** Round 1 sends the configured seed **unchanged**, so a run
that never reaches round 2 — and every single-round run — is byte-identical to one from before this
section existed. The offset composes with §14.5's per-attempt offset: `seed_for_round` runs first and
`seed_for_attempt` advances the result by the retry index, so round 1's first attempt is the pinned seed,
round 2 starts one past it, and a retry inside either round advances by the retry index. The round is
`FORENSICS_ROUND`, the same 1-based attempt index the driver sets once per attempt
(`set_forensics_round(attempt + 1)`) and the same index `call.usage` carries; it is 0 outside the
pipeline, and 0 is treated exactly like round 1 so the one-shot `post_chat` path (`bootstrap_m1`, the M0
spike) is untouched. `seed_for_round(Some(seed), r) = Some(seed.wrapping_add(r - 1))` for `r > 1`, else
the seed unchanged; `None` stays `None` in every round, because an unseeded run asks for **no** `seed`
key at all (§4.4, D4's byte-identity rule).

**What the records carry.** Every `call.usage` row keeps its per-call `seed=` **and** gains
`seed_mechanism=`, the same `backend:knob` spelling §14.21's knobs use: `llamacpp:seed` when the body
carried a `seed` key, `unset` when none was asked for. `unsupported` is reserved for a hosted backend
whose API has no seed field — the local transport never emits it, and it exists so a hosted run records a
refusal rather than a silently missing key. `run.summary` gains `seed_policy=per_round_offset` and the
manifest gains `seed_policy`, read back from the summary.

**Rows from before and after this section must never be pooled.** The offset is default-on, so a row's
own `seed=` value cannot say whether its later rounds were independent draws or duplicates of round 1.
A row that predates §14.23 has no `seed_policy` at all, and the manifest carries that absence as `None`
rather than a default — a fabricated `per_round_offset` would silently pool a fixed-seed row with an
offset one. Specifically, the batches that ran under the **old (fixed-seed) policy** are `paper-off`,
`smoke-pysnooper-3` and `smoke-sanic-1`: their rows must be read as fixed-seed regardless of what a
later table's `seed_policy` column would imply, and they must not be mixed into an arm comparison with
post-§14.23 runs.

**Nothing here changes when `VANTA_LLM_SEED` is unset**, and round 1 is unchanged when it is set. The
safety net is untouched: the output cap plus §14.13's nudged retry remains the bound (§14.21's budget and
§14.20's per-site caps are separate knobs and are off by default).

Tests: `src/llm/mod.rs::_seed_round_tests::round_one_is_unchanged_and_every_later_round_is_offset_by_its_index`
pins the arithmetic (`r = 1` unchanged, `r = n` is seed + n − 1, round 0 like round 1, `None` stays
`None`, wrapping at `u64::MAX`) and the three recorded spellings;
`tests/m3_session_log.rs::unset_seed_leaves_request_bodies_unseeded_and_the_verdict_unchanged` runs a
three-round attempt against the stub server and reads the `call.usage` rows rather than assuming them —
the unseeded run adds no `seed` key in any round and records `seed=unset seed_mechanism=unset`, the
seeded run reaches rounds 2 and 3 and each body carries exactly the seed its own row records
(`4242`, `4243`, `4244`) with `seed_mechanism=llamacpp:seed`, both summaries record
`seed_policy=per_round_offset`, and stripping the `seed` key makes each seeded body byte-identical to the
unseeded one;
`scripts/test_bugsinpy.py::TestRunJob::test_a_row_says_which_seed_policy_it_ran_under` asserts the
manifest reads `seed_policy` back and that a row predating the field comes back as `None` rather than a
default.

---

### 14.24 Hosted OpenAI backend, item 1 — backend selection and the /props skip

Vanta has only ever talked to one service: a local `llama-server`, reached over the OpenAI-compatible
`POST {base_url}/chat/completions`. That service has two properties the pipeline has silently depended on
— it publishes its own `/props`, and it accepts the full sampling surface — and both are false of the
hosted platform this section starts to support. The same pipeline must be able to run on the hosted
models without the local path moving a single byte, so the choice of backend becomes an explicit,
recorded fact rather than an assumption spread across the code.

**`VANTA_LLM_BACKEND=llamacpp|openai`, default `llamacpp`.** Parsed first in `LlmConfig::from_env`,
before anything that depends on it. The value is matched exactly: `llamacpp` or `openai`. An unknown
value — `llama.cpp`, `OpenAI`, `llamacp`, the empty string — is a `CONFIG_ERROR` that quotes the
offending value, never a fallback to the local backend. A silent fallback would be the worst possible
failure for this feature: a run that asked for the hosted model and got the local one would write a
manifest row whose model, cost and sampling columns all describe a different experiment.

**`base_url`'s default is a function of the backend.** `LlmBackend::default_base_url()` returns
`http://127.0.0.1:8080/v1` for `llamacpp` — the exact literal `Default::default()` used before this
section — and `https://api.openai.com/v1` for `openai`. `from_env` sets `cfg.base_url` from the backend
and then lets `VANTA_LLM_BASE_URL` override it, on either backend and with the same trailing-slash trim as
before, so the variable still wins for a proxy, a region, or this project's own mock test server. Keeping
the default next to the enum is what stops the pair from drifting; `Default for LlmConfig` now reads it
from there too.

**The backend is a field, not a guess about the URL.** `LlmConfig.backend` exists because the URL alone
cannot answer "does this endpoint publish `/props`?", "does it reject `temperature`?" or "does a token
cost money?" — and a proxy in front of `api.openai.com` would answer all three differently from a
substring match on the host. `LlmConfig::is_openai()` is the single predicate every backend-dependent
branch asks, so the question is asked one way in one place.

**`/props` is skipped on `openai`, not attempted and failed.** Two call sites probe the server:

- `llm::context_window_tokens` — on `openai` the probe is not made. `VANTA_LLM_CONTEXT_TOKENS` is taken
  as-is, with no disagreement check (there is no server value to disagree with), and an unset value keeps
  the existing `65536` backstop **with a louder warning**, because that number is far below a hosted
  model's real window and a silently-too-small guard refuses prompts that would fit. The `OnceCell`
  caching and the local three-way order (`VANTA_LLM_CONTEXT_TOKENS` → `/props` `n_ctx` → `65536`) are
  untouched for `llamacpp`.
- `llm::server_top_p_default` — returns `None` immediately. The manifest's `server_top_p_note` becomes the
  honest "the backend has no /props" rather than the outcome of a 404 fetched on purpose.

**`session.start` gains `llm_backend=`.** The token sits directly after `model=`, in the run-level record
written before attempt 1. The backend is part of a run's identity: a hosted row and a local row that
shared a model string would otherwise be indistinguishable in the log, even though they reject different
sampling fields, bill different amounts and probe different endpoints. `LlmConfig`'s `Display` gains a
matching `backend :` line, and its `context_window` line stops promising a `/props` fallback the hosted
backend never makes.

**Nothing here changes when `VANTA_LLM_BACKEND` is unset.** The default is `llamacpp`; the default
`base_url`, the `/props` probe, the disagreement warning, the key handling and every request body are
exactly as they were. `LlmConfig::default()` and `from_env()` with no new variables still produce the
pre-§14.24 config field for field, which is the property the existing pinning tests already enforce.

Tests: `src/config/mod.rs::tests::test_backend_enum_is_strict_and_round_trips` (exact accepted spellings;
`llama.cpp`/`OpenAI`/`""` rejected; both default endpoints), `test_default_backend_is_llamacpp` (the
default profile is unmoved), and `test_backend_selection_from_env` — **one** test, not six, because env
vars are process-global and `cargo test` runs threads in parallel; it walks unset → `llamacpp` → `openai`
→ explicit `VANTA_LLM_BASE_URL` override → trailing-slash trim → unknown value, asserting at each step
and leaning on `assert_eq!` on the whole `base_url` so "unchanged" is compared, not eyeballed.

---

## 15. Model screen

Vanta's LLM calls are served locally by `llama-server`. Before committing a model to the BugsInPy
waves, the candidates were screened on the toy fixture (`fixtures/toy_calc`), whose six bad cases and
three good cases are the same shape §14.8 uses as its end-to-end guard:

| model | bad cases | good cases | wall time |
|---|---|---|---|
| 27B | 6/6 | 3/3 | 39.0 min |
| 35B-A3B | 6/6 | 3/3 | 32.9 min |
| Flash | 6/6 | 3/3 | 15.3 min |
| Ornith-1.0 | 6/6 | 3/3 | 22.0 min |
| Coder-Next | 1/6 | — | 6.3 min |

Four of the five candidates reach the fixture's full score; Coder-Next reaches 1 of 6 bad cases and
is out (the good-case cell is one the screen did not fill in). The screen says which candidates are
still *eligible*, not which one to use: **the evaluation model is not chosen yet** — that choice is
to be made on the dev split (§14.7), not on these nine cases.

**Ornith-1.5 is shelved.** It runs into the `creator`/`attacker` **32768** output cap and the
`intent_audit` **2048** cap at Vanta's low sampling temperatures (§4.3 — 0.1 / 0.2 / 0.3 / 0.0),
producing truncated answers rather than wrong ones. That is a sampling-pin problem, not a model
verdict, and it is not a candidate for the waves until the pins can be overridden.

**Ornith-1.0 shows the same pattern, more rarely.** One attacker call in nine in the screening spent
the whole 32768-token window on reasoning and emitted no content — the same failure as 1.5's, just
rarer — which is why §14.5 retries a cut-off answer above temperature 0.0 instead of failing the call,
and counts truncations per role: `run.summary`'s `truncated_calls_attacker=` is the rate the paper
reports for the dev split, rather than something recovered by grepping a batch log.

**That at least one of those cut-off Attacker calls was the prompt, not the sampler.** Reading the
raw reply of a capped call (`vanta_sessions/<id>/debug/AttackerResponse_round_1_attempt_1_raw_response.txt`)
showed a repetition loop of one paragraph rather than a model still reasoning, triggered by the
contradiction between the prompt's both-trees `BehaviorPreservation` rule and its BOUNDARY-PROBING
RULE. The prompt was fixed and its version bumped (§14.10), so part of the "Ornith runs away" signal
this screen recorded was a Vanta bug that applies to every model; the retry (§14.5) still covers the
truncations that are genuinely sampling accidents.

Future work:

- **A temperature-override flag**, to be tested across all the models above on the dev split (§14.7).
  The temperatures are a fixed schedule (§4.3) and the schedule is part of a run's identity (§13), so
  the flag has to make the override visible in `run.summary` alongside the §14 flags. Ornith-1.5 is
  the model that motivates it: the question the flag answers is whether Vanta's low temperatures —
  rather than the models — are what the output caps are hitting.
