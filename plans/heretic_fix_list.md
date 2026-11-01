# Heretic runs — fix list

Evidence sessions in `vanta_sessions/`:

| Label in this list | Session |
|---|---|
| A v2 (broken) | `43f8143a` |
| B v2 (fixed, "real code") | `8c6750b2` |
| Attacker-only (broken) | `56eb4fd3` |
| C (auto-rank) | `40e84874` |
| B v1 (fixed, missing `model.py`) | `e8d393be` |

## What actually happened in each run

| Run | Verdict | What really decided it | Attacker's contribution |
|---|---|---|---|
| A v2 (broken) | ✅ Rejected | Native test | 3 tests, all passed, no catch |
| B v1 (fixed, missing model.py) | ✅ Rejected | mypy only | Both failing tests were bogus |
| B v2 (your real code) | ❌ False reject | Bogus test | Scrapped the one test that would have caught the planted bug |
| Attacker-only (broken) | ⚠️ Right verdict, wrong reason | Bogus test | Broken fake |
| C (auto-rank) | ❓ Rejected | A mypy finding that wasn't saved anywhere | 3 tests, all passed, duplicates of your own tests |

So far the Attacker has produced **zero real catches, plus three false failures**. Every
correct verdict came from the native suite or mypy.

**Correction 1:** B v1 was *not* "an Attacker test catching what native tests miss." That
test used a `MagicMock` model, which accepts any arguments, so the missing `model.py`
couldn't have broken it. Only mypy caught that.

**Correction 2:** The intent check did *not* get T1 backwards. The Attacker's T1 test mixed
two checks in one: "the new field defaults to `None`" (new behavior) and "`get_responses`
is called exactly as before" (preserved behavior). Labeled as a preservation test, that is
inconsistent, so scrapping it followed the rules. Relabeling it as `FeatureAcceptance`
would have been right, and it would have caught the broken patch.

## The three bogus failures, confirmed against heretic's real source

- **B v2 and B v1:** the tests build `HereticSettings(model="test-model")`. Heretic's
  `Settings` uses `CliSettingsSource(cli_parse_args=True)`, so under pytest it tries to
  parse pytest's own command-line arguments and exits. It fails on **any** code at all,
  original or patched.
- **Attacker-only:** the test's fake `FakeTensor` has no `.shape`, and `Model.get_responses`
  calls `inputs["input_ids"].shape[1]`. The test crashes inside its own fake before it ever
  reaches its assertion.
- **B v1's revised caching test:** it insists that `get_responses_batched` is called with
  exactly `(prompts, skip_special_tokens=True)`, but the issue says to thread the value
  through. It locks in old internals that the issue explicitly changes.

The Diagnostician labeled all three **Logic**: "the patch has a bug."

---

## P0: a failing generated test must not reject a patch unless the patch caused the failure

### 1. Save the evidence
Diagnostic payloads currently hold only `{"signal":"assertion_failed"}`: no output, no
traceback, no test source. Save, per failing test:
- the test source,
- pytest's failure output (truncated to ~4 KB),
- the exception type,
- the file and line of the innermost frame.

Also save **every NEW static finding** (file, line, code, message) in `evidence.json` and
print them in the console. Session `40e84874` was rejected for a mypy finding that appears
nowhere in the session files.

### 2. Patch-insensitive failures
When a test fails on BOTH the original and the patch with the same exception type raised at
the same file and line, it says nothing about the patch. Don't reject on it. Mark it
`test_defect`, send it back to the Attacker for revision with the error, and never count it
as Logic. (B v2: `HereticSettings` parsing pytest's arguments raises `SystemExit` on both
trees.)

### 3. Locate the fault before blaming the patch
Classify as **Logic** only if:
- (a) the `AssertionError` comes from the test's own `assert` or a mock `assert_*` call, **or**
- (b) the exception is raised in a file the patch touched **AND** the failing object isn't a
  type defined in the test file.

Exceptions on objects defined in the test (e.g. `'FakeTensor' object has no attribute
'shape'`), `SystemExit`, argparse/CLI errors, and errors raised in test setup before the code
under test is called → `test_defect`: send for revision, don't reject. (`56eb4fd3` round 2.)

### 4. Fix the rejection message
Name the real signal: list the regressed native tests, or list the new static findings, or
list the failing tests. Never print
`fails 0 test(s): (classified by the repo's own suite)` (`40e84874`).

### 5. Diagnostician order
If a test actually ran and failed, classify by its failure (`Logic` or `test_defect`).
Record new static errors separately; don't let them overwrite the test's class (`e8d393be`
shows real failures labeled Static).

---

## P1: intent handling

### 6. Relabel, don't scrap, preservation→feature
The real invariant: **a relabel must never make a test easier to admit or let it skip a BATE
check**, and relabels are driven ONLY by the intent verdict, never by a BATE outcome.

Two directions, both driven by the audit:

- **Promotion** — a test labeled `BehaviorPreservation` whose verdict is
  `issue_asks_for_behaviour=true` **and** `issue_requires_change=true` is relabelled
  `FeatureAcceptance` and admitted. It is safe because BATE's `FeatureAcceptance` rule then
  demands the probe FAIL on the original (it must prove the new behaviour is absent). (B v2's
  `test_keyword_rate_default_calls_get_responses_without_max_response_length` would then have
  caught the broken patch.)

- **Demotion** — an existing `FeatureAcceptance` (or `BugReproduction`) test whose verdict says
  the issue does NOT ask for the asserted behaviour is still relabelled
  `BehaviorPreservation` (when `issue_requires_change=false`). It is safe because BATE then
  demands the test PASS on the original. This direction is KEPT.

**Ceiling:** a test may be relabelled at most once. If a later verdict would flip it back,
scrap it instead. Log every relabel with `from`/`to` and the reason, and count relabels
separately in the summaries.

### 7. Attacker prompt rules (bump `prompt_version`)
- one intent per test; don't mix a new-feature assertion with a preservation assertion.
- Preservation tests must target the exact behaviour named in the issue's "must stay the
  same" clause, not the internal call arguments of functions the issue says to change.
- Don't instantiate application settings/config objects that read CLI args, env vars or
  files; use `MagicMock` or `model_construct()`.
- Fakes must implement every attribute the code under test touches.

### 8. Intent auditor rule (bump `prompt_version`)
A preservation test that asserts a function's internal downstream call is unchanged
contradicts the issue when the issue says the new value is threaded through that function.
Answer `issue_requires_change=true` for it.

---

## P2: correctness and measurement

### 9. Native test counts
Vanta counted 4 tests on master while pytest reports 5 passed, and 12 on the C patch while
pytest reports 15. Find out why (subtests? unittest classes? `run_tests.py`?), since the
native-collapse rule depends on these counts.

### 10. Reproduce the session `40e84874` mypy rejection
Run Vanta's exact static step on that session's `attempt_1_patched_tree`, show the new
finding(s), and tell how Vanta's mypy invocation differs from running
`mypy src/heretic/modifiers/abliteration.py` with the project venv.

### 11. Metrics script requirement
Count a review-mode `Rejected` as a **true catch** only if at least one rejecting signal
discriminates, i.e. it fails on the bad patch and passes on the known-good version (for
BugsInPy, the fixed commit). Otherwise label it `spurious_reject`. Report true catches,
spurious rejects and false approvals separately.

---

## Order of work

Do **P0 first (items 1–5)**, then **P1**, then **P2**.

After P0+P1, re-run the toy matrix and the four heretic runs (A v2, B v2, attacker-only, C)
so we can compare.
