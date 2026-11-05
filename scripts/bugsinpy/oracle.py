"""The hidden oracle — where a session's ground truth comes from (§5.2, §11 step 4).

This module is deliberately independent of Vanta. It takes a *patch* and a prepared
bug, and answers one question: **is this patch actually a fix?**

    patch_good  ≡  hidden_pass  ∧  suite_regressions == 0

That definition is the whole point of the harness (§5.3). Without it a `Rejected`
verdict on a real fix and a `Rejected` verdict on a broken patch are the same
number, and the resolve rate the thesis reports would be measuring the Creator's
politeness rather than Vanta's discrimination.

Four rules are load-bearing, and three of them are the kind that fail *silently*:

* **Test hunks are never applied.** The oracle always uses the fixed version's test
  files as-is. A patch that edits a test file therefore cannot reach the hidden
  test — it is not applied there (§5.2 step 2). Dropping them is also what makes the
  same code path safe for a *review-mode* candidate patch, which is arbitrary
  model output.
* **The source patch is applied to a fresh copy of ``oracle_tree/``**, never to
  ``oracle_tree/`` itself or to ``vanta_tree/``. Those two are shared by every run
  of a bug; mutating one in place would quietly re-grade every later session
  against a patched tree.
* **Failure sets are compared to the buggy baseline, not to zero.** ``vanta_tree/``
  is green by construction (§1.5), but only *by construction*: if a bug's baseline
  is not green, the residual set is the baseline and pre-existing failures are not
  charged to the patch (§5.2 step 4).
* **A hidden test that does not *pass* is a hidden failure.** ``skipped``,
  ``xfail``, ``deselected`` and "absent from the report" all count as not passing.
  A patch that makes the hidden test disappear must not read as a fix.

``apply_ok = false`` short-circuits to ``unobservable``: nothing is counted, and the
run is excluded from the rates (the M5 censoring rule, §5.2 step 2). An
un-upstreamed patch that will not apply is unattributable — it is a fact about the
artifact, not about Vanta, and both a `true_reject` and a `false_reject` would be
inventions.

An *applied* patch whose hidden run produced no pytest report at all is
``inconclusive`` (item B4): the grading is unreadable, so it is a third state —
neither a pass nor a definite failure — and is never pooled as a known-bad patch.

Everything third-party still runs in Shuru, through ``suite.py``: this module only
decides *which* commands run and how their reports compare.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from . import budget as budget_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .prepare import BugPrep

ORACLE_WORK_DIR = "oracle_work"

# ``git apply`` on a patch that rewrites a whole file is still fast; the suites are
# what take minutes. 300 s matches the review-mode apply used by Vanta itself.
APPLY_TIMEOUT_SECS = 300

APPLY_NO_ORACLE = "no_oracle"
"""The fifth apply class (item B6): there was no patch to grade **against**.

[VANTA-WHY] A bug with no ``FAIL_TO_PASS`` nodeids never reaches an apply at all, so
its result is not one of ``suite.git_apply_patch``'s three — but it is still *not* a
rejection, and item B6's rule reads a ``MaxAttemptsReached`` row's ``apply_ok = false``
as a repair failure only when ``git apply`` refused the patch. Naming this case keeps
the "did the patch get refused" question answerable without equating "no oracle" with
"the patch would not apply".
"""


@dataclass
class OracleResult:
    """One patch, graded. The keys of ``as_dict`` are §5.2 step 6's ``oracle.json``."""

    apply_ok: bool
    apply_detail: str = ""
    apply_outcome: str = ""
    """Which apply class this grading saw (§5.2, item B6).

    One of :data:`suite_mod.APPLY_APPLIED` / ``APPLY_REJECTED`` /
    ``APPLY_EXEC_FAILURE`` / ``APPLY_MISSING_PATCH`` — or :data:`NO_ORACLE` when
    there was nothing to grade against — and ``""`` on a row written before the
    field existed, which is read by item Q's rule unchanged.

    [VANTA-WHY] M3 item B6. ``apply_ok = false`` is three histories at once, and
    before this field the label could not tell them apart: a ``MaxAttemptsReached``
    run whose final patch ``git apply`` **refused** is a measured repair failure
    (``unresolved``), while one that wrote no patch at all, or whose apply command
    never ran, is a measurement failure (``unobservable``). Recording the class
    here keeps that distinction on the row rather than re-deriving it from a
    human-readable ``apply_detail`` that any wording change would silently invert.
    """

    hidden_pass: bool = False
    hidden_nodeids: list[str] = field(default_factory=list)
    hidden_outcomes: dict[str, str] = field(default_factory=dict)
    suite_regressions: list[str] = field(default_factory=list)
    patched_failures: list[str] = field(default_factory=list)
    patch_good: bool = False
    oracle_secs: float = 0.0
    error: str = ""
    inconclusive: bool = False
    """The patch applied but the grading is unreadable (item B4).

    Set when the hidden run produced no pytest report at all: the patch could be
    applied, so this is not the ``unobservable`` censor, but there is no evidence
    that it failed either. It is a third state — neither a pass nor a definite
    failure — and must never be pooled as a known-bad patch.
    """

    cache: Optional[list[dict[str, Any]]] = None
    """Which grading runs were served from the suite cache (M3 item S2).

    [VANTA-WHY] One entry per grading pair — ``{"side": "hidden"|"patched", …}``
    around that pair's own cache note (``key``, ``state``, ``source``,
    ``would_hit``). **Record-only**, exactly as item Q's fields are: nothing here
    may be read by a metric, a label or a verdict, and it is written beside the
    evidence rather than instead of it — the cached pair IS the measurement, so a
    hit changes no number, only where the number came from.

    ``None`` means "not recorded" and is not the same claim as ``[]``. The pair
    notes are appended only once a grading has actually consulted the cache, so an
    ``oracle.json`` written before this item — which has no ``cache`` key at all —
    must read back as unknown rather than as an empty list: ``[]`` would say the
    cache was consulted and held nothing, which for a row graded before the cache
    existed is a statement about a lookup that never happened. The point is
    auditability: a ``true_reject`` whose grading came from a cached pair must be
    traceable to the row that measured those bytes first.
    """

    @property
    def label_input(self) -> str:
        """What §5.3's table switches on: ``patch_good``, the censor, or the
        inconclusive grading.

        ``apply_ok = false`` is the censor; an applied patch whose hidden run was
        unreadable is ``inconclusive`` (item B4), not ``bad`` — reading it as
        ``bad`` would invent a ``true_reject`` out of a run that produced no
        evidence either way.
        """
        if not self.apply_ok:
            return "unobservable"
        if self.inconclusive:
            return "inconclusive"
        return "good" if self.patch_good else "bad"

    def as_dict(self) -> dict[str, Any]:
        return {
            "apply_ok": self.apply_ok,
            "apply_detail": self.apply_detail,
            "apply_outcome": self.apply_outcome,
            "hidden_pass": self.hidden_pass,
            "hidden_nodeids": sorted(self.hidden_nodeids),
            "hidden_outcomes": dict(sorted(self.hidden_outcomes.items())),
            "suite_regressions": sorted(self.suite_regressions),
            "suite_regressions_count": len(self.suite_regressions),
            "patched_failures": sorted(self.patched_failures),
            "patch_good": self.patch_good,
            "oracle_secs": round(self.oracle_secs, 3),
            "error": self.error,
            "inconclusive": self.inconclusive,
            # M3 item S2 — the cache notes of the two grading pairs, record-only.
            # Copied per entry so a caller cannot mutate the field through the dict.
            # ``None`` stays ``None``: "not recorded" is never written as ``[]``.
            "cache": None if self.cache is None else [dict(item) for item in self.cache],
        }


class OracleError(RuntimeError):
    """The oracle could not be run at all (as opposed to "the patch is bad")."""


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def hidden_command(
    test_cmd: Sequence[str],
    native_test_paths: Sequence[str],
    nodeids: Sequence[str],
) -> list[str]:
    """``pytest <nodeids> …`` — the native path argument is replaced, flags are kept.

    ``test_cmd`` runs the *native* suite (``python -m pytest tests -p no:cacheprovider
    --json-report …``); its path arguments are exactly what make it whole, so they are
    dropped and the hidden nodeids take their place. The flags stay, and the
    ``{report}`` placeholder is substituted later by ``suite.build_command`` as usual,
    so one report parser covers both kinds of run.

    [VANTA-WHY] ``native_test_paths`` is the split's recorded list of *those path
    arguments* — **not** ``BugPrep.test_paths``, which is the fix-side statement of
    which files Vanta takes from the fixed commit. The two are different lists, and
    passing the fix-side one here (which is what this function used to be handed)
    drops nothing at all for the corpus's commands, because those name a directory
    (``tests``) while the fix-side list names files inside it. The hidden run then
    collects the whole native suite *beside* the nodeids: it is no longer a hidden
    run, it costs a full suite twice, and a single unrelated collection error in the
    native tree interrupts the session, so every hidden nodeid reads ``absent`` and a
    good patch is graded ``bad``. That is exactly what gate B measured on fastapi/2 —
    467 collected, 0 run, 90 ``ModuleNotFoundError`` collectors, nodeid ``absent``.
    """
    paths = {str(p) for p in native_test_paths}
    kept = [str(part) for part in test_cmd if str(part) not in paths]
    if not nodeids:
        # Nothing to take the path argument's place, so the command stays whole —
        # dropping the path and running from the rootdir instead would be a second,
        # silent interpretation of a command that is not this function's to rewrite.
        return [str(part) for part in test_cmd]
    # The nodeids go immediately after the runner, so pytest receives them as
    # positional arguments whatever the flag order is. Keying on the *runner token*
    # rather than on "the first flag" matters: ``-m`` is a flag as well, and
    # inserting before it yields ``python <nodeid> -m pytest`` — which pytest never
    # sees, because ``python`` fails first.
    runner: Optional[int] = None
    for i, part in enumerate(kept):
        if part == "pytest" or part.endswith("/pytest"):
            runner = i
    insert_at = len(kept) if runner is None else runner + 1
    return kept[:insert_at] + [str(n) for n in nodeids] + kept[insert_at:]


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def score(
    *,
    hidden_pair: suite_mod.SuitePair,
    patched_pair: suite_mod.SuitePair,
    hidden_nodeids: Sequence[str],
    baseline_failures: Sequence[str],
) -> tuple[bool, dict[str, str], list[str], list[str]]:
    """Compare the two reports. Pure — no VM, no filesystem.

    Split out of ``grade`` for the same reason ``suite.compare_runs`` is: this is
    where a mistake is invisible. Reading "no regression" from an empty report, or
    counting a ``skipped`` hidden test as a pass, produces a plausible label rather
    than an error — so the rule has to be executable without a microVM.
    """
    # Both suites run twice (§1.3 step 3). A hidden nodeid that is flaky or missing
    # from one run has no stable outcome, and only a stable ``passed`` is a pass.
    stable = hidden_pair.stable_outcomes
    outcomes = {n: stable.get(n, "absent") for n in hidden_nodeids}
    hidden_pass = bool(hidden_nodeids) and all(
        outcomes[n] == "passed" for n in hidden_nodeids
    )

    baseline = set(baseline_failures)
    patched = set(patched_pair.stable_failures)
    regressions = sorted(patched - baseline)
    return hidden_pass, outcomes, regressions, sorted(patched)


def grade(
    *,
    prep: BugPrep,
    batch_dir: Path,
    checkpoint: str,
    patch_path: Path,
    label: str,
    disk_mb: int,
    baseline_failures: Optional[Sequence[str]] = None,
    timeout_secs: int = 1800,
    budget: Optional[budget_mod.Budget] = None,
) -> OracleResult:
    """Grade ``patch_path`` against ``prep``'s oracle. Never raises for a bad patch.

    ``label`` names the work tree and the report stems — the session id, or
    ``known_good`` / the candidate hash for review mode. It must be unique per
    grading, or a second grading would read the first one's reports.

    The work tree is left in place on purpose: §6.3's rerun-safety rule needs the
    graded tree to still exist for a session whose signal has to be re-run, so
    deleting it here would move a disk problem onto the correctness path.

    ``budget`` (M3 item Y1) is the invocation's ONE scheduler, threaded straight
    into both grading pairs so each pair's two runs may overlap exactly as a
    preparation suite's do (:func:`suite.run_suite_twice`). The two PAIRS stay
    sequential, and must: the patched suite only runs once the hidden pair has
    produced a readable report (item B4's guard returns before the second pair
    whenever it has not). ``None`` — a test, ``pool``, ``mutants`` — keeps the
    grading exactly the serial pair it was.
    """
    started = time.monotonic()
    # M3 item D1 — the grading runs cover the same *native* suite the in-session gate
    # does (``patched_pair`` runs ``prep.test_cmd`` in full), so the cap is derived
    # from prepare's own measurement by the one shared rule. The floor here is the
    # caller's current value (1800), so a bug whose suite was measured fast keeps its
    # cap and a slow one gets room; the rule can only ever raise it.
    timeout_secs = suite_mod.native_timeout_secs(prep.native_secs, floor=timeout_secs)
    # §3.4: ``hidden_nodeids`` is the declared ``FAIL_TO_PASS`` **plus** the recorded
    # collect-flips — the files that do not collect on the buggy source at all, so
    # they appear in no BugsInPy list yet are exactly the fail→pass flips the oracle
    # is for. Read through the one property so the oracle, the strip and the record
    # cannot disagree about what is hidden.
    result = OracleResult(apply_ok=False, hidden_nodeids=prep.hidden_nodeids)

    hidden_nodeids = prep.hidden_nodeids
    if not hidden_nodeids:
        result.error = "no FAIL_TO_PASS nodeids: this bug has no oracle to grade against"
        result.apply_outcome = APPLY_NO_ORACLE
        result.oracle_secs = time.monotonic() - started
        return result

    if not patch_path.is_file():
        result.apply_detail = f"patch not found: {patch_path}"
        result.error = result.apply_detail
        result.apply_outcome = suite_mod.APPLY_MISSING_PATCH
        result.oracle_secs = time.monotonic() - started
        return result

    work_rel = f"{ORACLE_WORK_DIR}/{prep.slug}__{_safe(label)}"
    work = batch_dir / work_rel
    # ``copy_tree`` replaces the destination, so a repeated grading of the same
    # label starts from the unpatched tree rather than from the previous patch.
    tree_mod.copy_tree(prep.root / "oracle_tree", work)

    apply_ok, apply_detail, apply_outcome = suite_mod.git_apply_patch(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=work_rel,
        patch_path=patch_path,
        disk_mb=disk_mb,
        drop_test_hunks=True,
        memory_mb=prep.vm_memory_mb,
        timeout_secs=APPLY_TIMEOUT_SECS,
    )
    result.apply_ok = apply_ok
    result.apply_detail = apply_detail
    result.apply_outcome = apply_outcome
    if not apply_ok:
        # Censored, not failed (§5.2 step 2): an un-appliable patch tells us nothing
        # about the patch's behaviour, and forcing it into `bad` would invent a
        # `true_reject` for Vanta out of a patch-formatting accident. Item B6 —
        # `apply_detail` alone cannot say whether `git apply` refused the diff or the
        # apply command never ran, so the class travels to the row separately.
        result.oracle_secs = time.monotonic() - started
        return result

    # [VANTA-WHY] ``extra_env`` is threaded here for the same reason every other
    # suite call in ``prepare`` threads it: the recorded per-project ``sandbox_env``
    # rule (fastapi's ``PYTHONPATH=./docs_src``) has to reach *every* run of the row,
    # the grading runs included, or the row is graded in an environment that cannot
    # even import its own test files. ``derive_hidden``'s oracle run passes it and
    # measured fastapi/2 as 775/776 passed; dropping it here made the same tree abort
    # with 44 ``ModuleNotFoundError``s and 0 tests collected, so every hidden nodeid
    # read ``absent`` and a good developer fix graded ``bad``. Omitting it was the
    # only asymmetry between "measuring" and "grading" the identical tree.
    sandbox_env = prep.sandbox_env or None
    # [VANTA-WHY] M3 item S2 — ``origin`` turns on the content-keyed cache for this
    # pair. A candidate patch's tree is the same bytes in every round, arm and seed,
    # and the hidden command is a function of the patch and the oracle, so the pair
    # is measured once per candidate rather than once per grading. The label is the
    # session or the candidate hash, so the entry records which row measured it.
    hidden_pair = suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=work_rel,
        # The *native* path arguments (the split's ``native_test_paths``), never
        # ``prep.test_paths`` — see :func:`hidden_command` for what handing it the
        # fix-side list costs.
        test_cmd=hidden_command(prep.test_cmd, prep.native_test_paths, hidden_nodeids),
        report_rel_stem=f"reports/{prep.slug}-{_safe(label)}-hidden",
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        extra_env=sandbox_env,
        memory_mb=prep.vm_memory_mb,
        precompile=prep.precompile,
        origin=f"{prep.slug}/{label}/hidden",
        budget=budget,
    )
    # M3 item S2 — record how the hidden pair was obtained BEFORE the guard below
    # can return: a censored grading is still a grading that may have been served
    # from the cache, and dropping the note there would make exactly the rows whose
    # evidence is weakest the ones whose provenance is least visible. Assigning
    # rather than appending is what keeps the field ``None`` until a grading has
    # actually reached the cache (see the field's docstring).
    if hidden_pair.cache is not None:
        result.cache = [{"side": "hidden", **hidden_pair.cache}]
    # [VANTA-WHY] Item B4. The patch applied, but if the hidden run wrote no
    # pytest report at all there is nothing to compare against: every hidden
    # nodeid would read ``absent``, ``score`` would say ``hidden_pass = false``
    # and ``patch_good = false``, and the grading would read as a *definite*
    # ``bad`` — inventing a ``true_reject`` out of a run that produced no
    # evidence either way, and (in the mutant pool) putting a patch no reviewer
    # can be credited for finding into a "known-bad" pool. This is a third
    # state, ``inconclusive``: the patch is not credited and not blamed. The
    # check runs before the patched suite so an unreadable hidden run does not
    # spend a second full suite for a result that is never returned.
    #
    # [VANTA-WHY] Item E3 — ``and`` BOTH reports, not ``or``. With ``or`` a run
    # where only ONE side wrote a report fell through to ``score``: the unreadable
    # side's nodeids all read ``absent``, ``hidden_pass`` came back false, and the
    # grade was a definite ``bad`` manufactured from a run that produced no
    # evidence. The error text is kept verbatim — ``pool.py`` matches the
    # ``"the hidden run produced no pytest report at all"`` substring.
    if not (hidden_pair.first.report_found and hidden_pair.second.report_found):
        result.inconclusive = True
        result.error = (
            "the hidden run produced no pytest report at all (exit "
            f"{hidden_pair.first.exit_code}/{hidden_pair.second.exit_code}); "
            f"tail: {hidden_pair.first.stdout_tail[-600:]!r}"
        )
        result.oracle_secs = time.monotonic() - started
        return result

    patched_pair = suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=work_rel,
        test_cmd=prep.test_cmd,
        report_rel_stem=f"reports/{prep.slug}-{_safe(label)}-patched",
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        extra_env=sandbox_env,
        memory_mb=prep.vm_memory_mb,
        precompile=prep.precompile,
        # M3 item S2 — the same keying as the hidden pair, on the native command
        # (the patched side of the regression reading).
        origin=f"{prep.slug}/{label}/patched",
        budget=budget,
    )
    # M3 item S2 — the patched pair's note, again before its guard can return.
    if patched_pair.cache is not None:
        result.cache = [*(result.cache or []), {"side": "patched", **patched_pair.cache}]

    # [VANTA-WHY] Item E3 — the mirror of the hidden guard, on the patched side.
    # A patched run that wrote no report leaves ``stable_failures`` empty, so
    # ``regressions == []`` and ``patch_good = hidden_pass``: a FALSE APPROVAL,
    # the one grading error that must never be manufactured from absent evidence.
    # Unreadable on the patched side is ``inconclusive``, never "no regressions".
    if not (patched_pair.first.report_found and patched_pair.second.report_found):
        result.inconclusive = True
        result.error = (
            "the patched run produced no pytest report at all (exit "
            f"{patched_pair.first.exit_code}/{patched_pair.second.exit_code}); "
            f"tail: {patched_pair.first.stdout_tail[-600:]!r}"
        )
        result.oracle_secs = time.monotonic() - started
        return result

    baseline = (
        prep.native_failure_baseline if baseline_failures is None else list(baseline_failures)
    )
    hidden_pass, outcomes, regressions, patched = score(
        hidden_pair=hidden_pair,
        patched_pair=patched_pair,
        hidden_nodeids=hidden_nodeids,
        baseline_failures=baseline,
    )
    result.hidden_pass = hidden_pass
    result.hidden_outcomes = outcomes
    result.suite_regressions = regressions
    result.patched_failures = patched
    result.patch_good = hidden_pass and not regressions
    result.oracle_secs = time.monotonic() - started
    return result


DEV_FIX_STEM = "dev_fix"
"""Label for the developer-fix self-check: names its work tree, its patch and its grading."""

SELFCHECK_DIR = "oracle/selfcheck"
"""Where ``prepare`` keeps the self-check's patch and its ``oracle.json``."""


def grade_dev_fix(
    *,
    prep: BugPrep,
    batch_dir: Path,
    checkpoint: str,
    disk_mb: int,
    timeout_secs: int = 1800,
    budget: Optional[budget_mod.Budget] = None,
) -> OracleResult:
    """Grade the developer fix against the oracle ``prep`` just derived (§11 step 2).

    [VANTA-WHY] The developer fix is the one patch that must be ``patch_good`` *by
    construction*, so grading it checks the **oracle**, not a model. Measured on
    PySnooper/2: ``bug_patch.txt`` touches only ``pysnooper/tracer.py`` while the fix
    commit also adds ``pycompat.PY2`` and ``utils.get_shortish_repr(..., custom_repr=)``,
    so the patched tree cannot import ``pysnooper``, all three ``FAIL_TO_PASS`` nodeids
    read ``absent``, and no patch of any kind could ever be graded good — every
    PySnooper/2 row was measuring the oracle. That is exactly the failure this check
    turns from a plausible wrong label into a refusal.

    The graded bytes are the ones ``runner.candidate_set`` will grade for the same bug:
    the same source (``bug_patch_txt``) through the same filter (test hunks dropped), so
    the two gradings are comparable rather than merely similar.
    """
    stem = f"{SELFCHECK_DIR}/{prep.slug}-{DEV_FIX_STEM}"
    patch_path = batch_dir / f"{stem}.patch"
    result = OracleResult(apply_ok=False, hidden_nodeids=prep.hidden_nodeids)

    text = suite_mod.drop_test_file_hunks(prep.bug.bug_patch_txt)
    if not text.strip():
        result.apply_detail = "the developer fix is nothing but test hunks"
        result.error = result.apply_detail
        result.apply_outcome = suite_mod.APPLY_MISSING_PATCH
        save(batch_dir / f"{stem}.oracle.json", result)
        return result

    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(text, encoding="utf-8")
    result = grade(
        prep=prep,
        batch_dir=batch_dir,
        checkpoint=checkpoint,
        patch_path=patch_path,
        label=DEV_FIX_STEM,
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        budget=budget,
    )
    save(batch_dir / f"{stem}.oracle.json", result)
    return result


def save(path: Path, result: OracleResult) -> Path:
    """Write ``oracle.json`` for a session (§5.2 step 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _safe(label: str) -> str:
    """Filesystem-safe stem for a label (session ids, candidate hashes, …)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label) or "unlabelled"
