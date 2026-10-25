"""``prepare``: checkout, SWE-bench split, hidden-test derivation, leak gate (§11 step 2).

No LLM, no Vanta. What this module produces, per bug:

``src_buggy/`` ``src_fixed/`` ``oracle_tree/`` ``vanta_tree/`` ``vanta_clean/``
    the trees from §1.2, with the hidden tests removed from the last two;
``FAIL_TO_PASS`` / ``PASS_TO_PASS``
    derived by running three suites twice each (§1.3), so a flaky nodeid can never
    become the oracle;
``context_sizes.md``
    every candidate measured against the window (§1.6);
``issue_texts_all.md`` + ``leak_gate/<bug>.json``
    the issue text and every token the gate removed (§2.2);
``plans/bugsinpy_split.json``
    rewritten with the derived half filled in.

Every suite run happens inside Shuru — ``prepare`` is not exempt from C3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import bake as bake_mod
from . import context as ctx
from . import issue as issue_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .bugsinpy_meta import BugInfo, MetadataError, load_bug_info
from .records import field_of, strings_of
from .split import Split, sha256_file, sha256_text

DISK_MB = 16384
"""Shuru disk size for suite runs. The base image is 1.5 GB and a checkpoint
inherits its layers, so this leaves room for a project's deps without growing the
copy-on-write clone needlessly."""


FULL = "full"
MEASURE_ONLY = "context_only"
"""A reserve is materialised and measured but its suites are not run (§8): that is
what ``context_sizes.md`` needs in order to justify swapping a rejected pilot bug
for a reserve, and it costs a checkout instead of hours of suite runs. Promoting a
reserve to the pilot means changing its ``role`` and re-running ``prepare``."""


UNDECLARED_F2P_REASON = (
    "flips between the buggy source and the fixed commit's tests, but it is not a test "
    "BugsInPy declares for this bug — its file is not bug.info's test_file and it is not "
    "named by run_test.sh — so BugsInPy never meant to run it and it is not part of the "
    "oracle. It is removed from FAIL_TO_PASS and from the PASS_TO_PASS suite, and it is "
    "hidden from vanta_tree so that the Creator cannot chase a test the benchmark does "
    "not ask for. Measured cause on PySnooper/2: the fix adds the test file for a new "
    "helper, so the test flips because the helper does not exist on the buggy source "
    "rather than because of the bug."
)
"""[VANTA-WHY] A flipping test is not automatically an oracle. PySnooper/2's fix adds
``tests/test_utils/test_ensure_tuple.py`` for a helper the fix also adds, so the test
fails on the buggy source for a reason that is not the bug — and the gate refused the
whole bug over it. Excluding it is the narrow fix; letting prepare carry it would put a
test the benchmark does not name into the Creator's tree *and* into the regression
baseline, where a patch could be marked a regression for failing a test BugsInPy never
asked to run. The rule is BugsInPy's own declaration, not a judgement about which tests
look related: ``bug.info``'s ``test_file`` plus the nodeids ``run_test.sh`` names."""


DEV_FIX_SELFCHECK_REASON = (
    "the bug's own developer fix does not grade patch_good against the oracle this batch "
    "just derived from it, so no candidate of any kind could be graded against this bug: a "
    "perfect patch would at best tie the developer fix, and every verdict, ECR and EIR on "
    "this bug would be measuring the oracle rather than Vanta. Refusing the bug here is the "
    "point — an oracle nothing can pass produces a plausible wrong label instead of an "
    "error, which is the failure mode this gate exists for. Measured cause on PySnooper/2: "
    "bug_patch.txt touches only pysnooper/tracer.py while the fix commit also adds "
    "pycompat.PY2 and utils.get_shortish_repr(..., custom_repr=), so the patched tree cannot "
    "import pysnooper and all three FAIL_TO_PASS nodeids read 'absent'."
)
"""[VANTA-WHY] The developer fix is the harness's own ground truth for "a patch that
fixes the bug", so it is the one candidate whose grading cannot be attributed to a model.
PySnooper/2 shipped a whole wave with it grading ``patch_good=false``, and the symptom was
indistinguishable from Vanta being conservative: the baseline approved a no-op, Vanta
rejected everything, and the gate the batch was supposed to enforce had never been asked
about the one patch it could not get wrong. Grading it at prepare time is cheap next to
the run it invalidates."""


class PrepareError(RuntimeError):
    """prepare could not proceed."""


@dataclass
class BugPrep:
    """Everything prepare learned about one bug."""

    bug: BugInfo
    root: Path
    role: str
    wave: int
    test_cmd: list[str]
    prepare_mode: str = FULL
    test_paths: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    f2p_undeclared: list[str] = field(default_factory=list)
    """Flipping tests BugsInPy does not declare: not an oracle, and hidden from the tree."""
    f2p_undeclared_reason: str = ""
    """Why they were removed, as one rule — see :data:`UNDECLARED_F2P_REASON`."""
    pass_to_pass_count: int = 0
    flaky_nodeids: list[str] = field(default_factory=list)
    preexisting_failures: list[str] = field(default_factory=list)
    native_failure_baseline: list[str] = field(default_factory=list)
    native_green_on_vanta_tree: bool = False
    native_skips_buggy: int = 0
    native_skips_fixed: int = 0
    """Stable ``skipped`` outcomes in the non-graded suite, one count per commit.

    [VANTA-WHY] Nothing grades that suite, so a skip there is fine — but it is the
    observation that shows a dependency the layer no longer has. On ``pandas/1``,
    dropping ``beautifulsoup4``, ``scikit-learn`` and ``psycopg2-binary`` turned the
    tests that ask for them into skips: no failure appeared anywhere, and a skip is not
    in :data:`suite.FAILURE_OUTCOMES`, so the run that lost three packages looked
    identical to the run that had them. Recorded per commit rather than merged, because
    the number that carries the information is the *difference* between the two trees —
    a fixed commit does not skip the tests its own fix adds.
    """
    hidden_files_deleted: list[str] = field(default_factory=list)
    partially_hidden_functions: list[str] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    context: Optional[ctx.ContextVerdict] = None
    issue: Optional[issue_mod.IssueText] = None
    deps: Optional[dict[str, Any]] = None

    @property
    def key(self) -> str:
        return self.bug.key

    @property
    def slug(self) -> str:
        """``PySnooper/1`` → ``PySnooper_1``, safe for a filename or a report stem."""
        return self.key.replace("/", "_")

    @property
    def runnable(self) -> bool:
        return (
            self.prepare_mode == FULL
            and not self.gate_failures
            and self.context is not None
            and self.context.context_ok
            and self.issue is not None
            and self.issue.ok
        )

    def to_dict(self) -> dict[str, Any]:
        """The writer for :meth:`from_dict`, and the only shape ``prepare.json`` has.

        ``runnable`` is recorded for the reader's benefit but recomputed by the
        property; everything else round-trips exactly, which is what the
        round-trip test in ``test_bugsinpy.py`` asserts.
        """
        return {
            "bug": self.key,
            "role": self.role,
            "wave": self.wave,
            "prepare_mode": self.prepare_mode,
            "runnable": self.runnable,
            "test_paths": self.test_paths,
            "fail_to_pass": sorted(self.fail_to_pass),
            "f2p_undeclared": sorted(self.f2p_undeclared),
            "f2p_undeclared_reason": self.f2p_undeclared_reason,
            "pass_to_pass_count": self.pass_to_pass_count,
            "flaky_nodeids": sorted(self.flaky_nodeids),
            "preexisting_failures": sorted(self.preexisting_failures),
            "native_failure_baseline": sorted(self.native_failure_baseline),
            "native_green_on_vanta_tree": self.native_green_on_vanta_tree,
            "native_skips_buggy": self.native_skips_buggy,
            "native_skips_fixed": self.native_skips_fixed,
            "hidden_files_deleted": self.hidden_files_deleted,
            "partially_hidden_functions": self.partially_hidden_functions,
            "gate_failures": self.gate_failures,
            "context": self.context.to_dict() if self.context else None,
            "issue": self.issue.to_dict() if self.issue else None,
            "deps": self.deps,
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        bug: BugInfo,
        root: Path,
        test_cmd: Sequence[str],
        where: str = "",
    ) -> "BugPrep":
        """Rebuild a prep from :meth:`to_dict`, or a :class:`RecordError` naming the field.

        Three fields are injected rather than read: ``bug``, ``root`` and
        ``test_cmd`` are not properties ``prepare`` recorded, they come from the
        BugsInPy metadata, the batch directory and the reviewed split — which is
        also why the runner re-reads ``bug.info`` on the way in. Every field
        ``prepare.json`` *does* carry is required, so a reader that fell behind the
        writer fails with the field's name instead of a constructor ``TypeError``.
        """
        owner = cls.__name__
        context_payload = field_of(payload, "context", owner=owner, where=where)
        issue_payload = field_of(payload, "issue", owner=owner, where=where)
        return cls(
            bug=bug,
            root=root,
            test_cmd=[str(part) for part in test_cmd],
            role=str(field_of(payload, "role", owner=owner, where=where)),
            wave=int(field_of(payload, "wave", owner=owner, where=where)),
            prepare_mode=str(field_of(payload, "prepare_mode", owner=owner, where=where)),
            test_paths=strings_of(payload, "test_paths", owner=owner, where=where),
            fail_to_pass=strings_of(payload, "fail_to_pass", owner=owner, where=where),
            f2p_undeclared=strings_of(payload, "f2p_undeclared", owner=owner, where=where),
            f2p_undeclared_reason=str(
                field_of(payload, "f2p_undeclared_reason", owner=owner, where=where)
            ),
            pass_to_pass_count=int(
                field_of(payload, "pass_to_pass_count", owner=owner, where=where)
            ),
            flaky_nodeids=strings_of(payload, "flaky_nodeids", owner=owner, where=where),
            preexisting_failures=strings_of(
                payload, "preexisting_failures", owner=owner, where=where
            ),
            native_failure_baseline=strings_of(
                payload, "native_failure_baseline", owner=owner, where=where
            ),
            native_green_on_vanta_tree=bool(
                field_of(payload, "native_green_on_vanta_tree", owner=owner, where=where)
            ),
            native_skips_buggy=int(
                field_of(payload, "native_skips_buggy", owner=owner, where=where)
            ),
            native_skips_fixed=int(
                field_of(payload, "native_skips_fixed", owner=owner, where=where)
            ),
            hidden_files_deleted=strings_of(
                payload, "hidden_files_deleted", owner=owner, where=where
            ),
            partially_hidden_functions=strings_of(
                payload, "partially_hidden_functions", owner=owner, where=where
            ),
            gate_failures=strings_of(payload, "gate_failures", owner=owner, where=where),
            context=None
            if context_payload is None
            else ctx.ContextVerdict.from_dict(context_payload, where=where),
            issue=None
            if issue_payload is None
            else issue_mod.IssueText.from_dict(issue_payload, where=where),
            deps=field_of(payload, "deps", owner=owner, where=where),
        )


# ---------------------------------------------------------------------------
# tree construction
# ---------------------------------------------------------------------------
def build_trees(bug: BugInfo, repo_cache: Path, bug_root: Path) -> list[str]:
    """Materialise ``src_buggy``, ``src_fixed`` and the pre-strip ``oracle_tree``."""
    bug_root.mkdir(parents=True, exist_ok=True)
    tree_mod.export_tree(repo_cache, bug.buggy_commit, bug_root / "src_buggy")
    tree_mod.export_tree(repo_cache, bug.fixed_commit, bug_root / "src_fixed")
    # `oracle_tree` is the pre-strip Vanta tree: buggy source + the FIXED commit's
    # test files, hidden tests intact. `vanta_tree`/`vanta_clean` are cut from it
    # after the hidden tests have been identified.
    test_paths = tree_mod.test_paths_from_fix(bug, bug_root / "src_fixed")
    tree_mod.compose_vanta_tree(
        bug_root / "src_buggy", bug_root / "src_fixed", bug_root / "oracle_tree", test_paths
    )
    return test_paths


# ---------------------------------------------------------------------------
# the dependency gate (§3.3) — before any oracle work
# ---------------------------------------------------------------------------
def check_deps(
    prep: BugPrep,
    *,
    checkpoint: str,
    batch_dir: Path,
    deps: dict[str, Any],
) -> set[str]:
    """Prove the bug's layer can run its suite *before* the derivation spends it.

    [VANTA-WHY] This is the measured failure it exists for. PySnooper/3's layer was
    baked from ``requirements.txt`` (0 bytes) and ``setup.sh`` (``setup.py install``
    plus ``decorator``), so ``python_toolbox`` and ``future`` were missing;
    ``tests/test_pysnooper.py`` died on import, ``--collect-only`` collected nothing,
    and prepare reported ``hidden-oracle derivation failed`` — a dependency problem
    reported as a property of the *bug*. The two need different fixes, and that
    message sent a debugging session after the wrong suspect.

    Both checks run inside the finished per-bug layer and both run the way the
    suites run — same interpreter, same cwd, same (absent) ``PYTHONPATH``, same
    pytest entry point:

    (a) the oracle tree's tests collect, with no collection error;
    (b) the project's own package imports **from the tree under test**, not from
        site-packages, where a ``setup.py install`` would have left a frozen copy
        that makes every patch grade the same.

    The two failures are raised as separate messages, so neither can be mistaken
    for the other or for the derivation. The collected nodeids are returned, so the
    derivation can use them instead of collecting the same tree a second time.
    """
    module = str(deps["import_name"])
    tree_rel = _rel(prep, "oracle_tree")

    try:
        nodeids = suite_mod.collect_only(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=prep.test_cmd,
            disk_mb=DISK_MB,
        )
    except suite_mod.SuiteError as exc:
        raise suite_mod.SuiteError(f"deps: collection failed: {exc}") from exc

    try:
        where = suite_mod.import_probe(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=prep.test_cmd,
            module=module,
            disk_mb=DISK_MB,
        )
    except suite_mod.SuiteError as exc:
        raise suite_mod.SuiteError(
            f"deps: {module} does not import from the tree under test: {exc}"
        ) from exc

    root = suite_mod.guest_tree_root(tree_rel)
    if not where.startswith(root + "/"):
        raise suite_mod.SuiteError(
            f"deps: package imports from site-packages, not the tree: {module} "
            f"loaded {where}, which is not under {root}"
        )
    return nodeids


# ---------------------------------------------------------------------------
# FAIL_TO_PASS / PASS_TO_PASS derivation (§1.3)
# ---------------------------------------------------------------------------
GRADED_SKIP_REASON = (
    "a FAIL_TO_PASS test is recorded as skipped, which is neither the failure the buggy "
    "commit has to show nor the pass the fixed commit has to show: skipped is not in "
    "suite.FAILURE_OUTCOMES, so FAIL_TO_PASS (= F_before \\ F_after) reads a test that "
    "skips on the fixed commit as one the fix makes pass, while oracle.score reads the "
    "same report and counts a skipped hidden test as *not* passing — a hidden test that "
    "can never pass, so every candidate would be failed for a reason that is not the bug. "
    "A graded test skips almost always because something it needs is missing from the "
    "layer (skip_if_no / importorskip / pytest.mark.skipif), so the pins the bake dropped "
    "are named here when it dropped any"
)
"""Why a graded skip is a gate failure and not a pass — see :func:`graded_skip_failure`."""


def graded_skip_failure(
    graded: Sequence[str],
    *,
    runs: Sequence[tuple[str, suite_mod.SuitePair]],
) -> str:
    """The gate reason for graded nodeids recorded as ``skipped``, or ``""``.

    [VANTA-WHY] ``skipped`` is the one outcome the two halves of the derivation disagree
    about, and both readings come from the same report: :data:`suite.FAILURE_OUTCOMES`
    cannot see it, so the flip the gate measures treats it as the pass half, and
    ``oracle.score`` treats it as not-passing. Reconciling that after the batch is baked
    is too late — the bug is in the run, and its oracle is unsatisfiable — so a graded
    test that skips on *either* commit refuses the bug here. Never as a pass, and never
    silently: the reason names the nodeid and the commit, and ``prepare``'s
    :func:`dropped_pin_note` adds the pins the bake dropped from the layer, which is the
    cause such a skip almost always has (§3.2).

    Reads :attr:`suite.SuitePair.stable_outcomes`, so a test that skips in only one of
    the two runs is flaky and is reported as flaky by :func:`derive_hidden` instead.
    """
    measured: list[str] = []
    for nodeid in graded:
        on = [
            label
            for label, pair in runs
            if pair.stable_outcomes.get(nodeid) == suite_mod.SKIPPED_OUTCOME
        ]
        if on:
            measured.append(f"{nodeid} skipped on {', '.join(on)}")
    if not measured:
        return ""
    return f"{GRADED_SKIP_REASON} Measured: {'; '.join(measured)}"


def skip_count(pair: suite_mod.SuitePair) -> int:
    """How many nodeids this run stably recorded as ``skipped``.

    Recorded, never gated: nothing grades the suite this counts, and a skip in it is a
    legitimate outcome. It is written down per commit because it is the only signal that
    a layer lost a dependency — see :attr:`BugPrep.native_skips_buggy`.
    """
    return sum(
        1
        for outcome in pair.stable_outcomes.values()
        if outcome == suite_mod.SKIPPED_OUTCOME
    )


def derive_hidden(
    prep: BugPrep,
    *,
    checkpoint: str,
    batch_dir: Path,
    collected: Optional[set[str]] = None,
) -> None:
    if collected is None:
        suite_mod.collect_only(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=_rel(prep, "oracle_tree"),
            test_cmd=prep.test_cmd,
            disk_mb=DISK_MB,
        )

    buggy = _suite(checkpoint, batch_dir, prep, "src_buggy", "src_buggy")
    before = _suite(checkpoint, batch_dir, prep, "oracle_tree", "oracle")
    after = _suite(checkpoint, batch_dir, prep, "src_fixed", "src_fixed")

    flaky = (
        buggy.flaky_nodeids
        | buggy.missing_from_one_run
        | before.flaky_nodeids
        | before.missing_from_one_run
        | after.flaky_nodeids
        | after.missing_from_one_run
    )
    prep.flaky_nodeids = sorted(flaky)

    f_buggy_pre = buggy.stable_failures
    f_before = before.stable_failures
    f_after = after.stable_failures
    prep.preexisting_failures = sorted(f_buggy_pre)

    declared, undeclared = split_declared_f2p(sorted(f_before - f_after), prep.bug)
    prep.fail_to_pass = declared
    prep.f2p_undeclared = undeclared
    prep.f2p_undeclared_reason = UNDECLARED_F2P_REASON if undeclared else ""

    if not prep.fail_to_pass:
        prep.gate_failures.append(
            "empty FAIL_TO_PASS: no test BugsInPy declares for this bug flips between the "
            "buggy source and the fixed commit's tests, so there is no oracle to grade "
            "against"
        )

    fixed_only = sorted(f_after - f_buggy_pre)
    if fixed_only:
        prep.gate_failures.append(
            "the fixed tree fails tests the buggy tree does not — the relaxed fixed-tree "
            f"gate (F_after \\ F_buggy_pre == empty) is violated: {fixed_only}"
        )

    # A skip is neither the failure nor the pass the flip is made of, and by the time
    # grading reads the report it is too late to say so (§3.2). Both commits, because a
    # graded test that skips on the buggy source is not a FAIL_TO_PASS either.
    skip_failure = graded_skip_failure(
        prep.fail_to_pass,
        runs=(("the buggy commit", buggy), ("the fixed commit", after)),
    )
    if skip_failure:
        prep.gate_failures.append(skip_failure)

    # The non-graded suite is not gated: a skip in it is a legitimate outcome. It is
    # recorded per commit, because a layer that lost a dependency shows up here and
    # nowhere else — no failure appears, and a skip is invisible to every failure set.
    prep.native_skips_buggy = skip_count(buggy)
    prep.native_skips_fixed = skip_count(after)


DROPPED_PIN_NOTE = (
    " — the bake dropped {pins} from this layer under §3.2's unresolvable-pin rule "
    "(nothing on this image installs it, and no site the graded tests reach imports "
    "it unconditionally), so if this failure is that module missing, the drop is why"
)


def dropped_pin_note(deps: Optional[Mapping[str, Any]]) -> str:
    """The clause that names the pins the bake dropped, for a gate-failure reason.

    [VANTA-WHY] Dropping a pin is a *prediction* — the import audit says nothing the
    graded tests reach imports it unconditionally — and ``prepare``'s gate is the
    measurement that can refute it. A gate failure whose reason does not name the
    drops leaves a reader to guess whether the bug is broken or the prediction was,
    and the two call for opposite responses: fix the bug, or put the pin back. So a
    bug that had a pin dropped says so in every reason the gate produces, and the
    reason already lists the sites the audit found in the layer's provenance.
    """
    dropped = (deps or {}).get("dropped") or []
    pins = [str(record.get("pin")) for record in dropped if record.get("pin")]
    return DROPPED_PIN_NOTE.format(pins=", ".join(pins)) if pins else ""


def split_declared_f2p(
    flipping: Sequence[str], bug: BugInfo
) -> tuple[list[str], list[str]]:
    """Split flipping nodeids into the ones BugsInPy declares and the ones it does not.

    ``declared`` is the oracle (``FAIL_TO_PASS``); ``undeclared`` is returned rather
    than dropped because the caller has to hide it from the tree as well as from the
    oracle — see :data:`UNDECLARED_F2P_REASON` for why both halves matter.
    """
    declared: list[str] = []
    undeclared: list[str] = []
    for nodeid in flipping:
        target = declared if _declared_by_bugsinpy(nodeid, bug) else undeclared
        target.append(nodeid)
    return declared, undeclared


def _declared_by_bugsinpy(nodeid: str, bug: BugInfo) -> bool:
    """The bug's own statement of which tests it means: ``test_file`` or ``run_test.sh``.

    The two ways a nodeid can be declared, and no third: the file ``bug.info`` names,
    or a nodeid the project's own runner names.
    """
    return tree_mod.nodeid_file(nodeid) == bug.test_file or _named_by_run_test(nodeid, bug)


def _named_by_run_test(nodeid: str, bug: BugInfo) -> bool:
    command = bug.run_test_command() or ""
    chain = tree_mod.nodeid_chain(nodeid)
    return bool(chain) and chain[-1] in command


def _suite(
    checkpoint: str,
    batch_dir: Path,
    prep: BugPrep,
    tree_name: str,
    stem_suffix: str,
) -> suite_mod.SuitePair:
    return suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=_rel(prep, tree_name),
        test_cmd=prep.test_cmd,
        report_rel_stem=f"reports/{prep.slug}-{stem_suffix}",
        disk_mb=DISK_MB,
    )


def strip_and_baseline(prep: BugPrep, *, checkpoint: str, batch_dir: Path) -> None:
    """Cut ``vanta_tree``/``vanta_clean`` from ``oracle_tree`` and re-run the suite.

    The pre-strip collected set is what decides "this file is entirely hidden" in
    the stripper, so it is read from the oracle run's report rather than guessed.

    [VANTA-WHY] The hidden set is ``FAIL_TO_PASS`` *plus* the undeclared flippers. A
    test the creator can read but the oracle does not grade is worse than either: the
    creator can spend a round satisfying it, and the native baseline can mark a patch a
    regression for failing a test BugsInPy never runs. Removing it from
    ``oracle_tree``'s derivative is what excludes it from the PASS_TO_PASS suite, since
    the suite is whatever the tree collects.
    """
    oracle = prep.root / "oracle_tree"
    vanta = prep.root / "vanta_tree"

    report = _read_report(batch_dir, f"{prep.slug}-oracle")
    # Through the shared reader: the oracle report of a bug whose own pytest pin caps
    # pytest-json-report at 0.2 is a flat list (§3.2), and reading only `["tests"]`
    # would make every file look uncollected — the hidden set would come out empty for
    # that bug and the stripper would not strip.
    collected_nodeids = [
        test["nodeid"]
        for test in suite_mod.report_tests(report, source=f"{prep.slug}-oracle report")
        if isinstance(test.get("nodeid"), str)
    ]
    collected_by_file = suite_mod.collect_by_file(collected_nodeids)

    tree_mod.copy_tree(oracle, vanta)
    try:
        strip = tree_mod.strip_hidden_tests(
            vanta,
            [*prep.fail_to_pass, *prep.f2p_undeclared],
            collected_by_file,
        )
    except tree_mod.StripError as exc:
        prep.gate_failures.append(f"hidden-test strip failed: {exc}")
        return

    prep.hidden_files_deleted = strip.deleted_files
    prep.partially_hidden_functions = strip.partially_hidden_functions

    after_strip = _suite(checkpoint, batch_dir, prep, "vanta_tree", "vanta")
    prep.native_failure_baseline = sorted(after_strip.stable_failures)
    prep.pass_to_pass_count = len(after_strip.stable_collected) - len(
        prep.native_failure_baseline
    )
    prep.native_green_on_vanta_tree = not prep.native_failure_baseline

    tree_mod.copy_tree(vanta, prep.root / "vanta_clean")


def selfcheck_dev_fix(prep: BugPrep, *, batch_dir: Path, checkpoint: str) -> bool:
    """Grade the developer fix with the real oracle; refuse the bug unless it is good.

    Returns whether the bug may continue. Called last, after the strip, because the
    regression comparison needs ``native_failure_baseline`` — the same baseline
    ``runner.grade_candidate`` leaves implicit when it grades a candidate, so the two
    gradings of the same bytes answer the same question.

    A bug with no ``FAIL_TO_PASS`` is passed through: the derivation has already gated
    it, and there is no oracle for this check to grade against.
    """
    # Deferred: ``oracle`` imports ``BugPrep`` from this module, so a module-level
    # import here would close the cycle before ``BugPrep`` exists.
    from . import oracle as oracle_mod

    if not prep.fail_to_pass:
        return True

    result = oracle_mod.grade_dev_fix(
        prep=prep, batch_dir=batch_dir, checkpoint=checkpoint, disk_mb=DISK_MB
    )
    if result.patch_good:
        return True

    prep.gate_failures.append(
        f"{DEV_FIX_SELFCHECK_REASON} Measured: apply_ok={result.apply_ok} "
        f"({result.apply_detail}), hidden_outcomes={result.hidden_outcomes}, "
        f"suite_regressions={result.suite_regressions}, error={result.error!r}"
    )
    return False


# ---------------------------------------------------------------------------
# issue text
# ---------------------------------------------------------------------------
def build_issue(prep: BugPrep, *, repo_cache: Path, batch_dir: Path) -> None:
    message = issue_mod.fix_commit_message(repo_cache, prep.bug.fixed_commit)
    report = _read_report(batch_dir, f"{prep.slug}-oracle")
    kinds = issue_mod.failure_kinds_from_report(report)

    hidden_sources = issue_mod.hidden_test_sources(
        prep.root / "oracle_tree", prep.fail_to_pass
    )
    answer_tokens = issue_mod.answer_token_counts(
        prep.bug.bug_patch_txt, hidden_sources, prep.fail_to_pass
    )
    tree_tokens = issue_mod.vanta_tree_token_counts(prep.root / "vanta_tree")
    prep.issue = issue_mod.build_issue_text(
        stripped_message=message,
        failure_kinds=kinds,
        answer_tokens=answer_tokens,
        tree_tokens=tree_tokens,
    )


# ---------------------------------------------------------------------------
# the review artifacts
# ---------------------------------------------------------------------------
def write_leak_gate(batch_dir: Path, prep: BugPrep) -> Path:
    path = batch_dir / "leak_gate" / f"{prep.slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = prep.issue.to_dict() if prep.issue else {"ok": False, "reason": "no issue text"}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_issue_texts(batch_dir: Path, preps: list[BugPrep]) -> Path:
    texts_dir = batch_dir / "issue_texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    for prep in preps:
        if prep.issue is None:
            continue
        body = [
            f"# {prep.key}",
            "",
            f"- role: `{prep.role}`  wave: **{prep.wave}**",
            f"- python: `{prep.bug.python_version}`  test_file: `{prep.bug.test_file}`",
            f"- FAIL_TO_PASS: {prep.fail_to_pass}",
            f"- undeclared F2P (excluded, hidden from vanta_tree): {prep.f2p_undeclared}",
            f"- failure kinds: {prep.issue.failure_kinds}",
            f"- gate: {'ok' if prep.issue.ok else '**NEEDS DECISION**'}",
            "",
            "## Frozen text (post-leak-gate)",
            "",
            prep.issue.text,
            "",
            "## Pre-gate text (before the leak gate)",
            "",
            prep.issue.pre_gate,
            "",
            "## Tokens the gate removed",
            "",
        ]
        if prep.issue.leak.removals:
            body += [
                "| token | class | occurrences in answer | in vanta_tree |",
                "|---|---|---|---|",
            ]
            body += [
                f"| `{r.token}` | {r.token_class} | {r.occurrences_in_answer} | "
                f"{r.occurrences_in_vanta_tree} |"
                for r in prep.issue.leak.removals
            ]
        else:
            body.append("— none —")
        if prep.issue.leak.ungated_prose:
            body += [
                "",
                "Prose tokens that exist only in the answer and were **not** gated "
                "(short, unadorned words carry no answer; listed so the choice is "
                "visible rather than implicit):",
                "",
                ", ".join(f"`{t}`" for t in sorted(set(prep.issue.leak.ungated_prose))),
            ]
        body.append("")
        (texts_dir / f"{prep.slug}.md").write_text("\n".join(body), encoding="utf-8")

    all_path = batch_dir / "issue_texts_all.md"
    needs = [prep for prep in preps if prep.issue is not None and not prep.issue.ok]
    summary = [
        "# Issue texts — all candidates, for hand review before any run",
        "",
        f"Generated by `prepare` on {utc_now()}.",
        "Read this before committing the split: it is the audit that costs nothing at run time.",
        "",
        "## NEEDS DECISION",
        "",
    ]
    if needs:
        summary += [
            "These texts are not actionable after the leak gate. Supply a replacement "
            "text or drop the bug; the harness does not guess.",
            "",
        ]
        summary += [f"- **{prep.key}** — {prep.issue.needs_decision}" for prep in needs]
    else:
        summary.append("None — every candidate produced an actionable text.")
    summary += ["", "---", ""]

    for prep in preps:
        if prep.issue is None:
            continue
        summary += [
            f"## {prep.key} ({prep.role}, wave {prep.wave})",
            "",
            prep.issue.text,
            "",
        ]
    all_path.write_text("\n".join(summary), encoding="utf-8")
    return all_path


def write_context_sizes(batch_dir: Path, preps: list[BugPrep]) -> Path:
    rows = [(prep.key, prep.role, prep.context) for prep in preps if prep.context is not None]
    path = batch_dir / "context_sizes.md"
    path.write_text(ctx.render_markdown(rows), encoding="utf-8")
    return path


def write_context_report(
    batch_dir: Path,
    preps: list[BugPrep],
    *,
    n_ctx: Optional[int] = None,
    n_ctx_source: Optional[str] = None,
) -> Path:
    """``context_report.json`` — the machine-readable half of §1.6.

    [VANTA-WHY] Measurements land here and in ``prepare.json``, never in
    ``plans/bugsinpy_split.json``. The split is an *input* the user reviews and
    edits; a stage that rewrites it silently turns a measurement into a decision,
    and the difference between "the plan says this" and "this run observed that" is
    the whole point of keeping the two files apart.

    ``prompt_tokens_real`` is null until a session has actually paid for the prompt:
    the estimate is ``ceil(source_bytes / 4)`` plus a frame and runs high (measured
    ~1.5x on the heretic sessions), so a pilot set chosen from the estimate alone is
    uncalibrated. ``prompt_tokens_real_source`` names the session that filled it in.

    The top-level ``n_ctx``/``n_ctx_source`` are the window the batch was *sized*
    against, which the caller resolved once (§1.6.1) — recorded here even when a
    batch measured no bug at all, so a report never leaves the window implicit.
    """
    path = batch_dir / "context_report.json"
    measured = [prep for prep in preps if prep.context is not None]
    windows = {prep.context.n_ctx for prep in measured}
    budgets = {prep.context.budget_tokens for prep in measured}
    # The frame is the selector's own number now (``--select-context``'s
    # ``prompt_frame_tokens``), not a Python constant: there is no second estimate
    # to keep in step, so it is read back from the measurement rather than mirrored.
    frames = {prep.context.selection.prompt_frame_tokens for prep in measured}
    resolved = n_ctx if n_ctx is not None else (next(iter(windows)) if len(windows) == 1 else None)
    resolved_source = (
        n_ctx_source
        if n_ctx_source is not None
        else (measured[0].context.n_ctx_source if len(windows) == 1 else None)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "context_report",
        "created_utc": utc_now(),
        "estimator": {
            # [VANTA-WHY] The estimator is the run's own selector, reached by
            # shelling out (§1.6.2 item 5). Naming it here is the point: this block
            # used to describe a Python mirror of the Rust walk, and a reader who
            # trusts a mirror is trusting the one thing the gate must not be.
            "source": "vanta_m3 --select-context --json (vanta::context)",
            "prompt_frame_tokens": next(iter(frames)) if len(frames) == 1 else None,
            "output_reserve_tokens": ctx.OUTPUT_RESERVE_TOKENS,
            "chat_overhead_tokens": ctx.CHAT_OVERHEAD_TOKENS,
            "margin_fraction": ctx.CONTEXT_MARGIN_FRACTION,
        },
        "n_ctx": resolved,
        "n_ctx_source": resolved_source,
        "windows_measured": sorted(windows),
        "budget_tokens": next(iter(budgets)) if len(budgets) == 1 else None,
        "budgets_measured": sorted(budgets),
        "fallback_window": bool(resolved_source and resolved_source.startswith("fallback"))
        or any(prep.context.n_ctx_source.startswith("fallback") for prep in measured),
        "bugs": [_context_record(prep) for prep in preps],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def prepare_manifest(
    *,
    split: Split,
    split_path: Path,
    split_sha_before: str,
    batch_dir: Path,
    commit: str,
    bugsinpy_root: Path,
    selected: list[str],
    broken: list[tuple[str, str]],
    preps: list[BugPrep],
    n_ctx: int,
    n_ctx_source: str,
) -> dict[str, Any]:
    """The whole ``prepare.json`` payload — the one writer ``run`` reads back.

    Kept as a function rather than inlined into ``cmd_prepare`` so the round-trip
    test builds the file with the same writer the stage does: a test that assembled
    the payload by hand would pin the reader against a *second* copy of the format,
    which is the drift this contract exists to prevent. Every dataclass below
    ``bugs`` is serialised by its own ``to_dict``, and ``run`` reads it back through
    the matching ``from_dict`` (``records.py``).

    ``n_ctx``/``n_ctx_source`` are the window the gate was sized against (§1.6.1):
    recorded here, and compared against the run's own window by
    ``runner.context_window_warning``.
    """
    return {
        "schema_version": 1,
        "kind": "prepare",
        "created_utc": utc_now(),
        "batch_dir": str(batch_dir),
        "split": str(split_path),
        "split_sha256_before": sha256_text(split.to_json()),
        "split_file_sha256_before": split_sha_before,
        "split_written": False,
        "bugsinpy_commit": commit,
        "bugsinpy_root": str(bugsinpy_root),
        # The window every verdict in this batch was sized against, and where that
        # number came from. ``run`` reads these back and warns when its own differs.
        "n_ctx": n_ctx,
        "n_ctx_source": n_ctx_source,
        "selected": list(selected),
        "breaks": [{"bug": key, "detail": detail} for key, detail in broken],
        "bugs": [prep.to_dict() for prep in preps],
    }


def _context_record(prep: BugPrep) -> dict[str, Any]:
    verdict = prep.context
    if verdict is None:
        return {
            "bug": prep.key,
            "wave": prep.wave,
            "role": prep.role,
            "select_role": None,
            "prepare_mode": prep.prepare_mode,
            "verdict": "not_measured",
            "context_ok": None,
            "estimated_prompt_tokens": None,
            "prompt_tokens_real": None,
            "prompt_tokens_real_source": None,
            "budget_tokens": None,
            "headroom_tokens": None,
            "n_ctx": None,
            "n_ctx_source": None,
            "mode": None,
            "source_bytes": None,
            "files_bytes": None,
            "files_tokens": None,
            "file_count": None,
            "tree_bytes": None,
            "tree_count": None,
            "selected_files": [],
            "dropped_for_budget": [],
            "pruned_count": None,
            "reasons": list(prep.gate_failures),
        }
    selection = verdict.selection
    # The selector's `selected_bytes`/`selected_tokens` are the file bytes PLUS the
    # tree listing's (``ContextSelection::total_bytes`` is
    # ``sum(files.bytes) + tree_bytes``), so the equality
    # ``selected_bytes == tree_bytes`` is the signature of a selection with no file
    # — a listing, always under budget, which is how all 11 dev bugs passed the
    # second measure-only run (§1.6.3). The two halves are therefore recorded
    # apart: `files_bytes`/`files_tokens` are what the file content cost,
    # `tree_bytes`/`tree_count` are the listing, and `source_bytes` is the
    # selector's own total, kept unchanged for continuity with earlier reports.
    files_bytes = sum(file.bytes for file in selection.files)
    files_tokens = sum(file.tokens for file in selection.files)
    return {
        "bug": prep.key,
        "wave": prep.wave,
        "role": prep.role,
        # The role the context was SELECTED for (`SELECT_ROLE`, the union of the
        # repair roles), not the split role above: it is written into the run's
        # `context.select` row, and a report that named only `pilot`/`reserve`
        # could not be lined up with the run's own row.
        "select_role": selection.role,
        "prepare_mode": prep.prepare_mode,
        "verdict": "ok" if verdict.context_ok else "reject",
        "context_ok": verdict.context_ok,
        "estimated_prompt_tokens": verdict.estimated_prompt_tokens,
        "prompt_tokens_real": None,
        "prompt_tokens_real_source": None,
        "budget_tokens": verdict.budget_tokens,
        "headroom_tokens": verdict.headroom_tokens,
        "n_ctx": verdict.n_ctx,
        "n_ctx_source": verdict.n_ctx_source,
        # The selection itself, straight from the run's own selector: how many bytes
        # and files went in, how big the tree listing was, which candidates the
        # ranking chose but the budget could not fit (`dropped_for_budget`) and how
        # many paths the walk pruned. `files_eligible`/`files_skipped_*` are gone
        # with the capped walk they described (§1.6.2).
        "mode": selection.mode,
        "source_bytes": selection.selected_bytes,
        "files_bytes": files_bytes,
        "files_tokens": files_tokens,
        "file_count": len(selection.files),
        "tree_bytes": selection.tree_bytes,
        "tree_count": selection.tree_count,
        # Per file, what an audit of the ranking needs: why it was picked
        # (`path` / `traceback` / `identifier` / `patch-touched` / `import-neighbour`
        # / `test` / `tree`), the score that ordering put on it, what it cost, and
        # whether it is shown whole or as an outline. Without this the report named
        # counts and could not be checked (§1.6.2 item 4).
        "selected_files": [file.to_dict() for file in selection.files],
        "dropped_for_budget": list(selection.dropped_for_budget),
        "pruned_count": len(selection.skipped),
        "reasons": verdict.reasons,
    }


def render_context_table(preps: list[BugPrep]) -> str:
    """The same numbers as ``context_report.json``, as a table on stdout."""
    head = (
        f"{'bug':<18} {'mode':<12} {'est tok':>8} {'real tok':>9} {'budget':>7} "
        f"{'headroom':>9} {'src B':>9} {'files':>6}  verdict"
    )
    lines = [head, "-" * len(head)]
    for prep in preps:
        verdict = prep.context
        if verdict is None:
            lines.append(
                f"{prep.key:<18} {prep.prepare_mode:<12} {'—':>8} {'—':>9} {'—':>7} "
                f"{'—':>9} {'—':>9} {'—':>6}  not measured"
            )
            continue
        lines.append(
            f"{prep.key:<18} {prep.prepare_mode:<12} {verdict.estimated_prompt_tokens:>8} "
            f"{'—':>9} {verdict.budget_tokens:>7} {verdict.headroom_tokens:>+9} "
            f"{verdict.selection.selected_bytes:>9} {len(verdict.selection.files):>6}  "
            f"{'ok' if verdict.context_ok else 'REJECT'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel(prep: BugPrep, tree_name: str) -> str:
    """Tree path relative to the batch directory (which Shuru mounts at /workspace)."""
    return f"{prep.root.relative_to(_batch_root(prep)).as_posix()}/{tree_name}"


def _batch_root(prep: BugPrep) -> Path:
    # worktrees/<slug>/<tree> — the batch directory is two levels up from the bug root.
    return prep.root.parent.parent


def _read_report(batch_dir: Path, stem: str) -> dict:
    path = batch_dir / f"reports/{stem}.run1.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_bug(split: Split, bugsinpy_root: Path, project: str, bug_id: int) -> BugInfo:
    try:
        return load_bug_info(bugsinpy_root, project, bug_id)
    except MetadataError as exc:
        raise PrepareError(str(exc)) from exc


def require_checkpoint(name: str) -> None:
    if not bake_mod.layer_exists(name):
        raise PrepareError(
            f"checkpoint {name!r} does not exist. Bake it first (design §11 step 3 for "
            "py3.8, §3.2 for the older interpreters); prepare runs every suite inside "
            "Shuru and will not run one on the host."
        )


def missing_checkpoints(names: list[str]) -> list[str]:
    """The selected checkpoints that do not exist, sorted.

    [VANTA-WHY] Returns the missing names instead of raising. A missing layer is
    still checked *before* any checkout, so the expensive part of ``prepare`` is
    never spent on a bug that cannot finish — but the *response* is per bug: the
    bug is flagged and the rest of the split is prepared (§3.2). A bug whose
    dependencies could not be read at bake time has no layer, and stopping the
    whole batch for it is what left a split with one unusable bug and no results
    for the others.

    The check is ``bake.layer_exists`` — the same function ``bake`` decides
    skip-vs-rebuild with. Two opinions about whether a layer exists is how this
    stage came to report "does not exist" about a layer the bake had just named.
    """
    return sorted({name for name in names if not bake_mod.layer_exists(name)})


# ---------------------------------------------------------------------------
# orchestration (§11 step 2)
# ---------------------------------------------------------------------------
def check_declared(entry: dict[str, Any], bug: BugInfo) -> list[str]:
    """The split is the protocol, so ``bug.info`` must still agree with it.

    A mismatch means either the split was edited by hand or BugsInPy's metadata
    moved under us. Both are fatal to reproducibility, and both are cheap to catch
    here rather than from a session whose ``FAIL_TO_PASS`` was derived from a
    different commit than the one the split claims.
    """
    mismatches: list[str] = []
    for key, actual in (
        ("python_version", bug.python_version),
        ("buggy_commit", bug.buggy_commit),
        ("fixed_commit", bug.fixed_commit),
        ("test_file", bug.test_file),
        ("pythonpath", bug.pythonpath),
    ):
        if entry.get(key) != actual:
            mismatches.append(
                f"split declares {key}={entry.get(key)!r}, BugsInPy metadata says {actual!r}"
            )
    return mismatches


def check_build_declared(entry: dict[str, Any], bug: BugInfo, *, repo: Path) -> list[str]:
    """``needs_build`` has to agree with the tree it is about, **in both directions**.

    [VANTA-WHY] This replaced a one-way check: "BugsInPy sets ``pythonpath``, so the
    split must say ``needs_build``". That direction was never the one that mattered —
    it only ever caught tqdm and cookiecutter, where BugsInPy's own metadata already
    said the package was imported out of a build directory. It said nothing about the
    trees that compile their own extensions, which is how the dev split came to
    declare ``needs_build: false`` for matplotlib/1, pandas/1 and spacy/4: their tests
    import a package that does not exist until the tree has been built, so the
    dependency gate's import probe would have failed with a message about a package
    that is not missing.

    The rule is :func:`bake.needs_build` — ``pythonpath`` pointing into the tree, or
    ``ext_modules`` *and* extension sources of the tree's own — and it is read from
    the **buggy commit**, because that is the tree the suite runs against
    (``oracle_tree`` is the buggy source with the fixed commit's test files). A wrong
    flag is fatal here and nowhere earlier: the split is the protocol, and every run's
    provenance carries this value. Flagged, not raised on (§3.2), so one mis-declared
    bug does not stop the rest of the batch.
    """
    declared = bool(entry.get("needs_build"))
    try:
        needed, why = bake_mod.needs_build_at(
            repo, bug.buggy_commit, pythonpath=bug.pythonpath
        )
    except (tree_mod.TreeError, bake_mod.BakeError) as exc:
        return [f"needs_build: the buggy tree could not be read: {exc}"]
    if declared == needed:
        return []
    return [
        f"split declares needs_build={declared!r} but the tree at {bug.buggy_commit[:12]} "
        f"says {needed!r}: {why}"
    ]


def prepare_one(
    entry: dict[str, Any],
    *,
    split: Split,
    bugsinpy_root: Path,
    repo_cache: Path,
    batch_dir: Path,
    deps_dir: Path,
    full: bool,
    n_ctx: int,
    n_ctx_source: str,
    vanta_bin: Path,
) -> BugPrep:
    """Everything ``prepare`` does for one bug, in the order §1.3 fixes.

    ``n_ctx`` is required and threaded into both prechecks: the gate is only worth
    running if it measures the window the *run* will use, and the caller resolves
    that once for the whole batch (§1.6.1) rather than each bug guessing.

    ``vanta_bin`` is the ``vanta_m3`` the batch's rows will measure: the gate
    shells out to it (§1.6.2 item 5), so the selection it measures is the run's
    own, not a second walk that only a test ever claimed agreed with it.
    """
    project = entry["project"]
    bug_id = int(entry["bug_id"])
    bug = load_bug(split, bugsinpy_root, project, bug_id)
    prep = BugPrep(
        bug=bug,
        root=batch_dir / "worktrees" / f"{project}_{bug_id}",
        role=entry.get("role", "pilot"),
        wave=int(entry["wave"]),
        test_cmd=[str(part) for part in entry["test_cmd"]],
        prepare_mode=FULL if full else MEASURE_ONLY,
    )

    mismatches = check_declared(entry, bug)
    # `needs_build` is checked against the buggy commit's own tree and not against the
    # metadata: it is a property of the files (§3.2), and the split is the protocol, so
    # a flag that disagrees with the tree is a mismatch like any other. Read through the
    # clone, so it runs before any tree is materialised.
    mismatches += check_build_declared(entry, bug, repo=repo_cache)
    if mismatches:
        prep.gate_failures.extend(mismatches)
        return prep

    prep.test_paths = build_trees(bug, repo_cache, prep.root)

    # Cheap gate first. The pre-strip tree is the same source as the final one
    # (the strip only removes test functions), so measuring it costs nothing and
    # saves the three double suite runs on a bug that cannot fit anyway — which is
    # the whole point of the precheck for a candidate like youtube-dl.
    early = ctx.precheck(
        prep.root / "oracle_tree",
        "",
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source,
        vanta_bin=vanta_bin,
    )
    if not early.context_ok:
        prep.context = early
        return prep

    if not full:
        prep.context = early
        return prep

    checkpoint = entry["checkpoint"]
    # The dependency gate comes first, and comes before the derivation, because a
    # missing dependency must never be reported as an oracle failure (§3.3). It
    # needs no oracle work at all: the layer is baked, the trees are on disk.
    try:
        deps = bake_mod.load_deps(deps_dir, checkpoint)
        prep.deps = deps
    except bake_mod.BakeError as exc:
        prep.gate_failures.append(f"deps: {exc}")
        return prep

    # Every reason the gate produces from here on names the pins this layer dropped,
    # when it dropped any (§3.2, ``dropped_pin_note``): the gate is what refutes the
    # prediction that the pin was not needed, and a reason that does not say which pins
    # were taken out leaves a reader unable to tell a broken bug from a wrong prediction.
    # Applied at every exit rather than at each `append`, because two of the steps below
    # record their own reasons deep inside themselves.
    def note_drops() -> None:
        suffix = dropped_pin_note(deps)
        if not suffix:
            return
        prep.gate_failures = [
            reason if reason.endswith(suffix) else reason + suffix
            for reason in prep.gate_failures
        ]

    try:
        collected = check_deps(
            prep, checkpoint=checkpoint, batch_dir=batch_dir, deps=deps
        )
    except suite_mod.SuiteError as exc:
        prep.gate_failures.append(str(exc))
        note_drops()
        return prep

    try:
        derive_hidden(
            prep, checkpoint=checkpoint, batch_dir=batch_dir, collected=collected
        )
    except (suite_mod.SuiteError, tree_mod.TreeError) as exc:
        prep.gate_failures.append(f"hidden-oracle derivation failed: {exc}")
        note_drops()
        return prep
    note_drops()

    try:
        strip_and_baseline(prep, checkpoint=checkpoint, batch_dir=batch_dir)
    except suite_mod.SuiteError as exc:
        prep.gate_failures.append(f"post-strip suite run failed: {exc}")
        note_drops()
        return prep
    note_drops()

    # The oracle's own validity check (§11 step 2). It runs before the issue text and
    # the authoritative context measurement because a bug refused here must not be
    # measured as a candidate; those two numbers would only describe a bug no run will
    # ever use.
    try:
        if not selfcheck_dev_fix(prep, batch_dir=batch_dir, checkpoint=checkpoint):
            note_drops()
            return prep
    except suite_mod.SuiteError as exc:
        prep.gate_failures.append(f"developer-fix self-check failed to run: {exc}")
        note_drops()
        return prep

    build_issue(prep, repo_cache=repo_cache, batch_dir=batch_dir)

    # Authoritative measurement: the tree Vanta will actually be given, with the
    # issue text it will actually be given. Smaller than `early`, never larger.
    prep.context = ctx.precheck(
        prep.root / "vanta_tree",
        prep.issue.text if prep.issue else "",
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source,
        vanta_bin=vanta_bin,
    )
    return prep


def derive_verdict(prep: BugPrep) -> dict[str, Any]:
    """Belongs in ``split.DERIVED_BUG_KEYS`` — nothing here is hand-authored."""
    issue_text = prep.issue.text if prep.issue else None
    return {
        "prepare_mode": prep.prepare_mode,
        "bug_patch_sha256": sha256_text(prep.bug.bug_patch_txt),
        "test_paths_from_fixed": prep.test_paths,
        "fail_to_pass": sorted(prep.fail_to_pass),
        "f2p_undeclared": sorted(prep.f2p_undeclared),
        "f2p_undeclared_reason": prep.f2p_undeclared_reason,
        "pass_to_pass_count": prep.pass_to_pass_count,
        "flaky_nodeids": sorted(prep.flaky_nodeids),
        "preexisting_failures": sorted(prep.preexisting_failures),
        "native_failure_baseline": sorted(prep.native_failure_baseline),
        "native_green_on_vanta_tree": prep.native_green_on_vanta_tree,
        "native_skips_buggy": prep.native_skips_buggy,
        "native_skips_fixed": prep.native_skips_fixed,
        "hidden_files_deleted": prep.hidden_files_deleted,
        "partially_hidden_functions": prep.partially_hidden_functions,
        "gate_failures": prep.gate_failures,
        "prompt_tokens_estimate": prep.context.estimated_prompt_tokens
        if prep.context
        else None,
        "context_ok": bool(prep.context and prep.context.context_ok),
        "context_reasons": prep.context.reasons if prep.context else ["not measured"],
        "issue_text": issue_text,
        "issue_text_sha256": sha256_text(issue_text) if issue_text is not None else None,
        "issue_ok": bool(prep.issue and prep.issue.ok),
        "issue_needs_decision": (
            prep.issue.needs_decision
            if prep.issue
            else "issue text was not built (context-rejected or measurement-only)"
        ),
        "deps": prep.deps,
    }


def render_summary(
    preps: list[BugPrep],
    *,
    n_ctx: Optional[int] = None,
    n_ctx_source: Optional[str] = None,
) -> str:
    """The ``prepare`` console + ``prepare_summary.md`` table.

    The window is printed above the table (§1.6.1) because every ``context tok``
    below it is meaningless without it: the first full prepare sized against the
    65536 fallback and rejected two PySnooper bugs, and nothing in the table said so.
    """
    lines: list[str] = []
    if n_ctx is not None:
        lines += [
            f"Context window: n_ctx={n_ctx} (source: {n_ctx_source or 'unknown'}) — "
            "every verdict below was sized against this window.",
            "",
        ]
    lines += [
        "| bug | role | wave | mode | verdict | F2P | collected | native fails | context tok | issue |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for prep in preps:
        if prep.gate_failures:
            verdict = "**gate**"
        elif prep.prepare_mode != FULL:
            verdict = "measured"
        elif prep.context and not prep.context.context_ok:
            verdict = "**context**"
        elif prep.issue and not prep.issue.ok:
            verdict = "**issue**"
        else:
            verdict = "ok"
        collected = prep.pass_to_pass_count + len(prep.native_failure_baseline)
        lines.append(
            "| {key} | {role} | {wave} | {mode} | {verdict} | {f2p} | {collected} | {fails} | "
            "{tokens} | {issue} |".format(
                key=prep.key,
                role=prep.role,
                wave=prep.wave,
                mode=prep.prepare_mode,
                verdict=verdict,
                f2p=len(prep.fail_to_pass),
                collected=collected if prep.prepare_mode == FULL else "—",
                fails=len(prep.native_failure_baseline)
                if prep.prepare_mode == FULL
                else "—",
                tokens=prep.context.estimated_prompt_tokens if prep.context else "—",
                issue="ok"
                if prep.issue and prep.issue.ok
                else ("**decision**" if prep.issue else "—"),
            )
        )
    lines.append("")
    for prep in preps:
        for failure in prep.gate_failures:
            lines.append(f"- **{prep.key}**: {failure}")
    for prep in preps:
        if prep.f2p_undeclared:
            lines.append(
                f"- **{prep.key}**: hidden and not graded — {len(prep.f2p_undeclared)} "
                "flipping test(s) BugsInPy does not declare for this bug, removed from "
                f"FAIL_TO_PASS and from vanta_tree: {prep.f2p_undeclared}"
            )
    lines.append("")
    return "\n".join(lines)
