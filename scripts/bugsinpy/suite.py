"""Run a project's suite inside Shuru and read pytest's JSON report back.

[VANTA-WHY] Every suite run in this harness happens in the microVM, including
``prepare``'s — BugsInPy source is third-party code and pytest imports it. C3 has
no exception for "just measuring" (design C3, §11 note).

[VANTA-WHY] Every suite is run **twice** (§1.3 step 3). A nodeid whose outcome
differs between the two runs is flagged flaky and excluded from both
``FAIL_TO_PASS`` and ``PASS_TO_PASS``. Without the second run a flaky test can be
selected as the oracle, at which point the whole bug's ground truth is noise — and
the noise is invisible, because a flaky failure looks exactly like a real one.

[VANTA-WHY] ``PYTHONDONTWRITEBYTECODE=1`` and ``-p no:cacheprovider``: the tree is
mounted read-write from the host, so ``__pycache__`` and ``.pytest_cache`` created
inside the VM land in the host tree, where they would be picked up by Vanta's
source walk (``__pycache__`` is skipped, but ``.pytest_cache`` is not) and change
what the Creator is shown.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import shuru

FAILURE_OUTCOMES = frozenset({"failed", "error"})

SKIPPED_OUTCOME = "skipped"
"""The outcome that is neither a failure nor a pass.

[VANTA-WHY] ``prepare``'s gate is built on :data:`FAILURE_OUTCOMES`, so a graded test
that *skips* is absent from both ``F_before`` and ``F_after`` and the flip the gate
measures cannot see it (:data:`prepare.GRADED_SKIP_REASON`); ``oracle.score``, reading
the same report, counts it as not passing. Named once here so the two readers cannot
disagree about which string they mean.
"""

SIGKILL_EXIT = 137
"""``128 + 9`` — the shell's code for a process the kernel ended with ``SIGKILL``.

[VANTA-WHY] This is how a suite killed for memory ends. The kernel's OOM killer sends
the signal, and the code is what the wrapper's ``set +e`` script records
(:func:`_exit_code`), so the state survives the kill and can be named *without* booting
anything. Measured on pandas/1: the whole ``pandas/tests`` run in a 2048 MB VM was
killed at ~61 % and the harness read the empty report as "there is no oracle".
pytest's own exit codes stop at 5, so 137 can only have come from a signal — which is
what makes "likely out of memory" a reading of the evidence rather than a guess.
"""

APPLY_APPLIED = "applied"
APPLY_REJECTED = "rejected"
APPLY_EXEC_FAILURE = "exec_failure"
APPLY_MISSING_PATCH = "missing_patch"
"""What happened when a patch was applied — the four classes ``apply_ok`` cannot tell apart.

[VANTA-WHY] M3 item B6. ``apply_ok = false`` said only "the tree does not carry this
patch", and three unrelated histories share that flag: ``git apply`` ran and **refused**
the diff (:data:`APPLY_REJECTED` — a real rejection, and for a ``MaxAttemptsReached`` run
a measured repair failure), there was no patch to apply at all
(:data:`APPLY_MISSING_PATCH`), and the apply command never completed — a VM that failed to
boot, an interpreter that died, a timeout (:data:`APPLY_EXEC_FAILURE`). Only the first is a
fact about the patch; the other two are facts about the harness, and reading them as a
Creator failure credits Vanta (or blames it) for an experiment that did not run. Named
once here because :func:`git_apply_patch` is the one function that can see ``git apply``'s
exit status, and ``oracle.grade`` is the one reader that turns it into a label.
"""


class SuiteError(RuntimeError):
    """The suite could not be run or its report could not be read."""


@dataclass
class SuiteRun:
    """One execution of the whole suite."""

    outcomes: dict[str, str] = field(default_factory=dict)
    exit_code: Optional[int] = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    report_found: bool = False
    report_error: str = ""
    """Why an existing report file could not be read, or ``""``.

    [VANTA-WHY] A report that is *there* and unreadable is a third state, not a missing
    one: ``_load_run`` used to let :func:`parse_report` raise on it, which turned a
    truncated or corrupt JSON — the shape a killed, mid-write run leaves — into an
    exception out of the whole bug rather than a named refusal. Kept as text so the
    reason can quote what the reader said.
    """

    secs: Optional[float] = None
    """Wall-clock seconds the pytest invocation itself took, measured IN the VM.

    [VANTA-WHY] M3 item D1. prepare has to record how long the native suite takes,
    because the in-session native gate caps the same suite at Rust
    ``harness::test_timeout_secs()`` (120 s from ``VANTA_TEST_TIMEOUT_SECS``) while
    prepare's own runs use 1800 s — measured on pandas/1, whose native run takes
    ~487 s, so the gate could never finish and the row ended ``native_vacuous``
    for a reason that was the CAP and not the tree (matplotlib/1's ~103 s was
    already at the edge). The cap handed to the binary is derived from THIS
    number, so it has to be the pytest invocation alone: the two ``date +%s``
    markers are emitted inside the VM around the command, excluding the VM boot
    and the pre-compile step, which the in-session cap does not cover either.
    ``None`` is a run that recorded no timing — every row before D1, and any run
    whose markers were lost.
    """

    @property
    def failures(self) -> set[str]:
        return {n for n, o in self.outcomes.items() if o in FAILURE_OUTCOMES}

    @property
    def collected(self) -> set[str]:
        return set(self.outcomes)


@dataclass
class SuitePair:
    """The two runs, plus what differs between them."""

    first: SuiteRun
    second: SuiteRun
    flaky_nodeids: set[str] = field(default_factory=set)
    missing_from_one_run: set[str] = field(default_factory=set)

    @property
    def stable_failures(self) -> set[str]:
        """Failures in both runs — the set a gate or a baseline should use."""
        return (self.first.failures & self.second.failures) - self.flaky_nodeids

    @property
    def stable_outcomes(self) -> dict[str, str]:
        return {
            n: o
            for n, o in self.first.outcomes.items()
            if n not in self.flaky_nodeids
            and n not in self.missing_from_one_run
            and self.second.outcomes.get(n) == o
        }

    @property
    def stable_collected(self) -> set[str]:
        return set(self.stable_outcomes)

    @property
    def max_run_secs(self) -> Optional[float]:
        """The slower of the two runs, in seconds, or ``None`` when neither timed.

        [VANTA-WHY] M3 item D1. The cap the in-session gate needs must cover the
        SLOWER run of the pair: the two runs are the same suite, but the first is
        cold and the second warm, so the number a per-bug cap has to clear is the
        larger one. Reading only the first run would under-size the cap by
        however much the cold run cost.
        """
        timed = [s for s in (self.first.secs, self.second.secs) if s is not None]
        return max(timed) if timed else None


def compare_runs(first: SuiteRun, second: SuiteRun) -> SuitePair:
    """Diff two runs of the same suite: flaky, missing, and demonstrably stable.

    Split out of ``run_suite_twice`` so the rule can be exercised without booting
    a VM. This rule is what stops a flaky test from becoming the oracle (§1.3 step
    3) — the one failure that would make every derived label for a bug noise — and
    a version of it that could only be tested by running the whole corpus would be
    a silent single point of failure.
    """
    pair = SuitePair(first=first, second=second)
    for nodeid in set(first.outcomes) | set(second.outcomes):
        a = first.outcomes.get(nodeid)
        b = second.outcomes.get(nodeid)
        if a is None or b is None:
            pair.missing_from_one_run.add(nodeid)
        elif a != b:
            pair.flaky_nodeids.add(nodeid)
    return pair


NATIVE_TIMEOUT_FLOOR_SECS = 120
"""The in-session native gate's cap before D1 — the floor a derived cap never goes below.

[VANTA-WHY] The Rust side's ``harness::test_timeout_secs()`` defaults to 120 s, and
that default is what D1 exists to replace with a per-bug number. Keeping it as the
FLOOR means the rule can only ever raise a cap: a bug whose suite is fast keeps the
old 120 s, and one whose suite cannot finish in 120 s gets room to finish.

It is the FALLBACK for :func:`native_timeout_floor`, not the floor itself (M3 item
F4): a run whose environment raises ``VANTA_TEST_TIMEOUT_SECS`` raises the floor with
it, so the two sides of the process boundary read one number.
"""

NATIVE_TIMEOUT_ENV = "VANTA_TEST_TIMEOUT_SECS"
"""The variable the Rust binary's own cap floor is read from.

[VANTA-WHY] M3 item F4 — ``harness::test_timeout_secs()`` resolves the cap the binary
uses (this variable, else 120 s), and ``driver.rs::resolve_native_timeout_secs``
refuses a ``--native-timeout-secs`` below it. The harness derives the value it sends
from prepare's measurement, so it has to measure against the SAME variable the binary
enforces, or the flag is refused before the run starts. Named once here so the reader,
the helper and the tests cannot drift apart.
"""


def native_timeout_floor(env: Mapping[str, str]) -> int:
    """The floor a native cap may not go below, from the run's EFFECTIVE environment.

    [VANTA-WHY] M3 item F4. The binary resolves ``harness::test_timeout_secs()`` from
    the environment it is given (:data:`NATIVE_TIMEOUT_ENV`, else
    :data:`NATIVE_TIMEOUT_FLOOR_SECS`) and refuses a ``--native-timeout-secs`` below
    its own floor. The harness sends ``max(floor, 3 × measured)``. If it measured
    against the fixed 120 while the operator's environment raised the binary's cap —
    ``VANTA_TEST_TIMEOUT_SECS=300``, exported by the batch or frozen by item 19's
    ``[env]`` — a bug whose suite is short or unmeasured sends 120, the binary refuses
    the flag, and the job ends ``infra_failure`` with no ``run.summary``: a fault
    invented by two readings of ONE number disagreeing. Reading the floor from the
    environment the child actually receives (:func:`runner.vanta_env`'s result) makes
    that impossible.

    The parse mirrors the Rust ``u64`` parse exactly — an optional leading ``+`` and
    ASCII digits only, kept only when positive — so a value one side accepts is not a
    different value on the other. Unset, unparsable, zero and negative all fall back
    to :data:`NATIVE_TIMEOUT_FLOOR_SECS`.
    """
    raw = env.get(NATIVE_TIMEOUT_ENV)
    if raw is None or not re.fullmatch(r"\+?[0-9]+", str(raw)):
        return NATIVE_TIMEOUT_FLOOR_SECS
    secs = int(raw)
    return secs if secs > 0 else NATIVE_TIMEOUT_FLOOR_SECS


def native_timeout_secs(native_secs: Optional[float], *, floor: int) -> int:
    """The wall-clock cap for a full native suite, from prepare's own measurement.

    [VANTA-WHY] M3 item D1. The rule is "3 × the measured run, never below the value
    already in use". A cap that is too small does not merely fail slowly: it makes the
    suite collect nothing, the native gate go vacuous and the row end ``infra_failure``
    — a fact about the CAP reported as a fault of the machine or the tree.
    Measured on pandas/1: the native run takes ~487 s against a 120 s in-session cap,
    so the gate could never finish there; matplotlib/1's ~103 s was already inside the
    margin.

    One rule, one helper, applied everywhere a full native suite runs — the in-session
    gate (floor :func:`native_timeout_floor`, which is the environment's own cap), the
    replay columns and the oracle grading (floor 1800) — so the three can never disagree
    about how long a bug's suite may take. ``floor`` is the caller's CURRENT cap, so the
    rule can only raise a cap, never lower one that was already large enough;
    ``native_secs is None`` (a row prepared before D1, or a run whose timing was lost)
    returns ``floor`` unchanged.
    """
    if native_secs is None:
        return floor
    return max(floor, math.ceil(3 * native_secs))


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def build_command(test_cmd: Sequence[str], report_guest_path: str) -> list[str]:
    substituted = [str(part).replace("{report}", report_guest_path) for part in test_cmd]
    if not any(report_guest_path in part for part in substituted):
        raise SuiteError(
            "test_cmd has no {report} placeholder; the JSON report would then be "
            "written to a path the harness cannot read back"
        )
    return substituted


def _shell_quote(parts: Sequence[str]) -> str:
    out = []
    for part in parts:
        if all(c.isalnum() or c in "-_./=:{}" for c in part):
            out.append(part)
        else:
            out.append("'" + part.replace("'", "'\\''") + "'")
    return " ".join(out)


def precompile_lines(
    precompile: Sequence[str], test_cmd: Sequence[str]
) -> list[str]:
    """The shell lines that run in every suite VM *before* pytest, one per rule.

    [VANTA-WHY] ``precompile`` is the split's recorded per-project rule (matplotlib/1:
    ``lib/matplotlib/widgets.py``). Its whole job is ordering: an invalid escape
    sequence is reported while the module is *compiled*, and matplotlib's pytest
    configuration escalates warnings to errors, so the first import of that module
    during collection dies as ``SyntaxError``. Compiling first leaves a valid ``.pyc``
    and nothing is compiled under pytest's filters at all.

    Three choices are load-bearing:

    * the interpreter is ``test_cmd[0]`` — the one about to run pytest. A ``.pyc`` is
      only read by the interpreter whose magic number it carries, so compiling with
      any other interpreter would leave a file pytest ignores and the warning would
      fire exactly as before.
    * ``-f`` forces the write. ``compileall`` otherwise skips a file whose ``.pyc``
      already records the source's mtime, and that check is second-resolution on the
      mtime alone — a patch applied in the same second as the previous compile would
      leave a stale ``.pyc`` and the warning would come back. Forcing removes the
      window entirely; the cost is one file's compile per VM run.
    * the lines are *newline*-joined, not ``&&``-joined: a compile that failed must
      not take pytest's report with it. The shell continues, pytest reports the
      compile error itself as a collection failure, and the gate reads the honest
      message instead of "no report at all".

    Returns an empty list when the row records nothing, so a bug without the rule runs
    exactly the script it ran before this key existed.
    """
    if not precompile:
        return []
    python = str(test_cmd[0]) if test_cmd else "python"
    return [_shell_quote([python, "-m", "compileall", "-q", "-f", *precompile])]


def run_suite_twice(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    report_rel_stem: str,
    disk_mb: int,
    timeout_secs: int = 1800,
    extra_env: Optional[dict[str, str]] = None,
    memory_mb: Optional[int] = None,
    precompile: Sequence[str] = (),
) -> SuitePair:
    """Run ``test_cmd`` twice inside the checkpoint and diff the two reports."""
    reports_dir = batch_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rel1 = f"{report_rel_stem}.run1.json"
    rel2 = f"{report_rel_stem}.run2.json"
    for rel in (rel1, rel2):
        (batch_dir / rel).unlink(missing_ok=True)

    cmd1 = _shell_quote(build_command(test_cmd, f"/workspace/{rel1}"))
    cmd2 = _shell_quote(build_command(test_cmd, f"/workspace/{rel2}"))

    script = "\n".join(
        [
            "set +e",
            *precompile_lines(precompile, test_cmd),
            # [VANTA-WHY] M3 item D1 — the wall-clock markers. `T0` opens the clock
            # immediately before the pytest invocation and `T1` closes it
            # immediately after, with `__VANTA_EXIT_n__` left in between so the
            # exit code is still `$?` of the pytest command and not of the
            # marker (`echo …=$(date +%s)` would consume it). Measuring here, in
            # the VM and around the command alone, is what makes the figure
            # comparable with the in-session cap, which also covers pytest only —
            # a number that included the boot would over-size every derived cap.
            "echo __VANTA_T0_1__=$(date +%s)",
            cmd1,
            "echo __VANTA_EXIT_1__=$?",
            "echo __VANTA_T1_1__=$(date +%s)",
            "echo __VANTA_T0_2__=$(date +%s)",
            cmd2,
            "echo __VANTA_EXIT_2__=$?",
            "echo __VANTA_T1_2__=$(date +%s)",
            "exit 0",
        ]
    )

    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)

    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["sh", "-c", script],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env,
    )

    first = _load_run(batch_dir / rel1, result, suffix="1")
    second = _load_run(batch_dir / rel2, result, suffix="2")
    pair = compare_runs(first, second)

    _clean_tree_noise(batch_dir / tree_rel)
    return pair


def collect_command(
    test_cmd: Sequence[str], ignore: Sequence[str] = ()
) -> list[str]:
    """``test_cmd`` reduced to a collection-only run — the collectability gate (§3.4).

    Split out of ``collect_only`` because the bake needs the *same* command: the
    per-bug layer's collect check must not be a second interpretation of
    ``test_cmd`` that can drift from the one prepare runs.

    ``ignore`` names test files the gate is to set aside — the recorded
    collect-flip files (see :func:`prepare.detect_collect_flip`), whose module-level
    import error on the buggy tree *is* the oracle rather than a broken environment.
    ``pytest --ignore`` keeps the file out of collection entirely, so the gate's pass
    is proven by a clean exit rather than inferred from a filtered error list.
    """
    return [
        str(part) for part in test_cmd if not str(part).startswith("--json-report")
    ] + ["--collect-only", "-q", "-p", "no:cacheprovider"] + [
        f"--ignore={path}" for path in ignore
    ]


def parse_collected(stdout: str) -> set[str]:
    """``-q`` collection output → nodeids. One nodeid per line, no spaces."""
    nodeids: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if "::" in line and " " not in line:
            nodeids.add(line)
    return nodeids


_COLLECT_ERROR_HEADER = re.compile(r"ERROR collecting (\S+)")
"""pytest's per-file collection-error heading, from its ``ERRORS`` section.

[VANTA-WHY] Read from the heading rather than from the short-summary line: only the
heading is followed by the traceback, and the traceback's own ``E   …`` line is the
*error type* the collect-flip record has to name. The short summary (``ERROR
tests/test_x.py``) is matched as the fallback below, so a pytest whose heading
format moved still yields the file, just without the type.
"""

_COLLECT_ERROR_TEXT = re.compile(r"^E\s+(\S.*)$")
"""The ``E   ImportError: …`` line — pytest's own one-line statement of the error."""

_COLLECT_ERROR_SUMMARY = re.compile(r"^ERROR\s+(\S+\.py)\b")
"""The short-summary ``ERROR tests/test_x.py`` line — the fallback, and the only
form available on a pytest old enough to omit the heading."""


def parse_collection_errors(output: str) -> dict[str, str]:
    """A collection run's output → ``{test file: the error's own line}``.

    [VANTA-WHY] The collect-flip rule (§3.4) is a statement about *which* files
    error: a file that errors here and collects cleanly on the fixed tree is the
    oracle, and a file that errors on both trees is a broken environment. A reader
    that returned only "there were errors" cannot tell those apart, and one that
    returned only the first (which is what :func:`collected_or_raise` deliberately
    does for the gate) cannot see a second hidden file. So this returns the whole
    map, file → error, and leaves the judgement to the caller.
    """
    errors: dict[str, str] = {}
    current: Optional[str] = None
    for raw in output.splitlines():
        line = raw.strip()
        heading = _COLLECT_ERROR_HEADER.search(line)
        if heading:
            current = heading.group(1)
            continue
        if current is None or current in errors:
            continue
        text = _COLLECT_ERROR_TEXT.match(line)
        if text:
            errors[current] = text.group(1).strip()
    for raw in output.splitlines():
        summary = _COLLECT_ERROR_SUMMARY.match(raw.strip())
        if summary:
            errors.setdefault(summary.group(1).split("::", 1)[0], "")
    return errors


_ERROR_LINE = re.compile(
    r"^(?:E\s+\S|ERROR\b|Traceback \(most recent call last\):|[\w.]*(?:Error|Exception)(?::|\b))"
)
"""Lines that name a failure, in pytest's own vocabulary.

``E   ImportError: cannot import name 'mini_toolbox'`` is the line that explains a
collection failure; the traceback header and pytest's ``ERROR`` summary line are
the two that point at *which* module failed. Everything else in a collection run is
progress, warnings, and the sandbox's own chatter.
"""


def failure_excerpt(output: str, *, limit: int = 25) -> str:
    """The failure lines of a run, or a bounded tail when there are none.

    [VANTA-WHY] The tail of a collection run is the wrong end of the output. On
    PySnooper/2 the real error was ``ImportError: cannot import name
    'mini_toolbox' from 'tests'``, printed in the middle; the last 800 characters
    were pytest's warnings summary plus the sandbox's "booting VM" lines, so the
    gate's message described neither the error nor the file that caused it, and the
    failure had to be re-run by hand inside the VM to be read. The excerpt is the
    error lines; the tail survives only as the fallback for output that has none.
    """
    lines = [line.rstrip() for line in output.splitlines() if _ERROR_LINE.match(line.strip())]
    if not lines:
        return f"tail: {output[-800:]!r}"
    kept = lines[:limit]
    more = f"\n    … {len(lines) - limit} more" if len(lines) > limit else ""
    return "error lines:\n" + "\n".join(f"    {line.strip()}" for line in kept) + more


def collected_or_raise(
    nodeids: set[str], *, exit_code: Optional[int], output: str
) -> set[str]:
    """A collectable suite: pytest exited cleanly and there is something to grade.

    ``exit_code`` is part of the check, not decoration. Measured on PySnooper/3: a
    test module that raises on import makes pytest collect 0 tests and exit 2 — and
    the same import failure inside *one* of several test files leaves a non-empty
    collection set. Without the exit-status check that gate would pass while the
    file holding the hidden test was never collected, and the bug would be graded
    against a suite it does not have (§3.4: "no collection errors").
    """
    if not nodeids:
        raise SuiteError(
            f"pytest collected nothing (exit {exit_code}); {failure_excerpt(output)}"
        )
    if exit_code != 0:
        raise SuiteError(
            f"pytest exited {exit_code} while collecting ({len(nodeids)} test(s) "
            f"collected): a collection error means the suite is not what the oracle "
            f"would grade; {failure_excerpt(output)}"
        )
    return nodeids


def guest_tree_root(tree_rel: str) -> str:
    """Where a tree appears inside the VM.

    One definition, so a probe can never disagree with the suites about which
    directory the suite ran in.
    """
    return f"/workspace/{tree_rel}"


@dataclass
class CollectionReport:
    """A collection run, with its errors kept rather than raised on (§3.4).

    ``nodeids`` is :func:`parse_collected`'s output and ``errors`` is
    :func:`parse_collection_errors`'s — the *whole* error map, not the first error.
    ``exit_code`` and ``output`` are kept so a caller that does decide to refuse can
    still report what pytest said, exactly as :func:`collected_or_raise` does.
    """

    nodeids: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)
    exit_code: Optional[int] = None
    output: str = ""


def collect_report(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    disk_mb: int,
    timeout_secs: int = 600,
    extra_env: Optional[dict[str, str]] = None,
    memory_mb: Optional[int] = None,
    precompile: Sequence[str] = (),
) -> CollectionReport:
    """Collect the whole suite, reporting every error instead of stopping at one.

    [VANTA-WHY] ``--continue-on-collection-errors`` is what makes this the right
    reader for the collect-flip rule and the wrong one for the gate. The gate must
    stop at the first error (§3.4, :func:`collected_or_raise`): a suite that does not
    collect is not the suite the oracle grades, and passing on a partial collection
    would grade a bug against tests it does not have. The collect-flip rule asks the
    opposite question — "are *all* of this tree's collection errors files that
    collect cleanly on the fixed tree?" — and a run that stopped at the first error
    could not answer it for a tree with two.

    Runs under the same interpreter, cwd and recorded ``sandbox_env`` as the graded
    suites, for the same reason :func:`collect_only` does (§3.4).
    """
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    if extra_env:
        env.update(extra_env)
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=[
            "sh",
            "-c",
            "\n".join(
                [
                    *precompile_lines(precompile, test_cmd),
                    _shell_quote(
                        collect_command(test_cmd) + ["--continue-on-collection-errors"]
                    ),
                ]
            ),
        ],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env,
    )
    output = f"{result.stdout}\n{result.stderr}"
    report = CollectionReport(
        nodeids=parse_collected(result.stdout),
        errors=parse_collection_errors(output),
        exit_code=result.exit_code,
        output=output,
    )
    _clean_tree_noise(batch_dir / tree_rel)
    return report


# ---------------------------------------------------------------------------
# building a tree that its suite imports out of a compiler (§3.4)
# ---------------------------------------------------------------------------
DEFAULT_BUILD_TIMEOUT_SECS = 3600
"""How long one tree's in-place build may take before it is abandoned.

Measured shapes, not padding: matplotlib's ``build_ext --inplace`` compiles the
vendored Agg and freetype trees (hundreds of translation units) and pandas builds its
Cython output, and a per-bug layer starts with a cold cache and no compiler cache at
all. It is longer than the whole-suite timeout (:func:`run_suite_twice`'s 1800 s) on
purpose: a build that runs long must be reported as a *build* failure, and a timeout
that shares the suite's number would report the two most different failures there are
in the same words.
"""


def build_tree_in_place(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    python: str,
    setup_args: Sequence[str],
    disk_mb: int,
    env: Optional[Mapping[str, str]] = None,
    timeout_secs: int = DEFAULT_BUILD_TIMEOUT_SECS,
    memory_mb: Optional[int] = None,
) -> None:
    """``python setup.py <setup_args>`` in the tree, inside the layer's VM.

    [VANTA-WHY] The failure this exists for: matplotlib/1's oracle tree was never
    built, so ``lib/matplotlib/tests/conftest.py`` died with ``ImportError: cannot
    import name 'ft2font' from partially initialized module 'matplotlib'`` — the C
    extension is compiled from ``src/ft2font.cpp`` into the package and nothing had
    compiled it. The generator is not the tree: it is a git export of the bug's commit,
    and a project whose package has C extensions is not importable until
    ``setup.py`` has run. The gate then reported it as ``deps: collection failed``,
    sending a reader after a missing dependency that was never missing.

    ``setup_args`` is the command, and the caller chooses it (:func:`prepare.setup_args_for`):
    there is one tree and two shapes of "built". ``env`` is the layer's own compiler
    environment — ``bake.COMPILER_ENV`` for a minor that installed a toolchain — and it
    has to be passed here rather than assumed: a checkpoint is a filesystem, and an
    export does not survive into a layer baked ``--from`` another
    (``bake.build_env_lines``), so without it the build runs the compiler the
    interpreter *recorded*, which is the machine it was built on.

    The build writes into the host tree (the batch directory is mounted ``rw``), so the
    compiled objects are there for every suite that runs afterwards.

    Raises :class:`SuiteError`, whose message names the tree and carries the error lines
    (:func:`failure_excerpt`), so a build failure is reported as itself and never as the
    collection it would otherwise have caused.
    """
    command = [python, "setup.py", *setup_args]
    env_full = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if env:
        env_full.update(env)
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["sh", "-c", _shell_quote(command)],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env_full,
    )
    where = guest_tree_root(tree_rel)
    if result.timed_out:
        raise SuiteError(
            f"build timed out after {timeout_secs}s in {where} "
            f"(`{_shell_quote(command)}`); {failure_excerpt(result.stdout + result.stderr)}"
        )
    if not result.ok:
        raise SuiteError(
            f"build failed in {where}: `{_shell_quote(command)}` exited "
            f"{result.exit_code}; {failure_excerpt(result.stdout + result.stderr)}"
        )


def collect_only(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    disk_mb: int,
    timeout_secs: int = 600,
    extra_env: Optional[dict[str, str]] = None,
    ignore: Sequence[str] = (),
    memory_mb: Optional[int] = None,
    precompile: Sequence[str] = (),
) -> set[str]:
    """``pytest --collect-only -q`` — the collectability gate (§3.4).

    Normalising every bug to pytest is what lets one report parser cover a
    corpus where half the projects ship ``unittest`` suites; pytest collects
    ``unittest.TestCase`` subclasses, ``unittest``'s own runner does not produce
    a JSON report we can read. If a bug does not collect here it is rejected from
    the split rather than run and failed.

    ``extra_env`` is the run's recorded ``sandbox_env`` (the project's own suite
    environment, e.g. ``PYTHONPATH=./docs_src`` for fastapi). It is passed so the
    gate collects under the *same* import path the graded suites run under: a gate
    that collected without it would reject a bug whose tests the full suite can
    run, and a gate that added import paths the suite does not have would accept
    one whose suite then dies on ``ModuleNotFoundError``.

    ``ignore`` names the recorded collect-flip files, and it is the *only* reason
    this gate may pass on a tree that has a collection error. It is not a filter
    over the error list: ``--ignore`` keeps those files out of collection
    altogether, so the exit is genuinely 0 and ``collected_or_raise`` still holds
    its full meaning. The caller has already proved, on both trees, that each
    ignored file errors on the buggy source and collects cleanly on the fixed one
    (:func:`prepare.detect_collect_flip`) — that error *is* the oracle there.
    """
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    if extra_env:
        env.update(extra_env)
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=[
            "sh",
            "-c",
            "\n".join(
                [
                    *precompile_lines(precompile, test_cmd),
                    _shell_quote(collect_command(test_cmd, ignore)),
                ]
            ),
        ],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env,
    )
    nodeids = collected_or_raise(
        parse_collected(result.stdout),
        exit_code=result.exit_code,
        output=f"{result.stdout} {result.stderr}",
    )
    _clean_tree_noise(batch_dir / tree_rel)
    return nodeids


def import_probe(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    module: str,
    disk_mb: int,
    timeout_secs: int = 300,
    extra_env: Optional[dict[str, str]] = None,
    memory_mb: Optional[int] = None,
    precompile: Sequence[str] = (),
) -> str:
    """``import <module>; print(<module>.__file__)``, run exactly as the suites run.

    [VANTA-WHY] The suites run ``python -m pytest`` with their cwd set to the tree
    root and the run's own environment — the interpreter's CWD prepend plus the
    recorded ``sandbox_env`` and nothing else. This probe reuses that invocation —
    same interpreter (``test_cmd[0]``), same cwd, same env, same ``extra_env`` — so
    it can never prove an import path the suites do not have. When ``extra_env`` is
    empty (the normal case) that is exactly "cwd and no ``PYTHONPATH``"; when the
    bug's project does declare one (fastapi's ``PYTHONPATH=./docs_src``), the probe
    sets the same value the suite does, because the suite's import path *is* the
    thing being checked. A probe that ran the module the suite does not import could
    pass while the suite still died on ``ModuleNotFoundError``. The version installed
    in site-packages is exactly what this must catch: a buggy copy there would be
    imported instead of the graded tree, and every patch would then grade the same.

    Returns the path the interpreter actually loaded.
    """
    if not module.isidentifier():
        raise SuiteError(
            f"import_probe was given {module!r}, which is not a Python identifier; "
            "the module name is read from the project's own setup.py"
        )
    python = str(test_cmd[0]) if test_cmd else "python"
    code = f"import {module} as _vanta_probe; print(_vanta_probe.__file__)"
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=[
            "sh",
            "-c",
            "\n".join(
                [
                    *precompile_lines(precompile, test_cmd),
                    _shell_quote([python, "-c", code]),
                ]
            ),
        ],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env,
    )
    where = next(
        (line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()),
        "",
    )
    if not result.ok or not where:
        raise SuiteError(
            f"`{python} -c` importing {module} in {guest_tree_root(tree_rel)} exited "
            f"{result.exit_code} without printing a path: "
            f"{(result.stderr + result.stdout).strip()[-800:]!r}"
        )
    return where


# ---------------------------------------------------------------------------
# report parsing
# ---------------------------------------------------------------------------
def report_tests(report: object, *, source: object = "the pytest report") -> list[dict]:
    """The per-test entries of a ``pytest-json-report`` document, in either schema.

    [VANTA-WHY] Two schemas are in play, because the plugin's release follows the
    bug's own pytest pin (§3.2): 0.1/0.2 write a **flat list**, one entry per test,
    while 0.3 and later write ``{"tests": [...]}``. The model is not what picks the
    schema — httpie/1's ``pytest==3.2.1`` is older than the plugin's first
    ``Requires-Dist`` on pytest, so only 0.2 will install there — and every reader of
    the file has to know both, or that one bug's report reads as "no tests" instead of
    as a verdict. ``source`` only names the report in the error message.

    A ``dict`` with no ``tests`` (or a valid empty one) is an empty report, not an
    error: a caller may hold ``{}`` because writing the file failed, and that is
    already the "the report says nothing" case. A top level — or a ``tests`` value —
    that is neither is a :class:`SuiteError`, because silently empty is how a real
    report turns into a wrong answer.

    One limitation of the 0.2 schema no reader can repair: it builds its list from
    the terminal reporter's ``passed`` and ``failed`` stats only, so a test that
    *errors* (the terminal reporter's ``error`` stat) is absent from the file rather
    than recorded as ``"error"``. An absent nodeid is excluded from the stable
    comparison (:func:`compare_runs`), which is the right treatment of a test the
    report says nothing about.
    """
    if isinstance(report, dict):
        tests = report.get("tests")
        if tests is None:
            return []
    elif isinstance(report, list):
        tests = report
    else:
        raise SuiteError(
            f"cannot read {source}: its top level is a {type(report).__name__}, not "
            "the mapping 0.3+ writes or the list 0.1/0.2 writes"
        )
    if not isinstance(tests, list):
        raise SuiteError(
            f"cannot read {source}: its `tests` is a {type(tests).__name__}, not the "
            "list 0.1/0.2 write"
        )
    return [test for test in tests if isinstance(test, dict)]


def parse_report(path: Path) -> dict[str, str]:
    """``{nodeid: outcome}`` from a ``pytest-json-report`` file, in either schema.

    The entries come from :func:`report_tests`, so both plugin schemas are read here
    for the same reason they are there.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"cannot read pytest report {path}: {exc}") from exc
    outcomes: dict[str, str] = {}
    for test in report_tests(data, source=f"pytest report {path}"):
        nodeid = test.get("nodeid")
        outcome = test.get("outcome")
        if isinstance(nodeid, str) and isinstance(outcome, str):
            outcomes[nodeid] = outcome
    return outcomes


def _load_run(report: Path, result: shuru.ExecResult, *, suffix: str) -> SuiteRun:
    run = SuiteRun(
        exit_code=_exit_code(result.stdout, suffix),
        timed_out=result.timed_out,
        stdout_tail=result.stdout[-4000:],
        stderr_tail=result.stderr[-2000:],
        secs=_marker_duration(result.stdout, suffix),
    )
    if report.is_file():
        try:
            run.outcomes = parse_report(report)
        except SuiteError as exc:
            # Reported, not raised: the file exists and cannot be read, which is a state
            # the run's own reader has to name (`run_state`) rather than an exception
            # that escapes the bug. See `SuiteRun.report_error`.
            run.report_error = str(exc)
        else:
            run.report_found = True
    return run


def run_state(run: SuiteRun) -> Optional[str]:
    """Why ``run`` is not a measurement, or ``None`` when it is one.

    [VANTA-WHY] A :class:`SuiteRun` with no report has empty ``outcomes``, so every set
    derived from it — :attr:`failures`, the ``F_before \\ F_after`` flip, the
    pre-existing-failure baseline — is empty for a reason that has nothing to do with
    the project's tests. Read as data, that emptiness says "no test fails". Measured on
    pandas/1: the harness refused the bug as having "no oracle" when its suite had in
    fact been killed by the OOM killer at ~61 %. A run that was killed, hit the cap, or
    left a report nothing can read measured nothing — the report's *absence* is itself
    the measurement, and `prepare` refuses the bug with this reason instead of deriving
    an oracle from the husk.

    The ending comes first, because the ending is what explains the report:
    ``timed_out`` and :data:`SIGKILL_EXIT` are the two ways a run is stopped before it
    can write one. A readable report beside a ``SIGKILL`` is still refused: pytest-json-report
    writes at session end, so a killed pytest cannot have produced it, and pytest's own
    codes stop at 5 — the file would be a stale one, not this run's.
    """
    if run.timed_out:
        return "timed out (hit the cap)"
    if run.exit_code == SIGKILL_EXIT:
        return f"killed (exit {SIGKILL_EXIT}, likely out of memory)"
    if run.report_error:
        return f"report unreadable ({run.report_error})"
    if run.report_found:
        return None
    if run.exit_code is None:
        return "wrote neither a report nor an exit status"
    return f"exited {run.exit_code} without a report"


def _exit_code(stdout: str, suffix: str) -> Optional[int]:
    marker = f"__VANTA_EXIT_{suffix}__="
    for line in stdout.splitlines():
        if line.startswith(marker):
            try:
                return int(line[len(marker) :].strip())
            except ValueError:
                return None
    return None


def _marker_time(stdout: str, name: str, suffix: str) -> Optional[int]:
    """The integer after ``__VANTA_<name>_<suffix>__=``, or ``None`` when absent.

    The reader for the ``date +%s`` markers :func:`run_suite_twice` writes around
    each pytest invocation. A run whose markers never arrived — an older script,
    a truncated stdout, a command that took the shell with it — returns ``None``
    rather than a wrong number, so a missing measurement can never be mistaken
    for a very fast run (which would derive a cap that is far too small).
    """
    marker = f"__VANTA_{name}_{suffix}__="
    for line in stdout.splitlines():
        if line.startswith(marker):
            try:
                return int(line[len(marker) :].strip())
            except ValueError:
                return None
    return None


def _marker_duration(stdout: str, suffix: str) -> Optional[float]:
    """The measured pytest wall time for ``suffix``, or ``None`` when not measurable.

    A negative or zero duration is refused as well as a missing one: ``date +%s``
    is second-resolution, and an inverted pair can only mean the markers belong to
    different runs (a mangled script), which is not a run time at all.
    """
    start = _marker_time(stdout, "T0", suffix)
    end = _marker_time(stdout, "T1", suffix)
    if start is None or end is None or end <= start:
        return None
    return float(end - start)


def _clean_tree_noise(tree: Path) -> None:
    """Remove the caches a suite run leaves in ``tree``.

    ``.hypothesis`` is hypothesis's example database. It is a *runtime* artefact of
    the run, not part of the project, and pandas/1's suite writes 145 example files
    into whichever tree it ran in — so a tree left uncleaned carries them into every
    copy taken from it (and into the tree the Creator reads). Removing it here is the
    same rule as ``__pycache__``/``.pytest_cache``: a run's leavings are not the tree.
    Called *after* both runs of a pair (:func:`run_suite_twice`), so the second run
    still sees the first run's database.
    """
    for name in (
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
    ):
        shutil.rmtree(tree / name, ignore_errors=True)


# ---------------------------------------------------------------------------
# diffing two failure sets (used by the fixed-tree gate and by the oracle)
# ---------------------------------------------------------------------------
def collect_by_file(nodeids: Sequence[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for nodeid in nodeids:
        out.setdefault(nodeid.split("::", 1)[0], set()).add(nodeid)
    return out


def git_apply_patch(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    patch_path: Path,
    disk_mb: int,
    drop_test_hunks: bool,
    memory_mb: Optional[int] = None,
    timeout_secs: int = 300,
) -> tuple[bool, str, str]:
    """Apply a patch inside the VM. Strict — no fuzzy fallback (§5.2).

    ``drop_test_hunks`` filters the diff by path before applying, so a Creator
    patch that edits a test file cannot reach the hidden test. Filtering is done
    with ``filterdiff``-free code (a small awk-free split in Python on the host)
    because the patch body is a host-side artifact; only ``git apply`` runs in the
    VM.

    Returns ``(ok, detail, outcome)``. [VANTA-WHY] M3 item B6 — ``outcome`` is the
    *class* of the result (:data:`APPLY_APPLIED` / :data:`APPLY_REJECTED` /
    :data:`APPLY_EXEC_FAILURE`), because ``ok = False`` alone cannot tell a patch
    ``git apply`` refused from a command that never ran. Only this function can
    see ``git apply``'s exit status and whether the run timed out, so the class is
    decided here and carried, rather than re-guessed from ``detail``'s text by
    every reader.
    """
    body = patch_path.read_text(encoding="utf-8")
    if drop_test_hunks:
        body = drop_test_file_hunks(body)
        if not body.strip():
            return (
                True,
                "patch contained only test hunks; nothing to apply",
                APPLY_APPLIED,
            )
    staged = batch_dir / "reports" / f"_apply_{patch_path.name}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(body, encoding="utf-8")
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["git", "apply", "--verbose", f"/workspace/{staged.relative_to(batch_dir)}"],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd=f"/workspace/{tree_rel}",
        timeout_secs=timeout_secs,
    )
    detail = (result.stdout + result.stderr).strip()[-2000:]
    if result.timed_out or result.exit_code is None:
        # The command did not finish, so nothing about the patch was learned:
        # neither an acceptance nor a refusal.
        return False, detail, APPLY_EXEC_FAILURE
    if result.exit_code != 0:
        return False, detail, APPLY_REJECTED
    return True, detail, APPLY_APPLIED


def drop_test_file_hunks(diff_text: str) -> str:
    """Remove whole ``diff --git`` sections whose target is a test file."""
    from .trees import is_test_path

    kept: list[str] = []
    current: list[str] = []
    current_is_test = False

    def flush() -> None:
        if current and not current_is_test:
            kept.extend(current)

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            current = [line]
            current_is_test = False
            parts = line.split()
            if len(parts) >= 4:
                target = parts[3]
                target = target[2:] if target.startswith("b/") else target
                current_is_test = is_test_path(target)
            continue
        current.append(line)
    flush()
    return "".join(kept)
