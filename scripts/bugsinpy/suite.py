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

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from . import budget as budget_mod
from . import shuru
from . import suite_cache

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

    def to_json(self) -> dict:
        """This run as JSON, for a cached pair (M3 item S1)."""
        return {
            "outcomes": self.outcomes,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "report_found": self.report_found,
            "report_error": self.report_error,
            "secs": self.secs,
        }

    @classmethod
    def from_json(cls, data: object) -> "SuiteRun":
        """The inverse of :meth:`to_json`.

        [VANTA-WHY] ``outcomes`` is REQUIRED: it is the run's whole measurement,
        so a payload without it is not a run at all. The scalar fields default,
        because the entry's digest and material have already proven the payload
        was written by this version of this module for this exact key — a missing
        scalar there is a shape this reader does not know, and refusing is the
        caller's job (the wrapper treats a decode failure as a corrupt entry).
        """
        if not isinstance(data, dict) or not isinstance(data.get("outcomes"), dict):
            raise ValueError("a cached run must carry an `outcomes` mapping")
        return cls(
            outcomes={str(k): str(v) for k, v in data["outcomes"].items()},
            exit_code=data.get("exit_code"),
            timed_out=bool(data.get("timed_out", False)),
            stdout_tail=str(data.get("stdout_tail") or ""),
            stderr_tail=str(data.get("stderr_tail") or ""),
            report_found=bool(data.get("report_found", False)),
            report_error=str(data.get("report_error") or ""),
            secs=data.get("secs"),
        )


@dataclass
class SuitePair:
    """The two runs, plus what differs between them."""

    first: SuiteRun
    second: SuiteRun
    flaky_nodeids: set[str] = field(default_factory=set)
    missing_from_one_run: set[str] = field(default_factory=set)
    cache: Optional[dict] = field(default=None, compare=False)
    """How this pair was obtained (M3 item S1), or ``None`` for a pair that was
    never looked up in the cache.

    [VANTA-WHY] The key, the state
    (``hit``/``miss``/``unreadable``/``unmeasured``/``off:no_cache``/
    ``off:unknown_checkpoint``), the ``created_by`` of the entry that was served
    and whether an entry existed that could not be used. ``off:unknown_checkpoint``
    (M3 item U3) is the one state with an EMPTY key: the layer has no recorded
    identity, so no key could be built for it and the cache was off entirely.
    Deliberately NOT part of :meth:`to_json`: it describes
    the LOOKUP, not the experiment, so storing it would make a served pair claim
    to have been served by whoever wrote it first. ``compare=False`` keeps it out
    of equality, so an existing test comparing two pairs still compares the two
    runs.
    """

    def to_json(self) -> dict:
        """The pair as JSON — the whole cached value (M3 item S1).

        [VANTA-WHY] BOTH runs are stored, not one: the flakiness comparison
        (``flaky_nodeids``/``missing_from_one_run``) is a reading of the PAIR, so
        a cache that stored a single run would make every later reader re-derive
        flakiness from one observation and call it stability.
        """
        return {
            "first": self.first.to_json(),
            "second": self.second.to_json(),
            "flaky_nodeids": sorted(self.flaky_nodeids),
            "missing_from_one_run": sorted(self.missing_from_one_run),
        }

    @classmethod
    def from_json(cls, data: object) -> "SuitePair":
        """The inverse of :meth:`to_json`; ``cache`` is left for the caller."""
        if not isinstance(data, dict):
            raise ValueError("a cached pair must be an object")
        return cls(
            first=SuiteRun.from_json(data["first"]),
            second=SuiteRun.from_json(data["second"]),
            flaky_nodeids=set(data.get("flaky_nodeids") or []),
            missing_from_one_run=set(data.get("missing_from_one_run") or []),
        )

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


def _run_script(
    suffix: str, cmd: str, precompile: Sequence[str], test_cmd: Sequence[str]
) -> str:
    """The shell script ONE of a pair's two runs executes in its own VM.

    [VANTA-WHY] M3 item S5. The markers keep the pair's ``1``/``2`` suffixes, so
    :func:`_exit_code` and :func:`_marker_duration` read a run exactly as they did
    when the two shared a VM — the change was which VM prints a marker, not what a
    marker is. The ``T0``/``T1`` pair still opens and closes the clock immediately
    around the pytest invocation (item D1): measuring in the VM and around the
    command alone is what makes the figure comparable with the in-session cap, and
    a number that included the boot would over-size every derived cap.

    The pre-compile runs in BOTH VMs and not only the first, because the two may
    overlap when the budget allows it (item S5). ``compileall -f`` writes each
    ``.pyc`` through ``importlib``'s atomic rename, so two identical compiles
    cannot corrupt one another; and a second run that skipped the step could race
    the first and import the module under pytest's warning filters — exactly the
    failure the step exists to prevent.
    """
    return "\n".join(
        [
            "set +e",
            *precompile_lines(precompile, test_cmd),
            f"echo __VANTA_T0_{suffix}__=$(date +%s)",
            cmd,
            # ``echo …=$(date +%s)`` consumes ``$?``, so the exit marker has to be
            # emitted before the closing time marker or the code read back would be
            # the marker's own.
            f"echo __VANTA_EXIT_{suffix}__=$?",
            f"echo __VANTA_T1_{suffix}__=$(date +%s)",
            "exit 0",
        ]
    )


# ---------------------------------------------------------------------------
# the per-VM root: a disposable clone of the tree, and nothing else (M3 V8)
# ---------------------------------------------------------------------------
VM_ROOT_DIR = ".vm_roots"
"""Where a run's private root is built — inside the batch directory.

[VANTA-WHY] M3 item V8. ``shuru run``'s ``--mount`` is a single, non-repeatable
flag (``shuru.py``): one boot sees exactly one host directory, so the ruling's
"mount the tree read-only beside a writable output directory" cannot be spelled in
one argv. The fix is to make that one directory *disposable* instead — a fresh
clone of the tree the VM is to work on, carrying nothing else the batch holds — so
the guest still reaches its tree at ``/workspace/<tree_rel>``, every report path
stays ``/workspace``-relative and byte-identical, and everything the batch holds
beside the tree (``oracle/``, other bugs' worktrees, the manifest, the suite cache)
is outside the VM's filesystem rather than merely denied.

The clone has to sit on the *same filesystem* as the tree it will replace, because
a mutating run adopts its tree back with an atomic rename (:meth:`VmRoot.adopt`),
which ``EXDEV``s across filesystems. ``batch_dir`` and its ``.vm_roots`` children
share one, so this is the only correct home for it. The guest cannot see it: the
mount root *is* the clone, so ``.vm_roots`` and every path beside it are above the
VM's filesystem root.
"""

ADOPTED_TREE_MAX_BYTES = 8 << 30
"""The largest tree a VM may hand back to be adopted (8 GiB).

Far above the largest measured build — matplotlib compiles its vendored Agg and
freetype trees — so it can only catch a runaway, never a legitimate build. A larger
one is a gate failure with the reason recorded, never a silently adopted tree.
"""

OUTPUT_MAX_BYTES = 256 << 20
"""The largest single file a run may hand back as an output (256 MiB).

A pytest JSON report for a whole project suite is a few megabytes; the cap exists
so that a "report" cannot be a way to make the host read a file that was never one.
"""


def _vm_copy(src: Path, dest: Path) -> None:
    """Copy ``src`` onto ``dest`` as an APFS clonefile, or a real copy where that fails.

    [VANTA-WHY] M3 item V8. The in-session sandbox seeds every VM this way
    (``harness::seed_sandbox_from``: ``cp -cR``, falling back to a recursive copy),
    and the reason is cost: ``clonefile(2)`` is copy-on-write, so a per-VM tree is a
    directory entry rather than a second copy of a project. A filesystem without it
    (a container overlay, a non-APFS volume) falls back to :func:`shutil.copytree`
    rather than failing the run — the isolation does not depend on which ran, only
    the cost does. ``cp -R`` keeps a symlink a symlink (macOS follows links only
    under ``-L``), which is what ``copytree(symlinks=True)`` does too.
    """
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["cp", "-cR", str(src), str(dest)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0 and dest.exists():
        return
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest, ignore_errors=True)
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
    else:
        shutil.copy2(src, dest)


def _validate_tree(tree: Path, *, max_bytes: int, where: str) -> None:
    """A tree a VM hands back must be a tree, and only a tree.

    [VANTA-WHY] M3 item V8 — the host adopts this tree, so the guest gets a say in
    what the host will afterwards read, hash and run on. Four things are refused,
    each because it turns "a directory of files" into something else:

    * a **symlink** whose target is outside the tree: the guest could point one at
      ``batch_dir/oracle`` — or anywhere else on the host — and the *host* would
      follow it while hashing or copying the adopted tree;
    * a **device, fifo or socket**: reading it is a leak or a block, and neither is
      a file in a source tree;
    * a **setuid/setgid bit**: a privilege the guest granted itself and the host
      would then carry;
    * more bytes than ``max_bytes`` (:data:`ADOPTED_TREE_MAX_BYTES`).

    A refusal raises :class:`SuiteError`. ``prepare`` records that as a gate failure
    with the reason, which is the whole point: a tree that fails validation is never
    silently replaced by the tree the VM did not touch.
    """
    if not tree.is_dir():
        raise SuiteError(f"{where} is not a directory, so it cannot replace the tree")
    root = tree.resolve()
    total = 0
    stack = [tree]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise SuiteError(f"cannot read {where} at {current}: {exc}") from exc
        for entry in entries:
            st = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                target = Path(os.path.realpath(entry.path))
                if target != root and root not in target.parents:
                    raise SuiteError(
                        f"{where} contains {entry.path}, a symlink to {target} outside "
                        "the tree; a tree may not point out of itself"
                    )
                continue
            if stat.S_ISDIR(st.st_mode):
                stack.append(Path(entry.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                raise SuiteError(
                    f"{where} contains {entry.path}, which is not a regular file"
                )
            if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
                raise SuiteError(
                    f"{where} contains {entry.path}, which carries a setuid/setgid bit"
                )
            total += st.st_size
            if total > max_bytes:
                raise SuiteError(
                    f"{where} is larger than the {max_bytes}-byte limit a tree may be"
                )


def _validate_output_file(path: Path, *, max_bytes: int, where: str) -> None:
    """A run's output must be a regular, bounded, symlink-free file before we read it.

    [VANTA-WHY] M3 item V8 — the host reads these files, so the VM that wrote them
    does not get to choose what they are. A symlink would make the "copy" read a
    path outside the clone, a device or fifo would make reading block or leak, and a
    file past the cap is not a report but a way to spend the host's memory. Refused
    loudly rather than skipped: an output that fails this is not a measurement, and
    reading it as "the suite said nothing" is the silent fallback the ruling forbids.
    """
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise SuiteError(f"{where} is a symbolic link; a run's output must be a real file")
    if not stat.S_ISREG(st.st_mode):
        raise SuiteError(f"{where} is not a regular file")
    if st.st_size > max_bytes:
        raise SuiteError(
            f"{where} is {st.st_size} bytes, past the {max_bytes}-byte cap an output "
            "may be"
        )


class VmRoot:
    """One VM's private root: a fresh clone of the tree, plus the scratch it carries.

    [VANTA-WHY] M3 item V8, and the answer to the one mount Shuru can express. See
    :data:`VM_ROOT_DIR` for why the VM's single mount is this directory. Three rules
    follow from it and are the whole contract:

    * **A clone per VM, never one per pair** — serial and parallel alike (M3 item
      V8/Q2). Each boot starts from the pristine tree's own bytes, so run 2 no longer
      inherits run 1's ``.hypothesis`` database and results do not depend on whether
      the budget let the two runs overlap. Recorded as a deliberate change: earlier
      rows were measured with the shared database.
    * **``carry`` is explicit** (:meth:`__init__`) — a build's configuration file, a
      staged patch, wheel staging. Anything the guest legitimately needs is named by
      the caller that needs it, so "what the VM can see" is a list a reader can check
      rather than "the batch directory".
    * **The host reads nothing until it has validated it** — :meth:`lift_outputs`
      for a run's files, :meth:`adopt` for a tree a mutating step changed.

    ``tree_rel`` is ``None`` for the few boots that need no tree at all (the
    interpreter probe), which still get a private, empty root rather than the batch
    directory.
    """

    def __init__(
        self,
        *,
        batch_dir: Path,
        tree_rel: Optional[str],
        carry: Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> None:
        self.batch_dir = Path(batch_dir)
        self.tree_rel = tree_rel
        self.carry = tuple(str(rel) for rel in carry)
        self.outputs = tuple(str(rel) for rel in outputs)
        self.path: Optional[Path] = None

    def __enter__(self) -> "VmRoot":
        base = self.batch_dir / VM_ROOT_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="run-", dir=str(base)))
        if self.tree_rel is not None:
            _vm_copy(self.batch_dir / self.tree_rel, self.path / self.tree_rel)
        for rel in self.carry:
            _vm_copy(self.batch_dir / rel, self.path / rel)
        for rel in self.outputs:
            # pytest-json-report opens the report it is told to write; it does not
            # create the directory, so the run's own parent has to exist in the clone
            # the same way it exists in the batch (:func:`_measure_suite`).
            (self.path / rel).parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc: object) -> bool:
        self.discard()
        return False

    @property
    def host(self) -> Path:
        """The directory to hand ``shuru.run`` as ``host_root``."""
        if self.path is None:
            raise SuiteError("this VM root was used outside its `with` block")
        return self.path

    def discard(self) -> None:
        """Delete this run's root. Called by ``__exit__``; safe to call again.

        [VANTA-WHY] M3 item X1. The ``.vm_roots`` holder itself is NOT removed.
        ``__enter__`` creates the holder and then ``mkdtemp``s inside it as two
        steps with no lock, so a sibling's ``discard`` that ``rmdir``ed the holder
        whenever it happened to be the only root left could delete it between the
        two — and ``mkdtemp`` then raised ``FileNotFoundError``. That is an
        ``OSError``, not a ``SuiteError``, so the callers that catch only
        ``SuiteError`` (:func:`prepare`) never recorded it as a gate failure and it
        escaped the bug, while the pool stored and re-raised it and stopped the whole
        batch. An empty holder left in the batch is harmless, so the removal is the
        thing to drop, not the creation to synchronise. Each run's own clone is still
        deleted here.
        """
        if self.path is None:
            return
        shutil.rmtree(self.path, ignore_errors=True)
        self.path = None

    def lift_outputs(self) -> None:
        """Copy this VM's declared outputs back to the host, after checking them.

        An output the run never wrote is skipped rather than refused: a VM that never
        booted, one the OOM killer ended and one that hit the cap all leave no report,
        and that absence is the run's own result — :func:`run_state` reads it. An
        output that *is* there must pass :func:`_validate_output_file` first.
        """
        for rel in self.outputs:
            src = self.host / rel
            if not src.exists() and not src.is_symlink():
                continue
            _validate_output_file(
                src, max_bytes=OUTPUT_MAX_BYTES, where=f"{rel} written by {self.host}"
            )
            dest = self.batch_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def adopt(self, *, max_bytes: int = ADOPTED_TREE_MAX_BYTES) -> str:
        """Swap this VM's tree in for the host's, and return the adopted tree's hash.

        [VANTA-WHY] M3 item V8/Q1. A step that *mutates* a tree — the in-place build,
        the patch apply — produces something every later step reads, so the guest's
        tree has to become the host's. It does so only after :func:`_validate_tree`,
        and the swap is two renames rather than a copy: POSIX ``rename`` cannot
        replace a non-empty directory, so the old tree steps aside first and is
        deleted once the new one is in place. If the second rename fails the old tree
        goes straight back and the error propagates — never a silent fallback to it.

        The returned hash is :func:`suite_cache.tree_fingerprint` of the adopted
        tree. Every later cache key is computed from the tree on disk at key time,
        so it is automatically the adopted one; the value is returned so ``prepare``
        can record *which* bytes that was on ``prepare.json``.
        """
        if self.tree_rel is None:
            raise SuiteError("this VM root has no tree to adopt")
        src = self.host / self.tree_rel
        _validate_tree(
            src,
            max_bytes=max_bytes,
            where=f"the tree {self.tree_rel} that {self.host} built",
        )
        dest = self.batch_dir / self.tree_rel
        replaced = dest.with_name(dest.name + ".replaced")
        if replaced.is_symlink() or replaced.is_file():
            replaced.unlink()
        elif replaced.is_dir():
            shutil.rmtree(replaced)
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            os.replace(dest, replaced)
        try:
            os.replace(src, dest)
        except OSError:
            if replaced.exists():
                os.replace(replaced, dest)
            raise
        if replaced.is_dir():
            shutil.rmtree(replaced, ignore_errors=True)
        return suite_cache.tree_fingerprint(dest)


def _run_one_vm(
    *,
    script: str,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    disk_mb: int,
    memory_mb: Optional[int],
    timeout_secs: int,
    env: Mapping[str, str],
    budget: Optional[budget_mod.Budget],
    carry: Sequence[str] = (),
    output_rel: Optional[str] = None,
) -> shuru.ExecResult:
    """Boot ONE VM for one run of a pair, holding a scheduler slot while it runs.

    [VANTA-WHY] M3 items S5/S6. Splitting the pair's single VM into two is what makes
    the runs independent, and this is the one place a run acquires its share of the
    host: ``memory_mb`` is the row's own allocation and the run takes one effective
    core (:func:`budget.Budget.slot`). The gate is held across ``shuru.run`` and
    released in its ``finally``, so a VM that fails to boot, times out or is killed
    frees the slot for the next run rather than wedging the queue.

    ``timeout_secs`` is now the cap for THIS run alone (the pair's two runs each get
    the full cap, where the shared script's cap covered both). That is the same
    reading the in-session gate makes of the same number — the cap is what one suite
    invocation may take — and it is why the derived cap (item D1) can be compared
    with it directly.

    [VANTA-WHY] M3 item V8. The VM mounts a fresh clone of the tree rather than the
    batch directory (:class:`VmRoot`), so the guest reaches its tree at
    ``/workspace/<tree_rel>`` exactly as before and everything the batch holds beside
    the tree — oracle, other bugs' worktrees, the manifest, the suite cache — is
    above the VM's filesystem root instead of merely writable-and-ignored. The clone
    is deleted when this run ends, so a run's writes into its tree are invisible to
    the next run and to the pristine tree. ``carry`` names batch-relative paths the
    run's script also needs (a build file, a staged patch); ``output_rel`` is the one
    file the run hands back, copied out and checked (:meth:`VmRoot.lift_outputs`)
    before anything reads it.
    """
    gate = budget.slot(memory_mb=memory_mb) if budget is not None else nullcontext()
    with gate:
        outputs = () if output_rel is None else (output_rel,)
        with VmRoot(
            batch_dir=batch_dir, tree_rel=tree_rel, carry=carry, outputs=outputs
        ) as root:
            result = shuru.run(
                checkpoint=checkpoint,
                host_root=root.host,
                command=["sh", "-c", script],
                disk_mb=disk_mb,
                memory_mb=memory_mb,
                guest_cwd=guest_tree_root(tree_rel),
                timeout_secs=timeout_secs,
                env=env,
            )
            root.lift_outputs()
            return result


def _run_pair(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    precompile: Sequence[str],
    disk_mb: int,
    memory_mb: Optional[int],
    timeout_secs: int,
    env: Mapping[str, str],
    rel1: str,
    rel2: str,
    cmd1: str,
    cmd2: str,
    budget: Optional[budget_mod.Budget],
    carry: Sequence[str] = (),
) -> tuple[SuiteRun, SuiteRun]:
    """Run the pair's two invocations in two separate VMs (M3 item S5).

    Each run gets its OWN :class:`shuru.ExecResult`, so ``_load_run`` reads the
    markers of the run it belongs to and nothing else — with the shared script the
    two runs' ``stdout_tail`` were the same buffer, and a failure in run 2 could
    only be told apart from one in run 1 by the suffix in the marker lines.

    The two VMs may overlap only when the budget allows it (:attr:`Budget.concurrent`).
    Without a budget — ``oracle``, ``replay`` and ``probe_xdist``'s calls, which do
    not pass one — the runs are still two VMs but strictly serial.

    [VANTA-WHY] M3 item V8/Q2. Each run gets its OWN clone of the pristine tree
    (:func:`_run_one_vm`), in serial and parallel mode alike, so run 2 never sees
    run 1's ``.hypothesis`` database and neither sees the host tree. That is a
    deliberate change from the shared-VM era, where run 2 started from the database
    run 1 had just written: the pair is what measures one tree twice, and a run whose
    starting state depends on its sibling's example cache is not two measurements of
    the same input. The two runs are therefore independent, which is what the
    pre-registration states.
    """
    script1 = _run_script("1", cmd1, precompile, test_cmd)
    script2 = _run_script("2", cmd2, precompile, test_cmd)
    common = {
        "checkpoint": checkpoint,
        "batch_dir": batch_dir,
        "tree_rel": tree_rel,
        "disk_mb": disk_mb,
        "memory_mb": memory_mb,
        "timeout_secs": timeout_secs,
        "env": env,
        "budget": budget,
        "carry": carry,
    }
    if budget is not None and budget.concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(
                _run_one_vm, script=script1, output_rel=rel1, **common
            )
            second_future = pool.submit(
                _run_one_vm, script=script2, output_rel=rel2, **common
            )
            result2 = second_future.result()
            result1 = first_future.result()
    else:
        result1 = _run_one_vm(script=script1, output_rel=rel1, **common)
        result2 = _run_one_vm(script=script2, output_rel=rel2, **common)
    return (
        _load_run(batch_dir / rel1, result1, suffix="1"),
        _load_run(batch_dir / rel2, result2, suffix="2"),
    )


# ---------------------------------------------------------------------------
# the suite cache (M3 items S1/S2/S3)
# ---------------------------------------------------------------------------
def cache_env(extra_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The environment a suite run is given, and the one its cache key hashes.

    [VANTA-WHY] One function, because the key has to describe the environment the
    run actually saw, and building the dict in two places is how a key silently
    stops covering ``extra_env``. That is not a cosmetic difference: the recorded
    per-project rule is fastapi/2's ``PYTHONPATH=./docs_src``, and the same tree
    measured 775 passed with it and 0 tests collected without it — so a key that
    missed it would serve one environment's result for another's.
    """
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)
    return env


def carry_material(batch_dir: Path, carry: Sequence[str]) -> Optional[list[dict]]:
    """The carried paths, digested, as they enter a key (:func:`cache_material`).

    [VANTA-WHY] M3 item V8. ``carry`` is an *input* to the run — ``probe_xdist``'s
    staged wheels are on the interpreter's ``PYTHONPATH`` — so two calls that differ
    only in what they carry must not share a key. A directory is hashed with
    :func:`suite_cache.tree_fingerprint` (the same digest a tree gets); a file is
    hashed by content. ``None`` when one of them cannot be read, which turns the cache
    off for the call exactly as an unreadable tree does, rather than keying on a path
    whose content was never read.
    """
    out: list[dict] = []
    for rel in carry:
        path = batch_dir / rel
        try:
            if path.is_dir():
                digest = suite_cache.tree_fingerprint(path)
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, NotADirectoryError):
            return None
        out.append({"rel": str(rel), "digest": digest})
    return out


def cache_material(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    disk_mb: int,
    timeout_secs: int,
    extra_env: Optional[Mapping[str, str]] = None,
    memory_mb: Optional[int] = None,
    precompile: Sequence[str] = (),
    carry: Sequence[str] = (),
) -> Optional[dict]:
    """The key material for one pair of runs, or ``None`` when it cannot be built.

    [VANTA-WHY] The key IS the input (M3 speed package): the exact tree bytes, the
    command, the checkpoint, the harness build, the suite's effective environment and
    the effective timeout cap. The cap is here because a run's TIMEOUT is a function
    of the cap and nothing else — a key without it would inherit one arm's timing
    out as a fact about the tree. ``None`` (an unreadable tree) turns the cache off
    for this call rather than raising: the caller is inside a run that is about to
    fail for its own reason, and a key that could raise would turn a permission
    problem into a harness crash.

    ``carry`` (M3 item V8) is the batch-relative paths a VM needs *beside* its tree,
    and it is hashed only when it is non-empty: an empty carry is the shape every
    caller but ``probe_xdist`` has, and folding an empty list into the rendering would
    invalidate every entry that describes the same experiment.

    ``checkpoint_id`` is the LAYER's identity, read from the provenance `bake`
    wrote (:func:`suite_cache.checkpoint_identity`), and it is not the checkpoint
    name (M3 item U3): the name is a label a re-bake reuses, so a name-keyed
    entry can outlive the layer it measured. It may be ``None`` — a layer baked
    before the stamp existed — and the material is still built, because the
    CALLER has to be able to tell "unknown identity" from "unreadable tree": the
    first is a per-bug cache miss that is recorded on the row, and the second is
    this function's ``None``.

    ``batch_dir`` is deliberately absent from the material (only the tree it holds
    is hashed): the whole point is that two batches measuring the same tree share
    one entry.
    """
    try:
        tree = suite_cache.tree_fingerprint(batch_dir / tree_rel)
    except (OSError, NotADirectoryError):
        return None
    material = {
        "schema": suite_cache.SCHEMA_VERSION,
        "kind": suite_cache.KIND_SUITE_PAIR,
        "harness": suite_cache.harness_build(),
        "tree": tree,
        "command": [str(part) for part in test_cmd],
        "env": {str(k): str(v) for k, v in sorted(cache_env(extra_env).items())},
        "precompile": [str(part) for part in precompile],
        "checkpoint": str(checkpoint),
        "checkpoint_id": suite_cache.checkpoint_identity(checkpoint),
        "disk_mb": int(disk_mb),
        "memory_mb": None if memory_mb is None else int(memory_mb),
        "timeout_secs": int(timeout_secs),
    }
    if carry:
        carried = carry_material(batch_dir, carry)
        if carried is None:
            return None
        material["carry"] = carried
    return material


def _write_reports(batch_dir: Path, report_rel_stem: str, pair: SuitePair) -> None:
    """Put a cached pair's reports back where a run would have left them.

    [VANTA-WHY] ``run_suite_twice`` is not the only reader of these files. The
    stripper reads ``reports/{slug}-oracle.run1.json``
    (:func:`prepare.strip_and_baseline`), and the replay baseline reads
    ``reports/{slug}-vanta.run1/2.json`` (:func:`replay._prepare_baseline_outcomes`).
    A hit that skipped the run without restoring the files would leave those readers
    looking at "the report says nothing" and deriving a wrong answer out of a
    correct cache hit. Only the runs that HAD a report get one, so a cached "no
    report" is still no report.
    """
    for suffix, run in (("1", pair.first), ("2", pair.second)):
        path = batch_dir / f"{report_rel_stem}.run{suffix}.json"
        if not run.report_found:
            path.unlink(missing_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "tests": [
                        {"nodeid": nodeid, "outcome": outcome}
                        for nodeid, outcome in sorted(run.outcomes.items())
                    ]
                }
            ),
            encoding="utf-8",
        )


_FLIGHTS: dict[str, threading.Event] = {}
"""The cache keys one thread in this process is measuring right now.

[VANTA-WHY] M3 items S5/S6 — the in-process single flight. ``S5`` makes three of
a bug's suite runs concurrent and ``S6`` makes rows concurrent, so two callers can
want the SAME key at the same time: the same tree demanded by two rows, or the
same candidate patch graded twice in one wave. Left alone, both would boot VMs for
one measurement and race on the entry file. One caller computes and the others
wait for its entry (or for its failure, in which case they measure it themselves).
The map is guarded by ``_FLIGHTS_GUARD`` and a member is removed in the leader's
``finally``, so a leader that raises still releases its waiters.
"""

_FLIGHTS_GUARD = threading.Lock()


def _single_flight(key: str, produce: Callable[[], SuitePair]) -> SuitePair:
    """Run ``produce`` once per ``key`` in this process; the rest wait for it.

    [VANTA-WHY] M3 items S5/S6. ``produce`` is the whole cache-aware path —
    lookup, then two VMs, then the write — so the flight covers exactly the work
    that must not happen twice for one key. A waiter does NOT receive the leader's
    value directly: it takes its own turn through ``produce`` once the leader has
    released the key, which means it re-reads the entry. That is deliberate. The
    leader may have measured nothing (item U1/U2's rule) or been asked not to write
    (``--no-cache``), and a waiter handed the leader's object would then be handed
    a pair the cache itself refuses to serve. Going back through ``produce`` makes
    the waiter's outcome the same experiment the cache would have given it anyway:
    a hit when the leader wrote a measurement, a fresh measurement when it did not.
    An empty ``key`` is a layer with no identity or a caller that did not opt in —
    no lookup and no write either way, so there is nothing to share and ``produce``
    runs straight through.
    """
    if not key:
        return produce()
    while True:
        with _FLIGHTS_GUARD:
            flight = _FLIGHTS.get(key)
            leader = flight is None
            if leader:
                flight = threading.Event()
                _FLIGHTS[key] = flight
        if leader:
            try:
                return produce()
            finally:
                with _FLIGHTS_GUARD:
                    _FLIGHTS.pop(key, None)
                flight.set()
        flight.wait()


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
    origin: Optional[str] = None,
    budget: Optional[budget_mod.Budget] = None,
    carry: Sequence[str] = (),
) -> SuitePair:
    """Run ``test_cmd`` twice inside the checkpoint and diff the two reports.

    ``origin`` names the caller, and setting it turns on the content-keyed cache
    (M3 items S2/S3): the same tree bytes + command + checkpoint + harness build +
    environment + cap are measured once and the WHOLE pair is served afterwards,
    across arms, seeds, attempts and batches. ``None`` — replay's call, and the
    default — is today's behaviour exactly, with no lookup and no write.

    The lookup still happens under ``--no-cache``: the flag forces a fresh run but
    the row records what the cache WOULD have returned, which is the only way to
    check the cache against a fresh measurement without a second experiment — a
    fresh measurement that succeeded then replaces the entry it could not use
    (M3 item U2).

    A layer with no recorded identity (M3 item U3) turns the cache off entirely
    and records that on the row: the key would otherwise carry the checkpoint's
    NAME, which a re-bake reuses, and an entry that outlives its layer is worse
    than no entry. See :func:`cache_material`.

    ``budget`` (M3 items S5/S6) is the one host-capacity scheduler. Each of the two
    runs takes a slot for the duration of its own VM; when the budget allows more
    than one run at a time the two overlap, otherwise they are serial. ``None``
    keeps the serial behaviour for callers that do not participate in a budget.

    The key is taken before the work and the whole lookup-measure-write path runs
    under :func:`_single_flight`, so two callers that want the same key at the same
    time — which S5 and S6 make ordinary — measure it once (M3 items S5/S6).

    ``carry`` (M3 item V8) names batch-relative paths the run needs beside its tree —
    ``probe_xdist``'s staged wheels, which the run imports off ``PYTHONPATH``. Each
    run's VM mounts its own clone of the tree (:class:`VmRoot`), so without this a
    path beside the tree would simply not exist inside the VM. Non-empty ``carry`` is
    part of the cache key (:func:`cache_material`); empty ``carry`` contributes
    nothing, so today's callers keep today's keys byte for byte.
    """
    material = (
        None
        if origin is None
        else cache_material(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=test_cmd,
            disk_mb=disk_mb,
            timeout_secs=timeout_secs,
            extra_env=extra_env,
            memory_mb=memory_mb,
            precompile=precompile,
            carry=carry,
        )
    )
    key = "" if material is None else suite_cache.key_of(material)
    # A layer with no identity is not an experiment anyone can share, so it takes
    # no flight (the body below turns the cache off for it, item U3).
    flight_key = key if material is not None and material.get("checkpoint_id") else ""
    return _single_flight(
        flight_key,
        lambda: _measure_suite(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=test_cmd,
            report_rel_stem=report_rel_stem,
            disk_mb=disk_mb,
            timeout_secs=timeout_secs,
            extra_env=extra_env,
            memory_mb=memory_mb,
            precompile=precompile,
            origin=origin,
            budget=budget,
            material=material,
            key=key,
            carry=carry,
        ),
    )


def _measure_suite(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    report_rel_stem: str,
    disk_mb: int,
    timeout_secs: int,
    extra_env: Optional[dict[str, str]],
    memory_mb: Optional[int],
    precompile: Sequence[str],
    origin: Optional[str],
    budget: Optional[budget_mod.Budget],
    material: Optional[dict],
    key: str,
    carry: Sequence[str] = (),
) -> SuitePair:
    """The cache-aware body of :func:`run_suite_twice`, run under its flight.

    [VANTA-WHY] M3 items S5/S6. The body is split out so one in-process flight can
    wrap "look up, then measure, then write" as a single unit: the key is computed
    by the caller (before the flight is taken) and handed in, and the entry the
    leader writes is what a waiter finds when it takes its own turn.
    """
    record = origin is not None
    condition = "miss"
    source = ""
    would_hit = False
    write = False
    if material is not None and not material.get("checkpoint_id"):
        # [VANTA-WHY] M3 item U3 — the layer's identity is unknown, so there is
        # nothing this run could be a repetition OF. The cache is off for this
        # call: no lookup (an entry found by name alone would be a measurement of
        # a layer that may no longer exist) and no write (an entry written now
        # would be inherited by whatever the name is next baked into). The row
        # still records the state, so a wave whose layers predate the stamp says
        # so per bug instead of looking like an ordinary miss.
        condition = "off:unknown_checkpoint"
        key = ""
        # Dropping the material is what makes "no lookup … and no write" true:
        # the two gates below are keyed on it, and leaving it set would look the
        # entry up under the empty key and then write a fresh one at
        # `cache_dir()/.json` — a junk entry inherited by the next layer baked
        # into this name, which is the very failure this item exists to stop.
        material = None
    if material is not None:
        entry = suite_cache.read_entry(material, key)
        source = "" if entry is None else str(entry.get("created_by") or "")
        disabled = suite_cache.cache_disabled()
        if entry is not None and not disabled:
            try:
                served = SuitePair.from_json(entry.get("payload"))
            except (KeyError, TypeError, ValueError):
                # An entry that is intact by `suite_cache`'s rules but that this
                # build cannot read: a corrupt-for-us entry. Never served; run
                # fresh and repair it.
                condition = "unreadable"
                would_hit = True
                write = True
            else:
                # [VANTA-WHY] M3 item U2 — the U1 rule, mirrored on this side.
                # The payload parsed, but a pair whose runs measured nothing is
                # not a measurement: `run_state` is exactly the check the rest of
                # the flow refuses a bug on, so an entry that fails it can only
                # have been written before the write gate existed. Present and
                # useless — recompute and overwrite it rather than serve an empty
                # suite as a stable oracle.
                states = (run_state(served.first), run_state(served.second))
                if any(state is not None for state in states):
                    condition = "unmeasured"
                    would_hit = True
                    write = True
                else:
                    served.cache = {
                        "key": key,
                        "state": "hit",
                        "source": source,
                        "would_hit": False,
                        "dir": str(suite_cache.cache_dir()),
                    }
                    _write_reports(batch_dir, report_rel_stem, served)
                    return served
        elif entry is not None:
            # [VANTA-WHY] M3 item U2 — a `--no-cache` run REPLACES the entry it
            # could not use, as `suite_cache.DISABLE_ENV` and the `--no-cache`
            # help both say. Leaving it alone meant a bad entry outlived the one
            # invocation that was asked to check it.
            condition = "off:no_cache"
            would_hit = True
            write = True
        else:
            write = not disabled

    reports_dir = batch_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rel1 = f"{report_rel_stem}.run1.json"
    rel2 = f"{report_rel_stem}.run2.json"
    for rel in (rel1, rel2):
        (batch_dir / rel).unlink(missing_ok=True)

    cmd1 = _shell_quote(build_command(test_cmd, f"/workspace/{rel1}"))
    cmd2 = _shell_quote(build_command(test_cmd, f"/workspace/{rel2}"))

    env = cache_env(extra_env)

    # [VANTA-WHY] M3 item S5 — TWO VMs, one per run, and the D1 markers ride on each
    # run's own script (``_run_script``): `T0` opens the clock immediately before the
    # pytest invocation and `T1` closes it immediately after, with `__VANTA_EXIT_n__`
    # left in between so the exit code is still `$?` of the pytest command and not of
    # the marker. Measuring in the VM and around the command alone is what makes the
    # figure comparable with the in-session cap, which also covers pytest only — a
    # number that included the boot would over-size every derived cap.
    first, second = _run_pair(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=tree_rel,
        test_cmd=test_cmd,
        precompile=precompile,
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        timeout_secs=timeout_secs,
        env=env,
        rel1=rel1,
        rel2=rel2,
        cmd1=cmd1,
        cmd2=cmd2,
        budget=budget,
        carry=carry,
    )
    pair = compare_runs(first, second)

    _clean_tree_noise(batch_dir / tree_rel)

    if material is not None:
        # [VANTA-WHY] M3 item U2 — ONLY A MEASUREMENT IS CACHED. `write` alone is
        # not enough: a plain miss sets it for every fresh run, including one whose
        # VM never booted, one the OOM killer ended, one that hit the cap under
        # load, or one that left a report nothing can read. Stored, such a pair is
        # served forever after as a stable oracle — `oracle.grade` reads it as
        # inconclusive and `prepare` reuses it for the same tree — which is the
        # defect, not the optimisation. `run_state` is the harness's own answer to
        # "did this run measure anything", so the gate is that answer for BOTH
        # runs. A `--no-cache` run that measured nothing therefore leaves the
        # entry it could not use intact, and still reports what it would have hit.
        if write and run_state(first) is None and run_state(second) is None:
            entry = suite_cache.build_entry(material, pair.to_json(), origin=origin)
            try:
                suite_cache.write_entry(entry)
            except OSError:
                # A cache that cannot be written must never fail the run: the
                # measurement is the product, the entry is an optimisation.
                pass
        # [VANTA-WHY] M3 item S7 — the DIRECTORY is recorded beside the key.
        # One suite cache root is shared by every batch unless a batch overrides
        # ``VANTA_SUITE_CACHE_DIR``, and the equivalence check runs two batches on
        # the same build and the same trees: left shared, the second batch is
        # served the first's entries and the comparison is vacuous. The key alone
        # cannot say which directory served it, so the row states the root this
        # measurement came from and the comparator can require the two batches to
        # name DIFFERENT roots.
        pair.cache = {
            "key": key,
            "state": condition,
            "source": source,
            "would_hit": would_hit,
            "dir": str(suite_cache.cache_dir()),
        }
    elif record:
        # [VANTA-WHY] M3 item U3 — the one path where the cache was asked for
        # and could not be used at all: the layer has no identity, so there is no
        # key, no lookup and no write, and the row says exactly that.
        pair.cache = {
            "key": "",
            "state": condition,
            "source": "",
            "would_hit": False,
            "dir": str(suite_cache.cache_dir()),
        }
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
    budget: Optional[budget_mod.Budget] = None,
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

    [VANTA-WHY] M3 item V8 — the VM mounts a clone of the tree, never the batch
    directory, so the collection run cannot read the oracle, the hidden tests or the
    manifest even by absolute path, and the caches collection leaves behind are
    deleted with the clone instead of landing in the tree the next step reads.
    """
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    if extra_env:
        env.update(extra_env)
    # M3 item Y1 — the collection VM holds one budget slot, like every other run's.
    gate = budget.slot(memory_mb=memory_mb) if budget is not None else nullcontext()
    with gate, VmRoot(batch_dir=batch_dir, tree_rel=tree_rel) as root:
        result = shuru.run(
            checkpoint=checkpoint,
            host_root=root.host,
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
    carry: Sequence[str] = (),
    budget: Optional[budget_mod.Budget] = None,
) -> str:
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

    The build's compiled objects have to be there for every suite that runs
    afterwards, which is why the VM's tree is adopted back onto the host (item V8/Q1,
    below) rather than discarded with the VM.

    Raises :class:`SuiteError`, whose message names the tree and carries the error lines
    (:func:`failure_excerpt`), so a build failure is reported as itself and never as the
    collection it would otherwise have caused.

    [VANTA-WHY] M3 item V8/Q1. The build now runs in a **clone** of the tree, and the
    host adopts the clone's result when — and only when — the build succeeded: the VM
    was the only writer, so the tree that replaces the host's is the tree the build
    actually produced, validated (:meth:`VmRoot.adopt`) and swapped in atomically. A
    build that fails or times out raises before the swap, so a half-built tree is
    never left where the next step would read it. The adopted tree's fingerprint is
    returned for ``prepare`` to record on ``prepare.json``; every later cache key is
    computed from the tree on disk at key time, so it is automatically the adopted
    one. ``carry`` names the batch-relative build files (:func:`prepare`) the build
    reads — written beside the trees, which the VM no longer sees as ``/workspace``
    unless they are carried in.
    """
    command = [python, "setup.py", *setup_args]
    env_full = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if env:
        env_full.update(env)
    where = guest_tree_root(tree_rel)
    # [VANTA-WHY] M3 item Y1 — the build's VM is one VM, so it holds one slot of the
    # invocation's budget exactly as a suite run does; a build that ran outside the
    # budget would let a concurrent prepare boot more VMs than the cap admits. Without
    # a budget (a test, `pool`, a serial prepare) it boots as it always did.
    gate = budget.slot(memory_mb=memory_mb) if budget is not None else nullcontext()
    with gate, VmRoot(batch_dir=batch_dir, tree_rel=tree_rel, carry=carry) as root:
        result = shuru.run(
            checkpoint=checkpoint,
            host_root=root.host,
            command=["sh", "-c", _shell_quote(command)],
            disk_mb=disk_mb,
            memory_mb=memory_mb,
            guest_cwd=where,
            timeout_secs=timeout_secs,
            env=env_full,
        )
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
        return root.adopt()


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
    budget: Optional[budget_mod.Budget] = None,
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

    [VANTA-WHY] M3 item V8 — the VM mounts a clone of the tree, never the batch
    directory, so the gate cannot read the oracle or the hidden tests even by
    absolute path.
    """
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    if extra_env:
        env.update(extra_env)
    # M3 item Y1 — the gate's VM holds one budget slot, like every other run's.
    gate = budget.slot(memory_mb=memory_mb) if budget is not None else nullcontext()
    with gate, VmRoot(batch_dir=batch_dir, tree_rel=tree_rel) as root:
        result = shuru.run(
            checkpoint=checkpoint,
            host_root=root.host,
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
    budget: Optional[budget_mod.Budget] = None,
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

    [VANTA-WHY] M3 item V8 — the VM mounts a clone of the tree, never the batch
    directory. The path the probe prints is therefore still ``/workspace/<tree_rel>``
    and comparable with :func:`guest_tree_root`, while the oracle beside the tree is
    outside the VM's filesystem root.
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
    # M3 item Y1 — the probe's VM holds one budget slot, like every other run's.
    gate = budget.slot(memory_mb=memory_mb) if budget is not None else nullcontext()
    with gate, VmRoot(batch_dir=batch_dir, tree_rel=tree_rel) as root:
        result = shuru.run(
            checkpoint=checkpoint,
            host_root=root.host,
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

    [VANTA-WHY] M3 item V8/Q2 — a suite run no longer writes into the host tree at
    all: it runs in a clone (:class:`VmRoot`) that is deleted with its VM, so this is
    now the belt to the clone's braces. It still runs, because the tree it guards is
    the one later steps copy, and because a run that predates item V8 — or a step
    that adopts a built tree — can have left the same artefacts behind.
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
    staged_rel = f"reports/_apply_{patch_path.name}"
    staged = batch_dir / staged_rel
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(body, encoding="utf-8")
    with VmRoot(
        batch_dir=batch_dir, tree_rel=tree_rel, carry=(staged_rel,)
    ) as root:
        result = shuru.run(
            checkpoint=checkpoint,
            host_root=root.host,
            command=["git", "apply", "--verbose", f"/workspace/{staged_rel}"],
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
        # [VANTA-WHY] M3 item V8/Q1 — the applied tree is what every later step
        # reads, so the host adopts the VM's tree. Only on a clean apply: a rejected
        # diff changed nothing the host should keep, and the clone is discarded with
        # its VM rather than replacing the tree with a partially patched one.
        root.adopt()
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
