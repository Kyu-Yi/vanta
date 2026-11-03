#!/usr/bin/env python3
"""probe_xdist.py — does the native suite measure the same under pytest-xdist? (M3 item S4)

WHY THIS EXISTS
---------------
The speed profile (``plans/private/M3-progress.md`` §"Step 5") ranks
``pytest-xdist`` inside the VM as one of the two biggest remaining wins — the native
test time is 83-87 % of a pandas/1 report and 4-8 workers could cut its ~403 s of
test time to ~60-100 s — and rates it **risk medium**, because "parallel test
order/flakiness can move a label". That is the whole question this probe answers,
and it is the one question a speed change is not allowed to answer by assumption:
if the suite collects a different nodeid set, or grades a nodeid differently, under
``-n auto``, then turning xdist on would silently change ``FAIL_TO_PASS`` /
``PASS_TO_PASS`` and every label derived from them.

So S4 is deliberately **not** an implementation. xdist is enabled nowhere in the
harness. This is a standalone probe, run by hand on one bug at a time, that measures
the serial suite and the same suite with ``-n auto`` appended, diffs the two
``nodeid -> outcome`` maps, and writes a JSON report plus one verdict line.

WHAT "EXACTLY LIKE THE HARNESS" MEANS HERE
------------------------------------------
Every suite run in this project happens inside Shuru, on a tree the harness built,
through ``suite.run_suite_twice``. This probe does not re-implement any of that: it
loads the bug's preps with the harness's own read-back (``runner.load_prepared``),
takes the split entry's ``checkpoint``, and calls ``suite.run_suite_twice`` twice —
once with the bug's own ``test_cmd`` and once with ``-n auto`` appended to it. The
VM script, the report path, the environment (``suite.cache_env``), the pre-compile
step and the timeout rule are therefore byte-for-byte the harness's, and nothing
here can drift from what a real row would run. The suite cache is left off
(``origin`` omitted), so neither run reads or writes it.

WHY THE BUG'S *OWN* PYTEST PIN MATTERS
--------------------------------------
A bug's checkpoint does not necessarily carry the split's ``pytest_version``: a
project that declares its own pytest pin gets it, and the layer *replaces* the
harness pin with it. Measured on pandas/1: the split records
``"pytest_version": "8.0.2"`` but the deps record
(``runs/bugsinpy/deps/vanta-bi-pandas-1.json``) installs ``pytest==5.4.2``, because
``requirements-dev.txt`` declares it, and pytest 8.0.2's own dependencies
(``exceptiongroup``, ``iniconfig``, ``tomli``, ``typing-extensions``) are recorded as
**orphans** ``replaced_by pytest==5.4.2``. An xdist built for pytest >= 7.0 would not
load there at all. So the probe reads the **actual** pytest version out of the
checkpoint at run time and checks the staged xdist's own ``Requires-Dist`` against
it — never the split's number.

WHERE XDIST MAY COME FROM
-------------------------
The VMs booted from a checkpoint have **no network device at all** (``shuru.run``'s
``--from`` path), so xdist can only come from:

  1. the checkpoint itself, when it already has ``xdist`` importable; or
  2. a wheel (or wheels) the operator staged offline, passed as
     ``--xdist-wheel-dir``. The probe checks each wheel's ``Requires-Dist`` against
     the checkpoint's pytest, and only then unpacks them under the batch directory and
     puts that directory on ``PYTHONPATH`` — so a wheel set the checkpoint cannot use,
     and a ``--dry-run``, write nothing at all (M3 item V5).

If neither is available — or the staged wheel's pytest requirement is not satisfied
by the checkpoint's pytest — the probe reports ``unavailable`` with the reason and
runs nothing. It never tries to install anything, and it never falls back to a
different mechanism.

USAGE (run from the repository root)
------------------------------------
    python3 scripts/probe_xdist.py \
        --batch runs/bugsinpy/m3-smoke1-PySnooper_3-k4 --bug PySnooper/3

    python3 scripts/probe_xdist.py \
        --batch runs/bugsinpy/m3-smoke2-pandas_1 --bug pandas/1 \
        --xdist-wheel-dir /path/to/staged-wheels

    # what would be run, booting nothing:
    python3 scripts/probe_xdist.py --batch ... --bug ... --dry-run

OUTPUT
------
One verdict line (``identical`` / ``not_identical`` / ``unavailable`` /
``serial_unavailable``, with the differing nodeids), and a JSON report at
``<batch>/probe_xdist/<slug>-xdist.json``. Exit codes: ``0`` the probe produced a
verdict (including ``unavailable``), ``2`` the probe could not get as far as a
suite — no batch, no tree, no checkpoint.

A ``--dry-run`` boots nothing and **writes no report** unless ``--report`` names one:
it prints the resolved xdist details and the plan, so the batch is read and never
touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from packaging.requirements import InvalidRequirement, Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402
from packaging.version import InvalidVersion, Version  # noqa: E402

from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
from bugsinpy import shuru as shuru_mod  # noqa: E402
from bugsinpy import suite as suite_mod  # noqa: E402
from bugsinpy.split import SplitError, load_split  # noqa: E402

DEFAULT_SPLIT = REPO_ROOT / "plans" / "bugsinpy_split.json"

XDIST_ARGS = ["-n", "auto"]
"""What is appended to the bug's own ``test_cmd`` for the parallel run.

``-n auto`` is the flag the mandate names, and it is appended *after* the report
options so the serial and parallel commands differ by exactly this suffix.
"""

SCHEMA_VERSION = 1

VERDICT_IDENTICAL = "identical"
VERDICT_DIFFERENT = "not_identical"
VERDICT_UNAVAILABLE = "unavailable"
VERDICT_SERIAL_BLANK = "serial_unavailable"
"""The four things this probe can say.

``unavailable`` is a **result**, not an error: it means xdist could not be resolved
offline for this bug and the parallel side was therefore never run. Reporting it is
the point — working around a missing offline wheel would be the thing this forbids.
``serial_unavailable`` is the same shape on the other side: the serial run measured
nothing, so there is no baseline to compare against.
"""

SOURCE_CHECKPOINT = "checkpoint"
SOURCE_STAGED_WHEEL = "staged_wheel"

PROBE_MARKER = "__VANTA_XDIST_PROBE__"
"""Prefix of the one line the in-VM preflight prints, so the rest of a boot's output
(a start-up-check banner, an interpreter warning) cannot be mistaken for the answer.
"""

PREFLIGHT_CODE = """\
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import pytest
    out["pytest"] = pytest.__version__
except Exception as exc:
    out["pytest"] = None
    out["pytest_error"] = f"{type(exc).__name__}: {exc}"
try:
    import xdist
    out["xdist"] = getattr(xdist, "__version__", None)
except Exception:
    out["xdist"] = None
print("%s" + json.dumps(out))
""" % PROBE_MARKER

PREFLIGHT_TIMEOUT_SECS = 180


# ---------------------------------------------------------------------------
# the pure part: one diff, one line
# ---------------------------------------------------------------------------
def outcome_diff(
    serial: Mapping[str, str],
    xdist: Mapping[str, str],
    excluded: Iterable[str] = (),
) -> dict[str, Any]:
    """Two ``nodeid -> outcome`` maps -> whether they agree, and where they do not.

    Pure, and the only place the comparison is made, so the rule can be exercised
    without booting a VM. Every outcome string is compared **raw** — ``passed``,
    ``failed``, ``error``, ``skipped`` and ``xfailed`` are all first-class here, not
    reduced to pass/fail the way the in-session native gate reduces them. That is
    deliberate and it is what the mandate asks for: xdist changes *scheduling*, and a
    test that flips between ``skipped`` and ``passed``, or between ``failed`` and
    ``error``, is exactly the kind of movement a gate that only counts pass/fail
    would hide.

    ``excluded`` is dropped from **both** sides before the diff, which is how the
    known-flaky set is kept from being reported as a difference (the same rule
    ``replay.baseline_parity`` applies to its two sides).

    The three lists are named for the side a nodeid is *missing* from, and a nodeid
    present on both with different outcomes lands in ``different_outcome`` — never in
    both.
    """

    excluded_set = {str(nodeid) for nodeid in excluded}
    left = {n: o for n, o in serial.items() if n not in excluded_set}
    right = {n: o for n, o in xdist.items() if n not in excluded_set}
    left_ids = set(left)
    right_ids = set(right)
    only_serial = sorted(left_ids - right_ids)
    only_xdist = sorted(right_ids - left_ids)
    different = sorted(n for n in left_ids & right_ids if left[n] != right[n])
    return {
        "identical": not (only_serial or only_xdist or different),
        "only_in_serial": only_serial,
        "only_in_xdist": only_xdist,
        "different_outcome": different,
    }


def verdict_line(report: Mapping[str, Any], *, cap: int = 5) -> str:
    """The probe's one line of output. Pure, so the shape is testable.

    The verdict is named first, then the counts, then — when the two sides differ —
    the differing nodeids themselves (capped, with the remainder counted), because
    "not identical" without the names is not actionable.
    """
    verdict = str(report["verdict"])
    head = f"probe-xdist {report['bug']}: {verdict}"
    if verdict not in (VERDICT_IDENTICAL, VERDICT_DIFFERENT):
        note = str(report.get("note") or "").strip()
        return f"{head} — {note}" if note else head

    diff = report["diff"]
    parts = [
        f"serial {report['serial']['collected']}",
        f"xdist {report['xdist']['collected']}",
        f"excluded {report['excluded_flaky']}",
    ]
    if verdict == VERDICT_DIFFERENT:
        parts.append(f"only_in_serial={len(diff['only_in_serial'])}")
        parts.append(f"only_in_xdist={len(diff['only_in_xdist'])}")
        parts.append(f"different_outcome={len(diff['different_outcome'])}")
    line = f"{head} ({', '.join(parts)})"

    differing = list(diff["only_in_serial"]) + list(diff["only_in_xdist"]) + list(
        diff["different_outcome"]
    )
    if differing:
        shown = ", ".join(differing[:cap])
        more = "" if len(differing) <= cap else f", +{len(differing) - cap} more"
        line += f" [{shown}{more}]"
    return line


# ---------------------------------------------------------------------------
# the pure part: what a staged wheel demands of pytest
# ---------------------------------------------------------------------------
def requires_dist(metadata: str) -> list[str]:
    """The ``Requires-Dist`` values of a ``METADATA`` file, in order. Pure."""
    out: list[str] = []
    for line in metadata.splitlines():
        if line.lower().startswith("requires-dist:"):
            out.append(line.split(":", 1)[1].strip())
    return out


def requirement_for(requirements: Sequence[str], name: str) -> Optional[str]:
    """The first requirement whose project name is ``name``, or ``None``. Pure.

    Matched through :func:`packaging.utils.canonicalize_name`, so ``pytest`` and
    ``Pytest`` are the same project, and an unparsable line is skipped rather than
    raising — a wheel whose metadata this build cannot read is a wheel whose
    requirement cannot be checked, which the caller reports as *unverified* rather
    than as satisfied.
    """
    wanted = canonicalize_name(name)
    for raw in requirements:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if canonicalize_name(req.name) == wanted:
            return raw
    return None


def requirement_satisfied(spec: Optional[str], version: Optional[str]) -> Optional[bool]:
    """Does ``version`` satisfy ``spec``? ``None`` when either cannot be read. Pure.

    The markers a requirement may carry (``; python_version < "3.11"``) are ignored
    deliberately: the only marker an xdist wheel uses is on an interpreter-version
    edge, and treating an unreadable case as ``None`` keeps the caller honest instead
    of guessing a boolean.
    """
    if not spec or not version:
        return None
    try:
        req = Requirement(spec)
        return req.specifier.contains(Version(version), prereleases=True)
    except (InvalidRequirement, InvalidVersion):
        return None


def metadata_of(wheel: Path) -> str:
    """The ``METADATA`` text inside a wheel, found without installing it."""
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(".dist-info/METADATA"):
                return archive.read(name).decode("utf-8", "replace")
    raise shuru_mod.ShuruError(f"{wheel} carries no *.dist-info/METADATA")


def _refuse_within(dest: Path, *, root: Path, what: str) -> None:
    """Refuse unless ``dest`` lies plainly inside ``root``, with no symlink on the way.

    [VANTA-WHY] M3 item V5. ``unpack_wheels`` CLEARS its destination with
    ``shutil.rmtree`` before extracting, and that destination is built by joining names
    onto the batch directory. ``rmtree`` follows a symlink, so a symlink planted at any
    name in that join — ``probe_xdist`` itself, or a deeper one — would make the
    clear-and-extract delete and write something outside the batch directory. Two rules
    remove that: the destination must be lexically under ``root``, and no component from
    ``root`` down to it may be a symlink. That second rule is the "symlinked ancestors"
    refusal: the cleanup stays inside the batch directory even when the batch directory
    has been tampered with. ``root`` itself is not checked — the caller has already
    resolved it.
    """
    try:
        dest.relative_to(root)
    except ValueError:
        raise shuru_mod.ShuruError(
            f"refusing {what}: {dest} is not inside {root}"
        ) from None
    component = dest
    while component != root:
        if component.is_symlink():
            raise shuru_mod.ShuruError(
                f"refusing {what}: {component} is a symlink, so clearing {dest} could "
                f"reach outside {root}"
            )
        component = component.parent


def unpack_wheels(wheels: Sequence[Path], dest: Path, *, root: Path) -> list[str]:
    """Unpack staged wheels into ``dest`` so an offline boot can import them.

    A wheel is a zip whose packages sit at the archive root, and a checkout of it
    carries the ``*.dist-info`` directory that ``importlib.metadata`` needs to find
    the ``pytest11`` entry point pytest uses to load a plugin. So unpacking a wheel
    and putting the directory on ``PYTHONPATH`` is enough to make it importable
    **without pip and without network** — which is the only kind of install these
    VMs can do.

    Members with a ``..`` component are refused rather than written: the wheels are
    the operator's, and a wheel that tries to escape the scratch directory is a
    staging mistake worth naming.

    [VANTA-WHY] M3 item U9 — ``dest`` is cleared first. It goes on ``PYTHONPATH``, and
    ``extractall`` only ever adds files: without this, a wheel staged by a **previous**
    probe run keeps its packages in ``dest``, so a run whose wheel set changed (or whose
    wheel was removed) imports a stale plugin from a directory the operator no longer
    names. Clearing the directory makes its contents exactly the wheels passed in, which
    is what the caller's ``staging`` claim means.

    [VANTA-WHY] M3 item V5 — ``root`` is the directory that clear is confined to, and
    :func:`_refuse_within` is asked before anything is removed or created. It is a
    required keyword rather than a defaulted one so no caller can forget it: the whole
    point is that ``rmtree`` never runs on a path the batch directory does not own.
    """
    _refuse_within(dest, root=root, what=f"staging directory {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if ".." in Path(member).parts:
                    raise shuru_mod.ShuruError(
                        f"refusing {wheel}: member {member!r} escapes the staging directory"
                    )
            archive.extractall(dest)
        published.append(wheel.name)
    return published


# ---------------------------------------------------------------------------
# the VM side
# ---------------------------------------------------------------------------
def parse_preflight(stdout: str) -> dict[str, Any]:
    """The preflight boot's answer: the line after :data:`PROBE_MARKER`, as JSON."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            return json.loads(line[len(PROBE_MARKER):])
    raise shuru_mod.ShuruError(
        "the preflight boot printed no answer line; the interpreter inside the "
        f"checkpoint could not be asked. Output tail:\n{stdout[-2000:]}"
    )


def _preflight(
    *,
    checkpoint: str,
    interpreter: str,
    batch_dir: Path,
    disk_mb: int,
    memory_mb: Optional[int],
    extra_env: Optional[Mapping[str, str]],
) -> dict[str, Any]:
    result = shuru_mod.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=[interpreter, "-c", PREFLIGHT_CODE],
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        guest_cwd="/workspace",
        timeout_secs=PREFLIGHT_TIMEOUT_SECS,
        env=suite_mod.cache_env(extra_env),
    )
    if not result.ok:
        raise shuru_mod.ShuruError(
            f"the preflight boot of {checkpoint} failed (exit {result.exit_code}, "
            f"timed_out={result.timed_out}):\n{result.stderr[-2000:]}"
        )
    return parse_preflight(result.stdout)


def _guest_path(batch_dir: Path, path: Path) -> str:
    """``batch_dir``'s host path as the VM sees it (``/workspace/<rel>``)."""
    return "/workspace/" + path.relative_to(batch_dir).as_posix()


# ---------------------------------------------------------------------------
# driving one bug
# ---------------------------------------------------------------------------
def _finish(report: dict[str, Any], out_path: Path) -> int:
    report["report_path"] = str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(verdict_line(report))
    print(f"report: {out_path}")
    return 0


def _print_dry_run_plan(report: Mapping[str, Any], *, will_write: bool) -> None:
    """Print what a dry run resolved, and what it would have run (M3 item U10).

    Everything ``_finish`` would have recorded, minus the measurements only a boot can
    produce: the checkpoint and tree, the command and the ``-n auto`` suffix, and the
    xdist the preflight resolved with its source, version and staging directory. It is
    the whole point of ``--dry-run`` — the operator sees the plan without the batch
    being touched.
    """
    xdist = report.get("xdist") or {}
    rows = [
        ("checkpoint", report.get("checkpoint")),
        ("tree", report.get("tree")),
        ("command", " ".join(str(p) for p in report.get("command") or [])),
        ("xdist args", " ".join(str(p) for p in report.get("xdist_args") or [])),
        ("xdist source", xdist.get("source")),
        ("xdist version", xdist.get("version")),
        ("pytest", xdist.get("pytest")),
        ("staging", xdist.get("staging") or "-"),
        ("wheels", ", ".join(xdist.get("wheels") or []) or "-"),
        ("timeout", f"{report.get('timeout_secs')}s"),
        ("disk / memory", f"{report.get('disk_mb')}MB / {report.get('memory_mb')}MB"),
    ]
    print(f"probe-xdist dry run: {report.get('bug')}")
    for name, value in rows:
        print(f"  {name}: {value}")
    if xdist.get("source") is None:
        print(f"  plan: nothing — {report.get('note')}")
    else:
        print(
            "  plan: run the suite serially, then again with the xdist args above, and "
            "diff the two nodeid->outcome maps"
        )
    if not will_write:
        print("  report: not written (pass --report to write one)")


def _stop(report: dict[str, Any], out_path: Path, args: argparse.Namespace) -> int:
    """End the probe without booting a suite — the report, or a dry run's plan.

    M3 item U10: a ``--dry-run`` writes a report only when ``--report`` names one, so
    the dry run stays a pure read of the batch. Every other early exit writes its
    report exactly as before.
    """
    if args.dry_run:
        _print_dry_run_plan(report, will_write=args.report is not None)
        if args.report is None:
            return 0
    return _finish(report, out_path)


def _resolve_wheel_dir(raw: Path, *, batch_dir: Path) -> Path:
    """The operator's ``--xdist-wheel-dir``, refused when it cannot be read safely.

    [VANTA-WHY] M3 item V5. The probe writes exactly one thing for a staged wheel: it
    CLEARS and then extracts into ``batch_dir/probe_xdist/<tree>/_wheels``. A wheel
    directory inside that scratch is refused, because ``unpack_wheels`` would delete the
    operator's own wheels — or, on a second run, feed the probe back the packages it
    extracted the first time, from a directory the operator no longer names. The
    containment test is made on the *resolved* path, so a symlink could otherwise let
    the same directory pass as one outside the scratch and slip past the refusal; the
    symlink itself is refused so the test stays a statement about what was named. The
    path is returned resolved, so every later use of it is the directory the refusal was
    made against.
    """
    given = Path(raw)
    if given.is_symlink():
        raise shuru_mod.ShuruError(
            f"refusing --xdist-wheel-dir {given}: it is a symlink, so the directory read "
            "here is not the directory named"
        )
    wheel_dir = given.resolve()
    scratch = (batch_dir / "probe_xdist").resolve()
    if wheel_dir == scratch or scratch in wheel_dir.parents:
        raise shuru_mod.ShuruError(
            f"refusing --xdist-wheel-dir {wheel_dir}: it is inside the probe's staging "
            f"area {scratch}, which unpack_wheels clears before it extracts"
        )
    return wheel_dir


def _resolve_xdist(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    interpreter: str,
    batch_dir: Path,
    tree_rel: str,
    disk_mb: int,
    memory_mb: Optional[int],
    sandbox_env: Mapping[str, str],
) -> dict[str, Any]:
    """Which xdist the parallel run gets, and the environment that makes it loadable.

    Returns a block with ``source`` (or ``reason`` when there is none) plus the
    ``extra_env`` the parallel ``run_suite_twice`` call has to be given.
    """
    probe = _preflight(
        checkpoint=checkpoint,
        interpreter=interpreter,
        batch_dir=batch_dir,
        disk_mb=disk_mb,
        memory_mb=memory_mb,
        extra_env=sandbox_env or None,
    )
    block: dict[str, Any] = {
        "source": None,
        "version": probe.get("xdist"),
        "pytest": probe.get("pytest"),
        "python": probe.get("python"),
        "stance": {},
        "reason": "",
    }
    if probe.get("xdist"):
        block["source"] = SOURCE_CHECKPOINT
        block["extra_env"] = dict(sandbox_env)
        return block

    if args.xdist_wheel_dir is None:
        block["reason"] = (
            f"the checkpoint has no importable pytest-xdist (pytest {probe.get('pytest')!r}) "
            "and no --xdist-wheel-dir was given, so the parallel side was not run"
        )
        return block

    wheel_dir = _resolve_wheel_dir(args.xdist_wheel_dir, batch_dir=batch_dir)
    wheels = sorted(p for p in wheel_dir.glob("*.whl") if p.is_file())
    if not wheels:
        block["reason"] = f"{wheel_dir} contains no *.whl, so the parallel side was not run"
        return block

    staging = batch_dir / "probe_xdist" / Path(tree_rel).parent.name / "_wheels"
    # [VANTA-WHY] M3 item V5 — everything above is a read of the operator's wheel
    # directory, and the stance loop below is a read of each wheel's METADATA. Nothing is
    # written to the batch directory before the wheels are known to be usable, so a dry
    # run — and a run whose wheels the checkpoint cannot use — leaves the batch exactly
    # as it found it. ``unpack_wheels`` is the one write, and it happens immediately
    # before the boot that needs the packages on ``PYTHONPATH``.
    block["wheels"] = [wheel.name for wheel in wheels]
    block["staging"] = str(staging)

    pytest_version = probe.get("pytest")
    xdist_spec: Optional[str] = None
    for wheel in wheels:
        requirements = requires_dist(metadata_of(wheel))
        spec = requirement_for(requirements, "pytest")
        if spec is not None:
            xdist_spec = spec
            block["stance"][wheel.name] = {
                "requires_pytest": spec,
                "satisfied": requirement_satisfied(spec, str(pytest_version) if pytest_version else None),
            }
        else:
            block["stance"][wheel.name] = {"requires_pytest": None, "satisfied": None}

    unsatisfied = [
        name for name, found in block["stance"].items() if found["satisfied"] is False
    ]
    if unsatisfied:
        block["reason"] = (
            f"the staged wheel(s) {', '.join(unsatisfied)} require a pytest the checkpoint "
            f"does not have (checkpoint pytest is {pytest_version!r}); "
            f"the requirement is {xdist_spec!r} — the parallel side was not run"
        )
        return block

    if not args.dry_run:
        unpack_wheels(wheels, staging, root=batch_dir)

    env = dict(sandbox_env)
    guest_staging = _guest_path(batch_dir, staging)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{guest_staging}:{existing}" if existing else guest_staging
    block["source"] = SOURCE_STAGED_WHEEL
    block["extra_env"] = env
    return block


def _run_bug(args: argparse.Namespace) -> int:
    batch_dir = Path(args.batch).resolve()
    split_path = (
        Path(args.split).resolve()
        if args.split is not None
        else (runner_mod.recorded_split_path(batch_dir) or DEFAULT_SPLIT)
    )
    split = load_split(split_path)
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    prep = next((p for p in batch.preps if p.key == args.bug), None)
    if prep is None:
        raise shuru_mod.ShuruError(
            f"{args.bug} is not one of the bugs {batch_dir / 'prepare.json'} records "
            f"({', '.join(sorted(p.key for p in batch.preps)) or 'none'})"
        )
    entry = batch.entries[args.bug]
    checkpoint = str(entry["checkpoint"])
    prepare_mod.require_checkpoint(checkpoint)

    tree_rel = f"{prep.root.relative_to(batch_dir).as_posix()}/src_buggy"
    if not (batch_dir / tree_rel).is_dir():
        raise shuru_mod.ShuruError(
            f"no src_buggy tree at {batch_dir / tree_rel}; prepare the batch again "
            "(the probe measures the tree the harness measured, it does not build one)"
        )

    disk_mb = prepare_mod.DISK_MB
    timeout_secs = args.timeout_secs or suite_mod.native_timeout_secs(
        prep.native_secs, floor=1800
    )
    serial_stem = f"reports/{prep.slug}-xdist-serial"
    parallel_stem = f"reports/{prep.slug}-xdist-auto"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bug": args.bug,
        "batch_dir": str(batch_dir),
        "split": str(split_path),
        "checkpoint": checkpoint,
        "tree": tree_rel,
        "command": [str(part) for part in prep.test_cmd],
        "xdist_args": list(XDIST_ARGS),
        "disk_mb": disk_mb,
        "memory_mb": prep.vm_memory_mb,
        "timeout_secs": timeout_secs,
        "excluded_flaky": 0,
        "serial": {"collected": 0, "state": None, "flaky": [], "missing": [], "secs": None},
        "xdist": {},
        "diff": {
            "identical": None,
            "only_in_serial": [],
            "only_in_xdist": [],
            "different_outcome": [],
        },
        "verdict": "",
        "note": "",
    }
    out_path = (
        Path(args.report).resolve()
        if args.report is not None
        else batch_dir / "probe_xdist" / f"{prep.slug}-xdist.json"
    )

    resolved = _resolve_xdist(
        args,
        checkpoint=checkpoint,
        interpreter=str(prep.test_cmd[0]),
        batch_dir=batch_dir,
        tree_rel=tree_rel,
        disk_mb=disk_mb,
        memory_mb=prep.vm_memory_mb,
        sandbox_env=prep.sandbox_env or {},
    )
    report["xdist"] = {
        key: value for key, value in resolved.items() if key != "extra_env"
    }
    if resolved["source"] is None:
        report["verdict"] = VERDICT_UNAVAILABLE
        report["note"] = str(resolved["reason"])
        return _stop(report, out_path, args)

    if args.dry_run:
        report["verdict"] = VERDICT_UNAVAILABLE
        report["note"] = "dry run: the preflight resolved xdist and nothing was booted"
        return _stop(report, out_path, args)

    print(
        f"probe-xdist {args.bug}: xdist {resolved.get('version')} from "
        f"{resolved['source']} (pytest {resolved.get('pytest')})",
        file=sys.stderr,
    )

    serial_pair = suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=tree_rel,
        test_cmd=[str(part) for part in prep.test_cmd],
        report_rel_stem=serial_stem,
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        extra_env=prep.sandbox_env or None,
        memory_mb=prep.vm_memory_mb,
        precompile=prep.precompile,
    )
    serial_state = suite_mod.run_state(serial_pair.first) or suite_mod.run_state(
        serial_pair.second
    )
    report["serial"] = {
        "collected": len(serial_pair.first.outcomes),
        "state": serial_state,
        "flaky": sorted(serial_pair.flaky_nodeids),
        "missing": sorted(serial_pair.missing_from_one_run),
        "secs": serial_pair.first.secs,
    }
    if serial_state is not None:
        report["verdict"] = VERDICT_SERIAL_BLANK
        report["note"] = f"the serial run measured nothing: {serial_state}"
        return _finish(report, out_path)

    parallel_pair = suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=tree_rel,
        test_cmd=[str(part) for part in prep.test_cmd] + list(XDIST_ARGS),
        report_rel_stem=parallel_stem,
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        extra_env=resolved["extra_env"] or None,
        memory_mb=prep.vm_memory_mb,
        precompile=prep.precompile,
    )
    parallel_state = suite_mod.run_state(parallel_pair.first) or suite_mod.run_state(
        parallel_pair.second
    )
    report["xdist"]["collected"] = len(parallel_pair.first.outcomes)
    report["xdist"]["state"] = parallel_state
    report["xdist"]["flaky"] = sorted(parallel_pair.flaky_nodeids)
    report["xdist"]["missing"] = sorted(parallel_pair.missing_from_one_run)
    report["xdist"]["secs"] = parallel_pair.first.secs
    if parallel_state is not None:
        report["verdict"] = VERDICT_UNAVAILABLE
        report["note"] = (
            f"the -n auto run measured nothing ({parallel_state}); xdist resolved to "
            f"{resolved['source']} but the suite did not run under it"
        )
        return _finish(report, out_path)

    excluded = (
        {str(nodeid) for nodeid in prep.flaky_nodeids}
        | serial_pair.flaky_nodeids
        | serial_pair.missing_from_one_run
        | parallel_pair.flaky_nodeids
        | parallel_pair.missing_from_one_run
    )
    report["excluded_flaky"] = len(excluded)
    report["diff"] = outcome_diff(
        serial_pair.first.outcomes, parallel_pair.first.outcomes, excluded
    )
    report["verdict"] = (
        VERDICT_IDENTICAL if report["diff"]["identical"] else VERDICT_DIFFERENT
    )
    return _finish(report, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="probe_xdist.py",
        description=(
            "Run a bug's native suite serially and again with `-n auto` inside Shuru, "
            "and report whether the two nodeid->outcome maps are identical. "
            "Enables xdist nowhere."
        ),
    )
    parser.add_argument(
        "--batch",
        required=True,
        type=Path,
        help="a prepared batch directory (the bug's tree must still be under worktrees/)",
    )
    parser.add_argument("--bug", required=True, help="the bug key, e.g. PySnooper/3")
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="the split the batch was prepared with (default: as prepare.json recorded)",
    )
    parser.add_argument(
        "--xdist-wheel-dir",
        type=Path,
        default=None,
        help="a directory of *.whl staged offline, used only when the checkpoint has no xdist",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=None,
        help="cap for each run_suite_twice boot (default: 3 x the bug's measured native run)",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="where the JSON report is written"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "resolve xdist, print the plan and exit, booting no suite. Writes no report "
            "unless --report names one (item U10)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return _run_bug(args)
    except (
        shuru_mod.ShuruError,
        runner_mod.RunnerError,
        prepare_mod.PrepareError,
        SplitError,
        OSError,
    ) as exc:
        print(f"probe-xdist: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
