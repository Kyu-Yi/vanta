"""Freeze a project's VCS-derived version into the tree the sandbox runs.

[VANTA-WHY] The seed this harness stages for a run is a copy of the execution
tree with ``SKIP_DIRS`` left out (``harness::copy_tree_excluding``), and `.git`
is one of those names — deliberately: the repository history carries the FIX
commit, so shipping it into the sandbox would hand the Creator the answer
(§1.2). The cost of that exclusion is the bug this module fixes. ``pandas`` and
``matplotlib`` derive ``__version__`` at import time with **versioneer**, which
runs ``git describe`` against the tree root; with no `.git` it falls through its
own chain to the placeholder ``0+unknown``. That placeholder is not merely
cosmetic — ``xarray``'s ``pdcompat`` does
``LooseVersion(pd.__version__) < "0.25.0"``, and ``LooseVersion`` cannot compare
``"0+unknown"`` with a version string, so it raises
``TypeError: '<' not supported between instances of 'str' and 'int'`` in every
module that imports xarray. On pandas/1 that is two test modules
(``pandas/tests/test_downstream.py``, ``pandas/tests/generic/test_to_xarray.py``)
and pytest stops the whole session with ``Interrupted: 2 errors during
collection`` (exit 2) — measured: ``collected=88592 bad_collectors=2``.

[VANTA-WHY] The fix is the one an sdist already performs: **freeze the version
into the version file**. ``setup.py sdist`` replaces ``pandas/_version.py`` with
versioneer's SHORT form — a header plus a literal ``version_json`` blob the
module returns verbatim — so a tarball with no `.git` still knows its version.
We do exactly that at **prepare** time, while the exported trees still hold
their `.git` (``trees.export_tree`` is a real ``git clone``), and write the
frozen file into the tree the sandbox is seeded from. In-session
``__version__`` then equals prepare's, byte for byte, without any `.git` in
either place.

The project is never guessed: a versioneer version file **names itself** in its
own ``get_config().versionfile_source``, and only a file whose declared source
path equals its own relative path is used. A tree with no versioneer and no
``setuptools_scm`` — PySnooper, whose version is a literal in ``setup.py`` — is
left exactly as it was.

[VANTA-WHY] M3 item B7 — measuring the version is the one thing this harness used
to do by *running third-party checkout code on the host*: ``runpy.run_path`` over
the project's own ``_version.py``, and an ``import setuptools_scm`` with ``cwd``
set to the tree. Both now run in **Shuru**, in a probe VM of their own that mounts
the prepare tree **read-only**, separate from every agent and test sandbox
(:func:`_guest_probe`). The tree is mounted because versioneer answers by running
``git describe`` against it and needs its ``.git`` — and ``.git`` must never reach
a sandbox the Creator or a graded run can read, which is exactly why the mount is a
dedicated, read-only one. Nothing the guest learns is written by the guest: the host
renders the frozen file itself (:func:`freeze`), every path it writes is contained
to the tree (:func:`_contained_target`), and the row says which sandbox the
measurement ran in (:data:`_PROBE_SANDBOX`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from . import shuru

# The sentinel the probe writes before its JSON payload, so a project's own
# import-time chatter (versioneer's ``verbose`` branch, a DeprecationWarning on
# stderr, anything a future version file prints) cannot corrupt the reading.
_PROBE_SENTINEL = "<<<VANTA-PIN>>>"

GUEST_ROOT = "/workspace"
"""Where the probe VM sees the tree it measures — the same mount point the suites use."""

PROBE_INTERPRETER = "python"
"""The interpreter the probe runs, as named inside the checkpoint.

Named ``python`` and not ``python3`` because that is the element the split's
``test_cmd`` starts with and the one ``runner.python_version_of`` reports: the probe
has to run the project's machinery under the *same* interpreter its suites run under
(the rule ``suite.import_probe`` follows), and a second spelling here could name an
interpreter the checkpoint's site-packages were not built for.
"""

PROBE_TIMEOUT_SECS = 300
"""A probe is one import and one ``git describe``; minutes, not the suite's timeout."""

_PROBE_NOTE_LIMIT = 300
"""How much of a failed probe's own complaint reaches :attr:`PinOutcome.reason`.

The reason is prose a reader sees in ``prepare.json`` and in the prepare table, not a
log: one bounded line says *which* failure it was, and the guest's full output is what
a re-run of the probe is for.
"""

_PROBE_SANDBOX = "shuru"
"""The value the row records for *which sandbox measured the version*.

Recorded rather than assumed because "the version was measured" and "the version was
measured without executing checkout code on the host" are different claims, and only
this field distinguishes a row prepared after B7 from one prepared before it.
"""

# Directories never descended while looking for a version file. The same
# reasoning as ``context::SKIP_DIRS``: a `.git` object database, a build tree and
# a vendored environment hold no project version file, and walking them costs
# far more than the search is worth on a tree like pandas'.
_PRUNE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".eggs",
        "htmlcov",
    }
)

# versioneer's placeholder for "I could not compute a version" — the string this
# whole module exists to keep out of a sandbox.
_PLACEHOLDER = "0+unknown"

# versioneer 0.15's ``SHORT_VERSION_PY``, reproduced so the frozen file is
# byte-identical to the one ``setup.py sdist`` would have written. Reproduced
# rather than imported because the constant lives in the *project's* vendored
# ``versioneer.py``, which the sandbox also carries and which is not on this
# host's path. ``versions_from_file`` reads it back with
# ``version_json = '''\n(.*)'''  # END VERSION_JSON``, so the shape is part of
# the format, not decoration.
_SHORT_VERSION_PY = """
# This file was generated by 'versioneer.py' (0.15) from
# revision-control system data, or from the parent directory name of an
# unpacked source archive. Distribution tarballs contain a pre-generated copy
# of this file.

from warnings import catch_warnings
with catch_warnings(record=True):
    import json
import sys

version_json = '''
%s
'''  # END VERSION_JSON


def get_versions():
    return json.loads(version_json)
"""


@dataclass(frozen=True)
class VersionPin:
    """One project's frozen version, and the file that now carries it."""

    kind: str
    """``"versioneer"`` or ``"setuptools_scm"`` — which machinery was frozen."""

    file: str
    """Tree-relative path of the version file that was rewritten."""

    contents: str
    """The exact bytes to write into :attr:`file`.

    Carried on the pin rather than re-derived at write time: measuring the
    version means running the project's own machinery against its ``.git``, and
    the write happens after that repository may already have been copied away.
    One measurement, one rendered file, one write.
    """

    version: str
    """The version string the tree reports, as prepare measured it."""

    full_revisionid: Optional[str]
    """versioneer's ``full-revisionid``, kept because the frozen form keeps it.

    [VANTA-WHY] Which small dict is frozen is not a free choice. versioneer's own
    ``write_to_version_file`` dumps the whole ``versions`` mapping — version,
    full-revisionid, dirty, error — and pandas' ``pandas/util/_print_versions.py``
    reads ``versions["full-revisionid"]`` back out of it, while
    ``pandas/tests/util/test_show_versions.py`` asserts the printed commit is 40
    hex digits. Dropping the field would turn a test that passes in prepare into
    one that fails in-session, which is exactly the prepare/in-session divergence
    the pin exists to remove. It reveals the **buggy tree's own** commit and
    never the fix's, and it is the string an sdist of that commit already
    carries.
    """

    sha256: str
    """sha256 of the frozen file's bytes, for the row."""


class VersionPinError(RuntimeError):
    """A freeze was asked to write outside the tree it was told to freeze."""


@dataclass(frozen=True)
class ProbeVM:
    """Where and how a probe runs: the checkpoint, the VM's size, and the interpreter.

    Carried as one object rather than as four parameters because every probe this
    module makes runs the same way, and a probe that ran under a *different*
    checkpoint or interpreter than its siblings would be measuring a different
    environment without saying so.
    """

    checkpoint: str
    """The Shuru checkpoint to boot — the project's own environment layer."""

    disk_mb: int
    """The VM's disk, passed straight to ``shuru run --disk-size``."""

    python: str = PROBE_INTERPRETER
    """The interpreter to run, named as it is inside the checkpoint."""

    memory_mb: Optional[int] = None
    """The checkpoint's recorded VM memory, or ``None`` for Shuru's own default."""


def _sentinel_payload(out: str) -> Any:
    """The JSON a probe wrote after :data:`_PROBE_SENTINEL`, or ``None``.

    One reader for both probes, so the sentinel convention has one spelling: a
    project's own import-time chatter (versioneer's ``verbose`` branch, a
    ``DeprecationWarning``, anything a future version file prints) lands before the
    marker and is ignored, and output with no marker at all answers ``None`` — "the
    probe said nothing", which is a different answer from any payload.
    """
    marker = out.find(_PROBE_SENTINEL)
    if marker < 0:
        return None
    try:
        return json.loads(out[marker + len(_PROBE_SENTINEL) :])
    except json.JSONDecodeError:
        return None


def _probe_note(result: shuru.ExecResult) -> str:
    """Why a probe that should have written the sentinel did not, as one line.

    [VANTA-WHY] M3 item P1. Before this, the only record of a failed measurement was
    "could not be measured": ``_guest_probe`` returned ``stdout`` and threw the rest of
    the result away, so a crash on import, a mount that refused a write, a VM that
    would not boot and a timeout all read the same and the cause could only be found by
    booting the probe again by hand. The guest's ``stderr`` is quoted first because
    that is where a traceback goes; the exit code is always named, because "exit 0 and
    said nothing" and "exit 127" are different failures. Bounded, because this text
    ends up inside :attr:`PinOutcome.reason` and from there inside ``prepare.json``.
    """
    if result.timed_out:
        return f"the probe timed out after {PROBE_TIMEOUT_SECS}s"
    ended = (
        f"exit code {result.exit_code}"
        if result.exit_code is not None
        else "no exit code"
    )
    detail = " ".join((result.stderr.strip() or result.stdout.strip()).split())
    if len(detail) > _PROBE_NOTE_LIMIT:
        detail = detail[: _PROBE_NOTE_LIMIT - 1] + "…"
    if not detail:
        return f"the probe wrote nothing and ended with {ended}"
    return f"the probe ended with {ended}: {detail}"


def _guest_probe(
    tree: Path,
    program: str,
    argv: Sequence[str] = (),
    *,
    vm: ProbeVM,
) -> shuru.ExecResult:
    """Run ``program`` over ``tree`` in a read-only Shuru VM; return its result.

    [VANTA-WHY] M3 item B7. Two properties are the whole reason this exists:

    * The mount is **read-only** (``shuru.run``'s ``allow_host_writes=False``), so
      nothing a probe does can change the tree, the batch or anything else on the
      host — and a probe that *tries* to write (``setuptools_scm`` writing its
      ``write_to`` file) fails inside the VM rather than editing the tree the run is
      seeded from.
    * The VM is **dedicated**: it exists for this probe, and it mounts the prepare
      tree, which carries ``.git``. versioneer cannot answer without a repository, and
      ``.git`` must never reach a sandbox the Creator or a graded run can read — so
      the history is exposed to this one throwaway boot and to nothing else.

    ``tree`` appears inside as :data:`GUEST_ROOT`, so callers pass guest paths (built
    from :data:`GUEST_ROOT`); the host path does not exist in the VM.

    The whole :class:`shuru.ExecResult`, not its ``stdout``: [VANTA-WHY] M3 item P1 —
    a probe that never reached its sentinel reported "could not be measured" and
    nothing else, because ``stderr``, the exit code and the timeout flag were dropped
    here. They are the only evidence of a failed measurement, and
    :func:`_probe_note` is the one reader of them.
    """
    return shuru.run(
        checkpoint=vm.checkpoint,
        host_root=tree,
        command=[vm.python, "-c", program, *argv],
        disk_mb=vm.disk_mb,
        memory_mb=vm.memory_mb,
        guest_cwd=GUEST_ROOT,
        timeout_secs=PROBE_TIMEOUT_SECS,
        allow_host_writes=False,
        # Writing ``__pycache__`` beside a probe's import is the one write a read-only
        # mount would still *attempt*; the suites set the same variable for the same
        # reason.
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
    )


def _probe(path: Path, tree: Path, *, vm: ProbeVM) -> tuple[Optional[dict[str, Any]], str]:
    """Load a versioneer version file in the probe VM and read both of its maps.

    In the VM rather than in this process: the file is third-party checkout code whose
    ``get_versions`` shells out to ``git``, so it runs where the harness runs every
    other piece of a project's code — inside Shuru, over a read-only mount
    (:func:`_guest_probe`). A ``runpy`` in *this* interpreter would also leave the
    project's module namespace in the harness process, the one place it must never be.

    Returns ``(payload, note)``. The payload is the probe's own answer or ``None``;
    the note is [VANTA-WHY] M3 item P1's one line about *why* there is no answer —
    empty whenever the payload arrived. Keeping it beside the payload rather than
    inside it means the caller reports a failed measurement without having to re-read
    the guest's stderr, and :func:`_versioneer_version_file` can carry a different note
    for each candidate file it could not measure.
    """
    program = (
        "import json, runpy, sys\n"
        "ns = runpy.run_path(sys.argv[1])\n"
        "cfg = ns['get_config']()\n"
        "payload = {\n"
        "    'vcs': getattr(cfg, 'VCS', None),\n"
        "    'style': getattr(cfg, 'style', None),\n"
        "    'tag_prefix': getattr(cfg, 'tag_prefix', None),\n"
        "    'parentdir_prefix': getattr(cfg, 'parentdir_prefix', None),\n"
        "    'versionfile_source': getattr(cfg, 'versionfile_source', None),\n"
        "    'versions': ns['get_versions'](),\n"
        "}\n"
        "sys.stdout.write('" + _PROBE_SENTINEL + "' + json.dumps(payload))\n"
    )
    result = _guest_probe(
        tree,
        program,
        [f"{GUEST_ROOT}/{path.relative_to(tree).as_posix()}"],
        vm=vm,
    )
    payload = _sentinel_payload(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("versions"), dict):
        return None, _probe_note(result)
    return payload, ""


def _walk_version_files(tree: Path):
    """Every ``_version.py`` outside the pruned directories, shallowest first."""
    stack: list[tuple[Path, int]] = [(tree, 0)]
    found: list[Path] = []
    while stack:
        directory, depth = stack.pop()
        if depth > 8:
            continue
        try:
            with os.scandir(directory) as entries:
                listing = sorted(entries, key=lambda e: e.name)
        except OSError:
            continue
        for entry in listing:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _PRUNE_DIRS:
                    stack.append((Path(entry.path), depth + 1))
            elif entry.name == "_version.py" and entry.is_file(follow_symlinks=False):
                found.append(Path(entry.path))
    yield from sorted(found, key=lambda p: len(p.parts))


def _declared_version_file(tree: Path) -> Optional[str]:
    """``versionfile_source`` from ``setup.cfg``'s ``[versioneer]`` section.

    The cheap path, and the one pandas takes. matplotlib keeps its versioneer
    configuration out of ``setup.cfg`` (a ``setup.cfg.template`` is rendered
    during its build), so the section is a shortcut and never the only route.
    """
    try:
        lines = (tree / "setup.cfg").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    inside = False
    for raw in lines:
        stripped = raw.split("#", 1)[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == "[versioneer]"
            continue
        if inside and stripped.startswith("versionfile_source"):
            _, _, value = stripped.partition("=")
            value = value.strip()
            if value:
                return value
    return None


def _looks_like_versioneer(path: Path) -> bool:
    """Whether ``path`` is worth a probe — a cheap pre-filter, never a decision."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "def get_versions" in text and "def get_config" in text


def _versioneer_version_file(
    tree: Path, *, vm: ProbeVM
) -> tuple[Optional[tuple[Path, dict[str, Any]]], list[tuple[Path, str]], bool]:
    """The tree's own versioneer version file and its measurement, and the rest.

    Returns ``(found, unmeasured, probed)``. ``found`` is ``(candidate, payload)``
    for the one versioneer-shaped file that names *itself* in
    ``get_config().versionfile_source`` — the property that makes this a detector
    rather than a guess: a vendored copy of some other project's ``_version.py``,
    or a ``_version.py`` that is not versioneer's, names a different path and is
    rejected. ``unmeasured`` is ``(path, note)`` for each versioneer-shaped file whose
    probe returned nothing at all, which on a healthy host is empty — an importable
    Python file with ``get_config``/``get_versions`` always probes. A non-empty list
    therefore says the measurement itself failed (a VM that would not boot, a file
    that raises on import), and the caller reports that — with the note the probe left
    ([VANTA-WHY] M3 item P1) — rather than reading it as "this project needs no
    pinning", which is the silent no-op this module exists to remove.

    ``probed`` is whether a probe VM was booted at all — the fact the row records.
    It is not ``bool(found or unmeasured)``: a tree whose only ``_version.py`` files
    are vendored copies of somebody else's boots a VM, gets an answer and rejects it,
    and that row must still say the measurement happened in the sandbox.
    """
    declared = _declared_version_file(tree)
    candidates: list[Path] = []
    if declared:
        configured = tree / declared
        if configured.is_file() and _looks_like_versioneer(configured):
            candidates.append(configured)
    candidates += [
        path for path in _walk_version_files(tree)
        if path not in candidates and _looks_like_versioneer(path)
    ]
    unmeasured: list[tuple[Path, str]] = []
    for candidate in candidates:
        rel = candidate.relative_to(tree).as_posix()
        payload, note = _probe(candidate, tree, vm=vm)
        if payload is None:
            unmeasured.append((candidate, note))
            continue
        if payload.get("vcs") != "git":
            continue
        named = payload.get("versionfile_source")
        if isinstance(named, str) and named.strip("./") == rel:
            return (candidate, payload), unmeasured, True
    return None, unmeasured, bool(candidates)


def _pyproject_setuptools_scm(tree: Path) -> Optional[dict[str, str]]:
    """The ``[tool.setuptools_scm]`` table's simple scalar keys, or ``None``.

    Hand-read rather than parsed with ``tomllib``: the harness runs on the
    system Python, and the fields this needs (``write_to``, ``version_file``)
    are single-line scalars. A file with no such table, or no usable scalar in
    it, is reported as absent so a project is never marked ``setuptools_scm``
    on the strength of a mention in prose.
    """
    manifest = tree / "pyproject.toml"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    inside = False
    scalars: dict[str, str] = {}
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == "[tool.setuptools_scm]"
            continue
        if not inside or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value:
            scalars[key] = value
    return scalars or None


def _setup_py_uses_scm(tree: Path) -> bool:
    try:
        text = (tree / "setup.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return "use_scm_version" in text


def _setuptools_scm_write_to(tree: Path, table: dict[str, str]) -> Optional[str]:
    """The file ``setuptools_scm`` is asked to write, if it is asked to write one."""
    for key in ("version_file", "write_to"):
        value = table.get(key)
        if value:
            return value
    return None


def _declared_version_files(tree: Path) -> set[str]:
    """The version file(s) the project's own configuration names. (item T4)

    [VANTA-WHY] M3 item T4. A versioneer-shaped ``_version.py`` that raises on import
    cannot be asked whether it names itself (``get_config().versionfile_source``), so
    the property :func:`_versioneer_version_file` uses to tell the project's own file
    from a vendored copy of somebody else's is unavailable for exactly the files that
    used to decide the outcome. The one ownership the harness *can* establish without
    running the file is the project's own configuration: versioneer's
    ``versionfile_source`` in ``setup.cfg``'s ``[versioneer]`` section, and
    setuptools_scm's ``write_to``/``version_file`` in ``pyproject.toml``'s
    ``[tool.setuptools_scm]`` table. Those are the paths this returns; a probe-failed
    candidate that is not one of them is a vendored copy and must be skipped rather
    than allowed to refuse the bug.
    """
    declared: set[str] = set()
    versioneer = _declared_version_file(tree)
    if versioneer:
        declared.add(PurePosixPath(versioneer).as_posix())
    table = _pyproject_setuptools_scm(tree)
    if table:
        write_to = _setuptools_scm_write_to(tree, table)
        if write_to:
            declared.add(PurePosixPath(write_to).as_posix())
    return declared


def _setuptools_scm_version(tree: Path, *, vm: ProbeVM) -> tuple[Optional[str], str]:
    """``setuptools_scm.get_version(root=.)`` in the probe VM, and why not, if not.

    In the VM for the same reason the versioneer probe is (:func:`_guest_probe`): the
    call imports the *checkout's* ``setuptools_scm`` and asks it to walk the checkout's
    ``.git``, so running it here would be host execution of third-party code over a
    repository this process must not open. ``root='.'`` resolves against
    ``guest_cwd``, which :func:`_guest_probe` sets to the mounted tree.

    Returns ``(version, note)`` — the same pair :func:`_probe` returns, for the same
    reason: [VANTA-WHY] M3 item P1, a probe that never reached its sentinel must say
    which failure it was. The program here exits ``3`` when the sandbox cannot import
    ``setuptools_scm`` and ``4`` when ``get_version`` raises, so the exit code in the
    note is itself the answer.
    """
    program = (
        "import json, sys\n"
        "try:\n"
        "    import setuptools_scm\n"
        "except Exception:\n"
        "    sys.exit(3)\n"
        "try:\n"
        "    value = setuptools_scm.get_version(root='.')\n"
        "except Exception:\n"
        "    sys.exit(4)\n"
        "sys.stdout.write('" + _PROBE_SENTINEL + "' + json.dumps(str(value)))\n"
    )
    result = _guest_probe(tree, program, vm=vm)
    value = _sentinel_payload(result.stdout)
    if isinstance(value, str) and value:
        return value, ""
    return None, _probe_note(result)


def _version_tuple(version: str) -> tuple[Any, ...]:
    """A best-effort ``version_tuple`` for the ``setuptools_scm`` frozen file.

    Only numbers and simple text segments survive; anything else becomes the raw
    token. The value is what a project's own ``__version_tuple__`` consumers read,
    and ``setuptools_scm`` itself renders it from the parsed version, so a tuple
    that is merely faithful to the string is the honest approximation here — the
    one fact this module is responsible for is the *version string*.
    """
    parts: list[Any] = []
    for token in re.split(r"[.+_-]", version):
        if not token:
            continue
        parts.append(int(token) if token.isdigit() else token)
    return tuple(parts)


def _setuptools_scm_contents(version: str) -> str:
    return (
        "# file generated by setuptools_scm\n"
        "# don't change, don't track in version control\n"
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = {_version_tuple(version)!r}\n"
    )


def _placeholder_free(versions: dict[str, Any]) -> Optional[str]:
    """Why the measured version is not usable, or ``None`` when it is."""
    version = versions.get("version")
    if not isinstance(version, str) or not version:
        return "the project reported no version"
    if version.startswith(_PLACEHOLDER):
        return (
            f"the project reported the placeholder {version!r}; its version "
            "machinery could not see a repository, so there is nothing to freeze"
        )
    if versions.get("error"):
        return f"the project reported a version error: {versions['error']!r}"
    return None


@dataclass(frozen=True)
class PinOutcome:
    """What ``measure`` found: a pin to write, or the reason there is none.

    ``reason is None`` and ``pin is None`` is the honest "this project needs no
    pinning" — PySnooper keeps its version as a literal in ``setup.py``, so its
    tree behaves identically with and without a repository. A ``reason`` is
    always the other case: version machinery *was* found and could not produce a
    real version. The two must stay distinguishable, because only the first is a
    no-op and only the second is a bug in this module's own reading of a project.
    """

    kind: Optional[str]
    pin: Optional["VersionPin"]
    reason: Optional[str]

    probed: bool = False
    """Whether a probe VM was booted to reach this outcome.

    [VANTA-WHY] M3 item B7. ``False`` for a tree whose version is a literal with no
    machinery to read — PySnooper, where no sandbox ever ran — and ``True`` whenever a
    probe booted, even one that only rejected a vendored ``_version.py``. Recorded so
    "measured in Shuru" is a claim each row makes rather than one a reader assumes, and
    so nothing silently reports a measurement that never happened.
    """

    measurement_failed: bool = False
    """Whether a version measurement was reached and did not yield a pin.

    [VANTA-WHY] M3 item P3. :attr:`reason` alone cannot answer this, and prepare must
    ask: the setuptools_scm-without-``write_to`` outcome also carries a ``reason`` — the
    project simply keeps no version file, which is a fact about its configuration rather
    than a failed reading — and prepare has always gone on to measure that bug. Every
    other reason means the machinery was reached and could not produce a usable version,
    and a bug left in that state runs its native suite against a tree whose project
    reports the ``0+unknown`` placeholder: on pandas/1 that aborts collection and ends
    the row ``native_vacuous``, which is why the caller refuses it at prepare time
    instead. ``True`` at exactly the three sites that mean it, so the one carve-out is a
    fact stated in the code rather than inferred from :attr:`probed`.
    """

    skipped: tuple[str, ...] = ()
    """Tree-relative paths of versioneer-shaped files that were passed over. (item T4)

    [VANTA-WHY] M3 item T4. A ``_version.py`` that matches the text pre-filter but
    raises on import cannot be asked whether it names itself, so the harness cannot
    call it the project's own version file — yet it used to decide the whole
    measurement, refusing a bug whose project had a perfectly measurable
    ``setuptools_scm`` version. A probe-failed candidate the project's own versioneer
    or setuptools configuration does *not* name is therefore recorded here and
    skipped: the measurement falls through to the other strategies, and a reader can
    still see *which* file was passed over rather than finding it vanished from the
    record.
    """


def measure(tree: Path, *, vm: ProbeVM) -> PinOutcome:
    """The version to freeze for ``tree``, and what to record about it.

    ``pin is None`` means exactly "change nothing", and :attr:`PinOutcome.reason`
    says which of the two ways that happened. The caller writes nothing in that
    case — the record is the whole of the change.

    ``vm`` is where both probes run (:func:`_guest_probe`): a read-only Shuru VM
    dedicated to this measurement, so the version is read without executing any
    checkout code on the host and without opening the tree's ``.git`` here.
    """
    found, unmeasured, probed = _versioneer_version_file(tree, vm=vm)

    # [VANTA-WHY] M3 item T4 — a probe-failed candidate decides the outcome only when
    # it is the project's own version file. Ownership is established from the project's
    # versioneer/setuptools configuration (:func:`_declared_version_files`); a candidate
    # that configuration does not name is an unverified vendored copy of somebody
    # else's, so it is recorded, skipped, and the measurement falls through to the
    # strategies below. A tree that names no version file at all keeps the old reading:
    # the walk found a version file and nothing contradicts it, so it is the project's.
    declared = _declared_version_files(tree)
    fatal: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []
    for path, note in unmeasured:
        (fatal if not declared or path.relative_to(tree).as_posix() in declared
         else skipped).append((path, note))
    skipped_paths = tuple(path.relative_to(tree).as_posix() for path, _ in skipped)

    if found is not None:
        candidate, payload = found
        versions = payload["versions"]
        unusable = _placeholder_free(versions)
        if unusable is not None:
            return PinOutcome(
                kind="versioneer",
                pin=None,
                reason=unusable,
                probed=probed,
                # [VANTA-WHY] M3 item P3 — the probe answered and the answer is the
                # placeholder: the machinery saw no repository, so there is nothing to
                # freeze and the sandbox would report ``0+unknown``. A failed
                # measurement like the others below.
                measurement_failed=True,
                skipped=skipped_paths,
            )
        contents = _short_version_contents(versions)
        pin = VersionPin(
            kind="versioneer",
            file=candidate.relative_to(tree).as_posix(),
            contents=contents,
            version=str(versions["version"]),
            full_revisionid=_str_or_none(versions.get("full-revisionid")),
            sha256=hashlib.sha256(contents.encode("utf-8")).hexdigest(),
        )
        return PinOutcome(kind="versioneer", pin=pin, reason=None, probed=probed,
                          skipped=skipped_paths)

    if fatal:
        # A versioneer-shaped file that could not be measured at all, and that the
        # project's own configuration names as its version source. On a healthy host
        # this cannot happen, so it is reported rather than folded into the "no
        # machinery here" answer — a silent no-op would leave the sandbox reporting a
        # placeholder version, which is the very bug being fixed.
        named = ", ".join(
            path.relative_to(tree).as_posix() for path, _ in fatal[:3]
        )
        more = f" (+{len(fatal) - 3} more)" if len(fatal) > 3 else ""
        reason = (
            f"a versioneer-shaped version file was found but could not be "
            f"measured: {named}{more}"
        )
        # [VANTA-WHY] M3 item P1 — the probe's own complaint, appended rather than
        # left in the guest's discarded stderr: this reason is the only place a reader
        # of the disabled row learns *which* failure it was.
        why = "; ".join(note for _, note in fatal[:3] if note)
        return PinOutcome(
            kind="versioneer",
            pin=None,
            reason=f"{reason} — {why}" if why else reason,
            probed=probed,
            # [VANTA-WHY] M3 item P3 — the file this project names as its own version
            # source exists and the probe could not read it. This is the pandas/1
            # shape: prepare went on, no pin was frozen, and the row died
            # ``native_vacuous`` two stages later with the cause nowhere in the gate.
            measurement_failed=True,
            skipped=skipped_paths,
        )

    table = _pyproject_setuptools_scm(tree)
    if table is None and not _setup_py_uses_scm(tree):
        return PinOutcome(kind=None, pin=None, reason=None, probed=probed,
                          skipped=skipped_paths)
    if table is None:
        table = {}
    write_to = _setuptools_scm_write_to(tree, table)
    if not write_to:
        return PinOutcome(
            kind="setuptools_scm",
            pin=None,
            reason=(
                "setuptools_scm is configured without a write_to/version_file, so "
                "the project keeps no version file for a sandbox to read"
            ),
            probed=probed,
            skipped=skipped_paths,
        )
    version, why = _setuptools_scm_version(tree, vm=vm)
    # A VM was booted to ask setuptools_scm, whatever it answered.
    probed = True
    if not version:
        return PinOutcome(
            kind="setuptools_scm",
            pin=None,
            reason=(
                "setuptools_scm could not compute a version in the probe sandbox"
                + (f" — {why}" if why else "")
            ),
            probed=probed,
            # [VANTA-WHY] M3 item P3 — a probe booted and answered nothing, so this is
            # a failed measurement too. The ``write_to``-less branch above is the only
            # reason that is not.
            measurement_failed=True,
            skipped=skipped_paths,
        )
    contents = _setuptools_scm_contents(version)
    pin = VersionPin(
        kind="setuptools_scm",
        file=write_to,
        contents=contents,
        version=version,
        full_revisionid=None,
        sha256=hashlib.sha256(contents.encode("utf-8")).hexdigest(),
    )
    return PinOutcome(kind="setuptools_scm", pin=pin, reason=None, probed=probed,
                      skipped=skipped_paths)


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _short_version_contents(versions: dict[str, Any]) -> str:
    """The file body for ``versions``, exactly as versioneer would write it.

    ``json.dumps(versions, sort_keys=True, indent=1, separators=(",", ": "))`` is
    versioneer 0.15's own rendering (``write_to_version_file``), reproduced so the
    frozen file is byte-identical to an sdist's.
    """
    body = json.dumps(versions, sort_keys=True, indent=1, separators=(",", ": "))
    return _SHORT_VERSION_PY % body


def _contained_target(tree: Path, rel: str) -> Path:
    """``rel`` resolved beneath ``tree``, or raise :class:`VersionPinError`.

    [VANTA-WHY] M3 item B7. ``pin.file`` is not always a path this module chose: for
    a ``setuptools_scm`` project it is the ``write_to``/``version_file`` string out of
    the checkout's own ``pyproject.toml``, and :func:`freeze` writes it with
    ``write_text``. Three strings would take that write out of the tree — an absolute
    path (``/etc/passwd``), a parent traversal (``../../x``) and a symlink the checkout
    planted that points outside the tree — and each is refused here, before the write,
    rather than discovered after it. The check resolves the deepest existing ancestor
    so a symlinked *directory* cannot launder the escape either.
    """
    raw = Path(rel)
    if raw.is_absolute() or rel.startswith("~"):
        raise VersionPinError(
            f"refusing to write outside the tree: {rel!r} is an absolute path"
        )
    if any(part == ".." for part in raw.parts):
        raise VersionPinError(
            f"refusing to write outside the tree: {rel!r} traverses a parent directory"
        )
    root = tree.resolve()
    target = root / raw
    # Resolve symlinks on the deepest already-existing ancestor, then re-attach the
    # parts that do not exist yet: a symlinked directory, or a symlink sitting exactly
    # on the target, would otherwise point the write outside the tree unnoticed.
    probe = target
    tail: list[str] = []
    while not os.path.lexists(probe) and probe != probe.parent:
        tail.append(probe.name)
        probe = probe.parent
    resolved = probe.resolve()
    for name in reversed(tail):
        resolved = resolved / name
    if root not in resolved.parents:
        raise VersionPinError(
            f"refusing to write outside the tree: {rel!r} resolves to {resolved}"
        )
    return target


def freeze(tree: Path, pin: VersionPin) -> None:
    """Write ``pin``'s frozen version file into ``tree``, contained to ``tree``."""
    target = _contained_target(tree, pin.file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(pin.contents, encoding="utf-8")


def as_record(pin: Optional[VersionPin], *, reason: Optional[str] = None,
              probed: bool = False, skipped: Sequence[str] = ()) -> dict[str, Any]:
    """The dict ``prepare.json`` and the manifest row carry.

    ``pinned`` is the load-bearing field: a reader must be able to tell "this
    row's version was frozen" from "this project needed no pinning" (``reason``
    is ``None``) and from "it needed one and the freeze failed" (``reason`` is
    set), because only the first is comparable to a post-pin row.

    ``probe_sandbox`` records *which sandbox measured the version*: ``"shuru"`` when a
    probe VM booted (:func:`measure`'s ``probed``), ``None`` when the project has no
    version machinery to read and nothing was measured at all. [VANTA-WHY] M3 item B7 —
    present on every row this writes, so a reader can tell a post-B7 row (measured
    without host execution of checkout code) from a pre-B7 one, which lacks the key.

    ``skipped`` names the versioneer-shaped files that were passed over because the
    project's own configuration did not name them (item T4), so a reader can see that
    a vendored copy was found and set aside rather than silently ignored.
    """
    sandbox = _PROBE_SANDBOX if probed else None
    if pin is None:
        return {"pinned": False, "kind": None, "file": None, "version": None,
                "full_revisionid": None, "sha256": None, "reason": reason,
                "probe_sandbox": sandbox, "skipped": list(skipped)}
    return {
        "pinned": True,
        "kind": pin.kind,
        "file": pin.file,
        "version": pin.version,
        "full_revisionid": pin.full_revisionid,
        "sha256": pin.sha256,
        "reason": None,
        "probe_sandbox": sandbox,
        "skipped": list(skipped),
    }
