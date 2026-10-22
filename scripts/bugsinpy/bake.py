"""Environment checkpoints, baked once and reused by every run (design §3, §11 step 3).

Two layers, because they change at different rates:

``vanta-bugsinpy-py<X.Y>``  the interpreter and the pinned pytest, one per Python
                            *minor* version used by the split.
``vanta-bi-<project>-<N>``  one bug's third-party dependencies on top of that.

[VANTA-WHY] The interpreter layer is keyed by minor version, not by the
``python_version`` a ``bug.info`` declares. BugsInPy writes patch levels
(``3.8.1`` on PySnooper, ``3.8.3`` on black); keying on those would bake three
identical 3.8 interpreters and then disagree with §3.2, which names the layers
``vanta-bugsinpy-py3.6/3.7/3.8``.

[VANTA-WHY] Debian trixie ships OpenSSL 3 and CPython 3.6/3.7/3.8 were built against
OpenSSL 1.1.1, so a naive "install Python 3.8" was expected to produce an
interpreter with no working ``ssl``. Measured on the base image (``vanta-python``,
2026-10): Debian 13, Python 3.13.5, ``OpenSSL 3.5.6``, ``uv`` absent, ``curl``,
``git``, ``make`` and ``gcc`` present, and no ``python`` at all (only
``/usr/bin/python3 -> python3.13``). The interpreter layer is therefore built by
``uv`` (python-build-standalone) into a virtualenv: measured, that yields CPython
3.8.20 carrying its own **OpenSSL 3.0.14**, so nothing compiles, nothing depends on
the base image's OpenSSL, and ``pip install`` works. py3.6/3.7 are handled the same
way if a standalone build exists for them, and otherwise by building OpenSSL 1.1.1
into the layer (§3.2); which one was used is recorded in
``/opt/vanta-deps-freeze.txt`` next to the layer's marker.

[VANTA-WHY] Every bake ends in ``touch /opt/vanta-deps-ok`` and is verified by
booting the result (``shuru.checkpoint_create(marker=…)``), because
``shuru checkpoint create`` caches a checkpoint *even when the inner command
fails*. The assertions before the ``touch`` are the wave-1 gate: the interpreter is
a **venv** (not the bare standalone), ``import ssl`` builds a context, a real
``pip install`` succeeds, and ``pytest --version`` plus ``import pytest_jsonreport``
answer. Each of those has failed for a real reason at least once:
``externally-managed-environment`` when the venv is skipped, and ``uv: not found``
when the installer's target is inferred from an empty ``HOME``.

The disk requirement is a measured fact, not a tuning knob: the base image is
8 GB, so ``--disk-size`` below that is refused outright (``shuru: --disk-size
4096MB is smaller than the base image (8192MB)``), which is why ``DISK_MB`` is
16 GB everywhere.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .bugsinpy_meta import read_metadata_text
from . import suite as suite_mod
from . import trees as tree_mod
from .shuru import ExecResult, checkpoint_create, checkpoint_delete, checkpoint_exists
from .split import Split, SplitError

DEFAULT_BASE = "vanta-python"
MARKER = "/opt/vanta-deps-ok"
"""Touched last, only after every assertion has passed."""

MIN_DISK_MB = 8192
"""The base image's disk. shuru refuses anything smaller."""

BAKE_DISK_MB = 16384
BAKE_TIMEOUT_SECS = 5400
"""90 min. py3.6/3.7 may compile OpenSSL 1.1.1 and CPython from source here."""

HARNESS_PACKAGES = frozenset({"pytest", "pytest-json-report"})
"""Installed by the *interpreter* layer and never touched by a per-bug layer.

[VANTA-WHY] A project's own ``pytest>=4.4.1`` (PySnooper's
``test_requirements.txt``) is a constraint on a package the harness already
provides. Installing the project's choice would downgrade pytest under
``pytest-json-report``, whose plugin is what produces the reports the oracle is
read from. The constraint is therefore *checked* against the harness pin and
recorded; a project that pins an incompatible pytest fails the bug loudly instead
of silently re-pinning the interpreter every wave.
"""

DEPS_DIRNAME = "deps"
"""Where ``bake`` leaves each per-bug layer's dependency provenance.

[VANTA-WHY] The provenance lives outside the layer, in ``runs/bugsinpy/deps/``,
because ``prepare`` has to read it *before* it runs anything inside the layer and
the checkpoint format offers no way to read a file out of a layer without booting
it. It is the record of what the bake decided — sources, pins, skipped lines — so
``prepare`` never has to re-derive a decision that needed the network.
"""


class BakeError(RuntimeError):
    """A layer could not be baked or did not verify."""


@dataclass(frozen=True)
class PytestPin:
    pytest: str
    json_report: str


def minor_version(version: str) -> str:
    """``"3.8.1"`` → ``"3.8"``. The unit an interpreter layer is shared at."""
    parts = str(version).split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise SplitError(f"cannot read a minor version out of {version!r}")
    return f"{parts[0]}.{parts[1]}"


def python_checkpoint_name(minor: str) -> str:
    return f"vanta-bugsinpy-py{minor}"


def bug_checkpoint_name(entry: dict[str, Any]) -> str:
    """The per-bug layer, from the committed split rather than recomputed."""
    name = entry["checkpoint"]
    expected = f"vanta-bi-{entry['project']}-{entry['bug_id']}"
    if name != expected:
        raise SplitError(
            f"{entry['project']}/{entry['bug_id']}: checkpoint {name!r} does not "
            f"follow the vanta-bi-<project>-<N> convention ({expected!r})"
        )
    return name


def pythons_in_split(split: Split, *, roles: Optional[set[str]] = None) -> list[str]:
    return sorted(
        {
            minor_version(entry["python_version"])
            for entry in split.bugs
            if roles is None or entry.get("role", "pilot") in roles
        }
    )


def pins_for_python(split: Split, minor: str) -> PytestPin:
    """The pytest pins every bug on ``minor`` declares, which must agree.

    Read from the split rather than from this file so the pins are part of the
    committed protocol (§3.2). Two bugs on the same interpreter with different pins
    are a contradiction, not a preference: the layer is shared.
    """
    pins = {
        (entry["pytest_version"], entry["json_report_version"])
        for entry in split.bugs
        if minor_version(entry["python_version"]) == minor
    }
    if not pins:
        raise SplitError(f"no bug in the split uses Python {minor}")
    if len(pins) > 1:
        raise SplitError(
            f"Python {minor} bugs disagree about the pytest pin: {sorted(pins)}. "
            "One interpreter layer cannot carry two versions."
        )
    pytest_version, json_report = pins.pop()
    return PytestPin(pytest=pytest_version, json_report=json_report)


# ---------------------------------------------------------------------------
# the interpreter layer
# ---------------------------------------------------------------------------
def python_recipe(minor: str, pin: PytestPin) -> str:
    """Install a standalone CPython and the pinned pytest, then assert both work."""
    venv = f"/opt/venv{minor.replace('.', '')}"
    return "\n".join(
        [
            "set -eux",
            # [VANTA-WHY] Measured on the base image: `sh -c` inside a fresh VM gets
            # HOME=/ (set, but to the root directory), so the uv installer's default
            # target became `//.local/bin` and the next line failed with
            # `uv: not found` (exit 127). The install directory and the cache
            # directory are therefore pinned explicitly, not inferred.
            'export HOME="${HOME:-/root}"',
            "export UV_INSTALL_DIR=/usr/local/bin",
            "export UV_CACHE_DIR=/var/cache/uv",
            "id -un",
            "curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv.sh",
            "sh /tmp/uv.sh",
            "command -v uv",
            "uv --version",
            f"uv venv --no-project --seed --python {minor} {venv}",
            # `python` is the name the split's test_cmd uses and `python3` is what
            # stray project scripts reach for; the base image ships neither (only
            # `python3` -> 3.13), so a missing wrapper means "ran under 3.13, no
            # pytest", which looks like a model failure rather than an environment
            # one.
            #
            # [VANTA-WHY] A wrapper *script*, never a symlink, and written with
            # `mv` rather than `>`. Both halves are measured. There is no PYTHONHOME:
            # `python -c "import sys; print(sys.prefix)"` through a call to
            # `/usr/local/bin/python` → symlink → `/opt/venv38/bin/python` → symlink
            # → the real standalone binary reported the *standalone* prefix, so pip
            # was the standalone's pip and answered
            # `error: externally-managed-environment` (PEP 668) — the interpreter
            # was reached but the venv was not seen. Invoking the same venv through
            # `exec /opt/venv38/bin/python "$@"` reported `sys.prefix=/opt/venv38`
            # and pip 25.0.1, i.e. the venv.
            #
            # [VANTA-WHY] And the redirect rule: in this same VM,
            # `printf … > /usr/local/bin/python` *while that path was the symlink
            # above* followed both links and overwrote the real 70904-byte CPython
            # binary with 43 bytes of shell script that execs the venv — an
            # infinite exec loop (measured: a later `/opt/venv38/bin/python -c …`
            # did not return within 5 s, and a whole probe sat there for 18
            # minutes). Writing to a scratch file and `mv`-ing it into place cannot
            # do that: `rename(2)` never follows the destination.
            f"for name in python python3; do printf '#!/bin/sh\\nexec {venv}/bin/python \"$@\"\\n' > /tmp/vanta-wrapper; chmod +x /tmp/vanta-wrapper; mv -f /tmp/vanta-wrapper \"/usr/local/bin/$name\"; done",
            "command -v python",
            "command -v python3",
            "python -V",
            # The assertions. These are the wave-1 gate (§11 step 3).
            #
            # The venv check is the PEP 668 guard: a bare interpreter is still a
            # working interpreter, so only asserting "python runs" would let the
            # externally-managed failure through to the pip line below.
            'python -c "import sys; assert sys.prefix != sys.base_prefix, sys.prefix"',
            'python -c "import ssl, sys; print(sys.version); print(ssl.OPENSSL_VERSION); print(sys.prefix)"',
            # Functional, not a version-string match: an interpreter whose _ssl
            # failed to build may still report a version, but it cannot make a
            # context. (Measured: the 3.8.20 standalone carries OpenSSL 3.0.14 and
            # builds a context fine, so `ssl` does not depend on the base image.)
            'python -c "import ssl; assert ssl.OPENSSL_VERSION; ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)"',
            # pip is not upgraded: `uv venv --seed` already installs the newest pip
            # that supports the target (25.0.1 on 3.8), so an upgrade is one more
            # network round trip for no change — and against a bare interpreter it
            # is the line that dies on PEP 668.
            f'python -m pip install --no-cache-dir "pytest=={pin.pytest}" '
            f'"pytest-json-report=={pin.json_report}"',
            "python -m pytest --version",
            # A real import of the plugin, so a wheel that installed but cannot be
            # imported fails here rather than inside a wave.
            'python -c "import pytest_jsonreport; print(\'json-report ok\')"',
            "python -m pip freeze > /opt/vanta-deps-freeze.txt",
            f"touch {MARKER}",
        ]
    )


# ---------------------------------------------------------------------------
# the per-bug layer
# ---------------------------------------------------------------------------
PIP_LINE_RE = re.compile(r"^(?:python[0-9.]*\s+-m\s+)?pip3?\s+install\s+(?P<args>.+?)\s*$")
"""BugsInPy's ``setup.sh`` is shell, so its dependency lines are shell too.

``python_toolbox`` on PySnooper/2 and ``decorator`` on PySnooper/3 exist *only*
here — ``requirements.txt`` is 0 bytes on all three — so a layer that read only
``requirements.txt`` would install nothing and the wave would die on
``ModuleNotFoundError``.
"""


SETUP_PY_RE = re.compile(
    r"^(?:python[0-9.]*\s+)?setup\.py\s+(?:install|develop|build)(?:\s|$)"
)
"""``python setup.py install`` — the project, not a dependency."""

_PROJECT_REASON = (
    "installs the project itself: a copy in site-packages would be imported instead "
    "of the graded tree, so every patch would grade the same"
)
_REQUIREMENTS_FILE_REASON = (
    "reads a requirements file (-r); its contents are not one of the four declared "
    "dependency sources, so nothing is guessed from it"
)
_VCS_REASON = (
    "a VCS/URL target: the harness cannot tell whether it is the project or a "
    "dependency, and installing the project would shadow the graded tree"
)
_SETUP_PY_REASON = (
    "runs the project's own setup.py: a non-editable install would put the buggy "
    "package in site-packages, where the tests could import it instead of the patched "
    "tree — every patch would then grade the same (design C3, §3.2)"
)
"""The four "we did not install this" reasons, kept as constants so the recorded
provenance and the tests quote the same words."""


@dataclass(frozen=True)
class SkippedLine:
    """A line that looked like an install and was deliberately not run."""

    source: str
    line: int
    text: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "line": self.line,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DepRef:
    """One requirement exactly as written, and where it was written."""

    name: str
    spec: str
    source: str
    pin_source: Optional[str] = None
    """The one source that wrote the exact ``==`` pin, when one did.

    [VANTA-WHY] Kept apart from ``source`` (which accumulates every spelling of the
    requirement) because it decides §3.2's date rule. A pin BugsInPy wrote itself is
    the benchmark's reference environment and is installed as written; a pin the
    *project* wrote is subject to the buggy commit's date like any other
    requirement. Asking "did any of these sources pin it?" after merging would
    mis-attribute a project pin whenever BugsInPy declared the same package as a
    range instead of a pin.
    """
    commit: Optional[str] = None
    """The commit this declaration was read from — the buggy one, or the fixed one.

    [VANTA-WHY] Per reference and not per bug, because the rule is per reference: a
    requirement the *fix* added was never in the graded tree, so holding it to the
    buggy commit's date would refuse a release the fix's own environment had; a
    requirement the buggy tree already declared must not be upgraded past it.
    Recorded as the sha, so the provenance names the commit rather than describing
    it.
    """
    commit_date: Optional[str] = None
    """``commit``'s own date — the cutoff :func:`resolve_pins` applies to this ref."""
    overridden: tuple[tuple[str, str], ...] = ()
    """The claims a BugsInPy reference pin won over, as ``(spec, source)`` pairs.

    [VANTA-WHY] Only :func:`merge_refs` sets this, and only when the group's exact
    ``==`` pin came from BugsInPy's own files and a project declaration asked for
    something the pin does not satisfy (luigi/1: ``tornado==5.0`` in BugsInPy's
    ``requirements.txt`` against ``tornado>=4.0,<5`` in ``setup.py``). The pin is
    the benchmark's validated environment and is installed anyway, so the claim it
    beat is not a conflict to raise — but it is a *decision*, and a decision that
    is not in the record is indistinguishable from a harness that never noticed.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "source": self.source,
            "pin_source": self.pin_source,
            "commit": self.commit,
            "commit_date": self.commit_date,
            "overridden": [{"spec": spec, "source": source} for spec, source in self.overridden],
        }


def _bare(spec: str) -> str:
    """``spec`` without the shell quoting it was written with.

    ``setup.sh`` is shell, so a requirement that contains a comma is normally
    quoted (``pip install "requests>=2,<3"``) and the token the shell would have
    removed the quotes from still carries them here. ``DepRef.spec`` keeps them —
    that is the line as written, and the provenance quotes it verbatim — but the
    parse must see what pip would have received.
    """
    bare = spec.strip()
    for quote in ("'", '"'):
        if len(bare) >= 2 and bare.startswith(quote) and bare.endswith(quote):
            return bare[1:-1]
    return bare


def _requirement(spec: str, source: str) -> Requirement:
    try:
        return Requirement(_bare(spec))
    except InvalidRequirement as exc:
        raise BakeError(
            f"{source}: {spec!r} is not a requirement this harness can resolve: {exc}"
        ) from exc


def dep_ref(spec: str, source: str) -> DepRef:
    return DepRef(name=canonicalize_name(_requirement(spec, source).name), spec=spec, source=source)


def _pip_args(args: str) -> tuple[list[str], Optional[str]]:
    """``(third-party specs, skip reason)`` for one ``pip install`` argument list.

    The reason is set when the whole line is dropped rather than partially read: a
    line that installs the project cannot be half-honoured, because the half that
    would be honoured is the buggy package landing in site-packages.
    """
    specs: list[str] = []
    for token in args.split():
        if token.startswith("-"):
            if token.startswith("-e") or token.startswith("--editable"):
                return [], _PROJECT_REASON
            if token.startswith(("-r", "--requirement")):
                return [], _REQUIREMENTS_FILE_REASON
            continue  # an ordinary pip flag (-U, --user, --no-deps, ...)
        if token.startswith((".", "/")) or token.endswith(".py"):
            return [], _PROJECT_REASON
        if token.startswith(
            ("git+", "hg+", "svn+", "bzr+", "http://", "https://")
        ) or "#egg=" in token:
            return [], _VCS_REASON
        specs.append(token)
    return specs, None


def parse_setup_sh(setup_sh: str) -> tuple[list[DepRef], list[SkippedLine]]:
    """``setup.sh`` → its third-party specs and every line that was skipped.

    [VANTA-WHY] The project's own install line is dropped deliberately, and the
    line is *recorded* rather than ignored. The graded tree is mounted at
    ``/workspace/...`` and every suite runs ``python -m pytest`` with its cwd set to
    the tree root, so ``import <package>`` resolves to the tree — the copy a run is
    being graded on. Baking ``setup.py install`` into a layer would instead put a
    *frozen* copy in site-packages, and a Creator patch could then pass while having
    no effect at all. Projects that need a real build
    (``pythonpath=<pkg>/build/lib/``) get it per tree, at prepare time, in Shuru.

    A line that is neither a pip install nor a ``setup.py`` run is recorded too:
    silently dropped shell is how a missing dependency becomes a mystery.
    """
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []
    seen: set[str] = set()
    for number, raw in enumerate(setup_sh.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        source = f"setup.sh line {number}"
        match = PIP_LINE_RE.match(line)
        if match:
            specs, reason = _pip_args(match.group("args"))
            if reason is not None:
                skipped.append(SkippedLine("setup.sh", number, line, reason))
                continue
            for spec in specs:
                if spec not in seen:
                    seen.add(spec)
                    refs.append(dep_ref(spec, source))
            continue
        if SETUP_PY_RE.match(line):
            skipped.append(SkippedLine("setup.sh", number, line, _SETUP_PY_REASON))
            continue
        skipped.append(SkippedLine("setup.sh", number, line, "not a pip install line"))
    return refs, skipped


def declared_deps(setup_sh: str) -> list[str]:
    """Third-party specs from ``setup.sh``, in file order, de-duplicated."""
    return [ref.spec for ref in parse_setup_sh(setup_sh)[0]]


def _requirement_line_reason(line: str) -> Optional[str]:
    """Why a requirements-file line is not a third-party spec, or ``None``."""
    if line.startswith(("-r", "--requirement")):
        return _REQUIREMENTS_FILE_REASON
    if line.startswith(("-e", "--editable")):
        return _PROJECT_REASON
    if line.startswith((".", "/")) or line.endswith(".py"):
        return _PROJECT_REASON
    if line.startswith(
        ("git+", "hg+", "svn+", "bzr+", "http://", "https://")
    ) or "#egg=" in line:
        return _VCS_REASON
    if line.startswith("-"):
        return "a pip option, not a requirement"
    return None


def requirements_refs(text: str, *, source: str) -> tuple[list[DepRef], list[SkippedLine]]:
    """One requirements file → its specs, with every non-spec line recorded."""
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []
    label = source.split(" line ", 1)[0]
    seen: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reason = _requirement_line_reason(line)
        if reason is not None:
            skipped.append(SkippedLine(label, number, line, reason))
            continue
        if line in seen:
            continue
        seen.add(line)
        refs.append(dep_ref(line, f"{source} line {number}"))
    return refs, skipped


# ---------------------------------------------------------------------------
# setup.py, read statically — never executed
# ---------------------------------------------------------------------------
def _setup_kwargs(setup_py: str) -> dict[str, ast.expr]:
    """``setup(...)``'s keyword arguments, as AST nodes.

    [VANTA-WHY] The project's ``setup.py`` is not run, on the host or anywhere else
    (design C3: nothing third-party executes outside Shuru, and "just asking
    setuptools what it wants" is executing it). It is Python; it can do anything.
    Reading it with ``ast`` is the only safe way to learn what it declares, and what
    ``ast`` cannot see — a ``**kwargs`` splat, a computed keyword name — is recorded
    as a note instead of guessed at.
    """
    try:
        tree = ast.parse(setup_py)
    except SyntaxError as exc:
        raise BakeError(f"setup.py does not parse as Python: {exc}") from exc

    assigned: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value

    kwargs: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_setup = (isinstance(func, ast.Attribute) and func.attr == "setup") or (
            isinstance(func, ast.Name) and func.id == "setup"
        )
        if not is_setup:
            continue
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            value = keyword.value
            if isinstance(value, ast.Name) and value.id in assigned:
                value = assigned[value.id]
            kwargs[keyword.arg] = value
    return kwargs


def _literal_nodes(node: ast.expr) -> list[ast.Constant]:
    """The string literals in ``node``, *with* their line numbers in ``setup.py``."""
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def _string_literals(node: ast.expr) -> list[str]:
    return [item.value for item in _literal_nodes(node)]


FILE_UNSAFE_CHARS = frozenset("><=!~;[] \t")
"""Characters a requirement has and a filename this harness reads does not.

[VANTA-WHY] Measured while writing the tests for this path:
``install_requires=['python_toolbox>=0.9.3']`` was read as *a file named
``python_toolbox>=0.9.3``*, because the literal has a dot in its last segment and
the filter below only required that. The requirement was then silently dropped and
recorded as "names a file which is not at the buggy commit" — a dependency missing
for a reason that reads like a fact about the commit. A version operator, an
extras bracket or a marker means the string is a requirement, never a filename.
"""


def _file_literals(node: ast.expr) -> list[str]:
    """String literals in ``node`` that name a file this harness could read.

    ``install_requires`` is not always a list of requirements: at some commits it is
    ``read_file('requirements.in')`` and at others
    ``open('requirements.txt').read().split('\\n')``, so the reader has to find the
    *file* the project itself names rather than assume a filename. The filter is
    deliberately narrow — the walk also sees ``'r'`` and ``'\\n'`` as literals, and
    a requirement is not a file (``FILE_UNSAFE_CHARS``).
    """
    out: list[str] = []
    for text in _string_literals(node):
        stripped = text.strip()
        if not stripped or "." not in stripped.rsplit("/", 1)[-1]:
            continue
        if any(char in stripped for char in FILE_UNSAFE_CHARS):
            continue
        if stripped.startswith(("http://", "https://", "git+")):
            continue
        if stripped not in out:
            out.append(stripped)
    return out


def _extras_tests_node(extras: Optional[ast.expr]) -> Optional[ast.expr]:
    """The value of ``extras_require={'tests': ...}``, when there is one."""
    if isinstance(extras, ast.Dict):
        for key, value in zip(extras.keys, extras.values):
            if isinstance(key, ast.Constant) and key.value == "tests":
                return value
    return None


def _entry_refs(
    node: Optional[ast.expr],
    *,
    source: str,
    read_file: Callable[[str], Optional[str]],
    notes: list[str],
    skip_files: Optional[set[str]] = None,
    commit_label: str = "the buggy commit",
) -> tuple[list[DepRef], list[SkippedLine]]:
    """One ``setup.py`` declaration → refs, following a file name when it has one."""
    if node is None:
        notes.append(f"{source}: not declared in setup.py")
        return [], []
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []
    files = _file_literals(node)
    for name in files:
        if skip_files is not None:
            if name in skip_files:
                continue
            skip_files.add(name)
        text = read_file(name)
        if text is None:
            notes.append(f"{source} names {name!r}, which is not at {commit_label}")
            continue
        part, part_skipped = requirements_refs(text, source=f"{source} ({name})")
        refs += part
        skipped += part_skipped
    if files:
        return refs, skipped
    for literal in _literal_nodes(node):
        text = literal.value.strip()
        if not text or text.startswith("#"):
            continue
        refs.append(dep_ref(text, f"{source} (setup.py) line {literal.lineno}"))
    return refs, skipped


def setup_py_refs(
    setup_py: str,
    read_file: Callable[[str], Optional[str]],
    notes: list[str],
    *,
    commit: Optional[str] = None,
    commit_date: Optional[str] = None,
    commit_label: str = "the buggy commit",
) -> tuple[list[DepRef], list[SkippedLine]]:
    """``setup.py`` → ``install_requires`` plus the project's test requirements.

    Two of the approved sources. ``setup.py`` itself is never run (design C3): it is
    Python, and asking setuptools what it wants *is* executing it. Everything here
    comes from the AST, and anything the AST cannot see is recorded as a note for a
    human rather than guessed at.

    Read once per commit (see :func:`bug_sources`), so ``commit``/``commit_date``
    stamp the refs with where they came from and ``commit_label`` says so in the
    notes — a note reading "not at the buggy commit" while the reader was looking at
    the fix would be a fact about the wrong tree.
    """
    kwargs = _setup_kwargs(setup_py)
    seen: set[str] = set()
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []
    for source, node in (
        ("install_requires", kwargs.get("install_requires")),
        ("tests_require", kwargs.get("tests_require")),
        ("extras_require['tests']", _extras_tests_node(kwargs.get("extras_require"))),
    ):
        part, part_skipped = _entry_refs(
            node,
            source=source,
            read_file=read_file,
            notes=notes,
            skip_files=seen,
            commit_label=commit_label,
        )
        if not part and node is not None and _file_literals(node):
            notes.append(f"{source}: declares no requirements")
        refs += part
        skipped += part_skipped

    # A root ``test_requirements.txt`` the project never names in ``setup.py`` is
    # still a declared test requirement, and the one source BugsInPy's own metadata
    # does not carry at all.
    if "test_requirements.txt" not in seen:
        text = read_file("test_requirements.txt")
        if text is None:
            notes.append(f"test_requirements.txt: not at {commit_label}")
        else:
            seen.add("test_requirements.txt")
            part, part_skipped = requirements_refs(text, source="test_requirements.txt")
            refs += part
            skipped += part_skipped
    return _stamped(refs, commit=commit, commit_date=commit_date), skipped


def _stamped(
    refs: Sequence[DepRef], *, commit: Optional[str], commit_date: Optional[str]
) -> list[DepRef]:
    """``refs`` with the commit they were read from attached.

    Applied at the edge of each reader rather than threaded through it: the parsing
    of a requirements file, a ``setup.sh`` line or a ``setup.py`` literal has nothing
    to do with which commit it came out of, and every one of them is also used
    directly by tests that care only about the parse.
    """
    if commit is None and commit_date is None:
        return list(refs)
    return [replace(ref, commit=commit, commit_date=commit_date) for ref in refs]


def import_name_of(setup_py: str) -> str:
    """The module the import-path check must load, named by ``setup.py`` itself.

    ``name='PySnooper'`` → ``pysnooper``: the canonical name is what PyPI knows and
    the import name is that name with dashes as underscores. When there is no
    ``name=`` the first ``packages=[...]`` literal is used instead. When neither is
    there the bug is flagged rather than guessed at — a wrong module name would make
    the check pass for the wrong reason.
    """
    kwargs = _setup_kwargs(setup_py)
    name = kwargs.get("name")
    if name is not None:
        for literal in _string_literals(name):
            if literal.strip():
                return canonicalize_name(literal.strip()).replace("-", "_")
    packages = kwargs.get("packages")
    if packages is not None:
        for literal in _string_literals(packages):
            candidate = literal.strip().split(".")[0]
            if candidate and candidate != "tests":
                return candidate
    raise BakeError(
        "setup.py declares neither name= nor a packages=[...] literal, so the "
        "import-path check cannot name the project's package; flag this bug "
        "(design §3.2)"
    )


# ---------------------------------------------------------------------------
# pyproject.toml, read statically — never executed (§3.2)
# ---------------------------------------------------------------------------
PYPROJECT_TEST_EXTRAS = ("test", "testing", "dev", "all")
"""Extra names whose requirements count as declared test requirements (§3.2).

[VANTA-WHY] ``setup.py``'s convention is a single ``tests`` extra; a
``pyproject.toml`` project's is plural and inconsistent — flit projects write
``test`` (fastapi's shape), poetry projects ``dev``, and many write ``all`` as the
union. All four are read. Which ones were taken is recorded in the ref's ``source``
and in a note, so "there are no test requirements" is never indistinguishable from
"there was no extra by that name".
"""


def _pyproject_data(pyproject: str) -> dict[str, Any]:
    """``pyproject.toml`` parsed as TOML — never executed (design C3).

    [VANTA-WHY] TOML is data, not code, so a parser is safe here in a way ``ast`` is
    for ``setup.py`` — the reason ``setup.py`` is walked rather than run does not
    apply. Unparseable TOML is still a refusal rather than an empty read: a
    half-read metadata file is how a missing dependency becomes a mystery.
    """
    try:
        return tomllib.loads(pyproject)
    except tomllib.TOMLDecodeError as exc:
        raise BakeError(f"pyproject.toml does not parse as TOML: {exc}") from exc


def _table(data: Mapping[str, Any], *path: str) -> Optional[dict[str, Any]]:
    """The TOML table at ``path``, or ``None`` when any segment is absent/not one."""
    node: Any = data
    for segment in path:
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node if isinstance(node, dict) else None


def _toml_requirement_array(
    value: Any, *, source: str, notes: list[str]
) -> list[DepRef]:
    """A PEP 621 / flit array of requirement strings → refs, exactly as written.

    A non-string entry is recorded rather than dropped: a metadata shape this
    harness does not read is a fact about the file, and silently skipping it would
    read as a project that declares less than it does.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        notes.append(
            f"{source}: expected an array of requirement strings, found "
            f"{type(value).__name__}"
        )
        return []
    refs: list[DepRef] = []
    for entry in value:
        if not isinstance(entry, str):
            notes.append(f"{source}: {entry!r} is not a requirement string; not read")
            continue
        text = entry.strip()
        if not text or text.startswith("#"):
            continue
        refs.append(dep_ref(text, source))
    return refs


def _extras_of(
    table: Mapping[str, Any], *, key: str, label: str, notes: list[str]
) -> list[tuple[str, Any]]:
    """``(extra name, its requirement array)`` from ``table[key]``, in file order."""
    extras = table.get(key)
    if extras is None:
        return []
    if not isinstance(extras, dict):
        notes.append(
            f"{label}: expected a table of extras, found {type(extras).__name__}"
        )
        return []
    return [(str(name), value) for name, value in extras.items()]


PEP440_OPERATORS = ("==", "!=", "<=", ">=", "<", ">", "~=")
"""The spellings that mean the same thing in poetry as in PEP 440."""


def _poetry_ref(
    name: str, value: Any, *, source: str, notes: list[str]
) -> Optional[DepRef]:
    """One ``[tool.poetry.dependencies]`` entry → a ref, or ``None`` (noted).

    [VANTA-WHY] Poetry's constraint syntax is not PEP 440: ``^2.22`` is a caret
    range, ``~1.4`` a tilde range, ``1.4.*`` a wildcard. Translating a caret range
    into bounds would mean reimplementing poetry's resolver in this harness, so only
    the spellings that can be read *exactly* are taken — a bare version (whose poetry
    meaning is ``==``), a PEP 440 specifier written as such, and ``*`` (no
    constraint) — and everything else is recorded and skipped. A guessed bound is
    worse than a recorded gap (§3.2), and the recorded gap names the entry.

    A bare version is turned into ``==`` *after* the operator check, never before:
    ``six1.12.0`` is a syntactically valid requirement name, so trying
    ``f"{name}{text}"`` first would silently declare a package called ``six1.12.0``
    instead of pinning ``six``.
    """
    if isinstance(value, dict):
        value = value.get("version")
    if value is None:
        notes.append(f"{source}: names no version this harness can read; not read")
        return None
    if not isinstance(value, str):
        notes.append(f"{source}: {value!r} is not a version string; not read")
        return None
    text = value.strip()
    if text in ("", "*"):
        return dep_ref(name, source)
    if text.startswith(PEP440_OPERATORS):
        return dep_ref(f"{name}{text}", source)
    try:
        Version(text)
    except InvalidVersion:
        notes.append(
            f"{source}: {value!r} is a poetry constraint this harness does not "
            "translate to PEP 440; not read"
        )
        return None
    return dep_ref(f"{name}=={text}", source)


def pyproject_refs(
    pyproject: str,
    notes: list[str],
    *,
    commit: Optional[str] = None,
    commit_date: Optional[str] = None,
    commit_label: str = "the buggy commit",
) -> list[DepRef]:
    """``pyproject.toml`` → the project's runtime deps plus its test requirements.

    Sources 3 and 4 (§3.2) for a commit packaged with ``pyproject.toml`` instead of
    ``setup.py`` — which is what ``fastapi/1`` is: a flit project with no
    ``setup.py`` at ``766157bf``, so reading ``setup.py`` alone refused the bug.
    Three spellings are read, all parsed and none executed:

    * **PEP 621** — ``[project] dependencies`` and ``[project.optional-dependencies]``;
    * **flit** — ``[tool.flit.metadata] requires`` and
      ``[tool.flit.metadata]["requires-extra"]``;
    * **poetry** — ``[tool.poetry.dependencies]``, when present (``python`` is
      skipped: it names the interpreter, not a distribution).

    A test extra (:data:`PYPROJECT_TEST_EXTRAS`) counts as a declared test
    requirement the way ``setup.py``'s ``tests`` extra does, and the extra a
    requirement came from is named in its ``source``. ``commit``/``commit_date``
    stamp the refs with where they were read from, and ``commit_label`` says so in
    the notes — the same contract as :func:`setup_py_refs`, because these refs go
    through the same both-commits union and the same per-declaration date rule.
    """
    data = _pyproject_data(pyproject)
    refs: list[DepRef] = []
    extras_taken: list[str] = []

    project = _table(data, "project")
    if project is not None:
        refs += _toml_requirement_array(
            project.get("dependencies"),
            source="project.dependencies (pyproject.toml)",
            notes=notes,
        )
        for extra, entries in _extras_of(
            project,
            key="optional-dependencies",
            label="project.optional-dependencies (pyproject.toml)",
            notes=notes,
        ):
            source = f"project.optional-dependencies['{extra}'] (pyproject.toml)"
            part = _toml_requirement_array(entries, source=source, notes=notes)
            if extra in PYPROJECT_TEST_EXTRAS:
                extras_taken.append(f"{extra} (PEP 621)")
                if not part:
                    notes.append(f"{source}: declares no requirements")
            refs += part

    flit = _table(data, "tool", "flit", "metadata")
    if flit is not None:
        refs += _toml_requirement_array(
            flit.get("requires"),
            source="tool.flit.metadata.requires (pyproject.toml)",
            notes=notes,
        )
        for extra, entries in _extras_of(
            flit,
            key="requires-extra",
            label="tool.flit.metadata.requires-extra (pyproject.toml)",
            notes=notes,
        ):
            source = f"tool.flit.metadata.requires-extra['{extra}'] (pyproject.toml)"
            part = _toml_requirement_array(entries, source=source, notes=notes)
            if extra in PYPROJECT_TEST_EXTRAS:
                extras_taken.append(f"{extra} (flit)")
                if not part:
                    notes.append(f"{source}: declares no requirements")
            refs += part

    poetry = _table(data, "tool", "poetry", "dependencies")
    if poetry is not None:
        for name in poetry:
            if name == "python":
                continue
            ref = _poetry_ref(
                str(name),
                poetry[name],
                source=f"tool.poetry.dependencies['{name}'] (pyproject.toml)",
                notes=notes,
            )
            if ref is not None:
                refs.append(ref)

    if project is None and flit is None and poetry is None:
        notes.append(
            "pyproject.toml: none of [project], [tool.flit.metadata] or "
            f"[tool.poetry.dependencies] is present at {commit_label}, so no "
            "dependency is declared here"
        )
    if extras_taken:
        notes.append("pyproject.toml: test extras taken: " + ", ".join(extras_taken))

    return _stamped(refs, commit=commit, commit_date=commit_date)


def import_name_of_pyproject(pyproject: str) -> str:
    """The module the import-path check must load, named by ``pyproject.toml``.

    The counterpart of :func:`import_name_of` for a commit with no ``setup.py``:
    ``[project].name`` first (PEP 621's own field), then flit's
    ``[tool.flit.metadata].module`` — the module flit actually builds, which is what
    an import loads — then flit's ``.name``, then poetry's ``[tool.poetry].name``.
    When none is there the bug is flagged rather than guessed at, for the same
    reason a wrong ``setup.py`` name would be: the check would pass against the
    wrong module.
    """
    data = _pyproject_data(pyproject)
    project = _table(data, "project")
    if project is not None:
        name = project.get("name")
        if isinstance(name, str) and name.strip():
            return canonicalize_name(name.strip()).replace("-", "_")
    flit = _table(data, "tool", "flit", "metadata")
    if flit is not None:
        module = flit.get("module")
        if isinstance(module, str) and module.strip():
            return module.strip().split(".")[0]
        name = flit.get("name")
        if isinstance(name, str) and name.strip():
            return canonicalize_name(name.strip()).replace("-", "_")
    poetry = _table(data, "tool", "poetry")
    if poetry is not None:
        name = poetry.get("name")
        if isinstance(name, str) and name.strip():
            return canonicalize_name(name.strip()).replace("-", "_")
    raise BakeError(
        "pyproject.toml declares no [project].name, no [tool.flit.metadata].module "
        "and no [tool.poetry].name, so the import-path check cannot name the "
        "project's package; flag this bug (design §3.2)"
    )


# ---------------------------------------------------------------------------
# the fifth source: the test side's import closure (§3.2)
# ---------------------------------------------------------------------------
PY38_STDLIB = frozenset(
    """
    __future__ __main__ _abc _ast _bisect _blake2 _bootlocale _codecs _collections
    _collections_abc _compat_pickle _contextvars _ctypes _curses _curses_panel _datetime
    _decimal _dummy_thread _elementtree _functools _hashlib _heapq _imp _io _json _locale
    _lzma _markupbase _md5 _msi _multibytecodec _opcode _operator _osx_support _pickle
    _posixsubprocess _py_abc _pydecimal _pyio _queue _random _sha1 _sha256 _sha512 _signal
    _sitebuiltins _socket _sqlite3 _ssl _stat _statistics _string _strptime _struct
    _symtable _thread _threading_local _tkinter _tracemalloc _uuid _warnings _weakref
    _weakrefset _winapi abc aifc antigravity argparse array ast asynchat asyncio asyncore
    atexit audioop base64 bdb binascii binhex bisect builtins bz2 cProfile calendar cgi
    cgitb chunk cmath cmd code codecs codeop collections colorsys compileall concurrent
    configparser contextlib contextvars copy copyreg crypt csv ctypes curses dataclasses
    datetime dbm decimal difflib dis distutils doctest dummy_threading email encodings
    ensurepip enum errno faulthandler fcntl filecmp fileinput fnmatch formatter fractions
    ftplib functools gc genericpath getopt getpass gettext glob grp gzip hashlib heapq hmac
    html http imaplib imghdr imp importlib inspect io ipaddress itertools json keyword
    lib2to3 linecache locale logging lzma mailbox mailcap marshal math mimetypes mmap
    modulefinder msilib msvcrt multiprocessing netrc nis nntplib nt ntpath nturl2path
    numbers operator optparse os ossaudiodev parser pathlib pdb pickle pickletools pipes
    pkgutil platform plistlib poplib posix posixpath pprint profile pstats pty pwd
    py_compile pyclbr pydoc pydoc_data queue quopri random re readline reprlib resource
    rlcompleter runpy sched secrets select selectors shelve shlex shutil signal site smtpd
    smtplib sndhdr socket socketserver spwd sqlite3 sre_compile sre_constants sre_parse
    ssl stat statistics string stringprep struct subprocess sunau symbol symtable sys
    sysconfig syslog tabnanny tarfile telnetlib tempfile termios test textwrap this
    threading time timeit tkinter token tokenize trace traceback tracemalloc tty turtle
    turtledemo types typing unicodedata unittest urllib uu uuid venv warnings wave weakref
    webbrowser winreg winsound wsgiref xdrlib xml xmlrpc zipapp zipfile zipimport zlib
    """.split()
)
"""Python 3.8's standard-library top-level module names, as a literal.

[VANTA-WHY] Committed rather than computed. The interpreter running the harness is
not the interpreter running the bug, and 3.8 predates ``sys.stdlib_module_names``
(added in 3.10), so asking this process what is standard library would answer for
the harness's own Python. Measured off the 3.8 module index — 210 top-level names,
which do include ``pwd`` and ``grp`` — plus CPython's private modules and the
platform and path modules the index omits: 281 names. Measured while writing this,
two of those omissions matter: ``ntpath``, which PySnooper/2's vendored
``tests/mini_toolbox/pathlib.py`` imports at module scope, and ``nt``, which it
imports under ``if os.name == 'nt'``. Neither is a distribution PyPI publishes, so
without them the closure would try to pin a package that does not exist — and fail
the bake for a module the target interpreter would never load.
"""

AMBIGUOUS_MODULE_DISTRIBUTIONS = frozenset(
    {
        "Crypto",
        "OpenSSL",
        "PIL",
        "attr",
        "bs4",
        "cv2",
        "dateutil",
        "dotenv",
        "jwt",
        "serial",
        "sklearn",
        "yaml",
    }
)
"""Modules whose import name is a different word from their distribution's.

[VANTA-WHY] ``import yaml`` is PyYAML, ``import bs4`` is beautifulsoup4, and no
canonicalisation recovers that. Guessing would fetch release dates for a package
nobody publishes and stop with a message about PyPI; refusing instead names the
actual fix, which is a reviewed mapping entry in the split's ``module_distributions``
(design §3.2) — a decision recorded in the protocol, never a guess in the harness.
A module *not* in this list is still not taken on faith: an unknown name has no
release dates, and :func:`resolve_pins` stops the bake on that.
"""


def module_distribution(module: str, *, distributions: Mapping[str, str]) -> str:
    """The distribution that provides an imported ``module``.

    The default is the module name canonicalised (``python_toolbox`` →
    ``python-toolbox``), which is what nearly every package does. The exceptions are
    the reviewed mappings in the split and the known-divergent names, which fail
    loudly rather than being guessed at.
    """
    reviewed = distributions.get(module)
    if reviewed:
        return canonicalize_name(reviewed)
    if module in AMBIGUOUS_MODULE_DISTRIBUTIONS:
        raise BakeError(
            f"the test side imports {module!r}, whose distribution is not named by its "
            "module, so the harness will not guess at it. Record the reviewed mapping "
            f"in the split's module_distributions ({module!r}: <distribution>) and bake "
            "again (design §3.2)."
        )
    return canonicalize_name(module)


@dataclass(frozen=True)
class OptionalImport:
    """A third-party import that is only reached on an optional path.

    [VANTA-WHY] Recorded, never installed. PySnooper/2's vendored helper imports
    ``scandir`` inside ``try: … except ImportError:`` for Python 2, and ``urllib``
    for the platform it is not running on. Installing those on 3.8 would provision a
    dependency the tree never loads — and a py2-only fallback does not install on 3.8
    at all, so the bake would fail over a branch that can never be taken.
    """

    module: str
    source: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"module": self.module, "source": self.source, "reason": self.reason}


_IMPORT_GUARDS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)
"""What a ``try`` must catch before its body's imports are optional. A bare
``except:`` counts too — it is how the Python 2 code writes the same thing."""

_VERSION_GUARDS = frozenset(
    {
        "PY2",
        "PY3",
        "PY27",
        "PY35",
        "PY36",
        "PY37",
        "PY38",
        "PY39",
        "python_implementation",
        "python_version",
        "version_info",
    }
)
"""Names that make an ``if`` a version check: ``sys.version_info[:2] >= (3, 6)``,
``if PY2:``, ``six.PY3``. Either branch is then the other branch's fallback."""


@dataclass(frozen=True)
class _ImportSite:
    module: str
    line: int
    optional: Optional[str]
    """Why this import is optional, or ``None`` when the module is always loaded."""


def _names_in(node: ast.AST) -> set[str]:
    """Every ``Name``/``Attribute`` a test mentions — ``sys.version_info`` → both."""
    out: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            out.add(item.id)
        elif isinstance(item, ast.Attribute):
            out.add(item.attr)
    return out


def _guard_reason(node: ast.Try) -> Optional[str]:
    """Why this ``try`` makes its body's imports optional, or ``None``."""
    for handler in node.handlers:
        if handler.type is None:
            return "behind a bare except"
        if _names_in(handler.type) & _IMPORT_GUARDS:
            return "behind an ImportError guard"
    return None


def _collect_imports(
    node: ast.AST, *, optional: Optional[str], out: list[_ImportSite]
) -> None:
    """Record every import reachable from ``node``, and whether it is optional.

    [VANTA-WHY] Optional propagates *into* a branch and never back out: an import
    inside ``try: import x  except ImportError:`` is a fallback, and so is one inside
    a function defined in that try. A relative import (``from . import helper``) is
    the tree's own file rather than a distribution, and is skipped — the absolute
    spelling of the same thing (``import tests``) is excluded by name instead, in
    :func:`test_import_closure`.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            out.append(_ImportSite(alias.name, node.lineno, optional))
        return
    if isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            out.append(_ImportSite(node.module, node.lineno, optional))
        return
    if isinstance(node, ast.Try):
        inner = _guard_reason(node) or optional
        for branch in (*node.body, *node.handlers, *node.orelse, *node.finalbody):
            _collect_imports(branch, optional=inner, out=out)
        return
    if isinstance(node, ast.If):
        guard = _names_in(node.test) & _VERSION_GUARDS
        inner = "behind a version check" if guard else optional
        for branch in (*node.body, *node.orelse):
            _collect_imports(branch, optional=inner, out=out)
        return
    for child in ast.iter_child_nodes(node):
        _collect_imports(child, optional=optional, out=out)


def test_side_paths(repo: Path, commit: str) -> list[str]:
    """The test-side Python files the tree at ``commit`` carries.

    The scope the closure walks (design §3.2). It is the *fixed* commit's list,
    because that is the test side the graded tree has (§1.2): pytest's own
    ``conftest.py``, the ``test_*``/``*_test.py`` modules, and every file under a
    ``test``/``tests`` directory — which is where a vendored helper like
    PySnooper/2's ``tests/mini_toolbox/`` lives.
    """
    return [
        path
        for path in tree_mod.list_files_at(repo, commit)
        if path.endswith(".py") and tree_mod.is_test_path(path)
    ]


def _test_local_modules(files: Sequence[str]) -> set[str]:
    """Top-level module names the test side provides for itself.

    ``tests/utils.py`` is importable as ``tests`` (the tree has ``tests/__init__.py``)
    and as ``utils`` (pytest puts the file's own directory on ``sys.path``); a file
    under ``tests/mini_toolbox/`` is importable as ``mini_toolbox``. None of those is
    a distribution, and resolving one against PyPI would either stop the bake for the
    wrong reason or install an unrelated package.
    """
    names: set[str] = set()
    for path in files:
        parts = path.split("/")
        names.update(parts[:-1])
        names.add(parts[-1].rsplit(".", 1)[0])
    return names


def _tree_local_modules(files: Sequence[str]) -> set[str]:
    """Module names the tree provides for itself by shipping Python source.

    [VANTA-WHY] The test side is not the only place the tree can put a module on
    ``sys.path``. fastapi's ``scripts/test.sh`` does ``export PYTHONPATH=./docs_src``
    before running pytest, so ``tests/test_tutorial/test_additional_responses/
    test_tutorial001.py`` opens with ``from additional_responses.tutorial001 import
    app`` — a directory of the repo's *own* tutorial sources, not a distribution
    (measured: the test-side-only exclusion set misses it, the bake then asks PyPI
    for ``additional-responses`` and dies on the 404).

    A directory that contains ``.py`` files is such a root wherever it sits, because
    both pytest's rootdir insertion and the project's own runner put directories on
    ``sys.path``. A ``.py`` file is deliberately *not* counted: ``fastapi/requests.py``
    is importable as ``fastapi.requests`` and never as a top-level ``requests``, and
    counting file stems would swallow the real distribution of that name.
    """
    names: set[str] = set()
    for path in files:
        if path.endswith(".py"):
            names.update(path.split("/")[:-1])
    return names


def tree_provided_modules(repo: Path, commit: str) -> set[str]:
    """The tree at ``commit``'s own modules, for :func:`test_import_closure`."""
    return _tree_local_modules(tree_mod.list_files_at(repo, commit))


def test_import_closure(
    *,
    files: Sequence[str],
    read_file: Callable[[str], Optional[str]],
    import_name: str,
    distributions: Optional[Mapping[str, str]] = None,
    stdlib: Iterable[str] = PY38_STDLIB,
    tree_modules: Iterable[str] = (),
) -> tuple[list[DepRef], list[OptionalImport], list[str]]:
    """The third-party modules the *test side* imports, and no source declares.

    [VANTA-WHY] Why this source exists: the four declaration sources say what a
    project *claims* it needs, and measured, that is a different set. PySnooper/2's
    fix vendors ``tests/mini_toolbox/pathlib.py``, which does ``import six`` at module
    scope; ``six`` is named by no requirements file, no ``setup.sh`` and no
    ``extras_require`` at either commit, and ``python-toolbox`` — whose sources the
    helper was copied from — declares no dependencies of its own. The tree under test
    imports it, so the layer has to have it.

    Read with ``ast``, never imported or executed: these are third-party files and
    nothing third-party runs on the host (design C3). A file that does not parse as
    Python 3 is recorded and skipped — the target interpreter could not collect it
    either, so its imports cannot be imports the suite makes.

    ``stdlib`` is a parameter so the walk is testable without a 3.8 interpreter and
    without the 280-name literal (design §3.2 pins the closure by module name, not by
    running anything). ``tree_modules`` is the rest of the tree's own modules
    (:func:`tree_provided_modules`) — a name the tree ships is not a distribution,
    wherever in the tree the sources that put it on ``sys.path`` live.
    """
    reviewed = distributions or {}
    stdlib_names = set(stdlib)
    local = _test_local_modules(files) | set(tree_modules)
    refs: list[DepRef] = []
    optional: list[OptionalImport] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    seen_optional: set[tuple[str, str]] = set()
    for path in sorted(files):
        text = read_file(path)
        if text is None:
            notes.append(f"import closure: {path} is in the tree but could not be read")
            continue
        try:
            parsed = ast.parse(text)
        except SyntaxError as exc:
            notes.append(f"import closure: {path} does not parse as Python 3 ({exc.msg})")
            continue
        sites: list[_ImportSite] = []
        _collect_imports(parsed, optional=None, out=sites)
        for site in sites:
            top = site.module.split(".")[0]
            if not top or top == "__future__":
                continue
            if top in stdlib_names or top == import_name or top in local:
                continue
            source = f"import closure: {path} line {site.line}"
            if site.optional is not None:
                if (top, source) not in seen_optional:
                    seen_optional.add((top, source))
                    optional.append(
                        OptionalImport(module=top, source=source, reason=site.optional)
                    )
                continue
            distribution = module_distribution(top, distributions=reviewed)
            if (distribution, path) in seen:
                continue
            seen.add((distribution, path))
            refs.append(DepRef(name=distribution, spec=top, source=source))
    return refs, optional, notes


def requirements_recipe(install_specs: Sequence[str]) -> str:
    """Install one bug's *resolved* dependencies on top of the interpreter layer.

    Takes pins (:func:`resolve_deps`) rather than the project's own requirement
    strings, because a layer has to be a fact: the requirements are ranges
    (``decorator>=4.3.0``), and "whatever pip resolves today" is a different
    environment every time the layer is rebuilt. Nothing here is a guess — every spec
    was named by one of the five approved sources and pinned to the newest release on
    or before the date of the commit that declared it (a fix-only requirement is held
    to the *fix's* date, everything else to the buggy commit's).
    """
    body = "\n".join(spec.strip() for spec in install_specs if spec.strip()).strip()

    lines = [
        "set -eux",
        'export HOME="${HOME:-/root}"',
        # Inherited from the interpreter layer, but asserted rather than assumed:
        # every install below runs through this `python`, and a bare standalone
        # interpreter answers PEP 668 instead of installing (measured). The
        # per-bug layer is also the first place a wrong wiring would be visible
        # as `ModuleNotFoundError` inside a wave.
        'python -c "import ssl, sys; assert sys.prefix != sys.base_prefix, sys.prefix; '
        'print(sys.version.split()[0], ssl.OPENSSL_VERSION, sys.prefix)"',
    ]
    if body:
        lines += [
            "cat > /tmp/req.txt <<'VANTA_REQUIREMENTS'",
            body,
            "VANTA_REQUIREMENTS",
            # No `|| true`: a bug whose dependencies do not install is a bug that
            # cannot be measured, and it must fail here rather than mid-wave.
            "python -m pip install --no-cache-dir -r /tmp/req.txt",
        ]
    lines += [
        "python -m pytest --version",
        "python -m pip freeze > /opt/vanta-deps-freeze.txt",
        f"touch {MARKER}",
    ]
    return "\n".join(lines)


def requirements_for(bugsinpy_root: Path, project: str, bug_id: int) -> str:
    path = bugsinpy_root / "projects" / project / "bugs" / str(bug_id) / "requirements.txt"
    if not path.is_file():
        return ""
    return read_metadata_text(path)


def setup_sh_for(bugsinpy_root: Path, project: str, bug_id: int) -> str:
    path = bugsinpy_root / "projects" / project / "bugs" / str(bug_id) / "setup.sh"
    if not path.is_file():
        return ""
    return read_metadata_text(path)


# ---------------------------------------------------------------------------
# every dependency source, unioned, then pinned by date (§3.2, §3.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepSources:
    """Everything the five approved sources declared for one bug, before pinning.

    Four of them are declarations, read at the buggy commit **and** at the fixed one:
    BugsInPy's per-bug ``requirements.txt``; the ``pip install`` lines in BugsInPy's
    ``setup.sh``; the project's own declared test requirements (``tests_require``,
    the ``tests`` extra, a root ``test_requirements.txt``); and ``install_requires``.
    The fifth is not a declaration at all — the test side's own import closure
    (:func:`test_import_closure`), read at the fixed commit because that is whose
    test side the graded tree carries. Nothing is taken from anywhere else, and no
    package is ever chosen by the harness.

    Both commits, because the graded tree is a mixture of the two (§1.2): the source
    under test is the buggy commit's, the test side is the fix's. A requirement only
    the fix declares is a requirement of the tree a run is graded on — see
    :func:`bug_sources`.
    """

    project: str
    bug_id: int
    checkpoint: str
    buggy_commit: str
    commit_date: str
    import_name: str
    fixed_commit: Optional[str] = None
    """The commit the fix's declarations and the test side's closure were read from."""
    fixed_commit_date: Optional[str] = None
    """``fixed_commit``'s date — the cutoff for a ref the fix declared."""
    refs: tuple[DepRef, ...] = ()
    optional_imports: tuple[OptionalImport, ...] = ()
    """Third-party imports on an optional path: recorded, never installed."""
    skipped: tuple[SkippedLine, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepConflict:
    """One package two sources disagree about. Never resolved automatically."""

    name: str
    claims: tuple[tuple[str, str], ...]

    def message(self) -> str:
        return (
            f"{self.name}: the declared sources disagree — "
            + "; ".join(f"{spec!r} from {source}" for spec, source in self.claims)
            + " (the harness does not pick one of them)"
        )


@dataclass(frozen=True)
class ResolvedDep:
    """One dependency, with the single version the layer will install."""

    name: str
    spec: str
    sources: tuple[str, ...]
    pin: str
    harness: bool
    released_at: Optional[str] = None
    """The pinned release's first upload date, when PyPI lists it (``YYYY-MM-DD``)."""
    pin_source: Optional[str] = None
    """The source that wrote the exact ``==`` pin, when the pin came from one."""
    exempt: bool = False
    """Whether this is a BugsInPy reference pin, so §3.2's date rule was not applied.

    A pip-freeze-style ``requirements.txt`` predates nothing: black/22's pins are from
    2020 while its buggy commit is from 2018, and the date rule would refuse every one
    of them. Rebuilding the benchmark's own environment is the point, so the pin is
    installed as written — and because that is a departure from the rule, the release
    date of every such pin is recorded, and any pin released *after* the buggy commit
    is listed at the top level of the provenance.
    """
    commit: Optional[str] = None
    """The commit whose declaration decided this pin — the source commit, recorded.

    [VANTA-WHY] The rule is per declaration, so the record has to say *which*
    declaration: "pinned to the newest release on or before the fix's date" is only
    checkable if the fix's sha is next to it, and a reader asking why a package the
    buggy tree never declared is in the layer gets an answer rather than a shrug.
    """
    commit_date: Optional[str] = None
    """``commit``'s date — the cutoff that was applied."""
    fixed: bool = False
    """Whether this declaration came from the fix's commit rather than the buggy one."""
    overridden: tuple[tuple[str, str], ...] = ()
    """The claims this pin beat, as ``(spec, source)`` pairs — normally none.

    [VANTA-WHY] Carried from :attr:`DepRef.overridden`, and recorded for the same
    reason ``exempt`` is: it is a departure from "the declarations decide", so the
    provenance has to say it out loud rather than leave a reader to notice that a
    ``>=4.0,<5`` bound is nowhere in the install set. It is empty for every ordinary
    pin, including a reference pin that agrees with the project.
    """

    @property
    def pin_rule(self) -> str:
        """How the pinned version was chosen, in one phrase, for the provenance."""
        if self.harness:
            return "provisioned by the interpreter layer"
        if self.exempt:
            return "BugsInPy's own exact pin, installed as written"
        if self.fixed:
            return "the newest release on or before the fixed commit's date"
        return "the newest release on or before the buggy commit's date"

    def as_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "spec": self.spec,
            "sources": list(self.sources),
            "pin": self.pin,
            "provisioned_by": "interpreter layer" if self.harness else "pinned at bake time",
            "pin_rule": self.pin_rule,
            "commit": self.commit,
            "commit_date": self.commit_date,
            "released_at": self.released_at,
        }
        if self.pin_source:
            data["pinned_by"] = self.pin_source
        if self.overridden:
            data["overridden"] = [
                {"spec": spec, "source": source} for spec, source in self.overridden
            ]
        return data


def _specifier_of(ref: DepRef) -> SpecifierSet:
    """The version constraint of one requirement — an empty set when there is none."""
    return _requirement(ref.spec, ref.source).specifier


def _specifier_text(ref: DepRef) -> str:
    """The version constraint of one requirement, as a string — ``""`` if none."""
    return str(_specifier_of(ref))


def _exact_version(specifier: SpecifierSet) -> Optional[Version]:
    """The single version of a ``==``-only constraint, or ``None``.

    ``==1.2.*`` is not a pin: it matches a range of releases, so it is a bound like
    any other and never the one claim a merge settles on.
    """
    clauses = list(specifier)
    if len(clauses) != 1 or clauses[0].operator != "==":
        return None
    try:
        return Version(clauses[0].version)
    except InvalidVersion:
        return None


BUGSINPY_PIN_SOURCES = ("requirements.txt", "setup.sh")
"""The source labels of the two files BugsInPy ships per bug (design §3.2).

Matched as a prefix of a ``DepRef.source`` label, which is how
:func:`requirements_refs` and :func:`parse_setup_sh` write them
(``requirements.txt line 7``). A project's own ``test_requirements.txt`` and a
``setup.py`` declaration that *names* ``requirements.txt`` both carry a different
prefix (``test_requirements.txt``, ``install_requires (requirements.txt) …``), so
they are not mistaken for BugsInPy's file.
"""


def reference_pin(ref: DepRef) -> bool:
    """Whether ``ref`` is an exact pin written in BugsInPy's own metadata.

    [VANTA-WHY] This is the one exception to §3.2's date rule. The rule exists so a
    layer cannot install a release the buggy tree could not have been built with;
    BugsInPy's ``requirements.txt``/``setup.sh`` *are* the reference environment that
    its own ``run_test.sh`` was validated against, so if the benchmark pinned an
    older-or-newer exact release, installing it is reproducing the benchmark rather
    than guessing. A range from those files is not exempt: only an exact ``==`` claim
    says which release the reference environment had, and the date rule still picks
    the release for a range. Which pins were exempt, and the date each was released,
    is recorded in the provenance (:func:`deps_provenance`).
    """
    if _exact_version(_specifier_of(ref)) is None:
        return False
    labels = (ref.pin_source or ref.source).split("; ")
    return any(label.startswith(BUGSINPY_PIN_SOURCES) for label in labels)


def _claims(refs: Sequence[DepRef]) -> tuple[tuple[str, str], ...]:
    """``(spec, source)`` for every claim in a group that has a constraint."""
    return tuple((ref.spec, ref.source) for ref in refs if _specifier_text(ref))


def merge_refs(refs: Sequence[DepRef]) -> tuple[list[DepRef], list[DepConflict]]:
    """One entry per package, plus every specifier disagreement that was found.

    Three cases, and nothing else is decided here:

    * A bare name is absorbed by a constrained one: PySnooper/3 declares ``decorator``
      in ``setup.sh`` and ``decorator>=4.3.0`` in ``install_requires``, which is one
      requirement written twice, not a conflict.
    * An exact ``==`` pin **intersects** with the bounds around it. black/22's
      ``requirements.txt`` pins ``attrs==19.3.0`` while the project's
      ``install_requires`` asks for ``attrs>=17.4.0``; the pin satisfies the bound, so
      the requirement *is* the pin. Which source wrote the pin travels with the merged
      ref in :attr:`DepRef.pin_source`, because that is what decides whether §3.2's
      date rule applies to it.
    * A pin that violates a bound is a conflict, **unless the pin is BugsInPy's own**
      (design §3.3). luigi/1 pins ``tornado==5.0`` in BugsInPy's ``requirements.txt``
      while the project's ``install_requires`` says ``tornado>=4.0,<5``; the ranges
      are jointly unsatisfiable, so one of them has to lose, and the rule is decided
      once rather than per bug: BugsInPy's pin is the environment its own
      ``run_test.sh`` was validated fail-before/pass-after in, and the dev-fix
      self-check re-verifies that claim afterwards. So the pin is installed as
      written — the same exemption :func:`reference_pin` already grants it — and the
      claim it beat travels in :attr:`DepRef.overridden` so the merge is not silent.
      A *project* pin that violates a bound is still a conflict: the project's own
      declarations disagree with each other and nothing in the harness can say which
      one the buggy tree was built with.
    * Anything else is a conflict and is never picked between (design §3.3). Two
      different ``==`` pins for one package are jointly unsatisfiable, whether or not
      one of them is BugsInPy's: a pin that never existed cannot be installed, so
      there is nothing to prefer. Two *different bounds* are a conflict too: their
      intersection is a constraint no source wrote, and a layer must install what was
      declared rather than a range the harness computed.
    """
    grouped: dict[str, list[DepRef]] = {}
    order: list[str] = []
    for ref in refs:
        if ref.name not in grouped:
            grouped[ref.name] = []
            order.append(ref.name)
        grouped[ref.name].append(ref)

    merged: list[DepRef] = []
    conflicts: list[DepConflict] = []
    for name in order:
        group = grouped[name]
        pins = [
            (ref, version)
            for ref in group
            if (version := _exact_version(_specifier_of(ref))) is not None
        ]
        if len({version for _, version in pins}) > 1:
            conflicts.append(DepConflict(name=name, claims=_claims(group)))
            continue
        if pins:
            pinned, version = pins[0]
            # Every pin in the group has the same version (checked above). When one of
            # them is BugsInPy's own, that is the one the merged ref speaks for: it is
            # what makes the pin exempt from §3.2's date rule, and it is the pin a
            # project declaration may not override.
            pinned = next((ref for ref, _ in pins if reference_pin(ref)), pinned)
            violated = tuple(
                # Collapsed to one entry per claim: the same declaration read at both
                # commits carries the same label, and a record that lists one claim
                # twice reads like a bug in the reader rather than a fact about luigi.
                dict.fromkeys(
                    (ref.spec, ref.source)
                    for ref in group
                    if ref is not pinned
                    and _specifier_text(ref)
                    and version not in _specifier_of(ref)
                )
            )
            if violated and not reference_pin(pinned):
                conflicts.append(DepConflict(name=name, claims=_claims(group)))
                continue
            merged.append(
                DepRef(
                    name=name,
                    spec=pinned.spec,
                    source="; ".join(ref.source for ref in group),
                    pin_source=pinned.source,
                    commit=pinned.commit,
                    commit_date=pinned.commit_date,
                    overridden=violated,
                )
            )
            continue
        specifiers = {_specifier_text(ref) for ref in group if _specifier_text(ref)}
        if len(specifiers) > 1:
            conflicts.append(DepConflict(name=name, claims=_claims(group)))
            continue
        chosen = next((ref for ref in group if _specifier_text(ref)), group[0])
        merged.append(
            DepRef(
                name=name,
                spec=chosen.spec,
                source="; ".join(ref.source for ref in group),
                commit=chosen.commit,
                commit_date=chosen.commit_date,
            )
        )
    return merged, conflicts


def _parseable(version: str) -> bool:
    try:
        Version(version)
    except InvalidVersion:
        return False
    return True


def _newest_release(available: Mapping[str, str]) -> tuple[str, str]:
    """``(version, date)`` of the newest parseable release PyPI lists."""
    return max(
        ((version, date) for version, date in available.items() if _parseable(version)),
        key=lambda item: Version(item[0]),
        default=("?", "?"),
    )


def resolve_pins(
    refs: Sequence[DepRef],
    *,
    releases: Mapping[str, Mapping[str, str]],
    commit_date: str,
    harness_pins: Mapping[str, str],
    buggy_commit: Optional[str] = None,
) -> list[ResolvedDep]:
    """Pin each dependency to the newest release on or before its declaration's date.

    Pure, and deliberately so: ``releases`` is passed in, so every rule here — the
    cutoff, the two exceptions, "no candidate" — is exercised by a test that never
    touches the network (design §3.2).

    The cutoff is **per declaration** and not per bug. ``commit_date`` is the buggy
    commit's date and the default; a ref carrying its own ``commit_date`` — one the
    *fix* declared, read at the fixed commit — is held to that date instead. A
    requirement the fix added was never in the buggy tree, so the buggy date would
    refuse the very release the fix's own environment had; and a ref the buggy tree
    already declared is stamped with the buggy date and so cannot be upgraded past
    it. ``buggy_commit`` is what tells the two apart, and the answer travels into
    :attr:`ResolvedDep.fixed` so the provenance can say which rule was used.

    Two exceptions, and both are checks rather than decisions:

    * A package the interpreter layer already provides. Its constraint is *checked*
      against the harness version and recorded; it is never installed over it,
      because the json-report plugin is what produces the report the oracle is read
      from and a project asking for ``pytest>=4.4.1`` would otherwise silently
      downgrade it.
    * A BugsInPy reference pin (:func:`reference_pin`). The date rule is what stops a
      layer from installing a release the buggy tree could not have been built with,
      and it is the wrong rule for the benchmark's *own* frozen environment, which is
      what ``requirements.txt`` in a pip-freeze-style bug is. Such a pin is installed
      as written, and its release date is recorded in the provenance so a late pin is
      visible rather than silent. When the pin also had to win over a project
      declaration the merge could not satisfy (:attr:`DepRef.overridden`), that
      decision is recorded with it rather than left implicit in the install list.

    For everything else the rule is unchanged, and it takes precedence over
    convenience: if the constraint has no release on or before the declaration's date,
    the bug cannot be rebuilt and the bake stops rather than installing today's
    version.
    """
    resolved: list[ResolvedDep] = []
    for ref in refs:
        requirement = _requirement(ref.spec, ref.source)
        cutoff = ref.commit_date or commit_date
        fixed = ref.commit is not None and ref.commit != buggy_commit
        if ref.name in HARNESS_PACKAGES:
            provided = harness_pins.get(ref.name)
            if provided is None:
                raise BakeError(
                    f"{ref.source}: {ref.name} is declared, but the interpreter layer "
                    f"does not provide it (harness pins: {dict(harness_pins)})"
                )
            if requirement.specifier and Version(provided) not in requirement.specifier:
                raise BakeError(
                    f"{ref.source}: {ref.spec} excludes the harness's "
                    f"{ref.name}=={provided}, which the interpreter layer installs and "
                    "pytest-json-report needs. The harness will not re-pin a package it "
                    "provides, and it will not downgrade pytest underneath the plugin — "
                    "this bug needs a decision."
                )
            resolved.append(
                ResolvedDep(
                    name=ref.name,
                    spec=ref.spec,
                    sources=(ref.source,),
                    pin=f"{ref.name}=={provided}",
                    harness=True,
                )
            )
            continue

        available = releases.get(ref.name)
        if not available:
            raise BakeError(
                f"{ref.source}: no release dates for {ref.name!r} were fetched; the "
                "bake cannot pin it (design §3.2 pins by date, not by guess)"
            )
        matching = [
            (version, date)
            for version, date in available.items()
            if _parseable(version)
            and (not requirement.specifier or Version(version) in requirement.specifier)
        ]
        exempt = reference_pin(ref)
        if exempt:
            pin_source = ref.pin_source or ref.source
            if not matching:
                newest = _newest_release(available)
                raise BakeError(
                    f"{ref.source}: {ref.spec} is a reference pin from BugsInPy's own "
                    f"metadata, and PyPI lists no such release of {ref.name!r} (newest "
                    f"{ref.name} is {newest[0]}, released {newest[1][:10]}). The "
                    "benchmark's environment cannot be reproduced from a pin whose "
                    "release does not exist."
                )
        else:
            pin_source = ref.pin_source
            matching = [item for item in matching if item[1][:10] <= cutoff]
            if not matching:
                newest = _newest_release(available)
                raise BakeError(
                    f"{ref.source}: {ref.spec} has no release on or before {cutoff} "
                    f"(newest {ref.name} is {newest[0]}, released {newest[1][:10]}). "
                    "The tree this requirement was declared in could not have been "
                    "built with it."
                )
        version, released_at = max(matching, key=lambda item: Version(item[0]))
        resolved.append(
            ResolvedDep(
                name=ref.name,
                spec=ref.spec,
                sources=(ref.source,),
                pin=f"{ref.name}=={version}",
                harness=False,
                released_at=released_at[:10],
                pin_source=pin_source,
                exempt=exempt,
                commit=ref.commit,
                commit_date=ref.commit_date,
                fixed=fixed,
                overridden=ref.overridden,
            )
        )
    return resolved


PYPI_JSON = "https://pypi.org/pypi/{name}/json"
PYPI_TIMEOUT_SECS = 60


def pypi_releases(name: str) -> dict[str, str]:
    """``{version: "YYYY-MM-DD"}`` for one package, from PyPI's JSON API.

    The first upload date of each version, which is when the tree at the buggy
    commit could first have installed it. Called only by the bake; everything that
    *decides* anything takes this mapping as an argument instead.
    """
    url = PYPI_JSON.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT_SECS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BakeError(f"cannot read release dates for {name!r} from {url}: {exc}") from exc

    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise BakeError(f"{url} returned no 'releases' mapping")
    dates: dict[str, str] = {}
    for version, files in releases.items():
        stamps = [
            str(item["upload_time_iso_8601"])
            for item in (files or [])
            if item.get("upload_time_iso_8601")
        ]
        if stamps:
            dates[str(version)] = min(stamps)[:10]
    if not dates:
        raise BakeError(f"{url} lists no released files for {name!r}")
    return dates


def resolve_deps(
    sources: DepSources,
    *,
    releases: Mapping[str, Mapping[str, str]],
    harness_pins: Mapping[str, str],
) -> list[ResolvedDep]:
    """Union → conflict check → date pin, for one bug."""
    merged, conflicts = merge_refs(sources.refs)
    if conflicts:
        raise BakeError(
            f"{sources.project}/{sources.bug_id}: "
            + " | ".join(conflict.message() for conflict in conflicts)
            + " — name the source that should win and the split will be amended rather "
            "than guessed at"
        )
    return resolve_pins(
        merged,
        releases=releases,
        commit_date=sources.commit_date,
        harness_pins=harness_pins,
        buggy_commit=sources.buggy_commit,
    )


def bug_sources(
    entry: Mapping[str, Any],
    *,
    bugsinpy_root: Path,
    repo: Path,
    module_distributions: Optional[Mapping[str, str]] = None,
) -> DepSources:
    """Read every approved source for one bug, at the buggy **and** the fixed commit.

    [VANTA-WHY] Both commits, because the graded tree is a mixture of the two (§1.2).
    The *source* under test is the buggy commit's; the *test side* is the fix's, which
    is what makes the bug a bug. Reading the declarations at the buggy commit alone
    therefore describes an environment the suite never runs in: PySnooper/2's fix
    vendors ``tests/mini_toolbox/pathlib.py``, whose ``import six`` is at module scope
    and named by no requirements file, no ``setup.sh`` and no ``extras_require`` at
    either commit. The tree under test imports it, so the layer has to have it.

    Read the other way, though, the fix is not authoritative either: a requirement the
    fix *adds* must not be back-dated, and one it *drops* was still there for the
    buggy code. So the two commits are unioned, each ref carries the commit it was
    read from (:attr:`DepRef.commit`), and :func:`resolve_pins` holds each to its own
    commit's date. A package both commits declare goes through the merge in
    :func:`merge_refs` — one requirement written twice, or a real conflict that stops
    the bake.

    BugsInPy's own ``requirements.txt``/``setup.sh`` are **not** read per commit: they
    are the benchmark's description of the bug, one file pair per bug, and they are the
    reference environment the bug's ``run_test.sh`` was validated against. They are
    left unstamped so they keep the buggy commit's date as their cutoff.

    The fifth source is the test side's own import closure (:func:`test_import_closure`),
    read at the fixed commit because that is whose test side the graded tree carries.
    A module whose distribution is not named by the module fails the bake rather than
    being guessed at; the fix is a reviewed ``module_distributions`` entry in the split.
    """
    project = str(entry["project"])
    bug_id = int(entry["bug_id"])
    buggy = str(entry["buggy_commit"])
    fixed = str(entry["fixed_commit"])
    buggy_date = tree_mod.commit_date(repo, buggy)
    fixed_date = tree_mod.commit_date(repo, fixed)
    notes: list[str] = []
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []

    # 1. BugsInPy's per-bug requirements.txt. Measured: 0 bytes for PySnooper 1-3,
    # which is exactly why the other sources exist.
    requirements_text = requirements_for(bugsinpy_root, project, bug_id)
    part, part_skipped = requirements_refs(requirements_text, source="requirements.txt")
    refs += part
    skipped += part_skipped
    if not requirements_text.strip():
        notes.append("requirements.txt (BugsInPy): empty for this bug")

    # 2. setup.sh, minus its install of the project itself.
    setup_sh = setup_sh_for(bugsinpy_root, project, bug_id)
    part, part_skipped = parse_setup_sh(setup_sh)
    refs += part
    skipped += part_skipped
    if not setup_sh.strip():
        notes.append("setup.sh (BugsInPy): empty for this bug")

    # The import name comes from the *buggy* commit's own metadata — ``setup.py``
    # or ``pyproject.toml``, whichever it carries: that is the package under test,
    # and the closure needs it to exclude the project's own modules. A commit with
    # neither is refused rather than guessed at (§3.2): the deps could be anywhere
    # and the module name would be a guess.
    read_buggy = partial(tree_mod.read_file_at, repo, buggy)
    buggy_setup = read_buggy("setup.py")
    buggy_pyproject = read_buggy("pyproject.toml")
    if buggy_setup is None and buggy_pyproject is None:
        raise BakeError(
            f"{project}/{bug_id}: neither setup.py nor pyproject.toml at {buggy}, so "
            "the declared dependencies and test requirements cannot be read. Flag "
            "this bug rather than guessing at its dependencies (design §3.2)."
        )
    import_name = (
        import_name_of(buggy_setup)
        if buggy_setup is not None
        else import_name_of_pyproject(buggy_pyproject)
    )

    # 3 + 4. The project's own declarations, once per commit — from ``setup.py``,
    # from ``pyproject.toml``, or from both if a commit carries both (a project
    # mid-migration does, and reading only one would under-provision).
    for label, sha, date in (("buggy", buggy, buggy_date), ("fixed", fixed, fixed_date)):
        read_at = partial(tree_mod.read_file_at, repo, sha)
        setup_py = read_at("setup.py")
        pyproject = read_at("pyproject.toml")
        if setup_py is None and pyproject is None:
            notes.append(
                f"setup.py and pyproject.toml: neither at the {label} commit "
                f"({sha[:12]})"
            )
            continue
        if setup_py is not None:
            part, part_skipped = setup_py_refs(
                setup_py,
                read_at,
                notes,
                commit=sha,
                commit_date=date,
                commit_label=f"the {label} commit",
            )
            refs += part
            skipped += part_skipped
        if pyproject is not None:
            refs += pyproject_refs(
                pyproject,
                notes,
                commit=sha,
                commit_date=date,
                commit_label=f"the {label} commit",
            )

    # 5. The test side's own imports, at the fixed commit. The tree's own modules
    # are excluded too: the test side is not the only thing that can put a name on
    # ``sys.path`` (fastapi's own ``scripts/test.sh`` adds ``docs_src``), and a name
    # the tree ships is a module it provides, never a distribution to install.
    closure_refs, optional, closure_notes = test_import_closure(
        files=test_side_paths(repo, fixed),
        read_file=partial(tree_mod.read_file_at, repo, fixed),
        import_name=import_name,
        distributions=module_distributions,
        tree_modules=tree_provided_modules(repo, fixed),
    )
    refs += _stamped(closure_refs, commit=fixed, commit_date=fixed_date)
    notes += closure_notes

    return DepSources(
        project=project,
        bug_id=bug_id,
        checkpoint=bug_checkpoint_name(entry),
        buggy_commit=buggy,
        commit_date=buggy_date,
        fixed_commit=fixed,
        fixed_commit_date=fixed_date,
        import_name=import_name,
        refs=tuple(refs),
        optional_imports=tuple(optional),
        skipped=tuple(skipped),
        notes=tuple(notes),
    )


def installs_of(resolved: Sequence[ResolvedDep]) -> tuple[str, ...]:
    """The pins the layer's recipe installs.

    One definition, used by both the recipe (via :class:`BugDeps`) and the
    provenance, so "what was recorded" and "what pip was told" cannot drift apart.
    """
    return tuple(dep.pin for dep in resolved if not dep.harness)


def deps_provenance(
    sources: DepSources,
    resolved: Sequence[ResolvedDep],
    *,
    harness_pins: Mapping[str, str],
) -> dict[str, Any]:
    """What the bake decided for one bug, as data a later stage can read.

    ``installs`` is **exactly** what the layer's recipe passes to pip: a package the
    interpreter layer already provides is not in it. It is not a loss of
    information — every dependency, provisioned or not, is in ``dependencies`` with
    ``provisioned_by`` saying which, and the harness pins are in their own key — but
    it does mean ``installs`` can be compared against the built recipe without a
    filter in between. It could not before: ``pytest==8.0.2`` appeared "installed"
    by the bug layer while the recipe never asked for it, which is the kind of
    contradiction that makes a provenance file not worth reading.

    ``pins_after_commit_date`` is the record of the places the layer's environment is
    newer than the *buggy* tree. Two of them are expected, and neither is an error:

    * a BugsInPy reference pin, installed as written — a pip-freeze-style
      ``requirements.txt`` is from a different era than the commit it is used with;
    * a requirement only the *fix* declared, held to the fix's own date — such a pin
      can legitimately postdate the buggy commit, since the fix's environment is the
      one that had it.

    Both are still the first thing to look at when a wave misbehaves against what the
    bug's date suggests, so this is a queryable list and not a sentence in a
    docstring. ``pin_rule`` distinguishes the two, and every dependency carries
    ``released_at``, ``pin_rule`` and the ``commit`` its declaration came from, so the
    same question can be asked of a pin that is *inside* the window.

    ``optional_imports`` is the other half of the closure: third-party modules the test
    side imports only on a Python-2 or platform-specific branch. They are recorded and
    deliberately **not** installed — installing a py2-only fallback on 3.8 would fail
    the bake over a branch that can never be taken.
    """
    return {
        "schema_version": 1,
        "project": sources.project,
        "bug_id": sources.bug_id,
        "checkpoint": sources.checkpoint,
        "buggy_commit": sources.buggy_commit,
        "commit_date": sources.commit_date,
        "fixed_commit": sources.fixed_commit,
        "fixed_commit_date": sources.fixed_commit_date,
        "import_name": sources.import_name,
        "harness_pins": dict(harness_pins),
        "installs": list(installs_of(resolved)),
        "dependencies": [dep.as_dict() for dep in resolved],
        "pins_after_commit_date": [
            {
                "name": dep.name,
                "pin": dep.pin,
                "released_at": dep.released_at,
                "pin_rule": dep.pin_rule,
            }
            for dep in resolved
            if dep.released_at is not None and dep.released_at > sources.commit_date
        ],
        "optional_imports": [item.as_dict() for item in sources.optional_imports],
        "skipped": [line.as_dict() for line in sources.skipped],
        "notes": list(sources.notes),
    }


@dataclass(frozen=True)
class BugDeps:
    """One bug's layer: what gets installed, and the record of why.

    ``installs`` is what the layer's recipe may put on the command line, so a
    package the interpreter layer already provides is **not** in it: re-pinning
    pytest would either be a no-op that re-downloads it or a silent downgrade
    underneath ``pytest-json-report``, which is the plugin the oracle is read from.
    The full set — including what was left to the harness — is in ``provenance``.
    """

    entry: dict[str, Any]
    checkpoint: str
    installs: tuple[str, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FlaggedBug:
    """One bug whose dependencies could not be read, and why (§3.2).

    [VANTA-WHY] A bug whose metadata cannot be read is one bug's problem, never the
    batch's. ``bake`` used to stop at the first one — ``fastapi/1``'s ``no setup.py``
    refusal took a whole dev-split bake down with it — which hides every later bug
    behind it and produces nothing to look at. The bug is recorded here with the
    reason it was refused, the rest of the split is resolved and baked, and the
    flagged list is printed separately from the §14.7 gate results: a flagged bug is
    not a gate failure, so it never silently promotes a reserve, and replacing it is
    a decision taken by hand (amend the split, §8).
    """

    key: str
    reason: str


def resolve_bug_deps(
    split: Split,
    entries: Sequence[dict[str, Any]],
    *,
    bugsinpy_root: Path,
    repo_for: Callable[[str], Path],
    fetch: Callable[[str], Mapping[str, str]] = pypi_releases,
) -> tuple[list[BugDeps], list[FlaggedBug]]:
    """Resolve every selected bug's layer, pinning by date from PyPI.

    Returns ``(layers, flagged)``. A bug whose sources cannot be read — no
    ``setup.py`` and no ``pyproject.toml`` at the buggy commit, an unparseable file,
    two sources that disagree, a release the interpreter layer cannot accept, a
    requirement with no release on or before its declaration's date — is **flagged**
    with the reason and skipped, rather than stopping the bake: one unreadable bug
    must not take the whole split down with it (§3.2). ``flagged`` is returned rather
    than printed so the caller can exclude those checkpoints from the recipes and
    print the list next to the §14.7 gate results, where it belongs.

    ``fetch`` is injected so the entire chain — union, conflict, cutoff, harness
    exception — is testable without the network.
    """
    out: list[BugDeps] = []
    flagged: list[FlaggedBug] = []
    distributions = split.module_distributions()
    for entry in entries:
        key = f"{entry['project']}/{entry['bug_id']}"
        try:
            sources = bug_sources(
                entry,
                bugsinpy_root=bugsinpy_root,
                repo=repo_for(str(entry["project"])),
                module_distributions=distributions,
            )
            pin = pins_for_python(split, minor_version(str(entry["python_version"])))
            harness_pins = {"pytest": pin.pytest, "pytest-json-report": pin.json_report}
            needed = sorted(
                {ref.name for ref in sources.refs if ref.name not in HARNESS_PACKAGES}
            )
            releases = {name: fetch(name) for name in needed}
            resolved = resolve_deps(sources, releases=releases, harness_pins=harness_pins)
        except BakeError as exc:
            flagged.append(FlaggedBug(key=key, reason=f"{type(exc).__name__}: {exc}"))
            continue
        out.append(
            BugDeps(
                entry=dict(entry),
                checkpoint=sources.checkpoint,
                installs=installs_of(resolved),
                provenance=deps_provenance(sources, resolved, harness_pins=harness_pins),
            )
        )
    return out, flagged


def deps_path(deps_dir: Path, checkpoint: str) -> Path:
    return deps_dir / f"{checkpoint}.json"


def write_deps(deps_dir: Path, deps: BugDeps) -> Path:
    deps_dir.mkdir(parents=True, exist_ok=True)
    path = deps_path(deps_dir, deps.checkpoint)
    path.write_text(
        json.dumps(deps.provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def load_deps(deps_dir: Path, checkpoint: str) -> dict[str, Any]:
    """The provenance ``bake`` recorded for one bug layer.

    [VANTA-WHY] Read from disk rather than re-derived: re-deriving needs PyPI
    release dates and the network, and the layer was baked from *these* pins. A bug
    with no provenance was baked by a harness that did not record what it installs,
    and the only honest answer is to rebuild it.
    """
    path = deps_path(deps_dir, checkpoint)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BakeError(
            f"no dependency provenance for {checkpoint} at {path}: re-run bake for "
            "this bug (design §3.2)"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BakeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "installs" not in data:
        raise BakeError(f"{path} is not dependency provenance")
    return data


def missing_deps(deps_dir: Path, checkpoints: Sequence[str]) -> list[str]:
    """The checkpoints under ``deps_dir`` with no provenance record, sorted.

    [VANTA-WHY] Returns the missing names instead of raising: a layer baked without
    a record is *one bug's* problem, not the batch's (§3.2), and ``prepare`` has to
    be able to flag that bug and prepare the rest. A bug whose dependencies could
    not be read at bake time has no layer and no record, so this is also where
    ``prepare`` sees bake's refusal — per bug, never as a stopped run.
    """
    return sorted(
        name for name in checkpoints if not deps_path(deps_dir, name).is_file()
    )


# ---------------------------------------------------------------------------
# baking
# ---------------------------------------------------------------------------
def bake(
    name: str,
    script: str,
    *,
    base: str = DEFAULT_BASE,
    disk_mb: int = BAKE_DISK_MB,
) -> ExecResult:
    """Bake and verify one layer. Network is allowed here and only here."""
    if disk_mb < MIN_DISK_MB:
        raise SplitError(
            f"disk_mb={disk_mb} is below the base image's {MIN_DISK_MB} MB; "
            "shuru refuses to boot a disk smaller than its base"
        )
    return checkpoint_create(
        name,
        script,
        base=base,
        disk_mb=disk_mb,
        allow_net=True,
        timeout_secs=BAKE_TIMEOUT_SECS,
        marker=MARKER,
    )


def entry_by_key(split: Split, key: str) -> dict[str, Any]:
    project, _, raw_id = key.partition("/")
    if not raw_id.isdigit():
        raise SplitError(f"--only expects PROJECT/BUG, got {key!r}")
    return split.bug(project, int(raw_id))


def select_layers(
    split: Split, *, pythons: list[str], only: list[str], roles: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """``(interpreter layers, bug entries)`` in bake order, de-duplicated."""
    known = pythons_in_split(split)
    unknown = [minor for minor in pythons if minor not in known]
    if unknown:
        raise SplitError(
            f"--pythons names {unknown}, which no bug in the split uses; known: {known}"
        )
    if only:
        entries = [entry_by_key(split, key) for key in only]
    else:
        entries = [
            entry for entry in split.bugs if entry.get("role", "pilot") in roles
        ]
    entries = [
        entry for entry in entries if minor_version(entry["python_version"]) in pythons
    ]
    minors = sorted(set(pythons) | {minor_version(e["python_version"]) for e in entries})
    return minors, entries


def layer_recipes(
    split: Split,
    *,
    pythons: list[str],
    entries: Sequence[dict[str, Any]],
    deps: Sequence[BugDeps],
) -> list[tuple[str, str, str]]:
    """``[(checkpoint name, base, script)]`` — every layer the bake would create.

    Pure string building (nothing is executed, nothing is downloaded), which is
    what makes the recipes testable without a VM: the assertions they contain are
    the wave-1 gate, and a gate that is only exercised by a real bake is a gate
    nobody re-checks.

    The per-bug layer's installs arrive as resolved pins rather than being read
    again from ``setup.sh``/``requirements.txt`` here: :func:`resolve_bug_deps` is
    the single place that talks to PyPI, and it is also the place that writes the
    provenance ``prepare`` reads back. Two readers of the same sources would be two
    chances to install something other than what was recorded.
    """
    by_checkpoint = {item.checkpoint: item for item in deps}
    layers: list[tuple[str, str, str]] = []
    for minor in pythons:
        layers.append(
            (
                python_checkpoint_name(minor),
                DEFAULT_BASE,
                python_recipe(minor, pins_for_python(split, minor)),
            )
        )
    for entry in entries:
        name = bug_checkpoint_name(entry)
        item = by_checkpoint.get(name)
        if item is None:
            raise BakeError(
                f"no resolved dependencies for {name}: resolve_bug_deps has to run "
                "before the recipes are built (design §3.2)"
            )
        layers.append(
            (
                name,
                python_checkpoint_name(minor_version(entry["python_version"])),
                requirements_recipe(item.installs),
            )
        )
    return layers


def render_plan(split: Split, *, pythons: list[str], entries: Sequence[dict[str, Any]]) -> str:
    """What the bake would create, and from which pins. No side effects."""
    lines = ["| layer | base | python | pytest | json-report |", "|---|---|---|---|---|"]
    for minor in pythons:
        pin = pins_for_python(split, minor)
        lines.append(
            f"| {python_checkpoint_name(minor)} | {DEFAULT_BASE} | {minor} | "
            f"{pin.pytest} | {pin.json_report} |"
        )
    for entry in entries:
        lines.append(
            f"| {bug_checkpoint_name(entry)} | "
            f"{python_checkpoint_name(minor_version(entry['python_version']))} | "
            f"{entry['python_version']} | {entry['pytest_version']} | "
            f"{entry['json_report_version']} |"
        )
    return "\n".join(lines) + "\n"


def children_of(split: Split) -> dict[str, str]:
    """``{per-bug layer: its interpreter layer}`` for **every** bug in the split.

    Not just the selected ones: :func:`plan_layers` needs the whole tree to find the
    stale children of a rebuilt base, and those are exactly the layers nobody
    selected to rebuild.
    """
    return {
        bug_checkpoint_name(entry): python_checkpoint_name(
            minor_version(str(entry["python_version"]))
        )
        for entry in split.bugs
    }


def layer_exists(name: str) -> bool:
    """The **single** existence check for a baked layer.

    [VANTA-WHY] ``bake``'s skip/rebuild decision and ``prepare``'s preflight both
    go through this one function, so "bake says the layer exists, prepare says it
    does not" can only ever mean the layer really changed in between. It did once,
    and the reason was not the check: ``checkpoint_create`` treats a failed build
    as "delete the name and report the error", so a rebuild that failed with
    *``checkpoint 'x' already exists``* deleted the perfectly good layer it had
    collided with. Two readers of one function cannot disagree about a name; they
    can only disagree about a name that moved. :func:`plan_layers` is what stops it
    from moving under them.
    """
    return checkpoint_exists(name)


def frozen(name: str) -> bool:
    """Kept as the bake's own word for the same check, so no caller has a reason
    to reach for ``shuru`` directly and invent a second opinion."""
    return layer_exists(name)


@dataclass(frozen=True)
class LayerAction:
    """What the bake will do to one layer, decided before anything is touched."""

    name: str
    base: str
    script: str
    delete: bool
    build: bool
    reason: str

    def describe(self) -> str:
        verb = []
        if self.delete:
            verb.append("delete")
        if self.build:
            verb.append("rebuild" if self.delete else "build")
        else:
            verb.append("skip")
        return f"{self.name}: {'+'.join(verb)} — {self.reason}"


def plan_layers(
    layers: Sequence[tuple[str, str, str]],
    *,
    force: bool,
    children: Mapping[str, str],
    exists: Callable[[str], bool] = layer_exists,
) -> list[LayerAction]:
    """``layers`` → what to delete, build and skip, in delete-then-build order.

    [VANTA-WHY] ``--force`` used to mean nothing to shuru: it was a *skip*
    suppression, not a replacement, so ``checkpoint create`` was called on a name
    that already existed, shuru refused, and ``checkpoint_create``'s failure path
    deleted the existing layer — turning "rebuild this" into "destroy this". A
    forced rebuild therefore has to delete first, and that delete has to be planned
    before any of it runs.

    [VANTA-WHY] A rebuilt interpreter layer invalidates everything stacked on it:
    ``vanta-bi-<project>-<N>`` inherits its interpreter, pytest and json-report from
    ``vanta-bugsinpy-py3.<v>``, so a child left behind is a layer ``prepare`` would
    happily run suites in — the exact failure mode of a stale layer that looks
    valid. Children of a rebuilt base are therefore deleted too. ``children`` is
    every per-bug layer in the split (not just the selected ones), because the stale
    child that matters most is the one nobody selected.

    A layer outside the selection is never deleted except as such a stale child,
    and every action carries a ``reason`` so the log says which case it was.

    Pure: ``exists`` is injected, which is what makes the ordering and the
    non-deletion rules testable without a microVM.
    """
    selected = {name for name, _, _ in layers}
    present = {name for name, _, _ in layers if exists(name)}
    rebuilt = {name for name, _, _ in layers if force or name not in present}

    actions: list[LayerAction] = []
    for name, base, script in layers:
        if force and name in present:
            reason = "--force: replaced (an existing layer is deleted first)"
        elif force:
            reason = "--force: built (nothing with this name yet)"
        elif name in present:
            reason = "already baked"
        else:
            reason = "not baked yet"
        actions.append(
            LayerAction(
                name=name,
                base=base,
                script=script,
                delete=force and name in present,
                build=force or name not in present,
                reason=reason,
            )
        )

    for name, base in sorted(children.items()):
        if name in selected or base not in rebuilt or not exists(name):
            continue
        actions.append(
            LayerAction(
                name=name,
                base=base,
                script="",
                delete=True,
                build=False,
                reason=(
                    f"stale layer on {base}, which is being rebuilt in this run; "
                    "deleted so prepare cannot run a suite in an environment baked "
                    "from the old base (it must be rebaked before it is used)"
                ),
            )
        )
    return actions


def delete_order(
    actions: Sequence[LayerAction], children: Mapping[str, str]
) -> list[LayerAction]:
    """The deletions, children before their base.

    The order is defensive rather than observed: shuru does not currently refuse to
    delete a layer that other layers derive from, but deleting a base first and then
    failing on a child would leave the tree in a state no plan describes.
    """
    return sorted(
        (action for action in actions if action.delete),
        key=lambda action: 0 if action.name in children else 1,
    )


def build_order(actions: Sequence[LayerAction]) -> list[LayerAction]:
    """The builds, in the order ``layer_recipes`` produced them: base before child.

    A child is baked ``--from`` its base, so this order is a correctness
    requirement, not a preference.
    """
    return [action for action in actions if action.build]


def summarise(actions: Sequence[LayerAction], *, failed: Sequence[str] = ()) -> str:
    """The bake's one-line-per-layer record, for the log and for the failure path."""
    lines = ["bake summary:"]
    for action in actions:
        state = "FAILED" if action.name in failed else "done" if action.build else "skipped"
        lines.append(f"  [{state}] {action.describe()}")
    if failed:
        lines.append(
            f"  {len(failed)} layer(s) failed: {', '.join(failed)} — nothing was "
            "left behind for prepare to treat as valid"
        )
    return "\n".join(lines)


def _bake_layer(name: str, script: str, base: str) -> ExecResult:
    """The default ``build`` callable.

    ``cmd_bake`` injects its own, so ``--disk-mb`` reaches the recipe; this is the
    same call with the default disk size, and it exists so the ordering logic has a
    real default rather than being untestable without one.
    """
    return bake(name, script, base=base)


def run_bake_actions(
    actions: Sequence[LayerAction],
    *,
    children: Mapping[str, str],
    exists: Callable[[str], bool] = layer_exists,
    remove: Callable[[str], None] = checkpoint_delete,
    build: Callable[[str, str, str], ExecResult] = _bake_layer,
    log: Callable[[str], None] = print,
) -> str:
    """Delete, then build, and stop at the first failure with a full summary.

    Takes the plan :func:`plan_layers` produced rather than re-deriving it: what is
    logged before the run has to be the same list that is executed, or the log is
    describing a different bake.

    [VANTA-WHY] Fail fast, and fail *clean*: a layer that failed to build must not
    survive under a name ``prepare`` will accept. ``checkpoint_create`` deletes on
    a failed inner command, but a failed ``delete`` of its own would leave the name
    in place, so the absence is verified here rather than assumed — this is the one
    post-condition where "probably" is not good enough, because a surviving
    half-baked layer is invisible to every later stage.

    Injected ``exists``/``remove``/``build``/``log`` so the whole sequence is a
    structural test rather than something only a real bake can exercise.
    """
    for action in delete_order(actions, children):
        log(f"[{action.name}] deleting — {action.reason}")
        remove(action.name)

    failed: list[str] = []
    for action in build_order(actions):
        log(f"[{action.name}] building from {action.base} — {action.reason}")
        result = build(action.name, action.script, action.base)
        if not result.ok:
            failed.append(action.name)
            if exists(action.name):
                remove(action.name)
            if exists(action.name):
                raise BakeError(
                    f"baking {action.name} failed and the layer could not be removed, "
                    "so it would be treated as valid by prepare; delete it by hand "
                    f"before running anything.\n\n{summarise(actions, failed=failed)}\n"
                    f"--- build log (tail) ---\n"
                    f"{(result.stdout + result.stderr).strip()[-2000:]}"
                )
            raise BakeError(
                f"baking {action.name} failed (exit {result.exit_code}): "
                f"{(result.stdout + result.stderr).strip()[-2000:]}\n\n"
                f"{summarise(actions, failed=failed)}"
            )
        log(f"[{action.name}] ok — marker {MARKER} verified")

    return summarise(actions)
