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
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

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

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "spec": self.spec, "source": self.source}


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
            notes.append(f"{source} names {name!r}, which is not at the buggy commit")
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
) -> tuple[list[DepRef], list[SkippedLine]]:
    """``setup.py`` → ``install_requires`` plus the project's test requirements.

    The last two of the four approved sources. ``setup.py`` itself is never run
    (design C3): it is Python, and asking setuptools what it wants *is* executing
    it. Everything here comes from the AST, and anything the AST cannot see is
    recorded as a note for a human rather than guessed at.
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
            node, source=source, read_file=read_file, notes=notes, skip_files=seen
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
            notes.append("test_requirements.txt: not at the buggy commit")
        else:
            seen.add("test_requirements.txt")
            part, part_skipped = requirements_refs(text, source="test_requirements.txt")
            refs += part
            skipped += part_skipped
    return refs, skipped


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


def requirements_recipe(install_specs: Sequence[str]) -> str:
    """Install one bug's *resolved* dependencies on top of the interpreter layer.

    Takes pins (:func:`resolve_deps`) rather than the project's own requirement
    strings, because a layer has to be a fact: the requirements are ranges
    (``decorator>=4.3.0``), and "whatever pip resolves today" is a different
    environment every time the layer is rebuilt. Nothing here is a guess — every
    spec was named by one of the four approved sources and pinned to a release that
    existed on the buggy commit's date.
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
    return path.read_text(encoding="utf-8", errors="replace")


def setup_sh_for(bugsinpy_root: Path, project: str, bug_id: int) -> str:
    path = bugsinpy_root / "projects" / project / "bugs" / str(bug_id) / "setup.sh"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# the four dependency sources, unioned, then pinned by date (§3.2, §3.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepSources:
    """Everything the four approved sources declared for one bug, before pinning.

    The four, in the order the design fixes them: BugsInPy's per-bug
    ``requirements.txt``; the ``pip install`` lines in BugsInPy's ``setup.sh``; the
    project's own declared test requirements at the buggy commit (``tests_require``,
    the ``tests`` extra, a root ``test_requirements.txt``); and ``install_requires``.
    Nothing is taken from anywhere else, and no package is ever chosen by the
    harness.
    """

    project: str
    bug_id: int
    checkpoint: str
    buggy_commit: str
    commit_date: str
    import_name: str
    refs: tuple[DepRef, ...] = ()
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "sources": list(self.sources),
            "pin": self.pin,
            "provisioned_by": "interpreter layer" if self.harness else "pinned at bake time",
        }


def _specifier_text(ref: DepRef) -> str:
    """The version constraint of one requirement, as a string — ``""`` if none."""
    return str(_requirement(ref.spec, ref.source).specifier)


def merge_refs(refs: Sequence[DepRef]) -> tuple[list[DepRef], list[DepConflict]]:
    """One entry per package, plus every specifier disagreement that was found.

    A bare name is absorbed by a constrained one: PySnooper/3 declares ``decorator``
    in ``setup.sh`` and ``decorator>=4.3.0`` in ``install_requires``, which is one
    requirement written twice, not a conflict. Two *different* specifier sets are
    returned as a conflict and never picked between (design §3.3: a bug with
    contradictory constraints is not reproducible from its declared sources).
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
        specifiers = {_specifier_text(ref) for ref in group if _specifier_text(ref)}
        if len(specifiers) > 1:
            conflicts.append(
                DepConflict(
                    name=name,
                    claims=tuple(
                        (ref.spec, ref.source) for ref in group if _specifier_text(ref)
                    ),
                )
            )
            continue
        chosen = next((ref for ref in group if _specifier_text(ref)), group[0])
        merged.append(
            DepRef(
                name=name,
                spec=chosen.spec,
                source="; ".join(ref.source for ref in group),
            )
        )
    return merged, conflicts


def _parseable(version: str) -> bool:
    try:
        Version(version)
    except InvalidVersion:
        return False
    return True


def resolve_pins(
    refs: Sequence[DepRef],
    *,
    releases: Mapping[str, Mapping[str, str]],
    commit_date: str,
    harness_pins: Mapping[str, str],
) -> list[ResolvedDep]:
    """Pin each dependency to the newest release on or before ``commit_date``.

    Pure, and deliberately so: ``releases`` is passed in, so every rule here — the
    cutoff, the harness exception, "no candidate" — is exercised by a test that
    never touches the network (design §3.2).

    The exception is a package the interpreter layer already provides. Its
    constraint is *checked* against the harness version and recorded; it is never
    installed over it, because the json-report plugin is what produces the report
    the oracle is read from and a project asking for ``pytest>=4.4.1`` would
    otherwise silently downgrade it.
    """
    resolved: list[ResolvedDep] = []
    for ref in refs:
        requirement = _requirement(ref.spec, ref.source)
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
        candidates = [
            (version, date)
            for version, date in available.items()
            if date[:10] <= commit_date
            and _parseable(version)
            and (not requirement.specifier or Version(version) in requirement.specifier)
        ]
        if not candidates:
            newest = max(
                (item for item in available.items() if _parseable(item[0])),
                key=lambda item: Version(item[0]),
                default=("?", "?"),
            )
            raise BakeError(
                f"{ref.source}: {ref.spec} has no release on or before {commit_date} "
                f"(newest {ref.name} is {newest[0]}, released {newest[1][:10]}). "
                "The buggy tree could not have been built with this requirement."
            )
        version = max(candidates, key=lambda item: Version(item[0]))[0]
        resolved.append(
            ResolvedDep(
                name=ref.name,
                spec=ref.spec,
                sources=(ref.source,),
                pin=f"{ref.name}=={version}",
                harness=False,
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
    )


def bug_sources(
    entry: Mapping[str, Any],
    *,
    bugsinpy_root: Path,
    repo: Path,
) -> DepSources:
    """Read the four approved sources for one bug, at the buggy commit.

    [VANTA-WHY] At the *buggy* commit and not at the fix: a dependency added by the
    fix is not a dependency the graded tree has, and installing it would provision
    an environment the bug never ran in.
    """
    project = str(entry["project"])
    bug_id = int(entry["bug_id"])
    commit = str(entry["buggy_commit"])
    notes: list[str] = []
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []

    # 1. BugsInPy's per-bug requirements.txt. Measured: 0 bytes for PySnooper 1-3,
    # which is exactly why the other three sources exist.
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

    # 3 + 4. The project's own declarations at the buggy commit.
    read_at = partial(tree_mod.read_file_at, repo, commit)
    setup_py = read_at("setup.py")
    if setup_py is None:
        raise BakeError(
            f"{project}/{bug_id}: no setup.py at {commit}, so install_requires and the "
            "declared test requirements cannot be read. Flag this bug rather than "
            "guessing at its dependencies (design §3.2)."
        )
    part, part_skipped = setup_py_refs(setup_py, read_at, notes)
    refs += part
    skipped += part_skipped

    return DepSources(
        project=project,
        bug_id=bug_id,
        checkpoint=bug_checkpoint_name(entry),
        buggy_commit=commit,
        commit_date=tree_mod.commit_date(repo, commit),
        import_name=import_name_of(setup_py),
        refs=tuple(refs),
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
    """
    return {
        "schema_version": 1,
        "project": sources.project,
        "bug_id": sources.bug_id,
        "checkpoint": sources.checkpoint,
        "buggy_commit": sources.buggy_commit,
        "commit_date": sources.commit_date,
        "import_name": sources.import_name,
        "harness_pins": dict(harness_pins),
        "installs": list(installs_of(resolved)),
        "dependencies": [dep.as_dict() for dep in resolved],
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


def resolve_bug_deps(
    split: Split,
    entries: Sequence[dict[str, Any]],
    *,
    bugsinpy_root: Path,
    repo_for: Callable[[str], Path],
    fetch: Callable[[str], Mapping[str, str]] = pypi_releases,
) -> list[BugDeps]:
    """Resolve every selected bug's layer, pinning by date from PyPI.

    ``fetch`` is injected so the entire chain — union, conflict, cutoff, harness
    exception — is testable without the network.
    """
    out: list[BugDeps] = []
    for entry in entries:
        sources = bug_sources(
            entry, bugsinpy_root=bugsinpy_root, repo=repo_for(str(entry["project"]))
        )
        pin = pins_for_python(split, minor_version(str(entry["python_version"])))
        harness_pins = {"pytest": pin.pytest, "pytest-json-report": pin.json_report}
        needed = sorted(
            {ref.name for ref in sources.refs if ref.name not in HARNESS_PACKAGES}
        )
        releases = {name: fetch(name) for name in needed}
        resolved = resolve_deps(sources, releases=releases, harness_pins=harness_pins)
        out.append(
            BugDeps(
                entry=dict(entry),
                checkpoint=sources.checkpoint,
                installs=installs_of(resolved),
                provenance=deps_provenance(sources, resolved, harness_pins=harness_pins),
            )
        )
    return out


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


def require_deps(deps_dir: Path, checkpoints: Sequence[str]) -> None:
    """Preflight every selected bug's provenance, before any checkout."""
    missing = sorted(name for name in checkpoints if not deps_path(deps_dir, name).is_file())
    if missing:
        raise BakeError(
            "no dependency provenance for "
            + ", ".join(missing)
            + f" under {deps_dir}. Re-run bake for these bugs — prepare reads the "
            "recorded pins rather than trusting a layer baked without one."
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
