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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .shuru import ExecResult, checkpoint_create, checkpoint_exists
from .split import Split, SplitError

DEFAULT_BASE = "vanta-python"
MARKER = "/opt/vanta-deps-ok"
"""Touched last, only after every assertion has passed."""

MIN_DISK_MB = 8192
"""The base image's disk. shuru refuses anything smaller."""

BAKE_DISK_MB = 16384
BAKE_TIMEOUT_SECS = 5400
"""90 min. py3.6/3.7 may compile OpenSSL 1.1.1 and CPython from source here."""


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


def _installs_the_project_itself(args: str) -> bool:
    """``python setup.py install`` / ``pip install -e .`` — the project, not a dep."""
    for token in args.split():
        if token in ("-e", "--editable") or token.startswith(("-e.", "-e/", ".", "/")):
            return True
        if token.endswith(".py"):
            return True
    return False


def declared_deps(setup_sh: str) -> list[str]:
    """Third-party specs from ``setup.sh``, in file order, de-duplicated.

    [VANTA-WHY] The project's own install line is dropped deliberately. The graded
    tree is mounted at ``/workspace/...`` and every suite runs ``python -m pytest``
    with its cwd set to the tree root, so ``import <package>`` resolves to the tree
    — the copy a run is being graded on. Baking ``setup.py install`` into a layer
    would instead put a *frozen* copy in site-packages and a Creator patch could
    then pass while having no effect at all. Projects that need a real build
    (``pythonpath=<pkg>/build/lib/``) get it per tree, at prepare time, in Shuru.
    """
    specs: list[str] = []
    for line in setup_sh.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIP_LINE_RE.match(line)
        if not match:
            continue
        args = match.group("args")
        if _installs_the_project_itself(args):
            continue
        specs.append(args)
    seen: set[str] = set()
    unique: list[str] = []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            unique.append(spec)
    return unique


def requirements_recipe(requirements_txt: str, extra_specs: Sequence[str] = ()) -> str:
    """Install one bug's dependencies on top of the interpreter layer."""
    body = "\n".join(
        [spec for spec in extra_specs if spec.strip()]
        + [line.strip() for line in requirements_txt.splitlines() if line.strip() and not line.strip().startswith("#")]
    ).strip()

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
    bugsinpy_root: Path,
) -> list[tuple[str, str, str]]:
    """``[(checkpoint name, base, script)]`` — every layer the bake would create.

    Pure string building (nothing is executed, nothing is downloaded), which is
    what makes the recipes testable without a VM: the assertions they contain are
    the wave-1 gate, and a gate that is only exercised by a real bake is a gate
    nobody re-checks.
    """
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
        base = python_checkpoint_name(minor_version(entry["python_version"]))
        specs = declared_deps(
            setup_sh_for(bugsinpy_root, entry["project"], entry["bug_id"])
        )
        requirements = requirements_for(
            bugsinpy_root, entry["project"], entry["bug_id"]
        )
        layers.append(
            (
                bug_checkpoint_name(entry),
                base,
                requirements_recipe(requirements, specs),
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


def frozen(name: str) -> bool:
    return checkpoint_exists(name)
