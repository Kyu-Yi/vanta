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
import io
import json
import re
import shlex
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Container, Iterable, Mapping, Optional, Sequence

from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .bugsinpy_meta import read_metadata_text
from . import suite as suite_mod
from . import trees as tree_mod
from .shuru import (
    DEV_SHM_ENV,
    DEV_SHM_MOUNTED,
    MP_SMOKE_CODE,
    MP_SMOKE_PREFIX,
    ExecResult,
    checkpoint_create,
    checkpoint_delete,
    checkpoint_exists,
    startup_check_lines,
)
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
"""Provisioned by the *interpreter* layer, for every bug on that Python minor.

A bug that declares one of these as a **range** — PySnooper's
``test_requirements.txt`` says ``pytest>=4.4.1`` — is a constraint on a package the
interpreter layer already carries, so it is *checked* against the harness pin and
recorded rather than installed: a range names no release, and the layer already has
a concrete one. A project that asks for a range the harness pin does not satisfy
fails the bug loudly instead of silently re-pinning the interpreter every wave.

[VANTA-WHY] An exact ``==`` pin is a different claim, and it wins. ``fastapi/1``'s
``requirements.txt`` says ``pytest==5.4.3``: that is the environment its own
``run_test.sh`` was validated fail-before/pass-after in, the same claim that makes
:func:`reference_pin` install any other BugsInPy pin as written. So the pin is
installed by the *per-bug* layer, on top of the interpreter layer's default, and the
two packages that declare a pytest requirement of their own are re-chosen to accept
it (:func:`plugin_pins`). The interpreter layer keeps the default
(:func:`pins_for_python`) — four bugs on Python 3.8 pin four different pytest
versions, which one shared interpreter layer cannot carry — and a bug with no pin of
its own runs under exactly that default.
"""

PLUGIN_PACKAGES = ("pytest-json-report", "pytest-metadata")
"""The two packages whose own requirements decide which release the layer can carry.

[VANTA-WHY] ``pytest-json-report`` is the plugin that writes the report the oracle
is read from, and every release from 0.3 on declares ``pytest>=3.3.2`` or higher.
``pytest-metadata`` it requires with **no** bound at all, so pip installs the newest
one — measured: 3.1.1, which declares ``pytest>=7.0.0``. Against a bug whose own pin
is pytest 5.4.3 that is not a warning but a broken layer (``pip check``: ``pytest-metadata
3.1.1 has requirement pytest>=7.0.0, but you have pytest 5.4.3``). Both are therefore
resolved against the bug's own pytest pin rather than left to pip, and both go into
the layer's install list and into its provenance.
"""

PLUGIN_SOURCES = {
    "pytest-json-report": "the report plugin (every bug)",
    "pytest-metadata": "the report plugin's metadata dependency",
}
"""What is recorded as having asked for a plugin pin — nothing *declared* it."""

DEPS_DIRNAME = "deps"
"""Where ``bake`` leaves each per-bug layer's dependency provenance.

[VANTA-WHY] The provenance lives outside the layer, in ``runs/bugsinpy/deps/``,
because ``prepare`` has to read it *before* it runs anything inside the layer and
the checkpoint format offers no way to read a file out of a layer without booting
it. It is the record of what the bake decided — sources, pins, skipped lines — so
``prepare`` never has to re-derive a decision that needed the network.
"""

BUILD_TOOLS = ("setuptools", "wheel", "cython")
"""The build-time toolchain, pinned by date for every layer, installed first (§3.2).

[VANTA-WHY] Measured on matplotlib/1: the pinned ``numpy==1.18.5`` has no wheel for
the sandbox (linux/aarch64, py3.8), so pip built it from source in an **isolated**
build environment — which takes *today's* setuptools, and setuptools has not carried
``distutils.msvccompiler`` for years, so numpy's own
``numpy/distutils/mingw32ccompiler.py`` died with ``ModuleNotFoundError``. A build-time
tool is as much a part of "the environment this tree was built in" as a dependency is,
so it is pinned by the same date rule: the newest release on or before the bug's commit
date, installed into the per-bug layer *before* its dependencies, and every dependency
install then runs with ``--no-build-isolation`` so a source build uses these pins
rather than a build environment that did not exist when the tree did.

Names are canonicalised the way every other name in this module is
(:func:`packaging.utils.canonicalize_name`), so ``cython`` is PyPI's ``Cython``.
"""

BUILD_REQUIRES_SOURCES = ("setup_requires (setup.py)", "build-system.requires (pyproject.toml)")
"""The two places a project declares **build** requirements (PEP 518 / setuptools).

These are not dependencies: nothing here is installed for the suite to import. They
say which tools the tree's own build needs (pandas needs Cython and numpy to compile
``pandas/_libs``), and that is what turns "pin setuptools, wheel and Cython" into a
per-project answer for ``numpy when it is a build requirement``.
"""

BUILD_TOOL_SOURCE = "the build toolchain (every bug)"
"""The ``source`` recorded for :data:`BUILD_TOOLS`' own three pins in the provenance.

A toolchain pin has no declaring line to name — it is the harness's own answer to
"what was this tree built with" — and the record has to distinguish it from a pin the
project asked for by name (``setup_requires=["numpy>=1.15"]``), which carries the line
that declared it.
"""

BUILD_PIN_RULE = "the newest release on or before the bug's commit date"
"""One phrase for the provenance, shared with §3.2's rule for dependencies."""

BUGSINPY_PIN_RULE = "BugsInPy's own exact pin, installed as written"
"""The ``pin_rule`` of a pin that came from BugsInPy's own files (:func:`reference_pin`).

A module constant because it is also the **predicate** two later rules key off: which
pins a conflict is allowed to involve (:func:`own_pin_names`), and which names the
layer's own ``pip check`` readings may tolerate (:func:`pip_check_conflicts`). Two
spellings of that one fact would let a conflict be tolerated in the record and refused
in the layer.
"""

PIP_TOOL = "pip"
"""The one build tool that is pinned by **name** rather than by date (§3.2).

[VANTA-WHY] Measured on sanic/1. Its ``requirements.txt`` is a pip freeze from 2020 —
``h11==0.9.0`` (line 40) and ``httpcore==0.3.0`` (line 44) — and httpcore 0.3.0's own
metadata says ``h11==0.8.*``. Two of BugsInPy's own exact pins, jointly unsatisfiable.
The environment they were frozen from installed both anyway, because pip before 20.3
*had* no resolver that could refuse: it installed each requirement in turn and printed
"pip's dependency resolver does not currently take into account all the packages that
are installed". Today's pip answers ``ResolutionImpossible`` instead and the layer never
gets built — which loses the benchmark's environment rather than reproducing it.

So the layer installs with a pip that still has the old behaviour:
``--use-deprecated=legacy-resolver``. Measured against pip's own tagged sources, that
option is in ``pip/_internal/cli/cmdoptions.py`` at 20.3, 20.3.4, 21.0, 21.3, 21.3.1,
23.1.2, 23.3.2, 24.0, 25.3, 26.1 and 26.2.2, and absent at 19.3.1, 20.0.2, 20.1 and
20.2.4 — the opt-out arrived with the resolver that made it necessary. That choice is not
a date rule either: sanic/1's commit date is 2020-05-12, *before* 20.3 existed, so the
date rule would pick a pip with no flag at all. It is a capability instead, and
:data:`PIP_PIN_VERSION` records which version and why.
"""

PIP_PIN_VERSION = "21.3.1"
"""The pinned ``pip``: the newest flag-bearing release every split's Python can run.

[VANTA-WHY] 21.3.1 is in the flag list above, and it is the *last* release that installs
on Python 3.6 — 22.0 moved its floor to 3.7, 24.1 to 3.8, 25.0 to 3.9 — and the full
split's oldest interpreter is 3.6.9. It is the newest of the candidates rather than the
oldest for a reason: the pins are 2020-era, but the legacy resolver still installs
*today's* newest release of everything the files left unpinned, and a newer pip reads
more of what those wheels declare.
"""

PIP_TOOL_SOURCE = "the legacy resolver's pip (every bug)"
"""The ``source`` recorded for the ``pip`` pin, next to the toolchain's own label."""

PIP_PIN_RULE = "the newest pip that still carries --use-deprecated=legacy-resolver"
"""How the ``pip`` pin's version was chosen — a capability, never a date."""

LEGACY_RESOLVER_FLAG = "--use-deprecated=legacy-resolver"
"""Every install *after* the toolchain carries this, and the reason is below.

[VANTA-WHY] One rule and not a flag per bug: a layer's environment *was* built by a pip
of this era, so every install in it that resolves this bug's pins reproduces those
semantics. The toolchain install is the one exception, and it is not an oversight: it is
the install that *puts this pip there*, so it runs under the interpreter layer's seeded
pip and there is nothing yet for the flag to change. The tolerance is not open-ended
either — what the legacy resolver installs a conflict for is judged afterwards, by the
``pip check`` gate at the end of the recipe (:func:`pip_check_conflicts`).
"""

PIP_CHECK_PREFIX = "VANTA-PIP-CHECK"
"""The marker the layer's ``pip check`` lines carry into the build log."""

PIP_CHECK_OUTPUT = "/opt/vanta-pip-check.txt"
"""Where the layer keeps its own ``pip check`` output, next to the freeze (§3.2).

The recipe echoes it back with :data:`PIP_CHECK_PREFIX` on every line, because a
checkpoint is a filesystem and reading a file out of one means booting it
(:data:`DEPS_DIRNAME`). The judgement is made on the host, in
:func:`pip_check_conflicts`, which is where the layer's own-pin names are known.
"""

PIP_CHECK_KEY = "pip_check"
"""The deps-provenance key the layer's ``pip check`` readings are recorded under."""

PIP_CHECK_CLEAN = "No broken requirements found."
"""What ``pip check`` writes when it has nothing to report — *no* conflicts, not a reading.

[VANTA-WHY] It reaches the gate as a reading like any other line, and the gate's rule is
that a line it cannot read is case (a) and refused (:func:`pip_check_conflicts`). Applied
to this sentence that rule refuses a layer for its clean bill of health: measured on
PySnooper/3, which installed its four pins, echoed this line, touched
``/opt/vanta-deps-ok``, had its checkpoint saved — and was then deleted with
``case (a) No broken requirements found. but the layer has nothing: not tolerated``. The
sentence is the whole content of :data:`PIP_CHECK_OUTPUT` when the layer is clean, so it
has to be recognised rather than read. It is matched exactly and not by prefix: a pip
whose wording changed would fall back to the unreadable rule and stop the bake, which is
the direction this gate has to fail in — but it would stop every layer at once, so a
wording change must be a one-line change here and not a silent pass.
"""

DEPS_FREEZE_OUTPUT = "/opt/vanta-deps-freeze.txt"
"""Where a layer keeps its own ``pip freeze``, next to ``pip check``'s output (§3.2)."""

PIP_FREEZE_PREFIX = "VANTA-PIP-FREEZE"
"""The marker the layer's ``pip freeze`` lines carry into the build log.

[VANTA-WHY] The freeze is the only reading of a layer that names *every* installed
distribution — the pins and whatever ``pip`` resolved besides them — and like
:data:`PIP_CHECK_OUTPUT` it is a file inside the checkpoint, which cannot be read
without booting it (:data:`DEPS_DIRNAME`). The recipe echoes it back line by line with
this prefix and the diff is taken on the host (:func:`unpinned_freeze`), because that is
where the pin list is known — a shell test inside the layer has nothing to compare to.
"""

CONSTRAINTS_FILE = "/tmp/constraints.txt"
"""The one constraints file every install in a bug layer is held to (§3.2).

[VANTA-WHY] ``fastapi/1`` measured what a step outside the pins costs. The layer's pin
list holds ``requests==2.23.0`` (the date rule's answer for the release the tree's
``requirements.txt`` named), but the layer's **first** install — the dated build tool
``flit==2.3.0`` — ran with no ``-c``, so pip resolved flit's own ``requires_dist`` against
the PyPI of the day and put ``requests-2.32.4`` in the layer. The install that followed
then downgraded requests to its pin and left ``charset-normalizer-3.5.2`` behind, which is
a dependency of 2.32.4 and of nothing in this layer's pin list — so the freeze gate
refused the layer, correctly, for a name no pin of it names. Every pin a layer installs
is an exact ``==`` (:func:`resolve_pins`, :func:`resolve_build_pins`, the closure walk),
which is what makes one constraint file per layer a complete answer: written from the
layer's whole pin list and passed to all four installs, it means pip cannot resolve a
pin's requirements to any version the layer has not already decided — in the toolchain
step, the build-requirement step, the early step or the dependency step.
"""

CONSTRAINTS_MARKER = "VANTA_CONSTRAINTS"
"""The heredoc delimiter the constraints file is written with, beside the other four."""

TRANSITIVE_PIN_RULE = "transitive, date rule"
"""The ``pin_rule`` of a pin the closure walk added, and not a declaration (§3.2).

[VANTA-WHY] Every other pin in a layer exists because something in the bug's tree, or
the build toolchain, or BugsInPy's ``requirements.txt``, named it. These exist because a
*release* named them: ``pip`` resolves those itself, at the version of the day, and the
date rule never sees them — ``fastapi/1`` measured the cost, an ``exceptiongroup==1.3.1``
from 2025 in a layer whose own pins hold ``typing-extensions==3.7.4.2``. The walk
(:func:`transitive_pins`) closes that gap by pinning the name itself, and the provenance
has to say that the pin is not a declaration: hence a phrase of its own and
:attr:`ResolvedDep.required_by` naming the pin that required it.
"""

ORPHAN_RULE = (
    "the interpreter layer holds it as a dependency of a harness pin this layer "
    "replaces, and this layer's pin list does not contain it"
)
"""Why a distribution is uninstalled from a bug layer rather than exempted (§3.2).

[VANTA-WHY] The mirror of :data:`TRANSITIVE_PIN_RULE`, and the other half of the same
fact: ``pip`` does not uninstall what it replaces, so a layer that installs
``pytest==5.4.3`` over the interpreter layer's ``8.0.2`` is left holding 8.0.2's own
dependencies — ``exceptiongroup==1.3.1``, ``iniconfig``, ``tomli`` — which no pin of it
names. Exempting them (which is what the freeze gate did) leaves a layer of 2020 with a
release of 2025 in it, and ``fastapi/1`` measured the cost: ``pip check`` refused the
layer on ``exceptiongroup 1.3.1 requires typing-extensions>=4.6.0`` against its own
``typing-extensions==3.7.4.2``, a contradiction nothing in the layer had declared. So
they are **removed**, before the layer's own ``pip check`` and before the freeze: the
recipe uninstalls each one by name, and the freeze gate is what reads that the removal
happened (:func:`unpinned_freeze`). Which names those are is the walk's third answer
(:func:`transitive_pins`), and the reason each was *there* is the record it returns with
them (:class:`OrphanedPin`).
"""

TREE_SATISFIED_RULE = (
    "the tree under test provides it, so the layer installs nothing for it"
)
"""Why a requirement a release declared is left to the tree rather than pinned (§3.2).

[VANTA-WHY] The tree under test ships its own distribution, and the suite imports *that*
one — the layer installs beside it, never over it. So a release that declares the
project's own name is asking for something the layer already has: measured, ``pandas/1``'s
closure walk pinned ``pandas==1.0.3`` because ``statsmodels==0.11.1`` and
``xarray==0.15.1`` each declare ``pandas``, and the recipe would have installed a released
pandas over the very tree the bug is graded on. The exemption is the one the import
closure already makes (:func:`tree_provided_modules`): the project's own distribution name
and every top-level module the tree ships are names the tree provides, whoever asks for
them. What the walk returns instead is the record (:class:`TreeSatisfied`), which is how
the provenance says the requirement *was* satisfied rather than quietly missing a pin.
"""

ORPHAN_PREFIX = "VANTA-ORPHAN"
"""The marker the recipe's uninstall step prints into the build log (§3.2).

[VANTA-WHY] The deps JSON is the record of *which* names were removed and why
(:class:`OrphanedPin`), and the freeze gate is what reads that the removal took:
:func:`unpinned_freeze` exempts ``provisioned`` and these names are deliberately not in
it, so a name the layer failed to uninstall is reported there rather than passed. This
prefix exists for the human reading the build log — ``pip uninstall`` prints what it
removed on lines nobody can grep, and the one thing a reader needs next to an empty
``VANTA-PIP-FREEZE`` block is *why* those names are absent from it.
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
SDIST_SYSTEM_PACKAGES: Mapping[str, tuple[str, ...]] = {
    # The evidence comment on each line is the review: the pinned sdist's own
    # ``setup.py``/``setup.cfg`` read (never executed) for ``libraries=``,
    # ``library_dirs``, a pkg-config/``find_library``/``ctypes.util`` call, or an
    # include of a header the archive does not carry.
    "asyncpg": (),  # Cython pgproto only; no libpq, no external header
    "bcrypt": ("libffi-dev",),
    "brotli": (),  # vendored c/; <vector> is libstdc++, which g++ brings
    "cffi": ("libffi-dev",),
    "cftime": (),  # cftime/_cftime.pyx only (numpy's headers); udunits is named in its
    # docstrings and nowhere else, and there is no site.cfg to point a build at one
    "coverage": (),  # coverage/ctracer/* only, all in the sdist
    "cython": (),  # its own setup.py names no library; Demos/ and docs/ do
    "distlib": (),  # PC/launcher.c is the Windows launcher, never built here
    "fastparquet": (),  # speedups.pyx/.c; numpy's headers, CPython's own, and omp.h —
    # which is the compiler's, and setup.py sets no -fopenmp for it to be used with
    "httptools": (),  # vendor/http-parser/ is inside the sdist
    "lxml": ("libxml2-dev", "libxslt1-dev"),
    "markupsafe": (),  # markup/_speedups.c; <Python.h> only
    "matplotlib": (
        "libfreetype-dev",
        "libpng-dev",
        "zlib1g-dev",
        "pkg-config",
    ),  # setupext.py: FreeType via pkg-config 'freetype2' and Png via 'libpng', both
    # default_libraries=[..., deplib('z')] — and get_pkg_config() is what locates them
    "multidict": (),  # multidict/_multidict.c + _multilib/*.h only
    "numpy": (),  # bundled f2c lapack_lite; get_info('lapack_opt', 0) falls back
    "numexpr": (),  # setup.py's libraries=/library_dirs= are inside the MKL branch
    # guarded by os.path.exists('site.cfg') — the sdist ships only site.cfg.example —
    # so the extension that gets built is the other one, whose libraries=['m']
    "peewee": ("libsqlite3-dev",),  # setup.py libraries=['sqlite3'] (optional ext)
    "pillow": ("libjpeg-dev",),
    "psutil": (),  # its libraries= are devstat/kvm/kstat/perfstat — all non-Linux
    "psycopg2": ("libpq-dev",),
    "psycopg2-binary": ("libpq-dev",),
    "pynacl": ("libffi-dev",),
    "pycryptodome": (),  # setup.py's get_libraries returns [] on purpose ("avoid linking
    # Python's dynamic library"), and every include it adds is src/ or src/libtom/ —
    # vendored — besides the compiler's own intrinsics headers
    "pyyaml": ("libyaml-dev",),  # setup.py libraries=['yaml'] for ext/_yaml.pyx
    "regex": (),  # regex_2/ and regex_3/ are in the sdist
    "ruamel-yaml-clib": (),  # libs come from its own ext_modules; it lists none
    "scikit-learn": ("gfortran", "libopenblas-dev", "liblapack-dev"),
    "scipy": ("gfortran", "libopenblas-dev", "liblapack-dev"),
    "sqlalchemy": (),  # cextension/{processors,resultproxy,utils}.c only
    "statsmodels": ("gfortran", "libopenblas-dev", "liblapack-dev"),
    "tornado": (),  # tornado/speedups.c only
    "ujson": (),  # vendored double-conversion; -lstdc++/-lm are the toolchain's
    "urwid": (),  # source/str_util.c only
    "uvloop": ("make", "autoconf", "automake", "libtool"),
    "websockets": (),  # src/websockets/speedups.c only
    "wrapt": (),  # wrapt/_wrappers.c; <Python.h> and <structmember.h> only
    "yarl": (),  # yarl/_quoting.c only
    "zope-interface": (),  # src/zope/interface/_zope_interface_coptimizations.c
    "zstandard": (),  # vendored c-ext/ only
}
"""Reviewed table: a package that compiles from sdist → the Debian trixie packages it needs.

[VANTA-WHY] The VM is ``linux/aarch64`` and most of these pins predate aarch64 wheels,
so pip builds them from source — and each missing ``-dev`` header costs a *whole bake
cycle* to discover, because the failure surfaces as that one bug's build error
(measured: ``cffi``'s wheel build died on ``c/_cffi_backend.c:15:10: fatal error:
ffi.h: No such file or directory``, i.e. on a package nothing in the split's metadata
mentions). The mapping is a *review*, in the same shape as the split's
``module_distributions``: it is the decision of which system library a measured sdist
builds against. Two things read it — :func:`_system_packages_for_py38` (the 3.8
interpreter layer's install list) and :func:`_review_of` (the preflight's gap check, so
a measured build that no entry covers is reported by name instead of discovered at
build time).

The names are canonical distribution names, because that is what the resolver pins and
what :func:`preflight_pin` classifies with. Three entries need a word:

* ``cryptography`` is **not here**, and its absence is a measure of this table's shape
  rather than an oversight. It was ``libssl-dev`` for as long as the review stood for
  the one build the split pins (``2.9.2``, the cffi/OpenSSL one), and that build is one
  no header can fix: ``2.9.2`` needs OpenSSL **1.1**, and this image is trixie, which
  ships OpenSSL 3 — from 35.0 on the sdist takes 3.x, and from 3.4 on it needs a Rust
  toolchain. So the pin is a :data:`UNBUILDABLE_ON_PLATFORM` one by its *version*, and
  the version is what the entry now carries. ``libssl-dev`` consequently leaves the
  interpreter layer (:func:`_system_packages_for_py38`), which is a fact about that
  layer and not about this table: nothing else the split builds links OpenSSL.
* ``pillow`` joins ``libjpeg-dev`` to the ``libpng-dev``/``libfreetype-dev``/``zlib1g-dev``
  the base list already carries: those three are for matplotlib's ``_png``/``ft2font``,
  and pillow needs the fourth for JPEG.
* ``uvloop`` vendors libuv and builds it with its own ``autogen.sh``, so it needs the
  autotools *and* ``make`` rather than a ``-dev`` header.

An entry with an empty tuple is a legitimate review result — "this package compiles,
and against nothing but libc and the compiler already in the layer" — and is how a
:func:`_review_of` gap gets closed without pretending a package needs something.

[VANTA-WHY] Every entry above was decided by reading the *pinned sdist*, not by
knowing the package: ``python3 scripts/bugsinpy_run.py bake --preflight-only`` names
the pins that compile and no entry covered, and each of those was re-read at its exact
pinned version for ``libraries=``, ``library_dirs``, a pkg-config/``find_library``/
``ctypes.util`` call, or an include of a header the archive does not carry. The one
finding that did not go to plan is worth keeping: the *raw* compiled-source count that
:func:`compiled_sources` produces is not evidence of a build. ``distlib`` 0.3.0's is
``PC/launcher.c`` — the Windows launcher, which nothing on Linux compiles — and
``cffi``'s includes ``demo/*.c``, which are examples. Both packages' entries are empty
for that reason, and the count stays in the record as a raw fact rather than as the
reason a build is said to need a library (:func:`_sdist_reason`).
"""

@dataclass(frozen=True)
class PlatformLimit:
    """One reason a package's sdist cannot be built here, and the releases it covers.

    [VANTA-WHY] The reason and the releases it is a reason *for* used to be two facts in
    two places — the table here carried the reason, and the boundary lived in a comment
    (:data:`SDIST_SYSTEM_PACKAGES`'s ``cryptography`` bullet said "only before 3.4" and
    then apologised that the entry could not carry it). ``cryptography`` is the case that
    made the apology untenable: which of its two impossible builds a pin runs into
    depends entirely on the version, so a name-keyed reason was one answer where there
    are two.

    ``bound`` is empty for a limit that holds for every release (``llvmlite``'s LLVM, and
    the other two seeded entries), and a bounded :class:`SpecifierSet` otherwise. Clauses
    are consulted **in order** and the first one whose bound covers the pin decides, so a
    name with two impossible windows lists the reason a reader should see first for the
    version at hand.
    """

    reason: str
    bound: SpecifierSet = SpecifierSet("")

    def covers(self, version: str) -> bool:
        """Whether this reason applies to ``version``.

        An empty bound is every release, and it is spelled out rather than left to
        ``SpecifierSet``: an empty set excludes prereleases from ``in`` (packaging reads
        the empty string as "no opinion about prereleases"), and "every release" has to
        mean every release. ``prereleases=True`` on the bounded case is the same
        arithmetic-and-not-install-policy decision — ``<35`` is a statement about release
        order, and a ``35.0rc1`` is below 35 whatever pip would do with it.

        A version nothing can be read for is not one this limit claims: the fall-through
        is the gap report, which is the answer that asks for a review rather than
        inventing one.
        """
        if not str(self.bound):
            return True
        try:
            return self.bound.contains(version, prereleases=True)
        except InvalidVersion:
            return False

    def __str__(self) -> str:
        return f"{self.bound or 'every release'}: {self.reason}"


UNBUILDABLE_ON_PLATFORM: Mapping[str, tuple[PlatformLimit, ...]] = {
    "cryptography": (
        PlatformLimit(
            "its sdist builds the C bindings against OpenSSL's headers, and every "
            "release before 35 needs the 1.1 ones, which this image does not carry: it "
            "is trixie, whose OpenSSL is 3.x, and cryptography only builds against 3.x "
            "from 35.0 on (its releases from 3.4 on need a Rust toolchain as well, "
            "which is the other half of what this table has to say about it)",
            bound=SpecifierSet("<35"),
        ),
        PlatformLimit(
            "its sdist builds the Rust binding, and this image carries no Rust "
            "toolchain at all",
            bound=SpecifierSet(">=35"),
        ),
    ),
    "llvmlite": (
        PlatformLimit(
            "its sdist builds against a specific LLVM release's C API; this image "
            "carries no LLVM, and the version it wants is not the one any current "
            "trixie package ships"
        ),
    ),
    "numba": (
        PlatformLimit(
            "it is built against llvmlite, which needs a specific LLVM release this "
            "image does not carry, so its own C extension cannot be built either"
        ),
    ),
    "pyarrow": (
        PlatformLimit(
            "its sdist needs the Arrow C++ toolchain (a full C++17 build plus its "
            "vendored third-party sources), which this image does not carry"
        ),
    ),
}
"""Packages whose *sdist* also cannot be built here, and why, as a review.

[VANTA-WHY] Different from a missing ``-dev`` header: no entry in
:data:`SDIST_SYSTEM_PACKAGES` would help, so a pin here is not a gap to close but a
fact to report. Preflight says so by name instead of letting the bake spend a cycle
building Arrow C++ to find out (measured: pandas/1's ``pyarrow==0.17.1``), and
:func:`decide_unresolvable` treats such a pin exactly like one with no Linux
distribution at all (§3.2): dropped when nothing imports it, and the bug flagged when
something does.

Each value is the name's :class:`PlatformLimit` clauses, in the order they are read, and
the first one covering the pin's version is what the preflight reports. A name with one
clause is a package no release of which can be built here; a name with several is one
whose impossible build changes with the version — which is the shape ``cryptography``
forced, and the reason the table is version-scoped at all rather than name-keyed.
"""


def _system_packages_for_py38() -> tuple[str, ...]:
    """The 3.8 interpreter layer's list: the base plus every entry of the table.

    Computed rather than written out twice, so "the layer carries what the reviewed
    table says it needs" is a property of the code and not of someone remembering to
    edit two lists. A table entry that is not in this union would be a gap that no
    preflight could close.
    """
    return tuple(
        sorted(
            set(_PY38_BASE_PACKAGES)
            | {package for packages in SDIST_SYSTEM_PACKAGES.values() for package in packages}
        )
    )


_PY38_BASE_PACKAGES = (
    # g++ brings gcc with it, which is also the driver LDSHARED names (§ COMPILER_ENV).
    "g++",
    "pkg-config",
    "libfreetype-dev",
    "libpng-dev",
    "zlib1g-dev",
)

SYSTEM_PACKAGES: Mapping[str, tuple[str, ...]] = {"3.8": _system_packages_for_py38()}
"""Build-time system packages, per Python minor, installed into the interpreter layer.

[VANTA-WHY] These are the *system* half of §3.2's build toolchain: the pinned
``setuptools``/``wheel``/``Cython`` are Python packages, but the extensions the
measured trees compile (matplotlib's ``_png``/``ft2font``, pandas' ``_libs``,
spacy's Cython modules) want a C++ compiler and the ``-dev`` headers they link
against. They go in the **interpreter** layer, not in a per-bug layer, because all
bugs on one minor share it — one environment, so a bug's layer is only ever what
its own metadata added.

All eleven dev-split bugs are 3.8, which is why 3.8 is the minor that has them.
The keys are a decision per minor and not a switch: a minor that is absent gets no
system packages, so the 3.6/3.7 layers keep the recipe they were baked with and are
neither rebuilt nor re-downloaded by this. Extending the list is a base-image change
and needs the same approval as adding to it the first time.

A compiler being present does not change a pure-Python bug's tests: ``needs_build``
(§3.2) decides per bug whether anything is compiled, and a bug that compiles nothing
never invokes the toolchain, so its verdict is the same as on a layer without one.
"""

_DPKG_QUERY_STATUS = "dpkg-query -W -f='${Package} ${Version} ${db:Status-Status}\\n'"
"""One line per package: name, version, and dpkg's own ``status:status`` field.

Not ``dpkg-query -W`` bare: for a package dpkg knows but has not installed that
prints an entry with a version and exits 0, which would read as present. The third
field is what distinguishes "installed" from "known".
"""


def system_package_lines(packages: Sequence[str]) -> list[str]:
    """Install, verify, and record the interpreter layer's system packages.

    [VANTA-WHY] Verified rather than trusted, for the same reason the pip and ssl
    checks below are: a layer that boots without a compiler looks baked, and the
    failure then surfaces mid-bake on a *bug's* layer as a build error whose text is
    ``g++: not found`` — i.e. as that bug's problem, on every wave, when the cause is
    the interpreter layer. A missing package therefore fails **this** layer's bake,
    which is what makes ``checkpoint_create`` delete it instead of caching it.

    The versions go to ``/opt/vanta-system-packages.txt`` inside the layer — the
    interpreter layer's provenance for its system half, beside the ``/opt/vanta-deps-freeze.txt``
    every layer already carries for its pip half. Like the freeze file it is part of
    the layer, because that is what makes it true of the layer: the record is written
    by the same script that does the install, and read back by booting the checkpoint
    (§3.2's ``python -m pip freeze`` pattern).

    Empty for a minor not in :data:`SYSTEM_PACKAGES`, so the recipe of a layer without
    them is unchanged, line for line.
    """
    if not packages:
        return []
    names = " ".join(packages)
    return [
        # The base image's apt lists are empty, so an install without this resolves
        # nothing. Network is allowed at bake time and only here (§3.2).
        "apt-get update",
        f"DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {names}",
        f"{_DPKG_QUERY_STATUS} {names} > /opt/vanta-system-packages.txt",
        "cat /opt/vanta-system-packages.txt",
        # `set -e` would stop at the first miss, so the misses are collected first and
        # the message names all of them: one bake tells you the whole list.
        f"missing=''; for pkg in {names}; do "
        f"{_DPKG_QUERY_STATUS} \"$pkg\" 2>/dev/null | grep -q ' installed$' "
        '|| missing="$missing $pkg"; done',
        '[ -z "$missing" ] || { echo "system packages missing from the layer after '
        f'apt-get:$missing" >&2; exit 1; }}',
    ]


_ERA_WARNING_FLAGS = (
    "-Wno-error=incompatible-pointer-types",
    "-Wno-error=implicit-function-declaration",
    "-Wno-error=int-conversion",
    "-Wno-error=implicit-int",
    "-Wno-error=return-mismatch",
    "-Wno-error=declaration-missing-parameter-type",
)
"""The six diagnostics GCC 14 promotes from warning to **error**, demoted back.

[VANTA-WHY] These are the *ancien régime* of C diagnostics: for the four decades the
2020 sdists in this benchmark were written in, an implicit function declaration, an
``int``↔pointer conversion, a mismatched pointer assignment and a missing parameter
type were warnings that scrolled past. GCC 14 (the base image's compiler) turns all
six into errors by default — the documented GCC 14 porting change — and that is the
measured shape of the failures this fixes: lxml's C sources failed on
``incompatible pointer type`` and ruamel.yaml.clib on a handler signature mismatch,
both 2020-era code that compiled under the GCC of its day.

``-Wno-error=``, not ``-Wno-``: the diagnostics stay *visible* as warnings and only
their promotion to failure is undone, which is exactly "the era's warning semantics"
and keeps a reader of the build log able to see what the old compiler saw.

``CFLAGS`` rather than ``CXXFLAGS`` because every one of the six is a C-only
diagnostic and the measured failures were in C sources; distutils'
``customize_compiler`` appends an environment ``CFLAGS`` to both ``cflags`` and
``ldshared``, so this is the variable the C compilation and the link line read.
"""

_ERA_FORTRAN_FLAGS = ("-fallow-argument-mismatch",)
"""GCC 10+'s Fortran counterpart to :data:`_ERA_WARNING_FLAGS`: one error, demoted.

[VANTA-WHY] ``gfortran`` reads the same legacy code the C list above is written for, and
GCC 10 promoted its *argument-mismatch* family from warning to **error** exactly as GCC
14 promoted the six C diagnostics. Measured on ``pandas/1``, whose layer builds
``scipy==1.4.1`` from its sdist: its ARPACK sources call ``dvout`` twice with the same
arity and a different rank in one position, and gfortran refused the pair —

``scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaitr.f:658:35:`` … ``376 | call dvout
(logfil, 1, rnorm, ndigit,`` / ``658 | call dvout (logfil, 2, xtemp, ndigit,`` …
``Error: Rank mismatch between actual argument at (1) and actual argument at (2) (scalar
and rank-1)``

— twice (``:658`` and ``:729``), after which the build stopped on ``error: Command
"/usr/bin/gfortran -Wall -g -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
… dnaitr.f -o dnaitr.o" failed with exit status 1`` and ``ERROR: Failed building wheel
for scipy``. Nothing there was killed, out of memory, or missing a BLAS library: the
compile ended in an ``Error:`` gfortran had printed as a ``Warning:`` in every release
before GCC 10. The mismatch is *between two calls of the same external subroutine*, which
is the shape this flag is documented for — "mismatches between different calls" — and it
is the whole of what the flag does: the diagnostic stays **visible** in the log and stops
being fatal, the same choice :data:`_ERA_WARNING_FLAGS` makes for C.

``FFLAGS``, not ``F77FLAGS``/``F90FLAGS``: ``scipy``/``numpy`` build Fortran through
``numpy.distutils``, whose ``FCompiler`` names ``FFLAGS`` as the environment variable of
its **generic** flag list (``flag_vars.flags = ('self.get_flags', 'FFLAGS', 'fflags',
…)`` in ``numpy/distutils/fcompiler/__init__.py``), so one variable reaches the
fixed-form ``f77`` sources that failed here *and* the free-form ones beside them; the
per-flavour variables are declared in the same table and cover only their own form.
"""

COMPILER_ENV: Mapping[str, str] = {
    "CC": "gcc",
    "CXX": "g++",
    "LDSHARED": "gcc -pthread -shared",
    "LDCXXSHARED": "g++ -pthread -shared",
    "CFLAGS": " ".join(_ERA_WARNING_FLAGS),
    "FFLAGS": " ".join(_ERA_FORTRAN_FLAGS),
}
"""The compiler a source build in these layers runs, the linker that follows it, and
the C and Fortran flags that restore the warning semantics of the era the sources were
written in.

[VANTA-WHY] What the interpreter recorded is printed by :func:`_sysconfig_probe`
before these exports, so it is a reading and not a guess; the baked
``vanta-bi-fastapi-1`` log carries ``recorded CC: cc``, ``recorded CXX: c++``,
``recorded LDSHARED: cc -shared -Wl,--exclude-libs,ALL`` and
``recorded LDCXXSHARED: c++ -shared``. uv's interpreter is a python-build-standalone
build, so those strings are its build machine's and an old-style ``distutils`` build
prefers them to anything the caller assumed — bare drivers that do resolve here, but
not the toolchain :data:`SYSTEM_PACKAGES` installs, and a ``CFLAGS`` exported into
the shell reaches the compile *and* the link line only through the override below.
These four are ``distutils``' documented override (``customize_compiler`` reads
``CC``/``CXX``/``LDSHARED`` from the environment before the recorded value), and
``g++ -pthread -shared`` is CPython's own Linux default for ``LDCXXSHARED``, so
nothing about the link line is invented here.

[VANTA-WHY] The ``error: [Errno 2]`` this mapping was first written for —
``numpy==1.18.5`` built from its sdist in the ``matplotlib/1`` layer, *after* the
BLAS/LAPACK fallback warnings, so it read as the compiler invocation — is not the
compiler. It is the VM's missing ``/dev/shm``, which CPython's multiprocessing needs
for the ``SemLock`` that numpy's parallel compile creates; measured, and fixed per
boot rather than here (:data:`bugsinpy.shuru.DEV_SHM_PATH`). The four driver
variables remain because they pin the toolchain the flags above are written for, and
their assertion is what keeps that a fact about the layer rather than a recipe line
nobody can read after the fact.

The fifth variable is the other half of the same fact — which compiler runs, and
what that compiler's *own* defaults make of 2020-era C — and it is in the same
mapping for the same reason: both are facts about the toolchain this layer runs
rather than about the tree being built, and a reader of a failed build should not
have to know which of the two it was. See :data:`_ERA_WARNING_FLAGS`.

The sixth is the same reading for the *other* compiler this toolchain installs.
``gfortran`` is in the layer for the same reason ``g++`` is (``SYSTEM_PACKAGES``
unions :data:`SDIST_SYSTEM_PACKAGES`, whose ``scipy``/``scikit-learn``/``statsmodels``
entries name it), and its defaults are the same kind of problem — one diagnostic that
was a warning in every release before GCC 10 and is an error in GCC 14 — so its answer
sits beside the C one rather than in a separate mechanism. See
:data:`_ERA_FORTRAN_FLAGS`.
"""

BUILD_ENV_RECORD = "/opt/vanta-build-env.txt"
"""Where a layer records the compiler environment its own builds run under.

Written by the same script that exports it, beside the interpreter layer's
``/opt/vanta-system-packages.txt`` and every layer's ``/opt/vanta-deps-freeze.txt``,
and printed into the bake log: the override above is what makes a source build work,
so what it actually was has to be a fact about the layer rather than a line in a
recipe nobody can read after the fact.
"""

BUILD_ENV_HEADER = f"vanta build env ({BUILD_ENV_RECORD}):"
BUILD_ENV_FOOTER = "vanta build env end"
"""The two lines that frame the record in the layer's own build log.

[VANTA-WHY] The record is echoed *inside* the build and framed, rather than read
back later with a probe that boots the layer again: the interpreter layer already
boots twice to verify its marker, and a third boot per layer would be paid twelve
times for a file the build itself just wrote. The frame is what makes the echo a
fact rather than a filter: ``run_bake_actions`` re-prints exactly the lines between
these two sentinels, so what appears in the bake log is the layer's own record —
not a guess assembled out of whatever the build happened to print. Exact match, not
a substring, because ``set -x`` echoes the ``echo`` that prints them.
"""

_SYSCONFIG_COMPILER_KEYS = ("CC", "CXX", "LDSHARED", "LDCXXSHARED")


def build_env_for(minor: Optional[str]) -> Mapping[str, str]:
    """The compiler environment a ``py<minor>`` layer's builds run with, or ``{}``.

    Keyed on the same fact as the toolchain itself: :func:`system_package_lines`
    installs ``g++`` (and ``gcc`` with it) only for the minors in
    :data:`SYSTEM_PACKAGES`, so a minor without them has no compiler to point these
    at and its recipe stays exactly as it was baked.
    """
    return COMPILER_ENV if minor in SYSTEM_PACKAGES else {}


def _sysconfig_probe() -> str:
    """A ``python -c`` that prints the interpreter's *own* recorded compiler settings.

    [VANTA-WHY] Printed rather than only overridden, because this is the evidence for
    why the overrides exist: a source build that dies on ``[Errno 2]`` is otherwise a
    mystery until someone boots the layer and looks. ``sysconfig`` reports what the
    interpreter was *built* with, not ``$CC``, so the line still answers "what did the
    standalone record" after the exports below.
    """
    prints = "; ".join(
        f"print('recorded {key}:', sysconfig.get_config_var({key!r}))"
        for key in _SYSCONFIG_COMPILER_KEYS
    )
    return f'python -c "import sysconfig; {prints}"'


def _build_env_assert(env: Mapping[str, str]) -> str:
    """A shell check that every variable was really set in the layer's shell.

    [VANTA-WHY] ``export`` in a recipe is not the same fact as the variable being set
    in the shell that runs the build, and getting it wrong reproduces the failure this
    environment exists to prevent — so it fails here, by name, rather than twenty
    minutes into a source build with an ``[Errno 2]`` that names no variable.
    """
    tests = " && ".join(f'[ -n "${name}" ]' for name in env)
    return (
        tests
        + " || { echo 'the compiler environment ("
        + ", ".join(env)
        + ") did not reach this layer; a source build would run the compiler the "
        "interpreter recorded for the machine it was built on' >&2; exit 1; }"
    )


def build_env_lines(env: Mapping[str, str]) -> list[str]:
    """Probe, export, check and record one layer's compiler environment.

    Empty when ``env`` is empty, so a layer whose minor has no toolchain keeps its
    recipe line for line (:func:`build_env_for`).

    The record is the variables *and* the compiler's own version *and* the fact that
    this layer's VM had a semaphore filesystem. The variables say which flags were
    asked for; the version says which compiler read them; and :data:`DEV_SHM_ENV`
    says the layer's ``multiprocessing`` can make a semaphore at all — which is the
    one fact about a layer that no compiler flag can supply and whose absence shows up
    as a *test failure* rather than a build error (:data:`bugsinpy.shuru.DEV_SHM_PATH`).
    Anything that reaches ``$CC`` without going through the environment
    (``sysconfig``'s recorded value) is what :func:`_sysconfig_probe` prints.
    """
    if not env:
        return []
    # The start-up check's variable is recorded beside the compiler's: same file, same
    # shape, one reader for both (DEV_SHM_GUARD).
    pattern = "|".join([*env, DEV_SHM_ENV])
    return [
        _sysconfig_probe(),
        *[f"export {name}={shlex.quote(value)}" for name, value in env.items()],
        _build_env_assert(env),
        # Asserted, not merely filtered for: `env | grep` *drops* a variable that is
        # not set, so a layer whose boot never ran the start-up check would record a
        # file that silently omits the one line saying multiprocessing works — and the
        # reader of that file cannot tell "absent" from "not recorded".
        f'[ -n "${DEV_SHM_ENV}" ] || {{ echo "the VM has no {DEV_SHM_ENV}: its '
        "start-up check did not run, so CPython's multiprocessing cannot create a "
        "semaphore and every test using a Pool, a Queue or a Lock would fail\" >&2; "
        "exit 1; }",
        # `set -e` sees the last command in a pipeline, so this is the record rather
        # than the check — the check is the line above it.
        f"env | grep -E '^({pattern})=' | sort > {BUILD_ENV_RECORD}",
        # The version, kept in the same `NAME=value` shape as the exports because the
        # reader of this file only keeps lines of that shape
        # (:func:`layer_provenance_lines`): a bare `gcc --version` line inside the
        # block would be dropped, and a fact that is dropped is not a record. The
        # assignment is its own statement so `set -e` catches a `gcc` that cannot run
        # — a `printf "$(gcc …)"` would report the `printf`'s status and record an
        # empty version silently.
        'CC_VERSION="$(gcc --version | head -n 1)"',
        '[ -n "$CC_VERSION" ] || { echo "gcc --version printed nothing; the compiler '
        'this layer builds with cannot say what it is" >&2; exit 1; }',
        f'printf "CC_VERSION=%s\\n" "$CC_VERSION" >> {BUILD_ENV_RECORD}',
        # Echoed between the two sentinels so the layer's own record reaches the bake
        # log (BUILD_ENV_HEADER): what the exports actually were is the one fact a
        # reader of a failed source build cannot recover from the recipe.
        f"echo {shlex.quote(BUILD_ENV_HEADER)}",
        f"cat {BUILD_ENV_RECORD}",
        f"echo {shlex.quote(BUILD_ENV_FOOTER)}",
    ]


_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
"""One ``NAME=value`` line of a build-environment record."""

_SMOKE_PREFIX = "toolchain smoke ok"
"""What the interpreter layer prints for each smoke check that passed: both extensions
built and imported, and (with :data:`_ERA_FORTRAN_FLAGS` reaching it) a fixed-form
Fortran rank mismatch compiled. One prefix for both, because the reader of a layer's log
(:func:`layer_provenance_lines`) keeps the smoke lines by it and a second spelling would
be a fact the reader drops."""


def layer_provenance_lines(output: str) -> list[str]:
    """The compiler-environment record and the toolchain smoke result in one log.

    ``output`` is a layer's own build output, so these are facts about *that* layer:
    the record is the file the build wrote inside it (:data:`BUILD_ENV_RECORD`), and
    the smoke lines are the interpreter layer's C, C++ and Fortran checks having
    passed.

    Three rules keep this from turning noise into provenance. The block is read only
    between the exact sentinels, because ``set -x`` echoes the commands that print
    them. Inside the block only ``NAME=value`` lines are kept, because the trace of
    the ``cat`` itself is a line in the same stream. The smoke lines are matched at
    the *start* of the stripped line, because the shell's trace of the command that
    prints each one contains the same words — measured shape of a log where the trace
    and the output are interleaved.
    """
    inside = False
    out: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if line == BUILD_ENV_HEADER:
            inside = True
            out.append(line)
            continue
        if line == BUILD_ENV_FOOTER:
            out.append(line)
            inside = False
            continue
        if inside:
            if _ENV_LINE_RE.match(line):
                out.append(line)
        elif line.startswith(_SMOKE_PREFIX):
            out.append(line)
    return out


_TOOLCHAIN_SMOKE_DIR = "/tmp/vanta-toolchain-smoke"
"""A scratch directory the interpreter layer builds a throwaway extension in."""

_TOOLCHAIN_SMOKE_C_SOURCE = """\
#include <Python.h>

/* [VANTA-WHY] The era's warning semantics, exercised rather than assumed. Assigning
   a `double *` to an `int *` is one of the six diagnostics GCC 14 made an error and
   the GCC these 2020 sources were written for printed as
   `warning: assignment from incompatible pointer type`. This file therefore compiles
   only if the flags in CFLAGS reached the compiler, which is the fact the smoke check
   exists to establish -- every later source build in this layer (numpy, then pandas)
   depends on the same flags having reached the same distutils path. */

static int vanta_era_pointer_check(void) {
    double *numbers = NULL;
    int *wrong;
    wrong = numbers;
    return wrong == NULL;
}

static PyObject *vanta_value(PyObject *self, PyObject *args) {
    return PyLong_FromLong(1);
}

static PyMethodDef vanta_methods[] = {
    {"value", vanta_value, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef vanta_module = {
    PyModuleDef_HEAD_INIT, "vanta_smoke_c", NULL, -1, vanta_methods,
};

PyMODINIT_FUNC PyInit_vanta_smoke_c(void) {
    if (!vanta_era_pointer_check()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "the layer's C flags did not reach the compiler");
        return NULL;
    }
    return PyModule_Create(&vanta_module);
}
"""

_TOOLCHAIN_SMOKE_CPP_SOURCE = """\
#include <Python.h>

static PyObject *vanta_value(PyObject *self, PyObject *args) {
    return PyLong_FromLong(2);
}

static PyMethodDef vanta_methods[] = {
    {"value", vanta_value, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef vanta_module = {
    PyModuleDef_HEAD_INIT, "vanta_smoke_cpp", NULL, -1, vanta_methods,
};

PyMODINIT_FUNC PyInit_vanta_smoke_cpp(void) {
    return PyModule_Create(&vanta_module);
}
"""

_TOOLCHAIN_SMOKE_F_SOURCE = """\
      subroutine vanta_rank_pair(scalar, vector)
c     [VANTA-WHY] The measured shape, in miniature: `dnaitr.f` calls
c     `dvout` twice, same arity, a different rank in one position --
c     `rnorm` (scalar) at 376 beside `xtemp` (rank-1) at 658 -- and
c     gfortran 14 refused the pair:
c       Error: Rank mismatch between actual argument at (1) and actual
c       argument at (2) (scalar and rank-1)
c     This file is that pair and nothing else: one implicit external
c     subroutine, called once with a scalar and once with a rank-1
c     array. `vanta_rank_consumer` is deliberately NOT defined here,
c     because the error is about a procedure the file has no interface
c     for -- a definition would hand gfortran the interface the ARPACK
c     sources do not have either -- and the file is fixed form, the
c     form that source is in (`-ffixed-form`, as its compile line
c     reads).
      integer scalar
      integer vector(2)
      call vanta_rank_consumer(scalar)
      call vanta_rank_consumer(vector)
      end
"""

_TOOLCHAIN_SMOKE_SETUP = """\
from setuptools import Extension, setup

setup(
    name="vanta-toolchain-smoke",
    version="0",
    ext_modules=[
        Extension("vanta_smoke_c", ["vanta_smoke_c.c"]),
        Extension("vanta_smoke_cpp", ["vanta_smoke_cpp.cpp"], language="c++"),
    ],
)
"""


def toolchain_smoke_lines(env: Mapping[str, str]) -> list[str]:
    """Build and import a tiny C **and** C++ extension, and compile a Fortran file.

    [VANTA-WHY] The failure the compiler environment prevents surfaces twenty minutes
    into a dependency's source build (numpy, then pandas). This is that build in
    miniature, on the layer where the toolchain is installed: ``setup.py build_ext``
    is the same ``distutils`` path that ran the recorded compiler, and the extension
    is imported afterwards so a build that produced an unloadable object file fails
    too. A broken toolchain therefore fails *this* layer's bake, before any per-bug
    layer is attempted, and one line of the log says so.

    The C++ half is not redundant: the measured trees compile C++ extensions
    (matplotlib's ``ft2font``, pandas' ``_libs``), and ``CC`` and ``CXX`` are two
    variables with two answers — a layer that can compile C and not C++ would pass a
    C-only smoke check.

    The C half also carries one diagnosis GCC 14 promoted to an error
    (:data:`_ERA_WARNING_FLAGS`), so this smoke fails on a layer where ``CFLAGS`` did
    not reach ``distutils``. That is the same class of failure as the compiler: a
    flag that is exported but never read is invisible until twenty minutes into
    lxml's C sources.

    The Fortran half is the same check for the other flags, and it is a *compile*
    rather than an extension build because the flags reach it by a different route:
    the trees' Fortran goes through ``numpy.distutils``, whose ``FCompiler`` appends
    an environment ``FFLAGS`` to its flag list (:data:`_ERA_FORTRAN_FLAGS`), while
    this layer's seeded flat ``distutils`` has no Fortran flag handling at all — so
    the smoke runs ``gfortran`` with the variable the build system reads, on a source
    carrying the rank mismatch ``pandas/1`` measured. ``gfortran`` is installed by
    this layer and not by the per-bug one (``SYSTEM_PACKAGES`` unions
    :data:`SDIST_SYSTEM_PACKAGES`), which is what makes it the layer to check.

    It runs under the interpreter layer's own **seeded** setuptools rather than a
    pinned toolchain: this layer's job is the interpreter and the compiler, and the
    toolchain is pinned per bug (§3.2) because it is the tree's build that has a date.

    Empty when ``env`` is empty, i.e. for a minor with no toolchain to smoke-test: it
    is keyed on the same fact as the packages that install the compiler
    (:func:`build_env_for`), so a 3.6/3.7 layer keeps the recipe it was baked with
    rather than being rebuilt to fail on a compiler it never had.
    """
    if not env:
        return []
    directory = _TOOLCHAIN_SMOKE_DIR
    return [
        f"mkdir -p {directory}",
        f"cat > {directory}/vanta_smoke_c.c <<'VANTA_SMOKE_C'",
        _TOOLCHAIN_SMOKE_C_SOURCE.rstrip("\n"),
        "VANTA_SMOKE_C",
        f"cat > {directory}/vanta_smoke_cpp.cpp <<'VANTA_SMOKE_CPP'",
        _TOOLCHAIN_SMOKE_CPP_SOURCE.rstrip("\n"),
        "VANTA_SMOKE_CPP",
        f"cat > {directory}/vanta_smoke_rank.f <<'VANTA_SMOKE_F'",
        _TOOLCHAIN_SMOKE_F_SOURCE.rstrip("\n"),
        "VANTA_SMOKE_F",
        f"cat > {directory}/setup.py <<'VANTA_SMOKE_SETUP'",
        _TOOLCHAIN_SMOKE_SETUP.rstrip("\n"),
        "VANTA_SMOKE_SETUP",
        f'cd {directory} && python setup.py build_ext --inplace && python -c "import '
        "vanta_smoke_c, vanta_smoke_cpp; assert vanta_smoke_c.value() == 1 and "
        f"vanta_smoke_cpp.value() == 2; print('{_SMOKE_PREFIX}: C and C++ "
        'extensions build and import\')"',
        # Unquoted `$FFLAGS`, because a flag list is more than one word and this is
        # the position the build system puts it in. `-ffixed-form` as the failing
        # command reads, so the smoke compiles the form that failed. No `|| true`:
        # an empty or unexported `FFLAGS` makes this file an `Error:` again, which
        # is the failure being checked for.
        f"cd {directory} && gfortran $FFLAGS -ffixed-form -c vanta_smoke_rank.f "
        f"-o vanta_smoke_rank.o && echo '{_SMOKE_PREFIX}: a Fortran rank mismatch "
        "compiles under FFLAGS'",
        # [VANTA-WHY] The kernel, not the compiler — and here rather than only at boot
        # because this is the layer whose record a reader consults afterwards. A VM
        # whose multiprocessing cannot create a semaphore makes every ``Pool``,
        # ``Queue`` and ``Lock`` in a later wave fail as a *test* failure, which reads
        # as a model error; ``DEV_SHM_GUARD`` runs the same check at every boot, and
        # this is the layer's own receipt that it passed on the interpreter the
        # suites will use (``python``, i.e. this layer's venv wrapper, not the base
        # image's 3.13).
        f"python -c {shlex.quote(MP_SMOKE_CODE)}",
        f'echo "vanta {MP_SMOKE_PREFIX}: Lock and a 2-worker Pool"',
    ]


def python_recipe(minor: str, pin: PytestPin) -> str:
    """Install a standalone CPython and the pinned pytest, then assert both work."""
    venv = f"/opt/venv{minor.replace('.', '')}"
    build_env = build_env_for(minor)
    return "\n".join(
        [
            "set -eux",
            *system_package_lines(SYSTEM_PACKAGES.get(minor, ())),
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
            # The start-up check, in the recipe as well as at every boot
            # (DEV_SHM_GUARD). The boot already ran it on this VM; running it here is
            # what puts the fact *in this layer's own log* and makes the export the
            # next line records a fact about the layer rather than about the wrapper
            # that happened to launch the build. Idempotent: it re-reads /proc/mounts
            # before mounting.
            *startup_check_lines(),
            # The compiler environment, set on the interpreter layer because that is
            # where the toolchain is installed and because every source build *above*
            # this layer (the smoke check below, then numpy, then pandas) inherits the
            # standalone's recorded compiler otherwise (COMPILER_ENV). Placed after the
            # wrapper exists, so the sysconfig probe reports *this* interpreter's
            # settings rather than the base image's 3.13.
            *build_env_lines(build_env),
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
            # The toolchain, exercised: a layer whose compiler is installed but whose
            # environment is wrong fails here, by name, instead of twenty minutes into
            # numpy (toolchain_smoke_lines). Before the marker, so a smoke failure makes
            # `checkpoint_create` delete the layer rather than cache it.
            *toolchain_smoke_lines(build_env),
            f"python -m pip freeze > {DEPS_FREEZE_OUTPUT}",
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
def _dict_literal_items(node: ast.expr) -> Optional[dict[str, ast.expr]]:
    """A dict literal's ``{str: value}`` items, or ``None`` if it is not a dict.

    Only constant string keys are read: a computed key (``"install_" + "requires"``)
    would have to be evaluated, and nothing here executes the project's code.
    """
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out[key.value] = value
    return out


def _setup_kwargs(setup_py: str) -> dict[str, ast.expr]:
    """``setup(...)``'s keyword arguments, as AST nodes.

    [VANTA-WHY] The project's ``setup.py`` is not run, on the host or anywhere else
    (design C3: nothing third-party executes outside Shuru, and "just asking
    setuptools what it wants" is executing it). It is Python; it can do anything.
    Reading it with ``ast`` is the only safe way to learn what it declares, and what
    ``ast`` cannot see — a computed keyword name, a dict built by a function — is
    recorded as a note instead of guessed at.

    A ``**kwargs`` splat is resolved too, because a project that assembles its
    metadata in a module-level dict has still written it down statically: sanic's
    ``setup.py`` opens ``setup_kwargs = {"name": "sanic", "packages": ["sanic"], …}``,
    appends ``setup_kwargs["install_requires"] = requirements`` and friends by
    subscript, and ends with ``setup(**setup_kwargs)``. Measured: without this the
    reader saw no ``name=`` and no ``packages=[...]`` and flagged sanic/1, whose
    metadata is all present. The literal's own keys and the subscript assignments are
    merged in source order, still by reading the AST — a splat whose name does not
    resolve to a dict literal contributes nothing, exactly as before.
    """
    try:
        tree = ast.parse(setup_py)
    except SyntaxError as exc:
        raise BakeError(f"setup.py does not parse as Python: {exc}") from exc

    assigned: dict[str, ast.expr] = {}
    subscripted: dict[str, dict[str, ast.expr]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned[target.id] = node.value
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                subscripted.setdefault(target.value.id, {})[target.slice.value] = node.value

    def deref(node: ast.expr) -> ast.expr:
        if isinstance(node, ast.Name) and node.id in assigned:
            return assigned[node.id]
        return node

    def splat_items(name: str) -> dict[str, ast.expr]:
        items: dict[str, ast.expr] = {}
        literal = assigned.get(name)
        if literal is not None:
            parsed = _dict_literal_items(literal)
            if parsed is not None:
                items.update(parsed)
        items.update(subscripted.get(name, {}))
        return {key: deref(value) for key, value in items.items()}

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
                if isinstance(keyword.value, ast.Name):
                    kwargs.update(splat_items(keyword.value.id))
                continue
            kwargs[keyword.arg] = deref(keyword.value)
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


def build_refs(
    setup_py: Optional[str],
    pyproject: Optional[str],
    notes: list[str],
    *,
    strict: bool = True,
) -> list[DepRef]:
    """The project's **build** requirements → refs, read statically, never installed as deps.

    PEP 518's ``[build-system].requires`` and setuptools' ``setup_requires`` (:data:`BUILD_REQUIRES_SOURCES`).
    Read for one reason: :func:`build_pins` has to know *which* tools to pin, and
    "numpy when it is a build requirement" is a question only the project's own
    metadata answers — pandas builds ``pandas/_libs`` against numpy and says so in
    both places, matplotlib says ``setup_requires=["numpy>=1.15"]``.

    Nothing here goes into the dependency install: these are the tools the tree is
    *built* with, and a build requirement that is also a runtime dependency is
    already pinned by the dependency set. Nothing here is executed either —
    ``setup.py`` is read with ``ast`` like every other source (§3.2) and the
    ``pyproject.toml`` is parsed as TOML.

    The declared constraint is not carried into the pin; only the *name* is read.
    The version comes from :data:`BUILD_PIN_RULE`, which is the point of the rule:
    pandas' build declares ``numpy==1.15.4`` because that is what *its own CI* used,
    and installing that in the layer would be a second source build of a numpy the
    dependency set then replaces.

    ``strict`` is what the two callers differ on, and the difference is whose file is
    being read. For the *tree* it is ``True``: an unparseable ``setup.py`` is this
    bug's own metadata and a :class:`BakeError` is the flag it deserves. For a
    **pinned release** (:func:`pypi_build_requirements`) it is ``False``: a 2020 sdist
    can carry Python 2 syntax that ``ast`` refuses, and that is not this bug's fault or
    a fact about the bug — the refusal is recorded as a note and the ``pyproject.toml``
    half is still read.
    """
    refs: list[DepRef] = []
    if setup_py is not None:
        try:
            kwargs = _setup_kwargs(setup_py)
        except BakeError:
            if strict:
                raise
            notes.append(
                "setup.py does not parse as Python, so its setup_requires could not "
                "be read; this is a reading, not a fact about the release"
            )
            kwargs = {}
        node = kwargs.get("setup_requires")
        if node is not None:
            for literal in _literal_nodes(node):
                text = literal.value.strip()
                if not text or text.startswith("#"):
                    continue
                refs.append(
                    dep_ref(text, f"setup_requires (setup.py) line {literal.lineno}")
                )
    if pyproject is not None:
        build_system = _table(_pyproject_data(pyproject), "build-system")
        if build_system is not None:
            refs += _toml_requirement_array(
                build_system.get("requires"),
                source="build-system.requires (pyproject.toml)",
                notes=notes,
            )
    return refs


COMPILED_SOURCE_SUFFIXES = (".pyx", ".c", ".cpp", ".cc", ".cxx")
"""What a tree has of its own to compile: Cython sources and C/C++ sources.

Suffixes and never directories: matplotlib's extension sources sit in ``extern/`` (the
vendored Agg, 63 files) and ``src/`` (17) alike, and both are compiled into the
package. A directory rule would have to guess which vendored directories count.
``.pxd`` is deliberately absent — a Cython *declaration* is not compiled.
"""


def needs_build(
    files: Sequence[str], setup_py: Optional[str], *, pythonpath: Optional[str] = None
) -> tuple[bool, str]:
    """Whether this tree must be built before its tests can import it, and **why**.

    [VANTA-WHY] Measured, not assumed. The dev split declared ``needs_build: false``
    for matplotlib/1 and pandas/1, whose tests import a package the tree itself has to
    compile — ``pandas/tests/dtypes/test_dtypes.py`` reaches ``pandas._libs``, and
    matplotlib's tests reach ``matplotlib._path`` — so the first thing that would have
    happened is the dependency gate's import probe failing with "does not import from
    the tree". That reads like a missing dependency and is not one, which is exactly
    the confusion §3.3 exists to prevent, in the other direction.

    The rule has two independent triggers, and both are properties of the tree rather
    than judgements about it:

    1. **BugsInPy says the import lives in a build directory** — it sets ``pythonpath``
       *inside* the tree (``tqdm/build/lib/``, ``cookiecutter/build/lib/``). The
       package its own ``run_test.sh`` imports only exists after a build, so this
       settles it whatever the tree ships.
    2. **The tree compiles extensions of its own.** ``setup(...)`` declares
       ``ext_modules`` *and* the tree carries sources to compile
       (:data:`COMPILED_SOURCE_SUFFIXES`). Both halves are load-bearing, and each
       excludes a measured shape: black declares ``ext_modules`` and has nothing to
       compile — its extension is the opt-in ``mypyc`` build of its own Python, off
       unless ``BLACK_USE_MYPYC`` is set, so its tests import pure Python; tornado
       ships ``tornado/speedups.c`` and declares no ``ext_modules``, because since 5.0
       that file is a stub kept for old compiled installs and is not built.

    The *reason* comes back with the answer, in both directions: the split carries a
    lone boolean, and a boolean that disagrees with its tree has to be able to say
    which half of the rule it disagrees with. ``prepare`` compares the two both ways
    and flags the bug rather than building a tree the split did not declare.
    """
    if pythonpath:
        return (
            True,
            f"BugsInPy sets pythonpath={pythonpath!r}: the suite imports the package "
            "from inside a build directory, so the tree has to be built first",
        )
    sources = sorted(path for path in files if path.endswith(COMPILED_SOURCE_SUFFIXES))
    declared = setup_py is not None and "ext_modules" in _setup_kwargs(setup_py)
    if declared and sources:
        sample = ", ".join(sources[:3]) + (", …" if len(sources) > 3 else "")
        return (
            True,
            f"setup.py declares ext_modules and the tree carries {len(sources)} "
            f"extension source(s) of its own ({sample}) for the tests to import",
        )
    if declared:
        return (
            False,
            "setup.py declares ext_modules but the tree carries no extension sources of "
            "its own, so nothing is compiled in-tree (black's mypyc extension is opt-in "
            "and compiles Python, not C)",
        )
    if sources:
        return (
            False,
            f"the tree carries {len(sources)} extension source(s) but setup.py declares "
            "no ext_modules, so nothing builds them (tornado's speedups.c is a stub)",
        )
    return (
        False,
        "setup.py declares no ext_modules and the tree carries no extension sources",
    )


def needs_build_at(
    repo: Path, commit: str, *, pythonpath: Optional[str] = None
) -> tuple[bool, str]:
    """:func:`needs_build` for a project's tree at ``commit``, read out of the clone.

    [VANTA-WHY] Read through ``git`` rather than from a checked-out tree, for the
    reason :func:`trees.read_file_at` exists: both callers need the answer *before*
    any tree is materialised — ``prepare`` as an early gate, the authoring script that
    writes the split before anything is materialised at all — and a checkout would be
    a second materialisation of the same commit that could disagree with the one the
    suites run.
    """
    files = tree_mod.list_files_at(repo, commit)
    setup_py = tree_mod.read_file_at(repo, commit, "setup.py")
    return needs_build(files, setup_py, pythonpath=pythonpath)


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

    ``module`` is the import spelling and may be dotted; the reviewed table is keyed by
    that spelling, so ``google.protobuf`` is matched before its top-level ``google`` —
    which is a distribution of its own on PyPI, and would otherwise be installed in
    protobuf's place. The fallback is always the *top-level* word, because
    ``import foo.bar`` is provided by distribution ``foo``, never by ``foo-bar``.
    """
    top = module.split(".")[0]
    for spelling in (module, top):
        reviewed = distributions.get(spelling)
        if reviewed:
            return canonicalize_name(reviewed)
    if top in AMBIGUOUS_MODULE_DISTRIBUTIONS:
        raise BakeError(
            f"the test side imports {module!r}, whose distribution is not named by its "
            "module, so the harness will not guess at it. Record the reviewed mapping "
            f"in the split's module_distributions ({module!r}: <distribution>) and bake "
            "again (design §3.2)."
        )
    return canonicalize_name(top)


def _can_name_a_distribution(module: str) -> bool:
    """Whether ``module`` could name a PyPI distribution at all.

    [VANTA-WHY] A PEP 508 name begins with a letter or a digit, so a module whose
    canonicalised spelling is not a valid one cannot be published under *any* name a
    merge could resolve: ``_csv`` canonicalises to ``-csv``. Those names are CPython's
    own private modules — ``_csv`` is the ``csv`` accelerator, which ships with every
    3.8 build — and they are the interpreter's, not the layer's. Measured: pandas/1's
    ``from _csv import Error`` reached the merge as ``-csv`` and stopped that bug over a
    name no one can install, rather than being recognised as the interpreter's.
    """
    try:
        Requirement(canonicalize_name(module))
    except InvalidRequirement:
        return False
    return True


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

TYPE_CHECKING_ONLY = "type-checking only"
"""Why an import cannot count as a runtime import: the block around it never runs.

[VANTA-WHY] An import inside ``if TYPE_CHECKING:`` is not an import the module makes.
``TYPE_CHECKING`` is ``False`` at run time, so the block is read by a type checker and
never executed — the distribution can be missing and the module still imports, which is
the whole point of writing it that way. Measured on pandas/1, this is the flag that
survived every other rule: ``pandas/core/arrays/boolean.py:31-32`` is
``if TYPE_CHECKING:`` / ``import pyarrow  # noqa: F401``, and the failing test
(``pandas/tests/dtypes/test_dtypes.py``) never reaches a runtime ``import pyarrow``, so
counting the block as evidence that the layer needs ``pyarrow`` refused a bug whose
graded tests pass without it. ``if False:`` is the same statement written without the
import from ``typing``."""

_LAZY_PREFIX = "lazy: inside "
"""The reason a function-scope import is recorded with, prefixed to its qualname.

[VANTA-WHY] An import inside a function or method body runs when that function is
called, not when the module is imported, so importing the module does not need the
distribution. That is not a leap of faith: ``prepare``'s gate is the empirical check —
if a graded test does call the function, the gate fails and the bug is reported with the
dropped pins named (:func:`prepare.dropped_pin_note`), which is a stop rather than a
silently short layer."""

_TYPE_CHECKING_NAMES = frozenset({"TYPE_CHECKING"})
"""The spellings :func:`_never_runs` treats as ``typing.TYPE_CHECKING`` by default."""


@dataclass(frozen=True)
class ImportSite:
    """One place in the bug's trees that names a module, and how it names it."""

    path: str
    line: int
    module: str
    optional: Optional[str]
    """Why reaching this import is not required, or ``None`` when it always runs."""
    commit: str = ""
    """The commit the file was read at, when the walk that found it knew one.

    Defaulted, because a caller that only wants the module names
    (:func:`test_import_closure`) reads one tree and has no second commit to name.
    """

    @property
    def spelling(self) -> str:
        """``path:line``, and the commit — where a reader goes to check the claim."""
        where = f"{self.path}:{self.line}"
        return f"{where} at {self.commit[:12]}" if self.commit else where

    @property
    def rendering(self) -> str:
        return f"{self.spelling} imports {self.module} — {self.optional or 'unconditional'}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "module": self.module,
            "optional": self.optional,
            "commit": self.commit or None,
        }


_OPTIONAL_CALLS: Mapping[str, str] = {
    "importorskip": "behind pytest.importorskip",
    "import_optional_dependency": "through import_optional_dependency",
}
"""Calls that name a module and mean "only if it is there".

[VANTA-WHY] A module a test only asks for through ``pytest.importorskip("x")`` is not
a module the layer has to carry: the test is written to be skipped without it. Measured,
this is most of pandas' 18 ``pyarrow`` mentions — the rest are ``try``/``except
ImportError`` guards, ``TYPE_CHECKING`` blocks and calls to pandas' own
``import_optional_dependency("pyarrow")`` — and reading any of them as "the tree
requires pyarrow" flags a bug whose failing test never goes near parquet.
``import_optional_dependency`` is the same statement in pandas' own vocabulary, and it
takes the module name as a string, so the call form has to be read rather than the
import statement.
"""


def _call_site(node: ast.Call) -> Optional[ImportSite]:
    """The site a guarded optional-import *call* names, or ``None`` for any other call."""
    func = node.func
    if isinstance(func, ast.Attribute):
        callee = func.attr
    elif isinstance(func, ast.Name):
        callee = func.id
    else:
        return None
    reason = _OPTIONAL_CALLS.get(callee)
    if reason is None or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    return ImportSite(path="", line=node.lineno, module=first.value, optional=reason)


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


def _type_checking_names(node: ast.AST) -> frozenset[str]:
    """The spellings of ``typing.TYPE_CHECKING`` the module under ``node`` binds.

    ``import typing`` and ``from typing import TYPE_CHECKING`` are both read by the
    ``TYPE_CHECKING`` name or attribute itself, so the only spelling this has to find is
    an **alias**: ``from typing import TYPE_CHECKING as TC`` leaves nothing called
    ``TYPE_CHECKING`` in the file, and the ``if TC:`` that follows is the same guard.
    The set is collected from the whole tree under ``node`` — the alias may be declared
    anywhere in the file — and always contains the bare name.
    """
    names = set(_TYPE_CHECKING_NAMES)
    for item in ast.walk(node):
        if isinstance(item, ast.ImportFrom) and not item.level and item.module == "typing":
            for alias in item.names:
                if alias.name == "TYPE_CHECKING":
                    names.add(alias.asname or alias.name)
    return frozenset(names)


def _never_runs(test: ast.AST, type_checking: frozenset[str] = _TYPE_CHECKING_NAMES) -> bool:
    """Whether an ``if`` with this test never runs its body, so its imports never run.

    The forms read here are the ones decidable from the test alone: a name bound to
    ``typing.TYPE_CHECKING`` (the bare name, an alias, the ``typing.TYPE_CHECKING``
    attribute) and the literal ``False``.

    [VANTA-WHY] The test is matched in its *exact* form, not as "any test that mentions
    the name". ``if not TYPE_CHECKING:`` and ``if x or TYPE_CHECKING:`` do run, so
    treating a mention as the guard would drop a pin a graded test actually needs.
    Anything else is left unguarded — the direction that keeps a pin rather than
    dropping one.
    """
    if isinstance(test, ast.Constant):
        return test.value is False
    if isinstance(test, ast.Name):
        return test.id in type_checking
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _scope_only_reason(reason: Optional[str]) -> bool:
    """Whether an optional reason answers the *audit's* question and not the layer's.

    [VANTA-WHY] ``type-checking only`` (:data:`TYPE_CHECKING_ONLY`) and
    ``lazy: inside <qualname>`` (:data:`_LAZY_PREFIX`) say "this import does not run when
    the module is imported". That is the question §3.2's unresolvable-pin rule asks — it is
    why a pin nothing on this image can install may be *dropped*
    (:func:`decide_unresolvable`) — and it is **not** a reason to leave an *installable*
    package out of a layer. Read as one it shrank the benchmark environment around the bug:
    measured on pandas/1, ``beautifulsoup4``, ``scikit-learn`` and ``psycopg2-binary``
    reach the layer through exactly such sites, so classifying them out of
    :func:`test_import_closure` stopped installing three packages the graded suite asked
    for, and the tests that skipped without them were the blind spot that hid it.

    So the two reasons are consulted by the audit walk (:func:`tree_import_sites` →
    :func:`import_auditor`) and by nothing else: :func:`test_import_closure` installs them
    like any other import, exactly as it did before the reasons existed.
    """
    return reason is not None and (
        reason == TYPE_CHECKING_ONLY or reason.startswith(_LAZY_PREFIX)
    )


def _collect_imports(
    node: ast.AST,
    *,
    optional: Optional[str],
    out: list[ImportSite],
    path: str = "",
    qualname: str = "",
    type_checking: Optional[frozenset[str]] = None,
) -> None:
    """Record every import reachable from ``node``, and whether it is optional.

    [VANTA-WHY] Optional propagates *into* a branch and never back out: an import
    inside ``try: import x  except ImportError:`` is a fallback, and so is one inside
    a function defined in that try. A relative import (``from . import helper``) is
    the tree's own file rather than a distribution, and is skipped — the absolute
    spelling of the same thing (``import tests``) is excluded by name instead, in
    :func:`test_import_closure`.

    Two structures make an import optional with no guard at all, and each is recorded
    with its own reason rather than as an unconditional site: a ``TYPE_CHECKING`` block
    (:data:`TYPE_CHECKING_ONLY`) and a function or method body (:data:`_LAZY_PREFIX`).
    Neither runs when the module is imported, which is the question this walk answers.

    ``path`` is the file being walked, recorded on every site so the rule that reads
    them (§3.2's, in :func:`tree_import_sites` and :func:`import_auditor`) can say
    *where* a module was named. It defaults to ``""`` so a caller that only wants the
    module names does not have to invent a filename. ``qualname`` is the enclosing
    ``Class.method``/``outer.inner`` the lazy reason is built from, and ``type_checking``
    is the file's spellings of ``TYPE_CHECKING``: both are derived from the module handed
    in on the first call and threaded down, because neither is a fact about one branch.
    """
    if type_checking is None:
        type_checking = _type_checking_names(node)
    if isinstance(node, ast.Import):
        for alias in node.names:
            out.append(ImportSite(path=path, line=node.lineno, module=alias.name, optional=optional))
        return
    if isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            out.append(
                ImportSite(path=path, line=node.lineno, module=node.module, optional=optional)
            )
        return
    if isinstance(node, ast.Call):
        site = _call_site(node)
        if site is not None:
            out.append(replace(site, path=path))
        # Not a return: the arguments are walked as usual, and a guarded call is not
        # the only thing a line can be.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        qual = f"{qualname}.{node.name}" if qualname else node.name
        body = {id(statement) for statement in node.body}
        for child in ast.iter_child_nodes(node):
            if id(child) not in body:
                # Only the body is lazy: a decorator and a default argument are
                # evaluated when the ``def`` runs, which is import time for a function
                # at module scope.
                _collect_imports(
                    child,
                    optional=optional,
                    out=out,
                    path=path,
                    qualname=qualname,
                    type_checking=type_checking,
                )
                continue
            # A reason that is already set is the nearer one (a guard, a version check,
            # a ``TYPE_CHECKING`` block) and stays; a nested function's own qualname is
            # the more specific answer to "where is this lazy".
            reason = optional
            if reason is None or reason.startswith(_LAZY_PREFIX):
                reason = f"{_LAZY_PREFIX}{qual}"
            _collect_imports(
                child,
                optional=reason,
                out=out,
                path=path,
                qualname=qual,
                type_checking=type_checking,
            )
        return
    if isinstance(node, ast.ClassDef):
        # A class body *does* run at import time, so nothing becomes optional on the
        # class's account; the name is carried only so a method's reason reads
        # ``Class.method`` rather than ``method``.
        qual = f"{qualname}.{node.name}" if qualname else node.name
        for child in ast.iter_child_nodes(node):
            _collect_imports(
                child,
                optional=optional,
                out=out,
                path=path,
                qualname=qual,
                type_checking=type_checking,
            )
        return
    if isinstance(node, ast.Try):
        inner = _guard_reason(node) or optional
        for branch in (*node.body, *node.handlers, *node.orelse, *node.finalbody):
            _collect_imports(
                branch,
                optional=inner,
                out=out,
                path=path,
                qualname=qualname,
                type_checking=type_checking,
            )
        return
    if isinstance(node, ast.If):
        if _never_runs(node.test, type_checking):
            for branch in node.body:
                _collect_imports(
                    branch,
                    optional=TYPE_CHECKING_ONLY,
                    out=out,
                    path=path,
                    qualname=qualname,
                    type_checking=type_checking,
                )
            # The ``else`` of a ``TYPE_CHECKING`` ``if`` is the branch that *does* run
            # (``pyarrow = None``), so it keeps the reason the ``if`` came in with.
            for branch in node.orelse:
                _collect_imports(
                    branch,
                    optional=optional,
                    out=out,
                    path=path,
                    qualname=qualname,
                    type_checking=type_checking,
                )
            return
        guard = _names_in(node.test) & _VERSION_GUARDS
        inner = "behind a version check" if guard else optional
        for branch in (*node.body, *node.orelse):
            _collect_imports(
                branch,
                optional=inner,
                out=out,
                path=path,
                qualname=qualname,
                type_checking=type_checking,
            )
        return
    for child in ast.iter_child_nodes(node):
        _collect_imports(
            child,
            optional=optional,
            out=out,
            path=path,
            qualname=qualname,
            type_checking=type_checking,
        )


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
    ``sys.path`` — and so are the ``.py`` files *directly* in a directory that is not a
    package, for the same reason: pytest inserts ``lib/`` for
    ``lib/matplotlib/tests/test_basic.py``, which is how matplotlib's own
    ``lib/pylab.py`` is imported as ``pylab`` (measured: the closure otherwise emits a
    distribution named ``pylab``, which PyPI does not publish — a 404 — and the bug
    stops), and ``scripts/`` for pandas'
    ``scripts/tests/test_validate_docstrings.py``, which is how
    ``scripts/validate_docstrings.py`` is imported as ``validate_docstrings``.

    A ``.py`` file *inside a package* is deliberately not counted: ``fastapi/requests.py``
    is importable as ``fastapi.requests`` and never as a top-level ``requests``, and
    counting those stems would swallow the real distribution of that name.
    """
    paths = list(files)
    packages = {
        path.rsplit("/", 1)[0] if "/" in path else "" for path in paths if path.endswith("__init__.py")
    }
    names: set[str] = set()
    for path in paths:
        if not path.endswith(".py"):
            continue
        parts = path.split("/")
        names.update(parts[:-1])
        stem = parts[-1][:-3]
        directory = "/".join(parts[:-1])
        if stem != "__init__" and directory not in packages:
            names.add(stem)
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

    This walk is the **install set** and nothing else decides it: the two reasons that
    only answer the audit's question — ``type-checking only`` and ``lazy: inside
    <qualname>`` — are not reasons to leave a package out of a layer
    (:func:`_scope_only_reason`), so a site carrying one is installed like any other
    import. Only a *guard* (an ``except ImportError``, a version check, an
    ``importorskip``) keeps a module in ``optional`` and out of the layer.
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
        sites: list[ImportSite] = []
        _collect_imports(parsed, optional=None, out=sites, path=path)
        for site in sites:
            top = site.module.split(".")[0]
            if not top or top == "__future__":
                continue
            if top in stdlib_names or top == import_name or top in local:
                continue
            if not _can_name_a_distribution(top):
                notes.append(
                    f"import closure: {path} line {site.line}: {top!r} cannot name a "
                    "distribution, so it is the interpreter's own module and nothing "
                    "is installed for it"
                )
                continue
            source = f"import closure: {path} line {site.line}"
            # ``type-checking only`` and ``lazy: inside <qualname>`` are the audit's
            # reasons, not this walk's: §3.2 consults them to decide whether a pin
            # nothing here can install may be dropped (:func:`decide_unresolvable`), and
            # they must not change which *installable* packages a layer carries — see
            # :func:`_scope_only_reason`. A guard (``except ImportError``, a version
            # check, ``importorskip``) is still this walk's reason, as it always was.
            if site.optional is not None and not _scope_only_reason(site.optional):
                if (top, source) not in seen_optional:
                    seen_optional.add((top, source))
                    optional.append(
                        OptionalImport(module=top, source=source, reason=site.optional)
                    )
                continue
            distribution = module_distribution(site.module, distributions=reviewed)
            if (distribution, path) in seen:
                continue
            seen.add((distribution, path))
            refs.append(DepRef(name=distribution, spec=top, source=source))
    return refs, optional, notes


def requirements_recipe(
    install_specs: Sequence[str],
    *,
    build_pins: Sequence[str] = (),
    early_pins: Sequence[str] = (),
    no_deps_pins: Sequence[str] = (),
    orphans: Sequence[str] = (),
    minor: Optional[str] = None,
) -> str:
    """Install one bug's *resolved* dependencies on top of the interpreter layer.

    Takes pins (:func:`resolve_deps`) rather than the project's own requirement
    strings, because a layer has to be a fact: the requirements are ranges
    (``decorator>=4.3.0``), and "whatever pip resolves today" is a different
    environment every time the layer is rebuilt. Nothing here is a guess — every spec
    was named by one of the five approved sources and pinned to the newest release on
    or before the date of the commit that declared it (a fix-only requirement is held
    to the *fix's* date, everything else to the buggy commit's).

    ``build_pins`` goes in **first**, in **two** installs, and the split between them
    is what makes the dependency install below run against a *dated* toolchain rather
    than today's:

    * the **toolchain** (:data:`BUILD_TOOLS`) alone, in one pip call, with pip's own
      build isolation — these three are the tools the isolation would otherwise
      supply, so they are the one install that cannot be made to depend on itself;
    * then **everything else** with ``--no-build-isolation``: the build requirements
      the project declares (numpy for matplotlib, Cython and numpy for pandas) and
      the dependency set;
    * and, between the two and the dependency set's own call, ``early_pins`` — the
      dependency pins that some pin in this layer declares as a *build* requirement
      (:func:`early_install_pins`), installed before the main list so the build that
      wants them finds them already there.

    [VANTA-WHY] ``early_pins`` is the third install because ``--no-build-isolation``
    makes setuptools resolve ``setup_requires`` itself, and with nothing installed that
    satisfies it ``easy_install`` reaches PyPI and installs **today's** release of it
    — which is the ``[VANTA-WHY]`` of :func:`pypi_build_requirements` seen from the
    build side, and the reason :func:`run_bake_actions` fails a layer that does it
    anyway. The pins are already in ``install_specs``; installing them first is what
    makes the difference, and pip installing an already-satisfied pin is a no-op, so
    they stay in the main list where the layer's record of what it installs is.

    [VANTA-WHY] The two-install split is the fix for a *measured* failure, and the
    single install it replaces looked correct in the log. matplotlib's ``setup.py``
    says ``setup_requires=["numpy>=1.15"]``, so numpy is a build pin and was written
    into the one pip call that keeps isolation: the log printed
    ``build pins: setuptools==47.1.1 …`` while pip built ``numpy==1.18.5`` in
    ``/tmp/pip-build-env-*/overlay`` with *today's* setuptools, and numpy died in
    ``numpy/distutils/mingw32ccompiler`` on a ``distutils`` module setuptools no
    longer carries — the exact failure the pins exist to prevent, one line above the
    assertion that said the pin had landed.

    ``--no-build-isolation`` is pip's supported way to say "build against what is
    already in this environment", which is exactly the pins just installed; pip
    refuses to build when a build requirement is missing from the environment rather
    than quietly isolating, so the flag either applies or the line fails. The recipe
    does not take that on trust either: after every non-isolated install it asserts
    that pip left **no** ``/tmp/pip-build-env-*`` directory behind
    (:func:`_no_isolated_build_env`), and every pin's version is asserted with
    ``importlib.metadata`` once both installs have run.

    ``minor`` is the interpreter minor this layer is built on, and it decides the
    **compiler environment** the installs below run under (:func:`build_env_for`). It
    is passed explicitly rather than inherited from the interpreter layer because a
    checkpoint is a filesystem: the ``export``ed ``CC``/``CXX`` the interpreter layer
    set are shell state and do not survive into a layer baked ``--from`` it. Without
    them the per-bug layer's source builds (matplotlib's and pandas' numpy) run the
    compiler the standalone recorded for its own build machine, which is the
    ``[Errno 2]`` this environment exists to prevent. A minor with no toolchain gets
    no exports, so such a layer's recipe is unchanged.

    Every install **after** the toolchain carries :data:`LEGACY_RESOLVER_FLAG`, and the
    toolchain call carries none because it is the install that puts
    :data:`PIP_PIN_VERSION` there (both spelled out above the recipe). Then, before the
    freeze, the layer runs ``pip check`` and echoes its output on
    :data:`PIP_CHECK_PREFIX`-marked lines: the legacy resolver tolerates a conflict
    between two of BugsInPy's own pins, and this is the reading that says which
    conflicts it actually installed (:func:`pip_check_conflicts`).

    Every install carries ``-c`` on one constraints file holding the layer's whole
    pin list (:data:`CONSTRAINTS_FILE`), written above the first of them. A pin list is only
    the layer's environment if every install in the layer is held to it: ``fastapi/1``
    measured the difference, a toolchain step that pulled today's ``requests`` for
    ``flit==2.3.0`` and left ``charset-normalizer`` in a layer whose pins name neither —
    which is what the freeze gate then refused. The file holds the same pins the installs
    hold, so it cannot change what a layer installs; it only stops a step resolving a
    dependency of a pin to a release no pin of this layer decided. It is also the reason
    ``no_deps_pins`` has to be its own pass and not a line left in the main one: a dropped
    name is in no constraints file either, so ``-c`` is silent about it.

    ``orphans`` is the one thing the recipe **removes**, and it goes here rather than in
    the pin list because it is the layer's own residue and not a dependency of anything in
    it (:data:`ORPHAN_RULE`): the interpreter layer's environment is inherited whole, and a
    ``pytest==5.4.3`` installed over its ``8.0.2`` leaves 8.0.2's ``exceptiongroup``,
    ``iniconfig`` and ``tomli`` behind. They are uninstalled by name after every install
    and before ``pip check`` — the check has to see the environment the layer ends up
    with, and every install below can pull a name back in. No ``|| true``: ``pip`` skipping
    a name that is not there is not a failure and anything else is, and the freeze gate is
    the second reading either way, since an orphan is exempted from nothing
    (:func:`unpinned_freeze`).

    ``build_pins`` is keyword-only and empty by default so that a caller with nothing
    to build — and every test of the dependency half — still gets a recipe, and the
    recipe says so rather than pretending to a toolchain it was not given.

    ``no_deps_pins`` is §3.2's dropped-dependency pass and the one subset of
    ``install_specs`` installed **separately**, with ``--no-deps``
    (:func:`dropped_dependents`): a pin whose own release declares a name the layer
    dropped, because leaving it in the main pass is what lets pip resolve that dropped
    name — at the newest release of the day, since the name is in no pin list and in no
    constraints file. Measured, ``sanic/1``'s ``pyopenssl==19.1.0`` declares
    ``cryptography (>=2.8)`` and the layer held ``cryptography 47.0.0`` for it. It runs
    **after** the main pass, so everything the pin declared is already installed (the
    closure walk pinned it) and ``--no-deps`` costs nothing but the dropped name, and
    **before** the orphan removal, which is the last thing before the reading.
    """
    no_deps = [spec.strip() for spec in no_deps_pins if spec.strip()]
    no_deps_set = set(no_deps)
    body = "\n".join(
        spec.strip()
        for spec in install_specs
        if spec.strip() and spec.strip() not in no_deps_set
    ).strip()
    toolchain, build_requires = _split_build_pins(build_pins)
    # The layer's whole pin list, written once and passed to every install below
    # (:data:`CONSTRAINTS_FILE`). One file and every install, because the failure it is
    # written against is exactly a *step* that resolved a name the layer's pins do not
    # contain: the toolchain's own `flit==2.3.0` pulled today's `requests` and left
    # `charset-normalizer` behind for the freeze gate to catch.
    constraints = constraint_lines(build_pins, early_pins, install_specs)
    held = f"-c {CONSTRAINTS_FILE} " if constraints else ""

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
        # Same start-up check as the interpreter layer's, and unconditional: a bug
        # layer with a 3.6/3.7 interpreter records no compiler environment
        # (:func:`build_env_for`) but its VM still needs a semaphore filesystem or its
        # suite fails on every ``Pool``/``Queue``/``Lock`` (DEV_SHM_GUARD).
        *startup_check_lines(),
        *build_env_lines(build_env_for(minor)),
        # Before the first install rather than beside the last: this is the one file
        # every install reads, and the toolchain step is one of the steps it holds.
        *constraints,
    ]
    if toolchain:
        lines += [
            "cat > /tmp/build-tools.txt <<'VANTA_BUILD_TOOLS'",
            "\n".join(toolchain),
            "VANTA_BUILD_TOOLS",
            # The toolchain itself, and the only install here that keeps isolation (and
            # the only one without the legacy-resolver flag): pip has to be able to
            # *build* a pinned tool that has no wheel for the platform, and at this
            # point there is nothing in the layer to build it against. It is also the
            # install that puts the pinned pip there, so the flag could not apply to it
            # even if it were otherwise safe. The constraints do apply to it: a build
            # tool is a distribution like any other, and its own requirements resolved
            # against the day's PyPI is the measured `fastapi/1` failure.
            f"python -m pip install --no-cache-dir {held}-r /tmp/build-tools.txt",
        ]
    if build_requires:
        lines += [
            "cat > /tmp/build-req.txt <<'VANTA_BUILD_REQUIRES'",
            "\n".join(build_requires),
            "VANTA_BUILD_REQUIRES",
            # The project's own build requirements, and *not* part of the isolated
            # call above: numpy is what failed there. Built against the toolchain
            # instead, so a source build sees the setuptools this tree was built
            # with rather than the one pip would fetch today.
            "python -m pip install --no-cache-dir "
            f"{held}--no-build-isolation {LEGACY_RESOLVER_FLAG} -r /tmp/build-req.txt",
            _no_isolated_build_env("the build requirements"),
        ]
    if toolchain or build_requires:
        # After both installs, because it asserts every pin — including the ones the
        # second install puts there. A pin that silently did not land is worse than
        # no pin: the build below would fall back on the interpreter layer's seeded
        # setuptools and look identical in the log.
        check = _build_pin_check(build_pins)
        if check:
            lines.append(check)
    early = [spec.strip() for spec in early_pins if spec.strip()]
    if early:
        lines += [
            "cat > /tmp/early-req.txt <<'VANTA_EARLY_REQUIRES'",
            "\n".join(early),
            "VANTA_EARLY_REQUIRES",
            # The dependency pins that another pin in this layer builds against,
            # installed before the main list so the build finds them: with
            # `--no-build-isolation` an unsatisfied setup_requires is fetched by
            # easy_install at today's version rather than the pinned one (§3.2). It
            # carries the flag for the same reason the main install does — it is a
            # dependency install and a dependency can itself be a source build.
            "python -m pip install --no-cache-dir "
            f"{held}--no-build-isolation {LEGACY_RESOLVER_FLAG} -r /tmp/early-req.txt",
            _no_isolated_build_env("the early build-requirement install"),
        ]
    if body:
        lines += [
            "cat > /tmp/req.txt <<'VANTA_REQUIREMENTS'",
            body,
            "VANTA_REQUIREMENTS",
            # No `|| true`: a bug whose dependencies do not install is a bug that
            # cannot be measured, and it must fail here rather than mid-wave. A
            # dependency pulled transitively can itself be a source build, which is
            # why this install carries the flag too.
            "python -m pip install --no-cache-dir "
            f"{held}--no-build-isolation {LEGACY_RESOLVER_FLAG} -r /tmp/req.txt",
            _no_isolated_build_env("the dependency install"),
        ]
    if no_deps:
        lines += [
            "cat > /tmp/no-deps.txt <<'VANTA_NO_DEPS'",
            "\n".join(no_deps),
            "VANTA_NO_DEPS",
            # §3.2's dropped-dependency pass, and the one install that must not resolve
            # anything: each of these pins declares a name the layer dropped, so pip's
            # own resolution would fetch it — at the newest release of the day, since a
            # dropped name is in no pin list and no constraints file. `--no-deps` stops
            # the resolution instead of accommodating it; the pins' other requirements
            # are already installed by the pass above, so nothing but the dropped name
            # is withheld, and `pip check` below is left to report exactly that
            # (:data:`CASE_DROPPED_DEPENDENCY`). Runs *after* the main pass so those
            # requirements are all there, and *before* the orphan removal so the
            # uninstalls it does are the last word on the environment.
            "python -m pip install --no-cache-dir "
            f"{held}--no-build-isolation {LEGACY_RESOLVER_FLAG} --no-deps "
            "-r /tmp/no-deps.txt",
            _no_isolated_build_env("the dropped-dependency install"),
        ]
    lines += [
        # The inheritance, undone: the interpreter layer's environment arrives whole
        # (`pip` does not uninstall what it replaces), so every name its *replaced*
        # harness pins brought in and this layer's pin list does not contain is removed
        # by name here (§3.2). After every install, because a pin below can pull one of
        # them back in, and before `pip check`, because the check has to read the
        # environment the layer actually ends up with.
        *orphan_removal_lines(orphans),
        # `pip check` reads what is *installed* and reports every requirement the
        # installed set does not satisfy, which is the only thing that can say whether
        # the legacy resolver left a contradiction behind. Its exit status is not the
        # verdict: a conflict between two of BugsInPy's own pins is tolerated (§3.2),
        # so the reading is taken here and judged on the host, where those names are
        # known (:func:`pip_check_conflicts`).
        f"python -m pip check > {PIP_CHECK_OUTPUT} 2>&1 || true",
        _pip_check_echo(),
        "python -m pytest --version",
        f"python -m pip freeze > {DEPS_FREEZE_OUTPUT}",
        _freeze_echo(),
        f"touch {MARKER}",
    ]
    return "\n".join(lines)


_BUILD_TOOL_NAMES = frozenset(
    canonicalize_name(name) for name in (*BUILD_TOOLS, PIP_TOOL)
)
"""The names the recipe's **first** install carries: the toolchain, plus ``pip`` itself.

``pip`` is in the toolchain call and not in the declared-build-requirements one
(:func:`_split_build_pins`) because that call is the first thing that runs, and the pip
pin has to be in place before any install that wants
:data:`LEGACY_RESOLVER_FLAG`.
"""


def _pip_check_echo() -> str:
    """One shell line that copies the layer's ``pip check`` output into the build log.

    The checkpoint holds :data:`PIP_CHECK_OUTPUT`, and nothing on the host can read it
    back without booting the layer (:data:`DEPS_DIRNAME`) — so the recipe reads it out
    as it goes, one :data:`PIP_CHECK_PREFIX`-marked line per ``pip check`` line. The
    marker is what the host filters on; the trailing ``print()`` is there because a
    ``pip check`` output that does not end in a newline would otherwise swallow the
    next line the recipe prints.
    """
    return (
        "python -c \"import sys; "
        f"sys.stdout.writelines('{PIP_CHECK_PREFIX} ' + line "
        f"for line in open('{PIP_CHECK_OUTPUT}')); print()\""
    )


def _freeze_echo() -> str:
    """One shell line that copies the layer's ``pip freeze`` into the build log.

    The same reading problem as :func:`_pip_check_echo`, one file over: the checkpoint
    holds :data:`DEPS_FREEZE_OUTPUT` and the host cannot open it, so the recipe echoes
    it out one :data:`PIP_FREEZE_PREFIX`-marked line per distribution. The judgement is
    the *diff* against the pin list, which only the host can take
    (:func:`unpinned_freeze`), so what the host needs from here is the reading and
    nothing else — hence one marker, one line, no opinion.
    """
    return (
        "python -c \"import sys; "
        f"sys.stdout.writelines('{PIP_FREEZE_PREFIX} ' + line "
        f"for line in open('{DEPS_FREEZE_OUTPUT}')); print()\""
    )


def orphan_removal_lines(orphans: Sequence[str]) -> list[str]:
    """The recipe lines that uninstall a bug layer's orphans, and say so in the log (§3.2).

    Empty when there is nothing to remove, which is every layer whose bug does not re-pin
    a harness package: the rule only fires on a layer that replaced one
    (:func:`transitive_pins`), so the ordinary recipe is unchanged and this is a step a
    reader sees only where it is warranted.

    The names go in by ``-r`` file rather than on the command line so that the line the log
    shows under ``set -x`` is the command and not the list, and each is echoed with
    :data:`ORPHAN_PREFIX` afterwards (:func:`_orphan_echo`) — the freeze below cannot show
    *why* a name the interpreter layer installed is absent from it, and this is the line
    that says.

    No ``--no-cache-dir`` — the step installs nothing — and no ``|| true``, for the reason
    the same flag is absent from the installs: ``pip`` skipping a name that is not there is
    a warning it exits ``0`` on, and anything else has to stop the layer
    (:data:`ORPHAN_RULE`).
    """
    names = [name for name in (spec.strip() for spec in orphans) if name]
    if not names:
        return []
    return [
        "cat > /tmp/orphans.txt <<'VANTA_ORPHANS'",
        "\n".join(names),
        "VANTA_ORPHANS",
        "python -m pip uninstall -y -r /tmp/orphans.txt",
        _orphan_echo(names),
    ]


def _orphan_echo(names: Sequence[str]) -> str:
    """One shell line saying, in the build log, which distributions were removed (§3.2).

    The same reading problem as :func:`_pip_check_echo`, without a file to read: the recipe
    already knows the names, and nothing on the host reads them back — the deps JSON is
    the record (:class:`OrphanedPin`) and the freeze gate is the verdict. This is for the
    reader of the log, one :data:`ORPHAN_PREFIX`-marked line per name, so that an
    uninstall a reader would otherwise have to infer from a missing freeze line is stated.
    """
    return (
        "python -c \"import sys; "
        f"sys.stdout.writelines('{ORPHAN_PREFIX} ' + name + chr(10) for name in "
        f"{sorted(names)!r})\""
    )


def _split_build_pins(build_pins: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(toolchain, declared build requirements)`` — the recipe's two installs.

    Split by **name** rather than by position, and the name wins whichever group it
    arrived in: a build requirement that is one of :data:`BUILD_TOOLS` (pandas
    declares Cython, matplotlib Cython and numpy) is installed by the toolchain call,
    because the second call is the one that has to run *against* the toolchain rather
    than fetch it. :data:`PIP_TOOL` is in the first group for a third reason — it is
    the pip the second call's :data:`LEGACY_RESOLVER_FLAG` needs, so it cannot be
    installed by that call. :func:`resolve_build_pins` already orders the toolchain
    first, so this only has to survive a caller that did not.
    """
    toolchain: list[str] = []
    build_requires: list[str] = []
    for spec in build_pins:
        stripped = spec.strip()
        if not stripped:
            continue
        name = canonicalize_name(stripped.partition("==")[0].strip())
        (toolchain if name in _BUILD_TOOL_NAMES else build_requires).append(stripped)
    return toolchain, build_requires


def _no_isolated_build_env(step: str) -> str:
    """One shell line that fails the recipe if pip left an isolated build env behind.

    [VANTA-WHY] ``/tmp/pip-build-env-*`` is pip's isolated build environment *made
    visible*, and ``--no-build-isolation`` is what stops one being created for
    ``step``. pip removes the directory on a normal exit, so this is a guard against
    the run that created one anyway — a failed or aborted isolated build — and not
    the main enforcement; but such a layer would be recorded as built, which is worse
    than a bake that stops. Written as a glob loop and not ``compgen -G``: the script
    runs under the image's ``/bin/sh`` (dash), where ``compgen`` does not exist.
    """
    return (
        "for d in /tmp/pip-build-env-*; do "
        '[ -e "$d" ] || continue; '
        f'echo "VANTA: {step} left an isolated build environment behind: $d"; '
        'ls -d "$d"; exit 1; '
        "done"
    )


def _build_pin_check(build_pins: Sequence[str]) -> str:
    """One line asserting that each build pin's version is the installed one.

    [VANTA-WHY] The failure this guards is not a wrong pin but an *absent* one: a
    toolchain that did not install (a yanked release, a name pip resolved elsewhere)
    leaves the dependency install to fall back on the interpreter layer's seeded
    setuptools, which is the exact environment the pins exist to replace — and pip
    reports nothing, because ``--no-build-isolation`` was satisfied. Read with
    ``importlib.metadata`` rather than ``import``: the pin names a *distribution*, and
    the module it provides is not always the same word (``cython`` → ``Cython``).
    """
    expected = []
    for pin in build_pins:
        name, _, version = pin.partition("==")
        if version:
            expected.append(f"({name!r}, {version!r})")
    if not expected:
        return ""
    return (
        'python -c "import importlib.metadata as md; '
        f"build_pins = [{', '.join(expected)}]; "
        "wrong = [(n, md.version(n), v) for n, v in build_pins if md.version(n) != v]; "
        "assert not wrong, wrong; "
        "print('build pins', build_pins)\""
    )


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
    build_refs: tuple[DepRef, ...] = ()
    """The project's declared **build** requirements, from both commits (:func:`build_refs`).

    Not dependencies and never installed as ones: they name the tools the tree is
    *built* with, which is what :func:`resolve_build_pins` turns into the layer's
    build pins. Read at the same two commits as the declarations, and stamped the
    same way, because a build requirement only the fix declares is held to the fix's
    date for the same reason a dependency is.
    """

    tree_modules: tuple[str, ...] = ()
    """The top-level modules the tree ships at the fixed commit (:func:`tree_provided_modules`).

    Read once, here, because two readers want the same answer and it is a walk of every
    file in the tree: the test side's own import closure, which must not resolve the
    tree's modules against PyPI, and §3.2's closure walk, which must not pin the tree's
    own distribution (:data:`TREE_SATISFIED_RULE`). Carrying it also means this record's
    consumers — the walk included — never touch the filesystem for it.
    """


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
    chosen_for_pytest: Optional[str] = None
    """The pytest pin this release was chosen to accept, when that is why it is here.

    [VANTA-WHY] ``pytest-json-report`` and ``pytest-metadata`` declare a *floor* on
    pytest, so a bug whose own pin is pytest 5.4.3 cannot carry the release the
    interpreter layer has. Which release was chosen is not a date rule, so it has to
    be recorded next to the pin: it is "the newest release whose declared pytest
    requirement accepts this pytest" (:func:`plugin_pins`), and without the pytest it
    accepts, the same pin would look arbitrary.
    """
    narrowed_by: tuple[tuple[str, str], ...] = ()
    """The exact pins whose own declared requirements narrowed this date-rule choice.

    [VANTA-WHY] §3.2's date rule answers "the newest release on or before this
    declaration's date", and that answer is not always installable: ``httpcore==0.3.0``
    declares ``h11==0.8.*``, so a layer holding both it and a date-picked ``h11`` is a
    layer pip refuses. When an exact pin in the same layer declares a requirement on
    this name, the newest release that satisfies *both* the declaration and the date is
    the answer, and each ``(pin, specifier)`` pair here says which pin narrowed it and
    with what — a departure from "newest before the date", so it is recorded the same
    way :attr:`overridden` is.
    """
    refused: tuple[tuple[str, str, str], ...] = ()
    """The releases the date rule would have picked and this layer's own pins refuse.

    [VANTA-WHY] :attr:`narrowed_by` is one direction of §3.2's case (a) — a pin in this
    layer declaring what release *this* name may be — and this is the other, which is
    what ``luigi/1`` measured: ``moto`` is a bare declaration, so nothing in the layer
    narrows *its* version, but ``moto==1.3.8`` declares ``idna (<2.9,>=2.5)`` while
    ``idna==2.9`` is BugsInPy's own pin, installed as written. The date rule's answer
    would be a layer whose own ``pip check`` reports case (a) and whose gate refuses it;
    the answer instead is the newest release on or before the date that the layer *can*
    hold — ``moto==1.3.7`` here, which declares nothing about ``idna``.

    Each entry is ``(rejected, held, declared)``: the release passed over, the pin in
    this layer whose version it does not satisfy, and what it requires of that name. The
    record therefore names both what was skipped and which pin skipped it, which is the
    whole departure from "newest before the date".
    """
    required_by: tuple[str, ...] = ()
    """The pins in this layer that declared this name, when nothing else did (§3.2).

    [VANTA-WHY] An empty tuple for a pin the tree declares — the ``sources`` and
    ``spec`` say that story. It is non-empty only for a *transitive* pick: a name
    ``pip`` would have resolved on its own, out of a release's own ``requires_dist``
    and against the wheel of the day. ``fastapi/1`` is what that costs — ``pytest``
    itself declares nothing about it and no pin in the layer names it, so the layer's
    environment is described by a list that does not mention what is in it. Which pin
    required it is the one thing a reader cannot recover from the pin, so it is
    recorded here, and :attr:`pin_rule` says the version came from the date rule.
    """

    @property
    def transitive(self) -> bool:
        """Whether this pin exists because another pin's declarations required it (§3.2)."""
        return bool(self.required_by)

    @property
    def pin_rule(self) -> str:
        """How the pinned version was chosen, in one phrase, for the provenance."""
        if self.harness:
            return "provisioned by the interpreter layer"
        if self.transitive:
            return TRANSITIVE_PIN_RULE
        if self.chosen_for_pytest:
            return (
                "the newest release whose declared pytest requirement accepts "
                f"pytest=={self.chosen_for_pytest}"
            )
        if self.exempt:
            return BUGSINPY_PIN_RULE
        if self.fixed:
            rule = "the newest release on or before the fixed commit's date"
        else:
            rule = "the newest release on or before the buggy commit's date"
        if not self.narrowed_by and not self.refused:
            return rule
        clauses: list[str] = []
        if self.narrowed_by:
            clauses.append(
                "that satisfies "
                + "; ".join(
                    f"{pin} declares {specifier}" for pin, specifier in self.narrowed_by
                )
            )
        if self.refused:
            clauses.append(
                "that this layer's own pins can hold, not "
                + "; ".join(
                    f"{rejected} declaring {declared}, which {held} does not satisfy"
                    for rejected, held, declared in self.refused
                )
            )
        return f"{rule} " + " and ".join(clauses)

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
        if self.required_by:
            data["required_by"] = list(self.required_by)
        if self.chosen_for_pytest:
            data["chosen_for_pytest"] = self.chosen_for_pytest
        if self.narrowed_by:
            data["narrowed_by"] = [
                {"pin": pin, "requires": specifier} for pin, specifier in self.narrowed_by
            ]
        if self.refused:
            data["refused"] = [
                {"pin": rejected, "held": held, "requires": declared}
                for rejected, held, declared in self.refused
            ]
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


def _specifier_accepts(specifier: SpecifierSet, version: str) -> bool:
    """Whether ``specifier`` accepts ``version`` — an empty specifier accepts any (§3.2).

    [VANTA-WHY] A requirement that names no version at all (``setuptools``) is satisfied
    by whatever the layer holds, which is what makes ``sanic/1``'s bare
    ``[build-system].requires`` entries ordering findings rather than problems. A clause
    is checked against the held version instead of only the name, because a name-only
    match records a satisfaction the build itself refuses: measured on ``luigi/1``,
    ``numpy==1.18.4``'s build needs ``Cython>=0.29.14`` and the layer holds
    ``cython==0.29.7`` — a recorded EARLY mapping whose ``satisfied_by`` does not satisfy
    it, and the build stops with ``RuntimeError: Building NumPy requires Cython >=
    0.29.14``.

    An *unparseable* held version is not a satisfaction: a clause that cannot be read
    against it is a question the build answers, and guessing yes here would record an
    ordering that does not hold.
    """
    if not str(specifier):
        return True
    return _parseable(version) and specifier.contains(version)


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


def _names_a_pre_release(specifier: SpecifierSet) -> bool:
    """Whether this requirement asks for a pre-release, which is what re-admits them.

    pip's own carve-out and the only one §3.2's date rule takes from it: a pre-release or
    a dev release is skipped unless the requirement *names* one — ``black==19.10b0`` does
    (a BugsInPy reference pin, installed as written), and so does a ``>=3.0.0a1`` bound,
    while ``pyparsing>=2.0.2`` does not. The direction matters: without the rule a range
    declaration gets the newest *listing*, and PyPI's listing holds pre-releases that sort
    above the release that was current — measured, ``matplotlib/1``'s ``pyparsing`` and
    ``jinja2`` and ``markupsafe`` all landed on an ``a1`` that pip would never have
    installed for those requirements; without the carve-out the benchmark's own
    pre-release pin could not be installed at all.

    A wildcard clause (``==1.0.*``) names no single version and is passed over, which is
    the conservative direction for a carve-out: it can only ever *deny* pre-releases.
    """
    for clause in specifier:
        try:
            if Version(clause.version).is_prerelease:
                return True
        except InvalidVersion:
            continue
    return False


def _date_rule_candidate(version: str, *specifiers: SpecifierSet) -> bool:
    """Whether the date rule may pick ``version`` for a requirement.

    ``specifiers`` are every requirement this layer holds for that name: the declaration's
    own (:func:`resolve_pins`), the union of the pins that declared it
    (:func:`_transitive_pin`), the refs that asked for a build tool
    (:func:`resolve_build_pins`) and, where this layer's own pinned releases constrain the
    same name, **theirs** — the ``requires_dist`` clause that narrows the choice is a
    requirement too, and a ``foo>=2.0a1`` in it is what makes a pre-release installable.
    A pre-release or dev release is refused unless one of them names one
    (:func:`_names_a_pre_release`); everything else is the caller's to filter, and the
    caller has already run :func:`_parseable`.
    """
    return not Version(version).is_prerelease or any(
        _names_a_pre_release(specifier) for specifier in specifiers
    )


def _newest_release(available: Mapping[str, str]) -> tuple[str, str]:
    """``(version, date)`` of the newest parseable release PyPI lists."""
    return max(
        ((version, date) for version, date in available.items() if _parseable(version)),
        key=lambda item: Version(item[0]),
        default=("?", "?"),
    )


def _settled_versions(
    refs: Sequence[DepRef], harness_pins: Mapping[str, str]
) -> dict[str, tuple[str, str]]:
    """``{name: (pin, version)}`` — the versions in this layer no date rule may move.

    Two kinds and neither is a date-rule choice: a BugsInPy reference pin
    (:func:`reference_pin`) is installed as written, and a package the interpreter layer
    provides (:data:`HARNESS_PACKAGES`) is never installed over — unless this bug writes
    its *own* exact pin for that same name, in which case the bug's pin wins and is the
    one recorded here (:func:`resolve_pins`' harness branch).

    [VANTA-WHY] This is the other half of §3.2's case (a), and it exists because the
    first half was not enough. ``_narrowing_requirements`` reads what the layer's exact
    pins declare *about* a name, which fixes a choice nothing else constrains — but
    ``luigi/1``'s ``moto`` is a bare declaration that nothing narrows, and its newest
    dated release (``1.3.8``) declares ``idna (<2.9,>=2.5)`` while ``idna==2.9`` is
    BugsInPy's own pin. Neither pin could move, the layer installed a contradiction, and
    the reading that found it was ``pip check`` *after* the build — case (a), refused.
    The settled set is what lets the choice be checked at pin time instead: a candidate
    that does not satisfy a settled pin is not an answer (:func:`_chosen_release`).
    """
    settled: dict[str, tuple[str, str]] = {
        canonicalize_name(name): (f"{name}=={version}", version)
        for name, version in harness_pins.items()
    }
    for ref in refs:
        if not reference_pin(ref):
            continue
        version = _exact_version(_specifier_of(ref))
        if version is not None:
            # After the harness entries, because the bug's own pin is installed *over*
            # the interpreter layer's version and is what the layer ends up holding.
            settled[canonicalize_name(ref.name)] = (f"{ref.name}=={version}", str(version))
    return settled


def _chosen_release(
    ref: DepRef,
    candidates: Sequence[tuple[str, str]],
    *,
    cutoff: str,
    settled: Mapping[str, tuple[str, str]],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> tuple[str, str, tuple[tuple[str, str, str], ...]]:
    """``(version, released_at, refused)`` — the newest candidate this layer can hold.

    [VANTA-WHY] §3.2's date rule asks for "the newest release on or before the
    declaration's date", and a date is not the only thing a layer has to satisfy: the
    pins the rule cannot move (:func:`_settled_versions`) are in the same environment,
    and a release whose own ``requires_dist`` does not satisfy one of them makes the
    layer a contradiction ``pip check`` reports as case (a) and the gate refuses. So the
    candidates are walked **newest first** and the first one the layer can hold is the
    answer, with every release passed over returned for the provenance
    (:attr:`ResolvedDep.refused`).

    A candidate whose declarations cannot be read is *taken*, not refused: an unreadable
    release is not evidence that it is incompatible — ``httpcore==0.3.0`` is read no
    other way — and the layer's own ``pip check`` is the backstop for it. That is the
    same conservative direction :func:`_narrowing_requirements` takes, and it is why this
    walk can only ever move a pin on a declaration that was actually read.

    The walk stops at the first answer, so an ordinary pin costs one metadata read — the
    newest candidate, which this layer's preflight reads anyway through the same cached
    reader — and a refusal costs one more.

    :class:`BakeError` when every candidate is refused: no release of this name is both
    on or before the declaration's date and installable beside the layer's own pins, and
    that is a decision for the split rather than something to install and discover
    (§3.2's case (a)).
    """
    refused: list[tuple[str, str, str]] = []
    for version, released_at in sorted(
        candidates, key=lambda item: Version(item[0]), reverse=True
    ):
        try:
            requires, _ = metadata_of(ref.name, str(version))
        except BakeError:
            return str(version), released_at, tuple(refused)
        conflict: Optional[tuple[str, str, str]] = None
        for raw in requires:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                continue
            if requirement.marker is not None:
                continue
            held = settled.get(canonicalize_name(requirement.name))
            if held is None:
                continue
            try:
                allowed = Version(held[1]) in requirement.specifier
            except InvalidVersion:
                continue
            if allowed:
                continue
            conflict = (
                f"{ref.name}=={version}",
                held[0],
                f"{requirement.name}{requirement.specifier}",
            )
            break
        if conflict is None:
            return str(version), released_at, tuple(refused)
        refused.append(conflict)
    raise BakeError(
        f"{ref.source}: {ref.spec} has no release on or before {cutoff} that this "
        "layer's own pins can hold — "
        + "; ".join(
            f"{rejected} declares {declared}, which {held} does not satisfy"
            for rejected, held, declared in refused
        )
        + ", and neither side is a version the date rule may move (§3.2)"
    )


def _narrowing_requirements(
    refs: Sequence[DepRef],
    *,
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> dict[str, list[tuple[str, str]]]:
    """``{name: [(pin, specifier)]}`` — what this layer's exact pins require of each name.

    [VANTA-WHY] The input to §3.2's case (a): only an **exact** pin can narrow a date-rule
    choice, because only an exact pin's own version is settled by the requirement itself
    (``httpcore==0.3.0``), and therefore only an exact pin's ``requires_dist`` can be read
    *before* the date rule runs. A range pin's version is not known yet, so it declares
    nothing that could be honoured here — and its own constraint is already in the merged
    refs. ``_exact_version`` is the same predicate the merge uses to decide what a claim
    can be settled on, so "an exact pin" means one thing in this file.

    A requirement carrying a marker is skipped: ``foo>=1; extra == "test"`` is only
    installed when the extra is asked for, and narrowing a layer's version on the word of
    a conditional requirement would pin something the bug's own environment never held.
    The conservative direction is deliberate — a conditional that *does* apply leaves a
    conflict for the ``pip check`` gate to report (:func:`pip_check_conflicts`) rather
    than a version chosen for a reason that cannot be checked.

    An exact pin whose own declarations cannot be read narrows nothing and is passed
    over, for the same conservative reason: what it asks for is unknown, and choosing a
    version for an unread requirement is a choice nothing can check. The layer's
    ``pip check`` still reads what pip installs, and the preflight still prints the pin
    as unknown (:data:`CASE_UNKNOWN`) — ``httpcore==0.3.0`` is unreadable from PyPI and
    is exactly this case, so its ``h11==0.8.*`` narrows nothing here and is caught after
    the build instead.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for ref in refs:
        version = _exact_version(_specifier_of(ref))
        if version is None:
            continue
        try:
            requires, _ = metadata_of(ref.name, str(version))
        except BakeError:
            continue
        for raw in requires:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                continue
            if requirement.marker is not None:
                continue
            text = str(requirement.specifier)
            if not text:
                continue
            name = canonicalize_name(requirement.name)
            pair = (f"{ref.name}=={version}", text)
            if pair not in out.setdefault(name, []):
                out[name].append(pair)
    return out


def resolve_pins(
    refs: Sequence[DepRef],
    *,
    releases: Mapping[str, Mapping[str, str]],
    commit_date: str,
    harness_pins: Mapping[str, str],
    buggy_commit: Optional[str] = None,
    metadata_of: Optional[
        Callable[[str, str], tuple[tuple[str, ...], Optional[str]]]
    ] = None,
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

    * A package the interpreter layer already provides, declared as a **range**. Its
      constraint is *checked* against the harness version and recorded; it is never
      installed over it, because a range names no release and the layer already
      carries one.
    * A BugsInPy reference pin (:func:`reference_pin`) — including one on a package
      the harness provides. The date rule is what stops a layer from installing a
      release the buggy tree could not have been built with, and it is the wrong rule
      for the benchmark's *own* frozen environment, which is what a pip-freeze-style
      ``requirements.txt`` is. Such a pin is installed as written, and its release
      date is recorded in the provenance so a late pin is visible rather than silent.
      ``pytest==5.4.3`` in fastapi/1's ``requirements.txt`` is this case: it is the
      environment that bug's ``run_test.sh`` was validated in, so it is installed by
      the per-bug layer rather than checked against the interpreter layer's default.
      When the pin also had to win over a project declaration the merge could not
      satisfy (:attr:`DepRef.overridden`), that decision is recorded with it rather
      than left implicit in the install list.

    For everything else the rule is unchanged, and it takes precedence over
    convenience: if the constraint has no release on or before the declaration's date,
    the bug cannot be rebuilt and the bake stops rather than installing today's
    version.

    "The newest release on or before the date" means the newest **release**, and a
    pre-release is not one: ``pip`` skips pre-releases and dev releases unless the
    requirement names one itself, and PyPI's listing is full of them sorting above the
    release that was current — measured, ``matplotlib/1``'s ``pyparsing>=2.0.2`` declaration
    took ``3.0.0a1``. So a candidate with a pre or dev segment is refused unless the
    requirement names one (:func:`_names_a_pre_release`), which is pip's own carve-out
    and is what keeps a BugsInPy reference pin like ``black==19.10b0`` installable. The
    narrowing below reads the same list this filter produced, so a narrowing claim can
    only ever take candidates away and never re-admit a pre-release (§3.2).

    One qualification on "the newest release on or before the date", and it is §3.2's
    case (a): when an **exact** pin elsewhere in this layer declares its own requirement
    on this name, the newest release the *date* allows is not the answer if it is one
    the other pin refuses — ``httpcore==0.3.0`` declares ``h11==0.8.*``, so a date-picked
    ``h11`` has to be one that satisfies both. The narrowing is read from the exact pins'
    ``requires_dist`` (``metadata_of``) and recorded
    (:attr:`ResolvedDep.narrowed_by`) *only when it changed the answer*, so an ordinary
    pin still reads as the plain date rule. ``metadata_of`` is optional for the same
    reason ``releases`` is an argument: a caller that passes none gets the date rule with
    nothing consulted, and every test of the rule itself stays offline.

    The same case, read the other way round, is the second qualification: the pins the
    date rule **cannot** move — BugsInPy's own ``==`` pins and the interpreter layer's
    harness pins (:func:`_settled_versions`) — are in the layer too, and a release that
    does not satisfy one of them is not an answer, however new it is (``moto==1.3.8``
    declares ``idna (<2.9,>=2.5)`` against ``idna==2.9``, so ``luigi/1`` pins ``1.3.7``
    and records why in :attr:`ResolvedDep.refused`). The two directions are one rule:
    the answer is the newest release on or before the declaration's date that this
    layer's other pins can hold.
    """
    narrowed = (
        _narrowing_requirements(refs, metadata_of=metadata_of)
        if metadata_of is not None
        else {}
    )
    settled = _settled_versions(refs, harness_pins) if metadata_of is not None else {}
    resolved: list[ResolvedDep] = []
    for ref in refs:
        requirement = _requirement(ref.spec, ref.source)
        cutoff = ref.commit_date or commit_date
        fixed = ref.commit is not None and ref.commit != buggy_commit
        if ref.name in HARNESS_PACKAGES and not reference_pin(ref):
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
                    "provides for a *range* — this bug needs a decision."
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
        claims = narrowed.get(canonicalize_name(ref.name), [])
        # Read here rather than at the narrowing filter below, because the carve-out has
        # to be known while the *pool* is built: a narrowing claim is a requirement on
        # this name like any other, and a ``foo>=2.0a1`` in one of them is what makes a
        # pre-release installable (§3.2). The pool itself is still the declaration's —
        # the claims only ever take candidates away, below.
        claim_specifiers = tuple(SpecifierSet(spec) for _, spec in claims if spec)
        matching = [
            (version, date)
            for version, date in available.items()
            if _parseable(version)
            and (not requirement.specifier or Version(version) in requirement.specifier)
            # ``pip`` would never take a pre-release for a requirement that does not
            # name one, and neither does the date rule: the newest listing is not the
            # newest release (§3.2). A requirement that names one keeps it.
            and _date_rule_candidate(
                version, requirement.specifier, *claim_specifiers
            )
        ]
        exempt = reference_pin(ref)
        narrowed_by: tuple[tuple[str, str], ...] = ()
        refused: tuple[tuple[str, str, str], ...] = ()
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
            # Installed as written, so the layer's other pins have no say here: this is
            # the release the benchmark's own environment held.
            version, released_at = max(matching, key=lambda item: Version(item[0]))
        else:
            pin_source = ref.pin_source
            dated = [item for item in matching if item[1][:10] <= cutoff]
            if not dated:
                newest = _newest_release(available)
                raise BakeError(
                    f"{ref.source}: {ref.spec} has no release on or before {cutoff} "
                    f"(newest {ref.name} is {newest[0]}, released {newest[1][:10]}). "
                    "The tree this requirement was declared in could not have been "
                    "built with it."
                )
            if claims:
                wanted = [
                    item
                    for item in dated
                    if all(Version(item[0]) in SpecifierSet(spec) for _, spec in claims)
                ]
                if not wanted:
                    raise BakeError(
                        f"{ref.source}: {ref.spec} has no release on or before {cutoff} "
                        "that satisfies "
                        + "; ".join(
                            f"{pin} declaring {spec}" for pin, spec in claims
                        )
                        + " — two pins of this layer cannot both hold, and neither is "
                        "BugsInPy's own, so nothing here can be installed as written "
                        "(§3.2)"
                    )
                if wanted != dated:
                    narrowed_by = tuple(claims)
                    dated = wanted
            if metadata_of is None:
                version, released_at = max(dated, key=lambda item: Version(item[0]))
            else:
                # The layer's *own* pins get the last word over the date: a release
                # whose declarations do not satisfy one of them is not an answer this
                # layer can hold (§3.2's case (a), found before the build rather than by
                # the layer's ``pip check`` after it).
                version, released_at, refused = _chosen_release(
                    ref,
                    dated,
                    cutoff=cutoff,
                    settled=settled,
                    metadata_of=metadata_of,
                )
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
                narrowed_by=narrowed_by,
                refused=refused,
            )
        )
    return resolved


PYPI_JSON = "https://pypi.org/pypi/{name}/json"


TRANSITIVE_SOURCE = "required by this layer's own pins"
"""The ``source`` on the synthetic reference a transitive pick is chosen through.

It reaches the reader only through a :class:`BakeError` message — a transitive pick has
no declaring source, which is the whole point of :attr:`ResolvedDep.required_by` — and it
is written as "required by this layer's own pins" rather than as a file name so that a
refusal names the pins in :attr:`ResolvedDep.required_by` and not a file: nothing in the
bug's tree wrote the requirement.
"""


def marker_applies(requirement: Requirement, *, python_minor: str, extra: str = "") -> bool:
    """Whether ``requirement``'s own marker holds **inside** the layer (§3.2).

    [VANTA-WHY] The three readers that move a settled version on a declaration
    (:func:`_chosen_release`, :func:`_narrowing_requirements`, :func:`metadata_conflicts`)
    skip every marker, and for their question that is right: a conditional requirement is
    not something to move a settled pin's, or to refuse a layer's, on — the conservative
    direction there is to leave the declaration unread and let the layer's own ``pip
    check`` report it. Everywhere a requirement is read *as an install* the question is
    the other way round — what will pip put in the environment — and there a marker is
    the question, because pip skips what does not apply. Measured, that is what
    ``pandas/1``'s early pass got wrong: ``scikit-learn==0.23.1``'s build declares
    ``numpy==1.13.3`` under ``python_version == "3.6"``, which is false for every layer
    this harness bakes, and the ordering finding named it anyway.

    ``extra`` is the extra the requirement was read *from*, and it is empty unless the
    requirement's own marker asks for one (:func:`marker_skips`): a declaration inside an
    extra is installed because that extra was requested, while PyPI's ``requires_dist``
    carries ``; extra == "test"`` entries that pip does not install unless the extra is
    asked for — so the default is the empty string and the closure walk
    (:func:`transitive_pins`) leaves it that way.

    Evaluated against the layer's own platform and never the host's:
    :func:`packaging.markers.default_environment` answers for the machine running the
    bake, which is a Mac, and would silently drop every ``sys_platform``-conditional
    requirement the layer really installs. Every variable for this layer's platform is
    supplied, so :class:`UndefinedEnvironmentName` reaches here in exactly one measured
    way: a marker on a variable ``packaging`` knows but this environment does not carry —
    ``dependency_groups``, which PEP 735 added and a layer is not a project with groups —
    and that is read as *not applying*, because pip cannot evaluate it either and the
    freeze gate (:func:`unpinned_freeze`) is what names the distribution if pip installed
    it anyway. A misspelt variable is not this branch: ``packaging`` refuses it at parse,
    so :func:`_requirement` raises and the ref is kept as unparseable rather than read as
    conditional (measured on ``packaging 26.3``).
    """
    if requirement.marker is None:
        return True
    environment = {
        "python_version": python_minor,
        "python_full_version": f"{python_minor}.0",
        "implementation_name": "cpython",
        "implementation_version": f"{python_minor}.0",
        "os_name": "posix",
        "sys_platform": "linux",
        "platform_machine": "aarch64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "extra": extra,
    }
    try:
        return bool(requirement.marker.evaluate(environment))
    except UndefinedEnvironmentName:
        return False


_SOURCE_LINE_RE = re.compile(r"\s+line\s+(\d+)$")
"""The trailing ``line N`` the file-backed readers write into a ref's ``source``.

Read back out here so a skipped requirement is recorded in the same shape as every other
:class:`SkippedLine` — the file in ``source`` and the number in ``line`` — rather than as
one string that no reader of the provenance can compare against the others.
"""

_EXTRA_CLAUSE_RE = re.compile(r"""extra\s*==\s*['"]([^'"]*)['"]""")
"""The extra a requirement's own marker asks for (``; extra == "test"``), if any."""


def marker_skips(
    refs: Sequence[DepRef], *, python_minor: str
) -> tuple[list[DepRef], list[SkippedLine]]:
    """``(refs whose markers hold in this layer, the ones dropped for not holding)``.

    §3.2's rule for every requirement that is read as something to **install**:
    ``setup.sh``'s lines, BugsInPy's ``requirements.txt``, ``install_requires``,
    ``tests_require``, the ``tests`` extra, a root ``test_requirements.txt``, a
    ``pyproject.toml``'s dependency arrays, ``setup_requires`` and
    ``[build-system].requires`` — the tree's own and, through
    :func:`early_install_pins`, a pinned release's. Applied to the *collected* refs, once,
    so the rule has one home instead of one per reader, and the refs are read here exactly
    as their readers wrote them: :attr:`DepRef.spec` is the requirement verbatim and
    therefore still carries the marker, which is why nothing had to be threaded down
    through the parsers to say so.

    A requirement whose marker is false for this layer is not something pip would
    install, and reading it as one describes an environment the layer never has: measured,
    ``pandas/1``'s early pass named ``scikit-learn==0.23.1``'s build requirement
    ``numpy==1.13.3`` under ``python_version == "3.6"``, and ``luigi/1``/``sanic/1`` each
    carry ``cffi>=1.8,!=1.11.3; platform_python_implementation != 'PyPy'``, which *is* true
    for CPython and stays.

    The extra: empty unless the requirement's own marker names one, which is the case that
    exists — ``extras_require["tests"]`` entries are read as declarations of an extra this
    harness does install, so an entry that says so about itself is not dropped for saying
    it. The direction is deliberately narrow: a *conditional* requirement whose condition
    is unrelated to its own extra still has to hold on this platform to be installed.

    A ref whose spec does not parse is kept rather than dropped: every ref here was built
    by :func:`dep_ref`, which already refused an unparseable requirement, so a refusal at
    this point is not a fact about the requirement and dropping one would be a silent loss
    of a declaration.
    """
    kept: list[DepRef] = []
    dropped: list[SkippedLine] = []
    for ref in refs:
        try:
            requirement = _requirement(ref.spec, ref.source)
        except BakeError:
            kept.append(ref)
            continue
        named = (
            _EXTRA_CLAUSE_RE.search(str(requirement.marker))
            if requirement.marker is not None
            else None
        )
        if marker_applies(
            requirement,
            python_minor=python_minor,
            extra=named.group(1) if named else "",
        ):
            kept.append(ref)
            continue
        source, line = ref.source, 0
        written = _SOURCE_LINE_RE.search(source)
        if written is not None:
            source, line = source[: written.start()], int(written.group(1))
        dropped.append(
            SkippedLine(
                source=source,
                line=line,
                text=ref.spec,
                reason=(
                    "its environment marker does not hold in this layer: "
                    f"{requirement.marker}"
                ),
            )
        )
    return kept, dropped


def _transitive_pin(
    name: str,
    claims: Sequence[tuple[str, str]],
    cutoff: str,
    *,
    releases: Mapping[str, Mapping[str, str]],
    settled: Mapping[str, tuple[str, str]],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> ResolvedDep:
    """The pin for one transitively required name, by the date rule (§3.2).

    ``claims`` is ``(requiring pin, specifier)`` for every pin in the layer that declared
    this name, and the answer is the newest release on or before ``cutoff`` that
    satisfies **all** of them — the same rule :func:`resolve_pins` applies to a
    declaration, with the declarations of a release standing in for the declarations of
    a file. The layer's own unsettled pins still get the last word
    (:func:`_chosen_release`), so a transitive pick cannot install a release one of them
    refuses any more than a declared pin can.

    [VANTA-WHY] An empty claim set cannot happen — the walk only ever asks about a name
    some pin declared — and no release satisfying the claims raises :class:`BakeError`
    rather than falling back to a newer one: the fallback is exactly what pip would have
    done on its own, and it is what put a 2025 ``exceptiongroup`` in a 2020 layer.

    The claims stand in for the declaration, so the pre-release rule is read off them:
    the newest release here has to be a *release* unless the claims name a pre-release
    (:func:`_date_rule_candidate`), which is what took ``matplotlib/1`` off
    ``jinja2==3.0.0a1`` (from ``sphinx==3.1.0``'s ``jinja2>=2.3``) and off the
    ``markupsafe==2.0.0a1`` that entry in turn required.
    """
    # Comma-joined, because this string is handed to :class:`SpecifierSet` below and
    # ``packaging`` reads clauses separated by commas: a space-joined pair of claims
    # (``>=0.14.0`` from one pin and ``>=0.5.3`` from another) is not a specifier set at
    # all and raises ``InvalidSpecifier``.
    spec = ",".join(sorted({specifier for _, specifier in claims if specifier}))
    claimed = SpecifierSet(spec)
    matching = [
        (version, date)
        for version, date in releases.items()
        if _parseable(version)
        and date[:10] <= cutoff
        and _date_rule_candidate(version, claimed)
        and all(
            not specifier or Version(version) in SpecifierSet(specifier)
            for _, specifier in claims
        )
    ]
    required_by = tuple(pin for pin, _ in claims)
    if not matching:
        raise BakeError(
            f"{TRANSITIVE_SOURCE}: {name}{spec} has no release on or before {cutoff} "
            f"(required by {'; '.join(required_by)}). Nothing here pins {name}, so pip "
            "would resolve it itself, at the newest release of the day — which is the "
            "layer this rule exists to refuse (§3.2)"
        )
    version, released_at, refused = _chosen_release(
        DepRef(name=name, spec=spec or name, source=TRANSITIVE_SOURCE),
        matching,
        cutoff=cutoff,
        settled=settled,
        metadata_of=metadata_of,
    )
    return ResolvedDep(
        name=name,
        spec=spec,
        sources=(),
        pin=f"{name}=={version}",
        harness=False,
        released_at=released_at[:10],
        commit_date=cutoff,
        refused=refused,
        required_by=required_by,
    )


def pin_cutoff(dep: ResolvedDep) -> Optional[str]:
    """The date a pin's own requirements are held to (§3.2).

    [VANTA-WHY] The **later** of the pin's declaration date and the date its release
    landed, because neither alone is right. A pin chosen by the date rule was released
    on or before the date it was declared, so for those the two agree and this is that
    date. A BugsInPy reference pin is installed *as written* and is not held to the
    date rule at all: ``pytest==5.4.3`` (2020-04-09) under a 2019-04-22 commit names a
    release a year after the tree it is used with, and its own requirements are that
    release's — holding them to the commit date would refuse ``pluggy<1.0,>=0.12``
    (whose 0.12.0 landed 2019-05-27), i.e. flag the bug for a dependency the reference
    environment itself could satisfy. A pin's requirements cannot pre-date the release
    that declares them, which is what ``max`` says.

    ``None`` for a pin with neither date: nothing can be said about its window, so the
    caller falls back to the layer's own commit date (:func:`transitive_pins`).
    """
    dates = [date for date in (dep.commit_date, dep.released_at) if date]
    return max(dates) if dates else None


@dataclass(frozen=True)
class OrphanedPin:
    """One distribution the interpreter layer holds that this layer does **not** (§3.2).

    [VANTA-WHY] ``pip`` does not uninstall what it replaces, so the environment a bug
    layer starts from is the interpreter layer's *whole* one: a layer installing
    ``pytest==5.4.3`` over ``8.0.2`` inherits 8.0.2's own dependencies and the base
    layer's ``pip-json-report``'s, and the ones its own pin list does not contain are
    left behind with nothing naming them. :data:`ORPHAN_RULE` is that fact; this record is
    what the layer does about it — the recipe uninstalls each one
    (:func:`requirements_recipe`), and the record says what was removed so a reader can
    check the claim rather than take it on trust.

    ``declared_by`` and ``declared`` are "why it was there", which is the half a reader
    cannot recover from the layer: the harness pin whose release brought the name in, and
    that release's own declaration of it — ``pytest==8.0.2`` declaring
    ``exceptiongroup >=1.0.0rc8 ; python_version < "3.11"``. Both are the base layer's
    *predicted* environment (:func:`transitive_pins`), taken at the versions the
    interpreter layer was built with.

    ``replaced_by`` is the other direction and the reason the name is removed rather than
    exempted: the pin *this* layer installs for the harness name above, ``pytest==5.4.3``
    for the example. It is empty only for a record built by hand — every orphan
    :func:`transitive_pins` returns has at least one, because an inherited name is
    orphaned only down a path from a harness pin this layer replaces, and the fix is the
    name of that pin.
    """

    name: str
    declared_by: tuple[str, ...] = ()
    """The pins of the **base** layer whose releases declared this name, sorted."""
    declared: tuple[str, ...] = ()
    """The declaration that brought it in, as the releasing metadata wrote it, sorted.

    A direct dependency of a harness pin writes it exactly as that pin's release did
    (``exceptiongroup >=1.0.0rc8 ; python_version < "3.11"``); one further down the
    inheritance names the release that declared *it*, which is the same fact one hop on.
    """
    replaced_by: tuple[str, ...] = ()
    """The pins of *this* layer for the harness names above, sorted."""

    @property
    def rule(self) -> str:
        """Why this name is removed, in one phrase, for the provenance."""
        return ORPHAN_RULE

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "rule": self.rule}
        if self.declared_by:
            data["declared_by"] = list(self.declared_by)
        if self.declared:
            data["declared"] = list(self.declared)
        if self.replaced_by:
            data["replaced_by"] = list(self.replaced_by)
        return data

    def describe(self) -> str:
        """One line for a build log or the preflight: the name and why it was there."""
        because = ", ".join(self.declared_by) or "the interpreter layer"
        instead = (
            f" (this layer installs {', '.join(self.replaced_by)})"
            if self.replaced_by
            else ""
        )
        return f"{self.name} — installed for {because}{instead}"


@dataclass(frozen=True)
class TreeSatisfied:
    """One requirement a release declared that the **tree under test** satisfies (§3.2).

    [VANTA-WHY] The exception to the closure walk, and it is the one name a layer must
    never pin: measured on ``pandas/1``, ``statsmodels==0.11.1`` and ``xarray==0.15.1``
    each declare ``pandas``, the walk pinned ``pandas==1.0.3`` for it, and the recipe
    would have installed a released pandas over the tree the bug is graded on — an
    environment where the suite imports the wrong source. The exemption is the one the
    import closure already makes (:func:`tree_provided_modules`): the bug's own
    distribution name and every top-level module the tree ships are the tree's, whoever
    asks for them. :data:`TREE_SATISFIED_RULE` is the fact; this record is what the walk
    returns instead of a pin, so a reader can check the claim rather than take it.

    ``required_by`` and ``declared`` are the half a reader cannot recover from the
    layer — which releases asked for the tree's own name, and how they asked — and they
    are the same two fields :class:`OrphanedPin` carries for the same reason.

    ``shipped`` is the tree's own spelling of the name: ``pandas`` for the distribution,
    or the directory a module lives in (``pylab``, ``docs_src``). It is what matched, and
    it is the one thing that says *which* half of the exemption applied.
    """

    name: str
    """The canonical name the requirement asked for, as the walk reads it."""
    shipped: tuple[str, ...] = ()
    """What the tree ships that answered it, sorted."""
    required_by: tuple[str, ...] = ()
    """The pins whose releases declared this name, sorted."""
    declared: tuple[str, ...] = ()
    """The declaration that asked for it, as the releasing metadata wrote it, sorted."""

    @property
    def rule(self) -> str:
        """Why nothing is installed for this name, in one phrase, for the provenance."""
        return TREE_SATISFIED_RULE

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "rule": self.rule}
        if self.shipped:
            data["shipped_by_the_tree"] = list(self.shipped)
        if self.required_by:
            data["required_by"] = list(self.required_by)
        if self.declared:
            data["declared"] = list(self.declared)
        return data

    def describe(self) -> str:
        """One line for a build log or the preflight: the name and who asked for it."""
        asked = ", ".join(self.required_by) or "no pin this layer installs"
        return f"{self.name} — declared by {asked}, provided by the tree ({', '.join(self.shipped) or self.name})"


def transitive_pins(
    pins: Sequence[tuple[str, str]],
    *,
    harness_pins: Mapping[str, str],
    commit_date: str,
    python_minor: str,
    releases: Mapping[str, Mapping[str, str]],
    fetch: Callable[[str], Mapping[str, str]],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
    cutoffs: Optional[Mapping[str, str]] = None,
    tree_satisfied: Optional[Mapping[str, str]] = None,
) -> tuple[
    tuple[ResolvedDep, ...],
    tuple[str, ...],
    tuple[OrphanedPin, ...],
    tuple[TreeSatisfied, ...],
]:
    """``(transitive pins, names provided, names orphaned, names the tree satisfies)``.

    [VANTA-WHY] Everything else in this file pins what a *file* declared. This walks what
    a *release* declares, because that is the half of a layer's environment the pin list
    never described: pip resolves a pin's own ``requires_dist`` against PyPI at install
    time, at the newest release that moment allows, and the date rule never sees the
    answer. ``fastapi/1`` measured the cost — ``exceptiongroup==1.3.1`` (2025) came in
    under a layer whose own pins hold ``typing-extensions==3.7.4.2``, and ``pip check``
    refused the layer for a contradiction no pin in it had declared.

    The walk starts at every pin the layer installs (:func:`pinned_parts`, build pins
    first) and follows each release's own declarations, markers evaluated for the layer's
    platform (:func:`marker_applies`), to the full transitive closure. A name that is
    already pinned is *satisfied* and stops the walk; a name nothing pins is pinned by
    the same date rule and the walk continues into **that** release's declarations
    (:func:`_transitive_pin`, :data:`TRANSITIVE_PIN_RULE`).

    The second half of the answer is the set of names this layer must **not** pin, and it
    is why the walk cannot simply start from the pins: the environment it is building is
    *inherited*. The interpreter layer installs ``pytest`` and ``pytest-json-report`` and
    everything those releases require, and pip does not uninstall what it replaces — a
    layer whose own ``requirements.txt`` pins ``pytest==5.4.3`` swaps the name out while
    ``pytest 8.0.2``'s own dependencies stay behind. So the **whole** closure of the
    harness pins is walked, at the versions the interpreter layer has, and every name the
    layer's own pins do not decide for themselves is reported back as *provided*: those
    are the names :func:`unpinned_freeze` exempts, because this layer neither installed
    them nor wrote them down. The closure of a provisioned name is resolved at the newest
    release and not by date, because the interpreter layer has no date rule — it is the
    harness's own environment and pip built it out of the day it was baked. That makes
    this set a *prediction* of the interpreter layer's environment: what it is used for is
    the **names**, which is what :func:`unpinned_freeze` compares, and a name that appears
    or disappears with a release is the one thing this cannot see.

    What this layer's own pins reach is pinned **regardless** of the inheritance, in the
    other direction: ``pytest 5.4.3`` declares ``pluggy (<1.0,>=0.12)`` and the
    interpreter layer holds a ``pluggy`` of its own, so a walk that stopped at a
    provisioned name would install pytest 5.4.3 over a pluggy it refuses. The names the
    layer pins win; the leftovers are what is exempt.

    The third answer is the exception to the second, and the only part of the inheritance
    that cannot be left alone (:data:`ORPHAN_RULE`). A name the layer *replaces* a harness
    pin for is not a name the interpreter layer provides *it* with — it is a leftover of a
    release this layer swapped out, and the walk knows which because it remembers, for
    every inherited name, which harness pins its path started at. ``pytest 8.0.2``'s
    ``exceptiongroup``, ``iniconfig`` and ``tomli`` came in under a ``pytest`` this layer
    re-pins to ``5.4.3``, the layer's own pins name none of them, its closure does not
    reach them (pytest 5.4.3 does not declare them) — and a layer holding a 2025
    ``exceptiongroup`` beside its own ``typing-extensions==3.7.4.2`` is the contradiction
    ``fastapi/1``'s ``pip check`` refused. Each is an :class:`OrphanedPin`: removed from
    the layer by the recipe and **not** exempted by the freeze gate, so a name left behind
    is reported rather than passed. Only a name *every* one of whose paths starts at a
    replaced harness pin is one of these: a name a kept harness pin also needs is that
    pin's dependency and stays (:func:`OrphanedPin`'s own note on ``replaced_by``).

    ``cutoffs`` maps a canonical pin name to the date that pin's **own** requirements are
    held to (:func:`pin_cutoff` — for a date-rule pin that is the date it was pinned
    against, and for a reference pin installed as written it is that release's own date,
    which is later); a name with no entry is held to ``commit_date``. A transitive pick's
    cutoff is the **newest** of its requirers' cutoffs, the rule
    :func:`resolve_build_pins` uses for the same reason: the requirement is the later
    one's own, so holding it to the earlier date would refuse the release that
    declaration's environment had.

    ``tree_satisfied`` is the fourth answer, and the one name the walk must **not** pin:
    the bug's own distribution name and every top-level module the tree ships, mapped to
    the tree's own spelling of it. The tree under test is never a transitive pin
    (:data:`TREE_SATISFIED_RULE`) — measured, ``pandas/1`` pinned ``pandas==1.0.3``
    because ``statsmodels==0.11.1`` and ``xarray==0.15.1`` declare ``pandas``, which
    would have installed a released pandas *over the tree the suite imports*. The
    exemption is the import closure's own (:func:`tree_provided_modules`), applied to
    the other half of the layer, and it holds in both walks: such a name is neither
    pinned here nor counted as something the interpreter layer provides, because neither
    is where it comes from. Each one is returned as a :class:`TreeSatisfied` record
    naming the releases that asked for it, so "satisfied by the tree" is a statement in
    the provenance rather than a name that quietly went missing.

    Pure apart from ``fetch``, which is called only for a name whose release dates are
    not already in ``releases``: the transitive picks, and the names the interpreter
    layer's provision walk has to descend through. A test that passes everything it
    needs never touches the network.
    """
    dates: dict[str, Mapping[str, str]] = dict(releases)
    cutoffs = cutoffs or {}
    tree = {canonicalize_name(name): name for name in (tree_satisfied or {})}
    # The tree's own names, and the two halves of the record each one leaves behind
    # (:class:`TreeSatisfied`): which releases asked for it, and how they asked. Both
    # walks below feed this, and neither descends through a name it holds.
    tree_asked_by: dict[str, set[str]] = {}
    tree_declared: dict[str, set[str]] = {}

    def note_tree(name: str, declared_by: str, version: str, raw: str) -> None:
        """Record one requirement the tree under test answers (:data:`TREE_SATISFIED_RULE`).

        ``name`` is the tree's own name the walk refused to pin, ``declared_by``/``version``
        the release whose metadata asked for it, and ``raw`` that declaration as written —
        the two halves a reader cannot recover from the layer.
        """
        tree_asked_by.setdefault(name, set()).add(f"{declared_by}=={version}")
        tree_declared.setdefault(name, set()).add(raw)

    settled: dict[str, tuple[str, str]] = {
        canonicalize_name(name): (f"{name}=={version}", version)
        for name, version in harness_pins.items()
    }
    for name, version in pins:
        # After the harness entries: the bug's own pin is installed *over* the
        # interpreter layer's version and is what the layer ends up holding.
        settled[canonicalize_name(name)] = (f"{name}=={version}", version)
    pinned = frozenset(canonicalize_name(name) for name, _ in pins)

    def releases_of(name: str) -> Mapping[str, str]:
        if name not in dates:
            dates[name] = dict(fetch(name))
        return dates[name]

    def declarations(name: str, version: str) -> tuple[tuple[str, str, str], ...]:
        """``((canonical name, specifier, raw), ...)`` for one release's own requirements.

        ``raw`` is the requirement as that release's own metadata wrote it, carried for the
        one reader that has to quote it back: an orphan's record says which release brought
        it in *and what that release declared* (:class:`OrphanedPin`), and the specifier
        alone would not be the declaration.
        """
        try:
            requires, _ = metadata_of(name, version)
        except BakeError:
            # A release whose declarations cannot be read describes nothing: it is not
            # evidence that it requires nothing (httpcore==0.3.0 is read no other way),
            # and the layer's own pip check is the backstop for what the walk missed.
            return ()
        out: list[tuple[str, str, str]] = []
        for raw in requires:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                continue
            if not marker_applies(requirement, python_minor=python_minor):
                continue
            out.append(
                (canonicalize_name(requirement.name), str(requirement.specifier), raw)
            )
        return tuple(out)

    # (A) Everything the interpreter layer is predicted to hold: the closure of **every**
    # harness pin, at the version the interpreter layer has. Descending through a name
    # this layer re-pins is not an oversight — pip does not uninstall what it replaces.
    # ``pytest 8.0.2``'s own dependencies stay in the environment after a layer installs
    # ``pytest 5.4.3`` over it, and that leftover is exactly what ``fastapi/1``'s ``pip
    # check`` failure was made of (``exceptiongroup``).
    #
    # Walked once per (name, harness pin) pair rather than once per name, so that a name
    # two harness pins reach carries **both** as its ``bringers`` *and* passes both on to
    # what it declares: phase (C) discriminates on exactly that set, and a name whose
    # descent stopped at the first pin to reach it would look like it came from one. A
    # harness pin is a *root* of that walk and not a node in it, so a path that reaches
    # one from another root stops there: ``pytest-json-report`` requires ``pytest``, and
    # counting ``pytest 8.0.2``'s declarations as the plugin's would attribute
    # ``exceptiongroup`` to a harness pin this layer keeps — the one misreading that makes
    # phase (C) miss the ``fastapi/1`` orphan.
    inherited: set[str] = set()
    roots = frozenset(canonicalize_name(source) for source in harness_pins)
    bringers: dict[str, set[str]] = {}
    declared_raw: dict[str, set[str]] = {}
    walking: list[tuple[str, str, str]] = []
    read_from: set[tuple[str, str]] = set()
    for source, version in harness_pins.items():
        canonical = canonicalize_name(source)
        inherited.add(canonical)
        bringers.setdefault(canonical, set()).add(source)
        if (canonical, source) not in read_from:
            read_from.add((canonical, source))
            walking.append((canonical, version, source))
    while walking:
        name, version, source = walking.pop()
        for required, _, raw in declarations(name, version):
            if required in roots:
                # The interpreter layer's own pin, and a root of this walk: the root it is
                # reached from is itself, so what it declares is its own closure and not
                # the reaching pin's. Left as the seed recorded it.
                continue
            if required in tree:
                # The tree's own name, whoever asks for it: neither something this layer
                # pins nor something the interpreter layer provides it *with* — the suite
                # imports it from the tree (:data:`TREE_SATISFIED_RULE`). Not descended,
                # because there is no release of it this layer is entitled to resolve.
                note_tree(required, name, version, raw)
                continue
            bringers.setdefault(required, set()).add(source)
            declared_raw.setdefault(required, set()).add(raw)
            inherited.add(required)
            if (required, source) in read_from:
                continue
            read_from.add((required, source))
            try:
                available = releases_of(required)
            except BakeError:
                # The walk only wanted the version in order to descend; the *name* is
                # what the set is for, and a release list that cannot be read is not
                # evidence the interpreter layer lacks it. Recorded and not descended,
                # which is the conservative direction: an over-large exemption is what
                # the layer's own ``pip check`` then has to catch.
                continue
            if available:
                walking.append(
                    (required, max(available, key=lambda item: Version(item)), source)
                )

    # The layer's own closure. Claims are collected per name before a version is chosen
    # for it — a name two pins declare is pinned once, against both — and a claim that
    # arrives after a choice was made re-chooses it, so the answer does not depend on the
    # order the pins were read in.
    claims: dict[str, list[tuple[str, str]]] = {}
    node_cutoffs: dict[str, set[str]] = {}
    chosen: dict[str, str] = {}
    records: dict[str, ResolvedDep] = {}
    queue: list[tuple[str, str, str]] = [
        (canonicalize_name(name), version, cutoffs.get(canonicalize_name(name), commit_date))
        for name, version in pins
    ]
    read: set[tuple[str, str]] = set()
    while queue:
        name, version, cutoff = queue.pop()
        if (name, version) in read:
            continue
        read.add((name, version))
        for required, specifier, raw in declarations(name, version):
            if required in pinned:
                continue
            if required in tree:
                # The project under test is never a transitive pin (:data:`TREE_SATISFIED_RULE`):
                # ``pandas/1``'s own closure walk pinned ``pandas==1.0.3`` because
                # ``statsmodels==0.11.1`` and ``xarray==0.15.1`` declare it, which would have
                # installed a released pandas over the tree the suite imports. Recorded as
                # satisfied by the tree rather than pinned, and not descended for the same
                # reason as above.
                note_tree(required, name, version, raw)
                continue
            node_cutoffs.setdefault(required, set()).add(cutoff)
            wanted = (f"{name}=={version}", specifier)
            bucket = claims.setdefault(required, [])
            if wanted not in bucket:
                bucket.append(wanted)
                bucket.sort()
            required_cutoff = max(node_cutoffs[required])
            record = _transitive_pin(
                required,
                claims[required],
                required_cutoff,
                releases=releases_of(required),
                settled=settled,
                metadata_of=metadata_of,
            )
            records[required] = record
            if chosen.get(required) != record.pin.partition("==")[2]:
                chosen[required] = record.pin.partition("==")[2]
                queue.append((required, chosen[required], required_cutoff))
    # (B) The exemption the freeze gate reads: the interpreter layer's names minus the
    # ones this layer decides for itself. A name this layer pins is not "supplied" — the
    # layer installs over it at a version of its own.
    #
    # (C) And minus the leftovers of a harness pin this layer *replaces*, which are the
    # one part of the interpreter layer's environment it may not leave alone
    # (:data:`ORPHAN_RULE`). ``settled`` already says which those are: it holds the
    # interpreter layer's version for every harness pin and the layer's own over the top of
    # it, so a harness pin whose entry moved is one this layer installs a different release
    # of — and a name every path of which starts at one of those is a dependency of a
    # release this layer swapped out, not a name the interpreter layer provides it with.
    # A name a *kept* harness pin also reaches is that pin's own dependency and stays —
    # ``bringers[name] <= replaced`` is the whole of that test, and the reason the walk
    # above is taken per (name, harness pin) pair rather than per name.
    replaced = {
        source
        for source, version in harness_pins.items()
        if settled[canonicalize_name(source)][1] != version
    }
    supplied = inherited - pinned - set(records)
    orphans = tuple(
        OrphanedPin(
            name=name,
            declared_by=tuple(sorted(bringers[name])),
            declared=tuple(sorted(declared_raw[name])),
            replaced_by=tuple(
                sorted(settled[canonicalize_name(pin)][0] for pin in bringers[name])
            ),
        )
        for name in sorted(supplied)
        if bringers[name] <= replaced
    )
    removed = {pin.name for pin in orphans}
    # The fourth answer, and the one the layer installs nothing for: a requirement a
    # release declared that the tree under test itself satisfies (:data:`TREE_SATISFIED_RULE`).
    # It is in neither of the other two — it is not a pin, and it is not a name the
    # interpreter layer provides — because there is a third place it comes from.
    satisfied = tuple(
        TreeSatisfied(
            name=name,
            shipped=(tree[name],),
            required_by=tuple(sorted(tree_asked_by[name])),
            declared=tuple(sorted(tree_declared[name])),
        )
        for name in sorted(tree_asked_by)
    )
    return (
        tuple(records[name] for name in sorted(records)),
        tuple(sorted(supplied - removed)),
        orphans,
        satisfied,
    )

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


PYPI_VERSION_JSON = "https://pypi.org/pypi/{name}/{version}/json"

PYPI_SIDECAR_SUFFIX = ".metadata"
"""PEP 658's published metadata file: the release file's URL plus this suffix.

[VANTA-WHY] It exists because the version JSON's ``info.requires_dist`` is **not**
readable for most of the dev split's pins. Measured against PyPI: it is ``null`` for
``matplotlib==3.2.1`` — a release that declares five requirements — for
``sqlalchemy==1.3.17``, ``numpy==1.18.4``, ``six==1.17.0``, ``pytz``, ``tornado``,
``kiwisolver``, ``pillow``, ``pyparsing`` and every ``httpcore``/``hpack``/
``hyperframe``/``hstspreload`` pin; across the split it is ``null`` for 60 of
``sanic/1``'s 145 pins, 106 of ``luigi/1``'s 173, 21 of ``pandas/1``'s 42 and 5 of
``matplotlib/1``'s 10. PyPI never returns ``[]``, so a ``null`` cannot be told apart
from "declares nothing" by that field alone — which is why reading it as "requires
nothing" (``requires_dist or ()``) was wrong for most of a layer, and why every caller
that cannot read a release is now told so instead (:data:`CASE_UNKNOWN`). The sidecar
is the same document pip itself reads, published beside the file, and it is where
those declarations were measured from.
"""

REQUIRES_DIST_HEADER = "requires-dist"
"""The Core Metadata field that carries one requirement of a release."""

WHEEL_METADATA_MEMBER = ".dist-info/METADATA"
"""Where a wheel keeps the document the sidecar would have published."""


def _requires_dist_fields(text: str) -> tuple[str, ...]:
    """The ``Requires-Dist`` fields of a Core Metadata document.

    Field-name-only matching, because this document is not the version JSON: PyPI's
    sidecar writes the field in mixed case (``Requires-Dist:``), separates the value
    with a colon, and the value is the requirement itself (``cycler (>=0.10)``).

    A document with no such field is a genuine **empty** reading — the release declares
    nothing — and not the unreadable case, which is why this returns a tuple and never
    raises: measured, ``pytest-json-report==0.2``'s sidecar has no ``Requires-Dist`` at
    all, and that is what keeps httpie/1's plugin pin at 0.2
    (:func:`newest_accepting_pytest`).
    """
    out: list[str] = []
    for line in text.splitlines():
        field, sep, value = line.partition(":")
        if not sep or field.strip().lower() != REQUIRES_DIST_HEADER:
            continue
        value = value.strip()
        if value:
            out.append(value)
    return tuple(out)


def _metadata_member(data: bytes) -> Optional[str]:
    """The ``METADATA`` document inside a wheel, or ``None`` when it holds none."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [
                name for name in archive.namelist()
                if name.endswith(WHEEL_METADATA_MEMBER)
            ]
            if not names:
                return None
            return archive.read(names[0]).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, OSError, KeyError):
        return None


@dataclass(frozen=True)
class PyPIFile:
    """One file a release is distributed as: the unit ``preflight`` classifies."""

    filename: str
    packagetype: str
    url: str = ""
    size: int = 0
    core_metadata: bool = False
    """Whether PyPI publishes this file's PEP 658 metadata beside it.

    [VANTA-WHY] ``core-metadata`` in the version JSON is a *mapping* — the sidecar's
    digest — when the file has one, and absent when it does not, so it is read for
    truth and carried as a flag. The sidecar is that file's own URL with
    :data:`PYPI_SIDECAR_SUFFIX` appended, which makes "is there one" a fact about the
    release rather than a 404 to discover per file (:func:`pypi_sidecar_text`).
    """

    @property
    def is_wheel(self) -> bool:
        return self.filename.endswith(".whl")

    @property
    def is_sdist(self) -> bool:
        # ``packagetype`` is the field that means it; the suffix is the fallback,
        # because the same JSON is read for releases old enough to predate the field.
        return self.packagetype == "sdist" or self.filename.endswith((".tar.gz", ".zip"))


def _release_files(listed: Any) -> tuple[PyPIFile, ...]:
    """The version JSON's ``urls`` list as :class:`PyPIFile`s.

    Shared by the two readers of one release — what it can be installed as
    (:func:`pypi_release_files`) and, when the JSON carries no declarations, which of
    its files the metadata should be read from (:func:`declared_requires`) — so that
    "what files does this release have" is answered once.
    """
    return tuple(
        PyPIFile(
            filename=str(item.get("filename") or ""),
            packagetype=str(item.get("packagetype") or ""),
            url=str(item.get("url") or ""),
            size=int(item.get("size") or 0),
            core_metadata=bool(item.get("core-metadata")),
        )
        for item in listed
        if isinstance(item, dict)
    )


def pypi_release_files(name: str, version: str) -> Optional[tuple[PyPIFile, ...]]:
    """Every file PyPI lists for one release, out of the release's version JSON.

    ``None`` when PyPI has no such release — an HTTP 404, which is an *answer* ("this
    release does not exist") and not a fetch that failed. The distinction is the whole
    reason this returns rather than raises for it: ``luigi/1``'s
    ``requests-async==0.5.0`` and ``sanic/1``'s ``pywin32==227`` are both class (c)
    (§3.2) for the same reason — nothing pip can install — and the preflight has to be
    able to say which of the two it found rather than stopping on the first. A release
    that **does** exist and lists no file is a non-``None`` empty tuple, which is a
    third thing again.

    ``urls`` and not ``info``: :func:`pypi_release_metadata` reads what the release
    *declares*, this reads what it can be **installed as** — the wheels, their tags,
    and the sdist. Called only by the bake, like :func:`pypi_releases`.
    """
    url = PYPI_VERSION_JSON.format(name=name, version=version)
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT_SECS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise BakeError(
            f"cannot read the files of {name}=={version} from {url}: {exc}"
        ) from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BakeError(
            f"cannot read the files of {name}=={version} from {url}: {exc}"
        ) from exc

    listed = payload.get("urls")
    if not isinstance(listed, list):
        raise BakeError(f"{url} returned no 'urls' list")
    return _release_files(listed)


def pypi_sidecar_text(file: PyPIFile) -> Optional[str]:
    """The PEP 658 metadata published beside one release file, or ``None`` for none.

    ``None`` is "PyPI published no sidecar for this file", returned rather than raised
    for the same reason :func:`pypi_release_files` returns ``None`` for a 404: it is an
    *answer*, and the caller's next reading is the wheel itself
    (:func:`wheel_metadata_text`). Measured, the sidecar is what makes the declarations
    of ``matplotlib==3.2.1`` (``cycler (>=0.10)``, ``kiwisolver (>=1.0.1)``,
    ``numpy (>=1.11)``, ``pyparsing (!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1)``,
    ``python-dateutil (>=2.1)``), of ``sqlalchemy==1.3.17`` (extras only) and of
    ``pytest-json-report==0.1``/``==0.2`` (nothing at all) readable, none of which the
    version JSON carries.
    """
    url = file.url + PYPI_SIDECAR_SUFFIX
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT_SECS) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise BakeError(
            f"cannot read the metadata beside {file.filename}: {exc}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BakeError(f"cannot read the metadata beside {file.filename}: {exc}") from exc


def wheel_metadata_text(file: PyPIFile) -> Optional[str]:
    """The ``METADATA`` document inside one wheel, or ``None`` when it holds none.

    The last resort for a release whose declarations the JSON does not carry and whose
    sidecar PyPI did not publish, and it downloads the wheel to read them — the same
    document the sidecar would have served without the download, which is why it is
    second and not first. A wheel past :data:`SDIST_MAX_BYTES` is not read: it is the
    same fuse the sdist listing uses, and a release past it is reported as unknown
    rather than guessed at.
    """
    try:
        with urllib.request.urlopen(file.url, timeout=PYPI_TIMEOUT_SECS) as response:
            data, _ = _read_capped(response, SDIST_MAX_BYTES)
    except (urllib.error.URLError, OSError) as exc:
        raise BakeError(f"cannot download {file.filename}: {exc}") from exc
    return _metadata_member(data)


def declared_requires(
    payload: Mapping[str, Any],
    *,
    name: str,
    version: str,
    sidecar_of: Callable[[PyPIFile], Optional[str]] = pypi_sidecar_text,
    wheel_of: Callable[[PyPIFile], Optional[str]] = wheel_metadata_text,
) -> tuple[str, ...]:
    """What one release declares it needs, or :class:`BakeError` for "unknown".

    [VANTA-WHY] Three readings in order, because the first is missing for most of the
    split's pins (:data:`PYPI_SIDECAR_SUFFIX`):

    #. ``info.requires_dist`` from the version JSON. Present means read — **including
       when it is an empty list**, which is the release saying it needs nothing.
    #. The PEP 658 sidecar of any file that carries one: the document pip itself
       installs from, without downloading the file.
    #. The ``METADATA`` inside the wheel, when the sidecar was not published.

    None of the three reaching a document is a **raise** and not an empty tuple:
    "could not be read" and "declares nothing" are different answers, and only one of
    them may be written into a layer's pin list as "nothing to honour". ``httpcore==0.3.0``
    is the pin this is for — sdist-only, no sidecar, ``requires_dist`` null — and the
    layer's own ``pip check`` is what reads its ``h11==0.8.*``
    (:func:`pip_check_conflicts`). ``setup.py`` is deliberately *not* executed to find
    out: an sdist's declarations cannot be had without running the code that writes
    them, and running a benchmark's build code during a preflight is not a reading. An
    sdist-only release therefore reads as unknown, which is what the preflight prints
    by name (:data:`CASE_UNKNOWN`) and what does not stop a bake.
    """
    info = payload.get("info")
    if not isinstance(info, dict):
        raise BakeError(f"PyPI's entry for {name}=={version} has no 'info' mapping")
    declared = info.get("requires_dist")
    if declared is not None:
        if isinstance(declared, str):
            return (declared,)
        return tuple(str(item) for item in declared)
    listed = payload.get("urls")
    files = _release_files(listed) if isinstance(listed, list) else ()
    for file in files:
        if not file.core_metadata:
            continue
        text = sidecar_of(file)
        if text is not None:
            return _requires_dist_fields(text)
    for file in files:
        if not file.is_wheel:
            continue
        text = wheel_of(file)
        if text is not None:
            return _requires_dist_fields(text)
    raise BakeError(
        f"PyPI carries no 'requires_dist' for {name}=={version}, and neither a wheel "
        "nor a PEP 658 sidecar to read it from, so what it declares is unknown"
    )


def pypi_release_metadata(
    name: str,
    version: str,
    *,
    sidecar_of: Callable[[PyPIFile], Optional[str]] = pypi_sidecar_text,
    wheel_of: Callable[[PyPIFile], Optional[str]] = wheel_metadata_text,
) -> tuple[tuple[str, ...], Optional[str]]:
    """``(requires_dist, requires_python)`` for one release, from PyPI.

    [VANTA-WHY] ``requires_dist`` is what the release itself declares, so "which release
    of pytest-json-report accepts this pytest" is answered by the packages rather than
    by a table in this file that would be stale the next time either of them ships.
    Measured against PyPI: 1.5.0 declares ``pytest>=3.8.0``, 1.2.4 ``pytest>=4.2.0``,
    0.3 ``pytest>=3.3.2``, and 0.1/0.2 declare no pytest requirement at all — those two
    read from their sidecar, because their version JSON carries no ``requires_dist`` —
    which is why httpie/1's ``pytest==3.2.1`` lands on 0.2.

    The declarations come from :func:`declared_requires`, which reads the JSON first and
    falls back to the published metadata. A release no reading reaches is a
    :class:`BakeError`, and every caller answers one as "unknown" rather than as
    "nothing" (:data:`CASE_UNKNOWN`): a layer is never told a pin declares no
    requirements because nobody could read it. ``requires_python`` is the version JSON's
    own field and is read whether or not the declarations came from there.

    ``sidecar_of`` and ``wheel_of`` are injected so the fallback chain is a structural
    test, the same way :func:`pypi_release_files`'s callers inject theirs. Called only
    by the bake, like :func:`pypi_releases`.
    """
    url = PYPI_VERSION_JSON.format(name=name, version=version)
    try:
        with urllib.request.urlopen(url, timeout=PYPI_TIMEOUT_SECS) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise BakeError(
            f"cannot read metadata for {name}=={version} from {url}: {exc}"
        ) from exc

    requires = declared_requires(
        payload, name=name, version=version, sidecar_of=sidecar_of, wheel_of=wheel_of
    )
    requires_python = payload["info"].get("requires_python")
    return requires, str(requires_python) if requires_python else None


SDIST_MAX_BYTES = 64 * 1024 * 1024
"""How much of an sdist preflight is willing to download in order to look inside it.

[VANTA-WHY] The sdist has to be read for real, because nothing cheaper answers the
question. PyPI's classifiers were measured useless for it: ``asyncpg``, ``cffi``,
``cryptography``, ``numpy``, ``regex``, ``sqlalchemy``, ``yarl`` and fourteen more of
the dev split's sdists that compile an extension carry no ``Programming Language ::
C`` classifier at all, and ``info``/``urls`` name the *files* without ever listing what
is inside the archive. The cap is a fuse and not a judgement about size: 64 MB is
above every sdist either split pins, and a release past it is reported as not
inspected rather than guessed at.
"""


def _read_capped(response: Any, max_bytes: int) -> tuple[bytes, bool]:
    """The body, or the first ``max_bytes`` of it, and whether it was cut."""
    chunks: list[bytes] = []
    read = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            return b"".join(chunks), False
        if read + len(chunk) > max_bytes:
            return b"".join([*chunks, chunk[: max_bytes - read]]), True
        chunks.append(chunk)
        read += len(chunk)


def _archive_names(data: bytes) -> tuple[str, ...]:
    """The member names of a tar or zip archive held in memory.

    Streaming (``r|*``) on purpose: a truncated tar still yields the members it did
    contain, which is the difference between "these are this sdist's compiled
    sources" and "nothing was read at all".
    """
    names: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|*") as archive:
            for member in archive:
                names.append(member.name)
        return tuple(names)
    except (tarfile.TarError, EOFError, ValueError):
        pass
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return tuple(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return ()


def sdist_file_list(
    file: PyPIFile, *, max_bytes: int = SDIST_MAX_BYTES
) -> tuple[tuple[str, ...], bool]:
    """``(member names, truncated)`` for one sdist.

    An sdist always has members, so "no names read" is reported as truncated: the
    listing is not the whole archive either way, and the caller must not read an
    empty list as "this sdist compiles nothing".
    """
    try:
        with urllib.request.urlopen(file.url, timeout=PYPI_TIMEOUT_SECS) as response:
            data, cut = _read_capped(response, max_bytes)
    except (urllib.error.URLError, OSError) as exc:
        raise BakeError(f"cannot download {file.filename}: {exc}") from exc
    names = _archive_names(data)
    return names, cut or not names


SDIST_METADATA_MEMBERS = ("pyproject.toml", "setup.py")
"""The two files a release declares its **build** requirements in (PEP 518 + setuptools)."""


def _top_level_member(name: str, wanted: Sequence[str]) -> Optional[str]:
    """``name`` as one of ``wanted``, when it is an sdist's own top-level copy of it.

    Top-level only (``<name>-<version>/<member>``), because that is where an sdist puts
    its own metadata: a nested copy is a vendored one, and reading it would attribute
    another project's build requirements to this release.
    """
    parts = name.split("/")
    if len(parts) != 2:
        return None
    return parts[1] if parts[1] in wanted else None


def _archive_texts(data: bytes, wanted: Sequence[str]) -> dict[str, str]:
    """``{member: text}`` for the top-level members of a tar or zip archive in memory.

    The counterpart of :func:`_archive_names`, which keeps the *names* because the
    question it answers is "does this build compile anything". This one keeps the
    *contents*, because "what does this build need" is written inside two files and
    nowhere in the listing.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            texts: dict[str, str] = {}
            for member in archive.getmembers():
                base = _top_level_member(member.name, wanted)
                if base is None or not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                texts[base] = handle.read().decode("utf-8", errors="replace")
            return texts
    except (tarfile.TarError, EOFError, ValueError):
        pass
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            texts = {}
            for name in archive.namelist():
                base = _top_level_member(name, wanted)
                if base is None or base in texts:
                    continue
                texts[base] = archive.read(name).decode("utf-8", errors="replace")
            return texts
    except (zipfile.BadZipFile, OSError, KeyError):
        return {}


def sdist_metadata_texts(
    file: PyPIFile, *, max_bytes: int = SDIST_MAX_BYTES
) -> dict[str, str]:
    """An sdist's own ``pyproject.toml`` and ``setup.py``, as text.

    Downloaded and read, never executed (design C3): both files are parsed as data —
    ``tomllib`` for the first, ``ast`` for the second — by the same
    :func:`build_refs` :func:`bug_sources` hands the tree's two files to.
    """
    try:
        with urllib.request.urlopen(file.url, timeout=PYPI_TIMEOUT_SECS) as response:
            data, _ = _read_capped(response, max_bytes)
    except (urllib.error.URLError, OSError) as exc:
        raise BakeError(f"cannot download {file.filename}: {exc}") from exc
    return _archive_texts(data, SDIST_METADATA_MEMBERS)


def pypi_build_requirements(
    name: str,
    version: str,
    *,
    files_of: Callable[[str, str], Optional[tuple[PyPIFile, ...]]] = pypi_release_files,
    texts_of: Callable[[PyPIFile], dict[str, str]] = sdist_metadata_texts,
) -> tuple[DepRef, ...]:
    """What one release declares it needs in order to be **built**, from its sdist.

    [VANTA-WHY] A second reading of the same two files :func:`build_refs` reads for the
    *tree* — PEP 518's ``[build-system].requires`` and setuptools' ``setup_requires`` —
    asked of the release instead of the project, because it is the only reading that
    answers it. PyPI's ``requires_dist`` (and therefore the sidecar and the wheel
    ``METADATA`` that :func:`declared_requires` falls back to) carries *runtime*
    requirements and never says "this needs cffi to build". Measured on the split's own
    pins: ``cryptography==2.9.2``'s ``pyproject.toml`` declares
    ``cffi>=1.8,!=1.11.3; platform_python_implementation != 'PyPy'`` and
    ``bcrypt==3.1.7``'s declares ``cffi>=1.1; python_implementation != 'PyPy'``, and
    neither release's ``METADATA`` mentions cffi as a build requirement anywhere.

    An empty tuple for a release with no sdist: a release pip installs from a wheel is
    never built here, so its build requirements are never exercised and nothing can
    fetch them. ``setup_py`` is read with ``strict=False`` (:func:`build_refs`) — a
    dependency's Python-2-era ``setup.py`` is a reading that failed, not this bug's
    fault.

    ``files_of``/``texts_of`` are injected so the whole reading is structural, the same
    way :func:`preflight_pin`'s pair is. Called only by the bake, like the other PyPI
    readers.
    """
    files = files_of(name, version)
    if files is None:
        return ()
    sdists = [file for file in files if file.is_sdist]
    if not sdists:
        return ()
    texts = texts_of(sdists[0])
    return tuple(
        build_refs(
            texts.get("setup.py"),
            texts.get("pyproject.toml"),
            notes=[],
            strict=False,
        )
    )


# ---------------------------------------------------------------------------
# preflight: what each pin installs as, before any layer is built
# ---------------------------------------------------------------------------
PREFLIGHT_SCHEMA_VERSION = 1

INSTALLABLE_WHEEL = "wheel"
"""Class (a): a wheel that installs on this image's interpreter without compiling."""

SDIST_ONLY = "sdist"
"""Class (b): no usable wheel, but a source distribution pip will build."""

NO_DISTRIBUTION = "unavailable"
"""Class (c): nothing pip can install here — PyPI has no such release, or the release
ships only Windows/macOS wheels and no sdist."""

UNBUILDABLE = "unbuildable"
"""A §3.2 finding that is not one of the three classes: a build no reviewed table entry
could make work on this image (:data:`UNBUILDABLE_ON_PLATFORM`).

The pin *is* sdist-only, so its class letter stays (b) — that is what pip will try — and
this is the separate word for the fact that the attempt cannot succeed. It is a recorded
outcome like :data:`NO_DISTRIBUTION`, not a class of its own.
"""

PREFLIGHT_CLASS_LETTERS: Mapping[str, str] = {
    INSTALLABLE_WHEEL: "a",
    SDIST_ONLY: "b",
    NO_DISTRIBUTION: "c",
}
"""The §3.2 class letters, so the printed table and the JSON agree on the vocabulary."""

TARGET_ARCHITECTURES = ("aarch64", "arm64")
"""The platform tags' architecture suffix that this image's ``linux-aarch64`` is.

Both spellings, because wheels are on PyPI under both and they mean the same machine.
"""


def _cpython_tag(python_minor: str) -> str:
    """``"3.8"`` → ``"cp38"``, the Python tag that interpreter's wheels carry."""
    major, _, minor = python_minor.partition(".")
    return f"cp{major}{minor or '0'}"


def _wheel_tag_parts(filename: str) -> Optional[tuple[str, str, str]]:
    """``(python, abi, platform)`` from a wheel filename, or ``None`` for not a wheel.

    PEP 427's ``{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl``: the
    distribution name and the version may not contain a dash, so the last three
    dash-separated fields are the tags whether or not a build tag is present.
    """
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        return None
    return parts[-3], parts[-2], parts[-1]


def _aarch64_platform(platform: str) -> bool:
    """Whether a wheel's platform tag names this image's architecture.

    ``manylinux2014_aarch64`` and its PEP 600 spelling ``manylinux_2_17_aarch64`` are
    one tag written two ways and both count; so do ``manylinux1``/``manylinux2010``,
    ``musllinux_*_aarch64`` and the bare ``linux_aarch64``.

    The matched tag is recorded per pin rather than collapsed into a boolean — those
    are not all the same claim about this image, which is glibc and would not take a
    musl libc build. Nothing in either split is affected (musllinux postdates every
    pin in both) and the record is what keeps that from being an assumption.
    """
    parts = platform.split("_")
    if len(parts) < 2 or parts[-1] not in TARGET_ARCHITECTURES:
        return False
    return parts[0] in (
        "manylinux",
        "manylinux1",
        "manylinux2010",
        "manylinux2014",
        "musllinux",
        "linux",
    )


def _tag_set(field: str) -> tuple[str, ...]:
    """The alternatives one wheel tag field names: ``py2.py3`` → ``py2``, ``py3``.

    [VANTA-WHY] A dot in a wheel's python/abi/platform field is not part of the tag:
    PEP 425 makes it the *compressed set* of tags the wheel also claims, so
    ``py2.py3-none-any`` is ``py2-none-any`` **and** ``py3-none-any``. Reading the
    field as one literal tag — or requiring every alternative to be acceptable —
    puts ``six``'s own wheel outside ``six``'s own interpreter, which is how this
    was caught: ``six-1.15.0-py2.py3-none-any.whl`` came back ``unavailable``.
    """
    return tuple(field.split("."))


def wheel_install_tag(file: PyPIFile, *, python_minor: str) -> Optional[str]:
    """The ``python-abi-platform`` triple that makes this wheel installable, or ``None``.

    A reason and not a boolean: the record has to say **which** of a release's files
    answered, because "some wheel was fine" is not checkable against PyPI afterwards.
    The triple returned is the expanded one — the alternative that actually matched,
    not the field it was spelled in — so the answer can be compared with pip's own
    tag listing rather than translated by hand.

    [VANTA-WHY] The tag has to be read. Measured: a classifier that accepted any
    non-``none-any`` wheel put ``numpy 1.18.5``, ``scipy 1.4.1`` and ``cffi 1.14.0`` in
    class (a) although none of the three ships a ``cp38`` ``aarch64`` wheel — and the
    ``cffi`` bake then failed on ``ffi.h``, which is precisely the discovery preflight
    exists to move forward. The three accepted shapes, and why each is one:

    * ``cp38``/``cp38m`` with an aarch64 platform tag — this interpreter's own build;
    * ``abi3`` from ``cp30`` to ``cp38`` on an aarch64 platform tag — the stable ABI is
      forward-compatible, so a wheel built for 3.7 installs on 3.8 and one built for
      3.9 does not;
    * ``py3``/``py2.py3``/``py3N`` (N ≤ the minor) with ``none`` and ``any``, or with
      an aarch64 platform tag — pure Python, which compiles nothing either way.

    A ``py2``-only wheel is **not** installable here and falls through to the sdist:
    ``py2-none-any`` expands to itself alone, and there is no ``py2`` in the set this
    interpreter answers to. That is the same rule that takes ``py2.py3``, and it is
    why the alternatives are enumerated instead of the field string being compared.
    """
    parts = _wheel_tag_parts(file.filename)
    if parts is None:
        return None
    major, _, minor = python_minor.partition(".")
    limit = int(minor) if minor.isdigit() else 0
    expected = f"cp{major}{minor}"
    concrete = (expected, f"{expected}m")
    stable = {f"cp{major}{n}" for n in range(limit + 1)}
    pure = {f"py{major}"} | {f"py{major}{n}" for n in range(limit + 1)}
    for python in _tag_set(parts[0]):
        for abi in _tag_set(parts[1]):
            for platform in _tag_set(parts[2]):
                on_arch = platform != "any" and _aarch64_platform(platform)
                if platform != "any" and not on_arch:
                    continue
                if abi == "none":
                    if python in pure:
                        return f"{python}-{abi}-{platform}"
                elif abi == "abi3":
                    if on_arch and python in stable:
                        return f"{python}-{abi}-{platform}"
                elif platform != "any" and python == expected and abi in concrete:
                    return f"{python}-{abi}-{platform}"
    return None


SDIST_COMPILED_SUFFIXES = COMPILED_SOURCE_SUFFIXES + (".f", ".f77", ".f90", ".f95")
"""Compiled-source suffixes, plus Fortran.

[VANTA-WHY] Fortran is on the list although no tree in either split compiles any: the
sdists that do are exactly the ones whose read is skipped for size (``scipy``, and
numpy's BLAS half), and a rule that silently omitted Fortran would report "compiles
nothing" for the two packages most likely to need ``gfortran`` — the wrong direction,
because it would let a missing compiler be discovered at build time.
"""

SDIST_DATA_DIRS = frozenset(
    {
        "test",
        "tests",
        "test_data",
        "testdata",
        "testing",
        "doc",
        "docs",
        "example",
        "examples",
    }
)
"""Path components that mean an sdist shipped the file as data, not as source to build.

[VANTA-WHY] Measured on ``protego`` 0.1.16: its sdist carries four ``.cc`` files and
every one is under ``tests/test_data/`` — renamed corpora, not C++. Counting them puts
a pure-Python package on the "needs a C compiler" list, and a review table is only
worth reading if every entry on it is real.
"""


def compiled_sources(members: Sequence[str]) -> tuple[str, ...]:
    """The sdist members whose suffix says the build compiles them.

    Suffix-matched on the basename and filtered by :data:`SDIST_DATA_DIRS`. PyPI's own
    classifiers cannot answer this (see :data:`SDIST_MAX_BYTES`), so the archive's own
    file list is what does: it is cheap, and it is the release's own claim about itself.
    """
    return tuple(
        name
        for name in members
        if name.endswith(SDIST_COMPILED_SUFFIXES)
        and not SDIST_DATA_DIRS.intersection(name.split("/")[:-1])
    )


@dataclass(frozen=True)
class PinConflict:
    """Two requirements of one layer that cannot both be satisfied (§3.2).

    [VANTA-WHY] The record of §3.2's two cases, and it is deliberately one type for
    both directions they are found from: the preflight reads every pin's
    ``requires_dist`` and compares it against the layer's other pins, and the layer's
    own ``pip check`` reads what pip *actually installed*. Both produce the same
    finding — "this pin declares a requirement this layer's other pin refuses" — and
    the decision that follows is the same, so one shape and one spelling of
    :attr:`case` keep the two from drifting.

    ``case`` is which side of §3.2's rule the conflict falls on. ``"b"`` means **both**
    sides are BugsInPy's own exact pins (:data:`BUGSINPY_PIN_RULE`): the environment
    they were frozen from installed both with a warning, so
    :data:`LEGACY_RESOLVER_FLAG` reproduces that and the conflict is tolerated —
    recorded, printed, and not a failure. ``"a"`` is everything else, and it is a
    failure: at least one side came from a decision this harness made (the date rule,
    or pip's own choice of a transitive dependency), so a layer holding both is a layer
    whose environment is not the benchmark's. ``"d"`` is the one *absence* that is
    expected rather than a finding (:data:`CASE_DROPPED_DEPENDENCY`): a requirement the
    layer dropped on purpose, so the pin that declares it is installed ``--no-deps``
    and the requirement is legitimately missing. ``"?"`` is not a conflict at all but
    the *absence* of the reading: no document could be reached for one of the pins, so
    what it declares is unknown (:data:`CASE_UNKNOWN`).

    ``pins`` is the two ``name==version`` spellings the finding is about, in the order
    it names them. It is carried because both readings produce prose
    (``"httpcore==0.3.0 declares h11==0.8.*"``) and the preflight has to ask a data
    question of it — is either side a pin §3.2's rule dropped? — that a sentence cannot
    answer.
    """

    case: str
    requires: str
    """The declaration, e.g. ``httpcore==0.3.0 declares h11==0.8.*``."""
    held: str
    """What the layer has instead, e.g. ``h11==0.9.0`` — ``"nothing"`` if absent."""
    source: str
    """Where the reading came from: ``"pip check"`` or ``"PyPI metadata"``."""
    pins: tuple[str, ...] = ()
    """The ``name==version`` spellings the finding is between, e.g. ``("h11==0.9.0",)``."""

    @property
    def tolerated(self) -> bool:
        """Whether §3.2 lets this layer keep the finding: an expected absence, or both
        sides its own pins (:data:`CASE_BOTH_OWN_PINS`, :data:`CASE_DROPPED_DEPENDENCY`).
        """
        return self.case in (CASE_BOTH_OWN_PINS, CASE_DROPPED_DEPENDENCY)

    def describe(self) -> str:
        """One line a reader can act on, with the case and the verdict in it."""
        if self.case == CASE_UNKNOWN:
            return f"case (?) unknown — {self.requires}"
        if self.case == CASE_DROPPED_DEPENDENCY:
            return (
                f"case ({self.case}) {self.requires}: dropped dependency, by §3.2's "
                "audit — tolerated"
            )
        verdict = (
            "tolerated (both sides are BugsInPy's own pins, installed as written)"
            if self.tolerated
            else "not tolerated — a conflict with a pin this harness chose"
        )
        return f"case ({self.case}) {self.requires} but the layer has {self.held}: {verdict}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "requires": self.requires,
            "held": self.held,
            "source": self.source,
            "pins": list(self.pins),
            "tolerated": self.tolerated,
        }


CASE_BOTH_OWN_PINS = "b"
"""§3.2's case (b): both sides are BugsInPy's own exact pins, so it is tolerated."""

CASE_HARNESS_CHOICE = "a"
"""§3.2's case (a): one side is a version this harness chose, so the bake fails."""

CASE_DROPPED_DEPENDENCY = "d"
"""§3.2's dropped dependency: a requirement this layer dropped on purpose is absent.

[VANTA-WHY] This is the one ``pip check`` reading that must not be a failure, and the
one that would otherwise be one. A pin this layer installs may declare a name §3.2's
unresolvable-pin rule dropped (:func:`decide_unresolvable`); the pin is installed
``--no-deps`` so pip cannot resolve that name at the newest release of the day
(:func:`dropped_dependents`), and ``pip check`` then reports the missing requirement —
``pyopenssl 19.1.0 requires cryptography, which is not installed``. That reading is the
measure working, not failing: the drop is recorded in the provenance with the audit that
justified it, and the alternative is the layer holding a release the harness never chose.
Tolerated **by name only**: the requirement the reading names has to be one this layer
dropped, so a missing requirement that is anything else is still case (a)
(:data:`CASE_HARNESS_CHOICE`) and still fails the layer.
"""

CASE_UNKNOWN = "?"
"""Neither case: no reading reached one of the pins, so nothing could be checked.

[VANTA-WHY] It is recorded rather than skipped, because "checked and clean" and "could
not be checked" are different answers and only one of them is reassuring. It is not a
failure either: the layer's own ``pip check`` (:func:`pip_check_conflicts`) reads what
pip *did* install and is what the gate uses, so a pin whose declarations could not be
read — ``httpcore==0.3.0``, sdist-only, no sidecar — has still been checked by the
layer. ``httpcore==0.3.0``'s own ``h11==0.8.*`` is the measured example of why this
case cannot be treated as "declares nothing": the version JSON carries no
``requires_dist``, pip reads the requirement from the sdist at resolution, and only the
layer's reading finds it.
"""


def own_pin_names(deps: Mapping[str, Any]) -> frozenset[str]:
    """The canonical names this layer installs as BugsInPy's own exact pin.

    [VANTA-WHY] Read off ``pin_rule`` rather than re-derived from the declarations: the
    provenance already answered "was this pin written by BugsInPy's files or chosen
    here", and :data:`BUGSINPY_PIN_RULE` is the one spelling of that answer
    (:func:`reference_pin` writes it). A second predicate would let a conflict be
    tolerated in the record and refused in the layer.

    This is what decides :attr:`PinConflict.case` for a reading that only names
    distributions — pip's ``pip check`` prose says nothing about where a version came
    from — so it is the layer's own-dependency names, and *not* the interpreter layer's
    harness pins: ``pytest`` is provisioned, not pinned as written, and a conflict
    against it is not one the benchmark's own environment had.
    """
    return frozenset(
        canonicalize_name(str(dep.get("name")))
        for dep in deps.get("dependencies") or ()
        if dep.get("pin_rule") == BUGSINPY_PIN_RULE
    )


def pin_narrowings(provenance: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """``(pin, reason)`` for every date-rule choice a pin in the same layer moved.

    [VANTA-WHY] Read off the provenance's own record — ``narrowed_by`` and ``refused``
    (:attr:`ResolvedDep`, and ``narrowed_by`` on :attr:`BuildPin` too) — rather than
    re-derived, for the same reason :func:`own_pin_names` reads ``pin_rule``: the answer
    already exists and one spelling of it is what keeps the record and the reading from
    disagreeing. The preflight's job is to put it in front of a reader *before* the layer
    is built, because a pin that is not simply "the newest release on or before the date"
    is the one thing a reader cannot see in the pin list itself — ``moto==1.3.7`` beside
    ``luigi/1``'s ``idna==2.9`` looks arbitrary until the reason is next to it. Nothing
    here is a finding to act on: a narrowed pin is a decision already taken, which is why
    this is printed as its own block and not counted against the verdict.

    Both halves of the layer are read because both can move for the same reason: a
    *dependency* is narrowed by what the layer's pins declare about it
    (:func:`_narrowing_requirements`), and a *build tool* by what the layer's pins need
    to build themselves (:func:`narrow_build_pins`). They are recorded in one shape —
    ``{"pin": <the pin that asked>, "requires": <its clause>}`` — so one loop reads both,
    and the moved build tool prints under NARROWED like any other moved pin rather than as
    an EARLY ordering.
    """
    out: list[tuple[str, str]] = []
    for group in ("build_pins", "dependencies"):
        for dep in provenance.get(group) or ():
            pin = str(dep.get("pin") or "")
            if not pin:
                continue
            for claim in dep.get("narrowed_by") or ():
                out.append((pin, f"{claim.get('pin')} declares {claim.get('requires')}"))
            for claim in dep.get("refused") or ():
                out.append(
                    (
                        pin,
                        f"{claim.get('pin')} declares {claim.get('requires')}, which "
                        f"{claim.get('held')} does not satisfy",
                    )
                )
    return tuple(out)


def _conflict_case(*names: str, own: Container[str]) -> str:
    """§3.2's case letter for a conflict between ``names`` — (b) only when all are own."""
    if names and all(canonicalize_name(name) in own for name in names):
        return CASE_BOTH_OWN_PINS
    return CASE_HARNESS_CHOICE


def pip_check_readings(output: str) -> tuple[str, ...]:
    """The layer's own ``pip check`` lines, lifted out of its build output.

    The recipe echoes :data:`PIP_CHECK_OUTPUT` on :data:`PIP_CHECK_PREFIX`-marked lines
    because a checkpoint is a filesystem (:data:`DEPS_DIRNAME`); this is the host's half
    of that, and it is the only place the marker is stripped back off.
    """
    return tuple(
        line[len(PIP_CHECK_PREFIX) :].strip()
        for line in output.replace("\r", "\n").splitlines()
        if line.startswith(PIP_CHECK_PREFIX)
    )


_REQUIREMENT_PROSE = re.compile(
    r"^(?P<package>\S+)\s+(?P<package_version>\S+)\s+"
    r"(?:has requirement|requires)\s+(?P<requires>.+?)"
    r"(?:,\s*but you have\s+(?P<held>\S+)\s+(?P<held_version>\S+?))?"
    r"(?:,\s*which is not installed)?\.?$"
)
"""pip's ``pip check`` sentences, as one pattern.

Two of them reach here and they differ only in the second half: ``A 1.0 has requirement
B<2, but you have B 2.0.`` and ``A 1.0 requires B<2, which is not installed.`` Measured
against pip 21.3.1's own ``check`` command. Both names come with their version — pip
spells a distribution the way ``pip freeze`` does, ``h11 0.9.0`` — and both are captured
because a finding is between two *versions*, not two names: the record spells them
``name==version`` like every other pin in this file.

Three details are the pattern rather than the shape of one sentence, and each was a
measured misreading:

* ``requires`` is ``.+?`` and not ``[^,]+?``: pip prints the specifier *as declared*, and
  ``B>=0.8,<0.9`` carries the comma this file splits its two halves on. A requirement that
  cannot contain one is what ``[^,]+?`` would have read as an unreadable line — case (a),
  refused — over a conflict that is the benchmark's own.
* ``held_version`` is lazy and the full stop is the pattern's own: with ``\\S+`` the
  version came back as ``0.9.0.``, so a finding named a pin ``h11==0.9.0.`` that matches
  nothing in the layer's install list and nothing in the dropped set.
* ``requires`` is lazy for the same reason: it stops at the first place the tail of the
  sentence can match, so the declaration is the requirement and not the requirement plus
  the clause about what the layer has.

Anything that does not match is kept as written and treated as case (a): a line this
cannot read is not a line that says the conflict is tolerable.

:data:`PIP_CHECK_CLEAN` is the one exception, and it is the exception that keeps this
rule from refusing a clean layer — it says there is no conflict rather than that there is
one this cannot read (:func:`pip_check_conflicts`).
"""


def _required_name(text: str) -> Optional[str]:
    """The canonical distribution name at the head of a requirement pip spelled out.

    ``cryptography`` and ``cryptography>=2.8`` have to answer the same, because which
    one pip writes depends on the requirement and not on the finding: its
    ``which is not installed`` sentence names the requirement as declared. A string
    that is not a requirement at all yields ``None``, and the caller then treats the
    reading as case (a) rather than deciding it is tolerable.
    """
    try:
        return canonicalize_name(Requirement(text).name)
    except InvalidRequirement:
        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", text.strip())
        return canonicalize_name(match.group(0)) if match else None


def pip_check_conflicts(
    readings: Iterable[str],
    *,
    own: Container[str],
    dropped: Container[str] = (),
) -> tuple[PinConflict, ...]:
    """Judge a layer's ``pip check`` readings by §3.2's rule, in order.

    ``own`` is the layer's :func:`own_pin_names`. Every conflict is returned, tolerated
    or not — the record is what the deps JSON and the bake output carry — and
    :attr:`PinConflict.tolerated` is the verdict the caller acts on.

    ``dropped`` is the canonical names §3.2 dropped for this layer, and it is what makes
    an *absence* tolerable instead of a failure: a reading that names no held version
    (``… requires cryptography, which is not installed``) is
    :data:`CASE_DROPPED_DEPENDENCY` when the requirement it names is a dropped name, and
    case (a) — a failure — for every other missing requirement. Nothing else about the
    reading changes, so a missing requirement this layer did *not* drop still stops the
    layer.

    A reading of :data:`PIP_CHECK_CLEAN` contributes nothing: it is pip saying it found
    no conflict, and a layer that says that has passed the gate.
    """
    names = {canonicalize_name(name) for name in dropped}
    out: list[PinConflict] = []
    for line in readings:
        text = line.strip()
        if not text or text == PIP_CHECK_CLEAN:
            continue
        match = _REQUIREMENT_PROSE.match(text)
        if match is None:
            out.append(
                PinConflict(
                    case=CASE_HARNESS_CHOICE,
                    requires=text,
                    held="nothing",
                    source="pip check",
                )
            )
            continue
        package = match.group("package")
        package_version = match.group("package_version")
        held = match.group("held")
        held_version = match.group("held_version")
        declared = match.group("requires").strip()
        if held is None:
            # A reading with no held version is an *absence*, not a conflict between two
            # pins, so §3.2's case (b) cannot apply to it — there is only one side. The
            # one absence this layer may have is a name it dropped on purpose
            # (:data:`CASE_DROPPED_DEPENDENCY`); every other missing requirement fails the
            # layer, whatever the name that declared it.
            required = _required_name(declared)
            if required is not None and required in names:
                out.append(
                    PinConflict(
                        case=CASE_DROPPED_DEPENDENCY,
                        requires=f"{package}=={package_version} declares {declared}",
                        held="nothing",
                        source="pip check",
                        pins=(f"{package}=={package_version}",),
                    )
                )
            else:
                out.append(
                    PinConflict(
                        case=CASE_HARNESS_CHOICE,
                        requires=f"{package}=={package_version} declares {declared}",
                        held="nothing",
                        source="pip check",
                        pins=(f"{package}=={package_version}",),
                    )
                )
            continue
        out.append(
            PinConflict(
                case=_conflict_case(
                    *(name for name in (package, held) if name), own=own
                ),
                requires=f"{package}=={package_version} declares "
                f"{match.group('requires').strip()}",
                held=f"{held}=={held_version}" if held else "nothing",
                source="pip check",
                pins=tuple(
                    spelling
                    for spelling in (
                        f"{package}=={package_version}",
                        f"{held}=={held_version}" if held else "",
                    )
                    if spelling
                ),
            )
        )
    return tuple(out)


def metadata_conflicts(
    pins: Sequence[tuple[str, str]],
    *,
    own: Container[str],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> tuple[PinConflict, ...]:
    """Every conflict §3.2's cases can be read from the pins' own ``requires_dist``.

    ``pins`` is one layer's ``(name, version)`` pairs (:func:`layer_pins`), and each
    pin's ``requires_dist`` is compared against the *other* pins: a pin whose metadata
    declares a requirement on a name this layer also pins, in a version that
    requirement excludes, is the conflict — no install needed, which is the point of
    doing it in the preflight.

    ``metadata_of`` is injected so the check is a structural test, and it is also why a
    release nothing could be read for is *recorded* (:data:`CASE_UNKNOWN`) rather than
    guessed at or passed over: :func:`pypi_release_metadata` raises when neither the
    version JSON, a wheel nor a sidecar carries the declarations
    (:func:`declared_requires`), the check cannot say anything about that pin, and "no
    conflicts found" must not be what a reader takes from an answer nobody was able to
    compute. The layer's own ``pip check`` is the backstop for those pins, and it is the
    reading the gate uses.
    """
    versions = {
        canonicalize_name(name): version for name, version in pins
    }
    out: list[PinConflict] = []
    for name, version in pins:
        try:
            requires, _ = metadata_of(name, version)
        except BakeError as exc:
            out.append(
                PinConflict(
                    case=CASE_UNKNOWN,
                    requires=f"{name}=={version} — {exc}",
                    held="unknown",
                    source="PyPI metadata",
                    pins=(f"{name}=={version}",),
                )
            )
            continue
        for raw in requires:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                continue
            if requirement.marker is not None:
                continue
            other = canonicalize_name(requirement.name)
            held = versions.get(other)
            if held is None:
                continue
            try:
                allowed = Version(held) in requirement.specifier
            except InvalidVersion:
                continue
            if allowed:
                continue
            out.append(
                PinConflict(
                    case=_conflict_case(name, other, own=own),
                    requires=f"{name}=={version} declares {requirement.specifier}",
                    held=f"{other}=={held}",
                    source="PyPI metadata",
                    pins=(f"{name}=={version}", f"{other}=={held}"),
                )
            )
    return tuple(out)


def pip_check_failure(conflicts: Sequence[PinConflict]) -> ExecResult:
    """A failed build's result, synthesised from the conflicts that stopped the layer.

    The layer installed and its marker was written; what failed is the *gate* after it
    (:func:`pip_check_conflicts`). It is reported through the same path a build failure
    takes — every failed layer, one line each, its own log kept — so a conflict names
    itself where the reader is already looking, rather than becoming a second kind of
    failure with its own reporting. The leading ``ERROR:`` is the same prefix pip's own
    fatal lines carry, which is what the one-line summary picks out
    (:func:`last_error_line`).
    """
    body = "\n".join(conflict.describe() for conflict in conflicts)
    return ExecResult(
        stdout="",
        stderr=f"ERROR: pip check found a conflict this layer may not keep:\n{body}",
        exit_code=1,
        timed_out=False,
    )


EASY_INSTALL_MARKERS = ("Searching for ", "Best match: ")
"""What setuptools' ``easy_install`` prints when it goes looking for a distribution.

Both, and not one: ``Searching for cffi`` is the first line it writes and ``Best
match: cffi 2.1.1`` is the one that names the version it settled on, so a single
marker would either miss a fetch that failed between the two or report a search whose
result is not in the log. They are also the two lines that cannot be anything else —
pip does not write either, and the recipe writes neither, so a match is a fetch.

Measured, and the reason the pair is spelled out rather than grepped loosely: on
``fastapi/1`` the layer installed ``cffi==1.14.0`` and ``cryptography==2.9.2`` declared
cffi in ``[build-system].requires``; with ``--no-build-isolation`` setuptools resolved
that declaration itself through ``easy_install``, which reached PyPI and installed
**today's** cffi 2.1.1 beside the pin. ``Installing cffi-2.1.1`` was the line that gave
it away and it is deliberately not a marker: pip's own ``Installing collected
packages:`` shares that prefix, and a marker that also matches pip's success line
would fail every layer.
"""


def easy_install_fetches(output: str) -> tuple[str, ...]:
    """The lines that say ``easy_install`` fetched a distribution during this build.

    Empty is the pass, and it is the answer a correct layer gives: every build
    requirement this layer's pins declare is either already installed — the early pass
    :func:`early_install_pins` orders, or a pin pip put there — or pip refuses to build
    at all. A match means something was fetched from outside the pin list, and which
    release it was is the point: ``cffi 2.1.1`` installed under a layer that pins
    ``cffi==1.14.0`` is a layer whose environment is not the one its provenance
    describes.
    """
    markers = EASY_INSTALL_MARKERS
    return tuple(
        line
        for raw in output.replace("\r", "\n").splitlines()
        if (line := raw.strip()).startswith(markers)
    )


def easy_install_failure(lines: Sequence[str]) -> ExecResult:
    """A failed build's result, synthesised from an ``easy_install`` fetch (§3.2).

    Through the same path :func:`pip_check_failure` uses and for the same reason: the
    install itself succeeded — the layer installed and its marker is there — so what
    failed is the gate after it, and reporting it as a build failure keeps one kind of
    failure in the summary instead of two. The leading ``ERROR:`` is the prefix
    :func:`last_error_line` picks out, so the one-line entry names this rather than the
    last line of pip's output.
    """
    body = "\n".join(lines)
    return ExecResult(
        stdout="",
        stderr=(
            "ERROR: setuptools' easy_install fetched a distribution while this layer "
            "was being built, so the layer is not the one its pin list describes "
            "(§3.2). Either the fetched package is a build requirement a pin in this "
            "layer satisfies — and it must be installed before the build that wants it "
            "(:func:`early_install_pins`) — or nothing here pins it and the layer "
            "cannot be built from its own pins:\n" + body
        ),
        exit_code=1,
        timed_out=False,
    )


def freeze_readings(output: str) -> tuple[str, ...]:
    """The layer's own ``pip freeze`` lines, lifted out of its build output.

    The host's half of :func:`_freeze_echo` and the only place :data:`PIP_FREEZE_PREFIX`
    is stripped back off, the same way :func:`pip_check_readings` is for
    :data:`PIP_CHECK_PREFIX`.
    """
    prefix = PIP_FREEZE_PREFIX
    out: list[str] = []
    for raw in output.replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        reading = line[len(prefix) :].strip()
        if reading:
            out.append(reading)
    return tuple(out)


def unpinned_freeze(
    output: str,
    *,
    pins: Sequence[tuple[str, str]],
    provided: Container[str] = (),
    dropped: Container[str] = (),
) -> tuple[str, ...]:
    """The distributions a layer installed that no pin of it names (§3.2).

    [VANTA-WHY] The closure walk (:func:`transitive_pins`) pins what a *release*
    declared, and this is what says the walk was complete. A name in the built layer
    that is in neither the pin list nor the interpreter layer's provision is one pip
    resolved by itself, out of the PyPI of the day — ``fastapi/1``'s
    ``exceptiongroup==1.3.1``, which no file, no pin and no walk had asked for. A pin
    list can only be as honest as the environment it describes, and the freeze is the
    layer's own account of what is in it, taken after the install rather than derived
    from the pins (`--use-deprecated=legacy-resolver` installs whatever it is given,
    so nothing before this point sees what pip added).

    Compared by **name**, through :func:`canonicalize_name` because ``pip freeze``
    spells a project per PEP 503 while a pin keeps the spelling its file used. The
    list's versions are the exact ``==`` pins the recipe installs, so a name on the
    list is at the version on the list; a *pinned* name installed at some other
    version is a different reading and a different gate
    (:func:`easy_install_fetches`), and this one deliberately does not double as it.

    ``provided`` is the names the interpreter layer supplies
    (:func:`transitive_pins`): every layer stacked on it inherits them, no bug's pin
    list mentions them, and without the exemption every layer would fail on
    ``pytest``'s own neighbours. It is a prediction of that environment and not a
    reading of it, which is what the interpreter layer's own freeze is kept for
    (:func:`_freeze_echo`).

    ``dropped`` is the one name this gate fails on **whatever** the pin list and the
    provision say, and it is the second reader of §3.2's dropped-dependency pass
    (:func:`dropped_dependents`). A dropped name is in neither ``pins`` nor ``provided``
    (:func:`drop_unresolvable_deps` never asks about a package the interpreter layer
    carries), so it would be reported above anyway — but only *by accident*, and only for
    a layer that happens to have a pin list. This makes it the rule it is: the layer's
    recipe installs the pins that declare a dropped name with ``--no-deps`` precisely so
    that pip cannot resolve it, so a dropped name in the freeze at **any** version means
    the pass did not take, and that is a failure even when the version would otherwise be
    exempt. Without it the leak the pass closes could reopen silently, because a leaked
    release is a name the pin list does not contain and would look like any other
    finding.

    A line that is not a distribution — pip's editable-install forms start with ``-``
    or ``#`` — is passed over: it is not a name to compare, and nothing here can say
    what it provides.
    """
    named = {canonicalize_name(name) for name, _ in pins} | {
        canonicalize_name(name) for name in provided
    }
    dropped_names = {canonicalize_name(name) for name in dropped}
    out: list[str] = []
    for reading in freeze_readings(output):
        if reading.startswith(("-", "#")):
            continue
        name = reading.partition("==")[0].partition(" @ ")[0].strip()
        if not name:
            continue
        if canonicalize_name(name) in dropped_names:
            out.append(reading)
            continue
        if canonicalize_name(name) in named:
            continue
        out.append(reading)
    return tuple(out)


def freeze_failure(lines: Sequence[str]) -> ExecResult:
    """A failed build's result, synthesised from the distributions no pin names (§3.2).

    Through the same path :func:`pip_check_failure` and :func:`easy_install_failure`
    use, and for the same reason: the layer installed and its marker was written, so
    what failed is the gate after it, and reporting it here keeps one kind of failure
    in the summary. The leading ``ERROR:`` is the prefix :func:`last_error_line` picks
    out, so the one-line entry names this rather than pip's last line.
    """
    body = "\n".join(lines)
    return ExecResult(
        stdout="",
        stderr=(
            "ERROR: this layer installed distributions its pin list does not name, so "
            "the layer is not the environment its provenance describes (§3.2). Every "
            "name has to come from a pin of this layer or from the interpreter layer "
            "it is stacked on; these came from neither, and pip resolved them itself "
            "at the newest release the day allowed:\n" + body
        ),
        exit_code=1,
        timed_out=False,
    )


@dataclass(frozen=True)
class PreflightPin:
    """One ``name==version`` pin and what PyPI says it can be installed as."""

    name: str
    version: str
    kind: str
    reason: str
    matched_tag: Optional[str] = None
    """The wheel tag that put this pin in class (a), when one did."""
    compiled: tuple[str, ...] = ()
    """The sdist members that look like sources a build compiles (class (b))."""
    sdist_truncated: bool = False
    """Whether the sdist was not read whole, so :attr:`compiled` may be incomplete."""
    gap: Optional[str] = None
    """Why this build is unreviewed, when it compiles and no table entry covers it."""
    unbuildable: Optional[str] = None
    """Why this sdist cannot be built on this image, when it cannot (a review, not a gap).

    A different answer from :attr:`gap` and they are mutually exclusive: a gap is a
    review someone has not done yet, and this is a build no entry in the reviewed table
    could make work. Both are recorded rather than passed over, because both are things
    the bake would otherwise discover as one bug's build error (§3.2).
    """

    @property
    def letter(self) -> str:
        """The §3.2 class letter, for a line a human reads."""
        return PREFLIGHT_CLASS_LETTERS.get(self.kind, "?")

    @property
    def compiles(self) -> bool:
        """Whether this sdist carries sources a build would compile.

        The one predicate the review table and the gap report both key off, so "needs a
        system library" cannot mean two different things in two places.
        """
        return self.kind == SDIST_ONLY and bool(self.compiled)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "pin": f"{self.name}=={self.version}",
            "class": self.letter,
            "kind": self.kind,
            "reason": self.reason,
        }
        if self.matched_tag is not None:
            data["matched_tag"] = self.matched_tag
        if self.kind == SDIST_ONLY:
            data["compiled_sources"] = list(self.compiled)
            data["sdist_truncated"] = self.sdist_truncated
        if self.gap is not None:
            data["gap"] = self.gap
        if self.unbuildable is not None:
            data["unbuildable"] = self.unbuildable
        return data


def preflight_pin(
    name: str,
    version: str,
    *,
    python_minor: str,
    files_of: Callable[[str, str], Optional[tuple[PyPIFile, ...]]] = pypi_release_files,
    sdist_of: Callable[[PyPIFile], tuple[tuple[str, ...], bool]] = sdist_file_list,
) -> PreflightPin:
    """Classify one pin as §3.2 class (a), (b) or (c).

    A wheel wins over an sdist when both are listed, because that is what pip picks.
    ``files_of`` returning ``None`` (PyPI has no such release) and a release that lists
    no file at all are both class (c) and get different reasons: the first is a pin
    that names a version nobody ever uploaded, the second a release that was uploaded
    without anything in it, and the remedy for a reader is different.
    """
    files = files_of(name, version)
    if files is None:
        return PreflightPin(
            name=name,
            version=version,
            kind=NO_DISTRIBUTION,
            reason=(
                "PyPI has no such release: the pin names a version that was never "
                "uploaded, so nothing can install it"
            ),
        )
    for file in files:
        tag = wheel_install_tag(file, python_minor=python_minor)
        if tag is not None:
            return PreflightPin(
                name=name,
                version=version,
                kind=INSTALLABLE_WHEEL,
                reason=f"installs from {file.filename} ({tag})",
                matched_tag=tag,
            )

    sdists = [file for file in files if file.is_sdist]
    if not sdists:
        wheels = [file for file in files if file.is_wheel]
        detail = (
            f"{len(wheels)} wheel(s) and none of them installable here"
            if wheels
            else "no files listed for it at all"
        )
        return PreflightPin(
            name=name,
            version=version,
            kind=NO_DISTRIBUTION,
            reason=(
                f"nothing pip can install on {_cpython_tag(python_minor)}/linux-aarch64: "
                f"{detail}, and no source distribution to build instead"
            ),
        )

    sdist = sdists[0]
    members, truncated = sdist_of(sdist)
    compiled = compiled_sources(members)
    review = _review_of(name, version, compiled=compiled, truncated=truncated)
    return PreflightPin(
        name=name,
        version=version,
        kind=SDIST_ONLY,
        reason=_sdist_reason(
            name,
            sdist=sdist,
            python_minor=python_minor,
            compiled=compiled,
            truncated=truncated,
            review=review,
        ),
        compiled=compiled,
        sdist_truncated=truncated,
        **review,
    )


def _review_note(name: str) -> str:
    """What the reviewed table says this build needs, as a clause a reader can search.

    ``reviewed: the toolchain alone`` is the empty tuple's clause and says the review
    happened — "no entry" and "an empty entry" must not read the same, because only the
    second one closes a gap (:func:`_review_of`).
    """
    packages = SDIST_SYSTEM_PACKAGES.get(canonicalize_name(name))
    if not packages:
        return "reviewed: the toolchain alone"
    return "reviewed: needs " + ", ".join(packages)


def _sdist_reason(
    name: str,
    *,
    sdist: PyPIFile,
    python_minor: str,
    compiled: Sequence[str],
    truncated: bool,
    review: Mapping[str, Optional[str]],
) -> str:
    """Why this pin is class (b), and what the reviewed table says about its build.

    [VANTA-WHY] The reason quotes the *review*, not the file list. ``compiled_sources``
    is a raw archive listing and it counts files no Linux build touches — ``distlib``
    0.3.0's only entry is ``PC/launcher.c`` (the Windows launcher) and ``cffi``'s
    include ``demo/*.c`` — so printing its sample made a Windows-only file read as the
    reason a build needs a system library. The count is still recorded per pin
    (``compiled_sources`` in the record); the authority for "what does this build need"
    is :data:`SDIST_SYSTEM_PACKAGES`, and a reader who wants to disagree with a line
    here argues with the table, not with a filename heuristic nothing could maintain.
    """
    head = (
        f"no {_cpython_tag(python_minor)} aarch64 wheel; pip builds "
        f"{sdist.filename} from source"
    )
    if review.get("unbuildable") is not None:
        return head
    if review.get("gap") is not None:
        return head + ", and no reviewed-table entry says what that build needs"
    if not compiled:
        detail = (
            "its file list was read only in part, so what it compiles is unknown"
            if truncated
            else "it carries no compiled sources, so it needs no compiler"
        )
        return f"{head}, and {detail}"
    return f"{head} — {_review_note(name)}"


def _review_of(
    name: str, version: str, *, compiled: Sequence[str], truncated: bool
) -> dict[str, Optional[str]]:
    """The reviewed table's verdict on one sdist build, as :class:`PreflightPin` fields.

    Three answers and each is a decision rather than a fallback:

    * a :data:`UNBUILDABLE_ON_PLATFORM` entry **whose bound covers ``version``** — the
      reason, because no system package would make this build work. Checked first: it is
      a fact about the package, and a gap line on top of it would read as if a review
      could close it. The table is version-scoped (:class:`PlatformLimit`) because
      ``cryptography`` has two impossible builds and which one a pin runs into is the
      pin's version; a version outside every clause is not claimed by any of them and
      falls through to the gap report below, which is the answer that asks for a review
      rather than inventing one;
    * a :data:`SDIST_SYSTEM_PACKAGES` entry — nothing to report. The entry may be an
      empty tuple ("this compiles, against nothing but libc and the layer's compiler"),
      which is a review result and not a missing entry;
    * neither, **and the build compiles something** — a named gap. A package whose sdist
      compiles nothing needs no entry: there is no library to name. ``truncated`` is not
      part of this test for the same reason it does not clear the count: a listing that
      was cut short *and* already showed compiled sources is a build that compiles, and
      that is the case the table most needs an entry for (``scipy`` and numpy's BLAS half
      are read no other way). A truncated read that showed **no** compiled source is the
      one case left alone — "unreviewed" cannot be said of a listing nobody finished.

    ``compiled`` only decides *whether to ask*; the table decides the answer, and the
    reason text never quotes the file list (:func:`_sdist_reason`). The list over-counts
    by construction — it is a suffix match over the whole archive, so it sees
    ``distlib``'s ``PC/launcher.c`` and ``cffi``'s ``demo/*.c`` — and the fix for that is
    an entry recording what the build really needs, which is what the table is.

    Looked up by :func:`packaging.utils.canonicalize_name`, so a pin spelled
    ``Zope.Interface`` finds the table's ``zope-interface``: the names on both sides are
    distribution names, and PEP 503 makes those one name.
    """
    canonical = canonicalize_name(name)
    for limit in UNBUILDABLE_ON_PLATFORM.get(canonical, ()):
        if limit.covers(version):
            return {"gap": None, "unbuildable": limit.reason}
    if canonical in SDIST_SYSTEM_PACKAGES or not compiled:
        return {"gap": None, "unbuildable": None}
    return {
        "gap": (
            f"unreviewed: {name} compiles from its sdist and SDIST_SYSTEM_PACKAGES has "
            "no entry for it, so which system library that build needs is unknown — add "
            "it to the table (the py3.8 layer's list is computed from the table) before "
            "baking, or the bake discovers it as this bug's build error"
        ),
        "unbuildable": None,
    }


def _pinned_parts(pin: str) -> Optional[tuple[str, str]]:
    """``(name, version)`` from an install spec, or ``None`` when it is not one.

    Every pin the layers carry is written by :func:`resolve_pins` or
    :func:`resolve_build_pins` as ``name==version``. Anything else is a shape this
    preflight does not recognise and must not guess about.
    """
    name, sep, version = pin.partition("==")
    if not sep or not name or not version:
        return None
    return name, version


def pinned_parts(*specs: str) -> tuple[tuple[str, str], ...]:
    """``(name, version)`` for each ``name==version`` spec, de-duplicated, in order.

    [VANTA-WHY] A spec that is not an exact pin is dropped rather than guessed at
    (:func:`_pinned_parts`), and a repeated one is dropped because a build requirement
    the dependency set also pins *is* that pin (:attr:`BuildPin.dependency_pin`) —
    counting it twice would report one package as two.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pin in specs:
        parts = _pinned_parts(pin)
        if parts is None or parts in seen:
            continue
        seen.add(parts)
        out.append(parts)
    return tuple(out)


def layer_pins(layer: BugDeps) -> tuple[tuple[str, str], ...]:
    """The layer's ``(name, version)`` pins, build pins first and with no repeats.

    Build pins first because that is the order the recipe installs them (§3.2).
    """
    return pinned_parts(*layer.build_pins, *layer.installs)


def constraint_pins(*spec_groups: Sequence[str]) -> tuple[str, ...]:
    """The layer's whole pin list, as one pip constraints file's lines (§3.2).

    Every group the recipe installs by — the toolchain and the declared build requirements
    (``build_pins``), the early pins, and the resolved dependency pins — because the file
    has to hold the *layer's* pins and not one install's: the failure it exists to prevent
    is a step that resolved a name no pin of this layer names (:data:`CONSTRAINTS_FILE`),
    and a step cannot be constrained by a list it is not given. A spec that is not an exact
    ``name==version`` is dropped rather than guessed at, and a repeated one is dropped too
    (:func:`pinned_parts`), so the file is a set of pins and not a set of spellings.
    """
    specs: list[str] = []
    for group in spec_groups:
        specs.extend(group)
    return tuple(f"{name}=={version}" for name, version in pinned_parts(*specs))


def constraint_lines(*spec_groups: Sequence[str]) -> list[str]:
    """The recipe lines that write the constraints file, ``[]`` when there are no pins.

    Empty for a recipe with nothing to install, so such a layer keeps the recipe it had and
    no install carries a ``-c`` pointing at a file that was never written.
    """
    constraints = constraint_pins(*spec_groups)
    if not constraints:
        return []
    return [
        f"cat > {CONSTRAINTS_FILE} <<'{CONSTRAINTS_MARKER}'",
        "\n".join(constraints),
        CONSTRAINTS_MARKER,
    ]


@dataclass(frozen=True)
class EarlyInstall:
    """A build requirement of one pin that another pin in the same layer satisfies.

    [VANTA-WHY] The record of one D1 finding, and it is a *pair* of pins rather than a
    fact about one: ``cryptography==2.9.2``'s ``[build-system].requires`` names cffi,
    and whether that name is a problem depends entirely on whether the same layer also
    holds a cffi pin — if it does, ``--no-build-isolation`` needs it installed
    *before* the build that wants it, and if it does not, setuptools' ``easy_install``
    fetches today's cffi instead (§3.2). So the finding names both sides, and the
    declaration itself, because "cffi" alone would not say which release asked for it
    or in what range.
    """

    pin: str
    """The pin whose *build* needs it, as ``name==version``."""
    requirement: DepRef
    """The declaration as the release's own metadata wrote it, with its source."""
    satisfied_by: str
    """The pin in the same layer that satisfies it, as ``name==version``."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pin": self.pin,
            "requires": self.requirement.spec,
            "source": self.requirement.source,
            "satisfied_by": self.satisfied_by,
        }


def early_install_pins(
    built: Sequence[tuple[str, str]],
    pinned: Sequence[tuple[str, str]],
    *,
    build_requires_of: Callable[[str, str], Sequence[DepRef]],
    python_minor: str,
    marker_skipped: Optional[list[SkippedLine]] = None,
) -> tuple[EarlyInstall, ...]:
    """The pins a source build in this layer needs installed **before** it runs.

    ``built`` is the pins pip will compile here and ``pinned`` every pin the layer
    holds (:func:`pinned_parts`) — two arguments and not one, because the two halves
    of the finding come from different places: what a pin's build *needs* is a fact
    about that release's sdist, and which version of it the layer *has* is the pin
    list. ``built`` is the caller's answer to "will pip build this here", because only
    a build runs ``setup.py`` at all: ``bcrypt==3.1.7`` declares cffi in
    ``[build-system].requires`` and installs from an aarch64 ``abi3`` wheel, so its
    declaration is never exercised and there is nothing to install early for it
    (measured on ``fastapi/1``'s log).

    A requirement the layer does **not** pin is not this function's business and is
    left to :func:`run_bake_actions`'s ``easy_install`` guard, which fails the layer
    that fetches one: what this returns is an ordering decision about pins the layer
    already holds, and it cannot invent a version for a name nothing pinned.

    The held pin must **satisfy** the requirement and not merely share its name
    (:func:`_specifier_accepts`), which is §3.2's build-requirement-wins rule read from
    this side: a name the layer pins too low is not an ordering finding but a pin that
    has to move (:func:`narrow_build_pins`), and it is printed as NARROWED rather than
    here. This is the one check both readers share — ``bake`` and the preflight
    (:func:`preflight_layers`) call this same function — so "every EARLY mapping is
    checked against its specifier" is a property of the function and not of one caller.
    Measured on ``luigi/1``, the name-only reading recorded
    ``numpy==1.18.4``'s build needing ``Cython>=0.29.14`` *satisfied by*
    ``cython==0.29.7``; a requirement with no clause (``setuptools``, ``wheel``) is still
    satisfied by whatever the layer holds.

    A pin whose own build requires itself is skipped: installing it early would mean
    building it to build it, which is the circularity the toolchain install exists to
    avoid, and no measured pin does it.

    ``build_requires_of`` is the reading (:func:`pypi_build_requirements`), injected so
    the whole decision is a structural test. A :class:`BakeError` from it is answered
    as "nothing known" — a release whose metadata could not be read is not a reason to
    fail a layer before it is built, and the layer's own build is the backstop.

    ``python_minor`` is the layer's interpreter and it is read here, on each release's own
    declarations, because this is the **early pass**'s half of §3.2's marker rule
    (:func:`marker_skips`): a build requirement that does not apply on this image is not
    something a build of it fetches. Measured, ``scikit-learn==0.23.1`` declares
    ``numpy==1.13.3`` under ``python_version == "3.6"``, and reading that as applying is
    what made ``pandas/1``'s ``early_pins`` name a numpy ordering for a python the layer is
    not — while ``cffi>=1.8,!=1.11.3; platform_python_implementation != 'PyPy'`` is true
    for CPython and stays. ``marker_skipped`` is the caller's list to record them in —
    the same "one out-parameter" shape :func:`build_refs` and :func:`pyproject_refs` use
    for ``notes`` — and it is optional because the preflight calls this too and the record
    belongs to the deps JSON the bake writes.
    """
    held = {
        canonicalize_name(name): (f"{name}=={version}", version)
        for name, version in pinned
    }
    out: list[EarlyInstall] = []
    seen: set[tuple[str, str]] = set()
    for name, version in built:
        own = canonicalize_name(name)
        spelling = f"{name}=={version}"
        try:
            refs = tuple(build_requires_of(name, version))
        except BakeError:
            continue
        refs, dropped = marker_skips(refs, python_minor=python_minor)
        if marker_skipped is not None:
            marker_skipped += dropped
        for ref in refs:
            required = canonicalize_name(ref.name)
            holding = held.get(required)
            if required == own or holding is None:
                continue
            # The held pin has to *satisfy* the clause, not merely share its name
            # (:func:`_specifier_accepts`): a mapping whose ``satisfied_by`` the build
            # itself refuses is a false record, and the unsatisfied case is not this
            # function's to order — the pin it names moves instead
            # (:func:`narrow_build_pins`), which is why it is printed as NARROWED and
            # never as EARLY.
            if not _specifier_accepts(_specifier_of(ref), holding[1]):
                continue
            satisfied_by = holding[0]
            if (spelling, satisfied_by) in seen:
                continue
            seen.add((spelling, satisfied_by))
            out.append(
                EarlyInstall(pin=spelling, requirement=ref, satisfied_by=satisfied_by)
            )
    return tuple(out)


@dataclass(frozen=True)
class PreflightBug:
    """One layer's preflight, over every pin it will install."""

    key: str
    checkpoint: str
    python_minor: str
    pins: tuple[PreflightPin, ...]
    conflicts: tuple[PinConflict, ...] = ()
    """The pairs of pins whose own metadata contradict each other (§3.2's cases).

    [VANTA-WHY] Read here rather than discovered by the layer's ``pip check``, because
    the point of the preflight is to say it *before* the layer is built: ``sanic/1``'s
    ``h11==0.9.0`` against ``httpcore==0.3.0``'s declared ``h11==0.8.*`` is a fact about
    the two releases, available from PyPI, and a bake that learns it from
    ``ResolutionImpossible`` has spent a build cycle on it. The layer's own reading is
    still taken (:func:`pip_check_conflicts`) — the two are not the same check, and the
    one that says what pip *did* is the one the gate uses.
    """

    narrowings: tuple[tuple[str, str], ...] = ()
    """The date-rule choices this layer's own pins moved, with the reason.

    [VANTA-WHY] A narrowed pin is a decision already taken and not a finding, so it is
    recorded here rather than counted by :func:`preflight_verdict` — but it is exactly
    what the pin list cannot show: ``moto==1.3.7`` in ``luigi/1`` is not "the newest
    release on or before the date" and looks arbitrary without the pin that moved it
    (:func:`pin_narrowings`). Printed by :func:`preflight_lines` before any build.
    """

    early: tuple[EarlyInstall, ...] = ()
    """The build requirements a pin in this layer needs and another pin here satisfies.

    [VANTA-WHY] Recorded because it is an *ordering* decision and nothing else in the
    report shows it: the two pins are both in the layer either way, and the only
    difference is that the bake installs the second one before the main list so the
    first one's build finds it (§3.2, :func:`early_install_pins`). A reader of the pin
    list cannot tell ``cffi`` apart from any other dependency; a reader of this key can
    tell that ``cryptography==2.9.2``'s build is the reason it is where it is.
    """

    skipped: tuple[SkippedLine, ...] = ()
    """The requirements §3.2's marker rule dropped, with the marker that dropped them.

    [VANTA-WHY] The preflight's half of the record :func:`deps_provenance` writes under
    the same key, and the reason it is here rather than on the terminal: what a build
    declaration says is measured, not read, so a `--preflight-only` run has to leave it
    somewhere. Measured, ``scipy==1.4.1``'s and ``scikit-learn==0.23.1``'s declarations
    each name ``numpy`` three times — once for ``python_version == '3.5'``, once for
    ``'3.6'`` and once for ``>= '3.8'`` — and only the last of the three applies to a
    py3.8 layer; the other two are dropped here, and the ``EARLY`` line the reader sees
    names the clause that holds instead of the first one written.
    """

    @property
    def refused(self) -> tuple[PinConflict, ...]:
        """The conflicts §3.2 does not tolerate: at least one side is a harness choice.

        Case (a) only. A case (?) reading is excluded because it is not a conflict at
        all — nobody could read that pin — and calling it refused would fail a layer over
        a fetch that did not happen.
        """
        return tuple(
            conflict for conflict in self.conflicts
            if conflict.case == CASE_HARNESS_CHOICE
        )

    @property
    def target(self) -> str:
        """The interpreter and platform the classification was made for."""
        return f"{_cpython_tag(self.python_minor)}/linux-aarch64"

    @property
    def to_build(self) -> tuple[PreflightPin, ...]:
        """The class (b) pins: pip will compile them from their sdist."""
        return tuple(pin for pin in self.pins if pin.kind == SDIST_ONLY)

    @property
    def unavailable(self) -> tuple[PreflightPin, ...]:
        """The class (c) pins: nothing pip can install here, so the layer cannot."""
        return tuple(pin for pin in self.pins if pin.kind == NO_DISTRIBUTION)

    def counts(self) -> dict[str, int]:
        """The three class counts, plus the three findings that are not a class.

        ``gap`` and ``unbuildable`` are counted separately because neither replaces the
        class: an unbuildable pin *is* sdist-only (that is what pip will try), and the
        extra word is what says the attempt cannot succeed.

        ``early`` is the third and counts :attr:`early` — the build requirements the
        layer satisfies from its own pins. It is a count of *findings* rather than of
        pins: the two pins it names are already counted in their own classes, and what
        the number says is how many ordering decisions the bake is about to take.
        """
        counts = {kind: 0 for kind in PREFLIGHT_CLASS_LETTERS}
        for pin in self.pins:
            counts[pin.kind] = counts.get(pin.kind, 0) + 1
        counts["gap"] = sum(1 for pin in self.pins if pin.gap is not None)
        counts["unbuildable"] = sum(1 for pin in self.pins if pin.unbuildable is not None)
        counts["early"] = len(self.early)
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "key": self.key,
            "checkpoint": self.checkpoint,
            "target": self.target,
            "counts": self.counts(),
            "pins": [pin.as_dict() for pin in self.pins],
            "conflicts": [conflict.as_dict() for conflict in self.conflicts],
            "narrowings": [
                {"pin": pin, "reason": reason} for pin, reason in self.narrowings
            ],
            "early_pins": [item.as_dict() for item in self.early],
            "skipped": [line.as_dict() for line in self.skipped],
        }


def preflight_classifier(
    *,
    files_of: Callable[[str, str], Optional[tuple[PyPIFile, ...]]] = pypi_release_files,
    sdist_of: Callable[[PyPIFile], tuple[tuple[str, ...], bool]] = sdist_file_list,
) -> Callable[[str, str, str], PreflightPin]:
    """``(name, version, python_minor) -> PreflightPin``, each pin fetched once per run.

    [VANTA-WHY] One instance per run, shared by §3.2's unresolvable-pin rule
    (:func:`drop_unresolvable_refs`) and by the table the bake prints
    (:func:`preflight_layers`). Both ask the same question about the same pins, and a
    second cache would pay PyPI twice for every answer. Cached across bugs rather than
    per bug, because the splits share their pins (numpy alone is in six of the eleven
    dev-split bugs), keyed ``(name, version, python_minor)`` so a package two bugs pin
    at the same version is fetched once — the same shape as the per-run ``releases``
    mapping :func:`resolve_bug_deps` keeps for the date fetches.

    ``files_of``/``sdist_of`` are injected so the whole classification is a structural
    test: the default pair is the network, and no test should need it.
    """
    cache: dict[tuple[str, str, str], PreflightPin] = {}

    def classify(name: str, version: str, python_minor: str) -> PreflightPin:
        key = (name, version, python_minor)
        if key not in cache:
            cache[key] = preflight_pin(
                name,
                version,
                python_minor=python_minor,
                files_of=files_of,
                sdist_of=sdist_of,
            )
        return cache[key]

    return classify


def cached_release_metadata(
    *,
    fetch: Callable[
        [str, str], tuple[tuple[str, ...], Optional[str]]
    ] = pypi_release_metadata,
) -> Callable[[str, str], tuple[tuple[str, ...], Optional[str]]]:
    """``(name, version) -> (requires_dist, requires_python)``, fetched once per run.

    [VANTA-WHY] The same shape as :func:`preflight_classifier`, and for the same
    reason: two places in one run ask what a release declares — §3.2's date rule, when
    an exact pin narrows a name the date rule would otherwise pick freely
    (:func:`_narrowing_requirements`), and the case-(a)/(b) check the preflight prints
    (:func:`metadata_conflicts`) — and the splits share their pins heavily, so a second
    cache would pay PyPI once per pin *per layer* for the same JSON. Keyed
    ``(name, version)`` and not ``(name, version, python_minor)`` like the classifier's:
    what a release declares does not depend on the interpreter reading it, and
    ``requires_python`` — the one field that does — is part of the answer rather than
    part of the key.

    ``fetch`` is injected so a test can hand it a dict; the default is the network.
    """
    cache: dict[tuple[str, str], tuple[tuple[str, ...], Optional[str]]] = {}

    def read(name: str, version: str) -> tuple[tuple[str, ...], Optional[str]]:
        key = (name, version)
        if key not in cache:
            cache[key] = fetch(name, version)
        return cache[key]

    return read


def cached_build_requirements(
    *,
    fetch: Callable[[str, str], Sequence[DepRef]] = pypi_build_requirements,
) -> Callable[[str, str], tuple[DepRef, ...]]:
    """``(name, version) -> what its sdist needs to be built``, read once per run.

    [VANTA-WHY] The third reader with this shape, and the one that pays the most: this
    one downloads a whole sdist (up to :data:`SDIST_MAX_BYTES`), while the other two ask
    for JSON, so a pin two layers hold being read twice is the difference between one
    400 KB download and two. ``bake`` builds one per run and hands it to both places
    that ask (:func:`early_install_pins` for the report and for the layer's own
    ``early_pins``).

    Keyed ``(name, version)`` like :func:`cached_release_metadata` and unlike the
    classifier: what an sdist declares it needs to build does not depend on the
    interpreter reading it. ``fetch`` is injected so a test can hand it a dict.
    """
    cache: dict[tuple[str, str], tuple[DepRef, ...]] = {}

    def read(name: str, version: str) -> tuple[DepRef, ...]:
        key = (name, version)
        if key not in cache:
            cache[key] = tuple(fetch(name, version))
        return cache[key]

    return read


def preflight_layers(
    layers: Sequence[BugDeps],
    *,
    classify: Optional[Callable[[str, str, str], PreflightPin]] = None,
    metadata_of: Callable[
        [str, str], tuple[tuple[str, ...], Optional[str]]
    ] = pypi_release_metadata,
    build_requires_of: Callable[[str, str], Sequence[DepRef]] = pypi_build_requirements,
) -> list[PreflightBug]:
    """Classify every pin of every layer, through one shared classifier.

    ``classify`` is the run's own :func:`preflight_classifier` when the caller has one —
    ``bake`` builds it once and hands the same instance to the drop decision and to this
    report — and a fresh one otherwise, so the cache still covers a whole call.

    ``metadata_of`` is the second reading each layer gets, and it is §3.2's case-(a)/(b)
    check: every pin's ``requires_dist``, against the layer's other pins
    (:func:`metadata_conflicts`). It comes from the same reader the date rule narrows
    itself with, so "what does this release declare" has one answer in the file —
    ``bake`` therefore hands it the run's :func:`cached_release_metadata` instance,
    which is also the one it gave :func:`resolve_bug_deps`, so a pin two layers hold is
    fetched once for both readings. It is the uncached network by default, and every
    test that calls this passes a stub.

    ``build_requires_of`` is the third reading, and it answers the question the other
    two cannot: what each pin's **build** needs (:func:`pypi_build_requirements`), which
    is in its sdist's ``pyproject.toml``/``setup.py`` and in none of the metadata forms
    the other two read. Its one use here is :func:`early_install_pins` — reporting the
    build requirements some *other* pin of the same layer satisfies, which the bake then
    installs before the main list (§3.2). It is the run's
    :func:`cached_build_requirements` instance when ``bake`` has one, because each answer
    costs a whole sdist download, and every test that calls this passes a stub.
    """
    classifier = classify or preflight_classifier()

    out: list[PreflightBug] = []
    for layer in layers:
        minor = minor_version(str(layer.entry["python_version"]))
        pins = layer_pins(layer)
        classified = tuple(classifier(name, version, minor) for name, version in pins)
        # §3.2's marker rule over the pinned releases' own build declarations. The drops
        # are collected here and recorded on the report rather than printed, because the
        # reading is what a ``--preflight-only`` run measures and the terminal is for
        # the findings a reader has to decide on (:attr:`PreflightBug.skipped`).
        marker_skipped: list[SkippedLine] = []
        out.append(
            PreflightBug(
                key=f"{layer.entry['project']}/{layer.entry['bug_id']}",
                checkpoint=layer.checkpoint,
                python_minor=minor,
                pins=classified,
                conflicts=metadata_conflicts(
                    pins,
                    own=own_pin_names(layer.provenance),
                    metadata_of=metadata_of,
                ),
                narrowings=pin_narrowings(layer.provenance),
                early=early_install_pins(
                    tuple(
                        (pin.name, pin.version)
                        for pin in classified
                        if pin.kind == SDIST_ONLY
                    ),
                    pins,
                    build_requires_of=build_requires_of,
                    # The same reading the bake makes, from the pins the layer recorded:
                    # §3.2's marker rule is applied here too, so the preflight's EARLY
                    # lines name what the build will actually want (:func:`marker_skips`).
                    python_minor=minor,
                    marker_skipped=marker_skipped,
                ),
                skipped=tuple(marker_skipped),
            )
        )
    return out


def _dropped_pins(
    drops: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[str]:
    """The ``name==version`` spellings the unresolvable-pin rule took out of the split."""
    return {str(record.get("pin")) for records in drops.values() for record in records}


def _unavailable_pins(
    reports: Sequence[PreflightBug], *, dropped: Container[str]
) -> dict[str, list[str]]:
    """``{pin: the bug keys it stops}`` for the pins nothing on this image can install."""
    findings: dict[str, list[str]] = {}
    for report in reports:
        for pin in report.unavailable:
            spelling = f"{pin.name}=={pin.version}"
            if spelling in dropped:
                continue
            findings.setdefault(spelling, []).append(report.key)
    return findings


def _finding_pins(
    reports: Sequence[PreflightBug], *, dropped: Container[str], finding: str
) -> dict[str, list[str]]:
    """``{pin: the bug keys it stops}`` for the pins carrying ``finding``.

    ``finding`` is the :class:`PreflightPin` attribute that holds the reason — ``gap``
    for a compiling sdist the reviewed table does not cover, ``unbuildable`` for one no
    entry could make work. They are separate questions and a pin answers exactly one of
    them, which is why this is asked once per finding rather than once for both.

    ``dropped`` is the set of ``pin`` strings the rule dropped — a pin that was dropped is
    *not* an open finding, so it is skipped: it is in the drops section with the audit that
    settled it, and printing it twice would make the run look like it has more problems
    than it has.
    """
    findings: dict[str, list[str]] = {}
    for report in reports:
        for pin in report.pins:
            if getattr(pin, finding) is None:
                continue
            spelling = f"{pin.name}=={pin.version}"
            if spelling in dropped:
                continue
            findings.setdefault(spelling, []).append(report.key)
    return findings


def preflight_lines(
    reports: Sequence[PreflightBug],
    *,
    dropped: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> list[str]:
    """One line per layer, then each finding once for the whole split.

    Per layer, **one** line: how many pins it installs and how each one is classified,
    plus what the unresolvable-pin rule took out of it. A bug with nothing to report
    gets that line too — "this layer is all wheels" is the answer to the question the
    table is asked, and a table with the quiet bugs missing cannot say whether a bug was
    checked.

    Then the findings, each printed once for the whole split rather than once per bug
    that has it: the pins the split shares are why a reader is here (numpy alone is in
    six of the eleven dev-split bugs), and the same GAP on six lines is six times the
    reading for one decision. ``<pin> — <bug keys>`` is the shape, so "which bugs does
    this unreviewed build stop" is answerable from the line.

    A layer's sdist-only pins are deliberately **not** listed: a pure-Python sdist needs
    no compiler and a noticed-disposition is a decision nobody has to take, so both live
    in the layer's ``.preflight.json`` and not on the terminal (§3.2). What is printed is
    what is still open — an unreviewed GAP, an UNBUILDABLE, an UNAVAILABLE pin the rule
    did not take, the pins whose own declarations contradict each other
    (:func:`_conflict_sections`), the pins the date rule moved for the layer's own pins
    (:func:`_narrowing_section`), the build requirements the layer installs early
    (:func:`_early_section`), and what the rule did take — and the pins' own reasons
    are in that JSON, next to the provenance.

    ``dropped`` is ``{checkpoint: that layer's dropped records}``, read from the
    provenance the run just wrote; the records are the dicts, so the audit's sites are
    in the JSON and the reason is repeated here.
    """
    drops = dropped or {}
    lines = [f"preflight: {len(reports)} layer(s)"]
    for report in reports:
        counts = report.counts()
        lines.append(
            f"[{report.checkpoint}] {report.key} ({report.target}): "
            f"{len(report.pins)} pin(s), {counts[INSTALLABLE_WHEEL]} wheel, "
            f"{counts[SDIST_ONLY]} sdist-only, {counts[NO_DISTRIBUTION]} unavailable, "
            f"{counts['gap']} gap, {counts['unbuildable']} unbuildable, "
            f"{counts['early']} early, "
            f"{len(drops.get(report.checkpoint, ()))} dropped"
        )

    dropped_pins = _dropped_pins(drops)
    gaps = _finding_pins(reports, dropped=dropped_pins, finding="gap")
    blocked = _finding_pins(reports, dropped=dropped_pins, finding="unbuildable")
    unavailable = _unavailable_pins(reports, dropped=dropped_pins)

    keys = {report.checkpoint: report.key for report in reports}
    lines += _section("unreviewed sdist build(s)", "GAP", gaps)
    lines += _section("unbuildable pin(s)", "UNBUILDABLE", blocked)
    lines += _section("pin(s) nothing can install", "UNAVAILABLE", unavailable)
    lines += _conflict_sections(reports, dropped=dropped_pins)
    lines += _narrowing_section(reports)
    lines += _early_section(reports, dropped=dropped_pins)
    if dropped_pins:
        lines += ["", f"dropped pin(s) ({len(dropped_pins)}) — §3.2's rule, with the "
                      "audit that settled it:"]
        for checkpoint, records in drops.items():
            for record in records:
                imports = record.get("imports") or {}
                lines.append(
                    f"  {record.get('pin')} ({record.get('outcome')}) — "
                    f"{record.get('reason')}"
                )
                lines.append(
                    f"    {keys.get(checkpoint, checkpoint)} — "
                    f"{imports.get('reason') or 'no import audit recorded'}"
                )
    return lines


def _section(title: str, label: str, findings: Mapping[str, Sequence[str]]) -> list[str]:
    """One printed block for one finding, or nothing when there is no such finding."""
    if not findings:
        return []
    lines = ["", f"{title} ({len(findings)}):"]
    for pin, keys in findings.items():
        lines.append(f"  {label} {pin} — {', '.join(keys)}")
    return lines


_CONFLICT_SECTIONS: Mapping[str, tuple[str, str]] = {
    CASE_HARNESS_CHOICE: (
        "pin conflict(s), §3.2 case (a) — one side is a version this harness chose, "
        "so the layer cannot hold both",
        "REFUSED",
    ),
    CASE_BOTH_OWN_PINS: (
        "pin conflict(s), §3.2 case (b) — both sides are BugsInPy's own pins, "
        "installed as written",
        "TOLERATED",
    ),
    CASE_UNKNOWN: (
        "pin(s) nothing could be read for — no PyPI 'requires_dist', no wheel "
        "metadata, no PEP 658 sidecar — so what they declare is unknown, and their "
        "layer's own pip check is the reading that covers them",
        "UNKNOWN",
    ),
}


def _conflict_sections(
    reports: Sequence[PreflightBug], *, dropped: Container[str]
) -> list[str]:
    """§3.2's cases, one section each, in the order a reader needs them.

    Case (a) first, because it is the only one that stops a layer: the two pins are
    there, the requirement is in one of them, and the other does not satisfy it — so a
    reader who reads one section reads that one. Then case (b), which is the benchmark's
    own environment reproduced, and last the pins nobody could read.

    A conflict with a **dropped** pin on either side is not printed: the layer will not
    install that pin, so no pair is left to conflict — the same reason
    (:func:`_unavailable_pins`) skips a dropped pin, and the drops have their own block
    below with the audit that settled them.

    There is no section for :data:`CASE_DROPPED_DEPENDENCY`, and it is the one case this
    report cannot produce: every conflict here comes from the pins' own metadata
    (:func:`metadata_conflicts`), and case (d) is a *reading* — the layer's own ``pip
    check`` saying a requirement is absent — so it exists only in
    :func:`run_bake_actions`' gate. The preflight would have to install the layer to have
    that reading, which is the layer's own build log, not this one.
    """
    grouped: dict[str, dict[str, list[str]]] = {
        case: {} for case in _CONFLICT_SECTIONS
    }
    for report in reports:
        for conflict in report.conflicts:
            if any(spelling in dropped for spelling in conflict.pins):
                continue
            grouped[conflict.case].setdefault(
                f"{conflict.requires} → {conflict.held}", []
            ).append(report.key)
    out: list[str] = []
    for case, (title, label) in _CONFLICT_SECTIONS.items():
        out += _section(title, label, grouped[case])
    return out


def _narrowing_section(reports: Sequence[PreflightBug]) -> list[str]:
    """One printed block for the date-rule choices a pin in the same layer moved.

    Not a finding and not a section per §3.2's cases (:func:`_conflict_sections`): the
    version was chosen, the pins do not conflict, and the block exists because the chosen
    version is *not* what the date rule alone would give. Printed once per pin with every
    layer that holds it, the same shape the other blocks use, so "which layers is this
    pin in" is answerable from the line.
    """
    grouped: dict[str, tuple[str, list[str]]] = {}
    for report in reports:
        for pin, reason in report.narrowings:
            grouped.setdefault(pin, (reason, []))[1].append(report.key)
    if not grouped:
        return []
    lines = [
        "",
        f"narrowed pin(s) ({len(grouped)}) — §3.2's date rule, moved to satisfy a pin in "
        "the same layer:",
    ]
    for pin, (reason, keys) in grouped.items():
        lines.append(f"  NARROWED {pin} — {reason} ({', '.join(keys)})")
    return lines


def _early_section(
    reports: Sequence[PreflightBug], *, dropped: Container[str]
) -> list[str]:
    """One printed block for the build requirements the layer installs early.

    [VANTA-WHY] Printed rather than left to the layer's ``.preflight.json`` because it
    is the one finding in this report that changes what the *recipe* does: every other
    line is about whether a pin can go in, and this one is about the order two pins go
    in. A reader comparing the pin list to the recipe sees a pin installed twice and
    needs a line that says why (§3.2, :func:`early_install_pins`).

    The line names both sides and the declaration, in the direction of the decision —
    *this* pin's build needs *that* pin — because "cffi" alone would not say which
    release asked for it; the source it came from is in the layer's JSON. A finding
    with a **dropped** pin on either side is skipped, like the other blocks: a dropped
    pin is not installed at all, so there is nothing to order it before.
    """
    grouped: dict[str, list[str]] = {}
    for report in reports:
        for item in report.early:
            if item.pin in dropped or item.satisfied_by in dropped:
                continue
            grouped.setdefault(
                f"{item.pin}'s build needs {item.requirement.spec} → "
                f"{item.satisfied_by}",
                [],
            ).append(report.key)
    return _section(
        "build requirement(s) the layer installs early", "EARLY", grouped
    )


def preflight_verdict(
    reports: Sequence[PreflightBug],
    *,
    dropped: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    flags: int = 0,
) -> str:
    """One line: whether this split is ready to bake.

    Ready means there is nothing left for a human to decide *before* the first layer is
    built: no unreviewed sdist build (§3.2's table), no **case (a)** pin conflict — two
    pins of one layer whose own declarations contradict each other with a version this
    harness chose on one side, which is the one conflict the layer's own ``pip check``
    will refuse (:func:`pip_check_conflicts`) — and no flagged bug (a bug whose
    dependencies could not be resolved). A dropped pin is a decision already recorded and
    not a blocker; if dropping one broke a bug, that shows up as a **flagged** bug — or
    at ``prepare``, which measures the buggy/fixed flip for real and names the drops when
    the gate fails. Either way the counts are printed, because "not ready" without them
    says nothing a reader can act on.

    Case (b) is not counted: it is the benchmark's own environment, tolerated on purpose
    (:data:`CASE_BOTH_OWN_PINS`), and a verdict that called it "not ready" would be
    asking for a decision §3.2 already took. A case (?) reading is not counted either —
    it is a fetch that did not happen, not a conflict.
    """
    dropped_pins = _dropped_pins(dropped or {})
    gaps = _finding_pins(reports, dropped=dropped_pins, finding="gap")
    conflicts = [
        (report, conflict)
        for report in reports
        for conflict in report.refused
        if not any(spelling in dropped_pins for spelling in conflict.pins)
    ]
    if not gaps and not conflicts and not flags:
        return "ready to bake"
    counts: list[str] = []
    if gaps:
        bugs = {key for keys in gaps.values() for key in keys}
        counts.append(f"{len(gaps)} unreviewed sdist build(s) across {len(bugs)} bug(s)")
    if conflicts:
        bugs = {report.key for report, _ in conflicts}
        counts.append(
            f"{len(conflicts)} pin conflict(s) across {len(bugs)} bug(s), none of them "
            "§3.2 case (b)"
        )
    if flags:
        counts.append(f"{flags} flagged bug(s)")
    return "not ready to bake: " + ", ".join(counts)


def preflight_path(deps_dir: Path, checkpoint: str) -> Path:
    return deps_dir / f"{checkpoint}.preflight.json"


def write_preflight(deps_dir: Path, report: PreflightBug) -> Path:
    """Record one layer's preflight next to its provenance, in the same directory.

    A sibling of ``<checkpoint>.json`` and not that file: ``prepare`` reads the
    provenance to find what the layer installs, and this is what the *bake* learned on
    the way there. Nothing globs ``*.json`` in ``deps/``, and the ``.preflight.json``
    suffix keeps it that way.
    """
    deps_dir.mkdir(parents=True, exist_ok=True)
    path = preflight_path(deps_dir, report.checkpoint)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# §3.2's unresolvable-pin rule: drop a pin nothing can install, or flag the bug
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DroppedPin:
    """A pin nothing on this image can install, and nothing the graded tests reach.

    [VANTA-WHY] Dropped *with a record*, never silently. A dependency the layer cannot
    install would otherwise be one bug's build error, and it is the same class of
    problem as a missing system package (§3.2): knowable before the layer is built, and
    the answer is a decision rather than a fallback. The record carries the declaration,
    why nothing can install it, and the audit that settles the other half of the rule —
    every import site the tree has for that module, and why each one counts as optional
    — so that "nothing the graded tests reach imports it" is a claim about the tree that
    can be re-checked rather than taken on trust.
    """

    name: str
    spec: str
    """The declaration as written, e.g. ``requests-async==0.5.0``."""
    sources: tuple[str, ...]
    """Every place it was declared."""
    version: Optional[str]
    """The version the pin names, when it names one."""
    outcome: str
    """Which finding: :data:`NO_DISTRIBUTION` or :data:`UNBUILDABLE`."""
    reason: str
    audit: Optional[ImportAudit] = None
    """What the bug's own trees say about the module this distribution provides.

    ``None`` only for a record built by hand — every record :func:`decide_unresolvable`
    makes carries one, and :meth:`as_dict` omits the key rather than inventing an audit.
    """

    @property
    def pin(self) -> str:
        return f"{self.name}=={self.version}" if self.version else self.name

    @property
    def import_note(self) -> str:
        """Why the sites found do not make this pin required, in one clause."""
        return self.audit.reason if self.audit is not None else ""

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "spec": self.spec,
            "pin": self.pin,
            "sources": list(self.sources),
            "outcome": self.outcome,
            "reason": self.reason,
        }
        if self.audit is not None:
            data["imports"] = self.audit.as_dict()
        return data


def decide_unresolvable(
    name: str,
    *,
    version: Optional[str],
    spec: str,
    sources: Sequence[str],
    outcome: str,
    reason: str,
    audit: ImportAudit,
) -> DroppedPin:
    """§3.2's one rule, for every pin nothing on this image can install.

    Two findings reach here and they are one rule rather than two:

    * a class (c) pin — ``luigi/1``'s ``requests-async==0.5.0``, a release PyPI never
      uploaded, and ``sanic/1``'s ``pywin32==227``, whose files are all foreign wheels
      and which ships no sdist;
    * a class (b) pin preflight marks :data:`UNBUILDABLE` — an sdist whose build no
      entry in the reviewed table could make work here (``pyarrow``'s Arrow C++,
      ``llvmlite``'s LLVM).

    Both are settled by the same question, asked of the bug itself: **does anything the
    graded tests reach import the module this distribution provides?** ``audit`` is that
    answer, and it is three answers rather than one, because "somewhere in the tree" is
    not the same claim as "somewhere in this bug's tests":

    * nothing imports it at all → dropped;
    * every site is behind a guard (:data:`_OPTIONAL_CALLS`, an ``except ImportError``, a
      version check) → dropped, and the record lists the sites so the claim is checkable;
    * sites exist but no unconditional one is in the failing test files' module-scope
      import closure → dropped: the tests this bug is graded on never reach it;
    * an unconditional site *is* in that closure → :class:`BakeError`, and
      :func:`resolve_bug_deps` flags the bug. A layer missing a module the graded tests
      import cannot run the suite, so baking it anyway would trade a named flag now for
      an unreadable failure later.

    [VANTA-WHY] The last answer is the whole point of the change from "does any file
    import it": measured on pandas/1, ``import pyarrow`` appears in 18 files, and every
    one of them is optional or lazy — pandas works without pyarrow, and the failing test
    is ``pandas/tests/dtypes/test_dtypes.py``, nowhere near parquet. Calling that a
    required pin flags a bug that would bake and grade perfectly well, and the fix is
    the *prepare* gate, which measures the buggy/fixed flip for real: if a dropped pin
    is what breaks the bug, the gate catches it and names the drops (§11 step 2).
    """
    if audit.required_by is not None:
        raise BakeError(
            f"{name} cannot be installed on this image ({reason}), and "
            f"{audit.required_by} imports it, so the bug cannot be baked without it. "
            "Amend the split or flag the bug: §3.2's unresolvable-pin rule only drops a "
            "pin every site imports optionally, or one no site in the failing tests' "
            "import closure reaches."
        )
    return DroppedPin(
        name=name,
        spec=spec,
        sources=tuple(sources),
        version=version,
        outcome=outcome,
        reason=reason,
        audit=audit,
    )


def exact_version(spec: str, source: str) -> Optional[str]:
    """The version an exact ``==`` pin names, or ``None`` for anything else.

    Only an exact pin can be judged *before* the date rule runs: a range or a bare name
    has no version until :func:`resolve_pins` picks one, and that half of the rule is
    applied to the resolved list instead (:func:`drop_unresolvable_deps`). ``==1.0.*``
    names a set of releases and not one, so it is not one of these either.
    """
    for clause in _requirement(spec, source).specifier:
        if clause.operator == "==" and not clause.version.endswith(".*"):
            return clause.version
    return None


def distribution_module_names(
    distribution: str, *, distributions: Mapping[str, str]
) -> tuple[str, ...]:
    """The import spellings ``distribution`` is normally known by.

    The split's reviewed ``module_distributions`` table inverted — ``Pillow`` → ``PIL``,
    ``PyYAML`` → ``yaml`` — plus PEP 503's own ``-`` → ``_`` spelling, which is what
    nearly every package is (``requests-async`` → ``requests_async``). The table is
    keyed by the import spelling, so inverting it is what recovers the name the tree
    would have written.

    Used only to ask "does anything import this", and there the extra spellings are the
    safe direction: a name that matches too much is a false *flag*, not a false drop.
    """
    canonical = canonicalize_name(distribution)
    names = {
        module
        for module, dist in distributions.items()
        if canonicalize_name(dist) == canonical
    }
    names.add(canonical.replace("-", "_"))
    return tuple(sorted(names))


def tree_import_sites(
    files: Sequence[str],
    *,
    read_file: Callable[[str], Optional[str]],
    stdlib: Iterable[str] = PY38_STDLIB,
) -> dict[str, tuple[ImportSite, ...]]:
    """``{module as imported: every site that imports it}`` over ``files``.

    [VANTA-WHY] The mirror image of :func:`test_import_closure`: that walk asks which
    distributions the test side needs, this asks whether *one* distribution is needed at
    all — the question §3.2's unresolvable-pin rule turns on. Read with ``ast`` for the
    same reason (nothing third-party runs on the host, design C3), and a file that does
    not parse is passed over rather than reported, exactly as the closure passes it
    over: the target interpreter could not collect it either.

    Both spellings are filed — ``google.protobuf`` and its top-level ``google`` — because
    a distribution provides a module and the tree may name either. Every site is kept and
    not just the first file, because the rule's answer is "every site of this module is
    optional", which one filename cannot settle and a reader cannot check.
    """
    stdlib_names = set(stdlib)
    sites: dict[str, list[ImportSite]] = {}
    for path in sorted(files):
        if not path.endswith(".py"):
            continue
        text = read_file(path)
        if text is None:
            continue
        try:
            parsed = ast.parse(text)
        except SyntaxError:
            continue
        found: list[ImportSite] = []
        _collect_imports(parsed, optional=None, out=found, path=path)
        for site in found:
            top = site.module.split(".")[0]
            if not top or top == "__future__" or top in stdlib_names:
                continue
            # The set, because the two spellings are the same name for a top-level
            # module (``import pyarrow``) and one site must not be filed twice.
            for name in {site.module, top}:
                sites.setdefault(name, []).append(site)
    return {module: tuple(records) for module, records in sites.items()}


def _module_of(path: str) -> Optional[str]:
    """The dotted name a tree file provides, or ``None`` when it is not importable.

    ``pandas/core/frame.py`` → ``pandas.core.frame``; ``pandas/core/__init__.py`` →
    ``pandas.core``.
    """
    if not path.endswith(".py"):
        return None
    parts = path[: -len(".py")].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _resolve_module(dotted: str, index: Mapping[str, str]) -> Optional[str]:
    """The file ``dotted`` names, or the file of its nearest importable parent."""
    parts = dotted.split(".")
    while parts:
        found = index.get(".".join(parts))
        if found is not None:
            return found
        parts.pop()
    return None


def _absolute_module(path: str, level: int, module: str) -> Optional[str]:
    """The absolute dotted name a relative import at ``path`` resolves to."""
    if not level:
        return module or None
    parts = path[: -len(".py")].split("/")
    if parts and parts[-1] != "__init__":
        parts = parts[:-1]
    keep = len(parts) - (level - 1)
    if keep <= 0:
        return None
    package = parts[:keep]
    return ".".join([*package, module] if module else package)


def _module_scope_imports(
    node: ast.AST,
    out: list[tuple[int, str, str]],
    *,
    type_checking: Optional[frozenset[str]] = None,
) -> None:
    """``(level, module, imported name)`` for every import a *module body* runs.

    [VANTA-WHY] A function body is not module scope, and that distinction is the whole
    of §3.2's rule for a lazily imported pin. ``import pandas`` runs when pandas is
    imported; the ``from pandas.core.arrays._arrow_utils import …`` inside a method body
    runs when that method is called — and ``_arrow_utils`` is where pandas' one
    unconditional ``import pyarrow`` lives. A graph that followed function bodies would
    put pyarrow in the import closure of every pandas test that reaches any array, which
    is exactly the false flag the rule exists to remove. A ``TYPE_CHECKING`` block — in
    any of its spellings, an alias included — and an ``if False:`` are skipped for the
    same reason: they never run, so a file only one of them names is not a file this one
    reaches.

    Relative imports are kept as ``(level, module)`` rather than dropped the way
    :func:`_collect_imports` drops them: inside a package the relative spelling is how
    most of its own modules are named, so a closure that ignored it would see only the
    handful of absolute imports at the top.
    """
    if type_checking is None:
        type_checking = _type_checking_names(node)
    if isinstance(node, ast.Import):
        for alias in node.names:
            out.append((0, alias.name, ""))
        return
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            out.append((node.level, node.module or "", alias.name))
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.If) and _never_runs(node.test, type_checking):
        return
    if isinstance(node, ast.Try):
        for branch in (*node.body, *node.handlers, *node.orelse, *node.finalbody):
            _module_scope_imports(branch, out, type_checking=type_checking)
        return
    for child in ast.iter_child_nodes(node):
        _module_scope_imports(child, out, type_checking=type_checking)


def module_import_graph(
    files: Sequence[str],
    *,
    read_file: Callable[[str], Optional[str]],
) -> dict[str, tuple[str, ...]]:
    """``{path: the tree files its module body imports}`` for one tree.

    Only the tree's own files are edges: a distribution is a leaf, and which
    distribution a module names is the other question (:func:`tree_import_sites`).
    """
    index: dict[str, str] = {}
    for path in files:
        name = _module_of(path)
        if name is not None:
            index.setdefault(name, path)
    graph: dict[str, tuple[str, ...]] = {}
    for path in files:
        if not path.endswith(".py"):
            continue
        text = read_file(path)
        refs: list[tuple[int, str, str]] = []
        if text is not None:
            try:
                _module_scope_imports(ast.parse(text), refs)
            except SyntaxError:
                refs = []
        targets: list[str] = []
        for level, module, name in refs:
            base = _absolute_module(path, level, module)
            if base is None:
                continue
            # ``from a.b import c`` may name a submodule of ``a.b`` or an attribute of
            # it; both are tried, so ``from pandas.core import arrays`` reaches
            # ``pandas/core/arrays/__init__.py``.
            candidates = [f"{base}.{name}", base] if name else [base]
            for candidate in candidates:
                target = _resolve_module(candidate, index)
                if target is not None and target != path:
                    targets.append(target)
        graph[path] = tuple(dict.fromkeys(targets))
    return graph


def import_closure(
    roots: Iterable[str], graph: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    """Every tree file the module-scope imports of ``roots`` reach, roots included."""
    seen: set[str] = set()
    queue = [root for root in roots if root in graph]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        queue.extend(graph.get(path, ()))
    return frozenset(seen)


@dataclass(frozen=True)
class ImportAudit:
    """What a bug's own trees say about one pin nothing on this image can install."""

    sites: tuple[ImportSite, ...]
    """Every place in either tree that names the module this distribution provides."""
    required_by: Optional[str]
    """The site that makes it required — file, line and commit — or ``None``."""
    reason: str
    """Why it is required, or why it is not, in one clause for the record."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_by": self.required_by,
            "reason": self.reason,
            "sites": [site.as_dict() for site in self.sites],
        }


def import_sites_of(
    distribution: str,
    *,
    sites: Mapping[str, Sequence[ImportSite]],
    distributions: Mapping[str, str],
) -> tuple[ImportSite, ...]:
    """Every site of every spelling ``distribution`` is named by, once each."""
    out: list[ImportSite] = []
    seen: set[tuple[str, str, int, str]] = set()
    for name in distribution_module_names(distribution, distributions=distributions):
        for site in sites.get(name, ()):
            key = (site.commit, site.path, site.line, site.module)
            if key in seen:
                continue
            seen.add(key)
            out.append(site)
    return tuple(sorted(out, key=lambda site: (site.commit, site.path, site.line)))


def import_auditor(
    repo: Path,
    commits: Sequence[str],
    *,
    distributions: Mapping[str, str],
    test_files: Sequence[str] = (),
    read_file: Optional[Callable[[str, Optional[str]], Optional[str]]] = None,
    list_files: Optional[Callable[[str], Sequence[str]]] = None,
) -> Callable[[str], ImportAudit]:
    """A lazy ``distribution -> ImportAudit`` for one bug.

    Lazy on purpose: the walk reads every ``.py`` of the tree at both commits through
    ``git show``, one fork per file, and it is only ever asked about a pin preflight
    could not install — a handful at most. Building the answer for every bug would cost
    a full tree read per bug to answer a question most bugs never ask.

    The rule, in the order the answers are decided:

    1. an unconditional import that is inside the failing test files' *module-scope*
       import closure → required, and the bug is flagged (:func:`decide_unresolvable`);
    2. every site optional — behind a guard (:data:`_OPTIONAL_CALLS`, an
       ``except ImportError``, a version check), inside a ``TYPE_CHECKING`` block
       (:data:`TYPE_CHECKING_ONLY`), or inside a function body
       (:data:`_LAZY_PREFIX`) → not required, whatever the tree looks like. None of
       those runs when the module is imported, so none is evidence the layer needs the
       distribution, and the record keeps every site and its reason;
    3. sites exist but none of the unconditional ones is in that closure → not required,
       because the tests this bug is graded on never reach it.

    [VANTA-WHY] Both commits, because the graded tree is a mixture (§1.2): the buggy
    source with the fix's test files. A file only the buggy commit carries can be the
    importer (a fix that removed an import), and a test file only the fix carries can
    be too, so either side alone would drop a pin the tree needs. The site keeps the
    commit it was read at, so the claim is checkable rather than a summary.
    """
    walk = list_files or (lambda commit: tree_mod.list_files_at(repo, commit))
    read = read_file or (lambda commit, path: tree_mod.read_file_at(repo, commit, path))
    state: Optional[tuple[dict[str, tuple[ImportSite, ...]], frozenset[str]]] = None

    def build() -> tuple[dict[str, tuple[ImportSite, ...]], frozenset[str]]:
        sites: dict[str, tuple[ImportSite, ...]] = {}
        seen: dict[str, list[ImportSite]] = {}
        closure: set[str] = set()
        for commit in dict.fromkeys(commits):
            if not commit:
                continue
            files = list(walk(commit))
            for module, records in tree_import_sites(
                files, read_file=partial(read, commit)
            ).items():
                seen.setdefault(module, []).extend(
                    replace(record, commit=commit) for record in records
                )
            graph = module_import_graph(files, read_file=partial(read, commit))
            roots = [path for path in test_files if path in graph]
            if roots:
                closure.update(import_closure(roots, graph))
            else:
                # Neither tree carries the failing test file, so the closure question
                # has no roots to answer from. An empty closure would drop every pin
                # nothing can install, so the whole tree counts as reachable instead:
                # the conservative direction, and the one that cannot bake a bug
                # without a module its own tests reach.
                closure.update(graph)
        sites = {module: tuple(records) for module, records in seen.items()}
        return sites, frozenset(closure)

    def audit(distribution: str) -> ImportAudit:
        nonlocal state
        if state is None:
            state = build()
        sites, closure = state
        found = import_sites_of(
            distribution, sites=sites, distributions=distributions
        )
        if not found:
            return ImportAudit((), None, "nothing in either tree imports it")
        for site in found:
            if site.optional is None and site.path in closure:
                return ImportAudit(
                    found,
                    site.spelling,
                    f"{site.spelling} imports it unconditionally, inside the failing "
                    "tests' import closure",
                )
        if all(site.optional is not None for site in found):
            return ImportAudit(
                found,
                None,
                f"every one of its {len(found)} import site(s) is behind an optional "
                "guard",
            )
        return ImportAudit(
            found,
            None,
            "no unconditional import of it is in the failing tests' import closure",
        )

    return audit


def uninstallable_reason(
    name: str,
    version: str,
    *,
    releases: Mapping[str, Mapping[str, str]],
    classify: Callable[[str, str, str], PreflightPin],
    python_minor: str,
) -> Optional[tuple[str, str]]:
    """``(outcome, reason)`` when nothing on this image can install ``name==version``.

    Three answers, one shape, because the road from here on is the same:

    * the pin names a release PyPI does not list at all — decided from the ``releases``
      mapping the bake already has (``requests-async==0.5.0``), so it costs no fetch;
    * the release's files say its format cannot work here — a class (c) release whose
      wheels are all foreign and which ships no sdist (``pywin32==227``);
    * the release is an sdist this image cannot build, named by
      :data:`UNBUILDABLE_ON_PLATFORM` (``pyarrow``, :data:`UNBUILDABLE`).

    ``None`` means "this pin is ordinary", which is the answer for a wheel, for an sdist
    the reviewed table covers, and for an sdist that compiles nothing.

    [VANTA-WHY] ``releases`` in and not one package's dates out of it: the rule is one
    callable for the whole bug, so it is bound once —
    ``partial(uninstallable_reason, releases=…, classify=…, python_minor=…)`` — and
    handed to both drop passes. The mapping answers the first finding for free; only a
    release that *does* exist costs a fetch, and that fetch is the run's shared
    preflight answer (:func:`preflight_classifier`), not a second one of its own.
    """
    known = releases.get(name)
    if known is not None and version not in known:
        newest, released = _newest_release(known)
        return NO_DISTRIBUTION, (
            f"PyPI lists no release {version} of {name}, so nothing can install it "
            f"(the newest {name} is {newest}, released {released[:10]})"
        )
    report = classify(name, version, python_minor)
    if report.unbuildable is not None:
        return UNBUILDABLE, report.unbuildable
    if report.kind == NO_DISTRIBUTION:
        return NO_DISTRIBUTION, report.reason
    return None


def _unresolvable_pin(
    name: str,
    version: str,
    *,
    spec: str,
    sources: Sequence[str],
    reason_of: Callable[[str, str], Optional[tuple[str, str]]],
    audit: Callable[[str], ImportAudit],
) -> Optional[DroppedPin]:
    """The record for one pin, or ``None`` when it is ordinary.

    Raises :class:`BakeError` through :func:`decide_unresolvable` when the graded tests
    reach an unconditional import of it — that raise is what flags the bug, and it is
    deliberate that it happens here rather than being returned as a value a caller could
    ignore. The audit is asked *after* ``reason_of`` says the pin cannot be installed, so
    an ordinary pin never pays for the tree walk.
    """
    finding = reason_of(name, version)
    if finding is None:
        return None
    outcome, reason = finding
    return decide_unresolvable(
        name,
        version=version,
        spec=spec,
        sources=sources,
        outcome=outcome,
        reason=reason,
        audit=audit(name),
    )


def drop_unresolvable_refs(
    refs: Sequence[DepRef],
    *,
    reason_of: Callable[[str, str], Optional[tuple[str, str]]],
    audit: Callable[[str], ImportAudit],
) -> tuple[tuple[DepRef, ...], list[DroppedPin]]:
    """Drop every declared pin whose own version nothing can install here.

    This runs **before** :func:`resolve_pins`, and that is the point of it: a BugsInPy
    reference pin naming a release PyPI never uploaded is a class (c) pin, and
    :func:`resolve_pins` refuses the whole bug on it before any later pass could look at
    it (``luigi/1``'s ``requests-async==0.5.0``). Only an exact pin can be judged here —
    the declaration itself has to name the version — so this is half the pass, and
    :func:`drop_unresolvable_deps` is the other half.
    """
    kept: list[DepRef] = []
    dropped: list[DroppedPin] = []
    for ref in refs:
        version = exact_version(ref.spec, ref.source)
        record = (
            None
            if version is None
            else _unresolvable_pin(
                ref.name,
                version,
                spec=ref.spec,
                sources=(ref.source,),
                reason_of=reason_of,
                audit=audit,
            )
        )
        if record is not None:
            dropped.append(record)
            continue
        kept.append(ref)
    return tuple(kept), dropped


def drop_unresolvable_deps(
    resolved: Sequence[ResolvedDep],
    *,
    reason_of: Callable[[str, str], Optional[tuple[str, str]]],
    audit: Callable[[str], ImportAudit],
) -> tuple[list[ResolvedDep], list[DroppedPin]]:
    """The same rule over what the date rule picked.

    The second half of the pass, and not redundant with
    :func:`drop_unresolvable_refs`: a dependency declared as a bare name or as a range —
    ``numpy``, which six bugs in the dev split declare that way — has no version until
    :func:`resolve_pins` chooses one, and only then can the release it picked be asked
    whether it installs here. Measured, that is pandas/1's ``pyarrow``, which comes from
    the import closure with no version at all and resolves to 0.17.1.

    It is called twice with two different kinds of pin, and that is the point: once by
    :func:`resolve_deps` over the layer's *declared* pins and once by
    :func:`resolve_bug_deps` over the **closure walk's** picks, which are produced after
    ``resolve_deps`` has already returned. A transitive pick is a pin chosen by the same
    date rule and installed the same way, so an unbuildable one is the same finding —
    ``luigi/1``'s and ``sanic/1``'s ``cryptography==2.8`` came in through ``moto`` and
    ``pyopenssl`` and were installed over a preflight that had already marked that
    release unbuildable here.

    A package the interpreter layer carries is never asked: this layer does not install
    it, so there is nothing to drop.
    """
    kept: list[ResolvedDep] = []
    dropped: list[DroppedPin] = []
    for dep in resolved:
        version = None if dep.harness else _pin_version(dep.pin)
        record = (
            None
            if version is None
            else _unresolvable_pin(
                dep.name,
                version,
                spec=dep.spec,
                # A declared pin's ``sources`` say which file asked for it; a transitive
                # pick has none — no file in the bug's tree named it — so what it
                # *was* declared by is the release whose ``requires_dist`` required it
                # (:attr:`ResolvedDep.required_by`). Without that the record would name
                # a distribution and no reason it is in the layer at all.
                sources=dep.sources or dep.required_by,
                reason_of=reason_of,
                audit=audit,
            )
        )
        if record is not None:
            dropped.append(record)
            continue
        kept.append(dep)
    return kept, dropped


def _declared_requirement(
    requires: Sequence[str], name: str
) -> Optional[SpecifierSet]:
    """The constraint a release's own metadata puts on ``name``, or ``None`` for none.

    ``None`` means the release does not mention ``name`` at all; an **empty**
    specifier set means it names ``name`` with no bound (``pytest-metadata`` is
    declared that way from ``pytest-json-report`` 0.5.0 on), which is a different
    claim and is why this returns ``Optional`` rather than a set that could be empty.
    """
    for raw in requires:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == canonicalize_name(name):
            return requirement.specifier
    return None


def newest_accepting_pytest(
    name: str,
    versions: Mapping[str, str],
    *,
    pytest_version: str,
    python_minor: Optional[str],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> str:
    """The newest release of ``name`` whose declared requirement accepts ``pytest_version``.

    Newest first, so the answer is the one the packages' own metadata allows rather
    than the one a date rule would have picked. A release that declares no pytest
    requirement accepts any pytest; a release whose ``Requires-Python`` excludes the
    layer's interpreter is skipped, because pip would refuse it — measured:
    ``pytest-metadata`` 2.0.4 declares ``pytest>=3.0.0,<8.0.0`` and ``>=3.7,<4.0``, so
    on the Python 3.6 bugs (tqdm/1) the newest acceptable release is 1.11.0, not 2.0.4.

    A release whose declarations cannot be read at all is skipped too, and that is the
    difference the reader's fallback chain makes: ``declares nothing`` and ``could not
    be read`` are the same shape to this loop and only the first may be read as
    "accepts any pytest". Measured, the fallback is what keeps httpie/1 at
    ``pytest-json-report==0.2``: 0.1 and 0.2 carry no ``requires_dist`` in their version
    JSON, and their sidecar is the document that says they declare nothing.
    """
    wanted = Version(pytest_version)
    for version in sorted(
        (item for item in versions if _parseable(item)), key=Version, reverse=True
    ):
        try:
            requires, requires_python = metadata_of(name, version)
        except BakeError:
            continue
        if requires_python and python_minor is not None:
            try:
                allowed = Version(f"{python_minor}.0") in SpecifierSet(requires_python)
            except (InvalidSpecifier, InvalidVersion):
                allowed = True
            if not allowed:
                continue
        specifier = _declared_requirement(requires, "pytest")
        if specifier is None or wanted in specifier:
            return version
    raise BakeError(
        f"no release of {name!r} declares a pytest requirement that accepts "
        f"pytest=={pytest_version}"
        + (f" on Python {python_minor}" if python_minor else "")
        + f" (newest {name} is {_newest_release(versions)[0]}). The layer cannot carry "
        "both, and the reporter the oracle is read from has to match the bug's pytest."
    )


def plugin_pins(
    *,
    pytest_version: str,
    python_minor: str,
    releases: Mapping[str, Mapping[str, str]],
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
    taken: Optional[Mapping[str, str]] = None,
) -> list[ResolvedDep]:
    """The report plugin's own pins, for a bug whose pytest is not the harness's.

    [VANTA-WHY] A bug that pins pytest itself changes what the plugin *can* be: every
    release of ``pytest-json-report`` from 0.3 on requires pytest 3.3.2 or newer — so
    httpie/1's ``pytest==3.2.1`` caps it at 0.2 — and from 0.5.0 on it declares a bare
    ``pytest-metadata``, which pip would resolve to 3.1.1, which requires pytest 7.
    Nothing declared either package for this bug, so the choice is recorded as
    :attr:`ResolvedDep.chosen_for_pytest` rather than left to pip, and the pins go
    into the layer's install list like any other dependency.

    Read in order, and each package is only chosen because a previous one declares it:
    the plugin itself is always installed (it is what the oracle's report is read
    from), while ``pytest-metadata`` is pinned only when the chosen plugin release
    actually asks for it — httpie/1's ``pytest-json-report==0.2`` declares no
    ``requires_dist`` at all, so nothing pulls metadata in and none is installed.

    ``taken`` maps every name this bug's own files already decided to the version they
    decided, so a plugin one of them pinned is not chosen twice — but its declared
    requirements are still read, so what *it* pulls in is still decided here.
    """
    decided: dict[str, str] = dict(taken or {})
    pulled_by: list[tuple[str, tuple[str, ...]]] = []
    out: list[ResolvedDep] = []
    for position, name in enumerate(PLUGIN_PACKAGES):
        version = decided.get(name)
        if version is None:
            if position and not any(
                _declared_requirement(requires, name) is not None
                for _, requires in pulled_by
            ):
                continue
            versions = releases.get(name)
            if not versions:
                raise BakeError(
                    f"no release dates for {name!r} were fetched; the layer's pytest is "
                    f"pytest=={pytest_version}, so the plugin cannot be chosen for it"
                )
            version = newest_accepting_pytest(
                name,
                versions,
                pytest_version=pytest_version,
                python_minor=python_minor,
                metadata_of=metadata_of,
            )
            decided[name] = version
            out.append(
                ResolvedDep(
                    name=name,
                    spec=f"{name}=={version}",
                    sources=(PLUGIN_SOURCES[name],),
                    pin=f"{name}=={version}",
                    harness=False,
                    released_at=versions[version][:10],
                    chosen_for_pytest=pytest_version,
                )
            )
        requires, _ = metadata_of(name, version)
        pulled_by.append((name, requires))
    return out


def _pin_version(pin: str) -> str:
    """The version half of a ``name==version`` pin."""
    return pin.partition("==")[2]


def own_pytest_pin(
    refs: Sequence[DepRef], *, harness_pytest: Optional[str] = None
) -> Optional[str]:
    """The pytest version this bug's own files pin **exactly**, or ``None``.

    [VANTA-WHY] Read off the *sources* rather than off the resolved dependencies, so
    ``bake`` can ask the question while it still has a chance to fetch what the answer
    needs: whether the report plugin has to be re-chosen decides whether its release
    dates are fetched at all. A reference pin always resolves to its own version —
    :func:`resolve_pins` exempts it from the date rule and installs it as written — so
    reading the file's text and reading the pin agree.

    A **range** is not a pin and yields ``None``: spacy/4's ``pytest>=4.6.5`` states a
    constraint on a package the interpreter layer already carries, and a range names
    no release for the layer to install. A pin equal to the harness's own version is
    also ``None``: the interpreter layer already has exactly that, so there is no
    plugin to re-choose and re-pinning it would be a no-op on the command line.
    """
    for ref in refs:
        if ref.name != "pytest" or not reference_pin(ref):
            continue
        version = _exact_version(_specifier_of(ref))
        if version is None:
            continue
        if harness_pytest is not None and version == Version(harness_pytest):
            continue
        return str(version)
    return None


def bugsinpy_refs(
    entry: Mapping[str, Any], *, bugsinpy_root: Path
) -> tuple[DepRef, ...]:
    """Every dependency BugsInPy's two per-bug files declare, without the repo.

    [VANTA-WHY] :func:`bug_sources` is the full reader — five sources, two commits,
    a repo clone — and it is what ``bake`` uses. This is the *plan* path's subset: the
    per-bug layer's pytest is decided by BugsInPy's own ``requirements.txt`` /
    ``setup.sh`` alone (:func:`own_pytest_pin`), and the plan has to be able to say so
    before anything is cloned or fetched. Reading the whole set of sources here rather
    than only the ``pytest`` lines keeps one parser per file, and the predicate that
    filters to pytest is the same one ``bake`` uses.
    """
    project = str(entry["project"])
    bug_id = int(entry["bug_id"])
    declared, _ = requirements_refs(
        requirements_for(bugsinpy_root, project, bug_id), source="requirements.txt"
    )
    installed, _ = parse_setup_sh(setup_sh_for(bugsinpy_root, project, bug_id))
    return tuple(declared) + tuple(installed)


def resolve_deps(
    sources: DepSources,
    *,
    releases: Mapping[str, Mapping[str, str]],
    harness_pins: Mapping[str, str],
    python_minor: Optional[str] = None,
    metadata_of: Optional[
        Callable[[str, str], tuple[tuple[str, ...], Optional[str]]]
    ] = None,
    reason_of: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
    audit: Optional[Callable[[str], ImportAudit]] = None,
    dropped: Optional[list[DroppedPin]] = None,
) -> list[ResolvedDep]:
    """Union → conflict check → date pin → the report plugin, for one bug.

    The plugin step runs only when this bug's own files pin pytest exactly and that
    pin is not the interpreter layer's (:func:`own_pytest_pin`): only then is there a
    plugin release to choose. Whatever it returns is *in addition to* the harness's
    own pins, which stay in the list as ``harness=True`` — the interpreter layer still
    carries them, and ``installs`` still excludes them (:func:`installs_of`).

    ``reason_of``, ``audit`` and ``dropped`` are §3.2's unresolvable-pin rule, and they
    come as a triple or not at all: ``dropped`` is the list the records are appended to,
    ``reason_of`` says whether one pin can be installed here at all, and ``audit`` says
    whether anything the graded tests reach imports it (:func:`import_auditor`). The
    rule is applied **around** the date rule, in two passes, because only one of them can
    run before it (:func:`drop_unresolvable_refs`, :func:`drop_unresolvable_deps`) — and
    *not* inside :func:`resolve_pins`, which stays pure and keeps refusing a pin it
    cannot date. Passing the triple is what a caller does when it wants the rule; a
    caller that passes none of it — every test that is about the date rule itself — gets
    the pins the declarations and the dates decide, with nothing dropped and nothing
    raised.

    ``metadata_of`` does double duty: it is the plugin step's "what does this release
    declare about pytest", and it is the *date rule's* case-(a) narrowing — what this
    layer's exact pins declare about each other (:func:`_narrowing_requirements`). One
    reader of ``requires_dist`` and not two, so a caller that passes none gets neither
    consultation.
    """
    merged, conflicts = merge_refs(sources.refs)
    if conflicts:
        raise BakeError(
            f"{sources.project}/{sources.bug_id}: "
            + " | ".join(conflict.message() for conflict in conflicts)
            + " — name the source that should win and the split will be amended rather "
            "than guessed at"
        )
    if dropped is not None:
        merged, dropped_refs = drop_unresolvable_refs(
            merged, reason_of=reason_of, audit=audit
        )
        dropped.extend(dropped_refs)
    resolved = resolve_pins(
        merged,
        releases=releases,
        commit_date=sources.commit_date,
        harness_pins=harness_pins,
        buggy_commit=sources.buggy_commit,
        metadata_of=metadata_of,
    )
    own = own_pytest_pin(sources.refs, harness_pytest=harness_pins.get("pytest"))
    if own is not None:
        resolved = resolved + plugin_pins(
            pytest_version=own,
            python_minor=python_minor or "",
            releases=releases,
            metadata_of=metadata_of or pypi_release_metadata,
            taken={dep.name: _pin_version(dep.pin) for dep in resolved},
        )
    if dropped is not None:
        resolved, dropped_deps = drop_unresolvable_deps(
            resolved, reason_of=reason_of, audit=audit
        )
        dropped.extend(dropped_deps)
    return resolved


@dataclass(frozen=True)
class BuildPin:
    """One build-time tool, pinned by date and installed before the dependencies (§3.2).

    [VANTA-WHY] The environment a tree was built in is not only its dependencies: pip
    builds a source distribution in a *fresh* environment unless told otherwise, and
    that environment is today's. Measured on matplotlib/1: the pinned
    ``numpy==1.18.5`` has no wheel for the sandbox (linux/aarch64, py3.8), so pip built
    it with a setuptools released years after matplotlib's own build, and numpy's
    ``numpy/distutils/mingw32ccompiler.py`` died on a ``distutils`` submodule that
    setuptools no longer carries. So the toolchain is pinned by the same rule as a
    dependency — the newest release on or before the bug's commit date — installed into
    the layer first, and the dependency install runs with ``--no-build-isolation``
    against it.
    """

    name: str
    pin: str
    source: str
    """What asked for this pin: :data:`BUILD_TOOL_SOURCE`, or the line that declared it."""
    released_at: Optional[str] = None
    dependency_pin: Optional[str] = None
    """The dependency pin this one copies, when both name the same package.

    [VANTA-WHY] pandas' ``setup_requires`` and its own dependency list both name numpy,
    and the layer can only hold one: pinning the build requirement by its own date and
    then installing the dependency's numpy over it would compile ``pandas/_libs``
    against a version that is immediately replaced. A build pin for a package the
    dependency set already pins *is* that pin, installed early — the record says which
    one it followed, so "why is numpy at this version" has one answer and not two.
    """
    capability: bool = False
    """Whether the version came from what the tool must be able to *do* (§3.2).

    [VANTA-WHY] Exactly one pin is like this: :data:`PIP_TOOL`, installed because the
    layer's environment needs a pip that still carries :data:`LEGACY_RESOLVER_FLAG`.
    That is a property of the tool and not of the date the bug is from — the bug's own
    date would name a pip from before the flag existed — so no release was fetched to
    choose it, ``released_at`` is empty, and a reader comparing the two keys would
    otherwise find a date that had no part in the decision.
    """
    narrowed_by: tuple[tuple[str, str], ...] = ()
    """The builds whose own requirements moved this pin off the date rule (§3.2).

    [VANTA-WHY] A date is not the only thing a build tool has to satisfy: the build
    itself reads the tools it runs under and refuses to go on. Measured on ``luigi/1``,
    the date rule names ``cython==0.29.7`` and ``numpy==1.18.4`` — a pin of the same
    layer, built here because it has no aarch64 wheel for py3.8 — stops with
    ``RuntimeError: Building NumPy requires Cython >= 0.29.14``. So a build
    requirement the layer's pin does not satisfy wins: the pin moves to the **oldest
    release that satisfies it**, and each ``(builder, specifier)`` pair here says which
    build moved it and with what, in the same shape
    :attr:`ResolvedDep.narrowed_by` records for the dependency half — so
    :func:`pin_narrowings` reads both halves with one spelling.
    """

    @property
    def pin_rule(self) -> str:
        """How the pinned version was chosen, in one phrase, for the provenance."""
        if self.capability:
            return PIP_PIN_RULE
        if self.dependency_pin is not None:
            return "the dependency set's pin for the same package"
        if self.narrowed_by:
            return "the oldest release that satisfies " + "; ".join(
                f"{pin} declares {specifier}" for pin, specifier in self.narrowed_by
            )
        return BUILD_PIN_RULE

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "pin": self.pin,
            "source": self.source,
            "pin_rule": self.pin_rule,
            "released_at": self.released_at,
        }
        if self.dependency_pin is not None:
            data["dependency_pin"] = self.dependency_pin
        if self.narrowed_by:
            data["narrowed_by"] = [
                {"pin": pin, "requires": specifier}
                for pin, specifier in self.narrowed_by
            ]
        return data


def resolve_build_pins(
    refs: Sequence[DepRef],
    resolved: Sequence[ResolvedDep],
    *,
    releases: Mapping[str, Mapping[str, str]],
    commit_date: str,
) -> list[BuildPin]:
    """The build-time tools this layer installs first, each pinned.

    Three kinds of name, one rule each:

    * :data:`PIP_TOOL` — the one name pinned by **capability** rather than by date
      (§3.2): the layer's environment is installed by a pip that still carries
      :data:`LEGACY_RESOLVER_FLAG`, and that is what :data:`PIP_PIN_VERSION` records.
      It needs no release lookup and would get a wrong answer from one, so it is
      emitted first and the date rule is never reached for it.
    * :data:`BUILD_TOOLS` — setuptools, wheel and Cython, the toolchain every bug's
      layer gets, held to the **bug's** commit date. There is no declaration to read
      a date from: the question is "what was available when this tree was built".
    * every name the project declares as a build requirement (:func:`build_refs`),
      held to the date of the commit that declared it, exactly as a dependency is. A
      build requirement only the fix declares was not there for the buggy tree, so
      the buggy date would refuse the very tool the fix's own build used.

    Pure, and takes ``releases`` as an argument for the same reason
    :func:`resolve_pins` does: the cutoff rule is exercised by a test that never
    touches the network.

    A name the dependency set already pins follows that pin (:attr:`BuildPin.dependency_pin`)
    rather than being dated twice — including the case the user's rule singles out,
    "numpy when it is a build requirement": pandas and matplotlib both declare numpy
    as a build requirement, both also depend on it, and the layer ends with the one
    numpy the dependency set resolved, installed early. A name the interpreter layer
    provides is skipped entirely: the harness's pytest is not something a tree's build
    gets to re-pin (§3.2), and :func:`resolve_pins` is where that constraint is
    checked.

    A build tool with no release on or before its cutoff is a ``BakeError`` and flags
    the bug, like every other date-rule refusal: a tree cannot have been built with a
    tool that did not exist yet, and installing today's is the failure this rule
    exists to prevent.

    "Release" is meant literally here too (§3.2): a pre-release or dev release is
    refused unless the refs that declared this name name one
    (:func:`_date_rule_candidate`), so a tree that declared ``cython>=3.0a1`` keeps its
    pre-release and :data:`BUILD_TOOLS` — which nothing declared — never gets one. What
    the toolchain installs is what the tree's own build ran against, and pip would not
    have taken a pre-release of ``setuptools`` for a bare field either.
    """
    by_name = {dep.name: dep for dep in resolved}
    declared: dict[str, list[DepRef]] = {}
    for ref in refs:
        declared.setdefault(canonicalize_name(ref.name), []).append(ref)
    ordered: list[str] = [
        canonicalize_name(PIP_TOOL),
        *(canonicalize_name(name) for name in BUILD_TOOLS),
    ]
    for ref in refs:
        name = canonicalize_name(ref.name)
        if name not in ordered:
            ordered.append(name)

    pins: list[BuildPin] = []
    for name in ordered:
        if name == canonicalize_name(PIP_TOOL):
            # No date and no release lookup: the pin is the capability, and ``sources``
            # / ``releases`` have nothing to say about it.
            pins.append(
                BuildPin(
                    name=PIP_TOOL,
                    pin=f"{PIP_TOOL}=={PIP_PIN_VERSION}",
                    source=PIP_TOOL_SOURCE,
                    capability=True,
                )
            )
            continue
        if name in HARNESS_PACKAGES:
            continue
        from_refs = declared.get(name, [])
        source = from_refs[0].source if from_refs else BUILD_TOOL_SOURCE
        dependency = by_name.get(name)
        if dependency is not None:
            pins.append(
                BuildPin(
                    name=name,
                    pin=dependency.pin,
                    source=source,
                    released_at=dependency.released_at,
                    dependency_pin=dependency.pin,
                )
            )
            continue
        cutoff = max(
            (ref.commit_date or commit_date for ref in from_refs), default=commit_date
        )
        available = releases.get(name)
        if not available:
            raise BakeError(
                f"{source}: no release dates for {name!r} were fetched, so the build "
                "toolchain cannot be pinned (design §3.2 pins by date, not by guess)"
            )
        declared_specifiers = SpecifierSet(
            ",".join(sorted({text for text in map(_specifier_text, from_refs) if text}))
        )
        matching = [
            (version, date)
            for version, date in available.items()
            if _parseable(version)
            and date[:10] <= cutoff
            and _date_rule_candidate(version, declared_specifiers)
        ]
        if not matching:
            newest = _newest_release(available)
            raise BakeError(
                f"{source}: {name!r} has no release on or before {cutoff} (newest is "
                f"{newest[0]}, released {newest[1][:10]}). A tree could not have been "
                "built with a tool that did not exist yet."
            )
        version, released_at = max(matching, key=lambda item: Version(item[0]))
        pins.append(
            BuildPin(
                name=name,
                pin=f"{name}=={version}",
                source=source,
                released_at=released_at[:10],
            )
        )
    return pins


def build_installs(pins: Sequence[BuildPin]) -> tuple[str, ...]:
    """The build pins' install specs, in the order the recipe installs them."""
    return tuple(pin.pin for pin in pins)


def narrow_build_pins(
    pins: Sequence[BuildPin],
    pinned: Sequence[tuple[str, str]],
    *,
    build_requires_of: Callable[[str, str], Sequence[DepRef]],
    python_minor: str,
    releases: Mapping[str, Mapping[str, str]],
) -> list[BuildPin]:
    """Move a build tool to the oldest release the layer's own builds accept (§3.2).

    [VANTA-WHY] §3.2's build-requirement-wins rule, and the one way the date rule is not
    the last word about a *build* pin. Measured on ``luigi/1``: the date rule names the
    toolchain's ``cython==0.29.7``, the layer also holds ``numpy==1.18.4`` — built here,
    because it has no aarch64 wheel for py3.8 — and numpy's own build reads the Cython it
    runs under and stops with ``RuntimeError: Building NumPy requires Cython >=
    0.29.14``. The build is the authority on what it needs: a requirement the layer's pin
    does not satisfy wins, and the pin moves to the **oldest release that satisfies it**
    — oldest, not newest, because the whole point of pinning by date is to stay as close
    to the bug's own era as the requirement allows, and ``>=0.29.14`` is a bound the
    build needs rather than a version it wants. **Every** claim on the name bounds the
    move and not only the ones the held pin fails: the release the pin moves to has to
    satisfy all of them, so a clause the held version happens to hold (``<0.29.16``
    beside ``>=0.29.14``) bounds it too. The move itself happens only when at least one
    claim does not hold as the pin stands — a name every claim already satisfies is left
    exactly where the date rule put it.

    ``pinned`` is the layer's whole pin list (:func:`pinned_parts` of the toolchain and
    the installs), because *any* of them may be the pin doing the building: numpy is a
    dependency here, not a tool. Only a build pin that the date rule alone chose can
    move — a pin copying the dependency set's answer (:attr:`BuildPin.dependency_pin`)
    and the capability pin (:data:`PIP_TOOL`) are left alone, or the layer would install
    the toolchain's version and then replace it with the dependency's, compiling against
    one release and importing another. A name the layer's own pins hold too low *and*
    resolve as a dependency therefore stays where it is; the record of the requirement
    is not an EARLY mapping either (:func:`early_install_pins` checks the clause), so
    nothing claims a satisfaction the build does not have.

    Each move is recorded on the pin itself (:attr:`BuildPin.narrowed_by`) in the shape
    :attr:`ResolvedDep.narrowed_by` uses, which is what makes it print as NARROWED
    through :func:`pin_narrowings` and never as EARLY.

    A build requirement nothing on PyPI satisfies is a ``BakeError`` and flags the bug,
    like every other unsatisfiable declaration (§3.2): the layer would be installed and
    then fail its own build, and the harness's job is to refuse that before the build
    runs. ``releases`` is the caller's, so the whole decision is exercised by a test
    that never touches the network.
    """
    movable = {
        canonicalize_name(pin.name): pin
        for pin in pins
        if pin.dependency_pin is None and not pin.capability
    }
    claims: dict[str, list[tuple[str, str]]] = {}
    for name, version in pinned:
        own = canonicalize_name(name)
        spelling = f"{name}=={version}"
        try:
            refs = tuple(build_requires_of(name, version))
        except BakeError:
            continue
        refs, _ = marker_skips(refs, python_minor=python_minor)
        for ref in refs:
            required = canonicalize_name(ref.name)
            if required == own or required not in movable:
                continue
            # Every claim is collected, including one the held pin already holds: what
            # moves has to satisfy *all* of them, and a clause the held version happens to
            # satisfy still bounds the release the pin moves to. Only the *decision* to
            # move reads the held version, below.
            claims.setdefault(required, []).append((spelling, str(_specifier_of(ref))))
    if not claims:
        return list(pins)
    out: list[BuildPin] = []
    for pin in pins:
        found = claims.get(canonicalize_name(pin.name))
        if not found or pin.dependency_pin is not None or pin.capability:
            out.append(pin)
            continue
        specifier = SpecifierSet(",".join(sorted({text for _, text in found})))
        if _specifier_accepts(specifier, _pin_version(pin.pin)):
            # Every claim already holds: the date rule's answer stands, and the move —
            # which would have to satisfy all of them anyway — is not made.
            out.append(pin)
            continue
        available = releases.get(canonicalize_name(pin.name))
        if not available:
            raise BakeError(
                f"{pin.source}: no release dates for {pin.name!r} were fetched, so the "
                "requirement this layer's own pins declare for it cannot be satisfied "
                "(design §3.2)"
            )
        matching = [
            version
            for version in available
            if _parseable(version)
            and _date_rule_candidate(version, specifier)
            and specifier.contains(version)
        ]
        if not matching:
            newest = _newest_release(available)
            raise BakeError(
                f"{found[0][0]} declares {specifier} for {pin.name!r}, and no release "
                f"satisfies it (newest is {newest[0]}, released {newest[1][:10]}). A "
                "layer cannot be built under a tool the requirement refuses."
            )
        version = min(matching, key=Version)
        out.append(
            replace(
                pin,
                pin=f"{pin.name}=={version}",
                released_at=available[version][:10],
                narrowed_by=tuple(dict.fromkeys(found)),
            )
        )
    return out


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
    minor = minor_version(str(entry["python_version"]))
    notes: list[str] = []
    refs: list[DepRef] = []
    skipped: list[SkippedLine] = []
    build_declared: list[DepRef] = []

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
    try:
        import_name = (
            import_name_of(buggy_setup)
            if buggy_setup is not None
            else import_name_of_pyproject(buggy_pyproject)
        )
    except BakeError:
        # Last resort, and only when the metadata itself cannot be read: the split may
        # carry a reviewed ``import_package`` for this bug (recorded in its
        # ``reviewed_changes``). Read exactly as a ``name=`` literal is, and the note
        # says the metadata was not what named it, so the override never looks like a
        # fact the file stated.
        override = entry.get("import_package")
        if override is None:
            raise
        import_name = canonicalize_name(str(override).strip()).replace("-", "_")
        notes.append(
            f"{project}/{bug_id}: the metadata at {buggy} did not name the import "
            f"package, so the reviewed split override import_package={override!r} was "
            "used; it is recorded in the split's reviewed_changes (design §3.2)."
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
        # The same two files' **build** requirements, read here so they are read at
        # the same two commits and stamped the same way. They are not dependencies and
        # never enter ``refs``: they name the tools the tree is built with, which is
        # §3.2's separate question and gets its own resolution
        # (:func:`resolve_build_pins`).
        build_declared += _stamped(
            build_refs(setup_py, pyproject, notes), commit=sha, commit_date=date
        )

    # 5. The test side's own imports, at the fixed commit. The tree's own modules
    # are excluded too: the test side is not the only thing that can put a name on
    # ``sys.path`` (fastapi's own ``scripts/test.sh`` adds ``docs_src``), and a name
    # the tree ships is a module it provides, never a distribution to install.
    tree_modules = sorted(tree_provided_modules(repo, fixed))
    closure_refs, optional, closure_notes = test_import_closure(
        files=test_side_paths(repo, fixed),
        read_file=partial(tree_mod.read_file_at, repo, fixed),
        import_name=import_name,
        distributions=module_distributions,
        tree_modules=tree_modules,
    )
    refs += _stamped(closure_refs, commit=fixed, commit_date=fixed_date)
    notes += closure_notes

    # §3.2's marker rule, applied once over everything the five sources declared — the
    # tree's own and its build requirements alike, because a ``setup_requires`` entry is
    # read as something to *install* just as an ``install_requires`` one is
    # (:func:`marker_skips`). Read here rather than in each parser so the rule has one
    # home; the refs still carry their markers verbatim, and the dropped ones are
    # recorded, not printed.
    refs, marker_skipped = marker_skips(refs, python_minor=minor)
    build_declared, build_marker_skipped = marker_skips(
        build_declared, python_minor=minor
    )
    skipped += marker_skipped + build_marker_skipped

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
        build_refs=tuple(build_declared),
        tree_modules=tuple(tree_modules),
    )


def installs_of(resolved: Sequence[ResolvedDep]) -> tuple[str, ...]:
    """The pins the layer's recipe installs.

    One definition, used by both the recipe (via :class:`BugDeps`) and the
    provenance, so "what was recorded" and "what pip was told" cannot drift apart.
    """
    return tuple(dep.pin for dep in resolved if not dep.harness)


def dropped_dependents(
    installs: Sequence[str],
    dropped: Sequence[DroppedPin],
    *,
    metadata_of: Callable[[str, str], tuple[tuple[str, ...], Optional[str]]],
) -> tuple[str, ...]:
    """The pins whose **own** release declares a name §3.2 dropped, as spec strings.

    [VANTA-WHY] A drop removes a name from ``installs``; it does not remove the
    requirement another pin wrote for that name, so pip resolves it at install time —
    and at the newest release of the day, because a dropped name is in no pin list and
    therefore in no ``-c`` constraints file either. Measured, ``sanic/1``: the date
    rule's ``cryptography==2.8`` and its declared ``cryptography==2.9.2`` are both
    unbuildable here and dropped, ``pyopenssl==19.1.0`` declares ``cryptography (>=2.8)``
    and ``service-identity==18.1.0`` declares it too, and the layer ended up holding
    ``cryptography 47.0.0`` — a release newer than the layer's own Python, whose
    ``requires_dist`` names a package the layer had just removed as an orphan, which is
    what its own ``pip check`` reported.

    So the pins that declare a dropped name are installed ``--no-deps`` in their own
    pass (:func:`requirements_recipe`). Nothing is lost by that: everything those pins
    require is already in ``installs`` — the closure walk pinned it — except the dropped
    name itself, which is precisely what may not be installed. ``pip check`` then reports
    the drop and nothing else, which is the one reading the gate tolerates by name
    (:data:`CASE_DROPPED_DEPENDENCY`).

    A pin is a dependent when its **own** release's ``requires_dist`` names a dropped
    name (:func:`_declared_requirement`), read at the version ``installs`` carries. A
    release whose metadata cannot be read is skipped rather than guessed at: the walk
    already recorded that release's declarations, and inventing a dependency here would
    be the second reader the rest of this module exists to avoid.
    """
    names = {canonicalize_name(record.name) for record in dropped}
    if not names:
        return ()
    out: list[str] = []
    for spec in installs:
        parts = _pinned_parts(spec)
        if parts is None:
            continue
        name, version = parts
        try:
            requires, _requires_python = metadata_of(name, version)
        except BakeError:
            continue
        if any(
            _declared_requirement(requires, dropped_name) is not None
            for dropped_name in names
        ):
            out.append(spec)
    return tuple(out)


def deps_provenance(
    sources: DepSources,
    resolved: Sequence[ResolvedDep],
    *,
    harness_pins: Mapping[str, str],
    build_pins: Sequence[BuildPin] = (),
    early: Sequence[EarlyInstall] = (),
    dropped: Sequence[DroppedPin] = (),
    no_deps: Sequence[str] = (),
    provisioned: Sequence[str] = (),
    orphans: Sequence[OrphanedPin] = (),
    tree_satisfied: Sequence[TreeSatisfied] = (),
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

    ``build_pins`` is the layer's other environment: not what the suite imports, but
    what the tree's own builds run *under* — the toolchain plus every build requirement
    the project declares, each pinned by date and installed before the dependencies
    (§3.2). It is a separate key rather than a flag on ``dependencies`` because it
    answers a different question: a reader asking "which Cython compiled
    ``pandas/_libs``" is not asking what pandas imports.

    ``early_pins`` is the ordering half of the same story and the one thing the other
    keys cannot show: a subset of ``installs`` that the recipe installs **first**,
    because a pin in this layer's own build declares it (``cryptography==2.9.2`` wants
    cffi, the layer holds ``cffi==1.14.0``) and under ``--no-build-isolation`` an
    unsatisfied declaration is fetched by ``easy_install`` at today's version instead
    (§3.2, :func:`early_install_pins`). Each entry names the pin that wants it, what it
    declared, and the pin that answers it — the same record the preflight prints, kept
    next to the pins it reorders.

    ``dropped`` is the other thing that is not in ``installs``, and it is the one
    exception to "a declared dependency is installed": a pin nothing on this image can
    install, which §3.2's unresolvable-pin rule drops *only* when no file in the bug's
    own tree imports the module it provides (:func:`decide_unresolvable`). The key is
    always present, empty when nothing was dropped, because "nothing was dropped" and
    "this file predates the rule" have to be distinguishable by a reader — and a layer
    whose install list is shorter than its declarations needs the record of why.

    ``no_deps`` is the other half of ``dropped`` and the one pass the pin list cannot
    describe (:func:`dropped_dependents`): the spec strings of the pins whose **own**
    release declares a dropped name. They are in ``installs`` — they are installed —
    but ``requirements_recipe`` moves them out of the main pass into their own
    ``--no-deps`` one, because a pin that declares a dropped name is what makes pip
    resolve that name at install time, and to the newest release of the day since the
    dropped name is in no pin list and in no constraints file. Measured, ``sanic/1``'s
    ``pyopenssl==19.1.0`` declares ``cryptography (>=2.8)`` and the layer ended up with
    ``cryptography 47.0.0``. The key is always present, empty when nothing was dropped.

    ``pip_check`` (:data:`PIP_CHECK_KEY`) is the layer's **own** verdict on itself, read
    from ``pip check`` after the install and written back here by
    :func:`record_pip_check`. It starts empty and stays empty for a layer whose install
    had nothing to report, which is most of them — and the preflight's reading of the
    same question from PyPI's metadata is not this key: that one is in the layer's
    ``.preflight.json``, and it is a prediction. This one is what pip did. For a layer
    that dropped something, the one expected reading is the dropped name itself
    (``<package> <ver> requires <dropped name>, which is not installed``), which
    :data:`CASE_DROPPED_DEPENDENCY` tolerates by name and which the freeze gate would
    otherwise have to fail on.

    ``provisioned`` is the mirror of ``harness_pins``: the canonical names the
    interpreter layer is predicted to hold and this layer neither pins nor picks
    (:func:`transitive_pins`) — the harness pins, everything their releases require at
    the versions the interpreter layer has, minus every name this layer decides for
    itself. It is written down because the freeze gate (:func:`unpinned_freeze`) exempts
    exactly these names from "this layer has to pin what it installs", and an exemption
    nothing records is one no reader can check: a name here that the interpreter layer
    does **not** in fact hold is a distribution this layer inherited from nowhere, which
    is the failure the gate exists to catch and would otherwise be its blind spot.

    ``orphans`` is the halves of that same prediction this layer may **not** inherit, and
    it is why the two keys are not one (:data:`ORPHAN_RULE`): a name the interpreter layer
    holds as a dependency of a harness pin this layer re-pins to another release is not
    something it provides the layer *with* — it is the old release's residue, no pin of the
    layer names it, and ``pip`` leaves it behind because it does not uninstall what it
    replaces. Each is an :class:`OrphanedPin`: the recipe uninstalls it by name, and it is
    **not** in ``provisioned``, so the freeze gate reports one whose removal did not take.
    The record carries ``declared_by``, ``declared`` and ``replaced_by`` — the harness pin
    whose release brought it in, what that release declared, and the pin this layer
    installed over it — because that is the half a reader cannot recover from the layer.
    Like ``dropped``, the key is always present and empty when nothing was removed, so
    "nothing was orphaned" and "this file predates the rule" are distinguishable.

    ``tree_satisfied`` is the third place a name can come from, and the only one that is
    neither installed nor inherited (:data:`TREE_SATISFIED_RULE`): a requirement a release
    declared that the **tree under test** provides for itself, which the closure walk
    therefore leaves unpinned. Measured, ``pandas/1`` is the case — ``statsmodels==0.11.1``
    and ``xarray==0.15.1`` each declare ``pandas``, and a released pandas installed over
    the tree would have been the wrong source for the suite that grades the bug. Each
    record is a :class:`TreeSatisfied` naming what asked for it and how it asked, because
    that is the half a reader cannot recover from the layer. Like ``dropped``, the key is
    always present and empty when the tree answered nothing, so "the tree provided
    nothing" and "this file predates the rule" are distinguishable.
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
        "provisioned": list(provisioned),
        "installs": list(installs_of(resolved)),
        "dropped": [pin.as_dict() for pin in dropped],
        "no_deps": list(no_deps),
        "orphans": [pin.as_dict() for pin in orphans],
        "tree_satisfied": [item.as_dict() for item in tree_satisfied],
        "build_pins": [pin.as_dict() for pin in build_pins],
        "early_pins": [item.as_dict() for item in early],
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
        PIP_CHECK_KEY: [],
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

    ``build_pins`` is installed *before* ``installs`` and is in the record for the
    same reason: ``requirements_recipe`` runs the two in that order and
    ``--no-build-isolation`` is what makes the order mean anything (§3.2).

    ``early_pins`` is a subset of ``installs`` the recipe installs *before* the main
    list (:func:`early_install_pins`): a pin this layer's own build declares, which has
    to be there when that build runs rather than beside it. It is carried rather than
    recomputed by :func:`layer_recipes`, for the reason ``installs`` is: the record
    next to the layer is the resolution's own answer, and a second reader of the sdists
    would be a second chance to install an order nobody recorded.
    """

    entry: dict[str, Any]
    checkpoint: str
    installs: tuple[str, ...]
    provenance: dict[str, Any]
    build_pins: tuple[str, ...] = ()
    early_pins: tuple[str, ...] = ()
    no_deps_pins: tuple[str, ...] = ()
    """The pins whose *own* release declares a name §3.2 dropped, so they are installed
    with ``--no-deps`` (:func:`dropped_dependents`).

    Spec strings, because the recipe installs them: they are in ``installs`` and in the
    freeze gate's pin list — they are installed, just not with their dependencies — and
    what ``requirements_recipe`` does with them is move them out of the main pass into
    their own ``--no-deps`` one, which is what stops pip resolving the dropped name at
    the newest release of the day. Empty for every layer with nothing dropped, which is
    most of them.
    """
    orphans: tuple[str, ...] = ()
    """The inherited names the layer's recipe has to **remove**, not install (§3.2).

    Canonical names of the interpreter layer's distributions that this layer's own pins
    do not contain and that nothing it keeps still needs (:class:`OrphanedPin`):
    ``requirements_recipe`` uninstalls each one after every install and before its ``pip
    check``, and the freeze gate exempts none of them. Empty for every layer that does not
    re-pin a harness package, which is most of them.
    """


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
    metadata_of: Callable[
        [str, str], tuple[tuple[str, ...], Optional[str]]
    ] = pypi_release_metadata,
    build_requires_of: Callable[
        [str, str], Sequence[DepRef]
    ] = pypi_build_requirements,
    classify: Optional[Callable[[str, str, str], PreflightPin]] = None,
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

    A bug that declares a pin *nothing on this image can install* is the same kind of
    answer and reaches the same place, by §3.2's unresolvable-pin rule: dropped with a
    record when nothing the graded tests reach imports it, flagged when something does
    (:func:`decide_unresolvable`). The rule's callables are built here, per bug —
    ``reason_of`` from the ``releases`` this bug just fetched and the run's shared
    preflight classifier, ``audit`` from the bug's own two trees and the test file the
    split declares — and :func:`resolve_deps` applies them around the date rule.

    ``fetch`` and ``metadata_of`` are injected so the entire chain — union, conflict,
    cutoff, harness exception, the report plugin's own pin — is testable without the
    network: the first answers "which releases exist, and when", the second "what does
    one release declare". ``classify`` answers "what does this release install as
    here"; ``bake`` builds one per run and shares it with the preflight table, so a pin
    is classified once however many times it is asked about. Injecting it — or a stub
    of it — is what keeps these tests offline.

    ``build_requires_of`` answers the third question, "what does this release need in
    order to be built", and it is used for exactly one decision: which of this layer's
    own pins some *other* pin's build needs installed before it
    (:func:`early_install_pins`), so the layer's ``early_pins`` can say so. It is read
    from the release's sdist and never executed; ``bake`` passes the run's
    :func:`cached_build_requirements`, so a pin two bugs hold is downloaded once.
    """
    out: list[BugDeps] = []
    flagged: list[FlaggedBug] = []
    distributions = split.module_distributions()
    classifier = classify or preflight_classifier()
    for entry in entries:
        key = f"{entry['project']}/{entry['bug_id']}"
        # The failing test files, which is what :func:`import_auditor` roots its closure
        # at: the split's ``test_file`` is BugsInPy's own statement of where the
        # bug-revealing test lives — the same declaration the oracle's FAIL_TO_PASS is
        # filtered by in ``prepare`` — so "the tests this bug is graded on" and this
        # list are one thing and not two.
        test_files = [str(entry["test_file"])] if entry.get("test_file") else []
        try:
            repo = repo_for(str(entry["project"]))
            sources = bug_sources(
                entry,
                bugsinpy_root=bugsinpy_root,
                repo=repo,
                module_distributions=distributions,
            )
            minor = minor_version(str(entry["python_version"]))
            pin = pins_for_python(split, minor)
            harness_pins = {"pytest": pin.pytest, "pytest-json-report": pin.json_report}
            # The build toolchain's names are fetched from PyPI too: they are pinned
            # by date like everything else, and a name the harness provides is left
            # alone (:func:`resolve_build_pins`).
            #
            # :data:`PIP_TOOL` is deliberately **not** here, and its absence is the
            # whole of "the pip pin needs no release dates": it is pinned by capability
            # — the newest release that still carries :data:`LEGACY_RESOLVER_FLAG` — and
            # :func:`resolve_build_pins` emits it without ever reading ``releases``
            # (§3.2). Fetching it would only add a name to a dict nothing looks it up
            # in; the date rule would answer a question about that name that has
            # nothing to do with why the version was chosen.
            build_names = sorted(
                {
                    canonicalize_name(name)
                    for name in (*BUILD_TOOLS, *(ref.name for ref in sources.build_refs))
                }
                - HARNESS_PACKAGES
            )
            # A bug that pins pytest itself needs pytest's **and** the *plugin's*
            # release dates, and both are asked here, before the fetch
            # (:func:`own_pytest_pin`), because both follow from the same fact: a pin
            # that is not the interpreter layer's own version.
            #
            # pytest itself, even though it is a :data:`HARNESS_PACKAGES` name the
            # interpreter layer already carries: the bug's ``pytest==X`` is a reference
            # pin (:func:`reference_pin`), so :func:`resolve_pins` exempts it from the
            # date rule and installs it as written — but it still reads
            # ``releases["pytest"]`` to check the release exists and to record the date
            # it was released. The harness-provides branch is gated on
            # ``not reference_pin(ref)``, so an exact pin does not take it, and
            # excluding the name here is what made fastapi/1, luigi/1, pandas/1 and
            # sanic/1 fail with "no release dates for 'pytest' were fetched".
            #
            # The plugin packages: which pytest-json-report/pytest-metadata release the
            # layer can carry is a question about that same pin. A bug without such a
            # pin keeps the interpreter layer's pair and needs none of the three — and
            # neither does one whose pin is exactly the harness's, which
            # :func:`own_pytest_pin` reads as no pin at all. No bug in either split
            # pins pytest at the layer's own version; if one did, it would need
            # pytest's dates for the reason above and this condition would have to be
            # ``own_pytest is not None or any(reference pin on pytest)``.
            own_pytest = own_pytest_pin(sources.refs, harness_pytest=pin.pytest)
            pytest_dates = (
                {"pytest", *PLUGIN_PACKAGES} if own_pytest is not None else set()
            )
            needed = sorted(
                {ref.name for ref in sources.refs if ref.name not in HARNESS_PACKAGES}
                | set(build_names)
                | pytest_dates
            )
            releases = {name: fetch(name) for name in needed}
            dropped: list[DroppedPin] = []
            # §3.2's unresolvable-pin rule's two callables, bound once for this bug: the
            # date rule's own mapping answers "does this release exist", the run's shared
            # classifier answers "what does it install as", and the bug's own two trees
            # answer "is anything the graded tests reach importing it" — lazily, because
            # the walk is a file read per ``.py`` and most bugs never ask. Bound here
            # rather than inline in the ``resolve_deps`` call because **both** halves of
            # the layer go through them: the declared pins inside ``resolve_deps`` and
            # the closure walk's picks below (:func:`drop_unresolvable_deps`), which is
            # what makes a transitive pick's audit the same rule and not a second one.
            reason_of = partial(
                uninstallable_reason,
                releases=releases,
                classify=classifier,
                python_minor=minor,
            )
            audit = import_auditor(
                repo,
                (sources.buggy_commit, sources.fixed_commit or ""),
                distributions=distributions,
                test_files=test_files,
            )
            resolved = resolve_deps(
                sources,
                releases=releases,
                harness_pins=harness_pins,
                python_minor=minor,
                metadata_of=metadata_of,
                reason_of=reason_of,
                audit=audit,
                dropped=dropped,
            )
            built = resolve_build_pins(
                sources.build_refs,
                resolved,
                releases=releases,
                commit_date=sources.commit_date,
            )
            # §3.2's build-requirement-wins rule, run before the toolchain's pin list
            # reaches anything else: a build tool one of this layer's own pins refuses to
            # be built under is moved to the oldest release that pin accepts, and the
            # record of the move travels on the pin itself (:func:`narrow_build_pins`).
            # It reads the pin list the *date rule* produced, which is why the two lines
            # below are computed twice — the second pair is what the walk and the recipe
            # see.
            built = narrow_build_pins(
                built,
                pinned_parts(*build_installs(built), *installs_of(resolved)),
                build_requires_of=build_requires_of,
                python_minor=minor,
                releases=releases,
            )
            # The layer's own pin list, before the closure walk adds to it — the walk
            # starts *at* these pins and stops at them, so a list that already held its
            # own answer would be a list about itself.
            build_specs = build_installs(built)
            own_pins = pinned_parts(*build_specs, *installs_of(resolved))
            # §3.2's closure walk, inside the ``try`` because a transitive requirement
            # this image cannot satisfy is one bug's flag and not the batch's — the same
            # answer ``resolve_deps``'s own refusal gets. ``cutoffs`` is per pin, not per
            # layer: a pin's own requirements come from the release that pin *is*, so the
            # window they are held to is that release's — which for a reference pin
            # installed as written is later than the buggy commit, and for a build pin is
            # its own release date (:func:`pin_cutoff`).
            transitive, provisioned, orphaned, tree_satisfied_pins = transitive_pins(
                own_pins,
                harness_pins=harness_pins,
                commit_date=sources.commit_date,
                python_minor=minor,
                releases=releases,
                fetch=fetch,
                metadata_of=metadata_of,
                # The names the tree under test provides for itself, so the closure walk
                # never pins one (:data:`TREE_SATISFIED_RULE`): the project's own import
                # name, and every top-level module the tree ships — the same exemption
                # ``bug_sources`` already makes for the test side's import closure
                # (:func:`tree_provided_modules`), applied to the other half of the layer.
                # Measured, this is what ``pandas/1``'s ``pandas==1.0.3`` was: a
                # transitive pick for the very distribution the bug is graded on, since
                # ``statsmodels==0.11.1`` and ``xarray==0.15.1`` each declare ``pandas``.
                tree_satisfied={
                    name: name
                    for name in (sources.import_name, *sources.tree_modules)
                },
                cutoffs={
                    **{
                        canonicalize_name(tool.name): tool.released_at
                        for tool in built
                        if tool.released_at
                    },
                    **{
                        canonicalize_name(dep.name): date
                        for dep in resolved
                        if (date := pin_cutoff(dep)) is not None
                    },
                },
            )
            # The closure walk's own picks get the same audit, and they are the half of
            # the layer it was measured on: ``luigi/1`` and ``sanic/1`` both install
            # ``cryptography==2.8`` because ``moto==1.3.7``/``pyopenssl==19.1.0``/
            # ``service-identity==18.1.0`` declare it, and that is the release the same
            # preflight marks :data:`UNBUILDABLE` here — while each layer's *own*
            # ``cryptography`` pin is dropped for exactly that reason. A pick nothing on
            # this image can install is a pin like any other: dropped with a record when
            # nothing the graded tests reach imports it, and the bug flagged when
            # something does (:func:`decide_unresolvable`). The record lands in the same
            # ``dropped`` list, so the provenance and the preflight's drop report read
            # the two halves as one answer.
            transitive, dropped_transitive = drop_unresolvable_deps(
                transitive, reason_of=reason_of, audit=audit
            )
            dropped.extend(dropped_transitive)
        except BakeError as exc:
            flagged.append(FlaggedBug(key=key, reason=f"{type(exc).__name__}: {exc}"))
            continue
        # The transitive picks are installed, recorded and gated exactly as the declared
        # ones are: they are the same kind of pin, chosen by the same rule, and the only
        # thing that distinguishes them is that no file in the bug's tree asked for them
        # (:attr:`ResolvedDep.required_by`). The orphans are the other half of the same
        # walk and the one part of it that is *removed*: they are carried to the recipe by
        # name and recorded with their reason, and they are deliberately not in
        # ``provisioned``, which is what makes the freeze gate their second reader.
        deps = (*resolved, *transitive)
        installs = installs_of(deps)
        pins = pinned_parts(*build_specs, *installs)
        # §3.2's dropped-dependency pass, and the one thing the pin list cannot say for
        # itself: a drop removes a name from ``installs``, it does not remove the
        # requirement another pin wrote for that name, so pip installs a release of it —
        # the newest of the day, since the name is in no pin list and in no constraints
        # file. ``sanic/1`` measured it: ``cryptography`` is dropped for both its declared
        # and its transitive pick, ``pyopenssl==19.1.0`` and ``service-identity==18.1.0``
        # declare it, and the layer ended up with ``cryptography 47.0.0``. The pins whose
        # own release declares a dropped name are therefore installed with ``--no-deps``
        # in their own pass (:func:`requirements_recipe`), which is what stops pip
        # resolving it at all; ``pip check`` then reports the dropped dependency itself and
        # nothing else, which is what the gate tolerates by name
        # (:data:`CASE_DROPPED_DEPENDENCY`).
        no_deps = dropped_dependents(installs, dropped, metadata_of=metadata_of)
        # §3.2's marker rule over the pinned releases' own build declarations, which is
        # this rule's early-pass half: a ``[build-system].requires`` entry that does not
        # hold on this image is not something the build orders anything for. The dropped
        # requirements are folded into the same ``skipped`` record the tree's declarations
        # write to, because a reader asking "what was read and not installed" is asking
        # one question of the whole layer.
        early_skipped: list[SkippedLine] = []
        early = early_install_pins(
            tuple(
                (name, version)
                for name, version in pins
                if classifier(name, version, minor).kind == SDIST_ONLY
            ),
            pins,
            build_requires_of=build_requires_of,
            python_minor=minor,
            marker_skipped=early_skipped,
        )
        if early_skipped:
            sources = replace(sources, skipped=(*sources.skipped, *early_skipped))
        out.append(
            BugDeps(
                entry=dict(entry),
                checkpoint=sources.checkpoint,
                installs=installs,
                provenance=deps_provenance(
                    sources,
                    deps,
                    harness_pins=harness_pins,
                    build_pins=built,
                    early=early,
                    dropped=dropped,
                    no_deps=no_deps,
                    provisioned=provisioned,
                    orphans=orphaned,
                    tree_satisfied=tree_satisfied_pins,
                ),
                build_pins=build_specs,
                # Spec strings, not ``pinned_parts``' ``(name, version)`` pairs: the
                # recipe installs these, and ``satisfied_by`` is already the pin's own
                # spelling. One pin can satisfy two builds, so the list is de-duplicated.
                early_pins=tuple(dict.fromkeys(item.satisfied_by for item in early)),
                no_deps_pins=no_deps,
                orphans=tuple(pin.name for pin in orphaned),
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


def record_pip_check(
    deps_dir: Path, checkpoint: str, conflicts: Sequence[PinConflict]
) -> Path:
    """Write a layer's own ``pip check`` conflicts back into its provenance.

    [VANTA-WHY] Back into the file rather than into a new one, and after the build
    rather than before it: the layer's reading exists only once pip has run, and
    ``prepare`` reads one provenance per bug. A second file would be a second thing to
    find, and the key is in :func:`deps_provenance` from the start precisely so that
    "nothing to report" (``[]``) and "this file predates the reading" (absent) stay
    distinguishable.

    Rewrites the whole file through :func:`load_deps`, so a provenance that is not one
    fails here the same way it fails everywhere else, and only :data:`PIP_CHECK_KEY` is
    touched — every other key stays byte-for-byte what the bake resolved.
    """
    data = load_deps(deps_dir, checkpoint)
    data[PIP_CHECK_KEY] = [conflict.as_dict() for conflict in conflicts]
    path = deps_path(deps_dir, checkpoint)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def layer_pin(deps: Mapping[str, Any], name: str) -> Optional[str]:
    """The version of ``name`` the layer's recipe installs, from its provenance.

    [VANTA-WHY] A bug whose own files pin a harness package is installed with *that*
    version, on top of the interpreter layer's default (§3.2), so the split's entry —
    which declares the interpreter layer's pair, one pair per Python minor — no longer
    says what the layer carries. The provenance does: it is written from the same
    resolved list :func:`installs_of` reads, so a run summary reporting the layer's
    pytest takes it from here rather than from the split.

    Three answers, in order of what the record actually says:

    * a dependency the bug's own sources named — its pin, or the interpreter layer's
      when §3.2 left it there;
    * a package no source named but the interpreter layer provides — recorded in
      ``harness_pins``, which is the pair that layer was *built* with, so it is what
      the bug's layer carries. Falling back to it is not borrowing the split's value:
      the split's entry is per Python minor and the same number only by convention,
      while this is the pin this bug was resolved against;
    * ``None`` when neither names it — a row that claims a version the layer never
      carried is worse than one that says it does not know.
    """
    harness_pins = deps.get("harness_pins") or {}
    for dep in deps.get("dependencies") or []:
        if dep.get("name") != name:
            continue
        if dep.get("provisioned_by") == "interpreter layer":
            return harness_pins.get(name)
        return _pin_version(str(dep.get("pin") or "")) or None
    return harness_pins.get(name)


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
        minor = minor_version(entry["python_version"])
        layers.append(
            (
                name,
                python_checkpoint_name(minor),
                requirements_recipe(
                    item.installs,
                    build_pins=item.build_pins,
                    early_pins=item.early_pins,
                    no_deps_pins=item.no_deps_pins,
                    orphans=item.orphans,
                    minor=minor,
                ),
            )
        )
    return layers


def render_plan(
    split: Split,
    *,
    pythons: list[str],
    entries: Sequence[dict[str, Any]],
    bugsinpy_root: Optional[Path] = None,
) -> str:
    """What the bake would create, and from which pins. No side effects.

    A bug's ``pytest`` cell is the version its layer will carry: its **own** pin when
    BugsInPy's files write one, read here from those two files alone — no repo, no
    network — because that is what ``bake`` installs on top of the interpreter layer
    (§3.2), and the interpreter layer's default otherwise. The split's entry declares
    the *interpreter* layer's pair (one pair per Python minor, checked by
    :func:`pins_for_python`), so printing it as the bug's own would name a pytest the
    layer will not have.

    The ``json-report`` cell of such a bug reads ``at bake``: which release the plugin
    can be is PyPI's answer, taken when this bug's layer is baked, and a plan that
    guessed it would be a second, unrecorded decision about the same question.
    ``bugsinpy_root`` is optional so the plan still renders where the clone is not
    there yet; the cell then falls back to the inherited pin.
    """
    lines = ["| layer | base | python | pytest | json-report |", "|---|---|---|---|---|"]
    for minor in pythons:
        pin = pins_for_python(split, minor)
        lines.append(
            f"| {python_checkpoint_name(minor)} | {DEFAULT_BASE} | {minor} | "
            f"{pin.pytest} | {pin.json_report} |"
        )
    for entry in entries:
        inherited = str(entry["pytest_version"])
        own = None
        if bugsinpy_root is not None:
            own = own_pytest_pin(
                bugsinpy_refs(entry, bugsinpy_root=bugsinpy_root),
                harness_pytest=inherited,
            )
        pytest_cell = own or inherited
        plugin_cell = "at bake" if own else str(entry["json_report_version"])
        lines.append(
            f"| {bug_checkpoint_name(entry)} | "
            f"{python_checkpoint_name(minor_version(entry['python_version']))} | "
            f"{entry['python_version']} | {pytest_cell} | {plugin_cell} |"
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


def summarise(
    actions: Sequence[LayerAction],
    *,
    failed: Sequence[str] = (),
    attempted: Optional[Sequence[str]] = None,
    survived: Sequence[str] = (),
) -> str:
    """The bake's one-line-per-layer record, for the log and for the failure path.

    [VANTA-WHY] Every label is a statement about what happened, and the earlier
    version's ``[done]`` was not: it was printed for any action whose plan said
    "build", including the four layers *after* the one that failed, none of which
    was ever attempted. A reader took that summary for a list of layers that exist —
    which is the one thing ``prepare`` acts on. So ``done`` now requires the build to
    have been attempted and to have returned, ``not started`` is its own label, and a
    delete-only action says ``deleted`` instead of hiding behind ``skipped``.

    ``attempted`` is *which* build actions were reached, in order. ``None`` means
    "not recorded — treat every planned build as attempted", which is what a caller
    summarising a plan rather than a run is asking for.
    """
    reached = None if attempted is None else set(attempted)
    failed_set = set(failed)
    lines = ["bake summary:"]
    for action in actions:
        if action.name in failed_set:
            state = "FAILED"
        elif action.build:
            state = "done" if reached is None or action.name in reached else "not started"
        elif action.delete:
            state = "deleted"
        else:
            state = "skipped"
        lines.append(f"  [{state}] {action.describe()}")
    if failed:
        lines.append(_failed_footer(failed, survived))
    return "\n".join(lines)


def _failed_footer(failed: Sequence[str], survived: Sequence[str]) -> str:
    """What the run left behind, said exactly — including when the answer is "a name".

    A failed build's checkpoint is deleted and the deletion *verified*
    (:func:`run_bake_actions`), because ``shuru checkpoint create`` saves the
    checkpoint before it reports the inner command's failure. That is the normal
    case, and it is the one that lets the summary claim nothing was left behind. When
    the delete does not take, the claim is not softened but replaced: the name is
    listed, and the reader is told that this one name *would* be accepted.
    """
    names = ", ".join(failed)
    if not survived:
        return (
            f"  {len(failed)} layer(s) failed: {names} — each was deleted again and "
            "its absence verified, so nothing was left behind for prepare to treat "
            "as valid"
        )
    return (
        f"  {len(failed)} layer(s) failed: {names}. These could not be removed and "
        f"still exist: {', '.join(survived)} — under those names a half-baked layer "
        "would be treated as valid by prepare (the marker is absent, but the name is "
        "not); delete them by hand before running anything else"
    )


_MIN_ERROR_LINE_CHARS = 4
"""Shorter than this, a line is not an error message.

Measured: a failed bake's message ended in ``failed (exit 1): i`` — the last thing
in the merged build output was a one-character fragment, so the tail of the log is
not the same thing as the error. The last line that has anything in it is.
"""

SHURU_LOG_PREFIX = "shuru:"
"""What every line of shuru's own CLI output starts with, and none of it is the error.

[VANTA-WHY] Measured, and it is the reason this constant exists: ``shuru checkpoint
create`` saves the checkpoint *before* reporting that the inner command failed, so the
last line of a failed layer's merged output is ``shuru: checkpoint 'vanta-bi-x' saved``
(:mod:`bugsinpy.shuru` says the same thing at the delete). Reporting that as "the last
meaningful line" named the layer that failed and nothing about why, on all three
failures of the nine-bug run. The inner command's own error is still in the log — it is
just above shuru's line, and this is what gets it read.
"""

ERROR_LINE_PREFIXES = ("ERROR", "error:", "fatal error")
"""What a line that *is* the failure starts with, in the three spellings measured.

A **prefix** test, and only that, because that is what a reader of the log can act on
without guessing which of two hundred lines matters: ``ERROR:`` is pip's own
(``ERROR: ResolutionImpossible``, ``ERROR: Cannot install …``), ``error:`` is a
compiler's (``error: command 'gcc' failed …``), and ``fatal error`` opens a C
compiler's fatal diagnostic (``fatal error: Python.h: No such file or directory``). A
line that starts with one of these is preferred over the last line that merely says
something (:func:`last_error_line`), because it is the one line of a two-hundred-line
build log that states the reason rather than the progress.

A compiler that prefixes its fatal diagnostic with the file it was reading
(``numpy/core/setup_common.py:9:10: fatal error: …``, which gcc does) is not matched by
this pass and falls through to the second one — which is a loss of precision and not of
information: the line is still printed, and it is still the line that says the most.
"""


def last_error_line(output: str) -> str:
    """The one line of a failed build that names the failure, or a placeholder.

    Two passes over the same list of lines, and the list is built once so both passes
    skip the same things:

    * shuru's own output (:data:`SHURU_LOG_PREFIX`) — never a failure, and, measured,
      the line that used to be reported for every failed layer;
    * blank lines, and ``set -x`` traces (``+ …``), which are the recipe rather than the
      result.

    The first pass takes the last line that starts with one of
    :data:`ERROR_LINE_PREFIXES`; the second, for a build that says nothing that way,
    takes the last line that is long enough to be a message rather than a fragment
    (:data:`_MIN_ERROR_LINE_CHARS`). ``\\r`` is treated as a line break because pip's
    progress output overwrites a line in place: without it the "line" is the whole
    carousel concatenated.
    """
    lines = [
        line
        for raw in output.replace("\r", "\n").splitlines()
        if (line := raw.strip())
        and not line.startswith(SHURU_LOG_PREFIX)
        and not line.startswith("+ ")
    ]
    for line in reversed(lines):
        if line.startswith(ERROR_LINE_PREFIXES):
            return line
    for line in reversed(lines):
        if len(line) >= _MIN_ERROR_LINE_CHARS:
            return line
    return "(no output)"


def bake_log_path(logs_dir: Path, name: str) -> Path:
    """Where one layer's full build log goes: ``<logs_dir>/<layer>.log``."""
    return logs_dir / f"{name}.log"


def clear_bake_logs(logs_dir: Path) -> int:
    """Empty the bake-logs directory, returning how many files it held.

    [VANTA-WHY] Emptied rather than accumulated: the summary prints this directory as
    *this* bake's logs, and a layer that is not in this run's plan must not be able to
    leave a file there that reads as though this run wrote it. The directory is the
    harness's own scratch output (``runs/bugsinpy/bake-logs``) and never a source of
    truth — the layer's checkpoint and its deps JSON are — so nothing is lost by
    replacing it per bake.
    """
    if not logs_dir.is_dir():
        return 0
    removed = 0
    for path in logs_dir.glob("*.log"):
        path.unlink()
        removed += 1
    return removed


def write_bake_log(logs_dir: Path, name: str, result: ExecResult) -> Path:
    """Write one layer's whole build output under its own name, and say where.

    The whole thing, not a tail: the one-line error in the summary is a pointer, and
    the reason a layer failed is usually nowhere near the end of its log — pip's
    resolution error is followed by shuru's own ``checkpoint … saved``, and a compiler's
    ``fatal error`` is followed by the recipe's remaining commands.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = bake_log_path(logs_dir, name)
    path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return path


def _with_gate_verdict(built: ExecResult, verdict: Optional[ExecResult]) -> ExecResult:
    """``built``'s own output, with the gate's verdict appended when the gate stopped it.

    [VANTA-WHY] The gate used to *replace* the result it judged, and the log is written
    from the result, so a refused layer's log held the verdict and nothing else — the
    install that had just succeeded, the pin list pip printed, and the ``pip check``
    output the verdict was built from were written nowhere. Measured on fastapi/1:
    a 253-byte log for a layer that had installed 98 pins, which is why the other side
    of its conflict (``exceptiongroup``, named by no pin list) could not be attributed
    after the fact. The one-line entry in the summary still comes from the verdict
    (:func:`_failure_report` reads what is passed to it), so nothing is reported twice.
    """
    if verdict is None:
        return built
    return ExecResult(
        stdout=built.stdout,
        stderr=built.stderr.rstrip("\n") + "\n" + verdict.stderr,
        exit_code=verdict.exit_code,
        timed_out=verdict.timed_out,
    )


def log_bake_line(line: str) -> None:
    """The default ``log`` for a bake: one line, flushed.

    [VANTA-WHY] A bake is usually piped into ``tee``, and a pipe makes ``print``
    block-buffered. The plan is printed with ``flush=True`` and then *nothing* — every
    ``building``/``build log``/``failed`` line waits in an 8 KB buffer until the
    process exits, and a run that is working looks hung. Measured: five minutes of
    baking, three failed layers, not one line after the plan.
    """
    print(line, flush=True)


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
    log: Callable[[str], None] = log_bake_line,
    logs_dir: Optional[Path] = None,
    own_pins: Optional[Mapping[str, Container[str]]] = None,
    record: Optional[Callable[[str, tuple[PinConflict, ...]], None]] = None,
    pins: Optional[Mapping[str, Sequence[tuple[str, str]]]] = None,
    provided: Optional[Mapping[str, Container[str]]] = None,
    dropped: Optional[Mapping[str, Container[str]]] = None,
) -> str:
    """Delete, then build every layer the plan asks for, and report what happened.

    Takes the plan :func:`plan_layers` produced rather than re-deriving it: what is
    logged before the run has to be the same list that is executed, or the log is
    describing a different bake.

    [VANTA-WHY] One failed *per-bug* layer does not stop the others. They are
    independent by construction — each is baked ``--from`` the interpreter layer and
    none is baked from another — so a bug that cannot resolve its dependencies says
    nothing about the next one, and stopping at the first turns a nine-bug run into
    nine runs to learn nine facts. Every failure is collected and reported together at
    the end. The interpreter layer is the exception and stops
    everything: it is the base of every layer stacked on it, so a layer that failed
    there would fail its children too, and each of those names would burn a bake
    cycle to learn the same thing.

    [VANTA-WHY] Fail *clean*: a layer that failed to build must not survive under a
    name ``prepare`` will accept. ``checkpoint_create`` deletes on a failed inner
    command, but a failed ``delete`` of its own would leave the name in place, so the
    absence is verified here rather than assumed — this is the one post-condition
    where "probably" is not good enough, because a surviving half-baked layer is
    invisible to every later stage. When the verification fails, that fact is carried
    into the summary by name (:func:`_failed_footer`) instead of the run claiming
    nothing was left behind.

    A built layer's own build-environment record and toolchain smoke result are
    re-printed here (:func:`layer_provenance_lines`), so the log of a run carries the
    layer's provenance rather than only the recipe that was supposed to produce it.
    There is nothing to print for a layer that was skipped: it was not built in this
    run, and what is being reported is this run.

    ``logs_dir`` is where each attempted layer's **whole** output is kept while the run
    happens (:func:`write_bake_log`), emptied first (:func:`clear_bake_logs`) and
    reported line by line through ``log``. It is what makes the one-line error in the
    failure report enough: the line says what happened, the file holds the two hundred
    lines above it. ``None`` writes nothing, which is what a structural test wants. A
    layer the gate refused keeps that same file, with the verdict appended to the
    output the install produced (:func:`_with_gate_verdict`) — the log of a refused
    layer is the only place its ``pip check`` reading and its install are recorded.

    ``own_pins`` and ``record`` are §3.2's own-pin gate. ``own_pins`` maps a layer's name
    to the canonical names it installs as **BugsInPy's own** pins
    (:func:`own_pin_names`); a layer the mapping does not name is not gated — the
    interpreter layer has no provenance to read its own pins from, and its install set is
    the harness's rather than the bug's. For a layer it does name, the layer's own
    ``pip check`` output (:func:`pip_check_readings`) is judged by §3.2's cases
    (:func:`pip_check_conflicts`): a case (b) conflict — both sides BugsInPy's own pins —
    is logged and tolerated, a case (a) one becomes the layer's failure through
    :func:`pip_check_failure`, and **every** conflict is handed to ``record`` so the deps
    JSON carries what pip actually said. This is the gate and the preflight is the
    prediction; a legacy-resolver install *succeeds* even when it is wrong, so nothing
    else in the run would catch it.

    [VANTA-WHY] Before either of those, **every** built layer is checked for a
    ``setuptools`` ``easy_install`` fetch (:func:`easy_install_fetches`) and fails
    through :func:`easy_install_failure` when one is found. It runs first and for every
    layer, own pins or not, because what it detects is not a conflict between pins but a
    package that entered the environment from outside the pin list: with
    ``--no-build-isolation`` an unsatisfied ``setup_requires``/``[build-system].requires``
    is resolved by setuptools itself, which reaches PyPI and installs *today's* release.
    A fetch is not tolerated the way a case (b) conflict is — the layer that results is
    not the one its provenance describes, and it would bake green. The early install pass
    (:func:`early_install_pins`, in the recipe) is how a layer whose own pins can satisfy
    such a declaration avoids the fetch; this is what says the measure worked.

    Injected ``exists``/``remove``/``build``/``record``/``log`` so the whole sequence is
    a structural test rather than something only a real bake can exercise.

    ``pins`` and ``provided`` are §3.2's freeze gate. ``pins`` maps a layer's name to its
    own pin list (:func:`layer_pins`) and ``provided`` to the names the interpreter layer
    supplies it (:func:`transitive_pins`, recorded as ``provisioned`` in the provenance —
    the orphans it removes are deliberately not among them); a layer the first of them
    does not name is not gated. What the gate reads is the layer's ``pip freeze``
    (:func:`unpinned_freeze`): every installed distribution has to be a pin of this layer
    or something the interpreter layer provides, and a name in neither is one pip resolved
    from the day's PyPI — or an orphan whose removal did not take, since a name the recipe
    uninstalls is in neither set on purpose (:data:`ORPHAN_RULE`) — and is failed through
    :func:`freeze_failure`. It runs after the own-pin gate because the two answer
    different questions — that one asks whether the pins can *hold* together, this one
    whether the pin list *describes* the layer — and it is the preflight's
    :func:`transitive_pins` that should make it pass: the walk pins what a release
    declared, so a name here is a declaration the walk could not read (an sdist with no
    metadata) rather than one it missed. ``dropped`` maps a layer to the canonical names
    §3.2 dropped for it, and it is passed to both gates: the own-pin gate tolerates the
    ``pip check`` reading that names one (:data:`CASE_DROPPED_DEPENDENCY`) and fails on
    every other missing requirement, and the freeze gate fails on a dropped name at any
    version whatever the pin list and provision say (:func:`unpinned_freeze`).
    """
    own = own_pins or {}
    pin_lists = pins or {}
    supplied = provided or {}
    dropped_names = dropped or {}
    logs: dict[str, Path] = {}
    if logs_dir is not None:
        clear_bake_logs(logs_dir)
        log(f"[logs] this bake's build logs: {logs_dir}")

    for action in delete_order(actions, children):
        log(f"[{action.name}] deleting — {action.reason}")
        remove(action.name)

    failed: list[str] = []
    results: dict[str, ExecResult] = {}
    attempted: list[str] = []
    survived: list[str] = []
    for action in build_order(actions):
        log(f"[{action.name}] building from {action.base} — {action.reason}")
        attempted.append(action.name)
        built = build(action.name, action.script, action.base)
        result = built
        verdict: Optional[ExecResult] = None
        if built.ok:
            # setuptools' easy_install, before the own-pin gate: a fetch is a failure
            # for *every* layer, not only a bug layer, and it says something the pin
            # gate cannot — the environment that was just built is not the one the
            # layer's own pins describe (§3.2).
            fetched = easy_install_fetches(built.stdout + built.stderr)
            if fetched:
                for line in fetched:
                    log(f"[{action.name}] easy_install: {line}")
                verdict = easy_install_failure(fetched)
                result = verdict
            elif action.name in own:
                conflicts = pip_check_conflicts(
                    pip_check_readings(built.stdout + built.stderr),
                    own=own[action.name],
                    dropped=dropped_names.get(action.name, ()),
                )
                for conflict in conflicts:
                    log(f"[{action.name}] {conflict.describe()}")
                if conflicts and record is not None:
                    record(action.name, conflicts)
                refused = tuple(
                    conflict
                    for conflict in conflicts
                    if conflict.case == CASE_HARNESS_CHOICE
                )
                if refused:
                    # The install succeeded and the marker is there; what failed is the
                    # gate after it. Reported through the same path a build failure
                    # takes, so a conflict names itself where the reader is already
                    # looking.
                    verdict = pip_check_failure(refused)
                    result = verdict
            if verdict is None and action.name in pin_lists:
                # The layer's own account of what it holds, after the install and not
                # from the pins: a legacy-resolver install succeeds whatever it is
                # given, so a name pip resolved itself is written nowhere else (§3.2).
                unpinned = unpinned_freeze(
                    built.stdout + built.stderr,
                    pins=pin_lists[action.name],
                    provided=supplied.get(action.name, ()),
                    dropped=dropped_names.get(action.name, ()),
                )
                if unpinned:
                    for line in unpinned:
                        log(f"[{action.name}] unpinned: {line}")
                    verdict = freeze_failure(unpinned)
                    result = verdict
        if logs_dir is not None:
            logs[action.name] = write_bake_log(
                logs_dir, action.name, _with_gate_verdict(built, verdict)
            )
            log(f"[{action.name}] build log: {logs[action.name]}")
        if not result.ok:
            failed.append(action.name)
            results[action.name] = result
            if exists(action.name):
                remove(action.name)
            if exists(action.name):
                survived.append(action.name)
            if action.base == DEFAULT_BASE:
                # An interpreter layer is the base of every layer stacked on it, so
                # nothing after this point is worth attempting. The layers that were
                # not reached are labelled ``not started`` in the summary.
                break
            continue
        log(f"[{action.name}] ok — marker {MARKER} verified")
        for line in layer_provenance_lines(result.stdout + result.stderr):
            log(f"[{action.name}] {line}")

    summary = summarise(
        actions, failed=failed, attempted=attempted, survived=survived
    )
    if failed:
        raise BakeError(_failure_report(failed, results, summary, logs))
    return summary


def _failure_report(
    failed: Sequence[str],
    results: Mapping[str, ExecResult],
    summary: str,
    logs: Optional[Mapping[str, Path]] = None,
) -> str:
    """Every failed layer's own error line, where its log is, then the summary.

    [VANTA-WHY] One line for **every** failed layer, and that line is the one line of
    the layer's output that names the failure (:func:`last_error_line`) rather than a
    tail: the tail of the last failure was all the nine-bug run printed, so the other
    two failures were represented by their layer names and nothing else — and what that
    tail ended in was shuru's ``checkpoint … saved``. Each layer's whole log is on disk
    (:func:`write_bake_log`) and pointed at here, which is what makes the one-line form
    enough.
    """
    lines = [f"{len(failed)} layer(s) failed to bake:"]
    for name in failed:
        result = results[name]
        state = (
            "timed out" if result.timed_out else f"exit {result.exit_code}"
        )
        lines.append(
            f"  {name}: {state} — {last_error_line(result.stdout + result.stderr)}"
        )
        path = (logs or {}).get(name)
        if path is not None:
            lines.append(f"    full build log: {path}")
    lines.append("")
    lines.append(summary)
    return "\n".join(lines)
