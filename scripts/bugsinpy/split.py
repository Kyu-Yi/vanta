"""The split file — the experiment's committed protocol.

``plans/bugsinpy_split.json`` is written by hand once (the declared half) and then
filled in by ``prepare`` (the derived half). It is committed **before any run**
(design §7.2), so it is simultaneously the input to the batch runner and the frozen
record of what the experiment was: which bugs, which configs, which seeds, which
pytest pins, which hidden nodeids, which issue text.

[VANTA-WHY] Unknown keys are an error, not ignored (§12.2.5). The split is the
protocol; a typo like ``"seed"`` for ``"seeds"`` must not silently run the whole
pilot with a default. That is the same reason the derived half is validated on
read: a hand-edited hash or nodeid list has to be rejected by the reader, not
discovered mid-wave.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from . import arms as arms_mod

SCHEMA_VERSION = 1

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "bugsinpy_commit",
        "bugsinpy_root",
        "configs",
        "seeds",
        "review_mode",
        "sampling",
        "module_distributions",
        "reviewed_changes",
        "bugs",
    }
)

OPTIONAL_TOP_LEVEL_KEYS = frozenset(
    {"bugsinpy_root", "module_distributions", "reviewed_changes"}
)

MODULE_DISTRIBUTIONS_DOC = """``{import module: distribution}``, for the test-import closure.

[VANTA-WHY] Some modules are not importable under their distribution's name
(``import yaml`` is PyYAML; ``import bs4`` is beautifulsoup4), and no
canonicalisation recovers that from the module alone. The closure (§3.2) refuses to
guess: it stops the bake and names the module. This mapping is where the answer goes,
in the protocol, where a reviewer can see the decision — rather than a table inside
the harness, where an unreviewed guess would silently become part of every layer.
"""

REVIEWED_CHANGE_KEYS = frozenset({"change", "reason", "decided_by", "rule", "measurements"})
"""``reviewed_changes`` is the split's audit trail (§8): every departure from the
first-authored protocol — an excluded bug, a promoted reserve, a changed selection
rule — is recorded here with its reason, in the split itself rather than in a
commit message or a design-doc prose block.

[VANTA-WHY] It lives in the protocol because the protocol is what a reviewer reads
to know what the experiment was. A substitution decided in chat and applied to
``bugs`` would be indistinguishable from a bug that was always there; the same
substitution *with its measurements* next to it is checkable. The measurements are
per-candidate and machine-readable for the same reason: "black/1 rejects" is an
assertion, whereas ``est 68376 / budget 65280 / source_truncated_by_cap`` is the
evidence, and a reader can re-run it.

The shape stays closed (unknown keys are an error) so a hand-edited record cannot
quietly grow an unvalidated field.
"""

MEASUREMENT_KEYS = frozenset(
    {"bug", "estimated_prompt_tokens", "headroom_tokens", "verdict", "reasons"}
)
MEASUREMENT_VERDICTS = ("pass", "reject")

# Hand-authored: the plan for the bug, known before any checkout.
DECLARED_BUG_KEYS = frozenset(
    {
        "project",
        "bug_id",
        "python_version",
        "buggy_commit",
        "fixed_commit",
        "test_file",
        "pythonpath",
        "needs_build",
        "native_test_paths",
        "test_cmd",
        "pytest_version",
        "json_report_version",
        "checkpoint",
        "wave",
        "role",
        "notes",
    }
)

OPTIONAL_DECLARED_BUG_KEYS = frozenset(
    {
        "import_package",
        "sandbox_env",
        "build_env",
        "build_files",
        "vm_memory_mb",
        "precompile",
    }
)
"""Declared keys a bug may carry but need not.

[VANTA-WHY] ``import_package`` is the last-resort override for the import name the
import-path check loads, used only when the bug's own ``setup.py``/``pyproject.toml``
cannot name the package statically (design §3.2). It is optional because the normal
route is the metadata itself: requiring it of every bug would put a redundant guess
next to a fact for all of them. A bug that sets it records the reason in the split's
``reviewed_changes``, the same way every other departure from the first-authored
protocol is recorded.

[VANTA-WHY] ``sandbox_env`` is the project's own statement of the environment its
suite runs in, recorded rather than hard-coded: fastapi's ``scripts/test.sh`` exports
``PYTHONPATH`` over the directory its example apps live in and ~250 of its tests
import a module the tree ships there (``from additional_responses.tutorial001 import
app``), so without it the suites die at collection with ``ModuleNotFoundError`` and the
bug reads as a broken tree. It is per bug in this file because the file is a list of
bugs — the rule is the *project's*, and a bug of a project that declares one carries
it. The recorded value is the *directory*, and it is not frozen: for the projects whose
own runner states it (fastapi), ``prepare`` reads the ``PYTHONPATH`` out of that bug's
buggy commit, because the directory moved between commits — the newer ones ship
``docs_src``, the older ones ``docs/src`` — and one recorded value served only one side
of the move (§8.6 round 10). What the row records is what every run of the row gets.
The one alternative that would have kept the tree free of harness-made files was a
``conftest.py`` written into the repo, and it was rejected: the tree the Attacker and
the Creator read must be exactly what the benchmark ships.

[VANTA-WHY] ``build_env`` is the project's own build rule, for a tree whose ``setup.py``
compiles with ``-Werror`` against sources written for an older toolchain. pandas/1 is the
case: ``setup.py`` appends ``-Werror`` to every compile (removed only when numpy < 1.16,
and the layer's numpy is 1.18.4), and GCC 14 promotes ``tokenizer.c``'s negative-index
idiom to ``error: array subscript 309 is above array bounds of 'double[309]'``, so the
tree dies in ``build_ext`` and the gate reports a compiler diagnostic that is not the
bug. The demotion is that project's own idiom — the same ``setup.py`` appends
``-Wno-error=deprecated-declarations`` for Python 3.8 — and it is per project for the
same reason ``sandbox_env`` is: it is a fact about how the project builds on this image,
not about the bug. It is a *separate* key from ``sandbox_env`` because it reaches the
build and nothing else: ``sandbox_env`` is the environment the graded tests run in, which
is the measurement, whereas this one is read by ``prepare``'s in-place build (§3.4) and
by nothing that grades a patch. A bug that declares one records the reason in the split's
``reviewed_changes``, like every other departure from the first-authored protocol.

[VANTA-WHY] ``build_files`` is the other half of that rule, for a build that selects its
configuration through a file rather than through a flag. matplotlib/1 is the case: its
``setupext.py`` reads the path of its config from ``MPLSETUPCFG`` (defaulting to
``setup.cfg``, which the tree does not ship — only ``setup.cfg.template``), and
``FreeType.do_custom_build`` returns immediately when ``system_freetype`` is true, so the
one honest offline route is to *name* the system FreeType the layer already carries
instead of letting the build try to download 2.6.1 from two GNU URLs, which is an
``IOError`` in a VM with no network. The file is written into the batch directory — the
guest's ``/workspace`` — and never into a tree, so it cannot change what the graded tests
read; the tree stays exactly what the benchmark ships, for the reason ``sandbox_env``
records. It is a map because a build may read more than one file, and the values are the
contents verbatim. A bug that declares one records the reason in the split's
``reviewed_changes``.

[VANTA-WHY] ``vm_memory_mb`` is the VM's memory in megabytes, for a project whose
whole suite does not fit the default a run VM boots with. pandas/1 is the case: on the
default 2048 MB the graded run of its suite is SIGKILLed mid-run by the kernel — measured
as exit 137 after 245 s at about 61 % with no pytest report written at all — which the
harness now refuses as a killed run rather than reading the absent report as an empty
oracle (``prepare.UNMEASURED_RUN_REASON``). It is a separate key from ``sandbox_env``
because it changes the VM's *size* rather than the environment the graded tests run in,
and it is per project for the same reason the others are: it is a fact about the project's
suite, not about the bug. A larger VM lowers how many VMs fit in parallel on one host, so
the number is recorded on the row — where the M4 queue reads it — rather than assumed. A
bug that declares one records the reason in the split's ``reviewed_changes``.

[VANTA-WHY] ``precompile`` is the list of the project's own files that every suite run
must byte-compile **before pytest starts**, with the interpreter that is about to run
pytest. matplotlib/1 is the case: its pytest configuration escalates warnings to errors
(``filterwarnings = error``), and CPython 3.8 emits an ``invalid escape sequence``
warning while *compiling* ``lib/matplotlib/widgets.py`` (line 1924, inside the docstring
opened at 1911) — so the first import of that module during collection turns a warning
about a docstring into ``SyntaxError`` and the gate reads a broken environment. Compiling
the file first leaves a valid ``.pyc`` beside it, the module is then loaded from bytecode
and nothing is compiled under pytest's filters at all. A whole-tree scan of the fixed
commit found **one** such warning, so the rule is one file and is not a general "compile
the tree" switch.

[VANTA-WHY] It is a *step*, not a value written into a tree once. A one-time ``.pyc`` goes
stale the moment a Creator patch edits that file (the import machinery validates the
``.pyc`` against the source's mtime), and the ``SyntaxError`` comes straight back — so the
step runs inside every VM run of the row, after the patch that run is measuring and before
its pytest, exactly as :attr:`sandbox_env` and :attr:`vm_memory_mb` reach every run. It is
per project for the same reason those are: it is a fact about the project's test
configuration, not about the bug. A bug that declares one records the reason in the
split's ``reviewed_changes``.
"""

VM_MEMORY_MB_MIN = 2048
"""The size a bug's ``vm_memory_mb`` must *exceed*.

[VANTA-WHY] This is the size a run VM boots at anyway (``shuru.DEFAULT_RUN_MEMORY_MB``,
which is shuru's own default). A bug that records this or less is asking for a VM no
bigger than the one it already gets, and the value that carries information is always
*more* — so a number at or below this is a mistake rather than a rule, and it is refused
here rather than quietly accepted as a declaration that changes nothing.
"""

SANDBOX_ENV_KEYS = frozenset({"PYTHONPATH", "MPLBACKEND"})
"""The keys ``sandbox_env`` may set, and nothing else.

[VANTA-WHY] The allow-list *is* the rule. ``PYTHONPATH`` is what the projects declare
for themselves; a general "set any variable" channel through the split would be a way
to change the environment the graded tests run in, which is the measurement, without
it showing up as a decision about the bug. A bug that needs another key is a change to
this set and to the split's ``reviewed_changes``, not a quiet value in a bug entry.

[VANTA-WHY] ``MPLBACKEND`` is the second key, admitted the same way, for matplotlib/1.
matplotlib reads it into ``rcParams['backend']`` at import (``lib/matplotlib/__init__.py``:
``if os.environ.get('MPLBACKEND'): rcParams['backend'] = os.environ.get('MPLBACKEND')``),
and the backend ``pyplot`` resolves is the one that is current *when pyplot is first
imported*. With nothing set the rcParams default is ``TkAgg``, so that import walks
``backends/backend_tkagg`` -> ``backends/_backend_tk`` -> ``_tkinter.__file__``, which the
layer's CPython 3.8 does not have: collection dies with ``AttributeError: module
'_tkinter' has no attribute '__file__'`` and the bug has no measurable suite at all.
Setting it is the project's own answer rather than a harness workaround -- matplotlib's
own test conftest also selects ``agg`` (``matplotlib.use('agg', force=True)``), only too
late, because ``use()`` imports pyplot on the way to ``switch_backend`` and pyplot has
already resolved the default -- and the value is the non-interactive ``Agg`` because the
suite's tests that do want another backend name it themselves in the child environment
they spawn (``test_backends_interactive.py``, ``test_backend_webagg.py``) or blank it
deliberately (``test_rcparams.py``), so a value in the parent cannot take a window away
from a test that asks for one.
"""

BUILD_ENV_KEYS = frozenset({"CFLAGS", "MPLSETUPCFG"})
"""The variables ``build_env`` may add to, and nothing else.

[VANTA-WHY] The allow-list *is* the rule, for the reason :data:`SANDBOX_ENV_KEYS`
records: a general channel through the split would be a way to change how the tree is
compiled — ``CC``, ``LDSHARED``, ``PYTHONPATH`` — without it showing up as a decision
about the bug. ``CFLAGS`` is the variable to name because the layer already sets it
(``bake.COMPILER_ENV``: the era's warning demotions and the stdint include), so a rule
that writes it *adds* its token to a value that is already there rather than putting it
in that value's place, and because it is the one the build's own configuration reads for
both the compile and the link line. A build's own ``-Werror`` lands after it on the
command line; GCC documents that a more specific ``-Wno-error=<name>`` has priority over
a bare ``-Werror`` independently of their position.

``MPLSETUPCFG`` is the second shape: it names the path of the file a build reads its own
configuration from — the other half of ``build_files``, which writes that file into the
batch directory. The layer never sets it (``bake.COMPILER_ENV`` names ``CC``, ``CXX``,
``LDSHARED``, ``LDCXXSHARED``, ``CFLAGS``, ``FFLAGS`` and nothing else), so the value is
set outright rather than appended to one that is there, which is what "added to the
layer's value" means for a variable the layer does not own. A bug that needs another
variable is a change to this set and to the split's ``reviewed_changes``, not a quiet
value in a bug entry.
"""

# Written back by ``prepare``. Nothing here may be hand-edited into the file
# without ``prepare`` re-deriving it, which is why the reader checks the hash.
DERIVED_BUG_KEYS = frozenset(
    {
        "prepare_mode",
        "bug_patch_sha256",
        "test_paths_from_fixed",
        "fail_to_pass",
        "f2p_undeclared",
        "f2p_undeclared_reason",
        "pass_to_pass_count",
        "flaky_nodeids",
        "preexisting_failures",
        "native_failure_baseline",
        "native_green_on_vanta_tree",
        "native_skips_buggy",
        "native_skips_fixed",
        "native_secs",
        "hidden_files_deleted",
        "partially_hidden_functions",
        "hidden_by_collect_flip",
        "collect_flip_nodeids",
        "collect_excluded",
        "gate_failures",
        "prompt_tokens_estimate",
        "context_ok",
        "context_reasons",
        "issue_text",
        "issue_text_sha256",
        "issue_text_commit",
        "issue_text_names_paths",
        "issue_text_names_symbols",
        "issue_ok",
        "issue_needs_decision",
        "deps",
        "builds",
        "version_pin",
    }
)

BUG_KEYS = DECLARED_BUG_KEYS | OPTIONAL_DECLARED_BUG_KEYS | DERIVED_BUG_KEYS

CONFIGS = arms_mod.CONFIGS
"""What a split's ``configs`` list may write — every arm spelling (§4.2).

[VANTA-WHY] M3 item 8 follow-up — read from ``arms.py`` rather than spelled out here,
so a new arm is added in one place. The list mixes the legacy names (``baseline``,
``vanta``) with the canonical arm names (``b0``, ``b1``, ``b2``, ``b3``, ``suite``);
``runner._require_arm_list`` refuses two spellings that normalise to one config, and
``vanta_planner`` stays absent on purpose: it is §4.4's placeholder, whose runs return
``InfraFailure``, so naming it is refused here rather than run.
"""
ROLES = ("pilot", "reserve")

SAMPLING_POLICY = "pinned_per_call_plus_server_top_p"
"""The split's declaration of where a run's sampling comes from (§4.3).

[VANTA-WHY] This replaced ``server_defaults``, which encoded a retracted premise:
that the harness lets the server decide everything. The pipeline has always pinned
``temperature`` / ``max_tokens`` **per call site** (creator 0.1, attacker_generate
0.2, attacker_revise/simplify 0.3, intent_audit 0.0), and only ``top_p`` is left to
the server. A policy string that says "server defaults" would have made the split
disagree with the run it describes, and the validator would then have rejected the
truthful file.
"""


class SplitError(RuntimeError):
    """The split file is missing, malformed, or contradicts itself."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class Split:
    """A loaded split file. ``bugs`` are plain dicts, validated on construction."""

    def __init__(self, data: dict[str, Any], path: Optional[Path] = None) -> None:
        self.path = path
        self.data = data

    # ---- accessors -------------------------------------------------------
    @property
    def bugsinpy_commit(self) -> str:
        return self.data["bugsinpy_commit"]

    @property
    def bugsinpy_root(self) -> str:
        return self.data.get("bugsinpy_root", "runs/bugsinpy/bugsinpy")

    @property
    def configs(self) -> list[str]:
        return list(self.data["configs"])

    @property
    def seeds(self) -> list[int]:
        return list(self.data["seeds"])

    @property
    def review_mode(self) -> bool:
        return bool(self.data.get("review_mode", False))

    @property
    def sampling(self) -> dict[str, Any]:
        return dict(self.data.get("sampling") or {})

    def module_distributions(self) -> dict[str, str]:
        """Reviewed ``{import module: distribution}`` overrides for the closure (§3.2).

        Empty when the key is absent, which is the normal case: nearly every module is
        importable under its distribution's canonical name.
        """
        return {
            str(module): str(distribution)
            for module, distribution in (self.data.get("module_distributions") or {}).items()
        }

    @property
    def bugs(self) -> list[dict[str, Any]]:
        return self.data["bugs"]

    def bug(self, project: str, bug_id: int) -> dict[str, Any]:
        for entry in self.bugs:
            if entry["project"] == project and entry["bug_id"] == bug_id:
                return entry
        raise SplitError(f"{project}/{bug_id} is not in the split")

    def bugs_in_wave(self, wave: int, role: str = "pilot") -> list[dict[str, Any]]:
        """Pilot bugs of one wave, in file order (the committed order, §7.1)."""
        return [
            entry
            for entry in self.bugs
            if entry.get("wave") == wave and entry.get("role", "pilot") == role
        ]

    def waves(self, role: str = "pilot") -> list[int]:
        return sorted({int(entry["wave"]) for entry in self.bugs if entry.get("role", "pilot") == role})

    # ---- mutation + persistence -----------------------------------------
    def set_derived(self, project: str, bug_id: int, **fields: Any) -> None:
        unknown = set(fields) - DERIVED_BUG_KEYS
        if unknown:
            raise SplitError(f"not derived fields: {sorted(unknown)}")
        self.bug(project, bug_id).update(fields)

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=False, ensure_ascii=False) + "\n"

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or self.path
        if target is None:
            raise SplitError("no path to save the split to")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target


def load_split(path: Path) -> Split:
    if not path.is_file():
        raise SplitError(f"no split file at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SplitError(f"{path}: top level must be an object")
    validate(data, path)
    return Split(data, path)


def validate(data: dict[str, Any], path: Optional[Path] = None) -> None:
    """Strict structural check. Raises ``SplitError`` on anything unexpected."""
    where = f"{path}: " if path else ""

    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise SplitError(f"{where}unknown top-level keys: {sorted(unknown)}")
    missing = TOP_LEVEL_KEYS - OPTIONAL_TOP_LEVEL_KEYS - set(data)
    if missing:
        raise SplitError(f"{where}missing top-level keys: {sorted(missing)}")

    if data["schema_version"] != SCHEMA_VERSION:
        raise SplitError(
            f"{where}schema_version {data['schema_version']!r}, expected {SCHEMA_VERSION}"
        )

    configs = data["configs"]
    if not isinstance(configs, list) or not configs:
        raise SplitError(f"{where}configs must be a non-empty list")
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        raise SplitError(f"{where}unknown configs {bad}; known: {list(CONFIGS)}")

    seeds = data["seeds"]
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise SplitError(f"{where}seeds must be a non-empty list of ints")

    sampling = data.get("sampling") or {}
    if sampling.get("policy") != SAMPLING_POLICY:
        raise SplitError(
            f"{where}sampling.policy must be {SAMPLING_POLICY!r}: the pipeline pins "
            "temperature/max_tokens per call site and top_p is read from the server's "
            "/props, so the harness adds no run-level value (design §4.3)"
        )

    bugs = data["bugs"]
    if not isinstance(bugs, list) or not bugs:
        raise SplitError(f"{where}bugs must be a non-empty list")

    seen: set[str] = set()
    for index, entry in enumerate(bugs):
        _validate_bug(entry, where, index, seen)

    reviewed = data.get("reviewed_changes")
    if reviewed is not None:
        if not isinstance(reviewed, list) or not reviewed:
            raise SplitError(f"{where}reviewed_changes must be a non-empty list when present")
        for index, entry in enumerate(reviewed):
            _validate_reviewed_change(entry, where, index)

    distributions = data.get("module_distributions")
    if distributions is not None:
        _validate_module_distributions(distributions, where)


def _validate_module_distributions(distributions: Any, where: str) -> None:
    tag = f"{where}module_distributions"
    if not isinstance(distributions, dict) or not distributions:
        raise SplitError(f"{tag} must be a non-empty object when present")
    for module, distribution in distributions.items():
        if not isinstance(module, str) or not module.strip():
            raise SplitError(f"{tag}: keys must be non-empty module names, got {module!r}")
        if not isinstance(distribution, str) or not distribution.strip():
            raise SplitError(
                f"{tag}[{module!r}] must be a non-empty distribution name, got "
                f"{distribution!r}"
            )


def _validate_reviewed_change(entry: Any, where: str, index: int) -> None:
    tag = f"{where}reviewed_changes[{index}]"
    if not isinstance(entry, dict):
        raise SplitError(f"{tag}: must be an object")

    unknown = set(entry) - REVIEWED_CHANGE_KEYS
    if unknown:
        raise SplitError(f"{tag}: unknown keys {sorted(unknown)}")

    for key in ("change", "reason"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SplitError(f"{tag}: {key} must be a non-empty string")

    for key in ("decided_by", "rule"):
        if key in entry and (not isinstance(entry[key], str) or not entry[key].strip()):
            raise SplitError(f"{tag}: {key} must be a non-empty string when present")

    measurements = entry.get("measurements")
    if measurements is None:
        return
    if not isinstance(measurements, list) or not measurements:
        raise SplitError(f"{tag}: measurements must be a non-empty list when present")
    for m_index, measurement in enumerate(measurements):
        _validate_measurement(measurement, tag, m_index)


def _validate_measurement(entry: Any, tag: str, index: int) -> None:
    where = f"{tag}.measurements[{index}]"
    if not isinstance(entry, dict):
        raise SplitError(f"{where}: must be an object")

    unknown = set(entry) - MEASUREMENT_KEYS
    if unknown:
        raise SplitError(f"{where}: unknown keys {sorted(unknown)}")

    bug = entry.get("bug")
    if not isinstance(bug, str) or not bug.strip():
        raise SplitError(f"{where}: bug must be a non-empty string")

    if entry.get("verdict") not in MEASUREMENT_VERDICTS:
        raise SplitError(f"{where}: verdict must be one of {list(MEASUREMENT_VERDICTS)}")

    for key in ("estimated_prompt_tokens", "headroom_tokens"):
        if not isinstance(entry.get(key), int):
            raise SplitError(f"{where}: {key} must be an int")

    reasons = entry.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        raise SplitError(f"{where}: reasons must be a list of strings")
    if entry["verdict"] == "reject" and not reasons:
        raise SplitError(
            f"{where}: a reject must carry its reasons — the precheck's verdict is only "
            "auditable if the record says which rules fired"
        )


def _validate_bug(entry: Any, where: str, index: int, seen: set[str]) -> None:
    tag = f"{where}bugs[{index}]"
    if not isinstance(entry, dict):
        raise SplitError(f"{tag}: must be an object")

    unknown = set(entry) - BUG_KEYS
    if unknown:
        raise SplitError(f"{tag}: unknown keys {sorted(unknown)}")
    missing = DECLARED_BUG_KEYS - set(entry)
    if missing:
        raise SplitError(f"{tag}: missing declared keys {sorted(missing)}")

    key = f"{entry['project']}/{entry['bug_id']}"
    tag = f"{where}{key}"
    if key in seen:
        raise SplitError(f"{tag}: duplicate bug entry")
    seen.add(key)

    if entry["role"] not in ROLES:
        raise SplitError(f"{tag}: role must be one of {list(ROLES)}")
    if not isinstance(entry["wave"], int) or entry["wave"] < 1:
        raise SplitError(f"{tag}: wave must be a positive int")

    cmd = entry["test_cmd"]
    if not isinstance(cmd, list) or not cmd:
        raise SplitError(f"{tag}: test_cmd must be a non-empty list")
    if not any("{report}" in str(part) for part in cmd):
        raise SplitError(
            f"{tag}: test_cmd must contain a '{{report}}' placeholder, otherwise the "
            "suite runner cannot read pytest's JSON report back"
        )

    paths = entry["native_test_paths"]
    if not isinstance(paths, list) or not paths:
        raise SplitError(f"{tag}: native_test_paths must be a non-empty list")

    if not entry["pytest_version"] or not entry["json_report_version"]:
        raise SplitError(
            f"{tag}: pytest_version and json_report_version are pinned per Python "
            "version and must be recorded (design §3.2)"
        )

    override = entry.get("import_package")
    if override is not None and (not isinstance(override, str) or not override.strip()):
        raise SplitError(
            f"{tag}: import_package must be a non-empty string — it is the module the "
            "import-path check loads, and an empty one would be a guess"
        )

    memory = entry.get("vm_memory_mb")
    if memory is not None:
        if isinstance(memory, bool) or not isinstance(memory, int):
            raise SplitError(
                f"{tag}: vm_memory_mb must be an integer number of megabytes — it is "
                "passed straight to shuru's --memory, and a value that is not a number "
                "cannot size a VM"
            )
        if memory <= VM_MEMORY_MB_MIN:
            raise SplitError(
                f"{tag}: vm_memory_mb {memory} does not exceed the {VM_MEMORY_MB_MIN} MB "
                "a run VM boots at anyway — a recorded size has to be larger than the "
                "default, or it declares something that is already true"
            )

    env = entry.get("sandbox_env")
    if env is not None:
        if not isinstance(env, dict) or not env:
            raise SplitError(
                f"{tag}: sandbox_env must be a non-empty object of KEY=VALUE strings — "
                "the project's own suite environment, or nothing at all; an empty one "
                "would record a rule that says nothing"
            )
        for key, value in env.items():
            if key not in SANDBOX_ENV_KEYS:
                raise SplitError(
                    f"{tag}: sandbox_env key {key!r} is not allowed — the only keys a "
                    f"bug may set are {sorted(SANDBOX_ENV_KEYS)}, because a general "
                    "environment channel would change what the graded tests run in"
                )
            if not isinstance(value, str) or not value.strip():
                raise SplitError(
                    f"{tag}: sandbox_env[{key!r}] must be a non-empty string — an empty "
                    "value is not the environment the project declares its suite runs in"
                )

    build = entry.get("build_env")
    if build is not None:
        if not isinstance(build, dict) or not build:
            raise SplitError(
                f"{tag}: build_env must be a non-empty object of NAME=VALUE strings — "
                "the project's own build rule, or nothing at all; an empty one would "
                "record a rule that says nothing"
            )
        if not entry["needs_build"]:
            raise SplitError(
                f"{tag}: build_env is declared but needs_build is false — the rule is "
                "read by the in-place build (§3.4), which a bug that needs no build "
                "never reaches, so this entry would be a rule that can never apply"
            )
        for name, value in build.items():
            if name not in BUILD_ENV_KEYS:
                raise SplitError(
                    f"{tag}: build_env key {name!r} is not allowed — the variables a "
                    f"bug may add to are {sorted(BUILD_ENV_KEYS)}, because a general "
                    "channel would be a way to change how the tree is compiled without "
                    "it showing up as a decision about the bug"
                )
            if not isinstance(value, str) or not value.strip():
                raise SplitError(
                    f"{tag}: build_env[{name!r}] must be a non-empty string — the value "
                    "is added to whatever the layer already sets for that variable, and "
                    "an empty one declares nothing"
                )

    files = entry.get("build_files")
    if files is not None:
        if not isinstance(files, dict) or not files:
            raise SplitError(
                f"{tag}: build_files must be a non-empty object of NAME=CONTENT strings "
                "— the build's own configuration file, or nothing at all; an empty one "
                "would record a file that is never written"
            )
        if not entry["needs_build"]:
            raise SplitError(
                f"{tag}: build_files is declared but needs_build is false — the file is "
                "written just before the in-place build (§3.4), which a bug that needs "
                "no build never reaches, so this entry would be a rule that can never "
                "apply"
            )
        for name, content in files.items():
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise SplitError(
                    f"{tag}: build_files name {name!r} must be a bare file name — the "
                    "file is written into the batch directory, which the guest sees as "
                    "/workspace, so a name carrying a path separator or a directory "
                    "step could put it somewhere other than beside the trees, and the "
                    "whole reason this rule is honest is that it cannot reach a tree"
                )
            if not isinstance(content, str) or not content.strip():
                raise SplitError(
                    f"{tag}: build_files[{name!r}] must be a non-empty string — the "
                    "contents are written verbatim, and an empty file declares nothing"
                )

    precompile = entry.get("precompile")
    if precompile is not None:
        if not isinstance(precompile, list) or not precompile:
            raise SplitError(
                f"{tag}: precompile must be a non-empty list of tree-relative file "
                "paths — the files an early compile has to leave bytecode for, or "
                "nothing at all; an empty list would record a step that does nothing"
            )
        for path in precompile:
            if (
                not isinstance(path, str)
                or not path.strip()
                or path.startswith("/")
                or "\\" in path
                or ".." in path.split("/")
                or not path.endswith(".py")
            ):
                raise SplitError(
                    f"{tag}: precompile entry {path!r} must be a tree-relative .py path "
                    "with no directory step — it is compiled from the tree root inside "
                    "the VM, in every run of this bug, so a path that could name a file "
                    "outside the tree (or a path the compiler cannot parse) would either "
                    "compile something the benchmark does not ship or fail silently"
                )

    # Derived fields, when present, must be self-consistent.
    issue = entry.get("issue_text")
    if issue is not None:
        digest = entry.get("issue_text_sha256")
        if not digest:
            raise SplitError(f"{tag}: issue_text without issue_text_sha256")
        if digest != sha256_text(issue):
            raise SplitError(f"{tag}: issue_text_sha256 does not match issue_text")


def pilot_bugs(split: Split) -> list[dict[str, Any]]:
    return [entry for entry in split.bugs if entry.get("role", "pilot") == "pilot"]


def groups_by_wave(bugs: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for entry in bugs:
        out.setdefault(int(entry["wave"]), []).append(entry)
    return dict(sorted(out.items()))
