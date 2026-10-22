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

OPTIONAL_DECLARED_BUG_KEYS = frozenset({"import_package"})
"""Declared keys a bug may carry but need not.

[VANTA-WHY] ``import_package`` is the last-resort override for the import name the
import-path check loads, used only when the bug's own ``setup.py``/``pyproject.toml``
cannot name the package statically (design §3.2). It is optional because the normal
route is the metadata itself: requiring it of every bug would put a redundant guess
next to a fact for all of them. A bug that sets it records the reason in the split's
``reviewed_changes``, the same way every other departure from the first-authored
protocol is recorded.
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
        "hidden_files_deleted",
        "partially_hidden_functions",
        "gate_failures",
        "prompt_tokens_estimate",
        "context_ok",
        "context_reasons",
        "issue_text",
        "issue_text_sha256",
        "issue_ok",
        "issue_needs_decision",
        "deps",
    }
)

BUG_KEYS = DECLARED_BUG_KEYS | OPTIONAL_DECLARED_BUG_KEYS | DERIVED_BUG_KEYS

CONFIGS = ("baseline", "vanta")  # `vanta_planner` is addressable but not implemented (§4.2)
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
