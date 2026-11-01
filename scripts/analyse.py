#!/usr/bin/env python3
"""Item 12 — one command that writes every number the paper quotes (§5).

    python3 scripts/analyse.py --batches 'runs/bugsinpy/luna-*' --out paper/generated/

[VANTA-WHY] The numbers in the paper must come from the batches' own records, not
from a notebook: a notebook's count cannot be re-derived from the artefacts, and
"why is this 0.61 and not 0.60" then has no answer. This script reads three things
per batch and nothing else —

    <batch>/manifest.jsonl            the rows (verdict, candidate, oracle, arm)
    <batch>/replay/<row-stem>.json    item 10's X[test][k] matrix
    <session_dir>/../<sid>_session.jsonl   the run's own summary line, when present

— and it writes files into ``--out``. It NEVER executes code, never calls a model
and never writes back into a batch.

Two rules shape the layout.

*Every metric is a plug-in over the same matrix* (M3.5 §6.6 fold-in 10). A metric is
a :class:`Metric` with a ``name`` and a ``run`` that takes the loaded
:class:`Analysis` and returns :class:`MetricResult` (files plus summary lines).
:data:`METRICS` is the registry; adding a number means adding an entry, never
editing a renderer. A metric that would need data no batch holds returns
``unavailable("<why>")`` rather than a zero — a zero is a measurement, and this
script must never report one it did not make.

*EIR/ECR are recomputed from the raw cells*, through item 10's own pure functions
(:func:`replay.transitions`, :func:`replay.rates`, :func:`replay.cells_for_view`),
never read back from the stored ``views`` aggregate. One implementation of the
counting rules is the point; a reader that trusted a second one could disagree with
the matrix it came from.

Determinism (§5 "regenerating twice gives identical files") is a hard requirement:
every dict is iterated in sorted order, every float is formatted to a fixed number
of places, and nothing here reads the clock, the environment or the machine name.
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import arms as arms_mod  # noqa: E402
from bugsinpy import replay as replay_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.jsonl"
DEFAULT_OUT = os.path.join("paper", "generated")
DEFAULT_SEED = replay_mod.DEFAULT_SEED
DEFAULT_BOOTSTRAP = 2000

# §5.3's label -> the five-way confusion cell the review table counts. A label
# outside this table is not a review decision (an infra failure, a censor) and is
# counted apart, never inside a TA/FA/TR/FR/NR rate.
CELL_TA = "TA"
CELL_FA = "FA"
CELL_TR = "TR"
CELL_FR = "FR"
CELL_NR = "NR"
CELL_ORDER = (CELL_TA, CELL_FA, CELL_TR, CELL_FR, CELL_NR)

LABEL_TO_CELL: dict[str, str] = {
    "true_approval": CELL_TA,
    "false_approval": CELL_FA,
    "true_reject": CELL_TR,
    "false_reject": CELL_FR,
    "needs_review_on_good": CELL_NR,
    "needs_review_on_bad": CELL_NR,
}

# "Was this row's verdict RIGHT?" — the paired test's per-row outcome. A refusal
# is neither: it is dropped from every pairing rather than scored as wrong, so a
# pair's McNemar table counts only the rows both arms decided.
LABEL_CORRECT: dict[str, bool] = {
    "true_approval": True,
    "true_reject": True,
    "false_approval": False,
    "false_reject": False,
}

# Candidate kind -> the paper's word for it (M3 plan item 12). `mutant` is cut by
# operator, because "a mutant was caught" and "an operator's mutants are caught"
# are different claims.
KIND_NAMES: dict[str, str] = {
    "dev_fix": "reference",
    "noop": "no-op",
    "failed_patch": "failed_patch",
}

# Item 12's paired comparisons (§5): B3 against each arm it is meant to beat, and
# the B3-on/B3-off ablation. `b1` is `b3` with the Attacker switched off (arms.py),
# which is what makes it the off-arm of that pair.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("b3", "judge", "B3 vs judge"),
    ("b3", "judge+suite", "B3 vs judge+suite"),
    ("b3", "b2", "B3 vs B2"),
    ("b3", "b1", "B3-on vs B3-off"),
)


class AnalysisError(RuntimeError):
    """The analysis cannot run at all (no batches, a malformed one)."""


# ---------------------------------------------------------------------------
# small numeric helpers — pure, so the tests can pin them
# ---------------------------------------------------------------------------
def unavailable(why: str) -> str:
    """The one token for a number this script cannot honestly produce.

    [VANTA-WHY] Never ``0``. A zero is a measurement — "we looked and found none" —
    and printing one where the input does not exist states a fact about the data
    that the data cannot support.
    """
    return f"unavailable({why})"


def label_for(record: Mapping[str, Any]) -> str:
    """One manifest row's §5.3 label, or ``""`` when the row carries no oracle."""
    payload = record.get("oracle")
    if not isinstance(payload, dict):
        return ""
    return runner_mod.label_for(
        str(record.get("verdict") or ""), runner_mod.oracle_from_dict(payload)
    )


def label_rule_for(record: Mapping[str, Any]) -> str:
    """Which rule read one manifest row's label, or ``""`` when it carries no oracle.

    [VANTA-WHY] M3 item B6 — old rows are never rewritten, so the rows already on disk
    were labelled by item Q's rule and only rows written after B6 carry a recorded apply
    class. The row states which rule read it (``label_rule``); a row that predates the
    field is read by the oracle's own shape, which is exactly item Q's set. Reporting
    the count per rule is what makes "no Phase 3 number depends on the legacy reading"
    checkable.
    """
    recorded = record.get("label_rule")
    if isinstance(recorded, str) and recorded:
        return recorded
    payload = record.get("oracle")
    if not isinstance(payload, dict):
        return ""
    return runner_mod.label_rule_for(runner_mod.oracle_from_dict(payload))


def label_rule_counts(analysis: "Analysis") -> dict[str, int]:
    """How many rows each label rule read — the report item B6 asks for.

    [VANTA-WHY] Item B6 — rows already on disk were labelled by item Q's rule and are
    never rewritten, so the only honest way to show "no Phase 3 number depends on the
    legacy reading" is to count the rules apart. A load with any ``legacy_item_q`` row
    is visibly a mixed load rather than a silently pooled one.
    """
    counts: dict[str, int] = {}
    for batch in analysis.batches:
        for record in batch.records:
            rule = label_rule_for(record)
            if rule:
                counts[rule] = counts.get(rule, 0) + 1
    return counts


def row_oracle(record: Mapping[str, Any]) -> Optional[Any]:
    payload = record.get("oracle")
    if not isinstance(payload, dict):
        return None
    return runner_mod.oracle_from_dict(payload)


def candidate_kind(record: Mapping[str, Any]) -> Optional[str]:
    """The paper's name for this row's candidate, or ``None`` for a row without one."""
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        return None
    kind = str(candidate.get("kind") or "")
    if kind == "mutant":
        operator = str(candidate.get("operator") or "?")
        return f"mutant:{operator}"
    return KIND_NAMES.get(kind, kind or "?")


def candidate_id(record: Mapping[str, Any]) -> Optional[str]:
    """The identity a pairing joins on: the candidate's own patch hash (§5)."""
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        return None
    raw = candidate.get("patch_sha256") or candidate.get("normalized_sha256")
    return str(raw) if raw else None


def approval_rate(counts: Mapping[str, int]) -> Optional[float]:
    """``(TA + FA) / (TA + FA + TR + FR)`` — refusals are outside the denominator."""
    decided = counts.get(CELL_TA, 0) + counts.get(CELL_FA, 0) + counts.get(CELL_TR, 0) + counts.get(CELL_FR, 0)
    if decided == 0:
        return None
    return (counts.get(CELL_TA, 0) + counts.get(CELL_FA, 0)) / decided


def rejection_rate(counts: Mapping[str, int]) -> Optional[float]:
    decided = counts.get(CELL_TA, 0) + counts.get(CELL_FA, 0) + counts.get(CELL_TR, 0) + counts.get(CELL_FR, 0)
    if decided == 0:
        return None
    return (counts.get(CELL_TR, 0) + counts.get(CELL_FR, 0)) / decided


def balanced_accuracy(counts: Mapping[str, int]) -> Optional[float]:
    """§5's balanced accuracy: the mean of the two per-class recalls.

    Sensitivity is "of the bad candidates, how many were rejected"; specificity is
    "of the good ones, how many were approved". Either denominator missing makes
    the whole quantity ``None`` — half of a balanced accuracy is not a balanced
    accuracy, and printing the remaining half would score an arm on one class only.
    """
    bad = counts.get(CELL_TR, 0) + counts.get(CELL_FA, 0)
    good = counts.get(CELL_TA, 0) + counts.get(CELL_FR, 0)
    if bad == 0 or good == 0:
        return None
    return 0.5 * (counts.get(CELL_TR, 0) / bad + counts.get(CELL_TA, 0) / good)


def mcnemar_exact(b: int, c: int) -> float:
    """The exact two-sided McNemar p-value for ``b``/``c`` discordant pairs.

    Under H0 each discordant pair is a fair coin, so the statistic is binomial;
    the exact two-sided p doubles the smaller tail and is capped at 1. Zero
    discordant pairs is ``1.0`` (no evidence), never ``0.0`` (proof of equality).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _percentile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile of a sorted list, by nearest rank (deterministic)."""
    ordered = sorted(values)
    index = int(round(q * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def cluster_bootstrap(
    by_cluster: Mapping[str, Sequence[Any]],
    statistic: Callable[[list[Any]], Optional[float]],
    *,
    seed: int,
    resamples: int,
) -> Optional[tuple[float, float]]:
    """A 95% interval for ``statistic``, resampling CLUSTERS (bugs) with replacement.

    [VANTA-WHY] The unit is the bug, not the row: one bug contributes many rows
    that share its difficulty, its suite and its oracle, so resampling rows would
    treat correlated observations as independent and print an interval far too
    narrow. The draw is seeded, so the interval is part of the reproducible output.

    Fewer than two clusters is ``None``: one cluster resampled with replacement is
    the same cluster every time, and the zero-width interval that would come out of
    it claims a precision the resampling never measured.
    """
    clusters = sorted(by_cluster)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        pooled: list[Any] = []
        for _ in range(len(clusters)):
            pooled.extend(by_cluster[clusters[rng.randrange(len(clusters))]])
        value = statistic(pooled)
        if value is not None:
            samples.append(float(value))
    if not samples:
        return None
    return (_percentile(samples, 0.025), _percentile(samples, 0.975))


def _fmt(value: Optional[float], places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _money(value: float) -> str:
    """A priced figure, to four places — the resolution Luna's rates need.

    Item 15's acceptance is a 5% match against the dashboard, and a single Luna
    row costs a few thousandths of a dollar, so three places would round most
    rows to ``$0.009`` and hide the difference. Never used for a hole: a number
    no row supports is rendered ``unavailable(...)``, not ``$0.0000``.
    """
    return f"${value:.4f}"


def _interval_text(interval: Optional[tuple[float, float]]) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _tex(text: str) -> str:
    """Escape the characters that would otherwise end or corrupt a LaTeX cell."""
    out = str(text)
    for needle, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        out = out.replace(needle, replacement)
    return out


# ---------------------------------------------------------------------------
# loading — the three inputs, and nothing else
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """One directory-matched batch: its rows and its replay records."""

    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    replays: list[dict[str, Any]] = field(default_factory=list)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a batch manifest, dropping a torn last line (§7.3, same rule as metrics).

    The manifest is append-only, so a resumed job appears twice; the LAST
    occurrence is the current one, which is what ``Manifest.find`` returns and
    therefore what a report must read.

    [VANTA-WHY] Item E6 + Q3 — a re-run of a job whose previous grade was
    INCONCLUSIVE appends a new row that names the session it supersedes. That
    superseded session is dropped from the reading as well, so a superseded
    inconclusive row can never be counted even if its own key were the latest.
    """
    latest: dict[str, dict[str, Any]] = {}
    superseded: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            supersedes = record.get("supersedes")
            if isinstance(supersedes, dict) and supersedes.get("session_id"):
                superseded.add(str(supersedes["session_id"]))
            key = str(record.get("key") or "")
            if key:
                latest[key] = record
            else:
                latest[f"#{len(latest)}"] = record
    rows = [latest[key] for key in sorted(latest)]
    return [row for row in rows if str(row.get("session_id") or "") not in superseded]


def load_replays(batch_dir: Path) -> list[dict[str, Any]]:
    """Every ``<batch>/replay/*.json`` record, in filename order."""
    out: list[dict[str, Any]] = []
    directory = batch_dir / replay_mod.REPLAY_DIRNAME
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def discover_batches(patterns: Sequence[str]) -> list[Path]:
    """Expand ``--batches`` globs to the batch directories that hold a manifest."""
    found: set[Path] = set()
    for pattern in patterns:
        for match in glob_mod.glob(pattern):
            path = Path(match)
            if path.is_file() and path.name == MANIFEST_NAME:
                path = path.parent
            if (path / MANIFEST_NAME).is_file():
                found.add(path.resolve())
    return sorted(found)


@dataclass
class Analysis:
    """The loaded inputs every metric reads — the one object the plug-ins share."""

    batches: list[Batch]
    seed: int = DEFAULT_SEED
    resamples: int = DEFAULT_BOOTSTRAP

    @property
    def rows(self) -> list[tuple[str, dict[str, Any]]]:
        """``(batch label, record)`` for every row, in a stable order."""
        out: list[tuple[str, dict[str, Any]]] = []
        for batch in self.batches:
            for record in batch.records:
                out.append((str(batch.path), record))
        return out

    @property
    def replays(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        for batch in self.batches:
            for record in batch.replays:
                out.append((str(batch.path), record))
        return out

    # -- the session log, read lazily and memoised (determinism: pure) --------
    _session_cache: dict[str, Optional[list[dict[str, Any]]]] = field(
        default_factory=dict, repr=False
    )

    def session_rows(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        """This row's ``<sid>_session.jsonl`` rows, or ``[]`` when it is absent."""
        raw = (record.get("artifacts") or {}).get("session_dir")
        if not raw:
            return []
        key = str(raw)
        if key in self._session_cache:
            return self._session_cache[key] or []
        directory = Path(raw)
        if not directory.is_absolute():
            directory = Path.cwd() / directory
            if not directory.is_dir():
                directory = REPO_ROOT / str(raw)
        path = directory.parent / f"{directory.name}_session.jsonl"
        rows: list[dict[str, Any]] = []
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                rows = []
        self._session_cache[key] = rows
        return rows

    def run_summary(self, record: Mapping[str, Any]) -> str:
        """The row's ``run.summary`` line, or ``""`` — the source of cost/retry counts."""
        summary = metrics_mod.run_summary_record(self.session_rows(record))
        return str(summary.get("summary") or "") if summary else ""

    def failure_classes(self, record: Mapping[str, Any]) -> Counter:
        """Count ``patched.result[...]`` rows by the failure class they carry."""
        counts: Counter = Counter()
        for row in self.session_rows(record):
            phase = str(row.get("phase") or "")
            if not phase.startswith(replay_mod.RESULT_SESSION_PHASE):
                continue
            match = re.search(r"failure_class=([A-Za-z_]+)", str(row.get("summary") or ""))
            if match:
                counts[match.group(1).lower()] += 1
        return counts


def load_analysis(
    patterns: Sequence[str],
    *,
    seed: int,
    resamples: int,
    allow_mixed_configs: bool = False,
) -> Analysis:
    paths = discover_batches(patterns)
    if not paths:
        raise AnalysisError(
            f"--batches {list(patterns)} matched no batch with a {MANIFEST_NAME}"
        )
    batches = [
        Batch(path=path, records=load_manifest(path / MANIFEST_NAME), replays=load_replays(path))
        for path in paths
    ]
    # Item 19 — refuse to pool rows the frozen config would make incomparable, BEFORE
    # any metric reads them. `allow_mixed_configs` is §Versioning's "unless told to".
    if not allow_mixed_configs:
        _refuse_mixed_configs(batches)
    return Analysis(batches=batches, seed=seed, resamples=resamples)


CONFIG_HASH_FIELD = "config_hash"
"""The row field item 19's frozen config writes (``frozen_config.FrozenConfig.record``).

[VANTA-WHY] Named once, because :func:`_refuse_mixed_configs` is the only reader that
turns the field into a refusal: a typo in a literal there would make the guard read
``None`` for every row and silently pool everything, and the row's own copies in
``runner.build_record`` would keep looking right."""

OVERRIDES_FIELD = "config_overrides"
"""The row field naming the overrides the frozen config was run with.

[VANTA-WHY] M3 item B8. ``config_hash`` is the hash of the config **file**, not of the
configuration a row was produced under: ``--config-override`` can waive a frozen
default or swap a sampling preset without touching the file, so two rows with the same
hash can be two different experiments. The rendered overrides are the other half of the
key. Named here for the same reason ``CONFIG_HASH_FIELD`` is — a typo would make the
guard read every row as "no overrides recorded" and pool them all."""


def _config_group(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """The ``(config_hash, config_overrides)`` key two rows must share to be poolable.

    Overrides are recorded as an unordered list of rendered strings
    (``frozen_config.Override.render``), so they are sorted into the key — two rows that
    applied the same overrides in a different order are the same configuration. A row
    with no ``config_overrides`` field at all is **not** the same claim as one that
    recorded an empty list, and gets its own key, exactly as a row with no
    ``config_hash`` does: the first says nothing about its overrides, the second says
    there were none, and pooling an unknown with a known under the known one's name is
    what the guard exists to stop.
    """
    value = record.get(CONFIG_HASH_FIELD)
    hash_key = None if value is None else str(value)
    raw = record.get(OVERRIDES_FIELD)
    if not isinstance(raw, list):
        return hash_key, None
    return hash_key, tuple(sorted(str(item) for item in raw))


def _refuse_mixed_configs(batches: Sequence[Batch]) -> None:
    """Item 19 — never pool rows from different frozen configurations.

    [VANTA-WHY] §Versioning's "rows from different configs are never pooled" is a
    ROW-level rule a batch directory cannot enforce: a batch resumed across a config
    edit carries both hashes in one manifest, and every metric here pools by bug and
    arm while ignoring which file produced a row. Two distinct hashes in one load is
    therefore a refusal rather than a warning — the numbers the caller is about to
    compute would be averages over two different experiments, and nothing downstream
    could tell. A row with no ``config_hash`` (one written before this field existed,
    or by a caller that supplied no frozen config) is its own group, never agreement
    with a hashed row.

    [VANTA-WHY] M3 item B8 — the key is the hash **and** the applied overrides
    (:func:`_config_group`): one file hash can be run with and without `--config-override`
    waivers, and those rows are as incomparable as two different files. So the guard
    reads both fields and the refusal names the overrides that made the groups differ.
    """
    groups: dict[tuple[Any, Any], list[str]] = {}
    for batch in batches:
        for record in batch.records:
            groups.setdefault(_config_group(record), []).append(str(batch.path))
    if len(groups) <= 1:
        return
    described: list[str] = []
    for (hash_key, overrides), where in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0] is not None,
            item[0][0] or "",
            item[0][1] is not None,
            item[0][1] or (),
        ),
    ):
        if hash_key is None:
            label = "none recorded"
        else:
            label = hash_key[:12]
            if overrides is None:
                label += " (overrides not recorded)"
            elif overrides:
                label += " (with overrides " + ", ".join(overrides) + ")"
        described.append(
            f"{label} ({len(where)} row(s) in {', '.join(sorted(set(where)))})"
        )
    raise AnalysisError(
        f"{len(groups)} distinct frozen configurations across the loaded rows: "
        + "; ".join(described)
        + ". Rows from different configs are never pooled (§Versioning). Analyse the "
        "batches separately, or pass --allow-mixed-configs to pool them on purpose."
    )


# ---------------------------------------------------------------------------
# the plug-in contract
# ---------------------------------------------------------------------------
@dataclass
class Artifact:
    """One output file: its name and its exact text."""

    filename: str
    text: str


@dataclass
class MetricResult:
    """What one metric produced: files to write, lines for ``summary.txt``."""

    name: str
    artifacts: list[Artifact] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


class Metric:
    """One plug-in over the loaded matrix (M3.5 §6.6 fold-in 10)."""

    name = "metric"

    def run(self, analysis: Analysis) -> MetricResult:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# figure writers — deterministic hand-written SVG, no third-party dependency
# ---------------------------------------------------------------------------
_WIDTH = 640
_HEIGHT = 360
_MARGIN = 56


def _svg_open(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111">',
    ]


def _svg_close() -> list[str]:
    return ["</g>", "</svg>", ""]


def figure_horizontal_bars(title: str, entries: Sequence[tuple[str, Optional[float]]]) -> str:
    """A labelled bar per entry; ``None`` draws an empty slot, never a zero bar."""
    slot = 24
    height = 60 + slot * max(1, len(entries))
    lines = _svg_open(_WIDTH, height)
    lines.append(f'<text x="{_MARGIN}" y="24" font-size="15">{_tex_text(title)}</text>')
    left = 180
    right = _WIDTH - 24
    top = 44
    for index, (label, value) in enumerate(entries):
        y = top + index * slot
        lines.append(f'<text x="8" y="{y + 12}">{_tex_text(label)}</text>')
        lines.append(
            f'<line x1="{left}" y1="{y + 10}" x2="{right}" y2="{y + 10}" stroke="#eee"/>'
        )
        if value is None:
            lines.append(f'<text x="{left + 4}" y="{y + 12}">n/a</text>')
            continue
        width = max(0.0, min(1.0, value)) * (right - left)
        lines.append(
            f'<rect x="{left}" y="{y + 2}" width="{width:.2f}" height="12" fill="#2f6fb3"/>'
        )
        lines.append(f'<text x="{left + width + 6:.2f}" y="{y + 12}">{value:.3f}</text>')
    lines.extend(_svg_close())
    return "\n".join(lines)


def _tex_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def figure_lines(
    title: str,
    xs: Sequence[str],
    series: Sequence[tuple[str, Sequence[Optional[float]], str]],
) -> str:
    """A line chart over categorical ``xs``; ``None`` breaks the line."""
    lines = _svg_open(_WIDTH, _HEIGHT)
    lines.append(f'<text x="{_MARGIN}" y="24" font-size="15">{_tex_text(title)}</text>')
    left, right = _MARGIN, _WIDTH - 24
    top, bottom = 44, _HEIGHT - 48
    lines.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="#fafafa" stroke="#ddd"/>')
    steps = max(1, len(xs))
    for index, name in enumerate(xs):
        x = left + (index + 0.5) * (right - left) / steps
        lines.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">{_tex_text(name)}</text>')
    for value, frac in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
        y = bottom - frac * (bottom - top)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#eee"/>')
        lines.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    legend_y = top + 12
    for name, values, colour in series:
        points: list[str] = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + (index + 0.5) * (right - left) / steps
            y = bottom - max(0.0, min(1.0, value)) * (bottom - top)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="2"/>')
        lines.append(f'<text x="{right - 150}" y="{legend_y}" fill="{colour}">— {_tex_text(name)}</text>')
        legend_y += 16
    lines.extend(_svg_close())
    return "\n".join(lines)


def figure_scatter(title: str, points: Sequence[tuple[str, float, float]]) -> str:
    """Fix rate (x) against regression rate (y), one label per point."""
    lines = _svg_open(_WIDTH, _HEIGHT)
    lines.append(f'<text x="{_MARGIN}" y="24" font-size="15">{_tex_text(title)}</text>')
    left, right = _MARGIN, _WIDTH - 24
    top, bottom = 44, _HEIGHT - 48
    lines.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="#fafafa" stroke="#ddd"/>')
    lines.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 30}" text-anchor="middle">fix rate</text>')
    lines.append(f'<text x="14" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {(top + bottom) / 2:.1f})">regression rate</text>')
    for label, x_value, y_value in points:
        x = left + max(0.0, min(1.0, x_value)) * (right - left)
        y = bottom - max(0.0, min(1.0, y_value)) * (bottom - top)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#c0504d"/>')
        lines.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}">{_tex_text(label)}</text>')
    lines.extend(_svg_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# metric 1 — review rates, per arm x candidate kind
# ---------------------------------------------------------------------------
def review_confusion(analysis: Analysis) -> dict[tuple[str, str], Counter]:
    """``{(arm, kind): Counter(cell)}`` over the rows that carry a candidate."""
    out: dict[tuple[str, str], Counter] = {}
    for _batch, record in analysis.rows:
        kind = candidate_kind(record)
        if kind is None:
            continue
        cell = LABEL_TO_CELL.get(label_for(record))
        if cell is None:
            continue
        key = (arms_mod.row_arm(record), kind)
        out.setdefault(key, Counter())[cell] += 1
    return out


class ReviewMetrics(Metric):
    name = "review"

    def run(self, analysis: Analysis) -> MetricResult:
        table = review_confusion(analysis)
        rows: list[str] = []
        summary: list[str] = ["", "== Review rates (per arm x candidate kind) =="]
        for (arm, kind), counts in sorted(table.items()):
            decided = sum(counts.get(cell, 0) for cell in (CELL_TA, CELL_FA, CELL_TR, CELL_FR))
            approval = approval_rate(counts)
            rejection = rejection_rate(counts)
            balanced = balanced_accuracy(counts)
            rows.append(
                " & ".join(
                    [
                        _tex(arm),
                        _tex(kind),
                        str(counts.get(CELL_TA, 0)),
                        str(counts.get(CELL_FA, 0)),
                        str(counts.get(CELL_TR, 0)),
                        str(counts.get(CELL_FR, 0)),
                        str(counts.get(CELL_NR, 0)),
                        f"{_fmt(approval)} {_interval_text(metrics_mod.wilson_interval(counts.get(CELL_TA, 0) + counts.get(CELL_FA, 0), decided))}",
                        f"{_fmt(rejection)} {_interval_text(metrics_mod.wilson_interval(counts.get(CELL_TR, 0) + counts.get(CELL_FR, 0), decided))}",
                        _fmt(balanced),
                    ]
                )
                + r" \\"
            )
            summary.append(
                f"  {arm:<12} {kind:<18} TA={counts.get(CELL_TA, 0)} FA={counts.get(CELL_FA, 0)} "
                f"TR={counts.get(CELL_TR, 0)} FR={counts.get(CELL_FR, 0)} NR={counts.get(CELL_NR, 0)} "
                f"approval={_fmt(approval)} rejection={_fmt(rejection)} balanced_acc={_fmt(balanced)}"
            )
        header = (
            r"% item 12 — review rates. TA/FA/TR/FR/NR are §5.3's cells; the rate"
            "\n"
            r"% columns carry a Wilson 95% interval over the decided rows (TA+FA+TR+FR)."
            "\n"
            r"\begin{tabular}{llrrrrrllr}"
            "\n"
            r"\hline"
            "\n"
            r"arm & kind & TA & FA & TR & FR & NR & approval & rejection & bal.\ acc \\"
            "\n"
            r"\hline"
            "\n"
        )
        body = "\n".join(rows) if rows else r"% no review rows in the input batches"
        tex = header + body + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n"
        bars = [
            (f"{arm}/{kind}", balanced_accuracy(counts))
            for (arm, kind), counts in sorted(table.items())
        ]
        return MetricResult(
            name=self.name,
            artifacts=[
                Artifact("review_rates.tex", tex),
                Artifact("review_rates.svg", figure_horizontal_bars("balanced accuracy by arm x kind", bars)),
            ],
            summary=summary,
        )


# ---------------------------------------------------------------------------
# metric 2 — paired comparisons (exact McNemar + cluster bootstrap by bug)
# ---------------------------------------------------------------------------
def pair_index(analysis: Analysis) -> dict[tuple[str, str], bool]:
    """``{(arm, bug#seed#candidate): verdict-was-right}``.

    The pairing key carries the seed as well as the bug, because two seeds of one
    bug are two instances of the experiment — pairing them onto one key would let
    the second quietly overwrite the first.
    """
    index: dict[tuple[str, str], bool] = {}
    for _batch, record in analysis.rows:
        identity = candidate_id(record)
        label = label_for(record)
        if identity is None or label not in LABEL_CORRECT:
            continue
        key = f"{record.get('bug')}#{record.get('seed')}#{identity}"
        index[(arms_mod.row_arm(record), key)] = LABEL_CORRECT[label]
    return index


def paired_counts(
    index: Mapping[tuple[str, str], bool], left: str, right: str
) -> tuple[list[tuple[str, bool, bool]], int, int]:
    """The shared rows, and the two discordant counts ``(b, c)``.

    ``b`` counts rows the left arm got right and the right arm wrong; ``c`` the
    reverse. Rows only one arm decided are outside the pairing — a McNemar test is
    over pairs, and a row with no partner is not one.
    """
    keys = sorted({key for (arm, key) in index if arm == left} & {key for (arm, key) in index if arm == right})
    shared = [(key, index[(left, key)], index[(right, key)]) for key in keys]
    b = sum(1 for _key, a_ok, b_ok in shared if a_ok and not b_ok)
    c = sum(1 for _key, a_ok, b_ok in shared if not a_ok and b_ok)
    return shared, b, c


class PairedMetrics(Metric):
    name = "paired"

    def run(self, analysis: Analysis) -> MetricResult:
        index = pair_index(analysis)
        rows: list[str] = []
        summary: list[str] = ["", "== Paired comparisons (exact McNemar, cluster bootstrap by bug) =="]
        for left, right, title in PAIRS:
            shared, b, c = paired_counts(index, left, right)
            if not shared:
                summary.append(f"  {title}: no paired rows")
                continue
            left_rate = sum(1 for _k, a_ok, _b_ok in shared if a_ok) / len(shared)
            right_rate = sum(1 for _k, _a_ok, b_ok in shared if b_ok) / len(shared)
            p_value = mcnemar_exact(b, c)

            by_bug: dict[str, list[tuple[bool, bool]]] = {}
            for key, a_ok, b_ok in shared:
                bug = key.split("#", 1)[0]
                by_bug.setdefault(bug, []).append((a_ok, b_ok))

            def difference(pairs: list[tuple[bool, bool]]) -> Optional[float]:
                if not pairs:
                    return None
                return sum(1 for a_ok, _b in pairs if a_ok) / len(pairs) - sum(
                    1 for _a, b_ok in pairs if b_ok
                ) / len(pairs)

            interval = cluster_bootstrap(
                by_bug, difference, seed=analysis.seed, resamples=analysis.resamples
            )
            rows.append(
                " & ".join(
                    [
                        _tex(title),
                        str(len(shared)),
                        str(b),
                        str(c),
                        f"{left_rate:.3f}",
                        f"{right_rate:.3f}",
                        f"{left_rate - right_rate:+.3f}",
                        f"{p_value:.4f}",
                        _interval_text(interval),
                    ]
                )
                + r" \\"
            )
            summary.append(
                f"  {title:<20} n={len(shared)} b={b} c={c} "
                f"diff={left_rate - right_rate:+.3f} mcnemar_p={p_value:.4f} ci={_interval_text(interval)}"
            )
        header = (
            r"% item 12 — paired comparisons. `b`/`c` are the discordant pairs"
            "\n"
            r"% (left-right, right-left); p is the exact two-sided McNemar value."
            "\n"
            r"\begin{tabular}{lrrrrrrrl}"
            "\n"
            r"\hline"
            "\n"
            r"pair & n & b & c & left & right & diff & p & 95\% CI \\"
            "\n"
            r"\hline"
            "\n"
        )
        body = "\n".join(rows) if rows else r"% no paired rows in the input batches"
        tex = header + body + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n"
        return MetricResult(name=self.name, artifacts=[Artifact("paired.tex", tex)], summary=summary)


# ---------------------------------------------------------------------------
# metric 3 — repair metrics
# ---------------------------------------------------------------------------
@dataclass
class RepairCounts:
    total: int = 0
    applied: int = 0
    fixed: int = 0
    regressed: int = 0
    fixed_first_attempt: int = 0
    attempt_hist: Counter = field(default_factory=Counter)
    adversarial_catches: int = 0
    replayed: int = 0
    inconclusive: int = 0
    """Repair rows whose grading was unreadable — the patch applied but the hidden run
    wrote no pytest report at all (``label_input`` = ``inconclusive``, item B4).

    [VANTA-WHY] M3 item B4's general rule, applied to the analysis: such a row is
    neither a fix nor a failure, so it is counted here and kept out of every rate —
    ``total`` is the fix-rate denominator, so leaving it in would read the row as
    "not fixed", i.e. blame the model on the strength of a run nobody could read.
    """

    @property
    def fix_rate(self) -> Optional[float]:
        return (self.fixed / self.total) if self.total else None

    @property
    def regression_rate(self) -> Optional[float]:
        return (self.regressed / self.applied) if self.applied else None

    @property
    def adversarial_catch_rate(self) -> Optional[float]:
        return (self.adversarial_catches / self.replayed) if self.replayed else None


def repair_counts(analysis: Analysis) -> RepairCounts:
    """§5's repair numbers, from the repair rows and their replay records.

    A regression is a test the patch broke that the baseline passed — the oracle's
    own ``suite_regressions``, which is already net of the baseline's pre-existing
    failures by construction, so there is no second baseline to subtract.

    The adversarial catch rate needs the replay matrix: a generated test that went
    pass -> fail across a transition while the row's own oracle recorded NO native
    regression is a regression the generated suite caught and the native suite did
    not. A row with no replay record is outside that denominator, not a zero.

    A row whose grading was unreadable (``inconclusive``) is in no denominator at all:
    it is neither a fix nor a failure, so it is counted apart (item B4's general rule).
    """
    counts = RepairCounts()
    by_session: dict[str, dict[str, Any]] = {}
    for _batch, replay in analysis.replays:
        session = str(replay.get("session_id") or "")
        if session:
            by_session[session] = replay

    for _batch, record in analysis.rows:
        if str(record.get("mode") or "") != arms_mod.REPAIR:
            continue
        oracle = row_oracle(record)
        if oracle is None:
            continue
        if oracle.inconclusive:
            # [VANTA-WHY] Item B4's general rule — an inconclusive grade (the patch
            # applied but the hidden run wrote no report) is neither a fix nor a
            # failure. It is held in its own count and kept out of every rate; only
            # `total` is the fix-rate denominator, so leaving it in would read the row
            # as "not fixed", i.e. bad, on the strength of an unreadable run.
            counts.inconclusive += 1
            continue
        counts.total += 1
        if oracle.apply_ok:
            counts.applied += 1
        if oracle.patch_good:
            counts.fixed += 1
            if record.get("fixed_at_attempt_1"):
                counts.fixed_first_attempt += 1
        if oracle.suite_regressions:
            counts.regressed += 1
        attempts = record.get("attempts_run")
        if isinstance(attempts, int):
            counts.attempt_hist[attempts] += 1

        replay = by_session.get(str(record.get("session_id") or ""))
        if replay is None:
            continue
        counts.replayed += 1
        generated_caught = False
        for entry in (replay.get("views") or {}).get(replay_mod.VIEW_GENERATED, {}).get("transitions", []):
            slot = (entry.get("by_source") or {}).get(replay_mod.SOURCE_VANTA_GENERATED)
            if slot and slot.get("n_pf", 0) > 0:
                generated_caught = True
                break
        if generated_caught and not oracle.suite_regressions:
            counts.adversarial_catches += 1
    return counts


class RepairMetrics(Metric):
    name = "repair"

    def run(self, analysis: Analysis) -> MetricResult:
        counts = repair_counts(analysis)
        attempt_text = ", ".join(
            f"{attempt}:{counts.attempt_hist[attempt]}" for attempt in sorted(counts.attempt_hist)
        ) or "n/a"
        rows = [
            ("repair rows", str(counts.total)),
            ("applied cleanly", str(counts.applied)),
            ("fixed (oracle patch_good)", str(counts.fixed)),
            ("inconclusive gradings (in no rate)", str(counts.inconclusive)),
            ("fix rate", _fmt(counts.fix_rate)),
            ("fixed at attempt 1", str(counts.fixed_first_attempt)),
            ("attempts_run histogram", _tex(attempt_text)),
            ("regressed (native)", str(counts.regressed)),
            ("regression rate (net of baseline)", _fmt(counts.regression_rate)),
            ("rows with a replay record", str(counts.replayed)),
            ("adversarial catch rate", _fmt(counts.adversarial_catch_rate)),
            ("cost per safely fixed defect", cost_per_fixed_defect(analysis)),
        ]
        tex = (
            r"% item 12 — repair metrics. `regressed` counts oracle suite_regressions;"
            "\n"
            r"% a regression is already net of the baseline's own failures."
            "\n"
            r"\begin{tabular}{lr}"
            "\n"
            r"\hline"
            "\n"
            + "\n".join(f"{_tex(name)} & {_tex(value)} \\\\" for name, value in rows)
            + "\n"
            + r"\hline"
            + "\n"
            + r"\end{tabular}"
            + "\n"
        )
        summary = ["", "== Repair metrics =="] + [f"  {name:<36} {value}" for name, value in rows]
        return MetricResult(
            name=self.name,
            artifacts=[
                Artifact("repair.tex", tex),
                Artifact(
                    "fix_regression.svg",
                    figure_scatter(
                        "fix rate vs regression rate per arm",
                        _fix_regression_points(analysis),
                    ),
                ),
            ],
            summary=summary,
        )


def _fix_regression_points(analysis: Analysis) -> list[tuple[str, float, float]]:
    """One point per arm: its fix rate against its regression rate."""
    by_arm: dict[str, RepairCounts] = {}
    by_session = {
        str(replay.get("session_id") or ""): replay
        for _b, replay in analysis.replays
    }
    for _batch, record in analysis.rows:
        if str(record.get("mode") or "") != arms_mod.REPAIR:
            continue
        oracle = row_oracle(record)
        if oracle is None:
            continue
        arm = arms_mod.row_arm(record)
        counts = by_arm.setdefault(arm, RepairCounts())
        if oracle.inconclusive:
            # Item B4's general rule — see `repair_counts`: an unreadable grading has
            # no place in this per-arm scatter's denominators either.
            counts.inconclusive += 1
            continue
        counts.total += 1
        if oracle.apply_ok:
            counts.applied += 1
        if oracle.patch_good:
            counts.fixed += 1
        if oracle.suite_regressions:
            counts.regressed += 1
        if str(record.get("session_id") or "") in by_session:
            counts.replayed += 1
    points: list[tuple[str, float, float]] = []
    for arm in sorted(by_arm):
        counts = by_arm[arm]
        if counts.fix_rate is None or counts.regression_rate is None:
            continue
        points.append((arm, counts.fix_rate, counts.regression_rate))
    return points


# ---------------------------------------------------------------------------
# metric 3b — per repair run: localization_hit beside fix-file coverage (M3 item A2)
# ---------------------------------------------------------------------------
def coverage_pair(text: Any) -> Optional[tuple[int, int]]:
    """One row's recorded ``fix_file_coverage`` as ``(shown_in_full, gold_files)``.

    The manifest field is the string ``"k/n"`` the runner wrote after the run
    (M3 item A2): k gold fix files the run was shown **in full**, over the n the
    fix touches. Anything else — ``None`` on a review row, a row from before the
    field existed, a malformed value — is ``None``: the row cannot answer, and a
    reader must never turn "no reading" into ``0/0`` (item B4's general rule, the
    same one :func:`unavailable` exists for).
    """
    if not isinstance(text, str) or "/" not in text:
        return None
    shown, _, total = text.partition("/")
    try:
        shown_i, total_i = int(shown), int(total)
    except ValueError:
        return None
    # [VANTA-WHY] M3 item E10 — a pair that cannot describe a coverage reading is not
    # one. `0/0` (no gold file to show), `1/0` and `5/3` (shown more than the fix
    # touches) all passed the parse above and were summed into
    # `with_reading`/`shown_full`/`gold_total`, so a pooled rate could exceed 1 or
    # divide by zero. `0/n` IS a reading ("none of the n shown whole"), so it stays.
    if total_i < 1 or shown_i < 0 or shown_i > total_i:
        return None
    return shown_i, total_i


class LocalizationMetrics(Metric):
    """M3 item A2 — fix-file coverage per repair run, reported beside ``localization_hit``.

    The two questions are asked of the same selection and answer different things:
    ``localization_hit`` says whether **any** gold file was among those the run was
    shown; ``fix_file_coverage`` says **how much of the fix** was shown whole
    (``k/n``), which are different numbers on a multi-file bug. Both are read from
    the manifest rows the runner wrote after each run — this metric computes
    neither, it only reports them side by side, so the column cannot become a term
    the selector optimises (the answer is fixed on the row before any analysis).

    A repair row with no coverage reading (a pre-A2 row, or a fix that touches only
    tests) prints ``—`` and is in no denominator; only rows that recorded a reading
    feed the pooled rate. Rows already on disk are never rewritten.
    """

    name = "localization"

    def run(self, analysis: Analysis) -> MetricResult:
        rows: list[tuple[str, str, str, str]] = []
        hits: dict[Optional[bool], int] = {True: 0, False: 0, None: 0}
        shown_full = 0
        gold_total = 0
        with_reading = 0
        for _batch, record in analysis.rows:
            if str(record.get("mode") or "") != arms_mod.REPAIR:
                continue
            hit = record.get("localization_hit")
            hit = hit if isinstance(hit, bool) else None
            hits[hit] += 1
            pair = coverage_pair(record.get("fix_file_coverage"))
            if pair is None:
                coverage_text = "—"
            else:
                with_reading += 1
                shown_full += pair[0]
                gold_total += pair[1]
                coverage_text = f"{pair[0]}/{pair[1]}"
            reason = record.get("fix_file_coverage_reason")
            rows.append(
                (
                    str(record.get("key") or ""),
                    {True: "yes", False: "no", None: "—"}[hit],
                    coverage_text,
                    str(reason) if reason else "",
                )
            )
        rows.sort(key=lambda row: row[0])

        if with_reading:
            pooled = _fmt(shown_full / gold_total)
        else:
            pooled = unavailable("no repair row recorded fix_file_coverage")
        summary = [
            "",
            "== Fix-file coverage per repair run (M3 item A2) ==",
            f"  rows with a coverage reading        {with_reading}",
            f"  pooled fix-file coverage            {shown_full}/{gold_total} ({pooled})"
            if with_reading
            else f"  pooled fix-file coverage            {pooled}",
            f"  localization_hit: hit {hits[True]} / miss {hits[False]} / unknown {hits[None]}",
        ]
        summary += [
            f"  {key:<40} hit={hit:<4} coverage={coverage:<5} {reason}".rstrip()
            for key, hit, coverage, reason in rows
        ]

        header = (
            r"% item 12 — M3 item A2: per repair run, `localization_hit` beside fix-file"
            "\n"
            r"% coverage (gold fix files shown in full / gold fix files). `reason` is"
            "\n"
            r"% blank when the row answered; a row with no reading is `—`, never `0/0`."
            "\n"
            r"\begin{tabular}{llll}"
            "\n"
            r"\hline"
            "\n"
            r"run & localization hit & fix-file coverage & reason \\"
            "\n"
            r"\hline"
            "\n"
        )
        body = "\n".join(
            " & ".join([_tex(key), hit, coverage, _tex(reason)]) + r" \\"
            for key, hit, coverage, reason in rows
        ) or r"% no repair rows in the input batches"
        tex = header + body + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n"
        return MetricResult(
            name=self.name,
            artifacts=[Artifact("localization.tex", tex)],
            summary=summary,
        )


# ---------------------------------------------------------------------------
# metric 4 — EIR/ECR per transition, pooled and per source
# ---------------------------------------------------------------------------
_COUNT_KEYS = ("n_pp", "n_pf", "n_fp", "n_ff", "n_censored")


def _zero() -> dict[str, int]:
    return {key: 0 for key in _COUNT_KEYS}


def _add(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key in _COUNT_KEYS:
        target[key] += int(source.get(key) or 0)


def view_transitions(
    analysis: Analysis, view: str
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]], dict[str, list[dict[str, dict[str, int]]]]]:
    """``(pooled, per_source, by_bug)`` for one view, recomputed from the cells.

    ``pooled[k]`` sums every row's counts for the transition ``k``; ``per_source``
    cuts the same counts by ``evidence_source``; ``by_bug`` keeps one entry per row
    so the cluster bootstrap can resample bugs rather than rows.
    """
    pooled: dict[str, dict[str, int]] = {}
    per_source: dict[str, dict[str, dict[str, int]]] = {}
    by_bug: dict[str, list[dict[str, dict[str, int]]]] = {}
    for _batch, replay in analysis.replays:
        if view not in (replay.get("views") or {}):
            continue
        cells = replay.get("cells") or []
        flaky = replay.get("flaky") or []
        entries = replay_mod.transitions(replay_mod.cells_for_view(cells, view), excluded=flaky)
        row_entries: dict[str, dict[str, dict[str, int]]] = {}
        for entry in entries:
            key = str(entry["k"])
            _add(pooled.setdefault(key, _zero()), entry)
            slot_map = per_source.setdefault(key, {})
            for source, slot in sorted((entry.get("by_source") or {}).items()):
                _add(slot_map.setdefault(source, _zero()), slot)
            row_entries[key] = {
                source: dict(slot) for source, slot in (entry.get("by_source") or {}).items()
            }
            summary_slot = _zero()
            _add(summary_slot, entry)
            row_entries[key]["__all__"] = summary_slot
        bug = str(replay.get("bug") or "")
        by_bug.setdefault(bug, []).append(row_entries)
    return pooled, per_source, by_bug


def _bootstrap_rate(
    by_bug: Mapping[str, Sequence[Mapping[str, Mapping[str, int]]]],
    transition: str,
    source: Optional[str],
    field: str,
    *,
    seed: int,
    resamples: int,
) -> Optional[tuple[float, float]]:
    bucket = source or "__all__"

    def statistic(rows: list[Mapping[str, Mapping[str, int]]]) -> Optional[float]:
        total = _zero()
        for row in rows:
            counts = row.get(transition)
            if counts and bucket in counts:
                _add(total, counts[bucket])
        return replay_mod.rates(total)[field]

    return cluster_bootstrap(by_bug, statistic, seed=seed, resamples=resamples)


class EirEcrMetrics(Metric):
    name = "eir_ecr"

    def run(self, analysis: Analysis) -> MetricResult:
        summary: list[str] = ["", "== EIR / ECR per transition (generated and oracle views) =="]
        rows: list[str] = []
        for view in sorted({view for _b, replay in analysis.replays for view in (replay.get("views") or {})}):
            pooled, per_source, by_bug = view_transitions(analysis, view)
            for transition in sorted(pooled):
                rates = replay_mod.rates(pooled[transition])
                denominators = rates["denominators"]
                ci_eir = _bootstrap_rate(by_bug, transition, None, "eir", seed=analysis.seed, resamples=analysis.resamples)
                ci_ecr = _bootstrap_rate(by_bug, transition, None, "ecr", seed=analysis.seed, resamples=analysis.resamples)
                stability = _stability(rates)
                rows.append(
                    " & ".join(
                        [
                            _tex(view),
                            "pooled",
                            _tex(transition),
                            str(denominators["eir"]),
                            f"{_fmt(rates['eir'])} {_interval_text(ci_eir)}",
                            str(denominators["ecr"]),
                            f"{_fmt(rates['ecr'])} {_interval_text(ci_ecr)}",
                            _fmt(rates["acc"]),
                            str(denominators["censored"]),
                            stability,
                        ]
                    )
                    + r" \\"
                )
                for source in sorted(per_source.get(transition, {})):
                    source_counts = per_source[transition][source]
                    source_rates = replay_mod.rates(source_counts)
                    source_den = source_rates["denominators"]
                    rows.append(
                        " & ".join(
                            [
                                _tex(view),
                                _tex(source),
                                _tex(transition),
                                str(source_den["eir"]),
                                _fmt(source_rates["eir"]),
                                str(source_den["ecr"]),
                                _fmt(source_rates["ecr"]),
                                _fmt(source_rates["acc"]),
                                str(source_den["censored"]),
                                _stability(source_rates),
                            ]
                        )
                        + r" \\"
                    )
                summary.append(
                    f"  {view:<10} {transition:<8} EIR={_fmt(rates['eir'])} (n={denominators['eir']}) "
                    f"ECR={_fmt(rates['ecr'])} (n={denominators['ecr']}) Acc={_fmt(rates['acc'])} "
                    f"censored={denominators['censored']} stable={stability}"
                )
        header = (
            r"% item 12 — EIR/ECR. Every rate prints its own denominator; `censored`"
            "\n"
            r"% is the count of transitions thrown away (an unobserved cell at either end)."
            "\n"
            r"% `stable` is the §5 check: it holds when ECR/EIR > Acc/(1-Acc)."
            "\n"
            r"\begin{tabular}{lllrrlrlrl}"
            "\n"
            r"\hline"
            "\n"
            r"view & source & k & n\_EIR & EIR (95\% CI) & n\_ECR & ECR (95\% CI) & Acc & cens. & stable \\"
            "\n"
            r"\hline"
            "\n"
        )
        body = "\n".join(rows) if rows else r"% no replay records in the input batches"
        tex = header + body + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n"
        figure = _eir_ecr_figure(analysis)
        return MetricResult(
            name=self.name,
            artifacts=[Artifact("eir_ecr.tex", tex), Artifact("eir_ecr.svg", figure)],
            summary=summary,
        )


def _stability(rates: Mapping[str, Any]) -> str:
    """The §5 stability check, or ``n/a`` when either side of it is undefined.

    ``ECR/EIR > Acc/(1-Acc)`` compares two ratios whose denominators can each be
    zero; an undefined comparison is reported as undefined, never as ``False``.
    """
    eir, ecr, acc = rates.get("eir"), rates.get("ecr"), rates.get("acc")
    if not eir or ecr is None or acc is None or acc in (0.0, 1.0):
        return "n/a"
    return "yes" if (ecr / eir) > (acc / (1.0 - acc)) else "no"


def _eir_ecr_figure(analysis: Analysis) -> str:
    """EIR and ECR against k for the generated view, with the stability threshold."""
    pooled, _per_source, _by_bug = view_transitions(analysis, replay_mod.VIEW_GENERATED)
    keys = sorted(pooled)
    eir = [replay_mod.rates(pooled[key])["eir"] for key in keys]
    ecr = [replay_mod.rates(pooled[key])["ecr"] for key in keys]
    threshold: list[Optional[float]] = []
    for key in keys:
        acc = replay_mod.rates(pooled[key])["acc"]
        threshold.append(None if acc is None or acc in (0.0, 1.0) else acc / (1.0 - acc))
    return figure_lines(
        "EIR / ECR per transition (generated view)",
        keys,
        (
            ("EIR", eir, "#c0504d"),
            ("ECR", ecr, "#2f6fb3"),
            ("Acc/(1-Acc)", threshold, "#888"),
        ),
    )


# ---------------------------------------------------------------------------
# metric 5 — pipeline metrics
# ---------------------------------------------------------------------------
class PipelineMetrics(Metric):
    name = "pipeline"

    def run(self, analysis: Analysis) -> MetricResult:
        generated = admitted_first = admitted_after = discarded = 0
        for _batch, replay in analysis.replays:
            for test in replay.get("tests") or []:
                if test.get("evidence_source") != replay_mod.SOURCE_VANTA_GENERATED:
                    continue
                generated += 1
                round_admitted = test.get("round_admitted")
                if round_admitted == 1:
                    admitted_first += 1
                elif isinstance(round_admitted, int) and round_admitted > 1:
                    admitted_after += 1
                if test.get("assertion_grounding_type") == replay_mod.GROUNDING_UNGROUNDED_SCRAPPED:
                    discarded += 1

        truncations = 0
        retries = 0
        patch_apply_failed = 0
        failure_classes: Counter = Counter()
        for _batch, record in analysis.rows:
            summary = analysis.run_summary(record)
            if summary:
                truncations += _int_field(summary, "truncated_calls")
                retries += _int_field(summary, "provider_retries")
            if str(record.get("stop_reason") or "") == "patch_application_failed":
                patch_apply_failed += 1
            failure_classes.update(analysis.failure_classes(record))

        class_text = ", ".join(
            f"{name}:{failure_classes[name]}" for name in sorted(failure_classes)
        ) or "n/a"
        rows = [
            ("tests generated", str(generated)),
            ("admitted first time", str(admitted_first)),
            ("admitted after revision", str(admitted_after)),
            ("discarded (ungrounded, scrapped)", str(discarded)),
            ("failure classes", _tex(class_text)),
            ("patch applications failed", str(patch_apply_failed)),
            ("truncated calls", str(truncations)),
            ("provider retries", str(retries)),
        ]
        tex = (
            r"% item 12 — pipeline metrics. Test counts come from the replay records;"
            "\n"
            r"% truncations and retries from each run's own summary line."
            "\n"
            r"\begin{tabular}{lr}"
            "\n"
            r"\hline"
            "\n"
            + "\n".join(f"{_tex(name)} & {value} \\\\" for name, value in rows)
            + "\n"
            + r"\hline"
            + "\n"
            + r"\end{tabular}"
            + "\n"
        )
        summary = ["", "== Pipeline metrics =="] + [f"  {name:<36} {value}" for name, value in rows]
        return MetricResult(name=self.name, artifacts=[Artifact("pipeline.tex", tex)], summary=summary)


def _int_field(summary: str, key: str) -> int:
    raw = metrics_mod.summary_field(summary, key)
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _float_field(summary: str, key: str) -> Optional[float]:
    """A ``call.usage`` summary field as ``float``, or ``None`` when absent.

    ``metrics.summary_field`` returns the field's TEXT — a summary line is a
    string — so a recorded number arrives as e.g. ``"0.0050"`` and has to be
    parsed before it can be summed. ``None`` (field missing, or the ``unset``
    placeholder) is "the call did not record this", which is kept distinct from
    a recorded ``0.0``.
    """
    raw = metrics_mod.summary_field(summary, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# metric 6 — cost and time (item 15)
# ---------------------------------------------------------------------------
# [VANTA-WHY] The pipeline prices every model call from its own versioned table
# (`llm::pricing`, PRICE_TABLE — a rate per token for input, for CACHED input
# and for output) and writes the money, the tokens and the `cost_basis` that
# produced it on the call's own `call.usage` row, plus the run's summed figure on
# the row's `llm.cost_usd`. This script never re-prices a token: a second price
# table here could disagree with the first (the harness makes the same argument
# for copying `cost_usd` rather than re-deriving it), so the numbers below SUM
# what the pipeline already priced. Where no row was priced — the local backend
# is `not_billed` — the metric says `unavailable(...)`, never a fabricated `0`.
CALL_USAGE_PHASE = "call.usage"

SITE_ROLE: dict[str, str] = {
    "creator": "creator",
    "attacker_generate": "attacker",
    "attacker_revise": "attacker",
    "attacker_simplify": "attacker",
    "intent_audit": "intent_audit",
    "judge": "judge",
}
"""The reportable role of each call site, mirroring ``llm::truncation_role``.

The Attacker has four call sites and one role, so "which role spent the money"
and "which of that role's calls did" are different questions; this map answers
the first. A site outside the map reports under its own name rather than being
dropped, so a call whose cost vanished from every role cannot make a per-role
total a lower bound nobody labelled. Costs are grouped by SITE in the record and
never re-derived from the row summary, which carries no per-role money.
"""


def _int_or_none(value: Any) -> Optional[int]:
    """A recorded count as ``int``, or ``None`` when it is absent/unusable.

    ``None`` is "not recorded" and is kept distinct from ``0``, which is
    "recorded, and it was zero" — the whole distinction this section preserves.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number_or_none(value: Any) -> Optional[float]:
    """A recorded money/wall figure as ``float``, or ``None`` when unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _row_price(record: Mapping[str, Any]) -> Optional[float]:
    """One row's priced spend, or ``None`` when the run priced nothing.

    Read from ``llm.cost_usd`` — the run's OWN total, which the pipeline sums
    from the calls it priced. ``None`` is the local backend (``not_billed``) and
    is NOT ``0.0``: "nothing was priced" and "priced, and it was free" are
    different facts, and averaging them together would understate the hosted
    cost.
    """
    llm = record.get("llm")
    if not isinstance(llm, dict):
        return None
    return _number_or_none(llm.get("cost_usd"))


@dataclass
class CostFacts:
    """Every cost/time figure item 12's table prints, summed from the records.

    Each ``Optional`` field is a measurement or a hole: ``None`` means no input
    row carried the fact, so the metric prints ``unavailable(...)`` rather than a
    zero. The dicts are keyed in sorted order when rendered, so the output is a
    pure function of the batches.
    """

    by_arm_cost: dict[str, float] = field(default_factory=dict)
    by_arm_verdicts: dict[str, int] = field(default_factory=dict)
    correct_cost: dict[str, float] = field(default_factory=dict)
    correct_count: dict[str, int] = field(default_factory=dict)
    fixed_cost: Optional[float] = None
    fixed_count: int = 0
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_ms: Optional[int] = None
    sandbox_s: Optional[float] = None
    by_role_cost: dict[str, float] = field(default_factory=dict)
    by_role_calls: dict[str, int] = field(default_factory=dict)


def _add_int(current: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return current
    return (current or 0) + value


def _add_float(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    return (current or 0.0) + value


def cost_facts(analysis: Analysis) -> CostFacts:
    """Sum item 15's cost/time facts from the rows and their session logs.

    Two passes over the same inputs: the manifest rows carry the money, the
    tokens and the grading wall, and the session's own ``call.usage`` rows carry
    the per-call money and the model wall. Nothing here reads a verdict beyond
    ``label_for`` (to recognise a CORRECT one) and nothing here writes anything
    back — the same record-only discipline as the rest of this script.
    """
    facts = CostFacts()
    for _batch, record in analysis.rows:
        price = _row_price(record)
        llm = record.get("llm") if isinstance(record.get("llm"), dict) else {}
        arm = arms_mod.row_arm(record)

        if price is not None:
            facts.by_arm_cost[arm] = facts.by_arm_cost.get(arm, 0.0) + price
        if str(record.get("verdict") or ""):
            facts.by_arm_verdicts[arm] = facts.by_arm_verdicts.get(arm, 0) + 1

        # §5.3's two CORRECT cells: an approval that should have happened and a
        # rejection that should have. A refusal (NR) and a mis-decision (FA/FR)
        # are outside the denominator, exactly as they are everywhere else.
        cell = LABEL_TO_CELL.get(label_for(record))
        if cell in (CELL_TA, CELL_TR):
            facts.correct_count[arm] = facts.correct_count.get(arm, 0) + 1
            if price is not None:
                facts.correct_cost[arm] = facts.correct_cost.get(arm, 0.0) + price

        # The grading sandbox's own wall, recorded per row. The in-session run's
        # wall (`timings.run_s`) mixes model wait with sandbox work, so it is not
        # used as "sandbox" here; the oracle run is the sandbox-only figure.
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        facts.sandbox_s = _add_float(facts.sandbox_s, _number_or_none(timings.get("oracle_s")))

        prompt = _int_or_none(llm.get("prompt_tokens"))
        cached = _int_or_none(llm.get("cached_tokens"))
        completion = _int_or_none(llm.get("completion_tokens_total"))
        facts.prompt_tokens = _add_int(facts.prompt_tokens, prompt)
        facts.cached_tokens = _add_int(facts.cached_tokens, cached)
        facts.completion_tokens = _add_int(facts.completion_tokens, completion)

        # A repair row whose patch is good and whose suite did not regress — the
        # defect it was asked to fix stayed fixed. Restricted to repair rows for
        # the same reason `repair_counts` is: a review row grades a candidate, it
        # does not fix one.
        oracle = row_oracle(record)
        if (
            str(record.get("mode") or "") == arms_mod.REPAIR
            and oracle is not None
            and oracle.patch_good
            and not oracle.suite_regressions
        ):
            facts.fixed_count += 1
            facts.fixed_cost = _add_float(facts.fixed_cost, price)

    _add_call_usage(analysis, facts)
    return facts


def _add_call_usage(analysis: Analysis, facts: CostFacts) -> None:
    """Fold the session's per-call ``call.usage`` rows into ``facts``.

    ``source=estimate`` rows are the pipeline's own character-derived guesses,
    not provider measurements, so they are excluded from the money and the wall
    — the same rule the pipeline applies when it sums its token totals.
    """
    for _batch, record in analysis.rows:
        for row in analysis.session_rows(record):
            if str(row.get("phase") or "") != CALL_USAGE_PHASE:
                continue
            summary = str(row.get("summary") or "")
            if (metrics_mod.summary_field(summary, "source") or "") == "estimate":
                continue
            role = role_for_site(metrics_mod.summary_field(summary, "site") or "?")
            facts.by_role_calls[role] = facts.by_role_calls.get(role, 0) + 1
            cost = _float_field(summary, "cost_usd")
            if cost is not None:
                facts.by_role_cost[role] = facts.by_role_cost.get(role, 0.0) + cost
            wall = _float_field(summary, "wall_ms")
            facts.model_ms = _add_int(
                facts.model_ms, None if wall is None else int(wall)
            )


def role_for_site(site: str) -> str:
    """The reportable role of a call site (see :data:`SITE_ROLE`)."""
    return SITE_ROLE.get(site, site)


def cost_per_fixed_defect(analysis: Analysis) -> str:
    """§5's "cost per safely fixed defect", or ``unavailable(...)``.

    The numerator is the priced spend of the rows that fixed a defect and did
    not regress; the denominator is how many those were. Neither a missing
    denominator nor an unpriced numerator is a zero.
    """
    facts = cost_facts(analysis)
    if facts.fixed_count == 0:
        return unavailable("no safely fixed defect row")
    if facts.fixed_cost is None:
        return unavailable("no safely fixed defect row was priced")
    return f"{_money(facts.fixed_cost / facts.fixed_count)} (n={facts.fixed_count})"


def _cost_by_arm(facts: CostFacts) -> str:
    parts: list[str] = []
    for arm in sorted(facts.by_arm_cost):
        count = facts.by_arm_verdicts.get(arm, 0)
        if count <= 0:
            continue
        parts.append(f"{arm} {_money(facts.by_arm_cost[arm] / count)}/verdict (n={count})")
    if not parts:
        return unavailable("no arm had both a verdict and a price")
    return " ".join(parts)


def _cost_per_correct_verdict(facts: CostFacts) -> str:
    counted = [arm for arm in ("b3", "judge") if facts.correct_count.get(arm, 0) > 0]
    if not counted:
        return unavailable("no correct verdict to price")
    if not facts.correct_cost:
        # The denominator exists but no row was priced — a local (`not_billed`)
        # run. That is "not measured", not "free", so it is unavailable, never
        # a per-arm `n/a` that reads like a figure someone could have made.
        return unavailable("no correct verdict was priced")
    parts: list[str] = []
    for arm in ("b3", "judge"):
        count = facts.correct_count.get(arm, 0)
        if count <= 0 or arm not in facts.correct_cost:
            parts.append(f"{arm} n/a")
            continue
        parts.append(f"{arm} {_money(facts.correct_cost[arm] / count)} (n={count})")
    return " ".join(parts)


def _cost_per_call_by_role(facts: CostFacts) -> str:
    parts: list[str] = []
    for role in sorted(facts.by_role_cost):
        calls = facts.by_role_calls.get(role, 0)
        if calls <= 0:
            continue
        parts.append(f"{role} {_money(facts.by_role_cost[role] / calls)}/call (n={calls})")
    if not parts:
        return unavailable("no call.usage row was priced")
    return " ".join(parts)


def _tokens_text(facts: CostFacts) -> str:
    if (
        facts.prompt_tokens is None
        and facts.cached_tokens is None
        and facts.completion_tokens is None
    ):
        return unavailable("no row reported tokens")

    def side(value: Optional[int]) -> str:
        return "n/a" if value is None else str(value)

    return f"{side(facts.prompt_tokens)} / {side(facts.cached_tokens)} / {side(facts.completion_tokens)}"


def _cache_hit_rate_text(facts: CostFacts) -> str:
    if facts.prompt_tokens is None or facts.prompt_tokens == 0:
        return unavailable("no prompt tokens were measured")
    if facts.cached_tokens is None:
        return unavailable("no cached-token count was reported")
    return f"{facts.cached_tokens / facts.prompt_tokens:.4f}"


def _wall_clock_text(facts: CostFacts) -> str:
    if facts.model_ms is None and facts.sandbox_s is None:
        return unavailable("no wall clock was recorded")
    model = "n/a" if facts.model_ms is None else f"{facts.model_ms / 1000.0:.3f}s"
    sandbox = "n/a" if facts.sandbox_s is None else f"{facts.sandbox_s:.3f}s"
    return f"model {model} / sandbox {sandbox}"


class CostTimeMetrics(Metric):
    name = "cost_time"

    def run(self, analysis: Analysis) -> MetricResult:
        facts = cost_facts(analysis)
        rows = [
            ("cost per call, by role", _cost_per_call_by_role(facts)),
            ("cost per verdict, by arm", _cost_by_arm(facts)),
            ("cost per correct verdict (B3 vs judge)", _cost_per_correct_verdict(facts)),
            ("cost per safely fixed defect", cost_per_fixed_defect(analysis)),
            ("tokens: input / cached / output", _tokens_text(facts)),
            ("cache hit rate", _cache_hit_rate_text(facts)),
            ("wall clock: model / sandbox", _wall_clock_text(facts)),
        ]
        tex = (
            r"% item 12 — cost and time. Every figure is summed from the pipeline's"
            "\n"
            r"% own priced records (`llm.cost_usd` and the `call.usage` rows); a"
            "\n"
            r"% figure no row supports is marked unavailable, never `0`."
            "\n"
            r"\begin{tabular}{ll}"
            "\n"
            r"\hline"
            "\n"
            + "\n".join(f"{_tex(name)} & {_tex(value)} \\\\" for name, value in rows)
            + "\n"
            + r"\hline"
            + "\n"
            + r"\end{tabular}"
            + "\n"
        )
        summary = ["", "== Cost and time =="] + [f"  {name:<40} {value}" for name, value in rows]
        return MetricResult(name=self.name, artifacts=[Artifact("cost_time.tex", tex)], summary=summary)


# ---------------------------------------------------------------------------
# the registry and the writer
# ---------------------------------------------------------------------------
METRICS: tuple[Metric, ...] = (
    ReviewMetrics(),
    PairedMetrics(),
    RepairMetrics(),
    LocalizationMetrics(),
    EirEcrMetrics(),
    PipelineMetrics(),
    CostTimeMetrics(),
)


def render_summary(analysis: Analysis, results: Sequence[MetricResult]) -> str:
    """The plain-text summary: the input batches, then every metric's lines."""
    lines = ["item 12 — analysis summary", "", "input batches:"]
    for batch in analysis.batches:
        lines.append(f"  {batch.path}  ({len(batch.records)} row(s), {len(batch.replays)} replay(s))")
    # Item B6 — which rule read each row. Rows already on disk keep item Q's reading
    # and are tagged, so a mixed load is visible here rather than silently pooled.
    rules = label_rule_counts(analysis)
    if rules:
        lines.append("")
        lines.append("label rules:")
        for rule in sorted(rules):
            lines.append(f"  {rule:<20} {rules[rule]}")
    for result in results:
        lines.extend(result.summary)
    lines.append("")
    return "\n".join(lines)


def run_analysis(analysis: Analysis) -> dict[str, str]:
    """Every output file's text, keyed by filename — the deterministic core."""
    files: dict[str, str] = {}
    results: list[MetricResult] = []
    for metric in METRICS:
        result = metric.run(analysis)
        results.append(result)
        for artifact in result.artifacts:
            files[artifact.filename] = artifact.text
    files["summary.txt"] = render_summary(analysis, results)
    return files


def write_outputs(files: Mapping[str, str], out_dir: Path) -> list[Path]:
    """Write every file under ``out_dir`` and return the paths, sorted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename in sorted(files):
        path = out_dir / filename
        path.write_text(files[filename], encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse.py",
        description=(
            "Item 12 — write every number in the paper from the batches' own "
            "manifest and replay records."
        ),
    )
    parser.add_argument(
        "--batches",
        action="append",
        required=True,
        metavar="GLOB",
        help="a batch glob (repeatable); each match must hold a manifest.jsonl",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"bootstrap seed (default {DEFAULT_SEED}); the intervals are a pure function of it",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=f"cluster-bootstrap resamples (default {DEFAULT_BOOTSTRAP})",
    )
    parser.add_argument(
        "--allow-mixed-configs",
        action="store_true",
        help=(
            "pool rows whose config_hash values differ (item 19). Off by default: rows "
            "produced under different frozen configurations are never pooled "
            "(§Versioning), and two distinct hashes in one load is a refusal rather than "
            "a warning. Pass this only to look at a mixed load on purpose."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = load_analysis(
            args.batches,
            seed=args.seed,
            resamples=args.bootstrap,
            allow_mixed_configs=args.allow_mixed_configs,
        )
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = run_analysis(analysis)
    for path in write_outputs(files, Path(args.out)):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
