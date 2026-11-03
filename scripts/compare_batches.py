#!/usr/bin/env python3
"""compare_batches.py — the pandas/1 equivalence check for M3 items S5 and S6.

WHY THIS EXISTS
---------------
S5 and S6 make the harness run suite runs and rows concurrently, and Nathan's
gate before either may be used for real is an equivalence check on one bug:
*"a serial prepare and run on the same build as the reference, then the parallel
version, compared per nodeid apart from the tests the serial pair marked flaky"*
(``plans/private/M3-progress.md``, "The gate before real use"). This script is
that comparison. It reads two batch directories produced by the *same build* — one
prepared and run with ``--prepare-jobs 1 --jobs 1``, one with the parallel
settings — and reports every difference that concurrency could have caused.

WHAT IS COMPARABLE, AND WHAT IS NOT
-----------------------------------
The two batches are two independent ``run`` invocations, so the model answered
twice. The parts of a batch that are a fact about the *tree and the scheduling*
are therefore comparable, and the parts the model produced are not:

  * **Comparable** (a difference here is a failure): the per-nodeid native sets
    ``prepare.json`` records (``fail_to_pass``, ``flaky_nodeids``,
    ``preexisting_failures``, ``native_failure_baseline``, ``collect_flip_nodeids``,
    ``f2p_undeclared``) and the native scalars beside them; whether a row exists at
    all for a job (``bug`` × ``config`` × ``mode`` × ``seed``); and every row's
    **time-limit behaviour** — ``timed_out``, ``stop_reason`` and
    ``native_timeout``, plus the ``scheduler`` block's wall time, deadline left and
    provider retries. This last group is the S6 half of the gate, stated by Nathan
    as *"Concurrency must not change outcomes through time limits ... If any does,
    the recommended --jobs is too high for this Mac."*
  * **Not comparable** (expected to differ): the patches the model wrote, and
    therefore the labels and oracle gradings derived from them. A row's identity
    includes its candidate patch's SHA-256 for exactly this reason. Two rows whose
    candidate bytes differ are reported in ``rows.unmatched_candidates`` with both
    labels side by side, and a *difference* there is a statement about the model,
    not about the scheduler.

The one place the two meet is a row whose candidate **bytes match**: then the
grading is a measurement of the same patch, so its **oracle** block must be
identical and a difference there fails the gate. Its **labels** are a different
matter (see below).

WHY A MATCHED ROW'S LABEL IS REPORTED AND NOT GRADED
----------------------------------------------------
Nathan's rule (b), overruled in part: *"for matched candidates the oracle result
must match (deterministic), but a label difference is reported side by side and
does not fail, because the Attacker writes new tests each run."* The oracle block
is the harness's own measurement of a fixed patch against a fixed hidden suite —
deterministic, so a difference is a finding. A label is produced from the model's
own report of what it changed, and the attack arm writes fresh tests every run, so
two runs of the same patch bytes can legitimately carry different labels. Those
differences are reported in ``rows.matched[*].label_differences`` and never fail
the gate. Rules (a), (c), (d) and (e) are unchanged.

CACHE ISOLATION
---------------
The two batches run on the same build and the same trees, so on ONE suite-cache
root the parallel batch would be served the serial batch's prepare suites and
in-session native gates — and the comparison would pass vacuously. Each batch is
therefore given its own root (``VANTA_SUITE_CACHE_DIR``, which drives both the
Python and the Rust caches) and this script proves it: it reads the cache root
each batch recorded (``prepare.json``'s per-tree ``scheduler.runs[*].cache.dir``)
and **fails** when the two batches name the same root, or when any recorded hit's
key is not an entry under its own batch's root. It reports each batch's
prepare-suite and native-gate hit/miss counts and the writer of every hit.

VERDICT
-------
``identical`` requires all five: no prepare-side nodeid difference, no matched-row
oracle difference, no group present in only one batch, no time-limit difference
(neither a matched row whose flags moved nor a group that hit a limit in the
parallel batch where the serial batch did not), and no cache-isolation problem.
It additionally requires that the comparison was not vacuous: at least one bug was
wanted and at least one row existed across the two batches. Two batches that
compared nothing (no ``prepare.json`` bugs, no rows) are ``not_identical``, never
``identical`` — ``all([])`` is ``True``, and a gate must not pass on absence.

USAGE (run from the repository root)
------------------------------------
    python3 scripts/compare_batches.py \
        --serial runs/bugsinpy/m3-eq-pandas_1-serial \
        --parallel runs/bugsinpy/m3-eq-pandas_1-parallel

    # only one bug, and an explicit report path:
    python3 scripts/compare_batches.py --serial A --parallel B --bug pandas/1 \
        --report /tmp/compare.json

OUTPUT
------
One verdict line, and a JSON report at ``<parallel>/compare_batches.json`` unless
``--report`` names one (``--no-report`` writes none). Exit codes: ``0`` identical,
``1`` not identical, ``2`` the two batches could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import manifest as manifest_mod  # noqa: E402
from bugsinpy import suite_cache as suite_cache_mod  # noqa: E402

# The nodeid map diff is S4's own primitive (`probe_xdist.outcome_diff`): the rule
# "two nodeid->outcome maps, minus a known-flaky set" already exists and is already
# tested, and a second copy of it here is exactly the drift this project's comments
# keep warning about. Imported, never re-implemented.
from probe_xdist import outcome_diff  # noqa: E402

SCHEMA_VERSION = 1

PREPARE_NAME = "prepare.json"

PREPARE_NODEID_FIELDS = (
    "fail_to_pass",
    "f2p_undeclared",
    "flaky_nodeids",
    "preexisting_failures",
    "native_failure_baseline",
    "collect_flip_nodeids",
)
"""The per-nodeid sets ``prepare.json`` records per bug — the S5 half of the gate."""

PREPARE_SCALAR_FIELDS = (
    "native_skips_buggy",
    "native_skips_fixed",
    "pass_to_pass_count",
    "native_green_on_vanta_tree",
    "prepare_mode",
    "vm_memory_mb",
)
"""The scalars beside them that are a fact about the tree, not about the clock."""

PREPARE_TIMING_FIELDS = ("native_secs",)
"""Measured, and expected to differ under load — recorded, never a failure."""

TIME_LIMIT_FLAGS = ("timed_out", "stop_reason", "native_timeout")
"""A row's time-limit behaviour. A difference in any of these fails the gate.

``timed_out`` is the harness killing the process; ``stop_reason`` is Vanta's own
deadline firing (``budget_exhausted``); ``native_timeout`` is the native gate
hitting its cap. All three are wall-clock, so all three can move under load.
"""

ORACLE_EXACT_FIELDS = (
    "apply_ok",
    "apply_outcome",
    "hidden_pass",
    "patch_good",
    "inconclusive",
    "suite_regressions_count",
)
"""The oracle's verdict-shaped fields. Compared only for a matched candidate."""

CACHE_HIT_STATE = "hit"
CACHE_MISS_STATE = "miss"
"""``suite.SuitePair.cache["state"]``, and the native gate's state, that mean hit/miss.

``hit`` is the one state a run was served by an entry; ``miss`` measured fresh. The
rest (``off:no_cache``, ``off:unknown_checkpoint``, ``unreadable``, ``unmeasured``)
are counted separately rather than folded into ``miss``: each is a different reason
the cache was not consulted, and the isolation check is about which state the two
batches reached, not about how many entries they kept.
"""

SESSIONS_DIRNAME = "vanta_sessions"
"""Where ``driver.rs`` writes ``<session>_evidence.json``, relative to the repo root.

[VANTA-WHY] The in-session native gate's cache records are NOT on the row: the row
carries only their count (``native_cache_hits``, on the ``run.summary`` log line),
and the records themselves ride ``<sid>_evidence.json``. Reading that file is the
only way to audit the Rust half of the cache from a batch — and the Rust half is
what a *review* row's native gate uses, so it is the half most likely to be served
from the other batch's directory if the two were left sharing one.
"""


class CompareError(RuntimeError):
    """A batch could not be read as a batch."""


# ---------------------------------------------------------------------------
# reading a batch
# ---------------------------------------------------------------------------
def read_prepare(batch_dir: Path) -> dict[str, Any]:
    """``prepare.json`` as a dict, or a :class:`CompareError` naming the batch."""
    path = Path(batch_dir) / PREPARE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompareError(f"{batch_dir} has no readable {PREPARE_NAME}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CompareError(f"{batch_dir}/{PREPARE_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CompareError(f"{batch_dir}/{PREPARE_NAME} is not an object")
    return payload


def read_rows(batch_dir: Path) -> list[dict[str, Any]]:
    """The batch's **current** rows (``manifest.effective_records``), never the raw list.

    The raw ``manifest.jsonl`` is append-only, so a resumed or re-run job appears
    twice; comparing raw rows would report a re-run as a difference. The filter is
    the same one every other reader uses (:func:`manifest.effective_records`), so
    this script and the digest can never disagree about which rows exist.
    """
    return manifest_mod.Manifest(Path(batch_dir)).effective_records()


def prepare_bugs(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``prepare.json``'s per-bug blocks keyed by the bug key it records."""
    out: dict[str, dict[str, Any]] = {}
    for block in payload.get("bugs") or []:
        if isinstance(block, dict) and block.get("bug"):
            out[str(block["bug"])] = block
    return out


# ---------------------------------------------------------------------------
# reading a batch's cache traffic
# ---------------------------------------------------------------------------
def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def prepare_suite_cache(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every prepare suite's recorded cache state — one record per bug × tree.

    Written by ``prepare._scheduler_record`` from ``suite.SuitePair.cache`` into
    ``prepare.json``'s per-bug ``scheduler.runs[<tree>].cache``. ``dir`` is the
    cache ROOT the measurement came from, which is the whole point: two batches on
    one root would be served each other's results.
    """
    out: list[dict[str, Any]] = []
    for bug, block in sorted(prepare_bugs(payload).items()):
        scheduler = block.get("scheduler")
        runs = scheduler.get("runs") if isinstance(scheduler, Mapping) else None
        if not isinstance(runs, Mapping):
            continue
        for tree, record in sorted(runs.items()):
            cache = record.get("cache") if isinstance(record, Mapping) else None
            if not isinstance(cache, Mapping):
                continue
            out.append(
                {
                    "bug": bug,
                    "where": str(tree),
                    "key": str(cache.get("key") or ""),
                    "state": str(cache.get("state") or ""),
                    "source": str(cache.get("source") or ""),
                    "dir": str(cache.get("dir") or ""),
                    "would_hit": bool(cache.get("would_hit")),
                }
            )
    return out


def evidence_path(session_dir: str) -> Path:
    """``vanta_sessions/<sid>`` → the sibling ``<sid>_evidence.json``.

    ``session_dir`` is repo-relative on a real row and absolute in a test; an
    absolute one is used as given so a batch built under ``tmp_path`` is readable.
    """
    directory = Path(session_dir)
    root = directory if directory.is_absolute() else REPO_ROOT / directory
    return root.parent / f"{directory.name}_evidence.json"


def native_cache_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every in-session native-gate cache record this batch's rows left behind.

    One record per attempt × side (``original``/``patched``), read from the
    session's evidence file. A row with no evidence file (never booted a native
    gate, or a cleaned ``vanta_sessions``) contributes nothing rather than raising:
    the absence shows up as a smaller count, and the isolation check runs on what
    *is* recorded. Direction is one-way on purpose — a missing file must not read
    as "isolated", because it is not evidence of isolation either way.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), Mapping) else None
        session_dir = str((artifacts or {}).get("session_dir") or "")
        if not session_dir:
            continue
        try:
            payload = json.loads(evidence_path(session_dir).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("native_cache") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            continue
        session = Path(session_dir).name
        for record in records:
            if not isinstance(record, Mapping):
                continue
            out.append(
                {
                    "bug": str(row.get("bug") or ""),
                    "where": f"{session}/a{record.get('attempt')}/{record.get('side')}",
                    "key": str(record.get("key") or ""),
                    "state": str(record.get("state") or ""),
                    "source": str(record.get("source") or ""),
                    "dir": "",
                    "would_hit": bool(record.get("would_hit")),
                }
            )
    return out


def cache_dirs(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The cache roots these records name, resolved, deduplicated and sorted."""
    found = {str(Path(str(r["dir"])).resolve()) for r in records if r.get("dir")}
    return sorted(found)


def cache_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One batch's cache traffic: the counts, the states, and who wrote each hit."""
    hits = [r for r in records if r["state"] == CACHE_HIT_STATE]
    misses = [r for r in records if r["state"] == CACHE_MISS_STATE]
    sources: dict[str, int] = {}
    for record in hits:
        name = str(record["source"] or "(unrecorded)")
        sources[name] = sources.get(name, 0) + 1
    return {
        "total": len(records),
        "hits": len(hits),
        "misses": len(misses),
        "other": len(records) - len(hits) - len(misses),
        "states": _counts(str(r["state"]) for r in records),
        "hit_sources": dict(sorted(sources.items())),
        "hit_keys": sorted({str(r["key"]) for r in hits if r["key"]}),
    }


# ---------------------------------------------------------------------------
# row identity
# ---------------------------------------------------------------------------
def candidate_sha(record: Mapping[str, Any]) -> str:
    """The reviewed patch's SHA-256, or ``""`` for a repair row.

    The candidate is a *dict* on the row (:meth:`runner.Candidate.as_dict`), and its
    ``patch_sha256`` is content-derived, so it is the same string in two batches iff
    the two reviewed the same bytes.
    """
    candidate = record.get("candidate")
    if isinstance(candidate, Mapping):
        return str(candidate.get("patch_sha256") or "")
    return ""


def row_identity(record: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    """``(bug, config, mode, seed, candidate_sha)`` — the job key minus ``batch_id``.

    ``manifest.job_key`` cannot be used directly: it embeds ``batch_id``, which is a
    property of the batch *directory*, so the two batches' keys can never be equal.
    Every other component of the key is kept, and kept in the same order.
    """
    return (
        str(record.get("bug") or ""),
        str(record.get("config") or ""),
        str(record.get("mode") or ""),
        int(record.get("seed") or 0),
        candidate_sha(record),
    )


def group_identity(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """``(bug, config, mode, seed)`` — one row-identity with the candidate dropped."""
    return row_identity(record)[:4]


def _label_of(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": record.get("label"),
        "label_rule": record.get("label_rule"),
        "label_reason": record.get("label_reason"),
        "verdict": record.get("verdict"),
    }


def _time_limits(record: Mapping[str, Any]) -> dict[str, Any]:
    """A row's time-limit behaviour, plus the calibration numbers beside it."""
    scheduler = record.get("scheduler") if isinstance(record.get("scheduler"), Mapping) else {}
    out = {
        "timed_out": record.get("timed_out"),
        "stop_reason": record.get("stop_reason"),
        "native_timeout": record.get("native_timeout"),
        "native_timeout_secs": record.get("native_timeout_secs"),
        "deadline_secs": record.get("deadline_secs"),
        "wall_s": (record.get("timings") or {}).get("run_s"),
    }
    # The scheduler block exists only on a row a concurrent invocation wrote; a
    # `--jobs 1` row has none, and `None` here means "not recorded", never zero.
    out["deadline_remaining_s"] = scheduler.get("deadline_remaining_s")
    out["peak_sandboxes"] = scheduler.get("peak_sandboxes")
    out["provider_retries"] = scheduler.get("provider_retries")
    out["provider_retries_rate_limit"] = scheduler.get("provider_retries_rate_limit")
    out["provider_retries_unavailable"] = scheduler.get("provider_retries_unavailable")
    return out


NO_STOP_REASON = (None, "", "unset", "none")
"""``stop_reason`` when nothing stopped the run.

``driver.rs`` writes ``stop_reason={}`` with ``stop_reason.unwrap_or("none")``, so a
row that reached its own verdict carries the literal string ``"none"`` — *not*
``None``. Verified against a real batch: every ordinary row reads ``'none'``. The
gate must treat that as "no limit hit", or every row would look like a timeout.
"""


def _hit_a_time_limit(record: Mapping[str, Any]) -> bool:
    """Whether this row's own time limits moved it — the S6 gate's unit."""
    if record.get("timed_out") is True or record.get("native_timeout") is True:
        return True
    return record.get("stop_reason") not in NO_STOP_REASON


def compare_oracle(
    serial: Mapping[str, Any], parallel: Mapping[str, Any], excluded: Iterable[str]
) -> dict[str, Any]:
    """Both rows' oracle blocks: the verdict fields, and the nodeid map diff."""
    left = serial.get("oracle") if isinstance(serial.get("oracle"), Mapping) else {}
    right = parallel.get("oracle") if isinstance(parallel.get("oracle"), Mapping) else {}
    exact = {
        name: [left.get(name), right.get(name)]
        for name in ORACLE_EXACT_FIELDS
        if left.get(name) != right.get(name)
    }
    left_map = left.get("hidden_outcomes") if isinstance(left.get("hidden_outcomes"), Mapping) else {}
    right_map = (
        right.get("hidden_outcomes") if isinstance(right.get("hidden_outcomes"), Mapping) else {}
    )
    nodeids = outcome_diff(left_map, right_map, excluded)
    return {
        "exact_differences": exact,
        "hidden_outcomes": nodeids,
        "identical": not exact and not nodeids["only_in_serial"]
        and not nodeids["only_in_xdist"] and not nodeids["different_outcome"],
    }


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def compare_prepare(
    serial: Mapping[str, Any], parallel: Mapping[str, Any], bugs: Sequence[str]
) -> dict[str, Any]:
    """The S5 half: are the two preparations' nodeid-level facts the same?"""
    left = prepare_bugs(serial)
    right = prepare_bugs(parallel)
    per_bug: dict[str, Any] = {}
    for bug in bugs:
        a = left.get(bug)
        b = right.get(bug)
        if a is None or b is None:
            per_bug[bug] = {
                "identical": False,
                "reason": f"prepared in only one batch: "
                f"serial={a is not None} parallel={b is not None}",
            }
            continue
        nodeids: dict[str, Any] = {}
        for field in PREPARE_NODEID_FIELDS:
            only_serial = sorted(set(a.get(field) or []) - set(b.get(field) or []))
            only_parallel = sorted(set(b.get(field) or []) - set(a.get(field) or []))
            if only_serial or only_parallel:
                nodeids[field] = {"only_in_serial": only_serial, "only_in_parallel": only_parallel}
        scalars = {
            field: [a.get(field), b.get(field)]
            for field in PREPARE_SCALAR_FIELDS
            if a.get(field) != b.get(field)
        }
        timing = {
            field: [a.get(field), b.get(field)]
            for field in PREPARE_TIMING_FIELDS
            if a.get(field) != b.get(field)
        }
        per_bug[bug] = {
            "nodeid_differences": nodeids,
            "scalar_differences": scalars,
            "timing": timing,
            "identical": not nodeids and not scalars,
        }
    return {"bugs": per_bug, "identical": all(b["identical"] for b in per_bug.values())}


def compare_rows(
    serial_rows: Sequence[dict[str, Any]],
    parallel_rows: Sequence[dict[str, Any]],
    excluded_by_bug: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """The S6 half: per-row presence, matched-candidate identity, and time limits."""
    left = {row_identity(r): r for r in serial_rows}
    right = {row_identity(r): r for r in parallel_rows}

    matched: list[dict[str, Any]] = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        bug = key[0]
        # [VANTA-WHY] Nathan's rule (b), overruled in part: the ORACLE must match
        # (it is the harness's deterministic measurement of these exact bytes),
        # but the LABEL is the model's own account of what it changed, and the
        # attack arm writes new tests each run — so two runs of one patch can
        # legitimately be labelled differently. Recorded side by side, never
        # graded. Rules (a), (c), (d) and (e) are untouched.
        labels = {
            name: [value_a, value_b]
            for name, value_a, value_b in (
                ("label", a.get("label"), b.get("label")),
                ("label_rule", a.get("label_rule"), b.get("label_rule")),
                ("label_reason", a.get("label_reason"), b.get("label_reason")),
                ("verdict", a.get("verdict"), b.get("verdict")),
            )
            if value_a != value_b
        }
        oracle = compare_oracle(a, b, excluded_by_bug.get(bug, ()))
        limits = {
            name: [_time_limits(a)[name], _time_limits(b)[name]]
            for name in TIME_LIMIT_FLAGS
            if _time_limits(a)[name] != _time_limits(b)[name]
        }
        matched.append(
            {
                "identity": list(key),
                "label_differences": labels,
                "oracle": oracle,
                "time_limit_differences": limits,
                "identical": oracle["identical"] and not limits,
            }
        )

    # A row present in one batch and not the other is only a finding when its GROUP
    # is missing: review candidates are expected to differ (the model answered
    # twice), so a review row whose candidate bytes are unique to one batch is
    # `unmatched_candidates`, not a missing row.
    left_groups = {group_identity(r) for r in serial_rows}
    right_groups = {group_identity(r) for r in parallel_rows}
    missing = [
        {"identity": list(group), "present_in": "serial"}
        for group in sorted(left_groups - right_groups)
    ] + [
        {"identity": list(group), "present_in": "parallel"}
        for group in sorted(right_groups - left_groups)
    ]

    only_left = [key for key in sorted(set(left) - set(right))]
    only_right = [key for key in sorted(set(right) - set(left))]
    unmatched = {
        "only_in_serial": [
            {"identity": list(key), **_label_of(left[key])}
            for key in only_left
            if group_identity(left[key]) in right_groups
        ],
        "only_in_parallel": [
            {"identity": list(key), **_label_of(right[key])}
            for key in only_right
            if group_identity(right[key]) in left_groups
        ],
    }

    # Time-limit behaviour at the group level: a group that hit a limit in the
    # parallel batch where the serial batch did not is the finding Nathan named.
    def hits_by_group(rows: Sequence[dict[str, Any]]) -> dict[tuple, int]:
        out: dict[tuple, int] = {}
        for row in rows:
            if _hit_a_time_limit(row):
                out[group_identity(row)] = out.get(group_identity(row), 0) + 1
        return out

    serial_hits = hits_by_group(serial_rows)
    parallel_hits = hits_by_group(parallel_rows)
    group_limit_differences = [
        {
            "identity": list(group),
            "serial_rows_hitting_a_limit": serial_hits.get(group, 0),
            "parallel_rows_hitting_a_limit": parallel_hits.get(group, 0),
        }
        for group in sorted(set(serial_hits) | set(parallel_hits))
        if serial_hits.get(group, 0) != parallel_hits.get(group, 0)
    ]

    return {
        "matched": matched,
        "missing_groups": missing,
        "unmatched_candidates": unmatched,
        "time_limit_differences": {
            "rows": [row for row in matched if row["time_limit_differences"]],
            "groups": group_limit_differences,
        },
        "identical": (
            all(row["identical"] for row in matched)
            and not missing
            and not group_limit_differences
        ),
    }


# ---------------------------------------------------------------------------
# cache isolation — the two batches must not have shared a root
# ---------------------------------------------------------------------------
def _unresolved_hits(
    label: str, dirs: Sequence[str], groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]]
) -> list[str]:
    """Every recorded hit whose key is not an entry under *this* batch's own roots.

    A hit is served from a directory by definition, so a key that resolves under no
    directory this batch named was served from one it did not name — the other
    batch's, which is the vacuous pass this whole check exists to stop.
    """
    problems: list[str] = []
    for kind, records in groups:
        for record in records:
            key = str(record["key"])
            if record["state"] != CACHE_HIT_STATE or not key:
                continue
            if any(suite_cache_mod.entry_path_in(root, key).is_file() for root in dirs):
                continue
            problems.append(
                f"{label} {kind} cache hit {key[:12]}… is not an entry under {label}'s own "
                f"cache directory ({', '.join(dirs)})"
            )
    return problems


def compare_cache(
    serial_prepare: Mapping[str, Any],
    parallel_prepare: Mapping[str, Any],
    serial_rows: Sequence[Mapping[str, Any]],
    parallel_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the two batches did not share a suite-cache root.

    Each batch's root is the one its prepare suites recorded; the in-session native
    gate uses the SAME root (one ``VANTA_SUITE_CACHE_DIR`` drives both caches), so a
    native hit is checked against it too. The check needs a recorded root to run at
    all, so ``checked`` says whether it did — it never fails for being un-runnable,
    because "no root recorded" is not evidence of a shared one, and a failure that
    fired on absence would fire on every batch that predates this item.
    """
    serial_suite = prepare_suite_cache(serial_prepare)
    parallel_suite = prepare_suite_cache(parallel_prepare)
    serial_native = native_cache_records(serial_rows)
    parallel_native = native_cache_records(parallel_rows)
    serial_dirs = cache_dirs(serial_suite)
    parallel_dirs = cache_dirs(parallel_suite)

    problems: list[str] = []
    checked = bool(serial_dirs and parallel_dirs)
    if checked:
        shared = sorted(set(serial_dirs) & set(parallel_dirs))
        if shared:
            problems.append(
                "both batches used the same suite cache directory, so neither computed "
                f"its own results: {', '.join(shared)}"
            )
        problems.extend(
            _unresolved_hits(
                "serial", serial_dirs, (("prepare", serial_suite), ("native", serial_native))
            )
        )
        problems.extend(
            _unresolved_hits(
                "parallel",
                parallel_dirs,
                (("prepare", parallel_suite), ("native", parallel_native)),
            )
        )

    return {
        "checked": checked,
        "problems": problems,
        "serial": {
            "dirs": serial_dirs,
            "prepare": cache_summary(serial_suite),
            "native": cache_summary(serial_native),
        },
        "parallel": {
            "dirs": parallel_dirs,
            "prepare": cache_summary(parallel_suite),
            "native": cache_summary(parallel_native),
        },
        "identical": not problems,
    }


def row_concurrency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The effective row concurrency a batch reached — what ``--jobs`` actually got.

    ``Budget.peak_cores`` is the high-water mark of rows holding the scheduler's
    slot and a row is charged exactly one core, so it IS the number of rows that ran
    at once. ``None`` means no row carried a ``scheduler`` block, which is what a
    ``--jobs 1`` batch looks like: concurrency is a fact about a budget, and a serial
    invocation writes no block to record one.
    """
    peaks = [
        int(row["scheduler"]["peak_cores"])
        for row in rows
        if isinstance(row.get("scheduler"), Mapping)
        and isinstance(row["scheduler"].get("peak_cores"), (int, float))
    ]
    return {
        "rows": len(rows),
        "rows_with_a_scheduler_block": len(peaks),
        "peak_cores": max(peaks) if peaks else None,
    }


def compare_batches(
    serial_dir: Path, parallel_dir: Path, *, bugs: Optional[Sequence[str]] = None
) -> dict[str, Any]:
    """Read both batches, compare them, and return the report payload."""
    serial_dir = Path(serial_dir)
    parallel_dir = Path(parallel_dir)
    if not serial_dir.is_dir():
        raise CompareError(f"--serial {serial_dir} is not a directory")
    if not parallel_dir.is_dir():
        raise CompareError(f"--parallel {parallel_dir} is not a directory")

    serial_prepare = read_prepare(serial_dir)
    parallel_prepare = read_prepare(parallel_dir)
    serial_rows = read_rows(serial_dir)
    parallel_rows = read_rows(parallel_dir)

    if bugs is None:
        wanted = sorted(set(prepare_bugs(serial_prepare)) | set(prepare_bugs(parallel_prepare)))
    else:
        wanted = list(bugs)
        # [VANTA-WHY] M3 item V2 — `--bug` must scope the ROWS half too. It scoped
        # only `compare_prepare`, so `--bug pandas/1` still compared every other
        # bug's rows and could answer `not_identical` for a difference in a bug the
        # operator did not ask about.
        in_scope = set(wanted)
        serial_rows = [row for row in serial_rows if str(row.get("bug") or "") in in_scope]
        parallel_rows = [row for row in parallel_rows if str(row.get("bug") or "") in in_scope]

    # The flaky set is the SERIAL batch's, per the spec ("apart from the tests the
    # serial pair marked flaky"), unioned with the parallel batch's so a nodeid
    # flaky on either side is dropped from both maps.
    def flaky(batch_dir: Path) -> dict[str, set[str]]:
        payload = read_prepare(batch_dir)
        return {
            bug: set(block.get("flaky_nodeids") or [])
            for bug, block in prepare_bugs(payload).items()
        }

    serial_flaky = flaky(serial_dir)
    parallel_flaky = flaky(parallel_dir)
    excluded_by_bug = {
        bug: serial_flaky.get(bug, set()) | parallel_flaky.get(bug, set()) for bug in wanted
    }

    prepare = compare_prepare(serial_prepare, parallel_prepare, wanted)
    rows = compare_rows(serial_rows, parallel_rows, excluded_by_bug)
    cache = compare_cache(serial_prepare, parallel_prepare, serial_rows, parallel_rows)

    # [VANTA-WHY] M3 item V2 — "identical" must mean "compared and equal", never
    # "compared nothing". Every half folds with `all(...)`, and `all([])` is True,
    # so two batches with no prepared bugs and no rows (a mistyped directory, a
    # batch that never ran, a prepare that died before its first bug) used to read
    # as `identical` — the one answer a gate must never give vacuously. Both halves
    # must have had material to compare: at least one bug wanted, and at least one
    # row across the two batches to match. A run whose review candidates all differ
    # is unaffected: it has rows, and their disagreement is `unmatched_candidates`.
    compared = {
        "bugs": len(wanted),
        "rows": len(serial_rows) + len(parallel_rows),
        "matched_rows": len(rows["matched"]),
        "ran": bool(wanted) and bool(serial_rows or parallel_rows),
    }
    identical = bool(
        compared["ran"] and prepare["identical"] and rows["identical"] and cache["identical"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "serial": {
            "batch_dir": str(serial_dir),
            "rows": len(serial_rows),
            "scheduler": serial_prepare.get("scheduler") or {},
            "concurrency": row_concurrency(serial_rows),
        },
        "parallel": {
            "batch_dir": str(parallel_dir),
            "rows": len(parallel_rows),
            "scheduler": parallel_prepare.get("scheduler") or {},
            "host_samples": _host_samples(parallel_dir),
            "concurrency": row_concurrency(parallel_rows),
        },
        "bugs": wanted,
        "compared": compared,
        "prepare": prepare,
        "cache": cache,
        "rows": rows,
        "verdict": "identical" if identical else "not_identical",
    }


def _host_samples(batch_dir: Path) -> Optional[dict[str, Any]]:
    """The batch's own host sampling, when ``run_digest.json`` recorded any."""
    try:
        payload = json.loads((Path(batch_dir) / "run_digest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = payload.get("host_samples") if isinstance(payload, Mapping) else None
    return block if isinstance(block, Mapping) else None


def verdict_line(report: Mapping[str, Any], *, cap: int = 8) -> str:
    """The one line an operator reads. Pure, so its shape is testable."""
    bugs = ", ".join(report.get("bugs") or []) or "-"
    head = f"compare-batches {bugs}: {report['verdict']}"
    prepare_bugs_report = (report.get("prepare") or {}).get("bugs") or {}
    prepare_diffs = sorted(b for b, block in prepare_bugs_report.items() if not block["identical"])
    rows = report.get("rows") or {}
    matched = rows.get("matched") or []
    row_diffs = [row for row in matched if not row["identical"]]
    label_diffs = [row for row in matched if row.get("label_differences")]
    limits = (rows.get("time_limit_differences") or {}).get("groups") or []
    missing = rows.get("missing_groups") or []
    unmatched = rows.get("unmatched_candidates") or {}
    n_unmatched = len(unmatched.get("only_in_serial") or []) + len(
        unmatched.get("only_in_parallel") or []
    )
    cache_problems = (report.get("cache") or {}).get("problems") or []
    parts = [
        f"prepare_diffs={len(prepare_diffs)}",
        f"matched={len(matched)}",
        f"row_diffs={len(row_diffs)}",
        f"label_diffs={len(label_diffs)}",
        f"time_limit_diffs={len(limits)}",
        f"missing_groups={len(missing)}",
        f"unmatched_candidates={n_unmatched}",
        f"cache_problems={len(cache_problems)}",
    ]
    line = f"{head} ({', '.join(parts)})"
    named: list[str] = []
    for bug in prepare_diffs:
        # A bug prepared in only one batch has `reason`, not the difference maps;
        # `.get` keeps the line printable instead of raising on exactly the case
        # that most needs naming.
        block = prepare_bugs_report[bug]
        for field in (block.get("nodeid_differences") or {}):
            named.append(f"{bug}:{field}")
        for field in (block.get("scalar_differences") or {}):
            named.append(f"{bug}:{field}")
        if not block.get("nodeid_differences") and not block.get("scalar_differences"):
            named.append(f"{bug}:{block.get('reason') or 'not_identical'}")
    if named:
        shown = ", ".join(named[:cap])
        more = "" if len(named) <= cap else f", +{len(named) - cap} more"
        line += f" [prepare {shown}{more}]"
    if limits:
        line += f" [time limits: {len(limits)} group(s)]"
    if cache_problems:
        line += f" [cache: {cache_problems[0]}]"
    serial_peak = ((report.get("serial") or {}).get("concurrency") or {}).get("peak_cores")
    parallel_peak = ((report.get("parallel") or {}).get("concurrency") or {}).get("peak_cores")
    if serial_peak is not None or parallel_peak is not None:
        line += f" [rows peak: serial={serial_peak} parallel={parallel_peak}]"
    # M3 item V2 — a `not_identical` that is really "nothing was compared" must say
    # so, or the operator reads the gate's most alarming answer as a real finding.
    compared = report.get("compared") or {}
    if compared and not compared.get("ran", True):
        line += (
            f" [nothing compared: wanted={compared.get('bugs')} bug(s), "
            f"{compared.get('rows')} row(s)]"
        )
    return line


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_batches.py",
        description=(
            "Compare the serial reference batch and the parallel batch of the same "
            "build: per-nodeid native outcomes, matched-candidate oracle results and "
            "labels, and every row's deadline/timeout behaviour (M3 items S5/S6)."
        ),
    )
    parser.add_argument("--serial", required=True, type=Path, help="the --jobs 1 reference batch")
    parser.add_argument("--parallel", required=True, type=Path, help="the parallel batch")
    parser.add_argument(
        "--bug",
        action="append",
        default=None,
        help="compare only this bug (repeatable; default: every bug either batch prepared)",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="where to write the JSON report"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="print the verdict and write no report"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        report = compare_batches(args.serial, args.parallel, bugs=args.bug)
    except CompareError as exc:
        print(f"compare-batches: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        # [VANTA-WHY] M3 item V2 — exit 1 means "the two batches differ" and nothing
        # else, so a batch that could not be *read* must not share that code. A
        # per-row permission error (`OSError`), a row whose `seed` is not an int and
        # a manifest with a stray byte (`UnicodeDecodeError`, a `ValueError`) all
        # used to escape `main` as an uncaught traceback and a non-zero status a
        # caller could not tell from a real difference.
        print(f"compare-batches: cannot read the batches: {exc!r}", file=sys.stderr)
        return 2
    print(verdict_line(report))
    if not args.no_report:
        path = args.report or (Path(args.parallel) / "compare_batches.json")
        print(f"report: {write_report(report, path)}")
    return 0 if report["verdict"] == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
