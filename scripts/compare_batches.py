#!/usr/bin/env python3
"""compare_batches.py — the pandas/1 equivalence check for M3 items S5 and S6.

WHY THIS EXISTS
---------------
S5 and S6 make the harness run suite runs and rows concurrently, and Nathan's
gate before either may be used for real is an equivalence check on one bug:
*"a serial prepare and run on the same build as the reference, then the parallel
version, compared per nodeid apart from the tests the serial pair marked flaky"*
(``plans/private/M3-progress.md``, "The gate before real use"). This script is
that comparison. It reads two batch directories — one prepared and run with
``--prepare-jobs 1 --jobs 1``, one with the parallel settings — and reports every
difference that concurrency could have caused.

ACROSS BUILDS (M3 item Z2)
--------------------------
The serial reference is not always re-run on the new build. To check Y1 — "every
new overlap charges the shared budget and results stay identical" — the parallel
batch of the NEW build is compared against the serial batch that already exists
from an OLDER build (``m3-equiv-*``). That is sound because the serial batch is
only a *reference tree*: what it must supply is the per-nodeid native facts and
the row's time-limit behaviour, none of which depend on the harness build. The
harness build DOES enter the suite-cache key, so two builds cannot be served each
other's measurements even on one root.

So the from-build facts are compared and **reported as information**: a difference
in ``harness_commit`` / ``vanta_build_commit`` (or their dirty flags) or in a
bug's ``checkpoint`` is recorded in the ``build`` block and never fails the gate.
A build difference is a fact about the two runs, not a disagreement about the
tree or the schedule.

WHAT IS COMPARABLE, AND WHAT IS NOT
-----------------------------------
The two batches are two independent ``run`` invocations, so the model answered
twice. The parts of a batch that are a fact about the *tree and the scheduling*
are therefore comparable, and the parts the model produced are not:

  * **Comparable** (a difference here is a failure): the per-nodeid native sets
    ``prepare.json`` records (``fail_to_pass``, ``flaky_nodeids``,
    ``preexisting_failures``, ``native_failure_baseline``, ``collect_flip_nodeids``,
    ``f2p_undeclared``) and the set of collect-gate exclusions (``collect_excluded``),
    plus the native scalars beside them; whether a row exists at
    all for a job (``bug`` × ``config`` × ``mode`` × ``seed``); and every row's
    **time-limit behaviour** — ``timed_out``, ``stop_reason`` and
    ``native_timeout``, plus the ``scheduler`` block's wall time, deadline left and
    provider retries. This last group is the S6 half of the gate, stated by Nathan
    as *"Concurrency must not change outcomes through time limits ... If any does,
    the recommended --jobs is too high for this Mac."* ``flaky_nodeids`` is the one
    exception inside this group — M3 item AA5, below.
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

THE FLAKY SET (M3 item AA5)
---------------------------
Which tests a batch happens to *catch* being flaky is a fact about the clock, not
about the tree: the same suite can flip a different test on each run, so two
batches of one tree will not agree on ``flaky_nodeids`` for free. A difference
that is *only* that is therefore given its own verdict, ``identical_except_flaky``
(exit 4), with the nodeids each side saw and the other did not listed per side —
and only when the difference is proven inert: nothing else in ``prepare`` or
``rows`` differs, and no matched row's label moved (a moved grading already fails
``rows``). Anything else is ``not_identical``, exactly as before this item.

WHY A MATCHED ROW'S LABEL IS GRADED (M3 item AB1)
-------------------------------------------------
Nathan's rule (b), overruled: a matched row is the SAME candidate bytes judged by
the harness on both sides, so every field of its record is a statement about those
bytes and all of them must agree. The oracle block is a measurement of a fixed
patch against a fixed hidden suite; the label (and its rule, reason and verdict) is
what the run concluded about that same patch. Two runs of one patch that end
differently are a real difference in the recorded result, not noise — the attack
arm's fresh tests change which tests ran, but they must not change how the same
bytes are graded, and when they do that IS the finding. A moved label therefore
fails ``rows`` exactly as a moved oracle does; it is still reported in
``rows.matched[*].label_differences`` for side-by-side reading. Rules (a), (c),
(d) and (e) are unchanged.

UNMATCHED CANDIDATES (M3 item AB1)
----------------------------------
A review row whose candidate bytes are unique to one batch was never compared:
there is no matched counterpart. That is neither a difference nor agreement, so it
gets its own verdict, ``incomparable`` (exit 5), naming the groups that have
candidates on one side only. It ranks below ``not_identical`` — a proven difference
is the stronger, more actionable statement — and above the soft passes
(``identical_unverified`` / ``identical_except_flaky`` / ``identical``), because a
comparison that could not be made must never read as one that passed.

CACHE ISOLATION
---------------
On one suite-cache root two batches of the SAME build and the SAME trees would be
served each other's prepare suites and in-session native gates — and the
comparison would pass vacuously. Each batch is therefore given its own root
(``VANTA_SUITE_CACHE_DIR``, which drives both the Python and the Rust caches) and
this script checks it: it reads the cache root each batch recorded
(``prepare.json``'s per-tree ``scheduler.runs[*].cache.dir``) and **fails** when
the two batches name the same root, or when a recorded hit resolves under the
*other* batch's root instead of its own. It reports each batch's prepare-suite and
native-gate hit/miss counts and the writer of every hit.

The check is *never skipped silently*, and it is honest about what it cannot see:

  * A side that recorded **no hit** has nothing to resolve, and passes trivially
    (last night's serial batch ran with the cache off, so this is the normal case
    for a cross-build comparison).
  * A recorded **hit** is verified against that side's own root — and, as a
    sharper finding, if it resolves under the OTHER side's root instead, the hit
    was demonstrably served from the other batch.
  * If a side recorded hits but its root no longer exists on disk, the hits cannot
    be checked at all. That side is ``isolation unverifiable`` and the verdict is
    ``identical_unverified`` — never a plain ``identical`` (which would claim a
    proof that was not made) and never a false ``not_identical`` (which would
    blame the batches for a directory the operator deleted).

VERDICT
-------
``identical`` requires all six: no prepare-side nodeid difference, no matched-row
difference (the oracle AND the label must match, both being statements about the
same candidate bytes), no group present in only one batch, no unmatched candidate
group, no time-limit difference (neither a matched row whose flags moved nor a
group that hit a limit in the parallel batch where the serial batch did not), and
no cache-isolation problem. It additionally requires that the comparison was not
vacuous: at least one bug was wanted and at least one row existed across the two
batches. Two batches that compared nothing (no ``prepare.json`` bugs, no rows) are
``not_identical``, never ``identical`` — ``all([])`` is ``True``, and a gate must
not pass on absence.

The verdicts, worst first: ``not_identical`` (exit 1, a proven difference),
``incomparable`` (exit 5, a group whose candidates exist on one side only, so its
rows were never compared), ``identical_unverified`` (exit 3, agreed but the cache
half of the proof was unavailable), ``identical_except_flaky`` (exit 4, agreed
apart from a flaky-set difference proven inert) and ``identical`` (exit 0). Build
stamp differences are reported and never change the verdict.

USAGE (run from the repository root)
------------------------------------
    python3 scripts/compare_batches.py \
        --serial runs/bugsinpy/m3-eq-pandas_1-serial \
        --parallel runs/bugsinpy/m3-eq-pandas_1-parallel

    # the new parallel batch against last night's (older-build) serial reference:
    python3 scripts/compare_batches.py \
        --serial runs/bugsinpy/m3-equiv-pandas_1-serial \
        --parallel runs/bugsinpy/m3-timing-pandas_1-parallel

    # only one bug, and an explicit report path:
    python3 scripts/compare_batches.py --serial A --parallel B --bug pandas/1 \
        --report /tmp/compare.json

OUTPUT
------
One verdict line, and a JSON report at ``<parallel>/compare_batches.json`` unless
``--report`` names one (``--no-report`` writes none). Exit codes: ``0`` identical,
``1`` not identical, ``2`` the two batches could not be read (or a report could
not be written), ``3`` ``identical_unverified`` (the batches agree, but one side's
cache isolation could not be verified), ``4`` ``identical_except_flaky`` (the
batches agree apart from a flaky-set difference proven inert — M3 item AA5), and
``5`` ``incomparable`` (a group's candidate bytes exist on one side only, so its
rows were never compared — M3 item AB1).
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
    "preexisting_failures",
    "native_failure_baseline",
    "collect_flip_nodeids",
    "collect_excluded",
)
"""The per-nodeid sets ``prepare.json`` records per bug — the S5 half of the gate.

[VANTA-WHY] M3 item AA5 — ``flaky_nodeids`` is deliberately NOT in this tuple.
Which tests a batch happens to *catch* in the act of being flaky is a fact about
the clock, not about the tree: the same suite can flip a different test on each
run. It is compared, but as its own category
(:func:`compare_prepare`'s ``flaky_differences``) and under the ``identical_except_flaky``
verdict, never as a plain nodeid difference.

``collect_excluded`` is a ``{file: reason}`` map, not a nodeid list; the set
difference runs over its *keys*, which is the fact worth comparing — the set of
files the collect gate had to set aside. Two preparations of one bug that set
aside different files are not the same preparation even when the nodeids agree.
"""

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

BUILD_STAMP_FIELDS = (
    "harness_commit",
    "harness_dirty",
    "vanta_build_commit",
    "vanta_build_dirty",
)
"""The from-build facts a row records (``runner._row_record``).

Two trees, two commits (§7.3): ``vanta_build_commit`` is what ``build.rs`` baked
into the binary and ``run.summary`` reports, ``harness_commit`` is this repo's HEAD
when the batch started. Across builds these differ **by design** — the whole point
of item Z2 is comparing a new-build parallel batch against an old-build serial
reference — so a difference here is *reported* and never fails the gate. The
per-bug ``checkpoint`` is compared beside them for the same reason.
"""

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


def _in_scope(
    records: Sequence[Mapping[str, Any]], bugs: Optional[Sequence[str]]
) -> list[Mapping[str, Any]]:
    """The records for the bugs in scope; ``None`` keeps every record.

    [VANTA-WHY] M3 item X3 — ``--bug`` must scope the cache half the same way it
    already scopes ``prepare`` and ``rows``. Before this, ``--bug pandas/1`` still
    checked every other bug's hits, so an unrelated bug's unresolvable entry could
    turn a scoped comparison into ``identical_unverified`` — the operator asking
    about one bug was told about another. ``None`` (no ``--bug``) is the identity,
    so the unscoped behaviour is bit-for-bit what it was.
    """
    if bugs is None:
        return list(records)
    wanted = {str(bug) for bug in bugs}
    return [record for record in records if str(record.get("bug") or "") in wanted]


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
    """A row's reported label, its rule, and the SOURCE the label was read from.

    [VANTA-WHY] M3-E follow-up — the label is the EFFECTIVE one
    (``manifest.label_of``): a MEASURED relabel (the oracle re-run on the row's
    reconstructed final patch) outranks a DERIVED one, which outranks the original
    the run wrote. ``label_source`` names which of the three this row resolved to,
    so a difference is readable rather than implied; it is REPORTED and not graded,
    because the same value read from two sources is the same conclusion about the
    candidate's bytes.
    """
    label, source = manifest_mod.label_of(record)
    return {
        "label": label or record.get("label"),
        "label_source": source,
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
    """The S5 half: are the two preparations' nodeid-level facts the same?

    [VANTA-WHY] M3 item AA5 — the returned block distinguishes two kinds of
    agreement. ``identical`` still means *nothing at all differs*, so every
    existing reader keeps its meaning. ``identical_except_flaky`` is the wider
    one: everything agrees apart from ``flaky_nodeids``, which is a difference the
    clock can manufacture on its own. The two are separate keys rather than a
    widened ``identical`` because the module's rule is that ``identical`` must
    never overclaim (see :func:`compare_batches`, which decides whether the flaky
    difference is inert before it may use the wider key).
    """
    left = prepare_bugs(serial)
    right = prepare_bugs(parallel)
    per_bug: dict[str, Any] = {}
    for bug in bugs:
        a = left.get(bug)
        b = right.get(bug)
        if a is None or b is None:
            per_bug[bug] = {
                "identical": False,
                "identical_except_flaky": False,
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
        # M3 item AA5 — the flaky set, compared in its own category.
        flaky_only_serial = sorted(set(a.get("flaky_nodeids") or []) - set(b.get("flaky_nodeids") or []))
        flaky_only_parallel = sorted(set(b.get("flaky_nodeids") or []) - set(a.get("flaky_nodeids") or []))
        flaky = (
            {"only_in_serial": flaky_only_serial, "only_in_parallel": flaky_only_parallel}
            if flaky_only_serial or flaky_only_parallel
            else {}
        )
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
            "flaky_differences": flaky,
            "scalar_differences": scalars,
            "timing": timing,
            "identical": not nodeids and not scalars and not flaky,
            "identical_except_flaky": not nodeids and not scalars,
        }
    return {
        "bugs": per_bug,
        "identical": all(b["identical"] for b in per_bug.values()),
        "identical_except_flaky": all(b["identical_except_flaky"] for b in per_bug.values()),
    }


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
        # [VANTA-WHY] M3 item AB1 — Nathan's rule (b), overruled. A matched row is
        # the same candidate bytes on both sides, so the ORACLE and the LABEL are
        # both statements about those bytes and both must agree. The oracle is the
        # harness's own deterministic measurement; the label is what the run
        # concluded about the same patch, and two runs of one patch that end
        # differently are a real difference in the recorded result. A moved label
        # therefore fails the row, exactly as a moved oracle does — the attack arm
        # writes new tests each run, but those tests must not change how the same
        # bytes are graded, and when they do that IS the finding. The label values
        # are still recorded in ``label_differences`` for side-by-side reading.
        labels = {
            name: [value_a, value_b]
            for name, value_a, value_b in (
                ("label", _label_of(a)["label"], _label_of(b)["label"]),
                ("label_rule", a.get("label_rule"), b.get("label_rule")),
                ("label_reason", a.get("label_reason"), b.get("label_reason")),
                ("verdict", a.get("verdict"), b.get("verdict")),
            )
            if value_a != value_b
        }
        # M3-E follow-up — which copy each side's label was read from. Reported
        # beside the grading rather than graded: the same value read from two
        # sources is the same conclusion about the same candidate bytes.
        label_sources = {
            "serial": _label_of(a)["label_source"],
            "parallel": _label_of(b)["label_source"],
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
                "label_sources": label_sources,
                "oracle": oracle,
                "time_limit_differences": limits,
                "identical": oracle["identical"] and not limits and not labels,
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
# build stamps — reported as information, never graded
# ---------------------------------------------------------------------------
def build_stamps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The distinct from-build facts one batch's rows recorded, per field and per bug.

    A set per field, not one value: a batch that ran rows from two invocations (or a
    ``--resume`` after a rebuild) can legitimately record more than one commit, and
    collapsing that to "the last one" would hide exactly the drift this item wants
    visible. The per-bug ``checkpoint`` is collected the same way, because a layer
    can be re-stamped between the two batches.
    """
    fields: dict[str, set[Any]] = {name: set() for name in BUILD_STAMP_FIELDS}
    checkpoints: dict[str, set[Any]] = {}
    for row in rows:
        for name in BUILD_STAMP_FIELDS:
            fields[name].add(row.get(name))
        checkpoints.setdefault(str(row.get("bug") or ""), set()).add(row.get("checkpoint"))
    return {
        # ``key=str`` because a field can hold ``None`` beside a string or a bool, and
        # ``None`` is not orderable against either — the mix is expected, not a bug.
        "fields": {name: sorted(values, key=str) for name, values in fields.items()},
        "checkpoints": {
            bug: sorted(values, key=str) for bug, values in sorted(checkpoints.items())
        },
    }


def compare_build(
    serial_rows: Sequence[Mapping[str, Any]],
    parallel_rows: Sequence[Mapping[str, Any]],
    bugs: Sequence[str],
) -> dict[str, Any]:
    """Both batches' build stamps, and every way they differ — **information only**.

    [VANTA-WHY] M3 item Z2 — this half never touches the verdict. The serial
    reference is an older build by construction (``m3-equiv-*``), so its
    ``harness_commit`` and ``vanta_build_commit`` are *expected* to differ from the
    new batch's, and a gate that failed on that would reject the very comparison it
    was asked to make. What the two batches must agree on is the tree and the
    schedule (``prepare`` and ``rows``); the build is reported so the operator can
    see it, and the suite cache — which keys on the build — is what actually stops
    two builds being served each other's measurements.
    """
    serial = build_stamps(serial_rows)
    parallel = build_stamps(parallel_rows)
    differences: list[str] = []
    for name in BUILD_STAMP_FIELDS:
        left, right = serial["fields"][name], parallel["fields"][name]
        if left != right:
            differences.append(f"{name}: serial={left} parallel={right}")
    for bug in bugs:
        left = serial["checkpoints"].get(bug, [])
        right = parallel["checkpoints"].get(bug, [])
        if left != right:
            differences.append(f"{bug} checkpoint: serial={left} parallel={right}")
    return {
        "serial": serial,
        "parallel": parallel,
        "differences": differences,
        "matches": not differences,
    }


# ---------------------------------------------------------------------------
# cache isolation — the two batches must not have shared a root
# ---------------------------------------------------------------------------
def _hit_isolation(
    label: str,
    own_dirs: Sequence[str],
    other_dirs: Sequence[str],
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[list[str], list[str]]:
    """One batch's recorded hits, sorted into *definite leaks* and *unverifiable*.

    A hit was served from a directory by definition, so asking where its key lives
    answers where it came from — three ways, and the three must not be collapsed:

    * **Under this batch's own root** — isolated. The normal re-prepared-tree case.
    * **Under the other batch's root** — a definite leak: this batch was served the
      other batch's measurement, the vacuous pass the check exists to stop.
    * **Under neither** — the honest gap. The entry may have been served from the
      other batch's root and *that* was deleted too, or this batch's own root is
      gone. Either way nothing here can be resolved, so it is an *unverifiable* side
      rather than a fabricated difference: reporting it as a problem would blame the
      batches for a directory the operator deleted.
    """
    problems: list[str] = []
    unverified: list[str] = []

    def _resolves(dirs: Sequence[str], key: str) -> bool:
        return any(suite_cache_mod.entry_path_in(root, key).is_file() for root in dirs)

    for kind, records in groups:
        for record in records:
            key = str(record["key"])
            if record["state"] != CACHE_HIT_STATE or not key:
                continue
            if _resolves(own_dirs, key):
                continue
            where = ", ".join(own_dirs) or "(no directory recorded)"
            if _resolves(other_dirs, key):
                problems.append(
                    f"{label} {kind} cache hit {key[:12]}… is not an entry under {label}'s own "
                    f"cache directory ({where}); it resolves under the other batch's root, so it "
                    f"was served from the other batch"
                )
                continue
            unverified.append(
                f"{label} {kind} cache hit {key[:12]}… is not an entry under {label}'s own cache "
                f"directory ({where}) and resolves under no directory the other batch named; "
                f"{label}'s cache isolation cannot be verified"
            )
    return problems, unverified


def compare_cache(
    serial_prepare: Mapping[str, Any],
    parallel_prepare: Mapping[str, Any],
    serial_rows: Sequence[Mapping[str, Any]],
    parallel_rows: Sequence[Mapping[str, Any]],
    bugs: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Prove the two batches did not share a suite-cache root — or say that it cannot.

    Each batch's root is the one its prepare suites recorded; the in-session native
    gate uses the SAME root (one ``VANTA_SUITE_CACHE_DIR`` drives both caches), so a
    native hit is checked against it too.

    [VANTA-WHY] M3 item Z2 — the check is **never skipped silently**. A side that
    recorded no hit has nothing to resolve and passes trivially; a side that recorded
    hits is resolved against its own root. ``checked`` still says whether both batches
    recorded a root (a shared-root test needs one on each side to mean anything), but
    it no longer gates the hit check: a hit whose key resolves nowhere is an
    *unverifiable* side, carried in ``unverified`` and turned into the
    ``identical_unverified`` verdict, never a plain ``identical`` (which would claim a
    proof that was not made) and never a false ``not_identical`` (which would blame
    the batches for a directory that is gone).

    [VANTA-WHY] M3 item X3 — ``bugs`` scopes the **hit checks and the summaries**
    to the bugs the operator asked about, and nothing else. The **root** stays
    read from *every* record on both sides (``cache_dirs`` and the shared-root
    test below): sharing a root is a fact about the whole batch, and a scoped
    ``--bug`` must not be able to hide it by leaving the shared root's records
    out of scope. ``None`` is the identity, so an unscoped comparison is
    unchanged.
    """
    all_serial_suite = prepare_suite_cache(serial_prepare)
    all_parallel_suite = prepare_suite_cache(parallel_prepare)
    serial_suite = _in_scope(all_serial_suite, bugs)
    parallel_suite = _in_scope(all_parallel_suite, bugs)
    serial_native = _in_scope(native_cache_records(serial_rows), bugs)
    parallel_native = _in_scope(native_cache_records(parallel_rows), bugs)
    # Root detection reads every record, deliberately unscoped (see the docstring).
    serial_dirs = cache_dirs(all_serial_suite)
    parallel_dirs = cache_dirs(all_parallel_suite)

    problems: list[str] = []
    unverified: list[str] = []
    checked = bool(serial_dirs and parallel_dirs)
    if checked:
        shared = sorted(set(serial_dirs) & set(parallel_dirs))
        if shared:
            problems.append(
                "both batches used the same suite cache directory, so neither computed "
                f"its own results: {', '.join(shared)}"
            )
    for label, own, other, groups in (
        ("serial", serial_dirs, parallel_dirs, (("prepare", serial_suite), ("native", serial_native))),
        (
            "parallel",
            parallel_dirs,
            serial_dirs,
            (("prepare", parallel_suite), ("native", parallel_native)),
        ),
    ):
        found, unresolved = _hit_isolation(label, own, other, groups)
        problems.extend(found)
        unverified.extend(unresolved)

    return {
        "checked": checked,
        "bugs": list(bugs) if bugs is not None else None,
        "problems": problems,
        "unverified": unverified,
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
        "verifiable": not unverified,
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
    #
    # [VANTA-WHY] T3 — read from the prepare payloads ALREADY loaded above rather
    # than re-reading ``prepare.json`` from disk. It is the same file the rest of
    # this comparison was built from, and a second read is a second chance to see a
    # batch that differs from the one being compared.
    def flaky_nodeids_by_bug(payload: Mapping[str, Any]) -> dict[str, set[str]]:
        return {
            bug: set(block.get("flaky_nodeids") or [])
            for bug, block in prepare_bugs(payload).items()
        }

    serial_flaky = flaky_nodeids_by_bug(serial_prepare)
    parallel_flaky = flaky_nodeids_by_bug(parallel_prepare)
    excluded_by_bug = {
        bug: serial_flaky.get(bug, set()) | parallel_flaky.get(bug, set()) for bug in wanted
    }

    prepare = compare_prepare(serial_prepare, parallel_prepare, wanted)
    rows = compare_rows(serial_rows, parallel_rows, excluded_by_bug)
    # [VANTA-WHY] M3 item X3 — `bugs` (the raw `--bug`, `None` when unset) scopes the
    # cache half's hit checks and summaries. `wanted` is deliberately NOT passed: it
    # is the all-bugs union when `--bug` is absent, and the root/shared-root test must
    # keep reading every record either way.
    cache = compare_cache(
        serial_prepare, parallel_prepare, serial_rows, parallel_rows, bugs=bugs
    )
    # [VANTA-WHY] M3 item Z2 — the build stamps of two batches that are *meant* to
    # come from different builds. Read and reported; never folded into the verdict.
    build = compare_build(serial_rows, parallel_rows, wanted)

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
    # [VANTA-WHY] M3 item AA5 — which tests a batch catches being flaky is a fact
    # about the clock, so a difference there is not a disagreement about the tree:
    # the timing batch's pandas/1 was `not_identical` from a flaky Hypothesis test
    # alone. M3 item AA5/AB1 — the flaky carve-out is allowed ONLY when the
    # flaky set is the only difference. A moved row grading OR a moved row label now
    # fails ``rows["identical"]`` (M3 item AB1: both are statements about the same
    # candidate bytes), so the carve-out cannot apply once either has moved; the
    # ``label_changed`` flag is kept for the report.
    flaky_diff_bugs = sorted(
        bug
        for bug, block in (prepare["bugs"] or {}).items()
        if block.get("flaky_differences")
    )
    label_changed = any(row.get("label_differences") for row in rows["matched"])
    unmatched = rows["unmatched_candidates"]
    incomparable_groups = sorted(
        {
            tuple(entry["identity"][:4])
            for entry in (unmatched.get("only_in_serial") or [])
            + (unmatched.get("only_in_parallel") or [])
        }
    )
    flaky_only = bool(flaky_diff_bugs) and not label_changed
    passed = bool(
        compared["ran"]
        and prepare["identical_except_flaky"]
        and rows["identical"]
        and cache["identical"]
    )
    # [VANTA-WHY] M3 item Z2/AA5/AB1 — five outcomes, worst first. A proven difference
    # (a moved oracle, label, time limit, prepare field or cache isolation) is
    # `not_identical` and wins over every softer verdict. `incomparable` is next: a
    # group whose candidate bytes exist on one side only was never compared — its rows
    # have no counterpart, so the comparison must not claim to have passed, and it is
    # not a difference either, hence its own verdict and its own exit code (5). Then
    # the soft passes: an unverifiable cache side, then a flaky-only difference proven
    # inert, then plain `identical`.
    if not passed:
        verdict = "not_identical"
    elif incomparable_groups:
        verdict = "incomparable"
    elif cache["unverified"]:
        verdict = "identical_unverified"
    elif flaky_only:
        verdict = "identical_except_flaky"
    else:
        verdict = "identical"
    flaky = {
        "bugs": flaky_diff_bugs,
        "differences": {
            bug: (prepare["bugs"][bug].get("flaky_differences") or {}) for bug in flaky_diff_bugs
        },
        "label_changed": label_changed,
        "discounted": flaky_only,
    }
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
        "build": build,
        "prepare": prepare,
        "cache": cache,
        "rows": rows,
        "flaky": flaky,
        "verdict": verdict,
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
    # M3 item AA5 — `identical` is False for a bug whose ONLY difference is the flaky
    # set, which the verdict discounts; a bug belongs in the failure list only when
    # the difference survives the carve-out. `.get(..., block["identical"])` keeps a
    # hand-built report (one that predates the wider key) reading as it always did.
    prepare_diffs = sorted(
        b
        for b, block in prepare_bugs_report.items()
        if not block.get("identical_except_flaky", block["identical"])
    )
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
    cache_unverified = (report.get("cache") or {}).get("unverified") or []
    build_diffs = (report.get("build") or {}).get("differences") or []
    flaky_bugs = (report.get("flaky") or {}).get("bugs") or []
    parts = [
        f"prepare_diffs={len(prepare_diffs)}",
        f"matched={len(matched)}",
        f"row_diffs={len(row_diffs)}",
        f"label_diffs={len(label_diffs)}",
        f"time_limit_diffs={len(limits)}",
        f"missing_groups={len(missing)}",
        f"unmatched_candidates={n_unmatched}",
        f"cache_problems={len(cache_problems)}",
        f"cache_unverified={len(cache_unverified)}",
        f"flaky_diffs={len(flaky_bugs)}",
        f"build_diffs={len(build_diffs)}",
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
    # M3 item AA5 — a flaky-only difference is discounted, not a failure, so it is
    # named as its own thing (with the nodeids each side saw and the other did not),
    # or the operator cannot tell which difference the verdict excused.
    if flaky_bugs:
        named_flaky: list[str] = []
        for bug in flaky_bugs:
            diff = (prepare_bugs_report.get(bug) or {}).get("flaky_differences") or {}
            serial_only = ", ".join(diff.get("only_in_serial") or []) or "-"
            parallel_only = ", ".join(diff.get("only_in_parallel") or []) or "-"
            named_flaky.append(
                f"{bug} serial-only=[{serial_only}] parallel-only=[{parallel_only}]"
            )
        shown = "; ".join(named_flaky[:cap])
        more = "" if len(named_flaky) <= cap else f"; +{len(named_flaky) - cap} more"
        excuse = "" if (report.get("flaky") or {}).get("discounted", True) else (
            " (withheld: a label moved)"
        )
        line += f" [flaky{excuse}: {shown}{more}]"
    # M3 item AB1 — an `incomparable` verdict means a group's candidate bytes exist on
    # one side only, so its rows were never compared; name those groups, or the
    # operator cannot tell which bug's review candidates went missing.
    incomparable_named: list[str] = []
    for entry in unmatched.get("only_in_serial") or []:
        incomparable_named.append(f"{entry['identity'][0]} serial-only")
    for entry in unmatched.get("only_in_parallel") or []:
        incomparable_named.append(f"{entry['identity'][0]} parallel-only")
    if incomparable_named:
        shown = ", ".join(incomparable_named[:cap])
        more = (
            "" if len(incomparable_named) <= cap else f", +{len(incomparable_named) - cap} more"
        )
        line += f" [incomparable: {shown}{more}]"
    if limits:
        line += f" [time limits: {len(limits)} group(s)]"
    if cache_problems:
        line += f" [cache: {cache_problems[0]}]"
    # M3 item Z2 — an unverifiable cache side is what makes the verdict
    # `identical_unverified`; name it, or the operator cannot tell why the gate did
    # not say plain `identical`.
    if cache_unverified:
        line += f" [cache unverified: {cache_unverified[0]}]"
    # M3 item Z2 — build differences never change the verdict, so they are named as
    # information: the operator comparing a new build against an old reference must
    # see WHICH of the two trees moved, not just that something did.
    if build_diffs:
        shown = "; ".join(build_diffs[:cap])
        more = "" if len(build_diffs) <= cap else f"; +{len(build_diffs) - cap} more"
        line += f" [build (information, not a failure): {shown}{more}]"
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


PER_BUG_DIRNAME = "compare_batches"
"""M3 item AA5 — where the per-bug reports go, under the parallel batch directory.

``<parallel>/compare_batches.json`` is the whole comparison; a file per bug beside
it (``<parallel>/compare_batches/<bug>.json``) is the same comparison as
``--bug <bug>`` would have produced, so a single bug's equivalence can be read —
and cited — on its own.
"""


def bug_slug(bug: str) -> str:
    """A bug key as a file name. ``pandas/1`` → ``pandas__1``.

    A bug key contains ``/`` by construction, so it cannot be a file name as it
    stands: ``pandas/1.json`` would write into a ``pandas`` directory rather than
    beside the other bugs, and ``write_report`` would be handed a path whose parent
    does not exist. The separator is doubled rather than dropped so the slug always
    shows where the split was.

    The mapping is injective over the keys a manifest actually writes — always
    ``<project>/<index>``, exactly one ``/``, and the index is digits-only — so two
    distinct bugs never share a file. It is *not* injective over arbitrary strings
    (a synthetic ``a__b`` and ``a/b`` both slug to ``a__b``); that is fine, because
    the only keys it is ever called with come from the manifest.
    """
    return str(bug).replace("/", "__")


def write_per_bug_reports(
    parallel_dir: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Path]:
    """Write one ``--bug``-scoped report per bug, and return bug → path.

    Each payload is a real re-run of :func:`compare_batches` scoped to one bug, not
    a slice of the whole report: scoping also narrows the cache half (M3 item X3)
    and the row set, so a slice would quietly disagree with what ``--bug`` prints.
    The comparison is pure file reading, so re-running it per bug is cheap; the
    comparisons happen in ``main``'s read phase and only the WRITES happen here, so
    a write failure carries its own code and message.
    """
    directory = Path(parallel_dir) / PER_BUG_DIRNAME
    out: dict[str, Path] = {}
    for bug, payload in payloads.items():
        out[bug] = write_report(payload, directory / f"{bug_slug(bug)}.json")
    return out


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
        # M3 item AA5/T3 — the per-bug reports are the same `--bug`-scoped
        # comparison, re-run over the same two batches, so a READ failure here must
        # be exit 2 ("cannot read"), never a half-written run that already printed a
        # verdict. Only the COMPARISONS happen here; every WRITE is deferred to the
        # write phase below, where a write failure gets its own code and message.
        # `--no-report` writes nothing at all, as it always has.
        per_bug_payloads = (
            {}
            if args.no_report
            else {
                bug: compare_batches(args.serial, args.parallel, bugs=[bug])
                for bug in report["bugs"]
            }
        )
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
        # [VANTA-WHY] T3 — a report that could not be *written* is "cannot read or
        # write the batches" (exit 2), not "the batches differ" (exit 1) and not a
        # verdict code. Before this the report write sat outside every `try`, so a
        # report path under an unwritable directory escaped `main` as an uncaught
        # traceback and a status a caller could not tell from a real difference.
        try:
            written = write_report(report, path)
            per_bug = write_per_bug_reports(args.parallel, per_bug_payloads)
        except OSError as exc:
            print(f"compare-batches: cannot write the report: {exc!r}", file=sys.stderr)
            return 2
        print(f"report: {written}")
        for bug in report["bugs"]:
            print(f"report [{bug}]: {per_bug[bug]}")
    # [VANTA-WHY] M3 item Z2/AA5/AB1 — five outcomes need five codes: 0 identical,
    # 3 `identical_unverified` (the batches agree, but a side's cache isolation could
    # not be resolved), 4 `identical_except_flaky` (the batches agree apart from a
    # flaky-set difference proven inert — M3 item AA5, its own code so a gate can
    # tell it from a plain pass), 5 `incomparable` (a group's candidate bytes exist on
    # one side only, so its rows were never compared — M3 item AB1, its own code so a
    # gate cannot mistake "could not compare" for "compared and passed"), and 1
    # `not_identical` (a proven difference — and, because the fold runs first, a
    # proven difference always wins over every softer verdict, including
    # `incomparable`).
    return {
        "identical": 0,
        "identical_unverified": 3,
        "identical_except_flaky": 4,
        "incomparable": 5,
    }.get(report["verdict"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
