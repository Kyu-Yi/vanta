#!/usr/bin/env python3
"""Feedback-loop harness for BugsInPy — the one command (design §7.1).

    python3 scripts/bugsinpy_run.py bake    --pythons 3.8 --yes   # §11 step 3
    python3 scripts/bugsinpy_run.py prepare --split plans/bugsinpy_split.json
    python3 scripts/bugsinpy_run.py run     --batch runs/bugsinpy/<batch>
    python3 scripts/bugsinpy_run.py report  --batch runs/bugsinpy/<batch>

``bake`` builds the Shuru checkpoints the split names — one interpreter layer per
Python minor version plus a thin per-bug dependency layer — and is the only stage
that needs network. ``prepare`` is the stage after it: it reads BugsInPy's own
metadata, cuts the SWE-bench trees, derives ``FAIL_TO_PASS`` / ``PASS_TO_PASS``,
strips the hidden tests, runs the context precheck and the leak gate, and writes
``prepare.json`` plus ``context_report.json`` into the batch directory. ``run``
consumes that prepared batch: per wave, baseline repair → vanta repair → baseline
review → vanta review (§7.1, D13), one ``vanta_m3`` process at a time, appending a
row per job to the manifest and resuming from ``state.json``. ``report`` reads the
manifest alone and re-derives the digest, the ``metrics.py --labels`` file and the
per-config scoring.

``run`` needs ``--batch`` because the batch directory is where the resume cursor
lives: a runner that invented a fresh directory on every start could never resume,
and ``prepare``'s artifacts would be in the other one. That is the whole of
"prepare-or-resume" — ``prepare`` once, then ``run`` as often as it takes.

Nothing third-party executes on the host. Checkouts and tree surgery are host-side
``git`` and filesystem work; every command that imports the project under test is
issued inside Shuru (design C3, ``suite.py``). ``run`` also invokes Vanta as a
subprocess, and Vanta's sandbox backend is Shuru.

Exit codes: ``0`` everything prepared/ran, ``1`` a bug or several did not prepare
(or a wave failed its gate), ``2`` the harness itself could not proceed (missing
checkpoint, moved metadata, missing ``vanta_m3``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import arms as arms_mod  # noqa: E402
from bugsinpy import bake as bake_mod  # noqa: E402
from bugsinpy import budget as budget_mod  # noqa: E402
from bugsinpy import bugsinpy_meta as meta_mod  # noqa: E402
from bugsinpy import compact as compact_mod  # noqa: E402
from bugsinpy import context as context_mod  # noqa: E402
from bugsinpy import frozen_config as frozen_mod  # noqa: E402
from bugsinpy import issue as issue_mod  # noqa: E402
from bugsinpy import issue_source  # noqa: E402
from bugsinpy import manifest as manifest_mod  # noqa: E402
from bugsinpy import mutants as mutants_mod  # noqa: E402
from bugsinpy import oracle as oracle_mod  # noqa: E402
from bugsinpy import pool as pool_mod  # noqa: E402
from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import records as records_mod  # noqa: E402
from bugsinpy import replay as replay_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
from bugsinpy import split as split_mod  # noqa: E402
from bugsinpy import suite as suite_mod  # noqa: E402
from bugsinpy import suite_cache  # noqa: E402
from bugsinpy import trees as tree_mod  # noqa: E402
from bugsinpy.shuru import ShuruError  # noqa: E402
from bugsinpy.split import Split, SplitError, load_split  # noqa: E402

EXPECTED_ERRORS = (
    SplitError,
    bake_mod.BakeError,
    prepare_mod.PrepareError,
    meta_mod.MetadataError,
    ShuruError,
    runner_mod.RunnerError,
    # Item 13 — a second queue over one batch directory is a refusal, not a crash.
    manifest_mod.BatchLocked,
    context_mod.ContextError,
    oracle_mod.OracleError,
    suite_mod.SuiteError,
    tree_mod.TreeError,
    tree_mod.StripError,
    issue_mod.IssueError,
    compact_mod.CompactError,
    frozen_mod.FrozenConfigError,
    records_mod.RecordError,
    mutants_mod.MutantError,
    pool_mod.PoolError,
    replay_mod.ReplayError,
)
"""Every failure the harness raises on purpose, and the reason the CLI prints them.

[VANTA-WHY] Each of these already carries the whole report in its message — which
stage refused, what it saw, what to do — so a traceback adds frames the reader has
to walk past to reach the sentence that matters, and it hides the exit code that
says *this is a refusal, not a crash*. An error class missing from this tuple exits
as an unhandled exception instead, which is how ``ContextError`` used to look.
"""

DEFAULT_SPLIT = REPO_ROOT / "plans" / "bugsinpy_split.json"
DEV_SPLIT = REPO_ROOT / "plans" / "bugsinpy_dev_split.json"
"""The tuning split (§8). The localization audit is for this one only: ranking
quality is a number one tunes on, and a tuning number computed over the
evaluation split is how a held-out set stops being held out."""

RUNS_DIR = REPO_ROOT / "runs" / "bugsinpy"

EXIT_OK = 0
EXIT_BUG_FAILED = 1
EXIT_HARNESS_FAILED = 2


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# selection — what the batch is, and how much of it to prepare
# ---------------------------------------------------------------------------
def bug_key(entry: dict[str, Any]) -> str:
    return f"{entry['project']}/{entry['bug_id']}"


def render_flagged(flagged: Sequence[tuple[str, str]]) -> str:
    """The end-of-run list of bugs that failed for their own reason (§3.2).

    [VANTA-WHY] Printed **separately** from the gate results on purpose: a flagged
    bug — one whose dependencies could not be read at bake time, or that a stage
    could not finish — is not a §14.7 gate failure, so the harness does not promote
    a reserve to replace it. The split is unchanged, the reason is here in full, and
    the decision is the reader's.
    """
    lines = [
        f"flagged bugs ({len(flagged)}) — not gate failures; no reserve was promoted "
        "and the split is unchanged:",
    ]
    for key, reason in flagged:
        lines.append(f"  - {key}: {reason}")
    return "\n".join(lines)


def select(
    split: Split,
    *,
    only: list[str],
    waves: list[int],
    reserves: str,
    measure_only: bool = False,
    refreeze: Sequence[str] = (),
) -> list[tuple[dict[str, Any], bool]]:
    """``[(entry, full)]`` in the committed file order.

    ``full`` means the whole §1.3 derivation — three suites, twice each, inside
    Shuru. A reserve is ``full=False``: it is checked out and context-measured so
    that ``context_sizes.md`` can justify promoting it, but its suites are not run
    until someone changes its ``role`` and re-runs prepare (§8). Naming a bug with
    ``--only`` is an explicit request, so it is always ``full``.

    ``refreeze`` is the same kind of request and is selected the same way: naming a
    bug with ``--refreeze PROJECT/BUG`` selects only it, in full, because the
    rewrite has to re-fetch the report and re-run the gate (§2.4.3) — a
    measure-only pass would read the very file the caller is saying is wrong.

    ``measure_only`` forces every candidate to ``full=False`` — the checkout, the
    SWE-bench trees and the §1.6 context precheck, and nothing that boots a VM.
    That is what makes the pilot set a *measured* choice: the sizes of all
    candidates, pilots and reserves, are available before a single checkpoint is
    baked or a wave scheduled. The numbers it produces are the pre-strip tree's
    (§1.6 step 1), so they are an upper bound on what Vanta is finally handed.
    """
    entries = split.bugs

    named = sorted(set(only) | set(refreeze))
    if named:
        known = {bug_key(entry) for entry in entries}
        unknown = sorted(set(named) - known)
        if unknown:
            raise SplitError(f"--only/--refreeze name bugs that are not in the split: {unknown}")
        wanted = set(named)
        return [(entry, not measure_only) for entry in entries if bug_key(entry) in wanted]

    if waves:
        wanted_waves = set(waves)
        entries = [e for e in entries if int(e["wave"]) in wanted_waves]

    selected: list[tuple[dict[str, Any], bool]] = []
    for entry in entries:
        if measure_only:
            selected.append((entry, False))
        elif entry.get("role", "pilot") == "pilot":
            selected.append((entry, True))
        elif reserves == "measure":
            selected.append((entry, False))
    return selected


def render_plan(selected: list[tuple[dict[str, Any], bool]]) -> str:
    lines = [
        "| bug | role | wave | py | checkpoint | mode | test_file | issue risk |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry, full in selected:
        lines.append(
            f"| {bug_key(entry)} | {entry.get('role', 'pilot')} | {entry['wave']} | "
            f"{entry['python_version']} | {entry['checkpoint']} | "
            f"{'full' if full else 'context-only'} | {entry['test_file']} | "
            f"{entry.get('issue_needs_decision') or '—'} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def cache_state_line(
    *, checkpoints: Sequence[str], unstamped: Sequence[str]
) -> str:
    """One line stating whether the selected layers can key the suite cache.

    [VANTA-WHY] M3 item U3 — the failure this guards against is silent. A layer whose
    provenance carries no ``checkpoint_id`` makes ``suite._measure_suite`` answer
    ``off:unknown_checkpoint``, so its suites are measured on every run and never reused,
    while every other line of the batch looks normal. The state is therefore *stated* in
    the summary rather than left to be inferred from a hit count of zero.
    """
    if suite_cache.cache_disabled():
        return (
            "Suite cache: OFF — --no-cache was passed; every suite is measured this run "
            "(U2 still replaces an entry it could not use)."
        )
    if not checkpoints:
        return "Suite cache: on — no layer is selected, so there is nothing to key."
    if unstamped:
        return (
            f"Suite cache: OFF — {len(unstamped)} of {len(checkpoints)} selected layer(s) "
            "carry no checkpoint_id, so no suite of theirs can be keyed or reused: "
            + ", ".join(unstamped)
            + ". Fix them once, without re-baking: "
            "`python3 scripts/bugsinpy_run.py stamp-existing`."
        )
    return (
        f"Suite cache: on — all {len(checkpoints)} selected layer(s) carry a "
        "checkpoint_id."
    )


def issues_dir_flag(args: argparse.Namespace) -> Optional[Path]:
    """The ``--issues-dir`` this invocation was given, or ``None`` for the committed default.

    ``None`` is not "no directory": it is "the flag was not given", and every stage then
    reads and writes the committed ``plans/bugsinpy_issues`` exactly as before. A caller
    that gives the flag gets a *required* directory — a report missing there is a refusal,
    never a fall back to the default (§2.4).
    """
    raw = getattr(args, "issues_dir", None)
    return None if raw is None else Path(raw)


def prepare_issues_dir(args: argparse.Namespace) -> Path:
    """The directory ``prepare`` freezes into and reads: ``--issues-dir``, else the default.

    A held-out split passes ``--issues-dir plans/private/bugsinpy_issues`` (gitignored) so
    its report text is frozen and verifiable without being published; the dev split omits
    the flag and gets the committed ``plans/bugsinpy_issues`` byte-for-byte.
    """
    given = issues_dir_flag(args)
    return (REPO_ROOT / issue_source.ISSUES_DIR) if given is None else given


def prepare_one_bug(
    *,
    entry: Mapping[str, Any],
    full: bool,
    split: Split,
    bugsinpy_root: Path,
    repo_cache: Path,
    batch_dir: Path,
    deps_dir: Path,
    issue_sources: Any,
    n_ctx: int,
    n_ctx_source: str,
    vanta_bin: Path,
    refreeze_keys: set[str],
    missing_layers: set[str],
    missing_provenance: set[str],
    budget: budget_mod.Budget,
    traceback_on: bool,
) -> tuple[str, Any]:
    """One bug's prepare, under the invocation's one budget (M3 item Y1).

    [VANTA-WHY] M3 item AA4. This body lived inside ``cmd_prepare`` as the closure
    ``prepare_bug``; the pipelined queue needs the very same work — a bug prepared
    to a frozen record before its rows start — without a second copy of it, so it
    is a function here and ``cmd_prepare`` and ``_queue_pipeline`` both call it.
    It prints the same two progress lines the serial prepare always printed
    (``[key] full ...`` and then ``ok``/``not runnable``/``FAILED``), so a pipeline
    prints the same story a serial ``prepare`` does.

    Returns ``("prep", BugPrep)`` or ``("broken", (key, detail))``; nothing is
    raised, because one bug's failure must not take the batch down (§3.2).
    """
    key = bug_key(entry)
    project = entry["project"]
    print(f"[{key}] {'full' if full else 'context-only'} ...", flush=True)
    checkpoint = entry["checkpoint"]
    if full and checkpoint in missing_layers:
        detail = (
            f"PrepareError: checkpoint {checkpoint} does not exist. Bake it first "
            "(design §11 step 3 for py3.8, §3.2 for the older interpreters)."
        )
        print(f"[{key}] FAILED — {detail}", flush=True)
        return "broken", (key, detail)
    if full and checkpoint in missing_provenance:
        detail = (
            f"BakeError: no dependency provenance for {checkpoint} under {deps_dir}. "
            "Its dependencies could not be read at bake time; re-run bake for it."
        )
        print(f"[{key}] FAILED — {detail}", flush=True)
        return "broken", (key, detail)
    try:
        url = meta_mod.project_github_url(bugsinpy_root, project)
        if not url:
            raise SplitError(f"no github_url for {project} in project.info")
        repo = tree_cache(repo_cache, project, url)
        prep = prepare_mod.prepare_one(
            entry,
            split=split,
            bugsinpy_root=bugsinpy_root,
            repo_cache=repo,
            batch_dir=batch_dir,
            deps_dir=deps_dir,
            issue_sources=issue_sources,
            full=full,
            n_ctx=n_ctx,
            n_ctx_source=n_ctx_source,
            vanta_bin=vanta_bin,
            refreeze=key in refreeze_keys,
            budget=budget,
        )
    except Exception as exc:  # one bug must not take the batch down
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[{key}] FAILED — {detail}", flush=True)
        if traceback_on:
            traceback.print_exc()
        return "broken", (key, detail)
    print(f"[{key}] {'ok' if prep.runnable else 'not runnable'}", flush=True)
    return "prep", prep


def cmd_prepare(args: argparse.Namespace) -> int:
    # Item 19 / Q3 — the frozen configuration, resolved exactly as `run` resolves
    # it and before anything else. `resolve` REFUSES the invocation when the
    # caller's own environment contradicts a setting the file freezes to a
    # literal, unless `--allow-override` waives it, and it returns the row block
    # (`config_hash` and friends, written into `prepare.json` below) and the
    # `[env]` settings to apply. prepare used to read `os.environ` alone for the
    # context window (§1.6.1), so a shell that exported none of the frozen
    # settings — the normal case — failed the gate even though the file freezes
    # `VANTA_LLM_CONTEXT_TOKENS`: the gate sized against nothing while every child
    # of the batch would have been handed the frozen window. Resolving here is
    # what lets it size against the window the run will actually use.
    frozen = frozen_mod.load()
    config_block, frozen_env = frozen_mod.resolve(
        frozen, environ=os.environ, allow_override=args.allow_override
    )
    print(
        f"[config] {frozen.name} {frozen.sha256[:12]} ({config_block['config_path']})"
        + (
            "; --allow-override waives "
            + ", ".join(config_block["config_overrides"])
            if config_block["config_overrides"]
            else ""
        ),
        flush=True,
    )
    # M3 items S1–S3 — `--no-cache` before the first suite of this stage.
    honour_no_cache(args)
    # Item B1 — the ONE effective environment, the caller's own with the frozen
    # `[env]` settings applied exactly as `runner.vanta_env` will apply them to
    # each child. The window the gate resolves below is read from it, so a clean
    # shell sizes the gate against the frozen value rather than refusing.
    invocation_env = effective_env(frozen_env)
    split_path: Path = args.split.resolve()
    split = load_split(split_path)
    if args.refreeze and args.measure_only:
        # The two flags want opposite things from the frozen file: --measure-only reads the
        # text a full prepare wrote and boots nothing, while --refreeze has to re-fetch the
        # report and re-run the gate. Silently honouring one would leave the caller believing
        # the other had happened (§2.4.3).
        raise SplitError(
            "--refreeze rewrites a frozen issue report, and --measure-only freezes nothing: "
            "a measure-only pass reads the file a full prepare wrote. The re-fetch and the "
            "rewrite have to run with suites (§2.4.3), so drop --measure-only."
        )
    selected = select(
        split,
        only=args.only,
        waves=parse_waves(args.waves),
        reserves=args.reserves,
        measure_only=args.measure_only,
        refreeze=args.refreeze,
    )
    if not selected:
        raise SplitError("the selection is empty")

    if args.plan:
        sys.stdout.write(render_plan(selected))
        return EXIT_OK

    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    commit = meta_mod.ensure_clone(bugsinpy_root, commit=split.bugsinpy_commit)
    if commit != split.bugsinpy_commit:
        raise SplitError(
            f"BugsInPy is at {commit}, the split pins {split.bugsinpy_commit}. "
            "Refusing to derive ground truth from metadata that moved."
        )

    # The context window, resolved once for the whole batch and from an *explicit*
    # source only (§1.6.1). This happens before the first checkout on purpose: with
    # the window unresolved the gate would size against the 65536 fallback and
    # reject bugs for a prompt no run would ever send, and it would do that after
    # hours of suite runs rather than before them. Read from the invocation's own
    # environment (`invocation_env`, item Q3): the frozen `[env]` settings are what
    # every child gets, so a shell that exported none of them still sizes the gate
    # against the frozen window instead of refusing.
    n_ctx, n_ctx_source = context_mod.explicit_context_window(
        n_ctx=args.n_ctx, environ=invocation_env
    )
    print(
        f"context window: n_ctx={n_ctx} (source: {n_ctx_source}) — the context gate "
        "is sized against this window",
        flush=True,
    )

    # The context gate shells out to the run's own selector (§1.6.2 item 5), so the
    # binary is not optional here and it may not be stale: a gate that measured an
    # older Vanta's selection would pass a bug with a verdict the run then refutes
    # with a different prompt. `run` warns about this (§7.3); `prepare` refuses,
    # because the number it is about to write down is the gate's own output.
    vanta_bin = runner_mod.require_vanta_bin(args.vanta_bin.resolve())
    stale_warning = runner_mod.stale_binary_warning(vanta_bin)
    if stale_warning:
        raise runner_mod.RunnerError(
            f"{stale_warning} `prepare` cannot size the context gate against it: the "
            "gate shells out to this binary, so every verdict it writes would describe "
            "a Vanta the batch will not run."
        )

    # Preflight the checkpoints for everything that will run a suite, and the
    # dependency provenance the bake recorded, before the first checkout: a missing
    # layer is a batch-level mistake (§3.3) and a layer baked without a record is
    # one prepare must not trust. Both sets are computed here — so the expensive
    # part of prepare is never spent on a bug that cannot finish — but the
    # *response* is per bug inside the loop below (§3.2): a bug whose dependencies
    # could not be read at bake time has no layer and no record, and stopping the
    # whole split for it is what the flag exists to avoid.
    selected_full = [entry for entry, full in selected if full]
    missing_layers = set(
        prepare_mod.missing_checkpoints([entry["checkpoint"] for entry in selected_full])
    )
    deps_dir = RUNS_DIR / "deps"
    missing_provenance = set(
        bake_mod.missing_deps(deps_dir, [entry["checkpoint"] for entry in selected_full])
    )
    # M3 item U3 — a layer baked before the identity existed has a provenance file and no
    # ``checkpoint_id``, so the suite cache answers ``off:unknown_checkpoint`` for every
    # suite of it and reuses nothing. `prepare` does NOT fix that for itself: silently
    # stamping a layer is exactly how a layer that changed would keep an identity it no
    # longer deserves (the stamp is refused by a boot-verify, and only `stamp-existing`
    # performs one). It names the layers, the command, and — below, in the summary — the
    # cache state, so the cache can never be off without the batch saying so.
    selected_checkpoints = [entry["checkpoint"] for entry in selected_full]
    unstamped = bake_mod.unstamped_deps(deps_dir, selected_checkpoints)
    cache_state = cache_state_line(
        checkpoints=selected_checkpoints, unstamped=unstamped
    )
    if unstamped and not suite_cache.cache_disabled():
        print(
            f"WARNING: the suite cache is OFF for {len(unstamped)} layer(s) with no "
            f"checkpoint_id: {', '.join(unstamped)}.\n"
            "  Every suite run of those layers is measured and never reused. Fix them "
            "once, without re-baking any of them:\n"
            "    python3 scripts/bugsinpy_run.py stamp-existing\n"
            "  then re-run this prepare.",
            file=sys.stderr,
            flush=True,
        )

    batch_dir: Path = (args.batch or (RUNS_DIR / f"prepare-{utc_stamp()}")).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = RUNS_DIR / "repos"
    # The frozen issue reports live under the directory this invocation named: the
    # committed `plans/bugsinpy_issues` by default (§2.4 — the dev bugs' reports are
    # committed), or `--issues-dir` as given (a held-out run passes the gitignored
    # `plans/private/bugsinpy_issues`, so its report text is frozen and verifiable
    # without being published). The API cache is machine-local and stays under
    # `runs/`, which `.gitignore` already ignores.
    issue_sources = prepare_mod.IssueSources(
        issues_dir=prepare_issues_dir(args),
        github_cache_dir=RUNS_DIR / "github-cache",
    )

    preps: list[prepare_mod.BugPrep] = []
    broken: list[tuple[str, str]] = []
    refreeze_keys = set(args.refreeze)

    # [VANTA-WHY] M3 item S5 — the ONE scheduler of this invocation, created here and
    # handed to every bug's suite runs. Default 1 keeps today's behaviour exactly
    # (one suite run at a time); a higher `--prepare-jobs` lets a bug's independent
    # derivations overlap, capped at host cores − 2 so the machine that owns the
    # checkpoints and the batch directory keeps room. It is created per invocation
    # rather than at import, so the recorded host load is this run's and no state
    # survives from an earlier `prepare` in the same process.
    budget = budget_mod.Budget(args.prepare_jobs, peak_vms=args.peak_vms)
    print(
        f"[scheduler] jobs={budget.jobs} concurrent={'yes' if budget.concurrent else 'no'} "
        f"cores={budget.cores}/{budget.cap_cores} memory="
        f"{'unbounded' if budget.cap_memory_mb is None else f'{budget.cap_memory_mb} MB'} "
        f"host_load={budget.host_load}",
        flush=True,
    )

    # [VANTA-WHY] M3 item Y1 — different bugs are independent, so every bug is
    # submitted at once and the ONE budget decides how many overlap, exactly as it
    # does for a bug's own derivations. Submitting them adds no concurrency the
    # operator did not ask for: at `--prepare-jobs 1` the budget is a single core,
    # so the gate admits one suite run at a time and a bug's second VM waits for the
    # first's slot to free; the pool below is `max(1, budget.cores)` workers, which
    # is one at that setting and therefore the same serial loop, in the same order,
    # producing the same `preps` and `broken` — the worker's index decides a bug's
    # place in both lists, never its completion order. This is what lets bug A's
    # vanta run (which starts when A's oracle pair lands, its real dependency)
    # overlap bug B's still-running derivations instead of waiting for the whole
    # batch to reach A's turn.
    slots: list[Optional[tuple[str, Any]]] = [None] * len(selected)

    def prepare_bug(index: int, entry: Mapping[str, Any], full: bool) -> None:
        """One bug's prepare, under the invocation's one budget (M3 item Y1)."""
        slots[index] = prepare_one_bug(
            entry=entry,
            full=full,
            split=split,
            bugsinpy_root=bugsinpy_root,
            repo_cache=repo_cache,
            batch_dir=batch_dir,
            deps_dir=deps_dir,
            issue_sources=issue_sources,
            n_ctx=n_ctx,
            n_ctx_source=n_ctx_source,
            vanta_bin=vanta_bin,
            refreeze_keys=refreeze_keys,
            missing_layers=missing_layers,
            missing_provenance=missing_provenance,
            budget=budget,
            traceback_on=args.traceback,
        )

    with ThreadPoolExecutor(max_workers=max(1, budget.cores)) as pool:
        futures = [
            pool.submit(prepare_bug, index, entry, full)
            for index, (entry, full) in enumerate(selected)
        ]
        for future in futures:
            future.result()

    for slot in slots:
        if slot is None:
            continue
        kind, value = slot
        if kind == "prep":
            preps.append(value)
        else:
            broken.append(value)

    for prep in preps:
        if prep.issue is not None:
            prepare_mod.write_leak_gate(batch_dir, prep)
    prepare_mod.write_issue_texts(batch_dir, preps)
    prepare_mod.write_context_sizes(batch_dir, preps)
    prepare_mod.write_signals(batch_dir, preps)
    prepare_mod.write_context_report(
        batch_dir, preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source
    )

    summary = prepare_mod.render_summary(
        preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source, cache_state=cache_state
    )
    (batch_dir / "prepare_summary.md").write_text(summary, encoding="utf-8")
    # The split is an input, never an output: how many bugs this run measured, and
    # which of them it rejected, belongs in the batch's own artifacts. Rehashing the
    # file after the work is the evidence that nothing here wrote to it.
    split_sha_before = split_mod.sha256_file(split_path)
    manifest = prepare_mod.prepare_manifest(
        split=split,
        split_path=split_path,
        split_sha_before=split_sha_before,
        batch_dir=batch_dir,
        commit=commit,
        bugsinpy_root=bugsinpy_root,
        selected=[bug_key(entry) for entry, _ in selected],
        broken=broken,
        preps=preps,
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source,
        config_block=config_block,
        scheduler=budget.record(),
    )
    (batch_dir / "prepare.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    split_sha_after = split_mod.sha256_file(split_path)
    if split_sha_after != split_sha_before:
        raise SplitError(
            f"{split_path} changed during prepare ({split_sha_before} → "
            f"{split_sha_after}). The split is an input; something wrote to it."
        )

    print()
    print(summary)
    print()
    print(prepare_mod.render_context_table(preps))
    if broken:
        print()
        print(render_flagged(broken))
    print()
    print(f"batch dir : {batch_dir}")
    print(f"split     : {split_path} (read-only; {split_sha_after[:12]} unchanged)")
    for artifact in (
        "context_report.json",
        "context_sizes.md",
        "signals.md",
        "issue_texts_all.md",
        "prepare_summary.md",
    ):
        print(f"  {artifact}")

    failed = [prep for prep in preps if not prep.runnable] + [
        None for _ in broken
    ]
    return EXIT_BUG_FAILED if failed else EXIT_OK


# ---------------------------------------------------------------------------
# bake — environment layers (§3, §11 step 3)
# ---------------------------------------------------------------------------
def parse_pythons(raw: str, split: Split) -> list[str]:
    if not raw:
        return bake_mod.pythons_in_split(split)
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def cmd_bake(args: argparse.Namespace) -> int:
    """Create the interpreter and per-bug checkpoints the split names.

    The only stage that needs network (``--allow-net``, inside Shuru) and the only
    one that runs before ``prepare``. Without ``--yes`` it prints the plan and
    stops: a bake downloads interpreters and installs packages for minutes, and a
    typo should not start one.

    Every ``--yes`` run starts with the preflight (§3.2): what each pin installs as
    on this image — written next to the provenance and printed before the first layer
    is built — and what each pin *declares* against the layer's other pins, which is
    §3.2's case (a)/(b): a pair pip would refuse to install is then visible before the
    build rather than as a layer that failed for a reason the summary had to guess at.
    ``--preflight-only`` stops after it, and does not need ``--yes`` — the preflight
    builds nothing, and the pins it classifies are the same ones either way.

    Every layer the build attempts leaves its whole output in this run's own
    ``runs/bugsinpy/bake-logs/<UTC stamp>/<layer>.log`` (:func:`bake.bake_logs_dir`),
    which is never emptied — an earlier bake's logs, the failed ones especially, stay
    readable — and each failed layer's one-line entry in the failure report names its
    own error and points at that file.
    """
    split = load_split(args.split.resolve())
    pythons = parse_pythons(args.pythons, split)
    roles = {"pilot", "reserve"} if args.reserves else {"pilot"}
    minors, entries = bake_mod.select_layers(
        split, pythons=pythons, only=args.only, roles=roles
    )
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()

    if not args.yes and not args.preflight_only:
        sys.stdout.write(
            bake_mod.render_plan(
                split, pythons=minors, entries=entries, bugsinpy_root=bugsinpy_root
            )
        )
        if not args.plan:
            print(
                "dry run: nothing was baked. Pass --yes to build these layers "
                "(Shuru, --allow-net, minutes per interpreter).",
                file=sys.stderr,
            )
        return EXIT_OK

    if not bugsinpy_root.is_dir():
        raise SplitError(
            f"no BugsInPy clone at {bugsinpy_root}; run `prepare --plan` first or "
            "clone it, so per-bug requirements can be read"
        )

    # Dependencies are resolved (and their provenance written) before anything is
    # baked, so a bug whose declared sources contradict each other, or that names a
    # release which did not exist yet at its commit date, is caught here rather than
    # producing a layer nobody can defend. This is the only place the harness talks
    # to PyPI; every decision it makes is taken from the dates that call returns.
    #
    # A bug that cannot be read is *flagged*, not fatal (§3.2): stopping the whole
    # bake at the first one hides every later bug behind it and produces nothing.
    # The resolvable bugs are baked, the flagged list is printed at the end, and the
    # exit code says a bug did not bake.
    repo_cache = RUNS_DIR / "repos"
    deps_dir = RUNS_DIR / "deps"

    def repo_for(project: str) -> Path:
        url = meta_mod.project_github_url(bugsinpy_root, project)
        if not url:
            raise SplitError(f"no github_url for {project} in project.info")
        return tree_cache(repo_cache, project, url)

    # One classifier for the whole run, shared by the two places that ask the same
    # question about the same pins: §3.2's unresolvable-pin rule (does this release
    # install here at all, which decides whether a pin is dropped or the bug flagged)
    # and the preflight table below. A second classifier would pay PyPI a second time
    # for every answer, and the splits share their pins heavily — numpy alone is in six
    # of the eleven dev-split bugs.
    classify = bake_mod.preflight_classifier()
    # One reader for the other question the two places ask about the same pins: what a
    # release *declares*. §3.2's date rule reads it when an exact pin narrows a name the
    # rule would otherwise pick freely, and the preflight's case-(a)/(b) check reads it
    # for every pin of every layer. Sharing one instance means one fetch per
    # ``name==version`` for the whole run, which matters because the two readings cover
    # the same pins and the splits share them.
    metadata_of = bake_mod.cached_release_metadata()
    # And one for the third question, which costs the most per answer: what a release
    # needs in order to be *built*. It is not in ``requires_dist`` or any wheel's
    # metadata — it is in the sdist's own ``pyproject.toml``/``setup.py`` — so this
    # reader downloads a whole sdist, and a pin the layers share being read once is the
    # difference between one download and several. Read by the resolution (which turns
    # it into each layer's ``early_pins``) and by the preflight table below.
    build_requires_of = bake_mod.cached_build_requirements()
    bug_deps, flagged = bake_mod.resolve_bug_deps(
        split,
        entries,
        bugsinpy_root=bugsinpy_root,
        repo_for=repo_for,
        classify=classify,
        metadata_of=metadata_of,
        build_requires_of=build_requires_of,
        # The VM every layer of this run is built in, recorded in each bug's provenance
        # so a layer that failed a source build can be told from one the VM was too
        # small for (§3.2). One decision for the run, like --disk-mb below it.
        build_vm=bake_mod.build_vm_record(args.cpus, args.memory),
    )

    # Every run ends with this list, empty or not: a reader of a run that stopped at a
    # failed build saw the plan and the layer that failed but had no way to tell
    # whether other bugs had been flagged before the build started. The `(0)` line is
    # the same statement made about a run where nothing was flagged.
    flagged_pairs = [(flag.key, flag.reason) for flag in flagged]

    def report_flagged() -> None:
        print(flush=True)
        print(render_flagged(flagged_pairs), flush=True)

    # A dropped pin is the one thing about a layer that a reader cannot see from the
    # install list alone — the declaration is gone and the layer is shorter than the
    # bug's own files say. It is not printed here per bug: the same pin is usually
    # dropped from several bugs, and the preflight below prints each drop once for the
    # whole split with the audit that settled it. The records are collected here, from
    # the provenance just written, so the two views are the same data.
    drops = {item.checkpoint: item.provenance.get("dropped") or [] for item in bug_deps}
    for item in bug_deps:
        path = bake_mod.write_deps(deps_dir, item)
        # The three per-layer lines are the build's own log, so a preflight-only run does
        # not print them: it read them to answer, and what it is for is the findings
        # below. They stay in the deps JSON either way, so nothing is lost by not
        # repeating the layer's install list once per bug before the run has started.
        if not args.preflight_only:
            print(f"[{item.checkpoint}] installs: {', '.join(item.installs) or '—'}")
            print(
                f"[{item.checkpoint}] build pins: {', '.join(item.build_pins) or '—'}",
                flush=True,
            )
            print(
                f"[{item.checkpoint}] early pins: {', '.join(item.early_pins) or '—'}",
                flush=True,
            )
            print(f"[{item.checkpoint}] provenance: {path}", flush=True)

    # The preflight, before the first layer is built (§3.2): what every pin installs
    # as on this image — cp38/linux-aarch64. A pin with no wheel here is built from
    # its sdist, and every system library that build needs is a whole bake cycle to
    # discover, so the answer is computed, printed and recorded next to the
    # provenance before anything is built.
    preflight = bake_mod.preflight_layers(
        bug_deps,
        classify=classify,
        metadata_of=metadata_of,
        build_requires_of=build_requires_of,
    )
    for report in preflight:
        path = bake_mod.write_preflight(deps_dir, report)
        print(f"[{report.checkpoint}] preflight: {path}", flush=True)
    if preflight:
        print("\n".join(bake_mod.preflight_lines(preflight, dropped=drops)), flush=True)

    if args.preflight_only:
        print(
            bake_mod.preflight_verdict(preflight, dropped=drops, flags=len(flagged)),
            flush=True,
        )
        print("preflight only: nothing was baked", flush=True)
        report_flagged()
        return EXIT_BUG_FAILED if flagged else EXIT_OK

    # A flagged bug has no resolved deps, so it must not reach the recipes — they
    # would refuse a checkpoint with nothing resolved for it, which is exactly the
    # stop the flag exists to avoid.
    flagged_keys = {flag.key for flag in flagged}
    baked_entries = [entry for entry in entries if bug_key(entry) not in flagged_keys]
    layers = bake_mod.layer_recipes(
        split, pythons=minors, entries=baked_entries, deps=bug_deps
    )

    # --force means "replace", which for shuru means delete-then-create, and a
    # rebuilt interpreter layer makes every layer stacked on it stale. Both are
    # decided up front (bake.plan_layers) so the deletions happen children-first and
    # a layer this run is not going to rebuild is never deleted for another reason.
    children = bake_mod.children_of(split)
    actions = bake_mod.plan_layers(layers, force=args.force, children=children)
    print(f"bake plan ({len(actions)} layers):", flush=True)
    for action in actions:
        print(f"  {action.describe()}", flush=True)

    # Every layer's whole build log is written here by run_bake_actions. This run gets
    # its own timestamped folder (bake.bake_logs_dir) and nothing in it is ever deleted,
    # so a layer that failed can be read in full and an earlier bake's logs — the failed
    # ones especially — are never wiped by this one.
    logs_dir = bake_mod.bake_logs_dir(RUNS_DIR)
    # §3.2's own-pin gate needs to know which pins are BugsInPy's own for the layer
    # being built, and to write back what the layer's own ``pip check`` found. Both are
    # per checkpoint and both come from the provenance this run just wrote.
    own_pins = {
        item.checkpoint: bake_mod.own_pin_names(item.provenance) for item in bug_deps
    }
    # The freeze gate's two halves, from the same provenance: the layer's own pin list,
    # which is what the layer's ``pip freeze`` has to be a subset of, and the names the
    # interpreter layer supplies it with, which no pin of the bug's mentions and which
    # every layer therefore inherits without pinning.
    pin_lists = {item.checkpoint: bake_mod.layer_pins(item) for item in bug_deps}
    provided = {
        item.checkpoint: item.provenance.get("provisioned") or ()
        for item in bug_deps
    }
    # The names §3.2 dropped for each layer, which both gates read by name: the own-pin
    # gate tolerates the ``pip check`` reading that names one and fails on any other
    # missing requirement, and the freeze gate fails on a dropped name at any version
    # (bake.CASE_DROPPED_DEPENDENCY, bake.unpinned_freeze). Read from the same
    # provenance: the recipe installs the pins that declare one with ``--no-deps``, so
    # a dropped name in the layer is the pass not having taken.
    dropped = {
        item.checkpoint: tuple(
            pin["name"] for pin in item.provenance.get("dropped") or ()
        )
        for item in bug_deps
    }
    # The names the tree under test provides, read from the same provenance
    # (bake.TREE_SATISFIED_RULE): the walk pins nothing for them, the recipe installs the
    # pins that declare one with ``--no-deps`` so pip cannot resolve them either, and the
    # own-pin gate therefore tolerates the ``pip check`` reading that names one
    # (bake.CASE_TREE_PROVIDED) while failing on any other missing requirement.
    tree_provided = {
        item.checkpoint: tuple(
            record["name"] for record in item.provenance.get("tree_satisfied") or ()
        )
        for item in bug_deps
    }
    # §8.6 R2(ii)'s two callables. The retry is asked once about a per-bug layer whose
    # build *failed*, with that layer's own merged output, and answers a second recipe
    # only when the failure names pins this rule may drop — a log that names none (R1's
    # ``setup.py egg_info`` crash), a pin §8.6's safeguard protects and a pin the graded
    # tests reach all answer None and leave the ordinary failure path alone. The material
    # it needs came through resolve_bug_deps on each BugDeps, so the closure and the
    # safeguard are the same two this layer was resolved with and not a second opinion.
    # record_drops writes what it dropped, and why, into that layer's own provenance.
    deps_by_checkpoint = {item.checkpoint: item for item in bug_deps}

    def retry_layers(name: str, built: bake_mod.ExecResult) -> Optional[bake_mod.RetryPlan]:
        item = deps_by_checkpoint.get(name)
        if item is None:
            return None
        return bake_mod.retry_plan(item, built.stdout + built.stderr)

    try:
        summary = bake_mod.run_bake_actions(
            actions,
            children=children,
            build=lambda name, script, base: bake_mod.bake(
                name,
                script,
                base=base,
                disk_mb=args.disk_mb,
                cpus=args.cpus,
                memory_mb=args.memory,
            ),
            logs_dir=logs_dir,
            own_pins=own_pins,
            record=lambda name, conflicts: bake_mod.record_pip_check(
                deps_dir, name, conflicts
            ),
            pins=pin_lists,
            provided=provided,
            dropped=dropped,
            tree_provided=tree_provided,
            retry=retry_layers,
            record_drops=lambda name, plan: bake_mod.record_retry(deps_dir, name, plan),
        )
    except BaseException:
        report_flagged()
        raise
    print(summary, flush=True)
    report_flagged()
    print("bake complete", flush=True)
    return EXIT_BUG_FAILED if flagged else EXIT_OK


# ---------------------------------------------------------------------------
# stamp-existing — an identity for the layers baked before U3 (M3 item U3)
# ---------------------------------------------------------------------------
def render_stamp_table(outcomes: Sequence[bake_mod.StampOutcome]) -> str:
    """One line per deps file: what it is, its identity, and what the boot said.

    [VANTA-WHY] M3 item U3 — the digest is printed beside the identity so the reader can
    check the invariant by hand (``checkpoint_id == sha256("<baked_at>\\x00<digest>")``),
    and the two are truncated because they are the same 64 hex characters in every row
    and the full values are in the files. ``baked`` versus ``stamped`` is the column that
    says whether a layer's identity came from a bake or from this command, which is the
    only way to tell a layer this step touched from one it skipped.

    ``interpreter`` and ``python`` are what the verify actually queried, read back from
    the boot rather than assumed: the pre-registration records *what was checked*, and a
    row that showed only "ok" would not say through which environment.
    """
    lines = [
        "| project | checkpoint | checkpoint id | digest | baked/stamped | "
        "interpreter | python | verify |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        lines.append(
            "| {project} | {checkpoint} | {cid} | {digest} | {origin} | {interp} | "
            "{pyver} | {verify} |".format(
                project=outcome.project or "—",
                checkpoint=outcome.checkpoint,
                cid=(outcome.checkpoint_id[:12] + "…")
                if outcome.checkpoint_id
                else "—",
                digest=(outcome.digest[:12] + "…") if outcome.digest else "—",
                origin=outcome.origin,
                interp=outcome.interpreter or "—",
                pyver=outcome.python_version or "—",
                verify=outcome.verify or "—",
            )
        )
    return "\n".join(lines)


def layer_interpreters() -> dict[str, str]:
    """``{project: test_cmd[0]}`` over both splits — the interpreter each suite runs.

    [VANTA-WHY] ``stamp-existing`` verifies a layer by asking the interpreter *inside it*
    what it can see installed, and the only answer that means anything is the one from the
    interpreter ``test_cmd[0]`` names — the one ``suite.run_suite_twice`` runs. Every
    project names bare ``python`` today, so this map only ever confirms the default; it
    exists so that a split which later names another interpreter makes this command
    *refuse* that layer rather than verify a different environment and stamp the layer on
    that basis. A project in neither split maps to nothing, and a layer of it is refused
    for the same reason: the interpreter is not knowable, so it is not guessed.
    """
    interpreters: dict[str, str] = {}
    for path in (DEV_SPLIT, DEFAULT_SPLIT):
        if not path.is_file():
            continue
        split = load_split(path)
        for entry in split.bugs:
            test_cmd = entry.get("test_cmd") or []
            if test_cmd:
                interpreters[str(entry["project"])] = str(test_cmd[0])
    return interpreters


def cmd_stamp_existing(args: argparse.Namespace) -> int:
    """Give every layer baked before M3 item U3 an identity, without re-baking it.

    The layers this fixes were baked before the stamp existed, so their provenance has no
    ``checkpoint_id`` and the suite cache reads ``off:unknown_checkpoint`` for each of
    them: every suite is measured and nothing is reused. Each layer is booted once,
    read-only, and asked — through the same interpreter its suite runs
    (:func:`layer_interpreters`) — what it can see installed; that listing is compared
    against the pins its provenance records. A layer that no longer matches is *left
    unstamped* and reported, not re-baked and not stopped for, so ``prepare`` keeps
    warning about it and its cache stays off. A layer whose checkpoint is simply gone is
    reported ``missing`` and is not a failure: there is nothing to act on. ``prepare``
    never does this for itself, on purpose: a stamp is an identity, and handing one to a
    layer that changed is the failure the boot-verify exists to refuse.

    A boot-verify that could not *read* the environment — no interpreter, a listing that
    failed or came back empty — refuses the layer with :data:`bake_mod.ENV_UNREADABLE`
    rather than reporting a mismatch, because an unread layer is a probe problem and a
    mismatch is a layer problem, and they want different responses.

    Writes exactly where ``bake.write_deps`` writes, in the same shape, so a layer's
    identity is fixed from here on. A later re-bake of the same layer still gets a fresh
    identity, because a bake never records the ``stamped_existing`` marker this step does.
    """
    deps_dir: Path = RUNS_DIR / "deps"
    if not deps_dir.is_dir():
        raise bake_mod.BakeError(
            f"no dependency provenance directory at {deps_dir}: nothing has been baked "
            "here, so there is no layer to stamp."
        )
    print(f"stamp-existing: {deps_dir}", flush=True)
    outcomes = bake_mod.stamp_existing(deps_dir, interpreters=layer_interpreters())
    if not outcomes:
        print("no deps files found — nothing to stamp.", flush=True)
        return EXIT_OK
    print(render_stamp_table(outcomes), flush=True)
    stamped = [o for o in outcomes if o.action == "stamped"]
    already = [o for o in outcomes if o.action == "already"]
    refused = [o for o in outcomes if o.action == "refused"]
    missing = [o for o in outcomes if o.action == "missing"]
    skipped = [o for o in outcomes if o.action == "not a layer's provenance"]
    print(
        f"\nstamped {len(stamped)}, already {len(already)}, missing {len(missing)}, "
        f"refused {len(refused)}, not a layer {len(skipped)}",
        flush=True,
    )
    if refused:
        print(
            "\nrefused layers were left exactly as they were, so `prepare` keeps warning "
            "about them and their cache stays off. Re-bake them (their environment no "
            "longer matches its record) or investigate the difference above.",
            file=sys.stderr,
            flush=True,
        )
    if missing:
        print(
            "missing layers have a provenance record but no checkpoint: there is nothing "
            "to stamp, and no cache entry to key.",
            flush=True,
        )
    if stamped:
        print(
            "the suite cache can now key every stamped layer; re-run `prepare`.",
            flush=True,
        )
    return EXIT_BUG_FAILED if refused else EXIT_OK


def tree_cache(repo_cache: Path, project: str, url: str) -> Path:
    """One upstream clone per project, reused across batches and bugs.

    Returned as a path, not a URL, because ``trees.export_tree`` clones *from* it:
    three checkouts of the same project (buggy, buggy+fixed-tests, fixed) then cost
    one network clone, and a reboot cannot lose a half-fetched repository.
    """
    from bugsinpy import trees as tree_mod

    return tree_mod.ensure_project_clone(repo_cache, project, url)


def parse_waves(raw: str) -> list[int]:
    if not raw:
        return []
    try:
        return [int(part) for part in raw.replace(",", " ").split()]
    except ValueError as exc:
        raise SplitError(f"--waves must be comma-separated integers, got {raw!r}") from exc


# ---------------------------------------------------------------------------
# run — the batch runner (§7.1, §7.4)
# ---------------------------------------------------------------------------
def select_preps(
    preps: list[prepare_mod.BugPrep],
    *,
    only: list[str],
    waves: list[int],
) -> list[prepare_mod.BugPrep]:
    """The bugs this invocation runs, in the split's own file order.

    Only *runnable* preps are eligible. ``prepare`` may have recorded a bug as a
    reserve (context-measured, its suites never run) or failed its gate, and running
    one of those would either grade against a tree with no derived oracle or fail
    inside Vanta for a reason the harness already knows. Refusing loudly beats a
    wave that quietly looks short.
    """
    chosen = preps
    if only:
        wanted = set(only)
        unknown = sorted(wanted - {prep.key for prep in chosen})
        if unknown:
            raise SplitError(f"--only names bugs that are not in this batch: {unknown}")
        chosen = [prep for prep in chosen if prep.key in wanted]
    if waves:
        wanted_waves = set(waves)
        chosen = [prep for prep in chosen if prep.wave in wanted_waves]
    if not chosen:
        raise SplitError("the selection is empty")
    blocked = [
        f"{prep.key} (prepare_mode={prep.prepare_mode}, gate: {'; '.join(prep.gate_failures) or 'none'})"
        for prep in chosen
        if not prep.runnable
    ]
    if blocked:
        raise SplitError(
            "these bugs did not prepare as runners, so a run against them would "
            f"grade a tree with no derived oracle: {blocked}"
        )
    return chosen


def effective_env(
    frozen_env: Mapping[str, str], *, base: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Item B1 — the environment a run's OWN checks and its children both see.

    [VANTA-WHY] The frozen ``[env]`` settings are applied by the harness
    (``runner.vanta_env``) to every child, on top of the caller's own
    environment. Before this, the two reads the invocation makes about its OWN
    configuration — §14.29's spend cap (``VANTA_MAX_SPEND_USD``) and item 14's
    endpoint preflight (``VANTA_LLM_BASE_URL``) — read ``os.environ`` alone, so a
    setting frozen in the file but not exported was honoured by the children and
    ignored by the checks meant to guard them: a frozen ``VANTA_MAX_SPEND_USD``
    left the batch *uncapped*. The effective environment is the one value used
    for both, exactly as ``vanta_env`` layers them (frozen wins over the caller).
    """

    return {**(os.environ if base is None else base), **frozen_env}


def max_spend_usd_from_env(env: Mapping[str, str]) -> Optional[float]:
    """§14.29 — ``VANTA_MAX_SPEND_USD``: the batch's budget, or ``None`` for no cap.

    Read here rather than inside ``run_batch`` so a test that drives the batch
    loop cannot be capped by whatever the developer happens to have exported:
    the cap belongs to the *invocation*, and this CLI is the invocation. Unset
    or empty means no cap, which is how every batch before this change ran. A
    value that is not a positive number is refused rather than ignored — a typo
    in a safety net must not quietly remove the net.
    """
    raw = (env.get("VANTA_MAX_SPEND_USD") or "").strip()
    if not raw:
        return None
    try:
        cap = float(raw)
    except ValueError:
        raise runner_mod.RunnerError(
            f"VANTA_MAX_SPEND_USD='{raw}' is not a number; unset it for no cap."
        )
    if not cap > 0:
        raise runner_mod.RunnerError(
            f"VANTA_MAX_SPEND_USD='{raw}' must be > 0; unset it for no cap."
        )
    return cap


def row_budget(
    args: argparse.Namespace, env: Mapping[str, str], *, flag: str
) -> budget_mod.Budget:
    """M3 item S6 — the ONE scheduler of a ``run``/``queue`` invocation.

    [VANTA-WHY] §14.33's closing rule: "``Default stays 1``, and ``1`` must be
    byte-identical to today". So a ``--jobs 1`` budget is created but never
    advertises itself — no ``[scheduler]`` line and, because
    :func:`runner.run_batch` guards its row block on ``budget.concurrent``, no
    ``scheduler`` key on any row. A serial batch is therefore the batch it was
    before this item, and the pool is a path ``--jobs 1`` does not enter rather
    than one that degenerates to serial.

    Above one, the backend gate is §14.33's second rule: parallel rows all call
    the same provider, so only the hosted backend may run them — every other
    backend is a local model that would be oversubscribed by its own concurrency.
    The check is on the *requested* jobs, before the host's core cap, so a
    smaller machine refuses ``--jobs 4`` for the same reason a large one does.
    """
    jobs = int(args.jobs)
    if jobs < 1:
        raise runner_mod.RunnerError(f"{flag} must be at least 1, not {jobs}.")
    if jobs > 1:
        backend = (env.get("VANTA_LLM_BACKEND") or "").strip()
        if backend.lower() != "openai":
            raise runner_mod.RunnerError(
                f"{flag}={jobs} needs VANTA_LLM_BACKEND=openai: concurrent rows all "
                "call the same provider (§14.33), and every other backend is a local "
                "model that this concurrency would oversubscribe. "
                f"VANTA_LLM_BACKEND={backend or '(unset)'}."
            )
    budget = budget_mod.Budget(jobs, peak_vms=args.peak_vms)
    if budget.concurrent:
        print(
            f"[scheduler] jobs={budget.jobs} concurrent=yes "
            f"cores={budget.cores}/{budget.cap_cores} memory="
            f"{'unbounded' if budget.cap_memory_mb is None else f'{budget.cap_memory_mb} MB'} "
            f"peak_vms={budget.peak_vms} host_load={budget.host_load}",
            flush=True,
        )
    return budget


def _start_host_sampler(
    batch_dir: Path, budget: budget_mod.Budget
) -> Optional[budget_mod.HostSampler]:
    """M3 item S6 — sample the host only when rows will actually overlap.

    A serial batch gains neither the file nor the thread: §14.33's ``--jobs 1``
    must be byte-identical to today, and a sampler that ran would write a new
    artifact into a batch directory that is supposed to be unchanged.
    """
    if not budget.concurrent:
        return None
    return budget_mod.HostSampler(batch_dir / budget_mod.HOST_SAMPLES_NAME).start()


def _stop_host_sampler(sampler: Optional[budget_mod.HostSampler]) -> Optional[dict]:
    """Stop the sampler and return its summary, or ``None`` when there was none."""
    if sampler is None:
        return None
    sampler.stop()
    return sampler.record()


def honour_no_cache(args: argparse.Namespace) -> None:
    """``--no-cache`` — force a fresh suite run, in BOTH processes of the stage.

    [VANTA-WHY] M3 items S1–S3. A suite is run by two processes and each looks for
    the cache by the same environment variable: the harness process itself (item
    S3's ``prepare`` gate and item S2's ``oracle.grade``, both in-process, reading
    ``suite_cache.cache_disabled()``) and the ``vanta_m3`` child it spawns (item
    S1's in-session native gate, reading ``suite_cache.rs``'s ``DISABLE_ENV``).
    Setting it once here covers both, because ``runner.vanta_env`` builds the
    child's environment from ``os.environ`` and only ever ADDS keys — and the
    frozen ``[env]`` does not name this one, so nothing overrides it.

    Called AFTER ``frozen_config.resolve``: that function refuses an invocation
    whose environment contradicts ``configs/v1-m3.toml``, and it reads
    ``os.environ``. A flag-set variable the file does not freeze is not a
    contradiction, but resolving first keeps this function from ever being the
    reason a refusal fires.

    The flag does not HIDE the cache: ``suite_cache.read_entry`` is still
    consulted so the row can record that the entry would have hit
    (``off:no_cache``, with ``would_hit`` in the row's own ``cache`` list), and
    the fresh payload then replaces the entry it could not use (M3 item U2) —
    unless the fresh run measured nothing, which `run_state` refuses on both
    sides. That is what makes the cache verifiable on demand.
    """
    if getattr(args, "no_cache", False):
        os.environ[suite_cache.DISABLE_ENV] = "1"


def cmd_run(args: argparse.Namespace) -> int:
    """Run a prepared batch: repairs first, then both reviews, one job at a time.

    The repo-root check comes before anything else. ``shuru run --mount`` requires
    the mounted host path to be inside the process CWD, so a run started from the
    wrong directory would mount nothing and fail per job instead of once.
    """
    runner_mod.require_repo_cwd()
    # Item 19 — the frozen configuration, resolved before anything else runs.
    # `load` reads `configs/v1-m3.toml`; `resolve` REFUSES the run when the caller's
    # own environment contradicts a setting the file freezes to a literal, unless
    # `--allow-override` waives it — and the waiver, with the values it waived, is
    # recorded on every row. It returns the row block (`config_hash` and friends)
    # and the `[env]` settings to apply. Computed once, here, because they describe
    # the whole invocation: one batch has one configuration.
    frozen = frozen_mod.load()
    config_block, frozen_env = frozen_mod.resolve(
        frozen, environ=os.environ, allow_override=args.allow_override
    )
    print(
        f"[config] {frozen.name} {frozen.sha256[:12]} ({config_block['config_path']})"
        + (
            "; --allow-override waives "
            + ", ".join(config_block["config_overrides"])
            if config_block["config_overrides"]
            else ""
        ),
        flush=True,
    )
    # M3 items S1–S3 — `--no-cache` before the first job of this stage.
    honour_no_cache(args)
    # Item B1 — the ONE effective environment: the caller's own, with the frozen
    # `[env]` settings applied exactly as `runner.vanta_env` will apply them to
    # each child. Every read this invocation makes about its OWN configuration
    # must see it — item 14's endpoint preflight (reads `VANTA_LLM_BASE_URL`) and
    # §14.29's spend cap (reads `VANTA_MAX_SPEND_USD`) — or a setting frozen in
    # the file but not exported is honoured by the children and ignored by the
    # checks that guard them. Built here, before the preflight, which used to run
    # first against `os.environ` alone.
    invocation_env = effective_env(frozen_env)
    # Item 14 — the same shape, and the same reason, as the repo-root check above:
    # a run against a model endpoint that is not there fails one job at a time, and
    # the only cheap moment to say so is before the first one. `--plan` is exempt
    # because printing the schedule sends no request, so it must not need one.
    if not args.plan:
        runner_mod.require_model_endpoint(
            preflight=partial(runner_mod.model_endpoint_preflight, env=invocation_env)
        )
    split = load_split(args.split.resolve())
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    # Item T3 — the preparation resolved the frozen configuration too and recorded it
    # in `prepare.json`; a run that contradicts it is a contradiction only
    # `--allow-override` waives, checked here so the refusal lands before the first
    # job rather than mid-wave. `--plan` runs the same check: the schedule a plan
    # prints would be a schedule for a batch nobody is allowed to run.
    config_mismatch = runner_mod.prepared_config_mismatch(batch.config_block, config_block)
    if config_mismatch and not args.allow_override:
        raise runner_mod.RunnerError(config_mismatch)
    # Before anything runs, and before --plan prints: a run whose window differs
    # from the one prepare sized against is measuring a different prompt than the
    # batch's verdicts describe (§1.6.1), and the cheap moment to notice is now. The
    # comparison reads the same effective environment every child inherits (item T5),
    # so a setting frozen in the file but not exported is not read as a divergence.
    window_warning = runner_mod.context_window_warning(
        batch, environ=invocation_env
    )
    if window_warning:
        print(f"WARNING: {window_warning}", file=sys.stderr, flush=True)
    waves = parse_waves(args.waves)
    selected = select_preps(batch.preps, only=args.only, waves=waves)
    vanta_bin = runner_mod.require_vanta_bin(args.vanta_bin.resolve())
    # Also before anything runs: the rows record two commits (§7.3), and a binary
    # built before the Rust sources moved on is measuring a Vanta that is not in
    # this repo. The check is cheap and it is the only moment it can still be acted
    # on — mid-wave it is just a note about results already produced.
    stale_warning = runner_mod.stale_binary_warning(vanta_bin)
    if stale_warning:
        print(f"WARNING: {stale_warning}", file=sys.stderr, flush=True)

    # §14.29 — the batch's budget, if the caller set one. Announced before the
    # first job so the number a run is capped at is on the record at the top of
    # the log rather than only in the environment that invoked it. Read from the
    # effective environment (item B1), so a cap frozen in `[env]` is honoured even
    # when the caller did not export it.
    max_spend_usd = max_spend_usd_from_env(invocation_env)
    if max_spend_usd is not None:
        print(
            f"[spend] VANTA_MAX_SPEND_USD=${max_spend_usd:g}: the batch stops before "
            "the first job that would start once its rows' cost has reached this.",
            flush=True,
        )

    # M3 item S6 — the ONE scheduler of this run. `--jobs 1` returns quietly and
    # changes nothing; above one it is announced here, before the first VM. The
    # host sampler rides along only for a run that will actually overlap rows.
    budget = row_budget(args, invocation_env, flag="--jobs")
    sampler = None if args.plan else _start_host_sampler(batch_dir, budget)

    code, results = runner_mod.run_batch(
        batch_dir=batch_dir,
        batch_id=batch_dir.name,
        split=split,
        preps=selected,
        entries=batch.entries,
        seeds=split.seeds,
        waves=waves or sorted({prep.wave for prep in selected}),
        vanta_bin=vanta_bin,
        max_attempts=args.max_attempts,
        fixed_rounds=args.fixed_rounds,
        fixed_rounds_declaration=frozen.fixed_rounds(),
        attempts_declaration=frozen.attempts(),
        disk_mb=args.disk_mb,
        show_hints=args.show_hints,
        # The `--issues-dir` this invocation named, or `None` for the committed
        # default. `run_batch` refuses the batch, by name, when a report it may read
        # is missing there — never a fall back and never a live re-fetch (§2.4).
        issues_dir=issues_dir_flag(args),
        plan_only=args.plan,
        max_spend_usd=max_spend_usd,
        # Item 14's per-row half. `--plan` returns from `run_batch` before the loop,
        # so passing the probe unconditionally cannot make a plan need an endpoint.
        # Bound to the effective environment (item B1) so the per-row probe asks the
        # same endpoint the children will use.
        preflight=partial(runner_mod.model_endpoint_preflight, env=invocation_env),
        # Item 19 — the frozen configuration resolved above: the block lands on every
        # row, the environment on every child. Item T3 — the preparation's own block
        # rides along so each row states both configurations.
        config_block=config_block,
        prep_config_block=batch.config_block,
        frozen_env=frozen_env,
        budget=budget,
    )
    if args.plan:
        return EXIT_OK

    # M3 item S6 — stop sampling before the report is derived, so the batch's own
    # host samples are complete and consistent with the rows.
    host_samples = _stop_host_sampler(sampler)

    # The report is derived from the manifest on disk, not from this invocation's
    # return value: a resumed run must summarise the whole batch, not the tail of it.
    records = manifest_mod.Manifest(batch_dir).effective_records()
    if records:
        labels_path, payload = runner_mod.write_batch_report(
            batch_dir, records, host_samples=host_samples
        )
        print()
        print(runner_mod.render_digest(payload))
        print()
        print(f"{payload['runs']} rows -> {batch_dir / 'run_digest.md'}")
        print(f"labels for metrics.py -> {labels_path}")

    broke = sum(result.failed for result in results)
    if broke:
        print(
            f"\n{broke} job(s) left no usable session and were written to "
            f"{batch_dir / 'failures.jsonl'}; their keys stay `failed` in "
            "state.json, so the next `run` re-runs them.",
            file=sys.stderr,
        )
        return EXIT_BUG_FAILED
    return code


def _parse_arms(raw: Union[str, Sequence[str]]) -> Optional[list[str]]:
    """``--arms`` as a spelling list, or ``None`` for the split's own arms.

    Empty means "not given", which is the whole protocol — the same default the
    ``run`` command has, so a queue with no ``--arms`` runs every arm.

    [VANTA-WHY] M3 item G — the flag is ``action="append"``, so a repeated
    ``--arms b1-k4 --arms b3-k4`` accumulates rather than silently keeping only
    the last value (which is how the scheduled ``b1-k4`` arm was dropped from a
    smoke run). Every element may itself be comma- or space-separated, so the
    older ``--arms a,b`` spelling keeps working.
    """
    if isinstance(raw, str):
        raw = [raw]
    spellings: list[str] = []
    for chunk in raw:
        spellings.extend(part for part in chunk.replace(",", " ").split() if part)
    return spellings or None


def _parse_seeds(raw: str) -> Optional[list[int]]:
    """``--seeds`` as an int list, or ``None`` for the split's own seeds."""
    if not raw.strip():
        return None
    try:
        return [int(part) for part in raw.replace(",", " ").split()]
    except ValueError as exc:
        raise SplitError(f"--seeds must be comma-separated integers, got {raw!r}") from exc


def _render_pipeline_schedule(
    batch_id: str,
    selected: Sequence[tuple[dict[str, Any], bool]],
    spellings: Sequence[str],
    seeds: Sequence[int],
    waves: Sequence[int],
) -> str:
    """The pipeline's plan: ``_render_schedule``'s table, read from the split.

    [VANTA-WHY] M3 item AA4 — ``runner._render_schedule`` prints this table from
    ``BugPrep`` objects, and in a pipeline none exist when the table is printed:
    the bugs it introduces are prepared by the very run that follows. The split
    names them already, and the row arithmetic is the same floor the queue's own
    renderer uses, so the table is the same table.
    """
    phases = arms_mod.phases_for(spellings)
    repairs = sum(1 for mode, _ in phases if mode == arms_mod.REPAIR)
    reviews = sum(1 for mode, _ in phases if mode == arms_mod.REVIEW)
    lines = [
        f"batch {batch_id} (pipelined: prepare then run, per bug)",
        f"order: {' → '.join(f'{mode} {written}' for mode, written in phases)}",
        "",
        "| wave | bug | checkpoint | py (pinned) | repair rows | review rows (floor) |",
        "|---|---|---|---|---|---|",
    ]
    total = 0
    for wave in waves:
        for entry, _full in selected:
            if int(entry["wave"]) != wave:
                continue
            repair = repairs * len(seeds)
            review = reviews * 2 * len(seeds)
            total += repair + review
            lines.append(
                f"| {wave} | {bug_key(entry)} | {entry['checkpoint']} | "
                f"{entry['python_version']} | {repair} | {review} |"
            )
    lines += [
        "",
        f"{total} rows if no bug yields a pooled failed patch; "
        f"up to {total * 2} if every bug yields two (§7.1).",
    ]
    return "\n".join(lines)


def _queue_pipeline(
    args: argparse.Namespace,
    *,
    split: Split,
    config_block: dict[str, Any],
    frozen_env: Mapping[str, str],
    invocation_env: Mapping[str, str],
    fixed_rounds_declaration: Optional[Mapping[str, int]],
    attempts_declaration: Optional[Mapping[str, int]],
) -> int:
    """M3 item AA4 — ``queue --pipeline``: prepare each bug, then start its rows.

    [VANTA-WHY] The serial shape is ``prepare`` for the whole split, then ``queue``.
    A bug's rows therefore wait for every other bug's prepare, however long the
    last one takes, even though a bug's rows depend on nothing but that bug's own
    prepare and its predecessors inside its own wave (D13, §7.1). This command runs
    the two stages as one: within a wave, every selected bug's prepare is submitted
    at once, and the moment one bug's prepare is final and frozen its rows are
    submitted — while the other bugs are still preparing. Waves stay sequential,
    because §8.1's gate is a statement about the earlier stages of the same batch.

    What does NOT change, deliberately:
      * ONE ``Budget`` governs every suite run AND every row (S5 + S6 + Y1). The
        budget's size is ``max(--prepare-jobs, --jobs)``; the prepare pool is
        ``--prepare-jobs`` wide and the row pool ``--jobs`` wide, so each flag keeps
        its own meaning while the memory/core gate is a single object.
      * ONE ``SpendGate`` for the whole batch (G1), created here exactly as
        ``_run_queue`` creates it.
      * Each bug's rows are run by ``runner._run_task`` — literally the queue's own
        per-bug unit — so a row is the row the queue would have written, D13 order
        included, and admission is unchanged.
      * The batch-level prepare record is the one ``cmd_prepare`` writes: the same
        ``prepare.json``/``prepare_summary.md``/``context_report.json``/
        ``signals.md``/``issue_texts_all.md``, from the same preps and broken list,
        written once every wave is done (a pipeline's prepare.json can only be
        complete at the end: later waves have not been prepared yet). Its
        ``scheduler`` block describes the whole pipeline invocation, which is what
        it is the record of.
      * ``--only``/``--waves``/``--reserves``/``--measure-only``/``--refreeze``
        select bugs exactly as they do in ``prepare``.
    """
    # The same two refusals `prepare` makes, in `prepare`'s own order and for the
    # same reasons: the flags are prepare flags and `--pipeline` runs the prepare.
    if args.refreeze and args.measure_only:
        raise SplitError(
            "--refreeze rewrites a frozen issue report, and --measure-only freezes nothing: "
            "a measure-only pass reads the file a full prepare wrote. The re-fetch and the "
            "rewrite have to run with suites (§2.4.3), so drop --measure-only."
        )
    selected = select(
        split,
        only=args.only,
        waves=parse_waves(args.waves),
        reserves=args.reserves,
        measure_only=args.measure_only,
        refreeze=args.refreeze,
    )
    if not selected:
        raise SplitError("the selection is empty")
    # `--plan` decides the selection and stops, exactly where `prepare --plan` stops:
    # before the clone, the context window and the binary. A plan that needed the
    # network or a build to print would be a plan of a different command.
    if args.plan:
        sys.stdout.write(render_plan(selected))
        return EXIT_OK

    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    commit = meta_mod.ensure_clone(bugsinpy_root, commit=split.bugsinpy_commit)
    if commit != split.bugsinpy_commit:
        raise SplitError(
            f"BugsInPy is at {commit}, the split pins {split.bugsinpy_commit}. "
            "Refusing to derive ground truth from metadata that moved."
        )
    n_ctx, n_ctx_source = context_mod.explicit_context_window(
        n_ctx=args.n_ctx, environ=invocation_env
    )
    print(
        f"context window: n_ctx={n_ctx} (source: {n_ctx_source}) — the context gate "
        "is sized against this window",
        flush=True,
    )
    # `prepare` refuses a stale binary (the gate shells out to it), and a pipeline
    # IS a prepare, so it refuses for the same reason here.
    vanta_bin = runner_mod.require_vanta_bin(args.vanta_bin.resolve())
    stale_warning = runner_mod.stale_binary_warning(vanta_bin)
    if stale_warning:
        raise runner_mod.RunnerError(
            f"{stale_warning} `prepare` cannot size the context gate against it: the "
            "gate shells out to this binary, so every verdict it writes would describe "
            "a Vanta the batch will not run."
        )

    selected_full = [entry for entry, full in selected if full]
    missing_layers = set(
        prepare_mod.missing_checkpoints([entry["checkpoint"] for entry in selected_full])
    )
    deps_dir = RUNS_DIR / "deps"
    missing_provenance = set(
        bake_mod.missing_deps(deps_dir, [entry["checkpoint"] for entry in selected_full])
    )
    selected_checkpoints = [entry["checkpoint"] for entry in selected_full]
    unstamped = bake_mod.unstamped_deps(deps_dir, selected_checkpoints)
    cache_state = cache_state_line(
        checkpoints=selected_checkpoints, unstamped=unstamped
    )
    if unstamped and not suite_cache.cache_disabled():
        print(
            f"WARNING: the suite cache is OFF for {len(unstamped)} layer(s) with no "
            f"checkpoint_id: {', '.join(unstamped)}.\n"
            "  Every suite run of those layers is measured and never reused. Fix them "
            "once, without re-baking any of them:\n"
            "    python3 scripts/bugsinpy_run.py stamp-existing\n"
            "  then re-run this prepare.",
            file=sys.stderr,
            flush=True,
        )

    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    batch_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = RUNS_DIR / "repos"
    # The pipelined prepare freezes into (and reads from) the directory the invocation
    # named — `--issues-dir`, or the committed default — exactly as `prepare` does.
    issue_sources = prepare_mod.IssueSources(
        issues_dir=prepare_issues_dir(args),
        github_cache_dir=RUNS_DIR / "github-cache",
    )

    # §14.33's backend gate for `--jobs`, same sentence `row_budget` uses: the
    # pipeline creates its own one Budget (S5 + S6), so it makes the check itself.
    if args.jobs > 1:
        backend = (invocation_env.get("VANTA_LLM_BACKEND") or "").strip()
        if backend.lower() != "openai":
            raise runner_mod.RunnerError(
                f"--jobs={args.jobs} needs VANTA_LLM_BACKEND=openai: concurrent rows all "
                "call the same provider (§14.33), and every other backend is a local "
                "model that this concurrency would oversubscribe. "
                f"VANTA_LLM_BACKEND={backend or '(unset)'}."
            )
    # The ONE scheduler of the whole invocation: a bug's suite runs and every row
    # charge come out of this one object. Its size is the wider of the two flags,
    # because each pool is capped by its own flag below while the resources are
    # gated here.
    budget = budget_mod.Budget(
        max(args.prepare_jobs, args.jobs), peak_vms=args.peak_vms
    )
    print(
        f"[scheduler] jobs={budget.jobs} concurrent={'yes' if budget.concurrent else 'no'} "
        f"cores={budget.cores}/{budget.cap_cores} memory="
        f"{'unbounded' if budget.cap_memory_mb is None else f'{budget.cap_memory_mb} MB'} "
        f"host_load={budget.host_load}",
        flush=True,
    )

    seeds = _parse_seeds(args.seeds) or split.seeds
    arm_spellings = _parse_arms(args.arms)
    max_spend_usd = max_spend_usd_from_env(invocation_env)
    if max_spend_usd is not None:
        print(
            f"[spend] VANTA_MAX_SPEND_USD=${max_spend_usd:g}: the queue stops claiming "
            "once its rows' cost has reached this.",
            flush=True,
        )
    # The queue's own refusals, made once here instead of once per bug — the same
    # five `_run_queue` makes before it prints the schedule.
    spellings = runner_mod._select_arms(split, arm_spellings)
    runner_mod._require_arm_list(spellings)
    runner_mod._require_arm_budget(spellings, args.max_attempts)
    runner_mod._require_fixed_rounds(
        spellings, args.fixed_rounds, fixed_rounds_declaration
    )
    runner_mod._require_attempts(spellings, args.max_attempts, attempts_declaration)
    runner_mod._require_seeds(split, seeds)

    entries = {bug_key(entry): entry for entry in split.bugs}
    waves = parse_waves(args.waves) or sorted(
        {int(entry["wave"]) for entry, _full in selected}
    )
    stop_file = runner_mod.queue_stop_path(batch_dir)
    manifest = manifest_mod.Manifest(batch_dir)
    spend_gate = (
        runner_mod.SpendGate(batch_dir, max_spend_usd, manifest)
        if max_spend_usd is not None
        else None
    )
    # `runner._run_task`'s own keyword set, built once: every bug of every wave is
    # run through the queue's per-bug unit, so a row is exactly the row a `queue`
    # would have written.
    run_kwargs: dict[str, Any] = dict(
        batch_dir=batch_dir,
        batch_id=batch_dir.name,
        split=split,
        entries=entries,
        seeds=seeds,
        vanta_bin=vanta_bin,
        max_attempts=args.max_attempts,
        fixed_rounds=args.fixed_rounds,
        fixed_rounds_declaration=fixed_rounds_declaration,
        attempts_declaration=attempts_declaration,
        disk_mb=args.disk_mb,
        show_hints=args.show_hints,
        # The `--issues-dir` this invocation named, so a `--show-hints` row of a
        # pipelined bug reads the same frozen report its prepare stage read.
        issues_dir=issues_dir_flag(args),
        max_spend_usd=max_spend_usd,
        spend_gate=spend_gate,
        preflight=partial(runner_mod.model_endpoint_preflight, env=invocation_env),
        config_block=config_block,
        # The preparation's block is this invocation's block: in a pipeline the
        # prepare and the run are one command, so what `prepare.json` will record
        # and what the rows carry cannot disagree.
        prep_config_block=config_block,
        frozen_env=frozen_env,
        arm_spellings=arm_spellings,
        budget=budget,
    )

    print(_render_pipeline_schedule(batch_dir.name, selected, spellings, seeds, waves))
    tail = ""
    if budget.concurrent:
        tail = (
            f"; one scheduler, cores={budget.cores}/{budget.cap_cores} "
            f"memory={'unbounded' if budget.cap_memory_mb is None else f'{budget.cap_memory_mb} MB'}"
        )
    print(
        f"[queue] {len(selected)} task(s) = wave × bug, up to {args.jobs} rows at once, "
        f"prepare up to {args.prepare_jobs}{tail}; stop file {stop_file}",
        flush=True,
    )
    if stop_file.exists():
        print(
            f"[queue] {stop_file} already exists: no job will be claimed. Delete it "
            "and run again.",
            flush=True,
        )

    owner = runner_mod.queue_lock(batch_dir)
    if not owner.acquire(blocking=False):
        raise manifest_mod.BatchLocked(str(owner.held_message))
    preps: list[prepare_mod.BugPrep] = []
    broken: list[tuple[str, str]] = []
    outcomes: list[Any] = []
    refreeze_keys = set(args.refreeze)
    sampler = _start_host_sampler(batch_dir, budget)
    try:
        for wave in waves:
            wave_selected = [
                (entry, full)
                for entry, full in selected
                if int(entry["wave"]) == wave
            ]
            if not wave_selected:
                continue
            print(
                f"\n=== wave {wave}: {len(wave_selected)} bug(s), up to "
                f"{min(args.jobs, len(wave_selected))} rows at once, prepare up to "
                f"{min(args.prepare_jobs, len(wave_selected))} ===",
                flush=True,
            )
            wave_outcomes: list[Any] = []
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as row_pool, \
                    ThreadPoolExecutor(max_workers=max(1, args.prepare_jobs)) as prep_pool:
                # Every bug of the wave is prepared at once (up to `--prepare-jobs`
                # in flight), and a bug's rows are submitted the moment THAT bug's
                # prepare returns — `as_completed`, not submission order — so one
                # bug's slow prepare cannot hold back another's rows. That is the
                # whole point of the item. The record stays deterministic anyway:
                # `preps` and `broken` are rebuilt in the split's own file order.
                prep_futures = {
                    prep_pool.submit(
                        prepare_one_bug,
                        entry=_entry,
                        full=_full,
                        split=split,
                        bugsinpy_root=bugsinpy_root,
                        repo_cache=repo_cache,
                        batch_dir=batch_dir,
                        deps_dir=deps_dir,
                        issue_sources=issue_sources,
                        n_ctx=n_ctx,
                        n_ctx_source=n_ctx_source,
                        vanta_bin=vanta_bin,
                        refreeze_keys=refreeze_keys,
                        missing_layers=missing_layers,
                        missing_provenance=missing_provenance,
                        budget=budget,
                        traceback_on=args.traceback,
                    ): index
                    for index, (_entry, _full) in enumerate(wave_selected)
                }
                prepared: dict[int, prepare_mod.BugPrep] = {}
                wave_broken: dict[int, tuple[str, str]] = {}
                row_futures: list[Any] = []
                for future in as_completed(prep_futures):
                    index = prep_futures[future]
                    kind, value = future.result()
                    if kind == "broken":
                        wave_broken[index] = value
                        continue
                    prep = value
                    prepared[index] = prep
                    if not prep.runnable:
                        reasons = (
                            "; ".join(prep.gate_failures) or "no gate reason recorded"
                        )
                        print(
                            f"  [queue] NOT RUNNABLE {prep.key}: {reasons} — its rows "
                            "are not run.",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if stop_file.exists():
                        print(
                            f"  [queue] {stop_file} appeared: not claiming "
                            f"{prep.key}.",
                            flush=True,
                        )
                        continue
                    row_futures.append(
                        row_pool.submit(
                            runner_mod._run_task,
                            wave=wave,
                            prep=prep,
                            stop_file=stop_file,
                            **run_kwargs,
                        )
                    )
                for index in sorted(prepared):
                    preps.append(prepared[index])
                for index in sorted(wave_broken):
                    broken.append(wave_broken[index])
                for future in row_futures:
                    wave_outcomes.append(future.result())
            outcomes.extend(wave_outcomes)
            # §8.1 — the same gate `_run_queue` applies: a wave whose gate failed,
            # or that was stopped deliberately, stops the next wave.
            stoppers = [
                outcome
                for outcome in wave_outcomes
                if outcome.reason in ("stop_file", "spend_cap", "gate")
            ]
            if stoppers:
                print(
                    f"\n=== queue stopped in wave {wave}: "
                    + ", ".join(f"{o.bug} ({o.reason})" for o in stoppers)
                    + " ===",
                    flush=True,
                )
                break
    finally:
        owner.release()
    host_samples = _stop_host_sampler(sampler)

    # The batch-level prepare record, exactly as `cmd_prepare` writes it. Written
    # once, here, because a pipelined batch's prepare is only complete when its last
    # wave has been prepared.
    for prep in preps:
        if prep.issue is not None:
            prepare_mod.write_leak_gate(batch_dir, prep)
    prepare_mod.write_issue_texts(batch_dir, preps)
    prepare_mod.write_context_sizes(batch_dir, preps)
    prepare_mod.write_signals(batch_dir, preps)
    prepare_mod.write_context_report(
        batch_dir, preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source
    )
    summary = prepare_mod.render_summary(
        preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source, cache_state=cache_state
    )
    (batch_dir / "prepare_summary.md").write_text(summary, encoding="utf-8")
    split_path = args.split.resolve()
    split_sha_before = split_mod.sha256_file(split_path)
    prepare_record = prepare_mod.prepare_manifest(
        split=split,
        split_path=split_path,
        split_sha_before=split_sha_before,
        batch_dir=batch_dir,
        commit=commit,
        bugsinpy_root=bugsinpy_root,
        selected=[bug_key(entry) for entry, _ in selected],
        broken=broken,
        preps=preps,
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source,
        config_block=config_block,
        scheduler=budget.record(),
    )
    (batch_dir / "prepare.json").write_text(
        json.dumps(prepare_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    split_sha_after = split_mod.sha256_file(split_path)
    if split_sha_after != split_sha_before:
        raise SplitError(
            f"{split_path} changed during prepare ({split_sha_before} → "
            f"{split_sha_after}). The split is an input; something wrote to it."
        )

    print()
    print(summary)
    print()
    print(prepare_mod.render_context_table(preps))
    if broken:
        print()
        print(render_flagged(broken))
    print()
    print(f"batch dir : {batch_dir}")
    print(f"split     : {split_path} (read-only; {split_sha_after[:12]} unchanged)")

    # The queue's own tail, unchanged.
    records = manifest_mod.Manifest(batch_dir).effective_records()
    if records:
        labels_path, payload = runner_mod.write_batch_report(
            batch_dir, records, host_samples=host_samples
        )
        print()
        print(runner_mod.render_digest(payload))
        print()
        print(f"{payload['runs']} rows -> {batch_dir / 'run_digest.md'}")
        print(f"labels for metrics.py -> {labels_path}")

    failed = sum(outcome.failed for outcome in outcomes)
    raised = [outcome.bug for outcome in outcomes if outcome.reason == "task_failed"]
    if failed:
        print(
            f"\n{failed} job(s) left no usable session and were written to "
            f"{batch_dir / 'failures.jsonl'}; their keys stay `failed` in "
            "state.json, so the next `queue` re-runs them.",
            file=sys.stderr,
        )
    if raised:
        print(
            f"\n{len(raised)} bug(s) raised and were skipped: {raised}. Re-run them "
            "once the cause is fixed; their claimed jobs keep their keys.",
            file=sys.stderr,
        )
    unprepared = [prep for prep in preps if not prep.runnable]
    unprepared_keys = {prep.key for prep in unprepared}
    if unprepared:
        print(
            f"\n{len(unprepared)} bug(s) did not prepare and their rows were not run: "
            f"{sorted(unprepared_keys)}.",
            file=sys.stderr,
            flush=True,
        )
    # §7.1 — every selected bug that is meant to run rows must have written at least
    # one. A bug that prepared runnable but was never claimed (the stop file, the
    # spend cap) leaves no outcome at all, so this row count is the only place it is
    # caught: without it a stopped run exits 0 having silently dropped every bug after
    # the stop. The floor is one row per selected *full* bug; a `full=False` reserve is
    # measured only and is not expected to write rows.
    rows_by_bug: dict[str, int] = {}
    for record in records:
        bug = str(record.get("bug") or "")
        if bug:
            rows_by_bug[bug] = rows_by_bug.get(bug, 0) + 1
    broken_keys = {key for key, _reason in broken}
    expected_bugs = [bug_key(entry) for entry, full in selected if full]
    no_rows = [
        key
        for key in expected_bugs
        if key not in unprepared_keys
        and key not in broken_keys
        and rows_by_bug.get(key, 0) == 0
    ]
    if no_rows:
        print(
            f"\nROW-COUNT FAILURE: {len(no_rows)} of {len(expected_bugs)} selected "
            f"bug(s) wrote no row: {sorted(no_rows)}. A run must write at least one row "
            f"per selected bug; the batch has {len(rows_by_bug)} bug(s) with rows. Do not "
            "read this batch as complete.",
            file=sys.stderr,
            flush=True,
        )
    if failed or raised or unprepared or broken or no_rows:
        return EXIT_BUG_FAILED
    return 1 if any(outcome.code != 0 for outcome in outcomes) else EXIT_OK


def cmd_queue(args: argparse.Namespace) -> int:
    """Item 13 — run a prepared batch's bugs several at a time.

    The same preamble as ``run`` — repo-root, endpoint, frozen configuration, split,
    prepared batch, window and stale-binary warnings, spend cap — because a queue
    is the same batch with a different claim loop, and none of those checks stop
    being true because several bugs run at once. The difference is the last call:
    ``run_queue`` owns the batch directory and runs up to ``--jobs`` bugs of one
    wave concurrently, delegating each bug to the same ``run_batch`` ``run`` uses.
    """
    runner_mod.require_repo_cwd()
    # Item 19 — the frozen configuration, resolved once for the whole queue. `jobs`
    # parallel bugs share one configuration; a queue split across two config files
    # would be two experiments sharing a batch directory.
    frozen = frozen_mod.load()
    config_block, frozen_env = frozen_mod.resolve(
        frozen, environ=os.environ, allow_override=args.allow_override
    )
    print(
        f"[config] {frozen.name} {frozen.sha256[:12]} ({config_block['config_path']})"
        + (
            "; --allow-override waives "
            + ", ".join(config_block["config_overrides"])
            if config_block["config_overrides"]
            else ""
        ),
        flush=True,
    )
    # M3 items S1–S3 — `--no-cache` before the first job of this queue, in the
    # parent, so every concurrent job's child inherits it.
    honour_no_cache(args)
    # Item B1 — the same ONE effective environment as `run`: the caller's own with
    # the frozen `[env]` applied, used by the preflight and the spend cap so a
    # setting frozen in the file but not exported is not silently dropped.
    invocation_env = effective_env(frozen_env)
    if not args.plan:
        runner_mod.require_model_endpoint(
            preflight=partial(runner_mod.model_endpoint_preflight, env=invocation_env)
        )
    split = load_split(args.split.resolve())
    # M3 item AA4 — `--pipeline` is the same command with the prepare stage folded
    # in: it prepares each bug and starts that bug's rows the moment its prepare is
    # final, instead of reading a batch `prepare` already finished. It needs no
    # `prepare.json`, so it branches before `load_prepared` (which raises "no
    # prepare.json in …" for a fresh directory) and writes one when it is done.
    if args.pipeline:
        return _queue_pipeline(
            args,
            split=split,
            config_block=config_block,
            frozen_env=frozen_env,
            invocation_env=invocation_env,
            fixed_rounds_declaration=frozen.fixed_rounds(),
            attempts_declaration=frozen.attempts(),
        )
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    # Item T3 — the same refusal as `run`, for the same reason: a queue shares one
    # configuration with its preparation or it is two experiments in one directory.
    config_mismatch = runner_mod.prepared_config_mismatch(batch.config_block, config_block)
    if config_mismatch and not args.allow_override:
        raise runner_mod.RunnerError(config_mismatch)
    # Item T5 — the same effective environment `run` compares against, for the same
    # reason: the queue's children resolve their window from the frozen `[env]`, so a
    # variable merely exported in this shell must not read as a divergence.
    window_warning = runner_mod.context_window_warning(
        batch, environ=invocation_env
    )
    if window_warning:
        print(f"WARNING: {window_warning}", file=sys.stderr, flush=True)
    waves = parse_waves(args.waves)
    selected = select_preps(batch.preps, only=args.only, waves=waves)
    # The queue's own selections, resolved here so a bad value is refused before
    # the schedule prints rather than once per bug: the arms and seeds a queue runs
    # are a subset of what the split declares, never a second protocol.
    arm_spellings = _parse_arms(args.arms)
    seeds = _parse_seeds(args.seeds) or split.seeds
    vanta_bin = runner_mod.require_vanta_bin(args.vanta_bin.resolve())
    stale_warning = runner_mod.stale_binary_warning(vanta_bin)
    if stale_warning:
        print(f"WARNING: {stale_warning}", file=sys.stderr, flush=True)

    # Read from the effective environment (item B1), so a cap frozen in `[env]` is
    # honoured even when the caller did not export it.
    max_spend_usd = max_spend_usd_from_env(invocation_env)
    if max_spend_usd is not None:
        print(
            f"[spend] VANTA_MAX_SPEND_USD=${max_spend_usd:g}: the queue stops claiming "
            "once its rows' cost has reached this.",
            flush=True,
        )

    # M3 item S6 — the ONE scheduler of this queue, shared by every worker and
    # threaded down into each per-bug `run_batch`; `--jobs 1` changes nothing.
    budget = row_budget(args, invocation_env, flag="--jobs")
    sampler = None if args.plan else _start_host_sampler(batch_dir, budget)

    code, outcomes = runner_mod.run_queue(
        batch_dir=batch_dir,
        batch_id=batch_dir.name,
        split=split,
        preps=selected,
        entries=batch.entries,
        seeds=seeds,
        waves=waves or sorted({prep.wave for prep in selected}),
        vanta_bin=vanta_bin,
        jobs=args.jobs,
        max_attempts=args.max_attempts,
        fixed_rounds=args.fixed_rounds,
        fixed_rounds_declaration=frozen.fixed_rounds(),
        attempts_declaration=frozen.attempts(),
        disk_mb=args.disk_mb,
        show_hints=args.show_hints,
        # The `--issues-dir` this invocation named, or `None` for the committed
        # default. With `--pipeline` the prepare stage froze into it; the queue
        # refuses a bug, by name, whose report it may read is missing there.
        issues_dir=issues_dir_flag(args),
        plan_only=args.plan,
        max_spend_usd=max_spend_usd,
        # Bound to the effective environment (item B1), as in `run`.
        preflight=partial(runner_mod.model_endpoint_preflight, env=invocation_env),
        config_block=config_block,
        prep_config_block=batch.config_block,
        frozen_env=frozen_env,
        arm_spellings=arm_spellings,
        budget=budget,
    )
    if args.plan:
        return EXIT_OK

    # M3 item S6 — stop sampling before the report is derived, so the batch's own
    # host samples are complete and consistent with the rows.
    host_samples = _stop_host_sampler(sampler)

    # Same tail as `run`: the report is the manifest on disk, so a resumed queue
    # summarises the whole batch and not the part this invocation happened to run.
    records = manifest_mod.Manifest(batch_dir).effective_records()
    if records:
        labels_path, payload = runner_mod.write_batch_report(
            batch_dir, records, host_samples=host_samples
        )
        print()
        print(runner_mod.render_digest(payload))
        print()
        print(f"{payload['runs']} rows -> {batch_dir / 'run_digest.md'}")
        print(f"labels for metrics.py -> {labels_path}")

    failed = sum(outcome.failed for outcome in outcomes)
    broken = [outcome.bug for outcome in outcomes if outcome.reason == "task_failed"]
    if failed:
        print(
            f"\n{failed} job(s) left no usable session and were written to "
            f"{batch_dir / 'failures.jsonl'}; their keys stay `failed` in "
            "state.json, so the next `queue` re-runs them.",
            file=sys.stderr,
        )
    if broken:
        print(
            f"\n{len(broken)} bug(s) raised and were skipped: {broken}. Re-run them "
            "once the cause is fixed; their claimed jobs keep their keys.",
            file=sys.stderr,
        )
    if failed or broken:
        return EXIT_BUG_FAILED
    return code


def _resolve_pool_split(args: argparse.Namespace, batch_dirs: Sequence[Path]) -> Split:
    """The split a ``pool`` invocation reads, resolving an omitted ``--split`` (#E).

    A ``pool`` stage reads *prepared* batches, so it has to read the split those
    batches were prepared with. Falling back to a default split when ``--split`` is
    omitted is what produced the bare "prepare.json names sanic/1, which the split
    does not" — the default happened not to name the batch's bugs. So:
    an omitted ``--split`` means "the batch's own recorded split"; a *given*
    ``--split`` must agree with that record, or the invocation is refused before
    anything is graded. A batch from before the field was recorded (``None``) leaves
    the caller with its default, exactly as before.
    """
    recorded = {batch: runner_mod.recorded_split_path(batch) for batch in batch_dirs}
    if args.split is not None:
        given = args.split.resolve()
        for batch_dir, path in recorded.items():
            if path is not None and path.resolve() != given:
                raise pool_mod.PoolError(
                    f"`--split {args.split}` disagrees with the split {path} that "
                    f"{batch_dir / 'prepare.json'} was prepared with. Omit --split to "
                    "use the batch's own, or pass that recorded path."
                )
        return load_split(given)
    distinct = {path.resolve() for path in recorded.values() if path is not None}
    if len(distinct) > 1:
        raise pool_mod.PoolError(
            "`--split` was omitted and the listed batches were prepared with "
            "different splits: " + ", ".join(sorted(str(path) for path in distinct)) + "."
        )
    if distinct:
        return load_split(distinct.pop())
    return load_split(DEFAULT_SPLIT)


def cmd_pool(args: argparse.Namespace) -> int:
    """`pool` owns two files; dispatch to the one this invocation named.

    Item 5's mutants file (``pool/<bug>.mutants.json``) is written by ``--mutants``;
    item 6's assembled review pool (``pool/<bug>.json``) — the file reviews read —
    is written by ``--build``. Exactly one is required, so a reader can never
    mistake which of the two a command produced.
    """
    if args.mutants and args.build:
        raise pool_mod.PoolError(
            "`pool` writes one file at a time: pass --mutants (item 5, the mutant "
            "pool) or --build (item 6, the assembled pool), not both."
        )
    if args.mutants:
        return cmd_pool_mutants(args)
    if args.build:
        return cmd_pool_build(args)
    raise pool_mod.PoolError(
        "`pool` owns two files. `--mutants` writes the reference-fix mutant pool "
        "(pool/<bug>.mutants.json, item 5); `--build` writes the assembled review "
        "pool (pool/<bug>.json, item 6) that reviews read. Pass exactly one."
    )


def cmd_pool_mutants(args: argparse.Namespace) -> int:
    """Item 5 — write and hash-lock each bug's reference-fix mutant pool.

    Repo-root check first, for the same reason every other stage does it: the whole
    preflight and grading run in Shuru, whose mount rule needs every tree inside the
    process CWD. ``--plan`` needs neither a VM nor a checkpoint — the candidates are
    a pure function of the reference fix — so it is exempt from the endpoint check
    the run stage does, and it boots nothing.
    """
    if len(args.batch) != 1:
        raise mutants_mod.MutantError(
            f"`pool --mutants` writes one batch's pool and takes exactly one --batch; "
            f"got {len(args.batch)}."
        )
    value = args.batch[0]
    if "=" in value:
        raise mutants_mod.MutantError(
            f"`--mutants` takes a bare `--batch DIR`, not a tag ({value!r}): its source "
            "is the reference fix, not a harvested batch. Tags are for `--build`."
        )
    if args.rebuild:
        raise mutants_mod.MutantError(
            "`--rebuild` belongs to `--build` (item 6); the mutants file is rewritten "
            "in place each time item 5 runs."
        )
    runner_mod.require_repo_cwd()
    batch_dir = runner_mod.require_batch_dir(Path(value).resolve())
    split = _resolve_pool_split(args, [batch_dir])
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    waves = parse_waves(args.waves)
    selected = select_preps(batch.preps, only=args.only, waves=waves)

    if args.plan:
        for prep in selected:
            for line in mutants_mod.plan_lines(prep):
                print(line)
        return EXIT_OK

    for index, prep in enumerate(selected, start=1):
        checkpoint = str(batch.entries[prep.key]["checkpoint"])
        print(f"[{index}/{len(selected)}] {prep.key} — grading mutants in Shuru…", flush=True)
        result = mutants_mod.run_mutants(
            prep=prep,
            batch_dir=batch_dir,
            checkpoint=checkpoint,
            disk_mb=args.disk_mb,
        )
        print(mutants_mod.render_pool_line(prep, result), flush=True)
    return EXIT_OK


def _pool_harvests(
    key: str, batches: list[dict[str, Any]], sessions: Path
) -> list[pool_mod.Harvest]:
    """Each tagged batch's Creator attempts for one bug, in the order listed."""
    harvests: list[pool_mod.Harvest] = []
    for entry in batches:
        prep = entry["preps"].get(key)
        if prep is None:
            continue
        harvests.append(
            pool_mod.harvest_batch(
                tag=entry["tag"],
                batch_dir=entry["dir"],
                sessions_dir=sessions,
                # Restricting the prep map to this one bug is what scopes the harvest
                # (and its `not_reviewable` count) to the bug being built; a record for
                # another bug is skipped by `harvest_batch` itself.
                preps={key: prep},
                manifest=entry["manifest"],
            )
        )
    return harvests


def cmd_pool_build(args: argparse.Namespace) -> int:
    """Item 6 — assemble, grade and hash-lock the review pool reviews read.

    Every listed batch is read through its own manifest (the row's ``session_id``,
    the ``artifacts.final_patch`` and the run's attempt count) and its own prepared
    trees (the per-attempt snapshots are diffed against that batch's ``vanta_clean``).
    The pool is written under ``--out``, which has no default: it spans several
    batches and arms and must not live inside whichever batch is listed first. A
    batch still running is refused — the pool is built after every repair batch has
    finished.
    """
    if args.out is None:
        raise pool_mod.PoolError(
            "`pool --build` needs `--out DIR`, and it has no default: the pool spans "
            "several batches and arms and must not live inside whichever batch is "
            "listed first. Nothing was read."
        )
    if not args.batch:
        raise pool_mod.PoolError(
            "`pool --build` needs at least one `--batch TAG=DIR` (TAG one of "
            f"{pool_mod.TAG_ORDER})."
        )
    pairs = [pool_mod.parse_batch_argument(value) for value in args.batch]
    runner_mod.require_repo_cwd()
    out_dir = args.out.resolve()
    sessions = runner_mod.sessions_dir()

    # Every listed batch's cursor is read before *any* prepared tree is loaded: the
    # rule is "the repair batches have finished", and a batch that is still running
    # has to refuse the build without half-reading the corpus the rule is about.
    located: list[tuple[str, Path]] = [
        (tag, runner_mod.require_batch_dir(raw.resolve())) for tag, raw in pairs
    ]
    still_running: list[str] = []
    for _, batch_dir in located:
        state = manifest_mod.Manifest(batch_dir).state()
        running = sorted(key for key, value in state.items() if value == manifest_mod.RUNNING)
        if running:
            shown = ", ".join(running[:4]) + ("…" if len(running) > 4 else "")
            still_running.append(
                f"{batch_dir} still has {len(running)} running job(s) ({shown})"
            )
    if still_running:
        raise pool_mod.PoolError(
            f"refusing to build {out_dir}: " + "; ".join(still_running) + ". The pool "
            "is built once every repair batch has finished; re-run when their "
            "state.json shows none running."
        )

    # The split is resolved from the batches themselves when `--split` is omitted (#E),
    # so it is read *after* they are located and refused-on-running. Nothing is graded
    # before this point.
    split = _resolve_pool_split(args, [batch_dir for _, batch_dir in located])
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()

    batches: list[dict[str, Any]] = []
    for tag, batch_dir in located:
        manifest = manifest_mod.Manifest(batch_dir)
        state = manifest.state()
        batch = runner_mod.load_prepared(
            batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
        )
        preps = {prep.key: prep for prep in batch.preps}
        batches.append(
            {
                "tag": tag,
                "dir": batch_dir,
                "batch": batch,
                "preps": preps,
                "manifest": manifest,
                "failed": sorted(key for key, value in state.items() if value == manifest_mod.FAILED),
            }
        )

    # One representative prep per bug, in the first batch's file order — the split is
    # the same in every batch, so a bug's prep differs only in its batch-relative root.
    representative: dict[str, prepare_mod.BugPrep] = {}
    for entry in batches:
        for prep in entry["batch"].preps:
            representative.setdefault(prep.key, prep)
    preps_in_order = [representative[prep.key] for prep in batches[0]["batch"].preps]
    waves = parse_waves(args.waves)
    selected = select_preps(preps_in_order, only=args.only, waves=waves)

    failed_keys = sum(len(entry["failed"]) for entry in batches)
    if failed_keys:
        print(
            f"note: {failed_keys} failed job(s) across the listed batches carry no "
            "patch to harvest; they are counted in the pool's header, not treated as "
            "an error.",
            file=sys.stderr,
        )

    if args.plan:
        for prep in selected:
            for line in pool_mod.plan_lines(
                prep, harvests=_pool_harvests(prep.key, batches, sessions)
            ):
                print(line)
        return EXIT_OK

    for index, prep in enumerate(selected, start=1):
        home = next(entry for entry in batches if prep.key in entry["preps"])
        checkpoint = str(home["batch"].entries[prep.key]["checkpoint"])
        mutants_file = home["dir"] / pool_mod.POOL_DIRNAME / f"{prep.slug}.mutants.json"
        # The header's `failed-keys` is this bug's own count, summed over every
        # listed batch — the batch-wide total above is only the note to the operator.
        bug_failed = sum(
            pool_mod.failed_keys_for(entry["manifest"].state(), prep.key)
            for entry in batches
        )
        print(
            f"[{index}/{len(selected)}] {prep.key} — grading the pool in Shuru…",
            flush=True,
        )
        result = pool_mod.run_pool(
            prep=home["preps"][prep.key],
            out_dir=out_dir,
            harvests=_pool_harvests(prep.key, batches, sessions),
            mutants_file=mutants_file,
            checkpoint=checkpoint,
            grade_batch_dir=home["dir"],
            disk_mb=args.disk_mb,
            rebuild=args.rebuild,
            failed_keys=bug_failed,
        )
        print(pool_mod.render_pool_line(prep, result), flush=True)
    return EXIT_OK


def cmd_localize(args: argparse.Namespace) -> int:
    """Ranking quality on the dev split, from evidence already on disk (§1.6.5, §1.6.11).

    No VM, no LLM, no selector: it reads the selection ``prepare`` measured
    (``context_report.json``) and each bug's fix-commit patch, and prints whether
    the fix's own files were among those shown **in full**, with the rank the
    selection gave each of them. That is what makes it a *view* of the ranking
    rather than an input to it — nothing in ``prepare`` calls this, and the split is
    the dev one so the number stays a tuning number.
    """
    split_path = args.split.resolve()
    if split_path != DEV_SPLIT.resolve():
        raise SplitError(
            f"the localization audit runs on the dev split only ({DEV_SPLIT}), because "
            "it exists to judge ranking quality — a number produced for a held-out set "
            "is a number one tunes on it. Refusing to audit "
            f"{split_path}."
        )
    split = load_split(split_path)
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    table, _rows = runner_mod.localization_table(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    print(table)
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Re-derive the batch report from the manifest, then score the review rows.

    A pure function of the evidence: no VM, no LLM, and nothing written that is not
    a view of ``manifest.jsonl``. The scoring step is ``scripts/metrics.py`` itself
    (§5.4, §9) — the harness does not re-implement the classifier the rest of the
    project already uses, because two implementations of "what counts as a false
    approval" is how two numbers disagree.
    """
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    records = manifest_mod.Manifest(batch_dir).effective_records()
    if not records:
        raise runner_mod.RunnerError(f"no manifest rows in {batch_dir}; run it first")
    labels_path, payload = runner_mod.write_batch_report(batch_dir, records)
    print(runner_mod.render_digest(payload))
    print()
    print(f"{payload['runs']} rows -> {batch_dir / 'run_digest.md'}")
    print(f"labels -> {labels_path}")
    if not any(record.get("candidate") for record in records):
        print(
            "\nno review rows yet: metrics.py scores labelled sessions, and only a "
            "review pass over a known-good/known-bad candidate produces one (§5.4)."
        )
        return EXIT_OK
    argv = [
        sys.executable,
        str(SCRIPTS_DIR / "metrics.py"),
        "--sessions-dir",
        str(runner_mod.sessions_dir()),
        "--labels",
        str(labels_path),
        # §9.1/§9.4: the pooled table is cut by provenance, and the oracle facts
        # its gauges read are the manifest's. Both are already on disk here, so
        # the report asks for both rather than printing the per-session view alone.
        "--pool-by-config",
        "--manifest",
        str(batch_dir / manifest_mod.MANIFEST_NAME),
    ]
    print()
    print("$ " + " ".join(argv), flush=True)
    return EXIT_OK if subprocess.run(argv, cwd=str(REPO_ROOT)).returncode == 0 else EXIT_BUG_FAILED


def _replay_session_id(path: Path) -> Optional[str]:
    """The ``session_id`` a replay file records, or ``None`` when it records none.

    ``None`` covers every way a file can fail to name a session — absent, not
    JSON, not an object, an empty or non-string ``session_id`` — because the only
    use is "is this file the one the current row's session wrote?", and a file
    that cannot say is not that file.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("session_id")
    return str(value) if isinstance(value, str) and value else None


def _replay_target(
    out_dir: Path, stem: str, session_id: str, *, force: bool
) -> tuple[Optional[Path], Path]:
    """Where this row's replay goes, and the file it would already be there as (G3).

    ``replay`` names a replay by the row's **key**, so a re-run of an inconclusive
    row — same key, NEW session — produced the path of the session it replaces.
    Skipping on the path alone then left the replacement with no replay at all,
    and F9's effective-session filter dropped the old file, so *neither* session
    had one. A file on disk is never rewritten: when the plain name holds another
    session's replay, this session's goes beside it as ``<stem>.<session>.json``
    and ``analyse.load_replays`` — which globs the directory and keeps the rows
    whose session is effective — picks the live one. Nothing else reads these
    files by name, so no lookup has to move with it.

    Returns ``(write_to, found)``. ``write_to`` is ``None`` when this session's
    own replay is already on disk and ``--force`` was not given; ``found`` is
    where it was found, so the caller can name it in the skip message. A session
    id is unique per run, so the qualified name is this session's and a re-run
    under it resumes on the same file rather than writing a second copy.
    """
    plain = out_dir / f"{stem}.json"
    if not plain.exists() or _replay_session_id(plain) == session_id:
        target = plain
    else:
        target = out_dir / f"{stem}.{session_id}.json"
    if target.exists() and not force and _replay_session_id(target) == session_id:
        return None, target
    return target, target


def cmd_replay(args: argparse.Namespace) -> int:
    """Item 10 — replay every stored attempt of a row onto one fixed test set S.

    [VANTA-WHY] Its own sub-command, never wired into ``run`` or ``pool``: a row's
    live verdicts are complete before it runs, and replay writes only under
    ``<batch>/replay/`` — the manifest, the transitions file, ``oracle/`` and
    ``vanta_sessions/`` are opened read-only, so no verdict, label, score or row on
    disk can move. The two checks up front are the same preflight every VM-booting
    stage does, because Shuru may only mount trees under the process CWD.

    The split resolves exactly as ``pool``'s does (:func:`_resolve_pool_split`): an
    omitted ``--split`` means "the split this batch was prepared with", and a given
    one must agree with that record. Reusing that function is deliberate — a second
    copy of the rule is how two stages come to disagree about which bugs a batch
    holds.
    """
    if len(args.batch) != 1:
        raise replay_mod.ReplayError(
            "`replay` replays one batch at a time and takes exactly one --batch; "
            f"got {len(args.batch)}."
        )
    runner_mod.require_repo_cwd()
    batch_dir = runner_mod.require_batch_dir(Path(args.batch[0]).resolve())
    split = _resolve_pool_split(args, [batch_dir])
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    waves = parse_waves(args.waves)
    selected = select_preps(batch.preps, only=args.only, waves=waves)
    preps = {prep.key: prep for prep in selected}
    views = replay_mod.parse_views(args.views)
    sessions = runner_mod.sessions_dir()

    # F3 — the rows that are current, through the one shared rule: a resumed or
    # re-run job appears more than once and a re-run of an inconclusive job names the
    # session it replaces, so the raw rows would replay a superseded session. The
    # helper keeps the last row per key and drops a superseded session; a row with no
    # key has no identity to replay under, so it is skipped as before.
    rows = [
        record
        for record in manifest_mod.effective_records(
            manifest_mod.Manifest(batch_dir).records()
        )
        if record.get("key") and replay_mod.is_repairable_row(record)
    ]
    if not rows:
        raise replay_mod.ReplayError(
            f"no repair rows to replay in {batch_dir}: replay needs a repair row "
            "with a session."
        )

    out_dir = batch_dir / replay_mod.REPLAY_DIRNAME
    for index, record in enumerate(rows, start=1):
        key = str(record.get("key"))
        prep = preps.get(str(record.get("bug") or ""))
        if prep is None:
            print(
                f"[{index}/{len(rows)}] {key} — skipped: this selection names no prep "
                f"for {record.get('bug')!r}",
                flush=True,
            )
            continue
        out_path, found = _replay_target(
            out_dir,
            replay_mod.row_stem(key),
            str(record.get("session_id") or ""),
            force=args.force,
        )
        if out_path is None:
            print(
                f"[{index}/{len(rows)}] {key} — skipped: {found.name} already "
                "exists (--force to redo)",
                flush=True,
            )
            continue
        checkpoint = str(batch.entries[prep.key]["checkpoint"])
        # A row that cannot be replayed is recorded, not dropped: an infra-failed run
        # and a run whose every attempt failed to apply both have no candidate code to
        # put in a column past the baseline, and raising here would take the rest of
        # the batch down with one row (item 10, replay fix 2). The reason tells the two
        # apart, and the note beside it is read from `replay_mod.SKIP_NOTES` so this
        # line cannot hardcode one meaning for both.
        reason = replay_mod.replay_skip_reason(record)
        if reason is not None:
            result = replay_mod.skipped_row(
                record=record,
                prep=prep,
                checkpoint=checkpoint,
                sessions_dir=sessions,
                views=views,
                seed=args.seed,
                sample_rate=args.sample_rate,
                reason=reason,
                batch_dir=batch_dir,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            note = replay_mod.SKIP_NOTES.get(reason, "recorded, not replayed")
            print(
                f"[{index}/{len(rows)}] {key} — skipped: {reason} ({note}) -> {out_path}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(rows)}] {key} — replaying {len(views)} view(s) in Shuru…",
            flush=True,
        )
        result = replay_mod.replay_row(
            record=record,
            prep=prep,
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            sessions_dir=sessions,
            views=views,
            seed=args.seed,
            sample_rate=args.sample_rate,
            disk_mb=args.disk_mb,
            timeout_secs=replay_mod.DEFAULT_TIMEOUT_SECS,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"    -> {out_path}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bugsinpy_run.py",
        description="BugsInPy harness: prepare the split, then run it (§7.1).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare",
        help="checkout, SWE-bench split, hidden oracles, context precheck, leak gate",
    )
    prep.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    prep.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="batch directory (default: runs/bugsinpy/prepare-<utc>)",
    )
    prep.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="prepare only these bugs (repeatable); forces full preparation",
    )
    prep.add_argument(
        "--refreeze",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help=(
            "re-fetch and rewrite this bug's frozen issue report "
            "(plans/bugsinpy_issues/<project>_<bug>.md), recording the date in its header "
            "(§2.4.3). Repeatable, and it selects only the bugs it names, in full. Without "
            "it an artifact is never replaced, however wrong it looks: a file whose hashes "
            "disagree with its header is refused and left untouched."
        ),
    )
    prep.add_argument("--waves", default="", help="comma-separated wave numbers")
    prep.add_argument(
        "--vanta-bin",
        type=Path,
        default=runner_mod.DEFAULT_VANTA_BIN,
        help=(
            "the vanta_m3 binary the context gate shells out to and the run will use "
            "(default: target/release/vanta_m3). `prepare` refuses a binary older than "
            "the Rust sources: the gate measures that binary's selection, so a stale "
            "one would write verdicts for a Vanta the batch will not run."
        ),
    )
    prep.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        metavar="TOKENS",
        help=(
            "the context window the context gate is sized against (§1.6.1). Checked "
            "against VANTA_LLM_CONTEXT_TOKENS and the server's /props n_ctx rather "
            "than overriding them: a run resolves the window from those two, so a "
            "flag that disagreed would size the gate against a prompt no run sends."
        ),
    )
    prep.add_argument(
        "--reserves",
        choices=("measure", "skip"),
        default="measure",
        help="what to do with role=reserve bugs (default: measure their context only)",
    )
    prep.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "context-measure every selected bug without running a suite or booting a "
            "VM (§1.6); this is how the pilot set is chosen from data"
        ),
    )
    prep.add_argument(
        "--plan",
        action="store_true",
        help="print the selected plan from the committed split and exit",
    )
    prep.add_argument("--traceback", action="store_true", help="print tracebacks")
    prep.add_argument(
        "--allow-override",
        action="store_true",
        help=(
            "prepare even though the environment contradicts configs/v1-m3.toml "
            "(item 19/Q3). Without it, a `VANTA_` variable the file freezes to a "
            "different value — or freezes to `unset` and the caller set — refuses "
            "prepare before the first checkout, because the gate would be sized "
            "against a window the run will not use. The waiver and the values it "
            "waived are recorded in `prepare.json` (`config_override`, "
            "`config_overrides`), so such a batch still cannot be pooled with one "
            "that did not need it."
        ),
    )
    prep.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "force a FRESH suite run instead of serving it from the M3 suite cache "
            "(items S1–S3), so a cached result can be checked against a fresh one on "
            "demand. The flag does not make the cache invisible: the entry is still "
            "read, the row records that it WOULD have hit (`off:no_cache`), and the "
            "fresh payload then replaces the entry it could not use."
        ),
    )
    prep.add_argument(
        "--prepare-jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "the maximum number of prepared-suite runs in flight at once (M3 item S5, "
            "default 1). Same meaning as `run --jobs` and `queue --jobs`: the memory "
            "and core budget can admit fewer, never more. Each bug's three independent "
            "derivations (buggy / oracle / fixed) are submitted together and overlap up "
            "to N at a time; the two runs of one pair are always two separate VMs and "
            "overlap only when N allows it. N is capped at host cores − 2, and each run "
            "holds its own recorded memory (vm_memory_mb) plus one core. Default 1: one "
            "run at a time, today's behaviour. A concurrent prepare records what it used "
            "on prepare.json and on every bug row (`scheduler`), because a measurement "
            "made beside another one is a different experiment from a serial one."
        ),
    )
    prep.add_argument(
        "--peak-vms",
        type=int,
        default=budget_mod.DEFAULT_PEAK_VMS,
        metavar="N",
        help=(
            "the scheduler's conservative charge for a ROW, used by `run`/`queue`; "
            "prepare charges each suite run its own memory instead, so this only sizes "
            f"the shared budget knob (default {budget_mod.DEFAULT_PEAK_VMS})."
        ),
    )
    prep.add_argument(
        "--issues-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "where prepare freezes (and reads) the frozen issue reports. Flag absent "
            "is the committed `plans/bugsinpy_issues`, byte-for-byte today's path. A "
            "held-out split points this at a gitignored directory (e.g. "
            "`plans/private/bugsinpy_issues`) so its report text is frozen and "
            "hashable without being published (§2.4.2)."
        ),
    )
    prep.set_defaults(func=cmd_prepare)

    bake = sub.add_parser(
        "bake",
        help="create the interpreter and per-bug Shuru checkpoints (the only networked stage)",
    )
    bake.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    bake.add_argument(
        "--pythons",
        default="",
        help="comma-separated minor versions, e.g. 3.8 (default: every version in the split)",
    )
    bake.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="bake per-bug layers for these bugs only (repeatable)",
    )
    bake.add_argument(
        "--reserves",
        action="store_true",
        help="include role=reserve bugs (default: pilots only, §8)",
    )
    bake.add_argument(
        "--disk-mb",
        type=int,
        default=bake_mod.BAKE_DISK_MB,
        help=f"shuru disk in MB; must be >= {bake_mod.MIN_DISK_MB} (the base image's)",
    )
    # The bake VM's size, which shuru's own defaults (2 CPUs, 2048 MB) are too small
    # for: ``pandas/1``'s scipy build is two parallel ``cc1plus`` over 2048 MB and the
    # kernel kills one. Sizing only — no layer's recipe depends on it — and recorded in
    # every layer's provenance (bake.BUILD_VM_KEY) and its build log.
    bake.add_argument(
        "--cpus",
        type=int,
        default=bake_mod.DEFAULT_CPUS,
        help="shuru --cpus for every layer's build VM",
    )
    bake.add_argument(
        "--memory",
        type=int,
        default=bake_mod.DEFAULT_MEMORY_MB,
        metavar="MB",
        help="shuru --memory (MB) for every layer's build VM",
    )
    bake.add_argument(
        "--plan",
        action="store_true",
        help="print the layer plan and exit (this is also the default without --yes)",
    )
    bake.add_argument(
        "--yes",
        action="store_true",
        help="actually bake: downloads interpreters and installs packages inside Shuru",
    )
    bake.add_argument(
        "--force",
        action="store_true",
        help="rebuild layers that already exist",
    )
    bake.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "resolve and classify every pin against PyPI, write the preflight records "
            "and exit without building anything (§3.2); does not need --yes"
        ),
    )
    bake.set_defaults(func=cmd_bake)

    stamp = sub.add_parser(
        "stamp-existing",
        help=(
            "give every checkpoint baked before M3 item U3 its identity, without "
            "re-baking it, so the suite cache can key on it"
        ),
    )
    stamp.set_defaults(func=cmd_stamp_existing)

    run = sub.add_parser(
        "run",
        help=(
            "run a prepared batch: baseline repair, vanta repair, baseline review, "
            "vanta review (§7.1); resumable"
        ),
    )
    run.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    run.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="the batch directory `prepare` created — the resume cursor lives there",
    )
    run.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="run only these bugs (repeatable)",
    )
    run.add_argument("--waves", default="", help="comma-separated wave numbers")
    run.add_argument(
        "--vanta-bin",
        type=Path,
        default=runner_mod.DEFAULT_VANTA_BIN,
        help="the vanta_m3 binary to invoke (default: target/release/vanta_m3)",
    )
    run.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Vanta's repair budget per run (default: 4, the binary's own). An arm "
            "that pins a budget owns it: B0 is one Creator call, so it runs 1 "
            "whether or not the flag is given, and an explicit different value is "
            "refused before the batch starts."
        ),
    )
    run.add_argument(
        "--fixed-rounds",
        type=int,
        default=None,
        help=(
            "state that this run spends EXACTLY K repair attempts (M3 item 9). A "
            "fixed-round run IS the arm, so schedule b1-k4/b3-k4; the flag is a "
            "caller's statement and is refused — before the batch starts — when no "
            "scheduled arm is a fixed-round arm, or when it disagrees with the K the "
            "arm (and the frozen config's [fixed_rounds] table) declares. The harness "
            "states the arm's own K on the child's argv; it does not need this flag."
        ),
    )
    run.add_argument(
        "--disk-mb",
        type=int,
        default=prepare_mod.DISK_MB,
        help=(
            "shuru disk in MB for every VM this stage boots; the default matches "
            f"the suites prepare ran ({prepare_mod.DISK_MB})"
        ),
    )
    run.add_argument(
        "--plan",
        action="store_true",
        help="print the schedule (and the honest floor on review rows) and exit",
    )
    run.add_argument(
        "--allow-override",
        action="store_true",
        help=(
            "run even though the environment contradicts configs/v1-m3.toml (item 19). "
            "Without it, a `VANTA_` variable the file freezes to a different value — or "
            "freezes to `unset` and the caller set — refuses the run before the first VM "
            "boots, because the rows would claim a configuration they did not measure. "
            "The waiver and the values it waived are recorded on every row "
            "(`config_override`, `config_overrides`), so such a batch still cannot be "
            "pooled with one that did not need it."
        ),
    )
    run.add_argument(
        "--show-hints",
        action="store_true",
        help=(
            "send each report's comments on top of its content (§2.4.4). Off by default and "
            "off for every batch a verdict rests on: the comments are pre-fix hints_text and "
            "they name the fix often enough that matplotlib/1's maintainer comment names a "
            "file and a line range. Setting it is an ablation arm — the rows record "
            "show_hints=true and take a `#hints` job key, so they can never be resumed as, or "
            "compared by name to, the default run."
        ),
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "force a FRESH suite run instead of serving it from the M3 suite cache "
            "(items S1–S3), so a cached result can be checked against a fresh one on "
            "demand. The flag does not make the cache invisible: the entry is still "
            "read, the row records that it WOULD have hit (`off:no_cache`, with "
            "`would_hit` in the row's own `cache` list), and the fresh payload then "
            "replaces the entry it could not use."
        ),
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "the maximum number of rows in flight at once across all bugs (default 1). "
            "Same meaning as `queue --jobs` and `prepare --prepare-jobs`: the memory "
            "and core budget can admit fewer, never more. `--jobs 1` is today's "
            "behaviour exactly. Requires VANTA_LLM_BACKEND=openai to exceed 1 (§14.33)."
        ),
    )
    run.add_argument(
        "--peak-vms",
        type=int,
        default=budget_mod.DEFAULT_PEAK_VMS,
        metavar="N",
        help=(
            "the scheduler's conservative charge for a row: assume up to N of its "
            f"microVMs run at once (default {budget_mod.DEFAULT_PEAK_VMS}), so a row is "
            "charged --peak-vms × its VM memory. Accounting only — no VM's memory or "
            "vCPUs change; each row records the peak it actually reached so this can be "
            "tuned from real runs."
        ),
    )
    run.add_argument(
        "--issues-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "where the frozen issue reports are read from. Flag absent is the "
            "committed `plans/bugsinpy_issues`, byte-for-byte today's path. A held-out "
            "split points this at its gitignored freeze directory (e.g. "
            "`plans/private/bugsinpy_issues`). When given the directory is REQUIRED: a "
            "report missing there refuses the bug by name — the run never falls back to "
            "the default folder and never re-fetches a report live (§2.4)."
        ),
    )
    run.set_defaults(func=cmd_run)

    queue = sub.add_parser(
        "queue",
        help=(
            "run a prepared batch's bugs several at a time (item 13): up to --jobs "
            "bugs of one wave at once, one shared spend cap, a STOP file, resumable"
        ),
    )
    queue.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    queue.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="the batch directory `prepare` created — the resume cursor lives there",
    )
    queue.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "M3 item AA4 — prepare and run in ONE invocation: each selected bug is "
            "prepared, and the moment its prepare is final and frozen its rows start, "
            "while the other bugs of its wave are still preparing. Waves stay "
            "sequential. One scheduler and one spend cap govern the whole pipeline, "
            "and the batch's prepare.json is written when the last wave is prepared. "
            "Off by default: without it a queue reads a batch `prepare` already "
            "finished, exactly as before."
        ),
    )
    queue.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="run only these bugs (repeatable)",
    )
    queue.add_argument("--waves", default="", help="comma-separated wave numbers")
    queue.add_argument(
        "--arms",
        action="append",
        default=[],
        metavar="SPELLING,...",
        help=(
            "the arm spellings to run, comma- or space-separated (repeatable; each "
            "value accumulates). Default: every arm the split declares. A spelling "
            "the split does not declare is refused — the split is the protocol and "
            "this flag is only a selection (§12.2.5)"
        ),
    )
    queue.add_argument(
        "--seeds",
        default="",
        metavar="N,...",
        help=(
            "the seeds to run, comma- or space-separated. Default: every seed the "
            "split declares; a value the split does not declare is refused"
        ),
    )
    queue.add_argument(
        "--vanta-bin",
        type=Path,
        default=runner_mod.DEFAULT_VANTA_BIN,
        help="the vanta_m3 binary to invoke (default: target/release/vanta_m3)",
    )
    queue.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Vanta's repair budget per run (default: 4, the binary's own). An arm "
            "that pins a budget owns it: an explicit different value is refused "
            "before the queue starts."
        ),
    )
    queue.add_argument(
        "--fixed-rounds",
        type=int,
        default=None,
        help=(
            "state that this queue's runs spend EXACTLY K repair attempts (M3 item 9). "
            "A fixed-round run IS the arm, so schedule b1-k4/b3-k4; the flag is a "
            "caller's statement and is refused — before the queue starts — when no "
            "scheduled arm is a fixed-round arm, or when it disagrees with the K the "
            "arm (and the frozen config's [fixed_rounds] table) declares."
        ),
    )
    queue.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "the maximum number of rows in flight at once across all bugs (default 1). "
            "Same meaning as `run --jobs` and `prepare --prepare-jobs`: the memory and "
            "core budget can admit fewer, never more. Waves stay sequential — the gate "
            "is a statement about the earlier stages (§8.1). Requires "
            "VANTA_LLM_BACKEND=openai to exceed 1 (§14.33)."
        ),
    )
    queue.add_argument(
        "--peak-vms",
        type=int,
        default=budget_mod.DEFAULT_PEAK_VMS,
        metavar="N",
        help=(
            "the scheduler's conservative charge for a row: assume up to N of its "
            f"microVMs run at once (default {budget_mod.DEFAULT_PEAK_VMS}), so a row is "
            "charged --peak-vms × its VM memory. Accounting only — no VM's memory or "
            "vCPUs change; each row records the peak it actually reached so this can be "
            "tuned from real runs."
        ),
    )
    queue.add_argument(
        "--disk-mb",
        type=int,
        default=prepare_mod.DISK_MB,
        help=(
            "shuru disk in MB for every VM this stage boots; the default matches "
            f"the suites prepare ran ({prepare_mod.DISK_MB})"
        ),
    )
    queue.add_argument(
        "--plan",
        action="store_true",
        help="print the schedule and the queue's task list, then exit",
    )
    queue.add_argument(
        "--allow-override",
        action="store_true",
        help=(
            "run even though the environment contradicts configs/v1-m3.toml (item 19); "
            "the waiver and the values it waived are recorded on every row"
        ),
    )
    queue.add_argument(
        "--show-hints",
        action="store_true",
        help=(
            "send each report's comments on top of its content (§2.4.4). Off by "
            "default and off for every batch a verdict rests on; such rows take a "
            "`#hints` job key and cannot be resumed as, or compared by name to, a "
            "default run."
        ),
    )
    queue.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "force a FRESH suite run instead of serving it from the M3 suite cache "
            "(items S1–S3), so a cached result can be checked against a fresh one on "
            "demand. The flag does not make the cache invisible: the entry is still "
            "read, the row records that it WOULD have hit (`off:no_cache`, with "
            "`would_hit` in the row's own `cache` list), and the fresh payload then "
            "replaces the entry it could not use."
        ),
    )
    # M3 item AA4 — the prepare-side flags, for `--pipeline`. They are exactly the
    # ones `prepare` takes and mean the same thing; without `--pipeline` they are
    # read by nothing, and a queue that reads a prepared batch needs none of them.
    queue.add_argument(
        "--prepare-jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "with --pipeline: the maximum number of prepared-suite runs in flight at "
            "once (M3 item S5, default 1), the same meaning as `prepare "
            "--prepare-jobs`. The prepare pool is N wide and the row pool is "
            "`--jobs` wide; both draw on the ONE scheduler, whose size is the wider "
            "of the two. Ignored without --pipeline, which prepares nothing."
        ),
    )
    queue.add_argument(
        "--refreeze",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help=(
            "with --pipeline: re-fetch and rewrite this bug's frozen issue report "
            "(prepare's own flag, §2.4.3). Repeatable, and it selects only the bugs it "
            "names, in full. Ignored without --pipeline."
        ),
    )
    queue.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        metavar="TOKENS",
        help=(
            "with --pipeline: the context window the context gate is sized against "
            "(§1.6.1, prepare's own flag). Checked against VANTA_LLM_CONTEXT_TOKENS "
            "and the server's /props n_ctx rather than overriding them. Ignored "
            "without --pipeline."
        ),
    )
    queue.add_argument(
        "--reserves",
        choices=("measure", "skip"),
        default="measure",
        help=(
            "with --pipeline: what to do with role=reserve bugs — measure their "
            "context only (default) or skip them (prepare's own flag). Ignored "
            "without --pipeline."
        ),
    )
    queue.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "with --pipeline: context-measure every selected bug without running a "
            "suite or booting a VM (prepare's own flag, §1.6). Ignored without "
            "--pipeline."
        ),
    )
    queue.add_argument(
        "--traceback", action="store_true", help="print tracebacks (with --pipeline)"
    )
    queue.add_argument(
        "--issues-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "where the frozen issue reports are read from (and, with --pipeline, "
            "frozen into). Flag absent is the committed `plans/bugsinpy_issues`, "
            "byte-for-byte today's path. A held-out split points this at its gitignored "
            "freeze directory (e.g. `plans/private/bugsinpy_issues`). When given the "
            "directory is REQUIRED: a report missing there refuses the bug by name — the "
            "run never falls back to the default folder and never re-fetches a report "
            "live (§2.4)."
        ),
    )
    queue.set_defaults(func=cmd_queue)

    pool = sub.add_parser(
        "pool",
        help=(
            "write and hash-lock a pool: `--mutants` the reference-fix mutant pool "
            "(item 5), `--build` the assembled review pool reviews read (item 6); "
            "grading runs in Shuru"
        ),
    )
    pool.add_argument(
        "--split",
        type=Path,
        default=None,
        help=(
            "the split to read the batches with. Omit it and each batch's own "
            "recorded split (its prepare.json) is used; a value that disagrees with "
            "that record is refused (#E)"
        ),
    )
    pool.add_argument(
        "--batch",
        action="append",
        default=[],
        metavar="[TAG=]DIR",
        help=(
            "a prepared batch directory (repeatable). `--mutants`: a bare DIR, exactly "
            "one. `--build`: TAG=DIR, at least one, TAG one of "
            f"{pool_mod.TAG_ORDER}, all sharing the priority order left to right"
        ),
    )
    pool.add_argument(
        "--mutants",
        action="store_true",
        help=(
            "write the reference-fix mutant pool (pool/<bug>.mutants.json, item 5). "
            "The assembled review pool (pool/<bug>.json) is written by --build."
        ),
    )
    pool.add_argument(
        "--build",
        action="store_true",
        help=(
            "assemble, grade and hash-lock the review pool (pool/<bug>.json, item 6) "
            "from every tagged batch's Creator attempts; needs --out"
        ),
    )
    pool.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "where --build writes pool/<bug>.json (required for --build, no default): "
            "the pool spans several batches, so it does not live inside any of them"
        ),
    )
    pool.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "with --build: replace an existing locked pool/<bug>.json, recording the "
            "replaced file's hash as `previous-sha256`. Without it an existing pool is "
            "never overwritten."
        ),
    )
    pool.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="pool only these bugs (repeatable)",
    )
    pool.add_argument("--waves", default="", help="comma-separated wave numbers")
    pool.add_argument(
        "--disk-mb",
        type=int,
        default=prepare_mod.DISK_MB,
        help=(
            "shuru disk in MB for every VM this stage boots; the default matches "
            f"the suites prepare ran ({prepare_mod.DISK_MB})"
        ),
    )
    pool.add_argument(
        "--plan",
        action="store_true",
        help=(
            "print each bug's candidate counts and the first candidates in sort order "
            "(for --build: per tagged batch's attempts), and exit; no VM"
        ),
    )
    pool.set_defaults(func=cmd_pool)

    rep = sub.add_parser(
        "report",
        help="re-derive the batch digest, the label file and the per-config metrics",
    )
    rep.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="the batch directory `run` wrote its manifest into",
    )
    rep.set_defaults(func=cmd_report)

    rpl = sub.add_parser(
        "replay",
        help=(
            "replay a finished repair row's stored attempts onto one shared test set "
            "in Shuru, writing the X[test][k] matrix under <batch>/replay (item 10)"
        ),
    )
    rpl.add_argument(
        "--batch",
        action="append",
        required=True,
        default=[],
        metavar="DIR",
        help="the batch directory `run` wrote its manifest into (exactly one)",
    )
    rpl.add_argument(
        "--split",
        type=Path,
        default=None,
        help=(
            "the split to read the batch with. Omit it and the batch's own recorded "
            "split (its prepare.json) is used, exactly as for `pool`"
        ),
    )
    rpl.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROJECT/BUG",
        help="replay only these bugs (repeatable)",
    )
    rpl.add_argument("--waves", default="", help="comma-separated wave numbers")
    rpl.add_argument(
        "--views",
        default=",".join(replay_mod.DEFAULT_VIEWS),
        help=(
            "comma-separated views to run (default "
            f"'{','.join(replay_mod.DEFAULT_VIEWS)}'): 'generated' is S (the native "
            "suite plus the admitted generated tests), 'oracle' the hidden nodeids"
        ),
    )
    rpl.add_argument(
        "--seed",
        type=int,
        default=replay_mod.DEFAULT_SEED,
        help=(
            "the seed for the re-execution flake sample (default "
            f"{replay_mod.DEFAULT_SEED}); the draw is a pure function of it"
        ),
    )
    rpl.add_argument(
        "--sample-rate",
        type=float,
        default=replay_mod.DEFAULT_SAMPLE_RATE,
        help=(
            "the share of cells re-executed to catch a flaky one (default "
            f"{replay_mod.DEFAULT_SAMPLE_RATE}); 0 draws none"
        ),
    )
    rpl.add_argument(
        "--force",
        action="store_true",
        help="redo a row whose <batch>/replay/<row-stem>.json already exists",
    )
    rpl.add_argument(
        "--disk-mb",
        type=int,
        default=prepare_mod.DISK_MB,
        help=(
            "shuru disk in MB for every VM this stage boots; the default matches "
            f"the suites prepare ran ({prepare_mod.DISK_MB})"
        ),
    )
    rpl.set_defaults(func=cmd_replay)

    loc = sub.add_parser(
        "localize",
        help=(
            "dev split only: did the selection a prepare measured contain the files "
            "the fix commit touched? (ranking quality, offline)"
        ),
    )
    loc.add_argument(
        "--split",
        type=Path,
        default=DEV_SPLIT,
        help=f"the dev split (default and only accepted value: {DEV_SPLIT})",
    )
    loc.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="a prepare batch directory — it must hold context_report.json",
    )
    loc.set_defaults(func=cmd_localize)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except EXPECTED_ERRORS as exc:
        # A bake that failed still owes the reader the summary of what it built,
        # deleted, skipped and failed — including the layers a failed interpreter
        # layer left unstarted — and every other expected error owes the reader its
        # own message; both are printed, not traced (see EXPECTED_ERRORS).
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_BUG_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
