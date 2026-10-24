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
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import bake as bake_mod  # noqa: E402
from bugsinpy import bugsinpy_meta as meta_mod  # noqa: E402
from bugsinpy import compact as compact_mod  # noqa: E402
from bugsinpy import context as context_mod  # noqa: E402
from bugsinpy import issue as issue_mod  # noqa: E402
from bugsinpy import manifest as manifest_mod  # noqa: E402
from bugsinpy import oracle as oracle_mod  # noqa: E402
from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import records as records_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
from bugsinpy import split as split_mod  # noqa: E402
from bugsinpy import suite as suite_mod  # noqa: E402
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
    context_mod.ContextError,
    oracle_mod.OracleError,
    suite_mod.SuiteError,
    tree_mod.TreeError,
    tree_mod.StripError,
    issue_mod.IssueError,
    compact_mod.CompactError,
    records_mod.RecordError,
)
"""Every failure the harness raises on purpose, and the reason the CLI prints them.

[VANTA-WHY] Each of these already carries the whole report in its message — which
stage refused, what it saw, what to do — so a traceback adds frames the reader has
to walk past to reach the sentence that matters, and it hides the exit code that
says *this is a refusal, not a crash*. An error class missing from this tuple exits
as an unhandled exception instead, which is how ``ContextError`` used to look.
"""

DEFAULT_SPLIT = REPO_ROOT / "plans" / "bugsinpy_split.json"
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
) -> list[tuple[dict[str, Any], bool]]:
    """``[(entry, full)]`` in the committed file order.

    ``full`` means the whole §1.3 derivation — three suites, twice each, inside
    Shuru. A reserve is ``full=False``: it is checked out and context-measured so
    that ``context_sizes.md`` can justify promoting it, but its suites are not run
    until someone changes its ``role`` and re-runs prepare (§8). Naming a bug with
    ``--only`` is an explicit request, so it is always ``full``.

    ``measure_only`` forces every candidate to ``full=False`` — the checkout, the
    SWE-bench trees and the §1.6 context precheck, and nothing that boots a VM.
    That is what makes the pilot set a *measured* choice: the sizes of all
    candidates, pilots and reserves, are available before a single checkpoint is
    baked or a wave scheduled. The numbers it produces are the pre-strip tree's
    (§1.6 step 1), so they are an upper bound on what Vanta is finally handed.
    """
    entries = split.bugs

    if only:
        known = {bug_key(entry) for entry in entries}
        unknown = sorted(set(only) - known)
        if unknown:
            raise SplitError(f"--only names bugs that are not in the split: {unknown}")
        wanted = set(only)
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
def cmd_prepare(args: argparse.Namespace) -> int:
    split_path: Path = args.split.resolve()
    split = load_split(split_path)
    selected = select(
        split,
        only=args.only,
        waves=parse_waves(args.waves),
        reserves=args.reserves,
        measure_only=args.measure_only,
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
    # hours of suite runs rather than before them.
    n_ctx, n_ctx_source = context_mod.explicit_context_window(n_ctx=args.n_ctx)
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

    batch_dir: Path = (args.batch or (RUNS_DIR / f"prepare-{utc_stamp()}")).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = RUNS_DIR / "repos"

    preps: list[prepare_mod.BugPrep] = []
    broken: list[tuple[str, str]] = []

    for entry, full in selected:
        key = bug_key(entry)
        project = entry["project"]
        print(f"[{key}] {'full' if full else 'context-only'} ...", flush=True)
        checkpoint = entry["checkpoint"]
        if full and checkpoint in missing_layers:
            detail = (
                f"PrepareError: checkpoint {checkpoint} does not exist. Bake it first "
                "(design §11 step 3 for py3.8, §3.2 for the older interpreters)."
            )
            broken.append((key, detail))
            print(f"[{key}] FAILED — {detail}", flush=True)
            continue
        if full and checkpoint in missing_provenance:
            detail = (
                f"BakeError: no dependency provenance for {checkpoint} under {deps_dir}. "
                "Its dependencies could not be read at bake time; re-run bake for it."
            )
            broken.append((key, detail))
            print(f"[{key}] FAILED — {detail}", flush=True)
            continue
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
                full=full,
                n_ctx=n_ctx,
                n_ctx_source=n_ctx_source,
                vanta_bin=vanta_bin,
            )
        except Exception as exc:  # one bug must not take the batch down
            detail = f"{type(exc).__name__}: {exc}"
            broken.append((key, detail))
            print(f"[{key}] FAILED — {detail}", flush=True)
            if args.traceback:
                traceback.print_exc()
            continue

        preps.append(prep)
        print(f"[{key}] {'ok' if prep.runnable else 'not runnable'}", flush=True)

    for prep in preps:
        if prep.issue is not None:
            prepare_mod.write_leak_gate(batch_dir, prep)
    prepare_mod.write_issue_texts(batch_dir, preps)
    prepare_mod.write_context_sizes(batch_dir, preps)
    prepare_mod.write_context_report(
        batch_dir, preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source
    )

    summary = prepare_mod.render_summary(preps, n_ctx=n_ctx, n_ctx_source=n_ctx_source)
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

    Every layer the build attempts leaves its whole output in
    ``runs/bugsinpy/bake-logs/<layer>.log`` (emptied at the start of each bake), and
    each failed layer's one-line entry in the failure report names its own error and
    points at that file.
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

    # Every layer's whole build log is written here by run_bake_actions, and the
    # directory is emptied first, so what it holds is always this bake's logs and a
    # layer that failed can be read in full rather than through the one error line the
    # summary prints.
    logs_dir = RUNS_DIR / "bake-logs"
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
    try:
        summary = bake_mod.run_bake_actions(
            actions,
            children=children,
            build=lambda name, script, base: bake_mod.bake(
                name, script, base=base, disk_mb=args.disk_mb
            ),
            logs_dir=logs_dir,
            own_pins=own_pins,
            record=lambda name, conflicts: bake_mod.record_pip_check(
                deps_dir, name, conflicts
            ),
            pins=pin_lists,
            provided=provided,
            dropped=dropped,
        )
    except BaseException:
        report_flagged()
        raise
    print(summary, flush=True)
    report_flagged()
    print("bake complete", flush=True)
    return EXIT_BUG_FAILED if flagged else EXIT_OK


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


def cmd_run(args: argparse.Namespace) -> int:
    """Run a prepared batch: repairs first, then both reviews, one job at a time.

    The repo-root check comes before anything else. ``shuru run --mount`` requires
    the mounted host path to be inside the process CWD, so a run started from the
    wrong directory would mount nothing and fail per job instead of once.
    """
    runner_mod.require_repo_cwd()
    split = load_split(args.split.resolve())
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()
    batch = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    # Before anything runs, and before --plan prints: a run whose window differs
    # from the one prepare sized against is measuring a different prompt than the
    # batch's verdicts describe (§1.6.1), and the cheap moment to notice is now.
    window_warning = runner_mod.context_window_warning(batch)
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
        disk_mb=args.disk_mb,
        plan_only=args.plan,
    )
    if args.plan:
        return EXIT_OK

    # The report is derived from the manifest on disk, not from this invocation's
    # return value: a resumed run must summarise the whole batch, not the tail of it.
    records = manifest_mod.Manifest(batch_dir).records()
    if records:
        labels_path, payload = runner_mod.write_batch_report(batch_dir, records)
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


def cmd_report(args: argparse.Namespace) -> int:
    """Re-derive the batch report from the manifest, then score the review rows.

    A pure function of the evidence: no VM, no LLM, and nothing written that is not
    a view of ``manifest.jsonl``. The scoring step is ``scripts/metrics.py`` itself
    (§5.4, §9) — the harness does not re-implement the classifier the rest of the
    project already uses, because two implementations of "what counts as a false
    approval" is how two numbers disagree.
    """
    batch_dir = runner_mod.require_batch_dir(args.batch.resolve())
    records = manifest_mod.Manifest(batch_dir).records()
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
        default=4,
        help="Vanta's repair budget per run (default: 4, the binary's own)",
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
    run.set_defaults(func=cmd_run)

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
