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
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import bake as bake_mod  # noqa: E402
from bugsinpy import bugsinpy_meta as meta_mod  # noqa: E402
from bugsinpy import manifest as manifest_mod  # noqa: E402
from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
from bugsinpy import split as split_mod  # noqa: E402
from bugsinpy.shuru import ShuruError  # noqa: E402
from bugsinpy.split import Split, SplitError, load_split, sha256_text  # noqa: E402

DEFAULT_SPLIT = REPO_ROOT / "plans" / "bugsinpy_split.json"
RUNS_DIR = REPO_ROOT / "runs" / "bugsinpy"

EXIT_OK = 0
EXIT_BUG_FAILED = 1
EXIT_HARNESS_FAILED = 2


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# selection — what the batch is, and how much of it to prepare
# ---------------------------------------------------------------------------
def bug_key(entry: dict[str, Any]) -> str:
    return f"{entry['project']}/{entry['bug_id']}"


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

    # Preflight the checkpoints for everything that will run a suite, before the
    # first checkout: a missing layer is a batch-level mistake (§3.3).
    prepare_mod.require_checkpoints(
        [entry["checkpoint"] for entry, full in selected if full]
    )
    # And the dependency provenance the bake recorded. prepare installs nothing; it
    # reads back what the layer was built from, so a layer baked without a record —
    # or by a harness that did not record one — is a batch-level mistake too, not a
    # per-bug surprise three suite runs later.
    deps_dir = RUNS_DIR / "deps"
    bake_mod.require_deps(deps_dir, [entry["checkpoint"] for entry, full in selected if full])

    batch_dir: Path = (args.batch or (RUNS_DIR / f"prepare-{utc_stamp()}")).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = RUNS_DIR / "repos"

    preps: list[prepare_mod.BugPrep] = []
    broken: list[tuple[str, str]] = []

    for entry, full in selected:
        key = bug_key(entry)
        project = entry["project"]
        print(f"[{key}] {'full' if full else 'context-only'} ...", flush=True)
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
    prepare_mod.write_context_report(batch_dir, preps)

    summary = prepare_mod.render_summary(preps)
    (batch_dir / "prepare_summary.md").write_text(summary, encoding="utf-8")
    # The split is an input, never an output: how many bugs this run measured, and
    # which of them it rejected, belongs in the batch's own artifacts. Rehashing the
    # file after the work is the evidence that nothing here wrote to it.
    split_sha_before = split_mod.sha256_file(split_path)
    (batch_dir / "prepare.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "prepare",
                "created_utc": utc_now(),
                "batch_dir": str(batch_dir),
                "split": str(split_path),
                "split_sha256_before": sha256_text(split.to_json()),
                "split_file_sha256_before": split_sha_before,
                "split_written": False,
                "bugsinpy_commit": commit,
                "bugsinpy_root": str(bugsinpy_root),
                "selected": [bug_key(entry) for entry, _ in selected],
                "breaks": [{"bug": key, "detail": detail} for key, detail in broken],
                "bugs": [prep.as_dict() for prep in preps],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
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
    """
    split = load_split(args.split.resolve())
    pythons = parse_pythons(args.pythons, split)
    roles = {"pilot", "reserve"} if args.reserves else {"pilot"}
    minors, entries = bake_mod.select_layers(
        split, pythons=pythons, only=args.only, roles=roles
    )
    bugsinpy_root = (REPO_ROOT / split.bugsinpy_root).resolve()

    if not args.yes:
        sys.stdout.write(
            bake_mod.render_plan(split, pythons=minors, entries=entries)
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
    # baked, so a bug whose four declared sources contradict each other, or that
    # names a release which did not exist yet at its commit date, fails the whole
    # bake here rather than producing a layer nobody can defend. This is the only
    # place the harness talks to PyPI; every decision it makes is taken from the
    # dates that call returns.
    repo_cache = RUNS_DIR / "repos"
    deps_dir = RUNS_DIR / "deps"

    def repo_for(project: str) -> Path:
        url = meta_mod.project_github_url(bugsinpy_root, project)
        if not url:
            raise SplitError(f"no github_url for {project} in project.info")
        return tree_cache(repo_cache, project, url)

    bug_deps = bake_mod.resolve_bug_deps(
        split, entries, bugsinpy_root=bugsinpy_root, repo_for=repo_for
    )
    for item in bug_deps:
        path = bake_mod.write_deps(deps_dir, item)
        print(f"[{item.checkpoint}] installs: {', '.join(item.installs) or '—'}")
        print(f"[{item.checkpoint}] provenance: {path}", flush=True)

    layers = bake_mod.layer_recipes(
        split, pythons=minors, entries=entries, deps=bug_deps
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

    summary = bake_mod.run_bake_actions(
        actions,
        children=children,
        build=lambda name, script, base: bake_mod.bake(
            name, script, base=base, disk_mb=args.disk_mb
        ),
    )
    print(summary, flush=True)
    print("bake complete", flush=True)
    return EXIT_OK


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
    preps, entries = runner_mod.load_prepared(
        batch_dir=batch_dir, split=split, bugsinpy_root=bugsinpy_root
    )
    waves = parse_waves(args.waves)
    selected = select_preps(preps, only=args.only, waves=waves)
    vanta_bin = runner_mod.require_vanta_bin(args.vanta_bin.resolve())

    code, results = runner_mod.run_batch(
        batch_dir=batch_dir,
        batch_id=batch_dir.name,
        split=split,
        preps=selected,
        entries=entries,
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
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except bake_mod.BakeError as exc:
        # A bake that failed still owes the reader the summary of what it rebuilt,
        # skipped and failed, so the message is printed rather than traced.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except prepare_mod.PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except meta_mod.MetadataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except ShuruError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except runner_mod.RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_HARNESS_FAILED
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_BUG_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
