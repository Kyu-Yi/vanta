#!/usr/bin/env python3
"""Feedback-loop harness for BugsInPy — the one command (design §7.1).

    python3 scripts/bugsinpy_run.py bake    --pythons 3.8 --yes   # §11 step 3
    python3 scripts/bugsinpy_run.py prepare --split plans/bugsinpy_split.json

``bake`` builds the Shuru checkpoints the split names — one interpreter layer per
Python minor version plus a thin per-bug dependency layer — and is the only stage
that needs network. ``prepare`` is the stage after it: it reads BugsInPy's own
metadata, cuts the SWE-bench trees, derives ``FAIL_TO_PASS`` / ``PASS_TO_PASS``,
strips the hidden tests, runs the context precheck and the leak gate, and writes
the derived half back into the committed split. Neither makes LLM calls or runs
Vanta — ``run`` (the batch runner) and ``report`` (metrics) are later stages.

Nothing third-party executes on the host. Checkouts and tree surgery are host-side
``git`` and filesystem work; every command that imports the project under test is
issued inside Shuru (design C3, ``suite.py``).

Exit codes: ``0`` everything prepared, ``1`` a bug or several did not prepare,
``2`` the harness itself could not proceed (missing checkpoint, moved metadata).
"""

from __future__ import annotations

import argparse
import json
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
from bugsinpy import prepare as prepare_mod  # noqa: E402
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
) -> list[tuple[dict[str, Any], bool]]:
    """``[(entry, full)]`` in the committed file order.

    ``full`` means the whole §1.3 derivation — three suites, twice each, inside
    Shuru. A reserve is ``full=False``: it is checked out and context-measured so
    that ``context_sizes.md`` can justify promoting it, but its suites are not run
    until someone changes its ``role`` and re-runs prepare (§8). Naming a bug with
    ``--only`` is an explicit request, so it is always ``full``.
    """
    entries = split.bugs

    if only:
        known = {bug_key(entry) for entry in entries}
        unknown = sorted(set(only) - known)
        if unknown:
            raise SplitError(f"--only names bugs that are not in the split: {unknown}")
        wanted = set(only)
        return [(entry, True) for entry in entries if bug_key(entry) in wanted]

    if waves:
        wanted_waves = set(waves)
        entries = [e for e in entries if int(e["wave"]) in wanted_waves]

    selected: list[tuple[dict[str, Any], bool]] = []
    for entry in entries:
        if entry.get("role", "pilot") == "pilot":
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
        split, only=args.only, waves=parse_waves(args.waves), reserves=args.reserves
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
                full=full,
            )
        except Exception as exc:  # one bug must not take the batch down
            detail = f"{type(exc).__name__}: {exc}"
            broken.append((key, detail))
            split.set_derived(
                entry["project"],
                int(entry["bug_id"]),
                gate_failures=[detail],
            )
            print(f"[{key}] FAILED — {detail}", flush=True)
            if args.traceback:
                traceback.print_exc()
            continue

        preps.append(prep)
        split.set_derived(
            entry["project"], int(entry["bug_id"]), **prepare_mod.derive_verdict(prep)
        )
        print(f"[{key}] {'ok' if prep.runnable else 'not runnable'}", flush=True)

    for prep in preps:
        if prep.issue is not None:
            prepare_mod.write_leak_gate(batch_dir, prep)
    prepare_mod.write_issue_texts(batch_dir, preps)
    prepare_mod.write_context_sizes(batch_dir, preps)

    summary = prepare_mod.render_summary(preps)
    (batch_dir / "prepare_summary.md").write_text(summary, encoding="utf-8")
    (batch_dir / "prepare.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "prepare",
                "created_utc": utc_now(),
                "batch_dir": str(batch_dir),
                "split": str(split_path),
                "split_sha256_before": sha256_text(split.to_json()),
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

    split.save(split_path)

    print()
    print(summary)
    print(f"batch dir : {batch_dir}")
    print(f"split     : {split_path}")
    for artifact in ("context_sizes.md", "issue_texts_all.md", "prepare_summary.md"):
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

    layers = bake_mod.layer_recipes(
        split, pythons=minors, entries=entries, bugsinpy_root=bugsinpy_root
    )
    for name, base, script in layers:
        if not args.force and bake_mod.frozen(name):
            print(f"[{name}] already baked — skipping (use --force to rebuild)", flush=True)
            continue
        print(f"[{name}] baking from {base} ...", flush=True)
        result = bake_mod.bake(name, script, base=base, disk_mb=args.disk_mb)
        if not result.ok:
            detail = (result.stdout + result.stderr).strip()[-2000:]
            raise ShuruError(
                f"baking {name} failed (exit {result.exit_code}): {detail}"
            )
        print(f"[{name}] ok — marker {bake_mod.MARKER} verified", flush=True)

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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SplitError as exc:
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
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_BUG_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
