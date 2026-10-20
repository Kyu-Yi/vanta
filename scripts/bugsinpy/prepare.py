"""``prepare``: checkout, SWE-bench split, hidden-test derivation, leak gate (§11 step 2).

No LLM, no Vanta. What this module produces, per bug:

``src_buggy/`` ``src_fixed/`` ``oracle_tree/`` ``vanta_tree/`` ``vanta_clean/``
    the trees from §1.2, with the hidden tests removed from the last two;
``FAIL_TO_PASS`` / ``PASS_TO_PASS``
    derived by running three suites twice each (§1.3), so a flaky nodeid can never
    become the oracle;
``context_sizes.md``
    every candidate measured against the window (§1.6);
``issue_texts_all.md`` + ``leak_gate/<bug>.json``
    the issue text and every token the gate removed (§2.2);
``plans/bugsinpy_split.json``
    rewritten with the derived half filled in.

Every suite run happens inside Shuru — ``prepare`` is not exempt from C3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import bake as bake_mod
from . import context as ctx
from . import issue as issue_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .bugsinpy_meta import BugInfo, MetadataError, load_bug_info
from .split import Split, sha256_text

DISK_MB = 16384
"""Shuru disk size for suite runs. The base image is 1.5 GB and a checkpoint
inherits its layers, so this leaves room for a project's deps without growing the
copy-on-write clone needlessly."""


FULL = "full"
MEASURE_ONLY = "context_only"
"""A reserve is materialised and measured but its suites are not run (§8): that is
what ``context_sizes.md`` needs in order to justify swapping a rejected pilot bug
for a reserve, and it costs a checkout instead of hours of suite runs. Promoting a
reserve to the pilot means changing its ``role`` and re-running ``prepare``."""


class PrepareError(RuntimeError):
    """prepare could not proceed."""


@dataclass
class BugPrep:
    """Everything prepare learned about one bug."""

    bug: BugInfo
    root: Path
    role: str
    wave: int
    test_cmd: list[str]
    prepare_mode: str = FULL
    test_paths: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass_count: int = 0
    flaky_nodeids: list[str] = field(default_factory=list)
    preexisting_failures: list[str] = field(default_factory=list)
    native_failure_baseline: list[str] = field(default_factory=list)
    native_green_on_vanta_tree: bool = False
    hidden_files_deleted: list[str] = field(default_factory=list)
    partially_hidden_functions: list[str] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    context: Optional[ctx.ContextVerdict] = None
    issue: Optional[issue_mod.IssueText] = None
    deps: Optional[dict[str, Any]] = None

    @property
    def key(self) -> str:
        return self.bug.key

    @property
    def slug(self) -> str:
        """``PySnooper/1`` → ``PySnooper_1``, safe for a filename or a report stem."""
        return self.key.replace("/", "_")

    @property
    def runnable(self) -> bool:
        return (
            self.prepare_mode == FULL
            and not self.gate_failures
            and self.context is not None
            and self.context.context_ok
            and self.issue is not None
            and self.issue.ok
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "bug": self.key,
            "role": self.role,
            "wave": self.wave,
            "prepare_mode": self.prepare_mode,
            "runnable": self.runnable,
            "test_paths": self.test_paths,
            "fail_to_pass": sorted(self.fail_to_pass),
            "pass_to_pass_count": self.pass_to_pass_count,
            "flaky_nodeids": sorted(self.flaky_nodeids),
            "preexisting_failures": sorted(self.preexisting_failures),
            "native_failure_baseline": sorted(self.native_failure_baseline),
            "native_green_on_vanta_tree": self.native_green_on_vanta_tree,
            "hidden_files_deleted": self.hidden_files_deleted,
            "partially_hidden_functions": self.partially_hidden_functions,
            "gate_failures": self.gate_failures,
            "context": self.context.as_dict() if self.context else None,
            "issue": self.issue.as_dict() if self.issue else None,
            "deps": self.deps,
        }


# ---------------------------------------------------------------------------
# tree construction
# ---------------------------------------------------------------------------
def build_trees(bug: BugInfo, repo_cache: Path, bug_root: Path) -> list[str]:
    """Materialise ``src_buggy``, ``src_fixed`` and the pre-strip ``oracle_tree``."""
    bug_root.mkdir(parents=True, exist_ok=True)
    tree_mod.export_tree(repo_cache, bug.buggy_commit, bug_root / "src_buggy")
    tree_mod.export_tree(repo_cache, bug.fixed_commit, bug_root / "src_fixed")
    # `oracle_tree` is the pre-strip Vanta tree: buggy source + the FIXED commit's
    # test files, hidden tests intact. `vanta_tree`/`vanta_clean` are cut from it
    # after the hidden tests have been identified.
    test_paths = tree_mod.test_paths_from_fix(bug)
    tree_mod.compose_vanta_tree(
        bug_root / "src_buggy", bug_root / "src_fixed", bug_root / "oracle_tree", test_paths
    )
    return test_paths


# ---------------------------------------------------------------------------
# the dependency gate (§3.3) — before any oracle work
# ---------------------------------------------------------------------------
def check_deps(
    prep: BugPrep,
    *,
    checkpoint: str,
    batch_dir: Path,
    deps: dict[str, Any],
) -> set[str]:
    """Prove the bug's layer can run its suite *before* the derivation spends it.

    [VANTA-WHY] This is the measured failure it exists for. PySnooper/3's layer was
    baked from ``requirements.txt`` (0 bytes) and ``setup.sh`` (``setup.py install``
    plus ``decorator``), so ``python_toolbox`` and ``future`` were missing;
    ``tests/test_pysnooper.py`` died on import, ``--collect-only`` collected nothing,
    and prepare reported ``hidden-oracle derivation failed`` — a dependency problem
    reported as a property of the *bug*. The two need different fixes, and that
    message sent a debugging session after the wrong suspect.

    Both checks run inside the finished per-bug layer and both run the way the
    suites run — same interpreter, same cwd, same (absent) ``PYTHONPATH``, same
    pytest entry point:

    (a) the oracle tree's tests collect, with no collection error;
    (b) the project's own package imports **from the tree under test**, not from
        site-packages, where a ``setup.py install`` would have left a frozen copy
        that makes every patch grade the same.

    The two failures are raised as separate messages, so neither can be mistaken
    for the other or for the derivation. The collected nodeids are returned, so the
    derivation can use them instead of collecting the same tree a second time.
    """
    module = str(deps["import_name"])
    tree_rel = _rel(prep, "oracle_tree")

    try:
        nodeids = suite_mod.collect_only(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=prep.test_cmd,
            disk_mb=DISK_MB,
        )
    except suite_mod.SuiteError as exc:
        raise suite_mod.SuiteError(f"deps: collection failed: {exc}") from exc

    try:
        where = suite_mod.import_probe(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=tree_rel,
            test_cmd=prep.test_cmd,
            module=module,
            disk_mb=DISK_MB,
        )
    except suite_mod.SuiteError as exc:
        raise suite_mod.SuiteError(
            f"deps: {module} does not import from the tree under test: {exc}"
        ) from exc

    root = suite_mod.guest_tree_root(tree_rel)
    if not where.startswith(root + "/"):
        raise suite_mod.SuiteError(
            f"deps: package imports from site-packages, not the tree: {module} "
            f"loaded {where}, which is not under {root}"
        )
    return nodeids


# ---------------------------------------------------------------------------
# FAIL_TO_PASS / PASS_TO_PASS derivation (§1.3)
# ---------------------------------------------------------------------------
def derive_hidden(
    prep: BugPrep,
    *,
    checkpoint: str,
    batch_dir: Path,
    collected: Optional[set[str]] = None,
) -> None:
    if collected is None:
        suite_mod.collect_only(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=_rel(prep, "oracle_tree"),
            test_cmd=prep.test_cmd,
            disk_mb=DISK_MB,
        )

    buggy = _suite(checkpoint, batch_dir, prep, "src_buggy", "src_buggy")
    before = _suite(checkpoint, batch_dir, prep, "oracle_tree", "oracle")
    after = _suite(checkpoint, batch_dir, prep, "src_fixed", "src_fixed")

    flaky = (
        buggy.flaky_nodeids
        | buggy.missing_from_one_run
        | before.flaky_nodeids
        | before.missing_from_one_run
        | after.flaky_nodeids
        | after.missing_from_one_run
    )
    prep.flaky_nodeids = sorted(flaky)

    f_buggy_pre = buggy.stable_failures
    f_before = before.stable_failures
    f_after = after.stable_failures
    prep.preexisting_failures = sorted(f_buggy_pre)
    prep.fail_to_pass = sorted(f_before - f_after)

    if not prep.fail_to_pass:
        prep.gate_failures.append(
            "empty FAIL_TO_PASS: the bug is not reproduced by the fixed commit's test "
            "files on the buggy source, so there is no oracle to grade against"
        )

    unowned = [
        nodeid
        for nodeid in prep.fail_to_pass
        if tree_mod.nodeid_file(nodeid) != prep.bug.test_file
        and not _named_by_run_test(nodeid, prep.bug)
    ]
    if unowned:
        prep.gate_failures.append(
            "FAIL_TO_PASS nodeid(s) outside bug.info's test_file and not named by "
            f"run_test.sh: {unowned} — the suite runner is not running what BugsInPy intends"
        )

    fixed_only = sorted(f_after - f_buggy_pre)
    if fixed_only:
        prep.gate_failures.append(
            "the fixed tree fails tests the buggy tree does not — the relaxed fixed-tree "
            f"gate (F_after \\ F_buggy_pre == empty) is violated: {fixed_only}"
        )


def _named_by_run_test(nodeid: str, bug: BugInfo) -> bool:
    command = bug.run_test_command() or ""
    chain = tree_mod.nodeid_chain(nodeid)
    return bool(chain) and chain[-1] in command


def _suite(
    checkpoint: str,
    batch_dir: Path,
    prep: BugPrep,
    tree_name: str,
    stem_suffix: str,
) -> suite_mod.SuitePair:
    return suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=batch_dir,
        tree_rel=_rel(prep, tree_name),
        test_cmd=prep.test_cmd,
        report_rel_stem=f"reports/{prep.slug}-{stem_suffix}",
        disk_mb=DISK_MB,
    )


def strip_and_baseline(prep: BugPrep, *, checkpoint: str, batch_dir: Path) -> None:
    """Cut ``vanta_tree``/``vanta_clean`` from ``oracle_tree`` and re-run the suite.

    The pre-strip collected set is what decides "this file is entirely hidden" in
    the stripper, so it is read from the oracle run's report rather than guessed.
    """
    oracle = prep.root / "oracle_tree"
    vanta = prep.root / "vanta_tree"

    report = _read_report(batch_dir, f"{prep.slug}-oracle")
    collected_nodeids = [
        test["nodeid"]
        for test in report.get("tests") or []
        if isinstance(test.get("nodeid"), str)
    ]
    collected_by_file = suite_mod.collect_by_file(collected_nodeids)

    tree_mod.copy_tree(oracle, vanta)
    try:
        strip = tree_mod.strip_hidden_tests(vanta, prep.fail_to_pass, collected_by_file)
    except tree_mod.StripError as exc:
        prep.gate_failures.append(f"hidden-test strip failed: {exc}")
        return

    prep.hidden_files_deleted = strip.deleted_files
    prep.partially_hidden_functions = strip.partially_hidden_functions

    after_strip = _suite(checkpoint, batch_dir, prep, "vanta_tree", "vanta")
    prep.native_failure_baseline = sorted(after_strip.stable_failures)
    prep.pass_to_pass_count = len(after_strip.stable_collected) - len(
        prep.native_failure_baseline
    )
    prep.native_green_on_vanta_tree = not prep.native_failure_baseline

    tree_mod.copy_tree(vanta, prep.root / "vanta_clean")


# ---------------------------------------------------------------------------
# issue text
# ---------------------------------------------------------------------------
def build_issue(prep: BugPrep, *, repo_cache: Path, batch_dir: Path) -> None:
    message = issue_mod.fix_commit_message(repo_cache, prep.bug.fixed_commit)
    report = _read_report(batch_dir, f"{prep.slug}-oracle")
    kinds = issue_mod.failure_kinds_from_report(report)

    hidden_sources = issue_mod.hidden_test_sources(
        prep.root / "oracle_tree", prep.fail_to_pass
    )
    answer_tokens = issue_mod.answer_token_counts(
        prep.bug.bug_patch_txt, hidden_sources, prep.fail_to_pass
    )
    tree_tokens = issue_mod.vanta_tree_token_counts(prep.root / "vanta_tree")
    prep.issue = issue_mod.build_issue_text(
        stripped_message=message,
        failure_kinds=kinds,
        answer_tokens=answer_tokens,
        tree_tokens=tree_tokens,
    )


# ---------------------------------------------------------------------------
# the review artifacts
# ---------------------------------------------------------------------------
def write_leak_gate(batch_dir: Path, prep: BugPrep) -> Path:
    path = batch_dir / "leak_gate" / f"{prep.slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = prep.issue.as_dict() if prep.issue else {"ok": False, "reason": "no issue text"}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_issue_texts(batch_dir: Path, preps: list[BugPrep]) -> Path:
    texts_dir = batch_dir / "issue_texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    for prep in preps:
        if prep.issue is None:
            continue
        body = [
            f"# {prep.key}",
            "",
            f"- role: `{prep.role}`  wave: **{prep.wave}**",
            f"- python: `{prep.bug.python_version}`  test_file: `{prep.bug.test_file}`",
            f"- FAIL_TO_PASS: {prep.fail_to_pass}",
            f"- failure kinds: {prep.issue.failure_kinds}",
            f"- gate: {'ok' if prep.issue.ok else '**NEEDS DECISION**'}",
            "",
            "## Frozen text (post-leak-gate)",
            "",
            prep.issue.text,
            "",
            "## Pre-gate text (before the leak gate)",
            "",
            prep.issue.pre_gate,
            "",
            "## Tokens the gate removed",
            "",
        ]
        if prep.issue.leak.removals:
            body += [
                "| token | class | occurrences in answer | in vanta_tree |",
                "|---|---|---|---|",
            ]
            body += [
                f"| `{r.token}` | {r.token_class} | {r.occurrences_in_answer} | "
                f"{r.occurrences_in_vanta_tree} |"
                for r in prep.issue.leak.removals
            ]
        else:
            body.append("— none —")
        if prep.issue.leak.ungated_prose:
            body += [
                "",
                "Prose tokens that exist only in the answer and were **not** gated "
                "(short, unadorned words carry no answer; listed so the choice is "
                "visible rather than implicit):",
                "",
                ", ".join(f"`{t}`" for t in sorted(set(prep.issue.leak.ungated_prose))),
            ]
        body.append("")
        (texts_dir / f"{prep.slug}.md").write_text("\n".join(body), encoding="utf-8")

    all_path = batch_dir / "issue_texts_all.md"
    needs = [prep for prep in preps if prep.issue is not None and not prep.issue.ok]
    summary = [
        "# Issue texts — all candidates, for hand review before any run",
        "",
        f"Generated by `prepare` on {utc_now()}.",
        "Read this before committing the split: it is the audit that costs nothing at run time.",
        "",
        "## NEEDS DECISION",
        "",
    ]
    if needs:
        summary += [
            "These texts are not actionable after the leak gate. Supply a replacement "
            "text or drop the bug; the harness does not guess.",
            "",
        ]
        summary += [f"- **{prep.key}** — {prep.issue.needs_decision}" for prep in needs]
    else:
        summary.append("None — every candidate produced an actionable text.")
    summary += ["", "---", ""]

    for prep in preps:
        if prep.issue is None:
            continue
        summary += [
            f"## {prep.key} ({prep.role}, wave {prep.wave})",
            "",
            prep.issue.text,
            "",
        ]
    all_path.write_text("\n".join(summary), encoding="utf-8")
    return all_path


def write_context_sizes(batch_dir: Path, preps: list[BugPrep]) -> Path:
    rows = [(prep.key, prep.role, prep.context) for prep in preps if prep.context is not None]
    path = batch_dir / "context_sizes.md"
    path.write_text(ctx.render_markdown(rows), encoding="utf-8")
    return path


def write_context_report(batch_dir: Path, preps: list[BugPrep]) -> Path:
    """``context_report.json`` — the machine-readable half of §1.6.

    [VANTA-WHY] Measurements land here and in ``prepare.json``, never in
    ``plans/bugsinpy_split.json``. The split is an *input* the user reviews and
    edits; a stage that rewrites it silently turns a measurement into a decision,
    and the difference between "the plan says this" and "this run observed that" is
    the whole point of keeping the two files apart.

    ``prompt_tokens_real`` is null until a session has actually paid for the prompt:
    the estimate is ``ceil(source_bytes / 4)`` plus a frame and runs high (measured
    ~1.5x on the heretic sessions), so a pilot set chosen from the estimate alone is
    uncalibrated. ``prompt_tokens_real_source`` names the session that filled it in.
    """
    path = batch_dir / "context_report.json"
    measured = [prep for prep in preps if prep.context is not None]
    windows = {prep.context.n_ctx for prep in measured}
    budgets = {prep.context.budget_tokens for prep in measured}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "context_report",
        "created_utc": utc_now(),
        "estimator": {
            "bytes_per_token": ctx.BYTES_PER_TOKEN_ESTIMATE,
            "prompt_frame_tokens": ctx.PROMPT_FRAME_TOKENS,
            "output_reserve_tokens": ctx.OUTPUT_RESERVE_TOKENS,
            "chat_overhead_tokens": ctx.CHAT_OVERHEAD_TOKENS,
            "margin_fraction": ctx.CONTEXT_MARGIN_FRACTION,
        },
        "n_ctx": next(iter(windows)) if len(windows) == 1 else None,
        "n_ctx_source": measured[0].context.n_ctx_source if len(windows) == 1 else None,
        "windows_measured": sorted(windows),
        "budget_tokens": next(iter(budgets)) if len(budgets) == 1 else None,
        "budgets_measured": sorted(budgets),
        "fallback_window": any(
            prep.context.n_ctx_source.startswith("fallback") for prep in measured
        ),
        "bugs": [_context_record(prep) for prep in preps],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _context_record(prep: BugPrep) -> dict[str, Any]:
    verdict = prep.context
    if verdict is None:
        return {
            "bug": prep.key,
            "wave": prep.wave,
            "role": prep.role,
            "prepare_mode": prep.prepare_mode,
            "verdict": "not_measured",
            "context_ok": None,
            "estimated_prompt_tokens": None,
            "prompt_tokens_real": None,
            "prompt_tokens_real_source": None,
            "budget_tokens": None,
            "headroom_tokens": None,
            "n_ctx": None,
            "n_ctx_source": None,
            "source_bytes": None,
            "file_count": None,
            "source_bytes_eligible": None,
            "files_eligible": None,
            "files_skipped_oversize": None,
            "files_skipped_budget": None,
            "reasons": list(prep.gate_failures),
        }
    stats = verdict.stats
    return {
        "bug": prep.key,
        "wave": prep.wave,
        "role": prep.role,
        "prepare_mode": prep.prepare_mode,
        "verdict": "ok" if verdict.context_ok else "reject",
        "context_ok": verdict.context_ok,
        "estimated_prompt_tokens": verdict.estimated_prompt_tokens,
        "prompt_tokens_real": None,
        "prompt_tokens_real_source": None,
        "budget_tokens": verdict.budget_tokens,
        "headroom_tokens": verdict.headroom_tokens,
        "n_ctx": verdict.n_ctx,
        "n_ctx_source": verdict.n_ctx_source,
        "source_bytes": stats.bytes_included,
        "file_count": stats.files_included,
        "source_bytes_eligible": stats.bytes_eligible,
        "files_eligible": stats.files_eligible,
        "files_skipped_oversize": stats.files_skipped_oversize,
        "files_skipped_budget": stats.files_skipped_budget,
        "reasons": verdict.reasons,
    }


def render_context_table(preps: list[BugPrep]) -> str:
    """The same numbers as ``context_report.json``, as a table on stdout."""
    head = (
        f"{'bug':<18} {'mode':<12} {'est tok':>8} {'real tok':>9} {'budget':>7} "
        f"{'headroom':>9} {'src B':>9} {'files':>6}  verdict"
    )
    lines = [head, "-" * len(head)]
    for prep in preps:
        verdict = prep.context
        if verdict is None:
            lines.append(
                f"{prep.key:<18} {prep.prepare_mode:<12} {'—':>8} {'—':>9} {'—':>7} "
                f"{'—':>9} {'—':>9} {'—':>6}  not measured"
            )
            continue
        lines.append(
            f"{prep.key:<18} {prep.prepare_mode:<12} {verdict.estimated_prompt_tokens:>8} "
            f"{'—':>9} {verdict.budget_tokens:>7} {verdict.headroom_tokens:>+9} "
            f"{verdict.stats.bytes_included:>9} {verdict.stats.files_included:>6}  "
            f"{'ok' if verdict.context_ok else 'REJECT'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel(prep: BugPrep, tree_name: str) -> str:
    """Tree path relative to the batch directory (which Shuru mounts at /workspace)."""
    return f"{prep.root.relative_to(_batch_root(prep)).as_posix()}/{tree_name}"


def _batch_root(prep: BugPrep) -> Path:
    # worktrees/<slug>/<tree> — the batch directory is two levels up from the bug root.
    return prep.root.parent.parent


def _read_report(batch_dir: Path, stem: str) -> dict:
    path = batch_dir / f"reports/{stem}.run1.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_bug(split: Split, bugsinpy_root: Path, project: str, bug_id: int) -> BugInfo:
    try:
        return load_bug_info(bugsinpy_root, project, bug_id)
    except MetadataError as exc:
        raise PrepareError(str(exc)) from exc


def require_checkpoint(name: str) -> None:
    if not bake_mod.layer_exists(name):
        raise PrepareError(
            f"checkpoint {name!r} does not exist. Bake it first (design §11 step 3 for "
            "py3.8, §3.2 for the older interpreters); prepare runs every suite inside "
            "Shuru and will not run one on the host."
        )


def require_checkpoints(names: list[str]) -> None:
    """Preflight the whole selected set, so a missing layer stops the batch up front.

    Checked before any checkout is made, not per bug: a missing checkpoint is a
    configuration error about the batch (the bake step has not been run for that
    wave), not a property of a bug, and discovering it after three full oracle
    derivations would waste the expensive part of ``prepare`` on work that cannot
    be completed.

    The check is ``bake.layer_exists`` — the same function ``bake`` decides
    skip-vs-rebuild with. Two opinions about whether a layer exists is how this
    stage came to report "does not exist" about a layer the bake had just named.
    """
    missing = sorted({name for name in names if not bake_mod.layer_exists(name)})
    if missing:
        raise PrepareError(
            "these checkpoints do not exist: "
            + ", ".join(missing)
            + ". Bake them first (design §11 step 3 for py3.8, §3.2 for the older "
            "interpreters). prepare runs every suite inside Shuru and will not run "
            "one on the host."
        )


# ---------------------------------------------------------------------------
# orchestration (§11 step 2)
# ---------------------------------------------------------------------------
def check_declared(entry: dict[str, Any], bug: BugInfo) -> list[str]:
    """The split is the protocol, so ``bug.info`` must still agree with it.

    A mismatch means either the split was edited by hand or BugsInPy's metadata
    moved under us. Both are fatal to reproducibility, and both are cheap to catch
    here rather than from a session whose ``FAIL_TO_PASS`` was derived from a
    different commit than the one the split claims.
    """
    mismatches: list[str] = []
    for key, actual in (
        ("python_version", bug.python_version),
        ("buggy_commit", bug.buggy_commit),
        ("fixed_commit", bug.fixed_commit),
        ("test_file", bug.test_file),
        ("pythonpath", bug.pythonpath),
    ):
        if entry.get(key) != actual:
            mismatches.append(
                f"split declares {key}={entry.get(key)!r}, BugsInPy metadata says {actual!r}"
            )
    if bug.pythonpath and not entry.get("needs_build"):
        mismatches.append(
            f"BugsInPy sets pythonpath={bug.pythonpath!r} but the split says needs_build=false"
        )
    return mismatches


def prepare_one(
    entry: dict[str, Any],
    *,
    split: Split,
    bugsinpy_root: Path,
    repo_cache: Path,
    batch_dir: Path,
    deps_dir: Path,
    full: bool,
) -> BugPrep:
    """Everything ``prepare`` does for one bug, in the order §1.3 fixes."""
    project = entry["project"]
    bug_id = int(entry["bug_id"])
    bug = load_bug(split, bugsinpy_root, project, bug_id)
    prep = BugPrep(
        bug=bug,
        root=batch_dir / "worktrees" / f"{project}_{bug_id}",
        role=entry.get("role", "pilot"),
        wave=int(entry["wave"]),
        test_cmd=[str(part) for part in entry["test_cmd"]],
        prepare_mode=FULL if full else MEASURE_ONLY,
    )

    mismatches = check_declared(entry, bug)
    if mismatches:
        prep.gate_failures.extend(mismatches)
        return prep

    prep.test_paths = build_trees(bug, repo_cache, prep.root)

    # Cheap gate first. The pre-strip tree is the same source as the final one
    # (the strip only removes test functions), so measuring it costs nothing and
    # saves the three double suite runs on a bug that cannot fit anyway — which is
    # the whole point of the precheck for a candidate like youtube-dl.
    early = ctx.precheck(prep.root / "oracle_tree", "")
    if not early.context_ok:
        prep.context = early
        return prep

    if not full:
        prep.context = early
        return prep

    checkpoint = entry["checkpoint"]
    # The dependency gate comes first, and comes before the derivation, because a
    # missing dependency must never be reported as an oracle failure (§3.3). It
    # needs no oracle work at all: the layer is baked, the trees are on disk.
    try:
        deps = bake_mod.load_deps(deps_dir, checkpoint)
        prep.deps = deps
    except bake_mod.BakeError as exc:
        prep.gate_failures.append(f"deps: {exc}")
        return prep

    try:
        collected = check_deps(
            prep, checkpoint=checkpoint, batch_dir=batch_dir, deps=deps
        )
    except suite_mod.SuiteError as exc:
        prep.gate_failures.append(str(exc))
        return prep

    try:
        derive_hidden(
            prep, checkpoint=checkpoint, batch_dir=batch_dir, collected=collected
        )
    except (suite_mod.SuiteError, tree_mod.TreeError) as exc:
        prep.gate_failures.append(f"hidden-oracle derivation failed: {exc}")
        return prep

    try:
        strip_and_baseline(prep, checkpoint=checkpoint, batch_dir=batch_dir)
    except suite_mod.SuiteError as exc:
        prep.gate_failures.append(f"post-strip suite run failed: {exc}")
        return prep

    build_issue(prep, repo_cache=repo_cache, batch_dir=batch_dir)

    # Authoritative measurement: the tree Vanta will actually be given, with the
    # issue text it will actually be given. Smaller than `early`, never larger.
    prep.context = ctx.precheck(
        prep.root / "vanta_tree", prep.issue.text if prep.issue else ""
    )
    return prep


def derive_verdict(prep: BugPrep) -> dict[str, Any]:
    """Belongs in ``split.DERIVED_BUG_KEYS`` — nothing here is hand-authored."""
    issue_text = prep.issue.text if prep.issue else None
    return {
        "prepare_mode": prep.prepare_mode,
        "bug_patch_sha256": sha256_text(prep.bug.bug_patch_txt),
        "test_paths_from_fixed": prep.test_paths,
        "fail_to_pass": sorted(prep.fail_to_pass),
        "pass_to_pass_count": prep.pass_to_pass_count,
        "flaky_nodeids": sorted(prep.flaky_nodeids),
        "preexisting_failures": sorted(prep.preexisting_failures),
        "native_failure_baseline": sorted(prep.native_failure_baseline),
        "native_green_on_vanta_tree": prep.native_green_on_vanta_tree,
        "hidden_files_deleted": prep.hidden_files_deleted,
        "partially_hidden_functions": prep.partially_hidden_functions,
        "gate_failures": prep.gate_failures,
        "prompt_tokens_estimate": prep.context.estimated_prompt_tokens
        if prep.context
        else None,
        "context_ok": bool(prep.context and prep.context.context_ok),
        "context_reasons": prep.context.reasons if prep.context else ["not measured"],
        "issue_text": issue_text,
        "issue_text_sha256": sha256_text(issue_text) if issue_text is not None else None,
        "issue_ok": bool(prep.issue and prep.issue.ok),
        "issue_needs_decision": (
            prep.issue.needs_decision
            if prep.issue
            else "issue text was not built (context-rejected or measurement-only)"
        ),
        "deps": prep.deps,
    }


def render_summary(preps: list[BugPrep]) -> str:
    """The ``prepare`` console + ``prepare_summary.md`` table."""
    lines = [
        "| bug | role | wave | mode | verdict | F2P | collected | native fails | context tok | issue |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for prep in preps:
        if prep.gate_failures:
            verdict = "**gate**"
        elif prep.prepare_mode != FULL:
            verdict = "measured"
        elif prep.context and not prep.context.context_ok:
            verdict = "**context**"
        elif prep.issue and not prep.issue.ok:
            verdict = "**issue**"
        else:
            verdict = "ok"
        collected = prep.pass_to_pass_count + len(prep.native_failure_baseline)
        lines.append(
            "| {key} | {role} | {wave} | {mode} | {verdict} | {f2p} | {collected} | {fails} | "
            "{tokens} | {issue} |".format(
                key=prep.key,
                role=prep.role,
                wave=prep.wave,
                mode=prep.prepare_mode,
                verdict=verdict,
                f2p=len(prep.fail_to_pass),
                collected=collected if prep.prepare_mode == FULL else "—",
                fails=len(prep.native_failure_baseline)
                if prep.prepare_mode == FULL
                else "—",
                tokens=prep.context.estimated_prompt_tokens if prep.context else "—",
                issue="ok"
                if prep.issue and prep.issue.ok
                else ("**decision**" if prep.issue else "—"),
            )
        )
    lines.append("")
    for prep in preps:
        for failure in prep.gate_failures:
            lines.append(f"- **{prep.key}**: {failure}")
    lines.append("")
    return "\n".join(lines)
