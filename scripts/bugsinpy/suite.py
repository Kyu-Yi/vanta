"""Run a project's suite inside Shuru and read pytest's JSON report back.

[VANTA-WHY] Every suite run in this harness happens in the microVM, including
``prepare``'s — BugsInPy source is third-party code and pytest imports it. C3 has
no exception for "just measuring" (design C3, §11 note).

[VANTA-WHY] Every suite is run **twice** (§1.3 step 3). A nodeid whose outcome
differs between the two runs is flagged flaky and excluded from both
``FAIL_TO_PASS`` and ``PASS_TO_PASS``. Without the second run a flaky test can be
selected as the oracle, at which point the whole bug's ground truth is noise — and
the noise is invisible, because a flaky failure looks exactly like a real one.

[VANTA-WHY] ``PYTHONDONTWRITEBYTECODE=1`` and ``-p no:cacheprovider``: the tree is
mounted read-write from the host, so ``__pycache__`` and ``.pytest_cache`` created
inside the VM land in the host tree, where they would be picked up by Vanta's
source walk (``__pycache__`` is skipped, but ``.pytest_cache`` is not) and change
what the Creator is shown.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from . import shuru

FAILURE_OUTCOMES = frozenset({"failed", "error"})


class SuiteError(RuntimeError):
    """The suite could not be run or its report could not be read."""


@dataclass
class SuiteRun:
    """One execution of the whole suite."""

    outcomes: dict[str, str] = field(default_factory=dict)
    exit_code: Optional[int] = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    report_found: bool = False

    @property
    def failures(self) -> set[str]:
        return {n for n, o in self.outcomes.items() if o in FAILURE_OUTCOMES}

    @property
    def collected(self) -> set[str]:
        return set(self.outcomes)


@dataclass
class SuitePair:
    """The two runs, plus what differs between them."""

    first: SuiteRun
    second: SuiteRun
    flaky_nodeids: set[str] = field(default_factory=set)
    missing_from_one_run: set[str] = field(default_factory=set)

    @property
    def stable_failures(self) -> set[str]:
        """Failures in both runs — the set a gate or a baseline should use."""
        return (self.first.failures & self.second.failures) - self.flaky_nodeids

    @property
    def stable_outcomes(self) -> dict[str, str]:
        return {
            n: o
            for n, o in self.first.outcomes.items()
            if n not in self.flaky_nodeids
            and n not in self.missing_from_one_run
            and self.second.outcomes.get(n) == o
        }

    @property
    def stable_collected(self) -> set[str]:
        return set(self.stable_outcomes)


def compare_runs(first: SuiteRun, second: SuiteRun) -> SuitePair:
    """Diff two runs of the same suite: flaky, missing, and demonstrably stable.

    Split out of ``run_suite_twice`` so the rule can be exercised without booting
    a VM. This rule is what stops a flaky test from becoming the oracle (§1.3 step
    3) — the one failure that would make every derived label for a bug noise — and
    a version of it that could only be tested by running the whole corpus would be
    a silent single point of failure.
    """
    pair = SuitePair(first=first, second=second)
    for nodeid in set(first.outcomes) | set(second.outcomes):
        a = first.outcomes.get(nodeid)
        b = second.outcomes.get(nodeid)
        if a is None or b is None:
            pair.missing_from_one_run.add(nodeid)
        elif a != b:
            pair.flaky_nodeids.add(nodeid)
    return pair


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def build_command(test_cmd: Sequence[str], report_guest_path: str) -> list[str]:
    substituted = [str(part).replace("{report}", report_guest_path) for part in test_cmd]
    if not any(report_guest_path in part for part in substituted):
        raise SuiteError(
            "test_cmd has no {report} placeholder; the JSON report would then be "
            "written to a path the harness cannot read back"
        )
    return substituted


def _shell_quote(parts: Sequence[str]) -> str:
    out = []
    for part in parts:
        if all(c.isalnum() or c in "-_./=:{}" for c in part):
            out.append(part)
        else:
            out.append("'" + part.replace("'", "'\\''") + "'")
    return " ".join(out)


def run_suite_twice(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    report_rel_stem: str,
    disk_mb: int,
    timeout_secs: int = 1800,
    extra_env: Optional[dict[str, str]] = None,
) -> SuitePair:
    """Run ``test_cmd`` twice inside the checkpoint and diff the two reports."""
    reports_dir = batch_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rel1 = f"{report_rel_stem}.run1.json"
    rel2 = f"{report_rel_stem}.run2.json"
    for rel in (rel1, rel2):
        (batch_dir / rel).unlink(missing_ok=True)

    cmd1 = _shell_quote(build_command(test_cmd, f"/workspace/{rel1}"))
    cmd2 = _shell_quote(build_command(test_cmd, f"/workspace/{rel2}"))

    script = "\n".join(
        [
            "set +e",
            cmd1,
            "echo __VANTA_EXIT_1__=$?",
            cmd2,
            "echo __VANTA_EXIT_2__=$?",
            "exit 0",
        ]
    )

    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)

    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["sh", "-c", script],
        disk_mb=disk_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env=env,
    )

    first = _load_run(batch_dir / rel1, result, suffix="1")
    second = _load_run(batch_dir / rel2, result, suffix="2")
    pair = compare_runs(first, second)

    _clean_tree_noise(batch_dir / tree_rel)
    return pair


def collect_command(test_cmd: Sequence[str]) -> list[str]:
    """``test_cmd`` reduced to a collection-only run — the collectability gate (§3.4).

    Split out of ``collect_only`` because the bake needs the *same* command: the
    per-bug layer's collect check must not be a second interpretation of
    ``test_cmd`` that can drift from the one prepare runs.
    """
    return [
        str(part) for part in test_cmd if not str(part).startswith("--json-report")
    ] + ["--collect-only", "-q", "-p", "no:cacheprovider"]


def parse_collected(stdout: str) -> set[str]:
    """``-q`` collection output → nodeids. One nodeid per line, no spaces."""
    nodeids: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if "::" in line and " " not in line:
            nodeids.add(line)
    return nodeids


def collected_or_raise(
    nodeids: set[str], *, exit_code: Optional[int], output: str
) -> set[str]:
    """A collectable suite: pytest exited cleanly and there is something to grade.

    ``exit_code`` is part of the check, not decoration. Measured on PySnooper/3: a
    test module that raises on import makes pytest collect 0 tests and exit 2 — and
    the same import failure inside *one* of several test files leaves a non-empty
    collection set. Without the exit-status check that gate would pass while the
    file holding the hidden test was never collected, and the bug would be graded
    against a suite it does not have (§3.4: "no collection errors").
    """
    if not nodeids:
        raise SuiteError(
            f"pytest collected nothing (exit {exit_code}); tail: "
            f"{output[-800:]!r}"
        )
    if exit_code != 0:
        raise SuiteError(
            f"pytest exited {exit_code} while collecting ({len(nodeids)} test(s) "
            f"collected): a collection error means the suite is not what the oracle "
            f"would grade; tail: {output[-800:]!r}"
        )
    return nodeids


def guest_tree_root(tree_rel: str) -> str:
    """Where a tree appears inside the VM.

    One definition, so a probe can never disagree with the suites about which
    directory the suite ran in.
    """
    return f"/workspace/{tree_rel}"


def collect_only(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    disk_mb: int,
    timeout_secs: int = 600,
) -> set[str]:
    """``pytest --collect-only -q`` — the collectability gate (§3.4).

    Normalising every bug to pytest is what lets one report parser cover a
    corpus where half the projects ship ``unittest`` suites; pytest collects
    ``unittest.TestCase`` subclasses, ``unittest``'s own runner does not produce
    a JSON report we can read. If a bug does not collect here it is rejected from
    the split rather than run and failed.
    """
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["sh", "-c", _shell_quote(collect_command(test_cmd))],
        disk_mb=disk_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    nodeids = collected_or_raise(
        parse_collected(result.stdout),
        exit_code=result.exit_code,
        output=f"{result.stdout} {result.stderr}",
    )
    _clean_tree_noise(batch_dir / tree_rel)
    return nodeids


def import_probe(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    test_cmd: Sequence[str],
    module: str,
    disk_mb: int,
    timeout_secs: int = 300,
) -> str:
    """``import <module>; print(<module>.__file__)``, run exactly as the suites run.

    [VANTA-WHY] The suites run ``python -m pytest`` with their cwd set to the tree
    root and no ``PYTHONPATH``, so the tree reaches ``sys.path`` through ``python
    -m``'s CWD prepend and nothing else. This probe reuses that invocation — same
    interpreter (``test_cmd[0]``), same cwd, same env, no ``PYTHONPATH``. A probe
    that set ``PYTHONPATH`` would prove an import path the suites do not have, and
    one that ran the module the suite does not import could pass while the suite
    still died on ``ModuleNotFoundError``. The version installed in site-packages
    is exactly what this must catch: a buggy copy there would be imported instead
    of the graded tree, and every patch would then grade the same.

    Returns the path the interpreter actually loaded.
    """
    if not module.isidentifier():
        raise SuiteError(
            f"import_probe was given {module!r}, which is not a Python identifier; "
            "the module name is read from the project's own setup.py"
        )
    python = str(test_cmd[0]) if test_cmd else "python"
    code = f"import {module} as _vanta_probe; print(_vanta_probe.__file__)"
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["sh", "-c", _shell_quote([python, "-c", code])],
        disk_mb=disk_mb,
        guest_cwd=guest_tree_root(tree_rel),
        timeout_secs=timeout_secs,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
    )
    where = next(
        (line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()),
        "",
    )
    if not result.ok or not where:
        raise SuiteError(
            f"`{python} -c` importing {module} in {guest_tree_root(tree_rel)} exited "
            f"{result.exit_code} without printing a path: "
            f"{(result.stderr + result.stdout).strip()[-800:]!r}"
        )
    return where


# ---------------------------------------------------------------------------
# report parsing
# ---------------------------------------------------------------------------
def parse_report(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"cannot read pytest report {path}: {exc}") from exc
    outcomes: dict[str, str] = {}
    for test in data.get("tests") or []:
        nodeid = test.get("nodeid")
        outcome = test.get("outcome")
        if isinstance(nodeid, str) and isinstance(outcome, str):
            outcomes[nodeid] = outcome
    return outcomes


def _load_run(report: Path, result: shuru.ExecResult, *, suffix: str) -> SuiteRun:
    run = SuiteRun(
        exit_code=_exit_code(result.stdout, suffix),
        timed_out=result.timed_out,
        stdout_tail=result.stdout[-4000:],
        stderr_tail=result.stderr[-2000:],
    )
    if report.is_file():
        run.outcomes = parse_report(report)
        run.report_found = True
    return run


def _exit_code(stdout: str, suffix: str) -> Optional[int]:
    marker = f"__VANTA_EXIT_{suffix}__="
    for line in stdout.splitlines():
        if line.startswith(marker):
            try:
                return int(line[len(marker) :].strip())
            except ValueError:
                return None
    return None


def _clean_tree_noise(tree: Path) -> None:
    for name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
        shutil.rmtree(tree / name, ignore_errors=True)


# ---------------------------------------------------------------------------
# diffing two failure sets (used by the fixed-tree gate and by the oracle)
# ---------------------------------------------------------------------------
def collect_by_file(nodeids: Sequence[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for nodeid in nodeids:
        out.setdefault(nodeid.split("::", 1)[0], set()).add(nodeid)
    return out


def git_apply_patch(
    *,
    checkpoint: str,
    batch_dir: Path,
    tree_rel: str,
    patch_path: Path,
    disk_mb: int,
    drop_test_hunks: bool,
    timeout_secs: int = 300,
) -> tuple[bool, str]:
    """Apply a patch inside the VM. Strict — no fuzzy fallback (§5.2).

    ``drop_test_hunks`` filters the diff by path before applying, so a Creator
    patch that edits a test file cannot reach the hidden test. Filtering is done
    with ``filterdiff``-free code (a small awk-free split in Python on the host)
    because the patch body is a host-side artifact; only ``git apply`` runs in the
    VM.
    """
    body = patch_path.read_text(encoding="utf-8")
    if drop_test_hunks:
        body = drop_test_file_hunks(body)
        if not body.strip():
            return True, "patch contained only test hunks; nothing to apply"
    staged = batch_dir / "reports" / f"_apply_{patch_path.name}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(body, encoding="utf-8")
    result = shuru.run(
        checkpoint=checkpoint,
        host_root=batch_dir,
        command=["git", "apply", "--verbose", f"/workspace/{staged.relative_to(batch_dir)}"],
        disk_mb=disk_mb,
        guest_cwd=f"/workspace/{tree_rel}",
        timeout_secs=timeout_secs,
    )
    ok = result.exit_code == 0
    detail = (result.stdout + result.stderr).strip()[-2000:]
    return ok, detail


def drop_test_file_hunks(diff_text: str) -> str:
    """Remove whole ``diff --git`` sections whose target is a test file."""
    from .trees import is_test_path

    kept: list[str] = []
    current: list[str] = []
    current_is_test = False

    def flush() -> None:
        if current and not current_is_test:
            kept.extend(current)

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            current = [line]
            current_is_test = False
            parts = line.split()
            if len(parts) >= 4:
                target = parts[3]
                target = target[2:] if target.startswith("b/") else target
                current_is_test = is_test_path(target)
            continue
        current.append(line)
    flush()
    return "".join(kept)
