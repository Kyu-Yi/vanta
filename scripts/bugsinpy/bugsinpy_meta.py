"""BugsInPy metadata: the pinned clone, ``bug.info`` and ``run_test.sh``.

[VANTA-WHY] The harness never calls ``bugsinpy-checkout``. That script builds its
own venv, writes into paths we do not control, and assumes it owns the working
tree; we need several checkouts of the same project side by side (buggy, buggy +
fixed tests, fixed) and we do the checkouts with plain ``git``. What we DO take
from BugsInPy is its metadata, which is the part with the ground truth in it.

``bug.info`` is ``key="value"`` per line, and at least one entry in the public
corpus (``youtube-dl/1``) writes ``fixed_commit_id ="..."`` with whitespace before
the ``=``. A parser that split on ``"`` or used a regex anchored on ``=`` would
silently return the wrong commit for that bug and check out the wrong tree, so the
parser splits on the FIRST ``=`` and strips both sides.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy"


class MetadataError(RuntimeError):
    """BugsInPy metadata is missing or malformed."""


@dataclass(frozen=True)
class BugInfo:
    """One ``projects/<p>/bug.info``, plus the bug-dir companions."""

    project: str
    bug_id: int
    python_version: str
    buggy_commit: str
    fixed_commit: str
    test_file: str
    pythonpath: Optional[str]
    requirements_txt: str
    run_test_sh: str
    bug_patch_txt: str
    bug_dir: Path

    @property
    def key(self) -> str:
        return f"{self.project}/{self.bug_id}"

    @property
    def test_file_abs(self) -> Path:
        return self.bug_dir / self.test_file

    def run_test_command(self) -> Optional[str]:
        """The single command line ``run_test.sh`` invokes, if it is a one-liner.

        Used only as a cross-check on which nodeid is the hidden test (design
        §1.3): BugsInPy's older projects shell out to ``python -m unittest`` while
        Vanta's harness parses pytest's JSON report, so this string is evidence,
        not the executor.
        """
        for raw in self.run_test_sh.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            return line
        return None


def parse_bug_info(text: str) -> dict[str, str]:
    """Parse ``key="value"`` lines, tolerating whitespace around ``=``."""
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key] = value
    return fields


def load_bug_info(bugsinpy_root: Path, project: str, bug_id: int) -> BugInfo:
    bug_dir = bugsinpy_root / "projects" / project / "bugs" / str(bug_id)
    if not bug_dir.is_dir():
        raise MetadataError(f"no such bug directory: {bug_dir}")

    info_path = bug_dir / "bug.info"
    if not info_path.is_file():
        raise MetadataError(f"missing bug.info: {info_path}")
    fields = parse_bug_info(info_path.read_text(encoding="utf-8", errors="replace"))

    try:
        return BugInfo(
            project=project,
            bug_id=bug_id,
            python_version=_required(fields, "python_version", info_path),
            buggy_commit=_required(fields, "buggy_commit_id", info_path),
            fixed_commit=_required(fields, "fixed_commit_id", info_path),
            test_file=_required(fields, "test_file", info_path),
            pythonpath=fields.get("pythonpath") or None,
            requirements_txt=_read_optional(bug_dir / "requirements.txt"),
            run_test_sh=_read_optional(bug_dir / "run_test.sh"),
            bug_patch_txt=_read_optional(bug_dir / "bug_patch.txt"),
            bug_dir=bug_dir,
        )
    except MetadataError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MetadataError(f"{info_path}: {exc}") from exc


def project_github_url(bugsinpy_root: Path, project: str) -> Optional[str]:
    info_path = bugsinpy_root / "projects" / project / "project.info"
    if not info_path.is_file():
        return None
    return parse_bug_info(info_path.read_text(encoding="utf-8", errors="replace")).get(
        "github_url"
    )


def ensure_clone(root: Path, *, commit: Optional[str] = None, url: str = BUGSINPY_URL) -> str:
    """Clone (or reuse) BugsInPy and return the checked-out commit SHA.

    [VANTA-WHY] The SHA is returned so ``prepare`` can freeze it into the prepared
    manifest. BugsInPy's metadata is mutable upstream: a bug's ``bug_patch.txt``
    can be corrected, and a split that recorded only "BugsInPy" would describe an
    experiment nobody can reproduce.
    """
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", url, str(root)], cwd=None)
    if commit:
        _git(["checkout", "--quiet", commit], cwd=root)
    return _git(["rev-parse", "HEAD"], cwd=root).strip()


def _git(args: list[str], *, cwd: Optional[Path]) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise MetadataError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _required(fields: dict[str, str], key: str, path: Path) -> str:
    value = fields.get(key)
    if not value:
        raise MetadataError(f"{path}: missing required key {key!r}")
    return value


def _read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
