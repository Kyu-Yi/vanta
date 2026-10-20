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

[VANTA-WHY] Every one of those files goes through :func:`read_metadata_text`, and
the encoding rule is not decoration. ``black/22``'s ``requirements.txt`` is UTF-16
with a byte-order mark, and reading it as UTF-8 with ``errors="replace"`` turned
its first line into ``'\ufffd\ufffda\x00c\x00c\x00e\x00s\x00s\x00...'`` — a string
that reached the requirement parser and failed there, as
``Expected package name at the start of dependency specifier``. That message names
the parser, not the file, so the real cause (an encoding the reader never
considered) took a detour through the wrong half of the pipeline to be found. The
reader therefore decodes what it can identify and **fails loudly on what it
cannot**: a wrong guess at an encoding corrupts a requirement silently, which is
exactly the failure a fix here is supposed to prevent rather than relocate.
"""

from __future__ import annotations

import codecs
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy"

BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    # After the utf-32 pair: BOM_UTF32_LE starts with BOM_UTF16_LE, so a UTF-32 file
    # tested against the utf-16 mark first would decode as utf-16 and come back with
    # a NUL between every character.
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)
"""Every byte-order mark this reader recognises, longest first."""


class MetadataError(RuntimeError):
    """BugsInPy metadata is missing or malformed."""


def decode_metadata(data: bytes, *, path: Path) -> str:
    """BugsInPy metadata bytes → text, or a clear error naming the problem.

    A mark is decoded by the codec it names. Without one the file must be UTF-8:
    anything else — a stray byte, or NULs from a UTF-16 file that lost its mark —
    is an error, because the alternative is a string that looks like text and is
    not, and the ``errors="replace"`` that produced such a string is what let a
    mojibake requirement travel as far as the requirement parser.
    """
    for bom, encoding in BOMS:
        if data.startswith(bom):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError as exc:
                raise MetadataError(
                    f"{path}: carries a {encoding} byte-order mark but does not decode "
                    f"as {encoding} (byte {exc.start} is 0x{data[exc.start]:02x}, first "
                    f"bytes {data[:16].hex(' ')})"
                ) from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetadataError(
            f"{path}: is not UTF-8 (byte {exc.start} is 0x{data[exc.start]:02x}, first "
            f"bytes {data[:16].hex(' ')}). BugsInPy's corpus really does contain "
            "non-UTF-8 files (black/22's requirements.txt is UTF-16), so this is read "
            "strictly rather than approximated: re-export the file as UTF-8, or with "
            "its mark, and re-run."
        ) from exc
    if "\x00" in text:
        raise MetadataError(
            f"{path}: decodes as UTF-8 but contains NUL bytes (first bytes "
            f"{data[:16].hex(' ')}), which is what UTF-16 without a byte-order mark "
            "looks like. This reader does not guess an encoding — a wrong guess "
            "corrupts a requirement without saying so — so the file needs its mark or "
            "a UTF-8 export."
        )
    return text


def read_metadata_text(path: Path) -> str:
    """One BugsInPy metadata file → text. The only reader the harness uses for them.

    Named for what it reads rather than for what it does, so a new caller reads as
    "this file comes from BugsInPy" at the call site. ``bug.info``, ``project.info``,
    ``requirements.txt``, ``setup.sh``, ``run_test.sh`` and ``bug_patch.txt`` all go
    through it — the last three because a project's own dependency lines and the
    issue text are exactly what a mojibake decode would quietly corrupt.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MetadataError(f"cannot read {path}: {exc}") from exc
    return decode_metadata(data, path=path)


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
    fields = parse_bug_info(read_metadata_text(info_path))

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
    return parse_bug_info(read_metadata_text(info_path)).get("github_url")


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
    return read_metadata_text(path)
