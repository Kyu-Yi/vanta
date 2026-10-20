"""The batch runner — one job at a time, resumable (design §7.1, §7.4, §11 step 6).

One *job* is one `vanta_m3` invocation plus its oracle grading and its manifest
row. A job is identified by ``(batch, bug, config, mode, seed, candidate)`` — the
last component is the candidate patch's ``patch_sha256`` for a review job and
empty for a repair job. That identity is the resume cursor (``state.json``), and
it is **content-based on purpose**: the pooled failed patch a review job grades
comes from a repair session whose id is a fresh random UUID on every attempt, so
keying on the session would make the same candidate look like a new one after a
resume and re-run it — and a counter would be worse still, because it would
conflate two different patches that happen to be the 3rd one produced.

[VANTA-WHY] Three rules from the design are load-bearing here, and each is the
kind that fails silently:

* **Order** (§7.1, D13). Within a wave: baseline repair → vanta repair → baseline
  review → vanta review. Repairs first, because a review candidate *is* a failed
  Creator patch and none can exist before the repairs ran. Baseline review second-
  to-last because it makes no LLM call at all (``VANTA_ATTACKER=off``), so it is
  the free plumbing check: it proves the candidate set, the strict ``--patch``
  apply, the oracle and the manifest row before a token is spent on the vanta
  review. And both configs review **exactly the same set at the same seed**, so
  every review row is a paired comparison whose only differing column is the
  config.
* **The review row's oracle is not re-run.** A review job grades the same bytes
  its candidate was graded with, so a second grading would repeat an experiment,
  not add evidence. The row copies ``candidate.oracle`` (§5.4, §7.3) — but only
  when the session actually produced a patch. ``PatchApplicationFailed`` and
  ``HarnessFailure`` leave no patch, and recording the *candidate's* good label
  against them would fabricate a ``true_reject``/``false_reject`` out of a run
  that told us nothing. Those rows are censored (``apply_ok = false`` →
  ``unobservable``, §5.2 step 2).
* **The mount rule forces the working directory.** ``shuru run --mount`` requires
  the host path to be inside the process CWD, and both Vanta's own backend and
  ``bugsinpy.shuru`` enforce it. So the runner refuses to start unless its CWD is
  the repo root and every batch directory is under ``runs/bugsinpy/``.

[VANTA-WHY] ``VANTA_SHURU_AUTO_DEPS=0`` is set, not merely left at its default.
Left on, Vanta's dependency self-heal would see the project's imports missing
from *its* checkpoint and build a derived one — which needs network, which a
per-run VM does not have. The per-bug checkpoint already has those deps (that is
what ``bake`` is for), so the self-heal can only turn a working environment into
an ``InfraFailure``.

Session directories are **not** moved into the batch. They stay in
``vanta_sessions/`` because ``--rerun-signals`` and ``metrics.py`` both address
them there, and because §5.4's discriminating check needs to re-run a review row's
signals afterwards. The manifest records the path instead of owning the files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import manifest as manifest_mod
from . import oracle as oracle_mod
from . import prepare as prepare_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .manifest import DONE, FAILED, RUNNING, Manifest, job_key
from .shuru import ShuruError
from .split import SAMPLING_POLICY, Split, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs" / "bugsinpy"
SESSIONS_DIRNAME = "vanta_sessions"
DEFAULT_VANTA_BIN = REPO_ROOT / "target" / "release" / "vanta_m3"

# §8.1's worst single run is ~30 min for a vanta repair; 3 h is a backstop, not an
# estimate. A run that hits it is a failure to report, not a result to grade.
RUN_TIMEOUT_SECS = 3 * 60 * 60
PROBE_TIMEOUT_SECS = 120
ORACLE_TIMEOUT_SECS = 1800

REPAIR = "repair"
REVIEW = "review"

DEV_FIX = "dev_fix"
NOOP = "noop"
FAILED_PATCH = "failed_patch"

# The most failed patches one bug may contribute to its review set (§7.1).
MAX_POOLED_CANDIDATES = 2

SESSION_LINE = re.compile(r"▌ Session:\s*(\S+)")
SUMMARY_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

PHASES: tuple[tuple[str, str], ...] = ((REPAIR, "baseline"), (REPAIR, "vanta"),
                                      (REVIEW, "baseline"), (REVIEW, "vanta"))
"""D13's order, spelled out. The config column is the split's own vocabulary; the
two are pinned together by :func:`config_switches`."""


class RunnerError(RuntimeError):
    """The runner cannot proceed — a broken preflight, not a bad run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def require_repo_cwd() -> Path:
    """Refuse to run from anywhere but the repo root (Shuru's mount rule)."""
    cwd = Path.cwd().resolve()
    if cwd != REPO_ROOT:
        raise RunnerError(
            f"the runner must be invoked with CWD = {REPO_ROOT} (got {cwd}): every "
            "tree lives under runs/bugsinpy/ and shuru may only mount paths inside "
            "the process CWD"
        )
    return cwd


def require_batch_dir(batch_dir: Path) -> Path:
    """Every batch dir must be inside the repo (same rule as :func:`require_repo_cwd`)."""
    resolved = batch_dir.resolve()
    cwd = require_repo_cwd()
    if cwd not in resolved.parents:
        raise RunnerError(
            f"batch directory {resolved} is not inside {cwd}; trees under it could "
            "not be mounted into a Shuru VM"
        )
    return resolved


def require_vanta_bin(path: Path) -> Path:
    if not path.is_file():
        raise RunnerError(
            f"no vanta_m3 binary at {path}. Build it first:\n"
            "    cargo build --release --bin vanta_m3"
        )
    return path


# ---------------------------------------------------------------------------
# config switches (§4.2) — one definition, shared with the record
# ---------------------------------------------------------------------------
def config_switches(config: str) -> tuple[bool, bool]:
    """``(attacker_enabled, planner_enabled)`` for a split config name.

    This is the inverse of ``benchmark_config_name`` in ``driver.rs``: the record
    the pipeline writes into ``run.summary`` names a config, and the manifest
    copies it. Deriving the *env* from the same switch pair is what keeps the two
    from being able to disagree.
    """
    if config == "baseline":
        return False, False
    if config == "vanta":
        return True, False
    if config == "vanta_planner":
        return True, True
    raise RunnerError(f"unknown config {config!r}; expected baseline, vanta, vanta_planner")


def vanta_env(
    *,
    config: str,
    seed: int,
    checkpoint: str,
    disk_mb: int,
    base: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """The controlled environment for one ``vanta_m3`` invocation (§7.4).

    Everything else — ``VANTA_LLM_BASE_URL``, ``VANTA_LLM_MODEL``, ``VANTA_API_KEY``
    — is inherited from the caller: the harness must not invent a model, and a
    copied value in this file would be one more thing that can drift from what
    actually ran.
    """
    env = dict(os.environ if base is None else base)
    attacker, planner = config_switches(config)
    env["VANTA_SANDBOX_BACKEND"] = "shuru"
    env["VANTA_SHURU_CHECKPOINT"] = checkpoint
    env["VANTA_SHURU_DISK_MB"] = str(disk_mb)
    # See the module docstring: the self-heal needs network the VM does not have.
    env["VANTA_SHURU_AUTO_DEPS"] = "0"
    env["VANTA_ATTACKER"] = "on" if attacker else "off"
    env["VANTA_PLANNER"] = "on" if planner else "off"
    # Always set, including for seed 0: `VANTA_LLM_SEED=0` is a request to pin 0,
    # while an unset variable means "no seed key in the body" (§4.4, D4).
    env["VANTA_LLM_SEED"] = str(seed)
    return env


# ---------------------------------------------------------------------------
# the test command handed to Vanta
# ---------------------------------------------------------------------------
def vanta_test_cmd(test_cmd: Sequence[str]) -> str:
    """The split's ``test_cmd`` as a shell string, minus Vanta's own report flags.

    Two facts collide here. ``TestCommand::from_shell`` splits on whitespace with
    **no quoting**, so this string must round-trip exactly — therefore any element
    containing whitespace is an error rather than a silent split. And Vanta
    appends its own ``--json-report --json-report-file=report.json -q``
    (``harness::build_test_cmd``), so passing the split's copies would make pytest
    receive two report files and the *last* one would win: the harness would then
    read a report Vanta never wrote.
    """
    kept: list[str] = []
    for part in test_cmd:
        token = str(part)
        if token.startswith("--json-report") or "{report}" in token:
            continue
        if any(ch.isspace() for ch in token):
            raise RunnerError(
                f"test_cmd element {token!r} contains whitespace; Vanta's "
                "--test-cmd is whitespace-split with no quoting, so this command "
                "cannot be passed through faithfully"
            )
        kept.append(token)
    if not kept:
        raise RunnerError(f"test_cmd {list(test_cmd)!r} is empty after dropping the report flags")
    return " ".join(kept)


# ---------------------------------------------------------------------------
# patches and candidates
# ---------------------------------------------------------------------------
def normalized_diff(text: str) -> str:
    """The dedup key's canonical form (§7.1).

    Two patches that differ only in an ``index`` line, in the ``a/``/``b/``
    prefixes or in trailing whitespace are the **same candidate**: they produce
    the same tree. Hashing the raw bytes would run the same experiment twice and
    count it twice.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("index "):
            continue
        if line.startswith("diff --git "):
            line = " ".join(
                token[2:] if token.startswith(("a/", "b/")) else token
                for token in line.split()
            )
        elif line.startswith("--- a/"):
            line = "--- " + line[6:]
        elif line.startswith("+++ b/"):
            line = "+++ " + line[6:]
        out.append(line.rstrip())
    return "\n".join(out)


def test_hunk_free(patch_text: str, *, what: str) -> str:
    """Drop test-file sections, then refuse an empty result (§5.4).

    Every candidate is applied with its test hunks dropped, so a candidate cannot
    smuggle a change into the test file — and so the *developer fix*, whose test
    hunks were computed against the buggy tests, still applies to ``vanta_tree``,
    which carries the fixed tests. An empty result means the patch consisted
    entirely of test changes: there is nothing to review, and grading it would
    grade a no-op with a real patch's name.
    """
    dropped = suite_mod.drop_test_file_hunks(patch_text)
    if not dropped.strip():
        raise RunnerError(
            f"{what}: dropping test-file hunks left nothing to apply — the patch "
            "changes only test files, so it cannot be a review candidate"
        )
    return dropped


def _first_source_file(root: Path) -> Optional[tuple[str, Path]]:
    """The lexicographically smallest non-test ``.py`` file — deterministic on purpose."""
    skip = {".git", "__pycache__", ".venv", "venv", "env", ".pytest_cache"}
    best: Optional[tuple[str, Path]] = None
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(part in skip for part in path.parts):
            continue
        if tree_mod.is_test_path(rel):
            continue
        if best is None or rel < best[0]:
            best = (rel, path)
    return best


def noop_patch_text(vanta_tree: Path) -> tuple[str, str]:
    """The whitespace-only floor case, as ``(relative path, diff text)``.

    [VANTA-WHY] A truly empty patch cannot be the floor. ``apply_patch_strict``
    returns "the patch is a no-op" when nothing changed, so an empty file would
    produce ``PatchApplicationFailed`` and the baseline review would measure
    *application*, not approval. The floor has to *apply cleanly and change
    nothing that can matter*, which is what appending one newline at EOF does.
    The file is chosen by sorted relative path so the same bug yields the same
    bytes on every run: the baseline and the vanta review must see one candidate,
    not two that differ by a coin flip.
    """
    chosen = _first_source_file(vanta_tree)
    if chosen is None:
        raise RunnerError(f"no non-test .py file under {vanta_tree}; cannot build the no-op floor")
    rel, path = chosen
    original = path.read_text(encoding="utf-8", errors="surrogateescape")
    patched = original + "\n"
    return rel, git_diff_no_index(original, patched, rel)


def git_diff_no_index(original: str, patched: str, rel: str) -> str:
    """A unified diff in exactly the shape Vanta's own ``write_final_patch`` emits.

    The trick it uses is repeated here on purpose: write the two contents under
    two scratch directories literally named ``a`` and ``b`` at the same relative
    path, then ``git diff --no-index --no-prefix -- a b``. ``--no-prefix`` plus
    dirs named ``a``/``b`` yields the standard ``a/<path>`` … ``b/<path>`` headers
    that ``git apply -p1`` expects, without a hand-written hunk header that could
    miscount (and be silently accepted by ``--recount``).
    """
    scratch = Path(_tempdir(prefix="vanta-noop-"))
    try:
        for name, text in (("a", original), ("b", patched)):
            target = scratch / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", errors="surrogateescape")
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--no-color", "--no-prefix", "--", "a", "b"],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        # `git diff --no-index` exits 1 whenever the trees differ — the normal case.
        if proc.returncode not in (0, 1):
            raise RunnerError(f"git diff --no-index failed: {proc.stderr.strip()[:400]}")
        return proc.stdout
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _tempdir(prefix: str) -> str:
    import tempfile

    return tempfile.mkdtemp(prefix=prefix)


@dataclass
class Candidate:
    """One patch under review, with the label it was graded with (§7.3)."""

    kind: str
    source: str
    patch_path: Path
    patch_sha256: str
    normalized_sha256: str
    oracle: oracle_mod.OracleResult

    @property
    def name(self) -> str:
        """Stable stem for filenames and for the job key: content, never a counter."""
        return f"{self.kind}-{self.patch_sha256[:12]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "patch_sha256": self.patch_sha256,
            "normalized_sha256": self.normalized_sha256,
            "patch_path": str(self.patch_path),
            "oracle": self.oracle.as_dict(),
        }


def write_candidate_patch(batch_dir: Path, prep: prepare_mod.BugPrep, kind: str, text: str,
                          *, source: str) -> Candidate:
    """Persist a test-hunk-free candidate and return it (un-graded)."""
    sha = sha256_text(text)
    dest = batch_dir / "candidates" / prep.slug / f"{kind}-{sha[:12]}.patch"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return Candidate(
        kind=kind,
        source=source,
        patch_path=dest,
        patch_sha256=sha,
        normalized_sha256=sha256_text(normalized_diff(text)),
        # Filled in by `grade_candidate`; a Candidate never travels un-graded.
        oracle=oracle_mod.OracleResult(apply_ok=False),
    )


def grade_candidate(
    *,
    batch_dir: Path,
    prep: prepare_mod.BugPrep,
    candidate: Candidate,
    checkpoint: str,
    disk_mb: int,
) -> Candidate:
    """Grade ``candidate`` once, caching the result keyed by its own identity.

    The cache is what makes a resume cheap: re-deriving a candidate set after a
    crash must not re-grade a patch the batch already graded, or a 3-hour batch
    would spend its first hour repeating known-good evidence.
    """
    cached = batch_dir / "oracle" / "candidates" / prep.slug / f"{candidate.name}.json"
    if cached.is_file():
        candidate.oracle = oracle_from_dict(_read_json(cached))
        return candidate
    result = oracle_mod.grade(
        prep=prep,
        batch_dir=batch_dir,
        checkpoint=checkpoint,
        patch_path=candidate.patch_path,
        label=f"cand-{prep.slug}-{candidate.name}",
        disk_mb=disk_mb,
        timeout_secs=ORACLE_TIMEOUT_SECS,
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    oracle_mod.save(cached, result)
    candidate.oracle = result
    return candidate


def oracle_from_dict(payload: dict[str, Any]) -> oracle_mod.OracleResult:
    """Rebuild an :class:`OracleResult` from its recorded ``as_dict`` form."""
    result = oracle_mod.OracleResult(
        apply_ok=bool(payload.get("apply_ok", False)),
        apply_detail=str(payload.get("apply_detail") or ""),
        hidden_pass=bool(payload.get("hidden_pass", False)),
        hidden_nodeids=[str(n) for n in payload.get("hidden_nodeids", [])],
        hidden_outcomes={str(k): str(v) for k, v in (payload.get("hidden_outcomes") or {}).items()},
        suite_regressions=[str(n) for n in payload.get("suite_regressions", [])],
        patched_failures=[str(n) for n in payload.get("patched_failures", [])],
        patch_good=bool(payload.get("patch_good", False)),
        oracle_secs=float(payload.get("oracle_secs") or 0.0),
        error=str(payload.get("error") or ""),
    )
    return result


def censored_oracle(prep: prepare_mod.BugPrep, reason: str) -> oracle_mod.OracleResult:
    """``apply_ok = false`` with a stated reason — §5.2's censor, built explicitly.

    Used where a run produced no patch of its own and there is therefore nothing
    to grade: a ``PatchApplicationFailed`` in review mode is a fact about the
    candidate's fit, and counting it as a ``true_reject`` would credit Vanta for
    a verdict it never reached.
    """
    return oracle_mod.OracleResult(
        apply_ok=False,
        apply_detail=reason,
        hidden_nodeids=sorted(prep.fail_to_pass),
        error=reason,
    )


def pool_failed_patches(
    *, batch_dir: Path, prep: prepare_mod.BugPrep, records: Sequence[dict[str, Any]]
) -> list[Candidate]:
    """The known-bad pool: failed Creator patches from *both* configs (§7.1).

    A pooled patch is a reviewable candidate when the oracle says it **applied
    cleanly and is not ``patch_good``** — ``apply_ok = true`` and (``hidden_pass =
    false`` or ``suite_regressions > 0``). ``InfraFailure`` runs and
    ``apply_ok = false`` patches are skipped: a patch that never applied is not a
    patch, so reviewing it would measure nothing. Deduplicated by the *normalized*
    diff, then the two with the smallest ``patch_sha256`` survive — a
    content-based, reproducible choice, so the candidate set can be recomputed
    from the manifest alone (§8.2 item 5).
    """
    pooled: dict[str, Candidate] = {}
    for record in records:
        oracle_payload = record.get("oracle") or {}
        if record.get("verdict") == "InfraFailure":
            continue
        if not oracle_payload.get("apply_ok"):
            continue
        if oracle_payload.get("patch_good"):
            continue
        patch_path = ((record.get("artifacts") or {}).get("final_patch")) or None
        if not patch_path:
            continue
        path = Path(patch_path)
        if not path.is_file():
            continue
        source = f"repair:{record.get('config')}:{record.get('session_id')}"
        try:
            text = test_hunk_free(
                path.read_text(encoding="utf-8", errors="replace"),
                what=f"{prep.key} {source}",
            )
        except RunnerError:
            # A pooled patch that is nothing but test hunks is not reviewable —
            # skip it rather than take the batch down.
            continue
        candidate = write_candidate_patch(
            batch_dir, prep, FAILED_PATCH, text, source=source
        )
        candidate.oracle = oracle_from_dict(oracle_payload)
        # Dedup by normalized diff, then keep the first by `patch_sha256`. The
        # dict key makes the dedup independent of iteration order.
        existing = pooled.get(candidate.normalized_sha256)
        if existing is None or candidate.patch_sha256 < existing.patch_sha256:
            pooled[candidate.normalized_sha256] = candidate
    ordered = sorted(pooled.values(), key=lambda c: c.patch_sha256)
    return ordered[:MAX_POOLED_CANDIDATES]


def candidate_set(
    *,
    batch_dir: Path,
    prep: prepare_mod.BugPrep,
    checkpoint: str,
    disk_mb: int,
    records: Sequence[dict[str, Any]],
) -> list[Candidate]:
    """The per-bug review set, identical for both configs (§5.4, §7.1).

    Always the developer fix (known-good by construction) and the no-op floor;
    plus up to two pooled failed patches. Built **once per bug** and reused by
    both configs' review passes, which is what makes those rows paired.
    """
    dev = write_candidate_patch(
        batch_dir,
        prep,
        DEV_FIX,
        test_hunk_free(prep.bug.bug_patch_txt, what=f"{prep.key} developer fix"),
        source="dev",
    )
    rel, noop_text = noop_patch_text(prep.root / "vanta_tree")
    noop = write_candidate_patch(
        batch_dir, prep, NOOP, test_hunk_free(noop_text, what=f"{prep.key} no-op"),
        source=f"noop:{rel}",
    )
    candidates = [dev, noop]
    candidates += pool_failed_patches(batch_dir=batch_dir, prep=prep, records=records)
    return [
        grade_candidate(
            batch_dir=batch_dir,
            prep=prep,
            candidate=candidate,
            checkpoint=checkpoint,
            disk_mb=disk_mb,
        )
        for candidate in candidates
    ]


# ---------------------------------------------------------------------------
# invoking vanta_m3 and finding what it produced
# ---------------------------------------------------------------------------
def sessions_dir() -> Path:
    return REPO_ROOT / SESSIONS_DIRNAME


def _session_stems() -> set[str]:
    directory = sessions_dir()
    if not directory.is_dir():
        return set()
    return {p.name[: -len("_session.jsonl")] for p in directory.glob("*_session.jsonl")}


@dataclass
class SessionOutcome:
    """What one ``vanta_m3`` invocation left behind."""

    session_id: Optional[str]
    exit_code: Optional[int]
    timed_out: bool
    decision: str
    summary: dict[str, str]
    raw_summary: str
    session_path: Optional[Path]
    debug_dir: Optional[Path]
    final_patch: Optional[Path]
    log_path: Path
    run_secs: float
    error: str = ""

    @property
    def usable(self) -> bool:
        """A session id and a parsed ``run.summary`` — nothing less counts as a row."""
        return self.session_id is not None and bool(self.decision) and not self.error


def parse_summary(text: str) -> dict[str, str]:
    """``key=value`` pairs of a ``run.summary`` line, in the order they appear.

    Order matters: the per-call sampling schedule is emitted site by site
    (``temperature_creator=… max_tokens_creator=…``), and the manifest reproduces
    it in that order so a reader compares it against the request bodies rather
    than against a re-sorted copy.
    """
    return {key: value for key, value in SUMMARY_PAIR.findall(text)}


def read_run_summary(session_path: Path) -> tuple[str, dict[str, str]]:
    """The last ``run.summary`` row's raw text and its parsed pairs."""
    raw = ""
    try:
        with open(session_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("phase") == "run.summary":
                    raw = str(record.get("summary") or "")
    except OSError:
        return "", {}
    return raw, parse_summary(raw)


def sampling_block(summary: dict[str, str]) -> dict[str, Any]:
    """§7.3's ``sampling`` block, read back from the pipeline's own record.

    The harness chooses no run-level value (§4.3): ``pinned`` is the per-call
    schedule the pipeline enforces, and ``server_top_p`` is the one parameter no
    call site pins, read from ``/props``. ``top_p=unset`` means the server
    published nothing — never a plausible-looking default, which would be a
    fabricated claim about a sampler nobody pinned.
    """
    sites: dict[str, dict[str, Any]] = {}
    for key, value in summary.items():
        if key.startswith("temperature_"):
            sites.setdefault(key[len("temperature_"):], {})["temperature"] = _number(value)
        elif key.startswith("max_tokens_"):
            sites.setdefault(key[len("max_tokens_"):], {})["max_tokens"] = _number(value)
    top_p = summary.get("top_p")
    return {
        # The split's own constant, not a copied literal: the manifest must name the
        # same policy the committed design declares, or the provenance block would
        # describe a design nobody reviewed.
        "source": SAMPLING_POLICY,
        "pinned": [{"site": site, **values} for site, values in sites.items()],
        "server_top_p": _number(top_p) if top_p and top_p != "unset" else None,
        "server_top_p_note": (
            "unset — /props did not publish it"
            if not top_p or top_p == "unset"
            else "read from the server's /props"
        ),
    }


def _number(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() and "." not in value else parsed


def invoke_vanta(
    *,
    vanta_bin: Path,
    prep: prepare_mod.BugPrep,
    issue_path: Path,
    env: dict[str, str],
    log_path: Path,
    max_attempts: int,
    candidate: Optional[Candidate] = None,
) -> SessionOutcome:
    """Run ``vanta_m3`` once, then discover and read what it wrote.

    The session id is a fresh random UUID generated *inside* the process, so it
    cannot be passed in — it is parsed from the ``▌ Session: {id}`` banner the
    driver prints, with a directory diff as the fallback for a banner that moved
    or a log that got truncated. The exit code is recorded but never trusted as
    the verdict: the verdict is the ``final_decision`` the session wrote down
    (§7.4's request-H contract).
    """
    argv = [
        str(vanta_bin),
        "--repo", str(prep.root / "vanta_tree"),
        "--issue", str(issue_path),
        "--test-cmd", vanta_test_cmd(prep.test_cmd),
        "--max-attempts", str(max_attempts),
    ]
    if candidate is not None:
        argv += ["--patch", str(candidate.patch_path)]

    before = _session_stems()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    exit_code: Optional[int] = None
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n")
        log.write("# env: " + " ".join(
            f"{k}={v}" for k, v in sorted(env.items())
            if k.startswith("VANTA_") and k not in {"VANTA_API_KEY"}
        ) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_SECS,
                stdin=subprocess.DEVNULL,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    elapsed = time.monotonic() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    session_id = _discover_session(log_text, before)
    outcome = SessionOutcome(
        session_id=session_id,
        exit_code=exit_code,
        timed_out=timed_out,
        decision="",
        summary={},
        raw_summary="",
        session_path=None,
        debug_dir=None,
        final_patch=None,
        log_path=log_path,
        run_secs=elapsed,
    )
    if session_id is None:
        outcome.error = (
            "no session id: neither the `▌ Session:` banner nor a new "
            f"vanta_sessions/*_session.jsonl appeared (exit {exit_code}, "
            f"timed_out={timed_out})"
        )
        return outcome

    session_path = sessions_dir() / f"{session_id}_session.jsonl"
    debug_dir = sessions_dir() / session_id / "debug"
    outcome.session_path = session_path
    outcome.debug_dir = debug_dir
    raw, summary = read_run_summary(session_path)
    outcome.raw_summary = raw
    outcome.summary = summary
    outcome.decision = summary.get("final_decision", "")
    patch = debug_dir / "final.patch"
    outcome.final_patch = patch if patch.is_file() else None
    if not outcome.decision:
        outcome.error = f"no run.summary in {session_path} (exit {exit_code})"
    return outcome


def _discover_session(log_text: str, before: set[str]) -> Optional[str]:
    matches = SESSION_LINE.findall(log_text)
    if matches:
        return matches[-1]
    after = _session_stems() - before
    if len(after) == 1:
        return after.pop()
    return None


# ---------------------------------------------------------------------------
# labels (§5.3) and the manifest row (§7.3)
# ---------------------------------------------------------------------------
def label_for(verdict: str, oracle: oracle_mod.OracleResult) -> str:
    """§5.3's label table, a pure function of (verdict, oracle).

    A verdict outside the three that produce a patch has no label of its own; its
    row carries the censor, so it lands on ``unobservable`` through the
    ``apply_ok`` branch rather than through a fabricated name.
    """
    if verdict == "InfraFailure":
        return "infra_failure"
    if not oracle.apply_ok:
        return "unobservable"
    if verdict == "AllTestsPassed":
        return "true_approval" if oracle.patch_good else "false_approval"
    if verdict == "Rejected":
        return "false_reject" if oracle.patch_good else "true_reject"
    if verdict == "MaxAttemptsReached":
        return "unresolved_but_fixed" if oracle.patch_good else "unresolved"
    return "unobservable"


def build_record(
    *,
    job: "Job",
    prep: prepare_mod.BugPrep,
    entry: dict[str, Any],
    outcome: SessionOutcome,
    oracle: oracle_mod.OracleResult,
    issue_sha256: str,
    prepare_s: float,
    oracle_secs: float,
    oracle_reused: bool,
    python_version: str,
    oracle_json: Optional[Path],
) -> dict[str, Any]:
    """One ``manifest.jsonl`` line (§7.3)."""
    summary = outcome.summary
    total = outcome.run_secs + oracle_secs
    return {
        "schema_version": manifest_mod.SCHEMA_VERSION,
        "kind": "run",
        "key": job.key,
        "batch_id": job.batch_id,
        "created_utc": utc_now(),
        "wave": job.wave,
        "bug": job.bug,
        "config": job.config,
        "mode": job.mode,
        "seed": job.seed,
        "session_id": outcome.session_id,
        "vanta_commit": summary.get("commit"),
        "vanta_dirty": _bool(summary.get("dirty")),
        "model": summary.get("model"),
        "sampling": sampling_block(summary),
        "checkpoint": job.checkpoint,
        # `python_version` is what the checkpoint ACTUALLY contains; the pinned one
        # is what BugsInPy's bug.info declares. §12.3 says they differ (3.8.20 vs
        # 3.8.1), and both are recorded so the gap is auditable rather than hidden
        # by writing the actual number into a field that claims to be the pin.
        "python_version": python_version or "unknown",
        "python_version_pinned": job.python_version_pinned,
        "pytest_version": entry.get("pytest_version"),
        "json_report_version": entry.get("json_report_version"),
        "candidate": job.candidate.as_dict() if job.candidate else None,
        "verdict": outcome.decision or None,
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        "attempts_run": _int(summary.get("attempts_run")),
        "fixed_at_attempt_1": _bool(summary.get("fixed_at_attempt_1")),
        "oracle": oracle.as_dict(),
        "label": label_for(outcome.decision, oracle),
        "timings": {
            "prepare_s": round(prepare_s, 3),
            "run_s": round(outcome.run_secs, 3),
            "oracle_s": round(oracle_secs, 3),
            "total_s": round(total, 3),
        },
        # True when the row reused the candidate's label instead of grading again
        # (§5.4): the same bytes were already graded, so a second grading would
        # repeat an experiment rather than add evidence.
        "oracle_reused": oracle_reused,
        "artifacts": {
            "session_dir": f"{SESSIONS_DIRNAME}/{outcome.session_id}",
            "final_patch": str(outcome.final_patch) if outcome.final_patch else None,
            "oracle_json": str(oracle_json) if oracle_json else None,
            "run_log": str(outcome.log_path),
            "graded_tree_diff": None,
            "issue_sha256": issue_sha256,
            "compact_state": "full",
        },
    }


def _bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.strip().lower() in {"true", "1", "yes"}


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------
@dataclass
class Job:
    batch_id: str
    wave: int
    bug: str
    prep: prepare_mod.BugPrep
    entry: dict[str, Any]
    config: str
    mode: str
    seed: int
    checkpoint: str
    python_version_pinned: str
    candidate: Optional[Candidate] = None

    @property
    def key(self) -> str:
        return job_key(
            batch_id=self.batch_id,
            bug=self.bug,
            config=self.config,
            mode=self.mode,
            seed=self.seed,
            candidate=self.candidate.patch_sha256 if self.candidate else None,
        )

    def label(self) -> str:
        suffix = f" {self.candidate.name}" if self.candidate else ""
        return f"[{self.bug} {self.config} {self.mode} seed={self.seed}{suffix}]"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
@dataclass
class BatchPaths:
    batch_dir: Path
    logs: Path
    issues: Path
    oracle: Path
    env_cache: Path
    failures: Path

    @classmethod
    def under(cls, batch_dir: Path) -> "BatchPaths":
        return cls(
            batch_dir=batch_dir,
            logs=batch_dir / "run_logs",
            issues=batch_dir / "issues",
            oracle=batch_dir / "oracle",
            env_cache=batch_dir / "env",
            failures=batch_dir / "failures.jsonl",
        )


def python_version_of(paths: BatchPaths, checkpoint: str, disk_mb: int) -> str:
    """The interpreter version the checkpoint actually contains, probed once.

    §7.3 records both this and the pinned version. Probing costs one boot per
    checkpoint and is cached, so a 100-run batch pays it once per environment.
    """
    cached = paths.env_cache / f"{checkpoint}.json"
    if cached.is_file():
        return str(_read_json(cached).get("python_version") or "")
    from . import shuru as shuru_mod

    probe = shuru_mod.run(
        checkpoint=checkpoint,
        host_root=paths.batch_dir,
        command=["sh", "-c", "python --version; python3 --version"],
        disk_mb=disk_mb,
        timeout_secs=PROBE_TIMEOUT_SECS,
    )
    text = (probe.stdout + probe.stderr).strip()
    version = ""
    match = re.search(r"Python (\d+\.\d+\.\d+)", text)
    if match:
        version = match.group(1)
    payload = {
        "checkpoint": checkpoint,
        "python_version": version,
        "probe_exit_code": probe.exit_code,
        "probe_timed_out": probe.timed_out,
        "probe_output": text[-2000:],
        "probed_utc": utc_now(),
    }
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return version


def run_job(
    *,
    job: Job,
    paths: BatchPaths,
    manifest: Manifest,
    vanta_bin: Path,
    max_attempts: int,
    disk_mb: int,
    python_version: str,
) -> Optional[dict[str, Any]]:
    """Execute one job end to end, or return ``None`` if it left no evidence.

    Ordering is the resume guarantee from ``manifest.py``: ``running`` before the
    work, the row fsynced, then ``done``. A job that produced no session id and no
    summary has no row to write — a half-written row would be a fabricated
    dataset entry — so its failure goes to ``failures.jsonl`` instead and the
    cursor stays ``failed``, which means it is re-run rather than skipped.
    """
    manifest.mark(job.key, RUNNING)
    prep = job.prep
    issue_path = paths.issues / f"{prep.slug}.txt"
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    issue_path.write_text(prep.issue.text if prep.issue else "", encoding="utf-8")
    issue_sha256 = sha256_text(issue_path.read_text(encoding="utf-8"))

    log_path = paths.logs / f"{_stem(job.key)}.log"
    env = vanta_env(
        config=job.config,
        seed=job.seed,
        checkpoint=job.checkpoint,
        disk_mb=disk_mb,
    )
    outcome = invoke_vanta(
        vanta_bin=vanta_bin,
        prep=prep,
        issue_path=issue_path,
        env=env,
        log_path=log_path,
        max_attempts=max_attempts,
        candidate=job.candidate,
    )
    if not outcome.usable:
        _append_failure(paths, job, outcome)
        manifest.mark(job.key, FAILED)
        print(f"  {job.label()} FAILED — {outcome.error}", flush=True)
        return None

    oracle_reused = job.candidate is not None
    oracle_json: Optional[Path] = None
    if oracle_reused:
        # The candidate carries the label its own bytes were graded with. Only
        # reuse it when this session produced a patch; otherwise the run never
        # applied the candidate, so the candidate's label describes a tree that
        # was not under test (§5.4).
        if outcome.final_patch is None:
            oracle = censored_oracle(
                prep, f"no final.patch for verdict {outcome.decision}"
            )
            oracle_reused = False
        else:
            oracle = job.candidate.oracle  # type: ignore[assignment]
    else:
        oracle_json = paths.oracle / f"{_stem(job.key)}.json"
        patch_path = outcome.final_patch
        if patch_path is None:
            # A repair run that reaches PatchApplicationFailed / HarnessFailure
            # writes no patch. `oracle.grade` censors a missing patch instead of
            # raising, which is the §5.2 reading: there is nothing to grade, so the
            # row is `unobservable` rather than a verdict about the model.
            patch_path = (outcome.debug_dir or paths.batch_dir) / "final.patch"
        oracle = oracle_mod.grade(
            prep=prep,
            batch_dir=paths.batch_dir,
            checkpoint=job.checkpoint,
            patch_path=patch_path,
            label=outcome.session_id or "unlabelled",
            disk_mb=disk_mb,
            timeout_secs=ORACLE_TIMEOUT_SECS,
        )
        oracle_mod.save(oracle_json, oracle)

    record = build_record(
        job=job,
        prep=prep,
        entry=job.entry,
        outcome=outcome,
        oracle=oracle,
        issue_sha256=issue_sha256,
        # `prepare` is a separate stage whose timings live in `prepare.json`; a run
        # row cannot claim a share of it, and the field keeps its §7.3 position with
        # the honest value.
        prepare_s=0.0,
        oracle_secs=0.0 if oracle_reused else oracle.oracle_secs,
        oracle_reused=oracle_reused,
        python_version=python_version,
        oracle_json=oracle_json,
    )
    manifest.append(record)
    manifest.mark(job.key, DONE)
    print(
        f"  {job.label()} → {record['verdict']} "
        f"patch_good={record['oracle']['patch_good']} label={record['label']} "
        f"({record['timings']['total_s']:.0f}s)",
        flush=True,
    )
    return record


def _append_failure(paths: BatchPaths, job: Job, outcome: SessionOutcome) -> None:
    paths.failures.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": utc_now(),
        "key": job.key,
        "bug": job.bug,
        "config": job.config,
        "mode": job.mode,
        "seed": job.seed,
        "session_id": outcome.session_id,
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        "error": outcome.error,
        "run_log": str(outcome.log_path),
    }
    with open(paths.failures, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _stem(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)


@dataclass
class WaveResult:
    wave: int
    records: list[dict[str, Any]] = field(default_factory=list)
    failed: int = 0
    """Jobs that left no usable session. Counted separately from ``records`` so the
    caller can tell "the wave is short" from "the wave is broken"."""


def run_batch(
    *,
    batch_dir: Path,
    batch_id: str,
    split: Split,
    preps: list[prepare_mod.BugPrep],
    entries: dict[str, dict[str, Any]],
    seeds: Sequence[int],
    waves: Sequence[int],
    vanta_bin: Path,
    max_attempts: int,
    disk_mb: int,
    plan_only: bool = False,
) -> tuple[int, list[WaveResult]]:
    """The whole batch: per wave, D13's four phases, then the wave-1 gate."""
    paths = BatchPaths.under(batch_dir)
    manifest = Manifest(batch_dir)
    # Not filtered down to the two known names: a filter here would drop
    # `vanta_planner` before the guard could refuse it, and a split that names the
    # placeholder must not start at all (§4.4).
    configs = list(split.configs)
    _require_phase_order(configs)

    plan = _render_schedule(batch_id, preps, entries, configs, seeds, waves)
    print(plan, flush=True)
    if plan_only:
        return 0, []

    checkpoint_pythons: dict[str, str] = {}
    for prep in preps:
        checkpoint = entries[prep.key]["checkpoint"]
        if checkpoint not in checkpoint_pythons:
            checkpoint_pythons[checkpoint] = python_version_of(paths, checkpoint, disk_mb)
            print(
                f"[env] {checkpoint}: python "
                f"{checkpoint_pythons[checkpoint] or 'unknown'} "
                f"(pinned {entries[prep.key]['python_version']})",
                flush=True,
            )

    results: list[WaveResult] = []
    for wave in waves:
        wave_preps = [p for p in preps if p.wave == wave]
        if not wave_preps:
            continue
        print(f"\n=== wave {wave}: {len(wave_preps)} bug(s) ===", flush=True)
        result = WaveResult(wave=wave)
        for mode, config in PHASES:
            if config not in configs:
                continue
            for prep in wave_preps:
                entry = entries[prep.key]
                checkpoint = entry["checkpoint"]
                for seed in seeds:
                    if mode == REPAIR:
                        jobs = [
                            Job(
                                batch_id=batch_id,
                                wave=wave,
                                bug=prep.key,
                                prep=prep,
                                entry=entry,
                                config=config,
                                mode=REPAIR,
                                seed=seed,
                                checkpoint=checkpoint,
                                python_version_pinned=entry["python_version"],
                            )
                        ]
                    else:
                        jobs = [
                            Job(
                                batch_id=batch_id,
                                wave=wave,
                                bug=prep.key,
                                prep=prep,
                                entry=entry,
                                config=config,
                                mode=REVIEW,
                                seed=seed,
                                checkpoint=checkpoint,
                                python_version_pinned=entry["python_version"],
                                candidate=candidate,
                            )
                            for candidate in _review_candidates(
                                batch_dir=batch_dir,
                                batch_id=batch_id,
                                manifest=manifest,
                                prep=prep,
                                configs=configs,
                                seed=seed,
                                checkpoint=checkpoint,
                                disk_mb=disk_mb,
                            )
                        ]
                    for job in jobs:
                        if manifest.is_done(job.key):
                            print(f"  {job.label()} done — skipping (resume)", flush=True)
                            found = manifest.find(job.key)
                            if found:
                                result.records.append(found)
                            continue
                        record = run_job(
                            job=job,
                            paths=paths,
                            manifest=manifest,
                            vanta_bin=vanta_bin,
                            max_attempts=max_attempts,
                            disk_mb=disk_mb,
                            python_version=checkpoint_pythons.get(checkpoint, ""),
                        )
                        if record:
                            result.records.append(record)
                        else:
                            result.failed += 1
        results.append(result)
        gate = wave_gate_failures(result.records)
        if gate:
            print(f"\n=== wave {wave} GATE FAILED ===", flush=True)
            for problem in gate:
                print(f"  - {problem}", flush=True)
            print(
                "Stopping before the next wave: the earlier stages are the ones "
                "whose failure invalidates every later number (§8.1).",
                flush=True,
            )
            return 1, results
    return 0, results


def _review_candidates(
    *,
    batch_dir: Path,
    batch_id: str,
    manifest: Manifest,
    prep: prepare_mod.BugPrep,
    configs: Sequence[str],
    seed: int,
    checkpoint: str,
    disk_mb: int,
) -> list[Candidate]:
    """The candidate set for one (bug, seed), from the repair rows already written.

    ``batch_id`` is threaded in rather than read off ``manifest.batch_dir.name``:
    it is the one field a caller could pass under a different name than the
    directory it writes into, and a lookup keyed on the wrong batch would silently
    find no repair rows and quietly shrink every review set to {dev fix, no-op}.
    """
    records: list[dict[str, Any]] = []
    for config in configs:
        key = job_key(
            batch_id=batch_id,
            bug=prep.key,
            config=config,
            mode=REPAIR,
            seed=seed,
        )
        found = manifest.find(key)
        if found:
            records.append(found)
    return candidate_set(
        batch_dir=batch_dir,
        prep=prep,
        checkpoint=checkpoint,
        disk_mb=disk_mb,
        records=records,
    )


REVIEW_CONFIGS = ("baseline", "vanta")
"""The pair §7.1 reviews, in the order the phases run.

``vanta_planner`` is deliberately absent: it is the §4.4 placeholder whose review
feedback comes from native+static only and whose runs return ``InfraFailure``, so a
split containing it cannot produce a third arm — only rows that fail the wave gate.
"""


def _require_phase_order(configs: Sequence[str]) -> None:
    """Refuse any config list but §7.1's pair, and refuse a reordered phase list.

    Two different mistakes land here and they need different sentences. A config
    list that is not exactly ``["baseline", "vanta"]`` is not a scheduling problem:
    the study has no third arm, and a review row with one config missing cannot be
    paired, so the batch must not start at all. A ``PHASES`` that no longer puts
    every repair before every review is a scheduling bug. Both are fatal, and
    neither is a preference — which is why this raises instead of reordering.
    """
    if list(configs) != list(REVIEW_CONFIGS):
        raise RunnerError(
            f"split configs must be exactly {list(REVIEW_CONFIGS)} for paired review "
            f"(got {list(configs)}); vanta_planner is a placeholder that returns "
            "InfraFailure and must not be in a run split"
        )
    expected = [(REPAIR, config) for config in configs] + [
        (REVIEW, config) for config in configs
    ]
    if list(PHASES) != expected:
        raise RunnerError(
            f"PHASES {PHASES} no longer matches {expected}: §7.1/D13 requires every "
            "repair pass to precede every review pass, and that ordering is a "
            "correctness property, not a preference"
        )


def wave_gate_failures(records: Sequence[dict[str, Any]]) -> list[str]:
    """§8.2's wave gate, as assertions over the rows a wave just wrote.

    Wave 1 is a go/no-go gate: if the oracle or the plumbing misbehaves there, the
    batch must stop and report rather than build the next wave on it. The checks
    are the ones whose failure invalidates every later number — not a threshold on
    model behaviour, which is the thing being measured.
    """
    problems: list[str] = []
    if not records:
        return ["the wave produced no records at all"]
    for record in records:
        tag = f"{record['bug']} {record['config']} {record['mode']}"
        if record.get("verdict") == "InfraFailure":
            problems.append(f"{tag}: InfraFailure — the environment, not the model")
        elif record.get("label") == "unobservable":
            problems.append(
                f"{tag}: unobservable — verdict={record.get('verdict')} "
                f"({record['oracle'].get('apply_detail') or record['oracle'].get('error')})"
            )
    # §8.2 item 5: the developer fix must grade good and the no-op must not. This
    # is the one pair of labels the harness knows before any run, so a disagreement
    # is proof that the oracle or the tree split is wrong.
    by_bug: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        candidate = record.get("candidate")
        if record["mode"] == REVIEW and candidate:
            by_bug.setdefault(record["bug"], {})[candidate["kind"]] = candidate
    for bug, kinds in sorted(by_bug.items()):
        dev = kinds.get(DEV_FIX)
        if dev is not None and not dev["oracle"].get("patch_good"):
            problems.append(f"{bug}: the developer fix graded patch_good=false")
        noop = kinds.get(NOOP)
        if noop is not None and noop["oracle"].get("patch_good"):
            problems.append(f"{bug}: the no-op floor graded patch_good=true")
    # §8.2: "paired vanta rows exist". Both configs review one set at one seed, so
    # a review row with no counterpart is not a pairing the study can use — and the
    # asymmetry would silently halve a comparison instead of failing.
    paired = {
        (record["bug"], record["seed"], record["candidate"]["normalized_sha256"], record["config"])
        for record in records
        if record["mode"] == REVIEW and record.get("candidate")
    }
    configs = {config for _, _, _, config in paired}
    for bug, seed, normalized, config in sorted(paired):
        for other in sorted(configs - {config}):
            if (bug, seed, normalized, other) not in paired:
                problems.append(
                    f"{bug} seed={seed} candidate={normalized[:12]}: {config} has a "
                    f"review row but {other} does not — the pair is incomplete"
                )
    return problems


def _render_schedule(
    batch_id: str,
    preps: Sequence[prepare_mod.BugPrep],
    entries: dict[str, dict[str, Any]],
    configs: Sequence[str],
    seeds: Sequence[int],
    waves: Sequence[int],
) -> str:
    """The plan, and an honest statement of what is not yet known.

    The review row count is a **floor**, not a prediction: the pooled failed
    patches do not exist until the repairs ran, so the set printed here is
    {developer fix, no-op} and the real one is 2–4 per bug (§7.1).
    """
    lines = [
        f"batch {batch_id}",
        f"order: {' → '.join(f'{mode} {config}' for mode, config in PHASES if config in configs)}",
        "",
        "| wave | bug | checkpoint | py (pinned) | repair rows | review rows (floor) |",
        "|---|---|---|---|---|---|",
    ]
    total = 0
    for wave in waves:
        for prep in preps:
            if prep.wave != wave:
                continue
            entry = entries[prep.key]
            repair = len(configs) * len(seeds)
            review = len(configs) * 2 * len(seeds)
            total += repair + review
            lines.append(
                f"| {wave} | {prep.key} | {entry['checkpoint']} | "
                f"{entry['python_version']} | {repair} | {review} |"
            )
    lines += [
        "",
        f"{total} rows if no bug yields a pooled failed patch; "
        f"up to {total * 2} if every bug yields two (§7.1).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading back a prepared batch (resume)
# ---------------------------------------------------------------------------
def load_prepared(
    *,
    batch_dir: Path,
    split: Split,
    bugsinpy_root: Path,
) -> tuple[list[prepare_mod.BugPrep], dict[str, dict[str, Any]]]:
    """Rebuild genuine :class:`BugPrep` objects from ``prepare.json``.

    No second persisted format: the runner reads the *same* artifact ``prepare``
    wrote, so a batch that was prepared in one process can be run in another
    without re-deriving three double suite runs per bug. Every field ``oracle.grade``
    reads is restored, and ``load_bug`` re-reads ``bug.info`` so a moved clone
    fails loudly here rather than silently grading against a tree nobody prepared.
    """
    payload = _read_json(batch_dir / "prepare.json")
    if not payload:
        raise RunnerError(
            f"no prepare.json in {batch_dir}; run `bugsinpy_run.py prepare --batch "
            f"{batch_dir}` first"
        )
    entries = {_bug_key(entry): entry for entry in split.bugs}
    preps: list[prepare_mod.BugPrep] = []
    for raw in payload.get("bugs", []):
        key = str(raw.get("bug"))
        entry = entries.get(key)
        if entry is None:
            raise RunnerError(f"prepare.json names {key}, which the split does not")
        project = str(entry["project"])
        bug_id = int(entry["bug_id"])
        bug = prepare_mod.load_bug(split, bugsinpy_root, project, bug_id)
        prep = prepare_mod.BugPrep(
            bug=bug,
            root=batch_dir / "worktrees" / f"{project}_{bug_id}",
            role=str(raw.get("role") or entry.get("role", "pilot")),
            wave=int(raw.get("wave") or entry["wave"]),
            test_cmd=[str(part) for part in entry["test_cmd"]],
            prepare_mode=str(raw.get("prepare_mode") or prepare_mod.FULL),
            test_paths=[str(p) for p in raw.get("test_paths", [])],
            fail_to_pass=[str(n) for n in raw.get("fail_to_pass", [])],
            pass_to_pass_count=int(raw.get("pass_to_pass_count") or 0),
            flaky_nodeids=[str(n) for n in raw.get("flaky_nodeids", [])],
            preexisting_failures=[str(n) for n in raw.get("preexisting_failures", [])],
            native_failure_baseline=[
                str(n) for n in raw.get("native_failure_baseline", [])
            ],
            native_green_on_vanta_tree=bool(raw.get("native_green_on_vanta_tree")),
            hidden_files_deleted=[str(p) for p in raw.get("hidden_files_deleted", [])],
            partially_hidden_functions=[
                str(p) for p in raw.get("partially_hidden_functions", [])
            ],
            gate_failures=[str(f) for f in raw.get("gate_failures", [])],
        )
        issue = raw.get("issue") or {}
        if issue:
            prep.issue = _issue_from_dict(issue)
        preps.append(prep)
    return preps, entries


def _issue_from_dict(payload: dict[str, Any]):
    """The smallest object ``.text``/``.ok`` satisfy for the runner's uses."""
    from .issue import IssueText

    text = str(payload.get("text") or "")
    issue = IssueText(text=text, pre_gate=str(payload.get("pre_gate") or ""), ok=bool(payload.get("ok")))
    issue.needs_decision = payload.get("needs_decision")
    issue.failure_kinds = list(payload.get("failure_kinds") or [])
    return issue


def _bug_key(entry: dict[str, Any]) -> str:
    return f"{entry['project']}/{entry['bug_id']}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_labels(records: Iterable[dict[str, Any]], path: Path) -> Path:
    """``metrics.py --labels`` input, built from the manifest (§5.4).

    Only review rows carry a label: the *candidate* has a ground truth the oracle
    established, so an approval there is checkable. A repair row's patch has no
    declared truth — that is what its own oracle column is for — so labelling it
    here would claim a study design the batch does not have.
    """
    labels: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate = record.get("candidate")
        session_id = record.get("session_id")
        if not candidate or not session_id:
            continue
        labels[str(session_id)] = {
            "known_bad": not bool(candidate["oracle"].get("patch_good")),
            "bug": record["bug"],
            "config": record["config"],
            "mode": record["mode"],
            "kind": candidate["kind"],
            "source": candidate["source"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def digest(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per (§4.4 provenance axes), plus the label histogram per config."""
    groups: dict[tuple, dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("config"),
            record.get("mode"),
            record.get("seed"),
            record.get("model"),
            record.get("vanta_commit"),
        )
        group = groups.setdefault(
            key,
            {
                "config": record.get("config"),
                "mode": record.get("mode"),
                "seed": record.get("seed"),
                "model": record.get("model"),
                "vanta_commit": record.get("vanta_commit"),
                "vanta_dirty": record.get("vanta_dirty"),
                "runs": 0,
                "labels": {},
            },
        )
        group["runs"] += 1
        label = str(record.get("label"))
        group["labels"][label] = group["labels"].get(label, 0) + 1
    return {"runs": len(records), "groups": list(groups.values())}


def render_digest(payload: dict[str, Any]) -> str:
    lines = [
        f"{'config':<10} {'mode':<8} {'seed':>4} {'model':<18} {'commit':<10} "
        f"{'runs':>5}  labels",
        "-" * 96,
    ]
    for group in payload.get("groups", []):
        commit = str(group.get("vanta_commit") or "")[:9]
        model = str(group.get("model") or "")[:18]
        labels = " ".join(f"{k}={v}" for k, v in sorted(group["labels"].items()))
        lines.append(
            f"{str(group['config']):<10} {str(group['mode']):<8} {str(group['seed']):>4} "
            f"{model:<18} {commit:<10} {group['runs']:>5}  {labels}"
        )
    return "\n".join(lines)


def write_batch_report(
    batch_dir: Path, records: Sequence[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    """Write the batch's report artifacts, from the manifest and nothing else.

    Read-only over the evidence: the rows are the dataset, and the digest and the
    labels file are two views of them that can be regenerated at any time — so a
    crash between a run and a report costs nothing, and a report can never disagree
    with the rows it summarises.
    """
    labels_path = write_labels(records, batch_dir / "run_labels.json")
    payload = digest(records)
    (batch_dir / "run_digest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (batch_dir / "run_digest.md").write_text(render_digest(payload) + "\n", encoding="utf-8")
    return labels_path, payload
