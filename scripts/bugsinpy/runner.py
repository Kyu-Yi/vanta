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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from . import bake as bake_mod
from . import compact as compact_mod
from . import arms as arms_mod
from . import context as ctx_mod
from . import issue_source
from . import manifest as manifest_mod
from . import oracle as oracle_mod
from . import prepare as prepare_mod
from . import records as records_mod
from . import redact as redact_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .manifest import DONE, FAILED, RUNNING, Manifest, job_key
from .shuru import ShuruError
from .split import SAMPLING_POLICY, Split, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs" / "bugsinpy"
SESSIONS_DIRNAME = "vanta_sessions"
DEFAULT_VANTA_BIN = REPO_ROOT / "target" / "release" / "vanta_m3"

VANTA_SOURCE_PATHS = ("src", "Cargo.toml", "Cargo.lock", "build.rs")
"""What "the binary matches the sources" is about (§7.3).

The binary embeds the commit ``build.rs`` saw when cargo last built it, so a row's
Vanta provenance is the **build** commit. The harness is a different tree with its
own commit, and it is deliberately absent from this list: a change under
``scripts/`` makes the binary old, not stale — the Rust it was built from is
untouched. Conflating the two is the bug that produced ``44f58ab`` in a digest
whose HEAD was ``b2efa49``.
"""

# §8.1's worst single run is ~30 min for a vanta repair; 3 h is a backstop, not an
# estimate. A run that hits it is a failure to report, not a result to grade.
RUN_TIMEOUT_SECS = 3 * 60 * 60
# §14.15 — how far short of `RUN_TIMEOUT_SECS` Vanta's own deadline is set. The
# margin has to cover the longest single LLM call the harness expects (5-10
# minutes on sanic's 50-65k-token prompts), because the deadline is checked
# BEFORE each call: the call already in flight when the clock runs out still
# finishes. Ten minutes is that call, plus the terminal write.
DEADLINE_MARGIN_SECS = 10 * 60
PROBE_TIMEOUT_SECS = 120
ORACLE_TIMEOUT_SECS = 1800

# The harness's two modes. Defined in the arm registry, which is keyed by them, and
# re-exported here because this module and the split's vocabulary share them.
REPAIR = arms_mod.REPAIR
REVIEW = arms_mod.REVIEW

DEV_FIX = "dev_fix"
NOOP = "noop"
FAILED_PATCH = "failed_patch"

# The most failed patches one bug may contribute to its review set (§7.1).
MAX_POOLED_CANDIDATES = 2

# §14.29 — where a batch that stopped at `VANTA_MAX_SPEND_USD` writes down why.
# A file rather than only a print: the terminal scrolls away and a wrapper reads
# the exit code, but whoever opens the directory later has to be able to tell a
# batch that ran out of budget from one that finished.
SPEND_CAP_NAME = "spend.json"

SESSION_LINE = re.compile(r"▌ Session:\s*(\S+)")
SUMMARY_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

# [VANTA-WHY] The schedule used to be a module-level ``PHASES`` constant, spelled out
# for §7.1's two configs and checked against the split's list at every batch start. It
# is now derived (``arms.phases_for``) from the split's own spellings, because M3 item
# 11 adds a repair-only arm (B0) that no four-element constant can express — and a
# derived order cannot drift from the list it came from, which is the only thing the
# old ``PHASES != expected`` check existed to catch.


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


# ---- item 14: is the configured model endpoint actually there? -------------
MODEL_UNREACHABLE = "model_unreachable"
"""The one token a batch that cannot reach its model refuses with (item 14).

[VANTA-WHY] A fixed token rather than a sentence, because this refusal has to be
told apart from a wave gate, a spend cap and a crash by something that is not a
human reading prose — a wrapper, or whoever opens the batch directory later. The
sentence after the token is for the human.
"""

MODEL_PREFLIGHT_TIMEOUT_SECS = 10
"""Seconds the tiny request may take before the endpoint counts as absent.

Ten, not the 3h a job gets: this is a liveness question, and a request that has
not been answered in ten seconds is a request that is not going to be. The
endpoint is either on this host or on a service that answers a GET in
milliseconds.
"""

BACKEND_DEFAULT_BASE_URLS = {
    "llamacpp": "http://127.0.0.1:8080/v1",
    "openai": "https://api.openai.com/v1",
}
"""Mirrors ``LlmBackend::default_base_url`` in ``src/config/mod.rs``.

[VANTA-WHY] The preflight has to probe the endpoint the *run* will use. When
``VANTA_LLM_BASE_URL`` is unset, that endpoint is the backend's default, and the
two backends do not share one: a preflight hard-coded to ``127.0.0.1:8080`` would
probe the local router while an ``openai`` run talked to the hosted service — a
liveness check on an endpoint nobody was going to call.
"""

KEY_ENV_BY_BACKEND = {"llamacpp": "VANTA_API_KEY", "openai": "OPENAI_API_KEY"}
"""Mirrors the key-name rule in ``LlmConfig::from_env`` — one name per backend.

[VANTA-WHY] The two names deliberately do not fall back to one another in the
Rust, and this copy must not either: on the openai backend the run reads
``OPENAI_API_KEY`` and nothing else, so a probe that accepted either name could
report the endpoint reachable with a credential the run will never send.
"""


@dataclass(frozen=True)
class ProbeReply:
    """What the endpoint answered. The status, and nothing the body contained.

    [VANTA-WHY] The body is deliberately not kept. A response from this endpoint
    can carry a provider error that echoes the credential back, and the preflight
    is the one place a body arrives before any redaction runs. There is nothing in
    it this needs: the status is the whole answer.
    """

    status: int
    detail: str = ""


def model_endpoint_base_url(env: Optional[Mapping[str, str]] = None) -> str:
    """The endpoint a run would use, resolved the way the Rust resolves it.

    ``VANTA_LLM_BASE_URL`` (trailing ``/`` trimmed) if it says anything, else the
    configured backend's default. An empty or whitespace-only variable counts as
    unset, exactly as in ``LlmConfig::from_env``.
    """
    environ = os.environ if env is None else env
    explicit = (environ.get("VANTA_LLM_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    backend = (environ.get("VANTA_LLM_BACKEND") or "").strip().lower()
    return BACKEND_DEFAULT_BASE_URLS.get(backend, BACKEND_DEFAULT_BASE_URLS["llamacpp"])


def _model_preflight_headers(env: Mapping[str, str]) -> dict[str, str]:
    """The Authorization header, if the run would send one.

    Read here and never printed: the value goes into the request the same way the
    real run's would, and this function's callers keep it out of every message.
    """
    backend = (env.get("VANTA_LLM_BACKEND") or "").strip().lower()
    key = (env.get(KEY_ENV_BY_BACKEND.get(backend, "VANTA_API_KEY")) or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _urllib_probe(url: str, headers: Mapping[str, str], timeout: float) -> ProbeReply:
    """The real request: a GET that keeps the status and drops the body."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return ProbeReply(status=int(response.status))
    except urllib.error.HTTPError as exc:
        # An HTTP error IS an answer — the endpoint is up and says so. Only a
        # transport failure means it is not there.
        return ProbeReply(status=int(exc.code), detail=str(exc.reason or ""))


def model_endpoint_preflight(
    *,
    env: Optional[Mapping[str, str]] = None,
    probe: Optional[Callable[[str, Mapping[str, str], float], ProbeReply]] = None,
) -> Optional[str]:
    """One tiny request to the configured endpoint (item 14).

    ``None`` when it answered, else a sentence saying why it did not.

    [VANTA-WHY] The request is a ``GET {base_url}/models``, not a chat completion.
    The batch asks this question once before it starts and once before every row,
    so the request has to be one a run can afford to repeat: a completion costs
    tokens on a billed backend and can take tens of seconds on a reasoning model,
    and a preflight that added either to every row would be worse than the
    overnight batch of failed rows it exists to prevent. ``/models`` is the one
    endpoint the OpenAI protocol requires of both backends this project talks to
    (the local llama.cpp router publishes it, and so does the hosted service), and
    it answers reachability and credential validity in one round trip.

    Failure means **the endpoint could not be reached**, or it answered 401/403 —
    the credential the run would send is one it will not authenticate with. Any
    other status is an answer, including a 404 from a server that publishes no
    model list: the endpoint is there, so the batch proceeds and, if the endpoint
    genuinely cannot serve, each row says so itself. Refusing on that 404 would
    turn "runs as before" into "refuses to start" for an endpoint that was fine.
    """
    environ = os.environ if env is None else env
    url = model_endpoint_base_url(environ) + "/models"
    ask = probe or _urllib_probe
    try:
        reply = ask(url, _model_preflight_headers(environ), MODEL_PREFLIGHT_TIMEOUT_SECS)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"no answer from {url} ({exc})"
    if reply.status in (401, 403):
        return (
            f"{url} answered HTTP {reply.status} — the endpoint is up but rejected "
            "the credential, so every row would fail to authenticate"
        )
    return None


def require_model_endpoint(
    *,
    preflight: Optional[Callable[[], Optional[str]]] = None,
    before: Optional[str] = None,
) -> None:
    """Refuse the batch when the model endpoint does not answer (item 14).

    Called once by ``run`` before the first job and once by :func:`run_batch`
    before each row. A refusal is a `RunnerError`, so the CLI prints one sentence
    and exits 2 — and because it is raised *before* ``run_job`` is called, no row
    at all is written for a model the harness could not reach: not a failed row,
    not a partial one.
    """
    reason = (preflight or model_endpoint_preflight)()
    if not reason:
        return
    where = f" (before {before})" if before else ""
    raise RunnerError(f"{MODEL_UNREACHABLE}: {reason}{where}")


# ---------------------------------------------------------------------------
# provenance: two trees, two commits (§7.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Revision:
    """A commit and whether the tree it was taken from was dirty.

    Both fields may be ``None``: ``commit`` when the revision could not be read at
    all, ``dirty`` when nothing could say. A missing commit is not the same as a
    clean tree, which is why it is not spelled ``""``.
    """

    commit: Optional[str] = None
    dirty: Optional[bool] = None


def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Stdout of a ``git`` command in *repo_root*, or ``None`` if it did not run."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_revision(repo_root: Path = REPO_ROOT) -> Revision:
    """The harness tree's own revision, right now.

    Read at the moment the batch starts rather than per row: the rows of one batch
    are one experiment, and a harness that changed halfway through it would make
    the wave-gate comparison a comparison of two harnesses (§4.4).
    """
    commit = _git(repo_root, "rev-parse", "--short", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return Revision(commit=commit or None, dirty=None if status is None else bool(status))


def _rustc_env(path: Path) -> dict[str, str]:
    """The ``cargo:rustc-env=KEY=VALUE`` pairs of one build-script output file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("cargo:rustc-env="):
            key, _, value = line[len("cargo:rustc-env=") :].partition("=")
            fields[key] = value
    return fields


def binary_revision(vanta_bin: Path) -> Revision:
    """The commit cargo baked into *vanta_bin* (``build.rs`` → ``rustc-env``).

    Read from cargo's build-script output next to the binary rather than by asking
    the binary: the pipeline does report the same pair in ``run.summary``, but only
    *after* a session has run, and the staleness check has to fire before the first
    job starts. An unreadable or ambiguous build directory yields an empty
    :class:`Revision` — never a guess.
    """
    commits: set[str] = set()
    dirty: Optional[bool] = None
    for output in sorted((vanta_bin.parent / "build").glob("vanta-*/output")):
        fields = _rustc_env(output)
        commit = fields.get("VANTA_BUILD_COMMIT")
        if commit:
            commits.add(commit)
            dirty = fields.get("VANTA_BUILD_DIRTY") == "true"
    if len(commits) != 1:
        return Revision()
    return Revision(commit=commits.pop(), dirty=dirty)


def changed_since(
    repo_root: Path, commit: str, paths: Sequence[str]
) -> Optional[list[str]]:
    """Paths among *paths* that differ between *commit* and the working tree.

    ``git diff <commit>`` compares against the working tree, so committed and
    uncommitted changes both count: a binary built at *commit* is equally stale
    either way.
    """
    out = _git(repo_root, "diff", "--name-only", commit, "--", *paths)
    if out is None:
        return None
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def stale_binary_message(
    *,
    binary: Revision,
    head: Revision,
    changed: Optional[Sequence[str]],
) -> Optional[str]:
    """The staleness warning as a pure function — ``None`` when the binary matches.

    Split from :func:`stale_binary_warning` so every branch is testable without a
    git repo, a cargo target directory or a binary on disk.
    """
    where = f"was built at {binary.commit}" if binary.commit else "has no readable build commit"
    head_text = f"HEAD is {head.commit}" if head.commit else "HEAD is unknown"
    if binary.commit is None or changed is None:
        return (
            "the vanta_m3 binary's build commit could not be read (no cargo "
            "build-script output beside it, or no git to diff with), so this batch "
            f"cannot check whether the Rust sources moved on since it was built "
            f"({head_text}). Rows still carry the binary's own commit from "
            "run.summary; rebuilding with `cargo build --release --bin vanta_m3` "
            "restores the check."
        )
    if not changed and not binary.dirty:
        return None
    reasons = []
    if changed:
        shown = ", ".join(changed[:8]) + ("…" if len(changed) > 8 else "")
        reasons.append(f"{len(changed)} Rust source path(s) changed since: {shown}")
    if binary.dirty:
        reasons.append(
            "it was built from a dirty tree, so no commit describes the sources it "
            "was built from"
        )
    return (
        f"the vanta_m3 binary {where} but {head_text}, and "
        + "; ".join(reasons)
        + ". Every row of this batch measures that binary, not the tree at HEAD: "
        "rebuild (`cargo build --release --bin vanta_m3`) or record the batch as "
        f"measuring {binary.commit}."
    )


def stale_binary_warning(vanta_bin: Path, *, repo_root: Path = REPO_ROOT) -> Optional[str]:
    """§7.3's run-start check, read off disk. Printing is the caller's job."""
    binary = binary_revision(vanta_bin)
    head = git_revision(repo_root)
    changed = (
        changed_since(repo_root, binary.commit, VANTA_SOURCE_PATHS)
        if binary.commit
        else None
    )
    return stale_binary_message(binary=binary, head=head, changed=changed)


# ---------------------------------------------------------------------------
# config switches (§4.2) — one definition, shared with the record
# ---------------------------------------------------------------------------
def config_switches(config: str) -> tuple[bool, bool]:
    """``(attacker_enabled, planner_enabled)`` for a split config name.

    This is the inverse of ``benchmark_config_name`` in ``driver.rs``: the record
    the pipeline writes into ``run.summary`` names a config, and the manifest
    copies it. Deriving the *env* from the same switch pair is what keeps the two
    from being able to disagree.

    [VANTA-WHY] M3 item 11 — the pair is read from ``arms.py`` rather than from three
    ``if``s here, so a new arm is added in one place. ``vanta_planner`` keeps its own
    answer: §4.4's placeholder is addressable but is deliberately NOT a registered
    arm, because every run of it returns ``InfraFailure``.
    """
    if config == "vanta_planner":
        return True, True
    try:
        entry = arms_mod.arm_for_config(config)
    except arms_mod.ArmError as exc:
        raise RunnerError(
            f"unknown config {config!r}; expected one of {list(arms_mod.CONFIGS)}"
        ) from exc
    return entry.attacker, entry.planner


def vanta_env(
    *,
    arm: str,
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

    [VANTA-WHY] Every switch, the attempt budget's owner and the arm name come from
    ONE ``arms.py`` entry, so a run cannot be configured as one arm and recorded as
    another. That is the whole of "the harness refuses before launching": there is no
    combination here for it to get wrong.
    """
    try:
        entry = arms_mod.arm(arm)
    except arms_mod.ArmError as exc:
        raise RunnerError(str(exc)) from exc
    env = dict(os.environ if base is None else base)
    env["VANTA_SANDBOX_BACKEND"] = "shuru"
    env["VANTA_SHURU_CHECKPOINT"] = checkpoint
    env["VANTA_SHURU_DISK_MB"] = str(disk_mb)
    # See the module docstring: the self-heal needs network the VM does not have.
    env["VANTA_SHURU_AUTO_DEPS"] = "0"
    env["VANTA_ATTACKER"] = "on" if entry.attacker else "off"
    env["VANTA_PLANNER"] = "on" if entry.planner else "off"
    # [VANTA-WHY] M3 item 8 — the B2 switch, set from the same entry as the arm name
    # so the two cannot disagree. `on` is both the binary's own default and the
    # setting every existing measurement was taken under, so a legacy arm sends
    # exactly the value it sent before this key existed.
    env["VANTA_BATE"] = "on" if entry.bate else "off"
    # [VANTA-WHY] M3 item 11 — the arm name, STATED rather than inferred. `b0` and
    # `baseline` are the same switch pair and differ only in `--max-attempts`, so
    # `benchmark_config_name` cannot name them; the binary records this value
    # verbatim and refuses it if the switches or the budget contradict it.
    env["VANTA_ARM"] = entry.name
    # [VANTA-WHY] M3 item 7 — the judge switch, set from the SAME entry as the arm
    # name so the two cannot disagree: the binary refuses a stated `VANTA_ARM` whose
    # judge setting contradicts `VANTA_JUDGE`. `off` is the binary's own default and
    # the setting every arm before the judge had, so a non-judge arm sends exactly
    # the value it sent before this key existed.
    env["VANTA_JUDGE"] = entry.judge or "off"
    # Always set, including for seed 0: `VANTA_LLM_SEED=0` is a request to pin 0,
    # while an unset variable means "no seed key in the body" (§4.4, D4).
    env["VANTA_LLM_SEED"] = str(seed)
    # §14.15 — Vanta's own deadline, set short of the timeout this module kills
    # the process with. The margin is what makes the difference between the two
    # outcomes: inside it Vanta notices the clock, stops between calls and writes
    # `final_decision` + `run.summary`; past it the harness SIGKILLs mid-call and
    # the row is the `no run.summary` shape the overnight batch produced. Ten
    # minutes is one long sanic call, which is the unit the margin has to cover.
    env["VANTA_DEADLINE_SECS"] = str(RUN_TIMEOUT_SECS - DEADLINE_MARGIN_SECS)
    return env


def env_log_line(env: Mapping[str, str]) -> str:
    """The run log's ``# env:`` header — Vanta's own configuration, redacted.

    [VANTA-WHY] The header is built from :func:`bugsinpy.redact.redact_env`, not
    from a name-prefix exclusion. The previous form dropped ``VANTA_API_KEY`` by
    NAME, which made the rule "the one variable we remembered to exclude": a new
    credential under a new name (``OPENAI_API_KEY``, or a ``VANTA_``-prefixed
    alias somebody adds later) would have been printed in full. Applying the rule
    to the whole dict means any secret-named entry, and any ``sk-`` token inside
    any value, comes out ``<redacted>`` — including one we have not invented yet.

    The ``VANTA_`` prefix filter stays on top, because the line is meant to
    describe *this harness's* configuration rather than dump the environment.

    [VANTA-WHY] Redaction applies to the RECORD, never to the call. The env
    handed to ``subprocess.run`` is the real one: the key has to survive to the
    child, or the run cannot authenticate.
    """
    safe = redact_mod.redact_env(env)
    return "# env: " + " ".join(
        f"{k}={v}" for k, v in sorted(safe.items()) if k.startswith("VANTA_")
    )


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
        hidden_nodeids=prep.hidden_nodeids,
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


# ---------------------------------------------------------------------------
# localization_hit — the paper metric, computed AFTER a run (§1.6.2 item 5)
# ---------------------------------------------------------------------------
def selected_paths(context_selection: str) -> list[str]:
    """The paths in a provenance line's ``selected=[…]`` list.

    The line is ``context::ContextSelection::provenance_line``: one space-free
    ``key=value`` blob whose ``selected`` list holds
    ``path@score[reasons]shown:bytesB/tok`` entries joined by ``;`` (and the
    literal ``none`` when nothing was shown). The path is everything before the
    first ``@``, and the list ends at ``:pruned=[``: taking the first ``]``
    instead would cut the entry off inside its own ``[reasons]`` brackets and
    silently report a shorter name than the one the run was shown. Returns ``[]``
    for ``none`` and for a line that carries no ``selected`` list.
    """
    if not context_selection or context_selection == "none":
        return []
    head = context_selection.partition(":pruned=[")[0]
    if "selected=[" not in head:
        return []
    body = head.partition("selected=[")[2]
    # The list is ``selected=[…]``: its closing ``]`` is still on ``body``. Without
    # dropping it the literal-empty form ``selected=[none]`` would come back as the
    # path ``none]`` — a path no tree has, counted as a localization miss forever.
    if body.endswith("]"):
        body = body[:-1]
    if body in ("", "none"):
        return []
    return [entry.split("@", 1)[0] for entry in body.split(";") if entry]


def fix_touched_paths(prep: prepare_mod.BugPrep) -> list[str]:
    """The source files the developer's fix touches (§1.6.2 item 5).

    The fix patch's **test** hunks are dropped first: BugsInPy's ``bug_patch.txt``
    usually updates the test file too, and a test path is never a file the run's
    source selection could show — the hidden tests are stripped from ``vanta_tree``
    before any run sees it. Counting a test path as a miss would report every bug
    as unlocalized, which is the opposite of a metric.
    """
    return fix_touched_paths_from_patch(prep.bug.bug_patch_txt)


def fix_touched_paths_from_patch(bug_patch_txt: str) -> list[str]:
    """The same question, asked of the patch text alone.

    Split out of :func:`fix_touched_paths` for the **offline** ranking audit
    (:func:`localization_table`, §1.6.5): that audit has the fix-commit patch and
    nothing else, and building a :class:`BugPrep` just to hand its own patch text
    back would be a fabricated record. One implementation, so the run-time metric
    and the audit cannot come to differ about what "touched" means.
    """
    parsed = tree_mod.parse_patch(suite_mod.drop_test_file_hunks(bug_patch_txt))
    return sorted(set(parsed.paths.changed) | set(parsed.paths.added))


def paths_hit(touched: Iterable[str], shown: Iterable[str]) -> Optional[bool]:
    """Whether any of ``touched`` is among ``shown``; ``None`` when nothing was touched.

    ``None`` — never ``False`` — for an empty ``touched`` set: a fix that changes
    only test files has no source path the selection could have shown, and pooling
    that as a miss would report a miss rate over rows that never had a chance.
    """
    touched = list(touched)
    if not touched:
        return None
    shown = set(shown)
    return any(path in shown for path in touched)


def shown_full(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    """The selected paths the run was shown **in full** — not outlined, not as a path.

    §1.6.11's narrowing of the audit's question. §1.6.5 asked whether a touched file
    was *among those shown*; shown is three forms, and a path the prompt prints as a
    bare tree entry is not a file the model can read. The audit now asks the stricter
    question — was the file the fix changed handed over whole? — because a hit on an
    outline or a path row says the prompt carried a *pointer* to the fix, which is a
    different finding from carrying the fix.
    """
    return [
        str(file["path"])
        for file in entries
        if file.get("path") and str(file.get("shown") or "") == "full"
    ]


def touched_rank_entry(path: str, entries: Iterable[Mapping[str, Any]]) -> str:
    """``path #rank (form)`` — where the selection put a file the fix changed (§1.6.11).

    Read three-valued, the way :func:`context.rank_of` reads the field, because the
    three states are three different findings and only one of them is a ranking
    failure:

    * **in the selection, rank recorded** — ``#3 (outline)``: the ranker put the
      fix's file third and showed it as an outline;
    * **in the selection, no rank recorded** — ``#?``: a report written before §1.6.9
      recorded ranks. The tolerant mark, never an invented number pulled from the
      array's order;
    * **not in the selection** — ``#—``: the ranking reached nothing for the file the
      fix changed.

    The form beside it is the same three forms the fill ranks in (§1.6.1), so a row
    where the fix's file was ranked but only outlined reads as that and not as a hit.
    """
    for file in entries:
        if str(file.get("path") or "") != path:
            continue
        rank = ctx_mod.rank_of(file)
        place = str(rank) if rank else "?"
        return f"{path} #{place} ({str(file.get('shown') or '?')})"
    return f"{path} #—"


def refusal_name(reasons: object) -> str:
    """The gate's own name for a row that was not a clean selection (§1.6.10), or ``—``.

    ``prepare`` writes each context row's ``reasons`` as ``name: what it means`` — the
    name is the vocabulary §1.6.10 fixed (``no_issue_grounded_selection``,
    ``selection_dropped_for_budget``, ``issue_source=none``) and the prose after the
    colon is the sentence a reader is shown elsewhere. The audit prints the name beside
    the row so a refusal does not read as a ranking miss: ``matplotlib/1``'s ``no`` is a
    selection the budget refused, and ``fastapi/1``'s ``no`` is a selection that ranked
    nothing the fix's files touch — the same cell, two different findings.

    The name is the token before the first colon, and only when that token is
    identifier-shaped (letters, digits, ``_``, ``=``). A reason that is a sentence — a
    measure-only run's ``frozen issue report: …`` — has no name and prints ``—``, the
    honest answer rather than a first word pretending to be one.
    """
    if isinstance(reasons, (str, bytes)) or not isinstance(reasons, Sequence):
        return "—"
    for reason in reasons:
        if not isinstance(reason, str) or ":" not in reason:
            continue
        name = reason.split(":", 1)[0].strip()
        if name and re.fullmatch(r"[a-z][a-z0-9_=]*", name):
            return name
    return "—"


def localization_hit(
    prep: prepare_mod.BugPrep, context_selection: Optional[str]
) -> Optional[bool]:
    """Whether a file the fix touched was among those the run was shown.

    ``None`` — not ``False`` — when the question cannot be answered: a summary with
    no ``context_selection`` (a batch from before §1.6.2 item 4), or a fix whose
    patch touches only test files (so nothing could have been shown). "No evidence"
    and "the selection missed" are two different findings and pooling them as
    ``False`` would report a miss rate over rows that never had a chance.
    """
    if context_selection is None:
        return None
    return paths_hit(fix_touched_paths(prep), selected_paths(context_selection))


CONTEXT_REPORT_NAME = "context_report.json"


def localization_table(
    *,
    batch_dir: Path,
    split: Split,
    bugsinpy_root: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Ranking quality per dev bug, **from the recorded selection** (§1.6.5).

    ``localization_hit`` as the paper metric is computed per run, from the
    ``context_selection=`` provenance the driver wrote. This is the same question
    asked one stage earlier and offline: read the selection ``prepare`` already
    measured (`context_report.json`), read each bug's fix-commit patch from
    BugsInPy, and report whether the fix's own files were among those shown. It
    **never runs the selector**, and nothing in ``prepare`` reads its answer — the
    audit is a view of evidence, so it cannot become a term the ranking optimises.

    The split is the **dev** split by construction: ranking quality is a tuning
    number, and a tuning number produced over the evaluation split is how a
    held-out set stops being held out. ``cmd_localize`` refuses any other.

    A bug whose report row is missing, or says ``not_measured``, is reported as
    ``None`` (printed ``—``) rather than as a miss: no selection was made, so
    there is nothing to judge.

    §1.6.11 widens the row: not *was the fix's file among those shown* but **was it
    shown in full**, and each touched file is printed with the **rank the selection
    gave it**. A path that reached the prompt as a tree entry, or as an outline, is a
    pointer to the fix rather than the fix, and a hit rate that counted it would
    report the ranking as better than the prompt the model read.
    """
    report = _read_json(batch_dir / CONTEXT_REPORT_NAME)
    if not report:
        raise RunnerError(
            f"no {CONTEXT_REPORT_NAME} in {batch_dir}. The audit reads the selection "
            "a prepare already measured, so run `bugsinpy_run.py prepare "
            "--measure-only --batch <dir>` (or a full prepare) first; it does not "
            "run the selector itself."
        )
    measured = {str(row.get("bug")): row for row in report.get("bugs", [])}
    rows: list[dict[str, Any]] = []
    for entry in split.bugs:
        key = _bug_key(entry)
        record = measured.get(key)
        # The fix patch through the harness's own reader (``load_bug`` →
        # ``bugsinpy_meta.load_bug_info``, the one that reads ``bug_patch.txt``), so a
        # moved clone fails here the way it fails in prepare instead of making the
        # audit quietly report every bug as a miss.
        bug = prepare_mod.load_bug(
            split, bugsinpy_root, str(entry["project"]), int(entry["bug_id"])
        )
        touched = fix_touched_paths_from_patch(bug.bug_patch_txt)
        entries = [
            file
            for file in (record or {}).get("selected_files", [])
            if file.get("path")
        ]
        # §1.6.11: a hit is a touched file shown **in full**, and the selection's full
        # set is the narrower one the hit is asked about. The rank column still prints
        # every touched file, so an outline or a path row is visible as a near miss.
        full = shown_full(entries)
        was_measured = bool(record) and record.get("verdict") != "not_measured"
        rows.append(
            {
                "bug": key,
                "selected": len(entries),
                "in_full": len(full),
                "top": [selection_entry(file) for file in entries[:3]],
                "touched": touched,
                "touched_ranks": [touched_rank_entry(path, entries) for path in touched],
                "hit": paths_hit(touched, full) if was_measured else None,
                # §1.6.10: the gate's own name for why the row is not a clean selection,
                # printed beside the hit so a refusal is not read as a ranking miss.
                "refused": refusal_name((record or {}).get("reasons")),
            }
        )
    return render_localization_table(rows), rows


def selection_entry(file: Mapping[str, Any]) -> str:
    """``path[reason:rule+reason:rule]`` — a selected file, and why it ranked (§1.6.4).

    The reason alone says which input named the file; the rule beside it says which
    of that signal's rules matched, and it is the half a reader needs here: `path`
    covers five of them, and a file the issue names outright is a different
    finding from one that matched a bare basename in another directory. A signal
    whose rule was not recorded — a ``context_report.json`` written before the
    selector recorded them — prints as the reason alone, never as a placeholder.
    """
    path = str(file.get("path") or "")
    rules = file.get("rules")
    if not isinstance(rules, Mapping):
        rules = {}
    parts = [
        f"{reason}:{rules[reason]}"
        if isinstance(rules.get(reason), str) and rules[reason]
        else str(reason)
        for reason in (file.get("reasons") or [])
    ]
    return f"{path}[{'+'.join(parts)}]" if parts else path


def render_localization_table(rows: list[dict[str, Any]]) -> str:
    """The six columns the audit is read for, aligned to their own widest cell.

    ``top 3 files`` carries each file's reasons *and* the rule that matched each
    one, so a row says why those three ranked and not only that they did. ``files in
    full`` is written ``in full / selected`` — the count the hit is asked over, beside
    the count the ranking chose, because a run that selected thirty files and showed
    two of them whole is a different finding from one that selected two. ``touched
    (rank)`` prints every file the fix changed with the place the selection gave it
    (§1.6.11), so a fix file that ranked 7th and was outlined is visible as the near
    miss it is. ``refused`` prints the gate's own name for a row that was not a clean
    selection (§1.6.10), so ``no`` on a budget refusal is not read as ``no`` on a
    ranking that reached nothing. The audit's question is whether the ranking put the
    fix's own files first, and a list of paths that agrees with the numbers while
    disagreeing with the recipe is exactly what a reader has to be able to see.

    Column widths come from the data rather than a constant: a path can be longer
    than any guessed width, and a table whose ``hit`` column has drifted a space to
    the right on the rows that matter is read as noise.
    """
    headers = ("bug", "files in full", "top 3 files", "touched (rank)", "hit", "refused")
    hits = {True: "yes", False: "no", None: "—"}
    body = [
        {
            "bug": str(row["bug"]),
            "files in full": f"{row['in_full']}/{row['selected']}",
            "top 3 files": ", ".join(row["top"]) or "—",
            "touched (rank)": ", ".join(row["touched_ranks"]) or "—",
            "hit": hits[row["hit"]],
            "refused": str(row["refused"]),
        }
        for row in rows
    ]
    widths = {head: max([len(head)] + [len(row[head]) for row in body]) for head in headers}

    def line(cells: dict[str, str]) -> str:
        return "  ".join(cells[head].ljust(widths[head]) for head in headers).rstrip()

    head = line({h: h for h in headers})
    lines = [head, "-" * len(head)]
    lines += [line(row) for row in body]
    tally = {verdict: sum(1 for r in rows if r["hit"] is verdict) for verdict in hits}
    lines += [
        "",
        f"localization_hit: hit {tally[True]} / miss {tally[False]} / unknown "
        f"{tally[None]} over {len(rows)} dev bug(s) — a hit is a file the fix touched "
        "shown **in full** (not merely selected, nor outlined), computed offline from "
        "the selection already recorded and never fed back into it (§1.6.11)",
    ]
    return "\n".join(lines)


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
        # §14.16 — the presence penalty is the one sampler key that can come from
        # an ENV OVERRIDE rather than a preset, so it is read back per site the
        # same way. It is present only when a non-zero penalty was pinned; a zero
        # and an unset penalty send byte-identical bodies.
        elif key.startswith("presence_penalty_"):
            sites.setdefault(key[len("presence_penalty_"):], {})["presence_penalty"] = _number(value)
    top_p = summary.get("top_p")
    # §14.16 — what `VANTA_PRESENCE_PENALTY` ASKED FOR, which `pinned` above can
    # only show when it is non-zero: `0.0` and `unset` send the same bodies, so a
    # table comparing the two arms of the ablation can only be cut from this
    # field. `unset` means the switch was never set.
    penalty = summary.get("presence_penalty")
    # §14.20 — what `VANTA_MAX_TOKENS_<SITE>` ASKED FOR, keyed by site. The
    # per-site `max_tokens` values inside `pinned` above are the EFFECTIVE cap
    # every body was sent under (the override already reached them); this is the
    # switch itself, which is what identifies an ablation arm in a batch table.
    # `unset` means the variable was never set, which is every run before §14.20.
    overrides_text = summary.get("max_tokens_overrides")
    overrides: Optional[dict[str, int]] = None
    if overrides_text and overrides_text != "unset":
        overrides = {
            site: int(cap)
            for site, _, cap in (
                pair.partition("=") for pair in overrides_text.split(",")
            )
            if site and cap
        }
    # §14.21 — the reasoning controls, at the same two levels §14.20 uses:
    # `budget`/`effort` are what the run ASKED FOR (the switch, which identifies
    # an ablation arm), and the `*_mechanism` fields name the transport that
    # implemented each one. The mechanism is what makes the same arm comparable
    # across backends — a hosted run records its own. `effort` here is the
    # EFFECTIVE level and defaults to `low`, because the six call sites have sent
    # `reasoning_effort: low` since v0.2; `None` means the record carried no such
    # key at all (a pre-§14.21 summary). `unset` from the pipeline reads back as
    # `None`, so a reader cannot mistake "no switch" for "a switch with no value".
    budget_text = summary.get("reasoning_budget")
    budget_mechanism = summary.get("reasoning_budget_mechanism")
    return {
        # The split's own constant, not a copied literal: the manifest must name the
        # same policy the committed design declares, or the provenance block would
        # describe a design nobody reviewed.
        "source": SAMPLING_POLICY,
        "pinned": [{"site": site, **values} for site, values in sites.items()],
        "presence_penalty": (
            _number(penalty) if penalty and penalty != "unset" else None
        ),
        "max_tokens_overrides": overrides,
        "reasoning": {
            "budget": (
                int(budget_text) if budget_text and budget_text != "unset" else None
            ),
            "budget_mechanism": (
                budget_mechanism if budget_mechanism and budget_mechanism != "unset" else None
            ),
            "effort": summary.get("reasoning_effort"),
            "effort_mechanism": summary.get("reasoning_effort_mechanism"),
        },
        "server_top_p": _number(top_p) if top_p and top_p != "unset" else None,
        "server_top_p_note": (
            "unset — /props did not publish it"
            if not top_p or top_p == "unset"
            else "read from the server's /props"
        ),
    }


def llm_block(summary: dict[str, str]) -> dict[str, Any]:
    """§14.12's per-run LLM cost block, read back from the pipeline's record.

    The pipeline counts the calls it made while it makes them (one ``call.usage``
    row per chat call); the harness only copies the totals. That is what makes a
    batch table and a session log unable to disagree about what a run cost — the
    alternative, recomputing from the log text, is what the runaway diagnosis
    could not do: the plain log carried neither a completion count nor a
    ``finish_reason``.

    ``by_site`` is those same three facts at CALL-SITE granularity. It exists
    beside ``truncated_calls_by_role`` because the Attacker's three sites are one
    role: "the revisions all ran away" and "the first generation ran away" are
    the same number per role and a different problem in fact.

    ``truncated_twice`` is the subset of the truncations that were cut off BOTH
    times the call asked (§14.5 item 2): the request, and the short ``answer
    directly`` re-ask appended to it. It cannot be derived from
    ``truncated_calls`` — two came back where one call gave up — and it is the
    number that says "this site cannot answer inside its cap", which is a
    different repair from "the model ran away once".

    ``revise_budget_per_attempt`` / ``revise_calls_used`` are §14.14's ceiling on
    the Attacker's remediation calls (revise in BATE plus simplify in the
    Diagnostician) and the run's total spend against it. They sit here because
    they are the other half of "is the Attacker stuck?": a truncation says one
    call could not finish, the budget says the round kept asking. The cap is a
    property of the design and is recorded per run so a manifest read years later
    does not have to trust a constant that has since moved.

    ``cached_tokens`` / ``reasoning_tokens`` / ``cost_usd`` / ``cost_basis`` are
    §14.29's usage facts that are not token counts, and the money. The pipeline
    prices each call from its own versioned table (``llm::pricing``) and writes
    the basis it used; the harness copies the number rather than re-pricing,
    because a second price table here could disagree with the first and there
    would be no way to tell which one a batch table used. ``cost_usd`` is a
    float or ``None``: ``None`` is "nothing was priced" — a local run, which is
    ``not_billed`` — and is NOT the same claim as ``0.0``. ``cost_basis`` names
    the rows that produced the figure (``gpt-6-luna/standard/2026-10-26``), so
    two rows priced from different dates are visible as such instead of being
    pooled silently.

    ``provider_retries`` / ``provider_retries_rate_limit`` /
    ``provider_retries_unavailable`` are §14.30's count of the times the
    PROVIDER refused a request and it was re-sent, split into the two classes it
    distinguishes (a rate-limit window rolling over, and ``flex`` capacity being
    unavailable). They are copied from the run-level counters, not summed over
    the ``call.usage`` rows, and that is deliberate: a call the provider kept
    refusing past its retry budget never answered and therefore writes no row,
    so a sum over the rows would under-report the runs that were throttled
    hardest. Beside ``truncated_calls`` because the two are the model's and the
    service's answers to the same question — "why was this call repeated?" —
    and a run that mixed them should not have to be inferred from a log.

    ``service_tier`` is §14.31's arm: the tier the run was served on, which is
    also the half of the price-table row name that ``cost_basis`` carries
    (``gpt-6-luna/flex/2026-10-26``). Copied from the run-level summary rather
    than parsed back out of ``cost_basis`` because the two are different claims
    — the basis is the row that priced the run, the tier is what the run asked
    for — and a batch grouped by arm should not have to split a formatted
    string to find it.

    ``cache_hit_rate`` is §14.32's share of the prompt the provider served from
    its own cache, copied from ``run.summary``'s ``llm_cache_hit_rate`` (the
    pipeline computes it from the two token totals it already records, so the
    three numbers on that line agree by construction). It is a float or ``None``:
    ``None`` is "no rate exists" — no call was measured, or no measured call
    reported a cache read — and is NOT the same claim as ``0.0``, which is "the
    cache was measured and served nothing". The row ALSO carries the raw
    ``cached_tokens``, so a rate computed over a subset of calls can be checked
    against the counts it came from.
    """
    by_site: dict[str, dict[str, Any]] = {}
    for key, value in summary.items():
        for prefix, field in (
            ("llm_calls_site_", "calls"),
            ("truncated_calls_site_", "truncated_calls"),
            ("completion_tokens_site_", "completion_tokens"),
        ):
            if key.startswith(prefix):
                by_site.setdefault(key[len(prefix):], {})[field] = _int(value)
    return {
        "calls": _int(summary.get("llm_calls")),
        "calls_measured": _int(summary.get("llm_calls_measured")),
        "prompt_tokens": _int(summary.get("llm_prompt_tokens")),
        "completion_tokens_total": _int(summary.get("completion_tokens_total")),
        # §14.29 — the money and the two non-token usage counts.
        "cached_tokens": _int(summary.get("llm_cached_tokens")),
        "reasoning_tokens": _int(summary.get("llm_reasoning_tokens")),
        # §14.32 — the share of the prompt the server served from its own cache,
        # copied from the run summary (the pipeline derives it from the two
        # totals on the same line, so a reader dividing those two gets this
        # value). `None` means no rate exists — nothing measured, or nothing
        # reported a cache read — and is NOT the same claim as `0.0`.
        "cache_hit_rate": _rate(summary.get("llm_cache_hit_rate")),
        "cost_usd": _cost(summary.get("cost_usd")),
        "cost_basis": summary.get("cost_basis"),
        # §14.31 — the tier the run was served on, beside the cost it selected.
        # Copied from the run-level summary (`standard`/`flex` on the hosted
        # backend, `unset` locally), not derived from `cost_basis`: the basis is
        # a formatted row NAME and this is the arm the run asked for, and a
        # reader grouping a batch by arm should not have to parse a string.
        "service_tier": summary.get("service_tier"),
        "truncated_calls": _int(summary.get("truncated_calls")),
        "truncated_twice": _int(summary.get("truncated_twice_calls")),
        # §14.30 — the service's refusals, split by class, copied verbatim from
        # the run-level counters (see the docstring: a never-answered call writes
        # no row, so these cannot be summed from `by_site`).
        "provider_retries": _int(summary.get("provider_retries")),
        "provider_retries_rate_limit": _int(summary.get("provider_retries_rate_limit")),
        "provider_retries_unavailable": _int(summary.get("provider_retries_unavailable")),
        # §14.5's three reportable roles, spelled out rather than derived from
        # the site map: the mapping site→role is the pipeline's
        # (`llm::truncation_role`), and a second copy of it here could drift.
        "truncated_calls_by_role": {
            "creator": _int(summary.get("truncated_calls_creator")),
            "attacker": _int(summary.get("truncated_calls_attacker")),
            "intent_audit": _int(summary.get("truncated_calls_intent_audit")),
        },
        "revise_budget_per_attempt": _int(summary.get("revise_budget_per_attempt")),
        "revise_calls_used": _int(summary.get("revise_calls_used")),
        "by_site": by_site,
    }


def _number(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() and "." not in value else parsed


def _cost(value: Optional[str]) -> Optional[float]:
    """§14.29 — a cost is a float or ``None``, never the string ``unset``.

    ``_number`` keeps a non-numeric token verbatim so a reader can see what the
    pipeline actually wrote; a field a batch sums cannot do that. The pipeline
    writes ``unset`` when nothing was priced and ``cost_basis`` says why
    (``not_billed``, ``no_usage``, ``unpriced_family``), so mapping every
    non-number to ``None`` loses nothing a reader cannot get from the basis, and
    keeps ``None`` (= nothing priced) distinct from ``0.0`` (= priced, and free).
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rate(value: Optional[str]) -> Optional[float]:
    """§14.32 — a share is a float or ``None``, never the string ``unset``.

    Same contract as ``_cost`` and for the same reason: a field a batch averages
    or thresholds cannot carry a non-numeric token. The pipeline writes ``unset``
    when no rate exists at all — nothing was measured, or nothing reported how
    much came from cache — and ``0.0000`` when the server measured a cache and
    served nothing from it. Collapsing both to ``None`` would erase that
    difference, so the pipeline keeps it in the text and this only decides what a
    reader can compute with.
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
    # §3.4 — the row's recorded suite environment, handed to every run of this bug
    # so Vanta's own suites see the same import path the preparation gates did.
    # One flag per key, the value never typed per bug: it comes from the split's
    # ``sandbox_env`` rule, and the binary refuses any key outside its allow-list.
    for key, value in sorted(prep.sandbox_env.items()):
        argv += ["--sandbox-env", f"{key}={value}"]
    if candidate is not None:
        argv += ["--patch", str(candidate.patch_path)]

    before = _session_stems()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    exit_code: Optional[int] = None
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n")
        log.write(env_log_line(env) + "\n\n")
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

    [VANTA-WHY] M3 item 7 — the judge arm's readable verdicts are scored EXACTLY as
    the test-driven arm's, so a judge row and the ``vanta`` row for the same candidate
    pair by label: ``JudgeApproved`` is an approval and lands on
    ``true_approval``/``false_approval`` like ``AllTestsPassed``; ``JudgeRejected`` is a
    rejection and lands on ``false_reject``/``true_reject`` like ``Rejected``.

    [VANTA-WHY] M3 item 7 — ``JudgeFailed`` (the judge answered and its answer was
    unreadable after the one retry) is its OWN label, checked BEFORE the ``apply_ok``
    branch: no patch is attached to that row, so routing it through ``apply_ok`` would
    report the censor's ``unobservable`` and hide the judge failure. It is deliberately
    NOT ``infra_failure``: the machine was fine and the call succeeded.
    """
    if verdict == "InfraFailure":
        return "infra_failure"
    if verdict == "JudgeFailed":
        return "judge_failed"
    if not oracle.apply_ok:
        return "unobservable"
    if verdict in ("AllTestsPassed", "JudgeApproved"):
        return "true_approval" if oracle.patch_good else "false_approval"
    if verdict in ("Rejected", "JudgeRejected"):
        return "false_reject" if oracle.patch_good else "true_reject"
    if verdict == "NeedsReview":
        # [WHY] §14.6: a refusal to decide is scored as itself, never as one of
        # the two decisions it declined to make. `patch_good` here would name the
        # row a `false_reject` (a rejection Vanta did not issue) and the negation
        # a `true_reject` (credit for a rejection it did not issue either). Both
        # readings would put a number the run never claimed into the wave's
        # rejection rate, so the label carries the ground truth it was refused
        # against AND the fact that the loop could not decide.
        return "needs_review_on_good" if oracle.patch_good else "needs_review_on_bad"
    if verdict == "MaxAttemptsReached":
        return "unresolved_but_fixed" if oracle.patch_good else "unresolved"
    return "unobservable"


def build_record(
    *,
    job: "Job",
    prep: prepare_mod.BugPrep,
    outcome: SessionOutcome,
    oracle: oracle_mod.OracleResult,
    issue_sha256: str,
    prepare_s: float,
    oracle_secs: float,
    oracle_reused: bool,
    python_version: str,
    oracle_json: Optional[Path],
    compaction: "compact_mod.Compaction",
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
        # [VANTA-WHY] M3 item 8 follow-up — the arm's canonical name, next to the
        # config it is part of the key of. `config` keeps its legacy value so every
        # job key and every existing script is untouched; `arm` is what analysis
        # groups by, and what a reader of one row needs to know which experimental
        # condition produced it (`b1` and `suite` are both `config=baseline`).
        "arm": job.arm,
        "mode": job.mode,
        "seed": job.seed,
        # §14.23 — how the pinned seed was turned into a per-round seed. Recorded
        # because the offset is DEFAULT-ON: a row's own `seed` cannot say whether its
        # later review rounds were independent draws or deterministic duplicates, so
        # a row that predates §14.23 (no `seed_policy` at all) must never be pooled
        # with one from after it. The pipeline's own spelling is kept.
        "seed_policy": summary.get("seed_policy"),
        # §2.4.4 — whether this row's prompt carried the report's comments as well as its
        # content. Recorded rather than inferred because the ablation's whole value is that
        # its rows are comparable to the default ones, and a row that does not say which
        # prompt it sent is a row nobody can put in the right arm afterwards.
        "show_hints": job.show_hints,
        "session_id": outcome.session_id,
        # Two trees, two commits (§7.3). `vanta_build_commit` is what `build.rs` baked
        # into the binary at cargo build time and what `run.summary` reports;
        # `harness_commit` is this repo's HEAD when the batch started — a different
        # commit whenever `scripts/` moved on after the last `cargo build`, which is
        # the normal case and not a defect. Recording one under the other's name is
        # what made a digest whose HEAD was b2efa49 claim 44f58ab.
        "harness_commit": job.harness.commit if job.harness else None,
        "harness_dirty": job.harness.dirty if job.harness else None,
        "vanta_build_commit": summary.get("commit"),
        "vanta_build_dirty": _bool(summary.get("dirty")),
        "model": summary.get("model"),
        "sampling": sampling_block(summary),
        # §14.22 — whether this row's ATTACKER prompt carried the extra
        # "one check is enough" sentence. Recorded for `show_hints`'s reason: the
        # attribution of an ablation row depends on knowing which prompt it sent,
        # and a row that does not say cannot be put in the right arm afterwards.
        # The pipeline's own spelling is kept (`on`/`off`, §14.3), so the manifest
        # and the command that produced the run can be read against each other.
        "attacker_concise": summary.get("attacker_concise"),
        # §14.12 — what the run's LLM calls actually cost, and how often the
        # server cut one off at its output cap. Read back from `run.summary`, so a
        # batch table can be cut by call site without opening a session log — and
        # so a runaway is a number the manifest carries rather than a surprise
        # found in a 2-hour log afterwards.
        "llm": llm_block(summary),
        "checkpoint": job.checkpoint,
        # `python_version` is what the checkpoint ACTUALLY contains; the pinned one
        # is what BugsInPy's bug.info declares. §12.3 says they differ (3.8.20 vs
        # 3.8.1), and both are recorded so the gap is auditable rather than hidden
        # by writing the actual number into a field that claims to be the pin.
        "python_version": python_version or "unknown",
        "python_version_pinned": job.python_version_pinned,
        # The pins the layer ACTUALLY carries, read back from its provenance rather
        # than from the split's entry (§3.2): an entry declares the *interpreter*
        # layer's pair, one pair per Python minor, while a bug whose own files pin
        # pytest is installed with that version on top of it. A bug with no pin of its
        # own carries the interpreter layer's pair, which its provenance records under
        # `harness_pins` — not the split's number by convention but the one this bug
        # was resolved against. `None` only when there is no provenance to read.
        "pytest_version": (
            bake_mod.layer_pin(prep.deps, "pytest") if prep.deps else None
        ),
        "json_report_version": (
            bake_mod.layer_pin(prep.deps, "pytest-json-report") if prep.deps else None
        ),
        "candidate": job.candidate.as_dict() if job.candidate else None,
        "verdict": outcome.decision or None,
        # §14.19 — which generated test's BATE baseline run died before pytest
        # graded it, when the environment rather than the test was broken. A run
        # that ends this way writes `verdict=HarnessFailure` and no patch, so its
        # label is `unobservable`; this field is what says WHY, and turns the class
        # into one row a batch table can count instead of a long trail of
        # revisions. `unset` on every ordinary run.
        "harness_error": summary.get("harness_error"),
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        # §14.15 — why a run that stopped itself stopped. `budget_exhausted` is
        # Vanta's OWN deadline firing, which ends the run as `NeedsReview` (exit 3)
        # with `run.summary` and the reason on disk. It is deliberately a different
        # field from `timed_out`, which is the harness killing the process: that
        # row usually has no `run.summary` to read at all, and the two must never
        # be pooled as one "stopped" outcome because they say opposite things —
        # one is a verdict Vanta reached, the other is a verdict it never reached.
        "stop_reason": summary.get("stop_reason"),
        "deadline_secs": _int(summary.get("deadline_secs")),
        "attempts_run": _int(summary.get("attempts_run")),
        "fixed_at_attempt_1": _bool(summary.get("fixed_at_attempt_1")),
        # §1.6.2 item 5 — whether the run's own selection included a file the fix
        # touched, computed here, AFTER the run and never during it: the selector
        # must not be able to see the answer, so the answer is only ever compared
        # against the selection afterwards. `null` means "no evidence", not "no".
        "localization_hit": localization_hit(prep, summary.get("context_selection")),
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
            "graded_tree_diff": compaction.diff,
            "issue_sha256": issue_sha256,
            "compact_state": compaction.state,
            "compact_reason": compaction.reason,
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
    arm: str
    """The canonical arm name this job runs (§4.2, ``arms.py``).

    [VANTA-WHY] M3 item 8 follow-up — ``config`` is the split's own column and stays
    ``baseline``/``vanta`` for the legacy arms so a job key on disk still resumes;
    ``arm`` is the arm's canonical name (``b1``/``suite``/``b3``/``b0``/``b2``), which
    is what analysis groups by. The two are not the same field for that reason: the
    key may not change, the arm must be nameable. It is ALSO what the harness passes
    as ``VANTA_ARM``, so the row's arm and the arm the binary ran are one value read
    from one place (``arms.py``).
    """
    mode: str
    seed: int
    checkpoint: str
    python_version_pinned: str
    candidate: Optional[Candidate] = None
    harness: Optional[Revision] = None
    """The harness's own revision, captured once when the batch started (§7.3).
    Carried on the job so the row can state it without a second `git` call per
    row, which would let two rows of one batch disagree about which harness ran
    them."""
    show_hints: bool = False
    """Send the report's comments as well as its content — **off by default** (§2.4.4).

    [VANTA-WHY] The comments are SWE-bench's ``hints_text`` and they name the answer often
    enough that ``matplotlib/1``'s maintainer comment names a file and a line range. They
    are gated and stored either way (``issue_source``), so this flag chooses only whether the
    prompt carries them; it defaults off because the default setting has to be the one a
    report that predates the fix supports, and a run that turns it on records that it did
    (``build_record``) and takes its own job key (``manifest.job_key``), so the ablation
    cannot be mistaken for the main result or resumed under its name.
    """

    @property
    def key(self) -> str:
        return job_key(
            batch_id=self.batch_id,
            bug=self.bug,
            config=self.config,
            mode=self.mode,
            seed=self.seed,
            candidate=self.candidate.patch_sha256 if self.candidate else None,
            show_hints=self.show_hints,
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


def issue_text_for(prep: prepare_mod.BugPrep, *, show_hints: bool) -> str:
    """The exact text written to ``issues/<slug>.txt``: the content, and the hints if asked.

    [VANTA-WHY] With ``show_hints`` off — the default, and every row of the main evaluation —
    the answer is ``prep.issue.text`` and nothing else: that string *is* the frozen artifact's
    gated content section, and a run has to send it byte-for-byte (§2.4.3). The comments are
    read here, from the same committed file whose hash a stage verified, rather than carried
    through ``prepare.json``: under the default setting there is then no copy of them anywhere
    on the path to a prompt, so "never shown" is a property of the code rather than of a flag
    somebody has to keep set. A report with no comments, or a hints section the gate emptied,
    leaves the text as it was — turning the ablation on cannot invent hints that do not exist.
    """
    text = prep.issue.text if prep.issue else ""
    if not show_hints:
        return text
    frozen = issue_source.load_or_none(
        issue_source.ISSUES_DIR, prep.bug.project, prep.bug.bug_id
    )
    if frozen is None or not frozen.hints:
        return text
    return f"{text}\n{frozen.hints}"


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
    issue_path.write_text(issue_text_for(prep, show_hints=job.show_hints), encoding="utf-8")
    issue_sha256 = sha256_text(issue_path.read_text(encoding="utf-8"))

    log_path = paths.logs / f"{_stem(job.key)}.log"
    env = vanta_env(
        arm=job.arm,
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

    # BEFORE the row is written, not after: `artifacts.compact_state` and
    # `graded_tree_diff` are the row's own statement about where the snapshots
    # went, and a row that said `full` about a tree compaction had just deleted
    # would misdescribe the session the manifest is the evidence for (§6.3, §7.3).
    compaction = compact_mod.compact_session(
        session_dir=sessions_dir() / outcome.session_id,
        base_tree=prep.root / "vanta_clean",
        verdict=outcome.decision,
        session_id=outcome.session_id,
        sessions_dir=sessions_dir(),
        base_commit=prep.bug.buggy_commit,
    )

    record = build_record(
        job=job,
        prep=prep,
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
        compaction=compaction,
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
        # The arm, for the same reason the row carries it: a failure record has to be
        # attributable to the experimental condition that produced it.
        "arm": job.arm,
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


def _spent_so_far(manifest: Manifest) -> float:
    """§14.29 — the batch's cost to date, summed from the rows already on disk.

    Read from the manifest rather than carried in a local counter, because the
    cap is a property of the *batch*: a resumed batch must not restart the
    count, and the rows that spent the money are on disk, not in this process's
    memory. A row that priced nothing contributes ``0.0`` — that is the local
    backend, and every run before this change.
    """
    return sum(_row_cost(record) for record in manifest.records())


def _row_cost(record: dict[str, Any]) -> float:
    """One row's ``llm.cost_usd``, ``None`` read as zero (see :func:`_cost`)."""
    llm = record.get("llm")
    if not isinstance(llm, dict):
        return 0.0
    cost = llm.get("cost_usd")
    return float(cost) if isinstance(cost, (int, float)) else 0.0


def _record_spend_cap(
    batch_dir: Path, cap_usd: float, spent_usd: float, job: Job
) -> None:
    """Stop at the batch's spend cap, and write down why (§14.29).

    Printed AND written: a batch that stopped early has to be distinguishable
    from one that finished, and the two readers who need to know look in
    different places — the person watching the terminal, and whoever opens the
    directory a week later. The exit code alone cannot say it, because 1 already
    means "a bug failed the wave gate" and 2 "the harness broke".

    The row that crossed the cap is already complete and on disk: the cap
    refuses the *next* job, never an in-flight one, so no partial row is written
    and "finish the current row" is structural rather than a promise. The
    unstarted jobs keep no key in ``state.json``, so this is not a kill switch —
    running the batch again without the variable picks up where it stopped.
    """
    payload = {
        "schema": 1,
        "reason": "spend_cap",
        "cap_usd": cap_usd,
        "spent_usd": round(spent_usd, 6),
        "stopped_utc": utc_now(),
        "first_unstarted_key": job.key,
    }
    path = batch_dir / SPEND_CAP_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)
    print(
        f"\n=== spend cap reached: ${spent_usd:.6f} >= ${cap_usd:.6f} "
        f"(VANTA_MAX_SPEND_USD) ===",
        flush=True,
    )
    print(
        f"  refusing to start {job.key}: the row that crossed the cap is complete, "
        "every later job is unstarted.",
        flush=True,
    )
    print(f"  reason recorded in {path}", flush=True)
    print(
        "  re-run this batch without VANTA_MAX_SPEND_USD to continue from here.",
        flush=True,
    )


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
    max_attempts: Optional[int] = None,
    disk_mb: int,
    show_hints: bool = False,
    plan_only: bool = False,
    max_spend_usd: Optional[float] = None,
    preflight: Optional[Callable[[], Optional[str]]] = None,
) -> tuple[int, list[WaveResult]]:
    """The whole batch: per wave, D13's four phases, then the wave-1 gate.

    ``show_hints`` is the with-hints ablation's switch (§2.4.4) and is off for every batch a
    verdict is meant to rest on: it sends the report's comments on top of its content, which
    changes what the prompt is, and so changes the job key (``manifest.job_key``) and is
    recorded on every row it touches.

    ``max_spend_usd`` is §14.29's cap (``VANTA_MAX_SPEND_USD``), ``None`` for no cap. It is
    checked BETWEEN jobs, against the cost of the rows already on disk: a job that would push
    the batch past the cap is not started, and the count it is compared against includes rows
    an earlier invocation of the same batch wrote, so a resume cannot double-spend.

    ``preflight`` is item 14's endpoint check, ``None`` for no per-row check. It is a
    zero-argument callable returning ``None`` or a sentence saying why the endpoint did not
    answer, and it is asked **before every row** — so an endpoint that goes away mid-batch
    stops the batch instead of writing a failed row for every job left. ``run`` passes the
    real probe; the default is ``None`` for the same reason ``max_spend_usd`` is read in
    the CLI rather than here: the check belongs to the invocation, and a test that drives this
    loop must not depend on whatever endpoint the developer happens to have running.

    ``max_attempts`` is the invocation's ``--max-attempts``, or ``None`` when the flag was
    NOT given. An arm that pins a budget owns it (B0 *is* "one Creator call"), so with no
    flag the harness states the pin and records it, and an explicit flag that contradicts
    it is refused by :func:`_require_arm_budget` before the schedule is printed. Every
    other arm runs what the invocation asked for, or the binary's own default.
    """
    paths = BatchPaths.under(batch_dir)
    manifest = Manifest(batch_dir)
    # §14.29 — the batch's spend to date, from the rows already on disk (a resumed
    # batch has some). `None` for a row that priced nothing.
    spent = _spent_so_far(manifest)
    # The harness's own revision, read once for the whole batch (§7.3). The binary's
    # is read per row from `run.summary`, because it belongs to the binary and not to
    # this tree.
    harness = git_revision()
    # The split's own spellings, NOT filtered down to a known list: a filter here
    # would drop `vanta_planner` before the guard could refuse it, and a split that
    # names the placeholder must not start at all (§4.4).
    spellings = list(split.configs)
    _require_arm_list(spellings)
    # M3 item 11 — refuse an explicit `--max-attempts` a pinned arm contradicts,
    # before the schedule is printed and before any VM boots.
    _require_arm_budget(spellings, max_attempts)
    # D13's order, mode-major, derived from the spellings themselves, so every repair
    # pass precedes every review pass by construction. The `config` column of a
    # phase's rows is the spelling's config value (`baseline` for both `b1` and
    # `suite`), which is what keeps a legacy job key byte-identical.
    phases = arms_mod.phases_for(spellings)
    # The config values whose repair rows a review pass may pool (§7.1): every
    # spelling that runs a repair at all, minus the review-only ones. A repair-only
    # arm contributes a candidate set; `suite` (review-only) contributes none.
    repair_configs = [
        arms_mod.config_of(written)
        for written in spellings
        if REPAIR in arms_mod.modes_of(written)
    ]

    plan = _render_schedule(batch_id, preps, entries, spellings, seeds, waves)
    print(plan, flush=True)
    if show_hints:
        print(
            "[prompt] --show-hints: every row's issue text is the report's content *plus* "
            "its comments (§2.4.4). These rows take a `#hints` job key and record "
            "show_hints=true; they are a separate arm and not comparable-by-name to a "
            "default run.",
            flush=True,
        )
    if plan_only:
        return 0, []

    checkpoint_pythons: dict[str, str] = {}
    print(
        f"[env] harness {harness.commit or 'unknown'}"
        f"{' (dirty)' if harness.dirty else ''}",
        flush=True,
    )
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
        for mode, written in phases:
            config = arms_mod.config_of(written)
            arm = arms_mod.arm_for(written, mode)
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
                                arm=arm,
                                mode=REPAIR,
                                seed=seed,
                                checkpoint=checkpoint,
                                python_version_pinned=entry["python_version"],
                                harness=harness,
                                show_hints=show_hints,
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
                                arm=arm,
                                mode=REVIEW,
                                seed=seed,
                                checkpoint=checkpoint,
                                python_version_pinned=entry["python_version"],
                                candidate=candidate,
                                harness=harness,
                                show_hints=show_hints,
                            )
                            for candidate in _review_candidates(
                                batch_dir=batch_dir,
                                batch_id=batch_id,
                                manifest=manifest,
                                prep=prep,
                                configs=repair_configs,
                                seed=seed,
                                checkpoint=checkpoint,
                                disk_mb=disk_mb,
                                show_hints=show_hints,
                            )
                        ]
                    for job in jobs:
                        if manifest.is_done(job.key):
                            print(f"  {job.label()} done — skipping (resume)", flush=True)
                            found = manifest.find(job.key)
                            if found:
                                result.records.append(found)
                            continue
                        # §14.29 — the spend cap is checked BEFORE the job starts,
                        # never during one: the check is between rows, so the row
                        # that crossed the cap is already complete and the rows
                        # that follow it are never half-written. `>=` rather than
                        # `>`: a batch that has reached its budget has spent it.
                        if max_spend_usd is not None and spent >= max_spend_usd:
                            _record_spend_cap(batch_dir, max_spend_usd, spent, job)
                            return 1, results
                        # Item 14 — before the row, not after it. Asked here rather
                        # than inside `run_job` because the refusal has to leave no
                        # trace at all: `run_job`'s first act is to mark the key
                        # `running` in `state.json`, and a key parked there is a row
                        # the next resume re-runs. Raising before the call means the
                        # cursor never moves and the batch continues from here once
                        # the endpoint is back.
                        if preflight is not None:
                            require_model_endpoint(preflight=preflight, before=job.key)
                        record = run_job(
                            job=job,
                            paths=paths,
                            manifest=manifest,
                            vanta_bin=vanta_bin,
                            # The arm owns the budget, not the invocation: B0 is
                            # "one Creator call, no feedback" (§M3 item 11), so a
                            # pinned arm states its budget here and an explicit
                            # `--max-attempts` that disagrees was already refused
                            # by `_require_arm_budget`. Every other arm runs what
                            # the invocation asked for, or the binary's default.
                            max_attempts=_arm_attempt_budget(arm, max_attempts),
                            disk_mb=disk_mb,
                            python_version=checkpoint_pythons.get(checkpoint, ""),
                        )
                        if record:
                            result.records.append(record)
                            spent += _row_cost(record)
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
    show_hints: bool = False,
) -> list[Candidate]:
    """The candidate set for one (bug, seed), from the repair rows already written.

    ``batch_id`` is threaded in rather than read off ``manifest.batch_dir.name``:
    it is the one field a caller could pass under a different name than the
    directory it writes into, and a lookup keyed on the wrong batch would silently
    find no repair rows and quietly shrink every review set to {dev fix, no-op}.

    ``show_hints`` is threaded in for the same reason it is threaded into the job
    list: it is part of the key (§2.4.4), so a with-hints batch's repair rows are
    keyed ``…#hints`` and a lookup that left the flag out would find none of them
    and silently shrink every review set.
    """
    records: list[dict[str, Any]] = []
    for config in configs:
        key = job_key(
            batch_id=batch_id,
            bug=prep.key,
            config=config,
            mode=REPAIR,
            seed=seed,
            show_hints=show_hints,
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


def _require_arm_list(spellings: Sequence[str]) -> None:
    """Refuse a split whose arm list cannot be scheduled, or names the placeholder.

    [VANTA-WHY] M3 item 8 follow-up — this used to require exactly
    ``["baseline", "vanta"]``, because §7.1's study had two arms and a review row
    whose counterpart was missing could not be paired. The pairing check is now an
    **analysis**-time property, checked by candidate key over the rows a batch
    wrote, so ANY arm list is allowed to run; what stays here is the set of
    refusals that are facts about the list itself rather than about pairing:

    * an unknown spelling (a typo must not quietly run a smaller study),
    * two spellings that normalise to ONE config — ``baseline`` and ``suite``, say.
      The job key has no arm column (``manifest.job_key``), so the second arm would
      be skipped as the first on a resume and the manifest would hold one
      experiment's row under the other's name,
    * ``vanta_planner``, §4.4's placeholder, whose runs return ``InfraFailure``.

    Repairs-before-reviews is deliberately NOT checked here: :func:`arms.phases_for`
    builds the order mode-major, so the schedule and this list cannot disagree — the
    old ``PHASES != expected`` comparison has nothing left to catch.
    """
    for written in spellings:
        if written == "vanta_planner":
            raise RunnerError(
                "vanta_planner is a placeholder that returns InfraFailure and must "
                "not be in a run split"
            )
    try:
        arms_mod.resolve_configs(spellings)
    except arms_mod.ArmError as exc:
        raise RunnerError(str(exc)) from exc


def _arm_attempt_budget(arm: str, requested: Optional[int]) -> int:
    """The attempt budget one arm runs with, or a refusal (→ :func:`arms.arm_attempts`)."""
    try:
        return arms_mod.arm_attempts(arm, requested)
    except arms_mod.ArmError as exc:
        raise RunnerError(str(exc)) from exc


def _require_arm_budget(spellings: Sequence[str], requested: Optional[int]) -> None:
    """Refuse an explicit ``--max-attempts`` that contradicts a pinned arm.

    [VANTA-WHY] M3 item 11 — B0's budget is 1 because that IS the arm ("one Creator
    call, no feedback"), so the harness **pins** it rather than letting the
    invocation decide. This is the pre-launch half of "refuse inconsistent
    combinations": a flag that disagrees is refused here, before the schedule is
    printed and before any VM boots. With no flag there is nothing to refuse — the
    arm's own budget is used and recorded on the row.
    """
    if requested is None:
        return
    for mode, written in arms_mod.phases_for(spellings):
        _arm_attempt_budget(arms_mod.arm_for(written, mode), requested)


def wave_gate_failures(records: Sequence[dict[str, Any]]) -> list[str]:
    """§8.2's wave gate, as assertions over the rows a wave just wrote.

    Wave 1 is a go/no-go gate: if the oracle or the plumbing misbehaves there, the
    batch must stop and report rather than build the next wave on it. The checks
    are the ones whose failure invalidates every later number — not a threshold on
    model behaviour, which is the thing being measured.

    [WHY] A row whose verdict is `NeedsReview` is deliberately NOT a failure here
    (§14.6). The loop refused to decide because no rule corroborated a rejection,
    which is the outcome §14 exists to produce, so gating on it would stop a wave
    for behaving as designed. What this gate still flags on such a row is the
    verdict-independent censor below: an `InfraFailure`, or a row whose oracle
    could not apply the candidate at all (`unobservable`) — a fact about the
    environment, not about the decision. A NeedsReview that graded normally
    carries `needs_review_on_good`/`needs_review_on_bad` and reaches the digest;
    its counts belong to `metrics.py`, which scores it as its own outcome.
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
    # [VANTA-WHY] M3 item 8 follow-up — §8.2's "paired vanta rows exist" is NOT
    # checked here any more. The check was "both configs review one set at one seed",
    # which a repair-only arm (B0) cannot satisfy and which an arbitrary arm list has
    # no fixed pair to test against. It moved to ANALYSIS time, where the rows on
    # disk are grouped by candidate key and a missing counterpart is visible on the
    # number it would have changed — the wave gate must not stop a wave for running
    # an arm list the split declared on purpose.
    return problems


def _render_schedule(
    batch_id: str,
    preps: Sequence[prepare_mod.BugPrep],
    entries: dict[str, dict[str, Any]],
    spellings: Sequence[str],
    seeds: Sequence[int],
    waves: Sequence[int],
) -> str:
    """The plan, and an honest statement of what is not yet known.

    The review row count is a **floor**, not a prediction: the pooled failed
    patches do not exist until the repairs ran, so the set printed here is
    {developer fix, no-op} and the real one is 2–4 per bug (§7.1).

    [VANTA-WHY] The row counts come from :func:`arms.phases_for`, not from the
    config list's length: a repair-only arm (B0) contributes a repair row and no
    review row, and a count derived from ``len(configs)`` would print a review
    figure for a phase the split does not run.
    """
    phases = arms_mod.phases_for(spellings)
    repairs = sum(1 for mode, _ in phases if mode == REPAIR)
    reviews = sum(1 for mode, _ in phases if mode == REVIEW)
    lines = [
        f"batch {batch_id}",
        f"order: {' → '.join(f'{mode} {written}' for mode, written in phases)}",
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
            repair = repairs * len(seeds)
            review = reviews * 2 * len(seeds)
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
@dataclass(frozen=True)
class PreparedBatch:
    """``prepare.json`` read back — the batch's preps, entries and its window."""

    batch_dir: Path
    preps: list[prepare_mod.BugPrep]
    entries: dict[str, dict[str, Any]]
    n_ctx: Optional[int]
    """The window ``prepare`` sized the context gate against (§1.6.1), as recorded.
    ``None`` means the batch predates the field, which :func:`context_window_warning`
    reports rather than inventing a value."""

    n_ctx_source: Optional[str]


def recorded_split_path(batch_dir: Path) -> Optional[Path]:
    """The split a batch was prepared with, as its ``prepare.json`` recorded it.

    ``prepare`` writes the split path it read (``_prepare_payload``'s ``split``),
    so a later stage given no ``--split`` can resolve the *same* split the batch's
    preps were derived from, instead of a default that may not even name the same
    bugs (#E). ``None`` when the batch predates the field or carries no
    ``prepare.json`` — the caller then keeps whatever default it had.
    """
    raw = _read_json(batch_dir / "prepare.json").get("split")
    return Path(str(raw)) if raw else None


def load_prepared(
    *,
    batch_dir: Path,
    split: Split,
    bugsinpy_root: Path,
) -> PreparedBatch:
    """Rebuild genuine :class:`BugPrep` objects from ``prepare.json``.

    No second persisted format: the runner calls the *same* ``to_dict``/``from_dict``
    pair ``prepare`` wrote with (``records.py``), so a batch prepared in one process
    can be run in another without re-deriving three double suite runs per bug, and a
    field the reader does not know about cannot be dropped silently. ``load_bug``
    re-reads ``bug.info`` so a moved clone fails loudly here rather than silently
    grading against a tree nobody prepared.
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
        try:
            prep = prepare_mod.BugPrep.from_dict(
                raw,
                bug=bug,
                root=batch_dir / "worktrees" / f"{project}_{bug_id}",
                test_cmd=[str(part) for part in entry["test_cmd"]],
                where=f"{key} of batch {batch_dir}",
            )
        except records_mod.RecordError as exc:
            # The record contract's error already names the field and the batch;
            # re-raising as a RunnerError is what turns it into a clean CLI message
            # instead of a traceback, without rewriting the message.
            raise RunnerError(f"{batch_dir / 'prepare.json'}: {exc}") from exc
        preps.append(prep)
    recorded_n_ctx = payload.get("n_ctx")
    if recorded_n_ctx is not None and not isinstance(recorded_n_ctx, int):
        raise RunnerError(
            f"{batch_dir / 'prepare.json'}: field 'n_ctx' is "
            f"{type(recorded_n_ctx).__name__}, not an integer"
        )
    return PreparedBatch(
        batch_dir=batch_dir,
        preps=preps,
        entries=entries,
        n_ctx=recorded_n_ctx,
        n_ctx_source=payload.get("n_ctx_source"),
    )


def context_window_warning(batch: PreparedBatch) -> Optional[str]:
    """Warn when this run's window is not the one ``prepare`` sized against (§1.6.1).

    ``prepare`` refuses to gate against the 65536 fallback, but the *run* still
    resolves its own window the way ``llm::context_window_tokens`` does — the
    variable, then ``/props``, then the fallback — and that can move between the two
    (a server restarted with a different ``-c``, an exported variable left out of the
    shell that starts the run). When it does, every context verdict in the batch was
    measured against a prompt the run will not send, which is exactly how PySnooper/1
    and /2 came to be rejected for a window no run would have used.

    Returns the message, or ``None`` when they agree; printing is the caller's job so
    the comparison is testable without a batch or a server.
    """
    if batch.n_ctx is None:
        return (
            f"{batch.batch_dir / 'prepare.json'} records no n_ctx, so this run cannot "
            "check that it uses the window prepare sized the context gate against. "
            "That batch predates the recorded window; re-run prepare before trusting "
            "its context verdicts."
        )
    run_n_ctx, run_source = ctx_mod.server_context_window()
    if run_n_ctx == batch.n_ctx:
        return None
    prepare_budget, _ = ctx_mod.budget_tokens(batch.n_ctx)
    run_budget, _ = ctx_mod.budget_tokens(run_n_ctx)
    return (
        f"context window mismatch: prepare sized this batch against n_ctx="
        f"{batch.n_ctx} (source: {batch.n_ctx_source or 'unrecorded'}) but this run "
        f"resolves n_ctx={run_n_ctx} (source: {run_source}). The prompt budget moves "
        f"with it ({prepare_budget} tok at prepare, {run_budget} tok now), so the "
        "batch's context verdicts describe a window this run will not use. Export "
        "VANTA_LLM_CONTEXT_TOKENS (or point VANTA_LLM_BASE_URL at the server prepare "
        "measured) so the two agree, or re-run prepare."
    )


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
            # [VANTA-WHY] M3 item 8 follow-up — the arm, backfilled for a row written
            # before the field existed (`arms.row_arm`), so a label file keys an old
            # row and a new one by the same condition.
            "arm": arms_mod.row_arm(record),
            "mode": record["mode"],
            "kind": candidate["kind"],
            "source": candidate["source"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def digest(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per (§4.4 provenance axes), plus the label histogram per config.

    Both commits are axes: a row produced by the harness at HEAD *and* the rows
    produced by a binary built three commits earlier are two different provenance
    claims, and pooling them would report one number for two experiments.

    [VANTA-WHY] M3 item 8 follow-up — the arm is an axis too. It is redundant with
    ``(config, mode)`` for every legacy row (``b1``/``suite``/``b3`` are a function
    of exactly those two), so adding it does not move a single legacy group; it is
    what separates two arms that share a config once the registry grows one
    (``b1`` and ``suite`` already share ``baseline``, and M3.5's ``b3-v2-*`` arms
    share ``vanta``). A row written before the field existed is backfilled from its
    own ``(mode, config)`` (:func:`arms.row_arm`), so old and new rows group alike.
    """
    groups: dict[tuple, dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("config"),
            arms_mod.row_arm(record),
            record.get("mode"),
            record.get("seed"),
            record.get("model"),
            record.get("harness_commit"),
            record.get("vanta_build_commit"),
        )
        group = groups.setdefault(
            key,
            {
                "config": record.get("config"),
                "arm": arms_mod.row_arm(record),
                "mode": record.get("mode"),
                "seed": record.get("seed"),
                "model": record.get("model"),
                "harness_commit": record.get("harness_commit"),
                "harness_dirty": record.get("harness_dirty"),
                "vanta_build_commit": record.get("vanta_build_commit"),
                "vanta_build_dirty": record.get("vanta_build_dirty"),
                "runs": 0,
                "labels": {},
                # §1.6.2 item 5 — the localization metric, pooled per group so the
                # digest answers "was the bug's own file in the prompt" without a
                # reader re-deriving it from `context_selection` strings. `unknown`
                # is rows that carried no evidence, kept apart from `miss`.
                "localization": {"hit": 0, "miss": 0, "unknown": 0},
            },
        )
        group["runs"] += 1
        label = str(record.get("label"))
        group["labels"][label] = group["labels"].get(label, 0) + 1
        hit = record.get("localization_hit")
        group["localization"][
            "unknown" if hit is None else ("hit" if hit else "miss")
        ] += 1
    return {"runs": len(records), "groups": list(groups.values())}


def render_digest(payload: dict[str, Any]) -> str:
    lines = [
        f"{'config':<10} {'arm':<7} {'mode':<8} {'seed':>4} {'model':<18} {'harness':<10} "
        f"{'build':<10} {'runs':>5}  labels  localization",
        "-" * 130,
    ]
    for group in payload.get("groups", []):
        harness = str(group.get("harness_commit") or "")[:9]
        build = str(group.get("vanta_build_commit") or "")[:9]
        model = str(group.get("model") or "")[:18]
        labels = " ".join(f"{k}={v}" for k, v in sorted(group["labels"].items()))
        loc = group.get("localization") or {}
        localization = " ".join(
            f"{key}={loc.get(key, 0)}" for key in ("hit", "miss", "unknown")
        )
        lines.append(
            f"{str(group['config']):<10} {str(group.get('arm') or ''):<7} "
            f"{str(group['mode']):<8} {str(group['seed']):>4} "
            f"{model:<18} {harness:<10} {build:<10} {group['runs']:>5}  {labels}  "
            f"{localization}"
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
