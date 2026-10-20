"""Context-budget precheck (design §1.6) — ``prepare`` rejects, it does not warn.

Vanta hands the Creator the **whole source concat**, capped by two constants in
``src/repo/mod.rs``: 64 KB per file and 256 KB in total. The cap is the part that
makes this precheck necessary rather than nice-to-have:

* A repo that fits is fine.
* A repo that does **not** fit is silently truncated. The Creator is shown an
  arbitrary subset of a large repo, the bug's own file may not be in it, and the
  run fails for a reason that has nothing to do with the model. On ``youtube-dl``
  (hundreds of extractor files) that is the expected outcome, and discovering it
  *after* a checkpoint has been baked and a wave scheduled costs hours.

So ``prepare`` measures every candidate against the window and **rejects** the
ones that will not fit, and writes ``context_sizes.md`` for all of them — rejects
and reserves alike, so the choice of pilot set is data rather than assertion.

The budget is not "some fraction of the window":

    budget = n_ctx − output reserve − chat overhead − margin       (§1.6.2)

The first two terms are what Vanta's own run-time guard subtracts
(``context_budget_exceeded`` in ``src/llm/mod.rs``, and ``max_tokens: 32768`` in
``src/agents/creator.rs``), so the precheck's boundary sits on the guard's
boundary; the margin is the harness's addition, reserved for feedback growth
across attempts. Both the token estimator and these constants are mirrors, not
independent choices — a precheck that disagrees with the guard either discards
candidates Vanta would have accepted, or spends a checkpoint, a wave and a night
on a bug that dies at attempt 1.

[VANTA-WHY] The walk here mirrors ``repo::collect_source_files`` constant for
constant (extensions, per-file cap, total cap, skip dirs). If it drifted, the
precheck would be measuring a different prompt from the one Vanta builds, which is
worse than not measuring at all. One honest difference is recorded in the report:
Vanta walks in ``read_dir`` order and we walk in sorted order, so when the budget
cap bites the two runs can drop *different* files. We therefore report the
totals as well as the capped estimate, and the rejection reason names truncation
explicitly instead of pretending the capped number is what Vanta will send.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .records import RecordError, field_of, strings_of

# ---- mirrors of src/repo/mod.rs -------------------------------------------
SOURCE_EXTENSIONS = frozenset(
    {"py", "rs", "go", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp"}
)
MAX_TOTAL_SOURCE_BYTES = 256 * 1024
MAX_FILE_BYTES = 64 * 1024
SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".idea",
        ".vscode",
    }
)

# ---- harness constants, all documented in §1.6 ----------------------------
CONTEXT_MARGIN_FRACTION = 0.25
"""Fraction of ``n_ctx`` the prompt must leave free. Not a magic number: the
Creator's feedback grows every attempt (test output is appended), and a prompt
that fits attempt 1 but not attempt 3 dies mid-run rather than at precheck."""

BYTES_PER_TOKEN_ESTIMATE = 4
"""Bytes of code per token — the *same* constant Vanta's own guard uses.

[VANTA-WHY] This mirrors ``TOKEN_ESTIMATE_BYTES_PER_TOKEN`` in ``src/llm/mod.rs``
byte for byte, ceil division included. It deliberately does **not** carry its own
calibration: the precheck's accept/reject boundary has to sit exactly where the
run-time guard's boundary sits, or a bug is rejected here for a prompt that Vanta
would have accepted (a silent loss of candidates) or accepted here for one that
dies at attempt 1 (a wasted checkpoint, wave and night). The value is 4 because
the ``heretic`` run measured 84500 estimated against 55300 real prompt tokens at
``/ 3`` — the Rust comment records it. Wave 1 re-measures the ratio from
``usage.prompt_tokens`` and §8.1 carries the number; if it moves, it moves in
``llm/mod.rs`` first and this constant follows."""

PROMPT_FRAME_TOKENS = 3000
"""System prompt, ``===FILE===`` framing, and the per-attempt scaffolding that is
present regardless of the source size."""

OUTPUT_RESERVE_TOKENS = 32_768
"""What the Creator asks the server to generate (``"max_tokens": 32768`` in
``src/agents/creator.rs``). The window is shared: prompt and completion come out of
the same ``n_ctx``, so this is subtracted before the margin is taken. Without it
the precheck would pass prompts that the guard below rejects — the exact
disagreement this module exists to prevent."""

CHAT_OVERHEAD_TOKENS = 256
"""``CHAT_OVERHEAD_TOKENS`` in ``src/llm/mod.rs``: the template and role markers the
server adds around the rendered messages. Small, but it is inside the guard's
comparison, so leaving it out would put the two boundaries 256 tokens apart."""

FALLBACK_CONTEXT_TOKENS = 65_536
FALLBACK_SOURCE = "fallback (server unreachable)"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
ENV_SOURCE = "VANTA_LLM_CONTEXT_TOKENS"
FLAG_SOURCE = "--n-ctx"
PROPS_SOURCE = "server /props"


class ContextError(RuntimeError):
    """The tree could not be measured."""


@dataclass
class ConcatStats:
    """What Vanta's source walk would hand the Creator for one tree."""

    files_eligible: int = 0
    bytes_eligible: int = 0
    files_included: int = 0
    bytes_included: int = 0
    files_skipped_oversize: int = 0
    files_skipped_budget: int = 0
    largest_files: list[tuple[str, int]] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.files_skipped_oversize > 0 or self.files_skipped_budget > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "files_eligible": self.files_eligible,
            "bytes_eligible": self.bytes_eligible,
            "files_included": self.files_included,
            "bytes_included": self.bytes_included,
            "files_skipped_oversize": self.files_skipped_oversize,
            "files_skipped_budget": self.files_skipped_budget,
            "largest_files": [[rel, size] for rel, size in self.largest_files],
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "ConcatStats":
        owner = cls.__name__
        largest = field_of(payload, "largest_files", owner=owner, where=where)
        if not isinstance(largest, list):
            raise RecordError(
                f"{owner} in {where}: field 'largest_files' is "
                f"{type(largest).__name__}, not a list"
            )
        return cls(
            files_eligible=int(field_of(payload, "files_eligible", owner=owner, where=where)),
            bytes_eligible=int(field_of(payload, "bytes_eligible", owner=owner, where=where)),
            files_included=int(field_of(payload, "files_included", owner=owner, where=where)),
            bytes_included=int(field_of(payload, "bytes_included", owner=owner, where=where)),
            files_skipped_oversize=int(
                field_of(payload, "files_skipped_oversize", owner=owner, where=where)
            ),
            files_skipped_budget=int(
                field_of(payload, "files_skipped_budget", owner=owner, where=where)
            ),
            largest_files=[(str(rel), int(size)) for rel, size in largest],
        )


def measure_tree(tree: Path) -> ConcatStats:
    """Walk ``tree`` exactly as ``repo::collect_source_files`` would.

    The walk is sorted here (Vanta's is not) purely so the numbers are stable
    across runs; the byte caps — the thing that actually decides rejection — are
    applied in the same order and with the same arithmetic.
    """
    stats = ConcatStats()
    candidates: list[tuple[str, int]] = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(tree).as_posix()
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            continue
        if path.suffix.lstrip(".") not in SOURCE_EXTENSIONS:
            continue
        size = path.stat().st_size
        stats.files_eligible += 1
        stats.bytes_eligible += size
        candidates.append((rel, size))

    total = 0
    for rel, size in candidates:
        if size > MAX_FILE_BYTES:
            stats.files_skipped_oversize += 1
            continue
        if total + size > MAX_TOTAL_SOURCE_BYTES:
            stats.files_skipped_budget += 1
            continue
        total += size
        stats.files_included += 1
        stats.bytes_included += size

    stats.largest_files = sorted(candidates, key=lambda item: -item[1])[:5]
    return stats


@dataclass
class ContextVerdict:
    context_ok: bool
    reasons: list[str]
    estimated_prompt_tokens: int
    budget_tokens: int
    n_ctx: int
    n_ctx_source: str
    stats: ConcatStats
    output_reserve_tokens: int = OUTPUT_RESERVE_TOKENS
    margin_tokens: int = 0

    @property
    def headroom_tokens(self) -> int:
        """How much the estimate could grow before it hits the budget."""
        return self.budget_tokens - self.estimated_prompt_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "context_ok": self.context_ok,
            "reasons": list(self.reasons),
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "budget_tokens": self.budget_tokens,
            # Derived, but recorded: it is the number a reader compares against the
            # estimate, and leaving it out would push the reader into recomputing it.
            "headroom_tokens": self.headroom_tokens,
            "n_ctx": self.n_ctx,
            "n_ctx_source": self.n_ctx_source,
            "output_reserve_tokens": self.output_reserve_tokens,
            "margin_tokens": self.margin_tokens,
            "stats": self.stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "ContextVerdict":
        owner = cls.__name__
        return cls(
            context_ok=bool(field_of(payload, "context_ok", owner=owner, where=where)),
            reasons=strings_of(payload, "reasons", owner=owner, where=where),
            estimated_prompt_tokens=int(
                field_of(payload, "estimated_prompt_tokens", owner=owner, where=where)
            ),
            budget_tokens=int(field_of(payload, "budget_tokens", owner=owner, where=where)),
            n_ctx=int(field_of(payload, "n_ctx", owner=owner, where=where)),
            n_ctx_source=str(field_of(payload, "n_ctx_source", owner=owner, where=where)),
            stats=ConcatStats.from_dict(
                field_of(payload, "stats", owner=owner, where=where), where=where
            ),
            output_reserve_tokens=int(
                field_of(payload, "output_reserve_tokens", owner=owner, where=where)
            ),
            margin_tokens=int(field_of(payload, "margin_tokens", owner=owner, where=where)),
        )


def estimate_tokens(text_bytes: int) -> int:
    """``bytes / 4``, rounded **up** — the Rust's ``conservative_token_estimate``.

    Ceiling rather than truncation because the guard must never under-count: a
    prompt that is one byte over a multiple of four is one token, not zero.
    """
    return -(-text_bytes // BYTES_PER_TOKEN_ESTIMATE)


def budget_tokens(n_ctx: int) -> tuple[int, int]:
    """``(budget, margin)`` for a window: the prompt budget the precheck enforces.

        budget = n_ctx − output reserve − chat overhead − margin
        margin = 25 % of n_ctx                                     (§1.6.2)

    The output reserve and the chat overhead are the two terms Vanta's own guard
    subtracts before it decides a prompt fits (``context_budget_exceeded``), so
    subtracting them here is what keeps the two boundaries on top of each other.
    The margin is the harness's addition, and it is the reason this precheck is
    stricter than the guard: it is reserved for feedback growth across attempts.
    """
    margin = int(n_ctx * CONTEXT_MARGIN_FRACTION)
    return n_ctx - OUTPUT_RESERVE_TOKENS - CHAT_OVERHEAD_TOKENS - margin, margin


def precheck(
    tree: Path,
    issue_text: str,
    *,
    n_ctx: Optional[int] = None,
    n_ctx_source: Optional[str] = None,
) -> ContextVerdict:
    stats = measure_tree(tree)
    if n_ctx is None:
        # ``explicit_context_window``, not ``server_context_window``: the gate
        # refuses to run against the 65536 fallback (see that function). A caller
        # that forgot to thread an explicit window therefore fails here instead of
        # producing a verdict nobody should act on.
        n_ctx, n_ctx_source = explicit_context_window()

    source_tokens = estimate_tokens(stats.bytes_included)
    issue_tokens = estimate_tokens(len(issue_text.encode("utf-8")))
    estimated = source_tokens + issue_tokens + PROMPT_FRAME_TOKENS
    budget, margin = budget_tokens(n_ctx)

    reasons: list[str] = []
    if stats.files_skipped_oversize or stats.files_skipped_budget:
        reasons.append(
            "source_truncated_by_cap: "
            f"{stats.files_skipped_oversize} file(s) over {MAX_FILE_BYTES} B, "
            f"{stats.files_skipped_budget} file(s) over the {MAX_TOTAL_SOURCE_BYTES} B "
            f"total budget — Vanta would show the Creator {stats.bytes_included} B of "
            f"{stats.bytes_eligible} B and the bug's own file may not be in it"
        )
    if estimated > budget:
        reasons.append(
            f"prompt_exceeds_budget: ~{estimated} tok estimated vs a budget of "
            f"{budget} tok (= n_ctx {n_ctx} − {OUTPUT_RESERVE_TOKENS} output − "
            f"{CHAT_OVERHEAD_TOKENS} overhead − {margin} margin)"
        )
    if budget <= 0:
        reasons.append(
            f"no_prompt_budget: n_ctx={n_ctx} ({n_ctx_source or 'unknown'}) is smaller "
            f"than the {OUTPUT_RESERVE_TOKENS} tok the Creator needs for its completion "
            "plus the margin; no prompt fits this window"
        )

    return ContextVerdict(
        context_ok=not reasons,
        reasons=reasons,
        estimated_prompt_tokens=estimated,
        budget_tokens=budget,
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source or "unknown",
        stats=stats,
        margin_tokens=margin,
    )


# ---------------------------------------------------------------------------
# the server's own `n_ctx`
# ---------------------------------------------------------------------------
def env_context_window() -> Optional[tuple[int, str]]:
    """``VANTA_LLM_CONTEXT_TOKENS`` as ``(n_ctx, source)``, or ``None`` if unset."""
    override = os.environ.get(ENV_SOURCE, "").strip()
    if not override:
        return None
    try:
        value = int(override)
    except ValueError as exc:
        raise ContextError(f"{ENV_SOURCE}={override!r} is not an integer") from exc
    if value <= 0:
        raise ContextError(f"{ENV_SOURCE}={override!r} is not a positive window")
    return value, ENV_SOURCE


def server_context_window() -> tuple[int, str]:
    """``n_ctx`` the way ``llm::context_window_tokens`` resolves it.

    Order: ``VANTA_LLM_CONTEXT_TOKENS`` (wins), else the server's ``/props``,
    else 65536. Same order as the Rust, so the precheck and the run-time guard
    agree about the window they are enforcing. The fallback is *returned* here —
    this is the run-time mirror, and a run really does fall back — but the
    precheck does not go through this function: see
    :func:`explicit_context_window`.
    """
    env = env_context_window()
    if env is not None:
        return env
    probed = probe_props_n_ctx()
    if probed is not None:
        return probed, PROPS_SOURCE
    return FALLBACK_CONTEXT_TOKENS, FALLBACK_SOURCE


def explicit_context_window(*, n_ctx: Optional[int] = None) -> tuple[int, str]:
    """The window to size the context gate against, from an **explicit** source.

    Either ``VANTA_LLM_CONTEXT_TOKENS``, ``--n-ctx``, or a reachable server's
    ``/props`` — and if none of the three is available this **refuses** rather than
    falling back to 65536.

    [VANTA-WHY] The fallback is not a neutral default, it is a wrong answer that
    looks like a right one. The first full prepare ran with the variable unset in
    the shell, sized against 65536, and got a 16128-token budget instead of 65280:
    PySnooper/1 and /2 were then rejected as "source_truncated_by_cap" for a window
    no run would ever have used. A gate is only worth running if it measures the
    prompt the run will actually send, so the missing window is an error here and
    not a warning in a report.

    ``--n-ctx`` is checked against the other two sources instead of overriding
    them: a run resolves the window from the variable, then ``/props``, then the
    fallback — it never reads this flag — so a flag that disagrees would size
    against a window the run will not use, which is the bug this whole function
    exists to prevent.
    """
    env = env_context_window()
    probed = probe_props_n_ctx()
    if n_ctx is not None:
        if env is not None and env[0] != n_ctx:
            raise ContextError(
                f"--n-ctx {n_ctx} disagrees with {ENV_SOURCE}={env[0]}. The variable "
                "wins at run time, so the flag would size the gate against a window "
                "the run will not use. Keep one of the two."
            )
        if probed is not None and probed != n_ctx:
            raise ContextError(
                f"--n-ctx {n_ctx} disagrees with the server's /props n_ctx ({probed}). "
                "A run reads /props when the variable is unset, so the flag would "
                "size the gate against a window the run will not use."
            )
        return n_ctx, FLAG_SOURCE
    if env is not None:
        return env
    if probed is not None:
        return probed, PROPS_SOURCE
    raise ContextError(
        "refusing to run the context gate: no explicit context size. Set "
        f"{ENV_SOURCE}, pass --n-ctx, or start the server so /props answers. "
        f"Sizing against the {FALLBACK_CONTEXT_TOKENS}-token fallback is not an "
        "option — that window is not the one the run will use, and a verdict "
        "measured against it (16128 tok of budget instead of 65280) rejects bugs "
        "for a reason that does not exist."
    )


def probe_props_n_ctx(timeout_secs: float = 5.0) -> Optional[int]:
    base = os.environ.get("VANTA_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    urls = []
    if base.endswith("/v1"):
        urls.append(base[: -len("/v1")] + "/props")
        urls.append(base + "/props")
    else:
        urls.append(base + "/props")
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout_secs) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            continue
        settings = body.get("default_generation_settings") or {}
        value = settings.get("n_ctx") or body.get("n_ctx")
        if isinstance(value, int) and value > 0:
            return value
    return None


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------
def render_markdown(rows: list[tuple[str, str, ContextVerdict]]) -> str:
    """``context_sizes.md`` — every candidate, not only the rejects (§1.6.3)."""
    budgets = {v.budget_tokens for _, _, v in rows}
    windows = {v.n_ctx for _, _, v in rows}
    lines = [
        "# Context sizes — every candidate, measured",
        "",
    ]
    if any(v.n_ctx_source.startswith("fallback") for _, _, v in rows):
        lines += [
            f"> **Not a decision.** No server answered `/props` and "
            f"`VANTA_LLM_CONTEXT_TOKENS` is unset, so every row below was measured "
            f"against the {FALLBACK_CONTEXT_TOKENS} fallback, not the window a run "
            f"will use. Start the server (or export `VANTA_LLM_CONTEXT_TOKENS`) and "
            f"re-run `prepare` before reading anything into these verdicts.",
            "",
        ]
    lines += [
        "## The budget",
        "",
        "A bug passes only if Vanta's prompt for it fits the window with room to spare:",
        "",
        "```",
        "budget = n_ctx − output reserve − chat overhead − margin",
        f"       = n_ctx − {OUTPUT_RESERVE_TOKENS} − {CHAT_OVERHEAD_TOKENS} "
        f"− {int(CONTEXT_MARGIN_FRACTION * 100)} % of n_ctx",
        "```",
        "",
        f"- **output reserve {OUTPUT_RESERVE_TOKENS} tok** — `\"max_tokens\": 32768` in "
        "`src/agents/creator.rs`. Prompt and completion share `n_ctx`.",
        f"- **chat overhead {CHAT_OVERHEAD_TOKENS} tok** — `CHAT_OVERHEAD_TOKENS` in "
        "`src/llm/mod.rs`.",
        f"- **margin {int(CONTEXT_MARGIN_FRACTION * 100)} %** — §1.6.2. The Creator's "
        "feedback grows every attempt, so a prompt that only just fits attempt 1 dies "
        "at attempt 3.",
        "",
        f"Estimator: `ceil(bytes / {BYTES_PER_TOKEN_ESTIMATE})` + {PROMPT_FRAME_TOKENS} "
        "tok of prompt frame — the same `bytes / 4` Vanta's own run-time guard uses "
        "(`TOKEN_ESTIMATE_BYTES_PER_TOKEN`), so the precheck's boundary is the guard's "
        "boundary.",
        "",
    ]
    if len(windows) == 1:
        n_ctx = next(iter(windows))
        budget, margin = budget_tokens(n_ctx)
        lines += [
            f"With `n_ctx = {n_ctx}` (source: {rows[0][2].n_ctx_source if rows else '—'}):",
            "",
            f"```\n{budget} = {n_ctx} − {OUTPUT_RESERVE_TOKENS} − "
            f"{CHAT_OVERHEAD_TOKENS} − {margin}\n```",
            "",
            f"i.e. about **{budget * BYTES_PER_TOKEN_ESTIMATE // 1024} KB** of source "
            "before the prompt frame, against Vanta's own "
            f"`MAX_TOTAL_SOURCE_BYTES = {MAX_TOTAL_SOURCE_BYTES}` "
            f"({MAX_TOTAL_SOURCE_BYTES // 1024} KB) cap.",
            "",
        ]
    elif budgets:
        lines += [
            "More than one window was measured in this run: budgets "
            + ", ".join(str(b) for b in sorted(budgets))
            + ". A single `n_ctx` for the whole batch is what §1.6 assumes; a mixed "
            "table means the server changed under the run.",
            "",
        ]

    lines += [
        "The cap that decides many of these is Vanta's own source walk "
        f"(`MAX_FILE_BYTES={MAX_FILE_BYTES}`, `MAX_TOTAL_SOURCE_BYTES={MAX_TOTAL_SOURCE_BYTES}`); "
        "a bug is rejected when the walk would *truncate* it, because the Creator would "
        "then be shown an arbitrary subset and the bug's own file may not be in it.",
        "",
        "| bug | role | verdict | n_ctx | src B (in/eligible) | files (in/eligible) | skipped (size/budget) | est. prompt tok | budget tok | headroom | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, role, verdict in rows:
        stats = verdict.stats
        lines.append(
            "| {key} | {role} | {verdict} | {n_ctx} ({src}) | {b_in}/{b_all} | {f_in}/{f_all} | "
            "{s_size}/{s_budget} | {est} | {budget} | {head} | {reasons} |".format(
                key=key,
                role=role,
                verdict="ok" if verdict.context_ok else "**reject**",
                n_ctx=verdict.n_ctx,
                src=verdict.n_ctx_source,
                b_in=stats.bytes_included,
                b_all=stats.bytes_eligible,
                f_in=stats.files_included,
                f_all=stats.files_eligible,
                s_size=stats.files_skipped_oversize,
                s_budget=stats.files_skipped_budget,
                est=verdict.estimated_prompt_tokens,
                budget=verdict.budget_tokens,
                head=verdict.headroom_tokens,
                reasons="; ".join(verdict.reasons) or "—",
            )
        )

    lines += ["", "## Largest files per candidate", ""]
    for key, _role, verdict in rows:
        if not verdict.stats.largest_files:
            continue
        listed = ", ".join(f"`{p}` ({size} B)" for p, size in verdict.stats.largest_files)
        lines.append(f"- **{key}**: {listed}")
    lines.append("")
    return "\n".join(lines)
