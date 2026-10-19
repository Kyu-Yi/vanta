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
ones that will not fit, with a documented margin (§1.6.2), and writes
``context_sizes.md`` for all of them — rejects and reserves alike, so the choice
of pilot set is data rather than assertion.

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

BYTES_PER_TOKEN_ESTIMATE = 3.6
"""Bytes of code per token. The repo's own local estimator runs ~1.5x high versus
``usage.prompt_tokens``; this constant is the calibration point, and wave 1
replaces it with the measured ratio (§8.1)."""

PROMPT_FRAME_TOKENS = 3000
"""System prompt, ``===FILE===`` framing, and the per-attempt scaffolding that is
present regardless of the source size."""

FALLBACK_CONTEXT_TOKENS = 65_536
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


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

    def as_dict(self) -> dict[str, object]:
        return {
            "context_ok": self.context_ok,
            "reasons": self.reasons,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "budget_tokens": self.budget_tokens,
            "n_ctx": self.n_ctx,
            "n_ctx_source": self.n_ctx_source,
            "source_bytes_included": self.stats.bytes_included,
            "source_bytes_eligible": self.stats.bytes_eligible,
            "files_included": self.stats.files_included,
            "files_eligible": self.stats.files_eligible,
            "files_skipped_oversize": self.stats.files_skipped_oversize,
            "files_skipped_budget": self.stats.files_skipped_budget,
        }


def estimate_tokens(text_bytes: int) -> int:
    return int(text_bytes / BYTES_PER_TOKEN_ESTIMATE)


def precheck(
    tree: Path,
    issue_text: str,
    *,
    n_ctx: Optional[int] = None,
    n_ctx_source: Optional[str] = None,
) -> ContextVerdict:
    stats = measure_tree(tree)
    if n_ctx is None:
        n_ctx, n_ctx_source = server_context_window()

    source_tokens = estimate_tokens(stats.bytes_included)
    issue_tokens = estimate_tokens(len(issue_text.encode("utf-8")))
    estimated = source_tokens + issue_tokens + PROMPT_FRAME_TOKENS
    budget = int(n_ctx * (1.0 - CONTEXT_MARGIN_FRACTION))

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
            f"{budget} tok ({int(CONTEXT_MARGIN_FRACTION * 100)}% of n_ctx={n_ctx} "
            "reserved for per-attempt feedback growth)"
        )

    return ContextVerdict(
        context_ok=not reasons,
        reasons=reasons,
        estimated_prompt_tokens=estimated,
        budget_tokens=budget,
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source or "unknown",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# the server's own `n_ctx`
# ---------------------------------------------------------------------------
def server_context_window() -> tuple[int, str]:
    """``n_ctx`` the way ``llm::context_window_tokens`` resolves it.

    Order: ``VANTA_LLM_CONTEXT_TOKENS`` (wins), else the server's ``/props``,
    else 65536. Same order as the Rust, so the precheck and the run-time guard
    agree about the window they are enforcing.
    """
    override = os.environ.get("VANTA_LLM_CONTEXT_TOKENS", "").strip()
    if override:
        try:
            value = int(override)
        except ValueError as exc:
            raise ContextError(
                f"VANTA_LLM_CONTEXT_TOKENS={override!r} is not an integer"
            ) from exc
        if value > 0:
            return value, "VANTA_LLM_CONTEXT_TOKENS"

    probed = probe_props_n_ctx()
    if probed is not None:
        return probed, "server /props"
    return FALLBACK_CONTEXT_TOKENS, "fallback (server unreachable)"


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
    lines = [
        "# Context sizes — every candidate, measured",
        "",
        f"Estimator: `bytes / {BYTES_PER_TOKEN_ESTIMATE}` + {PROMPT_FRAME_TOKENS} tok of "
        f"prompt frame. Budget: {int((1 - CONTEXT_MARGIN_FRACTION) * 100)} % of `n_ctx` "
        f"(margin {int(CONTEXT_MARGIN_FRACTION * 100)} %).",
        "",
        "The cap that decides most of these is Vanta's own source walk "
        f"(`MAX_FILE_BYTES={MAX_FILE_BYTES}`, `MAX_TOTAL_SOURCE_BYTES={MAX_TOTAL_SOURCE_BYTES}`); "
        "a bug is rejected when the walk would *truncate* it, because the Creator would "
        "then be shown an arbitrary subset and the bug's own file may not be in it.",
        "",
        "| bug | role | verdict | n_ctx | src B (in/eligible) | files (in/eligible) | skipped (size/budget) | est. prompt tok | budget tok | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, role, verdict in rows:
        stats = verdict.stats
        lines.append(
            "| {key} | {role} | {verdict} | {n_ctx} ({src}) | {b_in}/{b_all} | {f_in}/{f_all} | "
            "{s_size}/{s_budget} | {est} | {budget} | {reasons} |".format(
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
