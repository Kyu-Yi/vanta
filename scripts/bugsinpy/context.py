"""Context-budget precheck (design §1.6) — ``prepare`` rejects, it does not warn.

Vanta no longer hands a role "the first ~262 KB of the repo". It **selects** the
files the role is shown (`vanta::context`): the patch's own touched files and
their one-hop imports in review mode, the files the *issue text* names in repair
mode, filled greedily to the role's token budget. The old 256 KB/64 KB caps
applied *during* a `read_dir`-ordered walk and silently dropped whatever did not
fit, so on ``matplotlib`` (11.8 MB) the bug's own file usually was not in the
prompt and the run failed for a reason that had nothing to do with the model.
``prepare --measure-only`` rejected 10 of 11 dev/reserve bugs for exactly that
(``source_truncated_by_cap``).

The gate now measures **that selection**, not the whole repo, and it measures it
by running the SAME code the run runs:

    vanta_m3 --select-context --json --repo TREE --issue ISSUE \\
             --select-role ROLE --select-budget N [--patch PATCH]

[VANTA-WHY] Shelling out is the point. This module used to re-derive the walk
(extensions, caps, skip dirs) and the token estimate in Python, which meant the
gate could pass a bug whose prompt the run then refused — or refuse one the run
would have accepted — because the two "mirrors" were only ever asserted to agree
by a test. A gate that measures a different prompt from the one the run sends is
worse than no gate, so the walk and the ranking now have ONE implementation and
this module parses its JSON. It no longer walks a tree at all.

[VANTA-WHY] ``--select-context`` takes exactly the inputs a real run has — the
tree, the issue text, the patch under review (if any), the role and the budget —
and nothing else. It has no flag for ``FAIL_TO_PASS``, a hidden test, the fix
commit, the fix diff or ``bug.info``: a gate that could read the answer would size
the prompt around it. The harness's half of that rule is ``prepare``/``trees``
keeping the hidden tests out of the tree in the first place.

The budget is not "some fraction of the window":

    budget = n_ctx − output reserve − chat overhead − margin       (§1.6.2)

The first two terms are what Vanta's own run-time guard subtracts
(``context_budget_exceeded`` in ``src/llm/mod.rs``, whose ``max_tokens`` is the
request body's — the ``"creator"`` row of ``SAMPLING_SCHEDULE`` there, put into
the body by ``with_sampling``), so the precheck's boundary sits on the guard's
boundary; the margin is the harness's addition, reserved for feedback growth
across attempts. This is the one measurement the harness still owns, because the
*window* is the caller's (§1.6.1) and it is passed into the subcommand as
``--select-budget``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .records import RecordError, field_of, strings_of

# ---- the role label the gate selects for ---------------------------------
SELECT_ROLE = "creator+attacker+auditor+bate+diagnostician"
"""The role string the repair-mode selection is made for.

[VANTA-WHY] It must equal ``REPAIR_SOURCE_ROLES`` in ``src/pipeline/driver.rs``,
and a test asserts that. The label does not change WHAT is selected, but it is
written into the selection (and so into the run's ``context.select`` row), and a
gate that labelled the prompt differently from the run would make two identical
prompts look like two experiments.
"""

# ---- harness constants, all documented in §1.6 ----------------------------
CONTEXT_MARGIN_FRACTION = 0.25
"""Fraction of ``n_ctx`` the prompt must leave free. Not a magic number: the
Creator's feedback grows every attempt (test output is appended), and a prompt
that fits attempt 1 but not attempt 3 dies mid-run rather than at precheck."""

OUTPUT_RESERVE_TOKENS = 32_768
"""What the Creator asks the server to generate (the ``"creator"`` row of
``SAMPLING_SCHEDULE`` in ``src/llm/mod.rs``, which is what ``with_sampling`` puts
in the request body). The window is shared: prompt and completion come out of
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


# ---------------------------------------------------------------------------
# the selection — what the run's own selector returned, parsed
# ---------------------------------------------------------------------------
@dataclass
class SelectedFile:
    """One file in the selection: what it is, why it was picked, what it cost."""

    path: str
    score: int
    reasons: list[str]
    bytes: int
    tokens: int
    shown: str
    #: §1.6.4 — per signal, the rule that matched it, keyed by the reason's own
    #: name: ``{"path": "exact", "identifier": "camel"}``. A reason says which input
    #: named a file, not which of the rules inside that input did — `path` alone
    #: covers five — so this is what lets the audit print WHY a file ranked.
    #:
    #: Defaulted rather than required, unlike `Selection.grounded`: it is display
    #: only (no weight and no gate reads it), so a selection recorded before the
    #: selector recorded rules stays readable, and the audit prints the signal
    #: without its rule rather than refusing a selection it can already judge.
    rules: dict[str, str] = field(default_factory=dict)
    #: §1.6.9 — the file's 1-based rank in the selection, its place in score order.
    #: The array is already in that order, so this restates it; it is recorded
    #: because a report is read on its own, by analyses that sort the list their own
    #: way. ``0`` means the report did not record one (written before §1.6.9).
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "reasons": list(self.reasons),
            "rules": dict(self.rules),
            "bytes": self.bytes,
            "tokens": self.tokens,
            "shown": self.shown,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "SelectedFile":
        owner = cls.__name__
        return cls(
            path=str(field_of(payload, "path", owner=owner, where=where)),
            score=int(field_of(payload, "score", owner=owner, where=where)),
            reasons=strings_of(payload, "reasons", owner=owner, where=where),
            rules=rules_of(payload),
            bytes=int(field_of(payload, "bytes", owner=owner, where=where)),
            tokens=int(field_of(payload, "tokens", owner=owner, where=where)),
            shown=str(field_of(payload, "shown", owner=owner, where=where)),
            rank=rank_of(payload),
        )


def rules_of(payload: object) -> dict[str, str]:
    """``payload["rules"]`` as ``{signal: rule}``, or ``{}`` when it is absent.

    One of the two tolerant fields of this record, and deliberately so: §1.6.4
    added it so the ranking audit can say why a file ranked, nothing acts on it,
    and a report written before the selector recorded rules is still one the audit
    can read. Everything else here goes through :func:`field_of`, because everything
    else is a number the gate decides on.
    """
    if not isinstance(payload, Mapping):
        return {}
    rules = payload.get("rules")
    if not isinstance(rules, Mapping):
        return {}
    return {str(signal): str(rule) for signal, rule in rules.items()}


def rank_of(payload: object) -> int:
    """``payload["rank"]`` — the file's place in score order (§1.6.9) — or ``0``.

    Tolerant for the same reason as :func:`rules_of`: rank is recorded for readers
    of the report and nothing acts on it, so a report written before §1.6.9 does
    not become unreadable. ``0`` says plainly that no rank was recorded, rather
    than inventing one from the array's order, which is a reader's decision and not
    this function's.
    """
    if not isinstance(payload, Mapping):
        return 0
    rank = payload.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        return 0
    return rank


@dataclass
class Selection:
    """``vanta_m3 --select-context --json``'s answer, field for field.

    [VANTA-WHY] Every number the gate acts on comes from the run's own selector,
    including ``prompt_tokens_estimate`` and ``fits``. Recomputing either here
    would put the gate's boundary back on a second implementation — the failure
    this whole module was reduced to a shell-out to avoid.
    """

    role: str
    mode: str
    budget_tokens: int
    files: list[SelectedFile] = field(default_factory=list)
    #: §1.6.10 — whether the selection carries content grounded in the run's own
    #: inputs, as the selector's own reason vocabulary has it. The gate refuses on
    #: this and never re-derives it: a reason that named a file without grounding
    #: its body would be refused here without this module knowing its name.
    grounded: bool = False
    tree_count: int = 0
    selected_bytes: int = 0
    selected_tokens: int = 0
    tree_bytes: int = 0
    tree_tokens: int = 0
    dropped_for_budget: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    issue_tokens: int = 0
    prompt_frame_tokens: int = 0
    prompt_tokens_estimate: int = 0
    fits: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "mode": self.mode,
            "budget_tokens": self.budget_tokens,
            "file_count": len(self.files),
            "files": [file.to_dict() for file in self.files],
            "grounded": self.grounded,
            "tree_count": self.tree_count,
            "selected_bytes": self.selected_bytes,
            "selected_tokens": self.selected_tokens,
            "tree_bytes": self.tree_bytes,
            "tree_tokens": self.tree_tokens,
            "dropped_for_budget": list(self.dropped_for_budget),
            "skipped": [{"path": path, "reason": reason} for path, reason in self.skipped],
            "issue_tokens": self.issue_tokens,
            "prompt_frame_tokens": self.prompt_frame_tokens,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "fits": self.fits,
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "Selection":
        owner = cls.__name__
        files = field_of(payload, "files", owner=owner, where=where)
        if not isinstance(files, list):
            raise RecordError(
                f"{owner} in {where}: field 'files' is {type(files).__name__}, not a list"
            )
        skipped = field_of(payload, "skipped", owner=owner, where=where)
        if not isinstance(skipped, list):
            raise RecordError(
                f"{owner} in {where}: field 'skipped' is {type(skipped).__name__}, not a list"
            )
        return cls(
            role=str(field_of(payload, "role", owner=owner, where=where)),
            mode=str(field_of(payload, "mode", owner=owner, where=where)),
            budget_tokens=int(field_of(payload, "budget_tokens", owner=owner, where=where)),
            files=[
                SelectedFile.from_dict(item, where=f"{where} files[]") for item in files
            ],
            # Required, not defaulted (§1.6.10): a selector that does not answer
            # this is a gate that cannot judge, and reading a missing `grounded`
            # as `false` would refuse every bug the moment the field dropped off
            # the wire format.
            grounded=bool(field_of(payload, "grounded", owner=owner, where=where)),
            tree_count=int(field_of(payload, "tree_count", owner=owner, where=where)),
            selected_bytes=int(
                field_of(payload, "selected_bytes", owner=owner, where=where)
            ),
            selected_tokens=int(
                field_of(payload, "selected_tokens", owner=owner, where=where)
            ),
            tree_bytes=int(field_of(payload, "tree_bytes", owner=owner, where=where)),
            tree_tokens=int(field_of(payload, "tree_tokens", owner=owner, where=where)),
            dropped_for_budget=strings_of(
                payload, "dropped_for_budget", owner=owner, where=where
            ),
            skipped=[
                (
                    str(field_of(item, "path", owner=owner, where=where)),
                    str(field_of(item, "reason", owner=owner, where=where)),
                )
                for item in skipped
            ],
            issue_tokens=int(field_of(payload, "issue_tokens", owner=owner, where=where)),
            prompt_frame_tokens=int(
                field_of(payload, "prompt_frame_tokens", owner=owner, where=where)
            ),
            prompt_tokens_estimate=int(
                field_of(payload, "prompt_tokens_estimate", owner=owner, where=where)
            ),
            fits=bool(field_of(payload, "fits", owner=owner, where=where)),
        )


def select_context(
    tree: Path,
    issue_text: str,
    *,
    vanta_bin: Path,
    budget_tokens: int,
    role: str = SELECT_ROLE,
    patch_path: Optional[Path] = None,
) -> Selection:
    """Run the run's own selector and parse its answer.

    ``vanta_bin`` is the same ``vanta_m3`` the batch's rows will measure, so the
    gate cannot disagree with the run about what a prompt is — and ``prepare``
    refuses to start when that binary is older than the Rust sources (§7.3).
    """
    binary = Path(vanta_bin)
    if not binary.exists():
        raise ContextError(
            f"no vanta_m3 binary at {binary}: the context gate shells out to the "
            "run's own selector, so it cannot run without it. Build it with "
            "`cargo build --release --bin vanta_m3`."
        )

    argv = [
        str(binary),
        "--select-context",
        "--json",
        "--repo",
        str(tree),
        "--select-role",
        role,
        "--select-budget",
        str(budget_tokens),
    ]
    # [VANTA-WHY] The issue text goes through a file because that is how a run
    # receives it (`--issue PATH`) — the selector has no other way in, which is
    # the property the leak test pins.
    with tempfile.TemporaryDirectory() as tmp:
        if issue_text:
            issue_file = Path(tmp) / "issue.md"
            issue_file.write_text(issue_text, encoding="utf-8")
            argv += ["--issue", str(issue_file)]
        if patch_path is not None:
            argv += ["--patch", str(patch_path)]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise ContextError(
                f"`{' '.join(argv)}` failed (exit {exc.returncode}): "
                f"{(exc.stderr or '').strip()}"
            ) from exc
        except OSError as exc:
            raise ContextError(f"could not run {binary}: {exc}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContextError(
            f"{binary} --select-context did not print JSON: {exc}; stdout was "
            f"{completed.stdout[:200]!r}"
        ) from exc
    return Selection.from_dict(payload, where=f"select-context on {tree}")


@dataclass
class ContextVerdict:
    context_ok: bool
    reasons: list[str]
    estimated_prompt_tokens: int
    budget_tokens: int
    n_ctx: int
    n_ctx_source: str
    selection: Selection
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
            "selection": self.selection.to_dict(),
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
            selection=Selection.from_dict(
                field_of(payload, "selection", owner=owner, where=where), where=where
            ),
            output_reserve_tokens=int(
                field_of(payload, "output_reserve_tokens", owner=owner, where=where)
            ),
            margin_tokens=int(field_of(payload, "margin_tokens", owner=owner, where=where)),
        )


def budget_tokens(n_ctx: int) -> tuple[int, int]:
    """``(budget, margin)`` for a window: the prompt budget the precheck enforces.

        budget = n_ctx − output reserve − chat overhead − margin
        margin = 25 % of n_ctx                                     (§1.6.2)

    The output reserve and the chat overhead are the two terms Vanta's own guard
    subtracts before it decides a prompt fits (``context_budget_exceeded``), so
    subtracting them here is what keeps the two boundaries on top of each other.
    The margin is the harness's addition, and it is the reason this precheck is
    stricter than the guard: it is reserved for feedback growth across attempts.

    [VANTA-WHY] This is the *caller's* number now, not a mirror: it is passed to
    ``--select-context`` as ``--select-budget``, so the selector fills the budget
    the harness chose. ``vanta::context::budget_tokens`` computes the same thing
    for the run, and a test asserts the two agree.
    """
    margin = int(n_ctx * CONTEXT_MARGIN_FRACTION)
    return n_ctx - OUTPUT_RESERVE_TOKENS - CHAT_OVERHEAD_TOKENS - margin, margin


def precheck(
    tree: Path,
    issue_text: str,
    *,
    n_ctx: Optional[int] = None,
    n_ctx_source: Optional[str] = None,
    vanta_bin: Optional[Path] = None,
    role: str = SELECT_ROLE,
    patch_path: Optional[Path] = None,
    selection: Optional[Selection] = None,
) -> ContextVerdict:
    """Measure the *selection* for a role against the window the run will use.

    ``selection`` lets a caller that already ran the selector reuse its answer
    (``prepare`` measures the pre-strip tree first, then the final one, and only
    wants the shell-out when it will act on the number). Otherwise ``vanta_bin``
    is required — there is no Python fallback, by design.
    """
    if n_ctx is None:
        # ``explicit_context_window``, not ``server_context_window``: the gate
        # refuses to run against the 65536 fallback (see that function). A caller
        # that forgot to thread an explicit window therefore fails here instead of
        # producing a verdict nobody should act on.
        n_ctx, n_ctx_source = explicit_context_window()

    budget, margin = budget_tokens(n_ctx)

    if selection is None:
        if vanta_bin is None:
            raise ContextError(
                "the context gate measures the run's own selection, so it needs "
                "either the vanta_m3 binary (`vanta_bin=`) or an already-computed "
                "`selection=` — it has no walk of its own to fall back on."
            )
        selection = select_context(
            tree,
            issue_text,
            vanta_bin=vanta_bin,
            budget_tokens=budget,
            role=role,
            patch_path=patch_path,
        )

    estimated = selection.prompt_tokens_estimate

    reasons: list[str] = []
    if budget <= 0:
        reasons.append(
            f"no_prompt_budget: n_ctx={n_ctx} ({n_ctx_source or 'unknown'}) is smaller "
            f"than the {OUTPUT_RESERVE_TOKENS} tok the Creator needs for its completion "
            "plus the margin; no prompt fits this window"
        )
    if estimated > budget:
        reasons.append(
            f"prompt_exceeds_budget: ~{estimated} tok selected vs a budget of "
            f"{budget} tok (= n_ctx {n_ctx} − {OUTPUT_RESERVE_TOKENS} output − "
            f"{CHAT_OVERHEAD_TOKENS} overhead − {margin} margin)"
        )
    if selection.dropped_for_budget:
        # Not the old whole-repo truncation: this is the selection itself failing
        # to fit, which means even an outline of a file the ranking chose was left
        # out. The bug's own file is exactly what a ranking puts first, so this is
        # the case where a run would be shown a different prompt from the one the
        # ranking decided on.
        reasons.append(
            "selection_dropped_for_budget: "
            f"{len(selection.dropped_for_budget)} selected file(s) did not fit the "
            f"{budget} tok budget even as an outline (e.g. "
            f"{', '.join(selection.dropped_for_budget[:3])}); the run would be shown "
            "a smaller context than its ranking chose"
        )
    if not selection.grounded:
        # [VANTA-WHY] The verdict is the SELECTOR's own answer (§1.6.10):
        # `grounded` is the selector's reason vocabulary asking whether any
        # selected file's content is grounded in this run's own inputs, so a
        # reason that named a file without grounding its body is refused here
        # without this module learning the reason's name — the same reason the
        # walk lives in the binary and not here (§1.6.2 item 5).
        #
        # What it refuses is a listing, not a context: the prompt would be the
        # repository file tree — ``tree_count`` paths — and nothing else. That is
        # what the second measure-only run measured for all 11 dev bugs, and why
        # this reason exists: a listing always fits the budget, so the two checks
        # above pass a bug that was never shown any source (`selected_bytes ==
        # tree_bytes` exactly, §1.6.3). A selection is refused for the thing it is
        # missing, not for its size.
        reasons.append(
            "no_issue_grounded_selection: the selection carries no file grounded in "
            "this run's own inputs — only the repository file tree "
            f"({selection.tree_count} path(s), {selection.tree_bytes} B). A listing "
            "of paths says where code is, not what it does, so a role shown only "
            "that has no context to work from"
        )

    return ContextVerdict(
        context_ok=not reasons,
        reasons=reasons,
        estimated_prompt_tokens=estimated,
        budget_tokens=budget,
        n_ctx=n_ctx,
        n_ctx_source=n_ctx_source or "unknown",
        selection=selection,
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
    modes = {v.selection.mode for _, _, v in rows}
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
        "A bug passes only if the prompt Vanta SELECTS for it fits the window with room to spare:",
        "",
        "```",
        "budget = n_ctx − output reserve − chat overhead − margin",
        f"       = n_ctx − {OUTPUT_RESERVE_TOKENS} − {CHAT_OVERHEAD_TOKENS} "
        f"− {int(CONTEXT_MARGIN_FRACTION * 100)} % of n_ctx",
        "```",
        "",
        f"- **output reserve {OUTPUT_RESERVE_TOKENS} tok** — the `\"creator\"` row of "
        "`SAMPLING_SCHEDULE` in `src/llm/mod.rs`. Prompt and completion share `n_ctx`.",
        f"- **chat overhead {CHAT_OVERHEAD_TOKENS} tok** — `CHAT_OVERHEAD_TOKENS` in "
        "`src/llm/mod.rs`.",
        f"- **margin {int(CONTEXT_MARGIN_FRACTION * 100)} %** — §1.6.2. The Creator's "
        "feedback grows every attempt, so a prompt that only just fits attempt 1 dies "
        "at attempt 3.",
        "",
        "## What is measured",
        "",
        "The selection the run itself makes, obtained by running the run's own "
        "selector (`vanta_m3 --select-context --json`) over the tree and the issue "
        "text. In **repair** mode the files are ranked from the issue text (paths it "
        "names, traceback frames, symbols it mentions that a file defines); in "
        "**review** mode they are the patch's own touched files, their one-hop imports "
        "and the visible tests that exercise them. The token estimate, the prompt "
        "frame and the `fits` verdict are the subcommand's — the harness owns only "
        "the window it passes in as `--select-budget`, so the gate cannot disagree "
        "with the run about what a prompt is.",
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
            f"i.e. about **{budget * 4 // 1024} KB** of source before the prompt frame "
            "— the budget no longer coincides with a whole-repo cap, because there is "
            "no whole-repo cap: the caps apply to the SELECTED set (§1.6.2).",
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
    if len(modes) > 1:
        lines += [
            "More than one selection mode was measured in this run: "
            + ", ".join(sorted(modes))
            + ". `prepare` selects without a patch (repair mode), because at this "
            "stage there is no patch; a review row's selection is a different "
            "prompt and is not gated here.",
            "",
        ]

    lines += [
        "A bug is rejected when the *selected* prompt does not fit the budget, or "
        "when the selection itself had to drop a file it had ranked — the case where "
        "the run would be shown less than the ranking chose.",
        "",
        "| bug | role | verdict | n_ctx | mode | sel B | files | tree | dropped | pruned | est tok | budget tok | headroom tok | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, role, verdict in rows:
        selection = verdict.selection
        lines.append(
            "| {key} | {role} | {verdict} | {n_ctx} ({src}) | {mode} | {b_sel} | "
            "{f_sel} | {tree} | {dropped} | {pruned} | {est} | {budget} | {head} | "
            "{reasons} |".format(
                key=key,
                role=role,
                verdict="ok" if verdict.context_ok else "**reject**",
                n_ctx=verdict.n_ctx,
                src=verdict.n_ctx_source,
                mode=selection.mode,
                b_sel=selection.selected_bytes,
                f_sel=len(selection.files),
                tree=selection.tree_count,
                dropped=len(selection.dropped_for_budget),
                pruned=len(selection.skipped),
                est=verdict.estimated_prompt_tokens,
                budget=verdict.budget_tokens,
                head=verdict.headroom_tokens,
                reasons="; ".join(verdict.reasons) or "—",
            )
        )

    lines += ["", "## Selected files per candidate", ""]
    for key, _role, verdict in rows:
        selection = verdict.selection
        if not selection.files:
            lines.append(f"- **{key}**: no file selected")
            continue
        listed = ", ".join(
            f"`{file.path}` ({file.score}, {'+'.join(file.reasons)})"
            for file in selection.files[:8]
        )
        more = "" if len(selection.files) <= 8 else f", … {len(selection.files) - 8} more"
        lines.append(f"- **{key}**: {listed}{more}")
    lines.append("")
    return "\n".join(lines)
