"""The issue text's source: the project's own issue report, not the fix commit (§2.4).

[VANTA-WHY] The harness used to build the issue text out of
``git show -s --format=%B <fixed_commit>`` — the message of the commit that *fixed*
the bug. That text is written by the person who solved it, after they solved it, and
on the dev split it names the fix location outright for 7 of 11 bugs
(``is_string_dtype``, ``allowed_domains``, ``conllu2json``, ``beam_dataflow``,
``named_response_middlware``, ``response_model_exclude_defaults``, ``monkey patches to
renderer``). The leak gate cannot take those words back: it strips a token only when
it appears in the fix diff or the hidden test **and nowhere in the tree the Creator may
read**, and a symbol the developer fixed is by definition present in the buggy tree. A
gate whose whole rule is "this came from the answer" is not a gate against "this came
from the answer's author". The text the model should read is the *report* the bug was
filed against, which is what a real repair run gets.

The resolution is therefore a host-side, cached, one-off step:

1. Read the fix commit's message and pull the numbers it references
   (:func:`referenced_numbers`) — ``closes/fixes/resolves #N``, ``Merge pull request
   #N``, ``Issue N``, and the squash-merge ``(#N)``.
2. Ask the GitHub API whether that number is an issue or a pull request.
3. For a pull request, follow it to the issue it closes — its body's own closing
   keywords, or the cross-references on its timeline.
4. **Never** use the pull request's body or diff as issue text. The request named that
   specifically and the reason is the same as the one above: a pull request *is* the
   fix, so its prose is the answer's prose.

[VANTA-WHY] No ``gh`` and no ``git`` fallback. ``gh`` is a second implementation of
the same API behind a binary that may or may not be installed and whose output format
is not the API's; ``git`` cannot answer the question at all (a commit message is not a
web reference) and every ``git``-shaped shortcut here would be a way for the issue text
to come from somewhere other than the API response that was cached and hashed. One
implementation, one record.

[VANTA-WHY] ``VANTA_GITHUB_TOKEN`` is read from the environment and used read-only. It
is never written to the cache, never written to a log, and never part of a cache key —
the corpus is public repositories, so the token buys rate limit and nothing else. An
unauthenticated run is the default and works; it is simply slower, which is what the
on-disk cache and the rate-limit backoff are for.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from . import issue as issue_mod
from .records import RecordError
from .split import sha256_text

GITHUB_API = "https://api.github.com"
TOKEN_ENV = "VANTA_GITHUB_TOKEN"
USER_AGENT = "vanta-bugsinpy-issue-source"
DEFAULT_TIMEOUT_S = 30
MAX_BACKOFF_S = 3600
"""Longest a rate-limit backoff will wait before giving up and saying so."""

ISSUES_DIR = Path("plans/bugsinpy_issues")
"""Where the frozen issue reports are committed (§2.4 item 2)."""

FROZEN_HEADER_KEYS = (
    "issue",
    "resolved",
    "fetched",
    "fix-commit-date",
    "comments-included",
    "gate-tree-sha256",
    "gate-answer-sha256",
    "raw-sha256",
    "content-sha256",
    "raw-hints-sha256",
    "hints-sha256",
)
"""Every header line :func:`load_frozen` needs before it will hand out a text.

``raw-sha256`` and ``content-sha256`` are the pair the artifact is *for*: the raw report
is kept beside the gated text so a reader can see what the gate took out, and the gated
hash is what every stage checks the file against. ``raw-hints-sha256`` and ``hints-sha256``
are the same pair for the comments (§2.4.4), and they are in this list rather than optional
for a reason: the hints are an ablation arm, and an arm whose text is not covered by the
hash is one a hand edit can move without the file refusing itself.

The two ``gate-*-sha256`` lines are the gate's *inputs*, not its output (§2.4): the token
maps the freeze step computed from the buggy commit's source and from the fix diff plus
the test files ``bug.info`` declares."""

SOURCE_ISSUE = "issue"
SOURCE_VIA_PR = "via-pr"
SOURCE_NONE = "none"


class IssueSourceError(RuntimeError):
    """The issue report could not be resolved or read."""


# ---------------------------------------------------------------------------
# step 1 — what the fix commit message references
# ---------------------------------------------------------------------------
CROSS_REPO_RE = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)#(\d+)\b"
)
CLOSING_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]*#(\d+)", re.IGNORECASE
)
MERGE_RE = re.compile(r"\bMerge pull request #(\d+)\b")
ISSUE_WORD_RE = re.compile(r"\bissues?\s*#?(\d+)\b", re.IGNORECASE)
PAREN_RE = re.compile(r"\(#(\d+)\)")


@dataclass(frozen=True)
class RefCandidate:
    """One number a commit message points at, with the phrasing that pointed at it."""

    number: int
    kind: str
    owner: Optional[str] = None
    repo: Optional[str] = None

    @property
    def is_cross_repo(self) -> bool:
        return self.owner is not None


def _masked(message: str, spans: list[tuple[int, int]]) -> str:
    """``message`` with the given spans blanked, so a later pattern cannot re-read them."""
    if not spans:
        return message
    chars = list(message)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)


def referenced_numbers(message: str, *, owner: str, repo: str) -> list[RefCandidate]:
    """Numbers the fix commit's message references, most explicit phrasing first.

    Cross-repository references (``oh-my-fish/plugin-thefuck#11``) are recognised and
    reported, but only as candidates for *their own* repository: the text the model
    reads has to be this project's own report, and a number that belongs to a different
    project cannot be resolved against this one. They are also masked out of the scan
    for single-repository patterns, so ``plugin-thefuck#11`` cannot come back as
    ``thefuck#11``.

    Order is the point: ``Fix #869`` beats ``(#917)`` for ``thefuck/3`` because the
    closing keyword is what the committer meant, and the parenthesis is the pull
    request the commit was squash-merged through.
    """
    cross = [(m.start(), m.end()) for m in CROSS_REPO_RE.finditer(message)]
    plain = _masked(message, cross)

    out: list[RefCandidate] = []
    seen: set[tuple[Optional[str], Optional[str], int]] = set()

    def add(cand: RefCandidate) -> None:
        key = (cand.owner, cand.repo, cand.number)
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    for match in CROSS_REPO_RE.finditer(message):
        # Masked out of the single-repo patterns; kept as a candidate for its own repo.
        add(
            RefCandidate(
                number=int(match.group(3)),
                kind="cross-repo",
                owner=match.group(1),
                repo=match.group(2),
            )
        )
    for match in CLOSING_RE.finditer(plain):
        add(RefCandidate(number=int(match.group(1)), kind="closing", owner=owner, repo=repo))
    for match in MERGE_RE.finditer(plain):
        add(RefCandidate(number=int(match.group(1)), kind="merge", owner=owner, repo=repo))
    for match in ISSUE_WORD_RE.finditer(plain):
        add(RefCandidate(number=int(match.group(1)), kind="issue-word", owner=owner, repo=repo))
    for match in PAREN_RE.finditer(plain):
        add(RefCandidate(number=int(match.group(1)), kind="squash", owner=owner, repo=repo))
    return out


CLOSING_NUMBERS_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]*#(\d+)", re.IGNORECASE
)


def closing_numbers(text: str) -> list[int]:
    """Numbers a body says it closes, in order, deduplicated."""
    out: list[int] = []
    for match in CLOSING_NUMBERS_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in out:
            out.append(number)
    return out


# ---------------------------------------------------------------------------
# step 2/3 — the API client: cache first, one backoff, no gh, no git
# ---------------------------------------------------------------------------
def owner_repo(github_url: str) -> tuple[str, str]:
    """``https://github.com/owner/repo`` (or the same with a ``.git``) → ``(owner, repo)``."""
    text = (github_url or "").strip().rstrip("/")
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2 or "github.com" not in parts:
        raise IssueSourceError(
            f"{github_url!r} is not a github.com repository URL; the issue resolver has "
            "no other host to ask"
        )
    return parts[-2], parts[-1]


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: Any


class GitHubClient:
    """A tiny, cached, rate-limit-aware GET client for the public GitHub API.

    ``transport`` is the seam every test uses: it takes a URL path and the request
    headers and returns an :class:`HttpResponse`, so no test reaches the network and
    the caching/backoff logic is exercised against a scripted API rather than a live
    one. The default transport is :mod:`urllib` — the standard library, because the
    harness has no HTTP dependency and does not need one for a handful of GETs.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        token: Optional[str] = None,
        transport: Optional[Callable[[str, dict[str, str]], HttpResponse]] = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        max_backoff_s: int = MAX_BACKOFF_S,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        # Read once, held in memory, never written anywhere. The environment is read
        # here rather than in `default_client` so that a caller that passes a token
        # explicitly (a test) can never pick the environment up by accident.
        self._token = token if token is not None else (os.environ.get(TOKEN_ENV) or None)
        self._transport = transport or self._urllib_transport
        self._sleep = sleep
        self._timeout_s = timeout_s
        self._max_backoff_s = max_backoff_s
        self._now = now

    # -- public -----------------------------------------------------------
    def get(self, path: str) -> Any:
        """The cached JSON body for ``path`` (an API path like ``/repos/o/r/issues/1``)."""
        cached = self._read_cache(path)
        if cached is not None:
            return cached
        response = self._request(path)
        if response.status == 404:
            raise IssueSourceError(f"GET {path}: 404 — nothing at that reference")
        if response.status == 403 or response.status == 429:
            response = self._backoff(path, response)
        if response.status >= 400:
            raise IssueSourceError(f"GET {path}: HTTP {response.status}: {_short(response.body)}")
        self._write_cache(path, response.body)
        return response.body

    def get_or_none(self, path: str) -> Optional[Any]:
        """``get``, with a 404 read as "no such reference" instead of an error."""
        try:
            return self.get(path)
        except IssueSourceError as exc:
            if "404" in str(exc):
                return None
            raise

    def get_list(self, path: str) -> list[Any]:
        body = self.get(path)
        if not isinstance(body, list):
            raise IssueSourceError(f"GET {path}: expected a JSON array, got {type(body).__name__}")
        return body

    # -- cache ------------------------------------------------------------
    def cache_path(self, path: str) -> Path:
        return self.cache_dir / (sha256_text(path)[:32] + ".json")

    def _read_cache(self, path: str) -> Optional[Any]:
        target = self.cache_path(path)
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IssueSourceError(f"the cached response for {path} is unreadable: {exc}") from exc
        if payload.get("path") != path:
            raise IssueSourceError(
                f"the cache file {target} holds {payload.get('path')!r}, not {path!r}: "
                "the cache is keyed by a hash of the path, so this is a collision or a "
                "hand-edited file, and either way its body is not the answer to this "
                "question"
            )
        return payload.get("body")

    def _write_cache(self, path: str, body: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_path(path)
        payload = {"path": path, "cached_at": _utc_now(), "body": body}
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    # -- transport / backoff ---------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request(self, path: str) -> HttpResponse:
        try:
            return self._transport(path, self._headers())
        except IssueSourceError:
            raise
        except Exception as exc:  # transport failures are the caller's to see
            raise IssueSourceError(f"GET {path} failed: {exc}") from exc

    def _backoff(self, path: str, response: HttpResponse) -> HttpResponse:
        """Wait out a rate limit once, then retry — or say exactly what is blocking."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        retry_after = response.headers.get("Retry-After")
        wait_s = 0.0
        if retry_after:
            wait_s = float(retry_after)
        elif remaining == "0" and reset:
            wait_s = max(0.0, float(reset) - self._now())
        if wait_s <= 0 or wait_s > self._max_backoff_s:
            limit = " (no token: the unauthenticated limit is 60 requests per hour)" if not self._token else ""
            raise IssueSourceError(
                f"GET {path}: rate limited (HTTP {response.status}) and the wait would be "
                f"{wait_s:.0f}s, above the {self._max_backoff_s}s ceiling{limit}. Set "
                f"{TOKEN_ENV} to a token for a higher limit, or re-run later: the "
                "responses already fetched are cached."
            )
        self._sleep(wait_s)
        return self._request(path)

    def _urllib_transport(self, path: str, headers: dict[str, str]) -> HttpResponse:
        request = urllib.request.Request(f"{GITHUB_API}{path}", headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as handle:
                raw = handle.read()
                status = handle.status
                response_headers = {k: v for k, v in handle.headers.items()}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            response_headers = {k: v for k, v in (exc.headers or {}).items()}
        except urllib.error.URLError as exc:
            raise IssueSourceError(f"GET {path}: {exc.reason}") from exc
        text = raw.decode("utf-8", errors="replace")
        if "application/json" not in response_headers.get("Content-Type", ""):
            body: Any = text
        else:
            body = json.loads(text or "null")
        return HttpResponse(status=status, headers=response_headers, body=body)


def default_client(cache_dir: Path, **kwargs: Any) -> GitHubClient:
    return GitHubClient(cache_dir, **kwargs)


def _short(body: Any, limit: int = 300) -> str:
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return text[:limit]


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IssueComment:
    author: str
    created_at: str
    body: str


@dataclass(frozen=True)
class GateRecord:
    """The leak gate's result *and* its inputs, as the freeze step computed them (§2.4).

    [VANTA-WHY] The gate runs at freeze time and never again. That is the whole reason
    ``prepare``, a measure-only run and the run can hand Vanta byte-identical text: the
    gate's own inputs come from git alone — the buggy commit's source for ``tree_tokens``,
    the fix diff plus the test files ``bug.info`` declares for ``answer_tokens`` — so the
    step needs no suite run and a measure-only run (which has no ``FAIL_TO_PASS``) can read
    a text that was gated exactly as a real run's would have been. The two hashes are what
    a later reader compares against, and they are recorded rather than recomputed because
    re-gating would be a second answer to a question that already has one.
    """

    leak: "issue_mod.LeakResult"
    tree_sha256: str
    answer_sha256: str
    files: tuple[str, ...]
    """The test files the answer tokens came from — ``bug.info``'s declaration, resolved."""
    hints: Optional["issue_mod.LeakResult"] = None
    """The same gate over the report's **comments** — SWE-bench's ``hints_text`` (§2.4.4).

    [VANTA-WHY] Stored, gated and **never sent in the default setting**. A comment posted
    before the fix is under no obligation to avoid naming it, and on ``matplotlib/1`` one
    does: a maintainer says what the cause was and points at ``backend_agg.py``'s line
    range. That is the answer, and a report whose ``problem_statement`` is title-plus-body
    cannot carry it — so the comments are a section of their own, gated by the same rule as
    the text a run sends (:func:`issue_mod.gate_hints`) and read by nothing unless
    ``show_hints`` turns them on, which is off and recorded per row (``runner.build_record``).
    ``None`` means the report had no comments at all, which is different from comments that
    gated to nothing.
    """


@dataclass
class IssueReport:
    """The project's own report: title, body, and the comments up to the fix."""

    project: str
    bug_id: int
    source: str
    number: Optional[int] = None
    via_pr: Optional[int] = None
    url: str = ""
    title: str = ""
    body: str = ""
    comments: list[IssueComment] = field(default_factory=list)
    fetched_at: str = ""
    fix_commit_date: str = ""
    reason: str = ""
    """Why the source is ``none``; empty otherwise."""
    gate: Optional[GateRecord] = None
    """The gated text and the inputs it was gated against. Set before the write."""
    refrozen_at: str = ""
    """When an explicit refreeze replaced an artifact that was already on disk (§2.4.3)."""

    @property
    def slug(self) -> str:
        return f"{self.project}_{self.bug_id}"

    def content(self) -> str:
        """The Vanta-visible text: the **title and the body**. Not the comments (§2.4.4).

        [VANTA-WHY] This is SWE-bench's ``problem_statement``: the report as the person who
        filed it stated the problem, before anyone who knew the answer had typed into the
        thread. The comments are resolved with it and stored beside it, but they are
        ``hints_text`` — kept out of the default setting precisely because a fix is often
        named in them — and :meth:`hints` is where they go.

        **The reporter's own bytes.** Line endings folded to ``\\n`` and the newlines the
        framing adds at a part's edges dropped — :func:`_report_text`'s rule, and the only rule
        applied — with every space, tab and blank line *inside* a part left exactly as the API
        returned it. A code block in a report is Python the reporter pasted, and its
        indentation is part of what the Creator has to read: ``sanic/1``'s report carries
        ``    print('1')``, which is not the same statement as `` print('1')`` (§2.4.4).

        Empty for a ``none`` report — there is no text — and the frozen file says so by
        carrying nothing after its content marker.
        """
        parts: list[str] = []
        if self.title.strip():
            parts.append(f"# {self.title.strip()}")
        body = _report_text(self.body)
        if body:
            parts.append(body)
        return "\n\n".join(parts) + "\n" if parts else ""

    def hints(self) -> str:
        """The comments up to the fix's date — gated, stored, and **never shown by default**.

        Written the same way :meth:`content` is, so the two sections are gated, hashed and
        round-tripped by one rule, and empty when the report carries no comments: an empty
        hints section hashes to the empty string like an empty content section does, and the
        reader that finds nothing after the hints marker is reading a report with none, not a
        truncated file.
        """
        if not self.comments:
            return ""
        parts: list[str] = ["## Comments"]
        for comment in self.comments:
            parts.append(f"### {comment.author} — {comment.created_at}")
            text = _report_text(comment.body)
            if text:
                parts.append(text)
        return "\n\n".join(parts) + "\n"

    def resolved_as(self) -> str:
        if self.source == SOURCE_VIA_PR:
            return f"via PR #{self.via_pr}"
        if self.source == SOURCE_ISSUE:
            return "direct"
        return "none"


def resolve(
    client: GitHubClient,
    *,
    project: str,
    bug_id: int,
    github_url: str,
    message: str,
    fix_commit_date: str,
) -> IssueReport:
    """Resolve one bug's report, or return a report whose ``source`` is ``none``.

    Candidates are tried in the order :func:`referenced_numbers` produced. The first
    one that is an issue — directly, or through the pull request that closes it — wins.
    A candidate that does not exist (404) or that is a pull request which closes nothing
    is not an error: the message simply does not point at a report, and ``none`` says
    so. Transport and rate-limit failures **are** errors, because a network problem must
    never be recorded as "this bug has no issue" — that would silently shrink the
    evaluation.
    """
    owner, repo = owner_repo(github_url)
    candidates = referenced_numbers(message, owner=owner, repo=repo)
    same_repo = [c for c in candidates if c.owner == owner and c.repo == repo]
    if not candidates:
        return IssueReport(
            project=project,
            bug_id=bug_id,
            source=SOURCE_NONE,
            fetched_at=_utc_now(),
            fix_commit_date=fix_commit_date,
            reason="the fix commit's message references no issue or pull request number",
        )
    if not same_repo:
        return IssueReport(
            project=project,
            bug_id=bug_id,
            source=SOURCE_NONE,
            fetched_at=_utc_now(),
            fix_commit_date=fix_commit_date,
            reason=(
                "the fix commit's message references only "
                + ", ".join(f"{c.owner}/{c.repo}#{c.number}" for c in candidates)
                + ", which is a different repository: the text has to be this project's own report"
            ),
        )

    for cand in same_repo:
        found = _try_candidate(client, owner=owner, repo=repo, candidate=cand)
        if found is None:
            continue
        number, via_pr = found
        issue = client.get_or_none(f"/repos/{owner}/{repo}/issues/{number}")
        if issue is None:
            continue
        comments = _comments(client, owner=owner, repo=repo, number=number, until=fix_commit_date)
        return IssueReport(
            project=project,
            bug_id=bug_id,
            source=SOURCE_ISSUE if via_pr is None else SOURCE_VIA_PR,
            number=number,
            via_pr=via_pr,
            url=issue.get("html_url") or f"https://github.com/{owner}/{repo}/issues/{number}",
            title=(issue.get("title") or "").strip(),
            body=_report_text(issue.get("body") or ""),
            comments=comments,
            fetched_at=_utc_now(),
            fix_commit_date=fix_commit_date,
        )

    return IssueReport(
        project=project,
        bug_id=bug_id,
        source=SOURCE_NONE,
        fetched_at=_utc_now(),
        fix_commit_date=fix_commit_date,
        reason=(
            "every number the fix commit's message references is either absent or a pull "
            "request that closes no issue: "
            + ", ".join(f"#{c.number} ({c.kind})" for c in same_repo)
        ),
    )


def _try_candidate(
    client: GitHubClient, *, owner: str, repo: str, candidate: RefCandidate
) -> Optional[tuple[int, Optional[int]]]:
    """``(issue number, the PR it came through)``, or ``None`` if this candidate is no use."""
    payload = client.get_or_none(f"/repos/{owner}/{repo}/issues/{candidate.number}")
    if payload is None:
        return None
    if "pull_request" not in payload:
        return candidate.number, None
    via = _issue_closed_by_pr(client, owner=owner, repo=repo, payload=payload)
    return (via, candidate.number) if via is not None else None


def _issue_closed_by_pr(
    client: GitHubClient, *, owner: str, repo: str, payload: dict
) -> Optional[int]:
    """The issue a pull request closes: its body's closing keywords, then its timeline.

    The body is *read* here and only to find a number — it is never the issue text.
    It is read from the payload already in hand rather than from ``/pulls/N``: for a
    pull request the two endpoints carry the same body, and the second request would
    buy nothing but rate limit.
    """
    number = int(payload.get("number") or 0)
    for referenced in closing_numbers(payload.get("body") or ""):
        if referenced == number:
            continue
        target = client.get_or_none(f"/repos/{owner}/{repo}/issues/{referenced}")
        if isinstance(target, dict) and "pull_request" not in target:
            return referenced

    events = client.get_or_none(f"/repos/{owner}/{repo}/issues/{number}/timeline")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "cross-referenced":
            continue
        source = event.get("source") or {}
        if source.get("type") != "issue":
            continue
        referenced = (source.get("issue") or {}).get("number")
        if not isinstance(referenced, int) or referenced == number:
            continue
        return referenced
    return None


def _comments(
    client: GitHubClient, *, owner: str, repo: str, number: int, until: str
) -> list[IssueComment]:
    """Comments created at or before ``until``, in API order.

    The cut is the point: the fix commit's date is the last moment at which the report
    says what the reporter knew. A comment posted after the fix can only be about the
    fix, so it is not part of the report.
    """
    issue = client.get_or_none(f"/repos/{owner}/{repo}/issues/{number}") or {}
    if int(issue.get("comments") or 0) == 0:
        # The count is in the issue payload already; asking for the list costs a request
        # and can only return an empty array.
        return []
    payload = client.get_or_none(f"/repos/{owner}/{repo}/issues/{number}/comments")
    if not isinstance(payload, list):
        return []
    out: list[IssueComment] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        created = str(item.get("created_at") or "")
        if until and created > until:
            continue
        out.append(
            IssueComment(
                author=((item.get("user") or {}).get("login") or "unknown"),
                created_at=created,
                body=_report_text(str(item.get("body") or "")),
            )
        )
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the frozen artifact (§2.4 item 2)
# ---------------------------------------------------------------------------
def frozen_path(issues_dir: Path, project: str, bug_id: int) -> Path:
    return Path(issues_dir) / f"{project}_{bug_id}.md"


CONTENT_MARKER = "<!-- content -->"
RAW_MARKER = "<!-- raw -->"
GATE_MARKER = "<!-- gate -->"
HINTS_MARKER = "<!-- hints -->"
RAW_HINTS_MARKER = "<!-- raw-hints -->"
"""The frozen file's five sections, in this order, each introduced by its own line.

Explicit marker lines rather than blank ones, because a ``none`` report's text is empty
and a reader that split on "the first blank line after the header" could not tell that
from a truncated file. The raw report comes before the gated text so that a human reading
the committed artifact sees what the reporter wrote before seeing what the gate left.

[VANTA-WHY] ``raw-hints``/``hints`` are a second raw/gated pair, and they come last: what a
stage reads is the content section, and a file whose later sections were cut off in a copy
or a paste would lose only the section nothing sends. What they carry — the report's comments,
gated by the same rule — is stored so a with-hints ablation is possible later, and is sent by
nothing in the default setting."""


def _report_text(value: str) -> str:
    """The reporter's own text: line endings folded, and **only** its edge newlines dropped.

    [VANTA-WHY] This is the one rule applied to any part of a report — a title, a body, a
    comment — and it deliberately does almost nothing. ``\\r\\n`` and a lone ``\\r`` become
    ``\\n``, and the newlines that the framing around a part would otherwise duplicate are
    dropped; a space, a tab, a run of blanks and a line's indent are bytes of the text and are
    never touched. ``str.strip()`` — which is what this replaced — also eats a trailing space or
    the leading indentation of a body that opens with a code block, and a code block is Python
    the reporter pasted: ``sanic/1``'s ``    print('1')`` is a different statement as
    `` print('1')`` (§2.4.4).
    """
    return value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _canonical(text: str) -> str:
    """The one form a section body takes — in the file, and in the hash of it.

    Three rules, and they are the *same* rule for the writer and the reader: line endings are
    folded to ``\\n``; the blank lines the framing puts at a section's edges are dropped; and a
    non-empty body ends with exactly one ``\\n``. The first two are :func:`_report_text` — the
    same function that shapes a report's own parts, so the text stored is the text read — and the
    third is only about the file's shape.

    [VANTA-WHY] The fold is what the first dev freeze got wrong, and that failure was silent
    rather than loud. The API returns an issue body with whatever ``\\r\\n`` the reporter's
    editor used; the writer hashed that text as it stood and then wrote it, while the reader
    got it back through a universal-newline ``read_text``, so every line had already lost its
    ``\\r`` by the time it was hashed. Writer and reader therefore disagreed on **every** report
    that contained a carriage return, which is most of them — the pandas report hashes to
    ``2820bbb3…`` with its 89 carriage returns and ``0f9eb3c9…`` without them, the header held
    the first and the reader computed the second. Folding first makes them agree *by rule*
    rather than by both sides happening to do the same thing, and it makes the artifact a file
    that git, an editor and a diff all read the same way.

    [VANTA-WHY] The edge rule removes **newlines only, and never whitespace**. ``render_frozen``
    separates the sections with a blank line, so a section's own first and last character are
    never the ones the writer added as framing; dropping newlines recovers the body exactly.
    Stripping instead — which is what this did at first — also ate a trailing space or tab, and
    that is not framing: it is a byte of the text Vanta is given, and a rule that hides it would
    let a hand-edited artifact verify. ``test_an_edited_gated_text_is_refused_by_its_content_hash``
    is that byte, appended to the last line of a report, and the hash has to catch it.

    This is a rule about the **file**, applied to every section, and it is the *only* rule about
    whitespace anywhere in this path. It is not a licence to reformat: the gate substitutes
    tokens and changes nothing else (§2.4.4), so a section's bytes are the reporter's own.
    """
    folded = _report_text(text)
    return folded + "\n" if folded else ""


def _section(text: str, marker: str) -> tuple[str, str]:
    """``(before, after)`` the first line equal to ``marker``, that line dropped."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return "".join(lines[:index]), "".join(lines[index + 1 :])
    raise IssueSourceError(
        f"the frozen report has no {marker} line — it is not a file this reader may use"
    )


def render_frozen(report: IssueReport) -> str:
    """The artifact's exact bytes: provenance, the gate's record, raw/content, raw-hints/hints.

    [VANTA-WHY] The raw report is stored *beside* the gated text rather than instead of it.
    The gate's removals are only legible next to what was removed, and the paper's question
    (§2.4 item 6) is how much a fix-commit message would have inflated ``localization_hit``
    — a question nobody can answer from a gate report that no longer has the text it gated.
    Nothing after the content marker is read by a stage: ``FrozenIssue.content`` is the
    gated text, and that is what ``prepare``, a measure-only run and the run all send.

    The comments get their own raw/gated pair for the same reason and one more (§2.4.4):
    they are ``hints_text``, they name the answer often enough that ``matplotlib/1``'s
    maintainer comment names a file and a line range, and a section that is never sent still
    has to be gated and hashed — otherwise the ablation arm that later shows it would be
    showing text no gate ever saw.
    """
    if report.gate is None:
        raise IssueSourceError(
            f"{report.project}/{report.bug_id}: the report has no gate record, so it has no "
            "gated text. The frozen file stores the *gated* report (§2.4) — gate the text "
            "before writing it."
        )
    raw = report.content()
    raw_hints = report.hints()
    gated = report.gate.leak.text
    gated_hints = report.gate.hints.text if report.gate.hints else ""
    header = [
        f"# {report.project}/{report.bug_id} — the project's own issue report",
        "",
        f"- issue: {report.url or '—'}",
        f"- resolved: {report.resolved_as()}",
        f"- fetched: {report.fetched_at}",
        f"- fix-commit-date: {report.fix_commit_date}",
        f"- comments-included: {len(report.comments)}",
        f"- gate-tree-sha256: {report.gate.tree_sha256}",
        f"- gate-answer-sha256: {report.gate.answer_sha256}",
        f"- gate-removals: {len(report.gate.leak.removals)}",
        f"- raw-sha256: {sha256_text(_canonical(raw))}",
        f"- content-sha256: {sha256_text(_canonical(gated))}",
        f"- raw-hints-sha256: {sha256_text(_canonical(raw_hints))}",
        f"- hints-sha256: {sha256_text(_canonical(gated_hints))}",
    ]
    if report.refrozen_at:
        header.append(f"- refrozen: {report.refrozen_at}")
    if report.reason:
        header.append(f"- note: {report.reason}")
    gate = {
        "files": list(report.gate.files),
        "ok": report.gate.leak.ok,
        "reason": report.gate.leak.reason,
        "removals": [removal.to_dict() for removal in report.gate.leak.removals],
        "ungated_prose": list(report.gate.leak.ungated_prose),
        # No ``ok``/``reason`` for the hints: nobody asked whether they are actionable,
        # because nothing sends them by default (§2.4.4). ``None`` — not ``{}`` — when the
        # report had no comments, so "none resolved" and "resolved, gate found nothing"
        # stay distinguishable in the record.
        "hints": None
        if report.gate.hints is None
        else {
            "removals": [removal.to_dict() for removal in report.gate.hints.removals],
            "ungated_prose": list(report.gate.hints.ungated_prose),
        },
    }
    parts = [
        "\n".join(header),
        "",
        GATE_MARKER,
        "",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "",
        RAW_MARKER,
        "",
        _canonical(raw),
        CONTENT_MARKER,
        "",
        _canonical(gated),
        RAW_HINTS_MARKER,
        "",
        _canonical(raw_hints),
        HINTS_MARKER,
        "",
        _canonical(gated_hints),
    ]
    return "\n".join(parts) + "\n"


@dataclass
class FrozenIssue:
    """A frozen artifact, read back — the only shape ``prepare`` ever sees."""

    path: Path
    source: str
    url: str
    issue_number: Optional[int]
    via_pr: Optional[int]
    fetched_at: str
    fix_commit_date: str
    content_sha256: str
    title: str
    body: str
    comments: list[IssueComment]
    content: str
    raw: str
    """What the reporter wrote, before the gate — never sent to Vanta."""
    hints: str
    """The gated comments — the section no stage sends unless ``show_hints`` is on (§2.4.4)."""
    raw_hints: str
    """The comments before the gate — the record of what was removed from ``hints``."""
    hints_sha256: str
    """The header's hash of ``hints``, verified on read like ``content_sha256``."""
    hints_leak: "issue_mod.LeakResult"
    """The gate's substitutions in the comments. Its ``ok``/``reason`` were never asked for."""
    gate: "issue_mod.LeakResult"
    """The gate's record: what it removed, and the text it left (``content``)."""
    gate_files: tuple[str, ...]
    """The test files the answer tokens came from, for the F2P drift check (§2.4)."""
    gate_tree_sha256: str
    gate_answer_sha256: str
    reason: str = ""
    """The header's ``note`` — why the source is ``none``, when it is."""
    refrozen_at: str = ""
    """The header's ``refrozen`` date, when an explicit refreeze wrote this file."""

    def resolved_as(self) -> str:
        if self.source == SOURCE_VIA_PR:
            return f"via PR #{self.via_pr}"
        if self.source == SOURCE_ISSUE:
            return "direct"
        return "none"


def write_frozen(path: Path, report: IssueReport) -> Path:
    """Write the artifact — after :func:`parse_frozen` has accepted the exact text it will hold.

    [VANTA-WHY] The render is verified *before* it is written, through the same parser a stage
    reads with. The first dev freeze wrote five artifacts that its own reader refused, and
    nothing said so at the time: the refusal surfaced one stage later, as a gate failure on a
    bug whose file looked fine on disk, which is the slowest possible way to learn that the
    writer and the reader disagreed. Verifying first makes an internally inconsistent artifact
    impossible to create rather than merely detectable, and it leaves no file behind when the
    two do disagree — which is the one thing a later run could not tell from a valid one.

    ``newline=""`` because the bytes this writes are the bytes ``load_frozen`` hashes: a
    platform that translated ``\\n`` on the way out would put the hashes off by one byte per
    line, which is the same defect through a different door (§2.4.3).
    """
    target = Path(path)
    text = render_frozen(report)
    parse_frozen(text, path=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return target


def _header_block(text: str) -> tuple[dict[str, str], str]:
    """``(header, the rest)`` — the ``- key: value`` lines, up to the first blank line."""
    lines = text.splitlines(keepends=True)
    header: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("- ") and ":" in line:
            key, _, value = line[2:].partition(":")
            header[key.strip()] = value.strip()
        elif header and not line.strip():
            index += 1
            break
        index += 1
    return header, "".join(lines[index:])


def load_frozen(path: Path) -> FrozenIssue:
    """Read a frozen artifact from disk, and verify it (:func:`parse_frozen`)."""
    target = Path(path)
    try:
        # ``newline=""``: no translation in either direction, so the bytes hashed are the
        # bytes written. Reading with universal newlines is what let this reader lose one
        # ``\\r`` per line that the writer had hashed (§2.4.3).
        with target.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        raise IssueSourceError(f"cannot read the frozen issue report {target}: {exc}") from exc
    return parse_frozen(text, path=target)


def parse_frozen(text: str, *, path: Path) -> FrozenIssue:
    """Verify and read the artifact's text — the one parser, for the writer and the reader.

    The round trip is *the* guarantee §2.4 item 2 rests on: ``prepare``, a measure-only run
    and the run all read this file, so their issue texts are identical by construction. A
    file whose content no longer hashes to its header is not that file any more, and reading
    it anyway would put an unverifiable text in the prompt. The raw report is verified the
    same way, so the record of what the gate removed cannot be quietly rewritten either.

    Both hashes are taken over :func:`_canonical` of the parsed section, and the string handed
    out as ``content`` is that same canonical form — so "what is hashed" and "what Vanta is
    given" are one object, and not two things that have to be kept in step. The comments are
    parsed, hashed and verified the same way and handed out as ``hints``, which no stage reads
    in the default setting (§2.4.4) — a section that is never sent still has to refuse an edit,
    or the ablation arm would be an unreviewed text under a reviewed name.
    """
    target = Path(path)
    header, body = _header_block(text)
    for key in FROZEN_HEADER_KEYS:
        if key not in header:
            raise IssueSourceError(
                f"{target}: the header has no {key!r} line; a frozen report that cannot "
                "say how it was resolved and gated is not one the reader may use"
            )

    _, after_gate = _section(body, GATE_MARKER)
    gate_section, after_raw = _section(after_gate, RAW_MARKER)
    raw_section, after_content = _section(after_raw, CONTENT_MARKER)
    content_section, after_raw_hints = _section(after_content, RAW_HINTS_MARKER)
    raw_hints_section, after_hints = _section(after_raw_hints, HINTS_MARKER)
    raw = _canonical(raw_section)
    content = _canonical(content_section)
    raw_hints = _canonical(raw_hints_section)
    hints = _canonical(after_hints)

    if sha256_text(content) != header["content-sha256"]:
        raise IssueSourceError(
            f"{target}: content-sha256 in the header is {header['content-sha256']}, but the "
            f"content hashes to {sha256_text(content)} — the file was edited after it was "
            "frozen"
        )
    if sha256_text(raw) != header["raw-sha256"]:
        raise IssueSourceError(
            f"{target}: raw-sha256 in the header is {header['raw-sha256']}, but the raw "
            f"report hashes to {sha256_text(raw)} — the record of what the gate removed was "
            "edited after it was frozen"
        )
    if sha256_text(hints) != header["hints-sha256"]:
        raise IssueSourceError(
            f"{target}: hints-sha256 in the header is {header['hints-sha256']}, but the "
            f"hints hash to {sha256_text(hints)} — the comments were edited after the file "
            "was frozen"
        )
    if sha256_text(raw_hints) != header["raw-hints-sha256"]:
        raise IssueSourceError(
            f"{target}: raw-hints-sha256 in the header is {header['raw-hints-sha256']}, but "
            f"the raw comments hash to {sha256_text(raw_hints)} — the record of what the gate "
            "removed from the comments was edited after the file was frozen"
        )
    try:
        gate_payload = json.loads(_canonical(gate_section) or "{}")
    except ValueError as exc:
        raise IssueSourceError(f"{target}: the {GATE_MARKER} section is not JSON: {exc}") from exc
    hints_payload = gate_payload.get("hints")
    if hints_payload is None and header["comments-included"] != "0":
        raise IssueSourceError(
            f"{target}: the header says {header['comments-included']} comment(s) were "
            "included, but the gate record carries no hints block — the file does not say "
            "what was gated out of them"
        )
    gate = issue_mod.LeakResult(
        text=content,
        removals=[
            issue_mod.LeakRemoval.from_dict(item, where=f"{target} gate removals[]")
            for item in gate_payload.get("removals") or []
        ],
        ungated_prose=[str(word) for word in gate_payload.get("ungated_prose") or []],
        ok=bool(gate_payload.get("ok")),
        reason=gate_payload.get("reason"),
    )
    hints_leak = issue_mod.LeakResult(
        text=hints,
        removals=[
            issue_mod.LeakRemoval.from_dict(item, where=f"{target} gate hints removals[]")
            for item in (hints_payload or {}).get("removals") or []
        ],
        ungated_prose=[str(word) for word in (hints_payload or {}).get("ungated_prose") or []],
        # Not a verdict — see ``issue_mod.gate_hints``: the hints are never sent, so nobody
        # asked whether they are worth sending. Reconstructed rather than stored so the file
        # cannot appear to record an answer to a question that was never put.
        ok=True,
        reason=None,
    )

    source, via_pr = _parse_resolved(header["resolved"])
    title, body_text = _parse_content(content)
    comments = _parse_hints(hints)
    return FrozenIssue(
        path=target,
        source=source,
        url="" if header["issue"] == "—" else header["issue"],
        issue_number=_number_in_url(header["issue"]),
        via_pr=via_pr,
        fetched_at=header["fetched"],
        fix_commit_date=header["fix-commit-date"],
        content_sha256=header["content-sha256"],
        title=title,
        body=body_text,
        comments=comments,
        content=content,
        raw=raw,
        hints=hints,
        raw_hints=raw_hints,
        hints_sha256=header["hints-sha256"],
        hints_leak=hints_leak,
        gate=gate,
        gate_files=tuple(str(name) for name in gate_payload.get("files") or []),
        gate_tree_sha256=header["gate-tree-sha256"],
        gate_answer_sha256=header["gate-answer-sha256"],
        reason=header.get("note", ""),
        refrozen_at=header.get("refrozen", ""),
    )


def _parse_resolved(value: str) -> tuple[str, Optional[int]]:
    if value.startswith("via PR #"):
        return SOURCE_VIA_PR, int(value[len("via PR #") :])
    if value == "direct":
        return SOURCE_ISSUE, None
    if value == "none":
        return SOURCE_NONE, None
    raise IssueSourceError(f"the frozen report says resolved: {value!r}, which is not one of direct / via PR #N / none")


def _number_in_url(url: str) -> Optional[int]:
    match = re.search(r"/issues/(\d+)\b", url or "")
    return int(match.group(1)) if match else None


def _parse_content(content: str) -> tuple[str, str]:
    """The inverse of :meth:`IssueReport.content`: the title and the body, and nothing else.

    No comment handling at all, deliberately. A report's body is free text a reporter typed,
    and a line in it that happens to read ``## Comments`` is that reporter's line — the old
    reader would have taken everything after it as the comment thread, which is a way for the
    body's own bytes to be silently reclassified by their content. Now the split is by
    *section* (the markers), not by a heading a body may or may not contain.
    """
    title = ""
    body_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        body_lines.append(line)
    return title, _report_text("\n".join(body_lines))


def _parse_hints(hints: str) -> list[IssueComment]:
    """The inverse of :meth:`IssueReport.hints`: the comments, read back in order.

    The leading ``## Comments`` heading is dropped; each ``### author — created_at`` line
    opens a comment and everything up to the next one is its body. An empty hints section is
    a report with no comments, which is a fact about the report and not a truncated file —
    the marker line itself is what says the section is there (:func:`_section`).
    """
    comments: list[IssueComment] = []
    current: Optional[IssueComment] = None
    body_buf: list[str] = []
    for line in hints.splitlines():
        if line.startswith("### ") and " — " in line:
            if current is not None:
                comments.append(_with_body(current, body_buf))
            who, _, when = line[4:].partition(" — ")
            current = IssueComment(author=who.strip(), created_at=when.strip(), body="")
            body_buf = []
            continue
        if current is not None:
            body_buf.append(line)
    if current is not None:
        comments.append(_with_body(current, body_buf))
    return comments


def _with_body(comment: IssueComment, lines: Sequence[str]) -> IssueComment:
    """The comment's body, read back by the same rule that wrote it (:func:`_report_text`).

    Not ``.strip()``: a comment is as much the reporter's text as the body is, and an indented
    code block inside one has to come back with its indentation.
    """
    return IssueComment(
        author=comment.author,
        created_at=comment.created_at,
        body=_report_text("\n".join(lines)),
    )


def ensure_frozen(
    client: GitHubClient,
    *,
    issues_dir: Path,
    project: str,
    bug_id: int,
    github_url: str,
    message: str,
    fix_commit_date: str,
    gate: Callable[[str, str], GateRecord],
    refreeze: bool = False,
) -> FrozenIssue:
    """Read the frozen report, resolving and gating it once if it is not there yet.

    This is §2.4 item 1's "prepare step, cached": the *resolution* happens here, the network
    is touched at most once per bug ever (the client's cache answers every later run), and
    what it produced is written down and committed, so the text a run sees is a file in the
    repository rather than a live API response.

    ``gate`` is the freeze step's other half and is supplied by the caller because its inputs
    are the caller's: the buggy commit's source and the fix diff are on the caller's disk,
    and neither the API client nor this module walks a tree (§2.4). It is called once, on the
    report's two texts — the content and the comments (§2.4.4) — and what it returns is stored
    beside the raw text; after that no stage gates anything again, which is what "identical by
    construction" means here.

    ``refreeze`` is the *only* way an existing artifact is ever replaced, and it can only be
    set by someone naming the bug on the command line (§2.4.3). A file that exists is
    otherwise read, and read *once*: a refusal comes back to the caller as an error, never as
    a reason to write a new text, because an artifact whose hashes disagree with its own
    header is exactly the case where a silent rewrite would replace something nobody can
    verify with something nobody has reviewed. When it is set and the file did exist, the new
    artifact records a ``refrozen`` date, so a reader of the diff can tell a deliberate
    rewrite from drift rather than wondering why ``content-sha256`` moved.
    """
    target = frozen_path(issues_dir, project, bug_id)
    if target.is_file() and not refreeze:
        return load_frozen(target)
    report = resolve(
        client,
        project=project,
        bug_id=bug_id,
        github_url=github_url,
        message=message,
        fix_commit_date=fix_commit_date,
    )
    report.gate = gate(report.content(), report.hints())
    if target.is_file():
        report.refrozen_at = _utc_now()
    write_frozen(target, report)
    return load_frozen(target)


def load_or_none(issues_dir: Path, project: str, bug_id: int) -> Optional[FrozenIssue]:
    target = frozen_path(issues_dir, project, bug_id)
    return load_frozen(target) if target.is_file() else None


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "FROZEN_HEADER_KEYS",
    "GATE_MARKER",
    "GITHUB_API",
    "HINTS_MARKER",
    "ISSUES_DIR",
    "GateRecord",
    "IssueComment",
    "IssueReport",
    "IssueSourceError",
    "GitHubClient",
    "HttpResponse",
    "RAW_MARKER",
    "RAW_HINTS_MARKER",
    "RefCandidate",
    "SOURCE_ISSUE",
    "SOURCE_NONE",
    "SOURCE_VIA_PR",
    "TOKEN_ENV",
    "closing_numbers",
    "default_client",
    "ensure_frozen",
    "frozen_path",
    "load_frozen",
    "load_or_none",
    "owner_repo",
    "parse_frozen",
    "referenced_numbers",
    "render_frozen",
    "resolve",
    "write_frozen",
]
