"""Item 10 — replay a finished repair row's stored attempts onto one shared test set.

[VANTA-WHY] A repair row's live verdicts answer "did this patch pass", not "which
test changed its mind between these two attempts". The EIR/ECR rates need the
second question: for one fixed set of tests S, run every stored attempt's patch and
read the matrix ``X[test][k]``. That matrix never existed during the run — the live
run compared only what the Governor's shared set held at the time, attempt over
attempt — so it is rebuilt here, offline and deterministically, from the run's own
artifacts.

Three rules shape everything below.

* **Replay changes nothing that already exists.** Every write lands under
  ``<batch>/replay/``; the manifest, the transitions file, ``oracle/`` and
  ``vanta_sessions/`` are opened read-only. A row's live verdict, label and score
  are inputs, never outputs.
* **A cell is reused, not re-run, when the run already observed it.** The native
  suite already ran on each attempt's exact patch and the baseline ran in
  ``prepare``, so those cells are read back from the stored results — but only when
  the column's identity (a content hash of its patch) matches the id the live rows
  carry. A cell is never reused on a guess.
* **Everything that decides a number is a pure function** (:func:`cell_value`,
  :func:`transitions`, :func:`rates`, :func:`sample_cells`, :func:`flaky`,
  :func:`patch_id`). The Shuru layer below produces reports and decides nothing, so
  the acceptance test can exercise the counting rules with no VM at all.

The index convention (§4): column 0 is the *unmodified* baseline tree; column k
(1..K) is the tree the live graded run used for attempt k. So K stored attempts
produce K+1 columns, and the 0->1 transition is baseline-to-first-patch and nothing
else.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import arms as arms_mod
from . import compact as compact_mod
from . import manifest as manifest_mod
from . import oracle as oracle_mod
from . import pool as pool_mod
from . import runner as runner_mod
from . import shuru as shuru_mod
from . import split as split_mod
from . import suite as suite_mod
from . import trees as tree_mod


class ReplayError(RuntimeError):
    """The replay could not be run at all (as opposed to "a cell was censored")."""


SCHEMA_VERSION = 1
REPLAY_DIRNAME = "replay"

INDEX_CONVENTION = (
    "attempt 0 is the unmodified baseline tree; attempts 1..K are the Creator patches"
)

# The M3.5 §6.6 evidence_source vocabulary (§8, fold-in 8). These are the tokens a
# reader (and item 12's metrics) cuts by; the Rust enum they come from is
# Native/ExistingRepositoryTest, BenchmarkOracle and AttackerGenerated.
SOURCE_REPO_BASELINE = "repo_baseline"
SOURCE_BENCHMARK_ORACLE = "benchmark_oracle"
SOURCE_VANTA_GENERATED = "vanta_generated"

VIEW_GENERATED = "generated"
VIEW_ORACLE = "oracle"
DEFAULT_VIEWS = (VIEW_GENERATED, VIEW_ORACLE)

CELL_PASS = "pass"
CELL_FAIL = "fail"
CELL_ERROR = "error"
CELL_NOT_RUN = "not-run"

ORIGIN_BASELINE = "baseline"
ORIGIN_GRADED = "graded"
ORIGIN_COMPACTED = "compacted"
ORIGIN_MISSING = "missing"

DEFAULT_SAMPLE_RATE = 0.1
DEFAULT_SEED = 20261030

GENERATED_TEST_STEM = "vanta_replay"
"""The filename a generated test is written under inside a replay column.

One file per test, named ``vanta_replay_<NNN>.py`` by the test's position in the
sorted admitted set, so the mapping is a pure function of the row and a report's
nodeid can be read back to the ``generated:<test_name>`` matrix key.
"""


# ---------------------------------------------------------------------------
# pure functions — everything that decides a number (the acceptance test's target)
# ---------------------------------------------------------------------------
def patch_id(patch_text: str) -> str:
    """``"patch-<fnv1a64>"`` — the driver's ``stable_patch_fingerprint``, in Python.

    [VANTA-WHY] Byte-for-byte the algorithm at ``driver.rs:5456`` (FNV-1a over the
    patch's UTF-8 bytes, 16 lowercase hex digits, prefixed ``patch-``). This is the
    reuse guard's identity: a live transition row is only trusted for a column when
    its ``to_patch_attempt_id``/``from_patch_attempt_id`` equals this value, so the
    two implementations *must* agree or no cell would ever be reused. The Rust unit
    tests (``driver.rs:6589``) pin the Rust side; ``test_bugsinpy.py`` pins this one
    against a fixed vector.
    """
    hash_value = 0xCBF29CE484222325
    for byte in patch_text.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"patch-{hash_value:016x}"


def cell_value(
    report_outcome: Optional[str], collected: Iterable[str], infra: Optional[str]
) -> str:
    """One report entry -> one matrix cell (§6's censoring table).

    ``infra`` is the reason the *run* measured nothing (no report, an unreadable
    report, a timeout, a crash) — never ``None`` when the run's own reader refused
    it — and it censors the whole column, so it wins over anything the report says.

    ``passed`` is a pass; ``failed`` and ``error`` are both a fail. The plan's
    *"an import failure caused by the patch counts as fail, not censored"* is why
    ``error`` is not a censor: a collection or import error of a file in the
    column's tree is the patch's own doing. ``collected`` names the nodeids the run
    reached, so a caller can tell a test the report is silent about from one that
    was never collected at all; the cell is censored either way.
    """
    if infra:
        return CELL_ERROR
    if report_outcome == "passed":
        return CELL_PASS
    if report_outcome in suite_mod.FAILURE_OUTCOMES:
        return CELL_FAIL
    return CELL_NOT_RUN


_LIVE_RESULT_VALUES = {"Passed": CELL_PASS, "Failed": CELL_FAIL, "Error": CELL_FAIL}
"""A stored ``TestResult`` (``driver.rs``/``core``) -> a cell value.

``Error`` reads as ``fail`` for the same reason a report ``error`` does (§6): a
test that ran and did not hold its assertions is evidence about the candidate, not
an infrastructure failure. The only infrastructure failures are the run's own.
"""


def live_value(result: object) -> str:
    """Stored ``TestResult`` -> cell value, or :data:`CELL_NOT_RUN` when unknown."""
    return _LIVE_RESULT_VALUES.get(str(result), CELL_NOT_RUN)


def _is_observed(value: str) -> bool:
    return value in (CELL_PASS, CELL_FAIL)


def transitions(
    cells: Sequence[Mapping[str, Any]], *, excluded: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """The raw ``n_pp/n_pf/n_fp/n_ff/n_censored`` counts per consecutive column pair.

    [VANTA-WHY] A transition ``k -> k+1`` counts only when *both* cells are
    observed (pass or fail); anything else — a censored cell, a test absent from one
    of the two columns — moves ``n_censored`` and no rate. That is the plan's
    *"a transition is observed only if both cells are pass or fail"*, and it is why
    the counts are stored as five raw integers rather than as three ratios: a reader
    can always reconstruct the ratios, and can also see how much of the row was
    thrown away.

    ``excluded`` names test ids the flake sample flagged; they are dropped at this
    stage rather than edited out of ``"cells"``, so the raw record keeps every
    observation while the rates lose the flaky test (§9).
    """
    excluded_ids = {str(test_id) for test_id in excluded}
    by_test: dict[str, dict[int, Mapping[str, Any]]] = {}
    for cell in cells:
        by_test.setdefault(str(cell["test_id"]), {})[int(cell["k"])] = cell
    columns = sorted({int(cell["k"]) for cell in cells})

    out: list[dict[str, Any]] = []
    for older, newer in zip(columns, columns[1:]):
        counts = {"n_pp": 0, "n_pf": 0, "n_fp": 0, "n_ff": 0, "n_censored": 0}
        by_source: dict[str, dict[str, int]] = {}
        for test_id in sorted(by_test):
            if test_id in excluded_ids:
                continue
            source = str(by_test[test_id][older].get("evidence_source") or "")
            slot = by_source.setdefault(
                source, {"n_pp": 0, "n_pf": 0, "n_fp": 0, "n_ff": 0, "n_censored": 0}
            )
            older_cell = by_test[test_id].get(older)
            newer_cell = by_test[test_id].get(newer)
            if older_cell is None or newer_cell is None:
                counts["n_censored"] += 1
                slot["n_censored"] += 1
                continue
            a = str(older_cell.get("value"))
            b = str(newer_cell.get("value"))
            if not _is_observed(a) or not _is_observed(b):
                counts["n_censored"] += 1
                slot["n_censored"] += 1
                continue
            bucket = "n_pp" if a == CELL_PASS and b == CELL_PASS else (
                "n_pf" if a == CELL_PASS else ("n_fp" if b == CELL_PASS else "n_ff")
            )
            counts[bucket] += 1
            slot[bucket] += 1
        entry: dict[str, Any] = {"k": f"{older}->{newer}"}
        entry.update(counts)
        entry["by_source"] = by_source
        out.append(entry)
    return out


def rates(counts: Mapping[str, Any]) -> dict[str, Any]:
    """``{eir, ecr, acc, denominators}`` from one transition's raw counts.

    EIR is ``pass -> fail`` out of everything that started at pass; ECR is
    ``fail -> pass`` out of everything that started at fail; ``acc`` is the share of
    observed transitions whose newer cell passed. A zero denominator is ``None``,
    never ``0`` and never a ``ZeroDivisionError`` — "no transition started at pass"
    is not "the error-introduction rate is zero", and the difference matters in a
    paper. Both denominators ride along so a reader never has to re-derive them.
    """
    n_pp = int(counts.get("n_pp") or 0)
    n_pf = int(counts.get("n_pf") or 0)
    n_fp = int(counts.get("n_fp") or 0)
    n_ff = int(counts.get("n_ff") or 0)
    eir_den = n_pp + n_pf
    ecr_den = n_fp + n_ff
    observed = eir_den + ecr_den
    return {
        "eir": (n_pf / eir_den) if eir_den else None,
        "ecr": (n_fp / ecr_den) if ecr_den else None,
        "acc": ((n_pp + n_fp) / observed) if observed else None,
        "denominators": {
            "eir": eir_den,
            "ecr": ecr_den,
            "acc": observed,
            "observed": observed,
            "censored": int(counts.get("n_censored") or 0),
        },
    }


def sample_cells(
    cells: Sequence[Mapping[str, Any]], seed: int, rate: float
) -> list[dict[str, Any]]:
    """A seeded sample of cells to re-execute, drawn over the *sorted* cell list.

    [VANTA-WHY] Sorting first is what makes the draw a pure function of the cell
    set and the seed, so ``--seed`` reproduces the sample exactly in a fresh
    process — the plan's *"replaying twice gives the same matrix"*. The sample is
    drawn over **every** cell, reused ones included (Nathan, Q3): a reused cell is
    the only place a *live* observation can be caught being flaky.

    ``rate=0`` draws none. A positive rate on a small matrix draws at least one —
    10% of a five-column row must not round down to "no flake check at all", which
    is the whole instrument.
    """
    ordered = sorted(
        (dict(cell) for cell in cells), key=lambda c: (str(c["test_id"]), int(c["k"]))
    )
    if rate <= 0 or not ordered:
        return []
    count = min(len(ordered), max(1, round(rate * len(ordered))))
    return random.Random(seed).sample(ordered, count)


def flaky(
    cells: Sequence[Mapping[str, Any]], reexecutions: Mapping[tuple[str, int], str]
) -> set[str]:
    """Test ids whose re-executed cell disagrees with the observation on disk.

    ``reexecutions`` maps ``(test_id, k)`` to the value a second, independent
    execution produced. A cell not in the map was not sampled (or its column could
    not be resolved) and cannot be judged.
    """
    out: set[str] = set()
    for cell in cells:
        key = (str(cell["test_id"]), int(cell["k"]))
        if key not in reexecutions:
            continue
        if reexecutions[key] != str(cell.get("value")):
            out.add(key[0])
    return out


GROUNDING_UNGROUNDED_SCRAPPED = "ungrounded_scrapped"
GROUNDING_LITERAL = "literal_grounded"
GROUNDING_SYMPTOM = "symptom_grounded"


def assertion_grounding_type(assertions_grounded: Optional[bool], scrapped: bool) -> Optional[str]:
    """The M3.5 three-word grounding vocabulary, derived from item 1's tri-state.

    [VANTA-WHY] This is a *derivation*, flagged in the design: item 1 stores
    ``assertions_grounded`` as a tri-state, not these three words (the three words
    are the M3.5 vocabulary, where an explicit field arrives). The mapping is
    written here so a reader can disagree with it: a probe whose assertions the audit
    could not ground *and* which was scrapped is ``ungrounded_scrapped``; an
    ungrounded probe that survived is ``literal_grounded`` (item 1's loosening kept
    it because the issue itself names what it asserts); a grounded probe is
    ``symptom_grounded``; no audit ran -> ``None``.
    """
    if assertions_grounded is None:
        return None
    if assertions_grounded:
        return GROUNDING_SYMPTOM
    return GROUNDING_UNGROUNDED_SCRAPPED if scrapped else GROUNDING_LITERAL


def baseline_failure_signature(evidence: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse BATE's baseline account into ``{exception_type, repo_frame, ...}``.

    ``bate::baseline_evidence`` renders ``verdict: …`` then ``exception: <line>``
    then a bounded tail of pytest's output. This reads what is there and never
    guesses: an evidence text with no traceback frame yields ``None`` for
    ``repo_frame``, and a message that cannot be isolated yields ``None`` for
    ``message_sha256``. ``None`` overall means the artifact was absent or carried
    nothing parseable — never a fabricated signature.
    """
    if not evidence:
        return None
    exception_type: Optional[str] = None
    repo_frame: Optional[str] = None
    message: Optional[str] = None
    for raw in str(evidence).splitlines():
        line = raw.strip()
        if line.startswith("exception:"):
            detail = line[len("exception:") :].strip()
            if ":" in detail:
                head = detail.split(":", 1)[0].strip()
                if head and " " not in head:
                    exception_type = head
        elif line.startswith('File "'):
            # `File "<path>", line N, in <fn>` — the first frame whose file is not
            # a stdlib/site-packages path is the repo's own.
            try:
                path = line.split('"', 2)[1]
            except IndexError:
                continue
            if repo_frame is None and "site-packages" not in path and "/lib/python" not in path:
                repo_frame = path
        elif exception_type and detail_looks_like_message(line):
            message = line
    signature = {
        "exception_type": exception_type,
        "repo_frame": repo_frame,
        "assertion_file": None,
        "message_sha256": (
            hashlib.sha256(message.encode("utf-8")).hexdigest() if message else None
        ),
    }
    if not any(value for value in signature.values()):
        return None
    return signature


def detail_looks_like_message(line: str) -> bool:
    """A pytest ``E   …`` assertion line — the exception's own message."""
    return line.startswith("E ") or line.startswith("E\t")


# ---------------------------------------------------------------------------
# per-attempt trees (graded, not patched — the tree the live run graded)
# ---------------------------------------------------------------------------
def graded_patch(
    *, debug_dir: Path, base_tree: Path, attempt: int
) -> tuple[Optional[str], Optional[str], str]:
    """``(patch text, reason, origin)`` for one attempt's **graded** tree.

    [VANTA-WHY] ``pool.attempt_patch`` is the same three-case shape but hard-coded
    to ``attempt_<n>_patched_tree``. The replay must diff the **graded** tree —
    the one the live run put the generated tests in, with patch-added test files
    dropped and the existing tests restored — because that is the tree the live
    native cells were observed on, so the replay's code and the live code are the
    same code. Rather than add a second knob to ``pool.py``, this calls the pool's
    own ``_read_compacted`` / ``_covered_paths`` / ``drop_base_only_deletions`` and
    ``compact.tree_diff`` verbatim, so there is one implementation of each rule and
    ``pool.py`` is untouched.

    The third case — neither the tree nor the pinned diff is on disk — is censored,
    never guessed from ``final.patch``: ``final.patch`` is the *last* attempt's
    patch and attributing it to another attempt would be a fabricated observation.
    """
    tree = debug_dir / f"attempt_{attempt}_graded_tree"
    if tree.is_dir():
        try:
            text = compact_mod.tree_diff(base_tree, tree)
        except compact_mod.CompactError as exc:
            return None, f"the graded snapshot tree could not be diffed: {exc}", ""
        return (
            pool_mod.drop_base_only_deletions(text, pool_mod._covered_paths(debug_dir, tree.name)),
            None,
            ORIGIN_GRADED,
        )
    diff_path = debug_dir / f"{tree.name}.diff"
    hashes_path = debug_dir / f"{tree.name}.hashes.json"
    if diff_path.is_file() and hashes_path.is_file():
        text, reason = pool_mod._read_compacted(diff_path, hashes_path)
        if text is None:
            return None, reason, ""
        # The manifest outlives the tree it describes (compaction removes only the
        # tree), so the covered set is read the same way in both cases.
        return (
            pool_mod.drop_base_only_deletions(text, pool_mod._covered_paths(debug_dir, tree.name)),
            None,
            ORIGIN_COMPACTED,
        )
    if diff_path.is_file() or hashes_path.is_file():
        return None, "the graded .diff and .hashes.json are not both present", ""
    return None, "no graded snapshot tree and no compacted .diff/.hashes.json", ""


# ---------------------------------------------------------------------------
# reading a row's own artifacts (all opened read-only)
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _attempt_patch_text(debug_dir: Path, attempt: int) -> Optional[str]:
    """The Creator's patch for one attempt, if this run recorded it (item 9).

    [VANTA-WHY] Read for the *reuse guard* only: its hash is the column's identity,
    and the hash of the raw text is exactly what the driver stamped
    (``stable_patch_fingerprint`` over the patch body). Whitespace is significant —
    no normalisation here, or the id would stop matching the live rows.
    """
    path = debug_dir / f"attempt_{attempt}_creator_patch.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _attempt_tests(debug_dir: Path, attempt: int) -> Optional[list[dict[str, Any]]]:
    """The test set one attempt ran (item 9), or ``None`` when it was not stored.

    ``attempt_<n>_tests.json`` is a ``Vec<GeneratedTest>`` — the fixed shared set
    plus the round's new tests — so this is the row's admitted universe and the
    source of fold-in 2's ``round_admitted``.
    """
    payload = _read_json(debug_dir / f"attempt_{attempt}_tests.json")
    if not isinstance(payload, list):
        return None
    return [entry for entry in payload if isinstance(entry, dict)]


def _attempt_test_names(debug_dir: Path, attempt: int) -> Optional[list[str]]:
    entries = _attempt_tests(debug_dir, attempt)
    if entries is None:
        return None
    return [
        entry["test_name"]
        for entry in entries
        if isinstance(entry.get("test_name"), str)
    ]


def _attempt_generated_tests(debug_dir: Path, attempt: int) -> Optional[list[dict[str, Any]]]:
    """Item 10's driver artifact: every Attacker test of one round (fold-ins 3/4).

    ``None`` for a row whose batch predates the artifact — the caller then records
    ``driver_artifacts.generated_tests: "absent"`` and ``bate_status: null`` with a
    named reason, never a guessed status.
    """
    payload = _read_json(debug_dir / f"attempt_{attempt}_generated_tests.json")
    if not isinstance(payload, list):
        return None
    return [entry for entry in payload if isinstance(entry, dict)]


def _shared_tests(debug_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(debug_dir / "shared_tests.json")
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _transitions(sessions_dir: Path, session_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(sessions_dir / f"{session_id}_transitions.jsonl")


def _intent_checks(sessions_dir: Path, session_id: str) -> dict[str, dict[str, Any]]:
    """``{test_name: {assertions_grounded, scrapped}}`` from the session log.

    The log's ``intent.check[<name>]`` rows carry the facts as one ``key=value``
    line (``log_intent_checks``); this reads the two keys item 10's §8 fold-in 5
    needs and nothing else.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(sessions_dir / f"{session_id}_session.jsonl"):
        phase = str(row.get("phase") or "")
        if not phase.startswith("intent.check[") or not phase.endswith("]"):
            continue
        name = phase[len("intent.check[") : -1]
        summary = str(row.get("summary") or "")
        fields: dict[str, str] = {}
        for token in summary.split(" "):
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        out[name] = {
            "assertions_grounded": _tri_state(fields.get("assertions_grounded")),
            "scrapped": fields.get("scrapped") == "true",
        }
    return out


def _tri_state(text: Optional[str]) -> Optional[bool]:
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def is_repair_row(record: Mapping[str, Any]) -> bool:
    """A row item 10 may replay: a finished repair with a session and a patch."""
    if str(record.get("mode") or "") != arms_mod.REPAIR:
        return False
    if not record.get("session_id"):
        return False
    return bool((record.get("artifacts") or {}).get("final_patch"))


def row_stem(key: str) -> str:
    """A filesystem-safe stem for one row's files, derived from its manifest key."""
    out = [ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(key)]
    stem = "".join(out).strip("_")
    return stem[:120] or "row"


def row_seed(key: str, seed: int) -> int:
    """``(row key, seed)`` -> the per-row seed the sample is drawn with.

    Keeps :func:`sample_cells` a pure function of its own arguments while still
    making the draw differ between rows, which is what "a function of (row key, seed,
    sample_rate)" asks for.
    """
    digest = hashlib.sha256(f"{key}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------
@dataclass
class Column:
    """One column of the matrix — the tree one attempt graded."""

    k: int
    origin: str
    patch_id: Optional[str] = None
    patch_text: Optional[str] = None
    censor_reason: Optional[str] = None
    ran: bool = False

    @property
    def usable(self) -> bool:
        """Is the column's tree materialisable (whatever its origin)?"""
        return self.patch_text is not None or self.k == 0


@dataclass
class RowPlan:
    """Everything about one row that the counting stage needs, before any boot."""

    key: str
    bug: str
    session_id: str
    checkpoint: str
    prep: Any
    debug_dir: Path
    attempts: int
    columns: list[Column] = field(default_factory=list)
    native_ids: list[str] = field(default_factory=list)
    generated: list[dict[str, Any]] = field(default_factory=list)
    cells: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    sampled: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    generated_tests_present: bool = False
    generated_tests_reason: str = ""


def _column_patch_id(debug_dir: Path, attempt: int) -> Optional[str]:
    text = _attempt_patch_text(debug_dir, attempt)
    return patch_id(text) if text is not None else None


def build_columns(
    *, debug_dir: Path, base_tree: Path, attempts: int
) -> list[Column]:
    """K+1 columns: 0 is the baseline, 1..K are the graded trees (fold-in 1)."""
    columns = [Column(k=0, origin=ORIGIN_BASELINE)]
    for attempt in range(1, attempts + 1):
        text, reason, origin = graded_patch(
            debug_dir=debug_dir, base_tree=base_tree, attempt=attempt
        )
        pid = _column_patch_id(debug_dir, attempt)
        if text is None:
            columns.append(
                Column(k=attempt, origin=ORIGIN_MISSING, patch_id=pid, censor_reason=reason)
            )
        else:
            columns.append(
                Column(k=attempt, origin=origin, patch_id=pid, patch_text=text)
            )
    return columns


def reuse_cells(
    rows: Sequence[Mapping[str, Any]], column_ids: Mapping[int, Optional[str]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Cells the live run already observed, keyed ``(matrix key, k)`` (fold-in 2/8).

    [VANTA-WHY] The guard is the whole point of this function. A live transition row
    is trusted for a column only when that column's patch id equals the id the row
    stamped on that end — a content hash of the patch body, which is what makes "the
    same candidate" mean the same thing across runs. The matrix key follows §5: a
    native nodeid becomes ``native:<nodeid>``, a generated test ``generated:<name>``,
    never the bare nodeid (the live nodeid and the replay nodeid are different files).

    [VANTA-WHY] The ``+ 1`` is the index convention, not an off-by-one bug. A
    transition row's ``to_attempt``/``from_attempt`` are the **Governor's** attempt
    numbers, which start at 0 (``Governor::new`` sets ``current_attempt: 0``; the
    driver's loop stamps that number), while a replay column k is the **tree** the
    live run graded for that attempt, and those files are named ``attempt_<n>`` with
    n = governor attempt + 1 (``driver.rs`` writes ``attempt_{}_creator_patch.txt``
    and ``attempt_{}_graded_tree`` with ``attempt + 1``). Column ``k`` therefore
    holds ``attempt_<k>``, i.e. governor attempt ``k - 1``. Mapping governor attempt
    ``a`` to column ``a + 1`` makes the ids line up; without it every ``to_attempt =
    0`` row would look up column 0 (the baseline, whose id is ``None``) and be
    silently dropped, and no cell would ever be reused.
    """
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("censored"):
            continue
        source = str(row.get("evidence_source") or "")
        if source == "Native":
            evidence_source = SOURCE_REPO_BASELINE
            matrix_key = f"native:{row.get('test_id')}"
            test_name = row.get("test_name")
        elif source == "AttackerGenerated":
            evidence_source = SOURCE_VANTA_GENERATED
            test_name = row.get("test_name")
            matrix_key = f"generated:{test_name}"
        else:
            continue
        for end, attempt_key, id_key, result_key in (
            ("to", "to_attempt", "to_patch_attempt_id", "to_result"),
            ("from", "from_attempt", "from_patch_attempt_id", "from_result"),
        ):
            attempt = row.get(attempt_key)
            if attempt is None:
                continue
            attempt = int(attempt) + 1
            if column_ids.get(attempt) is None or row.get(id_key) != column_ids[attempt]:
                continue
            key = (matrix_key, attempt)
            if key in out:
                continue
            out[key] = {
                "test_id": matrix_key,
                "k": attempt,
                "value": live_value(row.get(result_key)),
                "observed_online": True,
                "raw": str(row.get(result_key)),
                "source": "live",
                "evidence_source": evidence_source,
                "test_name": test_name,
                "end": end,
            }
    return out


def baseline_cells(
    native_ids: Sequence[str], failure_baseline: Iterable[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Column 0's native cells from ``prepare``'s baseline (§7).

    [VANTA-WHY] The baseline was measured by ``prepare.strip_and_baseline`` on
    ``vanta_tree``, and ``vanta_clean`` is a copy of it, so a nodeid failing there
    fails on column 0 and every other is a pass. That is the design's *"column 0 by
    outcome-in-baseline-failures"*, and it is a reuse, so ``observed_online`` is
    true. It can only speak about nodeids the live run's shared set named — a
    nodeid nobody ever compared is simply not in column 0's universe rather than
    assumed to pass.
    """
    failures = {str(nodeid) for nodeid in failure_baseline}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for nodeid in native_ids:
        value = CELL_FAIL if nodeid in failures else CELL_PASS
        out[(f"native:{nodeid}", 0)] = {
            "test_id": f"native:{nodeid}",
            "k": 0,
            "value": value,
            "observed_online": True,
            "raw": "baseline failure" if value == CELL_FAIL else "baseline pass",
            "source": "live",
            "evidence_source": SOURCE_REPO_BASELINE,
            "test_name": nodeid,
        }
    return out


# ---------------------------------------------------------------------------
# the run layer — Shuru produces reports; everything above turns them into numbers
# ---------------------------------------------------------------------------
APPLY_TIMEOUT_SECS = 300
DEFAULT_TIMEOUT_SECS = 1800

DEFAULT_SPLIT = runner_mod.REPO_ROOT / "plans" / "bugsinpy_split.json"

GENERATED_CONFTEST_SHIM = (
    "import sys, os\n"
    "_here = os.path.dirname(os.path.abspath(__file__))\n"
    "for _p in (os.path.join(_here, 'src'), _here):\n"
    "    if _p not in sys.path:\n"
    "        sys.path.insert(0, _p)\n"
)
"""``harness::write_conftest``'s ``sys.path`` shim, byte-for-byte (``harness.rs:754``).

[VANTA-WHY] The live harness writes this into a *born-bare* sandbox root that holds
nothing but the source concat, which is why a ``src/`` layout needs the shim to be
importable at all. A replay column is a full copy of ``vanta_clean``, so it usually
already has the repo's own root ``conftest.py`` — and the two would collide on the
same path. The shim is therefore written **only when the column tree has no root
``conftest.py``** (a flagged, low-risk decision): overwriting a repo's conftest
would break its own pytest configuration, and a repo that ships one already knows
how to import itself.
"""


def _safe_label(text: object) -> str:
    out = [ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text)]
    return "".join(out).strip("_")[:80] or "row"


def evidence_source_for(test_id: str) -> str:
    """The M3.5 §6.6 token a test id belongs to (fold-in 8)."""
    if test_id.startswith("native:"):
        return SOURCE_REPO_BASELINE
    if test_id.startswith("oracle:"):
        return SOURCE_BENCHMARK_ORACLE
    return SOURCE_VANTA_GENERATED


def cells_for_view(
    cells: Sequence[Mapping[str, Any]], view: str
) -> list[dict[str, Any]]:
    """The cells one view owns — the oracle view, or everything else (§5/§6).

    [VANTA-WHY] Without this split the two views would be counted together: the
    oracle nodeids are disjoint from S, so their transitions would land in the S
    view's ``by_source`` and the pooled counts would silently mix the two.
    """
    if view == VIEW_ORACLE:
        keep = lambda tid: str(tid).startswith("oracle:")
    else:
        keep = lambda tid: not str(tid).startswith("oracle:")
    return [dict(cell) for cell in cells if keep(cell.get("test_id"))]


def parse_views(raw: str) -> list[str]:
    """``--views`` -> the views to run, in the canonical order and deduplicated.

    ``generated`` is S (the native suite plus every admitted generated test);
    ``oracle`` is the hidden nodeids, kept in their own view so their transitions
    never mix into S's counts (§5). An unknown name is refused rather than ignored:
    a typo that silently ran one view would read as a complete replay.
    """
    parts = {part for part in str(raw).replace(",", " ").split() if part}
    if not parts:
        return list(DEFAULT_VIEWS)
    unknown = sorted(parts - set(DEFAULT_VIEWS))
    if unknown:
        raise ReplayError(
            f"--views names unknown view(s) {unknown}; the views are {list(DEFAULT_VIEWS)}"
        )
    return [view for view in DEFAULT_VIEWS if view in parts]


def generated_files_map(names: Sequence[str]) -> dict[str, str]:
    """``{test_name: vanta_replay_<NNN>.py}`` by the sorted admitted position (§5).

    One file per test, so a report's file part reads straight back to a
    ``generated:<test_name>`` matrix key — a pure function of the row's admitted set.
    """
    ordered = sorted({str(name) for name in names if name})
    return {
        name: f"{GENERATED_TEST_STEM}_{index:03d}.py"
        for index, name in enumerate(ordered, start=1)
    }


def _nodeid_matrix_key(
    nodeid: str, by_file: Mapping[str, str], hidden: frozenset[str]
) -> str:
    """A report nodeid -> its matrix key (§5). Never the bare nodeid."""
    if nodeid in hidden:
        return f"oracle:{nodeid}"
    name = by_file.get(nodeid.split("::", 1)[0])
    if name is not None:
        return name
    return f"native:{nodeid}"


@dataclass
class ColumnRun:
    """What one column's boot produced, before any number is derived from it."""

    outcomes: dict[str, str] = field(default_factory=dict)
    """matrix key -> the first run's raw pytest outcome, only for nodeids it reached."""
    unstable: set[str] = field(default_factory=set)
    """matrix keys whose two runs in the one boot disagreed (the free flake rule)."""
    infra: Optional[str] = None
    """Non-``None`` when the run measured nothing — the whole column is censored."""
    ran: bool = False
    boots: int = 0
    secs: float = 0.0


def run_column(
    *,
    row_root: Path,
    k: int,
    base_tree: Path,
    patch_text: Optional[str],
    drop_test_hunks: bool,
    command: Sequence[str],
    generated: Mapping[str, str],
    generated_source: Mapping[str, str],
    hidden: frozenset[str],
    report_stem: str,
    checkpoint: str,
    prep: Any,
    disk_mb: int,
    timeout_secs: int,
) -> ColumnRun:
    """One column's tree, one boot, two runs (§10). Decides nothing.

    ``row_root`` is the *effective* batch dir — ``<batch>/replay/<row-stem>/`` — so
    ``run_suite_twice``'s reports, ``git_apply_patch``'s staged patch and every tree
    the VM reads all land inside ``<batch>/replay/`` and the manifest, the
    transitions file and ``vanta_sessions/`` are never written (§3).

    ``generated`` maps a ``generated:<name>`` matrix key to the file name it is
    written under *in this column*; ``command`` is the base pytest command (with its
    ``{report}`` placeholder); the generated part and item U's ``-o addopts=`` are
    appended only when this column actually needs generated files, so a column that
    runs only repo tests runs byte-for-byte the command the live native run used.

    The tree is deleted before returning: columns are transient by design (§3), and
    a pandas column is a few hundred megabytes.
    """
    run = ColumnRun()
    col_dir = row_root / f"col_{k}"
    tree_mod.copy_tree(base_tree, col_dir)

    if patch_text is not None:
        patch_path = row_root / f"col_{k}.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        started = time.monotonic()
        applied, detail = suite_mod.git_apply_patch(
            checkpoint=checkpoint,
            batch_dir=row_root,
            tree_rel=col_dir.name,
            patch_path=patch_path,
            disk_mb=disk_mb,
            drop_test_hunks=drop_test_hunks,
            memory_mb=prep.vm_memory_mb,
            timeout_secs=APPLY_TIMEOUT_SECS,
        )
        run.boots += 1
        run.secs += time.monotonic() - started
        if not applied:
            run.infra = f"the column's patch did not apply: {detail}"
            shutil.rmtree(col_dir, ignore_errors=True)
            return run

    for matrix_key, file_name in generated.items():
        (col_dir / file_name).write_text(generated_source[matrix_key], encoding="utf-8")
    if generated and not (col_dir / "conftest.py").exists():
        (col_dir / "conftest.py").write_text(GENERATED_CONFTEST_SHIM, encoding="utf-8")

    cmd = list(command)
    if generated:
        cmd = cmd + ["-o", "addopts="] + sorted(set(generated.values()))

    started = time.monotonic()
    pair = suite_mod.run_suite_twice(
        checkpoint=checkpoint,
        batch_dir=row_root,
        tree_rel=col_dir.name,
        test_cmd=cmd,
        report_rel_stem=report_stem,
        disk_mb=disk_mb,
        timeout_secs=timeout_secs,
        extra_env=prep.sandbox_env or None,
        memory_mb=prep.vm_memory_mb,
        precompile=prep.precompile,
    )
    run.boots += 1
    run.secs += time.monotonic() - started
    run.ran = True

    run.infra = suite_mod.run_state(pair.first) or suite_mod.run_state(pair.second)
    if run.infra:
        shutil.rmtree(col_dir, ignore_errors=True)
        return run

    by_file = {file_name: matrix_key for matrix_key, file_name in generated.items()}
    for nodeid, outcome in pair.first.outcomes.items():
        run.outcomes[_nodeid_matrix_key(nodeid, by_file, hidden)] = outcome
    for nodeid in pair.flaky_nodeids | pair.missing_from_one_run:
        run.unstable.add(_nodeid_matrix_key(nodeid, by_file, hidden))
    shutil.rmtree(col_dir, ignore_errors=True)
    return run


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------
def _column_test_cmd(view: str, prep: Any) -> list[str]:
    """The base pytest command a view runs, before generated files are appended."""
    if view == VIEW_ORACLE:
        return oracle_mod.hidden_command(
            prep.test_cmd, prep.native_test_paths, prep.hidden_nodeids
        )
    return [str(part) for part in prep.test_cmd]


def _base_tree(view: str, prep: Any) -> Path:
    return prep.root / ("oracle_tree" if view == VIEW_ORACLE else "vanta_clean")


def replay_row(
    *,
    record: Mapping[str, Any],
    prep: Any,
    checkpoint: str,
    batch_dir: Path,
    sessions_dir: Path,
    views: Sequence[str],
    seed: int,
    sample_rate: float,
    disk_mb: int,
    timeout_secs: int,
    progress: bool = True,
) -> dict[str, Any]:
    """Build one row's replay record. Writes only under ``<batch>/replay/``."""
    key = str(record.get("key") or "")
    session_id = str(record.get("session_id") or "")
    debug_dir = sessions_dir / session_id / "debug"
    row_root = batch_dir / REPLAY_DIRNAME / row_stem(key)
    row_root.mkdir(parents=True, exist_ok=True)

    attempts = pool_mod._attempt_total(record, debug_dir)
    columns = build_columns(
        debug_dir=debug_dir, base_tree=prep.root / "vanta_clean", attempts=attempts
    )
    column_ids = {column.k: column.patch_id for column in columns}

    transition_rows = _transitions(sessions_dir, session_id)
    native_ids = sorted(
        {
            str(row.get("test_id"))
            for row in transition_rows
            if str(row.get("evidence_source")) == "Native"
            and row.get("test_id")
            and not row.get("censored")
        }
    )

    reused = reuse_cells(transition_rows, column_ids)
    reused.update(baseline_cells(native_ids, prep.native_failure_baseline))

    shared_tests = _shared_tests(debug_dir)
    gen_names = sorted(
        {str(entry.get("test_name")) for entry in shared_tests if entry.get("test_name")}
    )
    gen_files = generated_files_map(gen_names)
    gen_source = {
        str(entry.get("test_name")): str(entry.get("test_source") or "")
        for entry in shared_tests
        if entry.get("test_name")
    }

    s_tests = [f"native:{nodeid}" for nodeid in native_ids] + [
        f"generated:{name}" for name in gen_names
    ]
    oracle_tests = [f"oracle:{nodeid}" for nodeid in prep.hidden_nodeids]
    test_ids = {"generated": s_tests, "oracle": oracle_tests}

    # §9: the sample is drawn over every S cell — reused ones included — and the
    # oracle view is not sampled. Sorting inside `sample_cells` makes the draw a
    # pure function of (cell set, per-row seed).
    s_universe = [
        {"test_id": test_id, "k": column.k}
        for test_id in sorted(set(s_tests))
        for column in columns
    ]
    sampled = sample_cells(s_universe, row_seed(key, seed), sample_rate)
    sampled_keys = {(str(cell["test_id"]), int(cell["k"])) for cell in sampled}

    cells: list[dict[str, Any]] = [dict(cell) for cell in reused.values()]
    reexecutions: dict[tuple[str, int], str] = {}
    unstable_ids: set[str] = set()
    unresolvable: list[str] = []
    ran_columns: set[int] = set()
    boots = 0
    secs = 0.0
    executed = 0
    sampled_resolved = 0

    for view in views:
        view_tests = test_ids[view]
        if not view_tests:
            continue
        hidden = frozenset(prep.hidden_nodeids) if view == VIEW_ORACLE else frozenset()
        command = _column_test_cmd(view, prep)
        base = _base_tree(view, prep)
        for column in columns:
            need = [tid for tid in view_tests if (tid, column.k) not in reused]
            sampled_reused = [
                tid
                for tid in view_tests
                if (tid, column.k) in sampled_keys and (tid, column.k) in reused
            ]
            union = sorted(set(need) | set(sampled_reused))
            if not union:
                continue
            if not column.usable:
                # The column's tree is gone: nothing to boot. Non-reused cells are
                # simply absent (transitions() counts them censored); a sampled
                # live cell has no second execution to compare against.
                unresolvable.extend(sampled_reused)
                continue
            generated = {
                tid: gen_files[tid[len("generated:") :]]
                for tid in union
                if tid.startswith("generated:")
            }
            run = run_column(
                row_root=row_root,
                k=column.k,
                base_tree=base,
                patch_text=column.patch_text,
                drop_test_hunks=view == VIEW_ORACLE,
                command=command,
                generated=generated,
                generated_source={
                    tid: gen_source[tid[len("generated:") :]] for tid in generated
                },
                hidden=hidden,
                report_stem=f"reports/{_safe_label(view)}-{column.k}",
                checkpoint=checkpoint,
                prep=prep,
                disk_mb=disk_mb,
                timeout_secs=timeout_secs,
            )
            boots += run.boots
            secs += run.secs
            ran_columns.add(column.k)
            if run.infra:
                unresolvable.extend(sampled_reused)
                continue
            unstable_ids |= run.unstable
            for tid in need:
                outcome = run.outcomes.get(tid)
                if outcome is None:
                    continue
                cells.append(
                    {
                        "test_id": tid,
                        "k": column.k,
                        "value": cell_value(outcome, [], None),
                        "observed_online": False,
                        "raw": outcome,
                        "source": "replay",
                        "evidence_source": evidence_source_for(tid),
                        "test_name": tid.split(":", 1)[1],
                    }
                )
                executed += 1
            for tid in sampled_reused:
                outcome = run.outcomes.get(tid)
                if outcome is None:
                    unresolvable.append(tid)
                    continue
                reexecutions[(tid, column.k)] = cell_value(outcome, [], None)
                sampled_resolved += 1

    # Two instruments answer the same question — is a cell stable? — and both feed
    # one `flaky` list: the free within-boot half (`unstable_ids`, where each
    # column's two runs in one boot disagreed on a nodeid) and the seeded
    # re-execution half (`flaky`, where a sampled cell's fresh run disagreed with the
    # observation on disk). The exclusion happens at the counts stage, not by
    # editing `cells`, so the raw record keeps every observation (§9).
    flaky_ids = set(flaky(cells, reexecutions)) | unstable_ids

    views_out: dict[str, Any] = {}
    for view in views:
        view_cells = cells_for_view(cells, view)
        entries = transitions(view_cells, excluded=flaky_ids)
        for entry in entries:
            entry["rates"] = rates(entry)
        views_out[view] = {"transitions": entries}

    intent_checks = _intent_checks(sessions_dir, session_id)
    gen_meta: dict[str, dict[str, Any]] = {}
    generated_tests_present = False
    for attempt in range(1, attempts + 1):
        entries = _attempt_generated_tests(debug_dir, attempt)
        if entries is None:
            continue
        generated_tests_present = True
        for entry in entries:
            name = str(entry.get("test_name") or "")
            if name and name not in gen_meta:
                gen_meta[name] = entry
    round_admitted: dict[str, int] = {}
    for attempt in range(1, attempts + 1):
        names = _attempt_test_names(debug_dir, attempt)
        if names is None:
            continue
        for name in names:
            round_admitted.setdefault(name, attempt)

    tests_out: list[dict[str, Any]] = []
    for test_id in sorted(set(s_tests) | set(oracle_tests)):
        if test_id.startswith("native:"):
            tests_out.append(
                {
                    "test_id": test_id,
                    "evidence_source": SOURCE_REPO_BASELINE,
                    "round_admitted": None,
                    "test_name": test_id[len("native:") :],
                    "assertion_grounding_type": None,
                }
            )
            continue
        if test_id.startswith("oracle:"):
            tests_out.append(
                {
                    "test_id": test_id,
                    "evidence_source": SOURCE_BENCHMARK_ORACLE,
                    "round_admitted": None,
                    "test_name": test_id[len("oracle:") :],
                    "assertion_grounding_type": None,
                }
            )
            continue
        name = test_id[len("generated:") :]
        meta = gen_meta.get(name)
        check = intent_checks.get(name) or {}
        tests_out.append(
            {
                "test_id": test_id,
                "evidence_source": SOURCE_VANTA_GENERATED,
                "round_admitted": round_admitted.get(name),
                "test_name": name,
                "replay_file": gen_files[name],
                "declared_intent": (meta or {}).get("declared_intent"),
                "bate_outcome": {
                    "status": (meta or {}).get("bate_status"),
                    "reason": (
                        (meta or {}).get("bate_reason")
                        if meta is not None
                        else "the driver artifact attempt_<n>_generated_tests.json is "
                        "absent for every attempt (a row written before item 10)"
                    ),
                },
                "baseline_failure_signature": baseline_failure_signature(
                    (meta or {}).get("baseline_evidence")
                ),
                "assertion_grounding_type": assertion_grounding_type(
                    check.get("assertions_grounded"), bool(check.get("scrapped"))
                ),
            }
        )

    reused_count = sum(1 for cell in cells if cell.get("observed_online"))
    if progress:
        print(
            f"  {prep.key}: {len(columns)} columns × {len(views)} view(s); "
            f"{reused_count} reused, {executed} executed, {boots} boot(s), "
            f"{secs:.1f}s",
            flush=True,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "replay",
        "key": key,
        "session_id": session_id,
        "bug": str(record.get("bug") or prep.key),
        "checkpoint": checkpoint,
        "index_convention": INDEX_CONVENTION,
        "columns": [
            {
                "k": column.k,
                "origin": column.origin,
                "patch_id": column.patch_id,
                "ran": column.k in ran_columns,
                "censor_reason": column.censor_reason,
            }
            for column in columns
        ],
        "tests": tests_out,
        "cells": sorted(cells, key=lambda cell: (str(cell["test_id"]), int(cell["k"]))),
        "views": views_out,
        "flaky": sorted(flaky_ids),
        "unresolvable": sorted(set(unresolvable)),
        "driver_artifacts": {
            "generated_tests": "present" if generated_tests_present else "absent"
        },
        "replay": {
            "boots": boots,
            "sandbox_secs": round(secs, 3),
            "seed": seed,
            "sample_rate": sample_rate,
            "sampled_cells": len(sampled),
            "sampled_resolved": sampled_resolved,
            "executed_cells": executed,
            "reused_cells": reused_count,
        },
    }
