#!/usr/bin/env python3
"""Author ``plans/bugsinpy_dev_split.json`` from the pinned clone (design §14.7).

Run once, by hand, before any gate has been run and before any Vanta result has
been looked at. The bug metadata (``python_version``, ``buggy_commit``,
``fixed_commit``, ``test_file``) is read from BugsInPy's own ``bug.info`` rather
than retyped, because a mistyped 40-hex commit checks out the wrong tree and
grades something else. Everything the split declares as a judgement —
``test_cmd``, ``native_test_paths``, ``role``, ``notes`` — is in the table below.

The order is the round-robin §14.7 records: one pass over the projects in the
clone's own order, each contributing its next py3.8 bug in ascending id, repeated
until there are ten candidates. ``PySnooper/3`` is appended last as an extra
``pilot`` labelled "seen during development".

    python3 scripts/bugsinpy/dev_split_author.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from bugsinpy import bugsinpy_meta as meta  # noqa: E402

SPLIT_PATH = REPO_ROOT / "plans" / "bugsinpy_dev_split.json"

# The evaluation split's pin, which the dev split must share: same BugsInPy
# commit, same interpreter layer, same pytest pins (§3.2).
BUGSINPY_COMMIT = "11c5f1eea954a42132cfd06bf257766a7963e0fd"
BUGSINPY_ROOT = "runs/bugsinpy/bugsinpy"
CREATED_UTC = "2026-10-21T04:57:36Z"

PYTEST_VERSION = "8.0.2"
JSON_REPORT_VERSION = "1.5.0"

ROUND_ROBIN_RULE = (
    "Round-robin across projects in clone order (fastapi, luigi, matplotlib, "
    "pandas, sanic, scrapy, spacy): take the first bug in each project that "
    "passes both gates, until 5. If a project has no bug that passes, skip it "
    "and continue. One pass over the projects gives their next py3.8 bug in "
    "ascending id and the pass repeats until 10 candidates are listed; the "
    "first 5 are pilots, the rest the substitution pool. The two gates are the "
    "existing ones -- the §1.6 context gate and the §8.2 dev-fix self-check -- "
    "neither is loosened and no cap is raised. Recorded before any gate was run "
    "and before any Vanta result was looked at."
)

# (project, bug_id, role, test path, test root, note)
#
# The test root is the project's own test directory, the same convention the
# evaluation split uses (``tests`` for PySnooper, ``tqdm/tests`` for tqdm): the
# suite the oracle derives FAIL_TO_PASS/PASS_TO_PASS from. For luigi/2 the
# file's own directory is ``test/contrib`` but the project's root is ``test``,
# so the root is ``test`` for both luigi entries.
POOL = [
    (
        "fastapi", 1, "pilot", "tests/test_jsonable_encoder.py", "tests",
        "Round-robin pass 1, fastapi. First py3.8 bug in the project.",
    ),
    (
        "luigi", 1, "pilot", "test/server_test.py", "test",
        "Round-robin pass 1, luigi. First py3.8 bug in the project.",
    ),
    (
        "matplotlib", 1, "pilot", "lib/matplotlib/tests/test_bbox_tight.py",
        "lib/matplotlib/tests",
        "Round-robin pass 1, matplotlib. First py3.8 bug in the project.",
    ),
    (
        "pandas", 1, "pilot", "pandas/tests/dtypes/test_dtypes.py", "pandas/tests",
        "Round-robin pass 1, pandas. First py3.8 bug in the project.",
    ),
    (
        "sanic", 1, "pilot", "tests/test_blueprints.py", "tests",
        "Round-robin pass 1, sanic. First py3.8 bug in the project. Pilot #5, "
        "the last of the one-per-project pass.",
    ),
    (
        "scrapy", 1, "reserve", "tests/test_spidermiddleware_offsite.py", "tests",
        "Round-robin pass 1, scrapy. Reserve #1: the substitution pool starts "
        "with another project's first bug, so promoting it keeps the dev set at "
        "five distinct projects.",
    ),
    (
        "spacy", 4, "reserve", "spacy/tests/regression/test_issue4665.py",
        "spacy/tests",
        "Round-robin pass 1, spacy. Reserve #2. spacy has exactly one py3.8 bug "
        "in the corpus, so this project cannot supply a second candidate.",
    ),
    (
        "fastapi", 2, "reserve", "tests/test_ws_router.py", "tests",
        "Round-robin pass 2, fastapi. Reserve #3, the first second-bug depth: "
        "used if fastapi/1 fails a gate.",
    ),
    (
        "luigi", 2, "reserve", "test/contrib/beam_dataflow_test.py", "test",
        "Round-robin pass 2, luigi. Reserve #4. The test root is the project's "
        "``test``, not the file's own ``test/contrib``.",
    ),
    (
        "matplotlib", 2, "reserve", "lib/matplotlib/tests/test_axes.py",
        "lib/matplotlib/tests",
        "Round-robin pass 2, matplotlib. Reserve #5, the tenth candidate.",
    ),
]

PYSNOOPER_NOTE = (
    "Seen during development. Items 1 and 2 of §14 were designed from this bug's "
    "round-4 false reject, so it is development data, not an evaluation bug: it "
    "is run here to check that a no-op still rejects via rule (b) and that the "
    "developer fix no longer false-rejects, and it is never reported as an "
    "evaluation result. The evaluation split records the same exclusion in its "
    "``reviewed_changes``."
)


def main() -> int:
    root = REPO_ROOT / BUGSINPY_ROOT
    bugs: list[dict] = []

    for project, bug_id, role, test_file, test_root, note in POOL:
        info = meta.load_bug_info(root, project, bug_id)
        if info.test_file != test_file:
            raise SystemExit(
                f"{project}/{bug_id}: the clone declares test_file "
                f"{info.test_file!r}, this table says {test_file!r}"
            )
        bugs.append(
            {
                "project": project,
                "bug_id": bug_id,
                "python_version": info.python_version,
                "buggy_commit": info.buggy_commit,
                "fixed_commit": info.fixed_commit,
                "test_file": info.test_file,
                "pythonpath": info.pythonpath,
                "needs_build": info.pythonpath is not None,
                "native_test_paths": [test_root],
                "test_cmd": [
                    "python",
                    "-m",
                    "pytest",
                    test_root,
                    "-p",
                    "no:cacheprovider",
                    "--json-report",
                    "--json-report-file={report}",
                ],
                "pytest_version": PYTEST_VERSION,
                "json_report_version": JSON_REPORT_VERSION,
                "checkpoint": f"vanta-bi-{project}-{bug_id}",
                "wave": 1,
                "role": role,
                "notes": note,
            }
        )

    # PySnooper/3, mirrored from the evaluation split so the two entries cannot
    # drift, then labelled.
    evaluation = json.loads(
        (REPO_ROOT / "plans" / "bugsinpy_split.json").read_text(encoding="utf-8")
    )
    pysnooper = next(
        entry
        for entry in evaluation["bugs"]
        if entry["project"] == "PySnooper" and entry["bug_id"] == 3
    )
    extra = {
        key: pysnooper[key]
        for key in (
            "project",
            "bug_id",
            "python_version",
            "buggy_commit",
            "fixed_commit",
            "test_file",
            "pythonpath",
            "needs_build",
            "native_test_paths",
            "test_cmd",
            "pytest_version",
            "json_report_version",
            "checkpoint",
        )
    }
    extra["wave"] = 1
    extra["role"] = "pilot"
    extra["notes"] = PYSNOOPER_NOTE
    bugs.append(extra)

    data = {
        "schema_version": 1,
        "created_utc": CREATED_UTC,
        "bugsinpy_commit": BUGSINPY_COMMIT,
        "bugsinpy_root": BUGSINPY_ROOT,
        "configs": ["baseline", "vanta"],
        "seeds": [0],
        "review_mode": True,
        "sampling": {
            "policy": "pinned_per_call_plus_server_top_p",
            "note": (
                "temperature/max_tokens come from the pipeline's per-call pins "
                "(design §4.3); top_p is the one value read from the server's "
                "/props. The harness adds no run-level value."
            ),
        },
        "reviewed_changes": [
            {
                "change": (
                    "This is the development split of §14.7: five py3.8 bugs, one "
                    "per project, from projects that appear in no evaluation wave, "
                    "plus PySnooper/3 labelled 'seen during development'. It is a "
                    "separate file so the evaluation split's bugs, waves and "
                    "sampling are not rewritten by the choice of development data."
                ),
                "reason": (
                    "Items 1-3 of §14 are developed on bugs no evaluation number "
                    "will be computed from, so no number is produced by a pipeline "
                    "tuned on the bug it measures. Flat directory order was "
                    "rejected: it makes the first ten candidates ten fastapi bugs, "
                    "so every item would be developed against one codebase's "
                    "idioms. Round-robin gives one bug per project."
                ),
                "decided_by": "user review, 2026-10-21",
                "rule": ROUND_ROBIN_RULE,
            },
            {
                "change": (
                    "Excluded from the candidate pool: ansible, keras and tornado. "
                    "No bug of theirs in the pinned clone declares "
                    "``python_version`` 3.8 (0 of 18, 0 of 45, 0 of 16), so the "
                    "§1.6 context gate can never pass for them and §14.7's "
                    "round-robin has nothing to take from these projects."
                ),
                "reason": (
                    "Recorded so the skip reads as a decision rather than an "
                    "omission: a reader comparing the clone's projects with the "
                    "seven in the round-robin would otherwise have to re-derive "
                    "it from the corpus. They are not evaluation-wave projects "
                    "either, and the same fact is noted in the plan doc (§14.7)."
                ),
                "decided_by": "user review, 2026-10-21",
            }
        ],
        "bugs": bugs,
    }

    SPLIT_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {SPLIT_PATH} ({len(bugs)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
