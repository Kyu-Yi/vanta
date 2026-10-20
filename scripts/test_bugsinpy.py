#!/usr/bin/env python3
"""Regression tests for the BugsInPy harness (``scripts/bugsinpy/``).

Run with:  python3 -m pytest scripts/test_bugsinpy.py -q

[WHY] Every test here pins a rule that would otherwise fail *silently and in the
wrong direction*. The harness decides what Vanta is allowed to see, what the
oracle is, and what counts as a leak — and a harness bug does not look like a bug,
it looks like a result. Three examples, each of which would produce a number
rather than an error:

* a hidden test left behind in ``vanta_tree/`` (the Creator reads the answer),
* a token stripped from the issue text that the buggy tree actually contains
  (the bug becomes unsolvable, and the low resolve rate looks like a model
  failure),
* a context precheck measuring a different walk from ``repo::collect_source_files``
  (bugs are rejected, or worse accepted, for the wrong reason).

Nothing in this file executes the project under test. The suite runner's Shuru
wrappers are covered structurally (``build_command``, ``drop_test_file_hunks``),
never by booting a VM.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bugsinpy import bake as bake_mod  # noqa: E402
from bugsinpy import bugsinpy_meta as meta_mod  # noqa: E402
from bugsinpy import compact as compact_mod  # noqa: E402
from bugsinpy import context as ctx  # noqa: E402
from bugsinpy import issue as issue_mod  # noqa: E402
from bugsinpy import manifest as manifest_mod  # noqa: E402
from bugsinpy import oracle as oracle_mod  # noqa: E402
from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import records as records_mod  # noqa: E402
from bugsinpy import runner as runner_mod  # noqa: E402
from bugsinpy import shuru as shuru_mod  # noqa: E402
from bugsinpy import split as split_mod  # noqa: E402
from bugsinpy import suite as suite_mod  # noqa: E402
from bugsinpy import trees as tree_mod  # noqa: E402
from bugsinpy.split import SplitError, load_split  # noqa: E402
import bugsinpy_run as run_mod  # noqa: E402

SPLIT_PATH = REPO_ROOT / "plans" / "bugsinpy_split.json"
BUGSINPY_CLONE = REPO_ROOT / "runs" / "bugsinpy" / "bugsinpy"

needs_clone = pytest.mark.skipif(
    not BUGSINPY_CLONE.is_dir(),
    reason="the BugsInPy clone is not present (runs/ is not committed)",
)


# ============================================================================
# fixture helpers
# ============================================================================
def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def blob(path: Path, size: int) -> Path:
    """A file of exactly ``size`` bytes (content irrelevant to the walk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def bug_info(**over) -> meta_mod.BugInfo:
    base = dict(
        project="Demo",
        bug_id=1,
        python_version="3.8.1",
        buggy_commit="b" * 40,
        fixed_commit="f" * 40,
        test_file="tests/test_demo.py",
        pythonpath=None,
        requirements_txt="",
        run_test_sh="",
        bug_patch_txt="",
        bug_dir=Path("/nonexistent"),
    )
    base.update(over)
    return meta_mod.BugInfo(**base)


def bug_entry(**over) -> dict:
    entry = dict(
        project="Demo",
        bug_id=1,
        python_version="3.8.1",
        buggy_commit="b" * 40,
        fixed_commit="f" * 40,
        test_file="tests/test_demo.py",
        pythonpath=None,
        needs_build=False,
        native_test_paths=["tests"],
        test_cmd=["python", "-m", "pytest", "tests", "--json-report-file={report}"],
        pytest_version="8.0.2",
        json_report_version="1.5.0",
        checkpoint="vanta-bi-demo-1",
        wave=1,
        role="pilot",
        notes="",
    )
    entry.update(over)
    return entry


def split_data(**over) -> dict:
    data = dict(
        schema_version=1,
        created_utc="2026-01-01T00:00:00Z",
        bugsinpy_commit="a" * 40,
        bugsinpy_root="runs/bugsinpy/bugsinpy",
        configs=["baseline", "vanta"],
        seeds=[0],
        review_mode=True,
        sampling={"policy": split_mod.SAMPLING_POLICY},
        bugs=[bug_entry()],
    )
    data.update(over)
    return data


def prep(**over) -> prepare_mod.BugPrep:
    base = dict(bug=bug_info(), root=Path("/nonexistent"), role="pilot", wave=1, test_cmd=["x"])
    base.update(over)
    return prepare_mod.BugPrep(**base)


def _bug_deps(
    entries, installs_by_checkpoint: dict | None = None
) -> list[bake_mod.BugDeps]:
    """The ``BugDeps`` :func:`layer_recipes` needs, built without PyPI or a repo.

    Resolving real dependencies is :func:`resolve_bug_deps`'s job and it is tested
    on its own; these tests are about the *shape* of the layers, so they hand the
    recipes the pins directly.
    """
    installs = installs_by_checkpoint or {}
    return [
        bake_mod.BugDeps(
            entry=dict(entry),
            checkpoint=bake_mod.bug_checkpoint_name(entry),
            installs=tuple(installs.get(bake_mod.bug_checkpoint_name(entry), ())),
            provenance={},
        )
        for entry in entries
    ]


# ============================================================================
# split.py — the committed protocol
# ============================================================================
class TestSplit:
    def test_real_split_loads_and_validates(self):
        split = load_split(SPLIT_PATH)
        assert split.seeds == [0], "the pilot runs a single seed (§8.3)"
        assert split.configs == ["baseline", "vanta"]
        assert split.review_mode is True
        assert split.sampling["policy"] == split_mod.SAMPLING_POLICY

    def test_real_split_pilot_set_is_frozen(self):
        """The split is the protocol, so a silent edit has to fail a test.

        Pinned deliberately: changing the pilot set is a decision, not a detail,
        and it must show up as a failing test rather than as a different number in
        a write-up.
        """
        split = load_split(SPLIT_PATH)
        pilot_keys = {f"{e['project']}/{e['bug_id']}" for e in split_mod.pilot_bugs(split)}
        reserve_keys = {
            f"{e['project']}/{e['bug_id']}"
            for e in split.bugs
            if e.get("role") == "reserve"
        }
        assert pilot_keys == {
            "PySnooper/2",
            "PySnooper/3",
            "black/22",
            "thefuck/1",
            "thefuck/2",
            "thefuck/3",
            "httpie/1",
            "httpie/2",
            "httpie/3",
            "cookiecutter/1",
        }
        assert reserve_keys == {
            "black/1",
            "black/2",
            "tqdm/1",
            "youtube-dl/1",
        }
        assert not (pilot_keys & reserve_keys)

    def test_real_split_reviewed_changes_carry_the_rule_and_its_measurements(self):
        """§8's substitution is the one place the split departs from §8's own plan.

        Pinned so the amendment cannot be edited away or have its evidence thinned:
        the rule that chose the substitute, the exclusion it substitutes for, and a
        measurement for every candidate the rule walked are all in the file a
        reviewer reads.
        """
        split = load_split(SPLIT_PATH)
        changes = split.data["reviewed_changes"]
        assert len(changes) == 2
        excluded, promoted = changes
        assert "PySnooper/1" in excluded["change"]
        assert "Python-2-only" in excluded["reason"]

        assert "black/22" in promoted["change"]
        assert "\"in a wave\" means already scheduled to run as a pilot" in promoted["rule"]
        measurements = promoted["measurements"]
        assert [m["bug"] for m in measurements] == [f"black/{i}" for i in range(1, 23)]
        # The rule stops at the FIRST passer, so exactly the last measurement passes
        # and every one before it carries the reason the precheck fired.
        assert [m["bug"] for m in measurements if m["verdict"] == "pass"] == ["black/22"]
        assert all(m["reasons"] for m in measurements[:-1])

    def test_reviewed_change_validates(self):
        data = split_data(
            reviewed_changes=[
                {
                    "change": "black/22 replaces PySnooper/1",
                    "reason": "the first context-gate passer in the py3.8 pool",
                    "decided_by": "review",
                    "rule": "first candidate that passes the gate wins",
                    "measurements": [
                        {
                            "bug": "black/1",
                            "estimated_prompt_tokens": 68376,
                            "headroom_tokens": -3096,
                            "verdict": "reject",
                            "reasons": ["source_truncated_by_cap: ..."],
                        }
                    ],
                }
            ]
        )
        split_mod.validate(data)

    def test_reviewed_change_needs_a_change_and_a_reason(self):
        for field in ("change", "reason"):
            entry = {"change": "x", "reason": "y"}
            entry[field] = ""
            with pytest.raises(SplitError, match=field):
                split_mod.validate(split_data(reviewed_changes=[entry]))

    def test_reviewed_change_rejects_unknown_keys(self):
        with pytest.raises(SplitError, match="unknown keys"):
            split_mod.validate(
                split_data(reviewed_changes=[{"change": "x", "reason": "y", "why": "z"}])
            )

    def test_a_rejected_measurement_must_say_which_rule_fired(self):
        """A bare "black/1 rejects" is an assertion; the reasons are the evidence."""
        with pytest.raises(SplitError, match="must carry its reasons"):
            split_mod.validate(
                split_data(
                    reviewed_changes=[
                        {
                            "change": "substitute",
                            "reason": "no reserve fits",
                            "measurements": [
                                {
                                    "bug": "black/1",
                                    "estimated_prompt_tokens": 68376,
                                    "headroom_tokens": -3096,
                                    "verdict": "reject",
                                    "reasons": [],
                                }
                            ],
                        }
                    ]
                )
            )

    def test_a_measurement_verdict_is_pass_or_reject(self):
        with pytest.raises(SplitError, match="verdict"):
            split_mod.validate(
                split_data(
                    reviewed_changes=[
                        {
                            "change": "substitute",
                            "reason": "no reserve fits",
                            "measurements": [
                                {
                                    "bug": "black/1",
                                    "estimated_prompt_tokens": 1,
                                    "headroom_tokens": 1,
                                    "verdict": "maybe",
                                    "reasons": [],
                                }
                            ],
                        }
                    ]
                )
            )

    def test_waves_group_the_pilot(self):
        split = load_split(SPLIT_PATH)
        assert split.waves() == [1, 2, 3]
        assert len(split.bugs_in_wave(1)) == 3

    def test_unknown_top_level_key_is_an_error(self):
        with pytest.raises(SplitError, match="unknown top-level keys"):
            split_mod.validate(split_data(seed=0))  # typo for `seeds`

    def test_unknown_bug_key_is_an_error(self):
        data = split_data(bugs=[bug_entry(seeds=[0])])
        with pytest.raises(SplitError, match="unknown keys"):
            split_mod.validate(data)

    def test_missing_declared_key_is_an_error(self):
        entry = bug_entry()
        del entry["test_cmd"]
        with pytest.raises(SplitError, match="missing declared keys"):
            split_mod.validate(split_data(bugs=[entry]))

    def test_schema_version_must_match(self):
        with pytest.raises(SplitError, match="schema_version"):
            split_mod.validate(split_data(schema_version=2))

    def test_configs_must_be_implemented(self):
        """`vanta_planner` is addressable but unimplemented (§4.2)."""
        with pytest.raises(SplitError, match="unknown configs"):
            split_mod.validate(split_data(configs=["vanta_planner"]))

    def test_seeds_must_be_a_non_empty_list_of_ints(self):
        with pytest.raises(SplitError, match="seeds"):
            split_mod.validate(split_data(seeds=[]))
        with pytest.raises(SplitError, match="seeds"):
            split_mod.validate(split_data(seeds=["0"]))

    def test_sampling_policy_must_state_where_sampling_comes_from(self):
        """The policy names the real source (§4.3): per-call pins + server top_p.

        The retired ``server_defaults`` claimed the harness lets the server decide
        everything, which is not what the pipeline does — it pins temperature and
        max_tokens at each call site. A split that says otherwise is rejected.
        """
        with pytest.raises(SplitError, match="sampling.policy"):
            split_mod.validate(split_data(sampling={"policy": "server_defaults"}))
        with pytest.raises(SplitError, match="sampling.policy"):
            split_mod.validate(split_data(sampling={"policy": "fixed", "temperature": 0.2}))

    def test_test_cmd_must_carry_the_report_placeholder(self):
        data = split_data(bugs=[bug_entry(test_cmd=["python", "-m", "pytest", "tests"])])
        with pytest.raises(SplitError, match="placeholder"):
            split_mod.validate(data)

    def test_duplicate_bug_is_an_error(self):
        data = split_data(bugs=[bug_entry(), bug_entry()])
        with pytest.raises(SplitError, match="duplicate"):
            split_mod.validate(data)

    def test_issue_text_hash_must_match(self):
        data = split_data(bugs=[bug_entry(issue_text="hello", issue_text_sha256="0" * 64)])
        with pytest.raises(SplitError, match="issue_text_sha256"):
            split_mod.validate(data)

        good = split_data(
            bugs=[
                bug_entry(
                    issue_text="hello",
                    issue_text_sha256=split_mod.sha256_text("hello"),
                )
            ]
        )
        split_mod.validate(good)

    def test_set_derived_rejects_undeclared_fields(self):
        split = split_mod.Split(split_data())
        with pytest.raises(SplitError, match="not derived fields"):
            split.set_derived("Demo", 1, seeds=[0])
        split.set_derived("Demo", 1, prepare_mode=prepare_mod.FULL)
        assert split.bug("Demo", 1)["prepare_mode"] == "full"

    def test_set_derived_on_unknown_bug_is_an_error(self):
        split = split_mod.Split(split_data())
        with pytest.raises(SplitError, match="not in the split"):
            split.set_derived("Nope", 9, prepare_mode=prepare_mod.FULL)

    def test_save_round_trips(self, tmp_path):
        split = split_mod.Split(split_data())
        split.set_derived("Demo", 1, prepare_mode=prepare_mod.MEASURE_ONLY)
        target = tmp_path / "split.json"
        split.save(target)
        again = load_split(target)
        assert again.bug("Demo", 1)["prepare_mode"] == "context_only"

    def test_to_json_is_newline_terminated(self):
        assert split_mod.Split(split_data()).to_json().endswith("}\n")


# ============================================================================
# bugsinpy_meta.py — reading BugsInPy's own metadata
# ============================================================================
class TestMetadata:
    def test_space_before_equals_is_tolerated(self):
        """`fixed_commit_id ="..."` is real: it appears in the youtube-dl bugs.

        A parser that anchored on `=` or split on `"` would silently check out
        the wrong tree, and the wrong tree is a *different* oracle, not an error.
        """
        fields = meta_mod.parse_bug_info(
            'python_version="3.7.0"\nfixed_commit_id ="1cc47c66"\n'
        )
        assert fields["fixed_commit_id"] == "1cc47c66"
        assert fields["python_version"] == "3.7.0"

    def test_quotes_stripped_and_noise_skipped(self):
        fields = meta_mod.parse_bug_info(
            '# comment\n\npython_version = "3.6.9" \ntest_file="t.py"\nnoise\n'
        )
        assert fields == {"python_version": "3.6.9", "test_file": "t.py"}

    def test_run_test_command_skips_comments_and_blanks(self):
        bug = bug_info(run_test_sh="#!/bin/sh\n# a note\n\npython -m unittest -q a.b.C\n")
        assert bug.run_test_command() == "python -m unittest -q a.b.C"

    def test_run_test_command_absent(self):
        assert bug_info(run_test_sh="# only a comment\n").run_test_command() is None

    def test_missing_bug_directory(self, tmp_path):
        with pytest.raises(meta_mod.MetadataError, match="no such bug directory"):
            meta_mod.load_bug_info(tmp_path, "Demo", 1)

    def test_missing_required_key(self, tmp_path):
        write(tmp_path / "projects" / "Demo" / "bugs" / "1" / "bug.info", 'python_version="3.8.1"\n')
        with pytest.raises(meta_mod.MetadataError, match="buggy_commit_id"):
            meta_mod.load_bug_info(tmp_path, "Demo", 1)

    @needs_clone
    def test_youtube_dl_fixed_commit_is_a_clean_sha(self):
        bug = meta_mod.load_bug_info(BUGSINPY_CLONE, "youtube-dl", 1)
        assert re.fullmatch(r"[0-9a-f]{40}", bug.fixed_commit), bug.fixed_commit

    @needs_clone
    def test_split_agrees_with_bugsinpy_metadata(self):
        """The declared half of the split must still match ``bug.info``.

        This is the guard that makes `prepare`'s derivation meaningful: if a
        commit or a test file moved, the derived `FAIL_TO_PASS` would belong to a
        different bug than the split claims.
        """
        split = load_split(SPLIT_PATH)
        for entry in split.bugs:
            bug = meta_mod.load_bug_info(BUGSINPY_CLONE, entry["project"], entry["bug_id"])
            assert prepare_mod.check_declared(entry, bug) == []
            # No checkout here: this assertion is about the declared split
            # agreeing with bug.info, and that lives in the first two sources.
            # The third source (whole test directories) needs a fixed tree, which
            # only exists after prepare, so a path that cannot exist is passed.
            assert bug.test_file in tree_mod.test_paths_from_fix(
                bug, Path("/nonexistent-fixed-tree")
            )


# ============================================================================
# trees.py — checkouts and the SWE-bench split
# ============================================================================
class TestIsTestPath:
    """Must mirror ``harness::is_native_test_path`` in Rust exactly."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("conftest.py", True),
            ("tests/conftest.py", True),
            ("a/b/conftest.py", True),
            ("test_x.py", True),
            ("tests/test_x.py", True),
            ("pkg/x_test.py", True),
            ("src/tests/helper.py", True),
            ("src/test/helper.py", True),
            ("foo.py", False),
            ("src/foo.py", False),
            ("src/tests_helper.py", False),
            ("tests/data.json", False),
            ("docs/test_x.md", False),
            ("tests/__init__.py", True),
        ],
    )
    def test_matches_rust_semantics(self, path, expected):
        assert tree_mod.is_test_path(path) is expected


class TestParsePatch:
    PATCH = """diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 context
-old line
+new line
diff --git a/new.py b/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+brand new
diff --git a/gone.py b/gone.py
deleted file mode 100644
index 4444444..0000000
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-bye
"""

    def test_classifies_paths(self):
        parsed = tree_mod.parse_patch(self.PATCH)
        assert parsed.paths.changed == ["foo.py"]
        assert parsed.paths.added == ["new.py"]
        assert parsed.paths.deleted == ["gone.py"]
        assert parsed.paths.all == ["foo.py", "gone.py", "new.py"]

    def test_records_added_and_removed_lines(self):
        parsed = tree_mod.parse_patch(self.PATCH)
        assert parsed.added_lines == ["new line", "brand new"]
        assert parsed.removed_lines == ["old line", "bye"]

    def test_empty_patch(self):
        parsed = tree_mod.parse_patch("")
        assert parsed.paths.all == []
        assert parsed.added_lines == []


class TestTestPathsFromFix:
    def test_unions_patch_test_paths_with_bug_info(self, tmp_path):
        """Union, not intersection: either source naming a path includes it."""
        bug = bug_info(
            test_file="tests/test_untouched.py",
            bug_patch_txt=TestParsePatch.PATCH,
        )
        assert tree_mod.test_paths_from_fix(bug, tmp_path / "fixed") == [
            "tests/test_untouched.py"
        ]

    def test_patch_that_touches_a_test_file(self, tmp_path):
        patch = (
            "diff --git a/tests/test_a.py b/tests/test_a.py\n"
            "--- a/tests/test_a.py\n"
            "+++ b/tests/test_a.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        bug = bug_info(test_file="tests/test_b.py", bug_patch_txt=patch)
        fixed = tmp_path / "fixed"
        write(fixed / "tests" / "test_a.py", "def test_a():\n    pass\n")
        write(fixed / "tests" / "test_b.py", "def test_b():\n    pass\n")
        assert tree_mod.test_paths_from_fix(bug, fixed) == [
            "tests/test_a.py",
            "tests/test_b.py",
        ]

    def test_syncs_the_whole_test_directory_not_only_the_named_module(self, tmp_path):
        """A test module is not self-contained; the helper beside it must come too.

        PySnooper/2 is the case in point: its test file does
        ``from . import mini_toolbox`` and ``tests/mini_toolbox/`` exists only in
        the fixed commit, so a tree holding just the named module dies at
        collection with ImportError (which the harness reads as "0 tests
        collected"). Only the named module is named by the first two sources;
        the third source is what brings the helper.
        """
        bug = bug_info(test_file="tests/test_pysnooper.py", bug_patch_txt="")
        fixed = tmp_path / "fixed"
        write(fixed / "tests" / "__init__.py", "")
        write(fixed / "tests" / "test_pysnooper.py", "from . import mini_toolbox\n")
        write(fixed / "tests" / "mini_toolbox" / "__init__.py", "CONTEXTMANAGER = 1\n")

        paths = tree_mod.test_paths_from_fix(bug, fixed)

        assert "tests/test_pysnooper.py" in paths, "the named module is still there"
        assert "tests/__init__.py" in paths, "the package marker a relative import needs"
        assert "tests/mini_toolbox/__init__.py" in paths, "the helper the module imports"

    def test_no_third_source_when_the_test_directory_is_absent(self, tmp_path):
        """The third source is a union, so a missing directory contributes nothing.

        This is what keeps the split-agreement guard (which has no checkout)
        meaningful instead of crashing on a path that does not exist.
        """
        bug = bug_info(test_file="tests/test_untouched.py", bug_patch_txt="")
        assert tree_mod.test_paths_from_fix(bug, tmp_path / "no-such-fixed") == [
            "tests/test_untouched.py"
        ]


class TestNodeIds:
    @pytest.mark.parametrize(
        "nodeid,chain",
        [
            ("tests/test_a.py::test_one", ["test_one"]),
            ("tests/test_a.py::TestC::test_one", ["TestC", "test_one"]),
            ("tests/test_a.py::test_one[3]", ["test_one"]),
            ("tests/test_a.py::TestC::test_one[3-abc]", ["TestC", "test_one"]),
        ],
    )
    def test_chain_strips_parametrisation(self, nodeid, chain):
        assert tree_mod.nodeid_chain(nodeid) == chain
        assert tree_mod.nodeid_file(nodeid) == "tests/test_a.py"


class TestComposeVantaTree:
    def test_takes_fixed_tests_and_buggy_source(self, tmp_path):
        buggy = tmp_path / "buggy"
        fixed = tmp_path / "fixed"
        write(buggy / "demo.py", "VALUE = 'buggy'\n")
        write(buggy / "tests" / "test_demo.py", "def test_demo():\n    assert False\n")
        write(fixed / "demo.py", "VALUE = 'fixed'\n")
        write(fixed / "tests" / "test_demo.py", "def test_demo():\n    assert True\n")

        dest = tmp_path / "vanta"
        report = tree_mod.compose_vanta_tree(buggy, fixed, dest, ["tests/test_demo.py"])

        assert (dest / "demo.py").read_text() == "VALUE = 'buggy'\n"
        assert (dest / "tests" / "test_demo.py").read_text() == (
            "def test_demo():\n    assert True\n"
        )
        assert report.copied_from_fixed == ["tests/test_demo.py"]
        assert report.dropped_fix_deleted == []

    def test_drops_a_test_the_fix_deleted(self, tmp_path):
        buggy = tmp_path / "buggy"
        fixed = tmp_path / "fixed"
        write(buggy / "demo.py", "x = 1\n")
        write(buggy / "tests" / "test_gone.py", "def test_gone():\n    pass\n")
        write(fixed / "demo.py", "x = 2\n")

        dest = tmp_path / "vanta"
        report = tree_mod.compose_vanta_tree(buggy, fixed, dest, ["tests/test_gone.py"])

        assert not (dest / "tests" / "test_gone.py").exists()
        assert report.dropped_fix_deleted == ["tests/test_gone.py"]

    def test_helper_the_kept_tests_import_is_present_and_the_f2p_test_is_absent(
        self, tmp_path
    ):
        """The whole PySnooper/2 chain in one place: copy, then strip.

        After ``test_paths_from_fix`` brings the helper into ``vanta_tree`` and
        ``strip_hidden_tests`` removes the hidden test, the tree must contain the
        helper the kept test imports and must not contain the fail-to-pass test —
        neither by name nor by body.
        """
        buggy = tmp_path / "buggy"
        fixed = tmp_path / "fixed"
        kept_test = (
            "from . import mini_toolbox\n"
            "\n"
            "\n"
            "def test_kept():\n"
            "    assert mini_toolbox.CONTEXTMANAGER\n"
        )
        f2p_test = (
            "def test_hidden_f2p():\n"
            "    assert 'F2P-BODY-MARKER' == 'leaked'\n"
        )
        write(buggy / "demo.py", "VALUE = 'buggy'\n")
        write(buggy / "tests" / "__init__.py", "")
        write(buggy / "tests" / "test_pysnooper.py", kept_test)
        write(fixed / "demo.py", "VALUE = 'fixed'\n")
        write(fixed / "tests" / "__init__.py", "")
        write(fixed / "tests" / "test_pysnooper.py", kept_test + "\n\n" + f2p_test)
        write(fixed / "tests" / "mini_toolbox" / "__init__.py", "CONTEXTMANAGER = 1\n")

        bug = bug_info(test_file="tests/test_pysnooper.py", bug_patch_txt="")
        test_paths = tree_mod.test_paths_from_fix(bug, fixed)

        dest = tmp_path / "vanta"
        tree_mod.compose_vanta_tree(buggy, fixed, dest, test_paths)
        rel = "tests/test_pysnooper.py"
        tree_mod.strip_hidden_tests(
            dest,
            [f"{rel}::test_hidden_f2p"],
            {rel: {f"{rel}::test_kept", f"{rel}::test_hidden_f2p"}},
        )

        assert (dest / "tests" / "mini_toolbox" / "__init__.py").is_file(), (
            "the helper the kept test imports must survive into vanta_tree"
        )
        text = (dest / rel).read_text()
        assert "def test_kept" in text
        assert "test_hidden_f2p" not in text, "the F2P test must not survive by name"
        assert "F2P-BODY-MARKER" not in text, "nor may its body appear anywhere"


class TestStripHiddenTests:
    MIXED = (
        '"""Docstring."""\n'
        "\n"
        "import os\n"
        "\n"
        "\n"
        "def test_hidden():\n"
        '    assert os.sep == "/"\n'
        "\n"
        "\n"
        "def test_kept():\n"
        '    """A test that stays."""\n'
        "    assert True\n"
    )

    PARAM = (
        "import pytest\n"
        "\n"
        "\n"
        '@pytest.mark.parametrize("value", [1, 2])\n'
        "def test_hidden(value):\n"
        "    assert value\n"
        "\n"
        "\n"
        "def test_kept():\n"
        "    assert True\n"
    )

    DECORATED = (
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.mark.skip\n"
        "@pytest.mark.slow\n"
        "def test_hidden():\n"
        "    assert True\n"
        "\n"
        "\n"
        "def test_kept():\n"
        "    assert True\n"
    )

    CLASSES = (
        "import unittest\n"
        "\n"
        "\n"
        "class TestThing(unittest.TestCase):\n"
        "    def test_hidden(self):\n"
        '        self.fail("boom")\n'
        "\n"
        "\n"
        "class TestOther(unittest.TestCase):\n"
        "    def test_kept(self):\n"
        "        pass\n"
    )

    def test_removes_only_the_named_function(self, tmp_path):
        rel = "tests/test_x.py"
        write(tmp_path / rel, self.MIXED)
        report = tree_mod.strip_hidden_tests(
            tmp_path,
            [f"{rel}::test_hidden"],
            {rel: {f"{rel}::test_hidden", f"{rel}::test_kept"}},
        )
        new = (tmp_path / rel).read_text()
        assert "def test_hidden" not in new
        assert "def test_kept" in new
        assert '"""Docstring."""' in new, "the strip must not rewrite the whole file"
        assert report.deleted_files == []
        assert report.removed_nodeids == [f"{rel}::test_hidden"]
        ast.parse(new)

    def test_removes_decorators_with_the_function(self, tmp_path):
        rel = "tests/test_x.py"
        write(tmp_path / rel, self.DECORATED)
        tree_mod.strip_hidden_tests(
            tmp_path,
            [f"{rel}::test_hidden"],
            {rel: {f"{rel}::test_hidden", f"{rel}::test_kept"}},
        )
        new = (tmp_path / rel).read_text()
        assert "test_hidden" not in new
        assert "@pytest.mark.skip" not in new, "a decorator left behind is still a leak"
        assert "def test_kept" in new
        ast.parse(new)

    def test_deletes_the_file_when_every_collected_test_is_hidden(self, tmp_path):
        rel = "tests/test_only.py"
        write(tmp_path / rel, "def test_hidden():\n    assert True\n")
        report = tree_mod.strip_hidden_tests(
            tmp_path, [f"{rel}::test_hidden"], {rel: {f"{rel}::test_hidden"}}
        )
        assert not (tmp_path / rel).exists()
        assert report.deleted_files == [rel]

    def test_fully_hidden_function_is_not_reported_as_partial(self, tmp_path):
        rel = "tests/test_x.py"
        write(tmp_path / rel, self.PARAM)
        report = tree_mod.strip_hidden_tests(
            tmp_path,
            [f"{rel}::test_hidden[1]", f"{rel}::test_hidden[2]"],
            {
                rel: {
                    f"{rel}::test_hidden[1]",
                    f"{rel}::test_hidden[2]",
                    f"{rel}::test_kept",
                }
            },
        )
        assert report.partially_hidden_functions == []

    def test_surviving_parametrisation_is_reported_as_partial(self, tmp_path):
        """The oracle named one parametrisation; the function leaves with both.

        Without this the shrunken native suite is unexplainable from the split:
        `FAIL_TO_PASS` would list one nodeid while the tree lost a function.
        """
        rel = "tests/test_x.py"
        write(tmp_path / rel, self.PARAM)
        report = tree_mod.strip_hidden_tests(
            tmp_path,
            [f"{rel}::test_hidden[1]"],
            {
                rel: {
                    f"{rel}::test_hidden[1]",
                    f"{rel}::test_hidden[2]",
                    f"{rel}::test_kept",
                }
            },
        )
        new = (tmp_path / rel).read_text()
        assert "def test_hidden" not in new
        assert report.partially_hidden_functions == [f"{rel}::test_hidden"]
        ast.parse(new)

    def test_emptied_class_gets_a_pass(self, tmp_path):
        rel = "tests/test_x.py"
        write(tmp_path / rel, self.CLASSES)
        report = tree_mod.strip_hidden_tests(
            tmp_path,
            [f"{rel}::TestThing::test_hidden"],
            {
                rel: {
                    f"{rel}::TestThing::test_hidden",
                    f"{rel}::TestOther::test_kept",
                }
            },
        )
        new = (tmp_path / rel).read_text()
        assert new.count("pass") == 2
        assert "test_hidden" not in new
        assert "class TestOther" in new
        assert report.removed_nodeids == [f"{rel}::TestThing::test_hidden"]
        ast.parse(new)

    def test_missing_nodeid_is_fatal(self, tmp_path):
        """A hidden test the strip cannot find stays visible in Vanta's tree."""
        rel = "tests/test_x.py"
        write(tmp_path / rel, "def test_other():\n    assert True\n")
        with pytest.raises(tree_mod.StripError, match="still be visible"):
            tree_mod.strip_hidden_tests(
                tmp_path,
                [f"{rel}::test_absent"],
                {rel: {f"{rel}::test_other"}},
            )

    def test_unparseable_file_is_fatal(self, tmp_path):
        rel = "tests/test_x.py"
        write(tmp_path / rel, "def broken(:\n")
        with pytest.raises(tree_mod.StripError, match="cannot parse"):
            tree_mod.strip_hidden_tests(tmp_path, [f"{rel}::test_x"], {rel: set()})


# ============================================================================
# context.py — the precheck must measure Vanta's walk, not a similar one
# ============================================================================
class TestContextConstants:
    def test_match_src_repo_mod_rs(self):
        """A drift here makes the precheck measure a different prompt.

        The constants are duplicated in Python and Rust on purpose (the harness
        must not link the crate), so the duplication is asserted rather than
        trusted.
        """
        source = (REPO_ROOT / "src" / "repo" / "mod.rs").read_text(encoding="utf-8")

        def string_list(name: str) -> set[str]:
            body = re.search(
                rf"const {name}: &\[&str\] = &\[(.*?)\];", source, re.DOTALL
            )
            assert body, f"{name} not found in src/repo/mod.rs"
            return set(re.findall(r'"([^"]+)"', body.group(1)))

        def number(name: str) -> int:
            body = re.search(rf"const {name}: usize = ([^;]+);", source)
            assert body, f"{name} not found in src/repo/mod.rs"
            return eval(body.group(1), {"__builtins__": {}}, {})  # noqa: S307

        assert string_list("SOURCE_EXTENSIONS") == ctx.SOURCE_EXTENSIONS
        assert string_list("SKIP_DIRS") == ctx.SKIP_DIRS
        assert number("MAX_FILE_BYTES") == ctx.MAX_FILE_BYTES
        assert number("MAX_TOTAL_SOURCE_BYTES") == ctx.MAX_TOTAL_SOURCE_BYTES

    def test_the_estimator_and_the_output_reserve_match_the_run_time_guard(self):
        """The precheck's boundary must be the guard's boundary, not near it.

        The same three numbers appear in three files: the token estimator and the
        chat overhead in ``src/llm/mod.rs`` (where ``context_budget_exceeded``
        compares ``prompt + output + overhead > n_ctx``), and the output reserve in
        ``src/agents/creator.rs``. If the Creator's ``max_tokens`` grows and this
        module keeps subtracting 32768, the precheck silently starts accepting
        prompts the guard rejects — and the run dies at attempt 1 after a checkpoint
        and a wave have been spent. So the duplication is asserted, not trusted.
        """
        llm = (REPO_ROOT / "src" / "llm" / "mod.rs").read_text(encoding="utf-8")
        creator = (REPO_ROOT / "src" / "agents" / "creator.rs").read_text(encoding="utf-8")

        bytes_per_token = re.search(
            r"const TOKEN_ESTIMATE_BYTES_PER_TOKEN: u32 = (\d+);", llm
        )
        overhead = re.search(r"const CHAT_OVERHEAD_TOKENS: u32 = (\d+);", llm)
        assert bytes_per_token, "TOKEN_ESTIMATE_BYTES_PER_TOKEN not found in src/llm/mod.rs"
        assert overhead, "CHAT_OVERHEAD_TOKENS not found in src/llm/mod.rs"
        assert int(bytes_per_token.group(1)) == ctx.BYTES_PER_TOKEN_ESTIMATE
        assert int(overhead.group(1)) == ctx.CHAT_OVERHEAD_TOKENS

        max_tokens = re.search(r'"max_tokens":\s*(\d+)', creator)
        assert max_tokens, '"max_tokens" not found in src/agents/creator.rs'
        assert int(max_tokens.group(1)) == ctx.OUTPUT_RESERVE_TOKENS

    def test_estimate_tokens_rounds_up_like_the_rust_does(self):
        """``conservative_token_estimate`` ceils; a truncated mirror under-counts."""
        rust = (REPO_ROOT / "src" / "llm" / "mod.rs").read_text(encoding="utf-8")
        assert "bytes % TOKEN_ESTIMATE_BYTES_PER_TOKEN != 0" in rust  # the ceil

        assert ctx.estimate_tokens(0) == 0
        assert ctx.estimate_tokens(1) == 1
        assert ctx.estimate_tokens(4) == 1
        assert ctx.estimate_tokens(5) == 2
        assert ctx.estimate_tokens(4097) == 1025


class TestContextBudget:
    """``budget = n_ctx − output reserve − overhead − 25 % of n_ctx`` (§1.6.2)."""

    def test_the_headers_window_yields_the_documented_budget(self):
        budget, margin = ctx.budget_tokens(131_072)
        assert budget == 65_280
        assert margin == 32_768
        assert 131_072 - ctx.OUTPUT_RESERVE_TOKENS - ctx.CHAT_OVERHEAD_TOKENS - margin == budget

    def test_the_budget_is_just_under_vantas_own_source_cap(self):
        """A coincidence worth keeping an eye on: 65280 tok ~ 255 KB < 256 KB."""
        budget, _ = ctx.budget_tokens(131_072)
        assert budget * ctx.BYTES_PER_TOKEN_ESTIMATE < ctx.MAX_TOTAL_SOURCE_BYTES

    def test_headroom_is_what_the_prompt_may_still_grow_by(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="test")
        assert verdict.headroom_tokens == verdict.budget_tokens - verdict.estimated_prompt_tokens
        assert verdict.output_reserve_tokens == ctx.OUTPUT_RESERVE_TOKENS
        assert verdict.to_dict()["headroom_tokens"] == verdict.headroom_tokens

    def test_a_window_below_the_output_reserve_has_no_prompt_budget(self, tmp_path):
        """Every real prompt is rejected *and* the reason says the window itself is
        the problem — otherwise a misconfigured server looks like a fat repo."""
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "", n_ctx=32_768, n_ctx_source="test")
        assert verdict.context_ok is False
        assert any(r.startswith("no_prompt_budget") for r in verdict.reasons)
        assert verdict.budget_tokens <= 0

    def test_the_source_cap_alone_can_exceed_the_budget(self, tmp_path):
        """The 256 KB cap is *not* the binding constraint — this budget is.

        Filling Vanta's own total cap exactly (untruncated, so the walk reports no
        rejection of its own) still estimates at 65536 tok, and 65536 + the 3000-tok
        frame is over the 65280 budget. So a repo can pass Vanta's cap and still be
        refused here, for a reason that is about the model's window rather than
        about the walk.
        """
        for index in range(4):  # 4 x 64 KB = the total cap exactly, none oversize
            blob(tmp_path / f"f{index}.py", ctx.MAX_FILE_BYTES)
        verdict = ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="test")

        assert verdict.stats.bytes_included == ctx.MAX_TOTAL_SOURCE_BYTES
        assert verdict.stats.truncated is False
        assert verdict.reasons == [
            r for r in verdict.reasons if r.startswith("prompt_exceeds_budget")
        ]
        assert verdict.estimated_prompt_tokens == 65_536 + ctx.PROMPT_FRAME_TOKENS
        assert verdict.estimated_prompt_tokens > verdict.budget_tokens
        assert verdict.context_ok is False

    def test_the_precheck_is_stricter_than_the_guard_by_exactly_the_margin(self, tmp_path):
        """A prompt between the budget and the guard's limit: Vanta would run it,
        the harness refuses it — and the gap *is* the feedback-growth margin.

        The issue text is what makes that band reachable: it is not part of the
        source walk and therefore not capped by ``MAX_TOTAL_SOURCE_BYTES``.
        """
        n_ctx = 131_072
        budget, margin = ctx.budget_tokens(n_ctx)
        guard_limit = n_ctx - ctx.OUTPUT_RESERVE_TOKENS - ctx.CHAT_OVERHEAD_TOKENS
        assert guard_limit - budget == margin

        write(tmp_path / "a.py", "x = 1\n")
        issue = "a" * ((budget + 512 - ctx.PROMPT_FRAME_TOKENS - 2) * 4)
        verdict = ctx.precheck(tmp_path, issue, n_ctx=n_ctx, n_ctx_source="test")
        assert verdict.estimated_prompt_tokens > budget
        assert verdict.estimated_prompt_tokens <= guard_limit
        assert verdict.context_ok is False


class TestRenderContextSizes:
    """``context_sizes.md`` is read to *choose the pilot set*, so it states the rule."""

    def test_the_report_shows_the_budget_derivation(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="test")
        text = ctx.render_markdown([("Demo/1", "pilot", verdict)])
        assert "budget = n_ctx" in text
        assert "65280 = 131072 − 32768 − 256 − 32768" in text
        assert "`TOKEN_ESTIMATE_BYTES_PER_TOKEN`" in text
        assert "| Demo/1 | pilot | ok | 131072 (test)" in text
        assert "| headroom |" in text

    def test_a_mixed_window_is_called_out_rather_than_averaged(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        rows = [
            ("Demo/1", "pilot", ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="a")),
            ("Demo/2", "reserve", ctx.precheck(tmp_path, "", n_ctx=65_536, n_ctx_source="b")),
        ]
        text = ctx.render_markdown(rows)
        assert "More than one window was measured" in text

    def test_a_fallback_window_is_flagged_as_not_a_decision(self, tmp_path):
        """The first measure-only run silently used the 65536 fallback and rejected
        13 of 14 candidates; a report that can be read as a decision must say so."""
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(
            tmp_path, "", n_ctx=ctx.FALLBACK_CONTEXT_TOKENS,
            n_ctx_source="fallback (server unreachable)",
        )
        text = ctx.render_markdown([("Demo/1", "pilot", verdict)])
        assert "**Not a decision.**" in text
        assert f"against the {ctx.FALLBACK_CONTEXT_TOKENS} fallback" in text
        assert "VANTA_LLM_CONTEXT_TOKENS" in text


class TestSelectMeasureOnly:
    """``--measure-only`` is how the pilot set becomes a measured choice (§8)."""

    def test_measure_only_forces_every_candidate_to_context_only(self):
        split = split_mod.Split(
            split_data(
                bugs=[
                    bug_entry(role="pilot", bug_id=1),
                    bug_entry(role="reserve", bug_id=2),
                    bug_entry(role="pilot", bug_id=3),
                ]
            )
        )
        selected = run_mod.select(
            split, only=[], waves=[], reserves="measure", measure_only=True
        )
        assert [full for _, full in selected] == [False, False, False]
        assert len(selected) == 3

    def test_measure_only_also_covers_an_explicitly_named_bug(self):
        """``--only X --measure-only`` must not quietly boot a VM for X: the point
        of the flag is that no third-party code runs on the host or in Shuru."""
        split = split_mod.Split(split_data())
        selected = run_mod.select(
            split, only=["Demo/1"], waves=[], reserves="measure", measure_only=True
        )
        assert selected == [(split.bugs[0], False)]

    def test_without_the_flag_a_pilot_is_still_full(self):
        split = split_mod.Split(split_data(bugs=[bug_entry(role="pilot")]))
        selected = run_mod.select(split, only=[], waves=[], reserves="measure")
        assert selected == [(split.bugs[0], True)]

    def test_the_flag_exists_on_the_prepare_subcommand(self):
        args = run_mod.build_parser().parse_args(["prepare", "--measure-only", "--plan"])
        assert args.measure_only is True
        args = run_mod.build_parser().parse_args(["prepare", "--plan"])
        assert args.measure_only is False


class TestContextReport:
    """§1.6 measurements live here and in ``prepare.json``, never in the split."""

    @staticmethod
    def _verdict(tmp_path, n_ctx=131_072, source="test"):
        write(tmp_path / "a.py", "x = 1\n")
        return ctx.precheck(tmp_path, "", n_ctx=n_ctx, n_ctx_source=source)

    def test_the_report_carries_what_a_swap_decision_needs(self, tmp_path):
        verdict = self._verdict(tmp_path)
        path = prepare_mod.write_context_report(tmp_path, [prep(context=verdict)])
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["kind"] == "context_report"
        assert report["n_ctx"] == 131_072
        assert report["budget_tokens"] == 65_280
        assert report["fallback_window"] is False
        bug = report["bugs"][0]
        assert bug["bug"] == "Demo/1"
        assert bug["verdict"] == "ok"
        assert bug["estimated_prompt_tokens"] == verdict.estimated_prompt_tokens
        assert bug["budget_tokens"] == 65_280
        assert bug["headroom_tokens"] == verdict.headroom_tokens
        assert bug["source_bytes"] == verdict.stats.bytes_included
        assert bug["file_count"] == verdict.stats.files_included

    def test_real_tokens_are_null_until_a_run_has_paid_for_the_prompt(self, tmp_path):
        """The estimate is uncalibrated (~1.5x high on heretic), so the real number
        has to be a separate, explicitly-empty field rather than the same one."""
        verdict = self._verdict(tmp_path)
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(context=verdict)]
            ).read_text(encoding="utf-8")
        )
        bug = report["bugs"][0]
        assert bug["prompt_tokens_real"] is None
        assert bug["prompt_tokens_real_source"] is None
        assert bug["prompt_tokens_real"] != bug["estimated_prompt_tokens"]

    def test_a_reject_carries_its_reasons(self, tmp_path):
        for index in range(5):
            blob(tmp_path / f"mod{index}.py", 60_000)
        verdict = ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="test")
        assert verdict.context_ok is False
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(context=verdict)]
            ).read_text(encoding="utf-8")
        )
        bug = report["bugs"][0]
        assert bug["verdict"] == "reject"
        assert bug["reasons"] == verdict.reasons
        assert "truncated" in bug["reasons"][0]

    def test_a_bug_that_was_never_measured_says_why(self, tmp_path):
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(gate_failures=["boom"])]
            ).read_text(encoding="utf-8")
        )
        bug = report["bugs"][0]
        assert bug["verdict"] == "not_measured"
        assert bug["context_ok"] is None
        assert bug["estimated_prompt_tokens"] is None
        assert bug["reasons"] == ["boom"]

    def test_a_fallback_window_is_flagged_in_the_json(self, tmp_path):
        verdict = self._verdict(
            tmp_path,
            n_ctx=ctx.FALLBACK_CONTEXT_TOKENS,
            source="fallback (server unreachable)",
        )
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(context=verdict)]
            ).read_text(encoding="utf-8")
        )
        assert report["fallback_window"] is True
        assert report["n_ctx"] == ctx.FALLBACK_CONTEXT_TOKENS

    def test_the_table_prints_both_token_columns(self, tmp_path):
        verdict = self._verdict(tmp_path)
        table = prepare_mod.render_context_table(
            [prep(context=verdict), prep(bug=bug_info(bug_id=2))]
        )
        assert "est tok" in table
        assert "real tok" in table
        assert str(verdict.estimated_prompt_tokens) in table
        assert "—" in table
        assert "Demo/2" in table
        assert "not measured" in table

    def test_the_report_records_the_window_the_batch_was_sized_against(self, tmp_path):
        """The caller-resolved window wins over the per-verdict one: it is the one
        number ``run`` compares its own against (§1.6.1)."""
        verdict = self._verdict(tmp_path, n_ctx=131_072, source="server /props")
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(context=verdict)], n_ctx=131_072, n_ctx_source="--n-ctx"
            ).read_text(encoding="utf-8")
        )
        assert report["n_ctx"] == 131_072
        assert report["n_ctx_source"] == "--n-ctx"

    def test_the_report_still_records_the_window_with_no_measurements(self, tmp_path):
        """A batch where every bug failed the gate has no verdict to copy from, so
        the window has to come from the caller or the report leaves it implicit."""
        report = json.loads(
            prepare_mod.write_context_report(
                tmp_path, [prep(gate_failures=["boom"])],
                n_ctx=131_072, n_ctx_source="VANTA_LLM_CONTEXT_TOKENS",
            ).read_text(encoding="utf-8")
        )
        assert report["n_ctx"] == 131_072
        assert report["n_ctx_source"] == "VANTA_LLM_CONTEXT_TOKENS"


class TestTheSummaryNamesTheWindow:
    """``prepare_summary.md`` states which window it sized against, and its source."""

    def test_the_window_and_its_source_are_printed_above_the_table(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "", n_ctx=131_072, n_ctx_source="--n-ctx")
        text = prepare_mod.render_summary(
            [prep(context=verdict)], n_ctx=131_072, n_ctx_source="--n-ctx"
        )
        assert "n_ctx=131072" in text
        assert "--n-ctx" in text
        lines = text.splitlines()
        assert lines[0].startswith("Context window:"), "the window must come first"

    def test_without_a_window_the_line_is_simply_absent(self):
        assert "Context window:" not in prepare_mod.render_summary([prep()])


class TestTheSplitIsAnInput:
    """The split is hand-authored and reviewed; no stage may write to it.

    The rule is enforced against the sources rather than promised in a comment,
    because the failure it prevents is silent: a stage that rewrites the split turns
    a measurement into a decision while looking like it only recorded something.
    """

    @staticmethod
    def _source(name: str) -> str:
        return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["bugsinpy_run.py", "bugsinpy/prepare.py"])
    def test_no_production_module_mutates_the_split(self, name):
        text = self._source(name)
        assert "set_derived(" not in text, f"{name} mutates the split"
        assert ".save(" not in text, f"{name} saves the split"

    def test_prepare_writes_the_context_report_instead(self):
        text = self._source("bugsinpy_run.py")
        assert "write_context_report(" in text
        assert "render_context_table(" in text
        assert "split.save(" not in text

    def test_the_committed_split_has_no_derived_keys(self):
        """A run that rewrote the split would leave derived keys in the commit."""
        data = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        for entry in data["bugs"]:
            assert not (set(entry) & split_mod.DERIVED_BUG_KEYS), entry["project"]


class TestMeasureTree:
    def test_filters_extensions_and_skip_dirs(self, tmp_path):
        write(tmp_path / "pkg" / "mod.py", "x = 1\n")
        write(tmp_path / "tests" / "test_a.py", "def test_a():\n    pass\n")
        write(tmp_path / "README.md", "not source\n")
        write(tmp_path / "__pycache__" / "mod.py", "cached\n")
        write(tmp_path / "build" / "gen.py", "generated\n")
        stats = ctx.measure_tree(tmp_path)
        assert stats.files_eligible == 2
        assert stats.files_included == 2
        assert stats.truncated is False

    def test_oversize_file_is_skipped_not_truncated(self, tmp_path):
        blob(tmp_path / "big.py", ctx.MAX_FILE_BYTES + 1)
        write(tmp_path / "small.py", "x = 1\n")
        stats = ctx.measure_tree(tmp_path)
        assert stats.files_eligible == 2
        assert stats.files_skipped_oversize == 1
        assert stats.files_included == 1

    def test_total_budget_bites(self, tmp_path):
        for index in range(5):
            blob(tmp_path / f"f{index}.py", 60 * 1024)
        stats = ctx.measure_tree(tmp_path)
        assert stats.files_eligible == 5
        assert stats.files_included == 4  # 4 x 60 KB fits under 256 KB, the 5th does not
        assert stats.files_skipped_budget == 1
        assert stats.bytes_included <= ctx.MAX_TOTAL_SOURCE_BYTES
        assert stats.truncated is True

    def test_reports_the_largest_files(self, tmp_path):
        blob(tmp_path / "a.py", 300)
        blob(tmp_path / "b.py", 900)
        stats = ctx.measure_tree(tmp_path)
        assert stats.largest_files[0][0] == "b.py"


class TestPrecheck:
    def test_accepts_a_tree_that_fits(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "hello", n_ctx=65536, n_ctx_source="test")
        assert verdict.context_ok is True
        assert verdict.reasons == []
        assert verdict.budget_tokens == ctx.budget_tokens(65536)[0]

    def test_rejects_a_truncated_source(self, tmp_path):
        blob(tmp_path / "big.py", ctx.MAX_FILE_BYTES + 1)
        verdict = ctx.precheck(tmp_path, "", n_ctx=200_000, n_ctx_source="test")
        assert verdict.context_ok is False
        assert len(verdict.reasons) == 1
        assert verdict.reasons[0].startswith("source_truncated_by_cap")
        assert "may not be in it" in verdict.reasons[0]

    def test_rejects_a_prompt_over_budget(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "", n_ctx=4000, n_ctx_source="test")
        assert verdict.context_ok is False
        assert verdict.reasons[0].startswith("prompt_exceeds_budget")

    def test_the_margin_is_reserved_for_feedback_growth(self, tmp_path):
        """Same tree, two windows: the tighter one leaves no room for attempt 2."""
        write(tmp_path / "a.py", "x" * 3000 + "\n")
        tight = ctx.precheck(tmp_path, "", n_ctx=48_000, n_ctx_source="test")
        roomy = ctx.precheck(tmp_path, "", n_ctx=96_000, n_ctx_source="test")
        assert tight.context_ok is False
        assert roomy.context_ok is True
        assert roomy.headroom_tokens > tight.headroom_tokens

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "12345")
        assert ctx.server_context_window() == (12345, "VANTA_LLM_CONTEXT_TOKENS")

    def test_bad_env_override_is_an_error(self, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "lots")
        with pytest.raises(ctx.ContextError, match="not an integer"):
            ctx.server_context_window()

    def test_falls_back_when_the_server_is_unreachable(self, monkeypatch):
        monkeypatch.delenv("VANTA_LLM_CONTEXT_TOKENS", raising=False)
        monkeypatch.setenv("VANTA_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        assert ctx.server_context_window() == (
            ctx.FALLBACK_CONTEXT_TOKENS,
            "fallback (server unreachable)",
        )


class TestExplicitContextWindow:
    """The gate may only size against a window the run will actually use (§1.6.1).

    [VANTA-WHY] The first full prepare ran with ``VANTA_LLM_CONTEXT_TOKENS`` unset
    in the shell, fell through to the 65536 fallback and rejected PySnooper/1 and /2
    for a 16128-token budget instead of 65280 — a rejection about a prompt no run
    would have sent. ``explicit_context_window`` therefore *refuses* where
    ``server_context_window`` (the run-time mirror) falls back, and the two are
    tested apart on purpose: the mirror must keep falling back, the gate must not.
    """

    @staticmethod
    def _no_sources(monkeypatch):
        monkeypatch.delenv("VANTA_LLM_CONTEXT_TOKENS", raising=False)
        monkeypatch.setenv("VANTA_LLM_BASE_URL", "http://127.0.0.1:1/v1")

    def test_it_refuses_when_no_source_offers_a_window(self, monkeypatch):
        self._no_sources(monkeypatch)
        with pytest.raises(ctx.ContextError, match="refusing to run the context gate"):
            ctx.explicit_context_window()

    def test_the_refusal_names_the_three_ways_out(self, monkeypatch):
        self._no_sources(monkeypatch)
        with pytest.raises(ctx.ContextError) as excinfo:
            ctx.explicit_context_window()
        message = str(excinfo.value)
        assert "VANTA_LLM_CONTEXT_TOKENS" in message
        assert "--n-ctx" in message
        assert "/props" in message
        assert str(ctx.FALLBACK_CONTEXT_TOKENS) in message

    def test_the_precheck_goes_through_the_refusal_not_the_fallback(self, tmp_path, monkeypatch):
        """A caller that forgot to thread a window fails here rather than producing
        a verdict sized against 65536 that somebody then acts on."""
        write(tmp_path / "a.py", "x = 1\n")
        self._no_sources(monkeypatch)
        with pytest.raises(ctx.ContextError, match="refusing to run the context gate"):
            ctx.precheck(tmp_path, "")

    def test_the_env_variable_is_an_explicit_source(self, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "131072")
        assert ctx.explicit_context_window() == (131072, "VANTA_LLM_CONTEXT_TOKENS")

    def test_a_flag_that_disagrees_with_the_variable_is_refused(self, monkeypatch):
        """The flag is *checked*, not obeyed: a run resolves the window from the
        variable, so a flag that disagreed would size the gate against a window the
        run will not use — the footgun in a different coat."""
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "131072")
        monkeypatch.setenv("VANTA_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        with pytest.raises(ctx.ContextError, match="disagrees with VANTA_LLM_CONTEXT_TOKENS"):
            ctx.explicit_context_window(n_ctx=65_536)

    def test_a_flag_that_agrees_is_recorded_as_its_own_source(self, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "131072")
        monkeypatch.setenv("VANTA_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        assert ctx.explicit_context_window(n_ctx=131_072) == (131_072, "--n-ctx")

    def test_the_flag_exists_on_the_prepare_subcommand(self):
        args = run_mod.build_parser().parse_args(["prepare", "--n-ctx", "131072", "--plan"])
        assert args.n_ctx == 131_072
        assert run_mod.build_parser().parse_args(["prepare", "--plan"]).n_ctx is None


# ============================================================================
# issue.py — the issue text must never leak the answer
# ============================================================================
class TestStripDiffAndCode:
    MESSAGE = (
        "Fix the parser\n"
        "\n"
        "The parser mishandled an empty flag and reported success.\n"
        "\n"
        "```python\n"
        "def parse(value):\n"
        "    return None\n"
        "```\n"
        "\n"
        "diff --git a/parser.py b/parser.py\n"
        "index abc..def 100644\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n"
        "\n"
        "    indented_trailer()\n"
        "\n"
        "Regards\n"
    )

    def test_removes_diffs_fences_and_indented_blocks(self):
        stripped = issue_mod.strip_diff_and_code(self.MESSAGE)
        assert "diff --git" not in stripped
        assert "@@" not in stripped
        assert "+++ b/parser.py" not in stripped
        assert "-old" not in stripped
        assert "+new" not in stripped
        assert "def parse" not in stripped
        assert "indented_trailer" not in stripped
        assert "Fix the parser" in stripped
        assert "mishandled an empty flag" in stripped
        assert "Regards" in stripped

    def test_removes_inline_code_spans(self):
        assert "value" not in issue_mod.strip_diff_and_code("Fix the `value` handling\n")

    def test_collapses_blank_runs(self):
        assert "\n\n\n" not in issue_mod.strip_diff_and_code("a\n\n\n\n\nb\n")


class TestLeakGate:
    def test_strips_an_answer_only_symbol(self):
        result = issue_mod.leak_gate(
            "The parser mishandles fix_widget_flag when the input is empty and short.",
            answer_tokens={"fix_widget_flag": 2},
            tree_tokens={},
        )
        assert "fix_widget_flag" not in result.text
        assert "[redacted]" in result.text
        assert result.text.replace("[redacted]", "").count("parser") == 1, (
            "only the leaking token may change"
        )
        assert len(result.removals) == 1
        assert result.removals[0].token == "fix_widget_flag"
        assert result.removals[0].occurrences_in_answer == 2
        assert result.removals[0].occurrences_in_vanta_tree == 0
        assert result.ok is True

    def test_keeps_a_token_the_tree_also_contains(self):
        """The whole point of the gate: present in the tree means not a leak."""
        result = issue_mod.leak_gate(
            "The widget handling is wrong and it should be corrected.",
            answer_tokens={"widget": 4},
            tree_tokens={"widget": 9},
        )
        assert "widget" in result.text
        assert result.removals == []

    def test_lists_short_prose_without_stripping_it(self):
        """Short unadorned words carry no answer; gating them shreds the text."""
        result = issue_mod.leak_gate(
            "The parser mishandles the empty input and returns the wrong answer now.",
            answer_tokens={"the": 7},
            tree_tokens={},
        )
        assert result.text.count("the") > 0
        assert result.removals == []
        assert "the" in result.ungated_prose

    def test_is_case_and_separator_insensitive(self):
        result = issue_mod.leak_gate(
            "The FixWidgetFlag path is broken and needs attention here.",
            answer_tokens={"fixwidgetflag": 1, "fix_widget_flag": 1},
            tree_tokens={},
        )
        assert "FixWidgetFlag" not in result.text

    def test_removes_a_path_or_literal_that_exists_only_in_the_answer(self):
        result = issue_mod.leak_gate(
            "The failure comes from hidden_helper.py and 4096 is returned instead.",
            answer_tokens={"hidden_helper.py": 1, "4096": 1},
            tree_tokens={},
        )
        assert "hidden_helper.py" not in result.text
        assert "4096" not in result.text

    def test_answer_shaped_rules(self):
        assert issue_mod.is_answer_shaped("fix_widget", "ident") is True
        assert issue_mod.is_answer_shaped("widget", "ident") is True  # six characters
        assert issue_mod.is_answer_shaped("widget2", "ident") is True
        assert issue_mod.is_answer_shaped("fix", "ident") is False
        assert issue_mod.is_answer_shaped("42", "number") is True
        assert issue_mod.is_answer_shaped('"x"', "string") is True
        assert issue_mod.is_answer_shaped("a.py", "path") is True
        assert issue_mod.is_answer_shaped("self.widget", "ident") is True  # bare name, 6 chars
        assert issue_mod.is_answer_shaped("self.fix", "ident") is False

    def test_normalise_token(self):
        assert issue_mod.normalise_token('"Foo"') == "foo"
        assert issue_mod.normalise_token("self.Foo") == "foo"
        assert issue_mod.normalise_token("foo-bar") == "foo_bar"

    def test_tokenize_classes(self):
        classes = [
            klass for _, klass in issue_mod.tokenize("tests/test_a.py::test_b and 42 and 'q'")
        ]
        assert "path" in classes
        assert "number" in classes
        assert "string" in classes
        assert "ident" in classes


class TestActionability:
    def test_too_few_words_is_not_actionable(self):
        ok, reason = issue_mod.is_actionable("the fix is small")
        assert ok is False
        assert "actionable words" in reason

    def test_mostly_redacted_is_not_actionable(self):
        ok, reason = issue_mod.is_actionable("[redacted] " * 10 + "a")
        assert ok is False
        assert "removed more text than it left" in reason

    def test_prose_is_actionable(self):
        ok, reason = issue_mod.is_actionable(
            "The parser mishandles an empty flag and reports success instead of failing."
        )
        assert ok is True
        assert reason is None


class TestBehaviouralLine:
    def test_names_the_failure_kind_and_nothing_else(self):
        line = issue_mod.default_behavioural_line("stripped", ["AssertionError"])
        assert "AssertionError" in line
        assert "==" not in line
        assert "expected" not in line.lower() or "Expected behaviour" in line

    def test_without_kinds_still_says_something_behavioural(self):
        line = issue_mod.default_behavioural_line("stripped", [])
        assert "fails" in line

    def test_failure_kinds_take_the_class_not_the_message(self):
        report = {
            "tests": [
                {
                    "nodeid": "tests/test_a.py::test_b",
                    "call": {"longrepr": "E   AssertionError: assert 1 == 2"},
                },
                {"nodeid": "tests/test_a.py::test_c", "call": {"outcome": "passed"}},
            ]
        }
        kinds = issue_mod.failure_kinds_from_report(report)
        assert kinds == ["AssertionError"]
        assert all("1 == 2" not in kind for kind in kinds)

    def test_failure_kinds_read_setup_errors(self):
        report = {
            "tests": [
                {"nodeid": "t.py::test_a", "setup": {"longrepr": "ImportError: no module"}}
            ]
        }
        assert issue_mod.failure_kinds_from_report(report) == ["ImportError"]


class TestBuildIssueText:
    def test_assembles_and_gates(self, tmp_path):
        issue = issue_mod.build_issue_text(
            stripped_message=(
                "Fix the parser\n\nThe parser mishandled an empty flag and it returned "
                "success instead of failing on the bad input.\n"
            ),
            failure_kinds=["AssertionError"],
            answer_tokens={"bad_helper": 1},
            tree_tokens={},
        )
        assert "The parser mishandled" in issue.text
        assert "AssertionError" in issue.text
        assert issue.ok is True
        assert issue.needs_decision is None

    def test_flags_an_unactionable_text_instead_of_guessing(self):
        """The behavioural hook is what makes the text readable, so an empty one
        is the only way to reach the flag — and the flag must fire, because the
        harness cannot invent a better issue text."""
        issue = issue_mod.build_issue_text(
            stripped_message="Fix it\n",
            failure_kinds=[],
            answer_tokens={},
            tree_tokens={},
            paraphrase=lambda body, kinds: "",
        )
        assert issue.ok is False
        assert issue.needs_decision, "an unactionable text must be flagged for a human"
        assert issue.pre_gate == "Fix it"
        assert issue.text == "Fix it"


# ============================================================================
# shuru.py — how an argv becomes a guest shell line (no VM)
# ============================================================================
class TestShuruCommandQuoting:
    """``run(command=...)`` receives an *argv*, and its elements must survive as
    single elements.

    [WHY] The first version joined the elements with ``&&`` and pasted them into the
    shell line. That is invisible while every element is a bare word — and every
    caller but the marker probe is. The probe passes
    ``["sh", "-c", "test -f /opt/vanta-deps-ok"]``, which became
    ``cd /workspace && sh && -c && test -f /opt/vanta-deps-ok``: a bare ``sh``
    reading an inherited stdin, blocking until its timeout, on two healthy
    checkpoints in a row. The bake then deleted a 17 GB layer that was fine and
    reported "marker missing", which is a *result*, not an error — exactly the
    class of bug this file exists to make impossible.
    """

    def _guest_line(self, monkeypatch, command, *, env=None):
        seen: dict = {}

        def fake_run(args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(shuru_mod, "_shuru_bin", lambda: "/usr/bin/shuru")
        monkeypatch.setattr(shuru_mod.subprocess, "run", fake_run)
        result = shuru_mod.run(
            checkpoint="vanta-x",
            host_root=Path.cwd(),
            command=command,
            disk_mb=8192,
            env=env,
        )
        assert result.ok
        # ``--`` then the guest's own argv: sh -c <line>
        guest = seen["args"]
        assert guest[guest.index("--") + 1 : guest.index("--") + 3] == ["sh", "-c"]
        assert seen["kwargs"].get("stdin") is not None, "stdin must be closed"
        return guest[guest.index("--") + 3]

    def test_a_quoted_argument_stays_one_argument(self, monkeypatch):
        line = self._guest_line(
            monkeypatch, ["sh", "-c", "test -f /opt/vanta-deps-ok"]
        )
        assert " 'test -f /opt/vanta-deps-ok'" in line
        assert "-c &&" not in line
        assert "&& sh &&" not in line

    def test_a_plain_argv_is_not_mangled(self, monkeypatch):
        line = self._guest_line(
            monkeypatch, ["git", "apply", "--verbose", "/workspace/p.diff"]
        )
        assert line.endswith("git apply --verbose /workspace/p.diff")

    def test_an_env_value_with_spaces_is_quoted(self, monkeypatch):
        line = self._guest_line(
            monkeypatch, ["python", "-c", "import x"], env={"PYTHONPATH": "/a b"}
        )
        assert "export PYTHONPATH='/a b'" in line

    def test_a_multiline_script_survives(self, monkeypatch):
        script = "set +e\necho one\necho two\nexit 0"
        line = self._guest_line(monkeypatch, ["sh", "-c", script])
        assert shlex.quote(script) in line


# ============================================================================
# suite.py — command construction and report reading (no VM)
# ============================================================================
class TestSuiteCommand:
    def test_substitutes_the_report_placeholder(self):
        cmd = suite_mod.build_command(
            ["python", "-m", "pytest", "tests", "--json-report-file={report}"],
            "/workspace/reports/a.json",
        )
        assert cmd[-1] == "--json-report-file=/workspace/reports/a.json"

    def test_requires_the_placeholder(self):
        with pytest.raises(suite_mod.SuiteError, match="placeholder"):
            suite_mod.build_command(["python", "-m", "pytest", "tests"], "/workspace/r.json")


class TestSuiteReport:
    def test_parse_report(self, tmp_path):
        path = write(
            tmp_path / "r.json",
            json.dumps(
                {
                    "tests": [
                        {"nodeid": "tests/test_a.py::test_b", "outcome": "failed"},
                        {"nodeid": "tests/test_a.py::test_c", "outcome": "passed"},
                    ]
                }
            ),
        )
        assert suite_mod.parse_report(path) == {
            "tests/test_a.py::test_b": "failed",
            "tests/test_a.py::test_c": "passed",
        }

    def test_parse_report_missing_file(self, tmp_path):
        with pytest.raises(suite_mod.SuiteError, match="cannot read"):
            suite_mod.parse_report(tmp_path / "nope.json")

    def test_collect_by_file(self):
        assert suite_mod.collect_by_file(
            ["tests/test_a.py::test_b", "tests/test_a.py::test_c", "t.py::test_d"]
        ) == {
            "tests/test_a.py": {"tests/test_a.py::test_b", "tests/test_a.py::test_c"},
            "t.py": {"t.py::test_d"},
        }


class TestSuitePair:
    def _pair(self, first, second):
        return suite_mod.compare_runs(
            suite_mod.SuiteRun(outcomes=first),
            suite_mod.SuiteRun(outcomes=second),
        )

    def test_stable_failures_excludes_flaky(self):
        pair = self._pair(
            {"a": "failed", "b": "failed", "c": "passed"},
            {"a": "failed", "b": "passed", "c": "passed"},
        )
        assert pair.flaky_nodeids == {"b"}
        assert pair.stable_failures == {"a"}
        assert pair.stable_collected == {"a", "c"}

    def test_a_nodeid_missing_from_one_run_is_not_stable(self):
        pair = self._pair({"a": "failed"}, {})
        assert pair.missing_from_one_run == {"a"}
        assert pair.stable_outcomes == {}

    def test_error_counts_as_a_failure(self):
        pair = self._pair({"a": "error"}, {"a": "error"})
        assert pair.stable_failures == {"a"}

    def test_a_run_that_differs_only_in_flaky_nodes_has_no_stable_failures(self):
        """The whole point of the second run: an unstable nodeid is never ground truth."""
        pair = self._pair({"a": "failed"}, {"a": "error"})
        assert pair.stable_failures == set()
        assert pair.stable_collected == set()


class TestDropTestFileHunks:
    DIFF = (
        "diff --git a/parser.py b/parser.py\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/tests/test_parser.py b/tests/test_parser.py\n"
        "--- a/tests/test_parser.py\n"
        "+++ b/tests/test_parser.py\n"
        "@@ -1 +1 @@\n"
        "-assert False\n"
        "+assert True\n"
    )

    def test_removes_test_sections_and_keeps_source(self):
        kept = suite_mod.drop_test_file_hunks(self.DIFF)
        assert "diff --git a/parser.py" in kept
        assert "+new" in kept
        assert "tests/test_parser.py" not in kept
        assert "assert True" not in kept

    def test_test_only_patch_reduces_to_nothing(self):
        only_test = self.DIFF.split("diff --git a/tests/", 1)
        body = "diff --git a/tests/" + only_test[1]
        assert suite_mod.drop_test_file_hunks(body).strip() == ""


# ============================================================================
# prepare.py — orchestration invariants
# ============================================================================
class TestPrepareInvariants:
    def test_derive_verdict_keys_are_exactly_the_derived_keys(self):
        """A key drift here writes an unvalidatable split, or silently drops one."""
        assert set(prepare_mod.derive_verdict(prep())) == split_mod.DERIVED_BUG_KEYS

    def test_every_derived_field_survives_set_derived(self):
        split = split_mod.Split(split_data())
        split.set_derived("Demo", 1, **prepare_mod.derive_verdict(prep()))
        entry = split.bug("Demo", 1)
        assert entry["prepare_mode"] == prepare_mod.FULL
        assert entry["context_ok"] is False

    def test_issue_hash_is_derived_from_the_text(self):
        issue = issue_mod.IssueText(
            text="Fix the parser.",
            pre_gate="Fix the parser.",
            leak=issue_mod.LeakResult(text="Fix the parser."),
            failure_kinds=[],
            ok=True,
            needs_decision=None,
        )
        verdict = prepare_mod.derive_verdict(prep(issue=issue))
        assert verdict["issue_text_sha256"] == split_mod.sha256_text("Fix the parser.")

    def test_runnable_requires_full_preparation(self):
        assert prep().runnable is False, "no context, no issue"
        assert prep(prepare_mode=prepare_mod.MEASURE_ONLY).runnable is False

    def test_runnable_requires_a_clean_gate(self):
        good = prep(
            context=ctx.ContextVerdict(True, [], 1, 2, 3, "test", ctx.ConcatStats()),
            issue=issue_mod.IssueText(
                text="x",
                pre_gate="x",
                leak=issue_mod.LeakResult(text="x"),
                failure_kinds=[],
                ok=True,
                needs_decision=None,
            ),
        )
        assert good.runnable is True
        with_gate = prep(
            context=good.context, issue=good.issue, gate_failures=["split drifted"]
        )
        assert with_gate.runnable is False

    def test_check_declared_matches(self):
        assert prepare_mod.check_declared(bug_entry(), bug_info()) == []

    def test_check_declared_flags_a_moved_commit(self):
        mismatches = prepare_mod.check_declared(bug_entry(), bug_info(fixed_commit="0" * 40))
        assert len(mismatches) == 1
        assert "fixed_commit" in mismatches[0]

    def test_check_declared_requires_needs_build_when_pythonpath_is_set(self):
        entry = bug_entry(pythonpath="demo/build/lib/")
        bug = bug_info(pythonpath="demo/build/lib/")
        mismatches = prepare_mod.check_declared(entry, bug)
        assert len(mismatches) == 1
        assert "needs_build" in mismatches[0]
        assert prepare_mod.check_declared(
            bug_entry(pythonpath="demo/build/lib/", needs_build=True), bug
        ) == []

    def test_require_checkpoints_names_every_missing_layer(self, monkeypatch):
        # Patched on ``bake``, not on ``prepare``: prepare has to reach the layer
        # through the same function the bake decides skip-vs-rebuild with, and this
        # is what makes that shared. A patch on prepare's own module namespace would
        # pass while the two stages used different checks.
        monkeypatch.setattr(
            bake_mod, "layer_exists", lambda name: name == "vanta-bugsinpy-py3.8"
        )
        with pytest.raises(prepare_mod.PrepareError) as excinfo:
            prepare_mod.require_checkpoints(
                ["vanta-bugsinpy-py3.8", "vanta-bi-thefuck-1", "vanta-bi-httpie-1"]
            )
        message = str(excinfo.value)
        assert "vanta-bi-httpie-1" in message
        assert "vanta-bi-thefuck-1" in message
        assert "vanta-bugsinpy-py3.8" not in message

    def test_prepare_and_bake_ask_the_same_question_about_a_layer(self, monkeypatch):
        """``require_checkpoint`` and ``bake.frozen`` must not be two opinions."""
        for seen in (True, False):
            monkeypatch.setattr(bake_mod, "layer_exists", lambda name, seen=seen: seen)
            assert bake_mod.frozen("vanta-bi-Demo-1") is seen
            if seen:
                prepare_mod.require_checkpoint("vanta-bi-Demo-1")
            else:
                with pytest.raises(prepare_mod.PrepareError, match="does not exist"):
                    prepare_mod.require_checkpoint("vanta-bi-Demo-1")

    def test_render_summary_lists_bugs_and_gate_failures(self):
        summary = prepare_mod.render_summary(
            [prep(), prep(bug=bug_info(project="Other", bug_id=2), gate_failures=["boom"])]
        )
        assert "Demo/1" in summary
        assert "**gate**" in summary
        assert "**Other/2**: boom" in summary

    def test_measure_only_bugs_report_no_suite_numbers(self):
        summary = prepare_mod.render_summary([prep(prepare_mode=prepare_mod.MEASURE_ONLY)])
        assert "measured" in summary
        assert "—" in summary


class TestPrepareRunRecordContract:
    """``prepare.json`` has one writer and one reader, and they cannot drift.

    [VANTA-WHY] The first smoke run crashed in ``runner.load_prepared`` with
    ``TypeError: IssueText.__init__() missing 3 required positional arguments:
    'leak', 'failure_kinds', and 'needs_decision'`` — ``IssueText`` had grown three
    fields and the runner's hand-written reader was still constructing the old
    shape. The reader and the writer were two copies of one format living in two
    modules. The round trip below is the test that would have caught it: it builds a
    prep the way ``prepare`` does, writes it with prepare's own writer, and reads it
    back with the runner's — asserting every field, not just the ones the crash
    happened to touch.

    Nothing here touches a VM or a clone: ``load_bug`` is injected, because the
    contract under test is the record's shape, not BugsInPy's metadata.
    """

    @staticmethod
    def _issue() -> issue_mod.IssueText:
        """A *real* ``IssueText`` from the real builder, with all six fields set.

        The message is nothing but answer-only tokens, so the leak gate strips all
        of them and ``is_actionable`` refuses the result: ``ok`` is False and
        ``needs_decision`` carries the reason. Those two are exactly the fields the
        old reader guessed at, so a round trip that loses them has to fail here.
        """
        built = issue_mod.build_issue_text(
            stripped_message="fix_widget_flag alpha_widget beta_widget",
            failure_kinds=["AssertionError"],
            answer_tokens={"fix_widget_flag": 2, "alpha_widget": 1, "beta_widget": 1},
            tree_tokens={},
            paraphrase=lambda body, kinds: "",
        )
        assert built.needs_decision is not None, "the fixture must exercise the field"
        assert built.ok is False
        assert len(built.leak.removals) == 3
        return built

    @staticmethod
    def _write_batch(tmp_path: Path, *, preps, n_ctx=131_072, n_ctx_source="--n-ctx"):
        """Write ``prepare.json`` with the writer the *stage* uses, not a copy of it."""
        split = split_mod.Split(split_data())
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest = prepare_mod.prepare_manifest(
            split=split,
            split_path=tmp_path / "split.json",
            split_sha_before="0" * 64,
            batch_dir=batch_dir,
            commit="a" * 40,
            bugsinpy_root=tmp_path / "bugsinpy",
            selected=["Demo/1"],
            broken=[],
            preps=list(preps),
            n_ctx=n_ctx,
            n_ctx_source=n_ctx_source,
        )
        (batch_dir / "prepare.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return split, batch_dir, manifest

    @staticmethod
    def _store_manifest(batch_dir: Path, manifest: dict) -> None:
        (batch_dir / "prepare.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _load(monkeypatch, split, batch_dir, tmp_path):
        monkeypatch.setattr(
            prepare_mod, "load_bug", lambda split, root, project, bug_id: bug_info()
        )
        return runner_mod.load_prepared(
            batch_dir=batch_dir, split=split, bugsinpy_root=tmp_path / "bugsinpy"
        )

    def test_a_prepared_bug_round_trips_every_field(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        write(tree / "pkg" / "mod.py", "def parse():\n    return None\n")
        issue = self._issue()
        verdict = ctx.precheck(tree, issue.text, n_ctx=131_072, n_ctx_source="--n-ctx")
        deps = {
            "checkpoint": "vanta-bi-demo-1",
            "installs": ["pytest==8.0.2", "pytest-json-report==1.5.0"],
            "dependencies": [{"name": "pytest", "version": "8.0.2"}],
        }
        original = prep(
            root=tmp_path / "batch" / "worktrees" / "Demo_1",
            prepare_mode=prepare_mod.FULL,
            test_paths=["tests"],
            fail_to_pass=["tests/test_demo.py::test_a"],
            pass_to_pass_count=7,
            flaky_nodeids=["tests/test_demo.py::test_flaky"],
            preexisting_failures=["tests/test_demo.py::test_pre"],
            native_failure_baseline=["tests/test_demo.py::test_native"],
            native_green_on_vanta_tree=True,
            hidden_files_deleted=["tests/test_hidden.py"],
            partially_hidden_functions=["hidden_fn"],
            gate_failures=[],
            context=verdict,
            issue=issue,
            deps=deps,
        )
        split, batch_dir, _ = self._write_batch(tmp_path, preps=[original])
        batch = self._load(monkeypatch, split, batch_dir, tmp_path)

        assert batch.n_ctx == 131_072
        assert batch.n_ctx_source == "--n-ctx"
        assert len(batch.preps) == 1
        loaded = batch.preps[0]

        # Injected rather than stored: the metadata, the batch directory and the
        # reviewed split's own command.
        assert loaded.bug == original.bug
        assert loaded.root == batch_dir / "worktrees" / "Demo_1"
        assert loaded.test_cmd == [
            "python", "-m", "pytest", "tests", "--json-report-file={report}"
        ]

        assert loaded.role == original.role
        assert loaded.wave == original.wave
        assert loaded.prepare_mode == original.prepare_mode
        assert loaded.test_paths == original.test_paths
        assert loaded.fail_to_pass == sorted(original.fail_to_pass)
        assert loaded.pass_to_pass_count == original.pass_to_pass_count
        assert loaded.flaky_nodeids == sorted(original.flaky_nodeids)
        assert loaded.preexisting_failures == sorted(original.preexisting_failures)
        assert loaded.native_failure_baseline == sorted(original.native_failure_baseline)
        assert loaded.native_green_on_vanta_tree is True
        assert loaded.hidden_files_deleted == original.hidden_files_deleted
        assert loaded.partially_hidden_functions == original.partially_hidden_functions
        assert loaded.gate_failures == original.gate_failures
        # The pair the crash was about, and the two that were never restored at all.
        assert loaded.issue == issue
        assert loaded.issue.needs_decision == issue.needs_decision
        assert loaded.issue.leak.removals == issue.leak.removals
        assert loaded.context == verdict
        assert loaded.deps == deps
        assert loaded.runnable == original.runnable
        assert batch.entries["Demo/1"]["checkpoint"] == "vanta-bi-demo-1"

    @pytest.mark.parametrize(
        "drop",
        [
            pytest.param(("issue", "leak"), id="issue.leak"),
            pytest.param(("issue", "needs_decision"), id="issue.needs_decision"),
            pytest.param(("context",), id="context"),
            pytest.param(("deps",), id="deps"),
        ],
    )
    def test_a_missing_field_names_the_field_and_the_batch(
        self, tmp_path, monkeypatch, drop
    ):
        """A reader that fell behind the writer has to say *what* and *where*.

        The crash this replaces was a ``TypeError`` naming neither the field nor the
        batch, which is why it took a full prepare and a smoke run to diagnose.
        """
        tree = tmp_path / "tree"
        write(tree / "a.py", "x = 1\n")
        issue = self._issue()
        verdict = ctx.precheck(tree, issue.text, n_ctx=131_072, n_ctx_source="--n-ctx")
        original = prep(context=verdict, issue=issue, deps={})
        split, batch_dir, manifest = self._write_batch(tmp_path, preps=[original])

        target = manifest["bugs"][0]
        for key in drop[:-1]:
            target = target[key]
        del target[drop[-1]]
        self._store_manifest(batch_dir, manifest)

        with pytest.raises(runner_mod.RunnerError) as excinfo:
            self._load(monkeypatch, split, batch_dir, tmp_path)
        message = str(excinfo.value)
        assert repr(drop[-1]) in message, message
        assert "Demo/1" in message, message
        assert str(batch_dir) in message, message

    def test_a_context_the_reader_failed_to_restore_is_loud(self, tmp_path, monkeypatch):
        """The other drift, found while fixing the first one.

        ``prepare`` wrote ``context`` and ``deps`` and the old reader restored
        *neither*, while every other field was read with a ``.get(name, default)``
        fallback that hid the same kind of gap. ``BugPrep.runnable`` requires
        ``context``, so a loaded batch could never pass ``select_preps`` — the run
        would have failed with "these bugs did not prepare as runners" about bugs
        that had prepared perfectly.
        """
        tree = tmp_path / "tree"
        write(tree / "a.py", "x = 1\n")
        issue = self._issue()
        verdict = ctx.precheck(tree, issue.text, n_ctx=131_072, n_ctx_source="--n-ctx")
        original = prep(context=verdict, issue=issue, deps={})
        assert original.runnable is False  # needs_decision: not a candidate to run

        runnable = prep(
            context=verdict,
            issue=issue_mod.IssueText(
                text="Fix the parser.",
                pre_gate="Fix the parser.",
                leak=issue_mod.LeakResult(text="Fix the parser."),
                failure_kinds=["AssertionError"],
                ok=True,
                needs_decision=None,
            ),
            deps={},
        )
        assert runnable.runnable is True

        split, batch_dir, _ = self._write_batch(tmp_path, preps=[runnable])
        loaded = self._load(monkeypatch, split, batch_dir, tmp_path).preps[0]
        assert loaded.runnable is True, "a restored prep must still be a runner"
        assert loaded.context is not None and loaded.deps == {}

    def test_a_bare_string_where_a_list_belongs_is_an_error(self, tmp_path, monkeypatch):
        """``strings_of`` checks the shape too: a bare string read as a list would
        silently become a list of characters."""
        original = prep()
        split, batch_dir, manifest = self._write_batch(tmp_path, preps=[original])
        manifest["bugs"][0]["fail_to_pass"] = "tests/test_demo.py::test_a"
        self._store_manifest(batch_dir, manifest)
        with pytest.raises(runner_mod.RunnerError) as excinfo:
            self._load(monkeypatch, split, batch_dir, tmp_path)
        assert "'fail_to_pass'" in str(excinfo.value)
        assert "not a list" in str(excinfo.value)

    def test_a_non_integer_recorded_window_is_an_error(self, tmp_path, monkeypatch):
        split, batch_dir, manifest = self._write_batch(tmp_path, preps=[prep()])
        manifest["n_ctx"] = "131072"
        self._store_manifest(batch_dir, manifest)
        with pytest.raises(runner_mod.RunnerError, match="'n_ctx'"):
            self._load(monkeypatch, split, batch_dir, tmp_path)

    def test_a_batch_with_no_prepare_json_says_how_to_make_one(self, tmp_path):
        split = split_mod.Split(split_data())
        with pytest.raises(runner_mod.RunnerError, match="no prepare.json"):
            runner_mod.load_prepared(
                batch_dir=tmp_path, split=split, bugsinpy_root=tmp_path / "bugsinpy"
            )


class TestContextWindowWarning:
    """A run whose window differs from the batch's is measuring a different prompt.

    ``prepare`` refuses to gate against a window the run will not use; this is the
    other half of that rule (§1.6.1) — the run still resolves its own window the way
    ``llm::context_window_tokens`` does, so the two can drift between the prepare
    and the run, and the drift is invisible without a warning.
    """

    @staticmethod
    def _batch(tmp_path, *, n_ctx, source="test"):
        return runner_mod.PreparedBatch(
            batch_dir=tmp_path, preps=[], entries={}, n_ctx=n_ctx, n_ctx_source=source
        )

    def test_no_warning_when_the_windows_agree(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "131072")
        assert runner_mod.context_window_warning(self._batch(tmp_path, n_ctx=131_072)) is None

    def test_a_mismatch_names_both_windows_and_both_budgets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "65536")
        warning = runner_mod.context_window_warning(
            self._batch(tmp_path, n_ctx=131_072, source="--n-ctx")
        )
        assert warning is not None
        assert "131072" in warning and "65536" in warning
        assert "65280" in warning, "the budget prepare gated against"
        assert "16128" in warning, "the budget this run would use"
        assert "--n-ctx" in warning
        assert "VANTA_LLM_CONTEXT_TOKENS" in warning

    def test_a_batch_that_recorded_no_window_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VANTA_LLM_CONTEXT_TOKENS", "131072")
        warning = runner_mod.context_window_warning(self._batch(tmp_path, n_ctx=None))
        assert warning is not None
        assert "records no n_ctx" in warning


# ============================================================================
# bake.py — the environment layers (§3, §11 step 3)
# ============================================================================
class TestBake:
    def test_minor_version_drops_the_patch_level(self):
        assert bake_mod.minor_version("3.8.1") == "3.8"
        assert bake_mod.minor_version("3.6.9") == "3.6"
        assert bake_mod.minor_version("3.8") == "3.8"

    def test_minor_version_rejects_a_non_version(self):
        with pytest.raises(SplitError, match="minor version"):
            bake_mod.minor_version("py38")

    def test_interpreter_layer_is_named_by_the_minor_version(self):
        """BugsInPy writes patch levels; §3.2 names the layers py3.6/3.7/3.8.

        Keying on ``3.8.1`` vs ``3.8.3`` would bake two identical interpreters and
        silently disagree with the design, so the name is pinned here.
        """
        assert bake_mod.python_checkpoint_name("3.8") == "vanta-bugsinpy-py3.8"
        assert "3.8.1" not in bake_mod.python_checkpoint_name("3.8")

    def test_real_split_pins_agree_per_minor_version(self):
        split = load_split(SPLIT_PATH)
        assert bake_mod.pythons_in_split(split) == ["3.6", "3.7", "3.8"]
        assert bake_mod.pins_for_python(split, "3.8") == bake_mod.PytestPin("8.0.2", "1.5.0")
        assert bake_mod.pins_for_python(split, "3.7") == bake_mod.PytestPin("7.4.4", "1.5.0")
        assert bake_mod.pins_for_python(split, "3.6") == bake_mod.PytestPin("6.2.5", "1.4.1")

    def test_disagreeing_pins_on_one_interpreter_are_an_error(self):
        data = split_data(
            bugs=[bug_entry(), bug_entry(bug_id=2, pytest_version="7.0.0")]
        )
        with pytest.raises(SplitError, match="disagree"):
            bake_mod.pins_for_python(split_mod.Split(data), "3.8")

    def test_unknown_python_is_an_error(self):
        with pytest.raises(SplitError, match="no bug in the split"):
            bake_mod.pins_for_python(split_mod.Split(split_data()), "3.9")

    def test_real_split_checkpoints_follow_the_committed_convention(self):
        split = load_split(SPLIT_PATH)
        for entry in split.bugs:
            assert bake_mod.bug_checkpoint_name(entry) == entry["checkpoint"]

    def test_bug_layer_name_must_not_drift_from_the_convention(self):
        assert bake_mod.bug_checkpoint_name(bug_entry(checkpoint="vanta-bi-Demo-1")) == (
            "vanta-bi-Demo-1"
        )
        with pytest.raises(SplitError, match="convention"):
            bake_mod.bug_checkpoint_name(bug_entry(checkpoint="whatever"))

    # ---- the interpreter recipe ------------------------------------------
    def _recipe(self) -> str:
        return bake_mod.python_recipe("3.8", bake_mod.PytestPin("8.0.2", "1.5.0"))

    def test_interpreter_recipe_asserts_ssl_then_pip_then_the_plugin(self):
        recipe = self._recipe()
        for needle in (
            "set -eu",
            "uv venv",
            "--python 3.8",
            "import ssl",
            '"pytest==8.0.2"',
            '"pytest-json-report==1.5.0"',
            "python -m pytest --version",
            "import pytest_jsonreport",
        ):
            assert needle in recipe, needle

    def test_ssl_check_is_functional_not_a_version_string(self):
        """An interpreter whose ``_ssl`` failed to build still prints a version."""
        assert "SSLContext" in self._recipe()

    def test_interpreter_recipe_provides_the_name_test_cmd_uses(self):
        recipe = self._recipe()
        assert "for name in python python3" in recipe
        assert "/usr/local/bin/$name" in recipe
        assert "/opt/venv38/bin/python" in recipe

    def test_interpreter_recipe_never_reaches_the_venv_through_a_symlink(self):
        """A symlink makes CPython resolve past the venv and see the standalone.

        Measured in Shuru: through ``/usr/local/bin/python`` → symlink →
        ``/opt/venv38/bin/python`` → symlink → the real binary, ``sys.prefix`` was
        the *standalone* prefix, so pip was the standalone's pip and answered
        ``error: externally-managed-environment`` (PEP 668).
        """
        recipe = self._recipe()
        assert "ln -sfn" not in recipe
        assert 'exec /opt/venv38/bin/python "$@"' in recipe

    def test_wrapper_is_moved_into_place_never_redirected_into_the_path(self):
        """`> /usr/local/bin/python` follows the link and destroys the interpreter.

        Measured: with that path a symlink to the venv, the redirect resolved both
        links and truncated the real 70904-byte CPython binary to 43 bytes of shell
        script that execs the venv — an infinite exec loop (a later
        ``/opt/venv38/bin/python -c …`` never returned). ``mv`` cannot: rename(2)
        does not follow the destination.
        """
        recipe = self._recipe()
        assert "> /tmp/vanta-wrapper" in recipe
        assert "mv -f /tmp/vanta-wrapper" in recipe
        assert "> /usr/local/bin/python" not in recipe

    def test_recipes_assert_the_interpreter_is_a_venv_not_the_standalone(self):
        """The PEP 668 guard: "python runs" is not enough.

        The standalone interpreter executes anything; it just refuses to install.
        Without this assertion the failure surfaces at the pip line — or, worse, at
        run time as a ``ModuleNotFoundError`` that looks like a model failure.
        """
        assert "sys.prefix != sys.base_prefix" in self._recipe()
        assert "sys.prefix != sys.base_prefix" in bake_mod.requirements_recipe([])

    def test_recipes_do_not_reinstall_pip(self):
        """``uv venv --seed`` already provides a current pip.

        Measured: against a bare interpreter, ``pip install --upgrade pip`` is the
        line that dies on PEP 668 — it was the wave-1 bake's first failure.
        """
        for recipe in (self._recipe(), bake_mod.requirements_recipe(["decorator"])):
            assert "--upgrade pip" not in recipe

    def test_marker_is_touched_last(self):
        for recipe in (self._recipe(), bake_mod.requirements_recipe(["decorator"])):
            assert recipe.strip().splitlines()[-1] == f"touch {bake_mod.MARKER}"

    # ---- the per-bug recipe ----------------------------------------------
    def test_declared_deps_reads_setup_sh_and_drops_the_self_install(self):
        setup = "python setup.py install\npip install python_toolbox\n"
        assert bake_mod.declared_deps(setup) == ["python_toolbox"]

    def test_declared_deps_drops_editable_and_path_installs(self):
        setup = "\n".join(
            [
                "pip install -e .",
                "pip install -e./pkg",
                "pip install ./local",
                "pip install decorator",
            ]
        )
        assert bake_mod.declared_deps(setup) == ["decorator"]

    def test_declared_deps_keeps_a_pinned_spec_verbatim(self):
        assert bake_mod.declared_deps('pip install "requests>=2,<3"\n') == [
            '"requests>=2,<3"'
        ]

    def test_declared_deps_accepts_python_m_pip_and_ignores_comments(self):
        setup = "# install deps\n\npython -m pip install decorator\npython3 -m pip install decorator\n"
        assert bake_mod.declared_deps(setup) == ["decorator"]

    def test_requirements_recipe_installs_the_resolved_pins_and_never_swallows_errors(self):
        """The recipe takes pins, not the project's own ranges.

        ``decorator>=4.3.0`` would install a different environment every time the
        layer is rebuilt; the layer has to be a fact, so what goes on the command
        line is what :func:`resolve_pins` decided.
        """
        recipe = bake_mod.requirements_recipe(["requests==2.0", "decorator==4.4.0"])
        assert "requests==2.0" in recipe
        assert "decorator==4.4.0" in recipe
        assert "|| true" not in recipe
        assert "VANTA_REQUIREMENTS" in recipe

    def test_requirements_recipe_without_deps_still_asserts_the_interpreter(self):
        recipe = bake_mod.requirements_recipe([])
        assert "import ssl" in recipe
        assert "python -m pytest --version" in recipe
        assert "cat >" not in recipe, "an empty heredoc would install nothing useful"

    @needs_clone
    def test_real_pysnooper_layers_are_the_interpreter_plus_one_thin_layer(self):
        """PySnooper's deps live in ``setup.sh``; ``requirements.txt`` is empty.

        This is the measured reason ``declared_deps`` exists at all: a layer that
        read only ``requirements.txt`` would install nothing and the wave-1 run
        would die on ``ModuleNotFoundError``.
        """
        root = BUGSINPY_CLONE
        assert bake_mod.requirements_for(root, "PySnooper", 1) == ""
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(root, "PySnooper", 1)) == []
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(root, "PySnooper", 2)) == [
            "python_toolbox"
        ]
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(root, "PySnooper", 3)) == [
            "decorator"
        ]

    def test_wave_1_layers_are_one_interpreter_and_three_bug_layers(self):
        split = load_split(SPLIT_PATH)
        minors, entries = bake_mod.select_layers(
            split, pythons=["3.8"], only=[], roles={"pilot"}
        )
        assert minors == ["3.8"]
        assert [entry["project"] for entry in entries] == ["PySnooper"] * 3
        layers = bake_mod.layer_recipes(
            split,
            pythons=minors,
            entries=entries,
            deps=_bug_deps(entries, {"vanta-bi-PySnooper-2": ("python-toolbox==0.9.3",)}),
        )
        assert [name for name, _, _ in layers] == [
            "vanta-bugsinpy-py3.8",
            "vanta-bi-PySnooper-1",
            "vanta-bi-PySnooper-2",
            "vanta-bi-PySnooper-3",
        ]
        assert layers[0][1] == bake_mod.DEFAULT_BASE
        assert all(base == "vanta-bugsinpy-py3.8" for _, base, _ in layers[1:])
        assert "python-toolbox==0.9.3" in layers[2][2]

    def test_a_bug_without_resolved_dependencies_is_refused_not_baked_dependency_free(self):
        """``layer_recipes`` must not silently build a layer that installs nothing.

        The layer's installs come from :func:`resolve_bug_deps`, and a bug missing
        from that list was never resolved — baking it anyway would produce exactly
        the PySnooper/3 failure this whole path exists to prevent.
        """
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        entry = split.bug("Demo", 1)
        with pytest.raises(bake_mod.BakeError, match="resolve_bug_deps"):
            bake_mod.layer_recipes(split, pythons=["3.8"], entries=[entry], deps=[])

    # ---- selection and the plan ------------------------------------------
    def test_select_layers_defaults_to_pilots_only(self):
        split = split_mod.Split(
            split_data(bugs=[bug_entry(), bug_entry(bug_id=2, role="reserve")])
        )
        minors, entries = bake_mod.select_layers(
            split, pythons=["3.8"], only=[], roles={"pilot"}
        )
        assert minors == ["3.8"]
        assert [entry["bug_id"] for entry in entries] == [1]

    def test_select_layers_rejects_an_unknown_python(self):
        with pytest.raises(SplitError, match="no bug in the split uses"):
            bake_mod.select_layers(
                split_mod.Split(split_data()), pythons=["3.11"], only=[], roles={"pilot"}
            )

    def test_plan_names_every_layer_with_its_base(self):
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        plan = bake_mod.render_plan(
            split, pythons=["3.8"], entries=[split.bug("Demo", 1)]
        )
        assert "| vanta-bugsinpy-py3.8 | vanta-python | 3.8 | 8.0.2 | 1.5.0 |" in plan
        assert "| vanta-bi-Demo-1 | vanta-bugsinpy-py3.8 | 3.8.1 |" in plan

    def test_disk_below_the_base_image_is_refused(self):
        """Measured: ``shuru: --disk-size 4096MB is smaller than the base image``."""
        with pytest.raises(SplitError, match="below the base image"):
            bake_mod.bake("vanta-bugsinpy-py3.8", "true", disk_mb=4096)


# ============================================================================
# --force means replace (§3.2) — the second half of the failed rebake
# ============================================================================
PY38 = "vanta-bugsinpy-py3.8"
"""The interpreter layer the three PySnooper bugs are baked on."""


def _layers(*names: str) -> list[tuple[str, str, str]]:
    """``layer_recipes``' output shape: ``(name, base, script)``, base before child."""
    return [
        (name, bake_mod.DEFAULT_BASE if name == PY38 else PY38, f"recipe for {name}")
        for name in names
    ]


class TestBakeLayerPlan:
    """What ``--force`` does to a layer that is already baked.

    Measured failure: ``bake --force`` passed ``checkpoint create`` a name that
    already existed, shuru refused, and ``checkpoint_create``'s failure path deleted
    the layer it had collided with. The next ``prepare`` then reported "these
    checkpoints do not exist: vanta-bi-PySnooper-1" about a layer the bake had just
    named, and ran PySnooper/3 against a *stale* layer — the old environment, baked
    without the dependency the fix was about. Every test here is about that pair of
    properties: a forced layer is replaced, and a stale child is not left behind
    looking valid.
    """

    CHILDREN = {"vanta-bi-Demo-1": PY38, "vanta-bi-Demo-2": PY38}

    def test_force_deletes_the_layer_before_recreating_it(self):
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=True,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        actions = {action.name: action for action in plan}
        assert actions[PY38].delete and actions[PY38].build
        assert actions["vanta-bi-Demo-1"].delete and actions["vanta-bi-Demo-1"].build
        assert "--force" in actions["vanta-bi-Demo-1"].reason

    def test_without_force_an_existing_layer_is_skipped_and_never_deleted(self):
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=False,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        assert [action.name for action in plan] == [PY38, "vanta-bi-Demo-1"]
        assert all(not action.delete and not action.build for action in plan)
        assert all(action.reason == "already baked" for action in plan)

    def test_a_layer_that_is_not_baked_yet_is_built_without_a_delete(self):
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=False,
            children=self.CHILDREN,
            exists=lambda name: name == PY38,
        )
        actions = {action.name: action for action in plan}
        assert not actions[PY38].delete and not actions[PY38].build
        assert not actions["vanta-bi-Demo-1"].delete and actions["vanta-bi-Demo-1"].build
        assert actions["vanta-bi-Demo-1"].reason == "not baked yet"

    def test_deletions_run_children_before_the_base_they_derive_from(self):
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"),
            force=True,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        order = [action.name for action in bake_mod.delete_order(plan, self.CHILDREN)]
        assert order[-1] == PY38, "the base is deleted last"
        assert set(order) == {PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"}

    def test_builds_run_base_before_child(self):
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=True,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        assert [action.name for action in bake_mod.build_order(plan)] == [
            PY38,
            "vanta-bi-Demo-1",
        ]

    def test_a_rebuilt_base_takes_its_unselected_children_with_it(self):
        """The stale layer that matters most is the one nobody selected.

        Rebuilding ``vanta-bugsinpy-py3.8`` invalidates every layer stacked on it,
        and ``vanta-bi-Demo-2`` is not in this run — so it would keep the old
        interpreter and keep being accepted by ``prepare``.
        """
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=True,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        actions = {action.name: action for action in plan}
        stale = actions["vanta-bi-Demo-2"]
        assert stale.delete and not stale.build
        assert PY38 in stale.reason and "stale layer" in stale.reason
        assert "rebaked before it is used" in stale.reason

    def test_a_base_that_is_not_rebuilt_never_deletes_its_children(self):
        """Skipping the base leaves its children valid, so nothing is deleted."""
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=False,
            children=self.CHILDREN,
            exists=lambda name: True,
        )
        assert [action.name for action in plan] == [PY38, "vanta-bi-Demo-1"]

    def test_a_layer_outside_the_selection_is_only_deleted_as_a_stale_child(self):
        """A child of a base that is *not* part of this run is left alone.

        ``--pythons 3.8 --only PySnooper/*`` must not touch a 3.7 layer.
        """
        children = dict(self.CHILDREN, **{"vanta-bi-Other-1": "vanta-bugsinpy-py3.7"})
        plan = bake_mod.plan_layers(
            _layers(PY38, "vanta-bi-Demo-1"),
            force=True,
            children=children,
            exists=lambda name: True,
        )
        assert "vanta-bi-Other-1" not in {action.name for action in plan}

    def test_every_per_bug_layer_in_the_split_is_a_known_child(self):
        """``children_of`` covers the reserves too, or the stale-child rule misses them."""
        split = split_mod.Split(
            split_data(
                bugs=[
                    bug_entry(checkpoint="vanta-bi-Demo-1"),
                    bug_entry(bug_id=2, checkpoint="vanta-bi-Demo-2", role="reserve"),
                    bug_entry(
                        bug_id=3,
                        checkpoint="vanta-bi-Demo-3",
                        python_version="3.7.3",
                        role="reserve",
                    ),
                ]
            )
        )
        assert bake_mod.children_of(split) == {
            "vanta-bi-Demo-1": PY38,
            "vanta-bi-Demo-2": PY38,
            "vanta-bi-Demo-3": "vanta-bugsinpy-py3.7",
        }


class TestRunBakeActions:
    """The delete/build sequence, driven with the VM calls injected out."""

    CHILDREN = {"vanta-bi-Demo-1": PY38, "vanta-bi-Demo-2": PY38}

    class Recorder:
        """Collects the calls, and owns which layers exist."""

        def __init__(self, present: set[str], failing: set[str] | None = None, sticky: set[str] | None = None):
            self.present = set(present)
            self.failing = set(failing or ())
            self.sticky = set(sticky or ())
            self.calls: list[tuple[str, str]] = []

        def exists(self, name: str) -> bool:
            return name in self.present

        def remove(self, name: str) -> None:
            self.calls.append(("remove", name))
            if name in self.sticky:
                return  # a delete that silently does nothing, as a failed one would
            self.present.discard(name)

        def build(self, name: str, script: str, base: str) -> bake_mod.ExecResult:
            self.calls.append(("build", name))
            if name in self.failing:
                # Measured in shuru: the checkpoint is saved *before* the inner
                # command's failure is reported, so a failed build leaves a name
                # behind. ``checkpoint_create`` deletes it; this asserts that if that
                # delete does not take, the name does not survive as "baked".
                self.present.add(name)
                return bake_mod.ExecResult(
                    stdout="", stderr=f"shuru: refusing to build {name}", exit_code=1, timed_out=False
                )
            self.present.add(name)
            return bake_mod.ExecResult(stdout="ok", stderr="", exit_code=0, timed_out=False)

    def _run(self, layers, recorder, *, force):
        return bake_mod.run_bake_actions(
            bake_mod.plan_layers(
                layers, force=force, children=self.CHILDREN, exists=recorder.exists
            ),
            children=self.CHILDREN,
            exists=recorder.exists,
            remove=recorder.remove,
            build=recorder.build,
            log=lambda line: None,
        )

    def test_all_deletions_happen_before_any_build(self):
        recorder = self.Recorder(present={PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"})
        self._run(_layers(PY38, "vanta-bi-Demo-1"), recorder, force=True)
        verbs = [verb for verb, _ in recorder.calls]
        assert verbs == ["remove", "remove", "remove", "build", "build"], recorder.calls
        assert [name for verb, name in recorder.calls if verb == "remove"] == [
            "vanta-bi-Demo-1",
            "vanta-bi-Demo-2",
            PY38,
        ]

    def test_a_failed_build_stops_the_run_and_leaves_nothing_behind(self):
        recorder = self.Recorder(
            present={PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"},
            failing={"vanta-bi-Demo-1"},
        )
        with pytest.raises(bake_mod.BakeError) as excinfo:
            self._run(_layers(PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"), recorder, force=True)
        message = str(excinfo.value)
        # Fail fast: the layer after the failure is not attempted.
        assert [name for verb, name in recorder.calls if verb == "build"] == [PY38, "vanta-bi-Demo-1"]
        # And the half-baked name is gone, so prepare cannot accept it.
        assert not recorder.exists("vanta-bi-Demo-1")
        assert "FAILED" in message and "vanta-bi-Demo-1" in message
        assert "bake summary:" in message

    def test_a_layer_that_survives_its_own_failure_is_reported_not_hidden(self):
        """If the cleanup delete does not take, say so instead of leaving it live."""
        recorder = self.Recorder(
            present={PY38, "vanta-bi-Demo-1"},
            failing={"vanta-bi-Demo-1"},
            sticky={"vanta-bi-Demo-1"},
        )
        with pytest.raises(bake_mod.BakeError, match="could not be removed"):
            self._run(_layers(PY38, "vanta-bi-Demo-1"), recorder, force=True)
        assert recorder.exists("vanta-bi-Demo-1"), "the test's own premise"

    def test_the_summary_names_what_was_rebuilt_skipped_and_failed(self):
        recorder = self.Recorder(present={PY38, "vanta-bi-Demo-1", "vanta-bi-Demo-2"})
        summary = self._run(_layers(PY38, "vanta-bi-Demo-1"), recorder, force=True)
        assert "[done] vanta-bi-Demo-1: delete+rebuild" in summary
        assert "[skipped] vanta-bi-Demo-2: delete" in summary
        assert "[done] vanta-bugsinpy-py3.8" in summary

    def test_without_force_a_baked_layer_is_not_touched(self):
        recorder = self.Recorder(present={PY38, "vanta-bi-Demo-1"})
        summary = self._run(_layers(PY38, "vanta-bi-Demo-1"), recorder, force=False)
        assert recorder.calls == []
        assert summary.count("[skipped]") == 2


# ============================================================================
# the four dependency sources (§3.2, §3.3) — the fix for the wave-1 failure
# ============================================================================
class TestSetupShParsing:
    """``setup.sh`` is shell: its pip lines are deps, its project install is not.

    Measured on PySnooper: ``setup.sh`` is the *only* source that carries
    ``python_toolbox`` (bug 2) and ``decorator`` (bug 3) — BugsInPy's
    ``requirements.txt`` is 0 bytes for all three bugs. A layer that read only
    ``requirements.txt`` installs nothing, and the failure surfaces as
    ``pytest collected nothing`` — a dependency problem wearing the costume of an
    oracle problem.
    """

    def test_pip_lines_are_kept_and_the_self_install_is_skipped_and_recorded(self):
        refs, skipped = bake_mod.parse_setup_sh(
            "python setup.py install\npip install python_toolbox\n"
        )
        assert [ref.spec for ref in refs] == ["python_toolbox"]
        assert refs[0].source == "setup.sh line 2"
        assert [line.line for line in skipped] == [1]
        assert skipped[0].text == "python setup.py install"
        assert skipped[0].reason == bake_mod._SETUP_PY_REASON

    def test_the_self_install_is_skipped_because_it_shadows_the_graded_tree(self):
        """The reason has to name the failure it prevents, or it reads as noise."""
        _, skipped = bake_mod.parse_setup_sh("python setup.py install\n")
        assert "site-packages" in skipped[0].reason
        assert "every patch would then grade the same" in skipped[0].reason

    def test_a_non_pip_shell_line_is_recorded_rather_than_silently_dropped(self):
        _, skipped = bake_mod.parse_setup_sh("export FOO=1\npip install decorator\n")
        assert [line.line for line in skipped] == [1]
        assert skipped[0].reason == "not a pip install line"

    def test_editable_path_and_requirements_file_lines_are_refused(self):
        refs, skipped = bake_mod.parse_setup_sh(
            "\n".join(
                [
                    "pip install -e .",
                    "pip install -e./pkg",
                    "pip install ./local",
                    "pip install -r requirements.txt",
                    "pip install decorator",
                ]
            )
        )
        assert [ref.spec for ref in refs] == ["decorator"]
        reasons = [line.reason for line in skipped]
        assert reasons[:3] == [bake_mod._PROJECT_REASON] * 3
        assert reasons[3] == bake_mod._REQUIREMENTS_FILE_REASON

    def test_a_vcs_target_is_refused_rather_than_guessed_at(self):
        refs, skipped = bake_mod.parse_setup_sh(
            "pip install git+https://github.com/x/y.git#egg=y\n"
        )
        assert refs == []
        assert skipped[0].reason == bake_mod._VCS_REASON

    def test_a_quoted_pinned_spec_is_parsed_without_its_shell_quotes(self):
        """The record keeps the line as written; the parse sees what pip received."""
        refs, _ = bake_mod.parse_setup_sh('pip install "requests>=2,<3"\n')
        assert refs[0].spec == '"requests>=2,<3"'
        assert bake_mod._specifier_text(refs[0]) == "<3,>=2"

    def test_a_duplicate_spec_is_not_declared_twice(self):
        refs, _ = bake_mod.parse_setup_sh("pip install decorator\npip install decorator\n")
        assert [ref.spec for ref in refs] == ["decorator"]

    def test_comments_and_blank_lines_yield_nothing(self):
        refs, skipped = bake_mod.parse_setup_sh("# deps\n\n   \n")
        assert refs == []
        assert skipped == []


class TestSetupPyStaticReader:
    """``setup.py`` is read with ``ast`` and never executed (design C3)."""

    def test_a_comment_only_file_named_by_setup_py_yields_no_refs(self):
        """Bug 1/2 name ``requirements.in``; bug 3 names ``requirements.txt``."""
        setup = "setup(name='X', install_requires=read_file('requirements.in'))\n"
        refs, _ = setup_py_refs(setup, {"requirements.in": "# no dependencies!\n"}, [])
        assert refs == []

    def test_a_comment_only_install_requires_is_noted_not_silently_empty(self):
        notes: list[str] = []
        setup_py_refs(
            "setup(name='X', install_requires=read_file('requirements.in'))\n",
            {"requirements.in": "# That's right baby! No dependencies!\n"},
            notes,
        )
        assert "install_requires: declares no requirements" in notes

    def test_install_requires_via_a_read_split_chain_keeps_every_entry(self):
        setup = (
            "setup(name='PySnooper',\n"
            "      install_requires=open('requirements.txt','r').read().split('\\n'))\n"
        )
        text = "decorator>=4.3.0\nfuture>=0.17.1\nsix>=1.12.0\n"
        refs, _ = setup_py_refs(setup, {"requirements.txt": text}, [])
        assert [ref.spec for ref in refs] == ["decorator>=4.3.0", "future>=0.17.1", "six>=1.12.0"]
        assert refs[0].source == "install_requires (requirements.txt) line 1"

    def test_a_trailing_blank_entry_is_skipped_not_resolved_as_a_package(self):
        """``split('\\n')`` on a newline-terminated file yields an empty string."""
        setup = (
            "setup(name='X',\n"
            "      tests_require=open('test_requirements.txt','r').read().split('\\n'))\n"
        )
        refs, _ = setup_py_refs(setup, {"test_requirements.txt": "python_toolbox>=0.9.3\n"}, [])
        assert [ref.spec for ref in refs] == ["python_toolbox>=0.9.3"]

    def test_tests_require_and_the_tests_extra_are_both_read(self):
        setup = (
            "setup(name='X',\n"
            "      tests_require=['python_toolbox>=0.9.3'],\n"
            "      extras_require={'tests': {'pytest', 'python-toolbox'}})\n"
        )
        refs, _ = setup_py_refs(setup, {}, [])
        assert {ref.name for ref in refs} == {"python-toolbox", "pytest"}
        assert any("tests_require (setup.py) line" in ref.source for ref in refs)
        assert any("extras_require['tests'] (setup.py) line" in ref.source for ref in refs)

    def test_a_root_test_requirements_file_is_read_even_when_setup_py_never_names_it(self):
        """PySnooper/3's ``test_requirements.txt`` is reachable only this way."""
        setup = "setup(name='PySnooper', install_requires=['decorator>=4.3.0'])\n"
        refs, _ = setup_py_refs(
            setup, {"test_requirements.txt": "python_toolbox>=0.9.3\npytest>=4.4.1\n"}, []
        )
        assert "python_toolbox>=0.9.3" == next(
            ref.spec for ref in refs if ref.name == "python-toolbox"
        )
        assert any(ref.source == "test_requirements.txt line 1" for ref in refs)

    def test_a_file_setup_py_names_but_the_commit_does_not_have_is_noted(self):
        notes: list[str] = []
        refs, _ = setup_py_refs(
            "setup(name='X', install_requires=read_file('requirements.in'))\n", {}, notes
        )
        assert refs == []
        assert any("requirements.in" in note and "not at the buggy commit" in note for note in notes)

    def test_a_declaration_that_is_absent_is_noted_per_source(self):
        notes: list[str] = []
        setup_py_refs("setup(name='X')\n", {}, notes)
        assert "install_requires: not declared in setup.py" in notes
        assert "tests_require: not declared in setup.py" in notes

    def test_setup_py_is_parsed_not_executed(self):
        """A ``setup.py`` that raises on import still yields its declarations.

        This is the C3 guarantee in a form a test can see: executing this file
        would kill the reader, so a reader that survives it read the AST.
        """
        setup = (
            "raise SystemExit('executing this setup.py would abort the host')\n"
            "setup(name='Demo', install_requires=['decorator>=4.3.0'])\n"
        )
        refs, _ = setup_py_refs(setup, {}, [])
        assert [ref.spec for ref in refs] == ["decorator>=4.3.0"]

    def test_import_name_comes_from_name_then_from_the_first_non_test_package(self):
        assert bake_mod.import_name_of("setup(name='PySnooper')\n") == "pysnooper"
        assert bake_mod.import_name_of("setup(packages=['pkg', 'tests'])\n") == "pkg"

    def test_import_name_that_cannot_be_read_is_flagged_not_guessed(self):
        with pytest.raises(bake_mod.BakeError, match="neither name= nor a packages="):
            bake_mod.import_name_of("setup(version='1.0')\n")

    def test_setup_py_that_is_not_python_is_an_error(self):
        with pytest.raises(bake_mod.BakeError, match="does not parse as Python"):
            bake_mod.setup_py_refs("setup(name=\n", {}, [])


def setup_py_refs(setup_py: str, files: dict, notes: list) -> tuple:
    """``bake_mod.setup_py_refs`` with a dict standing in for the buggy commit."""
    return bake_mod.setup_py_refs(setup_py, files.get, notes)


class TestResolvePinsByDate:
    """Pin by release date, with the dates passed in — no network in a test."""

    RELEASES = {
        "decorator": {"4.3.0": "2018-02-03", "4.4.0": "2019-03-16", "4.4.1": "2019-06-01"},
        "future": {"0.17.1": "2018-10-31"},
        "six": {"1.12.0": "2018-12-10", "1.13.0": "2020-05-31"},
        "python-toolbox": {"0.9.3": "2016-06-07"},
    }

    def _resolve(self, refs, *, commit_date="2019-04-22", harness_pins=None):
        return bake_mod.resolve_pins(
            refs,
            releases=self.RELEASES,
            commit_date=commit_date,
            harness_pins=harness_pins or {},
        )

    def test_the_newest_release_on_or_before_the_commit_date_wins(self):
        refs = [bake_mod.dep_ref("decorator>=4.3.0", "install_requires (requirements.txt) line 1")]
        resolved = self._resolve(refs)
        assert resolved[0].pin == "decorator==4.4.0"
        assert resolved[0].harness is False

    def test_a_release_newer_than_the_commit_is_excluded_even_when_it_is_newest(self):
        """4.4.1 exists, and the buggy tree could not have been built with it."""
        refs = [bake_mod.dep_ref("decorator", "setup.sh line 2")]
        assert self._resolve(refs)[0].pin == "decorator==4.4.0"

    def test_a_constraint_with_no_release_before_the_commit_date_is_an_error(self):
        """Installing today's version would provision an environment the bug never ran in."""
        refs = [bake_mod.dep_ref("six>=1.13.0", "install_requires (requirements.txt) line 3")]
        with pytest.raises(bake_mod.BakeError, match="no release on or before 2019-04-22"):
            self._resolve(refs)

    def test_the_error_names_the_newest_release_it_actually_found(self):
        refs = [bake_mod.dep_ref("six>=1.13.0", "install_requires (requirements.txt) line 3")]
        with pytest.raises(bake_mod.BakeError, match="newest six is 1.13.0"):
            self._resolve(refs)

    def test_a_package_with_no_release_dates_is_not_guessed_at(self):
        refs = [bake_mod.dep_ref("mystery", "setup.sh line 2")]
        with pytest.raises(bake_mod.BakeError, match="no release dates for 'mystery'"):
            self._resolve(refs)

    def test_an_unparseable_release_is_ignored_rather_than_crashing_the_pin(self):
        refs = [bake_mod.dep_ref("mystery", "setup.sh line 2")]
        resolved = bake_mod.resolve_pins(
            refs,
            releases={"mystery": {"not-a-version": "2019-01-01", "1.0": "2019-01-02"}},
            commit_date="2019-04-22",
            harness_pins={},
        )
        assert resolved[0].pin == "mystery==1.0"

    # ---- the harness exception -------------------------------------------
    def test_a_harness_provided_package_keeps_the_harness_version(self):
        """pytest is what ``pytest-json-report`` is installed against and what the
        oracle is read from; re-pinning it is a silent downgrade of the report."""
        refs = [bake_mod.dep_ref("pytest>=4.4.1", "test_requirements.txt line 2")]
        resolved = self._resolve(refs, harness_pins={"pytest": "8.0.2"})
        assert resolved[0].pin == "pytest==8.0.2"
        assert resolved[0].harness is True
        assert resolved[0].as_dict()["provisioned_by"] == "interpreter layer"

    def test_a_constraint_that_excludes_the_harness_version_needs_a_decision(self):
        refs = [bake_mod.dep_ref("pytest<5", "test_requirements.txt line 2")]
        with pytest.raises(bake_mod.BakeError, match="will not downgrade pytest"):
            self._resolve(refs, harness_pins={"pytest": "8.0.2"})

    def test_a_harness_package_the_layer_does_not_provide_is_an_error(self):
        refs = [bake_mod.dep_ref("pytest>=4.4.1", "test_requirements.txt line 2")]
        with pytest.raises(bake_mod.BakeError, match="does not provide it"):
            self._resolve(refs)

    def test_resolution_never_reaches_the_network(self, monkeypatch):
        """The dates are an argument, so a test can prove the pin offline."""
        monkeypatch.setattr(
            bake_mod,
            "pypi_releases",
            lambda name: pytest.fail("the pin path performed a network lookup"),
        )
        assert self._resolve([bake_mod.dep_ref("six", "setup.sh line 1")])[0].pin == "six==1.12.0"


class TestMergeRefs:
    """One requirement written twice is one requirement; two constraints disagree."""

    def test_a_bare_name_is_absorbed_by_a_constrained_one(self):
        """PySnooper/3: ``decorator`` in setup.sh, ``decorator>=4.3.0`` in setup.py."""
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("decorator", "setup.sh line 2"),
                bake_mod.dep_ref("decorator>=4.3.0", "install_requires (requirements.txt) line 1"),
            ]
        )
        assert conflicts == []
        assert [ref.spec for ref in merged] == ["decorator>=4.3.0"]
        assert "setup.sh line 2" in merged[0].source
        assert "install_requires" in merged[0].source

    def test_two_spellings_of_one_name_are_one_package(self):
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("python_toolbox>=0.9.3", "tests_require (test_requirements.txt) line 1"),
                bake_mod.dep_ref("python-toolbox", "extras_require['tests'] (setup.py) line 28"),
            ]
        )
        assert conflicts == []
        assert [ref.name for ref in merged] == ["python-toolbox"]

    def test_two_different_constraints_are_a_conflict_and_are_not_picked_between(self):
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("six>=1.12.0", "install_requires (requirements.txt) line 3"),
                bake_mod.dep_ref("six<1.11", "setup.sh line 4"),
            ]
        )
        assert merged == []
        assert len(conflicts) == 1
        assert conflicts[0].name == "six"
        assert "does not pick one of them" in conflicts[0].message()

    def test_resolve_deps_fails_the_bug_loudly_and_names_both_claims(self):
        sources = bake_mod.DepSources(
            project="Demo",
            bug_id=1,
            checkpoint="vanta-bi-Demo-1",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            import_name="demo",
            refs=(
                bake_mod.dep_ref("six>=1.12.0", "install_requires (requirements.txt) line 3"),
                bake_mod.dep_ref("six<1.11", "setup.sh line 4"),
            ),
        )
        with pytest.raises(bake_mod.BakeError) as excinfo:
            bake_mod.resolve_deps(sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={})
        message = str(excinfo.value)
        assert "Demo/1" in message
        assert "'six>=1.12.0' from install_requires (requirements.txt) line 3" in message
        assert "'six<1.11' from setup.sh line 4" in message


class TestDependencyProvenance:
    """What the bake decided is data a later stage reads back, not a re-derivation."""

    def _sources(self) -> bake_mod.DepSources:
        return bake_mod.DepSources(
            project="Demo",
            bug_id=3,
            checkpoint="vanta-bi-Demo-3",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            import_name="demo",
            refs=(
                bake_mod.dep_ref("decorator>=4.3.0", "install_requires (requirements.txt) line 1"),
                bake_mod.dep_ref("pytest>=4.4.1", "test_requirements.txt line 2"),
            ),
            skipped=(
                bake_mod.SkippedLine(
                    "setup.sh", 1, "python setup.py install", bake_mod._SETUP_PY_REASON
                ),
            ),
            notes=("requirements.txt (BugsInPy): empty for this bug",),
        )

    def test_every_dependency_records_its_source_its_pin_and_its_provisioner(self):
        sources = self._sources()
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={"pytest": "8.0.2"}
        )
        provenance = bake_mod.deps_provenance(sources, resolved, harness_pins={"pytest": "8.0.2"})
        by_name = {dep["name"]: dep for dep in provenance["dependencies"]}
        assert by_name["decorator"]["pin"] == "decorator==4.4.0"
        assert by_name["decorator"]["sources"] == ["install_requires (requirements.txt) line 1"]
        assert by_name["pytest"]["pin"] == "pytest==8.0.2"
        assert by_name["pytest"]["provisioned_by"] == "interpreter layer"
        assert provenance["import_name"] == "demo"
        assert provenance["harness_pins"] == {"pytest": "8.0.2"}

    def test_the_skipped_setup_sh_lines_are_part_of_the_record(self):
        sources = self._sources()
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={"pytest": "8.0.2"}
        )
        provenance = bake_mod.deps_provenance(sources, resolved, harness_pins={"pytest": "8.0.2"})
        # ``installs`` is what the layer's recipe passes to pip, so a package the
        # interpreter layer already provides is not in it (it is still recorded, in
        # ``dependencies`` with its ``provisioned_by``).
        assert provenance["installs"] == ["decorator==4.4.0"]
        assert [dep["name"] for dep in provenance["dependencies"]] == ["decorator", "pytest"]
        assert provenance["skipped"] == [
            {
                "source": "setup.sh",
                "line": 1,
                "text": "python setup.py install",
                "reason": bake_mod._SETUP_PY_REASON,
            }
        ]
        assert provenance["notes"] == ["requirements.txt (BugsInPy): empty for this bug"]

    def test_write_load_and_preflight_round_trip(self, tmp_path):
        deps_dir = tmp_path / "deps"
        item = bake_mod.BugDeps(
            entry={"project": "Demo", "bug_id": 3},
            checkpoint="vanta-bi-Demo-3",
            installs=("decorator==4.4.0",),
            provenance={"installs": ["decorator==4.4.0"], "import_name": "demo"},
        )
        path = bake_mod.write_deps(deps_dir, item)
        assert path.name == "vanta-bi-Demo-3.json"
        assert bake_mod.load_deps(deps_dir, "vanta-bi-Demo-3")["import_name"] == "demo"
        bake_mod.require_deps(deps_dir, ["vanta-bi-Demo-3"])

    def test_a_layer_with_no_recorded_provenance_must_be_rebaked(self, tmp_path):
        deps_dir = tmp_path / "deps"
        with pytest.raises(bake_mod.BakeError, match="re-run bake"):
            bake_mod.load_deps(deps_dir, "vanta-bi-Demo-3")
        with pytest.raises(bake_mod.BakeError, match="vanta-bi-Demo-3"):
            bake_mod.require_deps(deps_dir, ["vanta-bi-Demo-3"])

    def test_a_file_that_is_not_provenance_is_an_error_not_an_empty_provenance(self, tmp_path):
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()
        write(deps_dir / "vanta-bi-Demo-3.json", "{oops\n")
        with pytest.raises(bake_mod.BakeError, match="not valid JSON"):
            bake_mod.load_deps(deps_dir, "vanta-bi-Demo-3")
        write(deps_dir / "vanta-bi-Demo-3.json", json.dumps({"import_name": "demo"}))
        with pytest.raises(bake_mod.BakeError, match="not dependency provenance"):
            bake_mod.load_deps(deps_dir, "vanta-bi-Demo-3")


needs_bare_repos = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "PySnooper.git").is_dir(),
    reason="the bare project clones are not present (runs/ is not committed)",
)


@needs_bare_repos
@needs_clone
class TestRealPySnooperDependencies:
    """The measured bug, end to end: PySnooper 1-3 resolve to the right layers.

    PySnooper/3 is the failure this whole path was written for. Its layer used to
    be ``decorator`` alone, while ``tests/test_pysnooper.py`` imports
    ``python_toolbox``; prepare then reported a 0-test collection as a bug. The
    union of the four sources has to produce ``decorator``, ``future``, ``six`` and
    ``python-toolbox``, and must not install the project itself.
    """

    RELEASE_DATES = TestResolvePinsByDate.RELEASES

    def _entries(self) -> tuple[split_mod.Split, list[dict]]:
        split = load_split(SPLIT_PATH)
        entries = [split.bug("PySnooper", bug_id) for bug_id in (1, 2, 3)]
        return split, entries

    def _resolve(self, entries):
        return bake_mod.resolve_bug_deps(
            load_split(SPLIT_PATH),
            entries,
            bugsinpy_root=BUGSINPY_CLONE,
            repo_for=lambda project: REPO_ROOT / "runs" / "bugsinpy" / "repos" / f"{project}.git",
            fetch=lambda name: self.RELEASE_DATES[name],
        )

    def test_bug_three_installs_every_declared_dependency_and_not_the_project(self):
        _, entries = self._entries()
        item = next(d for d in self._resolve(entries) if d.checkpoint.endswith("-3"))
        assert set(item.installs) == {
            "decorator==4.4.0",
            "future==0.17.1",
            "python-toolbox==0.9.3",
            "six==1.12.0",
        }

    def test_the_project_is_never_among_the_installs(self):
        """A copy of ``pysnooper`` in site-packages would make every patch grade the same."""
        _, entries = self._entries()
        for item in self._resolve(entries):
            assert not any(spec.startswith("pysnooper") for spec in item.installs)

    def test_the_harness_pytest_is_recorded_but_never_installed_by_the_layer(self):
        """Both halves: out of the installs, still in the record.

        The layer must not re-install a package the interpreter layer provides (a
        downgrade underneath pytest-json-report would make the oracle's report
        unreadable), and the provenance must not claim the layer installed it
        either — a record that disagrees with the recipe is worse than no record.
        """
        _, entries = self._entries()
        item = next(d for d in self._resolve(entries) if d.checkpoint.endswith("-3"))
        assert not any(spec.startswith("pytest==") for spec in item.installs)
        pytest_dep = next(
            dep for dep in item.provenance["dependencies"] if dep["name"] == "pytest"
        )
        assert pytest_dep["pin"] == "pytest==8.0.2"
        assert pytest_dep["provisioned_by"] == "interpreter layer"
        assert "pytest==8.0.2" not in item.provenance["installs"]
        assert item.provenance["installs"] == list(item.installs)
        assert item.provenance["harness_pins"]["pytest"] == "8.0.2"

    def test_bug_one_and_two_get_python_toolbox_from_the_tests_extra(self):
        _, entries = self._entries()
        got = {
            item.checkpoint: set(item.installs)
            for item in self._resolve(entries)
            if not item.checkpoint.endswith("-3")
        }
        assert got == {
            "vanta-bi-PySnooper-1": {"python-toolbox==0.9.3"},
            "vanta-bi-PySnooper-2": {"python-toolbox==0.9.3"},
        }

    def test_the_setup_py_install_line_is_recorded_as_skipped_on_every_bug(self):
        _, entries = self._entries()
        for item in self._resolve(entries):
            skipped = item.provenance["skipped"]
            assert any(
                line["source"] == "setup.sh"
                and line["text"] == "python setup.py install"
                and "site-packages" in line["reason"]
                for line in skipped
            ), item.checkpoint

    def test_the_import_name_is_read_from_the_projects_own_setup_py(self):
        _, entries = self._entries()
        for item in self._resolve(entries):
            assert item.provenance["import_name"] == "pysnooper"


class TestDepGate:
    """The dependency gate runs before the derivation, with its own messages (§3.3).

    Measured: a missing ``python_toolbox`` made ``--collect-only`` collect nothing,
    and prepare reported ``hidden-oracle derivation failed``. A dependency problem
    and an oracle problem need different fixes, so they must not share a message.
    """

    @staticmethod
    def _prep(tmp_path: Path) -> prepare_mod.BugPrep:
        batch = tmp_path / "batch"
        return prep(root=batch / "worktrees" / "Demo_1")

    def _deps(self) -> dict:
        return {"import_name": "pysnooper"}

    def test_a_healthy_layer_returns_the_collected_nodeids(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            suite_mod,
            "collect_only",
            lambda **kwargs: seen.update(collect=kwargs) or {"tests/test_x.py::test_a"},
        )
        monkeypatch.setattr(
            suite_mod,
            "import_probe",
            lambda **kwargs: seen.update(probe=kwargs)
            or "/workspace/worktrees/Demo_1/oracle_tree/pysnooper/__init__.py",
        )
        bug_prep = self._prep(tmp_path)

        nodeids = prepare_mod.check_deps(
            bug_prep, checkpoint="chk", batch_dir=tmp_path / "batch", deps=self._deps()
        )

        assert nodeids == {"tests/test_x.py::test_a"}
        for call in (seen["collect"], seen["probe"]):
            assert call["checkpoint"] == "chk"
            assert call["tree_rel"] == "worktrees/Demo_1/oracle_tree"
            assert call["test_cmd"] == bug_prep.test_cmd, (
                "the gate must run exactly the invocation the suites run"
            )
        assert seen["probe"]["module"] == "pysnooper"

    def test_a_collection_failure_is_reported_as_a_dependency_failure(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            raise suite_mod.SuiteError("pytest collected nothing (exit 2)")

        monkeypatch.setattr(suite_mod, "collect_only", boom)
        monkeypatch.setattr(
            suite_mod, "import_probe", lambda **kwargs: pytest.fail("probe ran anyway")
        )

        with pytest.raises(suite_mod.SuiteError, match="^deps: collection failed:"):
            prepare_mod.check_deps(
                self._prep(tmp_path), checkpoint="chk", batch_dir=tmp_path / "batch",
                deps=self._deps(),
            )

    def test_an_import_from_site_packages_is_named_as_such(self, tmp_path, monkeypatch):
        """The failure a ``setup.py install`` in the bake would cause."""
        monkeypatch.setattr(suite_mod, "collect_only", lambda **kwargs: {"tests/x.py::t"})
        monkeypatch.setattr(
            suite_mod,
            "import_probe",
            lambda **kwargs: (
                "/opt/venv38/lib/python3.8/site-packages/pysnooper/__init__.py"
            ),
        )

        with pytest.raises(
            suite_mod.SuiteError, match="^deps: package imports from site-packages, not the tree:"
        ):
            prepare_mod.check_deps(
                self._prep(tmp_path), checkpoint="chk", batch_dir=tmp_path / "batch",
                deps=self._deps(),
            )

    def test_a_module_that_does_not_import_at_all_is_reported_distinctly(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(suite_mod, "collect_only", lambda **kwargs: {"tests/x.py::t"})

        def boom(**kwargs):
            raise suite_mod.SuiteError("exited 1 without printing a path")

        monkeypatch.setattr(suite_mod, "import_probe", boom)
        with pytest.raises(suite_mod.SuiteError, match="^deps: pysnooper does not import"):
            prepare_mod.check_deps(
                self._prep(tmp_path), checkpoint="chk", batch_dir=tmp_path / "batch",
                deps=self._deps(),
            )

    def test_the_derivation_reuses_the_gate_s_collection(self, tmp_path, monkeypatch):
        """Two collections of one tree are two chances to disagree."""
        calls: list = []
        monkeypatch.setattr(
            suite_mod, "collect_only", lambda **kwargs: calls.append(kwargs) or set()
        )

        class ReachedTheSuites(RuntimeError):
            pass

        def stop(*args, **kwargs):
            raise ReachedTheSuites()

        monkeypatch.setattr(prepare_mod, "_suite", stop)
        with pytest.raises(ReachedTheSuites):
            prepare_mod.derive_hidden(
                self._prep(tmp_path), checkpoint="chk", batch_dir=tmp_path / "batch",
                collected={"tests/x.py::t"},
            )
        assert calls == [], "`collected` was handed in; the gate already paid for it"


class TestImportProbe:
    """``import_probe`` reuses the suite invocation, not a convenient one."""

    @staticmethod
    def _result(stdout: str, *, exit_code: int = 0):
        return shuru_mod.ExecResult(
            stdout=stdout, stderr="", exit_code=exit_code, timed_out=False
        )

    def test_the_probe_runs_the_suite_interpreter_in_the_suite_cwd(self, monkeypatch):
        seen: dict = {}
        monkeypatch.setattr(
            suite_mod.shuru,
            "run",
            lambda **kwargs: seen.update(kwargs)
            or self._result("/workspace/w/Demo_1/oracle_tree/pysnooper/__init__.py\n"),
        )

        where = suite_mod.import_probe(
            checkpoint="chk",
            batch_dir=Path("/tmp/batch"),
            tree_rel="w/Demo_1/oracle_tree",
            test_cmd=["python", "-m", "pytest", "tests"],
            module="pysnooper",
            disk_mb=8192,
        )

        assert where == "/workspace/w/Demo_1/oracle_tree/pysnooper/__init__.py"
        assert seen["guest_cwd"] == "/workspace/w/Demo_1/oracle_tree"
        assert seen["command"][:2] == ["sh", "-c"], "the probe runs as one shell command"
        probe = seen["command"][2]
        assert "python -c" in probe, "the suite's own interpreter is used"
        assert "import pysnooper" in probe
        assert "PYTHONPATH" not in seen["env"], (
            "a PYTHONPATH would prove an import path the suites do not have"
        )
        assert "PYTHONPATH" not in probe

    def test_a_module_that_does_not_import_raises_rather_than_returning_a_path(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            suite_mod.shuru,
            "run",
            lambda **kwargs: shuru_mod.ExecResult(
                stdout="", stderr="ModuleNotFoundError: No module named 'pysnooper'",
                exit_code=1, timed_out=False,
            ),
        )
        with pytest.raises(suite_mod.SuiteError, match="ModuleNotFoundError"):
            suite_mod.import_probe(
                checkpoint="chk", batch_dir=Path("/tmp/batch"),
                tree_rel="w/Demo_1/oracle_tree",
                test_cmd=["python", "-m", "pytest", "tests"], module="pysnooper",
                disk_mb=8192,
            )

    def test_a_module_name_that_is_not_an_identifier_is_refused_before_any_vm(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            suite_mod.shuru, "run", lambda **kwargs: pytest.fail("a VM was booted")
        )
        with pytest.raises(suite_mod.SuiteError, match="not a Python identifier"):
            suite_mod.import_probe(
                checkpoint="chk", batch_dir=Path("/tmp/batch"),
                tree_rel="w/Demo_1/oracle_tree",
                test_cmd=["python", "-m", "pytest", "tests"], module="pysnooper; rm -rf /",
                disk_mb=8192,
            )

    def test_the_tree_root_the_suites_use_is_one_definition(self):
        assert suite_mod.guest_tree_root("w/Demo_1/oracle_tree") == (
            "/workspace/w/Demo_1/oracle_tree"
        )


class TestCollectedOrRaise:
    """A non-empty collection can still hide a collection error."""

    def test_a_clean_collection_passes(self):
        nodeids = {"tests/x.py::t"}
        assert suite_mod.collected_or_raise(nodeids, exit_code=0, output="") == nodeids

    def test_nothing_collected_is_an_error(self):
        with pytest.raises(suite_mod.SuiteError, match="collected nothing"):
            suite_mod.collected_or_raise(set(), exit_code=2, output="ImportError")

    def test_a_non_zero_exit_is_an_error_even_when_something_was_collected(self):
        """Measured: one bad test module leaves the others collected and pytest exits 2."""
        with pytest.raises(suite_mod.SuiteError, match="exited 2 while collecting"):
            suite_mod.collected_or_raise({"tests/x.py::t"}, exit_code=2, output="ImportError tail")

    def test_the_message_carries_the_error_not_the_tail_of_the_run(self):
        """PySnooper/2, verbatim in shape: the ImportError is in the middle, and the
        last 800 characters are warnings plus the sandbox's own boot chatter. The
        tail-only message sent the reader back into the VM to find the error."""
        output = (
            "shuru: booting VM…\n"
            "collected 0 items / 1 error\n\n"
            "==================================== ERRORS ====================================\n"
            "___________________ ERROR collecting tests/test_pysnooper.py ___________________\n"
            "ImportError while importing test module 'tests/test_pysnooper.py'.\n"
            "Hint: make sure your test modules/packages have valid Python names.\n"
            "Traceback:\n"
            "  /usr/lib/python3.8/importlib/__init__.py:127: in import_module\n"
            "      return _bootstrap._gcd_import(name[level:], package, level)\n"
            "      from . import mini_toolbox\n"
            "E   ImportError: cannot import name 'mini_toolbox' from 'tests'\n"
            "ERROR tests/test_pysnooper.py\n"
            "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
            "no tests collected, 1 error in 0.13s\n" + "warnings summary\n" * 200
        )
        with pytest.raises(suite_mod.SuiteError) as caught:
            suite_mod.collected_or_raise(set(), exit_code=2, output=output)
        message = str(caught.value)
        assert "cannot import name 'mini_toolbox'" in message
        assert "ERROR tests/test_pysnooper.py" in message
        assert "warnings summary" not in message, "the tail is what hid the error"
        assert "error lines:" in message

    def test_output_with_no_error_lines_still_reports_something(self):
        """The fallback, so a failure with an unfamiliar shape is not a blank message."""
        with pytest.raises(suite_mod.SuiteError, match="tail:"):
            suite_mod.collected_or_raise(set(), exit_code=2, output="nothing here\n" * 50)

    def test_the_excerpt_is_bounded(self):
        output = "\n".join(f"E   ValueError: {index}" for index in range(60))
        excerpt = suite_mod.failure_excerpt(output)
        assert excerpt.count("\n") == 26, "25 lines plus the '… N more' line"
        assert "more" in excerpt


# ============================================================================
# a tiny independent check of the strip, so a broken strip cannot pass silently
# ============================================================================
def test_strip_and_counter_agree_on_a_synthetic_file(tmp_path):
    """Cross-check: what the strip says it removed is really gone from the file.

    Written against the file contents rather than the report, because the report
    is what a buggy implementation would also get wrong.
    """
    rel = "tests/test_x.py"
    write(
        tmp_path / rel,
        "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n",
    )
    tree_mod.strip_hidden_tests(tmp_path, [f"{rel}::test_a"], {rel: {f"{rel}::test_a", f"{rel}::test_b"}})
    new = (tmp_path / rel).read_text()
    assert "test_a" not in new
    assert "assert 2" in new
    assert Counter(line for line in new.splitlines() if line.strip())["def test_b():"] == 1


# ============================================================================
# oracle.py — what makes a patch "good" (no VM)
# ============================================================================
def oracle_prep(**over) -> prepare_mod.BugPrep:
    prep = prepare_mod.BugPrep(
        bug=bug_info(),
        root=Path("/nonexistent/worktrees/Demo_1"),
        role="pilot",
        wave=1,
        test_cmd=[
            "python",
            "-m",
            "pytest",
            "tests",
            "-p",
            "no:cacheprovider",
            "--json-report",
            "--json-report-file={report}",
        ],
        test_paths=["tests"],
        fail_to_pass=["tests/test_demo.py::test_fix"],
        native_failure_baseline=["tests/test_demo.py::test_known_broken"],
    )
    for key, value in over.items():
        setattr(prep, key, value)
    return prep


def suite_pair(first: dict[str, str], second: dict[str, str] | None = None):
    a = suite_mod.SuiteRun(outcomes=dict(first), exit_code=0, report_found=True)
    b = suite_mod.SuiteRun(outcomes=dict(second or first), exit_code=0, report_found=True)
    return suite_mod.compare_runs(a, b)


class TestOracleHiddenCommand:
    def test_the_directory_argument_is_replaced_by_the_nodeids(self):
        cmd = oracle_mod.hidden_command(
            ["python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--json-report"],
            ["tests"],
            ["tests/test_a.py::test_b"],
        )
        assert cmd == [
            "python",
            "-m",
            "pytest",
            "tests/test_a.py::test_b",
            "-p",
            "no:cacheprovider",
            "--json-report",
        ]

    def test_the_nodeids_land_after_the_runner_not_after_the_first_flag(self):
        """``-m`` is a flag too: inserting before it gives ``python <nodeid> -m
        pytest``, which pytest never sees because ``python`` fails first."""
        cmd = oracle_mod.hidden_command(
            ["python", "-m", "pytest", "tests", "-p", "no:cacheprovider"],
            ["tests"],
            ["tests/test_a.py::test_b"],
        )
        assert cmd[3] == "tests/test_a.py::test_b"

    def test_a_nested_test_path_is_replaced_too(self):
        """tqdm's native path is ``tqdm/tests``, not ``tests``."""
        cmd = oracle_mod.hidden_command(
            ["python", "-m", "pytest", "tqdm/tests", "-p", "no:cacheprovider"],
            ["tqdm/tests"],
            ["tqdm/tests/tests_contrib.py::test_x"],
        )
        assert "tqdm/tests" not in cmd
        assert cmd[3] == "tqdm/tests/tests_contrib.py::test_x"

    def test_no_nodeids_leaves_the_flags(self):
        cmd = oracle_mod.hidden_command(["python", "-m", "pytest", "tests"], ["tests"], [])
        assert cmd == ["python", "-m", "pytest"]


class TestOracleScore:
    """``patch_good = hidden_pass ∧ suite_regressions == 0`` (§5.3).

    Each test here pins a way the rule could produce a *plausible wrong label*
    instead of an error, which is the failure mode this whole file guards.
    """

    def _score(self, hidden, patched, baseline=("tests/test_demo.py::test_known_broken",)):
        return oracle_mod.score(
            hidden_pair=hidden,
            patched_pair=patched,
            hidden_nodeids=["tests/test_demo.py::test_fix"],
            baseline_failures=list(baseline),
        )

    def test_a_passing_hidden_test_and_no_regression_is_good(self):
        hidden_pass, _, regressions, _ = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "passed"}),
            suite_pair({"tests/test_demo.py::test_known_broken": "failed"}),
        )
        assert hidden_pass is True
        assert regressions == []

    def test_a_skipped_hidden_test_is_not_a_pass(self):
        """A patch that makes the hidden test disappear must not read as a fix."""
        hidden_pass, outcomes, _, _ = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "skipped"}),
            suite_pair({}),
            baseline=(),
        )
        assert hidden_pass is False
        assert outcomes["tests/test_demo.py::test_fix"] == "skipped"

    def test_a_hidden_test_absent_from_the_report_is_not_a_pass(self):
        hidden_pass, outcomes, _, _ = self._score(
            suite_pair({"tests/test_other.py::test_other": "passed"}),
            suite_pair({}),
            baseline=(),
        )
        assert hidden_pass is False
        assert outcomes["tests/test_demo.py::test_fix"] == "absent"

    def test_a_flaky_hidden_test_is_not_a_pass(self):
        """Run twice (§1.3 step 3): passed then failed has no stable outcome."""
        hidden_pass, _, _, _ = self._score(
            suite_pair(
                {"tests/test_demo.py::test_fix": "passed"},
                {"tests/test_demo.py::test_fix": "failed"},
            ),
            suite_pair({}),
            baseline=(),
        )
        assert hidden_pass is False

    def test_a_preexisting_failure_is_not_charged_to_the_patch(self):
        _, _, regressions, patched = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "passed"}),
            suite_pair({"tests/test_demo.py::test_known_broken": "failed"}),
        )
        assert patched == ["tests/test_demo.py::test_known_broken"]
        assert regressions == []

    def test_a_new_failure_is_a_regression(self):
        hidden_pass, _, regressions, _ = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "passed"}),
            suite_pair(
                {
                    "tests/test_demo.py::test_known_broken": "failed",
                    "tests/test_demo.py::test_other": "failed",
                }
            ),
        )
        assert hidden_pass is True
        assert regressions == ["tests/test_demo.py::test_other"]

    def test_a_flaky_new_failure_is_not_a_regression(self):
        """Otherwise a flaky native test makes every patch look destructive."""
        _, _, regressions, _ = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "passed"}),
            suite_pair(
                {"tests/test_demo.py::test_other": "failed"},
                {"tests/test_demo.py::test_other": "passed"},
            ),
        )
        assert regressions == []

    def test_hidden_pass_alone_is_not_a_good_patch(self):
        """The review's correction: passing the hidden test while breaking the
        suite is *not* a good patch, so this must not be labelled `false_reject`."""
        hidden_pass, _, regressions, _ = self._score(
            suite_pair({"tests/test_demo.py::test_fix": "passed"}),
            suite_pair({"tests/test_demo.py::test_other": "failed"}),
        )
        assert hidden_pass is True
        assert regressions == ["tests/test_demo.py::test_other"]


class TestOracleGrade:
    """The orchestration around ``score``: trees, strictness, and censoring."""

    @pytest.fixture
    def wired(self, monkeypatch, tmp_path):
        """``grade`` with the VM and the tree copy replaced by recorders."""
        calls: dict = {"runs": [], "copies": []}

        def fake_copy(src, dest):
            calls["copies"].append((src, dest))
            dest.mkdir(parents=True, exist_ok=True)

        def fake_apply(**kwargs):
            calls["apply"] = kwargs
            return True, "applied"

        def fake_runs(**kwargs):
            calls["runs"].append(kwargs)
            stem = kwargs["report_rel_stem"]
            if stem.endswith("-hidden"):
                return calls["hidden_pair"]
            return calls["patched_pair"]

        monkeypatch.setattr(tree_mod, "copy_tree", fake_copy)
        monkeypatch.setattr(suite_mod, "git_apply_patch", fake_apply)
        monkeypatch.setattr(suite_mod, "run_suite_twice", fake_runs)
        calls["hidden_pair"] = suite_pair({"tests/test_demo.py::test_fix": "passed"})
        calls["patched_pair"] = suite_pair({})
        return calls

    def _grade(self, wired, tmp_path, **over):
        patch = write(tmp_path / "final.patch", "diff --git a/x b/x\n")
        return oracle_mod.grade(
            prep=oracle_prep(**over.pop("prep_over", {})),
            batch_dir=tmp_path / "batch",
            checkpoint="vanta-bi-Demo-1",
            patch_path=over.pop("patch_path", patch),
            label=over.pop("label", "sess1"),
            disk_mb=16384,
        )

    def test_a_good_patch_is_good(self, wired, tmp_path):
        result = self._grade(wired, tmp_path)
        assert result.apply_ok and result.hidden_pass and result.patch_good
        assert result.suite_regressions == []
        assert result.label_input == "good"

    def test_test_hunks_are_always_dropped(self, wired, tmp_path):
        """A Creator patch must not be able to reach the hidden test (§5.2 step 2)."""
        self._grade(wired, tmp_path)
        assert wired["apply"]["drop_test_hunks"] is True

    def test_the_patch_is_applied_to_a_copy_of_oracle_tree(self, wired, tmp_path):
        self._grade(wired, tmp_path)
        src, dest = wired["copies"][0]
        assert src == Path("/nonexistent/worktrees/Demo_1/oracle_tree")
        assert dest == tmp_path / "batch" / "oracle_work" / "Demo_1__sess1"

    def test_both_suites_run_in_the_work_tree_never_in_the_shared_trees(
        self, wired, tmp_path
    ):
        self._grade(wired, tmp_path)
        tree_rels = {run["tree_rel"] for run in wired["runs"]}
        assert tree_rels == {"oracle_work/Demo_1__sess1"}
        assert len(wired["runs"]) == 2

    def test_the_hidden_run_asks_for_the_hidden_nodeids(self, wired, tmp_path):
        self._grade(wired, tmp_path)
        hidden = next(r for r in wired["runs"] if r["report_rel_stem"].endswith("-hidden"))
        assert "tests/test_demo.py::test_fix" in hidden["test_cmd"]
        assert "tests" not in hidden["test_cmd"]

    def test_a_failing_hidden_test_makes_a_bad_patch(self, wired, tmp_path):
        wired["hidden_pair"] = suite_pair({"tests/test_demo.py::test_fix": "failed"})
        result = self._grade(wired, tmp_path)
        assert result.apply_ok is True
        assert result.patch_good is False
        assert result.label_input == "bad"

    def test_a_regression_makes_a_bad_patch_even_when_the_hidden_test_passes(
        self, wired, tmp_path
    ):
        wired["patched_pair"] = suite_pair({"tests/test_demo.py::test_other": "failed"})
        result = self._grade(wired, tmp_path)
        assert result.hidden_pass is True
        assert result.suite_regressions == ["tests/test_demo.py::test_other"]
        assert result.patch_good is False

    def test_an_unappliable_patch_is_censored_not_failed(self, monkeypatch, tmp_path):
        """§5.2 step 2: `unobservable`, excluded from the rates. Booting two suites
        to learn nothing would also be a waste of the only LLM slot's time budget."""
        def fake_apply(**kwargs):
            return False, "error: patch does not apply"

        monkeypatch.setattr(suite_mod, "git_apply_patch", fake_apply)
        monkeypatch.setattr(tree_mod, "copy_tree", lambda src, dest: dest.mkdir(parents=True, exist_ok=True))

        def must_not_run(**kwargs):
            raise AssertionError("no suite may run for a patch that would not apply")

        monkeypatch.setattr(suite_mod, "run_suite_twice", must_not_run)

        result = self._grade(wired=None, tmp_path=tmp_path)
        assert result.apply_ok is False
        assert result.patch_good is False
        assert result.label_input == "unobservable"
        assert "does not apply" in result.apply_detail

    def test_a_missing_patch_is_reported_not_crashed(self, wired, tmp_path):
        result = self._grade(wired, tmp_path, patch_path=tmp_path / "absent.patch")
        assert result.apply_ok is False
        assert "patch not found" in result.error

    def test_a_bug_with_no_hidden_nodeids_cannot_be_graded(self, wired, tmp_path):
        result = self._grade(
            wired, tmp_path, prep_over={"fail_to_pass": []}
        )
        assert result.apply_ok is False
        assert "no FAIL_TO_PASS" in result.error
        assert wired["runs"] == []

    def test_a_hidden_run_without_a_report_is_flagged(self, wired, tmp_path):
        """Not a crash: the patch is bad, and the reason is recorded for triage."""
        wired["hidden_pair"] = suite_mod.compare_runs(
            suite_mod.SuiteRun(exit_code=4, report_found=False),
            suite_mod.SuiteRun(exit_code=4, report_found=False),
        )
        result = self._grade(wired, tmp_path)
        assert result.hidden_pass is False
        assert "no pytest report" in result.error

    def test_oracle_json_has_the_documented_keys(self, wired, tmp_path):
        result = self._grade(wired, tmp_path)
        path = oracle_mod.save(tmp_path / "sessions" / "sess1" / "oracle.json", result)
        data = json.loads(path.read_text())
        assert set(data) >= {
            "apply_ok",
            "hidden_pass",
            "hidden_nodeids",
            "suite_regressions",
            "suite_regressions_count",
            "patch_good",
            "oracle_secs",
        }
        assert data["patch_good"] is True

    def test_the_label_is_made_filesystem_safe(self, wired, tmp_path):
        """Session ids and patch hashes are used as path components."""
        self._grade(wired, tmp_path, label="a/b c")
        assert wired["copies"][0][1].name == "Demo_1__a_b_c"


# ============================================================================
# §11 step 6 — manifest, resume, the candidate set and its gate
# ============================================================================
# The runner decides what a row *is* and what "already done" means. Both fail
# silently: a key that is not content-based re-runs a graded candidate after a
# resume (and the duplicate row looks like a fresh observation), and a cursor
# that trusts `running` skips a job whose artifacts are half-written.
def _diff(rel: str, added: str, *, index: str = "1111111..2222222") -> str:
    """A minimal non-test unified diff, in the shape `write_final_patch` emits."""
    return (
        f"diff --git a/{rel} b/{rel}\n"
        f"index {index} 100644\n"
        f"--- a/{rel}\n"
        f"+++ b/{rel}\n"
        "@@ -1,3 +1,4 @@\n"
        " one\n"
        " two\n"
        " three\n"
        f"+{added}\n"
    )


DIFF_LOOSE = _diff("pkg/mod.py", "four")
DIFF_TIDY = (
    # The same tree change, written the other way round: no `index` line, no
    # `a/`/`b/` prefixes, trailing whitespace on one line.
    "diff --git pkg/mod.py pkg/mod.py\n"
    "--- pkg/mod.py \n"
    "+++ pkg/mod.py\n"
    "@@ -1,3 +1,4 @@\n"
    " one\n"
    " two\n"
    " three\n"
    "+four\n"
)


def _pool_row(path: Path, text: str, **over) -> dict:
    """A manifest row whose patch exists on disk, as `pool_failed_patches` reads it."""
    write(path, text)
    row = {
        "verdict": "Rejected",
        "config": "vanta",
        "session_id": "sess-1",
        "oracle": {"apply_ok": True, "patch_good": False},
        "artifacts": {"final_patch": str(path)},
    }
    row.update(over)
    return row


class TestJobKey:
    """The key is the resume cursor, so its identity must be content, not a counter."""

    def test_a_repair_key_has_an_empty_candidate_component(self):
        key = manifest_mod.job_key(
            batch_id="b1", bug="Demo/1", config="vanta", mode="repair", seed=0
        )
        assert key == "b1#Demo/1#vanta#repair#0#"

    def test_a_review_key_embeds_the_candidate_patch_hash(self):
        key = manifest_mod.job_key(
            batch_id="b1",
            bug="Demo/1",
            config="baseline",
            mode="review",
            seed=0,
            candidate="a" * 64,
        )
        assert key.endswith("#" + "a" * 64)

    def test_two_candidates_of_one_bug_do_not_collide(self):
        """A review pass is one session *per candidate* (§7.1) — the second must not shadow the first."""
        first = manifest_mod.job_key(
            batch_id="b1", bug="Demo/1", config="vanta", mode="review", seed=0,
            candidate="a" * 64,
        )
        second = manifest_mod.job_key(
            batch_id="b1", bug="Demo/1", config="vanta", mode="review", seed=0,
            candidate="b" * 64,
        )
        assert first != second

    def test_a_pooled_patch_reuses_its_key_across_a_resume(self):
        """Session ids are fresh UUIDs; keying on one would make the same bytes look new."""
        candidate = split_mod.sha256_text(DIFF_LOOSE)
        assert manifest_mod.job_key(
            batch_id="b1", bug="Demo/1", config="vanta", mode="review", seed=0,
            candidate=candidate,
        ) == manifest_mod.job_key(
            batch_id="b1", bug="Demo/1", config="vanta", mode="review", seed=0,
            candidate=candidate,
        )


class TestManifest:
    """Evidence is append-only; the cursor is the only thing that is rewritten (§7.3)."""

    def test_rows_round_trip_one_json_object_per_line(self, tmp_path):
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.append({"key": "k1", "bug": "Demo/1"})
        manifest.append({"key": "k2", "bug": "Demo/1"})
        text = (tmp_path / manifest_mod.MANIFEST_NAME).read_text()
        assert text.endswith("\n")
        assert len(text.splitlines()) == 2
        assert [row["key"] for row in manifest.records()] == ["k1", "k2"]

    def test_appending_never_rewrites_an_earlier_row(self, tmp_path):
        """A row a paper cites must not be able to change under a reader."""
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.append({"key": "k1"})
        first = (tmp_path / manifest_mod.MANIFEST_NAME).read_text()
        manifest.append({"key": "k2"})
        assert (tmp_path / manifest_mod.MANIFEST_NAME).read_text().startswith(first)

    def test_a_torn_last_line_is_dropped_not_raised(self, tmp_path):
        """A process killed mid-write must leave the batch readable."""
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.append({"key": "k1"})
        with open(manifest.path, "a", encoding="utf-8") as handle:
            handle.write('{"key": "k2", "bug": "Dem')
        assert [row["key"] for row in manifest.records()] == ["k1"]

    def test_find_returns_the_latest_row_for_a_key(self, tmp_path):
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.append({"key": "k1", "label": "first"})
        manifest.append({"key": "k1", "label": "second"})
        assert manifest.find("k1")["label"] == "second"
        assert manifest.find("absent") is None

    def test_only_done_is_skipped_so_running_and_failed_are_re_run(self, tmp_path):
        """`running` means "no trustworthy row yet", and `failed` means "no evidence at all"."""
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.mark("k1", manifest_mod.DONE)
        manifest.mark("k2", manifest_mod.RUNNING)
        manifest.mark("k3", manifest_mod.FAILED)
        assert manifest.is_done("k1") is True
        assert manifest.is_done("k2") is False
        assert manifest.is_done("k3") is False
        assert manifest.is_done("never-seen") is False

    def test_mark_leaves_no_temp_file_behind(self, tmp_path):
        """`mark` writes a temp file then `os.replace`s it — atomicity is the whole point."""
        manifest = manifest_mod.Manifest(tmp_path)
        manifest.mark("k1", manifest_mod.DONE)
        assert [p.name for p in tmp_path.iterdir()] == [manifest_mod.STATE_NAME]

    def test_a_corrupt_cursor_reads_as_empty_rather_than_crashing(self, tmp_path):
        write(tmp_path / manifest_mod.STATE_NAME, "{not json")
        assert manifest_mod.Manifest(tmp_path).state() == {}

    def test_a_missing_cursor_reads_as_empty(self, tmp_path):
        assert manifest_mod.Manifest(tmp_path).state() == {}

    def test_pending_keeps_everything_that_is_not_done(self, tmp_path):
        class FakeJob:
            def __init__(self, key):
                self.key = key

        manifest = manifest_mod.Manifest(tmp_path)
        manifest.mark("k1", manifest_mod.DONE)
        assert [job.key for job in manifest.pending([FakeJob("k1"), FakeJob("k2")])] == ["k2"]

    def test_a_cursor_that_is_not_a_dict_reads_as_empty(self, tmp_path):
        write(tmp_path / manifest_mod.STATE_NAME, '["k1"]')
        assert manifest_mod.Manifest(tmp_path).state() == {}


class TestVantaTestCmd:
    """Vanta splits `--test-cmd` on whitespace with no quoting (`harness::build_test_cmd`)."""

    def test_it_drops_the_split_report_flag_but_keeps_the_target(self):
        assert runner_mod.vanta_test_cmd(
            ["python", "-m", "pytest", "tests", "--json-report-file={report}"]
        ) == "python -m pytest tests"

    def test_it_drops_every_json_report_spelling(self):
        assert runner_mod.vanta_test_cmd(
            ["python", "-m", "pytest", "tests", "--json-report", "--json-report-file={report}"]
        ) == "python -m pytest tests"

    def test_whitespace_in_an_element_is_an_error_not_a_silent_split(self):
        """A silently split element would run a different test target than the split named."""
        with pytest.raises(runner_mod.RunnerError, match="whitespace"):
            runner_mod.vanta_test_cmd(["python", "-m", "pytest", "tests/test a.py"])

    def test_a_command_that_is_nothing_but_report_flags_is_an_error(self):
        with pytest.raises(runner_mod.RunnerError, match="empty"):
            runner_mod.vanta_test_cmd(["--json-report-file={report}"])


class TestNormalizedDiff:
    """§7.1's dedup key: two patches that produce the same tree are one candidate."""

    def test_the_index_line_is_not_part_of_the_identity(self):
        assert "index " not in runner_mod.normalized_diff(DIFF_LOOSE)

    def test_a_and_b_prefixes_are_stripped(self):
        normalized = runner_mod.normalized_diff(DIFF_LOOSE)
        assert "diff --git pkg/mod.py pkg/mod.py" in normalized
        assert "--- pkg/mod.py" in normalized
        assert "+++ pkg/mod.py" in normalized
        assert "a/pkg" not in normalized and "b/pkg" not in normalized

    def test_trailing_whitespace_is_not_part_of_the_identity(self):
        assert runner_mod.normalized_diff(DIFF_LOOSE) == runner_mod.normalized_diff(DIFF_TIDY)

    def test_the_normal_form_is_idempotent(self):
        once = runner_mod.normalized_diff(DIFF_LOOSE)
        assert runner_mod.normalized_diff(once) == once

    def test_two_real_differences_stay_two_candidates(self):
        assert runner_mod.normalized_diff(DIFF_LOOSE) != runner_mod.normalized_diff(
            _diff("pkg/mod.py", "five")
        )


class TestNoopPatch:
    """The floor case must apply cleanly and change nothing(*§5.4*)."""

    def _tree(self, tmp_path: Path) -> Path:
        root = tmp_path / "vanta_tree"
        write(root / "pkg" / "zeta.py", "z = 1\n")
        write(root / "pkg" / "alpha.py", "a = 1\n")
        write(root / "tests" / "test_alpha.py", "def test_a():\n    assert True\n")
        write(root / "pkg" / "__pycache__" / "junk.py", "junk = 1\n")
        return root

    def test_the_source_file_is_chosen_by_sorted_path_so_the_bytes_are_stable(self, tmp_path):
        rel, _ = runner_mod.noop_patch_text(self._tree(tmp_path))
        assert rel == "pkg/alpha.py"

    def test_the_same_tree_yields_the_same_bytes(self, tmp_path):
        tree = self._tree(tmp_path)
        assert runner_mod.noop_patch_text(tree) == runner_mod.noop_patch_text(tree)

    def test_it_is_a_real_diff_and_not_an_empty_patch(self, tmp_path):
        """An empty patch is a no-op failure in `apply_patch_strict`, which would measure application."""
        _, text = runner_mod.noop_patch_text(self._tree(tmp_path))
        assert text.startswith("diff --git a/pkg/alpha.py b/pkg/alpha.py\n")
        assert "--- a/pkg/alpha.py" in text and "+++ b/pkg/alpha.py" in text
        assert "\n+\n" in text, "the floor change should be one added blank line"

    def test_it_survives_the_test_hunk_filter(self, tmp_path):
        _, text = runner_mod.noop_patch_text(self._tree(tmp_path))
        assert runner_mod.test_hunk_free(text, what="noop") == text

    def test_a_tree_with_no_non_test_module_is_an_error(self, tmp_path):
        write(tmp_path / "vanta_tree" / "tests" / "test_only.py", "def test_x(): pass\n")
        with pytest.raises(runner_mod.RunnerError, match="no-op floor"):
            runner_mod.noop_patch_text(tmp_path / "vanta_tree")


class TestTestHunkFree:
    """A candidate that is only test edits is not reviewable (§5.4)."""

    def test_a_test_only_patch_is_refused(self):
        with pytest.raises(runner_mod.RunnerError, match="changes only test files"):
            runner_mod.test_hunk_free(_diff("tests/test_mod.py", "assert True"), what="x")

    def test_the_test_section_is_dropped_and_the_source_section_kept(self):
        combined = _diff("tests/test_mod.py", "assert True") + _diff("pkg/mod.py", "four")
        kept = runner_mod.test_hunk_free(combined, what="x")
        assert "pkg/mod.py" in kept
        assert "tests/test_mod.py" not in kept


class TestPoolFailedPatches:
    """§7.1: the known-bad pool is deduplicated by normalized diff and capped at two."""

    def _pool(self, tmp_path, rows):
        return runner_mod.pool_failed_patches(
            batch_dir=tmp_path, prep=prep(root=tmp_path), records=rows
        )

    def test_equivalent_patches_collapse_to_one_candidate(self, tmp_path):
        rows = [
            _pool_row(tmp_path / "loose.patch", DIFF_LOOSE, session_id="sess-a"),
            _pool_row(tmp_path / "tidy.patch", DIFF_TIDY, session_id="sess-b"),
        ]
        pool = self._pool(tmp_path, rows)
        assert len(pool) == 1
        assert pool[0].normalized_sha256 == split_mod.sha256_text(
            runner_mod.normalized_diff(DIFF_LOOSE)
        )

    def test_the_survivor_is_the_smaller_hash_so_the_choice_is_reproducible(self, tmp_path):
        rows = [
            _pool_row(tmp_path / "loose.patch", DIFF_LOOSE, session_id="sess-a"),
            _pool_row(tmp_path / "tidy.patch", DIFF_TIDY, session_id="sess-b"),
        ]
        pool = self._pool(tmp_path, rows)
        assert pool[0].patch_sha256 == min(
            split_mod.sha256_text(DIFF_LOOSE), split_mod.sha256_text(DIFF_TIDY)
        )

    def test_at_most_two_candidates_survive_and_they_are_the_smallest_hashes(self, tmp_path):
        texts = [
            _diff("pkg/one.py", "a1"),
            _diff("pkg/two.py", "a2"),
            _diff("pkg/three.py", "a3"),
        ]
        rows = [
            _pool_row(tmp_path / f"p{i}.patch", text, session_id=f"sess-{i}")
            for i, text in enumerate(texts)
        ]
        pool = self._pool(tmp_path, rows)
        assert runner_mod.MAX_POOLED_CANDIDATES == 2
        assert [c.patch_sha256 for c in pool] == sorted(
            split_mod.sha256_text(text) for text in texts
        )[:2]

    def test_both_configs_contribute_and_the_source_says_which(self, tmp_path):
        rows = [
            _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"),
                      config="baseline", session_id="sess-base"),
            _pool_row(tmp_path / "b.patch", _diff("pkg/two.py", "a2"),
                      config="vanta", session_id="sess-vanta"),
        ]
        sources = {c.source for c in self._pool(tmp_path, rows)}
        assert sources == {"repair:baseline:sess-base", "repair:vanta:sess-vanta"}

    def test_pooled_candidates_keep_the_label_their_bytes_were_graded_with(self, tmp_path):
        row = _pool_row(
            tmp_path / "a.patch",
            _diff("pkg/one.py", "a1"),
            oracle={"apply_ok": True, "patch_good": False, "hidden_pass": False},
        )
        pool = self._pool(tmp_path, [row])
        assert pool[0].kind == runner_mod.FAILED_PATCH
        assert pool[0].oracle.apply_ok is True
        assert pool[0].oracle.patch_good is False

    def test_an_infra_failure_is_not_a_patch_to_review(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"),
                        verdict="InfraFailure")
        assert self._pool(tmp_path, [row]) == []

    def test_a_patch_that_never_applied_is_not_a_patch(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"),
                        oracle={"apply_ok": False, "patch_good": False})
        assert self._pool(tmp_path, [row]) == []

    def test_a_good_patch_is_not_known_bad(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"),
                        oracle={"apply_ok": True, "patch_good": True})
        assert self._pool(tmp_path, [row]) == []

    def test_a_row_without_a_patch_file_is_skipped(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"))
        row["artifacts"] = {"final_patch": str(tmp_path / "gone.patch")}
        assert self._pool(tmp_path, [row]) == []

    def test_a_repair_row_with_no_patch_at_all_is_skipped(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("pkg/one.py", "a1"))
        row["artifacts"] = {"final_patch": None}
        assert self._pool(tmp_path, [row]) == []

    def test_a_test_only_patch_is_skipped_rather_than_taking_the_batch_down(self, tmp_path):
        row = _pool_row(tmp_path / "a.patch", _diff("tests/test_mod.py", "assert True"))
        assert self._pool(tmp_path, [row]) == []


class TestCandidateSet:
    """The set is built once per bug and reused by both configs — that is what pairs the rows."""

    def _prep(self, tmp_path):
        root = tmp_path / "root"
        write(root / "vanta_tree" / "pkg" / "mod.py", "one\ntwo\nthree\n")
        return prep(root=root, bug=bug_info(bug_patch_txt=_diff("pkg/mod.py", "fix")))

    def test_it_is_the_developer_fix_then_the_noop_then_the_pooled_patches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(  # grading needs a VM; the ordering is what this test is about
            runner_mod, "grade_candidate", lambda **kw: kw["candidate"]
        )
        bug_prep = self._prep(tmp_path)
        rows = [_pool_row(tmp_path / "a.patch", _diff("pkg/other.py", "a1"))]
        candidates = runner_mod.candidate_set(
            batch_dir=tmp_path, prep=bug_prep, checkpoint="chk", disk_mb=8192, records=rows
        )
        assert [c.kind for c in candidates] == [
            runner_mod.DEV_FIX, runner_mod.NOOP, runner_mod.FAILED_PATCH
        ]

    def test_the_developer_fix_is_source_dev_and_carries_its_own_diff(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "grade_candidate", lambda **kw: kw["candidate"])
        candidates = runner_mod.candidate_set(
            batch_dir=tmp_path, prep=self._prep(tmp_path), checkpoint="chk", disk_mb=8192,
            records=[],
        )
        dev = candidates[0]
        assert dev.source == "dev"
        assert dev.patch_path.read_text() == _diff("pkg/mod.py", "fix")

    def test_a_bug_with_no_failed_patch_has_the_two_known_candidates_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "grade_candidate", lambda **kw: kw["candidate"])
        candidates = runner_mod.candidate_set(
            batch_dir=tmp_path, prep=self._prep(tmp_path), checkpoint="chk", disk_mb=8192,
            records=[],
        )
        assert [c.kind for c in candidates] == [runner_mod.DEV_FIX, runner_mod.NOOP]

    def test_the_noop_names_the_file_it_touched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "grade_candidate", lambda **kw: kw["candidate"])
        candidates = runner_mod.candidate_set(
            batch_dir=tmp_path, prep=self._prep(tmp_path), checkpoint="chk", disk_mb=8192,
            records=[],
        )
        assert candidates[1].source == "noop:pkg/mod.py"


class TestLabelFor:
    """§5.3's label table, as a pure function of (verdict, oracle)."""

    @pytest.mark.parametrize(
        "verdict,apply_ok,patch_good,expected",
        [
            ("AllTestsPassed", True, True, "true_approval"),
            ("AllTestsPassed", True, False, "false_approval"),
            ("Rejected", True, True, "false_reject"),
            ("Rejected", True, False, "true_reject"),
            ("MaxAttemptsReached", True, True, "unresolved_but_fixed"),
            ("MaxAttemptsReached", True, False, "unresolved"),
            ("InfraFailure", False, False, "infra_failure"),
            ("InfraFailure", True, True, "infra_failure"),
            # The censor wins over the verdict: a patch that never applied told us
            # nothing, so it cannot become a `true_reject` for Vanta (§5.2 step 2).
            ("Rejected", False, True, "unobservable"),
            ("PatchApplicationFailed", False, False, "unobservable"),
            ("HarnessFailure", False, False, "unobservable"),
            ("SomethingNobodyNamed", True, True, "unobservable"),
        ],
    )
    def test_the_table(self, verdict, apply_ok, patch_good, expected):
        oracle = oracle_mod.OracleResult(apply_ok=apply_ok, patch_good=patch_good)
        assert runner_mod.label_for(verdict, oracle) == expected

    def test_a_censored_oracle_states_its_reason(self):
        bug_prep = prep(fail_to_pass=["tests/test_a.py::test_b", "tests/test_a.py::test_a"])
        result = runner_mod.censored_oracle(bug_prep, "no final.patch for verdict Rejected")
        assert result.apply_ok is False
        assert result.patch_good is False
        assert result.label_input == "unobservable"
        assert result.apply_detail == "no final.patch for verdict Rejected"
        assert result.hidden_nodeids == sorted(bug_prep.fail_to_pass)


class TestParseSummary:
    """Key/value pairs, in the order the pipeline wrote them."""

    def test_pairs_are_read_in_order(self, tmp_path):
        path = write(
            tmp_path / "sess_session.jsonl",
            json.dumps({"phase": "session.start", "summary": "model=x"}) + "\n"
            + json.dumps({"phase": "run.summary", "summary": "attempts_run=1 commit=abc"}) + "\n",
        )
        raw, parsed = runner_mod.read_run_summary(path)
        assert raw == "attempts_run=1 commit=abc"
        assert list(parsed) == ["attempts_run", "commit"]

    def test_the_last_run_summary_wins(self, tmp_path):
        path = write(
            tmp_path / "sess_session.jsonl",
            json.dumps({"phase": "run.summary", "summary": "attempts_run=1"}) + "\n"
            + json.dumps({"phase": "run.summary", "summary": "attempts_run=3"}) + "\n",
        )
        assert runner_mod.read_run_summary(path)[1]["attempts_run"] == "3"

    def test_torn_and_foreign_lines_are_ignored(self, tmp_path):
        path = write(
            tmp_path / "sess_session.jsonl",
            '{"phase": "run.summary", "summary": "attempts_run=2"}\n{"phase": "run.sum\n',
        )
        assert runner_mod.read_run_summary(path)[1] == {"attempts_run": "2"}

    def test_a_missing_session_reads_as_empty(self, tmp_path):
        assert runner_mod.read_run_summary(tmp_path / "absent_session.jsonl") == ("", {})


class TestSamplingBlock:
    """§7.3's provenance block: the pipeline's per-call schedule plus the server's `top_p`."""

    SUMMARY = (
        "attempts_run=2 fixed_at_attempt_1=false final_decision=Rejected config=vanta "
        "mode=repair seed=0 model=qwen3.5-27b top_p=unset "
        "temperature_creator=0.1 max_tokens_creator=32768 "
        "temperature_attacker_generate=0.2 max_tokens_attacker_generate=32768 "
        "temperature_intent_audit=0.0 max_tokens_intent_audit=2048 "
        "commit=" + "d" * 40 + " dirty=false"
    )

    def test_the_policy_named_is_the_split_s_own_constant(self):
        block = runner_mod.sampling_block(runner_mod.parse_summary(self.SUMMARY))
        assert block["source"] == split_mod.SAMPLING_POLICY

    def test_the_sites_come_back_in_the_order_the_pipeline_emitted_them(self):
        block = runner_mod.sampling_block(runner_mod.parse_summary(self.SUMMARY))
        assert [entry["site"] for entry in block["pinned"]] == [
            "creator", "attacker_generate", "intent_audit"
        ]

    def test_numbers_keep_their_type(self):
        block = runner_mod.sampling_block(runner_mod.parse_summary(self.SUMMARY))
        by_site = {entry["site"]: entry for entry in block["pinned"]}
        assert by_site["creator"] == {"site": "creator", "temperature": 0.1, "max_tokens": 32768}
        assert by_site["intent_audit"]["temperature"] == 0.0
        assert by_site["intent_audit"]["max_tokens"] == 2048

    def test_an_unset_top_p_is_recorded_as_unset_not_as_a_plausible_default(self):
        block = runner_mod.sampling_block(runner_mod.parse_summary(self.SUMMARY))
        assert block["server_top_p"] is None
        assert "unset" in block["server_top_p_note"]

    def test_a_published_top_p_is_recorded_with_its_source(self):
        summary = runner_mod.parse_summary(self.SUMMARY.replace("top_p=unset", "top_p=0.95"))
        block = runner_mod.sampling_block(summary)
        assert block["server_top_p"] == 0.95
        assert "/props" in block["server_top_p_note"]

    def test_an_empty_summary_claims_nothing(self):
        block = runner_mod.sampling_block({})
        assert block["pinned"] == []
        assert block["server_top_p"] is None

    def test_the_pinned_schedule_matches_the_pipeline_constant(self):
        """`SAMPLING_SCHEDULE` is the only definition of the per-call schedule (§4.3)."""
        text = (REPO_ROOT / "src" / "llm" / "mod.rs").read_text(encoding="utf-8")
        block = text.split("pub const SAMPLING_SCHEDULE", 1)[1].split("];", 1)[0]
        entries = re.findall(r'\("([a-z_]+)",\s*([0-9.]+),\s*(\d+)\)', block)
        assert entries, "could not read SAMPLING_SCHEDULE out of src/llm/mod.rs"
        summary = runner_mod.parse_summary(
            " ".join(
                f"temperature_{site}={temperature} max_tokens_{site}={max_tokens}"
                for site, temperature, max_tokens in entries
            )
        )
        assert runner_mod.sampling_block(summary)["pinned"] == [
            {"site": site, "temperature": float(temperature), "max_tokens": int(max_tokens)}
            for site, temperature, max_tokens in entries
        ]


def _gate_row(
    *,
    bug: str = "Demo/1",
    config: str = "vanta",
    mode: str = "repair",
    seed: int = 0,
    verdict: str = "Rejected",
    label: str = "true_reject",
    kind: str | None = None,
    normalized: str | None = None,
    candidate_good: bool = False,
) -> dict:
    """A manifest row in the shape `wave_gate_failures` inspects."""
    row = {
        "bug": bug,
        "config": config,
        "mode": mode,
        "seed": seed,
        "verdict": verdict,
        "label": label,
        "session_id": f"{config}-{mode}-{kind or 'repair'}",
        "oracle": {"apply_ok": True, "patch_good": candidate_good, "apply_detail": "", "error": ""},
        "candidate": None,
    }
    if kind is not None:
        row["candidate"] = {
            "kind": kind,
            "source": "dev" if kind == runner_mod.DEV_FIX else "noop",
            "normalized_sha256": normalized or (kind[0] * 64),
            "oracle": {"apply_ok": True, "patch_good": candidate_good},
        }
    return row


def _healthy_wave() -> list[dict]:
    """Both configs, one repair row each, and the same two candidates reviewed by both."""
    rows: list[dict] = []
    for config in ("baseline", "vanta"):
        rows.append(_gate_row(config=config, mode="repair"))
        rows.append(
            _gate_row(config=config, mode="review", kind=runner_mod.DEV_FIX,
                      verdict="Rejected", label="false_reject", normalized="d" * 64,
                      candidate_good=True)
        )
        rows.append(
            _gate_row(config=config, mode="review", kind=runner_mod.NOOP,
                      verdict="Rejected", label="true_reject", normalized="n" * 64)
        )
    return rows


class TestWaveGate:
    """§8.2's go/no-go gate: only the failures that invalidate every later number."""

    def test_a_healthy_wave_reports_nothing(self):
        assert runner_mod.wave_gate_failures(_healthy_wave()) == []

    def test_an_empty_wave_is_a_failure(self):
        assert runner_mod.wave_gate_failures([]) == ["the wave produced no records at all"]

    def test_an_infra_failure_stops_the_wave(self):
        rows = _healthy_wave()
        rows[0]["verdict"] = "InfraFailure"
        problems = runner_mod.wave_gate_failures(rows)
        assert any("InfraFailure" in problem for problem in problems)

    def test_a_censored_row_stops_the_wave_and_names_the_reason(self):
        rows = _healthy_wave()
        rows[0]["verdict"] = "PatchApplicationFailed"
        rows[0]["label"] = "unobservable"
        rows[0]["oracle"]["apply_ok"] = False
        rows[0]["oracle"]["apply_detail"] = "does not apply"
        problems = runner_mod.wave_gate_failures(rows)
        assert any("unobservable" in problem and "does not apply" in problem for problem in problems)

    def test_a_developer_fix_that_grades_bad_proves_the_oracle_is_wrong(self):
        rows = _healthy_wave()
        for row in rows:
            if row["candidate"] and row["candidate"]["kind"] == runner_mod.DEV_FIX:
                row["candidate"]["oracle"]["patch_good"] = False
        problems = runner_mod.wave_gate_failures(rows)
        assert any("developer fix graded patch_good=false" in problem for problem in problems)

    def test_a_noop_that_grades_good_proves_the_oracle_is_wrong(self):
        rows = _healthy_wave()
        for row in rows:
            if row["candidate"] and row["candidate"]["kind"] == runner_mod.NOOP:
                row["candidate"]["oracle"]["patch_good"] = True
        problems = runner_mod.wave_gate_failures(rows)
        assert any("no-op floor graded patch_good=true" in problem for problem in problems)

    def test_a_review_row_without_its_counterpart_is_an_incomplete_pair(self):
        """The comparison is the point: a lone row silently halves it instead of failing."""
        rows = [
            row
            for row in _healthy_wave()
            if not (
                row["config"] == "baseline"
                and row["candidate"]
                and row["candidate"]["kind"] == runner_mod.DEV_FIX
            )
        ]
        problems = runner_mod.wave_gate_failures(rows)
        assert any("the pair is incomplete" in problem for problem in problems)

    def test_repair_rows_alone_do_not_trip_the_pairing_check(self):
        """There is no candidate to pair before the review passes run."""
        rows = [_gate_row(config=config, mode="repair") for config in ("baseline", "vanta")]
        assert runner_mod.wave_gate_failures(rows) == []


class TestDigestAndLabels:
    """The report is a view of the rows, built from the rows and nothing else."""

    ROWS = [
        {
            "bug": "Demo/1", "config": "vanta", "mode": "repair", "seed": 0,
            "model": "qwen3.5-27b",
            "harness_commit": "c" * 40, "harness_dirty": False,
            "vanta_build_commit": "d" * 40, "vanta_build_dirty": False,
            "label": "true_reject", "session_id": "sess-1", "candidate": None,
        },
        {
            "bug": "Demo/1", "config": "vanta", "mode": "review", "seed": 0,
            "model": "qwen3.5-27b",
            "harness_commit": "c" * 40, "harness_dirty": False,
            "vanta_build_commit": "d" * 40, "vanta_build_dirty": False,
            "label": "false_reject", "session_id": "sess-2",
            "candidate": {
                "kind": "dev_fix", "source": "dev",
                "oracle": {"apply_ok": True, "patch_good": True},
            },
        },
        {
            "bug": "Demo/1", "config": "baseline", "mode": "review", "seed": 0,
            "model": "qwen3.5-27b",
            "harness_commit": "c" * 40, "harness_dirty": False,
            "vanta_build_commit": "d" * 40, "vanta_build_dirty": False,
            "label": "true_reject", "session_id": "sess-3",
            "candidate": {
                "kind": "noop", "source": "noop:pkg/mod.py",
                "oracle": {"apply_ok": True, "patch_good": False},
            },
        },
    ]

    def test_only_review_rows_are_labelled(self, tmp_path):
        """A repair row's patch has no declared truth — that is what its oracle column is for."""
        path = runner_mod.write_labels(self.ROWS, tmp_path / "run_labels.json")
        labels = json.loads(path.read_text())
        assert set(labels) == {"sess-2", "sess-3"}
        assert labels["sess-2"] == {
            "known_bad": False, "bug": "Demo/1", "config": "vanta",
            "mode": "review", "kind": "dev_fix", "source": "dev",
        }
        assert labels["sess-3"]["known_bad"] is True

    def test_a_row_without_a_session_is_not_labelled(self, tmp_path):
        rows = [dict(self.ROWS[1], session_id=None)]
        labels = json.loads(runner_mod.write_labels(rows, tmp_path / "l.json").read_text())
        assert labels == {}

    def test_the_digest_groups_by_the_provenance_axes(self):
        payload = runner_mod.digest(self.ROWS)
        assert payload["runs"] == 3
        # config, mode, seed, model and both commits are the axes: a repair row and a
        # review row of the same config are two groups, not one.
        assert len(payload["groups"]) == 3
        vanta_review = [
            g for g in payload["groups"] if g["config"] == "vanta" and g["mode"] == "review"
        ]
        assert len(vanta_review) == 1
        assert vanta_review[0]["runs"] == 1
        assert vanta_review[0]["labels"] == {"false_reject": 1}
        assert vanta_review[0]["model"] == "qwen3.5-27b"
        assert vanta_review[0]["vanta_build_dirty"] is False
        assert vanta_review[0]["harness_dirty"] is False

    def test_the_model_is_one_of_the_grouping_axes(self):
        other = dict(self.ROWS[0], model="gemma-3-12b")
        assert len(runner_mod.digest(self.ROWS + [other])["groups"]) == 4

    def test_two_builds_of_vanta_do_not_share_a_group(self):
        other = dict(self.ROWS[0], vanta_build_commit="e" * 40)
        assert len(runner_mod.digest(self.ROWS + [other])["groups"]) == 4

    def test_two_harness_commits_do_not_share_a_group(self):
        """The commit that was reported as `vanta_commit` was the binary's; the
        harness's own is a separate axis now, so a harness that changed mid-batch
        cannot be pooled with the rows it did not run."""
        other = dict(self.ROWS[0], harness_commit="e" * 40)
        assert len(runner_mod.digest(self.ROWS + [other])["groups"]) == 4

    def test_the_rendered_digest_has_a_header_and_one_line_per_group(self):
        rendered = runner_mod.render_digest(runner_mod.digest(self.ROWS))
        lines = rendered.splitlines()
        assert "config" in lines[0] and "labels" in lines[0]
        assert "harness" in lines[0] and "build" in lines[0], (
            "both commits are provenance, and a column that names neither is the"
            " bug that made a digest claim 44f58ab for a batch at b2efa49"
        )
        assert len(lines) == 2 + len(runner_mod.digest(self.ROWS)["groups"])

    def test_write_batch_report_writes_all_three_artifacts(self, tmp_path):
        labels_path, payload = runner_mod.write_batch_report(tmp_path, self.ROWS)
        assert labels_path == tmp_path / "run_labels.json"
        assert payload["runs"] == 3
        assert json.loads((tmp_path / "run_digest.json").read_text()) == payload
        assert (tmp_path / "run_digest.md").read_text().startswith("config")


class TestStaleBinary:
    """§7.3: two commits, and a binary older than the Rust it claims to be built from.

    The bug this covers was a digest whose `commit` column read `44f58ab` while HEAD
    was `b2efa49` — the binary's build commit under a name that read like the
    harness's. Something must now say so *before* the wave runs, because after it the
    only honest option is to relabel the results.
    """

    HEAD = runner_mod.Revision(commit="b2efa49", dirty=False)
    BUILD = runner_mod.Revision(commit="44f58ab", dirty=False)

    def test_the_binary_matching_the_working_tree_is_silent(self):
        assert runner_mod.stale_binary_message(
            binary=self.HEAD, head=self.HEAD, changed=[]
        ) is None

    def test_rust_changes_since_the_build_are_named(self):
        message = runner_mod.stale_binary_message(
            binary=self.BUILD, head=self.HEAD,
            changed=["src/agents/attacker.rs", "Cargo.toml"],
        )
        assert message is not None
        assert "44f58ab" in message and "b2efa49" in message
        assert "src/agents/attacker.rs" in message

    def test_a_harness_only_change_is_not_a_stale_binary(self):
        """The normal case, and the one that produced the bad digest: HEAD moved on
        with `scripts/` while `src/` did not, so the binary is fine."""
        assert runner_mod.stale_binary_message(
            binary=self.BUILD, head=self.HEAD, changed=[]
        ) is None

    def test_a_dirty_build_has_no_commit_that_describes_it(self):
        message = runner_mod.stale_binary_message(
            binary=runner_mod.Revision(commit="44f58ab", dirty=True),
            head=self.HEAD, changed=[],
        )
        assert message is not None and "dirty" in message

    def test_an_unreadable_build_commit_is_admitted_rather_than_guessed(self):
        message = runner_mod.stale_binary_message(
            binary=runner_mod.Revision(), head=self.HEAD, changed=None
        )
        assert message is not None and "could not be read" in message

    def test_the_build_commit_is_read_from_cargos_build_script_output(self, tmp_path):
        binary = write(tmp_path / "target" / "release" / "vanta_m3", "")
        write(
            tmp_path / "target" / "release" / "build" / "vanta-ef12b856" / "output",
            "cargo:rerun-if-changed=build.rs\n"
            "cargo:rustc-env=VANTA_BUILD_COMMIT=44f58ab\n"
            "cargo:rustc-env=VANTA_BUILD_DIRTY=false\n",
        )
        assert runner_mod.binary_revision(binary) == runner_mod.Revision("44f58ab", False)

    def test_two_build_script_outputs_disagreeing_yield_no_commit(self, tmp_path):
        binary = write(tmp_path / "target" / "release" / "vanta_m3", "")
        for hash_ in ("aaaa", "bbbb"):
            write(
                tmp_path / "target" / "release" / "build" / f"vanta-{hash_}" / "output",
                f"cargo:rustc-env=VANTA_BUILD_COMMIT={hash_}\n",
            )
        assert runner_mod.binary_revision(binary) == runner_mod.Revision()

    def test_no_build_directory_yields_no_commit(self, tmp_path):
        binary = write(tmp_path / "target" / "release" / "vanta_m3", "")
        assert runner_mod.binary_revision(binary) == runner_mod.Revision()

    def test_the_warning_reads_git_and_the_binary_through_one_seam(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "git_revision", lambda *a, **k: self.HEAD)
        monkeypatch.setattr(runner_mod, "binary_revision", lambda path: self.BUILD)
        monkeypatch.setattr(runner_mod, "changed_since", lambda *a: ["src/lib.rs"])
        warning = runner_mod.stale_binary_warning(tmp_path / "vanta_m3")
        assert warning is not None and "src/lib.rs" in warning

    def test_the_rust_paths_exclude_the_harness_itself(self):
        """A warning that fired on every `scripts/` edit would be ignored, which is
        the one thing a run-start warning cannot afford."""
        assert "src" in runner_mod.VANTA_SOURCE_PATHS
        assert "build.rs" in runner_mod.VANTA_SOURCE_PATHS
        assert not any(path.startswith("scripts") for path in runner_mod.VANTA_SOURCE_PATHS)


class TestPhaseOrder:
    """D13: repairs first, then both reviews. The ordering is a correctness property."""

    def test_the_declared_order_is_repairs_then_reviews(self):
        assert runner_mod.PHASES == (
            ("repair", "baseline"),
            ("repair", "vanta"),
            ("review", "baseline"),
            ("review", "vanta"),
        )

    def test_the_split_s_own_configs_match_it(self):
        runner_mod._require_phase_order(["baseline", "vanta"])

    def test_naming_the_planner_placeholder_is_refused(self):
        """`vanta_planner` is a §4.4 placeholder: it would only add InfraFailure rows."""
        with pytest.raises(runner_mod.RunnerError) as caught:
            runner_mod._require_phase_order(["baseline", "vanta", "vanta_planner"])
        message = str(caught.value)
        assert "split configs must be exactly ['baseline', 'vanta'] for paired review" in message
        assert "got ['baseline', 'vanta', 'vanta_planner']" in message
        assert "vanta_planner is a placeholder that returns InfraFailure" in message
        assert "must not be in a run split" in message

    def test_the_message_is_about_the_config_list_not_about_the_ordering(self):
        """A one-config split cannot produce paired rows, so the sentence must say so."""
        with pytest.raises(runner_mod.RunnerError, match="must be exactly"):
            runner_mod._require_phase_order(["vanta"])

    def test_a_reordered_schedule_is_refused_rather_than_run(self, monkeypatch):
        monkeypatch.setattr(
            runner_mod,
            "PHASES",
            (
                ("review", "baseline"),
                ("repair", "baseline"),
                ("review", "vanta"),
                ("repair", "vanta"),
            ),
        )
        with pytest.raises(runner_mod.RunnerError, match="correctness property"):
            runner_mod._require_phase_order(["baseline", "vanta"])

    def test_the_paired_configs_are_the_two_the_design_names(self):
        assert runner_mod.REVIEW_CONFIGS == ("baseline", "vanta")


class TestVantaEnv:
    """§7.4's controlled environment, derived from the same switch pair the record names."""

    def _env(self, **over):
        args = dict(config="vanta", seed=0, checkpoint="chk", disk_mb=8192, base={})
        args.update(over)
        return runner_mod.vanta_env(**args)

    def test_config_switches(self):
        assert runner_mod.config_switches("baseline") == (False, False)
        assert runner_mod.config_switches("vanta") == (True, False)
        assert runner_mod.config_switches("vanta_planner") == (True, True)
        with pytest.raises(runner_mod.RunnerError, match="unknown config"):
            runner_mod.config_switches("turbo")

    def test_the_baseline_turns_the_attacker_off(self):
        assert self._env(config="baseline")["VANTA_ATTACKER"] == "off"
        assert self._env(config="vanta")["VANTA_ATTACKER"] == "on"

    def test_the_planner_placeholder_stays_off(self):
        assert self._env(config="vanta_planner")["VANTA_PLANNER"] == "on"
        assert self._env(config="vanta")["VANTA_PLANNER"] == "off"

    def test_seed_zero_is_still_sent_as_a_seed(self):
        """Unset means "no seed key in the body"; 0 is a request to pin 0 (§4.4)."""
        assert self._env(seed=0)["VANTA_LLM_SEED"] == "0"
        assert self._env(seed=7)["VANTA_LLM_SEED"] == "7"

    def test_the_dependency_self_heal_is_disabled(self):
        """Left on, it would build a derived checkpoint the per-run VM has no network for."""
        env = self._env()
        assert env["VANTA_SHURU_AUTO_DEPS"] == "0"
        assert env["VANTA_SANDBOX_BACKEND"] == "shuru"
        assert env["VANTA_SHURU_CHECKPOINT"] == "chk"
        assert env["VANTA_SHURU_DISK_MB"] == "8192"

    def test_it_does_not_invent_a_model_or_a_base_url(self):
        env = self._env(base={"VANTA_LLM_BASE_URL": "http://127.0.0.1:8080"})
        assert env["VANTA_LLM_BASE_URL"] == "http://127.0.0.1:8080"
        assert "VANTA_LLM_MODEL" not in env


class TestPreflight:
    """Shuru mounts only paths inside the process CWD, so both guards are load-bearing."""

    def test_the_runner_refuses_a_foreign_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(runner_mod.RunnerError, match="CWD"):
            runner_mod.require_repo_cwd()

    def test_the_repo_root_is_accepted(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        assert runner_mod.require_repo_cwd() == REPO_ROOT

    def test_a_batch_outside_the_repo_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        with pytest.raises(runner_mod.RunnerError, match="not inside"):
            runner_mod.require_batch_dir(tmp_path / "batch")

    def test_a_batch_under_runs_bugsinpy_is_accepted(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        assert runner_mod.require_batch_dir(runner_mod.RUNS_DIR / "wave1") == (
            runner_mod.RUNS_DIR / "wave1"
        )

    def test_a_missing_binary_says_how_to_build_it(self, tmp_path):
        with pytest.raises(runner_mod.RunnerError, match="cargo build"):
            runner_mod.require_vanta_bin(tmp_path / "vanta_m3")

    def test_a_present_binary_is_returned(self, tmp_path):
        binary = write(tmp_path / "vanta_m3", "#!/bin/sh\n")
        assert runner_mod.require_vanta_bin(binary) == binary


def _outcome(tmp_path: Path, **over) -> runner_mod.SessionOutcome:
    """One finished `vanta_m3` invocation, with a plausible `run.summary`."""
    base = dict(
        session_id="sess-1",
        exit_code=0,
        timed_out=False,
        decision="Rejected",
        summary={
            "attempts_run": "2",
            "fixed_at_attempt_1": "false",
            "final_decision": "Rejected",
            "config": "vanta",
            "mode": "repair",
            "seed": "0",
            "model": "qwen3.5-27b",
            "top_p": "unset",
            "temperature_creator": "0.1",
            "max_tokens_creator": "32768",
            "commit": "d" * 40,
            "dirty": "false",
        },
        raw_summary="attempts_run=2 final_decision=Rejected",
        session_path=tmp_path / "vanta_sessions" / "sess-1" / "sess-1_session.jsonl",
        debug_dir=None,
        final_patch=None,
        log_path=tmp_path / "run_logs" / "sess-1.log",
        run_secs=1.5,
    )
    base.update(over)
    return runner_mod.SessionOutcome(**base)


def _job(
    bug_prep,
    *,
    config="vanta",
    mode=runner_mod.REPAIR,
    candidate=None,
    harness=None,
) -> runner_mod.Job:
    return runner_mod.Job(
        batch_id="b1",
        wave=1,
        bug=bug_prep.key,
        prep=bug_prep,
        entry={
            "python_version": "3.8.1",
            "pytest_version": "8.0.2",
            "json_report_version": "1.5.0",
            "checkpoint": "chk",
        },
        config=config,
        mode=mode,
        seed=0,
        checkpoint="chk",
        python_version_pinned="3.8.1",
        candidate=candidate,
        harness=harness,
    )


class TestRunJob:
    """The row, the censor and the resume ordering — all three fail silently if wrong."""

    def _paths(self, tmp_path):
        return runner_mod.BatchPaths.under(tmp_path)

    def test_running_is_marked_before_the_work_and_done_only_after_the_row(
        self, tmp_path, monkeypatch
    ):
        paths = self._paths(tmp_path)
        manifest = manifest_mod.Manifest(tmp_path)
        bug_prep = prep(root=tmp_path / "root")
        job = _job(bug_prep)
        seen: dict[str, object] = {}
        real_append = manifest_mod.Manifest.append

        def spy_append(self, record):
            seen["state_at_append"] = self.state().get(record["key"])
            return real_append(self, record)

        def fake_invoke(**kwargs):
            seen["state_at_invoke"] = manifest.state().get(job.key)
            return _outcome(tmp_path)

        monkeypatch.setattr(manifest_mod.Manifest, "append", spy_append)
        monkeypatch.setattr(runner_mod, "invoke_vanta", fake_invoke)
        monkeypatch.setattr(
            oracle_mod,
            "grade",
            lambda **kwargs: oracle_mod.OracleResult(apply_ok=True, apply_detail="applied"),
        )

        record = runner_mod.run_job(
            job=job, paths=paths, manifest=manifest, vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4, disk_mb=8192, python_version="3.8.20",
        )

        assert seen["state_at_invoke"] == manifest_mod.RUNNING
        assert seen["state_at_append"] == manifest_mod.RUNNING, (
            "`done` before the row is flushed would let a crash skip a job with no evidence"
        )
        assert manifest.state()[job.key] == manifest_mod.DONE
        assert manifest.find(job.key)["key"] == job.key
        assert record["key"] == job.key

    def test_the_row_records_the_actual_and_the_pinned_interpreter(self, tmp_path, monkeypatch):
        """§12.3: the checkpoint holds 3.8.20 while the bug declares 3.8.1."""
        monkeypatch.setattr(runner_mod, "invoke_vanta", lambda **kwargs: _outcome(tmp_path))
        monkeypatch.setattr(
            oracle_mod, "grade", lambda **kwargs: oracle_mod.OracleResult(apply_ok=True)
        )
        record = runner_mod.run_job(
            job=_job(prep(root=tmp_path / "root")), paths=self._paths(tmp_path),
            manifest=manifest_mod.Manifest(tmp_path), vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4, disk_mb=8192, python_version="3.8.20",
        )
        assert record["python_version"] == "3.8.20"
        assert record["python_version_pinned"] == "3.8.1"
        assert record["sampling"]["source"] == split_mod.SAMPLING_POLICY
        assert record["artifacts"]["session_dir"] == "vanta_sessions/sess-1"
        assert record["oracle_reused"] is False

    def test_the_row_names_the_binary_s_build_commit_and_the_harness_s_own(
        self, tmp_path, monkeypatch
    ):
        """§7.3: two commits, because they are two trees and they routinely differ.

        `run.summary`'s `commit` is baked in at cargo build time; the harness commit
        is HEAD when the batch runs. Recording the first under a name that reads like
        the second is what made a digest claim 44f58ab for a batch whose HEAD was
        b2efa49.
        """
        monkeypatch.setattr(runner_mod, "invoke_vanta", lambda **kwargs: _outcome(tmp_path))
        monkeypatch.setattr(
            oracle_mod, "grade", lambda **kwargs: oracle_mod.OracleResult(apply_ok=True)
        )
        record = runner_mod.run_job(
            job=_job(
                prep(root=tmp_path / "root"),
                harness=runner_mod.Revision(commit="b2efa49", dirty=False),
            ),
            paths=self._paths(tmp_path), manifest=manifest_mod.Manifest(tmp_path),
            vanta_bin=tmp_path / "vanta_m3", max_attempts=4, disk_mb=8192,
            python_version="3.8.20",
        )
        assert record["vanta_build_commit"] == "d" * 40
        assert record["vanta_build_dirty"] is False
        assert record["harness_commit"] == "b2efa49"
        assert record["harness_dirty"] is False

    def test_a_row_without_a_harness_revision_says_unknown_rather_than_guessing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(runner_mod, "invoke_vanta", lambda **kwargs: _outcome(tmp_path))
        monkeypatch.setattr(
            oracle_mod, "grade", lambda **kwargs: oracle_mod.OracleResult(apply_ok=True)
        )
        record = runner_mod.run_job(
            job=_job(prep(root=tmp_path / "root")), paths=self._paths(tmp_path),
            manifest=manifest_mod.Manifest(tmp_path), vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4, disk_mb=8192, python_version="3.8.20",
        )
        assert record["harness_commit"] is None
        assert record["harness_dirty"] is None

    def test_a_review_row_reuses_the_candidate_label_and_never_re_grades(
        self, tmp_path, monkeypatch
    ):
        """The same bytes were graded already; a second grading repeats an experiment."""
        bug_prep = prep(root=tmp_path / "root", bug=bug_info(bug_patch_txt=_diff("pkg/mod.py", "fix")))
        candidate = runner_mod.write_candidate_patch(
            tmp_path, bug_prep, runner_mod.DEV_FIX, _diff("pkg/mod.py", "fix"), source="dev"
        )
        candidate.oracle = oracle_mod.OracleResult(apply_ok=True, patch_good=True)
        monkeypatch.setattr(
            runner_mod,
            "invoke_vanta",
            lambda **kwargs: _outcome(
                tmp_path, decision="AllTestsPassed", final_patch=tmp_path / "final.patch"
            ),
        )
        monkeypatch.setattr(
            oracle_mod, "grade", lambda **kwargs: pytest.fail("the oracle was re-run")
        )
        record = runner_mod.run_job(
            job=_job(bug_prep, mode=runner_mod.REVIEW, candidate=candidate),
            paths=self._paths(tmp_path), manifest=manifest_mod.Manifest(tmp_path),
            vanta_bin=tmp_path / "vanta_m3", max_attempts=4, disk_mb=8192,
            python_version="3.8.20",
        )
        assert record["oracle_reused"] is True
        assert record["label"] == "true_approval"
        assert record["artifacts"]["oracle_json"] is None
        assert record["candidate"]["kind"] == runner_mod.DEV_FIX

    def test_a_review_row_that_produced_no_patch_is_censored_not_graded(
        self, tmp_path, monkeypatch
    ):
        """A verdict the run never reached must not be credited to Vanta (§5.4)."""
        bug_prep = prep(root=tmp_path / "root", bug=bug_info(bug_patch_txt=_diff("pkg/mod.py", "fix")))
        candidate = runner_mod.write_candidate_patch(
            tmp_path, bug_prep, runner_mod.DEV_FIX, _diff("pkg/mod.py", "fix"), source="dev"
        )
        candidate.oracle = oracle_mod.OracleResult(apply_ok=True, patch_good=True)
        monkeypatch.setattr(
            runner_mod,
            "invoke_vanta",
            lambda **kwargs: _outcome(
                tmp_path, decision="PatchApplicationFailed", final_patch=None
            ),
        )
        monkeypatch.setattr(
            oracle_mod, "grade", lambda **kwargs: pytest.fail("the candidate was re-graded")
        )
        record = runner_mod.run_job(
            job=_job(bug_prep, mode=runner_mod.REVIEW, candidate=candidate),
            paths=self._paths(tmp_path), manifest=manifest_mod.Manifest(tmp_path),
            vanta_bin=tmp_path / "vanta_m3", max_attempts=4, disk_mb=8192,
            python_version="3.8.20",
        )
        assert record["oracle_reused"] is False
        assert record["label"] == "unobservable"
        assert record["oracle"]["apply_ok"] is False
        assert "no final.patch" in record["oracle"]["apply_detail"]

    def test_a_job_that_left_no_session_writes_no_row_and_stays_failed(
        self, tmp_path, monkeypatch
    ):
        """A half-written row would be a fabricated dataset entry."""
        paths = self._paths(tmp_path)
        manifest = manifest_mod.Manifest(tmp_path)
        job = _job(prep(root=tmp_path / "root"))
        monkeypatch.setattr(
            runner_mod,
            "invoke_vanta",
            lambda **kwargs: _outcome(
                tmp_path, session_id=None, decision="", error="no session banner"
            ),
        )
        record = runner_mod.run_job(
            job=job, paths=paths, manifest=manifest, vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4, disk_mb=8192, python_version="3.8.20",
        )
        assert record is None
        assert manifest.find(job.key) is None
        assert manifest.state()[job.key] == manifest_mod.FAILED
        assert manifest.is_done(job.key) is False
        failure = json.loads(paths.failures.read_text().splitlines()[0])
        assert failure["key"] == job.key
        assert failure["error"] == "no session banner"

    def test_the_verdict_comes_from_the_summary_not_the_exit_code(self, tmp_path, monkeypatch):
        """§7.4's request-H contract: a nonzero exit can still be a recorded verdict."""
        monkeypatch.setattr(
            runner_mod,
            "invoke_vanta",
            lambda **kwargs: _outcome(tmp_path, exit_code=101, decision="AllTestsPassed"),
        )
        monkeypatch.setattr(
            oracle_mod,
            "grade",
            lambda **kwargs: oracle_mod.OracleResult(apply_ok=True, patch_good=True),
        )
        record = runner_mod.run_job(
            job=_job(prep(root=tmp_path / "root")), paths=self._paths(tmp_path),
            manifest=manifest_mod.Manifest(tmp_path), vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4, disk_mb=8192, python_version="3.8.20",
        )
        assert record["exit_code"] == 101
        assert record["verdict"] == "AllTestsPassed"
        assert record["label"] == "true_approval"


class _FakeSplit:
    """A `Split` with only the field `run_batch` reads, so no split file is needed."""

    def __init__(self, configs):
        self.configs = list(configs)


class TestRunBatchLoop:
    """§7.1's phase loop: the schedule, not the work.

    `run_job` is replaced, so no VM boots and nothing is written outside `tmp_path`.
    What is under test is the order the phases run in, that both configs review one
    candidate set, and that `state.json` — not the file system — decides what a
    resumed run repeats.
    """

    DEV_ONE = _diff("pkg/mod.py", "developer-fix-one")
    DEV_TWO = _diff("pkg/mod.py", "developer-fix-two")
    FAILED_ONE = _diff("pkg/one.py", "baseline-attempt")
    FAILED_TWO = _diff("pkg/two.py", "vanta-attempt")
    GOOD = _diff("pkg/three.py", "a-real-fix")

    # -- fixture -----------------------------------------------------------
    @staticmethod
    def _tree(tmp_path: Path, name: str) -> Path:
        root = tmp_path / name
        write(root / "vanta_tree" / "pkg" / "mod.py", f"{name} = 1\nsecond = 2\nthird = 3\n")
        return root

    def _fixture(self, tmp_path, *, configs=("baseline", "vanta")):
        preps = [
            prep(
                bug=bug_info(bug_id=1, bug_patch_txt=self.DEV_ONE),
                root=self._tree(tmp_path, "one"),
                wave=1,
            ),
            prep(
                bug=bug_info(bug_id=2, bug_patch_txt=self.DEV_TWO),
                root=self._tree(tmp_path, "two"),
                wave=1,
            ),
        ]
        entries = {
            bug_prep.key: {
                "checkpoint": f"vanta-bi-{bug_prep.slug.lower()}",
                "python_version": "3.8.1",
            }
            for bug_prep in preps
        }
        return preps, entries, _FakeSplit(configs)

    # -- the stub ----------------------------------------------------------
    def _stub(self, tmp_path, calls, *, leave_running=None):
        """A `run_job` that writes a gate-passing row and records the schedule."""

        def stub(*, job, paths, manifest, vanta_bin, max_attempts, disk_mb, python_version):
            calls.append(job)
            record = self._row(tmp_path, job)
            manifest.append(record)
            stuck = leave_running is not None and leave_running(job)
            manifest.mark(job.key, manifest_mod.RUNNING if stuck else manifest_mod.DONE)
            return record

        return stub

    def _repair_result(self, job) -> tuple[str, bool]:
        """Bug 1's two repairs both fail — so they pool; bug 2's both succeed."""
        if job.bug.endswith("/1"):
            text = self.FAILED_ONE if job.config == "baseline" else self.FAILED_TWO
            return text, False
        return self.GOOD, True

    def _row(self, tmp_path: Path, job) -> dict:
        patch = None
        candidate = None
        if job.mode == runner_mod.REPAIR:
            text, good = self._repair_result(job)
            patch = write(
                tmp_path / "rows" / f"{job.bug.replace('/', '_')}-{job.config}.patch", text
            )
        else:
            candidate = job.candidate
            good = candidate.kind == runner_mod.DEV_FIX
        row = {
            "schema": manifest_mod.SCHEMA_VERSION,
            "key": job.key,
            "batch": job.batch_id,
            "bug": job.bug,
            "wave": job.wave,
            "config": job.config,
            "mode": job.mode,
            "seed": job.seed,
            "checkpoint": job.checkpoint,
            "session_id": f"{job.bug.replace('/', '_')}-{job.config}-{job.mode}",
            "verdict": "Rejected",
            "label": "false_reject" if good else "true_reject",
            "oracle": {"apply_ok": True, "patch_good": good, "apply_detail": "", "error": ""},
            "candidate": None,
            "artifacts": {
                "final_patch": str(patch) if patch else None,
                "session_dir": None,
                "oracle_json": None,
            },
        }
        if candidate is not None:
            row["candidate"] = {
                "kind": candidate.kind,
                "source": candidate.source,
                "patch_sha256": candidate.patch_sha256,
                "normalized_sha256": candidate.normalized_sha256,
                "oracle": {"apply_ok": True, "patch_good": good},
            }
        return row

    # -- driving it --------------------------------------------------------
    def _prepare(self, monkeypatch, tmp_path, *, configs=("baseline", "vanta"), running=None):
        preps, entries, split = self._fixture(tmp_path, configs=configs)
        calls: list = []
        monkeypatch.setattr(runner_mod, "python_version_of", lambda *a, **k: "3.8.20")
        monkeypatch.setattr(runner_mod, "grade_candidate", lambda **kwargs: kwargs["candidate"])
        monkeypatch.setattr(runner_mod, "run_job", self._stub(tmp_path, calls, leave_running=running))
        return preps, entries, split, calls, tmp_path / "batch"

    def _run(self, preps, entries, split, batch_dir):
        return runner_mod.run_batch(
            batch_dir=batch_dir,
            batch_id="b1",
            split=split,
            preps=preps,
            entries=entries,
            seeds=[0],
            waves=[1],
            vanta_bin=Path("/nonexistent/vanta_m3"),
            max_attempts=4,
            disk_mb=8192,
        )

    @staticmethod
    def _reviews(calls):
        return [job for job in calls if job.mode == runner_mod.REVIEW]

    def test_every_job_carries_the_batch_s_harness_revision(self, tmp_path, monkeypatch):
        """Read once, at the start: rows of one batch must not disagree about which
        harness produced them (§4.4)."""
        monkeypatch.setattr(
            runner_mod, "git_revision", lambda *a, **k: runner_mod.Revision("h" * 40, False)
        )
        preps, entries, split, calls, batch_dir = self._prepare(monkeypatch, tmp_path)
        self._run(preps, entries, split, batch_dir)
        assert calls, "the stub should have been called"
        assert {job.harness for job in calls} == {runner_mod.Revision("h" * 40, False)}

    def test_the_harness_revision_is_read_before_the_first_phase_runs(
        self, tmp_path, monkeypatch
    ):
        """A `git` call per row would let a mid-batch edit relabel the earlier rows."""
        seen: list[int] = []
        monkeypatch.setattr(
            runner_mod,
            "git_revision",
            lambda *a, **k: (seen.append(1), runner_mod.Revision("h" * 40, False))[1],
        )
        preps, entries, split, calls, batch_dir = self._prepare(monkeypatch, tmp_path)
        self._run(preps, entries, split, batch_dir)
        assert len(seen) == 1

    def test_every_repair_pass_runs_before_any_review_pass(self, tmp_path, monkeypatch):
        preps, entries, split, calls, batch_dir = self._prepare(monkeypatch, tmp_path)
        code, _ = self._run(preps, entries, split, batch_dir)
        assert code == 0, "the stub rows should pass the wave gate"
        modes = [job.mode for job in calls]
        assert modes[:4] == [runner_mod.REPAIR] * 4
        assert modes.count(runner_mod.REPAIR) == 4, "2 bugs x 2 configs"
        assert modes == [runner_mod.REPAIR] * 4 + [runner_mod.REVIEW] * 12

    def test_each_review_set_is_the_dev_fix_the_noop_and_the_pooled_repairs(
        self, tmp_path, monkeypatch
    ):
        preps, entries, split, calls, batch_dir = self._prepare(monkeypatch, tmp_path)
        self._run(preps, entries, split, batch_dir)
        kinds: dict[tuple, list] = {}
        for job in self._reviews(calls):
            kinds.setdefault((job.bug, job.config), []).append(job.candidate.kind)
        assert kinds[("Demo/1", "baseline")] == [
            runner_mod.DEV_FIX, runner_mod.NOOP, runner_mod.FAILED_PATCH, runner_mod.FAILED_PATCH
        ]
        assert kinds[("Demo/2", "baseline")] == [runner_mod.DEV_FIX, runner_mod.NOOP]
        pooled = {
            job.candidate.patch_sha256
            for job in self._reviews(calls)
            if job.candidate.kind == runner_mod.FAILED_PATCH
        }
        assert pooled == {
            split_mod.sha256_text(self.FAILED_ONE), split_mod.sha256_text(self.FAILED_TWO)
        }, "the pooled patches are the repair stubs' own bytes, deduplicated"
        assert all(
            job.candidate.source.startswith("repair:")
            for job in self._reviews(calls)
            if job.candidate.kind == runner_mod.FAILED_PATCH
        )

    def test_both_configs_review_the_same_candidate_set_at_the_same_seed(
        self, tmp_path, monkeypatch
    ):
        preps, entries, split, calls, batch_dir = self._prepare(monkeypatch, tmp_path)
        self._run(preps, entries, split, batch_dir)
        by_bug: dict[str, dict[str, set]] = {}
        for job in self._reviews(calls):
            identity = f"{job.candidate.kind}:{job.candidate.patch_sha256}"
            by_bug.setdefault(job.bug, {}).setdefault(job.config, set()).add(identity)
        for bug in ("Demo/1", "Demo/2"):
            assert by_bug[bug]["baseline"] == by_bug[bug]["vanta"], (
                "an asymmetric set would silently halve the comparison"
            )
        reviews = self._reviews(calls)
        baseline_keys = {job.key for job in reviews if job.config == "baseline"}
        vanta_keys = {job.key for job in reviews if job.config == "vanta"}
        assert baseline_keys and vanta_keys
        assert {
            key.replace("#baseline#", "#vanta#") for key in baseline_keys
        } == vanta_keys, "the paired keys differ in the config component only"

    def test_a_resumed_run_skips_done_rows_and_repeats_the_running_one(
        self, tmp_path, monkeypatch
    ):
        def mid_flight(job) -> bool:
            return (
                job.mode == runner_mod.REVIEW
                and job.bug.endswith("/1")
                and job.config == "vanta"
                and job.candidate.kind == runner_mod.NOOP
            )

        preps, entries, split, first, batch_dir = self._prepare(
            monkeypatch, tmp_path, running=mid_flight
        )
        code, _ = self._run(preps, entries, split, batch_dir)
        assert code == 0
        stuck = [job for job in first if mid_flight(job)]
        assert len(stuck) == 1
        manifest = manifest_mod.Manifest(batch_dir)
        assert manifest.state()[stuck[0].key] == manifest_mod.RUNNING
        assert len(manifest.records()) == len(first)

        # A second `run_batch` over the same manifest: the same work, resumed.
        preps, entries, split, second, same_batch_dir = self._prepare(monkeypatch, tmp_path)
        assert same_batch_dir == batch_dir
        code, _ = self._run(preps, entries, split, batch_dir)
        assert code == 0
        assert [job.key for job in second] == [stuck[0].key], (
            "only the key left `running` may be repeated"
        )
        manifest = manifest_mod.Manifest(batch_dir)
        assert manifest.state()[stuck[0].key] == manifest_mod.DONE
        assert all(manifest.is_done(key) for key in {job.key for job in first})
        assert len(manifest.records()) == len(first) + 1, (
            "the re-run appends a second row for the same key; the evidence stays append-only"
        )

    def test_a_split_that_names_the_planner_placeholder_is_refused_before_any_work(
        self, tmp_path, monkeypatch
    ):
        preps, entries, split, calls, batch_dir = self._prepare(
            monkeypatch, tmp_path, configs=("baseline", "vanta", "vanta_planner")
        )
        with pytest.raises(runner_mod.RunnerError, match="vanta_planner is a placeholder"):
            self._run(preps, entries, split, batch_dir)
        assert calls == [], "the guard must fire before the first job"
        assert not (batch_dir / manifest_mod.MANIFEST_NAME).exists()


class TestPythonVersionProbe:
    """The interpreter probe is one boot per checkpoint, cached for every later row."""

    def test_a_cached_probe_is_used_without_booting_a_vm(self, tmp_path, monkeypatch):
        paths = runner_mod.BatchPaths.under(tmp_path)
        write(paths.env_cache / "chk.json", json.dumps({"python_version": "3.8.20"}))
        monkeypatch.setattr(
            shuru_mod, "run", lambda **kwargs: pytest.fail("the checkpoint was re-probed")
        )
        assert runner_mod.python_version_of(paths, "chk", 8192) == "3.8.20"

    def test_a_probe_reads_the_version_and_caches_it(self, tmp_path, monkeypatch):
        paths = runner_mod.BatchPaths.under(tmp_path)

        class FakeResult:
            stdout = "Python 3.8.20"
            stderr = ""
            exit_code = 0
            timed_out = False

        monkeypatch.setattr(shuru_mod, "run", lambda **kwargs: FakeResult())
        assert runner_mod.python_version_of(paths, "chk", 8192) == "3.8.20"
        cached = json.loads((paths.env_cache / "chk.json").read_text())
        assert cached["checkpoint"] == "chk"
        assert cached["python_version"] == "3.8.20"


class TestCompaction:
    """§6.3 + D6: hold a rejecting session's snapshot until its rerun is recorded,
    and prove a compacted one replays byte-identically from base + diff.

    Everything lives under `tmp_path` and no VM boots. What these tests refuse to
    take on trust is the replay: they read the diff and the hashes back **off disk**
    and rebuild the tree from them, so a compaction that deleted the only copy
    without leaving a replayable artifact fails here rather than at the first
    `--rerun-signals` run against a real batch — where the failure would look like
    "the signal did not reproduce", i.e. a fabricated `spurious_reject`.
    """

    @staticmethod
    def _base(root: Path) -> Path:
        """The sanitized base: one file to edit, one to delete, one untouched."""
        base = root / "vanta_clean"
        write(base / "pkg" / "mod.py", "one = 1\nsecond = 2\n")
        write(base / "pkg" / "gone.py", "third = 3\n")
        write(base / "README.md", "hello\n")
        return base

    @staticmethod
    def _snapshot(session_dir: Path, name: str) -> Path:
        """A snapshot that edits, adds and deletes a file — every header shape the
        diff rewriter has to strip, including the two where git names both sides of
        the `diff --git` line after the same tree."""
        tree = session_dir / "debug" / name
        write(tree / "pkg" / "mod.py", "one = 1\nsecond = 22\nthird = 3\n")
        write(tree / "pkg" / "added.py", "four = 4\n")
        write(tree / "README.md", "hello\n")
        return tree

    @staticmethod
    def _write_rerun(sessions: Path, session_id: str, text: str) -> Path:
        return write(sessions / f"{session_id}_signal_reruns.json", text)

    # -- the rule ----------------------------------------------------------
    def test_a_rejecting_session_is_held_until_its_rerun_is_recorded(self, tmp_path):
        base = self._base(tmp_path)
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        tree = self._snapshot(session_dir, "attempt_1_graded_tree")

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=base,
            verdict="Rejected",
            session_id="sid",
            sessions_dir=sessions,
        )

        assert result.state == compact_mod.FULL
        assert "rerun" in result.reason
        assert result.diff is None
        assert result.trees == ["attempt_1_graded_tree"]
        assert tree.is_dir(), "the snapshot the rerun reads must still be there"
        assert not (tree.parent / f"{tree.name}.diff").exists()

    def test_a_rejected_session_compacts_once_the_rerun_artifact_exists(self, tmp_path):
        base = self._base(tmp_path)
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        tree = self._snapshot(session_dir, "attempt_1_graded_tree")
        self._write_rerun(
            sessions, "sid", json.dumps({"signals": [], "final_decision": "Rejected"})
        )

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=base,
            verdict="Rejected",
            session_id="sid",
            sessions_dir=sessions,
        )

        assert result.state == compact_mod.COMPACTED
        assert "rerun recorded" in result.reason
        assert not tree.exists()

    @pytest.mark.parametrize("verdict", ["AllTestsPassed", "MaxAttemptsReached", ""])
    def test_only_a_rejection_holds_a_snapshot(self, tmp_path, verdict):
        """A verdict with no rejection to explain has nothing a rerun could settle."""
        base = self._base(tmp_path)
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        tree = self._snapshot(session_dir, "attempt_1_graded_tree")

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=base,
            verdict=verdict,
            session_id="sid",
            sessions_dir=sessions,
        )

        assert compact_mod.has_rejecting_signal(verdict) is False
        assert result.state == compact_mod.COMPACTED
        assert not tree.exists()

    @pytest.mark.parametrize(
        "text",
        [
            '{"signals": [',  # a truncated write
            '{"signals": [], "final_decision": ""}',  # no decision
            '{"final_decision": "Rejected"}',  # no signals list
            "not json at all",
        ],
    )
    def test_a_truncated_rerun_artifact_does_not_count_as_recorded(self, tmp_path, text):
        """Half a rerun must not license deleting the tree it would have needed."""
        sessions = tmp_path / "vanta_sessions"
        self._write_rerun(sessions, "sid", text)

        assert compact_mod.rerun_recorded(sessions, "sid") is False
        state, _ = compact_mod.compaction_plan(
            verdict="Rejected", session_id="sid", sessions_dir=sessions
        )
        assert state == compact_mod.FULL

    def test_a_session_that_wrote_no_snapshot_reports_none(self, tmp_path):
        """`none` is a third state, not `full`: "never any" vs "still here" differ."""
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        (session_dir / "debug").mkdir(parents=True)

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=tmp_path / "missing",
            verdict="Rejected",
            session_id="sid",
            sessions_dir=sessions,
        )

        assert result.state == compact_mod.NONE
        assert result.diff is None
        assert list((session_dir / "debug").iterdir()) == []

    def test_compacting_without_a_base_tree_raises(self, tmp_path):
        """No base means no diff, and a diff-less deletion is unrecoverable."""
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        self._snapshot(session_dir, "attempt_1_graded_tree")

        with pytest.raises(compact_mod.CompactError, match="diff against"):
            compact_mod.compact_session(
                session_dir=session_dir,
                base_tree=tmp_path / "missing",
                verdict="AllTestsPassed",
                session_id="sid",
                sessions_dir=sessions,
            )

    # -- the replay --------------------------------------------------------
    def test_a_compacted_tree_replays_byte_identically_from_base_plus_diff(self, tmp_path):
        """§6.3's replayability proof, read back off disk rather than trusted.

        The test keeps the pre-deletion concat and hashes only as *expected* values;
        what it replays from is the diff and the artifact compaction actually wrote.
        """
        base = self._base(tmp_path)
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        tree = self._snapshot(session_dir, "attempt_1_graded_tree")
        expected_concat = compact_mod.tree_concat(tree)
        expected_hashes = compact_mod.tree_manifest(tree)

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=base,
            verdict="AllTestsPassed",
            session_id="sid",
            sessions_dir=sessions,
        )

        assert result.state == compact_mod.COMPACTED
        assert result.diff == "debug/attempt_1_graded_tree.diff"
        assert not tree.exists(), "the directory is replaced by the diff"

        artifact = json.loads((tree.parent / f"{tree.name}.hashes.json").read_text())
        assert artifact["file_count"] == len(expected_hashes)
        assert artifact["files"] == expected_hashes
        assert artifact["concat_sha256"] == split_mod.sha256_text(expected_concat)

        diff_text = (tree.parent / artifact["diff"]).read_text()
        assert artifact["diff_sha256"] == split_mod.sha256_text(diff_text)

        rebuilt = compact_mod.reconstruct(base, diff_text, tmp_path / "rebuilt")
        assert compact_mod.tree_concat(rebuilt) == expected_concat
        assert compact_mod.tree_manifest(rebuilt) == expected_hashes
        assert (
            split_mod.sha256_text(compact_mod.tree_concat(rebuilt))
            == artifact["concat_sha256"]
        )

    def test_reconstruct_refuses_a_diff_that_does_not_apply(self, tmp_path):
        """The replay is the evidence, so a bogus diff has to fail loudly."""
        base = self._base(tmp_path)

        with pytest.raises(compact_mod.CompactError, match="failed"):
            compact_mod.reconstruct(base, "not a patch\n", tmp_path / "rebuilt")

    def test_both_snapshots_are_diffed_before_they_are_deleted(self, tmp_path):
        """The graded tree is what the rerun reads; the patched one is its fallback."""
        base = self._base(tmp_path)
        sessions = tmp_path / "vanta_sessions"
        session_dir = sessions / "sid"
        self._snapshot(session_dir, "attempt_1_graded_tree")
        self._snapshot(session_dir, "attempt_1_patched_tree")

        result = compact_mod.compact_session(
            session_dir=session_dir,
            base_tree=base,
            verdict="AllTestsPassed",
            session_id="sid",
            sessions_dir=sessions,
        )

        assert result.state == compact_mod.COMPACTED
        assert result.trees == ["attempt_1_graded_tree", "attempt_1_patched_tree"]
        assert result.diff == "debug/attempt_1_graded_tree.diff", (
            "the row must point at the snapshot the rerun reads"
        )
        for name in result.trees:
            assert (session_dir / "debug" / f"{name}.diff").is_file()
            assert (session_dir / "debug" / f"{name}.hashes.json").is_file()
            assert not (session_dir / "debug" / name).exists()

    def test_the_hashes_and_the_concat_ignore_what_the_rerun_ignores(self, tmp_path):
        """Same filter as `rerun.rs::collect_tree_files`, or the evidence describes
        files nobody reads."""
        tree = tmp_path / "tree"
        write(tree / "pkg" / "mod.py", "one = 1\n")
        write(tree / "pkg" / "__pycache__" / "mod.cpython-38.pyc", "cached\n")
        write(tree / "pkg" / "stale.pyc", "cached\n")
        write(tree / "pkg" / "stale.pyo", "cached\n")
        write(tree / "pkg" / "mod_manifest.json", '{"paths": []}')
        write(tree / ".git" / "HEAD", "ref: refs/heads/main\n")

        assert compact_mod.tree_manifest(tree) == {
            "pkg/mod.py": split_mod.sha256_text("one = 1\n")
        }
        assert compact_mod.tree_concat(tree) == "# FILE: pkg/mod.py\none = 1\n"


class TestRunJobCompaction:
    """`run_job` applies §6.3 **before** it writes the row (§7.3's `artifacts`).

    A row that claimed `full` about a snapshot compaction had just deleted would
    misdescribe the session the manifest is the evidence for, so the state has to
    come from the compaction that actually ran.
    """

    def _fixture(self, tmp_path: Path, *, session_id: str = "sess-1") -> tuple[Path, Path]:
        base = tmp_path / "root" / "vanta_clean"
        write(base / "pkg" / "mod.py", "one = 1\n")
        sessions = tmp_path / "vanta_sessions"
        tree = sessions / session_id / "debug" / "attempt_1_graded_tree"
        write(tree / "pkg" / "mod.py", "one = 2\n")
        return sessions, tree

    def _run(self, tmp_path, monkeypatch, *, decision: str, sessions: Path) -> dict:
        monkeypatch.setattr(runner_mod, "sessions_dir", lambda: sessions)
        monkeypatch.setattr(
            runner_mod,
            "invoke_vanta",
            lambda **kwargs: _outcome(
                tmp_path, decision=decision, final_patch=tmp_path / "final.patch"
            ),
        )
        monkeypatch.setattr(
            oracle_mod,
            "grade",
            lambda **kwargs: oracle_mod.OracleResult(apply_ok=True, patch_good=True),
        )
        return runner_mod.run_job(
            job=_job(prep(root=tmp_path / "root")),
            paths=runner_mod.BatchPaths.under(tmp_path),
            manifest=manifest_mod.Manifest(tmp_path),
            vanta_bin=tmp_path / "vanta_m3",
            max_attempts=4,
            disk_mb=8192,
            python_version="3.8.20",
        )

    def test_a_compacted_session_is_reported_as_compacted_in_the_row(
        self, tmp_path, monkeypatch
    ):
        sessions, tree = self._fixture(tmp_path)

        record = self._run(
            tmp_path, monkeypatch, decision="AllTestsPassed", sessions=sessions
        )

        assert record["artifacts"]["compact_state"] == compact_mod.COMPACTED
        assert record["artifacts"]["graded_tree_diff"] == "debug/attempt_1_graded_tree.diff"
        assert not tree.exists()
        assert (sessions / "sess-1" / "debug" / "attempt_1_graded_tree.diff").is_file()

    def test_a_rejected_session_keeps_its_snapshot_and_the_row_says_full(
        self, tmp_path, monkeypatch
    ):
        sessions, tree = self._fixture(tmp_path)

        record = self._run(tmp_path, monkeypatch, decision="Rejected", sessions=sessions)

        assert record["artifacts"]["compact_state"] == compact_mod.FULL
        assert record["artifacts"]["graded_tree_diff"] is None
        assert "rerun" in record["artifacts"]["compact_reason"]
        assert tree.is_dir(), "§6.3: the rerun has not run yet"


class TestExpectedErrorsExitCleanly:
    """A refusal is a message and an exit code; only a harness bug is a traceback.

    Every error class in `EXPECTED_ERRORS` already spells out its own report, so a
    traceback buries the sentence that matters under frames the reader did not ask
    for. `ContextError` was missing from the list, so the context gate's refusals —
    the ones that tell you which window to export — arrived as a stack dump.
    """

    def test_a_context_error_is_a_line_not_a_traceback(self, tmp_path, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise ctx.ContextError("no explicit window source: export VANTA_LLM_CONTEXT_TOKENS")

        monkeypatch.setattr(runner_mod, "require_repo_cwd", lambda: REPO_ROOT)
        monkeypatch.setattr(run_mod, "load_split", boom)

        code = run_mod.main(["run", "--batch", str(tmp_path)])

        captured = capsys.readouterr()
        assert code == run_mod.EXIT_HARNESS_FAILED
        assert captured.err.startswith("error: no explicit window source")
        assert "Traceback" not in captured.err

    def test_the_harness_error_vocabulary_is_all_caught(self):
        """A class raised on purpose but missing here exits as an unhandled
        exception — which is indistinguishable from a crash in the harness."""
        assert set(run_mod.EXPECTED_ERRORS) >= {
            split_mod.SplitError,
            bake_mod.BakeError,
            prepare_mod.PrepareError,
            meta_mod.MetadataError,
            shuru_mod.ShuruError,
            runner_mod.RunnerError,
            ctx.ContextError,
            oracle_mod.OracleError,
            suite_mod.SuiteError,
            tree_mod.TreeError,
            tree_mod.StripError,
            issue_mod.IssueError,
            compact_mod.CompactError,
            records_mod.RecordError,
        }
