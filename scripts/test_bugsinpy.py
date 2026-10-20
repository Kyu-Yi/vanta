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
            "PySnooper/1",
            "PySnooper/2",
            "PySnooper/3",
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
            assert bug.test_file in tree_mod.test_paths_from_fix(bug)


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
    def test_unions_patch_test_paths_with_bug_info(self):
        """Union, not intersection: either source naming a path includes it."""
        bug = bug_info(
            test_file="tests/test_untouched.py",
            bug_patch_txt=TestParsePatch.PATCH,
        )
        assert tree_mod.test_paths_from_fix(bug) == ["tests/test_untouched.py"]

    def test_patch_that_touches_a_test_file(self):
        patch = (
            "diff --git a/tests/test_a.py b/tests/test_a.py\n"
            "--- a/tests/test_a.py\n"
            "+++ b/tests/test_a.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        bug = bug_info(test_file="tests/test_b.py", bug_patch_txt=patch)
        assert tree_mod.test_paths_from_fix(bug) == ["tests/test_a.py", "tests/test_b.py"]


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
        assert verdict.as_dict()["headroom_tokens"] == verdict.headroom_tokens

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
        monkeypatch.setattr(
            prepare_mod, "checkpoint_exists", lambda name: name == "vanta-bugsinpy-py3.8"
        )
        with pytest.raises(prepare_mod.PrepareError) as excinfo:
            prepare_mod.require_checkpoints(
                ["vanta-bugsinpy-py3.8", "vanta-bi-thefuck-1", "vanta-bi-httpie-1"]
            )
        message = str(excinfo.value)
        assert "vanta-bi-httpie-1" in message
        assert "vanta-bi-thefuck-1" in message
        assert "vanta-bugsinpy-py3.8" not in message

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
        assert "sys.prefix != sys.base_prefix" in bake_mod.requirements_recipe("", [])

    def test_recipes_do_not_reinstall_pip(self):
        """``uv venv --seed`` already provides a current pip.

        Measured: against a bare interpreter, ``pip install --upgrade pip`` is the
        line that dies on PEP 668 — it was the wave-1 bake's first failure.
        """
        for recipe in (self._recipe(), bake_mod.requirements_recipe("", ["decorator"])):
            assert "--upgrade pip" not in recipe

    def test_marker_is_touched_last(self):
        for recipe in (self._recipe(), bake_mod.requirements_recipe("", ["decorator"])):
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

    def test_requirements_recipe_installs_both_sources_and_never_swallows_errors(self):
        recipe = bake_mod.requirements_recipe("requests==2.0\n", ["decorator"])
        assert "decorator" in recipe
        assert "requests==2.0" in recipe
        assert "|| true" not in recipe
        assert "VANTA_REQUIREMENTS" in recipe

    def test_requirements_recipe_without_deps_still_asserts_the_interpreter(self):
        recipe = bake_mod.requirements_recipe("", [])
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

    @needs_clone
    def test_wave_1_layers_are_one_interpreter_and_three_bug_layers(self):
        split = load_split(SPLIT_PATH)
        minors, entries = bake_mod.select_layers(
            split, pythons=["3.8"], only=[], roles={"pilot"}
        )
        assert minors == ["3.8"]
        assert [entry["project"] for entry in entries] == ["PySnooper"] * 3
        layers = bake_mod.layer_recipes(
            split, pythons=minors, entries=entries, bugsinpy_root=BUGSINPY_CLONE
        )
        assert [name for name, _, _ in layers] == [
            "vanta-bugsinpy-py3.8",
            "vanta-bi-PySnooper-1",
            "vanta-bi-PySnooper-2",
            "vanta-bi-PySnooper-3",
        ]
        assert layers[0][1] == bake_mod.DEFAULT_BASE
        assert all(base == "vanta-bugsinpy-py3.8" for _, base, _ in layers[1:])
        assert "python_toolbox" in layers[2][2]

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
            "model": "qwen3.5-27b", "vanta_commit": "d" * 40, "vanta_dirty": False,
            "label": "true_reject", "session_id": "sess-1", "candidate": None,
        },
        {
            "bug": "Demo/1", "config": "vanta", "mode": "review", "seed": 0,
            "model": "qwen3.5-27b", "vanta_commit": "d" * 40, "vanta_dirty": False,
            "label": "false_reject", "session_id": "sess-2",
            "candidate": {
                "kind": "dev_fix", "source": "dev",
                "oracle": {"apply_ok": True, "patch_good": True},
            },
        },
        {
            "bug": "Demo/1", "config": "baseline", "mode": "review", "seed": 0,
            "model": "qwen3.5-27b", "vanta_commit": "d" * 40, "vanta_dirty": False,
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
        # config, mode, seed, model and commit are the axes: a repair row and a
        # review row of the same config are two groups, not one.
        assert len(payload["groups"]) == 3
        vanta_review = [
            g for g in payload["groups"] if g["config"] == "vanta" and g["mode"] == "review"
        ]
        assert len(vanta_review) == 1
        assert vanta_review[0]["runs"] == 1
        assert vanta_review[0]["labels"] == {"false_reject": 1}
        assert vanta_review[0]["model"] == "qwen3.5-27b"
        assert vanta_review[0]["vanta_dirty"] is False

    def test_the_model_is_one_of_the_grouping_axes(self):
        other = dict(self.ROWS[0], model="gemma-3-12b")
        assert len(runner_mod.digest(self.ROWS + [other])["groups"]) == 4

    def test_two_commits_do_not_share_a_group(self):
        other = dict(self.ROWS[0], vanta_commit="e" * 40)
        assert len(runner_mod.digest(self.ROWS + [other])["groups"]) == 4

    def test_the_rendered_digest_has_a_header_and_one_line_per_group(self):
        rendered = runner_mod.render_digest(runner_mod.digest(self.ROWS))
        lines = rendered.splitlines()
        assert "config" in lines[0] and "labels" in lines[0]
        assert len(lines) == 2 + len(runner_mod.digest(self.ROWS)["groups"])

    def test_write_batch_report_writes_all_three_artifacts(self, tmp_path):
        labels_path, payload = runner_mod.write_batch_report(tmp_path, self.ROWS)
        assert labels_path == tmp_path / "run_labels.json"
        assert payload["runs"] == 3
        assert json.loads((tmp_path / "run_digest.json").read_text()) == payload
        assert (tmp_path / "run_digest.md").read_text().startswith("config")


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


def _job(bug_prep, *, config="vanta", mode=runner_mod.REPAIR, candidate=None) -> runner_mod.Job:
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
