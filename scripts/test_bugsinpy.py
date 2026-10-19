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
from bugsinpy import context as ctx  # noqa: E402
from bugsinpy import issue as issue_mod  # noqa: E402
from bugsinpy import oracle as oracle_mod  # noqa: E402
from bugsinpy import prepare as prepare_mod  # noqa: E402
from bugsinpy import shuru as shuru_mod  # noqa: E402
from bugsinpy import split as split_mod  # noqa: E402
from bugsinpy import suite as suite_mod  # noqa: E402
from bugsinpy import trees as tree_mod  # noqa: E402
from bugsinpy.split import SplitError, load_split  # noqa: E402

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
        sampling={"policy": "server_defaults", "values_sent": None},
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
        assert split.sampling["policy"] == "server_defaults"

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

    def test_sampling_policy_must_be_server_defaults(self):
        """The harness never sets temperature/top_p/max_tokens (§4.3)."""
        with pytest.raises(SplitError, match="server_defaults"):
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

    def test_estimate_tokens(self):
        assert ctx.estimate_tokens(int(ctx.BYTES_PER_TOKEN_ESTIMATE * 360)) == 360
        assert ctx.estimate_tokens(0) == 0


class TestPrecheck:
    def test_accepts_a_tree_that_fits(self, tmp_path):
        write(tmp_path / "a.py", "x = 1\n")
        verdict = ctx.precheck(tmp_path, "hello", n_ctx=65536, n_ctx_source="test")
        assert verdict.context_ok is True
        assert verdict.reasons == []
        assert verdict.budget_tokens == int(65536 * (1 - ctx.CONTEXT_MARGIN_FRACTION))

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
        write(tmp_path / "a.py", "x" * 3000 + "\n")
        tight = ctx.precheck(tmp_path, "", n_ctx=4000, n_ctx_source="test")
        roomy = ctx.precheck(tmp_path, "", n_ctx=8000, n_ctx_source="test")
        assert tight.context_ok is False
        assert roomy.context_ok is True

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
