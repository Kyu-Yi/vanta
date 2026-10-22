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
import inspect
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import replace
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
DEV_SPLIT_PATH = REPO_ROOT / "plans" / "bugsinpy_dev_split.json"
BUGSINPY_CLONE = REPO_ROOT / "runs" / "bugsinpy" / "bugsinpy"

needs_clone = pytest.mark.skipif(
    not BUGSINPY_CLONE.is_dir(),
    reason="the BugsInPy clone is not present (runs/ is not committed)",
)

PROJECT_CLONES = REPO_ROOT / "runs" / "bugsinpy" / "repos"
needs_project_clones = pytest.mark.skipif(
    not (PROJECT_CLONES / "matplotlib.git").is_dir(),
    reason="the project clones are not present (runs/ is not committed)",
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
        checkpoint="vanta-bi-Demo-1",
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


def _heredoc(recipe: str, tag: str) -> str:
    """The body of one ``cat > … <<'TAG' … TAG`` block in a layer recipe."""
    return recipe.split(f"<<'{tag}'\n", 1)[1].split(f"\n{tag}\n", 1)[0]


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
            # PySnooper/2 is deliberately absent: it was dropped for a
            # `bug_patch.txt` that cannot pass the oracle (reviewed_changes[3]).
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

    def test_real_split_reviewed_changes_carry_the_rule_and_its_measurements(self):
        """§8's substitutions are where the split departs from §8's own plan.

        Pinned so an amendment cannot be edited away or have its evidence thinned:
        the rule that chose each substitute, the exclusion it substitutes for, and a
        measurement for every candidate the rule walked are all in the file a
        reviewer reads.
        """
        split = load_split(SPLIT_PATH)
        changes = split.data["reviewed_changes"]
        assert len(changes) == 6
        excluded, promoted, removed, dropped, development, mapped = changes
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

        # The third entry retires that promotion: black/22 is out of `bugs`, so wave
        # 1 is the two PySnooper bugs. Its evidence is the mechanical per-pin count
        # in the rule text, not a per-candidate measurement table — the exclusion is
        # a property of BugsInPy's own metadata, not a decision about the bug.
        assert "black/22 is removed from `bugs`" in removed["change"]
        assert "pip freeze" in removed["reason"]
        assert "aarch64" in removed["rule"]
        assert "measurements" not in removed

        # The fourth drops PySnooper/2, which is what makes wave 1 a single pilot
        # bug. Its evidence is the oracle's own grading of the developer fix (the
        # bug's `bug_patch.txt` cannot be imported), quoted from the batch it was
        # measured in.
        assert "PySnooper/2 is removed from `bugs`" in dropped["change"]
        assert "prepare-wave1-3" in dropped["reason"]
        assert "measurements" not in dropped

        # The fifth is not a protocol change but a reporting one: PySnooper/3 stays
        # in `bugs` and its wave still runs, and the record is what keeps its rows
        # out of the pooled results (§14.7 — §14's items 1 and 2 were designed from
        # its false reject, so it is development data).
        assert "PySnooper/3 is development data" in development["change"]
        assert "seen during development" in development["change"]
        assert "no config, seed, sampling or gate field is touched" in development["change"]
        assert "development case" in development["reason"]
        assert "measurements" not in development

        # The sixth is the reviewed module mapping: `psycopg2` is provided by two
        # distributions, so the closure refuses to choose and the pick is recorded
        # rather than guessed. Its evidence is the rule, not a per-candidate table.
        assert "psycopg2" in mapped["change"]
        assert "psycopg2-binary" in mapped["change"]
        assert "skip without a database server" in mapped["reason"]
        assert "date rule" in mapped["rule"]
        assert "measurements" not in mapped

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
        # Wave 1 is PySnooper/3 alone: PySnooper/2 was dropped and no substitute
        # was searched for (reviewed_changes[3]).
        assert len(split.bugs_in_wave(1)) == 1

    def test_dev_split_is_disjoint_from_the_evaluation_waves(self):
        """§14.7: items 1-3 are developed on bugs no evaluation number comes from.

        The development split is a second file precisely so that the choice of
        development data does not rewrite the evaluation split's bugs, waves or
        sampling. What has to hold is that the two sets meet at one bug only: the
        one §14 was designed from — PySnooper/3 — which the dev split carries as
        "seen during development" and the evaluation split excludes at reporting
        time through ``reviewed_changes``.
        """
        dev = load_split(DEV_SPLIT_PATH)
        evaluation = load_split(SPLIT_PATH)

        dev_keys = {f"{e['project']}/{e['bug_id']}" for e in dev.bugs}
        evaluation_keys = {f"{e['project']}/{e['bug_id']}" for e in evaluation.bugs}
        assert dev_keys & evaluation_keys == {"PySnooper/3"}
        assert len(dev.bugs) == 11, "5 pilots, 5 reserves and PySnooper/3"
        assert len(split_mod.pilot_bugs(dev)) == 6
        assert len([e for e in dev.bugs if e["role"] == "reserve"]) == 5

        # Project level: every other development project is in no evaluation wave
        # at all, pilot or reserve, so no item can be tuned on a project the paper
        # reports a number for.
        dev_projects = {e["project"] for e in dev.bugs}
        evaluation_projects = {e["project"] for e in evaluation.bugs}
        assert dev_projects & evaluation_projects == {"PySnooper"}

        seen = dev.bug("PySnooper", 3)
        assert seen["role"] == "pilot"
        assert "seen during development" in seen["notes"].lower()

        # The rule that chose the list is in the file, recorded before any gate ran
        # and before any Vanta result was looked at.
        rule = dev.data["reviewed_changes"][0]["rule"]
        assert "Round-robin across projects" in rule
        assert "until 5" in rule
        assert "skip it and continue" in rule

    def test_module_distributions_is_optional(self):
        """The split is valid without it: only a divergent module name needs a mapping."""
        assert split_mod.validate(split_data()) is None
        assert split_mod.Split(split_data()).module_distributions() == {}

    def test_a_reviewed_module_mapping_is_read_back(self):
        split = split_mod.Split(split_data(module_distributions={"yaml": "PyYAML"}))
        assert split.module_distributions() == {"yaml": "PyYAML"}

    def test_a_module_mapping_must_be_a_non_empty_object(self):
        with pytest.raises(SplitError, match="module_distributions"):
            split_mod.validate(split_data(module_distributions={}))
        with pytest.raises(SplitError, match="module_distributions"):
            split_mod.validate(split_data(module_distributions=["yaml"]))

    def test_a_module_mapping_needs_both_sides_named(self):
        with pytest.raises(SplitError, match="non-empty module names"):
            split_mod.validate(split_data(module_distributions={"": "PyYAML"}))
        with pytest.raises(SplitError, match="non-empty distribution name"):
            split_mod.validate(split_data(module_distributions={"yaml": "  "}))

    def test_the_reviewed_module_table_is_pinned_in_both_splits(self):
        """The divergent module names, as a reviewed decision, in the protocol.

        [VANTA-WHY] ``PIL`` (matplotlib/1) and ``dateutil`` (pandas/1) are the
        measured flags: the test side imports the module, the distribution is a
        different word, and no canonicalisation recovers it — so §3.2's closure
        refuses to guess and the answer goes in ``module_distributions`` where a
        reviewer sees it. ``pylab``, ``mpl_toolkits`` and ``pkg_resources`` are the
        other three names the closures reach that PyPI does not publish (measured:
        both 404). Pinned exactly, and identically in the development and evaluation
        splits, because a table that decides every layer's contents is the protocol,
        not a detail a later edit should move quietly.

        ``psycopg2`` is the reviewed exception to that "don't guess" rule being *refused*:
        ``import psycopg2`` is provided by two distributions that both exist on PyPI
        (``psycopg2`` and ``psycopg2-binary``, same module), so the harness refuses to
        pick one. The pick is the reviewer's, and it is recorded in the split — the
        table maps it to ``psycopg2-binary``, with the reason and the pin rule in
        ``reviewed_changes``. Until that was recorded the closure stopped the bug that
        reaches the module, which was the intended behaviour.
        """
        required = {
            "PIL": "Pillow",
            "dateutil": "python-dateutil",
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "bs4": "beautifulsoup4",
            "cv2": "opencv-python",
            "attr": "attrs",
            "jwt": "PyJWT",
            "dns": "dnspython",
            "Crypto": "pycryptodome",
            "google.protobuf": "protobuf",
            "OpenSSL": "pyOpenSSL",
            "magic": "python-magic",
            "serial": "pyserial",
            "usb": "pyusb",
            "zmq": "pyzmq",
            "MySQLdb": "mysqlclient",
            "pylab": "matplotlib",
            "mpl_toolkits": "matplotlib",
            "pkg_resources": "setuptools",
            "psycopg2": "psycopg2-binary",
        }
        dev = load_split(DEV_SPLIT_PATH).module_distributions()
        evaluation = load_split(SPLIT_PATH).module_distributions()
        assert dev == required
        assert evaluation == required
        assert dev["psycopg2"] == "psycopg2-binary", "the reviewer's pick, not a guess"

    def test_import_package_is_an_optional_declared_key(self):
        """The last-resort import name, carried only by a bug that needs it (§3.2).

        [VANTA-WHY] The normal route is the bug's own ``setup.py``/``pyproject.toml``;
        requiring ``import_package`` of every bug would put a redundant guess next to a
        fact. It is validated when present, and unknown keys are still an error, so the
        shape stays closed.
        """
        assert split_mod.validate(split_data()) is None
        assert split_mod.validate(split_data(bugs=[bug_entry(import_package="sanic")])) is None
        with pytest.raises(SplitError, match="import_package"):
            split_mod.validate(split_data(bugs=[bug_entry(import_package="  ")]))
        with pytest.raises(SplitError, match="import_package"):
            split_mod.validate(split_data(bugs=[bug_entry(import_package=1)]))

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

    # -- encodings ----------------------------------------------------------
    # BugsInPy's corpus is not uniformly UTF-8: black/22's requirements.txt is
    # UTF-16 with a mark. Reading it as UTF-8 with errors="replace" turned its
    # first line into mojibake that reached the requirement parser, which then
    # reported "Expected package name at the start of dependency specifier" —
    # an error naming the parser for a fault in the reader.
    def test_utf16_le_with_a_mark(self):
        data = "accessify==0.3.1\n".encode("utf-16")  # LE + BOM
        assert meta_mod.decode_metadata(data, path=Path("requirements.txt")) == (
            "accessify==0.3.1\n"
        )

    def test_utf16_be_with_a_mark(self):
        data = b"\xfe\xff" + "accessify==0.3.1\n".encode("utf-16-be")
        assert meta_mod.decode_metadata(data, path=Path("requirements.txt")) == (
            "accessify==0.3.1\n"
        )

    def test_utf8_sig_mark_is_stripped(self):
        data = b"\xef\xbb\xbf" + "typed-ast==1.4.0\n".encode("utf-8")
        text = meta_mod.decode_metadata(data, path=Path("requirements.txt"))
        assert text == "typed-ast==1.4.0\n"
        # Not merely "contains the line": a surviving U+FEFF would become part of
        # the first requirement's name.
        assert not text.startswith("\ufeff")

    def test_plain_utf8_passes_through(self):
        assert meta_mod.decode_metadata(b"a==1\n", path=Path("r.txt")) == "a==1\n"

    def test_utf16_without_a_mark_is_refused_not_guessed(self):
        """The one case where a guess *would* work is the one we refuse.

        BOM-less UTF-16LE of ASCII is valid UTF-8, so it decodes and leaves NULs.
        Guessing LE here would be right; guessing on a UTF-16BE file with the same
        bytes would not, and a wrong guess corrupts a requirement silently.
        """
        data = "accessify==0.3.1\n".encode("utf-16-le")
        with pytest.raises(meta_mod.MetadataError, match="NUL bytes"):
            meta_mod.decode_metadata(data, path=Path("requirements.txt"))

    def test_non_utf8_names_the_file_and_the_byte(self):
        data = b"caf\xe9==1.0\n"  # latin-1, not UTF-8
        with pytest.raises(meta_mod.MetadataError) as excinfo:
            meta_mod.decode_metadata(data, path=Path("projects/Demo/bugs/1/r.txt"))
        message = str(excinfo.value)
        assert "projects/Demo/bugs/1/r.txt" in message
        assert "0xe9" in message
        assert "not UTF-8" in message

    def test_read_metadata_text_reads_a_file(self, tmp_path):
        path = write(tmp_path / "requirements.txt", "decorator>=4.3.0\n")
        assert meta_mod.read_metadata_text(path) == "decorator>=4.3.0\n"

    def test_read_metadata_text_reports_an_unreadable_file(self, tmp_path):
        with pytest.raises(meta_mod.MetadataError, match="cannot read"):
            meta_mod.read_metadata_text(tmp_path / "absent.txt")

    def test_load_bug_info_reads_a_utf16_bug_info(self, tmp_path):
        """``bug.info`` itself is swept by the reader, not just ``requirements.txt``."""
        path = tmp_path / "projects" / "Demo" / "bugs" / "1" / "bug.info"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            (
                'python_version="3.8.3"\n'
                f'buggy_commit_id="{"b" * 40}"\n'
                f'fixed_commit_id="{"f" * 40}"\n'
                'test_file="tests/test_demo.py"\n'
            ).encode("utf-16")
        )
        bug = meta_mod.load_bug_info(tmp_path, "Demo", 1)
        assert bug.python_version == "3.8.3"
        assert bug.test_file == "tests/test_demo.py"

    def test_bake_reads_a_utf16_requirements_txt(self, tmp_path):
        """The reported failure, end to end: black/22's requirements.txt.

        Before the reader, this raised ``BakeError`` from the requirement parser
        with a message about dependency specifiers, on a file whose only fault was
        its encoding.
        """
        bug_dir = tmp_path / "projects" / "Demo" / "bugs" / "1"
        bug_dir.mkdir(parents=True, exist_ok=True)
        (bug_dir / "requirements.txt").write_bytes(
            "# a comment\naccessify==0.3.1\n\n-rmore.txt\n".encode("utf-16")
        )
        text = bake_mod.requirements_for(tmp_path, "Demo", 1)
        refs, skipped = bake_mod.requirements_refs(text, source="requirements.txt")
        assert [(ref.name, ref.spec) for ref in refs] == [("accessify", "accessify==0.3.1")]
        # And the line that is not a requirement is recorded, not silently dropped.
        assert [line.text for line in skipped] == ["-rmore.txt"]

    def test_bake_reads_a_utf16_setup_sh(self, tmp_path):
        bug_dir = tmp_path / "projects" / "Demo" / "bugs" / "1"
        bug_dir.mkdir(parents=True, exist_ok=True)
        (bug_dir / "setup.sh").write_bytes(
            "pip install decorator>=4.3.0\npython setup.py install\n".encode("utf-16")
        )
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(tmp_path, "Demo", 1)) == [
            "decorator>=4.3.0"
        ]

    @needs_clone
    def test_every_bugsinpy_file_the_split_reads_is_decodable(self):
        """Every metadata file of every bug in the split, read through the reader.

        Written as a sweep because the corpus is not uniformly UTF-8 and the next
        file like black/22's is not predictable: a bug whose ``setup.sh`` or
        ``bug.info`` is UTF-16 must fail here, in a second, rather than inside a
        bake after a checkpoint name has been created.
        """
        split = load_split(SPLIT_PATH)
        poisoned: list[str] = []
        for entry in split.bugs:
            project = str(entry["project"])
            bug_dir = BUGSINPY_CLONE / "projects" / project / "bugs" / str(entry["bug_id"])
            for name in (
                "bug.info",
                "requirements.txt",
                "setup.sh",
                "run_test.sh",
                "bug_patch.txt",
            ):
                path = bug_dir / name
                if not path.is_file():
                    continue
                try:
                    meta_mod.read_metadata_text(path)
                except meta_mod.MetadataError as exc:
                    poisoned.append(str(exc))
            assert meta_mod.read_metadata_text(
                BUGSINPY_CLONE / "projects" / project / "project.info"
            )
        assert poisoned == [], "\n".join(poisoned)

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
# F2P is the flipping set ∩ what BugsInPy declares (§1.3)
# ============================================================================
class TestDeclaredFailToPass:
    """A flipping test is not automatically the oracle.

    Measured on PySnooper/2: the fix adds ``tests/test_utils/test_ensure_tuple.py``
    for a helper it also adds, so that test flips because the helper does not exist
    on the buggy source — not because of the bug — and the gate used to refuse the
    whole bug over it. The rule is BugsInPy's own declaration, not a judgement about
    which tests look related: ``bug.info``'s ``test_file`` plus the nodeids
    ``run_test.sh`` names. Everything else is undeclared, so it is neither the
    oracle nor a test the Creator may read.
    """

    @staticmethod
    def _pair(failures: dict[str, str]) -> suite_mod.SuitePair:
        run = suite_mod.SuiteRun(outcomes=dict(failures), exit_code=0, report_found=True)
        return suite_mod.compare_runs(run, run)

    @staticmethod
    def _patched_suites(monkeypatch, pairs: dict[str, suite_mod.SuitePair]) -> None:
        monkeypatch.setattr(
            prepare_mod,
            "_suite",
            lambda checkpoint, batch_dir, prep, tree_name, stem_suffix: pairs[tree_name],
        )

    def test_the_bug_s_own_file_is_declared_and_anything_else_is_not(self):
        bug = bug_info(test_file="tests/test_pysnooper.py")
        declared, undeclared = prepare_mod.split_declared_f2p(
            [
                "tests/test_pysnooper.py::test_custom_repr",
                "tests/test_pysnooper.py::test_disable",
                "tests/test_utils/test_ensure_tuple.py::test_ensure_tuple",
            ],
            bug,
        )
        assert declared == [
            "tests/test_pysnooper.py::test_custom_repr",
            "tests/test_pysnooper.py::test_disable",
        ]
        assert undeclared == ["tests/test_utils/test_ensure_tuple.py::test_ensure_tuple"]

    def test_a_nodeid_only_the_projects_runner_names_is_still_declared(self):
        """``run_test.sh`` can name a nodeid in a file ``test_file`` does not."""
        bug = bug_info(
            test_file="tests/test_demo.py",
            run_test_sh="# a comment\npytest -q -s tests/other/test_named.py::test_named\n",
        )
        declared, undeclared = prepare_mod.split_declared_f2p(
            ["tests/other/test_named.py::test_named", "tests/stray.py::test_stray"], bug
        )
        assert declared == ["tests/other/test_named.py::test_named"]
        assert undeclared == ["tests/stray.py::test_stray"]

    def test_derive_hidden_keeps_the_declared_and_records_the_undeclared(
        self, tmp_path, monkeypatch
    ):
        """PySnooper/2's shape: four flip, three are BugsInPy's own test file."""
        bug = bug_info(
            test_file="tests/test_pysnooper.py",
            run_test_sh="pytest -q -s tests/test_pysnooper.py::test_custom_repr_single\n",
        )
        flipping = {
            "tests/test_pysnooper.py::test_custom_repr": "failed",
            "tests/test_pysnooper.py::test_custom_repr_single": "failed",
            "tests/test_pysnooper.py::test_disable": "failed",
            "tests/test_utils/test_ensure_tuple.py::test_ensure_tuple": "failed",
        }
        green = {nodeid: "passed" for nodeid in flipping}
        self._patched_suites(
            monkeypatch,
            {
                "src_buggy": self._pair(flipping),
                "oracle_tree": self._pair(flipping),
                "src_fixed": self._pair(green),
            },
        )
        bug_prep = prep(bug=bug, root=tmp_path / "batch" / "worktrees" / "Demo_1")

        prepare_mod.derive_hidden(
            bug_prep, checkpoint="chk", batch_dir=tmp_path / "batch", collected=set()
        )

        assert bug_prep.fail_to_pass == [
            "tests/test_pysnooper.py::test_custom_repr",
            "tests/test_pysnooper.py::test_custom_repr_single",
            "tests/test_pysnooper.py::test_disable",
        ]
        assert bug_prep.f2p_undeclared == [
            "tests/test_utils/test_ensure_tuple.py::test_ensure_tuple"
        ]
        assert bug_prep.f2p_undeclared_reason == prepare_mod.UNDECLARED_F2P_REASON
        assert bug_prep.gate_failures == [], "an undeclared flipper is not a gate failure"

    def test_an_empty_declared_intersection_is_still_a_gate_failure(
        self, tmp_path, monkeypatch
    ):
        """Every flipping test undeclared means there is no oracle — as PySnooper/1."""
        bug = bug_info(test_file="tests/test_pysnooper.py")
        flipping = {
            "tests/test_utils/test_ensure_tuple.py::test_ensure_tuple": "failed",
            "tests/test_other/test_new.py::test_new": "failed",
        }
        green = {nodeid: "passed" for nodeid in flipping}
        self._patched_suites(
            monkeypatch,
            {
                "src_buggy": self._pair(flipping),
                "oracle_tree": self._pair(flipping),
                "src_fixed": self._pair(green),
            },
        )
        bug_prep = prep(bug=bug, root=tmp_path / "batch" / "worktrees" / "Demo_1")

        prepare_mod.derive_hidden(
            bug_prep, checkpoint="chk", batch_dir=tmp_path / "batch", collected=set()
        )

        assert bug_prep.fail_to_pass == []
        assert bug_prep.f2p_undeclared == sorted(flipping)
        assert any("empty FAIL_TO_PASS" in f for f in bug_prep.gate_failures)

    @pytest.mark.parametrize("schema", ["mapping_0_3", "flat_list_0_2"])
    def test_strip_and_baseline_hides_the_undeclared_test_too(
        self, tmp_path, monkeypatch, schema
    ):
        """Hidden from the tree, and therefore from the P2P suite as well.

        Both plugin schemas: the collected nodeids come from the oracle's report
        whether that file is the 0.3+ mapping or the flat list 0.1/0.2 write, which is
        what a bug whose own pytest pin caps the plugin at 0.2 gets (§3.2). Reading
        only `["tests"]` left the collected set empty for that bug, so nothing was
        hidden from the Creator and nothing was stripped from the P2P suite.
        """
        root = tmp_path / "batch" / "worktrees" / "Demo_1"
        declared_file = "tests/test_pysnooper.py"
        undeclared_file = "tests/test_utils/test_ensure_tuple.py"
        write(
            root / "oracle_tree" / declared_file,
            "def test_custom_repr():\n    assert 1\n\n\ndef test_disable():\n    assert 2\n",
        )
        write(
            root / "oracle_tree" / undeclared_file,
            "def test_ensure_tuple():\n    assert 1\n",
        )
        entries = [
            {"nodeid": f"{declared_file}::test_custom_repr"},
            {"nodeid": f"{declared_file}::test_disable"},
            {"nodeid": f"{undeclared_file}::test_ensure_tuple"},
        ]
        report = {"tests": entries} if schema == "mapping_0_3" else entries
        monkeypatch.setattr(prepare_mod, "_read_report", lambda batch_dir, stem: report)
        seen: dict = {}

        def fake_suite(*args, **kwargs):
            seen["tree_name"] = args[3]
            run = suite_mod.SuiteRun(
                outcomes={f"{declared_file}::test_disable": "passed"},
                exit_code=0,
                report_found=True,
            )
            return suite_mod.compare_runs(run, run)

        monkeypatch.setattr(prepare_mod, "_suite", fake_suite)
        bug_prep = prep(
            root=root,
            test_paths=["tests"],
            fail_to_pass=[f"{declared_file}::test_custom_repr"],
            f2p_undeclared=[f"{undeclared_file}::test_ensure_tuple"],
        )

        prepare_mod.strip_and_baseline(
            bug_prep, checkpoint="chk", batch_dir=tmp_path / "batch"
        )

        assert not (root / "vanta_tree" / undeclared_file).exists(), (
            "the Creator must not be able to read a test the oracle does not grade"
        )
        kept = (root / "vanta_tree" / declared_file).read_text()
        assert "test_disable" in kept
        assert "test_custom_repr" not in kept
        assert bug_prep.hidden_files_deleted == [undeclared_file]
        assert seen["tree_name"] == "vanta_tree"


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

        ``context_budget_exceeded`` (in ``src/llm/mod.rs``) compares ``prompt +
        output + overhead > n_ctx``, and its ``output`` is the request body's
        ``"max_tokens"``. Every M3 call site gets that number from
        ``SAMPLING_SCHEDULE`` through ``with_sampling`` — eb7cab9 moved the
        Creator's literal out of ``src/agents/creator.rs`` and into the table, so
        the table is what this test reads, not a copy of the number. If the
        Creator's ``max_tokens`` grew and this module kept subtracting 32768, the
        precheck would silently accept prompts the guard rejects, and the run
        would die at attempt 1 after a checkpoint and a wave had been spent. So
        the value is asserted against the table the request bodies are built from.
        """
        llm = (REPO_ROOT / "src" / "llm" / "mod.rs").read_text(encoding="utf-8")

        bytes_per_token = re.search(
            r"const TOKEN_ESTIMATE_BYTES_PER_TOKEN: u32 = (\d+);", llm
        )
        overhead = re.search(r"const CHAT_OVERHEAD_TOKENS: u32 = (\d+);", llm)
        assert bytes_per_token, "TOKEN_ESTIMATE_BYTES_PER_TOKEN not found in src/llm/mod.rs"
        assert overhead, "CHAT_OVERHEAD_TOKENS not found in src/llm/mod.rs"
        assert int(bytes_per_token.group(1)) == ctx.BYTES_PER_TOKEN_ESTIMATE
        assert int(overhead.group(1)) == ctx.CHAT_OVERHEAD_TOKENS

        block = llm.split("pub const SAMPLING_SCHEDULE", 1)[1].split("];", 1)[0]
        pinned = {
            site: int(max_tokens)
            for site, _temperature, max_tokens in re.findall(
                r'\("([a-z_]+)",\s*([0-9.]+),\s*(\d+)\)', block
            )
        }
        assert pinned, "could not read SAMPLING_SCHEDULE out of src/llm/mod.rs"

        # The Creator asks for this many, and the gate reserves exactly it.
        assert pinned["creator"] == ctx.OUTPUT_RESERVE_TOKENS

        # The gate reserves one number for every site, so it must be at least the
        # largest cap: a site above it could still die at attempt 1 on a tree the
        # gate had passed. The auditor's 8192 (698933e) is below the reserve and
        # so needs no term of its own.
        assert max(pinned.values()) == ctx.OUTPUT_RESERVE_TOKENS, pinned

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

    @pytest.mark.parametrize("schema", ["mapping_0_3", "flat_list_0_2"])
    def test_failure_kinds_read_both_plugin_schemas(self, schema):
        """The 0.2 flat list carries the same `longrepr` blocks; it must be read."""
        entries = [
            {
                "nodeid": "t.py::test_a",
                "call": {"longrepr": "E   ValueError: bad flag"},
            }
        ]
        report = {"tests": entries} if schema == "mapping_0_3" else entries
        assert issue_mod.failure_kinds_from_report(report) == ["ValueError"]

    def test_failure_kinds_of_an_absent_report_are_empty_not_an_error(self):
        assert issue_mod.failure_kinds_from_report({}) == []


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
    """The reader must handle both plugin schemas, because which one is written
    depends on the bug's own pytest pin (§3.2): 0.1/0.2 write a flat list, 0.3+
    write ``{"tests": [...]}``. Both versions are actually in use — httpie/1's
    ``pytest==3.2.1`` caps the plugin at 0.2 — so both get a test.
    """

    TESTS = [
        {"nodeid": "tests/test_a.py::test_b", "outcome": "failed"},
        {"nodeid": "tests/test_a.py::test_c", "outcome": "passed"},
    ]
    EXPECTED = {
        "tests/test_a.py::test_b": "failed",
        "tests/test_a.py::test_c": "passed",
    }

    def test_parse_report_reads_the_mapping_schema_0_3_and_later_write(self, tmp_path):
        path = write(tmp_path / "r.json", json.dumps({"tests": self.TESTS}))
        assert suite_mod.parse_report(path) == self.EXPECTED

    def test_parse_report_reads_the_flat_list_schema_0_1_and_0_2_write(self, tmp_path):
        """The newest plugin that accepts httpie/1's pytest==3.2.1 is 0.2."""
        path = write(tmp_path / "r.json", json.dumps(self.TESTS))
        assert suite_mod.parse_report(path) == self.EXPECTED

    def test_a_missing_mapping_key_is_an_empty_report_not_an_error(self, tmp_path):
        path = write(tmp_path / "r.json", json.dumps({"created": 0.0}))
        assert suite_mod.parse_report(path) == {}

    def test_a_top_level_that_is_neither_schema_is_an_error(self, tmp_path):
        path = write(tmp_path / "r.json", json.dumps("passed"))
        with pytest.raises(suite_mod.SuiteError, match="top level"):
            suite_mod.parse_report(path)

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

    # ---- the shared entry reader, which prepare and issue also go through -----
    #
    # [VANTA-WHY] `parse_report` is not the only reader of the file. The stripper
    # (prepare) reads the collected nodeids out of the oracle's report and the issue
    # text (issue) reads exception classes out of it, and both used to look only under
    # `["tests"]`. On a bug whose own pin caps the plugin at 0.2 (httpie/1; §3.2) they
    # therefore saw ZERO collected tests and ZERO exception classes — a silently wrong
    # answer, not an error. One helper now knows both schemas and all three read it.

    def test_report_tests_reads_both_schemas(self):
        assert suite_mod.report_tests(self.TESTS) == self.TESTS
        assert suite_mod.report_tests({"tests": self.TESTS}) == self.TESTS

    def test_report_tests_an_absent_key_is_an_empty_report_not_an_error(self):
        assert suite_mod.report_tests({}) == []

    def test_report_tests_a_top_level_that_is_neither_schema_is_an_error(self):
        with pytest.raises(suite_mod.SuiteError, match="top level"):
            suite_mod.report_tests("passed")
        with pytest.raises(suite_mod.SuiteError, match="`tests` is a str"):
            suite_mod.report_tests({"tests": "nope"})


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

    def test_a_pythonpath_into_the_tree_needs_build_in_the_split(self, monkeypatch):
        """The clause lives in ``check_build_declared``, which reads the tree (§3.2).

        ``check_declared`` is metadata-only now: ``needs_build`` is a property of the
        cloned tree, not of BugsInPy's ``bug.info``, so it is checked against the
        clone in both directions rather than inferred from ``pythonpath``.
        """
        monkeypatch.setattr(
            bake_mod,
            "needs_build_at",
            lambda repo, commit, *, pythonpath=None: (
                True,
                "pythonpath points into the tree",
            ),
        )
        entry = bug_entry(pythonpath="demo/build/lib/")
        bug = bug_info(pythonpath="demo/build/lib/")
        assert prepare_mod.check_declared(entry, bug) == []
        repo = Path("/clones/demo.git")
        mismatches = prepare_mod.check_build_declared(entry, bug, repo=repo)
        assert len(mismatches) == 1
        assert "needs_build" in mismatches[0]
        assert prepare_mod.check_build_declared(
            bug_entry(pythonpath="demo/build/lib/", needs_build=True), bug, repo=repo
        ) == []

    def test_missing_checkpoints_names_every_missing_layer(self, monkeypatch):
        # Patched on ``bake``, not on ``prepare``: prepare has to reach the layer
        # through the same function the bake decides skip-vs-rebuild with, and this
        # is what makes that shared. A patch on prepare's own module namespace would
        # pass while the two stages used different checks.
        monkeypatch.setattr(
            bake_mod, "layer_exists", lambda name: name == "vanta-bugsinpy-py3.8"
        )
        # Reported, not raised on: a missing layer is one bug's problem and prepare
        # flags that bug and prepares the rest (§3.2).
        missing = prepare_mod.missing_checkpoints(
            ["vanta-bugsinpy-py3.8", "vanta-bi-thefuck-1", "vanta-bi-httpie-1"]
        )
        assert missing == ["vanta-bi-httpie-1", "vanta-bi-thefuck-1"]

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
            "checkpoint": "vanta-bi-Demo-1",
            "installs": ["pytest==8.0.2", "pytest-json-report==1.5.0"],
            "dependencies": [{"name": "pytest", "version": "8.0.2"}],
        }
        original = prep(
            root=tmp_path / "batch" / "worktrees" / "Demo_1",
            prepare_mode=prepare_mod.FULL,
            test_paths=["tests"],
            fail_to_pass=["tests/test_demo.py::test_a"],
            f2p_undeclared=["tests/test_utils/test_ensure_tuple.py::test_ensure_tuple"],
            f2p_undeclared_reason=prepare_mod.UNDECLARED_F2P_REASON,
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
        assert loaded.f2p_undeclared == sorted(original.f2p_undeclared)
        assert loaded.f2p_undeclared_reason == original.f2p_undeclared_reason
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
        assert batch.entries["Demo/1"]["checkpoint"] == "vanta-bi-Demo-1"

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

    # ---- the interpreter layer's system packages -------------------------
    #
    # [VANTA-WHY] The measured trees that compile extensions (matplotlib/1,2,
    # pandas/1, spacy/4) need a C++ compiler and the `-dev` headers they link
    # against, and every py3.8 bug shares one interpreter layer, so the packages go
    # there rather than into each bug's layer. The layer records their versions in
    # itself, and a package that is missing fails this layer's bake — not a bug's
    # build an hour later.

    SYSTEM_PACKAGES = ("g++", "pkg-config", "libfreetype-dev", "libpng-dev", "zlib1g-dev")

    def test_the_interpreter_recipe_installs_the_system_packages(self):
        recipe = self._recipe()
        assert (
            "apt-get install -y --no-install-recommends "
            + " ".join(self.SYSTEM_PACKAGES) in recipe
        )

    def test_the_system_packages_are_those_the_split_compiles_against(self):
        """One list, so the recipe and this expectation cannot drift apart."""
        assert bake_mod.SYSTEM_PACKAGES["3.8"] == self.SYSTEM_PACKAGES

    def test_the_layer_records_the_installed_versions_in_itself(self):
        """The interpreter layer's provenance for its non-pip half.

        Beside ``/opt/vanta-deps-freeze.txt``: written by the recipe that installs
        them, so it describes the layer rather than a host-side intention.
        """
        recipe = self._recipe()
        assert "dpkg-query -W" in recipe
        assert "/opt/vanta-system-packages.txt" in recipe

    def test_a_missing_system_package_fails_the_bake(self):
        """A layer without the compiler must not be cached under a valid name.

        The check is on dpkg's own status field: a package dpkg knows but has not
        installed still prints a version, so a version-only check would pass.
        """
        recipe = self._recipe()
        assert " installed$" in recipe
        assert "exit 1" in recipe
        assert "missing from the layer after apt-get:$missing" in recipe
        # Before the marker, or the layer is saved as built anyway.
        assert recipe.index("exit 1") < recipe.index(f"touch {bake_mod.MARKER}")

    def test_a_minor_without_system_packages_is_unchanged(self):
        """3.6/3.7 were baked without them, so their recipe must not move.

        Their layers are neither rebuilt nor re-downloaded by this decision — and
        that includes the compiler environment and the smoke check below, which are
        keyed on the same fact (there is no compiler on 3.6/3.7 to point them at).
        """
        recipe = bake_mod.python_recipe("3.6", bake_mod.PytestPin("6.2.5", "1.4.1"))
        assert "apt-get" not in recipe
        assert recipe.splitlines()[:2] == ["set -eux", 'export HOME="${HOME:-/root}"']
        assert "export CC" not in recipe
        assert "build_ext" not in recipe

    def test_the_system_package_lines_are_empty_for_a_minor_without_them(self):
        assert bake_mod.system_package_lines(()) == []

    # ---- the compiler the source builds run ------------------------------
    #
    # [VANTA-WHY] Measured on matplotlib/1: ``numpy==1.18.5`` has no wheel for the
    # sandbox (linux/aarch64, py3.8), so the layer builds it from source, and that
    # build ended in ``error: [Errno 2] No such file or directory`` — *after*
    # configure had read the tree, i.e. the compiler *invocation* failed. uv's
    # interpreter is a python-build-standalone build whose ``sysconfig`` records the
    # compiler its own build machine used, and an old-style ``distutils`` build runs
    # that string. ``gcc``/``g++`` are installed by the system packages above and were
    # never being called; ``CC``/``CXX``/``LDSHARED``/``LDCXXSHARED`` are distutils'
    # documented override.

    COMPILER_ENV = {
        "CC": "gcc",
        "CXX": "g++",
        "LDSHARED": "gcc -pthread -shared",
        "LDCXXSHARED": "g++ -pthread -shared",
    }

    def test_the_compiler_environment_is_the_distutils_override_for_this_toolchain(self):
        """One list, so the recipe and this expectation cannot drift apart."""
        assert bake_mod.COMPILER_ENV == self.COMPILER_ENV

    def test_the_interpreter_recipe_exports_the_compiler_for_source_builds(self):
        """Quoted as shell, because a linker line is more than one word."""
        recipe = self._recipe()
        assert "export CC=gcc" in recipe
        assert "export CXX=g++" in recipe
        assert "export LDSHARED='gcc -pthread -shared'" in recipe
        assert "export LDCXXSHARED='g++ -pthread -shared'" in recipe

    def test_the_recipe_prints_what_the_standalone_recorded(self):
        """The evidence for the override: sysconfig's own CC/CXX, not ``$CC``.

        Printed rather than only overridden, so a source build that dies on
        ``[Errno 2]`` is not a mystery until someone boots the layer and looks.
        """
        recipe = self._recipe()
        assert "sysconfig.get_config_var('CC')" in recipe
        assert "sysconfig.get_config_var('CXX')" in recipe

    def test_a_layer_fails_when_the_compiler_environment_did_not_reach_it(self):
        """By name, rather than reproducing the ``[Errno 2]`` this exists to prevent."""
        recipe = self._recipe()
        assert "did not reach this layer" in recipe
        for name in self.COMPILER_ENV:
            assert f'[ -n "${name}" ]' in recipe

    def test_the_compiler_environment_is_recorded_in_the_layer(self):
        """The layer's provenance for its compiler, beside its system packages.

        Beside ``/opt/vanta-system-packages.txt``: written by the recipe that sets
        it, so it describes the layer rather than a host-side intention.
        """
        recipe = self._recipe()
        assert bake_mod.BUILD_ENV_RECORD in recipe
        assert "env | grep -E" in recipe

    def test_a_source_build_runs_after_the_compiler_is_exported(self):
        """The export is what the build reads; before it, the recorded compiler runs."""
        recipe = self._recipe()
        assert recipe.index("export CC=gcc") < recipe.index("build_ext")
        assert recipe.index("export CC=gcc") < recipe.index(f"touch {bake_mod.MARKER}")

    def test_the_interpreter_recipe_smokes_the_toolchain_before_the_marker(self):
        """A broken toolchain fails the base layer, not twenty minutes into numpy."""
        recipe = self._recipe()
        assert "build_ext" in recipe
        assert "vanta_smoke_c" in recipe
        assert "vanta_smoke_cpp" in recipe
        assert recipe.index("build_ext") < recipe.index(f"touch {bake_mod.MARKER}")

    def test_the_smoke_check_builds_both_c_and_cpp(self):
        """``CC`` and ``CXX`` are two variables; a C-only smoke check passes a broken C++.

        The measured trees compile C++ extensions (matplotlib's ``ft2font``, pandas'
        ``_libs``), so the smoke check declares a ``language="c++"`` extension too.
        """
        assert 'language="c++"' in self._recipe()

    def test_the_compiler_environment_is_empty_for_a_minor_without_a_toolchain(self):
        """Keyed on ``SYSTEM_PACKAGES``, so a 3.6/3.7 layer gets neither lines."""
        assert bake_mod.build_env_for("3.8") == bake_mod.COMPILER_ENV
        assert bake_mod.build_env_for("3.6") == {}
        assert bake_mod.build_env_for(None) == {}
        assert bake_mod.build_env_lines({}) == []
        assert bake_mod.toolchain_smoke_lines({}) == []

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

    def test_the_toolchain_is_installed_first_and_the_builds_are_not_isolated(self):
        """The installs, in order, and which of them keeps pip's own isolation.

        Order is the whole point: pip builds a source distribution in a *fresh*
        environment unless told otherwise, so the dated toolchain has to be in the
        venv before the dependency install runs — and that install has to run with
        ``--no-build-isolation`` for the pins to be what the build uses. The
        toolchain call is the one install that keeps isolation, because at that point
        there is nothing in the layer for pip to build a pinned tool against.
        """
        recipe = bake_mod.requirements_recipe(
            ["numpy==1.18.5"],
            build_pins=["setuptools==40.8.0", "wheel==0.33.1", "cython==0.29.6"],
        )
        pip_lines = [
            line for line in recipe.splitlines() if line.startswith("python -m pip install")
        ]
        assert pip_lines == [
            "python -m pip install --no-cache-dir -r /tmp/build-tools.txt",
            "python -m pip install --no-cache-dir --no-build-isolation -r /tmp/req.txt",
        ]
        assert "VANTA_BUILD_TOOLS" in recipe
        assert recipe.index("setuptools==40.8.0") < recipe.index("--no-build-isolation")
        # The pinned versions are asserted, not assumed: a toolchain that did not land
        # would leave the build to the interpreter layer's seeded setuptools and look
        # identical in the log.
        assert "importlib.metadata" in recipe
        assert "('setuptools', '40.8.0')" in recipe
        assert "('cython', '0.29.6')" in recipe

    def test_a_declared_build_requirement_is_never_built_in_an_isolated_env(self):
        """The measured matplotlib failure, as a property of the recipe.

        ``setup.py`` says ``setup_requires=["numpy>=1.15"]``, so numpy is a *build*
        pin. Shared with the toolchain in one build install that kept pip's
        isolation, pip built ``numpy==1.18.5`` in ``/tmp/pip-build-env-*/overlay``
        with *today's* setuptools — which is why the log printed
        ``build pins: setuptools==47.1.1 …`` and numpy still died in
        ``numpy/distutils/mingw32ccompiler``, one line above the assertion that said
        the pin had landed.
        """
        recipe = bake_mod.requirements_recipe(
            ["decorator==4.4.0"],
            build_pins=[
                "setuptools==47.1.1",
                "wheel==0.33.1",
                "cython==0.29.6",
                "numpy==1.18.5",
            ],
        )
        pip_lines = [
            line for line in recipe.splitlines() if line.startswith("python -m pip install")
        ]
        assert pip_lines == [
            "python -m pip install --no-cache-dir -r /tmp/build-tools.txt",
            "python -m pip install --no-cache-dir --no-build-isolation -r /tmp/build-req.txt",
            "python -m pip install --no-cache-dir --no-build-isolation -r /tmp/req.txt",
        ]
        tools = _heredoc(recipe, "VANTA_BUILD_TOOLS")
        build_req = _heredoc(recipe, "VANTA_BUILD_REQUIRES")
        assert "setuptools==47.1.1" in tools
        assert "numpy==1.18.5" in build_req
        assert "numpy" not in tools
        assert recipe.index("VANTA_BUILD_REQUIRES") < recipe.index("--no-build-isolation")

    def test_every_non_isolated_install_checks_for_a_leftover_build_env(self):
        """``/tmp/pip-build-env-*`` is the isolated build env made visible (§3.2).

        pip removes it on a normal exit, so the guard catches the run that created
        one anyway — an isolated build that failed or was aborted — and fails the
        layer rather than recording one built with today's toolchain.
        """
        bare = bake_mod.requirements_recipe(["decorator==4.4.0"])
        assert bare.count("--no-build-isolation") == 1
        assert bare.count("/tmp/pip-build-env-*") == 1
        built = bake_mod.requirements_recipe(
            ["decorator==4.4.0"], build_pins=["setuptools==47.1.1", "numpy==1.18.5"]
        )
        assert built.count("--no-build-isolation") == 2
        assert built.count("/tmp/pip-build-env-*") == 2
        assert "exit 1" in built

    def test_a_recipe_without_build_pins_asks_for_no_toolchain(self):
        recipe = bake_mod.requirements_recipe(["decorator==4.4.0"])
        assert "/tmp/build-tools.txt" not in recipe
        assert "/tmp/build-req.txt" not in recipe
        # The dependency install stays non-isolated even with no build pins: the
        # interpreter layer is `uv venv --seed`, so it carries a setuptools and a
        # wheel, and a dependency pulled transitively can itself be a source build.
        assert "--no-build-isolation" in recipe

    def test_build_pins_without_dependencies_still_install_the_toolchain(self):
        recipe = bake_mod.requirements_recipe([], build_pins=["setuptools==40.8.0"])
        assert "VANTA_BUILD_TOOLS" in recipe
        assert "VANTA_REQUIREMENTS" not in recipe
        assert recipe.strip().splitlines()[-1] == f"touch {bake_mod.MARKER}"

    def test_the_per_bug_layer_sets_the_compiler_in_its_own_recipe(self):
        """The interpreter layer's ``export`` does not survive ``--from``.

        A checkpoint is a filesystem: the exported ``CC``/``CXX`` are shell state and
        a layer baked ``--from`` its interpreter layer starts a fresh shell without
        them. Without this the per-bug layer's source builds (matplotlib's and pandas'
        numpy) run the compiler the standalone recorded for its own build machine.
        """
        recipe = bake_mod.requirements_recipe(["numpy==1.18.5"], minor="3.8")
        assert "export CC=gcc" in recipe
        assert "export CXX=g++" in recipe
        assert recipe.index("export CC=gcc") < recipe.index("--no-build-isolation")

    def test_a_per_bug_recipe_without_a_toolchain_minor_is_unchanged(self):
        """The default is no environment, so every existing caller keeps its recipe."""
        assert "export CC" not in bake_mod.requirements_recipe(["decorator==4.4.0"])
        assert (
            "export CC"
            not in bake_mod.requirements_recipe(["decorator==4.4.0"], minor="3.6")
        )

    def test_layer_recipes_wires_the_minor_into_the_bug_recipe(self):
        """The minor comes from the bug's own entry, not from the interpreter layer."""
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        entry = split.bug("Demo", 1)
        deps = [
            bake_mod.BugDeps(
                entry=entry,
                checkpoint="vanta-bi-Demo-1",
                installs=("decorator==4.4.0",),
                provenance={},
            )
        ]
        layers = bake_mod.layer_recipes(split, pythons=["3.8"], entries=[entry], deps=deps)
        recipe = {name: script for name, _, script in layers}["vanta-bi-Demo-1"]
        assert "export CC=gcc" in recipe

    def test_a_bug_layer_recipe_carries_its_own_build_pins(self):
        """``layer_recipes`` is what wires the resolution into the layer's script."""
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        entry = split.bug("Demo", 1)
        deps = [
            bake_mod.BugDeps(
                entry=entry,
                checkpoint="vanta-bi-Demo-1",
                installs=("decorator==4.4.0",),
                provenance={},
                build_pins=("setuptools==40.8.0", "wheel==0.33.1", "cython==0.29.6"),
            )
        ]
        layers = bake_mod.layer_recipes(split, pythons=["3.8"], entries=[entry], deps=deps)
        recipe = {name: script for name, _, script in layers}["vanta-bi-Demo-1"]
        assert "setuptools==40.8.0" in recipe
        assert recipe.index("setuptools==40.8.0") < recipe.index("decorator==4.4.0")
        assert "--no-build-isolation" in recipe

    @needs_clone
    def test_real_pysnooper_layers_are_the_interpreter_plus_one_thin_layer(self):
        """PySnooper's deps live in ``setup.sh``; ``requirements.txt`` is empty.

        This is the measured reason ``declared_deps`` exists at all: a layer that
        read only ``requirements.txt`` would install nothing and the wave-1 run
        would die on ``ModuleNotFoundError``.
        """
        root = BUGSINPY_CLONE
        assert bake_mod.requirements_for(root, "PySnooper", 2) == ""
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(root, "PySnooper", 2)) == [
            "python_toolbox"
        ]
        assert bake_mod.declared_deps(bake_mod.setup_sh_for(root, "PySnooper", 3)) == [
            "decorator"
        ]

    def test_wave_1_layers_are_one_interpreter_and_one_bug_layer(self):
        split = load_split(SPLIT_PATH)
        minors, entries = bake_mod.select_layers(
            split, pythons=["3.8"], only=[], roles={"pilot"}
        )
        assert minors == ["3.8"]
        # Wave 1 is PySnooper/3 alone: black/22's promotion was retired (the third
        # reviewed_change) and PySnooper/2 was dropped (the fourth), so the
        # interpreter carries one bug layer.
        assert [entry["project"] for entry in entries] == ["PySnooper"]
        layers = bake_mod.layer_recipes(
            split,
            pythons=minors,
            entries=entries,
            deps=_bug_deps(entries, {"vanta-bi-PySnooper-3": ("decorator==4.4.0",)}),
        )
        assert [name for name, _, _ in layers] == [
            "vanta-bugsinpy-py3.8",
            "vanta-bi-PySnooper-3",
        ]
        assert layers[0][1] == bake_mod.DEFAULT_BASE
        assert all(base == "vanta-bugsinpy-py3.8" for _, base, _ in layers[1:])
        recipe_by_name = {name: script for name, _, script in layers}
        assert "decorator==4.4.0" in recipe_by_name["vanta-bi-PySnooper-3"]

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


def _wheel(filename: str) -> bake_mod.PyPIFile:
    return bake_mod.PyPIFile(
        filename=filename, packagetype="bdist_wheel", url=f"https://files.test/{filename}"
    )


def _sdist(filename: str) -> bake_mod.PyPIFile:
    return bake_mod.PyPIFile(
        filename=filename, packagetype="sdist", url=f"https://files.test/{filename}"
    )


class TestPinPreflight:
    """What every pin installs as, decided before anything is built (§3.2).

    Not ``TestPreflight``: that name is already taken by the runner's own preflight
    (CWD, the binary) further down this file, and a second class of the same name
    would shadow this one in the module namespace — collected as nothing.

    Every test injects ``files_of``/``sdist_of``: the defaults are PyPI and the
    network, and a preflight that can only be tested by reaching the network is a
    preflight nobody tests.
    """

    def _classify(
        self, files, *, members=(), truncated=False, pin="demo==1.0", minor="3.8"
    ):
        name, version = pin.split("==")
        return bake_mod.preflight_pin(
            name,
            version,
            python_minor=minor,
            files_of=lambda n, v: None if files is None else tuple(files),
            sdist_of=lambda file: (tuple(members), truncated),
        )

    def test_a_pure_python_wheel_installs_on_this_interpreter(self):
        """``py2.py3`` is a compressed tag set: the ``py3`` half is the one that answers."""
        pin = self._classify([_wheel("six-1.15.0-py2.py3-none-any.whl")])
        assert pin.kind == bake_mod.INSTALLABLE_WHEEL
        assert pin.matched_tag == "py3-none-any"
        assert pin.letter == "a"

    def test_a_compressed_platform_field_is_expanded_not_compared_literally(self):
        """``manylinux1_x86_64.manylinux2014_aarch64`` — one wheel, two platforms."""
        pin = self._classify(
            [_wheel("numpy-1.18.5-cp38-cp38-manylinux1_x86_64.manylinux2014_aarch64.whl")],
            pin="numpy==1.18.5",
        )
        assert pin.kind == bake_mod.INSTALLABLE_WHEEL
        assert pin.matched_tag == "cp38-cp38-manylinux2014_aarch64"

    def test_a_wheel_marked_for_another_cpython_does_not_install(self):
        for filename in (
            "numpy-1.18.5-cp39-cp39-manylinux2014_aarch64.whl",
            "numpy-1.18.5-cp36-cp36m-manylinux2014_aarch64.whl",
            "numpy-1.18.5-cp39-abi3-manylinux2014_aarch64.whl",
            "numpy-1.18.5-cp27-cp27m-manylinux2014_aarch64.whl",
            "numpy-1.18.5-py2-none-any.whl",
        ):
            pin = self._classify([_wheel(filename), _sdist("numpy-1.18.5.tar.gz")])
            assert pin.kind == bake_mod.SDIST_ONLY, filename
            assert pin.letter == "b"

    def test_a_wheel_for_another_architecture_does_not_install(self):
        pin = self._classify(
            [
                _wheel("numpy-1.18.5-cp38-cp38-manylinux2014_x86_64.whl"),
                _wheel("numpy-1.18.5-cp38-cp38-win_amd64.whl"),
                _sdist("numpy-1.18.5.tar.gz"),
            ]
        )
        assert pin.kind == bake_mod.SDIST_ONLY

    def test_an_abi3_wheel_installs_up_to_its_own_minor(self):
        pin = self._classify([_wheel("cryptography-3.0-cp37-abi3-manylinux2014_aarch64.whl")])
        assert pin.kind == bake_mod.INSTALLABLE_WHEEL

    def test_a_build_tag_on_the_filename_does_not_confuse_the_tag_read(self):
        pin = self._classify([_wheel("demo-1.0-1-cp38-cp38m-manylinux_2_17_aarch64.whl")])
        assert pin.matched_tag == "cp38-cp38m-manylinux_2_17_aarch64"

    def test_a_wheel_is_preferred_over_the_sdist_of_the_same_release(self):
        pin = self._classify(
            [
                _sdist("cffi-1.14.0.tar.gz"),
                _wheel("cffi-1.14.0-cp38-cp38-manylinux2014_aarch64.whl"),
            ],
            members=("cffi-1.14.0/c/_cffi_backend.c",),
        )
        assert pin.kind == bake_mod.INSTALLABLE_WHEEL

    def test_an_sdist_that_compiles_says_what_it_will_compile(self):
        pin = self._classify(
            [_sdist("cffi-1.14.0.tar.gz")],
            members=(
                "cffi-1.14.0/c/_cffi_backend.c",
                "cffi-1.14.0/setup.py",
                "cffi-1.14.0/doc/index.rst",
            ),
        )
        assert pin.kind == bake_mod.SDIST_ONLY
        assert pin.compiles
        assert pin.compiled == ("cffi-1.14.0/c/_cffi_backend.c",)
        assert "_cffi_backend.c" in pin.reason

    def test_an_sdist_that_compiles_nothing_needs_no_system_package(self):
        pin = self._classify(
            [_sdist("pager-3.3.tar.gz")], members=("pager-3.3/pager.py", "pager-3.3/setup.py")
        )
        assert pin.kind == bake_mod.SDIST_ONLY
        assert not pin.compiles
        assert "no compiled sources" in pin.reason

    def test_a_test_data_file_named_like_c_is_not_a_compiled_source(self):
        """Measured on protego 0.1.16: four ``.cc`` files, all renamed corpora."""
        pin = self._classify(
            [_sdist("protego-0.1.16.tar.gz")],
            members=(
                "protego-0.1.16/tests/test_data/neurips.cc",
                "protego-0.1.16/protego.py",
            ),
        )
        assert not pin.compiles

    def test_fortran_is_something_to_compile(self):
        pin = self._classify(
            [_sdist("scipy-1.4.1.tar.gz")], members=("scipy-1.4.1/scipy/linalg/src/x.f90",)
        )
        assert pin.compiles

    def test_an_sdist_that_was_not_read_whole_does_not_claim_to_compile_nothing(self):
        pin = self._classify([_sdist("matplotlib-3.2.1.tar.gz")], truncated=True)
        assert pin.sdist_truncated
        assert "not read whole" in pin.reason

    def test_a_pin_pypi_never_released_is_unavailable_and_says_so(self):
        pin = self._classify(None, pin="requests-async==0.5.0")
        assert pin.kind == bake_mod.NO_DISTRIBUTION
        assert pin.letter == "c"
        assert "no such release" in pin.reason

    def test_a_release_with_only_foreign_wheels_and_no_sdist_is_unavailable(self):
        pin = self._classify(
            [
                _wheel("pywin32-227-cp27-cp27m-win_amd64.whl"),
                _wheel("pywin32-227-cp38-cp38-win_amd64.whl"),
            ],
            pin="pywin32==227",
        )
        assert pin.kind == bake_mod.NO_DISTRIBUTION
        assert "2 wheel(s)" in pin.reason
        assert "no source distribution" in pin.reason

    def test_the_files_and_the_listing_of_a_shared_pin_are_fetched_once(self):
        """The cache is what makes a 232-pin preflight cheap; numpy is in six bugs."""
        entries = [
            bug_entry(checkpoint="vanta-bi-Demo-1"),
            bug_entry(bug_id=2, checkpoint="vanta-bi-Demo-2"),
        ]
        deps = [
            bake_mod.BugDeps(
                entry=dict(entry),
                checkpoint=bake_mod.bug_checkpoint_name(entry),
                installs=("numpy==1.18.5",),
                provenance={},
            )
            for entry in entries
        ]
        fetched: list[tuple[str, str]] = []
        listings: list[str] = []

        def files_of(name, version):
            fetched.append((name, version))
            return (_sdist(f"{name}-{version}.tar.gz"),)

        def sdist_of(file):
            listings.append(file.filename)
            return ("numpy-1.18.5/numpy/core/src/multiarray/arrayobject.c",), False

        reports = bake_mod.preflight_layers(deps, files_of=files_of, sdist_of=sdist_of)
        assert fetched == [("numpy", "1.18.5")]
        assert listings == ["numpy-1.18.5.tar.gz"]
        assert [report.checkpoint for report in reports] == [
            "vanta-bi-Demo-1",
            "vanta-bi-Demo-2",
        ]
        assert all(report.pins[0].compiles for report in reports)

    def test_the_layers_pins_keep_the_build_pins_first_and_drop_repeats(self):
        """A build requirement the dependency set also pins is that one pin, once."""
        layer = bake_mod.BugDeps(
            entry=bug_entry(),
            checkpoint="vanta-bi-Demo-1",
            installs=("numpy==1.18.5", "pytest==8.0.2"),
            provenance={},
            build_pins=("setuptools==44.1.0", "numpy==1.18.5"),
        )
        assert bake_mod.layer_pins(layer) == (
            ("setuptools", "44.1.0"),
            ("numpy", "1.18.5"),
            ("pytest", "8.0.2"),
        )

    def _report(self):
        return bake_mod.PreflightBug(
            key="Demo/1",
            checkpoint="vanta-bi-Demo-1",
            python_minor="3.8",
            pins=(
                bake_mod.PreflightPin("decorator", "4.4.0", bake_mod.INSTALLABLE_WHEEL, "w"),
                bake_mod.PreflightPin(
                    "cffi",
                    "1.14.0",
                    bake_mod.SDIST_ONLY,
                    "no cp38 aarch64 wheel; cffi-1.14.0.tar.gz builds 24 compiled source(s)",
                    compiled=("cffi-1.14.0/c/_cffi_backend.c",),
                ),
                bake_mod.PreflightPin(
                    "pywin32", "227", bake_mod.NO_DISTRIBUTION, "nothing pip can install"
                ),
            ),
        )

    def test_the_table_lists_every_sdist_and_unavailable_entry(self):
        lines = bake_mod.preflight_lines([self._report()])
        assert "preflight: 1 layer(s)" in lines[0]
        assert (
            "[vanta-bi-Demo-1] Demo/1 (cp38/linux-aarch64): 3 pin(s), 1 wheel, "
            "1 sdist-only, 1 unavailable" in lines[1]
        )
        assert "(b) cffi==1.14.0 — no cp38 aarch64 wheel" in lines[2]
        assert "(c) pywin32==227 — nothing pip can install" in lines[3]
        assert "(a)" not in "\n".join(lines), "a usable wheel is not a finding"

    def test_a_layer_with_nothing_to_report_still_gets_its_line(self):
        quiet = bake_mod.PreflightBug(
            key="Demo/1",
            checkpoint="vanta-bi-Demo-1",
            python_minor="3.8",
            pins=(
                bake_mod.PreflightPin("six", "1.15.0", bake_mod.INSTALLABLE_WHEEL, "w"),
            ),
        )
        lines = bake_mod.preflight_lines([quiet])
        assert len(lines) == 2, "one header and one line per layer, finding or not"
        assert "1 wheel, 0 sdist-only, 0 unavailable" in lines[1]

    def test_the_record_is_written_beside_the_provenance(self, tmp_path):
        path = bake_mod.write_preflight(tmp_path, self._report())
        assert path.name == "vanta-bi-Demo-1.preflight.json"
        assert tmp_path / "vanta-bi-Demo-1.preflight.json" == path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == bake_mod.PREFLIGHT_SCHEMA_VERSION
        assert data["checkpoint"] == "vanta-bi-Demo-1"
        assert data["target"] == "cp38/linux-aarch64"
        assert data["counts"] == {"wheel": 1, "sdist": 1, "unavailable": 1}
        assert [pin["pin"] for pin in data["pins"]] == [
            "decorator==4.4.0",
            "cffi==1.14.0",
            "pywin32==227",
        ]
        assert data["pins"][1]["compiled_sources"] == ["cffi-1.14.0/c/_cffi_backend.c"]
        assert data["pins"][1]["class"] == "b"
        assert "compiled_sources" not in data["pins"][0]

    def _cli(self, tmp_path, monkeypatch, *, argv=(), flagged=()):
        """Drive ``bake`` with resolution, preflight and the build all injected."""
        root = tmp_path / "bugsinpy"
        root.mkdir()
        entry = bug_entry(checkpoint="vanta-bi-Demo-1")
        split = split_mod.Split(split_data(bugs=[entry], bugsinpy_root=str(root)))
        deps = _bug_deps([entry], {"vanta-bi-Demo-1": ("decorator==4.4.0",)})
        written: list[str] = []
        monkeypatch.setattr(run_mod, "load_split", lambda path: split)
        monkeypatch.setattr(
            bake_mod, "resolve_bug_deps", lambda *a, **k: (deps, list(flagged))
        )
        monkeypatch.setattr(bake_mod, "write_deps", lambda *a, **k: tmp_path / "d.json")
        monkeypatch.setattr(
            bake_mod, "preflight_layers", lambda *a, **k: [self._report()]
        )
        monkeypatch.setattr(
            bake_mod,
            "write_preflight",
            lambda deps_dir, report: written.append(report.checkpoint) or (tmp_path / "p.json"),
        )
        monkeypatch.setattr(bake_mod, "plan_layers", lambda *a, **k: [])
        monkeypatch.setattr(bake_mod, "run_bake_actions", lambda *a, **k: "summary line")
        code = run_mod.main(
            [
                "bake",
                "--pythons",
                "3.8",
                "--split",
                str(tmp_path / "s.json"),
                *argv,
            ]
        )
        return code, written

    def test_preflight_only_classifies_and_builds_nothing(self, tmp_path, monkeypatch, capsys):
        code, written = self._cli(
            tmp_path, monkeypatch, argv=["--preflight-only"], flagged=[]
        )
        captured = capsys.readouterr()
        assert written == ["vanta-bi-Demo-1"], "the record is written for every layer"
        assert "(b) cffi==1.14.0" in captured.out
        assert "preflight only: nothing was baked" in captured.out
        assert "bake plan" not in captured.out, "no plan: no layer is touched"
        assert "bake complete" not in captured.out
        assert code == run_mod.EXIT_OK

    def test_preflight_only_needs_no_yes_and_still_reports_flagged_bugs(
        self, tmp_path, monkeypatch, capsys
    ):
        """The preflight builds nothing, so the ``--yes`` gate does not apply to it."""
        code, _ = self._cli(
            tmp_path,
            monkeypatch,
            argv=["--preflight-only"],
            flagged=[bake_mod.FlaggedBug("Demo/1", "BakeError: unreadable")],
        )
        captured = capsys.readouterr()
        assert "flagged bugs (1)" in captured.out
        assert "dry run: nothing was baked" not in captured.err
        assert code == run_mod.EXIT_BUG_FAILED

    def test_a_bake_writes_the_preflight_before_it_builds_anything(
        self, tmp_path, monkeypatch, capsys
    ):
        code, written = self._cli(tmp_path, monkeypatch, argv=["--yes"])
        captured = capsys.readouterr()
        assert written == ["vanta-bi-Demo-1"]
        assert "preflight: 1 layer(s)" in captured.out
        assert "bake complete" in captured.out
        assert code == run_mod.EXIT_OK


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

    def test_a_setup_kwargs_dict_passed_as_a_splat_is_read(self):
        """sanic/1's shape, quoted from that commit's ``setup.py``.

        [VANTA-WHY] Sanic assembles its metadata in a module-level dict literal,
        appends the requirement keys by subscript, and calls
        ``setup(**setup_kwargs)``. Measured: a reader that dropped the splat saw
        neither ``name=`` nor a ``packages=[...]`` literal and flagged the bug,
        although every value is written down statically.
        """
        setup = (
            "requirements = ['httptools>=0.0.10', 'uvloop>=0.5.1']\n"
            "setup_kwargs = {\n"
            "    'name': 'sanic',\n"
            "    'packages': ['sanic'],\n"
            "    'version': '19.12.0',\n"
            "}\n"
            "setup_kwargs['install_requires'] = requirements\n"
            "setup(**setup_kwargs)\n"
        )
        assert bake_mod.import_name_of(setup) == "sanic"

    def test_the_kwargs_dicts_subscripted_install_requires_is_a_declaration(self):
        """The four keys sanic adds by subscript are dependencies like any other."""
        setup = (
            "requirements = ['httptools>=0.0.10']\n"
            "setup_kwargs = {'name': 'sanic'}\n"
            "setup_kwargs['install_requires'] = requirements\n"
            "setup(**setup_kwargs)\n"
        )
        refs, _ = setup_py_refs(setup, {}, [])
        assert [ref.spec for ref in refs] == ["httptools>=0.0.10"]

    def test_a_splat_that_is_not_a_dict_literal_contributes_nothing(self):
        """A computed ``**kwargs`` is not guessed at; the bug is flagged as before."""
        with pytest.raises(bake_mod.BakeError, match="neither name= nor a packages="):
            bake_mod.import_name_of("setup(**make_kwargs())\n")

    def test_setup_py_that_is_not_python_is_an_error(self):
        with pytest.raises(bake_mod.BakeError, match="does not parse as Python"):
            bake_mod.setup_py_refs("setup(name=\n", {}, [])


def setup_py_refs(setup_py: str, files: dict, notes: list) -> tuple:
    """``bake_mod.setup_py_refs`` with a dict standing in for the buggy commit."""
    return bake_mod.setup_py_refs(setup_py, files.get, notes)


class TestPyprojectStaticReader:
    """``pyproject.toml`` is parsed as TOML and never executed (§3.2, C3).

    ``fastapi/1`` is the measured case: at ``766157bf`` there is no ``setup.py`` at
    all — it is a flit project — so a reader that only knew ``setup.py`` refused the
    bug at ``bake`` before ``prepare`` ever ran. Each spelling below is one that
    exists in the wild, and the test extras are the part of them that decides whether
    a bug's own tests can run at all.
    """

    def test_pep621_project_dependencies_are_read_as_written(self):
        refs = pyproject_refs(
            "[project]\n"
            'name = "demo"\n'
            'dependencies = ["starlette >=0.13.2,<0.14.0", "pydantic >=1.0,<2.0.0"]\n',
            [],
        )
        assert [ref.spec for ref in refs] == [
            "starlette >=0.13.2,<0.14.0",
            "pydantic >=1.0,<2.0.0",
        ]
        assert refs[0].source == "project.dependencies (pyproject.toml)"
        assert refs[0].name == "starlette"

    def test_a_pep621_test_extra_is_a_declared_test_requirement(self):
        notes: list[str] = []
        refs = pyproject_refs(
            "[project]\n"
            'name = "demo"\n'
            'dependencies = ["six"]\n'
            "[project.optional-dependencies]\n"
            'test = ["python_toolbox>=0.9.3", "pytest>=4.4.1"]\n'
            'doc = ["mkdocs"]\n',
            notes,
        )
        assert {ref.name for ref in refs} == {"six", "python-toolbox", "pytest", "mkdocs"}
        assert next(ref for ref in refs if ref.name == "python-toolbox").source == (
            "project.optional-dependencies['test'] (pyproject.toml)"
        )
        assert "pyproject.toml: test extras taken: test (PEP 621)" in notes

    def test_flit_requires_and_requires_extra_are_read(self):
        """fastapi's own shape at ``766157bf``, quoted from that commit."""
        notes: list[str] = []
        refs = pyproject_refs(
            "[tool.flit.metadata]\n"
            'module = "fastapi"\n'
            'requires = ["starlette ==0.13.2", "pydantic >=0.32.2,<2.0.0"]\n'
            "[tool.flit.metadata.requires-extra]\n"
            'test = ["requests", "pytest"]\n'
            'doc = ["mkdocs"]\n',
            notes,
        )
        assert [ref.spec for ref in refs] == [
            "starlette ==0.13.2",
            "pydantic >=0.32.2,<2.0.0",
            "requests",
            "pytest",
            "mkdocs",
        ]
        assert refs[2].source == "tool.flit.metadata.requires-extra['test'] (pyproject.toml)"
        assert "pyproject.toml: test extras taken: test (flit)" in notes

    def test_every_test_extra_name_is_taken_and_the_ones_taken_are_recorded(self):
        """``test``/``testing``/``dev``/``all`` are the conventions; ``doc`` is not one."""
        notes: list[str] = []
        refs = pyproject_refs(
            "[project]\n"
            'name = "demo"\n'
            "[project.optional-dependencies]\n"
            'test = ["six>=1.12.0"]\n'
            'testing = ["decorator>=4.3.0"]\n'
            'dev = ["future>=0.15.0"]\n'
            'all = ["requests>=2.0"]\n'
            'doc = ["mkdocs>=1.0"]\n',
            notes,
        )
        assert [ref.source for ref in refs] == [
            f"project.optional-dependencies['{extra}'] (pyproject.toml)"
            for extra in ("test", "testing", "dev", "all", "doc")
        ]
        assert notes == [
            "pyproject.toml: test extras taken: test (PEP 621), testing (PEP 621), "
            "dev (PEP 621), all (PEP 621)"
        ], "every extra that counts as a test requirement is named in the record"

    def test_poetry_dependencies_are_read_where_the_spelling_is_pep440(self):
        notes: list[str] = []
        refs = pyproject_refs(
            "[tool.poetry.dependencies]\n"
            'python = "^3.8"\n'
            'six = "1.12.0"\n'
            'attrs = ">=19.3.0"\n'
            'anything = "*"\n',
            notes,
        )
        assert [ref.spec for ref in refs] == ["six==1.12.0", "attrs>=19.3.0", "anything"]
        assert not any(ref.name == "python" for ref in refs), (
            "``python`` names the interpreter, not a distribution"
        )
        assert refs[0].source == "tool.poetry.dependencies['six'] (pyproject.toml)"
        assert notes == []

    def test_a_bare_poetry_version_is_a_pin_not_a_package_named_after_itself(self):
        """``six1.12.0`` parses as a requirement name, so the operator check comes first."""
        refs = pyproject_refs('[tool.poetry.dependencies]\nsix = "1.12.0"\n', [])
        assert [ref.name for ref in refs] == ["six"]
        assert refs[0].spec == "six==1.12.0"

    def test_a_poetry_constraint_this_harness_cannot_read_is_recorded_not_guessed(self):
        """``^2.22`` is a caret range: translating it would be guessing at bounds."""
        notes: list[str] = []
        refs = pyproject_refs('[tool.poetry.dependencies]\nrequests = "^2.22"\n', notes)
        assert refs == []
        assert notes == [
            "tool.poetry.dependencies['requests'] (pyproject.toml): '^2.22' is a "
            "poetry constraint this harness does not translate to PEP 440; not read"
        ]

    def test_a_pyproject_with_none_of_the_three_tables_declares_nothing_and_says_so(self):
        notes: list[str] = []
        refs = pyproject_refs('[build-system]\nrequires = ["flit_core"]\n', notes)
        assert refs == [], "``flit_core`` is the build backend, not a runtime dependency"
        assert notes == [
            "pyproject.toml: none of [project], [tool.flit.metadata] or "
            "[tool.poetry.dependencies] is present at the buggy commit, so no "
            "dependency is declared here"
        ]

    def test_a_pyproject_that_is_not_toml_is_a_refusal_never_an_empty_read(self):
        with pytest.raises(bake_mod.BakeError, match="does not parse as TOML"):
            bake_mod.pyproject_refs("[project\nname = 'demo'\n", [])

    def test_refs_are_stamped_with_the_commit_they_were_read_from(self):
        refs = bake_mod.pyproject_refs(
            '[project]\nname = "demo"\ndependencies = ["six"]\n',
            [],
            commit="f" * 40,
            commit_date="2019-07-30",
            commit_label="the fixed commit",
        )
        assert refs[0].commit == "f" * 40
        assert refs[0].commit_date == "2019-07-30"

    def test_import_name_comes_from_the_project_name_first(self):
        assert bake_mod.import_name_of_pyproject('[project]\nname = "FastAPI"\n') == "fastapi"

    def test_import_name_falls_back_to_flit_module_then_flit_name_then_poetry(self):
        assert (
            bake_mod.import_name_of_pyproject('[tool.flit.metadata]\nmodule = "fastapi"\n')
            == "fastapi"
        )
        assert (
            bake_mod.import_name_of_pyproject(
                '[tool.flit.metadata]\nname = "python-toolbox"\n'
            )
            == "python_toolbox"
        )
        assert bake_mod.import_name_of_pyproject('[tool.poetry]\nname = "fastapi"\n') == "fastapi"

    def test_import_name_that_cannot_be_read_is_flagged_not_guessed(self):
        with pytest.raises(bake_mod.BakeError, match=r"no \[project\]\.name"):
            bake_mod.import_name_of_pyproject('[build-system]\nrequires = ["flit_core"]\n')


def pyproject_refs(pyproject: str, notes: list) -> list:
    """``bake_mod.pyproject_refs`` with no commit stamping — a pure parse."""
    return bake_mod.pyproject_refs(pyproject, notes)


class TestResolvePinsByDate:
    """Pin by release date, with the dates passed in — no network in a test."""

    RELEASES = {
        "decorator": {"4.3.0": "2018-02-03", "4.4.0": "2019-03-16", "4.4.1": "2019-06-01"},
        "future": {"0.17.1": "2018-10-31"},
        "six": {"1.12.0": "2018-12-10", "1.13.0": "2020-05-31"},
        "python-toolbox": {"0.9.3": "2016-06-07"},
        # The build toolchain is date-pinned for every bug (§3.2), so any test that
        # runs the whole resolution path needs dates for it. The values are PyPI's
        # (TestBuildPins uses the same three), and both PySnooper trees — 2019-05-20
        # for bug 2, 2019-04-22 for bug 3 — have a release of each inside its window.
        "setuptools": {"40.8.0": "2019-02-04", "41.0.0": "2019-06-01"},
        "wheel": {"0.32.3": "2018-10-21", "0.33.1": "2019-04-06"},
        "cython": {"0.29.6": "2019-03-10", "0.29.13": "2019-08-30"},
        # The interpreter layer's own pair, for the harness-provides branch.
        "pytest": {"5.4.3": "2020-04-09", "8.0.2": "2023-12-30"},
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
        oracle is read from; re-pinning it is a silent downgrade of the report.

        A **range** is the case this covers: it names no release, so the layer has
        nothing to install and the harness version is the only answer. Its constraint
        is checked against that version rather than ignored.
        """
        refs = [bake_mod.dep_ref("pytest>=4.4.1", "test_requirements.txt line 2")]
        resolved = self._resolve(refs, harness_pins={"pytest": "8.0.2"})
        assert resolved[0].pin == "pytest==8.0.2"
        assert resolved[0].harness is True
        assert resolved[0].as_dict()["provisioned_by"] == "interpreter layer"

    def test_a_bugsinpys_own_pytest_pin_wins_over_the_harness_version(self):
        """The decision the four dev-pilot flags were about (§3.2).

        fastapi/1 and luigi/1 pin ``pytest==5.4.3``, pandas/1 ``5.4.2``, sanic/1
        ``5.2.1`` — all from BugsInPy's own ``requirements.txt``, so all reference
        pins — while the interpreter layer carries 8.0.2. Installing the harness pair
        over them asked the bug to run under a pytest it was never validated with,
        and the run flagged rather than pretending: pytest 8 refuses these suites.
        BugsInPy's exact ``==`` pin wins, exactly as it does for every other package,
        and the layer installs it *on top of* the interpreter layer's default. The
        report plugin then has to follow (:class:`TestTheReportPluginFollowsTheBugsPytest`),
        which is why this pin cannot be resolved in isolation from it.
        """
        refs = [bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7")]
        resolved = self._resolve(refs, harness_pins={"pytest": "8.0.2"})
        assert resolved[0].pin == "pytest==5.4.3"
        assert resolved[0].harness is False
        assert resolved[0].exempt is True
        # Released 2020-04-09, a year after the 2019-04-22 commit the test pins to:
        # the exemption is what lets it through, as it is for any late reference pin.
        assert resolved[0].released_at == "2020-04-09"
        assert resolved[0].pin_rule == "BugsInPy's own exact pin, installed as written"

    def test_a_project_pin_on_a_harness_package_is_not_a_reference_pin(self):
        """Only BugsInPy's own files are the reference environment (§3.2).

        A project that pins the version the harness already provides is satisfied by
        it and recorded as provisioned by the interpreter layer; one that pins a
        different version is refused rather than installed over it, because a project
        declaration is not the benchmark's validated environment.
        """
        agreeing = self._resolve(
            [bake_mod.dep_ref("pytest==8.0.2", "install_requires (setup.py) line 4")],
            harness_pins={"pytest": "8.0.2"},
        )
        assert agreeing[0].pin == "pytest==8.0.2"
        assert agreeing[0].harness is True
        with pytest.raises(bake_mod.BakeError, match="needs a decision"):
            self._resolve(
                [bake_mod.dep_ref("pytest==5.4.3", "install_requires (setup.py) line 4")],
                harness_pins={"pytest": "8.0.2"},
            )

    def test_a_constraint_that_excludes_the_harness_version_needs_a_decision(self):
        """A range names no release, so a range that excludes the harness is fatal."""
        refs = [bake_mod.dep_ref("pytest<5", "test_requirements.txt line 2")]
        with pytest.raises(
            bake_mod.BakeError, match="will not re-pin a package it provides"
        ):
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

    # ---- the BugsInPy reference-pin exception -----------------------------
    def test_a_bugsinpy_pin_is_installed_as_written_even_when_it_postdates_the_commit(self):
        """A pip-freeze-style ``requirements.txt`` *is* the reference environment.

        black/22's pins are from 2020 and its buggy commit is from 2018. The date rule
        would refuse every one of them, which would make the benchmark's own frozen
        environment unbakeable; the pin is what ``run_test.sh`` was validated against.
        """
        refs = [bake_mod.dep_ref("six==1.13.0", "requirements.txt line 12")]
        resolved = self._resolve(refs)
        assert resolved[0].pin == "six==1.13.0"
        assert resolved[0].exempt is True
        assert resolved[0].released_at == "2020-05-31"
        assert resolved[0].pin_rule == "BugsInPy's own exact pin, installed as written"
        assert resolved[0].as_dict()["pinned_by"] == "requirements.txt line 12"

    def test_a_setup_sh_pin_is_a_bugsinpy_pin_too(self):
        """Both of BugsInPy's per-bug files are the benchmark's environment."""
        refs = [bake_mod.dep_ref("decorator==4.4.1", "setup.sh line 2")]
        assert self._resolve(refs)[0].pin == "decorator==4.4.1"

    def test_a_range_from_bugsinpys_own_files_is_not_exempt(self):
        """Only an exact pin says which release the reference environment had."""
        refs = [bake_mod.dep_ref("six>=1.13.0", "requirements.txt line 12")]
        with pytest.raises(bake_mod.BakeError, match="no release on or before 2019-04-22"):
            self._resolve(refs)

    def test_the_same_pin_from_the_project_still_obeys_the_date_rule(self):
        refs = [bake_mod.dep_ref("six==1.13.0", "install_requires (setup.py) line 4")]
        with pytest.raises(bake_mod.BakeError, match="no release on or before 2019-04-22"):
            self._resolve(refs)

    def test_an_exempt_pin_naming_no_release_is_an_error_not_a_guess(self):
        refs = [bake_mod.dep_ref("six==9.9.9", "requirements.txt line 12")]
        with pytest.raises(bake_mod.BakeError, match="PyPI lists no such release of 'six'"):
            self._resolve(refs)

    def test_a_pin_inside_the_window_records_the_same_release_date(self):
        """The record is not only for late pins: every pin carries its release date."""
        refs = [bake_mod.dep_ref("decorator==4.4.0", "requirements.txt line 3")]
        resolved = self._resolve(refs)
        assert resolved[0].pin == "decorator==4.4.0"
        assert resolved[0].released_at == "2019-03-16"


class TestTheReportPluginFollowsTheBugsPytest:
    """A bug's own pytest decides which report plugin its layer can carry (§3.2).

    [VANTA-WHY] ``pytest-json-report`` declares a *floor* on pytest, and from 0.5.0 on
    it declares ``pytest-metadata``, whose own floor moves with the years: installed
    without a bound pip takes 3.1.1, which requires pytest 7. A layer that installs
    the bug's own ``pytest==5.4.3`` over the interpreter layer's pair (fastapi/1,
    luigi/1, pandas/1, sanic/1) would therefore end up with a plugin that refuses its
    own pytest, and the report the oracle is read from would never be written — the
    ``pip check`` signal was ``pytest-metadata 3.1.1 has requirement pytest>=7.0.0,
    but you have pytest 5.4.3``.

    So the plugin's release is chosen from the releases' **own** metadata, newest
    first, against the bug's pytest and the layer's interpreter; the choice is
    recorded (``chosen_for_pytest``); and ``pytest-metadata`` is pinned only when the
    chosen plugin release actually asks for it. Every value in ``METADATA`` below is
    what PyPI's per-release ``info`` says, so the rule is tested against the packages
    rather than against a table this file made up.
    """

    RELEASES = {
        "pytest-json-report": {
            "0.2": "2018-01-18",
            "0.3": "2018-01-19",
            "0.4": "2018-01-21",
            "1.5.0": "2022-03-15",
        },
        "pytest-metadata": {
            "1.11.0": "2020-11-27",
            "2.0.4": "2022-10-30",
            "3.1.1": "2024-02-12",
        },
    }

    METADATA = {
        # ``(requires_dist, requires_python)`` per release, measured from PyPI.
        ("pytest-json-report", "0.2"): ((), None),
        ("pytest-json-report", "0.3"): (("pytest (>=3.3.2)",), None),
        ("pytest-json-report", "0.4"): (("pytest (>=3.3.2)",), None),
        ("pytest-json-report", "1.5.0"): (("pytest (>=3.8.0)", "pytest-metadata"), None),
        ("pytest-metadata", "1.11.0"): (
            ("pytest (>=2.9.0)",),
            ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
        ),
        ("pytest-metadata", "2.0.4"): (("pytest (>=3.0.0,<8.0.0)",), ">=3.7,<4.0"),
        ("pytest-metadata", "3.1.1"): (("pytest>=7.0.0",), ">=3.8"),
    }

    @staticmethod
    def _metadata_of(name: str, version: str):
        return TestTheReportPluginFollowsTheBugsPytest.METADATA[(name, version)]

    def _plugins(self, pytest_version: str, *, python_minor: str, taken=None):
        return bake_mod.plugin_pins(
            pytest_version=pytest_version,
            python_minor=python_minor,
            releases=self.RELEASES,
            metadata_of=self._metadata_of,
            taken=taken,
        )

    def test_a_pytest_five_bug_gets_the_newest_plugin_that_still_accepts_it(self):
        """pytest 5.4.3 on 3.8: the plugin 1.5.0, and metadata 2.0.4 rather than 3.1.1.

        3.1.1 declares ``pytest>=7.0.0`` and is what an unbounded ``pytest-metadata``
        resolves to; 2.0.4 declares ``pytest>=3.0.0,<8.0.0`` with ``Requires-Python``
        ``>=3.7,<4.0``, so it is the newest release that can coexist with the layer.
        """
        pins = self._plugins("5.4.3", python_minor="3.8")
        assert [pin.pin for pin in pins] == [
            "pytest-json-report==1.5.0",
            "pytest-metadata==2.0.4",
        ]
        for pin in pins:
            assert pin.chosen_for_pytest == "5.4.3"
            assert pin.harness is False
        assert pins[0].released_at == "2022-03-15"
        assert pins[0].pin_rule == (
            "the newest release whose declared pytest requirement accepts pytest==5.4.3"
        )
        assert pins[0].as_dict()["chosen_for_pytest"] == "5.4.3"
        assert pins[1].as_dict()["provisioned_by"] == "pinned at bake time"

    def test_pytest_metadata_is_skipped_when_the_chosen_plugin_does_not_ask_for_it(self):
        """httpie/1: ``pytest==3.2.1`` caps the plugin at 0.2, which declares nothing.

        0.3 and later all require pytest 3.3.2 or newer, so 0.2 is the only release
        that accepts this bug's pytest — and its own metadata names no
        ``pytest-metadata``, so none is installed. Left to pip, an unbounded metadata
        would arrive anyway and refuse pytest 3.2.1.
        """
        pins = self._plugins("3.2.1", python_minor="3.7")
        assert [pin.pin for pin in pins] == ["pytest-json-report==0.2"]
        assert pins[0].pin_rule == (
            "the newest release whose declared pytest requirement accepts pytest==3.2.1"
        )

    def test_a_python_36_layer_gets_the_metadata_release_that_supports_it(self):
        """tqdm/1: 2.0.4's ``Requires-Python`` excludes 3.6, so the answer is older.

        This is the half of the rule that is not about pytest at all: pip would refuse
        the release outright, and the layer would have no metadata plugin.
        """
        pins = self._plugins("5.4.3", python_minor="3.6")
        assert [pin.pin for pin in pins] == [
            "pytest-json-report==1.5.0",
            "pytest-metadata==1.11.0",
        ]

    def test_a_plugin_the_bugs_own_files_pinned_is_not_chosen_twice(self):
        """``taken`` is every name the bug's own files already decided.

        A bug whose ``requirements.txt`` carries ``pytest-json-report==1.5.0`` gets
        that version installed by its own pin, so choosing one here would be a second
        claim about the same package — but the release's declared requirements still
        have to be read, or ``pytest-metadata`` would be left to pip.
        """
        assert self._plugins("5.4.3", python_minor="3.8", taken={"pytest-json-report": "1.5.0"}) == [
            pin
            for pin in self._plugins("5.4.3", python_minor="3.8")
            if pin.name == "pytest-metadata"
        ]
        # 0.4 declares only pytest, so nothing is pulled in and nothing is chosen.
        assert self._plugins("5.4.3", python_minor="3.8", taken={"pytest-json-report": "0.4"}) == []

    def test_no_plugin_release_accepting_the_bugs_pytest_is_an_error(self):
        """A guess here would be a layer whose report the oracle cannot read."""
        with pytest.raises(bake_mod.BakeError, match="declares a pytest requirement that accepts"):
            bake_mod.plugin_pins(
                pytest_version="3.2.1",
                python_minor="3.7",
                releases={"pytest-json-report": self.RELEASES["pytest-json-report"] | {"0.2": "2018-01-18"}},
                metadata_of=lambda name, version: (
                    ("pytest (>=3.3.2)",),
                    None,
                ),
            )

    def test_a_plugin_with_no_fetched_releases_is_an_error(self):
        with pytest.raises(bake_mod.BakeError, match="no release dates for 'pytest-json-report'"):
            bake_mod.plugin_pins(
                pytest_version="5.4.3",
                python_minor="3.8",
                releases={},
                metadata_of=self._metadata_of,
            )

    # ---- which pytest does the bug declare itself? ------------------------
    def test_own_pytest_pin_reads_an_exact_pin_from_bugsinpys_own_files(self):
        refs = [
            bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1"),
            bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"),
        ]
        assert bake_mod.own_pytest_pin(refs, harness_pytest="8.0.2") == "5.4.3"

    def test_own_pytest_pin_ignores_a_range(self):
        """spacy/4's ``pytest>=4.6.5`` names no release for the layer to install."""
        refs = [bake_mod.dep_ref("pytest>=4.6.5", "test_requirements.txt line 3")]
        assert bake_mod.own_pytest_pin(refs, harness_pytest="8.0.2") is None

    def test_own_pytest_pin_ignores_a_pin_equal_to_the_harness_version(self):
        """PySnooper/3's ``requirements.txt`` names the harness version, and that is a no-op."""
        refs = [bake_mod.dep_ref("pytest==8.0.2", "requirements.txt line 7")]
        assert bake_mod.own_pytest_pin(refs, harness_pytest="8.0.2") is None

    def test_own_pytest_pin_ignores_a_project_pin_on_pytest(self):
        """Only BugsInPy's own files are the reference environment (§3.2)."""
        refs = [bake_mod.dep_ref("pytest==5.4.3", "tests_require (setup.py) line 20")]
        assert bake_mod.own_pytest_pin(refs, harness_pytest="8.0.2") is None

    # ---- the whole chain, one bug ----------------------------------------
    @staticmethod
    def _sources(*refs: bake_mod.DepRef) -> bake_mod.DepSources:
        return bake_mod.DepSources(
            project="Demo",
            bug_id=1,
            checkpoint="vanta-bi-Demo-1",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            fixed_commit="f" * 40,
            fixed_commit_date="2019-07-30",
            import_name="demo",
            refs=tuple(refs),
        )

    def _resolve(self, *refs: bake_mod.DepRef, python_minor: str = "3.8"):
        releases = dict(TestResolvePinsByDate.RELEASES)
        releases.update(self.RELEASES)
        return bake_mod.resolve_deps(
            self._sources(*refs),
            releases=releases,
            harness_pins={"pytest": "8.0.2", "pytest-json-report": "1.5.0"},
            python_minor=python_minor,
            metadata_of=self._metadata_of,
        )

    def test_resolve_deps_adds_the_plugin_pins_to_the_bugs_own_pytest(self):
        """Both halves in one list: the bug's pytest *and* the plugin that accepts it."""
        resolved = self._resolve(
            bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1"),
            bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"),
        )
        assert [dep.pin for dep in resolved] == [
            "decorator==4.4.0",
            "pytest==5.4.3",
            "pytest-json-report==1.5.0",
            "pytest-metadata==2.0.4",
        ]
        assert bake_mod.installs_of(resolved) == (
            "decorator==4.4.0",
            "pytest==5.4.3",
            "pytest-json-report==1.5.0",
            "pytest-metadata==2.0.4",
        )

    def test_resolve_deps_leaves_a_bug_without_its_own_pin_alone(self):
        """PySnooper/3, black/1-2, matplotlib/1-2, cookiecutter/1, youtube-dl/1.

        No pin of their own, so the interpreter layer's pair stands and there is no
        plugin release to choose — the harness's own pin is the layer's.
        """
        resolved = self._resolve(
            bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1")
        )
        assert [dep.pin for dep in resolved] == ["decorator==4.4.0"]

    def test_resolve_deps_does_not_re_choose_a_plugin_the_bug_pinned_itself(self):
        resolved = self._resolve(
            bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"),
            bake_mod.dep_ref("pytest-json-report==0.4", "requirements.txt line 20"),
        )
        assert [dep.pin for dep in resolved] == [
            "pytest==5.4.3",
            "pytest-json-report==0.4",
        ]

    # ---- what the layers record and the plan shows ------------------------
    def test_layer_pin_reads_the_version_the_layer_actually_installs(self):
        """The split's entry declares the *interpreter* layer's pair, not the bug's.

        ``runner.build_record`` reads the run's row back through this, so a row that
        named the split's ``pytest_version`` would name a version the suite never ran
        under.
        """
        sources = self._sources(bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"))
        resolved = self._resolve(bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"))
        provenance = bake_mod.deps_provenance(
            sources,
            resolved,
            harness_pins={"pytest": "8.0.2", "pytest-json-report": "1.5.0"},
        )
        assert bake_mod.layer_pin(provenance, "pytest") == "5.4.3"
        assert bake_mod.layer_pin(provenance, "pytest-json-report") == "1.5.0"
        assert bake_mod.layer_pin(provenance, "pytest-metadata") == "2.0.4"
        assert bake_mod.layer_pin(provenance, "decorator") is None

    def test_layer_pin_reads_the_interpreter_layers_pair_when_the_bug_has_none(self):
        """A bug with no pin of its own: the interpreter layer's pair stands.

        Neither package is in ``dependencies`` here — the bug's sources never named
        them — so the answer comes from ``harness_pins``, which is the pair the
        interpreter layer was built with. Reporting ``None`` instead would drop the
        version from every such run's row.
        """
        sources = self._sources(bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1"))
        resolved = self._resolve(
            bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1")
        )
        provenance = bake_mod.deps_provenance(
            sources,
            resolved,
            harness_pins={"pytest": "8.0.2", "pytest-json-report": "1.5.0"},
        )
        assert bake_mod.layer_pin(provenance, "pytest") == "8.0.2"
        assert bake_mod.layer_pin(provenance, "pytest-json-report") == "1.5.0"
        assert bake_mod.layer_pin(provenance, "six") is None

    def test_layer_pin_reads_a_range_that_resolved_to_the_interpreter_layer(self):
        """spacy/4's ``pytest>=4.6.5`` is named, and §3.2 left it to the harness."""
        sources = self._sources(bake_mod.dep_ref("pytest>=4.6.5", "test_requirements.txt line 3"))
        resolved = self._resolve(
            bake_mod.dep_ref("pytest>=4.6.5", "test_requirements.txt line 3")
        )
        provenance = bake_mod.deps_provenance(
            sources,
            resolved,
            harness_pins={"pytest": "8.0.2", "pytest-json-report": "1.5.0"},
        )
        by_name = {dep["name"]: dep for dep in provenance["dependencies"]}
        assert by_name["pytest"]["provisioned_by"] == "interpreter layer"
        assert bake_mod.layer_pin(provenance, "pytest") == "8.0.2"

    @staticmethod
    def _clone(tmp_path: Path, requirements: str, setup_sh: str = "") -> Path:
        """One bug's two BugsInPy files, UTF-16 with a BOM as the real ones are."""
        bug_dir = tmp_path / "projects" / "Demo" / "bugs" / "1"
        bug_dir.mkdir(parents=True, exist_ok=True)
        (bug_dir / "requirements.txt").write_bytes(requirements.encode("utf-16"))
        (bug_dir / "setup.sh").write_bytes(setup_sh.encode("utf-16"))
        return tmp_path

    def test_bugsinpy_refs_reads_both_of_bugsinpys_files_without_a_repo(self, tmp_path):
        """The plan path has no clone and no network, so it reads these two alone."""
        root = self._clone(
            tmp_path,
            "decorator==4.4.1\npytest==5.4.3\n",
            "pip install six==1.12.0\npython setup.py install\n",
        )
        refs = bake_mod.bugsinpy_refs(bug_entry(), bugsinpy_root=root)
        assert [ref.spec for ref in refs] == [
            "decorator==4.4.1",
            "pytest==5.4.3",
            "six==1.12.0",
        ]
        assert bake_mod.own_pytest_pin(refs, harness_pytest="8.0.2") == "5.4.3"

    def test_the_plan_shows_the_bugs_own_pytest_and_defers_the_plugin(self, tmp_path):
        """A dry run has to name the pytest the layer will carry (§3.2).

        The ``json-report`` cell reads ``at bake``: which release the plugin can be is
        PyPI's answer, taken when the layer is built, and printing a guess would be a
        second, unrecorded decision about the same question.
        """
        root = self._clone(tmp_path, "pytest==5.4.3\n")
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        plan = bake_mod.render_plan(
            split,
            pythons=["3.8"],
            entries=[split.bug("Demo", 1)],
            bugsinpy_root=root,
        )
        assert "| vanta-bugsinpy-py3.8 | vanta-python | 3.8 | 8.0.2 | 1.5.0 |" in plan
        assert "| vanta-bi-Demo-1 | vanta-bugsinpy-py3.8 | 3.8.1 | 5.4.3 | at bake |" in plan


class TestTheFetchIncludesTheBugsOwnPytest:
    """A bug's exact ``pytest==X`` needs pytest's own release dates fetched.

    [VANTA-WHY] Measured on fastapi/1, luigi/1, pandas/1 and sanic/1: the bake failed
    with ``no release dates for 'pytest' were fetched; the bake cannot pin it``. The
    new rule is that the bug's own exact ``==`` pytest pin wins over the interpreter
    layer's, and that pin is a BugsInPy reference pin — exempt from the date rule, so
    :func:`resolve_pins` installs it as written. It still reads ``releases["pytest"]``,
    though: the exemption *records* the date the release landed and refuses a pin whose
    release does not exist. The fetch set excluded every :data:`HARNESS_PACKAGES` name
    and pytest is one, so the exemption and the fetch contradicted each other.
    """

    def _releases(self) -> dict:
        releases = dict(TestResolvePinsByDate.RELEASES)
        releases.update(TestTheReportPluginFollowsTheBugsPytest.RELEASES)
        return releases

    def _resolve(self, monkeypatch, *, refs):
        fetched: list[str] = []

        def fake_sources(entry, *, bugsinpy_root, repo, module_distributions=None):
            return bake_mod.DepSources(
                project="Demo",
                bug_id=1,
                checkpoint="vanta-bi-Demo-1",
                buggy_commit="b" * 40,
                commit_date="2019-04-22",
                fixed_commit="f" * 40,
                fixed_commit_date="2019-07-30",
                import_name="demo",
                refs=tuple(refs),
            )

        def fetch(name):
            fetched.append(name)
            return self._releases()[name]

        monkeypatch.setattr(bake_mod, "bug_sources", fake_sources)
        split = split_mod.Split(split_data(bugs=[bug_entry(checkpoint="vanta-bi-Demo-1")]))
        layers, flagged = bake_mod.resolve_bug_deps(
            split,
            split.bugs,
            bugsinpy_root=Path("/nonexistent"),
            repo_for=lambda project: Path("/nonexistent"),
            fetch=fetch,
            metadata_of=TestTheReportPluginFollowsTheBugsPytest._metadata_of,
        )
        return fetched, layers, flagged

    def test_the_bugs_own_pytest_is_fetched_so_its_pin_can_be_dated(self, monkeypatch):
        """The whole chain, with the fetch recorded rather than assumed."""
        fetched, layers, flagged = self._resolve(
            monkeypatch,
            refs=[
                bake_mod.dep_ref(
                    "decorator>=4.3.0", "install_requires (setup.py) line 1"
                ),
                bake_mod.dep_ref("pytest==5.4.3", "requirements.txt line 7"),
            ],
        )
        assert flagged == []
        assert "pytest" in fetched
        # The plugin releases follow from the same pin, so they are fetched too.
        assert "pytest-json-report" in fetched
        assert "pytest-metadata" in fetched
        assert layers[0].installs == (
            "decorator==4.4.0",
            "pytest==5.4.3",
            "pytest-json-report==1.5.0",
            "pytest-metadata==2.0.4",
        )

    def test_a_bug_without_its_own_pin_fetches_none_of_the_three(self, monkeypatch):
        """PySnooper/3, matplotlib/1-2 and the rest keep the interpreter layer's pair."""
        fetched, layers, flagged = self._resolve(
            monkeypatch,
            refs=[
                bake_mod.dep_ref(
                    "decorator>=4.3.0", "install_requires (setup.py) line 1"
                )
            ],
        )
        assert flagged == []
        assert "pytest" not in fetched
        assert "pytest-json-report" not in fetched
        assert "pytest-metadata" not in fetched
        assert layers[0].installs == ("decorator==4.4.0",)


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

    # ---- an exact pin narrows a bound instead of conflicting with it ------
    def test_an_exact_pin_intersects_with_the_bound_it_satisfies(self):
        """black/22: ``attrs==19.3.0`` in requirements.txt, ``attrs>=17.4.0`` in setup.py.

        The pin is inside the bound, so the requirement *is* the pin. The source that
        wrote the pin travels with the merged ref: it is what decides whether §3.2's
        date rule applies.
        """
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("attrs==19.3.0", "requirements.txt line 7"),
                bake_mod.dep_ref("attrs>=17.4.0", "install_requires (setup.py) line 43"),
            ]
        )
        assert conflicts == []
        assert [ref.spec for ref in merged] == ["attrs==19.3.0"]
        assert merged[0].pin_source == "requirements.txt line 7"
        assert bake_mod.reference_pin(merged[0]) is True

    def test_a_bugsinpys_pin_that_a_project_bound_excludes_wins_and_is_recorded(self):
        """luigi/1: ``tornado==5.0`` in BugsInPy's ``requirements.txt``, ``>=4.0,<5`` in setup.py.

        The two are jointly unsatisfiable, so one of them has to lose. The pin wins —
        it is the environment BugsInPy's own ``run_test.sh`` was validated
        fail-before/pass-after in — and the bound it beat is *recorded* rather than
        raised, so the override is visible in the provenance instead of looking like a
        harness that never read ``setup.py``.
        """
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("tornado==5.0", "requirements.txt line 137"),
                bake_mod.dep_ref("tornado>=4.0,<5", "install_requires (setup.py) line 41"),
                # The test side imports it bare, which is how luigi/1 reaches tornado
                # even without the two declarations. A bare name decides nothing.
                bake_mod.dep_ref("tornado", "import closure: test/server_test.py line 34"),
            ]
        )
        assert conflicts == []
        assert [ref.spec for ref in merged] == ["tornado==5.0"]
        assert merged[0].pin_source == "requirements.txt line 137"
        assert bake_mod.reference_pin(merged[0]) is True
        assert merged[0].overridden == (
            ("tornado>=4.0,<5", "install_requires (setup.py) line 41"),
        )
        # Both sources are still named: the override beats the claim, not the record.
        assert "install_requires (setup.py) line 41" in merged[0].source

    def test_a_project_pin_that_violates_another_sources_bound_is_still_a_conflict(self):
        """The licence is BugsInPy's, not "whoever pinned first".

        A pin the *project* wrote is subject to §3.2's date rule like any other
        requirement, and when it contradicts another of the project's own
        declarations nothing in the harness can say which one the buggy tree was built
        with — so it stays a conflict.
        """
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("attrs==17.1.0", "install_requires (setup.py) line 43"),
                bake_mod.dep_ref("attrs>=17.4.0", "tests_require (setup.py) line 7"),
            ]
        )
        assert merged == []
        assert conflicts[0].name == "attrs"
        assert conflicts[0].claims == (
            ("attrs==17.1.0", "install_requires (setup.py) line 43"),
            ("attrs>=17.4.0", "tests_require (setup.py) line 7"),
        )

    def test_two_different_pins_are_a_conflict(self):
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("six==1.11.0", "requirements.txt line 3"),
                bake_mod.dep_ref("six==1.12.0", "install_requires (setup.py) line 9"),
            ]
        )
        assert merged == []
        assert conflicts[0].name == "six"

    def test_a_project_pin_beside_bugsinpys_bound_is_not_a_reference_pin(self):
        """The pin's own source decides — not "did any of these sources pin it".

        A pin the project wrote is subject to the date rule even when BugsInPy declared
        the same package as a range; otherwise a project pin would inherit the
        benchmark's exemption.
        """
        merged, conflicts = bake_mod.merge_refs(
            [
                bake_mod.dep_ref("attrs>=17.4.0", "requirements.txt line 7"),
                bake_mod.dep_ref("attrs==19.3.0", "install_requires (setup.py) line 43"),
            ]
        )
        assert conflicts == []
        assert merged[0].spec == "attrs==19.3.0"
        assert merged[0].pin_source == "install_requires (setup.py) line 43"
        assert bake_mod.reference_pin(merged[0]) is False

    def test_a_wildcard_equality_is_a_bound_not_a_pin(self):
        """``==1.2.*`` matches a range of releases, so it pins nothing."""
        merged, conflicts = bake_mod.merge_refs(
            [bake_mod.dep_ref("six==1.12.*", "requirements.txt line 3")]
        )
        assert conflicts == []
        assert merged[0].pin_source is None
        assert bake_mod.reference_pin(merged[0]) is False

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


class TestUnionAcrossCommits:
    """A bug's dependencies are the union of what each commit declares (§3.2).

    [VANTA-WHY] The graded tree is a mixture of the two commits (§1.2): the source
    under test is the buggy commit's and the test side is the fix's. A requirement the
    *fix* added is therefore a requirement of the tree a run is graded on — but it was
    never in the buggy tree either, so it is held to the fix's own date instead of
    being back-dated to the buggy commit's. A requirement the *buggy* commit declared
    keeps the buggy date: the fix dropping it does not license an upgrade.
    """

    BUGGY = "b" * 40
    FIXED = "f" * 40
    BUGGY_DATE = "2019-04-22"
    FIXED_DATE = "2019-07-30"

    def _sources(self, *refs: bake_mod.DepRef) -> bake_mod.DepSources:
        return bake_mod.DepSources(
            project="Demo",
            bug_id=1,
            checkpoint="vanta-bi-Demo-1",
            buggy_commit=self.BUGGY,
            commit_date=self.BUGGY_DATE,
            fixed_commit=self.FIXED,
            fixed_commit_date=self.FIXED_DATE,
            import_name="demo",
            refs=tuple(refs),
        )

    def _ref(self, spec: str, source: str, *, fixed: bool) -> bake_mod.DepRef:
        return replace(
            bake_mod.dep_ref(spec, source),
            commit=self.FIXED if fixed else self.BUGGY,
            commit_date=self.FIXED_DATE if fixed else self.BUGGY_DATE,
        )

    def _resolve(self, *refs: bake_mod.DepRef) -> list[bake_mod.ResolvedDep]:
        return bake_mod.resolve_deps(
            self._sources(*refs), releases=TestResolvePinsByDate.RELEASES, harness_pins={}
        )

    def _provenance(self, *refs: bake_mod.DepRef) -> dict:
        sources = self._sources(*refs)
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={}
        )
        return bake_mod.deps_provenance(sources, resolved, harness_pins={})

    def test_a_requirement_only_the_fix_declares_is_held_to_the_fixed_date(self):
        """decorator 4.4.1 (2019-06-01) postdates the buggy commit; the fix's date allows it."""
        (dep,) = self._resolve(
            self._ref("decorator>=4.3.0", "install_requires (setup.py) line 1", fixed=True)
        )
        assert dep.pin == "decorator==4.4.1"
        assert dep.fixed is True
        assert dep.commit == self.FIXED
        assert dep.commit_date == self.FIXED_DATE
        assert dep.pin_rule == "the newest release on or before the fixed commit's date"

    def test_the_same_spec_on_the_buggy_commit_stops_at_the_buggy_date(self):
        """The cutoff is per declaration, not per bug: 4.4.1 is out of reach here."""
        (dep,) = self._resolve(
            self._ref("decorator>=4.3.0", "install_requires (setup.py) line 1", fixed=False)
        )
        assert dep.pin == "decorator==4.4.0"
        assert dep.fixed is False
        assert dep.commit == self.BUGGY
        assert dep.pin_rule == "the newest release on or before the buggy commit's date"

    def test_one_resolve_holds_each_ref_to_its_own_commit(self):
        """Both cutoffs in one call: the buggy ref is not silently upgraded by the fix."""
        by_name = {
            dep.name: dep
            for dep in self._resolve(
                self._ref("decorator>=4.3.0", "install_requires (setup.py) line 1", fixed=True),
                self._ref("future>=0.15.0", "install_requires (setup.py) line 2", fixed=False),
            )
        }
        assert by_name["decorator"].pin == "decorator==4.4.1"
        assert by_name["future"].pin == "future==0.17.1"

    def test_a_package_both_commits_declare_is_one_entry(self):
        """One requirement written twice is one requirement — the union, then the merge."""
        resolved = self._resolve(
            self._ref("six", "install_requires (setup.py) line 3", fixed=False),
            self._ref("six", "tests_require (setup.py) line 20", fixed=True),
        )
        assert len(resolved) == 1
        assert resolved[0].pin == "six==1.12.0"
        assert "install_requires (setup.py) line 3" in resolved[0].sources[0]
        assert "tests_require (setup.py) line 20" in resolved[0].sources[0]

    def test_the_two_commits_disagreeing_on_a_pin_fails_loudly(self):
        sources = self._sources(
            self._ref("six==1.11.0", "install_requires (setup.py) line 3", fixed=False),
            self._ref("six==1.12.0", "install_requires (setup.py) line 3", fixed=True),
        )
        with pytest.raises(bake_mod.BakeError) as excinfo:
            bake_mod.resolve_deps(
                sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={}
            )
        message = str(excinfo.value)
        assert "Demo/1" in message
        assert "'six==1.11.0'" in message
        assert "'six==1.12.0'" in message
        assert "does not pick one of them" in message

    def test_every_entry_records_the_commit_its_declaration_came_from(self):
        """The rule is per declaration, so the record has to name the declaration."""
        provenance = self._provenance(
            self._ref("decorator>=4.3.0", "install_requires (setup.py) line 1", fixed=True),
            self._ref("future>=0.15.0", "install_requires (setup.py) line 2", fixed=False),
        )
        by_name = {dep["name"]: dep for dep in provenance["dependencies"]}
        assert by_name["decorator"]["commit"] == self.FIXED
        assert by_name["decorator"]["commit_date"] == self.FIXED_DATE
        assert (
            by_name["decorator"]["pin_rule"]
            == "the newest release on or before the fixed commit's date"
        )
        assert by_name["future"]["commit"] == self.BUGGY
        assert by_name["future"]["commit_date"] == self.BUGGY_DATE
        assert provenance["fixed_commit"] == self.FIXED
        assert provenance["fixed_commit_date"] == self.FIXED_DATE

    def test_a_fix_only_pin_newer_than_the_buggy_commit_is_recorded_as_late(self):
        """The second expected reason a pin postdates the buggy tree, queryable."""
        provenance = self._provenance(
            self._ref("decorator>=4.3.0", "install_requires (setup.py) line 1", fixed=True)
        )
        assert provenance["installs"] == ["decorator==4.4.1"]
        assert provenance["pins_after_commit_date"] == [
            {
                "name": "decorator",
                "pin": "decorator==4.4.1",
                "released_at": "2019-06-01",
                "pin_rule": "the newest release on or before the fixed commit's date",
            }
        ]

    def test_without_the_buggy_sha_a_stamped_ref_counts_as_the_fix(self):
        """The documented default for a caller that passes no ``buggy_commit``."""
        ref = replace(
            bake_mod.dep_ref("decorator>=4.3.0", "install_requires (setup.py) line 1"),
            commit=self.FIXED,
            commit_date=self.FIXED_DATE,
        )
        (dep,) = bake_mod.resolve_pins(
            [ref],
            releases=TestResolvePinsByDate.RELEASES,
            commit_date=self.BUGGY_DATE,
            harness_pins={},
        )
        assert dep.pin == "decorator==4.4.1"
        assert dep.fixed is True


def fake_clone(monkeypatch, *, files: dict, dates: dict, requirements="", setup_sh="") -> None:
    """Point ``bug_sources`` at a dict-backed clone: ``(commit, path) -> text``.

    A real clone lives under ``runs/``, which is not committed, and what these tests
    measure is the *wiring* — which commit's file is read, and that both are — not
    git. Nothing is executed: handing back the text is the whole of the clone.
    """
    monkeypatch.setattr(
        bake_mod.tree_mod,
        "read_file_at",
        lambda repo, commit, path: files.get((commit, path)),
    )
    monkeypatch.setattr(
        bake_mod.tree_mod, "commit_date", lambda repo, commit: dates[commit]
    )
    monkeypatch.setattr(
        bake_mod, "requirements_for", lambda root, project, bug_id: requirements
    )
    monkeypatch.setattr(bake_mod, "setup_sh_for", lambda root, project, bug_id: setup_sh)
    monkeypatch.setattr(bake_mod, "test_side_paths", lambda repo, commit: [])
    monkeypatch.setattr(bake_mod, "tree_provided_modules", lambda repo, commit: set())
    monkeypatch.setattr(bake_mod, "test_import_closure", lambda **kwargs: ([], [], []))


class TestPyprojectAcrossCommits:
    """A ``pyproject.toml`` bug is read at both commits and unioned like any other.

    [VANTA-WHY] ``fastapi/1``'s refusal was not a missing file but a missing reader:
    the file was there, at both commits, and §3.2's union rule applies to it exactly
    as it does to ``setup.py``. A pyproject-only reader that stopped at the buggy
    commit would under-provision the graded tree, which is a mixture of the two
    (§1.2) — and the *fix* is where a requirement the bug's own tests need tends to
    be declared.
    """

    BUGGY = "b" * 40
    FIXED = "f" * 40
    BUGGY_DATE = "2019-04-22"
    FIXED_DATE = "2019-07-30"

    BUGGY_TOML = (
        "[tool.flit.metadata]\n"
        'module = "demo"\n'
        'requires = ["decorator >=4.3.0"]\n'
        "[tool.flit.metadata.requires-extra]\n"
        'test = ["requests"]\n'
    )
    FIXED_TOML = (
        "[tool.flit.metadata]\n"
        'module = "demo"\n'
        'requires = ["decorator >=4.3.0", "six >=1.12.0"]\n'
        "[tool.flit.metadata.requires-extra]\n"
        'test = ["responses"]\n'
    )
    RELEASES = {
        "decorator": {"4.3.0": "2018-02-03", "4.4.0": "2019-03-16", "4.4.1": "2019-06-01"},
        "six": {"1.12.0": "2018-12-10"},
        "requests": {"2.21.0": "2019-01-01"},
        "responses": {"0.10.6": "2019-05-21"},
    }

    def _sources(self, monkeypatch, *, buggy_toml=None, fixed_toml=None, setup=None):
        files = {}
        if buggy_toml is not None:
            files[(self.BUGGY, "pyproject.toml")] = buggy_toml
        if fixed_toml is not None:
            files[(self.FIXED, "pyproject.toml")] = fixed_toml
        if setup is not None:
            files[(self.BUGGY, "setup.py")] = setup
        fake_clone(
            monkeypatch,
            files=files,
            dates={self.BUGGY: self.BUGGY_DATE, self.FIXED: self.FIXED_DATE},
        )
        return bake_mod.bug_sources(
            bug_entry(checkpoint="vanta-bi-Demo-1"),
            bugsinpy_root=Path("/nonexistent"),
            repo=Path("/nonexistent"),
        )

    def _resolved(self, sources) -> dict:
        resolved = bake_mod.resolve_deps(
            sources, releases=self.RELEASES, harness_pins={}
        )
        return {dep.name: dep for dep in resolved}

    def test_a_pyproject_only_bug_is_read_rather_than_refused(self, monkeypatch):
        sources = self._sources(
            monkeypatch, buggy_toml=self.BUGGY_TOML, fixed_toml=self.FIXED_TOML
        )
        assert sources.import_name == "demo"
        assert {ref.name for ref in sources.refs} == {
            "decorator",
            "six",
            "requests",
            "responses",
        }

    def test_both_commits_are_read_and_each_ref_carries_its_own_commit(self, monkeypatch):
        sources = self._sources(
            monkeypatch, buggy_toml=self.BUGGY_TOML, fixed_toml=self.FIXED_TOML
        )
        by_name = {ref.name: ref for ref in sources.refs}
        assert by_name["requests"].commit == self.BUGGY
        assert by_name["requests"].commit_date == self.BUGGY_DATE
        assert by_name["responses"].commit == self.FIXED
        assert by_name["responses"].commit_date == self.FIXED_DATE

    def test_each_ref_is_pinned_by_the_date_of_the_commit_that_declared_it(self, monkeypatch):
        """One resolve, two cutoffs — and the fix's own declaration is reachable."""
        sources = self._sources(
            monkeypatch, buggy_toml=self.BUGGY_TOML, fixed_toml=self.FIXED_TOML
        )
        by_name = self._resolved(sources)
        assert by_name["decorator"].pin == "decorator==4.4.0"
        assert by_name["decorator"].pin_rule == (
            "the newest release on or before the buggy commit's date"
        )
        assert by_name["six"].pin == "six==1.12.0"
        assert by_name["six"].pin_rule == (
            "the newest release on or before the fixed commit's date"
        )
        assert by_name["responses"].pin == "responses==0.10.6", (
            "declared only by the fix, and released after the buggy commit's date: "
            "reading the fix is what makes it reachable at all"
        )

    def test_a_commit_with_neither_file_keeps_todays_refusal(self, monkeypatch):
        with pytest.raises(bake_mod.BakeError) as excinfo:
            self._sources(monkeypatch)
        message = str(excinfo.value)
        assert "neither setup.py nor pyproject.toml" in message
        assert "Flag this bug rather than guessing" in message

    def test_a_commit_carrying_both_files_has_both_read(self, monkeypatch):
        """A project mid-migration declares in both; reading one would under-provision."""
        sources = self._sources(
            monkeypatch,
            buggy_toml='[project]\nname = "demo"\ndependencies = ["future"]\n',
            fixed_toml='[project]\nname = "demo"\ndependencies = ["future"]\n',
            setup="setup(name='Demo', install_requires=['six>=1.12.0'])\n",
        )
        by_name = {ref.name: ref for ref in sources.refs}
        assert by_name["six"].source == "install_requires (setup.py) line 1"
        assert by_name["future"].source == "project.dependencies (pyproject.toml)"
        assert sources.import_name == "demo"


class TestImportPackageOverride:
    """``import_package`` is the last resort when the metadata cannot name the package.

    [VANTA-WHY] Sanic's ``setup.py`` is read by the ast reader (§3.2's fourth source),
    and in sanic/1's real file the reader now resolves ``name``/``packages`` from the
    module-level dict. The override exists for the shape the reader still cannot follow
    — a metadata dict built by a function — and only then: a bug whose ``setup.py``
    names its package is never overridden, and the note says the metadata was not what
    named it, so the value is never mistaken for a fact the file stated.
    """

    BUGGY = "b" * 40
    FIXED = "f" * 40
    UNREADABLE_SETUP = "setup(**make_kwargs())\n"

    def _sources(self, monkeypatch, entry):
        fake_clone(
            monkeypatch,
            files={(self.BUGGY, "setup.py"): self.UNREADABLE_SETUP},
            dates={self.BUGGY: "2019-04-22", self.FIXED: "2019-07-30"},
        )
        return bake_mod.bug_sources(
            entry, bugsinpy_root=Path("/nonexistent"), repo=Path("/nonexistent")
        )

    def test_the_override_is_used_when_the_metadata_cannot_name_the_package(self, monkeypatch):
        sources = self._sources(monkeypatch, bug_entry(import_package="sanic"))
        assert sources.import_name == "sanic"
        assert any("import_package='sanic'" in note for note in sources.notes)

    def test_without_the_override_the_bug_is_still_flagged(self, monkeypatch):
        with pytest.raises(bake_mod.BakeError, match="neither name= nor a packages="):
            self._sources(monkeypatch, bug_entry())

    def test_the_override_is_normalised_the_way_a_name_literal_is(self, monkeypatch):
        sources = self._sources(monkeypatch, bug_entry(import_package="PySnooper"))
        assert sources.import_name == "pysnooper"


class TestTestImportClosure:
    """The fifth source: the test side's own imports, read at the fixed commit (§3.2).

    [VANTA-WHY] PySnooper/2's fix vendors ``tests/mini_toolbox/pathlib.py``, which does
    ``import six`` at module scope. ``six`` is named by no requirements file, no
    ``setup.sh`` and no ``extras_require`` at either commit, and ``python-toolbox`` —
    whose sources the helper was copied from — declares no dependencies of its own, so
    the union of the four declarations is a no-op for that bug. The tree under test
    imports it, so the layer has to have it, and the test side's own source is the only
    place that fact exists.

    Read with ``ast`` and never by importing: these are third-party files and nothing
    third-party runs on the host (design C3). The whole walk is offline — no network, no
    VM, and no 3.8 interpreter, since ``stdlib`` is a parameter.
    """

    def _closure(
        self,
        files: dict,
        *,
        import_name: str = "pysnooper",
        distributions: dict | None = None,
        stdlib=None,
        tree_files: dict | None = None,
    ) -> tuple:
        return bake_mod.test_import_closure(
            files=sorted(files),
            read_file=files.get,
            import_name=import_name,
            distributions=distributions,
            stdlib=bake_mod.PY38_STDLIB if stdlib is None else stdlib,
            tree_modules=bake_mod._tree_local_modules(sorted(tree_files or {})),
        )

    def test_a_module_the_tree_ships_outside_the_test_side_is_not_a_distribution(self):
        """fastapi/1's measured case: ``docs_src`` goes on ``sys.path`` by hand.

        ``scripts/test.sh`` does ``export PYTHONPATH=./docs_src``, so a tutorial test's
        ``from additional_responses.tutorial001 import app`` is the repo's own
        directory. Nothing there is a declaration, so the closure emitted
        ``additional-responses`` and the bake died asking PyPI for it (404).
        """
        refs, optional, _ = self._closure(
            {
                "tests/test_tutorial/test_additional_responses/test_tutorial001.py": (
                    "from additional_responses.tutorial001 import app\n"
                )
            },
            tree_files={
                "docs_src/additional_responses/tutorial001.py": "",
                "tests/test_tutorial/test_additional_responses/test_tutorial001.py": "",
            },
        )
        assert refs == []
        assert optional == []

    def test_a_file_inside_a_package_does_not_make_its_stem_a_module(self):
        """``fastapi/requests.py`` is ``fastapi.requests``, never a top-level ``requests``.

        Only directories that ship ``.py`` are modules the tree provides, so the real
        distribution is still provisioned — the whole reason the rule is not
        "any name the tree mentions".
        """
        refs, _, _ = self._closure(
            {"tests/test_demo.py": "import requests\n"},
            tree_files={"fastapi/__init__.py": "", "fastapi/requests.py": ""},
        )
        assert [ref.spec for ref in refs] == ["requests"]

    def test_a_third_party_import_no_declaration_names_is_a_dependency(self):
        refs, optional, notes = self._closure(
            {"tests/mini_toolbox/pathlib.py": "import six\n"}
        )
        assert optional == []
        assert [ref.name for ref in refs] == ["six"]
        assert refs[0].spec == "six"
        assert refs[0].source == "import closure: tests/mini_toolbox/pathlib.py line 1"

    def test_the_source_names_the_file_and_the_line_it_was_found_on(self):
        refs, _, _ = self._closure({"tests/test_demo.py": "import os\n\nimport six\n"})
        assert [ref.source for ref in refs] == ["import closure: tests/test_demo.py line 3"]

    def test_a_dotted_import_is_recorded_under_its_top_level_module(self):
        refs, _, _ = self._closure({"tests/test_demo.py": "import six.moves\n"})
        assert [ref.spec for ref in refs] == ["six"]

    def test_the_same_module_imported_twice_in_one_file_is_one_dependency(self):
        refs, _, _ = self._closure(
            {"tests/test_demo.py": "import six\nfrom six import text_type\n"}
        )
        assert [ref.spec for ref in refs] == ["six"]

    def test_the_interpreter_stdlib_is_not_a_dependency(self):
        """3.8 has no ``sys.stdlib_module_names``, so the set travels as a literal."""
        refs, _, _ = self._closure({"tests/test_demo.py": "import os\nimport json\n"})
        assert refs == []
        refs, _, _ = self._closure(
            {"tests/test_demo.py": "import os\nimport json\n"}, stdlib=("os",)
        )
        assert [ref.spec for ref in refs] == ["json"]

    def test_the_project_is_not_a_dependency_of_itself(self):
        refs, _, _ = self._closure({"tests/test_demo.py": "import pysnooper\n"})
        assert refs == []

    def test_a_test_local_module_is_not_a_distribution(self):
        """``from tests import utils`` and ``import mini_toolbox`` are the tree's own files."""
        refs, optional, _ = self._closure(
            {
                "tests/test_demo.py": (
                    "from . import helper\nfrom tests import utils\nimport mini_toolbox\n"
                ),
                "tests/helper.py": "",
                "tests/mini_toolbox/__init__.py": "",
            }
        )
        assert refs == []
        assert optional == []

    def test_an_import_behind_an_importerror_guard_is_optional_and_not_installed(self):
        """A py2-only fallback must not break the bake over a branch 3.8 never takes."""
        refs, optional, _ = self._closure(
            {"tests/test_demo.py": "try:\n    import scandir\nexcept ImportError:\n    pass\n"}
        )
        assert refs == []
        assert optional == [
            bake_mod.OptionalImport(
                module="scandir",
                source="import closure: tests/test_demo.py line 2",
                reason="behind an ImportError guard",
            )
        ]

    def test_an_import_behind_a_version_check_is_optional(self):
        refs, optional, _ = self._closure(
            {"tests/test_demo.py": "import sys\nif sys.version_info < (3,):\n    import pathlib2\n"}
        )
        assert refs == []
        assert [item.module for item in optional] == ["pathlib2"]
        assert optional[0].reason == "behind a version check"

    def test_a_relative_import_is_the_trees_own_file(self):
        refs, optional, _ = self._closure({"tests/test_demo.py": "from . import helper\n"})
        assert refs == []
        assert optional == []

    def test_an_ambiguous_module_name_stops_the_bake_rather_than_guessing(self):
        with pytest.raises(bake_mod.BakeError, match="module_distributions"):
            self._closure({"tests/test_demo.py": "import yaml\n"})

    def test_a_reviewed_mapping_resolves_an_ambiguous_name(self):
        refs, _, _ = self._closure(
            {"tests/test_demo.py": "import yaml\n"}, distributions={"yaml": "PyYAML"}
        )
        assert refs[0].name == "pyyaml"
        assert refs[0].spec == "yaml"

    def test_a_dotted_reviewed_mapping_is_matched_before_its_top_level(self):
        """``google.protobuf`` is protobuf; ``google`` alone is a distribution too.

        [VANTA-WHY] The reviewed table is keyed by the import spelling, so the
        longest match has to win: a dotted import that fell through to its top-level
        word would install ``google`` — a real but unrelated release — in protobuf's
        place, and the layer would be wrong in a way no later stage can see.
        """
        assert (
            bake_mod.module_distribution("google.protobuf", distributions={"google.protobuf": "protobuf"})
            == "protobuf"
        )
        # The fallback is the top-level word, and only for names the table does not key.
        assert bake_mod.module_distribution("google.protobuf", distributions={}) == "google"
        assert bake_mod.module_distribution("six.moves", distributions={}) == "six"

    def test_a_reviewed_mapping_wins_over_the_ambiguous_refusal(self):
        """The table is consulted first, so a mapped ambiguous name never stops the bake."""
        assert (
            bake_mod.module_distribution("PIL", distributions={"PIL": "Pillow"}) == "pillow"
        )
        with pytest.raises(bake_mod.BakeError, match="module_distributions"):
            bake_mod.module_distribution("PIL", distributions={})

    def test_a_module_whose_name_cannot_be_a_distribution_is_its_interpreters(self):
        """``_csv`` is CPython's own module, so nothing is installed for it.

        [VANTA-WHY] pandas/1's ``from _csv import Error`` reached the merge as
        ``-csv`` (the canonicalised spelling) and stopped the bug over a name no one
        can publish — a PEP 508 name starts with a letter or a digit. It is the
        interpreter's accelerator for ``csv``, present in every 3.8 build, so the
        closure notes it and emits no reference rather than failing the bake.
        """
        assert bake_mod._can_name_a_distribution("_csv") is False
        assert bake_mod._can_name_a_distribution("csv") is True
        refs, optional, notes = self._closure(
            {"tests/test_demo.py": "from _csv import Error\n"}
        )
        assert refs == []
        assert optional == []
        assert any(
            "'_csv' cannot name a distribution" in note
            and "tests/test_demo.py line 1" in note
            for note in notes
        )

    def test_a_non_package_directory_ships_its_py_stems_as_modules(self):
        """matplotlib's ``lib/pylab.py`` imports as ``pylab`` (pytest inserts ``lib/``).

        [VANTA-WHY] PyPI publishes no ``pylab``, so counting the stem as a distribution
        stops matplotlib/1 with a 404 — the flag this rule removes. A ``.py`` directly
        in a directory that has no ``__init__.py`` is on ``sys.path`` as its own module;
        one *inside* a package is not, so ``fastapi/requests.py`` stays
        ``fastapi.requests`` and the real ``requests`` is still provisioned.
        """
        names = bake_mod._tree_local_modules(
            sorted(
                {
                    "lib/pylab.py": "",
                    "lib/matplotlib/__init__.py": "",
                    "fastapi/__init__.py": "",
                    "fastapi/requests.py": "",
                    "scripts/tests/test_validate_docstrings.py": "",
                    "scripts/validate_docstrings.py": "",
                }
            )
        )
        assert "pylab" in names, "a non-package directory's .py stem is a module"
        assert "validate_docstrings" in names
        assert "requests" not in names, "a .py inside a package is not a top-level module"
        assert "__init__" not in names

    def test_a_file_that_does_not_parse_is_noted_and_skipped(self):
        """The target interpreter could not collect it either, so its imports are not ours."""
        refs, _, notes = self._closure(
            {"tests/broken.py": "def f(:\n", "tests/test_demo.py": "import six\n"}
        )
        assert [ref.spec for ref in refs] == ["six"]
        assert any("tests/broken.py does not parse as Python 3" in note for note in notes)

    def test_a_file_in_the_tree_that_cannot_be_read_is_noted(self):
        refs, _, notes = self._closure(
            {"tests/gone.py": None, "tests/test_demo.py": "import six\n"}
        )
        assert [ref.spec for ref in refs] == ["six"]
        assert any("tests/gone.py is in the tree but could not be read" in n for n in notes)

    def test_future_imports_are_not_a_dependency(self):
        refs, _, _ = self._closure(
            {"tests/test_demo.py": "from __future__ import annotations\n"}
        )
        assert refs == []


class TestBuildPinsByDate:
    """The build-time toolchain is pinned by date like a dependency (§3.2).

    [VANTA-WHY] Measured on matplotlib/1: the pinned ``numpy==1.18.5`` has no wheel
    for the sandbox (linux/aarch64, py3.8), so pip built it in an *isolated* build
    environment that took setuptools released years after the tree — and numpy's
    ``numpy/distutils/mingw32ccompiler.py`` died on a ``distutils`` submodule that
    setuptools no longer carries. The rule these tests pin down: the toolchain is
    dated to the bug's commit, installed first, and the builds run against it.
    """

    TOOLCHAIN = {
        "setuptools": {"40.8.0": "2019-02-04", "41.0.0": "2019-06-01", "50.0.0": "2020-08-09"},
        "wheel": {"0.32.3": "2018-10-21", "0.33.1": "2019-04-06", "0.35.1": "2019-07-25"},
        "cython": {"0.29.6": "2019-03-10", "0.29.7": "2019-05-02"},
    }

    def _pins(self, refs=(), resolved=(), *, releases=None, commit_date="2019-04-22"):
        return bake_mod.resolve_build_pins(
            list(refs),
            list(resolved),
            releases=releases if releases is not None else dict(self.TOOLCHAIN),
            commit_date=commit_date,
        )

    def test_the_toolchain_gets_a_pin_for_every_bug_even_with_nothing_declared(self):
        """No project has to ask for setuptools, wheel and Cython: they are the answer
        to "what was this tree built with", and every layer gets them."""
        pins = self._pins()
        assert [pin.pin for pin in pins] == [
            "setuptools==40.8.0",
            "wheel==0.33.1",
            "cython==0.29.6",
        ]
        assert [pin.released_at for pin in pins] == [
            "2019-02-04",
            "2019-04-06",
            "2019-03-10",
        ]
        assert {pin.source for pin in pins} == {bake_mod.BUILD_TOOL_SOURCE}
        assert all(pin.pin_rule == bake_mod.BUILD_PIN_RULE for pin in pins)

    def test_a_release_newer_than_the_commit_is_excluded_even_when_it_is_newest(self):
        """setuptools 41.0.0 exists and the buggy tree could not have been built with it."""
        pins = {pin.name: pin for pin in self._pins()}
        assert pins["setuptools"].pin == "setuptools==40.8.0"

    def test_a_declared_build_requirement_is_held_to_the_commit_that_declared_it(self):
        """pandas' ``_libs`` are compiled against numpy, and ``pyproject.toml`` says so.

        The cutoff is the declaring commit's, exactly as for a dependency: a build
        requirement only the *fix* declares was not there for the buggy tree, and the
        buggy date would refuse the very numpy the fix's own build used.
        """
        releases = dict(self.TOOLCHAIN)
        releases["numpy"] = {
            "1.16.4": "2019-04-01",
            "1.17.0": "2019-07-26",
            "1.18.0": "2019-12-22",
        }
        at_buggy = bake_mod.DepRef(
            name="numpy", spec="numpy>=1.15", source="setup_requires (setup.py) line 275"
        )
        at_fixed = bake_mod.DepRef(
            name="numpy",
            spec="numpy>=1.15",
            source="build-system.requires (pyproject.toml)",
            commit="f" * 40,
            commit_date="2019-08-01",
        )
        buggy_view = {pin.name: pin for pin in self._pins([at_buggy], releases=releases)}
        fixed_view = {pin.name: pin for pin in self._pins([at_fixed], releases=releases)}
        assert buggy_view["numpy"].pin == "numpy==1.16.4"
        assert fixed_view["numpy"].pin == "numpy==1.17.0"
        assert fixed_view["numpy"].source == "build-system.requires (pyproject.toml)"
        assert fixed_view["numpy"].released_at == "2019-07-26"

    def test_a_build_requirement_the_dependencies_already_pin_follows_that_pin(self):
        """One numpy in the layer, or ``pandas/_libs`` is built against a version that
        is immediately replaced. The record says which pin it followed."""
        resolved = [
            bake_mod.ResolvedDep(
                name="numpy",
                spec="numpy==1.15.4",
                sources=("requirements.txt line 3",),
                pin="numpy==1.15.4",
                harness=False,
                released_at="2018-08-30",
                exempt=True,
            )
        ]
        ref = bake_mod.DepRef(
            name="numpy", spec="numpy>=1.15", source="setup_requires (setup.py) line 275"
        )
        # Only the toolchain's releases are passed, and numpy is absent from them on
        # purpose: a followed pin needs no date of its own, so if this path dated
        # numpy the call would fail with "no release dates for 'numpy'" instead of
        # pinning it. Two versions of one package in one layer is the failure.
        numpy_pin = {
            pin.name: pin for pin in self._pins([ref], resolved, releases=self.TOOLCHAIN)
        }["numpy"]
        assert numpy_pin.pin == "numpy==1.15.4"
        assert numpy_pin.dependency_pin == "numpy==1.15.4"
        assert numpy_pin.source == "setup_requires (setup.py) line 275"
        assert numpy_pin.pin_rule == "the dependency set's pin for the same package"

    def test_a_build_tool_with_no_release_before_the_commit_date_flags_the_bug(self):
        """A tree cannot have been built with a tool that did not exist yet, and
        installing today's is the failure this rule exists to prevent."""
        releases = dict(self.TOOLCHAIN)
        releases["wheel"] = {"0.33.1": "2019-06-01"}
        with pytest.raises(
            bake_mod.BakeError, match="'wheel' has no release on or before 2019-04-22"
        ):
            self._pins(releases=releases)

    def test_a_build_tool_with_no_release_dates_at_all_is_not_guessed_at(self):
        with pytest.raises(bake_mod.BakeError, match="no release dates for 'setuptools'"):
            self._pins(releases={"setuptools": {}})

    def test_a_package_the_interpreter_layer_provides_is_not_a_build_pin(self):
        """pytest is the harness's, and a tree's build does not get to re-pin it."""
        ref = bake_mod.DepRef(
            name="pytest", spec="pytest>=4.4.1", source="test_requirements.txt line 2"
        )
        # The releases mapping has no pytest in it at all: if the name were not skipped
        # this would raise, which is the assertion.
        assert [pin.name for pin in self._pins([ref])] == [
            "setuptools",
            "wheel",
            "cython",
        ]

    def test_the_declared_build_requirements_are_pinned_after_the_toolchain(self):
        """Order is the install order, so the toolchain comes first on the command line."""
        refs = [
            bake_mod.DepRef(
                name="numpy",
                spec="numpy>=1.15",
                source="build-system.requires (pyproject.toml)",
            ),
            bake_mod.DepRef(
                name="cython",
                spec="cython>=0.28",
                source="setup_requires (setup.py) line 271",
            ),
        ]
        releases = dict(self.TOOLCHAIN)
        releases["numpy"] = {"1.16.4": "2019-04-01"}
        pins = self._pins(refs, releases=releases)
        assert [pin.name for pin in pins] == ["setuptools", "wheel", "cython", "numpy"]
        # A toolchain name the project also declares records the declaration as its
        # source: "cython is here because this setup.py names it" says more than
        # "cython is here for every bug", and the toolchain rule still applied to the
        # pin's version — one pin, two reasons, and the more specific one recorded.
        assert pins[2].source == "setup_requires (setup.py) line 271"
        assert pins[2].pin == "cython==0.29.6"
        assert pins[3].source == "build-system.requires (pyproject.toml)"

    def test_build_installs_is_the_pins_in_install_order(self):
        assert bake_mod.build_installs(self._pins()) == (
            "setuptools==40.8.0",
            "wheel==0.33.1",
            "cython==0.29.6",
        )


class TestNeedsBuildIsARule:
    """``needs_build`` is read off the tree, in both directions (§3.2).

    [VANTA-WHY] Measured, and the reason this rule exists: the dev split declared
    ``needs_build: false`` for matplotlib/1 and pandas/1, whose tests import a package
    the tree compiles itself. The first thing that would have happened is the
    dependency gate's import probe failing with "does not import from the tree" — a
    dependency problem that is not a dependency problem, which is what §3.3 exists to
    keep apart. The old rule was ``info.pythonpath is not None``, which could only ever
    see tqdm and cookiecutter, where BugsInPy's metadata already said "build".
    """

    EXTENSION_SETUP = (
        "from setuptools import setup, Extension\n"
        "setup(name='demo', ext_modules=[Extension('demo._lib', [])])\n"
    )
    PLAIN_SETUP = "from setuptools import setup\nsetup(name='demo', packages=['demo'])\n"

    def test_a_pythonpath_into_the_tree_is_a_build_on_its_own(self):
        needs, why = bake_mod.needs_build([], None, pythonpath="demo/build/lib/")
        assert needs is True
        assert "pythonpath='demo/build/lib/'" in why

    def test_declared_extensions_plus_sources_in_the_tree_is_a_build(self):
        needs, why = bake_mod.needs_build(
            ["setup.py", "pandas/__init__.py", "pandas/_libs/algos.pyx"],
            self.EXTENSION_SETUP,
        )
        assert needs is True
        assert "1 extension source" in why and "pandas/_libs/algos.pyx" in why

    def test_the_sources_count_wherever_they_sit(self):
        """matplotlib's are in ``extern/`` (the vendored Agg) and ``src/``, both built."""
        needs, _ = bake_mod.needs_build(
            ["setup.py", "extern/agg24-svn/src/agg_arc.cpp", "src/_path_wrapper.cpp"],
            self.EXTENSION_SETUP,
        )
        assert needs is True

    def test_declared_extensions_with_no_sources_is_not_a_build(self):
        """black declares ``ext_modules``: the opt-in mypyc build of its own Python."""
        needs, why = bake_mod.needs_build(
            ["setup.py", "black/__init__.py"], self.EXTENSION_SETUP
        )
        assert needs is False
        assert "no extension sources" in why and "mypyc" in why

    def test_sources_with_no_declared_extensions_is_not_a_build(self):
        """tornado ships ``tornado/speedups.c`` and builds nothing: a stub since 5.0."""
        needs, why = bake_mod.needs_build(
            ["setup.py", "tornado/speedups.c"], self.PLAIN_SETUP
        )
        assert needs is False
        assert "no ext_modules" in why and "speedups.c" in why

    def test_a_pure_python_tree_is_not_a_build(self):
        needs, why = bake_mod.needs_build(
            ["setup.py", "demo/__init__.py"], self.PLAIN_SETUP
        )
        assert needs is False
        assert "no ext_modules" in why and "no extension sources" in why

    def test_a_cython_declaration_is_not_a_source(self):
        """``.pxd`` declares types for a compiler; it is not compiled itself."""
        needs, _ = bake_mod.needs_build(
            ["setup.py", "demo/attrs.pxd"], self.EXTENSION_SETUP
        )
        assert needs is False

    def test_a_tree_with_no_setup_py_is_read_rather_than_refused(self):
        """fastapi has no ``setup.py`` at all (flit): nothing is compiled, so no build."""
        needs, why = bake_mod.needs_build(["fastapi/__init__.py"], None)
        assert needs is False
        assert "no ext_modules" in why

    def test_the_reader_asks_the_clone_for_that_commit(self, monkeypatch):
        seen: dict = {}

        def fake_list(repo, commit):
            seen["list"] = (repo, commit)
            return ["setup.py", "demo/_lib.c"]

        def fake_read(repo, commit, path):
            seen["read"] = (repo, commit, path)
            return self.EXTENSION_SETUP

        monkeypatch.setattr(bake_mod.tree_mod, "list_files_at", fake_list)
        monkeypatch.setattr(bake_mod.tree_mod, "read_file_at", fake_read)
        needs, _ = bake_mod.needs_build_at(Path("/clones/demo.git"), "a" * 40)
        assert needs is True
        assert seen["list"] == (Path("/clones/demo.git"), "a" * 40)
        assert seen["read"] == (Path("/clones/demo.git"), "a" * 40, "setup.py")

    def test_an_unparseable_setup_py_is_an_error_not_a_silent_no(self):
        with pytest.raises(bake_mod.BakeError, match="does not parse"):
            bake_mod.needs_build(["setup.py"], "setup(name='demo'\n")


@needs_project_clones
class TestTheSplitsAgreeWithTheirTrees:
    """Both committed splits, checked against the trees they name (§3.2).

    [VANTA-WHY] This is the regression for the flag being hand-set: ``needs_build`` was
    ``info.pythonpath is not None``, so the four dev-split bugs whose trees compile C
    extensions of their own — matplotlib/1, matplotlib/2, pandas/1, spacy/4 — were
    declared ``false`` and nothing could disagree. Read through the clone, and every bug
    whose clone is present has to agree, so a *stale split* is a test failure rather
    than a prepare gate failure hours later.
    """

    def test_every_bug_in_both_splits_has_the_flag_its_tree_implies(self):
        checked = 0
        for path in (SPLIT_PATH, DEV_SPLIT_PATH):
            split = load_split(path)
            for entry in split.data["bugs"]:
                repo = PROJECT_CLONES / f"{entry['project']}.git"
                if not repo.is_dir():
                    continue
                bug = prepare_mod.load_bug(
                    path,
                    REPO_ROOT / split.data["bugsinpy_root"],
                    entry["project"],
                    int(entry["bug_id"]),
                )
                mismatches = prepare_mod.check_build_declared(entry, bug, repo=repo)
                assert mismatches == [], f"{path.name}: {entry['project']}/{entry['bug_id']}"
                checked += 1
        # A missing clone must not shrink this to nothing and still pass.
        assert checked >= 20, f"only {checked} bugs were checked"

    def test_the_trees_that_compile_extensions_are_declared(self):
        """The measured set, named: a rule that drifts must break these lines.

        ``bake.needs_build`` reads ``ext_modules`` at the buggy commit *and* a
        compiled extension source of the tree's own, so the dev split's three C
        trees are joined by the eval split's two, which BugsInPy itself builds out
        of a ``build/lib`` directory (``pythonpath``) before their tests can import.
        """
        for path, expected in (
            (SPLIT_PATH, {"cookiecutter/1", "tqdm/1"}),
            (DEV_SPLIT_PATH, {"matplotlib/1", "matplotlib/2", "pandas/1", "spacy/4"}),
        ):
            split = load_split(path)
            found = {
                f"{entry['project']}/{entry['bug_id']}"
                for entry in split.data["bugs"]
                if entry["needs_build"]
            }
            assert found == expected, f"{path.name}: {sorted(found)}"


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
        assert by_name["decorator"]["released_at"] == "2019-03-16"
        assert (
            by_name["decorator"]["pin_rule"]
            == "the newest release on or before the buggy commit's date"
        )
        assert by_name["pytest"]["pin"] == "pytest==8.0.2"
        assert by_name["pytest"]["provisioned_by"] == "interpreter layer"
        assert by_name["pytest"]["pin_rule"] == "provisioned by the interpreter layer"
        assert provenance["import_name"] == "demo"
        assert provenance["harness_pins"] == {"pytest": "8.0.2"}

    def test_a_reference_pin_released_after_the_buggy_commit_is_recorded_as_late(self):
        """The one departure from §3.2's date rule, as a queryable list.

        A pip-freeze-style bug is pinned to releases newer than its own commit, so
        ``pins_after_commit_date`` is where a reader finds out that a run's layer is
        newer than its tree — without re-deriving anything.
        """
        sources = bake_mod.DepSources(
            project="Demo",
            bug_id=4,
            checkpoint="vanta-bi-Demo-4",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            import_name="demo",
            refs=(bake_mod.dep_ref("six==1.13.0", "requirements.txt line 12"),),
        )
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={}
        )
        provenance = bake_mod.deps_provenance(sources, resolved, harness_pins={})
        assert provenance["installs"] == ["six==1.13.0"]
        assert provenance["pins_after_commit_date"] == [
            {
                "name": "six",
                "pin": "six==1.13.0",
                "released_at": "2020-05-31",
                "pin_rule": "BugsInPy's own exact pin, installed as written",
            }
        ]
        assert provenance["dependencies"][0]["pinned_by"] == "requirements.txt line 12"

    def test_a_reference_pin_that_beat_a_project_bound_records_what_it_beat(self):
        """The override is data in the record, not a sentence in a docstring.

        The layer installs one pin and the provenance says both where the pin came
        from and which declaration it contradicts — otherwise a reader comparing the
        record against ``setup.py`` finds a bound that was silently dropped.
        """
        sources = bake_mod.DepSources(
            project="Demo",
            bug_id=5,
            checkpoint="vanta-bi-Demo-5",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            import_name="demo",
            refs=(
                bake_mod.dep_ref("six==1.13.0", "requirements.txt line 12"),
                bake_mod.dep_ref("six<1.12", "install_requires (setup.py) line 9"),
            ),
        )
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={}
        )
        provenance = bake_mod.deps_provenance(sources, resolved, harness_pins={})
        assert provenance["installs"] == ["six==1.13.0"]
        dep = provenance["dependencies"][0]
        assert dep["sources"] == [
            "requirements.txt line 12; install_requires (setup.py) line 9"
        ]
        assert dep["pinned_by"] == "requirements.txt line 12"
        assert dep["pin_rule"] == "BugsInPy's own exact pin, installed as written"
        assert dep["overridden"] == [
            {"spec": "six<1.12", "source": "install_requires (setup.py) line 9"}
        ]

    def test_a_pin_inside_the_window_is_not_recorded_as_late(self):
        sources = self._sources()
        resolved = bake_mod.resolve_deps(
            sources, releases=TestResolvePinsByDate.RELEASES, harness_pins={"pytest": "8.0.2"}
        )
        provenance = bake_mod.deps_provenance(sources, resolved, harness_pins={"pytest": "8.0.2"})
        assert provenance["pins_after_commit_date"] == []

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

    def test_the_build_pins_are_recorded_with_what_asked_for_them(self):
        """The layer's *other* environment, and the numpy case the rule names.

        matplotlib declares ``setup_requires=["numpy>=1.15"]`` and depends on numpy;
        the layer can only hold one numpy, so the build pin is the dependency's pin —
        recorded as such, with the line that declared it.
        """
        sources = bake_mod.DepSources(
            project="Demo",
            bug_id=6,
            checkpoint="vanta-bi-Demo-6",
            buggy_commit="b" * 40,
            commit_date="2019-04-22",
            import_name="demo",
            refs=(bake_mod.dep_ref("numpy==1.15.4", "requirements.txt line 3"),),
            build_refs=(
                bake_mod.DepRef(
                    name="numpy",
                    spec="numpy>=1.15",
                    source="setup_requires (setup.py) line 275",
                ),
            ),
        )
        releases = {
            "numpy": {"1.15.4": "2018-08-30"},
            "setuptools": {"40.8.0": "2019-02-04"},
            "wheel": {"0.33.1": "2019-04-06"},
            "cython": {"0.29.6": "2019-03-10"},
        }
        resolved = bake_mod.resolve_deps(sources, releases=releases, harness_pins={})
        built = bake_mod.resolve_build_pins(
            sources.build_refs,
            resolved,
            releases=releases,
            commit_date=sources.commit_date,
        )
        provenance = bake_mod.deps_provenance(
            sources, resolved, harness_pins={}, build_pins=built
        )
        assert [pin["pin"] for pin in provenance["build_pins"]] == [
            "setuptools==40.8.0",
            "wheel==0.33.1",
            "cython==0.29.6",
            "numpy==1.15.4",
        ]
        by_name = {pin["name"]: pin for pin in provenance["build_pins"]}
        assert by_name["numpy"] == {
            "name": "numpy",
            "pin": "numpy==1.15.4",
            "source": "setup_requires (setup.py) line 275",
            "pin_rule": "the dependency set's pin for the same package",
            "released_at": "2018-08-30",
            "dependency_pin": "numpy==1.15.4",
        }
        assert by_name["setuptools"]["source"] == bake_mod.BUILD_TOOL_SOURCE
        assert by_name["setuptools"]["pin_rule"] == bake_mod.BUILD_PIN_RULE
        # And the dependency half is unchanged: numpy is installed once, as it was.
        assert provenance["installs"] == ["numpy==1.15.4"]

    def test_a_layer_with_no_build_pins_records_an_empty_list_not_a_missing_key(self):
        provenance = bake_mod.deps_provenance(self._sources(), [], harness_pins={})
        assert provenance["build_pins"] == []

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
        assert bake_mod.missing_deps(deps_dir, ["vanta-bi-Demo-3"]) == []

    def test_a_layer_with_no_recorded_provenance_must_be_rebaked(self, tmp_path):
        deps_dir = tmp_path / "deps"
        with pytest.raises(bake_mod.BakeError, match="re-run bake"):
            bake_mod.load_deps(deps_dir, "vanta-bi-Demo-3")
        # A missing record is reported, not raised on: it is one bug's problem and
        # prepare flags that bug rather than stopping the split (§3.2).
        assert bake_mod.missing_deps(deps_dir, ["vanta-bi-Demo-3"]) == ["vanta-bi-Demo-3"]

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
    """The measured bug, end to end: PySnooper 2-3 resolve to the right layers.

    PySnooper/3 is the failure this whole path was written for. Its layer used to
    be ``decorator`` alone, while ``tests/test_pysnooper.py`` imports
    ``python_toolbox``; prepare then reported a 0-test collection as a bug. The
    union of the four sources has to produce ``decorator``, ``future``, ``six`` and
    ``python-toolbox``, and must not install the project itself.
    """

    RELEASE_DATES = TestResolvePinsByDate.RELEASES

    def _entries(self) -> tuple[split_mod.Split, list[dict]]:
        split = load_split(SPLIT_PATH)
        three = split.bug("PySnooper", 3)
        # PySnooper/2 is no longer scheduled — it was dropped for a
        # `bug_patch.txt` that cannot pass the oracle (reviewed_changes[3]) — but
        # BugsInPy still ships it and bake's resolution for it is what these tests
        # measure, the vendored `import six` in particular. Its entry is rebuilt
        # from the clone's own `bug.info` so no commit is retyped, and it inherits
        # PySnooper/3's pins so the interpreter stays the one pin the split
        # declares.
        info = meta_mod.load_bug_info(BUGSINPY_CLONE, "PySnooper", 2)
        two = dict(three)
        two.update(
            bug_id=2,
            python_version=info.python_version,
            buggy_commit=info.buggy_commit,
            fixed_commit=info.fixed_commit,
            test_file=info.test_file,
            pythonpath=info.pythonpath,
            checkpoint="vanta-bi-PySnooper-2",
        )
        return split, [two, three]

    def _resolve(self, entries):
        return bake_mod.resolve_bug_deps(
            load_split(SPLIT_PATH),
            entries,
            bugsinpy_root=BUGSINPY_CLONE,
            repo_for=lambda project: REPO_ROOT / "runs" / "bugsinpy" / "repos" / f"{project}.git",
            fetch=lambda name: self.RELEASE_DATES[name],
        )

    def _layers(self, entries):
        layers, flagged = self._resolve(entries)
        assert flagged == [], f"PySnooper/2 and /3 must both resolve, got {flagged}"
        return layers

    def test_bug_three_installs_every_declared_dependency_and_not_the_project(self):
        _, entries = self._entries()
        item = next(d for d in self._layers(entries) if d.checkpoint.endswith("-3"))
        assert set(item.installs) == {
            "decorator==4.4.0",
            "future==0.17.1",
            "python-toolbox==0.9.3",
            "six==1.12.0",
        }

    def test_the_project_is_never_among_the_installs(self):
        """A copy of ``pysnooper`` in site-packages would make every patch grade the same."""
        _, entries = self._entries()
        for item in self._layers(entries):
            assert not any(spec.startswith("pysnooper") for spec in item.installs)

    def test_the_harness_pytest_is_recorded_but_never_installed_by_the_layer(self):
        """Both halves: out of the installs, still in the record.

        The layer must not re-install a package the interpreter layer provides (a
        downgrade underneath pytest-json-report would make the oracle's report
        unreadable), and the provenance must not claim the layer installed it
        either — a record that disagrees with the recipe is worse than no record.
        """
        _, entries = self._entries()
        item = next(d for d in self._layers(entries) if d.checkpoint.endswith("-3"))
        assert not any(spec.startswith("pytest==") for spec in item.installs)
        pytest_dep = next(
            dep for dep in item.provenance["dependencies"] if dep["name"] == "pytest"
        )
        assert pytest_dep["pin"] == "pytest==8.0.2"
        assert pytest_dep["provisioned_by"] == "interpreter layer"
        assert "pytest==8.0.2" not in item.provenance["installs"]
        assert item.provenance["installs"] == list(item.installs)
        assert item.provenance["harness_pins"]["pytest"] == "8.0.2"

    def test_bug_two_gets_python_toolbox_from_the_tests_extra(self):
        _, entries = self._entries()
        got = {
            item.checkpoint: set(item.installs)
            for item in self._layers(entries)
            if not item.checkpoint.endswith("-3")
        }
        assert got == {
            "vanta-bi-PySnooper-2": {"python-toolbox==0.9.3", "six==1.12.0"},
        }

    def test_bug_two_gets_six_from_the_test_sides_own_imports(self):
        """The vendored ``tests/mini_toolbox/pathlib.py`` does ``import six`` at module
        scope, and no declaration file at either commit names ``six``: the four-source
        union is a no-op for this bug. The fifth source is where it comes from, and it
        is held to the *fix's* date (2019-07-30), which is 1.12.0 — the buggy date would
        have allowed the same release, but the record has to say which declaration
        decided it."""
        _, entries = self._entries()
        item = next(d for d in self._layers(entries) if d.checkpoint.endswith("-2"))
        six = next(d for d in item.provenance["dependencies"] if d["name"] == "six")
        assert six["pin"] == "six==1.12.0"
        assert six["commit"] == "814abc34a098c1b98cb327105ac396f985d2413e"
        assert six["commit_date"] == "2019-07-30"
        assert six["pin_rule"] == "the newest release on or before the fixed commit's date"
        assert six["sources"] == ["import closure: tests/mini_toolbox/pathlib.py line 13"]
        assert "six==1.12.0" in item.installs

    def test_the_setup_py_install_line_is_recorded_as_skipped_on_every_bug(self):
        _, entries = self._entries()
        for item in self._layers(entries):
            skipped = item.provenance["skipped"]
            assert any(
                line["source"] == "setup.sh"
                and line["text"] == "python setup.py install"
                and "site-packages" in line["reason"]
                for line in skipped
            ), item.checkpoint

    def test_the_import_name_is_read_from_the_projects_own_setup_py(self):
        _, entries = self._entries()
        for item in self._layers(entries):
            assert item.provenance["import_name"] == "pysnooper"


needs_fastapi_repo = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "fastapi.git").is_dir(),
    reason="fastapi's bare clone is not present (runs/ is not committed)",
)


@needs_fastapi_repo
@needs_clone
class TestRealFastapiTutorialModules:
    """fastapi/1's flag, reproduced and fixed against the real files.

    [VANTA-WHY] The flag read "cannot read release dates for 'additional-responses'
    from PyPI: 404", which names the *consequence*: ``additional-responses`` is not a
    package anybody publishes, so something emitted a module that is not a
    distribution. The cause was the fifth source, not the reader — at the fixed commit
    ``tests/test_tutorial/test_additional_responses/test_tutorial001.py`` opens with
    ``from additional_responses.tutorial001 import app``, which is ``docs_src/``'s own
    directory (the repo's ``scripts/test.sh`` exports ``PYTHONPATH=./docs_src``), and
    the closure's local-module set only knew the test side. Both halves are asserted
    here so the diagnosis cannot drift back to "the reader picked up a TOML key".
    """

    BUGGY = "766157bfb4e7dfccba09ab398e8ec444d14e947c"
    FIXED = "3397d4d69a9c2d64c1219fcbf291ea5697a4abb8"
    REPO = REPO_ROOT / "runs" / "bugsinpy" / "repos" / "fastapi.git"
    TUTORIAL = "tests/test_tutorial/test_additional_responses/test_tutorial001.py"

    def _test_files(self) -> list[str]:
        files = tree_mod.list_files_at(self.REPO, self.FIXED)
        return [p for p in files if p.endswith(".py") and tree_mod.is_test_path(p)]

    def test_the_pyproject_reader_names_no_tutorial_directory_at_either_commit(self):
        """The file the flag pointed at: read at both commits and checked by name."""
        for commit in (self.BUGGY, self.FIXED):
            text = tree_mod.read_file_at(self.REPO, commit, "pyproject.toml")
            assert text is not None, commit
            names = {
                ref.name
                for ref in bake_mod.pyproject_refs(
                    text, [], commit=commit, commit_label="the commit"
                )
            }
            assert "additional-responses" not in names, commit
            assert {"starlette", "pydantic"} <= names, commit

    def test_the_tutorial_import_is_the_trees_own_directory_not_a_distribution(self):
        """Where ``additional-responses`` came from, and the one change that removes it."""
        test_files = self._test_files()
        naive, _, _ = bake_mod.test_import_closure(
            files=test_files,
            read_file=lambda path: tree_mod.read_file_at(self.REPO, self.FIXED, path),
            import_name="fastapi",
        )
        assert "additional_responses" in {
            ref.spec for ref in naive if ref.source.startswith(f"import closure: {self.TUTORIAL} ")
        }, "the tutorial import is what the closure used to emit"

        provided = bake_mod.tree_provided_modules(self.REPO, self.FIXED)
        assert "additional_responses" in provided
        refs, _, _ = bake_mod.test_import_closure(
            files=test_files,
            read_file=lambda path: tree_mod.read_file_at(self.REPO, self.FIXED, path),
            import_name="fastapi",
            tree_modules=provided,
        )
        names = {ref.spec for ref in refs}
        assert names == {
            "orjson",
            "pydantic",
            "pytest",
            "requests",
            "sqlalchemy",
            "starlette",
        }, sorted(names)
        assert "requests" in names, "fastapi/requests.py must not hide the distribution"

    def test_bug_sources_for_fastapi_one_has_no_tutorial_directory(self):
        """End to end through the wiring, not just the closure in isolation."""
        split = load_split(DEV_SPLIT_PATH)
        sources = bake_mod.bug_sources(
            split.bug("fastapi", 1),
            bugsinpy_root=BUGSINPY_CLONE,
            repo=self.REPO,
            module_distributions=split.module_distributions(),
        )
        names = {ref.name for ref in sources.refs}
        assert "additional-responses" not in names
        assert {
            ref.name
            for ref in sources.refs
            if ref.source.startswith("import closure:")
        } == {"orjson", "pydantic", "pytest", "requests", "sqlalchemy", "starlette"}
        assert {"requests", "starlette"} <= names


needs_luigi_repo = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "luigi.git").is_dir(),
    reason="luigi's bare clone is not present (runs/ is not committed)",
)


@needs_luigi_repo
@needs_clone
class TestRealLuigiTornadoPin:
    """luigi/1's flag, reproduced and settled against the real files.

    [VANTA-WHY] The flag read ``tornado: the declared sources disagree —
    'tornado==5.0' from requirements.txt line 137; 'tornado>=4.0,<5' from
    install_requires (setup.py) line 41``. Both claims are real and neither is a
    reader bug: BugsInPy's own ``requirements.txt`` pins exactly what the project's
    ``install_requires`` excludes, and the test side imports ``tornado`` bare besides.
    The resolution is therefore not a per-bug exception — it is §3.3's rule about
    BugsInPy's own pins — and it has to be checkable against the two files that
    produced it rather than against a fixture that paraphrases them.
    """

    BUGGY = "1164eb6b85b8a70f596dbb99452bec513e72c12e"
    FIXED = "aec5dc2ed8db53fc282a0bd24aabe59031b6d1ba"
    REPO = REPO_ROOT / "runs" / "bugsinpy" / "repos" / "luigi.git"
    # tornado's real release dates (the bake fetches these from PyPI).
    TORNADO = {
        "4.0.2": "2014-09-11",
        "4.2.1": "2015-07-17",
        "4.5.3": "2018-01-06",
        "5.0": "2018-03-05",
        "5.1": "2018-07-11",
    }

    def _merged(self):
        split = load_split(DEV_SPLIT_PATH)
        sources = bake_mod.bug_sources(
            split.bug("luigi", 1),
            bugsinpy_root=BUGSINPY_CLONE,
            repo=self.REPO,
            module_distributions=split.module_distributions(),
        )
        return bake_mod.merge_refs(sources.refs)

    def _tornado(self):
        merged, conflicts = self._merged()
        return next(ref for ref in merged if ref.name == "tornado"), conflicts

    def test_all_three_declarations_are_really_there(self):
        """The pin, the bound it contradicts, and the bare closure import."""
        tornado, conflicts = self._tornado()
        assert conflicts == []
        assert tornado.spec == "tornado==5.0"
        assert tornado.pin_source == "requirements.txt line 137"
        assert "install_requires (setup.py) line 41" in tornado.source
        assert "import closure: test/server_test.py line 34" in tornado.source

    def test_the_bugsinpy_pin_settles_it_instead_of_flagging_the_bug(self):
        tornado, _ = self._tornado()
        assert tornado.overridden == (
            ("tornado>=4.0,<5", "install_requires (setup.py) line 41"),
        )

    def test_the_pin_is_installed_as_written_and_the_override_travels_with_it(self):
        """End of the chain: what the layer installs, and what the record says.

        The merge is not allowed to settle a disagreement and then forget it, so the
        claim that lost is checkable in the provenance a later stage reads back.
        """
        tornado, _ = self._tornado()
        resolved = bake_mod.resolve_pins(
            [tornado],
            releases={"tornado": self.TORNADO},
            commit_date=tree_mod.commit_date(self.REPO, self.BUGGY),
            harness_pins={},
            buggy_commit=self.BUGGY,
        )
        assert resolved[0].pin == "tornado==5.0"
        assert resolved[0].exempt is True
        assert resolved[0].released_at == "2018-03-05"
        assert resolved[0].as_dict()["overridden"] == [
            {"spec": "tornado>=4.0,<5", "source": "install_requires (setup.py) line 41"}
        ]


needs_matplotlib_repo = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "matplotlib.git").is_dir(),
    reason="matplotlib's bare clone is not present (runs/ is not committed)",
)

needs_pandas_repo = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "pandas.git").is_dir(),
    reason="pandas' bare clone is not present (runs/ is not committed)",
)


class TestRealMatplotlibAndPandasClosure:
    """matplotlib/1's ``PIL`` and pandas/1's ``dateutil``, against the real files.

    [VANTA-WHY] Both flags are the fifth source: the test side imports a module whose
    distribution is a different word (``PIL`` is Pillow, ``dateutil`` is
    python-dateutil), and no canonicalisation recovers it, so the closure refuses to
    guess. The resolution is the split's reviewed ``module_distributions`` — the same
    decision for the evaluation waves — not a per-bug exception. pandas' ``psycopg2``
    is the same table used for a different reason: two distributions provide that
    module, so the entry is the reviewer's pick (the reason is in ``reviewed_changes``)
    and the harness still never guesses. The same two bugs
    exercise the other half of the source: matplotlib's ``lib/pylab.py`` and pandas'
    ``scripts/validate_docstrings.py`` live in directories with no ``__init__.py``, so
    pytest puts them on ``sys.path`` as modules the tree provides, and pandas'
    ``from _csv import Error`` is CPython's own. Every fact is asserted against the
    fixed commits the bake actually reads.
    """

    MATPLOTLIB_REPO = REPO_ROOT / "runs" / "bugsinpy" / "repos" / "matplotlib.git"
    PANDAS_REPO = REPO_ROOT / "runs" / "bugsinpy" / "repos" / "pandas.git"

    def _sources(self, project: str, bug_id: int, repo: Path):
        split = load_split(DEV_SPLIT_PATH)
        return split, bake_mod.bug_sources(
            split.bug(project, bug_id),
            bugsinpy_root=BUGSINPY_CLONE,
            repo=repo,
            module_distributions=split.module_distributions(),
        )

    @staticmethod
    def _by_spec(refs) -> dict:
        return {
            ref.spec: ref.name
            for ref in refs
            if ref.source.startswith("import closure:")
        }

    @needs_matplotlib_repo
    def test_matplotlib_resolves_pil_to_pillow_and_dateutil_to_python_dateutil(self):
        _, sources = self._sources("matplotlib", 1, self.MATPLOTLIB_REPO)
        by_spec = self._by_spec(sources.refs)
        assert by_spec["PIL"] == "pillow"
        assert by_spec["dateutil"] == "python-dateutil"
        assert "pylab" not in by_spec, "lib/pylab.py is the tree's own module"
        assert "mpl_toolkits" not in by_spec, "it ships as a package in matplotlib"

    @needs_matplotlib_repo
    def test_matplotlib_merges_without_a_conflict(self):
        _, sources = self._sources("matplotlib", 1, self.MATPLOTLIB_REPO)
        merged, conflicts = bake_mod.merge_refs(sources.refs)
        assert conflicts == []
        names = {ref.name for ref in merged}
        assert {"pillow", "python-dateutil", "numpy", "cycler"} <= names
        assert "pylab" not in names

    @needs_matplotlib_repo
    def test_without_the_reviewed_table_matplotlib_1_still_flags(self):
        """The mapping is what fixes it; unrecorded, the closure refuses (as today)."""
        split = load_split(DEV_SPLIT_PATH)
        with pytest.raises(bake_mod.BakeError, match="module_distributions"):
            bake_mod.bug_sources(
                split.bug("matplotlib", 1),
                bugsinpy_root=BUGSINPY_CLONE,
                repo=self.MATPLOTLIB_REPO,
                module_distributions={},
            )

    @needs_pandas_repo
    def test_pandas_resolves_the_five_divergent_modules(self):
        """``psycopg2`` included: the reviewer's pick is in the table, not the harness.

        [VANTA-WHY] Unlike the other four, ``psycopg2`` was never resolvable by
        canonicalisation *or* by preference: two distributions provide that module, so
        the closure refuses until a reviewer records which one — and it is now recorded
        (``psycopg2-binary``), with the reason in the split's ``reviewed_changes``. The
        table still decides it, so removing the entry puts the refusal back.
        """
        _, sources = self._sources("pandas", 1, self.PANDAS_REPO)
        by_spec = self._by_spec(sources.refs)
        assert by_spec["dateutil"] == "python-dateutil"
        assert by_spec["bs4"] == "beautifulsoup4"
        assert by_spec["sklearn"] == "scikit-learn"
        assert by_spec["pkg_resources"] == "setuptools"
        assert by_spec["psycopg2"] == "psycopg2-binary", "the reviewed pick, not a guess"

    @needs_pandas_repo
    def test_pandas_merges_without_a_conflict_and_csv_is_the_interpreters(self):
        split, sources = self._sources("pandas", 1, self.PANDAS_REPO)
        merged, conflicts = bake_mod.merge_refs(sources.refs)
        assert conflicts == []
        names = {ref.name for ref in merged}
        assert {"beautifulsoup4", "python-dateutil", "scikit-learn", "setuptools"} <= names
        assert "psycopg2-binary" in names, "mapped by the split; the source build needs libpq"
        assert "psycopg2" not in names, "the unmapped distribution is never installed"
        assert "-csv" not in names
        refs, _, notes = bake_mod.test_import_closure(
            files=bake_mod.test_side_paths(self.PANDAS_REPO, sources.fixed_commit),
            read_file=lambda path: tree_mod.read_file_at(
                self.PANDAS_REPO, sources.fixed_commit, path
            ),
            import_name=sources.import_name,
            distributions=split.module_distributions(),
            tree_modules=bake_mod.tree_provided_modules(
                self.PANDAS_REPO, sources.fixed_commit
            ),
        )
        assert "_csv" not in {ref.spec for ref in refs}
        assert any("'_csv' cannot name a distribution" in note for note in notes)


needs_sanic_repo = pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "bugsinpy" / "repos" / "sanic.git").is_dir(),
    reason="sanic's bare clone is not present (runs/ is not committed)",
)


@needs_sanic_repo
class TestRealSanicSetupKwargs:
    """sanic/1's flag, reproduced and fixed against the real ``setup.py``.

    [VANTA-WHY] The flag read ``setup.py declares neither name= nor a packages=[...]
    literal`` — true of the AST as the reader walked it, false as a fact about the bug,
    since sanic's ``setup.py`` opens ``setup_kwargs = {"name": "sanic", …}``, appends
    the requirement keys by subscript, and ends with ``setup(**setup_kwargs)``. The
    resolution is source 4's rule about a splatted dict literal, not a per-bug
    exception, so it is asserted against the file that produced it rather than against
    a fixture that paraphrases it.
    """

    BUGGY = "e7001b00747b659f7042b0534802b936ee8a53e0"
    FIXED = "44973125c15304b4262c51c78b5a86bd1daafa86"
    REPO = REPO_ROOT / "runs" / "bugsinpy" / "repos" / "sanic.git"

    def test_the_setup_py_names_its_package_through_the_kwargs_dict(self):
        setup_py = tree_mod.read_file_at(self.REPO, self.BUGGY, "setup.py")
        assert setup_py is not None
        assert "setup(**setup_kwargs)" in setup_py
        assert bake_mod.import_name_of(setup_py) == "sanic"

    def test_bug_sources_names_the_project_and_merges_without_a_conflict(self):
        split = load_split(DEV_SPLIT_PATH)
        sources = bake_mod.bug_sources(
            split.bug("sanic", 1),
            bugsinpy_root=BUGSINPY_CLONE,
            repo=self.REPO,
            module_distributions=split.module_distributions(),
        )
        assert sources.import_name == "sanic"
        _, conflicts = bake_mod.merge_refs(sources.refs)
        assert conflicts == []


class TestOneBugDoesNotStopTheBatch:
    """A bug whose dependencies cannot be read is flagged, never fatal (§3.2).

    [VANTA-WHY] ``fastapi/1``'s refusal stopped a whole dev-split ``bake`` at the
    first bug — which hides every later bug behind it and produces nothing to look
    at. ``resolve_bug_deps`` therefore returns ``(layers, flagged)``: the resolvable
    bugs are baked, the flagged checkpoints are kept out of the recipes, and the
    flagged list is printed *separately* from the §14.7 gate results, because
    flagged-unreadable is not one of those two gates and must not promote a reserve.
    """

    UNREADABLE = (
        "Demo/1: neither setup.py nor pyproject.toml at " + "b" * 40 + ", so the "
        "declared dependencies and test requirements cannot be read. Flag this bug "
        "rather than guessing at its dependencies (design §3.2)."
    )

    def _split(self) -> split_mod.Split:
        return split_mod.Split(
            split_data(
                bugs=[
                    bug_entry(checkpoint="vanta-bi-Demo-1"),
                    bug_entry(bug_id=2, checkpoint="vanta-bi-Demo-2"),
                ]
            )
        )

    def _resolve(self, monkeypatch, *, refusing=("Demo/1",)):
        def fake_sources(entry, *, bugsinpy_root, repo, module_distributions=None):
            if f"{entry['project']}/{entry['bug_id']}" in refusing:
                raise bake_mod.BakeError(self.UNREADABLE)
            return bake_mod.DepSources(
                project="Demo",
                bug_id=2,
                checkpoint="vanta-bi-Demo-2",
                buggy_commit="b" * 40,
                commit_date="2019-04-22",
                fixed_commit="f" * 40,
                fixed_commit_date="2019-07-30",
                import_name="demo",
                refs=(
                    replace(
                        bake_mod.dep_ref(
                            "decorator>=4.3.0", "install_requires (setup.py) line 1"
                        ),
                        commit="b" * 40,
                        commit_date="2019-04-22",
                    ),
                ),
            )

        monkeypatch.setattr(bake_mod, "bug_sources", fake_sources)
        split = self._split()
        return bake_mod.resolve_bug_deps(
            split,
            split.bugs,
            bugsinpy_root=Path("/nonexistent"),
            repo_for=lambda project: Path("/nonexistent"),
            fetch=lambda name: TestResolvePinsByDate.RELEASES[name],
        )

    def test_the_bug_after_the_unreadable_one_is_still_resolved(self, monkeypatch):
        layers, _ = self._resolve(monkeypatch)
        assert [item.checkpoint for item in layers] == ["vanta-bi-Demo-2"]
        assert layers[0].installs == ("decorator==4.4.0",)

    def test_the_unreadable_bug_is_flagged_with_its_reason(self, monkeypatch):
        layers, flagged = self._resolve(monkeypatch)
        assert layers, "the batch carried on"
        assert [(flag.key, flag.reason) for flag in flagged] == [
            ("Demo/1", f"BakeError: {self.UNREADABLE}")
        ]

    def test_a_split_where_every_bug_is_unreadable_still_returns_flags(self, monkeypatch):
        """Nothing resolvable is not an exception: it is two flags and no layers."""
        layers, flagged = self._resolve(monkeypatch, refusing=("Demo/1", "Demo/2"))
        assert layers == []
        assert [flag.key for flag in flagged] == ["Demo/1", "Demo/2"]

    def test_a_flagged_bug_is_kept_out_of_the_recipes(self, monkeypatch):
        """The exclusion ``cmd_bake`` performs — without it the flag would stop the bake."""
        layers, flagged = self._resolve(monkeypatch)
        split = self._split()
        with pytest.raises(bake_mod.BakeError, match="no resolved dependencies"):
            bake_mod.layer_recipes(
                split, pythons=["3.8"], entries=list(split.bugs), deps=layers
            )
        flagged_keys = {flag.key for flag in flagged}
        baked = [
            entry
            for entry in split.bugs
            if f"{entry['project']}/{entry['bug_id']}" not in flagged_keys
        ]
        names = [
            name
            for name, _, _ in bake_mod.layer_recipes(
                split, pythons=["3.8"], entries=baked, deps=layers
            )
        ]
        assert names == ["vanta-bugsinpy-py3.8", "vanta-bi-Demo-2"]

    def test_the_flagged_list_is_printed_apart_from_the_gate_results(self):
        text = run_mod.render_flagged(
            [("Demo/1", "BakeError: no setup.py ..."), ("FastAPI/1", "BakeError: b")]
        )
        assert "flagged bugs (2)" in text
        assert "not gate failures" in text, "a flag is not a §14.7 gate"
        assert "no reserve was promoted" in text, (
            "promoting a reserve for an unreadable bug is a hand decision"
        )
        assert "  - Demo/1: BakeError: no setup.py ..." in text
        assert "  - FastAPI/1: BakeError: b" in text

    def _bake_cli(self, tmp_path, monkeypatch, *, flagged, failure=None):
        """Drive ``bake --yes`` with the resolution and the build injected.

        The resolution is injected with a *resolvable* deps record, because a
        non-flagged entry always has one: ``layer_recipes`` refuses a checkpoint
        with nothing resolved (§3.2), and that refusal is its own test. A fake that
        returned no deps for a non-flagged bug would model a state the code
        deliberately cannot reach. ``write_deps`` is injected because the real one
        writes under ``runs/``.
        """
        root = tmp_path / "bugsinpy"
        root.mkdir()
        entry = bug_entry(checkpoint="vanta-bi-Demo-1")
        split = split_mod.Split(
            split_data(bugs=[entry], bugsinpy_root=str(root))
        )
        deps = _bug_deps([entry], {"vanta-bi-Demo-1": ("decorator==4.4.0",)})
        monkeypatch.setattr(run_mod, "load_split", lambda path: split)
        monkeypatch.setattr(
            bake_mod, "resolve_bug_deps", lambda *a, **k: (deps, list(flagged))
        )
        monkeypatch.setattr(bake_mod, "write_deps", lambda *a, **k: tmp_path / "d.json")
        monkeypatch.setattr(bake_mod, "plan_layers", lambda *a, **k: [])

        def build(*args, **kwargs):
            if failure is not None:
                raise failure
            return "summary line"

        monkeypatch.setattr(bake_mod, "run_bake_actions", build)
        return run_mod.main(
            ["bake", "--pythons", "3.8", "--yes", "--split", str(tmp_path / "s.json")]
        )

    def test_a_bake_that_fails_at_a_build_still_ends_with_the_flagged_list(
        self, tmp_path, monkeypatch, capsys
    ):
        """[VANTA-WHY] ``run_bake_actions`` raises on the first failed build.

        That is exactly the run whose reader has to know which bugs were flagged
        before it: the failure message names the layer that failed, and the flagged
        list is the other half of "what did this run not do".
        """
        code = self._bake_cli(
            tmp_path,
            monkeypatch,
            flagged=[bake_mod.FlaggedBug("Demo/1", "BakeError: unreadable")],
            failure=bake_mod.BakeError("baking vanta-bi-Demo-1 failed (exit 1)"),
        )
        captured = capsys.readouterr()
        assert "flagged bugs (1)" in captured.out
        assert "  - Demo/1: BakeError: unreadable" in captured.out
        assert "error: baking vanta-bi-Demo-1 failed" in captured.err
        assert code == run_mod.EXIT_HARNESS_FAILED

    def test_a_bake_that_flags_nothing_says_so(self, tmp_path, monkeypatch, capsys):
        """An empty list is a fact about the run, not a line to leave out."""
        code = self._bake_cli(tmp_path, monkeypatch, flagged=[])
        captured = capsys.readouterr()
        assert "flagged bugs (0)" in captured.out
        assert captured.out.rstrip().splitlines()[-1] == "bake complete"
        assert code == run_mod.EXIT_OK


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


class TestDevFixSelfCheck:
    """prepare grades the developer fix with the real oracle and refuses the bug if it fails.

    [VANTA-WHY] PySnooper/2 shipped a whole wave with its own developer fix grading
    ``patch_good = false``: ``bug_patch.txt`` is a strict subset of the fix commit, so the
    patched tree cannot import the package and *no* patch of any kind could be graded good.
    Nothing in the artifacts said so — every candidate simply graded bad, which is
    indistinguishable from Vanta being conservative. These tests pin the gate that turns it
    into a refusal at the cheapest point in the pipeline.
    """

    SOURCE_DIFF = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,3 +1,4 @@\n"
        " one\n"
        " two\n"
        " three\n"
        "+four\n"
    )
    TEST_DIFF = (
        "diff --git a/tests/test_demo.py b/tests/test_demo.py\n"
        "index 3333333..4444444 100644\n"
        "--- a/tests/test_demo.py\n"
        "+++ b/tests/test_demo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def test_fix():\n"
        "+    assert True\n"
        "     pass\n"
    )

    @pytest.fixture
    def graded(self, monkeypatch):
        """``grade`` recorded rather than run: no VM, and the arguments stay inspectable."""
        calls: dict = {}

        def fake_grade(**kwargs):
            calls.setdefault("gradings", []).append(kwargs)
            return calls["result"]

        monkeypatch.setattr(oracle_mod, "grade", fake_grade)
        calls["result"] = oracle_mod.OracleResult(
            apply_ok=True,
            hidden_pass=True,
            hidden_nodeids=["tests/test_demo.py::test_fix"],
            patch_good=True,
        )
        return calls

    def _run(self, tmp_path, *, text=None, fail_to_pass=None):
        prep = oracle_prep(
            bug=bug_info(bug_patch_txt=self.SOURCE_DIFF if text is None else text)
        )
        if fail_to_pass is not None:
            prep.fail_to_pass = list(fail_to_pass)
        ok = prepare_mod.selfcheck_dev_fix(
            prep, batch_dir=tmp_path / "batch", checkpoint="vanta-bi-Demo-1"
        )
        return ok, prep

    def test_a_good_developer_fix_lets_the_bug_continue(self, graded, tmp_path):
        ok, prep = self._run(tmp_path)
        assert ok is True
        assert prep.gate_failures == []

    def test_the_graded_patch_is_the_one_candidate_set_will_grade(self, graded, tmp_path):
        """Same source through the same filter as ``runner.candidate_set``: two gradings of
        one artifact have to be one question, not two similar ones."""
        self._run(tmp_path)
        call = graded["gradings"][0]
        assert call["patch_path"].name == "Demo_1-dev_fix.patch"
        assert call["patch_path"].read_text() == suite_mod.drop_test_file_hunks(self.SOURCE_DIFF)
        assert call["prep"].key == "Demo/1"
        assert call["label"] == oracle_mod.DEV_FIX_STEM

    def test_test_hunks_are_dropped_from_the_graded_patch(self, graded, tmp_path):
        ok, _ = self._run(tmp_path, text=self.SOURCE_DIFF + self.TEST_DIFF)
        assert ok is True
        written = (
            tmp_path / "batch" / oracle_mod.SELFCHECK_DIR / "Demo_1-dev_fix.patch"
        ).read_text()
        assert "pkg/mod.py" in written
        assert "test_demo.py" not in written

    def test_a_developer_fix_that_is_not_patch_good_refuses_the_bug(self, graded, tmp_path):
        graded["result"] = oracle_mod.OracleResult(
            apply_ok=True,
            hidden_pass=False,
            hidden_nodeids=["tests/test_demo.py::test_fix"],
            hidden_outcomes={"tests/test_demo.py::test_fix": "absent"},
            suite_regressions=["tests/test_demo.py::test_other"],
            patch_good=False,
        )
        ok, prep = self._run(tmp_path)
        assert ok is False
        assert prep.runnable is False
        failure = prep.gate_failures[0]
        assert len(prep.gate_failures) == 1, "one reason, not a wall of them"
        assert "developer fix" in failure
        assert "'absent'" in failure
        assert "tests/test_demo.py::test_other" in failure

    def test_the_refusal_quotes_the_reason_the_rule_exists_for(self):
        assert "PySnooper/2" in prepare_mod.DEV_FIX_SELFCHECK_REASON

    def test_the_grading_is_recorded_next_to_the_patch(self, graded, tmp_path):
        """A refused bug has to leave behind which patch it graded and what it saw."""
        self._run(tmp_path)
        saved = json.loads(
            (
                tmp_path / "batch" / oracle_mod.SELFCHECK_DIR / "Demo_1-dev_fix.oracle.json"
            ).read_text()
        )
        assert saved["patch_good"] is True
        assert saved["hidden_nodeids"] == ["tests/test_demo.py::test_fix"]

    def test_a_test_only_developer_fix_is_refused_without_grading(self, graded, tmp_path):
        """Nothing to apply means no suite may boot — the same rule ``test_hunk_free`` applies
        to a review candidate (§5.4)."""
        ok, prep = self._run(tmp_path, text=self.TEST_DIFF)
        assert ok is False
        assert "gradings" not in graded
        assert "nothing but test hunks" in prep.gate_failures[0]

    def test_a_bug_with_no_oracle_is_not_self_checked(self, graded, tmp_path):
        """An empty FAIL_TO_PASS is already gated by the derivation; adding a second,
        contradicting failure here would report the wrong cause."""
        ok, prep = self._run(tmp_path, fail_to_pass=[])
        assert ok is True
        assert prep.gate_failures == []
        assert "gradings" not in graded

    def test_prepare_one_runs_it_after_the_strip_and_before_the_issue(self):
        """Wiring, not behaviour: the check needs the post-strip baseline, and a bug it
        refuses must not go on to be measured as a candidate."""
        source = inspect.getsource(prepare_mod.prepare_one)
        assert 0 <= source.index("strip_and_baseline(")
        assert source.index("strip_and_baseline(") < source.index("selfcheck_dev_fix(")
        assert source.index("selfcheck_dev_fix(") < source.index("build_issue(")


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
            # §14.6: a refusal to decide is scored as itself, on both sides of the
            # ground truth — never as the rejection it declined to issue and never
            # as the approval it never made.
            ("NeedsReview", True, True, "needs_review_on_good"),
            ("NeedsReview", True, False, "needs_review_on_bad"),
            ("InfraFailure", False, False, "infra_failure"),
            ("InfraFailure", True, True, "infra_failure"),
            # The censor wins over the verdict: a patch that never applied told us
            # nothing, so it cannot become a `true_reject` for Vanta (§5.2 step 2).
            ("Rejected", False, True, "unobservable"),
            ("NeedsReview", False, True, "unobservable"),
            ("PatchApplicationFailed", False, False, "unobservable"),
            ("HarnessFailure", False, False, "unobservable"),
            ("SomethingNobodyNamed", True, True, "unobservable"),
        ],
    )
    def test_the_table(self, verdict, apply_ok, patch_good, expected):
        oracle = oracle_mod.OracleResult(apply_ok=apply_ok, patch_good=patch_good)
        assert runner_mod.label_for(verdict, oracle) == expected

    def test_a_needs_review_label_is_not_a_rejection_label(self):
        """§14.6: the two refusals must not collide with the four decisions.

        [WHY] `false_reject` is a scored Vanta error and `true_reject` is a scored
        Vanta catch. A refusal is neither, so the two families must be disjoint —
        a shared name would put a row the loop never decided into a rate that
        claims it did.
        """
        good = oracle_mod.OracleResult(apply_ok=True, patch_good=True)
        bad = oracle_mod.OracleResult(apply_ok=True, patch_good=False)
        refusal = {
            runner_mod.label_for("NeedsReview", good),
            runner_mod.label_for("NeedsReview", bad),
        }
        decided = {
            runner_mod.label_for(verdict, oracle)
            for verdict in ("AllTestsPassed", "Rejected", "MaxAttemptsReached")
            for oracle in (good, bad)
        }
        assert refusal == {"needs_review_on_good", "needs_review_on_bad"}
        assert refusal.isdisjoint(decided)

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

    def test_a_needs_review_row_does_not_stop_the_wave(self):
        """§14.6: a refusal to decide is the designed outcome, not a gate failure.

        [WHY] The label is taken from `label_for` rather than typed in, so this
        test fails if the label table and the gate ever disagree about what a
        refusal is called — which is exactly how the row would otherwise come
        back as a censored `unobservable` and stop the wave.
        """
        rows = _healthy_wave()
        for row in rows:
            if row["candidate"] and row["candidate"]["kind"] == runner_mod.DEV_FIX:
                good = bool(row["candidate"]["oracle"]["patch_good"])
                row["verdict"] = "NeedsReview"
                row["label"] = runner_mod.label_for(
                    "NeedsReview", oracle_mod.OracleResult(apply_ok=True, patch_good=good)
                )
        labels = {row["label"] for row in rows if row["verdict"] == "NeedsReview"}
        assert labels == {"needs_review_on_good"}
        assert runner_mod.wave_gate_failures(rows) == []

    def test_a_needs_review_row_the_oracle_could_not_grade_is_still_gated(self):
        """The censor is verdict-independent: it reports the plumbing, not the refusal.

        [WHY] `no final.patch` means the candidate was never applied, so the row
        says nothing about Vanta's decision-making and the wave's plumbing is
        what needs looking at. Keeping this gate armed is what stops the
        NeedsReview exemption above from becoming a blanket silence.
        """
        rows = _healthy_wave()
        rows[0]["verdict"] = "NeedsReview"
        rows[0]["label"] = "unobservable"
        rows[0]["oracle"]["apply_ok"] = False
        rows[0]["oracle"]["apply_detail"] = "no final.patch for verdict NeedsReview"
        problems = runner_mod.wave_gate_failures(rows)
        assert any(
            "unobservable" in problem and "no final.patch" in problem for problem in problems
        )


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
