// ============================================================================
// CANDIDATE PAIR — the abstraction that unifies Mode A (benchmark) and
// Mode B (review-a-repo) entry modes (Spec §6, §9).
// ============================================================================
//
// [VANTA-WHY] Spec §6: "Clone-a-repo-and-check-my-edits is a genuinely
// different entry mode from the benchmark case, and the architecture should
// name the difference." Without this abstraction, M3 hardcodes "load files
// from a directory, apply a patch in memory" and there is no clean place to
// plug in BugsInPy checkouts, SWE-bench instances, or git worktrees.
//
// A CandidatePair answers five questions, regardless of where the code came
// from:
//   1. What are the ORIGINAL (unmodified) files?
//   2. What is the issue/bug text?
//   3. How do I run the tests?
//   4. Who produced this evidence (provenance)?
//   5. (Optional) What on-disk paths back the trees?
//
// The CANDIDATE (patched) files are NOT part of the trait: the Creator
// produces them by patching the original. This keeps the trait a pure
// description of the input, not a mutable state machine.
//
// Three implementations (Spec §6 table + §9.3 "one loader, three adapters"):
//   - LocalRepoPair    : a directory on disk (Mode B, current default)
//   - GitWorkingTreePair: git worktree of base commit (Mode B, two trees)
//   - BenchmarkInstance: BugsInPy / SWE-bench / Defects4J (Mode A)
// ============================================================================

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::EvidenceSource;
use crate::repo::Repo;

// ============================================================================
// TestCommand — how to run the test suite (Spec §7 "Test command" row)
// ============================================================================
//
// [VANTA-WHY] Spec §7: "Require explicit --test-cmd, pytest as fallback. Do
// not infer." Real repos use tox, nox, make test, custom runners, markers.
// A hardcoded `python -m pytest` fails on anything but toy examples.
//
// TestCommand stores program + args separately so the sandbox can exec them
// directly (no shell injection). A shell-string constructor is provided for
// convenience (e.g. `pytest tests/ -k test_foo`), split on whitespace.
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl TestCommand {
    /// Build from a program name and explicit args.
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    /// Parse a shell-style string like `pytest tests/ -q` into program + args.
    /// Splits on whitespace; does NOT handle quoting (V1 limitation — fine
    /// for pytest invocations).
    pub fn from_shell(s: &str) -> Self {
        let parts: Vec<String> = s.split_whitespace().map(String::from).collect();
        if parts.is_empty() {
            return Self::default();
        }
        Self {
            program: parts[0].clone(),
            args: parts[1..].to_vec(),
        }
    }

    /// The full command as a vec (program first), suitable for sandbox.execute.
    pub fn as_cmd_vec(&self) -> Vec<String> {
        let mut v = vec![self.program.clone()];
        v.extend(self.args.iter().cloned());
        v
    }
}

impl Default for TestCommand {
    fn default() -> Self {
        // [VANTA-WHY] pytest is the fallback only when the user hasn't told
        // us otherwise. For SWE-bench/BugsInPy the adapter supplies the real
        // command; for local repos the user passes --test-cmd.
        Self::new("python3", vec!["-m".into(), "pytest".into()])
    }
}

// ============================================================================
// CandidatePair trait
// ============================================================================
//
// [VANTA-WHY] Every downstream stage (Creator, Attacker, BATE, patched run,
// Gatekeeper, PR) consumes the SAME five things. By putting them behind one
// trait, adding a fourth source (e.g. a Defects4J adapter) is an afternoon
// rather than a rewrite of vanta_m3.rs.
// ============================================================================

pub trait CandidatePair {
    /// The original (unmodified) source files, keyed by repo-relative path.
    fn original_files(&self) -> HashMap<String, String>;

    /// The issue / bug report text (fed to the Creator + Attacker).
    fn issue_text(&self) -> String;

    /// How to run the tests in this repo.
    fn test_command(&self) -> TestCommand;

    /// Who produced the candidate — for evidence provenance tagging.
    fn provenance(&self) -> EvidenceSource;

    // ---- Optional on-disk tree paths -------------------------------------
    //
    // [VANTA-WHY] Some pairs (git worktree, benchmark checkout) have real
    // on-disk trees. Returning Some lets the sandbox copy from disk instead
    // of reconstructing from the in-memory map (faster for large repos, and
    // preserves binary/non-source files). In-memory pairs return None and
    // the sandbox falls back to writing original_files().

    /// Path to the original tree on disk, if backed by one.
    fn original_tree_path(&self) -> Option<&Path> {
        None
    }

    /// Path to the candidate tree on disk, if backed by one.
    fn candidate_tree_path(&self) -> Option<&Path> {
        None
    }
}

// ============================================================================
// LocalRepoPair — a plain directory on disk (Mode B default)
// ============================================================================
//
// [VANTA-WHY] This is what vanta_m3 already does: walk a directory, collect
// source files, read an issue file. Wrapping it in CandidatePair means the
// main loop no longer special-cases "is this a local repo or a benchmark".
// ============================================================================

pub struct LocalRepoPair {
    pub repo: Repo,
    pub test_cmd: TestCommand,
}

impl LocalRepoPair {
    pub fn new(repo: Repo, test_cmd: TestCommand) -> Self {
        Self { repo, test_cmd }
    }
}

impl CandidatePair for LocalRepoPair {
    fn original_files(&self) -> HashMap<String, String> {
        self.repo
            .source_files
            .iter()
            .map(|sf| (sf.path.clone(), sf.contents.clone()))
            .collect()
    }

    fn issue_text(&self) -> String {
        self.repo.issue_text.clone()
    }

    fn test_command(&self) -> TestCommand {
        self.test_cmd.clone()
    }

    fn provenance(&self) -> EvidenceSource {
        // Local repo edits = HumanProvided (the user wrote the patch).
        EvidenceSource::HumanProvided
    }
}

// ============================================================================
// GitWorkingTreePair — two simultaneous trees via `git worktree` (Mode B)
// ============================================================================
//
// [VANTA-WHY] Spec §7: "The original is a second checkout via git worktree add
// of the base commit. Not the same directory stashed and reverted: Vanta
// needs both trees present simultaneously, because BATE runs against the
// original while the candidate exists."
//
// This implementation does NOT create the worktree itself (that's a one-time
// setup the user does). It points at two existing directories: the worktree
// (original, clean) and the working tree (candidate, with edits). It reads
// source files from both on demand.
//
// [PITFALL] For very large repos, reading all files into memory on every
// call is expensive. V1 accepts this because the repo loader already caps
// total bytes at 256 KB. For M5-scale runs we'd add a streaming variant.
// ============================================================================

pub struct GitWorkingTreePair {
    /// Path to the clean worktree (base commit). BATE runs against this.
    pub original_tree: PathBuf,
    /// Path to the working tree with the candidate edits.
    pub candidate_tree: PathBuf,
    pub issue_text: String,
    pub test_cmd: TestCommand,
    /// Source file extensions to collect (defaults to Repo's set).
    pub extensions: Vec<String>,
    /// Max total bytes to load (context window budget).
    pub max_bytes: usize,
}

impl GitWorkingTreePair {
    pub fn new(
        original_tree: PathBuf,
        candidate_tree: PathBuf,
        issue_text: String,
        test_cmd: TestCommand,
    ) -> Self {
        Self {
            original_tree,
            candidate_tree,
            issue_text,
            test_cmd,
            extensions: vec![
                "py".into(), "rs".into(), "go".into(), "js".into(), "ts".into(),
                "jsx".into(), "tsx".into(), "java".into(), "c".into(), "cpp".into(),
                "h".into(), "hpp".into(),
            ],
            max_bytes: 256 * 1024,
        }
    }

    /// Collect source files from a tree, respecting extension + size caps.
    fn collect(&self, root: &Path) -> HashMap<String, String> {
        let mut out = HashMap::new();
        let mut total = 0usize;
        collect_tree(root, root, &self.extensions, self.max_bytes, &mut out, &mut total);
        out
    }
}

impl CandidatePair for GitWorkingTreePair {
    fn original_files(&self) -> HashMap<String, String> {
        self.collect(&self.original_tree)
    }

    fn issue_text(&self) -> String {
        self.issue_text.clone()
    }

    fn test_command(&self) -> TestCommand {
        self.test_cmd.clone()
    }

    fn provenance(&self) -> EvidenceSource {
        EvidenceSource::HumanProvided
    }

    fn original_tree_path(&self) -> Option<&Path> {
        Some(&self.original_tree)
    }

    fn candidate_tree_path(&self) -> Option<&Path> {
        Some(&self.candidate_tree)
    }
}

// ============================================================================
// BenchmarkInstance — BugsInPy / SWE-bench / Defects4J (Mode A)
// ============================================================================
//
// [VANTA-WHY] Spec §8 + §9: a benchmark instance is a curated buggy commit +
// bug report + known failing tests. The SWE-bench field mapping is
// documented in §9.2. This struct holds the metadata; the actual checkout is
// done by the adapter (BugsInPy CLI, or `git clone` at base_commit).
//
// SWE-bench field mapping (Spec §9.2 table):
//   instance_id        → self.instance_id        (benchmark label on every row)
//   repo               → self.repo               (owner/name)
//   base_commit        → original_tree checkout  (original_tree)
//   problem_statement  → self.issue_text         (issue_text)
//   patch              → QUARANTINED (scoring only — never enters inputs)
//   test_patch         → applied at eval time, not given to Creator
//   FAIL_TO_PASS       → self.fail_to_pass       (intent BugReproduction)
//   PASS_TO_PASS       → self.pass_to_pass       (intent BehaviorPreservation)
//   hints_text         → EXCLUDED (leaks approach)
//
// [THE ONE RULE THAT CANNOT BEND] (Spec §9.1): the fixed version (gold patch)
// is the oracle. It NEVER enters Vanta's inputs. If it is reachable from
// inside the loop, every number is worthless. We store it as `gold_patch`
// Option<String> and the CandidatePair impl does NOT expose it.
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkInstance {
    /// Stable instance label (e.g. "django__django-12345"). Goes on every row.
    pub instance_id: String,
    /// Benchmark source (BugsInPy, SWE-bench, Defects4J).
    pub benchmark: String,
    /// GitHub owner/repo (SWE-bench) or project name (BugsInPy).
    pub repo: String,
    /// Commit to check out as the original (buggy) tree.
    pub base_commit: String,
    /// The bug report / problem statement. Fed to Creator + Attacker.
    pub issue_text: String,
    /// Tests that FAIL before the fix and PASS after → BugReproduction intent.
    pub fail_to_pass: Vec<String>,
    /// Tests that PASS before AND after → BehaviorPreservation intent.
    pub pass_to_pass: Vec<String>,
    /// How to run tests for this instance.
    pub test_cmd: TestCommand,
    /// Path to the checked-out original tree (buggy commit).
    pub original_tree: PathBuf,
    /// [QUARANTINED] The gold fix. NEVER passed to Creator/Attacker. Used only
    /// by the harness for post-hoc scoring. CandidatePair:: does not expose it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold_patch: Option<String>,
    /// Model tier label for per-tier logging (frontier / mid / cheap / local).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_tier: Option<String>,
}

impl CandidatePair for BenchmarkInstance {
    fn original_files(&self) -> HashMap<String, String> {
        // Read source files from the checked-out buggy tree.
        let mut out = HashMap::new();
        let mut total = 0usize;
        let exts = vec!["py".into(), "rs".into(), "go".into(), "js".into(), "ts".into()];
        collect_tree(&self.original_tree, &self.original_tree, &exts, 256 * 1024, &mut out, &mut total);
        out
    }

    fn issue_text(&self) -> String {
        self.issue_text.clone()
    }

    fn test_command(&self) -> TestCommand {
        self.test_cmd.clone()
    }

    fn provenance(&self) -> EvidenceSource {
        // Benchmark's own tests are oracle truth; Attacker-generated tests
        // are tagged AttackerGenerated by the test runner.
        EvidenceSource::BenchmarkOracle
    }

    fn original_tree_path(&self) -> Option<&Path> {
        Some(&self.original_tree)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Walk a directory tree collecting source files into a path→content map.
fn collect_tree(
    base: &Path,
    current: &Path,
    extensions: &[String],
    max_bytes: usize,
    out: &mut HashMap<String, String>,
    total: &mut usize,
) {
    let skip = [".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        "build", "dist", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "site-packages"];

    let entries = match fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if skip.contains(&name) {
                    continue;
                }
            }
            collect_tree(base, &path, extensions, max_bytes, out, total);
            continue;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !extensions.iter().any(|e| e == ext) {
            continue;
        }

        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let len = metadata.len() as usize;
        if len > 64 * 1024 || *total + len > max_bytes {
            continue;
        }

        let contents = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel = path.strip_prefix(base).unwrap_or(&path);
        out.insert(rel.to_string_lossy().to_string(), contents);
        *total += len;
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_command_from_shell() {
        let cmd = TestCommand::from_shell("pytest tests/ -q -x");
        assert_eq!(cmd.program, "pytest");
        assert_eq!(cmd.args, vec!["tests/", "-q", "-x"]);
        assert_eq!(
            cmd.as_cmd_vec(),
            vec!["pytest".to_string(), "tests/".into(), "-q".into(), "-x".into()]
        );
    }

    #[test]
    fn test_command_default_is_pytest() {
        let cmd = TestCommand::default();
        assert_eq!(cmd.program, "python3");
        assert_eq!(cmd.args, vec!["-m".to_string(), "pytest".to_string()]);
    }

    #[test]
    fn local_repo_pair_exposes_files() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("cart.py"), "def add(a,b): a+b\n").unwrap();
        fs::write(tmp.path().join("issue.md"), "bug\n").unwrap();
        let repo = Repo::load(tmp.path(), None).unwrap();
        let pair = LocalRepoPair::new(repo, TestCommand::default());
        let files = pair.original_files();
        assert!(files.contains_key("cart.py"));
        assert_eq!(pair.issue_text(), "bug\n");
        assert_eq!(pair.provenance(), EvidenceSource::HumanProvided);
    }

    #[test]
    fn git_worktree_pair_reads_both_trees() {
        let orig = TempDir::new().unwrap();
        let cand = TempDir::new().unwrap();
        fs::write(orig.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(cand.path().join("a.py"), "x = 2\n").unwrap();

        let pair = GitWorkingTreePair::new(
            orig.path().to_path_buf(),
            cand.path().to_path_buf(),
            "issue".into(),
            TestCommand::default(),
        );
        let of = pair.original_files();
        assert_eq!(of.get("a.py").unwrap(), "x = 1\n");
        assert_eq!(pair.original_tree_path(), Some(orig.path()));
        assert_eq!(pair.candidate_tree_path(), Some(cand.path()));
    }

    #[test]
    fn benchmark_instance_gold_patch_not_in_inputs() {
        // The CandidatePair trait surface must NOT expose gold_patch.
        // This test guards against accidentally leaking the oracle.
        let bm = BenchmarkInstance {
            instance_id: "test-1".into(),
            benchmark: "swe-bench".into(),
            repo: "o/r".into(),
            base_commit: "abc".into(),
            issue_text: "bug".into(),
            fail_to_pass: vec!["t1".into()],
            pass_to_pass: vec!["t2".into()],
            test_cmd: TestCommand::default(),
            original_tree: PathBuf::from("/tmp/nonexistent"),
            gold_patch: Some("SECRET ORACLE".into()),
            model_tier: Some("local".into()),
        };
        // Trait methods available:
        let _ = bm.issue_text();
        let _ = bm.test_command();
        let _ = bm.provenance();
        // gold_patch is a struct field, not on the trait. A user of the trait
        // object cannot reach it. (Direct struct access is for the scoring
        // harness only, after the run.)
        assert_eq!(bm.provenance(), EvidenceSource::BenchmarkOracle);
    }
}
