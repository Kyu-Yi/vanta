// ============================================================================
// REPO MODULE — load an arbitrary repository directory for Vanta to verify
// ============================================================================
//
// [VANTA-WHY] M2 hardcoded examples/buggy_python/cart.py. M3 needs to run on
// real repositories (heretic, SWE-bench, etc.). This module walks a directory,
// discovers source files, and reads the issue text.
//
// Design:
//   - Repo::load(dir) walks recursively, collecting source files by extension
//   - Source files limited to a max total size (context window budget)
//   - issue_text comes from --issue file OR auto-detected (issue.md, bug.md, etc.)
//   - Repo exposes source_files: Vec<SourceFile> for the Creator
// ============================================================================

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::agents::creator::SourceFile;

// [VANTA-WHY] Source file extensions we hand to the Creator.
// M3 target is Python (heretic, SWE-bench), but we keep a few common ones.
const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "rs", "go", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp",
];

// [VANTA-WHY] Context window budget. The Creator prompt includes ALL source files.
// Real repos can be huge; we cap total bytes to avoid blowing the context.
// 256 KB is enough for a focused bug fix in a small-to-medium repo.
const MAX_TOTAL_SOURCE_BYTES: usize = 256 * 1024;

// [VANTA-WHY] Per-file cap. A single giant file (e.g. generated code) shouldn't
// dominate the context. 64 KB per file.
const MAX_FILE_BYTES: usize = 64 * 1024;

// [VANTA-WHY] Directories we never descend into (noise, vendored deps, build
// artifacts). Keeps the walk fast and the context clean.
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", "target", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", ".idea", ".vscode",
];

pub struct Repo {
    pub root: PathBuf,
    pub source_files: Vec<SourceFile>,
    pub issue_text: String,
}

impl Repo {
    /// Load a repository directory.
    ///
    /// `repo_dir: path to the repo root.
    /// `issue_path`: optional path to the issue/bug description file.
    ///   If None, we auto-detect (issue.md, bug.md, etc.) inside repo_dir.
    pub fn load(
        repo_dir: &Path,
        issue_path: Option<&Path>,
    ) -> Result<Self, String> {
        if !repo_dir.is_dir() {
            return Err(format!(
                "Repo directory does not exist or is not a directory: {}",
                repo_dir.display()
            ));
        }

        info!(repo = %repo_dir.display(), "loading repo");

        // ---- Discover source files ----
        let source_files = capped_source_files(repo_dir);
        let total_bytes: usize = source_files.iter().map(|f| f.contents.len()).sum();

        info!(
            file_count = source_files.len(),
            total_bytes,
            "discovered source files"
        );

        if source_files.is_empty() {
            warn!(repo = %repo_dir.display(),
                "no source files found — Creator will have no context");
        }

        // ---- Read issue text ----
        let issue_text = match issue_path {
            Some(p) => {
                if p.is_file() {
                    fs::read_to_string(p).map_err(|e| {
                        format!("Failed to read issue file {}: {e}", p.display())
                    })?
                } else {
                    // Maybe issue_path is a string literal (the issue text itself)
                    p.to_string_lossy().to_string()
                }
            }
            None => auto_detect_issue(repo_dir)?,
        };

        info!(issue_len = issue_text.len(), "issue text loaded");

        Ok(Repo {
            root: repo_dir.to_path_buf(),
            source_files,
            issue_text,
        })
    }
}

// ----------------------------------------------------------------------------
// collect_source_files — recursive walk, collecting source files by extension
// ----------------------------------------------------------------------------
/// The repo's source files as the loader has always collected them: a
/// deterministic walk stopped by a per-file cap and a total byte cap.
///
/// [VANTA-WHY] §14.18 — this is a PROMPT budget, so the only callers left are
/// prompt-shaped: `Repo::load` (the Creator's fallback context, the import-path
/// probe) and the prompt fallback in the driver. The trees tests are executed
/// in come from `complete_tree_files`, because a cap that drops files silently
/// turns "the repo" into "the first 256 KB of the repo" — see that function.
pub fn capped_source_files(root: &Path) -> Vec<SourceFile> {
    let mut files = Vec::new();
    let mut total_bytes: usize = 0;
    collect_source_files(root, root, &mut files, &mut total_bytes);
    files
}

/// Every readable text file under `root`, as a `path -> contents` map.
///
/// [VANTA-WHY] §14.18 — this is the tree BATE's baseline, every revise, the
/// native suite, the graded patched tree, the dependency probe and the static
/// pass are executed against, so it has to be the REPOSITORY, not an excerpt of
/// it. It used to be `repo.source_files` (and `candidate::collect_tree`): the
/// same deterministic walk, but stopped at 64 KB per file and 256 KB in total,
/// and blind to every non-source extension. On sanic/1 that dropped
/// `sanic/websocket.py` and ALL of `tests/`, and both failures were silent:
/// the `pytest_sanic` entry-point plugin imports `sanic`, so pytest aborted in
/// `_preparse` before collecting anything and every generated test came back
/// `Error` (three identical mismatches, then a 32k-token revise against an
/// environment error no test can fix), while the repo's own suite "ran" with
/// zero tests and the Governor read that as no regressions.
///
/// This is the cap-free walk the context selection already uses — one
/// implementation, two consumers — so what the prompt may spend and what the
/// tests may see are decided independently instead of by the same byte cap.
/// It skips `SKIP_DIRS`, never follows a symlink, and leaves out anything that
/// is not valid UTF-8 (a binary file cannot be carried in the concat).
pub fn complete_tree_files(root: &Path) -> HashMap<String, String> {
    let (candidates, _skipped) = crate::context::walk_candidates(root);
    candidates
        .into_iter()
        .map(|candidate| (candidate.path, candidate.contents))
        .collect()
}

fn collect_source_files(
    base: &Path,
    current: &Path,
    out: &mut Vec<SourceFile>,
    total_bytes: &mut usize,
) {
    let entries = match fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };
    // [VANTA-WHY] `read_dir` order is filesystem-dependent, so once the byte cap
    // bites the same tree can yield a different subset on a different machine —
    // the bug's own file in the prompt here, absent there. Sorting makes the walk
    // (and, while it lasts, the truncation) reproducible. `context::walk_candidates`
    // is the cap-free replacement this walk feeds into; this stays as the loader
    // for the trees BATE and the harness run against.
    let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();

    for path in paths {
        // Skip directories in SKIP_DIRS
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if SKIP_DIRS.contains(&name) {
                    continue;
                }
            }
            collect_source_files(base, &path, out, total_bytes);
            continue;
        }

        // Check extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !SOURCE_EXTENSIONS.contains(&ext) {
            continue;
        }

        // Read file (with size cap)
        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let file_len = metadata.len() as usize;
        if file_len > MAX_FILE_BYTES {
            debug!(path = %path.display(), len = file_len, "skipping oversized source file (>64KB)");
            continue;
        }
        if *total_bytes + file_len > MAX_TOTAL_SOURCE_BYTES {
            debug!(path = %path.display(), "skipping source file (would exceed total budget)");
            continue;
        }

        let contents = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue, // binary or unreadable
        };

        // Relative path from repo root
        let rel = path.strip_prefix(base).unwrap_or(&path);
        out.push(SourceFile {
            path: rel.to_string_lossy().to_string(),
            contents,
        });
        *total_bytes += file_len;
    }
}

// ----------------------------------------------------------------------------
// auto_detect_issue — look for issue.md, bug.md, etc. inside the repo
// ----------------------------------------------------------------------------
fn auto_detect_issue(repo_dir: &Path) -> Result<String, String> {
    let candidates = [
        "issue.md",
        "bug.md",
        "ISSUE.md",
        "BUG.md",
        "issue.txt",
        "bug.txt",
        "README.md", // fallback: last resort
    ];

    for name in &candidates {
        let p = repo_dir.join(name);
        if p.is_file() {
            return fs::read_to_string(&p)
                .map_err(|e| format!("Failed to read {}: {e}", p.display()));
        }
    }

    Err(format!(
        "No issue file found in {}. Pass --issue PATH or place issue.md in the repo.",
        repo_dir.display()
    ))
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_repo_with_source_and_issue() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("cart.py"), "def add(a, b):\n    return a + b\n").unwrap();
        fs::write(tmp.path().join("issue.md"), "add(2,3) should return 5\n").unwrap();

        let repo = Repo::load(tmp.path(), None).unwrap();
        assert_eq!(repo.source_files.len(), 1);
        assert_eq!(repo.source_files[0].path, "cart.py");
        assert!(repo.issue_text.contains("add(2,3)"));
    }

    #[test]
    fn test_skip_noise_dirs() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("__pycache__")).unwrap();
        fs::write(tmp.path().join("__pycache__/junk.py"), "x = 1").unwrap();
        fs::write(tmp.path().join("real.py"), "def f(): pass").unwrap();
        fs::write(tmp.path().join("issue.md"), "bug\n").unwrap();

        let repo = Repo::load(tmp.path(), None).unwrap();
        assert_eq!(repo.source_files.len(), 1);
        assert_eq!(repo.source_files[0].path, "real.py");
    }

    #[test]
    fn test_explicit_issue_path() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("main.py"), "x = 1").unwrap();
        let issue_file = tmp.path().join("my_bug.txt");
        fs::write(&issue_file, "explicit issue text here").unwrap();

        let repo = Repo::load(tmp.path(), Some(&issue_file)).unwrap();
        assert!(repo.issue_text.contains("explicit issue text"));
    }

    #[test]
    fn test_missing_repo_errors() {
        let result = Repo::load(Path::new("/nonexistent/vanta/repo"), None);
        assert!(result.is_err());
    }

    /// [VANTA-WHY] §14.18 regression — how sanic/1 lost 100 % of its graded
    /// tests. The walk below is deterministic and sorted, so once the 256 KB
    /// total cap is spent it drops whatever sorts last. On sanic/1 that was
    /// `sanic/websocket.py` and all of `tests/`, and neither failure was
    /// visible: `pytest_sanic` is registered as a `pytest11` entry point, so
    /// pytest imported it during `_preparse`, that import reached
    /// `sanic/asgi.py` -> `sanic.websocket`, and pytest aborted BEFORE collecting
    /// anything. Every generated test then came back `Error` (never `Failed`),
    /// BATE scrapped all of them after three identical mismatches, and the
    /// repo's own suite "ran" with zero tests and read as no regressions.
    ///
    /// The assertion this test makes is the one that would have caught it: the
    /// tree the tests EXECUTE in must be the whole repo, and must not converge
    /// back onto the byte-capped dump the PROMPT is still allowed to fall back
    /// to. Both walks stay in the codebase — one is a prompt budget, the other
    /// is a filesystem — so the test pins the difference between them.
    #[test]
    fn test_execution_tree_is_not_the_capped_prompt_dump() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Six 60 KB source files. The 256 KB total budget fits four, so the
        // sorted walk drops the last two — `ee.py` by the total cap, and
        // `zz_module.py` because it sorts last, exactly like the module an
        // import chain reaches for at the end of the alphabet.
        let body = "x = 1\n".repeat(10_000); // 60_000 bytes
        for name in ["aa.py", "bb.py", "cc.py", "dd.py", "ee.py"] {
            fs::write(root.join(name), &body).unwrap();
        }
        fs::write(root.join("zz_module.py"), &body).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::write(root.join("tests/conftest.py"), "import sys\n").unwrap();

        let capped: Vec<String> = capped_source_files(root)
            .into_iter()
            .map(|file| file.path)
            .collect();
        let execution_tree = complete_tree_files(root);

        assert!(
            !capped.contains(&"zz_module.py".to_string()),
            "the prompt dump IS capped, so it must drop files once the budget is \
             spent — if this fails the fixture no longer exercises the cap: {capped:?}"
        );
        assert!(
            execution_tree.contains_key("zz_module.py"),
            "the tree tests execute in must NOT be the capped prefix; it was \
             missing {:?}",
            execution_tree.keys().collect::<Vec<_>>()
        );
        assert!(execution_tree.contains_key("tests/conftest.py"));
        assert!(
            execution_tree.len() > capped.len(),
            "the execution tree ({}) must not converge back onto the prompt dump ({})",
            execution_tree.len(),
            capped.len()
        );
        // A file in the execution tree is carried WHOLE — the cap must not have
        // truncated it either.
        assert_eq!(execution_tree["zz_module.py"], body);
    }

    /// [VANTA-WHY] §14.18 — the other half of the sanic/1 shape: the module the
    /// import chain needs lives in a directory the capped walk never reached, so
    /// the file is absent rather than truncated. Pinned separately because a
    /// future cap change could pass the byte assertion above while still
    /// pruning `tests/`.
    #[test]
    fn test_execution_tree_reaches_directories_the_cap_front_never_reaches() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Spend the whole 256 KB budget from a directory that sorts FIRST, so
        // the walk reaches `sanic/` with nothing left. Four files of exactly
        // 64 KB are the most the per-file cap admits, and together they are
        // exactly the total budget.
        fs::create_dir_all(root.join("aaa_filler")).unwrap();
        let body = "x".repeat(64 * 1024);
        for index in 0..4 {
            fs::write(root.join("aaa_filler").join(format!("f{index}.py")), &body).unwrap();
        }
        fs::create_dir_all(root.join("sanic")).unwrap();
        fs::write(root.join("sanic/websocket.py"), "class WebSocketConnection:\n    pass\n").unwrap();

        let capped: Vec<String> = capped_source_files(root)
            .into_iter()
            .map(|file| file.path)
            .collect();
        let execution_tree = complete_tree_files(root);

        assert!(!capped.contains(&"sanic/websocket.py".to_string()), "{capped:?}");
        assert!(execution_tree.contains_key("sanic/websocket.py"));
    }
}
