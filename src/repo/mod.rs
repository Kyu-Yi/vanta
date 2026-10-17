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
        let mut source_files = Vec::new();
        let mut total_bytes: usize = 0;
        collect_source_files(repo_dir, repo_dir, &mut source_files, &mut total_bytes);

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

    for entry in entries.flatten() {
        let path = entry.path();

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
}
