// ============================================================================
// SAFE PATH — confine a patch-supplied path to a tree root (M3 item T1)
// ============================================================================
//
// [VANTA-WHY] A Creator patch — and a `--patch` file, and a session's recorded
// test-integrity lists — is TEXT, and parts of it NAME files. Two of those
// sources are model output and neither is trusted: the `===FILE:` marker's path
// is taken verbatim (at any indentation, since item Q1), and a unified diff's
// `+++ b/...` header is read as a destination. `root.join(path)` is where a
// `===FILE: /home/user/.ssh/authorized_keys===` or a `../../x` becomes a WRITE
// outside the tree the run is allowed to touch.
//
// This module is the ONE place that decides whether a path may be used under a
// root, so the rule cannot drift between the apply and the writers.
//
//   * `unsafe_relative_path_reason` is a PURE predicate on the string — no I/O —
//     because the all-or-nothing apply (Contract B) is pure by design and must
//     decide which blocks it will accept without stat-ing anything.
//   * `confined_join` adds the on-disk half: `root.join(rel)` is RESOLVED
//     (following any existing symlink) and refused when it lands outside the
//     resolved `root`. The lexical gate alone cannot see that `pkg` is a
//     symlink to `/etc`.

use std::path::{Path, PathBuf};

/// The reason a patch's whole apply is refused when a block names a path it may
/// not write (M3 item T1). Recorded verbatim on the `patch.apply_failed` row.
pub const UNSAFE_PATH: &str = "unsafe_path";

/// Why `rel` may not be used as a path under a tree root, or `None` when it may.
///
/// Pure and filesystem-free. The cases are the ones a model can emit to leave
/// the tree: an absolute path (leading `/`), a Windows drive (`C:`) or UNC
/// (`\\server\share`) prefix, any `..` component, and an empty or NUL-bearing
/// string — which no path Vanta accepts may be.
pub fn unsafe_relative_path_reason(rel: &str) -> Option<&'static str> {
    if rel.is_empty() {
        return Some("empty path");
    }
    if rel.contains('\u{0}') {
        return Some("path contains a NUL byte");
    }
    // [VANTA-WHY] A padded path is not round-trippable: `build_source_concat`
    // emits `# FILE: {path}` and `parse_source_files` reads it back through
    // `trim()`, and the summary is a whitespace-delimited `key=value` line. A
    // path whose trim differs is a DIFFERENT path by the time it is written, so
    // it is refused rather than silently normalised. Interior spaces (a legal
    // filename byte) are untouched.
    if rel.trim() != rel {
        return Some("leading or trailing whitespace");
    }
    if rel.starts_with('/') || rel.starts_with('\\') {
        return Some("absolute or UNC path");
    }
    let bytes = rel.as_bytes();
    // `C:\x` / `C:/x`: a drive prefix is absolute on the system that mints it.
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Some("drive prefix");
    }
    // Split on BOTH separators. On unix a backslash is an ordinary filename
    // byte, but this decision must not depend on the host: `..\..\x` is a
    // traversal on the system that produced it, and a `..` component is never
    // something Vanta should write through.
    if rel.split(['/', '\\']).any(|component| component == "..") {
        return Some("parent-directory component");
    }
    None
}

/// `true` when `rel` passes the lexical gate.
pub fn is_safe_relative_path(rel: &str) -> bool {
    unsafe_relative_path_reason(rel).is_none()
}

/// Join `rel` onto `root`, refusing anything not safely contained.
///
/// [VANTA-WHY] The lexical gate is not enough on its own: `pkg` may be an
/// existing SYMLINK to a directory outside the root, and `pkg/evil.py` then
/// passes every string test while writing outside. `root.join(rel)` is
/// therefore resolved — the nearest EXISTING ancestor, so a file that does not
/// exist yet (its parent still to be created) is handled — and refused when the
/// resolved ancestor is not under the resolved `root`.
///
/// `Err` carries the `unsafe_path` reason so every caller records the same one
/// and nothing is written anywhere.
pub fn confined_join(root: &Path, rel: &str) -> Result<PathBuf, String> {
    if let Some(why) = unsafe_relative_path_reason(rel) {
        return Err(format!("{UNSAFE_PATH}: {rel} — {why}"));
    }
    let real_root = root.canonicalize().map_err(|e| {
        format!(
            "{UNSAFE_PATH}: cannot resolve the tree root {} ({e})",
            root.display()
        )
    })?;
    let joined = root.join(rel);
    // The nearest ancestor that exists on disk. `symlink_metadata` sees a
    // symlink AS ITSELF, so a link that points nowhere still stops the walk and
    // is refused by the canonicalize below rather than being walked through.
    let mut ancestor: &Path = joined.as_path();
    loop {
        if ancestor.symlink_metadata().is_ok() {
            break;
        }
        match ancestor.parent() {
            Some(parent) if parent != ancestor => ancestor = parent,
            _ => break,
        }
    }
    let real_ancestor = ancestor.canonicalize().map_err(|e| {
        format!("{UNSAFE_PATH}: {rel} cannot be resolved under the tree root ({e})")
    })?;
    // Component-wise, so `/a/bc` does not count as being under `/a/b`.
    if !real_ancestor.starts_with(&real_root) {
        return Err(format!(
            "{UNSAFE_PATH}: {rel} resolves outside the tree root"
        ));
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_parent_drive_and_unc_paths_are_refused() {
        assert_eq!(
            unsafe_relative_path_reason("/home/user/.ssh/authorized_keys"),
            Some("absolute or UNC path")
        );
        assert_eq!(unsafe_relative_path_reason(""), Some("empty path"));
        assert_eq!(
            unsafe_relative_path_reason("a\u{0}b.py"),
            Some("path contains a NUL byte")
        );
        assert_eq!(
            unsafe_relative_path_reason("../outside.py"),
            Some("parent-directory component")
        );
        assert_eq!(
            unsafe_relative_path_reason("pkg/../../outside.py"),
            Some("parent-directory component")
        );
        assert_eq!(
            unsafe_relative_path_reason("..\\outside.py"),
            Some("parent-directory component")
        );
        assert_eq!(unsafe_relative_path_reason("C:/x.py"), Some("drive prefix"));
        assert_eq!(
            unsafe_relative_path_reason("\\\\server\\share\\x.py"),
            Some("absolute or UNC path")
        );
    }

    #[test]
    fn ordinary_relative_paths_are_allowed() {
        for ok in [
            "cart.py",
            "pkg/mod.py",
            "a/b/c.py",
            "a.b.c",
            "x..y.py",
            "..hidden.py",
            "pkg/..hidden/ok.py",
        ] {
            assert!(is_safe_relative_path(ok), "{ok} must be allowed");
        }
    }

    #[test]
    fn confined_join_refuses_a_symlink_that_leaves_the_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();

        let err = confined_join(root.path(), "link/evil.py").unwrap_err();
        assert!(err.starts_with(UNSAFE_PATH), "got: {err}");
        assert!(
            !outside.path().join("evil.py").exists(),
            "confined_join must write nothing"
        );
        // An ordinary path under the root still joins.
        assert!(confined_join(root.path(), "pkg/ok.py").is_ok());
    }

    #[test]
    fn confined_join_refuses_the_lexical_cases_without_touching_the_disk() {
        let root = tempfile::tempdir().unwrap();
        for bad in ["/etc/passwd", "../x.py", "", "C:/x.py"] {
            let err = confined_join(root.path(), bad).unwrap_err();
            assert!(err.starts_with(UNSAFE_PATH), "{bad}: got {err}");
        }
    }
}
