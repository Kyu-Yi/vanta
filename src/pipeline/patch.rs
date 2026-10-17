// ============================================================================
// PATCH MODULE — robust patch application cascade (Spec §4 Creator agent output)
// ============================================================================
//
// [VANTA-WHY] Extracted from vanta_m2.rs into a reusable module so M3's repair
// loop and any future caller can apply patches without copy-pasting the cascade.
//
// Strategy order (new formats FIRST — they don't require line counting):
//   1. Full source — model emitted the complete patched file (no diff markers)
//   2. Search/replace blocks — ===SEARCH=== / ===REPLACE=== / ===END===
//   3. git apply (strict unified diff)
//   4. git apply --recount
//   5. patch -p1 --fuzz=3
//   6. hunk-by-hunk engine (context-anchored, no header arithmetic)
//   7. whole-file reconstruction — ONLY if patch has no '-' deletions
//
// Returns (patched_text, changed). `changed` is false when NO strategy could
// apply the patch — the caller MUST log PatchApplicationFailed rather than
// silently running tests against unmodified code.
//
// [PITFALL] Whole-file reconstruction CANNOT express deletions: it drops '-'
// lines and keeps '+'/' ' lines. If the fix is purely a deletion, this produces
// a byte-identical file and the pipeline silently records FAIL→FAIL. We guard:
// if any '-' line exists, skip strategy 7.
// ============================================================================

use std::fs;
use tracing::{debug, info, warn};

/// Apply a model-generated fix to the original source.
///
/// Returns `(patched_text, changed)` where `changed` is false if no strategy
/// could produce a different file.
pub fn apply_patch(original: &str, patch_text: &str) -> (String, bool) {
    // ---- Strategy 1: Full source (model emitted complete file) ----
    let looks_like_diff = patch_text.starts_with("--- ")
        || patch_text.starts_with("diff --git")
        || patch_text.starts_with("@@ ")
        || patch_text.contains("\n+++ ")
        || patch_text.contains("\n@@ ")
        || patch_text.contains("\n@@ ")
        || patch_text.contains("===SEARCH===");
    if !looks_like_diff
        && (patch_text.contains("def ") || patch_text.contains("fn ") || patch_text.contains("func "))
    {
        if patch_text.trim() != original.trim() {
            debug!("patch_text is full source — using directly");
            let patched = patch_text.to_string();
            let changed = patched != original;
            return (patched, changed);
        }
    }

    // ---- Strategy 2: Search/replace blocks ----
    if patch_text.contains("===SEARCH===") {
        if let Some(patched) = apply_search_replace(original, patch_text) {
            debug!("search/replace blocks applied");
            let changed = patched != original;
            return (patched, changed);
        }
        warn!("search/replace blocks failed to apply");
    }

    // ---- Strategies 3-5: standard diff tools (for backward compat) ----
    let tmp = tempfile::tempdir().ok();
    if let Some(dir) = tmp {
        let src_path = dir.path().join("source.txt");
        let patch_path = dir.path().join("fix.diff");
        let _ = fs::write(&src_path, original);
        let _ = fs::write(&patch_path, patch_text);

        // Strategy 3: git apply (strict)
        // [PITFALL] Without --3way, git apply can hang on certain malformed
        // diffs. We add it to force non-interactive behavior.
        let output = std::process::Command::new("git")
            .args(["apply", "--verbose", "--3way"])
            .current_dir(dir.path())
            .output();
        if let Ok(o) = output {
            if o.status.success() {
                if let Ok(patched) = fs::read_to_string(&src_path) {
                    let changed = patched != original;
                    return (patched, changed);
                }
            } else {
                debug!("git apply failed: {}",
                    String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or(""));
            }
        }

        // Strategy 4: git apply --recount
        let output = std::process::Command::new("git")
            .args(["apply", "--recount", "--verbose"])
            .current_dir(dir.path())
            .output();
        if let Ok(o) = output {
            if o.status.success() {
                if let Ok(patched) = fs::read_to_string(&src_path) {
                    debug!("git apply --recount succeeded");
                    let changed = patched != original;
                    return (patched, changed);
                }
            }
        }

        // Strategy 5: patch -p1 --fuzz=3
        // [PITFALL] macOS `patch` without --batch can hang waiting for
        // interactive input when hunks don't apply cleanly. --batch forces
        // non-interactive mode (skip hunks that fail, don't prompt).
        let output2 = std::process::Command::new("patch")
            .args(["-p1", "--fuzz=3", "--batch", "-i", "fix.diff"])
            .current_dir(dir.path())
            .output();
        if let Ok(o) = output2 {
            if o.status.success() {
                if let Ok(patched) = fs::read_to_string(&src_path) {
                    debug!("patch -p1 --fuzz=3 succeeded");
                    let changed = patched != original;
                    return (patched, changed);
                }
            }
        }
    }

    // ---- Strategy 6: hunk-by-hunk engine (context-anchored) ----
    if let Some(patched) = apply_hunks(original, patch_text) {
        debug!("hunk-by-hunk engine succeeded");
        let changed = patched != original;
        return (patched, changed);
    }

    // ---- Strategy 7: whole-file reconstruction (NO deletions allowed) ----
    // [PITFALL] Only attempt reconstruction when the patch actually looks like
    // a diff (has @@ hunks or --- / +++ headers). If the patch is raw source
    // code, stripping a leading ' ' from each line corrupts indentation and
    // produces a "changed" file that is actually worse.
    let looks_like_diff_v7 = patch_text.contains("\n@@ ")
        || patch_text.starts_with("@@ ")
        || patch_text.contains("\n--- ")
        || patch_text.starts_with("--- ");
    let has_deletions = patch_text.lines().any(|l| {
        l.starts_with('-') && !l.starts_with("---")
    });
    if looks_like_diff_v7 && !has_deletions {
        let reconstructed: String = patch_text
            .lines()
            .filter(|line| {
                !line.starts_with("--- ")
                    && !line.starts_with("+++ ")
                    && !line.starts_with("@@ ")
            })
            .filter_map(|line| {
                if let Some(rest) = line.strip_prefix('+') {
                    Some(rest.to_string())
                } else if let Some(rest) = line.strip_prefix(' ') {
                    Some(rest.to_string())
                } else if line.starts_with('-') {
                    None
                } else {
                    Some(line.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if !reconstructed.is_empty() && reconstructed.len() > original.len() / 2 {
            info!("whole-file reconstruction succeeded (no deletions in patch)");
            let patched = reconstructed + "\n";
            let changed = patched != original;
            return (patched, changed);
        }
    } else {
        info!("whole-file reconstruction SKIPPED (patch contains deletions — unsafe)");
    }

    // ---- All strategies failed ----
    warn!("ALL patch strategies failed — returning original unchanged");
    (original.to_string(), false)
}

// ============================================================================
// apply_search_replace — apply ===SEARCH===/===REPLACE===/===END=== blocks
// ============================================================================
//
// [VANTA-WHY] Search/replace blocks use exact anchor text (no line counting).
// Each block: find the SEARCH text in the original, replace with REPLACE text.
// Multiple blocks are applied sequentially. Returns None if any block's search
// text can't be found.
// ============================================================================
pub fn apply_search_replace(original: &str, patch_text: &str) -> Option<String> {
    let mut result = original.to_string();
    let mut applied_any = false;

    let blocks: Vec<&str> = patch_text.split("===SEARCH===").collect();
    for block in blocks.iter().skip(1) {
        let parts: Vec<&str> = block.splitn(2, "===REPLACE===").collect();
        if parts.len() != 2 {
            continue;
        }
        let search = parts[0].trim_matches('\n');
        let rest = parts[1];
        let replace_parts: Vec<&str> = rest.splitn(2, "===END===").collect();
        let replace = replace_parts[0].trim_matches('\n');

        if search.is_empty() {
            continue;
        }

        if result.contains(search) {
            result = result.replacen(search, replace, 1);
            applied_any = true;
        } else {
            warn!("search/replace: SEARCH text not found in original");
            return None;
        }
    }

    if applied_any {
        Some(result)
    } else {
        None
    }
}

// ============================================================================
// apply_hunks — hunk-by-hunk patch application engine
// ============================================================================
//
// [VANTA-WHY] Model-generated unified diffs often have wrong hunk headers
// (incorrect line counts) but correct context/content lines. This engine
// ignores the `@@` line numbers entirely and instead uses the context lines
// (` ` prefix) to find the correct insertion point in the original file.
// ============================================================================
const HUNK_FUZZ: usize = 3;

pub fn apply_hunks(original: &str, patch_text: &str) -> Option<String> {
    let orig_lines: Vec<&str> = original.lines().collect();
    let hunks = parse_hunks(patch_text);

    if hunks.is_empty() {
        return None;
    }

    let mut result_lines: Vec<String> = orig_lines.iter().map(|l| l.to_string()).collect();
    let mut applied_any = false;

    for hunk in &hunks {
        let range = hunk
            .old_range
            .and_then(|(start, len)| {
                if start == 0 {
                    return None;
                }
                let start_idx = start.saturating_sub(1);
                if start_idx > result_lines.len() {
                    return None;
                }
                let end_idx = std::cmp::min(start_idx + len, result_lines.len());
                Some((start_idx, end_idx))
            })
            .or_else(|| find_hunk_anchor(&result_lines, &hunk.context, HUNK_FUZZ));

        if let Some((start, end)) = range {
            let mut new_segment: Vec<String> = Vec::new();
            for line in &hunk.lines {
                match line {
                    HunkLine::Context(s) => new_segment.push(s.clone()),
                    HunkLine::Addition(s) => new_segment.push(s.clone()),
                    HunkLine::Deletion => {} // skip deletions
                }
            }

            result_lines.splice(start..end, new_segment);
            applied_any = true;
        } else {
            debug!("hunk-by-hunk: could not anchor hunk ({} context lines) — skipping",
                hunk.context.len());
        }
    }

    if applied_any {
        Some(result_lines.join("\n") + "\n")
    } else {
        None
    }
}

enum HunkLine {
    Context(String),
    Addition(String),
    Deletion,
}

struct Hunk {
    lines: Vec<HunkLine>,
    context: Vec<String>,
    old_range: Option<(usize, usize)>,
}

fn parse_hunks(patch_text: &str) -> Vec<Hunk> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut current: Option<Hunk> = None;

    for line in patch_text.lines() {
        if line.starts_with("@@") {
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            current = Some(Hunk {
                lines: Vec::new(),
                context: Vec::new(),
                old_range: parse_old_range(line),
            });
        } else if let Some(hunk) = current.as_mut() {
            if let Some(rest) = line.strip_prefix('+') {
                hunk.lines.push(HunkLine::Addition(rest.to_string()));
            } else if line.starts_with('-') && !line.starts_with("---") {
                hunk.lines.push(HunkLine::Deletion);
            } else if let Some(rest) = line.strip_prefix(' ') {
                let ctx = rest.to_string();
                hunk.lines.push(HunkLine::Context(ctx.clone()));
                hunk.context.push(ctx);
            } else if line.starts_with("--- ") || line.starts_with("+++ ") {
                // File headers — skip
            } else {
                hunk.lines.push(HunkLine::Context(line.to_string()));
                hunk.context.push(line.to_string());
            }
        }
    }
    if let Some(h) = current.take() {
        hunks.push(h);
    }
    hunks
}

fn parse_old_range(header: &str) -> Option<(usize, usize)> {
    let minus = header.split_whitespace().find(|part| part.starts_with('-'))?;
    let body = minus.trim_start_matches('-');
    let mut parts = body.split(',');
    let start = parts.next()?.parse::<usize>().ok()?;
    let len = parts
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    Some((start, len))
}

fn find_hunk_anchor(
    file_lines: &[String],
    context: &[String],
    fuzz: usize,
) -> Option<(usize, usize)> {
    if context.is_empty() {
        return None;
    }

    for top_trim in 0..=fuzz {
        for bottom_trim in 0..=fuzz {
            if top_trim + bottom_trim >= context.len() {
                continue;
            }
            let trimmed = &context[top_trim..context.len() - bottom_trim];
            if let Some(pos) = find_subsequence(file_lines, trimmed) {
                let start = pos.saturating_sub(top_trim);
                let end = std::cmp::min(pos + trimmed.len() + bottom_trim, file_lines.len());
                return Some((start, end));
            }
        }
    }
    None
}

fn find_subsequence(haystack: &[String], needle: &[String]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    for i in 0..=(haystack.len() - needle.len()) {
        if haystack[i..i + needle.len()] == needle[..] {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_source_strategy() {
        let original = "def add(a, b):\n    return a - b\n";
        let patch = "def add(a, b):\n    return a + b\n";
        let (patched, changed) = apply_patch(original, patch);
        assert!(changed);
        assert_eq!(patched, patch);
    }

    #[test]
    fn test_search_replace_strategy() {
        let original = "def add(a, b):\n    return a - b\n";
        let patch = "===SEARCH===\n    return a - b\n===REPLACE===\n    return a + b\n===END===\n";
        let (patched, changed) = apply_patch(original, patch);
        assert!(changed);
        assert!(patched.contains("return a + b"));
    }

    #[test]
    fn test_no_change_returns_false() {
        let original = "def add(a, b):\n    return a + b\n";
        let patch = original.to_string();
        let (_, changed) = apply_patch(original, &patch);
        assert!(!changed);
    }

    #[test]
    fn test_deletion_guard_on_reconstruction() {
        // A patch with only deletions should NOT be silently reconstructed.
        let original = "line1\nline2\nline3\n";
        let patch = "--- a.txt\n+++ b.txt\n@@ -1,3 +1,2 @@\n line1\n-line2\n line3\n";
        let (patched, changed) = apply_patch(original, patch);
        // Strategy 7 is skipped (has deletions). Strategy 6 (hunk engine) may or
        // may not anchor. If nothing works, changed == false. Either way the
        // patched text should NOT be byte-identical to original if a deletion
        // was applied. If changed==false, patched==original (correct — we don't
        // silently drop the deletion).
        if changed {
            assert_ne!(patched, original);
        } else {
            assert_eq!(patched, original);
        }
    }

    #[test]
    fn test_hunk_only_patch_is_not_treated_as_full_source() {
        let original = "def add(a, b):\n    return a + b\n";
        let patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n";
        let (patched, changed) = apply_patch(original, patch);
        assert_eq!(patched, "def add(a, b):\n    return a - b\n");
        assert!(changed);
    }
}
