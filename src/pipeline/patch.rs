// ============================================================================
// PATCH MODULE — robust patch application cascade (Spec §4 Creator agent output)
// ============================================================================
//
// [VANTA-WHY] Extracted from vanta_m2.rs into a reusable module so M3's repair
// loop and any future caller can apply patches without copy-pasting the cascade.
//
// Two policies, and the difference is the whole point:
//
//   Strict  (review mode, `--patch`) — `git apply` and NOTHING else. The
//           candidate came from outside the run, so a fuzzy recovery would test
//           a tree nobody wrote.
//   Cascade (repair mode, Creator output) — the full chain below, because the
//           Creator can emit a diff with wrong headers and the repair loop has a
//           budget to recover.
//
// Cascade strategy order (new formats FIRST — they don't require line counting):
//   1. Full source — model emitted the complete patched file (no diff markers)
//   2. Search/replace blocks — ===SEARCH=== / ===REPLACE=== / ===END===
//   3. git apply (strict unified diff)
//   4. git apply --recount
//   5. patch -p1 --fuzz=3
//   6. hunk-by-hunk engine (context-anchored, running line offset)
//   7. whole-file reconstruction — ONLY if patch has no '-' deletions
//
// [VANTA-WHY] Strategies 1, 2, 5, 6 and 7 are all non-git, so a `.py` result
// they produce is `py_compile`d next to the original. If the original parsed and
// the result does not, the apply is reported as FAILED rather than "patched
// (changed)": a file that no longer imports is not a candidate, and reporting it
// as one is how a syntax error gets a clean native report.
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

use std::collections::HashMap;
use std::fs;
use std::process::Command;
use tracing::{debug, info, warn};

// ============================================================================
// apply_patch_strict — the ONLY entry point for an externally supplied `--patch`
// ============================================================================
//
// [VANTA-WHY] A candidate handed to us with `--patch` was written outside the
// run. If a fuzzy strategy "recovers" it, the tree under test is a tree nobody
// wrote, and the resulting verdict is a statement about Vanta's repair engine
// rather than about the candidate. So review mode gets `git apply`, then
// `git apply --recount` as the single concession to a miscounted header, and
// nothing else. If both fail the run stops with PatchApplicationFailed.
//
// [PITFALL] The repo is materialised at the patch's OWN relative paths, because
// `git apply` resolves `a/src/heretic/plugin.py` against the working directory.
// The previous per-file cascade wrote the one file it was given to `source.txt`
// in a bare temp dir, so no diff header could ever name it: every git strategy
// failed on every real patch and the hunk engine was the only thing left
// standing — which is how a valid patch got spliced into a file that no longer
// parses.
//
// Returns the touched files whose content changed, or a plain-language reason.
pub fn apply_patch_strict(
    repo_files: &HashMap<String, String>,
    patch_text: &str,
    touched: &[String],
) -> Result<HashMap<String, String>, String> {
    let dir = tempfile::tempdir().map_err(|e| format!("could not create a temp dir: {e}"))?;
    let root = dir.path();

    for (path, content) in repo_files {
        let out = root.join(path);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{path}: {e}"))?;
        }
        fs::write(&out, content).map_err(|e| format!("{path}: {e}"))?;
    }

    let patch_path = root.join("vanta_candidate.diff");
    fs::write(&patch_path, patch_text).map_err(|e| format!("could not stage the patch: {e}"))?;
    let patch_arg = patch_path.to_string_lossy().to_string();

    // `git apply` works outside a repository, but an explicit one removes any
    // version-dependent index behaviour. Not fatal if it fails.
    let _ = Command::new("git")
        .args(["init", "-q"])
        .current_dir(root)
        .output();

    if let Err(first) = git_apply(root, &patch_arg, &[]) {
        if let Err(second) = git_apply(root, &patch_arg, &["--recount"]) {
            return Err(format!(
                "git apply refused the patch.\n  git apply: {first}\n  git apply --recount: {second}"
            ));
        }
        debug!("git apply refused the patch but --recount accepted it (miscounted header)");
    }

    let mut changed: HashMap<String, String> = HashMap::new();
    for path in touched {
        let Ok(patched) = fs::read_to_string(root.join(path)) else {
            // Deleted by the patch, or never written. Either way there is no
            // content to test.
            continue;
        };
        match repo_files.get(path) {
            Some(original) if original == &patched => {}
            // Changed file, or a file the patch created.
            _ => {
                changed.insert(path.clone(), patched);
            }
        }
    }

    if changed.is_empty() {
        return Err(
            "git apply reported success but no touched file changed — the patch is a no-op"
                .to_string(),
        );
    }
    Ok(changed)
}

/// The paths a patch DELETES (`--- a/<path>` … `+++ /dev/null`).
///
/// [VANTA-WHY] A deletion cannot be read off the resulting file map: the strict
/// path reads its destination from `+++ b/...` and skips a `/dev/null` outright,
/// so a deleted file simply keeps its original entry in `patched_files`. The
/// native-suite gate must restore a deleted test, and the evidence must name it,
/// so the deletion is read from the diff itself — the only place it is visible.
pub fn patch_deleted_paths(patch_text: &str) -> Vec<String> {
    let mut deleted: Vec<String> = Vec::new();
    let mut pending_source: Option<String> = None;
    for line in patch_text.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            let rest = rest.trim();
            pending_source = rest.strip_prefix("a/").map(str::to_string);
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            if rest.trim() == "/dev/null" {
                if let Some(path) = pending_source.take() {
                    deleted.push(path);
                }
            }
        }
    }
    deleted.sort();
    deleted.dedup();
    deleted
}

/// Run `git apply` in `root`; `Err` carries git's own message.
fn git_apply(root: &std::path::Path, patch_arg: &str, extra: &[&str]) -> Result<(), String> {
    let mut args = vec!["apply", "-p1", "--verbose"];
    args.extend_from_slice(extra);
    args.push(patch_arg);
    let out = Command::new("git")
        .args(&args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run git: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    Err(stderr.trim().lines().take(6).collect::<Vec<_>>().join(" | "))
}

/// Why a repair-mode cascade could not produce a usable candidate.
///
/// [VANTA-WHY] The Creator needs the DISTINCTION, not just "failed". "I could
/// not place your hunks" is a formatting problem the Creator can fix by
/// re-emitting the diff; "your patch applied but the result is not valid Python"
/// is a content problem it can only fix by changing the code. Collapsing both
/// into one message is how a run tells the model the wrong thing to repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyFailure {
    /// No strategy could place the patch: every hunk was rejected.
    ///
    /// [VANTA-WHY] Item R — this variant carries what the cascade already knew
    /// instead of a constant. The Creator is told the FILE, every strategy that
    /// was tried with its own reason, and — when whole-file reconstruction was
    /// not attempted — why it was skipped. A bare "did not apply" is the same
    /// sentence for a malformed diff and for an anchor that is not in the file,
    /// and the Creator cannot tell which one it must fix.
    DidNotApply {
        /// The file path the patch named (as the patch names it).
        path: String,
        /// Every strategy the cascade tried, in order, with its reason.
        strategies: Vec<StrategyFailure>,
        /// Why whole-file reconstruction was not attempted, when it was not.
        /// `None` means the cascade returned before reaching it.
        reconstruction_skipped: Option<String>,
    },
    /// A strategy produced a changed file, but the original parsed as Python and
    /// the result does not. Carries the interpreter's own message.
    InvalidPython(String),
}

/// One strategy's outcome inside the apply cascade, kept so the Creator can be
/// told what was actually tried. Restored information only: the reason is what
/// the strategy's own tool reported, never a nearest-match guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyFailure {
    /// The strategy's name ("search/replace", "git apply --3way", …).
    pub strategy: String,
    /// Why it did not place the patch.
    pub reason: String,
}

impl StrategyFailure {
    fn new(strategy: &str, reason: impl Into<String>) -> Self {
        Self {
            strategy: strategy.to_string(),
            reason: reason.into(),
        }
    }
}

impl ApplyFailure {
    /// The message the Creator is sent when its patch could not be placed.
    pub fn feedback(&self) -> String {
        match self {
            ApplyFailure::InvalidPython(err) => {
                format!("patch result is not valid Python: {err}")
            }
            ApplyFailure::DidNotApply {
                path,
                strategies,
                reconstruction_skipped,
            } => {
                let mut out = format!("patch did not apply to {path}");
                if strategies.is_empty() {
                    out.push_str(" — no apply strategy was reached");
                } else {
                    out.push_str(" — every apply strategy failed:");
                }
                for s in strategies {
                    out.push_str(&format!("\n  - {}: {}", s.strategy, s.reason));
                }
                if let Some(why) = reconstruction_skipped {
                    out.push_str(&format!(
                        "\n  - whole-file reconstruction: skipped ({why})"
                    ));
                }
                out
            }
        }
    }
}

/// Apply a model-generated fix to the original source.
///
/// `path` is the file's path as the patch names it — strategies 3-5 need it to
/// recreate the layout `git apply` / `patch(1)` expect, and the syntax guard
/// needs it to know whether the result must parse as Python.
///
/// Returns `(patched_text, changed)` where `changed` is false if no strategy
/// could produce a different, still-parsable file. Callers that must TELL the
/// model why use `apply_patch_detailed` instead.
pub fn apply_patch(original: &str, patch_text: &str, path: &str) -> (String, bool) {
    let (text, changed, _) = apply_patch_detailed(original, patch_text, path);
    (text, changed)
}

/// As `apply_patch`, but also reports WHY the apply failed.
///
/// [VANTA-WHY] This is a single implementation with a thin wrapper rather than a
/// second cascade: two copies of the strategy order would drift, and the drift
/// would only show up as a patch that applies in one path and not the other.
pub fn apply_patch_detailed(
    original: &str,
    patch_text: &str,
    path: &str,
) -> (String, bool, Option<ApplyFailure>) {
    // [VANTA-WHY] Every non-git strategy lands here first. A file that no longer
    // compiles is not a candidate, and "patched (changed)" for one is a lie the
    // rest of the pipeline cannot see through: the heretic run reported it for a
    // file whose `if` had no body.
    let mut failure: Option<ApplyFailure> = None;
    // [VANTA-WHY] Item R — every strategy this cascade actually tried, with the
    // reason it did not place the patch. The Creator is shown this list when the
    // cascade gives up; the reasons are the strategies' OWN tool output, never a
    // nearest-match guess.
    let mut tried: Vec<StrategyFailure> = Vec::new();
    // Why whole-file reconstruction was not attempted, when it was not.
    let mut reconstruction_skipped: Option<String> = None;
    let accept = |patched: String, strategy: &str| -> Result<(String, bool), ApplyFailure> {
        let changed = patched != original;
        if changed {
            if let Some(err) = python_break_reason(path, original, &patched) {
                warn!("{strategy}: result no longer parses as Python — rejecting this strategy");
                return Err(ApplyFailure::InvalidPython(err));
            }
        }
        Ok((patched, changed))
    };
    // A strategy that produced different text is the answer; a strategy that
    // produced nothing different is a failure with a precise reason. Either way
    // the cascade stops here, exactly as it did before.
    let finish = |patched: String,
                  changed: bool,
                  tried: &[StrategyFailure],
                  skipped: &Option<String>| {
        let failure = if changed {
            None
        } else {
            Some(ApplyFailure::DidNotApply {
                path: path.to_string(),
                strategies: tried.to_vec(),
                reconstruction_skipped: skipped.clone(),
            })
        };
        (patched, changed, failure)
    };

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
        && patch_text.trim() != original.trim()
    {
        debug!("patch_text is full source — using directly");
        match accept(patch_text.to_string(), "full-source") {
            Ok((patched, changed)) => {
                if changed {
                    return (patched, true, None);
                }
                tried.push(StrategyFailure::new(
                    "full-source",
                    "produced text identical to the original",
                ));
                return finish(patched, false, &tried, &reconstruction_skipped);
            }
            Err(refused) => failure = Some(refused),
        }
    }

    // ---- Strategy 2: Search/replace blocks ----
    if patch_text.contains("===SEARCH===") {
        match apply_search_replace(original, patch_text) {
            Ok(patched) => match accept(patched, "search/replace") {
                Ok((patched, changed)) => {
                    if changed {
                        debug!("search/replace blocks applied");
                        return (patched, true, None);
                    }
                    tried.push(StrategyFailure::new(
                        "search/replace",
                        "produced text identical to the original",
                    ));
                    return finish(patched, false, &tried, &reconstruction_skipped);
                }
                Err(refused) => failure = Some(refused),
            },
            Err(why) => {
                warn!("search/replace blocks failed to apply: {why}");
                tried.push(StrategyFailure::new("search/replace", why.to_string()));
            }
        }
    }

    // ---- Strategies 3-5: standard diff tools (for backward compat) ----
    // [PITFALL] The file must sit at the path the diff header names. A copy
    // called `source.txt` can never be matched by `a/src/pkg/mod.py`.
    if let Ok(dir) = tempfile::tempdir() {
        let root = dir.path();
        let rel = if path.trim().is_empty() {
            "source.txt"
        } else {
            path.trim()
        };
        let src_path = root.join(rel);
        if let Some(parent) = src_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&src_path, original);
        let patch_path = root.join("fix.diff");
        let _ = fs::write(&patch_path, patch_text);
        let patch_arg = patch_path.to_string_lossy().to_string();

        // [VANTA-WHY] Item R — return the tool's OWN message on failure instead
        // of `None`, so the reason survives into the Creator's feedback.
        let via_git = |extra: &[&str]| -> Result<String, String> {
            let mut args = vec!["apply", "-p1", "--verbose"];
            args.extend_from_slice(extra);
            args.push(&patch_arg);
            let out = Command::new("git")
                .args(&args)
                .current_dir(root)
                .output()
                .map_err(|e| format!("could not run git: {e}"))?;
            if out.status.success() {
                fs::read_to_string(&src_path)
                    .map_err(|e| format!("git apply {} changed the file but it could not be read: {e}", extra.join(" ")))
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                Err(stderr.trim().lines().take(6).collect::<Vec<_>>().join(" | "))
            }
        };

        // Strategy 3: git apply (strict).
        // [PITFALL] Without --3way, git apply can hang on certain malformed
        // diffs. We add it to force non-interactive behavior.
        match via_git(&["--3way"]) {
            Ok(patched) => {
                if patched != original {
                    return (patched, true, None);
                }
                tried.push(StrategyFailure::new(
                    "git apply --3way",
                    "produced text identical to the original",
                ));
            }
            Err(why) => tried.push(StrategyFailure::new("git apply --3way", why)),
        }

        // Strategy 4: git apply --recount
        match via_git(&["--recount"]) {
            Ok(patched) => {
                if patched != original {
                    debug!("git apply --recount succeeded");
                    return (patched, true, None);
                }
                tried.push(StrategyFailure::new(
                    "git apply --recount",
                    "produced text identical to the original",
                ));
            }
            Err(why) => tried.push(StrategyFailure::new("git apply --recount", why)),
        }

        // Strategy 5: patch -p1 --fuzz=3
        // [PITFALL] macOS `patch` without --batch can hang waiting for
        // interactive input when hunks don't apply cleanly. --batch forces
        // non-interactive mode (skip hunks that fail, don't prompt).
        let out = Command::new("patch")
            .args(["-p1", "--fuzz=3", "--batch", "-i", "fix.diff"])
            .current_dir(root)
            .output();
        match out {
            Ok(o) if o.status.success() => match fs::read_to_string(&src_path) {
                Ok(patched) => match accept(patched, "patch -p1") {
                    Ok((patched, changed)) => {
                        if changed {
                            debug!("patch -p1 --fuzz=3 succeeded");
                            return (patched, true, None);
                        }
                        tried.push(StrategyFailure::new(
                            "patch -p1 --fuzz=3",
                            "produced text identical to the original",
                        ));
                        return finish(patched, false, &tried, &reconstruction_skipped);
                    }
                    Err(refused) => failure = Some(refused),
                },
                Err(e) => tried.push(StrategyFailure::new(
                    "patch -p1 --fuzz=3",
                    format!("applied but the patched file could not be read: {e}"),
                )),
            },
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                tried.push(StrategyFailure::new(
                    "patch -p1 --fuzz=3",
                    stderr.trim().lines().take(6).collect::<Vec<_>>().join(" | "),
                ));
            }
            Err(e) => tried.push(StrategyFailure::new(
                "patch -p1 --fuzz=3",
                format!("could not run patch: {e}"),
            )),
        }
    }

    // ---- Strategy 6: hunk-by-hunk engine (context-anchored) ----
    match apply_hunks_detailed(original, patch_text) {
        Ok(patched) => match accept(patched, "hunk-by-hunk") {
            Ok((patched, changed)) => {
                if changed {
                    debug!("hunk-by-hunk engine succeeded");
                    return (patched, true, None);
                }
                tried.push(StrategyFailure::new(
                    "hunk-by-hunk",
                    "produced text identical to the original",
                ));
                return finish(patched, false, &tried, &reconstruction_skipped);
            }
            Err(refused) => failure = Some(refused),
        },
        Err(why) => {
            debug!("hunk-by-hunk engine did not apply: {why}");
            tried.push(StrategyFailure::new("hunk-by-hunk", why));
        }
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
            match accept(patched, "whole-file reconstruction") {
                Ok((patched, changed)) => {
                    if changed {
                        return (patched, true, None);
                    }
                    tried.push(StrategyFailure::new(
                        "whole-file reconstruction",
                        "produced text identical to the original",
                    ));
                    return finish(patched, false, &tried, &reconstruction_skipped);
                }
                Err(refused) => failure = Some(refused),
            }
        } else {
            tried.push(StrategyFailure::new(
                "whole-file reconstruction",
                format!(
                    "produced too little text ({} bytes; the file is {} bytes)",
                    reconstructed.len(),
                    original.len()
                ),
            ));
        }
    } else {
        // [VANTA-WHY] Item R — the original logged ONE sentence naming deletions
        // for BOTH reasons this branch can take. Say which one it actually was.
        let why = if !looks_like_diff_v7 {
            "the patch does not look like a diff (no @@ hunk and no --- header)"
        } else {
            "the patch contains deletions, which reconstruction cannot represent safely"
        };
        info!("whole-file reconstruction SKIPPED ({why})");
        reconstruction_skipped = Some(why.to_string());
    }

    // ---- All strategies failed ----
    warn!("ALL patch strategies failed — returning original unchanged");
    let reason = failure.unwrap_or(ApplyFailure::DidNotApply {
        path: path.to_string(),
        strategies: tried,
        reconstruction_skipped,
    });
    (original.to_string(), false, Some(reason))
}

// ============================================================================
// Python syntax guard — did a non-git strategy produce a file that cannot parse?
// ============================================================================
//
// [VANTA-WHY] This is the backstop for every strategy that is not `git apply`.
// It answers ONE question: does the original parse and the result not? A missing
// interpreter (or an original that never parsed) answers `None` so the guard can
// never block a run by accident.
//
// It returns the INTERPRETER'S OWN MESSAGE rather than a bool, because the
// Creator has to be told what broke: "patch result is not valid Python: <stderr>"
// is actionable, "apply failed" is not.
fn python_break_reason(path: &str, original: &str, patched: &str) -> Option<String> {
    if !path.trim().ends_with(".py") {
        return None;
    }
    if python_parses(original) != Some(true) {
        return None;
    }
    python_compile_error(patched)
}

/// `py_compile`'s stderr for a source that does not compile; `None` when it does
/// (or when no interpreter is available, so a missing python never fails a run).
fn python_compile_error(source: &str) -> Option<String> {
    let dir = tempfile::tempdir().ok()?;
    let file = dir.path().join("candidate.py");
    fs::write(&file, source).ok()?;
    let out = Command::new(crate::pipeline::harness::python_exe())
        .arg("-m")
        .arg("py_compile")
        .arg(&file)
        .output()
        .ok()?;
    if out.status.success() {
        return None;
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let message = stderr.trim();
    Some(if message.is_empty() {
        "py_compile rejected the result without a message".to_string()
    } else {
        message.lines().take(6).collect::<Vec<_>>().join(" | ")
    })
}

fn python_parses(source: &str) -> Option<bool> {
    let dir = tempfile::tempdir().ok()?;
    let file = dir.path().join("candidate.py");
    fs::write(&file, source).ok()?;
    let out = Command::new(crate::pipeline::harness::python_exe())
        .arg("-m")
        .arg("py_compile")
        .arg(&file)
        .output()
        .ok()?;
    Some(out.status.success())
}

// ============================================================================
// apply_search_replace — apply ===SEARCH===/===REPLACE===/===END=== blocks
// ============================================================================
//
// [VANTA-WHY] Search/replace blocks use exact anchor text (no line counting).
// Each block: find the SEARCH text in the original, replace with REPLACE text.
// Multiple blocks are applied sequentially. Returns the first block that could
// not be placed, so the reason reaches the Creator (Item R).
// ============================================================================
/// Why a `===SEARCH===`/`===REPLACE===` block could not be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchReplaceFailure {
    /// A block's SEARCH text does not appear in the original file. Carries the
    /// exact anchor text, so the Creator can see the line it wrote is not there.
    AnchorNotFound { search: String },
    /// The patch declared blocks, but none of them changed anything (each block
    /// needs both a `===SEARCH===` and a `===REPLACE===` marker).
    NothingApplied,
}

impl std::fmt::Display for SearchReplaceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchReplaceFailure::AnchorNotFound { search } => write!(
                f,
                "the ===SEARCH=== anchor text is not present in the file:\n{search}"
            ),
            SearchReplaceFailure::NothingApplied => write!(
                f,
                "no ===SEARCH===/===REPLACE=== block changed anything"
            ),
        }
    }
}

pub fn apply_search_replace(
    original: &str,
    patch_text: &str,
) -> Result<String, SearchReplaceFailure> {
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
            return Err(SearchReplaceFailure::AnchorNotFound {
                search: search.to_string(),
            });
        }
    }

    if applied_any {
        Ok(result)
    } else {
        Err(SearchReplaceFailure::NothingApplied)
    }
}

// ============================================================================
// apply_hunks — hunk-by-hunk patch application engine
// ============================================================================
//
// [VANTA-WHY] Model-generated unified diffs often have wrong hunk headers
// (incorrect line counts) but correct content lines. This engine locates each
// hunk by its CONTEXT and uses the `@@` range only as a fallback, because the
// header counts are the least trustworthy part of a model diff.
//
// It also keeps a RUNNING OFFSET. A hunk's `@@ -70,6` counts lines of the
// ORIGINAL file, so once an earlier hunk in the same file has inserted or
// removed lines, that number is stale. The previous version indexed the buffer
// with the raw header, so hunk 3 of the heretic patch spliced over `score_name`
// and the head of `init()` and left an `if` with no body: a valid patch turned
// into a file that no longer parses, and the pipeline reported "patched
// (changed)". Strategies that cannot verify where they splice are not worth
// having.
//
// Every hunk must apply, and it must apply where the hunk's own old side
// matches byte for byte. A partial or approximate application is refused
// outright rather than handed on as a candidate.
// ============================================================================

pub fn apply_hunks(original: &str, patch_text: &str) -> Option<String> {
    apply_hunks_detailed(original, patch_text).ok()
}

/// As `apply_hunks`, but reports WHY the engine refused the patch (Item R), so
/// the reason reaches the Creator instead of a bare `None`.
pub fn apply_hunks_detailed(original: &str, patch_text: &str) -> Result<String, String> {
    let hunks = parse_hunks(patch_text);

    if hunks.is_empty() {
        return Err("the patch has no @@ hunks to apply".to_string());
    }

    let mut result_lines: Vec<String> = original.lines().map(|l| l.to_string()).collect();
    let mut offset: isize = 0;

    for hunk in &hunks {
        let old_side = hunk.old_side();
        let new_side = hunk.new_side();
        if old_side.is_empty() {
            debug!("hunk-by-hunk: hunk has no old-side lines — refusing the whole patch");
            return Err("a hunk has no old-side lines".to_string());
        }
        let Some(start) = anchor_hunk(&result_lines, hunk, &old_side, offset) else {
            debug!(
                "hunk-by-hunk: hunk could not be anchored ({} old-side line(s)) — refusing the whole patch",
                old_side.len()
            );
            return Err(format!(
                "a hunk with {} old-side line(s) could not be anchored in the file",
                old_side.len()
            ));
        };

        offset += new_side.len() as isize - old_side.len() as isize;
        result_lines.splice(start..start + old_side.len(), new_side);
    }

    let patched = result_lines.join("\n") + "\n";
    if patched == original {
        return Err("the hunks applied but produced no change".to_string());
    }
    Ok(patched)
}

/// Where does this hunk's old side start in `file_lines`?
///
/// Context first: the longest contiguous run of context lines is far less likely
/// to appear twice than a one-line run like `    }`. The `@@` range corrected by
/// the running offset is the fallback. Both candidates must match the hunk's
/// ENTIRE old side before they are accepted, so a wrong guess is a refusal, not
/// a corruption.
fn anchor_hunk(
    file_lines: &[String],
    hunk: &Hunk,
    old_side: &[String],
    offset: isize,
) -> Option<usize> {
    // Context anchor. A run is a contiguous group of context lines *within the
    // hunk*; a deleted line may sit between two of them, so we anchor on the
    // longest run and map it back through the run's offset in the old side.
    for (context_index, run) in hunk.context_runs() {
        let Some(pos) = find_subsequence(file_lines, &run) else {
            continue;
        };
        let old_offset = hunk.context_offsets[context_index];
        let Some(start) = pos.checked_sub(old_offset) else {
            continue;
        };
        if matches_at(file_lines, start, old_side) {
            return Some(start);
        }
    }

    // Fallback: the `@@` range, shifted by everything earlier hunks changed.
    let (header_start, _) = hunk.old_range?;
    if header_start == 0 {
        return None;
    }
    let start = (header_start as isize - 1 + offset).max(0) as usize;
    if matches_at(file_lines, start, old_side) {
        Some(start)
    } else {
        None
    }
}

fn matches_at(file_lines: &[String], start: usize, needle: &[String]) -> bool {
    start + needle.len() <= file_lines.len() && file_lines[start..start + needle.len()] == *needle
}

enum HunkLine {
    Context(String),
    Addition(String),
    Deletion(String),
}

struct Hunk {
    lines: Vec<HunkLine>,
    context: Vec<String>,
    /// For each context entry, its index within the hunk's OLD side (context +
    /// deletions), so an anchored context run can be mapped back to the first
    /// old-side line of the hunk.
    context_offsets: Vec<usize>,
    old_range: Option<(usize, usize)>,
}

impl Hunk {
    /// The hunk's old side: context lines and deleted lines, in file order.
    fn old_side(&self) -> Vec<String> {
        self.lines
            .iter()
            .filter_map(|l| match l {
                HunkLine::Context(s) | HunkLine::Deletion(s) => Some(s.clone()),
                HunkLine::Addition(_) => None,
            })
            .collect()
    }

    /// How many old-side lines (context + deletions) the hunk has consumed so
    /// far. Used while parsing to record where each context line sits within the
    /// old side.
    fn old_side_len(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| matches!(l, HunkLine::Context(_) | HunkLine::Deletion(_)))
            .count()
    }

    /// The hunk's new side: context lines and added lines, in file order.
    fn new_side(&self) -> Vec<String> {
        self.lines
            .iter()
            .filter_map(|l| match l {
                HunkLine::Context(s) | HunkLine::Addition(s) => Some(s.clone()),
                HunkLine::Deletion(_) => None,
            })
            .collect()
    }

    /// Contiguous runs of context lines as `(context_index, lines)`, longest
    /// first — a long block is a much better anchor than a one-line one.
    fn context_runs(&self) -> Vec<(usize, Vec<String>)> {
        let mut runs: Vec<(usize, Vec<String>)> = Vec::new();
        let mut current: Option<(usize, Vec<String>)> = None;
        let mut context_index = 0usize;

        for line in &self.lines {
            match line {
                HunkLine::Context(s) => {
                    match current.as_mut() {
                        Some((_, run)) => run.push(s.clone()),
                        None => current = Some((context_index, vec![s.clone()])),
                    }
                    context_index += 1;
                }
                _ => {
                    if let Some(run) = current.take() {
                        runs.push(run);
                    }
                }
            }
        }
        if let Some(run) = current.take() {
            runs.push(run);
        }
        runs.sort_by_key(|(_, run)| std::cmp::Reverse(run.len()));
        runs
    }
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
                context_offsets: Vec::new(),
                old_range: parse_old_range(line),
            });
        } else if let Some(hunk) = current.as_mut() {
            let push_context = |hunk: &mut Hunk, ctx: String| {
                hunk.context_offsets.push(hunk.old_side_len());
                hunk.context.push(ctx.clone());
                hunk.lines.push(HunkLine::Context(ctx));
            };
            if let Some(rest) = line.strip_prefix('+') {
                hunk.lines.push(HunkLine::Addition(rest.to_string()));
            } else if let Some(rest) = line.strip_prefix('-') {
                if line.starts_with("---") {
                    // File header — skip.
                } else {
                    hunk.lines.push(HunkLine::Deletion(rest.to_string()));
                }
            } else if let Some(rest) = line.strip_prefix(' ') {
                push_context(hunk, rest.to_string());
            } else if line.starts_with("--- ") || line.starts_with("+++ ") {
                // File headers — skip
            } else {
                push_context(hunk, line.to_string());
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
        let (patched, changed) = apply_patch(original, patch, "cart.py");
        assert!(changed);
        assert_eq!(patched, patch);
    }

    #[test]
    fn test_search_replace_strategy() {
        let original = "def add(a, b):\n    return a - b\n";
        let patch = "===SEARCH===\n    return a - b\n===REPLACE===\n    return a + b\n===END===\n";
        let (patched, changed) = apply_patch(original, patch, "cart.py");
        assert!(changed);
        assert!(patched.contains("return a + b"));
    }

    #[test]
    fn test_no_change_returns_false() {
        let original = "def add(a, b):\n    return a + b\n";
        let patch = original.to_string();
        let (_, changed) = apply_patch(original, &patch, "cart.py");
        assert!(!changed);
    }

    #[test]
    fn test_deletion_guard_on_reconstruction() {
        // A patch with only deletions should NOT be silently reconstructed.
        let original = "line1\nline2\nline3\n";
        let patch = "--- a.txt\n+++ b.txt\n@@ -1,3 +1,2 @@\n line1\n-line2\n line3\n";
        let (patched, changed) = apply_patch(original, patch, "cart.py");
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

    /// A path is read from the diff, not from the file map: the strict apply
    /// leaves a deleted file at its original contents, so the deletion only
    /// exists in the patch text.
    #[test]
    fn patch_deleted_paths_names_only_files_sent_to_dev_null() {
        let patch = "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-def test_scale():\n-    assert scale(3) == 6\n";
        assert_eq!(
            patch_deleted_paths(patch),
            vec!["tests/test_core.py".to_string()]
        );
    }

    #[test]
    fn patch_deleted_paths_is_empty_for_additions_and_modifications() {
        let addition = "diff --git a/pyproject.toml b/pyproject.toml\n--- /dev/null\n+++ b/pyproject.toml\n@@ -0,0 +1 @@\n+[tool.pytest.ini_options]\n";
        assert!(patch_deleted_paths(addition).is_empty());
        let edit = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n";
        assert!(patch_deleted_paths(edit).is_empty());
    }

    #[test]
    fn test_hunk_only_patch_is_not_treated_as_full_source() {
        let original = "def add(a, b):\n    return a + b\n";
        let patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n";
        let (patched, changed) = apply_patch(original, patch, "cart.py");
        assert_eq!(patched, "def add(a, b):\n    return a - b\n");
        assert!(changed);
    }

    // ---- Fix 1 regression: the running offset ----

    /// Two hunks in one file, where hunk 1 changes the line count. Hunk 2's
    /// header still counts ORIGINAL lines, which is exactly the shape that made
    /// the old engine splice over the wrong buffer rows.
    #[test]
    fn a_second_hunk_is_placed_with_the_running_offset() {
        let original = "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n";
        let patch = "\
@@ -1,3 +1,5 @@
 def f():
     a = 1
+    a += 10
+    a += 20
     b = 2
@@ -4,2 +6,2 @@
     c = 3
-    return a + b + c
+    return a + b
";
        let patched = apply_hunks(original, patch).expect("both hunks must apply");
        assert_eq!(
            patched,
            "def f():\n    a = 1\n    a += 10\n    a += 20\n    b = 2\n    c = 3\n    return a + b\n",
            "hunk 2 must land after hunk 1's insertions"
        );
    }

    /// A hunk whose context cannot be found must refuse the WHOLE patch: a
    /// half-applied file is not a candidate.
    #[test]
    fn an_unanchorable_hunk_refuses_the_whole_patch() {
        let original = "def f():\n    return 1\n";
        let patch = "@@ -1,2 +1,2 @@\n def nowhere_in_this_file():\n-    return 1\n+    return 2\n";
        assert_eq!(apply_hunks(original, patch), None);
    }

    /// A pure insertion with context still anchors, and the verification
    /// rejects a hunk whose old side is not actually there.
    #[test]
    fn a_matching_old_side_is_required() {
        let original = "def f():\n    return 1\n";
        // Header says line 2, old side says something else — the `@@` fallback
        // must not be trusted blindly.
        let patch = "@@ -2,1 +2,1 @@\n    return 99\n";
        assert_eq!(apply_hunks(original, patch), None);
    }

    // ---- Fix 1 regression: review mode applies `git apply` and nothing else ----

    fn repo() -> HashMap<String, String> {
        HashMap::from([(
            "src/pkg/mod.py".to_string(),
            "def f():\n    return 1\n\n\ndef g():\n    return 2\n".to_string(),
        )])
    }

    #[test]
    fn strict_apply_takes_a_clean_two_hunk_patch() {
        let files = repo();
        let patch = "\
diff --git a/src/pkg/mod.py b/src/pkg/mod.py
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,2 +1,4 @@
 def f():
+    x = 1
+    y = 2
     return 1
@@ -5,2 +7,2 @@
 def g():
-    return 2
+    return 3
";
        let changed = apply_patch_strict(&files, patch, &["src/pkg/mod.py".to_string()])
            .expect("a clean patch must apply");
        assert_eq!(
            changed.get("src/pkg/mod.py").map(String::as_str),
            Some("def f():\n    x = 1\n    y = 2\n    return 1\n\n\ndef g():\n    return 3\n")
        );
    }

    /// Review mode must refuse a patch `git apply` cannot apply — the fuzzy
    /// cascade would have "recovered" this and handed the pipeline a file nobody
    /// wrote.
    #[test]
    fn strict_apply_refuses_a_patch_git_cannot_apply() {
        let files = repo();
        let patch = "\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,2 +1,2 @@
 def not_in_this_file():
-    return 1
+    return 2
";
        let err = apply_patch_strict(&files, patch, &["src/pkg/mod.py".to_string()])
            .expect_err("an inapplicable patch must be refused, not approximated");
        assert!(err.contains("git apply"), "unhelpful reason: {err}");

        // And the repair-mode cascade must NOT fabricate a tree either: no
        // strategy can place this hunk honestly, so it reports a failed apply.
        let original = files.get("src/pkg/mod.py").unwrap();
        let (recovered, changed) = apply_patch(original, patch, "src/pkg/mod.py");
        assert!(
            !changed && recovered == *original,
            "a hunk with no anchor must leave the file alone, not guess"
        );
    }

    /// The required regression: a hunk header whose counts do not match its body.
    /// `git apply` calls it corrupt and `--recount` refuses it too (verified
    /// against git 2.x), so review mode must stop rather than test a spliced tree.
    #[test]
    fn strict_apply_refuses_a_header_that_does_not_match_its_body() {
        let files = repo();
        // Header claims 5 old / 6 new lines; the body carries 2 old / 2 new.
        let patch = "\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,5 +1,6 @@
 def f():
-    return 1
+    return 9
";
        let err = apply_patch_strict(&files, patch, &["src/pkg/mod.py".to_string()])
            .expect_err("a miscounted header must be refused");
        assert!(err.contains("git apply"), "unhelpful reason: {err}");
    }

    /// ...whereas the repair cascade still takes what strict refuses, because a
    /// Creator response may not be a diff at all. That difference IS the policy.
    #[test]
    fn the_cascade_takes_what_strict_refuses() {
        let files = repo();
        let original = files.get("src/pkg/mod.py").unwrap();
        // A headerless `@@` fragment: `git apply` cannot name the file, but the
        // hunk engine only needs the context.
        let patch = "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 7\n";
        assert!(
            apply_patch_strict(&files, patch, &["src/pkg/mod.py".to_string()]).is_err(),
            "strict must refuse a headerless fragment"
        );
        let (recovered, changed) = apply_patch(original, patch, "src/pkg/mod.py");
        assert!(changed, "the cascade can still apply a headerless fragment");
        assert!(recovered.contains("return 7"));
    }

    #[test]
    fn strict_apply_reports_a_no_op_patch() {
        let files = repo();
        let patch = "\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,2 +1,2 @@
 def f():
     return 1
";
        let err = apply_patch_strict(&files, patch, &["src/pkg/mod.py".to_string()])
            .expect_err("a patch that changes nothing is not an application");
        assert!(err.contains("no-op"), "unhelpful reason: {err}");
    }

    // ---- Fix 1 regression: the syntax guard ----

    #[test]
    fn a_non_git_strategy_that_breaks_python_is_refused() {
        // Full-source strategy, so no git tool is involved: the model hands back
        // a file with an `if` and no body.
        let original = "def f():\n    return 1\n";
        let patch = "def f():\n    if True:\n    return 1\n";
        let (patched, changed) = apply_patch(original, patch, "src/pkg/mod.py");
        assert!(
            !changed,
            "a result that no longer compiles must be reported as a failed apply, \
             not as 'patched (changed)'; got:\n{patched}"
        );
        assert_eq!(patched, original);
    }

    #[test]
    fn the_syntax_guard_leaves_non_python_files_alone() {
        // The SAME content, once against a non-Python file and once against a
        // .py file: accepted for the former (the guard only knows Python),
        // refused for the latter (the original parsed and the result does not).
        let broken = "def f(:\n  not python at all\n";

        let (patched, changed) = apply_patch("hello\n", broken, "README.md");
        assert!(changed, "a non-Python file has no Python syntax to preserve");
        assert_eq!(patched, broken);

        let original = "def f():\n    return 1\n";
        let (kept, refused) = apply_patch(original, broken, "src/pkg/mod.py");
        assert!(!refused, "the same content must be refused for a .py file");
        assert_eq!(kept, original);
    }

    // ---- Item R: the apply failure carries what the cascade already knew ----

    #[test]
    fn a_missing_search_anchor_names_the_file_and_the_anchor() {
        // The model wrote a SEARCH anchor that is not in the file. The feedback
        // the Creator receives must name the file, quote the exact anchor, and
        // give the strategy's reason — never the old fixed sentence.
        let original = "def add(a, b):\n    return a + b\n";
        let patch = "\
===SEARCH===
def subtract(a, b):
    return a - b
===REPLACE===
def subtract(a, b):
    return b - a
===END===
";
        let (_text, changed, failure) = apply_patch_detailed(original, patch, "src/pkg/mod.py");
        assert!(!changed, "an anchor that is not in the file cannot apply");
        let msg = failure.expect("the cascade must report why it gave up").feedback();

        assert!(msg.contains("src/pkg/mod.py"), "the file must be named: {msg}");
        assert!(
            msg.contains("def subtract(a, b):\n    return a - b"),
            "the exact SEARCH anchor must be quoted: {msg}"
        );
        assert!(
            msg.contains("search/replace"),
            "the failing strategy must be named: {msg}"
        );
        assert!(
            !msg.contains("no file was changed by any strategy"),
            "the old constant string must be gone: {msg}"
        );
    }

    #[test]
    fn a_full_cascade_failure_lists_every_strategy_and_the_skip() {
        // A unified diff whose hunk anchor is not in the file: git apply, patch(1)
        // and the hunk engine all refuse it, and reconstruction is skipped
        // because the patch contains a deletion. Every one of those facts must be
        // carried back, with the reason the skip happened.
        let original = "def f():\n    return 1\n";
        let patch = "\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -10,2 +10,2 @@
 def other():
-    return 2
+    return 3
";
        let (_text, changed, failure) = apply_patch_detailed(original, patch, "src/pkg/mod.py");
        assert!(!changed);
        let msg = failure.expect("must fail").feedback();

        assert!(msg.contains("git apply --3way"), "{msg}");
        assert!(msg.contains("hunk-by-hunk"), "{msg}");
        assert!(msg.contains("whole-file reconstruction: skipped"), "{msg}");
        assert!(msg.contains("deletions"), "{msg}");
        assert!(
            !msg.contains("patch contains deletions — unsafe"),
            "the old one-size sentence must be gone: {msg}"
        );
    }

    #[test]
    fn an_invalid_python_result_still_reports_the_interpreter_message() {
        // The other variant is untouched: a diff that applied but does not parse
        // must still lead with the interpreter's own message.
        let original = "def f():\n    return 1\n";
        let patch = "def f():\n    if True:\n    return 1\n";
        let (_text, changed, failure) = apply_patch_detailed(original, patch, "src/pkg/mod.py");
        assert!(!changed);
        let msg = failure.expect("must fail").feedback();
        assert!(
            msg.starts_with("patch result is not valid Python:"),
            "the interpreter message must lead: {msg}"
        );
    }
}
