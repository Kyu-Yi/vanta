// ============================================================================
// CREATOR AGENT — issue + source files → patch (Spec §4 Creator agent)
// ============================================================================
//
// Spec §4: "Creator agent: Reads issue, source, and history; writes a minimal patch."
//
// M1 scope (per Master Plan M1 BUILD):
//   - CreatorRequest: issue_text, source_files, optional prior_feedback
//   - CreatorResponse: patch_text, explanation, files_touched, confidence
//   - Verify-First prompting in the system prompt from day one
//
// Verify-First (Spec §8, Liu & Meng 2026): instruct Creator to reason about
// verification BEFORE writing the patch. This is the intervention that the
// paper found drives EIR from 2% → 0% in reasoning domains. We evaluate it as
// a hypothesis in M4, but prompt for it from M1 so the data exists.
// ============================================================================

use crate::config::LlmConfig;
use crate::llm::{build_reqwest_client, extract_json_candidate, structured_call_with_retry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use tracing::{debug, info, trace, warn};

// ============================================================================
// CREATOR_PROMPT_VERSION — the run-identity tag for the Creator's prompt
// ============================================================================
//
// [VANTA-WHY] `session.start` records a `prompt_version` that names EVERY
// agent prompt in the run, so a later change to the Creator's Verify-First
// wording is visible in the log instead of looking like "no change". Bump this
// whenever the Creator system/user prompt text changes.
pub const CREATOR_PROMPT_VERSION: &str = "1";

// ============================================================================
// CreatorRequest — input to the Creator agent
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorRequest {
    pub issue_text: String,
    pub source_files: Vec<SourceFile>,
    pub prior_feedback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: String,
    pub contents: String,
}

// ============================================================================
// CreatorResponse — typed output from the Creator agent
// ============================================================================
//
// [VANTA-WHY] This is the SAME shape as the M0 spike CreatorResponse, now
// PROMOTED to a shared module so M1 Creator agent, M2 pipeline, and M0 spike
// all use ONE type (no copy-paste drift).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreatorResponse {
    pub patch_text: String,
    pub explanation: String,
    pub files_touched: Vec<String>,
    pub confidence: f64,
}

// ============================================================================
// creator_generate_patch() — call the LLM, get a typed CreatorResponse
// ============================================================================
//
// [VANTA-WHY] Uses structured_call_with_retry (3 attempts) + extract_json_candidate
// (4-stage cleaning pipeline) — the exact same robust pipeline from M0.
pub async fn creator_generate_patch(
    cfg: &LlmConfig,
    request: &CreatorRequest,
) -> Result<CreatorResponse, Box<dyn Error + Send + Sync>> {
    info!(
        issue_len = request.issue_text.len(),
        source_files = request.source_files.len(),
        "creator start"
    );

    let client = build_reqwest_client(cfg)?;

    // ---- Verify-First system prompt (Spec §8, Liu & Meng 2026) ----
    // [VANTA-WHY] "reason about verification BEFORE writing the patch".
    // The paper found this intervention suppresses error introduction (EIR).
    //
    // [VANTA-WHY] We ask for FULL SOURCE or SEARCH/REPLACE blocks, NOT unified diffs.
    // LLMs cannot reliably count lines for `@@ -a,b +c,d @@` hunk headers, and
    // a broken unified diff silently fails to apply (whole-file reconstruction
    // cannot express deletions). Full source eliminates counting entirely;
    // search/replace blocks use exact anchor text that requires no arithmetic.
    //
    // [VANTA-WHY] MULTI-FILE SUPPORT: For patches touching >1 file, separate
    // each file's content with `===FILE: path/to/file.py===` markers.
    let system_prompt = r#"You are Vanta Creator. Before writing any patch, THINK BRIEFLY (under 200 words) about how the change will be verified: what tests would pass/fail, what properties must hold. Then output the fix.

IMPORTANT: Keep your reasoning SHORT. The bulk of your output budget must go to the JSON answer in content.

YOU MUST output valid JSON. NO EXCEPTIONS.
  1. First character of content MUST be '{'
  2. Last character MUST be the matching '}'
  3. No prose, no markdown fences
  4. Do not stop until the full '}' is written

How to provide the fix in patch_text — CHOOSE ONE:

OPTION A (preferred for files under ~200 lines): Output the COMPLETE patched file content.
  Write the entire file as it should look after the fix, from the first line to the last.
  Do NOT use diff markers (---, +++, @@, +, -). Just the plain source code.

  If the patch touches MULTIPLE files, separate each with a marker line:
    ===FILE: path/to/file.py===
    <complete file content>
    ===FILE: path/to/other.py===
    <complete file content>

OPTION B (for larger files where full source is wasteful): Use SEARCH/REPLACE blocks.
  For a single file, just use the blocks.
  For multiple files, prefix each block group with ===FILE: path===:
    ===FILE: path/to/file.py===
    ===SEARCH===
    <exact lines from the original file, including indentation>
    ===REPLACE===
    <the replacement lines>
    ===END===

  The SEARCH text must match the original EXACTLY (whitespace included) so it can be found.

Required JSON schema:
{
  "patch_text":    string — the fix as full source (Option A) or search/replace blocks (Option B),
  "explanation":   string — 1 sentence,
  "files_touched": array of strings,
  "confidence":    number 0.0 to 1.0
}"#;

    // Build user prompt: issue + source files
    let mut user_parts: Vec<String> = Vec::new();
    user_parts.push(format!("ISSUE:\n{}\n", request.issue_text));
    for sf in &request.source_files {
        user_parts.push(format!("FILE: {}\n```\n{}\n```\n", sf.path, sf.contents));
    }
    if let Some(fb) = &request.prior_feedback {
        user_parts.push(format!("PRIOR FEEDBACK (from previous attempt, address this):\n{}\n", fb));
    }
    let user_prompt = user_parts.join("\n");

    debug!(system_len = system_prompt.len(), user_len = user_prompt.len(), "creator prompts built");

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Last response was not valid JSON. Write ONLY a JSON object beginning with '{{' and ending with '}}'. Keys: patch_text (full source or search/replace blocks), explanation, files_touched, confidence.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 32768,
        "reasoning_effort": "low",
    });

    let parse_closure = |raw: &str| -> Result<CreatorResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        trace!(json = %cleaned.chars().take(300).collect::<String>(), "creator extracted json");
        serde_json::from_str::<CreatorResponse>(cleaned).map_err(|e| {
            format!("CREATOR PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: CreatorResponse = structured_call_with_retry(
        &client, cfg, 3, build_messages, extras, parse_closure
    ).await?;

    // Sanity checks
    if !(0.0..=1.0).contains(&typed.confidence) {
        warn!(confidence = typed.confidence, "creator confidence out of [0,1]");
    }
    if typed.files_touched.is_empty() {
        warn!("creator files_touched is empty");
    }

    info!(
        patch_len = typed.patch_text.len(),
        files_touched = ?typed.files_touched,
        confidence = typed.confidence,
        "creator done"
    );

    Ok(typed)
}

// ============================================================================
// split_multi_file_patch — split patch_text into per-file patches
// ============================================================================
//
// [VANTA-WHY] For multi-file repos, the Creator emits multiple files separated
// by `===FILE: path/to/file.py===` markers. This function splits the combined
// patch_text into a map of (relative_path -> patch_content_for_that_file).
//
// If no ===FILE: markers are present, the whole patch_text is treated as a
// single-file patch and returned under the first entry in files_touched (or
// "source.txt" if files_touched is empty).
// ============================================================================
pub fn split_multi_file_patch(
    patch_text: &str,
    files_touched: &[String],
) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;

    let mut result = HashMap::new();

    // Check for multi-file markers
    if patch_text.contains("===FILE:") {
        let mut current_path: Option<String> = None;
        let mut current_content = String::new();

        for line in patch_text.lines() {
            if let Some(rest) = line.strip_prefix("===FILE:") {
                // Save previous file
                if let Some(p) = current_path.take() {
                    result.insert(p, current_content.trim_end().to_string());
                    current_content = String::new();
                }
                current_path = Some(rest.trim_end_matches("===").trim().to_string());
            } else if current_path.is_some() {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }
        // Save last file
        if let Some(p) = current_path {
            result.insert(p, current_content.trim_end().to_string());
        }
    } else {
        // Single file — use first files_touched entry or default
        let path = files_touched.first().cloned().unwrap_or_else(|| "source.txt".to_string());
        result.insert(path, patch_text.to_string());
    }

    result
}

/// Parse a standard git diff (with `diff --git a/x b/y`, `--- a/x`, `+++ b/y`
/// headers) into a map of file path → that file's hunks.
///
/// [VANTA-WHY] Used by `vanta_m3 --patch PATH` when verifying an EXISTING
/// developer patch (e.g. `git diff > my.patch`). The result is passed to
/// `apply_patch()` per file, just like the Creator's multi-file output.
pub fn split_patch_into_files(patch_text: &str) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;
    let mut result = HashMap::new();

    let mut current_path: Option<String> = None;
    let mut current_hunks = String::new();
    // Whether we're inside a file's diff section (after `+++ b/path`).
    let mut in_hunks = false;

    for line in patch_text.lines() {
        if line.starts_with("diff --git") {
            // Save previous file.
            if let Some(p) = current_path.take() {
                result.insert(p, current_hunks.trim_end().to_string());
                current_hunks = String::new();
            }
            in_hunks = false;
            // Extract the path from `diff --git a/src/foo.py b/src/foo.py`
            if let Some(path) = extract_path_from_diff_header(line) {
                current_path = Some(path);
            }
        } else if line.starts_with("+++ b/") {
            // `+++ b/src/foo.py` confirms the destination path — use it.
            let path = line.trim_start_matches("+++ b/").trim().to_string();
            // A path of "/dev/null" means the file was deleted; skip.
            if path != "/dev/null" {
                current_path = Some(path);
            }
            in_hunks = true;
        } else if line.starts_with("--- ") || line.starts_with("+++ ") {
            // Other --- / +++ headers (index, new file mode, etc.) — skip.
        } else if in_hunks && current_path.is_some() {
            current_hunks.push_str(line);
            current_hunks.push('\n');
        }
    }
    // Save last file.
    if let Some(p) = current_path.take() {
        result.insert(p, current_hunks.trim_end().to_string());
    }

    result
}

/// Extract the destination path from a `diff --git a/src/foo.py b/src/foo.py` line.
fn extract_path_from_diff_header(line: &str) -> Option<String> {
    // Format: `diff --git a/<path> b/<path>`
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 {
        // parts[2] = "a/path", parts[3] = "b/path"
        let b_path = parts[3];
        if let Some(rest) = b_path.strip_prefix("b/") {
            return Some(rest.to_string());
        }
        // Fallback: parts[2] without "a/"
        if let Some(rest) = parts[2].strip_prefix("a/") {
            return Some(rest.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_single_file() {
        let patch = "def add(a, b):\n    return a + b\n";
        let files = vec!["cart.py".to_string()];
        let result = split_multi_file_patch(patch, &files);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("cart.py"));
    }

    #[test]
    fn test_split_multi_file() {
        let patch = "===FILE: a.py===\ndef a(): pass\n===FILE: b.py===\ndef b(): pass\n";
        let files = vec!["a.py".to_string(), "b.py".to_string()];
        let result = split_multi_file_patch(patch, &files);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("a.py"));
        assert!(result.contains_key("b.py"));
        assert!(result["a.py"].contains("def a()"));
        assert!(result["b.py"].contains("def b()"));
    }
}
