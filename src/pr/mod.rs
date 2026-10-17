// ============================================================================
// PR MODULE — ship a Vanta-verified patch as a GitHub PR (Spec M3 BUILD)
// ============================================================================
//
// [VANTA-WHY] M3's definition of done: "Given a real Python repo and a bug
// description, Vanta produces a PR containing a patch plus Attacker-generated
// tests that pass baseline validation, with the whole loop running unattended."
//
// This module takes the Gatekeeper-approved patched files + the valid
// Attacker tests, drops them into the target repo on a fresh branch, commits,
// and opens a PR via the `gh` CLI. It does NOT reinvent git: it shells out to
// `git` and `gh` so auth is whatever the user already has configured.
//
// [PITFALL] We never commit to the user's current branch. We always create a
// new branch named `vanta/fix-<short-id>` so a failed run doesn't pollute
// `main`. The patched files are written directly into the working tree; the
// accepted tests go into `tests/vanta_<session>.py`.
// ============================================================================

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::core::TestEvidence;
use crate::session::SessionId;

/// Result of opening a PR.
#[derive(Debug, Clone)]
pub struct PrResult {
    pub branch: String,
    pub pr_url: Option<String>,
    pub commit_sha: Option<String>,
}

/// Open a PR containing the patched files + accepted tests.
///
/// - `repo_path`: the target repository working tree
/// - `patched_files`: map of relative path -> patched content
/// - `accepted_tests`: the Attacker-generated tests that passed BATE
/// - `evidence`: per-test evidence for the PR body
/// - `session_id`: for branch name uniqueness + test file naming
/// - `issue_text`: the original bug description (goes in the PR body)
pub fn open_pr(
    repo_path: &Path,
    patched_files: &std::collections::HashMap<String, String>,
    accepted_tests: &str,
    evidence: &[TestEvidence],
    session_id: &SessionId,
    issue_text: &str,
) -> Result<PrResult, Box<dyn std::error::Error + Send + Sync>> {
    // ---- 1. Create a fresh branch ----
    let short_id = &session_id.as_str()[..session_id.as_str().len().min(8)];
    let branch = format!("vanta/fix-{short_id}");
    run_git(repo_path, &["checkout", "-b", &branch])?;
    tracing::info!(branch = %branch, "PR: created branch");

    // ---- 2. Write patched files into the working tree ----
    for (rel_path, content) in patched_files {
        let abs = repo_path.join(rel_path);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&abs, content)?;
        run_git(repo_path, &["add", rel_path])?;
    }
    tracing::info!(count = patched_files.len(), "PR: wrote patched files");

    // ---- 3. Write accepted tests ----
    let tests_dir = repo_path.join("tests");
    fs::create_dir_all(&tests_dir)?;
    let tests_rel = format!("tests/vanta_{}.py", short_id);
    let tests_abs = repo_path.join(&tests_rel);
    fs::write(&tests_abs, accepted_tests)?;
    run_git(repo_path, &["add", &tests_rel])?;

    // ---- 4. Commit ----
    let msg = format!(
        "Vanta: auto-generated fix (session {short_id})\n\n\
         Bug:\n{}\n\n\
         Verification: {} test(s) passed baseline + patched validation.",
        issue_text.lines().next().unwrap_or("(see issue)"),
        evidence.iter().filter(|e| {
            e.outcome == crate::core::VerificationOutcome::VerifiedWithinBounds
        }).count()
    );
    run_git(repo_path, &["commit", "-m", &msg])?;
    let commit_sha = run_git(repo_path, &["rev-parse", "HEAD"]).ok()
        .map(|s| s.trim().to_string());

    // ---- 5. Open PR via `gh` ----
    let body = build_pr_body(issue_text, evidence);
    let title = format!("Vanta fix: {}", issue_text.lines().next().unwrap_or("auto-generated fix"));
    let pr_url = match run_gh(repo_path, &["pr", "create", "--title", &title, "--body", &body]) {
        Ok(url) => Some(url.trim().to_string()),
        Err(e) => {
            tracing::warn!(error = %e, "PR: gh pr create failed (push branch manually)");
            // Try pushing the branch at least
            let _ = run_git(repo_path, &["push", "-u", "origin", &branch]);
            None
        }
    };

    Ok(PrResult { branch, pr_url, commit_sha })
}

/// Build a human-readable PR body from the issue + evidence.
fn build_pr_body(issue_text: &str, evidence: &[TestEvidence]) -> String {
    let passed: Vec<&TestEvidence> = evidence
        .iter()
        .filter(|e| e.outcome == crate::core::VerificationOutcome::VerifiedWithinBounds)
        .collect();
    let failed: Vec<&TestEvidence> = evidence
        .iter()
        .filter(|e| e.outcome != crate::core::VerificationOutcome::VerifiedWithinBounds)
        .collect();

    let mut body = String::new();
    body.push_str("## Vanta Adversarial Verification Report\n\n");
    body.push_str("This patch was generated and verified by Vanta's adversarial loop.\n\n");
    body.push_str("### Issue\n```\n");
    body.push_str(issue_text);
    body.push_str("\n```\n\n");
    body.push_str(&format!("### Tests passed: {}\n", passed.len()));
    for e in &passed {
        body.push_str(&format!("- `{}` (intent={:?})\n", e.test_name, e.intent));
    }
    if !failed.is_empty() {
        body.push_str(&format!("\n### Tests failed: {}\n", failed.len()));
        for e in &failed {
            body.push_str(&format!(
                "- `{}` (intent={:?}, failure_class={:?})\n",
                e.test_name, e.intent, e.failure_class
            ));
        }
    }
    body.push_str("\n---\n*Generated by Vanta (M3 V1).*\n");
    body
}

// ---- git / gh wrappers -----------------------------------------------------

fn run_git(repo: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ).into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_gh(repo: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let out = Command::new("gh")
        .args(args)
        .current_dir(repo)
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "gh {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ).into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
