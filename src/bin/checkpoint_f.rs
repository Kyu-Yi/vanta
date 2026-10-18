// ============================================================================

#[cfg(test)]
mod tests {
    use super::resolve_base_sha;

    #[test]
    fn rejects_short_explicit_base_sha() {
        let err = resolve_base_sha(Some("b442dc8".into()), "80bedc327ef0a289d668c78fdf071bd0a7694379")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("full 40-character commit SHA"),
            "short SHAs must fail loudly, got: {err}"
        );
    }

    #[test]
    fn accepts_full_explicit_base_sha() {
        let sha = "b442dc8f21a44c182f00d3dcb629ec178402c769";
        assert_eq!(
            resolve_base_sha(Some(sha.into()), "80bedc327ef0a289d668c78fdf071bd0a7694379")
                .unwrap(),
            sha
        );
    }
}
// CHECKPOINT F — Open a GitHub pull request from Rust, unattended (octocrab)
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY
//
//   # | Rust / API concept             → Where Vanta reuses it later
//  ---+--------------------------------+----------------------------------------
//   1 | octocrab client + PAT auth     → M3's PR step. Auth is debugged HERE, on
//     |                                →   a throwaway repo, not inside the loop.
//  ---+--------------------------------+----------------------------------------
//   2 | Git Data API: blob → tree →    → Commits a file through the REST API. No
//     |   commit → ref                 →   local clone, no `git`/`gh` shell-out.
//  ---+--------------------------------+----------------------------------------
//   3 | Branch from a RECORDED sha     → The branch starts where Vanta tested, not
//     |                                →   at whatever the default branch points
//     |                                →   to when the PR is finally opened.
//  ---+--------------------------------+----------------------------------------
//   4 | Draft PR by default            → `--ready` opens it for review instead.
//  ---+--------------------------------+----------------------------------------
//   5 | Secret hygiene                 → The token never reaches the body, the
//     |                                →   title, or any printed line.
// ============================================================================

use std::error::Error;

use octocrab::Octocrab;
use octocrab::models::git::CreateTreeEntry;
use octocrab::params::repos::Reference;

type BoxError = Box<dyn Error + Send + Sync>;

/// Git file mode for a non-executable regular file.
const MODE_FILE: &str = "100644";
/// Git object type tag for a file's contents.
const TYPE_BLOB: &str = "blob";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHECKPOINT F — Create a branch, commit, and open a PR        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ---- 0. Inputs (env only; the token never leaves this process) ----
    // [VANTA-WHY] The PAT is read once and only ever handed to the client
    // builder. It is never formatted into the body, the title, or a log line —
    // that is the whole point of doing this checkpoint on its own.
    let token = std::env::var("VANTA_GITHUB_TOKEN").map_err(|_| {
        "VANTA_GITHUB_TOKEN is not set (needs contents + pull-requests write)"
    })?;
    let repo_slug = std::env::var("VANTA_PR_REPO")
        .map_err(|_| "VANTA_PR_REPO is not set (expected \"owner/repo\")")?;
    let (owner, repo) = repo_slug
        .split_once('/')
        .ok_or("VANTA_PR_REPO must be \"owner/repo\"")?;
    let ready = std::env::args().any(|arg| arg == "--ready");

    let octocrab = Octocrab::builder().personal_token(token).build()?;

    // ---- 1. Where does the default branch point RIGHT NOW? ----
    let repository = octocrab.repos(owner, repo).get().await?;
    let default_branch = repository
        .default_branch
        .clone()
        .ok_or("repository has no default branch")?;
    let default_tip = octocrab.commits(owner, repo).get(&default_branch).await?.sha;
    println!("▌ Default branch: {default_branch} @ {}", short(&default_tip));

    // ---- 2. The base is the commit Vanta TESTED, not today's tip ----
    // [VANTA-WHY] A PR opened minutes after a run may see a default branch that
    // has already moved. The branch must still start at the exact commit the
    // pipeline verified; the divergence is reported, never silently absorbed.
    let base_sha = resolve_base_sha(std::env::var("VANTA_PR_BASE_SHA").ok(), &default_tip)?;
    let base_moved = base_sha != default_tip;
    if base_moved {
        println!(
            "⚠ {default_branch} moved: tested at {} but now at {} — branching from the tested commit",
            short(&base_sha),
            short(&default_tip)
        );
    } else {
        println!("▌ Base commit: {} (default branch unchanged)", short(&base_sha));
    }

    // ---- 3. Resolve the base commit's tree so the new tree is built on it ----
    let base_commit = octocrab
        .git(owner, repo)
        .get_commit(&base_sha)
        .await
        .map_err(|e| {
            if std::env::var("VANTA_PR_BASE_SHA").is_ok() {
                format!(
                    "VANTA_PR_BASE_SHA={} does not exist on GitHub repo {owner}/{repo}: {e}",
                    base_sha
                )
            } else {
                format!(
                    "failed to resolve default-branch tip {} on GitHub repo {owner}/{repo}: {e}",
                    base_sha
                )
            }
        })?;
    let base_tree = base_commit.tree.sha.clone();

    // ---- 4. blob → tree → commit (a file written through the API) ----
    let file_path = std::env::var("VANTA_PR_FILE")
        .unwrap_or_else(|_| "vanta_checkpoint_f.txt".to_string());
    let content = std::env::var("VANTA_PR_CONTENT").unwrap_or_else(|_| {
        "Vanta build checkpoint F: branch + commit + pull request.\n".to_string()
    });
    let blob = octocrab
        .git(owner, repo)
        .create_blob(content)
        .encoding("utf-8")
        .send()
        .await?;
    println!("▌ Created blob {}", short(&blob.sha));

    let tree = octocrab
        .git(owner, repo)
        .create_tree(vec![CreateTreeEntry::new(file_path.clone(), MODE_FILE, TYPE_BLOB)
            .with_sha(blob.sha)])
        .base_tree(base_tree)
        .send()
        .await?;
    println!("▌ Created tree {}", short(&tree.sha));

    let branch = std::env::var("VANTA_PR_BRANCH").unwrap_or_else(|_| {
        format!("vanta/checkpoint-f-{}", short(&uuid::Uuid::new_v4().to_string()))
    });
    let commit = octocrab
        .git(owner, repo)
        .create_commit(format!("Vanta checkpoint F: add {file_path}"), tree.sha)
        .parents(vec![base_sha.clone()])
        .send()
        .await?;
    println!("▌ Created commit {}", short(&commit.sha));

    // ---- 5. Point a NEW ref at the commit (this is the branch) ----
    octocrab
        .git(owner, repo)
        .create_ref(&Reference::Branch(branch.clone()), commit.sha.clone())
        .await?;
    println!("▌ Created branch {branch}");

    // ---- 6. Open the PR, draft unless `--ready` ----
    let (vanta_commit, vanta_dirty) = vanta::session::vanta_revision();
    let session =
        std::env::var("VANTA_PR_SESSION").unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
    let decision =
        std::env::var("VANTA_PR_DECISION").unwrap_or_else(|_| "n/a (checkpoint F)".to_string());
    let tests = std::env::var("VANTA_PR_TESTS")
        .unwrap_or_else(|_| "(checkpoint F — no pipeline tests)".to_string());

    let body = build_body(
        &decision,
        &tests,
        &session,
        &vanta_commit,
        vanta_dirty,
        &base_sha,
        &default_branch,
        base_moved.then_some(default_tip.as_str()),
        &file_path,
    );
    let title = format!("Vanta checkpoint F: {file_path}");

    let pr = octocrab
        .pulls(owner, repo)
        .create(title, branch.clone(), default_branch.clone())
        .body(body)
        .draft(!ready)
        .send()
        .await?;

    let url = pr
        .html_url
        .as_ref()
        .map(|u| u.to_string())
        .unwrap_or_default();
    println!("\n▌ PR #{} — {url}", pr.number);
    println!(
        "▌ State: {}",
        if ready { "ready for review" } else { "DRAFT" }
    );
    println!(
        "▌ Head: {branch} → base: {default_branch} (from {})",
        short(&base_sha)
    );
    println!("\n✔ Checkpoint F complete: branch, commit and PR created via the GitHub API.");
    Ok(())
}

/// The PR body carries only run-level facts — never the credential.
#[allow(clippy::too_many_arguments)]
fn build_body(
    decision: &str,
    tests: &str,
    session: &str,
    vanta_commit: &str,
    vanta_dirty: bool,
    base_sha: &str,
    base_branch: &str,
    moved_to: Option<&str>,
    file_path: &str,
) -> String {
    let mut body = String::new();
    body.push_str("## Vanta pull request (build checkpoint F)\n\n");
    body.push_str(
        "Opened by Vanta's GitHub integration smoke test: branch → commit → PR via the \
         GitHub REST API. No local clone, no `git`/`gh` shell-out.\n\n",
    );
    body.push_str("### Decision\n");
    body.push_str(decision);
    body.push_str("\n\n### Tests run\n");
    body.push_str(tests);
    body.push_str("\n\n### Provenance\n");
    body.push_str(&format!("- session: `{session}`\n"));
    body.push_str(&format!(
        "- vanta_commit: `{vanta_commit}` (dirty={vanta_dirty})\n"
    ));
    body.push_str(&format!(
        "- prompt_version: `{}`\n",
        vanta::pipeline::prompt_versions()
    ));
    body.push_str(&format!("- changed file: `{file_path}`\n"));
    body.push_str("\n### Base\n");
    body.push_str(&format!(
        "Branch created from the tested commit `{base_sha}`.\n"
    ));
    if let Some(tip) = moved_to {
        body.push_str(&format!(
            "\n> ⚠ The default branch `{base_branch}` has moved since the tested commit \
             (now at `{tip}`). This branch was created from the exact commit Vanta tested, \
             so the diff is against tested code, not today's `{base_branch}` tip.\n"
        ));
    }
    body.push_str("\n---\n");
    body.push_str("*No credential is included in this body or its metadata.*\n");
    body
}

fn resolve_base_sha(explicit: Option<String>, default_tip: &str) -> Result<String, BoxError> {
    match explicit {
        None => Ok(default_tip.to_string()),
        Some(raw) => {
            let sha = raw.trim();
            if sha.len() != 40 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!(
                    "VANTA_PR_BASE_SHA must be a full 40-character commit SHA; got {sha:?}"
                )
                .into());
            }
            Ok(sha.to_string())
        }
    }
}

/// First 7 chars of a sha (or the whole string if shorter).
fn short(sha: &str) -> &str {
    &sha[..sha.len().min(7)]
}

// ============================================================================
// RECIPES — copy-paste templates for the M3 PR step
// ============================================================================
//
// ----------------------------------------------------------------------------
// RECIPE F1 — "Authenticate an octocrab client from a PAT"
//
//   let token = std::env::var("VANTA_GITHUB_TOKEN")?;   // never log this
//   let crab = octocrab::Octocrab::builder().personal_token(token).build()?;
//
//   → USE IN: M3 before opening a PR. Scopes: contents:write + pull-requests:write.
//
// ----------------------------------------------------------------------------
// RECIPE F2 — "Create a branch from a recorded base sha"
//
//   octocrab.git(owner, repo)
//       .create_ref(&Reference::Branch("vanta/fix-abc".into()), base_sha)
//       .await?;
//
//   → USE IN: M3 the branch must start at the commit the pipeline TESTED,
//     not at the default branch tip at PR time.
//
// ----------------------------------------------------------------------------
// RECIPE F3 — "Commit one file through the API (no local clone)"
//
//   let base = crab.git(o, r).get_commit(&base_sha).await?;        // .tree.sha
//   let blob = crab.git(o, r).create_blob(content).encoding("utf-8").send().await?;
//   let tree = crab.git(o, r)
//       .create_tree(vec![CreateTreeEntry::new(path, "100644", "blob").with_sha(blob.sha)])
//       .base_tree(base.tree.sha).send().await?;
//   let commit = crab.git(o, r)
//       .create_commit(msg, tree.sha).parents(vec![base_sha]).send().await?;
//
//   → USE IN: M3 applying the Gatekeeper-approved patch + accepted tests.
//
// ----------------------------------------------------------------------------
// RECIPE F4 — "Open the PR (draft by default)"
//
//   let pr = crab.pulls(o, r).create(title, branch, base_branch)
//       .body(body).draft(true).send().await?;
//   println!("{}", pr.html_url.unwrap());
//
//   → USE IN: M3 only after GatekeeperVerdict::Approved; draft unless told otherwise.
//
// ----------------------------------------------------------------------------
// RECIPE F5 — "Detect a moved base branch"
//
//   let tip = crab.commits(o, r).get(&default_branch).await?.sha;
//   let base_moved = base_sha != tip;
//
//   → USE IN: M3 to say so in the PR body instead of silently diffing against a
//     default branch that advanced after the run.
//
// ============================================================================
