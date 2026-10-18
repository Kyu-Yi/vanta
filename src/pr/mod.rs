// ============================================================================
// PR MODULE — ship a Vanta-verified patch as a GitHub PR (Spec M3 BUILD)
// ============================================================================
//
// [VANTA-WHY] The wired M3 PR path must use the SAME tested commit Vanta
// recorded at `session.start`, not whatever the default branch points at when
// the PR is opened later. This module therefore reads run provenance from the
// durable session log and opens the PR through the GitHub API with octocrab.
//
// [PITFALL] This module NEVER falls back to `gh` login state, `GITHUB_TOKEN`,
// or local git pushes. The only credential it accepts is `VANTA_GITHUB_TOKEN`,
// and the only branch it creates starts from the tested commit recorded in
// `session.start`.
// ============================================================================

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use octocrab::Octocrab;
use serde_json::{Value, json};

use crate::core::TestEvidence;
use crate::session::{SessionId, SessionRecord};

/// Result of opening a PR.
#[derive(Debug, Clone)]
pub struct PrResult {
    pub branch: String,
    pub pr_url: Option<String>,
    pub commit_sha: Option<String>,
}

#[derive(Debug, Clone)]
struct SessionStartMeta {
    tested_commit: String,
    prompt_version: String,
    vanta_commit: String,
    vanta_dirty: bool,
}

/// Fail early if the session-recorded tested commit cannot be used to open a PR.
pub async fn preflight_open_pr(
    repo_path: &Path,
    session_log_path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let token = std::env::var("VANTA_GITHUB_TOKEN").map_err(|_| {
        "VANTA_GITHUB_TOKEN is not set (this path never falls back to gh login)"
    })?;
    let repo_slug = repo_slug(repo_path)?;
    let (owner, repo) = repo_slug
        .split_once('/')
        .ok_or("repository slug must be owner/repo")?;
    let meta = read_session_start_meta(session_log_path)?;
    let octocrab = github_client(&token)?;

    let _: Value = octocrab
        .get(
            format!("/repos/{owner}/{repo}/git/commits/{}", meta.tested_commit),
            None::<&()>,
        )
        .await
        .map_err(|err| {
            format!(
                "tested_commit={} from session.start does not exist on GitHub repo {owner}/{repo}: {err}",
                meta.tested_commit
            )
        })?;
    Ok(())
}

/// Open a PR containing the patched files + accepted tests.
pub async fn open_pr(
    repo_path: &Path,
    patched_files: &HashMap<String, String>,
    accepted_tests: &str,
    evidence: &[TestEvidence],
    session_log_path: &Path,
    session_id: &SessionId,
    _issue_text: &str,
) -> Result<PrResult, Box<dyn std::error::Error + Send + Sync>> {
    let token = std::env::var("VANTA_GITHUB_TOKEN").map_err(|_| {
        "VANTA_GITHUB_TOKEN is not set (this path never falls back to gh login)"
    })?;
    let repo_slug = repo_slug(repo_path)?;
    let (owner, repo) = repo_slug
        .split_once('/')
        .ok_or("repository slug must be owner/repo")?;
    let meta = read_session_start_meta(session_log_path)?;
    let octocrab = github_client(&token)?;

    let repo_json: Value = octocrab
        .get(format!("/repos/{owner}/{repo}"), None::<&()>)
        .await?;
    let default_branch = json_str(&repo_json, &["default_branch"])?.to_string();

    let base_commit: Value = octocrab
        .get(
            format!("/repos/{owner}/{repo}/git/commits/{}", meta.tested_commit),
            None::<&()>,
        )
        .await
        .map_err(|err| {
            format!(
                "tested_commit={} from session.start does not exist on GitHub repo {owner}/{repo}: {err}",
                meta.tested_commit
            )
        })?;
    let base_tree = json_str(&base_commit, &["tree", "sha"])?.to_string();

    let short_id = &session_id.as_str()[..session_id.as_str().len().min(8)];
    let branch = format!("vanta/fix-{short_id}");
    let tests_path = format!("tests/vanta_{short_id}.py");

    let mut entries: Vec<(String, Value)> = patched_files
        .iter()
        .map(|(path, content)| {
            (
                path.clone(),
                json!({
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "content": content,
                }),
            )
        })
        .collect();
    entries.push((
        tests_path.clone(),
        json!({
            "path": tests_path,
            "mode": "100644",
            "type": "blob",
            "content": accepted_tests,
        }),
    ));
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let tree_entries: Vec<Value> = entries.into_iter().map(|(_, value)| value).collect();

    let tree: Value = octocrab
        .post(
            format!("/repos/{owner}/{repo}/git/trees"),
            Some(&json!({
                "base_tree": base_tree,
                "tree": tree_entries,
            })),
        )
        .await?;
    let tree_sha = json_str(&tree, &["sha"])?.to_string();

    let commit_message = format!("Vanta: auto-generated fix (session {short_id})");
    let commit: Value = octocrab
        .post(
            format!("/repos/{owner}/{repo}/git/commits"),
            Some(&json!({
                "message": commit_message,
                "tree": tree_sha,
                "parents": [meta.tested_commit],
            })),
        )
        .await?;
    let commit_sha = json_str(&commit, &["sha"])?.to_string();

    let _: Value = octocrab
        .post(
            format!("/repos/{owner}/{repo}/git/refs"),
            Some(&json!({
                "ref": format!("refs/heads/{branch}"),
                "sha": commit_sha,
            })),
        )
        .await?;

    let pr_body = build_pr_body(session_id, &meta, evidence);
    let pr: Value = octocrab
        .post(
            format!("/repos/{owner}/{repo}/pulls"),
            Some(&json!({
                "title": format!("Vanta: auto-generated fix (session {short_id})"),
                "head": branch,
                "base": default_branch,
                "body": pr_body,
                "draft": true,
            })),
        )
        .await?;

    Ok(PrResult {
        branch: format!("vanta/fix-{short_id}"),
        pr_url: pr
            .get("html_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        commit_sha: Some(commit_sha),
    })
}

fn github_client(token: &str) -> Result<Octocrab, Box<dyn std::error::Error + Send + Sync>> {
    let mut builder = Octocrab::builder().personal_token(token.to_string());
    if let Ok(base_uri) = std::env::var("VANTA_GITHUB_API_BASE") {
        builder = builder.base_uri(base_uri)?;
    }
    Ok(builder.build()?)
}

fn repo_slug(repo_path: &Path) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(slug) = std::env::var("VANTA_PR_REPO") {
        return Ok(slug);
    }

    let out = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .args(["remote", "get-url", "origin"])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "failed to resolve GitHub remote with `git -C {} remote get-url origin`: {}",
            repo_path.display(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    parse_repo_slug(String::from_utf8_lossy(&out.stdout).trim())
}

fn parse_repo_slug(remote: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let raw = if let Some(rest) = remote.strip_prefix("git@github.com:") {
        rest
    } else if let Some((_, rest)) = remote.split_once("github.com/") {
        rest
    } else {
        return Err(format!("unsupported GitHub remote URL: {remote}").into());
    };
    let trimmed = raw.trim_end_matches(".git");
    if trimmed.split('/').count() != 2 {
        return Err(format!("unsupported GitHub repo slug in remote URL: {remote}").into());
    }
    Ok(trimmed.to_string())
}

fn read_session_start_meta(
    session_log_path: &Path,
) -> Result<SessionStartMeta, Box<dyn std::error::Error + Send + Sync>> {
    let text = fs::read_to_string(session_log_path)?;
    let start: SessionRecord = text
        .lines()
        .map(serde_json::from_str::<SessionRecord>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .find(|record| record.phase == "session.start")
        .ok_or_else(|| {
            format!(
                "session.start record missing from session log {}",
                session_log_path.display()
            )
        })?;

    let fields = parse_summary_fields(&start.summary);
    let tested_commit = required_summary_field(&fields, "tested_commit")?;
    ensure_full_sha("tested_commit", &tested_commit)?;
    Ok(SessionStartMeta {
        tested_commit,
        prompt_version: required_summary_field(&fields, "prompt_version")?,
        vanta_commit: required_summary_field(&fields, "vanta_commit")?,
        vanta_dirty: required_summary_field(&fields, "vanta_dirty")? == "true",
    })
}

fn parse_summary_fields(summary: &str) -> HashMap<String, String> {
    summary
        .split_whitespace()
        .filter_map(|token| token.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn required_summary_field(
    fields: &HashMap<String, String>,
    key: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    fields
        .get(key)
        .cloned()
        .ok_or_else(|| format!("session.start summary is missing `{key}`").into())
}

fn ensure_full_sha(
    label: &str,
    sha: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if sha.len() != 40 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("{label} must be a full 40-character commit SHA; got {sha:?}").into());
    }
    Ok(())
}

fn build_pr_body(session_id: &SessionId, meta: &SessionStartMeta, evidence: &[TestEvidence]) -> String {
    let mut body = String::new();
    body.push_str("## Vanta pull request\n\n");
    body.push_str("### Decision\n");
    body.push_str("- decision: `Approved`\n\n");
    body.push_str("### Session\n");
    body.push_str(&format!("- session_id: `{}`\n", session_id.as_str()));
    body.push_str(&format!("- tested_commit: `{}`\n", meta.tested_commit));
    body.push_str(&format!(
        "- vanta_commit: `{}` (dirty={})\n",
        meta.vanta_commit, meta.vanta_dirty
    ));
    body.push_str(&format!("- prompt_version: `{}`\n\n", meta.prompt_version));
    body.push_str("### Evidence\n");
    for item in evidence {
        body.push_str(&format!(
            "- `{}` intent=`{:?}` outcome=`{:?}` failure_class=`{}`\n",
            item.test_name,
            item.intent,
            item.outcome,
            item.failure_class
                .as_ref()
                .map(|class| format!("{class:?}"))
                .unwrap_or_else(|| "None".to_string())
        ));
    }
    body.push_str("\n---\n");
    body.push_str("*No credential is included in this body or its metadata.*\n");
    body
}

fn json_str<'a>(
    value: &'a Value,
    path: &[&str],
) -> Result<&'a str, Box<dyn std::error::Error + Send + Sync>> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("GitHub response is missing `{}`", path.join(".")))?;
    }
    current
        .as_str()
        .ok_or_else(|| format!("GitHub response field `{}` is not a string", path.join(".")).into())
}

#[cfg(test)]
mod tests {
    use super::open_pr;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    use tempfile::TempDir;

    use crate::core::{
        EvidenceSource, FailureClass, TestEvidence, TestIntent, VerificationOutcome,
    };
    use crate::session::{SessionId, SessionRecord};

    static TEST_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[tokio::test]
    async fn open_pr_uses_tested_commit_from_session_start_and_never_serializes_token() {
        let _env_guard = TEST_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let repo = TestRepo::new();
        let session_id = SessionId::new("sess-pr");
        let session_log = repo.write_session_log(&session_id);
        let server = StubGithubServer::start();
        let token = "github_pat_TEST_TOKEN_DO_NOT_USE";

        unsafe {
            std::env::set_var("VANTA_GITHUB_TOKEN", token);
            std::env::set_var("VANTA_GITHUB_API_BASE", &server.base_url);
            std::env::remove_var("VANTA_PR_REPO");
        }

        let mut patched_files = HashMap::new();
        patched_files.insert(
            "calc.py".to_string(),
            "def shipping_band(weight):\n    return 'heavy'\n".to_string(),
        );
        let evidence = vec![TestEvidence {
            test_name: "tests/test_calc.py::test_boundary_1000".into(),
            outcome: VerificationOutcome::VerifiedWithinBounds,
            failure_class: Some(FailureClass::Logic),
            structured_payload: None,
            intent: TestIntent::BugReproduction,
            evidence_source: EvidenceSource::AttackerGenerated,
        }];

        let result = open_pr(
            repo.path(),
            &patched_files,
            "def test_boundary_1000():\n    assert True\n",
            &evidence,
            session_log.as_path(),
            &session_id,
            "shipping band boundary bug",
        )
        .await
        .unwrap();

        assert_eq!(result.commit_sha.as_deref(), Some("commit-new"));
        assert_eq!(result.pr_url.as_deref(), Some("http://stub/pull/7"));

        let requests = server.requests();
        let commit_body = requests
            .iter()
            .find(|req| req.path.ends_with("/git/commits") && req.method == "POST")
            .expect("commit request missing");
        assert!(
            commit_body
                .body
                .contains("b442dc8f21a44c182f00d3dcb629ec178402c769"),
            "commit parents must contain the full tested commit from session.start; got {}",
            commit_body.body
        );

        let pr_body = requests
            .iter()
            .find(|req| req.path.ends_with("/pulls") && req.method == "POST")
            .expect("pull request request missing");
        assert!(
            pr_body
                .body
                .contains("b442dc8f21a44c182f00d3dcb629ec178402c769"),
            "PR body must mention the tested commit from session.start; got {}",
            pr_body.body
        );
        assert!(
            pr_body
                .body
                .contains("attacker=2;intent_check=1;creator=1;diagnostician=1"),
            "PR body must include prompt_version from session.start; got {}",
            pr_body.body
        );
        assert!(
            !pr_body.body.contains(token),
            "the token must never appear in the PR request body"
        );

        unsafe {
            std::env::remove_var("VANTA_GITHUB_TOKEN");
            std::env::remove_var("VANTA_GITHUB_API_BASE");
        }
    }

    struct TestRepo {
        dir: TempDir,
    }

    impl TestRepo {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            run_git(dir.path(), &["init"]).unwrap();
            run_git(
                dir.path(),
                &["remote", "add", "origin", "https://github.com/Kyu-Yi/vanta-pr-test.git"],
            )
            .unwrap();
            Self { dir }
        }

        fn path(&self) -> &Path {
            self.dir.path()
        }

        fn write_session_log(&self, session_id: &SessionId) -> PathBuf {
            let path = self
                .dir
                .path()
                .join(format!("{}_session.jsonl", session_id.as_str()));
            let record = SessionRecord {
                session_id: session_id.clone(),
                patch_attempt_id: None,
                testing_iteration_id: None,
                test_id: None,
                phase: "session.start".into(),
                summary: "model=stub-model prompt_version=attacker=2;intent_check=1;creator=1;diagnostician=1 vanta_commit=d1e3744 vanta_dirty=true tested_commit=b442dc8f21a44c182f00d3dcb629ec178402c769".into(),
                timestamp: "2026-10-18T00:00:00Z".into(),
                benchmark_instance: None,
                benchmark_label: None,
            };
            fs::write(&path, format!("{}\n", serde_json::to_string(&record).unwrap())).unwrap();
            path
        }
    }

    fn run_git(repo: &Path, args: &[&str]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let out = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()?;
        if !out.status.success() {
            return Err(format!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
        Ok(())
    }

    #[derive(Clone, Debug)]
    struct RequestRecord {
        method: String,
        path: String,
        body: String,
    }

    struct StubGithubServer {
        base_url: String,
        requests: Arc<Mutex<Vec<RequestRecord>>>,
        stop: Arc<AtomicBool>,
        join: Option<thread::JoinHandle<()>>,
    }

    impl StubGithubServer {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let base_url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let requests_clone = Arc::clone(&requests);
            let stop_clone = Arc::clone(&stop);

            let join = thread::spawn(move || {
                while !stop_clone.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let mut request = Vec::new();
                            let mut buf = [0_u8; 4096];
                            let mut header_end = None;
                            let mut content_length = 0_usize;
                            loop {
                                match stream.read(&mut buf) {
                                    Ok(0) => break,
                                    Ok(n) => {
                                        request.extend_from_slice(&buf[..n]);
                                        if header_end.is_none() {
                                            if let Some(pos) =
                                                request.windows(4).position(|w| w == b"\r\n\r\n")
                                            {
                                                header_end = Some(pos + 4);
                                                let headers =
                                                    String::from_utf8_lossy(&request[..pos + 4]);
                                                for line in headers.lines() {
                                                    if let Some(value) = line
                                                        .strip_prefix("Content-Length:")
                                                        .or_else(|| {
                                                            line.strip_prefix("content-length:")
                                                        })
                                                    {
                                                        content_length =
                                                            value.trim().parse().unwrap_or(0);
                                                    }
                                                }
                                            }
                                        }
                                        if let Some(end) = header_end {
                                            if request.len() >= end + content_length {
                                                break;
                                            }
                                        }
                                    }
                                    Err(err)
                                        if err.kind() == std::io::ErrorKind::WouldBlock =>
                                    {
                                        thread::sleep(Duration::from_millis(10));
                                    }
                                    Err(_) => break,
                                }
                            }

                            let request_text = String::from_utf8_lossy(&request);
                            let mut lines = request_text.lines();
                            let first = lines.next().unwrap_or_default();
                            let mut first_parts = first.split_whitespace();
                            let method = first_parts.next().unwrap_or_default().to_string();
                            let path = first_parts.next().unwrap_or_default().to_string();
                            let body = header_end
                                .map(|end| String::from_utf8_lossy(&request[end..]).to_string())
                                .unwrap_or_default();
                            requests_clone
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .push(RequestRecord {
                                    method: method.clone(),
                                    path: path.clone(),
                                    body,
                                });

                            let response = if method == "GET" && path == "/repos/Kyu-Yi/vanta-pr-test"
                            {
                                json!({ "default_branch": "main" })
                            } else if method == "GET"
                                && path
                                    == "/repos/Kyu-Yi/vanta-pr-test/git/commits/b442dc8f21a44c182f00d3dcb629ec178402c769"
                            {
                                json!({
                                    "sha": "b442dc8f21a44c182f00d3dcb629ec178402c769",
                                    "tree": { "sha": "tree-base" }
                                })
                            } else if method == "POST"
                                && path == "/repos/Kyu-Yi/vanta-pr-test/git/trees"
                            {
                                json!({ "sha": "tree-new" })
                            } else if method == "POST"
                                && path == "/repos/Kyu-Yi/vanta-pr-test/git/commits"
                            {
                                json!({ "sha": "commit-new" })
                            } else if method == "POST"
                                && path == "/repos/Kyu-Yi/vanta-pr-test/git/refs"
                            {
                                json!({ "ref": "refs/heads/vanta/fix-sess-pr" })
                            } else if method == "POST" && path == "/repos/Kyu-Yi/vanta-pr-test/pulls"
                            {
                                json!({
                                    "number": 7,
                                    "html_url": "http://stub/pull/7"
                                })
                            } else {
                                json!({ "error": format!("unhandled {method} {path}") })
                            };

                            let response_body = response.to_string();
                            let status = if response.get("error").is_some() {
                                "404 Not Found"
                            } else {
                                "200 OK"
                            };
                            let http = format!(
                                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                response_body.len(),
                                response_body
                            );
                            let _ = stream.write_all(http.as_bytes());
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Self {
                base_url,
                requests,
                stop,
                join: Some(join),
            }
        }

        fn requests(&self) -> Vec<RequestRecord> {
            self.requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    impl Drop for StubGithubServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(self.base_url.trim_start_matches("http://"));
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }
}
