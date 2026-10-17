// ============================================================================
// SUBPROCESS BACKEND — fast-path executor for TRUSTED repositories (Spec §7)
// ============================================================================
//
// [VANTA-WHY] Spec §7 fast path: "build the SandboxBackend trait with a plain
// subprocess backend first (run pytest in a temp directory with timeouts). Ship
// V1 on that. Slot Shuru in as a second backend."
//
// [PITFALL] This is NOT a sandbox. It shares the host kernel, filesystem, and
// network. It's a controlled executor for trusted benchmark repos only.
// Untrusted/model-generated code MUST run in shuru.rs microVM.
//
// [SYNTAX] tokio::process::Command + process group (setpgid) for killing
// descendants. pytest spawns children; killing one PID leaves orphans.
// ============================================================================

use crate::sandbox::types::{ExecutionResult, SandboxBackend, SandboxKind};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

// [VANTA-WHY] Memory limit for sandboxed processes. Default 1 GiB address space.
// Spec §7: "Timeouts and memory limits enforced at process level".
// On macOS we set RLIMIT_AS (virtual address space) via libc::setrlimit
// in the child's pre_exec hook.
//
// [PITFALL] On macOS, RLIMIT_AS limits VIRTUAL address space, not RSS.
// torch/transformers map large virtual regions for MPS/GPU, so a 1 GiB
// (or even 8 GiB) limit kills the process with SIGKILL before pytest writes
// its report. Set VANTA_SANDBOX_MEMORY_MB=0 to disable the limit entirely
// for repos with heavy native deps.
fn memory_limit_bytes() -> Option<u64> {
    match std::env::var("VANTA_SANDBOX_MEMORY_MB") {
        Ok(s) => {
            let mb = s.parse::<u64>().unwrap_or(1024);
            if mb == 0 {
                None // disable limit
            } else {
                Some(mb * 1024 * 1024)
            }
        }
        Err(_) => Some(1 * 1024 * 1024 * 1024), // default 1 GiB
    }
}

pub struct SubprocessBackend {
    // [VANTA-WHY] temp_dir is held only for RAII — dropping SubprocessBackend
    // drops TempDir which deletes the directory. The compiler warns "never read"
    // because we never access the field, but it MUST exist to keep the dir alive.
    #[allow(dead_code)]
    temp_dir: TempDir,
    pub root: PathBuf,
}

impl SubprocessBackend {
    /// Create a subprocess backend with a fresh temp dir.
    pub fn new() -> Result<Self, std::io::Error> {
        let temp_dir = TempDir::new()?;
        let root = temp_dir.path().to_path_buf();
        info!(root = %root.display(), "subprocess sandbox spawned (temp dir)");
        Ok(Self { temp_dir, root })
    }

    /// Copy a source directory into the sandbox root.
    pub fn copy_repo_in(&self, src: &Path) -> Result<(), std::io::Error> {
        copy_dir_recursive(src, &self.root)?;
        debug!(src = %src.display(), "repo copied into sandbox");
        Ok(())
    }
}

// [SYNTAX] async fn in trait impl. Rust 1.75+ supports native async fn in traits.
impl SandboxBackend for SubprocessBackend {
    fn kind(&self) -> SandboxKind {
        SandboxKind::Subprocess
    }

    async fn spawn() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self::new()?)
    }

    async fn execute(
        &self,
        cmd: &[&str],
        cwd: &Path,
        timeout_secs: u64,
    ) -> Result<ExecutionResult, Box<dyn std::error::Error + Send + Sync>> {
        if cmd.is_empty() {
            return Err("execute: empty command".into());
        }

        let full_cwd = if cwd.is_absolute() {
            cwd.to_path_buf()
        } else {
            self.root.join(cwd)
        };

        debug!(cmd = ?cmd, cwd = %full_cwd.display(), timeout_secs, "executing command");

        let start = Instant::now();

        // [SYNTAX] Build the command. cmd[0] is the program, rest are args.
        let mut command = Command::new(cmd[0]);
        command
            .args(&cmd[1..])
            .current_dir(&full_cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // [VANTA-WHY] Process group: when we kill the leader, all children
            // (pytest workers, etc.) get SIGKILL too. Without this, pytest
            // subprocesses become orphans and survive the parent kill.
            .process_group(0);

        // [VANTA-WHY] Network disabled by default (best-effort on macOS).
        // We unset http_proxy/https_proxy and set no_proxy=*. True isolation
        // requires the microVM backend. We ALSO run a connectivity check
        // separately (see sandbox::mod.rs network_alive) so that if the
        // network IS killed mid-run we can classify the failure as
        // Infrastructure rather than Logic/Static.
        command.env_remove("http_proxy");
        command.env_remove("https_proxy");
        command.env_remove("HTTP_PROXY");
        command.env_remove("HTTPS_PROXY");
        command.env("no_proxy", "*");
        command.env("NO_PROXY", "*");

        // [VANTA-WHY] Memory limit enforced at process level (optional).
        // pre_exec runs in the forked child BEFORE exec, so setrlimit applies
        // to the process that will exec into python3/pytest.
        // Safety: pre_exec is unsafe because it runs in a forked child where
        // only async-signal-safe functions may be called. setrlimit is
        // async-signal-safe.
        if let Some(mem_limit) = memory_limit_bytes() {
            unsafe {
                command.pre_exec(move || {
                    let rlim = libc::rlimit {
                        rlim_cur: mem_limit,
                        rlim_max: mem_limit,
                    };
                    // RLIMIT_AS = 9 on macOS/Linux (address space limit)
                    libc::setrlimit(libc::RLIMIT_AS, &rlim);
                    Ok(())
                });
            }
        }

        // [SYNTAX] Spawn the child so we hold a Child handle we can kill on
        // timeout. tokio::process::Child::kill sends SIGKILL.
        let child = command.spawn()?;
        // [VANTA-WHY] Grab the PID BEFORE wait_with_output consumes the child. On
        // timeout we kill the process group (negative PID) via libc::kill. Because
        // we set process_group(0), the child is its own group leader and all
        // descendants are in the same group.
        let pid = child.id().unwrap_or(0) as i32;

        // [SYNTAX] tokio::time::timeout wraps the future. If it doesn't
        // complete within timeout_secs, returns Err(Elapsed).
        let output_result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await;

        let wall_time_ms = start.elapsed().as_millis();

        match output_result {
            Ok(Ok(output)) => {
                let result = ExecutionResult {
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    exit_code: output.status.code(),
                    wall_time_ms,
                    timed_out: false,
                };
                debug!(
                    exit_code = result.exit_code,
                    wall_time_ms,
                    "command completed"
                );
                Ok(result)
            }
            Ok(Err(e)) => Err(format!("subprocess execute failed: {e}").into()),
            Err(_elapsed) => {
                warn!(timeout_secs, "command timed out — killing process group");
                // [VANTA-WHY] Kill the entire process group. Negative PID sends
                // the signal to every process in the group, including pytest
                // workers and any grandchildren. SIGKILL (9) cannot be caught.
                if pid > 0 {
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
                Ok(ExecutionResult {
                    stdout: String::new(),
                    stderr: format!("TIMEOUT after {timeout_secs}s"),
                    exit_code: None,
                    wall_time_ms,
                    timed_out: true,
                })
            }
        }
    }

    fn destroy(self) {
        // TempDir is dropped here → auto-deletes the directory.
        info!(root = %self.root.display(), "subprocess sandbox destroyed (temp dir removed)");
    }
}

// ============================================================================
// Helper: recursive directory copy
// ============================================================================
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path)?;
        } else {
            std::fs::copy(&path, &dest_path)?;
        }
    }
    Ok(())
}
