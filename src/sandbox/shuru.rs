// ============================================================================
// SHURU BACKEND — macOS microVM via Shuru (Spec §7 real security boundary)
// ============================================================================
//
// [VANTA-WHY] Spec §7: "shuru.rs — Apple Virtualization.framework microVM via
// Shuru. Yes — the real boundary on macOS." Only the microVM backend is a
// security boundary. Subprocess is NOT.
//
// [PITFALL] Shuru CLI must be in PATH. If not, spawn() returns Infrastructure
// error. The orchestrator should fall back to subprocess (with a warning) for
// trusted repos, or refuse to run untrusted code.
//
// [VANTA-WHY] Shuru 0.7.0 API: one-off VMs only. `shuru run --mount HOST:GUEST
// -- CMD` boots a VM, mounts the host dir, runs CMD, and exits. There is NO
// persistent VM / detach / exec. So each execute() call boots a fresh VM.
//
// [VANTA-WHY] The base shuru OS image has NO python3. We create a checkpoint
// named "vanta-python" with python3 + pytest + ruff + mypy pre-installed. Every VM
// boots from this checkpoint via `shuru run --from vanta-python`. The checkpoint
// is created once (auto) on first spawn() if missing) and reused thereafter.
//
// Shuru CLI: https://github.com/superhq-ai/shuru
// ============================================================================

use crate::sandbox::types::{ExecutionResult, SandboxBackend, SandboxKind};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;
use tracing::{debug, info, warn};

// [VANTA-WHY] Default checkpoint name. Override with VANTA_SHURU_CHECKPOINT env var.
const DEFAULT_CHECKPOINT: &str = "vanta-python";

// [VANTA-WHY] Minimum disk size for any VM. Must be >= the base checkpoint's
// disk size (deps::BASE_DISK_MB) or shuru rejects the boot with
// "--disk-size X is smaller than the base image". Callers raise this via
// VANTA_SHURU_DISK_MB when a dependency checkpoint needs more space.
const DEFAULT_DISK_MB: u64 = crate::sandbox::deps::BASE_DISK_MB;

pub struct ShuruBackend {
    // [VANTA-WHY] temp_dir held for RAII — dropping ShuruBackend drops TempDir
    // which deletes the host directory that was mounted into VMs.
    #[allow(dead_code)]
    temp_dir: TempDir,
    pub root: PathBuf,
    checkpoint: String,
    disk_mb: u64,
}

impl ShuruBackend {
    pub fn shuru_available() -> bool {
        std::process::Command::new("shuru")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // checkpoint_exists — check if a named checkpoint is available
    // ------------------------------------------------------------------
    fn checkpoint_exists(name: &str) -> bool {
        std::process::Command::new("shuru")
            .args(["checkpoint", "list"])
            .output()
            .map(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                stdout.contains(name) || stderr.contains(name)
            })
            .unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // create_python_checkpoint — one-time setup: install python3 + pytest +
    // ruff + mypy and save as a checkpoint.
    // ------------------------------------------------------------------
    //
    // [VANTA-WHY] The base shuru OS image has no python3. We run an install
    // script that tries multiple package managers (apt-get, apk, dnf,
    // pacman) since we don't know the distro. After install, shuru saves the
    // disk state as a checkpoint named "vanta-python". This is a one-time
    // cost (a few minutes); subsequent VM boots are fast.
    // ------------------------------------------------------------------
    fn create_python_checkpoint(name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(checkpoint = %name, "creating python checkpoint (one-time setup, may take several minutes)...");

        // Multi-distro install script. Tries apt-get (Debian/Ubuntu),
        // apk (Alpine), dnf (Fedora/RHEL), pacman (Arch).
        //
        // [PITFALL] Debian 12+ marks python as "externally managed", so
        // `pip install` fails unless --break-system-packages is passed.
        // We also set DEBIAN_FRONTEND=noninteractive to suppress debconf
        // dialog warnings (no TTY in shuru VMs).
        let install_script = r#"
            set -e
            export DEBIAN_FRONTEND=noninteractive
            echo "=== Detecting package manager ==="
            if command -v apt-get >/dev/null 2>&1; then
                echo "Using apt-get"
                apt-get update -qq
                apt-get install -y -qq python3 python3-pip python3-venv
            elif command -v apk >/dev/null 2>&1; then
                echo "Using apk"
                apk add --no-cache python3 py3-pip
            elif command -v dnf >/dev/null 2>&1; then
                echo "Using dnf"
                dnf install -y python3 python3-pip
            elif command -v pacman >/dev/null 2>&1; then
                echo "Using pacman"
                pacman -Sy --noconfirm python python-pip
            else
                echo "ERROR: no supported package manager found"
                exit 1
            fi
            # Ensure pip module is available (some distros split it out)
            python3 -m ensurepip --upgrade 2>/dev/null || true
            echo "=== Installing pytest + ruff + mypy via pip ==="
            # --break-system-packages: required on Debian 12+ / externally-managed envs
            # --root-user-action=ignore: suppress pip's "running as root" warning
            python3 -m pip install --break-system-packages --root-user-action=ignore \
                pytest pytest-json-report ruff mypy
            echo "=== Verifying installs ==="
            python3 --version
            python3 -m pytest --version
            python3 -m ruff --version
            python3 -m mypy --version
            echo "=== Checkpoint setup complete ==="
        "#;

        let output = std::process::Command::new("shuru")
            .args([
                "checkpoint", "create", name,
                "--allow-net",
                "--disk-size", "8192",
                "--", "sh", "-c", install_script,
            ])
            .output()?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "failed to create python checkpoint:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            ).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        info!(checkpoint = %name, "python checkpoint created successfully");
        debug!(stdout = %stdout, "checkpoint create output");
        Ok(())
    }

    // ------------------------------------------------------------------
    // verify_deps_marker — boot the checkpoint and confirm the install marker
    // ------------------------------------------------------------------
    //
    // [VANTA-WHY] `shuru checkpoint create` saves the checkpoint EVEN IF the
    // inner install command fails (observed: pip hit "externally-managed-
    // environment" yet shuru still printed "checkpoint saved"). A silently
    // broken checkpoint would then be cached under its fingerprint and poison
    // every later run. We therefore have the install script `touch` a sentinel
    // file only on success, and verify the sentinel by booting the checkpoint.
    // ------------------------------------------------------------------
    fn verify_deps_marker(name: &str, disk_mb: u64) -> bool {
        let disk = disk_mb.to_string();
        std::process::Command::new("shuru")
            .args([
                "run",
                "--from", name,
                "--disk-size", &disk,
                "--",
                "sh", "-c", "test -f /opt/vanta-deps-ok",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // ensure_repo_checkpoint — auto-build a checkpoint with the repo's deps
    // ------------------------------------------------------------------
    //
    // [VANTA-WHY] Detects the repo's declared dependencies (requirements.txt /
    // pyproject.toml) and guarantees a checkpoint exists that has them
    // installed. The checkpoint is named `vanta-deps-<fingerprint>` so an
    // identical dep set is built once and reused. Returns:
    //   Ok(Some((name, disk_mb))) — use this checkpoint
    //   Ok(None)                  — repo declares no deps; use the base
    //   Err(_)                    — detection or install failed (caller warns
    //                               and falls back to the base checkpoint)
    // ------------------------------------------------------------------
    pub fn ensure_repo_checkpoint(
        repo_root: &Path,
    ) -> Result<Option<(String, u64)>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::sandbox::deps;

        if !Self::shuru_available() {
            return Ok(None);
        }

        let plan = deps::detect(repo_root);
        if plan.is_empty() {
            info!(repo = %repo_root.display(), "no declared dependencies — using base checkpoint");
            return Ok(None);
        }

        let name = plan.checkpoint_name();
        if Self::checkpoint_exists(&name) {
            info!(checkpoint = %name, disk_mb = plan.disk_mb, "reusing dependency checkpoint");
            return Ok(Some((name, plan.disk_mb)));
        }

        // Ensure the base python checkpoint exists (deps build FROM it).
        let base = std::env::var("VANTA_SHURU_BASE_CHECKPOINT")
            .unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string());
        if !Self::checkpoint_exists(&base) {
            Self::create_python_checkpoint(&base)?;
        }

        Self::build_deps_checkpoint(&name, &base, &plan)?;
        Ok(Some((name, plan.disk_mb)))
    }

    // ------------------------------------------------------------------
    // ensure_derived_checkpoint — self-heal: add modules pytest said are missing
    // ------------------------------------------------------------------
    //
    // [VANTA-WHY] Called when the tests import something the manifest did not
    // declare. We build a NEW checkpoint derived from the current one (the
    // parent is never mutated — shuru checkpoints are immutable) that adds the
    // missing packages, then the caller re-runs against it. Returns:
    //   Ok(Some((name, disk_mb))) — new checkpoint to use
    //   Ok(None)                  — nothing to add (empty module list)
    //   Err(_)                    — build failed
    // ------------------------------------------------------------------
    pub fn ensure_derived_checkpoint(
        parent: &str,
        parent_disk_mb: u64,
        missing_modules: &[String],
    ) -> Result<Option<(String, u64)>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::sandbox::deps;

        let plan = deps::derived_plan(parent_disk_mb, missing_modules);
        if plan.is_empty() {
            return Ok(None);
        }

        let name = plan.checkpoint_name();
        if Self::checkpoint_exists(&name) {
            info!(checkpoint = %name, "reusing derived dependency checkpoint");
            return Ok(Some((name, plan.disk_mb)));
        }

        if !Self::checkpoint_exists(parent) {
            return Err(format!("derived checkpoint parent '{parent}' not found").into());
        }

        Self::build_deps_checkpoint(&name, parent, &plan)?;
        Ok(Some((name, plan.disk_mb)))
    }

    // ------------------------------------------------------------------
    // build_deps_checkpoint — create NAME from BASE, install PLAN's packages
    // ------------------------------------------------------------------
    fn build_deps_checkpoint(
        name: &str,
        base: &str,
        plan: &crate::sandbox::deps::DependencyPlan,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            checkpoint = %name,
            base = %base,
            packages = plan.packages.len(),
            disk_mb = plan.disk_mb,
            "creating dependency checkpoint (one-time; installs may take minutes)..."
        );

        // `set -e` + `touch` marker: the sentinel is written ONLY if pip
        // succeeds, so a failed install is caught by verify_deps_marker below.
        let script = format!(
            "set -e\n\
             echo '=== Vanta: installing repo dependencies ==='\n\
             python3 -m pip install {args}\n\
             touch /opt/vanta-deps-ok\n\
             echo '=== Vanta: dependency install complete ==='\n",
            args = plan.pip_args()
        );

        let disk = plan.disk_mb.to_string();
        let status = std::process::Command::new("shuru")
            .args([
                "checkpoint", "create", name,
                "--from", base,
                "--disk-size", &disk,
                "--allow-net",
                "--", "sh", "-c", &script,
            ])
            .status()?;

        if !status.success() || !Self::verify_deps_marker(name, plan.disk_mb) {
            // Remove the (possibly broken) checkpoint so the next run retries.
            let _ = std::process::Command::new("shuru")
                .args(["checkpoint", "delete", name])
                .output();
            return Err(format!(
                "dependency checkpoint '{name}' failed to build (exit {:?}) — deleted; \
                 falling back to base checkpoint",
                status.code()
            )
            .into());
        }

        info!(checkpoint = %name, "dependency checkpoint created successfully");
        Ok(())
    }
}

impl SandboxBackend for ShuruBackend {
    fn kind(&self) -> SandboxKind {
        SandboxKind::Shuru
    }

    async fn spawn() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !Self::shuru_available() {
            warn!("shuru CLI not found in PATH — falling back to subprocess");
            return Err("ShuruBackend: 'shuru' CLI not found in PATH. Install from https://github.com/superhq-ai/shuru".into());
        }

        // Resolve checkpoint name from env or default.
        let checkpoint = std::env::var("VANTA_SHURU_CHECKPOINT")
            .unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string());

        // Resolve the disk size. Callers that build a dependency checkpoint set
        // VANTA_SHURU_DISK_MB to the size that checkpoint was created with;
        // `shuru run --disk-size` must be >= the checkpoint's base image size.
        let disk_mb = std::env::var("VANTA_SHURU_DISK_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DISK_MB)
            .max(DEFAULT_DISK_MB);

        // Ensure the python checkpoint exists. Create it if missing.
        if !Self::checkpoint_exists(&checkpoint) {
            info!(checkpoint = %checkpoint, "python checkpoint not found — creating (one-time)");
            Self::create_python_checkpoint(&checkpoint)?;
        }

        // Create a host temp dir INSIDE the current working directory.
        // [PITFALL] Shuru 0.7.0 restricts --mount to paths within CWD:
        //   "Error: mount host path '/var/folders/...' is outside the current
        //    working directory '/Users/kyu-yi/vanta'. Only paths within CWD
        //    can be mounted."
        // TempDir::new() puts dirs in /var/folders/... which is outside CWD.
        // We use TempDir::new_in(current_dir) so the mount path is within CWD.
        let cwd = std::env::current_dir()?;
        let temp_dir = TempDir::new_in(&cwd)?;
        let root = temp_dir.path().to_path_buf();
        info!(root = %root.display(), checkpoint = %checkpoint, disk_mb, "shuru backend spawned");
        Ok(Self { temp_dir, root, checkpoint, disk_mb })
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

        // [VANTA-WHY] Mount the host root dir at /workspace inside the VM.
        // Files the orchestrator wrote to self.root appear in /workspace.
        let host_root = self.root.to_string_lossy().to_string();
        let guest_mount = "/workspace";

        // Resolve cwd relative to the guest mount point.
        let guest_cwd = if cwd.as_os_str().is_empty() {
            guest_mount.to_string()
        } else if cwd.is_absolute() {
            cwd.to_string_lossy().to_string()
        } else {
            format!("{guest_mount}/{}", cwd.display())
        };

        // Build the command to run inside the VM.
        let inner_cmd = format!("cd {guest_cwd} && {}", cmd.join(" "));
        let cmd_str = cmd.join(" ");
        debug!(cmd = %cmd_str, guest_cwd = %guest_cwd, timeout_secs, "shuru exec");

        let start = Instant::now();

        // [SYNTAX] shuru run --from CHECKPOINT --disk-size SIZE --mount HOST:GUEST:rw --allow-host-writes -- <command>
        // --from starts from the checkpoint so python3/pytest/ruff/mypy (and any
        //   auto-installed repo deps) are available.
        // --disk-size must be >= the checkpoint's base image size, else shuru errors
        //   with "--disk-size 4096MB is smaller than the base image (8192MB)".
        //   self.disk_mb tracks the size the selected checkpoint was created with.
        // :rw allows the VM to write back to the host dir (for report.json).
        let disk = self.disk_mb.to_string();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("shuru")
                .args([
                    "run",
                    "--from", &self.checkpoint,
                    "--disk-size", &disk,
                    "--mount",
                    &format!("{host_root}:{guest_mount}:rw"),
                    "--allow-host-writes",
                    "--",
                    "sh",
                    "-c",
                    &inner_cmd,
                ])
                .output(),
        )
        .await;

        let wall_time_ms = start.elapsed().as_millis();

        match result {
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
                    "shuru command completed"
                );
                Ok(result)
            }
            Ok(Err(e)) => Err(format!("shuru exec failed: {e}").into()),
            Err(_) => {
                warn!(timeout_secs, "shuru command timed out");
                Ok(ExecutionResult {
                    stdout: String::new(),
                    stderr: format!("shuru exec TIMEOUT after {timeout_secs}s"),
                    exit_code: None,
                    wall_time_ms,
                    timed_out: true,
                })
            }
        }
    }

    fn destroy(self) {
        // TempDir is dropped here → auto-deletes the host mount directory.
        info!(root = %self.root.display(), "shuru backend destroyed (host temp dir removed)");
    }
}
