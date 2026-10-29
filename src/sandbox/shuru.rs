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

// ---------------------------------------------------------------------------
// The VM start-up check — DEV_SHM_GUARD
// ---------------------------------------------------------------------------
//
// [VANTA-WHY] A Shuru microVM boots with NO ``/dev/shm`` at all. CPython's
// ``multiprocessing`` creates its semaphores and shared-memory segments as files
// *in that directory* (``multiprocessing/synchronize.py`` ->
// ``_multiprocessing.SemLock`` -> ``sem_open``), so the first
// ``multiprocessing.Lock()`` in such a VM raises
// ``FileNotFoundError: [Errno 2] No such file or directory`` — with **no
// filename**, because the errno comes from the kernel rather than from a failed
// open of a named path.
//
// Measured, and the shape is why it cost a debugging session: it is the same
// error text a missing compiler produces. On the harness layer
// (``vanta-bugsinpy-py3.8``):
//
//     ls: cannot access '/dev/shm': No such file or directory
//     multiprocessing.Lock()  -> FileNotFoundError: [Errno 2]
//     after `mkdir -p /dev/shm && mount -t tmpfs -o rw,nosuid,nodev tmpfs /dev/shm`
//                             -> the lock is created
//
// [VANTA-WHY] Per boot, not per layer: a mount does not survive into a
// checkpoint, so "the checkpoint has /dev/shm" is not a property that could be
// baked in and read back later. Every boot re-establishes it.
//
// [VANTA-WHY] This is the same check the BugsInPy harness runs
// (``scripts/bugsinpy/shuru.py``, DEV_SHM_GUARD), in the other language this repo
// runs untrusted code from. One rule, two spellings — a second *rule* would drift.
//
// Without it, a bug whose tests use a ``Pool``, a ``Queue`` or a ``Lock`` fails
// inside the VM for a reason that has nothing to do with the code under test: a
// false failure that reads as a model error.
const DEV_SHM_GUARD: &str = r#"VANTA_DEV_SHM=''
if grep -q '^tmpfs /dev/shm ' /proc/mounts 2>/dev/null; then VANTA_DEV_SHM='tmpfs mounted'
elif mkdir -p /dev/shm && mount -t tmpfs -o rw,nosuid,nodev tmpfs /dev/shm; then VANTA_DEV_SHM='tmpfs mounted'
fi
export VANTA_DEV_SHM
[ -n "$VANTA_DEV_SHM" ] || { echo "vanta dev/shm: the VM has no /dev/shm and a tmpfs cannot be mounted on it, so CPython's multiprocessing cannot create a semaphore" >&2; exit 86; }
VANTA_MP_PY="$(command -v python3 || command -v python || true)"
if [ -n "$VANTA_MP_PY" ]; then
  "$VANTA_MP_PY" -c 'import multiprocessing as mp; lock = mp.Lock(); pool = mp.Pool(2); assert pool.map(abs, [-1, -2, -3]) == [1, 2, 3], "Pool.map"; pool.close(); pool.join()' || { echo "vanta multiprocessing smoke: FAILED -- /dev/shm is mounted but multiprocessing cannot use it (Lock and Pool are what every test in this VM would fail on)" >&2; exit 87; }
fi"#;

/// The guest shell line for one sandboxed command: the start-up check, then the command.
///
/// [VANTA-WHY] Split out of :func:`ShuruBackend::execute` so the boot step is
/// testable without booting a VM. The guard is the only part of that command line
/// no caller supplies, so a regression that dropped it would stay invisible until
/// a test using multiprocessing failed for a reason that had nothing to do with
/// the code under test.
///
/// A newline rather than ``&&`` between the two, because the guard ends in an
/// ``if`` block: the command's exit status has to be the command's, and an
/// ``&&``-chain makes every failure look like the chain's.
///
/// ``env`` is the run's recorded ``sandbox_env`` (e.g. ``PYTHONPATH=./docs_src``,
/// the import path fastapi's own ``scripts/test.sh`` exports). Each pair becomes
/// its own ``export K=<value>`` line between the guard and the ``cd`` — newline
/// separated for the same reason as above, so a failed export cannot swallow the
/// command. An empty ``env`` reproduces the previous line exactly, which is what
/// a run that never passed the flag gets.
///
/// ``precompile`` is the run's recorded per-project pre-compile rule. When it is
/// not empty, one ``<cmd[0]> -m compileall -q -f <files>`` line runs after the
/// ``cd`` and before the command, in the same tree and under the same
/// interpreter — see :func:`crate::sandbox::precompile_argv` for why. It is its
/// own line rather than an ``&&``-chain for the same reason the guard is: a
/// compile that failed must not take the test report with it. The exit status
/// the script reports is still the command's, because the command is last. An
/// empty ``precompile`` reproduces the previous line exactly.
fn guest_inner_command(
    guest_cwd: &str,
    cmd: &[&str],
    env: &[(String, String)],
    precompile: &[String],
) -> String {
    let mut exports = String::new();
    for (key, value) in env {
        exports.push_str(&format!("export {key}={}\n", shell_quote_value(value)));
    }
    let pre = crate::sandbox::precompile_argv(cmd, precompile);
    if pre.is_empty() {
        return format!(
            "{DEV_SHM_GUARD}\n{exports}cd {guest_cwd} && {command}",
            command = cmd.join(" ")
        );
    }
    let pre_line: Vec<String> = pre.iter().map(|arg| shell_quote_value(arg)).collect();
    format!(
        "{DEV_SHM_GUARD}\n{exports}cd {guest_cwd}\n{pre}\n{command}",
        pre = pre_line.join(" "),
        command = cmd.join(" ")
    )
}

/// Shell-quote one exported value.
///
/// [VANTA-WHY] The value is single-quoted with embedded single quotes escaped
/// (``'\''``) so a value with whitespace or shell metacharacters cannot break out
/// of the ``export`` line. A plain token (letters, digits, and the path-safe
/// punctuation this rule actually uses) is emitted unchanged, so the common
/// ``PYTHONPATH=./docs_src`` reads as it was written.
fn shell_quote_value(value: &str) -> String {
    let plain = !value.is_empty()
        && value.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '/' | '.' | '_' | '-' | ':' | '=' | '+' | ',' | '@')
        });
    if plain {
        return value.to_string();
    }
    let mut out = String::from("'");
    for c in value.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

pub struct RepoCheckpoint {
    pub name: String,
    pub disk_mb: u64,
    pub pip_freeze: String,
}

pub struct ShuruBackend {
    // [VANTA-WHY] temp_dir held for RAII — dropping ShuruBackend drops TempDir
    // which deletes the host directory that was mounted into VMs.
    #[allow(dead_code)]
    temp_dir: TempDir,
    pub root: PathBuf,
    checkpoint: String,
    disk_mb: u64,
    // [VANTA-WHY] The VM's memory, from VANTA_SHURU_MEMORY_MB. None — the value
    // every caller leaves unset — passes no --memory, so the boot line is exactly
    // what it was before this field existed and shuru picks its own default
    // (2048 MB). The harness sets it only for a project whose suite cannot fit that
    // default (the split's recorded `vm_memory_mb`), through `runner.vanta_env`, so
    // the sandbox the pipeline runs third-party code in is the same size the
    // preparation gates measured the bug in.
    memory_mb: Option<u64>,
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

    fn read_deps_freeze(
        name: &str,
        disk_mb: u64,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let disk = disk_mb.to_string();
        let output = std::process::Command::new("shuru")
            .args([
                "run",
                "--from",
                name,
                "--disk-size",
                &disk,
                "--",
                "sh",
                "-c",
                "cat /opt/vanta-deps-freeze.txt",
            ])
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "failed to read pip freeze from dependency checkpoint '{name}': {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    // ------------------------------------------------------------------
    // ensure_repo_checkpoint — auto-build a checkpoint with the repo's deps
    // ------------------------------------------------------------------
    //
    // [VANTA-WHY] Detects the repo's declared dependencies (requirements.txt /
    // pyproject.toml) and guarantees a checkpoint exists that has them
    // installed. The checkpoint is named `vanta-deps-<fingerprint>` so an
    // identical dep set is built once and reused. Returns:
    //   Ok(Some(RepoCheckpoint))  — use this checkpoint
    //   Ok(None)                  — repo declares no deps; use the base
    //   Err(_)                    — detection or install failed (caller warns
    //                               and falls back to the base checkpoint)
    // ------------------------------------------------------------------
    pub fn ensure_repo_checkpoint(
        repo_root: &Path,
    ) -> Result<Option<RepoCheckpoint>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::sandbox::deps;

        if !Self::shuru_available() {
            return Ok(None);
        }

        let plan = deps::detect(repo_root);
        if plan.is_empty() {
            info!(
                target: "vanta::sandbox::deps",
                repo = %repo_root.display(),
                "no declared dependencies — using base checkpoint"
            );
            return Ok(None);
        }
        if let Some(reason) = deps::policy_refusal_reason(&plan) {
            return Err(format!("dependency checkpoint refused: {reason}").into());
        }

        let name = plan.checkpoint_name();
        if Self::checkpoint_exists(&name) {
            match Self::read_deps_freeze(&name, plan.disk_mb) {
                Ok(pip_freeze) => {
                    info!(
                        target: "vanta::sandbox::deps",
                        checkpoint = %name,
                        disk_mb = plan.disk_mb,
                        "reusing dependency checkpoint"
                    );
                    return Ok(Some(RepoCheckpoint {
                        name,
                        disk_mb: plan.disk_mb,
                        pip_freeze,
                    }));
                }
                Err(err) => {
                    warn!(
                        target: "vanta::sandbox::deps",
                        checkpoint = %name,
                        error = %err,
                        "existing dependency checkpoint missing freeze metadata — rebuilding"
                    );
                    let _ = std::process::Command::new("shuru")
                        .args(["checkpoint", "delete", &name])
                        .output();
                }
            }
        }

        // Ensure the base python checkpoint exists (deps build FROM it).
        let base = std::env::var("VANTA_SHURU_BASE_CHECKPOINT")
            .unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string());
        if !Self::checkpoint_exists(&base) {
            Self::create_python_checkpoint(&base)?;
        }

        Self::build_deps_checkpoint(&name, &base, &plan)?;
        let pip_freeze = Self::read_deps_freeze(&name, plan.disk_mb)?;
        Ok(Some(RepoCheckpoint {
            name,
            disk_mb: plan.disk_mb,
            pip_freeze,
        }))
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
        if let Some(reason) = deps::policy_refusal_reason(&plan) {
            return Err(format!("dependency checkpoint refused: {reason}").into());
        }

        let name = plan.checkpoint_name();
        if Self::checkpoint_exists(&name) {
            info!(
                target: "vanta::sandbox::deps",
                checkpoint = %name,
                "reusing derived dependency checkpoint"
            );
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
            target: "vanta::sandbox::deps",
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
             python3 -m pip freeze > /opt/vanta-deps-freeze.txt\n\
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

        // Resolve the VM's memory. Unset — the normal case — passes no --memory and
        // lets shuru choose its own default. `runner.vanta_env` sets
        // VANTA_SHURU_MEMORY_MB from the row's recorded per-project `vm_memory_mb`, so
        // the sandbox the pipeline runs third-party code in matches the size the
        // preparation gates measured the bug in. A value at or below shuru's default is
        // refused by the split's validator, so a parsed value here is always meaningful.
        let memory_mb = std::env::var("VANTA_SHURU_MEMORY_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|mb| *mb > 0);

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
        Ok(Self { temp_dir, root, checkpoint, disk_mb, memory_mb })
    }

    async fn execute(
        &self,
        cmd: &[&str],
        cwd: &Path,
        timeout_secs: u64,
        env: &[(String, String)],
        precompile: &[String],
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

        // Build the command to run inside the VM. The start-up check comes first on
        // every boot (guest_inner_command): this is the single path
        // ``src/pipeline/harness.rs`` runs third-party code through, and a mount does
        // not persist in a checkpoint, so it cannot be folded into the layer.
        let inner_cmd = guest_inner_command(&guest_cwd, cmd, env, precompile);
        let cmd_str = cmd.join(" ");
        debug!(cmd = %cmd_str, guest_cwd = %guest_cwd, timeout_secs, "shuru exec");

        let start = Instant::now();

        // [SYNTAX] shuru run --from CHECKPOINT --disk-size SIZE [--memory MB] --mount HOST:GUEST:rw --allow-host-writes -- <command>
        // --from starts from the checkpoint so python3/pytest/ruff/mypy (and any
        //   auto-installed repo deps) are available.
        // --disk-size must be >= the checkpoint's base image size, else shuru errors
        //   with "--disk-size 4096MB is smaller than the base image (8192MB)".
        //   self.disk_mb tracks the size the selected checkpoint was created with.
        // --memory is present only when the harness recorded a size for this bug
        //   (self.memory_mb). A bigger VM lowers how many VMs fit in parallel, so an
        //   unrecorded bug sends no flag and boots shuru's own default.
        // :rw allows the VM to write back to the host dir (for report.json).
        let disk = self.disk_mb.to_string();
        let memory = self.memory_mb.map(|mb| mb.to_string());
        let mount = format!("{host_root}:{guest_mount}:rw");
        let mut args: Vec<&str> = vec!["run", "--from", &self.checkpoint, "--disk-size", &disk];
        if let Some(mem) = memory.as_deref() {
            args.push("--memory");
            args.push(mem);
        }
        args.extend_from_slice(&[
            "--mount",
            &mount,
            "--allow-host-writes",
            "--",
            "sh",
            "-c",
            &inner_cmd,
        ]);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("shuru").args(&args).output(),
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

#[cfg(test)]
mod tests {
    use super::*;

    // [VANTA-WHY] These are the only assertions about the guard that do not need a
    // VM. They exist because the failure it prevents is *silent* on the machine doing
    // the developing: a host has /dev/shm, so a boot that skipped the mount works
    // locally and fails inside the VM as a test failure that names no file.

    #[test]
    fn the_guard_mounts_a_tmpfs_and_stops_the_boot_without_one() {
        assert!(DEV_SHM_GUARD.contains("mkdir -p /dev/shm"));
        assert!(DEV_SHM_GUARD.contains("mount -t tmpfs"));
        assert!(DEV_SHM_GUARD.contains("/dev/shm"));
        // Loud, and before the command: the alternative is an `[Errno 2]` twenty
        // minutes into a source build that names no file and no variable.
        assert!(DEV_SHM_GUARD.contains("exit 86"));
        // Idempotent, because the guard also runs on the boot that created a layer.
        assert!(DEV_SHM_GUARD.contains("grep -q '^tmpfs /dev/shm ' /proc/mounts"));
    }

    #[test]
    fn the_guard_smokes_multiprocessing_on_whichever_interpreter_the_vm_has() {
        // Both halves of the smoke: `Lock()` is the call that raises the measured
        // FileNotFoundError, and a `Pool` is what numpy 1.18.5's build reached
        // through numpy/distutils/ccompiler.py.
        assert!(DEV_SHM_GUARD.contains("mp.Lock()"));
        assert!(DEV_SHM_GUARD.contains("mp.Pool(2)"));
        assert!(DEV_SHM_GUARD.contains("exit 87"));
        // Skipped, not failed, when the VM has no interpreter: the base image has
        // none before `create_python_checkpoint` installs one, and a VM that will
        // never run Python cannot have a Python multiprocessing problem.
        assert!(DEV_SHM_GUARD.contains("if [ -n \"$VANTA_MP_PY\" ]; then"));
    }

    #[test]
    fn every_sandboxed_command_runs_the_guard_first() {
        let inner = guest_inner_command("/workspace", &["python", "-m", "pytest"], &[], &[]);
        // First, so a VM that cannot provide a semaphore fails before the command
        // rather than inside it.
        assert!(inner.starts_with(DEV_SHM_GUARD));
        // And the command still ends the line, so its exit status is the line's.
        assert!(inner.ends_with("cd /workspace && python -m pytest"));
    }

    #[test]
    fn the_guard_does_not_swallow_the_command_or_its_status() {
        // A newline, not `&&`: the guard ends in an `if` block, and chaining it with
        // `&&` would make every failure read as the chain's.
        let inner = guest_inner_command("/workspace/x", &["sh", "-c", "exit 3"], &[], &[]);
        assert_eq!(inner, format!("{DEV_SHM_GUARD}\ncd /workspace/x && sh -c exit 3"));
    }

    // [VANTA-WHY] The recorded `sandbox_env` has to reach the guest line without
    // touching the case that never passes the flag: an empty env must reproduce
    // the previous command byte-for-byte, or a run that set nothing would still
    // change the command it shipped.

    #[test]
    fn an_empty_env_reproduces_the_old_guest_line_exactly() {
        assert_eq!(
            guest_inner_command("/workspace/t", &["python", "-m", "pytest"], &[], &[]),
            format!("{DEV_SHM_GUARD}\ncd /workspace/t && python -m pytest")
        );
    }

    #[test]
    fn a_recorded_env_becomes_an_export_before_the_command() {
        let inner = guest_inner_command(
            "/workspace/t",
            &["python", "-m", "pytest"],
            &[("PYTHONPATH".to_string(), "./docs_src".to_string())],
            &[],
        );
        assert!(inner.contains("export PYTHONPATH=./docs_src"));
        assert!(inner.ends_with("cd /workspace/t && python -m pytest"));
    }

    // [VANTA-WHY] The recorded `precompile` rule (item L) has to reach the guest
    // line the same way `sandbox_env` does, or the in-session suite runs of
    // matplotlib/1 die on the compile-time warning the rule exists to disarm. The
    // assertions below are the three load-bearing choices: same interpreter as the
    // command, the forcing `-f`, and its own line rather than an `&&`-chain.

    #[test]
    fn an_empty_precompile_reproduces_the_old_guest_line_exactly() {
        assert_eq!(
            guest_inner_command("/workspace/t", &["python3", "-m", "pytest"], &[], &[]),
            format!("{DEV_SHM_GUARD}\ncd /workspace/t && python3 -m pytest")
        );
    }

    #[test]
    fn a_recorded_precompile_compiles_before_the_command_with_its_interpreter() {
        let inner = guest_inner_command(
            "/workspace/t",
            &["python3", "-m", "pytest"],
            &[],
            &["lib/matplotlib/widgets.py".to_string()],
        );
        let lines: Vec<&str> = inner.lines().collect();
        let cd = lines.iter().position(|l| *l == "cd /workspace/t").unwrap();
        let pre = lines
            .iter()
            .position(|l| *l == "python3 -m compileall -q -f lib/matplotlib/widgets.py")
            .unwrap();
        // Its own line, after the `cd` (so the path is tree-relative) and before the
        // command (so nothing is compiled under pytest's filters) …
        assert!(cd < pre);
        // … and last, so the script's exit status is still the command's, and a
        // failed compile cannot take the test report with it.
        assert_eq!(*lines.last().unwrap(), "python3 -m pytest");
        // Not `&&`-chained to the command.
        assert!(!inner.contains("compileall -q -f lib/matplotlib/widgets.py &&"));
    }
}
