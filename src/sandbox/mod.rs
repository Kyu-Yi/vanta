// ============================================================================
// SANDBOX MODULE — execution backends (Spec §7)
// ============================================================================
// Spec §7: "A single Rust trait exposes spawn, execute, and destroy."
// Two backends: subprocess (fast path, trusted repos) + shuru (microVM security boundary).
// ============================================================================

pub mod deps;
pub mod shuru;
pub mod subprocess;

mod types;

pub use types::{ExecutionResult, SandboxBackend, SandboxKind};

use std::error::Error;
use std::sync::{Once, OnceLock};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendPreference {
    Auto,
    Shuru,
    Subprocess,
}

fn backend_preference() -> Result<BackendPreference, Box<dyn Error + Send + Sync>> {
    match std::env::var("VANTA_SANDBOX_BACKEND")
        .ok()
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        None | Some("auto") => Ok(BackendPreference::Auto),
        Some("shuru") => Ok(BackendPreference::Shuru),
        Some("subprocess") => Ok(BackendPreference::Subprocess),
        Some(other) => Err(format!(
            "invalid VANTA_SANDBOX_BACKEND={other:?}; expected auto, shuru, or subprocess"
        )
        .into()),
    }
}

pub fn requested_backend_label() -> Result<&'static str, Box<dyn Error + Send + Sync>> {
    Ok(match backend_preference()? {
        BackendPreference::Auto => "auto",
        BackendPreference::Shuru => "shuru",
        BackendPreference::Subprocess => "subprocess",
    })
}

pub fn backend_kind_label(kind: SandboxKind) -> &'static str {
    match kind {
        SandboxKind::Subprocess => "subprocess",
        SandboxKind::Shuru => "shuru",
    }
}

/// The complete set of environment keys `--sandbox-env` is allowed to inject.
///
/// [VANTA-WHY] The allow-list *is* the feature. A recorded per-project rule has
/// to be able to put the project's own environment in front of the baseline
/// gates, the native suite, the generated tests and the oracle — every one of
/// those runs must see the same value, or the bug is measured against a
/// different environment than the one its project declares. fastapi's own
/// `scripts/test.sh` exports `PYTHONPATH=./docs_src`, which ~250 of its tests
/// need at collection. matplotlib reads `MPLBACKEND` into `rcParams['backend']`
/// at import, and without it `pyplot` resolves the default to TkAgg, whose
/// import chain reaches `_tkinter.__file__` — absent from this layer's CPython
/// 3.8 — so collection itself dies and the suite is unmeasurable. A general
/// "set any variable" flag would instead be an env-injection channel across
/// the sandbox boundary, which is exactly what this flag refuses to be: only
/// the keys named here may cross.
pub const ALLOWED_SANDBOX_ENV_KEYS: &[&str] = &["PYTHONPATH", "MPLBACKEND"];

/// Parse one `--sandbox-env KEY=VALUE` specification.
///
/// [VANTA-WHY] The value comes only from the harness's recorded per-project
/// rule — it is never typed per bug — so the parser's whole job is to refuse to
/// grow into an env channel. The key must name a member of
/// `ALLOWED_SANDBOX_ENV_KEYS` (a rejected key is the flag refusing to be a
/// general env channel: the error names both the offending key and the allowed
/// set), and the value must be non-empty after trimming. The split is on the
/// FIRST `=`, so a value may itself contain `=`.
pub fn parse_sandbox_env(spec: &str) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    let Some((key, value)) = spec.split_once('=') else {
        return Err(format!("sandbox env {spec:?} is not KEY=VALUE (no '=')").into());
    };
    if key.is_empty() || !ALLOWED_SANDBOX_ENV_KEYS.contains(&key) {
        return Err(format!(
            "sandbox env key {key:?} is not allowed; allowed keys: {}",
            ALLOWED_SANDBOX_ENV_KEYS.join(", ")
        )
        .into());
    }
    if value.trim().is_empty() {
        return Err(format!("sandbox env {key} has an empty value").into());
    }
    Ok((key.to_string(), value.to_string()))
}

/// Parse one `--precompile PATH` specification: a tree-relative `.py` path.
///
/// [VANTA-WHY] The rule is the split's recorded per-project one, never typed per
/// bug, so this parser's job is to refuse a path the compile step could not
/// honour — the same shape of refusal `parse_sandbox_env` makes about keys. The
/// rules mirror the split's own validator (`scripts/bugsinpy/split.py`) so a row
/// that passed the split cannot fail here: non-empty, no absolute path, no
/// backslash, no `..` component, and `.py`. A path that named a file outside the
/// tree would compile something the benchmark does not ship; one the compiler
/// cannot parse (a directory, a non-Python file) would fail silently, and the
/// failure the rule exists to prevent would come back disguised as a suite error.
pub fn parse_precompile_path(spec: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let path = spec.trim();
    if path.is_empty() {
        return Err("precompile path is empty".into());
    }
    if path.starts_with('/') || path.contains('\\') || path.split('/').any(|part| part == "..") {
        return Err(format!(
            "precompile path {spec:?} must be tree-relative with no directory step"
        )
        .into());
    }
    if !path.ends_with(".py") {
        return Err(format!("precompile path {spec:?} must name a .py file").into());
    }
    Ok(path.to_string())
}

/// The pre-compile argv for a suite command: `<cmd[0]> -m compileall -q -f <files>`.
///
/// [VANTA-WHY] This is the in-session half of the harness's recorded `precompile`
/// rule (matplotlib/1: `lib/matplotlib/widgets.py`). Its whole job is ordering:
/// an invalid escape sequence is reported while the module is *compiled*, and
/// matplotlib's pytest configuration escalates warnings to errors, so the first
/// import of that module during collection dies as `SyntaxError`. Compiling first
/// leaves a valid `.pyc`, and nothing is compiled under pytest's filters at all.
///
/// Three choices are load-bearing and are the same three the Python harness's
/// `suite.precompile_lines` makes, so both halves of one rule behave identically:
///
/// * the interpreter is `cmd[0]` — the one about to run the tests. A `.pyc` is
///   only read by the interpreter whose magic number it carries, so compiling
///   with any other interpreter would leave a file the tests ignore and the
///   warning would fire exactly as before.
/// * `-f` forces the write. `compileall` otherwise skips a file whose `.pyc`
///   already records the source's mtime, and that check is second-resolution on
///   the mtime alone — a patch applied in the same second as the previous compile
///   would leave a stale `.pyc` and the warning would come back.
/// * the backends run it as its own line, not `&&`-chained to the command: a
///   compile that failed must not take the test report with it. The shell
///   continues, the suite reports the compile error itself as a collection
///   failure, and the gate reads the honest message instead of "no report at all".
///
/// Empty `precompile` returns an empty argv, so a bug without the rule runs
/// exactly the command it ran before the rule existed.
pub fn precompile_argv(cmd: &[&str], precompile: &[String]) -> Vec<String> {
    if precompile.is_empty() {
        return Vec::new();
    }
    let python = cmd.first().copied().unwrap_or("python3");
    let mut argv: Vec<String> = vec![
        python.to_string(),
        "-m".to_string(),
        "compileall".to_string(),
        "-q".to_string(),
        "-f".to_string(),
    ];
    argv.extend(precompile.iter().cloned());
    argv
}

/// The sandbox kind this process will actually use: the env preference, with
/// `auto` resolved through the shuru availability + boot test.
pub async fn actual_backend_kind() -> Result<SandboxKind, Box<dyn Error + Send + Sync>> {
    let backend = Backend::spawn().await?;
    let kind = backend.kind();
    backend.destroy();
    Ok(kind)
}

pub async fn actual_backend_label() -> Result<&'static str, Box<dyn Error + Send + Sync>> {
    Ok(backend_kind_label(actual_backend_kind().await?))
}

/// Whether the shuru backend can actually be used on this machine right now:
/// the CLI is present AND the VM boot test passes.
///
/// [VANTA-WHY] `actual_backend_kind` answers a different question — "what would
/// `auto` pick under the current env?" With `VANTA_SANDBOX_BACKEND=subprocess`
/// exported it reports `subprocess` even on a machine where shuru boots fine.
/// The rerun needs the machine fact, not the preference: it must know whether a
/// session recorded as `shuru` can be replayed at all, and whether an explicit
/// subprocess→shuru upgrade is even possible.
pub async fn shuru_usable() -> bool {
    shuru_available() && shuru_boot_ok().await
}

/// Cached result of the shuru boot test. We run it ONCE per process because
/// booting a VM takes ~1s and doing it on every Backend::spawn() (one per
/// test in run_tests_parallel) would make the pipeline crawl.
static SHURU_BOOT_OK: OnceLock<bool> = OnceLock::new();

/// Run the shuru boot test once and cache the result.
/// Returns true if shuru VMs boot successfully, false if shuru is
/// available but VMs don't boot (e.g. missing virtualization entitlement).
async fn shuru_boot_ok() -> bool {
    if let Some(&ok) = SHURU_BOOT_OK.get() {
        return ok;
    }
    let boot_test = tokio::process::Command::new("shuru")
        .args(["run", "--", "true"])
        .output()
        .await;
    let ok = boot_test
        .as_ref()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        let stderr = boot_test
            .ok()
            .and_then(|o| String::from_utf8(o.stderr).ok())
            .unwrap_or_default();
        let first_lines = stderr.lines().take(3).collect::<Vec<_>>().join(" | ");
        tracing::warn!(
            "shuru VM boot test failed — will use subprocess for this run. stderr: {}",
            first_lines
        );
        eprintln!("⚠ shuru VM boot test failed: {first_lines}");
    }
    let _ = SHURU_BOOT_OK.set(ok);
    ok
}

// ============================================================================
// network_alive() — quick connectivity check for Infrastructure classification
// ============================================================================
//
// [VANTA-WHY] M2 EXIT TEST: "Kill the network mid-run and it classifies as
// Infrastructure, not a model error." We need a way to detect that the network
// is down. This function attempts a TCP connection to a well-known resolver
// (Google DNS 8.8.8.8:53) with a 2-second timeout. If it fails, the network
// is considered dead → any test failure in that window is Infrastructure.
//
// We use a raw TCP connect (not DNS) because DNS itself needs the network and
// we want to test L3/L4 reachability, not name resolution.
// ============================================================================
pub async fn network_alive() -> bool {
    if std::env::var("VANTA_TEST_FORCE_NETWORK_DOWN")
        .ok()
        .as_deref()
        == Some("1")
    {
        return false;
    }
    use tokio::net::TcpStream;
    match tokio::time::timeout(Duration::from_secs(2), TcpStream::connect("8.8.8.8:53")).await {
        Ok(Ok(_)) => true,
        _ => false,
    }
}

// ============================================================================
// spawn_preferred_backend() — pick shuru if available, else subprocess
// ============================================================================
//
// [VANTA-WHY] Shuru is the real security boundary; subprocess is the fast
// path for trusted repos. We prefer shuru when the CLI is installed, and
// gracefully fall back to subprocess with a warning. This keeps the pipeline
// working on machines without shuru while using the stronger isolation when
// available.
//
// [PITFALL] We can't return `dyn SandboxBackend` easily because the trait
// uses async fn (returns unnameable futures). For M2 we use an enum dispatch
// pattern via a helper that returns one of the two concrete types. Callers
// that need a single type should use `spawn_subprocess()` or `spawn_shuru()`
// directly.
// ============================================================================

/// Returns true if the shuru CLI is available in PATH.
pub fn shuru_available() -> bool {
    shuru::ShuruBackend::shuru_available()
}

/// Spawn the preferred backend: shuru if available, else subprocess.
/// Returns a tagged enum so callers can match on the concrete type.
pub enum Backend {
    Subprocess(subprocess::SubprocessBackend),
    Shuru(shuru::ShuruBackend),
}

/// Guards the one-per-run announcement of which backend actually ran.
///
/// [VANTA-WHY] `Backend::spawn()` is called on every harness phase — dozens of
/// times in a single run (61 times in one observed Shuru run) — so announcing
/// the decision at each spawn floods the log with identical lines. The decision
/// cannot change within a process (the env var is fixed and the VM boot test is
/// memoized), so it is announced exactly once and later spawns stay silent.
static ANNOUNCE_BACKEND: Once = Once::new();

/// Announce the backend decision once per process (stdout).
fn announce_backend(line: &str) {
    ANNOUNCE_BACKEND.call_once(|| println!("{line}"));
}

/// Announce a subprocess fallback once per process (stderr — it is a warning).
fn announce_fallback(line: &str) {
    ANNOUNCE_BACKEND.call_once(|| eprintln!("{line}"));
}

impl Backend {
    pub async fn spawn() -> Result<Self, Box<dyn Error + Send + Sync>> {
        match backend_preference()? {
            BackendPreference::Subprocess => {
                let b = subprocess::SubprocessBackend::spawn().await?;
                tracing::warn!("using SUBPROCESS backend (forced by VANTA_SANDBOX_BACKEND)");
                announce_backend("▌ Backend: subprocess (forced, NO security boundary)");
                Ok(Backend::Subprocess(b))
            }
            BackendPreference::Shuru => {
                if !shuru_available() {
                    return Err(
                        "VANTA_SANDBOX_BACKEND=shuru but the shuru CLI is not available".into(),
                    );
                }
                if !shuru_boot_ok().await {
                    return Err(
                        "VANTA_SANDBOX_BACKEND=shuru but the shuru VM boot test failed".into(),
                    );
                }
                let b = shuru::ShuruBackend::spawn().await?;
                tracing::info!("using SHURU microVM backend (forced by VANTA_SANDBOX_BACKEND)");
                announce_backend("▌ Backend: SHURU microVM active (forced)");
                Ok(Backend::Shuru(b))
            }
            BackendPreference::Auto => {
                // [VANTA-WHY] `auto` degrades to subprocess, and a SILENT degrade
                // is indistinguishable from a real shuru run — which is exactly
                // how a missing CLI or a failed VM boot gets misread as "shuru
                // broke". Record the precise reason and print one greppable line
                // so a run can be audited for which boundary actually ran.
                let fallback_reason: String = if !shuru_available() {
                    "shuru CLI not found in PATH".to_string()
                } else if !shuru_boot_ok().await {
                    "shuru VM boot test failed (see warning above for stderr)".to_string()
                } else {
                    match shuru::ShuruBackend::spawn().await {
                        Ok(b) => {
                            tracing::info!("using SHURU microVM backend (preferred)");
                            announce_backend("▌ Backend: SHURU microVM active (auto-selected)");
                            return Ok(Backend::Shuru(b));
                        }
                        Err(e) => format!("shuru spawn failed: {e}"),
                    }
                };
                let b = subprocess::SubprocessBackend::spawn().await?;
                tracing::warn!(
                    reason = %fallback_reason,
                    "using SUBPROCESS backend — FALLBACK (no security boundary)"
                );
                announce_fallback(&format!(
                    "⚠ FALLBACK ACTIVE: subprocess backend (NO security boundary) — reason: {fallback_reason}"
                ));
                Ok(Backend::Subprocess(b))
            }
        }
    }

    pub fn kind(&self) -> SandboxKind {
        match self {
            Backend::Subprocess(b) => b.kind(),
            Backend::Shuru(b) => b.kind(),
        }
    }

    pub fn root(&self) -> &std::path::Path {
        match self {
            Backend::Subprocess(b) => &b.root,
            Backend::Shuru(b) => &b.root,
        }
    }

    pub async fn execute(
        &self,
        cmd: &[&str],
        cwd: &std::path::Path,
        timeout_secs: u64,
        env: &[(String, String)],
        precompile: &[String],
    ) -> Result<ExecutionResult, Box<dyn Error + Send + Sync>> {
        match self {
            Backend::Subprocess(b) => b.execute(cmd, cwd, timeout_secs, env, precompile).await,
            Backend::Shuru(b) => b.execute(cmd, cwd, timeout_secs, env, precompile).await,
        }
    }

    pub fn destroy(self) {
        match self {
            Backend::Subprocess(b) => b.destroy(),
            Backend::Shuru(b) => b.destroy(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn backend_preference_defaults_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Auto);
    }

    #[test]
    fn backend_preference_accepts_forced_values() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "shuru");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Shuru);
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "subprocess");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Subprocess);
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
    }

    #[test]
    fn backend_preference_rejects_unknown_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "mystery");
        }
        let err = backend_preference().unwrap_err().to_string();
        assert!(err.contains("invalid VANTA_SANDBOX_BACKEND"));
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
    }

    #[test]
    fn parse_sandbox_env_accepts_the_recorded_import_path() {
        assert_eq!(
            parse_sandbox_env("PYTHONPATH=./docs_src").unwrap(),
            ("PYTHONPATH".to_string(), "./docs_src".to_string())
        );
    }

    #[test]
    fn parse_sandbox_env_accepts_the_recorded_backend() {
        assert_eq!(
            parse_sandbox_env("MPLBACKEND=Agg").unwrap(),
            ("MPLBACKEND".to_string(), "Agg".to_string())
        );
    }

    #[test]
    fn parse_sandbox_env_rejects_a_key_the_boundary_must_not_carry() {
        let err = parse_sandbox_env("PATH=/x").unwrap_err().to_string();
        assert!(err.contains("PATH"));
        assert!(err.contains("PYTHONPATH"));
    }

    #[test]
    fn parse_sandbox_env_rejects_an_empty_value() {
        assert!(parse_sandbox_env("PYTHONPATH=").is_err());
    }

    #[test]
    fn parse_sandbox_env_rejects_a_missing_separator() {
        assert!(parse_sandbox_env("PYTHONPATH").is_err());
    }

    #[test]
    fn parse_sandbox_env_splits_on_the_first_equals_only() {
        assert_eq!(
            parse_sandbox_env("PYTHONPATH=./docs_src=./other").unwrap(),
            ("PYTHONPATH".to_string(), "./docs_src=./other".to_string())
        );
    }
}
