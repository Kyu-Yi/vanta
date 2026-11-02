// ============================================================================
// SUITE CACHE — content-keyed reuse of whole-suite runs (M3 item S1)
// ============================================================================
//
// [VANTA-WHY] A whole-suite run is the most expensive thing this harness does
// and the least variable one. Measured through prepare, pandas/1's own suite
// takes ~487 s per invocation, and the in-session native gate runs it twice per
// attempt — `native_original` against `native_patched`. `native_original` is a
// function of the BUG alone (the same original tree, the same test command, the
// same checkpoint), yet a Phase-3 arm-and-seed grid re-measured it once per arm,
// per seed and per attempt: ~10 714 s of a ~20 454 s grid measuring a constant.
//
// The rule is that the key IS the experiment. Everything that can move an
// outcome is in it, and nothing that cannot is:
//
//   * the tree the sandbox is seeded with (the source concat itself, hashed);
//   * the test command, its env and its precompile rule;
//   * the checkpoint, disk and memory the VM is created with (a too-small VM
//     KILLS the run, which is an outcome, not a resource detail);
//   * the wall-clock cap (a run's TIMEOUT is a function of the cap, and the cap
//     is derived per bug from prepare's own measurement — without it in the key
//     one arm's kill would be inherited as a fact about the tree);
//   * the harness build (commit + dirty), because the stored value describes a
//     run performed by a particular binary.
//
// Three rules make the reuse safe rather than merely fast:
//
//   * **A key is the input, so a hit is the same experiment.** See above.
//   * **A bad entry is never used.** An entry that is unreadable, carries
//     another schema, another key, other material, a payload that does not match
//     its own digest, or a payload this build cannot decode is treated as
//     absent: the caller recomputes and overwrites it.
//   * **Reuse is auditable.** Every entry records who wrote it, and the caller
//     is handed the key plus that identity, so `native_suite.summary` can say
//     which earlier session row its native result actually came from.
//
// Where it lives is deliberate (Nathan's speed-package mandate): under `runs/`
// — gitignored — read host-side only as data, never inside a `vanta_sessions`
// directory and never inside a sandbox. `VANTA_SUITE_CACHE_DIR` pins one
// absolute directory for a batch, so the Python harness's `suite_cache.py` and
// this module agree on where the entries are even though their cwd differs.
//
// [VANTA-WHY] A DIRTY BUILD IS NOT CACHED. `VANTA_BUILD_COMMIT` alone cannot
// separate two development builds of the same commit, and serving one's native
// result to the other would be serving a measurement of different code as if it
// were this one's. The stamp is still in the KEY (so a clean entry can never be
// read by another commit) and dirty disables reuse outright, with the reason
// recorded on the row instead of silently inherited.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::pipeline::candidate::TestCommand;

/// The entry format's version, and part of every key.
///
/// [VANTA-WHY] A key that did not carry the format version could be satisfied by
/// an entry written by an older shape of this module: the tree hash and the test
/// command would still agree, the payload would parse as *something*, and a field
/// added to the payload later would come back missing — read as "no regressions"
/// rather than as "this entry predates the field". Putting the version in the key
/// material makes a format change unable to read old entries at all: the key
/// changes, the entries are never found, and nothing has to be migrated/purged.
pub const SCHEMA_VERSION: u64 = 1;

/// The discriminator for the in-session gate's single `NativeSuiteReport` entry.
///
/// [VANTA-WHY] The Python harness stores a whole `SuitePair` under its own kind in
/// the same directory. Both hash a tree, so a shared key space without a
/// discriminator would let one kind's entry answer the other's lookup if the two
/// ever hashed the same bytes — and the two payloads are not even the same shape.
pub const KIND_NATIVE_REPORT: &str = "native_report";

/// Pins the cache root. Unset uses the repo-relative default below.
pub const CACHE_DIR_ENV: &str = "VANTA_SUITE_CACHE_DIR";

/// Set by `--no-cache` on `run`, `queue` and `prepare`.
///
/// [VANTA-WHY] The flag forces a FRESH run; it does not make the cache invisible.
/// `read_entry` is still consulted, and the note records `would_hit`, so a cached
/// result can be checked against a fresh one on demand — the only way to verify
/// the cache without a second experiment design. The entry is then refreshed.
pub const DISABLE_ENV: &str = "VANTA_NO_CACHE";

/// The default root, relative to the harness's cwd (the repo root).
///
/// `runner.vanta_env` always sets `CACHE_DIR_ENV` to an absolute path, so this is
/// only the fallback for a hand-run binary.
const DEFAULT_CACHE_DIR: &str = "runs/bugsinpy/suite_cache";

/// The build stamp recorded in every key: the compiled-in commit and dirtiness.
///
/// [VANTA-WHY] Build-time provenance rather than runtime discovery, for the same
/// reason `session::vanta_revision` is: a copied or isolated environment must not
/// be able to claim a revision it is not.
pub fn build_stamp() -> String {
    format!(
        "rs:{}:{}",
        env!("VANTA_BUILD_COMMIT"),
        env!("VANTA_BUILD_DIRTY")
    )
}

/// Why the cache cannot be USED on this invocation, or `None`.
///
/// Pure over its two inputs so a test does not have to mutate the process
/// environment to exercise either branch.
pub fn disabled_reason_from(no_cache_raw: &str, dirty: &str) -> Option<&'static str> {
    if flag_is_set(no_cache_raw) {
        return Some("no_cache");
    }
    if dirty == "true" {
        return Some("dirty_build");
    }
    None
}

/// [`disabled_reason_from`] over the real environment and this build.
pub fn disabled_reason() -> Option<&'static str> {
    disabled_reason_from(
        &std::env::var(DISABLE_ENV).unwrap_or_default(),
        env!("VANTA_BUILD_DIRTY"),
    )
}

fn flag_is_set(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// The cache root, overridden by [`CACHE_DIR_ENV`] when it is non-empty.
pub fn cache_dir_from(override_raw: &str, cwd: &Path) -> PathBuf {
    let trimmed = override_raw.trim();
    if trimmed.is_empty() {
        cwd.join(DEFAULT_CACHE_DIR)
    } else {
        PathBuf::from(trimmed)
    }
}

/// [`cache_dir_from`] over the real environment and cwd.
pub fn cache_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cache_dir_from(&std::env::var(CACHE_DIR_ENV).unwrap_or_default(), &cwd)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// The key material for one native-suite run, as a canonical JSON document.
///
/// [VANTA-WHY] A `BTreeMap` rendered by `serde_json` sorts its keys, so two
/// processes that mean the same experiment cannot produce two renderings of it —
/// which matters because the key is compared as a string across arms, seeds and
/// batches.
fn native_material(
    source: &str,
    test_cmd: &TestCommand,
    timeout_secs: u64,
    stamp: &str,
    checkpoint_raw: &str,
    disk_raw: &str,
    memory_raw: &str,
) -> Value {
    // Sorted so the recorded order of a `--sandbox-env` flag list cannot change
    // the key: the same environment is the same experiment whatever order the
    // arguments arrived in.
    let mut env = test_cmd.env.clone();
    env.sort();
    let mut material: BTreeMap<String, Value> = BTreeMap::new();
    material.insert("schema".to_string(), json!(SCHEMA_VERSION));
    material.insert("kind".to_string(), json!(KIND_NATIVE_REPORT));
    material.insert("harness".to_string(), json!(stamp));
    // [VANTA-WHY] The TREE is the source concat itself, because that is literally
    // everything the sandbox is seeded with (`seed_source_and_conftest` writes it
    // into the VM root). Hashing the concat rather than a directory means the key
    // describes exactly what the run could see, and nothing the walk order or a
    // stray byte-code file could vary.
    material.insert("tree".to_string(), json!(sha256_hex(source.as_bytes())));
    material.insert("command".to_string(), json!(test_cmd.as_cmd_vec()));
    material.insert("env".to_string(), json!(env));
    material.insert("precompile".to_string(), json!(test_cmd.precompile));
    material.insert("checkpoint".to_string(), json!(checkpoint_raw));
    material.insert("disk_mb".to_string(), json!(disk_raw));
    material.insert("memory_mb".to_string(), json!(memory_raw));
    material.insert("timeout_secs".to_string(), json!(timeout_secs));
    Value::Object(material.into_iter().collect())
}

/// A cache key: the hex digest plus the material it was derived from.
///
/// The material is carried alongside so the entry can store it and a reader can
/// re-verify that the file it found really belongs to this experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct Key {
    pub hex: String,
    pub material: Value,
}

/// The key for one in-session native-suite run.
pub fn native_key(source: &str, test_cmd: &TestCommand, timeout_secs: u64) -> Key {
    native_key_with(
        source,
        test_cmd,
        timeout_secs,
        &build_stamp(),
        &std::env::var("VANTA_SHURU_CHECKPOINT").unwrap_or_default(),
        &std::env::var("VANTA_SHURU_DISK_MB").unwrap_or_default(),
        &std::env::var("VANTA_SHURU_MEMORY_MB").unwrap_or_default(),
    )
}

/// [`native_key`] with every input explicit — the pure form a test drives.
pub fn native_key_with(
    source: &str,
    test_cmd: &TestCommand,
    timeout_secs: u64,
    stamp: &str,
    checkpoint_raw: &str,
    disk_raw: &str,
    memory_raw: &str,
) -> Key {
    let material = native_material(
        source,
        test_cmd,
        timeout_secs,
        stamp,
        checkpoint_raw,
        disk_raw,
        memory_raw,
    );
    let blob = serde_json::to_string(&material).unwrap_or_default();
    Key {
        hex: sha256_hex(blob.as_bytes()),
        material,
    }
}

fn payload_digest(payload: &Value) -> String {
    let blob = serde_json::to_string(payload).unwrap_or_default();
    sha256_hex(blob.as_bytes())
}

/// Where one key's entry lives. The first byte shards the directory so a cache
/// with tens of thousands of entries stays listable.
pub fn entry_path_from(root: &Path, key: &str) -> PathBuf {
    let shard = key.get(..2).unwrap_or("__");
    root.join(shard).join(format!("{key}.json"))
}

/// [`entry_path_from`] against the live cache root.
pub fn entry_path(key: &str) -> PathBuf {
    entry_path_from(&cache_dir(), key)
}

/// The validated entry for `key`, or `None`.
///
/// [VANTA-WHY] `None` covers corruption and mismatch in one place, because both
/// have the same consequence — the caller recomputes — and a caller that had to
/// tell them apart would have one more branch to get wrong. Every check is a
/// reason the entry cannot describe THIS experiment:
///
///   * unreadable or unparseable — a truncated write, a hand-edited file;
///   * another `schema` — an entry from a different format of this module;
///   * another `key` — the file is not where its own content says it belongs;
///   * other `material` — the key was recomputed from different inputs, so the
///     payload describes a different experiment even though it was stored here;
///   * a `payload` that does not match its own `payload_digest` — the file was
///     changed after it was written.
pub fn read_entry_from(root: &Path, key: &str, material: &Value) -> Option<Value> {
    let text = fs::read_to_string(entry_path_from(root, key)).ok()?;
    let data: Value = serde_json::from_str(&text).ok()?;
    if data.get("schema").and_then(Value::as_u64) != Some(SCHEMA_VERSION) {
        return None;
    }
    if data.get("key").and_then(Value::as_str) != Some(key) {
        return None;
    }
    if data.get("material") != Some(material) {
        return None;
    }
    let payload = data.get("payload")?;
    if data.get("payload_digest").and_then(Value::as_str) != Some(payload_digest(payload).as_str())
    {
        return None;
    }
    Some(data)
}

/// [`read_entry_from`] against the live cache root.
pub fn read_entry(key: &str, material: &Value) -> Option<Value> {
    read_entry_from(&cache_dir(), key, material)
}

/// Write `entry` for `key` atomically, so a reader never sees a torn file.
///
/// [VANTA-WHY] The queue runs several bugs at once, and a crashed or killed
/// writer must leave either the previous entry or none. A half-written file that
/// a later lookup found would be REFUSED (correctly) — losing a usable entry to
/// the crash of an unrelated process.
pub fn write_entry_to(
    root: &Path,
    key: &str,
    material: &Value,
    payload: &Value,
    origin: &str,
) -> std::io::Result<PathBuf> {
    let path = entry_path_from(root, key);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let entry = json!({
        "schema": SCHEMA_VERSION,
        "kind": material.get("kind").cloned().unwrap_or(Value::Null),
        "key": key,
        "material": material,
        "created_by": origin,
        "payload_digest": payload_digest(payload),
        "payload": payload,
    });
    let tmp = tempfile::NamedTempFile::new_in(
        path.parent().unwrap_or_else(|| Path::new(".")),
    )?;
    std::io::Write::write_all(&mut tmp.as_file(), entry.to_string().as_bytes())?;
    tmp.persist(&path)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    Ok(path)
}

/// [`write_entry_to`] against the live cache root.
pub fn write_entry(
    key: &str,
    material: &Value,
    payload: &Value,
    origin: &str,
) -> std::io::Result<PathBuf> {
    write_entry_to(&cache_dir(), key, material, payload, origin)
}

/// What a native-suite call should record about the cache, on the row.
///
/// [VANTA-WHY] `key` is the FULL 64-hex digest and not a prefix. A truncated tag
/// keeps a summary line short, but the entry's file name IS the key, so a prefix
/// cannot be used to find the entry the row is talking about — which would defeat
/// the only reason the field exists.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheNote {
    /// The 64-hex key of the run this note describes.
    pub key: String,
    /// `hit`, `miss`, `unreadable`, or `off:<reason>`.
    pub state: String,
    /// The `created_by` of the entry that was found, or `""`.
    pub source: String,
    /// True when an entry existed and was refused (either `--no-cache`/dirty, or
    /// a payload this build could not decode) — the "recompute and record it"
    /// half of the mandate.
    pub would_hit: bool,
}

impl CacheNote {
    /// The space-free token `native_suite.summary` carries for one side.
    ///
    /// `state|key|source|would_hit` — four fields, always, so a reader can split
    /// on `|` without a special case for "no entry" (an empty source field).
    pub fn token(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.state,
            self.key,
            self.source,
            if self.would_hit { 1 } else { 0 }
        )
    }
}

/// What to do for one native-suite invocation, decided before any VM work.
///
/// [VANTA-WHY] Split out of :func:`crate::pipeline::harness::run_native_suite` so
/// the DECISION is testable without booting a sandbox: the pure form takes the
/// disabled reason as an argument, so a test does not mutate the process
/// environment to exercise the `--no-cache` branch, and a "hit" test needs only a
/// temp directory and a written entry.
#[derive(Debug, Clone)]
pub struct CachePlan {
    /// The report to serve without running, when the entry is usable.
    pub reuse: Option<Value>,
    /// What the row records about this call.
    pub note: CacheNote,
    /// Whether a fresh run should overwrite the entry.
    pub write: bool,
}

/// The plan for one native-suite call: read the entry, decide, and record why.
pub fn plan_native_suite_from(
    root: &Path,
    key: &Key,
    disabled: Option<&str>,
) -> CachePlan {
    let found = read_entry_from(root, &key.hex, &key.material);
    let source = found
        .as_ref()
        .and_then(|entry| entry.get("created_by"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut note = CacheNote {
        key: key.hex.clone(),
        state: "miss".to_string(),
        source,
        would_hit: false,
    };
    let Some(entry) = found else {
        return CachePlan {
            reuse: None,
            note,
            write: disabled.is_none(),
        };
    };
    match disabled {
        Some(reason) => {
            note.state = format!("off:{reason}");
            note.would_hit = true;
            CachePlan {
                reuse: None,
                note,
                write: false,
            }
        }
        None => match entry.get("payload") {
            Some(payload) => {
                note.state = "hit".to_string();
                CachePlan {
                    reuse: Some(payload.clone()),
                    note,
                    write: false,
                }
            }
            None => {
                // An entry whose payload key is gone: treat it as corrupt, run
                // fresh and overwrite it, and say so rather than count it a miss.
                note.state = "unreadable".to_string();
                note.would_hit = true;
                CachePlan {
                    reuse: None,
                    note,
                    write: true,
                }
            }
        },
    }
}

/// [`plan_native_suite_from`] against the live root and environment.
pub fn plan_native_suite(key: &Key) -> CachePlan {
    plan_native_suite_from(&cache_dir(), key, disabled_reason())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cmd() -> TestCommand {
        TestCommand::from_shell("python3 -m pytest tests/")
    }

    fn key_for(origin: &str, cmd: &TestCommand) -> Key {
        let _ = origin;
        native_key_with(
            "# FILE: a.py\nx = 1\n",
            cmd,
            120,
            "rs:abc123:false",
            "vanta-python",
            "",
            "",
        )
    }

    #[test]
    fn the_key_is_a_function_of_its_inputs_only() {
        let cmd = test_cmd();
        let first = key_for("a", &cmd);
        let second = key_for("b", &cmd);
        assert_eq!(first.hex, second.hex, "the origin is not an input");

        let other = native_key_with(
            "# FILE: a.py\nx = 2\n",
            &cmd,
            120,
            "rs:abc123:false",
            "vanta-python",
            "",
            "",
        );
        assert_ne!(first.hex, other.hex, "the tree is an input");

        let capped = native_key_with(
            "# FILE: a.py\nx = 1\n",
            &cmd,
            121,
            "rs:abc123:false",
            "vanta-python",
            "",
            "",
        );
        assert_ne!(first.hex, capped.hex, "the cap is an input");

        let rebuilt = native_key_with(
            "# FILE: a.py\nx = 1\n",
            &cmd,
            120,
            "rs:def456:false",
            "vanta-python",
            "",
            "",
        );
        assert_ne!(first.hex, rebuilt.hex, "the harness build is an input");
    }

    #[test]
    fn the_key_ignores_the_order_of_the_recorded_environment() {
        let mut one = test_cmd();
        one.env = vec![
            ("PYTHONPATH".to_string(), "src".to_string()),
            ("PYTHONHASHSEED".to_string(), "0".to_string()),
        ];
        let mut two = test_cmd();
        two.env = vec![
            ("PYTHONHASHSEED".to_string(), "0".to_string()),
            ("PYTHONPATH".to_string(), "src".to_string()),
        ];
        assert_eq!(key_for("a", &one).hex, key_for("b", &two).hex);
    }

    #[test]
    fn a_written_entry_is_read_back_and_a_hit_needs_no_run() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let cmd = test_cmd();
        let key = key_for("writer", &cmd);
        let payload = json!({"ran": true, "tests": [["a.py::test_x", "Passed"]]});
        write_entry_to(root, &key.hex, &key.material, &payload, "sess1/attempt1/original").unwrap();

        let plan = plan_native_suite_from(root, &key, None);
        assert!(plan.reuse.is_some(), "a valid entry is served");
        assert_eq!(plan.note.state, "hit");
        assert_eq!(plan.note.source, "sess1/attempt1/original");
        assert!(!plan.note.would_hit);
        assert!(!plan.write, "a hit rewrites nothing");
        assert_eq!(
            plan.note.token(),
            format!("hit|{}|sess1/attempt1/original|0", key.hex)
        );
    }

    #[test]
    fn a_missing_entry_means_run_and_write() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_for("writer", &test_cmd());
        let plan = plan_native_suite_from(dir.path(), &key, None);
        assert!(plan.reuse.is_none());
        assert_eq!(plan.note.state, "miss");
        assert!(!plan.note.would_hit);
        assert!(plan.write);
    }

    #[test]
    fn no_cache_forces_a_fresh_run_but_still_reports_the_entry() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let key = key_for("writer", &test_cmd());
        let payload = json!({"ran": true});
        write_entry_to(root, &key.hex, &key.material, &payload, "sess1/attempt1/patched").unwrap();

        let plan = plan_native_suite_from(root, &key, Some("no_cache"));
        assert!(plan.reuse.is_none(), "--no-cache never serves the entry");
        assert_eq!(plan.note.state, "off:no_cache");
        assert!(plan.note.would_hit, "what the cache would have returned is recorded");
        assert_eq!(plan.note.source, "sess1/attempt1/patched");
        assert!(!plan.write, "a forced fresh run leaves the entry alone");
    }

    #[test]
    fn a_dirty_build_is_disabled_and_the_reason_is_recorded() {
        assert_eq!(disabled_reason_from("", "true"), Some("dirty_build"));
        assert_eq!(disabled_reason_from("1", "false"), Some("no_cache"));
        assert_eq!(disabled_reason_from("", "false"), None);
        assert_eq!(disabled_reason_from("TRUE", "false"), Some("no_cache"));
        assert_eq!(disabled_reason_from("0", "false"), None);
    }

    #[test]
    fn an_entry_that_does_not_match_its_own_digest_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let key = key_for("writer", &test_cmd());
        write_entry_to(root, &key.hex, &key.material, &json!({"ran": true}), "sess1").unwrap();
        // Corrupt the payload without touching the digest.
        let path = entry_path_from(root, &key.hex);
        let mut data: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        data["payload"] = json!({"ran": false});
        fs::write(&path, data.to_string()).unwrap();

        let plan = plan_native_suite_from(root, &key, None);
        assert!(plan.reuse.is_none(), "a mismatched payload is never served");
        assert_eq!(plan.note.state, "miss", "a refused entry reads as absent");
        assert!(plan.write, "and is overwritten by the fresh run");
    }

    #[test]
    fn an_entry_for_other_material_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let cmd = test_cmd();
        let key = key_for("writer", &cmd);
        let other = native_key_with(
            "# FILE: a.py\nx = 9\n",
            &cmd,
            120,
            "rs:abc123:false",
            "vanta-python",
            "",
            "",
        );
        // Write `key`'s document where `other`'s key would look for it.
        let entry = json!({
            "schema": SCHEMA_VERSION,
            "kind": KIND_NATIVE_REPORT,
            "key": key.hex,
            "material": key.material,
            "created_by": "sess1",
            "payload_digest": payload_digest(&json!({"ran": true})),
            "payload": {"ran": true},
        });
        let path = entry_path_from(root, &other.hex);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, entry.to_string()).unwrap();

        let plan = plan_native_suite_from(root, &other, None);
        assert!(plan.reuse.is_none(), "another experiment's entry is not ours");
        assert_eq!(plan.note.state, "miss");
    }

    #[test]
    fn a_tree_change_changes_the_key_even_at_the_same_size() {
        let cmd = test_cmd();
        let a = native_key_with("aaaa", &cmd, 120, "rs:x:false", "", "", "");
        let b = native_key_with("bbbb", &cmd, 120, "rs:x:false", "", "", "");
        assert_ne!(a.hex, b.hex);
    }

    #[test]
    fn the_cache_root_follows_the_override() {
        let cwd = Path::new("/repo");
        assert_eq!(
            cache_dir_from("", cwd),
            PathBuf::from("/repo/runs/bugsinpy/suite_cache")
        );
        assert_eq!(
            cache_dir_from("  /tmp/suite-cache  ", cwd),
            PathBuf::from("/tmp/suite-cache")
        );
    }
}
