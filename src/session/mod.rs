// ============================================================================
// SESSION MODULE — provenance IDs + attempt log to disk (Spec §7 Institutional Memory)
// ============================================================================
//
// WHAT YOU'LL LEARN HERE
//
//   # | Rust concept                  → Where Vanta reuses it later
//  ---+-------------------------------+---------------------------------------
//   1 | Newtype pattern               → Every ID type (SessionId, TestId, etc.) is a
//     |   (struct Wrapper(String))  →   compile-time guarantee you can't pass a
//     |                           →   SessionId where a TestId is expected.
//  ---+-------------------------------+---------------------------------------
//   2 | uuid::Uuid::new_v4()         → Unique IDs for every record. M5 experiment
//     |   (UUID v4 random)            →   results need to join across runs.
//  ---+-------------------------------+---------------------------------------
//   3 | serde_json::to_writer_pretty → Session log written to disk as JSON.
//     |   (pretty JSON file I/O)     →   M5 loads these back for EIR/ECR.
// ============================================================================

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info};
use uuid::Uuid;

use crate::core::{
    counts_toward_metrics, AttemptTransition, EcrEirSnapshot, EvidenceSource, FailureClass,
    TestIntent, TestResult,
};

// ============================================================================
// ID NEWTYPES — compile-time type safety for every record
// ============================================================================
//
// [SYNTAX] pub struct SessionId(String); — the "newtype" pattern. A tuple struct
// wrapping a String. The inner String is PRIVATE (no `pub` before the field), so
// only this module can construct a SessionId. External code gets SessionId::new() or
// SessionId::random().
//
// [VANTA-WHY] Without newtypes, all IDs are just Strings and you can accidentally
// pass a session_id where a test_id is expected (both String, both compile, both
// wrong). Newtypes make the type system catch this. M5's experiment joins
// (session_id → patch_attempt_id → test_id) depend on these being distinct.
//
// [RECIPE] To make a new ID type:
//   #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
//   pub struct ThingId(String);
//   impl ThingId {
//       pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
//       pub fn as_str(&self) -> &str { &self.0 }
//   }

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatchAttemptId(String);

impl PatchAttemptId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TestingIterationId(String);

impl TestingIterationId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TestId(String);

impl TestId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// SessionRecord — one attempt record in the session log
// ============================================================================
//
// [VANTA-WHY] Spec §8 Evidence provenance + Spec §7 Institutional Memory V1:
// "flat JSON incident files". Every attempt (Creator call, Attacker call, BATE
// validation, Diagnostician classification) produces a record. M5 loads all records
// for a session to compute EIR/ECR on the fixed shared test set.
//
// [PITFALL] Don't store raw unstructured strings here. Every field must be typed
// so M5 can filter by failure_class, intent, etc. without parsing prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: SessionId,
    /// [VANTA-WHY] `None` for a RUN-LEVEL record (e.g. `session.start`, written
    /// before attempt 1): there is no patch attempt yet, and inventing one would
    /// write an unobserved value to disk as a fact. A null here means "no patch
    /// attempt — exclude from any per-attempt grouping", exactly as a null
    /// `from_patch_attempt_id`/`to_patch_attempt_id` does on a transition row.
    pub patch_attempt_id: Option<PatchAttemptId>,
    pub testing_iteration_id: Option<TestingIterationId>,
    pub test_id: Option<TestId>,
    pub phase: String,
    pub summary: String,
    pub timestamp: String,
    // [VANTA-WHY] Spec §8 + §10 item 13: "benchmark_instance and
    // benchmark_label on every row." When running against BugsInPy/SWE-bench,
    // every record carries the instance ID so M5 can filter per-instance.
    // None for local/Mode-B runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_instance: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_label: Option<String>,
}

// ============================================================================
// TransitionRow — one persisted EIR/ECR transition (Spec §8; the M5 dataset)
// ============================================================================
//
// [VANTA-WHY] This is the RAW dataset the paper's analyses run on, written to
// vanta_sessions/{session_id}_transitions.jsonl. It is deliberately ROW-LEVEL,
// not a summary: the S^(0)-restricted rates, the defect-level cluster bootstrap,
// and the oracle-only robustness check all re-select rows AFTER the run, and none
// of them can be recomputed from an aggregate. Metrics are derived from these
// rows afterwards — raw data immutable, analysis repeatable — so a bug in the
// counting code costs a re-analysis, never a re-run of the model.
//
// [VANTA-WHY] session_id and the patch ids are stamped at persistence time:
// they are properties of the RUN, not of the comparison the Governor computes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRow {
    pub session_id: SessionId,
    /// Identity of the candidate the OLDER attempt ran, or `null` if that
    /// attempt produced no candidate to identify (skipped, censored, or its
    /// patch never applied).
    pub from_patch_attempt_id: Option<PatchAttemptId>,
    /// Identity of the candidate the NEWER attempt ran, or `null` (see above).
    ///
    /// [VANTA-WHY] Both sides are stored because they are not always different.
    /// In --patch (review) mode the candidate never changes, so a comparison
    /// between review rounds is ATTACK DEPTH against the same code, not a patch
    /// transition. With only the destination id those rows are indistinguishable
    /// from genuine repair transitions; with both, the patch-transition analysis
    /// is one filter — both ids non-null and `from_patch_attempt_id != to_patch_attempt_id`.
    ///
    /// [VANTA-WHY] A missing end is NEVER filled in from a neighbouring attempt.
    /// That would write a value nobody observed to disk as a fact, and it is
    /// wrong in both directions: a filled id that differs looks like a genuine
    /// patch transition, a filled id that matches looks like a review round. A
    /// null is filterable — it means "exclude this row" — so the analysis counts
    /// a row only when both ids are present and differ.
    pub to_patch_attempt_id: Option<PatchAttemptId>,
    pub from_attempt: u32,
    pub to_attempt: u32,
    /// Normalised body hash — the stable join key across attempts (not the name).
    pub test_id: String,
    /// Readability only; never used to match attempts.
    pub test_name: String,
    /// Identity of the question set this comparison ran on.
    pub shared_set_id: String,
    pub from_result: TestResult,
    pub to_result: TestResult,
    /// Both sides kept: EIR is judged on the NEW failure's class, ECR on the OLD.
    pub from_failure_class: Option<FailureClass>,
    pub to_failure_class: Option<FailureClass>,
    pub evidence_source: EvidenceSource,
    pub test_intent: TestIntent,
    /// [VANTA-WHY] Inverse of the Governor's `complete`. A censored row is a
    /// recorded HOLE, not a deleted row: it exists so the missing observation is
    /// visible and excludable from EIR/ECR rather than silently absent.
    pub censored: bool,
    /// Membership of S^(0), the shared set as it stood on the first attempt.
    /// The restricted analysis needs this per row and it cannot be rebuilt later.
    pub in_s0: bool,
}

impl TransitionRow {
    /// Stamp a Governor-computed transition with the run identities it needs to
    /// be joined from disk. An end with no real candidate id is passed as
    /// `None`; the row is then CENSORED, never filled in.
    pub fn from_transition(
        transition: &AttemptTransition,
        session_id: &SessionId,
        from_patch_attempt_id: Option<PatchAttemptId>,
        to_patch_attempt_id: Option<PatchAttemptId>,
    ) -> Self {
        // [VANTA-WHY] A missing candidate id on either end is a partially
        // observed transition: it identifies a hole and must never be treated as
        // complete. So it forces the censored flag regardless of what the
        // Governor reported for the comparison itself.
        let censored = !transition.complete
            || from_patch_attempt_id.is_none()
            || to_patch_attempt_id.is_none();
        Self {
            session_id: session_id.clone(),
            from_patch_attempt_id,
            to_patch_attempt_id,
            from_attempt: transition.from_attempt,
            to_attempt: transition.to_attempt,
            test_id: transition.test_id.clone(),
            test_name: transition.test_name.clone(),
            shared_set_id: transition.shared_set_id.clone(),
            from_result: transition.from_result.clone(),
            to_result: transition.to_result.clone(),
            from_failure_class: transition.from_failure_class.clone(),
            to_failure_class: transition.to_failure_class.clone(),
            evidence_source: transition.evidence_source.clone(),
            test_intent: transition.test_intent.clone(),
            censored,
            in_s0: transition.in_s0,
        }
    }

    /// The ONE definition of the *identity* half of the EIR/ECR counting rule: a
    /// row may move the metric only when it is a genuine patch transition — both
    /// ends observed AND different.
    ///
    /// [VANTA-WHY] Two things are deliberately NOT countable here, and both are
    /// kept on disk rather than dropped:
    /// - `from == to` is a review round: the candidate never changed, so the
    ///   comparison measures attack depth against one candidate, not a repair;
    /// - either end `null` (or `censored`) is a partially observed transition —
    ///   a recorded hole, and a hole is excluded, never filled in.
    ///
    /// The companion `crate::core::counts_toward_metrics` supplies the other
    /// half (the Logic-class gate). Together they are the whole rule, defined
    /// once and called by both the console and the M5 analysis layer.
    pub fn is_countable_patch_transition(&self) -> bool {
        !self.censored
            && self.from_patch_attempt_id.is_some()
            && self.to_patch_attempt_id.is_some()
            && self.from_patch_attempt_id != self.to_patch_attempt_id
    }
}

/// The ONE definition of EIR/ECR over persisted rows (Spec §8).
///
/// [VANTA-WHY] EIR/ECR are DERIVED FROM ROWS, never instead of them. This is
/// the single function the console's live print and the M5 analysis layer both
/// call, so there is exactly one definition of the metric and a single test
/// covers both. A row counts only if it is a genuine, complete patch transition
/// (`TransitionRow::is_countable_patch_transition`) whose failing side is
/// Logic-class (`counts_toward_metrics`); then fail->pass is ECR and pass->fail
/// is EIR. Review rounds, censored holes, and non-Logic transitions are excluded
/// here, at calculation time.
pub fn ecr_eir_from_rows(rows: &[TransitionRow]) -> EcrEirSnapshot {
    let mut snap = EcrEirSnapshot::default();
    for row in rows {
        if !row.is_countable_patch_transition() {
            continue;
        }
        if !counts_toward_metrics(
            &row.from_result,
            &row.to_result,
            &row.from_failure_class,
            &row.to_failure_class,
        ) {
            snap.excluded_non_logic += 1;
            continue;
        }
        snap.complete += 1;
        match (&row.from_result, &row.to_result) {
            (TestResult::Failed, TestResult::Passed) => snap.ecr_count += 1,
            (TestResult::Passed, TestResult::Failed) => snap.eir_count += 1,
            _ => {}
        }
    }
    snap
}

// ============================================================================
// TransitionWriter — append-only JSONL sink for transition rows (Spec §8)
// ============================================================================
//
// [VANTA-WHY] Rows are flushed as each transition is computed, not held in
// memory and written as one document at the end. A run can end early BY DESIGN
// (HarnessFailure aborts mid-run) or by accident (the sandbox has already filled
// the disk once); a single end-of-run write would lose every row from the
// session, including the good attempts that ran before the failure. An
// append-and-flush sink keeps the dataset equal to the work already done, so a
// truncated run is still analysable.
//
// [VANTA-WHY] JSONL rather than one JSON array: one row per line means a file
// cut off mid-write still parses line by line up to the last complete line, and
// appending never requires rewriting what is already on disk.
pub struct TransitionWriter {
    path: PathBuf,
    file: fs::File,
}

impl TransitionWriter {
    /// Create vanta_sessions/{session_id}_transitions.jsonl, truncating any
    /// existing file (session ids are unique, so a collision means a bug).
    pub fn open(base_dir: &Path, session_id: &SessionId) -> Result<Self, std::io::Error> {
        fs::create_dir_all(base_dir)?;
        let path = base_dir.join(format!("{}_transitions.jsonl", session_id.as_str()));
        let file = fs::File::create(&path)?;
        info!(path = %path.display(), "transition sink opened");
        Ok(Self { path, file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append rows, one JSON object per line, and flush before returning.
    ///
    /// [VANTA-WHY] The flush is the durability contract: a row that `append`
    /// returned for is on disk even if the process dies on the next statement.
    pub fn append(&mut self, rows: &[TransitionRow]) -> Result<(), std::io::Error> {
        for row in rows {
            let line = serde_json::to_string(row)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            writeln!(self.file, "{line}")?;
        }
        self.file.flush()
    }
}

// ============================================================================
// SessionLog — the per-session record log, durable as records are produced
// ============================================================================
//
// [VANTA-WHY] Spec §7 "Attempt log written to disk as JSON". This log is the
// ATTRIBUTION record for a run: the run-identifying facts (model, provider,
// prompts, Vanta build, backend, benchmark instance) live here, NOT on the
// transition rows. That makes durability a correctness property, not polish. A
// run can end early by design (HarnessFailure is terminal) or by accident (the
// sandbox has already filled the disk once); if the whole log were written once
// at the end, a crash would leave transition rows on disk saying what changed
// with no record of which model produced them — and an unattributed run is
// unusable to a study whose central comparison is across model tiers.
//
// [VANTA-WHY] `open` attaches an append-and-flush sink at
// vanta_sessions/{session_id}_session.jsonl, one record per line, so the very
// first record written — `session.start`, before attempt 1 — already carries the
// run identity to disk. `append` is the durable push; `push` is the
// in-memory-only form M1/M2 use before their single end-of-run write.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionLog {
    pub session_id: SessionId,
    pub records: Vec<SessionRecord>,
    /// [serde(skip)] A live file handle, not data: never serialised, and a
    /// deserialised log has no sink (its records are already on disk).
    #[serde(skip)]
    sink: Option<SessionLogSink>,
}

/// The append-and-flush JSONL sink backing a durable `SessionLog`.
#[derive(Debug)]
struct SessionLogSink {
    path: PathBuf,
    file: fs::File,
}

impl SessionLogSink {
    fn write_record(&mut self, record: &SessionRecord) -> Result<(), std::io::Error> {
        let line = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(self.file, "{line}")?;
        // The flush is the durability contract, exactly as for TransitionWriter:
        // a record `append` returned for is on disk even if the process dies on
        // the next statement.
        self.file.flush()
    }
}

impl SessionLog {
    pub fn new(session_id: SessionId) -> Self {
        info!(session_id = %session_id.as_str(), "session created");
        Self {
            session_id,
            records: Vec::new(),
            sink: None,
        }
    }

    /// Open the durable sink at vanta_sessions/{session_id}_session.jsonl.
    /// Every subsequent `append` is flushed to disk immediately.
    pub fn open(base_dir: &Path, session_id: SessionId) -> Result<Self, std::io::Error> {
        fs::create_dir_all(base_dir)?;
        let path = base_dir.join(format!("{}_session.jsonl", session_id.as_str()));
        let file = fs::File::create(&path)?;
        info!(
            session_id = %session_id.as_str(),
            path = %path.display(),
            "session log sink opened"
        );
        Ok(Self {
            session_id,
            records: Vec::new(),
            sink: Some(SessionLogSink { path, file }),
        })
    }

    /// Path of the durable sink, if this log was opened against a directory.
    pub fn sink_path(&self) -> Option<&Path> {
        self.sink.as_ref().map(|s| s.path.as_path())
    }

    /// Record + durably persist. With a sink attached the record is on disk when
    /// this returns; without one it is only held for a later `write_to_disk`.
    pub fn append(&mut self, record: SessionRecord) -> Result<(), std::io::Error> {
        if let Some(sink) = self.sink.as_mut() {
            sink.write_record(&record)?;
        }
        debug!(phase = %record.phase, "appending session record");
        self.records.push(record);
        Ok(())
    }

    /// In-memory only (M1/M2): collected here, written by `write_to_disk`.
    pub fn push(&mut self, record: SessionRecord) {
        debug!(
            phase = %record.phase,
            test_id = ?record.test_id,
            "appending session record"
        );
        self.records.push(record);
    }

    // [SYNTAX] pub fn write_to_disk(&self, base_dir: &Path) -> Result<PathBuf, std::io::Error>
    // Returns the path written. Creates base_dir if missing.
    //
    // [VANTA-WHY] Spec §7: "Attempt log written to disk as JSON — this becomes
    // M5's raw data". Path = vanta_sessions/{session_id}.json
    pub fn write_to_disk(&self, base_dir: &Path) -> Result<PathBuf, std::io::Error> {
        fs::create_dir_all(base_dir)?;
        let path = base_dir.join(format!("{}.json", self.session_id.as_str()));

        // [SYNTAX] File::create opens for writing (truncates if exists).
        let file = fs::File::create(&path)?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        info!(
            path = %path.display(),
            record_count = self.records.len(),
            "session log written"
        );
        Ok(path)
    }

    // [VANTA-WHY] Returns vanta_sessions/{session_id}/debug/ for forensics files
    // (raw LLM responses, BATE baselines, diagnostic JSON). The LLM layer writes
    // to VANTA_DEBUG_DIR; the binaries set that env var to this path so all
    // forensics are session-scoped.
    pub fn debug_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(self.session_id.as_str()).join("debug")
    }

    // Create the debug directory on disk. Returns the path.
    pub fn ensure_debug_dir(&self, base_dir: &Path) -> Result<PathBuf, std::io::Error> {
        let dir = self.debug_dir(base_dir);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}

// ============================================================================
// Helper: generate a fresh random session ID (UUID v4)
// ============================================================================
//
// [SYNTAX] Uuid::new_v4() generates a random UUID (version 4). to_string()
// gives "550e8400-e29b-41d4-a716-446655440000" format.
pub fn fresh_session_id() -> SessionId {
    SessionId::new(Uuid::new_v4().to_string())
}

pub fn fresh_patch_attempt_id() -> PatchAttemptId {
    PatchAttemptId::new(Uuid::new_v4().to_string())
}

pub fn fresh_test_id() -> TestId {
    TestId::new(Uuid::new_v4().to_string())
}

pub fn fresh_testing_iteration_id() -> TestingIterationId {
    TestingIterationId::new(Uuid::new_v4().to_string())
}

/// The Vanta revision that produced a run, plus whether the working tree was
/// DIRTY when the run started.
///
/// [VANTA-WHY] A commit hash alone is not reproducible provenance while
/// development is active: a run on a dirty tree includes edits that are not in
/// HEAD, so claiming HEAD would name a commit that does not reproduce the
/// binary that actually ran. Recording `dirty` separately makes the paper's rule
/// checkable rather than a matter of trust — a run that feeds the paper must
/// have `dirty == false`.
///
/// `VANTA_COMMIT` (if set, e.g. by CI) overrides the git lookup for the commit
/// only. `dirty` is always measured against the working tree, and a tree whose
/// state cannot be determined (git absent, or cwd not a repo) is reported as
/// dirty: an unverified provenance must never masquerade as a clean one.
pub fn vanta_revision() -> (String, bool) {
    let commit = std::env::var("VANTA_COMMIT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| git_stdout(&["rev-parse", "--short", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = match git_stdout(&["status", "--porcelain"]) {
        Some(out) => !out.is_empty(),
        None => true,
    };
    (commit, dirty)
}

/// Run `git <args>` and return trimmed stdout, or `None` if git is unavailable
/// or the command failed (e.g. the directory is not a repository).
fn git_stdout(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_unique() {
        let a = fresh_session_id();
        let b = fresh_session_id();
        assert_ne!(a, b);
    }

    #[test]
    fn newtype_roundtrip_via_string() {
        let id = SessionId::new("test-123");
        assert_eq!(id.as_str(), "test-123");
    }

    #[test]
    fn session_log_writes_and_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = SessionLog::new(SessionId::new("sess-test"));
        log.push(SessionRecord {
            session_id: SessionId::new("sess-test"),
            patch_attempt_id: Some(PatchAttemptId::new("pa-1")),
            testing_iteration_id: None,
            test_id: None,
            phase: "creator".into(),
            summary: "patch generated".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            benchmark_instance: None,
            benchmark_label: None,
        });
        let path = log.write_to_disk(dir.path()).unwrap();
        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("sess-test"));
        assert!(content.contains("creator"));
    }

    /// Build a minimal Governor transition for exercising the sink.
    fn transition(from_attempt: u32, to_attempt: u32) -> AttemptTransition {
        AttemptTransition {
            from_attempt,
            to_attempt,
            shared_set_id: "set-1".into(),
            test_id: "hash-1".into(),
            test_name: "test_x".into(),
            from_result: TestResult::Passed,
            to_result: TestResult::Failed,
            from_failure_class: None,
            to_failure_class: Some(FailureClass::Logic),
            complete: true,
            in_s0: true,
            evidence_source: EvidenceSource::AttackerGenerated,
            test_intent: TestIntent::BugReproduction,
        }
    }

    fn row(from_attempt: u32, to_attempt: u32, from_pid: &str, to_pid: &str) -> TransitionRow {
        TransitionRow::from_transition(
            &transition(from_attempt, to_attempt),
            &SessionId::new("s1"),
            Some(PatchAttemptId::new(from_pid)),
            Some(PatchAttemptId::new(to_pid)),
        )
    }

    /// The persisted row is the M5 dataset contract. A missing field cannot be
    /// reconstructed after the run, so the schema is asserted by name.
    #[test]
    fn transition_row_serialises_the_full_m5_schema() {
        let value: serde_json::Value =
            serde_json::to_value(row(0, 1, "p0", "p1")).unwrap();

        for key in [
            "session_id",
            "from_patch_attempt_id",
            "to_patch_attempt_id",
            "from_attempt",
            "to_attempt",
            "test_id",
            "test_name",
            "shared_set_id",
            "from_result",
            "to_result",
            "from_failure_class",
            "to_failure_class",
            "evidence_source",
            "test_intent",
            "censored",
            "in_s0",
        ] {
            assert!(value.get(key).is_some(), "row schema is missing `{key}`");
        }
        assert_eq!(value["in_s0"], serde_json::Value::Bool(true));
    }

    #[test]
    fn an_incomplete_transition_is_persisted_as_censored_not_dropped() {
        let mut incomplete = transition(0, 1);
        incomplete.complete = false;

        let stamped = TransitionRow::from_transition(
            &incomplete,
            &SessionId::new("s1"),
            Some(PatchAttemptId::new("p0")),
            Some(PatchAttemptId::new("p1")),
        );
        assert!(
            stamped.censored,
            "an unobserved side is a recorded hole, never a silently complete row"
        );
    }

    /// [VANTA-WHY] A skipped attempt leaves no candidate id for its end of the
    /// comparison. The row must record that as `null` and mark itself censored —
    /// never borrow a neighbouring attempt's id to stay "filterable". A null is
    /// filterable; it means "exclude this".
    #[test]
    fn a_missing_candidate_id_is_null_and_censored_never_filled_in() {
        let row = TransitionRow::from_transition(
            &transition(0, 1),
            &SessionId::new("s1"),
            None,
            Some(PatchAttemptId::new("p1")),
        );

        assert!(
            row.censored,
            "a transition with an unobserved end is partially observed, never complete"
        );
        assert!(row.from_patch_attempt_id.is_none());

        let value = serde_json::to_value(&row).unwrap();
        assert_eq!(
            value["from_patch_attempt_id"],
            serde_json::Value::Null,
            "an unobserved end must serialise as null, not as a guessed id"
        );
        assert_eq!(value["to_patch_attempt_id"], serde_json::json!("p1"));
    }

    /// A review round (`--patch`) compares the SAME candidate against a grown
    /// question set. Both patch ids must be recorded, and equal, so the
    /// patch-transition analysis can filter these rows out with one condition.
    #[test]
    fn review_round_rows_carry_equal_patch_ids() {
        let value = serde_json::to_value(row(1, 2, "same-patch", "same-patch")).unwrap();
        assert_eq!(value["from_patch_attempt_id"], value["to_patch_attempt_id"]);
    }

    /// The session log is durable as records arrive: a `session.start` record
    /// appended before attempt 1 is on disk even if the run dies next, which is
    /// what attributes any surviving transition rows to the run that produced
    /// them.
    #[test]
    fn session_log_records_are_durable_as_they_are_appended() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = SessionLog::open(dir.path(), SessionId::new("durable-session")).unwrap();
        let path = log.sink_path().unwrap().to_path_buf();

        log.append(SessionRecord {
            session_id: SessionId::new("durable-session"),
            // A run-level record: no patch attempt exists yet.
            patch_attempt_id: None,
            testing_iteration_id: None,
            test_id: None,
            phase: "session.start".into(),
            summary: "model=stub-model backend=subprocess".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            benchmark_instance: None,
            benchmark_label: None,
        })
        .unwrap();

        // The run "aborts" here; a fresh read must already see the record.
        let seen = fs::read_to_string(&path).unwrap();
        let records: Vec<SessionRecord> = seen
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].phase, "session.start");
        assert!(records[0].patch_attempt_id.is_none());
    }

    /// [VANTA-WHY] The durability contract: rows appended before an abort must be
    /// on disk, parseable line by line, even though the run never reached its
    /// end-of-run write.
    #[test]
    fn rows_survive_an_abort_after_attempt_two() {
        let dir = tempfile::tempdir().unwrap();
        let session = SessionId::new("abort-test");
        let mut writer = TransitionWriter::open(dir.path(), &session).unwrap();
        let path = writer.path().to_path_buf();

        writer.append(&[row(1, 2, "p0", "p1")]).unwrap();

        // The run "aborts" here — nothing else is written, the writer is never
        // touched again. A FRESH handle must already see the committed row.
        let seen = fs::read_to_string(&path).unwrap();
        assert_eq!(
            seen.lines().count(),
            1,
            "a row must reach disk as soon as it is appended, not at end of run"
        );

        // And it must parse line by line as JSONL.
        let rows: Vec<TransitionRow> = seen
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].from_attempt, 1);
        assert_eq!(rows[0].to_attempt, 2);
        assert_eq!(rows[0].from_patch_attempt_id, Some(PatchAttemptId::new("p0")));
    }

    #[test]
    fn each_append_adds_one_line_and_never_rewrites_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TransitionWriter::open(dir.path(), &SessionId::new("append-test")).unwrap();

        writer.append(&[row(1, 2, "a", "b")]).unwrap();
        writer.append(&[row(2, 3, "b", "c"), row(2, 3, "b", "c")]).unwrap();

        let text = fs::read_to_string(writer.path()).unwrap();
        let rows: Vec<TransitionRow> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 3, "appends accumulate; earlier rows are not rewritten");
        assert_eq!(rows[0].from_patch_attempt_id, Some(PatchAttemptId::new("a")));
        assert_eq!(rows[2].to_patch_attempt_id, Some(PatchAttemptId::new("c")));
    }

    /// Build a row with explicit results, classes and ids, for exercising the
    /// counting rule directly.
    #[allow(clippy::too_many_arguments)]
    fn counting_row(
        from_result: TestResult,
        to_result: TestResult,
        from_class: Option<FailureClass>,
        to_class: Option<FailureClass>,
        from_pid: Option<&str>,
        to_pid: Option<&str>,
    ) -> TransitionRow {
        let bare = AttemptTransition {
            from_attempt: 0,
            to_attempt: 1,
            shared_set_id: "set-1".into(),
            test_id: "hash-1".into(),
            test_name: "test_x".into(),
            from_result,
            to_result,
            from_failure_class: from_class,
            to_failure_class: to_class,
            complete: true,
            in_s0: true,
            evidence_source: EvidenceSource::AttackerGenerated,
            test_intent: TestIntent::BugReproduction,
        };
        TransitionRow::from_transition(
            &bare,
            &SessionId::new("s1"),
            from_pid.map(|s| PatchAttemptId::new(s)),
            to_pid.map(|s| PatchAttemptId::new(s)),
        )
    }

    /// [VANTA-WHY] The counting rule is a SINGLE function so the console and the
    /// M5 analysis layer share one definition of EIR/ECR. Exercising it here on
    /// rows covers both callers, and it pins the two exclusions: a review round
    /// (`from == to`) is attack depth, not a repair; a row with a null end is a
    /// censored hole. Neither may move the metric.
    #[test]
    fn the_counting_rule_counts_only_genuine_complete_patch_transitions() {
        let rows = vec![
            // A genuine patch transition that corrected a logic bug -> ECR.
            counting_row(TestResult::Failed, TestResult::Passed, Some(FailureClass::Logic), None, Some("p0"), Some("p1")),
            // A genuine patch transition that introduced a logic bug -> EIR.
            counting_row(TestResult::Passed, TestResult::Failed, None, Some(FailureClass::Logic), Some("p1"), Some("p2")),
            // Review round: the SAME candidate on both ends -> attack depth.
            counting_row(TestResult::Failed, TestResult::Passed, Some(FailureClass::Logic), None, Some("p2"), Some("p2")),
            // A null end: a censored hole, recorded and excluded, never filled in.
            counting_row(TestResult::Failed, TestResult::Passed, Some(FailureClass::Logic), None, None, Some("p3")),
        ];

        let snap = ecr_eir_from_rows(&rows);
        assert_eq!(snap.ecr_count, 1, "only the genuine repair transition is ECR");
        assert_eq!(snap.eir_count, 1, "only the genuine introduction is EIR");
        assert_eq!(
            snap.complete, 2,
            "review rounds and censored holes are excluded from the denominator too"
        );
    }

    /// A non-Logic failure is evidence about the harness, not the candidate, so
    /// it never moves EIR/ECR — but the exclusion is counted, not silent.
    #[test]
    fn a_non_logic_failure_is_excluded_from_eir_and_ecr() {
        let rows = vec![
            // "corrected" a harness blip: not a correction.
            counting_row(TestResult::Failed, TestResult::Passed, Some(FailureClass::Infrastructure), None, Some("p0"), Some("p1")),
            // "introduced" an error the environment caused: not an introduction.
            counting_row(TestResult::Passed, TestResult::Failed, None, Some(FailureClass::Static), Some("p1"), Some("p2")),
            // Infrastructure timeouts arrive as Error, but they are still
            // failing sides and must be excluded from the denominator too.
            counting_row(TestResult::Passed, TestResult::Error, None, Some(FailureClass::Infrastructure), Some("p2"), Some("p3")),
            // A real logic regression still counts.
            counting_row(TestResult::Passed, TestResult::Failed, None, Some(FailureClass::Logic), Some("p3"), Some("p4")),
        ];

        let snap = ecr_eir_from_rows(&rows);
        assert_eq!(snap.ecr_count, 0);
        assert_eq!(snap.eir_count, 1);
        assert_eq!(snap.complete, 1);
        assert_eq!(
            snap.excluded_non_logic, 3,
            "the exclusions must be visible so an evidence-starved run reads as thin"
        );
    }

    /// A row censored by the Governor (an unobserved side) is not counted even
    /// when both patch ids happen to be present and equal.
    #[test]
    fn a_censored_row_is_never_counted() {
        let mut censored = counting_row(
            TestResult::Failed,
            TestResult::Passed,
            Some(FailureClass::Logic),
            None,
            Some("p0"),
            Some("p1"),
        );
        censored.censored = true;

        let snap = ecr_eir_from_rows(&[censored]);
        assert_eq!(snap.complete, 0);
        assert_eq!(snap.ecr_count, 0);
        assert_eq!(snap.eir_count, 0);
    }
}
