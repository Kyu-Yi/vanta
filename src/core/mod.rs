// ============================================================================
// CORE MODULE — Shared domain types, used by EVERYTHING in Vanta
// ============================================================================
//
// Rule for what goes in this file vs other modules:
//   → IF a type is used by 2+ modules (bootstrap, M1 Creator/Attacker,
//     M2 Diagnostician/Sandbox, M3 Gatekeeper/Governor, M5 experiments),
//     put it HERE in core:: with full derives and fully-commented fields.
//
//   → IF a type is private to one module (e.g. LlmConfig is only in the LLM
//     layer, Config::LlmConfig), put it in that specific module.
//
// This is the FOUNDATIONAL file of the whole project. Every data shape in
// this module is reused unchanged from M0 through M5. When you add a new
// variant to VerificationOutcome 6 months from now, the compiler will find
// every forgotten match arm across the entire codebase. That's the point.
// ============================================================================

// [SYNTAX] use serde::{Serialize, Deserialize}; — brings the two derive
// macros into scope so we can write #[derive(Serialize, Deserialize)] below.
//
// [VANTA-WHY] Session logs (Spec §7 Institutional Memory, M1 session data
// model) are flat JSON files written to disk and reloaded between runs.
// Every enum and struct here must be serializable (Rust → JSON string) and
// deserializable (JSON string → typed Rust struct). Otherwise M5 experiment
// data is lost or you write manual string parsing code that drifts.
use serde::{Serialize, Deserialize};

// ============================================================================
// FailureClass — Phase 2 four-way Diagnostician (Spec §5)
// ============================================================================
//
// Spec §5 Phase 2 says every execution failure is triaged into exactly ONE of
// these 4 classes. Only Logic counts as a model error (the only class that
// feeds EIR/ECR). This enum appears in M2 Diagnostician output, M3 Governor
// filtering (exclude everything except Logic from EIR/ECR), and M5 tables
// (Attacker discard rate, static failure rate).
//
// [PITFALL] Beginners add a "Pass" or "None" variant to this enum. Wrong.
// Pass/fail is a different dimension from failure-class. A test that has NO
// failure has failure_class: Option<FailureClass>::None (see below).
// FailureClass only makes semantic sense when there WAS a failure to classify.
// ============================================================================

// [SYNTAX] Each #[derive(...)] line is SEPARATE and FULLY COMMENTED.
// This is deliberate learning scaffolding. When you come back in 3 months,
// you know which derive does what. You can copy-paste this derive comment
// block onto every new domain type without thinking.
//
// [SYNTAX] #[derive(Debug)] generates an implementation of std::fmt::Debug.
// Lets you write println!("{:?}", value) or dbg!(&value) for quick terminal
// inspection. Every domain type needs Debug for development diagnostics.
//
// [VANTA-WHY] During M2 sandbox debugging you will dump whole
// Vec<TestEvidence> to stderr with dbg!() to figure out why triage
// classified a timeout as Logic. Debug is the fastest way to see values.
#[derive(Debug)]

// [SYNTAX] #[derive(Serialize)] — Serde compile-time code generation.
// Produces a fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>.
// The generated code walks each field/variant and writes it out as JSON keys
// (or YAML, MessagePack, etc. — any Serde backend, not just JSON).
//
// [VANTA-WHY] Spec §7 Memory V1: flat JSON incident files on disk.
// Every evidence record ends up as JSON. Serialize is the "write to file" half.
#[derive(Serialize)]

// [SYNTAX] #[derive(Deserialize)] — reverse of Serialize.
// Produces a fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>.
// Takes a JSON string and turns it back into a validated Rust enum value.
// Invalid variant name → Deserialize error (never silently ignored).
//
// [VANTA-WHY] Re-running a previous session from disk: M5 script loads a
// directory of incident JSON files back into Vec<TestEvidence> and reruns the
// Governor on them to see how EIR/ECR estimates changed with more data.
// Deserialize is the "load from file" half.
#[derive(Deserialize)]

// [SYNTAX] #[derive(Clone)] — produces .clone() method that deep-copies self
// into a new owned value in fresh heap memory where applicable.
//
// [VANTA-WHY] The same TestEvidence record is referenced from BOTH the live
// pipeline (Phase 2) and the Governor's fixed shared test set (Phase 3).
// Cloning avoids borrow checker fights. For ~64-byte structs the cost is
// irrelevant vs correctness. If profiling in M5 shows hot clones, we switch
// to Rc<TestEvidence> then — but not a cycle before.
#[derive(Clone)]

// [SYNTAX] #[derive(PartialEq)] — enables assert_eq!(a, b) comparisons.
// Generated code: recursively compares each field/variant for equality.
//
// [VANTA-WHY] In M2 integration tests we assert:
//   assert_eq!(classified_failure_class, FailureClass::Logic);
// No PartialEq → no asserts. Also needed for Vec dedup, HashSet membership, etc.
#[derive(PartialEq)]

// [SYNTAX] pub enum Name { Variant, ... } — "enum" = sum type = "a value of
// this type is ONE OF these variants, never all at once" (contrast with struct
// which is "all fields exist simultaneously"). pub = any module importing this
// file can construct and match on this enum.
pub enum FailureClass {
    // [VANTA-WHY] Infrastructure = Spec §5 Phase 2 table row 1:
    // Backend spawn/transport failure, abnormal signals, timeouts, OOM,
    // health-check failure. Mechanism: structured result inspection, not
    // exit code alone; reset backend and rerun.
    // → NEVER counts as model error → ALWAYS excluded from EIR/ECR.
    Infrastructure,

    // [VANTA-WHY] Static = Spec §5 Phase 2 row 2:
    // Syntax errors, type errors, import failures, newly introduced lint
    // diagnostics. Mechanism: run BEFORE execution, DIFFERENCED against
    // baseline (only NEW diagnostics after patch count; pre-existing ignored).
    // → Tracked separately. NOT Logic → NOT EIR/ECR counted.
    Static,

    // [VANTA-WHY] Logic = Spec §5 Phase 2 row 3: THE ONLY CLASS THAT COUNTS.
    // Test ran, code executed, assertion failed. Mechanism: structured
    // diagnostic payload returned to the patch source as Creator feedback.
    // → This is what feeds Governor EIR/ECR calculation. The whole pipeline
    // exists to generate trustworthy Logic-class results.
    Logic,

    // [VANTA-WHY] Unknown = Spec §5 Phase 2 row 4: Unclassifiable.
    // Mechanism: rerun in fresh backend, capture logs, ask Attacker to
    // simplify the test, QUARANTINE if persistent (EXCLUDE from EIR/ECR, log
    // the reason). NOTE: Repository and benchmark oracle tests are NEVER
    // quarantined; they escalate to "human review required".
    Unknown,

    // [VANTA-WHY] TestDefect = the failure is a property of the TEST, not of
    // the candidate. Two shapes produce it: (a) the failure reproduces
    // IDENTICALLY on the unpatched tree (patch-insensitive — B v2's
    // HereticSettings parsed pytest's own argv and raised SystemExit on both
    // trees); (b) the exception was raised in an object the TEST defined, or in
    // test setup, before the code under test ran (56eb4fd3's FakeTensor had no
    // `.shape`). Blaming the patch for either is a FALSE finding about the
    // model, so this class is sent back to the Attacker for revision and NEVER
    // counted as Logic. → Excluded from EIR/ECR exactly like Infrastructure.
    TestDefect,
}

/// True when a failure was a genuine LOGIC failure — the only class that is
/// evidence about the candidate rather than about the harness or environment.
pub fn is_logic_failure(class: &Option<FailureClass>) -> bool {
    matches!(class, Some(FailureClass::Logic))
}

/// The ONE definition of whether a transition may move EIR/ECR on the Logic
/// gate — the half of the counting rule that is about failure CLASS, not
/// identity. Both the in-memory `AttemptTransition` and the persisted
/// `TransitionRow` route through here, so the rule has a single definition:
///
/// - a fail->pass is a correction only if the failure that cleared was Logic
///   (clearing a harness blip is not a correction);
/// - a pass->fail is an introduction only if the new failure is Logic (an
///   environment-induced import error is not an introduction);
/// - any non-pass result on either side is countable only if THAT side's
///   failure class is Logic. An Infrastructure/Static/Unknown `Error` still
///   counts as a failing side, so it is excluded too;
/// - an unchanged all-pass result carries no attribution, so it stays
///   COUNTABLE — under the CONDITIONAL definition below it joins EIR's
///   denominator (the row WAS passing at N) and never ECR's.
///
/// [VANTA-WHY] `source` is the ONE class that reopens the exclusion, and only
/// for itself. `FailureClass::Static` stays uncountable for Attacker tests (an
/// `import-not-found` is the environment's fault, not the candidate's), but a
/// row whose OWN source is `EvidenceSource::Static` is a blocking finding that
/// was already differenced against the pre-patch baseline — everything the
/// patch did not introduce is gone, so the finding that remains is the
/// candidate's and a finding that disappears on the next attempt is a real
/// correction. Without this, a static error fixed in N+1 would be structurally
/// invisible exactly like the native correction in 4d55123d.
///
/// The other half of the rule — that the row must be a genuine patch
/// transition, both ends observed and different — is
/// `TransitionRow::is_countable_patch_transition`.
pub fn counts_toward_metrics(
    from_result: &TestResult,
    to_result: &TestResult,
    from_failure_class: &Option<FailureClass>,
    to_failure_class: &Option<FailureClass>,
    source: &EvidenceSource,
) -> bool {
    let static_finding = matches!(source, EvidenceSource::Static);
    let side_is_countable = |result: &TestResult, class: &Option<FailureClass>| {
        matches!(result, TestResult::Passed)
            || is_logic_failure(class)
            || (static_finding && matches!(class, Some(FailureClass::Static)))
    };

    side_is_countable(from_result, from_failure_class)
        && side_is_countable(to_result, to_failure_class)
}

/// FNV-1a 64-bit, hex-encoded. Deterministic across runs and platforms.
fn fnv1a_hex(input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The first 12 hex characters of the SHA-256 of one role's system prompt
/// text(s), taken over the parts joined by U+001F.
///
/// [VANTA-WHY] §14.10 f — prompt provenance. A version LABEL is only as good as
/// the discipline of bumping it: `3588728` (the A3 follow-up) and `d4ea314` (the
/// A10 follow-up) both changed prompt wording after the `a2617b3` bump and left
/// the label alone, so two different prompts were logged under one tag and the
/// logs said "no change" about a change that moves verdicts. A content hash
/// cannot be forgotten: it is derived from the text that is sent.
///
/// [VANTA-WHY] Joined with U+001F (the ASCII unit separator, a character no
/// prompt contains) so the parts' boundaries are unambiguous — without a
/// separator, moving a word from the end of one prompt to the start of the next
/// would leave the concatenation identical and the change invisible.
///
/// The digest is TRUNCATED to 12 hex characters (48 bits): this is a change
/// detector for a handful of prompts, not an attack-resistant fingerprint, and a
/// short tag keeps `run.summary` readable.
pub fn prompt_content_hash(prompts: &[&str]) -> String {
    use sha2::{Digest, Sha256};

    let joined = prompts.join("\u{1f}");
    let digest = Sha256::digest(joined.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex.truncate(12);
    hex
}

/// Canonical identity of a test's PROBE, independent of the name it was given.
///
/// [VANTA-WHY] The shared test set exists so attempt N and attempt N+1 can be
/// compared on the SAME questions. Keying that set by test NAME breaks the
/// guarantee in both directions: the model reusing `test_edge_case` for a
/// genuinely different test silently swaps a body under an existing name, and
/// renaming the same probe (`test_x` -> `test_x_v2`) silently looks like new
/// coverage and double-counts in EIR/ECR. Identity must come from the CONTENT.
pub fn normalized_test_body_hash(test_source: &str) -> String {
    let mut canonical = String::new();
    for raw_line in test_source.lines() {
        let trimmed = raw_line.trim();
        // Comment-only lines and blank lines carry no probe semantics.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Collapse internal whitespace so pure formatting churn cannot fork identity.
        let collapsed = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
        // The function NAME is not part of the probe: two tests asserting the
        // same thing are the same probe even if one was renamed.
        canonical.push_str(&canonicalize_def_line(&collapsed));
        canonical.push('\n');
    }
    fnv1a_hex(canonical.trim_end())
}

/// `def test_x(a, b):` -> `def _(a, b):` — keep the signature, drop the name.
fn canonicalize_def_line(line: &str) -> String {
    if let Some(rest) = line.strip_prefix("def ") {
        if let Some(paren) = rest.find('(') {
            return format!("def _{}", &rest[paren..]);
        }
    }
    line.to_string()
}

/// Deterministic fingerprint of a shared-set snapshot — the identity of the
/// QUESTION SET a comparison was run on.
///
/// [VANTA-WHY] Order-independent by construction (the probe hashes are sorted
/// before hashing), so the same set always yields the same id regardless of
/// HashMap iteration order. It is stored on every transition row so a later
/// analysis can group rows by the question set they were asked without
/// re-deriving the set from the test corpus.
pub fn shared_set_fingerprint<'a, I>(hashes: I) -> String
where
    I: IntoIterator<Item = &'a String>,
{
    let mut sorted: Vec<&str> = hashes.into_iter().map(|h| h.as_str()).collect();
    sorted.sort_unstable();
    fnv1a_hex(&sorted.join(","))
}

// ============================================================================
// TestIntent — Declared intent for every Attacker-generated test (Spec §5)
// ============================================================================
//
// Spec §5 Table: Test Intent / Purpose / Expected on ORIGINAL / Expected on PATCHED
// BATE Phase 1 CANNOT WORK without this field. You literally cannot compare a
// test's baseline result against anything unless you know what the test was
// TRYING to do. A bug_reproduction test should FAIL on original code; a
// behavior_preservation test should PASS on original code. If you get that
// backwards, Phase 1 discards the test because its behavior contradicts its
// declared purpose.
// ============================================================================

#[derive(Debug)]       // [SYNTAX] same explanation as above → dbg!() during dev
#[derive(Serialize)]   // [VANTA-WHY] write to JSON session logs (M1)
#[derive(Deserialize)] // [VANTA-WHY] reload session logs for M5 experiments
#[derive(Clone)]       // [VANTA-WHY] share evidence between pipeline + governor
#[derive(PartialEq)]   // [VANTA-WHY] test assertions: check BATE classification
pub enum TestIntent {
    // [VANTA-WHY] Spec §5 Test Intents row 1:
    // Purpose = proves the reported defect EXISTS.
    // Expected on ORIGINAL → FAIL (proves bug was reproduced).
    // Expected on PATCHED → PASS (proves fix resolved the defect).
    // → FAIL/PASS is the correctness signal. Mismatch → BATE sends back for revision.
    BugReproduction,

    // [VANTA-WHY] Spec §5 Test Intents row 2:
    // Purpose = checks a REQUESTED NEW CAPABILITY works.
    // Expected on ORIGINAL → FAIL or unsupported (route didn't exist yet).
    // Expected on PATCHED → PASS.
    FeatureAcceptance,

    // [VANTA-WHY] Spec §5 Test Intents row 3:
    // Purpose = verifies UNRELATED WORKING BEHAVIOUR survived the patch.
    // Expected on ORIGINAL → PASS.
    // Expected on PATCHED → PASS.
    // → PASS/PASS is the preservation signal. FAIL on patched = REGRESSION.
    // This is the class that catches the "29.6% of SWE-bench patches diverge
    // behaviourally from correct fix" statistic from Spec §2 PatchDiff paper.
    BehaviorPreservation,
}

// ============================================================================
// TestResult — outcome of running a single test (used by BATE + sandbox)
// ============================================================================
#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq)]
pub enum TestResult {
    Passed,
    Failed,
    // [VANTA-WHY] Error = test runner itself errored (import/syntax/collect
    // failure), distinct from Failed = assertion failed during execution.
    Error,
}

// ============================================================================
// VerificationOutcome — Verifier backend 6-valued result (Spec §6)
// ============================================================================
//
// Spec §6 trait Verifier output enum. 6 values = "all pass" is NOT a simple
// decision rule. The Gatekeeper applies a VerificationPolicy (Required /
// BestEffort / Disabled) to decide how each outcome maps to reject / block /
// continue. Each outcome has documented semantics. Counterexample is a hard
// reject regardless of policy. VerifiedWithinBounds is NOT universal proof
// (it's bounded; record the bounds). Inconclusive/Unsupported/Timeout are
// FIRST-CLASS non-binary results. Real code produces them constantly.
// ============================================================================

#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq)]
pub enum VerificationOutcome {
    // [VANTA-WHY] Spec §6 Reading the outcomes honestly:
    // No violation found WITHIN the checked limits/assumptions. NOT universal
    // proof. A bounded checker may only explore loops to depth N. Record N.
    // Example from spec: CrossHair on Python finds no counterexample within
    // its timeout and path budget. This is the best "verified" signal we get.
    VerifiedWithinBounds,

    // [VANTA-WHY] STRONG REJECTION EVIDENCE.
    // Concrete counterexample input that makes the property FALSE.
    // Example from Spec §6: "charge=100, refund=100.01 — refund exceeds charge".
    // → REJECTS PATCH OUTRIGHT regardless of test outcomes. Always.
    // Hard-coded in M3 Gatekeeper policy tables.
    Counterexample,

    // [VANTA-WHY] Tried. Could not decide within budget. First-class result.
    // Real code triggers this constantly (recursion depth, symbolic explosion).
    // Policy mapping: Required → "Block / human review". BestEffort → report.
    Inconclusive,

    // [VANTA-WHY] Backend cannot handle THIS code at all: e.g. Python backend
    // asked to verify Rust, or endpoint uses stateful DB code unsupported by
    // symbolic execution. Policy: Required → Block/review, BestEffort → Continue.
    Unsupported,

    // [VANTA-WHY] Exceeded verification budget (wall-clock time or step count).
    // Policy mapping: Retry once with higher budget. Then: Required→Block/review.
    // BestEffort→Continue unverified. Always reported. Never silently treated as Verified.
    Timeout,

    // [VANTA-WHY] The verifier ITSELF misbehaved (segfault, internal error,
    // exit code 1 on valid input). Infrastructure-level. Not the patch's fault.
    // → Retry in fresh backend. If persists: escalate. NEVER attribute to patch.
    ToolFailure,
}

// ============================================================================
// EvidenceSource — Provenance: who wrote/produced a signal? (Spec §8)
// ============================================================================
//
// Spec §8 Evidence Provenance: "Sources are not equally trustworthy".
// Struct TestEvidence records: test_source (who wrote it) + execution_source
// (what ran it) separately. Struct VerificationEvidence has: property_source
// + outcome_source separately. Why? Example from Spec: solver correctly
// proves property "every refund < charge" — but the real policy allows
// refund EQUAL to charge. The solver was correct. The PROPERTY was wrong.
// The log must say "verified within bounds, property Attacker-generated".
// Never "mathematically proven correct".
// ============================================================================

#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq)]
pub enum EvidenceSource {
    // [VANTA-WHY] Tests written by humans and checked into the target repo
    // before Vanta ever ran. Gold standard. Never quarantined. If an
    // Unknown-class failure hits one of these → "human review required".
    ExistingRepositoryTest,

    // [VANTA-WHY] Benchmark oracle tests (BugsInPy ground truth, Defects4J
    // expected behaviour). Reference truth. NEVER quarantined even if Unknown.
    // Escalate to "verification inconclusive" instead. This is what makes
    // M5 numbers defensible against a paper reviewer.
    BenchmarkOracle,

    // [VANTA-WHY] Generated by Vanta's Attacker agent (LLM). Least-trusted
    // source of tests; subject to BATE Phase 1 validation AND max 3 retries
    // AND scrapping + discard-rate tracking. The 30-40% discard rate from
    // Attacker is a key reliability metric for the paper.
    AttackerGenerated,

    // [VANTA-WHY] Supplied by a human during manual review or when opening
    // an issue. Between ExistingRepositoryTest and AttackerGenerated in trust.
    HumanProvided,

    // [VANTA-WHY] Output of a formal verifier backend: CrossHair counterexample,
    // Kani proof, Z3 query result. HIGH trust for the engine's output. BUT:
    // see Spec §8 provenance warning — the PROPERTY the engine checked may
    // still be wrong. Provenance field must separate the two.
    FormalVerifier,

    // [VANTA-WHY] Evidence extracted directly from the issue description text:
    // "add(2,3) returns 4 but should be 7". Weakest form of property — may be
    // miswritten, ambiguous, or partial. Logged separately from repo tests.
    IssueDescription,

    // [VANTA-WHY] THE REPOSITORY'S OWN test suite, run on the candidate's tree
    // (Spec §5 native-suite gate). It is not `ExistingRepositoryTest` in the
    // sense above: that variant describes a repo test used as part of the shared
    // question set. These are the repo's existing tests run as a whole, and they
    // are the only evidence that does not depend on what the Attacker happened
    // to sample.
    //
    // [VANTA-WHY] Session 4d55123d approved a patch whose ONE real correction
    // was to a native test the previous attempt had broken. The correction was
    // invisible: the native suite wrote only session records, never transition
    // rows, so ECR read 0/0 and the only thing the run actually fixed counted
    // for nothing. A native row has the same fail->pass / pass->fail shape as a
    // shared-set row, keyed by the pytest nodeid, so it can be counted the same
    // way — and, like every other row, separately.
    Native,

    // [VANTA-WHY] A BLOCKING static finding: an error-class diagnostic (ruff
    // `F`/`E9`, or any mypy error) in a file the patch touches, differenced
    // against the pre-patch baseline. Its class is `FailureClass::Static`, and
    // that class is normally NOT countable — an environment-induced
    // `import-not-found` is a fact about the analysis environment, not about the
    // candidate. This source is the discriminator: baseline differencing already
    // removed every finding the patch did not introduce, so what is left IS the
    // candidate's doing, and a finding that disappears on the next attempt is a
    // genuine correction. Rows carry this source so the class gate can count
    // them without reopening the non-Logic exclusion for Attacker-generated
    // tests.
    Static,
}

/// Normalised identity of a blocking static finding: rule + file + message.
///
/// [VANTA-WHY] A static finding has no name and no body hash to key on, so its
/// identity has to be derived from what identifies it: the rule that fired, the
/// file it fired in, and its message. The line number is deliberately EXCLUDED —
/// a finding that survives an edit and only moves down the file is the SAME
/// finding, and keying on the line would read as "the old one was fixed and a
/// new one appeared", which is exactly the false correction this identity
/// exists to prevent. The message is collapsed to single spaces so pure
/// reflowing of a wrapped diagnostic (mypy wraps long type names) does not fork
/// the identity either.
pub fn static_finding_identity(rule: &str, file: &str, message: &str) -> String {
    let normalized = message.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{rule}|{file}|{normalized}")
}

// ============================================================================
// TestEvidence — Single test result record (composite of the above 4 enums)
// ============================================================================
//
// This is the CENTRAL DATA RECORD of Vanta. Every test produces one.
// Phase 1 (BATE) baseline validation produces a partial record. Phase 2
// (execution) fills in the outcome and failure class. Phase 3 (Governor)
// aggregates over collections of these records to compute EIR/ECR.
//
// Every field here maps to a Spec rule. When in doubt: go back to the spec.
// ============================================================================

#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct TestEvidence {
    // [VANTA-WHY] Stable unique name. Used for fixed shared set comparison.
    // Spec §8 Fixed shared test set: comparing attempt N vs N+1 on the SAME
    // saved tests requires identifying "did test T change outcome between runs".
    // Without a deterministic name you can't join records.
    pub test_name: String,

    // [VANTA-WHY] The 6-valued verification result (see enum above).
    // This field answers: "did this test/property verification pass?"
    pub outcome: VerificationOutcome,

    // [VANTA-WHY] Optional FailureClass (see enum explanation above).
    // Rule: TestEvidence.outcome == PASS / VerifiedWithinBounds → failure_class: None.
    // Rule: Any FAIL/non-verified outcome → failure_class must be Some(_)
    // after Diagnostician runs (may be None immediately after execution for
    // records that haven't been through triage yet).
    // Option expresses the MAY BE ABSENT semantics without polluting the enum.
    pub failure_class: Option<FailureClass>,

    // [VANTA-WHY] The Diagnostician's structured payload is what lets repair
    // mode feed the Creator the actual machine-readable reason for a failure
    // (ruff/mypy details, import/collect output, quarantine action, etc.),
    // rather than collapsing everything to a class label.
    pub structured_payload: Option<serde_json::Value>,

    // [VANTA-WHY] Declared TestIntent (BugReproduction / FeatureAcceptance /
    // BehaviorPreservation). ABSOLUTELY REQUIRED for BATE Phase 1. You cannot
    // judge a test on original code without knowing what it was INTENDED to do.
    // Spec §5 Test Intents: "A test cannot be judged against the original code
    // without knowing what it is for."
    pub intent: TestIntent,

    // [VANTA-WHY] Spec §8 Evidence provenance: WHO wrote this test?
    // Critical for trust attribution. Example: a Counterexample from source
    // FormalVerifier has different review consequences than a Logic failure
    // from source AttackerGenerated (which may be a wrong test, not a wrong patch).
    pub evidence_source: EvidenceSource,
}

// ============================================================================
// VerificationPolicy — how the Gatekeeper treats non-binary verifier outcomes
// ============================================================================
//
// Spec §6 Verification policy table. Because VerificationOutcome has 6 values
// (not pass/fail), "all pass" is NOT a decision rule. Each repo/path sets a
// policy. V1 default = BestEffort (ordinary application code).
//
// | Outcome            | Required            | BestEffort          |
// |--------------------|---------------------|---------------------|
// | VerifiedWithinBounds| Continue           | Continue            |
// | Counterexample     | Reject              | Reject              |
// | Inconclusive       | Block / human review| Report + continue   |
// | Unsupported        | Block / human review| Report + continue   |
// | Timeout            | Retry, then block   | Retry, then continue|
// | ToolFailure        | Infrastructure retry| Infrastructure retry|
// ============================================================================
#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq, Eq)]
pub enum VerificationPolicy {
    // [VANTA-WHY] Required = payment/auth/critical paths. Any non-verified
    // outcome blocks approval. Counterexample always rejects.
    Required,
    // [VANTA-WHY] BestEffort = V1 default for ordinary application code.
    // Inconclusive/Unsupported are reported but do not block. Counterexample
    // still rejects (hard rule regardless of policy).
    BestEffort,
    // [VANTA-WHY] Disabled = unsupported languages or no verifier backend.
    // The verification track is skipped entirely; only execution evidence
    // feeds the Gatekeeper.
    Disabled,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        VerificationPolicy::BestEffort
    }
}

// ============================================================================
// AttemptTransition — single EIR/ECR transition between two patch attempts
// ============================================================================
//
// Spec §8: "EIR and ECR describe transitions between patch attempts."
// Comparing attempt N vs N+1 on the SAME fixed shared test set:
//   - fail -> pass = ECR (Error Correction Rate: the new patch fixed a failure)
//   - pass -> fail = EIR (Error Introduction Rate: the new patch broke something)
//   - unchanged = no signal
//
// [VANTA-WHY] This is the raw material for the Governor. V1 uses Fixed-N
// (stop after N attempts) but logs every transition so M4's live EIR/ECR
// governor can be computed from the same data without rerunning.
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptTransition {
    pub from_attempt: u32,
    pub to_attempt: u32,
    // [VANTA-WHY] Identity of the QUESTION SET this row was compared on: a
    // fingerprint of the shared set as it stood on the previous attempt. Every
    // row from one comparison shares it. Because the shared set grows as the
    // Attacker adds tests, this id changes along an attempt chain — which is
    // exactly the non-stationarity the S^(0)-restricted analysis controls for,
    // and it makes that visible per row instead of only in aggregate.
    pub shared_set_id: String,
    // [VANTA-WHY] Identity is the normalised body hash, not the name. It is the
    // row's `test_id`: the stable join key across attempts. The name is kept for
    // readability only and is never used to match attempts.
    pub test_id: String,
    pub test_name: String,
    pub from_result: TestResult,
    pub to_result: TestResult,
    // [VANTA-WHY] Spec §8 requires a failure_class on every transition row. BOTH
    // sides are kept: EIR is judged on the class of the NEW failure, ECR on the
    // class of the OLD one, and a fail->fail transition has two. Storing only one
    // would make any later re-analysis (S^(0)-restricted, oracle-only) impossible
    // to compute from the log and force a full re-run.
    pub from_failure_class: Option<FailureClass>,
    pub to_failure_class: Option<FailureClass>,
    // [VANTA-WHY] true if both from and to results were observed (shared set
    // ran on both attempts). false = censored (one attempt didn't complete the
    // shared set), excluded from EIR/ECR estimation. The persisted row exposes
    // this as `censored` (the inverse).
    pub complete: bool,
    // [VANTA-WHY] Membership of S^(0) — the shared set as it stood on the first
    // attempt, before any Attacker test was folded in. The paper's restricted
    // analysis recomputes rates on S^(0) alone to control for the growing
    // question set; that subset cannot be reconstructed afterwards unless it is
    // marked per row here.
    pub in_s0: bool,
    // [VANTA-WHY] Provenance: who wrote the test and what it was meant to prove.
    // Carried onto every row so M5 can run the robustness check that recomputes
    // EIR/ECR on oracle tests only, excluding everything the Attacker generated.
    pub evidence_source: EvidenceSource,
    pub test_intent: TestIntent,
}

impl AttemptTransition {
    /// Whether this transition is allowed to move EIR/ECR.
    ///
    /// [VANTA-WHY] Only LOGIC-class failures count. A fail->pass transition only
    /// means the patch corrected something if the failure being corrected was a
    /// logic failure — clearing a harness blip is not a correction. A
    /// pass->fail transition only means the patch introduced an error if the new
    /// failure is a logic failure — an import error the environment caused is not
    /// an introduction. Unchanged results have nothing to attribute, so they stay
    /// in the denominator to keep the rate comparable with earlier runs.
    pub fn counts_toward_metrics(&self) -> bool {
        counts_toward_metrics(
            &self.from_result,
            &self.to_result,
            &self.from_failure_class,
            &self.to_failure_class,
            &self.evidence_source,
        )
    }
}

// ============================================================================
// EcrEirSnapshot — counted ECR/EIR over a set of complete transitions
// ============================================================================
//
// [VANTA-WHY] The rates are CONDITIONAL on the state at N, not a share of every
// observed row:
//
//   ECR = rows Failed at N that Pass at N+1  ÷  rows Failed at N
//   EIR = rows Passed at N that Fail at N+1  ÷  rows Passed at N
//
// The per-row alternative — divide by every countable row — answers a different
// and much weaker question, "what share of all observed tests changed in a good
// way", and its denominator is the SIZE OF THE SUITE. Session 4d55123d fixed one
// native regression among five native tests, so dividing by every row scores
// ECR(Native) = 1/5; a BugsInPy repo with 800 native tests would score ~0.1%
// for the one fix that mattered. That measures the repository, not the repair,
// and makes two repos incomparable. Conditional gives 1/1 in both.
//
// `complete` is kept because it describes the run's overall CHURN — how much of
// the observed suite moved at all. It is deliberately NOT either rate's
// denominator.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EcrEirSnapshot {
    pub ecr_count: usize, // rows Failed at N that Pass at N+1
    pub eir_count: usize, // rows Passed at N that Fail at N+1
    // [VANTA-WHY] The two CONDITIONAL denominators. A row belongs to at most
    // one of them — its state at N is Failed, Passed, or (for an Error) neither,
    // which moves no rate at all — so neither rate may be read off the other's
    // denominator.
    #[serde(default)]
    pub ecr_denominator: usize, // rows Failed at N
    #[serde(default)]
    pub eir_denominator: usize, // rows Passed at N
    pub complete: usize, // countable rows observed (churn; not a rate denominator)
    // [VANTA-WHY] Complete transitions dropped because a failure involved was
    // not Logic-class. Reported rather than silently discarded so a run that
    // excluded most of its evidence is visible instead of looking precise.
    #[serde(default)]
    pub excluded_non_logic: usize,
    // [VANTA-WHY] The same counts, split by where the evidence came from, and
    // additive to the totals above. A pooled ECR of 1/6 says nothing about
    // WHICH evidence moved: session 4d55123d's only correction was a native
    // test, and reading the number as an Attacker-test result would credit the
    // model for something the repository's own suite proved. `#[serde(default)]`
    // so snapshots written before the breakdown existed still load.
    #[serde(default)]
    pub attacker_generated: SourceBreakdown,
    #[serde(default)]
    pub native: SourceBreakdown,
    #[serde(default)]
    pub static_findings: SourceBreakdown,
}

/// ECR/EIR counts for ONE evidence source. See `EcrEirSnapshot`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SourceBreakdown {
    pub ecr_count: usize,
    pub eir_count: usize,
    #[serde(default)]
    pub ecr_denominator: usize,
    #[serde(default)]
    pub eir_denominator: usize,
    pub complete: usize,
    #[serde(default)]
    pub excluded_non_logic: usize,
}

impl SourceBreakdown {
    /// Conditional ECR for this source, e.g. `1/1 (100.00%)` or `n/a`.
    pub fn ecr_text(&self) -> String {
        EcrEirSnapshot::render_rate(self.ecr_count, self.ecr_denominator)
    }

    /// Conditional EIR for this source, e.g. `0/4 (0.00%)` or `n/a`.
    pub fn eir_text(&self) -> String {
        EcrEirSnapshot::render_rate(self.eir_count, self.eir_denominator)
    }
}

impl EcrEirSnapshot {
    pub fn ecr_rate(&self) -> f64 {
        if self.ecr_denominator == 0 {
            0.0
        } else {
            self.ecr_count as f64 / self.ecr_denominator as f64
        }
    }
    pub fn eir_rate(&self) -> f64 {
        if self.eir_denominator == 0 {
            0.0
        } else {
            self.eir_count as f64 / self.eir_denominator as f64
        }
    }

    /// Record one countable row in the totals AND in its source's breakdown.
    ///
    /// The row's state at N decides which denominator it joins — Failed joins
    /// ECR's, Passed joins EIR's, an Error joins neither — and the (from, to)
    /// pair decides whether it is a numerator. Both rates therefore stay
    /// conditional on what was broken / passing at N, whatever the size of the
    /// suite that produced the rows.
    pub fn record(&mut self, source: &EvidenceSource, from: &TestResult, to: &TestResult) {
        self.complete += 1;
        let breakdown = match source {
            EvidenceSource::Native => &mut self.native,
            EvidenceSource::Static => &mut self.static_findings,
            _ => &mut self.attacker_generated,
        };
        breakdown.complete += 1;
        match from {
            TestResult::Failed => {
                self.ecr_denominator += 1;
                breakdown.ecr_denominator += 1;
            }
            TestResult::Passed => {
                self.eir_denominator += 1;
                breakdown.eir_denominator += 1;
            }
            TestResult::Error => {}
        }
        match (from, to) {
            (TestResult::Failed, TestResult::Passed) => {
                self.ecr_count += 1;
                breakdown.ecr_count += 1;
            }
            (TestResult::Passed, TestResult::Failed) => {
                self.eir_count += 1;
                breakdown.eir_count += 1;
            }
            _ => {}
        }
    }

    /// Record one excluded (non-countable-class) row in the totals and in its
    /// source's breakdown.
    pub fn record_excluded(&mut self, source: &EvidenceSource) {
        self.excluded_non_logic += 1;
        match source {
            EvidenceSource::Native => self.native.excluded_non_logic += 1,
            EvidenceSource::Static => self.static_findings.excluded_non_logic += 1,
            _ => self.attacker_generated.excluded_non_logic += 1,
        }
    }

    /// `count/denominator (pct%)`, or `n/a` when the denominator is zero.
    ///
    /// [VANTA-WHY] A zero denominator must NEVER render as `0.00%`. "No
    /// transitions were countable" and "every countable transition was
    /// uncorrected" are opposite claims about a run, and a rate of 0.00% states
    /// the second one. `n/a` states the first.
    ///
    /// The denominator is the CONDITIONAL one (`ecr_denominator` /
    /// `eir_denominator`), never `complete`: dividing by every observed row
    /// would let the size of the repo's test suite decide the rate.
    pub fn render_rate(count: usize, denominator: usize) -> String {
        if denominator == 0 {
            "n/a".to_string()
        } else {
            format!(
                "{count}/{denominator} ({:.2}%)",
                count as f64 / denominator as f64 * 100.0
            )
        }
    }

    /// Combined ECR, e.g. `1/1 (100.00%)` or `n/a`.
    pub fn ecr_text(&self) -> String {
        Self::render_rate(self.ecr_count, self.ecr_denominator)
    }

    /// Combined EIR, e.g. `0/4 (0.00%)` or `n/a`.
    pub fn eir_text(&self) -> String {
        Self::render_rate(self.eir_count, self.eir_denominator)
    }

    /// ECR and EIR for one source, e.g. `ECR=1/1 (100.00%) EIR=0/4 (0.00%)`.
    fn source_text(breakdown: &SourceBreakdown) -> String {
        format!("ECR={} EIR={}", breakdown.ecr_text(), breakdown.eir_text())
    }

    /// The same metric per evidence source, so a pooled number can be read
    /// apart: `AttackerGenerated[ECR=n/a EIR=n/a] Native[ECR=1/1 (100.00%) …] …`
    pub fn per_source_text(&self) -> String {
        format!(
            "AttackerGenerated[{}] Native[{}] Static[{}]",
            Self::source_text(&self.attacker_generated),
            Self::source_text(&self.native),
            Self::source_text(&self.static_findings)
        )
    }
}

// ============================================================================
// GovernorDecision — why the repair loop stopped (or continued)
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GovernorDecision {
    // [VANTA-WHY] All shared-set tests passed on this attempt. Candidate for PR.
    AllTestsPassed,
    // [VANTA-WHY] A REVIEWED candidate demonstrably failed — a Logic failure, or
    // a repo-native test that passed before the patch and fails after it. In
    // review mode the candidate is fixed, so there is nothing to repair: the
    // answer is no. This is deliberately NOT MaxAttemptsReached. "Found a
    // regression" and "gave up after N tries" are different claims about the
    // model, and the PR and the paper both read this field to tell them apart.
    Rejected,
    // [VANTA-WHY] Reached the attempt budget without a clean pass. The number is
    // the run's own `--max-attempts` (the driver logs `attempts_run`); it is NOT
    // a fixed 4.
    MaxAttemptsReached,
    // [VANTA-WHY] §14.3. Review mode spent its whole attack depth on failures
    // that no corroborating rule covers: the candidate is not approved, and it
    // is not rejected either, because the only evidence against it might be a
    // wrong test. Reachable ONLY in review mode with `VANTA_CORROBORATE=on`, and
    // only once the budget is exhausted — a human has to look.
    //
    // Deliberately NOT an approval: the Gatekeeper still approves only on
    // `AllTestsPassed`, so this variant cannot inflate an approval count, and
    // the harness scores it as its own outcome (`needs_review_on_good` /
    // `needs_review_on_bad`) rather than as an approval, a rejection or an
    // unresolved run. It has its own exit code (3) so a caller can tell "a
    // human should look" from "no".
    NeedsReview,
    // [VANTA-WHY] M3 item 7 — the judge arms' positive verdict: one model call
    // read the issue, the SAME selected context the B3 arm would have seen, and
    // the candidate diff (plus the native suite's result under `judge+suite`),
    // and answered that the patch resolves the issue. Deliberately NOT
    // `AllTestsPassed`: that token claims a test ran and passed, and under the
    // judge no generated test ran at all — reading it as one would make a
    // verdict with no execution behind it indistinguishable from one with the
    // whole adversarial pipeline behind it, in the session log, in `metrics.py`
    // and in the approval-invariant checks. The harness maps it onto the same
    // TA/FA label `AllTestsPassed` gets, so a judge row still pairs with a Vanta
    // row on the same candidate.
    JudgeApproved,
    // [VANTA-WHY] M3 item 7 — the judge decided the candidate does NOT resolve
    // the issue. Same argument in the other direction: `Rejected` claims a test
    // observed a failure, and no test did here. Mapped onto the TR/FR labels for
    // pairing, recorded under its own token so the log says which mechanism
    // reached the verdict.
    JudgeRejected,
    // [VANTA-WHY] M3 item 7 — the judge answered with something unusable (an
    // unknown `verdict`, or JSON that would not parse) on BOTH the call and its
    // one retry, so there is no verdict at all. Its own token because it is none
    // of the neighbouring outcomes: not an approval and not a rejection (the
    // model did not decide), and not `InfraFailure` (the machine answered
    // perfectly well — it answered with text nobody can read as a verdict). It
    // gets its own exit code (4) and its own harness bucket (`judge_failed`),
    // and is never folded into `NeedsReview`, `Rejected` or `InfraFailure`.
    JudgeFailed,
    // [VANTA-WHY] An LLM call failed for every attempt the retry loop was
    // allowed. This says NOTHING about the candidate: the model never produced an
    // answer, so there is no evidence for or against the patch. It exists so the
    // run can end as a RECORDED outcome — final_decision, session summary,
    // evidence and transitions all on disk — instead of a crash with no verdict.
    // metrics.py must never score it as approval, rejection or unresolved.
    InfraFailure,
    // [VANTA-WHY] Patch failed to apply (changed == false). Cannot continue —
    // the Creator produced no usable change.
    PatchApplicationFailed,
    // [VANTA-WHY] Every patched test in the attempt classified as
    // Infrastructure. A real infrastructure fault hits one or two tests; a
    // 100% rate is either the setup or a transient correlated outage (six VMs
    // spawned in the same millisecond can all fail together). The discriminator
    // is REPRODUCIBILITY: retry once, and only if it collapses again is this a
    // broken harness rather than a blip. Terminal — nothing is left to retry.
    HarnessFailure,
    // [VANTA-WHY] Loop should continue (used internally; never a final decision).
    Continue,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity of a blocking static finding is rule + file + message. The
    /// line number is NOT part of it.
    #[test]
    fn a_static_finding_identity_ignores_the_line_it_sits_on() {
        let a = static_finding_identity("F821", "calc.py", "Undefined name `x`");
        let b = static_finding_identity("F821", "calc.py", "Undefined name `x`");
        assert_eq!(a, b, "the same finding must have one identity");

        // The identity does not even take a line, so a finding that only moved
        // down the file keeps it. A message that itself carries a colon (mypy
        // writes "path:line: error: ..." style text) must survive intact — the
        // identity is never split on one.
        let with_colon = static_finding_identity("index", "mod.py", "Incompatible type: int");
        assert_eq!(with_colon, "index|mod.py|Incompatible type: int");
    }

    /// A message that is re-wrapped by the analyzer is the SAME finding.
    #[test]
    fn a_static_finding_identity_normalises_whitespace_in_the_message() {
        let wrapped = static_finding_identity(
            "index",
            "mod.py",
            "Item of \"list\" has no\n   attribute \"shape\"",
        );
        let flat =
            static_finding_identity("index", "mod.py", "Item of \"list\" has no attribute \"shape\"");
        assert_eq!(
            wrapped, flat,
            "re-wrapping a diagnostic must not fork the identity"
        );
        assert_eq!(
            flat,
            r#"index|mod.py|Item of "list" has no attribute "shape""#,
            "the identity is readable: rule | file | normalised message"
        );
    }

    /// Different rule, file, or message -> different identity.
    #[test]
    fn a_static_finding_identity_separates_rule_file_and_message() {
        let base = static_finding_identity("F821", "calc.py", "Undefined name `x`");
        assert_ne!(base, static_finding_identity("F401", "calc.py", "Undefined name `x`"));
        assert_ne!(base, static_finding_identity("F821", "other.py", "Undefined name `x`"));
        assert_ne!(base, static_finding_identity("F821", "calc.py", "Undefined name `y`"));
    }

    /// A Static-class failure counts ONLY on a row whose own source is Static.
    ///
    /// [VANTA-WHY] This is the whole reason the gate takes a source: the static
    /// class is uncountable for Attacker evidence (an `import-not-found` blames
    /// the environment) but countable for a blocking finding that was already
    /// differenced against the baseline.
    #[test]
    fn a_static_class_failure_counts_only_for_static_source_rows() {
        let passed = TestResult::Passed;
        let failed = TestResult::Failed;
        let static_class = Some(FailureClass::Static);

        assert!(
            counts_toward_metrics(
                &failed,
                &passed,
                &static_class,
                &None,
                &EvidenceSource::Static
            ),
            "a blocking static error fixed on the next attempt is a correction"
        );
        assert!(
            counts_toward_metrics(
                &passed,
                &failed,
                &None,
                &static_class,
                &EvidenceSource::Static
            ),
            "a blocking static error introduced on the next attempt is an introduction"
        );
        assert!(
            !counts_toward_metrics(
                &failed,
                &passed,
                &static_class,
                &None,
                &EvidenceSource::AttackerGenerated
            ),
            "an Attacker test that merely classified Static must stay excluded"
        );
        assert!(
            !counts_toward_metrics(
                &failed,
                &passed,
                &static_class,
                &None,
                &EvidenceSource::Native
            ),
            "the allowance is for Static-source rows only, not for every source"
        );
    }

    /// Native rows carry Logic on their failing side, so a repo-native
    /// regression and its repair are countable under the ordinary rule.
    #[test]
    fn a_native_regression_counts_under_the_ordinary_logic_rule() {
        assert!(counts_toward_metrics(
            &TestResult::Failed,
            &TestResult::Passed,
            &Some(FailureClass::Logic),
            &None,
            &EvidenceSource::Native
        ));
        assert!(counts_toward_metrics(
            &TestResult::Passed,
            &TestResult::Failed,
            &None,
            &Some(FailureClass::Logic),
            &EvidenceSource::Native
        ));
    }

    /// A zero denominator renders `n/a`, never `0.00%`.
    #[test]
    fn a_zero_denominator_renders_not_applicable_never_a_zero_percent() {
        assert_eq!(EcrEirSnapshot::render_rate(0, 0), "n/a");
        assert_eq!(EcrEirSnapshot::render_rate(0, 3), "0/3 (0.00%)");
        assert_eq!(EcrEirSnapshot::render_rate(1, 2), "1/2 (50.00%)");

        let empty = EcrEirSnapshot::default();
        assert_eq!(empty.ecr_text(), "n/a");
        assert_eq!(empty.eir_text(), "n/a");
        assert!(
            !empty.per_source_text().contains("0.00%"),
            "an empty snapshot must not state a rate: {}",
            empty.per_source_text()
        );
    }

    /// The per-source breakdown is additive to the totals, and each rate's
    /// denominator is the row's state at N — not the row count.
    #[test]
    fn the_source_breakdown_adds_up_to_the_totals() {
        let mut snap = EcrEirSnapshot::default();
        snap.record(
            &EvidenceSource::AttackerGenerated,
            &TestResult::Passed,
            &TestResult::Passed,
        );
        snap.record(&EvidenceSource::Native, &TestResult::Failed, &TestResult::Passed);
        snap.record(&EvidenceSource::Static, &TestResult::Passed, &TestResult::Failed);
        snap.record_excluded(&EvidenceSource::AttackerGenerated);

        assert_eq!(snap.complete, 3);
        assert_eq!(snap.ecr_count, 1);
        assert_eq!(snap.eir_count, 1);
        assert_eq!(snap.excluded_non_logic, 1);
        assert_eq!(
            snap.ecr_denominator, 1,
            "only the Native row was FAILING at N, and it is the only ECR denominator"
        );
        assert_eq!(
            snap.eir_denominator, 2,
            "the Attacker and Static rows were PASSING at N, so they form EIR's denominator"
        );
        assert_eq!(snap.attacker_generated.complete, 1);
        assert_eq!(snap.native.ecr_count, 1);
        assert_eq!(snap.native.complete, 1);
        assert_eq!(snap.native.ecr_denominator, 1);
        assert_eq!(
            snap.native.eir_denominator, 0,
            "nothing was passing at N under the Native source, so EIR(Native) is n/a"
        );
        assert_eq!(snap.static_findings.eir_count, 1);
        assert_eq!(snap.static_findings.complete, 1);
        assert_eq!(snap.static_findings.eir_denominator, 1);
        assert_eq!(
            snap.native.complete + snap.static_findings.complete + snap.attacker_generated.complete,
            snap.complete
        );
    }

    /// A row whose `from` side is an Error belongs to NEITHER conditional
    /// denominator: it was not passing at N and it was not failing at N, so it
    /// can neither be fixed nor broken.
    #[test]
    fn an_error_at_n_joins_neither_conditional_denominator() {
        let mut snap = EcrEirSnapshot::default();
        snap.record(&EvidenceSource::Native, &TestResult::Error, &TestResult::Passed);

        assert_eq!(snap.complete, 1, "it is still a countable, observed row");
        assert_eq!(snap.ecr_count, 0);
        assert_eq!(snap.eir_count, 0);
        assert_eq!(snap.ecr_denominator, 0);
        assert_eq!(snap.eir_denominator, 0);
        assert_eq!(snap.ecr_text(), "n/a");
        assert_eq!(snap.eir_text(), "n/a");
    }

    /// The suite size must not move ECR: the same single fix scores `1/1`
    /// whether four other tests kept passing or eight hundred did.
    #[test]
    fn a_fix_scores_the_same_whatever_else_the_suite_contains() {
        let snapshot = |bystanders: usize| {
            let mut snap = EcrEirSnapshot::default();
            snap.record(&EvidenceSource::Native, &TestResult::Failed, &TestResult::Passed);
            for _ in 0..bystanders {
                snap.record(&EvidenceSource::Native, &TestResult::Passed, &TestResult::Passed);
            }
            snap
        };

        assert_eq!(snapshot(4).native.ecr_text(), "1/1 (100.00%)");
        assert_eq!(snapshot(800).native.ecr_text(), "1/1 (100.00%)");
        // ... while churn keeps growing, which is what `complete` is for.
        assert_eq!(snapshot(4).native.complete, 5);
        assert_eq!(snapshot(800).native.complete, 801);
    }
}
