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
/// - an unchanged result carries no attribution, so it stays COUNTABLE — it
///   remains in the denominator to keep the rate comparable across runs.
///
/// The other half of the rule — that the row must be a genuine patch
/// transition, both ends observed and different — is
/// `TransitionRow::is_countable_patch_transition`.
pub fn counts_toward_metrics(
    from_result: &TestResult,
    to_result: &TestResult,
    from_failure_class: &Option<FailureClass>,
    to_failure_class: &Option<FailureClass>,
) -> bool {
    match (from_result, to_result) {
        (TestResult::Failed, TestResult::Passed) => is_logic_failure(from_failure_class),
        (TestResult::Passed, TestResult::Failed) => is_logic_failure(to_failure_class),
        _ => true,
    }
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
        )
    }
}

// ============================================================================
// EcrEirSnapshot — counted ECR/EIR over a set of complete transitions
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EcrEirSnapshot {
    pub ecr_count: usize,   // fail -> pass
    pub eir_count: usize,   // pass -> fail
    pub complete: usize,    // number of complete transitions observed
    // [VANTA-WHY] Complete transitions dropped because a failure involved was
    // not Logic-class. Reported rather than silently discarded so a run that
    // excluded most of its evidence is visible instead of looking precise.
    #[serde(default)]
    pub excluded_non_logic: usize,
}

impl EcrEirSnapshot {
    pub fn ecr_rate(&self) -> f64 {
        if self.complete == 0 { 0.0 } else { self.ecr_count as f64 / self.complete as f64 }
    }
    pub fn eir_rate(&self) -> f64 {
        if self.complete == 0 { 0.0 } else { self.eir_count as f64 / self.complete as f64 }
    }
}

// ============================================================================
// GovernorDecision — why the repair loop stopped (or continued)
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GovernorDecision {
    // [VANTA-WHY] All shared-set tests passed on this attempt. Candidate for PR.
    AllTestsPassed,
    // [VANTA-WHY] Reached max attempts (Fixed-N = 4). Stopped without a clean pass.
    MaxAttemptsReached,
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
