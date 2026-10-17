// ============================================================================
// CHECKPOINT A — Ownership, Collections, Enums, Pattern Matching, Result
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY — every concept below is REUSED in M1-M5
// The exact line patterns you type here appear in Diagnostician, Governor,
// Attacker discard-rate counting, and the EIR/ECR tables for the thesis paper.
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | Vec<T> collections        → M2: storing 100+ Attacker-generated tests
//     |                           → M4: storing fixed shared test set for EIR/ECR
//     |                           → M1: storing Creator files_touched list
//  ---+---------------------------+-------------------------------------------
//   2 | &T borrowing (vs owned T) → Every fn parameter that reads a collection
//     |                           → Never pass Vec if &[T] will do (no clone!)
//     |                           → Saves 1000s of unnecessary copies in M2 runs
//  ---+---------------------------+-------------------------------------------
//   3 | match on enums (EXHAUST)  → Phase 2 Diagnostician 4-way triage
//     |                           → Gatekeeper 6-way VerificationPolicy table
//     |                           → Governor counting Logic vs Static vs Infr
//  ---+---------------------------+-------------------------------------------
//   4 | Option<T>                 → TestEvidence.failure_class is None for PASS
//     |                           → CreatorResponse.confidence optional later
//     |                           → LlmConfig.api_key: Some(key) / None (local)
//  ---+---------------------------+-------------------------------------------
//   5 | Result<T,E> + ? operator  → Every orchestration fn that can fail
//     |                           → HTTP errors, JSON parse errors, sandbox
//     |                           → ZERO unwraps allowed in non-test code paths
//  ---+---------------------------+-------------------------------------------
//   6 | main() -> Result<(), E>   → Errors bubble up as message, not panic trace
//     |                           → Benchmark harness (M5) prints Err & stops
//     |                           → Never leaves RUST_BACKTRACE in user face
//  ---+---------------------------+-------------------------------------------
//   7 | Counter variables + sum   → Governor EIR (break count) / ECR (fix count)
//     |                           → Attacker discard rate metric M2
//     |                           → Static failure rate / Logic failure split
//  ---+---------------------------+-------------------------------------------
//
// HOW TO USE THIS FILE: Compile + run. Read the comments top to bottom. Find
// each tagged [SYNTAX]/[VANTA-WHY]/[PITFALL]/[RECIPE]/[TRACE]. Then in a few
// days try re-writing triage_summary from blank file using only the footer
// recipes. That's when you know you've learned it.
// ============================================================================

// ----------------------------------------------------------------------------
// [SYNTAX] use vanta::core::*; = "bring all pub items from the vanta library's
// core module into local scope here". The `*` is a glob pattern = wildcard.
//
// [PITFALL] Glob `use ...::*` is FINE in standalone binaries like src/bin/*.rs
// because there's only one writer and one reader. Glob imports in a LIBRARY
// module (main src/*.rs, not src/bin/) are EVIL because they hide which types
// came from where. In library code, always import specific names:
//   use vanta::core::{VerificationOutcome, TestIntent};   <-- good in libs
//
// [VANTA-WHY] core/mod.rs in the library now has the REAL TestEvidence with
// the 5th field `evidence_source` (Spec §8 Evidence Provenance). The library
// crate root src/lib.rs re-exports core as pub. We use the library types
// here to guarantee checkpoint_a exercises the EXACT types M1+ code uses.
// Previously checkpoint_a had a local 4-field copy for pedagogical purposes;
// now refactored to the real shared 5-field version.
// ----------------------------------------------------------------------------
use vanta::core::*;

// ============================================================================
// triage_summary — PROCESS A COLLECTION → COUNT THINGS → ERROR IF EMPTY
// ============================================================================
//
// [RECIPE A1] (see footer for copy-paste version): Every time in Vanta you have
// "input = list of things, produce counts + per-item summary", this exact
// function shape is the template. It appears in:
//   - Diagnostician (M2): count Infrastructure / Static / Logic / Unknown
//   - Attacker reliability metric (M2): count scrapped vs accepted tests
//   - Governor (M4): count EIR transitions vs ECR transitions on shared set
//   - Paper results (M5): aggregate runs across model tiers CSV
// ============================================================================

// [SYNTAX] pub fn triage_summary(evidence: &[TestEvidence]) -> Result<(), String>
//
// Breakdown of every token:
//
//   pub fn                = public function (callable from other modules)
//   triage_summary        = name
//   evidence:             = parameter named `evidence`
//   &[TestEvidence]       = BORROWED SLICE of TestEvidence items.
//
//     [VANTA-WHY of &[T]]: We accept &[T] instead of Vec<T> for 3 reasons:
//       (1) BORROWED, not owned = caller retains ownership, no clone/copy needed
//       (2) SLICE = any contiguous memory. Caller can pass &vec, &array[..], or
//           a sub-slice like &evidence[0..5]. Works with all of them.
//       (3) Read-only signal = function body cannot accidentally add/remove
//           items from the caller's original Vec. (It'd need &mut [T] to mutate.)
//
//   -> Result<(), String>  = RETURN TYPE: Result with Ok variant holding ()
//     (unit = "nothing useful to return, function is called for side effects
//      of printing counts") and Err variant holding an owned String message.
//
//     [VANTA-WHY of Result]: Per Spec §3 + M0 exit test — zero panics in the
//     orchestration path. An empty evidence list could mean "Attacker produced
//     zero tests" or "sandbox died before Phase 2 started". We report that as
//     Err(String) and let the caller decide what to do (maybe retry, maybe
//     escalate). We never panic, because a panic mid-M5-benchmark corrupts the
//     8-hour experimental run and you lose a day of compute.
//
pub fn triage_summary(evidence: &[TestEvidence]) -> Result<(), String> {
    // ------------------------------------------------------------------------
    // Guard clause: empty input = early return with Err
    // ------------------------------------------------------------------------
    // [SYNTAX] .is_empty() on a slice = true iff length == 0.
    // [SYNTAX] return Err(...) = return EARLY from the function with the Err
    // variant, skipping all remaining code in the function body.
    // [SYNTAX] "str literal".to_string() = convert &'static str (compile-time
    // constant baked into binary) into owned String (heap-allocated, required
    // for the Err variant we return).
    //
    // [TRACE] What happens when someone calls triage_summary(&[]):
    //   1. evidence is an empty slice of length 0
    //   2. .is_empty() evaluates to true
    //   3. return Err(owned_string) exits the function right here
    //   4. The ? in main() catches the Err and returns it from main
    //   5. Rust runtime prints the Err string to stderr with NO backtrace
    //
    // [PITFALL] Beginners write: if evidence.len() == 0 { ... }. Works. But
    // .is_empty() is the idiomatic name (appears on Vec, String, HashSet,
    // HashMap, BTreeMap, every collection in std). Readers see .is_empty() and
    // instantly know intent. len() == 0 works but is 2 tokens longer and
    // requires a mental comparison parse.
    if evidence.is_empty() {
        return Err("CHECKPOINT_A_ERROR: No evidence to triage! Caller provided an empty list.".to_string());
    }

    // ------------------------------------------------------------------------
    // Counter variables = Recipe A1 (counting inside a loop)
    // ------------------------------------------------------------------------
    // [SYNTAX] `let mut` = declare a MUTABLE variable. Rust variables are
    // IMMUTABLE by default (unlike literally every other language you know).
    // Without `mut` the `verified_count += 1;` line below would fail with:
    //   "error[E0384]: cannot assign twice to immutable variable `verified_count`"
    //
    // [VANTA-WHY] This exact counter pattern appears in Governor §8 (Spec):
    //   EIR = number of (pass -> fail) transitions between attempt N and N+1
    //   ECR = number of (fail -> pass) transitions
    // You implement that by iterating over shared test set results and
    // incrementing EIR or ECR counters per-test based on the delta.
    let mut verified_count: u32      = 0; // VerifiedWithinBounds hits
    let mut counterexample_count: u32 = 0; // Counterexample (hard rejection)
    let mut undecided_count: u32     = 0; // Inconclusive | Unsupported | Timeout
    let mut tool_fail_count: u32     = 0; // ToolFailure (backend itself died)

    // ------------------------------------------------------------------------
    // Per-test triage loop — match exhaustively
    // ------------------------------------------------------------------------
    // [SYNTAX] for ev in evidence — iterates over each element of the slice
    // in order. `ev` here is a &TestEvidence (shared reference, not moved).
    //
    // [VANTA-WHY] Attacker writes N tests in M1. Phase 2 (Diagnostician) loops
    // over exactly that list. BATE Phase 1 loops over the same list. Verifier
    // backends loop over a property list. Everywhere you process a batch = for.
    for ev in evidence {

        // [SYNTAX] match &ev.outcome { ... } = pattern match BEHIND A REFERENCE.
        // We use `&` before ev.outcome because ev itself is a reference (we
        // got it from iterating a &[T] slice). Without the & we'd try to move
        // the VerificationOutcome out of the shared reference = compile error.
        //
        // [SYNTAX] => "arm body" = what to run when the pattern matches.
        //
        // [SYNTAX] | is OR in patterns: Inconclusive | Unsupported | Timeout
        // matches any ONE of those three variants and runs the same arm body.
        // This is how you GROUP cases together.
        //
        // [VANTA-WHY] This match on VerificationOutcome is the shape of the
        // Spec §6 Verification Policy table. Gatekeeper (M3) is literally:
        //   match (policy, outcome) {
        //       (Required, Inconclusive) => block_and_human_review(),
        //       (BestEffort, Inconclusive) => report_and_continue(),
        //       ...
        //   }
        // The Rust compiler FORCES you to write every arm. If you delete one,
        // it says "non-exhaustive patterns" and refuses to compile. That is
        // literally how you guarantee the Gatekeeper can't forget a case.
        //
        // [PITFALL] Beginners add `_ => ...` (wildcard = "match anything else")
        // as a catch-all. Fine sometimes, BUT IF YOU ADD A NEW OUTCOME VARIANT
        // SIX MONTHS LATER, the wildcard silently matches it and you have a
        // logic bug. Avoid wildcards on domain enums. Let the compiler catch
        // the missing arms for you. That is the entire value proposition.
        match &ev.outcome {
            VerificationOutcome::VerifiedWithinBounds => {
                verified_count += 1;
                println!("TEST {:<35} | OUTCOME: VerifiedWithinBounds  | INTENT: {:<20} | SOURCE: {:<22} | FC: {:?}",
                    ev.test_name, format_intent(&ev.intent), format_source(&ev.evidence_source), ev.failure_class);
            }
            VerificationOutcome::Counterexample => {
                counterexample_count += 1;
                println!("TEST {:<35} | OUTCOME: Counterexample         | INTENT: {:<20} | SOURCE: {:<22} | FC: {:?}",
                    ev.test_name, format_intent(&ev.intent), format_source(&ev.evidence_source), ev.failure_class);
            }
            VerificationOutcome::Inconclusive | VerificationOutcome::Unsupported | VerificationOutcome::Timeout => {
                undecided_count += 1;
                println!("TEST {:<35} | OUTCOME: UndecidedGroup          | INTENT: {:<20} | SOURCE: {:<22} | FC: {:?}",
                    ev.test_name, format_intent(&ev.intent), format_source(&ev.evidence_source), ev.failure_class);
            }
            VerificationOutcome::ToolFailure => {
                tool_fail_count += 1;
                println!("TEST {:<35} | OUTCOME: ToolFailure            | INTENT: {:<20} | SOURCE: {:<22} | FC: {:?}",
                    ev.test_name, format_intent(&ev.intent), format_source(&ev.evidence_source), ev.failure_class);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Print the final count summary table
    // ------------------------------------------------------------------------
    // [SYNTAX] println! with {} = Display format, {:?} = Debug format,
    // {:<N} = left-align within N-column width. Purely cosmetic for output.
    println!("\n=========================================================================");
    println!("TRIAGE COUNT SUMMARY ({} total evidence items)", evidence.len());
    println!("=========================================================================");
    println!("  VerifiedWithinBounds : {:>4}  (proof within checked limits)", verified_count);
    println!("  Counterexample       : {:>4}  (HARD REJECT — concrete input violates property)", counterexample_count);
    println!("  Undecided group      : {:>4}  (Inconclusive | Unsupported | Timeout)", undecided_count);
    println!("  ToolFailure          : {:>4}  (verifier backend itself misbehaved)", tool_fail_count);
    println!("=========================================================================\n");

    // ------------------------------------------------------------------------
    // Happy path return — Ok(())
    // ------------------------------------------------------------------------
    // [SYNTAX] Ok(()) = wrap unit in Ok variant. Functions with side effects
    // and no return value always have -> Result<(), E> and return Ok(()) here.
    //
    // [PITFALL] Forgetting the semicolon on the FINAL expression in a function
    // means it becomes the implicit return. This is correct for Ok(())! If you
    // added a semicolon after Ok(()) it becomes a statement (value thrown away)
    // and the function would end with implicit () instead of Result, causing a
    // type mismatch: "expected enum `Result`, found `()`". LAST EXPR = RETURN.
    Ok(())
}

// ----------------------------------------------------------------------------
// format_intent — tiny helper that turns TestIntent enum into displayable &str
// ----------------------------------------------------------------------------
// [SYNTAX] fn format_intent(intent: &TestIntent) -> &'static str
//   &'static str = string literal baked into the binary, lives for entire
//   program duration. Perfect for small fixed label mapping.
//
// [RECIPE] Every domain enum gets a -> &'static str display helper or a proper
// Display trait impl. In M2 we'll write format_failure_class, in M3 we'll
// write format_verification_policy. You'll call these helpers in every log line.
fn format_intent(intent: &TestIntent) -> &'static str {
    // [SYNTAX] match with each arm evaluating to a value. This entire match
    // expression IS the return value of the function (last expression rule).
    match intent {
        TestIntent::BugReproduction       => "BugReproduction",
        TestIntent::FeatureAcceptance     => "FeatureAcceptance",
        TestIntent::BehaviorPreservation  => "BehaviorPreservation",
    }
}

// ----------------------------------------------------------------------------
// format_source — format EvidenceSource enum for display output
// ----------------------------------------------------------------------------
// [SYNTAX] Same pattern as format_intent. Reusable Recipe for every enum.
// [VANTA-WHY] This 5th field was added in step 4 when we promoted TestEvidence
// to the library type. Spec §8 says you MUST separate who wrote a test from
// what ran it. In the output log lines this field lets readers distinguish:
// "Counterexample, FormalVerifier" (high trust hard reject) from:
// "Counterexample, AttackerGenerated" (possibly bad test needs review).
fn format_source(source: &EvidenceSource) -> &'static str {
    match source {
        EvidenceSource::ExistingRepositoryTest => "ExistingRepositoryTest",
        EvidenceSource::BenchmarkOracle        => "BenchmarkOracle",
        EvidenceSource::AttackerGenerated      => "AttackerGenerated",
        EvidenceSource::HumanProvided          => "HumanProvided",
        EvidenceSource::FormalVerifier         => "FormalVerifier",
        EvidenceSource::IssueDescription       => "IssueDescription",
    }
}

// ============================================================================
// main() — ENTRY POINT FOR CHECKPOINT_A BINARY
// ============================================================================
//
// [SYNTAX] fn main() -> Result<(), String> — since Rust Edition 2018 main can
// return a Result. If it returns Ok(()) the process exits 0. If it returns
// Err(msg), the runtime prints msg to stderr AND exits with code 1 (nonzero).
//
// [VANTA-WHY] This is EXACTLY how the Vanta CLI will handle errors in M3.
// If vanta run --repo foo barfs on a missing VANTA_LLM_API_KEY, the user sees:
//   Error: VANTA_LLM_API_KEY not set and no default available
//   <exits 1>
// Not a 30-line RUST_BACKTRACE that scares non-Rust users and looks broken.
// ============================================================================
fn main() -> Result<(), String> {

    // ------------------------------------------------------------------------
    // Build the evidence Vec: push 7 items covering every possible outcome
    // ------------------------------------------------------------------------
    // [SYNTAX] let mut evidence: Vec<TestEvidence> = Vec::new();
    //   Vec::new() = creates an EMPTY growable array on the heap. Initial
    //   capacity is 0; first push allocates, later pushes amortize reallocation.
    //   You can also write vec![] or vec![item1, item2] for literals. We use
    //   .push() one by one here so you can see each instantiation pattern.
    //
    // [VANTA-WHY] This is literally the data shape the Attacker writes.
    // In M1 the Attacker agent returns Vec<GeneratedTest>. In M2 that becomes
    // Vec<TestEvidence> after execution. The Governor consumes Vec<TestEvidence>
    // to compute EIR/ECR. Every phase of Vanta ends with a Vec of records.
    let mut evidence: Vec<TestEvidence> = Vec::new();

    // [SYNTAX] TestEvidence { field: value, ... } = struct literal syntax.
    // Every field MUST be provided (or derive Default + ..Default::default()).
    // .to_string() on str literals = create owned heap String for the struct.
    // None = Option's "absent" variant. Some(Enum::Variant) = Option's "present".
    // evidence_source = NEW 5th field (library-only): provenance per Spec §8.
    evidence.push(TestEvidence {
        test_name:       "test_refund_never_exceeds_charge".to_string(),
        outcome:         VerificationOutcome::VerifiedWithinBounds,
        failure_class:   None,                           // PASS = no failure class
        structured_payload: None,
        intent:          TestIntent::BehaviorPreservation,
        evidence_source: EvidenceSource::AttackerGenerated,  // LLM wrote this test
    });

    evidence.push(TestEvidence {
        test_name:       "test_add_2_plus_3_equals_5".to_string(),
        outcome:         VerificationOutcome::Counterexample,
        failure_class:   Some(FailureClass::Logic),       // HARD Logic failure = bug
        structured_payload: None,
        intent:          TestIntent::BugReproduction,
        evidence_source: EvidenceSource::BenchmarkOracle,  // Ground truth test
    });

    evidence.push(TestEvidence {
        test_name:       "test_api_v2_feature_flag_on".to_string(),
        outcome:         VerificationOutcome::Inconclusive,
        failure_class:   Some(FailureClass::Unknown),     // could not classify
        structured_payload: None,
        intent:          TestIntent::FeatureAcceptance,
        evidence_source: EvidenceSource::IssueDescription, // from issue text
    });

    evidence.push(TestEvidence {
        test_name:       "test_empty_cart_checkout".to_string(),
        outcome:         VerificationOutcome::Unsupported,
        failure_class:   None,                            // backend couldn't handle it
        structured_payload: None,
        intent:          TestIntent::FeatureAcceptance,
        evidence_source: EvidenceSource::AttackerGenerated,
    });

    evidence.push(TestEvidence {
        test_name:       "test_pagination_edge_cases".to_string(),
        outcome:         VerificationOutcome::Timeout,
        failure_class:   Some(FailureClass::Infrastructure), // timed out = infr
        structured_payload: None,
        intent:          TestIntent::BehaviorPreservation,
        evidence_source: EvidenceSource::ExistingRepositoryTest, // pre-existing repo test
    });

    evidence.push(TestEvidence {
        test_name:       "test_jwt_expired_token".to_string(),
        outcome:         VerificationOutcome::ToolFailure,
        failure_class:   Some(FailureClass::Infrastructure), // backend crashed
        structured_payload: None,
        intent:          TestIntent::BehaviorPreservation,
        evidence_source: EvidenceSource::HumanProvided,    // written during code review
    });

    evidence.push(TestEvidence {
        test_name:       "test_mobile_schema_backward_compat".to_string(),
        outcome:         VerificationOutcome::VerifiedWithinBounds,
        failure_class:   None,
        structured_payload: None,
        intent:          TestIntent::BehaviorPreservation,
        evidence_source: EvidenceSource::FormalVerifier,  // from CrossHair/Kani/Z3
    });

    // ------------------------------------------------------------------------
    // Call 1: real populated evidence list → must produce full triage + counts
    // ------------------------------------------------------------------------
    // [SYNTAX] &evidence = take a shared REFERENCE to the Vec. Rust implicitly
    // coerces Vec<T> to &[T] slice at the function boundary (Deref coercion).
    // So &Vec<T> automatically becomes &[T]. You could also write
    // evidence.as_slice() for explicitness; both are fine.
    //
    // [SYNTAX] ? = TRY OPERATOR (aka "question mark operator"). Full expansion:
    //   triage_summary(&evidence)?;
    // becomes conceptually:
    //   match triage_summary(&evidence) {
    //       Ok(unit_value) => unit_value,  // continue to next line
    //       Err(error_str)  => return Err(error_str.into()),  // BUBBLE UP
    //   }
    //
    // [VANTA-WHY of ?] In M2 the real Diagnostician call chain is:
    //   get_patch_bytes()?
    //   copy_to_sandbox()?
    //   run_bate_baseline()?
    //   run_execution()?
    //   triage_results()?
    //   write_session_log()?
    // Without ? each line would be a 5-line match block, turning 7 lines into
    // 35 lines of repetitive match error propagation. ? is the single biggest
    // ergonomic win in Rust for orchestration-style code. Learn it. Love it.
    println!("▶ CALL 1: triage_summary() with {} evidence items\n", evidence.len());
    triage_summary(&evidence)?;

    // ------------------------------------------------------------------------
    // Call 2: EMPTY evidence list → must RETURN ERR cleanly (NOT A PANIC)
    // ------------------------------------------------------------------------
    // [SYNTAX] &[] = empty slice literal. Type inference matches the function's
    // &[TestEvidence] parameter automatically because we annotated main as
    // Result<(), String>, so the Err type flows back up.
    //
    // [TRACE of this EXACT LINE]:
    //   1. We call triage_summary with an empty slice
    //   2. triage_summary hits is_empty() → returns Err(CHECKPOINT_A_ERROR: ...)
    //   3. The ? operator here sees Err(...) → expands to return Err(...)
    //   4. main() returns that Err to the Rust runtime
    //   5. Runtime prints the Err string to stderr:
    //      "Error: CHECKPOINT_A_ERROR: No evidence to triage! Caller provided an empty list."
    //   6. Process exits with exit code 1.
    //   7. NO "thread 'main' panicked at ..."
    //   8. NO RUST_BACKTRACE=1 output even if the env var is set
    //
    // This is exactly what we're proving to the exit test. We WANT the Err
    // here, so the program always terminates after call 1's happy path
    // printout followed by a clean Err message from call 2.
    println!("▶ CALL 2: triage_summary() with empty evidence list (expects clean Err)\n");
    triage_summary(&[])?;

    // Unreachable due to call 2 always returning Err. Still written for clarity
    // if someone comments out call 2.
    Ok(())
}

// ============================================================================
// 5 REUSABLE RECIPES FROM CHECKPOINT A
// ============================================================================
//
// COPY-PASTE THESE TEMPLATES DIRECTLY INTO M1-M5 CODE. Every pattern below is
// used at least 10 times in the full Vanta system.
//
// ----------------------------------------------------------------------------
// RECIPE A1 — "Process a collection, count things, error if empty"
//   The #1 function shape in Vanta (appears 15+ times across modules).
//
//   fn summarize(items: &[Thing]) -> Result<Counts, String> {
//       if items.is_empty() {
//           return Err("empty! need at least 1 item".into());
//       }
//       let mut count_a: u32 = 0;
//       let mut count_b: u32 = 0;
//       for item in items {
//           match item.category {
//               Category::A => count_a += 1,
//               Category::B => count_b += 1,
//           }
//       }
//       Ok(Counts { a: count_a, b: count_b })
//   }
//
//   → USE IN: Diagnostician failure counts, Attacker discard metrics (M2),
//     Governor EIR/ECR transition counts (M4), Paper results aggregation (M5).
//
// ----------------------------------------------------------------------------
// RECIPE A2 — "Exhaustive match on N outcomes, GROUPING arms with |"
//   Use on every domain enum. Avoid wildcards (_) unless you truly intend to.
//
//   match outcome {
//       VerificationOutcome::VerifiedWithinBounds => happy_path(),
//       VerificationOutcome::Counterexample       => hard_reject(),
//       VerificationOutcome::Inconclusive
//       | VerificationOutcome::Unsupported
//       | VerificationOutcome::Timeout            => report_and_continue(),
//       VerificationOutcome::ToolFailure          => retry_infra(),
//   }
//
//   → USE IN: Gatekeeper verification policy table (M3),
//     Diagnostician 4-way triage (M2: outcome → Infrastructure/Static/Logic/Unknown).
//
// ----------------------------------------------------------------------------
// RECIPE A3 — "Always accept &[T] (borrowed slice), never Vec<T>, in functions"
//   Callers can pass: &vec   or   &array[..]   or   foo[0..5]
//   No clones, no moves, no allocation. Works with 90% of collection params.
//
//   fn find_failures(tests: &[TestResult]) -> Vec<&TestResult> {
//       tests.iter().filter(|t| !t.passed).collect()
//   }
//
//   Exception: if the function needs to add/remove items, use &mut Vec<T>.
//   If it needs ownership (cache/store the value long-term), accept T.
//
// ----------------------------------------------------------------------------
// RECIPE A4 — "fn main() -> Result<(), String> and the ? chain"
//   Always. Under all circumstances. No unwrap() / expect() in main paths.
//
//   fn main() -> Result<(), String> {
//       let cfg = load_config()?;
//       let data = fetch_data(&cfg)?;
//       let report = analyze(data)?;
//       write_report(&report)?;
//       Ok(())
//   }
//
//   → USE IN: Every binary (src/bin/*.rs, the main vanta binary). This IS the
//     Vanta CLI entry in M3: `vanta run --repo ...` → main() → chain of ? calls.
//
// ----------------------------------------------------------------------------
// RECIPE A5 — "Option<T> for fields that may or may not exist on an instance"
//   Never invent a "NoThing" enum variant. Option expresses absence correctly.
//
//   struct TestRecord {
//       name: String,
//       failure: Option<FailureInfo>,   // PASS → None, FAIL → Some(details)
//       retried_after: Option<u64>,     // not retried → None
//   }
//
//   // Usage:
//   if let Some(info) = &record.failure {
//       eprintln!("FAIL: {}", info.message);
//   } else {
//       println!("PASS");
//   }
//
//   → USE IN: TestEvidence.failure_class (M2), any "optional field" ever.
//
// ============================================================================
