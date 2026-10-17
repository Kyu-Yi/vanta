// ============================================================================
// CHECKPOINT B — Serde JSON Roundtrip + Typed Struct Parsing
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY — every concept below is REUSED in M1-M5
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | #[derive(Serialize,       → CreatorResponse / AttackerResponse typed
//     |      Deserialize)]        →   structs (the actual M1 agent responses)
//     |                           → LlmConfig config file loading (M3)
//  ---+---------------------------+-------------------------------------------
//   2 | serde_json::to_string     → M1: convert typed request bodies to JSON
//     |   (typed_struct)          →   before sending to llama-server endpoint
//     |                           → M5 experiment checkpoint serialization
//  ---+---------------------------+-------------------------------------------
//   3 | serde_json::from_str      → M1: parse model JSON response into
//     |   (json_str)              →   CreatorResponse typed struct (not HashMap!)
//     |                           → M3: session.json rehydration on resume
//  ---+---------------------------+-------------------------------------------
//   4 | Result error handling     → M1: malformed model output → retry prompt
//     |   on JSON parse           →   without panicking the 8-hour experiment
//     |                           → M3: stale config version → Err + upgrade
//  ---+---------------------------+-------------------------------------------
//   5 | Typed struct vs untyped   → M1: access creator.patch_text as a field
//     |   HashMap<String,Value>   →   that COMPILES, not hash["patch_text"]!
//     |                           → No typos can exist; compiler catches them.
//  ---+---------------------------+-------------------------------------------
//
// M0 EXIT TEST RELEVANCE: This checkpoint specifically exercises "parses a
// typed JSON response without panicking" and "handles a deliberately malformed
// response via Result". Master Plan M0 exit test line items 4 and 5.
// ============================================================================

// ----------------------------------------------------------------------------
// [SYNTAX] use serde::{Serialize, Deserialize}; = bring the two derive macros
// into scope so we can write #[derive(Serialize, Deserialize)] on our structs
// and enums below. Trait macros: Serialize = "can convert to JSON (or wire
// format)", Deserialize = "can convert FROM JSON (or wire format) back into
// typed memory". Re-exported from serde crate.
//
// [VANTA-WHY] This is #1 most-used pair of derives in the codebase. Every
// response/request/struct crosses the network boundary at some point (HTTP
// request body ↔ model endpoint; checkpoint save ↔ filesystem; CSV ↔ paper
// analysis scripts). Serde is the lingua franca.
// ----------------------------------------------------------------------------
use serde::{Deserialize, Serialize};

// ----------------------------------------------------------------------------
// CreatorResponse — THE ACTUAL SHAPE OF M1 CREATOR AGENT RESPONSES
// ----------------------------------------------------------------------------
// This is NOT a checkpoint throwaway. M1 spike_m2 reuses EXACTLY this struct.
//
// Spec §7 Creator Agent → Output: EIR (Execution Intent Record) contains:
//   patch_text            = the git-diff-like patch the Creator wants applied
//   explanation           = 1-paragraph natural language justification
//   files_touched         = list of filenames claimed to be modified
//   confidence            = Creator's self-reported 0.0..=1.0 confidence
//
// [SYNTAX] Each derive is a SEPARATE attribute macro (one per line) so you can
// add/remove traits independently and see which trait something is missing.
//
// [SYNTAX] Debug — lets you println!("{:?}", response). Required for logging.
// [SYNTAX] Clone — response.clone() if downstream wants an owned copy.
// [SYNTAX] PartialEq — == comparison: assert_eq!(a, b) in tests.
//
// [PITFALL] If you put multiple derives on one line:
//   #[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
// It works, but you will spend 10 minutes on compile errors trying to figure
// out WHICH derive a missing trait came from. One-per-line is IDE-friendly
// (rust-analyzer shows a squiggle on exactly the right line when something
// breaks on a specific trait).
//
// [VANTA-WHY of typed struct over HashMap<String, Value>]:
// If Creator wrote a response and you access it via
//   let pt = map["patch_text"].as_str().unwrap();  ← BAD (3 bugs possible)
// You can't tell at compile time:
//   (1) did you spell "patch_text" correctly?
//   (2) is the field present (vs the model omitted it)?
//   (3) is the type correct (string vs array)?
//
// With the typed struct:
//   let pt = &response.patch_text;  ← COMPILE-TIME VERIFIED
// The compiler ensures (1) field exists, (2) name correct, (3) type known.
// This single design choice eliminates ~30% of runtime bugs you would otherwise
// chase on day 1 of M1.
// ----------------------------------------------------------------------------
#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
#[derive(Clone)]
#[derive(PartialEq)]
pub struct CreatorResponse {
    // [SYNTAX] patch_text: String = owned heap string field.
    // [VANTA-WHY] In M1 this is the MODEL OUTPUT. Spec requires this field.
    // Example real value: "--- a/src/stripe.rs\n+++ b/src/stripe.rs\n ..."
    pub patch_text: String,

    // [SYNTAX] explanation: String = natural language prose.
    // [VANTA-WHY] M3 Governor stores explanation next to each attempt.
    // Used for paper writeup qualitative appendix tables.
    pub explanation: String,

    // [SYNTAX] files_touched: Vec<String> = ordered list of files modified.
    //
    // [PITFALL] Beginners sometimes use [String; 3] (fixed-size array) if they
    // think "most patches touch 3 files". Don't. Number of files varies by
    // patch; array size must be known at compile time. Vec is dynamic.
    //
    // [VANTA-WHY] Spec §7 EIR lists files_touched as a required field.
    // M2 Diagnostician uses this to detect "Creator claimed it touched X but
    // actually only modified Y" discrepancy flag.
    pub files_touched: Vec<String>,

    // [SYNTAX] confidence: f64 = 64-bit IEEE floating point.
    // [VANTA-WHY] 0.0 = "this will definitely break". 1.0 = "100% confident".
    // M3 Gatekeeper can use low confidence (< 0.4) to route directly to
    // human review even if formal checks pass.
    pub confidence: f64,
}

// ============================================================================
// roundtrip — THE CORE EXERCISE: typed struct → JSON → typed struct
// ============================================================================
//
// Returns a Result where Err(String) = something went wrong during the
// serialize or deserialize step (including the actual error text from Serde
// which has line:column info for JSON syntax errors — VERY useful).
//
// [RECIPE B3] (see footer): Every LLM response handler in M1-M3 does exactly
// this: (a) GET/POST JSON bytes, (b) typed parse, (c) use typed fields.
pub fn roundtrip_demo() -> Result<(), String> {

    // ------------------------------------------------------------------------
    // Step 1: Build a typed CreatorResponse (what we expect the LLM to emit)
    // ------------------------------------------------------------------------
    // [SYNTAX] Struct literal with all 4 fields. Order doesn't matter (named
    // fields), but keeping order = struct-declaration order is conventional.
    //
    // [VANTA-WHY] In M1 this struct doesn't exist hand-written; it comes back
    // as JSON string from the model. We write it by hand here to demonstrate
    // roundtrip on a KNOWN GOOD input so you can compare serialize output
    // vs re-parse output for equality.
    let original: CreatorResponse = CreatorResponse {
        patch_text: r#"--- a/src/cart.rs
+++ b/src/cart.rs
@@ -12,7 +12,7 @@
 pub fn apply_discount(price: u64, pct: u8) -> u64 {
-    price * (100 - pct as u64) / 100
+    let discount = price.saturating_mul(100 - pct as u64) / 100;
+    discount
 }
"#.to_string(),
        explanation: "Fixed overflow in discount calc: direct mul could overflow \
                      on high-value carts; uses saturating_mul to cap at u64::MAX \
                      before division.".to_string(),
        files_touched: vec!["src/cart.rs".to_string()],
        confidence: 0.92,
    };

    println!("▶ STEP 1: Original CreatorResponse (typed struct)");
    println!("  patch_text    : {} chars", original.patch_text.len());
    println!("  explanation   : {}", &original.explanation[..80.min(original.explanation.len())]);
    println!("  files_touched : {:?}", original.files_touched);
    println!("  confidence    : {:.3}\n", original.confidence);

    // ------------------------------------------------------------------------
    // Step 2: Serialize struct → JSON string
    // ------------------------------------------------------------------------
    // [SYNTAX] serde_json::to_string(&value) -> Result<String, serde_json::Error>
    // Takes a reference (no ownership moved). Produces a single-line JSON
    // string. For pretty-printed 2-space-indented JSON: serde_json::to_string_pretty.
    //
    // [SYNTAX] .map_err(|e| format!("SERIALIZE_ERROR: {e}"))?;
    //   Chain: to_string returns Result<String, serde_json::Error>. Our function
    //   returns Result<(), String>. Different Err types. map_err transforms
    //   the concrete serde_json::Error into our generic String error message
    //   by formatting it. Then ? bubbles the result up if Err.
    //
    // [VANTA-WHY of map_err] The Rust ? operator does NOT auto-convert between
    // unrelated error types unless you implement From<T>. In M0/M1 fast code
    // we just String-wrap via map_err for every step. In M2 we'll build a
    // proper thiserror error enum. For now: String-everywhere keeps it simple.
    let json_str: String = serde_json::to_string_pretty(&original)
        .map_err(|e| format!("SERIALIZE_ERROR: {e}"))?;

    println!("▶ STEP 2: Serialized → JSON string");
    println!("{json_str}\n");

    // ------------------------------------------------------------------------
    // Step 3: Deserialize JSON string → new typed struct
    // ------------------------------------------------------------------------
    // [SYNTAX] serde_json::from_str::<CreatorResponse>(&json_str)
    //   ::<CreatorResponse> is TURBOFISH syntax = "explicitly tell the compiler
    //   which type we want to deserialize into". Without the turbofish, Rust
    //   would look at the `let roundtripped: CreatorResponse` type annotation
    //   on the left and infer it. Both work. Turbofish written here explicitly
    //   so beginners see "what type does from_str return" right at the call.
    //
    // [VANTA-WHY] THIS IS THE M1 SPIKE m2 LINE. Model returns JSON string from
    // its /chat/completions HTTP endpoint. You strip ```json fences, then call
    // exactly from_str::<CreatorResponse>. If it works → typed data. If fails
    // → retry the model with "previous response was not valid JSON matching
    // the schema, try again". You NEVER skip the typed-parse step (into a
    // HashMap) because of the bug-prevention reasons detailed above.
    let roundtripped: CreatorResponse = serde_json::from_str::<CreatorResponse>(&json_str)
        .map_err(|e| format!("DESERIALIZE_ERROR: {e}"))?;

    println!("▶ STEP 3: Deserialized → typed CreatorResponse (new instance)");
    println!("  patch_text    : {} chars", roundtripped.patch_text.len());
    println!("  files_touched : {:?}", roundtripped.files_touched);
    println!("  confidence    : {:.3}", roundtripped.confidence);

    // ------------------------------------------------------------------------
    // Step 4: Prove they're identical via PartialEq
    // ------------------------------------------------------------------------
    // [SYNTAX] assert_eq!(a, b) — macro. Panics if a != b. Since we derive
    // PartialEq on CreatorResponse, == works on the struct. (Rust does NOT
    // auto-generate equality for user-defined types; you must derive or impl.)
    //
    // [TRACE] What PartialEq actually checks: every field compared pair-wise:
    //   original.patch_text       == roundtripped.patch_text        (str equality)
    //   original.explanation      == roundtripped.explanation
    //   original.files_touched    == roundtripped.files_touched     (Vec element-wise)
    //   original.confidence       == roundtripped.confidence        (f64 bit equality)
    // All 4 must match.
    //
    // [PITFALL] f64 equality is NOT exact for values that came from arithmetic.
    // Use (a - b).abs() < 1e-9 for real math. Here we serialized→deserialized
    // without arithmetic, so exact match holds.
    assert_eq!(original, roundtripped,
        "ROUNDTRIP FAIL: original differs from deserialized instance! \
         If you see this: Serde introduced mutation or typed fields wrong.");

    println!("\n✔ ROUNDTRIP OK — original == roundtripped (PartialEq passed)\n");

    Ok(())
}

// ============================================================================
// malformed_demo — THE CRITICAL EXIT-TEST ITEM
//   "Can you handle a deliberately malformed JSON response without panicking?"
// ============================================================================
//
// [RECIPE B4] (footer): Every model response parse in M1-M3 follows this.
// Bad JSON comes from the model constantly (fence ```json not closed, commas
// missing after last array element, quote characters that are actually Unicode
// smart quotes the model decided to use instead of ASCII ", etc.). You MUST
// catch this as a Result Err and retry; you CANNOT let parse panics bubble up
// or they'll terminate a multi-hour M5 benchmark midway.
pub fn malformed_demo() -> Result<(), String> {

    // ------------------------------------------------------------------------
    // The bad JSON: deliberately missing comma after the "patch_text" line.
    // Valid JSON would be: "patch_text": "...",   ← comma!
    // But we wrote:        "patch_text": "..."    ← no comma. SYNTAX ERROR.
    //
    // [VANTA-WHY] In real M1/M2 runs the model produces stuff like this ALL
    // THE TIME. Small models especially (4B class) will miss commas 30% of
    // early calls. Larger models are better but still do it ~2%. You must
    // gracefully handle malformed JSON by retrying with a correction prompt.
    // ------------------------------------------------------------------------
    let bad_json: &str = r#"
{
    "patch_text": "fix: typo in check_totals",
    "explanation": "Typo in function name",
    "files_touched": ["src/ledger.rs"],
    "confidence": 0.7
}
"#;

    // We're also going to test MISSING FIELD JSON and WRONG TYPE JSON to show
    // typed-parses catching schema-level bugs (not just syntax bugs).

    let missing_field_json: &str = r#"
{
    "patch_text": "stuff",
    "explanation": "oops I forgot files_touched",
    "confidence": 0.7
}
"#;

    let wrong_type_json: &str = r#"
{
    "patch_text": "things",
    "explanation": "confidence should be a number not a string",
    "files_touched": ["a.rs"],
    "confidence": "I am 90% sure"
}
"#;

    // ------------------------------------------------------------------------
    // Syntax error test (missing comma above — actually the first one we
    // wrote IS valid after all since I included comma; let's make a REAL bad
    // one here — missing comma BETWEEN patch_text and explanation)
    // ------------------------------------------------------------------------
    let syntax_err_json: &str = r#"
{
    "patch_text": "x"
    "explanation": "y",
    "files_touched": ["a.rs"],
    "confidence": 0.7
}
"#;

    println!("▶ MALFORMED JSON TEST 1 / 3: Syntax error (missing comma between fields)");
    // [SYNTAX] match on the Result of parsing. For this deliberately bad
    // input we EXPECT an Err; if it returned Ok we'd have a correctness bug.
    match serde_json::from_str::<CreatorResponse>(syntax_err_json) {
        Ok(_) => {
            return Err("MALFORMED_DEMO_FAIL: syntax-error JSON somehow parsed! \
                        Check what you fed into from_str.".to_string());
        }
        Err(e) => {
            // [SYNTAX] e.line() and e.column() = Serde-specific helpers on
            // serde_json::Error that return human-readable location of the
            // failure. These are a godsend for debugging 4000-character model
            // JSON outputs where you can't just scan by eye.
            let line = e.line();
            let col  = e.column();
            println!("  HANDLED as Result::Err (NO PANIC) ✓");
            println!("  Serde says : line {line}, column {col}");
            println!("  Message    : {e}\n");
        }
    }

    println!("▶ MALFORMED JSON TEST 2 / 3: Missing required field 'files_touched'");
    match serde_json::from_str::<CreatorResponse>(missing_field_json) {
        Ok(_) => {
            return Err("MALFORMED_DEMO_FAIL: missing-field JSON parsed! \
                        Serde should require files_touched since no default.".to_string());
        }
        Err(e) => {
            println!("  HANDLED as Result::Err (NO PANIC) ✓");
            println!("  Message    : {e}\n");
        }
    }

    println!("▶ MALFORMED JSON TEST 3 / 3: Wrong field type (confidence is string not number)");
    match serde_json::from_str::<CreatorResponse>(wrong_type_json) {
        Ok(_) => {
            return Err("MALFORMED_DEMO_FAIL: wrong-type JSON parsed! \
                        f64 field accepted a string — something very wrong.".to_string());
        }
        Err(e) => {
            println!("  HANDLED as Result::Err (NO PANIC) ✓");
            println!("  Message    : {e}\n");
        }
    }

    // Just to also show the FIRST "bad_json" we wrote (which is actually
    // valid — check it parses so you can see the difference):
    println!("▶ REFERENCE: First candidate string we wrote is ACTUALLY VALID — confirms:");
    match serde_json::from_str::<CreatorResponse>(bad_json) {
        Ok(resp) => println!("  PARSED OK. files_touched = {:?}, confidence = {}\n",
                            resp.files_touched, resp.confidence),
        Err(e)   => println!("  Unexpectedly failed: {e}\n"),
    }

    println!("✔ ALL 3 MALFORMED CASES: Result.Err captured. ZERO panics.\n");

    Ok(())
}

// ============================================================================
// main() — Entry point
// ============================================================================
fn main() -> Result<(), String> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CHECKPOINT B — Serde roundtrip + typed parse + malformed handling");
    println!("═══════════════════════════════════════════════════════════════════\n");

    roundtrip_demo()?;
    malformed_demo()?;

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CHECKPOINT B EXIT: OK");
    println!("═══════════════════════════════════════════════════════════════════");
    Ok(())
}

// ============================================================================
// 5 REUSABLE RECIPES FROM CHECKPOINT B
// ============================================================================
//
// COPY-PASTE THESE TEMPLATES DIRECTLY INTO M1-M5 CODE.
//
// ----------------------------------------------------------------------------
// RECIPE B1 — "All response types: 5 derives minimum"
//
//   #[derive(Debug)]      // println!("{:?}", x)
//   #[derive(Serialize)]  // → JSON
//   #[derive(Deserialize)]// ← JSON
//   #[derive(Clone)]      // x.clone()
//   #[derive(PartialEq)]  // assert_eq! and ==
//   pub struct CreatorResponse {
//       pub patch_text: String,
//       pub explanation: String,
//       pub files_touched: Vec<String>,
//       pub confidence: f64,
//   }
//
//   → USE IN: EVERY struct/enum that crosses a network or persistence boundary.
//
// ----------------------------------------------------------------------------
// RECIPE B2 — "Serialize to JSON string, always map_err for non-panics"
//
//   let json = serde_json::to_string_pretty(&typed_struct)
//       .map_err(|e| format!("SERIALIZE_FAILED: {e}"))?;
//
//   → USE IN: M1 request body builder, M5 checkpoint save.
//
// ----------------------------------------------------------------------------
// RECIPE B3 — "Always deserialize into typed struct, NEVER HashMap<K,V>"
//
//   let parsed: CreatorResponse = serde_json::from_str::<CreatorResponse>(&body)
//       .map_err(|e| format!("PARSE_FAILED (line {} col {}): {}", e.line(), e.column(), e))?;
//   // Use: parsed.patch_text (compiler-verified field access)
//
//   Why not HashMap? Typed fields give:
//     • No typos like map["pacth_etxt"] (HashMap runtime None vs compile error!)
//     • Correct types (Vec<String> not a generic Value you have to as_str()/as_array())
//     • Required fields are enforced by Serde automatically
//
//   → USE IN: spike_m2 (step 12 of plan), M1 CreatorResponse parse, M1 AttackerResponse parse.
//
// ----------------------------------------------------------------------------
// RECIPE B4 — "Deliberate Result-typed handling of malformed model output"
//
//   match serde_json::from_str::<T>(&model_output_cleaned) {
//       Ok(typed)  => proceed_with(typed),
//       Err(e)     => retry_prompt_with_error_context(&e.to_string()),
//   }
//
//   Never:   serde_json::from_str::<T>(&s).unwrap()  ← PANICS if bad JSON!
//   Never:   let v: Value = serde_json::from_str(&s)?; v["field"].as_str()... ← untyped!
//
//   → USE IN: Every model-response parse in M1-M3. If bad JSON → retry loop.
//
// ----------------------------------------------------------------------------
// RECIPE B5 — "Markdown fence stripping before typed JSON parse (M1 step 12 add-on)"
//
//   // Small models often wrap JSON in: ```json\n{...}\n```
//   // Strip fence. Do this AFTER HTTP body, BEFORE from_str.
//   fn strip_markdown_fences(s: &str) -> &str {
//       let s = s.trim();
//       let s = s.strip_prefix("```json").unwrap_or(s).trim_start();
//       let s = s.strip_prefix("```").unwrap_or(s).trim_start();
//       let s = s.strip_suffix("```").unwrap_or(s).trim_end();
//       s
//   }
//
//   → USE IN: spike_m2 (step 12) — the exact line order in M1 response handler.
//
// ============================================================================
