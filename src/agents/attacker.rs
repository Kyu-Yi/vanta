// ============================================================================
// ATTACKER AGENT — patch → adversarial tests with declared intent (Spec §4)
// ============================================================================
//
// Spec §4: "Attacker agent: Given a patch, writes adversarial tests with
// declared intent targeting its weak points."
//
// Spec §5 Test Intents: each test declares WHY it exists:
//   - BugReproduction: proves the reported defect exists (FAIL on original)
//   - FeatureAcceptance: checks a new capability works
//   - BehaviorPreservation: checks unrelated working behaviour survived
//
// M1 scope (per Master Plan M1 BUILD):
//   - AttackerRequest: patch, original_source, issue_text
//   - GeneratedTest: test_name, test_source, intent, reason
//   - TestIntent enum (reused from core::TestIntent)
// ============================================================================

use crate::config::LlmConfig;
use crate::core::TestIntent;
use crate::llm::{build_reqwest_client, extract_json_candidate, structured_call_with_retry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use tracing::{debug, info, trace, warn};

// ============================================================================
// ATTACKER_PROMPT_VERSION — the run-identity tag for the Attacker's prompt
// ============================================================================
//
// [VANTA-WHY] The Attacker prompt is embedded in the binary with no other
// version tag, so two runs on the same commit that differ only in the prompt
// are indistinguishable in the session log. That matters here: a run BEFORE the
// boundary-probing rule and a run AFTER it are different experiments, and the
// boundary rule is exactly what the catch-rate comparison is about. Bump this
// string whenever the prompt text changes, so `session.start` names which
// prompt produced the rows.
//
// [VANTA-WHY] The tag is a bare version ("2.boundary-probe"), not
// "attacker.2.boundary-probe": `session.start` writes
// `prompt_version=attacker=<tag>;intent_check=<tag>;...`, so the prompt NAME is
// the key and repeating it inside the value would make the field ambiguous.
pub const ATTACKER_PROMPT_VERSION: &str = "3.review-context-bundle";

// ============================================================================
// INTENT_AUDITOR_PROMPT_VERSION — the run-identity tag for the intent check
// ============================================================================
//
// [VANTA-WHY] The intent auditor (audit_test_intent) is a SEPARATE prompt from
// the Attacker's and it is the one that decides whether a test is honest:
// demoting non-preservation labels and scrapping preservation probes the issue
// requires to change. A change to THAT wording changes what a run accepts, so
// it needs its own tag — otherwise editing the auditor would be logged as "no
// change" because the Attacker's tag is untouched.
pub const INTENT_AUDITOR_PROMPT_VERSION: &str = "2.review-context-bundle";

// ============================================================================
// AttackerRequest — input to the Attacker agent
// ============================================================================
//
// [VANTA-WHY] `existing_tests` is the shared set accumulated by PREVIOUS
// attempts (Spec §8). From round 2 on the Attacker is shown the probes it has
// already written so it can attack somewhere new: without this, every round
// re-derives the same boundary test, the fresh batch adds no coverage, and the
// run burns its attack depth re-running one probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackerRequest {
    pub patch_text: String,
    pub original_source: String,
    pub issue_text: String,
    /// Probes already in the fixed shared set. Empty on round 1.
    #[serde(default)]
    pub existing_tests: Vec<SharedProbe>,
}

/// One probe already in the shared set, as shown to the Attacker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SharedProbe {
    pub test_name: String,
    pub test_source: String,
}

// ============================================================================
// GeneratedTest — one adversarial test with DECLARED INTENT
// ============================================================================
//
// [VANTA-WHY] The `intent` field is what makes BATE (Phase 1) possible.
// Without it, you cannot judge whether a test's baseline result is correct.
// A BugReproduction test SHOULD fail on original code; a BehaviorPreservation
// test SHOULD pass. Mismatch → test is invalid → scrap after 3 retries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneratedTest {
    pub test_name: String,
    pub test_source: String,
    pub intent: TestIntent,
    pub reason: String,
    pub issue_quote: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentRelabel {
    pub test_name: String,
    pub from_intent: TestIntent,
    pub to_intent: TestIntent,
    pub reason: String,
}

// ============================================================================
// AttackerResponse — typed output: array of GeneratedTest
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackerResponse {
    pub tests: Vec<GeneratedTest>,
}

// ============================================================================
// AttackerTestCase — M0 spike compatibility (simpler shape)
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackerTestCase {
    pub test_name: String,
    pub test_source: String,
    pub expect_pass: bool,
}

// ============================================================================
// attacker_generate_tests() — call LLM, get typed AttackerResponse
// ============================================================================
pub async fn attacker_generate_tests(
    cfg: &LlmConfig,
    request: &AttackerRequest,
) -> Result<AttackerResponse, Box<dyn Error + Send + Sync>> {
    info!(patch_len = request.patch_text.len(), "attacker start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = r#"You are Vanta Attacker. Given a patch, write adversarial tests designed to BREAK it. Each test MUST declare an intent. Keep reasoning SHORT (under 100 words) — save output budget for the JSON answer.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "tests": [
    {
      "test_name":   string,
      "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body. NO module-level statements. Import at top, then `def test_...():` with body. Example:
                       import pytest
                       from cart import cart_total

                       def test_example():
                           assert cart_total([(100, 2)]) == 200

                       IMPORTANT: Do NOT import constants or symbols that only exist in the PATCHED code (e.g. MAX_TOTAL). Use literal values instead so the test can run on BOTH original and patched code.
      "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
      "reason":      string — why this test targets the patch,
      "issue_quote": string|null — for BugReproduction or FeatureAcceptance, quote the exact issue sentence that asks for this behaviour, and that sentence MUST name the same input the test asserts (a test asserting `shipping_band(500)` needs a quote about 500). For BehaviorPreservation use null
    }
  ]
}

Intent rules — each test MUST produce a DIFFERENT result on original vs patched (except BehaviorPreservation):
- BugReproduction: reproduces a bug that the patch fixes. The test MUST FAIL on the ORIGINAL code and PASS on the PATCHED code. It exercises the exact code path the patch changes.
- FeatureAcceptance: checks a NEW capability added by the patch. The test MUST FAIL on the ORIGINAL code (feature absent) and PASS on the PATCHED code (feature present).
- BehaviorPreservation: checks EXISTING behavior that should not change. The test MUST PASS on BOTH original and patched code. Test only functionality that existed before the patch.

CRITICAL:
- Every test must actually import and call the patched module's functions/classes. Do NOT mock away the code under test.
- Do NOT import symbols that only exist in the PATCHED code. Use hasattr/getattr or try/except to detect new features safely on original code.
- Use assertions (assert X == Y) so failures are clean, not crashes.
- Tests run with PYTHONPATH=src so `from heretic.modifier import X` works.
- Expected values come from the ISSUE and the ORIGINAL code's existing behaviour, never from the patch alone.
- A test may be BugReproduction or FeatureAcceptance ONLY if the quoted issue sentence explicitly asks for that behaviour AT THAT INPUT. If the issue does not ask for it, label it BehaviorPreservation.
- Quoting an issue sentence that does not mention your test's input does NOT authorise a non-preservation intent. A patch changing behaviour is NOT evidence the issue wanted the change — "the patch does X" is never a reason to label a test FeatureAcceptance.

IMPORT RULES (violating these makes the test ERROR on collection — worse than failing):
- Only import a name you can LITERALLY SEE defined in the ORIGINAL SOURCE above, and import it from the file where it is defined. Do NOT guess that a symbol lives in a module.
- When unsure, import the MODULE and access the attribute defensively, e.g. `import heretic.system as sysmod; fn = getattr(sysmod, "is_mps_available", None); assert fn is not None`. Never write `from m import name` for a name you have not seen defined.
- A test whose import raises is treated as an infra ERROR, not a clean failure — it is scrapped.

NO-REIMPLEMENTATION RULE:
- Never re-implement the patch's logic inside the test body (e.g. re-setting an env var yourself and asserting it is set). Such a test always passes on BOTH versions and proves nothing. Call the repo's real code instead.

INTENT-vs-API RULE:
- BehaviorPreservation may ONLY exercise APIs that already exist in the ORIGINAL source. If the behaviour you want to check uses a symbol the patch ADDS, it is a FeatureAcceptance test, not BehaviorPreservation.

BOUNDARY-PROBING RULE:
- A patch that widens, narrows or shifts a comparison (e.g. `<` → `<=`, or a changed threshold constant) changes behaviour ONLY at the exact boundary value. A test at a nearby interior value (e.g. 499 when the boundary is 500) returns the SAME result on original and patched code and proves nothing.
- For every comparison the patch changes, include a BehaviorPreservation test that asserts the ORIGINAL result at the EXACT boundary constant — the value named in the comparison itself, not its neighbour. Do this for every changed comparison the ISSUE does not ask to change.
- When the issue explicitly asks to change behaviour at a given boundary (e.g. "1000 should be standard"), do NOT write a preservation test for that boundary; cover it with a BugReproduction test instead.
- Example: a patch changing `if w < 500` to `if w <= 500` moves the ORIGINAL result at 500 from `"standard"` (because `< 500` is false at 500) to `"light"`. The preservation probe must therefore assert `shipping_band(500) == "standard"` — the original result AT 500 — and never assert `shipping_band(499)`, which returns the same band on both versions and proves nothing.
- Read the ORIGINAL result at the boundary out of the ORIGINAL SOURCE you were given. Never copy the patched value back as the expectation.

Write exactly 3 tests. Label each one honestly: use BugReproduction or FeatureAcceptance only when the issue asks for that behaviour at the input you assert, otherwise use BehaviorPreservation. Three BehaviorPreservation tests are a correct answer when the issue only describes one boundary."#;

    // [VANTA-WHY] Round >= 2: show the probes already in the shared set, with
    // the input literals each one pins, and forbid repeating them. A duplicate
    // probe adds no coverage but does consume a BATE slot and a patched run, so
    // the driver also drops exact duplicates before BATE; this section is what
    // makes the Attacker aim somewhere new instead of being silently discarded.
    let already_probed = if request.existing_tests.is_empty() {
        String::new()
    } else {
        let mut section = String::from(
            "\n\nALREADY IN THE SHARED TEST SET — DO NOT REPEAT ANY OF THESE:\n",
        );
        for probe in &request.existing_tests {
            let inputs = asserted_input_literals(&probe.test_source);
            section.push_str(&format!(
                "- {} (asserted inputs: [{}]):\n```\n{}\n```\n",
                probe.test_name,
                inputs.join(", "),
                probe.test_source
            ));
        }
        section.push_str(
            "Each test above already runs on every attempt. A new test that asserts the SAME input \
             with the SAME expected value is a duplicate, is discarded before it runs, and does not \
             count as an attack. Probe a DIFFERENT boundary, threshold, branch or input the patch \
             touches.\n",
        );
        section
    };

    let user_prompt = format!(
        "PATCH:\n```diff\n{}\n```\n\nORIGINAL SOURCE:\n```\n{}\n```\n\nISSUE:\n{}{}\n\nWrite AttackerResponse JSON.",
        request.patch_text, request.original_source, request.issue_text, already_probed
    );

    debug!(system_len = system_prompt.len(), user_len = user_prompt.len(), "attacker prompts built");

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object {{\"tests\":[...]}}. Each test has test_name, test_source, intent, reason.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 32768,
        "reasoning_effort": "low",
    });

    let parse_closure = |raw: &str| -> Result<AttackerResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        trace!(json = %cleaned.chars().take(300).collect::<String>(), "attacker extracted json");
        serde_json::from_str::<AttackerResponse>(cleaned).map_err(|e| {
            format!("ATTACKER PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: AttackerResponse = structured_call_with_retry(
        &client, cfg, 3, build_messages, extras, parse_closure
    ).await?;

    info!(test_count = typed.tests.len(), "attacker done");
    for t in &typed.tests {
        debug!(test_name = %t.test_name, intent = ?t.intent, src_len = t.test_source.len(), "attacker test");
    }

    Ok(typed)
}

// ============================================================================
// attacker_revise_test() — BATE feedback loop: revise a mismatched test
// ============================================================================
//
// [VANTA-WHY] Spec §7 Phase 1: "Mismatch returns to the Attacker for revision,
// max 3 retries, then scrap." When a test's baseline result doesn't match its
// declared intent, we don't just scrap it — we ask the Attacker to fix it.
// We pass the original test, the mismatch reason, and the original source so
// the Attacker knows what went wrong and can produce a corrected test.
//
// This is the "repair" half of BATE. The retry loop (validate_with_retry)
// handles flakiness; this handles genuinely wrong tests.
// ============================================================================
pub async fn attacker_revise_test(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    prior_test: &GeneratedTest,
    mismatch_reason: &str,
) -> Result<GeneratedTest, Box<dyn Error + Send + Sync>> {
    info!(test_name = %prior_test.test_name, "attacker revise_test start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = r#"You are Vanta Attacker. A test you wrote failed BATE (Baseline-Aware Test Evaluation). The test's actual result on the ORIGINAL (unpatched) code did not match its declared intent. Revise the test so its baseline result matches its intent.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "test_name":   string,
  "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body. NO module-level statements. Import at top, then `def test_...():` with body.
                   IMPORTANT: Do NOT import constants or symbols that only exist in the PATCHED code. Use literal values.
  "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
  "reason":      string — why this revised test now matches its intent,
  "issue_quote": string|null — for BugReproduction or FeatureAcceptance, quote the exact issue sentence that asks for this behaviour; for BehaviorPreservation use null
}

Intent rules (CRITICAL — these determine the expected baseline result):
- BugReproduction:      must FAIL on original code (the bug is present)
- FeatureAcceptance:    must FAIL on original code (feature doesn't exist yet)
- BehaviorPreservation: must PASS on original code (existing behavior)

If you cannot make the test match its intent, change the intent to match the test's actual behavior."#;

    let user_prompt = format!(
        "ISSUE:\n{issue_text}\n\n\
         ORIGINAL SOURCE:\n```\n{original_source}\n```\n\n\
         PRIOR TEST (failed BATE):\n- name: {name}\n- intent: {intent:?}\n- source:\n```python\n{source}\n```\n\n\
         BATE MISMATCH REASON:\n{reason}\n\n\
         Revise the test so its baseline result on the original source matches its declared intent. Expected values come from the issue and original code, never from the patch alone. Output JSON.",
        issue_text = issue_text,
        original_source = original_source,
        name = prior_test.test_name,
        intent = prior_test.intent,
        source = prior_test.test_source,
        reason = mismatch_reason,
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object with test_name, test_source, intent, reason, issue_quote.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 32768,
        "reasoning_effort": "low",
    });

    let parse_closure = |raw: &str| -> Result<GeneratedTest, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<GeneratedTest>(cleaned).map_err(|e| {
            format!("ATTACKER REVISE PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let revised: GeneratedTest = structured_call_with_retry(
        &client, cfg, 3, build_messages, extras, parse_closure
    ).await?;

    info!(test_name = %revised.test_name, intent = ?revised.intent, "attacker revise_test done");
    Ok(revised)
}

// ============================================================================
// attacker_simplify_test() — Unknown remediation: simplify a flaky/complex test
// ============================================================================
//
// [VANTA-WHY] Spec §7 Phase 2 Unknown handling: "rerun in a fresh sandbox,
// capture logs, return to the Attacker to simplify, quarantine if it persists."
// When a test produces an Unknown result (can't classify), we ask the Attacker
// to simplify it — reduce complexity, remove dependencies, make it more
// deterministic. If the simplified test still produces Unknown, we quarantine.
// ============================================================================
pub async fn attacker_simplify_test(
    cfg: &LlmConfig,
    original_source: &str,
    prior_test: &GeneratedTest,
) -> Result<GeneratedTest, Box<dyn Error + Send + Sync>> {
    info!(test_name = %prior_test.test_name, "attacker simplify_test start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = r#"You are Vanta Attacker. A test you wrote produced an UNKNOWN result (could not be classified as Pass/Fail/Infrastructure/Static). The test may be too complex, non-deterministic, or have side effects. Simplify it so it produces a clear PASS or FAIL.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "test_name":   string,
  "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body.
                   SIMPLIFY: remove network calls, time.sleep, randomness, complex setup. Use literal values. Keep the same intent.
  "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
  "reason":      string — what you simplified and why,
  "issue_quote": string|null — preserve the prior issue quote for BugReproduction/FeatureAcceptance; use null for BehaviorPreservation
}

Do NOT import constants or symbols that only exist in the PATCHED code."#;

    let user_prompt = format!(
        "ORIGINAL SOURCE:\n```\n{original_source}\n```\n\n\
         PRIOR TEST (Unknown result):\n- name: {name}\n- intent: {intent:?}\n- source:\n```python\n{source}\n```\n\n\
         Simplify this test so it produces a deterministic PASS or FAIL. Output JSON.",
        original_source = original_source,
        name = prior_test.test_name,
        intent = prior_test.intent,
        source = prior_test.test_source,
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object with test_name, test_source, intent, reason, issue_quote.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 32768,
        "reasoning_effort": "low",
    });

    let parse_closure = |raw: &str| -> Result<GeneratedTest, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<GeneratedTest>(cleaned).map_err(|e| {
            format!("ATTACKER SIMPLIFY PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let simplified: GeneratedTest = structured_call_with_retry(
        &client, cfg, 3, build_messages, extras, parse_closure
    ).await?;

    info!(test_name = %simplified.test_name, "attacker simplify_test done");
    Ok(simplified)
}

// ============================================================================
// INTENT AUDITOR — a separate call that never sees the patch
// ============================================================================
//
// [VANTA-WHY] The Attacker declares WHY a test exists, and BATE trusts that
// declaration: a BugReproduction/FeatureAcceptance test is ALLOWED to fail on
// the original code, a BehaviorPreservation test is not. That asymmetry is what
// let a bad patch write its own acceptance test — the Attacker read the patch,
// saw a changed behaviour, and labelled the change "FeatureAcceptance", so the
// change became the requirement it satisfied.
//
// The Attacker prompt forbids that. A prompt is not a guarantee. This call is
// the independent check: it is given the ISSUE, the ORIGINAL source and the
// test — never the patch, never the patched source. A non-preservation label
// survives only if this audit says the issue asks for that behaviour at the
// input the test asserts.
//
// [VANTA-WHY] The audit never PROMOTES a label and never turns a failing
// candidate into a passing one. It has two directions:
//   - a non-preservation label survives only if the audit says the issue asks
//     for that behaviour at the asserted input; otherwise it is demoted to
//     BehaviorPreservation. An auditor that cannot answer demotes rather than
//     letting the label through, because an unverified non-preservation label
//     must not stand;
//   - a BehaviorPreservation test is scrapped when the audit says the issue
//     REQUIRES the asserted behaviour to change at that input, because the test
//     then contradicts the issue. This is what makes the "the issue only asks to
//     change this one boundary" exception a checked fact rather than a sentence
//     the Attacker may or may not have honoured.
// EVERY test is audited and gets a record, preservation included.

/// Input to the intent audit.
///
/// [VANTA-WHY] Deliberately has NO patch field and no patched-source field, so
/// there is nothing for the call to leak. A regression test asserts the request
/// body the stub server receives contains neither.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCheckRequest {
    pub issue_text: String,
    pub original_source: String,
    pub test_name: String,
    pub test_source: String,
    pub declared_intent: TestIntent,
}

/// The auditor's answer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentVerdict {
    /// Does the issue ask for the behaviour this test asserts, at the input the
    /// test asserts?
    pub issue_asks_for_behaviour: bool,
    /// [VANTA-WHY] The question a preservation test needs. A BehaviorPreservation
    /// test asserts the ORIGINAL result, so `issue_asks_for_behaviour` is the
    /// wrong question for it — an issue that asks for a DIFFERENT result there
    /// answers false, which would wrongly keep a probe that contradicts the
    /// issue. This field asks the complementary question: does the issue require
    /// the behaviour at the asserted input to CHANGE from what the original code
    /// does? `true` means a preservation probe at that input contradicts the
    /// issue and must be scrapped. Defaulted so an older/short answer still
    /// parses, and a missing answer is never read as "the issue requires a
    /// change".
    #[serde(default)]
    pub issue_requires_change: bool,
    pub reason: String,
}

/// [VANTA-WHY] The deterministic quote evidence is RECORDED, never enforced. It
/// tells a reader whether the Attacker's quote was real and whether it named the
/// asserted input; the verdict is what decides the label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuoteEvidence {
    pub quote_present: bool,
    pub quote_in_issue: bool,
    pub quote_names_asserted_input: bool,
    pub asserted_inputs: Vec<String>,
}

/// One audit result. Emitted for EVERY test — kept, demoted or scrapped — so a
/// run's log shows the check ran and what it decided.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentCheckRecord {
    pub test_name: String,
    pub declared_intent: TestIntent,
    pub issue_asks_for_behaviour: bool,
    /// [VANTA-WHY] The complementary question a preservation probe needs: does
    /// the issue require the behaviour at the asserted input to CHANGE? Logged
    /// alongside the verdict so a reader can see why a preservation test was
    /// kept or scrapped, not just that it was.
    pub issue_requires_change: bool,
    /// [VANTA-WHY] True when a BehaviorPreservation test asserts behaviour the
    /// issue requires to CHANGE. It contradicts the issue, so it is scrapped
    /// rather than demoted: there is no honest label left for it.
    pub scrapped: bool,
    pub reason: String,
    pub evidence: QuoteEvidence,
}

/// [VANTA-WHY] The first line is the stub server's dispatch marker for this call
/// kind; an integration test uses it to find the audit requests and prove their
/// bodies carry no patch.
const INTENT_AUDITOR_SYSTEM_PROMPT: &str = r#"You are Vanta Intent Auditor. You are given an ISSUE, the ORIGINAL source of a repository, and ONE test. You are NOT given the patch and must not speculate about it.

Answer two questions about the input the test asserts:
1. "issue_asks_for_behaviour": does the ISSUE ask for the result this test asserts, AT THE INPUT THE TEST ASSERTS?
2. "issue_requires_change": does the ISSUE require the behaviour at that input to CHANGE from what the ORIGINAL source does there — that is, does the issue ask for a DIFFERENT result than the original returns at that input?

Rules:
- The issue is the only source of requirements. The original source shows what the behaviour WAS; it is evidence of existing behaviour, never of a new requirement.
- A test asserting `shipping_band(500) == "light"` is authorised only by an issue sentence that asks for that result at that input. A sentence about a DIFFERENT input does not authorise it.
- A sentence that forbids changing behaviour ("without changing the existing behaviour at the other thresholds") FORBIDS the change; it never authorises it and never requires one.
- An issue that asks for a change at ONE input does not require a change at any other input: answer "issue_requires_change" for the input THIS test asserts, not for the issue in general.
- If the issue does not ask for the asserted behaviour at the asserted input, answer "issue_asks_for_behaviour": false. Preserving existing behaviour is always a valid answer.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.
{"issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "reason": "one short sentence"}"#;

/// Audit one test's declared intent against the issue.
///
/// Never sees the patch: `IntentCheckRequest` has no field for it.
pub async fn attacker_check_intent(
    cfg: &LlmConfig,
    request: &IntentCheckRequest,
) -> Result<IntentVerdict, Box<dyn Error + Send + Sync>> {
    info!(
        test_name = %request.test_name,
        intent = ?request.declared_intent,
        "intent auditor start"
    );

    let client = build_reqwest_client(cfg)?;

    let user_prompt = format!(
        "ISSUE:\n{}\n\nORIGINAL SOURCE:\n```\n{}\n```\n\nTEST UNDER AUDIT ({}):\n```\n{}\n```\n\nDeclared intent: {:?}\n\nDoes the issue ask for this behaviour at the input this test asserts, and does it require the behaviour at that input to change? Answer with JSON.",
        request.issue_text,
        request.original_source,
        request.test_name,
        request.test_source,
        request.declared_intent
    );

    let sys_owned = INTENT_AUDITOR_SYSTEM_PROMPT.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            messages.push(json!({
                "role": "user",
                "content": "REPROMPT: output ONLY {\"issue_asks_for_behaviour\": true|false, \"issue_requires_change\": true|false, \"reason\": \"...\"}."
            }));
        }
        messages
    };

    // temperature 0: the audit must not be a coin flip. It also never sees the
    // patch, so there is no patch-shaped variance to average over.
    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 2048,
        "reasoning_effort": "low",
    });

    let parse_closure = |raw: &str| -> Result<IntentVerdict, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<IntentVerdict>(cleaned).map_err(|e| {
            format!(
                "INTENT AUDIT PARSE FAIL (line {} col {}): {}",
                e.line(),
                e.column(),
                e
            )
        })
    };

    let verdict: IntentVerdict =
        structured_call_with_retry(&client, cfg, 3, build_messages, extras, parse_closure).await?;

    info!(
        test_name = %request.test_name,
        asks = verdict.issue_asks_for_behaviour,
        "intent auditor done"
    );
    Ok(verdict)
}

/// Deterministic quote evidence for the session log. RECORDED ONLY.
pub fn observe_quote_evidence(test: &GeneratedTest, issue_text: &str) -> QuoteEvidence {
    let normalized_issue = normalize_issue_text(issue_text);
    let normalized_quote = test
        .issue_quote
        .as_deref()
        .map(normalize_issue_text)
        .filter(|quote| !quote.is_empty());

    let quote_present = normalized_quote.is_some();
    let quote_in_issue = normalized_quote
        .as_deref()
        .is_some_and(|quote| normalized_issue.contains(quote));

    let asserted_inputs = asserted_input_literals(&test.test_source);
    let quote_names_asserted_input = normalized_quote.as_deref().is_some_and(|quote| {
        asserted_inputs.is_empty()
            || asserted_inputs
                .iter()
                .any(|literal| quote_names_literal(quote, literal))
    });

    QuoteEvidence {
        quote_present,
        quote_in_issue,
        quote_names_asserted_input,
        asserted_inputs,
    }
}

/// Audit one test against the issue.
///
/// Returns the surviving test (`None` = scrapped), the relabel event if the test
/// was demoted, and the audit record. A record is returned for EVERY test,
/// preservation included, so the session log shows the check ran and what it
/// decided — the demote-only asymmetry is what made a preservation test the one
/// label nobody re-checked.
///
/// Three outcomes, one per way the declared intent can relate to the issue:
///   - non-preservation + issue asks for it  → keep.
///   - non-preservation + issue does NOT ask → demote to BehaviorPreservation.
///     An auditor that cannot answer takes this path: an unverified
///     non-preservation label must not stand.
///   - BehaviorPreservation + the issue REQUIRES the asserted behaviour to
///     change → SCRAP. The test contradicts the issue, so no honest label
///     remains for it. Without this, the "the issue only asks to change 1000"
///     exception lives in prompt wording alone and a probe asserting the old
///     value at 1000 rides into BATE, fails the patched side and rejects a
///     correct patch.
pub async fn audit_test_intent(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    test: &GeneratedTest,
) -> (Option<GeneratedTest>, Option<IntentRelabel>, Option<IntentCheckRecord>) {
    let evidence = observe_quote_evidence(test, issue_text);

    let request = IntentCheckRequest {
        issue_text: issue_text.to_string(),
        original_source: original_source.to_string(),
        test_name: test.test_name.clone(),
        test_source: test.test_source.clone(),
        declared_intent: test.intent.clone(),
    };

    let verdict = match attacker_check_intent(cfg, &request).await {
        Ok(verdict) => verdict,
        Err(e) => {
            warn!(
                test = %test.test_name,
                error = %e,
                "intent audit unavailable — demoting to BehaviorPreservation"
            );
            IntentVerdict {
                issue_asks_for_behaviour: false,
                issue_requires_change: false,
                reason: format!("intent audit unavailable ({e})"),
            }
        }
    };

    if matches!(test.intent, TestIntent::BehaviorPreservation) {
        if verdict.issue_requires_change {
            let reason = format!(
                "the issue requires the asserted behaviour to change at the asserted input: {}",
                verdict.reason
            );
            return (
                None,
                None,
                Some(IntentCheckRecord {
                    test_name: test.test_name.clone(),
                    declared_intent: test.intent.clone(),
                    issue_asks_for_behaviour: verdict.issue_asks_for_behaviour,
                    issue_requires_change: true,
                    scrapped: true,
                    reason,
                    evidence,
                }),
            );
        }
        return (
            Some(test.clone()),
            None,
            Some(IntentCheckRecord {
                test_name: test.test_name.clone(),
                declared_intent: test.intent.clone(),
                issue_asks_for_behaviour: verdict.issue_asks_for_behaviour,
                issue_requires_change: false,
                scrapped: false,
                reason: verdict.reason,
                evidence,
            }),
        );
    }

    if verdict.issue_asks_for_behaviour {
        return (
            Some(test.clone()),
            None,
            Some(IntentCheckRecord {
                test_name: test.test_name.clone(),
                declared_intent: test.intent.clone(),
                issue_asks_for_behaviour: true,
                issue_requires_change: verdict.issue_requires_change,
                scrapped: false,
                reason: verdict.reason,
                evidence,
            }),
        );
    }

    let mut demoted = test.clone();
    demoted.intent = TestIntent::BehaviorPreservation;
    demoted.issue_quote = None;

    let record = IntentCheckRecord {
        test_name: test.test_name.clone(),
        declared_intent: test.intent.clone(),
        issue_asks_for_behaviour: false,
        issue_requires_change: verdict.issue_requires_change,
        scrapped: false,
        reason: verdict.reason.clone(),
        evidence,
    };
    let event = IntentRelabel {
        test_name: test.test_name.clone(),
        from_intent: test.intent.clone(),
        to_intent: TestIntent::BehaviorPreservation,
        reason: format!(
            "the issue does not ask for {:?} at the asserted input: {}",
            test.intent, verdict.reason
        ),
    };
    (Some(demoted), Some(event), Some(record))
}

/// Audit a whole batch. Scrapped tests are dropped from the returned list; their
/// audit records (with `scrapped = true`) are still returned for logging.
pub async fn audit_tests_intent(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    tests: &[GeneratedTest],
) -> (
    Vec<GeneratedTest>,
    Vec<IntentRelabel>,
    Vec<IntentCheckRecord>,
) {
    let mut audited = Vec::with_capacity(tests.len());
    let mut relabels = Vec::new();
    let mut records = Vec::new();
    for test in tests {
        let (updated, relabel, record) =
            audit_test_intent(cfg, issue_text, original_source, test).await;
        if let Some(updated) = updated {
            audited.push(updated);
        }
        if let Some(relabel) = relabel {
            relabels.push(relabel);
        }
        if let Some(record) = record {
            records.push(record);
        }
    }
    (audited, relabels, records)
}

/// Whitespace-collapsed, markdown-free view of issue text for quote matching.
///
/// [VANTA-WHY] Issues wrap identifiers in markdown backticks, so an Attacker
/// that quotes the line literally *without* the backticks is not paraphrasing —
/// it is quoting. Dropping the ticks keeps such a quote admissible instead of
/// demoting a legitimate intent to BehaviourPreservation.
fn normalize_issue_text(text: &str) -> String {
    // Remove the ticks outright rather than replacing them with a space:
    // `"standard"`, must collapse to `"standard",` and not to `"standard" ,`.
    text.replace('`', "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Literal arguments appearing inside a call in the test source.
///
/// `assert shipping_band(500) == "light"` → `["500"]`.
///
/// [VANTA-WHY] These are the inputs a non-preservation test pins, and they are
/// the only deterministic link between a test and the issue sentence that is
/// supposed to authorise it. Without this, ANY verbatim sentence from the issue
/// is a licence: the fixture's real run quoted
/// "…without changing the existing behavior at the other thresholds." — a line
/// that forbids the very change the test encoded — and the quote check passed,
/// so the bad patch wrote its own acceptance test again.
fn asserted_input_literals(test_source: &str) -> Vec<String> {
    let chars: Vec<char> = test_source.chars().collect();
    let mut literals: Vec<String> = Vec::new();
    let mut depth = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        match chars[i] {
            '"' | '\'' => {
                let quote = chars[i];
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j] != quote {
                    j += 1;
                }
                if depth > 0 {
                    let literal: String = chars[start..j.min(chars.len())].iter().collect();
                    let literal = literal.trim().to_string();
                    if literal.chars().count() >= 2 && !literals.contains(&literal) {
                        literals.push(literal);
                    }
                }
                i = j + 1;
                continue;
            }
            '(' => {
                depth += 1;
                i += 1;
                continue;
            }
            ')' => {
                depth = depth.saturating_sub(1);
                i += 1;
                continue;
            }
            c if c.is_ascii_digit() && depth > 0 => {
                let starts_token = chars
                    .get(i.wrapping_sub(1))
                    .map_or(true, |p| !(p.is_ascii_alphanumeric() || *p == '_' || *p == '.'));
                if starts_token {
                    let mut j = i;
                    while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                        j += 1;
                    }
                    let literal: String = chars[i..j].iter().collect();
                    if !literals.contains(&literal) {
                        literals.push(literal);
                    }
                    i = j;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    literals
}

/// Does `quote` name `literal` as a standalone token?
///
/// Guards `500` against matching inside `1500`, and `2` against `20`.
fn quote_names_literal(quote: &str, literal: &str) -> bool {
    let mut from = 0usize;
    while let Some(offset) = quote[from..].find(literal) {
        let start = from + offset;
        let end = start + literal.len();
        let before_ok = quote[..start]
            .chars()
            .last()
            .map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '.'));
        let after_ok = quote[end..]
            .chars()
            .next()
            .map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '.'));
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

// [VANTA-WHY] There is deliberately NO deterministic relabel rule here any more.
// The quote-substring check (`relabel_test_to_issue_scope`) used to be the RULE:
// a non-preservation label survived iff its `issue_quote` appeared in the issue
// and named an asserted input. Sheriffing the label that way was still a lexical
// test pretending to be a semantic one — it can only ask whether a string
// appears, never whether the issue ASKS FOR the behaviour — and its own
// "anchored" case passed a quote that forbade the change it authorised. The
// substring evidence survives as `QuoteEvidence` on every `IntentCheckRecord`
// (a LOGGED FIELD, for readers and for re-analysis), while the decision is made
// by the patch-blind `attacker_check_intent` audit.

#[cfg(test)]
mod tests {
    use super::*;

    /// The real fixture issue, markdown and all — it is what exposed both
    /// defects: the backticked line and the sentence that forbids the change.
    const FIXTURE_ISSUE: &str = "`shipping_band(1000)` should return `\"standard\"`, not `\"heavy\"`.\n\nRight now the threshold logic treats weights strictly below `1000` grams as\n`\"standard\"` and everything else as `\"heavy\"`, so the exact `1000` gram edge\nfalls into the wrong band.\n\nPlease fix that boundary bug without changing the existing behavior at the\nother thresholds.\n";

    fn generated(intent: TestIntent, test_source: &str, issue_quote: Option<&str>) -> GeneratedTest {
        GeneratedTest {
            test_name: "test_boundary".into(),
            test_source: test_source.into(),
            intent,
            reason: "exercise a boundary".into(),
            issue_quote: issue_quote.map(str::to_string),
        }
    }

    // ---- quote evidence (RECORDED, never the rule) --------------------------

    #[test]
    fn quote_evidence_reports_a_missing_quote() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                None,
            ),
            FIXTURE_ISSUE,
        );
        assert!(!evidence.quote_present);
        assert!(!evidence.quote_in_issue);
        assert_eq!(evidence.asserted_inputs, vec!["500".to_string()]);
    }

    /// The exact failure the fixture matrix produced: a verbatim issue sentence
    /// that argues AGAINST the label. The evidence must make that visible.
    #[test]
    fn quote_evidence_shows_a_verbatim_quote_that_misses_the_asserted_input() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some(
                    "Please fix that boundary bug without changing the existing behavior at the other thresholds.",
                ),
            ),
            FIXTURE_ISSUE,
        );
        assert!(evidence.quote_present);
        assert!(
            evidence.quote_in_issue,
            "the quote really is verbatim in the issue — which is exactly why a \
             substring check is not an endorsement"
        );
        assert!(
            !evidence.quote_names_asserted_input,
            "the quote is about other thresholds, not the asserted 500g input"
        );
    }

    #[test]
    fn quote_evidence_marks_a_neighbouring_input_as_unmatched() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some("the exact `1000` gram edge falls into the wrong band."),
            ),
            FIXTURE_ISSUE,
        );
        assert!(evidence.quote_in_issue);
        assert!(!evidence.quote_names_asserted_input);
    }

    /// A quote that does name the asserted input is recorded as anchored, even
    /// though the Attacker dropped the markdown backticks.
    #[test]
    fn quote_evidence_accepts_a_quote_without_backticks() {
        for quote in [
            "shipping_band(1000) should return \"standard\", not \"heavy\".",
            "`shipping_band(1000)` should return `\"standard\"`, not `\"heavy\"`.",
        ] {
            let evidence = observe_quote_evidence(
                &generated(
                    TestIntent::BugReproduction,
                    "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                    Some(quote),
                ),
                FIXTURE_ISSUE,
            );
            assert!(evidence.quote_in_issue, "quote={quote}");
            assert!(evidence.quote_names_asserted_input, "quote={quote}");
        }
    }

    #[test]
    fn quote_evidence_matching_normalizes_whitespace() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("shipping_band(1000)   should return  \"standard\""),
            ),
            "shipping_band(1000)\nshould return \"standard\" and not \"heavy\"",
        );
        assert!(evidence.quote_in_issue);
        assert!(evidence.quote_names_asserted_input);
    }

    /// A test with no literal call arguments cannot be anchored; the evidence
    /// must say so rather than claiming the quote matches.
    #[test]
    fn quote_evidence_reports_an_unanchorable_test() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert compute() is not None\n",
                Some("A new compute helper must exist."),
            ),
            "A new compute helper must exist.",
        );
        assert!(evidence.quote_in_issue);
        assert!(evidence.asserted_inputs.is_empty());
        assert!(
            evidence.quote_names_asserted_input,
            "with nothing to anchor on, presence is all the evidence can report"
        );
    }

    #[test]
    fn asserted_input_literals_reads_call_arguments_only() {
        assert_eq!(
            asserted_input_literals("def t():\n    assert shipping_band(500) == \"light\"\n"),
            vec!["500".to_string()]
        );
        assert!(asserted_input_literals("def t():\n    assert add(2, 3) == 5\n")
            .iter()
            .all(|l| l == "2" || l == "3"));
        // `== 1500` is an expected value, not an argument.
        assert_eq!(
            asserted_input_literals("def t():\n    assert shipping_band(500) == 1500\n"),
            vec!["500".to_string()]
        );
    }

    #[test]
    fn quote_names_literal_respects_token_boundaries() {
        assert!(quote_names_literal("shipping_band(500) is light", "500"));
        assert!(!quote_names_literal("shipping_band(1500) is heavy", "500"));
        assert!(quote_names_literal("add(2, 3)", "2"));
        assert!(!quote_names_literal("add(20, 3)", "2"));
    }
}
