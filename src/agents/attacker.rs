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
use tracing::{debug, info, trace};

// ============================================================================
// AttackerRequest — input to the Attacker agent
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackerRequest {
    pub patch_text: String,
    pub original_source: String,
    pub issue_text: String,
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

Write exactly 3 tests. Label each one honestly: use BugReproduction or FeatureAcceptance only when the issue asks for that behaviour at the input you assert, otherwise use BehaviorPreservation. Three BehaviorPreservation tests are a correct answer when the issue only describes one boundary."#;

    let user_prompt = format!(
        "PATCH:\n```diff\n{}\n```\n\nORIGINAL SOURCE:\n```\n{}\n```\n\nISSUE:\n{}\n\nWrite AttackerResponse JSON.",
        request.patch_text, request.original_source, request.issue_text
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

pub fn relabel_test_to_issue_scope(
    test: &GeneratedTest,
    issue_text: &str,
) -> (GeneratedTest, Option<IntentRelabel>) {
    if matches!(test.intent, TestIntent::BehaviorPreservation) {
        return (test.clone(), None);
    }

    let normalized_issue = normalize_issue_text(issue_text);
    let normalized_quote = test
        .issue_quote
        .as_deref()
        .map(normalize_issue_text)
        .filter(|quote| !quote.is_empty());

    let quote_in_issue = normalized_quote
        .as_deref()
        .is_some_and(|quote| normalized_issue.contains(quote));

    // The quote must also be ABOUT the behaviour under test. Presence alone is
    // not endorsement.
    let anchors = asserted_input_literals(&test.test_source);
    let quote_names_input = normalized_quote.as_deref().is_some_and(|quote| {
        anchors.is_empty() || anchors.iter().any(|a| quote_names_literal(quote, a))
    });

    if quote_in_issue && quote_names_input {
        return (test.clone(), None);
    }

    let reason = match (test.issue_quote.as_deref().map(str::trim), quote_in_issue) {
        (Some("") | None, _) => format!(
            "intent {:?} requires an issue_quote that appears in the issue text; no usable quote was provided",
            test.intent
        ),
        (Some(_), false) => format!(
            "intent {:?} requires an issue_quote that appears in the issue text; quote not found: {:?}",
            test.intent,
            test.issue_quote.as_deref().unwrap_or_default()
        ),
        (Some(_), true) => format!(
            "intent {:?} requires an issue_quote that names the behaviour under test; quote does not mention any asserted input {:?}",
            test.intent, anchors
        ),
    };

    let mut relabeled = test.clone();
    relabeled.intent = TestIntent::BehaviorPreservation;
    relabeled.issue_quote = None;

    (
        relabeled,
        Some(IntentRelabel {
            test_name: test.test_name.clone(),
            from_intent: test.intent.clone(),
            to_intent: TestIntent::BehaviorPreservation,
            reason,
        }),
    )
}

pub fn relabel_tests_to_issue_scope(
    tests: &[GeneratedTest],
    issue_text: &str,
) -> (Vec<GeneratedTest>, Vec<IntentRelabel>) {
    let mut relabeled_tests = Vec::with_capacity(tests.len());
    let mut events = Vec::new();
    for test in tests {
        let (updated, event) = relabel_test_to_issue_scope(test, issue_text);
        relabeled_tests.push(updated);
        if let Some(event) = event {
            events.push(event);
        }
    }
    (relabeled_tests, events)
}

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

    #[test]
    fn non_preservation_test_without_issue_quote_is_relabelled() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                None,
            ),
            FIXTURE_ISSUE,
        );
        assert_eq!(test.intent, TestIntent::BehaviorPreservation);
        assert!(test.issue_quote.is_none());
        assert_eq!(event.unwrap().from_intent, TestIntent::FeatureAcceptance);
    }

    /// The exact failure the fixture matrix produced: a verbatim issue sentence
    /// that argues AGAINST the label, accepted by presence alone.
    #[test]
    fn verbatim_quote_that_does_not_name_the_asserted_input_is_relabelled() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some(
                    "Please fix that boundary bug without changing the existing behavior at the other thresholds.",
                ),
            ),
            FIXTURE_ISSUE,
        );
        assert_eq!(
            test.intent,
            TestIntent::BehaviorPreservation,
            "a quote about another threshold must not authorise the 500g behaviour"
        );
        let event = event.expect("relabel must be recorded");
        assert!(
            event.reason.contains("does not mention any asserted input"),
            "got {}",
            event.reason
        );
    }

    #[test]
    fn quote_naming_a_neighbouring_input_is_not_enough() {
        let (test, _) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some("the exact `1000` gram edge falls into the wrong band."),
            ),
            FIXTURE_ISSUE,
        );
        assert_eq!(test.intent, TestIntent::BehaviorPreservation);
    }

    /// A legitimate intent survives, even though the Attacker dropped the
    /// markdown backticks when quoting.
    #[test]
    fn quote_naming_the_asserted_input_keeps_intent_without_backticks() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::BugReproduction,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("shipping_band(1000) should return \"standard\", not \"heavy\"."),
            ),
            FIXTURE_ISSUE,
        );
        assert_eq!(test.intent, TestIntent::BugReproduction);
        assert!(event.is_none(), "got {:?}", event.map(|e| e.reason));
    }

    #[test]
    fn quote_naming_the_asserted_input_keeps_intent_with_backticks() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::BugReproduction,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("`shipping_band(1000)` should return `\"standard\"`, not `\"heavy\"`."),
            ),
            FIXTURE_ISSUE,
        );
        assert_eq!(test.intent, TestIntent::BugReproduction);
        assert!(event.is_none(), "got {:?}", event.map(|e| e.reason));
    }

    #[test]
    fn issue_quote_matching_normalizes_whitespace() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("shipping_band(1000)   should return  \"standard\""),
            ),
            "shipping_band(1000)\nshould return \"standard\" and not \"heavy\"",
        );
        assert_eq!(test.intent, TestIntent::FeatureAcceptance);
        assert!(event.is_none());
    }

    /// A test with no literal call arguments cannot be anchored; presence is the
    /// most the check can ask for, so it must not demote silently.
    #[test]
    fn unanchorable_test_falls_back_to_presence_check() {
        let (test, event) = relabel_test_to_issue_scope(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert compute() is not None\n",
                Some("A new compute helper must exist."),
            ),
            "A new compute helper must exist.",
        );
        assert_eq!(test.intent, TestIntent::FeatureAcceptance);
        assert!(event.is_none());
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
