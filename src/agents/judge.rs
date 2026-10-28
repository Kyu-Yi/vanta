// ============================================================================
// JUDGE AGENT — issue + context + candidate patch → verdict (M3 item 7)
// ============================================================================
//
// [VANTA-WHY] M3 item 7. The judge arm replaces the whole adversarial pipeline
// with ONE model call that reads the same material the `vanta` arm's roles read
// and answers a single question: does this candidate resolve the issue? It is
// the study's "LLM-as-judge" comparison against the test-driven arms, so it is
// deliberately NOT a pass/fail test run — no tool use, no execution — and its
// verdict is recorded under its own tokens (`JudgeApproved`/`JudgeRejected`/
// `JudgeFailed`) rather than borrowed from the test-driven ones, so the log
// always says which mechanism reached the answer.
//
// [VANTA-WHY] The judge is shown the SAME context selection and budget the
// `vanta` review arm is (plans/M3-plan.md item 7's fairness rule): the driver
// renders `select_review_prompt`'s output, the same value the Attacker, the
// auditor, BATE and the Diagnostician are shown, and passes it here verbatim.
// Sampling matches too — the same model, the same `reasoning_effort` the other
// call sites send, and the `judge` site's pins in `SAMPLING_SCHEDULE`.

use crate::config::LlmConfig;
use crate::llm::{build_reqwest_client, extract_json_candidate, structured_call_with_retry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, trace};

// ============================================================================
// JUDGE_PROMPT_VERSION — the run-identity tag for the judge's prompt
// ============================================================================
//
// [VANTA-WHY] The plan requires the judge's prompt text to be versioned and
// hashed, exactly as the other prompts are, so a wording change is visible in
// the log instead of looking like "no change". Bump this whenever the system or
// user prompt text below changes; the content hash
// (`core::prompt_content_hash`) is the belt to this version tag's braces.
pub const JUDGE_PROMPT_VERSION: &str = "1";

// ============================================================================
// JUDGE_SYSTEM_PROMPT — the judging instruction, as a const
// ============================================================================
//
// [VANTA-WHY] Hoisted out of the call site for the same reason `CREATOR_SYSTEM_PROMPT`
// is: the hash must be taken over the text the model ACTUALLY sees, so "the
// prompt the call sends" and "the prompt the hash covers" are one value by
// construction.
pub const JUDGE_SYSTEM_PROMPT: &str = r#"You are Vanta Judge. You are given an issue, the repository context a developer would need to understand the change, and a candidate patch. Decide whether the candidate patch correctly resolves the issue WITHOUT breaking existing behaviour.

You are the ONLY judge. No test is run and you must not run anything: read the issue and the diff and reason about them. Judge the patch on its own merits — do not assume the patch author is right.

YOU MUST output valid JSON. NO EXCEPTIONS.
  1. First character of content MUST be '{'
  2. Last character MUST be the matching '}'
  3. No prose, no markdown fences
  4. Do not stop until the full '}' is written

Required JSON schema:
{
  "verdict":    "approve" | "reject",
  "confidence": number 0.0 to 1.0,
  "reason":     string — 1 to 3 sentences naming what in the diff decides it
}

Answer "approve" only when the diff genuinely fixes the issue and does not regress existing behaviour. When you are not sure, answer "reject"."#;

/// The judging prompt the call site sends.
///
/// Routed through an accessor so the hash is taken over the exact text the model
/// receives (see `creator_system_prompt` for the incident that makes that worth
/// pinning).
pub fn judge_system_prompt() -> &'static str {
    JUDGE_SYSTEM_PROMPT
}

// ============================================================================
// JudgeVerdict — the typed verdict
// ============================================================================
//
// [VANTA-WHY] An enum rather than a free string, so an unknown `verdict` value
// is a PARSE error and takes the retry path: "the model said something we cannot
// read as a verdict" is a different fact from "the model approved", and folding
// the first into either verdict would record a decision nobody made. Two failed
// attempts end the run as `JudgeFailed`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JudgeVerdict {
    Approve,
    Reject,
}

// ============================================================================
// JudgeRequest / JudgeResponse — the typed I/O of the judge agent
// ============================================================================

/// Everything the judge is shown. Every field is a value the driver already
/// built for the `vanta` arm, so the fairness rule (item 7) is enforced by the
/// shape: there is no field here the test-driven arm did not also have.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeRequest {
    pub issue_text: String,
    /// The rendered `select_review_prompt` selection — the same string the
    /// `vanta` review arm's roles are shown.
    pub context: String,
    /// The candidate patch (the diff under review).
    pub candidate_diff: String,
    /// The native test suite's result on the candidate, under `judge+suite`
    /// only: which tests failed, with the pass/fail summary. `None` under the
    /// plain judge, where no suite is run.
    pub suite_result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeResponse {
    pub verdict: JudgeVerdict,
    pub confidence: f64,
    pub reason: String,
}

// ============================================================================
// JudgeError — a wrong ANSWER versus a broken CALL
// ============================================================================
//
// [VANTA-WHY] M3 item 7 — the two failures need opposite handling, so they are
// two types rather than one string. "The judge answered and its answer cannot be
// read as a verdict" is the judge arm's own outcome (`JudgeFailed`, its own exit
// code and its own bucket, counted and never retried as infrastructure). "The
// call never produced an answer at all" is every other call site's failure and
// ends the run as `InfraFailure`. Folding them together was the alternative: it
// would either hide a genuinely broken candidate behind `infra_failure`, or
// report a dead server as a verdict about the patch.
#[derive(Debug)]
pub enum JudgeError {
    /// The judge did not answer with a readable verdict — malformed JSON after
    /// the one retry, or an answer the server cut off at the cap so it was never
    /// handed to the parser. This is item 7's `JudgeFailed`.
    Malformed(String),
    /// The call itself failed — an HTTP error, the run's deadline, a prompt that
    /// cannot fit the window. No verdict was ever attempted.
    Call(String),
}

impl JudgeError {
    /// The message, whichever kind it is, for the session log.
    pub fn message(&self) -> &str {
        match self {
            JudgeError::Malformed(m) | JudgeError::Call(m) => m,
        }
    }
}

/// Did a `structured_call_with_retry` failure come from an answer the model DID
/// give, rather than from the call never completing?
///
/// [VANTA-WHY] M3 item 7 — the two markers the LLM layer writes are the whole
/// distinction: `PARSE ERR` is an answer that arrived and could not be read, and
/// `TRUNCATED` is an answer the server stopped mid-way (also not a verdict). An
/// HTTP error, the deadline marker and the context-overflow refusal carry
/// neither, so they stay infrastructure. This is a reading of the LLM layer's own
/// wording, kept here rather than in the driver so the classification lives
/// beside the prompt it is about.
fn is_unreadable_answer(message: &str) -> bool {
    message.contains("PARSE ERR") || message.contains("TRUNCATED")
}

// ============================================================================
// judge_candidate() — call the LLM once (plus one retry), get a typed verdict
// ============================================================================
//
// [VANTA-WHY] `max_attempts: 2`, not `cfg.retry_attempts`: the plan fixes the
// judge at "one call per verdict, plus one retry if the JSON is malformed". A
// third attempt would make the judge arm's call count depend on the invocation
// rather than on the arm, which is exactly what the arm registry exists to
// prevent.
pub async fn judge_candidate(
    cfg: &LlmConfig,
    request: &JudgeRequest,
) -> Result<JudgeResponse, JudgeError> {
    info!(
        issue_len = request.issue_text.len(),
        context_len = request.context.len(),
        diff_len = request.candidate_diff.len(),
        with_suite = request.suite_result.is_some(),
        "judge start"
    );

    let client = build_reqwest_client(cfg)
        .map_err(|e| JudgeError::Call(format!("could not build the HTTP client: {e}")))?;

    let system_prompt = judge_system_prompt();

    let mut user_parts: Vec<String> = Vec::new();
    user_parts.push(format!("ISSUE:\n{}\n", request.issue_text));
    user_parts.push(format!(
        "REPOSITORY CONTEXT (the same selection the test-driven arm is shown):\n{}\n",
        request.context
    ));
    user_parts.push(format!("CANDIDATE PATCH:\n{}\n", request.candidate_diff));
    if let Some(suite) = &request.suite_result {
        user_parts.push(format!(
            "NATIVE TEST SUITE RESULT ON THE CANDIDATE:\n{}\n",
            suite
        ));
    }
    user_parts.push(
        "Decide whether this patch correctly resolves the issue without breaking existing \
         behaviour. Answer with the JSON object described in your instructions."
            .to_string(),
    );
    let user_prompt = user_parts.join("\n");

    debug!(
        system_len = system_prompt.len(),
        user_len = user_prompt.len(),
        "judge prompts built"
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = "REPROMPT (attempt #2): Last response was not valid JSON or the \
                        `verdict` was not \"approve\" or \"reject\". Write ONLY a JSON object \
                        beginning with '{' and ending with '}'. Keys: verdict (\"approve\" or \
                        \"reject\"), confidence (0.0-1.0), reason."
                .to_string();
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    // [VANTA-WHY] The same `reasoning_effort` every other call site sends, and
    // the `judge` site's own pins from `SAMPLING_SCHEDULE` — the fairness rule
    // item 7 states (same model, effort and sampling as the arm it is compared
    // with).
    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "judge",
    );

    let parse_closure = |raw: &str| -> Result<JudgeResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        trace!(json = %cleaned.chars().take(300).collect::<String>(), "judge extracted json");
        serde_json::from_str::<JudgeResponse>(cleaned).map_err(|e| {
            format!("JUDGE PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: JudgeResponse = structured_call_with_retry(
        &client, cfg, 2, "", "judge", build_messages, extras, parse_closure
    )
    .await
    .map_err(|e| {
        let message = e.to_string();
        if is_unreadable_answer(&message) {
            JudgeError::Malformed(message)
        } else {
            JudgeError::Call(message)
        }
    })?;

    info!(
        verdict = ?typed.verdict,
        confidence = typed.confidence,
        "judge done"
    );

    Ok(typed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verdict enum is the parse gate: an unknown value must not deserialize
    /// into a verdict, or a malformed answer would be recorded as a decision the
    /// model never made.
    #[test]
    fn an_unknown_verdict_is_not_a_judge_response() {
        let ok: JudgeResponse =
            serde_json::from_str(r#"{"verdict":"approve","confidence":0.9,"reason":"r"}"#).unwrap();
        assert_eq!(ok.verdict, JudgeVerdict::Approve);
        let reject: JudgeResponse =
            serde_json::from_str(r#"{"verdict":"reject","confidence":0.1,"reason":"r"}"#).unwrap();
        assert_eq!(reject.verdict, JudgeVerdict::Reject);
        assert!(
            serde_json::from_str::<JudgeResponse>(
                r#"{"verdict":"maybe","confidence":0.5,"reason":"r"}"#
            )
            .is_err(),
            "an unknown verdict must fail to parse so it takes the retry path"
        );
    }

    /// The judge site must be in the schedule, or `sampling_pin_table` panics when
    /// the request body is built.
    #[test]
    fn the_judge_site_is_in_the_sampling_schedule() {
        assert!(
            crate::llm::SAMPLING_SCHEDULE.iter().any(|(site, ..)| *site == "judge"),
            "`judge` must be a site in SAMPLING_SCHEDULE"
        );
    }
}
