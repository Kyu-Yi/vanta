// ============================================================================
// LLM MODULE — shared HTTP + structured-output cleaning helpers
// ============================================================================
//
// These helpers were PROMOTED from bootstrap/mod.rs (M0 spike) into a shared
// module so that Creator (agents/creator.rs), Attacker (agents/attacker.rs),
// and the M0 spike (bootstrap/mod.rs) can ALL reuse them without duplication.
//
// WHAT YOU'LL LEARN HERE
//
//   # | Rust concept                  → Where Vanta reuses it later
//  ---+-------------------------------+---------------------------------------
//   1 | reqwest::Client + json!()     → Every LLM call in M1-M5 goes through
//     |   (HTTP POST JSON body)        →   post_chat() — single source of truth
//  ---+-------------------------------+---------------------------------------
//   2 | extract_first_valid_json      → Brace-balanced multi-candidate scanner
//     |   (robust JSON extraction)    →   + serde try-parse on EACH candidate.
//     |                           →   The #1 robustness pattern for small models.
//  ---+-------------------------------+---------------------------------------
//   3 | structured_call_with_retry<T> → N-attempt retry loop with corrective
//     |   (generic async retry)       →   REPROMPT tails. p=0.5 → 3 attm = 87.5%.
// ============================================================================

use crate::config::LlmConfig;
use serde_json::{json, Value};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, trace, warn};

// Re-export the agent response types so bootstrap can use them too.
pub use crate::agents::attacker::{AttackerResponse, AttackerTestCase};
pub use crate::agents::creator::CreatorResponse;

// ============================================================================
// FORENSICS: write raw LLM responses + parsed JSON to disk for debugging
// ============================================================================
//
// [VANTA-WHY] User explicitly requested "extensive debugging and logging".
// When VANTA_DEBUG_DIR is set, every structured_call_with_retry attempt writes:
//   {debug_dir}/{type_name}_attempt_{n}_raw_response.txt   — raw model output
//   {debug_dir}/{type_name}_attempt_{n}_parsed.json        — parsed typed result (on success)
// This lets us retroactively inspect what the model returned when a parse fails,
// which is invaluable for linking new additions and tracking issues.

fn debug_dir() -> Option<PathBuf> {
    std::env::var("VANTA_DEBUG_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
}

fn write_forensics(type_name: &str, attempt: u32, suffix: &str, content: &str) {
    let Some(dir) = debug_dir() else { return };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!(err = %e, "forensics: failed to create debug dir");
        return;
    }
    let filename = format!("{type_name}_attempt_{attempt}_{suffix}");
    let path = dir.join(filename);
    match std::fs::write(&path, content) {
        Ok(_) => debug!(path = %path.display(), "forensics written"),
        Err(e) => warn!(err = %e, path = %path.display(), "forensics write failed"),
    }
}

// ============================================================================
// HELPER: build_reqwest_client(&cfg) → reqwest::Client
// ============================================================================
pub fn build_reqwest_client(cfg: &LlmConfig) -> Result<reqwest::Client, reqwest::Error> {
    use std::time::Duration;
    let mut headers = reqwest::header::HeaderMap::new();

    if let Some(key) = &cfg.api_key {
        use reqwest::header::{HeaderValue, AUTHORIZATION};
        let mut val = HeaderValue::from_str(&format!("Bearer {key}"))
            .expect("API key contains non-ASCII bytes");
        val.set_sensitive(true);
        headers.insert(AUTHORIZATION, val);
    }

    reqwest::Client::builder()
        .timeout(Duration::from_secs(cfg.request_timeout_secs))
        .default_headers(headers)
        .build()
}

// ============================================================================
// HELPER: strip_markdown_fences(&str) -> &str
// ============================================================================
pub fn strip_markdown_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s).trim_start();
    let s = s.strip_prefix("```").unwrap_or(s).trim_start();
    let s = s.strip_suffix("```").unwrap_or(s).trim_end();
    s
}

const CHAT_OVERHEAD_TOKENS: u32 = 256;

fn render_messages_for_token_estimate(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|message| {
            let role = message["role"].as_str().unwrap_or("unknown");
            let content = message["content"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| message["content"].to_string());
            format!("<{role}>\n{content}\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn conservative_token_estimate(text: &str) -> u32 {
    let bytes = text.as_bytes().len() as u32;
    if bytes == 0 { 0 } else { bytes / 3 + u32::from(bytes % 3 != 0) }
}

fn context_budget_exceeded(
    prompt_tokens: u32,
    requested_output_tokens: u32,
    context_window_tokens: u32,
) -> bool {
    prompt_tokens
        .saturating_add(requested_output_tokens)
        .saturating_add(CHAT_OVERHEAD_TOKENS)
        > context_window_tokens
}

fn tokenize_url_candidates(cfg: &LlmConfig) -> Vec<String> {
    let base = cfg.base_url.trim_end_matches('/');
    let mut urls = Vec::new();
    if let Some(root) = base.strip_suffix("/v1") {
        urls.push(format!("{root}/tokenize"));
    }
    urls.push(format!("{base}/tokenize"));
    urls.dedup();
    urls
}

async fn tokenize_via_server(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    text: &str,
) -> Option<u32> {
    for url in tokenize_url_candidates(cfg) {
        let Ok(resp) = client
            .post(&url)
            .json(&json!({ "content": text }))
            .send()
            .await
        else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.json::<Value>().await else {
            continue;
        };
        if let Some(tokens) = body["tokens"].as_array() {
            return Some(tokens.len() as u32);
        }
        if let Some(count) = body["count"].as_u64() {
            return Some(count as u32);
        }
        if let Some(count) = body["token_count"].as_u64() {
            return Some(count as u32);
        }
    }
    None
}

// ============================================================================
// HELPER: extract_first_json_object(&str) -> Option<&str> (kept for pedagogy)
// ============================================================================
#[allow(dead_code)]
pub fn extract_first_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut prev_was_backslash = false;
    let mut end_idx: Option<usize> = None;

    for (i, &b) in bytes[start..].iter().enumerate() {
        let abs_idx = start + i;
        match b {
            b'"' if !prev_was_backslash => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 { end_idx = Some(abs_idx); break; }
            }
            _ => {}
        }
        prev_was_backslash = b == b'\\' && !prev_was_backslash;
    }
    let end = end_idx?;
    Some(&s[start..=end])
}

// ============================================================================
// HELPER: extract_first_valid_json(&str) -> Option<&str>
//   Iterate ALL balanced brace pairs, try serde_json parse on each,
//   return first that parses. Solves "Rust code block in prose" failure.
// ============================================================================
pub fn extract_first_valid_json(s: &str) -> Option<&str> {
    let mut cursor: usize = 0;
    let s_bytes = s.as_bytes();
    let mut attempts: u32 = 0;
    const MAX_ATTEMPTS: u32 = 200;

    while cursor < s_bytes.len() && attempts < MAX_ATTEMPTS {
        attempts += 1;
        let rel_start = match s[cursor..].find('{') {
            Some(i) => i,
            None => break,
        };
        let abs_start = cursor + rel_start;

        let bytes = s.as_bytes();
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut prev_was_backslash = false;
        let mut abs_end: Option<usize> = None;

        for (i, &b) in bytes[abs_start..].iter().enumerate() {
            let abs = abs_start + i;
            match b {
                b'"' if !prev_was_backslash => in_string = !in_string,
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth == 0 { abs_end = Some(abs); break; }
                }
                _ => {}
            }
            prev_was_backslash = b == b'\\' && !prev_was_backslash;
        }

        let end = match abs_end {
            Some(e) => e,
            None => break,
        };

        let candidate = &s[abs_start..=end];
        let parse_result: Result<serde_json::Value, _> = serde_json::from_str(candidate);
        match parse_result {
            Ok(_v) => return Some(candidate),
            Err(_) => { cursor = end + 1; }
        }
    }
    None
}

// ============================================================================
// HELPER: extract_json_candidate(raw: &str) -> Result<&str, String>
//   4-stage pipeline: fences → prose scan → fallthrough → sanity.
// ============================================================================
pub fn extract_json_candidate<'a>(raw: &'a str) -> Result<&'a str, String> {
    let stage1 = strip_markdown_fences(raw);
    let stage2: &str = if let Some(json_slice) = extract_first_valid_json(stage1) {
        json_slice
    } else if stage1.starts_with('{') && stage1.ends_with('}') {
        stage1
    } else {
        return Err(format!(
            "NO valid JSON found across all brace-balanced candidates. starts={:?}, len={}",
            stage1.chars().take(200).collect::<String>(),
            stage1.len()
        ));
    };
    if stage2.is_empty() {
        return Err("EMPTY after all cleaning stages".into());
    }
    if !stage2.starts_with('{') || !stage2.ends_with('}') {
        return Err(format!(
            "candidate not JSON object (starts={:?}, ends={:?})",
            stage2.chars().next(), stage2.chars().last()
        ));
    }
    Ok(stage2)
}

// ============================================================================
// HELPER: post_chat(&client, &cfg, messages, extra_fields) -> Result<String>
//   Returns assistant content string (fusion: content + reasoning_content).
// ============================================================================
pub async fn post_chat(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    messages: Vec<Value>,
    extra_fields: Value,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let url = cfg.chat_completions_url();

    let body: Value = json!({
        "model": cfg.model_id,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 0.2,
    });

    let mut full_body = body.as_object().cloned().unwrap_or_default();
    if let Some(obj) = extra_fields.as_object() {
        for (k, v) in obj {
            full_body.insert(k.clone(), v.clone());
        }
    }
    let full_body = Value::Object(full_body);

    let requested_output_tokens = full_body["max_tokens"].as_u64().unwrap_or(16_384) as u32;
    let prompt_render = render_messages_for_token_estimate(&messages);
    let estimate = conservative_token_estimate(&prompt_render);
    let mut prompt_tokens = estimate;
    let mut token_source = "estimate";
    // [VANTA-WHY] Ask the server for an exact count only when the conservative
    // estimate is already about to reject. The /tokenize probe re-sends the whole
    // prompt — exactly the cost the narrowed review context exists to avoid — so
    // it must never run on the common path. This keeps the guard fail-closed
    // (the estimate over-counts) while still rescuing a borderline-but-fitting
    // request that the estimate would have refused.
    if context_budget_exceeded(estimate, requested_output_tokens, cfg.context_window_tokens) {
        if let Some(exact) = tokenize_via_server(client, cfg, &prompt_render).await {
            prompt_tokens = exact;
            token_source = "llama-server";
        }
    }
    info!(
        prompt_tokens,
        requested_output_tokens,
        context_window_tokens = cfg.context_window_tokens,
        token_source,
        "llm prompt budget"
    );
    if context_budget_exceeded(
        prompt_tokens,
        requested_output_tokens,
        cfg.context_window_tokens,
    ) {
        return Err(format!(
            "Prompt would exceed model context: prompt_tokens={} + max_tokens={} + overhead={} > context_window={}",
            prompt_tokens,
            requested_output_tokens,
            CHAT_OVERHEAD_TOKENS,
            cfg.context_window_tokens,
        )
        .into());
    }

    let t = Instant::now();
    let resp = client.post(&url).json(&full_body).send().await
        .map_err(|e| format!("HTTP POST to {url} FAILED: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        let txt = resp.text().await.unwrap_or_else(|_| String::from("<could not read body>"));
        return Err(format!(
            "HTTP ERROR {code} from {url}. Body snippet: {}",
            &txt[..300.min(txt.len())]
        ).into());
    }

    let json_resp: Value = resp.json().await
        .map_err(|e| format!("FAILED to parse HTTP response as JSON: {e}"))?;

    let choices = json_resp["choices"].as_array()
        .ok_or("response JSON missing 'choices' array")?;
    let first_choice = choices.first()
        .ok_or("choices array was empty")?;
    let msg = &first_choice["message"];
    let content = msg["content"].as_str().unwrap_or("").trim().to_string();
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("").trim().to_string();

    if content.is_empty() && reasoning.is_empty() {
        return Err("Assistant response: BOTH content AND reasoning_content empty".into());
    }

    debug!(content_len = content.len(), reasoning_len = reasoning.len(), "model response received");
    trace!(content = %content.chars().take(500).collect::<String>(), "model content");
    trace!(reasoning = %reasoning.chars().take(500).collect::<String>(), "model reasoning");

    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create("/tmp/vanta_last_model_output.txt") {
            let _ = writeln!(f, "=== CONTENT ({len_c}) ===\n{content}\n\n=== REASONING ({len_r}) ===\n{reasoning}\n",
                len_c = content.len(), len_r = reasoning.len());
        }
    }

    let combined = format!(
        "===CONTENT===\n{}\n===END_CONTENT===\n\n===REASONING===\n{}\n===END_REASONING===",
        content, reasoning
    );

    let ms = t.elapsed().as_millis();
    let tok_total = json_resp["usage"]["total_tokens"].as_u64().unwrap_or(0);
    debug!(status = status.as_u16(), elapsed_ms = ms, total_tokens = tok_total, "http post complete");

    Ok(combined)
}

// ============================================================================
// HELPER: structured_call_with_retry<T> — N-attempt retry + typed parse
// ============================================================================
pub async fn structured_call_with_retry<Fm, Fp, T>(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    max_attempts: u32,
    mut build_messages: Fm,
    extras_for_attempt: Value,
    mut parse_into_typed: Fp,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
where
    Fm: FnMut(u32) -> Vec<Value>,
    Fp: FnMut(&str) -> Result<T, String>,
    T: std::fmt::Debug + serde::Serialize,
{
    let mut last_err: Option<String> = None;

    for attempt in 1..=max_attempts {
        let messages = build_messages(attempt);
        if attempt > 1 {
            warn!(attempt, max_attempts, "retrying structured call");
        }

        let raw = match post_chat(client, cfg, messages, extras_for_attempt.clone()).await {
            Ok(r) => r,
            Err(http_e) => {
                let msg = format!("attempt {attempt} HTTP err: {http_e}");
                warn!(%msg);
                last_err = Some(msg);
                continue;
            }
        };

        let type_name = std::any::type_name::<T>().split("::").last().unwrap_or("T");

        // [FORENSICS] Write the raw model response to disk before parsing,
        // so we can inspect exactly what the model returned if parsing fails.
        write_forensics(type_name, attempt, "raw_response.txt", &raw);

        match parse_into_typed(&raw) {
            Ok(typed) => {
                debug!(attempt, type_name, "parsed typed response");
                // [FORENSICS] Write the successfully parsed typed result as JSON.
                if let Ok(json) = serde_json::to_string_pretty(&serde_json::to_value(&typed).unwrap_or(json!(null))) {
                    write_forensics(type_name, attempt, "parsed.json", &json);
                }
                return Ok(typed);
            }
            Err(parse_err) => {
                let preview: String = raw.chars().take(240).collect();
                last_err = Some(format!(
                    "attempt {attempt} PARSE ERR: {}\n  model-preview: {preview}",
                    parse_err
                ));
            }
        }
    }

    Err(format!(
        "structured_call_with_retry all {} attempts FAILED. Last:\n{}",
        max_attempts,
        last_err.as_deref().unwrap_or("(no details)")
    ).into())
}

#[cfg(test)]
mod _cleaning_tests {
    use super::{
        context_budget_exceeded, conservative_token_estimate, extract_first_json_object,
        extract_json_candidate, strip_markdown_fences,
    };

    #[test]
    fn strips_all_fence_variants() {
        assert_eq!(strip_markdown_fences("```json\n{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("```{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("{}\n"), "{}");
    }

    #[test]
    fn extracts_plain_json() {
        assert_eq!(extract_first_json_object("{\"a\":1}"), Some("{\"a\":1}"));
    }

    #[test]
    fn extracts_json_after_prose() {
        let s = "Sure! Here you go:\n{\"x\":\"hello\",\"y\":[1,2]}";
        assert_eq!(extract_first_json_object(s), Some("{\"x\":\"hello\",\"y\":[1,2]}"));
    }

    #[test]
    fn pipeline_handles_prose_then_fenced_json() {
        let s = "Okay let's see...\n```json\n{\"ok\":true}\n```\nHope that helps";
        let r = extract_json_candidate(s).unwrap();
        assert_eq!(r, "{\"ok\":true}");
    }

    #[test]
    fn pipeline_handles_plain_json_no_wrapper() {
        let s = "{\"a\": 1, \"b\": [1,2,3]}";
        let r = extract_json_candidate(s).unwrap();
        assert_eq!(r, s);
    }

    #[test]
    fn pipeline_rejects_non_object() {
        assert!(extract_json_candidate("just text").is_err());
    }

    #[test]
    fn conservative_token_estimate_grows_with_input() {
        assert!(
            conservative_token_estimate("a b c d e f g h")
                > conservative_token_estimate("a b")
        );
    }

    #[test]
    fn context_budget_exceeded_counts_prompt_output_and_overhead() {
        assert!(context_budget_exceeded(60_000, 6_000, 65_536));
        assert!(!context_budget_exceeded(20_000, 2_000, 65_536));
    }
}
