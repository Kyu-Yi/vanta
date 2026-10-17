// ============================================================================
// BOOTSTRAP MODULE — M0 spike launcher (kept for regression verification)
// ============================================================================
//
// M0 SPIKE DELIVERABLES:
//   m1_single_call     — single unstructured chat completion
//   m2_structured_json — structured typed CreatorResponse JSON + malformed handling
//   m3_two_agent_chain — Creator → Attacker chain
//
// NOTE: All helper functions (post_chat, extract_first_valid_json,
// structured_call_with_retry, build_reqwest_client) and agent response types
// (CreatorResponse, AttackerResponse) were PROMOTED to crate::llm and
// crate::agents in M1. This module now imports them — no local duplication.
// ============================================================================

use crate::agents::attacker::AttackerResponse;
use crate::agents::creator::CreatorResponse;
use crate::config::LlmConfig;
use crate::llm::{
    build_reqwest_client, extract_json_candidate, post_chat, structured_call_with_retry,
};
use serde_json::json;
use std::error::Error;

// ============================================================================
// run() — entry point. Sequentially runs m1 → m2 → m3 via ? chain.
// ============================================================================
pub async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M0 SPIKE — local llama-server via reqwest             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let cfg: LlmConfig = LlmConfig::from_env()?;
    println!("┌─ Resolved LLM config ───────────────────────────────────────┐");
    println!("{cfg}");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("▌ M1 START — single unstructured chat completion");
    let m1_out = m1_single_call(&cfg).await?;
    println!("  ✔ M1 OK — preview = {m1_out:?}");
    println!("▌ M1 END\n");

    println!("▌ M2 START — structured JSON → typed CreatorResponse");
    let m2_out = m2_structured_json(&cfg).await?;
    println!("  ✔ M2 OK — files_touched = {:?}, confidence = {}",
        m2_out.files_touched, m2_out.confidence);
    println!("▌ M2 END\n");

    println!("▌ M3 START — Creator → Attacker chained via ? operator");
    let m3_out = m3_two_agent_chain(&cfg).await?;
    println!("  ✔ M3 OK — attacker wrote {} test cases", m3_out.tests.len());
    for (i, t) in m3_out.tests.iter().enumerate() {
        println!("     [{:>2}] {} (intent={:?})", i + 1, t.test_name, t.intent);
    }
    println!("▌ M3 END\n");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  M0 SPIKE: ALL 3 MILESTONES PASSED ✔                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}

// ============================================================================
// MILESTONE 1 (m1_single_call)
// ============================================================================
pub async fn m1_single_call(cfg: &LlmConfig) -> Result<String, Box<dyn Error + Send + Sync>> {
    let client = build_reqwest_client(cfg)?;
    let messages = vec![
        json!({"role": "system", "content": "You are a concise assistant. Answer in 1 sentence max."}),
        json!({"role": "user", "content": "In one short sentence, explain what a Result<T,E> is in Rust."}),
    ];
    let text = post_chat(&client, cfg, messages, json!({})).await?;
    let preview = text.chars().take(120).collect::<String>();
    if preview.trim().is_empty() {
        return Err("m1_single_call: model returned EMPTY response".into());
    }
    Ok(preview)
}

// ============================================================================
// MILESTONE 2 (m2_structured_json) — structured JSON → typed CreatorResponse
// ============================================================================
pub async fn m2_structured_json(cfg: &LlmConfig)
    -> Result<CreatorResponse, Box<dyn Error + Send + Sync>>
{
    let client = build_reqwest_client(cfg)?;

    let system_prompt = r#"You are Creator. YOU MUST output valid JSON. NO EXCEPTIONS.

INSTRUCTIONS YOU MUST OBEY:
  1. Do NOT write prose or thinking in your final answer. Thinking goes in reasoning_content ONLY.
  2. In content field: START WRITING WITH '{'. First character MUST be '{'.
  3. Write EXACTLY one JSON object. Last character MUST be matching '}'.
  4. Do NOT wrap JSON in fences.
  5. Do NOT stop until full '}' is written.

Required JSON schema:
{
  "patch_text":     string — git diff patch,
  "explanation":    string — 1 sentence,
  "files_touched":  array of strings,
  "confidence":     number 0.0 to 1.0
}

Example:
{"patch_text":"--- a/src/add.rs\n+++ b/src/add.rs\n@@ -1 +1 @@\n-pub fn add(a:u8,b:u8)->u8{a+b}\n+pub fn add(a:u8,b:u8)->u8{a.checked_add(b).unwrap_or(u8::MAX)}","explanation":"Replaced plain add with checked_add.","files_touched":["src/add.rs"],"confidence":0.94}

REMEMBER: begin '{' end '}'."#.to_string();

    let user_prompt_base = "Please patch this tiny buggy function:
```rust
// src/discount.rs
pub fn apply_discount(price_cents: u32, pct: u8) -> u32 {
    price_cents * (100 - pct as u32) / 100
}
```
Write CreatorResponse JSON.";

    let user_base_owned = user_prompt_base.to_string();
    let sys_owned = system_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_base_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY a valid JSON object beginning with '{{' and ending with '}}'. Keys: patch_text, explanation, files_touched, confidence.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2400,
        "seed": 42,
    });

    let parse_closure = |raw: &str| -> Result<CreatorResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<CreatorResponse>(cleaned).map_err(|e| {
            format!("M2 PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: CreatorResponse = structured_call_with_retry(
        &client, cfg, 3, build_messages, extras, parse_closure
    ).await?;

    // ---- BUILT-IN MALFORMED CASE (M0 exit test: must NOT panic) ----
    let bad = r#"{ "patch_text": "x"  "explanation": "y", "files_touched": ["a"], "confidence": 0.5 }"#;
    match serde_json::from_str::<CreatorResponse>(bad) {
        Ok(_) => return Err("M2 MALFORMED TEST FAILED: bad JSON parsed OK".into()),
        Err(e) => {
            eprintln!("  ✔ M2 builtin malformed case caught (line {} col {}): {}",
                      e.line(), e.column(), e);
        }
    }

    if !(0.0..=1.0).contains(&typed.confidence) {
        return Err(format!("M2: confidence out of range [0,1]: {}", typed.confidence).into());
    }
    if typed.files_touched.is_empty() {
        return Err("M2: files_touched is empty".into());
    }
    Ok(typed)
}

// ============================================================================
// MILESTONE 3 (m3_two_agent_chain) — Creator → Attacker chain
// ============================================================================
pub async fn m3_two_agent_chain(cfg: &LlmConfig)
    -> Result<AttackerResponse, Box<dyn Error + Send + Sync>>
{
    let client = build_reqwest_client(cfg)?;

    // ---- CREATOR PHASE ----
    let creator_system = r#"You are Vanta Creator. YOU MUST output valid JSON. NO EXCEPTIONS.

INSTRUCTIONS YOU MUST OBEY:
  1. Thinking goes in reasoning_content ONLY.
  2. content field MUST start with '{'.
  3. One JSON object, ending with matching '}'.
  4. No fences, no prose.
  5. Do not stop until full '}' is written.

Required schema:
{ "patch_text": string, "explanation": string, "files_touched": [string], "confidence": number }

REMEMBER: begin '{' end '}'."#.to_string();

    let creator_user_base = r#"Patch this file to fix integer overflow. Return CreatorResponse JSON.

```rust
// src/cart.rs
pub fn cart_total(items: &[u32]) -> u32 {
    items.iter().sum()
}
```"#.to_string();

    let creator_sys_owned = creator_system.to_string();
    let creator_build_msgs = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": creator_sys_owned.clone()}),
            json!({"role": "user", "content": creator_user_base.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{attempt}): Write ONLY CreatorResponse JSON. Start '{{' end '}}'.",
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };
    let creator_extras = json!({
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2400,
        "seed": 42,
    });
    let creator_parse = |raw: &str| -> Result<CreatorResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<CreatorResponse>(cleaned)
            .map_err(|e| format!("M3 Creator PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e))
    };

    let creator: CreatorResponse = structured_call_with_retry(
        &client, cfg, 4, creator_build_msgs, creator_extras, creator_parse
    ).await?;

    let patch_text = creator.patch_text.clone();
    if patch_text.is_empty() {
        return Err("M3 Creator returned EMPTY patch_text".into());
    }
    eprintln!("  M3↳ Creator done — patch_text={} chars, files_touched={:?}",
        patch_text.len(), creator.files_touched);

    // ---- ATTACKER PHASE ----
    let attacker_system = r###"You are Vanta Attacker. YOU MUST output valid JSON. NO EXCEPTIONS.

INSTRUCTIONS:
  1. content starts with '{', ends with '}'.
  2. No prose, no fences.
  3. Do not stop until full '}'.

Required schema:
{
  "tests": [
    { "test_name": string, "test_source": string, "expect_pass": bool }
  ]
}

Write exactly 2 tests."###.to_string();

    let patch_preview: String = patch_text.chars().take(600).collect();
    let attacker_user_base = format!(
r#"Patch:
```diff
{patch_preview}
```
Write Attacker JSON with "tests" array of 2 items. Valid JSON only."#
    );

    let atk_sys = attacker_system.clone();
    let atk_usr = attacker_user_base.clone();
    let attacker_build_msgs = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": atk_sys.clone()}),
            json!({"role": "user", "content": atk_usr.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT attempt #{attempt}: Write ONLY JSON {{\"tests\":[...]}}. 2 tests.",
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };
    let attacker_extras = json!({
        "response_format": {"type": "json_object"},
        "max_tokens": 2400,
        "temperature": 0.15,
        "seed": 42,
    });
    let attacker_parse = |raw: &str| -> Result<AttackerResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<AttackerResponse>(cleaned)
            .map_err(|e| format!("M3 Attacker PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e))
    };

    let typed_attacker: AttackerResponse = structured_call_with_retry(
        &client, cfg, 4, attacker_build_msgs, attacker_extras, attacker_parse
    ).await?;

    if typed_attacker.tests.is_empty() {
        return Err("M3: Attacker wrote 0 tests".into());
    }
    Ok(typed_attacker)
}
