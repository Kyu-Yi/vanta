// ============================================================================
// vanta_m1 — M1 one-shot pipeline: Creator → Attacker → Session Log
// ============================================================================
//
// M1 SCOPE (per Spec): agents + session log. NO execution, NO sandbox, NO BATE.
//
// Flow:
//   1. Read sample buggy Python file + issue from examples/buggy_python/
//   2. Creator agent → patch_text (Verify-First prompted)
//   3. Attacker agent → Vec<GeneratedTest> (each with declared intent)
//   4. SessionLog written to vanta_sessions/{session_id}.json
//
// M2 adds: sandbox execution + BATE baseline validation + Diagnostician.
// ============================================================================

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use vanta::agents::attacker::{AttackerRequest, AttackerResponse};
use vanta::agents::creator::{CreatorRequest, CreatorResponse, SourceFile};
use vanta::config::LlmConfig;
use vanta::session::{fresh_patch_attempt_id, fresh_session_id, SessionLog, SessionRecord};
use vanta::logging::init_logging;

// Path to the sample bug repo (relative to project root when run via cargo).
const BUGGY_DIR: &str = "examples/buggy_python";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M1 — Creator → Attacker → Session Log                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let cfg = LlmConfig::from_env()?;
    println!("{cfg}\n");

    // ---- 1. Load sample bug repo ----
    let cart_path = PathBuf::from(BUGGY_DIR).join("cart.py");
    let issue_path = PathBuf::from(BUGGY_DIR).join("issue.md");
    let cart_source = fs::read_to_string(&cart_path)?;
    let issue_text = fs::read_to_string(&issue_path)?;
    println!("▌ Loaded {} ({} bytes) + issue.md ({} bytes)",
        cart_path.display(), cart_source.len(), issue_text.len());

    let source_files = vec![SourceFile {
        path: "cart.py".to_string(),
        contents: cart_source.clone(),
    }];

    // ---- 2. Creator agent ----
    let session_id = fresh_session_id();
    let patch_attempt_id = fresh_patch_attempt_id();
    let mut session_log = SessionLog::new(session_id.clone());

    // [FORENSICS] Create session debug dir + point VANTA_DEBUG_DIR at it so
    // the LLM layer writes raw responses here: vanta_sessions/{id}/debug/
    let debug_dir = session_log.ensure_debug_dir(std::path::Path::new("vanta_sessions"))?;
    // SAFETY: single-threaded CLI; no other thread reads env concurrently.
    unsafe { std::env::set_var("VANTA_DEBUG_DIR", &debug_dir); }
    println!("▌ Forensics dir → {}", debug_dir.display());

    println!("\n▌ CREATOR phase — generating patch...");
    let creator_request = CreatorRequest {
        issue_text: issue_text.clone(),
        source_files,
        prior_feedback: None,
    };
    let creator_response: CreatorResponse =
        vanta::agents::creator::creator_generate_patch(&cfg, &creator_request).await?;

    println!("  ✔ Creator done — patch_text={} chars, files_touched={:?}, confidence={}",
        creator_response.patch_text.len(),
        creator_response.files_touched,
        creator_response.confidence);

    session_log.push(SessionRecord {
        session_id: session_id.clone(),
        patch_attempt_id: Some(patch_attempt_id.clone()),
        testing_iteration_id: None,
        test_id: None,
        phase: "creator".to_string(),
        summary: format!("patch_text={} chars, confidence={}",
            creator_response.patch_text.len(), creator_response.confidence),
        timestamp: now_iso(),
        benchmark_instance: None,
        benchmark_label: None,
    });

    // ---- 3. Attacker agent ----
    println!("\n▌ ATTACKER phase — generating adversarial tests...");
    let attacker_request = AttackerRequest {
        patch_text: creator_response.patch_text.clone(),
        original_source: cart_source,
        issue_text: issue_text.clone(),
        // M1 is a single-shot demo: no previous rounds, no shared set.
        existing_tests: Vec::new(),
    };
    let attacker_response: AttackerResponse =
        vanta::agents::attacker::attacker_generate_tests(&cfg, &attacker_request).await?;

    println!("  ✔ Attacker done — {} tests generated:", attacker_response.tests.len());
    for (i, t) in attacker_response.tests.iter().enumerate() {
        println!("    [{:>2}] {} (intent={:?}) — {}", i + 1, t.test_name, t.intent, t.reason);
        session_log.push(SessionRecord {
            session_id: session_id.clone(),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: None,
            test_id: None,
            phase: format!("attacker.test[{}]", t.test_name),
            summary: format!("intent={:?}, reason={}", t.intent, t.reason),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        });
    }

    // ---- 4. Write session log to disk ----
    let log_path = session_log.write_to_disk(std::path::Path::new("vanta_sessions"))?;
    println!("\n▌ Session log written → {}", log_path.display());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  M1 PIPELINE COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}

fn now_iso() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("{secs}")
}
