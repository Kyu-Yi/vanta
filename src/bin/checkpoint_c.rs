// ============================================================================
// CHECKPOINT C — Async/Await + Sequential vs Concurrent Timing with tokio::join!
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | #[tokio::main] runtime    → THE ENTRY POINT of every M1+ binary
//     |                           → 100% of I/O (HTTP, sandbox, file I/O) is async
//  ---+---------------------------+-------------------------------------------
//   2 | async fn + .await         → M1: three milestone calls chained via ?
//     |                           → M2: Phase 1 (BATE) + Phase 2 (exec) + log
//  ---+---------------------------+-------------------------------------------
//   3 | Sequential await pattern  → M1: m1_single_call → m2_structured → m3_chain
//     | (a.await; b.await;)       → Must run in ORDER because each depends on
//     |                           → previous outputs (two-agent chain NEEDS this)
//  ---+---------------------------+-------------------------------------------
//   4 | tokio::join!(a,b)         → M2: run 8 test suites in parallel, each
//     |                           →   400ms → total 500ms instead of 3200ms
//     |                           → M3: run 3 verifier backends (Formal +
//     |                           →   Fuzz + Static) concurrently on 1 patch
//  ---+---------------------------+-------------------------------------------
//   5 | Instant + elapsed timing  → Governor M4: measure EIR/ECR step times
//     |                           → Paper M5: report wall-clock per attempt
//     |                           → Timeout checks (if phase_2 > 300s → abort)
// ============================================================================

// ----------------------------------------------------------------------------
// [SYNTAX] use std::time::{Instant, Duration}; — bring timing primitives from
// the standard library.
//   Instant  = a monotonic (never goes backwards even if OS clock changes!)
//              timestamp. Perfect for measuring durations of code sections.
//   Duration = length of time with sub-nanosecond precision. Usually created
//              via Duration::from_millis(u64) or Duration::from_secs_f64(f64).
// ----------------------------------------------------------------------------
use std::time::{Duration, Instant};

// ----------------------------------------------------------------------------
// greet — async task that sleeps N milliseconds, then returns a greeting String
// ----------------------------------------------------------------------------
//
// [SYNTAX] async fn greet(delay_ms: u64, name: &str) -> String { ... }
//   `async fn` = declares an asynchronous function. Calling an async fn does
//   NOT execute the body. Instead, it returns a Future (conceptually: "a
//   suspended computation that will produce a String later when polled by
//   the runtime"). To actually RUN that future, you .await it somewhere
//   inside another async fn or block.
//
//   Conceptual expansion of async fn:
//     fn greet(args...) -> impl Future<Output = String> + Send + ... { async move { ... } }
//   But you write `async fn` and let the compiler handle it.
//
// [SYNTAX] tokio::time::sleep(duration).await; — yields control BACK TO THE
// TOKIO RUNTIME SCHEDULER for the requested duration. Key difference vs the
// SYNCHRONOUS equivalent std::thread::sleep(Duration::from_millis(N)):
//     std::thread::sleep → BLOCKS THE OS THREAD for N ms. Wastes the thread,
//       no other task can run on it. In a 8-thread runtime: 8 sleeps and you
//       have 0 threads left to do actual work.
//     tokio::time::sleep().await → YIELDS. The thread goes and runs OTHER
//       pending tasks while time passes. When the deadline elapses, this task
//       is rescheduled. This is the entire point of async.
//
// [VANTA-WHY] In M2, an Attacker writes 100 tests. If you run them:
//   Sequential  : 100 tests × 0.2s each = 20 WALL SECONDS
//   Concurrent  : 100 sleeps interleaved ≈ 0.4s WALL SECONDS (the long pole)
// Async concurrency on the single-threaded runtime already wins 50× for
// I/O-bound or sleep-bound work. Real sandbox execution is I/O-bound.
async fn greet(delay_ms: u64, name: &str) -> String {
    // [SYNTAX] eprintln! writes to STDERR (separate stream from STDOUT).
    // Good for debug timings / progress info so it doesn't get captured if
    // the binary's stdout is piped to a file.
    eprintln!("  [task spawn info] greet(delay={delay_ms}ms, name={name}) started. \
               Will sleep {delay_ms}ms then return String.");

    tokio::time::sleep(Duration::from_millis(delay_ms)).await;

    let result = format!("Hello, {name}! (after {delay_ms}ms)");
    eprintln!("  [task done] greet({delay_ms}ms, {name}) → \"{result}\"");
    result
}

// ============================================================================
// #[tokio::main] — THIS IS THE LINE THAT ENABLES ASYNC IN RUST PROGRAMS
// ============================================================================
//
// [SYNTAX] #[tokio::main] = attribute macro that wraps the following async fn
// (here: async fn main()) in a synchronous runtime bootstrap.
//
// [CONCEPTUAL EXPANSION — what the macro writes for you]:
//
//   fn main() {
//       let rt = tokio::runtime::Builder::new_multi_thread()
//           .enable_all()
//           .build()
//           .unwrap();
//       rt.block_on(async {
//           // YOUR ASYNC BODY HERE
//       });
//   }
//
// There's also #[tokio::main(flavor = "current_thread")] which builds a
// SINGLE-THREADED runtime. For M1-M3 local model development, single-threaded
// is fine and actually easier to reason about timing-wise. We use the default
// multi-threaded here because the plan + Cargo.toml already pulled in full.
//
// [PITFALL] YOU CANNOT CALL .await OUTSIDE AN ASYNC CONTEXT.
//   This fails:
//     fn main() { greet(500, "foo").await; }   // ERROR: `.await` is only valid
//                                                // inside async fn or blocks
//   Fix: put #[tokio::main] on main() and declare it async.
//
// [PITFALL] DO NOT CALL BLOCKING SYNC I/O inside async code.
//   Bad:  async fn foo() { std::fs::read_to_string("huge.txt"); }  ← blocks thread
//   Good: async fn foo() { tokio::fs::read_to_string("huge.txt").await; }  ← yields
//   The tokio::fs (and tokio::net, etc.) modules exist for exactly this reason.
//
#[tokio::main]
async fn main() -> Result<(), String> {

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CHECKPOINT C — Sequential .await vs tokio::join! concurrent timing");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Both paths run 2 tasks:");
    println!("    Task A = greet(500ms, \"Alice\")   — 500ms simulated work");
    println!("    Task B = greet(500ms, \"Bob\")     — 500ms simulated work");
    println!();
    println!("  Expected results:");
    println!("    SEQUENTIAL (a.await then b.await): ~1000ms total (sum of delays)");
    println!("    CONCURRENT (join!(a,b))         : ~500ms  total (MAX delay, overlap)");
    println!();

    // ========================================================================
    // EXPERIMENT 1: SEQUENTIAL AWAIT
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("▶ EXPERIMENT 1: Sequential — a.await; b.await;");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // [SYNTAX] let start = Instant::now(); capture the current monotonic time.
    let seq_start: Instant = Instant::now();

    // ---- THE TWO TASKS RUN BACK-TO-BACK ------------------------------------
    // greet(500, "Alice") creates the Future object, schedules it with the
    // runtime, and .await WAITS HERE (but yields the thread) until it's done.
    // ONLY AFTER it returns does greet(500, "Bob") even START.
    //
    // This is: a finishes at t=500ms, then b STARTS at t=500ms, b finishes at
    // t=1000ms. Total ≈ 1000ms.
    let a_seq: String = greet(500, "Alice").await;
    let b_seq: String = greet(500, "Bob").await;
    // -----------------------------------------------------------------------

    let seq_elapsed: Duration = seq_start.elapsed();
    let seq_ms: u128 = seq_elapsed.as_millis();

    println!();
    println!("  Sequential results:");
    println!("    a = \"{a_seq}\"");
    println!("    b = \"{b_seq}\"");
    println!("    Total elapsed = {seq_ms} ms  (expect ≈ 1000)");
    println!();

    // Assert sequential is close enough to 1000ms (allow +-150ms scheduler jitter)
    assert!((900..=1200).contains(&seq_ms),
        "Sequential timing FAILED: expected ~1000ms wall time, got {seq_ms}ms. \
         If you see this, verify tokio::time::sleep (not std::thread::sleep) \
         is being used.");
    println!("  ✔ Sequential timing OK: {seq_ms} ms ≈ 1000 ms (sum of two delays)");
    println!();

    // ========================================================================
    // EXPERIMENT 2: CONCURRENT — tokio::join!
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("▶ EXPERIMENT 2: Concurrent — tokio::join!(task_a, task_b)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let join_start: Instant = Instant::now();

    // ---- CRITICAL DIFFERENCE ----------------------------------------------
    // Step 1: Create BOTH futures FIRST WITHOUT AWAITING.
    //   Creating a future = just making the lazy Future object. No sleep, no work.
    //   Both exist on the stack as values; nothing has started yet.
    let future_a = greet(500, "Alice");  // NOT started yet
    let future_b = greet(500, "Bob");    // NOT started yet

    // Step 2: tokio::join!(fa, fb) = AWAIT BOTH CONCURRENTLY.
    //   The tokio runtime polls both futures in an interleaved fashion on the
    //   same scheduler thread. When future_a hits its sleep().await, it YIELDS.
    //   The runtime immediately polls future_b which hits its sleep. Now BOTH
    //   are sleeping simultaneously on the scheduler. After 500ms elapses, BOTH
    //   wake and produce their return values. join! returns a tuple (A,B).
    //
    // This is: both START at ~t=0, both FINISH at ~t=500ms. Total ≈ 500ms.
    //
    // [SYNTAX] let (a, b) = tokio::join!(fa, fb);
    //   Destructuring assignment. join! always returns a tuple with exactly
    //   the same number of elements and in the same order as the inputs.
    //
    // [VANTA-WHY] This concurrency is the reason you spent 3 days on tokio
    // instead of just writing synchronous code. In M3 Phase 3 (Verification)
    // for one Creator patch you will run:
    //   tokio::join!(
    //       formal_verify(&patch),       // T_s = 400ms
    //       fuzz_suite(&patch, 3_000),   // T_f = 800ms
    //       static_lint(&patch),         // T_l = 100ms
    //   )
    //   Total elapsed ≈ 800ms (long pole) vs 1300ms (sum). At M5 scale this
    //   shaves hours off experimental runs.
    let (a_joint, b_joint): (String, String) = tokio::join!(future_a, future_b);
    // -----------------------------------------------------------------------

    let join_elapsed: Duration = join_start.elapsed();
    let join_ms: u128 = join_elapsed.as_millis();

    println!();
    println!("  Concurrent (tokio::join!) results:");
    println!("    a = \"{a_joint}\"");
    println!("    b = \"{b_joint}\"");
    println!("    Total elapsed = {join_ms} ms  (expect ≈ 500)");
    println!();

    // Assert concurrent < sequential AND close to 500ms (allow jitter window)
    assert!((400..=700).contains(&join_ms),
        "Concurrent timing FAILED: expected ~500ms wall time, got {join_ms}ms. \
         (Hint: if you see ~1000ms, you accidentally awaited inside the \
         future-creation step, or you used std::thread::sleep which blocks \
         the scheduler thread rather than yielding.)");
    assert!(join_ms < seq_ms,
        "Join! should be strictly faster than sequential (overlap!): \
         join={join_ms}ms, seq={seq_ms}ms");
    println!("  ✔ Concurrent timing OK: {join_ms} ms ≈ 500 ms (one overlap window)");
    println!();
    println!("  ⚡ Speedup factor = {:.2}×  (ideal 2.0×, your results will be near this)",
             seq_ms as f64 / join_ms.max(1) as f64);
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CHECKPOINT C EXIT: OK");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

// ============================================================================
// 5 REUSABLE RECIPES FROM CHECKPOINT C
// ============================================================================
//
// COPY-PASTE THESE TEMPLATES DIRECTLY INTO M1-M5 CODE.
//
// ----------------------------------------------------------------------------
// RECIPE C1 — "All binaries: #[tokio::main] on async main"
//
//   #[tokio::main]
//   async fn main() -> Result<(), String> {
//       let cfg = load_config().await?;
//       run_pipeline(&cfg).await?;
//       Ok(())
//   }
//
//   → USE IN: Literally every async binary (all of them after M0, because
//     they all do HTTP calls). In main.rs (which we already converted — see
//     main.rs for the thin-wrapper version).
//
// ----------------------------------------------------------------------------
// RECIPE C2 — "Simulated-delay async task (for tests and checkpoints)"
//
//   async fn simulated_api_call(ms: u64) -> String {
//       tokio::time::sleep(Duration::from_millis(ms)).await;  // yield, NOT block!
//       "done".to_string()
//   }
//
//   NEVER USE std::thread::sleep inside async code. It blocks the runtime thread.
//
// ----------------------------------------------------------------------------
// RECIPE C3 — "Sequential execution: .await one by one"
//   Use when each step DEPENDS on previous outputs (chain pattern, pipeline).
//
//   let step_1_result = step_1().await?;
//   let step_2_result = step_2(step_1_result.field).await?;
//   let step_3_result = step_3(&step_2_result).await?;
//
//   → USE IN: M1 3-milestone chain, M2 BATE→execution→diagnostician pipeline.
//
// ----------------------------------------------------------------------------
// RECIPE C4 — "Concurrent execution: tokio::join!(a, b, c)"
//   Use when tasks are INDEPENDENT (no data dependency between them).
//
//   let (formal_result, fuzz_result, lint_result) = tokio::join!(
//       run_formal(patch.clone()),
//       run_fuzz(patch.clone(), iterations),
//       run_lint(patch.clone()),
//   );
//
//   → USE IN: M3 Phase 3 (run 3 verifier backends on same patch),
//     M2 (run batches of tests in parallel).
//
// ----------------------------------------------------------------------------
// RECIPE C5 — "Instant timing wrap any block"
//
//   let t = Instant::now();
//   let r = expensive_thing().await?;
//   let ms = t.elapsed().as_millis();
//   eprintln!("expensive_thing took {ms} ms");
//
//   → USE IN: Every phase of every milestone for instrumentation + M5
//     benchmark CSV columns. Add this pattern ALWAYS; you will regret not
//     having timings when you debug M3 slow runs.
//
// ============================================================================
