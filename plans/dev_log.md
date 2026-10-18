# Vanta development log

Chronological notes on development runs. **These are development observations,
not evidence.** They are on the toy fixture only, are not part of any M5
dataset, and must not be cited as results. Formal measurements live in the
session/transition JSONL under `vanta_sessions/`.

---

## Attachment-stage boundary rule — review-mode catch rate (before / after)

- **Date:** 2026-10-18
- **Scope:** `fixtures/toy_calc` (toy fixture only), `vanta_m3` and `vanta_m2`
  review mode, backends `subprocess` and `shuru`, `--max-attempts 3` (M3) /
  N=1 (M2), n as noted per batch.
- **Change under test:** `attacker.2.boundary-probe` — the Attacker prompt asks
  for a `BehaviorPreservation` probe at the OLD boundary value of every
  comparison the patch changes, and the patch-blind intent check scraps a
  preservation probe the issue requires to change.

### Before the boundary rule — `2/5` rejected

The bad patch's only regression is at exactly `500g`
(`shipping_band(500)`: original `"standard"` → patched `"light"`). Before the
rule, the Attacker usually probed `499`, which returns the same band on both
sides and proves nothing, so the bad patch was caught only when the Attacker
happened to test `500`.

```
==> vanta_m3 review_bad @ subprocess: rejected 2/5, approved 3, other 0
==> SUMMARY: rejected 2/5, approved 3, other 0
```

> Caveat: the pre-rule runs were written into the shared, non-timestamped run
> directory and were later **overwritten** by a subsequent batch, so only 3
> `prompt_version=unversioned` sessions survive. The numbers above are from the
> pasted terminal record, not from logs on disk.

### After the boundary rule — `5 of 6` rejected, but it still misses

| Batch | Backend | Scenario | Outcome |
|---|---|---|---|
| `2026-10-18T0923…` ×5 | subprocess | `vanta_m3 review_bad` | rejected **5/5** |
| `2026-10-18T092657` | shuru | `vanta_m3 review_bad` | **rejected** (probe landed on 500) |
| `2026-10-18T095250` | shuru | `vanta_m3 review_bad` | **APPROVED — miss** |
| `2026-10-18T095250` | shuru | `vanta_m3 review_good` | approved |
| `2026-10-18T095250` | shuru | `vanta_m3 repair` | approved |
| `2026-10-18T095250` | shuru | `vanta_m2 review_bad` | **APPROVED — miss** |

Across both backends the bad patch was rejected **5 of 6** times with the rule in
place (5/5 on subprocess, 0/1 on shuru).

**Why the shuru M3 run missed.** The Attacker probed `499`, `501`, `999`,
`1000`, `1001`, `1500`, `250`, `0` and a negative input — it went around `500`
on both sides and never landed on it. The `499`/`501` probes return the same
band on original and patched (`499 < 500` and `499 <= 500` are both true), so
nothing catches the regression. The M2 run tested `499` and `1001` and likewise
skipped `500`. This is Attacker sampling variance, not a backend fault.

**Why it is not a shuru problem.** The four shuru runs contain no
`Infrastructure` classification, no `HarnessFailure` and no timeout, and every
sandbox was a real microVM (`▌ Backend: SHURU microVM active (forced)`). Every
Attacker-written test executed inside a microVM and returned a real pass/fail;
the subprocess backend would have approved the same tests.

### Honest summary

| Scenario | Before | After |
|---|---|---|
| `vanta_m3 review_bad` rejected | 2/5 | 5/6 (both backends) |
| `vanta_m3 review_good` approved | — | 5/5 |

n=5 (subprocess) + n=2 (shuru), one toy fixture. The correct reading is
**"rejection rate went up, still sometimes misses"** — not "fixed", and not
"variance". The deterministic boundary-obligation mechanism that would close it
is deliberately deferred (`plans/m1_m2_implementation_plan.md` §M3-Addendum 6).

A development result. **Not evidence.**

## 2026-10-18 — Review-mode prompt narrowing

Review mode no longer sends the whole repository to the Attacker. The Attacker,
the intent audit and the attacker revision now receive the files the patch
touches, the in-repo files those touched files import (one level deep), and the
repo's existing tests that cover them. The execution tree used by BATE is
unchanged — only the prompt context is narrowed. Without this, a large repo
(`heretic`, ~261 KB of source) cannot fit the model context at all.

Every LLM call now logs a prompt budget and fails closed if the request would
exceed the model context (`VANTA_LLM_CONTEXT_TOKENS`, raised to 131072 for the
local server).

Regression check on Shuru (`review_bad` x5, `review_good` x5), same toy fixture:

| Binary | Scenario | Result |
|---|---|---|
| `vanta_m3` | review_bad | rejected 5/5 |
| `vanta_m3` | review_good | approved 5/5 |
| `vanta_m2` | review_bad | rejected 3/5 |

**Attack depth matters.** M3 (3 attack rounds) rejected 5/5; M2 (1 round)
rejected 3/5 — same patch, same model, same backend. Both M2 approvals came
from a round-1 attack set that contained no test at the 500 boundary at all
(`1000` / `499` / `1001`). M2 was never 5/5 on this scenario in earlier batches
either (1/1, then 0/1), so this is the one-round baseline behaving as expected,
not a regression from the narrowing. M3 is the binding regression; M2 is kept
as the contrast baseline.

A development result. **Not evidence.**

## Follow-up

- [ ] Narrow the Creator context before the BugsInPy loader lands: pass the
  failing file, traceback-referenced files, and one-level in-repo imports
  instead of the whole repository on large projects like pandas or keras.
