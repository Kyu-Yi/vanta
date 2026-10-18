#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/fixtures/toy_calc"
RUN_ROOT="${HOME}/vanta-runs/toy_calc"
# [VANTA-WHY] Every invocation gets its OWN timestamped batch directory and
# writes nothing outside it. Before this, each batch reused
# `<backend>_<binary>_<scenario>[_runN]` under a shared root, so a second batch
# silently overwrote the first — and the pre-boundary-rule 2/5 catch-rate record
# was lost that way. Seconds precision (not just minutes) so two batches started
# in the same minute still cannot collide.
BATCH_STAMP="$(date +%Y-%m-%dT%H%M%S)"
HOST_PYTHON="$(command -v python3)"

usage() {
  cat <<'EOF'
Usage:
  zsh fixtures/toy_calc/run_matrix.sh [all|shuru|subprocess]
      Run the whole matrix: M3 review bad, M3 review good, M3 repair, M2 review bad.

  zsh fixtures/toy_calc/run_matrix.sh [all|shuru|subprocess] --only <scenario> [--repeat N]
      Run ONE scenario, optionally N times, and report how many runs REJECTED the
      candidate. The Attacker is random, so a single outcome says little; --repeat
      measures a CATCH RATE instead of one coin flip.
      <scenario> is review_bad, review_good or repair (or the full <binary>:<scenario>).

Each invocation writes to its own timestamped batch directory and never
overwrites an earlier batch:
  ~/vanta-runs/toy_calc/<timestamp>_<backend>/
      <binary>_<scenario>[_runN]/     (fresh repo copy)
      logs/<binary>_<scenario>[.runN].log
EOF
}

BACKEND_ARG="all"
ONLY=""
REPEAT=1
while (( $# > 0 )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --repeat) REPEAT="${2:-}"; shift 2 ;;
    --repeat=*) REPEAT="${1#--repeat=}"; shift ;;
    all|shuru|subprocess) BACKEND_ARG="$1"; shift ;;
    *) usage; exit 1 ;;
  esac
done

if (( REPEAT > 1 )) && [[ -z "$ONLY" ]]; then
  echo "--repeat needs --only <scenario>: repeating the whole matrix is not what this is for" >&2
  exit 1
fi

BACKENDS=()
case "$BACKEND_ARG" in
  all)
    BACKENDS=("subprocess" "shuru")
    ;;
  shuru|subprocess)
    BACKENDS=("$BACKEND_ARG")
    ;;
esac

if "$HOST_PYTHON" - <<'PY'
import importlib.util, sys
ok = all(importlib.util.find_spec(name) for name in ("pytest", "pytest_jsonreport"))
sys.exit(0 if ok else 1)
PY
then
  SUBPROCESS_TEST_CMD="$HOST_PYTHON -m pytest -q"
else
  echo "python3 is missing pytest or pytest-json-report; create a venv before running subprocess cases" >&2
  exit 1
fi

copy_fixture_repo() {
  local run_dir="$1"
  rm -rf "$run_dir"
  mkdir -p "$(dirname "$run_dir")"
  /bin/cp -R "$FIXTURE_DIR" "$run_dir"
  rm -rf "$run_dir/run_logs"
  git -C "$run_dir" init -q
  git -C "$run_dir" config user.name "Vanta Fixture"
  git -C "$run_dir" config user.email "vanta-fixture@example.com"
  git -C "$run_dir" add .
  git -C "$run_dir" commit -qm "Fixture baseline"
}

# Run one scenario in a fresh run dir and classify the GOVERNOR decision.
# Prints one of: rejected | approved | other:<exit-code>.
# Does NOT call `exit` on a failing binary: a rejection exits non-zero by design,
# and that is the outcome we are here to count.
execute_case() {
  local backend="$1"
  local binary="$2"
  local scenario="$3"
  local test_cmd="$4"
  local run_dir="$5"
  local log_path="$6"

  copy_fixture_repo "$run_dir"

  local extra=()
  case "${binary}:${scenario}" in
    vanta_m3:review_bad)  extra=(--patch "$run_dir/bad.patch" --max-attempts 3) ;;
    vanta_m3:review_good) extra=(--patch "$run_dir/good.patch" --max-attempts 3) ;;
    vanta_m3:repair)      extra=(--max-attempts 3) ;;
    vanta_m2:review_bad)  extra=(--patch "$run_dir/bad.patch") ;;
    *) echo "unknown scenario ${binary}/${scenario}" >&2; return 2 ;;
  esac

  local exit_code=0
  (
    cd "$run_dir"
    VANTA_SANDBOX_BACKEND="$backend" \
      cargo run --manifest-path "$ROOT_DIR/Cargo.toml" --bin "$binary" -- \
      --repo "$run_dir" \
      --issue "$run_dir/issue.md" \
      --test-cmd "$test_cmd" \
      "${extra[@]}"
  ) >"$log_path" 2>&1 || exit_code=$?

  if grep -q "FINAL: Rejected" "$log_path"; then
    echo "rejected"
  elif grep -q "FINAL: AllTestsPassed" "$log_path"; then
    echo "approved"
  else
    echo "other:${exit_code}"
  fi
}

# Ordered run cases, "<binary>:<scenario>".
CASES=(
  "vanta_m3:review_bad"
  "vanta_m3:review_good"
  "vanta_m3:repair"
  "vanta_m2:review_bad"
)

matches_only() {
  local case_id="$1"
  [[ -z "$ONLY" ]] && return 0
  [[ "$case_id" == "$ONLY" ]] && return 0
  [[ "${case_id#*:}" == "$ONLY" ]] && return 0
  return 1
}

total_rejected=0
total_approved=0
total_other=0
total_runs=0

for backend in "${BACKENDS[@]}"; do
  BATCH_DIR="${RUN_ROOT}/${BATCH_STAMP}_${backend}"
  BATCH_LOG_DIR="${BATCH_DIR}/logs"
  mkdir -p "$BATCH_LOG_DIR"

  if [[ "$backend" == "subprocess" ]]; then
    TEST_CMD="$SUBPROCESS_TEST_CMD"
  else
    TEST_CMD="python3 -m pytest -q"
  fi

  for case_id in "${CASES[@]}"; do
    matches_only "$case_id" || continue
    binary="${case_id%%:*}"
    scenario="${case_id#*:}"

    rejected=0
    approved=0
    other=0

    for i in $(seq 1 "$REPEAT"); do
      if (( REPEAT > 1 )); then
        run_dir="$BATCH_DIR/${binary}_${scenario}_run${i}"
        log_path="$BATCH_LOG_DIR/${binary}_${scenario}.run${i}.log"
      else
        run_dir="$BATCH_DIR/${binary}_${scenario}"
        log_path="$BATCH_LOG_DIR/${binary}_${scenario}.log"
      fi

      echo
      echo "==> backend=${backend} binary=${binary} scenario=${scenario} run=${i}/${REPEAT}"
      echo "    repo=${run_dir}"
      echo "    log=${log_path}"

      outcome="$(execute_case "$backend" "$binary" "$scenario" "$TEST_CMD" "$run_dir" "$log_path")"
      case "$outcome" in
        rejected) rejected=$((rejected + 1)) ;;
        approved) approved=$((approved + 1)) ;;
        *)        other=$((other + 1)) ;;
      esac
      echo "    outcome=${outcome}"
    done

    total_rejected=$((total_rejected + rejected))
    total_approved=$((total_approved + approved))
    total_other=$((total_other + other))
    total_runs=$((total_runs + REPEAT))

    echo
    echo "==> ${binary} ${scenario} @ ${backend}: rejected ${rejected}/${REPEAT}, approved ${approved}, other ${other}"
  done
done

if (( REPEAT > 1 )); then
  echo
  echo "==> SUMMARY: rejected ${total_rejected}/${total_runs}, approved ${total_approved}, other ${total_other}"
fi

