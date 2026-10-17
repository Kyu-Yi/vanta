#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/fixtures/toy_calc"
RUN_ROOT="${HOME}/vanta-runs/toy_calc"
LOG_DIR="$RUN_ROOT/logs"
HOST_PYTHON="$(command -v python3)"

mkdir -p "$LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  zsh fixtures/toy_calc/run_matrix.sh [all|shuru|subprocess]

Runs:
  - M3 review bad patch
  - M3 review good patch
  - M3 repair buggy original
  - M2 review bad patch

Each run starts from a fresh copy outside the vanta repo:
  ~/vanta-runs/toy_calc/<backend>_<binary>_<scenario>/
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

BACKENDS=()
case "${1:-all}" in
  all)
    BACKENDS=("subprocess" "shuru")
    ;;
  shuru|subprocess)
    BACKENDS=("${1}")
    ;;
  *)
    usage
    exit 1
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

run_case() {
  local backend="$1"
  local binary="$2"
  local scenario="$3"
  local test_cmd="$4"
  shift 4

  local run_dir="$RUN_ROOT/${backend}_${binary}_${scenario}"
  local log_path="$LOG_DIR/${backend}_${binary}_${scenario}.log"
  copy_fixture_repo "$run_dir"

  echo
  echo "==> backend=${backend} binary=${binary} scenario=${scenario}"
  echo "    repo=${run_dir}"
  echo "    log=${log_path}"

  (
    cd "$run_dir"
    VANTA_SANDBOX_BACKEND="$backend" \
      cargo run --manifest-path "$ROOT_DIR/Cargo.toml" --bin "$binary" -- \
      --repo "$run_dir" \
      --issue "$run_dir/issue.md" \
      --test-cmd "$test_cmd" \
      "$@"
  ) 2>&1 | tee "$log_path"
}

for backend in "${BACKENDS[@]}"; do
  if [[ "$backend" == "subprocess" ]]; then
    TEST_CMD="$SUBPROCESS_TEST_CMD"
  else
    TEST_CMD="python3 -m pytest -q"
  fi

  run_case "$backend" vanta_m3 review_bad "$TEST_CMD" \
    --patch "$RUN_ROOT/${backend}_vanta_m3_review_bad/bad.patch" \
    --max-attempts 3

  run_case "$backend" vanta_m3 review_good "$TEST_CMD" \
    --patch "$RUN_ROOT/${backend}_vanta_m3_review_good/good.patch" \
    --max-attempts 3

  run_case "$backend" vanta_m3 repair "$TEST_CMD" \
    --max-attempts 3

  run_case "$backend" vanta_m2 review_bad "$TEST_CMD" \
    --patch "$RUN_ROOT/${backend}_vanta_m2_review_bad/bad.patch"
done
