#!/usr/bin/env bash
set -euo pipefail

# Strong-scaling benchmark driver for fluxpulse_run_mpi.
#
# Usage examples:
#   bash scaling/bench_mpi_scaling.sh \
#     --exe ./build/fluxpulse_run_mpi \
#     --config configs/bench_2q_noisy.json \
#     --ranks 1,2,4,8 \
#     --repeat 3 \
#     --batch-size 64
#
#   bash scaling/bench_mpi_scaling.sh \
#     --exe ./build/fluxpulse_run_mpi \
#     --config configs/1q_t1.json \
#     --ranks 1,2,4 \
#     --mode static \
#     --output scaling/strong_scaling.csv
#
# Notes:
# - This script measures wall-clock time externally with /usr/bin/time.
# - It assumes the total trajectory count comes from the JSON config and therefore
#   remains fixed across ranks (strong scaling).
# - It writes one CSV row per run plus summary columns speedup/efficiency.
# - Result files produced by the app are left to the app configuration. To avoid
#   overwriting outputs between runs, prefer benchmark configs with unique or
#   disposable result paths.

usage() {
  cat <<'EOF'
Usage:
  bench_mpi_scaling.sh --exe <path> --config <path> [options]

Required:
  --exe <path>            Path to fluxpulse_run_mpi executable
  --config <path>         Benchmark config JSON

Options:
  --compiled-program <p>  Optional compiled program JSON
  --ranks <list>          Comma-separated MPI ranks (default: 1,2,4)
  --repeat <n>            Repeats per rank (default: 3)
  --batch-size <n>        Pass --batch_size <n> to the MPI app
  --mode <m>              dynamic | static (default: dynamic)
  --mpirun <cmd>          MPI launcher command (default: mpirun)
  --output <path>         CSV output path (default: scaling/strong_scaling.csv)
  --label <text>          Benchmark label written to CSV
  --extra-arg <arg>       Extra argument passed through to the MPI app; repeatable
  --help                  Show this help

CSV columns:
  label,config,mode,batch_size,ranks,repeat,elapsed_sec,best_rank1_sec,speedup,efficiency
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

EXE=""
CONFIG=""
COMPILED_PROGRAM=""
RANKS_CSV="1,2,4"
REPEAT=3
BATCH_SIZE=""
MODE="dynamic"
MPIRUN_CMD="mpirun"
OUTPUT="scaling/strong_scaling.csv"
LABEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exe)
      EXE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --compiled-program)
      COMPILED_PROGRAM="$2"
      shift 2
      ;;
    --ranks)
      RANKS_CSV="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mpirun)
      MPIRUN_CMD="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --extra-arg)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$EXE" ]] || { echo "ERROR: --exe is required" >&2; usage; exit 2; }
[[ -n "$CONFIG" ]] || { echo "ERROR: --config is required" >&2; usage; exit 2; }
[[ -x "$EXE" ]] || { echo "ERROR: executable not found or not executable: $EXE" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }
[[ "$REPEAT" =~ ^[0-9]+$ ]] || { echo "ERROR: --repeat must be an integer" >&2; exit 2; }
[[ "$MODE" == "dynamic" || "$MODE" == "static" ]] || { echo "ERROR: --mode must be dynamic or static" >&2; exit 2; }

require_cmd "$MPIRUN_CMD"
require_cmd /usr/bin/time
require_cmd awk
require_cmd python3

mkdir -p "$(dirname "$OUTPUT")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

IFS=',' read -r -a RANKS <<< "$RANKS_CSV"
if [[ ${#RANKS[@]} -eq 0 ]]; then
  echo "ERROR: --ranks must contain at least one value" >&2
  exit 2
fi

if [[ -z "$LABEL" ]]; then
  LABEL="$(basename "$CONFIG")"
fi

CSV_ESCAPED_CONFIG="$CONFIG"
CSV_ESCAPED_LABEL="$LABEL"
CSV_ESCAPED_MODE="$MODE"
CSV_ESCAPED_BATCH_SIZE="${BATCH_SIZE:-default}"

echo "label,config,mode,batch_size,ranks,repeat,elapsed_sec,best_rank1_sec,speedup,efficiency" > "$OUTPUT"

rank1_best=""

run_one() {
  local ranks="$1"
  local rep="$2"
  local time_file="$TMP_DIR/time_r${ranks}_rep${rep}.txt"
  local app_log="$TMP_DIR/app_r${ranks}_rep${rep}.log"

  local -a cmd
  cmd=("$MPIRUN_CMD" -n "$ranks" "$EXE" "$CONFIG")

  if [[ -n "$COMPILED_PROGRAM" ]]; then
    cmd+=(--compiled_program "$COMPILED_PROGRAM")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi
  if [[ "$MODE" == "static" ]]; then
    cmd+=(--static)
  else
    cmd+=(--dynamic)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[bench] ranks=$ranks repeat=$rep mode=$MODE batch_size=${BATCH_SIZE:-default}" >&2
  /usr/bin/time -p -o "$time_file" "${cmd[@]}" > "$app_log" 2>&1

  awk '/^real / { print $2; found=1 } END { if (!found) exit 1 }' "$time_file"
}

calc_metrics() {
  local elapsed="$1"
  local base="$2"
  local ranks="$3"
  python3 - "$elapsed" "$base" "$ranks" <<'PY'
import sys
elapsed = float(sys.argv[1])
base = float(sys.argv[2])
ranks = int(sys.argv[3])
speedup = base / elapsed if elapsed > 0 else 0.0
eff = speedup / ranks if ranks > 0 else 0.0
print(f"{speedup:.8f},{eff:.8f}")
PY
}

best_of_list() {
  python3 - "$@" <<'PY'
import sys
vals = [float(x) for x in sys.argv[1:]]
print(f"{min(vals):.8f}")
PY
}

for ranks in "${RANKS[@]}"; do
  [[ "$ranks" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid rank count: $ranks" >&2; exit 2; }
  (( ranks >= 1 )) || { echo "ERROR: rank count must be >= 1" >&2; exit 2; }

  times=()
  for ((rep = 1; rep <= REPEAT; ++rep)); do
    elapsed="$(run_one "$ranks" "$rep")"
    times+=("$elapsed")
  done

  best_elapsed="$(best_of_list "${times[@]}")"

  if [[ "$ranks" == "1" ]]; then
    rank1_best="$best_elapsed"
  fi

  if [[ -z "$rank1_best" ]]; then
    echo "ERROR: rank 1 must be included in --ranks to compute strong-scaling speedup" >&2
    exit 2
  fi

  metrics="$(calc_metrics "$best_elapsed" "$rank1_best" "$ranks")"
  speedup="${metrics%,*}"
  efficiency="${metrics#*,}"

  printf '%s,%s,%s,%s,%s,%s,%.8f,%.8f,%.8f,%.8f\n' \
    "$CSV_ESCAPED_LABEL" \
    "$CSV_ESCAPED_CONFIG" \
    "$CSV_ESCAPED_MODE" \
    "$CSV_ESCAPED_BATCH_SIZE" \
    "$ranks" \
    "$REPEAT" \
    "$best_elapsed" \
    "$rank1_best" \
    "$speedup" \
    "$efficiency" >> "$OUTPUT"
done

echo "[bench] wrote: $OUTPUT" >&2
cat "$OUTPUT"
