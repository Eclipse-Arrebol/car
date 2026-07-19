#!/usr/bin/env bash
set -euo pipefail

# Personalized FL comparison under the submitted-aware C-state.
#
# Usage on the server:
#   cd /path/to/car_charge
#   bash run_personalized_fl_cstate.sh smoke
#   bash run_personalized_fl_cstate.sh full
#
# Optional environment overrides:
#   LOAD=80 ROUNDS=100 TRAIN_SEEDS="101 102" EVAL_SEEDS="1001 1002" bash run_personalized_fl_cstate.sh full
#   FORCE=1 bash run_personalized_fl_cstate.sh full

MODE="${1:-full}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
DRIVER="${DRIVER:-$PROJECT_ROOT/paper_personalized_fl_cstate.py}"

LOAD="${LOAD:-60}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPISODES="${LOCAL_EPISODES:-1}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-144}"
NUM_STATIONS="${NUM_STATIONS:-4}"
NUM_CHARGERS_PER_STATION="${NUM_CHARGERS_PER_STATION:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPSILON="${EPSILON:-1.0}"
EPSILON_DECAY="${EPSILON_DECAY:-0.98}"
SAVE_EVERY="${SAVE_EVERY:-20}"
FORCE="${FORCE:-0}"

TRAIN_SEEDS="${TRAIN_SEEDS:-101 102 103 104 105}"
EVAL_SEEDS="${EVAL_SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010}"

if [[ "$MODE" == "smoke" ]]; then
  ROUNDS="${ROUNDS_SMOKE:-2}"
  TRAIN_SEEDS="${TRAIN_SEEDS_SMOKE:-101}"
  EVAL_SEEDS="${EVAL_SEEDS_SMOKE:-1001 1002}"
  FORCE="${FORCE_SMOKE:-1}"
elif [[ "$MODE" != "full" && "$MODE" != "train-only" && "$MODE" != "eval-only" && "$MODE" != "summarize" ]]; then
  echo "Unknown mode: $MODE"
  echo "Expected one of: smoke, full, train-only, eval-only, summarize"
  exit 2
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HETERO_REWARD=1

CLIENT_SPECS="old_city:1.3,new_city:1.0,suburb:0.7"
ROOT_CKPT="checkpoints_personalized_fl_cstate"
OUT_DIR="evaluation/personalized_fl_cstate"
LOG_DIR="$OUT_DIR/logs"
LONG_CSV="$OUT_DIR/personalized_fl_cstate_long.csv"
SUMMARY_CSV="$OUT_DIR/personalized_fl_cstate_summary.csv"

mkdir -p "$ROOT_CKPT" "$OUT_DIR" "$LOG_DIR"

if [[ ! -f "$DRIVER" ]]; then
  echo "Missing driver: $DRIVER"
  exit 1
fi

run_step() {
  local name="$1"
  local log_path="$2"
  shift 2
  echo
  echo "[personalized-fl-c] START $name"
  echo "[personalized-fl-c] log -> $log_path"
  echo "[personalized-fl-c] cmd -> $PYTHON $*"
  local start
  start="$(date +%s)"
  "$PYTHON" "$@" 2>&1 | tee -a "$log_path"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -ne 0 ]]; then
    echo "[personalized-fl-c] FAILED $name with exit code $code. See $log_path"
    exit "$code"
  fi
  local end
  end="$(date +%s)"
  echo "[personalized-fl-c] DONE $name in $((end - start))s"
}

all_exist() {
  local path
  for path in "$@"; do
    [[ -f "$path" ]] || return 1
  done
  return 0
}

common_args=(
  "--num-evs" "$LOAD"
  "--num-stations" "$NUM_STATIONS"
  "--num-chargers-per-station" "$NUM_CHARGERS_PER_STATION"
  "--steps-per-episode" "$STEPS_PER_EPISODE"
  "--batch-size" "$BATCH_SIZE"
  "--network" "station_only"
  "--no-use-action-mask"
  "--epsilon" "$EPSILON"
  "--epsilon-decay" "$EPSILON_DECAY"
  "--save-every" "$SAVE_EVERY"
)

echo "[personalized-fl-c] project=$PROJECT_ROOT"
echo "[personalized-fl-c] mode=$MODE"
echo "[personalized-fl-c] load=$LOAD"
echo "[personalized-fl-c] train seeds=$TRAIN_SEEDS"
echo "[personalized-fl-c] eval seeds=$EVAL_SEEDS"
echo "[personalized-fl-c] rounds/episodes=$ROUNDS"
echo "[personalized-fl-c] local episodes per FL round=$LOCAL_EPISODES"
echo "[personalized-fl-c] state=C submitted-aware heading; no z correction to wait/service"
echo "[personalized-fl-c] output long csv=$LONG_CSV"
echo "[personalized-fl-c] HETERO_REWARD=1"

if [[ "$MODE" != "summarize" ]]; then
  for seed in $TRAIN_SEEDS; do
    tag="cstate_${LOAD}ev_r${ROUNDS}_seed${seed}"
    prefix="$ROOT_CKPT/$tag"

    old_dir="${prefix}_local_old_city"
    new_dir="${prefix}_local_new_city"
    sub_dir="${prefix}_local_suburb"
    central_dir="${prefix}_centralized"
    fedavg_dir="${prefix}_fedavg"
    fedprox_dir="${prefix}_fedprox"
    fedset_dir="${prefix}_fedsetrl"

    old_final="$old_dir/model_final.pth"
    new_final="$new_dir/model_final.pth"
    sub_final="$sub_dir/model_final.pth"
    central_final="$central_dir/central_final.pth"
    fedavg_final="$fedavg_dir/global_final.pth"
    fedprox_final="$fedprox_dir/global_final.pth"
    fedset_old="$fedset_dir/old_city_final.pth"
    fedset_new="$fedset_dir/new_city_final.pth"
    fedset_sub="$fedset_dir/suburb_final.pth"

    if [[ "$MODE" != "eval-only" ]]; then
      if [[ "$FORCE" == "1" || ! -f "$old_final" ]]; then
        run_step "train_${tag}_local_old_city" "$LOG_DIR/train_${tag}_local_old_city.log" \
          "$DRIVER" train-local "${common_args[@]}" --seed "$seed" --episodes "$ROUNDS" \
          --city old_city --ue-scale 1.3 --save-dir "$old_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} local_old_city"
      fi

      if [[ "$FORCE" == "1" || ! -f "$new_final" ]]; then
        run_step "train_${tag}_local_new_city" "$LOG_DIR/train_${tag}_local_new_city.log" \
          "$DRIVER" train-local "${common_args[@]}" --seed "$seed" --episodes "$ROUNDS" \
          --city new_city --ue-scale 1.0 --save-dir "$new_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} local_new_city"
      fi

      if [[ "$FORCE" == "1" || ! -f "$sub_final" ]]; then
        run_step "train_${tag}_local_suburb" "$LOG_DIR/train_${tag}_local_suburb.log" \
          "$DRIVER" train-local "${common_args[@]}" --seed "$seed" --episodes "$ROUNDS" \
          --city suburb --ue-scale 0.7 --save-dir "$sub_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} local_suburb"
      fi

      if [[ "$FORCE" == "1" || ! -f "$central_final" ]]; then
        run_step "train_${tag}_centralized" "$LOG_DIR/train_${tag}_centralized.log" \
          "$DRIVER" train-centralized "${common_args[@]}" --seed "$seed" --rounds "$ROUNDS" \
          --local-episodes "$LOCAL_EPISODES" --client-specs "$CLIENT_SPECS" --save-dir "$central_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} centralized"
      fi

      if [[ "$FORCE" == "1" || ! -f "$fedavg_final" ]]; then
        run_step "train_${tag}_fedavg" "$LOG_DIR/train_${tag}_fedavg.log" \
          "$DRIVER" train-fed "${common_args[@]}" --seed "$seed" --rounds "$ROUNDS" \
          --local-episodes "$LOCAL_EPISODES" --client-specs "$CLIENT_SPECS" \
          --fed-method fedavg --save-dir "$fedavg_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} fedavg"
      fi

      if [[ "$FORCE" == "1" || ! -f "$fedprox_final" ]]; then
        run_step "train_${tag}_fedprox" "$LOG_DIR/train_${tag}_fedprox.log" \
          "$DRIVER" train-fed "${common_args[@]}" --seed "$seed" --rounds "$ROUNDS" \
          --local-episodes "$LOCAL_EPISODES" --client-specs "$CLIENT_SPECS" \
          --fed-method fedprox --fedprox-mu 0.1 --save-dir "$fedprox_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} fedprox"
      fi

      if [[ "$FORCE" == "1" ]] || ! all_exist "$fedset_old" "$fedset_new" "$fedset_sub"; then
        run_step "train_${tag}_fedsetrl" "$LOG_DIR/train_${tag}_fedsetrl.log" \
          "$DRIVER" train-fed "${common_args[@]}" --seed "$seed" --rounds "$ROUNDS" \
          --local-episodes "$LOCAL_EPISODES" --client-specs "$CLIENT_SPECS" \
          --fed-method fedsetrl --save-dir "$fedset_dir"
      else
        echo "[personalized-fl-c] SKIP train ${tag} fedsetrl"
      fi
    fi

    if [[ "$MODE" != "train-only" ]]; then
      if ! all_exist "$old_final" "$new_final" "$sub_final" "$central_final" "$fedavg_final" "$fedprox_final" "$fedset_old" "$fedset_new" "$fedset_sub"; then
        echo "[personalized-fl-c] WARN missing checkpoints for ${tag}; skip eval"
        continue
      fi

      eval_seed_args=()
      for eval_seed in $EVAL_SEEDS; do
        eval_seed_args+=("$eval_seed")
      done

      run_step "eval_${tag}" "$LOG_DIR/eval_${tag}.log" \
        "$DRIVER" eval \
        --out "$LONG_CSV" \
        --resume \
        --train-seed "$seed" \
        --num-evs "$LOAD" \
        --num-stations "$NUM_STATIONS" \
        --num-chargers-per-station "$NUM_CHARGERS_PER_STATION" \
        --steps-per-episode "$STEPS_PER_EPISODE" \
        --network station_only \
        --no-use-action-mask \
        --client-specs "$CLIENT_SPECS" \
        --eval-seeds "${eval_seed_args[@]}" \
        --model-specs \
        "Local:old_city:$old_final" \
        "Local:new_city:$new_final" \
        "Local:suburb:$sub_final" \
        "Centralized:*:$central_final" \
        "FedAvg:*:$fedavg_final" \
        "FedProx:*:$fedprox_final" \
        "FedSetRL:old_city:$fedset_old" \
        "FedSetRL:new_city:$fedset_new" \
        "FedSetRL:suburb:$fedset_sub"
    fi
  done
fi

if [[ "$MODE" != "train-only" && -f "$LONG_CSV" ]]; then
  run_step "summarize_personalized_fl_cstate" "$LOG_DIR/summarize.log" \
    "$DRIVER" summarize --csv "$LONG_CSV" --out "$SUMMARY_CSV"
fi

echo
echo "[personalized-fl-c] ALL DONE"
echo "[personalized-fl-c] checkpoints -> $ROOT_CKPT"
echo "[personalized-fl-c] long csv -> $LONG_CSV"
echo "[personalized-fl-c] summary csv -> $SUMMARY_CSV"
