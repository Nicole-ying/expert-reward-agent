#!/usr/bin/env bash
set -uo pipefail
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

CONFIG="configs/env002_rpv4_old_v4_noexpert.yaml"
PREFIX="rpv4_old_v4_noexpert"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

mkdir -p logs

# 64-core host: 5 seeds × 12 cores = cores 0..59 (cores 60..63 left free).
declare -A CORES=(
  [0]="0-11"
  [1]="12-23"
  [2]="24-35"
  [3]="36-47"
  [4]="48-59"
)

echo "===== bipedal rpv4_old_v4_noexpert: 5 seeds × 12 cores, parallel ====="
echo "CONFIG=$CONFIG  ROUNDS=$ROUNDS  TIMESTEPS=$TOTAL_TIMESTEPS  EVAL=$EVAL_EPISODES"

PIDS=()
for seed in 0 1 2 3 4; do
  logfile="logs/rpv4_old_v4_noexpert_bipedal_seed${seed}.log"
  echo "SEED $seed -> cores ${CORES[$seed]} -> $logfile ($(date '+%H:%M:%S'))"
  OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 \
    taskset -c "${CORES[$seed]}" \
    python -m pipeline.run_iterative_experiment \
      --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" \
      --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" \
      --eval-episodes "$EVAL_EPISODES" \
      > "$logfile" 2>&1 &
  PIDS+=("$!")
done

echo "Launched ${#PIDS[@]} seeds (pids: ${PIDS[*]})"
status=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "SEED $i DONE — $(date '+%H:%M:%S')"
  else
    echo "SEED $i FAILED (exit=$rc) — see logs/rpv4_old_v4_noexpert_bipedal_seed${i}.log"
    status=1
  fi
done
echo "ALL DONE — $(date) (status=$status)"
exit $status
