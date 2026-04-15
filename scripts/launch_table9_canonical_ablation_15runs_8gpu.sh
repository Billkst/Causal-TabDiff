#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/coding/Causal-TabDiff"
OUTPUT_ROOT="$ROOT/outputs/table9_canonical_ablation_15runs"
LOG_ROOT="$ROOT/logs/table9_canonical_ablation_15runs"
TRAIN_LOG_DIR="$LOG_ROOT/training"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TRAIN_LOG_DIR"

run_task () {
  local gpu="$1"
  local variant="$2"
  local seed="$3"

  echo "[GPU $gpu] START variant=$variant seed=$seed"

  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_ablation_layer2_official.py \
    --variant "$variant" \
    --seed "$seed" \
    --output_root "$OUTPUT_ROOT" \
    --log_dir "$TRAIN_LOG_DIR" \
    --pretrain_epochs 50 \
    --finetune_epochs 200 \
    --batch_size 64 \
    --num_workers 4 \
    --use_cached_dataset \
    > "$LOG_ROOT/${variant}_seed${seed}_gpu${gpu}.launcher.log" 2>&1

  echo "[GPU $gpu] DONE variant=$variant seed=$seed"
}

worker_0() {
  run_task 0 wo_Discriminator 42
  run_task 0 wo_Discriminator 52
}

worker_1() {
  run_task 1 wo_Discriminator 62
  run_task 1 wo_Discriminator 72
}

worker_2() {
  run_task 2 wo_Discriminator 82
  run_task 2 wo_Dual_Attention 42
}

worker_3() {
  run_task 3 wo_Dual_Attention 52
  run_task 3 wo_Dual_Attention 62
}

worker_4() {
  run_task 4 wo_Dual_Attention 72
  run_task 4 wo_Dual_Attention 82
}

worker_5() {
  run_task 5 wo_Oversampling 42
  run_task 5 wo_Oversampling 52
}

worker_6() {
  run_task 6 wo_Oversampling 62
  run_task 6 wo_Oversampling 72
}

worker_7() {
  run_task 7 wo_Oversampling 82
}

export -f run_task

worker_0 &
worker_1 &
worker_2 &
worker_3 &
worker_4 &
worker_5 &
worker_6 &
worker_7 &

wait
echo "ALL TABLE9 CANONICAL ABLATION RUNS FINISHED."
