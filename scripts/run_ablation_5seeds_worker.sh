#!/usr/bin/env bash
set -u

WORKER_ID="${1:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT" || exit 1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submission_v3_5seeds/ablation_layer2}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-logs/training_ablation_5seeds}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-logs/ablation_5seeds}"
STATUS_ROOT="${STATUS_ROOT:-${OUTPUT_ROOT}/status}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run_ablation_layer2_official.py}"

mkdir -p "$OUTPUT_ROOT" "$TRAINING_LOG_DIR" "$WORKER_LOG_DIR" "$STATUS_ROOT"

case "$WORKER_ID" in
  0) RUNS=("wo_Oversampling 42") ;;
  1) RUNS=("wo_Oversampling 52") ;;
  2) RUNS=("wo_Oversampling 62") ;;
  3) RUNS=("wo_Oversampling 72") ;;
  4) RUNS=("wo_Oversampling 82") ;;
  5) RUNS=("wo_Discriminator 42" "wo_Discriminator 52" "wo_Dual_Attention 42") ;;
  6) RUNS=("wo_Discriminator 62" "wo_Discriminator 72" "wo_Dual_Attention 52") ;;
  7) RUNS=("wo_Discriminator 82" "wo_Dual_Attention 62" "wo_Dual_Attention 72" "wo_Dual_Attention 82") ;;
  *)
    echo "Unknown worker id: ${WORKER_ID}. Expected 0-7." | tee -a "${WORKER_LOG_DIR}/worker_invalid.log"
    exit 2
    ;;
esac

echo "Worker ${WORKER_ID} started at $(date -Iseconds)"
echo "Assigned runs: ${RUNS[*]}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Training log dir: ${TRAINING_LOG_DIR}"
echo "Dataloader workers per run: ${DATALOADER_WORKERS}"
echo "Assigned physical GPU: ${ABLATION_GPU_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

for run_spec in "${RUNS[@]}"; do
  read -r variant seed <<< "$run_spec"
  variant_dir="${OUTPUT_ROOT}/${variant}"
  pred_path="${variant_dir}/${variant}_seed${seed}.npz"
  efficiency_path="${variant_dir}/${variant}_seed${seed}_efficiency.json"
  status_file="${STATUS_ROOT}/${variant}_seed${seed}.status"
  done_file="${STATUS_ROOT}/${variant}_seed${seed}.done"
  failed_file="${STATUS_ROOT}/${variant}_seed${seed}.failed"

  mkdir -p "$variant_dir"

  if [[ "${FORCE:-0}" != "1" && -s "$pred_path" && -s "$efficiency_path" ]]; then
    echo "SKIP existing ${variant} seed ${seed}: ${pred_path}"
    {
      echo "status=skipped_existing"
      echo "variant=${variant}"
      echo "seed=${seed}"
      echo "pred_path=${pred_path}"
      echo "efficiency_path=${efficiency_path}"
      echo "timestamp=$(date -Iseconds)"
    } > "$status_file"
    touch "$done_file"
    rm -f "$failed_file"
    continue
  fi

  rm -f "$done_file" "$failed_file"

  cmd=(
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT"
    --variant "$variant"
    --seed "$seed"
    --output_root "$OUTPUT_ROOT"
    --log_dir "$TRAINING_LOG_DIR"
    --num_workers "$DATALOADER_WORKERS"
    --use_cached_dataset
  )

  if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
    cmd+=(--debug_mode)
  fi

  start_ts="$(date +%s)"
  echo "RUN ${variant} seed ${seed} at $(date -Iseconds)"
  echo "GPU binding: worker=${WORKER_ID}, physical_gpu=${ABLATION_GPU_ID:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CMD: ${cmd[*]}"

  "${cmd[@]}"
  rc="$?"
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"

  if [[ "$rc" -eq 0 && -s "$pred_path" && -s "$efficiency_path" ]]; then
    {
      echo "status=ok"
      echo "exit_code=${rc}"
      echo "variant=${variant}"
      echo "seed=${seed}"
      echo "elapsed_sec=${elapsed_sec}"
      echo "pred_path=${pred_path}"
      echo "efficiency_path=${efficiency_path}"
      echo "timestamp=$(date -Iseconds)"
    } > "$status_file"
    touch "$done_file"
    rm -f "$failed_file"
    echo "OK ${variant} seed ${seed}: elapsed=${elapsed_sec}s"
  else
    {
      echo "status=failed"
      echo "exit_code=${rc}"
      echo "variant=${variant}"
      echo "seed=${seed}"
      echo "elapsed_sec=${elapsed_sec}"
      echo "pred_path=${pred_path}"
      echo "efficiency_path=${efficiency_path}"
      echo "timestamp=$(date -Iseconds)"
    } > "$status_file"
    cp "$status_file" "$failed_file"
    echo "FAILED ${variant} seed ${seed}: exit=${rc}, elapsed=${elapsed_sec}s"
  fi
done

echo "Worker ${WORKER_ID} finished at $(date -Iseconds)"
