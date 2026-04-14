#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT" || exit 1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_ablation_v2_5seeds}"
LOG_ROOT="${LOG_ROOT:-logs/local_ablation_v2_5seeds}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-${LOG_ROOT}/training}"
STATUS_ROOT="${STATUS_ROOT:-${OUTPUT_ROOT}/status}"
BATCH_SIZE="${BATCH_SIZE:-94739}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run_ablation_layer2_official.py}"
RUN_SET="${RUN_SET:-20run}"
FORCE="${FORCE:-0}"

SEEDS=(42 52 62 72 82)
case "$RUN_SET" in
  20run)
    VARIANTS=(Full_Model_v2 wo_Discriminator_v2 wo_Oversampling_v2 wo_Time_Attention_v2)
    ;;
  15run)
    VARIANTS=(Full_Model_v2 wo_Discriminator_v2 wo_Oversampling_v2)
    ;;
  smoke)
    SEEDS=(42)
    VARIANTS=(Full_Model_v2)
    OUTPUT_ROOT="${OUTPUT_ROOT}_smoke"
    LOG_ROOT="${LOG_ROOT}_smoke"
    TRAINING_LOG_DIR="${LOG_ROOT}/training"
    STATUS_ROOT="${OUTPUT_ROOT}/status"
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}. Use 20run, 15run, or smoke." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TRAINING_LOG_DIR" "$STATUS_ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${LOG_ROOT}/launcher_${RUN_STAMP}.log"

{
  echo "Local ablation v2 serial launcher"
  echo "Time: $(date -Iseconds)"
  echo "Project root: $PROJECT_ROOT"
  echo "Run set: $RUN_SET"
  echo "Variants: ${VARIANTS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Output root: $OUTPUT_ROOT"
  echo "Log root: $LOG_ROOT"
  echo "Training log dir: $TRAINING_LOG_DIR"
  echo "Status root: $STATUS_ROOT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICE"
  echo "Protocol: batch_size=${BATCH_SIZE}, num_workers=${DATALOADER_WORKERS}, use_cached_dataset=true"
} | tee "$MASTER_LOG"

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    pred_path="${OUTPUT_ROOT}/${variant}/${variant}_seed${seed}.npz"
    eff_path="${OUTPUT_ROOT}/${variant}/${variant}_seed${seed}_efficiency.json"
    status_file="${STATUS_ROOT}/${variant}_seed${seed}.status"
    run_log="${LOG_ROOT}/${variant}_seed${seed}_${RUN_STAMP}.launcher.log"

    mkdir -p "${OUTPUT_ROOT}/${variant}"

    if [[ "$FORCE" != "1" && -s "$pred_path" && -s "$eff_path" ]]; then
      echo "SKIP existing ${variant} seed ${seed}: ${pred_path}" | tee -a "$MASTER_LOG"
      {
        echo "status=skipped_existing"
        echo "variant=${variant}"
        echo "seed=${seed}"
        echo "pred_path=${pred_path}"
        echo "efficiency_path=${eff_path}"
        echo "timestamp=$(date -Iseconds)"
      } > "$status_file"
      continue
    fi

    echo "START ${variant} seed ${seed} | $(date -Iseconds)" | tee -a "$MASTER_LOG"
    echo "Log: ${run_log}" | tee -a "$MASTER_LOG"
    {
      echo "status=running"
      echo "variant=${variant}"
      echo "seed=${seed}"
      echo "cuda_visible_devices=${CUDA_DEVICE}"
      echo "batch_size=${BATCH_SIZE}"
      echo "num_workers=${DATALOADER_WORKERS}"
      echo "timestamp=$(date -Iseconds)"
    } > "$status_file"

    start_ts="$(date +%s)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -u "$TRAIN_SCRIPT" \
      --variant "$variant" \
      --seed "$seed" \
      --output_root "$OUTPUT_ROOT" \
      --log_dir "$TRAINING_LOG_DIR" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$DATALOADER_WORKERS" \
      --use_cached_dataset \
      > "$run_log" 2>&1
    rc="$?"
    end_ts="$(date +%s)"
    elapsed_sec="$((end_ts - start_ts))"

    if [[ "$rc" -eq 0 && -s "$pred_path" && -s "$eff_path" ]]; then
      echo "DONE ${variant} seed ${seed} | elapsed=${elapsed_sec}s" | tee -a "$MASTER_LOG"
      {
        echo "status=ok"
        echo "exit_code=${rc}"
        echo "variant=${variant}"
        echo "seed=${seed}"
        echo "elapsed_sec=${elapsed_sec}"
        echo "pred_path=${pred_path}"
        echo "efficiency_path=${eff_path}"
        echo "timestamp=$(date -Iseconds)"
      } > "$status_file"
    else
      echo "FAILED ${variant} seed ${seed} | exit=${rc} elapsed=${elapsed_sec}s" | tee -a "$MASTER_LOG"
      {
        echo "status=failed"
        echo "exit_code=${rc}"
        echo "variant=${variant}"
        echo "seed=${seed}"
        echo "elapsed_sec=${elapsed_sec}"
        echo "pred_path=${pred_path}"
        echo "efficiency_path=${eff_path}"
        echo "timestamp=$(date -Iseconds)"
      } > "$status_file"
      if [[ "${FAIL_FAST:-0}" == "1" ]]; then
        exit "$rc"
      fi
    fi
  done
done

echo "ALL REQUESTED LOCAL ABLATION V2 RUNS FINISHED | $(date -Iseconds)" | tee -a "$MASTER_LOG"
echo "Aggregate: python -u scripts/aggregate_local_ablation_v2_5seeds.py --output-root $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
