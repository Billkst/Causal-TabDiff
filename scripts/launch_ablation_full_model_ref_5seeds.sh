#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT" || exit 1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submission_v3_5seeds/ablation_layer2_full_model_ref}"
LOG_ROOT="${LOG_ROOT:-logs/ablation_5seeds_full_model_ref}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-logs/training_ablation_5seeds_full_model_ref}"
STATUS_ROOT="${STATUS_ROOT:-${OUTPUT_ROOT}/status}"
BATCH_SIZE="${BATCH_SIZE:-94739}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run_ablation_layer2_official.py}"
SEEDS=(42 52 62 72 82)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TRAINING_LOG_DIR" "$STATUS_ROOT" "$OUTPUT_ROOT/Full_Model"

echo "Full_Model reference 5-seed launch"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Launcher log root: $LOG_ROOT"
echo "Training log dir: $TRAINING_LOG_DIR"
echo "Status root: $STATUS_ROOT"
echo "Run stamp: $RUN_STAMP"
echo "Protocol: variant=Full_Model batch_size=${BATCH_SIZE} num_workers=${DATALOADER_WORKERS} use_cached_dataset=true"
echo "GPU mapping: seed42->GPU0 seed52->GPU1 seed62->GPU2 seed72->GPU3 seed82->GPU4"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu_id="$i"
  pred_path="${OUTPUT_ROOT}/Full_Model/Full_Model_seed${seed}.npz"
  eff_path="${OUTPUT_ROOT}/Full_Model/Full_Model_seed${seed}_efficiency.json"
  launcher_log="${LOG_ROOT}/Full_Model_seed${seed}_${RUN_STAMP}.launcher.log"
  status_file="${STATUS_ROOT}/Full_Model_seed${seed}.status"

  if [[ "${FORCE:-0}" != "1" && -s "$pred_path" && -s "$eff_path" ]]; then
    echo "SKIP existing Full_Model seed ${seed}: ${pred_path}"
    {
      echo "status=skipped_existing"
      echo "variant=Full_Model"
      echo "seed=${seed}"
      echo "gpu_id=${gpu_id}"
      echo "pred_path=${pred_path}"
      echo "efficiency_path=${eff_path}"
      echo "timestamp=$(date -Iseconds)"
    } > "$status_file"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu_id" ABLATION_GPU_ID="$gpu_id" \
    nohup "$PYTHON_BIN" -u "$TRAIN_SCRIPT" \
      --variant Full_Model \
      --seed "$seed" \
      --output_root "$OUTPUT_ROOT" \
      --log_dir "$TRAINING_LOG_DIR" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$DATALOADER_WORKERS" \
      --use_cached_dataset \
      > "$launcher_log" 2>&1 &
  worker_pid="$!"
  echo "$worker_pid" > "${STATUS_ROOT}/Full_Model_seed${seed}_${RUN_STAMP}.pid"
  {
    echo "status=launched"
    echo "variant=Full_Model"
    echo "seed=${seed}"
    echo "gpu_id=${gpu_id}"
    echo "pid=${worker_pid}"
    echo "pred_path=${pred_path}"
    echo "efficiency_path=${eff_path}"
    echo "launcher_log=${launcher_log}"
    echo "timestamp=$(date -Iseconds)"
  } > "$status_file"
  echo "Started Full_Model seed ${seed}: gpu=${gpu_id}, pid=${worker_pid}, log=${launcher_log}"
done

{
  echo "run_stamp=${RUN_STAMP}"
  echo "variant=Full_Model"
  echo "seeds=42,52,62,72,82"
  echo "output_root=${OUTPUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "training_log_dir=${TRAINING_LOG_DIR}"
  echo "status_root=${STATUS_ROOT}"
  echo "batch_size=${BATCH_SIZE}"
  echo "num_workers=${DATALOADER_WORKERS}"
  echo "use_cached_dataset=true"
  echo "gpu_mapping=42:0,52:1,62:2,72:3,82:4"
  echo "ablation_variant_policy=do_not_rerun_existing_15_ablation_variants"
} > "${STATUS_ROOT}/launch_${RUN_STAMP}.manifest"

echo "Launch manifest: ${STATUS_ROOT}/launch_${RUN_STAMP}.manifest"
echo "Monitor launcher logs: tail -f ${LOG_ROOT}/Full_Model_seed*_${RUN_STAMP}.launcher.log"
echo "Monitor training logs: tail -f ${TRAINING_LOG_DIR}/ablation_layer2_Full_Model_seed*.log"
