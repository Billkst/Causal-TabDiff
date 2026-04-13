#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT" || exit 1

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submission_v3_5seeds/ablation_layer2}"
LOG_ROOT="${LOG_ROOT:-logs/ablation_5seeds}"
STATUS_ROOT="${STATUS_ROOT:-${OUTPUT_ROOT}/status}"
WORKER_COUNT="${WORKER_COUNT:-8}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

echo "Ablation 5-seed launch"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Log root: $LOG_ROOT"
echo "Status root: $STATUS_ROOT"
echo "Worker count: $WORKER_COUNT"
echo "Run stamp: $RUN_STAMP"

for worker_id in $(seq 0 "$((WORKER_COUNT - 1))"); do
  worker_log="${LOG_ROOT}/worker_${worker_id}_${RUN_STAMP}.launcher.log"
  CUDA_VISIBLE_DEVICES="$worker_id" ABLATION_GPU_ID="$worker_id" \
    nohup bash scripts/run_ablation_5seeds_worker.sh "$worker_id" > "$worker_log" 2>&1 &
  worker_pid="$!"
  echo "$worker_pid" > "${STATUS_ROOT}/worker_${worker_id}_${RUN_STAMP}.pid"
  echo "Started worker ${worker_id}: gpu=${worker_id}, pid=${worker_pid}, log=${worker_log}"
done

{
  echo "run_stamp=${RUN_STAMP}"
  echo "worker_count=${WORKER_COUNT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "status_root=${STATUS_ROOT}"
  echo "variants=wo_Discriminator,wo_Dual_Attention,wo_Oversampling"
  echo "seeds=42,52,62,72,82"
  echo "full_model_policy=reuse_existing_official_5seed_do_not_rerun"
  echo "gpu_mapping=worker_id_equals_cuda_visible_devices"
} > "${STATUS_ROOT}/launch_${RUN_STAMP}.manifest"

echo "Launch manifest: ${STATUS_ROOT}/launch_${RUN_STAMP}.manifest"
echo "Monitor: tail -f ${LOG_ROOT}/worker_*_${RUN_STAMP}.launcher.log"
echo "Check completeness: python -u scripts/aggregate_ablation_5seeds.py --check-only"
