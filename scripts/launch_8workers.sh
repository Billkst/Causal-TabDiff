#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="outputs/submission_v3_5seeds/param_discussion_layer2"
LOG_ROOT="logs/5seeds_run"
CHECKPOINT_ROOT="checkpoints/5seeds_run"

mkdir -p "$OUTPUT_ROOT"/{focal_gamma,oversample_ratio,disc_weight}
mkdir -p "$LOG_ROOT"/progress
mkdir -p "$CHECKPOINT_ROOT"

echo "============================================"
echo "5-Seed 70-Run Launcher | 8 Workers"
echo "Time: $(date -u)"
echo "bs=94739 nw=0 seeds=42,0,1,2,3"
echo "Output: $OUTPUT_ROOT"
echo "Logs:   $LOG_ROOT"
echo "============================================"

bash scripts/run_worker.sh W0 0 \
  "disc_weight|0.2|42" \
  "disc_weight|0.01|0" \
  "disc_weight|0.05|2" \
  "oversample_ratio|7.5|2" \
  "oversample_ratio|2.5|0" \
  "focal_gamma|3.0|0" \
  "focal_gamma|2.0|3" \
  "oversample_ratio|10.0|2" \
  "focal_gamma|0.0|0" \
  > "$LOG_ROOT/worker_W0_stdout.log" 2>&1 &

bash scripts/run_worker.sh W1 1 \
  "disc_weight|0.2|0" \
  "disc_weight|0.01|1" \
  "disc_weight|0.05|3" \
  "oversample_ratio|7.5|3" \
  "oversample_ratio|2.5|1" \
  "focal_gamma|3.0|1" \
  "focal_gamma|4.0|42" \
  "oversample_ratio|10.0|0" \
  "focal_gamma|1.0|3" \
  > "$LOG_ROOT/worker_W1_stdout.log" 2>&1 &

bash scripts/run_worker.sh W2 2 \
  "disc_weight|0.2|1" \
  "disc_weight|0.01|2" \
  "disc_weight|0.0|42" \
  "oversample_ratio|7.5|42" \
  "disc_weight|0.1|1" \
  "focal_gamma|3.0|2" \
  "focal_gamma|4.0|3" \
  "focal_gamma|1.0|2" \
  > "$LOG_ROOT/worker_W2_stdout.log" 2>&1 &

bash scripts/run_worker.sh W3 3 \
  "disc_weight|0.2|2" \
  "disc_weight|0.01|3" \
  "disc_weight|0.0|0" \
  "oversample_ratio|7.5|0" \
  "disc_weight|0.1|2" \
  "focal_gamma|3.0|3" \
  "oversample_ratio|10.0|42" \
  "focal_gamma|1.0|42" \
  "focal_gamma|0.0|1" \
  > "$LOG_ROOT/worker_W3_stdout.log" 2>&1 &

bash scripts/run_worker.sh W4 4 \
  "disc_weight|0.2|3" \
  "disc_weight|0.05|42" \
  "disc_weight|0.05|1" \
  "oversample_ratio|7.5|1" \
  "disc_weight|0.1|3" \
  "focal_gamma|2.0|42" \
  "focal_gamma|4.0|0" \
  "oversample_ratio|10.0|3" \
  > "$LOG_ROOT/worker_W4_stdout.log" 2>&1 &

bash scripts/run_worker.sh W5 5 \
  "oversample_ratio|1.0|42" \
  "oversample_ratio|1.0|2" \
  "disc_weight|0.0|1" \
  "disc_weight|0.1|42" \
  "oversample_ratio|2.5|2" \
  "focal_gamma|2.0|0" \
  "focal_gamma|4.0|1" \
  "focal_gamma|1.0|0" \
  "focal_gamma|0.0|2" \
  > "$LOG_ROOT/worker_W5_stdout.log" 2>&1 &

bash scripts/run_worker.sh W6 6 \
  "oversample_ratio|1.0|0" \
  "oversample_ratio|1.0|3" \
  "disc_weight|0.0|2" \
  "disc_weight|0.1|0" \
  "oversample_ratio|2.5|3" \
  "focal_gamma|2.0|1" \
  "focal_gamma|4.0|2" \
  "focal_gamma|1.0|1" \
  "focal_gamma|0.0|3" \
  > "$LOG_ROOT/worker_W6_stdout.log" 2>&1 &

bash scripts/run_worker.sh W7 7 \
  "oversample_ratio|1.0|1" \
  "disc_weight|0.01|42" \
  "disc_weight|0.05|0" \
  "disc_weight|0.0|3" \
  "oversample_ratio|2.5|42" \
  "focal_gamma|3.0|42" \
  "focal_gamma|2.0|2" \
  "oversample_ratio|10.0|1" \
  "focal_gamma|0.0|42" \
  > "$LOG_ROOT/worker_W7_stdout.log" 2>&1 &

echo "All 8 workers launched. PIDs:"
jobs -l
echo ""
echo "Monitor: tail -f $LOG_ROOT/worker_W*_stdout.log"
echo "Progress: cat $LOG_ROOT/progress/worker_*_PROGRESS.json"

wait
echo ""
echo "============================================"
echo "ALL WORKERS FINISHED | $(date -u)"
echo "============================================"
