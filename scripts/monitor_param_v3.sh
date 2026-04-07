#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

echo "========================================"
echo "Submission V3 Parameter Discussion Monitor"
echo "Time: $(date)"
echo "========================================"

DEST="outputs/submission_v3/param_discussion_layer2"
LOG_ROOT="logs/param_discussion_layer2_v3"

EXPECTED_NEW=(
    "focal_gamma/focal_gamma_4.0_seed42"
    "oversample_ratio/oversample_ratio_2.5_seed42"
    "oversample_ratio/oversample_ratio_7.5_seed42"
    "disc_weight/disc_weight_0.01_seed42"
    "disc_weight/disc_weight_0.05_seed42"
    "disc_weight/disc_weight_0.1_seed42"
    "disc_weight/disc_weight_0.2_seed42"
)

echo ""
echo "--- New Runs Status ---"
for run in "${EXPECTED_NEW[@]}"; do
    npz_official="${DEST}/${run}_official.npz"
    npz_raw="${DEST}/${run}.npz"
    param=$(echo "$run" | cut -d'/' -f1)
    base=$(basename "$run")
    log="${LOG_ROOT}/${param}/${base}.log"
    
    if [ -f "$npz_official" ]; then
        echo "  ✅ ${run} — DONE (official)"
    elif [ -f "$npz_raw" ]; then
        echo "  ✅ ${run} — DONE (pending rename)"
    elif [ -f "$log" ]; then
        last_line=$(tail -c 200 "$log" 2>/dev/null | strings | tail -1)
        echo "  🔄 ${run} — RUNNING: ${last_line:0:80}"
    else
        echo "  ⏳ ${run} — QUEUED"
    fi
done

echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo "--- Process Check ---"
ps aux | grep "run_param_discussion" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "CMD:", substr($0, index($0,$11))}' || echo "  No running process"
