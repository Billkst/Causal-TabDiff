#!/bin/bash
export DATASET_METADATA_PATH='src/data/dataset_metadata_noleak.json'
export METRICS_ALIGN_FAKE_PREVALENCE=1

echo "=========================================================="
echo "Starting FULL baseline evaluation sequentially (No-Leak + Optimal F1 env)"
echo "Time: $(date)"
echo "=========================================================="

# Keys need to match the actual dictionary keys exactly
MODELS=("CausalForest (Classic)" "STaSy (ICLR 23)" "TabSyn (ICLR 24)" "TabDiff (ICLR 25)" "TSDiff (ICLR 23)")

for model in "${MODELS[@]}"; do
    echo ">>> Running ${model}..."
    /home/UserData/miniconda/envs/causal_tabdiff/bin/python run_baselines.py --model "${model}"
done

echo "=========================================================="
echo "All baseline evaluations completed. markdown_report.md updated."
echo "Time: $(date)"
echo "=========================================================="
