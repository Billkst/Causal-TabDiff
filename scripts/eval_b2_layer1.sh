#!/bin/bash
echo "=== 自动评估 Layer1 结果 ==="
SEEDS=(42 52 62 72 82)

for seed in "${SEEDS[@]}"; do
    pred_file="outputs/b2_baseline/layer1/causal_forest_seed${seed}_predictions.npz"
    if [ -f "$pred_file" ]; then
        echo "评估 CausalForest seed=${seed}"
        python evaluate_real_anchors.py \
            --predictions_file $pred_file \
            --output_dir outputs/b2_baseline/layer1 \
            --model_name CausalForest_seed${seed} \
            > logs/b2_baseline/layer1/causal_forest_seed${seed}_eval.log 2>&1
    fi
done

for seed in "${SEEDS[@]}"; do
    pred_file="outputs/tslib_models/itransformer_seed${seed}_predictions.npz"
    if [ -f "$pred_file" ]; then
        echo "评估 iTransformer seed=${seed}"
        python -c "
import numpy as np
import sys
sys.path.insert(0, 'src')
from evaluation.metrics import compute_all_metrics
from evaluation.plots import generate_all_plots
import json, os

data = np.load('$pred_file')
metrics = compute_all_metrics(data['y_true'], data['y_pred'], threshold=0.5)
os.makedirs('outputs/b2_baseline/layer1', exist_ok=True)
with open('outputs/b2_baseline/layer1/iTransformer_seed${seed}_metrics.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
               for k, v in metrics.items() if k != 'confusion_matrix'}, f, indent=2)
y_pred_binary = (data['y_pred'] >= 0.5).astype(int)
generate_all_plots(data['y_true'], data['y_pred'], y_pred_binary, 
                  'outputs/b2_baseline/layer1/iTransformer_seed${seed}')
print('✓ iTransformer seed=${seed} 评估完成')
" > logs/b2_baseline/layer1/itransformer_seed${seed}_eval.log 2>&1
    fi
done

echo "✓ Layer1 评估完成"
