#!/bin/bash
set -euo pipefail

SEEDS=(42 52 62 72 82)
FORMAL_DIR="outputs/b2_baseline/formal_runs"
mkdir -p "$FORMAL_DIR/layer1" "$FORMAL_DIR/tstr" "$FORMAL_DIR/layer2" logs/b2_lockdown

echo "[1/5] CausalForest formal rerun"
for seed in "${SEEDS[@]}"; do
  python train_causal_forest_b2.py --seed "$seed" --output_dir "$FORMAL_DIR/layer1"
  python evaluate_model.py \
    --model_type baseline \
    --predictions_file "$FORMAL_DIR/layer1/causal_forest_seed${seed}_predictions.npz" \
    --output_dir "$FORMAL_DIR/layer1" \
    --model_name "CausalForest_seed${seed}"
done

echo "[2/5] Layer1 formal reruns: iTransformer/TSDiff/STaSy"
for seed in "${SEEDS[@]}"; do
  python train_tslib_models.py --model itransformer --seed "$seed" --epochs 50 --output_dir "$FORMAL_DIR/layer1"
  python evaluate_model.py --model_type baseline --predictions_file "$FORMAL_DIR/layer1/itransformer_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/layer1" --model_name "iTransformer_seed${seed}"
  python train_diffusion_layer1_formal.py --model tsdiff --seed "$seed" --epochs 5 --output_dir "$FORMAL_DIR/layer1"
  python evaluate_model.py --model_type baseline --predictions_file "$FORMAL_DIR/layer1/tsdiff_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/layer1" --model_name "TSDiff_seed${seed}"
  python train_diffusion_layer1_formal.py --model stasy --seed "$seed" --epochs 5 --output_dir "$FORMAL_DIR/layer1"
  python evaluate_model.py --model_type baseline --predictions_file "$FORMAL_DIR/layer1/stasy_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/layer1" --model_name "STaSy_seed${seed}"
done

echo "[3/5] TSTR formal reruns/recomputes"
for seed in "${SEEDS[@]}"; do
  python recompute_tstr_pipeline_predictions.py --model stasy --seed "$seed" --pipeline_path "outputs/tstr_baselines/stasy_seed${seed}_pipeline.pkl" --output_dir "$FORMAL_DIR/tstr"
  python evaluate_tstr.py --predictions_file "$FORMAL_DIR/tstr/stasy_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/tstr" --model_name "stasy_seed${seed}"
  python recompute_tstr_pipeline_predictions.py --model tsdiff --seed "$seed" --pipeline_path "outputs/tstr_baselines/tsdiff_seed${seed}_pipeline.pkl" --output_dir "$FORMAL_DIR/tstr"
  python evaluate_tstr.py --predictions_file "$FORMAL_DIR/tstr/tsdiff_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/tstr" --model_name "tsdiff_seed${seed}"
  for model in tabsyn tabdiff survtraj sssd; do
    python train_generative_strict.py --model "$model" --seed "$seed" --epochs 30 --n_synthetic 1000 --output_dir "$FORMAL_DIR/tstr"
    if [ -f "$FORMAL_DIR/tstr/${model}_seed${seed}_predictions.npz" ]; then
      python evaluate_tstr.py --predictions_file "$FORMAL_DIR/tstr/${model}_seed${seed}_predictions.npz" --output_dir "$FORMAL_DIR/tstr" --model_name "${model}_seed${seed}"
    fi
  done
done

echo "[4/5] Layer2 formal reruns"
for seed in "${SEEDS[@]}"; do
  python train_tslib_layer2.py --model itransformer --seed "$seed" --epochs 30 --output_dir "$FORMAL_DIR/layer2"
  python evaluate_layer2.py --predictions_file "$FORMAL_DIR/layer2/itransformer_seed${seed}_layer2.npz" --output_dir "$FORMAL_DIR/layer2" --model_name "iTransformer_seed${seed}"
  python train_tslib_layer2.py --model timexer --seed "$seed" --epochs 30 --output_dir "$FORMAL_DIR/layer2"
  python evaluate_layer2.py --predictions_file "$FORMAL_DIR/layer2/timexer_seed${seed}_layer2.npz" --output_dir "$FORMAL_DIR/layer2" --model_name "TimeXer_seed${seed}"
  python train_generative_layer2.py --model sssd --seed "$seed" --epochs 30 --output_dir "$FORMAL_DIR/layer2"
  python evaluate_layer2.py --predictions_file "$FORMAL_DIR/layer2/sssd_seed${seed}_layer2.npz" --output_dir "$FORMAL_DIR/layer2" --model_name "SSSD_seed${seed}"
  python train_generative_layer2.py --model survtraj --seed "$seed" --epochs 30 --output_dir "$FORMAL_DIR/layer2"
  python evaluate_layer2.py --predictions_file "$FORMAL_DIR/layer2/survtraj_seed${seed}_layer2.npz" --output_dir "$FORMAL_DIR/layer2" --model_name "SurvTraj_seed${seed}"
done

echo "[5/5] 生成正式表格和一致性检查"
python scripts/finalize_b2_official_5seed.py

echo "✓ baseline minimal lockdown completed"
