#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

for seed in 42 52 62 72 82; do
  echo "========== Starting Seed $seed at $(date) =========="
  conda run --no-capture-output -n causal_tabdiff python -u run_phase2_twostage.py \
    --seed $seed --pretrain_epochs 50 --finetune_epochs 200 --batch_size 256
  echo "========== Seed $seed Done at $(date) =========="
  echo ""
done

echo "ALL 5 SEEDS COMPLETE at $(date)"
