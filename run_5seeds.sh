#!/bin/bash
for seed in 42 52 62 72 82; do
  echo "=== Starting seed $seed ==="
  nohup conda run -n causal_tabdiff python -u run_experiment_landmark.py \
    --seed $seed \
    --epochs 100 \
    --batch_size 512 \
    > logs/training/run_landmark_seed${seed}_5seed.log 2>&1 &
  echo "Seed $seed started, PID: $!"
  sleep 2
done
echo "All 5 seeds launched"
