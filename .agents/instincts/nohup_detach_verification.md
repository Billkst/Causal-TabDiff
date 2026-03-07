---
description: For long-running experiments, never rely on tool background session alone; always use true nohup/screen/tmux detachment with PID and liveness verification.
triggers:
  - user asks to run baseline/training/evaluation expected >5 minutes
  - command uses run_baselines.py or run_experiment.py
  - any request includes "后台", "nohup", "断线不断" or "守护"
confidence: true
---

# Instinct: Real Detach Verification

## Mistake Captured
Using an IDE/tool-level background session (`isBackground=true`) was incorrectly treated as equivalent to daemonized execution. This can still be tied to session lifecycle and risks termination after disconnect/reset.

## BAD Pattern
```bash
# BAD: tool background or plain & without robust detach guarantees
python run_baselines.py --model "TabDiff (ICLR 25)" &
```

## GOOD Pattern
```bash
# GOOD: true detach + log redirect + PID persistence + liveness check
mkdir -p logs/evaluation
nohup env DATASET_METADATA_PATH=src/data/dataset_metadata_noleak.json \
  /home/UserData/miniconda/envs/causal_tabdiff/bin/python run_baselines.py --model "TabDiff (ICLR 25)" \
  >> logs/evaluation/nohup_tabdiff_noleak.log 2>&1 < /dev/null &
echo $! > logs/evaluation/nohup_tabdiff_noleak.pid
ps -p "$(cat logs/evaluation/nohup_tabdiff_noleak.pid)" -o pid,ppid,stat,etime,cmd
```

## Mandatory Completion Signal
After launch, always return all of:
1. PID file path
2. Log file path
3. Liveness proof (`ps -p <pid> ...`)
4. Fresh log tail showing new progress lines

## Failure Recovery
If job stops unexpectedly:
1. verify PID existence and process liveness,
2. inspect last 80 log lines,
3. relaunch with true `nohup` command,
4. re-verify with PID + log growth.
