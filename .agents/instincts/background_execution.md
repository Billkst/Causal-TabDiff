---
description: Always use remote-resilient daemon methods (nohup/screen/tmux) for long-running scripts (like training models) to prevent termination when the user disconnects or the pseudoterminal drops.
triggers:
  - user asks to run a training, evaluation, or heavy computational script
  - any script involving `epochs`, `fit`, `train`, `baseline`, or heavy `run_*.py` files
  - command is expected to take longer than 5 minutes
confidence: true
---

# Instinct: Daemonize Long-Running Processes

## Context/Observation
When the user asks to start heavy training loops or large-scale evaluation baselines on remote servers, simply using the standard terminal execution or backgrounding with `&` without disconnecting standard streams will bind the process to the current VS Code terminal session (`pts`). 
If the user's connection drops, they close their laptop, or the VS Code remote connection resets, a `SIGHUP` signal is sent, completely killing the running experiment and causing catastrophic loss of ephemeral state (like models training in-memory without checkpoints).

## Action Rule
1. **Never** run long-duration Python code (like `run_baselines.py` or `run_experiment.py`) natively in the foreground or just as a standard bash background command (`python script.py &`).
2. **Always** wrap execution using `nohup` (or `screen`/`tmux` if an interactive viewer is requested). 
3. **Always** redirect output strictly to a log file (`> log.log 2>&1 &`). 
4. **Validation**: Explicitly inform the user that the process is completely detached ("daemonized") and can survive a network disconnection.

### Example
**Bad (Vulnerable to disconnect):**
```bash
python run_baselines.py --model "TabSyn"
```
**Good (Resilient):**
```bash
nohup python run_baselines.py > logs/evaluation/nohup_tabsyn.log 2>&1 &
```
