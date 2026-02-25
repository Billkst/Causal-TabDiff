#!/bin/bash
# ==============================================================================
# Causal-TabDiff: Server Full Baseline Evaluation Script
# ==============================================================================

# 1. 自动守护进程隔离 (Self-Daemonizing SIGHUP Protection)
# 如果脚本在控制台交互式运行，自动挂载 nohup 以后台脱机模式重启自身，保护整个跑批循环不被断线掐除
if [ -t 1 ]; then
    mkdir -p logs
    echo "⚠️ 检测到活动终端！正在触发全量跑批任务后台剥离 (Nohup Detachment)..."
    nohup bash "$0" "$@" > logs/server_master.log 2>&1 &
    echo "✅ 任务已隐式入驻守护进程！主控日志: logs/server_master.log. PID: $!"
    echo "   您可以放心地关闭 SSH 窗口或者关机了。"
    exit 0
fi

# ================= 以下为主跑批逻辑 (执行于后台守护态) ==================

# 2. 显卡锁死 (GPU Visibility)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 3. 确立独立的日志流分流存储目标 (Log Rotation Directory)
mkdir -p logs

# 3.1 选择解释器：优先使用目标环境中的 python，避免 conda run 额外缓冲
ENV_PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"
if [ ! -x "$ENV_PYTHON" ]; then
    echo "⚠️ 未找到 $ENV_PYTHON，回退到 conda run 模式。"
fi

# 4. 自动化模型清单 (Model Array for Sequential Pipeline)
# 🚨 严格合规审查：绝不附带 --debug_mode！确保全部 Epoch 和真实全量数据释放！
MODELS=(
    "CausalForest (Classic)"
    "STaSy (ICLR 23)"
    "TabSyn (ICLR 24)"
    "TabDiff (ICLR 25)"
    "TSDiff (ICLR 23)"
)

echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] Causal-TabDiff 终极服务器全量基线评估跑批启动！"

for MODEL in "${MODELS[@]}"; do
    # 提取括号前的单词作为纯净的日志文件标识，例："TabSyn (ICLR 24)" -> "tabsyn"
    LOG_NAME=$(echo "$MODEL" | awk '{print tolower($1)}')
    LOG_FILE="logs/${LOG_NAME}_full.log"
    
    echo "================================================================"
    echo "⏳ [跑批中] 当前挂载评估任务: $MODEL"
    echo "   标准输出与跟踪进度重定向至 -> $LOG_FILE"
    echo "🕒 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 5. 模型级自动化排队执行串行 (Sequential Automated Dispatch)
    # 利用我们在 run_baselines.py 刚才新增的 --model 单点驱动参数独立切片运行
    if [ -x "$ENV_PYTHON" ]; then
        BASELINE_LOG_FILE="$LOG_FILE" BASELINE_DISABLE_STREAM=1 "$ENV_PYTHON" -u run_baselines.py --model "$MODEL" > "$LOG_FILE" 2>&1 &
    else
        BASELINE_LOG_FILE="$LOG_FILE" BASELINE_DISABLE_STREAM=1 conda run --no-capture-output -n causal_tabdiff python -u run_baselines.py --model "$MODEL" > "$LOG_FILE" 2>&1 &
    fi
    MODEL_PID=$!
    while kill -0 "$MODEL_PID" 2>/dev/null; do
        NOW_TS=$(date +%s)
        if [ -f "$LOG_FILE" ]; then
            LOG_MTIME=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo "$NOW_TS")
            STALE_SEC=$((NOW_TS - LOG_MTIME))
        else
            STALE_SEC=-1
        fi
        PROC_STAT=$(ps -p "$MODEL_PID" -o %cpu,%mem,etime --no-headers 2>/dev/null | xargs)
        echo "⏱ [$MODEL] 仍在运行... $(date '+%Y-%m-%d %H:%M:%S') | pid=$MODEL_PID | cpu/mem/etime=${PROC_STAT:-N/A} | log_stale=${STALE_SEC}s"
        if [ "$STALE_SEC" -ge 300 ] 2>/dev/null; then
            echo "⚠️ [$MODEL] 超过 300s 无新日志，可能在底层库长时间计算（如 forest fit），请继续观察。"
        fi
        sleep 60
    done
    wait "$MODEL_PID"
    EXIT_CODE=$?
    echo "🧾 [$MODEL] 退出码: $EXIT_CODE"
    echo "📌 [$MODEL] 最近日志摘要(最后5行):"
    tail -n 5 "$LOG_FILE" 2>/dev/null || true
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ [$MODEL] 执行失败，已中断后续模型。"
        exit $EXIT_CODE
    fi
    
    # 由于整个 bash 已经在外层的 nohup 中脱机运行，所以这里直接串行执行即可，无需额外的后台标记和 wait
    
    echo "✅ [$MODEL] 基线演算结案收敛！结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] 所有的基线模型全量数据跑批完毕！成绩单汇总已成形！"
