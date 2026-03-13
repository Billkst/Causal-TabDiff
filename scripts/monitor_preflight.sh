#!/bin/bash
echo "=== B2 Baseline Preflight 监控 ==="
echo ""
echo "运行中的训练任务:"
ps aux | grep -E "(train_tslib|train_tstr)" | grep -v grep | awk '{print $2, $11, $12, $13, $14}'
echo ""
echo "最新日志状态:"
echo "- iTransformer:" $(tail -1 logs/b2_baseline/preflight/itransformer_seed42.log 2>/dev/null || echo "未启动")
echo "- TimeXer:" $(tail -1 logs/b2_baseline/preflight/timexer_layer2_seed42.log 2>/dev/null || echo "未启动")
echo "- TabSyn:" $(tail -1 logs/b2_baseline/preflight/tabsyn_tstr_seed42.log 2>/dev/null || echo "未启动")
echo "- STaSy:" $(tail -1 logs/b2_baseline/preflight/stasy_tstr_seed42.log 2>/dev/null || echo "未启动")
echo "- TabDiff:" $(tail -1 logs/b2_baseline/preflight/tabdiff_tstr_seed42.log 2>/dev/null || echo "未启动")
echo "- TSDiff:" $(tail -1 logs/b2_baseline/preflight/tsdiff_tstr_seed42.log 2>/dev/null || echo "未启动")
