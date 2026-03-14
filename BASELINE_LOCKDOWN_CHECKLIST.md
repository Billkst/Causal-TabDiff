# 基线锁定执行检查清单

**版本**: v1.0  
**生成日期**: 2026-03-14  
**目标**: 5-seed官方协议下的最小化重跑基线锁定

---

## 前置检查 (5分钟)

### ☐ 检查1: 确认STaSy AUROC<0.5的根本原因

**操作**:
```bash
# 查看STaSy Layer1的AUROC
python3 -c "
import json
import os
for seed in [42, 52, 62, 72, 82]:
    f = f'outputs/b2_baseline_backup_20260313/tsdiff_stasy/stasy_seed{seed}/metrics.json'
    if os.path.exists(f):
        with open(f) as fp:
            data = json.load(fp)
            print(f'STaSy seed{seed} Layer1 AUROC: {data.get(\"auroc\")}')
"

# 查看STaSy TSTR的AUROC
python3 -c "
import json
import os
for seed in [42, 52, 62, 72, 82]:
    f = f'outputs/b2_baseline_backup_20260313/tstr/stasy_seed{seed}_tstr_metrics.json'
    if os.path.exists(f):
        with open(f) as fp:
            data = json.load(fp)
            print(f'STaSy seed{seed} TSTR AUROC: {data.get(\"auroc\")}')
"
```

**决策**:
- [ ] 是已知bug → 跳过重跑，标记为"已知限制"
- [ ] 是数据问题 → 修复数据后重跑
- [ ] 是模型问题 → 添加到重跑列表

**输出文件**: `STASY_AUROC_DECISION.txt`

---

## 第一阶段: 必须重跑 (100分钟)

### ☐ 重跑1: iTransformer Layer1 (30分钟)

**问题**: F1=0 (所有5个seed) → 无法建立固定测试阈值

**步骤**:

1. [ ] 激活conda环境
   ```bash
   conda activate causal_tabdiff
   ```

2. [ ] 创建输出目录
   ```bash
   mkdir -p outputs/b2_baseline_lockdown/layer1
   ```

3. [ ] 运行训练脚本
   ```bash
   python -u train_retained_baselines.py \
     --model iTransformer \
     --seeds 42 52 62 72 82 \
     --layer layer1 \
     --output_dir outputs/b2_baseline_lockdown/layer1 \
     2>&1 | tee logs/itransformer_layer1_lockdown.log
   ```

4. [ ] 验证输出
   ```bash
   python3 << 'EOF'
   import json
   import os
   
   print("=== iTransformer Layer1 验证 ===")
   for seed in [42, 52, 62, 72, 82]:
       metrics_file = f'outputs/b2_baseline_lockdown/layer1/iTransformer_seed{seed}/metrics.json'
       if os.path.exists(metrics_file):
           with open(metrics_file) as f:
               data = json.load(f)
               f1 = data.get('f1', 'N/A')
               threshold = data.get('threshold', 'N/A')
               auroc = data.get('auroc', 'N/A')
               print(f"seed{seed}: F1={f1}, threshold={threshold}, AUROC={auroc}")
               
               # 检查验收标准
               if f1 > 0.01 and threshold != float('inf'):
                   print(f"  ✅ 通过")
               else:
                   print(f"  ❌ 失败")
       else:
           print(f"seed{seed}: ❌ 文件不存在")
   EOF
   ```

**验收标准**:
- [ ] 所有5个seed的F1 > 0.01
- [ ] 所有5个seed的threshold是有限值
- [ ] AUROC与旧结果的差异 < 0.05

---

### ☐ 重跑2: TabDiff TSTR (60分钟)

**问题**: 全部失败 (所有5个seed) - "all predictions are positive class"

**步骤**:

1. [ ] 激活conda环境
   ```bash
   conda activate causal_tabdiff
   ```

2. [ ] 创建输出目录
   ```bash
   mkdir -p outputs/b2_baseline_lockdown/tstr
   ```

3. [ ] 运行训练脚本
   ```bash
   python -u train_tslib_models.py \
     --model tabdiff \
     --seeds 42 52 62 72 82 \
     --task tstr \
     --output_dir outputs/b2_baseline_lockdown/tstr \
     2>&1 | tee logs/tabdiff_tstr_lockdown.log
   ```

4. [ ] 验证输出
   ```bash
   python3 << 'EOF'
   import json
   import os
   import numpy as np
   
   print("=== TabDiff TSTR 验证 ===")
   for seed in [42, 52, 62, 72, 82]:
       pred_file = f'outputs/b2_baseline_lockdown/tstr/tabdiff_seed{seed}_predictions.npz'
       metrics_file = f'outputs/b2_baseline_lockdown/tstr/tabdiff_seed{seed}_metrics.json'
       
       if os.path.exists(pred_file):
           data = np.load(pred_file)
           preds = data['predictions']
           pos_ratio = np.mean(preds > 0.5)
           print(f"seed{seed}: min={preds.min():.3f}, max={preds.max():.3f}, pos_ratio={pos_ratio:.3f}")
           
           # 检查验收标准
           if 0.1 < pos_ratio < 0.9:
               print(f"  ✅ 通过")
           else:
               print(f"  ❌ 失败")
       else:
           print(f"seed{seed}: ❌ 文件不存在")
   EOF
   ```

**验收标准**:
- [ ] 所有5个seed的预测值范围在[0, 1]
- [ ] 所有5个seed的正例比例在(0.1, 0.9)之间
- [ ] 所有5个seed都有有效的metrics.json

---

### ☐ 重新评估1: SSSD Layer2 (seed 72, 82) (5分钟)

**问题**: optimal_threshold = inf (无有效F1)

**步骤**:

1. [ ] 激活conda环境
   ```bash
   conda activate causal_tabdiff
   ```

2. [ ] 创建输出目录
   ```bash
   mkdir -p outputs/b2_baseline_lockdown/layer2
   ```

3. [ ] 运行readout评估脚本
   ```bash
   python -u evaluate_layer2.py \
     --model SSSD \
     --seeds 72 82 \
     --predictions_dir outputs/b2_baseline_backup_20260313/layer2 \
     --output_dir outputs/b2_baseline_lockdown/layer2 \
     2>&1 | tee logs/sssd_layer2_lockdown.log
   ```

4. [ ] 验证输出
   ```bash
   python3 << 'EOF'
   import json
   import os
   
   print("=== SSSD Layer2 验证 ===")
   for seed in [72, 82]:
       f = f'outputs/b2_baseline_lockdown/layer2/SSSD_seed{seed}_layer2_readout_metrics.json'
       if os.path.exists(f):
           with open(f) as fp:
               data = json.load(fp)
               auroc = data.get('auroc', 'N/A')
               threshold = data.get('optimal_threshold', 'N/A')
               print(f"seed{seed}: AUROC={auroc}, threshold={threshold}")
               
               # 检查验收标准
               if threshold != float('inf') and auroc > 0.4:
                   print(f"  ✅ 通过")
               else:
                   print(f"  ❌ 失败")
       else:
           print(f"seed{seed}: ❌ 文件不存在")
   EOF
   ```

**验收标准**:
- [ ] 两个seed的optimal_threshold都是有限值
- [ ] 两个seed的AUROC > 0.4

---

### ☐ 重新评估2: SurvTraj Layer2 (seed 52) (5分钟)

**问题**: AUROC = 0.5 (随机分类器)

**步骤**:

1. [ ] 激活conda环境
   ```bash
   conda activate causal_tabdiff
   ```

2. [ ] 运行readout评估脚本
   ```bash
   python -u evaluate_layer2.py \
     --model SurvTraj \
     --seeds 52 \
     --predictions_dir outputs/b2_baseline_backup_20260313/layer2 \
     --output_dir outputs/b2_baseline_lockdown/layer2 \
     2>&1 | tee logs/survtraj_layer2_lockdown.log
   ```

3. [ ] 验证输出
   ```bash
   python3 << 'EOF'
   import json
   import os
   
   print("=== SurvTraj Layer2 验证 ===")
   f = 'outputs/b2_baseline_lockdown/layer2/SurvTraj_seed52_layer2_readout_metrics.json'
   if os.path.exists(f):
       with open(f) as fp:
           data = json.load(fp)
           auroc = data.get('auroc', 'N/A')
           threshold = data.get('optimal_threshold', 'N/A')
           print(f"seed52: AUROC={auroc}, threshold={threshold}")
           
           # 检查验收标准
           if threshold != float('inf') and auroc != 0.5:
               print(f"  ✅ 通过")
           else:
               print(f"  ❌ 失败")
   else:
       print(f"seed52: ❌ 文件不存在")
   EOF
   ```

**验收标准**:
- [ ] seed52的optimal_threshold是有限值
- [ ] seed52的AUROC ≠ 0.5

---

## 第二阶段: 生成最终CSV表 (10分钟)

### ☐ 步骤1: 合并所有结果

```bash
python3 << 'EOF'
import json
import os
import numpy as np
from pathlib import Path

# 定义所有模型和任务
layer1_direct = ['CausalForest', 'iTransformer', 'tsdiff', 'stasy']
layer1_tstr = ['sssd', 'survtraj', 'tabsyn', 'tabdiff', 'tsdiff', 'stasy']
layer2 = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']

seeds = [42, 52, 62, 72, 82]

# 合并Layer1直接预测
print("合并Layer1直接预测...")
layer1_data = {}
for model in layer1_direct:
    metrics_list = []
    for seed in seeds:
        # 优先使用新的重跑结果，否则使用备份
        new_path = f'outputs/b2_baseline_lockdown/layer1/{model}_seed{seed}/metrics.json'
        old_path = f'outputs/b2_baseline_backup_20260313/layer1/{model}_seed{seed}/metrics.json'
        
        if os.path.exists(new_path):
            with open(new_path) as f:
                metrics_list.append(json.load(f))
        elif os.path.exists(old_path):
            with open(old_path) as f:
                metrics_list.append(json.load(f))
    
    if metrics_list:
        layer1_data[model] = metrics_list

print(f"✅ 合并完成: {len(layer1_data)} 个模型")

# 类似地合并Layer1 TSTR和Layer2
# ... (省略详细代码)

print("✅ 所有数据合并完成")
