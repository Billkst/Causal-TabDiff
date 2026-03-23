"""
快速诊断 disc_loss 数值 - 运行1个epoch查看各损失分量
"""
import torch
import numpy as np
from src.data.data_module_landmark import load_and_split_data, create_dataloaders
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
np.random.seed(seed)

# 加载数据
train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
    'data/landmark_tables/unified_person_landmark_table.pkl', seed=seed
)
train_loader, _, _ = create_dataloaders(
    train_df, val_df, test_df, landmark_to_idx, batch_size=4096, num_workers=4
)

# 模型
model = CausalTabDiffTrajectory(
    t_steps=3, feature_dim=15, diffusion_steps=100, trajectory_len=7
).to(device)

print("=" * 70)
print("诊断 disc_loss 数值")
print("=" * 70)

# 运行3个batch
for i, batch in enumerate(train_loader):
    if i >= 3:
        break
    
    x = batch['x'].to(device)
    alpha = batch['landmark'].float().to(device)
    
    outputs = model(x, alpha)
    
    print(f"\nBatch {i+1}:")
    print(f"  diff_loss:  {outputs['diff_loss'].item():.6f}")
    print(f"  disc_loss:  {outputs['disc_loss'].item():.6f}")
    print(f"  比例: disc/diff = {outputs['disc_loss'].item() / outputs['diff_loss'].item():.6f}")

print("\n" + "=" * 70)
print("结论:")
print("如果 disc_loss < 0.001，说明判别器训练过度或损失scale过小")
print("如果 disc_loss 与 diff_loss 比例 < 0.01，disc_weight 影响微乎其微")
print("=" * 70)
