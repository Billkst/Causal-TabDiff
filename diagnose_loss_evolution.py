"""
追踪训练过程中各损失分量的演化
"""
import torch
import torch.nn.functional as F
import numpy as np
from src.data.data_module_landmark import load_and_split_data, create_dataloaders
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
np.random.seed(seed)

train_df, val_df, test_df, landmark_to_idx = load_and_split_data(
    'data/landmark_tables/unified_person_landmark_table.pkl', seed=seed
)
train_loader, _, _ = create_dataloaders(
    train_df, val_df, test_df, landmark_to_idx, batch_size=4096, num_workers=4
)

model = CausalTabDiffTrajectory(
    t_steps=3, feature_dim=15, diffusion_steps=100, trajectory_len=7
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

print("Epoch | DiffLoss | DiscLoss | TrajLoss | RiskLoss | TotalLoss")
print("-" * 70)

for epoch in range(5):
    model.train()
    diff_losses, disc_losses, traj_losses, risk_losses = [], [], [], []
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只跑10个batch加速
            break
        
        x = batch['x'].to(device)
        alpha = batch['landmark'].float().to(device)
        y_2year = batch['y_2year'].to(device)
        traj_target = batch['trajectory_target'].to(device)
        traj_mask = batch['trajectory_valid_mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(x, alpha)
        
        diff_loss = outputs['diff_loss']
        disc_loss = outputs['disc_loss']
