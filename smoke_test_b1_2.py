import sys
import torch
import torch.nn.functional as F
from src.data.data_module_landmark import get_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import numpy as np

print("=== B1-2 Smoke Test: End-to-End Pipeline ===\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'

print("1. Loading data...")
train_loader = get_dataloader(table_path, 'train', batch_size=8, seed=42, debug_n_persons=100)
val_loader = get_dataloader(table_path, 'val', batch_size=8, seed=42, debug_n_persons=100)

batch = next(iter(train_loader))
print(f"✓ Data loaded")
print(f"  x: {batch['x'].shape}")
print(f"  y_2year: {batch['y_2year'].shape}")
print(f"  trajectory_target: {batch['trajectory_target'].shape}")
print(f"  trajectory_valid_mask: {batch['trajectory_valid_mask'].shape}\n")

print("2. Initializing model...")
t_steps = batch['x'].shape[1]
feature_dim = batch['x'].shape[2]
trajectory_len = batch['trajectory_target'].shape[1]

model = CausalTabDiffTrajectory(t_steps, feature_dim, diffusion_steps=10, trajectory_len=trajectory_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"✓ Model initialized (t_steps={t_steps}, feature_dim={feature_dim}, trajectory_len={trajectory_len})\n")

print("3. Testing forward pass...")
x = batch['x'].to(device)
y_2year = batch['y_2year'].to(device)
traj_target = batch['trajectory_target'].to(device)
traj_mask = batch['trajectory_valid_mask'].to(device)

alpha_target = torch.rand(x.shape[0], 1).to(device) * 0.8 + 0.1

outputs = model(x, alpha_target)
print(f"✓ Forward pass successful")
print(f"  diff_loss: {outputs['diff_loss'].item():.4f}")
print(f"  disc_loss: {outputs['disc_loss'].item():.4f}")
print(f"  trajectory: {outputs['trajectory'].shape}")
print(f"  risk_2year: {outputs['risk_2year'].shape}\n")

print("4. Testing loss computation...")
loss_diff = outputs['diff_loss']
loss_disc = outputs['disc_loss']

traj_pred = outputs['trajectory']
loss_traj = model.compute_trajectory_loss(traj_pred, traj_target, traj_mask)

risk_pred = outputs['risk_2year']
loss_2year = F.binary_cross_entropy(risk_pred, y_2year)

total_loss = loss_diff + 0.5 * loss_disc + loss_traj + loss_2year
print(f"✓ Loss computed")
print(f"  loss_traj: {loss_traj.item():.4f}")
print(f"  loss_2year: {loss_2year.item():.4f}")
print(f"  total_loss: {total_loss.item():.4f}\n")

print("5. Testing backward pass...")
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
print(f"✓ Backward pass successful\n")

print("6. Running mini training (2 epochs)...")
for epoch in range(2):
    model.train()
    total_loss_epoch = 0
    
    for batch in train_loader:
        x = batch['x'].to(device)
        y_2year = batch['y_2year'].to(device)
        traj_target = batch['trajectory_target'].to(device)
        traj_mask = batch['trajectory_valid_mask'].to(device)
        
        alpha_target = torch.rand(x.shape[0], 1).to(device) * 0.8 + 0.1
        
        optimizer.zero_grad()
        outputs = model(x, alpha_target)
        
        loss = (outputs['diff_loss'] + 0.5 * outputs['disc_loss'] + 
                model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask) +
                F.binary_cross_entropy(outputs['risk_2year'], y_2year))
        
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
    
    print(f"  Epoch {epoch+1}/2 | Loss: {total_loss_epoch/len(train_loader):.4f}")

print(f"✓ Training successful\n")

print("7. Running validation and computing metrics...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        x = batch['x'].to(device)
        y_2year = batch['y_2year'].to(device)
        alpha_target = torch.rand(x.shape[0], 1).to(device) * 0.8 + 0.1
        
        outputs = model(x, alpha_target)
        risk_pred = outputs['risk_2year'].cpu().numpy()
        y_true = y_2year.cpu().numpy()
        
        all_preds.append(risk_pred)
        all_labels.append(y_true)

all_preds = np.concatenate(all_preds, axis=0).flatten()
all_labels = np.concatenate(all_labels, axis=0).flatten().astype(int)

if len(np.unique(all_labels)) > 1:
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
else:
    auroc = auprc = float('nan')

all_preds_binary = (all_preds > 0.5).astype(int)
f1 = f1_score(all_labels, all_preds_binary, zero_division=0)
cm = confusion_matrix(all_labels, all_preds_binary)

print(f"✓ Validation complete\n")
print(f"=== Minimal Metrics (Smoke Test Only) ===")
print(f"AUROC: {auroc:.4f}" if not np.isnan(auroc) else "AUROC: N/A (single class)")
print(f"AUPRC: {auprc:.4f}" if not np.isnan(auprc) else "AUPRC: N/A (single class)")
print(f"F1:    {f1:.4f}")
print(f"Confusion Matrix:\n{cm}\n")

print("=== B1-2 Smoke Test PASSED ===")
print("\nNote: These metrics are from debug mode (100 persons, 2 epochs).")
print("NOT representative of final performance.")
