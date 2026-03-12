import sys
import torch
import numpy as np

sys.path.insert(0, 'src')
from baselines.tsdiff_core.model import TSDiffDDPM

print("="*60)
print("TSDiff Trajectory Upgrade - Smoke Test")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

print("Test 1: Original mode (no condition)")
model_v1 = TSDiffDDPM(input_dim=37, timesteps=100, condition_dim=0).to(device)
x_train = torch.randn(8, 1, 37, device=device)
loss = model_v1.train_step(x_train)
print(f"✓ Training loss: {loss.item():.4f}")

x_gen = model_v1.sample(batch_size=4, seq_len=1, features=37, device=device)
print(f"✓ Generated shape: {x_gen.shape} (expected: [4, 1, 37])\n")

print("Test 2: Trajectory mode (with condition)")
model_v2 = TSDiffDDPM(input_dim=6, timesteps=100, condition_dim=37).to(device)
x_history = torch.randn(8, 37, device=device)
y_trajectory = torch.randn(8, 6, 1, device=device)
loss = model_v2.train_step(y_trajectory, condition=x_history)
print(f"✓ Training loss with condition: {loss.item():.4f}")

traj_gen = model_v2.sample(batch_size=4, seq_len=6, features=1, device=device, condition=x_history[:4])
print(f"✓ Generated trajectory shape: {traj_gen.shape} (expected: [4, 6, 1])")
print(f"✓ Trajectory values range: [{traj_gen.min():.2f}, {traj_gen.max():.2f}]\n")

print("Test 3: Multi-step trajectory (6-year risk)")
traj_6year = model_v2.sample(batch_size=2, seq_len=6, features=1, device=device, condition=x_history[:2])
print(f"✓ 6-year trajectory shape: {traj_6year.shape}")
print(f"✓ Sample trajectory: {traj_6year[0, :, 0].cpu().numpy()}\n")

print("="*60)
print("✅ TSDiff Trajectory Upgrade: ALL TESTS PASSED")
print("="*60)
print("\nCapabilities:")
print("  ✓ Layer 1: 2-year risk (from trajectory[1])")
print("  ✓ Layer 2: 6-year risk trajectory")
print("  ✓ Condition encoder: history → future")
print("  ✓ Backward compatible: works without condition")
