import sys
import torch
import numpy as np

print("=== Smoke Test: Landmark-Based Data Pipeline ===\n")

try:
    from src.data.data_module_landmark import get_landmark_dataloader
    print("✓ Import successful")
    
    print("\n1. Testing data loading...")
    train_loader = get_landmark_dataloader('data', split='train', batch_size=4, seed=42, debug_mode=True)
    print(f"✓ Train loader created")
    
    print("\n2. Testing batch structure...")
    batch = next(iter(train_loader))
    
    print(f"  x shape: {batch['x'].shape}")
    print(f"  y_2year shape: {batch['y_2year'].shape}")
    print(f"  risk_trajectory shape: {batch['risk_trajectory'].shape}")
    print(f"  alpha_target shape: {batch['alpha_target'].shape}")
    print(f"  landmark shape: {batch['landmark'].shape}")
    print(f"  history_length shape: {batch['history_length'].shape}")
    
    assert batch['x'].dim() == 3, "x should be 3D (batch, time, features)"
    assert batch['y_2year'].dim() == 2, "y_2year should be 2D"
    assert batch['risk_trajectory'].dim() == 2, "risk_trajectory should be 2D"
    print("✓ Shapes correct")
    
    print("\n3. Testing value ranges...")
    assert torch.all((batch['y_2year'] == 0) | (batch['y_2year'] == 1)), "y_2year must be binary"
    assert torch.all((batch['risk_trajectory'] >= 0) & (batch['risk_trajectory'] <= 1)), "trajectory must be in [0,1]"
    assert torch.all((batch['landmark'] >= 0) & (batch['landmark'] <= 2)), "landmark must be in {0,1,2}"
    print("✓ Value ranges valid")
    
    print("\n4. Testing splits...")
    val_loader = get_landmark_dataloader('data', split='val', batch_size=4, seed=42, debug_mode=True)
    test_loader = get_landmark_dataloader('data', split='test', batch_size=4, seed=42, debug_mode=True)
    print(f"✓ All splits loaded")
    
    print("\n5. Testing model interface...")
    from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
    
    t_steps = batch['x'].shape[1]
    feature_dim = batch['x'].shape[2]
    trajectory_len = batch['risk_trajectory'].shape[1]
    
    model = CausalTabDiffTrajectory(t_steps, feature_dim, diffusion_steps=10, trajectory_len=trajectory_len)
    
    x = batch['x']
    alpha = batch['alpha_target']
    
    outputs = model(x, alpha)
    
    print(f"  diff_loss: {outputs['diff_loss'].item():.4f}")
    print(f"  disc_loss: {outputs['disc_loss'].item():.4f}")
    print(f"  trajectory shape: {outputs['trajectory'].shape}")
    print(f"  risk_2year shape: {outputs['risk_2year'].shape}")
    
    assert outputs['trajectory'].shape == batch['risk_trajectory'].shape
    assert outputs['risk_2year'].shape == batch['y_2year'].shape
    print("✓ Model outputs correct")
    
    print("\n=== ALL TESTS PASSED ===")
    print("\nNext steps:")
    print("1. Run full training: python run_experiment_landmark.py --debug_mode")
    print("2. Run baselines: python run_baselines_landmark.py --debug_mode")
    print("3. Scale to full data (remove --debug_mode)")
    print("4. Run 5-seed experiments")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
