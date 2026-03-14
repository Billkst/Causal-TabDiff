#!/usr/bin/env python
"""Smoke test for TabDiff_strict repair"""
import torch
import numpy as np
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/baselines'))

def test_tabdiff_strict():
    """Minimal smoke test"""
    from tabdiff_landmark_strict import TabDiffLandmarkStrictWrapper
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create wrapper
    wrapper = TabDiffLandmarkStrictWrapper(seq_len=10, feature_dim=5)
    print("✓ Wrapper created")
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 10, 5)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    train_loader = [{'x': x, 'y_2year': y}]
    print("✓ Data created")
    
    # Test fit (1 epoch)
    try:
        wrapper.fit(train_loader, epochs=1, device=device)
        print("✓ Fit success (1 epoch)")
    except Exception as e:
        print(f"✗ Fit failed: {e}")
        return False
    
    # Test sample
    try:
        X_syn, Y_syn = wrapper.sample(10, device)
        assert X_syn.shape == (10, 10, 5), f"Wrong X shape: {X_syn.shape}"
        assert Y_syn.shape == (10, 1), f"Wrong Y shape: {Y_syn.shape}"
        print("✓ Sample success (10 samples)")
    except Exception as e:
        print(f"✗ Sample failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    result = test_tabdiff_strict()
    
    # Save result
    os.makedirs('outputs/model_repairs', exist_ok=True)
    os.makedirs('logs/model_repairs', exist_ok=True)
    
    output = {
        'model': 'TabDiff_strict',
        'status': 'PASS' if result else 'FAIL',
        'timestamp': str(np.datetime64('now'))
    }
    
    with open('outputs/model_repairs/tabdiff_strict_repair_result.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResult: {output['status']}")
    sys.exit(0 if result else 1)
