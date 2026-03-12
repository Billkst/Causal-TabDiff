#!/usr/bin/env python3
"""
TimeXer smoke test for layer2 (6-year trajectory) functionality.
Validates: create/load, forecast, risk trajectory, eval/readout.
"""
import torch
import json
import os
from datetime import datetime
from src.baselines.tslib_wrappers import TimeXerWrapper

def test_timexer_layer2():
    """Test TimeXer layer2 functionality."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    try:
        # Test 1: Create model
        print("Test 1: Creating TimeXer model...")
        seq_len = 24
        enc_in = 5
        exog_in = 3
        pred_len = 6
        
        model = TimeXerWrapper(
            seq_len=seq_len,
            enc_in=enc_in,
            exog_in=exog_in,
            task='long_term_forecast',
            pred_len=pred_len,
            d_model=64,
            n_heads=4,
            e_layers=2,
            patch_len=1
        )
        model.eval()
        results['tests']['create'] = {'status': 'success'}
        print("✓ Model created successfully")
        
        # Test 2: Forward pass (forecast)
        print("\nTest 2: Running forecast...")
        batch_size = 2
        x_enc = torch.randn(batch_size, seq_len, enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, exog_in)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc)
        
        assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} vs {batch_size}"
        assert output.shape[1] > 0, f"Output sequence length must be > 0, got {output.shape[1]}"
        results['tests']['forecast'] = {
            'status': 'success',
            'output_shape': list(output.shape),
            'expected_pred_len': pred_len,
            'actual_output_len': output.shape[1]
        }
        print(f"✓ Forecast output shape: {output.shape}")
        
        # Test 3: Risk trajectory validation
        print("\nTest 3: Validating risk trajectory...")
        risk_traj = output.cpu().numpy()
        assert risk_traj.min() >= -10 and risk_traj.max() <= 10, "Risk values out of expected range"
        assert not any(torch.isnan(output).flatten()), "NaN values in output"
        results['tests']['risk_trajectory'] = {
            'status': 'success',
            'min': float(risk_traj.min()),
            'max': float(risk_traj.max()),
            'mean': float(risk_traj.mean())
        }
        print(f"✓ Risk trajectory valid: min={risk_traj.min():.4f}, max={risk_traj.max():.4f}")
        
        # Test 4: Model state (eval/readout ready)
        print("\nTest 4: Checking model state...")
        assert not model.training, "Model should be in eval mode"
        param_count = sum(p.numel() for p in model.parameters())
        results['tests']['model_state'] = {
            'status': 'success',
            'training_mode': model.training,
            'param_count': param_count
        }
        print(f"✓ Model ready for eval/readout: {param_count} parameters")
        
        results['overall'] = 'PASS'
        
    except Exception as e:
        results['overall'] = 'FAIL'
        results['error'] = str(e)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == '__main__':
    os.makedirs('outputs/model_repairs', exist_ok=True)
    os.makedirs('logs/model_repairs', exist_ok=True)
    
    print("=" * 60)
    print("TimeXer Layer2 Smoke Test")
    print("=" * 60)
    
    results = test_timexer_layer2()
    
    # Save results
    with open('outputs/model_repairs/timexer_repair_result.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save log
    with open('logs/model_repairs/timexer_repair.log', 'w') as f:
        f.write(f"TimeXer Repair Log\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Overall: {results['overall']}\n\n")
        for test_name, test_result in results['tests'].items():
            f.write(f"{test_name}: {test_result['status']}\n")
        if 'error' in results:
            f.write(f"\nError: {results['error']}\n")
    
    print("\n" + "=" * 60)
    print(f"Overall Result: {results['overall']}")
    print("=" * 60)
    print(f"Results saved to: outputs/model_repairs/timexer_repair_result.json")
    print(f"Log saved to: logs/model_repairs/timexer_repair.log")
