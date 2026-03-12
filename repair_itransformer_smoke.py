#!/usr/bin/env python
"""Smoke test for iTransformer and TimeXer import fix."""
import sys
import os
import json
import torch
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

log_file = 'logs/model_repairs/itransformer_repair.log'
result_file = 'outputs/model_repairs/itransformer_repair_result.json'

def log_msg(msg):
    """Log to both stdout and file."""
    print(msg, flush=True)
    with open(log_file, 'a', buffering=1) as f:
        f.write(msg + '\n')
        f.flush()

def test_itransformer():
    """Test iTransformer import and basic operations."""
    log_msg('\n=== Testing iTransformer ===')
    try:
        from src.baselines.tslib_wrappers import iTransformerWrapper
        log_msg('✓ iTransformer import successful')
        
        # Layer 1: Create and forward
        model = iTransformerWrapper(seq_len=24, enc_in=5, task='classification', num_class=2)
        x = torch.randn(2, 24, 5)
        out = model(x)
        log_msg(f'✓ Layer 1 (classification): input {x.shape} -> output {out.shape}')
        
        # Layer 2: Forecast
        model_f = iTransformerWrapper(seq_len=24, enc_in=5, task='long_term_forecast', pred_len=6)
        out_f = model_f(x)
        log_msg(f'✓ Layer 2 (forecast): input {x.shape} -> output {out_f.shape}')
        
        return True, 'iTransformer passed all tests'
    except Exception as e:
        log_msg(f'✗ iTransformer failed: {str(e)}')
        log_msg(traceback.format_exc())
        return False, str(e)

def test_timexer():
    """Test TimeXer import and basic operations."""
    log_msg('\n=== Testing TimeXer ===')
    try:
        from src.baselines.tslib_wrappers import TimeXerWrapper
        log_msg('✓ TimeXer import successful')
        
        # Layer 1: Create and forward
        model = TimeXerWrapper(seq_len=24, enc_in=5, exog_in=3, task='classification', num_class=2)
        x = torch.randn(2, 24, 5)
        out = model(x)
        log_msg(f'✓ Layer 1 (classification): input {x.shape} -> output {out.shape}')
        
        # Layer 2: Forecast
        model_f = TimeXerWrapper(seq_len=24, enc_in=5, exog_in=3, task='long_term_forecast', pred_len=6)
        out_f = model_f(x)
        log_msg(f'✓ Layer 2 (forecast): input {x.shape} -> output {out_f.shape}')
        
        return True, 'TimeXer passed all tests'
    except Exception as e:
        log_msg(f'✗ TimeXer failed: {str(e)}')
        log_msg(traceback.format_exc())
        return False, str(e)

if __name__ == '__main__':
    log_msg(f'[{datetime.now().isoformat()}] Starting iTransformer/TimeXer smoke tests')
    
    it_pass, it_msg = test_itransformer()
    tx_pass, tx_msg = test_timexer()
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'repair_method': 'importlib dynamic loading',
        'iTransformer': {'passed': it_pass, 'message': it_msg},
        'TimeXer': {'passed': tx_pass, 'message': tx_msg},
        'overall_success': it_pass and tx_pass
    }
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    log_msg(f'\n[{datetime.now().isoformat()}] Results saved to {result_file}')
    sys.exit(0 if result['overall_success'] else 1)
