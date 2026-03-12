"""
所有模型的快速 smoke test
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from baselines.tsdiff_core.model import TSDiffDDPM
from baselines.tslib_wrappers import iTransformerWrapper, TimeXerWrapper

def test_tsdiff_trajectory():
    print("\n=== Test 1: TSDiff Trajectory Mode ===")
    model = TSDiffDDPM(input_dim=37, timesteps=100, condition_dim=37)
    
    # Test training
    x = torch.randn(8, 6, 1)
    condition = torch.randn(8, 37)
    loss = model.train_step(x, condition)
    print(f"✓ Training loss: {loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(4, seq_len=6, features=1, device='cpu', condition=condition[:4])
    print(f"✓ Sample shape: {samples.shape} (expected: [4, 6, 1])")
    assert samples.shape == (4, 6, 1)
    print("✓ TSDiff trajectory mode passed")

def test_itransformer():
    print("\n=== Test 2: iTransformer ===")
    model = iTransformerWrapper(seq_len=3, enc_in=37, task='classification', num_class=2)
    
    x = torch.randn(8, 3, 37)
    output = model(x)
    print(f"✓ Output shape: {output.shape} (expected: [8, 2])")
    assert output.shape == (8, 2)
    print("✓ iTransformer passed")

def test_timexer():
    print("\n=== Test 3: TimeXer ===")
    model = TimeXerWrapper(seq_len=3, enc_in=37, exog_in=4, task='classification', num_class=2)
    
    x_enc = torch.randn(8, 3, 37)
    x_mark = torch.randn(8, 3, 4)
    output = model(x_enc, x_mark)
    print(f"✓ Output shape: {output.shape} (expected: [8, 2])")
    assert output.shape == (8, 2)
    print("✓ TimeXer passed")

if __name__ == '__main__':
    print("=== 开始所有模型 Smoke Test ===")
    test_tsdiff_trajectory()
    test_itransformer()
    test_timexer()
    print("\n=== 所有测试通过 ✓ ===")
