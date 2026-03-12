#!/usr/bin/env python
"""
TabSyn_strict Smoke Test - 验证修复
"""
import sys
import os
import json
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
log_dir = project_root / 'logs' / 'model_repairs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'tabsyn_strict_repair.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test 1: Model creation"""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Creation")
    logger.info("=" * 60)
    
    try:
        from baselines.tabsyn_core.vae.model import Model_VAE
        
        # Test parameters
        num_layers = 2
        d_numerical = 100
        categories = []
        d_token = 64
        n_head = 1
        factor = 32
        bias = True
        
        logger.info(f"Creating Model_VAE with:")
        logger.info(f"  num_layers={num_layers}")
        logger.info(f"  d_numerical={d_numerical}")
        logger.info(f"  categories={categories}")
        logger.info(f"  d_token={d_token}")
        logger.info(f"  n_head={n_head}")
        logger.info(f"  factor={factor}")
        logger.info(f"  bias={bias}")
        
        model = Model_VAE(
            num_layers=num_layers,
            d_numerical=d_numerical,
            categories=categories,
            d_token=d_token,
            n_head=n_head,
            factor=factor,
            bias=bias
        )
        
        logger.info("✓ Model created successfully")
        logger.info(f"Model type: {type(model)}")
        return model, True
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, False

def test_forward_pass(model):
    """Test 2: Forward pass and return value unpacking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Forward Pass & Return Value Unpacking")
    logger.info("=" * 60)
    
    try:
        batch_size = 10
        d_numerical = 100
        
        x_num = torch.randn(batch_size, d_numerical)
        x_cat = None
        
        logger.info(f"Input shapes: x_num={x_num.shape}, x_cat={x_cat}")
        
        # Test forward pass
        output = model(x_num, x_cat)
        logger.info(f"Forward output type: {type(output)}")
        logger.info(f"Forward output length: {len(output) if isinstance(output, tuple) else 'not tuple'}")
        
        # Unpack 4 values
        recon_x_num, recon_x_cat, mu_z, std_z = output
        
        logger.info(f"✓ Successfully unpacked 4 return values:")
        logger.info(f"  recon_x_num shape: {recon_x_num.shape}")
        logger.info(f"  recon_x_cat type: {type(recon_x_cat)}")
        logger.info(f"  mu_z shape: {mu_z.shape}")
        logger.info(f"  std_z shape: {std_z.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_training_step(model):
    """Test 3: Training step"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Training Step")
    logger.info("=" * 60)
    
    try:
        batch_size = 10
        d_numerical = 100
        
        x_num = torch.randn(batch_size, d_numerical)
        x_cat = None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        logger.info("Running 1 training step...")
        
        optimizer.zero_grad()
        recon_x_num, recon_x_cat, mu_z, std_z = model(x_num, x_cat)
        
        # Compute loss
        recon_loss = torch.nn.functional.mse_loss(recon_x_num, x_num)
        kl_loss = -0.5 * torch.sum(1 + std_z - mu_z.pow(2) - std_z.exp()) / batch_size
        loss = recon_loss + 0.001 * kl_loss
        
        logger.info(f"  Recon loss: {recon_loss.item():.6f}")
        logger.info(f"  KL loss: {kl_loss.item():.6f}")
        logger.info(f"  Total loss: {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()
        
        logger.info("✓ Training step completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_wrapper():
    """Test 4: Wrapper integration"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Wrapper Integration")
    logger.info("=" * 60)
    
    try:
        from baselines.tabsyn_landmark_strict import TabSynLandmarkStrictWrapper
        
        seq_len = 10
        feature_dim = 10
        
        logger.info(f"Creating wrapper with seq_len={seq_len}, feature_dim={feature_dim}")
        
        wrapper = TabSynLandmarkStrictWrapper(seq_len, feature_dim)
        logger.info("✓ Wrapper created successfully")
        
        # Test that wrapper can access Model_VAE
        logger.info("Checking wrapper can import Model_VAE...")
        from baselines.tabsyn_core.vae.model import Model_VAE
        logger.info("✓ Model_VAE import successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Wrapper integration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Starting TabSyn_strict Smoke Tests")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    results = {}
    
    # Test 1: Model creation
    model, success = test_model_creation()
    results['model_creation'] = success
    if not success:
        logger.error("Stopping tests - model creation failed")
        return results
    
    # Test 2: Forward pass
    results['forward_pass'] = test_forward_pass(model)
    
    # Test 3: Training step
    results['training_step'] = test_training_step(model)
    
    # Test 4: Wrapper
    results['wrapper_integration'] = test_wrapper()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return results

if __name__ == '__main__':
    results = main()
    
    # Save results
    output_dir = Path(__file__).parent / 'outputs' / 'model_repairs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / 'tabsyn_strict_repair_result.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {result_file}")
    
    sys.exit(0 if all(results.values()) else 1)
