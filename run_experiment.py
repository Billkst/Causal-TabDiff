import argparse
import torch
import logging
from tqdm import tqdm
from src.data.data_module import get_dataloader
from src.models.causal_tabdiff import CausalTabDiff

import os

# Ensure log directory exists
os.makedirs('logs/training', exist_ok=True)

# Configure logging mapping
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training/run.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Causal-TabDiff Experiment")
    parser.add_argument('--debug_mode', action='store_true', help='Force truncation and CPU execution for Phase 3 smoke test')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to datasets')
    args = parser.parse_args()

    # Apply debug constraints
    if args.debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG MODE ENABLED. Using constrained dataset and forcing CPU.")
        device = torch.device('cpu')
        epochs = 2
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = 100

    logger.info(f"Using device: {device}")

    # Load Data
    logger.debug("Loading dataset...")
    dataloader = get_dataloader(data_dir=args.data_dir, batch_size=512 if not args.debug_mode else 4, debug_mode=args.debug_mode)
    
    # Infer dimensions from first batch
    sample_batch = next(iter(dataloader))
    t_steps = sample_batch['x'].shape[1]
    feature_dim = sample_batch['x'].shape[2]

    logger.debug(f"Data shapes inferred - T steps: {t_steps}, Feature Dim: {feature_dim}")

    # Initialize Model (Fallback Protocol active)
    model = CausalTabDiff(
        t_steps=t_steps, 
        feature_dim=feature_dim, 
        diffusion_steps=100 if not args.debug_mode else 10 # fast debug
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Starting Training Loop...")
    model.train()
    for epoch in range(epochs):
        epoch_diff_loss = 0
        epoch_disc_loss = 0
        
        # In debug mode, we severely limit training size to avoid local overload
        num_batches = len(dataloader) if not args.debug_mode else min(2, len(dataloader))
        
        for i, batch in enumerate(dataloader):
            if args.debug_mode and i >= 2:
                 break
                 
            x = batch['x'].to(device)
            alpha_tgt = batch['alpha_target'].to(device)
            
            optimizer.zero_grad()
            diff_loss, disc_loss = model(x, alpha_tgt)
            loss = diff_loss + 0.5 * disc_loss
            loss.backward()
            optimizer.step()
            
            epoch_diff_loss += diff_loss.item()
            epoch_disc_loss += disc_loss.item()
            
        logger.debug(f"Epoch {epoch+1}/{epochs} | Diff Loss: {epoch_diff_loss/num_batches:.4f} | Disc Loss: {epoch_disc_loss/num_batches:.4f}")

    logger.info("Training Complete.")
    
    # Run a dummy sampling validation
    logger.info("Testing Gradient Guided Sampling...")
    model.eval()
    with torch.no_grad():
         # Re-enable grad just for the target tensor during sampling if needed by guidance
         sample_alpha = torch.tensor([[0.8]], device=device)
         sampled = model.sample_with_guidance(batch_size=1, alpha_target=sample_alpha, guidance_scale=2.0)
         logger.debug(f"Sampled trajectory shape: {sampled.shape}")
         logger.info("Sampling Successful.")

if __name__ == '__main__':
    main()
