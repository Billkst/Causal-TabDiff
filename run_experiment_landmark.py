import argparse
import torch
import logging
from src.data.data_module_landmark import get_landmark_dataloader
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
import os

os.makedirs('logs/training', exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training/run_landmark.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.debug_mode:
        args.epochs = 2
        args.batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    train_loader = get_landmark_dataloader(args.data_dir, 'train', args.batch_size, args.seed, args.debug_mode)
    val_loader = get_landmark_dataloader(args.data_dir, 'val', args.batch_size, args.seed, args.debug_mode)
    
    sample = next(iter(train_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['risk_trajectory'].shape[1]

    model = CausalTabDiffTrajectory(t_steps, feature_dim, 100 if not args.debug_mode else 10, trajectory_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Training started")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = batch['alpha_target'].to(device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['risk_trajectory'].to(device)
            
            optimizer.zero_grad()
            outputs = model(x, alpha)
            
            loss_diff = outputs['diff_loss']
            loss_disc = outputs['disc_loss']
            loss_traj = F.binary_cross_entropy(outputs['trajectory'], traj_target)
            loss_2year = F.binary_cross_entropy(outputs['risk_2year'], y_2year)
            
            loss = loss_diff + 0.5 * loss_disc + loss_traj + loss_2year
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")
    
    logger.info("Training complete")

if __name__ == '__main__':
    main()
