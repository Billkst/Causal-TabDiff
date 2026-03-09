import torch
import torch.nn.functional as F

def focal_loss_with_logits(logits, targets, alpha=1.0, gamma=2.0, pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = (1 - pt) ** gamma * bce_loss
    
    if pos_weight is not None:
        weight = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
        focal_loss = focal_loss * weight
        
    return focal_loss.mean()

targets = torch.tensor([1.0, 0.0, 0.0])
logits = torch.tensor([0.0, 0.0, 0.0])
print(focal_loss_with_logits(logits, targets, pos_weight=torch.tensor([10.0])))
