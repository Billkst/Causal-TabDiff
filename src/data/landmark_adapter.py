"""
Landmark 数据适配层 - 将 landmark batch 转换为 generative baselines 期望的格式
"""
import torch
import numpy as np


def adapt_landmark_to_generative(batch, device):
    adapted = {}
    
    x = batch['x'].to(device)
    B, T, D = x.shape
    
    adapted['x'] = x
    adapted['y'] = batch['y_2year'].to(device)
    adapted['alpha_target'] = batch['landmark'].float().to(device)
    adapted['x_cat_raw'] = torch.zeros(B, T, 0, dtype=torch.long, device=device)
    
    return adapted
