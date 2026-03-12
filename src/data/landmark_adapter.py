"""
Landmark 数据适配层 - 将 landmark batch 转换为 generative baselines 期望的格式
"""
import torch
import numpy as np


def adapt_landmark_to_generative(batch, device):
    """
    将 landmark dataloader 的 batch 转换为 generative baselines 期望的格式
    
    Landmark batch 格式:
        - x: (B, seq_len, features)
        - y_2year: (B, 1)
        - trajectory_target: (B, 6)
        - trajectory_valid_mask: (B, 6)
        - landmark: (B, 1)
        - history_length: (B, 1)
        - pid: list
    
    Generative baselines 期望格式:
        - x: (B, T, D)
        - alpha_target: (B, 1) - 用 landmark 替代
        - y: (B, 1) - 用 y_2year 替代
        - x_cat_raw: (B, T, num_cats) - 如果有类别特征
    """
    adapted = {}
    
    adapted['x'] = batch['x'].to(device)
    adapted['y'] = batch['y_2year'].to(device)
    adapted['alpha_target'] = batch['landmark'].float().to(device)
    
    B, T, D = adapted['x'].shape
    adapted['x_cat_raw'] = torch.zeros(B, T, 0, device=device)
    
    return adapted
