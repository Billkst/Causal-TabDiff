"""
TSLib model wrappers for iTransformer and TimeXer.
Supports both Layer 1 (2-year risk) and Layer 2 (6-year trajectory).
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/TSLib'))

from models.iTransformer import Model as iTransformerModel
from models.TimeXer import Model as TimeXerModel
import argparse


class iTransformerWrapper(nn.Module):
    """iTransformer wrapper for landmark-conditioned risk prediction."""
    
    def __init__(self, seq_len, enc_in, task='classification', num_class=2, pred_len=6, d_model=128, n_heads=4, e_layers=2):
        super().__init__()
        self.task = task
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.config = argparse.Namespace(
            task_name=task,
            seq_len=seq_len,
            pred_len=pred_len if task == 'long_term_forecast' else 0,
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=1 if task == 'long_term_forecast' else enc_in,
            num_class=num_class,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_model * 4,
            dropout=0.1,
            embed='timeF',
            freq='h',
            activation='gelu',
            factor=3,
            output_attention=False
        )
        self.model = iTransformerModel(self.config)
    
    def forward(self, x, padding_mask=None):
        if self.task == 'classification':
            return self.model.classification(x, padding_mask)
        else:
            output = self.model.forecast(x, None, None, None)
            if len(output.shape) == 3:
                return output.squeeze(-1)
            return output


class TimeXerWrapper(nn.Module):
    """TimeXer wrapper with exogenous variable support."""
    
    def __init__(self, seq_len, enc_in, exog_in, task='classification', num_class=2, pred_len=6, 
                 d_model=128, n_heads=4, e_layers=2, patch_len=1):
        super().__init__()
        self.task = task
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.exog_in = exog_in
        
        self.config = argparse.Namespace(
            task_name=task,
            seq_len=seq_len,
            pred_len=pred_len if task == 'long_term_forecast' else 0,
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=1 if task == 'long_term_forecast' else enc_in,
            num_class=num_class,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_model * 4,
            dropout=0.1,
            patch_len=patch_len,
            use_norm=True,
            activation='gelu',
            output_attention=False,
            ex_dim=exog_in,
            features='M',
            embed='timeF',
            freq='h',
            factor=3
        )
        self.model = TimeXerModel(self.config)
    
    def forward(self, x_enc, x_mark_enc=None):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], self.exog_in, device=x_enc.device)
        
        if self.task == 'classification':
            return self.model.classification(x_enc, x_mark_enc)
        else:
            output = self.model.forecast(x_enc, x_mark_enc, None, None)
            if len(output.shape) == 3:
                return output.squeeze(-1)
            return output
