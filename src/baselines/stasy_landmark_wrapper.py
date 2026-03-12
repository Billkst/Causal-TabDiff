"""
STaSy Landmark Wrapper - 最小化重写版本
"""
import torch
import numpy as np
import sys
import os


class STaSyLandmarkWrapper:
    def __init__(self, seq_len, feature_dim):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.fitted = False
        self.model = None
        self.ema = None
        self.sde = None
    
    def fit(self, train_loader, epochs, device):
        stasy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'stasy_core'))
        if stasy_path not in sys.path:
            sys.path.insert(0, stasy_path)
        
        import ml_collections
        from models import utils as stasy_mutils
        import sde_lib
        import losses as stasy_losses
        from models.ema import ExponentialMovingAverage
        
        config = ml_collections.ConfigDict()
        config.device = device
        config.training = ml_collections.ConfigDict()
        config.training.sde = 'vesde'
        config.training.continuous = True
        config.training.reduce_mean = True
        
        config.model = ml_collections.ConfigDict()
        config.model.name = 'ncsnpp_tabular'
        config.model.ema_rate = 0.9999
        config.model.nf = 64
        config.model.hidden_dims = (256, 512, 512, 256)
        config.model.sigma_min = 0.01
        config.model.sigma_max = 10.
        config.model.activation = "swish"
        config.model.num_scales = 1000
        
        config.optim = ml_collections.ConfigDict()
        config.optim.optimizer = 'Adam'
        config.optim.lr = 2e-3
        config.optim.weight_decay = 0
        
        config.data = ml_collections.ConfigDict()
        config.data.image_size = self.seq_len * self.feature_dim + 1
        
        self.model = stasy_mutils.create_model(config).to(device)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate)
        optimizer = stasy_losses.get_optimizer(config, self.model.parameters())
        self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=50)
        
        optimize_fn = stasy_losses.optimization_manager(config)
        train_step_fn = stasy_losses.get_step_fn(self.sde, train=True, optimize_fn=optimize_fn,
                                                 reduce_mean=True, continuous=True, likelihood_weighting=False)
        
        state = dict(optimizer=optimizer, model=self.model, ema=self.ema, step=0)
        
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch['x'].cpu().numpy()
                y = batch['y_2year'].cpu().numpy()
                
                x_flat = x.reshape(x.shape[0], -1)
                xy = np.concatenate([x_flat, y], axis=1)
                xy_tensor = torch.tensor(xy, dtype=torch.float32, device=device)
                
                train_step_fn(state, xy_tensor)
        
        self.fitted = True
    
    def sample(self, n_samples, device):
        if not self.fitted:
            return torch.randn(n_samples, self.seq_len, self.feature_dim, device=device), torch.randn(n_samples, 1, device=device)
        
        import sampling
        
        sampling_shape = (n_samples, self.seq_len * self.feature_dim + 1)
        sampling_fn = sampling.get_sampling_fn(None, self.sde, sampling_shape, lambda x: x, eps=1e-5)
        
        self.ema.copy_to(self.model.parameters())
        samples, _ = sampling_fn(self.model, sampling_shape=sampling_shape)
        self.ema.restore(self.model.parameters())
        
        samples = torch.clamp(samples, -3.0, 3.0)
        
        X_syn = samples[:, :-1].reshape(n_samples, self.seq_len, self.feature_dim)
        Y_syn = samples[:, -1:]
        
        return X_syn, Y_syn
