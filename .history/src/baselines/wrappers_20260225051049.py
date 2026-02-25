import torch
import numpy as np
from .base import BaselineWrapper
import logging

logger = logging.getLogger(__name__)

class CausalForestWrapper(BaselineWrapper):
    """
    Wrapper for EconML Causal Forest (Traditional ML baseline).
    """
    def __init__(self, t_steps, feature_dim, **kwargs):
        super().__init__(t_steps, feature_dim, **kwargs)
        from econml.dml import CausalForestDML
        from sklearn.linear_model import Ridge
        # EconML model configured for Multi-treatment Multi-Variate outputs
        self.model = CausalForestDML(
            model_y=Ridge(), 
            model_t=Ridge(),
            discrete_treatment=False,
            n_estimators=100,
            random_state=42
        ) 
        self.fitted = False

    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training Causal Forest {'(Debug)' if debug_mode else ''}...")
        logger.info("[CausalForest] Epoch 1/1 - collecting training batches...")
        
        # Collect all batches into a single dataset
        X_list, T_list = [], []
        
        max_batches = min(2, len(dataloader)) if debug_mode else len(dataloader)
        for i, batch in enumerate(dataloader):
            if debug_mode and i >= 2: break
            
            x = batch['x'].cpu().numpy() # [Batch, T, D]
            alpha_tgt = batch['alpha_target'].cpu().numpy() # [Batch, 1]
            
            b_size = x.shape[0]
            x_flat = x.reshape(b_size, -1)
            y_flat = batch['y'].cpu().numpy().reshape(b_size, -1) # [Batch, 1]
            xy_flat = np.concatenate([x_flat, y_flat], axis=1) # [Batch, D + 1]
            
            X_list.append(xy_flat)
            T_list.append(alpha_tgt.reshape(-1))
            batch_idx = i + 1
            if batch_idx % 20 == 0 or batch_idx == max_batches:
                logger.info(f"[CausalForest] Epoch 1/1 - collected batch {batch_idx}/{max_batches}")
            
        X_all = np.concatenate(X_list, axis=0)
        T_all = np.concatenate(T_list, axis=0)
        
        # We treat the concatenated [X, Y] features as the multidimensional outcome we want to predict or measure effect on. 
        # CausalForestDML natively supports multi-dimensional Y.
        XY_all = X_all.copy() 
        # Use a dummy W of ones since we are generating X, Y given T marginally here.
        W_all = np.ones((XY_all.shape[0], 1))
        logger.info(f"[CausalForest] Epoch 1/1 - fitting model with N={XY_all.shape[0]}, D={XY_all.shape[1]}")
        
        self.model.fit(Y=XY_all, T=T_all, X=W_all, cache_values=True)
        self.XY_mean = np.mean(XY_all, axis=0, keepdims=True)
        logger.info("[CausalForest] Epoch 1/1 - fit completed")
        self.fitted = True

    def sample(self, batch_size, alpha_target, device):
        if not self.fitted:
             return torch.randn(batch_size, self.t_steps, self.feature_dim, device=device)
             
        # Predict counterfactual Y given a new treatment T (alpha_target)
        # We estimate the marginal effect and add it to the base marginal expectation
        T_new = alpha_target.cpu().numpy().reshape(-1)
        W_new = np.ones((batch_size, 1))
        
        # CATE: E[Y(T=1) - Y(T=0) | X]
        # For continuous, it's the gradient. 
        cate = self.model.effect(W_new, T0=np.zeros_like(T_new), T1=T_new)
        
        # Reconstruct base (intercept) + effect
        # Approximate the factual base XY_0 = XY - cate*T_orig (using training mean as a naive base)
        XY_base = self.XY_mean # using historical mean cached from fit
        XY_cf = XY_base + cate
        
        X_cf = XY_cf[:, :-1]
        Y_cf = XY_cf[:, -1:]
        # Reshape X back to [Batch, T, D]
        X_cf_reshaped = X_cf.reshape(batch_size, self.t_steps, self.feature_dim) # This has analog bits!
        
        # We must decode this into semantic categorical integers (5D)
        import os, json
        meta_path = 'src/data/dataset_metadata.json'
        if not os.path.exists(meta_path):
             meta_path = 'dataset_metadata.json'
        with open(meta_path, 'r') as f:
             meta = json.load(f)
             
        D_discrete = len(meta['columns'])
        X_cf_semantic = np.zeros((batch_size, self.t_steps, D_discrete))
        
        for t in range(self.t_steps):
            analog_offset = 0
            for i_col, col_meta in enumerate(meta['columns']):
                dim = col_meta['dim']
                feat = X_cf_reshaped[:, t, analog_offset : analog_offset + dim]
                if col_meta['type'] == 'continuous':
                    X_cf_semantic[:, t, i_col] = feat.squeeze(-1)
                else:
                    # Convert Analog Bits (-1 or 1) back to integer.
                    # Since it's a regression output, we threshold at 0 and treat >0 as bit=1.
                    bits = (feat > 0).astype(int)
                    idx_val = np.zeros((batch_size,), dtype=int)
                    for b in range(dim):
                        idx_val = (idx_val << 1) | bits[:, b]
                    X_cf_semantic[:, t, i_col] = idx_val
                analog_offset += dim

        return torch.tensor(X_cf_semantic[:, -1, :], dtype=torch.float32, device=device), \
               torch.tensor(Y_cf, dtype=torch.float32, device=device)

class STaSyWrapper(BaselineWrapper):
    def __init__(self, t_steps, feature_dim, **kwargs):
        super().__init__(t_steps, feature_dim, **kwargs)
        self.fitted = False
        import sys, os
        self.stasy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'stasy_core'))
        if self.stasy_path not in sys.path:
            sys.path.insert(0, self.stasy_path)

    def _get_config(self):
        import ml_collections
        import torch
        config = ml_collections.ConfigDict()
        config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        config.training = training = ml_collections.ConfigDict()
        training.batch_size = 64
        training.sde = 'vesde'
        training.continuous = True
        training.reduce_mean = True
        training.spl = False
        training.likelihood_weighting = False
        
        config.model = model = ml_collections.ConfigDict()
        model.layer_type = 'concatsquash'
        model.name = 'ncsnpp_tabular'
        model.scale_by_sigma = False
        model.ema_rate = 0.9999
        model.activation = 'elu'
        model.nf = 64
        model.hidden_dims = (256, 512, 1024, 1024, 512, 256)
        model.conditional = True
        model.embedding_type = 'fourier'
        model.fourier_scale = 16
        model.conv_size = 3
        model.sigma_min = 0.01
        model.sigma_max = 10.
        model.num_scales = 50
        model.alpha0 = 0.3
        model.beta0 = 0.95
        
        config.optim = optim = ml_collections.ConfigDict()
        optim.weight_decay = 0
        optim.optimizer = 'Adam'
        optim.lr = 2e-3
        optim.beta1 = 0.9
        optim.eps = 1e-8
        optim.warmup = 5000
        optim.grad_clip = 1.
        
        config.data = data = ml_collections.ConfigDict()
        # +1 for Y, +1 for T (alpha_target). STaSy joint generation.
        data.image_size = self.t_steps * self.feature_dim + 2 
        
        config.sampling = sampling = ml_collections.ConfigDict()
        sampling.method = 'ode'
        sampling.predictor = 'euler_maruyama'
        sampling.corrector = 'none'
        sampling.probability_flow = True
        sampling.snr = 0.16
        sampling.noise_removal = True
        
        return config

    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training STaSy {'(Debug)' if debug_mode else ''}...")
        import torch, numpy as np
        from models import utils as stasy_mutils
        import sde_lib
        import losses as stasy_losses
        from models import ncsnpp_tabular # Registers the model in _MODELS dict
        
        config = self._get_config()
        config.device = device
        
        self.score_model = stasy_mutils.create_model(config)
        self.score_model.to(device)
        from models.ema import ExponentialMovingAverage
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=config.model.ema_rate)
        self.optimizer = stasy_losses.get_optimizer(config, self.score_model.parameters())
        self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=50)
        
        optimize_fn = stasy_losses.optimization_manager(config)
        
        train_step_fn = stasy_losses.get_step_fn(self.sde, train=True, optimize_fn=optimize_fn,
                                                 reduce_mean=config.training.reduce_mean, continuous=config.training.continuous,
                                                 likelihood_weighting=False, workdir=None, spl=False)
                                                 
        self.state = dict(optimizer=self.optimizer, model=self.score_model, ema=self.ema, step=0, epoch=0)
        
        total_epochs = epochs if not debug_mode else 5
        log_every = 20
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            max_batches = min(2, len(dataloader)) if debug_mode else len(dataloader)
            for i, batch in enumerate(dataloader):
                if debug_mode and i >= 2: break
                
                x = batch['x'].cpu().numpy()
                alpha_tgt = batch['alpha_target'].cpu().numpy()
                y = batch['y'].cpu().numpy()
                
                b_size = x.shape[0]
                x_flat = x.reshape(b_size, -1)
                y_flat = y.reshape(b_size, -1)
                t_flat = alpha_tgt.reshape(b_size, -1)
                # Concatenate X, T, Y for joint generation
                xyt_flat = np.concatenate([x_flat, t_flat, y_flat], axis=1)
                
                xyt_tensor = torch.tensor(xyt_flat, dtype=torch.float32, device=device)
                loss = train_step_fn(self.state, xyt_tensor)
                loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                epoch_loss += loss_value
                epoch_batches += 1
                batch_idx = i + 1
                if batch_idx % log_every == 0 or batch_idx == max_batches:
                    logger.info(f"[STaSy] Epoch {epoch + 1}/{total_epochs} - batch {batch_idx}/{max_batches}, loss={loss_value:.6f}")
            avg_loss = epoch_loss / max(1, epoch_batches)
            logger.info(f"[STaSy] Epoch {epoch + 1}/{total_epochs} - avg_loss={avg_loss:.6f}, batches={epoch_batches}")
                
        self.fitted = True

    def sample(self, batch_size, alpha_target, device):
        if not self.fitted:
             return torch.randn(batch_size, self.t_steps, self.feature_dim, device=device), \
                    torch.randn(batch_size, 1, device=device)
                    
        import sampling, torch
        config = self._get_config()
        sampling_shape = (batch_size, config.data.image_size)
        
        def dummy_scaler(x): return x
        
        sampling_fn = sampling.get_sampling_fn(config, self.sde, sampling_shape, dummy_scaler, eps=1e-5)
        
        self.ema.copy_to(self.score_model.parameters())
        samples, n = sampling_fn(self.score_model, sampling_shape=sampling_shape)
        self.ema.restore(self.score_model.parameters())
        
        XYT_cf = samples.detach().clone()
        # Physics SDE clipping: Untamed 1-epoch SDE output causes massive divergence due to sigma_max.
        XYT_cf = torch.clamp(XYT_cf, -3.0, 3.0) 

        # Shape: [Batch, (T*D) + T_dim(1) + Y_dim(1)] 
        # Last index is Y, second last is T
        Y_cf = XYT_cf[:, -1:]
        T_cf = XYT_cf[:, -2:-1] # Ignoring this generated T eventually, relying on alpha_target logic? No, wait. 
        X_cf = XYT_cf[:, :-2]
        
        X_cf_reshaped = X_cf.view(batch_size, self.t_steps, self.feature_dim)
        
        # Decode Analog Bits back to semantic integers for STaSy
        import os, json
        meta_path = 'src/data/dataset_metadata.json'
        if not os.path.exists(meta_path):
             meta_path = 'dataset_metadata.json'
        with open(meta_path, 'r') as f:
             meta = json.load(f)
             
        D_discrete = len(meta['columns'])
        X_cf_semantic = torch.zeros((batch_size, self.t_steps, D_discrete), device=device)
        
        for t in range(self.t_steps):
            analog_offset = 0
            for i_col, col_meta in enumerate(meta['columns']):
                dim = col_meta['dim']
                feat = X_cf_reshaped[:, t, analog_offset : analog_offset + dim]
                if col_meta['type'] == 'continuous':
                    X_cf_semantic[:, t, i_col:i_col+1] = feat
                else:
                    bits = (feat > 0).long()
                    idx_val = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                    for b in range(dim):
                        idx_val = (idx_val << 1) | bits[:, b:b+1]
                    X_cf_semantic[:, t, i_col:i_col+1] = idx_val.float()
                analog_offset += dim
                
        return X_cf_semantic[:, -1, :], Y_cf

class TSDiffWrapper(BaselineWrapper):
    def __init__(self, t_steps, feature_dim, **kwargs):
        super().__init__(t_steps, feature_dim, **kwargs)
        self.fitted = False

    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training TSDiff (ICLR 2023) {'(Debug)' if debug_mode else ''}...")
        
        import torch, json, os, sys
        self.tsdiff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tsdiff_core'))
        if self.tsdiff_path not in sys.path:
            sys.path.insert(0, self.tsdiff_path)
        from model import TSDiffDDPM
        
        meta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_metadata.json'))
        with open(meta_path, 'r') as f:
             self.metadata = json.load(f)
             
        # Feature sizes
        self.d_discrete = len(self.metadata['columns'])
        
        # Determine total dimensions
        sample_batch = next(iter(dataloader))
        # Total Features = (T_steps * Original Num/AnalogDims) + T_target + Y
        x_dim = sample_batch['x'].reshape(sample_batch['x'].shape[0], -1).shape[1]
        self.total_dim = x_dim + 1 + 1 # + alpha_target + Y
        
        # Using 50 timesteps in DDPM for speed instead of 1000
        self.model = TSDiffDDPM(input_dim=self.total_dim, timesteps=50 if debug_mode else 1000).to(device)
        self.model.train()
        
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        
        total_epochs = epochs if not debug_mode else 5
        log_every = 20
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            max_batches = min(2, len(dataloader)) if debug_mode else len(dataloader)
            for i, batch in enumerate(dataloader):
                if debug_mode and i >= 2: break
                
                # Format: [Batch, T, Analog]
                x_num = batch['x'].to(device).reshape(batch['x'].shape[0], -1)
                t_tgt = batch['alpha_target'].to(device).reshape(-1, 1)
                y_tgt = batch['y'].to(device).reshape(-1, 1)
                
                # ADAPTER PATTERN: Concatenate 2D features
                x_combined_2d = torch.cat([x_num, t_tgt, y_tgt], dim=1) # [Batch, Features]
                
                # ADAPTER PATTERN: Inject fake Sequence Dimension = 1 (3D format for Time Series Diffusion)
                x_combined_3d = x_combined_2d.unsqueeze(1) # [Batch, 1, Features]
                
                optimizer.zero_grad()
                loss = self.model.train_step(x_combined_3d)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                epoch_loss += loss_value
                epoch_batches += 1
                batch_idx = i + 1
                if batch_idx % log_every == 0 or batch_idx == max_batches:
                    logger.info(f"[TSDiff] Epoch {epoch + 1}/{total_epochs} - batch {batch_idx}/{max_batches}, loss={loss_value:.6f}")
            avg_loss = epoch_loss / max(1, epoch_batches)
            logger.info(f"[TSDiff] Epoch {epoch + 1}/{total_epochs} - avg_loss={avg_loss:.6f}, batches={epoch_batches}")
                
        self.fitted = True

    def sample(self, batch_size, alpha_target, device):
        import torch
        if not self.fitted:
             return torch.zeros((batch_size, self.d_discrete), device=device), \
                    torch.zeros(batch_size, 1, device=device)
                    
        self.model.eval()
        
        # 1. ADAPTER PATTERN: Instruct TSDiff to generate [Batch, Seq=1, Features]
        samples_3d = self.model.sample(batch_size, seq_len=1, features=self.total_dim, device=device)
        
        # 2. ADAPTER PATTERN: Squeeze fake Sequence dimension back to 2D
        samples_2d = samples_3d.squeeze(1) # [Batch, Features]
        
        # Distribute the dimensions
        Y_cf_tensor = samples_2d[:, -1:]
        T_cf = samples_2d[:, -2:-1] 
        X_cf_analog = samples_2d[:, :-2]
        
        X_cf_reshaped = X_cf_analog.view(batch_size, self.t_steps, -1)
        
        # Reconstruct continuous and discrete classes correctly
        X_cf_semantic = torch.zeros((batch_size, self.t_steps, self.d_discrete), device=device)
        for t in range(self.t_steps):
            analog_offset = 0
            for i_col, col_meta in enumerate(self.metadata['columns']):
                dim = col_meta['dim']
                feat = X_cf_reshaped[:, t, analog_offset : analog_offset + dim]
                if col_meta['type'] == 'continuous':
                    X_cf_semantic[:, t, i_col:i_col+1] = feat
                else:
                    bits = (feat > 0).long()
                    idx_val = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                    for b in range(dim):
                        idx_val = (idx_val << 1) | bits[:, b:b+1]
                    X_cf_semantic[:, t, i_col:i_col+1] = idx_val.float()
                analog_offset += dim
        
        # Apply Y binarization threshold
        Y_cf_tensor = (Y_cf_tensor > 0.5).float()
                
        # Return pure sequence-less final state
        return X_cf_semantic[:, -1, :], Y_cf_tensor

class TabSynWrapper(BaselineWrapper):
    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training TabSyn (VAE + Diffusion) {'(Debug)' if debug_mode else ''}...")
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import sys, os
        self.tabsyn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tabsyn_core'))
        if self.tabsyn_path not in sys.path:
            sys.path.insert(0, self.tabsyn_path)
            
        from .tabsyn_core.vae.model import Model_VAE
        from .tabsyn_core.model import MLPDiffusion, Model
        
        # 1. Collect all data
        import json, os
        meta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_metadata.json'))
        with open(meta_path, 'r') as f:
             self.metadata = json.load(f)
             
        self.k_classes_list = [c['classes'] for c in self.metadata['categorical']]
        
        all_x_cont_flat, all_x_cat_flat = [], []
        
        for batch in dataloader:
            x_batch = batch['x'].to(device) # [B, T=3, D_orig_analog]
            x_cat_raw = batch['x_cat_raw'].to(device) # [B, T=3, num_cats]
            t_batch = batch['alpha_target'].to(device) # [B, 1]
            y_batch = batch['y'].to(device) # [B, 1]
            b_size = x_batch.shape[0]
            
            # Extract only continuous slices across all T
            x_cont_t_list = []
            for t in range(self.t_steps):
                offset = 0
                for col_meta in self.metadata['columns']:
                    dim = col_meta['dim']
                    if col_meta['type'] == 'continuous':
                        x_cont_t_list.append(x_batch[:, t, offset:offset+dim])
                    offset += dim
            x_cont_flat = torch.cat(x_cont_t_list, dim=1) if len(x_cont_t_list) > 0 else torch.zeros(b_size, 0).to(device)
            
            x_num = torch.cat([x_cont_flat, t_batch], dim=1) # [B, T*cont + 1]
            
            x_cat_t_flat = x_cat_raw.reshape(b_size, -1)
            x_cat = torch.cat([x_cat_t_flat, y_batch.long().view(b_size, 1)], dim=1)
            
            all_x_cont_flat.append(x_num)
            all_x_cat_flat.append(x_cat)
            
        X_num = torch.cat(all_x_cont_flat, dim=0)
        X_cat = torch.cat(all_x_cat_flat, dim=0)
        
        self.d_numerical = X_num.shape[1]
        
        self.categories = []
        for _ in range(self.t_steps):
            self.categories.extend(self.k_classes_list)
        self.categories.append(self.metadata['y_col']['classes'])
        
        self.d_token = 4
        
        self.vae = Model_VAE(num_layers=2, d_numerical=self.d_numerical, categories=self.categories if len(self.categories)>0 else [], d_token=self.d_token, n_head=1, factor=32, bias=True).to(device)
        vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=1e-5)
        
        from torch.utils.data import TensorDataset, DataLoader
        
        if X_cat.shape[1] > 0:
             ds = TensorDataset(X_num, X_cat)
        else:
             ds = TensorDataset(X_num, torch.zeros((X_num.shape[0], 1)))
             
        dl = DataLoader(ds, batch_size=4096, shuffle=True)
        
        vae_epochs = 50 if debug_mode else 4000
        self.vae.train()
        vae_log_every = 20
        for ep in range(vae_epochs):
            ep_loss_sum = 0.0
            ep_batches = 0
            vae_max_batches = len(dl)
            for batch_arr in dl:
                vae_optimizer.zero_grad()
                
                batch_num = batch_arr[0].to(device)
                batch_cat = batch_arr[1].to(device) if len(self.categories) > 0 else None
                
                Recon_X_num, Recon_X_cat, mu_z, std_z = self.vae(batch_num, batch_cat)
                
                mse_loss = (batch_num - Recon_X_num).pow(2).mean() if self.d_numerical > 0 else 0.0
                
                ce_loss = 0.0
                import torch.nn.functional as F
                if batch_cat is not None:
                    for c_idx in range(len(self.categories)):
                        ce_loss += F.cross_entropy(Recon_X_cat[c_idx], batch_cat[:, c_idx])
                    ce_loss = ce_loss / len(self.categories)
                
                temp = 1 + std_z - mu_z.pow(2) - std_z.exp()
                loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
                
                loss = mse_loss + ce_loss + 1e-2 * loss_kld
                loss.backward()
                vae_optimizer.step()
                loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                ep_loss_sum += loss_value
                ep_batches += 1
                if ep_batches % vae_log_every == 0 or ep_batches == vae_max_batches:
                    logger.info(f"[TabSyn-VAE] Epoch {ep + 1}/{vae_epochs} - batch {ep_batches}/{vae_max_batches}, loss={loss_value:.6f}")
            vae_avg_loss = ep_loss_sum / max(1, ep_batches)
            logger.info(f"[TabSyn-VAE] Epoch {ep + 1}/{vae_epochs} - avg_loss={vae_avg_loss:.6f}, batches={ep_batches}")
                
        self.vae.eval()
        with torch.no_grad():
            self.train_z = self.vae.get_embedding(X_num, X_cat if len(self.categories)>0 else None)
            
        self.train_z = self.train_z[:, 1:, :]
        B, num_tokens, token_dim = self.train_z.size()
        in_dim = num_tokens * token_dim
        self.train_z = self.train_z.view(B, in_dim).cpu().numpy()
            
        self.z_mean = torch.tensor(self.train_z.mean(0), dtype=torch.float32, device=device)
        self.z_std = torch.tensor(self.train_z.std(0), dtype=torch.float32, device=device)
        
        train_z_tensor = (torch.tensor(self.train_z, device=device) - self.z_mean) / 2
        
        self.denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        self.diff_model = Model(denoise_fn=self.denoise_fn, hid_dim=in_dim).to(device)
        diff_optimizer = torch.optim.Adam(self.diff_model.parameters(), lr=1e-3, weight_decay=0)
        
        diff_dl = DataLoader(TensorDataset(train_z_tensor), batch_size=4096, shuffle=True)
        diff_epochs = 50 if debug_mode else 10000
        diff_log_every = 20
        
        self.diff_model.train()
        for ep in range(diff_epochs):
            ep_loss_sum = 0.0
            ep_batches = 0
            diff_max_batches = len(diff_dl)
            for (bz,) in diff_dl:
                diff_optimizer.zero_grad()
                loss = self.diff_model(bz)
                loss.mean().backward()
                diff_optimizer.step()
                loss_value = float(loss.mean().item()) if hasattr(loss, 'mean') else float(loss)
                ep_loss_sum += loss_value
                ep_batches += 1
                if ep_batches % diff_log_every == 0 or ep_batches == diff_max_batches:
                    logger.info(f"[TabSyn-Diffusion] Epoch {ep + 1}/{diff_epochs} - batch {ep_batches}/{diff_max_batches}, loss={loss_value:.6f}")
            diff_avg_loss = ep_loss_sum / max(1, ep_batches)
            logger.info(f"[TabSyn-Diffusion] Epoch {ep + 1}/{diff_epochs} - avg_loss={diff_avg_loss:.6f}, batches={ep_batches}")
                
        self.fitted = True

    def sample(self, batch_size, alpha_target, device):
        if not getattr(self, 'fitted', False):
             return torch.randn(batch_size, self.t_steps, self.feature_dim, device=device), \
                    torch.randn(batch_size, 1, device=device)

        from .tabsyn_core.diffusion_utils import sample as tabsyn_sample
        import torch, numpy as np
        
        self.diff_model.eval()
        self.vae.eval()
        
        with torch.no_grad():
            z_sampled = tabsyn_sample(self.diff_model.denoise_fn_D, batch_size, self.diff_model.denoise_fn_D.hid_dim, num_steps=50, device=device)
            z_sampled = z_sampled * 2 + self.z_mean
            
            z_sampled = z_sampled.reshape(batch_size, -1, self.d_token)
            
            h = self.vae.VAE.decoder(z_sampled)
            recon_x_num, recon_x_cat = self.vae.Reconstructor(h)
            
            x_cont_cf_flat = recon_x_num[:, :-1]
            
            cat_preds = []
            if recon_x_cat is not None:
                for logits in recon_x_cat:
                    cat_preds.append(logits.argmax(dim=-1, keepdim=True).float())
                cat_preds_tensor = torch.cat(cat_preds, dim=1) # [B, T*num_cats + 1]
            else:
                cat_preds_tensor = torch.zeros((batch_size, 1), device=device)
                
            Y_cf_tensor = cat_preds_tensor[:, -1:]
            pred_cats_only = cat_preds_tensor[:, :-1]
            
            
            # Interleave into properly sized array
            D_discrete = len(self.metadata['columns'])
            X_cf = torch.zeros((batch_size, self.t_steps, D_discrete), device=device)
            
            cont_idx = 0
            cat_idx = 0
            
            for t in range(self.t_steps):
                for i_col, col_meta in enumerate(self.metadata['columns']):
                    dim = col_meta['dim']
                    
                    if col_meta['type'] == 'continuous':
                        # Pull from x_cont_cf_flat
                        feat = x_cont_cf_flat[:, cont_idx : cont_idx + dim]
                        cont_idx += dim
                        X_cf[:, t, i_col : i_col + 1] = feat
                    else:
                        idx_val = pred_cats_only[:, cat_idx].float()
                        cat_idx += 1
                        X_cf[:, t, i_col : i_col + 1] = idx_val.unsqueeze(1)    
                    
        return X_cf[:, -1, :], Y_cf_tensor

class TabDiffWrapper(BaselineWrapper):
    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training TabDiff (ICLR 2025) {'(Debug)' if debug_mode else ''}...")
        
        from .tabdiff_core.modules.main_modules import UniModMLP, Model
        from .tabdiff_core.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
        
        # Determine exact dataset metadata config
        import os
        metadata_path = 'src/data/dataset_metadata.json'
        if not os.path.exists(metadata_path):
             metadata_path = 'dataset_metadata.json' # Fallback if run elsewhere
             
        with open(metadata_path, 'r') as f:
            import json
            meta = json.load(f)
            
        columns = meta['columns']
        d_numerical = 0
        categories = []
        
        # We need to know how many continuous features vs categorical features exist.
        # TabDiff expects d_numerical, and categories (list of num_classes).
        # We will flatten X over t_steps.
        
        for t in range(self.t_steps):
            for col in columns:
                if col['type'] == 'continuous':
                    d_numerical += 1
                else:
                    # TabDiff needs the NUMBER OF CLASSES, which is 'classes' inside meta['categorical']
                    pass # We will populate categories array correctly below
                    
        # Find classes for categorical
        cat_classes_list = [c['classes'] for c in meta['categorical']]
        for t in range(self.t_steps):
            categories.extend(cat_classes_list)
            
        # Add alpha_target (continuous)
        d_numerical += 1
        
        # Add Y (categorical)
        categories.append(meta['y_col']['classes'])
        
        categories_np = np.array(categories)
        self.metadata = meta
        self.d_numerical = d_numerical
        self.categories = categories_np
        self.k_classes_list = cat_classes_list
        
        # Build Model
        backbone = UniModMLP(
            d_numerical=d_numerical,
            categories=categories_np + 1, # +1 for mask token on each category
            num_layers=3 if debug_mode else 6,
            d_token=192 if debug_mode else 256,
            dim_t=128 if debug_mode else 512,
            use_mlp=True,
            n_head=1 if debug_mode else 8,
            factor=4,
            bias=True
        )
        model = Model(backbone, sigma_data=0.5, precond=True)
        model.to(device)
        
        self.model = UnifiedCtimeDiffusion(
            num_classes=categories_np,
            num_numerical_features=d_numerical,
            denoise_fn=model,
            y_only_model=None,
            num_timesteps=10 if debug_mode else 50,
            scheduler='power_mean_per_column',
            cat_scheduler='log_linear_per_column',
            noise_dist='uniform_t', # Using default setup
            edm_params={'sigma_data': 0.5, 'precond': True},
            noise_schedule_params={'rho': 7.0} if d_numerical > 0 else {}, # Simplify config
            sampler_params={'stochastic_sampler': False, 'second_order_correction': True},
            device=device
        )
        self.model.to(device)
        self.model.train()
        
        from torch.optim import AdamW
        optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        total_epochs = epochs if not debug_mode else 5
        log_every = 20
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            max_batches = min(2, len(dataloader)) if debug_mode else len(dataloader)
            for i, batch in enumerate(dataloader):
                if debug_mode and i >= 2: break
                
                x_batch = batch['x'].to(device) # [B, T, D_orig_analog]
                x_cat_raw = batch['x_cat_raw'].to(device) # [B, T, num_cats]
                t_batch = batch['alpha_target'].to(device).view(-1, 1) # [B, 1]
                y_batch = batch['y'].to(device).view(-1, 1) # [B, 1]
                b_size = x_batch.shape[0]
                
                # Extract only continuous slices across all T
                x_cont_t_list = []
                for t in range(self.t_steps):
                    offset = 0
                    for col_meta in self.metadata['columns']:
                        dim = col_meta['dim']
                        if col_meta['type'] == 'continuous':
                            x_cont_t_list.append(x_batch[:, t, offset:offset+dim])
                        offset += dim
                
                x_cont_flat = torch.cat(x_cont_t_list, dim=1) if len(x_cont_t_list) > 0 else torch.zeros(b_size, 0).to(device)
                x_num = torch.cat([x_cont_flat, t_batch], dim=1) # [B, T*cont + 1]
                
                x_cat_t_flat = x_cat_raw.reshape(b_size, -1)
                y_cat = (y_batch > 0.5).long()
                x_cat = torch.cat([x_cat_t_flat, y_cat], dim=1)
                
                # Patch missing values mapping (-1 -> 0)
                x_cat = torch.where(x_cat < 0, torch.zeros_like(x_cat), x_cat)
                
                x_combined = torch.cat([x_num, x_cat], dim=1)

                optimizer.zero_grad()
                d_loss, c_loss = self.model.mixed_loss(x_combined)
                loss = d_loss + c_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                epoch_loss += loss_value
                epoch_batches += 1
                batch_idx = i + 1
                if batch_idx % log_every == 0 or batch_idx == max_batches:
                    logger.info(f"[TabDiff] Epoch {epoch + 1}/{total_epochs} - batch {batch_idx}/{max_batches}, loss={loss_value:.6f}")
            avg_loss = epoch_loss / max(1, epoch_batches)
            logger.info(f"[TabDiff] Epoch {epoch + 1}/{total_epochs} - avg_loss={avg_loss:.6f}, batches={epoch_batches}")
                
        self.fitted = True

    @torch.no_grad()
    def sample(self, batch_size, alpha_target, device):
        if not self.fitted:
             return torch.randn(batch_size, self.t_steps, self.feature_dim, device=device), \
                    torch.randn(batch_size, 1, device=device)
        
        self.model.eval()
        
        sample_out = self.model.sample(batch_size).to(device)
        
        x_num_out = sample_out[:, :self.d_numerical]
        x_cat_out = sample_out[:, self.d_numerical:]
        
        x_cont_cf_flat = x_num_out[:, :-1]
        
        pred_cats_only = x_cat_out[:, :-1]
        Y_cf_tensor = x_cat_out[:, -1:]
        
        # Clean discrete output reconstruction without Analog Bits corruption!
        D_discrete = len(self.metadata['columns'])
        X_cf = torch.zeros((batch_size, self.t_steps, D_discrete), device=device)
        
        cont_idx = 0
        cat_idx = 0
        
        for t in range(self.t_steps):
            for i_col, col_meta in enumerate(self.metadata['columns']):
                if col_meta['type'] == 'continuous':
                    dim = col_meta['dim']
                    # Pull from x_cont_cf_flat
                    feat = x_cont_cf_flat[:, cont_idx : cont_idx + dim]
                    cont_idx += dim
                    X_cf[:, t, i_col : i_col + 1] = feat
                else:
                    # Straight uncorrupted Category Indices!
                    idx_val = pred_cats_only[:, cat_idx:cat_idx+1].float()
                    cat_idx += 1
                    X_cf[:, t, i_col : i_col + 1] = idx_val
                    
        Y_cf_tensor = (Y_cf_tensor > 0.5).float()
        
        return X_cf[:, -1, :], Y_cf_tensor

class CausalTabDiffWrapper(BaselineWrapper):
    """
    Wrapper for our proposed model to keep the evaluation loop unified.
    """
    def __init__(self, t_steps, feature_dim, **kwargs):
        super().__init__(t_steps, feature_dim, **kwargs)
        from src.models.causal_tabdiff import CausalTabDiff
        # If in debug mode, keep diffusion steps small, otherwise 100 for evaluation
        # Let's read from kwargs or default to a small number
        diffusion_steps = kwargs.get('diffusion_steps', 100)
        self.model = CausalTabDiff(t_steps=t_steps, feature_dim=feature_dim, diffusion_steps=diffusion_steps)
        
    def fit(self, dataloader, epochs, device, debug_mode=False):
        logger.info(f"Training Causal-TabDiff {'(Debug)' if debug_mode else ''}...")
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            num_batches = len(dataloader) if not debug_mode else min(2, len(dataloader))
            epoch_loss = 0
            log_every = 20
            for i, batch in enumerate(dataloader):
                if debug_mode and i >= 2: break
                x = batch['x'].to(device)
                alpha_tgt = batch['alpha_target'].to(device)
                
                optimizer.zero_grad()
                # Assuming causal_tabdiff architecture will be updated to output Y
                # For now, placeholder diff_loss
                diff_loss, disc_loss = self.model(x, alpha_tgt)
                loss = diff_loss + 0.5 * disc_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_idx = i + 1
                if batch_idx % log_every == 0 or batch_idx == num_batches:
                    logger.info(f"[Causal-TabDiff] Epoch {epoch + 1}/{epochs} - batch {batch_idx}/{num_batches}, loss={loss.item():.6f}")
            avg_loss = epoch_loss / max(1, num_batches)
            logger.info(f"[Causal-TabDiff] Epoch {epoch + 1}/{epochs} - avg_loss={avg_loss:.6f}, batches={num_batches}")

    def sample(self, batch_size, alpha_target, device):
        self.model.eval()
        with torch.no_grad():
            sampled = self.model.sample_with_guidance(batch_size=batch_size, alpha_target=alpha_target, guidance_scale=2.0)
            
        import os, json
        meta_path = 'src/data/dataset_metadata.json'
        if not os.path.exists(meta_path):
             meta_path = 'dataset_metadata.json'
        with open(meta_path, 'r') as f:
             meta = json.load(f)
             
        D_discrete = len(meta['columns'])
        X_cf_semantic = torch.zeros((batch_size, self.t_steps, D_discrete), device=device)
        
        for t in range(self.t_steps):
            analog_offset = 0
            for i_col, col_meta in enumerate(meta['columns']):
                dim = col_meta['dim']
                feat = sampled[:, t, analog_offset : analog_offset + dim]
                if col_meta['type'] == 'continuous':
                    X_cf_semantic[:, t, i_col:i_col+1] = feat
                else:
                    bits = (feat > 0).long()
                    idx_val = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                    for b in range(dim):
                        idx_val = (idx_val << 1) | bits[:, b:b+1]
                    X_cf_semantic[:, t, i_col:i_col+1] = idx_val.float()
                analog_offset += dim
        
        return X_cf_semantic[:, -1, :], torch.randn(batch_size, 1, device=device)
