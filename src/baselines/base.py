from abc import ABC, abstractmethod

class BaselineWrapper(ABC):
    """
    Abstract interface for all baseline models to ensure a unified evaluation pipeline.
    """
    def __init__(self, t_steps, feature_dim, **kwargs):
        self.t_steps = t_steps
        self.feature_dim = feature_dim
        
    @abstractmethod
    def fit(self, dataloader, epochs, device, debug_mode=False):
        """
        Train the baseline model.
        """
        pass
        
    @abstractmethod
    def sample(self, batch_size, alpha_target, device):
        """
        Generate/predict counterfactual samples given alpha_target.
        Returns: tensor of shape (batch_size, t_steps, feature_dim)
        """
        pass
