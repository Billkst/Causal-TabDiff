import unittest
import torch
from src.models.causal_tabdiff import CausalTabDiff

class TestCausalTabDiff(unittest.TestCase):
    def setUp(self):
        """
        Step B: Test-Driven Initialization
        Setting up the isolated CausalTabDiff module.
        """
        self.batch_size = 4
        self.t_steps = 3
        self.feature_dim = 16
        self.model = CausalTabDiff(t_steps=self.t_steps, feature_dim=self.feature_dim, heads=4).to('cpu')
        
    def test_forward_loss_computation(self):
        """Ensure the forward pass computes diffusion and energy losses correctly."""
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        
        diff_loss, disc_loss = self.model(x_0, alpha_target)
        
        self.assertIsNotNone(diff_loss)
        self.assertIsNotNone(disc_loss)
        self.assertTrue(diff_loss.item() > 0)
        self.assertTrue(disc_loss.item() > 0)

    def test_extreme_guidance_sampling(self):
        """
        Step B: Boundary Condition Test - Extreme Guidance
        Ensure the sampling method does not crash and the gradient flows when 'delay=extreme_value'
        or 'guidance_scale' is very large.
        """
        alpha_target = torch.tensor([[0.9], [0.1]]) # Two distinct causal environments
        
        with torch.enable_grad():
             # Test with normal guidance
             out_normal = self.model.sample_with_guidance(2, alpha_target, guidance_scale=1.0)
             # Test with extreme guidance
             out_extreme = self.model.sample_with_guidance(2, alpha_target, guidance_scale=100.0)
             
        self.assertEqual(out_normal.shape, (2, self.t_steps, self.feature_dim))
        self.assertEqual(out_extreme.shape, (2, self.t_steps, self.feature_dim))
        
        # Output should diverge if guidance gradient implies a pull
        self.assertFalse(torch.allclose(out_normal, out_extreme))

    def test_zero_guidance_sampling(self):
         """
         Step B: Boundary Condition Test - Zero Effect (No Causal Guidance)
         Should devolve to standard DDPM without errors.
         """
         alpha_target = torch.tensor([[0.5]])
         
         with torch.enable_grad():
             out_zero = self.model.sample_with_guidance(1, alpha_target, guidance_scale=0.0)
             
         self.assertEqual(out_zero.shape, (1, self.t_steps, self.feature_dim))

if __name__ == '__main__':
    unittest.main()
