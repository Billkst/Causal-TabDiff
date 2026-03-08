import os
import unittest
from unittest.mock import Mock

import torch

from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import NLSTDataset
from src.models.causal_tabdiff import CausalTabDiff


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOLEAK_METADATA_PATH = os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


class TestCausalTabDiffBlockers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['DATASET_METADATA_PATH'] = NOLEAK_METADATA_PATH
        cls.dataset = NLSTDataset(data_dir=DATA_DIR, debug_mode=False)
        cls.empirical_positive_rate = float((cls.dataset.y.reshape(-1) > 0.5).mean())

    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 8
        self.t_steps = 3
        self.feature_dim = 7
        self.device = torch.device('cpu')
        self.model = CausalTabDiff(
            t_steps=self.t_steps,
            feature_dim=self.feature_dim,
            diffusion_steps=10,
            heads=1,
        ).to(self.device)

    def test_sampling_reproducible_given_fixed_seed(self):
        alpha_target = torch.full((2, 1), 0.3, device=self.device)

        torch.manual_seed(123)
        out_first = self.model.sample_with_guidance(2, alpha_target, guidance_scale=1.0)

        torch.manual_seed(123)
        out_second = self.model.sample_with_guidance(2, alpha_target, guidance_scale=1.0)

        self.assertTrue(
            torch.allclose(out_first, out_second),
            '固定随机种子后，采样输出应可复现。',
        )

    def test_alpha_condition_changes_samples_under_fixed_seed(self):
        alpha_low = torch.full((2, 1), 0.1, device=self.device)
        alpha_high = torch.full((2, 1), 0.9, device=self.device)

        torch.manual_seed(456)
        out_low = self.model.sample_with_guidance(2, alpha_low, guidance_scale=1.0)

        torch.manual_seed(456)
        out_high = self.model.sample_with_guidance(2, alpha_high, guidance_scale=1.0)

        self.assertFalse(
            torch.allclose(out_low, out_high),
            '在固定噪声下，不同 alpha 条件应导致不同样本。',
        )

    def test_wrapper_sample_y_is_not_placeholder_randomness(self):
        wrapper = CausalTabDiffWrapper(
            t_steps=self.t_steps,
            feature_dim=self.feature_dim,
            diffusion_steps=10,
        )
        deterministic_sample = torch.zeros(self.batch_size, self.t_steps, self.feature_dim)
        wrapper.model.sample_with_guidance = Mock(return_value=deterministic_sample)
        alpha_target = torch.full((self.batch_size, 1), 0.4)

        _, y_first = wrapper.sample(self.batch_size, alpha_target, self.device)
        _, y_second = wrapper.sample(self.batch_size, alpha_target, self.device)

        self.assertTrue(
            torch.allclose(y_first, y_second),
            '若主生成轨迹固定，Y 输出不应在两次调用间随机漂移。',
        )

    def test_wrapper_sample_positive_rate_matches_dataset_prior(self):
        wrapper = CausalTabDiffWrapper(
            t_steps=self.t_steps,
            feature_dim=self.feature_dim,
            diffusion_steps=10,
        )
        deterministic_sample = torch.zeros(512, self.t_steps, self.feature_dim)
        wrapper.model.sample_with_guidance = Mock(return_value=deterministic_sample)
        alpha_target = torch.full((512, 1), 0.6)

        _, y_sampled = wrapper.sample(512, alpha_target, self.device)
        sampled_positive_rate = float((y_sampled > 0.5).float().mean().item())

        self.assertLessEqual(
            abs(sampled_positive_rate - self.empirical_positive_rate),
            0.05,
            (
                '生成 Y 的阳性率应接近真实数据基率；'
                f'当前真实基率={self.empirical_positive_rate:.4f}, 生成基率={sampled_positive_rate:.4f}'
            ),
        )


if __name__ == '__main__':
    unittest.main()