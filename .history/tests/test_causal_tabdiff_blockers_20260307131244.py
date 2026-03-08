import os
import unittest
from unittest.mock import Mock

import numpy as np
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

    def test_dataset_can_bind_real_treatment_column(self):
        previous_alpha_col = os.environ.get('ALPHA_TREATMENT_COLUMN')
        os.environ['ALPHA_TREATMENT_COLUMN'] = 'cigsmok'
        try:
            dataset = NLSTDataset(data_dir=DATA_DIR, debug_mode=True)
            expected = dataset.merged_df['cigsmok'].fillna(0).astype(float).to_numpy().reshape(-1, 1)
            self.assertEqual(dataset.alpha_target_source, 'cigsmok')
            self.assertTrue(
                np.allclose(dataset.alpha_target, expected),
                '启用真实 treatment 列后，alpha_target 应与 cigsmok 对齐。',
            )
        finally:
            if previous_alpha_col is None:
                os.environ.pop('ALPHA_TREATMENT_COLUMN', None)
            else:
                os.environ['ALPHA_TREATMENT_COLUMN'] = previous_alpha_col

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

    def test_joint_forward_with_outcome_loss(self):
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        y_target = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])
        pos_weight = torch.tensor([2.0])

        diff_loss, disc_loss, outcome_loss = self.model(
            x_0,
            alpha_target,
            y_target=y_target,
            pos_weight=pos_weight,
        )

        self.assertTrue(torch.isfinite(diff_loss))
        self.assertTrue(torch.isfinite(disc_loss))
        self.assertTrue(torch.isfinite(outcome_loss))
        self.assertGreater(outcome_loss.item(), 0.0)

    def test_joint_forward_with_ranking_loss_stays_finite(self):
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        y_target = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])

        _, _, outcome_loss = self.model(
            x_0,
            alpha_target,
            y_target=y_target,
            pos_weight=torch.tensor([2.0]),
            rank_loss_weight=0.5,
        )

        self.assertTrue(torch.isfinite(outcome_loss))
        self.assertGreater(outcome_loss.item(), 0.0)

    def test_model_outcome_scores_change_with_alpha(self):
        x = torch.zeros(4, self.t_steps, self.feature_dim)
        alpha_low = torch.zeros(4, 1)
        alpha_high = torch.ones(4, 1)

        score_low = self.model.predict_outcome_proba(x, alpha_low)
        score_high = self.model.predict_outcome_proba(x, alpha_high)

        self.assertFalse(
            torch.allclose(score_low, score_high),
            '主模型内置 outcome head 应对 alpha 条件有响应。',
        )

    def test_model_v2_cumulative_risk_changes_with_alpha(self):
        x = torch.zeros(4, self.t_steps, self.feature_dim)
        alpha_low = torch.zeros(4, 1)
        alpha_high = torch.ones(4, 1)

        risk_low = self.model.predict_cumulative_risk(x, alpha_low)
        risk_high = self.model.predict_cumulative_risk(x, alpha_high)

        self.assertTrue(torch.isfinite(risk_low).all())
        self.assertTrue(torch.isfinite(risk_high).all())
        self.assertFalse(
            torch.allclose(risk_low, risk_high),
            'V2 轨迹风险读出应对 alpha 条件有响应。',
        )

    def test_joint_forward_with_trajectory_risk_head_stays_finite(self):
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        y_target = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])

        diff_loss, disc_loss, outcome_loss = self.model(
            x_0,
            alpha_target,
            y_target=y_target,
            pos_weight=torch.tensor([2.0]),
            use_trajectory_risk_head=True,
            risk_smoothness_weight=0.1,
            cf_consistency_weight=0.1,
        )

        self.assertTrue(torch.isfinite(diff_loss))
        self.assertTrue(torch.isfinite(disc_loss))
        self.assertTrue(torch.isfinite(outcome_loss))
        self.assertGreater(outcome_loss.item(), 0.0)

    def test_joint_forward_with_v21_fidelity_anchor_stays_finite(self):
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        y_target = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])

        diff_loss, disc_loss, outcome_loss = self.model(
            x_0,
            alpha_target,
            y_target=y_target,
            pos_weight=torch.tensor([2.0]),
            use_trajectory_risk_head=True,
            risk_smoothness_weight=0.1,
            cf_consistency_weight=0.1,
            denoise_recon_weight=0.1,
            batch_moment_weight=0.05,
        )

        self.assertTrue(torch.isfinite(diff_loss))
        self.assertTrue(torch.isfinite(disc_loss))
        self.assertTrue(torch.isfinite(outcome_loss))
        self.assertGreater(diff_loss.item(), 0.0)

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

    def test_sampling_with_late_guidance_schedule_is_reproducible(self):
        alpha_target = torch.full((2, 1), 0.4, device=self.device)

        torch.manual_seed(321)
        out_first = self.model.sample_with_guidance(
            2,
            alpha_target,
            guidance_scale=1.5,
            guidance_schedule='late',
            guidance_power=2.0,
        )

        torch.manual_seed(321)
        out_second = self.model.sample_with_guidance(
            2,
            alpha_target,
            guidance_scale=1.5,
            guidance_schedule='late',
            guidance_power=2.0,
        )

        self.assertTrue(torch.allclose(out_first, out_second))

    def test_model_without_noise_head_stays_finite(self):
        alt_model = CausalTabDiff(
            t_steps=self.t_steps,
            feature_dim=self.feature_dim,
            diffusion_steps=10,
            heads=1,
            use_noise_head=False,
        ).to(self.device)
        x_0 = torch.randn(self.batch_size, self.t_steps, self.feature_dim)
        alpha_target = torch.rand(self.batch_size, 1)
        y_target = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])

        diff_loss, disc_loss, outcome_loss = alt_model(
            x_0,
            alpha_target,
            y_target=y_target,
            pos_weight=torch.tensor([2.0]),
            use_trajectory_risk_head=True,
        )

        self.assertTrue(torch.isfinite(diff_loss))
        self.assertTrue(torch.isfinite(disc_loss))
        self.assertTrue(torch.isfinite(outcome_loss))

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