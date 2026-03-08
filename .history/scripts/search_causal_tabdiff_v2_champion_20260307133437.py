import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_baselines import compute_metrics, estimate_trainable_params_m
from src.baselines.wrappers import CausalTabDiffWrapper
from src.data.data_module import NLSTDataset


SEED = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_SEED', '7'))
TRAIN_SIZE = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_TRAIN_SIZE', '8192'))
EVAL_SIZE = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_EVAL_SIZE', '4096'))
BATCH_SIZE = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_BATCH_SIZE', '256'))
EPOCHS = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_EPOCHS', '8'))
DIFFUSION_STEPS = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_DIFFUSION_STEPS', '20'))
SEARCH_PROFILE = os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_PROFILE', 'champion').strip().lower()
OUTPUT_PATH = os.environ.get(
    'CAUSAL_TABDIFF_V2_SEARCH_OUTPUT_PATH',
    os.path.join(PROJECT_ROOT, 'logs', 'testing', f'causal_tabdiff_v2_{SEARCH_PROFILE}_search.md'),
)
BASELINE_REPORT_PATH = os.environ.get(
    'CAUSAL_TABDIFF_V2_BASELINE_REPORT_PATH',
    os.path.join(PROJECT_ROOT, 'markdown_report.md'),
)

METRIC_DIRECTIONS = {
    'ATE_Bias': 'lower',
    'Wasserstein': 'lower',
    'CMD': 'lower',
    'TSTR_AUC': 'higher',
    'TSTR_F1': 'higher',
}

SOFT_METRICS = ['Params(M)', 'AvgInfer(ms/sample)']
FORMAL_GATE = {
    'TSTR_AUC': 0.58,
    'TSTR_PR_AUC': 0.04,
    'TSTR_F1': 0.05,
    'TSTR_F1_RealPrev': 0.05,
}

META = [
    {'name': 'race', 'type': 'categorical', 'dim': 4},
    {'name': 'cigsmok', 'type': 'categorical', 'dim': 1},
    {'name': 'gender', 'type': 'categorical', 'dim': 1},
    {'name': 'age', 'type': 'continuous', 'dim': 1},
]

SEARCH_PROFILES = {
    'champion': [
        {
            'name': 'instinct_anchor_075_g10_s010_c005',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'v2_pass_anchor_100_g10_s010_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'risklight_100_g10_s005_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.05,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'riskstrong_100_g10_s020_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.20,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'cflight_100_g10_s010_c002',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.02,
        },
        {
            'name': 'cfstrong_100_g10_s010_c010',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.10,
        },
        {
            'name': 'bothlight_100_g10_s005_c002',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.05,
            'cf_consistency_weight': 0.02,
        },
        {
            'name': 'bothstrong_075_g10_s020_c010',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.20,
            'cf_consistency_weight': 0.10,
        },
        {
            'name': 'guidelow_100_g05_s010_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 0.5,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'guidehigh_100_g15_s010_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.5,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'noguideblend_075_g00_s010_c005',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 0.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
        },
        {
            'name': 'nosmooth_100_g10_s000_c005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.00,
            'cf_consistency_weight': 0.05,
        },
    ],
    'balance': [
        {
            'name': 'balance_task_anchor_out050_ep10_d30',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.5,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.50,
            'epochs': 10,
            'diffusion_steps': 30,
        },
        {
            'name': 'balance_task_anchor_out025_ep12_d30',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.5,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.25,
            'epochs': 12,
            'diffusion_steps': 30,
        },
        {
            'name': 'balance_mid_out050_ep12_d40',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.50,
            'epochs': 12,
            'diffusion_steps': 40,
        },
        {
            'name': 'balance_mid_out025_ep10_d40',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.25,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'balance_blend_out050_ep10_d30',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 1.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.50,
            'epochs': 10,
            'diffusion_steps': 30,
        },
        {
            'name': 'balance_genheavy_out010_ep14_d40',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 0.5,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.10,
            'epochs': 14,
            'diffusion_steps': 40,
        },
    ],
    'refine': [
        {
            'name': 'refine_bridge_out035_ep10_d35_g125',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'refine_bridge_out040_ep10_d35_g125',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'refine_bridge_out045_ep10_d35_g125',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.45,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'refine_tasklean_out035_ep10_d35_g150',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'refine_balance_out035_ep10_d40_g125',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'refine_balance_out030_ep10_d40_g125',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.30,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'refine_gatelean_out030_ep12_d40_g100',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.00,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.30,
            'epochs': 12,
            'diffusion_steps': 40,
        },
        {
            'name': 'refine_tasklean_out040_ep12_d35_g150',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'epochs': 12,
            'diffusion_steps': 35,
        },
    ],
    'fidelity': [
        {
            'name': 'fidelity_anchor_out035_ep10_d35_g150_m100',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'fidelity_out038_ep10_d40_g140_m100',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.40,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.38,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'fidelity_out032_ep10_d40_g140_m100',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.40,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.32,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'fidelity_out030_ep12_d45_g125_m100',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.30,
            'epochs': 12,
            'diffusion_steps': 45,
        },
        {
            'name': 'fidelity_out035_ep10_d40_g140_m090',
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.40,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'epochs': 10,
            'diffusion_steps': 40,
        },
        {
            'name': 'fidelity_out025_ep12_d45_g125_m090',
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.25,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.25,
            'epochs': 12,
            'diffusion_steps': 45,
        },
    ],
    'v21': [
        {
            'name': 'v21_anchor_lateg_out035_d35_r010_m005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.10,
            'batch_moment_weight': 0.05,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v21_moment_stronger_lateg_out035_d35_r010_m010',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.10,
            'batch_moment_weight': 0.10,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v21_lowguide_lateg_out035_d35_r015_m005',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.15,
            'batch_moment_weight': 0.05,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v21_blend_lateg_out035_d35_r010_m005',
            'sample_model_score_weight': 0.75,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.10,
            'batch_moment_weight': 0.05,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v21_balance_lateg_out030_d40_r010_m010',
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.30,
            'denoise_recon_weight': 0.10,
            'batch_moment_weight': 0.10,
            'epochs': 10,
            'diffusion_steps': 40,
        },
    ],
    'v21_ablate': [
        {
            'name': 'ablate_noisehead_only_oldguide',
            'use_noise_head': True,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.0,
            'batch_moment_weight': 0.0,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'ablate_recon_only_oldguide',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.10,
            'batch_moment_weight': 0.05,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'ablate_lateg_only',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.0,
            'batch_moment_weight': 0.0,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'ablate_lightrecon_oldguide',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.0,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'ablate_noisehead_lateg',
            'use_noise_head': True,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'late',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.0,
            'batch_moment_weight': 0.0,
            'epochs': 10,
            'diffusion_steps': 35,
        },
    ],
    'v22_refine': [
        {
            'name': 'v22_base_const_g150_r000_m000',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.00,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_lightrecon_g150_r002_m000',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.02,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_lightrecon_g150_r003_m000',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_reconmoment_g150_r003_m002',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.02,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_reconmoment_g150_r005_m002',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.05,
            'batch_moment_weight': 0.02,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_lightrecon_g125_r003_m000',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_reconmoment_g125_r003_m002',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.02,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v22_lightrecon_out030_g150_r003_m000',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.30,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
    ],
    'v23_gate': [
        {
            'name': 'v23_anchor_g150_r003_o035_m100',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g175_r003_o035_m100',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.75,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g150_r004_o035_m100',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.04,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g150_r003_o040_m100',
            'use_noise_head': False,
            'sample_model_score_weight': 1.00,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g150_r003_o040_m090',
            'use_noise_head': False,
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g175_r002_o040_m090',
            'use_noise_head': False,
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.75,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'denoise_recon_weight': 0.02,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g150_r003_o035_m090',
            'use_noise_head': False,
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.50,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.35,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
        {
            'name': 'v23_g125_r003_o040_m090',
            'use_noise_head': False,
            'sample_model_score_weight': 0.90,
            'sample_guidance_scale': 1.25,
            'sample_guidance_schedule': 'constant',
            'sample_guidance_power': 2.0,
            'risk_smoothness_weight': 0.10,
            'cf_consistency_weight': 0.05,
            'outcome_loss_weight': 0.40,
            'denoise_recon_weight': 0.03,
            'batch_moment_weight': 0.00,
            'epochs': 10,
            'diffusion_steps': 35,
        },
    ],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_semantic_real_x(batch, device):
    x_analog = batch['x'].to(device)
    cat_raw = batch['x_cat_raw'].to(device).float()
    real_x = torch.zeros((x_analog.shape[0], len(META)), device=device)

    analog_offset = 0
    cat_idx = 0
    for i_col, col_meta in enumerate(META):
        if col_meta['type'] == 'continuous':
            real_x[:, i_col:i_col + 1] = x_analog[:, -1, analog_offset:analog_offset + 1]
        else:
            real_x[:, i_col:i_col + 1] = cat_raw[:, -1, cat_idx:cat_idx + 1]
            cat_idx += 1
        analog_offset += col_meta['dim']
    return real_x


def parse_metric_cell(cell: str):
    text = cell.strip()
    if text == 'N/A':
        return float('nan')
    if '±' in text:
        text = text.split('±', 1)[0].strip()
    return float(text)


def parse_baseline_report(report_path: str):
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    table_lines = [line for line in lines if line.strip().startswith('|')]
    if len(table_lines) < 3:
        raise RuntimeError(f'Cannot parse baseline markdown table from {report_path}')

    headers = [part.strip() for part in table_lines[0].strip('|').split('|')]
    rows = []
    for line in table_lines[2:]:
        parts = [part.strip() for part in line.strip('|').split('|')]
        if len(parts) != len(headers):
            continue
        row = dict(zip(headers, parts))
        rows.append(row)
    return rows


def build_reference_frontier(report_path: str):
    rows = parse_baseline_report(report_path)
    frontier = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = [parse_metric_cell(row[metric]) for row in rows if metric in row]
        values = [v for v in values if np.isfinite(v)]
        if not values:
            frontier[metric] = float('nan')
        elif direction == 'lower':
            frontier[metric] = min(values)
        else:
            frontier[metric] = max(values)

    for metric in SOFT_METRICS:
        values = [parse_metric_cell(row[metric]) for row in rows if metric in row]
        values = [v for v in values if np.isfinite(v)]
        frontier[metric] = min(values) if values else float('nan')

    return rows, frontier


def evaluate_config(config, dataset: NLSTDataset, device: torch.device):
    set_seed(SEED)
    n = len(dataset)
    indices = np.random.permutation(n)
    train_idx = indices[:TRAIN_SIZE]
    eval_idx = indices[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False)
    first_batch = next(iter(train_loader))

    local_diffusion_steps = int(config.get('diffusion_steps', DIFFUSION_STEPS))
    local_epochs = int(config.get('epochs', EPOCHS))
    outcome_loss_weight = float(config.get('outcome_loss_weight', 1.0))

    wrapper = CausalTabDiffWrapper(
        t_steps=first_batch['x'].shape[1],
        feature_dim=first_batch['x'].shape[2],
        diffusion_steps=local_diffusion_steps,
        use_noise_head=bool(config.get('use_noise_head', True)),
        outcome_loss_weight=outcome_loss_weight,
        use_trajectory_risk_head=True,
        sample_use_trajectory_risk=True,
        risk_smoothness_weight=config['risk_smoothness_weight'],
        cf_consistency_weight=config['cf_consistency_weight'],
        sample_model_score_weight=config['sample_model_score_weight'],
        sample_guidance_scale=config['sample_guidance_scale'],
        sample_guidance_schedule=config.get('sample_guidance_schedule', 'constant'),
        sample_guidance_power=float(config.get('sample_guidance_power', 2.0)),
        denoise_recon_weight=float(config.get('denoise_recon_weight', 0.0)),
        batch_moment_weight=float(config.get('batch_moment_weight', 0.0)),
    )

    train_start = time.perf_counter()
    wrapper.fit(train_loader, epochs=local_epochs, device=device, debug_mode=False)
    train_seconds = time.perf_counter() - train_start
    params_m = estimate_trainable_params_m(wrapper)

    all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for batch in eval_loader:
            alpha = batch['alpha_target'].to(device)
            fake_x, fake_y = wrapper.sample(batch_size=alpha.shape[0], alpha_target=alpha, device=device)
            real_x = build_semantic_real_x(batch, device)
            all_real_x.append(real_x)
            all_fake_x.append(fake_x)
            all_real_y.append(batch['y'].to(device))
            all_fake_y.append(fake_y)
            all_alpha.append(alpha)
    infer_seconds = time.perf_counter() - infer_start

    real_x_full = torch.cat(all_real_x, dim=0)
    fake_x_full = torch.cat(all_fake_x, dim=0)
    real_y_full = torch.cat(all_real_y, dim=0)
    fake_y_full = torch.cat(all_fake_y, dim=0)
    alpha_full = torch.cat(all_alpha, dim=0)

    metrics = compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)
    real_y_rate = float((real_y_full > 0.5).float().mean().item())
    fake_y_rate = float((fake_y_full > 0.5).float().mean().item())

    metrics['Params(M)'] = float(params_m)
    metrics['AvgInfer(ms/sample)'] = float(1000.0 * infer_seconds / max(1, real_x_full.shape[0]))

    formal_gate_pass = all(metrics[key] >= threshold for key, threshold in FORMAL_GATE.items())
    return {
        'config': config,
        'metrics': metrics,
        'formal_gate_pass': formal_gate_pass,
        'train_seconds': float(train_seconds),
        'infer_seconds': float(infer_seconds),
        'real_y_rate': real_y_rate,
        'fake_y_rate': fake_y_rate,
        'epochs': local_epochs,
        'diffusion_steps': local_diffusion_steps,
        'outcome_loss_weight': outcome_loss_weight,
    }


def compare_against_frontier(result, frontier):
    wins = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        value = result['metrics'][metric]
        ref = frontier[metric]
        if not np.isfinite(value) or not np.isfinite(ref):
            wins[metric] = False
        elif direction == 'lower':
            wins[metric] = value < ref
        else:
            wins[metric] = value > ref

    soft_status = {}
    for metric in SOFT_METRICS:
        value = result['metrics'][metric]
        ref = frontier[metric]
        if not np.isfinite(value) or not np.isfinite(ref):
            soft_status[metric] = 'unknown'
        elif value <= ref * 1.5:
            soft_status[metric] = 'acceptable'
        elif value <= ref * 2.0:
            soft_status[metric] = 'borderline'
        else:
            soft_status[metric] = 'weak'

    result['hard_wins'] = wins
    result['hard_win_count'] = int(sum(bool(v) for v in wins.values()))
    result['table_all_win'] = bool(all(wins.values()))
    result['soft_status'] = soft_status
    return result


def rank_key(result):
    metrics = result['metrics']
    soft_accept = sum(v == 'acceptable' for v in result['soft_status'].values())
    return (
        int(result['table_all_win']),
        result['hard_win_count'],
        int(result['formal_gate_pass']),
        soft_accept,
        metrics['TSTR_AUC'],
        metrics['TSTR_F1'],
        metrics['TSTR_PR_AUC'],
        metrics['TSTR_F1_RealPrev'],
        -metrics['ATE_Bias'],
        -metrics['Wasserstein'],
        -metrics['CMD'],
    )


def fmt(value):
    if not np.isfinite(value):
        return 'N/A'
    return f'{value:.6f}'


def main() -> None:
    os.environ.setdefault('DATASET_METADATA_PATH', os.path.join(PROJECT_ROOT, 'src', 'data', 'dataset_metadata_noleak.json'))
    os.environ.setdefault('ALPHA_TREATMENT_COLUMN', 'cigsmok')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    baseline_rows, frontier = build_reference_frontier(BASELINE_REPORT_PATH)
    if SEARCH_PROFILE not in SEARCH_PROFILES:
        raise ValueError(f'Unknown search profile: {SEARCH_PROFILE}')
    profile_configs = SEARCH_PROFILES[SEARCH_PROFILE]
    limit = int(os.environ.get('CAUSAL_TABDIFF_V2_SEARCH_LIMIT', str(len(profile_configs))))
    configs = profile_configs[: max(1, min(limit, len(profile_configs)))]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NLSTDataset(data_dir=os.path.join(PROJECT_ROOT, 'data'), debug_mode=False)

    results = []
    wall_start = time.perf_counter()
    for idx, config in enumerate(configs, start=1):
        print(f'=== [{idx}/{len(configs)}] {config["name"]} ===')
        evaluated = evaluate_config(config, dataset, device)
        compare_against_frontier(evaluated, frontier)
        results.append(evaluated)
        print(
            f"hard_wins={evaluated['hard_win_count']}/5 table_all_win={evaluated['table_all_win']} "
            f"gate={evaluated['formal_gate_pass']} auc={evaluated['metrics']['TSTR_AUC']:.6f} "
            f"f1={evaluated['metrics']['TSTR_F1']:.6f} ate={evaluated['metrics']['ATE_Bias']:.6f}"
        )
    total_seconds = time.perf_counter() - wall_start

    best = sorted(results, key=rank_key, reverse=True)[0]

    lines = []
    lines.append('# CausalTabDiff V2 Champion Search')
    lines.append('')
    lines.append(f'- Generated at: {datetime.now().isoformat()}')
    lines.append(f'- Device: {device}')
    lines.append(f'- Seed: {SEED}')
    lines.append(f'- Search profile: {SEARCH_PROFILE}')
    lines.append(f'- Train size: {TRAIN_SIZE}')
    lines.append(f'- Eval size: {EVAL_SIZE}')
    lines.append(f'- Batch size: {BATCH_SIZE}')
    lines.append(f'- Epochs: {EPOCHS}')
    lines.append(f'- Diffusion steps: {DIFFUSION_STEPS}')
    lines.append(f'- Treatment source: {os.environ.get("ALPHA_TREATMENT_COLUMN")}')
    lines.append(f'- Metadata path: {os.environ.get("DATASET_METADATA_PATH")}')
    lines.append(f'- Baseline report path: {BASELINE_REPORT_PATH}')
    lines.append(f'- Total wall time seconds: {total_seconds:.2f}')
    lines.append('')
    lines.append('## Instinct Anchors Used')
    lines.append('')
    lines.append('- Forced noleak metadata and real treatment `cigsmok`.')
    lines.append('- Search anchored around the noleak instinct defaults: early `0.50/0.50` blend for directional safety, later stronger candidate around `0.75 model score / 1.0 guidance`.')
    lines.append('- V2 search kept `use_trajectory_risk_head=True` and `sample_use_trajectory_risk=True` for all runs.')
    lines.append('')
    lines.append('## Baseline Frontier From Current Table')
    lines.append('')
    for row in baseline_rows:
        model_name = row.get('Model', 'Unknown')
        lines.append(
            f"- {model_name}: ATE_Bias={row.get('ATE_Bias', 'N/A')}, Wasserstein={row.get('Wasserstein', 'N/A')}, "
            f"CMD={row.get('CMD', 'N/A')}, TSTR_AUC={row.get('TSTR_AUC', 'N/A')}, TSTR_F1={row.get('TSTR_F1', 'N/A')}"
        )
    lines.append('')
    lines.append('- Frontier targets:')
    for metric in list(METRIC_DIRECTIONS.keys()) + SOFT_METRICS:
        lines.append(f'  - {metric}: {fmt(frontier[metric])}')
    lines.append('')
    lines.append('## Best Current Candidate')
    lines.append('')
    best_cfg = best['config']
    lines.append(f"- Name: {best_cfg['name']}")
    lines.append(f"- Hard wins: {best['hard_win_count']}/5")
    lines.append(f"- Table all-win: {best['table_all_win']}")
    lines.append(f"- Formal gate pass: {best['formal_gate_pass']}")
    lines.append(f"- Params(M): {fmt(best['metrics']['Params(M)'])} ({best['soft_status']['Params(M)']})")
    lines.append(f"- AvgInfer(ms/sample): {fmt(best['metrics']['AvgInfer(ms/sample)'])} ({best['soft_status']['AvgInfer(ms/sample)']})")
    for metric in ['ATE_Bias', 'Wasserstein', 'CMD', 'TSTR_AUC', 'TSTR_PR_AUC', 'TSTR_F1', 'TSTR_F1_RealPrev', 'TSTR_F1_FakePrev']:
        lines.append(f"- {metric}: {fmt(best['metrics'][metric])}")
    lines.append('')
    lines.append('## All Config Results')
    lines.append('')
    for result in sorted(results, key=rank_key, reverse=True):
        cfg = result['config']
        lines.append(f"### {cfg['name']}")
        lines.append('')
        lines.append(
            f"- config: use_noise_head={bool(cfg.get('use_noise_head', True))}, model_w={cfg['sample_model_score_weight']:.2f}, guidance={cfg['sample_guidance_scale']:.2f}, "
            f"smooth={cfg['risk_smoothness_weight']:.2f}, cf={cfg['cf_consistency_weight']:.2f}, "
            f"outcome_w={result['outcome_loss_weight']:.2f}, epochs={result['epochs']}, diffusion_steps={result['diffusion_steps']}, "
            f"guidance_schedule={cfg.get('sample_guidance_schedule', 'constant')}, guidance_power={float(cfg.get('sample_guidance_power', 2.0)):.1f}, "
            f"recon={float(cfg.get('denoise_recon_weight', 0.0)):.2f}, moment={float(cfg.get('batch_moment_weight', 0.0)):.2f}"
        )
        lines.append(f"- hard_wins: {result['hard_win_count']}/5")
        lines.append(f"- table_all_win: {result['table_all_win']}")
        lines.append(f"- formal_gate_pass: {result['formal_gate_pass']}")
        lines.append(
            f"- soft_status: Params(M)={result['soft_status']['Params(M)']}, "
            f"AvgInfer(ms/sample)={result['soft_status']['AvgInfer(ms/sample)']}"
        )
        lines.append(f"- train_seconds: {result['train_seconds']:.2f}")
        lines.append(f"- infer_seconds: {result['infer_seconds']:.2f}")
        lines.append(f"- real_y_rate: {result['real_y_rate']:.4f}")
        lines.append(f"- fake_y_rate: {result['fake_y_rate']:.4f}")
        for metric in ['ATE_Bias', 'Wasserstein', 'CMD', 'TSTR_AUC', 'TSTR_PR_AUC', 'TSTR_F1', 'TSTR_F1_RealPrev', 'TSTR_F1_FakePrev', 'Params(M)', 'AvgInfer(ms/sample)']:
            lines.append(f"- {metric}: {fmt(result['metrics'][metric])}")
        lines.append(
            '- hard_metric_wins: ' + ', '.join(
                f"{metric}={'Y' if result['hard_wins'][metric] else 'N'}" for metric in METRIC_DIRECTIONS
            )
        )
        lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    if best['table_all_win']:
        lines.append('- A single-seed V2 configuration already clears the current 5-baseline table frontier on all reported hard metrics.')
    else:
        lines.append('- No searched V2 configuration yet dominates all current table metrics simultaneously; continue with targeted search around the best candidate rather than broad expansion.')
    if best['formal_gate_pass']:
        lines.append('- The best searched configuration also passes the formal prevalence-aware gate, so it is eligible for the next multi-seed confirmation stage.')
    else:
        lines.append('- The best searched configuration still fails the formal prevalence-aware gate, so do not escalate to multi-seed until the gate is recovered.')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote V2 champion search report to {OUTPUT_PATH}')
    print(f'best_config={best_cfg["name"]}')
    print(f'best_hard_wins={best["hard_win_count"]}')
    print(f'best_table_all_win={best["table_all_win"]}')
    print(f'best_formal_gate_pass={best["formal_gate_pass"]}')


if __name__ == '__main__':
    main()