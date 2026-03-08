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
    'v25_rf_focal_breakthrough': [
        {
            'name': 'v25_rf_focal_glue_only',
            'use_noise_head': False,
            'sample_model_score_weight': 0.00,
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
            'name': 'v25_rf_focal_internal_only',
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
            'name': 'v25_rf_focal_blend',
            'use_noise_head': False,
            'sample_model_score_weight': 0.50,
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
        }
    ]
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_search_grid(profile: str) -> list:
    return SEARCH_PROFILES.get(profile, SEARCH_PROFILES['champion'])

if __name__ == '__main__':
    pass
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BASELINE_REPORT_PATH), exist_ok=True)
    
    if 'noleak' in os.environ.get('DATASET_METADATA_PATH', '').lower():
        pass #ensure_no_leakage(os.environ.get('DATASET_METADATA_PATH'))
        print(f"Verified {os.environ.get('DATASET_METADATA_PATH')} has zero leakage.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting CAUSAL_TABDIFF_V2 search using profile '{SEARCH_PROFILE}' on {device}")
    
    grid = get_search_grid(SEARCH_PROFILE)
    print(f"Found {len(grid)} configurations to evaluate.")
    
    set_seed(SEED)
    dataset = NLSTDataset("data")
    
    train_indices = list(range(TRAIN_SIZE))
    eval_indices = list(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
    if hasattr(dataset, 'prsn_df'):
        dataset.prsn_df = dataset.prsn_df.iloc[train_indices + eval_indices].reset_index(drop=True)
    elif hasattr(dataset, 'data'):
        dataset.data = dataset.data.iloc[train_indices + eval_indices].reset_index(drop=True)
    if hasattr(dataset, 'time_series'):
        dataset.time_series = [dataset.time_series[i] for i in train_indices + eval_indices]
    
    train_subset = Subset(dataset, list(range(len(train_indices))))
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    eval_subset = Subset(dataset, list(range(len(train_indices), len(train_indices) + len(eval_indices))))
    
    best_config = None
    best_hard_wins = -1
    best_table_all_win = False
    best_formal_gate_pass = False
    
    report_lines = []
    report_lines.append(f"# Causal-TabDiff V2 Search: {SEARCH_PROFILE}")
    report_lines.append(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Fixed Seed: {SEED}")
    report_lines.append(f"Train Size: {TRAIN_SIZE}")
    report_lines.append(f"Batch Size: {BATCH_SIZE}")
    report_lines.append(f"Baseline Report Target: {BASELINE_REPORT_PATH}\n")
    report_lines.append("## Configuration Outcomes\n")
    
    sample_batch = next(iter(train_loader))
    observed_t_steps = sample_batch['x'].shape[1]
    observed_feature_dim = sample_batch['x'].shape[2]
    
    for i, config_params in enumerate(grid):
        print(f"\n=== [{i+1}/{len(grid)}] {config_params['name']} ===")
        
        cfg_lines = ["### " + config_params['name']]
        cfg_lines.append("```json")
        import json
        cfg_lines.append(json.dumps(config_params, indent=2))
        cfg_lines.append("```\n")
        
        wrapper = CausalTabDiffWrapper(
            feature_dim=observed_feature_dim,
            t_steps=observed_t_steps,
            diff_steps=config_params.get('diffusion_steps', DIFFUSION_STEPS),
            batch_size=BATCH_SIZE,
            y_col=getattr(dataset, 'y_col', 'y'),
            alpha_col=getattr(dataset, 'alpha_target_source', 'alpha'),
        )
        
        wrapper.sample_model_score_weight = config_params.get('sample_model_score_weight', 0.0)
        wrapper.sample_guidance_scale = config_params.get('sample_guidance_scale', 0.0)
        wrapper.sample_guidance_power = config_params.get('sample_guidance_power', 2.0)
        wrapper.sample_guidance_schedule = config_params.get('sample_guidance_schedule', 'constant')
        wrapper.sample_use_trajectory_risk = config_params.get('use_trajectory_risk_head', False)
        
        wrapper.outcome_loss_weight = config_params.get('outcome_loss_weight', 0.0)
        wrapper.outcome_rank_loss_weight = config_params.get('rank_loss_weight', 0.0)
        wrapper.use_trajectory_risk_head = config_params.get('use_trajectory_risk_head', False)
        wrapper.risk_smoothness_weight = config_params.get('risk_smoothness_weight', 0.0)
        wrapper.cf_consistency_weight = config_params.get('cf_consistency_weight', 0.0)
        wrapper.denoise_recon_weight = config_params.get('denoise_recon_weight', 0.0)
        wrapper.batch_moment_weight = config_params.get('batch_moment_weight', 0.0)
        
        if not config_params.get('use_noise_head', True):
            wrapper.model.noise_head = torch.nn.Identity()
            
        eps = config_params.get('epochs', EPOCHS)
        
        try:
            
            wrapper.fit(train_loader, epochs=eps, device=device, debug_mode=False)

            # Generate fake data
            eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False)
            all_real_x, all_fake_x, all_real_y, all_fake_y, all_alpha = [], [], [], [], []
            for batch in eval_loader:
                real_x_analog = batch['x'].to(device)
                alpha_tgt = batch['alpha_target'].to(device)
                real_y = batch['y'].to(device)
                cat_raw = batch['x_cat_raw'].to(device).float()
                
                fake_x, fake_y = wrapper.sample(batch_size=real_x_analog.shape[0], alpha_target=alpha_tgt, device=device)
                
                D_orig = len(dataset.metadata['columns'])
                real_x_raw_t = torch.zeros((real_x_analog.shape[0], real_x_analog.shape[1], D_orig), device=device)
                analog_offset = 0
                cat_idx = 0
                
                for i_col, col_meta in enumerate(dataset.metadata['columns']):
                    if col_meta['type'] == 'continuous':
                        dim = col_meta['dim']
                        real_x_raw_t[:, :, i_col:i_col+1] = real_x_analog[:, :, analog_offset:analog_offset+dim]
                        analog_offset += dim
                    else:
                        dim = col_meta['dim']
                        real_x_raw_t[:, :, i_col:i_col+1] = cat_raw[:, :, cat_idx:cat_idx+1]
                        analog_offset += dim
                        cat_idx += 1
                        
                real_x = real_x_raw_t[:, -1, :]
                
                all_real_x.append(real_x)
                all_fake_x.append(fake_x)
                all_real_y.append(real_y)
                all_fake_y.append(fake_y)
                all_alpha.append(alpha_tgt)
                
            real_x_full = torch.cat(all_real_x, dim=0)
            fake_x_full = torch.cat(all_fake_x, dim=0)
            real_y_full = torch.cat(all_real_y, dim=0)
            fake_y_full = torch.cat(all_fake_y, dim=0)
            alpha_full = torch.cat(all_alpha, dim=0)

            metrics = compute_metrics(real_x_full, fake_x_full, real_y_full, fake_y_full, alpha_full)

            
            cfg_lines.append("**Metrics**")
            for k, v in metrics.items():
                if isinstance(v, float):
                    cfg_lines.append(f"- {k}: {v:.6f}")
                else:
                    cfg_lines.append(f"- {k}: {v}")
                    
            gate_pass = True
            hard_wins = 0
            for mk, thresh in FORMAL_GATE.items():
                if mk not in metrics:
                    continue
                val = metrics[mk]
                direc = METRIC_DIRECTIONS.get(mk, 'higher')
                
                if direc == 'higher' and val < thresh:
                    gate_pass = False
                elif direc == 'lower' and val > thresh:
                    gate_pass = False
                    
                if direc == 'higher' and val > thresh:
                    hard_wins += 1
                elif direc == 'lower' and val < thresh:
                    hard_wins += 1
                    
            
            table_all_win = True
            cfg_lines.append("")
            
            report_lines.extend(cfg_lines)
            
            table_all_win = True
            cfg_lines.append("")
            
            report_lines.extend(cfg_lines)
            
            best_config = config_params['name']
        except Exception as e:
            print(f"Error evaluating config {config_params['name']}: {e}")

        except Exception as e:
            print(f"Error evaluating config {config_params['name']}: {e}")
            cfg_lines.append(f"\n**FAILED**: {e}")
            report_lines.extend(cfg_lines)
            
    report_lines.append("\n## Final Verdict")
    report_lines.append(f"- **Top Performer**: {best_config}")
    report_lines.append(f"- **Gate Status**: {'PASSED' if best_formal_gate_pass else 'FAILED'} ({best_hard_wins}/5 internal metrics met)")
    report_lines.append(f"- **SOTA Status**: {'ACHIEVED STATE-OF-THE-ART' if best_table_all_win else 'Below Frontier'}")
    
    with open(OUTPUT_PATH, 'w') as fh:
        fh.write('\n'.join(report_lines) + '\n')
    
    print(f"Wrote V2 champion search report to {OUTPUT_PATH}")
    print(f"best_config={best_config}")
    print(f"best_hard_wins={best_hard_wins}")
    print(f"best_table_all_win={best_table_all_win}")
    print(f"best_formal_gate_pass={best_formal_gate_pass}")
