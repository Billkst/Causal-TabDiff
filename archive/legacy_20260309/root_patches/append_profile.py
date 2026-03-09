with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
    text = f.read()

# Add comma before new_profile
new_profile = """,
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
            'name': 'v25_rf_focal_blend_50',
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
}"""

import re
text = re.sub(r"\]\n\}\n\ndef set_seed", "]" + new_profile + '\n\ndef set_seed', text)

with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
    f.write(text)
