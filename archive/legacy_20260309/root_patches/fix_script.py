with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
    text = f.read()

import re
# Revert my bad sed commands
text = re.sub(r'"v24_champion_final".*?}

def set_seed', 
'''"v24_champion_final": [
        {
            "name": "v24_focal_internal_100",
            "use_noise_head": False,
            "sample_model_score_weight": 1.00,
            "sample_guidance_scale": 1.50,
            "sample_guidance_schedule": "constant",
            "sample_guidance_power": 2.0,
            "risk_smoothness_weight": 0.10,
            "cf_consistency_weight": 0.05,
            "outcome_loss_weight": 0.35,
            "denoise_recon_weight": 0.03,
            "batch_moment_weight": 0.00,
            "epochs": 10,
            "diffusion_steps": 35,
        },
        {
            "name": "v24_focal_internal_50",
            "use_noise_head": False,
            "sample_model_score_weight": 0.50,
            "sample_guidance_scale": 1.50,
            "sample_guidance_schedule": "constant",
            "sample_guidance_power": 2.0,
            "risk_smoothness_weight": 0.10,
            "cf_consistency_weight": 0.05,
            "outcome_loss_weight": 0.35,
            "denoise_recon_weight": 0.03,
            "batch_moment_weight": 0.00,
            "epochs": 10,
            "diffusion_steps": 35,
        }
    ]
}

def set_seed''', text, flags=re.DOTALL)

with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
    f.write(text)
