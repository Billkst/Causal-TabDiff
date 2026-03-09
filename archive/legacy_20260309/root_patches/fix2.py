with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
    lines = f.readlines()

out = []
skip = False
for line in lines:
    if line.strip().startswith('"v24_champion_final"'):
        skip = True
    if line.strip().startswith('def set_seed'):
        skip = False
        out.append("    'v24_champion_final': [\n")
        out.append("        {\n")
        out.append("            'name': 'v24_focal_internal_100',\n")
        out.append("            'use_noise_head': False,\n")
        out.append("            'sample_model_score_weight': 1.00,\n")
        out.append("            'sample_guidance_scale': 1.50,\n")
        out.append("            'sample_guidance_schedule': 'constant',\n")
        out.append("            'sample_guidance_power': 2.0,\n")
        out.append("            'risk_smoothness_weight': 0.10,\n")
        out.append("            'cf_consistency_weight': 0.05,\n")
        out.append("            'outcome_loss_weight': 0.35,\n")
        out.append("            'denoise_recon_weight': 0.03,\n")
        out.append("            'batch_moment_weight': 0.00,\n")
        out.append("            'epochs': 10,\n")
        out.append("            'diffusion_steps': 35,\n")
        out.append("        },\n")
        out.append("        {\n")
        out.append("            'name': 'v24_focal_blend_50',\n")
        out.append("            'use_noise_head': False,\n")
        out.append("            'sample_model_score_weight': 0.50,\n")
        out.append("            'sample_guidance_scale': 1.50,\n")
        out.append("            'sample_guidance_schedule': 'constant',\n")
        out.append("            'sample_guidance_power': 2.0,\n")
        out.append("            'risk_smoothness_weight': 0.10,\n")
        out.append("            'cf_consistency_weight': 0.05,\n")
        out.append("            'outcome_loss_weight': 0.35,\n")
        out.append("            'denoise_recon_weight': 0.03,\n")
        out.append("            'batch_moment_weight': 0.00,\n")
        out.append("            'epochs': 10,\n")
        out.append("            'diffusion_steps': 35,\n")
        out.append("        }\n")
        out.append("    ]\n")
        out.append("}\n\n")
    if not skip:
        out.append(line)

with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
    f.writelines(out)
