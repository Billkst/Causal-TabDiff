import re

def fix_script():
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
        content = f.read()

    old_string = "wrapper.fit(train_loader, epochs=eps, device=device, debug_mode=False)\n            metrics = compute_metrics(wrapper, dataset, eval_subset, device)"
    
    replacement = """
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
"""
    # Try regex if exact match fails
    content = re.sub(r'wrapper\.fit\([^)]+\)\s+metrics = compute_metrics\([^)]+\)', replacement, content)
    
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
        f.write(content)

fix_script()
