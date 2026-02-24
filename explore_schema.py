import sys
import os
import torch
import numpy as np

# ensure src is in path
sys.path.insert(0, os.path.abspath('.'))

from src.data.data_module import get_dataloader

def infer_schema(dataloader):
    # Collect all data to infer properly
    all_x = []
    all_t = []
    all_y = []
    
    for batch in dataloader:
        all_x.append(batch['x'].numpy())
        all_t.append(batch['alpha_target'].numpy())
        all_y.append(batch['y'].numpy())
        
    X = np.concatenate(all_x, axis=0) # [B, T_steps, D]
    T_mat = np.concatenate(all_t, axis=0) # [B, 1]
    Y_mat = np.concatenate(all_y, axis=0) # [B, 1]
    
    # Flatten X over time for TabSyn
    b_size = X.shape[0]
    X_flat = X.reshape(b_size, -1)
    
    joint_data = np.concatenate([X_flat, T_mat, Y_mat], axis=1)
    D_total = joint_data.shape[1]
    
    cat_cols = []
    cont_cols = []
    categories = []
    
    print("-" * 50)
    print(f"Total Joint Features (Flattened X + T + Y): {D_total}")
    for col_idx in range(D_total):
        col_data = joint_data[:, col_idx]
        unique_vals = np.unique(col_data)
        num_unique = len(unique_vals)
        
        # Heuristic: if <= 15 unique values or all values are integers, treat as categorical
        is_integer_like = np.all(np.mod(unique_vals, 1) == 0)
        # Even if they are floats like -1.0, 1.0 (analog bits) they are discrete
        
        if num_unique <= 15 or is_integer_like:
            cat_cols.append(col_idx)
            categories.append(num_unique)
            print(f"Col {col_idx}: Categorical | Unique Vals: {num_unique} | Ex: {unique_vals[:5]}")
        else:
            cont_cols.append(col_idx)
            print(f"Col {col_idx}: Continuous  | Unique Vals: {num_unique} | Range: [{np.min(col_data):.2f}, {np.max(col_data):.2f}]")

    print("-" * 50)
    print(f"Continuous Columns : {cont_cols}")
    print(f"Categorical Columns: {cat_cols}")
    print(f"Categories Count   : {categories}")
    
if __name__ == "__main__":
    dataloader = get_dataloader('data', batch_size=256, debug_mode=True)
    infer_schema(dataloader)
