import json
import os
import pandas as pd
import numpy as np

def generate_metadata():
    """
    Globally scans the dataset schema from the raw CSV level
    and serializes an immutable metadata JSON for all downstream pipelines to read transparently.
    It considers any column with <= 15 unique values as categorical, regardless of string or numeric.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(data_dir, '..', '..', 'data'))
    
    prsn_path = os.path.join(base_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_prsn_idc_20210527.csv')
    
    # 彻底禁止一切 Mock Data 生成与 Try-Catch，直接读取原生唯一宽表
    merged_df = pd.read_csv(prsn_path)
    
    if 'cancyr' not in merged_df.columns:
        raise KeyError("CRITICAL ERROR: 'cancyr' not found in prsn_df during metadata generation!")
        
    merged_df['cancyr'] = merged_df['cancyr'].fillna(0).astype(int)

    y_col = 'cancyr'
    
    # Analyze columns
    continuous_cols = []
    categorical_cols = []
    
    # Note: excluding raw IDs, unstructured strings or post-treatment outcomes that leak the target prediction
    exclude_cols = ['pid', 'dataset_version', 'candx_days', 'canc_free_days', 'canc_rpt_link', 'de_stag_7thed', 'de_stag', 'de_grade', 'de_type', y_col]
    
    for col in merged_df.columns:
        if col in exclude_cols: 
            continue
            
        unique_vals = merged_df[col].dropna().unique()
        num_unique = len(unique_vals)
        
        # Academic rule: <= 15 unique -> Categorical
        if num_unique <= 15:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)
            
    meta = {
        'columns': [],
        'continuous': [],
        'categorical': [],
        'y_col': {
            'name': y_col,
            'classes': int(merged_df[y_col].nunique()),
            'analog_bits': 0
        }
    }
    
    for c in continuous_cols:
        meta['continuous'].append({'name': c, 'dim': 1})
        
    for c in categorical_cols:
        unique_vals = sorted([int(x) if float(x).is_integer() else x for x in merged_df[c].dropna().unique()])
        k_classes = len(unique_vals)
        max_idx = k_classes - 1
        m_bits = int(np.ceil(np.log2(max_idx + 1e-5))) if max_idx > 0 else 1
        
        meta['categorical'].append({
            'name': c, 
            'classes': k_classes, 
            'analog_bits': m_bits,
            'val_to_idx': {str(v): i for i, v in enumerate(unique_vals)},
            'idx_to_val': {str(i): v for i, v in enumerate(unique_vals)}
        })
        
    for col in merged_df.columns:
        if col in exclude_cols: continue
        if col in continuous_cols:
            meta['columns'].append({'name': col, 'type': 'continuous', 'dim': 1})
        else:
            # find analog bits
            meta_cat = next(item for item in meta['categorical'] if item['name'] == col)
            meta['columns'].append({'name': col, 'type': 'categorical', 'dim': meta_cat['analog_bits']})
            
    meta_path = os.path.join(data_dir, 'dataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"Static global schema generated at {meta_path}.")
    print("Categorical features detected:", categorical_cols)
    print("Continuous features detected:", continuous_cols)

if __name__ == "__main__":
    generate_metadata()
