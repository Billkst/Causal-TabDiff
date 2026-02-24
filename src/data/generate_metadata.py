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
    canc_path = os.path.join(base_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_canc_idc_20210527.csv')
    screen_path = os.path.join(base_dir, 'nlst.780.idc.delivery.052821', 'nlst_780_screen_idc_20210527.csv')
    
    try:
        prsn_df = pd.read_csv(prsn_path)
        canc_df = pd.read_csv(canc_path)
        
        # We simply need a unified schema. Let's merge PRSN and CANC
        merged_df = pd.merge(prsn_df, canc_df[['pid', 'cancyr']], on='pid', how='left')
        merged_df['cancyr'] = merged_df['cancyr'].fillna(0).astype(int)
        
        # Add basic dummy age if missing
        if 'age' not in merged_df.columns:
            merged_df['age'] = np.random.randint(50, 80, size=len(merged_df))
            
    except Exception as e:
        print(f"Real data not found: {e}. Generating comprehensive mock data.")
        size = 1000
        merged_df = pd.DataFrame({
            'pid': range(size),
            'age': np.random.normal(60, 5, size),
            'bmi': np.random.normal(25, 3, size),
            'gender': np.random.choice([1, 2], size),
            'smoke_hist': np.random.choice([0, 1, 2, 3], size),
            'screen_group': np.random.choice([1, 2], size),
            'cancyr': np.random.choice([0, 1], size, p=[0.9, 0.1])
        })

    y_col = 'cancyr'
    
    # Analyze columns
    continuous_cols = []
    categorical_cols = []
    
    for col in merged_df.columns:
        if col in ['pid', y_col]: 
            continue # Exclude ID and Label from X features
            
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
        if col in ['pid', y_col]: continue
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
