with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
    text = f.read()

import re
old_text = """    dataset = NLSTDataset("data")
    
    train_indices = list(range(TRAIN_SIZE))
    eval_indices = list(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
    
    dataset.data = dataset.data.iloc[train_indices + eval_indices].reset_index(drop=True)
    dataset.time_series = [dataset.time_series[i] for i in train_indices + eval_indices]"""

new_text = """    dataset = NLSTDataset("data")
    
    train_indices = list(range(TRAIN_SIZE))
    eval_indices = list(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
    if hasattr(dataset, 'prsn_df'):
        dataset.prsn_df = dataset.prsn_df.iloc[train_indices + eval_indices].reset_index(drop=True)
    else:
        dataset.data = dataset.data.iloc[train_indices + eval_indices].reset_index(drop=True)
    dataset.time_series = [dataset.time_series[i] for i in train_indices + eval_indices]"""

text = text.replace(old_text, new_text)
with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
    f.write(text)
