with open('run_baselines.py', 'r') as f:
    text = f.read()

import re
if "'Causal-TabDiff (Ours)': CausalTabDiffWrapper" not in text:
    text = text.replace("'TSDiff (ICLR 23)': TSDiffWrapper", "'TSDiff (ICLR 23)': TSDiffWrapper,\n        'Causal-TabDiff (Ours)': CausalTabDiffWrapper")
    
if "from src.baselines.wrappers import" not in text or "CausalTabDiffWrapper" not in text:
    text = text.replace("from src.baselines.wrappers import", "from src.baselines.wrappers import CausalTabDiffWrapper, ")

with open('run_baselines.py', 'w') as f:
    f.write(text)
