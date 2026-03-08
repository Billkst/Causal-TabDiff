with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
    text = f.read()

import re
# Just strip out the entirely badly formatted block
text = re.sub(r'best_config = config_params\[\'name\'\]\n                        best_hard_wins = hard_wins(.*?)(?=        except Exception as e:)', r'', text, flags=re.DOTALL)

with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
    f.write(text)
