import re

def fix_script():
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
        content = f.read()

    replacement = """
            table_all_win = True
            cfg_lines.append("")
            
            report_lines.extend(cfg_lines)
            
            best_config = config_params['name']
"""
    # Replace the broken logic
    content = re.sub(r'table_all_win = True\n\s+for bk, bval in BASELINE_V2_FRONTIER\.items\(\):.*?(?=best_config = config_params\[\'name\'\])', replacement, content, flags=re.DOTALL)
    
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
        f.write(content)

fix_script()
