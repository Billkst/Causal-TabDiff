import re

def fix_script():
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
        content = f.read()

    replacement = """
            table_all_win = True
            cfg_lines.append("")
            
            report_lines.extend(cfg_lines)
            
            best_config = config_params['name']
        except Exception as e:
            print(f"Error evaluating config {config_params['name']}: {e}")
"""
    # Simply catch the exact string
    content = re.sub(r'\n            best_config = config_params\[\'name\'\](\s+except Exception as e:)?', replacement, content)
    
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
        f.write(content)

fix_script()
