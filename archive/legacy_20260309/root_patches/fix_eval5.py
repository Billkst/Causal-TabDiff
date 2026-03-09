def fix_script():
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'r') as f:
        lines = f.readlines()
        
    out = []
    for line in lines:
        if "best_hard_wins = hard_wins" in line:
            line = line.replace("    best_hard_wins = hard_wins", "            best_hard_wins = hard_wins")
        if "best_formal_gate_pass = gate_pass" in line:
            line = line.replace("    best_formal_gate_pass = gate_pass", "            best_formal_gate_pass = gate_pass")
        if "best_table_all_win = table_all_win" in line:
            line = line.replace("    best_table_all_win = table_all_win", "            best_table_all_win = table_all_win")
        out.append(line)
        
    with open('scripts/search_causal_tabdiff_v2_champion.py', 'w') as f:
        f.writelines(out)

fix_script()
