"""
Positional Leverage Analyzer (Canonical Positions)
==================================================
Calculates the empirical leverage of core grappling positions:
    Leverage Score = Total Transitions (n) * Positional Equity Swing (Delta_V)

Features:
---------
- Accurately counts all bouts reached (including St -> Guard pulls and -> End).
- Directional equity cliff: Advancing to top pins vs conceding to bottom pins.
- Standing (St) is excluded from the final rankings.
- Conflates _T and _B into canonical parent positions (OG, CG, HG, SC, FM, BM, 50-50).

Usage:
------
    python positional_leverage.py [csv_path]
"""

import sys
import os
import pandas as pd
import numpy as np
import bjj_core

def conflate_base(state_name):
    """Extracts base position name (e.g. OG_T -> OG, SC_B -> SC)."""
    s = str(state_name).strip()
    for suffix in ['_T', '_B']:
        if s.endswith(suffix):
            return s[:-2]
    return s

def compute_positional_leverage(csv_path="Matches26.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Target file '{csv_path}' was not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'\s*/\s*', '/', regex=True)

    # 1. Expand dual athlete perspective with directional split mode preserved
    dual_df = bjj_core.expand_dual_athlete_transitions(df, split_mode=True)

    # Clean strings and tag canonical parent states
    dual_df['Source_Clean'] = dual_df['Source_Resolved'].astype(str).str.strip()
    dual_df['Target_Clean'] = dual_df['Target_Resolved'].astype(str).str.strip()
    dual_df['Source_Canonical'] = dual_df['Source_Clean'].apply(conflate_base)
    dual_df['Target_Canonical'] = dual_df['Target_Clean'].apply(conflate_base)

    # 2. Compute empirical Win Equity V(S) for all directional states (_T vs _B)
    state_bouts = dual_df.groupby(['Match ID', 'Athlete_Role', 'Source_Clean'])['Is_Winner'].first().reset_index()
    equity_map = state_bouts.groupby('Source_Clean')['Is_Winner'].mean().to_dict()

    # 3. Ground truth traffic: Tally total engagements and unique bouts BEFORE removing St
    # (Divide transition count by 2 to account for mirrored dual perspectives)
    total_engagements = (dual_df.groupby('Source_Canonical').size() / 2).astype(int).to_dict()
    total_bouts = dual_df.groupby('Source_Canonical')['Match ID'].nunique().to_dict()

    # 4. Filter transitions for calculating the ground exit cliff:
    # Exclude self-loops (e.g., OG -> OG adjustments), standing (St), and terminal match ends
    ignored_destinations = {'St', 'End', 'nan', '', 'None'}
    exit_mask = (~dual_df['Target_Canonical'].isin(ignored_destinations)) & \
                (dual_df['Source_Canonical'] != dual_df['Target_Canonical'])
    ground_exits = dual_df[exit_mask].copy()
    ground_exits['Target_Equity'] = ground_exits['Target_Clean'].map(equity_map).fillna(0.50)

    # 5. Calculate Leverage Metrics per Canonical Position
    records = []
    # Evaluate positions, excluding Standing ('St')
    all_positions = [p for p in set(dual_df['Source_Canonical']) if p not in ignored_destinations]

    for pos in sorted(all_positions):
        n_transitions = total_engagements.get(pos, 0)
        n_bouts = total_bouts.get(pos, 0)
        
        pos_exits = ground_exits[ground_exits['Source_Canonical'] == pos]

        if not pos_exits.empty:
            v_max = pos_exits['Target_Equity'].max()
            v_min = pos_exits['Target_Equity'].min()
            delta_v = round(v_max - v_min, 3)
        else:
            v_max, v_min, delta_v = 0.50, 0.50, 0.000

        # Leverage Score = Transition Volume * Delta V Cliff
        leverage_score = round(n_transitions * delta_v, 1)

        records.append({
            'Position': pos,
            'Advancement Equity (Max)': round(v_max, 2),
            'Concession Equity (Min)': round(v_min, 2),
            'Delta_V (Cliff)': delta_v,
            'Transitions (n)': n_transitions,
            'Bouts Reached': n_bouts,
            'Leverage_Score': leverage_score
        })

    lev_df = pd.DataFrame(records).sort_values(by='Leverage_Score', ascending=False).reset_index(drop=True)
    lev_df.index += 1

    print("\n============================== POSITIONAL LEVERAGE RANKING ==============================")
    print(lev_df.to_string())
    print("=========================================================================================\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Matches26.csv"
    compute_positional_leverage(path)