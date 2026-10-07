import sys
import pandas as pd
import numpy as np
import bjj_core

def compute_positional_leverage(csv_path="Matches26.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'\s*/\s*', '/', regex=True)
    
    # Expand to dual perspective (Athlete A and B)
    dual_df = bjj_core.expand_dual_athlete_transitions(df)
    
    # 1. Compute empirical win equity V(S) for each state
    state_bouts = dual_df.groupby(['Match ID', 'Athlete_Role', 'Source_Resolved']).first().reset_index()
    state_equity = state_bouts.groupby('Source_Resolved')['Win'].agg(
        Total_Bouts='count',
        Wins='sum'
    )
    state_equity['Win_Equity'] = (state_equity['Wins'] / state_equity['Total_Bouts']).round(3)
    
    # 2. Compute transition outcomes and equity delta
    dual_df = dual_df.merge(
        state_equity[['Win_Equity']], 
        left_on='Target_Resolved', 
        right_index=True, 
        how='left'
    ).rename(columns={'Win_Equity': 'Target_Equity'}).fillna({'Target_Equity': 0.5})
    
    # 3. Calculate Leverage per State
    records = []
    for state, group in dual_df.groupby('Source_Resolved'):
        if state in ['End']:
            continue
        n_transitions = len(group)
        n_bouts = group['Match ID'].nunique()
        v_current = state_equity.loc[state, 'Win_Equity'] if state in state_equity.index else 0.5
        
        # Best exit equity vs worst exit equity
        v_max = group['Target_Equity'].max()
        v_min = group['Target_Equity'].min()
        delta_v = v_max - v_min
        
        # Expected Win Impact (EWI) = Volume * Delta V
        leverage_score = n_transitions * delta_v
        
        records.append({
            'State': state,
            'Current_V': v_current,
            'Exit_V_Max': round(v_max, 2),
            'Exit_V_Min': round(v_min, 2),
            'Delta_V_Cliff': round(delta_v, 2),
            'Transitions': n_transitions,
            'Bouts': n_bouts,
            'Leverage_Score': round(leverage_score, 1)
        })
        
    lev_df = pd.DataFrame(records).sort_values(by='Leverage_Score', ascending=False).reset_index(drop=True)
    lev_df.index += 1
    
    print("\n========================= POSITIONAL LEVERAGE RANKING =========================")
    print(lev_df.to_string())
    print("===============================================================================\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Matches26.csv"
    compute_positional_leverage(path)