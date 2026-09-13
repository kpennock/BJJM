"""
bjj_core.py
===========
Shared data ingestion, perspective resolution, and filtering pipeline
for BJJ match analysis tools (markov.py and report.py).
"""

import os
import sys
import pandas as pd
import numpy as np

SPLIT_STATES = {'CG', 'OG', 'FM', 'BM', 'HG', 'SC'}

def parse_timestamp_duration(ts_str):
    """Calculates duration in seconds from 'm:ss/m:ss' timestamps."""
    if pd.isna(ts_str) or '/' not in str(ts_str):
        return 0.0
    parts = str(ts_str).strip().split('/')
    def to_seconds(t_val):
        m, s = map(float, t_val.strip().split(':'))
        return m * 60 + s
    try:
        return abs(to_seconds(parts[0]) - to_seconds(parts[1]))
    except Exception:
        return 0.0

def parse_adv_pen(adv_str):
    """Parses 'a/b', '/-1', or '1/-1' into integer tuples (A, B)."""
    if pd.isna(adv_str) or '/' not in str(adv_str):
        return (0, 0)
    parts = str(adv_str).strip().split('/')
    a = int(parts[0]) if parts[0].strip() not in ['', '+', '-'] else 0
    b = int(parts[1]) if len(parts) > 1 and parts[1].strip() not in ['', '+', '-'] else 0
    return (a, b)

def resolve_perspective_state(state_name, athlete_id, dom_athlete, split_mode=False):
    """Maps symmetric/asymmetric states to Top (_T) or Bottom (_B)."""
    raw_state = str(state_name).strip()
    if not split_mode or raw_state not in SPLIT_STATES:
        return raw_state

    dom_norm = str(dom_athlete).strip().lower()
    ath_norm = str(athlete_id).strip().lower()

    if dom_norm in ['none', 'neutral', 'n', 'nan', '']:
        return raw_state

    return f"{raw_state}_T" if ath_norm == dom_norm else f"{raw_state}_B"

def parse_onesided_pairs(pairs_str):
    """Parses string '1/A,2/A,3/B' into a lookup dict: {'1': 'A', '2': 'A', '3': 'B'}."""
    if not pairs_str:
        return {}
    mapping = {}
    tokens = [t.strip() for t in str(pairs_str).split(',') if t.strip()]
    for token in tokens:
        if '/' in token:
            m_id, comp = token.split('/', 1)
            mapping[m_id.strip()] = comp.strip().upper()
    return mapping

def expand_dual_athlete_transitions(df, split_mode=False, onesided_map=None):
    """
    Expands transitions into athlete-specific trajectories with lookahead
    dominance resolution and point/submission metrics.
    """
    processed = []
    df = df.reset_index(drop=True)
    has_win = 'Win' in df.columns
    is_onesided = bool(onesided_map)

    match_col = next((c for c in ['Match ID', 'MatchID', 'Match_ID', 'Match'] if c in df.columns), None)

    for i in range(len(df)):
        row = df.iloc[i]
        curr_m_id = str(row.get(match_col, '')).strip() if match_col else ""

        if is_onesided:
            if curr_m_id not in onesided_map:
                continue
            allowed_roles = [onesided_map[curr_m_id]]
        else:
            allowed_roles = ['A', 'B']

        src_raw = str(row.get('Primary State', '')).strip()
        tgt_raw = str(row.get('Next State', '')).strip()
        src_dom = str(row.get('Dominant Athlete', 'None')).strip()
        tgt_dom = src_dom

        action = str(row.get('Exit Action', '')).strip().lower()
        instigator = str(row.get('Action Instigator', '')).strip().upper()
        subaction = str(row.get('Subaction', '')).strip()

        pts_val = row.get('Points', 0)
        try:
            points = int(pts_val) if not pd.isna(pts_val) and pts_val != '' else 0
        except Exception:
            points = 0

        # Lookahead within same match to determine consolidated target dominance
        if i + 1 < len(df):
            next_row = df.iloc[i + 1]
            next_m_id = str(next_row.get(match_col, '')).strip() if match_col else ""
            next_pri_state = str(next_row.get('Primary State', '')).strip()
            next_dom = str(next_row.get('Dominant Athlete', 'None')).strip()

            if (curr_m_id == next_m_id) and (tgt_raw == next_pri_state):
                if next_dom.lower() not in ['none', 'neutral', 'n', 'nan', '']:
                    tgt_dom = next_dom

        # Action fallback for sweeps/passes
        if tgt_dom == src_dom and action in ['sw', 'sweep', 'gpass', 'pass', 'sc']:
            if instigator in ['A', 'B']:
                tgt_dom = instigator

        name_a = row.get('Name A', 'Athlete A')
        name_b = row.get('Name B', 'Athlete B')
        base = row.to_dict()
        win_val = str(row.get('Win', '')).strip().lower()

        # Athlete A perspective
        if 'A' in allowed_roles:
            src_a = resolve_perspective_state(src_raw, 'A', src_dom, split_mode=split_mode)
            tgt_a = resolve_perspective_state(tgt_raw, 'A', tgt_dom, split_mode=split_mode)
            rec_a = base.copy()
            rec_a.update({
                'Focal_Athlete': name_a,
                'Athlete_Role': 'A',
                'Source_Resolved': src_a,
                'Target_Resolved': tgt_a,
                'Scored_Points': points if instigator == 'A' else 0,
                'Executed_Sub': (action in ['sub', 'submission'] and instigator == 'A'),
                'Is_Winner': (win_val in ['a', str(name_a).lower()]) if has_win else True
            })
            processed.append(rec_a)

        # Athlete B perspective
        if 'B' in allowed_roles:
            src_b = resolve_perspective_state(src_raw, 'B', src_dom, split_mode=split_mode)
            tgt_b = resolve_perspective_state(tgt_raw, 'B', tgt_dom, split_mode=split_mode)
            rec_b = base.copy()
            rec_b.update({
                'Focal_Athlete': name_b,
                'Athlete_Role': 'B',
                'Source_Resolved': src_b,
                'Target_Resolved': tgt_b,
                'Scored_Points': points if instigator == 'B' else 0,
                'Executed_Sub': (action in ['sub', 'submission'] and instigator == 'B'),
                'Is_Winner': (win_val in ['b', str(name_b).lower()]) if has_win else True
            })
            processed.append(rec_b)

    return pd.DataFrame(processed)

def filter_dataset(df, args):
    """Filters expanded DataFrame based on CLI options."""
    filtered = df.copy()

    if getattr(args, 'match_id', None) is not None:
        m_col = next((c for c in ['Match ID', 'MatchID', 'Match_ID', 'Match'] if c in filtered.columns), None)
        if m_col:
            filtered = filtered[filtered[m_col].astype(str) == str(args.match_id)]

    if getattr(args, 'name', None):
        name_cols = [c for c in ['Name A', 'Name B', 'Focal_Athlete'] if c in filtered.columns]
        if name_cols:
            mask = False
            for c in name_cols:
                mask = mask | (filtered[c].astype(str).str.lower() == args.name.lower())
            filtered = filtered[mask]

    if getattr(args, 'belt', None) and 'Belt' in filtered.columns:
        filtered = filtered[filtered['Belt'].astype(str).str.lower() == args.belt.lower()]

    if getattr(args, 'age', None) is not None and 'Age' in filtered.columns:
        filtered = filtered[filtered['Age'].astype(str) == str(args.age)]

    if getattr(args, 'weight', None):
        w_col = next((c for c in ['Weight Class', 'Weight', 'WeightClass'] if c in filtered.columns), None)
        if w_col:
            filtered = filtered[filtered[w_col].astype(str).str.lower() == args.weight.lower()]

    if getattr(args, 'win_only', False):
        filtered = filtered[filtered['Is_Winner'] == True]

    return filtered

def load_and_preprocess(csv_path):
    """Loads CSV, standardizes column names, and computes duration seconds."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' was not found.")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'\s*/\s*', '/', regex=True)

    time_col = next((c for c in ['Timestamp In/Out', 'Timestamp In / Out'] if c in df.columns), None)
    if time_col:
        df['Duration_Sec'] = df[time_col].apply(parse_timestamp_duration)
    else:
        df['Duration_Sec'] = 0.0

    return df