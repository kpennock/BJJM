import sys
import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print('>>> [1/4] markov.py loaded successfully.')

SPLIT_STATES = {'CG', 'OG', 'FM', 'BM', 'HG', 'SC'}

def parse_timestamp_duration(ts_str):
    if pd.isna(ts_str) or '/' not in str(ts_str):
        return 0.0
    parts = str(ts_str).strip().split('/')
    parts = str(ts_str).strip().split('/')
    def to_seconds(t_val):
        m, s = map(float, t_val.strip().split(':'))
        return m * 60 + s
    try:
        return abs(to_seconds(parts[0]) - to_seconds(parts[1]))
    except Exception:
        return 0.0

def resolve_perspective_state(state_name, athlete_id, dom_athlete):
    raw_state = str(state_name).strip()
    if raw_state not in SPLIT_STATES:
        return raw_state
    dom_norm = str(dom_athlete).strip().lower()
    ath_norm = str(athlete_id).strip().lower()
    if dom_norm in ['none', 'neutral', 'n', 'nan', '']:
        return raw_state
    return f'{raw_state}_T' if ath_norm == dom_norm else f'{raw_state}_B'

def expand_dual_athlete_transitions(df):
    processed = []
    has_win = 'Win' in df.columns
    for _, row in df.iterrows():
        src_raw = str(row.get('Primary State', '')).strip()
        tgt_raw = str(row.get('Next State', '')).strip()
        dom = str(row.get('Dominant Athlete', 'None')).strip()

        name_a = row.get('Name A', 'Athlete A')
        name_b = row.get('Name B', 'Athlete B')

        src_a = resolve_perspective_state(src_raw, 'A', dom)
        tgt_a = resolve_perspective_state(tgt_raw, 'A', dom)
        src_b = resolve_perspective_state(src_raw, 'B', dom)
        tgt_b = resolve_perspective_state(tgt_raw, 'B', dom)

        base = row.to_dict()
        win_val = str(row.get('Win', '')).strip().lower()

        rec_a = base.copy()
        rec_a.update({
            'Focal_Athlete': name_a,
            'Athlete_Role': 'A',
            'Source_Resolved': src_a,
            'Target_Resolved': tgt_a,
            'Is_Winner': (win_val in ['a', str(name_a).lower()]) if has_win else True
        })
        processed.append(rec_a)

        rec_b = base.copy()
        rec_b.update({
            'Focal_Athlete': name_b,
            'Athlete_Role': 'B',
            'Source_Resolved': src_b,
            'Target_Resolved': tgt_b,
            'Is_Winner': (win_val in ['b', str(name_b).lower()]) if has_win else True
        })
        processed.append(rec_b)

    return pd.DataFrame(processed)

def filter_dataset(df, args):
    filtered = df.copy()
    if args.match_id is not None:
        m_col = next((c for c in ['Match ID', 'MatchID', 'Match_ID', 'Match'] if c in filtered.columns), None)
        if m_col:
            filtered = filtered[filtered[m_col].astype(str) == str(args.match_id)]
    if args.name:
        name_cols = [c for c in ['Name A', 'Name B', 'Focal_Athlete'] if c in filtered.columns]
        if name_cols:
            mask = False
            for c in name_cols:
                mask = mask | (filtered[c].astype(str).str.lower() == args.name.lower())
            filtered = filtered[mask]
    if args.belt and 'Belt' in filtered.columns:
        filtered = filtered[filtered['Belt'].astype(str).str.lower() == args.belt.lower()]
    if args.age is not None and 'Age' in filtered.columns:
        filtered = filtered[filtered['Age'].astype(str) == str(args.age)]
    if args.weight:
        w_col = next((c for c in ['Weight Class', 'Weight', 'WeightClass'] if c in filtered.columns), None)
        if w_col:
            filtered = filtered[filtered[w_col].astype(str).str.lower() == args.weight.lower()]
    if args.win_only:
        filtered = filtered[filtered['Is_Winner'] == True]
    return filtered

def analyze_and_plot(df, title_suffix=''):
    counts = pd.crosstab(df['Source_Resolved'], df['Target_Resolved'], dropna=False)
    all_states = sorted(list(set(df['Source_Resolved']).union(set(df['Target_Resolved']))))
    counts = counts.reindex(index=all_states, columns=all_states, fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    G = nx.DiGraph()
    for s in all_states:
        G.add_node(s)
    for src in probs.index:
        for tgt in probs.columns:
            c = counts.loc[src, tgt]
            if c > 0:
                G.add_edge(src, tgt, weight=probs.loc[src, tgt], count=c)

    plt.figure(figsize=(13, 8))
    pos = nx.spring_layout(G, seed=42, k=1.8)

    node_colors = []
    for node in G.nodes():
        if '_T' in node:
            node_colors.append('#a3e635')
        elif '_B' in node:
            node_colors.append('#f87171')
        else:
            node_colors.append('#93c5fd')

    nx.draw_networkx_nodes(G, pos, node_size=2600, node_color=node_colors, edgecolors='#334155', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    edges = G.edges()
    weights = [max(1.0, G[u][v]['weight'] * 3.5) for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='#64748b', arrowsize=16, connectionstyle='arc3,rad=0.15')

    edge_labels = {(u, v): f"{d['weight']:.2f}\n(n={d['count']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
    edge_labels = {(u, v): f"{d['weight']:.2f}\n(n={d['count']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)

    plt.title(f'BJJ State Transitions {title_suffix}', fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    out_png = 'bjj_transitions.png'
    plt.savefig(out_png, dpi=300)
    print(f'>>> [4/4] Plot saved to: {out_png}')

    try:
        plt.show()
    except Exception as e:
        print(f'>>> [Note] Skipped GUI window: {e}')

    return probs

def main():
    print('>>> [2/4] Initializing arguments...')
    parser = argparse.ArgumentParser(description='Process BJJ Markov Chain transitions.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('--match-id', '-m', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--belt', type=str, default=None)
    parser.add_argument('--age', type=int, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--win-only', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f'Error: File {args.csv_path} was not found.')
        return

    print(f'>>> [3/4] Reading dataset: {args.csv_path}')
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip().str.replace(r'\s*/\s*', '/', regex=True)

    dual_df = expand_dual_athlete_transitions(df)
    filtered = filter_dataset(dual_df, args)

    if filtered.empty:
        print('Error: No transition data matched the specified filter criteria.')
        return

    filter_info = []
    if args.match_id: filter_info.append(f'Match={args.match_id}')
    if args.name: filter_info.append(f'Name={args.name}')
    if args.belt: filter_info.append(f'Belt={args.belt}')
    if args.age: filter_info.append(f'Age={args.age}')
    if args.weight: filter_info.append(f'Weight={args.weight}')
    if args.win_only: filter_info.append('Winners Only')
    title_suffix = f"({', '.join(filter_info)})" if filter_info else '(All Data)'

    probs = analyze_and_plot(filtered, title_suffix=title_suffix)

    print('\n--- Transition Probability Matrix ---')
    print(probs.round(2))

if __name__ == '__main__':
    main()
