import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def parse_timestamp_duration(ts_str):
    """Calculates duration in seconds from 'm:ss/m:ss' strings."""
    if pd.isna(ts_str) or '/' not in str(ts_str):
        return 0.0
    parts = str(ts_str).strip().split('/')
    def to_seconds(t_val):
        t_val = t_val.strip()
        m, s = map(float, t_val.split(':'))
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
    a = int(parts[0]) if parts[0].strip() != '' else 0
    b = int(parts[1]) if parts[1].strip() != '' else 0
    return (a, b)

def format_state(row):
    """Constructs state name incorporating dominance when applicable."""
    base = str(row['Primary State']).strip()
    dom = str(row.get('Dominant Athlete', 'None')).strip()
    if dom and dom.lower() not in ['none', 'neutral', 'nan', '']:
        return f"{base} ({dom})"
    return base

def analyze_bjj_data(df):
    # 1. Clean headers: strip spaces and normalize internal slashes
    df.columns = df.columns.str.strip().str.replace(r'\s*/\s*', '/', regex=True)

    time_col = 'Timestamp In/Out'
    adv_col = 'Adv/Pen'

    # 2. Parse durations and advantage/penalties
    df['Duration_Sec'] = df[time_col].apply(parse_timestamp_duration) if time_col in df.columns else 0.0
    if adv_col in df.columns:
        df['Adv_Pen_Tuple'] = df[adv_col].apply(parse_adv_pen)
        df['Adv_Pen_A'] = df['Adv_Pen_Tuple'].apply(lambda x: x[0])
        df['Adv_Pen_B'] = df['Adv_Pen_Tuple'].apply(lambda x: x[1])

    # 3. Standardize Source and Target States
    df['Source_State'] = df.apply(format_state, axis=1)
    df['Target_State'] = df['Next State'].astype(str).str.strip()

    # 4. Generate transition matrix
    transition_counts = pd.crosstab(df['Source_State'], df['Target_State'], dropna=False)
    all_states = sorted(list(set(df['Source_State']).union(set(df['Target_State']))))
    transition_counts = transition_counts.reindex(index=all_states, columns=all_states, fill_value=0)
    
    # Calculate row-normalized probabilities
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0.0)

    # 5. Build and visualize NetworkX DiGraph
    G = nx.DiGraph()
    for state in all_states:
        G.add_node(state)

    for src in transition_probs.index:
        for tgt in transition_probs.columns:
            count = transition_counts.loc[src, tgt]
            if count > 0:
                G.add_edge(src, tgt, weight=transition_probs.loc[src, tgt], count=count)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)

    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color="#d6e8fa", edgecolors="#2b5c8f")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    edges = G.edges()
    weights = [G[u][v]['weight'] * 3.5 for u, v in edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=weights, edge_color="#4a5568",
        arrowsize=16, connectionstyle="arc3,rad=0.1"
    )

    edge_labels = {(u, v): f"{d['weight']:.2f}\n(n={d['count']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)

    plt.title("BJJ State Transition Graph", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return transition_probs, df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python markov.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    prob_matrix, enriched_df = analyze_bjj_data(df)

    print("\n--- Transition Probability Matrix ---")
    print(prob_matrix.round(2))

    print("\n--- Parsed Preview ---")
    preview_cols = ['Timestamp In/Out', 'Source_State', 'Duration_Sec', 'Exit Action', 'Target_State']
    cols = [c for c in preview_cols if c in enriched_df.columns]
    print(enriched_df[cols].head())