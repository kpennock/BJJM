import io
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def parse_timestamp_duration(ts_str):
    """Calculates duration in seconds from countdown/countup timestamp pair (e.g.

    '4:00/3:48').
    """
    if pd.isna(ts_str) or "/" not in str(ts_str):
        return 0.0

    parts = str(ts_str).strip().split("/")

    def to_seconds(t_val):
        t_val = t_val.strip()
        m, s = map(float, t_val.split(":"))
        return m * 60 + s

    t1 = to_seconds(parts[0])
    t2 = to_seconds(parts[1])
    return abs(t1 - t2)


def parse_adv_pen(adv_str):
    """Parses compound adv/pen tokens (e.g., '1/-1', '/-1', '2/0') into (A, B)

    integers.
    """
    if pd.isna(adv_str) or "/" not in str(adv_str):
        return (0, 0)
    parts = str(adv_str).strip().split("/")
    a = int(parts[0]) if parts[0] != "" else 0
    b = int(parts[1]) if parts[1] != "" else 0
    return (a, b)


def format_state(row, state_col="Primary State"):
    """Constructs a consistent node label including dominance perspective."""
    base = str(row[state_col]).strip()
    dom = (
        str(row["Dominant Athlete"]).strip()
        if "Dominant Athlete" in row
        else "None"
    )

    if dom and dom.lower() not in ["none", "neutral", "nan", ""]:
        return f"{base} ({dom})"
    return base


def analyze_bjj_data(df):
    # 1. Parse timestamps, dwell times, and adv/pen
    df["Duration_Sec"] = df["Timestamp In / Out"].apply(
        parse_timestamp_duration
    )
    df["Adv_Pen_Tuple"] = df["Adv/Pen"].apply(parse_adv_pen)
    df["Adv_Pen_A"] = df["Adv_Pen_Tuple"].apply(lambda x: x[0])
    df["Adv_Pen_B"] = df["Adv_Pen_Tuple"].apply(lambda x: x[1])

    # 2. Normalize Current State and Next State labels
    df["Source_State"] = df.apply(
        lambda r: format_state(r, "Primary State"), axis=1
    )
    df["Target_State"] = df["Next State"].astype(str).str.strip()

    # 3. Compute Transition Count and Probability Matrix
    transition_counts = pd.crosstab(
        df["Source_State"], df["Target_State"], dropna=False
    )
    all_states = sorted(
        list(set(df["Source_State"]).union(set(df["Target_State"])))
    )

    # Reindex to ensure square matrix covering all observed states
    transition_counts = transition_counts.reindex(
        index=all_states, columns=all_states, fill_value=0
    )
    transition_probs = transition_counts.div(
        transition_counts.sum(axis=1), axis=0
    ).fillna(0.0)

    # 4. Generate Graph Visualization
    G = nx.DiGraph()

    for state in all_states:
        G.add_node(state)

    for src in transition_probs.index:
        for tgt in transition_probs.columns:
            prob = transition_probs.loc[src, tgt]
            count = transition_counts.loc[src, tgt]
            if count > 0:
                G.add_edge(src, tgt, weight=prob, count=count)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Draw nodes and structural layout
    nx.draw_networkx_nodes(
        G, pos, node_size=2800, node_color="#d6e8fa", edgecolors="#2b5c8f"
    )
    nx.draw_networkx_labels(
        G, pos, font_size=9, font_weight="bold", font_family="sans-serif"
    )

    # Draw directed edges
    edges = G.edges()
    weights = [G[u][v]["weight"] * 3.5 for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=weights,
        edge_color="#4a5568",
        arrowsize=18,
        connectionstyle="arc3,rad=0.1",
    )

    # Edge labels showing transition probabilities
    edge_labels = {
        (u, v): f"{d['weight']:.2f}\n(n={d['count']})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3
    )

    plt.title("BJJ State Transition Graph", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return transition_probs, df

