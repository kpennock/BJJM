"""
BJJ Markov Chain Transition Analyzer
====================================
Usage:
    python markov.py [csv_path] [options]

CLI Options & Modes:
    -s, --split              Enable Perspective Split State Mode (_T / _B).
    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode (e.g. -s_one "1/A,2/A").
    -m, --match-id <ID>      Filter records to a single Match ID.
    --name <string>          Filter transitions involving a specific athlete by name.
    --belt <rank>            Filter by belt rank (e.g., --belt Grey).
    --age <int>              Filter by athlete age (e.g., --age 11).
    --weight <class>         Filter by weight category (e.g., --weight Light).
    --win-only               Filter transitions to winning competitors only.
    -h, --help               Display argument help and exit.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import bjj_core

def analyze_and_plot(df, title_suffix='', split_mode=False):
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

    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=2.2)

    node_colors = []
    for node in G.nodes():
        if split_mode:
            if '_T' in node:
                node_colors.append('#86efac')
            elif '_B' in node:
                node_colors.append('#fca5a5')
            else:
                node_colors.append('#bae6fd')
        else:
            node_colors.append('#bae6fd')

    node_radius = 2600
    nx.draw_networkx_nodes(
        G, pos, node_size=node_radius, node_color=node_colors,
        edgecolors='#1e293b', linewidths=1.8, ax=ax
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8.5, font_weight='bold', font_family='sans-serif', ax=ax
    )

    arc_rad = 0.22
    edges = list(G.edges())
    weights = [max(1.2, G[u][v]['weight'] * 3.5) for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=weights,
        edge_color='#334155',
        arrows=True,
        arrowsize=28,
        arrowstyle='-|>',
        min_source_margin=30,
        min_target_margin=30,
        connectionstyle=f'arc3,rad={arc_rad}',
        ax=ax
    )

    for u, v, data in G.edges(data=True):
        if u == v:
            x, y = pos[u]
            lx, ly = x, y + 0.12
        else:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            offset_factor = 0.55 * arc_rad
            lx = mx - offset_factor * dy
            ly = my + offset_factor * dx

        label_text = f"{data['weight']:.2f} (n={data['count']})"
        ax.text(
            lx, ly, label_text,
            fontsize=7.5,
            fontweight='semibold',
            color='#0f172a',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#ffffff',
                edgecolor='#94a3b8',
                linewidth=0.8,
                alpha=0.92
            ),
            zorder=10
        )

    ax.set_title(f'BJJ State Transitions {title_suffix}', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    fig.tight_layout()

    out_png = 'bjj_transitions_onesided.png' if 'One-Sided' in title_suffix else (
        'bjj_transitions_split.png' if split_mode else 'bjj_transitions_unsplit.png'
    )
    fig.savefig(out_png, dpi=300)
    print(f'>>> Diagram saved to: {out_png}')

    try:
        plt.show()
    except Exception as e:
        print(f'>>> [Note] Pop-up window skipped: {e}')

    return probs

def main():
    parser = argparse.ArgumentParser(description='Process BJJ Markov Chain transitions.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided split state mode with match/competitor pairs (e.g. "1/A,2/A,3/B")')
    parser.add_argument('--match-id', '-m', type=str, default=None, help='Filter by Match ID')
    parser.add_argument('--name', type=str, default=None, help='Filter by athlete name')
    parser.add_argument('--belt', type=str, default=None, help='Filter by belt rank')
    parser.add_argument('--age', type=int, default=None, help='Filter by age')
    parser.add_argument('--weight', type=str, default=None, help='Filter by weight class')
    parser.add_argument('--win-only', action='store_true', help='Filter to winning athlete transitions')

    args = parser.parse_args()

    try:
        raw_df = bjj_core.load_and_preprocess(args.csv_path)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    onesided_map = bjj_core.parse_onesided_pairs(args.split_onesided) if args.split_onesided else None
    split_active = args.split or bool(onesided_map)

    dual_df = bjj_core.expand_dual_athlete_transitions(raw_df, split_mode=split_active, onesided_map=onesided_map)
    filtered = bjj_core.filter_dataset(dual_df, args)

    if filtered.empty:
        print('Error: No transitions match the specified filters.')
        return

    filter_info = []
    if args.split_onesided:
        filter_info.append(f"One-Sided: {args.split_onesided}")
    elif args.split:
        filter_info.append("Split Mode")
    else:
        filter_info.append("Unsplit Mode")

    if args.match_id: filter_info.append(f'Match={args.match_id}')
    if args.name: filter_info.append(f'Name={args.name}')
    if args.belt: filter_info.append(f'Belt={args.belt}')
    if args.age: filter_info.append(f'Age={args.age}')
    if args.weight: filter_info.append(f'Weight={args.weight}')
    if args.win_only: filter_info.append('Winners Only')
    title_suffix = f"({', '.join(filter_info)})" if filter_info else ''

    print(f'>>> Processing {len(filtered)} transition records {title_suffix}...')
    probs = analyze_and_plot(filtered, title_suffix=title_suffix, split_mode=split_active)

    print('\n--- Transition Probability Matrix ---')
    print(probs.round(2))

if __name__ == '__main__':
    main()