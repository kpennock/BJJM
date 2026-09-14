"""
BJJ Markov Chain Transition Analyzer
====================================
Usage:
    python markov.py [csv_path] [options]

Differential Comparison Modes:
    --compare-win            Compare Winners vs. Losers: ΔP = P(win) - P(loss).
    --group_1 <str> --group_2 <str>
                             Compare any two groups/teams: ΔP = P(group_1) - P(group_2).
    --heatmap                Render differential output as a 2D diverging heatmap.

Filtering & Threshold Options:
    -lk, --link_count <int>  Node In-Degree Threshold (n >= lk).
                             Prunes entire states (rows/columns/nodes) entered fewer
                             than lk total times across the evaluated dataset.
    -tk, --trans_count <int> Transition Edge Count Threshold (n >= tk).
                             Suppresses specific (source -> target) transitions where
                             the transition count is below tk. In graphs, edges with
                             count < tk are not drawn. In heatmaps, cells with
                             count < tk are zeroed out.

Perspective & View Modes:
    -s, --split              Enable Perspective Split State Mode (_T / _B).
    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode (e.g. -s_one "1/A,2/A").

Metadata Filter Options:
    -m, --match-id <ID>      Filter records to a single Match ID.
    --name <string>          Filter transitions by athlete name.
    --group <string>         Filter transitions to a single team/group.
    --belt <rank>            Filter by belt rank (e.g., --belt Grey).
    --age <int>              Filter by athlete age (e.g., --age 11).
    --weight <class>         Filter by weight category (e.g., --weight Light).
    --win-only               Filter transitions to winning competitors only.
    -h, --help               Display argument help and exit.

Examples:
---------
    1. Focus on primary hubs, but see all exits:
       python markov.py markov_sample.csv -s --compare-win --heatmap -lk 3

    2. Look at all states, but only display transitions repeated at least twice:
       python markov.py markov_sample.csv -s --compare-win --heatmap -tk 2

    3. Strict high-confidence mode (frequent states AND frequent transitions):
       python markov.py markov_sample.csv -s --compare-win --heatmap -lk 3 -tk 2
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import bjj_core

def compute_prob_matrix(df, allowed_states=None, min_incoming_links=0):
    raw_counts = pd.crosstab(df['Source_Resolved'], df['Target_Resolved'], dropna=False)
    all_states = sorted(list(set(df['Source_Resolved']).union(set(df['Target_Resolved']))))
    counts = raw_counts.reindex(index=all_states, columns=all_states, fill_value=0)

    # 1. State/Node-level filtering (-lk)
    if allowed_states is not None:
        # Strictly intersect with the pre-qualified allowed states
        valid_states = sorted([s for s in allowed_states if s in counts.index or s in counts.columns])
        counts = counts.reindex(index=valid_states, columns=valid_states, fill_value=0)
        all_states = valid_states
    elif min_incoming_links > 0:
        incoming_totals = counts.sum(axis=0)
        valid_states = sorted(incoming_totals[incoming_totals >= min_incoming_links].index.tolist())
        counts = counts.reindex(index=valid_states, columns=valid_states, fill_value=0)
        all_states = valid_states

    if len(all_states) == 0:
        return pd.DataFrame(), pd.DataFrame(), []

    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0).fillna(0.0)
    return probs, counts, all_states

def plot_differential_heatmap(delta_p, total_counts=None, min_trans_count=0, label_1="Cohort 1", label_2="Cohort 2"):
    if delta_p.empty:
        print("Warning: Differential matrix is empty; skipping heatmap.")
        return

    display_delta = delta_p.copy()

    # Zero out cells that do not satisfy the transition count threshold (-tk)
    if min_trans_count > 0 and total_counts is not None:
        mask = total_counts < min_trans_count
        display_delta[mask] = 0.0

    fig, ax = plt.subplots(figsize=(12, 10))
    states = list(display_delta.index)

    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cax = ax.imshow(
        display_delta.values,
        cmap='coolwarm_r',
        norm=norm,
        interpolation='nearest'
    )
    cax.format_cursor_data = lambda data: ""

    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha='right', fontsize=9, fontweight='bold')
    ax.set_yticklabels(states, fontsize=9, fontweight='bold')

    for i in range(len(states)):
        for j in range(len(states)):
            val = display_delta.iloc[i, j]
            if abs(val) >= 0.01:
                color = 'white' if abs(val) > 0.45 else 'black'
                ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                        color=color, fontsize=8, fontweight='bold')

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'ΔP: Favors {label_2} (< 0)  vs.  Favors {label_1} (> 0)', fontsize=10, fontweight='bold')

    title_suffix = f" [Trans Count >= {min_trans_count}]" if min_trans_count > 0 else ""
    ax.set_title(f"State Transition Differential Heatmap [{label_1} vs. {label_2}]{title_suffix}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Target State", fontsize=11, fontweight='bold')
    ax.set_ylabel("Source State", fontsize=11, fontweight='bold')
    fig.tight_layout()

    out_png = 'bjj_differential_heatmap.png'
    fig.savefig(out_png, dpi=300)
    print(f'>>> Differential Heatmap saved to: {out_png}')
    try:
        plt.show()
    except Exception as e:
        print(f'>>> [Note] GUI window skipped: {e}')

def plot_differential_graph(delta_p, total_counts=None, min_trans_count=0, label_1="Cohort 1", label_2="Cohort 2", split_mode=False):
    if delta_p.empty:
        print("Warning: Differential matrix is empty; skipping graph.")
        return

    G = nx.DiGraph()
    states = list(delta_p.index)
    for s in states:
        G.add_node(s)

    edge_colors = []
    edge_widths = []
    active_edges = []

    for src in states:
        for tgt in states:
            diff = delta_p.loc[src, tgt]
            c = total_counts.loc[src, tgt] if total_counts is not None else min_trans_count
            if abs(diff) >= 0.02 and c >= min_trans_count:
                G.add_edge(src, tgt, weight=diff, count=c)
                active_edges.append((src, tgt))
                edge_colors.append('#16a34a' if diff > 0 else '#dc2626')
                edge_widths.append(max(1.5, abs(diff) * 6.0))

    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=2.2)

    node_colors = []
    for node in G.nodes():
        if split_mode:
            if '_T' in node:
                node_colors.append('#bbf7d0')
            elif '_B' in node:
                node_colors.append('#fecaca')
            else:
                node_colors.append('#e2e8f0')
        else:
            node_colors.append('#e2e8f0')

    node_radius = 2600
    nx.draw_networkx_nodes(
        G, pos, node_size=node_radius, node_color=node_colors,
        edgecolors='#1e293b', linewidths=1.8, ax=ax
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8.5, font_weight='bold', font_family='sans-serif', ax=ax
    )

    arc_rad = 0.22
    nx.draw_networkx_edges(
        G, pos,
        edgelist=active_edges,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=26,
        arrowstyle='-|>',
        min_source_margin=30,
        min_target_margin=30,
        connectionstyle=f'arc3,rad={arc_rad}',
        ax=ax
    )

    for u, v in active_edges:
        diff = delta_p.loc[u, v]
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

        label_color = '#15803d' if diff > 0 else '#b91c1c'
        ax.text(
            lx, ly, f"{diff:+.2f}",
            fontsize=8,
            fontweight='bold',
            color=label_color,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='#ffffff',
                edgecolor=label_color,
                linewidth=1.0,
                alpha=0.92
            ),
            zorder=10
        )

    title_suffix = f" [Trans Count >= {min_trans_count}]" if min_trans_count > 0 else ""
    ax.set_title(f"Differential Transitions [Green = Favors {label_1} | Red = Favors {label_2}]{title_suffix}",
                 fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    fig.tight_layout()

    out_png = 'bjj_differential_graph.png'
    fig.savefig(out_png, dpi=300)
    print(f'>>> Differential Graph saved to: {out_png}')
    try:
        plt.show()
    except Exception as e:
        print(f'>>> [Note] GUI window skipped: {e}')

def analyze_and_plot(probs, counts, all_states, title_suffix='', split_mode=False, min_trans_count=0):
    if len(all_states) == 0:
        print("Warning: No nodes satisfy the specified threshold.")
        return

    G = nx.DiGraph()
    for s in all_states:
        G.add_node(s)
    for src in probs.index:
        for tgt in probs.columns:
            c = counts.loc[src, tgt]
            if c >= max(1, min_trans_count):
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
        print(f'>>> [Note] GUI window skipped: {e}')

def main():
    parser = argparse.ArgumentParser(description='Process BJJ Markov Chain transitions.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided split state mode with match/competitor pairs (e.g. "1/A,2/A,3/B")')
    parser.add_argument('-lk', '--link_count', type=int, default=0,
                        help='Minimum incoming transition count threshold (n >= lk)')
    parser.add_argument('-tk', '--trans_count', type=int, default=0,
                        help='Minimum transition sample threshold (n >= tk; suppresses edges/cells below tk)')
    parser.add_argument('--compare-win', action='store_true',
                        help='Visualize differential transitions: P(win) - P(loss)')
    parser.add_argument('--group_1', type=str, default=None, help='First group for differential comparison')
    parser.add_argument('--group_2', type=str, default=None, help='Second group for differential comparison')
    parser.add_argument('--heatmap', action='store_true',
                        help='Render comparison as a 2D diverging heatmap instead of a graph')
    parser.add_argument('--match-id', '-m', type=str, default=None, help='Filter by Match ID')
    parser.add_argument('--name', type=str, default=None, help='Filter by athlete name')
    parser.add_argument('--group', type=str, default=None, help='Filter by specific team/group')
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

    # Differential Comparison Pipeline
    is_group_comp = (args.group_1 is not None and args.group_2 is not None)
    if args.compare_win or is_group_comp:
        if is_group_comp:
            c1_name = args.group_1
            c2_name = args.group_2
            df_1 = filtered[filtered['Athlete_Group'].str.lower() == c1_name.lower()]
            df_2 = filtered[filtered['Athlete_Group'].str.lower() == c2_name.lower()]
        else:
            c1_name = "Winners"
            c2_name = "Losers"
            df_1 = filtered[filtered['Is_Winner'] == True]
            df_2 = filtered[filtered['Is_Winner'] == False]

        if df_1.empty or df_2.empty:
            print(f"Error: Comparison requires data for both '{c1_name}' ({len(df_1)} rows) and '{c2_name}' ({len(df_2)} rows).")
            return

        combined_df = pd.concat([df_1, df_2])

        # Node-Level Qualification (-lk): Filter strictly on incoming links
        if args.link_count > 0:
            inbound_counts = combined_df['Target_Resolved'].value_counts()
            allowed_states = sorted(inbound_counts[inbound_counts >= args.link_count].index.tolist())
        else:
            allowed_states = sorted(list(set(combined_df['Source_Resolved']).union(set(combined_df['Target_Resolved']))))

        # Compute probability and count matrices for both cohorts restricted to qualified states
        p_1, c_1, _ = compute_prob_matrix(df_1, allowed_states=allowed_states)
        p_2, c_2, _ = compute_prob_matrix(df_2, allowed_states=allowed_states)

        all_union_states = sorted(allowed_states)
        p_1 = p_1.reindex(index=all_union_states, columns=all_union_states, fill_value=0.0)
        p_2 = p_2.reindex(index=all_union_states, columns=all_union_states, fill_value=0.0)

        c_1 = c_1.reindex(index=all_union_states, columns=all_union_states, fill_value=0)
        c_2 = c_2.reindex(index=all_union_states, columns=all_union_states, fill_value=0)
        total_counts = c_1.add(c_2, fill_value=0)

        delta_p = p_1 - p_2

        threshold_notes = []
        if args.link_count > 0:
            threshold_notes.append(f"Inbound >= {args.link_count}")
        if args.trans_count > 0:
            threshold_notes.append(f"Trans Count >= {args.trans_count}")
        thresh_str = f" [Thresholds: {', '.join(threshold_notes)}]" if threshold_notes else ""

        print("\n" + "=" * 65)
        print(f"DIFFERENTIAL TRANSITION MATRIX: ΔP = P({c1_name}) - P({c2_name}){thresh_str}")
        print(f"Positive (+): Favors {c1_name} | Negative (-): Favors {c2_name}")
        print("=" * 65)
        print(delta_p.round(2))

        if args.heatmap:
            plot_differential_heatmap(
                delta_p, total_counts=total_counts, min_trans_count=args.trans_count,
                label_1=c1_name, label_2=c2_name
            )
        else:
            plot_differential_graph(
                delta_p, total_counts=total_counts, min_trans_count=args.trans_count,
                label_1=c1_name, label_2=c2_name, split_mode=split_active
            )
        return

    filter_info = []
    if args.split_onesided:
        filter_info.append(f"One-Sided: {args.split_onesided}")
    elif args.split:
        filter_info.append("Split Mode")
    else:
        filter_info.append("Unsplit Mode")

    if args.link_count > 0: filter_info.append(f'Inbound>={args.link_count}')
    if args.trans_count > 0: filter_info.append(f'TransCount>={args.trans_count}')
    if args.match_id: filter_info.append(f'Match={args.match_id}')
    if args.name: filter_info.append(f'Name={args.name}')
    if args.group: filter_info.append(f'Group={args.group}')
    if args.belt: filter_info.append(f'Belt={args.belt}')
    if args.age: filter_info.append(f'Age={args.age}')
    if args.weight: filter_info.append(f'Weight={args.weight}')
    if args.win_only: filter_info.append('Winners Only')
    title_suffix = f"({', '.join(filter_info)})" if filter_info else ''

    probs, counts, all_states = compute_prob_matrix(
        filtered, min_incoming_links=args.link_count
    )

    print(f'>>> Processing {len(filtered)} transition records {title_suffix}...')
    analyze_and_plot(
        probs, counts, all_states, title_suffix=title_suffix,
        split_mode=split_active, min_trans_count=args.trans_count
    )

    print('\n--- Transition Probability Matrix ---')
    print(probs.round(2))

if __name__ == '__main__':
    main()