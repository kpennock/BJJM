"""
BJJ Match Performance Reporter
==============================
Usage:
    python report.py [csv_path] [options]

CLI Options & Modes:
    -s, --split              Enable Perspective Split State Mode (_T / _B).
    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode (e.g. -s_one "1/A,2/A").
    -lk, --link_count <int>  Threshold for minimum incoming transition links (n >= lk).
    --group_1 <str> --group_2 <str>
                             Compare dwell times and control for two specific teams/groups.
    -m, --match-id <ID>      Filter records to a single Match ID.
    --name <string>          Filter transitions by athlete name.
    --group <string>         Filter transitions to a single team/group.
    --belt <rank>            Filter by belt rank (e.g., --belt Grey).
    --age <int>              Filter by athlete age (e.g., --age 11).
    --weight <class>         Filter by weight category (e.g., --weight Light).
    --win-only               Filter metrics to winning competitors only.
    -h, --help               Display argument help and exit.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

import bjj_core

def print_reports(df, args):
    report_df = df.copy()

    if args.link_count > 0:
        inbound_counts = bjj_core.get_incoming_link_counts(report_df)
        valid_states = {state for state, count in inbound_counts.items() if count >= args.link_count}
        report_df = report_df[
            report_df['Source_Resolved'].isin(valid_states) | 
            report_df['Target_Resolved'].isin(valid_states)
        ]
        print(f"\n[Filter Active] Minimum Inbound Threshold: n >= {args.link_count}")
        print(f">> Qualified States: {sorted(list(valid_states))}")

    if report_df.empty:
        print("\nNo states satisfy the minimum incoming transition threshold.")
        return

    total_duration = report_df['Duration_Sec'].sum()

    # 1. State Dwell Time Report
    print("\n" + "=" * 60)
    print("1. TOTAL DWELL TIME BY STATE")
    print("=" * 60)
    time_rep = report_df.groupby('Source_Resolved')['Duration_Sec'].agg(['sum', 'count', 'mean']).reset_index()
    time_rep.columns = ['State', 'Total Time (s)', 'Entries', 'Avg Dwell / Entry (s)']
    time_rep['% Total Time'] = ((time_rep['Total Time (s)'] / total_duration) * 100).round(1) if total_duration > 0 else 0.0
    time_rep = time_rep.sort_values(by='Total Time (s)', ascending=False)
    print(time_rep.to_string(index=False))

    # 2. Scoring by Exit Action
    print("\n" + "=" * 60)
    print("2. SCORING BY EXIT ACTION")
    print("=" * 60)
    scoring_df = report_df[report_df['Scored_Points'] > 0]
    if scoring_df.empty:
        print("No points recorded for current selection.")
    else:
        action_rep = scoring_df.groupby('Exit Action')['Scored_Points'].agg(['sum', 'count']).reset_index()
        action_rep.columns = ['Action', 'Total Points', 'Success Count']
        print(action_rep.sort_values(by='Total Points', ascending=False).to_string(index=False))

    # 3. Submission Overview
    subs_df = report_df[report_df['Executed_Sub'] == True]
    print("\n" + "=" * 60)
    print(f"3. SUBMISSION METRICS (Total Count: {len(subs_df)})")
    print("=" * 60)

    if subs_df.empty:
        print("No submissions recorded for current selection.")
    else:
        print(">> By Originating State:")
        sub_state = subs_df.groupby('Source_Resolved').size().reset_index(name='Subs')
        print(sub_state.sort_values(by='Subs', ascending=False).to_string(index=False))

        print("\n>> By Specific Subaction (Technique):")
        subs_df_copy = subs_df.copy()
        subs_df_copy['Subaction_Clean'] = subs_df_copy['Subaction'].replace('', np.nan).fillna('Unspecified')
        sub_tech = subs_df_copy.groupby('Subaction_Clean').size().reset_index(name='Subs')
        print(sub_tech.sort_values(by='Subs', ascending=False).to_string(index=False))

    # 4. Cohort Comparative Breakdown
    is_group_comp = (args.group_1 is not None and args.group_2 is not None)
    if is_group_comp:
        c1_name, c2_name = args.group_1, args.group_2
        cohort_1 = report_df[report_df['Athlete_Group'].str.lower() == c1_name.lower()]
        cohort_2 = report_df[report_df['Athlete_Group'].str.lower() == c2_name.lower()]
    else:
        c1_name, c2_name = "Winners", "Losers"
        cohort_1 = report_df[report_df['Is_Winner'] == True]
        cohort_2 = report_df[report_df['Is_Winner'] == False]

    if not cohort_1.empty and not cohort_2.empty:
        print("\n" + "=" * 60)
        print(f"4. COMPARATIVE METRICS: {c1_name} vs. {c2_name}")
        print("=" * 60)

        top_states = [s for s in report_df['Source_Resolved'].unique() if '_T' in s]
        bottom_states = [s for s in report_df['Source_Resolved'].unique() if '_B' in s]

        c1_top = cohort_1[cohort_1['Source_Resolved'].isin(top_states)]['Duration_Sec'].sum()
        c1_bot = cohort_1[cohort_1['Source_Resolved'].isin(bottom_states)]['Duration_Sec'].sum()
        c2_top = cohort_2[cohort_2['Source_Resolved'].isin(top_states)]['Duration_Sec'].sum()
        c2_bot = cohort_2[cohort_2['Source_Resolved'].isin(bottom_states)]['Duration_Sec'].sum()

        print(f"Top Control Time:   {c1_name} = {c1_top:.1f}s | {c2_name} = {c2_top:.1f}s")
        print(f"Bottom Guard Time:  {c1_name} = {c1_bot:.1f}s | {c2_name} = {c2_bot:.1f}s")

        c1_dwell = cohort_1.groupby('Source_Resolved')['Duration_Sec'].mean().rename(f'{c1_name}_Avg_Dwell')
        c2_dwell = cohort_2.groupby('Source_Resolved')['Duration_Sec'].mean().rename(f'{c2_name}_Avg_Dwell')

        comp_dwell = pd.concat([c1_dwell, c2_dwell], axis=1).fillna(0.0)
        comp_dwell['Dwell_Delta'] = (comp_dwell[f'{c1_name}_Avg_Dwell'] - comp_dwell[f'{c2_name}_Avg_Dwell']).round(1)
        print(f"\n>> Mean Dwell Time (s) per Position:")
        print(comp_dwell.round(1).to_string())

def main():
    parser = argparse.ArgumentParser(description='Generate Performance Reports from BJJ Match Data.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided mode with match/athlete pairs (e.g. "1/A,2/A")')
    parser.add_argument('-lk', '--link_count', type=int, default=0,
                        help='Minimum incoming transition threshold (n >= lk)')
    parser.add_argument('--group_1', type=str, default=None, help='First group for comparison')
    parser.add_argument('--group_2', type=str, default=None, help='Second group for comparison')
    parser.add_argument('--match-id', '-m', type=str, default=None, help='Filter by Match ID')
    parser.add_argument('--name', type=str, default=None, help='Filter by athlete name')
    parser.add_argument('--group', type=str, default=None, help='Filter by specific team/group')
    parser.add_argument('--belt', type=str, default=None, help='Filter by belt rank')
    parser.add_argument('--age', type=int, default=None, help='Filter by age')
    parser.add_argument('--weight', type=str, default=None, help='Filter by weight class')
    parser.add_argument('--win-only', action='store_true', help='Filter to winning athlete perspective only')

    args = parser.parse_args()

    try:
        raw_df = bjj_core.load_and_preprocess(args.csv_path)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    onesided_map = bjj_core.parse_onesided_pairs(args.split_onesided) if args.split_onesided else None
    split_active = args.split or bool(onesided_map)

    expanded = bjj_core.expand_dual_athlete_transitions(raw_df, split_mode=split_active, onesided_map=onesided_map)
    filtered = bjj_core.filter_dataset(expanded, args)

    if filtered.empty:
        print('No data matches the selected filters.')
        return

    print_reports(filtered, args)

if __name__ == '__main__':
    main()