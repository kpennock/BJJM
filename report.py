"""
BJJ Match Performance Reporter
==============================
Aggregates state dwell times, scoring actions, submission analytics, and
comparative metrics across matches. Supports canonical unsplit states, perspective
split states (_T / _B), designated one-sided match views, and cross-cohort comparisons.

Usage:
------
    python report.py [csv_path] [options]

Positional Arguments:
---------------------
    csv_path                 Path to the input CSV file. Defaults to
                             'markov_sample.csv' if omitted.

View & Perspective Modes:
-------------------------
    -s, --split              Enable Perspective Split State Mode (_T / _B).
                             Splits symmetric/asymmetric states (CG, OG, FM,
                             BM, HG, SC) into Top (_T) and Bottom (_B) positions
                             relative to each athlete's perspective.
                             Default (without -s): Canonical unsplit states.

    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode.
                             Isolates reporting metrics strictly from the
                             viewpoint of a designated competitor (A or B) across
                             specified matches.
                             Format: Comma-separated "match_id/competitor" pairs.
                             Example: -s_one "1/A,2/A,3/B"

Threshold & Filtering Options:
------------------------------
    -lk, --link_count <int>  Minimum incoming transition threshold (n >= lk).
                             Excludes states entered fewer than lk total times.

    -m, --match-id <ID>      Filter records to a single match by its Match ID (e.g., -m 1).

    --name <string>          Filter transitions involving a specific athlete by name
                             (e.g., --name "Eymen Agah Saygili").

    --group <string>         Filter transitions to a single team/group.

    --belt <rank>            Filter by belt rank (e.g., --belt Grey, --belt yellow).

    --age <int>              Filter by athlete age (e.g., --age 11).

    --weight <class>         Filter by weight category (e.g., --weight Light,
                             --weight Feather).

    --win-only               Filter metrics exclusively to the perspective trajectory
                             of winning competitors.

Differential & Cohort Comparison Modes:
---------------------------------------
    --compare-by <attr>      Attribute column to compare across (belt, age, weight, group, win).
    --cohort_1 <val>         First cohort value for differential comparison.
    --cohort_2 <val>         Second cohort value for differential comparison.
    --group_1 <str> --group_2 <str>
                             Legacy shorthand for team/group comparisons.
                             (Equivalent to: --compare-by group --cohort_1 <str> --cohort_2 <str>)
    Default Comparison:      When no cohort flags are specified, Section 4 defaults
                             to comparing Winners vs. Losers.

    -h, --help               Display the built-in argument help manual and exit.
    
    e.g.python report.py markov_sample.csv -s --compare-by belt --cohort_1 Yellow --cohort_2 Grey
        python report.py markov_sample.csv -s --compare-by age --cohort_1 11 --cohort_2 12
Output Sections:
----------------
    1. Total Dwell Time by State:
       Aggregates cumulative time (seconds), total entries, average dwell per entry,
       and percentage of total match time spent in each state.

    2. Scoring by Exit Action:
       Summarizes total points scored and successful execution counts broken down
       by transition action (e.g., sweeps, guard passes, back takes).

    3. Submission Metrics:
       Provides total submission counts, broken down both by the originating
       positional state and by the specific finishing technique (subaction).

    4. Cohort Comparative Breakdown:
       Compares top control time, bottom guard time, and average dwell time
       per position between two cohorts (e.g., belts, weights, ages, teams, or win/loss).

Example Calls:
--------------
    1. Standard run on default file (unsplit states):
       python report.py

    2. Compare Belts (Yellow vs. Grey) in Split State View:
       python report.py markov_sample.csv -s --compare-by belt --cohort_1 Yellow --cohort_2 Grey

    3. Compare Weight Classes (Light vs. Middle):
       python report.py markov_sample.csv -s --compare-by weight --cohort_1 Light --cohort_2 Middle

    4. Compare Age Divisions with link threshold:
       python report.py markov_sample.csv -s --compare-by age --cohort_1 11 --cohort_2 12 -lk 2

    5. Compare Teams / Academies:
       python report.py markov_sample.csv -s --compare-by group --cohort_1 "Norther Tribe" --cohort_2 "XType BJJ"
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
    attr = None
    c1_name = None
    c2_name = None

    if args.compare_by and args.cohort_1 is not None and args.cohort_2 is not None:
        attr = args.compare_by.strip().lower()
        c1_name = str(args.cohort_1).strip()
        c2_name = str(args.cohort_2).strip()
    elif args.group_1 is not None and args.group_2 is not None:
        attr = 'group'
        c1_name = str(args.group_1).strip()
        c2_name = str(args.group_2).strip()
    else:
        attr = 'win'
        c1_name = "Winners"
        c2_name = "Losers"

    col_map = {
        'belt': 'Belt',
        'age': 'Age',
        'weight': next((c for c in ['Weight Class', 'Weight', 'WeightClass'] if c in report_df.columns), 'Weight Class'),
        'group': 'Athlete_Group',
        'win': 'Is_Winner'
    }

    target_col = col_map.get(attr)
    if not target_col or target_col not in report_df.columns:
        print(f"\n[Comparison Skipped] Target column for '{attr}' not found in dataset.")
        return

    if attr == 'win':
        cohort_1 = report_df[report_df['Is_Winner'] == True]
        cohort_2 = report_df[report_df['Is_Winner'] == False]
    else:
        cohort_1 = report_df[report_df[target_col].astype(str).str.lower() == c1_name.lower()]
        cohort_2 = report_df[report_df[target_col].astype(str).str.lower() == c2_name.lower()]

    if not cohort_1.empty and not cohort_2.empty:
        print("\n" + "=" * 60)
        print(f"4. COMPARATIVE METRICS ({attr.upper()}): {c1_name} vs. {c2_name}")
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
    elif attr != 'win':
        print(f"\n[Comparison Skipped] Requires records for both '{c1_name}' ({len(cohort_1)} rows) and '{c2_name}' ({len(cohort_2)} rows) in '{target_col}'.")

def main():
    parser = argparse.ArgumentParser(description='Generate Performance Reports from BJJ Match Data.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided mode with match/athlete pairs (e.g. "1/A,2/A")')
    parser.add_argument('-lk', '--link_count', type=int, default=0,
                        help='Minimum incoming transition threshold (n >= lk)')

    # Generalized Cohort Differential Flags
    parser.add_argument('--compare-by', type=str, default=None,
                        help='Attribute column to compare across (belt, age, weight, group, win)')
    parser.add_argument('--cohort_1', type=str, default=None,
                        help='First cohort value for comparison')
    parser.add_argument('--cohort_2', type=str, default=None,
                        help='Second cohort value for comparison')

    # Legacy team flags
    parser.add_argument('--group_1', type=str, default=None, help='First group for comparison (legacy)')
    parser.add_argument('--group_2', type=str, default=None, help='Second group for comparison (legacy)')

    # Metadata filter options
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