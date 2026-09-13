"""
BJJ Match Performance Reporter
==============================
Aggregates state dwell times, scoring actions, and submission analytics
from BJJ match transition logs. Supports canonical unsplit states, perspective
split states (_T / _B), and designated one-sided match views.

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
                             Default (without -s): Evaluates canonical unsplit states.

    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode.
                             Isolates the reporting metrics strictly from the
                             viewpoint of a designated competitor (A or B) across
                             specified matches.
                             Format: Comma-separated "match_id/competitor" pairs.
                             Example: -s_one "1/A,2/A,3/B"

Match & Metadata Filter Options:
--------------------------------
    -m, --match-id <ID>      Filter records to a single match by its Match ID (e.g., -m 1).

    --name <string>          Filter transitions involving a specific athlete by name
                             (e.g., --name "Eymen Agah Saygili").

    --belt <rank>            Filter by belt rank (e.g., --belt Grey, --belt yellow).

    --age <int>              Filter by athlete age (e.g., --age 11).

    --weight <class>         Filter by weight category (e.g., --weight Light,
                             --weight Feather).

    --win-only               Filter metrics exclusively to the perspective trajectory
                             of winning competitors.

    -h, --help               Display the built-in argument help manual and exit.

Output Sections:
----------------
    1. Total Dwell Time by State:
       Aggregates cumulative time (seconds), total entries, average dwell per entry,
       and percentage of total match time spent in each state.

    2. Scoring by Exit Action:
       Summarizes total points scored and successful execution counts broken down
       by the specific transition action (e.g., sweeps, guard passes, back takes).

    3. Submission Metrics:
       Provides total submission counts, broken down both by the originating
       positional state and by the specific finishing technique (subaction).

Example Calls:
--------------
    1. Standard run on default file (unsplit states):
       python report.py

    2. Run on a specific CSV file:
       python report.py my_match_data.csv

    3. Run with split-state tracking (_T / _B):
       python report.py markov_sample.csv -s

    4. Analyze one-sided perspectives for matches 1 and 2:
       python report.py markov_sample.csv -s_one "1/A,2/A"

    5. Generate a report for winning competitors in a specific demographic:
       python report.py markov_sample.csv -s --belt Grey --weight Light --age 11 --win-only

    6. Isolate a single match by ID:
       python report.py markov_sample.csv -m 2
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

import bjj_core

def print_reports(df):
    total_duration = df['Duration_Sec'].sum()

    # 1. State Dwell Time Report
    print("\n" + "=" * 60)
    print("1. TOTAL DWELL TIME BY STATE")
    print("=" * 60)
    time_rep = df.groupby('Source_Resolved')['Duration_Sec'].agg(['sum', 'count', 'mean']).reset_index()
    time_rep.columns = ['State', 'Total Time (s)', 'Entries', 'Avg Dwell / Entry (s)']
    time_rep['% Total Time'] = ((time_rep['Total Time (s)'] / total_duration) * 100).round(1) if total_duration > 0 else 0.0
    time_rep = time_rep.sort_values(by='Total Time (s)', ascending=False)
    print(time_rep.to_string(index=False))

    # 2. Scoring by Exit Action
    print("\n" + "=" * 60)
    print("2. SCORING BY EXIT ACTION")
    print("=" * 60)
    scoring_df = df[df['Scored_Points'] > 0]
    if scoring_df.empty:
        print("No points recorded for current selection.")
    else:
        action_rep = scoring_df.groupby('Exit Action')['Scored_Points'].agg(['sum', 'count']).reset_index()
        action_rep.columns = ['Action', 'Total Points', 'Success Count']
        print(action_rep.sort_values(by='Total Points', ascending=False).to_string(index=False))

    # 3. Submission Overview
    subs_df = df[df['Executed_Sub'] == True]
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

def main():
    parser = argparse.ArgumentParser(description='Generate Performance Reports from BJJ Match Data.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided mode with match/athlete pairs (e.g. "1/A,2/A")')
    parser.add_argument('--match-id', '-m', type=str, default=None, help='Filter by Match ID')
    parser.add_argument('--name', type=str, default=None, help='Filter by athlete name')
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

    print_reports(filtered)

if __name__ == '__main__':
    main()