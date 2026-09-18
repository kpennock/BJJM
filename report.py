"""
BJJ Match Performance Reporter
==============================
Aggregates state dwell times, comparative scoring actions by split state,
submission analytics, cross-cohort metrics, and empirical positional win
conversion rates across BJJ match records.

Key Features & Sections:
------------------------
1. State Dwell Normalization:
   Normalizes duration metrics per match (seconds / match) to prevent
   rare single-entry stalls from distorting comparative profiles.

2. Comparative Scoring by Originating State:
   Contrasts point production by originating position (Source_Resolved)
   between Winners and Losers, quantifying where decisive scoring occurs.

3. Submission Metrics:
   Details submission frequencies broken down by originating state
   and specific finishing technique (subaction).

4. Cohort Comparative Breakdown:
   Compares top control time, bottom guard time, and average time spent
   per position per match between two designated cohorts.

5. Positional Win Conversion Rate:
   Measures the empirical win rate and victory count once each position or
   perspective-split state (_T / _B) is achieved across unique bouts.

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
                             Splits dynamic guards and dominant pins relative
                             to Top Athlete designation.
                             Default (without -s): Canonical unsplit states.

    -s_one, --split_onesided <pairs>
                             One-Sided Split State Mode.
                             Isolates metrics strictly from the perspective of
                             a designated competitor across specified matches.
                             Format: Comma-separated "match_id/competitor" pairs.
                             Example: -s_one "1/A,2/A,3/B"

Threshold & Filtering Options:
------------------------------
    -lk, --link_count <int>  Minimum incoming transition threshold (n >= lk).
                             Prunes states entered fewer than lk total times.

    -m, --match-id <ID>      Filter records to a single Match ID (e.g., -m 1).

    --name <string>          Filter transitions involving a specific competitor
                             by name (e.g., --name "Eymen Agah Saygili").

    --group <string>         Filter transitions to a single team/group.

    --belt <rank>            Filter by belt rank (e.g., --belt Grey, --belt Yellow).

    --age <int>              Filter by athlete age (e.g., --age 11).

    --weight <class>         Filter by weight class (e.g., --weight Light).

    --win-only               Filter metrics exclusively to the trajectory of
                             winning competitors.

Differential & Cohort Comparison Modes:
---------------------------------------
    --compare-by <attr>      Attribute column to compare across (belt, age, weight, group, win).
    --cohort_1 <val>         First cohort value for differential comparison.
    --cohort_2 <val>         Second cohort value for differential comparison.
    --group_1 <str> --group_2 <str>
                             Legacy shorthand for team/group comparisons.
    Default Comparison:      When no cohort flags are specified, Section 4 defaults
                             to comparing Winners vs. Losers.

    -h, --help               Display the built-in argument help manual and exit.

Example Commands:
-----------------
    1. Run standard report with split states (_T / _B):
       python report.py markov_PanKids2026.csv -s

    2. Run with link count threshold (prunes rare entries):
       python report.py markov_PanKids2026.csv -s -lk 2

    3. Compare Belts (Yellow vs. Grey):
       python report.py markov_PanKids2026.csv -s --compare-by belt --cohort_1 Yellow --cohort_2 Grey

    4. Filter by Weight Class and Belt:
       python report.py markov_PanKids2026.csv -s --belt Grey --weight Light
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

    # Count distinct matches in the active selection
    match_col = next((c for c in ['Match ID', 'MatchID', 'Match_ID', 'Match'] if c in report_df.columns), None)
    total_distinct_matches = report_df[match_col].nunique() if match_col else 1
    total_duration = report_df['Duration_Sec'].sum()

    # -------------------------------------------------------------------------
    # 1. State Dwell Time Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"1. TOTAL DWELL TIME BY STATE (Sample: {total_distinct_matches} Matches)")
    print("=" * 70)
    time_rep = report_df.groupby('Source_Resolved').agg(
        Total_Time=('Duration_Sec', 'sum'),
        Entries=('Duration_Sec', 'count')
    ).reset_index()

    time_rep['Time / Match (s)'] = (time_rep['Total_Time'] / total_distinct_matches).round(1)
    time_rep['Avg / Entry (s)'] = (time_rep['Total_Time'] / time_rep['Entries']).round(1)
    time_rep['% Total Time'] = ((time_rep['Total_Time'] / total_duration) * 100).round(1) if total_duration > 0 else 0.0

    time_rep = time_rep.rename(columns={'Source_Resolved': 'State', 'Total_Time': 'Total Time (s)'})
    time_rep = time_rep[['State', 'Total Time (s)', 'Entries', 'Time / Match (s)', 'Avg / Entry (s)', '% Total Time']]
    time_rep = time_rep.sort_values(by='Total Time (s)', ascending=False)
    print(time_rep.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Scoring by Originating Split State (Winners vs. Losers)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. SCORING BY ORIGINATING SPLIT STATE (Winners vs. Losers)")
    print("=" * 70)
    scoring_df = report_df[report_df['Scored_Points'] > 0]
    if scoring_df.empty:
        print("No points recorded for current selection.")
    else:
        win_scores = scoring_df[scoring_df['Is_Winner'] == True].groupby('Source_Resolved')['Scored_Points'].agg(['sum', 'count'])
        lose_scores = scoring_df[scoring_df['Is_Winner'] == False].groupby('Source_Resolved')['Scored_Points'].agg(['sum', 'count'])

        all_scoring_states = sorted(list(set(scoring_df['Source_Resolved'])))
        score_comp = pd.DataFrame(index=all_scoring_states)
        score_comp['Winner_Pts'] = win_scores['sum']
        score_comp['Winner_Scores (n)'] = win_scores['count']
        score_comp['Loser_Pts'] = lose_scores['sum']
        score_comp['Loser_Scores (n)'] = lose_scores['count']
        score_comp = score_comp.fillna(0).astype(int)

        score_comp['Net_Pts (W - L)'] = score_comp['Winner_Pts'] - score_comp['Loser_Pts']
        score_comp = score_comp.sort_values(by=['Winner_Pts', 'Net_Pts (W - L)'], ascending=[False, False])
        print(score_comp.to_string())

        print("\n>> Scoring Breakdown by Specific Action & State:")
        detailed = scoring_df.groupby(['Source_Resolved', 'Exit Action', 'Is_Winner'])['Scored_Points'].agg(['sum', 'count']).reset_index()
        detailed['Cohort'] = detailed['Is_Winner'].map({True: 'Winner', False: 'Loser'})
        detailed = detailed.rename(columns={'Source_Resolved': 'State', 'sum': 'Total Pts', 'count': 'Executions'})
        detailed = detailed.sort_values(by=['State', 'Cohort', 'Total Pts'], ascending=[True, False, False])
        print(detailed[['State', 'Exit Action', 'Cohort', 'Total Pts', 'Executions']].to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. Submission Overview
    # -------------------------------------------------------------------------
    subs_df = report_df[report_df['Executed_Sub'] == True]
    print("\n" + "=" * 70)
    print(f"3. SUBMISSION METRICS (Total Count: {len(subs_df)})")
    print("=" * 70)

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

    # -------------------------------------------------------------------------
    # 4. Cohort Comparative Breakdown
    # -------------------------------------------------------------------------
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
        print("\n" + "=" * 70)
        print(f"4. COMPARATIVE METRICS ({attr.upper()}): {c1_name} vs. {c2_name}")
        print("=" * 70)

        c1_matches = cohort_1[match_col].nunique() if match_col else 1
        c2_matches = cohort_2[match_col].nunique() if match_col else 1

        top_states = [s for s in report_df['Source_Resolved'].unique() if '_T' in s]
        bottom_states = [s for s in report_df['Source_Resolved'].unique() if '_B' in s]

        c1_top_total = cohort_1[cohort_1['Source_Resolved'].isin(top_states)]['Duration_Sec'].sum()
        c1_bot_total = cohort_1[cohort_1['Source_Resolved'].isin(bottom_states)]['Duration_Sec'].sum()
        c2_top_total = cohort_2[cohort_2['Source_Resolved'].isin(top_states)]['Duration_Sec'].sum()
        c2_bot_total = cohort_2[cohort_2['Source_Resolved'].isin(bottom_states)]['Duration_Sec'].sum()

        print(f"Bouts Evaluated:    {c1_name} = {c1_matches} matches | {c2_name} = {c2_matches} matches")
        print(f"Top Control Time:   {c1_name} = {c1_top_total / c1_matches:.1f}s/match ({c1_top_total:.1f}s total) | "
              f"{c2_name} = {c2_top_total / c2_matches:.1f}s/match ({c2_top_total:.1f}s total)")
        print(f"Bottom Guard Time:  {c1_name} = {c1_bot_total / c1_matches:.1f}s/match ({c1_bot_total:.1f}s total) | "
              f"{c2_name} = {c2_bot_total / c2_matches:.1f}s/match ({c2_bot_total:.1f}s total)")

        all_states = sorted(list(set(cohort_1['Source_Resolved']).union(set(cohort_2['Source_Resolved']))))
        c1_sums = cohort_1.groupby('Source_Resolved')['Duration_Sec'].sum().reindex(all_states, fill_value=0.0)
        c2_sums = cohort_2.groupby('Source_Resolved')['Duration_Sec'].sum().reindex(all_states, fill_value=0.0)

        c1_per_match = (c1_sums / c1_matches).rename(f'{c1_name}_s/Match')
        c2_per_match = (c2_sums / c2_matches).rename(f'{c2_name}_s/Match')

        comp_dwell = pd.concat([c1_per_match, c2_per_match], axis=1)
        comp_dwell['Delta_s/Match'] = (comp_dwell[f'{c1_name}_s/Match'] - comp_dwell[f'{c2_name}_s/Match']).round(1)

        print(f"\n>> Average Time Spent per Position per Match (seconds):")
        print(comp_dwell.round(1).to_string())
    elif attr != 'win':
        print(f"\n[Comparison Skipped] Requires records for both '{c1_name}' ({len(cohort_1)} rows) and '{c2_name}' ({len(cohort_2)} rows) in '{target_col}'.")

    # -------------------------------------------------------------------------
    # 5. Positional Win Conversion Rate (Split States)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. POSITIONAL WIN CONVERSION RATE (Success Rate by State Achieved)")
    print("=" * 70)

    if match_col:
        # Collect all unique visits per athlete per match in each state
        src_visits = report_df[[match_col, 'Athlete_Role', 'Source_Resolved', 'Is_Winner']].rename(
            columns={'Source_Resolved': 'State'}
        )
        tgt_visits = report_df[[match_col, 'Athlete_Role', 'Target_Resolved', 'Is_Winner']].rename(
            columns={'Target_Resolved': 'State'}
        )
        combined_visits = pd.concat([src_visits, tgt_visits]).drop_duplicates(
            subset=[match_col, 'Athlete_Role', 'State']
        )

        # Exclude terminal absorbing match-end state
        combined_visits = combined_visits[combined_visits['State'] != 'End']

        # Total entries across all transitions (gross entry frequency)
        gross_entries = report_df['Target_Resolved'].value_counts()

        summary_rows = []
        for state, group in combined_visits.groupby('State'):
            bouts_reached = len(group)
            wins = int(group['Is_Winner'].sum())
            losses = bouts_reached - wins
            win_pct = round((wins / bouts_reached) * 100, 1) if bouts_reached > 0 else 0.0
            total_entries = int(gross_entries.get(state, len(group)))

            summary_rows.append({
                'State': state,
                'Total Entries': total_entries,
                'Bouts Reached': bouts_reached,
                'Wins': wins,
                'Losses': losses,
                'Win Rate (%)': win_pct
            })

        conv_df = pd.DataFrame(summary_rows)
        conv_df = conv_df.sort_values(by=['Win Rate (%)', 'Bouts Reached'], ascending=[False, False])
        print(conv_df.to_string(index=False))
    else:
        print("[Skipped] 'Match ID' column required to compute bout-level conversion.")


def main():
    parser = argparse.ArgumentParser(description='Generate Performance Reports from BJJ Match Data.')
    parser.add_argument('csv_path', nargs='?', default='markov_sample.csv', help='Path to input CSV file')
    parser.add_argument('-s', '--split', action='store_true', help='Enable split states mode (_T and _B)')
    parser.add_argument('-s_one', '--split_onesided', type=str, default=None,
                        help='One-sided mode with match/athlete pairs (e.g. "1/A,2/A")')
    parser.add_argument('-lk', '--link_count', type=int, default=0,
                        help='Minimum incoming transition threshold (n >= lk)')

    # Generalized Cohort Comparison Flags
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