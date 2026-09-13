"""
opening_conversion.py
=====================
Empirical Win Rate Calculator Conditioned on First Split State Established.

Description:
------------
Quantifies first-mover advantage by tracking each competitor's trajectory
until the first asymmetric split state (_T or _B) is consolidated.
Symmetric states (St, Fty) and terminal nodes (End) are skipped until
positional control is secured.

Analytical Formulation:
-----------------------
    P(Win | First Split State -> S) = N_wins(S) / N_entries(S)

Where:
    - S: The first consolidated split state (ending in _T or _B).
    - N_entries(S): Total entries into state S across evaluated matches.
    - N_wins(S): Entries where the athlete won the match.

Usage:
------
    python opening_conversion.py [csv_path]

Example:
--------
    python opening_conversion.py markov_sample.csv
"""

import sys
import pandas as pd
from bjj_core import expand_dual_athlete_transitions


def calculate_first_split_state_win_rates(csv_path: str) -> None:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{csv_path}' was not found.")
        return

    # Normalize column headers
    df.columns = df.columns.str.strip().str.replace(r"\s*/\s*", "/", regex=True)

    match_col = next(
        (c for c in ["Match ID", "MatchID", "Match_ID", "Match"] if c in df.columns),
        None,
    )

    if not match_col:
        print("Error: No 'Match ID' column found in the dataset.")
        return

    # Expand records into dual perspectives (Athlete A and B) with split mode active
    dual_df = expand_dual_athlete_transitions(df, split_mode=True)

    # Filter strictly for resolved split states (_T or _B)
    split_transitions = dual_df[
        dual_df["Target_Resolved"].str.endswith(("_T", "_B"))
    ]

    if split_transitions.empty:
        print("Error: No split states (_T or _B) were established in this dataset.")
        return

    # Isolate the very first split state for each athlete perspective per match
    first_split_states = (
        split_transitions.groupby([match_col, "Athlete_Role"])
        .first()
        .reset_index()
    )

    # Aggregate counts and empirical conversion rates
    summary = []
    for state, group in first_split_states.groupby("Target_Resolved"):
        total_occurrences = len(group)
        wins = int(group["Is_Winner"].sum())
        losses = total_occurrences - wins
        win_prob = round(wins / total_occurrences, 2) if total_occurrences > 0 else 0.0

        summary.append({
            "First Split State": state,
            "Total Entries (n)": total_occurrences,
            "Wins": wins,
            "Losses": losses,
            "Win Probability": win_prob,
        })

    summary_df = pd.DataFrame(summary).sort_values(
        by=["Win Probability", "Total Entries (n)"],
        ascending=[False, False],
    )

    print("\n" + "=" * 65)
    print("WIN PROBABILITY BY FIRST ASYMMETRIC CONSOLIDATED STATE (_T / _B)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "markov_sample.csv"
    calculate_first_split_state_win_rates(target_csv)