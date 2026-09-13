"""
opening_conversion.py
=====================
Empirical Opening Transition Win Rate Calculator for BJJ Match Analytics.

Description:
------------
This script quantifies first-mover advantage across tournament matches.
It isolates each competitor's very first transition out of the neutral standing
state (`St` -> non-`St`) and calculates the empirical win probability conditioned
on the resulting destination position or split state (_T / _B).

Analytical Formulation:
-----------------------
    P(Win | First Exchange -> S) = N_wins(S) / N_entries(S)

Where:
    - S: The target state reached after leaving the opening standing exchange.
    - N_entries(S): Total athlete entries into state S on the first exchange.
    - N_wins(S): Entries where the athlete won the match.

Usage:
------
    python opening_conversion.py [csv_path]

Examples:
---------
    1. Run on default sample file:
       python opening_conversion.py

    2. Run on a specific tournament file:
       python opening_conversion.py markov_sample.csv
"""

import sys
import pandas as pd
from bjj_core import expand_dual_athlete_transitions


def calculate_opening_win_rates(csv_path: str) -> None:
    """
    Parses tournament match logs, isolates the first non-standing transition
    for each athlete perspective, and computes historical win probabilities.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{csv_path}' was not found.")
        return

    df.columns = df.columns.str.strip().str.replace(r"\s*/\s*", "/", regex=True)

    match_col = next(
        (c for c in ["Match ID", "MatchID", "Match_ID", "Match"] if c in df.columns),
        None,
    )

    if not match_col:
        print("Error: No 'Match ID' column found. Ensure matches are separated by an ID.")
        return

    # Expand records into dual perspectives (_T / _B)
    dual_df = expand_dual_athlete_transitions(df)

    # Isolate transitions leaving the initial standing exchange (St -> non-St)
    opening_transitions = dual_df[
        (dual_df["Source_Resolved"] == "St") & 
        (dual_df["Target_Resolved"] != "St")
    ]

    if opening_transitions.empty:
        print("Error: No transitions leaving 'St' were found in the dataset.")
        return

    # Select the first opening transition per athlete trajectory per match
    first_exchanges = (
        opening_transitions.groupby([match_col, "Athlete_Role"])
        .first()
        .reset_index()
    )

    # Aggregate total entries, wins, losses, and conversion probabilities
    summary = []
    for state, group in first_exchanges.groupby("Target_Resolved"):
        total_occurrences = len(group)
        wins = int(group["Is_Winner"].sum())
        losses = total_occurrences - wins
        win_prob = round(wins / total_occurrences, 2) if total_occurrences > 0 else 0.0

        summary.append({
            "First Exchange State": state,
            "Total Entries (n)": total_occurrences,
            "Wins": wins,
            "Losses": losses,
            "Win Probability": win_prob,
        })

    summary_df = pd.DataFrame(summary).sort_values(
        by=["Win Probability", "Total Entries (n)"], 
        ascending=[False, False]
    )

    print("\n" + "=" * 65)
    print("WIN PROBABILITY BY STATE AFTER FIRST STANDING EXCHANGE")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "markov_sample.csv"
    calculate_opening_win_rates(target_csv)