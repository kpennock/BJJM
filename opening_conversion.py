"""
opening_conversion.py
=====================
Empirical Win Rate Calculator and Visualizer Conditioned on First Split State.

Description:
------------
Quantifies first-mover advantage by tracking each competitor's trajectory
until the first asymmetric split state (_T or _B) is consolidated.
Outputs a tabular summary to console and generates a bidirectional horizontal
diverging bar chart centered at the 50% neutral equilibrium mark.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from bjj_core import expand_dual_athlete_transitions


def plot_conversion_bars(summary_df: pd.DataFrame) -> None:
    """
    Renders a bidirectional horizontal diverging bar chart centered at 50% win probability.
    """
    if summary_df.empty:
        return

    # Sort so top-performing states appear at the top of the vertical axis
    plot_df = summary_df.sort_values(by="Win Probability", ascending=True).copy()

    # Center delta around neutral 0.50
    plot_df["Delta_50"] = plot_df["Win Probability"] - 0.50

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.55)))

    # Assign green for winner advantage (>50%) and red for disadvantage (<50%)
    bar_colors = [
        "#16a34a" if val >= 0 else "#dc2626" for val in plot_df["Delta_50"]
    ]

    # Draw horizontal bars starting from the 0.50 baseline
    bars = ax.barh(
        plot_df["First Split State"],
        plot_df["Delta_50"],
        left=0.50,
        color=bar_colors,
        edgecolor="#1e293b",
        linewidth=1.2,
        height=0.6,
    )

    # 50% Neutral Baseline
    ax.axvline(0.50, color="#475569", linestyle="--", linewidth=1.5, alpha=0.85)

    # Annotate bars with Win % and sample size (n)
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        win_pct = int(round(row["Win Probability"] * 100))
        n_count = int(row["Total Entries (n)"])
        label_text = f"{win_pct}% (n={n_count})"

        val = row["Delta_50"]
        if val >= 0:
            ax.text(
                row["Win Probability"] + 0.02,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#15803d",
            )
        else:
            ax.text(
                row["Win Probability"] - 0.02,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                va="center",
                ha="right",
                fontsize=8.5,
                fontweight="bold",
                color="#b91c1c",
            )

    ax.set_xlim(-0.05, 1.15)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50% (Equilibrium)", "75%", "100%"], fontweight="semibold")

    ax.set_title(
        "Win Probability Conditioned on First Consolidated Split State (_T / _B)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Empirical Win Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("First Consolidated State", fontsize=10, fontweight="bold")

    # Styling and grid
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = "opening_conversion.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n>>> Visualization saved to: {out_png}")

    try:
        plt.show()
    except Exception as e:
        print(f">>> [Note] GUI window skipped: {e}")


def calculate_first_split_state_win_rates(csv_path: str) -> None:
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
        print("Error: No 'Match ID' column found in the dataset.")
        return

    # Expand records into dual perspectives (_T / _B)
    dual_df = expand_dual_athlete_transitions(df, split_mode=True)

    # Filter strictly for resolved split states (_T or _B)
    split_transitions = dual_df[
        dual_df["Target_Resolved"].str.endswith(("_T", "_B"))
    ]

    if split_transitions.empty:
        print("Error: No split states (_T or _B) were established in this dataset.")
        return

    # Select the very first split state for each athlete perspective per match
    first_split_states = (
        split_transitions.groupby([match_col, "Athlete_Role"])
        .first()
        .reset_index()
    )

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

    plot_conversion_bars(summary_df)


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "markov_sample.csv"
    calculate_first_split_state_win_rates(target_csv)