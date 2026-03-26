"""
Classify loans from snapshot data into soft-liquidation patterns.

Reads the parquet snapshot file, computes per-loan classification flags,
and writes filtered CSVs for each pattern.

Usage:
    uv run python analysis/classify_loans.py
    uv run python analysis/classify_loans.py --snapshots path/to/snapshots.parquet --output analysis/data
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def classify(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-loan classification from time-series snapshots.

    Returns a DataFrame with one row per (loan_id, user, market) and columns:
        n1, n2, collateral_deposited, went_below_sl, went_into_sl,
        sl_max_depth, went_below_then_above_sl, into_then_above_sl,
        time_in_sl, time_below_sl, total_loan_time, initial_debt, end_debt,
        loan_start_time, loan_start_block, loan_end_time, loan_end_block,
        should_be_liquidated
    """
    groups = snapshots_df.groupby(["loan_id", "user", "market"])

    rows = []
    for (loan_id, user, market), g in groups:
        g = g.sort_values("epoch_time")

        went_into_sl = bool(g["in_sl"].any())
        went_below_sl = bool(g["below_sl"].any())
        sl_max_depth = float(g["sl_depth"].max()) if went_into_sl else 0.0

        # "into then above": entered SL and later had a snapshot above SL
        if went_into_sl:
            first_in = g[g["in_sl"]].iloc[0].name
            after_in = g.loc[first_in:]
            into_then_above = bool(after_in["above_sl"].any())
        else:
            into_then_above = False

        # "below then above": went fully below SL band and later recovered above
        if went_below_sl:
            first_below = g[g["below_sl"]].iloc[0].name
            after_below = g.loc[first_below:]
            below_then_above = bool(after_below["above_sl"].any())
        else:
            below_then_above = False

        # Time calculations
        interval = g["epoch_time"].diff().median() if len(g) > 1 else 0
        time_in_sl = float(g["in_sl"].sum() * interval)
        time_below_sl = float(g["below_sl"].sum() * interval)
        total_loan_time = float(g["epoch_time"].iloc[-1] - g["epoch_time"].iloc[0])

        rows.append({
            "loan_id": loan_id,
            "user": user,
            "market": market,
            "n1": int(g["n1"].iloc[0]),
            "n2": int(g["n2"].iloc[0]),
            "collateral_deposited": float(g["total_deposited"].iloc[0]),
            "went_below_sl": went_below_sl,
            "went_into_sl": went_into_sl,
            "sl_max_depth": sl_max_depth,
            "went_below_then_above_sl": below_then_above,
            "into_then_above_sl": into_then_above,
            "time_in_sl": time_in_sl,
            "time_below_sl": time_below_sl,
            "total_loan_time": total_loan_time,
            "initial_debt": float(g["debt"].iloc[0]),
            "end_debt": float(g["debt"].iloc[-1]),
            "loan_start_time": g["epoch_time"].iloc[0],
            "loan_start_block": int(g["block_number"].iloc[0]),
            "loan_end_time": g["epoch_time"].iloc[-1],
            "loan_end_block": int(g["block_number"].iloc[-1]),
            "should_be_liquidated": bool(g["liq_eligible"].any()),
        })

    return pd.DataFrame(rows)


def write_filtered_csvs(details: pd.DataFrame, output_dir: Path):
    """Write the 4 pattern CSVs + the full details CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    details.to_csv(output_dir / "crvusd_user_loan_details.csv", index=False)

    into_sl = details[details["went_into_sl"]]
    into_sl.to_csv(output_dir / "loans_into_sl.csv", index=False)

    into_then_above = details[details["went_into_sl"] & details["into_then_above_sl"]]
    into_then_above.to_csv(output_dir / "loans_into_then_above_sl.csv", index=False)

    below_sl = details[details["went_below_sl"]]
    below_sl.to_csv(output_dir / "loans_below_sl.csv", index=False)

    under_then_above = details[details["went_below_sl"] & details["went_below_then_above_sl"]]
    under_then_above.to_csv(output_dir / "loans_under_then_above_sl.csv", index=False)

    print(f"Total loans:           {len(details):>6}")
    print(f"Into SL:               {len(into_sl):>6}")
    print(f"Into SL → recovered:   {len(into_then_above):>6}")
    print(f"Below SL:              {len(below_sl):>6}")
    print(f"Below SL → recovered:  {len(under_then_above):>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify loans from snapshot data")
    parser.add_argument("--snapshots", type=Path,
                        default=DATA_DIR / "crvusd_user_loan_snapshots.parquet")
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    print(f"Loading snapshots from {args.snapshots}...")
    df = pd.read_parquet(args.snapshots)
    print(f"  {len(df)} snapshots loaded")

    print("Classifying loans...")
    details = classify(df)

    write_filtered_csvs(details, args.output)
    print("Done.")
