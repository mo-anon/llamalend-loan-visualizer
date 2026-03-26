"""
Find interesting loans from the classified CSV data.

Filters loans by soft-liquidation pattern, collateral size, time in SL, depth,
and market. Outputs loan configs ready to use with the video pipeline.

Usage:
    # Most dramatic: fell below SL band then recovered
    uv run python analysis/find_interesting_loans.py --pattern under_then_above

    # Large loans that entered SL for at least 2 days
    uv run python analysis/find_interesting_loans.py --pattern into_sl --min-collateral 50 --min-time-in-sl 172800

    # Deep SL penetration, any pattern
    uv run python analysis/find_interesting_loans.py --min-depth 0.8

    # Filter by market (controller address prefix)
    uv run python analysis/find_interesting_loans.py --pattern below_sl --market 0x8472

    # Sort by depth or time
    uv run python analysis/find_interesting_loans.py --pattern into_then_above --sort depth --limit 20
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

PATTERN_FILES = {
    "into_sl": "loans_into_sl.csv",
    "into_then_above": "loans_into_then_above_sl.csv",
    "below_sl": "loans_below_sl.csv",
    "under_then_above": "loans_under_then_above_sl.csv",
}

SORT_KEYS = {
    "depth": "sl_max_depth",
    "time": "time_in_sl",
    "collateral": "collateral_deposited",
    "debt": "initial_debt",
    "duration": "total_loan_time",
}


def load_loans(pattern: str | None, data_dir: Path) -> pd.DataFrame:
    if pattern and pattern in PATTERN_FILES:
        path = data_dir / PATTERN_FILES[pattern]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run classify_loans.py first.")
        return pd.read_csv(path)
    return pd.read_csv(data_dir / "crvusd_user_loan_details.csv")


def format_time(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def main():
    parser = argparse.ArgumentParser(description="Find interesting LLAMMA loans")
    parser.add_argument("--pattern", choices=list(PATTERN_FILES.keys()),
                        help="SL pattern to filter by")
    parser.add_argument("--market", type=str, help="Controller address (prefix match)")
    parser.add_argument("--user", type=str, help="User address (prefix match)")
    parser.add_argument("--min-collateral", type=float, default=0)
    parser.add_argument("--min-depth", type=float, default=0,
                        help="Minimum SL depth (0-1)")
    parser.add_argument("--min-time-in-sl", type=float, default=0,
                        help="Minimum time in SL (seconds)")
    parser.add_argument("--sort", choices=list(SORT_KEYS.keys()), default="depth",
                        help="Sort by (default: depth)")
    parser.add_argument("--asc", action="store_true", help="Sort ascending")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    df = load_loans(args.pattern, args.data_dir)

    # Apply filters
    if args.market:
        df = df[df["market"].str.lower().str.startswith(args.market.lower())]
    if args.user:
        df = df[df["user"].str.lower().str.startswith(args.user.lower())]
    if args.min_collateral > 0:
        df = df[df["collateral_deposited"] >= args.min_collateral]
    if args.min_depth > 0:
        df = df[df["sl_max_depth"] >= args.min_depth]
    if args.min_time_in_sl > 0:
        df = df[df["time_in_sl"] >= args.min_time_in_sl]

    sort_col = SORT_KEYS[args.sort]
    df = df.sort_values(sort_col, ascending=args.asc)
    df = df.head(args.limit)

    if df.empty:
        print("No loans match the filters.")
        return

    # Print results
    print(f"{'#':>3}  {'Market':<12} {'User':<12} {'Blocks':>22}  "
          f"{'Depth':>5} {'Time SL':>8} {'Collat':>10} {'Debt':>12}")
    print("-" * 95)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        market_short = row["market"][:10] + ".."
        user_short = row["user"][:10] + ".."
        block_range = f"{int(row['loan_start_block']):>10} → {int(row['loan_end_block']):<10}"
        depth = f"{row['sl_max_depth']:.2f}"
        time_sl = format_time(row["time_in_sl"])
        collat = f"{row['collateral_deposited']:,.1f}"
        debt = f"{row['initial_debt']:,.0f}"

        print(f"{i:>3}  {market_short:<12} {user_short:<12} {block_range}  "
              f"{depth:>5} {time_sl:>8} {collat:>10} {debt:>12}")

    # Print full details for easy copy-paste into the pipeline
    print("\n" + "=" * 95)
    print("Pipeline configs (copy-paste ready):\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"# --- Loan {i} ---")
        print(f"CONTROLLER_ADDRESS = \"{row['market']}\"")
        print(f"USER_ADDRESS       = \"{row['user']}\"")
        print(f"START_BLOCK        = {int(row['loan_start_block'])}")
        print(f"END_BLOCK          = {int(row['loan_end_block'])}")
        print()


if __name__ == "__main__":
    main()
