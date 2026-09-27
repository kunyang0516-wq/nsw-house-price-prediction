"""Build the modelling frame (macro as-of + panel features).

Usage
-----
    python pipelines/02_build_features.py
    python pipelines/02_build_features.py --refresh-macro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.macro_asof import load_macro_panel, macro_coverage_report
from src.features.pipeline import build_features, load_clean
from src.utils.config import OUT_CLEAN_PARQUET, PROCESSED_DIR
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-macro", action="store_true",
                        help="re-download the RBA snapshots instead of using data/external")
    parser.add_argument("--sample", type=int, default=None,
                        help="use only the first N cleaned rows (smoke test)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not OUT_CLEAN_PARQUET.exists():
        print(f"Missing {OUT_CLEAN_PARQUET}; run pipelines/01_build_clean.py first.")
        return 2

    clean = load_clean()
    if args.sample:
        clean = clean.head(args.sample)
        print(f"Smoke test on the first {len(clean):,} rows.")

    macro = load_macro_panel(refresh=args.refresh_macro)
    print("RBA snapshots loaded:")
    print(f"  cash rate: {len(macro.cash_rate):,} effective dates "
          f"({macro.cash_rate['effective_date'].min():%Y-%m-%d} .. "
          f"{macro.cash_rate['effective_date'].max():%Y-%m-%d})")
    print(f"  CPI:       {len(macro.cpi):,} quarters "
          f"({macro.cpi['period_end'].min():%Y-%m-%d} .. "
          f"{macro.cpi['period_end'].max():%Y-%m-%d}), "
          f"publication lag assumed {macro.cpi_publication_lag_days} days")

    frame, report = build_features(clean=clean, macro=macro, save=not args.sample)

    if not args.sample:
        write_json(PROCESSED_DIR / "features.meta.json", report)
        print("\nPanel coverage by year:")
        coverage = macro_coverage_report(frame)
        print(coverage.tail(10).to_string(index=False))
        print("\nNULL counts on key features:")
        key_features = ["area_sqm", "dist_cbd", "cash_rate_asof", "cpi_yoy_asof",
                        "pc_med_price_12m", "pc_n_sales_12m", "development_type", "log_price"]
        print(frame[[c for c in key_features if c in frame.columns]].isna().sum().to_string())
        print(f"\nRows with a legal development label: {frame['development_type'].notna().sum():,} "
              f"({frame['development_type'].notna().mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
