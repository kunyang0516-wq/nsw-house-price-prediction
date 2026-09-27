"""Build the cleaned analysis frame (leakage-safe).

Usage
-----
    python pipelines/01_build_clean.py                 # full run
    python pipelines/01_build_clean.py --limit-chunks 2  # smoke test
    python pipelines/01_build_clean.py --iqr-mode off    # no trimming (robust-loss run)

Outlier policy has two layers:

* a **corruption fence** fitted on the training span and applied to every row —
  it only removes values that cannot be a house transaction;
* a **year-specific quality fence** (default: fitted on the whole cleaned sample)
  which flags sample-level extremes. It is applied to the training side only, and
  fitting it per year is what stops a 2001-2015 price ceiling from rejecting
  legitimate 2023 sales.

Logs a staged row-count report, the per-year fence table, and a comparison
against the group's notebook numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import build_clean
from src.data.contract import (
    GROUP_CLEAN_ROWS,
    assert_all,
    check_clean,
    postcode_coverage,
)
from src.data.outliers import (
    apply_iqr_filter,
    fit_global_bounds,
    fit_iqr_bounds,
)
from src.data.postcode_features import (
    build_development_labels,
    build_transport_features,
)
from src.utils.config import (
    OUT_CLEAN_PARQUET,
    OUT_POSTCODE_DEVELOPMENT,
    OUT_POSTCODE_TRANSPORT,
    REPORTS_DIR,
    SETTINGS,
    ensure_dirs,
)
from src.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-chunks", type=int, default=None,
                        help="only read the first N raw chunks (smoke test)")
    parser.add_argument("--iqr-mode",
                        choices=["per_year_full_sample", "per_year_train_only",
                                 "global_train_only", "off"],
                        default="per_year_full_sample",
                        help="per_year_full_sample (default, option A): year-specific quality "
                             "fences fitted on the whole cleaned sample, so a 2023 sale is "
                             "judged against 2023 prices rather than a 2001-2015 ceiling; "
                             "per_year_train_only: the original, which mis-flags recent sales")
    parser.add_argument("--drop-unjudged", action="store_true",
                        help="also drop rows the training fences never judged (later years); "
                             "default keeps them flagged so the validation span survives")
    parser.add_argument("--train-end", default=SETTINGS.windows.train_end,
                        help="last contract date whose rows may be used to FIT the fences")
    parser.add_argument("--skip-reference", action="store_true",
                        help="reuse existing postcode lookup CSVs instead of rebuilding them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    policy = SETTINGS.cleaning

    # ------------------------------------------------------------------ #
    # 1. Postcode reference tables (development labels + distances)
    # ------------------------------------------------------------------ #
    if args.skip_reference and OUT_POSTCODE_TRANSPORT.exists() and OUT_POSTCODE_DEVELOPMENT.exists():
        transport = pd.read_csv(OUT_POSTCODE_TRANSPORT, dtype={"postcode": "string"})
        development = pd.read_csv(OUT_POSTCODE_DEVELOPMENT, dtype={"postcode": "string"})
        print("Reusing existing postcode reference tables.")
    else:
        print("Building postcode development labels ...")
        development = build_development_labels(SETTINGS.paths["raw_property"], policy)
        print(f"  labelled postcodes: {len(development):,}")
        for column in [c for c in development.columns if c.startswith('development_type_')]:
            counts = development[column].value_counts(dropna=False).to_dict()
            print(f"  {column}: {counts}")

        print("Building postcode transport features ...")
        transport = build_transport_features()
        print(f"  postcode rows: {len(transport):,}")

    # ------------------------------------------------------------------ #
    # 2. Cleaning pass (no fitted statistics)
    # ------------------------------------------------------------------ #
    print("Cleaning raw records in chunks ...")
    pre_outlier, report = build_clean(policy, transport_features=transport,
                                     development_lookup=development,
                                     max_chunks=args.limit_chunks)
    print("\n" + report.summary())
    print(f"\nPre-outlier frame: {len(pre_outlier):,} rows, {len(pre_outlier.columns)} columns")

    if args.limit_chunks:
        print("\n--limit-chunks supplied: skipping IQR fit and parquet export.")
        return 0

    # ------------------------------------------------------------------ #
    # 3. Outlier fences (audit M1 / L4)
    # ------------------------------------------------------------------ #
    dates = pd.to_datetime(pre_outlier["contract_date"], errors="coerce")
    train_mask = dates.le(pd.Timestamp(args.train_end))
    train_rows = int(train_mask.sum())
    print(f"\nTraining span (<= {args.train_end}): {train_rows:,} rows "
          f"of {len(pre_outlier):,} ({train_rows / len(pre_outlier):.1%})")

    bounds = None
    if args.iqr_mode != "off":
        by = ("year",) if args.iqr_mode.startswith("per_year") else ()
        if args.iqr_mode == "per_year_train_only":
            fit_frame = pre_outlier.loc[train_mask]
        else:
            # Option A: year-specific quality fences are fitted on the FULL sample.
            # A naive fence fitted on 2001-2015 and applied to 2023 is the wrong
            # measuring stick: the training-span price ceiling is ~$2.34M while the
            # 2023 99th percentile is ~$2.9M, so legitimate recent sales were being
            # rejected as outliers. Year-specific bounds fix that, and because the
            # upper bound for a year is derived mostly from other years (the fence
            # uses the whole sample) the self-reference is ~1/n per year.
            fit_frame = pre_outlier
        bounds = fit_iqr_bounds(fit_frame, by=by, multiplier=policy.iqr_multiplier)
        print(f"  fences fitted: {len(bounds.bounds)} ({args.iqr_mode}, by={by or 'global'}, "
              f"on {'full sample' if args.iqr_mode != 'per_year_train_only' else 'training span'})")

    # Corruption fence: fit on the TRAINING SPAN ONLY and apply it to every row.
    # It only has to catch values that cannot be a house transaction (a 2.7e9 sqm
    # parcel, an $875M "residence"), so a training-span fit is both sufficient and
    # conservative.
    global_bounds = fit_global_bounds(pre_outlier.loc[train_mask],
                                      multiplier=policy.iqr_multiplier)
    clean, outlier_report = apply_iqr_filter(
        pre_outlier, bounds, global_bounds=global_bounds)
    print("\nOutlier step:", outlier_report)
    if bounds is not None:
        bounds_frame = bounds.to_frame().sort_values(["column", "group"])
        out_path = REPORTS_DIR / "tables" / "iqr_bounds.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bounds_frame.to_csv(out_path, index=False)
        print(f"  fences written to {out_path}")

        # Per-year fence values and how hard each year is trimmed: this is the
        # table that would have exposed the stale-ceiling problem immediately.
        per_year = (bounds_frame.pivot_table(index="group", columns="column",
                                             values="value_upper", aggfunc="first"))
        per_year = per_year.rename(columns={"purchase_price": "price_upper",
                                            "area_sqm": "area_upper"})
        # The fence table is indexed by string year labels (built with
        # astype("string")); the date year is an integer, so align them explicitly
        # or every assignment below silently becomes NaN.
        numeric_index = pd.to_numeric(per_year.index, errors="coerce")
        per_year = per_year.loc[numeric_index.notna()].copy()
        per_year.index = numeric_index[numeric_index.notna()].astype(int)
        kept = clean["iqr_keep"].astype(bool)
        by_year = dates.dt.year
        per_year["rows_kept"] = kept.groupby(by_year).sum()
        counts = by_year.value_counts()
        per_year["rows_flagged"] = counts.sub(per_year["rows_kept"], fill_value=0)
        total = per_year["rows_kept"] + per_year["rows_flagged"]
        per_year["flagged_share"] = (per_year["rows_flagged"] / total).round(4)
        per_year = per_year[["price_upper", "area_upper", "rows_kept",
                             "rows_flagged", "flagged_share"]]
        per_year.to_csv(REPORTS_DIR / "tables" / "iqr_bounds_by_year.csv")
        print(f"  per-year fence table written to {REPORTS_DIR / 'tables' / 'iqr_bounds_by_year.csv'}")
        print(per_year.tail(12).to_string())

        print(f"\n  rows outside the year fence (flagged, training-side only): "
              f"{outlier_report['rows_quality_fail_flagged']:,} "
              f"({outlier_report['rows_quality_fail_flagged'] / len(pre_outlier):.1%})")
        print(f"  rows failing the corruption fence (dropped outright): "
              f"{outlier_report['rows_corruption_fail']:,}")

    # ------------------------------------------------------------------ #
    # 4. Contract + coverage report, then persist
    # ------------------------------------------------------------------ #
    checks = check_clean(clean)
    for check in checks:
        print(check)
    assert_all(checks)

    coverage = postcode_coverage(clean)
    print("\nCoverage:", coverage)
    print(f"Group notebook for comparison: {GROUP_CLEAN_ROWS:,} rows "
          f"(single full-sample fence); this run: {len(clean):,} rows")

    clean.to_parquet(OUT_CLEAN_PARQUET, index=False)
    print(f"\nWrote {OUT_CLEAN_PARQUET} ({OUT_CLEAN_PARQUET.stat().st_size / 1024**2:,.1f} MiB)")

    write_json(OUT_CLEAN_PARQUET.with_suffix(".meta.json"), {
        "rows": int(len(clean)),
        "columns": list(clean.columns),
        "coverage": coverage,
        "group_notebook_rows": GROUP_CLEAN_ROWS,
        "iqr": outlier_report,
        "iqr_bounds": bounds.as_dict() if bounds is not None else None,
        "cleaning_policy": {
            "required_fields": list(policy.required_fields),
            "required_development_column": policy.required_development_column,
            "development_windows": [
                {"name": w.name, "start": w.start, "end": w.end, "label_as_of": w.label_as_of}
                for w in policy.development_windows
            ],
        },
        "stage_counts": {k: int(v) for k, v in report.stage_counts.items()},
        "plausibility_gate": report.plausibility,
        "duplicate_rows_removed": int(report.duplicate_rows_removed),
        "invalid_value_rows_removed": int(report.invalid_value_rows_removed),
    })
    print("Wrote", OUT_CLEAN_PARQUET.with_suffix('.meta.json'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
