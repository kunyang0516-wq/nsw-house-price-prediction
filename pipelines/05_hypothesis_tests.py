"""H1/H2 hypothesis tests with clustered uncertainty and a ratio-bias placebo.

Usage
-----
    python pipelines/05_hypothesis_tests.py
    python pipelines/05_hypothesis_tests.py --max-rows 300000 --n-boot 50   # smoke test

Outputs (reports/tables):
    h1_headline_correlations.csv, h1_by_year.csv, h1_within_location.csv,
    h2_ratio_correlation.csv, h2_placebo_ratio.csv, h2_elasticity.csv,
    h2_unit_price_by_area_band.csv, hypothesis_summary.json,
    figures/h2_ratio_placebo.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.hypothesis_tests import (
    neighbourhood_robustness,
    test_h1_area_price,
    test_h2_unit_price,
)
from src.features.pipeline import OUT_FEATURES_PARQUET
from src.utils.config import REPORTS_DIR
from src.utils.io import write_json

TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--max-rows", type=int, default=800_000,
                        help="cap rows used for the bootstrap/placebo loops "
                             "(point estimates still use the full sample)")
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--n-placebo", type=int, default=200)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    full_rows = len(frame)
    print(f"Full feature frame: {full_rows:,} rows")

    analysis = frame
    if args.max_rows and args.max_rows < full_rows:
        analysis = frame.sample(n=args.max_rows, random_state=36103)
        print(f"Bootstrap/placebo subsample: {len(analysis):,} rows "
              f"(point estimates reported on the same rows for consistency)")

    print("\n--- H1: area vs price ---")
    h1 = test_h1_area_price(analysis, n_boot=args.n_boot)
    headline = pd.DataFrame(h1.headline)
    print(headline[["measure", "coefficient", "ci_low", "ci_high", "n", "n_postcodes"]].to_string(index=False))
    print("\nWithin-location estimate (postcode + year FE, clustered by postcode):")
    print(pd.DataFrame([h1.within_location])[["label", "coef", "se", "ci_low", "ci_high", "n", "n_clusters"]].to_string(index=False))
    print("\nH1 interpretation:")
    print("  " + h1.interpretation)

    print("\n--- H2: unit price and area (ratio-bias placebo) ---")
    h2 = test_h2_unit_price(analysis, n_placebo=args.n_placebo)
    print(f"  observed corr(log area, log unit price): {h2.observed['coefficient']:+.4f} "
          f"(n={h2.observed['n']:,})")
    draws = h2.placebo["coefficient"]
    print(f"  shuffled-price placebo: median {draws.median():+.4f}, "
          f"2.5-97.5% [{draws.quantile(0.025):+.4f}, {draws.quantile(0.975):+.4f}] "
          f"({len(draws)} draws)")
    print(f"  elasticity (log price ~ log area | postcode+year FE): {h2.elasticity['coef']:.4f} "
          f"(se {h2.elasticity['se']:.4f}), test beta=1 -> p = {h2.elasticity.get('p_value', float('nan')):.3g}")
    print("\n  Median unit price by area decile:")
    print(h2.area_bands[["area_band", "median_area", "median_unit_price", "unit_price_index", "n"]]
          .to_string(index=False))
    print("\nH2 interpretation:")
    print("  " + h2.interpretation)

    print("\n--- Robustness: adding the lagged neighbourhood price ---")
    robustness = neighbourhood_robustness(analysis)
    print("\n  H1 (area elasticity under three specifications):")
    show = [c for c in ("label", "coef", "se", "ci_low", "ci_high", "n", "n_clusters") if c in robustness.h1_table.columns]
    print(robustness.h1_table[show].to_string(index=False))
    print("\n  H2 (elasticity, beta = 1 test):")
    show = [c for c in ("label", "coef", "se", "p_value", "n") if c in robustness.h2_table.columns]
    print(robustness.h2_table[show].to_string(index=False))
    print("\n  Bad-control caveat:")
    print("  " + robustness.bad_control_note)
    print("\n  " + robustness.interpretation)

    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for name, table in {**h1.to_frames(), **h2.to_frames(), **robustness.to_frames()}.items():
            table.to_csv(TABLES_DIR / f"{name}.csv", index=False)

        mechanical_share = (abs(draws.median()) / abs(h2.observed["coefficient"])
                            if h2.observed["coefficient"] else float("nan"))
        write_json(TABLES_DIR / "hypothesis_summary.json", {
            "rows_full": int(full_rows),
            "rows_used": int(len(analysis)),
            "h1": {
                "headline": h1.headline,
                "within_location": h1.within_location,
                "interpretation": h1.interpretation,
            },
            "h2": {
                "observed_ratio_correlation": h2.observed["coefficient"],
                "placebo_median": float(draws.median()),
                "placebo_ci": [float(draws.quantile(0.025)), float(draws.quantile(0.975))],
                "mechanical_share_of_observed": float(mechanical_share),
                "elasticity": h2.elasticity,
                "interpretation": h2.interpretation,
            },
            "neighbourhood_robustness": {
                "bad_control_note": robustness.bad_control_note,
                "interpretation": robustness.interpretation,
                "h1": robustness.h1_table.to_dict(orient="records"),
                "h2": robustness.h2_table.to_dict(orient="records"),
            },
            "elapsed_seconds": round(time.time() - started, 1),
        })
        make_placebo_plot(h2, FIGURES_DIR / "h2_ratio_placebo.png")
        print(f"\nWrote hypothesis tables to {TABLES_DIR}")
        print(f"Wrote {FIGURES_DIR / 'h2_ratio_placebo.png'}")
    print(f"Elapsed: {time.time() - started:,.1f}s")
    return 0


def make_placebo_plot(h2, path: Path) -> None:
    """Show the observed ratio correlation against its shuffled-price null."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draws = h2.placebo["coefficient"].to_numpy(dtype=float)
    observed = h2.observed["coefficient"]
    accent, grey, ink2 = "#2a78d6", "#b4b2ac", "#6b6963"

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    ax.hist(draws, bins=30, color=grey, alpha=0.85,
            label=f"shuffled-price placebo (n={len(draws)})")
    ax.axvline(float(np.median(draws)), color=ink2, lw=1.6, ls="-",
               label=f"placebo median {np.median(draws):+.3f}")
    ax.axvline(observed, color=accent, lw=2.4, ls="--",
               label=f"observed {observed:+.3f}")
    mechanical = abs(np.median(draws)) / abs(observed) if observed else float("nan")
    ax.set_xlabel("corr(log area, log unit price)")
    ax.set_ylabel("placebo draws")
    ax.set_title("H2: how much of the unit-price/area association is the ratio itself?\n"
                 f"{mechanical:.0%} of the observed correlation is reproduced with prices shuffled",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.012, 0.012,
             "Placebo shuffles purchase price across dwellings, leaving area untouched, so any "
             "remaining association comes from area sitting in the ratio's denominator.",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor="white", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
