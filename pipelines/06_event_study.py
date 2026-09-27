"""E1 Metro opening event study (single event, not-yet-treated controls).

Deliverables
------------
* `event_study_coefficients.csv` — event-time coefficients for every specification
* `event_study_pretrend.csv`    — the joint pre-trend test (the causal gate)
* `event_study_placebo.csv`     — shifted-date and random-treatment placebos
* `event_study_robustness.csv`  — radius x outcome grid
* `event_study_panel.csv`       — the balanced panel actually estimated
* `event_study_plot.png`        — coefficient path with 95% CI

Usage
-----
    python pipelines/06_event_study.py
    python pipelines/06_event_study.py --sample 400000   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal.event_study import (
    balance_panel,
    coefficients_to_wide,
    dose_response,
    estimate_event_study,
    placebo_random_treatment,
    placebo_shifted_event,
)
from src.data.events import (
    EventStudySpec,
    assign_treatment,
    build_event_panel,
    event_month,
    get_event,
    treatment_summary,
    write_spec,
)
from src.features.pipeline import OUT_FEATURES_PARQUET
from src.utils.config import REPORTS_DIR
from src.utils.io import write_json

TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--transport", default="data/interim/postcode_transport_features.csv")
    parser.add_argument("--event", default="E1")
    parser.add_argument("--radius-km", type=float, default=2.0)
    parser.add_argument("--control-min-km", type=float, default=3.0)
    parser.add_argument("--window-months", type=int, default=24)
    parser.add_argument("--outcome", default="log_median_unit_price")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    spec = EventStudySpec(event_id=args.event, radius_km=args.radius_km,
                          control_min_km=args.control_min_km,
                          window_months=args.window_months, outcome=args.outcome)

    transport = pd.read_csv(args.transport, dtype={"postcode": "string"})

    # Property-level distance distribution from the geocoded address sample. This
    # replaces the postcode centroid as the treatment variable: a postcode can
    # span kilometres, so its centroid mis-assigns homes near the boundary.
    geocode_features = None
    try:
        from src.data.events import _stations_at_event
        from src.data.geocode_features import (
            build_postcode_geocode_distances,
            compare_treatment_definitions,
        )
        geocode_features, geocode_report = build_postcode_geocode_distances(
            stations=_stations_at_event(spec.event_id))
        print("Geocoded property distances:")
        print(f"  addresses used        : {geocode_report.points_used:,}")
        print(f"  station entrances used: {geocode_report.entrances_used}")
        print(f"  postcodes kept        : {geocode_report.postcodes_kept}"
              f" (dropped {geocode_report.postcodes_dropped_few_points} with < "
              f"{spec.min_geocoded_points} addresses)")
        comparison = compare_treatment_definitions(transport, geocode_features,
                                                   radius_km=spec.radius_km)
        disagree = comparison.loc[~comparison["agree"]]
        print(f"  arm disagreement vs centroid at {spec.radius_km} km: "
              f"{len(disagree)} of {len(comparison)} postcodes")
        if len(disagree):
            print(disagree[["postcode", "dist_metro", "dist_metro_p25",
                            "centroid_treated", "geocode_treated"]]
                  .head(10).to_string(index=False))
    except FileNotFoundError as exc:
        print(f"  geocoded distances unavailable ({exc}); falling back to the centroid.")

    treatment = assign_treatment(transport, spec, geocode_features=geocode_features)
    summary = treatment_summary(treatment, radius_km=spec.radius_km)
    print("\nTreatment assignment (pre-opening distance snapshot):")
    print(f"  treated postcodes ({spec.radius_km} km): {summary['treated_postcodes']} -> {summary['treated_list']}")
    print(f"  control postcodes (>= {spec.control_min_km} km, same SA4s): {summary['control_postcodes']}")
    print(f"  mean distance: treated {summary['treated_mean_dist_km']:.2f} km, "
          f"control {summary['control_mean_dist_km']:.2f} km")
    print(f"  distance source: {summary.get('distance_source_counts')}")
    if "postcodes_disagreeing_with_centroid" in summary:
        print(f"  postcodes where the geocoded definition disagrees with the centroid: "
              f"{summary['postcodes_disagreeing_with_centroid']} of {summary['postcodes_compared']}")

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    if args.sample and args.sample < len(frame):
        step = max(1, len(frame) // args.sample)
        frame = frame.sort_values("contract_date").iloc[::step]
        print(f"Smoke test on a {len(frame):,}-row chronological sample.")

    panel = build_event_panel(frame, treatment, spec)
    balanced = balance_panel(panel, value_col=spec.outcome)
    print(f"\nEvent panel: {len(panel):,} postcode-months, "
          f"{panel['postcode'].nunique()} postcodes, "
          f"{panel['month'].nunique()} months, window ±{spec.window_months}")
    print("Postcode-months per arm:")
    print(panel.groupby("group", observed=True).agg(
        postcodes=("postcode", "nunique"), months=("month", "nunique"),
        mean_sales=("n_sales", "mean"), median_unit_price=("median_unit_price", "median")).to_string())

    # --- main specification ------------------------------------------------
    main_result = estimate_event_study(balanced, spec.outcome,
                                       reference_period=spec.reference_period)
    print(f"\nMain specification ({spec.outcome}, ±{spec.window_months}m, ref {spec.reference_period}):")
    print(main_result.coefficients[["event_time", "coef", "se", "ci_low", "ci_high"]].to_string(index=False))
    print(f"\nPre-trend joint test: Wald {main_result.pre_trend.get('wald_statistic', float('nan')):.2f}, "
          f"dof {main_result.pre_trend.get('dof')}, "
          f"p = {main_result.pre_trend.get('p_value', float('nan')):.4f}")
    post_mean = main_result.coefficients.loc[
        main_result.coefficients["event_time"] >= 0, "coef"].mean()
    print(f"Mean post-period coefficient: {post_mean:+.4f} log10 points "
          f"(~{10 ** post_mean - 1:+.1%} on the median price per sqm)")

    # --- placebo tests -----------------------------------------------------
    print("\nPlacebos (both should be near zero):")
    shifted = placebo_shifted_event(balanced, spec.outcome, shift_months=36,
                                    reference_period=spec.reference_period)
    random_t = placebo_random_treatment(balanced, spec.outcome, n_draws=20,
                                        reference_period=spec.reference_period)
    for label, result in (("shifted event date", shifted), ("random treatment", random_t)):
        post = result.coefficients.loc[result.coefficients["event_time"] >= 0, "coef"]
        mean_post = post.mean() if len(post) else float("nan")
        print(f"  {label:20s} mean post coef {mean_post:+.4f} | "
              f"pre-trend p {result.pre_trend.get('p_value', float('nan')):.3f}")

    # --- robustness grid ---------------------------------------------------
    robustness_rows = []
    coefficients = {"main": main_result,
                    "placebo_shifted_date": shifted,
                    "placebo_random_treatment": random_t}

    for radius in spec.robustness_radii:
        variant = EventStudySpec(event_id=spec.event_id, radius_km=radius,
                                 control_min_km=spec.control_min_km,
                                 window_months=spec.window_months, outcome=spec.outcome)
        variant_treatment = assign_treatment(transport, variant)
        variant_panel = build_event_panel(frame, variant_treatment, variant)
        variant_balanced = balance_panel(variant_panel, value_col=variant.outcome)
        result = estimate_event_study(variant_balanced, variant.outcome,
                                      reference_period=variant.reference_period)
        label = f"radius_{radius:g}km"
        coefficients[label] = result
        robustness_rows.append({
            "specification": label,
            "radius_km": radius,
            "outcome": variant.outcome,
            "treated_postcodes": int(variant_treatment["is_treated"].sum()),
            "mean_post_coef": float(result.coefficients.loc[
                result.coefficients["event_time"] >= 0, "coef"].mean()),
            "pretrend_p_value": result.pre_trend.get("p_value", float("nan")),
            "n_obs": result.n_obs,
        })

    for outcome in spec.robustness_outcomes:
        result = estimate_event_study(balanced, outcome, reference_period=spec.reference_period)
        label = f"outcome_{outcome}"
        coefficients[label] = result
        robustness_rows.append({
            "specification": label,
            "radius_km": spec.radius_km,
            "outcome": outcome,
            "treated_postcodes": summary["treated_postcodes"],
            "mean_post_coef": float(result.coefficients.loc[
                result.coefficients["event_time"] >= 0, "coef"].mean()),
            "pretrend_p_value": result.pre_trend.get("p_value", float("nan")),
            "n_obs": result.n_obs,
        })

    robustness = pd.DataFrame(robustness_rows)
    main_row = {
        "specification": "main",
        "radius_km": spec.radius_km,
        "outcome": spec.outcome,
        "treated_postcodes": summary["treated_postcodes"],
        "mean_post_coef": float(post_mean),
        "pretrend_p_value": main_result.pre_trend.get("p_value", float("nan")),
        "n_obs": main_result.n_obs,
    }
    robustness = pd.concat([pd.DataFrame([main_row]), robustness], ignore_index=True)
    print("\nRobustness grid:")
    print(robustness.to_string(index=False))

    # --- continuous exposure (preferred design given cluster counts) -------
    print("\nDose-response specification (post x log(1+km), postcode + month FE):")
    dose_rows = []
    for km in (5.0, 10.0):
        row = dose_response(balanced, spec.outcome, treatment, max_km=km)
        dose_rows.append(row)
        print(f"  within {km:>4.0f} km: beta {row['coef_post_x_log_dist']:+.4f} "
              f"(se {row['se']:.4f}, p {row['p_value']:.3g}, "
              f"{row['n_postcodes']} postcodes, {row['n_obs']:,} obs) -> {row['interpretation']}")
    dose_table = pd.DataFrame(dose_rows)

    # --- persist -----------------------------------------------------------
    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        main_result.coefficients.to_csv(TABLES_DIR / "event_study_coefficients.csv", index=False)
        coefficients_to_wide(coefficients).to_csv(TABLES_DIR / "event_study_specifications.csv", index=False)
        robustness.to_csv(TABLES_DIR / "event_study_robustness.csv", index=False)
        dose_table.to_csv(TABLES_DIR / "event_study_dose_response.csv", index=False)
        pd.DataFrame([main_result.pre_trend]).to_csv(TABLES_DIR / "event_study_pretrend.csv", index=False)
        balanced.to_parquet(TABLES_DIR / "event_study_panel.parquet", index=False)

        placebo_rows = []
        for label, result in (("shifted_event_date", shifted), ("random_treatment", random_t)):
            post = result.coefficients.loc[result.coefficients["event_time"] >= 0, "coef"]
            placebo_rows.append({
                "placebo": label,
                "mean_post_coef": float(post.mean()) if len(post) else float("nan"),
                "max_abs_post_coef": float(post.abs().max()) if len(post) else float("nan"),
                "pretrend_p_value": result.pre_trend.get("p_value",
                                                         result.pre_trend.get("p_value_median", float("nan"))),
                "pretrend_p_share_below_0_05": result.pre_trend.get("p_value_share_below_0_05", float("nan")),
                "n_obs": result.n_obs,
            })
        pd.DataFrame(placebo_rows).to_csv(TABLES_DIR / "event_study_placebo.csv", index=False)

        make_event_plot(main_result, spec, FIGURES_DIR / "event_study_plot.png",
                        placebo=random_t)
        write_spec(spec)
        write_json(TABLES_DIR / "event_study_meta.json", {
            "spec": spec.to_dict(),
            "event": {k: str(v) for k, v in get_event(spec.event_id).to_dict().items()},
            "event_month": str(event_month(spec.event_id)),
            "treatment_summary": summary,
            "main": {
                "mean_post_coef": float(post_mean),
                "implied_pct_on_unit_price": float(10 ** post_mean - 1),
                "pretrend": main_result.pre_trend,
                "n_obs": main_result.n_obs,
                "n_postcodes": main_result.n_postcodes,
            },
            "dose_response": dose_rows,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"\nWrote tables to {TABLES_DIR}")
        print(f"Wrote {FIGURES_DIR / 'event_study_plot.png'}")
    print(f"Elapsed: {time.time() - started:,.1f}s")
    return 0


def make_event_plot(result, spec, path: Path, placebo=None) -> None:
    """Event-study coefficient plot with 95% CI and the pre-trend verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result.coefficients.dropna(subset=["event_time"]).copy()
    table = table.loc[table["term"].str.startswith("et_")]
    # Insert the omitted reference period at zero.
    reference = pd.DataFrame([{"event_time": spec.reference_period, "coef": 0.0,
                               "ci_low": 0.0, "ci_high": 0.0, "term": "reference"}])
    table = pd.concat([table, reference], ignore_index=True).sort_values("event_time")

    accent, grey, ink, ink2 = "#2a78d6", "#b4b2ac", "#0b0b0b", "#6b6963"
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    ax.axvspan(table["event_time"].min() - 0.5, -0.5, color="#f5f4f1", zorder=0)
    ax.axhline(0, color=grey, lw=1.2, zorder=1)
    ax.axvline(-0.5, color="#d8d6d0", lw=1.4, zorder=1)
    ax.errorbar(table["event_time"], table["coef"],
                yerr=[table["coef"] - table["ci_low"], table["ci_high"] - table["coef"]],
                fmt="o", color=accent, ecolor=accent, elinewidth=1.3, capsize=2.5,
                markersize=4.5, zorder=4, label=f"{spec.outcome} (95% CI)")

    if placebo is not None:
        pt = placebo.coefficients.dropna(subset=["event_time"])
        pt = pt.loc[pt["term"].str.startswith("et_")]
        ax.plot(pt["event_time"], pt["coef"], "s--", color=grey, markersize=3,
                lw=1.2, alpha=0.9, zorder=3, label="placebo: random treatment")

    ax.set_xlabel("Months relative to Metro opening (26 May 2019)")
    ax.set_ylabel("Coefficient (log10 median price per sqm)")
    p_value = result.pre_trend.get("p_value", float("nan"))
    verdict = ("pre-trend test passes (p >= 0.05)" if p_value >= 0.05
               else "pre-trend test REJECTS parallel trends (p < 0.05)")
    ax.set_title(f"Event study: Metro Northwest opening\n{verdict} — joint pre-trend "
                 f"p = {p_value:.3f}, {result.n_postcodes} postcodes, "
                 f"{result.n_obs:,} postcode-months",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.012, 0.012,
             "Controls: same SA4 regions, >= 3 km from any new station. "
             "Clustered by postcode. Dashed placebo reassigns treatment randomly.",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, facecolor="white", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
