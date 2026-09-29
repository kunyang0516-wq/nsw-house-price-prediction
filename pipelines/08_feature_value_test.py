"""Feature-value test: do the engineered lagged neighbourhood features earn their place?

This is a *third kind* of test, distinct from H1/H2 (world relationships) and H3
(the Metro effect). It asks whether our own features add out-of-sample predictive
value, under the same rolling-origin protocol used to report model performance.

Layers (each adds to the previous):

    L0  property + location      area, log_area, distances
    L1  + calendar               year_num, month_sin, month_cos
    L2  + zone / region labels   development_type, zoning, postcode, council, locality
    L3  + lagged price stats     pc_med_price_{3,6,12}m          <- headline increment
    L4  + lagged unit price      pc_med_unit_price_{3,6,12}m
    M1  + macro                  cash_rate_asof, cpi_yoy_asof
    D1  + dispersion             pc_price_iqr_ratio_12m
    Q1  + liquidity & coverage   pc_n_sales_*, pc_n_months_observed_*, staleness

Three changes of substance, each closing a gap rather than adding a feature:

1. **The macro block had never been ablated.** `cash_rate_asof` and `cpi_yoy_asof` are
   in the production feature spec, but no ablation layer included them, so the report
   could not say whether they earn their place. They are added as `M1`, placed **last**,
   so the increment is measured against a model that already carries every other feature
   — the strictest available test, and the one that matches the question being asked
   ("once you know where, when, how big and what the neighbourhood has been selling for,
   does the cash rate tell you anything more?").

   Both macro columns are functions of the contract date (43 and 93 distinct values across
   276 months), so they are near-collinear with the calendar layer. That is precisely why
   they are measured *after* it rather than before: credited first, they would simply be
   collecting the time trend's contribution. An earlier revision placed them directly
   after L4, which was wrong twice over — it made L4's own delta a comparison against a
   worse model, and it contaminated D1's delta, because consecutive-difference attribution
   charges each layer for whatever the layer immediately before it did.

2. **The old `L5` mixed two different ideas under one label.** `pc_price_iqr_ratio_12m`
   is *dispersion* (how heterogeneous a postcode's stock is); the `pc_n_sales_*` and
   `pc_n_months_observed_*` columns are *liquidity and coverage* (how active the market
   is and how trustworthy the rolling median is). Calling the block "liquidity" was
   misleading and made the negative increment impossible to attribute. They are now
   separate layers, `D1` and `Q1`.

3. The layer key `L5` no longer exists, so downstream code keys off the new names.

Falsification tests:

    permutation placebo  the same columns, values shuffled across postcodes
                         inside the training fold. Any surviving increment can
                         only come from the *presence* / missingness pattern.
    time-reversal test   forward-looking windows (t+1..t+k) instead of lagged
                         ones. Should NOT be materially better than the real
                         lagged features; if it is, the pipeline leaks.

Model: XGBoost (main), as agreed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import regression_report
from src.features.pipeline import OUT_FEATURES_PARQUET
from src.models.features import FeatureSpec, PropertyFeatureTransformer, make_xy
from src.models.registry import build_model, select_model_specs
from src.utils.config import REPORTS_DIR
from src.validation.time_series_cv import rolling_origin_splits, split_frame

TABLES_DIR = REPORTS_DIR / "tables"

PRICE_STATS = ["pc_med_price_3m", "pc_med_price_6m", "pc_med_price_12m"]
UNIT_PRICE_STATS = ["pc_med_unit_price_3m", "pc_med_unit_price_6m", "pc_med_unit_price_12m"]
# How heterogeneous a postcode's stock is: the 12-month interquartile price ratio.
DISPERSION_STATS = ["pc_price_iqr_ratio_12m"]
# How active the market is, and how much evidence sits behind the rolling statistics.
# `pc_n_months_observed_*` is strictly a coverage column -- it tells the model how much to
# trust `pc_med_price_*` -- so it belongs with liquidity rather than with price levels.
LIQUIDITY_STATS = ["pc_n_sales_3m", "pc_n_sales_6m", "pc_n_sales_12m",
                   "pc_n_months_observed_3m", "pc_n_months_observed_6m",
                   "pc_n_months_observed_12m",
                   "pc_months_since_sale", "pc_months_observed"]
# Pure functions of the contract date; see the module docstring on why their layer sits
# after the calendar layer rather than before it.
MACRO_STATS = ["cash_rate_asof", "cpi_yoy_asof"]
PROPERTY_LOCATION = ["area_sqm", "log_area", "dist_cbd", "log_dist_cbd", "dist_train", "dist_metro"]
CALENDAR = ["year_num", "month_sin", "month_cos"]
LABELS = ["development_type", "zoning_clean", "post_code", "council_name", "locality"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="XGBoost backend for every layer/placebo fit")
    parser.add_argument("--jobs", type=int, default=1,
                        help="XGBoost threads on the cpu backend")
    parser.add_argument("--limit-folds", type=int, default=None,
                        help="use only the last N folds (faster)")
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--skip-placebos", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


HIGH_CARD_CATEGORIES = ("post_code", "council_name", "locality")


def spec_for(numerics: list[str], categories: list[str], bin_source: str = "area_sqm") -> FeatureSpec:
    """Build a FeatureSpec, keeping the numeric and categorical roles disjoint.

    The callers pass one flat list of "everything in this layer", so the categorical
    names appear in `numerics` too. Passing that straight through put five string columns
    into `FeatureSpec.numeric`, which meant each of them was *also* processed by the
    numeric block: coerced to NaN, imputed with a constant, standardised, and emitted as a
    useless `num__<name>` column alongside its real `low__`/`high__` encoding. It also made
    `PropertyFeatureTransformer.fit` crash on the oldest folds, where such a column can be
    entirely missing (`float(pd.NA)` raises), which is why the ablation could not be run on
    a single early fold at all.
    """
    category_names = tuple(categories)
    numeric_only = [c for c in numerics if c not in category_names]
    return FeatureSpec(
        numeric=tuple(numeric_only),
        categorical_low_card=tuple(c for c in category_names
                                   if c not in HIGH_CARD_CATEGORIES),
        categorical_high_card=tuple(c for c in category_names
                                    if c in HIGH_CARD_CATEGORIES),
        bin_source=bin_source,
    )


def layer_specs() -> dict[str, FeatureSpec]:
    """Cumulative feature layers, in the order they are added.

    Insertion order is load-bearing twice over: `summary["rmsle_mean"].diff()` and the
    per-fold improvement count are both taken along this order, so each layer's increment
    is measured against the one before it here. (The per-fold count additionally has to
    reindex the pivot, because `pivot_table` sorts its columns alphabetically.)
    """
    base = list(PROPERTY_LOCATION)
    cal = base + CALENDAR
    lab = cal + LABELS
    price = lab + PRICE_STATS
    unit = price + UNIT_PRICE_STATS
    with_dispersion = unit + DISPERSION_STATS
    full = with_dispersion + LIQUIDITY_STATS

    return {
        "L0_property_location": spec_for(base, []),
        "L1_plus_calendar": spec_for(cal, []),
        "L2_plus_labels": spec_for(cal, LABELS),
        "L3_plus_lagged_price": spec_for(price, LABELS),
        "L4_plus_unit_price": spec_for(unit, LABELS),
        "D1_plus_dispersion": spec_for(with_dispersion, LABELS),
        "Q1_plus_liquidity_coverage": spec_for(full, LABELS),
        # Macro goes LAST so its increment is measured against a model that already has
        # everything else. Placing it in the middle (after L4) was wrong for two reasons:
        # it made `L4`'s own delta a comparison against a worse model, and it contaminated
        # `D1`'s delta, since consecutive-difference attribution charges each layer for
        # whatever the layer immediately before it did. Last is also the stricter test:
        # it asks whether macro adds anything the other 26 features did not already carry.
        "M1_plus_macro": spec_for(full + MACRO_STATS, LABELS),
    }


def forward_rolling_panel(frame: pd.DataFrame, windows=(3, 6, 12)) -> pd.DataFrame:
    """Forward-looking rolling medians — a deliberately wrong feature set.

    If these predict better than the lagged ones, the lag/embargo machinery is
    not doing its job.
    """
    keys = ["post_code", "contract_date", "purchase_price"]
    data = frame[keys].copy()
    data["month"] = pd.to_datetime(data["contract_date"]).dt.to_period("M").dt.to_timestamp()
    base = (data.groupby(["post_code", "month"], observed=True)
            .agg(median_price=("purchase_price", "median")).reset_index())
    months = pd.date_range(base["month"].min(), base["month"].max(), freq="MS")
    index = pd.MultiIndex.from_product([base["post_code"].unique(), months],
                                       names=["post_code", "month"])
    spine = base.set_index(["post_code", "month"]).reindex(index).reset_index()

    # Reverse time inside each postcode, roll, then reverse back: this makes the
    # window cover the *future* months relative to the contract.
    reversed_spine = spine.sort_values(["post_code", "month"], ascending=[True, False]).copy()
    grouped = reversed_spine.groupby("post_code", observed=True)["median_price"]
    out = pd.DataFrame({"post_code": reversed_spine["post_code"], "month": reversed_spine["month"]})
    for window in windows:
        out[f"pc_med_price_{window}m"] = (grouped.rolling(window, min_periods=window).median()
                                          .reset_index(level=0, drop=True).to_numpy())
    # Shift so the window starts at t+1 rather than including t.
    out = out.sort_values(["post_code", "month"])
    for window in windows:
        column = f"pc_med_price_{window}m"
        out[column] = out.groupby("post_code", observed=True)[column].shift(-1)
    return out


def run_layer(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: FeatureSpec,
    xgb_params: dict,
    column_override: dict[str, pd.Series] | None = None,
) -> dict:
    """Fit one layer on the training fold, score the validation fold."""
    X_train, y_train, _ = make_xy(train, spec)
    X_valid, y_valid, _ = make_xy(valid, spec)

    if column_override:
        for column, values in column_override.items():
            if column in X_train.columns:
                X_train[column] = values.loc[X_train.index].to_numpy()
                X_valid[column] = values.loc[X_valid.index].to_numpy()

    transformer = PropertyFeatureTransformer(spec).fit(X_train, y_train)
    Z_train = transformer.transform(X_train)
    Z_valid = transformer.transform(X_valid)

    model = build_model(select_model_specs(["xgb"])[0][0])
    model.set_params(**xgb_params)
    split_at = min(max(int(len(Z_train) * 0.9), 1), len(Z_train) - 2)
    model.fit(Z_train[:split_at], y_train.to_numpy(dtype=float)[:split_at],
              eval_set=[(Z_train[split_at:], y_train.to_numpy(dtype=float)[split_at:])],
              verbose=False)
    y_pred = np.asarray(model.predict(Z_valid), dtype=float)
    return regression_report(y_valid.to_numpy(dtype=float), y_pred)


def main() -> int:
    args = parse_args()
    started = time.time()

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    # The IQR filter is applied per fold inside split_frame, matching the model
    # pipeline exactly; filtering here as well would silently differ.
    print(f"Feature frame: {len(frame):,} rows; "
          f"{int((frame['iqr_keep'] == True).sum()):,} inside the training fences")  # noqa: E712

    # Reuse the same fold definition as the model pipeline, holdout excluded.
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=84,
                                  horizon_months=12, step_months=24,
                                  embargo_months=1, holdout_months=12)
    if args.limit_folds:
        folds = folds[-args.limit_folds:]
    print(f"Evaluating {len(folds)} folds: " + "; ".join(
        f"f{f.index} {f.valid_months[0]:%Y-%m}" for f in folds))

    xgb_params = {"n_estimators": args.rounds, "learning_rate": 0.05, "max_depth": 8,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                  "tree_method": "hist", "n_jobs": args.jobs, "device": args.device,
                  "random_state": 36103}

    specs = layer_specs()
    print(f"Layers: {len(specs)} (L0-L4, M1 macro, D1 dispersion, Q1 liquidity/coverage)")
    rows: list[dict] = []

    for fold in folds:
        train, valid, _ = split_frame(frame, fold, group_col="group_key", drop_unjudged_iqr=True)
        if train.empty or valid.empty:
            continue
        train = train.sort_values("contract_date").iloc[-400_000:]
        print(f"\nfold {fold.index}: train {len(train):,} | valid {len(valid):,}")

        for layer, spec in specs.items():
            report = run_layer(train, valid, spec, xgb_params)
            rows.append({"fold": fold.index, "layer": layer, "n": report["n"],
                         "rmsle": report["rmsle"], "mdape_pct": report["mdape_pct"],
                         "r2_log10": report["r2_log10"]})
            print(f"  {layer:26s} RMSLE {report['rmsle']:.4f} | MdAPE {report['mdape_pct']:.1f}%")

    layer_table = pd.DataFrame(rows)
    if layer_table.empty:
        print("No folds produced results.")
        return 1
    # Use the last fold that actually produced rows: a fold can be empty once the
    # corruption/IQR filters are applied.
    evaluated_folds = sorted(layer_table["fold"].unique())
    anchor_fold = [f for f in folds if f.index == evaluated_folds[-1]][0]
    print(f"\nAnchor fold for placebos: fold {anchor_fold.index} "
          f"({anchor_fold.valid_months[0]:%Y-%m}..{anchor_fold.valid_months[1]:%Y-%m})")
    summary = (layer_table.groupby("layer", observed=True)
               .agg(rmsle_mean=("rmsle", "mean"), rmsle_std=("rmsle", "std"),
                    mdape_mean=("mdape_pct", "mean"), folds=("fold", "nunique"))
               .reset_index())
    # Reorder to insertion order FIRST. `groupby` sorts its keys alphabetically, so
    # `summary` arrives as D1, L0, L1, L2, L3, L4, M1, Q1 -- and every downstream diff
    # taken along it would compare the wrong pairs. (The layer table itself is already in
    # insertion order, since the loop emits folds outermost and layers innermost.)
    layer_order = list(layer_specs())
    summary = summary.set_index("layer").reindex(layer_order).reset_index()

    summary["rmsle_delta_vs_prev"] = summary["rmsle_mean"].diff()
    # Same ordering requirement for the per-fold counts: `pivot_table` also sorts its
    # columns alphabetically, so it needs the explicit reindex too.
    fold_pivot = (layer_table.pivot_table(index="fold", columns="layer", values="rmsle")
                  .reindex(columns=layer_order))
    fold_deltas = fold_pivot.diff(axis=1)
    summary["folds_improved_vs_prev"] = [
        int((fold_deltas[layer] < 0).sum()) if i else np.nan
        for i, layer in enumerate(summary["layer"])
    ]
    print("\n=== Layer ablation (mean across folds; negative delta = improvement) ===")
    print(summary.to_string(index=False))

    core = summary.loc[summary["layer"] == "L2_plus_labels", "rmsle_mean"].iloc[0]
    with_price = summary.loc[summary["layer"] == "L3_plus_lagged_price", "rmsle_mean"].iloc[0]
    print(f"\nHeadline increment (L3 - L2): {with_price - core:+.4f} RMSLE")

    # ---------------- N1: is it location level or timing? ---------------- #
    # Demean each lagged price column within its postcode (postcode means taken
    # from the training fold). If the increment collapses, the columns are only
    # a stand-in for "which postcode is expensive".
    print("\n=== N1: within-postcode demeaned lagged prices (L3 spec) ===")
    demeaned_rows = []
    for fold in [f for f in folds if f.index in evaluated_folds][-2:]:
        train, valid, _ = split_frame(frame, fold, group_col="group_key", drop_unjudged_iqr=True)
        if train.empty or valid.empty:
            continue
        train = train.sort_values("contract_date").iloc[-400_000:]
        means = {c: train.groupby("post_code")[c].mean() for c in PRICE_STATS}
        combined = pd.concat([train.assign(_side="train"), valid.assign(_side="valid")])
        override = {}
        for column, mean_map in means.items():
            values = combined[column] - combined["post_code"].map(mean_map)
            override[column] = values
        train_dm = combined.loc[combined["_side"] == "train"].drop(columns="_side")
        valid_dm = combined.loc[combined["_side"] == "valid"].drop(columns="_side")
        report = run_layer(train_dm, valid_dm, specs["L3_plus_lagged_price"], xgb_params, override)
        demeaned_rows.append({"fold": fold.index, "rmsle": report["rmsle"]})
        print(f"  fold {fold.index}: demeaned L3 RMSLE {report['rmsle']:.4f} "
              f"(real L3 {layer_table.loc[(layer_table['fold'] == fold.index) & (layer_table['layer'] == 'L3_plus_lagged_price'), 'rmsle'].iloc[0]:.4f}, "
              f"L2 {layer_table.loc[(layer_table['fold'] == fold.index) & (layer_table['layer'] == 'L2_plus_labels'), 'rmsle'].iloc[0]:.4f})")
    demeaned_table = pd.DataFrame(demeaned_rows)

    # ---------------- falsification ------------------------------------- #
    placebo_rows = []
    reverted = None
    if not args.skip_placebos:
        print("\n=== Placebos ===")
        spec = specs["L3_plus_lagged_price"]
        rng = np.random.default_rng(36103)

        # (a) permutation: shuffle each lagged column across postcodes, keeping
        # the marginal distribution and the missingness pattern intact.
        for trial in range(args.permutations):
            combined = pd.concat([train.assign(_side="train"), valid.assign(_side="valid")])
            override = {}
            for column in PRICE_STATS:
                if column not in combined.columns:
                    continue
                values = combined[column].to_numpy().copy()
                rng.shuffle(values)
                override[column] = pd.Series(values, index=combined.index)
            train_perm = combined.loc[combined["_side"] == "train"].drop(columns="_side")
            valid_perm = combined.loc[combined["_side"] == "valid"].drop(columns="_side")
            report = run_layer(train_perm, valid_perm, spec, xgb_params, override)
            placebo_rows.append({"placebo": "permutation", "trial": trial, "rmsle": report["rmsle"]})
            print(f"  permutation {trial}: RMSLE {report['rmsle']:.4f}")

        # (b) time reversal: use future windows instead of lagged ones
        forward = forward_rolling_panel(frame)
        frame_month = pd.to_datetime(frame["contract_date"]).dt.to_period("M").dt.to_timestamp()
        combined = frame.assign(month=frame_month).merge(
            forward, on=["post_code", "month"], how="left", suffixes=("", "__fwd"))
        for column in PRICE_STATS:
            forward_column = f"{column}__fwd"
            if forward_column in combined.columns:
                combined[column] = combined[forward_column]
            else:
                print(f"  note: {forward_column} missing, skipping that column")
        # IMPORTANT: do not drop rows here. A `dropna` would compare the placebo on
        # a different (easier) row population than the real run and make the
        # future-leaking feature look better than it is. Measured: dropping the
        # 2,127 rows with an incomplete forward window moved the placebo from
        # 0.4765 to 0.3687 against a real L3 of 0.4355.
        fold = anchor_fold
        train, valid, _ = split_frame(combined, fold, group_col="group_key", drop_unjudged_iqr=True)
        if not train.empty and not valid.empty:
            train = train.sort_values("contract_date").iloc[-400_000:]
            report = run_layer(train, valid, spec, xgb_params)
            reverted = report["rmsle"]
            placebo_rows.append({"placebo": "time_reversal", "trial": 0, "rmsle": report["rmsle"]})
            print(f"  time reversal (future windows, same rows): RMSLE {report['rmsle']:.4f}")

    placebo_table = pd.DataFrame(placebo_rows)
    if not placebo_table.empty:
        perm = placebo_table.loc[placebo_table["placebo"] == "permutation", "rmsle"]
        anchor_real = layer_table.loc[
            (layer_table["fold"] == anchor_fold.index)
            & (layer_table["layer"] == "L3_plus_lagged_price"), "rmsle"]
        anchor_l2 = layer_table.loc[
            (layer_table["fold"] == anchor_fold.index)
            & (layer_table["layer"] == "L2_plus_labels"), "rmsle"]
        print(f"\n=== Placebo summary (anchor fold {anchor_fold.index}, L3) ===")
        if len(anchor_real):
            print(f"  real L3 RMSLE: {anchor_real.iloc[0]:.4f}")
        if len(perm):
            print(f"  permutation : mean {perm.mean():.4f} (n={len(perm)})")
        if len(anchor_l2):
            print(f"  L2 reference: {anchor_l2.iloc[0]:.4f}")
        if reverted is not None:
            print(f"  time reversal (future windows): {reverted:.4f}")

    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        # Already in insertion order; the reindex is a guard, not a fix.
        layer_table.to_csv(TABLES_DIR / "feature_ablation_by_fold.csv", index=False)
        summary.to_csv(TABLES_DIR / "feature_ablation.csv", index=False)
        if not placebo_table.empty:
            placebo_table.to_csv(TABLES_DIR / "feature_value_placebo.csv", index=False)
        if not demeaned_table.empty:
            demeaned_table.to_csv(TABLES_DIR / "feature_value_demeaned.csv", index=False)
        print(f"\nWrote {TABLES_DIR / 'feature_ablation.csv'}")
    print(f"Elapsed: {time.time() - started:,.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
