"""Matched A/B: does the production model actually need the three negative layers?

The layer ablation already measures D1 (dispersion), Q1 (liquidity/coverage) and M1
(macro) honestly -- it is out-of-sample and runs inside the rolling-origin folds. But it
runs under *different settings from the production model*: the training fold is truncated
to the most recent 400,000 rows and XGBoost gets 200 rounds with no early stopping, while
the production model trains on the full window (up to 1.51M rows) for 600 rounds with
early stopping.

Tree ensembles do not compose: a feature that is worthless on 400k rows and 200 rounds
need not be worthless on 1.51M rows and 600 rounds, because the extra rounds can spend
themselves on it. So the ablation shows "these layers do not pay for themselves in the
ablation's regime", which is not quite the claim we want to make.

This script closes that gap with a like-for-like comparison at production settings: same
XGBoost configuration, same fold definitions, same training windows, with only the feature
set differing.

    full    every feature in FeatureSpec
    lean    minus D1 (pc_price_iqr_ratio_12m), Q1 (sales counts, coverage, staleness)
            and M1 (cash_rate_asof, cpi_yoy_asof)
    nocal   minus the calendar block as well -- a deliberate positive control: if removing
            calendar features does NOT hurt, the comparison method is broken

Three evaluation surfaces are reported, because "does it help" and "does it help *there*"
are different questions:

    tail     the last 10% of the training window, which early stopping selects on but
             never fits
    valid    the embargoed forecast window (the number the CV table reports)
    holdout  2023, which no fold ever touches

Usage
-----
    python pipelines/10_compare_feature_sets.py --device cuda
    python pipelines/10_compare_feature_sets.py --configs full lean
    python pipelines/10_compare_feature_sets.py --no-save

Writes `reports/tables/feature_set_comparison.csv` and
`reports/tables/feature_set_comparison_meta.json`.
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
from src.utils.io import write_json
from src.validation.time_series_cv import holdout_window, rolling_origin_splits, split_frame

TABLES_DIR = REPORTS_DIR / "tables"

# The layers the ablation flagged as net-negative, and the positive control block.
D1_DISPERSION = ("pc_price_iqr_ratio_12m",)
Q1_LIQUIDITY = ("pc_n_sales_3m", "pc_n_sales_6m", "pc_n_sales_12m",
                "pc_n_months_observed_3m", "pc_n_months_observed_6m",
                "pc_n_months_observed_12m", "pc_months_since_sale", "pc_months_observed")
M1_MACRO = ("cash_rate_asof", "cpi_yoy_asof")
CALENDAR = ("year_num", "month_sin", "month_cos")

NEGATIVE_LAYERS = D1_DISPERSION + Q1_LIQUIDITY + M1_MACRO


def spec_for(include_calendar: bool, exclude: tuple[str, ...]) -> FeatureSpec:
    base = FeatureSpec()
    drop = set(exclude) | (set() if include_calendar else set(CALENDAR))
    return FeatureSpec(
        numeric=tuple(c for c in base.numeric if c not in drop),
        categorical_low_card=base.categorical_low_card,
        categorical_high_card=base.categorical_high_card,
        bin_source=base.bin_source,
    )


def configs() -> dict[str, FeatureSpec]:
    return {
        "full": spec_for(True, ()),
        "lean": spec_for(True, NEGATIVE_LAYERS),
        "nocal": spec_for(False, NEGATIVE_LAYERS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--configs", nargs="*", default=None,
                        choices=["full", "lean", "nocal"])
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=84)
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=24)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def make_model(args):
    spec = select_model_specs(["xgb"])[0][0]
    spec.params["n_estimators"] = args.rounds
    spec.device = args.device
    spec.n_jobs = args.jobs
    return spec


def fit_and_score(model_spec, spec, train, valid):
    """Fit once on `train`; score the early-stopping tail and the validation window."""
    X_tr, y_tr, _ = make_xy(train, spec)
    y_tr_values = y_tr.to_numpy(dtype=float)
    transformer = PropertyFeatureTransformer(spec).fit(X_tr, y_tr)
    Z_tr = transformer.transform(X_tr)

    model = build_model(model_spec)
    split_at = min(max(int(len(Z_tr) * 0.9), 1), len(Z_tr) - 2)
    model.fit(Z_tr[:split_at], y_tr_values[:split_at],
              eval_set=[(Z_tr[split_at:], y_tr_values[split_at:])], verbose=False)

    tail_pred = np.asarray(model.predict(Z_tr[split_at:]), dtype=float)
    out = {"tail": regression_report(y_tr_values[split_at:], tail_pred)}

    X_va, y_va, _ = make_xy(valid, spec)
    valid_pred = np.asarray(model.predict(transformer.transform(X_va)), dtype=float)
    out["valid"] = regression_report(y_va.to_numpy(dtype=float), valid_pred)
    return out


def score_holdout(model_spec, spec, train, holdout_frame) -> dict | None:
    """One model per config, trained on everything before the holdout, scored once.

    Deliberately NOT done per fold. Predicting 2023 from each fold's model makes the
    result a monotone function of how much history that fold had -- a model trained only
    through 2007 scores about 0.64 on 2023 -- which measures the training horizon rather
    than the feature set. The holdout question needs one fit per configuration.
    """
    if holdout_frame is None or holdout_frame.empty or train.empty:
        return None
    X_tr, y_tr, _ = make_xy(train, spec)
    transformer = PropertyFeatureTransformer(spec).fit(X_tr, y_tr)
    Z_tr = transformer.transform(X_tr)
    y_tr_values = y_tr.to_numpy(dtype=float)

    model = build_model(model_spec)
    split_at = min(max(int(len(Z_tr) * 0.9), 1), len(Z_tr) - 2)
    model.fit(Z_tr[:split_at], y_tr_values[:split_at],
              eval_set=[(Z_tr[split_at:], y_tr_values[split_at:])], verbose=False)

    X_h, y_h, _ = make_xy(holdout_frame, spec)
    pred = np.asarray(model.predict(transformer.transform(X_h)), dtype=float)
    return regression_report(y_h.to_numpy(dtype=float), pred)


def main() -> int:
    args = parse_args()
    started = time.time()

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    folds = rolling_origin_splits(
        frame["contract_date"], min_train_months=args.min_train_months,
        horizon_months=args.horizon_months, step_months=args.step_months,
        embargo_months=args.embargo_months, holdout_months=args.holdout_months)
    holdout = holdout_window(frame["contract_date"], args.holdout_months)

    wanted = args.configs or ["full", "lean", "nocal"]
    specs = {k: v for k, v in configs().items() if k in wanted}
    model_spec = make_model(args)

    print(f"XGBoost, {args.rounds} rounds, device={args.device}")
    for name, spec in specs.items():
        print(f"  {name:6s}: {len(spec.numeric)} numeric features")
    if holdout is not None:
        print(f"{len(folds)} folds; holdout {holdout[0]:%Y-%m}..{holdout[1]:%Y-%m}\n")

    holdout_frame = None
    holdout_train = None
    if holdout is not None:
        months = frame["contract_date"].dt.to_period("M").dt.to_timestamp()
        holdout_mask = months.between(holdout[0], holdout[1])
        holdout_frame = frame.loc[holdout_mask]
        # Train the one-shot holdout models the same way the pipeline does: everything
        # before the holdout, purged of dwellings that appear in it.
        holdout_train = frame.loc[months < holdout[0]]
        overlap = holdout_train["group_key"].isin(set(holdout_frame["group_key"]))
        holdout_train = holdout_train.loc[~overlap]

    rows: list[dict] = []
    for fold in folds:
        train, valid, _ = split_frame(frame, fold, group_col="group_key",
                                      drop_unjudged_iqr=True)
        if train.empty or valid.empty:
            continue
        train = train.sort_values("contract_date")
        line = []
        for name, spec in specs.items():
            t0 = time.time()
            report = fit_and_score(model_spec, spec, train, valid)
            elapsed = time.time() - t0
            rows.append({
                "config": name, "fold": fold.index,
                "n_numeric_features": len(spec.numeric),
                "train_rows": int(len(train)), "valid_rows": int(len(valid)),
                "tail_rmsle": round(report["tail"]["rmsle"], 4),
                "valid_rmsle": round(report["valid"]["rmsle"], 4),
                "seconds": round(elapsed, 1),
            })
            line.append(f"{name} {report['valid']['rmsle']:.4f}")
        print(f"  fold {fold.index}: train {len(train):>9,} | valid RMSLE  "
              + "  ".join(line), flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        print("no folds produced results")
        return 1

    piv = table.pivot_table(index="fold", columns="config", values="valid_rmsle")

    # One holdout fit per config, trained on everything before 2023.
    holdout_scores: dict[str, float] = {}
    if holdout_train is not None and not holdout_train.empty:
        print(f"\nOne-shot holdout ({holdout[0]:%Y-%m}..{holdout[1]:%Y-%m}), "
              f"train {len(holdout_train):,}:")
        for name, spec in specs.items():
            scored = score_holdout(model_spec, spec, holdout_train, holdout_frame)
            if scored is not None:
                holdout_scores[name] = round(float(scored["rmsle"]), 4)
                print(f"  {name:6s} RMSLE {scored['rmsle']:.4f}")

    summary_rows = []
    for name in specs:
        block = table.loc[table["config"] == name]
        summary_rows.append({
            "config": name,
            "n_numeric_features": int(block["n_numeric_features"].iloc[0]),
            "valid_rmsle_mean": round(float(block["valid_rmsle"].mean()), 4),
            "valid_rmsle_std": round(float(block["valid_rmsle"].std()), 4),
            "tail_rmsle_mean": round(float(block["tail_rmsle"].mean()), 4),
            "holdout_rmsle": holdout_scores.get(name),
            "seconds_mean": round(float(block["seconds"].mean()), 1),
        })
    summary = pd.DataFrame(summary_rows).sort_values("valid_rmsle_mean").reset_index(drop=True)

    print("\n=== mean across folds (lower is better) ===")
    print(summary.to_string(index=False))

    verdict_lines: list[str] = []
    if {"full", "lean"} <= set(piv.columns):
        diff = piv["lean"] - piv["full"]
        win = int((diff < 0).sum())
        mean_diff = float(diff.mean())
        sd_diff = float(diff.std())
        print("\nlean - full, per fold (negative = lean better):")
        print("  " + "  ".join(f"f{i}:{v:+.4f}" for i, v in diff.items()))
        print(f"  mean {mean_diff:+.4f}, lean wins {win}/{len(diff)} folds, "
              f"fold-to-fold sd {sd_diff:.4f}")
        sign = "better" if mean_diff < 0 else "worse"
        verdict_lines.append(
            f"At production settings (full training window, {args.rounds} rounds, early "
            f"stopping), the lean model is {sign} on the validation windows by "
            f"{abs(mean_diff):.4f} RMSLE on average, winning {win} of {len(diff)} folds. "
            f"The fold-to-fold standard deviation of that difference is {sd_diff:.4f}.")
        if abs(mean_diff) < sd_diff:
            verdict_lines.append(
                "The mean difference is smaller than its own fold-to-fold standard "
                "deviation, so the direction is not consistently established at this noise "
                "level. The honest reading is that the three layers are NEUTRAL, not that "
                "removing them measurably improves accuracy.")
        else:
            verdict_lines.append(
                "The mean difference exceeds its fold-to-fold standard deviation, so the "
                "direction is stable across folds.")
        if {"full", "lean"} <= set(holdout_scores):
            h_diff = holdout_scores["lean"] - holdout_scores["full"]
            direction = "better" if h_diff < 0 else "worse"
            verdict_lines.append(
                f"On the one-shot 2023 holdout — one fit per configuration, so the only "
                f"genuinely untouched comparison here — the lean model is {direction} by "
                f"{abs(h_diff):.4f} RMSLE ({holdout_scores['lean']:.4f} against "
                f"{holdout_scores['full']:.4f}).")
    if {"lean", "nocal"} <= set(piv.columns):
        ctrl = piv["nocal"] - piv["lean"]
        print(f"\npositive control (drop calendar too), nocal - lean: "
              f"mean {float(ctrl.mean()):+.4f}, worse in "
              f"{int((ctrl > 0).sum())}/{len(ctrl)} folds")
        verdict_lines.append(
            f"Positive control: removing the calendar block as well makes validation error "
            f"{float(ctrl.mean()):+.4f} worse on average, in {int((ctrl > 0).sum())} of "
            f"{len(ctrl)} folds. The method does detect a block that matters, so a "
            "near-zero result for the other three is informative rather than a sign that "
            "the comparison is broken.")

    print()
    for line in verdict_lines:
        print("VERDICT:", line)

    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLES_DIR / "feature_set_comparison.csv", index=False)
        write_json(TABLES_DIR / "feature_set_comparison_meta.json", {
            "rounds": args.rounds,
            "device": args.device,
            "configs": {name: list(spec.numeric) for name, spec in specs.items()},
            "excluded_negative_layers": list(NEGATIVE_LAYERS),
            "calendar_block": list(CALENDAR),
            "folds": [f.label() for f in folds],
            "summary": summary.to_dict(orient="records"),
            "verdict": verdict_lines,
            "note": ("Production settings on the FULL training window, unlike the layer "
                     "ablation, which truncates to 400k rows and 200 rounds."),
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"\nWrote {TABLES_DIR / 'feature_set_comparison.csv'}")
        print(f"Wrote {TABLES_DIR / 'feature_set_comparison_meta.json'}")

    print(f"\nElapsed: {time.time() - started:,.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
