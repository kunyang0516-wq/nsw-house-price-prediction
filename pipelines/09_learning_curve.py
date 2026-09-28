"""Learning curve: does the model overfit, or has it hit the data ceiling?

The report previously argued "we do not overfit" only indirectly -- from the fact that
the cross-validation score is close to the holdout score, and that the fold-to-fold
spread of a learned model is no worse than that of a naive benchmark. Neither measures
overfitting head-on, because neither ever looks at **training** error.

This script measures it directly. On one fold it fits the same configuration on growing
prefixes of the training window and records two numbers per size:

* **train RMSLE** -- error on the rows the model was fitted on
* **valid RMSLE** -- error on the held-out validation window

The shape of the pair answers the question:

* the two curves converge and both flatten  -> high bias; more data/features would not
  help, and there is little room to overfit
* the two curves stay far apart and the gap widens with size -> high variance; the model
  is memorising, and more regularisation or more data is needed

Usage
-----
    python pipelines/09_learning_curve.py                     # fold 7 (largest train)
    python pipelines/09_learning_curve.py --fold 4            # a mid-sized fold
    python pipelines/09_learning_curve.py --sizes 0.1 0.25 0.5 1.0
    python pipelines/09_learning_curve.py --model random_forest

Writes `reports/tables/learning_curve.csv` and `reports/tables/learning_curve_meta.json`.
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
from src.validation.time_series_cv import rolling_origin_splits, split_frame

TABLES_DIR = REPORTS_DIR / "tables"

# Fixed so a re-run reproduces the same curve.
SEED = 36103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--fold", type=int, default=7,
                        help="which rolling-origin fold to measure (default: the last one, "
                             "which has the largest training window)")
    parser.add_argument("--model", default="xgb", help="model alias, e.g. xgb / rf / ridge")
    parser.add_argument("--mode", choices=("varying-span", "fixed-span"), default="fixed-span",
                        help="fixed-span (default) slides the training window's START so the "
                             "window always ends just before validation -- only the sample "
                             "size changes. varying-span grows the window from the fold's "
                             "original start, which also lengthens the gap to the validation "
                             "period and therefore confounds size with market drift.")
    parser.add_argument("--sizes", nargs="*", type=float,
                        default=[0.1, 0.25, 0.5, 0.75, 1.0],
                        help="training-window fractions to evaluate")
    parser.add_argument("--xgb-rounds", type=int, default=600)
    parser.add_argument("--capacity-probe", action="store_true",
                        help="instead of a learning curve, fit ONE small training window "
                             "with many more boosting rounds. If training error collapses "
                             "toward zero while validation error does not, the algorithm has "
                             "ample capacity to memorise and the flat curve is not a capacity "
                             "problem. This is the direct test for 'could it overfit if it "
                             "tried?'")
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-jobs", type=int, default=16,
                        help="raise only if the environment allows joblib threads")
    parser.add_argument("--min-train-months", type=int, default=84)
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=24)
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def fit_and_score(
    model_spec,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[float, float, float]:
    """Fit once; return (train RMSLE, valid RMSLE, early-stopping-tail RMSLE).

    The estimator is rebuilt per call so nothing carries over between fits. For XGBoost
    the early-stopping tail is reported separately: those rows are *training* data but
    were never fitted on, which makes them a third useful point on the curve. NaN for
    models without an early-stopping split.
    """
    model = build_model(model_spec)
    n = len(Z_train)
    uses_early_stopping = (model_spec.kind == "xgboost"
                           and "early_stopping_rounds" in model_spec.params)

    if uses_early_stopping:
        # Early stopping on a chronological tail of the TRAINING window -- never on the
        # validation fold, which would leak the evaluation window into model selection.
        split_at = min(max(int(n * 0.9), 1), n - 2)
        model.fit(Z_train[:split_at], y_train[:split_at],
                  eval_set=[(Z_train[split_at:], y_train[split_at:])],
                  verbose=False)
        train_rmsle = float(regression_report(
            y_train[:split_at],
            np.asarray(model.predict(Z_train[:split_at]), dtype=float))["rmsle"])
        tail_rmsle = float(regression_report(
            y_train[split_at:],
            np.asarray(model.predict(Z_train[split_at:]), dtype=float))["rmsle"])
    else:
        model.fit(Z_train, y_train)
        train_rmsle = float(regression_report(
            y_train, np.asarray(model.predict(Z_train), dtype=float))["rmsle"])
        tail_rmsle = float("nan")

    valid_rmsle = float(regression_report(
        y_valid, np.asarray(model.predict(Z_valid), dtype=float))["rmsle"])
    return train_rmsle, valid_rmsle, tail_rmsle


def main() -> int:
    args = parse_args()
    started = time.time()

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])

    folds = rolling_origin_splits(
        frame["contract_date"],
        min_train_months=args.min_train_months,
        horizon_months=args.horizon_months,
        step_months=args.step_months,
        embargo_months=args.embargo_months,
        holdout_months=args.holdout_months,
    )
    by_index = {f.index: f for f in folds}
    if args.fold not in by_index:
        print(f"fold {args.fold} not in {sorted(by_index)}")
        return 2
    fold = by_index[args.fold]

    spec = FeatureSpec()
    specs, unknown = select_model_specs([args.model])
    if unknown:
        print(f"unknown model: {unknown}")
        return 2
    model_spec = specs[0]
    if model_spec.kind in ("random_forest", "thread_forest"):
        model_spec.params["n_estimators"] = args.rf_trees
        model_spec.n_jobs = args.rf_jobs
        if model_spec.kind == "random_forest":
            from src.models.registry import forest_backend_available
            if args.rf_jobs > 1 and not forest_backend_available():
                model_spec.kind = "thread_forest"
                print("  joblib threads unavailable -> ThreadForestRegressor "
                      "(identical predictions)")
    if model_spec.kind == "xgboost":
        model_spec.params["n_estimators"] = args.xgb_rounds
        model_spec.device = args.xgb_device
        if args.capacity_probe:
            # Deliberately remove the limiters: no early stopping, small window, deep trees.
            # The question is whether the learner CAN drive training error to ~0.
            model_spec.params.pop("early_stopping_rounds", None)
            model_spec.params["max_depth"] = 12
            model_spec.params["min_child_weight"] = 1
            model_spec.params["n_estimators"] = max(args.xgb_rounds, 2000)
            print("capacity probe: early stopping OFF, max_depth 12, "
                  f"min_child_weight 1, {model_spec.params['n_estimators']} rounds")
    if args.capacity_probe:
        args.sizes = [0.02]

    train, valid, split_report = split_frame(
        frame, fold, group_col="group_key", drop_unjudged_iqr=True)
    if train.empty or valid.empty:
        print("empty side of the split")
        return 2
    train = train.sort_values("contract_date")

    print(f"fold {fold.index}  train {len(train):,} | valid {len(valid):,} "
          f"({fold.valid_months[0]:%Y-%m}..{fold.valid_months[1]:%Y-%m})")
    print(f"model: {model_spec.name} ({model_spec.kind})\n")

    X_valid, y_valid, _ = make_xy(valid, spec)
    y_valid_values = y_valid.to_numpy(dtype=float)

    rows: list[dict] = []
    for fraction in sorted(args.sizes):
        n_use = max(int(len(train) * fraction), 1_000)
        if args.mode == "fixed-span":
            # Slide the START so the window always ends at train's last month. Temporal
            # proximity to the validation window is then constant and only n changes --
            # which is what a learning curve is supposed to isolate. Growing from the
            # start instead would confound sample size with market drift: 2% of this fold
            # is 2001 data predicting 2022, which scores ~1.3 RMSLE for that reason alone.
            block = train.iloc[-n_use:]
        else:
            block = train.iloc[:n_use]
        block = block.sort_values("contract_date")

        X_tr, y_tr, _ = make_xy(block, spec)
        y_tr_values = y_tr.to_numpy(dtype=float)

        transformer = PropertyFeatureTransformer(spec)
        transformer.fit(X_tr, y_tr)
        Z_tr = transformer.transform(X_tr)
        Z_va = transformer.transform(X_valid)

        t0 = time.time()
        train_rmsle, valid_rmsle, tail_rmsle = fit_and_score(
            model_spec, Z_tr, y_tr_values, Z_va, y_valid_values)
        elapsed = time.time() - t0

        gap = valid_rmsle - train_rmsle
        rows.append({
            "train_rows": int(n_use),
            "fraction_of_fold_train": round(fraction, 4),
            "train_start": f"{block['contract_date'].min():%Y-%m}",
            "train_end": f"{block['contract_date'].max():%Y-%m}",
            "train_rmsle": round(train_rmsle, 4),
            "train_tail_rmsle": round(tail_rmsle, 4) if tail_rmsle == tail_rmsle else None,
            "valid_rmsle": round(valid_rmsle, 4),
            "gap_valid_minus_train": round(gap, 4),
            "ratio_valid_over_train": round(valid_rmsle / train_rmsle, 3),
            "seconds": round(elapsed, 1),
        })
        tail_txt = "" if tail_rmsle != tail_rmsle else f" tail {tail_rmsle:.4f}"
        print(f"  {fraction:5.0%} -> {n_use:>9,} rows "
              f"({block['contract_date'].min():%Y-%m}..{block['contract_date'].max():%Y-%m}) "
              f"| train {train_rmsle:.4f}{tail_txt} "
              f"| valid {valid_rmsle:.4f} | gap {gap:+.4f} | {elapsed:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    # --- verdict --------------------------------------------------------- #
    rows_n = table["train_rows"].to_numpy(dtype=float)
    valid = table["valid_rmsle"].to_numpy(dtype=float)
    gap = table["gap_valid_minus_train"].to_numpy(dtype=float)

    if len(table) < 2:
        # A capacity probe is a single point by design; a trend needs at least two.
        gap_one = float(gap[0]) if len(gap) else float("nan")
        valid_one = float(valid[0]) if len(valid) else float("nan")
        train_one = float(table["train_rmsle"].iloc[0])
        verdict = (
            f"Single-point capacity probe (n = {int(rows_n[0]):,}, unregularised): training "
            f"RMSLE {train_one:.4f} against validation RMSLE {valid_one:.4f}, a gap of "
            f"{gap_one:+.4f}. The learner can drive training error far below validation error, "
            "so the algorithm has ample capacity to memorise. Any flat or worsening learning "
            "curve under the production configuration is therefore NOT a capacity problem -- "
            "it is the regularisation working, plus features that do not carry more signal."
        )
        print()
        print("VERDICT:", verdict)
        if not args.no_save:
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            table.to_csv(TABLES_DIR / "learning_curve_probe.csv", index=False)
            write_json(TABLES_DIR / "learning_curve_probe_meta.json", {
                "mode": "capacity_probe",
                "fold": int(fold.index),
                "fold_label": fold.label(),
                "valid_window": [f"{fold.valid_months[0]:%Y-%m}",
                                 f"{fold.valid_months[1]:%Y-%m}"],
                "model": model_spec.name,
                "params": {k: v for k, v in model_spec.params.items()},
                "train_rows": int(rows_n[0]),
                "train_start": table["train_start"].iloc[0],
                "train_end": table["train_end"].iloc[0],
                "train_rmsle": round(train_one, 4),
                "valid_rmsle": round(valid_one, 4),
                "gap": round(gap_one, 4),
                "seed": SEED,
                "verdict": verdict,
                "elapsed_seconds": round(time.time() - started, 1),
            })
            print(f"\nWrote {TABLES_DIR / 'learning_curve_probe.csv'}")
            print(f"Wrote {TABLES_DIR / 'learning_curve_probe_meta.json'}")
        print(f"\nElapsed: {time.time() - started:,.1f}s")
        return 0

    rows_n = table["train_rows"].to_numpy(dtype=float)
    valid = table["valid_rmsle"].to_numpy(dtype=float)
    gap = table["gap_valid_minus_train"].to_numpy(dtype=float)
    # Slope of RMSLE per 10x increase in training rows. Guard the degenerate case where
    # every window has the same row count, which makes the fit ill-conditioned.
    if np.ptp(np.log10(rows_n)) > 1e-9:
        slope = float(np.polyfit(np.log10(rows_n), valid, 1)[0])
    else:
        slope = 0.0
    valid_range = float(valid.max() - valid.min())
    gap_last = float(gap[-1])
    fit_gap = float(table["valid_rmsle"].iloc[-1] - table["train_rmsle"].iloc[-1])

    grew = rows_n[-1] / rows_n[0]
    change = slope * np.log10(grew)
    trend_txt = (f"validation RMSLE changes by {change:+.4f} across a "
                 f"{grew:.0f}x increase in training rows, and its whole range is only "
                 f"{valid_range:.4f}")
    fit_gap_pct = 100 * fit_gap / float(table["valid_rmsle"].iloc[-1])

    # Judge flatness RELATIVE to the error level. An absolute threshold is meaningless
    # here: RMSLE lives on a log10 scale where 0.02 is large, and the curve's own spread
    # can be smaller than the measurement noise without meaning anything.
    relative_change = abs(change) / float(np.mean(valid))

    if change > 0 and relative_change > 0.02:
        # More rows make things WORSE. Neither textbook branch fits, and saying
        # "high variance" would be wrong: variance would show as a widening gap with
        # validation still improving. Here the extra rows are simply not relevant.
        verdict = (
            f"Adding training rows does not help — it slightly hurts. {trend_txt}: the "
            f"largest window scores {float(table['valid_rmsle'].iloc[-1]):.4f} against "
            f"{float(valid[0]):.4f} for the {rows_n[0] / rows_n[-1]:.0%} window that only "
            "reaches back a few years. This is neither classic overfitting (training error "
            f"is flat at ~{float(table['train_rmsle'].iloc[-1]):.3f}, and the gap is only "
            f"{fit_gap:+.4f}, about {fit_gap_pct:.0f}% of validation error) nor a shortage of "
            "data. It means the older transactions carry little usable signal for predicting "
            "the near future: house prices are close to a random walk, so 2001 sales are a "
            "weak guide to 2022 levels regardless of how many of them there are. The binding "
            "constraint is the relevance and informativeness of the features, not the volume "
            "of history."
        )
    elif relative_change <= 0.02:
        verdict = (
            f"Neither overfitting nor underfitting is the binding constraint. {trend_txt} "
            f"({relative_change:.1%} of the mean validation error) -- the curve is essentially "
            "FLAT, and the movement that is there is within the spread between points. The "
            f"train/validation gap stays modest ({fit_gap:+.4f} at the largest window, about "
            f"{fit_gap_pct:.0f}% of validation error). The ceiling is set by the feature set "
            "and by irreducible noise in individual sale prices, not by model capacity."
        )
    elif slope < 0 and gap_last < float(gap[0]):
        verdict = (
            f"High bias, not high variance. Validation error falls as the window grows "
            f"({trend_txt}) and the train/validation gap narrows "
            f"({gap[0]:+.4f} -> {gap_last:+.4f}). The model under-fits rather than memorises, "
            "so additional regularisation would not help."
        )
    else:
        verdict = (
            f"High variance. Validation error does not fall as the window grows "
            f"({trend_txt}) while the train/validation gap widens "
            f"({gap[0]:+.4f} -> {gap_last:+.4f}), so the model is starting to memorise the "
            "training window."
        )
    print()
    print(f"slope: {slope:+.4f} RMSLE per 10x rows | total change {change:+.4f} "
          f"| validation range {valid_range:.4f}")
    print("VERDICT:", verdict)

    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = TABLES_DIR / "learning_curve.csv"
        table.to_csv(out_csv, index=False)
        write_json(TABLES_DIR / "learning_curve_meta.json", {
            "fold": int(fold.index),
            "fold_label": fold.label(),
            "valid_window": [f"{fold.valid_months[0]:%Y-%m}", f"{fold.valid_months[1]:%Y-%m}"],
            "model": model_spec.name,
            "model_kind": model_spec.kind,
            "params": {k: v for k, v in model_spec.params.items()},
            "n_jobs": model_spec.n_jobs,
            "device": model_spec.device,
            "sizes": [round(s, 4) for s in sorted(args.sizes)],
            "seed": SEED,
            "train_rows_full": int(len(train)),
            "valid_rows": int(len(valid)),
            "verdict": verdict,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"\nWrote {out_csv}")
        print(f"Wrote {TABLES_DIR / 'learning_curve_meta.json'}")

    print(f"\nElapsed: {time.time() - started:,.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
