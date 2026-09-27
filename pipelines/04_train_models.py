"""Rolling-origin training and evaluation for the three model families.

Design (D1/D2 + audit L1/L2/L3):

* Folds come from `rolling_origin_splits` (expanding window, 12-month horizon,
  1-month embargo) and every fold is **purged by `group_key`** so a dwelling that
  sells twice cannot sit on both sides of a split.
* The feature transformer is fitted **inside each fold** on that fold's training
  rows only: imputation medians, category frequencies, target encodings, area bin
  edges and standardisation all come from the training fold.
* The naive benchmark predicts the postcode's trailing 12-month median price,
  which is itself a lagged feature — a fair, non-trivial bar.

Usage
-----
    python pipelines/04_train_models.py                       # full run, all models
    python pipelines/04_train_models.py --models ridge naive  # subset
    python pipelines/04_train_models.py --sample 200000       # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compare_models, regression_report
from src.features.pipeline import OUT_FEATURES_PARQUET
from src.models.features import FeatureSpec, PropertyFeatureTransformer, make_xy
from src.models.registry import (
    ModelSpec,
    build_model,
    forest_backend_available,
    naive_global_predict,
    naive_postcode_month_predict,
    select_model_specs,
)
from src.utils.config import REPORTS_DIR
from src.utils.io import write_json
from src.validation.time_series_cv import (
    holdout_window,
    rolling_origin_splits,
    split_frame,
)

TABLES_DIR = REPORTS_DIR / "tables"


def check_xgb_device(device: str) -> None:
    """Fail fast (and loudly) if a GPU was requested but is unusable.

    Without this, ``device="cuda"`` raises deep inside the first ``fit`` call
    after the fold's feature engineering has already run.
    """
    if device != "cuda":
        return
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - xgboost is a hard dependency
        raise SystemExit(f"xgboost is not importable: {exc}")
    if not xgb.build_info().get("USE_CUDA", False):
        raise SystemExit(
            "This xgboost build has no CUDA support (USE_CUDA: False). "
            "Re-run without --xgb-device cuda.")
    try:
        import numpy as np
        from xgboost import XGBRegressor
        probe = XGBRegressor(n_estimators=1, device="cuda", tree_method="hist")
        probe.fit(np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.float32))
    except Exception as exc:
        raise SystemExit(
            f"CUDA backend requested but a probe fit failed: {type(exc).__name__}: {exc}\n"
            "Re-run without --xgb-device cuda (CPU is ~7.5x slower but always works).")
    print("xgboost CUDA backend verified with a probe fit.")


def check_rf_threads(n_jobs: int) -> None:
    """Report which forest backend will be used, and fail only if none can run.

    sklearn's forest parallelises through joblib, which on Windows builds a
    ``multiprocessing`` thread pool whose queue needs a named pipe; confined /
    sandboxed shells deny that with ``PermissionError: [WinError 5]``. Rather than
    aborting, the caller swaps in ``ThreadForestRegressor`` (see
    ``src/models/thread_forest.py``), which uses ``concurrent.futures`` and gives
    numerically identical results. So a missing joblib pool is a warning, not an
    error -- only a total lack of threads is fatal.
    """
    if n_jobs <= 1:
        return
    if forest_backend_available():
        print(f"Random forest: sklearn backend, n_jobs={n_jobs}.")
        return
    from src.models.thread_forest import threads_available
    if threads_available():
        print(f"joblib thread pool unavailable (sandboxed shell) -> "
              f"ThreadForestRegressor with {n_jobs} threads instead; "
              "predictions are identical to sklearn's.")
    else:
        raise SystemExit(
            f"--rf-jobs {n_jobs} requested but this environment cannot create "
            "threads at all. Re-run with --rf-jobs 1 (much slower).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(OUT_FEATURES_PARQUET))
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--holdout-months", type=int, default=12,
                        help="reserve the final N months as an untouched test set "
                             "(no fold validates inside it)")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=400_000,
                        help="cap training rows per fold (chronological tail kept); "
                             "0 disables the cap")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-jobs", type=int, default=1,
                        help="raise only if the environment allows joblib threads")
    parser.add_argument("--xgb-rounds", type=int, default=600)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu",
                        help="XGBoost backend. 'cuda' measured ~7.5x faster than "
                             "single-threaded cpu on this data; falls back loudly if "
                             "the installed build has no GPU support")
    parser.add_argument("--xgb-jobs", type=int, default=1,
                        help="XGBoost threads (ignored by the cuda backend)")
    parser.add_argument("--drop-unjudged-iqr", action="store_true",
                        help="deprecated: rows outside the fence are now always excluded "
                             "unless --keep-iqr-outside is passed")
    parser.add_argument("--keep-iqr-outside", action="store_true",
                        help="keep rows the training fences flagged as implausible. "
                             "NOT recommended: the raw data contains a 2.7e9 sqm area and "
                             "$100M+ 'house' prices that make linear models explode.")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
def predict_one(
    model_spec: ModelSpec,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_valid: np.ndarray,
    valid: pd.DataFrame,
    train_median_log: float,
) -> tuple[np.ndarray, object | None]:
    """Validation predictions for one model on one fold, plus the fitted model."""
    if model_spec.kind == "naive_postcode_month":
        return naive_postcode_month_predict(valid, train_median_log), None
    if model_spec.kind == "naive_global":
        return naive_global_predict(valid, train_median_log), None

    model = build_model(model_spec)
    n = len(Z_train)

    if model_spec.kind == "xgboost":
        # Early stopping on a chronological tail of the TRAINING fold. Using the
        # validation fold would leak the evaluation window into model selection.
        #
        # On the cuda backend `fit`/`predict` move host numpy to the booster's
        # device internally, which is the intended path in xgboost 3.x: neither
        # `DMatrix` nor `QuantileDMatrix` takes a `device` argument any more, and
        # the sklearn wrapper rejects a DMatrix eval_set outright
        # ("Not supported type for data"). Passing numpy straight through is
        # both correct and fastest here; the one-off "mismatched devices"
        # warning on the first predict is harmless.
        split_at = min(max(int(n * 0.9), 1), n - 2)
        model.fit(Z_train[:split_at], y_train[:split_at],
                  eval_set=[(Z_train[split_at:], y_train[split_at:])],
                  verbose=False)
    else:
        model.fit(Z_train, y_train)

    return np.asarray(model.predict(Z_valid), dtype=float), model


def importance_rows(model_spec: ModelSpec, model, names: list[str]) -> list[dict]:
    if model is None or model_spec.kind not in ("ridge", "random_forest", "thread_forest",
                                                "xgboost"):
        return []
    if model_spec.kind == "ridge":
        values = np.abs(np.asarray(model.coef_).ravel())
        kind = "abs_coefficient"
    else:
        values = np.asarray(model.feature_importances_).ravel()
        kind = "importance"
    order = np.argsort(values)[::-1][:40]
    return [{
        "model": model_spec.name,
        "kind": kind,
        "rank": rank,
        "feature": names[index] if index < len(names) else f"f{index}",
        "value": float(values[index]),
    } for rank, index in enumerate(order, start=1)]


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    started = time.time()
    check_xgb_device(args.xgb_device)
    check_rf_threads(args.rf_jobs)

    frame = pd.read_parquet(args.features)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"])
    if args.sample and args.sample < len(frame):
        # Keep the chronological spread rather than the first N rows.
        step = max(1, len(frame) // args.sample)
        frame = frame.sort_values("contract_date").iloc[::step].copy()
        print(f"Smoke test on a {len(frame):,}-row chronological sample.")

    spec = FeatureSpec()
    folds = rolling_origin_splits(
        frame["contract_date"],
        min_train_months=args.min_train_months,
        horizon_months=args.horizon_months,
        step_months=args.step_months,
        embargo_months=args.embargo_months,
        holdout_months=args.holdout_months,
    )
    holdout = holdout_window(frame["contract_date"], args.holdout_months)
    print(f"{len(folds)} rolling-origin folds:")
    for fold in folds:
        print("  " + fold.label())
    if holdout is not None:
        print(f"Holdout reserved: {holdout[0]:%Y-%m} .. {holdout[1]:%Y-%m} "
              f"(never validated on by any fold)")

    # Excluding rows the training fences flagged is the default: leaving them in
    # feeds a $555M 'house' and a 2.7e9 sqm parcel to the models, which makes the
    # linear model's predictions diverge (observed: a $9.1e11 prediction).
    filter_iqr = not args.keep_iqr_outside
    print(f"IQR-flagged rows: {'excluded' if filter_iqr else 'KEPT (not recommended)'}")

    specs, unknown = select_model_specs(args.models)
    if unknown:
        print(f"Warning: unknown model names ignored: {sorted(unknown)}")
    for spec_item in specs:
        if spec_item.kind in ("random_forest", "thread_forest"):
            spec_item.params["n_estimators"] = args.rf_trees
            spec_item.n_jobs = args.rf_jobs
            if spec_item.kind == "random_forest" and args.rf_jobs > 1 \
                    and not forest_backend_available():
                # joblib cannot build its thread pool here (sandboxed Windows
                # denies the queue's named pipe), so sklearn's own forest would
                # silently run on one core. Swap in the thread-backed estimator:
                # same algorithm, same seeds, identical predictions.
                spec_item.kind = "thread_forest"
                print("  joblib threads unavailable -> using ThreadForestRegressor "
                      f"(n_jobs={args.rf_jobs}); predictions are identical")
        if spec_item.kind == "xgboost":
            spec_item.params["n_estimators"] = args.xgb_rounds
            spec_item.device = args.xgb_device
            spec_item.n_jobs = args.xgb_jobs
    print("Models:", [s.name for s in specs])
    for spec_item in specs:
        if spec_item.kind == "xgboost":
            print(f"  xgboost backend: device={spec_item.device} n_jobs={spec_item.n_jobs}")
        if spec_item.kind == "random_forest":
            print(f"  random_forest: n_estimators={spec_item.params['n_estimators']} "
                  f"n_jobs={spec_item.n_jobs}")

    fold_rows: list[dict] = []
    pooled: dict[str, dict[str, list]] = {}
    predictions: list[pd.DataFrame] = []
    importance: list[dict] = []

    for fold in folds:
        train, valid, split_report = split_frame(
            frame, fold, group_col="group_key", drop_unjudged_iqr=filter_iqr)
        if train.empty or valid.empty:
            print(f"fold {fold.index}: empty side, skipped")
            continue

        capped = False
        if args.max_train_rows and len(train) > args.max_train_rows:
            # Keep the most recent training rows: the model is predicting the
            # near future, so the tail of the training window matters most.
            train = train.sort_values("contract_date").iloc[-args.max_train_rows:]
            capped = True

        print(f"\nfold {fold.index}: train {len(train):,}{' (capped)' if capped else ''} "
              f"| valid {len(valid):,} "
              f"| purged {split_report['group_overlap_removed']:,} rows "
              f"({split_report.get('group_overlap_share', 0):.2%}) sharing a dwelling with validation",
              flush=True)

        X_train, y_train, _ = make_xy(train, spec)
        X_valid, y_valid, _ = make_xy(valid, spec)
        y_train_values = y_train.to_numpy(dtype=float)
        y_valid_values = y_valid.to_numpy(dtype=float)
        train_median_log = float(np.nanmedian(y_train_values))

        transformer = PropertyFeatureTransformer(spec)
        transformer.fit(X_train, y_train)
        Z_train = transformer.transform(X_train)
        Z_valid = transformer.transform(X_valid)
        names = list(transformer.get_feature_names_out())

        is_last_fold = fold.index == folds[-1].index

        for model_spec in specs:
            y_pred, fitted = predict_one(model_spec, Z_train, y_train_values, Z_valid,
                                         valid, train_median_log)
            report = regression_report(y_valid_values, y_pred, label=model_spec.name)
            report.update({
                "fold": fold.index,
                "valid_start": fold.valid_months[0].strftime("%Y-%m"),
                "valid_end": fold.valid_months[1].strftime("%Y-%m"),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
            })
            fold_rows.append(report)

            store = pooled.setdefault(model_spec.name, {"y_true": [], "y_pred": []})
            store["y_true"].append(y_valid_values)
            store["y_pred"].append(y_pred)
            predictions.append(pd.DataFrame({
                "model": model_spec.name,
                "fold": fold.index,
                "contract_date": valid["contract_date"].to_numpy(),
                "post_code": valid["post_code"].to_numpy(),
                "development_type": valid["development_type"].to_numpy(),
                # Carried here so the error diagnostics below never have to
                # re-join the feature frame: (contract_date, post_code) is far
                # from unique (up to ~86 sales share one key), so a merge on it
                # is many-to-many and silently multiplies the diagnostic counts.
                "area_sqm": valid["area_sqm"].to_numpy(dtype=float),
                "dist_cbd": valid["dist_cbd"].to_numpy(dtype=float),
                "y_true_log10": y_valid_values,
                "y_pred_log10": y_pred,
                "abs_pct_error": np.abs(10 ** y_pred - 10 ** y_valid_values) / (10 ** y_valid_values),
            }))

            if is_last_fold:
                importance.extend(importance_rows(model_spec, fitted, names))

            print(f"  {model_spec.name:22s} RMSLE {report['rmsle']:.4f} | "
                  f"MAE ${report['mae_aud']:,.0f} | MdAPE {report['mdape_pct']:.1f}% | "
                  f"R2 {report['r2_log10']:.3f}  [{time.time() - started:,.0f}s]", flush=True)

    if not fold_rows:
        print("No folds produced results.")
        return 1

    fold_table = pd.DataFrame(fold_rows)
    pooled_table = compare_models([
        regression_report(np.concatenate(store["y_true"]),
                          np.concatenate(store["y_pred"]), label=name)
        for name, store in pooled.items()
    ])
    predictions_table = pd.concat(predictions, ignore_index=True)

    print("\n=== Pooled across folds (ranked by RMSLE) ===")
    print(pooled_table.to_string(index=False))
    print("\n=== Per-fold RMSLE stability ===")
    print(fold_table.groupby("model")["rmsle"].agg(["mean", "std", "min", "max"])
          .sort_values("mean").to_string())

    best = str(pooled_table.iloc[0]["model"])
    print(f"\nBest pooled model: {best}")

    # ------------------------------------------------------------------ #
    # Final holdout: train on everything before it, predict once, never tune.
    # ------------------------------------------------------------------ #
    holdout_table = None
    if holdout is not None:
        holdout_table = evaluate_holdout(frame, spec, specs, holdout, filter_iqr, best)
        if holdout_table is not None:
            print("\n=== Final holdout (untouched by fold selection) ===")
            print(holdout_table.to_string(index=False))

    if not args.no_save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        fold_table.to_csv(TABLES_DIR / "model_scores_by_fold.csv", index=False)
        pooled_table.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
        predictions_table.to_parquet(TABLES_DIR / "predictions.parquet", index=False)
        if holdout_table is not None:
            holdout_table.to_csv(TABLES_DIR / "model_comparison_holdout.csv", index=False)
        if importance:
            pd.DataFrame(importance).to_csv(TABLES_DIR / "feature_importance.csv", index=False)

        # Error diagnostics for the best model, by year / area band / region type.
        # All the grouping columns already ride along on `predictions_table`, so
        # this stays one row per prediction.
        diagnostics = predictions_table.loc[predictions_table["model"] == best].copy()
        diagnostics["year"] = pd.to_datetime(diagnostics["contract_date"]).dt.year.astype(str)
        diagnostics["area_band"] = pd.qcut(
            diagnostics["area_sqm"].rank(method="first"), 5,
            labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"])
        diagnostics["region_type"] = diagnostics["development_type"]

        grouped = []
        for column in ("year", "area_band", "region_type"):
            for value, block in diagnostics.groupby(column, observed=True):
                if len(block) < 200:
                    continue
                grouped.append({
                    "grouped_by": column,
                    "group": str(value),
                    "n": int(len(block)),
                    **{k: v for k, v in regression_report(
                        block["y_true_log10"].to_numpy(), block["y_pred_log10"].to_numpy()).items()
                       if k not in ("model", "n")},
                })
        pd.DataFrame(grouped).to_csv(TABLES_DIR / "model_scores_grouped.csv", index=False)

        write_json(TABLES_DIR / "training_meta.json", {
            "folds": [f.label() for f in folds],
            "min_train_months": args.min_train_months,
            "horizon_months": args.horizon_months,
            "step_months": args.step_months,
            "embargo_months": args.embargo_months,
            "holdout_months": args.holdout_months,
            "holdout_window": ([f"{holdout[0]:%Y-%m}", f"{holdout[1]:%Y-%m}"]
                               if holdout is not None else None),
            "group_key": "address|post_code",
            "iqr_flagged_rows_excluded": bool(filter_iqr),
            "max_train_rows": args.max_train_rows,
            "rows": int(len(frame)),
            "models": [s.name for s in specs],
            "best_model": best,
            "feature_spec": spec.to_dict(),
            "elapsed_seconds": round(time.time() - started, 1),
        })
        for name in ("model_comparison.csv", "model_scores_by_fold.csv",
                     "model_scores_grouped.csv", "predictions.parquet"):
            print(f"Wrote {TABLES_DIR / name}")
        if holdout_table is not None:
            print(f"Wrote {TABLES_DIR / 'model_comparison_holdout.csv'}")

    print(f"\nElapsed: {time.time() - started:,.1f}s")
    return 0


def evaluate_holdout(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    specs: list[ModelSpec],
    holdout: tuple[pd.Timestamp, pd.Timestamp],
    filter_iqr: bool,
    best_model: str,
) -> pd.DataFrame | None:
    """Train once on everything before the holdout and score it, untouched.

    This is the only genuinely out-of-sample number in the report: no fold
    selection, feature choice or hyper-parameter was informed by it. It must be
    run once, on frozen settings, and never used to tune anything.
    """
    start, end = holdout
    months = pd.to_datetime(frame["contract_date"]).dt.to_period("M").dt.to_timestamp()
    holdout_mask = months.between(start, end)
    train_mask = months < start

    pool = frame
    if filter_iqr and "iqr_keep" in frame.columns:
        # Training side only; the holdout is scored on the full population.
        keep = frame["iqr_keep"] == True  # noqa: E712
        train_mask = train_mask & keep
    if "iqr_corruption_ok" in frame.columns:
        corruption = frame["iqr_corruption_ok"] == True  # noqa: E712
        train_mask = train_mask & corruption
        holdout_mask = holdout_mask & corruption

    train = frame.loc[train_mask]
    holdout_rows = frame.loc[holdout_mask]
    if train.empty or holdout_rows.empty:
        print("Holdout evaluation skipped: empty side of the split.")
        return None

    # Purge dwellings that appear in the holdout from training, as in every fold.
    if "group_key" in frame.columns:
        overlap = train["group_key"].isin(set(holdout_rows["group_key"]))
        print(f"Holdout: purged {int(overlap.sum()):,} training rows sharing a dwelling "
              f"with the holdout ({overlap.mean():.2%})")
        train = train.loc[~overlap]

    X_train, y_train, _ = make_xy(train, spec)
    X_hold, y_hold, _ = make_xy(holdout_rows, spec)
    y_train_values = y_train.to_numpy(dtype=float)
    y_hold_values = y_hold.to_numpy(dtype=float)
    train_median_log = float(np.nanmedian(y_train_values))

    transformer = PropertyFeatureTransformer(spec)
    transformer.fit(X_train, y_train)
    Z_train = transformer.transform(X_train)
    Z_hold = transformer.transform(X_hold)

    print(f"Holdout {start:%Y-%m}..{end:%Y-%m}: train {len(train):,} | holdout {len(holdout_rows):,}")

    rows = []
    for model_spec in specs:
        y_pred, _ = predict_one(model_spec, Z_train, y_train_values, Z_hold,
                                holdout_rows, train_median_log)
        report = regression_report(y_hold_values, y_pred, label=model_spec.name)
        report["is_best_from_cv"] = model_spec.name == best_model
        rows.append(report)
    return pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
