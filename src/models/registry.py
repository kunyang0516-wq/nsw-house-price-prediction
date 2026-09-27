"""Model definitions behind one fit/predict interface.

Three families, as specified: linear regression (interpretable baseline),
random forest, and gradient boosting. The baselines are deliberately dumb so the
honest question — "does the model beat the local median?" — is answered.

All models predict ``log10(purchase_price)`` and are trained on the matrix
produced by `PropertyFeatureTransformer`, which is fitted on the training fold
inside every fold, so nothing here needs to know about leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RANDOM_SEED = 36103


@dataclass
class ModelSpec:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    needs_scaling: bool = True
    # XGBoost only: "cpu" or "cuda". The installed xgboost build reports
    # USE_CUDA: True, and `device="cuda"` measured 7.5x faster than the
    # single-threaded CPU default on this data.
    device: str = "cpu"
    # RandomForestRegressor only.
    n_jobs: int = 1


def default_model_specs() -> list[ModelSpec]:
    """Baselines plus the three requested model families.

    Names are the canonical identifiers used on the command line and in the
    report tables; aliases are handled by `select_model_specs`.
    """
    return [
        ModelSpec("naive_postcode_month", "naive_postcode_month", {}, needs_scaling=False),
        ModelSpec("naive_global_median", "naive_global", {}, needs_scaling=False),
        ModelSpec("linear_ridge", "ridge",
                  {"alpha": 1.0, "max_iter": 2000, "random_state": RANDOM_SEED}),
        # n_jobs=1 on purpose: joblib's thread backend needs named pipes, which
        # are unavailable in this environment, and single-threaded training is
        # also more reproducible. Raise it via --rf-jobs when running elsewhere.
        ModelSpec("random_forest", "random_forest",
                  {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 20,
                   "max_features": 0.4, "random_state": RANDOM_SEED},
                  n_jobs=1),
        # tree_method="hist" works on both backends; `device` selects between
        # them. Single-threaded CPU is the safe default (reproducible anywhere),
        # but it is ~7.5x slower than the GPU on this data, so the training
        # script exposes --xgb-device / --xgb-jobs.
        ModelSpec("xgboost", "xgboost",
                  {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 8,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "min_child_weight": 5, "reg_lambda": 1.0,
                   "tree_method": "hist", "random_state": RANDOM_SEED,
                   "early_stopping_rounds": 50},
                  device="cpu"),
    ]


# Convenient short names people type on the command line.
MODEL_ALIASES: dict[str, str] = {
    "naive": "naive_postcode_month",
    "naive_pc": "naive_postcode_month",
    "naive_global": "naive_global_median",
    "lr": "linear_ridge",
    "linear": "linear_ridge",
    "ridge": "linear_ridge",
    "rf": "random_forest",
    "forest": "random_forest",
    "xgb": "xgboost",
    "boost": "xgboost",
}


def select_model_specs(names: list[str] | None) -> tuple[list[ModelSpec], list[str]]:
    """Resolve user-supplied names/aliases to specs. Returns (specs, unknown)."""
    specs = default_model_specs()
    if not names:
        return specs, []
    by_name = {s.name: s for s in specs}
    selected: list[ModelSpec] = []
    unknown: list[str] = []
    for raw in names:
        key = raw.strip().lower()
        canonical = MODEL_ALIASES.get(key, raw)
        if canonical in by_name:
            if by_name[canonical] not in selected:
                selected.append(by_name[canonical])
        else:
            unknown.append(raw)
    return (selected or specs), unknown


def build_model(spec: ModelSpec):
    """Instantiate the estimator for a spec (fresh per fold)."""
    if spec.kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**spec.params)
    if spec.kind == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=spec.n_jobs, **spec.params)
    if spec.kind == "thread_forest":
        # Same algorithm, but tree building runs on a ThreadPoolExecutor instead
        # of joblib -> multiprocessing, which confined Windows shells deny. See
        # src/models/thread_forest.py; results are identical to sklearn's.
        from src.models.thread_forest import ThreadForestRegressor
        return ThreadForestRegressor(n_jobs=spec.n_jobs, **spec.params)
    if spec.kind == "xgboost":
        from xgboost import XGBRegressor
        # `device` is passed explicitly rather than through params so that the
        # CPU fallback stays automatic on machines without a CUDA build.
        return XGBRegressor(device=spec.device, n_jobs=spec.n_jobs, **spec.params)
    raise ValueError(f"Unknown model kind: {spec.kind}")


def forest_backend_available() -> bool:
    """True if sklearn's own forest can parallelise via joblib *here*.

    Must exercise joblib, not merely ``ThreadPoolExecutor``: sklearn's forest
    goes through ``joblib.Parallel``, which builds a
    ``multiprocessing.pool.ThreadPool`` whose ``SimpleQueue`` needs a named pipe.
    A plain ``ThreadPoolExecutor`` works fine in a confined shell while joblib's
    does not, so testing the wrong primitive would report a backend that still
    raises ``PermissionError: [WinError 5]`` at fit time.
    """
    try:
        from joblib import Parallel, delayed
        Parallel(n_jobs=2, prefer="threads")(delayed(int)(1) for _ in range(2))
        return True
    except (PermissionError, OSError, ImportError):
        return False


# --------------------------------------------------------------------------- #
# Naive baselines: no fitting, only past information
# --------------------------------------------------------------------------- #
def naive_postcode_month_predict(frame: pd.DataFrame, fallback_log: float) -> np.ndarray:
    """Predict each row with its postcode's trailing 12-month median price.

    This is the benchmark that matters: it uses only information available
    before the contract month (the rolling feature is lagged one month), and it
    captures the dominant driver of a house price — location and timing — with a
    single number. A model that cannot beat it is not adding value.

    ``fallback_log`` is already on the **log10 price** scale (it is the training
    median of ``log_price``), and it is used verbatim. Falling back through the
    shorter windows keeps the benchmark meaningful where the 12-month window is
    too sparse to be observed under a strict full-window rule.
    """
    values = frame["pc_med_price_12m"].to_numpy(dtype=float) if "pc_med_price_12m" in frame else np.full(len(frame), np.nan)
    for column in ("pc_med_price_6m", "pc_med_price_3m"):
        if column in frame.columns:
            values = np.where(np.isnan(values), frame[column].to_numpy(dtype=float), values)
    values = np.where(np.isnan(values), np.power(10.0, fallback_log), values)
    # The rolling columns are in AUD; the target is log10(AUD).
    return np.log10(np.clip(values, 1.0, None))


def naive_global_predict(frame: pd.DataFrame, fallback_log: float) -> np.ndarray:
    """Predict the training-set median (already log10) for every row."""
    return np.full(len(frame), fallback_log)
