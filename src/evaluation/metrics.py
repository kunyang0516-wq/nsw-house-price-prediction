"""Evaluation metrics for a log10-price target.

The target is ``log10(purchase_price)``, so every model is scored in log space
first and then converted back to dollars for interpretability. Two conventions
that matter:

* **RMSLE** is the headline metric: in log10 space it is the RMSE divided by
  ``log10(e)``, i.e. it is symmetric in *relative* terms. Expensive homes do not
  dominate it the way they dominate RMSE in dollars.
* **MdAPE** (median absolute percentage error) is reported alongside because the
  price distribution is skewed; a handful of very expensive or very cheap sales
  can move the mean-based metrics a lot.
* Back-transforming a *mean* prediction is biased (Jensen). We therefore report
  dollar metrics on ``10**prediction`` (the median in the original scale) and
  state that explicitly rather than silently correcting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG10_E = np.log10(np.e)


def rmsle_from_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> float:
    """RMSE in natural-log units, computed from log10 inputs."""
    diff = (np.asarray(y_pred_log10) - np.asarray(y_true_log10)) * np.log(10.0)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred_log10) - np.asarray(y_true_log10))))


def rmse_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> float:
    diff = np.asarray(y_pred_log10) - np.asarray(y_true_log10)
    return float(np.sqrt(np.mean(diff ** 2)))


def r2_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> float:
    y_true = np.asarray(y_true_log10, dtype=float)
    y_pred = np.asarray(y_pred_log10, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def dollar_metrics(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> dict:
    """MAE / RMSE / MdAPE on the original AUD scale (median back-transform)."""
    true_aud = np.power(10.0, np.asarray(y_true_log10, dtype=float))
    pred_aud = np.power(10.0, np.asarray(y_pred_log10, dtype=float))
    error = pred_aud - true_aud
    ape = np.abs(error) / np.where(true_aud == 0, np.nan, true_aud)
    return {
        "mae_aud": float(np.nanmean(np.abs(error))),
        "rmse_aud": float(np.sqrt(np.nanmean(error ** 2))),
        "mdape_pct": float(np.nanmedian(ape) * 100),
        "mape_pct": float(np.nanmean(ape) * 100),
    }


def regression_report(
    y_true_log10: np.ndarray,
    y_pred_log10: np.ndarray,
    label: str = "model",
) -> dict:
    return {
        "model": label,
        "n": int(len(y_true_log10)),
        "rmsle": rmsle_from_log10(y_true_log10, y_pred_log10),
        "rmse_log10": rmse_log10(y_true_log10, y_pred_log10),
        "mae_log10": mae_log10(y_true_log10, y_pred_log10),
        "r2_log10": r2_log10(y_true_log10, y_pred_log10),
        **dollar_metrics(y_true_log10, y_pred_log10),
    }


def grouped_report(
    frame: pd.DataFrame,
    y_true_log10: pd.Series,
    y_pred_log10: pd.Series,
    by: str,
    label: str,
) -> pd.DataFrame:
    """Metrics per group — a single average can hide "only accurate downtown"."""
    data = pd.DataFrame({
        "group": frame[by].to_numpy(),
        "y_true": np.asarray(y_true_log10, dtype=float),
        "y_pred": np.asarray(y_pred_log10, dtype=float),
    })
    rows = []
    for value, block in data.groupby("group", observed=True):
        if len(block) < 50:
            continue
        rows.append({
            "model": label,
            "grouped_by": by,
            "group": value,
            "n": len(block),
            **{k: v for k, v in regression_report(block["y_true"].to_numpy(),
                                                  block["y_pred"].to_numpy()).items()
               if k not in ("model", "n")},
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def compare_models(reports: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(reports)
    ordered = ["model", "n", "rmsle", "rmse_log10", "mae_log10", "r2_log10",
               "mae_aud", "rmse_aud", "mdape_pct", "mape_pct"]
    return frame[[c for c in ordered if c in frame.columns]].sort_values("rmsle").reset_index(drop=True)
