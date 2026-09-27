"""Two-way fixed-effects event study for the Metro opening (E1).

Why the simple estimator is the right one here (design D3): every treated
postcode is treated on the *same* day (2019-05-26), so there is exactly one
cohort. The negative-weighting problem that motivates Callaway-Sant'Anna /
Sun-Abraham only arises with staggered adoption; with a single cohort the TWFE
event study is not an approximation.

What is implemented:

* a balanced ``postcode x month`` panel inside the event window,
* within (two-way) demeaning to absorb postcode and calendar-month fixed effects
  without materialising thousands of dummies,
* event-time coefficients with the reference period omitted,
* **cluster-robust standard errors by postcode**,
* a **joint pre-trend test** (Wald on the pre-period coefficients) — the gate for
  whether a causal reading is allowed at all,
* placebo tests (shifted event date, randomly assigned "treatment") that must
  come back near zero,
* a robustness grid over radius and outcome.

The estimator is intentionally hand-rolled: the full result set (coefficients,
clustered SEs, pre-trend statistic, placebo draws) is reproducible from the
saved tables without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    coefficients: pd.DataFrame      # event_time, coef, se, ci_low, ci_high, n
    pre_trend: dict                 # wald statistic, dof, p-value
    n_obs: int
    n_postcodes: int
    n_months: int
    outcome: str
    reference_period: int

    def post_mean(self) -> float:
        post = self.coefficients.loc[self.coefficients["event_time"] >= 0, "coef"]
        return float(post.mean()) if len(post) else float("nan")


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #
def balance_panel(panel: pd.DataFrame, value_col: str, count_col: str | None = "n_sales") -> pd.DataFrame:
    """Reindex to a complete postcode x month grid (missing months become NaN).

    Balancing matters for the two-way within transform: an unbalanced panel with
    incidental gaps would otherwise let the fixed effects absorb different
    information for different postcodes.
    """
    months = pd.date_range(panel["month"].min(), panel["month"].max(), freq="MS")
    index = pd.MultiIndex.from_product([sorted(panel["postcode"].unique()), months],
                                       names=["postcode", "month"])
    frame = (panel.set_index(["postcode", "month"])
             .reindex(index)
             .reset_index())
    frame["treated"] = frame.groupby("postcode")["treated"].transform("max")
    if count_col:
        frame[count_col] = frame[count_col].fillna(0)
    # Keep the real event time alongside any placebo re-labelling.
    if "event_time" in frame.columns and "event_time_original" not in frame.columns:
        frame["event_time_original"] = frame["event_time"]
    return frame.sort_values(["postcode", "month"]).reset_index(drop=True)


def add_event_time(frame: pd.DataFrame, event_month: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    out["event_time"] = ((out["month"].dt.year - event_month.year) * 12
                         + (out["month"].dt.month - event_month.month))
    return out


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def _demean_two_way(values: np.ndarray, groups: np.ndarray, times: np.ndarray,
                    tol: float = 1e-10, max_iter: int = 200) -> tuple[np.ndarray, int, int]:
    """Alternating projections: remove group and time means from every column.

    Returns the residualised array plus the number of absorbed group and time
    levels (needed for the degrees-of-freedom correction).
    """
    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    group_codes, _ = pd.factorize(groups)
    time_codes, _ = pd.factorize(times)
    n_groups = group_codes.max() + 1
    n_times = time_codes.max() + 1

    for _ in range(max_iter):
        previous = matrix.copy()
        # remove group means
        sums = np.zeros((n_groups, matrix.shape[1]))
        np.add.at(sums, group_codes, matrix)
        counts = np.bincount(group_codes, minlength=n_groups).reshape(-1, 1)
        matrix -= (sums / np.where(counts == 0, 1, counts))[group_codes]
        # remove time means
        sums = np.zeros((n_times, matrix.shape[1]))
        np.add.at(sums, time_codes, matrix)
        counts = np.bincount(time_codes, minlength=n_times).reshape(-1, 1)
        matrix -= (sums / np.where(counts == 0, 1, counts))[time_codes]
        if np.max(np.abs(matrix - previous)) < tol:
            break
    return matrix, n_groups, n_times


def _cluster_cov(design: np.ndarray, residuals: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Cluster-robust (CR1) covariance for an OLS coefficient vector."""
    n, k = design.shape
    xtx_inv = np.linalg.pinv(design.T @ design)
    codes, uniques = pd.factorize(clusters)
    g = len(uniques)

    meat = np.zeros((k, k))
    for code in range(g):
        mask = codes == code
        score = design[mask].T @ residuals[mask]
        meat += np.outer(score, score)

    # Small-sample correction: (G / (G-1)) * ((n-1) / (n-k))
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return xtx_inv @ meat @ xtx_inv * correction


def estimate_event_study(
    panel: pd.DataFrame,
    outcome: str,
    reference_period: int = -1,
    min_group_size: int = 5,
    extra_regressors: list[str] | None = None,
    treated_col: str = "treated",
    cluster_col: str = "postcode",
) -> EventStudyResult:
    """TWFE event study: outcome ~ event-time dummies + postcode FE + month FE.

    ``min_group_size`` drops event-time buckets with fewer than this many
    observations in either arm, which keeps bins at the window edge (and the
    fully binned endpoints) from being estimated off a handful of sales.
    """
    frame = panel.dropna(subset=[outcome]).copy()
    frame = frame.loc[frame[treated_col].notna()]
    if frame.empty:
        raise ValueError("No observations with a non-missing outcome.")

    event_times = sorted(frame["event_time"].dropna().unique())
    # Bucket the window edges so the tails are not estimated from thin cells.
    lo, hi = int(min(event_times)), int(max(event_times))
    frame["event_bucket"] = (frame["event_time"].clip(lower=lo + 2, upper=hi - 2)
                             .round().astype("Int64"))

    counts = (frame.dropna(subset=["event_bucket"])
              .groupby(["event_bucket", treated_col], observed=True)[outcome]
              .size().unstack(fill_value=0))
    keep_buckets = counts.index[(counts.min(axis=1) >= min_group_size)]
    frame = frame.loc[frame["event_bucket"].isin(keep_buckets)]

    buckets = [int(b) for b in sorted(frame["event_bucket"].dropna().unique())
               if int(b) != reference_period]
    if not buckets:
        raise ValueError("No event-time buckets left after filtering.")

    dummies = {}
    for bucket in buckets:
        # Interaction with treatment: the control group identifies the
        # calendar-time fixed effects, the interaction identifies the effect.
        dummies[f"et_{bucket}"] = ((frame["event_bucket"] == bucket)
                                   & (frame[treated_col] == 1)).astype(float).to_numpy()

    z_columns = [dummies[key] for key in dummies]
    if extra_regressors:
        for column in extra_regressors:
            z_columns.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    z = np.column_stack(z_columns)

    y = frame[outcome].to_numpy(dtype=float)
    groups = frame[cluster_col].to_numpy()
    times = frame["month"].to_numpy()

    combined = np.column_stack([y, z])
    residualised, n_groups, n_times = _demean_two_way(combined, groups, times)
    y_res = residualised[:, 0]
    z_res = residualised[:, 1:]

    beta, *_ = np.linalg.lstsq(z_res, y_res, rcond=None)
    fitted = z_res @ beta
    resid = y_res - fitted

    covariance = _cluster_cov(z_res, resid, groups)
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))

    coefficient_names = list(dummies.keys())
    if extra_regressors:
        coefficient_names += list(extra_regressors)
    table = pd.DataFrame({
        "term": coefficient_names,
        "event_time": [int(name.split("_")[1]) if name.startswith("et_") else np.nan
                       for name in coefficient_names],
        "coef": beta,
        "se": se,
    })
    table["ci_low"] = table["coef"] - 1.96 * table["se"]
    table["ci_high"] = table["coef"] + 1.96 * table["se"]
    table["n"] = len(frame)
    table["absorbed_group_levels"] = n_groups
    table["absorbed_time_levels"] = n_times
    table = table.sort_values("event_time").reset_index(drop=True)

    # --- joint pre-trend test ------------------------------------------------
    pre = table.loc[table["event_time"] < reference_period]
    pre_index = [coefficient_names.index(f"et_{int(b)}") for b in pre["event_time"]]
    pre_trend = {"n_pre_coefficients": len(pre_index)}
    if pre_index:
        beta_pre = beta[pre_index]
        cov_pre = covariance[np.ix_(pre_index, pre_index)]
        statistic = float(beta_pre @ np.linalg.pinv(cov_pre) @ beta_pre)
        dof = len(pre_index)
        try:
            from scipy import stats
            p_value = float(1 - stats.chi2.cdf(statistic, dof))
        except Exception:
            p_value = float("nan")
        pre_trend.update({"wald_statistic": statistic, "dof": dof, "p_value": p_value,
                          "max_abs_pre_coef": float(pre["coef"].abs().max()),
                          "interpretation": (
                              "p < 0.05 rejects parallel pre-trends; report the design as "
                              "descriptive rather than causal")})
    else:
        pre_trend.update({"wald_statistic": float("nan"), "dof": 0, "p_value": float("nan")})

    return EventStudyResult(
        coefficients=table,
        pre_trend=pre_trend,
        n_obs=int(len(frame)),
        n_postcodes=int(frame[cluster_col].nunique()),
        n_months=int(frame["month"].nunique()),
        outcome=outcome,
        reference_period=reference_period,
    )


# --------------------------------------------------------------------------- #
# placebo / robustness
# --------------------------------------------------------------------------- #
def placebo_shifted_event(
    panel: pd.DataFrame,
    outcome: str,
    shift_months: int = -36,
    reference_period: int = -1,
    **kwargs,
) -> EventStudyResult:
    """Re-label a fake event earlier in time; effects should be ~0.

    ``shift_months`` moves the fake event date; the estimation window stays
    [-24, -1] months relative to the *fake* event, so the whole placebo window
    sits before the real opening and contains only pre-treatment data.
    """
    frame = panel.copy()
    if "event_time_original" not in frame.columns:
        frame["event_time_original"] = frame["event_time"]
    frame["event_time"] = frame["event_time_original"] - shift_months
    # Re-index inside the placebo window so the fake event itself sits at 0.
    window = frame["event_time"].between(-24, 24)
    frame = frame.loc[window].copy()
    frame["event_time"] = frame["event_time"] + 24
    if frame.empty:
        raise ValueError(
            f"Shifted-date placebo has no observations (shift {shift_months}m); "
            "widen the panel window or pick a smaller shift."
        )
    return estimate_event_study(frame, outcome, reference_period=23, **kwargs)


def placebo_random_treatment(
    panel: pd.DataFrame,
    outcome: str,
    seed: int = 36103,
    reference_period: int = -1,
    n_draws: int = 1,
    **kwargs,
) -> EventStudyResult:
    """Randomly reassign treatment across the same postcodes.

    With very few treated clusters (8 postcodes at the 2 km radius) a single draw
    can reject by chance, so the report should read this together with the
    shifted-date placebo rather than on its own.
    """
    frame = panel.copy()
    if "event_time_original" not in frame.columns:
        frame["event_time_original"] = frame["event_time"]
    postcodes = sorted(frame["postcode"].unique())
    rng = np.random.default_rng(seed)
    share = frame.groupby("postcode")["treated"].max().mean()
    results = []
    for draw in range(max(1, n_draws)):
        n_treated = max(1, int(round(len(postcodes) * share)))
        chosen = set(rng.choice(postcodes, size=n_treated, replace=False))
        variant = frame.copy()
        variant["treated"] = variant["postcode"].isin(chosen).astype(int)
        variant["event_time"] = variant["event_time_original"]
        results.append(estimate_event_study(variant, outcome,
                                            reference_period=reference_period, **kwargs))
    if len(results) == 1:
        return results[0]
    # Aggregate draws into one synthetic result: mean coefficient, mean SE.
    stacked = pd.concat([r.coefficients.assign(draw=i) for i, r in enumerate(results)],
                        ignore_index=True)
    aggregated = (stacked.groupby("event_time", as_index=False)
                  .agg(coef=("coef", "mean"), se=("se", "mean"),
                       ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"),
                       n=("n", "max")))
    aggregated["term"] = ["et_" + str(int(v)) for v in aggregated["event_time"]]
    p_values = [r.pre_trend.get("p_value", float("nan")) for r in results]
    return EventStudyResult(
        coefficients=aggregated,
        pre_trend={"n_pre_coefficients": results[0].pre_trend.get("n_pre_coefficients"),
                   "p_value_median": float(np.nanmedian(p_values)),
                   "p_value_share_below_0_05": float(np.mean(np.asarray(p_values) < 0.05)),
                   "draws": len(results)},
        n_obs=results[0].n_obs,
        n_postcodes=results[0].n_postcodes,
        n_months=results[0].n_months,
        outcome=outcome,
        reference_period=reference_period,
    )


def dose_response(
    panel: pd.DataFrame,
    outcome: str,
    treatment: pd.DataFrame,
    max_km: float = 10.0,
    post_from: int = 0,
    cluster_col: str = "postcode",
    label: str = "dose_response",
) -> dict:
    """Continuous-exposure difference-in-differences: intensity instead of a ring.

    ``outcome = postcode FE + month FE + beta * post x log(1 + distance_km)``.

    With only 8-12 treated postcodes a binary treated/control contrast leans on a
    handful of clusters. A distance gradient uses the whole postcode set within
    ``max_km`` and asks a sharper question: do prices rise *more* where the new
    station is closer? Read ``beta < 0`` as "closer to the station, bigger
    post-opening increase" (distance enters positively, so a negative
    interaction means the gradient flattens/inverts).
    """
    frame = panel.copy()
    distance = treatment.set_index("postcode")["dist_metro_pre"]
    frame["dist_km"] = frame["postcode"].map(distance)
    frame = frame.loc[frame["dist_km"].notna() & frame["dist_km"].le(max_km)]
    frame = frame.dropna(subset=[outcome])
    if frame.empty:
        raise ValueError("No observations within the dose-response distance band.")

    frame["log_dist"] = np.log1p(frame["dist_km"])
    frame["post"] = (frame["event_time"] >= post_from).astype(float)
    frame["post_x_logdist"] = frame["post"] * frame["log_dist"]

    z = frame[["post_x_logdist"]].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    groups = frame[cluster_col].astype("string").to_numpy()
    times = frame["month"].to_numpy()

    combined = np.column_stack([y, z])
    residualised, n_groups, n_times = _demean_two_way(combined, groups, times)
    y_res = residualised[:, 0]
    z_res = residualised[:, 1:]

    beta, *_ = np.linalg.lstsq(z_res, y_res, rcond=None)
    resid = y_res - z_res @ beta
    covariance = _cluster_cov(z_res, resid, groups)
    se = float(np.sqrt(max(covariance[0, 0], 0.0)))

    try:
        from scipy import stats
        p_value = float(2 * (1 - stats.norm.cdf(abs(beta[0] / se)))) if se > 0 else float("nan")
    except Exception:
        p_value = float("nan")

    return {
        "specification": label,
        "outcome": outcome,
        "max_km": max_km,
        "coef_post_x_log_dist": float(beta[0]),
        "se": se,
        "ci_low": float(beta[0] - 1.96 * se),
        "ci_high": float(beta[0] + 1.96 * se),
        "p_value": p_value,
        "n_obs": int(len(frame)),
        "n_postcodes": int(frame[cluster_col].nunique()),
        "interpretation": (
            "negative interaction = the post-opening change is larger closer to the "
            "station" if beta[0] < 0 else
            "positive interaction = the post-opening change is larger further from the station"
        ),
    }


def coefficients_to_wide(results: dict[str, EventStudyResult]) -> pd.DataFrame:
    """Stack several specifications into one tidy table for the report."""
    rows = []
    for label, result in results.items():
        for _, row in result.coefficients.iterrows():
            if pd.isna(row["event_time"]):
                continue
            rows.append({
                "specification": label,
                "outcome": result.outcome,
                "event_time": int(row["event_time"]),
                "coef": row["coef"],
                "se": row["se"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n": row["n"],
            })
    return pd.DataFrame(rows)
