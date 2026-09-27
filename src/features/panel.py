"""Postcode panel and rolling neighbourhood features — time-safe by construction.

Two leakage rules are enforced here (initial_v0/03_leakage_audit.md L1/L7):

* **No contemporaneous information.** A feature for month *t* is built from
  months up to and including *t-1* (the `shift(1)` in `_lag`). A property
  contracted in month *t* therefore never sees another sale from month *t*.
* **No future information.** Rolling windows are trailing only, the monthly
  spine is complete (gaps become explicit NaN, never a forward-filled value),
  and counts are integer-valued so a partially observed month is visible in the
  data rather than silently averaged away.

Helper columns are prefixed with ``_`` and dropped before the frame is returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOWS: tuple[int, ...] = (3, 6, 12)

# Columns that exist in the panel only to build features.
_HELPER_PREFIX = "_"


def to_month(series: pd.Series) -> pd.Series:
    """Contract date -> month period (the panel's time index)."""
    dates = pd.to_datetime(series, errors="coerce")
    # Normalise to the first of the month so merging on the spine is exact.
    return dates.dt.to_period("M").dt.to_timestamp()


def build_postcode_month_panel(
    df: pd.DataFrame,
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
    postcode_col: str = "post_code",
    date_col: str = "contract_date",
) -> pd.DataFrame:
    """Aggregate transactions to `post_code x month` (descriptive + feature input).

    Aggregation itself does not leak — the leakage risk lives in how the result
    is *used*, which `add_rolling_features` handles.
    """
    frame = pd.DataFrame({
        "postcode": df[postcode_col].astype("string"),
        "month": to_month(df[date_col]),
        "price": pd.to_numeric(df[value_col], errors="coerce"),
        "area": pd.to_numeric(df[area_col], errors="coerce"),
    }).dropna(subset=["postcode", "month", "price"])

    frame["unit_price"] = frame["price"] / frame["area"].replace(0, np.nan)

    panel = (frame.groupby(["postcode", "month"], observed=True)
             .agg(n_sales=("price", "size"),
                  median_price=("price", "median"),
                  mean_price=("price", "mean"),
                  median_unit_price=("unit_price", "median"),
                  median_area=("area", "median"),
                  p25_price=("price", lambda s: s.quantile(0.25)),
                  p75_price=("price", lambda s: s.quantile(0.75)))
             .reset_index())
    return panel


def complete_monthly_spine(
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Reindex every postcode onto a complete monthly grid.

    Missing months are materialised as NaN with ``n_sales = 0`` so that:
      * rolling windows count *calendar* months (not observations), and
      * a gap is visible rather than interpolated.
    """
    months = pd.date_range(
        start or panel["month"].min(),
        end or panel["month"].max(),
        freq="MS",
    )
    index = pd.MultiIndex.from_product([panel["postcode"].unique(), months],
                                       names=["postcode", "month"])
    out = (panel.set_index(["postcode", "month"])
           .reindex(index)
           .reset_index())
    out["n_sales"] = out["n_sales"].fillna(0).astype("int64")
    out["_observed"] = out["median_price"].notna()
    return out.sort_values(["postcode", "month"]).reset_index(drop=True)


def _lag(frame: pd.DataFrame, columns: list[str], group: str = "postcode") -> pd.DataFrame:
    """Shift the named columns one month back inside each postcode.

    This single line is what makes the features safe: a row for month *t* can
    only ever describe months <= t-1.
    """
    out = frame.copy()
    out[columns] = out.groupby(group, observed=True)[columns].shift(1)
    return out


def add_rolling_features(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    group: str = "postcode",
    strict_windows: bool = True,
) -> pd.DataFrame:
    """Trailing rolling neighbourhood features, all lagged one month.

    Produces, per window *w*:
      ``pc_med_price_{w}m``, ``pc_n_sales_{w}m``, ``pc_med_unit_price_{w}m``,
      ``pc_price_iqr_ratio_{w}m``, ``pc_n_months_observed_{w}m``
    plus ``pc_months_since_sale``.

    ``strict_windows=True`` requires the window to be *fully* observed
    (``min_periods == w``). The previous floor of ``w // 2`` produced a
    "6-month median" from as few as three observed months, so the column name
    overstated the evidence behind the value. The companion
    ``pc_n_months_observed_{w}m`` column makes coverage explicit instead: the
    model can separate "no data" from "partially observed window" and fall back
    to the shorter windows when the longer ones are thin.
    """
    frame = panel.sort_values([group, "month"]).copy()

    base_cols = ["median_price", "n_sales", "median_unit_price", "p25_price", "p75_price"]
    for column in base_cols:
        if column not in frame.columns:
            raise KeyError(f"Panel is missing '{column}' — build it with build_postcode_month_panel.")

    # Rule out same-month information first, then roll.
    lagged = _lag(frame, base_cols, group=group)
    grouped = lagged.groupby(group, observed=True)

    features = pd.DataFrame({group: frame[group], "month": frame["month"]})
    # Coverage counts are built from the same one-month-lagged observation flag.
    observed_lagged = frame["median_price"].notna().astype(float).groupby(
        frame[group], observed=True).shift(1)

    for window in windows:
        min_periods = window if strict_windows else max(1, window // 2)
        roll = grouped.rolling(window, min_periods=min_periods)

        def _values(name: str, agg: str):
            # `rolling` on a groupby yields a MultiIndex (group, month); drop the
            # duplicated level and realign positionally to the sorted frame.
            return roll[name].agg(agg).reset_index(level=0, drop=True).to_numpy()

        features[f"pc_med_price_{window}m"] = _values("median_price", "median")
        features[f"pc_med_unit_price_{window}m"] = _values("median_unit_price", "median")
        features[f"pc_n_sales_{window}m"] = _values("n_sales", "sum")
        features[f"pc_price_iqr_ratio_{window}m"] = _values("p75_price", "median") / _values("p25_price", "median")
        features[f"pc_n_months_observed_{window}m"] = (
            observed_lagged.groupby(frame[group], observed=True)
            .rolling(window, min_periods=1).sum()
            .reset_index(level=0, drop=True).to_numpy()
        )

    # Staleness: how long since this postcode last recorded a sale.
    observed_month = frame["month"].where(frame["median_price"].notna())
    last_observed = (observed_month.groupby(frame[group], observed=True).ffill())
    months_since = (frame["month"] - last_observed).dt.days / 30.44
    features["pc_months_since_sale"] = months_since.to_numpy()

    features["pc_months_observed"] = frame.groupby(group, observed=True).cumcount().to_numpy()
    return features


def attach_panel_features(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    postcode_col: str = "post_code",
    date_col: str = "contract_date",
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
) -> pd.DataFrame:
    """Join the monthly panel onto transaction rows.

    ``panel`` may be either the base aggregate (`build_postcode_month_panel`) or
    that aggregate with rolling features attached.

    Same-month aggregates are kept for EDA/reporting but **renamed** with a
    ``contemporaneous_`` prefix, because they describe sales in the very month of
    the contract — including other dwellings sold that month. They are not
    prediction features; the lagged ``pc_*`` columns are.
    """
    merged = df.copy()
    merged["month"] = to_month(merged[date_col])
    # Normalise the join key: the panel stores postcodes as 4-char strings
    # (from read_postcode_key) while raw frames may hold numeric or object
    # postcodes. Silent dtype mismatch would produce an all-NaN merge here.
    merged["_postcode_key"] = merged[postcode_col].astype("string").str.strip().str.zfill(4)

    base_columns = [c for c in panel.columns
                    if c not in ("postcode", "month") and not c.startswith("_")
                    and not c.startswith("pc_")]
    rolling_columns = [c for c in panel.columns if c.startswith("pc_")]

    # 1) same-month aggregates, explicitly labelled
    if base_columns:
        left = (panel[["postcode", "month", *base_columns]]
                .rename(columns={"postcode": "_postcode_key",
                                 **{c: f"contemporaneous_{c}" for c in base_columns}}))
        merged = merged.merge(left, on=["_postcode_key", "month"],
                              how="left", validate="many_to_one")

    # 2) lagged rolling features (the safe ones)
    if rolling_columns:
        right = (panel[["postcode", "month", *rolling_columns]]
                 .rename(columns={"postcode": "_postcode_key"}))
        merged = merged.merge(right, on=["_postcode_key", "month"],
                              how="left", validate="many_to_one")

    return merged.drop(columns=["month", "_postcode_key"], errors="ignore")
