"""Outlier handling that cannot leak the target's future.

The group's notebook applied one log-IQR fence to the whole 2001-2023 sample
($70k-$3.73M, 238-2,016 sqm) and removed 425,347 rows (18.5%), which trims the
*target's* tails using information from the whole period.

This module instead:

  * estimates fences **inside a training span only** (`fit_iqr_bounds`),
  * supports **per-year** bounds, so the price drift across 23 years does not
    put a thumb on the scale for any single year,
  * can be switched off entirely (`mode="off"`) so the report can compare
    trimming against robust loss on the untrimmed target.

The fitted bounds are plain values (a dict), so they can be persisted and
re-used at prediction time without touching the data again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_COL = "purchase_price"
AREA_COL = "area_sqm"
TARGET_COLS = (PRICE_COL, AREA_COL)

# Reserved group key for the fence fitted on the whole training span. Every
# contract is judged against at least this fence, so `iqr_keep` is never NA.
GLOBAL_GROUP = "__all__"


@dataclass
class IQRBounds:
    """Fitted log10-space fences, keyed by group value ("__all__" for global)."""

    multiplier: float = 1.5
    columns: tuple[str, ...] = TARGET_COLS
    by: tuple[str, ...] = ()
    bounds: dict = field(default_factory=dict)  # {(group_key, column): (lower, upper)}
    train_rows: int = 0
    train_min_date: pd.Timestamp | None = None
    train_max_date: pd.Timestamp | None = None

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for (group, column), (lower, upper) in self.bounds.items():
            rows.append({
                "group": group,
                "column": column,
                "log10_lower": lower,
                "log10_upper": upper,
                "value_lower": 10 ** lower,
                "value_upper": 10 ** upper,
            })
        return pd.DataFrame(rows)

    def as_dict(self) -> dict:
        return {
            "multiplier": self.multiplier,
            "columns": list(self.columns),
            "by": list(self.by),
            "train_rows": int(self.train_rows),
            "train_span": (
                [str(self.train_min_date), str(self.train_max_date)]
                if self.train_min_date is not None else None
            ),
            "bounds": [
                {"group": str(g), "column": c, "lower": lo, "upper": hi}
                for (g, c), (lo, hi) in self.bounds.items()
            ],
        }


def fit_iqr_bounds(
    train: pd.DataFrame,
    by: tuple[str, ...] = ("year",),
    multiplier: float = 1.5,
    date_col: str = "contract_date",
) -> IQRBounds:
    """Fit log10-space fences on the training rows only.

    Two sets of fences are stored:

    * one per group (per contract year when ``by=("year",)``), and
    * one **global** fence for the training span, stored under ``__all__``.

    The global fence matters because a contract can fall in a year the training
    span never covered. Without it those rows are simply *unjudged* and any
    corruption in them survives: the largest area in this dataset is
    2.7e9 sqm in a 2023 contract, and the per-year fences only reach 2015.
    """
    frame = train.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.loc[frame[list(TARGET_COLS)].gt(0).all(axis=1) & np.isfinite(frame[list(TARGET_COLS)]).all(axis=1)]

    if "year" in by and "year" not in frame.columns:
        frame["year"] = frame[date_col].dt.year

    logs = np.log10(frame[list(TARGET_COLS)])

    fitted = IQRBounds(
        multiplier=multiplier,
        by=tuple(by),
        train_rows=int(len(frame)),
        train_min_date=frame[date_col].min(),
        train_max_date=frame[date_col].max(),
    )

    def _add(group_key: str, index) -> None:
        subset = logs.loc[index]
        q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
        iqr = q3 - q1
        for column in TARGET_COLS:
            fitted.bounds[(group_key, column)] = (
                float(q1[column] - multiplier * iqr[column]),
                float(q3[column] + multiplier * iqr[column]),
            )

    # Global fence first so it always exists as the fallback.
    _add(GLOBAL_GROUP, frame.index)

    if by:
        keys = frame[list(by)].astype("string").agg("|".join, axis=1)
        for group_key, index in keys.groupby(keys).groups.items():
            _add(str(group_key), index)

    return fitted


def flag_outliers(
    df: pd.DataFrame,
    bounds: IQRBounds,
    date_col: str = "contract_date",
) -> pd.DataFrame:
    """Per-row trim flags for every fitted column, plus an overall `iqr_keep`.

    Rows in a group with no fitted fence (e.g. years after the training span)
    get ``iqr_keep = <NA>``: they are *unjudged*, not dropped. Callers decide
    whether to keep them (`iqr_keep.fillna(True)`), drop them
    (`== True`), or treat them as a robustness subset.
    """
    frame = df
    if "year" in bounds.by and "year" not in frame.columns:
        frame = frame.assign(year=pd.to_datetime(frame[date_col], errors="coerce").dt.year)

    if bounds.by:
        keys = frame[list(bounds.by)].astype("string").agg("|".join, axis=1)
    else:
        keys = pd.Series("__all__", index=frame.index)

    logs = np.log10(frame[list(bounds.columns)])

    per_column: dict[str, pd.Series] = {}
    for column in bounds.columns:
        if column not in frame.columns:
            continue
        lows = {g: lo for (g, c), (lo, hi) in bounds.bounds.items() if c == column}
        highs = {g: hi for (g, c), (lo, hi) in bounds.bounds.items() if c == column}
        inside = logs[column].ge(keys.map(lows)) & logs[column].le(keys.map(highs))
        per_column[column] = inside.astype("boolean")

    if not per_column:
        raise ValueError(f"None of the fitted columns {bounds.columns} are present in the frame.")

    combined = None
    for column, flags in per_column.items():
        combined = flags if combined is None else (combined & flags)

    result = frame.copy()
    for column, flags in per_column.items():
        result[f"iqr_keep_{column}"] = flags
    result["iqr_keep"] = combined
    return result


def fit_global_bounds(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    area_cap: float | None = None,
    price_cap: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Corruption fence fitted on the **whole** sample (log10 space).

    A data-quality gate, not a trimming device. It is deliberately generous and
    its only job is to remove values that cannot be a residence-house
    transaction: the raw file contains a 2.7e9 sqm area and $100M+ "house"
    prices. Because it applies to every row, validation years that have no
    per-year fence are still protected.
    """
    frame = df.loc[df[list(TARGET_COLS)].gt(0).all(axis=1)
                   & np.isfinite(df[list(TARGET_COLS)]).all(axis=1)]
    logs = np.log10(frame[list(TARGET_COLS)])
    q1, q3 = logs.quantile(0.25), logs.quantile(0.75)
    iqr = q3 - q1

    bounds: dict[str, tuple[float, float]] = {}
    for column in TARGET_COLS:
        bounds[column] = (float(q1[column] - multiplier * iqr[column]),
                          float(q3[column] + multiplier * iqr[column]))
    if area_cap is not None:
        bounds[AREA_COL] = (bounds[AREA_COL][0], min(bounds[AREA_COL][1], float(np.log10(area_cap))))
    if price_cap is not None:
        bounds[PRICE_COL] = (bounds[PRICE_COL][0], min(bounds[PRICE_COL][1], float(np.log10(price_cap))))
    return bounds


def mark_quality(
    df: pd.DataFrame,
    global_bounds: dict[str, tuple[float, float]],
    train_bounds: IQRBounds | None = None,
    date_col: str = "contract_date",
) -> pd.DataFrame:
    """Attach two independent flags.

    ``iqr_corruption_ok``
        passes the whole-sample corruption fence -> apply to training **and**
        validation rows.
    ``iqr_keep``
        additionally passes its own year's training fence. This is the *training
        quality* policy: applying it to later years would delete every year after
        the training span and silently empty the validation windows, which is
        exactly the failure this design avoids.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    if "year" not in out.columns:
        out["year"] = dates.dt.year

    corruption = pd.Series(True, index=out.index)
    for column, (lower, upper) in global_bounds.items():
        if column in out.columns:
            logs = np.log10(out[column])
            corruption &= logs.ge(lower) & logs.le(upper)

    quality = pd.Series(True, index=out.index)
    if train_bounds is not None and train_bounds.bounds:
        # Look the fences up through a Series instead of a dict. Two reasons:
        #   1. `Series.map(dict)` does not coerce dtype differences (pandas
        #      StringDtype keys vs an Int64/object year column), so it silently
        #      returned all-NaN — which marked every row in a year without its own
        #      fence as failing and emptied the training side of the post-2015
        #      folds while leaving validation untouched.
        #   2. Reindexing a Series with `.reindex(...)` gives explicit NA handling.
        year_keys = out["year"].astype("int64").astype(str)

        def _fence_series(column: str, position: int) -> pd.Series:
            values = {str(g): bounds_v[position]
                      for (g, c), bounds_v in train_bounds.bounds.items() if c == column}
            # Fallback for years beyond the fitting span.
            if (GLOBAL_GROUP, column) in train_bounds.bounds:
                values[GLOBAL_GROUP] = train_bounds.bounds[(GLOBAL_GROUP, column)][position]
            lookup = pd.Series(values, dtype="float64")
            # Rows in unfenced years take the global entry; `reindex` would give
            # NaN, so map the key first and only then look it up.
            resolved = year_keys.where(year_keys.isin(lookup.index), GLOBAL_GROUP)
            return pd.Series(lookup.reindex(resolved).to_numpy(), index=out.index)

        for column in train_bounds.columns:
            if column not in out.columns:
                continue
            logs = np.log10(out[column])
            inside = logs.ge(_fence_series(column, 0)) & logs.le(_fence_series(column, 1))
            quality &= inside.fillna(False)

    out["iqr_corruption_ok"] = corruption.astype(bool)
    out["iqr_keep"] = (quality & corruption).astype(bool)
    return out


def apply_iqr_filter(
    df: pd.DataFrame,
    bounds: IQRBounds | None,
    date_col: str = "contract_date",
    drop_unjudged: bool = False,
    global_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach both quality flags and report what each removes.

    Rows failing the corruption fence are dropped outright. Rows failing only the
    per-year quality fence stay in the frame with ``iqr_keep = False`` so the
    model pipeline can apply that policy to the training side alone.
    """
    if bounds is None and global_bounds is None:
        out = df.reset_index(drop=True)
        return out, {"mode": "off", "rows_removed": 0, "rows_remaining": len(out)}

    global_bounds = global_bounds or {}
    marked = mark_quality(df, global_bounds, bounds, date_col)

    corruption_fail = ~marked["iqr_corruption_ok"]
    quality_only_fail = marked["iqr_corruption_ok"] & ~marked["iqr_keep"]
    out = marked.loc[~corruption_fail].reset_index(drop=True)

    report = {
        "mode": "iqr",
        "by": list(bounds.by) if bounds is not None else [],
        "multiplier": bounds.multiplier if bounds is not None else None,
        "train_span": (
            f"{bounds.train_min_date:%Y-%m-%d}..{bounds.train_max_date:%Y-%m-%d}"
            if bounds is not None and bounds.train_min_date is not None else None
        ),
        "global_corruption_fence": {
            column: {"log10": [lo, hi], "value": [10 ** lo, 10 ** hi]}
            for column, (lo, hi) in global_bounds.items()
        },
        "rows_before": int(len(df)),
        "rows_corruption_fail": int(corruption_fail.sum()),
        "rows_quality_fail_flagged": int(quality_only_fail.sum()),
        "rows_removed": int(corruption_fail.sum()),
        "rows_remaining": int(len(out)),
        "rows_flagged_for_training_only": int((~out["iqr_keep"]).sum()),
    }
    return out, report
