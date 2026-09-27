"""Time-series cross-validation with grouping and an embargo gap.

Design fixed in initial_v0/02_next_steps.md §6.3 and §0 (D1/D2):

  expanding |---- train ----|--embargo--|-- valid --|
             |------- train -------|--embargo--|-- valid --|
             |---------- train ----------|--embargo--|-- valid --|

* **Expanding** by default (a sliding window is a one-flag change) because the
  sample is long and the process is non-stationary.
* **Embargo** ≥ 1 month between the training end and the validation start. It
  absorbs (a) the lagged panel features, which describe month *t-1*, and
  (b) repeat sales of the same dwelling that straddle the split.
* **Grouping** by `group_key` = ``address|postcode``: 63.5% of rows share a key
  with another row, so a random or purely time-based split lets the same
  dwelling appear in both folds and inflates the score (audit M2 / T3).

The splitter works on monthly keys, which keeps it cheap on 1.8M rows and makes
the fold boundaries human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/validation split, identified by month."""

    index: int
    train_months: tuple[pd.Timestamp, pd.Timestamp]
    valid_months: tuple[pd.Timestamp, pd.Timestamp]
    embargo_months: int

    def label(self) -> str:
        return (f"fold{self.index}: train {self.train_months[0]:%Y-%m}..{self.train_months[1]:%Y-%m}"
                f" | embargo {self.embargo_months}m"
                f" | valid {self.valid_months[0]:%Y-%m}..{self.valid_months[1]:%Y-%m}")


def _month_floor(value) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def rolling_origin_splits(
    months: pd.Series | pd.DatetimeIndex,
    min_train_months: int = 120,
    horizon_months: int = 12,
    step_months: int = 12,
    embargo_months: int = 1,
    mode: str = "expanding",
    window_months: int | None = None,
    holdout_months: int = 0,
) -> list[Fold]:
    """Generate forward-chaining folds over the observed months.

    Parameters
    ----------
    months
        Contract dates (or any dates) of the rows; only the span matters.
    min_train_months
        Minimum length of the first training window.
    horizon_months
        Length of each validation window.
    step_months
        How far the origin advances between folds.
    embargo_months
        Gap inserted between training end and validation start.
    mode
        "expanding" (train always starts at the first month) or "sliding"
        (train is the last `window_months` months).
    holdout_months
        Reserve the final N months as an untouched test set: no fold may
        validate inside it. Every earlier fold still trains on data that
        precedes its own validation window, so the holdout stays unseen until a
        model is frozen.
    """
    if mode not in ("expanding", "sliding"):
        raise ValueError("mode must be 'expanding' or 'sliding'")
    if mode == "sliding" and not window_months:
        raise ValueError("sliding mode requires window_months")

    observed = pd.to_datetime(pd.Series(months)).dropna()
    if observed.empty:
        raise ValueError("No dates supplied.")
    first = _month_floor(observed.min())
    last = _month_floor(observed.max())
    all_months = pd.date_range(first, last, freq="MS")

    # Last month a fold is allowed to validate in.
    valid_ceiling = all_months[-(holdout_months + 1)] if holdout_months else all_months[-1]

    folds: list[Fold] = []
    index = 0
    train_end_offset = min_train_months - 1
    while True:
        valid_start_offset = train_end_offset + embargo_months + 1
        valid_end_offset = valid_start_offset + horizon_months - 1
        if valid_start_offset >= len(all_months):
            break
        if all_months[valid_start_offset] > valid_ceiling:
            break

        valid_start = all_months[valid_start_offset]
        valid_end = all_months[min(valid_end_offset, len(all_months) - 1)]
        if valid_end > valid_ceiling:
            valid_end = valid_ceiling
        train_end = all_months[train_end_offset]

        if mode == "expanding":
            train_start = first
        else:
            train_start = all_months[max(0, train_end_offset - window_months + 1)]

        folds.append(Fold(index=index,
                          train_months=(train_start, train_end),
                          valid_months=(valid_start, valid_end),
                          embargo_months=embargo_months))
        index += 1
        train_end_offset += step_months

        if valid_end >= valid_ceiling:
            break

    return folds


def holdout_window(
    months: pd.Series | pd.DatetimeIndex,
    holdout_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Start and end month of the reserved holdout period."""
    if not holdout_months:
        return None
    observed = pd.to_datetime(pd.Series(months)).dropna()
    all_months = pd.date_range(_month_floor(observed.min()), _month_floor(observed.max()), freq="MS")
    if holdout_months >= len(all_months):
        raise ValueError("holdout_months covers the whole sample.")
    return all_months[-holdout_months], all_months[-1]


def month_keys(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Map dates to month-start timestamps for cheap membership tests."""
    return pd.to_datetime(pd.Series(dates)).dt.to_period("M").dt.to_timestamp()


def split_frame(
    df: pd.DataFrame,
    fold: Fold,
    date_col: str = "contract_date",
    group_col: str | None = "group_key",
    drop_unjudged_iqr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (train, valid) row subsets for one fold, plus a small report.

    Two quality policies are applied asymmetrically on purpose:

    * **corruption fence** (``iqr_corruption_ok``) applies to both sides — it
      only removes values that cannot be a house transaction.
    * **per-year quality fence** (``iqr_keep``) applies to the **training side
      only** when ``drop_unjudged_iqr`` is set. Applying it to validation would
      delete every year after the training span (no fence exists for those years)
      and silently empty the later validation windows, so the validation period
      is scored on the population rather than on a self-selected subset.

    ``group_col`` is enforced: any group key present in the validation window is
    removed from training, so a dwelling cannot be on both sides of a split.
    """
    frame = df
    months = month_keys(frame[date_col])

    train_mask = months.between(fold.train_months[0], fold.train_months[1])
    valid_mask = months.between(fold.valid_months[0], fold.valid_months[1])

    report = {
        "fold": fold.index,
        "train_rows_before_group_filter": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "embargo_months": fold.embargo_months,
        "group_overlap_removed": 0,
    }

    # Corruption gate: both sides.
    if "iqr_corruption_ok" in frame.columns:
        corruption = frame["iqr_corruption_ok"] == True  # noqa: E712
        report["corruption_rows_excluded_train"] = int((train_mask & ~corruption).sum())
        report["corruption_rows_excluded_valid"] = int((valid_mask & ~corruption).sum())
        train_mask = train_mask & corruption
        valid_mask = valid_mask & corruption

    # Training-quality gate: training side only.
    if drop_unjudged_iqr and "iqr_keep" in frame.columns:
        keep = frame["iqr_keep"] == True  # noqa: E712
        report["iqr_quality_rows_excluded_train"] = int((train_mask & ~keep).sum())
        report["iqr_filtered"] = True
        train_mask = train_mask & keep

    if group_col and group_col in frame.columns:
        valid_groups = set(frame.loc[valid_mask, group_col].dropna().unique())
        train_groups = frame.loc[train_mask, group_col]
        overlap = train_groups.isin(valid_groups)
        report["group_overlap_removed"] = int(overlap.sum())
        report["group_overlap_share"] = float(overlap.mean()) if len(train_groups) else 0.0
        train_mask = train_mask & ~overlap

    train = frame.loc[train_mask]
    valid = frame.loc[valid_mask]
    report["train_rows"] = int(len(train))
    report["valid_months"] = [str(fold.valid_months[0]), str(fold.valid_months[1])]
    report["train_months"] = [str(fold.train_months[0]), str(fold.train_months[1])]
    return train, valid, report


def summarize_folds(folds: list[Fold], df: pd.DataFrame, date_col: str = "contract_date") -> pd.DataFrame:
    """Table of fold boundaries with row counts — goes straight into the report."""
    rows = []
    for fold in folds:
        train, valid, report = split_frame(df, fold, date_col=date_col, group_col=None)
        rows.append({
            "fold": fold.index,
            "train_start": fold.train_months[0].strftime("%Y-%m"),
            "train_end": fold.train_months[1].strftime("%Y-%m"),
            "valid_start": fold.valid_months[0].strftime("%Y-%m"),
            "valid_end": fold.valid_months[1].strftime("%Y-%m"),
            "embargo_months": fold.embargo_months,
            "train_rows": len(train),
            "valid_rows": len(valid),
        })
    return pd.DataFrame(rows)


def assert_no_leakage(df: pd.DataFrame, fold: Fold, date_col: str = "contract_date",
                      group_col: str | None = "group_key") -> None:
    """Hard checks for one fold — used by tests and before every training run."""
    train, valid, _ = split_frame(df, fold, date_col=date_col, group_col=group_col)
    if train.empty or valid.empty:
        raise AssertionError(f"{fold.label()}: empty side of the split")

    train_max = month_keys(train[date_col]).max()
    valid_min = month_keys(valid[date_col]).min()
    gap = (valid_min.year - train_max.year) * 12 + (valid_min.month - train_max.month)
    if gap < fold.embargo_months:
        raise AssertionError(
            f"{fold.label()}: embargo violated — train ends {train_max:%Y-%m}, "
            f"validation starts {valid_min:%Y-%m} (gap {gap} < {fold.embargo_months})"
        )

    if group_col and group_col in df.columns:
        overlap = set(train[group_col]) & set(valid[group_col])
        if overlap:
            raise AssertionError(f"{fold.label()}: {len(overlap)} group keys appear in both sides")
