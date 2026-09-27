"""Cross-validation splitter contracts (audit T3, §6.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.time_series_cv import (
    assert_no_leakage,
    month_keys,
    rolling_origin_splits,
    split_frame,
    summarize_folds,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """36 months x 40 dwellings, each dwelling selling twice."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2018-01-01", "2020-12-01", freq="MS")
    rows = []
    for dwelling in range(40):
        for sale in range(2):
            month = months[rng.integers(0, len(months))]
            rows.append({
                "contract_date": month + pd.Timedelta(days=int(rng.integers(0, 27))),
                "purchase_price": float(400_000 + 1_000 * dwelling + rng.normal(0, 5_000)),
                "area_sqm": 600.0,
                "group_key": f"dwelling-{dwelling}",
            })
    return pd.DataFrame(rows)


def test_folds_are_forward_chaining(frame):
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=1)
    assert len(folds) >= 3
    for earlier, later in zip(folds, folds[1:]):
        assert later.valid_months[0] > earlier.valid_months[0], "validation windows must move forward"
        assert later.train_months[1] >= earlier.train_months[1], "training origin must not move back"


def test_embargo_is_respected(frame):
    embargo = 2
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=embargo)
    for fold in folds:
        train, valid, _ = split_frame(frame, fold, group_col=None)
        gap_months = (valid["contract_date"].dt.to_period("M").min()
                      - train["contract_date"].dt.to_period("M").max()).n
        assert gap_months > embargo, f"fold {fold.index}: gap {gap_months} <= embargo {embargo}"


def test_group_keys_are_purged(frame):
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=1)
    for fold in folds:
        train, valid, report = split_frame(frame, fold, group_col="group_key")
        assert not (set(train["group_key"]) & set(valid["group_key"]))
        assert report["group_overlap_removed"] >= 0
        assert_no_leakage(frame, fold, group_col="group_key")


def test_sliding_mode_keeps_window_length(frame):
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=1,
                                  mode="sliding", window_months=12)
    for fold in folds:
        length = (fold.train_months[1].to_period("M") - fold.train_months[0].to_period("M")).n + 1
        assert length == 12


def test_summary_shape(frame):
    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=1)
    table = summarize_folds(folds, frame)
    assert list(table.columns) == ["fold", "train_start", "train_end", "valid_start",
                                   "valid_end", "embargo_months", "train_rows", "valid_rows"]
    assert (table["train_rows"] > 0).all()
    assert (table["valid_rows"] > 0).all()


def test_invalid_mode_raises(frame):
    with pytest.raises(ValueError):
        rolling_origin_splits(frame["contract_date"], mode="random")
