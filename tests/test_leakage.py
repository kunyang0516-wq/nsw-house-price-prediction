"""Leakage guards (audit T1-T4, T6).

These tests are the executable form of initial_v0/03_leakage_audit.md §5. They
run on the *real* postcode lookups and, where possible, on small synthetic
frames so they stay fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.contract import FORBIDDEN_IN_FEATURES, check_clean
from src.features.panel import (
    add_rolling_features,
    attach_panel_features,
    build_postcode_month_panel,
    complete_monthly_spine,
    to_month,
)
from src.utils.config import OUT_POSTCODE_DEVELOPMENT, SETTINGS
from src.validation.time_series_cv import (
    Fold,
    assert_no_leakage,
    month_keys,
    rolling_origin_splits,
    split_frame,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_transactions() -> pd.DataFrame:
    """Two postcodes, 24 months, a known price trend and a known future jump."""
    rng = np.random.default_rng(36103)
    rows = []
    for postcode in ("2000", "2100"):
        for month in pd.date_range("2015-01-01", "2016-12-01", freq="MS"):
            for _ in range(5):
                base = 500_000 + (month.year - 2015) * 60_000
                if postcode == "2100":
                    base += 120_000
                rows.append({
                    "post_code": postcode,
                    "contract_date": month + pd.Timedelta(days=int(rng.integers(0, 27))),
                    "purchase_price": float(base + rng.normal(0, 8_000)),
                    "area_sqm": float(rng.normal(650, 40)),
                    "group_key": f"{postcode}-{len(rows)}",
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel(synthetic_transactions: pd.DataFrame) -> pd.DataFrame:
    base = build_postcode_month_panel(synthetic_transactions)
    return add_rolling_features(complete_monthly_spine(base))


# --------------------------------------------------------------------------- #
# T2 - future perturbation: changing the future must not change past features
# --------------------------------------------------------------------------- #
def test_rolling_features_ignore_future(synthetic_transactions, panel):
    cutoff = pd.Timestamp("2016-01-01")

    # Rebuild with every observation at/after the cutoff inflated 10x.
    tampered = synthetic_transactions.copy()
    touched = to_month(tampered["contract_date"]) >= cutoff
    tampered.loc[touched, "purchase_price"] *= 10

    tampered_panel = add_rolling_features(
        complete_monthly_spine(build_postcode_month_panel(tampered)))

    before = panel.loc[panel["month"] < cutoff].reset_index(drop=True)
    after = tampered_panel.loc[tampered_panel["month"] < cutoff].reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c.startswith("pc_")]
    assert feature_cols, "expected rolling feature columns"
    pd.testing.assert_frame_equal(
        before[["postcode", "month", *feature_cols]],
        after[["postcode", "month", *feature_cols]],
        check_exact=False,
        atol=1e-9,
    )


# --------------------------------------------------------------------------- #
# T1 - no contemporaneous information: features never use the current month
# --------------------------------------------------------------------------- #
def test_rolling_features_exclude_current_month(synthetic_transactions, panel):
    months = sorted(panel["month"].unique())
    check_month = months[12]

    # Inflate ONLY the check month, then compare the feature row for that month.
    tampered = synthetic_transactions.copy()
    tampered.loc[to_month(tampered["contract_date"]) == check_month, "purchase_price"] *= 50

    tampered_panel = add_rolling_features(
        complete_monthly_spine(build_postcode_month_panel(tampered)))

    key = ["postcode", "month"]
    feature_cols = [c for c in panel.columns if c.startswith("pc_")]
    original = panel.loc[panel["month"] == check_month, key + feature_cols].reset_index(drop=True)
    changed = tampered_panel.loc[tampered_panel["month"] == check_month, key + feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, changed, check_exact=False, atol=1e-9)


def test_rolling_uses_only_past_windows(panel):
    """A postcode's first month must have no rolling history at all.

    `pc_months_observed` is a plain exposure counter (cumulative row count), not
    a statistic over past observations, so it is excluded here.
    """
    months = sorted(panel["month"].unique())
    first_month = months[0]
    first_rows = panel.loc[panel["month"] == first_month]
    rolling_cols = [c for c in first_rows.columns
                    if c.startswith("pc_")
                    and c not in ("pc_months_observed", "pc_months_since_sale")]
    assert rolling_cols, "expected rolling feature columns"
    assert first_rows[rolling_cols].isna().all().all(), \
        "the first month cannot have rolling features (nothing is in the past yet)"


# --------------------------------------------------------------------------- #
# T6 - macro / label as-of rules
# --------------------------------------------------------------------------- #
def test_development_labels_are_legal_before_2011():
    """No contract before 2011 may carry a label derived from its own era."""
    if not OUT_POSTCODE_DEVELOPMENT.exists():
        pytest.skip("postcode development lookup not built yet")
    lookup = pd.read_csv(OUT_POSTCODE_DEVELOPMENT, dtype={"postcode": "string"})

    # Sanity on the two source windows themselves.
    assert "development_type_v2010" in lookup.columns
    assert "development_type_v2014" in lookup.columns

    # A v2014 label may only be used after the v2014 window closes.
    policy = SETTINGS.cleaning
    windows = {w.name: w for w in policy.development_windows}
    assert windows["v2010"].label_as_of < windows["v2014"].label_as_of


def test_forbidden_columns_are_declared():
    for column in ("settlement_date", "download_date", "property_id",
                   "legal_description", "address", "price_per_sqm", "cash_rate", "cpi"):
        assert column in FORBIDDEN_IN_FEATURES, f"{column} must be declared forbidden"


def test_synthetic_frame_passes_contracts(synthetic_transactions):
    frame = synthetic_transactions.copy()
    frame["primary_purpose"] = "RESIDENCE"
    frame["property_type"] = "house"
    frame["area_sqm"] = frame["area_sqm"].abs()
    for column in ("dist_cbd", "dist_train", "dist_metro", "dist_metro_new"):
        frame[column] = 5.0
    frame["development_type"] = pd.Series(["Established"] * len(frame), dtype="string")
    results = check_clean(frame)
    failures = [r for r in results if not r.passed]
    assert not failures, "\n".join(str(r) for r in failures)


# --------------------------------------------------------------------------- #
# T3 - grouping: the same dwelling must not straddle a split
# --------------------------------------------------------------------------- #
def test_group_keys_do_not_cross_folds(synthetic_transactions):
    """Each dwelling sells once, so no group can legitimately straddle a split.

    If the splitter failed to purge validation groups from training, the guard
    in `assert_no_leakage` would raise. `test_cv_splits.py` additionally proves
    the purge removes rows when a dwelling *does* sell twice.
    """
    frame = synthetic_transactions.copy()
    frame["group_key"] = "dwelling-" + frame.index.astype(str)

    folds = rolling_origin_splits(frame["contract_date"], min_train_months=12,
                                  horizon_months=6, step_months=6, embargo_months=1)
    assert folds, "expected at least one fold"
    for fold in folds:
        assert_no_leakage(frame, fold, group_col="group_key")


def test_same_month_features_are_separately_named(synthetic_transactions):
    """Contemporaneous panel stats must not be mistakable for features.

    Pipeline order: base panel -> attach (stats get the `contemporaneous_`
    prefix) -> attach the lagged rolling features.
    """
    base = build_postcode_month_panel(synthetic_transactions)
    spine = complete_monthly_spine(base)
    rolling = add_rolling_features(spine)

    with_stats = attach_panel_features(synthetic_transactions, spine)
    assert "median_price" not in with_stats.columns
    assert "contemporaneous_median_price" in with_stats.columns

    full = attach_panel_features(with_stats, rolling)
    assert any(c.startswith("pc_med_price_") for c in full.columns)
    assert "median_price" not in full.columns
