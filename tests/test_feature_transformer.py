"""Regression tests for the fold-aware feature transformer.

`PropertyFeatureTransformer` had no test coverage at all, which is how a crash went
unnoticed: `fit` computed a numeric imputation value with `float(median)`, and when a
column had **no usable value anywhere in the training fold** the median is `pd.NA`, so
`float(pd.NA)` raised `TypeError: float() argument must be a string or a real number, not
'NAType'`.

That is not a contrived case. `development_type` is NA for every contract before 2011 by
construction -- the earliest legally-timed label window closes in 2010, which is the
project's own leakage fix -- so the oldest rolling-origin folds have no usable value for
it at all. The ablation could therefore not be run on an early fold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.features import FeatureSpec, PropertyFeatureTransformer, make_xy


def _frame(n: int = 300, seed: int = 36103) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "contract_date": pd.date_range("2015-01-01", periods=n, freq="D"),
        "area_sqm": rng.normal(600, 60, n),
        "log_area": rng.normal(2.8, 0.1, n),
        "dist_cbd": rng.uniform(2, 60, n),
        "year_num": rng.uniform(2001, 2023, n),
        "post_code": rng.choice(["2000", "2153", "2768"], n),
        "development_type": rng.choice(["Established", "Greenfield"], n),
        "log_price": rng.normal(6.0, 0.3, n),
    })


SPEC = FeatureSpec(
    numeric=("area_sqm", "log_area", "dist_cbd", "year_num"),
    categorical_low_card=("development_type",),
    categorical_high_card=("post_code",),
)


def test_fit_does_not_crash_when_a_numeric_column_is_entirely_missing():
    """The crash: an all-NA numeric column makes the median `pd.NA`."""
    frame = _frame()
    frame["dist_cbd"] = np.nan          # nothing usable in this fold
    X, y, _ = make_xy(frame, SPEC)

    transformer = PropertyFeatureTransformer(SPEC)
    transformer.fit(X, y)               # used to raise TypeError

    assert transformer.impute_values_["dist_cbd"] == 0.0
    assert "dist_cbd" in transformer.degenerate_imputations_


def test_degenerate_column_is_recorded_not_silent():
    frame = _frame()
    frame["dist_cbd"] = np.nan
    X, y, _ = make_xy(frame, SPEC)
    transformer = PropertyFeatureTransformer(SPEC).fit(X, y)

    # A healthy column must NOT be recorded as degenerate.
    assert "area_sqm" not in transformer.degenerate_imputations_
    assert transformer.impute_values_["area_sqm"] == pytest.approx(
        float(frame["area_sqm"].median()), rel=1e-9)


def test_transform_still_produces_finite_output_for_a_degenerate_fold():
    frame = _frame()
    frame["dist_cbd"] = np.nan
    X, y, _ = make_xy(frame, SPEC)
    transformer = PropertyFeatureTransformer(SPEC).fit(X, y)
    Z = transformer.transform(X)

    assert np.isfinite(Z).all(), "constant imputation must not produce NaN/inf"
    assert len(transformer.get_feature_names_out()) == Z.shape[1]


def test_all_missing_categorical_is_handled():
    """Categoricals go through frequency/target encoding, which fills '<missing>'."""
    frame = _frame()
    frame["development_type"] = pd.NA
    X, y, _ = make_xy(frame, SPEC)
    transformer = PropertyFeatureTransformer(SPEC).fit(X, y)
    Z = transformer.transform(X)

    assert np.isfinite(Z).all()


def test_numeric_and_categorical_roles_are_disjoint():
    """A column must not be processed as both a number and a category.

    `pipelines/08_feature_value_test.py` used to pass one flat list of "everything in this
    layer" as `FeatureSpec.numeric`, so the five string columns were also coerced to
    numbers, imputed with a constant and standardised -- emitting a useless
    `num__<name>` alongside their real encoding, and crashing `fit` on any fold where such
    a column was wholly missing.
    """
    frame = _frame()
    X, y, _ = make_xy(frame, SPEC)
    transformer = PropertyFeatureTransformer(SPEC).fit(X, y)
    names = list(transformer.get_feature_names_out())

    numeric_prefixes = tuple(f"num__{c}" for c in SPEC.numeric)
    category_names = set(SPEC.categorical_low_card) | set(SPEC.categorical_high_card)
    assert not (set(SPEC.numeric) & category_names), "spec itself must be disjoint"

    for name in names:
        for category in category_names:
            assert not (name.startswith(f"num__{category}")
                        and name in numeric_prefixes), (
                f"{name} treats the categorical '{category}' as numeric")


def test_length_mismatch_between_X_and_y_is_rejected():
    frame = _frame()
    X, y, _ = make_xy(frame, SPEC)
    with pytest.raises(ValueError):
        PropertyFeatureTransformer(SPEC).fit(X, y.iloc[:-1])
