"""Regression test for the quality-fence key mapping.

A post-training-span year must fall back to the global training fence, not be
rejected outright. Before the fix, `Series.map` with a dict keyed by numpy strings
returned all-NaN for DataFrame columns of other string dtypes, so **every row
after 2015 was silently dropped from training** while the validation windows were
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.outliers import (
    GLOBAL_GROUP,
    fit_global_bounds,
    fit_iqr_bounds,
    mark_quality,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Two well-behaved years plus one year beyond the fitting span.

    The out-of-span year deliberately shares the *same* price and area
    distribution as the fitted years. That is what isolates the key-mapping bug:
    if the fixture used a higher price level the corruption fence would reject
    those rows on its own and the test would pass for the wrong reason (as it did
    on the first attempt at this test).
    """
    rng = np.random.default_rng(7)
    rows = []
    for year, base in ((2014, 500_000), (2015, 510_000), (2019, 520_000)):
        for _ in range(400):
            rows.append({
                "contract_date": pd.Timestamp(f"{year}-06-15"),
                "purchase_price": float(base + rng.normal(0, 20_000)),
                "area_sqm": float(rng.normal(600, 40)),
            })
    return pd.DataFrame(rows)


def test_year_beyond_span_uses_the_global_fence(frame):
    train = frame.loc[frame["contract_date"].dt.year <= 2015]
    bounds = fit_iqr_bounds(train, by=("year",))
    global_bounds = fit_global_bounds(train)

    marked = mark_quality(frame, global_bounds, bounds)

    beyond = marked.loc[marked["contract_date"].dt.year == 2019]
    assert len(beyond) > 0
    # The whole point: these rows are judged by the global fence, not dropped.
    assert beyond["iqr_keep"].sum() > 0, "2019 rows were all rejected (map returned NaN)"
    assert beyond["iqr_keep"].mean() > 0.5, (
        f"only {beyond['iqr_keep'].mean():.1%} of 2019 rows kept; expected most to pass")
    # In-span years are judged by their own fence and should also mostly pass.
    inside = marked.loc[marked["contract_date"].dt.year.between(2014, 2015)]
    assert inside["iqr_keep"].mean() > 0.9


def test_global_group_is_present_and_used(frame):
    train = frame.loc[frame["contract_date"].dt.year <= 2015]
    bounds = fit_iqr_bounds(train, by=("year",))
    assert (GLOBAL_GROUP, "purchase_price") in bounds.bounds
    assert "2014" in {g for (g, _) in bounds.bounds}
    assert "2019" not in {g for (g, _) in bounds.bounds}


def test_obvious_corruption_is_still_rejected(frame):
    train = frame.loc[frame["contract_date"].dt.year <= 2015]
    bounds = fit_iqr_bounds(train, by=("year",))
    global_bounds = fit_global_bounds(train)

    corrupted = frame.copy()
    corrupted.loc[corrupted.index[0], "area_sqm"] = 1.53e9
    corrupted.loc[corrupted.index[1], "purchase_price"] = 8.75e8
    marked = mark_quality(corrupted, global_bounds, bounds)
    assert not bool(marked.loc[corrupted.index[0], "iqr_corruption_ok"])
    assert not bool(marked.loc[corrupted.index[1], "iqr_corruption_ok"])
