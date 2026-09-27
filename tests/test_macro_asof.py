"""Macro as-of join guards (audit §3.3 / T6).

These run against synthetic series so they are fast and offline; one optional
test exercises the real RBA snapshots when they exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.macro_asof import (
    CASH_RATE_SNAPSHOT,
    CPI_SNAPSHOT,
    MacroPanel,
    assert_macro_asof_valid,
    fetch_cpi,
    load_macro_panel,
    macro_coverage_report,
)


@pytest.fixture(scope="module")
def synthetic_panel() -> MacroPanel:
    """Quarterly CPI with a 28-day publication lag and three rate changes."""
    cash = pd.DataFrame({
        "effective_date": pd.to_datetime(["2018-01-01", "2018-09-01", "2019-06-04", "2020-03-19"]),
        "cash_rate": [1.50, 1.00, 0.75, 0.25],
    })
    quarter_ends = pd.date_range("2017-03-31", "2020-12-31", freq="QE")
    cpi = pd.DataFrame({
        "period_end": quarter_ends,
        "cpi_index": np.linspace(90.0, 110.0, len(quarter_ends)),
    })
    cpi["release_date"] = cpi["period_end"] + pd.Timedelta(days=28)
    return MacroPanel(cash_rate=cash, cpi=cpi)


def test_cash_rate_uses_latest_effective_date(synthetic_panel):
    dates = pd.Series(pd.to_datetime([
        "2018-01-01",   # change on the day itself
        "2018-06-30",   # between changes
        "2018-09-01",
        "2019-06-03",   # day before a change
        "2019-06-04",   # change day
        "2021-01-01",   # after the last change
    ]))
    result = synthetic_panel.as_of(dates)
    assert result["cash_rate_asof"].tolist() == [1.50, 1.50, 1.00, 1.00, 0.75, 0.25]


def test_cpi_respects_publication_lag(synthetic_panel):
    """A quarter that has ended but not been published must not be used."""
    # 2019-01-31: 2018 Q4 ended 2018-12-31 but releases 2019-01-28, so Q4 is usable.
    # 2019-01-10: only 2018 Q3 (released 2018-10-28) is usable.
    dates = pd.Series(pd.to_datetime(["2019-01-10", "2019-01-31"]))
    result = synthetic_panel.as_of(dates)
    assert result.loc[0, "cpi_asof_period_end"] == pd.Timestamp("2018-09-30")
    assert result.loc[1, "cpi_asof_period_end"] == pd.Timestamp("2018-12-31")


def test_never_uses_a_release_after_the_contract(synthetic_panel):
    dates = pd.Series(pd.date_range("2018-01-01", "2020-12-31", freq="D"))
    result = synthetic_panel.as_of(dates)
    joined = pd.DataFrame({"contract_date": dates}).join(result)
    assert_macro_asof_valid(joined)  # must not raise


def test_perturbing_future_macro_changes_nothing_past(synthetic_panel):
    """T2 for macro: an as-of lookup is a pure function of the past."""
    past = pd.Series(pd.to_datetime(["2019-01-15", "2019-06-20", "2020-01-05"]))
    baseline = synthetic_panel.as_of(past)

    tampered_cash = synthetic_panel.cash_rate.copy()
    tampered_cash.loc[tampered_cash["effective_date"] > "2019-12-31", "cash_rate"] = 99.0
    tampered_cpi = synthetic_panel.cpi.copy()
    tampered_cpi.loc[tampered_cpi["release_date"] > "2019-12-31", "cpi_index"] = 999.0
    tampered = MacroPanel(cash_rate=tampered_cash, cpi=tampered_cpi)

    after = tampered.as_of(past)
    pd.testing.assert_frame_equal(baseline, after)


def test_validator_catches_a_leaky_join(synthetic_panel):
    joined = pd.DataFrame({
        "contract_date": pd.to_datetime(["2019-01-05"]),
        "cash_rate_asof": [0.75],
        # A release a month in the future: must be rejected.
        "cpi_asof_release_date": [pd.Timestamp("2019-02-28")],
        "cpi_asof_period_end": [pd.Timestamp("2018-12-31")],
    })
    with pytest.raises(AssertionError):
        assert_macro_asof_valid(joined)


def test_coverage_report_shape(synthetic_panel):
    dates = pd.Series(pd.date_range("2018-01-01", "2019-12-31", freq="D"))
    joined = pd.DataFrame({"contract_date": dates}).join(synthetic_panel.as_of(dates))
    report = macro_coverage_report(joined)
    assert {"year", "rows", "cash_rate_mean", "cpi_lag_days_median"} <= set(report.columns)
    assert (report["cpi_lag_days_min"] >= 0).all()


@pytest.mark.skipif(not CASH_RATE_SNAPSHOT.exists(), reason="RBA cash-rate snapshot not downloaded")
def test_real_cash_rate_snapshot_is_monotonic():
    frame = pd.read_csv(CASH_RATE_SNAPSHOT, parse_dates=["effective_date"])
    assert frame["effective_date"].is_monotonic_increasing
    assert not frame["effective_date"].duplicated().any()
    assert frame["cash_rate"].between(0, 25).all()


@pytest.mark.skipif(not CPI_SNAPSHOT.exists(), reason="CPI snapshot not downloaded")
def test_real_cpi_snapshot_has_publication_lag():
    frame = pd.read_csv(CPI_SNAPSHOT, parse_dates=["period_end", "release_date"])
    assert (frame["release_date"] > frame["period_end"]).all()
    assert frame["period_end"].is_monotonic_increasing
