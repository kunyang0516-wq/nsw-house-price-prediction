"""Monthly / as-of macro features: RBA cash rate target and CPI.

Why this module exists (audit M3): the group's notebook attached a **calendar-year
average** cash rate and a hard-coded annual CPI to every contract. Both are
functions of the whole year, so a January contract was given a variable that
embeds rate decisions made in December. That is temporal leakage.

This module instead builds an *as-of* join:

* **cash rate** — the RBA publishes the target with an effective date, and the
  rate changes on that day. So the value available on contract date *t* is the
  most recent target with ``effective_date <= t``. No publication lag.
* **CPI** — the index is published about four weeks after the quarter ends. The
  value available on *t* is therefore the most recent quarter whose
  ``period_end + publication_lag`` is ``<= t``. Using ``period_end <= t`` would
  be wrong: for the first ~4 weeks of a quarter the index does not exist yet.

Sources (both official RBA statistical tables, stable CSV endpoints):
  * F1.1 "Interest Rates and Yields – Money Market" -> Cash Rate Target (daily)
  * G1 "Consumer Price Inflation" -> CPI index, quarterly

Snapshots are written to ``data/external/`` so a run is reproducible offline;
the fetch is skipped when the snapshot already exists (use ``--refresh`` to
overwrite).
"""

from __future__ import annotations

import io
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import EXTERNAL_DIR

RBA_F1_URL = "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"
RBA_G1_URL = "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv"

# The HTML history table is the authoritative change log: one row per effective
# date with a clean "Cash rate target %" column (405 rows, back to 1990-01).
# The A2 CSV carries the same information but its early rows are ragged
# ("17.00 to 17.50" in a single field) and its column count varies by row.
RBA_CASH_RATE_URL = "https://www.rba.gov.au/statistics/cash-rate/"

CASH_RATE_SNAPSHOT = EXTERNAL_DIR / "rba_cash_rate_target.csv"
CPI_SNAPSHOT = EXTERNAL_DIR / "rba_cpi_index_quarterly.csv"

# RBA CSV files carry ~10 rows of title/units/notes before the header row.
_RBA_SKIPROWS = 10

# Quarterly CPI is published roughly four weeks after the quarter ends.
CPI_PUBLICATION_LAG_DAYS = 28


@dataclass(frozen=True)
class MacroPanel:
    """As-of macro series plus the metadata needed to audit the join."""

    cash_rate: pd.DataFrame  # effective_date, cash_rate
    cpi: pd.DataFrame  # period_end, release_date, cpi_index
    money_market_series: str = "Cash Rate Target"
    cpi_series: str = "GCPIAG"
    cpi_publication_lag_days: int = CPI_PUBLICATION_LAG_DAYS

    def as_of(self, dates: pd.Series) -> pd.DataFrame:
        """Return macro columns available on each supplied date.

        Output columns: ``cash_rate_asof``, ``cpi_index_asof``,
        ``cpi_asof_period_end``, ``cpi_asof_release_date``, ``cpi_yoy_asof``.
        """
        target = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

        cash = self.cash_rate.sort_values("effective_date")
        cash_sorted = cash["cash_rate"].to_numpy()
        cash_dates = cash["effective_date"].to_numpy()

        cpi = self.cpi.sort_values("release_date").copy()
        cpi["cpi_yoy"] = cpi["cpi_index"].pct_change(4) * 100
        cpi_sorted = cpi[["cpi_index", "period_end", "release_date", "cpi_yoy"]].to_numpy()
        cpi_release = cpi["release_date"].to_numpy()

        cash_out = np.full(len(target), np.nan)
        cpi_index = np.full(len(target), np.nan)
        cpi_period = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
        cpi_release_out = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
        cpi_yoy = np.full(len(target), np.nan)

        cash_pos = np.searchsorted(cash_dates, target.to_numpy(), side="right") - 1
        cpi_pos = np.searchsorted(cpi_release, target.to_numpy(), side="right") - 1

        has_cash = cash_pos >= 0
        cash_out[has_cash] = cash_sorted[cash_pos[has_cash]]

        has_cpi = cpi_pos >= 0
        cpi_index[has_cpi] = cpi_sorted[cpi_pos[has_cpi], 0].astype(float)
        cpi_period[has_cpi] = cpi_sorted[cpi_pos[has_cpi], 1].astype("datetime64[ns]")
        cpi_release_out[has_cpi] = cpi_sorted[cpi_pos[has_cpi], 2].astype("datetime64[ns]")
        cpi_yoy[has_cpi] = cpi_sorted[cpi_pos[has_cpi], 3].astype(float)

        return pd.DataFrame({
            "cash_rate_asof": cash_out,
            "cpi_index_asof": cpi_index,
            "cpi_asof_period_end": cpi_period,
            "cpi_asof_release_date": cpi_release_out,
            "cpi_yoy_asof": cpi_yoy,
        })

    def meta(self) -> dict:
        return {
            "money_market_series": self.money_market_series,
            "cpi_series": self.cpi_series,
            "cpi_publication_lag_days": self.cpi_publication_lag_days,
            "cash_rate_span": [
                str(self.cash_rate["effective_date"].min().date()),
                str(self.cash_rate["effective_date"].max().date()),
            ],
            "cpi_span": [
                str(self.cpi["period_end"].min().date()),
                str(self.cpi["period_end"].max().date()),
            ],
            "cpi_index_base_note": (
                "RBA G1 GCPIAG index. The index base differs from the annual CPI "
                "values hard-coded in the group's notebook (e.g. G1 2023Q4 = 96.81 "
                "on a 2023-24=100 basis vs their 134.43), so levels are NOT "
                "comparable across the two sources. Levels are also not "
                "comparable across index rebases. Use the YoY rate as the feature."
            ),
        }


# --------------------------------------------------------------------------- #
# Fetch / load
# --------------------------------------------------------------------------- #
def _fetch_rba_table(url: str) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    frame = pd.read_csv(io.BytesIO(payload), skiprows=_RBA_SKIPROWS)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _parse_rba_number(value) -> float:
    """Extract the first number from an RBA cell ("17.00 to 17.50" -> 17.5)."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else np.nan


def fetch_cash_rate(path: Path = CASH_RATE_SNAPSHOT, refresh: bool = False) -> pd.DataFrame:
    """Daily RBA cash rate target keyed by effective date.

    Source: the cash-rate history table on the RBA website (one row per change,
    with the day the new target took effect). Because the target changes *on*
    the effective date, the value known on contract date ``t`` is simply the most
    recent row with ``effective_date <= t`` — no publication lag applies.
    """
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["effective_date"])

    request = urllib.request.Request(RBA_CASH_RATE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tables = pd.read_html(io.StringIO(response.read().decode("utf-8", "replace")))

    selected = None
    for table in tables:
        columns = [str(c) for c in table.columns]
        date_col = next((c for c in columns if "effective date" in c.lower()), None)
        rate_col = next((c for c in columns if "cash rate target" in c.lower()), None)
        if date_col and rate_col:
            selected = pd.DataFrame({
                "effective_date": pd.to_datetime(table[date_col], format="mixed",
                                                 dayfirst=True, errors="coerce"),
                "cash_rate": [ _parse_rba_number(v) for v in table[rate_col] ],
            })
            break
    if selected is None:
        raise ValueError("No cash-rate history table found on the RBA page.")

    selected = selected.dropna(subset=["effective_date", "cash_rate"])
    conflicts = selected.groupby("effective_date")["cash_rate"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Conflicting cash rates for one effective date.")

    frame = (selected.drop_duplicates("effective_date")
             .sort_values("effective_date")
             .reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def fetch_cpi(
    path: Path = CPI_SNAPSHOT,
    refresh: bool = False,
    lag_days: int = CPI_PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Quarterly CPI index with an explicit, documented publication date."""
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["period_end", "release_date"])

    raw = _fetch_rba_table(RBA_G1_URL)
    if "GCPIAG" not in raw.columns and "Series ID" not in raw.columns:
        raise ValueError(f"Unexpected RBA G1 layout. Columns: {list(raw.columns)}")

    frame = pd.DataFrame({
        "period_end": pd.to_datetime(raw["Series ID"], format="%d/%m/%Y", errors="coerce"),
        "cpi_index": pd.to_numeric(raw["GCPIAG"], errors="coerce"),
    }).dropna()

    frame = (frame.drop_duplicates("period_end")
             .sort_values("period_end")
             .reset_index(drop=True))
    # Publication lag is an assumption, not data: make it explicit and stored.
    frame["release_date"] = frame["period_end"] + pd.Timedelta(days=lag_days)
    frame["publication_lag_days"] = lag_days

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_macro_panel(refresh: bool = False, lag_days: int = CPI_PUBLICATION_LAG_DAYS) -> MacroPanel:
    cash = fetch_cash_rate(refresh=refresh)
    cpi = fetch_cpi(refresh=refresh, lag_days=lag_days)
    return MacroPanel(cash_rate=cash, cpi=cpi, cpi_publication_lag_days=lag_days)


# --------------------------------------------------------------------------- #
# Leakage checks (audit §3.3 / T6)
# --------------------------------------------------------------------------- #
def assert_macro_asof_valid(joined: pd.DataFrame, date_col: str = "contract_date") -> None:
    """Hard guarantees about the as-of join. Raise if any is violated."""
    dates = pd.to_datetime(joined[date_col], errors="coerce")

    released = joined["cpi_asof_release_date"]
    bad = released.notna() & (released > dates)
    if bad.any():
        raise AssertionError(
            f"{int(bad.sum())} rows use a CPI release that post-dates the contract."
        )

    if "cpi_asof_period_end" in joined:
        inside = joined["cpi_asof_period_end"].notna() & (joined["cpi_asof_period_end"] > dates)
        if inside.any():
            raise AssertionError(
                f"{int(inside.sum())} rows use a CPI quarter that had not even ended."
            )

    missing_cash = joined["cash_rate_asof"].isna()
    if missing_cash.any():
        raise AssertionError(
            f"{int(missing_cash.sum())} rows have no cash rate on or before the contract date."
        )


def macro_coverage_report(joined: pd.DataFrame, date_col: str = "contract_date") -> pd.DataFrame:
    """Per-year summary: how stale is the CPI information at contract time?"""
    frame = joined.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["year"] = frame[date_col].dt.year
    frame["cpi_lag_days"] = (frame[date_col] - frame["cpi_asof_period_end"]).dt.days
    return (frame.groupby("year", observed=True)
            .agg(rows=(date_col, "size"),
                 cash_rate_mean=("cash_rate_asof", "mean"),
                 cpi_index_median=("cpi_index_asof", "median"),
                 cpi_yoy_median=("cpi_yoy_asof", "median"),
                 cpi_lag_days_min=("cpi_lag_days", "min"),
                 cpi_lag_days_median=("cpi_lag_days", "median"),
                 cpi_lag_days_max=("cpi_lag_days", "max"))
            .reset_index())
