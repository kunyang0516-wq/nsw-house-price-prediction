"""Data contract: the assertions that guard every stage boundary.

If a contract fails, stop — do not "fix" the data silently. The numbers below
were measured on the current raw extract; when the source data is refreshed,
update EXPECTED deliberately and note it in the changelog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.clean import DUPLICATE_KEY
from src.utils.io import read_postcode_key


@dataclass(frozen=True)
class ContractResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.detail}"


# Measured on nsw_property_data.csv (610.9 MiB) — see initial_v0/01_eda_walkthrough.md
EXPECTED_RAW_ROWS = 4_854_814
EXPECTED_RAW_COLUMNS = 17

# Measured on the group's cleaning output (their single full-sample IQR fence)
GROUP_CLEAN_ROWS = 1_867_040
GROUP_UNIQUE_POSTCODES = 545

# The forbidden list is the machine-readable form of initial_v0/03_leakage_audit.md §1
FORBIDDEN_IN_FEATURES = (
    "purchase_price",
    "settlement_date",
    "download_date",
    "property_id",
    "legal_description",
    "address",
    "price_per_sqm",
    "cash_rate",
    "cpi",
)


def check_raw(df: pd.DataFrame, expected_rows: int | None = EXPECTED_RAW_ROWS) -> list[ContractResult]:
    """Pre-cleaning contract for the raw file."""
    results = [
        ContractResult("raw.rows", expected_rows is None or len(df) == expected_rows,
                       f"{len(df):,} rows (expected {expected_rows:,})" if expected_rows else f"{len(df):,} rows"),
        ContractResult("raw.columns", len(df.columns) == EXPECTED_RAW_COLUMNS,
                       f"{len(df.columns)} columns"),
    ]
    return results


def check_clean(df: pd.DataFrame, expected_rows: int | None = None) -> list[ContractResult]:
    """Post-cleaning contract. Every rule here maps to a leakage-audit line."""
    checks: list[ContractResult] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(ContractResult(name, bool(ok), detail))

    # --- schema / identity -------------------------------------------------
    add("clean.has_group_key", "group_key" in df.columns,
        "group_key present" if "group_key" in df.columns else "group_key MISSING")

    # --- de-duplication (identity key, D1/D2) ------------------------------
    # Whole-row duplicates are the wrong test: the modelling frame keeps a subset
    # of columns, so rows that differ only in a dropped column reappear as
    # duplicates there and double-weight one transaction.
    #
    # A frame that never went through `build_clean` (hand-built test frames)
    # lacks `property_id` and `area`, the two columns that exist only before the
    # cleaning stage trims the frame. The rule does not apply there.
    #
    # NOTE: `build_clean` drops `area` at the very end, so on the final
    # `clean.parquet` only `property_id` is left and the full key can no longer
    # be re-checked. The authoritative assertion therefore lives in
    # `build_clean` itself (before the column drop); this is the second line of
    # defence for frames that still carry every key column.
    key = [c for c in DUPLICATE_KEY if c in df.columns]
    if not any(c in df.columns for c in ("property_id", "area")):
        add("clean.no_duplicate_identity", True,
            "n/a - not a cleaning-pipeline frame (no property_id/area columns)")
    elif len(key) == len(DUPLICATE_KEY):
        dup = int(df.duplicated(subset=key).sum())
        add("clean.no_duplicate_identity", dup == 0,
            f"0 duplicate sale events on {key}" if dup == 0
            else f"{dup:,} duplicate sale events on {key}")
    elif "area" not in df.columns:
        # Post-`build_clean` frame: `area` was intentionally dropped there, so
        # fall back to the identity key that survives.
        surviving = ["property_id", "contract_date", "purchase_price", "area_sqm"]
        if all(c in df.columns for c in surviving):
            dup = int(df.duplicated(subset=surviving).sum())
            add("clean.no_duplicate_identity", dup == 0,
                f"0 duplicate sale events on {surviving}" if dup == 0
                else f"{dup:,} duplicate sale events on {surviving}")
        else:
            add("clean.no_duplicate_identity", True,
                "n/a - identity columns already trimmed by build_clean")
    else:
        add("clean.no_duplicate_identity", False,
            f"identity key incomplete: missing {sorted(set(DUPLICATE_KEY) - set(key))}")

    # --- group_key integrity (M2) ------------------------------------------
    # A group_key must not straddle two dwellings. `area_sqm` is near-unique to a
    # dwelling, so a group_key covering many distinct areas signals the
    # bare-street-name collapse this fallback exists to prevent.
    #
    # The ceiling is a *corruption* guard, not a precision target: addresses
    # without a house number are an apartment block or a whole estate in the
    # source data (the worst case is 359 units on 'BRODIE SPARK DR, WOLLI
    # CREEK', which carries no number and no strata lot for 12,619 of its
    # rows). No key can separate those, so the guard only has to catch a
    # regression that is an order of magnitude worse than the observed 359.
    if {"group_key", "area_sqm"} <= set(df.columns):
        per_key = df.groupby("group_key", observed=True)["area_sqm"].nunique()
        worst = int(per_key.max()) if len(per_key) else 0
        add("clean.group_key_not_over_merged", worst <= 1000,
            f"max distinct areas within one group_key: {worst}")

    # --- category filters (group's rule) -----------------------------------
    purpose_ok = df["primary_purpose"].str.strip().str.upper().eq("RESIDENCE").all()
    add("clean.purpose_residence", purpose_ok, "all RESIDENCE" if purpose_ok else "non-RESIDENCE rows present")

    type_ok = df["property_type"].str.strip().str.lower().eq("house").all()
    add("clean.property_type_house", type_ok, "all house" if type_ok else "non-house rows present")

    # --- value rules -------------------------------------------------------
    price_area_ok = bool((df[["purchase_price", "area_sqm"]].gt(0) & np.isfinite(df[["purchase_price", "area_sqm"]])).all().all())
    add("clean.positive_price_area", price_area_ok, "price & area finite and > 0")

    # --- absolute plausibility (the guard that the IQR fence cannot provide) --
    # These are the same constants as `PlausibilityPolicy`. They exist as
    # assertions, not just as a filter, so that a future change to `area_type`
    # units (e.g. an acre code treated as square metres) fails loudly here
    # instead of silently feeding 1e9-sqm parcels to the model.
    from src.utils.config import SETTINGS as _SETTINGS
    limits = _SETTINGS.plausibility
    area = pd.to_numeric(df["area_sqm"], errors="coerce")
    price = pd.to_numeric(df["purchase_price"], errors="coerce")
    unit = price / area.replace(0, np.nan)

    add("clean.area_within_plausible_range",
        bool(area.between(limits.area_sqm_min, limits.area_sqm_max).all()),
        f"area_sqm in [{limits.area_sqm_min:,.0f}, {limits.area_sqm_max:,.0f}]"
        f" | observed {area.min():,.0f}..{area.max():,.0f}")
    add("clean.price_within_plausible_range",
        bool(price.between(limits.price_min, limits.price_max).all()),
        f"price in [{limits.price_min:,.0f}, {limits.price_max:,.0f}]"
        f" | observed {price.min():,.0f}..{price.max():,.0f}")
    add("clean.unit_price_within_plausible_range",
        bool(unit.between(limits.unit_price_min, limits.unit_price_max).all()),
        f"price_per_sqm in [{limits.unit_price_min:,.0f}, {limits.unit_price_max:,.0f}]"
        f" | observed {unit.min():,.2f}..{unit.max():,.2f}")
    add("clean.no_megaparcel", bool(area.le(limits.area_sqm_max).all()),
        f"{int(area.gt(limits.area_sqm_max).sum())} rows above the plausible area ceiling")

    dist_cols = [c for c in ("dist_cbd", "dist_train", "dist_metro", "dist_metro_new") if c in df.columns]
    dist_ok = not df[dist_cols].lt(0).any().any()
    add("clean.non_negative_distances", dist_ok, f"{len(dist_cols)} distance columns >= 0")

    # --- target leakage guards --------------------------------------------
    # price_per_sqm is target / feature: it must never appear in the clean frame.
    contaminated = [c for c in ("price_per_sqm", "target_encoded_price") if c in df.columns]
    add("clean.no_derived_target_feature", not contaminated,
        "no target-derived columns in frame" if not contaminated else f"leaky columns present: {contaminated}")

    add("clean.cash_rate_is_asof", "cash_rate" not in df.columns or "cash_rate_asof" in df.columns,
        "annual cash_rate absent or superseded by cash_rate_asof")

    # --- time span ---------------------------------------------------------
    dates = pd.to_datetime(df["contract_date"], errors="coerce")
    add("clean.contract_date_parsed", dates.notna().all(),
        f"{dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}" if dates.notna().any() else "no parseable dates")

    if expected_rows is not None:
        add("clean.row_count", len(df) == expected_rows,
            f"{len(df):,} rows (expected {expected_rows:,})")

    # --- development label legality ---------------------------------------
    # `development_type` is the single legality-checked column: it is filled only
    # for contracts that fall *after* a label window has closed. Before 2011 no
    # window has closed, so a contract there must carry no label at all. The
    # versioned source columns (`development_type_v2010`, `_v2014`) are lookup
    # artefacts and must never be features.
    if "development_type" in df.columns:
        pre = dates.le(pd.Timestamp("2010-12-31"))
        illegal = int((pre & df["development_type"].notna()).sum())
        add("clean.dev_label_legality", illegal == 0,
            f"{illegal} pre-2011 rows carry a legal label (expected 0)")
        labelled = int(df["development_type"].notna().sum())
        add("clean.dev_label_present_from_2011", labelled > 0,
            f"{labelled:,} rows carry a legal label")

    return checks


def assert_all(results: list[ContractResult]) -> None:
    """Raise with a readable summary if any contract failed."""
    failures = [r for r in results if not r.passed]
    if failures:
        lines = "\n".join(str(r) for r in results)
        raise AssertionError(f"{len(failures)} contract(s) failed:\n{lines}")


def postcode_coverage(df: pd.DataFrame) -> dict:
    """Small descriptive block used in reports and logs."""
    postcodes = read_postcode_key(df["post_code"]).dropna()
    return {
        "rows": int(len(df)),
        "unique_postcodes": int(postcodes.nunique()),
        "unique_group_keys": int(df["group_key"].nunique()) if "group_key" in df else None,
        "rows_on_duplicated_group_keys": int(df["group_key"].duplicated(keep=False).sum()) if "group_key" in df else None,
    }
