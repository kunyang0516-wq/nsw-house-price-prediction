"""Assemble the modelling frame: clean transactions + macro as-of + panel features.

Deterministic, fold-independent preparation. Nothing here estimates a statistic
from the target, so it is safe to run once over the whole history. Anything that
*does* learn from data (imputation medians, target encodings, bin edges,
standardisation) lives in `src/models/features.py` and is fitted per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.macro_asof import (
    MacroPanel,
    assert_macro_asof_valid,
    load_macro_panel,
    macro_coverage_report,
)
from src.features.panel import (
    add_rolling_features,
    attach_panel_features,
    build_postcode_month_panel,
    complete_monthly_spine,
)
from src.utils.config import OUT_CLEAN_PARQUET, PROCESSED_DIR

OUT_FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
OUT_PANEL_PARQUET = PROCESSED_DIR / "panel_postcode_month.parquet"

# Columns carried into the modelling frame (leakage-safe subset only).
BASE_COLUMNS = (
    "contract_date", "purchase_price", "area_sqm", "post_code", "council_name",
    "locality", "zoning", "development_type", "dist_cbd", "dist_train",
    "dist_metro", "dist_metro_new", "group_key", "iqr_keep", "iqr_corruption_ok",
)


def load_clean(path=OUT_CLEAN_PARQUET) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["contract_date"] = pd.to_datetime(frame["contract_date"], errors="coerce")
    return frame


def clean_zoning(series: pd.Series, min_count: int = 500) -> pd.Series:
    """Collapse rare zoning codes so the low-cardinality block stays small."""
    codes = series.astype("string").str.strip().str.upper()
    counts = codes.value_counts()
    rare = counts[counts < min_count].index
    return codes.where(~codes.isin(rare), "OTHER")


def add_calendar_features(frame: pd.DataFrame, date_col: str = "contract_date") -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = dates.dt.year.astype("Int64").astype("string")
    out["month_num"] = dates.dt.month.astype("Int64").astype("string")
    out["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    out["year_num"] = dates.dt.year + (dates.dt.month - 1) / 12
    return out


def add_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Monotone transforms only — no target information involved."""
    out = frame.copy()
    # Explicit float cast: parquet round-trips integer columns as nullable Int64,
    # and masked integer arrays reject float operations (clip, log) downstream.
    out["area_sqm"] = pd.to_numeric(out["area_sqm"], errors="coerce").astype("float64")
    out["log_area"] = np.log10(out["area_sqm"].where(out["area_sqm"] > 0))

    dist = pd.to_numeric(out["dist_cbd"], errors="coerce").astype("float64")
    out["dist_cbd"] = dist
    out["log_dist_cbd"] = np.log10(dist.where(dist > 0))
    # Interaction: a large plot far from the CBD is a different product from a
    # large plot close in. Kept explicit so the linear model can use it.
    out["dist_cbd_x_area"] = out["log_dist_cbd"] * out["log_area"]
    return out


def add_target(frame: pd.DataFrame, price_col: str = "purchase_price") -> pd.DataFrame:
    out = frame.copy()
    price = pd.to_numeric(out[price_col], errors="coerce").astype("float64")
    out[price_col] = price
    out["log_price"] = np.log10(price.where(price > 0))
    return out


def build_features(
    clean: pd.DataFrame | None = None,
    macro: MacroPanel | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return the prepared modelling frame plus a report dict."""
    if clean is None:
        clean = load_clean()
    if macro is None:
        macro = load_macro_panel()

    frame = clean[[c for c in BASE_COLUMNS if c in clean.columns]].copy()
    frame["post_code"] = frame["post_code"].astype("string").str.strip().str.zfill(4)

    # --- macro as-of (the M3 fix) -----------------------------------------
    frame = pd.concat([frame.reset_index(drop=True),
                       macro.as_of(frame["contract_date"]).reset_index(drop=True)], axis=1)
    assert_macro_asof_valid(frame)

    # --- calendar / geometry / target -------------------------------------
    frame = add_calendar_features(frame)
    frame = add_geometry_features(frame)
    frame = add_target(frame)

    # --- postcode x month panel and its lagged rolling features ------------
    base_panel = build_postcode_month_panel(frame)
    spine = complete_monthly_spine(base_panel)
    rolling = add_rolling_features(spine)
    frame = attach_panel_features(frame, spine)
    frame = attach_panel_features(frame, rolling)

    frame["zoning_clean"] = clean_zoning(frame["zoning"])

    required = ["log_price", "area_sqm", "dist_cbd", "cash_rate_asof", "post_code"]
    before = len(frame)
    frame = frame.dropna(subset=[c for c in required if c in frame.columns]).reset_index(drop=True)
    dropped = before - len(frame)

    report = {
        "rows": int(len(frame)),
        "dropped_for_required_features": int(dropped),
        "macro": macro.meta(),
        "columns": list(frame.columns),
    }

    if verbose:
        print(f"Feature frame: {len(frame):,} rows ({dropped:,} dropped for required features)")
        print("Macro coverage by year:")
        print(macro_coverage_report(frame).to_string(index=False))

    if save:
        OUT_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(OUT_FEATURES_PARQUET, index=False)
        spine.to_parquet(OUT_PANEL_PARQUET, index=False)
        if verbose:
            print(f"Wrote {OUT_FEATURES_PARQUET} ({OUT_FEATURES_PARQUET.stat().st_size / 1024**2:,.1f} MiB)")
            print(f"Wrote {OUT_PANEL_PARQUET} ({OUT_PANEL_PARQUET.stat().st_size / 1024**2:,.1f} MiB)")
    return frame, report
