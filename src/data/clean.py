"""Leakage-safe re-implementation of the group's cleaning pipeline.

Kept from the merged notebook (see initial_v0/03_leakage_audit.md §6.1):
  * chunked reading with an explicit dtype map
  * left-joining the postcode lookups *before* filtering
  * the required-field / purpose / house filters and their staged counts
  * the `df_before_iqr` snapshot so the outlier step is re-runnable
  * area-unit conversion (M -> sqm, H -> hectare * 10 000)
  * de-duplication and the post-cleaning assertions

Changed (audit items):
  * **M3** annual `cash_rate` / `cpi` are NOT attached here; they arrive later as
    monthly as-of columns (`src/data/macro_asof.py`). This module does not touch
    anything that was unknown at the contract date.
  * **M1 / L4** the single full-sample log-IQR fence is replaced by per-year
    bounds fitted on the training span only (`src/data/outliers.py`).
  * **M2** a `group_key` column is produced for CV grouping: the address when it
    carries a house number, `property_id` when it does not.
  * **§2.2** development labels are attached through `attach_legal_development_type`,
    so a contract never carries a label built from its own or future years.
  * **D1/D2** de-duplication uses an explicit identity key
    (`property_id + contract_date + purchase_price + area_sqm`) instead of
    comparing whole rows, so it stays correct after later stages drop columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.postcode_features import (
    attach_legal_development_type,
    join_transport,
)
from src.utils.config import OUT_POSTCODE_TRANSPORT, SETTINGS, CleaningPolicy
from src.utils.io import iter_raw_chunks, read_postcode_key

# A leading house number, optionally after a unit/lot prefix:
#   "12 SMITH ST", "UNIT 5/12 SMITH ST", "5/12 SMITH ST", "LOT 3 SMITH RD".
# Deliberately NOT matching a bare street name ("SMITH ST"), which is the case
# `add_group_key` has to fall back on.
_HOUSE_NUMBER_RE = re.compile(
    r"^\s*(?:(?:UNIT|U|SHOP|SUITE|LOT|FLAT|LEVEL|APT|APARTMENT)\s*)?\d", re.IGNORECASE)


def has_house_number(address: pd.Series) -> pd.Series:
    """True where an address starts with a house/unit/lot number."""
    return address.astype("string").str.match(_HOUSE_NUMBER_RE, na=False).fillna(False)


# Identity key for de-duplication: one sale event.
DUPLICATE_KEY: tuple[str, ...] = (
    "property_id", "contract_date", "purchase_price", "area",
)

STAGE_LABELS = (
    "Original rows",
    "Dropped outside plausible date window",
    "After required-field filter",
    "After purpose filter",
    "After house filter",
)


@dataclass
class CleaningReport:
    """Staged row counts plus the field-level missing profile (both are logs)."""

    stage_counts: pd.Series = field(default_factory=lambda: pd.Series(0, index=list(STAGE_LABELS), dtype="int64"))
    missing_counts: pd.Series = field(default_factory=pd.Series)
    duplicate_rows_removed: int = 0
    invalid_value_rows_removed: int = 0
    negative_distance_rows: int = 0
    plausibility: dict = field(default_factory=dict)

    def add(self, other: "CleaningReport") -> None:
        self.stage_counts = self.stage_counts.add(other.stage_counts, fill_value=0)
        self.missing_counts = self.missing_counts.add(other.missing_counts, fill_value=0)
        self.duplicate_rows_removed += other.duplicate_rows_removed
        self.invalid_value_rows_removed += other.invalid_value_rows_removed
        self.negative_distance_rows += other.negative_distance_rows

    def summary(self) -> str:
        lines = []
        previous = None
        for stage, count in self.stage_counts.items():
            removed = "" if previous is None else f"  | removed {previous - count:,}"
            lines.append(f"{stage}: {count:,}{removed}")
            previous = count
        lines.append(f"Duplicate rows removed: {self.duplicate_rows_removed:,}")
        lines.append(
            f"Invalid price/area or negative-distance rows removed: {self.invalid_value_rows_removed:,}"
            f" (of which negative distances: {self.negative_distance_rows:,})"
        )
        if self.plausibility:
            lines.append(
                f"Plausibility gate removed: {self.plausibility.get('rows_removed', 0):,}"
                f" of {self.plausibility.get('rows_before', 0):,} rows"
                f" ({self.plausibility.get('rows_removed', 0) / max(self.plausibility.get('rows_before', 1), 1):.4%})"
            )
            for key, value in self.plausibility.items():
                if key.startswith("rule_"):
                    lines.append(f"    {key[5:]}: {value:,}")
        lines.append("Missing values before filtering (counts overlap):")
        for column, count in self.missing_counts.items():
            lines.append(f"  {column}: {count:,}")
        return "\n".join(lines)


def parse_contract_dates(raw: pd.Series) -> pd.Series:
    """Parse the three date shapes present in the source file.

    ISO (2014-01-31), compact (20140131) and anything else day-first. Same
    strategy as the group's notebook; only wrapped in a named function.
    """
    text = raw.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    parsed = parsed.fillna(compact)
    remaining = parsed.isna() & text.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text.loc[remaining], format="mixed", dayfirst=True, errors="coerce"
        )
    return parsed


def clean_chunk(
    chunk: pd.DataFrame,
    policy: CleaningPolicy,
    transport_features: pd.DataFrame,
    transport_feature_columns: list[str],
    development_lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the group's filter chain to one raw chunk.

    Returns the retained rows plus a per-chunk report. No fitted statistics are
    computed here, so this function is safe to run over the full history.
    """
    joined = join_transport(chunk, transport_features, transport_feature_columns)

    for column in ("purchase_price", "area"):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")

    joined["contract_date"] = parse_contract_dates(joined["contract_date"])
    joined["post_code"] = joined["post_code"].str.strip().replace("", pd.NA)
    joined["area_type"] = joined["area_type"].str.strip().replace("", pd.NA)

    # --- versioned development labels (no annual macro columns here) --------
    if development_lookup is not None:
        keys = read_postcode_key(joined["post_code"])
        for column in development_lookup.columns:
            if column.startswith("development_type_"):
                # Map by postcode value. A positional `.map(series)` would align
                # on the lookup's index and silently produce all-NA.
                mapping = dict(zip(development_lookup["postcode"].astype("string"),
                                   development_lookup[column]))
                joined[column] = keys.map(mapping).astype("string")
    joined = attach_legal_development_type(joined, policy)

    # --- plausibility window (drops corrupted dates such as 0015-08-29) -----
    in_window = joined["contract_date"].between(
        pd.Timestamp(policy.contract_date_min), pd.Timestamp(policy.contract_date_max))
    outside_window = int((~in_window).sum())
    joined = joined.loc[in_window]
    if len(joined) == 0:
        raise ValueError("Every row fell outside the plausible contract-date window.")

    # --- staged filters ----------------------------------------------------
    required = [c for c in policy.required_fields]
    if policy.required_development_column:
        required.append(policy.required_development_column)
    required = [c for c in required if c in joined.columns]

    missing = joined[required].isna().sum()
    complete = joined.dropna(subset=required)
    purposes = complete["primary_purpose"].str.strip().str.upper()
    selected_purpose = complete.loc[purposes.eq(policy.purpose_keep).fillna(False)]
    houses = selected_purpose.loc[
        selected_purpose["property_type"].str.strip().str.lower().eq(policy.property_type_keep).fillna(False)
    ]

    report = CleaningReport(stage_counts=pd.Series({
        "Original rows": len(chunk),
        "Dropped outside plausible date window": outside_window,
        "After required-field filter": len(complete),
        "After purpose filter": len(selected_purpose),
        "After house filter": len(houses),
    }), missing_counts=missing)

    return houses.copy(), report


def convert_area_units(df: pd.DataFrame, policy: CleaningPolicy) -> pd.DataFrame:
    """M -> sqm, H -> hectare * 10 000. Unknown codes stay missing."""
    out = df.copy()
    codes = out["area_type"].astype("string").str.strip().str.upper()
    factors = codes.map(dict(policy.area_unit_factors))
    out["area_sqm"] = out["area"] * factors
    out["_area_unit"] = codes
    return out


def add_group_key(
    df: pd.DataFrame,
    postcode_col: str = "post_code",
    verbose: bool = True,
) -> pd.DataFrame:
    """A per-dwelling key for CV grouping (audit M2). Never a feature.

    The raw ``address`` field carries the unit number ("910 B/6 NANCARROW AVE"),
    so ``address|postcode`` is the closest thing to "one dwelling" available.
    But 1.13% of addresses carry **no house number** at all -- they are bare
    street names such as ``' WOODBURY PARK DR, MARDI'``, where up to 62
    different houses in different price bands collapse into a single key. Those
    rows get grouped together for purging and are then almost never eligible for
    a validation fold.

    For those rows the key falls back to ``property_id``, which splits them back
    into (near-)individual dwellings: the largest remaining group falls from 62
    rows to 25, and the number of groups of 20+ rows drops from 1,124 to 25.

    ``property_id`` is deliberately *not* used as the primary key. It is a
    parcel/estate identifier, not a dwelling identifier: one Norwest
    ``property_id`` covers 42 addresses across two streets in a 2020-21
    development, and of the 39,529 property_ids spanning several address
    spellings, only 663 have the same contract date under two spellings --
    i.e. the overwhelming majority are genuinely different houses, not one
    dwelling written two ways. Grouping on it would merge whole estates into one
    CV group and shrink the effective validation set.
    """
    out = df.copy()
    address = out["address"].astype("string").str.strip().str.upper()
    postcode = read_postcode_key(out[postcode_col])
    address_key = address.fillna("<missing>") + "|" + postcode.fillna("<missing>")

    if "property_id" not in out.columns:
        out["group_key"] = address_key
        out["group_key_source"] = "address"
        return out

    numbered = has_house_number(address).to_numpy(dtype=bool)
    pid = out["property_id"].astype("string").str.strip()
    # `.notna()` / `.ne("")` on an Arrow-backed string column yield nullable
    # booleans; combining them with `&` raises "boolean value of NA is
    # ambiguous". Resolve to plain bool first.
    pid_present = pid.notna().fillna(False).to_numpy(dtype=bool)
    pid_nonempty = pid.fillna("").ne("").to_numpy(dtype=bool)
    fallback = (~numbered) & pid_present & pid_nonempty
    if verbose:
        print(f"group_key from address: {int(numbered.sum()):,} rows | "
              f"from property_id fallback (no house number): {int(fallback.sum()):,} rows")
    out["group_key"] = address_key.astype("object")
    out.loc[fallback, "group_key"] = pid.loc[fallback]
    # Keep the provenance: it is what makes the fallback auditable downstream.
    out["group_key_source"] = "address"
    out.loc[fallback, "group_key_source"] = "property_id"
    out["group_key"] = out["group_key"].astype("string")
    out["group_key_source"] = out["group_key_source"].astype("string")
    return out


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Drop rows that are the same *record*, not merely the same values.

    A whole-row ``drop_duplicates`` is the wrong instrument here, because later
    stages drop columns: rows that differ only in ``settlement_date``,
    ``legal_description`` or ``strata_lot_number`` are distinct rows at this
    point but become byte-identical in the modelling frame, where they
    double-weight one transaction. Deduplicating on an explicit identity key is
    stable under any later column selection.

    ``subset`` is that key. It must uniquely describe one sale event:
    ``property_id + contract_date + purchase_price + area_sqm``. Measured on the
    full history this removes ~30,300 rows, and **no** removed row sits in a
    group with more than one distinct price -- so no genuine concurrent sale is
    being destroyed (verified, and pinned by
    ``tests/test_dedup_and_grouping.py``). Adding ``area_sqm`` to the key keeps
    the six same-price cases whose area differs.

    Falls back to whole-row comparison when the key columns are absent (e.g. on
    a frame that has already been trimmed down).
    """
    before = len(df)
    if subset:
        key = [c for c in subset if c in df.columns]
    else:
        key = []
    if key and len(key) == len(subset):
        out = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    else:
        out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, before - len(out)


def drop_invalid_values(
    df: pd.DataFrame,
    distance_columns: tuple[str, ...] = ("dist_cbd", "dist_train", "dist_metro", "dist_metro_new"),
) -> tuple[pd.DataFrame, int, int]:
    """price > 0 and area_sqm > 0, and no negative distances. Zero is allowed."""
    present = [c for c in distance_columns if c in df.columns]
    negative = df[present].lt(0).any(axis=1)
    valid = df["purchase_price"].gt(0) & df["area_sqm"].gt(0) & np.isfinite(df["purchase_price"]) & np.isfinite(df["area_sqm"])
    keep = valid & ~negative
    out = df.loc[keep].copy().reset_index(drop=True)
    return out, int((~keep).sum()), int(negative.sum())


def apply_plausibility_gate(
    df: pd.DataFrame,
    policy: "PlausibilityPolicy | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop records that cannot be residence-house transactions (layer A).

    Runs **before** any statistic is fitted, and before the IQR fence, so corrupt
    values cannot widen the quartiles used to judge them. Each rule is recorded
    separately so the counts can go into the report.

    Distinct from the IQR fence on purpose: this gate uses fixed domain constants
    (see `PlausibilityPolicy`), it never depends on the sample, and it is the only
    check that can catch a value that is impossible in absolute terms — for
    example a 1.53e9 sqm house, which is neither negative nor zero and sits in a
    year that no training-span fence covers.
    """
    from src.utils.config import SETTINGS

    policy = policy or SETTINGS.plausibility
    out = df
    report: dict = {"enabled": bool(policy.enabled)}

    if not policy.enabled:
        report["rows_removed"] = 0
        report["rows_remaining"] = len(df)
        return out, report

    area = pd.to_numeric(out["area_sqm"], errors="coerce")
    price = pd.to_numeric(out["purchase_price"], errors="coerce")
    unit = price / area.replace(0, np.nan)

    rules = {
        "area_sqm_below_min": area.lt(policy.area_sqm_min),
        "area_sqm_above_max": area.gt(policy.area_sqm_max),
        "price_below_min": price.lt(policy.price_min),
        "price_above_max": price.gt(policy.price_max),
        "unit_price_below_min": unit.lt(policy.unit_price_min),
        "unit_price_above_max": unit.gt(policy.unit_price_max),
    }
    violated = pd.Series(False, index=out.index)
    for name, mask in rules.items():
        mask = mask.fillna(False)
        report[f"rule_{name}"] = int(mask.sum())
        violated |= mask

    out = out.loc[~violated].reset_index(drop=True)
    report.update({
        "rows_before": int(len(df)),
        "rows_removed": int(violated.sum()),
        "rows_remaining": int(len(out)),
        "bounds": {
            "area_sqm": [policy.area_sqm_min, policy.area_sqm_max],
            "purchase_price": [policy.price_min, policy.price_max],
            "price_per_sqm": [policy.unit_price_min, policy.unit_price_max],
        },
    })
    return out, report


def build_clean(
    policy: CleaningPolicy | None = None,
    transport_features: pd.DataFrame | None = None,
    development_lookup: pd.DataFrame | None = None,
    progress: bool = True,
    max_chunks: int | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Full pass over the raw CSV -> the pre-outlier clean frame.

    The returned frame still contains the IQR-trimmable rows; outlier removal is
    a separate, train-only step (`src/data/outliers.py`). `max_chunks` supports
    smoke tests without reading the whole 610 MiB file.
    """
    policy = policy or SETTINGS.cleaning
    from src.utils.config import RAW_PROPERTY_CSV

    if transport_features is None:
        transport_features = pd.read_csv(OUT_POSTCODE_TRANSPORT, dtype={"postcode": "string"})
    transport_features = transport_features.replace(r"^\s*$", pd.NA, regex=True)
    transport_features["postcode"] = transport_features["postcode"].astype("string").str.strip().str.zfill(4)
    feature_columns = [c for c in transport_features.columns if c != "postcode"]

    parts: list[pd.DataFrame] = []
    report = CleaningReport()
    for index, chunk in enumerate(iter_raw_chunks(RAW_PROPERTY_CSV, policy.chunk_size)):
        if max_chunks is not None and index >= max_chunks:
            break
        cleaned, chunk_report = clean_chunk(
            chunk, policy, transport_features, feature_columns, development_lookup
        )
        parts.append(cleaned)
        report.add(chunk_report)
        if progress:
            print(f"  chunk {index:>3}: kept {len(cleaned):,} of {len(chunk):,}", flush=True)

    if not parts:
        raise ValueError("No chunks were read - check RAW_PROPERTY_CSV.")

    df = pd.concat(parts, ignore_index=True)
    del parts

    df, report.duplicate_rows_removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    # Authoritative de-duplication assertion. It has to happen HERE: `area` is
    # dropped at the end of this function, so `check_clean` on the saved frame
    # cannot re-derive the full identity key.
    leftover = int(df.duplicated(subset=list(DUPLICATE_KEY)).sum())
    if leftover:
        raise AssertionError(
            f"{leftover:,} duplicate sale events survived de-duplication "
            f"on {list(DUPLICATE_KEY)}")
    df = convert_area_units(df, policy)
    # Invalid price/area and negative distances are rejected before the unit
    # column is dropped, so the counts stay auditable.
    df, removed, negative = drop_invalid_values(df)
    report.invalid_value_rows_removed = removed
    report.negative_distance_rows = negative
    # Layer A: absolute plausibility gate, before any statistic is fitted.
    df, plausibility = apply_plausibility_gate(df)
    report.plausibility = plausibility
    df = add_group_key(df)
    df = df.drop(columns=["area", "area_type", "_area_unit"], errors="ignore")
    return df, report
