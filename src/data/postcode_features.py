"""Postcode-level reference features: development labels and transport distances.

Two leakage fixes relative to the group's notebook:

1. **Development labels are versioned.** The group labelled every postcode from
   the 2011-2014 vacancy baseline and applied it to contracts from 2001 onward.
   For a 2005 contract that label encodes information that did not exist yet.
   We therefore build two label sets and expose a single, legality-checked
   column (`development_type`) that is NULL where no legal label exists:
     - `development_type_v2010` : baseline 2001-2010 -> legal for contracts >= 2011
     - `development_type_v2014` : baseline 2011-2014 -> legal for contracts >= 2015
   Reporting/EDA may use either explicitly; the model uses `development_type`.

2. **Metro distance gets an as-of variant.** `stationentrances2020_v4.csv` is a
   2020 snapshot, so `dist_metro` describes a network that only opened on
   2019-05-26. It is kept for EDA, and `dist_metro_asof(contract_date)` is
   derived from the station opening dates in `data/external/events.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import (
    OUT_POSTCODE_DEVELOPMENT,
    OUT_POSTCODE_TRANSPORT,
    POSTCODE_CSV,
    STATION_CSV,
    CleaningPolicy,
)
from src.utils.io import iter_raw_chunks, read_postcode_key, write_json

R_EARTH_KM = 6371.0

METRO13 = [
    "Tallawong", "Rouse Hill", "Kellyville", "Bella Vista", "Norwest",
    "Hills Showground", "Castle Hill", "Cherrybrook", "Epping",
    "Macquarie University", "Macquarie Park", "North Ryde", "Chatswood",
]
METRO8 = METRO13[:8]
SYDNEY_CBD = np.array([[-33.8688, 151.2093]])


def nearest_km(lat: np.ndarray, lon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from each (lat, lon) to the nearest point."""
    if len(points) == 0:
        raise ValueError("No reference points available.")
    a = np.radians(np.asarray(lat, dtype=float))[:, None]
    b = np.radians(points[:, 0])[None, :]
    delta_lon = np.radians(np.asarray(lon, dtype=float))[:, None] - np.radians(points[:, 1])[None, :]
    cosine = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(delta_lon)
    return (R_EARTH_KM * np.arccos(np.clip(cosine, -1.0, 1.0))).min(axis=1)


# --------------------------------------------------------------------------- #
# Development labels
# --------------------------------------------------------------------------- #
def build_development_labels(
    raw_path: Path,
    policy: CleaningPolicy,
    postcode_development_path: Path = OUT_POSTCODE_DEVELOPMENT,
) -> pd.DataFrame:
    """Vacancy-share development labels for each configured baseline window.

    Follows the group's rule exactly (house-classified transactions priced above
    AUD 10,000 with purpose RESIDENCE or VACANT LAND; vacant share >= 15% and at
    least 50 baseline sales -> Greenfield, otherwise Established, else missing),
    but runs it once per window so each label has a known information cutoff.
    """
    usecols = ["contract_date", "purchase_price", "post_code", "primary_purpose", "property_type"]
    frames: list[pd.DataFrame] = []

    for window in policy.development_windows:
        parts = []
        for chunk in iter_raw_chunks(raw_path, policy.chunk_size, usecols=usecols):
            dates = pd.to_datetime(chunk["contract_date"].str.strip(), format="mixed", errors="coerce")
            prices = pd.to_numeric(chunk["purchase_price"], errors="coerce")
            purposes = chunk["primary_purpose"].str.strip().str.upper()
            keep = (
                dates.between(window.start, window.end)
                & prices.gt(10_000)
                & purposes.isin(["RESIDENCE", "VACANT LAND"])
                & chunk["property_type"].str.strip().str.lower().eq("house").fillna(False)
            )
            selected = pd.DataFrame({
                "postcode": read_postcode_key(chunk.loc[keep, "post_code"]),
                "is_vacant": purposes.loc[keep].eq("VACANT LAND").astype(int),
            }).dropna(subset=["postcode"])
            parts.append(selected.groupby("postcode").agg(
                vacant_sales=("is_vacant", "sum"), n_base=("is_vacant", "size")))

        agg = pd.concat(parts).groupby(level=0).sum() if parts else pd.DataFrame(
            columns=["vacant_sales", "n_base"])
        agg["base_vac"] = 100 * agg["vacant_sales"] / agg["n_base"]
        label = pd.Series("Established", index=agg.index, dtype="string")
        label.loc[agg["base_vac"].ge(policy.dev_vacant_share_threshold)] = "Greenfield"
        label.loc[agg["n_base"].lt(policy.dev_min_baseline_sales)] = pd.NA

        frames.append(pd.DataFrame({
            "postcode": agg.index,
            f"base_vac_{window.name}": agg["base_vac"].to_numpy(),
            f"n_base_{window.name}": agg["n_base"].to_numpy(),
            window.column(): label.to_numpy(),
        }))

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="postcode", how="outer")

    merged.to_csv(postcode_development_path, index=False)
    write_json(postcode_development_path.with_name("development_labels_meta.json"), {
        "windows": [
            {"name": w.name, "start": w.start, "end": w.end, "label_as_of": w.label_as_of,
             "column": w.column()}
            for w in policy.development_windows
        ],
        "min_baseline_sales": policy.dev_min_baseline_sales,
        "vacant_share_threshold": policy.dev_vacant_share_threshold,
        "rule": "vacant share >= threshold and n_base >= min -> Greenfield, else Established, else missing",
    })
    return merged


def development_label_meta(policy: CleaningPolicy) -> dict[str, str]:
    """Map each versioned label column to the last date at which it is illegal."""
    return {w.column(): w.label_as_of for w in policy.development_windows}


def attach_legal_development_type(
    df: pd.DataFrame,
    policy: CleaningPolicy,
    date_col: str = "contract_date",
) -> pd.DataFrame:
    """Add the single legality-checked `development_type` column.

    For each contract date, use the newest label whose window has closed before
    the contract. Contracts before the first window closes get NULL — that is the
    honest answer, not a back-filled label.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    legal = pd.Series(pd.NA, index=out.index, dtype="string")

    for window in sorted(policy.development_windows, key=lambda w: w.label_as_of):
        column = window.column()
        if column not in out.columns:
            continue
        # Strictly *after* the window closes. A contract inside the window (or in
        # the window's final year) would otherwise be labelled with statistics
        # computed from its own year — the boundary the contract test enforces.
        usable = dates.gt(pd.Timestamp(window.label_as_of))
        legal.loc[usable] = out.loc[usable, column]

    out["development_type"] = legal
    return out


# --------------------------------------------------------------------------- #
# Transport features
# --------------------------------------------------------------------------- #
def build_transport_features(
    station_path: Path = STATION_CSV,
    postcode_path: Path = POSTCODE_CSV,
    output_path: Path = OUT_POSTCODE_TRANSPORT,
) -> pd.DataFrame:
    """Postcode-level distances to CBD, all stations and the Metro stations.

    Unchanged from the group's logic: these are postcode-centroid great-circle
    distances, not property-level walking distances. Kept static for EDA.
    """
    entrances = pd.read_csv(station_path, low_memory=False)
    postcodes = pd.read_csv(postcode_path, low_memory=False)

    for column in ("LAT", "LONG"):
        entrances[column] = pd.to_numeric(entrances[column], errors="coerce")
    entrances = entrances.dropna(subset=["Train_Station", "LAT", "LONG"])
    stations = (entrances.groupby("Train_Station")
                .agg(lat=("LAT", "mean"), lon=("LONG", "mean")).reset_index())

    postcodes = postcodes.loc[
        (postcodes["state"] == "NSW")
        & (postcodes["type"] == "Delivery Area")
        & (postcodes["postcode"] >= 2000)
        & postcodes["lat"].notna() & postcodes["long"].notna()
        & postcodes["lat"].ne(0) & postcodes["long"].ne(0)
    ]
    centroids = (postcodes.groupby("postcode")
                 .agg(lat=("lat", "mean"), lon=("long", "mean"),
                      locality=("locality", "first"), sa4=("sa4name", "first"))
                 .reset_index())

    missing = sorted(set(METRO13) - set(stations["Train_Station"]))
    if missing:
        raise ValueError(f"Metro stations missing from source: {missing}")

    lat = centroids["lat"].to_numpy()
    lon = centroids["lon"].to_numpy()

    def points(names: list[str]) -> np.ndarray:
        return stations.loc[stations["Train_Station"].isin(names), ["lat", "lon"]].to_numpy()

    centroids["dist_train"] = nearest_km(lat, lon, stations[["lat", "lon"]].to_numpy())
    centroids["dist_metro"] = nearest_km(lat, lon, points(METRO13))
    centroids["dist_metro_new"] = nearest_km(lat, lon, points(METRO8))
    centroids["dist_cbd"] = nearest_km(lat, lon, SYDNEY_CBD)

    distance_columns = ["dist_cbd", "dist_train", "dist_metro", "dist_metro_new"]
    centroids[distance_columns] = centroids[distance_columns].round(2)

    features = centroids[["postcode", "locality", "sa4", *distance_columns]].copy()
    # `postcode` is an int key here (from the source), not the 4-char string.
    if features["postcode"].duplicated().any():
        raise ValueError("Transport postcodes must be unique.")
    if not np.isfinite(features[distance_columns]).all().all():
        raise ValueError("Non-finite transport distances.")

    features.to_csv(output_path, index=False)
    return features


def join_transport(
    chunk: pd.DataFrame,
    transport_features: pd.DataFrame,
    feature_columns: list[str],
    postcode_col: str = "post_code",
    transport_key: str = "_postcode_key",
) -> pd.DataFrame:
    """Left-join the postcode lookup onto a raw chunk (group's logic, kept as-is).

    Merging before the cleaning filters is what lets unmatched postcodes be
    removed by the required-field rule instead of silently surviving with NA.
    """
    properties = chunk.drop(columns=feature_columns, errors="ignore").copy()
    properties[transport_key] = read_postcode_key(properties[postcode_col])
    joined = properties.merge(
        transport_features,
        left_on=transport_key,
        right_on="postcode",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(joined) != len(chunk):
        raise AssertionError("Transport join changed the row count.")
    return joined.drop(columns=[transport_key, "postcode"])
