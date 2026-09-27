"""Property-level station distances from the geocoded address sample.

The postcode centroid distance used earlier is a crude proxy: a postcode can span
several kilometres, so "the centroid is 2 km from a station" says little about the
properties inside it. `data/geocode_results.csv` holds up to 10 geocoded addresses
per postcode, which is enough to describe the **distribution** of actual property
distance to a station instead of a single point.

Two things this module deliberately does:

1. **Distances are measured to station entrances, not station centroids.** The
   source file has 1,074 entrance coordinates; averaging them per station loses
   real geometry (two entrances of one station can be 200 m apart, which matters
   for a 500 m ring).
2. **Only stations open at the event date are used.** For E1 all 13 opened on
   2019-05-26, so this is exact; it also makes the module correct if a later event
   with staggered openings is added.

What this is *not* for: it must never become a model feature. It covers 6,347
addresses against 1.8M transactions (~0.35%), and distance is time-invariant, so
it adds nothing to price prediction. Its job is treatment definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.postcode_features import METRO13, SYDNEY_CBD, nearest_km
from src.utils.config import EXTERNAL_DIR, PROJECT_ROOT, STATION_CSV

GEOCODE_PATH = PROJECT_ROOT / "data" / "geocode_results.csv"
OUT_GEOCODE_DISTANCES = PROJECT_ROOT / "data" / "interim" / "postcode_geocode_distances.csv"

# Complement probabilities turned into distances.
SHARE_GRID = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
MIN_POINTS_PER_POSTCODE = 3   # below this the distribution is not credible


@dataclass
class GeocodeDistanceReport:
    postcodes_in: int
    postcodes_kept: int
    postcodes_dropped_few_points: int
    points_used: int
    entrances_used: int
    event_id: str

    def to_dict(self) -> dict:
        return {
            "postcodes_in": self.postcodes_in,
            "postcodes_kept": self.postcodes_kept,
            "postcodes_dropped_few_points": self.postcodes_dropped_few_points,
            "points_used": self.points_used,
            "entrances_used": self.entrances_used,
            "event_id": self.event_id,
        }


def load_station_entrances(
    station_path=STATION_CSV,
    station_names: list[str] | None = None,
) -> np.ndarray:
    """Entrance coordinates (lat, lon) for the named stations.

    Returns every entrance rather than one averaged point per station, so the
    geometry of large interchanges is preserved.
    """
    entrances = pd.read_csv(station_path, low_memory=False)
    for column in ("LAT", "LONG"):
        entrances[column] = pd.to_numeric(entrances[column], errors="coerce")
    entrances = entrances.dropna(subset=["Train_Station", "LAT", "LONG"])
    if station_names is not None:
        entrances = entrances.loc[entrances["Train_Station"].isin(station_names)]
    if entrances.empty:
        raise ValueError("No station entrances available for the requested stations.")
    return entrances[["LAT", "LONG"]].to_numpy()


def build_postcode_geocode_distances(
    geocode_path=GEOCODE_PATH,
    station_path=STATION_CSV,
    stations: list[str] | None = None,
    min_points: int = MIN_POINTS_PER_POSTCODE,
    output_path=OUT_GEOCODE_DISTANCES,
) -> tuple[pd.DataFrame, GeocodeDistanceReport]:
    """Postcode-level distribution of property-to-station-entrance distance.

    For each postcode with at least ``min_points`` geocoded addresses, computes:

    * ``n_geocoded`` — addresses available
    * ``dist_metro_p{p}`` for p in 10/25/50/75/90/100 — the p-th percentile of
      that postcode's property distances (``p=25`` is the headline: "at least
      three quarters of homes are within this far")
    * ``share_within_1km`` / ``_2km`` / ``_3km`` / ``_5km`` — the fraction of
      geocoded homes inside a walking-relevant radius, which serves as a
      continuous treatment intensity
    * ``dist_metro_share50`` — the distance within which half the homes sit
    """
    stations = stations or METRO13
    entrances = load_station_entrances(station_path, stations)
    points = pd.read_csv(geocode_path)
    points["postcode"] = (points["post_code"].astype("Int64").astype("string").str.zfill(4))

    counts = points.groupby("postcode").size()
    keep = counts[counts >= min_points].index
    dropped = int((counts < min_points).sum())
    points = points.loc[points["postcode"].isin(keep)].copy()
    lat = pd.to_numeric(points["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(points["lon"], errors="coerce").to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon)
    points, lat, lon = points.loc[valid], lat[valid], lon[valid]

    # Re-check the point count *after* dropping unusable coordinates, otherwise a
    # postcode can be reported with more geocodes than it actually contributes and
    # slip past the `min_geocoded_points` guard in the treatment assignment.
    usable = points.groupby("postcode").size()
    keep_usable = usable[usable >= min_points].index
    dropped += int((~counts.index.isin(keep_usable) & (counts >= min_points)).sum())
    points = points.loc[points["postcode"].isin(keep_usable)].copy()
    lat = pd.to_numeric(points["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(points["lon"], errors="coerce").to_numpy()

    # Distance from every geocoded address to the nearest station entrance.
    points["dist_km"] = nearest_km(lat, lon, entrances)

    grouped = points.groupby("postcode")["dist_km"]
    rows = []
    for postcode, series in grouped:
        values = np.sort(series.to_numpy())
        row = {"postcode": postcode, "n_geocoded": int(len(values))}
        for probability in SHARE_GRID:
            # Complement: the distance within which `probability` of homes sit.
            row[f"dist_metro_share{int(probability * 100)}"] = float(
                np.quantile(values, probability))
        for radius in (1.0, 2.0, 3.0, 5.0):
            row[f"share_within_{int(radius)}km"] = float((values <= radius).mean())
        row["dist_metro_mean"] = float(values.mean())
        row["dist_metro_spread_km"] = float(values.max() - values.min())
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("postcode").reset_index(drop=True)

    # The headline variable: "three quarters of homes are within this far".
    features["dist_metro_p25"] = features["dist_metro_share25"]
    features["dist_metro_p50"] = features["dist_metro_share50"]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)

    report = GeocodeDistanceReport(
        postcodes_in=int(len(counts)),
        postcodes_kept=int(len(features)),
        postcodes_dropped_few_points=dropped,
        points_used=int(len(points)),
        entrances_used=int(len(entrances)),
        event_id="",
    )
    return features, report


def compare_treatment_definitions(
    centroid: pd.DataFrame,
    geocode: pd.DataFrame,
    radius_km: float = 2.0,
    centroid_column: str = "dist_metro",
    geocode_column: str = "dist_metro_p25",
) -> pd.DataFrame:
    """Cross-tab of treated/control under the centroid and geocode definitions.

    This is the diagnostic that shows whether the crude centroid proxy was
    mis-assigning postcodes: a postcode whose centroid is far but whose homes are
    close (or vice versa) changes arm depending on the definition.
    """
    frame = centroid[["postcode", centroid_column, "sa4"]].merge(
        geocode[["postcode", geocode_column, "n_geocoded"]], on="postcode", how="inner")
    frame["centroid_treated"] = frame[centroid_column].le(radius_km)
    frame["geocode_treated"] = frame[geocode_column].le(radius_km)
    frame["agree"] = frame["centroid_treated"] == frame["geocode_treated"]
    return frame.sort_values([centroid_column]).reset_index(drop=True)
