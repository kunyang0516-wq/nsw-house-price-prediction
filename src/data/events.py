"""Event definitions, treatment assignment and the event-time panel (design D3/D4).

E1 (the v0 study) is the opening of Sydney Metro Northwest on **2019-05-26**,
when all 13 Tallawong-Chatswood stations opened simultaneously. Two design
choices follow from that:

* **Single event date, single cohort.** Staggered-adoption estimators
  (Callaway-Sant'Anna, Sun-Abraham) exist to handle cohorts treated at different
  times. With one date they collapse to the two-way fixed-effects event study, so
  the simple estimator is not an approximation here — it is exact.
* **Control group = not-yet-treated.** Postcodes in the same SA4 regions that are
  far from any new station. We exclude the never-served framing because every
  inner-Sydney postcode is eventually served; "not yet treated" is the honest
  description of the counterfactual inside the window.

Treatment is defined by the **pre-opening** metro distance snapshot (D4): a
postcode is treated if its centroid is within ``radius_km`` of a station that is
open at the event date. Using the 2020 station snapshot unconditionally would
"time travel" — a 2016 contract cannot be near a station that did not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import EXTERNAL_DIR, INTERIM_DIR

EVENTS_CSV = EXTERNAL_DIR / "events.csv"
NW_SA4 = (
    "Sydney - Ryde",
    "Sydney - Parramatta",
    "Sydney - Baulkham Hills and Hawkesbury",
    "Sydney - Blacktown",
)


@dataclass(frozen=True)
class EventStudySpec:
    """Everything the event study needs, in one auditable place (pre-registration)."""

    event_id: str = "E1"
    radius_km: float = 2.0            # main specification (D4)
    control_min_km: float = 3.0       # controls must be at least this far away
    window_months: int = 24           # [-24, +24]
    reference_period: int = -1        # omitted event-time bucket
    cluster: str = "postcode"
    outcome: str = "log_median_unit_price"   # main outcome (D3: single, pre-declared)
    # Treatment definition. `dist_metro_p25` comes from the geocoded address
    # sample: "at least 75% of geocoded homes in this postcode are within this
    # far". It is preferred over the postcode centroid because a postcode can
    # span kilometres, so its centroid mis-assigns the arm for many homes.
    # Fall back to `dist_metro` (centroid, 2020 snapshot) when no geocoding exists.
    treated_variable: str = "dist_metro_p25"
    fallback_variable: str = "dist_metro"
    min_geocoded_points: int = 3
    robustness_radii: tuple[float, ...] = (1.0, 3.0)
    robustness_outcomes: tuple[str, ...] = ("log_median_price", "n_sales")
    sa4_filter: tuple[str, ...] = NW_SA4

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "radius_km": self.radius_km,
            "control_min_km": self.control_min_km,
            "window_months": self.window_months,
            "reference_period": self.reference_period,
            "cluster": self.cluster,
            "outcome": self.outcome,
            "treated_variable": self.treated_variable,
            "fallback_variable": self.fallback_variable,
            "min_geocoded_points": self.min_geocoded_points,
            "robustness_radii": list(self.robustness_radii),
            "robustness_outcomes": list(self.robustness_outcomes),
            "sa4_filter": list(self.sa4_filter),
        }


def load_events(path=EVENTS_CSV) -> pd.DataFrame:
    events = pd.read_csv(path)
    events["event_date"] = pd.to_datetime(events["event_date"])
    return events


def get_event(event_id: str = "E1", path=EVENTS_CSV) -> pd.Series:
    events = load_events(path)
    match = events.loc[events["event_id"] == event_id]
    if match.empty:
        raise KeyError(f"event_id {event_id!r} not found in {path}")
    return match.iloc[0]


def event_month(event_id: str = "E1", path=EVENTS_CSV) -> pd.Timestamp:
    """Month bucket (first of month) that contains the event date."""
    return get_event(event_id, path)["event_date"].to_period("M").to_timestamp()


def assign_treatment(
    transport_features: pd.DataFrame,
    spec: EventStudySpec,
    path=EVENTS_CSV,
    geocode_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Postcode-level treatment table from the pre-opening distance snapshot.

    Distance is taken from the **geocoded address sample** when available
    (``dist_metro_p25``: at least three quarters of geocoded homes in the postcode
    are within this many km of a station entrance), falling back to the postcode
    centroid (``dist_metro``) for postcodes without enough geocoded addresses.

    Returns ``postcode, dist_metro_pre, treated, sa4, locality`` plus the event
    month, so downstream code never re-derives the definition.
    """
    event = get_event(spec.event_id, path)
    frame = transport_features.copy()
    frame["postcode"] = frame["postcode"].astype("string").str.strip().str.zfill(4)

    if geocode_features is None:
        try:
            from src.data.geocode_features import build_postcode_geocode_distances
            geocode_features, _ = build_postcode_geocode_distances(
                stations=_stations_at_event(spec.event_id, path))
        except FileNotFoundError:
            geocode_features = None

    centroid = pd.to_numeric(frame.get("dist_metro"), errors="coerce")
    if geocode_features is not None and not geocode_features.empty:
        merge_columns = ["postcode", spec.treated_variable]
        for extra in ("n_geocoded", "share_within_1km", "share_within_2km",
                      "dist_metro_p50", "dist_metro_spread_km"):
            if extra in geocode_features.columns:
                merge_columns.append(extra)
        merged = frame.merge(geocode_features[merge_columns].rename(
            columns={"postcode": "_geo_key"}), left_on="postcode",
            right_on="_geo_key", how="left").drop(columns=["_geo_key"])
        geocoded = pd.to_numeric(merged.get(spec.treated_variable), errors="coerce")
        enough = geocoded.notna()
        if "min_geocoded_points" in spec.__dataclass_fields__ and "n_geocoded" in merged:
            enough &= pd.to_numeric(merged["n_geocoded"], errors="coerce").ge(spec.min_geocoded_points)
        frame["dist_metro_geocoded"] = geocoded
        frame["dist_metro_pre"] = geocoded.where(enough, centroid)
        frame["treatment_distance_source"] = np.where(enough, "geocoded_p25", "centroid_fallback")
    else:
        frame["dist_metro_geocoded"] = np.nan
        frame["dist_metro_pre"] = centroid
        frame["treatment_distance_source"] = "centroid"

    frame["dist_metro_centroid"] = centroid
    frame["treated"] = frame["dist_metro_pre"].le(spec.radius_km)

    # Controls: same SA4 universe, far enough away that the station is not
    # plausibly part of their local market.
    in_scope = frame["sa4"].isin(spec.sa4_filter)
    frame["in_scope_sa4"] = in_scope
    frame["is_control"] = in_scope & frame["dist_metro_pre"].gt(spec.control_min_km)
    frame["is_treated"] = in_scope & frame["treated"]

    frame["event_id"] = spec.event_id
    frame["event_month"] = event_month(spec.event_id, path)
    frame["event_date"] = event["event_date"]
    return frame


def _stations_at_event(event_id: str, path=EVENTS_CSV) -> list[str]:
    """Stations already open at the event date (all 13 for E1)."""
    event = get_event(event_id, path)
    names = [name.strip() for name in str(event["opened_stations"]).split("|") if name.strip()]
    return names or list(METRO13)


def treatment_summary(treatment: pd.DataFrame, radius_km: float = 2.0) -> dict:
    """Counts per arm, plus how often the two distance definitions disagree."""
    treated = treatment.loc[treatment["is_treated"]]
    control = treatment.loc[treatment["is_control"]]
    sources = treatment["treatment_distance_source"].value_counts().to_dict() \
        if "treatment_distance_source" in treatment.columns else {}
    summary = {
        "treated_postcodes": int(treated["postcode"].nunique()),
        "treated_list": sorted(treated["postcode"].unique().tolist()),
        "control_postcodes": int(control["postcode"].nunique()),
        "treated_mean_dist_km": float(treated["dist_metro_pre"].mean()) if len(treated) else None,
        "control_mean_dist_km": float(control["dist_metro_pre"].mean()) if len(control) else None,
        "distance_source_counts": sources,
    }
    if "dist_metro_centroid" in treatment.columns:
        both = (treatment[["postcode", "dist_metro_centroid", "dist_metro_pre"]]
                .drop_duplicates("postcode")
                .dropna(subset=["dist_metro_centroid", "dist_metro_pre"]))
        if len(both):
            disagree = ((both["dist_metro_centroid"] <= radius_km)
                        != (both["dist_metro_pre"] <= radius_km)).sum()
            summary["postcodes_compared"] = int(len(both))
            summary["postcodes_disagreeing_with_centroid"] = int(disagree)
    return summary


def build_event_panel(
    transactions: pd.DataFrame,
    treatment: pd.DataFrame,
    spec: EventStudySpec,
    date_col: str = "contract_date",
    postcode_col: str = "post_code",
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
) -> pd.DataFrame:
    """`postcode x month` panel restricted to treated + control postcodes.

    Columns: ``postcode, month, event_time, treated, group, n_sales,
    median_price, median_unit_price`` (+ ``log_*`` transforms and ``n_sales``
    used as the count outcome).
    """
    frame = pd.DataFrame({
        "postcode": transactions[postcode_col].astype("string").str.strip().str.zfill(4),
        "month": pd.to_datetime(transactions[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
        "price": pd.to_numeric(transactions[value_col], errors="coerce"),
        "area": pd.to_numeric(transactions[area_col], errors="coerce"),
    }).dropna(subset=["postcode", "month", "price"])

    keep = treatment.loc[treatment["is_treated"] | treatment["is_control"], ["postcode", "is_treated"]]
    frame = frame.merge(keep, on="postcode", how="inner", validate="many_to_one")
    if frame.empty:
        raise ValueError("No transactions fall in the treated/control postcode set.")

    frame["unit_price"] = frame["price"] / frame["area"].replace(0, np.nan)
    panel = (frame.groupby(["postcode", "month", "is_treated"], observed=True)
             .agg(n_sales=("price", "size"),
                  median_price=("price", "median"),
                  median_unit_price=("unit_price", "median"),
                  median_area=("area", "median"))
             .reset_index()
             .rename(columns={"is_treated": "treated"}))

    event_start = event_month(spec.event_id)
    panel["event_time"] = ((panel["month"].dt.year - event_start.year) * 12
                           + (panel["month"].dt.month - event_start.month))
    window = panel["event_time"].between(-spec.window_months, spec.window_months)
    panel = panel.loc[window].copy()

    panel["treated"] = panel["treated"].astype(int)
    panel["group"] = np.where(panel["treated"] == 1, "Station", "No station")
    panel["log_median_price"] = np.log10(panel["median_price"].where(panel["median_price"] > 0))
    panel["log_median_unit_price"] = np.log10(
        panel["median_unit_price"].where(panel["median_unit_price"] > 0))
    panel["month_index"] = ((panel["month"].dt.year - panel["month"].dt.year.min()) * 12
                            + (panel["month"].dt.month - panel["month"].dt.month.min()) + 1)
    return panel.sort_values(["postcode", "month"]).reset_index(drop=True)


def write_spec(spec: EventStudySpec, path=None) -> None:
    """Persist the pre-registration so the reported design can be checked."""
    path = path or (INTERIM_DIR / f"event_study_spec_{spec.event_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
