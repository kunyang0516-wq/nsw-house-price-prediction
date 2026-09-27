"""Single source of truth for paths, sample windows and cleaning policy.

Deliberately plain Python (no YAML dependency) so the pipeline runs in the
project's existing venv. Import ``SETTINGS`` everywhere instead of hard-coding
paths or constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw inputs currently live in the repository root (the EDA notebook reads them
# from there). They are treated as READ-ONLY.
RAW_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_PROPERTY_CSV = RAW_DIR / "nsw_property_data.csv"
POSTCODE_CSV = RAW_DIR / "australian_postcodes.csv"
STATION_CSV = RAW_DIR / "stationentrances2020_v4.csv"

# Outputs
OUT_POSTCODE_DEVELOPMENT = INTERIM_DIR / "postcode_development.csv"
OUT_POSTCODE_TRANSPORT = INTERIM_DIR / "postcode_transport_features.csv"
OUT_CLEAN_PARQUET = PROCESSED_DIR / "clean.parquet"
OUT_DEV_LABELS_META = INTERIM_DIR / "development_labels_meta.json"


# --------------------------------------------------------------------------- #
# Cleaning policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DevelopmentWindow:
    """A vacancy-share baseline window used to label a postcode.

    ``label_as_of`` is the information cutoff that makes the label legal for a
    contract: a label built from [start, end] may only be used for contracts
    after ``label_as_of``.
    """

    name: str
    start: str
    end: str
    label_as_of: str

    def column(self) -> str:
        return f"development_type_{self.name}"


@dataclass(frozen=True)
class PlausibilityPolicy:
    """Absolute bounds separating "physically impossible" from "statistically unusual".

    These are **domain constants, not sample quantiles**, and that distinction is
    the whole point:

    * an IQR fence is *relative* — it widens with a dirtier sample, moves when the
      window changes, and cannot be written into a methods section;
    * these bounds can be stated before looking at the data ("a NSW residence
      house is not 15 hectares"), so a reader can judge them.

    They also must run **before** any statistic is fitted, otherwise the corrupt
    values distort the very quartiles used to judge them.

    Observed pathology this catches: a 2020 "house" with ``area_sqm = 1.53e9``
    and a "residence" priced at ``$875,300,000``.

    The unit-price bounds matter most: they are the only check that catches
    *joint* errors, where price and area each look acceptable but their ratio does
    not. The lower bound is deliberately loose ($10/sqm) because genuine
    large-lot rural sales sit near $40-50/sqm; those are real, not corrupt.
    """

    area_sqm_min: float = 20.0
    area_sqm_max: float = 10_000.0        # ~1 hectare; above this it is a farm/acreage
    price_min: float = 10_000.0
    price_max: float = 50_000_000.0       # the NSW record is ~$130M; $50M+ needs review
    unit_price_min: float = 10.0          # below this the record is probably land, not a house
    unit_price_max: float = 50_000.0      # above this no Australian residence exists
    enabled: bool = True


@dataclass(frozen=True)
class CleaningPolicy:
    """Mirrors the group's seven cleaning steps, with leakage fixes applied."""

    chunk_size: int = 250_000

    # Step 1: required fields at the initial filter (group's list, minus the
    # annual economic columns, which are re-introduced later as monthly as-of).
    required_fields: tuple[str, ...] = (
        "purchase_price",
        "area",
        "area_type",
        "contract_date",
        "post_code",
        "dist_cbd",
    )
    # Deliberately NOT required: the development label. A legally-timed label
    # only exists from 2011 onward (see development_windows), so demanding it
    # would throw away the 2001-2010 training span. Missing labels are kept as
    # NaN and handled at the feature level.
    required_development_column: str | None = None
    required_distance_columns: tuple[str, ...] = ("dist_cbd",)

    # Plausibility window for contract dates. The raw file contains corrupted
    # values outside it (observed: 0015-08-29), which are dropped as data errors.
    contract_date_min: str = "2001-01-01"
    contract_date_max: str = "2023-12-31"

    # Plausible recorded area for a *residence house* in square metres.
    # Measured on the cleaned sample: 4.8% of rows exceed 20,000 sqm (a round
    # value that looks like a sentinel), 6.2% exceed 10,000, and the maximum is
    # 2.7e9 — clearly corrupted. The unit-price/area relationship also has a
    # visible break above roughly 2,000-3,000 sqm, which is the standard-lot /
    # acreage boundary. Analyses restricted to standard residential lots use this
    # cap; the unrestricted sample is reported as a robustness check.
    area_plausible_min: float = 100.0
    area_plausible_max: float = 5_000.0

    # Step 2: purpose / type filters (group's rule)
    purpose_keep: str = "RESIDENCE"
    property_type_keep: str = "house"

    # Step 3: label windows.
    #   v2010 -> legal for every contract after 2010-12-31 (the leakage-safe one)
    #   v2014 -> the group's original window; retro-informs pre-2011 contracts
    development_windows: tuple[DevelopmentWindow, ...] = (
        DevelopmentWindow("v2010", "2001-01-01", "2010-12-31", "2010-12-31"),
        DevelopmentWindow("v2014", "2011-01-01", "2014-12-31", "2014-12-31"),
    )
    dev_min_baseline_sales: int = 50
    dev_vacant_share_threshold: float = 15.0

    # Step 4: unit conversion
    area_unit_factors: tuple[tuple[str, float], ...] = (("M", 1.0), ("H", 10_000.0))

    # Step 5: log-IQR outlier policy.
    # ``by=("year",)`` + a full-sample fit is the fix for the group's single
    # full-sample fence. Two things matter: the fence is *per year* (a 2001-2015
    # ceiling must not judge a 2023 sale) and it is applied to the training side
    # only (applying it to validation would empty every post-2015 window).
    # ``mode`` may be overridden to "off" for the robustness run requested in the
    # audit (robust loss instead of trimming).
    iqr_mode: str = "per_year_full_sample"  # {per_year_full_sample, per_year_train_only, global_train_only, off}
    iqr_multiplier: float = 1.5

    # Columns that must never reach the modelling matrix (audit section 1)
    forbidden_feature_columns: tuple[str, ...] = (
        "purchase_price",  # the target
        "settlement_date",  # later than contract
        "download_date",  # data snapshot date (2024)
        "property_id",  # strata plan id, not a property id
        "legal_description",  # parcel fingerprint
        "address",  # used only as the CV group key
        "price_per_sqm",  # target / feature ratio
        "cash_rate",  # annual mean -> leakage (see macro_asof)
        "cpi",  # annual mean -> leakage
        "development_type",  # unversioned label
    )


@dataclass(frozen=True)
class SampleWindows:
    """Time-based windows used by the modelling pipeline."""

    # The clean dataset is built over the full history, but IQR bounds and any
    # other fitted quantity may only learn from the training span.
    train_end: str = "2015-12-31"
    valid_start: str = "2016-01-01"
    embargo_months: int = 1


@dataclass(frozen=True)
class Settings:
    paths: dict = field(default_factory=lambda: {
        "raw_property": RAW_PROPERTY_CSV,
        "postcode": POSTCODE_CSV,
        "station": STATION_CSV,
    })
    cleaning: CleaningPolicy = CleaningPolicy()
    plausibility: PlausibilityPolicy = PlausibilityPolicy()
    windows: SampleWindows = SampleWindows()
    random_seed: int = 36103


SETTINGS = Settings()


def ensure_dirs() -> None:
    """Create output directories (never touches raw inputs)."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
                      REPORTS_DIR / "figures", REPORTS_DIR / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
