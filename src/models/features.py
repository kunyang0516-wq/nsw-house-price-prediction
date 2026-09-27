"""sklearn-compatible, fold-aware feature transformer.

Everything that *learns from data* lives here so it can be fitted on a training
fold and applied to a validation fold (audit §0 rule 2, L2/L3):

* median imputation and missing indicators for numeric columns
* category **frequency** encoding
* **smoothed target encoding** for high-cardinality categoricals (out-of-fold,
  never on the whole sample)
* quantile binning of area (thresholds from the training fold only)
* optional standardisation

Columns that are already safe numerics (log transforms, distances, calendar,
macro as-of, rolling postcode features) are passed through untouched. No step
here touches ``purchase_price`` except the target encoder, which only ever sees
the training fold's target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class FeatureSpec:
    """Declares which clean columns play which role.

    Calendar features are numeric on purpose. Encoding `year` as a *frequency*
    (as an earlier revision did) tells the model how common a year was in the
    training fold, not where that year sits in time, so a tree could not
    extrapolate a trend at all. `year_num` is continuous (year + month/12) and
    `month_sin`/`month_cos` give seasonality a smooth periodic encoding.
    """

    numeric: tuple[str, ...] = (
        "area_sqm", "log_area", "dist_cbd", "log_dist_cbd", "dist_train",
        "dist_metro", "dist_cbd_x_area", "cash_rate_asof", "cpi_yoy_asof",
        "year_num", "month_sin", "month_cos",
        "pc_med_price_3m", "pc_med_unit_price_3m", "pc_n_sales_3m", "pc_n_months_observed_3m",
        "pc_med_price_6m", "pc_med_unit_price_6m", "pc_n_sales_6m", "pc_n_months_observed_6m",
        "pc_med_price_12m", "pc_med_unit_price_12m", "pc_n_sales_12m",
        "pc_price_iqr_ratio_12m", "pc_n_months_observed_12m",
        "pc_months_since_sale", "pc_months_observed",
    )
    categorical_low_card: tuple[str, ...] = (
        "development_type", "zoning_clean",
    )
    categorical_high_card: tuple[str, ...] = (
        "post_code", "council_name", "locality",
    )
    # Binned area is created inside the transformer from `area_sqm`.
    bin_source: str = "area_sqm"
    bin_quantiles: int = 10
    target_smoothing: float = 20.0
    standardize: bool = True

    def all_source_columns(self) -> list[str]:
        """Source columns, de-duplicated and order-preserving.

        `area_sqm` appears both as a numeric feature and as the bin source; a
        duplicated name would make `frame[[...]]` return a DataFrame instead of a
        Series and silently break the numeric blocks.
        """
        seen: dict[str, None] = {}
        for column in (*self.numeric, *self.categorical_low_card,
                       *self.categorical_high_card, self.bin_source):
            seen.setdefault(column, None)
        return list(seen)

    def to_dict(self) -> dict:
        return {
            "numeric": list(self.numeric),
            "categorical_low_card": list(self.categorical_low_card),
            "categorical_high_card": list(self.categorical_high_card),
            "bin_source": self.bin_source,
            "bin_quantiles": self.bin_quantiles,
            "target_smoothing": self.target_smoothing,
            "standardize": self.standardize,
        }


class PropertyFeatureTransformer(BaseEstimator, TransformerMixin):
    """Fit on a training fold, transform either fold into a numeric matrix."""

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self.feature_names_: list[str] = []
        self.impute_values_: dict[str, float] = {}
        self.frequency_maps_: dict[str, pd.Series] = {}
        self.target_maps_: dict[str, pd.Series] = {}
        self.bin_edges_: np.ndarray | None = None
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_scale_: np.ndarray | None = None
        self.global_target_mean_: float | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureTransformer":
        spec = self.spec
        if y is None:
            raise ValueError("Target encoder needs y (log price) to fit.")

        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_target_mean_ = float(np.nanmean(y))

        # Numeric imputation values
        for column in spec.numeric:
            if column in X.columns:
                self.impute_values_[column] = float(pd.to_numeric(X[column], errors="coerce").median())

        # Quantile bin edges for area
        if spec.bin_source in X.columns:
            values = pd.to_numeric(X[spec.bin_source], errors="coerce").dropna()
            if len(values):
                edges = np.unique(np.quantile(values, np.linspace(0, 1, spec.bin_quantiles + 1)))
                self.bin_edges_ = edges if len(edges) > 1 else None

        # Frequency + smoothed target encoding, training fold only
        for column in (*spec.categorical_low_card, *spec.categorical_high_card):
            if column not in X.columns:
                continue
            keys = X[column].astype("string").fillna("<missing>")
            self.frequency_maps_[column] = keys.value_counts(normalize=True)

            stats = pd.DataFrame({"key": keys, "y": y}).groupby("key")["y"].agg(["mean", "size"])
            smooth = spec.target_smoothing
            encoded = (stats["mean"] * stats["size"] + self.global_target_mean_ * smooth) / (stats["size"] + smooth)
            self.target_maps_[column] = encoded

        self._build_names(X)
        matrix = self._to_matrix(X, fitting=True)
        if spec.standardize:
            self.scaler_mean_ = np.nanmean(matrix, axis=0)
            scale = np.nanstd(matrix, axis=0)
            scale[scale == 0] = 1.0
            self.scaler_scale_ = scale
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_target_mean_ is None:
            raise RuntimeError("Transformer is not fitted.")
        matrix = self._to_matrix(X, fitting=False)
        if self.spec.standardize:
            matrix = (matrix - self.scaler_mean_) / self.scaler_scale_
        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)

    # ------------------------------------------------------------------ #
    def _original_columns(self) -> list[str]:
        spec = self.spec
        return [*spec.numeric, *spec.categorical_low_card, *spec.categorical_high_card]

    def _build_names(self, X: pd.DataFrame) -> None:
        names: list[str] = []
        for column in self.spec.numeric:
            if column in X.columns:
                names.append(column)
                names.append(f"{column}__isna")
        for column in self.spec.categorical_low_card:
            if column in X.columns:
                names.append(f"{column}__freq")
        for column in self.spec.categorical_high_card:
            if column in X.columns:
                names.append(f"{column}__freq")
                names.append(f"{column}__te")
        if self.bin_edges_ is not None:
            names.append(f"{self.spec.bin_source}__bin")
        self.feature_names_ = names

    def _to_matrix(self, X: pd.DataFrame, fitting: bool) -> np.ndarray:
        spec = self.spec
        blocks: list[np.ndarray] = []

        for column in spec.numeric:
            if column not in X.columns:
                continue
            raw = pd.to_numeric(X[column], errors="coerce")
            missing = raw.isna().to_numpy(dtype=float)
            fill = self.impute_values_.get(column, 0.0)
            blocks.append(raw.fillna(fill).to_numpy(dtype=float).reshape(-1, 1))
            blocks.append(missing.reshape(-1, 1))

        for column in spec.categorical_low_card:
            if column not in X.columns:
                continue
            keys = X[column].astype("string").fillna("<missing>")
            freq = keys.map(self.frequency_maps_.get(column, pd.Series(dtype=float))).astype(float)
            blocks.append(freq.fillna(0.0).to_numpy().reshape(-1, 1))

        for column in spec.categorical_high_card:
            if column not in X.columns:
                continue
            keys = X[column].astype("string").fillna("<missing>")
            freq = keys.map(self.frequency_maps_.get(column, pd.Series(dtype=float))).astype(float)
            target = keys.map(self.target_maps_.get(column, pd.Series(dtype=float))).astype(float)
            blocks.append(freq.fillna(0.0).to_numpy().reshape(-1, 1))
            blocks.append(target.fillna(self.global_target_mean_).to_numpy().reshape(-1, 1))

        if self.bin_edges_ is not None and spec.bin_source in X.columns:
            values = pd.to_numeric(X[spec.bin_source], errors="coerce")
            binned = pd.cut(values, bins=self.bin_edges_, labels=False, include_lowest=True)
            blocks.append(binned.astype(float).fillna(-1).to_numpy().reshape(-1, 1))

        if not blocks:
            raise ValueError("No feature blocks could be built — check the FeatureSpec against the frame.")
        return np.hstack(blocks)


def make_xy(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    target_column: str = "log_price",
    date_column: str = "contract_date",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a prepared frame into (X, y, dates) without dropping information."""
    y = pd.to_numeric(frame[target_column], errors="coerce")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    columns = [c for c in spec.all_source_columns() if c in frame.columns]
    X = frame.loc[:, columns].copy()
    return X, y, dates
