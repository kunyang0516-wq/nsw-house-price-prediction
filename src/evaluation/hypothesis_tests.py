"""Hypothesis tests for the two EDA claims (audit §3, next-steps §3).

H1 — area vs price
    The notebook reports Pearson -0.0315, Spearman -0.1082, log-Pearson -0.1112
    on the whole sample with no uncertainty and no controls. Here:
      * bootstrap CIs, resampling **postcodes** (rows are not independent),
      * the same statistics per year, to show whether a single number is even
        meaningful across 23 years,
      * a within-location estimate: log price on log area with postcode and year
        fixed effects, clustered by postcode. If the pooled negative correlation
        is a between-location artefact, beta flips sign once location is absorbed.

H2 — "larger area, lower price per square metre"
    This hypothesis is partly an artefact of the ratio: ``price_per_sqm`` has the
    predictor in its denominator, so a negative correlation appears even when
    price and area are independent. Three tests separate the artefact from the
    economics:
      1. **placebo**: shuffle price, recompute the ratio correlation. Whatever
         survives under the placebo is the mechanical component.
      2. **symmetry/elasticity**: estimate log price on log area. beta < 1 is the
         honest statement of "unit price falls with size"; beta = 1 means no
         discount. Testing beta = 1 avoids the ratio entirely.
      3. **within-area-band medians**: descriptive check of monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.causal.event_study import _cluster_cov, _demean_two_way


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cluster_bootstrap_indices(postcodes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample postcode clusters with replacement, return row positions.

    Row positions per postcode are precomputed once per call to keep the 300-draw
    bootstrap affordable on ~1M rows.
    """
    codes, uniques = pd.factorize(postcodes)
    positions = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    draws = rng.integers(0, len(uniques), size=len(uniques))
    return np.concatenate([positions[d] for d in draws])


def restrict_residential_area(
    frame: pd.DataFrame,
    area_col: str = "area_sqm",
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop implausible recorded areas before the area-price tests.

    The cleaned sample contains 4.8% of rows above 20,000 sqm (a suspiciously
    round value), a maximum of 2.7e9 sqm, and a visible break in the unit-price
    curve above roughly 2,000-3,000 sqm. Those rows look like acreage or corrupted
    `area` entries rather than the standard residential lots the EDA describes.
    """
    from src.utils.config import SETTINGS
    minimum = SETTINGS.cleaning.area_plausible_min if minimum is None else minimum
    maximum = SETTINGS.cleaning.area_plausible_max if maximum is None else maximum

    values = pd.to_numeric(frame[area_col], errors="coerce")
    keep = values.between(minimum, maximum)
    report = {
        "area_min": minimum,
        "area_max": maximum,
        "rows_before": int(len(frame)),
        "rows_dropped": int((~keep).sum()),
        "share_dropped": float((~keep).mean()),
        "rows_after": int(keep.sum()),
    }
    return frame.loc[keep].copy(), report


def bootstrap_correlation_ci(
    frame: pd.DataFrame,
    x: str,
    y: str,
    method: str = "pearson",
    log_transform: bool = False,
    n_boot: int = 300,
    cluster_col: str = "post_code",
    seed: int = 36103,
) -> dict:
    """Point estimate plus a postcode-clustered bootstrap CI."""
    data = frame[[x, y, cluster_col]].dropna()
    if log_transform:
        data = data.loc[(data[x] > 0) & (data[y] > 0)]
        xs = np.log10(data[x].to_numpy(dtype=float))
        ys = np.log10(data[y].to_numpy(dtype=float))
    else:
        xs = data[x].to_numpy(dtype=float)
        ys = data[y].to_numpy(dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if method == "spearman":
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    point = corr(xs, ys)
    postcodes = data[cluster_col].astype("string").to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = _cluster_bootstrap_indices(postcodes, rng)
        draws.append(corr(xs[index], ys[index]))
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    return {
        "measure": ("log10 " if log_transform else "") + method,
        "x": x,
        "y": y,
        "n": int(len(data)),
        "n_postcodes": int(pd.Series(postcodes).nunique()),
        "coefficient": point,
        "ci_low": float(np.percentile(valid, 2.5)) if len(valid) else float("nan"),
        "ci_high": float(np.percentile(valid, 97.5)) if len(valid) else float("nan"),
        "boot_std": float(np.std(valid)) if len(valid) else float("nan"),
        "bootstrap_draws": int(len(valid)),
    }


def partial_out(
    frame: pd.DataFrame,
    y: str,
    x: str,
    absorb: list[str],
    cluster_col: str = "post_code",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Residualise y and x on the absorbed fixed effects (two-way demeaning)."""
    data = frame[[y, x, *absorb, cluster_col]].dropna()
    groups = data[absorb[0]].astype("string").to_numpy()
    times = data[absorb[1]].astype("string").to_numpy() if len(absorb) > 1 else np.zeros(len(data))
    residualised, n_groups, n_times = _demean_two_way(
        data.loc[:, [y, x]].to_numpy(dtype=float), groups, times)
    return (data[y].to_numpy(dtype=float), residualised[:, 0], residualised[:, 1],
            n_groups + n_times)


def _one_dim(series_like) -> np.ndarray:
    """Coerce a possibly-duplicated column selection to a 1-D string array."""
    values = series_like
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return values.astype("string").to_numpy().ravel()


def fe_regression(
    frame: pd.DataFrame,
    y: str,
    x: str,
    absorb: list[str],
    cluster_col: str = "post_code",
    test_value: float | None = None,
    label: str = "",
    controls: list[str] | None = None,
    interactions: dict[str, str] | None = None,
) -> dict:
    """Fixed-effects regression with clustered SEs.

    ``controls`` are additional regressors (residualised alongside ``x``);
    ``interactions`` maps a new name to ``"<a>*<b>"``-style products of existing
    columns. Use `describe_bad_control` when a control sits on the causal path
    from ``x`` to ``y``.
    """
    controls = controls or []
    interactions = interactions or {}
    columns = list(dict.fromkeys([y, x, *absorb, cluster_col, *controls]))
    data = frame.loc[:, columns].dropna().copy()

    regressors: list[str] = [x, *controls]
    for name, expression in interactions.items():
        left, _, right = expression.partition("*")
        if left not in data.columns or right not in data.columns:
            raise KeyError(f"Interaction {expression!r} references a missing column.")
        data[name] = pd.to_numeric(data[left], errors="coerce") * pd.to_numeric(data[right], errors="coerce")
        regressors.append(name)
    data = data.dropna(subset=regressors)
    if data.empty:
        raise ValueError(f"No rows left for {label or x}.")

    groups = _one_dim(data.loc[:, absorb[0]])
    times = _one_dim(data.loc[:, absorb[1]]) if len(absorb) > 1 else np.zeros(len(data))

    matrix = np.column_stack([data[y].to_numpy(dtype=float)]
                             + [data[c].to_numpy(dtype=float) for c in regressors])
    residualised, n_groups, n_times = _demean_two_way(matrix, groups, times)
    y_res = residualised[:, 0]
    x_res = residualised[:, 1:]

    keep = [i for i in range(x_res.shape[1]) if np.std(x_res[:, i]) > 1e-12]
    if not keep:
        raise ValueError(f"All regressors are collinear with the absorbed effects ({label}).")
    x_res = x_res[:, keep]
    kept_names = [regressors[i] for i in keep]

    beta, *_ = np.linalg.lstsq(x_res, y_res, rcond=None)
    resid = y_res - x_res @ beta
    covariance = _cluster_cov(x_res, resid, data[cluster_col].astype("string").to_numpy())
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))

    result = {
        "label": label or f"{y} ~ {x}",
        "y": y,
        "x": x,
        "absorbed": "+".join(absorb),
        "regressors": "+".join(kept_names),
        "n": int(len(data)),
        "n_clusters": int(pd.Series(data[cluster_col]).nunique()),
        "absorbed_levels": int(n_groups + n_times),
    }
    for position, name in enumerate(kept_names):
        result[f"coef__{name}"] = float(beta[position])
        result[f"se__{name}"] = float(se[position])
    # Primary coefficient kept under the plain names for compatibility.
    result.update({
        "coef": float(beta[0]),
        "se": float(se[0]),
        "ci_low": float(beta[0] - 1.96 * se[0]),
        "ci_high": float(beta[0] + 1.96 * se[0]),
    })
    if test_value is not None and se[0] > 0:
        z = (float(beta[0]) - test_value) / float(se[0])
        try:
            from scipy import stats
            p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        except Exception:
            p_value = float("nan")
        result.update({"test_value": test_value, "z_statistic": float(z), "p_value": p_value})
    return result


def describe_bad_control() -> str:
    """The standing caveat for any regression that controls for `pc_med_price_*`."""
    return (
        "`pc_med_price_*` is a function of prices in the same postcode, and area is "
        "a component of price. It therefore sits *downstream* of the regressor on "
        "the causal path (area -> price -> neighbourhood median). Conditioning on "
        "it absorbs part of the very variation being estimated and biases the area "
        "coefficient toward zero, so these estimates are a lower bound and are "
        "reported for transparency only, never as the headline."
    )


# --------------------------------------------------------------------------- #
# H1
# --------------------------------------------------------------------------- #
@dataclass
class H1Result:
    headline: list[dict] = field(default_factory=list)
    per_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    within_location: dict = field(default_factory=dict)
    area_scope: dict = field(default_factory=dict)
    interpretation: str = ""

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "h1_headline_correlations": pd.DataFrame(self.headline),
            "h1_by_year": self.per_year,
            "h1_within_location": pd.DataFrame([self.within_location]),
            "h1_area_scope": pd.DataFrame([self.area_scope]),
        }


def test_h1_area_price(
    frame: pd.DataFrame,
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
    n_boot: int = 300,
    seed: int = 36103,
    restrict_area: bool = True,
    area_min: float | None = None,
    area_max: float | None = None,
) -> H1Result:
    """Area vs price: pooled correlations, per-year correlations, FE estimate."""
    data = frame.copy()
    area_scope: dict = {}
    if restrict_area:
        data, area_scope = restrict_residential_area(data, area_col, area_min, area_max)

    data["log_area"] = np.log10(data[area_col].where(data[area_col] > 0))
    data["log_price"] = np.log10(data[value_col].where(data[value_col] > 0))
    data["year"] = pd.to_datetime(data["contract_date"]).dt.year
    data = data.dropna(subset=["log_area", "log_price", "post_code", "year"])

    headline = [
        bootstrap_correlation_ci(data, area_col, value_col, "pearson", False, n_boot, seed=seed),
        bootstrap_correlation_ci(data, area_col, value_col, "spearman", False, n_boot, seed=seed),
        bootstrap_correlation_ci(data, area_col, value_col, "pearson", True, n_boot, seed=seed),
    ]

    per_year_rows = []
    for year, block in data.groupby("year", observed=True):
        if len(block) < 500:
            continue
        row = bootstrap_correlation_ci(block, area_col, value_col, "pearson", True,
                                       n_boot=max(50, n_boot // 5), seed=seed)
        row["year"] = int(year)
        per_year_rows.append(row)
    per_year = pd.DataFrame(per_year_rows)

    within = fe_regression(data, "log_price", "log_area",
                           absorb=["post_code", "year"], label="log price ~ log area | postcode + year FE")

    pooled = next(r for r in headline if r["measure"] == "log10 pearson")
    sign_flip = (pooled["coefficient"] < 0 < within["coef"])
    interpretation = (
        "Pooled log-log correlation is negative, but the within-location FE "
        "estimate is " + ("positive" if within["coef"] > 0 else "negative") +
        f" ({within['coef']:.3f}). " +
        ("The negative pooled association is therefore largely a between-location "
         "artefact: within a postcode and year, larger recorded area is associated "
         "with a *higher* price."
         if sign_flip else
         "The association is not explained away by location and time alone.") +
        " Scope is standard residential lots"
        + (f" ({area_scope['area_min']:.0f}-{area_scope['area_max']:.0f} sqm, "
           f"{area_scope['rows_dropped']:,} rows / {area_scope['share_dropped']:.1%} excluded as "
           "implausible or acreage)" if area_scope else "")
        + ", so this does not extend to acreage or semi-rural parcels."
    )
    return H1Result(headline=headline, per_year=per_year, within_location=within,
                    area_scope=area_scope, interpretation=interpretation)


# --------------------------------------------------------------------------- #
# H2
# --------------------------------------------------------------------------- #
@dataclass
class RobustnessResult:
    """H1/H2 robustness: what changes when neighbourhood price is controlled for."""

    h1_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    h2_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    bad_control_note: str = ""
    interpretation: str = ""

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "h1_robustness_neighbourhood": self.h1_table,
            "h2_robustness_neighbourhood": self.h2_table,
        }


def neighbourhood_robustness(
    frame: pd.DataFrame,
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
    lag_col: str = "pc_med_price_3m",
    restrict_area: bool = True,
) -> RobustnessResult:
    """Re-run H1/H2 with the lagged neighbourhood price in the specification.

    Three specifications per hypothesis, so the direction of the bias is visible:

    H1
      R0  main:            log price ~ log area | postcode + year FE
      R1  bad control:     R0 + log pc_med_price_3m
      R2  interaction:     R0 + log area x log pc_med_price_3m   (preferred)

    H2 (elasticity, test beta = 1)
      R0  main, R1 + log neighbourhood price
    """
    data = frame.copy()
    area_scope: dict = {}
    if restrict_area:
        data, area_scope = restrict_residential_area(data, area_col)

    data["log_area"] = np.log10(data[area_col].where(data[area_col] > 0))
    data["log_price"] = np.log10(data[value_col].where(data[value_col] > 0))
    data["year"] = pd.to_datetime(data["contract_date"]).dt.year
    data["log_lag_price"] = np.log10(pd.to_numeric(data[lag_col], errors="coerce").where(
        pd.to_numeric(data[lag_col], errors="coerce") > 0))
    data = data.dropna(subset=["log_area", "log_price", "post_code", "year"])
    usable = data["log_lag_price"].notna()
    print(f"  neighbourhood robustness: {int(usable.sum()):,} of {len(data):,} rows have "
          f"a lagged neighbourhood price ({usable.mean():.1%})")

    # Centre the neighbourhood-price term so the main effect in the interaction
    # specification reads as the elasticity at the average market level rather
    # than at an arbitrary zero.
    data["log_lag_price_c"] = data["log_lag_price"] - data["log_lag_price"].mean()

    h1_rows = []
    h1_rows.append(fe_regression(data, "log_price", "log_area",
                                 absorb=["post_code", "year"],
                                 label="H1-R0 main (postcode + year FE)"))
    h1_rows.append(fe_regression(data, "log_price", "log_area",
                                 absorb=["post_code", "year"],
                                 controls=["log_lag_price"],
                                 label="H1-R1 + neighbourhood price (bad control)"))
    h1_rows.append(fe_regression(data, "log_price", "log_area",
                                 absorb=["post_code", "year"],
                                 controls=["log_lag_price_c"],
                                 interactions={"log_area_x_lag_c": "log_area*log_lag_price_c"},
                                 label="H1-R2 + area x neighbourhood price (centred)"))

    h2_rows = []
    h2_rows.append(fe_regression(data, "log_price", "log_area",
                                 absorb=["post_code", "year"], test_value=1.0,
                                 label="H2-R0 main elasticity (test beta = 1)"))
    h2_rows.append(fe_regression(data, "log_price", "log_area",
                                 absorb=["post_code", "year"], test_value=1.0,
                                 controls=["log_lag_price"],
                                 label="H2-R1 + neighbourhood price (bad control)"))

    h1_table = pd.DataFrame(h1_rows)
    h2_table = pd.DataFrame(h2_rows)

    base = float(h1_table.loc[0, "coef"])
    controlled = float(h1_table.loc[1, "coef"])
    interaction_name = "log_area_x_lag_c"
    interaction = float(h1_table.loc[2, f"coef__{interaction_name}"])
    interaction_se = float(h1_table.loc[2, f"se__{interaction_name}"])
    interaction_z = interaction / interaction_se if interaction_se > 0 else float("nan")
    shift = (controlled - base) / abs(base)
    within = abs(shift) < 0.05
    interpretation = (
        f"Controlling for the lagged neighbourhood price moves the area elasticity from "
        f"{base:+.3f} to {controlled:+.3f} ({shift:+.1%}). "
        + ("The estimate barely moves, so the area gradient does **not** appear to run through "
           "neighbourhood price momentum and the bad-control concern is, empirically, second-order "
           "here. That is a finding, not a reassurance: it means the pooled negative correlation "
           "was about *between-postcode* differences, which the postcode fixed effects already absorb."
           if within else
           "The estimate shifts materially, which is the bad-control signature: conditioning on a "
           "mediator absorbs part of the effect. R0 remains the headline and R1 is a lower bound.")
        + f" The interaction term is {interaction:+.3f} (se {interaction_se:.3f}, z {interaction_z:+.1f}), "
        + ("so the area gradient does vary with the local market level"
           if abs(interaction_z) > 2 else
           "so there is no evidence that the area gradient varies with the local market level")
        + ". In the interaction specification the main effect is evaluated at the average "
        "neighbourhood price because the term is centred."
    )
    return RobustnessResult(h1_table=h1_table, h2_table=h2_table,
                            bad_control_note=describe_bad_control(),
                            interpretation=interpretation)


@dataclass
class H2Result:
    observed: dict = field(default_factory=dict)
    placebo: pd.DataFrame = field(default_factory=pd.DataFrame)
    elasticity: dict = field(default_factory=dict)
    area_bands: pd.DataFrame = field(default_factory=pd.DataFrame)
    area_scope: dict = field(default_factory=dict)
    interpretation: str = ""

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "h2_ratio_correlation": pd.DataFrame([self.observed]),
            "h2_placebo_ratio": self.placebo,
            "h2_elasticity": pd.DataFrame([self.elasticity]),
            "h2_unit_price_by_area_band": self.area_bands,
            "h2_area_scope": pd.DataFrame([self.area_scope]),
        }


def test_h2_unit_price(
    frame: pd.DataFrame,
    value_col: str = "purchase_price",
    area_col: str = "area_sqm",
    n_placebo: int = 200,
    seed: int = 36103,
    restrict_area: bool = True,
    area_min: float | None = None,
    area_max: float | None = None,
) -> H2Result:
    """Unit-price hypothesis with an explicit ratio-bias placebo."""
    data = frame.copy()
    area_scope: dict = {}
    if restrict_area:
        data, area_scope = restrict_residential_area(data, area_col, area_min, area_max)

    data["log_area"] = np.log10(data[area_col].where(data[area_col] > 0))
    data["log_price"] = np.log10(data[value_col].where(data[value_col] > 0))
    data["price_per_sqm"] = data[value_col] / data[area_col]
    data["log_unit_price"] = np.log10(data["price_per_sqm"].where(data["price_per_sqm"] > 0))
    data["year"] = pd.to_datetime(data["contract_date"]).dt.year
    data = data.dropna(subset=["log_area", "log_unit_price", "log_price", "post_code"])

    rng = np.random.default_rng(seed)

    def ratio_corr(block: pd.DataFrame) -> float:
        values = block[["log_area", "log_unit_price"]].to_numpy(dtype=float)
        if np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
            return float("nan")
        return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])

    observed_coef = ratio_corr(data)
    observed = {
        "measure": "corr(log area, log unit price)",
        "coefficient": observed_coef,
        "n": int(len(data)),
    }

    # Placebo: shuffle price within the sample. Area is untouched, so any
    # remaining negative correlation can only come from the ratio's denominator.
    log_area = data["log_area"].to_numpy(dtype=float)
    price_values = data[value_col].to_numpy(dtype=float)
    area_values = data[area_col].to_numpy(dtype=float)
    draws = []
    for _ in range(n_placebo):
        shuffled = rng.permutation(price_values)
        unit = shuffled / area_values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_unit = np.log10(np.where(unit > 0, unit, np.nan))
        mask = np.isfinite(log_unit)
        if mask.sum() > 10 and np.std(log_area[mask]) > 0 and np.std(log_unit[mask]) > 0:
            draws.append(float(np.corrcoef(log_area[mask], log_unit[mask])[0, 1]))
    draws = np.asarray(draws, dtype=float)
    placebo = pd.DataFrame({
        "placebo_draw": np.arange(1, len(draws) + 1),
        "coefficient": draws,
    })
    placebo_median = float(np.median(draws)) if len(draws) else float("nan")
    placebo_ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if len(draws) else (np.nan, np.nan)

    elasticity = fe_regression(data, "log_price", "log_area",
                               absorb=["post_code", "year"], test_value=1.0,
                               label="log price ~ log area | postcode + year FE (test beta=1)")

    bands = data.copy()
    bands["area_band"] = pd.qcut(bands["log_area"].rank(method="first"), 10, labels=False)
    area_bands = (bands.groupby("area_band", observed=True)
                  .agg(median_area=(area_col, "median"),
                       median_unit_price=("price_per_sqm", "median"),
                       median_price=(value_col, "median"),
                       n=(value_col, "size"))
                  .reset_index())
    area_bands["unit_price_index"] = (area_bands["median_unit_price"]
                                      / area_bands["median_unit_price"].iloc[0] * 100)

    mechanical_share = (abs(placebo_median) / abs(observed_coef)) if observed_coef else float("nan")
    interpretation = (
        f"Observed corr(log area, log unit price) = {observed_coef:.3f}. "
        f"Under the shuffled-price placebo the same statistic is {placebo_median:.3f} "
        f"(95% band {placebo_ci[0]:.3f}..{placebo_ci[1]:.3f}), so about "
        f"{mechanical_share:.0%} of the observed association is the mechanical "
        "ratio effect. The elasticity estimate (log price on log area, within "
        f"postcode and year) is {elasticity['coef']:.3f} with a test of beta=1 "
        f"giving p={elasticity.get('p_value', float('nan')):.3g}; "
        + ("a slope below 1 means unit price genuinely falls with size"
           if elasticity["coef"] < 1 else
           "a slope at or above 1 means no unit-price discount")
        + " once location and time are held fixed."
    )
    return H2Result(observed=observed, placebo=placebo, elasticity=elasticity,
                    area_bands=area_bands, area_scope=area_scope,
                    interpretation=interpretation)
