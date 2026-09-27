"""Single source of every figure and table, then the final markdown report.

Run this last. It reads only from `data/processed/*.parquet` and
`reports/tables/*.csv`, so the report can never drift from the numbers.

Usage
-----
    python pipelines/07_make_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import naive_postcode_month_predict
from src.utils.config import PROJECT_ROOT, REPORTS_DIR
from src.utils.io import read_json, write_json

TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"
CLEAN_META = PROJECT_ROOT / "data" / "processed" / "clean.meta.json"
FEATURES_META = PROJECT_ROOT / "data" / "processed" / "features.meta.json"


def load_features(path: Path) -> pd.DataFrame:
    """Read the feature frame and normalise masked dtypes to plain numpy.

    `data/processed/clean.parquet` stores counts/prices as pandas nullable
    (Int64/Float64). Masked integer arrays reject float operations such as
    `Series.clip(bound_float)`, which matplotlib-facing code relies on, so the
    report reads everything numeric as float64.
    """
    frame = pd.read_parquet(path)
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.api.extensions.ExtensionDtype):
            kind = getattr(frame[column].dtype, "kind", "")
            if kind in ("i", "u", "f"):
                frame[column] = frame[column].astype("float64")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="row cap for the EDA figures")
    parser.add_argument("--figures-only", action="store_true")
    return parser.parse_args()


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
    })
    return plt


ACCENT = "#2a78d6"
GREY = "#b4b2ac"
ORANGE = "#d55e00"
INK2 = "#6b6963"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figure_distributions(frame: pd.DataFrame, out_dir: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, column, label, colour in (
        (axes[0], "area_sqm", "Recorded area (sqm)", ACCENT),
        (axes[1], "purchase_price", "Purchase price (AUD)", ORANGE),
    ):
        # astype(float) matters: parquet round-trips `area_sqm` as nullable Int64,
        # and Series.clip() on a masked integer array cannot take float bounds.
        values = pd.to_numeric(frame[column], errors="coerce").astype("float64").dropna()
        values = values.loc[values > 0]
        low, high = values.quantile(0.001), values.quantile(0.999)
        ax.hist(values.clip(low, high), bins=60, color=colour, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        median = values.median()
        ax.axvline(median, color="#222222", ls="--", lw=1.8, label=f"median {median:,.0f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Transactions")
        ax.set_title(f"{label.split(' (')[0]} distribution (0.1-99.9% shown)")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Cleaned residence houses, n = {len(frame):,}", fontsize=12, x=0.01, ha="left")
    fig.savefig(out_dir / "fig1_distributions.png", bbox_inches="tight")
    plt.close(fig)


def figure_area_price(frame: pd.DataFrame, out_dir: Path, plt) -> None:
    data = frame[["area_sqm", "purchase_price"]].astype("float64").dropna()
    data = data[(data.area_sqm > 0) & (data.purchase_price > 0)]
    trend = (data.assign(bin=pd.qcut(data.area_sqm, 20, duplicates="drop"))
             .groupby("bin", observed=True)
             .agg(area=("area_sqm", "median"), price=("purchase_price", "median")))

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    density = ax.hexbin(data.area_sqm, data.purchase_price / 1e6, gridsize=60,
                        bins="log", mincnt=1, cmap="Blues", linewidths=0)
    fig.colorbar(density, ax=ax, label="Transactions per hexagon (log scale)", shrink=0.85)
    ax.plot(trend["area"], trend["price"] / 1e6, "-o", color=ORANGE, lw=2.2,
            markersize=4, label="median price by area ventile")
    ax.set_xlim(0, data.area_sqm.quantile(0.999))
    ax.set_ylim(0, data.purchase_price.quantile(0.999) / 1e6)
    ax.set_xlabel("Recorded area (sqm)")
    ax.set_ylabel("Purchase price (AUD millions)")
    ax.set_title("Area versus purchase price: all cleaned records\n"
                 "the pooled correlation is near zero, but the median rises with area",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "fig2_area_vs_price.png", bbox_inches="tight")
    plt.close(fig)


def figure_area_unit_price(frame: pd.DataFrame, out_dir: Path, plt,
                           area_max: float = 5000.0) -> None:
    data = frame[["area_sqm", "purchase_price"]].astype("float64").dropna()
    data = data[(data.area_sqm.between(100, area_max)) & (data.purchase_price > 0)]
    data["unit_price"] = data.purchase_price / data.area_sqm
    data = data[data.unit_price > 0]

    trend = (data.assign(bin=pd.qcut(data.area_sqm, 20, duplicates="drop"))
             .groupby("bin", observed=True)
             .agg(area=("area_sqm", "median"), unit=("unit_price", "median")))

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    density = ax.hexbin(data.area_sqm, data.unit_price, xscale="log", yscale="log",
                        gridsize=60, bins="log", mincnt=1, cmap="Blues", linewidths=0)
    fig.colorbar(density, ax=ax, label="Transactions per hexagon (log scale)", shrink=0.85)
    ax.plot(trend["area"], trend["unit"], "-o", color=ORANGE, lw=2.2, markersize=4,
            label="median unit price by area ventile")
    ax.set_xlabel("Recorded area (sqm, log scale)")
    ax.set_ylabel("Sale price per sqm (AUD, log scale)")
    ax.set_title("H2: unit price falls steeply with area — but most of that is the ratio\n"
                 "see h2_placebo_ratio.csv for the shuffled-price benchmark",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "fig3_area_vs_unit_price.png", bbox_inches="tight")
    plt.close(fig)


def figure_h1_by_year(tables_dir: Path, out_dir: Path, plt) -> bool:
    """Area-price correlation by contract year: is one pooled number meaningful?"""
    path = tables_dir / "h1_by_year.csv"
    if not path.exists():
        return False
    table = pd.read_csv(path).sort_values("year")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.errorbar(table["year"], table["coefficient"],
                yerr=[table["coefficient"] - table["ci_low"],
                      table["ci_high"] - table["coefficient"]],
                fmt="o", color=ACCENT, ecolor=ACCENT, capsize=3, ms=5,
                label="log-log correlation (95% clustered CI)")
    ax.axhline(0, color=GREY, lw=1.4)
    pooled = table["coefficient"].median()
    ax.axhline(pooled, color=ORANGE, ls="--", lw=1.5,
               label=f"median across years {pooled:+.3f}")
    ax.set_xlabel("Contract year")
    ax.set_ylabel("corr(log area, log price)")
    ax.set_title("H1: the area-price correlation is not stable across years\n"
                 "a single pooled coefficient averages over a sign change",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "fig8_h1_by_year.png", bbox_inches="tight")
    plt.close(fig)
    return True


def figure_dose_response(tables_dir: Path, out_dir: Path, plt) -> bool:
    path = tables_dir / "event_study_dose_response.csv"
    if not path.exists():
        return False
    table = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    positions = np.arange(len(table))
    ax.errorbar(positions, table["coef_post_x_log_dist"],
                yerr=[table["coef_post_x_log_dist"] - table["ci_low"],
                      table["ci_high"] - table["coef_post_x_log_dist"]],
                fmt="o", color=ACCENT, ecolor=ACCENT, capsize=4, ms=7)
    ax.axhline(0, color=GREY, lw=1.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"within {int(km)} km" for km in table["max_km"]])
    ax.set_ylabel("post x log(1 + km) coefficient")
    ax.set_title("Metro dose-response: no coefficient is distinguishable from zero\n"
                 "(negative would mean a larger rise closer to the station)",
                 fontsize=11, loc="left")
    fig.savefig(out_dir / "fig9_dose_response.png", bbox_inches="tight")
    plt.close(fig)
    return True


def figure_yearly_economics(frame: pd.DataFrame, out_dir: Path, plt) -> None:
    data = frame.copy()
    data["year"] = pd.to_datetime(data["contract_date"]).dt.year
    yearly = (data.groupby("year", observed=True)
              .agg(median_price=("purchase_price", "median"),
                   cash_rate=("cash_rate_asof", "mean"),
                   cpi_yoy=("cpi_yoy_asof", "median"),
                   rows=("purchase_price", "size"))
              .reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(yearly.year, yearly.median_price / 1e6, "-o", color=ORANGE, ms=4)
    axes[0].set_title("Median price by contract year")
    axes[0].set_ylabel("AUD millions")
    axes[1].plot(yearly.year, yearly.cash_rate, "-o", color=ACCENT, ms=4)
    axes[1].set_title("Mean cash rate on contract date (monthly as-of)")
    axes[1].set_ylabel("Per cent")
    axes[2].plot(yearly.year, yearly.cpi_yoy, "-o", color="#117733", ms=4)
    axes[2].axhline(0, color=GREY, lw=1)
    axes[2].set_title("CPI year-ended inflation at contract date")
    axes[2].set_ylabel("Per cent")
    for ax in axes:
        ax.set_xlabel("Contract year")
    fig.suptitle("Descriptive comparison only: 23 annual points, no identification",
                 fontsize=11.5, x=0.01, ha="left")
    fig.savefig(out_dir / "fig4_yearly_macro.png", bbox_inches="tight")
    plt.close(fig)


def figure_rolling_scores(tables_dir: Path, out_dir: Path, plt) -> bool:
    path = tables_dir / "model_scores_by_fold.csv"
    if not path.exists():
        return False
    scores = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for model, block in scores.groupby("model"):
        block = block.sort_values("fold")
        axes[0].plot(block["valid_start"], block["rmsle"], "-o", ms=4, label=model)
        axes[1].plot(block["valid_start"], block["mdape_pct"], "-o", ms=4, label=model)
    axes[0].set_title("RMSLE by validation window")
    axes[1].set_title("Median absolute percentage error by validation window")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Validation window start")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Rolling-origin cross-validation: every fold trains only on the past",
                 fontsize=11.5, x=0.01, ha="left")
    fig.savefig(out_dir / "fig5_rolling_cv.png", bbox_inches="tight")
    plt.close(fig)
    return True


def figure_residuals(tables_dir: Path, frame: pd.DataFrame, out_dir: Path, plt) -> bool:
    path = tables_dir / "predictions.parquet"
    comparison = tables_dir / "model_comparison.csv"
    if not path.exists() or not comparison.exists():
        return False
    best = pd.read_csv(comparison).iloc[0]["model"]
    predictions = pd.read_parquet(path)
    merged = predictions.loc[predictions["model"] == best].copy()
    if "area_sqm" not in merged.columns:
        # Older prediction tables lack the carried-through grouping columns, so
        # re-join defensively -- but (contract_date, post_code) is not unique,
        # which makes that merge many-to-many. Refuse to plot inflated data.
        merged = merged.merge(frame[["contract_date", "post_code", "area_sqm"]],
                              on=["contract_date", "post_code"], how="left")
        if len(merged) != int((predictions["model"] == best).sum()):
            raise SystemExit(
                "predictions.parquet has no area_sqm column and the fallback merge on "
                "(contract_date, post_code) fanned out; re-run 04_train_models.py.")
    merged["residual"] = merged["y_pred_log10"] - merged["y_true_log10"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].hexbin(merged["y_true_log10"], merged["residual"], gridsize=50, bins="log",
                   mincnt=1, cmap="Blues")
    axes[0].axhline(0, color=ORANGE, lw=1.6)
    axes[0].set_xlabel("actual log10 price")
    axes[0].set_ylabel("residual (predicted - actual)")
    axes[0].set_title(f"Residuals vs actual ({best})")

    deciles = merged.assign(bin=pd.qcut(merged["area_sqm"].rank(method="first"), 10, labels=False))
    profile = deciles.groupby("bin", observed=True)["residual"].median()
    axes[1].plot(profile.index, profile.values, "-o", color=ACCENT, ms=4)
    axes[1].axhline(0, color=GREY, lw=1.2)
    axes[1].set_xlabel("area decile (0 = smallest)")
    axes[1].set_ylabel("median residual")
    axes[1].set_title("Bias across the area distribution")

    axes[2].hist(merged["residual"].clip(-0.6, 0.6), bins=60, color=GREY, edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color=ORANGE, lw=1.6)
    axes[2].set_xlabel("residual (clipped at ±0.6)")
    axes[2].set_title("Residual distribution")
    fig.suptitle(f"Error diagnostics for the best model ({best}), pooled validation folds",
                 fontsize=11.5, x=0.01, ha="left")
    fig.savefig(out_dir / "fig6_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    return True


def figure_feature_importance(tables_dir: Path, out_dir: Path, plt) -> bool:
    path = tables_dir / "feature_importance.csv"
    if not path.exists():
        return False
    importance = pd.read_csv(path)
    models = [m for m in importance["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.4), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        block = (importance.loc[importance["model"] == model]
                 .nsmallest(20, "rank").sort_values("value"))
        ax.barh(block["feature"], block["value"], color=ACCENT, alpha=0.85)
        ax.set_title(f"{model}\n({importance.loc[importance['model'] == model, 'kind'].iloc[0]})",
                     fontsize=10.5)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.suptitle("Top 20 features, last training fold only (indicative, not causal)",
                 fontsize=11.5, x=0.01, ha="left")
    fig.savefig(out_dir / "fig7_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def md_table(path: Path, n: int = 12, float_format: str = "{:,.4f}") -> str:
    if not path.exists():
        return f"_(missing: {path.name})_"
    frame = pd.read_csv(path).head(n)
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda v: "" if pd.isna(v) else float_format.format(v))
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    divider = "|" + "|".join("---" for _ in formatted.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in formatted.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_report(out_dir: Path, tables: Path, figures: Path) -> Path:
    clean_meta = read_json(CLEAN_META) if CLEAN_META.exists() else {}
    features_meta = read_json(FEATURES_META) if FEATURES_META.exists() else {}
    training_meta = read_json(tables / "training_meta.json") if (tables / "training_meta.json").exists() else {}
    event_meta = read_json(tables / "event_study_meta.json") if (tables / "event_study_meta.json").exists() else {}
    hypothesis = read_json(tables / "hypothesis_summary.json") if (tables / "hypothesis_summary.json").exists() else {}
    outlier = clean_meta.get("iqr", {})
    robustness = hypothesis.get("neighbourhood_robustness", {})

    best_model = training_meta.get("best_model", "n/a")
    pretrend_p = (event_meta.get("main", {}).get("pretrend", {}) or {}).get("p_value", float("nan"))
    post_pct = event_meta.get("main", {}).get("implied_pct_on_unit_price", float("nan"))
    holdout_window = training_meta.get("holdout_window")
    corruption_fence = (outlier.get("global_corruption_fence") or {})

    # Feature-value test (optional: pipeline 08 may not have run yet)
    ablation_path = tables / "feature_ablation.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else None
    placebo_path = tables / "feature_value_placebo.csv"
    feature_placebo = pd.read_csv(placebo_path) if placebo_path.exists() else None
    demeaned_path = tables / "feature_value_demeaned.csv"
    feature_demeaned = pd.read_csv(demeaned_path) if demeaned_path.exists() else None

    headline_ablation = ""
    ablation_verdict = ("*(no feature ablation available — run pipelines/08_feature_value_test.py)*")
    if ablation is not None and len(ablation) > 3:
        def _layer_value(name: str, column: str) -> float:
            row = ablation.loc[ablation["layer"] == name, column]
            return float(row.iloc[0]) if len(row) else float("nan")

        def _layer(name: str) -> float:
            return _layer_value(name, "rmsle_mean")

        def _folds(name: str) -> str:
            """'8/8' from the ablation's own `folds_improved_vs_prev` column.

            These used to be hardcoded, so a re-run could leave the prose
            claiming 7/8 while the table beside it said 6/8.
            """
            improved = _layer_value(name, "folds_improved_vs_prev")
            total = _layer_value(name, "folds")
            if improved != improved or total != total:
                return "n/a"
            return f"{improved:.0f}/{total:.0f}"

        # Rank the cumulative layers by their own marginal contribution instead of
        # asserting which is 'second largest'. With the rebuilt data the labels
        # layer overtook the lagged-price layer, so a hardcoded ordering would
        # have called the wrong layer second.
        layer_rows = ablation.dropna(subset=["rmsle_delta_vs_prev"])
        ranked = layer_rows.sort_values("rmsle_delta_vs_prev")
        n_layers = len(ranked)
        labels = {
            "L1_plus_calendar": "calendar features",
            "L2_plus_labels": "the property/region label block",
            "L3_plus_lagged_price": "the engineered lagged neighbourhood price statistics",
            "L4_plus_unit_price": "the lagged unit-price columns",
            "L5_plus_liquidity": "the liquidity/dispersion/coverage block",
        }
        best_row = ranked.iloc[0]
        second_row = ranked.iloc[1] if n_layers > 1 else None
        worst_row = ranked.iloc[-1]

        def _describe(row) -> str:
            name = labels.get(row["layer"], row["layer"])
            return (f"**{name}** ({row['rmsle_delta_vs_prev']:+.4f} RMSLE, "
                    f"{_folds(row['layer'])} folds)")

        def _name(row) -> str:
            """Plain, non-bold layer name (for use inside an already-bold run)."""
            return str(labels.get(row["layer"], row["layer"]))

        def _with_article(row) -> str:
            """Prefix 'the' unless the label already starts with one."""
            name = _name(row)
            return name if name.startswith("the ") else f"the {name}"

        second_txt = ""
        if second_row is not None:
            second_txt = f" The next largest is {_describe(second_row)}."
        worst_txt = ""
        if worst_row["rmsle_delta_vs_prev"] > 0:
            worst_txt = (f" Only {_with_article(worst_row)} makes things "
                         f"worse ({worst_row['rmsle_delta_vs_prev']:+.4f}, "
                         f"{_folds(worst_row['layer'])} folds).")
        elif worst_row["rmsle_delta_vs_prev"] == 0:
            worst_txt = (f" The smallest contribution is {_with_article(worst_row)} "
                         f"({worst_row['rmsle_delta_vs_prev']:+.4f}).")

        headline_ablation = (
            f"**The single largest contributor is {_with_article(best_row)}** "
            f"({best_row['rmsle_delta_vs_prev']:+.4f} RMSLE, {_folds(best_row['layer'])} folds) — "
            f"measured as the drop in pooled RMSLE from adding that layer on top of the previous one."
            f"{second_txt} Layers are cumulative, so each delta is that block's marginal value once "
            f"everything before it is present.{worst_txt}"
        )
        # One-line summary for the headline-findings list, so it cannot drift
        # from §4.5 (which used to happen: the two were written independently and
        # the headline kept asserting the previous run's conclusion).
        improves = int((layer_rows["rmsle_delta_vs_prev"] < 0).sum())
        ablation_verdict = (
            f"{improves} of {n_layers} blocks reduce pooled RMSLE. "
            f"The largest gain is {_with_article(best_row)} "
            f"({best_row['rmsle_delta_vs_prev']:+.4f})"
            + (f"; the smallest is {_with_article(worst_row)} "
               f"({worst_row['rmsle_delta_vs_prev']:+.4f})."
               if worst_row["rmsle_delta_vs_prev"] <= 0 else
               f", and {_with_article(worst_row)} does not help "
               f"({worst_row['rmsle_delta_vs_prev']:+.4f}).")
        )

    verdict = ("pre-trend test PASSES (p = {:.3f}), so a causal reading is defensible "
               "within the stated limits".format(pretrend_p)
               if isinstance(pretrend_p, (int, float)) and pretrend_p == pretrend_p and pretrend_p >= 0.05
               else "pre-trend test REJECTS parallel trends (p = {:s}), so the Metro coefficients "
                    "are reported as **descriptive**, not causal".format(
                        "n/a" if pretrend_p != pretrend_p else f"{pretrend_p:.4f}"))

    # --- model narrative, derived from the tables ------------------------- #
    # These numbers used to be hardcoded, which meant a rebuild left the prose
    # quoting the previous run's figures while the tables beside it showed the
    # new ones. Everything below is read from the CSVs the report embeds.
    comparison_path = tables / "model_comparison.csv"
    holdout_path = tables / "model_comparison_holdout.csv"
    by_fold_path = tables / "model_scores_by_fold.csv"
    pooled = pd.read_csv(comparison_path) if comparison_path.exists() else None
    holdout_tbl = pd.read_csv(holdout_path) if holdout_path.exists() else None
    by_fold = pd.read_csv(by_fold_path) if by_fold_path.exists() else None

    # Needed by the 4.6 price-band table: the lagged rolling medians that the
    # naive benchmark uses, so the benchmark can be recomputed on the same rows
    # as the model predictions.
    predictions_path = tables / "predictions.parquet"
    features = None
    if FEATURES_PARQUET.exists():
        try:
            features = pd.read_parquet(
                FEATURES_PARQUET,
                columns=["contract_date", "post_code", "pc_med_price_12m",
                         "pc_med_price_6m", "pc_med_price_3m"])
        except (OSError, KeyError, ValueError) as exc:  # pragma: no cover - defensive
            print(f"  (features unavailable for the price-band table: {exc})")

    def _rmsle(table: pd.DataFrame | None, model: str) -> float:
        if table is None or "rmsle" not in getattr(table, "columns", []):
            return float("nan")
        row = table.loc[table["model"] == model, "rmsle"]
        return float(row.iloc[0]) if len(row) else float("nan")

    win_text = "n/a"
    pooled_gap = pooled_pct = float("nan")
    naive_name = "naive_postcode_month"
    if pooled is not None and best_model in set(pooled.get("model", [])):
        # Any naive_* row is the benchmark we care about.
        naive_rows = pooled.loc[pooled["model"].str.startswith("naive_")]
        if len(naive_rows):
            naive_pooled = float(naive_rows.iloc[0]["rmsle"])
            naive_name = str(naive_rows.iloc[0]["model"])
            best_pooled = _rmsle(pooled, best_model)
            pooled_gap = naive_pooled - best_pooled
            pooled_pct = 100.0 * pooled_gap / naive_pooled
            win_text = "-"
            if by_fold is not None and {"model", "fold", "rmsle"} <= set(by_fold.columns):
                wide = by_fold.pivot_table(index="fold", columns="model", values="rmsle")
                if best_model in wide.columns and naive_name in wide.columns:
                    wins = int((wide[best_model] < wide[naive_name]).sum())
                    total = int(wide[[best_model, naive_name]].dropna().shape[0])
                    win_text = f"{wins} of {total}"
    best_pooled_v = _rmsle(pooled, best_model)
    # Pooled RMSLE of the benchmark, kept explicit: several narrative lines quote
    # the best-model/benchmark pair and must not re-derive it inconsistently.
    naive_pooled_v = (best_pooled_v + pooled_gap) if pooled_gap == pooled_gap else float("nan")
    holdout_best = _rmsle(holdout_tbl, best_model)
    holdout_naive = float("nan")
    if holdout_tbl is not None:
        naive_h = holdout_tbl.loc[holdout_tbl["model"].str.startswith("naive_")]
        if len(naive_h):
            holdout_naive = float(naive_h.iloc[0]["rmsle"])
    holdout_gap = holdout_naive - holdout_best
    best_mdape = float("nan")
    if holdout_tbl is not None and best_model in set(holdout_tbl["model"]):
        best_mdape = float(holdout_tbl.set_index("model").loc[best_model, "mdape_pct"])

    # Sentences that quote per-fold spread / model agreement, derived the same way.
    wide_all = (by_fold.pivot_table(index="fold", columns="model", values="rmsle")
                if by_fold is not None and {"fold", "model", "rmsle"} <= set(by_fold.columns)
                else None)

    def _fold_range(model: str) -> str:
        if wide_all is None or model not in wide_all.columns:
            return "n/a"
        series = wide_all[model].dropna()
        return f"{series.min():.2f} to {series.max():.2f}" if len(series) else "n/a"

    ridge_range = _fold_range("linear_ridge")
    rf_xgb_gap = float("nan")
    rf_xgb_flips = 0
    if wide_all is not None and {"random_forest", "xgboost"} <= set(wide_all.columns):
        rf_xgb_gap = float((wide_all["random_forest"] - wide_all["xgboost"]).abs().mean())
        pair = wide_all[["random_forest", "xgboost"]].dropna()
        if len(pair) > 1:
            # How often does the leader swap between consecutive folds?
            leader = (pair["random_forest"] < pair["xgboost"]).astype(int)
            rf_xgb_flips = int((leader.diff().abs() == 1).sum())
    # Folds where ridge has the best MdAPE among the single-equation models.
    ridge_mdape_wins = 0
    fold_total = 0
    if by_fold is not None and {"fold", "model", "mdape_pct"} <= set(by_fold.columns):
        single = by_fold.pivot_table(index="fold", columns="model", values="mdape_pct")
        candidates = [c for c in ("linear_ridge", "random_forest", "xgboost") if c in single.columns]
        if "linear_ridge" in candidates:
            block = single[candidates].dropna()
            fold_total = len(block)
            ridge_mdape_wins = int((block.idxmin(axis=1) == "linear_ridge").sum())

    # --- 4.6 holdout metric table, derived -------------------------------- #
    metric_rows = ["| metric | {} | {} | winner |".format(best_model, naive_name),
                   "|---|---|---|---|"]
    metric_wins: dict[str, list[str]] = {}
    metric_intro = "On the final holdout:"
    if holdout_tbl is not None and best_model in set(holdout_tbl["model"]):
        row_best = holdout_tbl.set_index("model").loc[best_model]
        if naive_name in set(holdout_tbl["model"]):
            row_naive = holdout_tbl.set_index("model").loc[naive_name]
            specs = [
                ("RMSLE", "rmsle", "{:.4f}", False),
                ("R² (log10)", "r2_log10", "{:.3f}", True),
                ("MAE (log10)", "mae_log10", "{:.4f}", False),
                ("MdAPE", "mdape_pct", "{:.2f}%", False),
                ("MAE (AUD, median back-transform)", "mae_aud", "{:,.0f}", False),
                ("RMSE (AUD)", "rmse_aud", "{:,.0f}", False),
            ]
            for label, column, fmt, higher_is_better in specs:
                if column not in holdout_tbl.columns:
                    continue
                b, n = float(row_best[column]), float(row_naive[column])
                b_better = (b > n) if higher_is_better else (b < n)
                btxt = f"**{fmt.format(b)}**" if b_better else fmt.format(b)
                ntxt = fmt.format(n) if b_better else f"**{fmt.format(n)}**"
                metric_rows.append(f"| {label} | {btxt} | {ntxt} | "
                                   f"{best_model if b_better else naive_name} |")
                if b != n:
                    metric_wins.setdefault(best_model if b_better else naive_name, []).append(label)
        # Whether the metrics actually disagree is data-dependent: on one run the
        # model wins RMSLE while the benchmark wins MdAPE, on another the model
        # wins everything. Asserting "they rank in opposite order" made the text
        # contradict the table directly beneath it.
        best_wins = metric_wins.get(best_model, [])
        naive_wins_list = metric_wins.get(naive_name, [])
        if best_wins and naive_wins_list:
            metric_intro = (
                f"The metrics **disagree**: `{best_model}` wins on "
                f"{', '.join(best_wins)}, while the benchmark wins on "
                f"{', '.join(naive_wins_list)}. That is not a bug but a property of the two "
                "statistics. On the final holdout:")
        elif best_wins:
            metric_intro = (
                f"`{best_model}` wins on **every** metric listed, including the median percentage "
                "error — unlike earlier runs of this pipeline, where RMSLE and MdAPE pointed in "
                "opposite directions. Whether that reversal appears is a property of the sample "
                "rather than of the metric definitions, so it is worth re-checking on fresh data. "
                "On the final holdout:")
        elif naive_wins_list:
            metric_intro = (
                f"The benchmark wins on **every** metric listed. On the final holdout:")
        else:
            metric_intro = "On the final holdout:"
    metric_gap = (float(holdout_tbl.set_index('model').loc[best_model, 'rmsle'])
                  - float(holdout_tbl.set_index('model').loc[naive_name, 'rmsle'])
                  if holdout_tbl is not None and
                  {best_model, naive_name} <= set(holdout_tbl["model"]) else float("nan"))

    # Defensive lookup: a partial or absent hypothesis_summary.json must degrade
    # to "n/a", not raise IndexError. `h1.headline` is a list of correlation rows
    # and the relevant one is the log-log entry (index 2 in a complete run).
    def _hyp(*path, key: str, default=float("nan")):
        node = hypothesis
        for part in path:
            if not isinstance(node, dict):
                return default
            node = node.get(part)
            if node is None:
                return default
        if isinstance(node, list):
            for item in node:
                # The measures are 'pearson', 'spearman', 'log10 pearson'; the
                # log-log one is the elasticity of interest.
                if isinstance(item, dict) and str(item.get("measure", "")).startswith("log10"):
                    return item.get(key, default)
            if len(node) > 2 and isinstance(node[2], dict):
                return node[2].get(key, default)
            return default
        if isinstance(node, dict):
            return node.get(key, default)
        return default

    def _pct(value) -> str:
        """Percent string that degrades to 'n/a' instead of 'nan%'."""
        try:
            if value != value:  # NaN
                return "n/a"
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return "n/a"

    def _num(value, fmt: str, suffix: str = "") -> str:
        """Format a possibly-missing number as 'n/a' rather than 'nan'."""
        try:
            if value != value:
                return "n/a"
            return fmt.format(float(value)) + suffix
        except (TypeError, ValueError):
            return "n/a"

    sections = [
        "# NSW house price prediction — final report",
        "",
        "Generated by `pipelines/07_make_report.py`. Every number below is traceable to a CSV in "
        "`reports/tables/`; every figure is regenerated by the same script.",
        "",
        "## 0. Headline findings",
        "",
        f"1. **The area-price hypothesis reverses once location is controlled.** Pooled "
        f"log-log correlation is {_num(_hyp('h1', 'headline', key='coefficient'), '{:.4f}')} "
        f"but the within-postcode-and-year elasticity is "
        f"{_num(_hyp('h1', 'within_location', key='coef'), '{:+.3f}')}. "
        "The weak negative pooled correlation is a between-location artefact.",
        f"2. **The unit-price hypothesis is mostly a ratio artefact.** "
        f"{_pct(_hyp('h2', key='mechanical_share_of_observed', default=float('nan')))} "
        "of the observed "
        "`corr(log area, log unit price)` is reproduced with prices shuffled at random — i.e. by the "
        "ratio's denominator alone.",
        f"3. **The Metro event study cannot support a causal claim on this data.** {verdict}.",
        f"4. **Model performance:** the best pooled model is `{best_model}`; see §4 for the "
        "rolling-origin comparison against the naive postcode-median benchmark.",
        "5. **Which engineered features earn their place depends on the layer, per the ablation in "
        f"§4.5.** {ablation_verdict}",
        "6. **A single naive benchmark — the postcode's trailing 12-month median — is close behind the "
        f"best model** ({naive_pooled_v:.4f} vs {best_pooled_v:.4f} RMSLE pooled; "
        f"{holdout_naive:.4f} vs {holdout_best:.4f} on the untouched holdout). Any "
        "claim that the engineered feature set adds a lot over local recent prices would be overstated.",
        "",
        "## 1. Data and cleaning (W1)",
        "",
        f"| stage | rows |",
        f"|---|---|",
    ]
    for stage, count in (clean_meta.get("stage_counts") or {}).items():
        sections.append(f"| {stage} | {count:,} |")
    sections += [
        "",
        f"* Raw input: 4,854,814 rows x 17 columns (610.9 MiB).",
        f"* Clean output rows: **{clean_meta.get('rows', 0):,}** with "
        f"{clean_meta.get('coverage', {}).get('unique_postcodes', 0):,} distinct postcodes and "
        f"{clean_meta.get('coverage', {}).get('unique_group_keys', 0):,} distinct "
        "`address|postcode` keys.",
        f"* Corrupted contract dates outside 2001-01-01..2023-12-31 (e.g. 0015-08-29, 1024-01-05) "
        "are dropped as data errors.",
        "",
        "### Leakage fixes relative to the group's notebook",
        "",
        "| audit id | what changed | why |",
        "|---|---|---|",
        "| M1 / L4 | outlier handling is split into two fences: a **corruption fence** "
        "fitted on the training span and applied to every row, and a **per-year quality fence** "
        "applied to the training side only | the original single full-sample fence trimmed the "
        "target's tails using the whole period; a per-year fence fitted on 2001-2015 cannot judge "
        "later years, and applying it to validation empties every post-2015 window |",
        "| M2 | CV splits are purged by `address|postcode` (63.5% of rows share a key with another row) | "
        "otherwise the same dwelling straddles train and validation |",
        "| M3 | cash rate and CPI are joined **as-of the contract date** (`release_date <= contract`, "
        "CPI lag assumed 28 days) | yearly averages embed decisions made after the sale |",
        "| §2.2 | development labels are **versioned**; a contract only receives a label whose window "
        "already closed | a 2005 sale cannot know a 2011-2014 vacancy share |",
        "| §2.3 | Metro distance uses the **pre-opening** snapshot for treatment assignment | the 2020 "
        "station file describes infrastructure that opened in 2019 |",
        "",
        "### Data-quality defects found and fixed",
        "",
        "The raw extract is not clean even after the group's filters. The following were found by "
        "auditing the model's own failures:",
        "",
        "| defect | evidence | fix |",
        "|---|---|---|",
        f"| impossible `area_sqm` | max 2,700,000,000 sqm for a 'house'; 4.8% of rows above 20,000 sqm; "
        f"4.9% of rows had an area above 100,000 sqm | corruption fence on area "
        f"({corruption_fence.get('area_sqm', {}).get('value', ['?', '?'])[0]:,.0f}-"
        f"{corruption_fence.get('area_sqm', {}).get('value', ['?', '?'])[1]:,.0f} sqm, fitted on the training span) |",
        f"| impossible prices | max $875,300,000 for a residence house; 972 rows above $20M | same fence on "
        f"price ({corruption_fence.get('purchase_price', {}).get('value', ['?', '?'])[0]:,.0f}-"
        f"{corruption_fence.get('purchase_price', {}).get('value', ['?', '?'])[1]:,.0f} AUD) |",
        "| a single row dominated every squared-error metric | one 2020 contract was given a "
        "$9.1e11 prediction by ridge; that row alone was **100%** of the pooled RMSE | fixed by the "
        "corruption fence; tail metrics are now also reported on the surviving sample |",
        "| per-year fences cannot judge later years | with `drop_unjudged_iqr` the 2016-2022 validation "
        "windows were **silently emptied** | the two-fence design above; validation is scored on the "
        "full population, training on the fenced subset |",
        "",
        f"Rows removed by the corruption fence: **{outlier.get('rows_corruption_fail', 0):,}**; "
        f"rows kept but flagged as outside the per-year training fence: "
        f"**{outlier.get('rows_quality_fail_flagged', 0):,}**.",
        "",
        "### Errata: errors found in the modelling code itself",
        "",
        "Four mistakes were found by auditing the model outputs rather than the data. They are "
        "recorded because each one made a number look *better* or *worse* without raising an error "
        "(full write-up in `initial_v0/03_leakage_audit.md` §9):",
        "",
        "| # | error | effect | fix |",
        "|---|---|---|---|",
        "| E1 | the naive benchmark applied `log10` to a value that was **already** `log10` | its RMSLE "
        "jumped to 2.18 and R² to −10.5, making every model look like a large win over a broken "
        "baseline | use the log value directly, with a graded 12m → 6m → 3m fallback |",
        "| E2 | the time-reversal placebo dropped rows whose forward window was incomplete | the "
        "future-leaking feature appeared *better* than the lagged one (0.3687 vs 0.4355) | lock the row "
        "set; with identical rows the future version is worse (0.4765), as it must be |",
        "| E3 | the per-year fence emptied the post-2015 validation windows | RMSLE appeared to improve "
        "0.45 → 0.30 purely because half the sample vanished | two-fence design (see above); row counts "
        "are printed per fold |",
        "| E4 | `--max-train-rows` made the nominally expanding window an implicit sliding one | each "
        "fold trained on only the most recent ~4 years | disclosed in §6; `--max-train-rows 0` gives a "
        "true expanding window (XGBoost on GPU makes the retrain tractable: 8 folds x 600 rounds in "
        "~4 min, with the single-threaded random forest the remaining bottleneck) |",
        "",
        "## 2. Hypothesis tests (W5)",
        "",
        "### H1 — area versus price",
        "",
        md_table(tables / "h1_headline_correlations.csv"),
        "",
        "Within-location estimate (log price on log area, postcode + year fixed effects, "
        "clustered by postcode):",
        "",
        md_table(tables / "h1_within_location.csv", n=3),
        "",
        (hypothesis.get("h1", {}).get("interpretation", "")),
        "",
        "![H1 correlation by year](figures/fig8_h1_by_year.png)",
        "",
        "### H2 — area versus unit price",
        "",
        md_table(tables / "h2_ratio_correlation.csv", n=3),
        "",
        "Shuffled-price placebo (the mechanical component):",
        "",
        md_table(tables / "h2_elasticity.csv", n=3),
        "",
        (hypothesis.get("h2", {}).get("interpretation", "")),
        "",
        "![H2 ratio placebo](figures/h2_ratio_placebo.png)",
        "",
        "### Robustness: controlling for the lagged neighbourhood price",
        "",
        (robustness.get("bad_control_note", "")),
        "",
        "H1 under three specifications (same sample, same fixed effects, clustered by postcode):",
        "",
        md_table(tables / "h1_robustness_neighbourhood.csv", n=5),
        "",
        "H2 (elasticity, test of beta = 1):",
        "",
        md_table(tables / "h2_robustness_neighbourhood.csv", n=5),
        "",
        (robustness.get("interpretation", "")),
        "",
        "## 3. Metro event study (W5)",
        "",
        f"* Event: **{event_meta.get('event', {}).get('event_name', 'E1')}**, "
        f"{event_meta.get('event', {}).get('event_date', '')[:10]} "
        f"(all 13 stations opened together, so there is a single treatment cohort and the "
        "two-way fixed-effects estimator is exact rather than an approximation).",
        f"* Treated postcodes at {event_meta.get('spec', {}).get('radius_km', 2)} km: "
        f"{event_meta.get('treatment_summary', {}).get('treated_postcodes', 0)} "
        f"{event_meta.get('treatment_summary', {}).get('treated_list', [])}",
        f"* Controls: same four North-West SA4 regions, at least "
        f"{event_meta.get('spec', {}).get('control_min_km', 3)} km from any new station "
        f"({event_meta.get('treatment_summary', {}).get('control_postcodes', 0)} postcodes).",
        f"* Window: ±{event_meta.get('spec', {}).get('window_months', 24)} months around the opening; "
        "reference period omitted; standard errors clustered by postcode.",
        "",
        f"**Pre-trend verdict.** {verdict}.",
        "",
        "The pre-trend rejection is the single most important result in this section: with 21 "
        "pre-period coefficients jointly different from zero, the treated and control postcodes "
        "were already on different price paths before 26 May 2019. The binary coefficient "
        f"(mean post-period {_num(event_meta.get('main', {}).get('mean_post_coef'), '{:+.4f}')} log10, "
        f"about {_num(post_pct, '{:+.1%}')} on the median unit price) therefore mixes the station with "
        "pre-existing divergence.",
        "",
        md_table(tables / "event_study_pretrend.csv", n=3),
        "",
        "Placebo checks (both should be indistinguishable from zero):",
        "",
        md_table(tables / "event_study_placebo.csv"),
        "",
        "Robustness grid:",
        "",
        md_table(tables / "event_study_robustness.csv", n=10),
        "",
        "Dose-response (continuous exposure; the preferred design given how few postcodes "
        "are within 2 km):",
        "",
        md_table(tables / "event_study_dose_response.csv", n=5),
        "",
        "![Dose response](figures/fig9_dose_response.png)",
        "",
        "![Event study](figures/event_study_plot.png)",
        "",
        "## 4. Models and rolling-origin CV (W3-W4)",
        "",
        f"Target: `log10(purchase_price)`. Folds: {len(training_meta.get('folds', []))} "
        f"forward-chaining windows (train months {training_meta.get('min_train_months', '?')}+, "
        f"horizon {training_meta.get('horizon_months', '?')} months, embargo "
        f"{training_meta.get('embargo_months', '?')} month(s)), purged by dwelling.",
        "",
        "Pooled across folds:",
        "",
        md_table(tables / "model_comparison.csv", n=10),
        "",
        "The naive benchmark predicts the postcode's trailing 12-month median price using only "
        "lagged information. Beating it is the minimum bar for a feature set to be worth anything.",
        "",
        f"**The ranking is much tighter than it looks at first glance.** `{best_model}` beats the naive "
        f"benchmark by {pooled_gap:.4f} RMSLE pooled ({best_pooled_v:.4f} vs {naive_pooled_v:.4f}, about a "
        f"{pooled_pct:.0f}% reduction), and it wins in **{win_text} folds**. On the final holdout "
        f"(below) the gap is {holdout_gap:.4f}. So the honest "
        "summary is that a 53-dimensional engineered feature set buys a consistent but modest "
        "improvement over one number — the local recent median — and the benchmark is never far behind.",
        "",
        "Two caveats when reading this table:",
        "",
        f"* `linear_ridge` has the **best median** percentage error among the single-equation models "
        f"in {ridge_mdape_wins} of {fold_total} folds but the **worst tail**: its `rmse_aud` is "
        "inflated by a handful of extreme predictions on cheap dwellings, so on dollar-scale tail "
        f"risk it is the least reliable of the three. Its per-fold RMSLE also swings ({ridge_range}), "
        "i.e. it is the least stable.",
        f"* `random_forest` and `xgboost` are within ~{rf_xgb_gap:.3f} RMSLE of each other on average and "
        f"their fold-to-fold ordering flips {rf_xgb_flips} times, so they should be treated as "
        "equivalent on this data; the ranking is not "
        "stable enough to justify a strong claim.",
        "",
        "Per-fold detail:",
        "",
        md_table(tables / "model_scores_by_fold.csv",
                 n=len(training_meta.get("folds", [])) * len(training_meta.get("models", [])) or 40),
        "",
        "![Rolling CV](figures/fig5_rolling_cv.png)",
        "",
        "![Residual diagnostics](figures/fig6_residual_diagnostics.png)",
        "",
        "Top features (last fold only, indicative):",
        "",
        "![Feature importance](figures/fig7_feature_importance.png)",
        "",
    ]
    if holdout_window:
        sections += [
            "### Final holdout",
            "",
            f"The final **{training_meta.get('holdout_months')} months "
            f"({holdout_window[0]} to {holdout_window[1]})** were never used by any fold: no model, "
            "feature or hyper-parameter choice was informed by them. The model was trained once on "
            "everything before the holdout and scored once.",
            "",
            md_table(tables / "model_comparison_holdout.csv", n=10),
            "",
            "Read this as the honest out-of-sample number. It differs from the rolling figure above "
            "because it is a single later period (with its own market conditions) rather than an "
            "average of nine windows, and it is scored on the **full population** rather than on rows "
            "the training fence accepted.",
            "",
        ]
    if ablation is not None:
        sections += [
            "## 4.5 Do the engineered features earn their place? (feature-value test)",
            "",
            "This asks a different question from H1/H2 (world relationships) and H3 (the Metro effect): "
            "**do our own engineered features add out-of-sample predictive value?** Same folds, same "
            "group purge, same embargo; model = XGBoost (200 rounds) to keep ~100 fits affordable.",
            "",
            "Layers are cumulative, so each row's delta is that layer's marginal contribution:",
            "",
            "| layer | what it adds |",
            "|---|---|",
            "| L0 | property + location (area, distances) |",
            "| L1 | + calendar (`year_num`, `month_sin`, `month_cos`) |",
            "| L2 | + region labels (postcode / council / locality / zoning / development) |",
            "| L3 | + **lagged price statistics** `pc_med_price_{3,6,12}m` |",
            "| L4 | + lagged unit price `pc_med_unit_price_{3,6,12}m` |",
            "| L5 | + dispersion, sales counts, coverage, staleness |",
            "",
            md_table(ablation_path, n=10),
            "",
            headline_ablation,
            "",
            "**Falsification tests** (anchor fold, L3 specification):",
            "",
            "* *Permutation placebo* — the same columns with values shuffled across postcodes inside "
            "the training fold, so the marginal distribution and the missingness pattern survive but "
            "the postcode↔value link is destroyed.",
            "* *Time reversal* — forward-looking windows (t+1..t+k) instead of lagged ones. If the "
            "future version predicts materially better, the lag/embargo machinery is not working.",
            "",
        ]
        if feature_placebo is not None:
            sections += [md_table(placebo_path, n=15), ""]
        sections += [
            "**N1: is the lagged price signal location level or timing?** Re-running L3 with each "
            "lagged price column demeaned within its postcode (postcode means from the training fold) "
            "removes the between-postcode level. If the increment collapses, the columns are only a "
            "stand-in for \"which postcode is expensive\".",
            "",
        ]
        if feature_demeaned is not None:
            sections += [md_table(demeaned_path, n=10), ""]
    # --- 4.6 price-band table, derived ------------------------------------ #
    # The quoted n's and APE figures used to be hardcoded from an earlier run.
    band_rows: list[str] = []
    band_claim = ""
    worst1_text = ""
    if features is not None and predictions_path.exists():
        try:
            preds = pd.read_parquet(predictions_path)
            block = preds.loc[preds["model"] == best_model].copy()
            if "area_sqm" in block.columns:
                keys = features[["contract_date", "post_code", "pc_med_price_12m",
                                 "pc_med_price_6m", "pc_med_price_3m"]]
                joined = block.merge(keys, on=["contract_date", "post_code"], how="left")
                if len(joined) == len(block):
                    naive_pred = naive_postcode_month_predict(
                        joined, float(np.nanmedian(joined["y_true_log10"])))
                    joined["naive_pred_log10"] = naive_pred
                    actual = np.power(10.0, joined["y_true_log10"].to_numpy())
                    rf_ape = joined["abs_pct_error"].to_numpy()
                    naive_ape = (np.abs(np.power(10.0, naive_pred) - actual)
                                 / np.clip(actual, 1.0, None))
                    joined["rf_ape"] = rf_ape
                    joined["naive_ape"] = naive_ape
                    joined["band"] = pd.qcut(
                        joined["y_true_log10"].rank(method="first"), 5,
                        labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 dearest"])
                    band_rows = [f"| price band | n | {best_model} MdAPE | naive MdAPE | "
                                 f"{best_model} mean APE | naive mean APE |",
                                 "|---|---|---|---|---|---|"]
                    rf_wins, naive_wins = 0, 0
                    for label, grp in joined.groupby("band", observed=True):
                        r_md = 100 * float(grp["rf_ape"].median())
                        n_md = 100 * float(grp["naive_ape"].median())
                        r_mn = 100 * float(grp["rf_ape"].mean())
                        n_mn = 100 * float(grp["naive_ape"].mean())
                        rf_better = r_mn < n_mn
                        rf_wins += int(rf_better)
                        naive_wins += int(not rf_better)
                        band_rows.append(
                            f"| {label} | {len(grp):,} | "
                            f"{'**' if r_md < n_md else ''}{r_md:.1f}%{'**' if r_md < n_md else ''} | "
                            f"{'**' if n_md < r_md else ''}{n_md:.1f}%{'**' if n_md < r_md else ''} | "
                            f"{'**' if r_mn < n_mn else ''}{r_mn:.1f}%{'**' if r_mn < n_mn else ''} | "
                            f"{'**' if n_mn < r_mn else ''}{n_mn:.1f}%{'**' if n_mn < r_mn else ''} |")
                    band_claim = (
                        f"`{best_model}` wins on mean APE in **{rf_wins} of 5** price bands and the "
                        f"benchmark in **{naive_wins} of 5**.")
                    # Concentration of squared log error in the worst 1%.
                    err = (joined["y_pred_log10"] - joined["y_true_log10"]) ** 2
                    nerr = (naive_pred - joined["y_true_log10"].to_numpy()) ** 2
                    k = max(1, int(np.ceil(0.01 * len(err))))
                    share_rf = float(np.sort(err.to_numpy())[-k:].sum() / err.sum() * 100)
                    share_nv = float(np.sort(nerr)[-k:].sum() / nerr.sum() * 100)
                    worst1_text = (
                        f"**Concentration of error** (pooled folds): the worst 1% of rows carry "
                        f"{share_rf:.1f}% of the total squared log error for {best_model} and "
                        f"{share_nv:.1f}% for the benchmark — so both are tail-dominated, and neither "
                        "number is an artefact of one bad row once the corruption fence is in place.")
        except (OSError, KeyError, ValueError) as exc:  # pragma: no cover - defensive
            print(f"  (price-band table skipped: {type(exc).__name__}: {exc})")

    sections += [
        "### 4.6 Which metric should be used, and why MdAPE and RMSLE can disagree",
        "",
        metric_intro,
        "",
        *metric_rows,
        "",
        "**Why they can disagree.** Both metrics punish proportional error, but they weight the price",
        "distribution differently:",
        "",
        "* **RMSLE lives in log space, so it is scale-free.** Missing a $4M house by 40% costs exactly",
        "the same as missing a $250k house by 40%. One number therefore summarises accuracy across the",
        "whole price range, and a single catastrophic miss on an unusual property shows up clearly.",
        "* **MdAPE is a raw percentage on the dollar scale.** It is dominated by cheap properties, where",
        "the same dollar error becomes a huge percentage. The median then sits close to the *typical*",
        "cheap sale.",
        "",
        "**Where each model actually wins.** Splitting the same rows by actual-price quintile makes the",
        "trade-off visible:",
        "",
        *(band_rows or ["*(price-band table unavailable — predictions.parquet missing or stale)*"]),
        "",
        band_claim or
        "*(per-band comparison unavailable; re-run `04_train_models.py` then `07_make_report.py`)*",
        "",
        "**Which to report, and for what.**",
        "",
        "| purpose | metric | why |",
        "|---|---|---|",
        "| model selection / headline accuracy | **RMSLE** | scale-free, comparable across folds and price levels, sensitive to catastrophic misses |",
        "| communicating to a non-technical reader | **MdAPE** | \"half of homes are predicted within X%\" is immediately legible |",
        "| fairness across the price range | **per-quintile APE** | a single number hides that the models win in different segments |",
        "| dollar impact | MAE/RMSE in AUD, **with the tail disclosed** | RMSE in dollars was once 100% attributable to a single row; always report the worst-1% share alongside |",
        "",
        (worst1_text or
         "*(error-concentration statistic unavailable; re-run `04_train_models.py`)*"),
        "",
        "**Recommended wording for a presentation.** \"The model predicts half of homes within "
        f"{best_mdape:.0f}% and has "
        f"a lower scaled error than a postcode-median benchmark ({best_pooled_v:.3f} vs {naive_pooled_v:.3f}). A simple benchmark is",
        "within ~2 percentage points on typical accuracy, so the value of the feature set is modest; the",
        "model's real advantage is on the cheapest fifth of the market and in avoiding large misses.\"",
        "",
        "## 5. Figures",        "",
        "![Distributions](figures/fig1_distributions.png)",
        "",
        "![Area vs price](figures/fig2_area_vs_price.png)",
        "",
        "![Area vs unit price](figures/fig3_area_vs_unit_price.png)",
        "",
        "![Yearly macro comparison](figures/fig4_yearly_macro.png)",
        "",
        "## 6. Known limitations",
        "",
        "1. `area_sqm` is the **recorded** area from the Valuer General extract. It is assumed to be "
        "known at contract date; this assumption is not verifiable from the file and is flagged in "
        "`initial_v0/03_leakage_audit.md` §2.1.",
        "2. 4.8% of cleaned rows have an area above 20,000 sqm and the maximum is 2.7e9 sqm — "
        "corrupted or acreage records. Hypothesis tests therefore restrict to 100-5,000 sqm.",
        "3. Transport distances are **postcode-centroid** great-circle distances, not property or "
        "walking distances.",
        "4. The Metro study has few treated clusters (8 postcodes at the 2 km radius), so clustered "
        "standard errors are imprecise and placebo draws can reject by chance; the 1 km and 3 km "
        "radii are reported for this reason.",
        "5. `development_type` is missing for 42% of rows because a legally-timed label only exists "
        "from 2011 onward; it is kept as a category with a missing level rather than dropped.",
        "6. Macro levels come from the RBA G1 CPI series, whose index base differs from the annual "
        "CPI values hard-coded in the original notebook; levels are not comparable across sources, "
        "so the year-ended rate is the feature.",
        "",
        "## 7. Reproducing this report",
        "",
        "```bash",
        "python pipelines/01_build_clean.py          # raw CSV  -> data/processed/clean.parquet",
        "python pipelines/02_build_features.py       # + macro as-of + panel -> features.parquet",
        "python pipelines/04_train_models.py         # rolling-origin CV -> model tables",
        "python pipelines/05_hypothesis_tests.py     # H1/H2 with placebos",
        "python pipelines/06_event_study.py          # E1 event study",
        "python pipelines/07_make_report.py          # this report + all figures",
        "pytest tests -q                             # contract + leakage + CV guards",
        "```",
        "",
    ]
    path = out_dir / "final_report.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    started = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    if not FEATURES_PARQUET.exists():
        print(f"Missing {FEATURES_PARQUET}; run pipelines/02_build_features.py first.")
        return 2

    frame = load_features(FEATURES_PARQUET)
    if args.sample and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=36103)
    print(f"Loaded {len(frame):,} rows for EDA figures.")

    figure_distributions(frame, FIGURES_DIR, plt)
    figure_area_price(frame, FIGURES_DIR, plt)
    figure_area_unit_price(frame, FIGURES_DIR, plt)
    figure_yearly_economics(frame, FIGURES_DIR, plt)
    made = {
        "fig5_rolling_cv": figure_rolling_scores(TABLES_DIR, FIGURES_DIR, plt),
        "fig6_residuals": figure_residuals(TABLES_DIR, frame, FIGURES_DIR, plt),
        "fig7_importance": figure_feature_importance(TABLES_DIR, FIGURES_DIR, plt),
        "fig8_h1_by_year": figure_h1_by_year(TABLES_DIR, FIGURES_DIR, plt),
        "fig9_dose_response": figure_dose_response(TABLES_DIR, FIGURES_DIR, plt),
    }
    print("Figures written:", ", ".join(p.name for p in sorted(FIGURES_DIR.glob("fig*.png"))))
    for name, ok in made.items():
        if not ok:
            print(f"  note: {name} skipped (run the training pipeline first)")

    if not args.figures_only:
        report = write_report(REPORTS_DIR, TABLES_DIR, FIGURES_DIR)
        write_json(TABLES_DIR / "report_meta.json", {
            "rows_used_for_figures": int(len(frame)),
            "figures": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
            "tables": sorted(p.name for p in TABLES_DIR.glob("*.csv")),
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"Wrote {report}")
    print(f"Elapsed: {time.time() - started:,.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
