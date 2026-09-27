"""Report which downstream artifacts are older than the feature frame.

The pipeline is expensive enough that stages get re-run piecemeal, and a stale
table does not announce itself: `07_make_report.py` will happily build a report
whose prose is fresh and whose numbers are not. This prints, for every artifact
stage, whether it is newer than `data/processed/features.parquet` and (where the
stage records it) which `rows` count it was computed from.

Usage
-----
    python pipelines/00_check_staleness.py
    python pipelines/00_check_staleness.py --exit-code   # non-zero if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import OUT_CLEAN_PARQUET, REPORTS_DIR
from src.features.pipeline import OUT_FEATURES_PARQUET

TABLES = REPORTS_DIR / "tables"

# stage label -> (producer script, artifact paths, the input that stage consumes)
STAGES: list[tuple[str, str, list[str], str]] = [
    ("01 clean", "01_build_clean.py", [str(OUT_CLEAN_PARQUET)], "raw CSV"),
    ("02 features", "02_build_features.py", [str(OUT_FEATURES_PARQUET)],
     str(OUT_CLEAN_PARQUET)),
    ("04 train", "04_train_models.py", [
        "model_comparison.csv", "model_scores_by_fold.csv",
        "model_comparison_holdout.csv", "predictions.parquet",
        "training_meta.json", "feature_importance.csv", "model_scores_grouped.csv"],
     str(OUT_FEATURES_PARQUET)),
    ("05 hypotheses", "05_hypothesis_tests.py", [
        "hypothesis_summary.json", "h1_headline_correlations.csv",
        "h2_elasticity.csv", "h2_placebo_ratio.csv"], str(OUT_FEATURES_PARQUET)),
    ("06 event study", "06_event_study.py", [
        "event_study_meta.json", "event_study_coefficients.csv",
        "event_study_pretrend.csv", "event_study_dose_response.csv"],
     str(OUT_FEATURES_PARQUET)),
    ("08 feature value", "08_feature_value_test.py", [
        "feature_ablation.csv", "feature_ablation_by_fold.csv",
        "feature_value_placebo.csv", "feature_value_demeaned.csv"],
     str(OUT_FEATURES_PARQUET)),
    ("07 report", "07_make_report.py", [
        "report_meta.json", str(REPORTS_DIR / "final_report.md")],
     str(OUT_FEATURES_PARQUET)),
]

# stage -> (file, key) recording how many rows that stage actually consumed,
# when it records one at all.
ROW_RECORD: dict[str, tuple[str, str]] = {
    "04 train": ("training_meta.json", "rows"),
    "06 event study": ("event_study_meta.json", "rows"),
}


def newest(paths: list[Path]) -> tuple[Path | None, float]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None, 0.0
    best = max(existing, key=lambda p: p.stat().st_mtime)
    return best, best.stat().st_mtime


def rows_recorded(stage: str) -> int | None:
    """The row count the stage recorded for its own input, if any."""
    if stage not in ROW_RECORD:
        return None
    name, key = ROW_RECORD[stage]
    path = TABLES / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-code", action="store_true",
                        help="exit 1 when any stage is stale or missing")
    args = parser.parse_args()

    if not OUT_FEATURES_PARQUET.exists():
        print(f"Missing {OUT_FEATURES_PARQUET}; nothing to compare against.")
        return 2
    feature_mtime = OUT_FEATURES_PARQUET.stat().st_mtime
    frame_rows = len(pd.read_parquet(OUT_FEATURES_PARQUET, columns=["post_code"]))
    print(f"features.parquet: {frame_rows:,} rows, "
          f"written {pd.Timestamp(feature_mtime, unit='s'):%Y-%m-%d %H:%M:%S}\n")

    stale: list[str] = []
    for stage, script, names, input_path in STAGES:
        paths = [Path(n) if str(n).startswith(str(REPORTS_DIR)) or ":" in str(n)
                 else TABLES / n for n in names]
        missing = [p.name for p in paths if not p.exists()]
        last, mtime = newest(paths)
        source = Path(input_path)
        source_mtime = source.stat().st_mtime if source.exists() else None
        if not last:
            status = "MISSING"
        elif source_mtime is None:
            # A source stage (its input is the raw CSV): it can only be checked
            # for existence, not for freshness.
            status = "ok"
        elif mtime >= source_mtime:
            status = "ok"
        else:
            status = "STALE"
        if status != "ok":
            stale.append(stage)
        when = f"{pd.Timestamp(mtime, unit='s'):%Y-%m-%d %H:%M}" if last else "-"
        rec = rows_recorded(stage)
        rec_txt = "" if rec is None else f"  (consumed {rec:,} rows)"
        note = f"  missing: {', '.join(missing)}" if missing else ""
        print(f"{status:8s} {stage:16s} {when}  vs {source.name:16s} "
              f"-> {script}{rec_txt}{note}")

    print()
    consistency = check_consistency()
    print()

    if stale:
        order = ["04_train_models.py", "05_hypothesis_tests.py",
                 "06_event_study.py", "08_feature_value_test.py", "07_make_report.py"]
        print("Re-run with:")
        for stage in stale:
            for candidate in order:
                if candidate.startswith(stage.split()[0]):
                    print(f"  .\\.venv\\Scripts\\python.exe pipelines\\{candidate}")
        return 1 if args.exit_code else 0
    if consistency:
        print("All stages are at least as new as features.parquet.")
        return 0
    return 1 if args.exit_code else 0


def check_consistency() -> bool:
    """Cross-table checks: do the derived tables agree with each other?

    A stage can be freshly written and still be internally inconsistent -- the
    grouped-diagnostics bug (a many-to-many merge on
    ``(contract_date, post_code)``) inflated every ``n`` by ~2.5x while all the
    timestamps looked fine. These checks are the machine-readable form of the
    objective's "tables are mutually consistent" clause.
    """
    ok = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        if not passed:
            ok = False
        print(f"{'ok  ' if passed else 'FAIL'}     {name:36s} {detail}")

    preds_path = TABLES / "predictions.parquet"
    grouped_path = TABLES / "model_scores_grouped.csv"
    comparison_path = TABLES / "model_comparison.csv"
    holdout_path = TABLES / "model_comparison_holdout.csv"
    meta_path = TABLES / "training_meta.json"

    # 1. training_meta rows must equal the frame the model was trained on.
    if meta_path.exists() and OUT_FEATURES_PARQUET.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = meta.get("rows")
        actual = len(pd.read_parquet(OUT_FEATURES_PARQUET, columns=["post_code"]))
        report("training_meta.rows == features rows", recorded == actual,
               f"{recorded:,} vs {actual:,}")
        report("training_meta.max_train_rows == 0",
               meta.get("max_train_rows") == 0,
               f"{meta.get('max_train_rows')} (0 = true expanding window)")

    # 2. The grouped diagnostics must have exactly one row per prediction.
    if preds_path.exists() and grouped_path.exists() and comparison_path.exists():
        preds = pd.read_parquet(preds_path, columns=["model"])
        best = pd.read_csv(comparison_path).iloc[0]["model"]
        true_n = int((preds["model"] == best).sum())
        grouped = pd.read_csv(grouped_path)
        for column in ("year", "area_band"):
            block = grouped.loc[grouped["grouped_by"] == column, "n"]
            if len(block):
                report(f"grouped[{column}].n sums to predictions", int(block.sum()) == true_n,
                       f"{int(block.sum()):,} vs {true_n:,}")
        if "area_sqm" not in pd.read_parquet(preds_path).columns:
            report("predictions.parquet carries area_sqm", False,
                   "re-run 04_train_models.py (diagnostics would fan out)")
        else:
            report("predictions.parquet carries area_sqm", True, "present")

    # 3. Every model in the comparison tables must have predictions.
    if preds_path.exists() and comparison_path.exists() and holdout_path.exists():
        have = set(pd.read_parquet(preds_path, columns=["model"])["model"].unique())
        pooled = set(pd.read_csv(comparison_path)["model"])
        hold = set(pd.read_csv(holdout_path)["model"])
        report("pooled models have predictions", pooled <= have,
               f"missing: {sorted(pooled - have) or 'none'}")
        report("holdout models == pooled models", hold == pooled,
               f"only in one: {sorted(hold ^ pooled) or 'none'}")

    # 4. Fold rows per model must be identical (a model silently dropped a fold).
    folds_path = TABLES / "model_scores_by_fold.csv"
    if folds_path.exists():
        by_fold = pd.read_csv(folds_path)
        counts = by_fold.groupby("model")["fold"].nunique()
        report("same fold count for every model", counts.nunique() == 1,
               f"{counts.to_dict()}")

    # 5. The report must be at least as new as the *data* tables it embeds.
    #
    # Excluded deliberately: `report_meta.json` and `report_cn_meta.json` are
    # written BY the report, and `training_meta.json` / `event_study_meta.json`
    # carry a wall-clock `elapsed_seconds` that changes on every re-run. All four
    # are re-written (and therefore re-timestamped) on each invocation, including
    # read-only ones, so comparing them against the report is a comparison
    # against noise: reverting them with `git checkout` alone is enough to make
    # the report look stale. What matters is that the report post-dates the
    # tables whose *contents* it reproduces.
    report_path = REPORTS_DIR / "final_report.md"
    if report_path.exists():
        volatile = {
            "report_meta.json", "report_cn_meta.json",
            "training_meta.json", "event_study_meta.json",
        }
        embedded = [p for p in sorted(TABLES.glob("*.csv")) + sorted(TABLES.glob("*.json"))
                    if p.name not in volatile]
        newest_table, mtime = newest(embedded)
        if newest_table is not None:
            report("final_report newer than tables", report_path.stat().st_mtime >= mtime,
                   f"report {pd.Timestamp(report_path.stat().st_mtime, unit='s'):%H:%M} vs "
                   f"{newest_table.name} {pd.Timestamp(mtime, unit='s'):%H:%M}")

    # 6. No hardcoded model numbers may survive in the report prose.
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        if comparison_path.exists():
            best = pd.read_csv(comparison_path).iloc[0]
            best_rmsle = f"{float(best['rmsle']):.4f}"
            report("report quotes the current best RMSLE",
                   best_rmsle in text or f"{float(best['rmsle']):.3f}" in text,
                   f"looking for {best_rmsle}")

    return ok


if __name__ == "__main__":
    raise SystemExit(main())
