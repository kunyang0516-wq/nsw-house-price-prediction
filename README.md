# NSW house price prediction — 36103 AT2

End-to-end analysis of NSW residence-house sales (2001–2023, 1.87M transactions after cleaning):
leakage-safe data preparation, hypothesis tests for the EDA claims, rolling-origin forecasting with
three model families, and an event study of the Sydney Metro Northwest opening.

```
raw CSV ──► clean ──► features ──► rolling CV models ──► report
                 └──► postcode × month panel ──► Metro event study
```

## Quick start

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe pandas numpy scipy matplotlib lxml pyarrow scikit-learn xgboost pytest

python pipelines/01_build_clean.py        # raw 610 MiB CSV -> data/processed/clean.parquet
python pipelines/02_build_features.py     # + monthly RBA macro as-of + panel features
python pipelines/04_train_models.py       # rolling-origin CV, 3 model families + baselines
python pipelines/05_hypothesis_tests.py   # H1/H2 with clustered CIs and a ratio-bias placebo
python pipelines/06_event_study.py        # Metro Northwest event study (E1)
python pipelines/08_feature_value_test.py # layered ablation + placebos
python pipelines/07_make_report.py        # every figure + reports/final_report.md
pytest tests -q                           # data contract, leakage and CV guards
```

Each pipeline is independently runnable and idempotent; outputs land in `data/processed`,
`data/interim` and `reports/`. Raw data is never modified.

## Layout

| path | contents |
|---|---|
| `src/data/` | `load`, `contract` (assertions), `clean`, `postcode_features`, `outliers`, `macro_asof`, `events` |
| `src/features/` | `panel.py` (postcode×month panel + lagged rolling features), `pipeline.py` (assembles the modelling frame) |
| `src/models/` | `features.py` (fold-aware transformer), `registry.py` (models + baselines) |
| `src/validation/` | `time_series_cv.py` (expanding/sliding folds, embargo, group purge) |
| `src/causal/` | `event_study.py` (TWFE event study, dose-response, placebos) |
| `src/evaluation/` | `metrics.py`, `hypothesis_tests.py` |
| `pipelines/` | `01_build_clean` → `07_make_report`, the only entry points |
| `tests/` | 22 guards: data contract, leakage, macro as-of, CV splits |
| `reports/` | `final_report.md`, `figures/`, `tables/` |
| `initial_v0/` | the design documents this implementation follows |
| `attic/` | **not used by the pipeline** — archived raw material, notebook output, team scripts, and the Chinese-language design documents and report (see `attic/README.md`; also gitignored) |

### What stays in the repository root, and why

| file | role |
|---|---|
| `nsw_property_data.csv` | **primary input** (610.9 MiB raw extract) — read by `pipelines/01_build_clean.py` |
| `australian_postcodes.csv` | postcode centroid / locality lookup, read by `01_build_clean.py` |
| `stationentrances2020_v4.csv` | station entrance coordinates, read by `01_build_clean.py` and `src/data/geocode_features.py` |
| `36103_AT2_combined.ipynb` | the group's original notebook — submission evidence |
| `README.md`, `SCAFFOLD.md` | project documentation |

The three data files must stay at the root because `src/utils/config.py` resolves them
there (`RAW_DIR = PROJECT_ROOT`). **Moving them breaks `01_build_clean.py`** — if you want
them under `data/raw/`, change `RAW_DIR` in `config.py` in the same commit.

`.gitignore` is tuned so that a plain `git add .` commits exactly the right set:

| committed | why |
|---|---|
| `src/`, `pipelines/`, `tests/`, `config/`, `notebooks/` | the code |
| `initial_v0/`, `README.md`, `SCAFFOLD.md` | the design documents |
| `reports/final_report*.md`, `reports/figures/*.png` | the deliverables |
| `reports/tables/*.csv` | **the evidence** - every number in the report traces back to one of these |
| `data/external/*.csv`, `data/interim/*.csv` | small pipeline **inputs** (event table, RBA snapshots, postcode lookups) |

| ignored | why |
|---|---|
| `nsw_property_data.csv`, `australian_postcodes.csv`, `stationentrances2020_v4.csv` | 618 MiB of raw input; re-downloadable, and `01_build_clean.py` reads them from the root |
| `data/raw/`, `data/processed/` | large intermediates (`clean.parquet`, `features.parquet`) |
| `*.parquet` | `predictions.parquet` and the event-study panel are multi-megabyte byproducts |
| `attic/` | the archive of unused files |
| `.venv/`, `__pycache__/`, `.pytest_cache/`, `pytest-cache-files-*/`, `.ipynb_checkpoints/` | environment and caches |
| `_*.py`, `*.log` | scratch scripts and run logs |

> **Two `.gitignore` bugs were found and fixed while doing this.** A bare `data/`
> also matches `src/data/` - which silently excluded the **entire data layer**
> (`clean.py`, `contract.py`, `outliers.py`, `macro_asof.py`, `events.py`,
> `geocode_features.py`, `postcode_features.py`) from the repository. The same
> applied to a bare `models/` versus a future `src/models/`. Check with
> `git ls-files -o -i --exclude-standard | grep -v __pycache__` - it should print
> nothing except what you intend to ignore.

**Note on the original notebook.** `36103_AT2_combined.ipynb` is left in place at the repository
root as the group's submission evidence; `notebooks/archived/` is where a frozen copy belongs once
the team is done with the notebook. No implementation logic lives in it any more — everything is in
`src/` and driven from `pipelines/`.

## Leakage controls this project enforces

The starting notebook (three member notebooks merged, `36103_AT2_combined.ipynb`) is sound
descriptive EDA but had five leakage paths that had to be closed. Full audit:
`initial_v0/03_leakage_audit.md`.

| # | issue in the original | fix |
|---|---|---|
| M1 | one log-IQR fence fitted on the whole 2001–2023 sample removes 18.5% of rows and trims the **target's** tails | fences fitted per contract year, **on the training span only**; rows outside the fence are flagged (`iqr_keep`) rather than silently deleted |
| M2 | 63.5% of rows share an `address\|postcode` key with another row, so a dwelling can straddle a split | every CV fold purges validation group keys from training, so leftover repeat sales cannot inflate scores |
| M3 | annual mean cash rate and annual hard-coded CPI attached to every contract | monthly **as-of** join: cash rate from the RBA change log (same-day effect, no lag), CPI on `release_date` with an explicit 28-day publication lag |
| §2.2 | postcode development labels built from 2011–2014 applied to contracts from 2001 | labels are **versioned**; a contract only receives a label whose window has already closed, so pre-2011 rows carry `NA` |
| §2.3 | metro distance computed from a 2020 station snapshot used across 2001–2023 | treatment assignment uses the **pre-opening** distance; static columns remain for EDA only |

Everything that learns from data (imputation medians, target encodings, area bin edges,
standardisation, IQR fences) is fitted **inside each training fold and applied to the validation
fold**, so the validation period never informs the model that predicts it.

## Headline results

* **H1 (area vs price).** Pooled log-log correlation is negative but the within-postcode-and-year
  elasticity is **positive** (~0.27): the pooled negative sign is a between-location artefact.
* **H2 (unit price vs area).** ~85% of the strong negative `corr(log area, log unit price)` is
  reproduced by a shuffled-price placebo — it is the ratio's denominator, not an economic discount.
  The elasticity of price with respect to area is ~0.27, far below 1.
* **Metro event study.** The joint pre-trend test **rejects** parallel trends in every specification,
  so the binary station coefficient cannot be read causally; the distance-gradient specification is
  negative as expected (larger rises closer to a station) but not distinguishable from zero.
* **Forecasting.** Rolling-origin CV (expanding window, 12-month horizon, 1-month embargo) compares
  two naive benchmarks with ridge, random forest and XGBoost on `log10(price)`; see
  `reports/tables/model_comparison.csv` and `reports/final_report.md` §4.

## Known limitations

1. 4.8% of cleaned rows have area > 20,000 sqm and the maximum is 2.7e9 sqm (corrupted or acreage);
   hypothesis tests restrict to 100–5,000 sqm.
2. `area_sqm` is taken to be known at contract date — stated as an assumption, not verified.
3. Distances are postcode-centroid great-circle, not property or walking distances.
4. The Metro study has only 8–12 treated postcodes, so clustered standard errors are imprecise.
5. `development_type` is `NA` for 42% of rows because a legally-timed label only exists from 2011.
6. Macro levels come from the RBA G1 CPI series, whose index base differs from the annual CPI values
   hard-coded in the original notebook; the year-ended rate is used as the feature instead.

## Environment

Python 3.12, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1,
matplotlib 3.11.2, lxml 6.1.3, pyarrow 25.0.1. Tree models run with `n_jobs=1`: joblib's thread
backend needs named pipes, which are unavailable in this environment.
