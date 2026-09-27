# NSW house price prediction — 36103 AT2

End-to-end analysis of NSW residence-house sales (2001–2023, **1,720,833 transactions** after
cleaning): leakage-safe data preparation, hypothesis tests for the EDA claims, rolling-origin
forecasting with three model families, and an event study of the Sydney Metro Northwest opening.

```
raw CSV ──► clean ──► features ──► rolling CV models ──► report
                 └──► postcode × month panel ──► Metro event study
```

## Data — what you must supply, and where the rest comes from

**The raw extract is not in this repository** (610.9 MiB exceeds GitHub's limit). Everything
else the cleaning stage needs is committed, so `01_build_clean.py` runs offline.

| file | where it goes | how to get it |
|---|---|---|
| `nsw_property_data.csv` | **repository root** | **you must supply it** — see below |
| `australian_postcodes.csv` | repository root | optional; auto-downloaded from [matthewproctor/australianpostcodes](https://github.com/matthewproctor/australianpostcodes) if absent |
| `stationentrances2020_v4.csv` | repository root | optional; auto-downloaded from [Transport for NSW Open Data](https://opendata.transport.nsw.gov.au/) if absent |

`data/interim/postcode_development.csv` and `data/interim/postcode_transport_features.csv` are
committed, so the two lookups above can be skipped entirely with `--skip-reference` — that is the
recommended way to reproduce, because it needs no network and cannot drift with an upstream source:

```bash
python pipelines/01_build_clean.py --skip-reference
```

### About `nsw_property_data.csv`

It is the NSW **Valuer General Property Sales Information** bulk extract, filtered to house
residence sales. Its 17 columns are `property_id`, `download_date`, `council_name`, `address`,
`post_code`, `property_type`, `strata_lot_number`, `property_name`, `area_type`, `contract_date`,
`settlement_date`, `zoning`, `nature_of_property`, `primary_purpose`, `legal_description`,
`purchase_price`, `area`.

Note that **it carries no coordinates**: the `dist_cbd` / `dist_train` / `dist_metro` columns are
*derived*, joined from the postcode lookup built by `build_transport_features()`. That lookup
(and the `dist_metro` as-of variant used for treatment assignment) comes from
`australian_postcodes.csv` + `stationentrances2020_v4.csv`.

The exact file used here is:

| property | value |
|---|---|
| size | 610.9 MiB |
| rows | **4,854,814** |
| columns | 17 |
| expected `clean.parquet` rows | **1,720,833** |

**Row counts are only comparable if you hold the same extract.** A different export (different
date range, councils, or filters) will still run end to end, but every downstream number — the
model scores, the hypothesis estimates, the fences — will differ. `01_build_clean.py` prints a
staged row-count report so you can tell immediately which case you are in.

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

To run the whole chain in one process (about 60 minutes), see `pipelines/run_chain.py` and
`pipelines/NOTES.md`; that is also where the `--rf-jobs` / `--xgb-device` speed-ups are explained.

## Layout

| path | contents |
|---|---|
| `src/data/` | `load`, `contract` (assertions), `clean`, `postcode_features`, `outliers`, `macro_asof`, `events` |
| `src/features/` | `panel.py` (postcode×month panel + lagged rolling features), `pipeline.py` (assembles the modelling frame) |
| `src/models/` | `features.py` (fold-aware transformer), `registry.py` (models + baselines), `thread_forest.py` (thread-parallel random forest for shells where joblib's pool is blocked) |
| `src/validation/` | `time_series_cv.py` (expanding/sliding folds, embargo, group purge) |
| `src/causal/` | `event_study.py` (TWFE event study, dose-response, placebos) |
| `src/evaluation/` | `metrics.py`, `hypothesis_tests.py` |
| `pipelines/` | `00_check_staleness` then `01_build_clean` → `07_make_report`; `run_chain.py` runs them in order |
| `tests/` | 55 guards: data contract, leakage, macro as-of, CV splits, de-duplication and grouping, thread-forest equivalence |
| `reports/` | `final_report.md`, `figures/`, `tables/` |
| `initial_v0/` | the design documents this implementation follows |

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
| `nsw_property_data.csv` | 610.9 MiB, **not re-downloadable** — you must supply the same extract (see the Data section) |
| `australian_postcodes.csv`, `stationentrances2020_v4.csv` | 7 MiB; re-downloadable, and `01_build_clean.py` fetches them automatically if absent |
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

* **H1 (area vs price).** Pooled log-log correlation is **−0.1115** but the within-postcode-and-year
  elasticity is **+0.2463**: the pooled negative sign is a between-location artefact (bigger houses
  sit on cheaper land, mostly further out), not evidence that area lowers price.
* **H2 (unit price vs area).** The observed `corr(log area, log unit price)` is **−0.5149**, but a
  shuffled-price placebo reproduces **−0.4368** of it — i.e. **~85%** is the ratio's denominator,
  not an economic discount. The elasticity of price with respect to area is **+0.2463**, far below 1,
  which is the part of the claim that does survive.
* **Metro event study.** The joint pre-trend test **rejects** parallel trends in every specification
  (p ≈ 1.1e-06 on the headline one), so the station coefficients are reported as **descriptive, not
  causal** — treated postcodes were already on different price paths before the 2019 opening.
* **Forecasting.** Rolling-origin CV (expanding window, 12-month horizon, 1-month embargo, purged by
  dwelling) compares two naive benchmarks with ridge, random forest and XGBoost on `log10(price)`.
  The best pooled model is **`xgboost`, RMSLE 0.3182** (holdout **0.3305**), against **0.3622**
  (holdout 0.3718) for the single-number postcode-median benchmark — a consistent but *modest* 12%
  improvement. See `reports/tables/model_comparison.csv` and `reports/final_report.md` §4.

## Known limitations

1. The raw extract contains values that cannot be a house sale (an `area` of 2.7e9 sqm, a 555M AUD
   "residence"). These are removed by the plausibility gate, so `clean.parquet` tops out at
   **1,603 sqm** and no row exceeds 10,000 sqm. The cost is disclosed rather than hidden: the gate
   drops 147,519 of 2,233,883 rows (6.6%), and `01_build_clean.py` reports each rule's count.
2. `area_sqm` is taken to be known at contract date — stated as an assumption, not verified.
3. Distances are postcode-centroid great-circle, not property or walking distances.
4. The Metro study has only 8–12 treated postcodes, so clustered standard errors are imprecise.
5. `development_type` is `NA` for **43%** of rows, because a legally-timed label only exists from
   2011 (the `v2010` window needs 2001–2010 sales to close first).
6. Macro levels come from the RBA G1 CPI series, whose index base differs from the annual CPI values
   hard-coded in the original notebook; the year-ended rate is used as the feature instead.
7. `data/processed/` is not committed, so a fresh clone must run `01` and `02` (about 5 minutes)
   before any modelling stage. `reports/tables/` **is** committed, so every number in the report
   can be audited without re-running anything.

## Environment

Python 3.12, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1,
matplotlib 3.11.2, lxml 6.1.3, pyarrow 25.0.1.

XGBoost runs on CUDA (`--xgb-device cuda`); random forest has no GPU path and is parallelised
across CPU threads (`--rf-jobs N`). sklearn's forest delegates to `joblib`, which builds a
`multiprocessing.pool.ThreadPool` whose queue needs a named pipe — confined Windows shells deny
that with `PermissionError: [WinError 5]`, so the forest would silently run on one core. When
joblib's pool is unavailable, `src/models/thread_forest.py` builds the trees on a
`ThreadPoolExecutor` instead, reusing sklearn's own bootstrap sampling and `_fit`, and
`tests/test_thread_forest.py` asserts bit-identical predictions against `RandomForestRegressor`.
`04_train_models.py` probes joblib at startup and reports which backend it will use.
