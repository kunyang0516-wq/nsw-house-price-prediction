# initial_v0 - design documents

> The plan this implementation follows, written **before** the code, plus the audits written
> **after** things went wrong. Chinese originals are kept in `attic/chinese/initial_v0/`.

## Contents

| file | what it is |
|---|---|
| `00_project_walkthrough.md` | **Read this first.** Folder map, data flow, all 16 cleaning steps with measured row counts, a file-by-file explanation of every module, key numbers, reading order |
| `01_eda_walkthrough.md` | Section-by-section reading of the group's original `36103_AT2_combined.ipynb` (34 code cells): what each part does, its row counts, and the three hard facts about the data that shape everything downstream |
| `02_next_steps.md` | The plan: decisions D1-D4, hypothesis-testing ladder, monthly macro as-of design, Event Study specification, modelling and rolling-CV design, packaging structure, milestones W1-W6 |
| `03_leakage_audit.md` | The audit that governs the code: **column-by-column verdicts** for all 24 cleaned columns, three project-specific leakage paths, nine feature-engineering red lines, six automated tests, a cell-by-cell audit of the notebook, and **three rounds of post-hoc findings (sections 8-10) covering 9 defects** found while building |

## The decisions these documents lock in (D1-D4)

| # | decision |
|---|---|
| **D1** | target = `log10(purchase_price)`, granularity = **per transaction** |
| **D2** | keep duplicate addresses; cross-validation is **grouped by `address|postcode`** |
| **D3** | Event Study v0 = **single event E1** (Sydney Metro Northwest, 2019-05-26), controls = **not-yet-treated** |
| **D4** | treatment distance from the **geocoded address sample** (property-level distribution), main radius **2 km** |

## Why the Chinese originals are archived rather than deleted

They are the documents the project was actually designed against, and section 8-10 of `03` are a
chronological record of what failed and why. Keeping them preserves that record. The English files
are the current authoritative versions; if the two disagree, the English one wins.

## Where the numbers come from

Every figure in these documents is reproduced by the pipeline:

| claim | source |
|---|---|
| cleaning row counts, fence values | `data/processed/clean.meta.json`, `reports/tables/iqr_bounds_by_year.csv` |
| model scores, fold definitions, holdout | `reports/tables/training_meta.json`, `model_comparison{,_holdout}.csv` |
| hypothesis tests | `reports/tables/hypothesis_summary.json`, `h1_*.csv`, `h2_*.csv` |
| event study | `reports/tables/event_study_meta.json`, `event_study_*.csv` |
| feature ablation | `reports/tables/feature_ablation.csv`, `feature_value_*.csv` |

## State of play

| item | status |
|---|---|
| cleaning, features, CV, models, hypothesis tests, event study, reports | done and reproducible |
| plausibility gate + option-A per-year fences | implemented and applied |
| `mark_quality` key-mapping bug (audit F1) | fixed, regression test added |
| true expanding window | verified: 503k -> 1.55M training rows across folds |
| **model retrain on the rebuilt data** | **not yet run** - the model numbers in the reports come from the previous data pass |
| property-level geocoding of in-scope sales | optional v1 (needs ~110k geocodes) |
| E2/E4 multi-event stacked design | v1; E2's core stations opened 2024-08, beyond the current data |
