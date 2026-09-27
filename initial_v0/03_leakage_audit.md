# 03 - Data leakage audit

> **Purpose:** run this table over any cleaning or feature-engineering logic before reusing it.
> Every row must answer "did this value exist and was it public on contract date *t*?"
> If it cannot, treat it as leakage until shown otherwise.
>
> Status: OK / conditional / banned / to confirm.

---

## 0. Three principles

1. **Availability.** A feature must exist objectively on or before contract date *t*, **and be
   public**. Existence is not publication: a CPI quarter may have ended while its release is still
   four weeks away.
2. **Fit once, inside the fold.** Anything estimated from data - IQR bounds, means, quantile edges,
   target encodings, imputation values - must be fitted on the training fold and applied outward.
   Fitting on the whole sample is leakage.
3. **Independent units.** The split unit must separate time, space **and** dwelling. The same
   `address|postcode` (63.5% of rows share a key) must not straddle train and test.

---

## 1. Column-by-column verdicts (the 24 cleaned columns)

| column | verdict | reason / handling |
|---|---|---|
| `purchase_price` | target | only ever `log10()` of it; never a feature |
| `contract_date` | OK | source of year, month, season; do **not** derive "days since download" from it |
| `settlement_date` | **banned** | settlement is **after** contract; any duration derived from it is future information |
| `download_date` | **banned** | 2024-02 snapshot date |
| `property_id` | **banned** | a **strata plan number**, not a property id (one value appears 24,421 times); encodes "which building" and leaks across folds |
| `strata_lot_number` | conditional | missingness may indicate a strata property; the **raw value** reveals building identity, so binarise only |
| `property_name` | conditional | mostly NaN; at most "is it present" |
| `legal_description` | **banned** | parcel fingerprint; links repeat sales of the same parcel |
| `address` | **group key only** | contains the unit number, so it is a near-unique identifier; never a feature |
| `post_code` | OK categorical | known at contract; confirm boundaries did not change |
| `council_name` | OK categorical | same |
| `locality` | OK categorical | overlaps postcode; keep one or use a hierarchy |
| `property_type` | constant | all `house` after cleaning; drop |
| `primary_purpose` | constant | all `RESIDENCE`; drop |
| `nature_of_property` | near-constant | `R` for 1,867,014 of 1,867,040; drop |
| `zoning` | OK | known at contract; merge rare codes |
| `area_sqm` | **to confirm (important)** | see section 2.1 - if it comes from a government assessment register, the recorded value may post-date the sale |
| `cash_rate` | **banned as implemented** | annual calendar-day weighted mean -> monthly as-of instead (section 3.1) |
| `cpi` | **banned as implemented** | annual mean plus no publication lag (section 3.2) |
| `development_type` | restructured | see section 2.2 |
| `dist_cbd` | OK | static geography; keep the coordinate snapshot fixed |
| `dist_train` | OK | same, though new stations do open during the window (section 2.3) |
| `dist_metro` | time-varying | computed from a **2020** snapshot but used across 2001-2023; for pre-2019 contracts this is a facility that did not exist yet. See section 2.3 |
| `dist_metro_new` | time-varying | same, and strongly collinear with `dist_metro`; keep only one |
| `price_per_sqm` (derived) | **banned** | target over regressor: leakage plus ratio bias |

---

## 2. Three project-specific leakage paths

### 2.1 The vintage of `area_sqm`

- the raw `area` is missing for 1,485,716 of 4,854,814 rows (30.6%) and mixes units (3.419 `H` vs
  727.2 `M`)
- risk: a register's `area` is often the **current** assessed value, not the value at the time of
  the transfer notification
- handling: confirm the data dictionary with the team; failing that, (1) state the "area is known at
  contract date" assumption in the report, (2) run a sensitivity test replacing `area_sqm` with a
  postcode-by-area-band indicator, and (3) never build a ratio of area to price

### 2.2 When `development_type` was constructed

- built from the **2011-2014** vacancy share and applied to contracts from 2001
- for 2001-2010 contracts the label contains information that did not yet exist
- two safe options:
  - **A (chosen)**: version the label. `development_type_v2010` (baseline 2001-2010) is legal from
    2011 onward; `development_type_v2014` from 2015. A single legality-checked `development_type`
    column is `NA` where no legal label exists
  - **B**: use a genuinely pre-determined proxy (2001 population density, distance to CBD, the
    vacancy share as of the contract year)
- either way the label carries a version stamp, not one anonymous column

### 2.3 "Time travel" in the distance features

- `dist_metro` uses **2020** station coordinates, but the 13 Northwest stations opened on
  **2019-05-26**
- consequence: a facility that did not exist is treated as a 2001-2018 feature. For modelling this
  is leakage; **for the event study it is fatal**, since the design assumes no station before opening
- correct approach: give each station an `open_date`, build
  `dist_metro_t = min{distance to stations with open_date <= t}`, and define the event study's
  treatment from a **pre-opening** snapshot

> **Update - implemented.** The treatment variable no longer uses the postcode centroid. It uses the
> **property-level distance distribution** from the geocoded sample
> (`src/data/geocode_features.py`): distances measured to station **entrances** (52 relevant
> entrances), using only stations **already open** at the event date (`_stations_at_event`). The
> headline variable `dist_metro_p25` is the distance within which 75% of a postcode's geocoded homes
> sit. At the 2 km threshold **6 postcodes change arm**, and the treated set grows from 8 to 11 -
> evidence that the centroid was mis-assigning boundary postcodes. Postcodes with fewer than 3 valid
> addresses (91 of them) fall back to the centroid, and the source counts are recorded in
> `event_study_meta.json`.

---

## 3. Macro leakage and its fix

### 3.1 Cash rate

| item | as implemented originally | target |
|---|---|---|
| grain | annual calendar-day weighted mean | daily target rate, as-of the contract date |
| timing | includes rate decisions made **after** the sale | most recent `effective_date <= contract_date` |
| extra risk | scrapes the RBA page (`rba_cash_rate_source.html` did not exist locally, so every run fetched HTML) | fetch the official table and snapshot it under `data/external/` |

### 3.2 CPI

- store both `period_end` and `release_date`
- join on `release_date <= contract_date`, **not** `period_end <= contract_date`
- quarterly CPI is published roughly four weeks after quarter end, so a January 2022 contract can
  only use the 2021 Q3 print
- if annual CPI is retained, use it for **descriptive EDA only**

### 3.3 Three mandatory assertions

```
assert (macro.release_date <= contract_date).all()      # publication lag
assert feature_frame.isna().sum() matches expectation  # no ffill fabricating values
# perturbation: shuffle macro values after t; features at t must not change
```

---

## 4. Feature-engineering red lines

| # | red line | note |
|---|---|---|
| L1 | **rolling regional statistics** | postcode-level "past N months" medians/counts/volatility must (1) end at t-1 or earlier, (2) add the publication lag, (3) be NaN when insufficient, **never forward-filled** |
| L2 | **target encoding** | computed inside the training fold only, with leave-one-out or smoothing |
| L3 | **quantile binning, standardisation, imputation** | thresholds from the training fold; whole-sample `qcut` or `.mean()` is leakage |
| L4 | **IQR trimming** | per year, fitted for the purpose, applied to the training side only; never a single full-sample fence, and never used to delete validation rows |
| L5 | **duplicate addresses** | 63.5% of rows share a key -> group-aware CV |
| L6 | **pipeline ordering** | a `FunctionTransformer` or feature step computed outside CV is still leakage; anything depending on y or on whole-sample distributions belongs inside the fold |
| L7 | **lagged target** | `lag(price)` must genuinely shift, within group, with NaN rather than the current value |
| L8 | **geocoding** | encoded coordinates are fine as features (known at contract), but the lookup must not carry information learned later |
| L9 | **target transform** | back-transform with `10**pred` for MAE/RMSE; note that a mean back-transform carries smearing bias |

---

## 5. Automated defence (`tests/`)

| test | method | pass condition |
|---|---|---|
| T1 temporal leakage | `corr(f_t, y_{t+k}) - corr(f_t, y_t)` for k = 1, 3, 12 | no systematic inflation |
| T2 future perturbation | shuffle all raw data after the test period, recompute features | test-period features identical |
| T3 group isolation | random KFold versus `address|postcode`-grouped KFold | the difference is the leakage magnitude and is disclosed |
| T4 whole-sample fit | features computed outside the pipeline compared with in-pipeline | identical |
| T5 single-feature audit | fit a shallow model per feature | any unusually strong single feature must be explained |
| T6 as-of assertions | section 3.3 | all green |

Implemented as `tests/test_leakage.py`, `test_cv_splits.py`, `test_macro_asof.py` and
`test_quality_fence.py`.

---

## 6. Cell-by-cell audit of `36103_AT2_combined.ipynb`

Verdicts: reusable / needs rework / must not enter the modelling pipeline (EDA only).

### 6.1 Cleaning (cells 3-37)

| cell | content | verdict | issue | replaced by |
|---|---|---|---|---|
| 3 | path constants, `CHUNK_SIZE` | reusable | - | `config/settings.py` |
| 5 | 1,000-row preview + chunked count | reusable | - | `src/data/load_raw.py` |
| 7 | annual cash rate (RBA day-weighted mean) + hard-coded annual CPI | **must not be used** | **M3**: a full-year mean includes decisions made after the sale; CPI has no stated source or lag | `src/data/macro_asof.py` |
| 9 | postcode development labels (2011-2014 vacancy share) | needs rework | section 2.2 | versioned labels in `postcode_features.py` |
| 11 | station/postcode coordinates + four great-circle distances (2020 snapshot) | needs rework | section 2.3 | static distances for EDA plus a time-varying/geocoded version |
| 13 | missing-value check | reusable | - | folded into the contract |
| 15 | chunked left join with `validate="many_to_one"` and a length assertion | reusable | the most careful code in the notebook | `join_transport` |
| 17 | cleaning rules (three date formats, numeric coercion, required fields, purpose/house filters, counts) | reusable with edits | `years.map(cash_rate/cpi)` imports M3; the label imports section 2.2 | `clean_chunk` |
| 19 | chunked application, concatenation, staged counts | reusable | - | `build_clean` |
| 21 | whole-row de-duplication (1,102 rows) | reusable with edits | M2: the real problem is 63.5% of rows sharing a key, which this misses | keep, plus a `group_key` for CV |
| 23 | positive price/area, non-negative distances | reusable | - | same |
| 25 | area unit conversion M/H to sqm, `df_before_iqr` snapshot | reusable | the snapshot pattern is worth keeping | `convert_area_units` |
| 27 | log-IQR with a **single full-sample fence** | **must not be used** | L4: one fence across 23 years removes 425,347 rows (18.5%) from the **target's** tails | per-year fences, training side only |
| 29 | post-cleaning assertions | reusable | - | `validate_clean` |
| 31, 33 | boxplot and histogram | EDA | - | `07_make_report.py` |
| 35 | CSV export | needs rework | 335 MB CSV; already dropped `area`/`area_type`/`sa4`; no `group_key` | parquet plus `group_key` |
| 37 | annual economic scatter | EDA only | 23 points, min/max already truncated by the IQR filter; not evidence about rates | keep as a descriptive chart, labelled "n = 23 annual aggregates, not causal" |

### 6.2 Feature engineering (cells 40-61)

| cell | content | verdict | issue | replaced by |
|---|---|---|---|---|
| 40 | read the cleaned CSV (5 columns) | reusable | - | `load_clean` |
| 42 | descriptive statistics | EDA | - | report |
| 44 | hexbin + 15-bin median trend (`qcut`) | EDA | whole-sample `qcut` is fine for display, L3 leakage as a feature | display kept; bins refitted per fold if used as a feature |
| 46 | three correlations | needs rework | no CI, no p-value, no stratification | `hypothesis_tests.py` (clustered bootstrap + two-way FE) |
| 48 | `price_per_sqm` | **banned** | target divided by a regressor | the object of the H2 test, never a feature |
| 50 | unit-price log-log hexbin | EDA | same `qcut` note | display kept; H2 uses the placebo and elasticity |
| 53 | read cleaned data, window to 2023-04-30 | reusable | - | `build_panel` |
| 54 | locality / SA4 labels | reusable | - | same |
| 55 | development filter and hard-coded `METRO_PC` | needs rework | depends on the 2020 snapshot; only 5 treated postcodes | derived from the event table and the geocoded distances |
| 56 | `post_code x year` panel with a `stn` flag | needs rework | aggregation is safe as description, but `stn` is defined after the fact and would leak as a feature | `post_code x month` panel with an as-of treatment variable |
| 57 | 2011 = 100 index + parallel-trend check | descriptive | index construction is correct; only the Established group is checked | kept as a preliminary check; the formal test is in `event_study.py` |
| 58 | sales-volume comparison | descriptive | - | kept |
| 59, 60 | fig1, fig2 | EDA | the wording already says "descriptive", which is correct | kept |
| 61 | date range, thresholds, category counts | reusable | - | folded into the contract |

### 6.3 The four audit questions, answered

| question | cells hit | conclusion |
|---|---|---|
| (1) whole-sample thresholds before splitting? | **27** (full-sample IQR), 44/50 (`qcut`) | 27 had to change (L4); 44/50 only if used as features |
| (2) any use of `settlement_date` / `download_date` / `property_id` / `legal_description` / raw `address`? | none | clean - the notebook never touched these, which is a genuine strength |
| (3) cross-time aggregates on the training period? | **7** (annual macro), 57/58 (full-period panel statistics) | 7 had to change (M3); 57/58 are descriptive only |
| (4) current snapshots back-filled into history? | **11** (2020 stations), 9 (2011-2014 labels), 55 (hard-coded postcodes) | all three had to become as-of |

### 6.4 Summary

**Reusable as-is** (about 70% of the cleaning logic): chunked reading, the `validate="many_to_one"`
join, required-field / purpose / house filters, unit conversion, the pre-IQR snapshot, whole-row
de-duplication, post-cleaning assertions, panel aggregation, and the figure-drawing code.

**Had to be rewritten** (4 places): cell 7 annual macro; cell 27 full-sample IQR; cells 9/55
development labels and treatment definition; cell 11 static distances.

**Had to be added** (2): `group_key`; rolling regional features using past data only.

---

## 7. Relationship to the other documents

- cleaning facts: `01_eda_walkthrough.md` section 1
- M1/M2/M3 fixes: `02_next_steps.md` section 2.2
- monthly macro: `02_next_steps.md` section 4
- event-study treatment construction: `02_next_steps.md` section 5.2 (this document's section 2.3 is
  its precondition)

---

## 8. First round of post-hoc findings: five data-side defects

Sections 1-7 were written **before** implementation. Five further defects surfaced once the code
ran, all of the "the audit did not cover this, and it would have produced a wrong conclusion" type.

### D1. `area_sqm` values of impossible magnitude

- **symptom:** ridge produced a single **$9.1e11** prediction and pushed the dollar RMSE in the
  model comparison to $1.03e9
- **cause:** a 2020 "house" with `area_sqm = 1.53e9` (1.53 billion square metres), about 1,020
  standard deviations from the mean after standardisation
- **extent:** 4.8% of rows exceed 20,000 sqm; 4.9% exceed 100,000; the maximum is 2.7e9
- **why nothing stopped it:** the per-year fences only existed for 2001-2015, and that row is from
  2020 - unjudged
- **lesson:** **outlier filtering must cover every year**, or the dirtiest records sit outside the
  fence

### D2. `purchase_price` values of impossible magnitude

- **symptom:** a maximum "residence" price of **$875,300,000**; 972 rows above $20M; a $555M row
  inside the training window
- **effect:** contaminates training (linear coefficients shift) and makes dollar-scale metrics
  meaningless
- **fix:** the same corruption fence as area

### D3. A per-year fence fitted on the training span silently emptied later validation windows

- **symptom:** with the filter enabled, all folds after 2015 reported `empty side, skipped`, while
  the report still showed "8 folds"
- **cause:** per-year fences existed only for 2001-2015; for later years "unjudged" was treated as
  "failed"
- **effect:** metrics improved dramatically (RMSLE 0.45 -> 0.30) purely because the sample halved
- **fix:** two independent fences -
  - `iqr_corruption_ok`: fitted on the training span, applied to **every** row; removes only
    impossible values
  - `iqr_keep`: the per-year quality fence, applied to the **training side only**
  Validation is then scored on the full population rather than a self-selected subset

### D4. A contract assertion checked the wrong column

- **symptom:** `dev_label_legality` reported 968,666 violations indefinitely
- **cause:** it tested the **source** column `development_type_v2010` (populated for all years)
  instead of the legality-checked `development_type` (populated only after the window closes). The
  data was correct; the check was not
- **lesson:** **contract assertions need auditing too** - a failing check does not always mean the
  data is wrong

### D5. Feature engineering: `min_periods` overstated what the column name promised

- **symptom:** `pc_med_price_6m` produced a value from as few as three observed months
- **cause:** `min_periods = window // 2`
- **fix:** require a fully observed window, and emit `pc_n_months_observed_{3,6,12}m` so coverage is
  explicit. Cost: 12-month missingness rose from 1.5% to 8%

### The pattern across these five

| pattern | symptom | check |
|---|---|---|
| **fence coverage incomplete in time** | the dirtiest record sits outside the fitting window | for every filter ask "what does it do to rows after the training period?" |
| **a filter silently changes the sample** | metrics improve because the sample shrank | print row counts on both sides of every filter |
| **an assertion checks the wrong column** | a check that always fails, or always passes | when an assertion fails, verify the assertion first |
| **a lenient parameter overstates a column name** | a "6-month" window built from 3 months | pair every rolling statistic with a coverage counter |

---

## 9. Second round: four defects found in the models and placebos

### E1. The naive benchmark was double-log-transformed

- **symptom:** `naive_postcode_month` RMSLE jumped from 0.68 to **2.18** and R² to **-10.5**, with
  predictions around log10 = 0.75 (about $5.60)
- **cause:** `train_median_log` was already log10, and the function applied `np.log10` again
- **effect:** the benchmark column of the comparison table was meaningless, making every model look
  like a large win over a broken baseline
- **fix:** use the log value directly, with a graded 12m -> 6m -> 3m fallback
- **lesson:** **verify the baseline independently**; while it is broken, no "beats the baseline"
  claim means anything

### E2. Sample-selection bias in the time-reversal placebo

- **symptom:** swapping the lagged feature for a **future** window improved RMSLE from 0.4355 to
  **0.3687** - which looks like future leakage
- **cause:** the placebo applied a `dropna` after the swap, removing 2,127 validation rows with an
  incomplete forward window; the surviving rows were in more active postcodes and therefore easier
- **verification:** holding the row set fixed gives **0.4765** for the future version, **worse** than
  the real 0.4355, as it must be
- **lesson:** **any placebo or ablation must lock the row set**, or the "effect" may just be a
  different sample

### E3. Metrics "improved" after a fence emptied the validation windows

- same mechanism as D3, recorded separately because the lesson is about interpretation: when a
  metric improves, suspect the sample before celebrating

### E4. `--max-train-rows` made "expanding" an implicit sliding window

- **symptom:** `training_meta.json` records `min_train_months: 84`, but each fold actually trained on
  the most recent 400,000 rows (about 3.75-4.25 years)
- **effect:** early folds discarded history; documentation did not match behaviour
- **fix:** disclosed in the report; `--max-train-rows 0` gives a true expanding window
- **lesson:** metadata must record the **actual** training volume, not only a lower-bound constraint

### Four general checks from round two

1. baselines need their own correctness test, not merely "it ran"
2. placebos and ablations must lock the row set and report row counts on both sides
3. every claimed improvement must be accompanied by the change in sample size
4. metadata records actual training volume, not just hyper-parameters

---

## 10. Third round: two defects that only surface under specific conditions

### F1. `Series.map(dict)` dtype mismatch (a real bug, silent)

- **symptom:** the expanding window's training set did not grow as expected; a per-year diagnostic
  showed **100% of every year from 2016 onward being rejected**
- **cause:** `mark_quality` looked fences up with `keys.map(lows)`, where `lows` was keyed by
  `astype("string")` (pandas StringDtype) while the `year` column was Int64.
  **`Series.map(dict)` does not coerce dtypes**, so it returned all-NaN, and `fillna(False)` then
  marked everything as failing
- **effect:** the training side was emptied while the **validation side was unaffected** (the quality
  fence only applies to training), so every per-fold metric looked normal and the "expanding window"
  plateaued at 1.12M rows
- **fix:** resolve keys then look up through a `Series`; regression test added in
  `tests/test_quality_fence.py` (**fails before the fix, passes after**)
- **why it hid:** the bug only triggers for years without their own fence, and every existing
  assertion looked at totals
- **a trap in the test itself:** the first version of the regression test used a higher price level
  for the out-of-span year, so the **corruption fence** rejected those rows and the test "passed" for
  the wrong reason. The fixture must keep the same distribution

### F2. The quality fence's fitting window did not match the years it judged

- **symptom:** even after F1, many **legitimate** expensive sales from 2016 onward were flagged as
  outliers
- **cause:** per-year fences were fitted on the **training span (2001-2015)** but applied to 2023
  prices. The training-span ceiling was **$2.34M** while the 2023 99th percentile is **$2.9M** - the
  ruler was wrong
- **effect:** later folds had their training set wrongly compressed, so "expanding" was misnamed
- **fix (option A):** fit the per-year quality fence on the **full sample**. Ceilings now move with
  the year:

  | year | before (global) | after | share flagged |
  |---|---|---|---|
  | 2015 | $2.34M | $3.71M | 1.7% |
  | 2021 | $2.34M | $5.73M | 1.9% |
  | 2023 | $2.34M | $5.18M | 1.1% |

  Flagged shares are now stable at **1.1%-2.3%** instead of 100%
- **methodological note:** this is a **one-off dataset-level decision** (12.5% of rows trimmed,
  stated in the methods), not a per-fold statistic. It never touches the target and it is disclosed.
  Using a 2001-2015 ruler on 2023 prices was the actual error

### Three new general checks from round three

5. **unify key types explicitly** in any lookup; never rely on `Series.map(dict)` coercion
6. **any rule that is fitted by time and applied by year must emit a per-year diagnostic table**
   (`reports/tables/iqr_bounds_by_year.csv`) - both F1 and F2 are immediately visible there
7. **test fixtures must not introduce an extra dimension**: if a fixture also trips another rule,
   a passing test may be passing for the wrong reason
