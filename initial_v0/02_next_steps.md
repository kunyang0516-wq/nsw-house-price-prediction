# 02 - Plan (v0)

> The design this implementation follows. Decisions D1-D4 and the milestone structure were fixed
> here **before** the code; the audits in `03_leakage_audit.md` record what went wrong afterwards.

---

## 0. Locked decisions (v0 freeze)

| # | decision | consequence |
|---|---|---|
| **D1** | target = **`log10(purchase_price)`**, granularity = **per transaction** | headline metric RMSLE; the event study uses a `postcode x month` panel, which is an estimation choice and does not conflict |
| **D2** | keep all duplicate addresses; CV is **grouped by `address|postcode`** | preserves repeat sales; a "first sale per key only" variant is run as a sensitivity check |
| **D3** | Event Study v0 = **single event E1** (2019-05-26 opening), controls = **not-yet-treated** | establishes a baseline; E2/E4 stacking is deferred to v1 |
| **D4** | treatment distance from the **geocoded property sample**, main radius **2 km** | no property-level geocoding of the full sample; finer rings deferred |

**Hard constraints implied by D1-D4**

- `price_per_sqm`, any contemporaneous or future-derived area statistic, and anything derived from
  `settlement_date` are **forbidden as features**
- the splitter signature is `rolling_origin_splits(..., group_key="address|postcode", embargo>=1)`
- every step that estimates a parameter from data (IQR fences, imputation, bin edges, target
  encoding) is fitted **inside the training fold only**
- the event study declares **one** main outcome in advance; the others are robustness

---

## 1. From the current notebook to the finished project

Descriptive EDA -> falsifiable hypothesis tests -> reproducible feature engineering ->
leakage-free rolling-origin modelling -> quasi-experimental causal estimation -> packaged delivery.

---

## 2. Data contract and the three things that had to change

### 2.1 The contract (`src/data/contract.py`)

Assertions that fail loudly:

| check | expected |
|---|---|
| rows | 1,867,040 in the original; **1,774,829** now (changes recorded explicitly) |
| distinct postcodes | 545 originally, **596** now |
| date span | 2001-01-01 to 2023-12-31 |
| `property_type` / `primary_purpose` | all `house` / all `RESIDENCE` |
| `development_type` | in {Established, Greenfield} or missing (never a stale value) |
| distances | non-negative, non-missing for the required column |
| price / area | strictly positive and finite |
| **absolute plausibility** | area 20-10,000 sqm; price $10k-$50M; unit price $10-$50k/sqm |
| primary key | `address + post_code` may repeat; `property_id` is **banned** as an identifier |

### 2.2 The three changes

| # | original | problem | fix |
|---|---|---|---|
| **M1** | one log-IQR fence over the whole 2001-2023 sample | trims the **target's** tails using all periods | per-year fences, fitted on the full sample, applied to the training side only; rows outside are flagged rather than silently deleted |
| **M2** | 63.5% of rows share an `address|postcode` key | the same dwelling can straddle a split | every fold purges validation group keys from training |
| **M3** | annual mean cash rate and annual hard-coded CPI on every row | uses decisions made after the sale | monthly **as-of** join: cash rate by effective date, CPI on `release_date` with a 28-day publication lag |

---

## 3. How to test the EDA hypotheses

The principle: EDA produces **observations**; a test produces a **falsifiable statement with an
effect size, uncertainty, and at least one design that excludes competing explanations**. Three
levels, each stricter than the last:

```
L1 description + uncertainty  ->  L2 conditioning  ->  L3 quasi-experimental design
```

### H1 - area and price

**Falsifiable statement.** Controlling for location and time, `log10(area_sqm)` has no association
with `log10(price)` (beta = 0).

- **L1**: add bootstrap CIs **clustered by postcode** (rows are not independent), plus per-year
  and per-SA4 stratification; if yearly coefficients differ in sign, one pooled number is
  meaningless
- **L2**: `log10(price) = beta * log10(area) + f(distance) + year FE + postcode FE`, with standard
  errors clustered by postcode. Beta then measures the within-location, within-year gradient. The
  expectation was a **sign flip** once location is absorbed (a Simpson reversal), which is itself
  the finding
- **L3**: the sample is IQR-fenced to a narrow area band, so the conclusion is stated as holding
  only within that band

### H2 - larger area, lower unit price

This is a trap: `price_per_sqm = price / area` puts the regressor in the denominator, so a negative
correlation appears even when price and area are independent.

Three tests:

1. **placebo**: shuffle price, recompute the ratio correlation. Whatever survives is the mechanical
   component
2. **elasticity**: estimate log price on log area. beta < 1 means unit price falls with size;
   beta = 1 means no discount. This avoids the ratio entirely
3. **within-band medians**: a descriptive monotonicity check

### H3 - the Metro effect

**Falsifiable statement.** Before and after opening, treated postcodes follow the same price path
as controls (all event-time coefficients are zero).

Checks: a joint pre-trend test (the admission gate), an event-study coefficient plot, robustness to
the control definition / radius / outcome, placebo event dates, heterogeneity by development type
and distance, and a SUTVA check separating near from far controls.

---

## 4. Monthly cash rate and CPI

**Principle:** on contract date *t*, use only macro values **already published** by *t*.

- **cash rate**: keep the RBA **effective date**; the value available on *t* is the most recent
  change with `effective_date <= t`. The target changes on the day, so there is no publication lag.
  Derived features: rate level, months since the last change, change over the last 3 months
- **CPI**: store both `period_end` and `release_date` and join on `release_date <= contract_date`.
  Quarterly CPI is published roughly four weeks after quarter end, so a January 2022 contract can
  only use the 2021 Q3 print
- features: `cpi_yoy` (year-ended inflation) and a real price deflator
- **annual versions stay for descriptive EDA only**; modelling uses the `*_asof` columns, clearly
  named apart

Tests: `release_date <= contract_date` must always hold; shuffling macro values after *t* must not
change features at *t*.

---

## 5. Event Study design (H3)

### 5.1 The event table

| event | stations | date | note |
|---|---|---|---|
| **E1** | Tallawong to Chatswood, 13 stations | 2019-05-26 | the v0 study |
| E2 | City & Southwest | 2024-08 onward | beyond the current data range |
| E4 | existing-line upgrades | to be confirmed | placebo or control |

Opening dates must be verified per station, with a source URL, in `data/external/events.csv`.

### 5.2 Treatment and exposure

1. **treatment (binary, v0)**: distance to the nearest new station <= 2 km, with 1 km and 3 km as
   robustness
   - the distance measure is `dist_metro_p25` from the **geocoded property sample**: at least 75%
     of geocoded homes in the postcode are within that distance. Distances are measured to station
     **entrances**, and only stations open at the event date are used. Postcodes with fewer than 3
     valid addresses fall back to the centroid
   - measured effect of the change: at 2 km, **6 postcodes change arm**, all of the form "centroid
     far but most homes close"; the treated set grows from 8 to **11**
2. **exposure intensity**: `share_within_1km` as a continuous treatment intensity for the
   dose-response specification
3. **treatment timing**: v0 is a single date; v1 assigns each postcode the opening month of its
   first new station (staggered)
4. **controls**: same SA4 regions, at least 3 km from any new station ("not-yet-treated" within the
   window)

### 5.3 Panel and outcomes

`postcode x month` (density verified at about 22 sales per postcode per month within 3 km).
Outcomes: log median price, log median unit price, sales count, turnover.

### 5.4 Estimation

1. **TWFE event study**: `y_pt = sum_k beta_k D_pt^k + postcode FE + month FE`, reference period
   `k = -1`. With a **single cohort this is exact**, not an approximation
2. **stacked difference-in-differences** for v1's multiple events: one stack per event, each with
   contemporaneous not-yet-treated controls, then precision-weighted pooling
3. **Callaway-Sant'Anna** and **Sun-Abraham** for v1, to avoid TWFE's negative weights under
   staggered adoption
4. standard errors clustered by postcode

### 5.5 Window and placebos

Window [-24, +24] months with binned endpoints; reference period `k = -1`. Placebos: shift the
event date earlier; assign treatment at random; use an unrelated outcome; separate near from far
controls. Each is reported, and a rejected pre-trend is reported as a rejection rather than tuned
away.

### 5.6 Pre-registration

Treatment definition, radius, window, reference period, control construction, main outcome,
clustering level and any multiple-testing correction are recorded in
`data/interim/event_study_spec_E1.json` **before** estimation.

---

## 6. Modelling

### 6.1 Target and granularity

`log10(purchase_price)`, per transaction, features restricted to what is knowable at the **contract
date**.

### 6.2 Feature layers

| layer | features | note |
|---|---|---|
| A property | area, area bin, zoning, council, development label | known at contract |
| B location | `log(dist_cbd)`, `dist_train`, `dist_metro`, locality, SA4 | strongly collinear; check VIF, keep one Metro distance |
| C regional history (**past only**) | rolling 3/6/12-month median price, unit price, sales count, dispersion, coverage | rolling windows with an embargo |
| D calendar | `year_num` (continuous), `month_sin`, `month_cos` | encoded numerically, not by frequency |
| E macro | `cash_rate_asof`, `cpi_yoy_asof` | monthly as-of only |

**Forbidden:** `price_per_sqm`, any contemporaneous or future-derived statistic, anything from
`settlement_date`.

### 6.3 Rolling-origin CV

Expanding window by default, 12-month horizon, 1-month embargo, purged by `address|postcode`, with
a final 12-month holdout reserved and scored **once** after settings are frozen.

### 6.4 Models and metrics

Baselines (postcode rolling median; global median), ridge, random forest, XGBoost. Headline
**RMSLE**, plus MAE/RMSE in dollars and MdAPE, all reported per fold and per segment
(year, SA4, price band). **RMSLE and MdAPE can rank models differently** - see the report's
section 4.6 for the full treatment.

---

## 7. Packaging

```
src/{data,features,models,validation,causal,evaluation,utils}
pipelines/01..08           # the only entry points
tests/                     # contract, leakage, macro as-of, CV
reports/{figures,tables}   # every number traced
initial_v0/                # design and audits
```

Rules: no logic in notebooks; `pipelines/` is the only entry point; `reports/` is the only place
figures are drawn; raw data is read-only.

---

## 8. Milestones

| stage | content | acceptance |
|---|---|---|
| W1 | data contract, cleaning as code, M1 | pipeline runs; contract tests green; row-count differences recorded |
| W2 | monthly macro, rolling features, leakage tests | as-of tests green; "annual vs monthly" metric table |
| W3 | baselines, metric standard, rolling-CV skeleton | baseline scores fixed; CV figure produced |
| W4 | three model families, tuning, comparison | comparison table, residual diagnostics, importance |
| W5 | event study, placebos, pre-trend test | coefficient plot, pre-trend table, cohort ATTs |
| W6 | packaging, final report | one command reproduces every figure |

---

## 9. Still open

- whether to add a "1-month and 3-month-ahead" prediction variant
- verifying E2/E4 opening dates station by station
- whether the Metro main outcome should be unit price or total price
- sampling the duplicate addresses to confirm whether repeats are one dwelling sold twice or
  different units at one address
- if the pre-trend test fails, whether a purely descriptive framing is acceptable (it was, and is)

---

## 10. Self-check list

- [ ] `pytest tests -q` green (contract, leakage, CV, macro as-of)
- [ ] `01_build_clean.py` rebuilds from raw
- [ ] every figure regenerated by `07_make_report.py`, with only expected diffs
- [ ] new features pass the future-perturbation test
- [ ] the event study's pre-trend test is reported, pass or fail
- [ ] every number in the report traces to a CSV in `reports/tables/`
