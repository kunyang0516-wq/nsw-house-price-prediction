# 01 - Reading `36103_AT2_combined.ipynb`

> A factual reading of the group's notebook: what each section does, the numbers its own stored
> outputs report, and where the design is unsafe. Nothing here is an estimate - the figures are
> quoted from the notebook's saved outputs and were re-checked against the raw file.
>
> Next steps are in `02_next_steps.md`; the leakage verdict is in `03_leakage_audit.md`.

---

## 0. Overall structure

The notebook is three member notebooks merged in dependency order: 62 cells, 34 of them code,
with clear boundaries.

| section | cells | content | output |
|---|---|---|---|
| header | 0-2 | merge note + Part 1 title | - |
| **Part 1 cleaning** | 3-37 | builds the modelling sample from the raw CSV | `nsw_residence_house_cleaned.csv`, `postcode_development_preclean.csv`, `postcode_transport_features.csv` |
| **Part 2 area-price EDA** | 38-50 | area/price relationship, correlations, unit-price hypothesis | figures (not saved) |
| **Part 3 postcode / Metro** | 51-61 | station vs non-station postcode comparison | `fig1_timing.png`, `fig2_price_vs_volume.png` |

The header itself warns: *"Outputs below come from three separate runs and are not mutually
consistent. Restart and run all before use."* - so the notebook must be re-run before delivery.

---

## 1. Part 1 - data cleaning (cells 3-37)

### 1.1 Input and scale (cells 3, 5)

- `nsw_property_data.csv`: 610.9 MiB, **4,854,814 rows x 17 columns**
- processed with `chunksize=250_000` so the whole file never has to fit in memory
- the raw grain is an individual property transfer, mixing units, houses, vacant land and
  several purposes

### 1.2 Annual economic lookups (cell 7)

- `cpi_by_year`: 23 values for 2001-2023, **hard-coded in the code**, with no stated source
  (the notebook itself notes it is unverified)
- `cash_rate`: scraped from the RBA cash-rate history table (with a local HTML cache), forward
  filled to daily and then **averaged over each calendar year** (2001 = 5.06, 2023 = 3.87)
- Both are therefore **full-year averages**, not the value knowable on the contract date - the
  direct cause of the time-leakage finding M3

### 1.3 Postcode development labels (cell 9)

- baseline: **2011-2014** transactions with `property_type = house`, purpose in
  {RESIDENCE, VACANT LAND}, price above $10,000
- per postcode, vacant share = VACANT LAND sales / baseline sales
- rule: at least 50 baseline sales and share >= 15% gives `Greenfield`, otherwise `Established`;
  too few sales gives missing
- output `postcode_development_preclean.csv`
- **risk**: the label is built from 2011-2014 information and applied to every year, including
  2001 - forward-looking for the early period

### 1.4 Postcode transport features (cell 11)

- sources: `stationentrances2020_v4.csv` (a **2020 snapshot**, cached locally) and
  `australian_postcodes.csv` (postcode centroid coordinates)
- station entrances are averaged to one point per station; postcodes to one centroid each
- four great-circle distances in km: `dist_cbd` (to -33.8688, 151.2093), `dist_train` (all
  stations), `dist_metro` (the 13 Northwest Metro stations), `dist_metro_new` (the first 8)
- output `postcode_transport_features.csv`, one row per postcode
- **risk**: postcode-centroid grain, and a 2020 station snapshot that cannot reconstruct the
  network as it existed in earlier years

### 1.5 The cleaning pipeline (cells 17, 19, 21, 23, 25, 27, 29)

| stage | rule | rows remaining | removed here |
|---|---|---|---|
| raw | - | 4,854,814 | - |
| 1 required fields | `purchase_price, area, area_type, cash_rate, cpi, contract_date, post_code, development_type, dist_cbd` all present | 3,321,221 | -1,533,593 |
| 2 purpose | `primary_purpose == RESIDENCE` | 2,539,858 | -781,363 |
| 3 type | `property_type == house` | 2,293,496 | -246,362 |
| 4 duplicates | rows identical on every column | 2,292,394 | -1,102 |
| 5 invalid values | price > 0, area > 0, four distances non-negative | 2,292,387 | -7 |
| 6 unit conversion | `area_type` M x1, H x10,000 to give `area_sqm` | 2,292,387 | 0 |
| 7 log-IQR outliers | log10(price) and log10(area) each within Q1-1.5IQR .. Q3+1.5IQR, drop if either fails | **1,867,040** | -425,347 |

Missing values before filtering (counts overlap): `area` 1,485,716; `area_type` 1,485,785;
`cash_rate`/`cpi` 34,130 each; `development_type` 26,645; `dist_cbd` 11,211.

Stage 7's original-unit bounds: **price $70,061.92-$3,728,701.62; area 238.57-2,015.84 sqm**.

Validation (cell 29): 1,867,040 rows x 24 columns, contract dates 2001-01-01 to 2023-12-31.

**Exported columns**: `property_id, download_date, council_name, purchase_price, address,
post_code, property_type, strata_lot_number, property_name, contract_date, settlement_date,
zoning, nature_of_property, primary_purpose, legal_description, locality, dist_cbd, dist_train,
dist_metro, dist_metro_new, cash_rate, cpi, development_type, area_sqm`.

### 1.6 Part 1 figures (cells 31, 33, 37)

- boxplots of area and price on linear axes
- histograms with 60 equal-width bins and a median line
- annual economic comparison: median/min/max price by contract year against annual `cash_rate`
  and `cpi`

The cell 37 scatter has only **23 points** (one per year), and its min/max lines are already
truncated by the IQR filter. It is a descriptive picture, not evidence that interest rates move
house prices.

### 1.7 Annual sample sizes (re-checked)

| year | 2001 | 2005 | 2010 | 2015 | 2020 | 2021 | 2023 |
|---|---|---|---|---|---|---|---|
| rows | 70,966 | 67,974 | 75,442 | 91,448 | 87,187 | 106,818 | 73,751 |

Only **545 postcodes** survive into the final sample - a number that matters for Part 3.

---

## 2. Part 2 - area and price EDA (cells 38-50)

- reads the cleaned CSV (5 columns), asserts types and positivity, and **does no further cleaning**
- descriptive statistics (cell 42): median area 645 sqm, median price $480,000
  (1st/99th percentiles: 259.2-1,718 sqm; $92,500-$2,900,000)
- area vs price (cell 44): all 1,867,040 rows as a hexbin with a log colour scale, plus a median
  trend across 15 equal-frequency area groups
- **correlations (cell 46)**: Pearson **-0.0315**, Spearman **-0.1082**, log10 Pearson **-0.1112**
- **Hypothesis 2** (cells 47-50): `price_per_sqm = purchase_price / area_sqm`, log-log hexbin,
  covering 1,798,174 of 1,867,040 rows (96.3%)

The section already states the key risk: area appears in both the predictor and the denominator,
so there is mathematical coupling, and "a negative correlation cannot demonstrate an area
discount, let alone causation". That statement is correct and has to be tested with a falsifiable
design (see `02_next_steps.md` section 3). Note also that all three correlations are reported
without confidence intervals, p-values, per-year or per-region stratification, or any controls.

---

## 3. Part 3 - postcode and Metro analysis (cells 51-61)

- window 2001-01-01 to 2023-04-30, price above $10,000
- groups: `METRO_PC = [2126, 2153, 2154, 2155, 2762]` (2155/2762 Greenfield, the rest
  Established), controls restricted to four SA4 regions; crossed with Greenfield/Established to
  give four cells
- panel: `post_code x year` with median price, sales count and development type, keeping
  `n_sales >= 20`
- price index: **2011 = 100**, computed as the median of member-postcode median prices

Key outputs:

- **parallel-trend check** (Established, Station minus No station): 2011 = 0.0, 2012 = -0.8,
  2013 = -2.8, 2014 = -1.8, 2015 = -0.9 index points
- **2023 index gap** (Jan-Apr only): Established **+50.2 pt**, Greenfield **+20.6 pt**
- **2021 sales per postcode**: Established 186 to 495 (+308), Greenfield 223 to 922 (+699)
- `fig1_timing.png`: the 2014 tunnelling and 2019-05-26 opening markers with the index series
- `fig2_price_vs_volume.png`: the 2023 price gap and the 2021 volume gap

Three things need strengthening:

1. **groups are defined after the fact** from 2011-2014 labels and 2020 station locations, with no
   pre-treatment matching or covariate balance check
2. the parallel-trend check covers only the **Established** group for 2011-2015; the Greenfield
   group is visibly non-parallel (2013: 118.5 vs 106.9, a gap of -11.6)
3. station postcodes are intrinsically higher-turnover inner or new-release areas (sales counts
   differ by 4-8x), so "the station effect" and "the area is different" are fully mixed

Part 3 is therefore a descriptive comparison and cannot support a causal claim. The Event Study in
`02_next_steps.md` section 5 is the route to a design with identification.

---

## 4. Three hard facts about the data

### 4.1 `property_id` is a strata plan number, not a property identifier

Re-checked: 4,854,814 rows contain only **1,944,986 distinct `property_id`** values. The extreme
case, `property_id = 4205410`, appears **24,421 times** across 262 different units in one Ryde
strata scheme (`910 B/6 NANCARROW AVE`, `105 A/6 NANCARROW AVE`, ...).

**Consequence:** `property_id` cannot be used for de-duplication, group-aware splitting or
repeat-sale matching. A dwelling has to be identified by `address + post_code`.

On the cleaned sample:

| metric | value |
|---|---|
| cleaned rows | 1,867,040 |
| distinct `address|post_code` | 1,158,796 |
| rows on a duplicated key | 1,184,938 (**63.5%**) |
| most repeats for one key | 62 |

So the same address appears on average 1.6 times, which creates a real risk of the same dwelling
landing in both training and test sets.

### 4.2 Transport distances are postcode-centroid based

`dist_metro` is the distance from a postcode centroid to the nearest Metro station. Within a 3 km
radius:

| radius | rows | postcodes |
|---|---|---|
| <= 1 km | 59,752 | 6 |
| <= 2 km | 77,989 | 11 |
| <= 3 km | 110,928 | 18 |
| <= 5 km | 188,660 | 36 |

Only **6 postcodes** sit within 1 km, so the treated group would be very small. Finer distance
rings require **property-level coordinates** - the repository already has a geocoded sample that
can be used for this.

### 4.3 Transaction density supports a monthly panel

Within 3 km and 2016-2021: 28,348 sales across 18 postcodes over 72 months, about **21.9 sales per
postcode per month**. A `postcode x month` panel is therefore statistically viable, which is the
precondition for the Event Study.

---

## 5. Code-quality notes

- `area` is missing for 1,485,716 raw rows (30.6%), and after the IQR stage the surviving area
  range is only 238-2,016 sqm. Large-lot, farm and acreage houses are effectively excluded, so the
  final sample is closer to "typical suburban block". That is a hard limit on how far any
  area-price conclusion generalises.
- Stage 7 uses a **single full-sample fence** across 23 years and every postcode. Prices in 2001
  and 2023 differ by multiples, so one fence systematically favours some years and regions.
- The same logic is recomputed several times (cell 19 rebuilds the data; cells 31 and 33 re-import),
  which is exactly what the packaged version removes.
