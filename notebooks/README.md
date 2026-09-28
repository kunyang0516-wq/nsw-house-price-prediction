# notebooks

Two walkthrough notebooks that explain the project step by step. They are for reading
and understanding, not for producing results:

| notebook | covers | cells |
|---|---|---|
| `01_data_and_features.ipynb` | loading the raw CSV, the postcode lookups, cleaning, the three outlier layers, feature engineering and the leakage traps | 22 markdown, 22 code |
| `02_analysis_and_results.ipynb` | H1/H2 hypothesis tests and the ratio-bias placebo, the Metro event study, rolling-origin cross-validation, the feature ablation, and the report | 16 markdown, 25 code |

**They write nothing to disk.** The save steps are commented out; `01` rebuilds the
cleaning and feature frames in memory, and `02` reads the tables under `reports/tables/`.
The real files are produced by `pipelines/`, which runs the same code:

```powershell
python pipelines/run_chain.py --rf-jobs 16      # whole chain, ~60 minutes
python pipelines/00_check_staleness.py          # freshness + cross-table consistency
```

Because these notebooks import from `src` rather than reimplementing anything, the
numbers they display match the report exactly.

`archived/` is where the group's original `36103_AT2_combined.ipynb` belongs once the
team is finished with it. Until then that notebook stays at the repository root as
submission evidence.
