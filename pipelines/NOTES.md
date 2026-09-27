# pipelines

Numbered so the order is unambiguous. Run them in sequence:

```
00_check_staleness.py   which tables predate features.parquet + cross-table consistency
01_build_clean.py       raw CSV        -> data/processed/clean.parquet
02_build_features.py    clean          -> data/processed/features.parquet
04_train_models.py      rolling-origin CV + final holdout -> reports/tables/
05_hypothesis_tests.py  H1/H2, robustness, ratio-bias placebo
06_event_study.py       Metro E1 event study (TWFE + placebos + dose response)
07_make_report.py       every figure + reports/final_report.md
08_feature_value_test.py  layered ablation + permutation/time-reversal placebos
run_chain.py            runs 01 -> 07 in one process, in order
```

`run_chain.py` exists because `--rf-jobs` needs threads and joblib's pool cannot
be built in a confined shell (see below). Running every stage in a *single*
process keeps joblib on its thread path and keeps peak memory to one frame at a
time:

```powershell
.\.venv\Scripts\python.exe -u pipelines\run_chain.py --rf-jobs 16
```

`--from 04` resumes at a stage, and stage numbers (`01 02`) select a subset. A
full run takes about 60 minutes: 01 and 02 are ~5 min together, 04 dominates at
~43 min, and 05/06/07/08 add ~15 min.

The stages are seeded throughout (`random_state=36103`, `default_rng(36103)`), so
re-running 05 and 08 reproduces the same numbers.

`00_check_staleness.py` compares each stage's artifacts against the modification
time of the frame that stage consumes, and against the row count it recorded when
it ran. A stale table is invisible otherwise: `07_make_report.py` will build a
report with fresh prose and old numbers. Run it after any rebuild of
`clean.parquet` / `features.parquet`:

```
.\.venv\Scripts\python.exe pipelines\00_check_staleness.py
```

Note that `04_train_models.py`, `05_hypothesis_tests.py`,
`06_event_study.py`, `07_make_report.py` and `08_feature_value_test.py` all write
into `reports/tables/`, so they must not be run concurrently.


`03` was never needed: the postcode panel is produced as a byproduct of `02`
(`data/processed/panel_postcode_month.parquet`).

`09_make_report_cn.py` wrote a Chinese counterpart of the report. It now lives in
`attic/chinese/` along with the Chinese design documents, because the repository's
deliverables are English. It can be run from there if a Chinese report is needed
again (it reads the same tables, so it cannot drift from the English version).

## Running the expensive step (`04_train_models.py`)

`--max-train-rows` defaults to **400,000**, which silently caps every "expanding"
window to a ~4-year sliding one. Pass `--max-train-rows 0` for a genuine expanding
window. Budget roughly:

* XGBoost is offloaded to the GPU with `--xgb-device cuda`, verified at runtime
  by a probe fit.
* Random forest has no GPU path in scikit-learn, so it is parallelised across CPU
  threads (`--rf-jobs N`).

### Measured timings (full expanding window, 1.72M rows, 8 folds + holdout)

| stage | wall clock |
| --- | --- |
| XGBoost 600 rounds, CUDA | ~4 min (8 folds) |
| Random forest 200 trees, 16 threads (`ThreadForestRegressor`) | 100–376 s per fold, ~36 min total |
| Ridge | ~12 s for all 8 folds |
| **whole chain 01→07** | **59 min** |

### The `ThreadForestRegressor` fallback

sklearn's forest calls `joblib.Parallel`, which builds a
`multiprocessing.pool.ThreadPool` whose queue needs a named pipe. Confined /
sandboxed Windows shells deny that with `PermissionError: [WinError 5]`, so
`RandomForestRegressor(n_jobs>1)` **silently runs on one core** — this machine then
sits at 7% total CPU. `plain threading.Thread` and `ThreadPoolExecutor` work fine;
only joblib's pool factory is blocked.

`src/models/thread_forest.py` therefore builds the trees itself with
`ThreadPoolExecutor`, reusing sklearn's own `_generate_sample_indices` and
`DecisionTreeRegressor._fit` with the same seed order as `BaseForest.fit`.
`tests/test_thread_forest.py` asserts `np.array_equal` against
`RandomForestRegressor` on the same data, so the swap cannot change a result.

`04_train_models.py` probes joblib at startup and prints which backend it will
use, switching automatically only when joblib is unavailable. In a normal
terminal the built-in sklearn path is used unchanged.

Results are only written after **every** fold finishes, so an interrupted run
leaves nothing behind but stdout — redirect it to a file and watch it from a
second terminal:

```powershell
.\.venv\Scripts\python.exe -u pipelines\run_chain.py --rf-jobs 16 > reports\tables\run.log 2>&1
# second terminal:
Get-Content reports\tables\run.log -Wait -Tail 30
```

Avoid `2>&1 | Tee-Object -FilePath ...`: PowerShell buffers that pipeline, so the
log file is not created until the process exits.


