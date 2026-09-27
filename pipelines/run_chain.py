"""Run the rebuild + retrain chain in ONE process.

Stage selection via argv, so the caller can stop before the long RF stage:

    python _run_all.py                 # all stages
    python _run_all.py 01 02           # only these stage numbers
    python _run_all.py --rf-jobs 1     # override the RF thread count
    python _run_all.py --from 04       # start at stage 04

Single process on purpose: joblib's default loky backend spawns worker
*processes* whose result pipes a confined shell denies
(PermissionError: [WinError 5]). Running every stage in-process keeps joblib on
its thread path, which is allowed.

Stage 06 is omitted: its artifacts are already newer than the rebuilt frame and
it does not depend on the model tables.
"""

from __future__ import annotations

import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_RF_JOBS = "16"

STAGES: list[tuple[str, str]] = [
    ("01", "pipelines/01_build_clean.py"),
    ("02", "pipelines/02_build_features.py"),
    ("04", "pipelines/04_train_models.py"),
    ("05", "pipelines/05_hypothesis_tests.py"),
    ("08", "pipelines/08_feature_value_test.py"),
    ("07", "pipelines/07_make_report.py"),
]

# Stage-specific arguments; tokens `{rf_jobs}` are substituted from --rf-jobs.
STAGE_ARGS: dict[str, list[str]] = {
    "04": ["--min-train-months", "84", "--step-months", "24",
           "--holdout-months", "12", "--max-train-rows", "0",
           "--rf-trees", "200", "--rf-jobs", "{rf_jobs}",
           "--xgb-rounds", "600", "--xgb-device", "cuda"],
    "08": ["--device", "cuda"],
}


def parse() -> tuple[list[str], str, str | None]:
    only: list[str] = []
    rf_jobs = DEFAULT_RF_JOBS
    from_stage: str | None = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--rf-jobs":
            rf_jobs = args[i + 1]
            i += 2
        elif token == "--from":
            from_stage = args[i + 1]
            i += 2
        else:
            only.append(token)
            i += 1
    return only, rf_jobs, from_stage


def main() -> int:
    only, rf_jobs, from_stage = parse()
    selected = [s for s in STAGES if not only or s[0] in only]
    if from_stage:
        order = [s[0] for s in STAGES]
        selected = [s for s in selected if order.index(s[0]) >= order.index(from_stage)]
    if not selected:
        print(f"No stages selected (requested {only}).")
        return 2

    print(f"RF threads: {rf_jobs} | stages: {[s[0] for s in selected]}", flush=True)
    started = time.time()
    for index, (number, script) in enumerate(selected, start=1):
        path = ROOT / script
        if not path.exists():
            print(f"[{index}/{len(selected)}] SKIP {script}: not found", flush=True)
            continue
        extra = [a.replace("{rf_jobs}", rf_jobs) for a in STAGE_ARGS.get(number, [])]
        print(f"\n{'=' * 78}\n[{index}/{len(selected)}] {script}  "
              f"(+{(time.time() - started) / 60:.1f} min elapsed)\n{'=' * 78}", flush=True)
        saved = sys.argv
        sys.argv = [str(path), *extra]
        stage_start = time.time()
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            if code != 0:
                print(f"\n!!! {script} exited {code}; stopping.", flush=True)
                return code
        except BaseException:
            print(f"\n!!! {script} raised; stopping.", flush=True)
            traceback.print_exc()
            return 1
        finally:
            sys.argv = saved
        print(f"--- {script} done in {(time.time() - stage_start) / 60:.1f} min", flush=True)

    print(f"\nSELECTED STAGES COMPLETE in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
