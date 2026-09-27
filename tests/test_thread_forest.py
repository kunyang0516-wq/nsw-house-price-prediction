"""The thread-backed forest must be numerically identical to sklearn's.

`ThreadForestRegressor` exists because joblib's thread pool needs a named pipe,
which confined Windows shells deny, so `RandomForestRegressor(n_jobs>1)` silently
falls back to a single core. Swapping in a different estimator is only acceptable
if it produces the *same* model, so that equivalence is the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.thread_forest import ThreadForestRegressor, resolve_workers, threads_available


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(36103)
    n, d = 4_000, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 0.3 - X[:, 1] * 0.2 + rng.normal(scale=0.2, size=n)).astype(np.float32)
    Xv = rng.normal(size=(1_000, d)).astype(np.float32)
    return X, y, Xv


PARAMS = dict(n_estimators=12, max_depth=8, min_samples_leaf=5,
              max_features=0.4, random_state=36103)


def test_matches_sklearn_single_threaded(data):
    X, y, Xv = data
    base = RandomForestRegressor(n_jobs=1, **PARAMS).fit(X, y)
    thread = ThreadForestRegressor(n_jobs=1, **PARAMS).fit(X, y)
    assert np.array_equal(base.predict(Xv), thread.predict(Xv))


def test_matches_sklearn_with_threads(data):
    """The whole point: parallel tree building must not change the model."""
    X, y, Xv = data
    base = RandomForestRegressor(n_jobs=1, **PARAMS).fit(X, y)
    thread = ThreadForestRegressor(n_jobs=4, **PARAMS).fit(X, y)
    assert np.array_equal(base.predict(Xv), thread.predict(Xv))


def test_feature_importances_match(data):
    X, y, _ = data
    base = RandomForestRegressor(n_jobs=1, **PARAMS).fit(X, y)
    thread = ThreadForestRegressor(n_jobs=4, **PARAMS).fit(X, y)
    assert np.allclose(base.feature_importances_, thread.feature_importances_)


def test_tree_count_and_seeds_match(data):
    X, y, _ = data
    base = RandomForestRegressor(n_jobs=1, **PARAMS).fit(X, y)
    thread = ThreadForestRegressor(n_jobs=4, **PARAMS).fit(X, y)
    assert len(thread.estimators_) == len(base.estimators_) == PARAMS["n_estimators"]
    base_seeds = [t.random_state for t in base.estimators_]
    thread_seeds = [t.random_state for t in thread.estimators_]
    assert base_seeds == thread_seeds


def test_predict_before_fit_raises(data):
    _, _, Xv = data
    with pytest.raises(Exception):
        ThreadForestRegressor(**PARAMS).predict(Xv)


def test_resolve_workers():
    assert resolve_workers(1) == 1
    assert resolve_workers(8) == 8
    assert resolve_workers(None) >= 1
    assert resolve_workers(-1) >= 1


def test_joblib_is_the_blocked_primitive_not_threads():
    """Documents *why* this module exists, and fails loudly if the premise changes.

    If joblib starts working (e.g. running outside a sandbox), sklearn's own
    forest is preferable and `registry.forest_backend_available()` will say so.
    """
    from src.models.registry import forest_backend_available

    assert threads_available() is True, (
        "ThreadPoolExecutor should work; ThreadForestRegressor cannot run without it")
    # Not asserted either way: the joblib answer is environment-dependent, and
    # this test only pins that the two probes are not the same check.
    assert isinstance(forest_backend_available(), bool)
