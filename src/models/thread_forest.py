"""Run sklearn's random forest on a `ThreadPoolExecutor`.

Why this exists
---------------
``RandomForestRegressor(n_jobs>1)`` delegates to ``joblib``, which builds a
``multiprocessing.pool.ThreadPool`` whose queue needs a named pipe. Confined /
sandboxed Windows shells deny that with ``PermissionError: [WinError 5]``, so the
forest silently falls back to a single core -- about 7x slower on this machine
(16 cores, of which the run was using one).

sklearn already anticipates the fix in its own source: *"we prefer the threading
backend as the Cython code for fitting the trees is internally releasing the
Python GIL"*. So the trees parallelise well across threads; only joblib's pool
construction is the obstacle.

This module therefore builds the trees itself, with `concurrent.futures`, while
using sklearn's own primitives for everything that affects the numbers:

* ``_generate_sample_indices`` for the bootstrap draw, and
* ``DecisionTreeRegressor._fit`` with the bootstrap counts passed as
  ``sample_weight`` -- exactly what ``_parallel_build_trees`` does.
* the tree seeds are drawn from ``check_random_state(random_state)`` in the same
  order as ``BaseForest.fit``, so **predictions are identical** to
  ``RandomForestRegressor`` fitted with the same parameters.

Measured``RandomForestRegressor`` equivalence is asserted in
``tests/test_thread_forest.py``. Prediction is averaged over trees; the small
float summation order is the only difference and is bounded by float64 rounding.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

__all__ = ["ThreadForestRegressor", "threads_available", "resolve_workers"]


def resolve_workers(n_jobs: int | None) -> int:
    """Positive worker count; -1 and None mean 'all cores'."""
    if n_jobs is None or n_jobs < 0:
        return max(1, os.cpu_count() or 2)
    return max(1, n_jobs)


def threads_available() -> bool:
    """True if a ThreadPoolExecutor can be created in this environment."""
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(int, [1, 2]))
        return True
    except (PermissionError, OSError):
        return False


def _build_one(tree, X, y, n_samples, n_samples_bootstrap, bootstrap):
    """Fit one tree on a bootstrap resample, mirroring `_parallel_build_trees`."""
    if bootstrap:
        indices = _generate_sample_indices(tree.random_state, n_samples,
                                           n_samples_bootstrap, None)
        sample_weight = np.bincount(indices, minlength=n_samples)
    else:
        sample_weight = None
    tree._fit(X, y, sample_weight=sample_weight, check_input=False)
    return tree


class ThreadForestRegressor(RegressorMixin, BaseEstimator):
    """`RandomForestRegressor` that parallelises tree building with threads.

    Parameters mirror the sklearn estimator for the subset that affects the
    result. ``n_jobs`` counts *threads*, not processes.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 min_samples_leaf: int = 1, max_features=1.0,
                 bootstrap: bool = True, n_jobs: int = -1,
                 random_state: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X, y, **fit_kwargs):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        n_samples_bootstrap = n_samples if self.bootstrap else None

        random_state = check_random_state(self.random_state)
        trees = []
        for _ in range(self.n_estimators):
            # Same seed stream as sklearn's BaseForest.fit.
            seed = random_state.randint(np.iinfo(np.int32).max)
            trees.append(DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed))

        workers = min(resolve_workers(self.n_jobs), max(1, len(trees)))
        if workers == 1:
            self.estimators_ = [
                _build_one(t, X, y, n_samples, n_samples_bootstrap, self.bootstrap)
                for t in trees]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                self.estimators_ = list(pool.map(
                    lambda t: _build_one(t, X, y, n_samples, n_samples_bootstrap,
                                         self.bootstrap),
                    trees))
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        return self

    # -- prediction ------------------------------------------------------- #
    def predict(self, X, n_jobs: int | None = None):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X)
        workers = min(resolve_workers(n_jobs if n_jobs is not None else self.n_jobs),
                      max(1, len(self.estimators_)))

        def one(tree):
            return tree.predict(X)

        if workers == 1:
            preds = [one(t) for t in self.estimators_]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                preds = list(pool.map(one, self.estimators_))
        return np.mean(np.asarray(preds, dtype=np.float64), axis=0)

    # -- sklearn API parity ----------------------------------------------- #
    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimators_")
        all_importances = np.array([t.feature_importances_ for t in self.estimators_])
        return all_importances.mean(axis=0)
