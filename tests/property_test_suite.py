# -*- coding: utf-8 -*-
"""Property-based tests: the regimes nobody thought to write down.

The conformance suite fixes fourteen regimes chosen by hand. That is a good
list because each entry came from a real failure, but it is still a list
someone wrote, and the bug that started this whole effort was found by a user
doing something nobody had listed.

Hypothesis generates the data instead. It varies shape, missingness rate and
missingness structure, and when it finds a failure it shrinks the input to the
smallest case that still fails, which is usually more informative than the
original.

What is asserted here are *properties*, statements that must hold for every
input rather than for one example:

  - fitting then predicting yields finite, correctly shaped output, or the
    estimator refuses clearly
  - probabilities lie in [0, 1] and sum to 1
  - predictions do not depend on the order of the rows
  - predicting a subset agrees with predicting everything and then subsetting
  - a fit is reproducible

Run
---
    pytest property_test_suite.py -q

Skips itself when hypothesis is absent, so it is additive rather than a new
hard dependency.
"""
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp                              # noqa: E402

try:
    import MissLearn as ML
except ImportError:                                        # pragma: no cover
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import MissLearn as ML

from MissLearn._conformance import EmptyFeatureError                   # noqa: E402


# A deliberately small, fast set. Property tests earn their keep by running
# many examples, so the per-example cost has to stay low.
REGRESSORS = ["MissLinear", "MissRidgeRegressor", "MissBayesRegressor",
              "MissNeighborsRegressor"]
CLASSIFIERS = ["MissLogistic", "MissRidgeClassifier", "MissBayesClassifier",
               "MissNeighborsClassifier"]
FAST = {"MissLinear": dict(compute_se=False),
        "MissRidgeRegressor": dict(compute_se=False),
        "MissLogistic": dict(compute_se=False),
        "MissRidgeClassifier": dict(compute_se=False)}

SETTINGS = settings(
    max_examples=25,
    deadline=None,                       # fits are slow and vary by machine
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.function_scoped_fixture],
)

# Values are bounded well inside float64. Unbounded floats would generate
# overflow cases that say nothing about missing-data handling, which is the
# subject here.
_values = st.floats(min_value=-1e3, max_value=1e3,
                    allow_nan=False, allow_infinity=False, width=64)


@st.composite
def incomplete_matrix(draw, min_rows=12, max_rows=45, min_cols=1, max_cols=5):
    """An X with missingness, structured the way real data is.

    Three structures are drawn, not one. Scattered missingness is the easy
    case; blockwise is what instrument failures produce and keeps the pattern
    count low; column-biased is what happens when one measurement is expensive.
    """
    n = draw(st.integers(min_rows, max_rows))
    p = draw(st.integers(min_cols, max_cols))
    X = np.asarray(draw(hnp.arrays(np.float64, (n, p), elements=_values)))

    style = draw(st.sampled_from(["scattered", "blockwise", "column_biased"]))
    rate = draw(st.floats(0.0, 0.6))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))

    if style == "scattered":
        X[rng.random((n, p)) < rate] = np.nan
    elif style == "blockwise":
        if p >= 2 and n >= 4:
            c0 = draw(st.integers(0, p - 1))
            X[: n // 2, c0] = np.nan
        else:
            X[rng.random((n, p)) < rate] = np.nan
    else:
        col = draw(st.integers(0, p - 1))
        X[rng.random(n) < min(0.9, rate * 1.5), col] = np.nan

    # A column with no observed values is a refusal by contract, tested in the
    # conformance suite. Leave at least one value so these properties are
    # about the regimes where a fit is expected to be possible.
    for j in range(p):
        if np.all(np.isnan(X[:, j])):
            X[0, j] = 0.0
    return X


def _target(X, task, draw_seed=0):
    rng = np.random.default_rng(draw_seed)
    n, p = X.shape
    y = np.nan_to_num(X, nan=0.0) @ rng.normal(size=p) + rng.normal(scale=0.3, size=n)
    if task == "classification":
        y = (y > np.median(y)).astype(float)
        if len(np.unique(y)) < 2:            # degenerate split
            y[0] = 1.0 - y[0]
    return y


def _fit(name, X, y):
    est = getattr(ML, name)(**FAST.get(name, {}))
    return est.fit(X, y)


# Refusals that are part of the contract rather than failures. Anything else
# propagates and fails the test, which is the point: an unexpected exception
# type is exactly what a property test should surface.
ACCEPTABLE = (EmptyFeatureError, ValueError, np.linalg.LinAlgError)


def _agreement_tol(y, *preds):
    """Absolute tolerance for "these two fits gave the same answer".

    A flat absolute tolerance is the wrong instrument here. The intercept is
    recovered as ``mu_Y - beta @ mu_X``, a difference of two quantities that
    can each be far larger than the result, so when they nearly cancel the
    absolute error left behind is set by *their* magnitude and not by the
    intercept's. On one generated case the two fits agreed to seven
    significant figures on ``coef_`` and ten on ``loglik_``, yet a flat
    ``atol=1e-6`` failed them because the intercept was 0.364 - 0.364 and the
    residue was 1.7e-6.

    Scaling by the data instead keeps the test honest without asking an
    iterative optimiser for ten significant digits. It stays strict where it
    matters: the divergences this test was written to catch differ by 1e-2 to
    1e+1, two to three orders of magnitude above anything this returns.
    """
    scale = 1.0
    for arr in (y,) + preds:
        arr = np.asarray(arr, dtype=float)
        if arr.size and np.isfinite(arr).any():
            scale = max(scale, float(np.nanmax(np.abs(arr[np.isfinite(arr)]))))
    return 1e-6 * scale


@pytest.mark.parametrize("name", REGRESSORS)
@SETTINGS
@given(X=incomplete_matrix())
def test_regressor_predicts_finite_or_refuses(name, X):
    """Either a usable prediction, or a refusal. Never silent nonsense."""
    y = _target(X, "regression")
    try:
        est = _fit(name, X, y)
    except ACCEPTABLE:
        return
    pred = np.asarray(est.predict(X), dtype=float)
    assert pred.shape[0] == X.shape[0]
    assert np.all(np.isfinite(pred)), (
        "fitted and predicted, but the predictions are not finite")


@pytest.mark.parametrize("name", CLASSIFIERS)
@SETTINGS
@given(X=incomplete_matrix(min_cols=2))
def test_classifier_probabilities_are_probabilities(name, X):
    y = _target(X, "classification")
    try:
        est = _fit(name, X, y)
    except ACCEPTABLE:
        return
    P = np.asarray(est.predict_proba(X), dtype=float)
    assert np.all(np.isfinite(P)), "predict_proba is not finite"
    assert P.min() >= -1e-9 and P.max() <= 1 + 1e-9
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", REGRESSORS[:2])
@SETTINGS
@given(X=incomplete_matrix())
def test_row_order_does_not_change_predictions(name, X):
    """Permuting the rows must permute the predictions, nothing more.

    A model whose answer depends on the order rows arrived in is reading
    something it should not. Pattern grouping makes this worth checking here
    specifically: rows are reordered internally by missingness pattern.

    On the tolerance. The defects this test was written for were failures of
    identifiability, not of precision: a constant column gave coefficients of
    -324 and -1.68e6 for the same data, and a near-degenerate one gave
    7.75e+09, diverging by 8.3e-02 relative. What remains after those are
    fixed is the optimiser stopping at slightly different points on a flat
    surface, which on an ill-conditioned generated design leaves agreement to
    five or six significant figures, or about 7e-05 relative. The threshold
    sits between the two: roughly fifteen times above the noise and eighty
    times below the smallest real defect, so it is a gap rather than a line
    drawn to make the suite pass.

    Making this exact rather than tolerant would mean canonicalising the row
    order inside every fit, so that a permutation could not change the
    floating-point summation order at all. That is worth doing and is not
    done here.
    """
    y = _target(X, "regression")
    perm = np.random.default_rng(0).permutation(X.shape[0])
    try:
        a = _fit(name, X, y).predict(X)
        b = _fit(name, X[perm], y[perm]).predict(X[perm])
    except ACCEPTABLE:
        return
    assert np.allclose(np.asarray(a)[perm], np.asarray(b), rtol=1e-3,
                       atol=_agreement_tol(y, a, b), equal_nan=True)


@pytest.mark.parametrize("name", REGRESSORS[:2])
@SETTINGS
@given(X=incomplete_matrix(min_rows=16))
def test_prediction_is_subset_invariant(name, X):
    """Predicting half the rows agrees with predicting all and slicing."""
    y = _target(X, "regression")
    try:
        est = _fit(name, X, y)
        full = np.asarray(est.predict(X), dtype=float)
        half = np.asarray(est.predict(X[::2]), dtype=float)
    except ACCEPTABLE:
        return
    assert np.allclose(full[::2], half, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", REGRESSORS[:2] + CLASSIFIERS[:2])
@SETTINGS
@given(X=incomplete_matrix())
def test_fit_is_reproducible(name, X):
    """The same data must give the same answer twice."""
    task = "classification" if name in CLASSIFIERS else "regression"
    y = _target(X, task)
    try:
        a = _fit(name, X, y).predict(X)
        b = _fit(name, X, y).predict(X)
    except ACCEPTABLE:
        return
    assert np.allclose(np.asarray(a, float), np.asarray(b, float),
                       equal_nan=True)


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-q", "--tb=short"]))
