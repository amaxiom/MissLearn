# -*- coding: utf-8 -*-
"""Cross-estimator conformance suite.

Why this exists
---------------
A user found that MissLASSOClassifier raised on high missingness while
MissLogistic and MissLASSORegressor degraded gracefully. The complete-case
seeding fallback existed in the siblings and simply had not been applied to
that one class. The defect that matters is not "a class had a bug", it is
"behaviour diverged between classes that are meant to be interchangeable".

Per-class discipline cannot prevent that, because nothing checks the classes
against each other. scikit-learn and imbalanced-learn avoid the whole category
by running one set of checks over every estimator. This is that mechanism for
MissLearn: every estimator is driven through every degenerate regime, and a
regime that one estimator survives is expected of all of its siblings.

How to read a failure
---------------------
A failure here usually means one of two things.

  1. A genuine divergence: this estimator lacks a guard its siblings have.
     Fix the estimator, do not relax the test.
  2. The regime is legitimately impossible for this family. Then the estimator
     should raise a clear, documented error rather than a LinAlgError from
     deep in a factorisation, and the expectation belongs in KNOWN_FAILURES
     with a reason.

Silent failure is never acceptable. An estimator that fits, predicts, and
returns NaN is worse than one that raises, because nothing tells the caller.
That is checked explicitly.

Run
---
    python conformance_test_suite.py
    pytest conformance_test_suite.py -v
"""
import contextlib
import functools
import inspect
import io
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

try:
    import MissLearn as ML
except ImportError:                                       # pragma: no cover
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import MissLearn as ML

SEED = 0


# ===========================================================================
# The estimator registry: every fittable model, with the settings that make
# the suite fast without changing what is being tested.
# ===========================================================================

REGRESSORS = {
    "MissLinear":             dict(compute_se=False),
    "MissRidgeRegressor":     dict(compute_se=False),
    "MissLASSORegressor":     dict(),
    "MissBayesRegressor":     dict(),
    "MissNeighborsRegressor": dict(),
    "MissSupportRegressor":   dict(),
    "MissGaussianRegressor":  dict(n_restarts=1),
    "MissMixedRegressor":     dict(compute_se=False),
}

CLASSIFIERS = {
    "MissLogistic":            dict(compute_se=False),
    "MissRidgeClassifier":     dict(compute_se=False),
    "MissLASSOClassifier":     dict(),
    "MissBayesClassifier":     dict(),
    "MissNeighborsClassifier": dict(),
    "MissSupportClassifier":   dict(),
    "MissGaussianClassifier":  dict(n_restarts=1),
    "MissMixedClassifier":     dict(compute_se=False),
}

NEEDS_GROUPS = {"MissMixedRegressor", "MissMixedClassifier"}

ALL_ESTIMATORS = [(n, "regression") for n in REGRESSORS] + \
                 [(n, "classification") for n in CLASSIFIERS]


def make(name):
    kw = REGRESSORS.get(name, CLASSIFIERS.get(name, {}))
    return getattr(ML, name)(**kw)


def fit_of(est, name, X, y, g):
    return est.fit(X, y, groups=g) if name in NEEDS_GROUPS else est.fit(X, y)


def predict_of(est, name, X, g):
    return est.predict(X, groups=g) if name in NEEDS_GROUPS else est.predict(X)


# ===========================================================================
# Degenerate regimes. Each returns (X, y, groups).
#
# These are the conditions real data actually arrives in, not adversarial
# constructions: a sensor that failed for a whole campaign gives an all-NaN
# column, a pilot study gives p > n, and a rare outcome gives 5% positives.
# ===========================================================================

def _base(n=120, p=5, task="regression", seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(scale=0.4, size=n)
    if task == "classification":
        y = (y > np.median(y)).astype(float)
    return X, y, rng.integers(0, 4, size=n)


def _holes(X, rate, seed=SEED):
    rng = np.random.default_rng(seed)
    X[rng.random(X.shape) < rate] = np.nan
    return X


def r_clean(task):
    X, y, g = _base(task=task)
    return _holes(X, 0.15), y, g


def _r_missing(rate):
    def f(task):
        X, y, g = _base(task=task)
        return _holes(X, rate), y, g
    return f


def r_no_complete_cases(task):
    """Every row has a hole, so complete-case seeding has nothing to use.

    This is the regime that produced the reported bug.
    """
    X, y, g = _base(task=task)
    rng = np.random.default_rng(SEED)
    for i in range(X.shape[0]):
        X[i, rng.integers(0, X.shape[1])] = np.nan
    return X, y, g


def r_all_nan_column(task):
    X, y, g = _base(task=task)
    X = _holes(X, 0.15)
    X[:, 2] = np.nan
    return X, y, g


def r_constant_feature(task):
    X, y, g = _base(task=task)
    X = _holes(X, 0.15)
    X[:, 1] = 3.0
    return X, y, g


def r_collinear(task):
    X, y, g = _base(task=task)
    X[:, 3] = X[:, 0]
    return _holes(X, 0.15), y, g


def r_wide(task):
    X, y, g = _base(n=25, p=40, task=task)
    return _holes(X, 0.15), y, g


def r_tiny(task):
    X, y, g = _base(n=12, p=3, task=task)
    return _holes(X, 0.15), y, g


def r_single_feature(task):
    X, y, g = _base(p=1, task=task)
    return _holes(X, 0.20), y, g


def r_missing_y(task):
    X, y, g = _base(task=task)
    X = _holes(X, 0.15)
    rng = np.random.default_rng(SEED + 1)
    y = y.copy()
    y[rng.random(len(y)) < 0.20] = np.nan
    return X, y, g


def r_imbalanced(task):
    X, y, g = _base(task=task)
    X = _holes(X, 0.15)
    if task == "classification":
        y = np.zeros_like(y)
        y[:6] = 1.0
    return X, y, g


def r_extreme_scale(task):
    X, y, g = _base(task=task)
    X[:, 0] *= 1e6
    X[:, 1] *= 1e-6
    return _holes(X, 0.15), y, g


REGIMES = {
    "clean_15pct":        r_clean,
    "missing_50pct":      _r_missing(0.50),
    "missing_70pct":      _r_missing(0.70),
    "missing_90pct":      _r_missing(0.90),
    "no_complete_cases":  r_no_complete_cases,
    "all_nan_column":     r_all_nan_column,
    "constant_feature":   r_constant_feature,
    "collinear_features": r_collinear,
    "wide_p_gt_n":        r_wide,
    "tiny_n":             r_tiny,
    "single_feature":     r_single_feature,
    "missing_y":          r_missing_y,
    "imbalanced_5pct":    r_imbalanced,
    "extreme_scale":      r_extreme_scale,
}


# ===========================================================================
# Known divergences, recorded so they are visible rather than tolerated.
#
# Every entry is a work item, not a permanent exemption. Each is a case where
# siblings cope and this estimator does not, which is the exact defect class
# this suite exists to prevent. When one is fixed its test reports XPASS and
# the entry should be deleted.
#
# Established 2026-08-01 by running all 16 estimators over all 14 regimes:
# 9 of 224 cells failed.
# ===========================================================================

# Regimes where the correct behaviour is a uniform, informative refusal
# rather than a fit. A fully missing column has no conditional distribution to
# marginalise over, so every estimator must decline in the same way and say
# which column is at fault.
#
# This regime previously produced four different behaviours across sixteen
# estimators: silent NaN predictions, finite labels hiding NaN probabilities,
# an unhelpful NaN message, and three that coped. It is now a single contract,
# enforced here.
REFUSAL_REGIMES = {
    "all_nan_column": "EmptyFeatureError",
}


KNOWN_FAILURES = {
    # MissLinear declines p >= n deliberately, and this entry records a
    # decision rather than a defect.
    #
    # It estimates the full joint MVN of [y, X]: 902 free parameters at p = 40
    # with 25 rows. The likelihood is unidentified, so there is no fit to be
    # had. Conditioning the seed was tried and rejected: it removed the
    # LinAlgError only by substituting a slow optimisation returning
    # coefficients that look ordinary and mean nothing, which is the trade the
    # fully-missing-column path exists to prevent.
    #
    # The siblings genuinely cope rather than merely surviving: the penalized
    # families shrink explicitly and are identified here. So this divergence
    # is real, intended, and the right behaviour, and the estimator now says
    # so in one sentence naming the alternatives instead of raising from
    # inside a factorisation.
    ("MissLinear", "wide_p_gt_n"):
        "Refuses by design: the joint MVN is unidentified at p >= n. Raises "
        "ValueError naming MissRidgeRegressor and MissLASSORegressor, which "
        "shrink and are defined in this regime.",
    # The ten all_nan_column entries that stood here are gone because the
    # regime is fixed, not because the expectation was relaxed. Every
    # estimator now raises EmptyFeatureError naming the column; see
    # REFUSAL_REGIMES above and MissLearn/_conformance.py.
}

# Divergences that appear only in the probability path.
#
# These are kept separate from KNOWN_FAILURES deliberately. All four estimators
# below return perfectly finite hard labels under this regime, so a check that
# only exercised predict declared them healthy; the first parity matrix run did
# exactly that and missed them. Keying the exemption to the specific test keeps
# the fit/predict expectation strict, which is what makes an XPASS meaningful.
KNOWN_PROBA_FAILURES = {
    # Empty, and worth recording why.
    #
    # Four entries lived here: MissLogistic, MissRidgeClassifier,
    # MissLASSOClassifier and MissMixedClassifier returned finite class labels
    # whose predict_proba was entirely NaN on a fully missing column. The label
    # path concealed the failure completely, which is what made it the most
    # dangerous divergence found. That regime now refuses uniformly, so the
    # exemptions are deleted rather than carried.
    #
    # Keep this table. It is separate from KNOWN_FAILURES so that a divergence
    # in the probability path can be declared without weakening the fit and
    # predict expectation, and the first sweep of this library missed exactly
    # that class of defect by checking predict alone.
}


def _maybe_xfail(name, regime, table=None):
    key = (name, regime)
    for tbl in ((table,) if table is not None else (KNOWN_FAILURES,)):
        if key in tbl:
            pytest.xfail("known divergence: %s" % tbl[key])


# ===========================================================================
# 1. The core conformance check, over the full cross product
# ===========================================================================

@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_degenerate_regime(name, task, regime):
    """Fit and predict must succeed, and predictions must be usable.

    'Usable' means finite and correctly shaped. An estimator that returns NaN
    predictions has failed even though it raised nothing.
    """
    _maybe_xfail(name, regime)
    X, y, g = REGIMES[regime](task)
    est = make(name)

    if regime in REFUSAL_REGIMES:
        # The contract here is a clear refusal, identical across estimators.
        expected = REFUSAL_REGIMES[regime]
        with pytest.raises(Exception) as excinfo:
            fit_of(est, name, X, y, g)
        assert type(excinfo.value).__name__ == expected, (
            "expected %s, got %s: %s" % (expected,
                                         type(excinfo.value).__name__,
                                         excinfo.value))
        assert "column" in str(excinfo.value), (
            "the refusal must name the offending column, or the user cannot "
            "act on it")
        return

    fitted = fit_of(est, name, X, y, g)
    assert fitted is est, "fit must return self (scikit-learn contract)"

    pred = np.asarray(predict_of(est, name, X, g), dtype=float)
    assert pred.shape[0] == X.shape[0], (
        "predict returned %d rows for %d input rows" % (pred.shape[0], X.shape[0]))
    assert np.all(np.isfinite(pred)), (
        "predictions contain NaN or inf: silent failure is worse than raising")


# ===========================================================================
# 2. Probabilities must be probabilities
# ===========================================================================

@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("name", sorted(CLASSIFIERS))
def test_predict_proba_valid(name, regime):
    # Both tables apply here: an estimator that cannot fit at all obviously
    # cannot produce probabilities either.
    _maybe_xfail(name, regime, KNOWN_FAILURES)
    _maybe_xfail(name, regime, KNOWN_PROBA_FAILURES)
    if regime in REFUSAL_REGIMES:
        # Not a skip. There is no fitted model to take probabilities from, but
        # the refusal is itself the contract and is worth asserting: an
        # estimator that declines on fit must also decline on predict_proba
        # rather than answering from a half-initialised state. Skipping here
        # left eight cells asserting nothing at all.
        est = make(name)
        with pytest.raises(Exception):
            fit_of(est, name, *REGIMES[regime]("classification"))
        with pytest.raises(Exception):
            est.predict_proba(REGIMES[regime]("classification")[0])
        return
    X, y, g = REGIMES[regime]("classification")
    est = fit_of(make(name), name, X, y, g)
    if not hasattr(est, "predict_proba"):
        pytest.skip("%s exposes no predict_proba" % name)

    P = np.asarray(est.predict_proba(X), dtype=float)
    assert np.all(np.isfinite(P)), "probabilities contain NaN or inf"
    assert P.min() >= -1e-9 and P.max() <= 1 + 1e-9, "probabilities outside [0, 1]"
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "rows do not sum to 1"


# ===========================================================================
# 3. The scikit-learn estimator contract, applied to every class
# ===========================================================================

@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_sklearn_contract(name, task):
    """get_params / set_params / clone, and the fitted-attribute conventions."""
    from sklearn.base import clone

    est = make(name)
    params = est.get_params()
    assert isinstance(params, dict)

    cloned = clone(est)
    assert cloned.get_params() == params, "clone did not preserve parameters"

    est.set_params(**params)                      # round-trip must be accepted

    X, y, g = REGIMES["clean_15pct"](task)
    fit_of(est, name, X, y, g)

    assert est.n_features_in_ == X.shape[1]
    assert hasattr(est, "feature_names_in_")
    for attr in ("n_samples_fit_", "n_missing_X_", "missing_rate_X_",
                 "n_complete_", "n_partial_", "copula_used_"):
        assert hasattr(est, attr), "missing common fitted attribute %s" % attr
    if task == "classification":
        assert hasattr(est, "classes_")


# ===========================================================================
# 4. Determinism
# ===========================================================================

@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_deterministic(name, task):
    """The same data must give the same answer twice.

    Several families use random restarts or bootstrap draws internally. Those
    are seeded, so a difference here means an unseeded source of randomness,
    which makes every published number unreproducible.

    Compared on the continuous surface and bit for bit, not on ``predict``
    under a tolerance, because that is how this check missed a real one.
    MissLASSOClassifier seeded from liblinear without a random_state and drifted
    by 1.6e-04 in its coefficients between two fits of the same matrix, but a
    classifier's ``predict`` returns labels, and the labels were identical, so
    the comparison never saw the drift. sklearn's ``check_fit_idempotent``
    caught it on ``decision_function``, which this suite was not looking at.

    Bit-exact rather than allclose because there is no tolerance at which a
    reproducible fit is acceptable-but-different. Every estimator here meets
    it, so the strict form costs nothing and states the real requirement.
    """
    X, y, g = REGIMES["clean_15pct"](task)
    a = fit_of(make(name), name, X, y, g)
    b = fit_of(make(name), name, X, y, g)

    surfaces = [("predict", np.asarray(predict_of(a, name, X, g), float),
                 np.asarray(predict_of(b, name, X, g), float))]
    for method in ("predict_proba", "decision_function"):
        if hasattr(a, method):
            surfaces.append((method,
                             np.asarray(getattr(a, method)(X), float),
                             np.asarray(getattr(b, method)(X), float)))
    if hasattr(a, "coef_"):
        surfaces.append(("coef_", np.asarray(a.coef_, float),
                         np.asarray(b.coef_, float)))

    drifted = ["%s by %.3e" % (what, float(np.nanmax(np.abs(u - v))))
               for what, u, v in surfaces if not np.array_equal(u, v)]
    assert not drifted, (
        "two identical fits disagree: %s. A difference this small still means "
        "an unseeded source of randomness, and it reaches published numbers."
        % "; ".join(drifted))


# ---------------------------------------------------------------------------
# Row order is not information
# ---------------------------------------------------------------------------

#: Regimes the permutation check runs on. It costs an extra fit wherever it
#: runs, so it is scoped rather than applied to all fourteen: the control,
#: because an estimator that is not stable on ordinary data is not stable
#: anywhere, and the degenerate ones, which is where the defects this exists
#: for actually lived.
ORDER_REGIMES = ["clean_15pct", "constant_feature", "no_complete_cases",
                 "extreme_scale"]


@pytest.mark.parametrize("regime",
                         [r for r in ORDER_REGIMES if r in REGIMES])
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_row_order_does_not_change_predictions(name, task, regime):
    """Permuting the rows must permute the predictions and change nothing else.

    This is the axis the suite was missing, and its absence is why a whole
    family of defects lived here undetected. Every other test looks at a
    single fit, and a large finite number passes them however arbitrary it
    is. ``constant_feature`` in particular passed throughout while
    MissLinear was returning a coefficient of -324 on one row order and
    -1.68e6 on another, because nothing ever fitted the same data twice.

    Groups are permuted with their rows rather than regenerated. Rebuilding a
    per-row array from reordered data assigns each row a different subject,
    which compares two different problems and reports a false failure.
    """
    X, y, g = REGIMES[regime](task)
    try:
        base = predict_of(fit_of(make(name), name, X, y, g), name, X, g)
    except Exception:
        pytest.skip("refuses this regime; covered by test_degenerate_regime")

    rng = np.random.default_rng(0)
    perm = rng.permutation(X.shape[0])
    gp = None if g is None else np.asarray(g)[perm]
    try:
        shuffled = predict_of(
            fit_of(make(name), name, X[perm], np.asarray(y)[perm], gp),
            name, X[perm], gp)
    except Exception as exc:
        pytest.fail("fitted the rows in one order and refused the same rows "
                    "in another (%s)" % type(exc).__name__)

    a = np.asarray(base, dtype=float)[perm]
    b = np.asarray(shuffled, dtype=float)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True), (
        "row order changed the predictions by %.3e; the same data in a "
        "different order is the same data"
        % float(np.nanmax(np.abs(a - b))))


# ===========================================================================
# 5. Feature names propagate, so reports name the measurement
# ===========================================================================

@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_feature_names_from_dataframe(name, task):
    pd = pytest.importorskip("pandas")
    X, y, g = REGIMES["clean_15pct"](task)
    cols = ["feat_%d" % j for j in range(X.shape[1])]
    est = fit_of(make(name), name, pd.DataFrame(X, columns=cols), y, g)
    assert list(est.feature_names_in_) == cols, (
        "column names did not reach feature_names_in_, so warnings and "
        "summaries will say X4 instead of naming the measurement")


# ===========================================================================
# 6. The guarantee itself: no NEW sibling divergence
#
# The tests above pin individual cells. This one states the property that the
# suite exists to defend, so that adding an estimator or a regime cannot
# quietly reintroduce the original defect.
# ===========================================================================

@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_no_undeclared_sibling_divergence(regime):
    """Within a task, a regime some estimators survive is expected of all.

    Any divergence must be declared in KNOWN_FAILURES with a reason. An
    undeclared one fails here, which is the check that would have caught the
    reported MissLASSOClassifier bug on the commit that introduced it.
    """
    for task, family in (("regression", REGRESSORS),
                         ("classification", CLASSIFIERS)):
        X, y, g = REGIMES[regime](task)
        if regime in REFUSAL_REGIMES:
            continue          # uniform refusal is checked by its own test
        coped, diverged = [], []
        for name in sorted(family):
            if (name, regime) in KNOWN_FAILURES:
                continue
            declared_proba = (name, regime) in KNOWN_PROBA_FAILURES
            try:
                est = fit_of(make(name), name, X, y, g)
                pred = np.asarray(predict_of(est, name, X, g), dtype=float)
                ok = pred.shape[0] == X.shape[0] and np.all(np.isfinite(pred))
                reason = "non-finite or wrongly shaped predictions"
                # The probability path has to be checked here as well. Four
                # classifiers return finite labels while predict_proba is all
                # NaN, so a predict-only sweep reports them healthy.
                if ok and not declared_proba and hasattr(est, "predict_proba"):
                    P = np.asarray(est.predict_proba(X), dtype=float)
                    if not np.all(np.isfinite(P)):
                        ok, reason = False, "predict_proba returns NaN"
                    elif not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
                        ok, reason = False, "predict_proba rows do not sum to 1"
            except Exception as exc:
                ok, reason = False, "%s: %s" % (type(exc).__name__,
                                                str(exc).split("\n")[0][:70])
            (coped if ok else diverged).append(
                name if ok else "%s (%s)" % (name, reason))

        assert not (coped and diverged), (
            "undeclared sibling divergence in %s under %s.\n"
            "  coped:     %s\n"
            "  diverged:  %s\n"
            "Either fix the estimator, or declare it in KNOWN_FAILURES with a "
            "reason." % (task, regime, ", ".join(coped), "; ".join(diverged)))



# ===========================================================================
# 7. The invariant axes
#
# Sections 1 to 6 grew one at a time, each in response to a defect that had
# already shipped. The four axes below are the techniques that found the
# August 2026 defects, written down so they run on every commit instead of
# when somebody thinks to probe by hand. Each one caught something real:
#
#   options    four options accepted a misspelling and silently selected the
#              other branch, copula on all sixteen estimators among them
#   accessors  MissBayes returned exactly 1/p for every feature when one
#              effect size came out nan, which sums to 1 and reads as a result
#   numerics   an underflow crashed MissMixed and a nan reached the MissLASSO
#              gradient, both at boundaries no test visited
#   accuracy   nothing anywhere graded whether a fit predicted better than
#              the mean, so a divergence to r-squared -6.7e6 passed everything
#
# They share one fit per (estimator, regime) through _fitted below, so adding
# an axis costs assertions rather than optimisations.
# ===========================================================================

#: Regimes the invariant axes run on. Scoped like ORDER_REGIMES and for the
#: same reason: each axis costs a fit per cell. The control is here because an
#: estimator that misbehaves on ordinary data misbehaves everywhere, and the
#: rest are the four regimes where the August defects actually lived.
INVARIANT_REGIMES = ["clean_15pct", "constant_feature", "collinear_features",
                     "extreme_scale", "no_complete_cases"]


@functools.lru_cache(maxsize=None)
def _fitted(name, task, regime):
    """One fit per (estimator, regime), shared by every axis below.

    The axes are read-only by construction: they inspect attributes and call
    predict, summary and the accessors, none of which may mutate the
    estimator. Anything that needs a fresh fit, such as the determinism
    checks, must not use this.
    """
    X, y, g = REGIMES[regime](task)
    return fit_of(make(name), name, X, y, g), X, y, g


def _cell(name, task, regime):
    """Fetch the shared fit, skipping cells whose refusal is covered elsewhere."""
    if (name, regime) in KNOWN_FAILURES:
        pytest.skip("declared in KNOWN_FAILURES")
    try:
        return _fitted(name, task, regime)
    except Exception as exc:
        pytest.skip("refuses this regime (%s); covered by "
                    "test_degenerate_regime" % type(exc).__name__)


# ---------------------------------------------------------------------------
# 7a. Options: a misspelling must not select a different model
# ---------------------------------------------------------------------------

def option_fixture(task):
    """A deliberately small matrix, used only by the option axes.

    These axes ask whether an option is wired up, which is a question about
    branches rather than about statistics, so they do not need the shared
    n=120. The Gaussian Process families are O(n^3) and the option axis fits
    each of them once per documented value: at n=120 that was 282 s for
    MissGaussianClassifier alone and it tripled the whole suite. At n=40 the
    same branches are exercised for a fortieth of the time.
    """
    X, y, g = _base(n=40, p=4, task=task)
    return _holes(X, 0.15), y, g


#: (legal values, one misspelling that must be refused). A None misspelling
#: means the parameter is a bool, where there is nothing to misspell.
_COPULA = (("auto", True, False), "atuo")
_METHOD = (("L-BFGS-B", "TNC", "Powell", "SLSQP"), "not_a_solver")
_WEIGHTS = (("distance", "uniform"), "unifrom")
_METRIC = (("euclidean", "mahalanobis"), "mahalnobis")
_STRUCTURE = (("full", "naive"), "nieve")
_GP_KERNEL = (("rbf", "matern52", "matern12"), "matern32")
_SVM_KERNEL = (("rbf", "linear", "poly"), "sigmoid")
_GAMMA = (("scale", "auto"), "Scale")
_ARD = ((True, False), None)
_WARM_START = ((True, False), None)


def options_of(name):
    """The enumerable options of one estimator, resolved by family.

    ``kernel`` means different things in _gp and _svm, so the table is built
    per estimator rather than keyed by parameter name alone.
    """
    opts = {"copula": _COPULA}
    if "warm_start" in inspect.signature(getattr(ML, name).__init__).parameters:
        opts["warm_start"] = _WARM_START
    if "LASSO" in name:
        opts["method"] = _METHOD
    if "Neighbors" in name:
        opts["weights"] = _WEIGHTS
        opts["metric"] = _METRIC
    if "Bayes" in name:
        opts["structure"] = _STRUCTURE
    if "Gaussian" in name:
        opts["kernel"] = _GP_KERNEL
        opts["ard"] = _ARD
    if "Support" in name:
        opts["kernel"] = _SVM_KERNEL
        opts["gamma"] = _GAMMA
    return opts


#: Parameters deliberately outside the option matrix, with the reason. Like
#: KNOWN_FAILURES these are work items, not permanent exemptions.
UNCHECKED_OPTIONS = {
    "compute_se": "a bool whose two values are both legal and are covered by "
                  "the registry, which fits half the estimators with it off",
    "method": "restricted to bounds-capable solvers in MissLASSO only. The "
              "other four families that take it pass it straight to scipy, so "
              "an unusable solver is accepted there and the misspelling comes "
              "back in scipy's words rather than MissLearn's. Widening the "
              "check to them is a behaviour change on estimators that "
              "currently accept BFGS and work.",
    "shrinkage": "'auto' or a float in [0, 1], so it is not an enumeration; "
                 "the float path is checked by the unit suite",
    "task": "set by the library from the estimator class, never by the caller",
}


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_every_string_option_is_in_the_matrix(name, task):
    """A new option must be enumerated here or exempted with a reason.

    This is what makes the axis systematic rather than a snapshot. The four
    silent-fallback defects were all in options that existed for months; the
    check that would have caught them is one that notices an option nobody
    enumerated.
    """
    known = set(options_of(name)) | set(UNCHECKED_OPTIONS)
    params = inspect.signature(getattr(ML, name).__init__).parameters
    unregistered = sorted(
        p for p, v in params.items()
        if p not in ("self", "kwargs")
        and isinstance(v.default, (str, bool))
        and p not in known)
    assert not unregistered, (
        "%s takes %s, which no test enumerates. Add it to options_of() with "
        "its legal values and a misspelling, or to UNCHECKED_OPTIONS with the "
        "reason it cannot be enumerated." % (name, unregistered))


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_documented_option_values_all_fit(name, task):
    """Every value the docstring offers must actually produce a fit."""
    X, y, g = option_fixture(task)
    for param, (values, _) in sorted(options_of(name).items()):
        for value in values:
            kw = dict(REGRESSORS.get(name, CLASSIFIERS.get(name, {})))
            kw[param] = value
            est = getattr(ML, name)(**kw)
            try:
                pred = predict_of(fit_of(est, name, X, y, g), name, X, g)
            except Exception as exc:
                pytest.fail("%s=%r is documented but %s: %s"
                            % (param, value, type(exc).__name__, exc))
            assert np.all(np.isfinite(np.asarray(pred, float))), (
                "%s=%r fitted but predicts non-finitely" % (param, value))


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_a_misspelled_option_is_refused(name, task):
    """The defect this axis exists for.

    These options are consumed as ``== 'mahalanobis'`` or ``== 'full'``, so an
    unrecognised value takes the other branch. Nothing is unset and nothing is
    missing, so nothing complains, and the user gets a different model from
    the one they asked for. On the fixtures in the unit suite the differences
    reached 2.9 in prediction units, and for ``structure`` a misspelling of
    ``'full'`` selected naive independence, the opposite model.
    """
    X, y, g = option_fixture(task)
    for param, (_, bad) in sorted(options_of(name).items()):
        if bad is None:
            continue                      # a bool cannot be misspelled
        kw = dict(REGRESSORS.get(name, CLASSIFIERS.get(name, {})))
        kw[param] = bad
        with pytest.raises(ValueError, match=param):
            fit_of(getattr(ML, name)(**kw), name, X, y, g)


#: Options whose values are expected to agree across two fresh fits, with the
#: reason. Everything else must demonstrably change the model.
NO_DIFFERENCE_EXPECTED = {
    "warm_start": "reuses the solution of the previous fit on the same "
                  "estimator object, so two fresh fits have nothing to differ "
                  "about, and it is adopted only when it improves on the EM "
                  "optimum, which on ordinary data it does not. Both values "
                  "are still required to fit, which the check above does.",
}


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_options_are_not_silently_equivalent(name, task):
    """An option that never changes the answer is not an option.

    Written the other way round from the refusal check on purpose. If a
    refactor makes one branch unreachable, the refusal check still passes and
    this one does not.
    """
    X, y, g = option_fixture(task)

    def fit_with(param, value):
        """The continuous output, since labels hide a difference that is real.

        Comparing predict here reported method and gamma as doing nothing on
        two classifiers. They move the decision function by 5.1e-03 and
        1.0e-02; the labels simply did not cross a boundary. That is the same
        mistake test_deterministic used to make in the other direction.
        """
        kw = dict(REGRESSORS.get(name, CLASSIFIERS.get(name, {})))
        kw[param] = value
        est = fit_of(getattr(ML, name)(**kw), name, X, y, g)
        for method in ("decision_function", "predict_proba"):
            if hasattr(est, method):
                return np.asarray(getattr(est, method)(X), float)
        return np.asarray(predict_of(est, name, X, g), float)

    for param, (values, _) in sorted(options_of(name).items()):
        if len(values) < 2 or param in NO_DIFFERENCE_EXPECTED:
            continue
        first = fit_with(param, values[0])
        assert any(not np.array_equal(first, fit_with(param, v))
                   for v in values[1:]), (
            "%s makes no difference to %s on ordinary data, so either the "
            "branch is unreachable or the option is not wired up"
            % (param, name))


# ---------------------------------------------------------------------------
# 7b. Accessors: what a user reads after fitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_importances_are_a_distribution(name, task, regime):
    """feature_importances_ must be one number per feature, summing to 1."""
    est, X, _, _ = _cell(name, task, regime)
    imp = np.asarray(est.feature_importances_, dtype=float)
    assert imp.shape == (X.shape[1],), "wrong shape %s" % (imp.shape,)
    assert np.all(np.isfinite(imp)), "non-finite importances"
    assert np.all(imp >= 0), "negative importance"
    assert abs(imp.sum() - 1.0) < 1e-8, "sums to %.8f" % imp.sum()


@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_importances_are_not_the_placeholder(name, task, regime):
    """Exact uniformity is the signature of a fallback, not of a ranking.

    Both MissBayes accessors end in ``w / total if total > 0 else
    ones / len(w)``, which is a reasonable answer when every weight is
    genuinely zero and a lie when one weight is nan, because nan fails the
    ``> 0`` test too. That is how a constant column turned
    [0.78, 0.05, 0, 0.18] into 1/p four times over, with no nan left in the
    output to show for it.

    Three families report the fraction of rows in which each feature is
    observed rather than a predictive contribution, which their docstrings
    state. Their spread is small, 0.02 on these fixtures, but never exactly
    zero, so this check separates them from the fallback without needing to
    know which kind of measure it is looking at.
    """
    est, X, _, _ = _cell(name, task, regime)
    if X.shape[1] < 2:
        pytest.skip("a single feature is uniform by definition")
    imp = np.asarray(est.feature_importances_, dtype=float)
    assert not np.all(imp == imp[0]), (
        "every feature scored exactly 1/p, which is what the uniform "
        "fallback returns when a weight came out nan")


@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task",
                         [(n, t) for n, t in ALL_ESTIMATORS
                          if t == "regression"],
                         ids=[n for n, t in ALL_ESTIMATORS
                              if t == "regression"])
def test_prediction_interval_contains_the_prediction(name, task, regime):
    """An interval that excludes its own point estimate is not an interval."""
    est, X, _, g = _cell(name, task, regime)
    if not hasattr(est, "predict_interval"):
        pytest.skip("no predict_interval")
    kw = {"groups": g} if name in NEEDS_GROUPS else {}
    lo, hi = est.predict_interval(X, **kw)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    point = np.asarray(predict_of(est, name, X, g), dtype=float)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), \
        "non-finite interval"
    assert np.all(hi >= lo), "upper bound below lower bound"
    inside = (point >= lo - 1e-8) & (point <= hi + 1e-8)
    assert np.all(inside), ("the point estimate lies outside its own interval "
                            "on %d of %d rows" % (int((~inside).sum()),
                                                  len(point)))


@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_summary_names_every_feature_and_shows_no_nan(name, task, regime):
    """summary() is the report a reader trusts, so it must not print nan.

    MissLASSO printed four columns of literal nan on every row, under
    headings saying p_value and CI, because it does not compute standard
    errors by default. A note above the table explained it, and a table
    reading nan still looks like a fit that failed.
    """
    est, _, _, _ = _cell(name, task, regime)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        est.summary()
    text = buf.getvalue()
    assert text.strip(), "summary() printed nothing"
    assert "nan" not in text.lower(), (
        "summary() prints nan, which reads as a failed fit rather than as a "
        "quantity that was not computed")
    unnamed = [str(n) for n in est.feature_names_in_ if str(n) not in text]
    assert not unnamed, "summary() never names %s" % unnamed


# ---------------------------------------------------------------------------
# 7c. Numerical hygiene
# ---------------------------------------------------------------------------

#: Cells that emit a numerical warning, with what it is and why it is left.
#: Work items, like KNOWN_FAILURES.
KNOWN_NUMERICAL_WARNINGS = {
    ("MissLinear", "collinear_features"):
        "scipy's finite-difference gradient subtracts two objective values "
        "where the objective correctly returned inf for an out-of-domain "
        "parameter, so inf - inf gives nan inside _numdiff. Predictions and "
        "loglik_ are finite and converged_ is True. Removing the warning "
        "itself means returning a large finite penalty instead of inf from "
        "the objective, which changes the optimisation surface of a working "
        "estimator, so it is left."
        "\n\n"
        "The cell is clean again, but it was not, and the reason is worth "
        "keeping. This entry used to read 'measured consequence: none', "
        "citing that with compute_se=True every standard error is finite. "
        "That was the wrong test: finite is not the same as correct. Given "
        "two identical columns, whose individual coefficients are "
        "unidentified while their sum is not, MissLinear reported a standard "
        "error of exactly 0.0000 on them in 7 to 16 draws out of 20, across n "
        "from 40 to 300 and missingness from 0 to 25 per cent, and printed a "
        "confidence interval of zero width around a coefficient of -223.20. "
        "MissRidgeRegressor was false in 0 of 20 in every one of those twelve "
        "configurations, because its penalty genuinely determines the split."
        "\n\n"
        "Two things about the diagnosis were wrong on the first pass and are "
        "recorded so they are not repeated. The z statistics and p values "
        "were never affected: z_stats_ and pvalues_ already derive from "
        "np.where(se > 0, se, np.nan) and were correctly NaN throughout, so "
        "only se_ and the interval built from it ever lied. And psd_jitter "
        "was not the cause, though it looked like it: dropping its floor from "
        "1e-4 to 1e-10 changes the coefficients but leaves the standard "
        "errors at 0.000 throughout."
        "\n\n"
        "The cause was np.sqrt(np.maximum(np.diag(Var), 0.0)) in the standard "
        "error paths. A delta-method or inverse-Hessian variance goes "
        "negative when the Hessian is not positive definite or the Jacobian "
        "is ill conditioned, which is a computation reporting its own "
        "failure; flooring it at zero turned that into a claim of exact "
        "knowledge. All eight such sites now route through "
        "standard_errors_from_variance, which returns NaN instead, and the "
        "rate is 0 of 20 in all twelve configurations. Identified "
        "coefficients in the same fit keep their standard errors in 8 to 14 "
        "draws out of 15; a negative diagonal entry is not always confined to "
        "the unidentified pair, and where it is not, the entry is withheld "
        "wherever it sits.",
}


@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_fit_emits_no_numerical_warning(name, task, regime):
    """Overflow, underflow and invalid-value warnings are defects in waiting.

    Each of the August numerical defects announced itself as a RuntimeWarning
    before it announced itself as a failure: tau_sq underflowing to zero
    before it raised ZeroDivisionError, and v_g squared overflowing before it
    wrote nan into a gradient. Warnings are not printed by default in a
    notebook, so nobody saw them. Here they fail.

    This needs its own fit, since warnings are raised during the fit and the
    shared one has already happened.
    """
    if (name, regime) in KNOWN_FAILURES:
        pytest.skip("declared in KNOWN_FAILURES")
    X, y, g = REGIMES[regime](task)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            est = fit_of(make(name), name, X, y, g)
            predict_of(est, name, X, g)
        except Exception as exc:
            pytest.skip("refuses this regime (%s)" % type(exc).__name__)
    numerical = sorted({str(w.message) for w in caught
                        if issubclass(w.category, RuntimeWarning)})
    declared = KNOWN_NUMERICAL_WARNINGS.get((name, regime))
    if declared:
        if not numerical:
            # Not necessarily stale. Several declared warnings are raised
            # inside a dependency rather than here, so whether the cell warns
            # depends on the installed version. The MissLinear collinear cell
            # is one: the warning comes from scipy's finite-difference
            # gradient, and it appears on scipy 1.17 and not on 1.13, which is
            # what Python 3.9 resolves. Deleting the entry would then make
            # every newer-scipy environment fail the undeclared-warning
            # assertion below instead, so one static declaration cannot
            # satisfy both while the behaviour depends on scipy.
            #
            # This is a skip rather than a failure, and it names the versions
            # so the -ra summary says where it was silent. Retiring an entry
            # means confirming the warning is gone across the supported
            # dependency range, not in one run.
            import numpy as _np
            import scipy as _sp
            pytest.skip(
                "declared in KNOWN_NUMERICAL_WARNINGS and silent here "
                "(numpy %s, scipy %s). Confirm across the supported range "
                "before deleting the entry." % (_np.__version__,
                                                _sp.__version__))
        return
    assert not numerical, (
        "fit emitted %s. Either guard the boundary or record the cell in "
        "KNOWN_NUMERICAL_WARNINGS with the measured consequence."
        % "; ".join(m[:70] for m in numerical))


# ---------------------------------------------------------------------------
# 7d. Accuracy: a fit that explains nothing is not a fit
# ---------------------------------------------------------------------------

#: Cells that fall below the floor, with the diagnosis. Work items.
#:
#: Empty since 16 August 2026. The one entry it held, MissMixedRegressor under
#: extreme_scale at r-squared -6.7e6, was the defect this axis found on its
#: first run and it is fixed: _utils.feature_scale no longer measures a
#: column's spread against the widest column in the design, which classed an
#: ordinary Gaussian column scaled by 1e-06 as degenerate. That estimator now
#: reaches 0.8373 on the same data, and MissRidgeRegressor and
#: MissLASSORegressor improved from 0.7334 to 0.846 alongside it.
KNOWN_ACCURACY_FAILURES = {}


def _trivial_baseline_score(y, pred, task):
    """Score against the answer that ignores X entirely."""
    observed = ~np.isnan(np.asarray(y, dtype=float))
    y_obs = np.asarray(y, dtype=float)[observed]
    p_obs = np.asarray(pred, dtype=float)[observed]
    if task == "regression":
        spread = ((y_obs - y_obs.mean()) ** 2).sum()
        if spread <= 0:
            return None, "r2"
        return 1.0 - ((y_obs - p_obs) ** 2).sum() / spread, "r2"
    majority = max(np.mean(y_obs == v) for v in np.unique(y_obs))
    return (p_obs == y_obs).mean() - majority, "lift"


#: How far below the trivial baseline a fit may fall before it is broken
#: rather than merely poor. Deliberately generous: measured over all fourteen
#: regimes, every cell that is not diverging sits above -0.16 r-squared and
#: -0.04 lift, and the one that is diverging sits at -6.7e6. Nothing lives in
#: between, so the floor does not need to be finely tuned to separate them.
R2_FLOOR = -1.0
LIFT_FLOOR = -0.25


@pytest.mark.parametrize("regime", INVARIANT_REGIMES)
@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_fit_is_not_worse_than_ignoring_the_features(name, task, regime):
    """Nothing anywhere graded whether a fit predicts better than the mean.

    Every other test here asks whether the output is well formed: finite,
    correctly shaped, summing to one, the same twice. A model that has
    diverged passes all of that, because 2.6e04 is a perfectly finite number.
    MissMixedRegressor reached r-squared -6.7e6 under extreme_scale while
    reporting converged_ True, and the suite called it healthy.

    The floor is not a performance target. It asks only that the fit is not
    arbitrarily worse than predicting the mean, which is the weakest possible
    statement that still separates a diverged fit from a hard problem.
    """
    est, X, y, g = _cell(name, task, regime)
    pred = np.asarray(predict_of(est, name, X, g), dtype=float)
    if not np.all(np.isfinite(pred)):
        pytest.skip("non-finite predictions; covered by test_degenerate_regime")
    score, metric = _trivial_baseline_score(y, pred, task)
    if score is None:
        pytest.skip("target has no spread to explain")

    floor = R2_FLOOR if metric == "r2" else LIFT_FLOOR
    declared = KNOWN_ACCURACY_FAILURES.get((name, regime))
    if declared:
        if score >= floor:
            pytest.fail("declared in KNOWN_ACCURACY_FAILURES but now scores "
                        "%s=%.4f, above the floor; delete the entry"
                        % (metric, score))
        return
    assert score >= floor, (
        "%s=%.4g under %s, below the floor of %.2f. A fit this far beneath "
        "the trivial baseline has diverged rather than found the problem "
        "hard; check the scale handling before widening the floor."
        % (metric, score, regime, floor))


# ---------------------------------------------------------------------------
# 7e. Siblings that differ only in their penalty must agree
# ---------------------------------------------------------------------------

#: Estimators that are the same model fitted the same way, differing only in
#: how the coefficients are penalised. A large gap between them on identical
#: data is a defect in one of them, not a modelling difference.
#:
#: Deliberately narrow. Measured across all fourteen regimes, these groups
#: agree to a median of 0.0013 (regression) and 0.0083 (classification), while
#: a group spanning different model classes (linear, naive Bayes, neighbours,
#: Gaussian process) has a median spread of 0.0808 and reaches 0.6571. Those
#: wide gaps are real differences in what the models can represent, so a check
#: scoped that widely would be noise. This one is only meaningful because the
#: members are the same estimator with a different penalty term.
AGREEING_GROUPS = {
    "linear_regression": ("regression", ["MissLinear", "MissRidgeRegressor",
                                         "MissLASSORegressor"]),
    "linear_classification": ("classification", ["MissLogistic",
                                                 "MissRidgeClassifier",
                                                 "MissLASSOClassifier"]),
}

#: How far apart two of them may score on the same data. Taken from the
#: measurement rather than chosen: every legitimate gap across the whole
#: matrix is at or below 0.0305, and the one defect sits at 0.1128, so this
#: has about a factor of two either side.
SIBLING_GAP = 0.06

#: Gaps that are real, with the reason. Work items like the tables above.
KNOWN_SIBLING_GAPS = {
    ("linear_regression", "wide_p_gt_n"):
        "0.7417 between MissRidgeRegressor at 0.7304 and MissLASSORegressor at "
        "-0.0113, with MissLinear declining the regime outright. At p=40 and "
        "n=25 an L1 penalty of 1.0 shrinks essentially every coefficient to "
        "zero while an L2 penalty retains the signal, which is the textbook "
        "difference between the two penalties rather than a defect in either. "
        "This is the one regime where the group's members are not "
        "interchangeable.",
}


def _sibling_scores(group, regime):
    """Score each member of a group on one regime, skipping declared refusals."""
    task, names = AGREEING_GROUPS[group]
    scores = {}
    for name in names:
        if (name, regime) in KNOWN_FAILURES:
            continue
        try:
            est, X, y, g = _fitted(name, task, regime)
            pred = np.asarray(predict_of(est, name, X, g), dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(pred)):
            continue
        value, _ = _trivial_baseline_score(y, pred, task)
        if value is not None:
            scores[name] = (value, y)
    return scores


def _granularity(y, task):
    """The smallest difference the metric can express.

    Accuracy on n observed rows moves in steps of 1/n, so on the twelve-row
    regime two members differing by a single sample are 0.083 apart and that
    is not evidence of anything. r-squared is continuous and has no such
    floor.
    """
    if task == "regression":
        return 0.0
    observed = int(np.sum(~np.isnan(np.asarray(y, dtype=float))))
    return (2.0 / observed) if observed else 0.0


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("group", sorted(AGREEING_GROUPS))
def test_penalised_siblings_agree(group, regime):
    """The gap the accuracy floor is too generous to see.

    ``test_fit_is_not_worse_than_ignoring_the_features`` asks whether a fit
    beat the mean, which is the weakest question worth asking and is exactly
    what a diverged model fails. It cannot see an estimator that is merely
    much worse than the sibling beside it: MissLinear scored 0.7334 under
    extreme_scale while MissRidgeRegressor and MissLASSORegressor reached
    0.846 on the same data, and every test in this suite passed, because
    0.7334 is a perfectly respectable number in isolation.

    That gap was found by reading a table by hand. This is the check that
    finds the next one.
    """
    task, _ = AGREEING_GROUPS[group]
    scores = _sibling_scores(group, regime)
    if len(scores) < 2:
        pytest.skip("fewer than two members fitted this regime")

    values = {n: v for n, (v, _) in scores.items()}
    best_name = max(values, key=values.get)
    worst_name = min(values, key=values.get)
    gap = values[best_name] - values[worst_name]
    tolerance = max(SIBLING_GAP,
                    _granularity(scores[worst_name][1], task))

    declared = KNOWN_SIBLING_GAPS.get((group, regime))
    if declared:
        if gap <= tolerance:
            pytest.fail("declared in KNOWN_SIBLING_GAPS but the group now "
                        "agrees to %.4f; delete the entry" % gap)
        return

    assert gap <= tolerance, (
        "%s scored %.4f where %s reached %.4f on the same data, a gap of "
        "%.4f against a tolerance of %.4f. These differ only in their "
        "penalty, so one of them is doing something the other is not; check "
        "what each excludes or rescales before widening the tolerance."
        % (worst_name, values[worst_name], best_name, values[best_name],
           gap, tolerance))


# ---------------------------------------------------------------------------
# 7f. Numeric parameters that cannot describe anything
# ---------------------------------------------------------------------------

#: parameter -> (values that are not a description of anything, why)
#:
#: Fifty-four of these were accepted across eleven estimators. Most were
#: harmless in the sense that the fit came back with the same r-squared as at
#: the default, which is how they came to be recorded as benign and left. That
#: was not a safe reading: a negative tolerance took MissLASSORegressor from
#: 0.9869 to -0.0007, a fit explaining nothing, with converged unset and no
#: warning. The other ten estimators hid it by being unaffected.
INVALID_NUMERIC = {
    'max_iter':     ((0, -1), 'a budget of no iterations is not a budget'),
    'tol':          ((0.0, -1e-3), 'a tolerance is a positive distance'),
    'n_quadrature': ((0, -1), 'quadrature needs at least one node'),
    'n_restarts':   ((-1,), 'a negative count of restarts'),
    'fe_ridge':     ((-1.0,), 'a negative ridge rewards large coefficients'),
    'n_neighbors':  ((0, -1), 'a neighbourhood of nobody'),
    'var_smoothing': ((-1.0,), 'a negative variance floor'),
    'alpha':        ((-1.0,), 'a negative penalty'),
    'l2_reg':       ((-1.0,), 'a negative penalty'),
    'degree':       ((0, -1), 'a polynomial of negative degree'),
    'C':            ((0.0, -1.0), 'a non-positive inverse penalty'),
    'epsilon':      ((-1.0,), 'a negative width for the insensitive tube'),
    'noise_var_init': ((0.0, -1.0), 'a variance that is not positive'),
    'max_iter_newton': ((0, -1), 'a Newton budget of no steps'),
}

#: Numeric parameters deliberately not swept, with the reason.
UNSWEPT_NUMERIC = {
    'random_state': 'any integer is a valid seed, and None means unseeded',
    'n_jobs': 'scikit-learn defines -1 as all cores, so negatives are legal',
    'max_samples': 'a fraction or a count, checked by the ensemble itself',
    'n_estimators': 'MissEnsemble is not in this registry; it takes an '
                    'estimator argument and is exercised by the unit suite',
    'm': 'MissImputer is not in this registry, for the same reason',
    'n_components': 'not a parameter of any estimator in this registry',
    'coef0': 'the constant term of a polynomial or sigmoid kernel, where '
             'every real value including a negative one is meaningful',
}


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_invalid_numeric_parameters_are_refused(name, task):
    """A value that cannot describe anything must be refused, not absorbed.

    scikit-learn raises for every one of these, so accepting them is a
    contract difference before it is a correctness problem. The correctness
    problem is real too, and it was invisible: ten of the eleven estimators
    that take ``tol`` returned an unchanged r-squared at ``tol=-1e-3``, and
    the eleventh returned -0.0007.
    """
    X, y, g = option_fixture(task)
    params = inspect.signature(getattr(ML, name).__init__).parameters
    for pname, (values, why) in sorted(INVALID_NUMERIC.items()):
        if pname not in params:
            continue
        for value in values:
            kw = dict(REGRESSORS.get(name, CLASSIFIERS.get(name, {})))
            kw[pname] = value
            with pytest.raises(ValueError, match=pname):
                fit_of(getattr(ML, name)(**kw), name, X, y, g)


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_every_numeric_parameter_is_swept_or_exempt(name, task):
    """A new numeric parameter must be swept here or exempted with a reason.

    The counterpart to test_every_string_option_is_in_the_matrix, and it
    exists for the same reason: these fifty-four values were accepted for
    months, and no test could notice because no test knew the parameters
    were there.
    """
    known = set(INVALID_NUMERIC) | set(UNSWEPT_NUMERIC) | set(UNCHECKED_OPTIONS)
    params = inspect.signature(getattr(ML, name).__init__).parameters
    unswept = sorted(
        p for p, v in params.items()
        if p not in ('self', 'kwargs')
        and isinstance(v.default, (int, float))
        and not isinstance(v.default, bool)
        and p not in known)
    assert not unswept, (
        "%s takes %s, which no test sweeps for values that cannot describe "
        "anything. Add it to INVALID_NUMERIC with those values, or to "
        "UNSWEPT_NUMERIC with the reason it has none." % (name, unswept))


@pytest.mark.parametrize("name,task", ALL_ESTIMATORS,
                         ids=[n for n, _ in ALL_ESTIMATORS])
def test_the_defaults_are_themselves_valid(name, task):
    """Whatever the sweep refuses, the shipped defaults must not trip.

    Cheap, and it is the failure mode a tightened guard actually has: the
    check that rejects nonsense also rejecting the value every user gets for
    saying nothing.
    """
    X, y, g = option_fixture(task)
    est = fit_of(make(name), name, X, y, g)
    assert np.all(np.isfinite(np.asarray(predict_of(est, name, X, g), float)))

if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v", "--tb=short", "-ra"]))
