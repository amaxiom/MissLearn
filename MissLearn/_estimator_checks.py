# -*- coding: utf-8 -*-
"""Conformance checks for estimators that accept missing data.

``sklearn.utils.estimator_checks.check_estimator`` verifies the scikit-learn
contract. It says nothing about how an estimator behaves when the data is
incomplete, because scikit-learn estimators mostly refuse incomplete data.

That leaves a gap. An estimator can satisfy every scikit-learn check and still
fit happily on a fully missing column and return ``NaN`` predictions, or raise
at 90% missingness where an equivalent estimator degrades gracefully. Those are
not hypotheticals: both occurred in MissLearn, and neither was detectable by
any check that existed.

This module is the missing counterpart. It drives an estimator through the
degenerate regimes that incomplete data actually produces and reports what it
does in each. It is deliberately generic: nothing here depends on MissLearn,
so it is usable against any NaN-tolerant estimator following the scikit-learn
API.

Usage
-----
    from MissLearn import check_missing_data_estimator

    report = check_missing_data_estimator(MyEstimator())
    print(report)                       # human-readable summary
    assert report.ok                    # or fail a test

    # In a parametrised test suite:
    for regime, outcome in report.items():
        assert outcome.acceptable, outcome.explain()

What "acceptable" means
-----------------------
Not "did not raise". An estimator has three defensible responses to a
degenerate regime, and one indefensible one:

  fit and predict usefully           acceptable
  refuse, clearly, naming the cause  acceptable
  refuse with an opaque error        marginal, reported
  fit and return NaN or nonsense     never acceptable

Answering differently to the same question also counts as nonsense. Every
check above looks at one fit, and a large finite number passes them however
arbitrary it is, so an estimator whose answer depended on the order the rows
arrived in went unnoticed: a whole family of degenerate-column defects
survived 398 passing tests here and was eventually found by a property test.
The determinism check refits the same data, and refits it with the rows
permuted, and requires the same answer both times. It caught
MissMixedRegressor moving by 6.3e-02 the first time it ran.

The last is singled out because it is the most dangerous and the easiest to
ship. An exception stops a pipeline; a silent ``NaN`` propagates into whatever
consumes the prediction. Four MissLearn regressors did exactly this on a fully
missing column, and four classifiers returned plausible class labels whose
``predict_proba`` was entirely ``NaN``, so the label path concealed it
completely.
"""
from collections import OrderedDict

import numpy as np

__all__ = ["check_missing_data_estimator", "MissingDataReport",
           "RegimeOutcome", "REGIMES", "DETERMINISM_REGIMES"]


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

def _base(n, p, task, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(scale=0.4, size=n)
    if task == "classification":
        y = (y > np.median(y)).astype(float)
    return X, y


def _holes(X, rate, seed):
    rng = np.random.default_rng(seed)
    X = X.copy()
    X[rng.random(X.shape) < rate] = np.nan
    return X


def _r_rate(rate):
    def make(task, seed):
        X, y = _base(120, 5, task, seed)
        return _holes(X, rate, seed), y
    return make


def _r_no_complete_cases(task, seed):
    X, y = _base(120, 5, task, seed)
    rng = np.random.default_rng(seed)
    X = X.copy()
    for i in range(X.shape[0]):
        X[i, rng.integers(0, X.shape[1])] = np.nan
    return X, y


def _r_all_nan_column(task, seed):
    X, y = _base(120, 5, task, seed)
    X = _holes(X, 0.15, seed)
    X[:, 2] = np.nan
    return X, y


def _r_one_observed_cell(task, seed):
    """A column with exactly one observed value: estimable in principle,
    barely."""
    X, y = _base(120, 5, task, seed)
    X = _holes(X, 0.15, seed)
    X[1:, 3] = np.nan
    return X, y


def _r_missing_y(task, seed):
    X, y = _base(120, 5, task, seed)
    X = _holes(X, 0.15, seed)
    rng = np.random.default_rng(seed + 1)
    y = y.copy()
    y[rng.random(len(y)) < 0.20] = np.nan
    return X, y


def _r_blockwise(task, seed):
    """Whole groups of columns missing together, as when an instrument fails.

    Distinguished from scattered missingness because the cost of pattern-based
    methods is driven by the number of distinct patterns, not the rate.
    """
    X, y = _base(120, 6, task, seed)
    X = X.copy()
    X[:40, 0:3] = np.nan
    X[40:80, 3:6] = np.nan
    return X, y


def _r_wide(task, seed):
    X, y = _base(25, 40, task, seed)
    return _holes(X, 0.15, seed), y


def _r_single_feature(task, seed):
    X, y = _base(120, 1, task, seed)
    return _holes(X, 0.20, seed), y


def _r_extreme_scale(task, seed):
    X, y = _base(120, 5, task, seed)
    X = X.copy()
    X[:, 0] *= 1e6
    X[:, 1] *= 1e-6
    return _holes(X, 0.15, seed), y


REGIMES = OrderedDict([
    ("clean_15pct",       (_r_rate(0.15),  "a routine rate, as a control")),
    ("missing_50pct",     (_r_rate(0.50),  "half the cells absent")),
    ("missing_90pct",     (_r_rate(0.90),  "extreme; the covariance is barely estimable")),
    ("no_complete_cases", (_r_no_complete_cases, "every row has a hole, so complete-case seeding has nothing to use")),
    ("all_nan_column",    (_r_all_nan_column, "a column with no observed values anywhere")),
    ("one_observed_cell", (_r_one_observed_cell, "a column observed exactly once")),
    ("missing_y",         (_r_missing_y,   "an incomplete target")),
    ("blockwise",         (_r_blockwise,   "columns missing in blocks, as when an instrument fails")),
    ("wide_p_gt_n",       (_r_wide,        "more features than samples")),
    ("single_feature",    (_r_single_feature, "p = 1")),
    ("extreme_scale",     (_r_extreme_scale, "features differing by twelve orders of magnitude")),
])


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

class RegimeOutcome(object):
    """What an estimator did in one regime."""

    FITTED = "fitted"
    REFUSED_CLEARLY = "refused clearly"
    REFUSED_OPAQUELY = "refused opaquely"
    SILENT_NONSENSE = "fitted, output unusable"

    def __init__(self, regime, kind, detail=""):
        self.regime = regime
        self.kind = kind
        self.detail = detail

    @property
    def acceptable(self):
        return self.kind in (self.FITTED, self.REFUSED_CLEARLY)

    def explain(self):
        if self.kind == self.SILENT_NONSENSE:
            return ("%s: %s. This is the failure mode worth removing first. "
                    "An exception stops a pipeline; a silent NaN propagates "
                    "into whatever consumes the prediction."
                    % (self.regime, self.detail))
        if self.kind == self.REFUSED_OPAQUELY:
            return ("%s: refused, but the error does not say why (%s). "
                    "Refusing is defensible; refusing without naming the cause "
                    "leaves the user unable to act."
                    % (self.regime, self.detail))
        return "%s: %s" % (self.regime, self.kind)

    def __repr__(self):
        return "<%s %s>" % (self.regime, self.kind)


#: How far two fits of the same data may differ before it counts as the
#: estimator having answered differently. Generous on purpose: an iterative
#: optimiser is entitled to floating-point noise, and the failures this is
#: built to catch are nothing like noise. The degenerate-column family
#: produced coefficients of -324 against -1.68e6, and predictions differing
#: by 8.3% relative; the estimators that pass do so at exactly 0.0.
DETERMINISM_RTOL = 1e-6
DETERMINISM_ATOL = 1e-9

#: Regimes the determinism check runs on by default.
#:
#: It costs two extra fits wherever it runs, so applying it to all eleven
#: tripled the time of a suite that already took several minutes, which is
#: not a price worth paying on every commit. These four keep almost all of
#: the detection: the control, because an estimator that is not reproducible
#: on ordinary data is not reproducible anywhere; the two structurally
#: degenerate regimes, which is where the defects this was built for live;
#: and the conditioning one. When MissMixedRegressor was caught it failed on
#: the control by 1.9e-04 and on no_complete_cases by 6.3e-02, so the subset
#: would have found it twice over. Pass determinism='all' for the full set.
DETERMINISM_REGIMES = frozenset({
    "clean_15pct", "no_complete_cases", "all_nan_column", "extreme_scale",
})


def _comparison_surface(est, Xa):
    """The continuous output where the estimator has one, else the prediction.

    Determinism has to be compared on this rather than on ``predict``. A
    classifier's prediction is a label, and two fits that genuinely differ
    agree on almost every label: MissLASSOClassifier drifted by 1.6e-04 in its
    coefficients between two fits of the same matrix, seeded from liblinear
    without a random_state, and this check reported it deterministic because
    the labels matched. sklearn's own check_fit_idempotent caught it on
    decision_function. Regressors are unaffected, since for them predict is
    already the continuous output.

    Falls back rather than raising, because a third-party estimator may expose
    the method and refuse the call, for instance when prediction needs a
    keyword this helper does not have.
    """
    for method in ("decision_function", "predict_proba"):
        if hasattr(est, method):
            try:
                return np.asarray(getattr(est, method)(Xa), dtype=float)
            except Exception:
                pass
    return np.asarray(est.predict(Xa), dtype=float)


def _determinism_failures(estimator, X, y, fitted, fit_kwargs, clone):
    """Refit the same data and check the answer is the same.

    This is the axis the suite was missing. Every other check asks whether
    one fit produced something defensible, and a large finite number passes
    that however arbitrary it is. Nothing fitted the same data twice, so an
    estimator whose answer depended on the order the rows arrived in looked
    perfectly healthy: the degenerate-column family survived 398 passing
    tests here and was found by a property test instead.

    Two things are checked, both of which must hold for any estimator worth
    the name:

      refitting identical data reproduces the answer
      permuting the rows permutes the predictions and changes nothing else

    A failure is graded as silent nonsense rather than a refusal, because
    that is what it is: the estimator did not decline, it returned two
    different answers to the same question and reported success both times.
    """
    failures = []

    # Resolve callable fit_kwargs once, against the original data. Resolving
    # them again per fit looks equivalent and is not: a callable such as
    # ``lambda X, y: np.arange(len(X)) // 5`` assigns by position, so
    # regenerating it from permuted rows gives each row a different group and
    # the comparison is between two different problems. That mistake made
    # MissMixedRegressor look non-deterministic when it was exact.
    base_kw = {k: (v(X, y) if callable(v) else v)
               for k, v in fit_kwargs.items()}

    def _permuted_kw(perm):
        """Per-row arguments follow their rows; anything else is unchanged."""
        out = {}
        for k, val in base_kw.items():
            arr = np.asarray(val)
            out[k] = (arr[perm] if arr.ndim >= 1 and arr.shape[0] == X.shape[0]
                      else val)
        return out

    def _fit(Xa, ya, kw):
        est = clone(estimator)
        est.fit(Xa, ya, **kw)
        return _comparison_surface(est, Xa)

    pred = _comparison_surface(fitted, X)
    what = ("decision_function" if hasattr(fitted, "decision_function")
            else "predict_proba" if hasattr(fitted, "predict_proba")
            else "predictions")

    # Same data twice.
    try:
        again = _fit(X, y, base_kw)
    except Exception as exc:                             # pragma: no cover
        return ["refitting identical data raised %s" % type(exc).__name__]
    if again.shape == pred.shape and not np.allclose(
            again, pred, rtol=DETERMINISM_RTOL, atol=DETERMINISM_ATOL,
            equal_nan=True):
        failures.append(
            "refitting identical data changed %s (max difference %.3e)"
            % (what, float(np.nanmax(np.abs(again - pred)))))

    # Same data, rows reordered. Seeded from the data so the permutation is
    # reproducible for a given regime rather than varying between runs.
    rng = np.random.default_rng(abs(hash(X.shape)) % (2 ** 31))
    perm = rng.permutation(X.shape[0])
    try:
        permuted = _fit(X[perm], np.asarray(y)[perm], _permuted_kw(perm))
    except Exception as exc:
        return failures + [
            "fitted the data but refused the same rows in a different order "
            "(%s)" % type(exc).__name__]
    if permuted.shape == pred.shape and not np.allclose(
            pred[perm], permuted, rtol=DETERMINISM_RTOL,
            atol=DETERMINISM_ATOL, equal_nan=True):
        failures.append(
            "row order changed %s (max difference %.3e); the same data in a "
            "different order is the same data"
            % (what, float(np.nanmax(np.abs(pred[perm] - permuted)))))

    return failures


class MissingDataReport(object):
    """The outcome of every regime, with a readable summary."""

    def __init__(self, estimator_name, outcomes):
        self.estimator_name = estimator_name
        self.outcomes = outcomes

    #: The regime an estimator has to actually fit to be participating at
    #: all. It is described in REGIMES as "a routine rate, as a control".
    CONTROL_REGIME = "clean_15pct"

    def items(self):
        return self.outcomes.items()

    @property
    def participates(self):
        """Did the estimator fit the control regime?

        Refusing every regime used to score as passing, because a clean
        refusal is graded acceptable and a plain sklearn Ridge refuses all
        eleven with a tidy message about NaN. That produced the inversion
        this property exists to stop: Ridge, which cannot take a missing
        value at all, reported ok; HistGradientBoosting, which fits ten
        regimes out of eleven, reported not ok. Handling nothing is not
        conformance, however clearly it is announced.
        """
        control = self.outcomes.get(self.CONTROL_REGIME)
        if control is not None:
            return control.kind == RegimeOutcome.FITTED
        # No control regime in this report; fall back to "fitted something".
        return any(o.kind == RegimeOutcome.FITTED
                   for o in self.outcomes.values())

    @property
    def ok(self):
        return (self.participates
                and all(o.acceptable for o in self.outcomes.values()))

    def __bool__(self):
        """``if report:`` should mean what ``report.ok`` means.

        Without this, truth-testing the report returned True for every
        object, including one full of failures.
        """
        return self.ok

    __nonzero__ = __bool__          # Python 2 spelling, harmless to keep

    @property
    def problems(self):
        return [o for o in self.outcomes.values() if not o.acceptable]

    def __str__(self):
        W = 22
        lines = ["", "Missing-data conformance: %s" % self.estimator_name,
                 "=" * 64]
        for name, o in self.outcomes.items():
            mark = "ok  " if o.acceptable else "FAIL"
            lines.append("  %s %-*s %s" % (mark, W, name, o.kind))
            if not o.acceptable and o.detail:
                lines.append("       %s" % o.detail[:70])
        lines.append("-" * 64)
        if not self.participates:
            lines.append(
                "  NOT A MISSING-DATA ESTIMATOR: it refused the control "
                "regime")
            lines.append(
                "  (%s, a routine 15%% rate). Refusing every regime is not"
                % self.CONTROL_REGIME)
            lines.append(
                "  conformance, however clearly the refusal is worded.")
            lines.append(
                "  If fit needs an extra argument, pass it through "
                "fit_kwargs.")
        elif self.ok:
            lines.append("  every regime handled acceptably")
        else:
            lines.append("  %d regime(s) need attention:" % len(self.problems))
            for o in self.problems:
                lines.append("    - " + o.explain())
        return "\n".join(lines)

    __repr__ = __str__


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

def _is_classifier(est):
    return getattr(est, "_estimator_type", None) == "classifier"


def _clear_error(exc):
    """Does the message let the user act?

    A message naming the offending column, the shape, or the condition is
    actionable. A bare LinAlgError from inside a factorisation is not, even
    though declining was the right decision.
    """
    text = str(exc).lower()
    if not text.strip():
        return False
    # Two ways a message can be actionable: it names the thing that is wrong,
    # or it names what to do about it. The list was originally all nouns, so
    # a message like "Provide 'estimator' or 'estimators'" scored as opaque
    # despite telling the user exactly what to do next.
    informative = ("column", "feature", "sample", "class", "identified",
                   "missing", "observed", "sparse", "shape", "expect",
                   "requires", "not supported", "n_samples", "parameters",
                   "provide", "specify", "must be", "use ", "pass ",
                   "instead", "got ", "cannot fit")
    opaque = ("not positive definite", "singular matrix", "0-dimensional",
              "index out of", "must be at least two-dimensional")
    if any(o in text for o in opaque):
        return False
    return any(k in text for k in informative)


def check_missing_data_estimator(estimator, task=None, seed=0,
                                 fit_kwargs=None, determinism='default'):
    """Drive ``estimator`` through the degenerate missing-data regimes.

    Parameters
    ----------
    estimator : estimator instance
        Must follow the scikit-learn API and accept ``NaN`` in ``X``. A fresh
        clone is fitted for each regime, so the instance passed is untouched.
    task : {'regression', 'classification'}, optional
        Inferred from ``_estimator_type`` when not given.
    seed : int
        Seeds the generated data, so a report is reproducible.
    fit_kwargs : dict, optional
        Extra arguments for ``fit``. A value may be a callable taking
        ``(X, y)`` and returning the argument, which is how a per-row
        argument has to be supplied: the regimes use different row counts,
        so no fixed array fits them all::

            check_missing_data_estimator(
                MissMixedRegressor(),
                fit_kwargs={'groups': lambda X, y: np.arange(len(X)) // 5})

    determinism : {'default', 'all', 'off'}
        Where to check that refitting the same data, and refitting it
        with the rows reordered, give the same answer. 'default' runs
        it on DETERMINISM_REGIMES, which is four of the eleven and
        keeps the cost near 1.5x rather than 3x. 'all' runs it
        everywhere; 'off' skips it.

    Returns
    -------
    MissingDataReport

    Notes
    -----
    The report is advisory, not a pass/fail gate. An estimator may legitimately
    decline a regime: a model that estimates a full joint covariance is not
    identified when p exceeds n, and saying so plainly is the correct
    behaviour. What the report insists on is that the decision be visible.
    """
    from sklearn.base import clone

    task = task or ("classification" if _is_classifier(estimator)
                    else "regression")
    fit_kwargs = fit_kwargs or {}
    name = type(estimator).__name__
    outcomes = OrderedDict()

    for regime, (make, _why) in REGIMES.items():
        X, y = make(task, seed)
        try:
            est = clone(estimator)
        except Exception:                                # pragma: no cover
            est = estimator
        # A per-row argument cannot be a fixed array here: the regimes do not
        # all use the same number of rows (most are n=120, wide_p_gt_n is
        # n=25), so one groups vector is right for ten of them and a length
        # mismatch on the eleventh. That mismatch surfaced as "refused
        # opaquely", which reads as a fault in the estimator rather than in
        # how it was called. Any callable value is resolved against the
        # regime's own data instead.
        resolved = {k: (v(X, y) if callable(v) else v)
                    for k, v in fit_kwargs.items()}
        try:
            est.fit(X, y, **resolved)
            pred = np.asarray(est.predict(X), dtype=float)

            bad = []
            if pred.shape[0] != X.shape[0]:
                bad.append("predict returned %d rows for %d inputs"
                           % (pred.shape[0], X.shape[0]))
            if not np.all(np.isfinite(pred)):
                bad.append("predictions contain NaN or inf")

            if _is_classifier(est) and hasattr(est, "predict_proba"):
                # Checked explicitly. Four classifiers once returned finite
                # labels whose probabilities were entirely NaN, and a
                # predict-only sweep called them healthy.
                P = np.asarray(est.predict_proba(X), dtype=float)
                if not np.all(np.isfinite(P)):
                    bad.append("predict_proba contains NaN, while predict "
                               "returned finite labels")
                elif not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
                    bad.append("predict_proba rows do not sum to 1")

            if determinism == 'all' or (determinism == 'default'
                                        and regime in DETERMINISM_REGIMES):
                bad.extend(_determinism_failures(estimator, X, y, est,
                                                 fit_kwargs, clone))

            if bad:
                outcomes[regime] = RegimeOutcome(
                    regime, RegimeOutcome.SILENT_NONSENSE, "; ".join(bad))
            else:
                outcomes[regime] = RegimeOutcome(regime, RegimeOutcome.FITTED)

        except Exception as exc:
            kind = (RegimeOutcome.REFUSED_CLEARLY if _clear_error(exc)
                    else RegimeOutcome.REFUSED_OPAQUELY)
            outcomes[regime] = RegimeOutcome(
                regime, kind, "%s: %s" % (type(exc).__name__,
                                          str(exc).split("\n")[0][:90]))

    return MissingDataReport(name, outcomes)
