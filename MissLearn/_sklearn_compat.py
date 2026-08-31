# -*- coding: utf-8 -*-
"""Declared exceptions to the scikit-learn estimator contract.

``check_estimator`` encodes assumptions that are correct for almost every
estimator. Two of them are incompatible with what this library is for, one is
a numerical tolerance that cannot be met without making the output worse, and
one is an upstream limitation in libsvm that affects two classes only.
Declaring them is honest; quietly skipping the checks, or contorting the
library to satisfy them, would not be.

scikit-learn 1.6 accepts an ``expected_failed_checks`` mapping so that a
project can state this in a form its own tooling understands:

    from sklearn.utils.estimator_checks import check_estimator
    from MissLearn._sklearn_compat import expected_failed_checks

    check_estimator(est, expected_failed_checks=expected_failed_checks(est))

Every entry needs a reason a reviewer can evaluate. "Known issue" is not one.
An entry that can be removed by fixing the library should be fixed instead.
"""
__all__ = ["EXPECTED_FAILED_CHECKS", "PER_ESTIMATOR_FAILED_CHECKS",
           "expected_failed_checks"]


#: Checks that cannot pass, with the reason each is out of reach.
EXPECTED_FAILED_CHECKS = {
    "check_supervised_y_no_nan": (
        "Requires the estimator to raise on NaN in y. MissLearn accepts it "
        "deliberately: a row with an unobserved response still informs the "
        "feature distribution and is used by the likelihood, which is a "
        "central capability rather than an oversight. Infinity in y IS "
        "rejected, since no conditional distribution makes an infinite "
        "observation meaningful. Passing this check would mean deleting the "
        "feature."
    ),
    "check_methods_subset_invariance": (
        "decision_function is log(p / (1 - p)). Where a model is certain, p "
        "is within 1e-8 of 0 or 1, and the logarithm turns a difference far "
        "below any meaningful tolerance into a large difference in log-odds. "
        "predict_proba itself is subset-invariant to 6.7e-15 in the worst "
        "case across the classifier family and exactly zero for three of "
        "them, measured over several subsets of a full matrix; only the "
        "logged form moves, and 6.7e-15 is seven orders below the tolerance "
        "the check applies. Meeting the check would require clipping "
        "probabilities so "
        "coarsely that genuine confidence became unrepresentable, which is a "
        "worse output for a real user."
    ),
}


#: Exceptions that belong to one estimator rather than to the library.
_LIBSVM_READONLY_PICKLE = (
    "Inherited from scikit-learn, not ours. The check memmaps the "
    "fitted estimator so that every array attribute becomes read-only, "
    "then pickles it and predicts; libsvm's Cython probability path "
    "needs writable buffers and raises 'buffer source array is "
    "read-only'. Measured on scikit-learn 1.6.1: a plain "
    "SVC(probability=True) fails this check with the identical error, "
    "while SVC(probability=False) and SVR both pass, and so does "
    "MissSupportRegressor. A classifier has to offer predict_proba, "
    "which needs probability=True, so the only ways out are to drop "
    "that method or to copy libsvm's fitted arrays on every call, which "
    "would be a permanent cost to work around an upstream limitation."
)

PER_ESTIMATOR_FAILED_CHECKS = {
    # The dispatcher resolves to MissSupportClassifier on a classification
    # target and inherits the same libsvm limitation through it.
    "MissSupport": {
        "check_estimators_pickle": _LIBSVM_READONLY_PICKLE,
    },
    "MissSupportClassifier": {
        "check_estimators_pickle": _LIBSVM_READONLY_PICKLE,
    },
}


def expected_failed_checks(estimator=None):
    """Return the declared exceptions, for ``check_estimator``.

    Takes an estimator so that the signature matches what scikit-learn's
    ``parametrize_with_checks`` expects as a callback, and so a per-estimator
    exception has somewhere to go. Entries in EXPECTED_FAILED_CHECKS apply to
    the whole library; entries in PER_ESTIMATOR_FAILED_CHECKS apply only to
    the named class, so an exception granted to one estimator is not silently
    extended to its siblings.
    """
    declared = dict(EXPECTED_FAILED_CHECKS)
    if estimator is not None:
        name = getattr(type(estimator), "__name__", None)
        declared.update(PER_ESTIMATOR_FAILED_CHECKS.get(name, {}))
    return declared
