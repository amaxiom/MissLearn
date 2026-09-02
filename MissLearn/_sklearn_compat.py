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
from sklearn.base import is_classifier as _sk_is_classifier
from sklearn.base import is_regressor as _sk_is_regressor

__all__ = ["EXPECTED_FAILED_CHECKS", "PER_ESTIMATOR_FAILED_CHECKS",
           "expected_failed_checks", "CONTRACT_EXEMPT",
           "is_classifier_safe", "is_regressor_safe"]



#: Classes that ``check_estimator`` should not be run against at all, as
#: opposed to individual checks they are expected to fail. The distinction
#: matters: an entry in EXPECTED_FAILED_CHECKS says "this estimator meets the
#: contract except here"; an entry below says "this class is not the kind of
#: thing the contract describes". Both need a reason a reviewer can weigh.
#:
#: Keep this list as short as the truth allows. A class is easier to exempt
#: than to fix, and the fourteen estimators that were once left out of the
#: continuous-integration list were not passing quietly: they were not being
#: asked.
CONTRACT_EXEMPT = {
    "MissImputer": (
        "scikit-learn sees a transform method and requires the transformer "
        "contract, under which transform returns one array of shape "
        "(n_samples, n_features_out). MissImputer.transform returns a list of "
        "m completed datasets, which is what multiple imputation is: the m "
        "draws are combined afterwards by Rubin's rules, and returning a "
        "single array would discard the between-imputation variance that "
        "makes the estimate honest. Declaring transformer_tags to satisfy the "
        "discovery would state a contract the class does not meet, and the "
        "transformer checks would then fail on the return value instead. "
        "\n\nIt is still a BaseEstimator, which it needs to be: scikit-learn's "
        "check_is_fitted calls get_tags from 1.7, so a class without "
        "__sklearn_tags__ raises AttributeError from inside its own transform. "
        "get_params, set_params and clone work, fit sets n_features_in_ and "
        "feature_names_in_, and the estimators themselves need no imputation "
        "step because they marginalise instead."
    ),
}

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


# ---------------------------------------------------------------------------
# Total forms of the two estimator-type questions
# ---------------------------------------------------------------------------
# Through scikit-learn 1.6, is_classifier and is_regressor read the
# _estimator_type attribute and returned False for any object without it.
# From 1.7 they go through get_tags, which raises AttributeError when nothing
# in the object's MRO defines __sklearn_tags__. Eleven public classes in this
# package are not BaseEstimator subclasses, so on 1.7 and later the plain
# functions raise rather than answer for MissImputer, MissEnsemble,
# MissExplainer, MissPreprocessor and the rest.
#
# Asking whether an arbitrary object is a classifier is a total question, and
# an object that declares no estimator type is not one. These return that
# answer instead of raising. Only AttributeError is caught, so a real failure
# inside a genuine __sklearn_tags__ implementation still propagates.
#
# The deeper repair is to give the estimator-like classes among those eleven a
# proper scikit-learn identity, which would also bring them within reach of
# check_estimator. That is tracked separately; it is a wider change than a
# compatibility shim.

def is_classifier_safe(estimator) -> bool:
    """``sklearn.base.is_classifier`` that answers False instead of raising.

    Returns False for an object with no ``__sklearn_tags__`` and no
    ``_estimator_type``, which is what scikit-learn itself did through 1.6.
    """
    try:
        return bool(_sk_is_classifier(estimator))
    except AttributeError:
        return False


def is_regressor_safe(estimator) -> bool:
    """``sklearn.base.is_regressor`` that answers False instead of raising.

    The counterpart of :func:`is_classifier_safe`; see its note.
    """
    try:
        return bool(_sk_is_regressor(estimator))
    except AttributeError:
        return False
