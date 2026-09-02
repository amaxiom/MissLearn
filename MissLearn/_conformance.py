# -*- coding: utf-8 -*-
"""Behaviour shared by every estimator, applied in one place.

Why this module exists
----------------------
MissLearn had sixteen estimators that were meant to be interchangeable, and
their behaviour in degenerate regimes was whatever each class happened to
implement. A user found the consequence within twenty minutes:
``MissLASSOClassifier`` raised on high missingness while ``MissLogistic`` and
``MissLASSORegressor`` degraded gracefully, because a fallback present in both
siblings had never been applied to that one class.

Per-class discipline cannot prevent that, because nothing compares the classes
with each other. The fix is structural: behaviour that should be identical
across estimators is defined once, here, and applied to every class through
the same wrapping point. A cross-estimator conformance suite
(``tests/conformance_test_suite.py``) then holds the guarantee.

What lives here
---------------
Semantics that belong to every estimator, not to any one family:

``check_no_empty_features``
    Refuse a feature with no observed values, uniformly and with a message
    naming the columns.

``route_multiclass``
    Let the binary classifiers accept more than two classes by delegating to
    one-vs-rest internally, so users are not required to know which wrapper to
    reach for.

This module deliberately holds no family-specific logic. Anything that is true
of one estimator and not its siblings belongs in that estimator.
"""
from typing import Optional

import numpy as np
from ._sklearn_compat import is_classifier_safe

__all__ = ["validate_input", "check_not_sparse", "encode_labels", "check_no_complex_data", "check_no_empty_features", "check_penalty", "check_positive_int", "check_choice", "check_copula",
           "check_tolerance", "check_positive_float", "public_parameter_name",
           "check_common_parameters", "check_feature_names",
           "check_n_features", "route_multiclass", "EmptyFeatureError"]


def _allow_nan_kwarg():
    """The keyword that permits NaN, whichever scikit-learn is installed.

    Renamed from force_all_finite to ensure_all_finite in 1.6. The package
    supports 1.1 upwards, so the name is resolved rather than assumed.
    """
    import inspect as _inspect
    from sklearn.utils.validation import check_array as _ca
    params = _inspect.signature(_ca).parameters
    return ("ensure_all_finite" if "ensure_all_finite" in params
            else "force_all_finite")


def validate_input(estimator, X, y=None, check_y=True, min_samples=2):
    """Validate X and y the way scikit-learn expects, while permitting NaN.

    MissLearn wrote its own input handling, which accepted a great deal that
    scikit-learn requires an estimator to reject: one-dimensional X, empty
    arrays, single samples, sparse matrices, infinities, a continuous target
    handed to a classifier, and ``y=None``. That accounted for most of the
    hundred individual ``check_estimator`` failures, and none of it was
    family-specific, so it is one validator rather than sixteen.

    scikit-learn's own ``check_array`` is used deliberately instead of
    hand-rolled equivalents. Several of the checks assert on the *text* of the
    error, so reimplementing the logic would produce estimators that reject
    the right input for the right reason and still fail conformance. Reusing
    the validator makes the messages correct by construction, and means they
    match what a user has already seen from every other scikit-learn
    estimator.

    The one deliberate departure: ``NaN`` is permitted, since marginalising
    over it is the entire purpose of the library. Infinity is not, because
    there is no conditional distribution that makes an infinite observation
    meaningful.

    Returns
    -------
    (X, y) validated, with y unchanged when it was not supplied.
    """
    from sklearn.utils.validation import check_array

    allow_nan = {_allow_nan_kwarg(): "allow-nan"}

    check_no_complex_data(X)
    # min_samples is 2 at fit and 1 at predict. Requiring two rows at
    # predict was a regression introduced here: predicting for a single
    # row is ordinary use, and scikit-learn's subset-invariance check
    # does exactly that.
    X = check_array(X, accept_sparse=False, dtype="numeric",
                    ensure_2d=True, ensure_min_samples=min_samples,
                    ensure_min_features=1, **allow_nan)

    if y is None:
        if check_y:
            raise ValueError(
                "%s requires y to be passed, but the target y is None."
                % type(estimator).__name__)
        return X, None

    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        # scikit-learn requires the flattening to be announced: a caller who
        # passes a column vector has probably made a shape error, and silently
        # accepting it hides that.
        import warnings as _w
        from sklearn.exceptions import DataConversionWarning
        _w.warn("A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example "
                "using ravel().", DataConversionWarning, stacklevel=2)
        y = y.ravel()

    # NaN in y is meaningful here: those rows still inform the feature
    # distribution. Infinity never is, and scikit-learn checks for it.
    if y.dtype.kind in "fc" and np.isinf(np.asarray(y, dtype=float)).any():
        raise ValueError("Input y contains infinity or a value too large for "
                         "dtype('float64').")

    # Some absent outcomes are the point of this library; all of them are not.
    # With nothing observed there is no supervised problem left, and the three
    # families disagreed about it: MissRidgeRegressor refused, MissLinear
    # fitted and then predicted NaN from NaN coefficients, and
    # MissMixedRegressor fitted a zero model and predicted finite numbers from
    # no data at all. prefit_check has worded this refusal since it was
    # written; the estimators simply did not all call it. Refusing here means
    # every estimator inherits the same answer.
    if y.size and _isnan_safe(y).all():
        raise ValueError(
            "%s: y is entirely NaN, so there is no observed outcome to fit. "
            "Absent entries in y are supported and still inform the feature "
            "distribution, but at least one outcome must be observed."
            % type(estimator).__name__)

    if is_classifier_safe(estimator):
        from sklearn.utils.multiclass import type_of_target
        observed = y[~_isnan_safe(y)] if y.dtype.kind == "f" else y
        if observed.size:
            kind = type_of_target(observed)
            if kind in ("continuous", "continuous-multioutput"):
                raise ValueError(
                    "Unknown label type: %s. %s expects a classification "
                    "target, but y looks continuous."
                    % (kind, type(estimator).__name__))

    if len(y) != X.shape[0]:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: "
            "[%d, %d]" % (X.shape[0], len(y)))
    return X, y


class EmptyFeatureError(ValueError):
    """A feature column contains no observed values.

    A ``ValueError`` subclass, so existing code that catches ``ValueError``
    keeps working, while callers that want to detect this specific condition
    can do so without matching on message text.
    """


def is_missing_label(y) -> np.ndarray:
    """Boolean mask of the absent entries in a label vector.

    A label can be absent in four ways and they are not interchangeable:
    float ``nan``, Python ``None``, ``pandas.NA`` and ``pandas.NaT``. Only the
    first answers to ``np.isnan``; ``None`` is not a float, and ``pd.NA``
    raises ``TypeError`` when its truth value is taken, so a test written for
    one of them silently or loudly fails on the others.

    This is the single definition used by the whole library. It was previously
    written once correctly in ``_multiclass`` and once incorrectly here, and
    every classifier used the incorrect one.
    """
    y = np.asarray(y)
    try:
        return np.isnan(y.astype(float))
    except (ValueError, TypeError):
        # Object or extension dtype: inspect each entry, because pd.NA raises
        # rather than answering when compared.
        mask = np.zeros(y.shape[0], dtype=bool)
        for i, v in enumerate(y.ravel()):
            try:
                mask[i] = v is None or bool(v != v)
            except (TypeError, ValueError):
                mask[i] = True      # pd.NA, pd.NaT and anything like them
        return mask


def encode_labels(estimator, y):
    """Map non-numeric class labels to integers, remembering the mapping.

    A classifier that only accepts 0 and 1 forces the caller to encode labels
    by hand and to decode predictions afterwards, which no other scikit-learn
    classifier requires. The internals here genuinely need numbers, since the
    likelihood is written over a numeric target, so the encoding happens at
    the boundary instead of being pushed onto the user.

    NaN is preserved rather than encoded. A missing label is not a class, and
    those rows still contribute to the feature distribution, so turning them
    into a category would silently invent an outcome.

    Returns
    -------
    (y_numeric, classes) where classes is None when no encoding was needed.
    """
    y = np.asarray(y)
    if y.dtype.kind in "biufc":
        return y, None

    # One test for absence, shared with the rest of the library. The former
    # test here was `isinstance(v, float) and np.isnan(v)`, which let None and
    # pandas NA through to np.unique and raised on the sort.
    missing = is_missing_label(y)
    observed = np.array([v for v, m in zip(y.ravel(), missing) if not m],
                        dtype=object)
    classes = np.unique(observed)
    lookup = {c: i for i, c in enumerate(classes)}
    out = np.full(y.shape[0], np.nan, dtype=float)
    for i, (v, m) in enumerate(zip(y.ravel(), missing)):
        if m:
            continue            # absent stays absent, it is not a category
        out[i] = lookup[v]
    return out, classes


def check_not_sparse(X, estimator_name="estimator"):
    """Refuse sparse input in scikit-learn's wording.

    These estimators are dense-only, and the reason is semantic rather than an
    implementation gap: a sparse matrix cannot distinguish a structural zero
    from a missing entry, and that distinction is the whole subject of this
    library. Storing NaN sparsely defeats the point of sparsity in any case.

    Refused here, before coercion. Letting a sparse matrix reach np.asarray
    produced a 0-d object array and an error several layers down that named
    neither sparsity nor the estimator.
    """
    if hasattr(X, "toarray") and hasattr(X, "tocsr"):
        raise TypeError(
            "A sparse matrix was passed, but dense data is required by %s. "
            "Use X.toarray() to convert to a dense numpy array. Note that a "
            "sparse matrix cannot represent missing entries distinctly from "
            "zeros, which this estimator requires." % estimator_name)


def check_no_complex_data(X):
    """Refuse complex input, with the wording scikit-learn expects.

    Every estimator here casts to float internally, which silently discards
    the imaginary part: a caller who passes complex data gets numbers back
    that answer a question they did not ask. Refusing is the only safe
    response.

    The message text matters and is not arbitrary. scikit-learn's
    ``check_complex_data`` asserts on the phrase, so an estimator that raises
    for the right reason with different wording still fails conformance. Two
    MissLearn classes were in exactly that position: they raised, and were
    rejected for the message.
    """
    if np.iscomplexobj(np.asarray(X)):
        raise ValueError("Complex data not supported\n%s" % (np.asarray(X),))


def check_no_empty_features(X, feature_names=None, estimator_name="estimator"):
    """Raise if any column of ``X`` is entirely missing.

    FIML marginalises a missing entry over the conditional distribution implied
    by the observed entries. A column with no observed values anywhere has no
    such distribution: there is nothing to condition on and nothing to
    estimate. The honest response is to say so.

    Before this check existed the sixteen estimators disagreed about it, which
    is exactly the drift this module was created to stop. Four regressors fitted
    happily and returned ``NaN`` predictions; four classifiers returned finite
    class labels whose ``predict_proba`` was entirely ``NaN``, so the label path
    concealed the failure; two raised a message about NaN input that did not
    mention which column caused it; and three coped. Silent ``NaN`` is the worst
    of those outcomes, because it propagates into whatever consumes the
    prediction instead of stopping the pipeline.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    feature_names : sequence of str, optional
        Used to name the offending columns. Falls back to positional indices.
    estimator_name : str
        Named in the message so the error is traceable in a pipeline.

    Raises
    ------
    EmptyFeatureError
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return

    empty = np.flatnonzero(np.all(np.isnan(X), axis=0))
    if empty.size == 0:
        return

    if feature_names is not None and len(feature_names) >= X.shape[1]:
        labels = ["'%s' (column %d)" % (feature_names[j], j) for j in empty]
    else:
        labels = ["column %d" % j for j in empty]

    listed = ", ".join(labels[:8]) + (", ..." if len(labels) > 8 else "")
    raise EmptyFeatureError(
        "%s cannot fit: %d feature%s ha%s no observed values at all (%s). "
        "A fully missing column has no conditional distribution to marginalise "
        "over, so it carries no information and cannot be estimated. Drop the "
        "column, or use MissRecommender, whose preprocessing_ attribute reports "
        "which columns to drop rather than model."
        % (estimator_name, empty.size, "s" if empty.size > 1 else "",
           "ve" if empty.size > 1 else "s", listed))


def check_n_features(estimator, X):
    """Raise if ``X`` has a different number of columns than ``fit`` saw.

    Part of the scikit-learn estimator contract: predicting with the wrong
    number of features is a caller error and must be reported as one, not
    absorbed. Silently accepting a mismatched matrix produces predictions that
    correspond to no model the user has fitted.

    The message follows scikit-learn's wording, so a user who has seen it from
    any other estimator recognises it immediately.

    Raises
    ------
    ValueError
    """
    expected = getattr(estimator, "n_features_in_", None)
    if expected is None:
        return                       # not fitted, or a stage that has no notion
    X = np.asarray(X)
    if X.ndim != 2:
        return                       # shape errors belong to the caller's own path
    if X.shape[1] != expected:
        raise ValueError(
            "X has %d features, but %s is expecting %d features as input."
            % (X.shape[1], type(estimator).__name__, expected))


def _is_classifier(estimator) -> bool:
    return is_classifier_safe(estimator)


def route_multiclass(estimator, X, y, fit_kwargs) -> Optional[object]:
    """Fit a one-vs-rest wrapper when a binary classifier is given K > 2.

    The binary classifiers are binary by construction: a logistic link, a
    two-class Gaussian discriminant, a binary SVM. Multi-class support was
    available through ``MissMulticlass``, but only to users who knew to reach
    for it, and ``check_estimator`` fails any classifier that rejects three
    classes. Routing internally removes both problems at once and matches what
    a scikit-learn user expects a classifier to do.

    Two guards prevent recursion, and both are needed. Each sub-estimator of
    a routing is marked ``_is_multiclass_member`` and declined, which is why
    decomposing into K binary problems terminates. ``MissMulticlass`` itself
    is marked ``_is_multiclass_router`` and declined, which matters because
    it is a classifier: without that, fitting one on three classes would
    route it into another router until the stack ended. The second guard was
    unnecessary only while ``MissMulticlass`` was a plain object rather than
    a scikit-learn estimator.

    Returns
    -------
    The fitted ``MissMulticlass``, or ``None`` when no routing is needed, in
    which case the caller proceeds with its own fit.
    """
    if not _is_classifier(estimator):
        return None
    if getattr(estimator, "_is_multiclass_member", False):
        return None                       # a sub-problem of an outer routing
    if getattr(estimator, "_is_multiclass_router", False):
        return None                       # already a router; routing it recurses

    y_arr = np.asarray(y)
    observed = y_arr[~_isnan_safe(y_arr)]
    if observed.size == 0:
        return None
    n_classes = np.unique(observed).size
    if n_classes <= 2:
        return None

    import copy
    from ._multiclass import MissMulticlass

    template = copy.deepcopy(estimator)
    template._is_multiclass_member = True
    router = MissMulticlass(template)
    router.fit(X, y, **(fit_kwargs or {}))
    return router


def _isnan_safe(a):
    """isnan that tolerates non-float dtypes such as string labels."""
    try:
        return np.isnan(a)
    except (TypeError, ValueError):
        return np.zeros(np.shape(a), dtype=bool)


def check_penalty(value, name="alpha", estimator_name="estimator"):
    """Reject a negative regularisation strength.

    A negative penalty is not weak regularisation, it is a reward for large
    coefficients: the objective is unbounded below and the fit runs away.
    MissLASSOClassifier and MissRidgeClassifier accepted alpha=-1 and
    returned coefficients of 16.7 and 18.0 where the true values were about
    2, while their regressor siblings refused the same input. The regressors
    were not being careful; they seed from sklearn's Lasso and Ridge, whose
    own parameter validation happened to catch it. Relying on that is how
    two of four estimators ended up guarded and two did not, so the check
    lives here and every one of them calls it.

    Parameters
    ----------
    value : float
        The penalty strength to check.
    name : str
        Parameter name, for the message.
    estimator_name : str
        Estimator name, for the message.

    Raises
    ------
    ValueError
        If *value* is negative or not finite.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "%s: %s must be a non-negative number, got %r."
            % (estimator_name, name, value))
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(
            "%s: %s must be >= 0, got %r. A negative penalty rewards large "
            "coefficients rather than shrinking them, so the objective has "
            "no minimum and the fit does not mean anything."
            % (estimator_name, name, value))
    return v


def check_positive_int(value, name, estimator_name="estimator", minimum=1):
    """Reject a count that cannot describe anything.

    A count of zero or below does not fail loudly, it quietly changes the
    model. n_neighbors=0 returned an R-squared of exactly 0 with every
    prediction equal to the training mean, and n_neighbors=-1 fitted a
    different k from the one asked for and looked entirely plausible doing
    it. On a polynomial kernel, degree=0 collapsed every prediction to the
    same value and degree=-1 reached an R-squared of -3.7e6.

    Parameters
    ----------
    value : int
        The count to check.
    name : str
        Parameter name, for the message.
    estimator_name : str
        Estimator name, for the message.
    minimum : int
        Smallest permitted value, 1 by default.

    Raises
    ------
    ValueError
        If *value* is not an integer of at least *minimum*.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                "%s: %s must be an integer >= %d, got %r."
                % (estimator_name, name, minimum, value))
        if as_int != value:
            raise ValueError(
                "%s: %s must be a whole number >= %d, got %r."
                % (estimator_name, name, minimum, value))
        value = as_int
    if value < minimum:
        raise ValueError(
            "%s: %s must be >= %d, got %r. A smaller value does not weaken "
            "the model, it silently replaces it with a different one."
            % (estimator_name, name, minimum, value))
    return int(value)


def public_parameter_name(estimator, *candidates):
    """The name among *candidates* that this estimator's constructor accepts.

    ``MissRidgeClassifier`` exposes ``alpha`` and aliases it to ``l2_reg``
    with a property, because it inherits MissLogistic's likelihood and that
    is what the likelihood calls it. The guard therefore refused
    ``alpha=-1`` with a message about ``l2_reg``, naming a parameter the
    caller had never heard of and cannot set. Reporting the name the
    constructor takes costs one signature lookup and means the message
    matches the code the user wrote.
    """
    import inspect as _inspect
    params = _inspect.signature(type(estimator).__init__).parameters
    for candidate in candidates:
        if candidate in params:
            return candidate
    return candidates[0]


def check_positive_float(value, name, estimator_name="estimator", what=""):
    """Reject a quantity that has to be strictly positive.

    Variances, widths and scales are all of this shape: zero is not a small
    one, it is the absence of the thing. ``noise_var_init=0`` and
    ``noise_var_init=-1`` were both accepted by the Gaussian process, which
    then fitted to an r-squared of 0.6787 either way, so nothing about the
    output said the starting variance was impossible.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: %s must be a positive number, got %r."
                         % (estimator_name, name, value))
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("%s: %s must be > 0, got %r.%s"
                         % (estimator_name, name, value,
                            (" " + what) if what else ""))
    return v


def check_common_parameters(estimator):
    """Validate the parameters that several estimators share.

    Called from two places, because there are two kinds of estimator here and
    only one of them runs the usual fit machinery. The concrete classes reach
    this through ``MissBase._store_fit_metadata``, which every one of their
    fits already calls. The task dispatchers do not: they construct a
    concrete estimator and delegate, so anything they never pass on is never
    checked. ``n_quadrature`` is exactly that case, since it belongs to the
    classifier's quadrature and a dispatcher resolving to a regressor drops
    it silently, leaving ``MissRidge(n_quadrature=0)`` accepted on a
    regression target.

    Each parameter is only checked where it exists, so this can be called on
    anything without knowing which of them it takes.
    """
    who = type(estimator).__name__
    if hasattr(estimator, 'copula'):
        check_copula(estimator.copula, who)
    if hasattr(estimator, 'max_iter'):
        check_positive_int(estimator.max_iter, 'max_iter', who)
    if hasattr(estimator, 'tol'):
        check_tolerance(estimator.tol, 'tol', who)
    if hasattr(estimator, 'n_quadrature'):
        check_positive_int(estimator.n_quadrature, 'n_quadrature', who)
    if hasattr(estimator, 'n_restarts'):
        # Zero restarts is a real choice: fit once from the seed and stop.
        check_positive_int(estimator.n_restarts, 'n_restarts', who, minimum=0)
    if hasattr(estimator, 'noise_var_init'):
        check_positive_float(estimator.noise_var_init, 'noise_var_init', who,
                             'A variance of zero is not a small variance.')
    if hasattr(estimator, 'max_iter_newton'):
        check_positive_int(estimator.max_iter_newton, 'max_iter_newton', who)
    if hasattr(estimator, 'fe_ridge'):
        # A negative ridge rewards large coefficients instead of shrinking
        # them, which is the unbounded-objective failure check_penalty
        # documents.
        check_penalty(estimator.fe_ridge, 'fe_ridge', who)


def check_tolerance(value, name="tol", estimator_name="estimator"):
    """Reject a convergence tolerance that is not a positive distance.

    A tolerance is how close is close enough, so zero means never close
    enough and a negative value means a distance smaller than one that
    cannot exist. Neither describes a stopping rule, and scipy's optimisers
    do not treat them as errors, they just behave oddly.

    Mostly the damage was invisible: at ``tol=0`` or ``tol=-1e-3`` nine of
    the eleven estimators that take the parameter returned exactly the same
    r-squared as at the default. MissLASSORegressor did not. A negative
    tolerance took it from 0.9869 to **-0.0007**, which is a fit that
    explains nothing, reported with no warning and ``converged`` unset. That
    is the case this exists for, and it is why "measured benign" was not a
    safe conclusion to draw from the other ten.

    Parameters
    ----------
    value : float
        The tolerance to check.
    name : str
        Parameter name, for the message.
    estimator_name : str
        Estimator name, for the message.

    Raises
    ------
    ValueError
        If *value* is not a finite number strictly greater than zero.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: %s must be a positive number, got %r."
                         % (estimator_name, name, value))
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(
            "%s: %s must be > 0, got %r. A tolerance is how close counts as "
            "converged, so zero can never be met and a negative one asks for "
            "a distance that does not exist."
            % (estimator_name, name, value))
    return v


def check_choice(value, options, name, estimator_name="estimator"):
    """Reject a string option that is not one of the documented choices.

    An unrecognised option is the quietest failure in the library, because
    the code that consumes it asks ``== 'mahalanobis'`` or ``== 'full'`` and
    treats everything else as the other branch. Nothing is unset and nothing
    is missing, so nothing complains; the user simply gets a different model
    from the one they asked for. Three options behaved this way. A typo in
    MissNeighbors ``metric`` silently gave euclidean where mahalanobis was
    asked for, and predictions moved by up to 1.5. A typo in ``weights``
    silently gave distance weighting, moving predictions by up to 2.9. Worst
    of the three, MissBayes ``structure`` tested only for ``'full'``, so any
    misspelling of it, ``'ful'`` included, selected naive independence and
    moved predictions by up to 2.8.

    Parameters
    ----------
    value : str
        The option value to check.
    options : sequence of str
        The permitted values.
    name : str
        Parameter name, for the message.
    estimator_name : str
        Estimator name, for the message.

    Raises
    ------
    ValueError
        If *value* is not one of *options*.
    """
    if value in options:
        return value
    allowed = ", ".join(repr(o) for o in options)
    message = ("%s: %s must be one of %s; got %r."
               % (estimator_name, name, allowed, value))
    if isinstance(value, str):
        near = [o for o in options if o.lower() == value.lower().strip()]
        if near:
            message += " Did you mean %r?" % near[0]
    raise ValueError(message)


def check_copula(value, estimator_name="estimator"):
    """Reject a ``copula`` string that is not ``'auto'``.

    ``copula`` takes a bool or the word ``'auto'``, and it is consumed as
    ``if self.copula == 'auto': ... elif self.copula:``. Any other string
    misses the first test and is then merely truthy, so it silently forces
    the transform on. ``copula='atuo'`` fitted without complaint and moved
    predictions by 1.4 against ``copula=False`` on plain data. This is the
    widest-reaching member of the family in :func:`check_choice`'s docstring,
    because ``copula`` is a parameter of all twenty-three estimators.

    Only strings are inspected. A bool, or anything else whose truthiness is
    what the caller intended, is left alone: the defect is specifically that
    a misspelling of the one legal word cannot be distinguished from a
    deliberate True.

    Parameters
    ----------
    value : bool or str
        The value to check.
    estimator_name : str
        Estimator name, for the message.

    Raises
    ------
    ValueError
        If *value* is a string other than ``'auto'``.
    """
    if isinstance(value, str) and value != 'auto':
        message = ("%s: copula must be True, False, or 'auto'; got %r."
                   % (estimator_name, value))
        if value.lower().strip() == 'auto':
            message += " Did you mean 'auto'?"
        raise ValueError(message)
    return value


def check_feature_names(estimator, names):
    """The columns at predict must be the columns the model was fitted on.

    Recording ``feature_names_in_`` and never checking it is worse than not
    recording it, because the count still matches and nothing complains. A
    DataFrame whose columns are in a different order is the ordinary way this
    happens, through a reindex, a join, or a select written from memory, and
    it produced predictions correlated -1.000 with the correct ones: on a
    model with coefficients (3.0, -0.015, -3.0), swapping the first and third
    columns negates every prediction. scikit-learn refuses exactly this case
    and this now matches it.

    An ndarray at predict is allowed against a model fitted on a DataFrame:
    there are no names to disagree, and the feature count is checked
    separately.

    Parameters
    ----------
    estimator : fitted estimator
        Read for ``feature_names_in_``; absent means nothing to check.
    names : list of str or None
        Column names of the data being predicted, or None if it had none.

    Raises
    ------
    ValueError
        If both sets of names exist and differ in content or in order.
    """
    # Only names that actually came from the data can disagree with anything.
    # feature_names_in_ is always populated, with synthetic X0..Xp when the
    # fit saw a plain array, and comparing those against a DataFrame's real
    # columns rejects the perfectly ordinary case of fitting on an ndarray
    # and predicting on a frame. _feature_names_from_data is the flag that
    # tells the two apart.
    fitted = getattr(estimator, '_feature_names_from_data', None)
    if not fitted or names is None:
        return
    fitted = [str(c) for c in fitted]
    names = [str(c) for c in names]
    if fitted == names:
        return

    who = type(estimator).__name__
    if sorted(fitted) == sorted(names):
        raise ValueError(
            "%s was fitted on columns %s but is being asked to predict on "
            "the same columns in a different order, %s. The values would be "
            "read into the wrong coefficients, so the predictions would be "
            "wrong without being obviously wrong. Reorder the columns to "
            "match, for example X[list(model.feature_names_in_)]."
            % (who, fitted, names))
    missing = [c for c in fitted if c not in names]
    unexpected = [c for c in names if c not in fitted]
    detail = []
    if missing:
        detail.append("missing %s" % missing)
    if unexpected:
        detail.append("unexpected %s" % unexpected)
    raise ValueError(
        "%s was fitted on columns %s but received %s (%s)."
        % (who, fitted, names, "; ".join(detail)))
