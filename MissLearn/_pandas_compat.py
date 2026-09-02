"""
_pandas_compat.py  --  Transparent pandas DataFrame / Series support.

Provides:
    _coerce_X(X)  ->  (np.ndarray, feature_names_or_None)
    _coerce_y(y)  ->  np.ndarray
    enable_dataframe_support(cls)
        Patches fit / predict / predict_proba / predict_interval /
        decision_function / score on a class so that DataFrames and Series
        are accepted and silently converted, and feature names are stored
        in self.feature_names_in_ during fit.

Applied at import time in __init__.py to every exported MissLearn model
class.  No model files need to be modified.

Design notes
------------
* Detection is duck-typed (no hard pandas import required at patch time).
  Pandas is only imported inside the model methods if a DataFrame is
  actually passed.

* The wrapper stores feature_names_in_ on the *instance*, not the class,
  so multiple fitted instances are independent.

* summary() methods continue to use X0/X1/... labels internally.
  Users can access the name mapping via model.feature_names_in_.

* Applying enable_dataframe_support() twice to the same class is a no-op
  (idempotent via _PANDAS_ENABLED sentinel attribute).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import functools
import numpy as np
from ._sklearn_compat import is_classifier_safe, is_regressor_safe


# ---------------------------------------------------------------------------
# Coercion utilities
# ---------------------------------------------------------------------------

def _is_dataframe(obj) -> bool:
    """True if obj looks like a pandas DataFrame (duck-typed)."""
    return (type(obj).__name__ == 'DataFrame'
            and hasattr(obj, 'to_numpy')
            and hasattr(obj, 'columns'))


def _is_series(obj) -> bool:
    """True if obj looks like a pandas Series (duck-typed)."""
    return (type(obj).__name__ == 'Series'
            and hasattr(obj, 'to_numpy')
            and hasattr(obj, 'name'))


def _coerce_X(X) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Convert X to float64 ndarray.

    Returns
    -------
    X_arr : ndarray of shape (n, p)
    names : list of str if X was a DataFrame, else None
    """
    if _is_dataframe(X):
        names = [str(c) for c in X.columns]
        try:
            return X.to_numpy(dtype=float, na_value=np.nan), names
        except (ValueError, TypeError):
            # Non-numeric columns (e.g. string categoricals destined for
            # MissPreprocessor): pass through as object; downstream
            # validation still rejects them where floats are required.
            return X.to_numpy(), names
    try:
        return np.asarray(X, dtype=float), None
    except (ValueError, TypeError):
        return np.asarray(X, dtype=object), None


def _coerce_y(y) -> np.ndarray:
    """Convert y to a 1-D float64 ndarray; non-numeric labels (e.g. string
    classes for MissMulticlass) pass through unconverted."""
    if _is_series(y) or _is_dataframe(y):
        try:
            return y.to_numpy(dtype=float, na_value=np.nan).ravel()
        except (ValueError, TypeError):
            return y.to_numpy().ravel()
    try:
        return np.asarray(y, dtype=float)
    except (ValueError, TypeError):
        return np.asarray(y)


def _coerce_groups(groups):
    """Convert groups to ndarray (preserves dtype -- may be str or int)."""
    if groups is None:
        return None
    if _is_series(groups) or _is_dataframe(groups):
        return groups.to_numpy().ravel()
    return np.asarray(groups)


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------


def _supervised(est):
    """True for estimators whose fit takes a target."""
    return is_classifier_safe(est) or is_regressor_safe(est)


def _requires_y(est):
    """Supervised estimators require y; scikit-learn checks that y=None raises."""
    return _supervised(est)


def _wrap_fit(method):
    """Wrap fit(...) to accept DataFrames / Series transparently.

    Supports every fit arity used across MissLearn without assuming a fixed
    one, so the wrapper never changes the underlying signature contract:

        fit(X)              unsupervised   (e.g. MissImputer)
        fit(X, y)           supervised models
        fit(X, y, groups)   mixed-effects models

    plus ``y`` / ``groups`` passed by keyword.  Arguments are forwarded
    unchanged except for DataFrame/Series -> ndarray coercion: the first
    positional after X is treated as ``y`` and the second as ``groups``.
    """
    def _fit(self, X, *args, **kwargs):
        from ._conformance import (check_no_complex_data,
                                   check_no_empty_features, check_not_sparse,
                                   encode_labels, route_multiclass,
                                   validate_input)

        check_no_complex_data(X)
        check_not_sparse(X, type(self).__name__)
        X_arr, names = _coerce_X(X)
        args = list(args)
        if len(args) >= 1 and args[0] is not None:   # positional y
            args[0] = _coerce_y(args[0])
        if len(args) >= 2:                       # positional groups
            args[1] = _coerce_groups(args[1])
        if kwargs.get('y') is not None:
            kwargs = dict(kwargs)
            kwargs['y'] = _coerce_y(kwargs['y'])
        if kwargs.get('groups') is not None:
            kwargs = dict(kwargs)
            kwargs['groups'] = _coerce_groups(kwargs['groups'])
        # Publish the column names BEFORE delegating as well as after. Setting
        # them only afterwards left anything that consults them *during* fit
        # working from placeholder labels: MissPreprocessor builds its
        # compatibility report inside fit, so every warning named 'X4' rather
        # than the column the user would recognise, even though the attribute
        # looked correct once fit returned.
        if names is not None:
            self.feature_names_in_ = names

        # Behaviour that must be identical across every estimator is applied
        # here rather than in each class, because sixteen independent
        # implementations is what allowed them to drift apart. See
        # _conformance.py.
        # scikit-learn's own validator, with NaN permitted. Placed here so
        # every estimator rejects the same malformed input in the same words.
        # Unsupervised fits (MissImputer) legitimately have no y.
        self._label_classes_ = None
        if _supervised(self) and is_classifier_safe(self):
            _raw = args[0] if args else kwargs.get('y')
            if _raw is not None:
                _enc, _cls = encode_labels(self, _raw)
                if _cls is not None:
                    self._label_classes_ = _cls
                    if args:
                        args[0] = _enc
                    else:
                        kwargs = dict(kwargs); kwargs['y'] = _enc
        _y_present = bool(args) or kwargs.get('y') is not None
        # An estimator whose job is to accept raw data and make it numeric
        # cannot be handed only numeric data. _coerce_X already passes an
        # object array through for exactly this case, with a comment naming
        # MissPreprocessor, and then this validation rejected it one step
        # later: a DataFrame with a string categorical column failed with
        # "could not convert string to float" before MissPreprocessor.fit
        # was ever entered, which is the one input the class exists to
        # handle. Such an estimator validates and encodes for itself.
        if getattr(self, '_ACCEPTS_RAW_INPUT', False):
            return method(self, X_arr, *args, **kwargs)
        if _supervised(self):
            _yv = args[0] if args else kwargs.get('y')
            X_arr, _yv = validate_input(self, X_arr, _yv,
                                        check_y=_y_present or _requires_y(self))
            if args:
                args[0] = _yv
            elif kwargs.get('y') is not None:
                kwargs = dict(kwargs); kwargs['y'] = _yv
        else:
            X_arr, _ = validate_input(self, X_arr, None, check_y=False)

        check_no_empty_features(
            X_arr, feature_names=names, estimator_name=type(self).__name__)

        # A binary classifier handed three or more classes routes through
        # one-vs-rest rather than refusing. Cleared first so that refitting an
        # estimator on binary data does not leave a stale router behind.
        self._multiclass_router_ = None
        y_for_route = args[0] if args else kwargs.get('y')
        if y_for_route is not None:
            route_kwargs = dict(kwargs)
            route_kwargs.pop('y', None)
            if len(args) >= 2 and 'groups' not in route_kwargs:
                route_kwargs['groups'] = args[1]
            router = route_multiclass(self, X_arr, y_for_route, route_kwargs)
            if router is not None:
                self._multiclass_router_ = router
                # Report the caller's own labels, not the integer codes the
                # router was fitted on. Restoring these only on the
                # non-routed path meant a three-class string target came back
                # as [0., 1., 2.], which is the combination of two features
                # that each worked alone.
                self.classes_ = (self._label_classes_
                                 if getattr(self, '_label_classes_', None) is not None
                                 else router.classes_)
                _its = [getattr(e, 'n_iter_', None)
                        for e in getattr(router, 'estimators_', [])]
                _its = [i for i in _its if i is not None]
                if _its:
                    self.n_iter_ = int(max(_its))
                self.n_features_in_ = X_arr.shape[1]
                if names is not None:
                    self.feature_names_in_ = names
                    self._feature_names_from_data = list(names)
                return self

        result = method(self, X_arr, *args, **kwargs)
        if getattr(self, '_label_classes_', None) is not None:
            self.classes_ = self._label_classes_
        # Set it again afterwards, so a model that assigns its own placeholder
        # labels during fit still ends up with the true column names.
        if names is not None:
            self.feature_names_in_ = names
            # Also after the fit, not only before it: the inner
            # _validate_and_convert sees an ndarray by this point and clears
            # the provenance flag, so anything set beforehand is lost.
            self._feature_names_from_data = list(names)
        return result
    # Preserve the wrapped signature. scikit-learn introspects fit to check
    # that the second parameter is y, and to build metadata routing. Without
    # this the wrapper presents every estimator as fit(X, *args, **kwargs),
    # so check_estimator rejected the entire library with "Expected y or Y as
    # second argument". functools.wraps sets __wrapped__, which is what makes
    # inspect.signature look through to the real signature.
    functools.wraps(method)(_fit)
    _fit.__name__ = 'fit'
    return _fit


def _raw_attribute(cls, name):
    """The class attribute itself, without letting a descriptor resolve it.

    ``getattr(cls, name)`` runs ``__get__``, which for an ``available_if``
    method returns the underlying function and loses the predicate. Walking
    the MRO gives the descriptor object, so the caller can tell the two
    apart. Returns None when no class in the MRO defines the name, which is
    the case a plain ``hasattr`` would also have to handle, except that
    ``hasattr`` would additionally be satisfied by ``__getattr__``
    delegation and wrap a method that does not belong to this class at all.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return None


def _wrap_predict(method):
    """Wrap predict(X, **kw) / predict_proba / predict_interval /
    decision_function to accept DataFrames."""
    def _pred(self, X, *args, **kwargs):
        # Positional args (e.g. MissImputer.transform(X, y),
        # MissMixed.predict(X, groups)) are forwarded unchanged: the
        # methods coerce them internally, and converting here could
        # change dtypes (e.g. string group labels).
        from ._conformance import (check_n_features, check_no_complex_data,
                                   check_feature_names)

        check_no_complex_data(X)
        X_arr, _names = _coerce_X(X)
        # feature_names_in_ was recorded at fit and never read, so a frame
        # with the right columns in the wrong order sailed through the count
        # check and produced silently wrong answers.
        check_feature_names(self, _names)
        if 'groups' in kwargs and kwargs['groups'] is not None:
            kwargs = dict(kwargs)
            kwargs['groups'] = _coerce_groups(kwargs['groups'])

        # Part of the scikit-learn contract, and applied here for the same
        # reason as everything else in this wrapper: sixteen estimators cannot
        # be relied on to implement one check identically.
        # Same validator as fit, so predict refuses the same malformed
        # input in the same words rather than failing further in.
        # An estimator that accepts raw input at fit has to accept it at
        # predict too, for the same reason and with the same consequence: the
        # column it one-hot encoded during fit is still a string column when
        # the caller predicts on it.
        raw_ok = getattr(self, '_ACCEPTS_RAW_INPUT', False)
        if not raw_ok and getattr(self, 'n_features_in_', None) is not None:
            from ._conformance import validate_input as _vi
            X_arr, _ = _vi(self, X_arr, None, check_y=False, min_samples=1)
        if not raw_ok:
            check_n_features(self, X_arr)

        # If fit routed a multi-class problem through one-vs-rest, every
        # prediction method has to follow it there. Delegating in fit alone
        # would leave predict answering from an unfitted binary model.
        router = getattr(self, '_multiclass_router_', None)
        if router is not None:
            target = getattr(router, method.__name__, None)
            out = (target(X_arr, *args, **kwargs) if target is not None
                   else method(self, X_arr, *args, **kwargs))
        else:
            out = method(self, X_arr, *args, **kwargs)

        # Decode once, after whichever path produced the output. Doing it only
        # on the direct path meant a routed multi-class fit returned integer
        # codes while classes_ correctly reported the original labels, so the
        # two disagreed: each feature worked alone and the combination did not.
        _cls = getattr(self, '_label_classes_', None)
        if _cls is not None and method.__name__ == 'predict':
            idx = np.asarray(out)
            if idx.dtype.kind in 'fiu':
                out = np.asarray(_cls)[idx.astype(int)]
        return out
    _pred.__name__ = method.__name__
    _pred.__doc__  = method.__doc__
    return _pred


def _wrap_score(method):
    """Wrap score(X, y, **kw) to accept DataFrames / Series."""
    def _score(self, X, y, **kwargs):
        X_arr, _ = _coerce_X(X)
        y_arr    = _coerce_y(y)
        if 'groups' in kwargs and kwargs['groups'] is not None:
            kwargs = dict(kwargs)
            kwargs['groups'] = _coerce_groups(kwargs['groups'])
        return method(self, X_arr, y_arr, **kwargs)
    _score.__name__ = 'score'
    _score.__doc__  = method.__doc__
    return _score


# ---------------------------------------------------------------------------
# Class patcher
# ---------------------------------------------------------------------------

_PREDICT_METHODS = (
    'predict', 'predict_proba', 'predict_interval', 'decision_function',
    'transform',
)

def enable_dataframe_support(cls):
    """
    Patch a MissLearn model class to accept pandas DataFrames / Series.

    Idempotent: calling twice on the same class has no effect.

    Parameters
    ----------
    cls : model class to patch (modified in-place)

    Returns
    -------
    cls (for use as a decorator)
    """
    if getattr(cls, '_PANDAS_ENABLED', False):
        return cls

    if hasattr(cls, 'fit'):
        cls.fit = _wrap_fit(cls.fit)

    for name in _PREDICT_METHODS:
        raw = _raw_attribute(cls, name)
        if raw is None:
            continue
        check = getattr(raw, 'check', None)
        underlying = getattr(raw, 'fn', None)
        if check is not None and underlying is not None:
            # An available_if method. Reading it off the class hands back the
            # bare function, so wrapping the result of getattr and assigning
            # that would replace the descriptor with a plain function and
            # throw the gating away. The task dispatchers use available_if to
            # hide predict_proba on a regression fit, and this is where that
            # was being silently undone: the attribute came back visible and
            # raised only when called, which is what four scikit-learn checks
            # per dispatcher were failing on. Rebuild the descriptor around
            # the wrapped function instead.
            from sklearn.utils.metaestimators import available_if
            setattr(cls, name, available_if(check)(_wrap_predict(underlying)))
        else:
            setattr(cls, name, _wrap_predict(raw))

    if hasattr(cls, 'score'):
        cls.score = _wrap_score(cls.score)

    cls._PANDAS_ENABLED = True
    return cls
