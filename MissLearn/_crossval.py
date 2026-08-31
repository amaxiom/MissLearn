"""
_crossval.py  --  Cross-validation utilities that preserve NaN.

sklearn's cross_val_score / KFold pipeline calls check_array() which
rejects NaN by default.  The utilities here bypass that validation step
entirely, so MissLearn models (which handle NaN internally via FIML) work
without modification.

Classes
-------
MissKFold               -- K-fold splitter, no NaN stripping
MissStratifiedKFold     -- Stratified K-fold (stratifies on non-NaN y)

Functions
---------
miss_cross_val_score    -- Drop-in replacement for sklearn's cross_val_score
miss_cross_validate     -- Multi-metric version; optionally returns train scores

All functions accept groups= for MissMixed-style estimators and thread the
corresponding group subsets to fit() and optionally score().

Scoring
-------
Passing scoring=None uses the estimator's own .score() method.
Passing a callable receives (estimator, X_test, y_test) and returns a float.
Passing a string uses a small built-in registry of common metrics:

    Regression  : 'r2', 'neg_mse', 'neg_rmse', 'neg_mae'
    Classification : 'accuracy', 'f1', 'roc_auc'

For 'roc_auc' and 'f1' the estimator must have predict_proba.

An unrecognised scoring string raises before any fold runs, listing the names
that are accepted. A fold that fails while scoring with a valid metric warns
and contributes NaN instead, leaving the remaining folds usable: a degenerate
split is a fact about the data, a misspelled metric is not.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# MissKFold
# ---------------------------------------------------------------------------

class MissKFold:
    """
    K-fold cross-validation splitter that does not call sklearn's
    check_array and therefore accepts arrays containing NaN.

    Parameters
    ----------
    n_splits : int, default 5
    shuffle : bool, default False
    random_state : int or None, default None

    See Also
    --------
    MissStratifiedKFold : the same splitter with class-balanced folds.
    miss_cross_val_score : scores an estimator over these folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state=None,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : array-like of shape (n, ...)
        y, groups : ignored (kept for sklearn API compatibility)

        Yields
        ------
        train : ndarray of int
        test  : ndarray of int
        """
        n       = np.asarray(X).shape[0]
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        # Allocate test_mask once and reuse each fold (avoids per-fold allocation)
        test_mask = np.zeros(n, dtype=bool)
        start = 0
        for size in fold_sizes:
            test_mask[:] = False
            test_mask[indices[start: start + size]] = True
            yield indices[~test_mask], indices[test_mask]
            start += size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# MissStratifiedKFold
# ---------------------------------------------------------------------------

class MissStratifiedKFold:
    """
    Stratified K-fold splitter that handles NaN in y.

    Observed (non-NaN) y values are stratified normally.  Rows where y
    is NaN are distributed across folds proportionally (randomly assigned).

    Parameters
    ----------
    n_splits : int, default 5
    shuffle : bool, default True
    random_state : int or None, default None
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state=None,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Yields (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : array-like of shape (n, ...)
        y : array-like of shape (n,)  -- may contain NaN
        groups : ignored
        """
        from sklearn.model_selection import StratifiedKFold as _SKF

        n      = np.asarray(X).shape[0]
        y_arr  = np.asarray(y, dtype=float)

        nan_mask = np.isnan(y_arr)
        obs_idx  = np.where(~nan_mask)[0]
        nan_idx  = np.where(nan_mask)[0]

        if len(obs_idx) < self.n_splits:
            warnings.warn(
                f"MissStratifiedKFold: fewer observed y values ({len(obs_idx)}) "
                f"than n_splits ({self.n_splits}).  Falling back to MissKFold.",
                UserWarning, stacklevel=2,
            )
            yield from MissKFold(
                self.n_splits, self.shuffle, self.random_state
            ).split(X, y)
            return

        # Guard: every class must have at least n_splits members for
        # StratifiedKFold to work.  If any class is too small, fall back.
        y_obs_vals = y_arr[obs_idx]
        classes, counts = np.unique(y_obs_vals[~np.isnan(y_obs_vals)],
                                    return_counts=True)
        if len(classes) == 0 or counts.min() < self.n_splits:
            warnings.warn(
                f"MissStratifiedKFold: at least one class has fewer than "
                f"n_splits={self.n_splits} observed samples.  "
                f"Falling back to MissKFold.",
                UserWarning, stacklevel=2,
            )
            yield from MissKFold(
                self.n_splits, self.shuffle, self.random_state
            ).split(X, y)
            return

        # Stratify on observed y
        skf = _SKF(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        # Assign NaN rows to folds uniformly
        rng = np.random.default_rng(self.random_state)
        nan_fold = rng.integers(0, self.n_splits, size=len(nan_idx))

        splits = list(skf.split(obs_idx, y_arr[obs_idx]))

        for fold, (obs_train_local, obs_test_local) in enumerate(splits):
            train_obs = obs_idx[obs_train_local]
            test_obs  = obs_idx[obs_test_local]

            train_nan = nan_idx[nan_fold != fold]
            test_nan  = nan_idx[nan_fold == fold]

            train_idx = np.sort(np.concatenate([train_obs, train_nan]))
            test_idx  = np.sort(np.concatenate([test_obs,  test_nan]))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Scoring registry
# ---------------------------------------------------------------------------

def _compute_score(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scoring,
    score_groups=None,
) -> float:
    """Compute a single fold score using the requested scoring method."""

    # None or 'auto': use estimator's own score()
    if scoring is None or scoring == 'auto':
        kw = {} if score_groups is None else {'groups': score_groups}
        return float(estimator.score(X_test, y_test, **kw))

    # Callable: scorer(estimator, X_test, y_test)
    if callable(scoring):
        return float(scoring(estimator, X_test, y_test))

    # String shortcuts
    obs   = ~np.isnan(y_test)
    y_obs = y_test[obs]
    X_obs = X_test[obs]

    # If no observed y values in the test fold, no string metric is computable.
    if len(y_obs) == 0:
        return float('nan')

    if scoring == 'r2':
        y_pred = estimator.predict(X_obs)
        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if scoring in ('neg_mse', 'neg_mean_squared_error'):
        y_pred = estimator.predict(X_obs)
        return float(-np.mean((y_obs - y_pred) ** 2))

    if scoring in ('neg_rmse', 'neg_root_mean_squared_error'):
        y_pred = estimator.predict(X_obs)
        return float(-np.sqrt(np.mean((y_obs - y_pred) ** 2)))

    if scoring in ('neg_mae', 'neg_mean_absolute_error'):
        y_pred = estimator.predict(X_obs)
        return float(-np.mean(np.abs(y_obs - y_pred)))

    if scoring == 'accuracy':
        y_pred = estimator.predict(X_obs)
        return float(np.mean(y_obs == y_pred))

    if scoring == 'f1':
        from sklearn.metrics import f1_score
        y_pred = estimator.predict(X_obs)
        return float(f1_score(y_obs, y_pred, average='binary', zero_division=0))

    if scoring == 'roc_auc':
        from sklearn.metrics import roc_auc_score
        proba = estimator.predict_proba(X_obs)
        y_score = proba[:, 1] if proba.ndim == 2 else proba
        return float(roc_auc_score(y_obs, y_score))

    raise ValueError(
        f"Unknown scoring string '{scoring}'.  Supported: "
        f"{', '.join(repr(s) for s in sorted(SCORER_NAMES))}, "
        f"or pass a callable scorer(estimator, X_test, y_test)."
    )


#: Every scoring string ``_compute_score`` understands. Defined next to it so
#: the two cannot drift apart.
SCORER_NAMES = frozenset({
    'r2',
    'neg_mse', 'neg_mean_squared_error',
    'neg_rmse', 'neg_root_mean_squared_error',
    'neg_mae', 'neg_mean_absolute_error',
    'accuracy', 'f1', 'roc_auc',
})


def _validate_scoring(scoring):
    """Reject an unusable scoring specification before any fold runs.

    A fold that fails is warned about and skipped, which is right when one
    split happens to be degenerate and the others are fine. A misspelled
    scorer is not that: it cannot succeed on any fold, so the same warning
    repeats once per split and the caller is handed an array of NaN for what
    is really a typo. Checking once, up front, turns that into one error that
    says what the valid names are.
    """
    if scoring is None or callable(scoring):
        return
    if isinstance(scoring, (list, tuple, dict)):
        names = scoring.values() if isinstance(scoring, dict) else scoring
        for s in names:
            _validate_scoring(s)
        return
    if isinstance(scoring, str):
        if scoring == 'auto' or scoring in SCORER_NAMES:
            return
        raise ValueError(
            "Unknown scoring string %r.  Supported: %s, or pass a callable "
            "scorer(estimator, X_test, y_test)."
            % (scoring, ', '.join(repr(s) for s in sorted(SCORER_NAMES))))
    raise TypeError(
        "scoring must be None, a string, a callable, or a collection of "
        "those; got %s." % type(scoring).__name__)


# ---------------------------------------------------------------------------
# Warm-start state stripping (per-fold independence)
# ---------------------------------------------------------------------------

def _strip_warm_start_state(est) -> None:
    """
    Remove warm-start optimizer state (_theta_opt_) from a per-fold estimator
    copy, including template estimators nested inside wrappers
    (MissMulticlass / MissEnsemble / MissPreprocessor), so no fold's fit is
    initialised from parameters trained on another fold's rows.

    Only instance attributes are touched (__dict__ check), so class
    attributes or properties named _theta_opt_ on third-party estimators
    cannot raise here.
    """
    seen = set()
    stack = [est]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        d = getattr(obj, '__dict__', None)
        if not isinstance(d, dict):
            continue
        d.pop('_theta_opt_', None)
        # Nested templates / fitted members used by MissLearn wrappers
        for attr in ('estimator', '_est', 'model_'):
            stack.append(d.get(attr))
        for item in d.get('estimators', []) or []:
            stack.append(item[1] if isinstance(item, tuple) else item)
        for item in d.get('estimators_', []) or []:
            stack.append(item[1] if isinstance(item, tuple) else item)


# ---------------------------------------------------------------------------
# miss_cross_val_score
# ---------------------------------------------------------------------------

def miss_cross_val_score(
    estimator,
    X,
    y,
    cv: Union[int, Any] = 5,
    scoring=None,
    groups=None,
    stratified: bool = False,
    return_estimators: bool = False,
    random_state=None,
    **fit_params,
) -> np.ndarray:
    """
    Cross-validated score for MissLearn (and NaN-native) estimators.

    Equivalent to sklearn's cross_val_score but does NOT call check_array,
    so arrays containing NaN are passed through to the estimator unchanged.

    Parameters
    ----------
    estimator : fitted or unfitted MissLearn model
    X : array-like or DataFrame of shape (n, p), may contain NaN
    y : array-like or Series of shape (n,), may contain NaN
    cv : int or splitter, default 5
        Number of folds, or a MissKFold / MissStratifiedKFold instance.
    scoring : None, str, or callable, default None
        How each fold is scored::

            None     -- use estimator.score()
            str      -- one of 'r2', 'neg_mse', 'neg_rmse', 'neg_mae',
                        'accuracy', 'f1', 'roc_auc'
            callable -- scorer(estimator, X_test, y_test) -> float
    groups : array-like, optional
        Group labels for MissMixed-based estimators.  The corresponding
        group subset is passed to both fit() and score().
    stratified : bool, default False
        If True, use MissStratifiedKFold instead of MissKFold.
        Ignored if cv is already a splitter object.
    return_estimators : bool, default False
        If True, return (scores, fitted_estimators) instead of scores.
    **fit_params : forwarded to estimator.fit()

    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Score for each fold, in fold order. A fold whose fit or scoring failed
        holds NaN rather than being dropped, so the length is always n_splits
        and position i is always fold i; use ``np.nanmean`` to summarise.
        ``fitted_estimators`` holds None in the same position.
    fitted_estimators : list of estimators (only if return_estimators=True)
    """
    # Coerce to numpy (handle DataFrames)
    from ._pandas_compat import _coerce_X, _coerce_y, _coerce_groups
    X_arr, _ = _coerce_X(X)
    y_arr    = _coerce_y(y)
    # Fail on a bad scorer name here, not once per fold.
    _validate_scoring(scoring)
    groups_arr = _coerce_groups(groups)

    n = X_arr.shape[0]

    # Build splitter
    if isinstance(cv, int):
        if stratified:
            splitter = MissStratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            splitter = MissKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = cv

    scores:     List[float]  = []
    estimators: list         = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_arr, y_arr, groups_arr)
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Build fit kwargs, subsetting groups and any other array fit_params
        # (IndexError covers scalar/0-d params, which have no shape[0])
        kw: Dict[str, Any] = {}
        for k, v in fit_params.items():
            try:
                arr = np.asarray(v)
                if arr.shape[0] == n:
                    kw[k] = arr[train_idx]
                    continue
            except (TypeError, ValueError, IndexError):
                pass
            kw[k] = v

        if groups_arr is not None:
            kw['groups'] = groups_arr[train_idx]

        # Fit.  Each fold must start from an independent state: carrying
        # warm-start parameters across folds (or in from an already-fitted
        # estimator) would leak other folds' test rows into this fold's
        # optimisation starting point.
        est = copy.deepcopy(estimator)
        _strip_warm_start_state(est)
        try:
            est.fit(X_train, y_train, **kw)
        except Exception as exc:
            # NaN in the fold's slot, not a missing slot. Dropping the entry
            # shortened the returned array, which the docstring promises is
            # shape (n_splits,), and shifted every later fold's index, so a
            # caller pairing scores against fold metadata silently misaligned
            # and could not tell which fold had gone. The scoring failure a few
            # lines below already substitutes NaN, and miss_cross_validate
            # does the same for a failed fit; this path was the odd one out.
            warnings.warn(
                f"miss_cross_val_score: fold {fold} fit failed: {exc}.  "
                f"Substituting NaN.",
                UserWarning, stacklevel=2,
            )
            scores.append(float('nan'))
            if return_estimators:
                estimators.append(None)
            continue

        # Score-time groups (for MissMixed predict / score)
        score_groups = groups_arr[test_idx] if groups_arr is not None else None

        try:
            s = _compute_score(est, X_test, y_test, scoring, score_groups)
        except Exception as exc:
            warnings.warn(
                f"miss_cross_val_score: fold {fold} scoring failed: {exc}.  "
                f"Substituting NaN.",
                UserWarning, stacklevel=2,
            )
            s = float('nan')

        scores.append(s)
        if return_estimators:
            estimators.append(est)

    scores_arr = np.array(scores)
    if return_estimators:
        return scores_arr, estimators
    return scores_arr


# ---------------------------------------------------------------------------
# miss_cross_validate
# ---------------------------------------------------------------------------

def miss_cross_validate(
    estimator,
    X,
    y,
    cv: Union[int, Any] = 5,
    scoring=None,
    groups=None,
    stratified: bool = False,
    return_train_score: bool = False,
    return_estimators: bool = False,
    random_state=None,
    **fit_params,
) -> Dict[str, Any]:
    """
    Multi-metric cross-validation for MissLearn estimators.

    Parameters
    ----------
    estimator, X, y, cv, groups, stratified, **fit_params
        As in miss_cross_val_score.
    scoring : None, str, list of str, or dict {name: scorer}, default None
        None        -- {'score': estimator.score()}
        str         -- {'score': named_metric}
        list of str -- {name: named_metric for name in scoring}
        dict        -- {name: scorer_callable_or_string}
    return_train_score : bool, default False
        Also compute scores on the training fold.
    return_estimators : bool, default False
        Include fitted estimators in the returned dict.

    Returns
    -------
    dict with keys:
        'fit_time'         : ndarray (n_splits,)
        'test_<name>'      : ndarray (n_splits,) for each metric
        'train_<name>'     : ndarray (n_splits,) if return_train_score
        'estimators'       : list of fitted models if return_estimators
    """
    import time
    from ._pandas_compat import _coerce_X, _coerce_y, _coerce_groups

    X_arr, _   = _coerce_X(X)
    y_arr      = _coerce_y(y)
    # Fail on a bad scorer name here, not once per fold.
    _validate_scoring(scoring)
    groups_arr = _coerce_groups(groups)
    n          = X_arr.shape[0]

    # Normalise scoring to {name: scorer}
    if scoring is None:
        scorers = {'score': None}
    elif isinstance(scoring, str):
        scorers = {'score': scoring}
    elif isinstance(scoring, (list, tuple)):
        scorers = {s: s for s in scoring}
    elif isinstance(scoring, dict):
        scorers = scoring
    else:
        scorers = {'score': scoring}

    # Build splitter
    if isinstance(cv, int):
        splitter = (MissStratifiedKFold if stratified else MissKFold)(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    else:
        splitter = cv

    results: Dict[str, list] = {
        'fit_time': [],
        **{f'test_{name}': [] for name in scorers},
    }
    if return_train_score:
        for name in scorers:
            results[f'train_{name}'] = []
    if return_estimators:
        results['estimators'] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_arr, y_arr, groups_arr)
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # (IndexError covers scalar/0-d params, which have no shape[0])
        kw: Dict[str, Any] = {}
        for k, v in fit_params.items():
            try:
                arr = np.asarray(v)
                if arr.shape[0] == n:
                    kw[k] = arr[train_idx]
                    continue
            except (TypeError, ValueError, IndexError):
                pass
            kw[k] = v
        if groups_arr is not None:
            kw['groups'] = groups_arr[train_idx]

        # Each fold must start from an independent state: carrying
        # warm-start parameters across folds (or in from an already-fitted
        # estimator) would leak other folds' test rows into this fold's
        # optimisation starting point.
        est = copy.deepcopy(estimator)
        _strip_warm_start_state(est)
        t0  = time.perf_counter()
        try:
            est.fit(X_train, y_train, **kw)
        except Exception as exc:
            warnings.warn(
                f"miss_cross_validate: fold {fold} fit failed: {exc}.",
                UserWarning, stacklevel=2,
            )
            results['fit_time'].append(float('nan'))
            for name in scorers:
                results[f'test_{name}'].append(float('nan'))
                if return_train_score:
                    results[f'train_{name}'].append(float('nan'))
            # Keep 'estimators' aligned with the per-fold score arrays
            if return_estimators:
                results['estimators'].append(None)
            continue
        results['fit_time'].append(time.perf_counter() - t0)

        score_groups_test  = groups_arr[test_idx]  if groups_arr is not None else None
        score_groups_train = groups_arr[train_idx] if groups_arr is not None else None

        for name, scorer in scorers.items():
            try:
                results[f'test_{name}'].append(
                    _compute_score(est, X_test, y_test, scorer, score_groups_test)
                )
            except Exception as exc:
                warnings.warn(f"fold {fold} test scoring '{name}' failed: {exc}",
                              UserWarning, stacklevel=2)
                results[f'test_{name}'].append(float('nan'))

            if return_train_score:
                try:
                    results[f'train_{name}'].append(
                        _compute_score(est, X_train, y_train, scorer,
                                       score_groups_train)
                    )
                except Exception as exc:
                    warnings.warn(f"fold {fold} train scoring '{name}' failed: {exc}",
                                  UserWarning, stacklevel=2)
                    results[f'train_{name}'].append(float('nan'))

        if return_estimators:
            results['estimators'].append(est)

    return {k: np.array(v) if k != 'estimators' else v
            for k, v in results.items()}
