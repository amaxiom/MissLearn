"""
_multiclass.py  --  One-vs-Rest multi-class wrapper for MissLearn classifiers.

MissMulticlass
    Wraps any binary MissLearn classifier (or NaN-native tree classifier)
    and extends it to K > 2 classes via the One-vs-Rest (OvR) strategy.

    For K = 2, the underlying estimator is used directly (no OvR overhead).
    For K > 2, K binary classifiers are fitted: classifier k treats class k
    as positive (1) and all others as negative (0).  NaN in y is preserved
    across all sub-problems so each binary FIML model handles missing
    outcomes correctly.  Predicted probabilities are normalised to sum to 1
    using the standard OvR normalisation.

OvR and FIML
    The joint MVN distribution of X is re-estimated independently for each
    binary sub-problem.  This is slightly less efficient than sharing a
    single estimate across sub-problems, but avoids any coupling between
    sub-model internals and keeps the implementation simple and robust.

Usage
-----
    from MissLearn import MissRidgeClassifier, MissMulticlass
    import numpy as np

    clf = MissMulticlass(MissRidgeClassifier(alpha=1.0))
    clf.fit(X, y)                      # y has values {0, 1, 2}
    proba = clf.predict_proba(X_test)  # shape (n, 3)
    preds = clf.predict(X_test)        # class labels

    # Works transparently with MissEnsemble:
    from MissLearn import MissEnsemble
    ens = MissEnsemble(estimator=MissMulticlass(MissRidgeClassifier()), n_estimators=20)
    ens.fit(X, y)
"""

from __future__ import annotations

import copy
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._base import MissTags


class MissMulticlass(ClassifierMixin, MissTags, BaseEstimator):
    """
    One-vs-Rest multi-class extension for binary MissLearn classifiers.

    Parameters
    ----------
    estimator : binary MissLearn classifier
        Any classifier with fit / predict_proba.  Cloned once per class.
    strategy : str, default 'ovr'
        Currently only 'ovr' (One-vs-Rest) is supported.

    Attributes
    ----------
    classes_        : ndarray of shape (K,) -- sorted unique class labels
    estimators_     : list of K fitted binary classifiers
    n_classes_      : int
    feature_importances_ : ndarray (K-weighted average, if base supports it)
    feature_importances_std_ : ndarray
    """

    #: Read by ``route_multiclass`` so that a router is never routed into
    #: another router. Before this class was a scikit-learn estimator it was
    #: not recognised as a classifier, so the routing function declined on
    #: its first test and the question never came up.
    _is_multiclass_router = True

    def __init__(self, estimator=None, strategy: str = 'ovr'):
        # No validation here. scikit-learn requires a constructor to store
        # its arguments and nothing else: clone() rebuilds an estimator from
        # get_params and must never fail, and a parameter checked at
        # construction cannot be swept by a grid search. strategy is
        # validated in fit, where it would first be used.
        self.estimator = estimator
        self.strategy  = strategy

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        params = {'estimator': self.estimator, 'strategy': self.strategy}
        if deep and hasattr(self.estimator, 'get_params'):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f'estimator__{k}'] = v
        return params

    def set_params(self, **params):
        est_params = {}
        for k, v in params.items():
            if k.startswith('estimator__'):
                est_params[k[len('estimator__'):]] = v
            else:
                setattr(self, k, v)
        if est_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**est_params)
        return self

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit one binary classifier per class (K > 2) or the estimator
        directly (K = 2).

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        y : ndarray of shape (n,), may contain NaN
            Class labels.  Any comparable type (int, float, str).
        **kwargs : forwarded to each sub-estimator's fit()
                   (e.g. groups for MissMixed-based estimators)
        """
        if self.strategy != 'ovr':
            raise ValueError(
                f"MissMulticlass: strategy='{self.strategy}' is not "
                f"supported.  Only 'ovr' (One-vs-Rest) is currently "
                f"available."
            )
        if self.estimator is None:
            # A meta-estimator with no default cannot satisfy the
            # scikit-learn contract, which constructs every estimator with no
            # arguments. MissLogistic is the binary classifier a one-vs-rest
            # wrapper is usually built on, so it is the default, in the same
            # way BaggingClassifier falls back to a decision tree. Imported
            # here rather than at module scope: _logistic imports the routing
            # helper from _conformance, which would close an import loop.
            from ._logistic import MissLogistic
            self._est_ = MissLogistic()
        else:
            self._est_ = self.estimator
        # The scikit-learn contract requires both of these after fit, and
        # nothing set them while this class was a plain object outside the
        # reach of check_estimator. Column names are taken before the
        # conversion to ndarray discards them.
        _names = list(getattr(X, 'columns', [])) or None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        if _names is not None and len(_names) == self.n_features_in_:
            self.feature_names_in_ = np.asarray([str(c) for c in _names],
                                                dtype=object)

        # Unique observed classes
        y_obs = y[~_is_nan(y)]
        self.classes_ = np.unique(y_obs)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"MissMulticlass: found only {self.n_classes_} class(es) in y.  "
                f"At least 2 classes are required."
            )

        # ------------------------------------------------------------------
        # Binary case: delegate directly
        # ------------------------------------------------------------------
        if self.n_classes_ == 2:
            self._binary = True
            self._est    = copy.deepcopy(self._est_)
            y_bin = _to_binary(y, self.classes_[1])   # positive = second class
            self._est.fit(X, y_bin, **kwargs)
            self.estimators_ = [self._est]
        else:
            # ------------------------------------------------------------------
            # Multi-class OvR
            # ------------------------------------------------------------------
            self._binary    = False
            self.estimators_ = []

            for k, cls_label in enumerate(self.classes_):
                y_bin = _to_binary(y, cls_label)     # NaN preserved
                est_k = copy.deepcopy(self._est_)
                try:
                    est_k.fit(X, y_bin, **kwargs)
                    self.estimators_.append(est_k)
                except Exception as exc:
                    warnings.warn(
                        f"MissMulticlass: sub-classifier for class "
                        f"'{cls_label}' failed to fit: {exc}.  "
                        f"This class will receive uniform probabilities.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.estimators_.append(None)

        # Feature importances: weighted average over sub-estimators
        self._aggregate_feature_importances(X.shape[1])
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predicted class probabilities, shape (n, K).

        For K = 2 delegates to the underlying estimator directly.
        For K > 2 uses OvR normalisation::

            p_k = P(y=k | X) from sub-classifier k, then row-normalise.
        """
        # Every estimator in the library reports an unfitted call through
        # check_is_fitted. This class did not, because it was not a scikit-learn
        # estimator and so was never put through check_estimator.
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = self.n_classes_

        if self._binary:
            return self._est.predict_proba(X, **kwargs)

        # Collect P(positive) from each sub-classifier
        raw = np.full((n, K), 1.0 / K)   # fallback for failed classifiers
        for k, est_k in enumerate(self.estimators_):
            if est_k is None:
                continue
            try:
                p2 = est_k.predict_proba(X, **kwargs)   # (n, 2)
                raw[:, k] = p2[:, 1]
            except Exception:
                pass

        # Normalise rows so they sum to 1
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return raw / row_sums

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predicted class labels."""
        # Every estimator in the library reports an unfitted call through
        # check_is_fitted. This class did not, because it was not a scikit-learn
        # estimator and so was never put through check_estimator.
        check_is_fitted(self)
        proba = self.predict_proba(X, **kwargs)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        For K = 2: signed log-odds from underlying estimator.
        For K > 2: matrix of shape (n, K) of per-class log-odds.

        The K > 2 scores are derived from the same sub-classifier
        probabilities that :meth:`predict_proba` uses, and never from a
        sub-classifier's own ``decision_function``, even where one exists.

        Taking the raw score when it was available is what this used to do,
        and it made the two rankings disagree. ``predict`` is the argmax of
        ``predict_proba``, and scikit-learn requires the argmax of
        ``decision_function`` to match it; but a sub-classifier's raw score
        and its calibrated probability are related by a monotone map fitted
        *per sub-classifier*, so a score that is highest for class 2 can
        carry a probability that is not. On MissSupportClassifier, where
        Platt scaling fits separate parameters for every one-vs-rest problem,
        the two disagreed on 3 of 300 rows and ``check_classifiers_train``
        failed. Plain scikit-learn SVC passes the same check, so this was
        ours rather than inherited.

        The logit is strictly increasing, so taking it of each class's
        probability leaves the ranking identical to ``predict_proba`` by
        construction, and row-normalisation there divides a row by a positive
        constant, which cannot reorder it either. The two now agree for
        arithmetic reasons rather than by luck.
        """
        # Every estimator in the library reports an unfitted call through
        # check_is_fitted. This class did not, because it was not a scikit-learn
        # estimator and so was never put through check_estimator.
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if self._binary:
            if hasattr(self._est, 'decision_function'):
                return self._est.decision_function(X, **kwargs)
            # No decision_function on the base estimator: derive the signed
            # log-odds of the positive class (classes_[1]) from predict_proba,
            # keeping the documented 1-D binary contract.
            p = self._est.predict_proba(X, **kwargs)[:, 1].clip(1e-10, 1 - 1e-10)
            return np.log(p / (1 - p))

        n, K = X.shape[0], self.n_classes_
        # 1/K matches the fallback predict_proba uses for a sub-classifier
        # that could not be fitted, so an absent class scores the same in
        # both places.
        fallback = np.log((1.0 / K) / (1.0 - 1.0 / K)) if K > 1 else 0.0
        scores = np.full((n, K), fallback)
        for k, est_k in enumerate(self.estimators_):
            if est_k is None:
                continue
            try:
                p = est_k.predict_proba(X, **kwargs)[:, 1]
            except Exception:
                continue
            p = np.clip(p, 1e-10, 1 - 1e-10)
            scores[:, k] = np.log(p / (1 - p))
        return scores

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Accuracy, excluding rows where y is NaN."""
        X   = np.asarray(X, dtype=float)
        y   = np.asarray(y)
        mask = ~_is_nan(y)
        if mask.sum() == 0:
            return float('nan')
        y_pred = self.predict(X, **kwargs)
        return float(np.mean(y[mask] == y_pred[mask]))

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def _aggregate_feature_importances(self, p: int) -> None:
        fi_list = []
        for est_k in self.estimators_:
            if est_k is not None and hasattr(est_k, 'feature_importances_'):
                fi_list.append(est_k.feature_importances_)
        if not fi_list:
            return
        fi_arr = np.array(fi_list)             # (n_valid, p)
        self.feature_importances_     = fi_arr.mean(axis=0)
        self.feature_importances_std_ = fi_arr.std(axis=0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        W = 66
        print()
        print('=' * W)
        print(f"{'MissMulticlass  --  One-vs-Rest Extension':^{W}}")
        print('=' * W)
        print(f"  Classes  : {self.classes_.tolist()}")
        print(f"  n_classes: {self.n_classes_}")
        print(f"  Strategy : {self.strategy}")
        # the resolved sub-estimator, so the report does not read
        # NoneType when the default was used
        _base = getattr(self, "_est_", None) or self.estimator
        print(f"  Base est : {type(_base).__name__}")
        if self.n_classes_ > 2:
            print()
            print(f"  Sub-classifiers ({self.n_classes_} binary OvR models):")
            for k, (cls_label, est_k) in enumerate(
                zip(self.classes_, self.estimators_)
            ):
                status = 'fitted' if est_k is not None else 'FAILED'
                print(f"    [{k}] class '{cls_label}' vs. rest  [{status}]")
        if hasattr(self, 'feature_importances_'):
            print()
            fi  = self.feature_importances_
            std = self.feature_importances_std_
            order = np.argsort(fi)[::-1]
            bar_w = 24
            print("  Feature Importances (avg across OvR classifiers):")
            for j in order:
                pct    = fi[j] * 100
                filled = int(round(pct / 100 * bar_w))
                bar    = '#' * filled + ' ' * (bar_w - filled)
                print(f"    X{j:<4} [{bar}] {pct:5.1f}%  +/-{std[j]*100:4.1f}%")
        print('=' * W)
        print()

    def __repr__(self) -> str:
        return (f"MissMulticlass(estimator={self.estimator!r}, "
                f"strategy='{self.strategy}')")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nan(y: np.ndarray) -> np.ndarray:
    """Return boolean mask of NaN positions.

    Handles numeric NaN, Python None, and pandas NA/NaT values safely.
    The per-element loop guards against pd.NA whose comparison raises TypeError.
    """
    try:
        return np.isnan(y.astype(float))
    except (ValueError, TypeError):
        # Object array: may contain None, pd.NA, or strings
        mask = np.zeros(len(y), dtype=bool)
        for i, v in enumerate(y):
            try:
                mask[i] = v is None or bool(v != v)   # v != v is True only for float NaN
            except (TypeError, ValueError):
                mask[i] = True   # pd.NA and similar
        return mask


def _to_binary(y: np.ndarray, positive_class) -> np.ndarray:
    """
    Convert y to a binary array: 1.0 if y == positive_class, 0.0 otherwise.
    NaN values in y remain NaN.

    The comparison is made only on the observed entries. Comparing the whole
    array at once fails for a pandas nullable dtype: ``y == positive_class``
    yields a BooleanArray containing ``pd.NA``, and ``np.where`` then has to
    take the truth value of ``pd.NA``, which raises
    ``TypeError: boolean value of NA is ambiguous``. :func:`_is_nan` above was
    written to handle exactly that value and this function then handed it
    straight to numpy, so a ``string``-dtype Series with a missing label
    reached the multiclass fit and died two frames later.
    """
    nan_mask = _is_nan(y)
    y_arr    = np.asarray(y, dtype=object)
    y_bin    = np.zeros(y_arr.shape[0], dtype=float)
    observed = ~nan_mask
    if observed.any():
        y_bin[observed] = (y_arr[observed] == positive_class).astype(float)
    y_bin[nan_mask] = np.nan
    return y_bin
