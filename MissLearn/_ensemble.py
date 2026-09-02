"""
_ensemble.py  --  Bootstrap-aggregated ensemble for MissLearn and NaN-native
                  tree models.

MissEnsemble
    Wraps one or more MissLearn / NaN-native tree estimators in a bootstrap-
    aggregated ensemble.  Supports:

    * Homogeneous bagging  -- one estimator type cloned N times
    * Heterogeneous weighted ensemble -- explicit named list with per-estimator
      weights

    NaN-native non-MissLearn estimators accepted:
        sklearn  : HistGradientBoosting{Classifier,Regressor}  (any version)
                   DecisionTree*, RandomForest*, ExtraTrees*    (sklearn >= 1.3)

        XGBoost  : XGBClassifier, XGBRegressor
        LightGBM : LGBMClassifier, LGBMRegressor
        CatBoost : CatBoostClassifier, CatBoostRegressor

    All other estimators are rejected with a clear error message.

Philosophy
    Bagging is philosophically orthogonal to missing-data handling.  Each base
    estimator handles NaN in its own way (FIML for MissLearn models, native
    split logic for tree models).  The ensemble layer sees no missing data -- it
    only aggregates fitted predictions.

    When 'groups' is supplied (for MissMixed base estimators), bootstrap
    sampling is performed at the group level (cluster bootstrap): subjects are
    drawn with replacement and duplicate draws are relabelled so the LME treats
    them as distinct subjects.
"""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict

import numpy as np
from ._sklearn_compat import is_classifier_safe, is_regressor_safe

from ._base import MissBase
from sklearn.utils import check_random_state

# ---------------------------------------------------------------------------
# Allow-list constants
# ---------------------------------------------------------------------------

# sklearn CART-based trees that gained NaN support in 1.3
_SKLEARN_CART = frozenset({
    'DecisionTreeClassifier',  'DecisionTreeRegressor',
    'RandomForestClassifier',  'RandomForestRegressor',
    'ExtraTreesClassifier',    'ExtraTreesRegressor',
})

# sklearn histogram-based boosting -- NaN-native from introduction
_SKLEARN_HIST = frozenset({
    'HistGradientBoostingClassifier',
    'HistGradientBoostingRegressor',
})

_XGB_NAMES      = frozenset({'XGBClassifier',      'XGBRegressor'})
_LGBM_NAMES     = frozenset({'LGBMClassifier',     'LGBMRegressor'})
_CATBOOST_NAMES = frozenset({'CatBoostClassifier', 'CatBoostRegressor'})

_SKLEARN_NAN_NATIVE = _SKLEARN_CART | _SKLEARN_HIST
_ALL_ALLOWED        = _SKLEARN_NAN_NATIVE | _XGB_NAMES | _LGBM_NAMES | _CATBOOST_NAMES


# ---------------------------------------------------------------------------
# Estimator compatibility helpers
# ---------------------------------------------------------------------------

def _is_misslearn(est) -> bool:
    """Return True if *est* is a MissLearn model.

    Asks whether it inherits from MissBase rather than where its class was
    declared. The module test refused a user subclass of MissLogistic with a
    message saying it was not NaN-native, which was both wrong and unhelpful:
    a subclass of a MissLearn estimator handles NaN exactly as well as its
    parent. The module check is kept as a fallback so that anything inside
    the package that does not inherit from MissBase still counts.
    """
    if isinstance(est, MissBase):
        return True
    return type(est).__module__.startswith('MissLearn')


def _check_estimator_compatible(est, label: str = 'estimator') -> None:
    """Raise a descriptive error if *est* cannot handle NaN inputs."""
    name = type(est).__name__

    if _is_misslearn(est):
        return                                  # always OK

    if name in _SKLEARN_CART:
        import sklearn
        try:
            from packaging.version import Version as V
            if V(sklearn.__version__) < V('1.3'):
                raise ValueError(
                    f"{label} ({name}) requires scikit-learn >= 1.3 for native "
                    f"NaN support (installed: {sklearn.__version__}).  Use "
                    f"HistGradientBoosting* or upgrade scikit-learn."
                )
        except ImportError:
            pass                                # packaging not available; skip check
        return

    if name in _SKLEARN_HIST:
        return

    if name in _XGB_NAMES:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{label} ({name}) requires the 'xgboost' package "
                f"(pip install xgboost)."
            )
        return

    if name in _LGBM_NAMES:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{label} ({name}) requires the 'lightgbm' package "
                f"(pip install lightgbm)."
            )
        return

    if name in _CATBOOST_NAMES:
        try:
            import catboost  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{label} ({name}) requires the 'catboost' package "
                f"(pip install catboost)."
            )
        return

    # Not in any allowed set
    allowed = (
        "any MissLearn model; or the following NaN-native tree models: "
        + ", ".join(sorted(_SKLEARN_NAN_NATIVE))
        + " (sklearn >= 1.3 for CART variants); "
        + "XGBClassifier/XGBRegressor; "
        + "LGBMClassifier/LGBMRegressor; "
        + "CatBoostClassifier/CatBoostRegressor."
    )
    raise ValueError(
        f"{label} ({name}) is not supported by MissEnsemble.  "
        f"Only estimators that handle NaN inputs natively are accepted.  "
        f"Supported estimators: {allowed}"
    )


def _clone_with_seed(est, seed: int):
    """Deep-copy *est* and set its random_state to *seed* where possible."""
    est2 = copy.deepcopy(est)
    try:
        est2.set_params(random_state=seed)
    except (ValueError, TypeError):
        pass
    return est2


def _has_predict_proba(est) -> bool:
    return hasattr(est, 'predict_proba')


def _filter_kwargs(est, kwargs: dict) -> dict:
    """
    Strip 'groups' from kwargs for non-MissLearn estimators, which do not
    accept that argument.
    """
    if not kwargs or _is_misslearn(est):
        return kwargs
    return {k: v for k, v in kwargs.items() if k != 'groups'}


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def _detect_task(y: np.ndarray) -> str:
    """Return 'regression' or 'classification' inferred from *y*.

    Classification is inferred when y contains only integer-valued
    observations (no fractional part) and at most 20 unique values.
    This covers binary {0,1} and multi-class {0,1,2,...,K-1}.
    """
    y_obs = y[~np.isnan(y)]
    if y_obs.size == 0:
        raise ValueError(
            "MissEnsemble: y contains no observed (non-NaN) values."
        )
    unique = np.unique(y_obs)
    # All values are whole numbers and cardinality is small enough
    if np.all(y_obs == np.floor(y_obs)) and len(unique) <= 20:
        return 'classification'
    return 'regression'


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _row_bootstrap(n: int, rng, max_samples: float):
    """
    Standard row bootstrap.

    Returns
    -------
    boot_idx : 1-D int array of sampled row indices (with replacement)
    oob_idx  : 1-D int array of rows not sampled
    """
    k = max(1, int(n * max_samples))
    boot = rng.integers(0, n, size=k)
    oob  = np.setdiff1d(np.arange(n), boot)
    return boot.astype(np.intp), oob.astype(np.intp)


def _cluster_bootstrap(groups: np.ndarray, rng, max_samples: float):
    """
    Cluster (group-level) bootstrap.

    Groups are sampled with replacement.  A group drawn k > 1 times appears
    under k distinct labels (original + '__b1', '__b2', ...) so that MissMixed
    treats them as separate subjects.

    Returns
    -------
    row_idx    : 1-D int array of row indices into the original data
    new_groups : 1-D array of (relabelled) group labels
    oob_idx    : 1-D int array of rows belonging to groups not drawn at all
    """
    unique_g = np.unique(groups)
    k        = max(1, int(len(unique_g) * max_samples))
    sampled  = rng.choice(unique_g, size=k, replace=True)

    rows: list   = []
    labels: list = []
    counts: dict = defaultdict(int)

    for g in sampled:
        idx = np.where(groups == g)[0]
        c   = counts[g]
        counts[g] += 1
        label = g if c == 0 else f"{g}__b{c}"
        rows.append(idx)
        labels.extend([label] * len(idx))

    row_idx    = np.concatenate(rows).astype(np.intp)
    new_groups = np.array(labels)

    # OOB: groups never selected
    selected   = np.array(list(counts.keys()))
    oob_g      = unique_g[~np.isin(unique_g, selected)]
    oob_idx    = (np.where(np.isin(groups, oob_g))[0].astype(np.intp)
                  if len(oob_g) else np.array([], dtype=np.intp))
    return row_idx, new_groups, oob_idx


# ---------------------------------------------------------------------------
# Module-level fit helper (required for joblib pickling)
# ---------------------------------------------------------------------------

def _fit_one(est, X_boot: np.ndarray, y_boot: np.ndarray, fit_kw: dict):
    est.fit(X_boot, y_boot, **fit_kw)
    return est


def _subset_kwargs(kwargs: dict, idx: np.ndarray, n: int) -> dict:
    """
    Subset any (n,)-length array-valued fit kwarg (e.g. sample_weight) to
    the given row indices so it stays aligned with a bootstrap resample.
    Non-array and wrong-length values pass through unchanged.
    """
    out = {}
    for k, v in kwargs.items():
        try:
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] == n:
                out[k] = arr[idx]
                continue
        except (TypeError, ValueError):
            pass
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# MissEnsemble
# ---------------------------------------------------------------------------

class MissEnsemble:
    """
    Bootstrap-aggregated ensemble for MissLearn and NaN-native tree models.

    Two usage modes:

    **Homogeneous bagging** -- pass a single ``estimator`` with
    ``n_estimators`` bootstrap copies::

        MissEnsemble(estimator=MissRidgeRegressor(), n_estimators=100)

    **Heterogeneous weighted ensemble** -- pass an explicit named list with
    optional per-estimator weights::

        MissEnsemble(
            estimators=[
                ('ridge', MissRidgeRegressor()),
                ('hgb',   HistGradientBoostingRegressor()),
                ('dt',    DecisionTreeRegressor()),
            ],
            weights=[2, 1, 1],
        )

    All base estimators must either be MissLearn models or NaN-native tree
    models (see module docstring for the full allow-list).  Passing any other
    estimator raises a ``ValueError`` with the list of supported options.

    Parameters
    ----------
    estimator : unfitted estimator, optional
        Base estimator for homogeneous bagging.  Mutually exclusive with
        ``estimators``.
    estimators : list of (str, estimator) tuples, optional
        Named estimators for a heterogeneous ensemble.  Mutually exclusive
        with ``estimator``.
    n_estimators : int, default 100
        Number of bootstrap copies.  Only used with ``estimator``.
    weights : array-like of float, optional
        Per-estimator weights for prediction averaging.  Need not sum to one
        (normalised internally).  Defaults to uniform weights.
    bootstrap : bool, default True
        If True each estimator is fit on a bootstrap sample of the training
        data.  If False all estimators are fit on the full dataset (simple
        weighted average).
    max_samples : float, default 1.0
        Fraction of training rows (or groups for cluster bootstrap) to draw
        per estimator.  Must be in (0, 1].
    max_features : float, default 1.0
        Fraction of features to use per estimator.  Must be in (0, 1].
        Values < 1.0 enable the random-subspace method.
    oob_score : bool, default False
        Compute an out-of-bag score for each estimator.  Only meaningful
        when ``bootstrap=True``.
    n_jobs : int, default 1
        Number of parallel fitting jobs.  -1 uses all available CPUs.
    random_state : int or None, default None

    Attributes
    ----------
    estimators_ : list of (name, fitted_estimator, feat_idx) tuples
    weights_ : ndarray of normalised per-estimator weights
    task_ : 'regression' or 'classification'
    classes_ : ndarray  (classification only)
    feature_importances_ : ndarray  (if any base estimator exposes it)
    feature_importances_std_ : ndarray  (weighted std across estimators)
    oob_scores_ : dict  name -> float  (if oob_score=True)
    n_estimators_ : int
    n_features_in_ : int
    """

    def __init__(
        self,
        estimator=None,
        estimators=None,
        n_estimators: int = 100,
        weights=None,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state=None,
    ):
        self.estimator    = estimator
        self.estimators   = estimators
        self.n_estimators = n_estimators
        self.weights      = weights
        self.bootstrap    = bootstrap
        self.max_samples  = max_samples
        self.max_features = max_features
        self.oob_score    = oob_score
        self.n_jobs       = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        return {
            'estimator':    self.estimator,
            'estimators':   self.estimators,
            'n_estimators': self.n_estimators,
            'weights':      self.weights,
            'bootstrap':    self.bootstrap,
            'max_samples':  self.max_samples,
            'max_features': self.max_features,
            'oob_score':    self.oob_score,
            'n_jobs':       self.n_jobs,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_estimator_list(self):
        """Return list of (name, unfitted_estimator) from init parameters."""
        if self.estimator is not None and self.estimators is not None:
            raise ValueError(
                "Provide either 'estimator' (homogeneous bagging) or "
                "'estimators' (heterogeneous ensemble), not both."
            )
        if self.estimator is not None:
            return [
                (f"est_{i}", copy.deepcopy(self.estimator))
                for i in range(self.n_estimators)
            ]
        if self.estimators is not None:
            return [
                (name, copy.deepcopy(est))
                for name, est in self.estimators
            ]
        raise ValueError(
            "Provide 'estimator' (homogeneous) or 'estimators' (heterogeneous)."
        )

    def _norm_weights(self, n: int) -> np.ndarray:
        if self.weights is None:
            return np.ones(n) / n
        w = np.asarray(self.weights, dtype=float)
        if len(w) != n:
            raise ValueError(
                f"len(weights)={len(w)} does not match the number of "
                f"estimators={n}."
            )
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative.")
        total = w.sum()
        if total == 0:
            raise ValueError("weights must not all be zero.")
        return w / total

    def _feature_mask(self, p: int, rng) -> np.ndarray:
        n_feat = max(1, int(p * self.max_features))
        return np.sort(rng.choice(p, size=n_feat, replace=False))

    def _validate_task_consistency(self, named_ests, task: str) -> None:
        """Warn if a base estimator looks inconsistent with the detected task."""
        for i, (name, est) in enumerate(named_ests):
            if not _is_misslearn(est):
                continue
            est_name = type(est).__name__
            # Ask scikit-learn what the estimator is, rather than reading the
            # class name for a suffix. The name test passed on
            # MissRidgeRegressor and failed on MissLinear, which is equally a
            # regressor and simply does not carry the word, so the two
            # estimators most likely to be reached for were the two the guard
            # could not see. The failure was then deferred from fit to
            # predict_proba, which raised AttributeError from inside
            # _collect_proba instead of a diagnosable error at the door.
            is_reg_model = is_regressor_safe(est)
            is_clf_model = is_classifier_safe(est)
            if task == 'classification' and is_reg_model:
                raise ValueError(
                    f"estimators[{i}] ({name}): y was detected as "
                    f"classification but {est_name} is a regressor.  "
                    f"Use the corresponding Classifier variant."
                )
            if task == 'regression' and is_clf_model:
                raise ValueError(
                    f"estimators[{i}] ({name}): y was detected as "
                    f"regression but {est_name} is a classifier.  "
                    f"Use the corresponding Regressor variant."
                )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit the ensemble.

        Parameters
        ----------
        X : array-like of shape (n, p), may contain NaN
        y : array-like of shape (n,), may contain NaN
        groups : array-like of shape (n,), optional keyword argument
            Subject / group labels for MissMixed base estimators.  When
            supplied, bootstrap sampling is performed at the group level
            (cluster bootstrap) so group structure is preserved.

        Returns
        -------
        self
        """
        # Out-of-bag rows are the rows a bootstrap draw left out, so without
        # bootstrapping there are none and oob_scores_ came back an empty
        # dict with nothing said. Asking for a score and silently getting no
        # score is the failure worth refusing; sklearn refuses it too.
        if self.oob_score and not self.bootstrap:
            raise ValueError(
                "MissEnsemble: oob_score=True requires bootstrap=True. "
                "Out-of-bag scoring is measured on the rows each bootstrap "
                "draw left out, and with bootstrap=False every estimator "
                "sees every row, so there is nothing to score against.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        rng  = check_random_state(self.random_state)

        groups     = kwargs.get('groups', None)
        groups_arr = np.asarray(groups) if groups is not None else None

        # --- resolve and validate estimators ---
        named_ests = self._resolve_estimator_list()
        n_est      = len(named_ests)

        for i, (name, est) in enumerate(named_ests):
            _check_estimator_compatible(est, label=f"estimators[{i}] ({name})")

        # --- detect task and validate consistency ---
        self.task_ = _detect_task(y)
        self._validate_task_consistency(named_ests, self.task_)

        # --- normalised weights ---
        self.weights_ = self._norm_weights(n_est)

        # --- build setup list (all random choices made here for reproducibility) ---
        setup_list = []   # (name, cloned_est, X_boot, y_boot, fit_kw, oob_rows, feat_idx)

        for i, (name, est) in enumerate(named_ests):
            # Each estimator gets a deterministic seed derived from the master rng
            seed    = int(rng.randint(0, 2**31))
            est2    = _clone_with_seed(est, seed)
            est_rng = np.random.default_rng(seed)

            # Feature subsampling
            feat_idx = (self._feature_mask(p, est_rng)
                        if self.max_features < 1.0 else np.arange(p))
            X_sub = X[:, feat_idx]

            if self.bootstrap:
                if groups_arr is not None:
                    boot_rows, boot_groups, oob_rows = _cluster_bootstrap(
                        groups_arr, est_rng, self.max_samples
                    )
                    X_boot = X_sub[boot_rows]
                    y_boot = y[boot_rows]
                    fit_kw = _subset_kwargs(
                        {k: v for k, v in kwargs.items() if k != 'groups'},
                        boot_rows, n,
                    )
                    fit_kw['groups'] = boot_groups
                else:
                    boot_rows, oob_rows = _row_bootstrap(n, est_rng, self.max_samples)
                    X_boot = X_sub[boot_rows]
                    y_boot = y[boot_rows]
                    fit_kw = _subset_kwargs(dict(kwargs), boot_rows, n)
            else:
                X_boot   = X_sub
                y_boot   = y
                boot_rows = np.arange(n)
                oob_rows  = np.array([], dtype=np.intp)
                fit_kw    = dict(kwargs)

            # Non-MissLearn estimators do not accept the 'groups' kwarg.
            fit_kw = _filter_kwargs(est2, fit_kw)

            # sklearn/tree models handle NaN in X but not in y.
            # Strip rows where y is NaN from bootstrap sample for non-MissLearn estimators.
            if not _is_misslearn(est2):
                valid = ~np.isnan(y_boot)
                if not np.all(valid):
                    X_boot = X_boot[valid]
                    y_boot = y_boot[valid]
                    if 'groups' in fit_kw:
                        fit_kw = dict(fit_kw)
                        fit_kw['groups'] = np.asarray(fit_kw['groups'])[valid]

            setup_list.append((name, est2, X_boot, y_boot, fit_kw, oob_rows, feat_idx))

        # --- fit (parallel or sequential) ---
        if self.n_jobs == 1:
            fitted = [
                _fit_one(est, Xb, yb, fkw)
                for _, est, Xb, yb, fkw, _, _ in setup_list
            ]
        else:
            from joblib import Parallel, delayed
            fitted = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one)(est, Xb, yb, fkw)
                for _, est, Xb, yb, fkw, _, _ in setup_list
            )

        # --- store fitted estimators ---
        self.estimators_  = []
        self.oob_indices_ = []
        self.oob_scores_  = {}

        for (name, _, _, _, _, oob_rows, feat_idx), est_fit in zip(setup_list, fitted):
            self.estimators_.append((name, est_fit, feat_idx))
            self.oob_indices_.append(oob_rows)

            if self.oob_score and len(oob_rows) > 0:
                X_oob = X[:, feat_idx][oob_rows]
                y_oob = y[oob_rows]
                oob_kw: dict = {}
                if groups_arr is not None:
                    oob_kw['groups'] = groups_arr[oob_rows]
                # For non-MissLearn estimators, strip NaN-y from OOB set
                if not _is_misslearn(est_fit):
                    valid_oob = ~np.isnan(y_oob)
                    X_oob = X_oob[valid_oob]
                    y_oob = y_oob[valid_oob]
                    if 'groups' in oob_kw:
                        oob_kw['groups'] = np.asarray(oob_kw['groups'])[valid_oob]
                try:
                    oob_kw_f = _filter_kwargs(est_fit, oob_kw)
                    self.oob_scores_[name] = float(
                        est_fit.score(X_oob, y_oob, **oob_kw_f)
                    )
                except Exception:
                    self.oob_scores_[name] = float('nan')

        # --- classes (classification) ---
        if self.task_ == 'classification':
            # Derive from y so every observed class is represented even when
            # a bootstrap member never saw one (rare classes can be absent
            # from a resample).  Member probabilities are aligned to these
            # classes in _collect_proba.
            y_obs = y[~np.isnan(y)]
            self.classes_ = np.unique(y_obs)

        # --- feature importances ---
        self._aggregate_feature_importances(p)

        self.n_features_in_ = p
        self.n_estimators_  = n_est
        self._homogeneous   = self.estimator is not None   # for summary display
        return self

    # ------------------------------------------------------------------
    # Feature importance aggregation
    # ------------------------------------------------------------------

    def _aggregate_feature_importances(self, p: int) -> None:
        fi_sum = np.zeros(p)
        w_sum  = 0.0
        for (_, est, feat_idx), w in zip(self.estimators_, self.weights_):
            if hasattr(est, 'feature_importances_'):
                fi = np.zeros(p)
                fi[feat_idx] = est.feature_importances_
                fi_sum += w * fi
                w_sum  += w
        if w_sum > 0:
            self.feature_importances_ = fi_sum / w_sum
            # weighted variance -> std
            fi_sq = np.zeros(p)
            for (_, est, feat_idx), w in zip(self.estimators_, self.weights_):
                if hasattr(est, 'feature_importances_'):
                    fi = np.zeros(p)
                    fi[feat_idx] = est.feature_importances_
                    fi_sq += w * (fi - self.feature_importances_) ** 2
            self.feature_importances_std_ = np.sqrt(fi_sq / w_sum)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _collect(self, X: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """
        Collect raw outputs from all estimators.

        Returns ndarray of shape (n_est, n) for scalar outputs (predict,
        decision_function) or (n_est, n, k) for array outputs (predict_proba).
        """
        results = []
        for _, est, feat_idx in self.estimators_:
            X_sub = X[:, feat_idx]
            kw    = _filter_kwargs(est, kwargs)
            results.append(getattr(est, method)(X_sub, **kw))
        return np.array(results)

    def _collect_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Collect predict_proba outputs aligned to self.classes_.

        A member fitted on a bootstrap resample may have seen only a subset
        of the classes; its probability columns are mapped to the ensemble's
        class order (unseen classes get probability 0).
        """
        n = X.shape[0]
        classes = np.asarray(self.classes_, dtype=float)
        K = len(classes)
        results = []
        for _, est, feat_idx in self.estimators_:
            X_sub = X[:, feat_idx]
            kw    = _filter_kwargs(est, kwargs)
            p_est = np.asarray(est.predict_proba(X_sub, **kw), dtype=float)
            est_classes = np.asarray(
                getattr(est, 'classes_', self.classes_), dtype=float
            )
            if p_est.shape[1] == K and np.array_equal(est_classes, classes):
                results.append(p_est)
                continue
            aligned = np.zeros((n, K))
            for j, c in enumerate(est_classes):
                hit = np.where(classes == c)[0]
                if hit.size:
                    aligned[:, hit[0]] = p_est[:, j]
            results.append(aligned)
        return np.array(results)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict targets.

        Regression  : weighted mean of base estimator predictions.
        Classification : argmax of weighted mean predicted probabilities.
        """
        X = np.asarray(X, dtype=float)
        if self.task_ == 'regression':
            preds = self._collect(X, 'predict', **kwargs)   # (n_est, n)
            return np.average(preds, axis=0, weights=self.weights_)
        else:
            proba = self._collect_proba(X, **kwargs)   # (n_est, n, k)
            avg   = np.average(proba, axis=0, weights=self.weights_)
            return self.classes_[np.argmax(avg, axis=1)]

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Weighted mean predicted class probabilities.  Classification only.
        """
        if self.task_ != 'classification':
            raise AttributeError(
                "predict_proba is only available when task_ == 'classification'."
            )
        X = np.asarray(X, dtype=float)
        proba = self._collect_proba(X, **kwargs)   # (n_est, n, k)
        return np.average(proba, axis=0, weights=self.weights_)

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.05, **kwargs
    ):
        """
        Empirical prediction interval from the spread of ensemble predictions.

        Regression only.  Returns the alpha/2 and 1-alpha/2 empirical
        quantiles of the B base estimator predictions.

        Note: this interval captures epistemic (model) uncertainty.  For
        homogeneous bagging it approximates a bootstrap confidence interval.
        For heterogeneous ensembles the spread reflects model disagreement
        rather than aleatoric noise.

        Returns
        -------
        lo, hi : ndarrays of shape (n,)
        """
        if self.task_ != 'regression':
            raise AttributeError(
                "predict_interval is only available when task_ == 'regression'."
            )
        X     = np.asarray(X, dtype=float)
        preds = self._collect(X, 'predict', **kwargs)   # (n_est, n)
        lo    = np.quantile(preds, alpha / 2,       axis=0)
        hi    = np.quantile(preds, 1.0 - alpha / 2, axis=0)
        return lo, hi

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        R² for regression, accuracy for classification.
        Rows where y is NaN are excluded.
        """
        X      = np.asarray(X, dtype=float)
        y      = np.asarray(y, dtype=float)
        mask   = ~np.isnan(y)
        y_pred = self.predict(X, **kwargs)
        y_o    = y[mask]
        p_o    = y_pred[mask]
        if self.task_ == 'regression':
            ss_res = float(np.sum((y_o - p_o) ** 2))
            ss_tot = float(np.sum((y_o - y_o.mean()) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            return float(np.mean(y_o == p_o))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, feature_names=None) -> None:
        """Print a fitted-ensemble report.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for the columns of X, used to label the feature-importance
            block. Resolved in priority order: this argument, then a
            ``feature_names_in_`` recorded at fit time (which the pandas
            compatibility layer sets when a DataFrame is passed), and only
            then generic ``X0..Xp``. Without it an importance table names
            columns as ``X17``, which is unreadable against a real data
            dictionary.
        """
        W = 76
        print()
        print('=' * W)
        print(f"{'MissEnsemble':^{W}}")
        print('=' * W)
        print(f"  Task            : {self.task_}")
        print(f"  Estimators      : {self.n_estimators_}")
        print(f"  Bootstrap       : {self.bootstrap}"
              + (f"  (max_samples={self.max_samples})" if self.bootstrap else ""))
        if self.max_features < 1.0:
            print(f"  max_features    : {self.max_features}")
        print()

        has_oob = self.oob_score and bool(self.oob_scores_)

        # --- estimator table ---
        if self._homogeneous and self.n_estimators_ > 6:
            # Compact display for large homogeneous ensembles
            _, est0, _ = self.estimators_[0]
            type_name  = type(est0).__name__
            w_each     = self.weights_[0]
            print(f"  Homogeneous bagging: {self.n_estimators_} x {type_name}")
            print(f"  Weight per estimator: {w_each:.4f}")
            if has_oob:
                oob_vals = [v for v in self.oob_scores_.values()
                            if np.isfinite(v)]
                if oob_vals:
                    print(f"  OOB score: mean={np.mean(oob_vals):.4f}  "
                          f"std={np.std(oob_vals):.4f}  "
                          f"min={np.min(oob_vals):.4f}  "
                          f"max={np.max(oob_vals):.4f}")
        else:
            # Full row per estimator
            hdr = f"  {'#':<4} {'Name':<22} {'Type':<30} {'Weight':>7}"
            if has_oob:
                hdr += f"  {'OOB score':>10}"
            print(hdr)
            print('  ' + '-' * (W - 2))
            for i, ((name, est, _), w) in enumerate(
                zip(self.estimators_, self.weights_)
            ):
                line = (f"  {i:<4} {name:<22} {type(est).__name__:<30} {w:>7.3f}")
                if has_oob:
                    sc = self.oob_scores_.get(name, float('nan'))
                    line += f"  {sc:>10.4f}"
                print(line)
            print('  ' + '-' * (W - 2))

        # --- feature importances ---
        if hasattr(self, 'feature_importances_'):
            print()
            fi    = self.feature_importances_
            std   = getattr(self, 'feature_importances_std_', np.zeros_like(fi))
            order = np.argsort(fi)[::-1]
            bar_w = 24
            names = feature_names
            if names is None:
                names = getattr(self, 'feature_names_in_', None)
            if names is None or len(names) < len(fi):
                names = [f"X{j}" for j in range(len(fi))]
            NW = max(6, min(28, max(len(str(names[j])) for j in order)))

            print("  Feature Importances (weighted average across estimators):")
            for j in order:
                pct    = fi[j] * 100
                filled = int(round(pct / 100 * bar_w))
                bar    = '#' * filled + ' ' * (bar_w - filled)
                print(f"    {str(names[j]):<{NW}} [{bar}] "
                      f"{pct:5.1f}%  +/-{std[j]*100:4.1f}%")

        # --- note on interval semantics ---
        if self.task_ == 'regression':
            print()
            if self._homogeneous:
                print("  predict_interval: empirical bootstrap CI (epistemic + "
                      "sampling uncertainty).")
            else:
                print("  predict_interval: empirical spread of model predictions "
                      "(epistemic / model-disagreement only; does not include "
                      "aleatoric noise).")

        print('=' * W)
        print()
