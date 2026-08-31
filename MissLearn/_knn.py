"""
MissLearn._knn
--------------
K-Nearest Neighbours models with native missing-data support.

Distance is computed on the subset of features that are observed in BOTH
the query row and each candidate training row (available-case distance),
scaled by sqrt(p / n_shared) to compensate for the reduced dimensionality.
Under Missing-At-Random, E[d_full^2] = (p / n_shared) * E[d_partial^2], so
the scaled partial distance is an unbiased estimator of the full distance.
No imputation is performed at any stage.

Distance metrics
----------------
'euclidean' (default)
    Standardized Euclidean on shared features.  Each feature is scaled by
    its training standard deviation so all dimensions contribute equally.

'mahalanobis'
    Accounts for inter-feature correlations via the correlation matrix
    estimated from complete training cases.  Reduces to standardized
    Euclidean when features are uncorrelated.

Weights
-------
'distance' (default): w_j = 1 / d_j.  Closer neighbours contribute more.
    Exact matches (d=0) are handled without division by zero.

'uniform':            all K neighbours contribute equally.

Reproducibility
---------------
Results are fully deterministic.  Tied distances are broken by training-set
index (smallest index wins) via stable sort -- guaranteed to produce the
same result regardless of BLAS implementation, because the tie-breaking is
done in Python after the distance computation.

Performance
-----------
The distance matrix is computed in two nested loops over unique missingness
patterns.  Within each (query_pattern, train_pattern) pair the computation
is fully vectorized via BLAS-accelerated matrix multiplication:
    sq_dist = a^2 + b^2 - 2*a@b.T   (n_q_grp, n_t_grp)
"""

import numpy as np
from ._utils import feature_scale
from collections import defaultdict
from scipy.stats import norm as _scipy_norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase, MissTags, only_for
from ._conformance import check_common_parameters, check_choice, check_positive_int
from ._copula import RankNormalTransformer, needs_copula


# ======================================================================
# Shared base
# ======================================================================

class _MissNeighborsBase(MissBase):
    """
    Shared fit / distance / neighbour logic for regressor and classifier.
    Subclasses must implement predict(), score(), and summary().
    """

    # ------------------------------------------------------------------ #
    # Fit internals
    # ------------------------------------------------------------------ #

    def _fit_base(self, X):
        """
        Standardize X, store training data, precompute pattern groups.
        Optionally estimate correlation matrix for Mahalanobis metric.
        Called after any copula transform has been applied.
        """
        n, p = X.shape

        # Per-feature statistics on observed (non-NaN) values
        self.mu_    = np.nanmean(X, axis=0)
        # feature_scale applies the shared absolute-and-relative rule:
        # a constant column, a column with under two observed values, and
        # a column whose spread is negligible beside the widest one all
        # get a divisor of 1.0 rather than being amplified.
        self.sigma_ = feature_scale(X)

        Z = (X - self.mu_) / self.sigma_
        self.Z_train_ = Z       # standardized, NaN preserved
        self.X_train_ = X       # original scale, NaN preserved

        if self.metric == 'mahalanobis':
            complete = ~np.isnan(X).any(axis=1)
            Z_cc = Z[complete]
            if Z_cc.shape[0] >= p + 1:
                Corr = np.corrcoef(Z_cc, rowvar=False)
                min_eig = float(np.linalg.eigvalsh(Corr).min())
                jitter  = max(1e-6, -min_eig + 1e-4) if min_eig < 1e-4 else 1e-6
                self.Corr_X_ = Corr + jitter * np.eye(p)
            else:
                self.Corr_X_ = np.eye(p)   # fallback

        # Precompute training patterns once; used in every distance call
        groups = defaultdict(list)
        for j, z_j in enumerate(Z):
            key = tuple(np.where(~np.isnan(z_j))[0])
            if key:
                groups[key].append(j)
        self._train_patterns_ = [
            (np.array(k), np.array(v)) for k, v in groups.items()
        ]

        # Joint-Gaussian conditional moments for expected squared distances
        # (euclidean metric): E||z_q - z_t||^2 = ||F_q - F_t||^2 + s_q + s_t.
        # Far more reliable under missingness than rescaled available-case
        # distances, and no imputed dataset is ever created.
        if self.metric != 'mahalanobis':
            from ._imputer import _ConditionalMoments
            self._moments_ = _ConditionalMoments().fit(Z)
            self._F_train_, self._s_train_ = self._moments_.transform(Z)

    # ------------------------------------------------------------------ #
    # Standardize a query matrix using training statistics
    # ------------------------------------------------------------------ #

    def _standardize(self, X):
        """Standardize query X with training mu_ / sigma_. NaN preserved."""
        return (X - self.mu_) / self.sigma_

    # ------------------------------------------------------------------ #
    # Vectorized available-case distance matrix
    # ------------------------------------------------------------------ #

    def _compute_distances(self, Z_query):
        """
        Return the (n_query, n_train) distance matrix.

        euclidean (default): expected squared distance under the fitted
        joint Gaussian; missing dims contribute their conditional mean
        difference plus their conditional variance (Eirola et al., 2013):
            E||z_q − z_t||² = ||F_q − F_t||² + s_q + s_t.

        Every pair is comparable (no unreachable rows), and rows with more
        missingness are correctly treated as more distant/uncertain.

        mahalanobis: legacy available-case path; shared observed features,
        whitened by the training correlation, scaled by sqrt(p / n_shared).
        Entries where two rows share no features remain np.inf.
        """
        if self.metric != 'mahalanobis':
            F_q, s_q = self._moments_.transform(Z_query)
            sq_q  = np.sum(F_q ** 2, axis=1, keepdims=True)
            sq_t  = np.sum(self._F_train_ ** 2, axis=1)
            cross = F_q @ self._F_train_.T
            sq_d  = np.maximum(sq_q + sq_t - 2.0 * cross, 0.0)
            sq_d += s_q[:, None] + self._s_train_[None, :]
            return np.sqrt(sq_d)

        n_q  = Z_query.shape[0]
        n_t  = self.Z_train_.shape[0]
        p    = self.n_features_in_
        D    = np.full((n_q, n_t), np.inf)
        maha = self.metric == 'mahalanobis'

        # Group query rows by observed pattern
        q_groups = defaultdict(list)
        for i, z_i in enumerate(Z_query):
            key = tuple(np.where(~np.isnan(z_i))[0])
            if key:
                q_groups[key].append(i)

        for q_key, q_idxs in q_groups.items():
            q_arr = np.array(q_key)

            for t_arr, t_idxs in self._train_patterns_:
                shared = np.intersect1d(q_arr, t_arr)
                m = len(shared)
                if m == 0:
                    continue

                # Positions of the shared features within q_arr (sorted)
                q_pos = np.searchsorted(q_arr, shared)

                # Sub-matrices: (n_q_grp, m) and (n_t_grp, m)
                Z_q = Z_query[np.ix_(q_idxs, q_arr)][:, q_pos]
                Z_t = self.Z_train_[np.ix_(t_idxs, shared)]

                if maha:
                    # Cholesky-whiten via the correlation sub-matrix
                    try:
                        L   = np.linalg.cholesky(
                            self.Corr_X_[np.ix_(shared, shared)]
                        )
                        Z_q = np.linalg.solve(L, Z_q.T).T
                        Z_t = np.linalg.solve(L, Z_t.T).T
                    except np.linalg.LinAlgError:
                        pass   # fall back to standardized Euclidean

                # Squared Euclidean: a²+b²-2ab  (n_q_grp, n_t_grp)
                sq_q    = np.sum(Z_q ** 2, axis=1, keepdims=True)
                sq_t    = np.sum(Z_t ** 2, axis=1)
                cross   = Z_q @ Z_t.T
                sq_dist = np.maximum(sq_q + sq_t - 2.0 * cross, 0.0)

                # Scale to estimate full-dimensional distance
                D[np.ix_(q_idxs, t_idxs)] = np.sqrt(float(p) / m * sq_dist)

        return D

    # ------------------------------------------------------------------ #
    # Neighbour selection with deterministic tie-breaking
    # ------------------------------------------------------------------ #

    def _get_neighbors(self, D):
        """
        For each query row return (nn_indices, nn_distances) for the
        n_neighbors nearest training rows.

        Tie-breaking: distances sorted stably, so equal distances are
        broken by ascending training index.  Fully deterministic regardless
        of BLAS floating-point variation (the sort is Python-side).
        """
        k  = self.n_neighbors
        result = []

        for i in range(D.shape[0]):
            d        = D[i]
            finite   = np.where(np.isfinite(d))[0]
            n_f      = len(finite)

            if n_f == 0:
                result.append((np.array([], dtype=int), np.array([])))
                continue

            k_eff    = min(k, n_f)
            d_finite = d[finite]

            if k_eff < n_f:
                # argpartition is O(n) but selects arbitrarily among values
                # tied at the k-th distance.  Gather every candidate within
                # the k-th smallest distance, then stable-sort so ties break
                # by ascending training index as documented.
                kth  = np.partition(d_finite, k_eff - 1)[k_eff - 1]
                cand = np.where(d_finite <= kth)[0]
                order = np.argsort(d_finite[cand], kind='stable')
                sel   = cand[order][:k_eff]
            else:
                sel = np.argsort(d_finite, kind='stable')

            result.append((finite[sel], d_finite[sel]))

        return result

    # ------------------------------------------------------------------ #
    # Feature importance proxy
    # ------------------------------------------------------------------ #

    @property
    def feature_importances_(self):
        """
        Feature availability rate as proxy importance (sums to 1).

        For KNN there is no coefficient vector; instead, the fraction of
        training rows in which each feature is observed is used as a
        natural proxy: features that are always missing cannot contribute
        to any distance computation.
        """
        check_is_fitted(self)
        avail = np.mean(~np.isnan(self.X_train_), axis=0)
        total = avail.sum()
        return avail / total if total > 0 else avail


# ======================================================================
# MissNeighborsRegressor
# ======================================================================

class MissNeighborsRegressor(RegressorMixin, _MissNeighborsBase):
    """
    K-Nearest Neighbours regression with native missing-data support.

    Distances are computed on available shared features (scaled to estimate
    the full-dimensional distance).  The prediction is the distance-weighted
    mean of the K nearest neighbours' y values.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours (default 5).
    weights : {'distance', 'uniform'}
        'distance' (default): weight = 1/d.  Exact matches receive full
        weight; no division-by-zero.
        'uniform': all K neighbours contribute equally.
    metric : {'euclidean', 'mahalanobis'}
        Distance metric (default 'euclidean').
    copula : bool or 'auto', default 'auto'
        If True, apply a rank-normal transform to each feature before
        computing distances.  Useful for heavy-tailed or skewed features
        where Euclidean distance on the raw scale is poorly calibrated.
        'auto' applies the transform when any feature shows ``|skewness|``>1
        or ``|excess kurtosis|``>2.

    Attributes
    ----------
    Z_train_ : ndarray, shape (n, p)
        Standardized (and optionally copula-transformed) training X.
    y_train_ : ndarray, shape (n,)
        Training targets (original scale, NaN preserved).
    mu_, sigma_ : ndarray, shape (p,)
        Per-feature training mean and std used for standardization.
    n_neighbors_ : int
        Actual number of neighbours available (may be < n_neighbors for
        very small training sets).
    copula_used_ : bool
    """

    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean',
                 copula='auto'):
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.copula      = copula

    def fit(self, X, y):
        """
        Fit: store (optionally transformed) training data.

        No model optimization is performed; all computation happens at
        predict time.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.
        y : array-like, shape (n,).    NaN treated as missing.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        check_positive_int(self.n_neighbors, 'n_neighbors', type(self).__name__)
        # Consumed as `== 'uniform'` and `== 'mahalanobis'` below, so an
        # unrecognised value selects the other branch without a word.
        check_choice(self.weights, ('distance', 'uniform'), 'weights',
                     type(self).__name__)
        check_choice(self.metric, ('euclidean', 'mahalanobis'), 'metric',
                     type(self).__name__)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        if self.copula == 'auto':
            # X only. Unlike the other regressors this one leaves y on its
            # original scale, because it predicts by averaging neighbouring
            # y values and the copula is here to calibrate feature distances.
            # Passing y in let a skewed target trigger a feature transform
            # that had nothing to do with it.
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        # Keep the original-scale X for summary()/score paths, which apply
        # the copula transform themselves (X_train_ is post-copula).
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        self.y_train_ = y.copy()
        self._fit_base(X)

        # Overall fallback prediction (used when no neighbours are reachable)
        obs_y = ~np.isnan(y)
        self._y_fallback = float(np.mean(y[obs_y])) if obs_y.any() else 0.0
        self._y_global_std = float(np.std(y[obs_y], ddof=1)) if obs_y.sum() > 1 else 1.0
        self.n_neighbors_ = min(self.n_neighbors, int(obs_y.sum()))

        return self

    # ------------------------------------------------------------------ #
    # Internal prediction helpers
    # ------------------------------------------------------------------ #

    def _aggregate(self, neighbors, uninformative=None):
        """
        Aggregate neighbour y values into predictions.

        Returns (y_hat, y_se) arrays where y_se is the weighted std of
        neighbour y values (used by predict_interval).

        Parameters
        ----------
        neighbors : list of (indices, distances), one entry per query row.
        uninformative : boolean ndarray or None
            Rows of the query with no observed feature at all. For these the
            expected distance is identical to every training point, so the
            neighbours returned are an arbitrary subset and their spread is a
            sample of k response values rather than a statement about the
            query. The marginal spread of the response is used instead, which
            is what the two existing fallbacks below already use when the
            neighbourhood itself is unusable. Without this the interval for a
            row nothing is known about came out narrower than for a row
            something is known about.
        """
        n         = len(neighbors)
        y_hat     = np.empty(n)
        y_se      = np.empty(n)
        y_train   = self.y_train_
        fallback  = self._y_fallback
        g_std     = self._y_global_std
        uniform   = self.weights == 'uniform'

        for i, (nn_idx, nn_dist) in enumerate(neighbors):
            if len(nn_idx) == 0:
                y_hat[i] = fallback
                y_se[i]  = g_std
                continue

            y_nn   = y_train[nn_idx]
            obs_y  = ~np.isnan(y_nn)
            if not obs_y.any():
                y_hat[i] = fallback
                y_se[i]  = g_std
                continue

            y_obs  = y_nn[obs_y]
            d_obs  = nn_dist[obs_y]

            if uniform:
                w = np.ones(len(y_obs))
            else:
                zero = d_obs == 0.0
                if zero.any():
                    # Exact match: use only zero-distance neighbours
                    y_obs  = y_obs[zero]
                    w      = np.ones(zero.sum())
                else:
                    w = 1.0 / d_obs

            w_sum     = w.sum()
            w_norm    = w / w_sum
            mu        = float(np.dot(w_norm, y_obs))
            var       = float(np.dot(w_norm, (y_obs - mu) ** 2))
            y_hat[i]  = mu
            if uninformative is not None and uninformative[i]:
                # Nothing observed in this query row, so the neighbourhood is
                # arbitrary and its spread says nothing about the query. Fall
                # back to the marginal spread, as the two branches above do
                # when the neighbourhood itself is unusable.
                y_se[i] = g_std
            else:
                y_se[i] = float(np.sqrt(var)) if var > 0 else 0.0

        return y_hat, y_se

    def _predict_internal(self, X):
        """Predict on already-transformed X (after copula if any)."""
        Z         = self._standardize(X)
        D         = self._compute_distances(Z)
        neighbors = self._get_neighbors(D)
        y_hat, _  = self._aggregate(neighbors)
        return y_hat, neighbors

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, X):
        """
        Predict E[Y | X_obs].

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.

        Returns
        -------
        y_hat : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        y_hat, _ = self._predict_internal(X)
        return y_hat

    def predict_interval(self, X, alpha=0.05):
        """
        Empirical prediction interval from the spread of neighbour y values.

        The interval is y_hat ± z_{alpha/2} * (weighted std of K neighbour y
        values).  It widens automatically when the nearest neighbours are
        spread out (high local variance) or when few neighbours are available
        (missing features reduce the effective neighbourhood).

        Parameters
        ----------
        X     : array-like, shape (n, p).  NaN allowed.
        alpha : float, significance level (default 0.05 for 95%).

        Returns
        -------
        lower, upper : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)

        Z         = self._standardize(X)
        D         = self._compute_distances(Z)
        neighbors = self._get_neighbors(D)
        # Rows with no observed feature: the neighbourhood is arbitrary for
        # these, so _aggregate uses the marginal spread rather than the
        # spread of whichever k points came back.
        y_hat, y_se = self._aggregate(
            neighbors, uninformative=np.isnan(X).all(axis=1))

        z     = _scipy_norm.ppf(1.0 - alpha / 2.0)
        lower = y_hat - z * y_se
        upper = y_hat + z * y_se
        return lower, upper

    def score(self, X, y):
        """R² on the subset where y is not NaN."""
        check_is_fitted(self)
        X, y  = self._validate_and_convert(X, y)
        y_hat = self.predict(X)
        obs   = ~np.isnan(y)
        y_o, yh_o = y[obs], y_hat[obs]
        ss_res = np.sum((y_o - yh_o) ** 2)
        ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def summary(self):
        """Print a model summary."""
        check_is_fitted(self)
        W = 72
        print()
        print('=' * W)
        print('MissNeighborsRegressor  --  KNN Regression  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  n_neighbors     : {self.n_neighbors}  "
              f"(effective: {self.n_neighbors_})")
        print(f"  Weights         : {self.weights}")
        print(f"  Metric          : {self.metric}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (distances on normal-transformed scale)")
        print(f"  Train R²        : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        print("  Feature Availability (used as proxy importance):")
        avail = np.mean(~np.isnan(self.X_train_), axis=0)
        order = np.argsort(avail)[::-1]
        for i in order:
            name = self.feature_names_in_[i]
            bar  = self._bar(avail[i])
            print(f"  {name:>14}: {bar}  {avail[i] * 100:5.1f}% available")
        print('=' * W)
        print()


# ======================================================================
# MissNeighborsClassifier
# ======================================================================

class MissNeighborsClassifier(ClassifierMixin, _MissNeighborsBase):
    """
    K-Nearest Neighbours classification with native missing-data support.

    Distances are computed on available shared features.  The prediction
    is the distance-weighted majority vote among K nearest neighbours.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours (default 5).
    weights : {'distance', 'uniform'}
        Voting scheme (default 'distance').
    metric : {'euclidean', 'mahalanobis'}
        Distance metric (default 'euclidean').
    copula : bool or 'auto', default 'auto'
        Rank-normal transform on features before distance computation.

    Attributes
    ----------
    classes_ : ndarray, shape (2,)
    Z_train_, y_train_, mu_, sigma_ : see MissNeighborsRegressor
    copula_used_ : bool
    """

    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean',
                 copula='auto'):
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.copula      = copula

    def fit(self, X, y):
        """
        Fit: validate binary labels, store training data.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.
        y : array-like, shape (n,).    Binary {0, 1}; NaN treated as missing.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        check_positive_int(self.n_neighbors, 'n_neighbors', type(self).__name__)
        # Consumed as `== 'uniform'` and `== 'mahalanobis'` below, so an
        # unrecognised value selects the other branch without a word.
        check_choice(self.weights, ('distance', 'uniform'), 'weights',
                     type(self).__name__)
        check_choice(self.metric, ('euclidean', 'mahalanobis'), 'metric',
                     type(self).__name__)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        y_obs = y[~np.isnan(y)]
        _classes = np.unique(y_obs)
        # Int cast only for integer-valued labels; truncating fractional
        # labels would break the classes_[1] comparisons below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissNeighborsClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )

        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        # Keep the original-scale X for summary()/score paths, which apply
        # the copula transform themselves (X_train_ is post-copula).
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        self.y_train_ = y.copy()
        self._fit_base(X)

        # Fallback class probability (class proportions from observed labels)
        n_pos = float(np.sum(y_obs == self.classes_[1]))
        self._p1_fallback = n_pos / len(y_obs) if len(y_obs) > 0 else 0.5
        self.n_neighbors_ = min(self.n_neighbors, int((~np.isnan(y)).sum()))

        return self

    # ------------------------------------------------------------------ #
    # Internal probability computation
    # ------------------------------------------------------------------ #

    def _class_proba(self, neighbors):
        """
        Compute P(Y=1 | X_obs) for each query row from its neighbours.
        Returns ndarray of shape (n_query,).
        """
        n       = len(neighbors)
        p1      = np.empty(n)
        y_train = self.y_train_
        uniform = self.weights == 'uniform'

        for i, (nn_idx, nn_dist) in enumerate(neighbors):
            if len(nn_idx) == 0:
                p1[i] = self._p1_fallback
                continue

            y_nn  = y_train[nn_idx]
            obs_y = ~np.isnan(y_nn)
            if not obs_y.any():
                p1[i] = self._p1_fallback
                continue

            y_obs = y_nn[obs_y]
            d_obs = nn_dist[obs_y]

            if uniform:
                w = np.ones(len(y_obs))
            else:
                zero = d_obs == 0.0
                if zero.any():
                    y_obs = y_obs[zero]
                    w     = np.ones(zero.sum())
                else:
                    w = 1.0 / d_obs

            p1[i] = float(np.dot(w, y_obs == self.classes_[1]) / w.sum())

        return p1

    def _run_predict(self, X):
        """Predict on already-transformed X; returns (proba_matrix, neighbors)."""
        Z         = self._standardize(X)
        D         = self._compute_distances(Z)
        neighbors = self._get_neighbors(D)
        p1        = self._class_proba(neighbors)
        proba     = np.column_stack([1.0 - p1, p1])
        proba     = np.clip(proba, 1e-15, 1.0 - 1e-15)
        proba    /= proba.sum(axis=1, keepdims=True)
        return proba, neighbors

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_proba(self, X):
        """
        Class probabilities P(Y=0|X_obs) and P(Y=1|X_obs).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.

        Returns
        -------
        proba : ndarray, shape (n, 2)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        proba, _ = self._run_predict(X)
        return proba

    def predict(self, X):
        """Predict binary class labels."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Log-odds score P(Y=1) / P(Y=0).  +inf / -inf at probability 1 / 0.
        """
        check_is_fitted(self)
        proba  = self.predict_proba(X)
        p1     = np.clip(proba[:, 1], 1e-15, 1.0 - 1e-15)
        return np.log(p1 / (1.0 - p1))

    def score(self, X, y):
        """Accuracy on the subset where y is not NaN."""
        check_is_fitted(self)
        X, y   = self._validate_and_convert(X, y)
        obs    = ~np.isnan(y)
        y_pred = self.predict(X[obs])
        return float(np.mean(y_pred == y[obs]))

    def summary(self):
        """Print a model summary."""
        check_is_fitted(self)
        W = 72
        print()
        print('=' * W)
        print('MissNeighborsClassifier  --  KNN Classification (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Classes         : {self.classes_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  n_neighbors     : {self.n_neighbors}  "
              f"(effective: {self.n_neighbors_})")
        print(f"  Weights         : {self.weights}")
        print(f"  Metric          : {self.metric}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (distances on normal-transformed scale)")
        print(f"  Train accuracy  : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        print("  Feature Availability (used as proxy importance):")
        avail = np.mean(~np.isnan(self.X_train_), axis=0)
        order = np.argsort(avail)[::-1]
        for i in order:
            name = self.feature_names_in_[i]
            bar  = self._bar(avail[i])
            print(f"  {name:>14}: {bar}  {avail[i] * 100:5.1f}% available")
        print('=' * W)
        print()


# ======================================================================
# MissNeighbors  --  unified auto-selecting wrapper
# ======================================================================

class MissNeighbors(MissTags, BaseEstimator):
    """
    Unified KNN model that automatically selects regression or classification
    based on the observed values of y.

    Detection rule (applied at fit time):
        All observed y in {0, 1}  ->  MissNeighborsClassifier
        Otherwise                 ->  MissNeighborsRegressor

    The underlying model is stored in ``model_`` and the task in ``task_``
    ('regression' or 'classification') after fitting.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours (default 5).
    weights : {'distance', 'uniform'}
        Voting / averaging scheme (default 'distance').
    metric : {'euclidean', 'mahalanobis'}
        Distance metric (default 'euclidean').
    copula : bool or 'auto', default 'auto'
        Rank-normal feature transform (default False).
    """


    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean',
                 copula='auto'):
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.copula      = copula

    @staticmethod

    def _detect_task(y):
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        obs   = y_arr[~np.isnan(y_arr)]
        if len(obs) == 0:
            raise ValueError("y has no observed (non-NaN) values.")
        if set(np.unique(obs).tolist()) <= {0.0, 1.0}:
            return 'classification'
        return 'regression'


    def fit(self, X, y):
        """
        Detect task and fit the appropriate KNN model.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.
        y : array-like, shape (n,).    NaN treated as missing.

        Returns
        -------
        self
        """
        # The dispatcher constructs a concrete estimator and hands
        # the work over, so any parameter it does not pass on is
        # never validated by anything. n_quadrature is the case
        # that exposed it: it belongs to the classifier, and on a
        # regression target it was simply dropped, leaving
        # n_quadrature=0 accepted here and refused by the sibling.
        check_common_parameters(self)
        self.task_ = self._detect_task(y)
        kw = dict(n_neighbors=self.n_neighbors, weights=self.weights,
                  metric=self.metric, copula=self.copula)
        if self.task_ == 'classification':
            self.model_ = MissNeighborsClassifier(**kw)
        else:
            self.model_ = MissNeighborsRegressor(**kw)
        self.model_.fit(X, y)
        return self


    def __getattr__(self, name):
        if name.startswith('_') or name in ('model_', 'task_'):
            raise AttributeError(name)
        try:
            model = object.__getattribute__(self, 'model_')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'. "
                "Has the model been fitted yet?"
            )
        return getattr(model, name)


    def predict(self, X):
        """Predict target values (regression) or class labels (classification)."""
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)

    @available_if(only_for('classification'))
    def predict_proba(self, X):
        """Class probabilities.  Only available for classification."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'classification':
            raise AttributeError(
                "predict_proba is only available when task_ == 'classification'."
            )
        return self.model_.predict_proba(X)

    @available_if(only_for('classification'))
    def decision_function(self, X):
        """Log-odds scores.  Only available for classification."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'classification':
            raise AttributeError(
                "decision_function is only available when task_ == 'classification'."
            )
        return self.model_.decision_function(X)

    @available_if(only_for('regression'))
    def predict_interval(self, X, alpha=0.05):
        """Empirical prediction interval.  Only available for regression."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'regression':
            raise AttributeError(
                "predict_interval is only available when task_ == 'regression'."
            )
        return self.model_.predict_interval(X, alpha=alpha)


    def score(self, X, y):
        """R² (regression) or accuracy (classification) on non-NaN targets."""
        check_is_fitted(self, 'model_')
        return self.model_.score(X, y)


    def summary(self):
        """Print the model summary."""
        check_is_fitted(self, 'model_')
        return self.model_.summary()
