"""
MissLearn._svm
--------------
Support Vector Machine models with native missing-data support.

Available-case kernel
---------------------
The key to handling missingness without imputation is redefining the kernel
between two observations x, x' to use only the features observed in BOTH
rows (the "available-case" intersection), then scaling the result by p/m:

    m = |obs(x) ∩ obs(x')|          shared observed features
    s = p / m                        MAR-motivated scale factor

    RBF    : K(x,x') = exp(-γ · s · Σ_shared (x_j−x'_j)²)
    Linear : K(x,x') = s · Σ_shared x_j x'_j
    Poly   : K(x,x') = (γ · s · Σ_shared x_j x'_j + r)^d

Under Missing-At-Random, E[d_full²] = (p/m) · E[d_shared²], so the scaled
partial kernel approximates what the full-feature kernel would compute.
Row pairs sharing no features get K = 0 and contribute nothing to the SVM
decision; predictions for fully-missing rows fall back to the bias term.

The kernel matrices are passed to sklearn's LIBSVM solver via
kernel='precomputed'.  The solver itself is unchanged -- only the kernel
definition is adapted for missingness.

Prediction intervals (regression)
----------------------------------
SVR has no native uncertainty output.  MissSupportRegressor derives empirical
intervals from the training-residual standard deviation σ_train, inflated
by the fraction of missing features in each query row:

    half-width_i = z_{α/2} · σ_train · sqrt(p / max(m_i, 1))

where m_i = |obs(x_i)|.  Intervals widen automatically as features are
missing, consistent with all other MissLearn regression models.

Reproducibility
---------------
Fully deterministic.  LIBSVM is deterministic for a given kernel matrix.
SVC with probability=True uses Platt scaling (internal 5-fold CV) seeded
with random_state=0.  The kernel computation uses stable NumPy operations
with no randomness.

Performance
-----------
Kernel computation groups row pairs by missingness pattern and vectorizes
the inner products / squared distances within each block via BLAS matrix
multiply, identical in structure to MissNeighbors.  The training kernel matrix
is symmetrized after computation to cancel any floating-point asymmetry.
"""

import numpy as np
from ._utils import feature_scale
from collections import defaultdict
from scipy.stats import norm as _scipy_norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.svm import SVR, SVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase, MissTags, only_for
from ._conformance import check_common_parameters, check_positive_int
from ._copula import RankNormalTransformer, needs_copula


# ======================================================================
# Shared base
# ======================================================================

class _MissSupportBase(MissBase):
    """
    Shared gamma resolution, vectorized available-case kernel computation,
    and feature importances for MissSupportRegressor and MissSupportClassifier.
    """

    # ------------------------------------------------------------------ #
    # Gamma
    # ------------------------------------------------------------------ #

    def _resolve_gamma(self, Z):
        """
        Compute the γ scalar for RBF / poly kernels.

        'scale' : 1 / (p × mean_variance_of_the_expected-kernel_embedding).
                  The embedding is (F, √s·e_i), so its total variance is the
                  variance of the conditionally-filled features plus the mean
                  per-row conditional variance; matching the scale of the
                  expected squared distances the kernel actually sees.

        'auto'  : 1 / p
        float   : used directly
        """
        p = Z.shape[1]
        if self.gamma == 'scale':
            F, s = self._moments_of(Z) if hasattr(self, '_moments_') else (None, None)
            if F is not None:
                total_var = float(F.var(axis=0).sum() + s.mean())
            else:
                total_var = float(np.nansum(
                    [np.nanvar(Z[:, j]) for j in range(p)]
                ))
            return 1.0 / total_var if total_var > 0 else 1.0 / p
        elif self.gamma == 'auto':
            return 1.0 / p
        else:
            try:
                g = float(self.gamma)
            except (TypeError, ValueError):
                raise ValueError(
                    "gamma must be 'scale', 'auto', or a positive number; "
                    "got %r." % (self.gamma,))
            # A negative gamma turns exp(-gamma * d^2) into exp(+|gamma| *
            # d^2). The kernel stops being positive semi-definite, the fit
            # proceeds anyway, and the result is unusable rather than merely
            # poor: gamma=-1 on ordinary data returned an R^2 of -1.7e34.
            # Nothing downstream notices, so it has to be caught here.
            if not np.isfinite(g) or g <= 0.0:
                raise ValueError(
                    "gamma must be positive; got %r. A non-positive gamma "
                    "makes the kernel non-PSD, and the model fits without "
                    "complaint on a kernel that is no longer a similarity."
                    % (self.gamma,))
            return g

    # ------------------------------------------------------------------ #
    # Available-case kernel matrix
    # ------------------------------------------------------------------ #

    def _moments_of(self, Z):
        """(F, s) conditional moments for Z, cached for the training matrix."""
        if Z is self._Z_train_obs_:
            return self._F_train_, self._s_train_
        return self._moments_.transform(Z)

    def _compute_kernel(self, Z_a, Z_b):
        """
        Expected kernel under the fitted joint Gaussian, shape (n_a, n_b).

        Both Z_a and Z_b must be already standardized (NaN preserved).
        Missing dims contribute their conditional mean plus conditional
        variance to the expected squared distance / inner product:
            E||z_a − z_b||² = ||F_a − F_b||² + s_a + s_b   (a ≠ b)
            E<z_a, z_b>     = <F_a, F_b>                    (a ≠ b)
            E||z_a||²       = ||F_a||² + s_a                (diagonal)

        These are exact Euclidean quantities in an augmented space (each
        row embeds as (F_i, √s_i·e_i) with orthogonal uncertainty axes),
        so every kernel built on them stays PSD; unlike rescaled
        available-case kernels.  Identical objects (Z_a is Z_b) are treated
        as the same realisation, giving an exact-zero self-distance.
        """
        kname  = self.kernel
        gamma_ = self._gamma_
        same   = Z_a is Z_b

        F_a, s_a = self._moments_of(Z_a)
        F_b, s_b = (F_a, s_a) if same else self._moments_of(Z_b)

        cross = F_a @ F_b.T

        if kname == 'rbf':
            sq_a = np.sum(F_a ** 2, axis=1, keepdims=True)
            sq_b = np.sum(F_b ** 2, axis=1)
            sq_d = np.maximum(sq_a + sq_b - 2.0 * cross, 0.0)
            sq_d = sq_d + s_a[:, None] + s_b[None, :]
            if same:
                np.fill_diagonal(sq_d, 0.0)
            return np.exp(-gamma_ * sq_d)

        if kname in ('linear', 'poly'):
            G = cross
            if same:
                G = G.copy()
                G[np.diag_indices_from(G)] = np.sum(F_a ** 2, axis=1) + s_a
            if kname == 'linear':
                return G
            return (gamma_ * G + self.coef0) ** self.degree

        raise ValueError(
            f"kernel must be 'rbf', 'linear', or 'poly'; got {kname!r}"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _standardize(self, X):
        """Standardize with training mu / sigma; NaN preserved."""
        return (X - self._mu_) / self._sigma_

    def _kernel_test(self, X):
        """Return K(standardize(X), Z_train_obs_) -- (n_test, n_train_obs)."""
        return self._compute_kernel(self._standardize(X), self._Z_train_obs_)

    # ------------------------------------------------------------------ #
    # Feature importances
    # ------------------------------------------------------------------ #

    @property
    def feature_importances_(self):
        """
        Feature importances, normalized to sum to 1.

        Linear kernel: absolute reconstructed weight coefficients
            w_j = Σ_{i∈SVs} α_i · z_ij   (NaN entries in z treated as 0).

        RBF / poly: feature availability rates -- the fraction of training
            rows in which each feature is observed.  Features that are
            always missing cannot contribute to any kernel evaluation.
        """
        check_is_fitted(self)
        if self.kernel == 'linear':
            sv_idx = self._svm_.support_
            Z_sv   = self._Z_train_obs_[sv_idx, :]
            Z_sv_f = np.where(np.isnan(Z_sv), 0.0, Z_sv)
            alpha  = self._svm_.dual_coef_.ravel()   # (n_sv,)
            imp    = np.abs(alpha @ Z_sv_f)          # (p,)
        else:
            imp = np.mean(~np.isnan(self._X_orig_), axis=0).astype(float)

        total = imp.sum()
        return imp / total if total > 0 else np.full(len(imp), 1.0 / len(imp))


# ======================================================================
# MissSupportRegressor
# ======================================================================

class MissSupportRegressor(RegressorMixin, _MissSupportBase):
    """
    Support Vector Regression with native missing-data support.

    Uses an available-case kernel -- distances / inner products are computed
    only on the features observed in both rows, scaled by p/m to estimate
    the full-dimensional quantity under MAR.  The fitted solver is sklearn's
    SVR with kernel='precomputed'.

    Parameters
    ----------
    C : float
        Regularization strength (default 1.0).  Smaller values impose more
        regularization and produce a wider margin.
    epsilon : float
        Width of the SVR insensitive tube (default 0.1).  Residuals inside
        the tube incur no loss.
    kernel : {'rbf', 'linear', 'poly'}
        Available-case kernel type (default 'rbf').
    gamma : float or {'scale', 'auto'}
        Kernel coefficient for 'rbf' and 'poly' (default 'scale').
        'scale' = 1 / (p × mean_feature_variance).
    degree : int
        Degree for the polynomial kernel (default 3).
    coef0 : float
        Independent term in the polynomial kernel (default 0.0).
    copula : bool or 'auto', default 'auto'
        Rank-normal transform applied to X and y before fitting (default
        False).  Predictions are inverse-transformed back to the original
        y scale.  'auto' triggers when any feature or y shows
        ``|skewness|`` > 1 or ``|excess kurtosis|`` > 2.

    Attributes
    ----------
    _svm_        : sklearn SVR instance
    n_support_   : int -- number of support vectors
    _sigma_train_: float -- std of training residuals (used for intervals)
    feature_importances_ : ndarray (p,)
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale',
                 degree=3, coef0=0.0, copula='auto'):
        self.C       = C
        self.epsilon = epsilon
        self.kernel  = kernel
        self.gamma   = gamma
        self.degree  = degree
        self.coef0   = coef0
        self.copula  = copula

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit SVR on the available-case kernel matrix.

        Rows with missing y are excluded from the SVM fit.  Missing values
        in X are handled by the available-case kernel during both fitting and
        prediction -- no rows are dropped for X-missingness alone.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.
        y : array-like, shape (n,).    NaN treated as missing (rows excluded).

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        check_positive_int(self.degree, 'degree', type(self).__name__)
        self._store_fit_metadata(X, y)
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        # ------ copula -----------------------------------------------
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            self._copula_y_ = RankNormalTransformer().fit(y.reshape(-1, 1))
            X = self._copula_X_.transform(X)
            y = self._copula_y_.transform(y.reshape(-1, 1))[:, 0]

        # ------ standardize X ----------------------------------------
        self._mu_    = np.nanmean(X, axis=0)
        # feature_scale applies the shared absolute-and-relative rule:
        # a constant column, a column with under two observed values, and
        # a column whose spread is negligible beside the widest one all
        # get a divisor of 1.0 rather than being amplified.
        self._sigma_ = feature_scale(X)
        Z = self._standardize(X)

        # ------ standardize y ----------------------------------------
        # SVR's C and epsilon are only meaningful on a unit response scale
        # (sklearn's own guidance); fit on standardized y and unscale
        # predictions at the API boundary.
        _y_obs_vals = y[~np.isnan(y)]
        self._y_center_ = float(np.mean(_y_obs_vals)) if _y_obs_vals.size else 0.0
        _sy = (float(np.std(_y_obs_vals, ddof=1))
               if _y_obs_vals.size > 1 else 1.0)
        self._y_scale_ = _sy if np.isfinite(_sy) and _sy >= 1e-8 else 1.0
        y = np.where(np.isnan(y), np.nan,
                     (y - self._y_center_) / self._y_scale_)

        # ------ drop rows with missing y -----------------------------
        obs_y = ~np.isnan(y)
        self._Z_train_obs_ = Z[obs_y]
        self._y_train_obs_ = y[obs_y]

        # ------ conditional moments for the expected kernel ----------
        # The joint Gaussian is fitted on ALL rows (missing-y rows still
        # inform the feature distribution); moments are cached for the
        # training kernel rows.
        from ._imputer import _ConditionalMoments
        self._moments_ = _ConditionalMoments().fit(Z)
        self._F_train_, self._s_train_ = self._moments_.transform(
            self._Z_train_obs_
        )

        # ------ kernel -----------------------------------------------
        self._gamma_ = self._resolve_gamma(self._Z_train_obs_)
        K_train = self._compute_kernel(self._Z_train_obs_, self._Z_train_obs_)
        K_train = (K_train + K_train.T) / 2.0     # symmetrize

        # ------ fit SVR ----------------------------------------------
        self._svm_ = SVR(kernel='precomputed', C=self.C, epsilon=self.epsilon)
        self._svm_.fit(K_train, self._y_train_obs_)
        self.n_support_ = int(self._svm_.support_.shape[0])

        # ------ training residuals for predict_interval --------------
        y_hat_tr = self._svm_.predict(K_train)
        resid    = self._y_train_obs_ - y_hat_tr
        self._sigma_train_ = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0

        return self

    # ------------------------------------------------------------------ #
    # Predict
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
        y_hat = self._svm_.predict(self._kernel_test(X))
        y_hat = self._y_center_ + self._y_scale_ * y_hat
        if self.copula_used_:
            y_hat = self._copula_y_.inverse_transform(y_hat.reshape(-1, 1))[:, 0]
        return y_hat

    def predict_interval(self, X, alpha=0.05):
        """
        Empirical prediction interval.

        Width = 2 · z_{α/2} · σ_train · sqrt(p / max(m_i, 1))

        where σ_train is the std of training residuals and m_i counts the
        observed features in query row i.  The interval is widest for
        fully-missing rows (m_i = 0 → scaled to sqrt(p) × σ_train) and
        narrowest for complete rows (m_i = p → ± z · σ_train).

        Parameters
        ----------
        X     : array-like, shape (n, p).  NaN allowed.
        alpha : float, significance level (default 0.05 → 95%).

        Returns
        -------
        lower, upper : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)

        # Count observed features BEFORE any transform (original missingness)
        m_i   = (~np.isnan(X)).sum(axis=1).astype(float)
        m_i   = np.maximum(m_i, 1.0)
        scale = np.sqrt(float(self.n_features_in_) / m_i)

        X_k = X.copy()
        if self.copula_used_:
            X_k = self._copula_X_.transform(X_k)

        y_hat_n = self._svm_.predict(self._kernel_test(X_k))    # standardized space

        z     = _scipy_norm.ppf(1.0 - alpha / 2.0)
        half  = z * self._sigma_train_ * scale
        lower = self._y_center_ + self._y_scale_ * (y_hat_n - half)
        upper = self._y_center_ + self._y_scale_ * (y_hat_n + half)

        if self.copula_used_:
            lower = self._copula_y_.inverse_transform(lower.reshape(-1, 1))[:, 0]
            upper = self._copula_y_.inverse_transform(upper.reshape(-1, 1))[:, 0]

        return lower, upper

    # ------------------------------------------------------------------ #
    # Score / summary
    # ------------------------------------------------------------------ #

    def score(self, X, y):
        """R² on the subset where y is not NaN."""
        check_is_fitted(self)
        X, y  = self._validate_and_convert(X, y)
        y_hat = self.predict(X)
        obs   = ~np.isnan(y)
        y_o, yh = y[obs], y_hat[obs]
        ss_res  = np.sum((y_o - yh) ** 2)
        ss_tot  = np.sum((y_o - np.mean(y_o)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def summary(self):
        """Print a model summary."""
        check_is_fitted(self)
        W   = 72
        p   = self.n_features_in_
        imp = self.feature_importances_
        print()
        print('=' * W)
        print('MissSupportRegressor  --  SVM Regression  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations   : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features       : {p}")
        print(f"  Missing (X)    : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)    : {self.n_missing_y_}")
        gamma_str = (f"  gamma={self._gamma_:.4g}" if self.kernel != 'linear' else '')
        print(f"  Kernel         : {self.kernel}{gamma_str}"
              + (f"  degree={self.degree}" if self.kernel == 'poly' else ''))
        print(f"  C={self.C}   epsilon={self.epsilon}")
        print(f"  Support vectors: {self.n_support_} / {len(self._y_train_obs_)}")
        print(f"  sigma_train    : {self._sigma_train_:.4f}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula         : {label}")
        elif self.copula_used_:
            print(f"  Copula         : yes")
        print(f"  Train R²       : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        label = ('weight |w_j|' if self.kernel == 'linear'
                 else 'availability')
        print(f"  Feature importances ({label}, normalized):")
        order = np.argsort(imp)[::-1]
        for j in order:
            name = self.feature_names_in_[j]
            bar  = self._bar(imp[j])
            print(f"  {name:>14}: {bar}  {imp[j] * 100:5.1f}%  "
                  f"(avail {avail[j] * 100:.0f}%)")
        print('=' * W)
        print()


# ======================================================================
# MissSupportClassifier
# ======================================================================

class MissSupportClassifier(ClassifierMixin, _MissSupportBase):
    """
    Support Vector Classification with native missing-data support.

    Uses an available-case kernel (see module docstring).  The fitted solver
    is sklearn's SVC with kernel='precomputed' and probability=True, which
    adds Platt scaling for calibrated class probabilities.

    Parameters
    ----------
    C : float
        Regularization strength (default 1.0).
    kernel : {'rbf', 'linear', 'poly'}
        Available-case kernel type (default 'rbf').
    gamma : float or {'scale', 'auto'}
        Kernel coefficient for 'rbf' and 'poly' (default 'scale').
    degree : int
        Polynomial kernel degree (default 3).
    coef0 : float
        Polynomial independent term (default 0.0).
    copula : bool or 'auto', default 'auto'
        Rank-normal transform of X only (y is discrete; default False).

    Attributes
    ----------
    classes_     : ndarray (2,)
    class_prior_ : ndarray (2,) -- [P(Y=0), P(Y=1)]
    _svm_        : sklearn SVC instance (probability=True, random_state=0)
    n_support_   : int
    feature_importances_ : ndarray (p,)
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 degree=3, coef0=0.0, copula='auto'):
        self.C      = C
        self.kernel = kernel
        self.gamma  = gamma
        self.degree = degree
        self.coef0  = coef0
        self.copula = copula

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit SVC on the available-case kernel matrix.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN treated as missing.
        y : array-like, shape (n,).    Binary {0,1}; NaN treated as missing.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        check_positive_int(self.degree, 'degree', type(self).__name__)
        self._store_fit_metadata(X, y)
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        y_obs = y[~np.isnan(y)]
        _classes = np.sort(np.unique(y_obs))
        # Int cast only for integer-valued labels; truncating fractional
        # labels would break the classes_[1] comparisons below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissSupportClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )
        n_pos = float(np.sum(y_obs == self.classes_[1]))
        self.class_prior_ = np.array([1.0 - n_pos / len(y_obs),
                                      n_pos / len(y_obs)])

        # ------ copula (X only -- y is discrete) ---------------------
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        # ------ standardize X ----------------------------------------
        self._mu_    = np.nanmean(X, axis=0)
        # feature_scale applies the shared absolute-and-relative rule:
        # a constant column, a column with under two observed values, and
        # a column whose spread is negligible beside the widest one all
        # get a divisor of 1.0 rather than being amplified.
        self._sigma_ = feature_scale(X)
        Z = self._standardize(X)

        # ------ drop rows with missing y -----------------------------
        obs_y = ~np.isnan(y)
        self._Z_train_obs_ = Z[obs_y]
        # Binarize against classes_[1] so the SVC's proba column 1 always
        # corresponds to classes_[1] regardless of the original label values
        # (astype(int) would truncate fractional labels).
        self._y_train_obs_ = (y[obs_y] == self.classes_[1]).astype(int)

        # ------ conditional moments for the expected kernel ----------
        # The joint Gaussian is fitted on ALL rows (missing-y rows still
        # inform the feature distribution); moments are cached for the
        # training kernel rows.
        from ._imputer import _ConditionalMoments
        self._moments_ = _ConditionalMoments().fit(Z)
        self._F_train_, self._s_train_ = self._moments_.transform(
            self._Z_train_obs_
        )

        # ------ kernel -----------------------------------------------
        self._gamma_ = self._resolve_gamma(self._Z_train_obs_)
        K_train = self._compute_kernel(self._Z_train_obs_, self._Z_train_obs_)
        K_train = (K_train + K_train.T) / 2.0

        # ------ fit SVC (probability=True for predict_proba) ---------
        self._svm_ = SVC(kernel='precomputed', C=self.C,
                         probability=True, random_state=0)
        self._svm_.fit(K_train, self._y_train_obs_)
        self.n_support_ = int(self._svm_.support_.shape[0])

        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict_proba(self, X):
        """
        Class probabilities [P(Y=0|X_obs), P(Y=1|X_obs)] via Platt scaling.

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
        proba = self._svm_.predict_proba(self._kernel_test(X))
        proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """
        Predict binary class labels (argmax of predict_proba).

        Using predict_proba (rather than the raw SVM decision) is consistent
        with all other MissLearn classifiers and avoids the minor discrepancy
        that can arise between the raw decision function and Platt-calibrated
        probabilities.
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Raw SVM decision scores (signed distance to the separating hyperplane).

        Larger values favour class 1 and the ordering matches
        ``predict_proba`` exactly, but the sign is not quite the same
        boundary as the label. ``predict`` is the argmax of the
        Platt-calibrated probability, and Platt fits an intercept, so the
        point where this score crosses zero is not in general the point where
        the probability crosses a half. Rows in that narrow band, and rows
        whose two probabilities tie exactly, can have a positive score and a
        predicted label of class 0. Use ``predict`` or ``predict_proba`` when
        the label matters, and this when the geometry does.
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        return self._svm_.decision_function(self._kernel_test(X))

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
        W   = 72
        p   = self.n_features_in_
        imp = self.feature_importances_
        print()
        print('=' * W)
        print('MissSupportClassifier  --  SVM Classification  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations   : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features       : {p}")
        print(f"  Classes        : {self.classes_}")
        print(f"  Class priors   : P(Y={self.classes_[0]})={self.class_prior_[0]:.3f}  "
              f"P(Y={self.classes_[1]})={self.class_prior_[1]:.3f}")
        print(f"  Missing (X)    : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)    : {self.n_missing_y_}")
        gamma_str = (f"  gamma={self._gamma_:.4g}" if self.kernel != 'linear' else '')
        print(f"  Kernel         : {self.kernel}{gamma_str}"
              + (f"  degree={self.degree}" if self.kernel == 'poly' else ''))
        print(f"  C={self.C}")
        print(f"  Support vectors: {self.n_support_} / {len(self._y_train_obs_)}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula         : {label}")
        elif self.copula_used_:
            print(f"  Copula         : yes")
        print(f"  Train accuracy : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        label = ('weight |w_j|' if self.kernel == 'linear'
                 else 'availability')
        print(f"  Feature importances ({label}, normalized):")
        order = np.argsort(imp)[::-1]
        for j in order:
            name = self.feature_names_in_[j]
            bar  = self._bar(imp[j])
            print(f"  {name:>14}: {bar}  {imp[j] * 100:5.1f}%  "
                  f"(avail {avail[j] * 100:.0f}%)")
        print('=' * W)
        print()


# ======================================================================
# MissSupport  --  unified auto-selecting wrapper
# ======================================================================

class MissSupport(MissTags, BaseEstimator):
    """
    Unified SVM that automatically selects regression or classification
    based on the observed values of y.

    Detection rule (applied at fit time):
        All observed y in {0, 1}  ->  MissSupportClassifier
        Otherwise                 ->  MissSupportRegressor

    After fitting, ``model_`` holds the underlying model and ``task_``
    holds 'regression' or 'classification'.

    Parameters
    ----------
    C : float
        Regularization strength (default 1.0).
    epsilon : float
        SVR insensitive tube width -- regression only (default 0.1).
    kernel : {'rbf', 'linear', 'poly'}
        Available-case kernel (default 'rbf').
    gamma : float or {'scale', 'auto'}
        Kernel coefficient for rbf / poly (default 'scale').
    degree : int
        Polynomial kernel degree (default 3).
    coef0 : float
        Polynomial independent term (default 0.0).
    copula : bool or 'auto', default 'auto'
        Rank-normal feature transform (default False).
    """


    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale',
                 degree=3, coef0=0.0, copula='auto'):
        self.C       = C
        self.epsilon = epsilon
        self.kernel  = kernel
        self.gamma   = gamma
        self.degree  = degree
        self.coef0   = coef0
        self.copula  = copula

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
        Detect task and fit the appropriate SVM model.

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
        kw = dict(C=self.C, kernel=self.kernel, gamma=self.gamma,
                  degree=self.degree, coef0=self.coef0, copula=self.copula)
        if self.task_ == 'classification':
            self.model_ = MissSupportClassifier(**kw)
        else:
            self.model_ = MissSupportRegressor(epsilon=self.epsilon, **kw)
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
        """Signed SVM scores.  Only available for classification."""
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
