"""
MissLearn._bayes
----------------
Gaussian generative (Bayes) models with native missing-data support.

Classification  (MissBayesClassifier, MissBayes + binary y)
------------------------------------------------------------------
Gaussian class-conditional model:
    P(Y=k | X_obs) ∝ P(Y=k) * N(X_obs ; μ_k[obs], Σ_k[obs, obs])

With structure='full' (default) each class carries a full covariance
matrix estimated from available cases (pairwise moments, shrunk toward
the diagonal), so correlated features are weighted correctly instead of
double-counted.  Missing features are marginalised out exactly; the
observed subvector of an MVN is itself MVN; correct under MCAR/MAR with
no imputation.  structure='naive' recovers classic Gaussian Naive Bayes
(diagonal Σ_k).

Regression  (MissBayesRegressor, MissBayes + continuous y)
------------------------------------------------------------------
Gaussian generative model:
    Y ~ N(μ_Y, σ²_Y)                     [prior on Y]
    X | Y ~ N(a + b·Y,  T)               [linear-Gaussian evidence]

With structure='full' (default) T is the full residual covariance of X
given Y (equivalently, (X, Y) is jointly MVN).  The posterior is Gaussian:
    1/σ²_post = 1/σ²_Y + b_obsᵀ T_obs⁻¹ b_obs
    μ_post    = σ²_post × ( μ_Y/σ²_Y + b_obsᵀ T_obs⁻¹ (X_obs − a_obs) )

structure='naive' restricts T to its diagonal (τ_j²), recovering the
classic naive-Bayes posterior; with correlated features that variant
double-counts evidence and is retained only for reference.

Parameters (b_j, a_j, T) are fitted from available-case pairwise moments -
no imputation.  Missing features are excluded from the posterior update.
All-missing rows fall back to the prior (μ_Y, σ_Y).

Reproducibility
---------------
All computations are closed-form (no optimization, no stochastic components).
Results are exactly reproducible across platforms.

Performance
-----------
Prediction groups query rows by missingness pattern; each pattern costs one
small linear solve (regressor) or one Cholesky per class (classifier), with
all row arithmetic vectorized.  Fit is O(n·p²).
"""

import numpy as np
from collections import defaultdict
from scipy.stats import norm as _scipy_norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase, MissTags, only_for
from ._conformance import check_common_parameters, check_choice, check_penalty
from ._copula import RankNormalTransformer, needs_copula
from ._utils import mvn_logpdf_batch


# ======================================================================
# Shared base
# ======================================================================

class _MissBayesBase(MissBase):
    """Structural base shared by regressor and classifier."""

    @staticmethod
    def _pairwise_cov(R, diag, min_pairs=3):
        """
        Available-case pairwise covariance matrix of the columns of R
        (NaN where unobserved), with the diagonal forced to *diag*.

        Off-diagonal entries with fewer than *min_pairs* jointly observed
        rows are set to 0 (no evidence of correlation).

        Returns
        -------
        C        : (p, p) covariance matrix
        min_pair : int, smallest jointly-observed count among estimated
                   pairs (len(R) when p == 1); drives auto-shrinkage.
        """
        p = R.shape[1]
        C = np.diag(np.asarray(diag, dtype=float))
        obs = ~np.isnan(R)
        min_pair = R.shape[0]
        for j in range(p):
            for k in range(j + 1, p):
                both = obs[:, j] & obs[:, k]
                n_jk = int(both.sum())
                min_pair = min(min_pair, n_jk)
                if n_jk >= min_pairs:
                    c = float(np.cov(R[both, j], R[both, k], ddof=1)[0, 1])
                    C[j, k] = C[k, j] = c
        return C, min_pair

    @staticmethod
    def _shrink_psd(C, lam):
        """
        Shrink C toward its diagonal by lam and repair to PSD by clipping
        eigenvalues at a small floor.  Keeps the diagonal unchanged.
        """
        d = np.diag(C).copy()
        C = (1.0 - lam) * C + lam * np.diag(d)
        # PSD repair: pairwise-complete covariances need not be PSD
        eigvals, eigvecs = np.linalg.eigh(C)
        floor = max(1e-12, 1e-8 * float(eigvals.max())) if eigvals.size else 1e-12
        if eigvals.min() < floor:
            C = (eigvecs * np.maximum(eigvals, floor)) @ eigvecs.T
            C = 0.5 * (C + C.T)
            # Restore the (well-estimated) per-feature variances
            sd_ratio = np.sqrt(d / np.maximum(np.diag(C), 1e-12))
            C = C * np.outer(sd_ratio, sd_ratio)
        return C

    @staticmethod
    def _auto_shrinkage(pair_counts_min, p):
        """
        Heuristic shrinkage intensity: more features / fewer jointly observed
        rows → stronger shrinkage toward the diagonal.
        """
        n_eff = max(float(pair_counts_min), 1.0)
        return float(min(0.9, max(0.05, p / (p + n_eff))))


# ======================================================================
# MissBayesRegressor
# ======================================================================

class MissBayesRegressor(RegressorMixin, _MissBayesBase):
    """
    Gaussian Naive Bayes regression with native missing-data support.

    Generative model:
        Y       ~ N(μ_Y, σ²_Y)
        X_j | Y ~ N(a_j + b_j·Y,  τ_j²)    independently for each j

    The posterior Y | X_obs is Gaussian (closed form):
        1/σ²_post = 1/σ²_Y  +  Σ_{j∈obs}  b_j² / τ_j²
        μ_post    = σ²_post × ( μ_Y/σ²_Y  +  Σ_{j∈obs}  b_j·(X_j − a_j)/τ_j² )

    Parameters b_j, a_j and the residual covariance T are estimated from
    available cases -- no imputation.  All-missing rows fall back to the
    prior (μ_Y, σ_Y) with the widest possible interval.

    Parameters
    ----------
    var_smoothing : float
        Added to the diagonal of T as var_smoothing × max_j Var(X_j),
        preventing exact-zero residual variance (sklearn-style
        regularization).  Default 1e-9.
    copula : bool or 'auto', default 'auto'
        If True, apply a rank-normal copula transform to X and y before fitting.
        Predictions are inverse-transformed back to the original y scale.
        'auto' triggers when any feature or y shows ``|skewness|``>1 or
        ``|excess kurtosis|``>2.
    structure : 'full' or 'naive'
        'full' (default) models the full residual covariance T of X given Y,
        so correlated features are weighted correctly.  'naive' restricts T
        to its diagonal; the classic naive-Bayes posterior, which
        double-counts evidence from correlated features.
    shrinkage : 'auto' or float in [0, 1]
        Intensity of shrinkage of T's off-diagonal toward 0 (structure='full'
        only).  'auto' scales with p and the available pairwise sample sizes.

    Attributes
    ----------
    mu_Y_, sigma2_Y_   : float  -- prior mean and variance of y
    slopes_            : ndarray (p,)  -- b_j coefficients
    intercepts_        : ndarray (p,)  -- a_j intercepts
    tau2_              : ndarray (p,)  -- residual variances τ_j² (diag of T)
    resid_cov_         : ndarray (p, p) -- residual covariance T
    """

    def __init__(self, var_smoothing=1e-9, copula='auto',
                 structure='full', shrinkage='auto'):
        self.var_smoothing = var_smoothing
        self.copula        = copula
        self.structure     = structure
        self.shrinkage     = shrinkage

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the Gaussian generative model from available (X_j, Y) pairs.

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
        # Added to the covariance diagonal, so a negative value breaks
        # positive-definiteness and every prediction comes back NaN.
        check_penalty(self.var_smoothing, 'var_smoothing',
                      type(self).__name__)
        # Tested only for 'full' below, so any misspelling of it, 'ful'
        # included, quietly selects naive independence instead.
        check_choice(self.structure, ('full', 'naive'), 'structure',
                     type(self).__name__)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        # Store originals for summary/score (predict handles copula internally)
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        # Copula transform of both X and y (regression: y is also modelled)
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            self._copula_y_ = RankNormalTransformer().fit(y.reshape(-1, 1))
            X = self._copula_X_.transform(X)
            y = self._copula_y_.transform(y.reshape(-1, 1))[:, 0]

        # Prior on Y
        y_obs = y[~np.isnan(y)]
        self.mu_Y_     = float(np.mean(y_obs)) if len(y_obs) > 0 else 0.0
        self.sigma2_Y_ = float(np.var(y_obs, ddof=1)) if len(y_obs) > 1 else 1.0
        self.sigma2_Y_ = max(self.sigma2_Y_, 1e-12)

        # Available-case OLS of X_j on Y for each feature
        slopes     = np.zeros(p)
        intercepts = np.zeros(p)
        tau2       = np.full(p, self.sigma2_Y_)   # sensible default

        for j in range(p):
            mask_j = ~np.isnan(X[:, j]) & ~np.isnan(y)
            if mask_j.sum() < 3:
                continue
            xj      = X[mask_j, j]
            yj      = y[mask_j]
            var_y_j = float(np.var(yj, ddof=1))
            if var_y_j < 1e-12:
                continue
            cov_xy    = float(np.cov(xj, yj, ddof=1)[0, 1])
            b_j       = cov_xy / var_y_j
            a_j       = float(np.mean(xj)) - b_j * float(np.mean(yj))
            resid     = xj - (a_j + b_j * yj)
            tau2_j    = float(np.var(resid, ddof=1)) if len(resid) > 1 else 1.0
            slopes[j]     = b_j
            intercepts[j] = a_j
            tau2[j]       = max(tau2_j, 1e-12)

        # Var-smoothing
        x_vars  = np.array([np.nanvar(X[:, j]) for j in range(p)])
        max_var = float(x_vars.max()) if x_vars.size > 0 else 1.0
        if max_var > 0:
            tau2 += self.var_smoothing * max_var

        self.slopes_     = slopes
        self.intercepts_ = intercepts
        self.tau2_       = tau2

        # Residual covariance T of X given Y.  With structure='full' the
        # off-diagonal is estimated from available-case residual pairs, so
        # the posterior weights correlated evidence correctly instead of
        # double-counting it (the naive-Bayes failure mode).
        if self.structure == 'full' and p > 1:
            resid = np.where(
                np.isnan(X) | np.isnan(y)[:, None],
                np.nan,
                X - (intercepts[None, :] + np.outer(y, slopes)),
            )
            T, min_pair = self._pairwise_cov(resid, tau2)
            lam = (self._auto_shrinkage(min_pair, p)
                   if self.shrinkage == 'auto' else float(self.shrinkage))
            T = self._shrink_psd(T, lam)
            self._shrinkage_used_ = lam
        else:
            T = np.diag(tau2)
            self._shrinkage_used_ = 1.0
        self.resid_cov_ = T

        # Precision contribution per feature: b_j^2 / tau_j^2
        self._prec_contrib_ = slopes ** 2 / tau2

        return self

    # ------------------------------------------------------------------ #
    # Prediction internals
    # ------------------------------------------------------------------ #

    def _predict_params(self, X):
        """
        Compute posterior (mu_post, sigma_post) for all query rows.

        Groups by missingness pattern; each pattern costs one small linear
        solve of T_obs (the residual covariance restricted to the observed
        features), with all row arithmetic vectorized.  With a diagonal T
        (structure='naive') this reduces exactly to the classic naive-Bayes
        posterior update.

        Returns (mu_post, sigma_post), shape (n,).
        """
        n = X.shape[0]
        mu_post    = np.full(n, self.mu_Y_)
        sigma_post = np.full(n, np.sqrt(self.sigma2_Y_))

        prec_prior = 1.0 / self.sigma2_Y_
        b = self.slopes_
        a = self.intercepts_
        T = self.resid_cov_

        groups = defaultdict(list)
        for i, x_i in enumerate(X):
            key = tuple(np.where(~np.isnan(x_i))[0])
            groups[key].append(i)

        for obs_key, idxs in groups.items():
            if not obs_key:   # all-missing: stays at prior
                continue
            obs_arr  = np.array(obs_key)
            idxs_arr = np.array(idxs)

            b_obs = b[obs_arr]
            T_obs = T[np.ix_(obs_arr, obs_arr)]
            try:
                w = np.linalg.solve(T_obs, b_obs)        # T_obs^{-1} b_obs
            except np.linalg.LinAlgError:
                # Singular T_obs: fall back to the diagonal (naive) update
                w = b_obs / np.maximum(np.diag(T_obs), 1e-12)

            prec_post = prec_prior + float(b_obs @ w)
            var_post  = 1.0 / max(prec_post, 1e-300)

            Z_grp = X[np.ix_(idxs_arr, obs_arr)] - a[obs_arr]   # (n_grp, m)
            mu_post[idxs_arr]    = var_post * (self.mu_Y_ * prec_prior + Z_grp @ w)
            sigma_post[idxs_arr] = np.sqrt(var_post)

        return mu_post, sigma_post

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, X):
        """
        Posterior mean E[Y | X_obs].

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
        mu, _ = self._predict_params(X)
        if self.copula_used_:
            mu = self._copula_y_.inverse_transform(mu.reshape(-1, 1))[:, 0]
        return mu

    def predict_interval(self, X, alpha=0.05):
        """
        Posterior credible interval [mu_post ± z * sigma_post].

        The interval widens automatically with missingness: complete rows have
        the tightest intervals (maximum posterior precision); all-missing rows
        fall back to the prior width.

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
        mu, sigma = self._predict_params(X)
        z     = _scipy_norm.ppf(1.0 - alpha / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        if self.copula_used_:
            lower = self._copula_y_.inverse_transform(lower.reshape(-1, 1))[:, 0]
            upper = self._copula_y_.inverse_transform(upper.reshape(-1, 1))[:, 0]
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

    @property
    def feature_importances_(self):
        """
        Normalized precision contribution b_j²/τ_j² per feature, sums to 1.

        Reflects each feature's contribution to the posterior precision gain
        over the prior.  Features with slope ≈ 0 or high residual variance
        contribute little and rank low.
        """
        check_is_fitted(self)
        w     = self._prec_contrib_.copy()
        total = w.sum()
        return w / total if total > 0 else np.ones(len(w)) / len(w)

    def summary(self):
        """Print a model summary."""
        check_is_fitted(self)
        W = 76
        p = self.n_features_in_
        print()
        print('=' * W)
        print('MissBayesRegressor  --  Gaussian NB Regression  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features     : {p}")
        print(f"  Missing (X)  : {self.n_missing_X_} ({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)  : {self.n_missing_y_}")
        print(f"  Prior mean   : {self.mu_Y_:.4f}     prior std: {np.sqrt(self.sigma2_Y_):.4f}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula       : {label}")
        elif self.copula_used_:
            print(f"  Copula       : yes  (X and y transformed to normal scale)")
        print(f"  Train R²     : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        imp   = self.feature_importances_
        order = np.argsort(imp)[::-1]
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        header = (f"  {'Feature':>14}  {'slope b_j':>10}  {'intercept':>10}  "
                  f"{'noise tau':>10}  {'importance':>10}  {'avail':>6}")
        print(header)
        print('  ' + '-' * (W - 2))
        for j in order:
            name = self.feature_names_in_[j]
            print(f"  {name:>14}  {self.slopes_[j]:>10.4f}  {self.intercepts_[j]:>10.4f}  "
                  f"{np.sqrt(self.tau2_[j]):>10.4f}  {imp[j] * 100:>9.1f}%  "
                  f"{avail[j] * 100:>5.1f}%")
        print('=' * W)
        print()


# ======================================================================
# MissBayesClassifier
# ======================================================================

class MissBayesClassifier(ClassifierMixin, _MissBayesBase):
    """
    Gaussian class-conditional classification with native missing-data support.

    P(Y=k | X_obs) ∝ P(Y=k) * N(X_obs ; μ_k[obs], Σ_k[obs, obs])

    Missing features are marginalised out exactly (the observed subvector of
    an MVN is itself MVN) -- correct under MCAR/MAR, no imputation.  With
    structure='full' (default) each class carries a full covariance matrix
    estimated from available cases and shrunk toward its diagonal, so
    correlated features are weighted correctly.  structure='naive' restricts
    Σ_k to its diagonal, recovering classic Gaussian Naive Bayes.  Features
    never observed with a particular class fall back to global feature
    statistics.

    Parameters
    ----------
    var_smoothing : float
        Added to diag(Σ_k) as var_smoothing × max_jk Var(X_j), preventing
        zero-variance degeneracy (sklearn GaussianNB convention). Default 1e-9.
    copula : bool or 'auto', default 'auto'
        If True, apply a rank-normal copula transform to X (not y) before
        fitting.  'auto' triggers when any feature shows ``|skewness|``>1 or
        ``|excess kurtosis|``>2.
    structure : 'full' or 'naive'
        'full' (default) estimates a full per-class covariance (QDA-style,
        shrunk toward the diagonal for stability); 'naive' keeps the
        classic diagonal (independent-features) model.
    shrinkage : 'auto' or float in [0, 1]
        Intensity of shrinkage of each Σ_k's off-diagonal toward 0
        (structure='full' only).  'auto' scales with p and the available
        per-class pairwise sample sizes.

    Attributes
    ----------
    classes_     : ndarray (2,) -- class labels
    class_prior_ : ndarray (2,) -- [P(Y=0), P(Y=1)]
    mu_jk_       : ndarray (p, 2) -- per-feature per-class means
    sigma2_jk_   : ndarray (p, 2) -- per-feature per-class variances
    cov_k_       : list of 2 (p, p) ndarrays -- per-class covariance matrices
    """

    def __init__(self, var_smoothing=1e-9, copula='auto',
                 structure='full', shrinkage='auto'):
        self.var_smoothing = var_smoothing
        self.copula        = copula
        self.structure     = structure
        self.shrinkage     = shrinkage

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Estimate per-class Gaussian parameters from available cases.

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
        # Added to the covariance diagonal, so a negative value breaks
        # positive-definiteness and every prediction comes back NaN.
        check_penalty(self.var_smoothing, 'var_smoothing',
                      type(self).__name__)
        # Tested only for 'full' below, so any misspelling of it, 'ful'
        # included, quietly selects naive independence instead.
        check_choice(self.structure, ('full', 'naive'), 'structure',
                     type(self).__name__)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        y_obs = y[~np.isnan(y)]
        _classes = np.sort(np.unique(y_obs))
        # Int cast only for integer-valued labels; truncating fractional
        # labels would break the classes_-based comparisons below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissBayesClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )

        # Copula on X only (y is discrete -- no transform)
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        # Class priors
        n_pos = float(np.sum(y_obs == self.classes_[1]))
        pi1   = n_pos / len(y_obs) if len(y_obs) > 0 else 0.5
        self.class_prior_ = np.array([1.0 - pi1, pi1])

        # Per-class per-feature Gaussian parameters
        mu_jk     = np.zeros((p, 2))
        sigma2_jk = np.ones((p, 2))

        for ki, kv in enumerate(self.classes_):
            kv_f = float(kv)
            for j in range(p):
                mask = (~np.isnan(X[:, j])) & (~np.isnan(y)) & (y == kv_f)
                n_jk = int(mask.sum())
                if n_jk >= 2:
                    mu_jk[j, ki]     = float(np.mean(X[mask, j]))
                    sigma2_jk[j, ki] = float(np.var(X[mask, j], ddof=1))
                else:
                    # Fall back to global feature statistics
                    gm = ~np.isnan(X[:, j])
                    if gm.sum() >= 2:
                        mu_jk[j, ki]     = float(np.mean(X[gm, j]))
                        sigma2_jk[j, ki] = float(np.var(X[gm, j], ddof=1))
                    # else: keep defaults (0, 1) -- fully unobserved feature

        # Var-smoothing (sklearn-style)
        x_vars  = np.array([np.nanvar(X[:, j]) for j in range(p)])
        max_var = float(x_vars.max()) if x_vars.size > 0 else 1.0
        if max_var > 0:
            sigma2_jk += self.var_smoothing * max_var

        self.mu_jk_     = mu_jk
        self.sigma2_jk_ = sigma2_jk

        # Per-class covariance matrices.  With structure='full' the
        # off-diagonal is estimated from available-case pairs within each
        # class (centred on that class's means) and shrunk toward the
        # diagonal, so correlated features are weighted correctly instead
        # of double-counted.
        cov_k = []
        for ki, kv in enumerate(self.classes_):
            if self.structure == 'full' and p > 1:
                in_class = (~np.isnan(y)) & (y == float(kv))
                R_k = np.where(np.isnan(X[in_class]), np.nan,
                               X[in_class] - mu_jk[:, ki][None, :])
                C_k, min_pair = self._pairwise_cov(R_k, sigma2_jk[:, ki])
                lam = (self._auto_shrinkage(min_pair, p)
                       if self.shrinkage == 'auto' else float(self.shrinkage))
                cov_k.append(self._shrink_psd(C_k, lam))
            else:
                cov_k.append(np.diag(sigma2_jk[:, ki]))
        self.cov_k_ = cov_k

        # Feature importance: Cohen's d effect size between classes.
        #
        # A feature with no within-class spread makes pooled_std zero and the
        # ratio 0/0 = nan. The default var_smoothing floors the variances so
        # this does not arise, but var_smoothing=0 is accepted and then a
        # single constant column produced one nan, which made the sum in
        # feature_importances_ nan, its ``total > 0`` test false, and every
        # feature came back at exactly 1/p. That is the worst possible
        # failure for this attribute: it sums to 1, contains no nan, and
        # reads as a finding, while the real answer on the fixture that found
        # it was [0.78, 0.05, 0, 0.18]. Predictions were unaffected, so
        # nothing else signalled it.
        gap        = np.abs(mu_jk[:, 1] - mu_jk[:, 0])
        pooled_std = np.sqrt((sigma2_jk[:, 0] + sigma2_jk[:, 1]) / 2.0)
        usable     = pooled_std > 0.0
        effect     = np.zeros(len(pooled_std), dtype=float)
        np.divide(gap, pooled_std, out=effect, where=usable)

        # A feature that does not vary within either class but whose class
        # means differ separates them perfectly, so it belongs above every
        # feature that does vary. Ranked by a finite margin rather than inf,
        # which would leave the normalisation meaningless.
        perfect = (~usable) & (gap > 0.0)
        if perfect.any():
            ceiling = float(np.max(effect[~perfect])) if (~perfect).any() else 0.0
            effect[perfect] = max(ceiling, 1.0) * 10.0

        effect[~np.isfinite(effect)] = 0.0
        self._effect_size_ = effect

        return self

    # ------------------------------------------------------------------ #
    # Prediction internals
    # ------------------------------------------------------------------ #

    def _log_posteriors(self, X):
        """
        Compute (unnormalized) log P(Y=k | X_obs) for all query rows.

        Groups by missingness pattern; the observed subvector of an MVN is
        itself MVN, so log P(X_obs | Y=k) = log N(x_obs; μ_k[obs], Σ_k[obs,obs])

       ; one batched MVN evaluation per pattern per class.  With diagonal
        Σ_k (structure='naive') this equals the classic per-feature product.

        Returns ndarray, shape (n, 2).
        """
        n         = X.shape[0]
        log_prior = np.log(self.class_prior_)          # (2,)
        log_post  = np.broadcast_to(log_prior, (n, 2)).copy()

        groups = defaultdict(list)
        for i, x_i in enumerate(X):
            key = tuple(np.where(~np.isnan(x_i))[0])
            groups[key].append(i)

        for obs_key, idxs in groups.items():
            if not obs_key:   # all-missing: prior only
                continue
            obs_arr  = np.array(obs_key)
            idxs_arr = np.array(idxs)
            X_grp    = X[np.ix_(idxs_arr, obs_arr)]    # (n_grp, m)

            for ki in range(2):
                mu_obs  = self.mu_jk_[obs_arr, ki]
                cov_obs = self.cov_k_[ki][np.ix_(obs_arr, obs_arr)]
                lps = mvn_logpdf_batch(X_grp, mu_obs, cov_obs)
                if not np.all(np.isfinite(lps)):
                    # Non-PD submatrix (mvn_logpdf_batch signals with -inf):
                    # fall back to the diagonal model for this pattern/class
                    var_obs = np.maximum(np.diag(cov_obs), 1e-12)
                    diff2   = (X_grp - mu_obs[None, :]) ** 2 / var_obs[None, :]
                    lps = (-0.5 * np.log(2.0 * np.pi * var_obs).sum()
                           - 0.5 * diff2.sum(axis=1))
                log_post[idxs_arr, ki] += lps

        return log_post

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_proba(self, X):
        """
        Class probabilities [P(Y=0|X_obs), P(Y=1|X_obs)].

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
        log_post = self._log_posteriors(X)
        # Numerically stable softmax via log-sum-exp
        log_Z    = log_post.max(axis=1, keepdims=True)
        probs    = np.exp(log_post - log_Z)
        probs   /= probs.sum(axis=1, keepdims=True)
        probs    = np.clip(probs, 1e-15, 1.0 - 1e-15)
        probs   /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """Predict binary class labels."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Log-odds: log P(Y=1|X_obs) / P(Y=0|X_obs).
        Equal to the log-posterior difference (log-likelihood ratio + log prior ratio).
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

    @property
    def feature_importances_(self):
        """
        Normalized Cohen's d effect sizes per feature, sums to 1.

        Cohen's d = ``|μ_j1 − μ_j0|`` / sqrt((σ²_j0 + σ²_j1)/2) measures
        how well class means are separated relative to within-class spread.
        """
        check_is_fitted(self)
        w     = self._effect_size_.copy()
        total = w.sum()
        return w / total if total > 0 else np.ones(len(w)) / len(w)

    def summary(self):
        """Print a model summary."""
        check_is_fitted(self)
        W = 80
        p = self.n_features_in_
        print()
        print('=' * W)
        print('MissBayesClassifier  --  Gaussian NB Classification  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features     : {p}")
        print(f"  Classes      : {self.classes_}")
        print(f"  Class priors : P(Y={self.classes_[0]})={self.class_prior_[0]:.3f}  "
              f"P(Y={self.classes_[1]})={self.class_prior_[1]:.3f}")
        print(f"  Missing (X)  : {self.n_missing_X_} ({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)  : {self.n_missing_y_}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula       : {label}")
        elif self.copula_used_:
            print(f"  Copula       : yes  (X transformed to normal scale)")
        print(f"  Train acc    : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        imp   = self.feature_importances_
        order = np.argsort(imp)[::-1]
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        c0, c1 = self.classes_[0], self.classes_[1]
        header = (f"  {'Feature':>14}  {'mu_'+str(c0):>8}  {'mu_'+str(c1):>8}  "
                  f"{'std_'+str(c0):>8}  {'std_'+str(c1):>8}  "
                  f"{'effect_d':>9}  {'import':>7}  {'avail':>6}")
        print(header)
        print('  ' + '-' * (W - 2))
        for j in order:
            name = self.feature_names_in_[j]
            print(f"  {name:>14}  {self.mu_jk_[j, 0]:>8.3f}  {self.mu_jk_[j, 1]:>8.3f}  "
                  f"{np.sqrt(self.sigma2_jk_[j, 0]):>8.3f}  {np.sqrt(self.sigma2_jk_[j, 1]):>8.3f}  "
                  f"{self._effect_size_[j]:>9.3f}  {imp[j] * 100:>6.1f}%  "
                  f"{avail[j] * 100:>5.1f}%")
        print('=' * W)
        print()


# ======================================================================
# MissBayes  --  unified auto-selecting wrapper
# ======================================================================

class MissBayes(MissTags, BaseEstimator):
    """
    Unified Naive Bayes model that automatically selects regression or
    classification based on the observed values of y.

    Detection rule (applied at fit time):
        All observed y in {0, 1}  ->  MissBayesClassifier
        Otherwise                 ->  MissBayesRegressor

    The underlying model is stored in ``model_`` and the task in ``task_``
    ('regression' or 'classification') after fitting.

    Parameters
    ----------
    var_smoothing : float
        Regularization for per-class (or per-feature) variance estimates.
        Default 1e-9.
    copula : bool or 'auto', default 'auto'
        Rank-normal feature transform (default False).
    structure : 'full' or 'naive'
        'full' (default) models the full covariance so correlated features
        are weighted correctly; 'naive' keeps the classic diagonal model.
    shrinkage : 'auto' or float in [0, 1]
        Off-diagonal shrinkage intensity (structure='full' only).
    """


    def __init__(self, var_smoothing=1e-9, copula='auto',
                 structure='full', shrinkage='auto'):
        self.var_smoothing = var_smoothing
        self.copula        = copula
        self.structure     = structure
        self.shrinkage     = shrinkage

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
        Detect task and fit the appropriate Naive Bayes model.

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
        kw = dict(var_smoothing=self.var_smoothing, copula=self.copula,
                  structure=self.structure, shrinkage=self.shrinkage)
        if self.task_ == 'classification':
            self.model_ = MissBayesClassifier(**kw)
        else:
            self.model_ = MissBayesRegressor(**kw)
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
        """Posterior credible interval.  Only available for regression."""
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
