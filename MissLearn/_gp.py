"""
MissLearn._gp
-------------
Gaussian Process models with native missing-data support.

Marginalized kernel (Bayesian treatment of missing features)
-------------------------------------------------------------
The kernel between x and x' is computed by integrating out missing feature
values, assuming they are drawn from the empirical marginal distribution
N(0, 1) in the standardized space (after mean/std scaling of each feature).

For each feature j, the kernel contribution K_j(x_j, x'_j) is:

    Both observed   : K_j(x_j, x'_j) = k_1d(|z_j − z'_j| / ls_j)
    One missing     : K_j = E_{z_j ~ N(0,1)}[k_1d(|z_j − z'_j| / ls_j)]
    Both missing    : K_j = E_{z,z' ~ N(0,1)}[k_1d(|z − z'| / ls_j)]

    (where z = (x − μ)/σ is the standardized value)

These one-dimensional expectations are evaluated using Gauss-Hermite
quadrature (Q = 20 nodes, machine-precision accurate for smooth integrands).
The resulting 1D expectations have closed forms for the RBF kernel:

    One missing     : K_j = ls / sqrt(ls²+1) * exp(-½ z_obs² / (ls²+1))
    Both missing    : K_j = ls / sqrt(ls²+2)

The full kernel is the product of the per-feature contributions:

    K_marg(x, x') = σ_f² ∏_j K_j(x_j, x'_j)

This product kernel is a valid Mercer kernel (PSD by construction as an
expectation of a product of PSD kernels), solving the positive-definiteness
issue of the raw available-case kernel.

Predictive uncertainty for missing test features
------------------------------------------------
When test-point features are missing, the kernel K(x*, x_train) is
smaller (the GH marginal is below the obs-obs value), increasing the
posterior variance.  For a fully-missing test point the posterior
collapses toward the prior:

    K(x*_fully_missing, x_train_i) ≈ small → posterior var ≈ σ_f²

This gives the correct behaviour: maximum uncertainty for points with no
observed features, decreasing as more features are available.

Automatic Relevance Determination (ARD)
-----------------------------------------
With ard=True each feature gets its own length scale ls_j.  A large ls_j
means the model is insensitive to feature j; a small ls_j means the feature
is informative.  Normalized 1/ls_j gives a principled importance ranking.

Hyperparameter optimisation
-----------------------------
Log marginal likelihood (LML) is maximised via L-BFGS-B using analytical
gradients (fast, exact).  n_restarts restarts are drawn from a fixed-seed
RNG so results are reproducible regardless of the global numpy state.

Reproducibility
---------------
Fully deterministic.  L-BFGS-B is deterministic.  Random restarts use
rng(0) internally so they are independent of global random state.  Laplace
Newton iteration is deterministic.

Performance
-----------
Kernel computation is vectorised over the n² pairs; per-feature GH sums
are O(Q=20) and fast.  For n > ~2 000, GP is inherently O(n³) --
consider sparse/inducing-point approximations for large datasets.
"""

import warnings
import numpy as np
from ._utils import feature_scale
from scipy.optimize import minimize
from scipy.stats import norm as _scipy_norm
from scipy.special import expit as _sigmoid
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._conformance import check_common_parameters
from ._base import MissBase, MissTags, only_for
from ._copula import RankNormalTransformer, needs_copula


# ======================================================================
# Gauss-Hermite nodes/weights (module-level cache, computed once)
# Nodes t_i and weights w_i satisfy: ∫ exp(-t²) f(t) dt ≈ Σ w_i f(t_i)
# ======================================================================
_GH_Q = 20
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(_GH_Q)
_INV_SQRTPI = 1.0 / np.sqrt(np.pi)

# For miss-miss integration: d = z-z' ~ N(0,2) → d = 2t
_GH_T_2 = 2.0 * _GH_T     # nodes for miss-miss case
# For obs-miss / miss-obs: z_miss = √2 · t
_GH_T_SQRT2 = np.sqrt(2.0) * _GH_T  # nodes for obs-miss / miss-obs case


# ======================================================================
# 1D kernel primitives
# ======================================================================

def _k1d(r, kernel):
    """
    1D kernel value k(r) at (vectorised) normalised distance r = |z-z'| / ls.

    Returns array of same shape as r.
    """
    if kernel == 'rbf':
        return np.exp(-0.5 * r * r)
    elif kernel == 'matern52':
        s = np.sqrt(5.0) * r
        return (1.0 + s + s * s / 3.0) * np.exp(-s)
    elif kernel == 'matern12':
        return np.exp(-r)
    raise ValueError(f"kernel must be 'rbf', 'matern52', or 'matern12'; got {kernel!r}")


def _dk1d_dlogs(r, k_val, kernel):
    """
    ∂k_1d(r) / ∂(log ls) = -r · k'(r)  (vectorised).

    Derivation: r = |z-z'|/ls, so ∂r/∂(log ls) = -r.
    Chain rule: ∂k/∂(log ls) = k'(r) · (-r).
    """
    if kernel == 'rbf':
        # k(r) = exp(-r²/2), k'(r) = -r·exp(-r²/2)
        # -r·k'(r) = r²·exp(-r²/2) = r²·k_val
        return r * r * k_val
    elif kernel == 'matern52':
        # k(r) = (1+√5r+5r²/3)exp(-√5r)
        # ∂k/∂(log ls) = (5/3)r²(1+√5r)exp(-√5r)
        s = np.sqrt(5.0) * r
        return (5.0 / 3.0) * r * r * (1.0 + s) * np.exp(-s)
    elif kernel == 'matern12':
        # k(r) = exp(-r), ∂k/∂(log ls) = r·exp(-r) = r·k_val
        return r * k_val
    raise ValueError(f"Unknown kernel: {kernel!r}")


# ======================================================================
# Marginalized kernel construction
# ======================================================================

def _marg_kernel(Z_a, Z_b, ls_arr, sf, kernel, compute_grad=False):
    """
    Compute the marginalized product kernel K_marg(Z_a, Z_b) and optionally
    the per-feature log-derivatives ∂ log K_marg / ∂(log ls_j).

    Parameters
    ----------
    Z_a, Z_b : ndarray, shapes (n_a, p) and (n_b, p). NaN = missing.
    ls_arr   : ndarray (p,) -- per-feature length scales.
    sf       : float -- signal std (σ_f).
    kernel   : str -- 'rbf', 'matern52', or 'matern12'.
    compute_grad : bool -- if True, also compute log-derivatives.

    Returns
    -------
    K    : ndarray (n_a, n_b) -- K_marg(Z_a, Z_b).
    logd : list of p (n_a, n_b) arrays -- ∂ log K_j / ∂(log ls_j), or []
           if compute_grad is False.
    """
    n_a, p = Z_a.shape
    n_b = Z_b.shape[0]

    obs_a = ~np.isnan(Z_a)              # (n_a, p)  True = observed
    obs_b = ~np.isnan(Z_b)              # (n_b, p)
    Zaf = np.where(obs_a, Z_a, 0.0)    # NaN → 0 for arithmetic
    Zbf = np.where(obs_b, Z_b, 0.0)

    log_K = np.zeros((n_a, n_b))       # accumulate log product
    logd  = []

    for j in range(p):
        lsj  = ls_arr[j]
        zaj  = Zaf[:, j]               # (n_a,)
        zbj  = Zbf[:, j]               # (n_b,)
        oa   = obs_a[:, j]             # (n_a,) bool
        ob   = obs_b[:, j]             # (n_b,) bool

        # Missingness-pattern masks for each pair (i, k)
        oo = oa[:, None] & ob[None, :]   # (n_a, n_b)
        om = oa[:, None] & ~ob[None, :]
        mo = ~oa[:, None] & ob[None, :]
        # mm = everything else (both missing)

        # ---- obs-obs ----
        r_oo = np.abs(zaj[:, None] - zbj[None, :]) / lsj   # (n_a, n_b)
        k_oo = _k1d(r_oo, kernel)                           # (n_a, n_b)

        # ---- obs-miss: z_a observed, z_b missing ~ N(0,1) ----
        # K_om[i] = (1/√π) Σ_m w_m k(|z_ai - √2 t_m| / ls_j)
        r_om_gh = np.abs(zaj[:, None] - _GH_T_SQRT2[None, :]) / lsj   # (n_a, Q)
        k_om_gh = _k1d(r_om_gh, kernel)                                # (n_a, Q)
        k_om    = _INV_SQRTPI * (k_om_gh * _GH_W[None, :]).sum(axis=1) # (n_a,)

        # ---- miss-obs: z_a missing ~ N(0,1), z_b observed ----
        r_mo_gh = np.abs(_GH_T_SQRT2[:, None] - zbj[None, :]) / lsj   # (Q, n_b)
        k_mo_gh = _k1d(r_mo_gh, kernel)                                # (Q, n_b)
        k_mo    = _INV_SQRTPI * (k_mo_gh * _GH_W[:, None]).sum(axis=0) # (n_b,)

        # ---- miss-miss: z_a, z_b both ~ N(0,1), d = z_a-z_b ~ N(0,2) ----
        # K_mm = (1/√π) Σ_m w_m k(|2 t_m| / ls_j)
        r_mm_gh = np.abs(_GH_T_2) / lsj    # (Q,)
        k_mm_gh = _k1d(r_mm_gh, kernel)    # (Q,)
        k_mm    = float(_INV_SQRTPI * (k_mm_gh * _GH_W).sum())  # scalar

        # Combine into K_j (n_a, n_b), then add log to accumulator
        K_j = np.where(oo, k_oo,
              np.where(om, k_om[:, None],
              np.where(mo, k_mo[None, :],
              k_mm)))
        # Guard against zero (shouldn't happen for well-scaled data)
        log_K += np.log(np.maximum(K_j, 1e-300))

        if compute_grad:
            # ∂ log K_j / ∂(log ls_j) = (∂K_j_marg / ∂(log ls_j)) / K_j_marg

            # obs-obs: dk/d(log ls) / k = log-derivative of k w.r.t. log ls
            safe_k_oo = np.where(k_oo > 1e-300, k_oo, 1.0)
            gd_oo = _dk1d_dlogs(r_oo, k_oo, kernel) / safe_k_oo

            # obs-miss: (1/√π) Σ_m w_m ∂k_1d(r_m)/∂(log ls) / K_om
            dk_om = _INV_SQRTPI * (
                _dk1d_dlogs(r_om_gh, k_om_gh, kernel) * _GH_W[None, :]
            ).sum(axis=1)  # (n_a,)
            safe_k_om = np.where(k_om > 1e-300, k_om, 1.0)
            gd_om = np.where(k_om > 1e-300, dk_om / safe_k_om, 0.0)

            # miss-obs
            dk_mo = _INV_SQRTPI * (
                _dk1d_dlogs(r_mo_gh, k_mo_gh, kernel) * _GH_W[:, None]
            ).sum(axis=0)  # (n_b,)
            safe_k_mo = np.where(k_mo > 1e-300, k_mo, 1.0)
            gd_mo = np.where(k_mo > 1e-300, dk_mo / safe_k_mo, 0.0)

            # miss-miss
            dk_mm = float(_INV_SQRTPI * (_dk1d_dlogs(r_mm_gh, k_mm_gh, kernel) * _GH_W).sum())
            gd_mm = dk_mm / k_mm if k_mm > 1e-300 else 0.0

            G_j = np.where(oo, gd_oo,
                  np.where(om, gd_om[:, None],
                  np.where(mo, gd_mo[None, :],
                  gd_mm)))
            logd.append(G_j)

    K = sf * sf * np.exp(log_K)
    return K, logd


# ======================================================================
# Shared base
# ======================================================================

class _MissGaussianBase(MissBase):
    """Shared utilities: standardisation, kernel build, hyperparameter opt."""

    def _build_K(self, Z_a, Z_b, compute_grad=False):
        """Return (K_marg, logd) from the marginalized kernel."""
        return _marg_kernel(Z_a, Z_b, self._ls_arr_, self._sf_,
                            self.kernel, compute_grad=compute_grad)

    def _set_hp_from_log(self, params):
        """Unpack log-space params into self._ls_arr_, _sf_, _sn_."""
        p = self.n_features_in_
        if self.ard:
            self._ls_arr_ = np.exp(params[:p])
            self._sf_     = float(np.exp(params[p]))
            self._sn_     = float(np.exp(params[p + 1]))
            self.length_scale_ = self._ls_arr_.copy()
        else:
            ls = float(np.exp(params[0]))
            self._ls_arr_ = np.full(p, ls)
            self._sf_     = float(np.exp(params[1]))
            self._sn_     = float(np.exp(params[2]))
            self.length_scale_ = ls
        self.signal_var_ = self._sf_ ** 2
        self.noise_var_  = self._sn_ ** 2

    def _n_params(self):
        """Number of log-space hyperparameters."""
        return (self.n_features_in_ + 2) if self.ard else 3

    def _chol_solve(self, L, rhs):
        """Solve K_y x = rhs given Cholesky L (K_y = L L^T)."""
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))

    def _safe_cholesky(self, K, base_jitter=1e-8):
        """
        Cholesky using eigenvalue-clipping as the primary strategy.

        Clips negative eigenvalues to a small positive floor, then adds a
        minimal diagonal jitter.  Faster than the escalating-jitter loop on
        ill-conditioned matrices and avoids try/except branching overhead.
        """
        K = (K + K.T) * 0.5
        # Fast path: try plain Cholesky first (works for well-conditioned K)
        try:
            return np.linalg.cholesky(K + base_jitter * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            pass
        # Eigenvalue-clip path: guaranteed PSD, no looping
        eigvals, eigvecs = np.linalg.eigh(K)
        min_eig = float(eigvals.min())
        jitter  = max(base_jitter, -min_eig + 1e-6) if min_eig < 1e-6 else base_jitter
        K_psd   = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T
        K_psd   = (K_psd + K_psd.T) * 0.5 + jitter * np.eye(K.shape[0])
        return np.linalg.cholesky(K_psd)

    # ------------------------------------------------------------------
    # Gradient of LML w.r.t. log hyperparameters
    # ------------------------------------------------------------------

    def _lml_gradient(self, W_eff, K_f, logd, sn, ard):
        """
        Compute gradient of log-marginal-likelihood w.r.t. log hyperparameters.

        ∂LML/∂(log ls_j)  = ½ tr(W_eff · ∂K_f/∂(log ls_j))
                           = ½ Σ_{ik} W_eff[i,k] · K_f[i,k] · logd_j[i,k]

        ∂LML/∂(log σ_f)   = tr(W_eff · K_f)
        ∂LML/∂(log σ_n)   = σ_n² · tr(W_eff)

        Parameters
        ----------
        W_eff : (n, n) -- αα^T − K_y^{-1} for regression; a·a^T − R for clf.
        K_f   : (n, n) -- marginalised signal kernel.
        logd  : list of p (n, n) arrays -- ∂ log K_j / ∂(log ls_j).
                For isotropic, the caller passes a list with one summed element.

        sn    : float -- noise std (0 for classifier).
        ard   : bool
        """
        p = self.n_features_in_
        WK = W_eff * K_f         # element-wise product (used for ls and sf)

        if ard:
            grad_ls = [0.5 * float(np.sum(WK * logd[j])) for j in range(p)]
        else:
            # Isotropic: logd has p elements; gradient = sum over features
            combined = logd[0]
            for j in range(1, p):
                combined = combined + logd[j]
            grad_ls = [0.5 * float(np.sum(WK * combined))]

        grad_sf = float(np.sum(WK))          # 0.5 * tr(W_eff * 2*K_f)
        grad_sn = sn * sn * float(np.trace(W_eff))
        return np.array(grad_ls + [grad_sf, grad_sn])

    # ------------------------------------------------------------------
    # Hyperparameter optimisation with multiple restarts
    # ------------------------------------------------------------------

    def _optimise(self, neg_lml_and_grad, n_params, y_std, has_noise=True):
        """
        Minimise neg_lml_and_grad using L-BFGS-B with n_restarts restarts.
        All restarts use rng(0) so results are independent of global state.
        Returns (best_params, best_lml).

        Parameter layout: [log_ls..., log_sf, log_sn] when has_noise, else
        [log_ls..., log_sf] (GP classifier).  The sigma bounds are shifted
        by log(y_std) so amplitude/noise scales track the response scale
        (the regressor centres y but does not rescale it internally).
        """
        rng = np.random.default_rng(0)     # fixed seed -- reproducible

        log_y_std = np.log(max(y_std, 1e-6))

        n_ls      = n_params - 2 if has_noise else n_params - 1
        bounds_ls = [(-5.0, 3.0)] * n_ls
        bounds_sf = (-3.0 + log_y_std, 3.0 + log_y_std)
        if has_noise:
            bounds = bounds_ls + [bounds_sf, (-8.0 + log_y_std, 1.0 + log_y_std)]
        else:
            bounds = bounds_ls + [bounds_sf]

        best_lml    = np.inf
        best_params = None

        for trial in range(1 + self.n_restarts):
            if trial == 0:
                x0 = np.zeros(n_params)
                if has_noise:
                    x0[-2] = log_y_std
                    x0[-1] = log_y_std - 2.0
                else:
                    x0[-1] = log_y_std
            else:
                x0 = rng.uniform(-3.0, 2.0, n_params)
                if has_noise:
                    x0[-2] = rng.uniform(log_y_std - 2.0, log_y_std + 2.0)
                    x0[-1] = rng.uniform(log_y_std - 6.0, log_y_std)
                else:
                    x0[-1] = rng.uniform(log_y_std - 2.0, log_y_std + 2.0)

            try:
                res = minimize(neg_lml_and_grad, x0, jac=True,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 1000, 'ftol': 1e-10,
                                        'gtol': 1e-6})
                if res.fun < best_lml:
                    best_lml    = res.fun
                    best_params = res.x.copy()
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
                pass

        # Whether any restart returned a usable optimum. Five of the
        # iterative estimators in this library expose converged_ and this one
        # did not, so a fit in which every restart failed came back looking
        # like any other, carrying hyperparameters nobody chose.
        self._optimiser_converged_ = best_params is not None

        if best_params is None:
            warnings.warn(
                "%s: every hyperparameter restart failed; falling back to "
                "default hyperparameters. The fit still predicts, but from a "
                "kernel that was not fitted to these data. converged_ is "
                "False and log_marginal_likelihood_ is -inf."
                % type(self).__name__,
                UserWarning, stacklevel=2,
            )
            best_params = np.zeros(n_params)
            best_params[-2 if has_noise else -1] = log_y_std
        return best_params, -best_lml

    @property
    def feature_importances_(self):
        """
        Normalised feature importances, sums to 1.

        ARD     : 1/ls_k per feature (small ls → sensitive to that feature).
        Isotropic: feature availability rate (features often missing cannot
                   influence kernel evaluations as strongly).
        """
        check_is_fitted(self)
        if self.ard:
            w = 1.0 / self._ls_arr_
        else:
            w = np.mean(~np.isnan(self._X_orig_), axis=0).astype(float)
        total = w.sum()
        return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


# ======================================================================
# MissGaussianRegressor
# ======================================================================

class MissGaussianRegressor(RegressorMixin, _MissGaussianBase):
    """
    Gaussian Process Regression with native missing-data support.

    The predictive distribution is the exact Bayesian posterior:
        f* | y, X, X*  ~  N(μ*, Σ*)

    where the marginalised kernel is used throughout.  Posterior variance
    increases naturally as features are missing -- no heuristic multiplier.

    Parameters
    ----------
    kernel : {'rbf', 'matern52', 'matern12'}
        Covariance kernel (default 'rbf').
    ard : bool
        Automatic Relevance Determination: fit one length scale per feature
        (default False).
    n_restarts : int
        Number of random restarts for hyperparameter optimisation (default 3).
    noise_var_init : float
        Starting guess for noise variance (default 0.1).
    copula : bool or 'auto', default 'auto'
        Rank-normal transform of X and y before fitting (default 'auto').

    Attributes
    ----------
    length_scale_ : float or ndarray (p,)
    signal_var_   : float
    noise_var_    : float
    log_marginal_likelihood_ : float
    feature_importances_ : ndarray (p,)
    """

    def __init__(self, kernel='rbf', ard=False, n_restarts=3,
                 noise_var_init=0.1, copula='auto'):
        self.kernel          = kernel
        self.ard             = ard
        self.n_restarts      = n_restarts
        self.noise_var_init  = noise_var_init
        self.copula          = copula

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Optimise kernel hyperparameters by maximising log marginal likelihood,
        then cache the Cholesky factor for O(n²) prediction.
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        self._store_fit_metadata(X, y)
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        # Copula
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            self._copula_y_ = RankNormalTransformer().fit(y.reshape(-1, 1))
            X = self._copula_X_.transform(X)
            y = self._copula_y_.transform(y.reshape(-1, 1))[:, 0]

        # Standardise X (missing handled by NaN-aware stats)
        self._mu_    = np.nanmean(X, axis=0)
        # feature_scale applies the shared absolute-and-relative rule:
        # a constant column, a column with under two observed values, and
        # a column whose spread is negligible beside the widest one all
        # get a divisor of 1.0 rather than being amplified.
        self._sigma_ = feature_scale(X)
        Z = (X - self._mu_) / self._sigma_

        # Drop rows with missing y
        obs_y = ~np.isnan(y)
        self._Z_train_ = Z[obs_y]
        self._y_train_ = y[obs_y]

        # Centre y.  The GP prior has mean zero, so an uncentred response makes
        # the posterior revert toward 0 (not toward mean(y)) away from the
        # training data, and forces the amplitude sigma_f to absorb the mean --
        # which it cannot, because bounds_sf in _optimise is set from the
        # response *scale*, not its location.  The mean is added back in
        # predict() and predict_interval().
        self._y_mean_  = float(self._y_train_.mean()) if self._y_train_.size else 0.0
        self._y_train_ = self._y_train_ - self._y_mean_

        n, p = self._Z_train_.shape
        n_hp  = self._n_params()
        y_std = float(np.std(self._y_train_)) if len(self._y_train_) > 1 else 1.0

        # Objective: neg LML + analytical gradient
        def _obj(params):
            self._set_hp_from_log(params)
            K_f, logd = self._build_K(self._Z_train_, self._Z_train_,
                                      compute_grad=True)
            sn  = self._sn_
            K_y = K_f + sn * sn * np.eye(n)
            K_y = (K_y + K_y.T) * 0.5

            try:
                L = np.linalg.cholesky(K_y)
            except np.linalg.LinAlgError:
                # Should rarely occur with the PSD marginalized kernel
                try:
                    L = self._safe_cholesky(K_f, base_jitter=sn * sn)
                except np.linalg.LinAlgError:
                    return 1e10, np.zeros(n_hp)

            alpha = self._chol_solve(L, self._y_train_)

            lml = (-0.5 * float(self._y_train_ @ alpha)
                   - np.sum(np.log(np.diag(L)))
                   - 0.5 * n * np.log(2.0 * np.pi))

            K_inv = self._chol_solve(L, np.eye(n))
            W_eff = np.outer(alpha, alpha) - K_inv
            grad  = self._lml_gradient(W_eff, K_f, logd, sn, self.ard)

            return -lml, -grad

        # Optimise
        params_opt, lml_opt = self._optimise(_obj, n_hp, y_std)
        self._set_hp_from_log(params_opt)
        self.log_marginal_likelihood_ = lml_opt
        self.converged_ = bool(getattr(self, '_optimiser_converged_', True))

        # Cache Cholesky and alpha for O(n²) predictions
        K_f, _ = self._build_K(self._Z_train_, self._Z_train_)
        K_y    = K_f + self._sn_ ** 2 * np.eye(n)
        self._L_train_     = self._safe_cholesky(K_y)
        self._alpha_train_ = self._chol_solve(self._L_train_, self._y_train_)

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _posterior(self, X_transformed):
        """
        Returns (mu_n, std_post_n, std_pred_n) in the model's internal space.
        """
        Z_test = (X_transformed - self._mu_) / self._sigma_
        K_st, _ = self._build_K(Z_test, self._Z_train_)  # (n_test, n_train)

        mu_n = K_st @ self._alpha_train_

        # V = L^{-1} K_st^T
        V        = np.linalg.solve(self._L_train_, K_st.T)
        k_ss     = np.full(Z_test.shape[0], self._sf_ ** 2)   # K(x*,x*)=σ_f²
        var_post = np.maximum(k_ss - np.sum(V * V, axis=0), 0.0)

        return mu_n, np.sqrt(var_post), np.sqrt(var_post + self._sn_ ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, X):
        """Posterior mean E[f* | y, X, X*]."""
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        X_k = self._copula_X_.transform(X) if self.copula_used_ else X
        mu_n, _, _ = self._posterior(X_k)
        # Undo the y-centring of fit() before any copula inverse transform
        mu_n = mu_n + self._y_mean_
        if self.copula_used_:
            return self._copula_y_.inverse_transform(mu_n.reshape(-1, 1))[:, 0]
        return mu_n

    def predict_std(self, X):
        """
        Posterior standard deviation σ*(x) -- model uncertainty only.

        Smallest at training points; grows with distance and missingness.
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        X_k = self._copula_X_.transform(X) if self.copula_used_ else X
        _, std_post, _ = self._posterior(X_k)
        return std_post

    def predict_interval(self, X, alpha=0.05):
        """
        Exact Bayesian predictive interval, including observation noise σ_n.

        half-width = z_{α/2} · sqrt(σ*²(x) + σ_n²)

        Returns
        -------
        lower, upper : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        X_k = self._copula_X_.transform(X) if self.copula_used_ else X
        mu_n, _, std_pred_n = self._posterior(X_k)

        z    = _scipy_norm.ppf(1.0 - alpha / 2.0)
        # Undo the y-centring of fit() before any copula inverse transform
        lo_n = self._y_mean_ + mu_n - z * std_pred_n
        hi_n = self._y_mean_ + mu_n + z * std_pred_n

        if self.copula_used_:
            lo = self._copula_y_.inverse_transform(lo_n.reshape(-1, 1))[:, 0]
            hi = self._copula_y_.inverse_transform(hi_n.reshape(-1, 1))[:, 0]
        else:
            lo, hi = lo_n, hi_n

        return lo, hi

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
        W = 72
        p = self.n_features_in_
        print()
        print('=' * W)
        print('MissGaussianRegressor  --  GP Regression  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations  : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features      : {p}")
        print(f"  Missing (X)   : {self.n_missing_X_} ({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)   : {self.n_missing_y_}")
        print(f"  Kernel        : {self.kernel}  (ARD: {self.ard})")
        if self.ard:
            ls_str = '  '.join(f'{v:.3g}' for v in self.length_scale_)
            print(f"  length_scale  : [{ls_str}]")
        else:
            print(f"  length_scale  : {self.length_scale_:.4g}")
        print(f"  signal_var    : {self.signal_var_:.4g}")
        print(f"  noise_var     : {self.noise_var_:.6f}")
        print(f"  LML           : {self.log_marginal_likelihood_:.4f}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula        : {label}")
        elif self.copula_used_:
            print(f"  Copula        : yes")
        print(f"  Train R²      : {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        imp   = self.feature_importances_
        order = np.argsort(imp)[::-1]
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        label = ('1/ls (ARD)' if self.ard else 'availability')
        print(f"  Feature importances ({label}, normalized):")
        for j in order:
            name = self.feature_names_in_[j]
            bar  = self._bar(imp[j])
            ls_j = (self._ls_arr_[j] if self.ard else self.length_scale_)
            print(f"  {name:>14}: {bar}  {imp[j] * 100:5.1f}%  "
                  f"(ls={ls_j:.3g},  avail {avail[j] * 100:.0f}%)")
        print('=' * W)
        print()


# ======================================================================
# MissGaussianClassifier
# ======================================================================

class MissGaussianClassifier(ClassifierMixin, _MissGaussianBase):
    """
    Gaussian Process Classification with native missing-data support.

    Uses a Laplace approximation to the posterior:
        p(f | y) ≈ N(f̂, (W + K^{-1})^{-1})

    where f̂ is the mode and W = diag(σ(f̂)(1-σ(f̂))).

    Predictive probability uses the probit approximation:
        p(y*=1 | x*) ≈ σ(μ* / sqrt(1 + π/8 · σ*²))

    Parameters
    ----------
    kernel : {'rbf', 'matern52', 'matern12'}
    ard : bool
    n_restarts : int
    max_iter_newton : int
    copula : bool or 'auto', default 'auto'
    """

    def __init__(self, kernel='rbf', ard=False, n_restarts=3,
                 max_iter_newton=50, copula='auto'):
        self.kernel          = kernel
        self.ard             = ard
        self.n_restarts      = n_restarts
        self.max_iter_newton = max_iter_newton
        self.copula          = copula

    # ------------------------------------------------------------------
    # Laplace mode finding (Rasmussen & Williams Algorithm 3.1)
    # ------------------------------------------------------------------

    def _laplace_mode(self, K, y):
        """
        Find mode f̂ of log p(f|y) = log p(y|f) − ½ f^T K^{-1} f.

        Returns: f_hat, W_vec, L_W, a
        """
        n = len(y)
        f = np.zeros(n)
        laplace_converged = False

        for _ in range(self.max_iter_newton):
            pi     = _sigmoid(f)
            W_vec  = np.maximum(pi * (1.0 - pi), 1e-10)
            W_sqrt = np.sqrt(W_vec)
            grad   = y - pi

            B = np.eye(n) + (W_sqrt[:, None] * K) * W_sqrt[None, :]
            try:
                L_W = self._safe_cholesky(B)
            except np.linalg.LinAlgError:
                # The mode was not reached. What follows still returns
                # quantities, and they are not the mode's.
                laplace_converged = False
                break

            b   = W_vec * f + grad
            Kb  = K @ b
            v   = np.linalg.solve(L_W, W_sqrt * Kb)
            a   = b - W_sqrt * np.linalg.solve(L_W.T, v)
            f_new = K @ a

            if np.max(np.abs(f_new - f)) < 1e-6 * (1.0 + np.max(np.abs(f))):
                f = f_new
                laplace_converged = True
                break
            f = f_new

        # Final quantities at the point reached, which is the mode only if
        # the loop above converged; _laplace_converged_ records which.
        self._laplace_converged_ = bool(laplace_converged)
        pi     = _sigmoid(f)
        W_vec  = np.maximum(pi * (1.0 - pi), 1e-10)
        W_sqrt = np.sqrt(W_vec)
        B      = np.eye(n) + (W_sqrt[:, None] * K) * W_sqrt[None, :]
        L_W    = self._safe_cholesky(B)
        b      = W_vec * f + (y - pi)
        Kb     = K @ b
        v      = np.linalg.solve(L_W, W_sqrt * Kb)
        a      = b - W_sqrt * np.linalg.solve(L_W.T, v)

        return f, W_vec, L_W, a

    # ------------------------------------------------------------------
    # Laplace log marginal likelihood and gradient
    # ------------------------------------------------------------------

    def _laplace_lml_and_grad(self, params, Z, y):
        """
        Laplace approximate log marginal likelihood and analytical gradient.
        """
        n, p = Z.shape
        self._set_hp_from_log(params)

        K_f, logd = self._build_K(Z, Z, compute_grad=True)
        K = K_f + 1e-8 * np.eye(n)    # minimal jitter (K_f is PSD)

        f_hat, W_vec, L_W, a = self._laplace_mode(K, y)

        log_lik = float(y @ f_hat - np.sum(np.logaddexp(0.0, f_hat)))
        lml = (log_lik
               - 0.5 * float(a @ f_hat)
               - np.sum(np.log(np.diag(L_W))))

        # Gradient: W_eff = a a^T − (K + W^{-1})^{-1} = a a^T − W^{½} B^{-1} W^{½}
        W_sqrt = np.sqrt(W_vec)
        B_inv  = np.linalg.solve(L_W.T, np.linalg.solve(L_W, np.eye(n)))
        R      = W_sqrt[:, None] * B_inv * W_sqrt[None, :]   # (K+W^{-1})^{-1}
        W_eff  = np.outer(a, a) - R

        # Explicit part: ½ tr(W_eff ∂K/∂θ).  GP classifier has no σ_n;
        # compute gradient then drop the log_sn term.
        grad_full = self._lml_gradient(W_eff, K_f, logd, 0.0, self.ard)
        grad = grad_full[:-1]    # remove the log_sn element

        # Implicit part (R&W Algorithm 5.1): the mode f̂ moves with θ
        # through −½ log|B|.  s2_i = −½ [K − K R K]_ii · ∂³ log p(y_i|f_i)/∂f_i³,
        # correction per θ_j is s2ᵀ s3_j with s3_j = (I − K R) ∂K_j (y − π̂).
        pi_hat = _sigmoid(f_hat)
        d3     = -pi_hat * (1.0 - pi_hat) * (1.0 - 2.0 * pi_hat)
        KR     = K @ R
        post_diag = np.diag(K) - np.einsum('ij,ji->i', KR, K)
        s2     = -0.5 * post_diag * d3
        resid  = y - pi_hat

        def _implicit(dK):
            b = dK @ resid
            return float(s2 @ (b - KR @ b))

        if self.ard:
            corr = [_implicit(K_f * logd[j]) for j in range(p)]
        else:
            combined = logd[0]
            for j in range(1, p):
                combined = combined + logd[j]
            corr = [_implicit(K_f * combined)]
        corr.append(_implicit(2.0 * K_f))    # ∂K/∂(log σ_f) = 2 K_f
        grad = grad + np.array(corr)

        return -lml, -grad

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Find optimal kernel hyperparameters via Laplace LML, then cache
        the Laplace posterior for prediction.
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        self._store_fit_metadata(X, y)
        self._X_orig_ = X.copy()
        self._y_orig_ = y.copy()

        y_obs = y[~np.isnan(y)]
        _classes = np.sort(np.unique(y_obs))
        # Int cast only for integer-valued labels; truncating fractional
        # labels would break the classes_[1] binarization below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissGaussianClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )

        # Binarize against classes_[1]: the Bernoulli likelihood in
        # _laplace_mode assumes y ∈ {0, 1}.  NaN (missing y) preserved.
        y = np.where(np.isnan(y), np.nan,
                     (y == self.classes_[1]).astype(float))
        y_obs = y[~np.isnan(y)]
        n_pos = float(np.sum(y_obs == 1))
        self.class_prior_ = np.array([1.0 - n_pos / len(y_obs),
                                      n_pos / len(y_obs)])

        # Copula on X only
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        self._mu_    = np.nanmean(X, axis=0)
        # feature_scale applies the shared absolute-and-relative rule:
        # a constant column, a column with under two observed values, and
        # a column whose spread is negligible beside the widest one all
        # get a divisor of 1.0 rather than being amplified.
        self._sigma_ = feature_scale(X)
        Z = (X - self._mu_) / self._sigma_

        obs_y = ~np.isnan(y)
        self._Z_train_ = Z[obs_y]
        self._y_train_ = y[obs_y]

        n, p = self._Z_train_.shape
        # GP classifier has no σ_n: params = [log_ls..., log_sf]
        n_hp = (p + 1) if self.ard else 2

        def _obj(params):
            # Append dummy log_sn=0 so _set_hp_from_log works correctly
            params_full = np.append(params, 0.0)
            neg_lml, neg_grad = self._laplace_lml_and_grad(
                params_full, self._Z_train_, self._y_train_
            )
            return neg_lml, neg_grad   # neg_grad already trimmed in _laplace_lml_and_grad

        best_params, lml_opt = self._optimise(_obj, n_hp, 1.0, has_noise=False)
        params_full = np.append(best_params, 0.0)
        self._set_hp_from_log(params_full)
        self.log_marginal_likelihood_ = lml_opt
        self.converged_ = bool(
            getattr(self, '_optimiser_converged_', True)
            and getattr(self, '_laplace_converged_', True))
        self.noise_var_ = 0.0    # no noise term in GP classifier

        # Cache Laplace posterior
        K_f, _ = self._build_K(self._Z_train_, self._Z_train_)
        K  = K_f + 1e-8 * np.eye(n)
        self._K_train_ = K
        f_hat, W_vec, L_W, a = self._laplace_mode(K, self._y_train_)
        self._f_hat_ = f_hat
        self._W_vec_ = W_vec
        self._L_W_   = L_W
        self._a_     = a

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _latent_posterior(self, X_transformed):
        """Return (f_star_mean, f_star_var) for transformed test X."""
        Z_test = (X_transformed - self._mu_) / self._sigma_
        K_st, _ = self._build_K(Z_test, self._Z_train_)    # (n_test, n_train)

        f_mu   = K_st @ self._a_

        W_sqrt = np.sqrt(self._W_vec_)
        V      = np.linalg.solve(self._L_W_, W_sqrt[:, None] * K_st.T)
        k_ss   = np.full(Z_test.shape[0], self._sf_ ** 2)
        f_var  = np.maximum(k_ss - np.sum(V * V, axis=0), 0.0)

        return f_mu, f_var

    def predict_proba(self, X):
        """
        Class probabilities [P(Y=0|X_obs), P(Y=1|X_obs)].

        Uses the probit approximation:
            p(y*=1) ≈ σ(μ* / sqrt(1 + π/8 · σ*²))
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        X_k = self._copula_X_.transform(X) if self.copula_used_ else X
        f_mu, f_var = self._latent_posterior(X_k)

        kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8.0)
        p1    = _sigmoid(kappa * f_mu)
        p1    = np.clip(p1, 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Predict binary class labels (argmax of predict_proba)."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Latent GP posterior mean f*(x) -- signed score for class 1.
        Positive favours class 1, negative favours class 0.
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        X_k = self._copula_X_.transform(X) if self.copula_used_ else X
        f_mu, _ = self._latent_posterior(X_k)
        return f_mu

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
        p = self.n_features_in_
        print()
        print('=' * W)
        print('MissGaussianClassifier  --  GP Classification  (missing-data aware)'.center(W))
        print('=' * W)
        print(f"  Observations  : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features      : {p}")
        print(f"  Classes       : {self.classes_}")
        print(f"  Class priors  : P(Y={self.classes_[0]})={self.class_prior_[0]:.3f}  "
              f"P(Y={self.classes_[1]})={self.class_prior_[1]:.3f}")
        print(f"  Missing (X)   : {self.n_missing_X_} ({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)   : {self.n_missing_y_}")
        print(f"  Kernel        : {self.kernel}  (ARD: {self.ard})")
        if self.ard:
            ls_str = '  '.join(f'{v:.3g}' for v in self.length_scale_)
            print(f"  length_scale  : [{ls_str}]")
        else:
            print(f"  length_scale  : {self.length_scale_:.4g}")
        print(f"  signal_var    : {self.signal_var_:.4g}")
        print(f"  LML (approx)  : {self.log_marginal_likelihood_:.4f}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula        : {label}")
        elif self.copula_used_:
            print(f"  Copula        : yes")
        print(f"  Train accuracy: {self.score(self._X_orig_, self._y_orig_):.4f}")
        print('-' * W)
        imp   = self.feature_importances_
        order = np.argsort(imp)[::-1]
        avail = np.mean(~np.isnan(self._X_orig_), axis=0)
        label = ('1/ls (ARD)' if self.ard else 'availability')
        print(f"  Feature importances ({label}, normalized):")
        for j in order:
            name = self.feature_names_in_[j]
            bar  = self._bar(imp[j])
            ls_j = (self._ls_arr_[j] if self.ard else self.length_scale_)
            print(f"  {name:>14}: {bar}  {imp[j] * 100:5.1f}%  "
                  f"(ls={ls_j:.3g},  avail {avail[j] * 100:.0f}%)")
        print('=' * W)
        print()


# ======================================================================
# MissGaussian  --  unified auto-selecting wrapper
# ======================================================================

class MissGaussian(MissTags, BaseEstimator):
    """
    Unified Gaussian Process model that automatically selects regression or
    classification based on the observed values of y.

    Detection rule:
        All observed y in {0, 1}  →  MissGaussianClassifier
        Otherwise                 →  MissGaussianRegressor

    Parameters
    ----------
    kernel : {'rbf', 'matern52', 'matern12'}
    ard : bool
    n_restarts : int
    max_iter_newton : int
    noise_var_init : float
    copula : bool or 'auto', default 'auto'
    """


    def __init__(self, kernel='rbf', ard=False, n_restarts=3,
                 max_iter_newton=50, noise_var_init=0.1, copula='auto'):
        self.kernel          = kernel
        self.ard             = ard
        self.n_restarts      = n_restarts
        self.max_iter_newton = max_iter_newton
        self.noise_var_init  = noise_var_init
        self.copula          = copula

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
        """Detect task and fit the appropriate GP model."""
        # The dispatcher constructs a concrete estimator and hands
        # the work over, so any parameter it does not pass on is
        # never validated by anything. n_quadrature is the case
        # that exposed it: it belongs to the classifier, and on a
        # regression target it was simply dropped, leaving
        # n_quadrature=0 accepted here and refused by the sibling.
        check_common_parameters(self)
        self.task_ = self._detect_task(y)
        kw = dict(kernel=self.kernel, ard=self.ard,
                  n_restarts=self.n_restarts, copula=self.copula)
        if self.task_ == 'classification':
            self.model_ = MissGaussianClassifier(max_iter_newton=self.max_iter_newton, **kw)
        else:
            self.model_ = MissGaussianRegressor(noise_var_init=self.noise_var_init, **kw)
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
        """Posterior mean (regression) or class labels (classification)."""
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
        """Latent GP mean scores.  Only available for classification."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'classification':
            raise AttributeError(
                "decision_function is only available when task_ == 'classification'."
            )
        return self.model_.decision_function(X)

    @available_if(only_for('regression'))
    def predict_interval(self, X, alpha=0.05):
        """Bayesian predictive interval.  Only available for regression."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'regression':
            raise AttributeError(
                "predict_interval is only available when task_ == 'regression'."
            )
        return self.model_.predict_interval(X, alpha=alpha)

    @available_if(only_for('regression'))
    def predict_std(self, X):
        """Posterior standard deviation.  Only available for regression."""
        check_is_fitted(self, 'model_')
        if self.task_ != 'regression':
            raise AttributeError(
                "predict_std is only available when task_ == 'regression'."
            )
        return self.model_.predict_std(X)


    def score(self, X, y):
        """R² (regression) or accuracy (classification) on non-NaN targets."""
        check_is_fitted(self, 'model_')
        return self.model_.score(X, y)


    def summary(self):
        """Print the model summary."""
        check_is_fitted(self, 'model_')
        return self.model_.summary()
