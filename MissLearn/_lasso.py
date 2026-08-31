"""
MissLearn._lasso
----------------
LASSO-penalized FIML models with native missing data support.

Both models use variable splitting: write beta = u - v with u, v >= 0 so that
the non-differentiable L1 penalty alpha * ||beta||_1 becomes the linear,
differentiable term alpha * (sum(u) + sum(v)) under non-negativity bounds.
L-BFGS-B handles the bounds natively; no smoothing approximation is used.
At the optimizer's solution, complementarity (u_j * v_j = 0) holds and
u_j + v_j = |beta_j|, so the penalty equals the true L1 norm.

MissLASSORegressor
    Models Y | X ~ N(beta @ X + intercept, sigma^2) with X ~ N(mu_X, Sigma_X).
    Sparse beta via L1 penalty; identical FIML log-likelihood structure to
    MissRidgeRegressor.  Missing X handled via pattern-grouped conditional normals.

MissLASSOClassifier
    FIML logistic regression with L1 penalty on slope coefficients.
    Missing X marginalized via Gauss-Hermite quadrature (same as MissLogistic).

MissLASSO
    Unified wrapper: fits MissLASSOClassifier when y is binary {0, 1},
    MissLASSORegressor otherwise.  Determined at fit time.

Standard errors are not computed by default.  The L1 penalty creates a
non-differentiable objective whose Hessian is undefined at zero coefficients,
making standard asymptotic SEs theoretically invalid.  Set compute_se=True
to attempt SE estimation from the optimizer Hessian, but interpret with care:
SEs are only meaningful for coefficients that are clearly non-zero.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase, MissTags, only_for
from ._conformance import check_common_parameters, check_choice, check_penalty
from ._copula import RankNormalTransformer, needs_copula
from ._utils import (
    sigmoid, pack_cholesky, unpack_cholesky, mvn_logpdf_batch,
    conditional_normal_params, integrate_logistic_normal,
    logistic_normal_with_grads, numerical_hessian,
    standard_errors_from_variance,
    psd_jitter, feature_scale
)

_ZERO_THRESH = 1e-4   # |coef| below this is reported as zero in summary

from ._utils import prep_conditional_terms as _prep_conditional_terms


# ======================================================================
# MissLASSORegressor
# ======================================================================

class MissLASSORegressor(RegressorMixin, MissBase):
    """
    LASSO-penalized FIML linear regression with native missing data support.

    Models Y | X ~ N(beta @ X + intercept, sigma^2) and X ~ N(mu_X, Sigma_X).
    The L1 penalty on beta promotes sparse solutions.  Missing X is handled
    via closed-form marginalization over the conditional normal distribution
    of the missing features given the observed ones.

    Parameters
    ----------
    alpha : float
        LASSO regularization strength (default 1.0).  Larger values give
        stronger sparsity.  alpha=0 recovers unpenalized FIML regression.
    max_iter : int
        Maximum optimizer iterations (default 3000; LASSO often needs more
        iterations than Ridge due to the bound constraints).
    tol : float
        Convergence tolerance (default 1e-7).
    method : {'L-BFGS-B', 'TNC', 'Powell', 'SLSQP'}
        scipy.optimize.minimize method (default 'L-BFGS-B'). The L1 term is
        written by variable splitting, beta = u - v with u, v >= 0, so the
        penalty only equals ``|beta|`` while those bounds hold. Solvers that
        cannot handle bounds discard them and the objective runs away, so
        the choice is restricted to those that can.
    compute_se : bool
        Compute standard errors (default False).  SEs are theoretically
        ill-defined at zero coefficients under L1 regularization; only
        meaningful for coefficients that are clearly non-zero.
    copula : bool or 'auto', default 'auto'
        Gaussian copula transform. 'auto' (the default) applies it
        only when the marginals are skewed or heavy-tailed enough to
        warrant it; True forces it on and False off.

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
        Estimated regression coefficients (sparse at optimum).
    intercept_ : float
    n_nonzero_ : int
        Number of non-zero coefficients (``|coef_j|`` > 1e-4).
    sigma_sq_ : float
    mu_X_ : ndarray, shape (p,)
    Sigma_X_ : ndarray, shape (p, p)
    loglik_ : float
    aic_, bic_ : float
    copula_used_ : bool
    """

    def __init__(self, alpha=1.0, max_iter=3000, tol=1e-7, method='L-BFGS-B',
                 compute_se=False, copula='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.compute_se = compute_se
        self.copula = copula

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Pattern grouping
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Standard errors (optional, use with caution)
    # ------------------------------------------------------------------ #

    def _compute_se(self, theta_opt, nll_fn, hess_inv=None):
        """
        SEs from the (small) reduced-parameter Hessian.

        theta layout: [intercept, u (p), v (p), log_sigma]; the Hessian is
        mapped to beta space via d(beta_j)/d(u_j)=1, d(beta_j)/d(v_j)=-1.
        """
        p = self.n_features_in_
        n_theta = len(theta_opt)
        if hess_inv is not None:
            Var_theta = hess_inv
        else:
            try:
                H = numerical_hessian(nll_fn, theta_opt, eps=1e-4)
                H += np.eye(n_theta) * 1e-8
                Var_theta = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return np.full(p + 1, np.nan)
        J = np.zeros((p + 1, n_theta))
        J[0, 0] = 1.0                              # intercept
        for j in range(p):
            J[j + 1, 1 + j]     =  1.0            # d beta_j / d u_j
            J[j + 1, 1 + p + j] = -1.0            # d beta_j / d v_j
        Var_beta = J @ Var_theta @ J.T
        return standard_errors_from_variance(Var_beta)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the LASSO-penalized FIML regression model.

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
        check_penalty(self.alpha, 'alpha', 'MissLASSORegressor')
        # The L1 term is written by variable splitting, beta = u - v with
        # u, v >= 0, so the penalty sum(u + v) only equals |beta| while those
        # bounds hold. Solvers that cannot handle bounds discard them, the
        # penalty turns into a reward for large coefficients and the
        # objective runs away: CG reached an R-squared of -2.3e6 here.
        # Nelder-Mead respects the bounds but does not optimise this problem,
        # returning the training mean at an R-squared of exactly 0 while
        # reporting convergence.
        check_choice(self.method, ('L-BFGS-B', 'TNC', 'Powell', 'SLSQP'),
                     'method', 'MissLASSORegressor')
        self._store_fit_metadata(X, y)
        n, p = X.shape

        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)
            y_obs_mask = ~np.isnan(y)
            self._copula_y_ = RankNormalTransformer().fit(
                y[y_obs_mask].reshape(-1, 1)
            )
            y = y.copy()
            y[y_obs_mask] = self._copula_y_.transform(
                y[y_obs_mask].reshape(-1, 1)
            ).ravel()

        # Penalized likelihoods are only scale-coherent on standardized data
        # (glmnet convention): fit on standardized X and y, then convert all
        # public parameters back to the raw scale below (the FIML model is
        # affine-equivariant).  Without this, a large ||beta|| arising purely
        # from units makes the penalty dominate the likelihood and the
        # optimum collapses to beta ~ 0 with sigma ~ sd(y).
        _mx = np.nanmean(X, axis=0)
        _mx = np.where(np.isfinite(_mx), _mx, 0.0)
        _sx = feature_scale(X)
        _y_obs = ~np.isnan(y)
        _my = float(np.mean(y[_y_obs])) if _y_obs.any() else 0.0
        _sy = float(np.std(y[_y_obs], ddof=1)) if _y_obs.sum() > 1 else 1.0
        if not (np.isfinite(_sy) and _sy >= 1e-8):
            _sy = 1.0
        X = (X - _mx) / _sx
        y = np.where(np.isnan(y), np.nan, (y - _my) / _sy)

        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_cc = X[complete_mask]
        y_cc = y[complete_mask]

        mu_X0 = np.nanmean(X, axis=0)
        mu_X0 = np.where(np.isfinite(mu_X0), mu_X0, 0.0)

        # Prefer complete cases; fall back to a mean-imputed seed when there
        # are too few (FIML still fits on the real partial data below).
        if len(y_cc) >= p + 2:
            X_seed, y_seed, X_cov = X_cc, y_cc, X_cc
        else:
            obs_y  = ~np.isnan(y)
            X_imp  = np.where(np.isnan(X), mu_X0, X)
            X_seed, y_seed, X_cov = X_imp[obs_y], y[obs_y], X_imp

        from sklearn.linear_model import Lasso as _Lasso
        lasso0 = _Lasso(alpha=self.alpha, max_iter=5000)
        lasso0.fit(X_seed, y_seed)
        beta0     = lasso0.coef_
        intercept0 = float(lasso0.intercept_)
        sigma0    = max(np.std(y_seed - lasso0.predict(X_seed)), 1e-4)

        u0 = np.maximum(beta0, 0.0) + 1e-6
        v0 = np.maximum(-beta0, 0.0) + 1e-6

        # psd_jitter also promotes the 0-d array np.cov returns for a
        # single column, which is what made this estimator fail at p = 1
        # where MissLASSOClassifier coped.
        Sigma_X0 = psd_jitter(np.cov(X_cov, rowvar=False))

        # --- Stage 1: X-moments by EM (full-information, includes y) -----
        # Estimating the MVN nuisance parameters once removes p + p(p+1)/2
        # dimensions from the optimizer below, whose finite-difference
        # gradients otherwise dominate the runtime.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(np.column_stack([X, y]))
        mu_X_opt    = _fitter.mu_[:p]
        Sigma_X_opt = _fitter.Sigma_[:p, :p]

        # --- Stage 2: penalized conditional ML for (intercept, beta, sigma)
        # Pattern-constant terms are precomputed, so each NLL call is O(n p).
        obs_y_mask = ~np.isnan(y)
        y_y = y[obs_y_mask]
        F_y, cond_groups = _prep_conditional_terms(X[obs_y_mask], mu_X_opt,
                                                   Sigma_X_opt)

        # Vectorized group structures: group_id per row, group sizes, and the
        # (G, p, p) tensor of Sigma_c scattered to full-p coordinates, so the
        # NLL and its analytic gradient are pure array expressions (no Python
        # loop over patterns, no finite differences).
        n_y_rows = len(y_y)
        G = len(cond_groups)
        group_id = np.empty(n_y_rows, dtype=np.intp)
        n_g = np.empty(G)
        M = np.zeros((G, p, p))
        for g, (rows, mis, Sc) in enumerate(cond_groups):
            group_id[rows] = g
            n_g[g] = len(rows)
            if mis.size:
                M[g][np.ix_(mis, mis)] = Sc
        _LOG2PI = np.log(2.0 * np.pi)
        _bad_grad = np.zeros(2 * p + 2)

        def _nll_grad(theta_r):
            intercept = theta_r[0]
            u_r = theta_r[1:p + 1]
            v_r = theta_r[p + 1:2 * p + 1]
            beta = u_r - v_r
            sigma_sq = float(np.exp(2.0 * theta_r[2 * p + 1]))

            Mb  = M @ beta                                   # (G, p)
            v_g = sigma_sq + Mb @ beta                       # (G,)
            if not np.all(np.isfinite(v_g)) or np.any(v_g <= 0):
                return np.inf, _bad_grad
            resid = y_y - intercept - F_y @ beta
            rss_g = np.bincount(group_id, weights=resid * resid, minlength=G)

            nll = (0.5 * float(n_g @ (_LOG2PI + np.log(v_g)))
                   + 0.5 * float(np.sum(rss_g / v_g)))
            if self.alpha > 0:
                nll += self.alpha * float(np.sum(u_r) + np.sum(v_r))

            # Analytic gradient
            with np.errstate(over='ignore', invalid='ignore'):
                w_v = 0.5 * n_g / v_g - 0.5 * rss_g / v_g ** 2   # dNLL/dv_g
            # v_g passed the positivity check above but can still be small
            # enough that squaring it lands in the denormals, at which point
            # the division overflows to inf and the inf * 0 in g_beta below
            # writes nan into the gradient. SLSQP walks there. A nan gradient
            # is not a small numerical blemish, it is a direction the
            # optimiser cannot use, so treat it as the out-of-domain case the
            # positivity check already handles.
            if not np.all(np.isfinite(w_v)):
                return np.inf, _bad_grad
            r_w = resid / v_g[group_id]
            g_int  = -float(np.sum(r_w))
            g_beta = -(F_y.T @ r_w) + 2.0 * (w_v[:, None] * Mb).sum(axis=0)
            g_logs = float(np.sum(w_v)) * 2.0 * sigma_sq
            grad = np.concatenate([
                [g_int], g_beta + self.alpha, -g_beta + self.alpha, [g_logs]
            ])
            return nll, grad

        def _reduced_nll(theta_r):
            return _nll_grad(theta_r)[0]

        theta0 = np.concatenate([[intercept0], u0, v0, [np.log(sigma0)]])
        bounds = ([(None, None)] + [(0.0, None)] * (2 * p) + [(None, None)])

        # Powell is derivative-free. Handing it a gradient makes scipy warn
        # on every fit about a jac it will not use, so give it the value-only
        # form that _compute_se already relies on.
        if self.method == 'Powell':
            result = minimize(
                fun=_reduced_nll,
                x0=theta0,
                method=self.method,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'ftol': self.tol},
            )
        else:
            result = minimize(
                fun=_nll_grad,
                x0=theta0,
                jac=True,
                method=self.method,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'ftol': self.tol,
                         'gtol': self.tol * 1e-2},
            )

        theta_opt = result.x
        self._theta_opt_ = result.x
        intercept_opt = float(theta_opt[0])
        beta_opt      = theta_opt[1:p + 1] - theta_opt[p + 1:2 * p + 1]
        sigma_opt     = float(np.exp(theta_opt[2 * p + 1]))

        # Sparsity is judged on the standardized scale (unit-free threshold)
        self.n_nonzero_  = int(np.sum(np.abs(beta_opt) > _ZERO_THRESH))

        # Convert every public parameter back to the raw (unstandardized)
        # scale.  The FIML model is affine-equivariant, so predict() and
        # predict_interval() work unchanged on these raw-scale parameters.
        self.coef_       = beta_opt * (_sy / _sx)
        self.intercept_  = (_my + _sy * intercept_opt
                            - float(self.coef_ @ _mx))
        self.sigma_sq_   = (_sy ** 2) * sigma_opt ** 2
        self.mu_X_       = _mx + _sx * mu_X_opt
        self.Sigma_X_    = Sigma_X_opt * np.outer(_sx, _sx)

        # result.fun includes the L1 penalty; add it back so loglik_/AIC/BIC
        # reflect the data likelihood only (same convention as MissRidge).
        # The full-information log-likelihood is the conditional part (from
        # the optimizer) plus the X-marginal part (from the EM stage); the
        # Jacobian term moves it from the standardized to the raw scale.
        u_opt = theta_opt[1:p + 1]
        v_opt = theta_opt[p + 1:2 * p + 1]
        penalty = self.alpha * float(np.sum(u_opt) + np.sum(v_opt))
        _marg_ll   = _mvn_loglik(X, mu_X_opt, Sigma_X_opt)
        _n_obs_col = np.sum(~np.isnan(X), axis=0)
        _jacobian  = (-float(_n_obs_col @ np.log(_sx))
                      - float(_y_obs.sum()) * np.log(_sy))
        self.loglik_    = (-(float(result.fun) - penalty) + _marg_ll
                           + _jacobian)
        # Parameter count: (intercept, beta, sigma) + MVN moments
        n_params        = (p + 2) + p + p * (p + 1) // 2
        self.aic_       = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_       = n_params * np.log(n) - 2.0 * self.loglik_
        self.converged_ = bool(result.success)
        # scikit-learn expects n_iter_ wherever max_iter is exposed.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # SEs always use the numerical Hessian of the reduced NLL: the
        # optimizer's low-rank inverse-Hessian approximation is unreliable
        # for inference, and the reduced parameter space makes the exact
        # computation cheap.
        hess_inv = None

        if self.compute_se:
            se_std = self._compute_se(theta_opt, _reduced_nll, None)
            # Rescale to raw units: exact for slopes; the intercept uses a
            # diagonal delta approximation (covariances with the slopes are
            # ignored, as elsewhere in the penalized models).
            se_beta = se_std[1:] * (_sy / _sx)
            se_int  = _sy * np.sqrt(
                se_std[0] ** 2 + float(np.sum(((_mx / _sx) * se_std[1:]) ** 2))
            )
            self.se_ = np.concatenate([[se_int], se_beta])
        else:
            self.se_ = np.full(p + 1, np.nan)

        coefs_all = np.concatenate([[self.intercept_], self.coef_])
        safe_se   = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = coefs_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        sigma_X = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        sigma_Y = np.sqrt(max(
            self.sigma_sq_ + float(self.coef_ @ self.Sigma_X_ @ self.coef_),
            1e-12
        ))
        self.coef_std_ = self.coef_ * sigma_X / sigma_Y
        self.sigma_X_  = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def _predict_fiml(self, X):
        """Pattern-grouped FIML prediction on the internal scale."""
        beta      = self.coef_
        intercept = self.intercept_
        mu_X      = self.mu_X_
        Sigma_X   = self.Sigma_X_
        n, p      = X.shape
        y_hat     = np.empty(n)

        obs_matrix = ~np.isnan(X)
        complete   = obs_matrix.all(axis=1)
        all_miss   = ~obs_matrix.any(axis=1)
        partial    = ~complete & ~all_miss

        if complete.any():
            y_hat[complete] = intercept + X[complete] @ beta
        if all_miss.any():
            y_hat[all_miss] = intercept + beta @ mu_X
        if partial.any():
            X_part    = X[partial]
            part_idxs = np.where(partial)[0]
            groups    = defaultdict(list)
            for loc_i, x_i in enumerate(X_part):
                groups[tuple(np.where(~np.isnan(x_i))[0])].append(loc_i)
            all_idx = np.arange(p)
            for obs_key, loc_idxs in groups.items():
                obs_idx   = np.array(obs_key)
                mis_idx   = np.setdiff1d(all_idx, obs_idx)
                x_batch   = X_part[np.ix_(loc_idxs, obs_idx)]
                Sigma_obs = Sigma_X[np.ix_(obs_idx, obs_idx)]
                Sigma_mo  = Sigma_X[np.ix_(mis_idx, obs_idx)]
                K         = np.linalg.solve(Sigma_obs, Sigma_mo.T).T
                mu_c      = mu_X[mis_idx] + (x_batch - mu_X[obs_idx]) @ K.T
                y_hat[part_idxs[loc_idxs]] = (
                    intercept + x_batch @ beta[obs_idx] + mu_c @ beta[mis_idx]
                )
        return y_hat

    def predict(self, X):
        """
        Predict E[Y | X_obs].  NaN in X is treated as missing.

        Parameters
        ----------
        X : array-like, shape (n, p)

        Returns
        -------
        y_hat : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        y_hat = self._predict_fiml(X)
        if self.copula_used_:
            y_hat = self._copula_y_.inverse_transform_1d(y_hat, col=0)
        return y_hat

    def predict_interval(self, X, alpha=0.05):
        """
        Prediction interval reflecting missing-feature uncertainty.

        Parameters
        ----------
        X     : array-like, shape (n, p).  NaN allowed.
        alpha : float, significance level (default 0.05).

        Returns
        -------
        lower, upper : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)

        z        = norm.ppf(1.0 - alpha / 2.0)
        beta     = self.coef_
        mu_X     = self.mu_X_
        Sigma_X  = self.Sigma_X_
        sigma_sq = self.sigma_sq_
        n, p     = X.shape

        y_hat_internal = self._predict_fiml(X)
        se_pred = np.empty(n)

        obs_matrix = ~np.isnan(X)
        complete   = obs_matrix.all(axis=1)
        all_miss   = ~obs_matrix.any(axis=1)
        partial    = ~complete & ~all_miss

        if complete.any():
            se_pred[complete] = np.sqrt(sigma_sq)
        if all_miss.any():
            se_pred[all_miss] = np.sqrt(sigma_sq + float(beta @ Sigma_X @ beta))
        if partial.any():
            X_part    = X[partial]
            part_idxs = np.where(partial)[0]
            groups    = defaultdict(list)
            for loc_i, x_i in enumerate(X_part):
                groups[tuple(np.where(~np.isnan(x_i))[0])].append(loc_i)
            all_idx = np.arange(p)
            for obs_key, loc_idxs in groups.items():
                obs_idx   = np.array(obs_key)
                mis_idx   = np.setdiff1d(all_idx, obs_idx)
                beta_mis  = beta[mis_idx]
                Sigma_obs = Sigma_X[np.ix_(obs_idx, obs_idx)]
                Sigma_mo  = Sigma_X[np.ix_(mis_idx, obs_idx)]
                Sigma_mm  = Sigma_X[np.ix_(mis_idx, mis_idx)]
                K         = np.linalg.solve(Sigma_obs, Sigma_mo.T).T
                Sigma_c   = 0.5 * ((Sigma_mm - K @ Sigma_mo.T) + (Sigma_mm - K @ Sigma_mo.T).T)
                var_y     = sigma_sq + float(beta_mis @ Sigma_c @ beta_mis)
                se_pred[part_idxs[loc_idxs]] = np.sqrt(max(var_y, 0.0))

        lower = y_hat_internal - z * se_pred
        upper = y_hat_internal + z * se_pred
        if self.copula_used_:
            lower = self._copula_y_.inverse_transform_1d(lower, col=0)
            upper = self._copula_y_.inverse_transform_1d(upper, col=0)
        return lower, upper

    def score(self, X, y):
        """R² on the subset where y is not NaN."""
        check_is_fitted(self)
        X, y = self._validate_and_convert(X, y)
        y_hat = self.predict(X)
        obs   = ~np.isnan(y)
        y_o, yh_o = y[obs], y_hat[obs]
        ss_res = np.sum((y_o - yh_o) ** 2)
        ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self, alpha=0.05):
        """
        Print model summary.

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 74
        print()
        print('=' * W)
        print('MissLASSORegressor  --  LASSO-Penalized FIML Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}  "
              f"(non-zero: {self.n_nonzero_})")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  Alpha (LASSO)   : {self.alpha}")
        print(f"  Converged       : {self.converged_}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (coefficients on normal-transformed scale)")
        print(f"  Log-likelihood  : {self.loglik_:.4f}")
        print(f"  AIC             : {self.aic_:.4f}   BIC: {self.bic_:.4f}")
        print(f"  Residual std    : {np.sqrt(self.sigma_sq_):.4f}  "
              f"(var: {self.sigma_sq_:.4f})")
        if not self.compute_se:
            print("  Note: SE/p-value not computed (L1 penalty; set compute_se=True)")
        print('-' * W)
        print("  Coefficients:")
        for line in self._coef_table_lines(stat_label='z_stat', alpha=alpha):
            print('    ' + line)
        print()
        print("  Feature Importances (normalized standardized |coef|):")
        for line in self._importance_lines():
            print(line)
        print('=' * W)
        print()


# ======================================================================
# MissLASSOClassifier
# ======================================================================

class MissLASSOClassifier(ClassifierMixin, MissBase):
    """
    LASSO-penalized FIML logistic regression with native missing data support.

    Slope coefficients are penalized with an L1 norm via variable splitting.
    Missing X is marginalized via Gauss-Hermite quadrature over the
    conditional normal distribution of the missing features.

    Parameters
    ----------
    alpha : float
        LASSO regularization strength (default 1.0).  Applied on the
        internally standardized feature scale (glmnet convention), so it is
        unit-free and does not need retuning when the features are
        re-expressed in different units.
    max_iter : int
        Maximum optimizer iterations (default 3000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : {'L-BFGS-B', 'TNC', 'Powell', 'SLSQP'}
        scipy.optimize.minimize method (default 'L-BFGS-B'). The L1 term is
        written by variable splitting, beta = u - v with u, v >= 0, so the
        penalty only equals ``|beta|`` while those bounds hold. Solvers that
        cannot handle bounds discard them and the objective runs away, so
        the choice is restricted to those that can.
    n_quadrature : int
        Gauss-Hermite nodes for the small-variance branch of the
        logistic-normal integral (default 20). Wide variances use a
        step-plus-remainder rule that does not consult it.
    compute_se : bool
        Compute standard errors (default False).
    copula : bool or 'auto', default 'auto'
        Gaussian copula transform on X (default False).

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
    intercept_ : float
    n_nonzero_ : int
    odds_ratios_ : ndarray, shape (p+1,)
    mu_X_, Sigma_X_ : ndarray
    classes_ : ndarray
    loglik_, aic_, bic_ : float
    copula_used_ : bool
    """

    def __init__(self, alpha=1.0, max_iter=3000, tol=1e-7, method='L-BFGS-B',
                 n_quadrature=20, compute_se=False, copula='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_quadrature = n_quadrature
        self.compute_se = compute_se
        self.copula = copula

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Pattern grouping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_patterns(X):
        p = X.shape[1]
        all_idx = np.arange(p)
        groups  = defaultdict(list)
        for i, x_i in enumerate(X):
            obs_key = tuple(np.where(~np.isnan(x_i))[0])
            if obs_key:
                groups[obs_key].append(i)
        return [
            (np.array(obs_key),
             np.setdiff1d(all_idx, obs_key),
             np.array(idxs))
            for obs_key, idxs in groups.items()
        ]

    # ------------------------------------------------------------------ #
    # Single-row probability (used in predict_proba)
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Standard errors (optional)
    # ------------------------------------------------------------------ #

    def _compute_se(self, theta_opt, nll_fn, hess_inv=None):
        """
        SEs from the (small) reduced-parameter Hessian.

        theta layout: [intercept, u (p), v (p)]; the Hessian is mapped to
        beta space via d(beta_j)/d(u_j)=1, d(beta_j)/d(v_j)=-1.
        """
        p       = self.n_features_in_
        n_theta = len(theta_opt)
        if hess_inv is not None:
            Var_theta = hess_inv
        else:
            try:
                H = numerical_hessian(nll_fn, theta_opt, eps=1e-4)
                H += np.eye(n_theta) * 1e-8
                Var_theta = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return np.full(p + 1, np.nan)
        J = np.zeros((p + 1, n_theta))
        J[0, 0] = 1.0
        for j in range(p):
            J[j + 1, 1 + j]     =  1.0
            J[j + 1, 1 + p + j] = -1.0
        Var_beta = J @ Var_theta @ J.T
        return standard_errors_from_variance(Var_beta)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the LASSO-penalized FIML logistic regression model.

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
        check_penalty(self.alpha, 'alpha', 'MissLASSOClassifier')
        # The L1 term is written by variable splitting, beta = u - v with
        # u, v >= 0, so the penalty sum(u + v) only equals |beta| while those
        # bounds hold. Solvers that cannot handle bounds discard them, the
        # penalty turns into a reward for large coefficients and the
        # objective runs away: CG reached an R-squared of -2.3e6 here.
        # Nelder-Mead respects the bounds but does not optimise this problem,
        # returning the training mean at an R-squared of exactly 0 while
        # reporting convergence.
        check_choice(self.method, ('L-BFGS-B', 'TNC', 'Powell', 'SLSQP'),
                     'method', 'MissLASSOClassifier')
        self._store_fit_metadata(X, y)
        n, p = X.shape

        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        y_vals      = y[~np.isnan(y)]
        _classes = np.unique(y_vals)
        # Int cast only for integer-valued labels; truncating fractional
        # labels would break the classes_[1] binarization below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissLASSOClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )

        # Binarize against classes_[1] so any two labels work, keeping the
        # likelihood's y==1 encoding consistent with predict's classes_-based
        # mapping.  NaN (missing y) preserved.
        y = np.where(np.isnan(y), np.nan,
                     (y == self.classes_[1]).astype(float))

        # Penalized likelihoods are only scale-coherent on standardized data
        # (glmnet convention; see MissLASSORegressor.fit): on raw units the L1
        # penalty charges each coefficient for the size its feature's units
        # force it to have rather than for model complexity, so small-unit
        # features are annihilated and large-unit features escape the penalty
        # entirely.  y is binary and the model is on the logit scale, so only
        # the fixed effects transform; every public attribute is converted back
        # to the original feature scale after the fit.
        _mx = np.nanmean(X, axis=0)
        _mx = np.where(np.isfinite(_mx), _mx, 0.0)
        _sx = feature_scale(X)
        X = (X - _mx) / _sx
        # Stored so the prediction paths can run in the same standardised
        # space: the conditional solves are ill-conditioned in raw units when
        # the feature scales span orders of magnitude, even though the affine
        # conversion of the parameters themselves is exact.
        self._x_mean_, self._x_scale_ = _mx, _sx

        # Seed the optimiser from complete cases; if there are too few (large
        # p or high missingness) fall back to a mean-imputed seed rather than
        # failing, as in MissLogistic and MissLASSORegressor.  The FIML
        # likelihood below still integrates over every observed entry, so
        # this only sets the starting point.
        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_cc = X[complete_mask]
        y_cc = y[complete_mask]

        if len(y_cc) >= p + 2:
            X_seed, y_seed = X_cc, y_cc
        else:
            _mu0 = np.nanmean(X, axis=0)
            _mu0 = np.where(np.isfinite(_mu0), _mu0, 0.0)
            _obs = ~np.isnan(y)
            X_seed = np.where(np.isnan(X), _mu0, X)[_obs]
            y_seed = y[_obs]

        from sklearn.linear_model import LogisticRegression
        # If the seed rows contain a single class, fall back to a zero
        # seed rather than failing (same behavior as MissLogistic).
        if len(np.unique(y_seed)) >= 2:
            # random_state matters here and nowhere else in the file: this is
            # the only sklearn solver MissLearn uses that consumes an RNG.
            # liblinear shuffles its coordinate order, so with the default
            # random_state=None the seed moved on every call and carried
            # through to the final coefficients, which drifted by 1.6e-04
            # across five fits of identical data. The regressor sibling seeds
            # from Lasso, which is deterministic, which is why only the
            # classifier was affected.
            lr0 = LogisticRegression(
                penalty='l1', solver='liblinear', C=1.0 / max(self.alpha, 1e-6),
                max_iter=1000, random_state=0,
            )
            lr0.fit(X_seed, y_seed)
            intercept0 = float(lr0.intercept_[0])
            beta0_X    = lr0.coef_.ravel()
        else:
            intercept0 = 0.0
            beta0_X    = np.zeros(p)

        u0 = np.maximum(beta0_X, 0.0) + 1e-6
        v0 = np.maximum(-beta0_X, 0.0) + 1e-6

        # --- Stage 1: X-moments by EM (full-information) -----------------
        # Estimating the MVN nuisance parameters once removes p + p(p+1)/2
        # dimensions from the optimizer below, whose finite-difference
        # gradients otherwise dominate the runtime.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(X)
        mu_X_opt    = _fitter.mu_
        Sigma_X_opt = _fitter.Sigma_

        # --- Stage 2: penalized conditional ML for (intercept, beta) -----
        # Pattern-constant terms are precomputed (vectorized GH quadrature
        # per pattern), so each NLL call is O(n p).
        _obs_y = ~np.isnan(y)
        y_y = y[_obs_y]
        F_y, cond_groups = _prep_conditional_terms(X[_obs_y], mu_X_opt,
                                                   Sigma_X_opt)

        # Vectorized group structures: group_id per row and the (G, p, p)
        # tensor of Sigma_c scattered to full-p coordinates.  The Bernoulli
        # NLL and its analytic gradient evaluate the Gauss-Hermite sum for
        # ALL rows at once (complete rows have v_g = 0, so their GH nodes
        # collapse to sigmoid(a) exactly); no Python loop over patterns,
        # no finite differences.
        n_y_rows = len(y_y)
        G = len(cond_groups)
        group_id = np.empty(n_y_rows, dtype=np.intp)
        M = np.zeros((G, p, p))
        for g, (rows, mis, Sc) in enumerate(cond_groups):
            group_id[rows] = g
            if mis.size:
                M[g][np.ix_(mis, mis)] = Sc
        _is_pos = (y_y == 1)
        _bad_grad = np.zeros(2 * p + 1)

        def _nll_grad(theta_r):
            intercept = theta_r[0]
            u_r = theta_r[1:p + 1]
            v_r = theta_r[p + 1:2 * p + 1]
            beta = u_r - v_r

            Mb  = M @ beta                                    # (G, p)
            v_g = np.maximum(Mb @ beta, 0.0)                  # (G,)
            if not np.all(np.isfinite(v_g)):
                return np.inf, _bad_grad
            a = intercept + F_y @ beta                        # (n,)
            v_row = v_g[group_id]                             # (n,)

            # One rule for fitting and for predicting; see the matching note
            # in _logistic.py for why Gauss-Hermite alone cannot hold this
            # integral once v_g grows.
            p1, dp_da, dp_dv = logistic_normal_with_grads(
                a, v_row, self.n_quadrature)
            p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)

            nll = -(float(np.sum(np.log(p1[_is_pos])))
                    + float(np.sum(np.log1p(-p1[~_is_pos]))))
            if self.alpha > 0:
                nll += self.alpha * float(np.sum(u_r) + np.sum(v_r))

            # Analytic gradient. The helper supplies dp1/da and dp1/dv; the
            # chain rule out to (intercept, u, v) is this estimator's own.
            d     = np.where(_is_pos, -1.0 / p1, 1.0 / (1.0 - p1))  # dNLL/dp1
            g_a   = d * dp_da                             # per-row dNLL/da
            g_vg  = np.bincount(group_id, weights=d * dp_dv, minlength=G)
            g_int  = float(np.sum(g_a))
            g_beta = F_y.T @ g_a + 2.0 * (g_vg[:, None] * Mb).sum(axis=0)
            grad = np.concatenate([
                [g_int], g_beta + self.alpha, -g_beta + self.alpha
            ])
            return nll, grad

        def _reduced_nll(theta_r):
            return _nll_grad(theta_r)[0]

        theta0 = np.concatenate([[intercept0], u0, v0])
        bounds = [(None, None)] + [(0.0, None)] * (2 * p)

        # Powell is derivative-free. Handing it a gradient makes scipy warn
        # on every fit about a jac it will not use, so give it the value-only
        # form that _compute_se already relies on.
        if self.method == 'Powell':
            result = minimize(
                fun=_reduced_nll,
                x0=theta0,
                method=self.method,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'ftol': self.tol},
            )
        else:
            result = minimize(
                fun=_nll_grad,
                x0=theta0,
                jac=True,
                method=self.method,
                bounds=bounds,
                options={'maxiter': self.max_iter, 'ftol': self.tol,
                         'gtol': self.tol * 1e-2},
            )

        theta_opt = result.x
        self._theta_opt_ = result.x
        beta_opt = theta_opt[1:p + 1] - theta_opt[p + 1:2 * p + 1]

        # Standardised-scale estimates, kept for the prediction paths
        b0_std   = float(theta_opt[0])
        beta_std = beta_opt
        self._b0_std_      = b0_std
        self._beta_std_    = beta_std
        self._mu_X_std_    = mu_X_opt
        self._Sigma_X_std_ = Sigma_X_opt
        # [intercept, coef] on the standardised scale, used by predict_proba
        # and decision_function (both run in the standardised space).
        self._beta_full  = np.concatenate([[b0_std], beta_std])
        # Sparsity is judged on the standardized scale (unit-free threshold),
        # matching MissLASSORegressor.
        self.n_nonzero_  = int(np.sum(np.abs(beta_std) > _ZERO_THRESH))

        # --- convert public parameters back to the original feature scale ---
        # The logit is affine in X, so b0_std + beta_std @ (x - mx)/sx equals
        # intercept_ + coef_ @ x exactly.
        self.intercept_  = float(b0_std - float((beta_std / _sx) @ _mx))
        self.coef_       = beta_std / _sx
        self.mu_X_       = _mx + _sx * mu_X_opt
        self.Sigma_X_    = Sigma_X_opt * np.outer(_sx, _sx)

        # result.fun includes the L1 penalty; add it back so loglik_/AIC/BIC
        # reflect the data likelihood only.  The full-information
        # log-likelihood is the conditional part plus the X-marginal part
        # (from the EM stage), plus the Jacobian of the X standardisation which
        # moves it onto the raw scale so AIC/BIC stay comparable across unit
        # choices (y is binary, so there is no y term).
        u_opt = theta_opt[1:p + 1]
        v_opt = theta_opt[p + 1:2 * p + 1]
        penalty = self.alpha * float(np.sum(u_opt) + np.sum(v_opt))
        _marg_ll = _mvn_loglik(X, mu_X_opt, Sigma_X_opt)
        _n_obs_col = np.sum(~np.isnan(X), axis=0)
        _jacobian  = -float(_n_obs_col @ np.log(_sx))
        self.loglik_    = -(float(result.fun) - penalty) + _marg_ll + _jacobian
        # Parameter count: (intercept, beta) + MVN moments
        n_params        = (p + 1) + p + p * (p + 1) // 2
        self.aic_       = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_       = n_params * np.log(n) - 2.0 * self.loglik_
        self.converged_ = bool(result.success)
        # scikit-learn expects n_iter_ wherever max_iter is exposed.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # Overflow to inf is the intended reading, not a fault; see the note
        # at the matching line in _logistic.py. MissMixedClassifier already
        # guarded this and its two siblings did not.
        beta_all          = np.concatenate([[self.intercept_], self.coef_])
        with np.errstate(over='ignore'):
            self.odds_ratios_ = np.exp(beta_all)

        # SEs always use the numerical Hessian of the reduced NLL: the
        # optimizer's low-rank inverse-Hessian approximation is unreliable
        # for inference, and the reduced parameter space makes the exact
        # computation cheap.
        hess_inv = None

        if self.compute_se:
            se_std = self._compute_se(theta_opt, _reduced_nll, None)
            # Delta-method rescale to raw units: exact for the slopes; the
            # intercept uses a diagonal approximation (covariances with the
            # slopes ignored, as elsewhere in the penalized models).
            se_beta = se_std[1:] / _sx
            se_int  = np.sqrt(
                se_std[0] ** 2 + float(np.sum(((_mx / _sx) * se_std[1:]) ** 2))
            )
            self.se_ = np.concatenate([[se_int], se_beta])
        else:
            self.se_ = np.full(p + 1, np.nan)

        safe_se       = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = beta_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        # Raw scale throughout, so the product is invariant to the internal
        # standardisation.
        sigma_X        = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        self.coef_std_ = self.coef_ * sigma_X
        self.sigma_X_  = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_proba(self, X):
        """
        P(Y=0|X_obs) and P(Y=1|X_obs).  NaN in X treated as missing.

        Parameters
        ----------
        X : array-like, shape (n, p)

        Returns
        -------
        ndarray, shape (n, 2)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        # All conditional solves run in the standardised space of the fit; the
        # resulting logit is identical to a raw-space computation but
        # numerically stable when the feature scales span orders of magnitude.
        X = (X - self._x_mean_) / self._x_scale_
        beta = self._beta_full
        mu_X = self._mu_X_std_
        Sigma_X = self._Sigma_X_std_

        beta_0 = beta[0]
        beta_X = beta[1:]
        n = X.shape[0]
        p1 = np.full(n, np.nan)

        all_missing_mask = np.isnan(X).all(axis=1)
        if all_missing_mask.any():
            v_full = float(beta_X @ Sigma_X @ beta_X)
            p1[all_missing_mask] = integrate_logistic_normal(
                float(beta_0 + beta_X @ mu_X), v_full, self.n_quadrature
            )

        patterns = self._group_patterns(X)
        for obs_idx, mis_idx, row_idxs in patterns:
            x_obs_batch = X[np.ix_(row_idxs, obs_idx)]
            beta_obs = beta_X[obs_idx]

            if len(mis_idx) == 0:
                a_vals = beta_0 + x_obs_batch @ beta_obs
                p1[row_idxs] = sigmoid(a_vals)
            else:
                beta_mis = beta_X[mis_idx]
                mu_obs = mu_X[obs_idx]
                Sigma_obs = Sigma_X[np.ix_(obs_idx, obs_idx)]
                Sigma_mo = Sigma_X[np.ix_(mis_idx, obs_idx)]
                Sigma_mm = Sigma_X[np.ix_(mis_idx, mis_idx)]
                K = np.linalg.solve(Sigma_obs, Sigma_mo.T).T
                Sigma_c = Sigma_mm - K @ Sigma_mo.T
                Sigma_c = 0.5 * (Sigma_c + Sigma_c.T)
                v = float(beta_mis @ Sigma_c @ beta_mis)

                mu_mis = mu_X[mis_idx]
                mu_c_batch = mu_mis + (x_obs_batch - mu_obs) @ K.T
                a_vals = beta_0 + x_obs_batch @ beta_obs + mu_c_batch @ beta_mis

                p1[row_idxs] = integrate_logistic_normal(a_vals, v, self.n_quadrature)

        p1 = np.clip(p1, 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Predict binary class labels."""
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def decision_function(self, X):
        """Log-odds scores.  Missing X filled with conditional normal mean."""
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        # Standardised space of the fit (see predict_proba)
        X = (X - self._x_mean_) / self._x_scale_
        beta      = self._beta_full
        log_odds  = np.empty(X.shape[0])
        obs_matrix = ~np.isnan(X)
        complete   = obs_matrix.all(axis=1)
        all_miss   = ~obs_matrix.any(axis=1)
        partial    = ~complete & ~all_miss

        if complete.any():
            log_odds[complete] = beta[0] + X[complete] @ beta[1:]
        if all_miss.any():
            log_odds[all_miss] = float(beta[0] + beta[1:] @ self._mu_X_std_)
        if partial.any():
            for i in np.where(partial)[0]:
                obs_idx = np.where(obs_matrix[i])[0]
                mis_idx = np.where(~obs_matrix[i])[0]
                mu_c, _ = conditional_normal_params(
                    self._mu_X_std_, self._Sigma_X_std_,
                    obs_idx, mis_idx, X[i, obs_idx]
                )
                x_filled = X[i].copy()
                x_filled[mis_idx] = mu_c
                log_odds[i] = beta[0] + beta[1:] @ x_filled
        return log_odds

    def score(self, X, y):
        """Accuracy on the subset where y is not NaN."""
        check_is_fitted(self)
        X, y = self._validate_and_convert(X, y)
        obs    = ~np.isnan(y)
        y_pred = self.predict(X[obs])
        return float(np.mean(y_pred == y[obs]))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self, alpha=0.05):
        """
        Print model summary.

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 80
        print()
        print('=' * W)
        print('MissLASSOClassifier  --  LASSO-Penalized FIML Logistic Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}  "
              f"(non-zero: {self.n_nonzero_})")
        print(f"  Classes         : {self.classes_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  Alpha (LASSO)   : {self.alpha}")
        print(f"  Converged       : {self.converged_}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (coefficients on normal-transformed X scale)")
        print(f"  Log-likelihood  : {self.loglik_:.4f}")
        print(f"  AIC             : {self.aic_:.4f}   BIC: {self.bic_:.4f}")
        if not self.compute_se:
            print("  Note: SE/p-value not computed (L1 penalty; set compute_se=True)")
        print('-' * W)
        print("  Coefficients (log-odds scale) with Odds Ratios:")

        or_header = f"{'odds_ratio':>12}"
        or_rows   = [f"{self.odds_ratios_[i]:>12.4f}"
                     for i in range(len(self.odds_ratios_))]

        for line in self._coef_table_lines(
            stat_label='z_stat', alpha=alpha,
            extra_header=or_header, extra_rows=or_rows
        ):
            print('    ' + line)

        print()
        print("  Feature Importances (normalized standardized |coef|):")
        for line in self._importance_lines():
            print(line)
        print('=' * W)
        print()


# ======================================================================
# MissLASSO  --  unified auto-selecting wrapper
# ======================================================================

class MissLASSO(MissTags, BaseEstimator):
    """
    Unified LASSO-penalized FIML model that automatically selects between
    regression and classification based on the observed values of y.

    Detection rule (applied at fit time):
        All observed y in {0, 1}  ->  MissLASSOClassifier
        Otherwise                 ->  MissLASSORegressor

    The underlying model is stored in ``model_`` and the task in ``task_``
    ('regression' or 'classification') after fitting.  All methods
    (predict, predict_proba, score, summary, ...) delegate to ``model_``;
    fitted attributes are accessible directly via attribute delegation.

    Parameters
    ----------
    alpha : float
        LASSO regularization strength (default 1.0).
    max_iter : int
        Maximum optimizer iterations (default 3000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : {'L-BFGS-B', 'TNC', 'Powell', 'SLSQP'}
        scipy.optimize.minimize method (default 'L-BFGS-B'). The L1 term is
        written by variable splitting, beta = u - v with u, v >= 0, so the
        penalty only equals ``|beta|`` while those bounds hold. Solvers that
        cannot handle bounds discard them and the objective runs away, so
        the choice is restricted to those that can.
    n_quadrature : int
        GH nodes for the small-variance branch of the classification integral
        (default 20).  Ignored for regression.
    compute_se : bool
        Attempt SE computation (default False; see class docstrings).
    copula : bool or 'auto', default 'auto'
        Gaussian copula transform. 'auto' (the default) applies it
        only when the marginals are skewed or heavy-tailed enough to
        warrant it; True forces it on and False off.
    """


    def __init__(self, alpha=1.0, max_iter=3000, tol=1e-7, method='L-BFGS-B',
                 n_quadrature=20, compute_se=False, copula='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_quadrature = n_quadrature
        self.compute_se = compute_se
        self.copula = copula

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
        Detect task and fit the appropriate LASSO-penalized FIML model.

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
        if self.task_ == 'classification':
            self.model_ = MissLASSOClassifier(
                alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
                method=self.method, n_quadrature=self.n_quadrature,
                compute_se=self.compute_se, copula=self.copula,
            )
        else:
            self.model_ = MissLASSORegressor(
                alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
                method=self.method, compute_se=self.compute_se, copula=self.copula,
            )
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
        """Prediction interval.  Only available for regression."""
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


    def summary(self, alpha=0.05):
        """Print the model summary."""
        check_is_fitted(self, 'model_')
        return self.model_.summary(alpha=alpha)
