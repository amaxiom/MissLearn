"""
MissLearn._ridge
----------------
Ridge-penalized FIML models with native missing data support.

MissRidgeRegressor
    Models Y | X ~ N(beta @ X + intercept, sigma^2) with X ~ N(mu_X, Sigma_X).
    Beta is a free parameter directly penalized by an L2 (Ridge) term.  This
    differs from MissLinear, where beta is derived analytically from the joint
    covariance: here it is a free parameter in the optimization, so the Ridge
    penalty acts directly and the regularization strength is explicit.

    For observations with missing X, the marginal distribution of Y given only
    the observed predictors is also normal (closed-form):
        E[Y | X_obs]   = intercept + beta_obs @ x_obs + beta_mis @ mu_{mis|obs}
        Var[Y | X_obs] = sigma^2   + beta_mis @ Sigma_{mis|obs} @ beta_mis

    Sigma_{mis|obs} is constant within each missingness pattern, so it is
    computed once per pattern group rather than once per observation.

MissRidgeClassifier
    Ridge-penalized FIML logistic regression.  Thin subclass of MissLogistic
    that uses alpha as the regularization parameter name (matching sklearn's
    Ridge convention) and provides a dedicated summary line.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase, MissTags, only_for
from ._conformance import check_common_parameters, check_penalty
from ._copula import RankNormalTransformer, needs_copula
from ._logistic import MissLogistic
from ._utils import (
    pack_cholesky, unpack_cholesky, mvn_logpdf_batch,
    conditional_normal_params, numerical_hessian, feature_scale,
    standard_errors_from_variance
)


# ======================================================================
# MissRidgeRegressor
# ======================================================================

class MissRidgeRegressor(RegressorMixin, MissBase):
    """
    Ridge-penalized FIML linear regression with native missing data support.

    Models the conditional distribution Y | X ~ N(beta @ X + intercept, sigma^2)
    and the marginal predictor distribution X ~ N(mu_X, Sigma_X).  Missing
    values in X and y are handled without imputation.

    Unlike MissLinear (which derives beta analytically from a joint covariance),
    beta here is a free parameter subject to an explicit L2 penalty.  This makes
    the regularization strength directly controllable and avoids the assumption
    that Y and X are jointly normal.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength (default 1.0).  Larger values give
        stronger shrinkage toward zero.  alpha=0 recovers unpenalized FIML
        regression (similar to MissLinear but with a conditional model).
    max_iter : int
        Maximum optimizer iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    compute_se : bool
        If True (default), compute standard errors from the Hessian diagonal.
        Because beta is a free parameter, no delta method is needed.
    copula : bool or 'auto', default 'auto'
        If True, apply a marginal Gaussian copula transform to X (and y) before
        fitting.  If 'auto', apply when data appears non-normal (``|skewness|`` > 1
        or ``|excess kurtosis|`` > 2 on any column).

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
    intercept_ : float
    sigma_sq_ : float
        Estimated residual variance.
    se_ : ndarray, shape (p+1,)
        Standard errors: [se_intercept, se_coef_0, ..., se_coef_{p-1}].
    pvalues_ : ndarray, shape (p+1,)
    z_stats_ : ndarray, shape (p+1,)
    coef_std_ : ndarray, shape (p,)
        Standardized coefficients: coef_j * sigma_Xj / sigma_Y_marginal.
    mu_X_ : ndarray, shape (p,)
    Sigma_X_ : ndarray, shape (p, p)
    loglik_ : float
    aic_, bic_ : float
    copula_used_ : bool
        Whether the copula transform was actually applied (resolved from 'auto').
    """

    def __init__(self, alpha=1.0, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 compute_se=True, copula='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.compute_se = compute_se
        self.copula = copula

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    def _pack_params(self, intercept, beta, sigma, mu_X, Sigma_X):
        """
        Pack (intercept, beta, sigma, mu_X, Sigma_X) into an unconstrained vector:
            theta = [intercept (1), beta (p), log(sigma) (1),
                     mu_X (p), L_X_vec (p*(p+1)//2)]

        sigma is stored as log(sigma) to enforce positivity.
        """
        L_vec, _ = pack_cholesky(Sigma_X)
        return np.concatenate([[intercept], beta, [np.log(sigma)], mu_X, L_vec])

    # ------------------------------------------------------------------ #
    # Missingness pattern grouping
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Standard errors
    # ------------------------------------------------------------------ #

    def _compute_se(self, theta_opt, nll_fn, hess_inv=None):
        """
        Compute standard errors from the inverse Hessian of the reduced
        conditional NLL (theta = [intercept, beta, log_sigma]).

        (intercept, beta) are the first p+1 elements of theta, so no delta
        method is needed; their SEs come directly from the Hessian diagonal.
        """
        n_theta = len(theta_opt)
        p = self.n_features_in_

        if hess_inv is not None:
            Var_theta = hess_inv
        else:
            try:
                H = numerical_hessian(nll_fn, theta_opt, eps=1e-4)
                H += np.eye(n_theta) * 1e-8
                Var_theta = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return np.full(p + 1, np.nan)

        se_all = standard_errors_from_variance(Var_theta)
        return se_all[:p + 1]   # [se_intercept, se_beta_1, ..., se_beta_p]

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the Ridge-penalized FIML regression model.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    NaN values treated as missing.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        check_penalty(self.alpha, 'alpha', 'MissRidgeRegressor')
        self._store_fit_metadata(X, y)
        n, p = X.shape

        # Resolve copula='auto'
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

        # Seed the optimiser.  Prefer complete cases; if there are too few
        # (large p or high missingness), fall back to a mean-imputed seed
        # rather than failing; the FIML likelihood below still uses every
        # observed entry, so this only sets the starting point.
        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_cc = X[complete_mask]
        y_cc = y[complete_mask]

        mu_X0 = np.nanmean(X, axis=0)
        mu_X0 = np.where(np.isfinite(mu_X0), mu_X0, 0.0)

        if len(y_cc) >= p + 2:
            X_seed, y_seed, X_cov = X_cc, y_cc, X_cc
        else:
            obs_y  = ~np.isnan(y)
            X_imp  = np.where(np.isnan(X), mu_X0, X)
            X_seed, y_seed, X_cov = X_imp[obs_y], y[obs_y], X_imp

        from sklearn.linear_model import Ridge as _Ridge
        r0 = _Ridge(alpha=self.alpha)
        r0.fit(X_seed, y_seed)
        beta0 = r0.coef_
        intercept0 = float(r0.intercept_)
        sigma0 = max(np.std(y_seed - r0.predict(X_seed)), 1e-4)

        # --- Stage 1: X-moments by EM (full-information, includes y) -----
        # Estimating the MVN nuisance parameters once removes p + p(p+1)/2
        # dimensions from the optimizer below, whose finite-difference
        # gradients otherwise dominate the runtime.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        from ._utils import prep_conditional_terms
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(np.column_stack([X, y]))
        mu_X_opt    = _fitter.mu_[:p]
        Sigma_X_opt = _fitter.Sigma_[:p, :p]

        # --- Stage 2: penalized conditional ML for (intercept, beta, sigma)
        # Pattern-constant terms are precomputed, so each NLL call is O(n p).
        _obs_y_rows = ~np.isnan(y)
        y_y = y[_obs_y_rows]
        F_y, cond_groups = prep_conditional_terms(X[_obs_y_rows], mu_X_opt,
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
        _bad_grad = np.zeros(p + 2)

        def _nll_grad(theta_r):
            intercept = theta_r[0]
            beta = theta_r[1:p + 1]
            sigma_sq = float(np.exp(2.0 * theta_r[p + 1]))

            Mb  = M @ beta                                   # (G, p)
            v_g = sigma_sq + Mb @ beta                       # (G,)
            if not np.all(np.isfinite(v_g)) or np.any(v_g <= 0):
                return np.inf, _bad_grad
            resid = y_y - intercept - F_y @ beta
            rss_g = np.bincount(group_id, weights=resid * resid, minlength=G)

            nll = (0.5 * float(n_g @ (_LOG2PI + np.log(v_g)))
                   + 0.5 * float(np.sum(rss_g / v_g)))
            if self.alpha > 0:
                nll += 0.5 * self.alpha * float(np.dot(beta, beta))

            # Analytic gradient
            w_v = 0.5 * n_g / v_g - 0.5 * rss_g / v_g ** 2   # dNLL/dv_g
            r_w = resid / v_g[group_id]
            g_int  = -float(np.sum(r_w))
            g_beta = -(F_y.T @ r_w) + 2.0 * (w_v[:, None] * Mb).sum(axis=0)
            if self.alpha > 0:
                g_beta = g_beta + self.alpha * beta
            g_logs = float(np.sum(w_v)) * 2.0 * sigma_sq
            return nll, np.concatenate([[g_int], g_beta, [g_logs]])

        def _reduced_nll(theta_r):
            return _nll_grad(theta_r)[0]

        theta0 = np.concatenate([[intercept0], beta0, [np.log(sigma0)]])

        result = minimize(
            fun=_nll_grad,
            x0=theta0,
            jac=True,
            method=self.method,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'gtol': self.tol * 1e-2,
            }
        )

        theta_opt = result.x
        self._theta_opt_ = result.x
        intercept_opt = float(theta_opt[0])
        beta_opt      = theta_opt[1:p + 1]
        sigma_opt     = float(np.exp(theta_opt[p + 1]))

        # Convert every public parameter back to the raw (unstandardized)
        # scale.  The FIML model is affine-equivariant, so predict() and
        # predict_interval() work unchanged on these raw-scale parameters.
        self.coef_      = beta_opt * (_sy / _sx)
        self.intercept_ = (_my + _sy * intercept_opt
                           - float(self.coef_ @ _mx))
        self.sigma_sq_  = (_sy ** 2) * sigma_opt ** 2
        self.mu_X_      = _mx + _sx * mu_X_opt
        self.Sigma_X_   = Sigma_X_opt * np.outer(_sx, _sx)

        # result.fun includes the ridge penalty; recover the pure data
        # log-likelihood for interpretable AIC / BIC.  The full-information
        # log-likelihood is the conditional part (from the optimizer) plus
        # the X-marginal part (from the EM stage); the Jacobian term moves
        # it from the standardized to the raw scale.
        ridge_penalty = 0.5 * self.alpha * float(np.dot(beta_opt, beta_opt))
        _marg_ll   = _mvn_loglik(X, mu_X_opt, Sigma_X_opt)
        _n_obs_col = np.sum(~np.isnan(X), axis=0)
        _jacobian  = (-float(_n_obs_col @ np.log(_sx))
                      - float(_y_obs.sum()) * np.log(_sy))
        self.loglik_ = (-float(result.fun) + ridge_penalty + _marg_ll
                        + _jacobian)
        # Parameter count: (intercept, beta, sigma) + MVN moments
        n_params = (p + 2) + p + p * (p + 1) // 2
        self.aic_ = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_ = n_params * np.log(n) - 2.0 * self.loglik_
        self.converged_ = bool(result.success)
        # scikit-learn expects n_iter_ wherever max_iter is exposed.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # SEs always come from a numerical Hessian of the reduced NLL: with
        # analytic gradients L-BFGS-B converges in a handful of iterations,
        # leaving its low-rank inverse-Hessian approximation near the
        # identity; useless for inference.  The reduced NLL is vectorised,
        # so the exact numerical Hessian costs only milliseconds.
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
        safe_se = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = coefs_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        # Standardized coefficients (computed from raw-scale parameters)
        sigma_X = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        # Marginal std of y: sqrt(sigma^2 + beta @ Sigma_X @ beta)
        sigma_Y_marginal = np.sqrt(max(
            self.sigma_sq_ + float(self.coef_ @ self.Sigma_X_ @ self.coef_),
            1e-12
        ))
        self.coef_std_ = self.coef_ * sigma_X / sigma_Y_marginal
        self.sigma_X_ = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, X):
        """
        Predict E[Y | X_obs] for each row.

        Complete rows: intercept + beta @ x.
        Partial rows:  intercept + beta_obs @ x_obs + beta_mis @ mu_{mis|obs}.
        All-missing:   intercept + beta @ mu_X  (unconditional mean of Y).

        Parameters
        ----------
        X : array-like, shape (n_new, p).  NaN allowed.

        Returns
        -------
        y_hat : ndarray, shape (n_new,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        y_hat = self._predict_fiml(X)
        if self.copula_used_:
            y_hat = self._copula_y_.inverse_transform_1d(y_hat, col=0)
        return y_hat

    def _predict_fiml(self, X):
        """
        Core FIML prediction on the internal (possibly transformed) scale.

        Rows are grouped by missingness pattern so that each pattern incurs
        a single Cholesky solve rather than one solve per row.
        """
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
                K         = np.linalg.solve(Sigma_obs, Sigma_mo.T).T  # (|mis|, |obs|)
                mu_c      = mu_X[mis_idx] + (x_batch - mu_X[obs_idx]) @ K.T  # (n_grp, |mis|)
                y_hat[part_idxs[loc_idxs]] = (
                    intercept + x_batch @ beta[obs_idx] + mu_c @ beta[mis_idx]
                )

        return y_hat

    def predict_interval(self, X, alpha=0.05):
        """
        Compute prediction intervals reflecting missing-feature uncertainty.

        Var[Y | X_obs] = sigma^2 + beta_mis @ Sigma_{mis|obs} @ beta_mis,
        which is larger when more features are missing.

        Parameters
        ----------
        X     : array-like, shape (n_new, p).  NaN allowed.
        alpha : float, significance level (default 0.05 for 95% interval).

        Returns
        -------
        lower, upper : ndarray, shape (n_new,)
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
                Sigma_c   = Sigma_mm - K @ Sigma_mo.T
                Sigma_c   = 0.5 * (Sigma_c + Sigma_c.T)
                var_y     = sigma_sq + float(beta_mis @ Sigma_c @ beta_mis)
                se_pred[part_idxs[loc_idxs]] = np.sqrt(max(var_y, 0.0))

        lower = y_hat_internal - z * se_pred
        upper = y_hat_internal + z * se_pred

        if self.copula_used_:
            lower = self._copula_y_.inverse_transform_1d(lower, col=0)
            upper = self._copula_y_.inverse_transform_1d(upper, col=0)

        return lower, upper

    def score(self, X, y):
        """
        R² on the subset of observations where y is not NaN.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)

        Returns
        -------
        float, R² in (-inf, 1]
        """
        check_is_fitted(self)
        X, y = self._validate_and_convert(X, y)
        y_hat = self.predict(X)
        obs = ~np.isnan(y)
        y_o, yh_o = y[obs], y_hat[obs]
        ss_res = np.sum((y_o - yh_o) ** 2)
        ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self, alpha=0.05):
        """
        Print a comprehensive model summary.

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 74
        print()
        print('=' * W)
        print('MissRidgeRegressor  --  Ridge-Penalized FIML Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  Alpha (Ridge)   : {self.alpha}")
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
# MissRidgeClassifier
# ======================================================================

class MissRidgeClassifier(MissLogistic):
    """
    Ridge-penalized FIML logistic regression with native missing data support.

    Equivalent to MissLogistic with an explicit regularization strength alpha
    (matching sklearn's Ridge naming convention).  alpha=0 recovers exact FIML
    logistic regression; larger alpha shrinks coefficients toward zero.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength on slope coefficients (default 1.0).
        The intercept is not penalized.
    max_iter, tol, method, n_quadrature, compute_se, copula
        Passed through to MissLogistic; see MissLogistic for details.
    """

    def __init__(self, alpha=1.0, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 n_quadrature=20, compute_se=True, copula='auto'):
        self.alpha = alpha
        super().__init__(
            max_iter=max_iter, tol=tol, method=method,
            n_quadrature=n_quadrature, compute_se=compute_se,
            l2_reg=alpha, copula=copula,
        )

    @property
    def l2_reg(self):
        """Alias for ``alpha``, read by the inherited MissLogistic NLL.

        This class subclasses MissLogistic, whose objective reads
        ``self.l2_reg``, but ridge is spelled ``alpha`` everywhere in
        scikit-learn and in the rest of this library, so ``alpha`` is the
        parameter and this is the adapter.

        ``MissRidgeRegressor`` deliberately has no counterpart. It does not
        inherit MissLogistic's objective, so there is nothing to adapt, and
        adding a second public name for its penalty would be a wart rather
        than symmetry. Anyone comparing the two classes' attribute lists will
        find this missing on the regressor; that is the reason.

        The scikit-learn parameter contract is unaffected: ``get_params``
        reports ``alpha`` and not ``l2_reg``, so ``clone`` has one name to
        copy, and ``set_params(l2_reg=...)`` raises rather than silently
        writing to a name the estimator does not declare.
        """
        return self.alpha

    @l2_reg.setter
    def l2_reg(self, value):
        self.alpha = value

    def summary(self, alpha=0.05):
        """
        Print a comprehensive model summary.

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 80
        print()
        print('=' * W)
        print('MissRidgeClassifier  --  Ridge-Penalized FIML Logistic Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Classes         : {self.classes_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  Alpha (Ridge)   : {self.alpha}")
        print(f"  Converged       : {self.converged_}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (coefficients on normal-transformed X scale)")
        print(f"  Log-likelihood  : {self.loglik_:.4f}")
        print(f"  AIC             : {self.aic_:.4f}   BIC: {self.bic_:.4f}")
        print('-' * W)
        print("  Coefficients (log-odds scale) with Odds Ratios:")

        or_header = f"{'odds_ratio':>12}"
        or_rows = [f"{self.odds_ratios_[i]:>12.4f}"
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
# MissRidge  --  unified auto-selecting wrapper
# ======================================================================

class MissRidge(MissTags, BaseEstimator):
    """
    Unified Ridge-penalized FIML model that automatically selects between
    regression and classification based on the observed values of y.

    Detection rule (applied at fit time):
        If every observed (non-NaN) value of y is in {0, 1}  ->  classification
                                                               (MissRidgeClassifier)

        Otherwise                                             ->  regression
                                                               (MissRidgeRegressor)

    After fitting, the underlying model is stored in ``model_`` and the
    detected task in ``task_`` ('regression' or 'classification').
    All public methods (predict, predict_proba, score, summary, etc.) are
    forwarded to ``model_``; fitted attributes (``coef_``, ``intercept_``, ``classes_``,
    ...) are also accessible directly on this object via attribute delegation.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength (default 1.0).
    max_iter : int
        Maximum optimizer iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    n_quadrature : int
        Gauss-Hermite quadrature nodes for classification (default 20).
        Ignored for regression.
    compute_se : bool
        Compute standard errors (default True).
    copula : bool or 'auto', default 'auto'
        Gaussian copula transform. 'auto' (the default) applies it
        only when the marginals are skewed or heavy-tailed enough to
        warrant it; True forces it on and False off.  See MissRidgeRegressor
        and MissRidgeClassifier for details.
    """


    def __init__(self, alpha=1.0, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 n_quadrature=20, compute_se=True, copula='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_quadrature = n_quadrature
        self.compute_se = compute_se
        self.copula = copula

    # ------------------------------------------------------------------ #
    # Task detection
    # ------------------------------------------------------------------ #

    @staticmethod

    def _detect_task(y):
        """
        Return 'classification' if all observed y values are in {0, 1},
        else 'regression'.
        """
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        obs = y_arr[~np.isnan(y_arr)]
        if len(obs) == 0:
            raise ValueError("y has no observed (non-NaN) values.")
        unique_vals = set(np.unique(obs).tolist())
        if unique_vals <= {0.0, 1.0}:
            return 'classification'
        return 'regression'

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #


    def fit(self, X, y):
        """
        Detect task type and fit the appropriate Ridge-penalized FIML model.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    NaN values treated as missing.

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
            self.model_ = MissRidgeClassifier(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                method=self.method,
                n_quadrature=self.n_quadrature,
                compute_se=self.compute_se,
                copula=self.copula,
            )
        else:
            self.model_ = MissRidgeRegressor(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                method=self.method,
                compute_se=self.compute_se,
                copula=self.copula,
            )

        self.model_.fit(X, y)
        return self

    # ------------------------------------------------------------------ #
    # Attribute delegation
    # ------------------------------------------------------------------ #


    def __getattr__(self, name):
        """Delegate fitted-attribute lookups to the underlying model."""
        if name.startswith('_') or name in ('model_', 'task_'):
            raise AttributeError(name)
        try:
            model = object.__getattribute__(self, 'model_')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                "The model has not been fitted yet."
            )
        return getattr(model, name)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #


    def predict(self, X):
        """
        Predict target values (regression) or class labels (classification).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.

        Returns
        -------
        ndarray, shape (n,)
        """
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)

    @available_if(only_for('classification'))
    def predict_proba(self, X):
        """
        Predict class probabilities.  Only available for classification.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.

        Returns
        -------
        ndarray, shape (n, 2)

        Raises
        ------
        AttributeError if task is regression.
        """
        check_is_fitted(self, 'model_')
        if self.task_ != 'classification':
            raise AttributeError(
                "predict_proba is only available when task_ == 'classification'."
            )
        return self.model_.predict_proba(X)

    @available_if(only_for('classification'))
    def decision_function(self, X):
        """
        Compute log-odds scores.  Only available for classification.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.

        Returns
        -------
        ndarray, shape (n,)

        Raises
        ------
        AttributeError if task is regression.
        """
        check_is_fitted(self, 'model_')
        if self.task_ != 'classification':
            raise AttributeError(
                "decision_function is only available when task_ == 'classification'."
            )
        return self.model_.decision_function(X)

    @available_if(only_for('regression'))
    def predict_interval(self, X, alpha=0.05):
        """
        Prediction interval incorporating missing-feature uncertainty.
        Only available for regression.

        Parameters
        ----------
        X     : array-like, shape (n, p).  NaN allowed.
        alpha : float, significance level (default 0.05 for 95% interval).

        Returns
        -------
        lower, upper : ndarray, shape (n,)

        Raises
        ------
        AttributeError if task is classification.
        """
        check_is_fitted(self, 'model_')
        if self.task_ != 'regression':
            raise AttributeError(
                "predict_interval is only available when task_ == 'regression'."
            )
        return self.model_.predict_interval(X, alpha=alpha)


    def score(self, X, y):
        """
        R² (regression) or accuracy (classification) on non-NaN targets.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)

        Returns
        -------
        float
        """
        check_is_fitted(self, 'model_')
        return self.model_.score(X, y)


    def summary(self, alpha=0.05):
        """
        Print the model summary (delegates to the underlying model).

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self, 'model_')
        return self.model_.summary(alpha=alpha)
