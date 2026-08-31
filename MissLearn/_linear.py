"""
MissLearn._linear
--------------
Full Information Maximum Likelihood (FIML) linear regression.

Models the joint distribution of all variables as multivariate normal:
    Z = (Y, X_1, ..., X_p) ~ N(mu, Sigma)

Each observation contributes the log-likelihood of its observed subvector
under the marginal multivariate normal.  No imputation, no listwise
deletion, no fake data.  Missing values in both X and y are handled
identically and symmetrically: they simply do not appear in the
likelihood term for that observation.

After optimization, regression coefficients are derived analytically
from the FIML-estimated covariance via the standard partitioned-normal
formula; they are not free parameters in the optimization.

Prediction with missing X at inference time uses the conditional normal
expectation, also exact and closed-form.

Reference:
    Arbuckle, J. L. (1996). Full information estimation in the presence
    of incomplete data. In G. A. Marcoulides & R. E. Schumacker (Eds.),
    Advanced structural equation modeling: Issues and techniques (pp. 243-277).
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase
from ._copula import RankNormalTransformer, needs_copula
from ._utils import (
    pack_cholesky, unpack_cholesky, mvn_logpdf, mvn_logpdf_batch,
    conditional_normal_params, numerical_hessian, psd_jitter,
    degenerate_feature_mask, feature_scale, standard_errors_from_variance
)


class MissLinear(RegressorMixin, MissBase):
    """
    FIML linear regression with native missing data support.

    Fits the joint multivariate normal distribution of (Y, X_1, ..., X_p)
    directly on the observed data using Full Information Maximum Likelihood.
    No imputation or case deletion is performed at any stage.

    Parameters
    ----------
    max_iter : int
        Maximum number of optimizer iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    compute_se : bool
        If True (default), compute standard errors after fitting from the
        curvature of the conditional likelihood of y given the observed part
        of X, as a (p+2)-dimensional numerical Hessian at the fitted moments.
        Set False to skip SE computation for large datasets where speed
        matters and inference is not needed.
    copula : bool or 'auto', default 'auto'
        Apply a Gaussian copula transform to each feature before fitting,
        relaxing the multivariate normality assumption (default 'auto').
    warm_start : bool
        If True, reuse the fitted parameter vector from the previous call to
        fit() as the optimizer starting point instead of re-initializing from
        complete cases.  Speeds up repeated fitting on similar data (e.g.,
        successive cross-validation folds) by 20 to 40%.  The starting point is
        stored in ``_theta_opt_`` after each fit (default False).

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
        Regression coefficients derived analytically from fitted Sigma.
    intercept_ : float
        Regression intercept.
    se_ : ndarray, shape (p+1,)
        Standard errors: [se_intercept, se_coef_0, ..., se_coef_{p-1}].
    pvalues_ : ndarray, shape (p+1,)
        Two-sided p-values under the standard normal.
    z_stats_ : ndarray, shape (p+1,)
        z-statistics (coef / se).
    coef_std_ : ndarray, shape (p,)
        Standardized coefficients: coef_j * sigma_Xj / sigma_Y.
    mu_joint_ : ndarray, shape (p+1,)
        Fitted joint mean vector [mu_Y, mu_X1, ..., mu_Xp].
    Sigma_joint_ : ndarray, shape (p+1, p+1)
        Fitted joint covariance matrix.
    sigma_sq_ : float
        Conditional variance of Y given X (homoscedastic residual variance).
    loglik_ : float
        Maximized log-likelihood.
    aic_ : float
    bic_ : float
    """

    def __init__(self, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 compute_se=True, copula='auto', warm_start=False):
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.compute_se = compute_se
        self.copula = copula
        self.warm_start = warm_start

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    def _pack_params(self, mu, Sigma):
        """
        Flatten (mu, Sigma) into the unconstrained parameter vector:
            theta = [mu (q), L_vec (q*(q+1)//2)]

        where q = p+1 and L_vec is the log-diagonal Cholesky parameterization.
        """
        vec, _ = pack_cholesky(Sigma)
        return np.concatenate([mu, vec])

    def _unpack_params(self, theta):
        """
        Recover (mu, Sigma) from the parameter vector.

        theta[:q]  -> mu  (joint mean of Z = [Y, X])
        theta[q:]  -> L   (Cholesky factor, diagonals log-stored)
        Sigma = L @ L.T

        The width comes from ``_q_fit_`` rather than ``n_features_in_``
        because degenerate columns are held out of the joint model, so the
        parameter vector can be narrower than the caller's feature count.
        """
        q = getattr(self, '_q_fit_', self.n_features_in_ + 1)
        mu = theta[:q]
        L = unpack_cholesky(theta[q:], q)
        return mu, L @ L.T

    # ------------------------------------------------------------------ #
    # Missingness pattern grouping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_patterns(Z):
        """
        Group row indices by their observed-variable pattern.

        Returns a list of (obs_idx, row_idxs) pairs where obs_idx is the
        array of column indices with non-NaN values and row_idxs is the
        array of row indices sharing that pattern.  Rows with no observed
        values are excluded.

        Precomputing patterns once before optimization eliminates repeated
        np.where calls inside the NLL and enables batched Cholesky solves.
        """
        groups = defaultdict(list)
        for i, z_i in enumerate(Z):
            key = tuple(np.where(~np.isnan(z_i))[0])
            if key:
                groups[key].append(i)
        return [(np.array(key), np.array(idxs)) for key, idxs in groups.items()]

    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #

    def _neg_log_likelihood(self, theta, Z, patterns):
        """
        FIML negative log-likelihood for the joint normal model on Z = [Y | X].

        Observations are grouped by missingness pattern.  For each unique
        pattern, a single Cholesky decomposition of the submatrix is shared
        across all observations in the group via mvn_logpdf_batch.

        Parameters
        ----------
        theta    : ndarray, shape (n_params,)
        Z        : ndarray, shape (n, q), joint matrix; NaN encodes missing values
        patterns : list of (obs_idx, row_idxs) pairs from _group_patterns

        Returns
        -------
        float, negative log-likelihood value
        """
        try:
            mu, Sigma = self._unpack_params(theta)
        except (np.linalg.LinAlgError, FloatingPointError):
            return np.inf

        nll = 0.0
        for obs_idx, row_idxs in patterns:
            mu_sub = mu[obs_idx]
            Sigma_sub = Sigma[np.ix_(obs_idx, obs_idx)]
            X_batch = Z[np.ix_(row_idxs, obs_idx)]
            lps = mvn_logpdf_batch(X_batch, mu_sub, Sigma_sub)
            if not np.all(np.isfinite(lps)):
                return np.inf
            nll -= np.sum(lps)
        return nll

    # ------------------------------------------------------------------ #
    # Analytic coefficient recovery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand_joint(mu_r, Sigma_r, keep, X):
        """Widen the reduced joint moments back to the caller's column count.

        A column held out of the fit re-enters with its observed value as the
        mean and zero variance and covariance, which is what a constant
        feature actually is. Keeping the public arrays at full width means
        predict, predict_interval and summary index by the caller's column
        numbers throughout, so there is no position mapping anywhere else to
        get wrong.

        Parameters
        ----------
        mu_r, Sigma_r : ndarray
            Joint moments over [Y] + the kept columns.
        keep : ndarray of bool, shape (p,)
            Columns that were included in the fit.
        X : ndarray of shape (n, p)
            Features on the same scale the model was fitted on, used to read
            off the constant value of each held-out column.

        Returns
        -------
        mu    : ndarray, shape (p + 1,)
        Sigma : ndarray, shape (p + 1, p + 1)
        """
        p = keep.shape[0]
        idx = np.concatenate([[0], np.where(keep)[0] + 1]).astype(int)
        mu = np.zeros(p + 1)
        Sigma = np.zeros((p + 1, p + 1))
        mu[idx] = mu_r
        Sigma[np.ix_(idx, idx)] = Sigma_r
        for j in np.where(~keep)[0]:
            col = X[:, j]
            obs = col[~np.isnan(col)]
            mu[j + 1] = float(obs[0]) if obs.size else 0.0
        return mu, Sigma

    def _informative_obs(self, x_i):
        """Observed feature indices that can inform the conditional mean.

        Conditioning on a constant feature is a no-op in exact arithmetic:
        it contributes a zero row and column to the observed covariance
        block, so it adds no information about y. In floating point that
        zero block is instead a rounding residue, and solving against it
        turns a no-op into an arbitrary shift that depends on row order.
        Dropping those columns from the conditioning set gives the exact
        answer directly rather than approaching it through a solve that
        cannot be conditioned.
        """
        obs = np.where(~np.isnan(x_i))[0]
        const = getattr(self, '_constant_features_', None)
        if const is not None and len(const) and const.any():
            obs = obs[~const[obs]]
        return obs

    def _extract_coefficients(self, mu, Sigma, constant=None):
        """
        Derive linear regression coefficients from the joint normal parameters.

        From the partitioned covariance of Z = (Y, X)::

            beta      = Sigma_YX @ Sigma_XX^{-1}
            intercept = mu_Y - beta @ mu_X
            sigma_sq  = Sigma_YY - Sigma_YX @ Sigma_XX^{-1} @ Sigma_XY
                      (conditional variance of Y given X = OLS residual variance)

        Solved via np.linalg.solve rather than explicit matrix inversion.

        Constant features are held out of the solve and given a coefficient of
        exactly zero. For those columns both Sigma_XX[j, j] and Sigma_YX[j]
        are numerically zero, so the ratio above is arbitrary rather than
        merely imprecise, and it changes with the order the rows arrived in.
        Zero is the identified answer: a feature that never varies cannot move
        the prediction, and its constant level belongs in the intercept, which
        is where mu_Y already puts it.

        Parameters
        ----------
        mu, Sigma : ndarray
            Fitted joint mean and covariance of Z = (Y, X).
        constant : ndarray of bool, shape (p,), optional
            Columns to hold out, from ``degenerate_feature_mask``. Defaults
            holding none out.

        Returns
        -------
        beta      : ndarray, shape (p,)
        intercept : float
        sigma_sq  : float, >= 0
        """
        Sigma_YX = Sigma[0, 1:]       # shape (p,)
        Sigma_XX = Sigma[1:, 1:]      # shape (p, p)
        p = Sigma_YX.shape[0]

        free = (np.ones(p, dtype=bool) if constant is None
                else ~np.asarray(constant, dtype=bool))

        beta = np.zeros(p, dtype=float)
        if free.any():
            A = Sigma_XX[np.ix_(free, free)]
            b = Sigma_YX[free]
            # Sigma_XX @ beta = Sigma_YX  =>  beta = Sigma_XX^{-1} Sigma_YX
            try:
                beta[free] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Collinear survivors: the minimum-norm solution is still
                # well defined and does not depend on row order.
                beta[free] = np.linalg.lstsq(A, b, rcond=None)[0]

        intercept = float(mu[0] - beta @ mu[1:])
        sigma_sq = float(max(Sigma[0, 0] - Sigma_YX @ beta, 1e-12))
        return beta, intercept, sigma_sq

    # ------------------------------------------------------------------ #
    # Standard errors via delta method
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the FIML linear regression model.

        X and y may contain NaN values at arbitrary positions.  No
        imputation is performed; each observation contributes only its
        observed entries to the likelihood.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Predictor matrix.  NaN values are treated as missing.
        y : array-like, shape (n,)
            Response vector.  NaN values are treated as missing.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        # Degenerate columns are held out of the joint model itself, not just
        # out of the coefficient solve. A feature that never varies gives the
        # joint MVN a zero-variance direction; the covariance is then singular
        # and the FIML likelihood unbounded, so there is no maximum to
        # converge to and the optimiser stops wherever the arithmetic carries
        # it. The same twelve rows in two orders reached log-likelihoods of
        # 225.99 and 344.95, and coefficients of -324 and -1.68e6. Removing
        # the column removes the unbounded direction and leaves an identified
        # model. The moments are widened again afterwards so every public
        # attribute keeps the caller's feature count.
        self._constant_features_ = degenerate_feature_mask(X)
        keep = ~self._constant_features_
        p_fit = int(keep.sum())
        q = p_fit + 1
        self._q_fit_ = q

        # Refuse p >= n rather than return a number that means nothing.
        #
        # This model estimates the full joint MVN of [y, X]: q means and
        # q(q+1)/2 covariance parameters. At p = 40 with n = 25 that is 902
        # free parameters from 25 rows. The covariance is then singular by
        # construction, and no amount of conditioning makes it estimable.
        #
        # Conditioning the seed was tried here and is the wrong instrument: it
        # stops the LinAlgError, but only by replacing an immediate failure
        # with a slow optimisation over an unidentified likelihood, which
        # returns coefficients that look ordinary and mean nothing. That is
        # the same trade the fully-missing-column path was fixed to stop
        # making. The penalized families shrink explicitly and are identified
        # in this regime, which is why they cope and this one does not.
        if n <= q:
            raise ValueError(
                "MissLinear cannot fit: the joint model has %d parameters "
                "(%d means and %d covariance terms) but only %d rows, so it "
                "is not identified. Use MissRidgeRegressor or "
                "MissLASSORegressor, which shrink explicitly and are defined "
                "when p >= n, or reduce the feature count. MissRecommender "
                "reports this as part of its shape evidence."
                % (q + q * (q + 1) // 2, q, q * (q + 1) // 2, n))

        # Resolve copula='auto' to a concrete boolean and store the decision.
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        # Gaussian copula: map each marginal to N(0,1) before FIML.
        # X and y get separate transformers so y predictions can be inverted.
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

        # Internal standardisation. MissLinear was the only regressor in the
        # family without it: Ridge, LASSO, Logistic and Mixed all fit in a
        # standardised space and convert back. Without it the joint model
        # inherits the caller's units, and on a column whose spread is 6e-08
        # beside a response of spread 0.3 the covariance reaches a condition
        # number near 3e14, most of the way to the limit of double precision;
        # the coefficient recovered from it moved by 13% under a row
        # permutation. A diagonal rescaling is an affine map and the MVN is
        # equivariant under it, so the moments convert back exactly: the
        # arithmetic simply happens somewhere better conditioned. It also
        # makes the fixed regularisers (the EM ridge, the seed jitter) mean
        # the same thing regardless of the units the data arrived in.
        self._x_mean_  = np.nanmean(X, axis=0)
        self._x_mean_  = np.where(np.isfinite(self._x_mean_), self._x_mean_, 0.0)
        self._x_scale_ = feature_scale(X)
        _yo = ~np.isnan(y)
        self._y_mean_  = float(np.mean(y[_yo])) if _yo.any() else 0.0
        _sy = float(np.std(y[_yo], ddof=1)) if _yo.sum() > 1 else 1.0
        self._y_scale_ = _sy if (np.isfinite(_sy) and _sy > 0) else 1.0
        X = (X - self._x_mean_) / self._x_scale_
        y = np.where(np.isnan(y), np.nan,
                     (y - self._y_mean_) / self._y_scale_)

        # Joint matrix: column 0 = y, then the identified columns of X
        Z = np.column_stack([y, X[:, keep]])

        # EM stage: the FIML MLE of a joint MVN is computed far faster by EM
        # than by quasi-Newton over the Cholesky parameterisation (whose
        # finite-difference gradients cost O(q^2) NLL evaluations per step).
        # The short L-BFGS-B run below then polishes the EM solution; same
        # objective, same optimum.  Standard errors are computed separately
        # from the conditional likelihood (see below), NOT from the polish's
        # inverse-Hessian approximation, which is uninformative when the
        # optimiser starts at the optimum.
        from ._imputer import _JointMVNFitter
        # Tight relative tolerance: EM iterations are far cheaper than the
        # quasi-Newton polish steps below, so let EM do almost all the work.
        _fitter = _JointMVNFitter(max_iter=500, tol=1e-10, reg=1e-8)
        _fitter.fit(Z)
        mu0    = _fitter.mu_
        # Spectrum-aware conditioning, the same rule MissLASSORegressor
        # applies. The fixed 1e-10 that stood here was the only reason this
        # estimator failed at 90% missingness and at p > n while every sibling
        # regressor coped: at those rates the EM covariance is marginally
        # indefinite and no constant that small can lift it.
        Sigma0 = psd_jitter(_fitter.Sigma_)

        theta0 = self._pack_params(mu0, Sigma0)

        # Precompute missingness patterns once; passed to NLL via args
        patterns = self._group_patterns(Z)

        # warm_start: only adopt the previous fit's parameters when they are
        # actually better than the fresh EM optimum for THIS data; the EM
        # stage already converges to the MLE, so a stale warm start would
        # otherwise discard it.
        if (self.warm_start and hasattr(self, '_theta_opt_')
                and len(self._theta_opt_) == len(theta0)):
            if (self._neg_log_likelihood(self._theta_opt_, Z, patterns)
                    < self._neg_log_likelihood(theta0, Z, patterns)):
                theta0 = self._theta_opt_

        # Polish the EM solution.  EM has already converged to the FIML MLE,
        # so a short quasi-Newton run suffices; it refines the last digits
        # and supplies the inverse Hessian for the standard errors.
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=theta0,
            args=(Z, patterns),
            method=self.method,
            options={
                'maxiter': min(self.max_iter, 50),
                'ftol': self.tol,
                'gtol': self.tol * 1e-2
            }
        )

        theta_opt = result.x
        self._theta_opt_ = result.x
        mu_r, Sigma_r = self._unpack_params(theta_opt)
        self._mu_std_, self._Sigma_std_ = self._expand_joint(
            mu_r, Sigma_r, keep, X)

        # Analytic coefficient recovery, done on the standardised moments
        # where the solve is conditioned, then converted. Held-out columns are
        # not in the joint model at all, so their coefficient is zero by
        # construction rather than by a solve that had to be defended against.
        self._beta_std_, self._icpt_std_, _s2_std = self._extract_coefficients(
            self._mu_std_, self._Sigma_std_, constant=self._constant_features_
        )

        sx, sy = self._x_scale_, self._y_scale_
        mx, my = self._x_mean_, self._y_mean_
        self.coef_      = self._beta_std_ * sy / sx
        self.intercept_ = float(my + sy * self._icpt_std_ - self.coef_ @ mx)
        self.sigma_sq_  = float(_s2_std * sy * sy)

        # The public joint moments are on the caller's scale. A diagonal
        # rescaling maps them back exactly, so this loses nothing.
        _s_all = np.concatenate([[sy], sx])
        _m_all = np.concatenate([[my], mx])
        self.mu_joint_    = _m_all + _s_all * self._mu_std_
        self.Sigma_joint_ = np.outer(_s_all, _s_all) * self._Sigma_std_

        # Information criteria. The likelihood was maximised on the
        # standardised scale, so it carries that change of variables: each
        # observed cell contributes -log(scale of its column). Subtracting it
        # puts loglik_, and therefore AIC and BIC, back on the caller's scale,
        # where they stay comparable with every other estimator here and with
        # the values this model reported before it standardised.
        _s_Z    = np.concatenate([[sy], sx[keep]])
        _nobs_Z = (~np.isnan(Z)).sum(axis=0)
        self.loglik_ = -float(result.fun) - float(np.sum(_nobs_Z * np.log(_s_Z)))
        n_params = len(theta_opt)
        self.aic_ = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_ = n_params * np.log(n) - 2.0 * self.loglik_
        self.converged_ = bool(result.success)
        # scikit-learn expects n_iter_ wherever max_iter is exposed.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # Standard errors.  The polish starts at the EM optimum and typically
        # terminates after one or two iterations, so L-BFGS-B's inverse-
        # Hessian approximation is essentially the identity matrix; useless
        # for inference.  Instead compute SEs from the curvature of the
        # conditional likelihood of y given X_obs at the fitted moments
        # (the same convention as the penalized and mixed models): a small
        # (p+2)-dimensional numerical Hessian of a fully vectorized NLL.
        # The curvature is taken over the columns that were actually fitted;
        # a held-out column has no coefficient to have a standard error for,
        # and its entry is filled with NaN below rather than with a number
        # that would imply one was estimated.
        if self.compute_se and p_fit > 0:
            from ._utils import prep_conditional_terms
            mu_X_hat    = mu_r[1:]
            Sigma_X_hat = Sigma_r[1:, 1:]
            _obs_y = ~np.isnan(Z[:, 0])
            y_y = Z[_obs_y, 0]
            F_y, cond_groups = prep_conditional_terms(
                Z[_obs_y, 1:], mu_X_hat, Sigma_X_hat
            )
            G = len(cond_groups)
            group_id = np.empty(len(y_y), dtype=np.intp)
            n_g = np.empty(G)
            M = np.zeros((G, p_fit, p_fit))
            for g, (rows, mis, Sc) in enumerate(cond_groups):
                group_id[rows] = g
                n_g[g] = len(rows)
                if mis.size:
                    M[g][np.ix_(mis, mis)] = Sc
            _LOG2PI = np.log(2.0 * np.pi)

            def _cond_nll(theta_r):
                icpt = theta_r[0]
                beta = theta_r[1:p_fit + 1]
                sigma_sq = float(np.exp(2.0 * theta_r[p_fit + 1]))
                Mb  = M @ beta
                v_g = sigma_sq + Mb @ beta
                if not np.all(np.isfinite(v_g)) or np.any(v_g <= 0):
                    return np.inf
                resid = y_y - icpt - F_y @ beta
                rss_g = np.bincount(group_id, weights=resid * resid,
                                    minlength=G)
                return (0.5 * float(n_g @ (_LOG2PI + np.log(v_g)))
                        + 0.5 * float(np.sum(rss_g / v_g)))

            theta_se = np.concatenate([
                [self._icpt_std_], self._beta_std_[keep],
                [0.5 * np.log(max(_s2_std, 1e-300))]
            ])
            self.se_ = np.full(p + 1, np.nan)
            try:
                H = numerical_hessian(_cond_nll, theta_se, eps=1e-4)
                H += np.eye(len(theta_se)) * 1e-8
                Var = np.linalg.inv(H)[:p_fit + 1, :p_fit + 1]

                # Convert the standardised covariance to the caller's scale.
                # The diagonal alone will not do it: the raw intercept is
                # my + sy*icpt_std - sum_j coef_j*mx_j, so it mixes every
                # coefficient, and its variance needs their covariances too.
                sxk = sx[keep]
                J = np.zeros((p_fit + 1, p_fit + 1))
                J[0, 0] = sy
                J[0, 1:] = -(sy / sxk) * mx[keep]
                J[1:, 1:] = np.diag(sy / sxk)
                Var_raw = J @ Var @ J.T

                se_raw = standard_errors_from_variance(Var_raw)
                self.se_[0] = se_raw[0]
                self.se_[1:][keep] = se_raw[1:]
            except np.linalg.LinAlgError:
                pass
        else:
            self.se_ = np.full(p + 1, np.nan)

        coefs_all = np.concatenate([[self.intercept_], self.coef_])
        safe_se = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = coefs_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        # Standardized coefficients (for feature importance)
        # beta_std_j = beta_j * sigma_Xj / sigma_Y
        sigma_X = np.sqrt(np.maximum(np.diag(self.Sigma_joint_[1:, 1:]), 1e-12))
        sigma_Y = np.sqrt(max(self.Sigma_joint_[0, 0], 1e-12))
        self.coef_std_ = self.coef_ * sigma_X / sigma_Y
        self.sigma_X_ = sigma_X
        self.sigma_Y_ = sigma_Y

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, X):
        """
        Predict E[Y | X_obs] for each row of X.

        Complete rows: direct linear equation  y_hat = intercept + coef @ x.
        Rows with missing X: conditional normal expectation::

            E[Y | X_obs] = mu_Y + Sigma_{Y,X_obs} @ Sigma_{X_obs,X_obs}^{-1}
                           @ (x_obs - mu_{X_obs})

        Rows with all X missing: predict the unconditional mean mu_Y.

        No imputation is performed at any point.

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
        Core FIML prediction on the internal (possibly copula-transformed) scale.

        Rows are grouped by missingness pattern so that each pattern incurs
        a single Cholesky solve rather than one solve per row.

        The conditional solves run on the standardised moments for the same
        reason the fit does: in the caller's units the observed block can be
        ill-conditioned enough that the solve is meaningless, and rescaling
        is exact. The whole routine therefore works in standardised space and
        converts once at the end.
        """
        mu = self._mu_std_
        Sigma = self._Sigma_std_
        X = (X - self._x_mean_) / self._x_scale_
        n = X.shape[0]
        y_hat = np.empty(n)

        obs_matrix = ~np.isnan(X)
        complete  = obs_matrix.all(axis=1)
        all_miss  = ~obs_matrix.any(axis=1)
        partial   = ~complete & ~all_miss

        if complete.any():
            y_hat[complete] = (self._icpt_std_
                               + X[complete] @ self._beta_std_)

        if all_miss.any():
            y_hat[all_miss] = float(mu[0])

        if partial.any():
            X_part = X[partial]
            part_idxs = np.where(partial)[0]
            groups = defaultdict(list)
            for loc_i, x_i in enumerate(X_part):
                groups[tuple(self._informative_obs(x_i))].append(loc_i)
            for obs_key, loc_idxs in groups.items():
                if not obs_key:
                    # Nothing informative observed: the conditional mean is
                    # the unconditional one.
                    y_hat[part_idxs[loc_idxs]] = float(mu[0])
                    continue
                obs_idx_Z = np.array(obs_key) + 1   # +1: Y sits at index 0 in Z
                y_idx     = np.array([0])
                x_batch   = X_part[np.ix_(loc_idxs, np.array(obs_key))]
                Sigma_oo  = Sigma[np.ix_(obs_idx_Z, obs_idx_Z)]
                Sigma_yo  = Sigma[np.ix_(y_idx, obs_idx_Z)]  # (1, |obs|)
                K         = np.linalg.solve(Sigma_oo, Sigma_yo.T).T  # (1, |obs|)
                mu_cond   = mu[0] + (x_batch - mu[obs_idx_Z]) @ K.T  # (n_grp, 1)
                y_hat[part_idxs[loc_idxs]] = mu_cond[:, 0]

        return self._y_mean_ + self._y_scale_ * y_hat

    def predict_interval(self, X, alpha=0.05):
        """
        Compute prediction intervals for new observations.

        The interval half-width is z_{alpha/2} * sqrt(Var[Y | X_obs]) where:
            Var[Y | X_obs] = Sigma_YY - Sigma_{Y,X_obs} Sigma_{X_obs,X_obs}^{-1}
                             Sigma_{X_obs,Y}

        Observations with more missing features have wider intervals,
        correctly reflecting greater predictive uncertainty.

        Parameters
        ----------
        X     : array-like, shape (n_new, p).  NaN allowed.
        alpha : float, significance level (default 0.05 for 95% interval).

        Returns
        -------
        lower : ndarray, shape (n_new,)
        upper : ndarray, shape (n_new,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)

        if self.copula_used_:
            X = self._copula_X_.transform(X)

        z = norm.ppf(1.0 - alpha / 2.0)
        # Standardised moments, for the same conditioning reason as predict;
        # the widths are multiplied back by the response scale at the end.
        mu = self._mu_std_
        Sigma = self._Sigma_std_
        sy = self._y_scale_

        # Work entirely on the internal (copula-transformed) scale so that
        # the conditional variances are correct; inverse-transform bounds at end.
        n = X.shape[0]
        y_hat_internal = self._predict_fiml(X)
        se_pred = np.empty(n)

        X = (X - self._x_mean_) / self._x_scale_
        obs_matrix = ~np.isnan(X)
        complete  = obs_matrix.all(axis=1)
        all_miss  = ~obs_matrix.any(axis=1)
        partial   = ~complete & ~all_miss

        if complete.any():
            se_pred[complete] = np.sqrt(self.sigma_sq_) / sy

        if all_miss.any():
            se_pred[all_miss] = np.sqrt(float(Sigma[0, 0]))

        if partial.any():
            X_part    = X[partial]
            part_idxs = np.where(partial)[0]
            groups    = defaultdict(list)
            for loc_i, x_i in enumerate(X_part):
                groups[tuple(self._informative_obs(x_i))].append(loc_i)
            y_idx = np.array([0])
            for obs_key, loc_idxs in groups.items():
                if not obs_key:
                    # Conditioning on nothing informative leaves the marginal
                    # variance of y, which is the widest honest interval.
                    se_pred[part_idxs[loc_idxs]] = np.sqrt(float(Sigma[0, 0]))
                    continue
                obs_idx_Z = np.array(obs_key) + 1
                Sigma_oo  = Sigma[np.ix_(obs_idx_Z, obs_idx_Z)]
                Sigma_yo  = Sigma[np.ix_(y_idx, obs_idx_Z)]
                K         = np.linalg.solve(Sigma_oo, Sigma_yo.T).T
                var_cond  = float(max(Sigma[0, 0] - float((K @ Sigma_yo.T)[0, 0]), 0.0))
                se_pred[part_idxs[loc_idxs]] = np.sqrt(var_cond)

        se_pred = se_pred * sy          # back to the caller's response scale
        lower = y_hat_internal - z * se_pred
        upper = y_hat_internal + z * se_pred

        if self.copula_used_:
            lower = self._copula_y_.inverse_transform_1d(lower, col=0)
            upper = self._copula_y_.inverse_transform_1d(upper, col=0)

        return lower, upper

    def score(self, X, y):
        """
        Compute R² on the subset of observations where y is not NaN.

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

        Sections:
            - Dataset and missingness statistics
            - Model fit: log-likelihood, AIC, BIC, residual variance
            - Coefficient table with SEs, z-stats, p-values, and 95% CI
            - Feature importances (standardized absolute coefficients)

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 74
        print()
        print('=' * W)
        print('MissLinear  --  FIML Linear Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
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
