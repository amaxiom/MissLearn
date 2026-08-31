"""
MissLearn._logistic
----------------
Full Information Maximum Likelihood (FIML) logistic regression.

Models the joint distribution of the observed data as:
    P(Y=1 | X)           -- logistic link with linear predictor
    P(X_1,...,X_p)       -- multivariate normal on predictors

For each observation, the FIML log-likelihood contribution is:
    log P(y_i, X_obs_i) = log P(y_i | X_obs_i) + log P(X_obs_i)

where P(y_i | X_obs_i) is computed by marginalizing over missing X:
    P(y=1 | X_obs) = integral sigma(beta @ x) P(x_mis | x_obs) d x_mis

The key mathematical simplification that makes this tractable:
Because sigma(beta @ x) depends on the missing features only through
the scalar s = beta_mis @ x_mis, and s is normally distributed (being
a linear combination of a normal vector), the integral reduces to 1D
regardless of how many features are missing:
    P(y=1 | X_obs) = E_t[sigma(a + t)],  t ~ N(0, v)
    a = beta_0 + beta_obs @ x_obs + beta_mis @ mu_mis|obs
    v = beta_mis @ Sigma_mis|obs @ beta_mis

This 1D integral is evaluated exactly via Gauss-Hermite quadrature.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._base import MissBase
from ._conformance import public_parameter_name, check_penalty
from ._copula import RankNormalTransformer, needs_copula
from ._utils import (
    sigmoid, pack_cholesky, unpack_cholesky, mvn_logpdf, mvn_logpdf_batch,
    conditional_normal_params, integrate_logistic_normal,
    logistic_normal_with_grads, numerical_hessian, feature_scale,
    standard_errors_from_variance
)


class MissLogistic(ClassifierMixin, MissBase):
    """
    FIML logistic regression with native missing data support.

    Jointly estimates logistic regression coefficients and the multivariate
    normal distribution of predictors using Full Information Maximum
    Likelihood.  No imputation or case deletion is performed at any stage.
    Missing X at inference time is handled via the same conditional normal
    expectation used during training.

    Parameters
    ----------
    max_iter : int
        Maximum optimizer iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    n_quadrature : int
        Number of Gauss-Hermite nodes for the 1-D logistic-normal integral
        (default 20). Applies only where the marginalised variance is small,
        which is where Gauss-Hermite is accurate to about 1e-11; above that
        the integral is taken by a step-plus-remainder rule that does not
        consult this argument. Raising it no longer changes the fit, which is
        the point: it used to, by up to 0.16 in a coefficient.
    compute_se : bool
        If True (default), compute standard errors after fitting.
    l2_reg : float
        L2 penalty on slope coefficients (default 0.01).  Prevents divergence
        under quasi-separation.  Set to 0 for exact unpenalized FIML.
        The penalty is applied on the internally standardized scale, so it is
        unit-free: the value does not have to be retuned when the features are
        re-expressed in different units, and at the default it is negligible
        for well-scaled data (the standardized coefficients are O(1)) while
        still bounding the quasi-separated directions.
    copula : bool or 'auto', default 'auto'
        Apply a Gaussian copula transform to each feature before fitting,
        relaxing the multivariate normality assumption on X (default False).
    warm_start : bool
        If True, reuse the fitted parameter vector from the previous call to
        fit() as the optimizer starting point instead of re-initializing from
        complete cases.  Speeds up repeated fitting on similar data (e.g.,
        successive cross-validation folds) by 20 to 40%.  The starting point is
        stored in ``_theta_opt_`` after each fit (default False).

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
        Log-odds coefficients for each predictor.
    intercept_ : float
        Log-odds intercept.
    se_ : ndarray, shape (p+1,)
        Standard errors: [se_intercept, se_coef_0, ..., se_coef_{p-1}].
    pvalues_ : ndarray, shape (p+1,)
    z_stats_ : ndarray, shape (p+1,)
    odds_ratios_ : ndarray, shape (p+1,)
        exp(coef) for each term including intercept.
    coef_std_ : ndarray, shape (p,)
        Standardized coefficients: coef_j * sigma_Xj.
    mu_X_ : ndarray, shape (p,)
        Fitted mean of the predictor distribution.
    Sigma_X_ : ndarray, shape (p, p)
        Fitted covariance of the predictor distribution.
    classes_ : ndarray, shape (2,)
    loglik_ : float
    aic_, bic_ : float
    """

    def __init__(self, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 n_quadrature=20, compute_se=True, l2_reg=0.01, copula='auto',
                 warm_start=False):
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_quadrature = n_quadrature
        self.compute_se = compute_se
        self.l2_reg = l2_reg
        self.copula = copula
        self.warm_start = warm_start

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    def _pack_params(self, beta, mu_X, Sigma_X):
        """
        Flatten (beta, mu_X, Sigma_X) into the unconstrained parameter vector:
            theta = [beta (p+1), mu_X (p), L_X_vec (p*(p+1)//2)]
        """
        L_vec, _ = pack_cholesky(Sigma_X)
        return np.concatenate([beta, mu_X, L_vec])

    # ------------------------------------------------------------------ #
    # Missingness pattern grouping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_patterns(X):
        """
        Group row indices by (obs_idx, mis_idx) missingness pattern.

        Returns a list of (obs_idx, mis_idx, row_idxs) triples.  Rows with
        no observed X features are excluded.

        Precomputing patterns once before optimization allows:
        - batched mvn_logpdf via a single Cholesky per pattern
        - Sigma_c and v (quadrature variance) computed once per pattern
          rather than once per observation
        """
        p = X.shape[1]
        all_idx = np.arange(p)
        groups = defaultdict(list)
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
    # Probability computation for one observation (used in predict_proba)
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
        conditional NLL (theta = [intercept, beta]).

        If hess_inv is provided (the optimizer's approximate inverse Hessian),
        it is used directly, avoiding numerical Hessian computation.

        Returns
        -------
        se : ndarray, shape (p+1,)  -- [se_intercept, se_coef_0, ...]
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

        # The first p+1 diagonal entries correspond to beta directly
        se_all = standard_errors_from_variance(Var_theta)
        return se_all[:p + 1]

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the FIML logistic regression model.

        X may contain NaN values at arbitrary positions.  y must be
        binary (0/1) and may also contain NaN (those rows inform the
        predictor distribution but not the outcome parameters).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    Binary labels {0, 1}; NaN allowed.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        X, y = self._canonical_fit_order(X, y)
        # Validated here rather than in each subclass: MissRidgeClassifier
        # is a thin subclass that aliases alpha onto l2_reg and inherits
        # this fit, so a check written into its own class body would
        # never run. That inheritance is why it was missed once before.
        check_penalty(self.l2_reg,
                      public_parameter_name(self, 'l2_reg', 'alpha'),
                      type(self).__name__)
        self._store_fit_metadata(X, y)
        n, p = X.shape

        # Resolve copula='auto' to a concrete boolean and store the decision.
        # y is binary and excluded from the normality check.
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        # Gaussian copula: map each X marginal to N(0,1) before FIML.
        # y is binary and is never transformed.
        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X = self._copula_X_.transform(X)

        # Encode class labels and validate
        y_vals = y[~np.isnan(y)]
        _classes = np.unique(y_vals)
        # Cast to int only when the labels are integer-valued; astype(int)
        # on fractional labels (e.g. {0.5, 1.5}) would truncate them and
        # break the classes_[1] binarization below.
        if _classes.size and np.all(_classes == np.floor(_classes)):
            _classes = _classes.astype(int)
        self.classes_ = _classes
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissLogistic requires exactly 2 classes; found {self.classes_}."
            )

        # Binarize against classes_[1] so any two labels work (e.g. {1, 2}
        # or {-1, 1}), keeping the likelihood's y==1 encoding consistent
        # with predict's classes_-based mapping.  NaN (missing y) preserved.
        y = np.where(np.isnan(y), np.nan,
                     (y == self.classes_[1]).astype(float))

        # Internal standardisation of X (the convention of the penalized
        # models; see MissLASSORegressor.fit).  The L2 penalty below is only
        # scale-coherent on standardized data: on raw units a feature measured
        # in small units needs a correspondingly large coefficient, the penalty
        # charges for that magnitude rather than for model complexity, and the
        # optimum shrinks that coefficient toward zero purely because of the
        # units.  y is binary and the model is on the logit scale, so only the
        # fixed effects transform; every public attribute is converted back to
        # the original feature scale after the fit.
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
        # failing.  The FIML likelihood below still integrates over every
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

        # Initial beta from sklearn logistic regression on the seed rows.
        # If the seed happens to contain a single class, fall back to zeros.
        from sklearn.linear_model import LogisticRegression
        if len(np.unique(y_seed)) >= 2:
            lr0 = LogisticRegression(max_iter=1000, solver='lbfgs')
            lr0.fit(X_seed, y_seed)
            beta0 = np.concatenate([lr0.intercept_, lr0.coef_.ravel()])
        else:
            beta0 = np.zeros(p + 1)

        # --- Stage 1: X-moments by EM (full-information) -----------------
        # Estimating the MVN nuisance parameters once removes p + p(p+1)/2
        # dimensions from the optimizer below, whose finite-difference
        # gradients otherwise dominate the runtime.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        from ._utils import prep_conditional_terms
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(X)
        mu_X_opt    = _fitter.mu_
        Sigma_X_opt = _fitter.Sigma_

        # --- Stage 2: conditional ML for (intercept, beta) ---------------
        # Pattern-constant terms are precomputed (vectorized GH quadrature
        # per pattern), so each NLL call is O(n p).  Rows with observed y
        # but no observed X contribute their marginal P(y) term through the
        # all-missing pattern (F = mu_X, Sigma_c = Sigma_X).
        _obs_y_rows = ~np.isnan(y)
        y_y = y[_obs_y_rows]
        F_y, cond_groups = prep_conditional_terms(X[_obs_y_rows], mu_X_opt,
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
        _bad_grad = np.zeros(p + 1)

        def _nll_grad(theta_r):
            intercept = theta_r[0]
            beta = theta_r[1:]

            Mb  = M @ beta                                    # (G, p)
            v_g = np.maximum(Mb @ beta, 0.0)                  # (G,)
            if not np.all(np.isfinite(v_g)):
                return np.inf, _bad_grad
            a = intercept + F_y @ beta                        # (n,)
            v_row = v_g[group_id]                             # (n,)

            # One rule for fitting and for predicting. Gauss-Hermite alone
            # cannot hold this integral once v_g grows: it is a step of width
            # about 1/sqrt(2 v_g) in the quadrature variable, and the error
            # reaches 0.004 by v = 25 whatever the node count. v_g is the
            # variance of the missing features' linear contribution, so it
            # grows with the coefficients, which is to say the rule was least
            # accurate exactly where the signal was strongest.
            p1, dp_da, dp_dv = logistic_normal_with_grads(
                a, v_row, self.n_quadrature)
            p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)

            nll = -(float(np.sum(np.log(p1[_is_pos])))
                    + float(np.sum(np.log1p(-p1[~_is_pos]))))
            if self.l2_reg > 0:
                nll += 0.5 * self.l2_reg * float(np.dot(beta, beta))

            # Analytic gradient. The helper supplies dp1/da and dp1/dv; the
            # chain rule out to (intercept, beta) is this estimator's own.
            d     = np.where(_is_pos, -1.0 / p1, 1.0 / (1.0 - p1))  # dNLL/dp1
            g_a   = d * dp_da                             # per-row dNLL/da
            g_vg  = np.bincount(group_id, weights=d * dp_dv, minlength=G)
            g_int  = float(np.sum(g_a))
            g_beta = F_y.T @ g_a + 2.0 * (g_vg[:, None] * Mb).sum(axis=0)
            if self.l2_reg > 0:
                g_beta = g_beta + self.l2_reg * beta
            return nll, np.concatenate([[g_int], g_beta])

        def _reduced_nll(theta_r):
            return _nll_grad(theta_r)[0]

        theta0 = beta0

        if self.warm_start and hasattr(self, '_theta_opt_'):
            if len(self._theta_opt_) == len(theta0):
                theta0 = self._theta_opt_
            # else: p changed between fits; fall back to fresh initialisation
        # else: theta0 is already set from complete-case initialisation above

        # Optimize
        result = minimize(
            fun=_nll_grad,
            x0=theta0,
            jac=True,
            method=self.method,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'gtol': self.tol * 1e-2
            }
        )

        theta_opt = result.x
        self._theta_opt_ = result.x
        beta_opt = theta_opt

        # Standardised-scale estimates, kept for the prediction paths
        b0_std   = float(beta_opt[0])
        beta_std = beta_opt[1:]
        self._b0_std_      = b0_std
        self._beta_std_    = beta_std
        self._mu_X_std_    = mu_X_opt
        self._Sigma_X_std_ = Sigma_X_opt
        # [intercept, coef] on the standardised scale, used by predict_proba
        # and decision_function (both run in the standardised space).
        self._beta_full = beta_opt

        # --- convert public parameters back to the original feature scale ---
        # The logit is affine in X, so eta = b0_std + beta_std @ (x - mx)/sx
        # equals intercept_ + coef_ @ x exactly.
        self.coef_ = beta_std / _sx
        self.intercept_ = float(b0_std - float(self.coef_ @ _mx))
        self.mu_X_ = _mx + _sx * mu_X_opt
        self.Sigma_X_ = Sigma_X_opt * np.outer(_sx, _sx)

        # Information criteria.  result.fun includes the L2 penalty; add it
        # back so loglik_/AIC/BIC reflect the data likelihood only (same
        # convention as MissRidgeRegressor).  The full-information
        # log-likelihood is the conditional part plus the X-marginal part
        # (from the EM stage), plus the Jacobian of the X standardisation
        # which moves it onto the raw scale so AIC/BIC stay comparable across
        # unit choices (y is binary, so there is no y term).
        penalty = 0.5 * self.l2_reg * float(np.dot(beta_std, beta_std))
        _marg_ll = _mvn_loglik(X, mu_X_opt, Sigma_X_opt)
        _n_obs_col = np.sum(~np.isnan(X), axis=0)
        _jacobian = -float(_n_obs_col @ np.log(_sx))
        self.loglik_ = -(float(result.fun) - penalty) + _marg_ll + _jacobian
        # Parameter count: (intercept, beta) + MVN moments
        n_params = (p + 1) + p + p * (p + 1) // 2
        self.aic_ = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_ = n_params * np.log(n) - 2.0 * self.loglik_
        self.converged_ = bool(result.success)
        # scikit-learn expects n_iter_ wherever max_iter is exposed.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # Odds ratios. Overflow to inf is the intended reading rather than a
        # fault: a coefficient past about 709 means the predictor separates
        # the classes, where the odds ratio really is unbounded and the MLE
        # does not exist. The warning is silenced so that it cannot mask a
        # real one; the value is left alone, because clipping it would report
        # a finite odds ratio for a variable that has none.
        beta_all = np.concatenate([[self.intercept_], self.coef_])
        with np.errstate(over='ignore'):
            self.odds_ratios_ = np.exp(beta_all)

        # SEs always come from a numerical Hessian of the reduced NLL: with
        # analytic gradients L-BFGS-B converges in a handful of iterations,
        # leaving its low-rank inverse-Hessian approximation near the
        # identity; useless for inference.  The reduced NLL is vectorised,
        # so the exact numerical Hessian costs only milliseconds.
        if self.compute_se:
            se_std = self._compute_se(theta_opt, _reduced_nll, None)
            # Delta-method rescale to raw units: exact for the slopes; the
            # intercept uses a diagonal approximation (covariances with the
            # slopes ignored, as elsewhere in the penalized models).
            se_beta = se_std[1:] / _sx
            se_int = np.sqrt(
                se_std[0] ** 2 + float(np.sum(((_mx / _sx) * se_std[1:]) ** 2))
            )
            self.se_ = np.concatenate([[se_int], se_beta])
        else:
            self.se_ = np.full(p + 1, np.nan)

        safe_se = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = beta_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        # Standardized coefficients: coef_j * sigma_Xj (raw scale throughout,
        # so the product is invariant to the internal standardisation)
        sigma_X = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        self.coef_std_ = self.coef_ * sigma_X
        self.sigma_X_ = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_proba(self, X):
        """
        Estimate class probabilities P(Y=0 | X_obs) and P(Y=1 | X_obs).

        For complete rows: direct sigmoid evaluation.
        For rows with missing X: Gauss-Hermite quadrature over the
        conditional distribution of X_mis | X_obs.

        No imputation is performed.

        Parameters
        ----------
        X : array-like, shape (n_new, p).  NaN allowed.

        Returns
        -------
        proba : ndarray, shape (n_new, 2)
            Column 0 is P(Y=0); column 1 is P(Y=1).
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

        # Rows with NO observed features: use marginal P(Y=1) by integrating
        # over the full prior X ~ N(mu_X, Sigma_X).
        all_missing_mask = np.isnan(X).all(axis=1)
        if all_missing_mask.any():
            v_full = float(beta_X @ Sigma_X @ beta_X)
            p1[all_missing_mask] = integrate_logistic_normal(
                float(beta_0 + beta_X @ mu_X), v_full, self.n_quadrature
            )

        patterns = self._group_patterns(X)
        for obs_idx, mis_idx, row_idxs in patterns:
            x_obs_batch = X[np.ix_(row_idxs, obs_idx)]  # (n_group, |obs|)
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
                K = np.linalg.solve(Sigma_obs, Sigma_mo.T).T  # (|mis|, |obs|)
                Sigma_c = Sigma_mm - K @ Sigma_mo.T
                Sigma_c = 0.5 * (Sigma_c + Sigma_c.T)
                v = float(beta_mis @ Sigma_c @ beta_mis)

                mu_mis = mu_X[mis_idx]
                mu_c_batch = mu_mis + (x_obs_batch - mu_obs) @ K.T  # (n_group, |mis|)
                a_vals = beta_0 + x_obs_batch @ beta_obs + mu_c_batch @ beta_mis

                p1[row_idxs] = integrate_logistic_normal(a_vals, v, self.n_quadrature)

        p1 = np.clip(p1, 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : array-like, shape (n_new, p).  NaN allowed.

        Returns
        -------
        y_pred : ndarray, shape (n_new,), dtype int
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Compute log-odds (linear predictor) for complete cases.
        For rows with missing X, returns the marginal linear predictor
        using E[X_mis | X_obs] in place of the missing values.

        Parameters
        ----------
        X : array-like, shape (n_new, p).  NaN allowed.

        Returns
        -------
        log_odds : ndarray, shape (n_new,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)
        # Standardised space of the fit (see predict_proba)
        X = (X - self._x_mean_) / self._x_scale_
        beta = self._beta_full
        log_odds = np.empty(X.shape[0])

        for i, x_i in enumerate(X):
            obs_mask = ~np.isnan(x_i)
            if obs_mask.all():
                log_odds[i] = beta[0] + beta[1:] @ x_i
            elif not obs_mask.any():
                log_odds[i] = float(beta[0] + beta[1:] @ self._mu_X_std_)
            else:
                obs_idx = np.where(obs_mask)[0]
                mis_idx = np.where(~obs_mask)[0]
                x_obs = x_i[obs_mask]
                mu_c, _ = conditional_normal_params(
                    self._mu_X_std_, self._Sigma_X_std_, obs_idx, mis_idx, x_obs
                )
                x_filled = x_i.copy()
                x_filled[~obs_mask] = mu_c
                log_odds[i] = beta[0] + beta[1:] @ x_filled
        return log_odds

    def score(self, X, y):
        """
        Classification accuracy on the subset where y is not NaN.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)

        Returns
        -------
        float, accuracy in [0, 1]
        """
        check_is_fitted(self)
        X, y = self._validate_and_convert(X, y)
        obs = ~np.isnan(y)
        y_pred = self.predict(X[obs])
        return float(np.mean(y_pred == y[obs]))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self, alpha=0.05):
        """
        Print a comprehensive model summary.

        Sections:
            - Dataset and missingness statistics
            - Model fit: log-likelihood, AIC, BIC
            - Coefficient table with SEs, z-stats, p-values, odds ratios, CI
            - Feature importances (standardized absolute coefficients)

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 80
        print()
        print('=' * W)
        print('MissLogistic  --  FIML Logistic Regression'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Classes         : {self.classes_}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
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

        # Extra column: odds ratio
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
