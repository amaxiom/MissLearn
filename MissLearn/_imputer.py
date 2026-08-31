"""
_imputer.py  --  Multiple imputation via the joint MVN estimated by FIML.

MissImputer
    Fits a joint multivariate normal distribution to X (and optionally y)
    using Full Information Maximum Likelihood, then draws m complete datasets
    from the conditional distribution of each missing value given its observed
    neighbours.

    This is proper Bayesian multiple imputation: each draw reflects both the
    uncertainty about the missing value (conditional variance) and -- when
    posterior=True -- the uncertainty about the MVN parameters themselves
    (parameter uncertainty, implemented via a non-parametric bootstrap over
    the FIML estimates).

    The resulting m datasets can be:
    (a) Returned directly for downstream modelling.
    (b) Used to fit m copies of any estimator, then combined via Rubin's rules
        using MissImputer.combine().

Rubin's Rules (combine method)
    Given m estimates theta_1, ..., theta_m and their variances V_1, ..., V_m:
        theta_bar = mean(theta_i)
        W         = mean(V_i)           within-imputation variance
        B         = var(theta_i)        between-imputation variance
        T         = W + (1 + 1/m) * B  total variance
        df        = (m-1) * (1 + W/((1+1/m)*B))^2  degrees of freedom

Usage
-----
    from MissLearn import MissImputer
    import numpy as np

    imp = MissImputer(m=20, random_state=0)
    imp.fit(X)

    # Option A: get m complete datasets
    datasets = imp.transform(X)         # list of m ndarrays, shape (n, p)

    # Option B: mean imputation (for quick baselines only)
    X_mean = imp.transform_mean(X)      # single ndarray, conditional means

    # Option C: fit a downstream model on each imputed dataset and combine
    from sklearn.linear_model import Ridge
    results = imp.fit_transform_combine(
        X, y,
        estimator=Ridge(alpha=1.0),
        param='coef_',
        param_var='coef_var_',          # optional: attribute holding variances
    )
"""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted
from scipy import stats as sp_stats

from ._utils import conditional_normal_params, degenerate_feature_mask


# ---------------------------------------------------------------------------
# Joint MVN fitter (X-only, no response)
# ---------------------------------------------------------------------------

class _JointMVNFitter:
    """
    Fit a joint MVN to X using Full Information Maximum Likelihood (FIML).

    Uses available-case pairwise estimates for robustness, then refines
    using a regularised EM algorithm:
        E-step: fill missing entries with conditional expectations
        M-step: update mu and Sigma from the completed data

    Parameters
    ----------
    max_iter : int, default 200
    tol      : float, default 1e-6   convergence criterion on log-likelihood change
    reg      : float, default 1e-6   ridge added to Sigma diagonal for PD guarantee
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float    = 1e-6,
        reg: float    = 1e-6,
    ):
        self.max_iter = max_iter
        self.tol      = tol
        self.reg      = reg

    def fit(self, X: np.ndarray) -> '_JointMVNFitter':
        n, p = X.shape

        # --- pairwise available-case initialisation ---
        mu    = np.nanmean(X, axis=0)
        Sigma = np.zeros((p, p))
        # Compute upper triangle only then mirror (halves the work)
        for j in range(p):
            mask_j = ~np.isnan(X[:, j])
            m_jj   = mask_j.sum()
            Sigma[j, j] = np.var(X[mask_j, j], ddof=1) if m_jj > 1 else 0.0
            for k in range(j + 1, p):
                mask = mask_j & ~np.isnan(X[:, k])
                if mask.sum() < 2:
                    Sigma[j, k] = Sigma[k, j] = 0.0
                else:
                    Sigma[j, k] = Sigma[k, j] = np.cov(X[mask, j], X[mask, k])[0, 1]
        # ensure PD
        Sigma = Sigma + self.reg * np.eye(p)

        nan_mask = np.isnan(X)

        # Group rows by missingness pattern once: each EM iteration then
        # costs one solve per pattern (batched fill) instead of one per row.
        pattern_groups = []
        _by_mis = {}
        for i in range(n):
            _by_mis.setdefault(tuple(np.where(nan_mask[i])[0]), []).append(i)
        for mis_key, rows in _by_mis.items():
            mis = np.array(mis_key, dtype=int)
            if mis.size == 0:
                continue
            obs = np.setdiff1d(np.arange(p), mis)
            pattern_groups.append((np.array(rows), obs, mis))

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # --- E-step: compute conditional means and cross-products ---
            X_filled  = X.copy()
            # S_extra accumulates E[z_m z_m^T | z_obs] for the EM covariance
            # (accounts for variance in missing dims, not just means)
            S_extra   = np.zeros((p, p))  # sum of conditional covariances

            for rows, obs, mis in pattern_groups:
                if obs.size == 0:
                    # fully missing rows: fill with marginal mean
                    X_filled[np.ix_(rows, mis)] = mu[mis]
                    S_extra[np.ix_(mis, mis)] += len(rows) * Sigma[np.ix_(mis, mis)]
                    continue
                S_oo = Sigma[np.ix_(obs, obs)]
                S_mo = Sigma[np.ix_(mis, obs)]
                try:
                    K = np.linalg.solve(S_oo, S_mo.T).T
                except np.linalg.LinAlgError:
                    K = S_mo @ np.linalg.pinv(S_oo)
                Sigma_c = Sigma[np.ix_(mis, mis)] - K @ S_mo.T
                X_filled[np.ix_(rows, mis)] = (
                    mu[mis] + (X[np.ix_(rows, obs)] - mu[obs]) @ K.T
                )
                S_extra[np.ix_(mis, mis)] += len(rows) * Sigma_c

            # --- M-step: update mu and Sigma ---
            mu = X_filled.mean(axis=0)
            # demeaned outer products
            Z  = X_filled - mu
            Sigma_new = (Z.T @ Z + S_extra) / n + self.reg * np.eye(p)
            # symmetrise
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)

            # --- log-likelihood (complete-data proxy for convergence) ---
            try:
                ll = _mvn_loglik(X, mu, Sigma_new)
            except np.linalg.LinAlgError:
                ll = prev_ll

            prev_ll_old = prev_ll
            prev_ll = ll
            Sigma = Sigma_new
            # Relative criterion: an absolute tolerance on a log-likelihood
            # of magnitude ~n·p runs every fit to max_iter for nothing.
            if abs(ll - prev_ll_old) < self.tol * (1.0 + abs(ll)):
                break

        self.mu_     = mu
        self.Sigma_  = Sigma
        self.n_iter_ = iteration + 1
        return self

    def log_likelihood(self, X: np.ndarray) -> float:
        return _mvn_loglik(X, self.mu_, self.Sigma_)


def _mvn_loglik(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """FIML log-likelihood for X ~ N(mu, Sigma) with missing data.

    Rows are grouped by observed pattern so each pattern costs a single
    Cholesky decomposition (batched over its rows).
    """
    n, p = X.shape
    ll   = 0.0
    nan_mask = np.isnan(X)

    groups = {}
    for i in range(n):
        groups.setdefault(tuple(np.where(~nan_mask[i])[0]), []).append(i)

    for obs_key, rows in groups.items():
        if not obs_key:
            continue
        obs  = np.array(obs_key)
        rows = np.array(rows)
        S_oo = Sigma[np.ix_(obs, obs)]
        k    = len(obs)
        try:
            L = np.linalg.cholesky(S_oo)
        except np.linalg.LinAlgError:
            continue
        diff = X[np.ix_(rows, obs)] - mu[obs]
        V    = np.linalg.solve(L, diff.T)
        ll  += float(
            -0.5 * len(rows) * (k * np.log(2 * np.pi)
                                + 2 * np.sum(np.log(np.diag(L))))
            - 0.5 * np.sum(V ** 2)
        )
    return ll


# ---------------------------------------------------------------------------
# Conditional moments (expected-distance support for KNN / SVM)
# ---------------------------------------------------------------------------

class _ConditionalMoments:
    """
    Joint-Gaussian conditional moments for rows with missing entries.

    Fits N(mu, Sigma) by EM (available-case ML via _JointMVNFitter), then
    ``transform(X)`` returns (F, s):

        F : X with each missing entry replaced by its conditional mean
            given that row's observed entries;

        s : per-row summed conditional variance of the missing entries.

    For independent rows q ≠ t the expected squared Euclidean distance is
        E||x_q − x_t||² = ||F_q − F_t||² + s_q + s_t

    (Eirola et al., 2013).  Equivalently, row i embeds as (F_i, √s_i·e_i)
    with mutually orthogonal uncertainty axes, so expected distances are
    exact Euclidean distances in the augmented space; kernels built on
    them remain PSD.  No imputed dataset is ever created: the distance
    itself is the model expectation.
    """

    def __init__(self, max_iter=100, tol=1e-5, reg=1e-6):
        self.max_iter = max_iter
        self.tol      = tol
        self.reg      = reg

    def fit(self, X: np.ndarray) -> '_ConditionalMoments':
        """Fit the joint Gaussian, excluding columns that cannot inform it.

        A degenerate column gives the covariance a zero-variance direction,
        and conditioning against it is a division by rounding residue rather
        than the no-op it is in exact arithmetic. MissSupportRegressor drifted
        by 5.4e-04 between two row orders of the same data because of it, and
        by 2.2e-15 once the column was removed. The column is put back
        afterwards with its constant as the mean and zero variance, which is
        what it is, so callers still index by their own column numbers.
        """
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        self.degenerate_ = degenerate_feature_mask(X)
        keep = ~self.degenerate_

        fitter = _JointMVNFitter(max_iter=self.max_iter, tol=self.tol,
                                 reg=self.reg)
        if keep.all():
            fitter.fit(X)
            self.mu_    = fitter.mu_
            self.Sigma_ = fitter.Sigma_
            return self

        mu    = np.zeros(p)
        Sigma = np.zeros((p, p))
        for j in np.where(self.degenerate_)[0]:
            col = X[:, j]
            obs = col[~np.isnan(col)]
            mu[j] = float(obs[0]) if obs.size else 0.0
        if keep.any():
            fitter.fit(X[:, keep])
            idx = np.where(keep)[0]
            mu[idx] = fitter.mu_
            Sigma[np.ix_(idx, idx)] = fitter.Sigma_

        self.mu_    = mu
        self.Sigma_ = Sigma
        return self

    def transform(self, X: np.ndarray):
        """Return (F, s); pattern-grouped, one solve per unique pattern."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        F = X.copy()
        s = np.zeros(n)

        mu, Sigma = self.mu_, self.Sigma_

        groups = defaultdict(list)
        nan_mask = np.isnan(X)
        for i in range(n):
            groups[tuple(np.where(nan_mask[i])[0])].append(i)

        for mis_key, idxs in groups.items():
            if not mis_key:
                continue
            mis = np.array(mis_key)
            obs = np.setdiff1d(np.arange(p), mis)
            rows = np.array(idxs)

            # Conditioning on a degenerate column adds nothing and makes the
            # observed block singular. A degenerate column that is *missing*
            # needs no special case: its row and column of Sigma are exactly
            # zero, so the formulas below already return its constant as the
            # conditional mean and zero as the conditional variance.
            deg = getattr(self, 'degenerate_', None)
            if deg is not None and len(deg) == p and deg.any():
                obs = obs[~deg[obs]]

            if len(obs) == 0:
                F[np.ix_(rows, mis)] = mu[mis]
                s[rows] = float(np.trace(Sigma))
                continue

            S_oo = Sigma[np.ix_(obs, obs)]
            S_mo = Sigma[np.ix_(mis, obs)]
            S_mm = Sigma[np.ix_(mis, mis)]
            try:
                K = np.linalg.solve(S_oo, S_mo.T).T          # (|mis|, |obs|)
            except np.linalg.LinAlgError:
                K = S_mo @ np.linalg.pinv(S_oo)
            var_c = np.maximum(np.diag(S_mm - K @ S_mo.T), 0.0)

            X_obs = X[np.ix_(rows, obs)]
            F[np.ix_(rows, mis)] = mu[mis] + (X_obs - mu[obs]) @ K.T
            s[rows] = float(var_c.sum())

        return F, s


# ---------------------------------------------------------------------------
# MissImputer
# ---------------------------------------------------------------------------

class MissImputer:
    """
    Multiple imputation by draws from the joint MVN conditional distribution.

    For each missing entry X_{ij}, draws a value from:
        X_{ij} | X_{i,obs} ~ N(mu_cond, Sigma_cond)

    where mu_cond and Sigma_cond come from the partitioned-normal formula
    applied to the FIML-estimated joint MVN of X.

    Parameters
    ----------
    m : int, default 20
        Number of imputed datasets to generate.
    include_y : bool, default False
        If True, y is treated as an additional variable in the joint MVN
        (augmenting X).  Imputed datasets then include the imputed y column
        as the last column.
    max_iter : int, default 200
        Maximum EM iterations for the joint MVN fit.
    tol : float, default 1e-6
        Convergence tolerance for the EM algorithm.
    reg : float, default 1e-6
        Diagonal regularisation added to Sigma for numerical stability.
    posterior : bool, default False
        If True, draw MVN parameters (mu, Sigma) from a non-parametric
        bootstrap before each imputation draw, propagating parameter
        uncertainty.  This gives "proper" multiple imputation in Rubin's
        sense but increases computation by m-fold.
    random_state : int or None, default None

    Attributes
    ----------
    mu_     : ndarray, shape (p,) or (p+1,) if include_y
    Sigma_  : ndarray, shape (p, p) or (p+1, p+1) if include_y
    n_iter_ : int  -- EM iterations used
    """

    def __init__(
        self,
        m: int                   = 20,
        include_y: bool          = False,
        max_iter: int            = 200,
        tol: float               = 1e-6,
        reg: float               = 1e-6,
        posterior: bool          = False,
        random_state             = None,
    ):
        self.m            = m
        self.include_y    = include_y
        self.max_iter     = max_iter
        self.tol          = tol
        self.reg          = reg
        self.posterior    = posterior
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MissImputer':
        """
        Estimate the joint MVN distribution of X (and optionally y).

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        y : ndarray of shape (n,), optional; used only if include_y=True
        """
        X = np.asarray(X, dtype=float)
        if self.include_y:
            if y is None:
                raise ValueError("MissImputer: include_y=True requires y to be passed.")
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
            Z     = np.hstack([X, y_arr])
        else:
            Z = X

        self._fitter = _JointMVNFitter(
            max_iter=self.max_iter, tol=self.tol, reg=self.reg
        )
        self._fitter.fit(Z)
        self.mu_     = self._fitter.mu_
        self.Sigma_  = self._fitter.Sigma_
        self.n_iter_ = self._fitter.n_iter_
        self._Z_train = Z        # kept for posterior bootstrap
        return self

    # ------------------------------------------------------------------
    # transform: draw m imputed datasets
    # ------------------------------------------------------------------

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Draw m imputed datasets.

        Rows are grouped by missingness pattern before drawing so that the
        conditional covariance Sigma_c (which depends only on the pattern,
        not on the observed values) is computed once per unique pattern rather
        than once per row.  This gives a 20 to 40% speedup when n is large and
        the number of distinct patterns is small relative to n.

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        y : ndarray of shape (n,), required if include_y=True

        Returns
        -------
        datasets : list of m ndarrays, each of shape (n, p) [or (n, p+1) if include_y]
            Each dataset has all NaN replaced by draws from the conditional MVN.
            Observed values are unchanged.
        """
        # Every other estimator in the library reports an unfitted call
        # through check_is_fitted. This module did not, so the first thing
        # a caller saw was AttributeError on mu_, an internal name, raised
        # from inside a helper rather than at the door.
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if self.include_y:
            if y is None:
                raise ValueError("MissImputer: include_y=True requires y.")
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
            Z     = np.hstack([X, y_arr])
        else:
            Z = X.copy()

        from collections import defaultdict

        rng      = np.random.default_rng(self.random_state)
        datasets = []

        # Precompute missingness patterns once (Z is constant across draws)
        nan_mask_base  = np.isnan(Z)
        pattern_groups = defaultdict(list)
        for i in range(Z.shape[0]):
            key = (tuple(np.where(~nan_mask_base[i])[0]),
                   tuple(np.where( nan_mask_base[i])[0]))
            pattern_groups[key].append(i)

        for imp in range(self.m):
            # --- optionally re-sample MVN parameters from bootstrap ---
            if self.posterior:
                n_tr = self._Z_train.shape[0]
                boot_idx = rng.integers(0, n_tr, size=n_tr)
                boot_fitter = _JointMVNFitter(
                    max_iter=self.max_iter, tol=self.tol, reg=self.reg
                )
                boot_fitter.fit(self._Z_train[boot_idx])
                mu_use    = boot_fitter.mu_
                Sigma_use = boot_fitter.Sigma_
            else:
                mu_use    = self.mu_
                Sigma_use = self.Sigma_

            Z_imp = Z.copy()

            for (obs_t, mis_t), row_idxs in pattern_groups.items():
                obs_idx = np.array(obs_t)
                mis_idx = np.array(mis_t)
                if len(mis_idx) == 0:
                    continue

                # Compute conditional params once per pattern (not per row)
                if len(obs_idx) == 0:
                    mu_c    = mu_use[mis_idx]
                    Sigma_c = Sigma_use[np.ix_(mis_idx, mis_idx)]
                else:
                    # batch: mu_c varies per row, Sigma_c is constant for the pattern
                    Sigma_oo = Sigma_use[np.ix_(obs_idx, obs_idx)]
                    Sigma_mo = Sigma_use[np.ix_(mis_idx, obs_idx)]
                    Sigma_mm = Sigma_use[np.ix_(mis_idx, mis_idx)]
                    K        = np.linalg.solve(Sigma_oo, Sigma_mo.T).T
                    Sigma_c  = Sigma_mm - K @ Sigma_mo.T
                    Sigma_c  = 0.5 * (Sigma_c + Sigma_c.T)

                Sigma_c = Sigma_c + self.reg * np.eye(len(mis_idx))

                # Draw once per row but Sigma_c is shared -- use batch draw
                n_grp = len(row_idxs)
                if len(obs_idx) == 0:
                    draws = rng.multivariate_normal(mu_c, Sigma_c, size=n_grp)
                else:
                    # mu_c is row-specific: mu_mis + (x_obs - mu_obs) @ K.T
                    x_obs_batch = Z[np.ix_(row_idxs, obs_idx)]
                    mu_c_batch  = mu_use[mis_idx] + (x_obs_batch - mu_use[obs_idx]) @ K.T
                    # Draw each row separately (mu differs per row, Sigma shared)
                    draws = np.vstack([
                        rng.multivariate_normal(mu_c_batch[j], Sigma_c)
                        for j in range(n_grp)
                    ])

                for j, i in enumerate(row_idxs):
                    Z_imp[i, mis_idx] = draws[j]

            datasets.append(Z_imp)

        return datasets

    # ------------------------------------------------------------------
    # transform_mean: conditional mean imputation (deterministic)
    # ------------------------------------------------------------------

    def transform_mean(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return a single dataset where each missing value is replaced by its
        conditional mean given the observed values in the same row.

        This is equivalent to a single deterministic imputation (m=1, no noise).
        Appropriate for exploring the data or as a baseline; NOT recommended as
        the only imputation for inference (underestimates variance).

        Returns
        -------
        X_imputed : ndarray of shape (n, p) [or (n, p+1) if include_y]
        """
        # Every other estimator in the library reports an unfitted call
        # through check_is_fitted. This module did not, so the first thing
        # a caller saw was AttributeError on mu_, an internal name, raised
        # from inside a helper rather than at the door.
        check_is_fitted(self)
        from collections import defaultdict

        X = np.asarray(X, dtype=float)
        if self.include_y:
            if y is None:
                raise ValueError("MissImputer: include_y=True requires y.")
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
            Z_out = np.hstack([X, y_arr])
        else:
            Z_out = X.copy()   # single copy: read observed, write missing in-place

        # Precompute missingness patterns once
        nan_mask_base  = np.isnan(Z_out)
        pattern_groups = defaultdict(list)
        for i in range(Z_out.shape[0]):
            key = (tuple(np.where(~nan_mask_base[i])[0]),
                   tuple(np.where( nan_mask_base[i])[0]))
            pattern_groups[key].append(i)

        for (obs_t, mis_t), row_idxs in pattern_groups.items():
            obs_idx = np.array(obs_t)
            mis_idx = np.array(mis_t)
            if len(mis_idx) == 0:
                continue

            if len(obs_idx) == 0:
                # Fully missing: fill with marginal mean (same for all rows in group)
                for i in row_idxs:
                    Z_out[i, mis_idx] = self.mu_[mis_idx]
            else:
                # Compute Sigma_c components once per pattern
                Sigma_oo = self.Sigma_[np.ix_(obs_idx, obs_idx)]
                Sigma_mo = self.Sigma_[np.ix_(mis_idx, obs_idx)]
                K        = np.linalg.solve(Sigma_oo, Sigma_mo.T).T
                # mu_c is row-specific: mu_mis + (x_obs - mu_obs) @ K.T
                x_obs_batch = Z_out[np.ix_(row_idxs, obs_idx)]
                mu_c_batch  = self.mu_[mis_idx] + (x_obs_batch - self.mu_[obs_idx]) @ K.T
                for j, i in enumerate(row_idxs):
                    Z_out[i, mis_idx] = mu_c_batch[j]

        return Z_out

    # ------------------------------------------------------------------
    # Rubin's rules: combine m estimates
    # ------------------------------------------------------------------

    @staticmethod
    def combine(
        estimates: np.ndarray,
        variances: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Combine m parameter estimates using Rubin's (1987) rules.

        Parameters
        ----------
        estimates : ndarray of shape (m,) or (m, k)
            Parameter estimate from each imputed dataset.
        variances : ndarray of shape (m,) or (m, k), optional
            Within-imputation variance (e.g. squared standard errors) for
            each estimate.  If None, only the point estimate and
            between-imputation variance are returned (no total variance or df).

        Returns
        -------
        dict with keys:
            'estimate'    : pooled estimate (shape () or (k,))
            'within_var'  : W -- mean within-imputation variance (if variances given)
            'between_var' : B -- between-imputation variance
            'total_var'   : T -- total variance W + (1+1/m)*B (if variances given)
            'se'          : sqrt(T) (if variances given)
            'df'          : Rubin degrees of freedom (if variances given and m > 1)
            'p_value'     : two-sided p-value vs 0 (scalar estimate only, if variances given)
        """
        estimates = np.asarray(estimates, dtype=float)
        m         = estimates.shape[0]

        theta_bar = estimates.mean(axis=0)
        B         = np.var(estimates, axis=0, ddof=1) if m > 1 else np.zeros_like(theta_bar)

        result = {'estimate': theta_bar, 'between_var': B, 'm': m}

        if variances is not None:
            variances = np.asarray(variances, dtype=float)
            # W and B must come from the same m. Averaging a shorter list of
            # variances against a longer list of estimates went unnoticed and
            # produced a pooled standard error 11.5% too small on three
            # estimates against two variances, with m still reported as
            # three. A pooled SE is the number that gets published, so a
            # quiet 11% is worse than a loud refusal.
            if variances.shape[0] != m:
                raise ValueError(
                    "estimates and variances must cover the same imputed "
                    "datasets: got %d estimate(s) and %d variance(s). "
                    "Rubin's rules combine the within-imputation variance of "
                    "each estimate with the variance between them, so a "
                    "mismatch has no defined answer."
                    % (m, variances.shape[0]))
            W = variances.mean(axis=0)
            T = W + (1.0 + 1.0 / m) * B
            result['within_var'] = W
            result['total_var']  = T
            result['se']         = np.sqrt(np.maximum(T, 0.0))

            if m > 1:
                # Rubin (1987) degrees of freedom. Not Barnard-Rubin, which is
                # what this was labelled: that is the 1999 small-sample
                # correction built from the complete-data degrees of freedom,
                # and it is not what is computed here.
                #
                # r is the relative increase in variance due to missingness,
                # and it is exactly zero when every imputation returns the
                # same estimate. The degrees of freedom are then infinite,
                # which is right, since with no between-imputation variance
                # the reference distribution is the normal; but reaching it
                # through 1/0 announced itself with a RuntimeWarning.
                r = np.atleast_1d(np.asarray(
                    (1.0 + 1.0 / m) * B / np.maximum(W, 1e-300), dtype=float))
                # The infinite case is filled in, not computed. np.where would
                # evaluate both arms, and the arm for r = 0 overflows on the
                # way to the answer it is not going to use: 1/1e-300 squared
                # is 1e600, which warns before it is discarded.
                df_arr = np.full(r.shape, np.inf)
                positive = r > 0.0
                if positive.any():
                    df_arr[positive] = (m - 1) * (1.0 + 1.0 / r[positive]) ** 2
                df = df_arr if np.ndim(B) else float(df_arr[0])
                result['df'] = df

                # p-value only for scalar estimate
                if theta_bar.ndim == 0:
                    t_stat = float(theta_bar) / float(result['se'])
                    pv     = 2.0 * sp_stats.t.sf(abs(t_stat), df=float(df))
                    result['p_value'] = pv

        return result

    # ------------------------------------------------------------------
    # Convenience: fit m models and combine
    # ------------------------------------------------------------------

    def fit_transform_combine(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator,
        param: str,
        param_var: Optional[str] = None,
        **fit_kwargs,
    ) -> dict:
        """
        Fit *estimator* on each of the m imputed datasets and combine
        parameter estimates using Rubin's rules.

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        y : ndarray of shape (n,), may contain NaN
        estimator : any sklearn-compatible estimator with fit()
        param : str
            Attribute name of the parameter to pool (e.g. '``coef_``').
        param_var : str or None
            Attribute holding the **standard error** of *param*, not its
            variance: whatever is found there is squared before entering
            Rubin's rules. Use '``se_``' for MissLearn models. An attribute
            that already holds a variance yields a variance squared, and a
            pooled standard error that nothing in the output marks as wrong,
            so the name of this argument is the one thing about it not to
            trust. If None, Rubin's pooled standard error is not computed.
        **fit_kwargs : forwarded to estimator.fit()

        Returns
        -------
        dict from MissImputer.combine()
            Plus 'fitted_estimators': list of m fitted estimators.
        """
        # Every other estimator in the library reports an unfitted call
        # through check_is_fitted. This module did not, so the first thing
        # a caller saw was AttributeError on mu_, an internal name, raised
        # from inside a helper rather than at the door.
        check_is_fitted(self)
        X   = np.asarray(X, dtype=float)
        y   = np.asarray(y, dtype=float)

        if self.include_y:
            # Datasets are (n, p+1) with the jointly imputed y last; use it
            # so rows with missing y contribute rather than being dropped.
            datasets = self.transform(X, y)
        else:
            datasets = self.transform(X)

        estimates_list = []
        variances_list = []
        fitted_ests    = []

        for k, data in enumerate(datasets):
            if self.include_y:
                X_imp = data[:, :-1]
                y_imp = data[:, -1]
            else:
                X_imp = data
                y_imp = y.copy()
            est = copy.deepcopy(estimator)
            # Strip NaN-y rows for estimators that can't handle them
            obs = ~np.isnan(y_imp)
            try:
                est.fit(X_imp[obs], y_imp[obs], **fit_kwargs)
            except TypeError:
                # estimator doesn't accept fit_kwargs
                est.fit(X_imp[obs], y_imp[obs])

            val = getattr(est, param, None)
            if val is None:
                warnings.warn(
                    f"MissImputer.fit_transform_combine: estimator has no "
                    f"attribute '{param}' after fitting imputed dataset {k}.",
                    UserWarning, stacklevel=2,
                )
                continue
            val_arr = np.atleast_1d(np.asarray(val, dtype=float))

            if param_var is not None:
                # Rubin's rules require estimates and variances from the SAME
                # set of imputations; drop the whole imputation if its
                # variance cannot be extracted, rather than silently pooling
                # mismatched sets.
                v = getattr(est, param_var, None)
                v_arr = None
                if v is not None:
                    v_arr = np.atleast_1d(np.asarray(v, dtype=float))
                    # MissLearn se_ includes intercept at index 0;
                    # strip it if the length is one more than the estimate.
                    if len(v_arr) == len(val_arr) + 1:
                        v_arr = v_arr[1:]
                    if len(v_arr) != len(val_arr) or not np.all(np.isfinite(v_arr)):
                        v_arr = None
                if v_arr is None:
                    warnings.warn(
                        f"MissImputer.fit_transform_combine: could not extract "
                        f"'{param_var}' for imputed dataset {k}; dropping this "
                        f"imputation from pooling.",
                        UserWarning, stacklevel=2,
                    )
                    continue
                variances_list.append(v_arr ** 2)  # se -> variance

            estimates_list.append(val_arr)
            fitted_ests.append(est)

        if not estimates_list:
            raise RuntimeError(
                "MissImputer.fit_transform_combine: no successful fits."
            )

        combined = self.combine(
            np.array(estimates_list),
            np.array(variances_list) if variances_list else None,
        )
        combined['fitted_estimators'] = fitted_ests
        return combined

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        # Every other estimator in the library reports an unfitted call
        # through check_is_fitted. This module did not, so the first thing
        # a caller saw was AttributeError on mu_, an internal name, raised
        # from inside a helper rather than at the door.
        check_is_fitted(self)
        W = 66
        p = len(self.mu_)
        print()
        print('=' * W)
        print(f"{'MissImputer  --  Multiple Imputation via Joint MVN':^{W}}")
        print('=' * W)
        print(f"  Imputations (m) : {self.m}")
        print(f"  Variables (p)   : {p}{' (includes y)' if self.include_y else ''}")
        print(f"  EM iterations   : {self.n_iter_}")
        print(f"  Posterior draws : {self.posterior}")
        print()
        print("  Fitted marginal means:")
        for j, mj in enumerate(self.mu_):
            label = f"y" if (self.include_y and j == p - 1) else f"X{j}"
            print(f"    {label:<6}  mu={mj:8.4f}  sd={np.sqrt(self.Sigma_[j,j]):8.4f}")
        print()
        print("  Correlation matrix (fitted joint MVN):")
        sd   = np.sqrt(np.diag(self.Sigma_))
        corr = self.Sigma_ / np.outer(sd, sd)
        np.set_printoptions(precision=3, suppress=True, linewidth=W - 4)
        for row in corr:
            print("   ", row)
        np.set_printoptions()
        print('=' * W)
        print()

    def __repr__(self) -> str:
        return (f"MissImputer(m={self.m}, include_y={self.include_y}, "
                f"posterior={self.posterior}, random_state={self.random_state})")
