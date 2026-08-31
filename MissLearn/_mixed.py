"""
MissLearn._mixed
----------------
Full Information Maximum Likelihood (FIML) linear mixed-effects model (LME)
and generalized linear mixed model (GLMM) with native missing data support.

No imputation. No listwise deletion. No fake data.
Each observation contributes exactly the information it contains.

MissMixedRegressor
    FIML random-intercept LME for continuous outcomes.

    Model:
        y_ij = beta_0 + x_ij^T beta + b_i + eps_ij
        b_i  ~ N(0, tau^2)          (subject-level random intercept)
        eps_ij ~ N(0, sigma^2)      (within-subject residual)

    Missing outcomes: handled by FIML -- the observed subvector of y for
    each subject is used; unobserved rows contribute zero to the likelihood.

    Missing covariates: marginalized analytically using the conditional-
    normal approach (same as all other MissLearn models): for each row with
    partial X, the missing-covariate contribution to the linear predictor is
    replaced by its conditional expectation given X_obs, and the additional
    variance enters V_i on the diagonal via the Woodbury formula.

MissMixedClassifier
    FIML random-intercept GLMM for binary outcomes (logistic link).

    Model:
        y_ij | b_i ~ Bernoulli(sigma(beta_0 + x_ij^T beta + b_i))
        b_i ~ N(0, tau^2)

    The random intercept is integrated out via Gauss-Hermite quadrature
    (20 nodes by default). Missing covariates are handled using the probit
    approximation: P(y=1 | x_obs, b) ~ sigma((eta_base + b) / sqrt(1 + pi/8 * v)),
    where v = Var[x_mis^T beta_mis | x_obs].

MissMixed
    Auto-selecting wrapper: detects regression vs. classification from y.
"""

import warnings

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.special import expit as _expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._conformance import check_common_parameters
from ._base import MissBase, MissTags, only_for
from ._copula import RankNormalTransformer, needs_copula
from ._utils import (
    pack_cholesky, unpack_cholesky, mvn_logpdf, mvn_logpdf_batch,
    numerical_hessian, feature_scale, standard_errors_from_variance,
    integrate_logistic_normal
)

# tau_sq reaches the optimiser as exp(2*log_tau) and underflows to exactly
# 0.0 once log_tau drops below about -354, at which point the Python float
# division 1.0/tau_sq raises ZeroDivisionError and the fit dies. SLSQP walks
# there on ordinary data. The limit itself is not a problem: tau_sq -> 0 is
# the no-random-effect case, where the shrinkage term vanishes and the
# likelihood tends to the ordinary regression one, so the region is floored
# rather than refused.
_TAU_SQ_FLOOR = np.finfo(float).tiny


def _resolve_groups(groups, n):
    """Fill in ``groups`` when the caller did not supply any.

    ``groups`` used to be a required positional argument, which made
    ``fit(X, y)`` impossible and left these three estimators unusable with
    every scikit-learn utility that calls it that way: cross-validation,
    pipelines, and ``check_estimator``, which failed twenty-five checks on
    MissMixedRegressor alone for this one reason.

    Absent groups means one group, so the model reduces to its fixed-effects
    counterpart. A single random intercept shared by every row is absorbed
    into the fixed intercept, tau goes to zero, and what remains is ordinary
    FIML regression. Measured on data with no group structure: r-squared
    0.9796 against MissLinear's 0.9796, with tau_sq_ at 1.9e-07.

    The tempting alternative, giving every row its own group, is wrong and
    quietly so. It hands each observation a free intercept, which is
    confounded with the residual and inflated the same fit to 0.9860. A
    default that makes a model look better than it is has no business being
    the one you get for saying nothing.
    """
    if groups is not None:
        return groups
    return np.zeros(int(n), dtype=np.int64)

# ======================================================================
# Module-level constants
# ======================================================================

_SQRT2 = np.sqrt(2.0)
_PI_OVER_8 = np.pi / 8.0


# ======================================================================
# Shared helper functions
# ======================================================================

def _group_by_subject(groups):
    """
    Build an ordered mapping from subject label to row indices.

    Returns
    -------
    dict {label: ndarray of int}
    """
    groups = np.asarray(groups)
    mapping = defaultdict(list)
    for i, g in enumerate(groups):
        mapping[g].append(i)
    return {g: np.array(idxs) for g, idxs in mapping.items()}


def _logpdf_x_row(x_row, mu_X, Sigma_X):
    """
    log P(x_obs | mu_X, Sigma_X) for one row with possible missing values.

    Only observed dimensions contribute; the marginal MVN over those
    dimensions is evaluated via Cholesky.

    Returns 0.0 if no values are observed.
    """
    obs_idx = np.where(~np.isnan(x_row))[0]
    if len(obs_idx) == 0:
        return 0.0
    return mvn_logpdf(
        x_row[obs_idx],
        mu_X[obs_idx],
        Sigma_X[np.ix_(obs_idx, obs_idx)],
    )


def _subject_log_integrand(b, eta_scaled, sign, inv_scale, tau_sq):
    """log of a subject's integrand at each value of the random effect.

        g(b) = sum_j log sigma( sign_j * (eta_j + b / scale_j) )
               + log N(b; 0, tau^2)

    ``b`` may be an array, in which case the return has its shape.
    """
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    u = sign[None, :] * (eta_scaled[None, :] + b[:, None] * inv_scale[None, :])
    log_prior = (-0.5 * b * b / tau_sq
                 - 0.5 * np.log(2.0 * np.pi * tau_sq))
    return _log_sigmoid(u).sum(axis=1) + log_prior


def _subject_mode(eta_scaled, sign, inv_scale, tau_sq, max_iter=60, tol=1e-11):
    """Mode of a subject's log-integrand, and the curvature there.

    g is strictly concave: its second derivative is

        g''(b) = -sum_j (1/scale_j)^2 sigma(u_j) sigma(-u_j) - 1 / tau^2

    every term of which is negative. So the mode is unique and Newton from
    zero reaches it in a handful of steps without safeguarding.

    Returns
    -------
    b_hat : float
    sd_hat : float
        ``1 / sqrt(-g''(b_hat))``, the width of the Gaussian that matches the
        integrand's curvature at its peak.
    """
    inv_tau_sq = 1.0 / tau_sq
    b = 0.0
    curv = inv_tau_sq
    for _ in range(max_iter):
        u = sign * (eta_scaled + b * inv_scale)
        s = _expit(-u)                       # sigma(-u)
        g1 = float(np.sum(sign * inv_scale * s) - b * inv_tau_sq)
        curv = float(np.sum(inv_scale * inv_scale * s * (1.0 - s))
                     + inv_tau_sq)
        step = g1 / curv                     # curv = -g''
        b_new = b + step
        if not np.isfinite(b_new):
            break
        converged = abs(b_new - b) <= tol * (1.0 + abs(b))
        b = b_new
        if converged:
            break
    u = sign * (eta_scaled + b * inv_scale)
    s = _expit(-u)
    curv = float(np.sum(inv_scale * inv_scale * s * (1.0 - s)) + inv_tau_sq)
    if not (np.isfinite(curv) and curv > 0.0):
        curv = inv_tau_sq
    return b, 1.0 / np.sqrt(curv)


def _adaptive_nodes(eta_scaled, sign, inv_scale, tau_sq, gh_t, gh_log_w):
    """Adaptive Gauss-Hermite nodes and log weights for one subject.

    Plain Gauss-Hermite spreads its nodes over the *prior* for the random
    effect, at width tau. Once a subject carries several observations the
    likelihood pins the effect into a much narrower interval, which may sit far
    from zero, and most of the nodes then land where the integrand is
    negligible. The error grows with both tau and the number of observations
    per subject: 4e-10 at tau = 1, but 1.7e-3 at tau = 2 and, with twenty
    observations, 0.62 at tau = 3, where even 320 nodes leave 6e-4.

    Recentring on the mode and rescaling to the curvature there (Liu and
    Pierce 1994) puts the nodes where the integrand actually lives. Writing
    b = b_hat + sqrt(2) * sd_hat * z,

        INT exp(g(b)) db = sqrt(2) sd_hat INT exp(g(...)) dz
                         = sqrt(2) sd_hat INT [exp(g(...)) exp(z^2)] exp(-z^2) dz

    so the Gauss-Hermite weights apply to the bracketed function and each
    weight picks up exp(t_k^2). With one node this is the Laplace
    approximation; the unadapted rule is the special case b_hat = 0,
    sd_hat = tau.

    Returns
    -------
    b_nodes : ndarray, shape (Q,)
    log_w : ndarray, shape (Q,)
        Log weights with the exp(t_k^2) factor and the sqrt(2) * sd_hat
        Jacobian already folded in, so that ``logsumexp(log_w + g(b_nodes))``
        is the log of the integral.
    """
    b_hat, sd_hat = _subject_mode(eta_scaled, sign, inv_scale, tau_sq)
    b_nodes = b_hat + _SQRT2 * sd_hat * gh_t
    log_w = gh_log_w + gh_t * gh_t + np.log(_SQRT2 * sd_hat)
    return b_nodes, log_w


def _log_sigmoid(x):
    """
    Numerically stable log-sigmoid: log sigma(x) = -log(1 + exp(-x)).

    Works element-wise on arrays. Clips x to [-500, 500] to avoid overflow.
    """
    return -np.log1p(np.exp(-np.clip(x, -500.0, 500.0)))


def _build_x_patterns(X):
    """
    Group row indices by X-missingness pattern.

    Parameters
    ----------
    X : ndarray, shape (n, p)

    Returns
    -------
    patterns : list of (obs_idx, mis_idx, row_idxs)
        obs_idx  : ndarray of observed feature indices (sorted)
        mis_idx  : ndarray of missing feature indices
        row_idxs : ndarray of row indices in X with this pattern
    """
    p = X.shape[1]
    all_idx = np.arange(p)
    pattern_map = defaultdict(list)
    for i, x_row in enumerate(X):
        key = tuple(np.where(~np.isnan(x_row))[0])
        pattern_map[key].append(i)
    return [
        (np.array(key, dtype=np.intp),
         np.setdiff1d(all_idx, key),
         np.array(idxs, dtype=np.intp))
        for key, idxs in pattern_map.items()
    ]


def _row_contribs_batched(X, x_patterns, beta, mu_X, Sigma_X):
    """
    Compute the per-row (mu_adj, delta) arrays efficiently.

    For each unique missingness pattern, the conditional-normal quantities
    (K, Sigma_cond) are computed once and applied to all rows in the group.

    Parameters
    ----------
    X         : ndarray, shape (n, p)
    x_patterns: output of _build_x_patterns(X)
    beta      : ndarray, shape (p,)   -- fixed-effect slopes (no intercept)
    mu_X      : ndarray, shape (p,)
    Sigma_X   : ndarray, shape (p, p)

    Returns
    -------
    mu_adj_all : ndarray, shape (n,)   E[x^T beta | x_obs] per row
    delta_all  : ndarray, shape (n,)   Var[x_mis^T beta_mis | x_obs] per row
    """
    n          = X.shape[0]
    mu_adj_all = np.empty(n)
    delta_all  = np.zeros(n)

    for obs_idx, mis_idx, row_idxs in x_patterns:
        if len(obs_idx) == 0:
            # All-missing rows: unconditional linear predictor mean
            mu_adj_all[row_idxs] = float(mu_X @ beta)
            delta_all[row_idxs]  = float(beta @ Sigma_X @ beta)

        elif len(mis_idx) == 0:
            # Complete rows: direct linear predictor
            x_batch              = X[np.ix_(row_idxs, obs_idx)]
            mu_adj_all[row_idxs] = x_batch @ beta[obs_idx]
            # delta stays 0

        else:
            # Partial rows: conditional-normal adjustment
            x_batch    = X[np.ix_(row_idxs, obs_idx)]
            beta_mis   = beta[mis_idx]
            Sigma_oo   = Sigma_X[np.ix_(obs_idx, obs_idx)]
            Sigma_mo   = Sigma_X[np.ix_(mis_idx, obs_idx)]
            Sigma_mm   = Sigma_X[np.ix_(mis_idx, mis_idx)]
            K          = np.linalg.solve(Sigma_oo, Sigma_mo.T).T  # (|mis|, |obs|)
            Sigma_cond = Sigma_mm - K @ Sigma_mo.T
            Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
            delta      = float(beta_mis @ Sigma_cond @ beta_mis)

            # mu_mis_cond for all rows in this group: (n_pat, |mis|)
            mu_mis_cond          = mu_X[mis_idx] + (x_batch - mu_X[obs_idx]) @ K.T
            mu_adj_all[row_idxs] = (
                x_batch @ beta[obs_idx] + mu_mis_cond @ beta_mis
            )
            delta_all[row_idxs]  = max(delta, 0.0)

    return mu_adj_all, delta_all


# ======================================================================
# MissMixedRegressor
# ======================================================================

class MissMixedRegressor(RegressorMixin, MissBase):
    """
    FIML random-intercept linear mixed-effects model with native missing
    data support.

    Fits the model:
        y_ij = beta_0 + x_ij^T beta + b_i + eps_ij,
        b_i  ~ N(0, tau^2),    eps_ij ~ N(0, sigma^2).

    For each subject i, the marginal distribution of the *observed* outcome
    subvector is::

        y_i_obs ~ N(mu_i_obs, V_i_obs),

    where V_i_obs = tau^2 * J J^T + diag(sigma^2 + delta_j).

    J is the ones vector for the observed rows, and delta_j = beta_mis^T
    Sigma_{mis|obs} beta_mis absorbs the uncertainty from missing covariates
    in row j.  The matrix inversion and determinant are computed in O(n_i)
    via the Woodbury / matrix-determinant lemma (compound symmetry + diagonal).

    Parameters
    ----------
    max_iter : int
        Lower bound on the L-BFGS-B iteration budget (default 2000). The
        budget actually used is ``max(max_iter, 300 * (p + 3))``, because the
        reduced objective is optimised with a numerical gradient costing
        ``2(p + 3)`` evaluations per step and a fixed budget is exhausted on
        wide problems well before convergence. From ``p = 4`` upward the
        scaled term already exceeds the default, so raising `max_iter` has an
        effect only above that floor and lowering it has none.
    tol : float
        Convergence tolerance on function value and gradient (default 1e-7).
        This is the operative stopping rule: `ftol` on the function value.
        The projected-gradient test is left loose, because for an
        unnormalised large-sample likelihood the gradient norm at the optimum
        is naturally O(n).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    compute_se : bool
        If True (default), compute standard errors from the Hessian diagonal.
    copula : bool or 'auto', default 'auto'
        If True, apply a marginal Gaussian copula transform to X and y before
        fitting.  If 'auto', apply when data appears non-normal (``|skewness|`` > 1
        or ``|excess kurtosis|`` > 2 on any column).
    fe_ridge : float, default 1e-2
        Vanishing ridge on the internally standardised fixed effects.  It is
        negligible for a well-conditioned design (the standardised
        coefficients are O(1)) but regularises the unidentified directions
        under strongly collinear features, keeping prediction for new groups
        stable instead of letting the coefficients drift in the collinear
        null space.  Set to 0.0 to recover the unpenalised FIML estimator.
        Excluded from the reported log-likelihood, AIC and BIC.

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
        Fixed-effect slope estimates.
    intercept_ : float
        Fixed-effect intercept.
    tau_sq_ : float
        Estimated random-intercept variance.
    sigma_sq_ : float
        Estimated within-subject residual variance.
    icc_ : float
        Intraclass correlation: tau^2 / (tau^2 + sigma^2).
    blup_ : dict {group_label: float}
        Best Linear Unbiased Predictors of the random intercept for each
        training subject.  New subjects receive BLUP = 0.
    n_groups_ : int
        Number of distinct subjects / clusters in the training set.
    se_ : ndarray, shape (p+1,)
        Standard errors: [se_intercept, se_coef_0, ..., se_coef_{p-1}].
    z_stats_ : ndarray, shape (p+1,)
    pvalues_ : ndarray, shape (p+1,)
    coef_std_ : ndarray, shape (p,)
        Standardized fixed-effect coefficients.
    mu_X_ : ndarray, shape (p,)
    Sigma_X_ : ndarray, shape (p, p)
    loglik_ : float
    aic_, bic_ : float
    converged_ : bool
    copula_used_ : bool
    """

    def __init__(self, max_iter=2000, tol=1e-7, method='L-BFGS-B',
                 compute_se=True, copula='auto', fe_ridge=1e-2):
        self.max_iter   = max_iter
        self.tol        = tol
        self.method     = method
        self.compute_se = compute_se
        self.copula     = copula
        self.fe_ridge   = fe_ridge

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    def _pack_params(self, intercept, beta, tau, sigma, mu_X, Sigma_X):
        L_vec, _ = pack_cholesky(Sigma_X)
        return np.concatenate([
            [intercept], beta, [np.log(tau), np.log(sigma)], mu_X, L_vec
        ])

    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Best Linear Unbiased Predictors
    # ------------------------------------------------------------------ #

    def _compute_blups(self, X, y, subject_map, mu_adj_all, delta_all):
        """
        Compute the BLUP of b_i for every training subject.

        b_hat_i = tau^2 * J^T V_i^{-1} (y_i_obs - mu_i_obs)

        Uses pre-batched (mu_adj_all, delta_all) for efficiency.
        """
        intercept = self.intercept_
        tau_sq    = self.tau_sq_
        sigma_sq  = self.sigma_sq_

        blup = {}
        for subj, row_idxs in subject_map.items():
            y_i        = y[row_idxs]
            obs_y_mask = ~np.isnan(y_i)

            if not obs_y_mask.any():
                blup[subj] = 0.0
                continue

            obs_global = row_idxs[obs_y_mask]
            mu_vec     = intercept + mu_adj_all[obs_global]
            delta_vec  = delta_all[obs_global]

            y_obs   = y_i[obs_y_mask]
            r       = y_obs - mu_vec
            d_vec   = sigma_sq + delta_vec
            d_inv   = 1.0 / d_vec
            denom   = 1.0 / max(tau_sq, _TAU_SQ_FLOOR) + np.sum(d_inv)
            Dinv_r  = d_inv * r
            Vinv_r  = Dinv_r - d_inv * (np.sum(Dinv_r) / denom)

            blup[subj] = tau_sq * float(np.sum(Vinv_r))

        return blup

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y, groups=None):
        """
        Fit the FIML random-intercept LME model.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    NaN values treated as missing.
        groups : array-like, shape (n,)
            Subject / cluster identifier for each observation.
            All observations sharing the same label are treated as
            measurements from one subject and contribute a shared random
            intercept.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        groups = _resolve_groups(groups, len(y))
        # Checked before the reorder, which zips the three together and
        # otherwise fails inside numpy with a concatenation message that
        # names neither groups nor the lengths involved.
        if len(np.asarray(groups)) != len(y):
            raise ValueError(
                "%s: groups has %d entries but X and y have %d rows. Each "
                "row needs the label of the subject it belongs to."
                % (type(self).__name__, len(np.asarray(groups)), len(y)))
        # X, y and groups are reordered together: permuting two of the three
        # would attach observations to the wrong subject.
        X, y, groups = self._canonical_fit_order_with_groups(X, y, groups)
        self._store_fit_metadata(X, y)
        self.groups_fit_ = np.asarray(groups)
        n, p = X.shape

        # Resolve copula='auto'
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X, y)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X                = self._copula_X_.transform(X)
            y_obs_mask       = ~np.isnan(y)
            self._copula_y_  = RankNormalTransformer().fit(
                y[y_obs_mask].reshape(-1, 1)
            )
            y = y.copy()
            y[y_obs_mask] = self._copula_y_.transform(
                y[y_obs_mask].reshape(-1, 1)
            ).ravel()

        # Internal standardisation (the convention of the penalized models):
        # the fixed-effect optimiser in stage two is ill-conditioned when the
        # features differ widely in scale or are collinear, which on real
        # high-dimensional data drives the coefficients to divergence.  Fit on
        # standardised X and y, then convert every public parameter back to
        # the original scale (the random-intercept LME is affine-equivariant).
        _mx = np.nanmean(X, axis=0)
        _mx = np.where(np.isfinite(_mx), _mx, 0.0)
        _sx = feature_scale(X)
        _yobs = ~np.isnan(y)
        _my = float(np.mean(y[_yobs])) if _yobs.any() else 0.0
        _sy = float(np.std(y[_yobs], ddof=1)) if _yobs.sum() > 1 else 1.0
        if not (np.isfinite(_sy) and _sy >= 1e-8):
            _sy = 1.0
        _ny = int(_yobs.sum())
        X = (X - _mx) / _sx
        y = np.where(np.isnan(y), np.nan, (y - _my) / _sy)
        # Stored so prediction can run in the same standardised space: the
        # conditional solves are ill-conditioned in raw units when feature
        # scales span orders of magnitude, even though the affine conversion
        # of the parameters themselves is exact.
        self._x_mean_, self._x_scale_ = _mx, _sx
        self._y_mean_, self._y_scale_ = _my, _sy

        subject_map    = _group_by_subject(groups)
        self.n_groups_ = len(subject_map)

        # Pre-compute missingness patterns once (reused across all NLL calls)
        x_patterns = _build_x_patterns(X)

        # Seed the optimiser from complete cases; if there are too few (large
        # p or high missingness) fall back to a mean-imputed seed rather than
        # failing.  The FIML likelihood below still integrates over every
        # observed entry, so this only sets the starting point.
        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_cc          = X[complete_mask]
        y_cc          = y[complete_mask]

        mu_X0 = np.nanmean(X, axis=0)
        mu_X0 = np.where(np.isfinite(mu_X0), mu_X0, 0.0)

        if len(y_cc) >= p + 2:
            X_seed, y_seed = X_cc, y_cc
        else:
            obs_y  = ~np.isnan(y)
            X_imp  = np.where(np.isnan(X), mu_X0, X)
            X_seed, y_seed = X_imp[obs_y], y[obs_y]

        from sklearn.linear_model import LinearRegression
        if len(y_seed) >= 2:
            lr = LinearRegression()
            lr.fit(X_seed, y_seed)
            beta0      = lr.coef_
            intercept0 = float(lr.intercept_)
            resid      = y_seed - lr.predict(X_seed)
            resid_sd   = float(np.std(resid))
        else:
            # Degenerate seed: start flat and let the likelihood do the work.
            beta0      = np.zeros(p)
            intercept0 = float(np.nanmean(y)) if np.any(~np.isnan(y)) else 0.0
            resid_sd   = float(np.nanstd(y)) if np.any(~np.isnan(y)) else 1.0
        if not np.isfinite(resid_sd) or resid_sd <= 0:
            resid_sd = 1.0
        sigma0     = max(resid_sd * 0.8, 1e-4)
        tau0       = max(resid_sd * 0.4, 1e-4)

        # --- Stage 1: X-moments by EM (full-information) -----------------
        # Estimating the MVN nuisance parameters once removes p + p(p+1)/2
        # dimensions from the optimizer below (and from the SE Hessian),
        # whose finite-difference evaluations otherwise dominate runtime.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        from ._utils import prep_conditional_terms
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(X)
        mu_X_opt    = _fitter.mu_
        Sigma_X_opt = _fitter.Sigma_

        # --- Stage 2: reduced conditional ML over [int, beta, ltau, lsig]
        # Pattern-constant terms are precomputed: F (conditionally filled
        # covariates) and the per-pattern conditional covariances stacked
        # into a (G, p, p) tensor, so each NLL call is pure arithmetic.
        F_all, cond_groups = prep_conditional_terms(X, mu_X_opt, Sigma_X_opt)
        G = len(cond_groups)
        group_id = np.empty(n, dtype=np.intp)
        M = np.zeros((G, p, p))
        for g, (rows, mis, Sc) in enumerate(cond_groups):
            group_id[rows] = g
            if mis.size:
                M[g][np.ix_(mis, mis)] = Sc

        # A vanishing ridge on the standardised fixed effects. It has no
        # effect when the design is well conditioned (the coefficients are
        # O(1) on the standardised scale, so the penalty is negligible), but
        # it regularises the unidentified directions when the features are
        # strongly collinear, giving a unique well-posed solution and stable
        # prediction for new groups instead of a coefficient vector that
        # drifts in the collinear null space. The penalty is excluded from
        # the reported log-likelihood, AIC and BIC.
        fe_ridge = float(getattr(self, 'fe_ridge', 1e-2))

        def _reduced_nll(theta_r):
            intercept = theta_r[0]
            beta      = theta_r[1:p + 1]
            # The other end of the same exponential. Underflow is floored at
            # the divisions below; overflow warns here and then propagates inf
            # through log1p into the objective, which the optimiser rejects
            # anyway, so the exponent is clamped to keep both ends finite and
            # the fit quiet. 700 is just below where exp leaves float64.
            tau_sq    = float(np.exp(min(2.0 * theta_r[p + 1], 700.0)))
            sigma_sq  = float(np.exp(min(2.0 * theta_r[p + 2], 700.0)))

            Mb  = M @ beta
            v_g = np.maximum(Mb @ beta, 0.0)
            delta_all  = v_g[group_id]
            mu_adj_all = F_all @ beta

            nll = 0.0
            for subj, row_idxs in subject_map.items():
                y_i        = y[row_idxs]
                obs_y_mask = ~np.isnan(y_i)
                if not obs_y_mask.any():
                    continue
                obs_global = row_idxs[obs_y_mask]
                n_obs      = int(obs_y_mask.sum())
                r     = y_i[obs_y_mask] - (intercept + mu_adj_all[obs_global])
                d_vec = sigma_sq + delta_all[obs_global]
                if np.any(d_vec <= 0) or not np.all(np.isfinite(d_vec)):
                    return np.inf
                d_inv     = 1.0 / d_vec
                JtDinvJ   = float(np.sum(d_inv))
                denom     = 1.0 / max(tau_sq, _TAU_SQ_FLOOR) + JtDinvJ
                log_det_V = float(np.sum(np.log(d_vec))) \
                    + np.log1p(tau_sq * JtDinvJ)
                Dinv_r    = d_inv * r
                Vinv_r    = Dinv_r - d_inv * (np.sum(Dinv_r) / denom)
                nll += 0.5 * (n_obs * np.log(2.0 * np.pi) + log_det_V
                              + float(r @ Vinv_r))
            return nll + 0.5 * fe_ridge * float(beta @ beta)

        theta0 = np.concatenate([
            [intercept0], beta0, [np.log(tau0)], [np.log(sigma0)]
        ])

        # The reduced NLL is optimised with a numerical gradient, which costs
        # 2(p+3) evaluations per step; on wide problems the default function
        # budget (15000) is exhausted long before convergence, so both the
        # function and iteration budgets are scaled with the parameter count.
        # Convergence is via the function-value tolerance (ftol): for an
        # unnormalised large-sample likelihood the gradient norm at the
        # optimum is naturally O(n), so the projected-gradient test (gtol) is
        # left loose and is not the operative stopping rule.
        result = minimize(
            fun=_reduced_nll,
            x0=theta0,
            method=self.method,
            options={
                'maxiter': max(self.max_iter, 300 * (p + 3)),
                'maxfun':  2000 * (p + 3),
                'ftol':    self.tol,
                'gtol':    max(self.tol, 1e-5),
            },
        )

        theta_opt = result.x
        # Standardised-scale estimates (everything below the EM was fitted on
        # standardised X and y; convert to the original scale at the end).
        b0_s   = float(theta_opt[0])
        beta_s = theta_opt[1:p + 1]
        # Same clamp as the objective uses, for the same reason and so the
        # reported variance components cannot disagree with the value the
        # likelihood was evaluated at. Written as exp(x)**2 rather than
        # exp(2x), this overflows once x passes about 354.
        tau_s2 = float(np.exp(min(2.0 * theta_opt[p + 1], 700.0)))
        sig_s2 = float(np.exp(min(2.0 * theta_opt[p + 2], 700.0)))

        # BLUPs on the standardised scale (uses self.* below), then scaled to y
        self.intercept_ = b0_s
        self.tau_sq_    = tau_s2
        self.sigma_sq_  = sig_s2
        _Mb        = M @ beta_s
        delta_all  = np.maximum(_Mb @ beta_s, 0.0)[group_id]
        mu_adj_all = F_all @ beta_s
        blup_s     = self._compute_blups(X, y, subject_map, mu_adj_all, delta_all)

        # Standard errors on the standardised scale (small reduced Hessian)
        if self.compute_se:
            try:
                H  = numerical_hessian(_reduced_nll, theta_opt, eps=1e-4)
                H += np.eye(len(theta_opt)) * 1e-8
                se_s = standard_errors_from_variance(np.linalg.inv(H))[:p + 1]
            except np.linalg.LinAlgError:
                se_s = np.full(p + 1, np.nan)
        else:
            se_s = np.full(p + 1, np.nan)

        # result.fun carries the fixed-effect ridge penalty; strip it so the
        # reported log-likelihood, AIC and BIC reflect the pure model fit.
        penalty_opt = 0.5 * fe_ridge * float(beta_s @ beta_s)
        loglik_std = (-float(result.fun) + penalty_opt
                      + _mvn_loglik(X, mu_X_opt, Sigma_X_opt))

        # Standardised-scale parameters, kept for the prediction paths
        self._beta_std_     = beta_s
        self._b0_std_       = b0_s
        self._tau_sq_std_   = tau_s2
        self._sigma_sq_std_ = sig_s2
        self._mu_X_std_     = mu_X_opt
        self._Sigma_X_std_  = Sigma_X_opt
        self._blup_std_     = blup_s

        # --- convert every public parameter back to the original scale -----
        self.coef_      = beta_s * (_sy / _sx)
        self.intercept_ = _my + _sy * b0_s - float(self.coef_ @ _mx)
        self.tau_sq_    = (_sy ** 2) * tau_s2
        self.sigma_sq_  = (_sy ** 2) * sig_s2
        self.mu_X_      = _mx + _sx * mu_X_opt
        self.Sigma_X_   = Sigma_X_opt * np.outer(_sx, _sx)
        self.icc_       = self.tau_sq_ / (self.tau_sq_ + self.sigma_sq_)
        self.converged_ = bool(result.success)
        if not self.converged_:
            warnings.warn(
                "%s: the likelihood optimiser stopped without converging "
                "after %d iterations (%s). The fitted parameters are wherever "
                "it happened to stop, not a maximum, so coefficients, "
                "standard errors and predictions should all be treated as "
                "unreliable. This is most often an ill-conditioned design: "
                "predictors on wildly different scales, near-duplicate "
                "columns, or strongly non-normal margins that the joint "
                "Gaussian working model cannot represent. Consider "
                "copula='auto', which maps each margin to a normal score, or "
                "removing redundant predictors."
                % (type(self).__name__, int(getattr(result, 'nit', 0)),
                   str(getattr(result, 'message', 'no message'))[:80]),
                UserWarning, stacklevel=2,
            )
        # scikit-learn expects n_iter_ wherever max_iter is exposed, and
        # both penalised siblings set it here. These two did not, so
        # check_non_transformer_estimators_n_iter failed on them alone.
        self.n_iter_ = int(getattr(result, 'nit', 0))
        self.blup_      = {g: _sy * b for g, b in blup_s.items()}

        # Full-information log-likelihood with the Jacobian of the X/y
        # standardisation (matches the penalized-model convention).
        _n_obs_col   = np.sum(~np.isnan(X), axis=0)
        _jac         = -float(_n_obs_col @ np.log(_sx)) - _ny * np.log(_sy)
        self.loglik_ = loglik_std + _jac
        n_params     = (p + 3) + p + p * (p + 1) // 2
        self.aic_    = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_    = n_params * np.log(n) - 2.0 * self.loglik_

        # Standard errors: exact rescale for slopes; diagonal delta for the
        # intercept (covariances with slopes ignored, as in the other models).
        se_beta = se_s[1:] * (_sy / _sx)
        se_int  = _sy * np.sqrt(
            se_s[0] ** 2 + float(np.sum(((_mx / _sx) * se_s[1:]) ** 2))
        )
        self.se_ = np.concatenate([[se_int], se_beta])

        coefs_all     = np.concatenate([[self.intercept_], self.coef_])
        safe_se       = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_ = coefs_all / safe_se
        self.pvalues_ = self._pvalues_from_zstat(self.z_stats_)

        # Standardized coefficients (from raw-scale parameters)
        sigma_X          = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        sigma_Y_marginal = np.sqrt(max(
            self.sigma_sq_ + self.tau_sq_
            + float(self.coef_ @ self.Sigma_X_ @ self.coef_),
            1e-12,
        ))
        self.coef_std_ = self.coef_ * sigma_X / sigma_Y_marginal
        self.sigma_X_  = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, X, groups=None):
        """
        Predict continuous outcomes.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None
            Subject labels.  Training subjects receive their BLUP adjustment;
            new subjects receive zero random effect (population-level).

        Returns
        -------
        y_hat : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)

        # All conditional solves run in the standardised space of the fit;
        # raw-unit solves are exact in principle but numerically unstable
        # when the feature scales span orders of magnitude.
        Xs = (X - self._x_mean_) / self._x_scale_
        beta      = self._beta_std_
        intercept = self._b0_std_
        mu_X      = self._mu_X_std_
        Sigma_X   = self._Sigma_X_std_
        groups_arr = np.asarray(groups) if groups is not None else None

        x_pat          = _build_x_patterns(Xs)
        mu_adj_all, _  = _row_contribs_batched(Xs, x_pat, beta, mu_X, Sigma_X)
        y_hat          = intercept + mu_adj_all

        if groups_arr is not None:
            for i, g in enumerate(groups_arr):
                y_hat[i] += self._blup_std_.get(g, 0.0)

        y_hat = self._y_mean_ + self._y_scale_ * y_hat

        if self.copula_used_:
            y_hat = self._copula_y_.inverse_transform_1d(y_hat, col=0)

        return y_hat

    def predict_interval(self, X, groups=None, alpha=0.05):
        """
        Prediction intervals accounting for all sources of uncertainty.

        Variance components:
          - sigma^2        : within-subject residual noise (always present)
          - delta_j        : uncertainty from missing covariates in row j
          - tau^2          : between-subject random-effect variance
                             (included for unknown subjects; absorbed by the
                             BLUP for known subjects)

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None
            Known training subjects receive BLUP adjustment and narrower
            intervals.  Omit (or pass None) for population-level intervals.
        alpha : float
            Significance level (default 0.05 for 95% interval).

        Returns
        -------
        lower, upper : ndarray, shape (n,)
        """
        check_is_fitted(self)
        X_arr = self._validate_and_convert(X)
        if self.copula_used_:
            X_arr = self._copula_X_.transform(X_arr)

        # Standardised space throughout, as in predict()
        Xs        = (X_arr - self._x_mean_) / self._x_scale_
        z         = norm.ppf(1.0 - alpha / 2.0)
        beta      = self._beta_std_
        mu_X      = self._mu_X_std_
        Sigma_X   = self._Sigma_X_std_
        tau_sq    = self._tau_sq_std_
        sigma_sq  = self._sigma_sq_std_
        intercept = self._b0_std_
        n         = Xs.shape[0]

        groups_arr = np.asarray(groups) if groups is not None else None

        x_pat                    = _build_x_patterns(Xs)
        mu_adj_all, delta_all    = _row_contribs_batched(
            Xs, x_pat, beta, mu_X, Sigma_X
        )
        y_hat   = intercept + mu_adj_all
        se_pred = np.empty(n)

        for i in range(n):
            delta = delta_all[i]
            if groups_arr is not None:
                g         = groups_arr[i]
                b_hat     = self._blup_std_.get(g, 0.0)
                y_hat[i] += b_hat
                var_i     = sigma_sq + delta           # BLUP absorbs tau^2
            else:
                var_i     = sigma_sq + tau_sq + delta  # full marginal variance
            se_pred[i] = np.sqrt(max(var_i, 0.0))

        y_hat   = self._y_mean_ + self._y_scale_ * y_hat
        se_pred = self._y_scale_ * se_pred
        lower = y_hat - z * se_pred
        upper = y_hat + z * se_pred

        if self.copula_used_:
            lower = self._copula_y_.inverse_transform_1d(lower, col=0)
            upper = self._copula_y_.inverse_transform_1d(upper, col=0)

        return lower, upper

    def score(self, X, y, groups=None):
        """
        R² on the subset of observations where y is not NaN.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        groups : array-like, shape (n,) or None

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X, y  = self._validate_and_convert(X, y)
        y_hat = self.predict(X, groups=groups)
        obs   = ~np.isnan(y)
        y_o, yh_o = y[obs], y_hat[obs]
        ss_res    = np.sum((y_o - yh_o) ** 2)
        ss_tot    = np.sum((y_o - np.mean(y_o)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self, alpha=0.05):
        """
        Print a comprehensive model summary including fixed effects,
        random-effect variance, ICC, and feature importances.

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self)
        W = 74
        print()
        print('=' * W)
        print('MissMixedRegressor  --  FIML Random-Intercept LME'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Subjects/Groups : {self.n_groups_}")
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
        print(f"  tau (rand. int) : {np.sqrt(self.tau_sq_):.4f}  "
              f"(var: {self.tau_sq_:.4f})")
        print(f"  sigma (resid)   : {np.sqrt(self.sigma_sq_):.4f}  "
              f"(var: {self.sigma_sq_:.4f})")
        print(f"  ICC             : {self.icc_:.4f}")
        print('-' * W)
        print("  Fixed Effects:")
        for line in self._coef_table_lines(stat_label='z_stat', alpha=alpha):
            print('    ' + line)
        print()
        print("  Feature Importances (normalized standardized |coef|):")
        for line in self._importance_lines():
            print(line)
        print('=' * W)
        print()


# ======================================================================
# MissMixedClassifier
# ======================================================================

class MissMixedClassifier(ClassifierMixin, MissBase):
    """
    FIML random-intercept GLMM for binary outcomes with native missing
    data support.

    Fits the model:
        y_ij | b_i ~ Bernoulli(sigma(beta_0 + x_ij^T beta + b_i)),
        b_i ~ N(0, tau^2).

    The random intercept b_i is integrated out numerically via
    *adaptive* Gauss-Hermite quadrature: for each subject the nodes are
    recentred on the mode of that subject's integrand and rescaled to the
    curvature there (Liu and Pierce 1994), rather than being placed on the
    shared prior. A subject with many observations has a posterior for b_i
    far narrower than the prior, and unadapted nodes then sample almost
    entirely where the integrand is negligible.  Missing covariates in X are
    handled
    using the probit approximation: for row j with missing predictors,

        P(y_ij = 1 | x_ij_obs, b_i)
            ~ sigma((eta_base_ij + b_i) / sqrt(1 + pi/8 * v_j))

    where eta_base_ij = E[x_ij^T beta | x_ij_obs] and v_j is the
    conditional variance of the missing-feature linear contribution.

    Parameters
    ----------
    n_quadrature : int
        Number of adaptive Gauss-Hermite nodes for integrating over b_i
        (default 20). The nodes are recentred on each subject's own posterior
        mode and scaled to the curvature there, so 20 is accurate well past
        the random-effect scales that the unadapted rule could hold.
    max_iter : int
        Maximum L-BFGS-B iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    compute_se : bool
        If True (default), compute standard errors from the Hessian diagonal.
    copula : bool or 'auto', default 'auto'
        Copula transform for X (y is binary; it is never transformed).

    Attributes
    ----------
    coef_ : ndarray, shape (p,)
    intercept_ : float
    tau_sq_ : float
        Estimated random-intercept variance.
    classes_ : ndarray, shape (2,)
    odds_ratios_ : ndarray, shape (p+1,)
    blup_ : dict {group_label: float}
        Posterior-mean BLUPs of the random intercept for training subjects.
    n_groups_ : int
    se_ : ndarray, shape (p+1,)
    z_stats_, pvalues_ : ndarray, shape (p+1,)
    coef_std_ : ndarray, shape (p,)
    mu_X_ : ndarray, shape (p,)
    Sigma_X_ : ndarray, shape (p, p)
    loglik_, aic_, bic_ : float
    converged_ : bool
    copula_used_ : bool
    """

    def __init__(self, n_quadrature=20, max_iter=2000, tol=1e-7,
                 method='L-BFGS-B', compute_se=True, copula='auto'):
        self.n_quadrature = n_quadrature
        self.max_iter     = max_iter
        self.tol          = tol
        self.method       = method
        self.compute_se   = compute_se
        self.copula       = copula

    # ------------------------------------------------------------------ #
    # Parameter packing / unpacking
    # ------------------------------------------------------------------ #

    def _pack_params(self, intercept, beta, tau, mu_X, Sigma_X):
        L_vec, _ = pack_cholesky(Sigma_X)
        return np.concatenate(
            [[intercept], beta, [np.log(tau)], mu_X, L_vec]
        )

    # ------------------------------------------------------------------ #
    # Subject log-likelihood (GH quadrature)
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Negative log-likelihood
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # BLUPs (posterior mean of b_i via GH quadrature)
    # ------------------------------------------------------------------ #

    def _compute_blups(self, X, y, subject_map, gh_t, gh_w):
        """
        Compute the posterior mean E[b_i | y_i_obs, X_i_obs] for each
        training subject using the GH quadrature already set up for the NLL.

        E[b_i | data_i] = (1/Z_i) integral b * P(y_i|X_i,b) N(b;0,tau^2) db
                        ~ Sigma_m (b_m * w_unnorm_m) / Sigma_m w_unnorm_m
        """
        beta      = self.coef_
        intercept = self.intercept_
        tau       = np.sqrt(self.tau_sq_)
        mu_X      = self.mu_X_
        Sigma_X   = self.Sigma_X_

        tau_sq      = max(tau * tau, _TAU_SQ_FLOOR)
        gh_log_w    = np.log(gh_w)
        n_q         = len(gh_t)

        # Precompute eta_base and v_miss for all rows using batched patterns
        x_patterns   = _build_x_patterns(X)
        mu_adj_all, delta_all = _row_contribs_batched(
            X, x_patterns, beta, mu_X, Sigma_X
        )
        eta_base_all = intercept + mu_adj_all

        blup = {}
        for subj, row_idxs in subject_map.items():
            y_i        = y[row_idxs]
            obs_y_mask = ~np.isnan(y_i)

            if not obs_y_mask.any():
                blup[subj] = 0.0
                continue

            obs_global = row_idxs[obs_y_mask]
            y_obs      = y_i[obs_y_mask]
            eta_base   = eta_base_all[obs_global]
            v_miss     = delta_all[obs_global]

            scale      = np.sqrt(1.0 + _PI_OVER_8 * v_miss)
            inv_scale  = 1.0 / scale
            eta_scaled = eta_base * inv_scale
            sign       = 2.0 * y_obs - 1.0

            # The same adaptive rule the likelihood uses. A posterior mean is
            # a ratio, so a shared error in numerator and denominator cancels
            # in part, but not once the nodes miss the posterior altogether,
            # which is precisely what they do at large tau.
            b_nodes, log_w = _adaptive_nodes(
                eta_scaled, sign, inv_scale, tau_sq, gh_t, gh_log_w
            )
            log_p_nodes = log_w + _subject_log_integrand(
                b_nodes, eta_scaled, sign, inv_scale, tau_sq
            )
            max_lp         = log_p_nodes.max()
            weights_unnorm = np.exp(log_p_nodes - max_lp)
            Z              = weights_unnorm.sum()
            blup[subj]     = float(np.dot(b_nodes, weights_unnorm) / Z)

        return blup

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y, groups=None):
        """
        Fit the FIML random-intercept GLMM.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    Binary {0, 1}.  NaN allowed.
        groups : array-like, shape (n,).  Subject / cluster labels.

        Returns
        -------
        self
        """
        X, y = self._validate_and_convert(X, y)
        groups = _resolve_groups(groups, len(y))
        # Checked before the reorder, which zips the three together and
        # otherwise fails inside numpy with a concatenation message that
        # names neither groups nor the lengths involved.
        if len(np.asarray(groups)) != len(y):
            raise ValueError(
                "%s: groups has %d entries but X and y have %d rows. Each "
                "row needs the label of the subject it belongs to."
                % (type(self).__name__, len(np.asarray(groups)), len(y)))
        # X, y and groups are reordered together: permuting two of the three
        # would attach observations to the wrong subject.
        X, y, groups = self._canonical_fit_order_with_groups(X, y, groups)
        self._store_fit_metadata(X, y)
        self.groups_fit_ = np.asarray(groups)
        n, p = X.shape

        y_obs_vals    = y[~np.isnan(y)]
        self.classes_ = np.array(sorted(set(y_obs_vals.tolist())))
        if len(self.classes_) != 2:
            raise ValueError(
                f"MissMixedClassifier requires exactly 2 classes; "
                f"found {self.classes_}."
            )

        # Binarize against classes_[1]: the Bernoulli likelihood (and the
        # BLUP code) encode outcomes via sign = 2y - 1, which requires
        # y ∈ {0, 1}.  NaN (missing y) preserved.
        y = np.where(np.isnan(y), np.nan,
                     (y == self.classes_[1]).astype(float))

        # Copula applies to X only (y is binary; it is never transformed)
        if self.copula == 'auto':
            self.copula_used_ = needs_copula(X)
        else:
            self.copula_used_ = bool(self.copula)

        if self.copula_used_:
            self._copula_X_ = RankNormalTransformer().fit(X)
            X               = self._copula_X_.transform(X)

        # Internal standardisation of X (the convention of MissMixedRegressor
        # and the penalized models).  The conditional solves in stage two are
        # ill-conditioned when the features differ widely in scale, exactly
        # the failure mode that broke the regressor on the Parkinsons voice
        # features.  y is binary and the random intercept lives on the logit
        # scale, so tau and the BLUPs are invariant to this rescaling; only
        # the fixed effects transform, and every public attribute is converted
        # back to the original feature scale after the fit.
        _mx = np.nanmean(X, axis=0)
        _mx = np.where(np.isfinite(_mx), _mx, 0.0)
        _sx = feature_scale(X)
        X = (X - _mx) / _sx
        self._x_mean_, self._x_scale_ = _mx, _sx

        subject_map    = _group_by_subject(groups)
        self.n_groups_ = len(subject_map)

        # Pre-compute missingness patterns once (reused across all NLL calls)
        x_patterns = _build_x_patterns(X)

        # GH nodes for integrating over the random intercept
        gh_t, gh_w = np.polynomial.hermite.hermgauss(self.n_quadrature)

        # Seed the optimiser from complete cases; if there are too few (large
        # p or high missingness) fall back to a mean-imputed seed rather than
        # failing.  The FIML likelihood below still integrates over every
        # observed entry, so this only sets the starting point.
        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_cc          = X[complete_mask]
        y_cc          = y[complete_mask]

        mu_X0 = np.nanmean(X, axis=0)
        mu_X0 = np.where(np.isfinite(mu_X0), mu_X0, 0.0)

        if len(y_cc) >= p + 2:
            X_seed, y_seed = X_cc, y_cc
        else:
            obs_y  = ~np.isnan(y)
            X_imp  = np.where(np.isnan(X), mu_X0, X)
            X_seed, y_seed = X_imp[obs_y], y[obs_y]

        from sklearn.linear_model import LogisticRegression
        # If the seed contains a single class, fall back to a zero seed rather
        # than failing (same behaviour as MissLogistic).
        if len(np.unique(y_seed)) >= 2:
            lr = LogisticRegression(max_iter=500, C=1.0)
            lr.fit(X_seed, y_seed)
            beta0      = lr.coef_.ravel()
            intercept0 = float(lr.intercept_[0])
        else:
            beta0      = np.zeros(p)
            intercept0 = 0.0
        tau0       = 1.0

        # --- Stage 1: X-moments by EM (full-information) -----------------
        # Removes p + p(p+1)/2 dimensions from the optimizer and SE Hessian.
        from ._imputer import _JointMVNFitter, _mvn_loglik
        from ._utils import prep_conditional_terms
        _fitter = _JointMVNFitter(max_iter=100, tol=1e-5, reg=1e-6)
        _fitter.fit(X)
        mu_X_opt    = _fitter.mu_
        Sigma_X_opt = _fitter.Sigma_

        # --- Stage 2: reduced conditional ML over [int, beta, log_tau] ---
        F_all, cond_groups = prep_conditional_terms(X, mu_X_opt, Sigma_X_opt)
        G = len(cond_groups)
        group_id = np.empty(n, dtype=np.intp)
        M = np.zeros((G, p, p))
        for g, (rows, mis, Sc) in enumerate(cond_groups):
            group_id[rows] = g
            if mis.size:
                M[g][np.ix_(mis, mis)] = Sc
        _gh_log_w = np.log(gh_w)

        def _reduced_nll(theta_r):
            intercept = theta_r[0]
            beta      = theta_r[1:p + 1]
            tau       = float(np.exp(theta_r[p + 1]))
            tau_sq    = max(tau * tau, _TAU_SQ_FLOOR)

            Mb  = M @ beta
            v_g = np.maximum(Mb @ beta, 0.0)
            delta_all    = v_g[group_id]
            eta_base_all = intercept + F_all @ beta

            nll = 0.0
            for subj, row_idxs in subject_map.items():
                y_i        = y[row_idxs]
                obs_y_mask = ~np.isnan(y_i)
                if not obs_y_mask.any():
                    continue
                obs_global = row_idxs[obs_y_mask]
                y_obs      = y_i[obs_y_mask]
                scale      = np.sqrt(1.0 + _PI_OVER_8 * delta_all[obs_global])
                inv_scale  = 1.0 / scale
                eta_scaled = eta_base_all[obs_global] * inv_scale
                sign       = 2.0 * y_obs - 1.0

                # Nodes placed on this subject's own posterior rather than on
                # the shared prior; see _adaptive_nodes for why the unadapted
                # rule loses whole nats once a subject carries several
                # observations and tau is large.
                b_nodes, log_w = _adaptive_nodes(
                    eta_scaled, sign, inv_scale, tau_sq, gh_t, _gh_log_w
                )
                log_p_nodes = log_w + _subject_log_integrand(
                    b_nodes, eta_scaled, sign, inv_scale, tau_sq
                )
                max_lp    = log_p_nodes.max()
                log_lik_i = max_lp + np.log(np.sum(np.exp(log_p_nodes - max_lp)))
                if not np.isfinite(log_lik_i):
                    return np.inf
                nll -= float(log_lik_i)
            return nll

        theta0 = np.concatenate([[intercept0], beta0, [np.log(tau0)]])

        # Bound log(tau): with near-separable data the GLMM likelihood is
        # almost flat in large tau, and an unbounded scale can drift to
        # astronomically large random-effect variances that break the GH
        # integration and new-subject predictions.  tau in [1e-3, 30] covers
        # every realistic logistic random-intercept scale.
        _bounds = ([(None, None)] * (p + 1)
                   + [(np.log(1e-3), np.log(30.0))])

        result = minimize(
            fun=_reduced_nll,
            x0=theta0,
            method=self.method,
            bounds=_bounds,
            options={
                'maxiter': self.max_iter,
                'ftol':    self.tol,
                'gtol':    max(self.tol, 1e-5),
            },
        )

        theta_opt = result.x
        intercept_opt = float(theta_opt[0])
        beta_opt      = theta_opt[1:p + 1]
        tau_opt       = float(np.exp(theta_opt[p + 1]))

        self.converged_ = bool(result.success)
        if not self.converged_:
            warnings.warn(
                "%s: the likelihood optimiser stopped without converging "
                "after %d iterations (%s). The fitted parameters are wherever "
                "it happened to stop, not a maximum, so coefficients, "
                "standard errors and predictions should all be treated as "
                "unreliable. This is most often an ill-conditioned design: "
                "predictors on wildly different scales, near-duplicate "
                "columns, or strongly non-normal margins that the joint "
                "Gaussian working model cannot represent. Consider "
                "copula='auto', which maps each margin to a normal score, or "
                "removing redundant predictors."
                % (type(self).__name__, int(getattr(result, 'nit', 0)),
                   str(getattr(result, 'message', 'no message'))[:80]),
                UserWarning, stacklevel=2,
            )
        # scikit-learn expects n_iter_ wherever max_iter is exposed, and
        # both penalised siblings set it here. These two did not, so
        # check_non_transformer_estimators_n_iter failed on them alone.
        self.n_iter_ = int(getattr(result, 'nit', 0))

        # Standardised-scale estimates are set on self first so _compute_blups
        # (which reads self.coef_/intercept_/mu_X_/Sigma_X_) runs in the same
        # standardised space as the fit; the BLUPs are on the logit scale and
        # therefore invariant to the X rescaling.
        self.intercept_ = intercept_opt
        self.coef_      = beta_opt
        self.tau_sq_    = tau_opt ** 2
        self.mu_X_      = mu_X_opt
        self.Sigma_X_   = Sigma_X_opt
        blup = self._compute_blups(X, y, subject_map, gh_t, gh_w)

        # Standardised-scale parameters, kept for the prediction paths
        self._beta_std_    = beta_opt
        self._b0_std_      = intercept_opt
        self._mu_X_std_    = mu_X_opt
        self._Sigma_X_std_ = Sigma_X_opt
        self._blup_std_    = blup

        # Conditional log-likelihood (invariant to X standardisation) plus the
        # X-marginal on the standardised scale.
        loglik_std = -float(result.fun) + _mvn_loglik(X, mu_X_opt, Sigma_X_opt)

        # --- convert public parameters back to the original feature scale ---
        self.coef_      = beta_opt / _sx
        self.intercept_ = intercept_opt - float(self.coef_ @ _mx)
        self.tau_sq_    = tau_opt ** 2                 # invariant (logit scale)
        self.mu_X_      = _mx + _sx * mu_X_opt
        self.Sigma_X_   = Sigma_X_opt * np.outer(_sx, _sx)
        self.blup_      = blup                          # invariant (logit scale)

        # Full-information log-likelihood with the Jacobian of the X
        # standardisation (y is binary, so no y term).
        _n_obs_col   = np.sum(~np.isnan(X), axis=0)
        _jac         = -float(_n_obs_col @ np.log(_sx))
        self.loglik_ = loglik_std + _jac
        n_params     = (p + 2) + p + p * (p + 1) // 2
        self.aic_    = 2.0 * n_params - 2.0 * self.loglik_
        self.bic_    = n_params * np.log(n) - 2.0 * self.loglik_

        # Standard errors from the (small) reduced numerical Hessian, computed
        # on the standardised scale then mapped to the original scale.
        if self.compute_se:
            try:
                H  = numerical_hessian(_reduced_nll, theta_opt, eps=1e-4)
                H += np.eye(len(theta_opt)) * 1e-8
                Var      = np.linalg.inv(H)
                se_s     = standard_errors_from_variance(Var)[:p + 1]
            except np.linalg.LinAlgError:
                se_s = np.full(p + 1, np.nan)
        else:
            se_s = np.full(p + 1, np.nan)

        # Exact rescale for the slopes; diagonal delta for the intercept
        # (covariances with the slopes ignored, as in the other models).
        se_beta  = se_s[1:] / _sx
        se_int   = np.sqrt(se_s[0] ** 2
                           + float(np.sum(((_mx / _sx) * se_s[1:]) ** 2)))
        self.se_ = np.concatenate([[se_int], se_beta])

        coefs_all         = np.concatenate([[self.intercept_], self.coef_])
        safe_se           = np.where(self.se_ > 0, self.se_, np.nan)
        self.z_stats_     = coefs_all / safe_se
        self.pvalues_     = self._pvalues_from_zstat(self.z_stats_)
        with np.errstate(over='ignore'):
            self.odds_ratios_ = np.exp(coefs_all)

        # Standardized coefficients
        sigma_X          = np.sqrt(np.maximum(np.diag(self.Sigma_X_), 1e-12))
        # Approximate marginal std of logit: sqrt(pi^2/3 + tau^2)
        sigma_Y          = np.sqrt(np.pi ** 2 / 3.0 + self.tau_sq_)
        self.coef_std_   = self.coef_ * sigma_X / sigma_Y
        self.sigma_X_    = sigma_X

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_proba(self, X, groups=None):
        """
        Predict class probabilities via GH quadrature (new subjects) or
        BLUP-adjusted sigmoid (known training subjects).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

        Returns
        -------
        proba : ndarray, shape (n, 2).  Columns: [P(y=0), P(y=1)].
        """
        check_is_fitted(self)
        X = self._validate_and_convert(X)
        if self.copula_used_:
            X = self._copula_X_.transform(X)

        # All conditional solves run in the standardised space of the fit; the
        # resulting logit is identical to a raw-space computation but numerically
        # stable when the feature scales span orders of magnitude.
        X          = (X - self._x_mean_) / self._x_scale_
        beta       = self._beta_std_
        intercept  = self._b0_std_
        tau        = np.sqrt(self.tau_sq_)   # invariant to X standardisation
        mu_X       = self._mu_X_std_
        Sigma_X    = self._Sigma_X_std_
        blup_map   = self._blup_std_
        n          = X.shape[0]
        p1         = np.empty(n)

        groups_arr = np.asarray(groups) if groups is not None else None

        # Batched row contributions
        x_patterns_pred          = _build_x_patterns(X)
        mu_adj_all, delta_all    = _row_contribs_batched(
            X, x_patterns_pred, beta, mu_X, Sigma_X
        )
        eta_base_all = intercept + mu_adj_all    # (n,)
        scale_all    = np.sqrt(1.0 + _PI_OVER_8 * delta_all)  # (n,)
        eta_scaled_all = eta_base_all / scale_all             # (n,)

        for i in range(n):
            scale_i    = scale_all[i]
            eta_sc_i   = eta_scaled_all[i]

            if groups_arr is not None and groups_arr[i] in blup_map:
                # Known subject: shift by BLUP (logit scale, invariant)
                b_hat   = blup_map[groups_arr[i]]
                eta_eff = eta_sc_i + b_hat / scale_i
                p1[i]   = float(1.0 / (1.0 + np.exp(
                    -np.clip(eta_eff, -500.0, 500.0)
                )))
            else:
                # Unknown subject: no observations to condition on, so this is
                # the plain expectation of a logistic over the random-effect
                # prior. That is exactly integrate_logistic_normal, which is
                # accurate for any variance; Gauss-Hermite on its own lost
                # 0.03 in the probability once tau/scale grew.
                p1[i] = float(integrate_logistic_normal(
                    np.array([eta_sc_i]), float((tau / scale_i) ** 2)
                )[0])

        p1 = np.clip(p1, 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, groups=None):
        """
        Predict class labels (0 or 1).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

        Returns
        -------
        y_pred : ndarray, shape (n,), labels from ``self.classes_``
        """
        check_is_fitted(self)
        proba = self.predict_proba(X, groups=groups)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def decision_function(self, X, groups=None):
        """
        Log-odds of class 1.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

        Returns
        -------
        scores : ndarray, shape (n,)
        """
        check_is_fitted(self)
        proba = self.predict_proba(X, groups=groups)
        p     = np.clip(proba[:, 1], 1e-15, 1.0 - 1e-15)
        return np.log(p / (1.0 - p))

    def score(self, X, y, groups=None):
        """
        Accuracy on non-NaN observed outcomes.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        groups : array-like, shape (n,) or None

        Returns
        -------
        float
        """
        check_is_fitted(self)
        X_arr, y_arr = self._validate_and_convert(X, y)
        y_pred       = self.predict(X_arr, groups=groups)
        obs          = ~np.isnan(y_arr)
        if not obs.any():
            return 0.0
        return float(np.mean(y_pred[obs] == y_arr[obs]))

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
        W = 78
        print()
        print('=' * W)
        print('MissMixedClassifier  --  FIML Random-Intercept GLMM'.center(W))
        print('=' * W)
        print(f"  Observations    : {self.n_samples_fit_}  "
              f"(complete: {self.n_complete_},  partial: {self.n_partial_})")
        print(f"  Subjects/Groups : {self.n_groups_}")
        print(f"  Features        : {self.n_features_in_}")
        print(f"  Classes         : {list(self.classes_)}")
        print(f"  Missing (X)     : {self.n_missing_X_} "
              f"({self.missing_rate_X_ * 100:.1f}%)")
        if self.n_missing_y_ > 0:
            print(f"  Missing (y)     : {self.n_missing_y_}")
        print(f"  Converged       : {self.converged_}")
        if self.copula == 'auto':
            label = 'auto (applied)' if self.copula_used_ else 'auto (not applied)'
            print(f"  Copula          : {label}")
        elif self.copula_used_:
            print(f"  Copula          : yes  (X on normal-transformed scale)")
        print(f"  Log-likelihood  : {self.loglik_:.4f}")
        print(f"  AIC             : {self.aic_:.4f}   BIC: {self.bic_:.4f}")
        print(f"  tau (rand. int) : {np.sqrt(self.tau_sq_):.4f}  "
              f"(var: {self.tau_sq_:.4f})")
        print('-' * W)
        print("  Fixed Effects (log-odds scale) with Odds Ratios:")

        or_header = f"{'odds_ratio':>12}"
        or_rows   = [f"{self.odds_ratios_[i]:>12.4f}"
                     for i in range(len(self.odds_ratios_))]

        for line in self._coef_table_lines(
            stat_label='z_stat', alpha=alpha,
            extra_header=or_header, extra_rows=or_rows,
        ):
            print('    ' + line)

        print()
        print("  Feature Importances (normalized standardized |coef|):")
        for line in self._importance_lines():
            print(line)
        print('=' * W)
        print()


# ======================================================================
# MissMixed  --  unified auto-selecting wrapper
# ======================================================================

class MissMixed(MissTags, BaseEstimator):
    """
    Unified FIML mixed-effects model that automatically selects between
    regression (MissMixedRegressor) and classification (MissMixedClassifier)
    based on the observed values of y.

    Detection rule (applied at fit time):
        If every observed (non-NaN) value of y is in {0, 1}  ->  classification
        Otherwise                                             ->  regression

    After fitting, the underlying model is stored in ``model_`` and the
    detected task in ``task_`` ('regression' or 'classification').
    All public methods (predict, score, summary, etc.) are forwarded to
    ``model_``; fitted attributes (``coef_``, ``intercept_``, ``blup_``, ...) are also
    accessible directly on this object via attribute delegation.

    Parameters
    ----------
    n_quadrature : int
        Adaptive Gauss-Hermite nodes (classification only, default 20).
    max_iter : int
        Maximum optimizer iterations (default 2000).
    tol : float
        Convergence tolerance (default 1e-7).
    method : str
        scipy.optimize.minimize method (default 'L-BFGS-B').
    compute_se : bool
        Compute standard errors (default True).
    copula : bool or 'auto', default 'auto'
        Gaussian copula transform. 'auto' (the default) applies it
        only when the marginals are skewed or heavy-tailed enough to
        warrant it; True forces it on and False off.
    """

    def __init__(self, n_quadrature=20, max_iter=2000, tol=1e-7,
                 method='L-BFGS-B', compute_se=True, copula='auto'):
        self.n_quadrature = n_quadrature
        self.max_iter     = max_iter
        self.tol          = tol
        self.method       = method
        self.compute_se   = compute_se
        self.copula       = copula

    # ------------------------------------------------------------------ #
    # Task detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_task(y):
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        obs   = y_arr[~np.isnan(y_arr)]
        if len(obs) == 0:
            raise ValueError("y has no observed (non-NaN) values.")
        if set(np.unique(obs).tolist()) <= {0.0, 1.0}:
            return 'classification'
        return 'regression'

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X, y, groups=None):
        """
        Detect task and fit the appropriate FIML mixed-effects model.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN values treated as missing.
        y : array-like, shape (n,).    NaN values treated as missing.
        groups : array-like, shape (n,).  Subject / cluster labels.

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
            self.model_ = MissMixedClassifier(
                n_quadrature=self.n_quadrature,
                max_iter=self.max_iter,
                tol=self.tol,
                method=self.method,
                compute_se=self.compute_se,
                copula=self.copula,
            )
        else:
            self.model_ = MissMixedRegressor(
                max_iter=self.max_iter,
                tol=self.tol,
                method=self.method,
                compute_se=self.compute_se,
                copula=self.copula,
            )

        self.model_.fit(X, y, groups=groups)
        return self

    # ------------------------------------------------------------------ #
    # Attribute delegation
    # ------------------------------------------------------------------ #

    def __getattr__(self, name):
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

    def predict(self, X, groups=None):
        """
        Predict outcomes (regression) or class labels (classification).

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

        Returns
        -------
        ndarray, shape (n,)
        """
        check_is_fitted(self, 'model_')
        return self.model_.predict(X, groups=groups)

    @available_if(only_for('classification'))
    def predict_proba(self, X, groups=None):
        """
        Predict class probabilities.  Only available for classification.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

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
        return self.model_.predict_proba(X, groups=groups)

    @available_if(only_for('classification'))
    def decision_function(self, X, groups=None):
        """
        Log-odds scores.  Only available for classification.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None

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
        return self.model_.decision_function(X, groups=groups)

    @available_if(only_for('regression'))
    def predict_interval(self, X, groups=None, alpha=0.05):
        """
        Prediction intervals.  Only available for regression.

        Parameters
        ----------
        X : array-like, shape (n, p).  NaN allowed.
        groups : array-like, shape (n,) or None
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
        return self.model_.predict_interval(X, groups=groups, alpha=alpha)

    def score(self, X, y, groups=None):
        """
        R² (regression) or accuracy (classification) on non-NaN targets.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        groups : array-like, shape (n,) or None

        Returns
        -------
        float
        """
        check_is_fitted(self, 'model_')
        return self.model_.score(X, y, groups=groups)

    def summary(self, alpha=0.05):
        """
        Print the model summary (delegates to the underlying model).

        Parameters
        ----------
        alpha : float, CI significance level (default 0.05).
        """
        check_is_fitted(self, 'model_')
        return self.model_.summary(alpha=alpha)
