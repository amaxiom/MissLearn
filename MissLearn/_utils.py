"""
MissLearn._utils
-------------
Core numerical utilities shared by MissLinear and MissLogistic.
All functions are stateless and operate on numpy arrays.
"""

import warnings

import numpy as np
from collections import defaultdict
from scipy.special import expit, ndtr

# Module-level cache for Gauss-Hermite nodes and weights, keyed by n_points.
# Avoids recomputing hermgauss on every NLL evaluation during optimization.
# Lives for the process lifetime.  Call clear_cache() to release if needed.
_gh_cache: dict = {}

# Gauss-Legendre nodes for the wide-variance branch of
# integrate_logistic_normal, keyed by node count. Same lifetime as _gh_cache.
_gl_cache: dict = {}

#: Above this variance the Gauss-Hermite branch is replaced. Gauss-Hermite is
#: exact to about 1e-11 below it and the replacement is exact to 1e-15 above,
#: so the two agree to far better than either matters at the crossover.
_SPLIT_ABOVE_V = 1.0

#: Half-width of the correction integral. The remainder it integrates is
#: bounded by exp(-|s|), so the tail past 25 is under 1.4e-11 and truncating
#: there costs about 1e-13. Widening the panel does not buy accuracy, it
#: spends it: the nodes spread over ground the integrand has already vacated,
#: and the same 40 nodes per panel that reach 2e-13 at a half-width of 25
#: manage only 3.5e-9 at 40. Narrower is worse again, since below about 20 the
#: truncated tail starts to dominate at large v.
_SPLIT_HALF_WIDTH = 25.0

#: Nodes per panel for the correction integral. The integrand is analytic on
#: each panel, so convergence is spectral: 40 per panel holds the worst error
#: over the whole range to 4e-9, reached at the crossover variance where the
#: Gaussian is narrowest against the fixed panel width. Doubling it to 80 buys
#: machine precision, which nothing here needs, at twice the working array.
_SPLIT_NODES = 40


def clear_cache():
    """Release cached Gauss-Hermite nodes and weights."""
    _gh_cache.clear()


# ============================================================
# Activation
# ============================================================

def sigmoid(x):
    """Numerically stable sigmoid via scipy.special.expit."""
    return expit(x)


# ============================================================
# Conditional-likelihood precomputation (two-stage FIML fits)
# ============================================================

def prep_conditional_terms(X_rows, mu_X, Sigma_X):
    """
    Precompute the pattern-constant quantities of the conditional likelihood
    p(y | X_obs) given fixed X-moments (mu_X, Sigma_X):

        F      : X_rows with missing entries replaced by conditional means
                 E[X_mis | X_obs] (all-missing rows get mu_X)

        groups : list of (row_indices, mis_idx, Sigma_c) triples, one per
                 missingness pattern; Sigma_c is the conditional covariance
                 of the missing block (None for complete rows).

    With these fixed, the conditional linear predictor is intercept + F @ beta
    and the marginalisation variance is beta[mis]' Sigma_c beta[mis], so each
    NLL evaluation is O(n p) instead of O(patterns p^3); the optimizer no
    longer needs the MVN nuisance parameters at all.
    """
    n, p = X_rows.shape
    F = X_rows.copy()
    groups = []
    by_pattern = defaultdict(list)
    for i in range(n):
        by_pattern[tuple(np.where(np.isnan(X_rows[i]))[0])].append(i)

    for mis_key, rows in by_pattern.items():
        rows = np.array(rows)
        mis = np.array(mis_key, dtype=int)
        if mis.size == 0:
            groups.append((rows, mis, None))
            continue
        obs = np.setdiff1d(np.arange(p), mis)
        if obs.size == 0:
            F[np.ix_(rows, mis)] = mu_X[mis]
            groups.append((rows, mis, Sigma_X[np.ix_(mis, mis)]))
            continue
        S_oo = Sigma_X[np.ix_(obs, obs)]
        S_mo = Sigma_X[np.ix_(mis, obs)]
        try:
            K = np.linalg.solve(S_oo, S_mo.T).T
        except np.linalg.LinAlgError:
            K = S_mo @ np.linalg.pinv(S_oo)
        Sc = Sigma_X[np.ix_(mis, mis)] - K @ S_mo.T
        Sc = 0.5 * (Sc + Sc.T)
        F[np.ix_(rows, mis)] = (
            mu_X[mis] + (X_rows[np.ix_(rows, obs)] - mu_X[obs]) @ K.T
        )
        groups.append((rows, mis, Sc))

    return F, groups


# ============================================================
# Cholesky parameterization of covariance matrices
# ============================================================

def _diag_pos(i):
    """
    Return the position of diagonal element (i, i) in a row-major
    lower-triangular flattened vector (as produced by np.tril_indices).

    Row i of the lower triangle has i+1 elements, starting at global
    index i*(i+1)/2. The diagonal (i,i) is the last element in that row,
    so its global index is i*(i+1)/2 + i = i*(i+3)/2.

    Verified: i=0->0, i=1->2, i=2->5, i=3->9 (matches tril_indices).
    """
    return i * (i + 3) // 2


def psd_jitter(Sigma, floor=1e-4, base=1e-6):
    """Return Sigma nudged onto the positive-definite cone.

    A covariance estimated from heavily incomplete data is often only
    marginally positive definite, and at 90% missingness or p > n it is
    routinely indefinite by a small amount. Cholesky then fails, and the
    caller sees ``LinAlgError: Matrix is not positive definite`` from deep
    inside a factorisation, with nothing pointing at the cause.

    The jitter is chosen from the spectrum rather than fixed: if the smallest
    eigenvalue is already comfortable the nudge is negligible, and if it is
    negative the matrix is lifted just clear of the floor. A fixed constant
    cannot do this, being simultaneously too large for a well-conditioned
    matrix and too small for a badly conditioned one. MissLinear used a fixed
    1e-10 and was the only regressor that failed at 90% missingness and at
    p > n, while MissLASSORegressor already applied this rule and coped.

    Also guarantees a 2-D result. ``np.cov`` returns a 0-d array for a single
    column, which is what made MissLASSORegressor fail at p = 1 where its own
    classifier succeeded.

    Parameters
    ----------
    Sigma : array_like
        Symmetric covariance estimate. Scalars and 0-d arrays are promoted.
    floor : float
        Target lower bound on the smallest eigenvalue.
    base : float
        Jitter applied even when the matrix is already comfortable.

    Returns
    -------
    ndarray of shape (p, p), symmetric and positive definite.
    """
    S = np.atleast_2d(np.asarray(Sigma, dtype=np.float64))
    if S.shape[0] != S.shape[1]:
        S = S.reshape(1, 1)
    S = 0.5 * (S + S.T)                     # enforce exact symmetry first
    try:
        min_eig = float(np.linalg.eigvalsh(S).min())
    except np.linalg.LinAlgError:
        min_eig = -np.inf
    jitter = max(base, -min_eig + floor) if min_eig < floor else base
    return S + np.eye(S.shape[0]) * jitter


def constant_feature_mask(X):
    """Flag columns whose observed values never vary.

    A feature that takes one value carries no information about the response,
    so its coefficient is not identifiable: in the fitted joint covariance
    both its variance and its covariance with y are numerically zero, and the
    slope recovered as their ratio is whatever rounding error the optimiser
    happened to leave behind. On a single all-zero column that produced a
    coefficient of -324 in one row order and -1.68e6 in another, which is the
    same fit reported two different ways.

    The degeneracy cannot be detected from the covariance alone. A relative
    rank tolerance compares each variance against the largest one, so when
    every column is constant there is nothing left to compare against and the
    noise looks full-rank. Comparing against the variance of y instead would
    wrongly flag a genuine feature measured in small units. The data is the
    only unambiguous reference, so the test is made here, on the observed
    values, before any of that scale information is lost.

    Columns with fewer than two observed values are also flagged: a variance
    cannot be estimated from one point, and the same unidentifiability
    follows.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix, may contain NaN.

    Returns
    -------
    ndarray of bool, shape (p,)
        True where the column is constant across its observed entries.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, p = X.shape
    mask = np.zeros(p, dtype=bool)
    for j in range(p):
        col = X[:, j]
        obs = col[~np.isnan(col)]
        if obs.size < 2:
            mask[j] = True
        else:
            mask[j] = bool(obs.max() == obs.min())
    return mask


def canonical_row_order(Z):
    """Permutation putting rows into an order that does not depend on input order.

    Every accumulation in a FIML fit, the EM sufficient statistics, the
    pattern-grouped likelihood sums, the optimiser's own arithmetic, is a
    floating-point sum, and floating-point addition is not associative. Two
    orderings of the same rows therefore produce objective values that differ
    in the last bits, and on a flat or ill-conditioned surface the optimiser
    amplifies that into genuinely different stopping points: on a fourteen-row
    design the log-likelihood differed in the sixth significant figure and
    predictions by 0.186.

    Sorting the rows before any of that happens removes the cause rather than
    tightening a tolerance against the symptom. The fit sees the same sequence
    of numbers whatever order the caller supplied, so it is permutation-exact
    by construction. Nothing needs to be unsorted afterwards: a fit returns
    parameters, not per-row output, and prediction is already row-wise.

    Rows are ordered lexicographically by column, with a column's missingness
    ranking ahead of its value so that NaN is distinguishable from any number
    rather than being mapped onto one. Rows that tie on every key are
    identical and contribute identically, so their relative order cannot
    matter.

    Parameters
    ----------
    Z : ndarray of shape (n, k)
        Rows to order; may contain NaN.

    Returns
    -------
    ndarray of int, shape (n,)
        Indices that sort the rows.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    keys = []
    # np.lexsort treats the LAST key as most significant, so walk the columns
    # backwards to make column 0 the primary one.
    for j in range(Z.shape[1] - 1, -1, -1):
        col = Z[:, j]
        nan = np.isnan(col)
        keys.append(np.where(nan, 0.0, col))   # finer: the value
        keys.append(nan)                       # coarser: observed before missing
    return np.lexsort(keys)


def degenerate_feature_mask(X, rtol=1e-12):
    """Columns that cannot support an identified coefficient.

    What this is for, stated precisely, because it was doing two jobs and
    only one of them was its own. The numerical rescue is already handled
    elsewhere: :func:`psd_jitter` lifts a singular joint covariance onto the
    positive-definite cone, and a constant column then comes out with a
    coefficient of exactly 0.0 rather than raising ``LinAlgError``. What
    exclusion adds on top of that is *honest reporting*. A held-out column
    gets NaN for its standard error, z statistic and p value, which says the
    coefficient is not identifiable, rather than reporting it as exactly zero
    with a confidence interval around it. That distinction is the whole
    purpose of this mask, and it is a question about identifiability alone.

    So the union is of identifiability tests only:

    * :func:`constant_feature_mask` asks whether the column varies at all.
      It is the only test that can answer when a column has too few observed
      values to take a spread or a magnitude from.
    * ``sd <= 0`` catches a column that does vary, but by so little that its
      variance underflows to exactly zero: observed values of 0 and 3.07e-177
      are not constant, yet a model fitted over them has a zero-variance
      direction just the same, and MissLinear returned non-finite predictions
      from it.
    * The resolution test catches a spread that is the representation error
      of the column's own values, as for a column of 1e+06 plus noise of
      1e-10, where centring is catastrophic cancellation.

    **The relative-to-widest test was removed on 17 August 2026.** It asked
    whether a column's spread was negligible beside the *widest* spread in
    the design, which is not a question about identifiability at all: a
    column with a small spread next to a large one is perfectly identifiable.
    It was withdrawn from :func:`feature_scale` the day before for being
    unable to order its two cases correctly, and it failed here in the same
    way, classing a Gaussian column scaled by 1e-06 as unidentifiable. That
    left MissLinear scoring 0.7334 under the conformance ``extreme_scale``
    regime where MissRidgeRegressor and MissLASSORegressor, which share its
    model and differ only in penalty, both reached 0.846.

    Being reluctant to exclude is the right way to be wrong here, and the
    risk is asymmetric: a column wrongly kept is caught downstream by the
    jitter, which gives it a coefficient of zero, while a column wrongly
    dropped is silently unavailable to the model with nothing to catch it.
    Measured across all fourteen conformance regimes and both tasks, dropping
    the relative term changes exactly one cell, ``extreme_scale``, which is
    the one it was getting wrong.

    No absolute floor on the standard deviation is applied here, unlike the
    version of :func:`feature_scale` that once had one. Declining to divide
    by a small number is harmless; dropping a column from the model is not,
    and a dataset recorded entirely in small units would lose every feature
    it has.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix, may contain NaN.
    rtol : float
        Smallest spread, as a fraction of the column's own largest absolute
        value, that still counts as an identifiable spread.

    Returns
    -------
    ndarray of bool, shape (p,)
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    const = constant_feature_mask(X)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd), sd, 0.0)

    # The column's own magnitude sets the resolution its values are stored
    # at, the same measure :func:`feature_scale` uses, so the two functions
    # cannot disagree about which columns are real.
    magnitude = np.where(np.isfinite(X), np.abs(X), 0.0).max(axis=0) \
        if X.size else np.zeros(X.shape[1])

    return const | (sd <= 0.0) | (sd <= rtol * magnitude)


def feature_scale(X, rtol=1e-12):
    """Per-column standardisation divisor that will not amplify noise.

    Standardising divides each column by its standard deviation, so the
    divisor has to be a spread the column really has. Every estimator here
    once guarded that with the same absolute floor, ``sd >= 1e-8``, written
    out separately in eleven places. An absolute floor is unit-dependent and
    it let through the case that prompted the rewrite: a column whose
    observed values were 0 and 5.96e-08, beside features with a spread near
    300, which was divided up to order one, fitted, and divided back down to
    a coefficient of 7.75e+09 that moved by a hundred million between two row
    orderings of the same data.

    That was replaced by a floor relative to the *widest* column in the
    design, and this is the second rewrite, on 16 August 2026, because the
    relative floor measures the wrong thing. How small a column's spread is
    next to some other column's says nothing about whether the spread is
    real. Measured on the two cases it has to separate, the column that must
    be standardised sits **71 times further below the widest column** than the
    pathological one ever did, so no choice of threshold could have kept one
    and dropped the other. What it did instead was class an ordinary Gaussian
    column scaled by 1e-06 as degenerate and leave it unstandardised, after
    which MissMixedRegressor needed a coefficient of 1e+06 to compensate,
    drove sigma_sq to 1e+305 and reached an r-squared of -6.7e+06 on data
    where MissLinear managed 0.73.

    The test is therefore per column and scale-free: a spread counts when it
    is large compared with the resolution of that column's own values.

      * A Gaussian column scaled by 1e-06 has a spread of 0.41 relative to its
        own magnitude, exactly like any other Gaussian column, and is
        standardised. So is the 0-and-5.96e-08 column, at 0.42; its
        instability was a row-ordering failure and is now prevented directly
        by :func:`canonical_row_order`, which fixes the summation order so a
        permutation cannot change the answer at all.
      * A column of 1e+06 plus noise of 1e-10 has a relative spread of
        1.0e-16, below the float64 epsilon of 2.2e-16. Its "spread" is the
        representation error of its own values, centring it is catastrophic
        cancellation, and it is refused. This is the case an absolute floor
        cannot see either, since 1e-10 clears any fixed threshold.

    Those sit twelve orders of magnitude apart on this measure, so *rtol* is a
    gap rather than a line: anywhere from about 1e-14 to 1e-06 gives the same
    answers on all three.

    This is the complement to :func:`constant_feature_mask`, which asks
    whether a column varies at all and is the only one that can answer when a
    column has no observed values to take a magnitude from. Neither test
    subsumes the other.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix, may contain NaN.
    rtol : float
        Smallest spread, as a fraction of the column's own largest absolute
        value, that still counts as a spread worth dividing by.

    Returns
    -------
    ndarray of shape (p,)
        The divisor per column: the standard deviation where that is
        meaningful, and 1.0 where dividing would amplify noise.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.size == 0:
        return np.ones(X.shape[1], dtype=np.float64)
    # ddof=1 warns about degrees of freedom on a column with fewer than two
    # observed values, which is an ordinary thing for this package to be
    # handed. Such a column gets a NaN spread, is caught below and is not
    # standardised; a divisor function has no business warning about it.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd), sd, 0.0)

    # The column's own magnitude sets the resolution its values are stored
    # at. Missing entries contribute nothing rather than being skipped by
    # nanmax, which warns on an all-NaN column and returns NaN for it.
    magnitude = np.where(np.isfinite(X), np.abs(X), 0.0).max(axis=0)

    resolvable = sd > rtol * magnitude
    return np.where((sd > 0.0) & resolvable, sd, 1.0)


def pack_cholesky(Sigma):
    """
    Compute the Cholesky factor L of Sigma = L @ L.T and pack L into a
    1-D parameter vector suitable for unconstrained optimization.

    Diagonal entries are stored as log(L_ii) to enforce L_ii > 0 without
    bound constraints. Off-diagonal entries are stored directly.

    Parameters
    ----------
    Sigma : ndarray, shape (p, p), symmetric positive definite

    Returns
    -------
    vec : ndarray, shape (p*(p+1)//2,)
        Packed Cholesky vector; diagonals are log-transformed.
    L   : ndarray, shape (p, p)
        The Cholesky factor before log-transforming diagonals.
    """
    p = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    rows, cols = np.tril_indices(p)
    vec = L[rows, cols].copy().astype(np.float64)
    # Vectorised diagonal log-transform (replaces Python loop)
    diag_pos = np.array([i * (i + 3) // 2 for i in range(p)])
    vec[diag_pos] = np.log(np.diag(L))
    return vec, L


def unpack_cholesky(vec, p):
    """
    Reconstruct the lower-triangular Cholesky factor L from a packed vector.

    Diagonal entries are stored as logs and are exponentiated on recovery,
    guaranteeing positive diagonals. Returns L such that Sigma = L @ L.T
    is symmetric positive definite by construction.

    Parameters
    ----------
    vec : ndarray, shape (p*(p+1)//2,)
    p   : int, matrix dimension

    Returns
    -------
    L : ndarray, shape (p, p)
    """
    L = np.zeros((p, p), dtype=np.float64)
    rows, cols = np.tril_indices(p)
    L[rows, cols] = vec
    diag_pos = np.array([i * (i + 3) // 2 for i in range(p)])
    L[np.arange(p), np.arange(p)] = np.exp(vec[diag_pos])
    return L


# ============================================================
# Multivariate normal log-PDF
# ============================================================

def mvn_logpdf(x, mu, Sigma):
    """
    Evaluate the multivariate normal log-PDF at x via Cholesky decomposition.

    log p(x) = -k/2 log(2pi) - sum_i log(L_ii) - 0.5 ||L^{-1}(x-mu)||^2

    Works uniformly for k = 1, ..., p without special-casing scalars.

    Parameters
    ----------
    x     : ndarray, shape (k,)
    mu    : ndarray, shape (k,)
    Sigma : ndarray, shape (k, k)

    Returns
    -------
    float  (-inf if Sigma is not positive definite)
    """
    k = len(x)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf
    diff = x - mu
    v = np.linalg.solve(L, diff)                        # L v = diff
    log_det = 2.0 * np.sum(np.log(np.diag(L)))         # log|Sigma|
    maha_sq = float(np.dot(v, v))                       # (x-mu)^T Sigma^{-1} (x-mu)
    return -0.5 * (k * np.log(2.0 * np.pi) + log_det + maha_sq)


def mvn_logpdf_batch(X_batch, mu, Sigma):
    """
    Vectorized MVN log-PDF for multiple observations sharing the same (mu, Sigma).

    Performs a single Cholesky decomposition and triangular solve for all
    observations, replacing the per-row cost of mvn_logpdf in the NLL loop.

    Parameters
    ----------
    X_batch : ndarray, shape (n, k)
    mu      : ndarray, shape (k,)
    Sigma   : ndarray, shape (k, k)

    Returns
    -------
    log_probs : ndarray, shape (n,)  (-inf entries if Sigma is not PD)
    """
    k = Sigma.shape[0]
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return np.full(len(X_batch), -np.inf)
    diff = X_batch - mu                        # (n, k)
    V = np.linalg.solve(L, diff.T)            # (k, n)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    maha_sq = np.sum(V ** 2, axis=0)          # (n,)
    return -0.5 * (k * np.log(2.0 * np.pi) + log_det + maha_sq)


# ============================================================
# Conditional normal distribution
# ============================================================

def conditional_normal_params(mu, Sigma, obs_idx, mis_idx, x_obs):
    """
    Compute parameters of the conditional distribution Z_mis | Z_obs = x_obs
    for Z ~ N(mu, Sigma), using the partitioned normal formulae.

    Conditioning formulae:
        mu_c  = mu_m + Sigma_mo @ Sigma_oo^{-1} @ (x_obs - mu_o)
        Sig_c = Sigma_mm - Sigma_mo @ Sigma_oo^{-1} @ Sigma_om

    The linear system Sigma_oo @ K.T = Sigma_mo.T is solved via numpy's
    solver (internally Cholesky/LU) rather than computing the inverse
    explicitly.

    Parameters
    ----------
    mu      : ndarray, shape (p,)
    Sigma   : ndarray, shape (p, p)
    obs_idx : 1-D array-like of int, indices of observed variables
    mis_idx : 1-D array-like of int, indices of missing variables
    x_obs   : ndarray, shape (len(obs_idx),), observed values

    Returns
    -------
    mu_cond    : ndarray, shape (len(mis_idx),)
    Sigma_cond : ndarray, shape (len(mis_idx), len(mis_idx))
    """
    obs_idx = np.asarray(obs_idx)
    mis_idx = np.asarray(mis_idx)

    Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
    Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]   # shape: (|mis|, |obs|)
    Sigma_mm = Sigma[np.ix_(mis_idx, mis_idx)]

    # K = Sigma_mo @ Sigma_oo^{-1}  solved as  Sigma_oo K.T = Sigma_mo.T
    K = np.linalg.solve(Sigma_oo, Sigma_mo.T).T  # shape: (|mis|, |obs|)

    mu_cond = mu[mis_idx] + K @ (x_obs - mu[obs_idx])
    Sigma_cond = Sigma_mm - K @ Sigma_mo.T
    # Symmetrize to remove floating-point asymmetry introduced by K computation
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
    return mu_cond, Sigma_cond


# ============================================================
# 1-D Gauss-Hermite integration for logistic-normal expectation
# ============================================================

def _integrate_wide(a_arr, v):
    """E[sigma(a + t)], t ~ N(0, v), for variances Gauss-Hermite cannot hold.

    Gauss-Hermite fails here structurally rather than by a shortage of nodes.
    In its own variable the integrand is sigma(a + sqrt(2v) x), a step of width
    about 1/sqrt(2v); past v of roughly 50 no practical node count resolves it,
    and raising n_points from 20 to 640 still leaves an error above 1e-4 at
    v = 100. The worst error over a in [-6, 6] at the shipped 20 nodes reaches
    0.004 by v = 25 and 0.03 by v = 100, and v is the variance of the missing
    features' linear contribution, so it grows with both the number of missing
    features and the size of the coefficients: a fitted model with |beta| up to
    3.6 had a median v of 23 across its missingness patterns.

    Splitting the logistic into a step plus a remainder removes the difficulty:

        sigma(s) = H(s) + r(s),        r(s) = sigma(s) - H(s)

    with H the unit step at 0. Substituting s = a + t,

        E[sigma(a + t)] = P(S > 0) + INT r(s) N(s; a, v) ds,   S ~ N(a, v)
                        = Phi(a / sqrt(v)) + INT r(s) N(s; a, v) ds

    The first term is exact and carries all of the v dependence. The second has
    a support that does not widen with v at all, because |r(s)| < exp(-|s|), so
    ordinary Gauss-Legendre handles it. r jumps by -1 at the origin, so the
    correction is integrated as two panels meeting there; a single panel
    straddling the jump converges at 1/n instead of spectrally.

    Accurate to about 2e-13 for every v from 0.1 to 2000.
    """
    s, wr = _split_nodes()
    sd = np.sqrt(v)
    dens = np.exp(-0.5 * ((s[None, :] - a_arr[:, None]) / sd) ** 2) / (
        sd * np.sqrt(2.0 * np.pi))
    return ndtr(a_arr / sd) + dens @ wr


def _split_nodes():
    """Nodes for the correction integral, and weights with r(s) folded in.

    Two panels meeting at the origin, because r jumps by -1 there and a single
    panel straddling the jump converges at 1/n instead of spectrally. Cached
    for the process lifetime alongside the Gauss-Hermite nodes.
    """
    key = ('split', _SPLIT_NODES, _SPLIT_HALF_WIDTH)
    if key not in _gl_cache:
        x, w = np.polynomial.legendre.leggauss(_SPLIT_NODES)
        half = 0.5 * _SPLIT_HALF_WIDTH
        s = np.concatenate([half * (x - 1.0), half * (x + 1.0)])
        ws = np.concatenate([half * w, half * w])
        # r(s) = sigma(s) - H(s), written to avoid cancellation on either side
        r = np.where(s < 0.0, sigmoid(s), sigmoid(s) - 1.0)
        _gl_cache[key] = (s, ws * r)
    return _gl_cache[key]


def logistic_normal_with_grads(a, v_row, n_points=20):
    """E[sigma(a + t)] with t ~ N(0, v), and its derivatives, row by row.

    The value that :func:`integrate_logistic_normal` returns, plus dp/da and
    dp/dv, so that a fitting likelihood can use the same rule its predictions
    use. Every estimator that marginalises a logistic link over missing
    features needs exactly these three quantities: the chain rule from
    (a, v) out to the coefficients is the estimator's own business and differs
    between them, but the integral and its two derivatives do not.

    ``v_row`` is per row rather than scalar, because a fit evaluates every
    missingness pattern in one call and each pattern carries its own variance.
    Rows are split between the two rules by their own v, so a single call
    mixes them freely.

    For the wide branch, differentiating

        p = Phi(a / sqrt(v)) + INT r(s) N(s; a, v) ds

    under the integral sign gives

        dp/da = phi(a/sqrt(v)) / sqrt(v)
                + INT r(s) N(s; a, v) (s - a) / v ds
        dp/dv = -a phi(a/sqrt(v)) / (2 v^(3/2))
                + INT r(s) N(s; a, v) [ (s-a)^2 / (2v^2) - 1 / (2v) ] ds

    both of which reuse the density already formed for the value.

    Parameters
    ----------
    a        : ndarray, shape (n,)
    v_row    : ndarray, shape (n,), non-negative
    n_points : int, Gauss-Hermite nodes for the narrow branch

    Returns
    -------
    p, dp_da, dp_dv : ndarray, each shape (n,)
    """
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    v_row = np.atleast_1d(np.asarray(v_row, dtype=np.float64))
    if v_row.shape != a.shape:
        v_row = np.broadcast_to(v_row, a.shape)

    p = np.empty_like(a)
    dp_da = np.empty_like(a)
    dp_dv = np.zeros_like(a)
    wide = v_row > _SPLIT_ABOVE_V

    if not wide.all():
        idx = np.flatnonzero(~wide)
        if n_points not in _gh_cache:
            _gh_cache[n_points] = np.polynomial.hermite.hermgauss(n_points)
        t_k, w_k = _gh_cache[n_points]
        w_norm = w_k / np.sqrt(np.pi)

        c = np.sqrt(2.0 * v_row[idx])
        Z = a[idx][:, None] + c[:, None] * t_k[None, :]
        S = sigmoid(Z)
        Sp = S * (1.0 - S)
        p[idx] = S @ w_norm
        dp_da[idx] = Sp @ w_norm
        # c = sqrt(2v), so dc/dv = 1/c; at v = 0 the derivative is left at 0,
        # which is what the Gauss-Hermite form has always reported there.
        dp_dc = (Sp * t_k[None, :]) @ w_norm
        safe = c > 1e-12
        dp_dv[idx] = np.where(safe, dp_dc / np.where(safe, c, 1.0), 0.0)

    if wide.any():
        idx = np.flatnonzero(wide)
        s, wr = _split_nodes()
        v = v_row[idx]
        sd = np.sqrt(v)
        u = a[idx] / sd
        phi = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)

        diff = s[None, :] - a[idx][:, None]
        dens = np.exp(-0.5 * diff * diff / v[:, None]) / (
            sd[:, None] * np.sqrt(2.0 * np.pi))

        p[idx] = ndtr(u) + dens @ wr
        dp_da[idx] = phi / sd + (dens * (diff / v[:, None])) @ wr
        dp_dv[idx] = (-a[idx] * phi / (2.0 * v * sd)
                      + (dens * (diff * diff / (2.0 * v[:, None] ** 2)
                                 - 0.5 / v[:, None])) @ wr)

    return p, dp_da, dp_dv


def integrate_logistic_normal(a, v, n_points=20):
    """
    Compute E[sigma(a + t)] where t ~ N(0, v).

    Uses Gauss-Hermite quadrature for small v and a step-plus-remainder split
    above ``_SPLIT_ABOVE_V``; see :func:`_integrate_wide` for why the second
    branch exists.

    This function reduces ANY-dimensional marginalization over missing features
    to a single 1-D integral.  The key identity is:

        If beta_mis is a vector and x_mis ~ N(mu_c, Sigma_c), then
        the scalar s = beta_mis @ x_mis is also normally distributed:
            s ~ N(beta_mis @ mu_c,  beta_mis @ Sigma_c @ beta_mis)

        Therefore:
            E[sigma(beta_0 + beta_obs @ x_obs + beta_mis @ x_mis)]
            = E_t[sigma(a + t)],   t ~ N(0, v)

        where
            a = beta_0 + beta_obs @ x_obs + beta_mis @ mu_c
            v = beta_mis @ Sigma_c @ beta_mis

    Physicist's Gauss-Hermite: integral f(x) exp(-x^2) dx = sum_i w_i f(x_i)
    Substituting t = sqrt(2v) * s (so t ~ N(0,v)) gives:
        E_t[sigma(a+t)] = (1/sqrt(pi)) * sum_i w_i * sigma(a + sqrt(2v)*s_i)

    Parameters
    ----------
    a        : float or ndarray, linear predictor offset (scalar or 1-D array)
    v        : float >= 0, variance of the missing-feature linear contribution
    n_points : int, number of Gauss-Hermite nodes (default 20). Applies only
               to the small-variance branch, where 20 nodes are accurate to
               about 1e-11. Above _SPLIT_ABOVE_V the split rule is used and
               this argument is not consulted.

    Returns
    -------
    float or ndarray in (0, 1); matches the shape of a
    """
    # Treat 0-d numpy arrays (e.g. np.float64 scalars) as scalars throughout.
    _scalar = np.isscalar(a) or np.ndim(a) == 0
    if v <= 1e-12:
        # Deterministic: no missing-feature variance to integrate over
        return float(sigmoid(a)) if _scalar else sigmoid(np.asarray(a))
    if v > _SPLIT_ABOVE_V:
        result = _integrate_wide(np.atleast_1d(np.asarray(a, dtype=float)), v)
        result = np.clip(result, 1e-15, 1.0 - 1e-15)
        return float(result[0]) if _scalar else result
    if n_points not in _gh_cache:
        _gh_cache[n_points] = np.polynomial.hermite.hermgauss(n_points)
    nodes, weights = _gh_cache[n_points]
    t = np.sqrt(2.0 * v) * nodes                          # (Q,)
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))     # (n,)
    vals = sigmoid(a_arr[:, None] + t[None, :])            # (n, Q)
    result = vals @ weights / np.sqrt(np.pi)               # (n,)
    result = np.clip(result, 1e-15, 1.0 - 1e-15)
    return float(result[0]) if _scalar else result


def standard_errors_from_variance(Var):
    """Standard errors from a variance matrix, NaN where there is no variance.

    A delta-method or inverse-Hessian variance can come out negative: the
    Hessian was not positive definite, or the Jacobian was ill conditioned
    enough that the quadratic form lost its sign. That is a computation
    reporting its own failure.

    These were previously floored at zero, which turned "this could not be
    computed" into "this coefficient is known exactly": the reported standard
    error was 0.0000 and the confidence interval collapsed onto the point
    estimate, the most confident statement the table can make, arrived at
    because the arithmetic broke. Two identical predictor columns produced it
    in most draws, their individual coefficients being unidentified while
    their sum is not, and MissLinear then printed coefficients of -223.20 and
    +225.04 with intervals of zero width.

    NaN is the reading the rest of the library already uses for a coefficient
    that is not identified, and z_stats_ and pvalues_ already derive from
    ``np.where(se > 0, se, np.nan)``, so they were NaN in this case while the
    standard error beside them was not. Returning NaN here makes the three
    agree.

    Exact zeros take the same route as negatives, deliberately. A standard
    error of exactly zero asserts a coefficient known without error, which no
    finite sample supports, and it is the same value the downstream guard
    already rejects.

    Parameters
    ----------
    Var : ndarray of shape (k, k)
        Variance matrix; only its diagonal is read.

    Returns
    -------
    ndarray of shape (k,)
        Square roots of the positive diagonal entries, NaN elsewhere.
    """
    d = np.diag(np.asarray(Var, dtype=np.float64))
    return np.sqrt(np.where(d > 0.0, d, np.nan))


# ============================================================
# Numerical Hessian for standard error computation
# ============================================================

def numerical_hessian(f, x, eps=1e-4):
    """
    Compute the Hessian of a scalar function f at x via symmetric
    central finite differences.

    Mixed-partial formula (fourth-order accurate for smooth f):
        H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4 eps^2)

    Requires 2*n*(n+1) function evaluations for an n-dimensional x.

    Parameters
    ----------
    f   : callable, R^n -> R
    x   : ndarray, shape (n,)
    eps : float, finite-difference step (default 1e-4, balances truncation
          and cancellation error for double-precision arithmetic)

    Returns
    -------
    H : ndarray, shape (n, n), symmetric
    """
    n = len(x)
    H  = np.zeros((n, n))
    ei = np.zeros(n)
    ej = np.zeros(n)
    for i in range(n):
        ei[i] = eps
        for j in range(i, n):
            ej[j] = eps
            H[i, j] = (
                f(x + ei + ej) - f(x + ei - ej)
                - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * eps ** 2)
            H[j, i] = H[i, j]
            ej[j] = 0.0
        ei[i] = 0.0
    return H
