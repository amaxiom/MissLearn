"""
MissLearn._copula
-----------------
Marginal Gaussian copula transform for relaxing the joint normality assumption.

By Sklar's theorem, any joint distribution decomposes into:
    - marginal CDFs for each variable (estimated non-parametrically here)
    - a dependence structure (Gaussian copula: normal correlation matrix)

Transforming each variable to the normal scale via its empirical CDF
makes the Gaussian copula model applicable to any continuous distribution,
while leaving the FIML optimization and quadrature logic entirely unchanged.

Usage:
    copula=True in MissLinear or MissLogistic applies this transform
    automatically before fitting and inverts it on output.

Reference:
    Liu, H. et al. (2012). High-dimensional semiparametric Gaussian copula
    graphical models. Annals of Statistics, 40(4), 2293-2326.
"""

import numpy as np
from scipy.stats import norm   # skew/kurtosis computed locally, see _skew_kurtosis


#: A column with fewer distinct observed values than this is treated as
#: discrete and passed through untouched.
#:
#: Sklar's theorem gives a unique copula only when the marginals are
#: continuous. For a discrete marginal the empirical CDF is a step function,
#: the copula is not identified, and a rank-normal map is a relabelling of the
#: categories rather than a route to normality. Indicator columns also trigger
#: the automatic heuristic by construction, because an indicator with
#: prevalence outside roughly [0.25, 0.75] has skewness above 1 no matter what
#: it encodes, so without this rule one-hot dummies both invited the transform
#: and absorbed it.
#:
#: Three is the smallest threshold that excludes constant and binary columns,
#: the two cases where transforming has no defensible reading.
MIN_DISTINCT_FOR_COPULA = 3


def _is_discrete(obs):
    """True if the observed values are too few and too repetitive to transform."""
    return np.unique(obs).size < MIN_DISTINCT_FOR_COPULA


class RankNormalTransformer:
    """
    Per-column empirical CDF transform to the standard normal scale.

    For each feature j with n_j observed values, the transform is:
        u_j(x) = (rank(x among obs_j) - 0.375) / (n_j + 0.25)   [Blom 1958]
        z_j(x) = Phi^{-1}(u_j(x))

    Blom's plotting-position formula avoids Phi^{-1}(0) = -inf and
    Phi^{-1}(1) = +inf at the boundary ranks.

    Tied values all receive the mean normal score of the block they form,
    so the transform is a well-defined function of x even when a column is
    discrete. This also keeps the interpolation table strictly increasing,
    which np.interp requires in both directions.

    NaN values are preserved through both transform and inverse_transform.
    Values outside the training range are clipped to the boundary quantile
    (the most extreme observed normal score).

    On a column with few distinct values the transform is a monotone
    relabelling and cannot add information; see needs_copula, which decides
    whether transforming is worthwhile at all.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_features_ : int
    """

    def fit(self, X):
        """
        Estimate empirical CDFs from observed (non-NaN) values.

        Parameters
        ----------
        X : ndarray, shape (n, p), may contain NaN

        Returns
        -------
        self
        """
        _, p = X.shape
        self.n_features_ = p
        self._sorted_obs = []
        self._normal_scores = []
        self._passthrough_ = np.zeros(p, dtype=bool)

        for j in range(p):
            obs = X[:, j][~np.isnan(X[:, j])]
            if _is_discrete(obs):
                # Left on its own scale; see MIN_DISTINCT_FOR_COPULA. A column
                # with nothing left to estimate from arrives here too, with
                # zero or one observed value, and takes the same route.
                #
                # It used to raise instead. That made the copula the only part
                # of a fit that could refuse data the rest of the library
                # accepts: the same matrix fitted without complaint under
                # copula=False, and degenerate columns are already dropped
                # downstream. With copula='auto' as the default, any fold
                # sparse enough to empty a column killed the fit, in a library
                # whose subject is missing data.
                self._passthrough_[j] = True
                self._sorted_obs.append(None)
                self._normal_scores.append(None)
                continue

            obs_sorted = np.sort(obs)
            n_obs = len(obs_sorted)
            u = (np.arange(1, n_obs + 1) - 0.375) / (n_obs + 0.25)
            scores = norm.ppf(u)

            # Repeated values must all receive the same normal score, and
            # np.interp requires a strictly increasing first array. Collapse
            # each block of tied values to the mean of the scores it spans
            # (the mid-rank, or van der Waerden, convention).
            #
            # Passing the raw sorted values through instead leaves duplicate
            # entries in the interpolation table, and np.interp then resolves
            # every member of a tie block to the block's *largest* score. The
            # starkest case was a binary column, which no longer reaches this
            # branch: 0 mapped to +1.18 and 1 to +3.12, giving a column of mean
            # 1.41 and standard deviation 0.62 where the transform is defined
            # to deliver N(0, 1). Any repeated value is distorted the same way,
            # in proportion to how much of the column its block spans, and
            # which score a block landed on moved with the block boundaries, so
            # results shifted between cross-validation folds as well.
            uniq, inverse = np.unique(obs_sorted, return_inverse=True)
            counts = np.bincount(inverse)
            block_mean = np.bincount(inverse, weights=scores) / counts

            self._sorted_obs.append(uniq)
            self._normal_scores.append(block_mean)

        return self

    def transform(self, X):
        """
        Map X to the standard normal scale column-wise.  NaN -> NaN.

        Uses linear interpolation within the training range; clips to
        the boundary normal score outside it.

        Parameters
        ----------
        X : ndarray, shape (n, p)

        Returns
        -------
        Z : ndarray, shape (n, p), approximately N(0, 1) column-wise
        """
        X = np.asarray(X, dtype=np.float64)
        Z = np.full_like(X, np.nan)
        for j in range(self.n_features_):
            if self._passthrough_[j]:
                Z[:, j] = X[:, j]
                continue
            obs_mask = ~np.isnan(X[:, j])
            if obs_mask.any():
                Z[obs_mask, j] = np.interp(
                    X[obs_mask, j],
                    self._sorted_obs[j],
                    self._normal_scores[j],
                )
        return Z

    def inverse_transform(self, Z):
        """
        Map Z from the normal scale back to the original scale.  NaN -> NaN.

        Parameters
        ----------
        Z : ndarray, shape (n, p)

        Returns
        -------
        X : ndarray, shape (n, p)
        """
        Z = np.asarray(Z, dtype=np.float64)
        X = np.full_like(Z, np.nan)
        for j in range(self.n_features_):
            if self._passthrough_[j]:
                X[:, j] = Z[:, j]
                continue
            obs_mask = ~np.isnan(Z[:, j])
            if obs_mask.any():
                X[obs_mask, j] = np.interp(
                    Z[obs_mask, j],
                    self._normal_scores[j],
                    self._sorted_obs[j],
                )
        return X

    def inverse_transform_1d(self, z_vals, col):
        """
        Inverse-transform a 1-D array for a single column.

        Parameters
        ----------
        z_vals : ndarray, shape (n,)
        col    : int, column index

        Returns
        -------
        x_vals : ndarray, shape (n,)
        """
        if self._passthrough_[col]:
            return np.asarray(z_vals, dtype=np.float64)
        return np.interp(
            z_vals,
            self._normal_scores[col],
            self._sorted_obs[col],
        )


# ============================================================
# Automatic copula selection heuristic
# ============================================================

def needs_copula(X, y=None):
    """
    Return True if any column shows substantial departure from normality.

    Checks Fisher's skewness and excess kurtosis on each column's observed
    (non-NaN) values.  Thresholds:
        ``|skewness|``      > 1.0   -- moderate asymmetry (exponential has ~2)
        ``|excess kurtosis|`` > 2.0 -- substantial tail departure (chi-sq(4) has 3)

    Columns with fewer than 8 observed values are skipped.

    Parameters
    ----------
    X : ndarray, shape (n, p), may contain NaN
    y : ndarray, shape (n,) or None -- pass y only from an estimator that
        also transforms y. The regressors do; the classifiers do not, and
        neither does MissNeighborsRegressor, which transforms features to
        calibrate distances and averages y on its original scale.

    Returns
    -------
    bool
    """
    cols = [X[:, j] for j in range(X.shape[1])]
    if y is not None:
        cols.append(y)

    for col in cols:
        obs = col[~np.isnan(col)]
        if len(obs) < 8 or _is_discrete(obs):
            continue
        g1, g2 = _skew_kurtosis(obs)
        if abs(g1) > 1.0 or abs(g2) > 2.0:
            return True
    return False


def _skew_kurtosis(obs):
    """Fisher skewness and excess kurtosis, computed here rather than imported.

    These are two moment ratios, and scipy's versions of them route through
    its array-API compatibility layer, which under scipy 1.17 against numpy
    1.26 reaches ``numpy.dtypes.VoidDType`` through a lazy import that does
    not always resolve; it surfaced as an AttributeError from deep inside
    scipy during a coverage run. That path used to be reached only when
    somebody asked for copula='auto' explicitly. It is now the default, so
    every fit of every estimator goes through it, and a third-party
    incompatibility in the hot path of every fit is not worth a dependency
    for eight lines of arithmetic.

    Matches ``scipy.stats.skew(obs)`` and
    ``scipy.stats.kurtosis(obs, fisher=True)`` at their defaults (biased
    moment estimators).
    """
    obs = np.asarray(obs, dtype=np.float64)
    d = obs - obs.mean()

    # Both statistics are dimensionless, so the arithmetic is made
    # dimensionless before it starts. Computing them from raw moments is not
    # scale-free even though the answer is: for values around 1e120 the cube
    # overflows while the square is still finite, giving inf/inf, and below
    # about 1e-150 both underflow to zero, giving 0/0. Either way the result
    # is NaN, and needs_copula reads abs(NaN) > 1.0 as False, so a column
    # would silently stop triggering the transform when its units changed.
    # Above 1e160 the square overflows too and the constant-column guard
    # below claims it, which is the same silent answer by a different route.
    span = np.max(np.abs(d)) if d.size else 0.0
    if not np.isfinite(span) or span <= 0.0:
        return 0.0, 0.0          # constant column: no asymmetry, no tails
    d = d / span                 # now within [-1, 1] whatever the units were

    m2 = np.mean(d ** 2)
    if not np.isfinite(m2) or m2 <= 0.0:
        return 0.0, 0.0
    z = d / np.sqrt(m2)          # standardised, so no power of it can escape
    g1 = float(np.mean(z ** 3))
    g2 = float(np.mean(z ** 4) - 3.0)
    if not (np.isfinite(g1) and np.isfinite(g2)):
        return 0.0, 0.0
    return g1, g2
