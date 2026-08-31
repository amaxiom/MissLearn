"""
_sensitivity.py  --  MNAR sensitivity analysis via delta-adjustment.

MissSensitivity
    Tests how robust model conclusions are to departures from the Missing At
    Random (MAR) assumption underlying FIML.

    Method: delta-adjustment (shift method).
    ----------------------------------------
    For each delta in a grid, the imputed values of missing Y are shifted by
    delta (in units of sigma_Y by default).  The model is then refitted on
    m multiply-imputed, delta-shifted datasets and estimates are pooled via
    Rubin's rules.  This traces a sensitivity curve: how do estimates, signs,
    and significance change as the MNAR departure grows?

    Interpretation of delta:
        delta = 0   : MAR assumption (same as the original FIML model)
        delta > 0   : missing Y values are systematically HIGHER than MAR
                      would predict (e.g. sicker patients tend to drop out
                      and also have worse outcomes)

        delta < 0   : missing Y values are systematically LOWER than MAR
                      would predict

    Tipping-point analysis:
        The smallest ``|delta|`` at which a specified conclusion changes.  For a
        regression coefficient: the delta at which the estimate crosses zero,
        or at which the confidence interval first excludes/includes zero.
        Reported in units of sigma_Y for interpretability.

    Philosophy note:
        MNAR cannot be detected from observed data alone.  This analysis does
        not test whether MNAR holds; it quantifies the *fragility* of
        conclusions: large tipping points indicate robust conclusions, small
        ones indicate conclusions that are sensitive to MNAR departures.

Reference:
    Van Buuren (2018). Flexible Imputation of Missing Data, Ch. 8.
    Carpenter & Kenward (2013). Multiple Imputation and its Application, Ch. 5.

Usage
-----
    from MissLearn import MissRidgeRegressor, MissSensitivity
    import numpy as np

    model = MissRidgeRegressor(alpha=1.0, compute_se=False)
    sens  = MissSensitivity(model, delta_range=(-3, 3), n_delta=25, m=10)
    sens.fit(X, y)

    sens.summary()
    tp = sens.tipping_point(coef_idx=0)
    print(f"Coefficient 0 tips at delta={tp:.2f} sigma_Y")
"""

from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Tuple

import numpy as np

from ._imputer import MissImputer


# ---------------------------------------------------------------------------
# MissSensitivity
# ---------------------------------------------------------------------------

class MissSensitivity:
    """
    MNAR sensitivity analysis via delta-adjustment (shift method).

    Parameters
    ----------
    estimator : fitted or unfitted MissLearn estimator
        Must implement fit(X, y) and expose ``coef_`` after fitting.
        MissRidgeRegressor and MissLinear are the primary intended uses.
        Any sklearn-compatible regressor also works.
    delta_range : tuple of (float, float), default (-3, 3)
        Range of delta values to sweep, in units of sigma_Y
        (if standardise=True) or raw outcome units (if standardise=False).
    n_delta : int, default 25
        Number of evenly-spaced delta values across delta_range.
    m : int, default 10
        Number of imputed datasets per delta value.
        Higher m reduces Monte Carlo noise in the pooled estimates.
    standardise : bool, default True
        If True, delta is measured in units of sigma_Y (standard deviation
        of the observed Y values).  Recommended: keeps delta interpretable
        across datasets with different outcome scales.
    alpha : float, default 0.05
        Significance level for the tipping-point CI analysis.
    imputer_reg : float, default 1e-6
        Regularisation passed to the internal MissImputer.
    random_state : int or None, default None

    Attributes
    ----------
    delta_grid_     : ndarray (n_delta,) -- raw delta values used
    delta_std_grid_ : ndarray (n_delta,) -- delta in sigma_Y units
    coef_curves_    : ndarray (n_delta, p) -- pooled coefficient at each delta
    coef_se_curves_ : ndarray (n_delta, p) -- Rubin SE at each delta (or NaN)
    intercept_curve_: ndarray (n_delta,)
    baseline_idx_   : int -- index of delta closest to 0 in ``delta_grid_``
    sigma_y_        : float -- std of observed Y (for standardisation)
    feature_names_  : list of str
    n_miss_y_       : int -- number of missing Y observations
    """

    def __init__(
        self,
        estimator,
        delta_range: Tuple[float, float] = (-3.0, 3.0),
        n_delta: int     = 25,
        m: int           = 10,
        standardise: bool = True,
        alpha: float     = 0.05,
        imputer_reg: float = 1e-6,
        random_state      = None,
    ):
        self.estimator    = estimator
        self.delta_range  = delta_range
        self.n_delta      = n_delta
        self.m            = m
        self.standardise  = standardise
        self.alpha        = alpha
        self.imputer_reg  = imputer_reg
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            **fit_kwargs) -> 'MissSensitivity':
        """
        Run the sensitivity analysis over the delta grid.

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        y : ndarray of shape (n,), may contain NaN
        feature_names : list of str, optional
            Coefficient labels for the summary table.  Defaults to
            X0, X1, ...
        **fit_kwargs : forwarded to each estimator.fit() call

        Raises
        ------
        AttributeError
            If the estimator exposes no ``coef_`` after fitting, since there is
            then no coefficient path to report.
        """
        X   = np.asarray(X, dtype=float)
        y   = np.asarray(y, dtype=float)
        n, p = X.shape

        # Observed y statistics for standardisation
        y_obs        = y[~np.isnan(y)]
        self.sigma_y_  = float(np.std(y_obs, ddof=1)) if len(y_obs) > 1 else 1.0
        self.mean_y_   = float(np.mean(y_obs))
        self.n_miss_y_ = int(np.sum(np.isnan(y)))

        if self.n_miss_y_ == 0:
            warnings.warn(
                "MissSensitivity: y has no missing values.  "
                "Sensitivity analysis is trivially flat (all deltas give the same fit).  "
                "Results are returned for completeness.",
                UserWarning, stacklevel=2,
            )

        # Delta grid
        d_lo, d_hi = self.delta_range
        self.delta_std_grid_ = np.linspace(d_lo, d_hi, self.n_delta)
        if self.standardise:
            self.delta_grid_ = self.delta_std_grid_ * self.sigma_y_
        else:
            self.delta_grid_     = self.delta_std_grid_
            self.delta_std_grid_ = self.delta_grid_ / max(self.sigma_y_, 1e-10)

        self.baseline_idx_ = int(np.argmin(np.abs(self.delta_grid_)))

        # Fit MissImputer jointly on (X, y): draws for missing y must
        # condition on X (and missing X on observed X and y) so that the
        # delta=0 baseline reproduces the MAR fit before delta shifts.
        imputer = MissImputer(
            m=self.m,
            reg=self.imputer_reg,
            random_state=self.random_state,
            include_y=True,
        )
        imputer.fit(X, y)

        # Miss-y mask
        miss_y_mask = np.isnan(y)

        # Generate the imputations once (shared across all deltas); each is
        # (n, p+1) with the jointly imputed y in the last column.
        # Use conditional mean imputation for a deterministic NaN-free baseline.
        Z_mean_imp     = imputer.transform_mean(X, y)
        X_mean_imp     = Z_mean_imp[:, :p]
        y_mean_imp     = Z_mean_imp[:, p]
        Z_imputed_list = imputer.transform(X, y)   # list of m stochastic draws

        # Fit baseline model once (at delta=0) to get feature count and names.
        # Use mean-imputed X so sklearn estimators (which reject NaN) also work.
        _base_est = copy.deepcopy(self.estimator)
        y_base    = y.copy()
        if self.n_miss_y_ > 0:
            y_base[miss_y_mask] = y_mean_imp[miss_y_mask]  # conditional means
        try:
            _base_est.fit(X_mean_imp, y_base, **fit_kwargs)
        except Exception:
            _base_est.fit(X_mean_imp, y_base)

        self._base_est = _base_est

        # Delta-adjustment reports how each *coefficient* moves as the MNAR
        # shift grows, so an estimator without coef_ has nothing to report.
        # Falling back to zeros here would print a table of 0.0000 with a
        # STABLE verdict against every row, which reads as "the conclusions
        # are robust" when in fact nothing was computed.
        if not hasattr(_base_est, 'coef_'):
            raise AttributeError(
                f"{type(_base_est).__name__} does not expose coef_ after "
                f"fitting, so a coefficient sensitivity curve cannot be "
                f"computed. MissSensitivity requires a model with linear "
                f"coefficients: MissLinear, MissRidgeRegressor, "
                f"MissLASSORegressor, MissLogistic, MissRidgeClassifier or "
                f"MissLASSOClassifier. Generative, kernel and "
                f"distance-based families (MissBayes, MissSupport, "
                f"MissNeighbors, MissGaussian) have no coefficient vector; "
                f"run the sensitivity analysis on a linear model fitted to "
                f"the same data instead."
            )

        p_coef = len(np.atleast_1d(_base_est.coef_))
        if feature_names is not None:
            if len(feature_names) != p_coef:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries but the "
                    f"model has {p_coef} coefficients.")
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"X{j}" for j in range(p_coef)]

        # Pre-allocate result arrays
        coef_curves       = np.full((self.n_delta, p_coef), np.nan)
        coef_se_curves    = np.full((self.n_delta, p_coef), np.nan)
        intercept_curve   = np.full(self.n_delta, np.nan)

        # --- sweep delta grid ---
        for di, delta in enumerate(self.delta_grid_):
            estimates_list  = []
            variances_list  = []
            intercepts_list = []

            for k in range(self.m):
                Z_imp = Z_imputed_list[k % len(Z_imputed_list)]  # reuse if m > len
                X_imp = Z_imp[:, :p]

                # Build y for this imputed dataset: conditional draws for
                # missing entries, shifted by delta (the MNAR departure)
                y_imp = y.copy()
                if self.n_miss_y_ > 0:
                    y_imp[miss_y_mask] = Z_imp[miss_y_mask, p] + delta

                est = copy.deepcopy(self.estimator)
                try:
                    est.fit(X_imp, y_imp, **fit_kwargs)
                except TypeError:
                    try:
                        est.fit(X_imp, y_imp)
                    except Exception as exc:
                        warnings.warn(
                            f"MissSensitivity: fit failed at delta={delta:.3f}, "
                            f"imputation {k}: {exc}",
                            UserWarning, stacklevel=2,
                        )
                        continue
                except Exception as exc:
                    warnings.warn(
                        f"MissSensitivity: fit failed at delta={delta:.3f}, "
                        f"imputation {k}: {exc}",
                        UserWarning, stacklevel=2,
                    )
                    continue

                # The baseline fit already established that coef_ exists, so
                # its absence here means this particular fit failed silently;
                # skip the draw rather than contributing zeros to the pool.
                if not hasattr(est, 'coef_'):
                    warnings.warn(
                        f"MissSensitivity: no coef_ after fit at "
                        f"delta={delta:.3f}, imputation {k}; draw skipped.",
                        UserWarning, stacklevel=2,
                    )
                    continue
                coef = np.atleast_1d(np.asarray(est.coef_, dtype=float))
                estimates_list.append(coef)
                intercepts_list.append(float(getattr(est, 'intercept_', 0.0)))

                se = getattr(est, 'se_', None)
                if se is not None:
                    se_arr = np.atleast_1d(np.asarray(se, dtype=float))
                    # se_ in MissLearn includes intercept at index 0
                    if len(se_arr) == p_coef + 1:
                        se_arr = se_arr[1:]   # drop intercept SE
                    if len(se_arr) == p_coef:
                        variances_list.append(se_arr ** 2)

            if estimates_list:
                est_arr = np.array(estimates_list)       # (k_ok, p_coef)
                coef_curves[di]     = est_arr.mean(axis=0)
                coef_se_curves[di]  = est_arr.std(axis=0, ddof=1) if len(est_arr) > 1 \
                    else np.zeros(p_coef)

                # Rubin total variance if within-imputation SEs available
                if variances_list and len(variances_list) == len(estimates_list):
                    W = np.array(variances_list).mean(axis=0)
                    B = np.var(est_arr, axis=0, ddof=1) if len(est_arr) > 1 \
                        else np.zeros(p_coef)
                    m_eff = len(estimates_list)
                    T = W + (1.0 + 1.0 / m_eff) * B
                    coef_se_curves[di] = np.sqrt(np.maximum(T, 0.0))

                intercept_curve[di] = np.mean(intercepts_list)

        self.coef_curves_     = coef_curves
        self.coef_se_curves_  = coef_se_curves
        self.intercept_curve_ = intercept_curve
        return self

    # ------------------------------------------------------------------
    # Tipping-point analysis
    # ------------------------------------------------------------------

    def tipping_point(
        self,
        coef_idx: int  = 0,
        threshold: float = 0.0,
        method: str    = 'sign',
    ) -> Optional[float]:
        """
        Find the smallest ``|delta|`` (in sigma_Y units) at which a conclusion
        about coefficient *coef_idx* changes.

        Parameters
        ----------
        coef_idx : int
            Index into ``coef_curves_`` (i.e., which coefficient to analyse).
        threshold : float, default 0.0
            The value the coefficient must cross.  Default 0 tests for a
            sign change.
        method : str, default 'sign'
            Which criterion defines the tipping point::

                'sign'  -- where coef - threshold changes sign
                'ci'    -- where the (1-alpha) CI first excludes or includes
                           threshold (requires coef_se_curves_)

        Returns
        -------
        float or None
            Tipping-point delta in sigma_Y units, or None if the conclusion
            does not change across the entire delta_range.
        """
        # Reject an unrecognised method rather than falling through to 'ci'.
        # The dispatch below used to be sign-or-everything-else, so 'Sign'
        # with a capital, or any typo, quietly ran the other criterion and
        # returned a number that looked like an answer.
        if method not in ('sign', 'ci'):
            raise ValueError(
                "method must be 'sign' or 'ci', got %r. 'sign' tips where "
                "the coefficient crosses the threshold; 'ci' tips where its "
                "confidence interval starts or stops excluding it."
                % (method,))

        n_coef = self.coef_curves_.shape[1]
        if not -n_coef <= coef_idx < n_coef:
            raise IndexError(
                "coef_idx %d is out of range; the fitted model has %d "
                "coefficient(s), so valid indices are 0 to %d."
                % (coef_idx, n_coef, n_coef - 1))

        coef = self.coef_curves_[:, coef_idx] - threshold
        base = coef[self.baseline_idx_]

        # A delta whose fits all failed holds NaN, by design, and NaN must not
        # be read as a crossing. It would be: np.sign(nan) != np.sign(base) is
        # True, and in the confidence-interval branch below the negation turns
        # a NaN comparison into True the same way. One failed fit at delta 0.5
        # turned a coefficient path running 1.96 to 1.99, nowhere near zero,
        # into "tips at delta=+0.50, mildly sensitive to MNAR". The whole
        # point of this tool is to say how far a conclusion can be pushed
        # before it breaks, so inventing a breaking point is the worst
        # available failure.
        finite = np.isfinite(coef)

        if not np.isfinite(base):
            raise ValueError(
                "the baseline fit at delta=0 produced no coefficient, so "
                "there is nothing to measure departures from. Every tipping "
                "point is expressed relative to it. Check the warnings from "
                "fit(): the baseline draws failed."
            )

        if method == 'sign':
            sign_changes = np.where(
                finite & (np.sign(coef) != np.sign(base)))[0]
        else:  # 'ci'
            se = self.coef_se_curves_[:, coef_idx]
            # Without standard errors every comparison below is False, the
            # negation makes every delta look like a crossing, and the
            # function returns the delta nearest zero: a report of maximal
            # fragility produced by having computed nothing. An estimator
            # fitted with compute_se=False lands here, and MissLASSORegressor
            # does so by default. Refusing is the only honest answer.
            if not np.any(np.isfinite(se)):
                raise ValueError(
                    "method='ci' needs standard errors, and none are "
                    "available: coef_se_curves_ is entirely NaN. Refit the "
                    "sensitivity analysis with an estimator that computes "
                    "them (compute_se=True), or use method='sign', which "
                    "needs only the coefficient path."
                )
            z  = 1.96  # approximate for large df
            ci_lo = coef - z * se
            ci_hi = coef + z * se
            # At baseline, check whether 0 is inside or outside CI
            # Deltas with no standard error are excluded here for the same
            # reason as above. The all-NaN case is refused outright, but a
            # partly-NaN column reached the negated branch below and every
            # missing delta came back as a crossing.
            finite = finite & np.isfinite(se)
            if ci_lo[self.baseline_idx_] > 0 or ci_hi[self.baseline_idx_] < 0:
                # Baseline: significant.  Tip = first delta where CI includes 0
                sign_changes = np.where(
                    finite & (ci_lo <= 0) & (ci_hi >= 0))[0]
            else:
                # Baseline: not significant.  Tip = first delta where CI excludes 0
                sign_changes = np.where(
                    finite & ~((ci_lo <= 0) & (ci_hi >= 0)))[0]

        if len(sign_changes) == 0:
            return None

        # Find the crossing closest to delta=0
        deltas_at_change = self.delta_std_grid_[sign_changes]
        return float(deltas_at_change[np.argmin(np.abs(deltas_at_change))])

    # ------------------------------------------------------------------
    # Robustness verdict
    # ------------------------------------------------------------------

    def verdict(self, coef_idx: int = 0, threshold: float = 0.0) -> str:
        """
        Return a plain-language robustness verdict for coefficient *coef_idx*.

        Uses conventional thresholds:
            ``|tip|`` >= 1.0 sigma_Y : robust to moderate MNAR
            ``|tip|`` >= 0.5 sigma_Y : mildly sensitive
            ``|tip|``  < 0.5 sigma_Y : sensitive (fragile conclusion)
            None                 : conclusion stable across entire delta_range
        """
        tp = self.tipping_point(coef_idx=coef_idx, threshold=threshold)
        if tp is None:
            return (f"[ROBUST]  Conclusion for X{coef_idx} is stable across the "
                    f"entire delta range [{self.delta_range[0]:.1f}, "
                    f"{self.delta_range[1]:.1f}] sigma_Y.")
        atp = abs(tp)
        if atp >= 1.0:
            label = "ROBUST"
            desc  = "robust to moderate MNAR departures"
        elif atp >= 0.5:
            label = "MILD"
            desc  = "mildly sensitive to MNAR"
        else:
            label = "SENSITIVE"
            desc  = "sensitive -- conclusion fragile under small MNAR departures"
        return (f"[{label}]  X{coef_idx} tips at delta={tp:+.2f} sigma_Y: {desc}.")

    # ------------------------------------------------------------------
    # Convenience: sensitivity table
    # ------------------------------------------------------------------

    def sensitivity_table(self) -> np.ndarray:
        """
        Return a structured array with columns [delta_std, coef_j, se_j]
        for each coefficient.

        Returns
        -------
        ndarray of shape (n_delta, 1 + 2*p_coef)
            Columns: [delta_std, coef_0, se_0, coef_1, se_1, ...]
        """
        p = self.coef_curves_.shape[1]
        # Interleave coef / se columns in one allocation (avoids repeated column_stack copies)
        coef_se = np.empty((self.n_delta, 2 * p))
        coef_se[:, 0::2] = self.coef_curves_
        coef_se[:, 1::2] = self.coef_se_curves_
        return np.column_stack([self.delta_std_grid_, coef_se])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        p   = self.coef_curves_.shape[1]
        bas = self.baseline_idx_

        # Size the label column to the longest name so real feature names
        # (channels, assays, analytes) do not push the numeric columns out
        # of alignment with their headers.
        labels = [self.feature_names_[j] if j < len(self.feature_names_)
                  else f"X{j}" for j in range(p)]
        NW = max(6, min(24, max((len(s) for s in labels), default=6)))
        W  = max(70, NW + 64)

        print()
        print('=' * W)
        print(f"{'MissSensitivity  --  MNAR Delta-Adjustment Analysis':^{W}}")
        print('=' * W)
        print(f"  Method        : delta-adjustment (shift in imputed missing Y)")
        print(f"  Delta range   : [{self.delta_range[0]:.1f}, {self.delta_range[1]:.1f}] sigma_Y")
        print(f"  Grid points   : {self.n_delta}")
        print(f"  Imputations   : {self.m} per delta")
        print(f"  sigma_Y       : {self.sigma_y_:.4f}")
        print(f"  Missing Y     : {self.n_miss_y_} obs")
        print()

        print(f"  {'Coef':<{NW}}  {'Baseline':>10}  {'at delta=-1':>12}  "
              f"{'at delta=+1':>12}  {'Tipping pt':>12}  {'Verdict':>10}")
        print(f"  {'-'*NW}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

        # find grid indices closest to -1 and +1 sigma_Y
        idx_m1 = int(np.argmin(np.abs(self.delta_std_grid_ + 1.0)))
        idx_p1 = int(np.argmin(np.abs(self.delta_std_grid_ - 1.0)))

        for j in range(p):
            base_val = self.coef_curves_[bas, j]
            m1_val   = self.coef_curves_[idx_m1, j]
            p1_val   = self.coef_curves_[idx_p1, j]
            tp       = self.tipping_point(coef_idx=j)
            tp_str   = f"{tp:+.2f}" if tp is not None else "stable"
            atp      = abs(tp) if tp is not None else 999
            verd     = "ROBUST" if atp >= 1.0 else ("MILD" if atp >= 0.5 else "SENSITIVE")
            if tp is None:
                verd = "STABLE"
            print(f"  {labels[j]:<{NW}}  {base_val:>10.4f}  {m1_val:>12.4f}  "
                  f"{p1_val:>12.4f}  {tp_str:>12}  {verd:>10}")

        print()
        print("  Interpretation: delta = 1 means missing Y values are 1 sigma_Y higher")
        print("  than MAR predicts. Tipping pt = smallest |delta| where conclusion changes.")
        print('=' * W)
        print()

    def __repr__(self) -> str:
        return (f"MissSensitivity(estimator={type(self.estimator).__name__}, "
                f"delta_range={self.delta_range}, n_delta={self.n_delta}, "
                f"m={self.m})")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_y_marginal(
    mean_y: float,
    sigma_y: float,
    n_miss: int,
    rng,
) -> np.ndarray:
    """
    Draw n_miss Y values from N(mean_y, sigma_y^2).

    This represents the MAR baseline imputation for missing Y before the
    delta shift is applied.  The marginal parameters come from the observed
    Y values in the training data.

    Returns
    -------
    y_draws : ndarray of shape (n_miss,)
    """
    return rng.normal(loc=mean_y, scale=max(sigma_y, 1e-10), size=n_miss)
