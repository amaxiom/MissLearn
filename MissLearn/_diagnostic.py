"""
_diagnostic.py  --  Missing data mechanism diagnostics for MissLearn.

MissDiagnostic
    Analyses a dataset to assess whether FIML is appropriate given the
    likely missing data mechanism (MCAR / MAR / MNAR).

    Tests and summaries provided
    ----------------------------
    little_mcar_test()       Little's (1988) chi-squared test of MCAR
    mar_plausibility()       Logistic regression of each missingness
                             indicator on all other observed variables;
                             significant fit = missingness predictable
                             from observed data (consistent with MAR)

    pattern_summary()        Frequency table of unique missingness patterns
    missingness_correlations()
                             Pairwise phi-coefficient correlation between
                             missingness indicators (which variables tend
                             to be missing together)

    descriptive_by_missingness()
                             Compare distributions of observed variables
                             between rows where a target variable is
                             missing vs observed (Mann-Whitney U test)

    summary()                Full plain-language report with verdict

Mechanism primer
----------------
MCAR  Missing Completely At Random  -- missingness is independent of all
      data, observed and unobserved.  Least restrictive; FIML is fully
      valid.  Little's test is the standard formal check.

MAR   Missing At Random  -- missingness may depend on observed variables
      but not on the missing values themselves.  FIML remains valid.
      Formal distinction from MNAR is not testable from data alone, but
      the MAR plausibility test gives useful evidence.

MNAR  Missing Not At Random  -- missingness depends on the unobserved
      values.  FIML may be biased.  No test can confirm MNAR from the
      observed data alone, but low MAR plausibility raises suspicion.

References
----------
Little, R. J. A. (1988).  A test of missing completely at random for
    multivariate data with missing values.  Journal of the American
    Statistical Association, 83(404), 1198-1202.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2, mannwhitneyu


# ---------------------------------------------------------------------------
# MissDiagnostic
# ---------------------------------------------------------------------------

class MissDiagnostic:
    """
    Missing data mechanism diagnostics.

    Parameters
    ----------
    X : array-like of shape (n, p), may contain NaN
    y : array-like of shape (n,), optional, may contain NaN
        If provided, y is appended as an additional column for pattern
        and mechanism analysis.
    feature_names : list of str, optional
        Column labels.  Auto-generated (X0, X1, ...) if not provided.
    alpha : float, default 0.05
        Significance level for all hypothesis tests.

    Call fit() to run all diagnostics, then summary() for the full report.
    Individual methods (little_mcar_test, mar_plausibility, etc.) can be
    called separately without fitting all diagnostics.
    """

    def __init__(
        self,
        X,
        y=None,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ):
        X_arr = np.asarray(X, dtype=float)
        if y is not None:
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
            self.X_       = np.hstack([X_arr, y_arr])
            y_name        = ['y']
        else:
            self.X_       = X_arr
            y_name        = []

        n, p_full = self.X_.shape
        self.n_   = n
        self.p_   = p_full

        if feature_names is not None:
            self.feature_names_ = list(feature_names) + y_name
        else:
            self.feature_names_ = [f"X{j}" for j in range(X_arr.shape[1])] + y_name

        self.alpha = alpha

        # Results populated by fit() or individual methods
        self._little_done     = False
        self._mar_done        = False
        self._pattern_done    = False
        self._corr_done       = False
        self._desc_done       = False

    # ------------------------------------------------------------------
    # Helper: missingness indicator matrix
    # ------------------------------------------------------------------

    def _R(self) -> np.ndarray:
        """Binary missingness indicator matrix, shape (n, p). 1 = missing."""
        return np.isnan(self.X_).astype(float)

    def _obs_mean(self) -> np.ndarray:
        """Column means ignoring NaN."""
        return np.nanmean(self.X_, axis=0)

    def _pairwise_cov(self) -> np.ndarray:
        """
        Pairwise available-case covariance matrix.
        Uses all rows where both columns are observed for each (i, j) pair.
        """
        p = self.p_
        C = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                mask = ~np.isnan(self.X_[:, i]) & ~np.isnan(self.X_[:, j])
                m = mask.sum()
                if m > 1:
                    xi = self.X_[mask, i]
                    xj = self.X_[mask, j]
                    C[i, j] = C[j, i] = np.cov(xi, xj)[0, 1]
        # Diagonal: marginal variances
        for i in range(p):
            mask = ~np.isnan(self.X_[:, i])
            if mask.sum() > 1:
                C[i, i] = np.var(self.X_[mask, i], ddof=1)
        return C

    def _col_label(self, j: int) -> str:
        if j < len(self.feature_names_):
            return self.feature_names_[j]
        return f"col{j}"

    # ------------------------------------------------------------------
    # Little's MCAR test
    # ------------------------------------------------------------------

    def little_mcar_test(self) -> Dict:
        """
        Little's (1988) chi-squared test of MCAR.

        Computes a test statistic based on the Mahalanobis distance between
        the pattern-wise means and the overall estimated means.  Uses
        pairwise available-case estimates for the mean and covariance
        (an approximation to the EM-based estimates in the original paper).

        Returns
        -------
        dict with keys: statistic, df, pvalue, significant, patterns_used
        """
        X    = self.X_
        n, p = X.shape

        mu_hat    = self._obs_mean()
        Sigma_hat = self._pairwise_cov()

        # Group rows by missingness pattern
        R = np.isnan(X)
        pattern_keys = {}
        for i in range(n):
            key = tuple(R[i].astype(int))
            pattern_keys.setdefault(key, []).append(i)

        d2           = 0.0
        df_accum     = 0
        patterns_used = 0

        for pattern, row_indices in pattern_keys.items():
            obs_cols = [j for j, m in enumerate(pattern) if m == 0]
            if len(obs_cols) == 0:
                continue   # all missing: no contribution

            n_k   = len(row_indices)
            if n_k < 2:
                continue   # need at least 2 rows to estimate a mean reliably

            # Mean of observed variables for this pattern
            y_k   = np.nanmean(X[np.ix_(row_indices, obs_cols)], axis=0)
            mu_k  = mu_hat[obs_cols]

            # Covariance submatrix
            Sigma_k = Sigma_hat[np.ix_(obs_cols, obs_cols)]

            # Regularise slightly to avoid singularity
            Sigma_k = Sigma_k + 1e-8 * np.eye(len(obs_cols))

            try:
                # Use solve instead of inv: more numerically stable and avoids
                # forming the full inverse matrix
                v = np.linalg.solve(Sigma_k, y_k - mu_k)
            except np.linalg.LinAlgError:
                continue

            diff = y_k - mu_k
            contrib = n_k * float(diff @ v)
            if np.isfinite(contrib):
                d2 += contrib
                df_accum += len(obs_cols)
                patterns_used += 1

        # Degrees of freedom: total observed cells - number of observed variables
        p_observed = int(np.any(~R, axis=0).sum())
        df = max(1, df_accum - p_observed)

        pvalue = float(1.0 - chi2.cdf(d2, df))

        self.little_statistic_    = d2
        self.little_df_           = df
        self.little_pvalue_       = pvalue
        self.little_significant_  = pvalue < self.alpha
        self.little_patterns_used_ = patterns_used
        self._little_done         = True

        return {
            'statistic':    d2,
            'df':           df,
            'pvalue':       pvalue,
            'significant':  pvalue < self.alpha,
            'patterns_used': patterns_used,
        }

    # ------------------------------------------------------------------
    # MAR plausibility
    # ------------------------------------------------------------------

    def mar_plausibility(self) -> Dict:
        """
        For each variable with missing values, fit a logistic regression
        predicting its missingness indicator from all other observed
        variables (mean-imputed for diagnostic purposes only).

        A significant fit (likelihood ratio test) means the missingness
        is predictable from observed data, which is consistent with MAR.

        Returns
        -------
        dict: {col_name: {'lr_statistic', 'df', 'pvalue', 'significant'}}
        """
        from sklearn.linear_model import LogisticRegression

        X  = self.X_
        n, p = X.shape
        R  = np.isnan(X)

        # Mean-impute predictors (for this diagnostic only) -- vectorised
        col_means = np.nanmean(X, axis=0)
        nan_mask  = np.isnan(X)
        X_imp     = np.where(nan_mask, col_means[None, :], X)

        results: Dict[str, dict] = {}

        for j in range(p):
            n_missing = int(R[:, j].sum())
            if n_missing == 0 or n_missing == n:
                continue   # nothing to test

            R_j     = R[:, j].astype(int)
            pi      = R_j.mean()

            if pi <= 0 or pi >= 1:
                continue

            # Predictors: all other columns (mean-imputed)
            X_pred_cols = [k for k in range(p) if k != j]
            if not X_pred_cols:
                continue
            X_pred = X_imp[:, X_pred_cols]

            # Null log-likelihood (intercept only)
            log_lik_null = float(
                n * (pi * np.log(pi) + (1.0 - pi) * np.log(1.0 - pi))
            )

            # Fit logistic regression
            try:
                lr = LogisticRegression(
                    C=1e6, max_iter=500, solver='lbfgs', random_state=0
                )
                lr.fit(X_pred, R_j)
                proba = lr.predict_proba(X_pred)
                log_lik_full = float(
                    np.sum(np.log(
                        proba[np.arange(n), R_j].clip(1e-10, 1.0)
                    ))
                )
            except Exception:
                continue

            lr_stat = max(0.0, 2.0 * (log_lik_full - log_lik_null))
            df      = len(X_pred_cols)
            pvalue  = float(1.0 - chi2.cdf(lr_stat, df))

            results[self._col_label(j)] = {
                'n_missing':    n_missing,
                'pct_missing':  100.0 * n_missing / n,
                'lr_statistic': lr_stat,
                'df':           df,
                'pvalue':       pvalue,
                'significant':  pvalue < self.alpha,
            }

        self.mar_results_  = results
        self._mar_done     = True
        return results

    # ------------------------------------------------------------------
    # Pattern summary
    # ------------------------------------------------------------------

    def pattern_summary(self) -> List[dict]:
        """
        Frequency table of unique missingness patterns.

        Returns list of dicts sorted by frequency (descending), each with
        keys: 'missing_cols', 'n', 'pct', 'pattern_vector'.
        """
        X    = self.X_
        n, p = X.shape
        R    = np.isnan(X)

        pattern_map: Dict[tuple, int] = {}
        for i in range(n):
            key = tuple(R[i].astype(int))
            pattern_map[key] = pattern_map.get(key, 0) + 1

        patterns = []
        for pat, count in sorted(pattern_map.items(), key=lambda x: -x[1]):
            missing_cols = [self._col_label(j) for j, m in enumerate(pat) if m]
            patterns.append({
                'missing_cols':    missing_cols if missing_cols else ['(none)'],
                'n':               count,
                'pct':             100.0 * count / n,
                'pattern_vector':  np.array(pat),
            })

        self.patterns_       = patterns
        self.n_patterns_     = len(patterns)
        self._pattern_done   = True
        return patterns

    # ------------------------------------------------------------------
    # Missingness correlations
    # ------------------------------------------------------------------

    def missingness_correlations(self) -> np.ndarray:
        """
        Pairwise phi-coefficient correlation between missingness indicators.

        Variables that tend to be missing together have high positive
        correlations (shared mechanism).  Variables that are rarely missing
        at the same time have negative correlations.

        Returns
        -------
        corr : ndarray of shape (p, p)
        """
        R = self._R()
        # Only include columns with any missingness
        has_missing = R.any(axis=0)
        R_sub = R[:, has_missing]
        cols  = [self._col_label(j) for j in range(self.p_) if has_missing[j]]

        if R_sub.shape[1] < 2:
            self.miss_corr_        = np.array([[1.0]])
            self.miss_corr_cols_   = cols
            self._corr_done        = True
            return self.miss_corr_

        # Pearson on binary indicators = phi coefficient
        corr = np.corrcoef(R_sub.T)
        corr = np.nan_to_num(corr, nan=0.0)

        self.miss_corr_      = corr
        self.miss_corr_cols_ = cols
        self._corr_done      = True
        return corr

    # ------------------------------------------------------------------
    # Descriptive comparison by missingness
    # ------------------------------------------------------------------

    def descriptive_by_missingness(self) -> Dict[str, Dict]:
        """
        For each variable j with missing values, compare the distribution
        of all other variables between rows where j is missing vs observed.

        Uses the two-sided Mann-Whitney U test (non-parametric).
        A significant p-value suggests that missingness in j is associated
        with the values of another variable, which is evidence for MAR (or
        possibly MNAR if the association is with j itself).

        Returns
        -------
        dict: {col_j: {col_k: {'mean_observed', 'mean_missing',
                                'u_statistic', 'pvalue', 'significant'}}}
        """
        X    = self.X_
        n, p = X.shape
        R    = np.isnan(X)

        desc: Dict[str, Dict] = {}

        for j in range(p):
            if R[:, j].sum() == 0:
                continue
            miss_mask = R[:, j]
            obs_mask  = ~miss_mask

            if obs_mask.sum() < 2 or miss_mask.sum() < 2:
                continue

            col_j_results: Dict[str, dict] = {}
            for k in range(p):
                if k == j:
                    continue
                vals_obs  = X[obs_mask,  k]
                vals_miss = X[miss_mask, k]
                # Remove NaN from each group
                vals_obs  = vals_obs[~np.isnan(vals_obs)]
                vals_miss = vals_miss[~np.isnan(vals_miss)]

                if len(vals_obs) < 2 or len(vals_miss) < 2:
                    continue

                try:
                    stat, pval = mannwhitneyu(
                        vals_obs, vals_miss, alternative='two-sided'
                    )
                except Exception:
                    continue

                col_j_results[self._col_label(k)] = {
                    'mean_observed': float(np.mean(vals_obs)),
                    'mean_missing':  float(np.mean(vals_miss)),
                    'u_statistic':   float(stat),
                    'pvalue':        float(pval),
                    'significant':   float(pval) < self.alpha,
                }

            if col_j_results:
                desc[self._col_label(j)] = col_j_results

        self.desc_results_ = desc
        self._desc_done    = True
        return desc

    # ------------------------------------------------------------------
    # fit  (run all diagnostics at once)
    # ------------------------------------------------------------------

    def fit(self):
        """
        Run all diagnostic tests and store results.

        Returns self.
        """
        self.pattern_summary()
        self.little_mcar_test()
        self.mar_plausibility()
        self.missingness_correlations()
        self.descriptive_by_missingness()
        return self

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """
        Print a full plain-language diagnostic report.
        Runs any tests not yet computed.
        """
        if not self._pattern_done:
            self.pattern_summary()
        if not self._little_done:
            self.little_mcar_test()
        if not self._mar_done:
            self.mar_plausibility()
        if not self._corr_done:
            self.missingness_correlations()

        W = 70
        print()
        print('=' * W)
        print(f"{'MissDiagnostic  --  Missing Data Mechanism Report':^{W}}")
        print('=' * W)
        print(f"  n = {self.n_}  |  p = {self.p_}  |  "
              f"significance level alpha = {self.alpha}")

        # Overall missingness summary
        R = self._R()
        total_cells = self.n_ * self.p_
        missing_cells = int(R.sum())
        print(f"  Overall missingness: {missing_cells}/{total_cells} cells "
              f"({100.*missing_cells/total_cells:.1f}%)")
        col_miss = R.mean(axis=0)
        for j in range(self.p_):
            if col_miss[j] > 0:
                bar_w  = 20
                filled = int(round(col_miss[j] * bar_w))
                bar    = '#' * filled + ' ' * (bar_w - filled)
                print(f"    {self._col_label(j):<12} [{bar}] "
                      f"{col_miss[j]*100:5.1f}%")

        # ------------------------------------------------------------------
        # Missingness patterns
        # ------------------------------------------------------------------
        print()
        print(f"  {'Missingness Patterns':}")
        print('  ' + '-' * (W - 2))
        max_show = 10
        for i, pat in enumerate(self.patterns_[:max_show]):
            miss_str = ', '.join(pat['missing_cols'])
            print(f"  {i+1:>3}. n={pat['n']:>4} ({pat['pct']:5.1f}%)  "
                  f"missing: {miss_str}")
        if self.n_patterns_ > max_show:
            print(f"  ... and {self.n_patterns_ - max_show} more patterns.")
        print(f"  Total unique patterns: {self.n_patterns_}")

        # ------------------------------------------------------------------
        # Little's MCAR test
        # ------------------------------------------------------------------
        print()
        print(f"  Little's MCAR Test")
        print('  ' + '-' * (W - 2))
        print(f"  chi2({self.little_df_}) = {self.little_statistic_:.3f}  "
              f"p = {self.little_pvalue_:.4f}  "
              f"(patterns used: {self.little_patterns_used_})")
        if not self.little_significant_:
            print(f"  Result: Cannot reject MCAR (p >= {self.alpha}).  "
                  f"Data is consistent with MCAR.")
            print(f"  FIML is fully appropriate under MCAR.")
        else:
            print(f"  Result: MCAR rejected (p < {self.alpha}).  "
                  f"Data is NOT missing completely at random.")
            print(f"  This does not invalidate FIML -- see MAR analysis below.")

        # ------------------------------------------------------------------
        # MAR plausibility
        # ------------------------------------------------------------------
        print()
        print(f"  MAR Plausibility  (logistic regression of missingness "
              f"on observed variables)")
        print('  ' + '-' * (W - 2))

        if not self.mar_results_:
            print("  No variables with testable missingness.")
        else:
            n_sig = sum(1 for r in self.mar_results_.values() if r['significant'])
            print(f"  {'Variable':<14} {'%Missing':>9}  "
                  f"{'LR stat':>9}  {'df':>4}  {'p-value':>9}  {'MAR evidence'}")
            print('  ' + '-' * (W - 2))
            for col_name, res in self.mar_results_.items():
                flag = 'YES *' if res['significant'] else 'no'
                print(f"  {col_name:<14} {res['pct_missing']:>8.1f}%  "
                      f"{res['lr_statistic']:>9.3f}  "
                      f"{res['df']:>4}  "
                      f"{res['pvalue']:>9.4f}  {flag}")
            print()
            if n_sig == 0:
                print(f"  No variables show significant MAR evidence.  "
                      f"Missingness does not appear predictable from observed data.")
            elif n_sig == len(self.mar_results_):
                print(f"  All {n_sig} variable(s) with missingness show evidence "
                      f"of MAR.  FIML remains appropriate.")
            else:
                print(f"  {n_sig}/{len(self.mar_results_)} variable(s) show MAR "
                      f"evidence.  FIML is likely appropriate.")

        # ------------------------------------------------------------------
        # Missingness correlations
        # ------------------------------------------------------------------
        if hasattr(self, 'miss_corr_') and len(self.miss_corr_cols_) >= 2:
            print()
            print(f"  Missingness Correlations (phi coefficients, "
                  f"|r| > 0.3 shown)")
            print('  ' + '-' * (W - 2))
            cols = self.miss_corr_cols_
            corr = self.miss_corr_
            found_any = False
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r = corr[i, j]
                    if abs(r) > 0.3:
                        direction = 'often missing together' if r > 0 \
                                    else 'rarely missing together'
                        print(f"  {cols[i]} -- {cols[j]}: "
                              f"r = {r:.3f}  ({direction})")
                        found_any = True
            if not found_any:
                print("  No strong pairwise missingness correlations (|r| <= 0.3).")

        # ------------------------------------------------------------------
        # Descriptive by missingness (strongest associations only)
        # ------------------------------------------------------------------
        if self._desc_done and self.desc_results_:
            print()
            print(f"  Descriptive Comparison by Missingness "
                  f"(significant associations, alpha={self.alpha})")
            print('  ' + '-' * (W - 2))
            any_shown = False
            for col_j, comparisons in self.desc_results_.items():
                sig_comps = {k: v for k, v in comparisons.items()
                             if v['significant']}
                if not sig_comps:
                    continue
                print(f"  When '{col_j}' is missing vs observed:")
                for col_k, res in sig_comps.items():
                    diff = res['mean_missing'] - res['mean_observed']
                    print(f"    {col_k:<12}  "
                          f"mean(obs)={res['mean_observed']:>8.3f}  "
                          f"mean(miss)={res['mean_missing']:>8.3f}  "
                          f"diff={diff:+.3f}  "
                          f"p={res['pvalue']:.4f}")
                any_shown = True
            if not any_shown:
                print("  No significant associations found.")

        # ------------------------------------------------------------------
        # Overall verdict
        # ------------------------------------------------------------------
        print()
        print(f"  {'VERDICT':}")
        print('  ' + '=' * (W - 2))
        self._print_verdict()
        print('=' * W)
        print()

    def _print_verdict(self) -> None:
        """Print a plain-language recommendation."""
        mcar_ok  = not self.little_significant_
        mar_ok   = (self.mar_results_ and
                    any(r['significant'] for r in self.mar_results_.values()))
        n_mar_sig = sum(1 for r in self.mar_results_.values()
                        if r['significant']) if self.mar_results_ else 0
        n_mar_tot = len(self.mar_results_)

        if mcar_ok:
            print("  [OK] Data is consistent with MCAR.  FIML is fully "
                  "appropriate and will produce unbiased estimates.")
        elif n_mar_tot == 0:
            print("  [WARN] MCAR was rejected but no variables had testable "
                  "missingness for the MAR plausibility test.  Proceed with "
                  "FIML but treat results with caution.")
        elif n_mar_sig > 0:
            frac = n_mar_sig / n_mar_tot
            if frac >= 0.5:
                print(f"  [OK] MCAR was rejected but {n_mar_sig}/{n_mar_tot} "
                      f"variables show MAR evidence -- missingness is "
                      f"predictable from observed data.  FIML remains valid "
                      f"under MAR.")
            else:
                print(f"  [WARN] MCAR was rejected and only {n_mar_sig}/"
                      f"{n_mar_tot} variables show MAR evidence.  FIML is "
                      f"likely still appropriate but consider whether any "
                      f"missing values depend on the unobserved data itself "
                      f"(MNAR).  If so, results may be biased.")
        else:
            print("  [WARN] MCAR was rejected and no MAR evidence was found.  "
                  "Missingness is not predictable from observed variables.  "
                  "MNAR cannot be ruled out.  FIML results may be biased.  "
                  "Consider a sensitivity analysis or domain knowledge about "
                  "the missing data mechanism before relying on FIML estimates.")

        print()
        print("  Note: MAR vs MNAR cannot be formally distinguished from the")
        print("  observed data alone.  The above is a plausibility assessment,")
        print("  not a definitive test.  Domain knowledge should inform the")
        print("  final judgement.")
