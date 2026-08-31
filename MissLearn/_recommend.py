"""
Model recommendation for incomplete data.

Given an incomplete design matrix, this module gathers evidence about the
data (shape, missingness mechanism, conditional-mean linearity, tail
behaviour, grouping structure) and turns that evidence into a ranked,
justified choice of MissLearn estimator.

The recommender is deliberately a *triage* tool, not an oracle.  Every score
it produces is a heuristic built from the diagnostics below, and it reports
the reasons alongside the ranking so the user can disagree with it.
Cross-validation on the actual task remains the arbiter; the value here is
that the shortlist is defensible and the structural mistakes (a Gaussian
process on 9000 rows, a fixed-effects model on clustered data, imputing a
column that is 90% absent) are caught before any fitting happens.

Usage
-----
    from MissLearn import MissRecommender

    rec = MissRecommender(feature_names=cols).fit(X, y)
    rec.summary()                 # readable report with reasons
    model = rec.make_estimator()  # the top-ranked model, configured
    model.fit(X, y)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._diagnostic import MissDiagnostic
from ._validate import prefit_check

__all__ = ['MissRecommender', 'recommend_model']


# ---------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------
# Each entry maps a family key to its regression and classification classes,
# plus a short description of what the model assumes.  The recommender scores
# family keys, then resolves to a concrete class once the task is known.

_CANDIDATES = {
    'MissLinear': {
        'reg': 'MissLinear', 'clf': 'MissLogistic',
        'label': 'linear / logistic regression',
        'assumes': 'linear conditional mean, no strong collinearity',
    },
    'MissRidge': {
        'reg': 'MissRidgeRegressor', 'clf': 'MissRidgeClassifier',
        'label': 'L2-penalized linear / logistic',
        'assumes': 'linear conditional mean, many or collinear predictors',
    },
    'MissLASSO': {
        'reg': 'MissLASSORegressor', 'clf': 'MissLASSOClassifier',
        'label': 'L1-penalized linear / logistic',
        'assumes': 'linear conditional mean, a sparse subset of predictors',
    },
    'MissBayes': {
        'reg': 'MissBayesRegressor', 'clf': 'MissBayesClassifier',
        'label': 'full-covariance generative Gaussian',
        'assumes': 'class-conditional / joint normality of the predictors',
    },
    'MissNeighbors': {
        'reg': 'MissNeighborsRegressor', 'clf': 'MissNeighborsClassifier',
        'label': 'k-nearest neighbours, expected distance',
        'assumes': 'local smoothness, no global functional form',
    },
    'MissSupport': {
        'reg': 'MissSupportRegressor', 'clf': 'MissSupportClassifier',
        'label': 'support vector, expected kernel',
        'assumes': 'a smooth nonlinear decision surface',
    },
    'MissGaussian': {
        'reg': 'MissGaussianRegressor', 'clf': 'MissGaussianClassifier',
        'label': 'Gaussian process, marginalised kernel',
        'assumes': 'smoothness, and n small enough for exact O(n^3) inference',
    },
    'MissMixed': {
        'reg': 'MissMixedRegressor', 'clf': 'MissMixedClassifier',
        'label': 'random-intercept mixed effects',
        'assumes': 'observations clustered within groups',
    },
}

_FAMILY_ORDER = list(_CANDIDATES)


#: How far above the trivial baseline a linearity probe must score before the
#: comparison between the two probes counts as evidence. Both arms can sit at
#: or below the baseline, and the ratio of their lifts then still resolves to a
#: number and still names a winner.
_PROBE_SKILL_MARGIN = 0.01


class _Score:
    """Accumulates a score and the reasons that produced it."""

    def __init__(self, family: str):
        self.family = family
        self.points = 0.0
        self.reasons: List[str] = []
        self.vetoed = False
        self.veto_reason = ''

    def add(self, points: float, reason: str) -> None:
        self.points += points
        sign = '+' if points >= 0 else ''
        self.reasons.append(f"{sign}{points:g}  {reason}")

    def veto(self, reason: str) -> None:
        self.vetoed = True
        self.veto_reason = reason


class MissRecommender:
    """
    Recommend a MissLearn estimator from the structure of incomplete data.

    Parameters
    ----------
    task : {'auto', 'regression', 'classification'}, default 'auto'
        Detected from y when 'auto'.
    groups : array-like of shape (n,), optional
        Cluster label per row (patient, site, batch).  When supplied, the
        recommender estimates the intraclass correlation and will promote
        the mixed-effects family if clustering is material.  Without this
        argument the mixed-effects family cannot be recommended at all,
        because it has no grouping variable to use.
    feature_names : list of str, optional
    alpha : float, default 0.05
        Significance level for the mechanism tests.
    drop_threshold : float, default 0.60
        Columns missing more than this fraction are recommended for removal
        rather than for modelling.  Above roughly this level the conditional
        moments of the column are driven by a small and probably unrepresent-
        ative subset of rows, and retaining it adds parameters without adding
        information.
    high_missing : float, default 0.30
        Overall missing-cell fraction above which the recommender treats the
        problem as missingness-dominated.
    gp_max_n : int, default 1000
        Row count above which the Gaussian process family is vetoed on cost.
    probe_nonlinearity : bool, default True
        Run a cheap mean-imputed comparison of a linear model against a
        nearest-neighbour model to detect curvature in the conditional mean.
        This is the only part of the recommender that fits anything.
    probe_max_n : int, default 2000
        Subsample size for the nonlinearity probe.
    random_state : int, default 0

    Attributes
    ----------
    task_            : str, the resolved task
    recommended_     : str, class name of the top-ranked estimator
    ranked_          : list of dicts, all families with score and reasons
    evidence_        : dict, everything the decision was based on
    preprocessing_   : dict, drop_columns / copula / standardise advice
    followups_       : list of dicts, recommended post-fit analyses
    notes_           : list of str, cost and caveat warnings
    diagnostic_      : the fitted MissDiagnostic
    """

    def __init__(
        self,
        task: str = 'auto',
        groups: Optional[Sequence] = None,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        drop_threshold: float = 0.60,
        high_missing: float = 0.30,
        gp_max_n: int = 1000,
        probe_nonlinearity: bool = True,
        probe_max_n: int = 2000,
        random_state: int = 0,
    ):
        self.task = task
        self.groups = groups
        self.feature_names = feature_names
        self.alpha = alpha
        self.drop_threshold = drop_threshold
        self.high_missing = high_missing
        self.gp_max_n = gp_max_n
        self.probe_nonlinearity = probe_nonlinearity
        self.probe_max_n = probe_max_n
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Task detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_task(y: np.ndarray) -> str:
        y_obs = y[~np.isnan(y)]
        if y_obs.size == 0:
            raise ValueError("y is entirely missing; cannot detect the task.")
        uniq = np.unique(y_obs)
        if uniq.size <= 2:
            return 'classification'
        # Integer-valued with few levels is treated as multiclass.
        if uniq.size <= 20 and np.allclose(uniq, np.round(uniq)):
            return 'classification'
        return 'regression'

    # ------------------------------------------------------------------
    # Evidence: intraclass correlation
    # ------------------------------------------------------------------

    @staticmethod
    def _icc(y: np.ndarray, groups: np.ndarray) -> Optional[float]:
        """
        One-way random-effects intraclass correlation on the observed y.

        Returns None when there are too few groups to estimate it.
        """
        obs = ~np.isnan(y)
        y_o, g_o = y[obs], np.asarray(groups)[obs]
        labels = np.unique(g_o)
        if labels.size < 2 or y_o.size <= labels.size:
            return None

        grand = y_o.mean()
        ss_between = 0.0
        ss_within = 0.0
        for lab in labels:
            yi = y_o[g_o == lab]
            if yi.size == 0:
                continue
            ss_between += yi.size * (yi.mean() - grand) ** 2
            ss_within += float(np.sum((yi - yi.mean()) ** 2))

        df_b = labels.size - 1
        df_w = y_o.size - labels.size
        if df_b <= 0 or df_w <= 0:
            return None

        ms_b = ss_between / df_b
        ms_w = ss_within / df_w
        # Balanced-design approximation to the group size.
        k = y_o.size / labels.size
        var_b = max(0.0, (ms_b - ms_w) / k)
        total = var_b + ms_w
        return float(var_b / total) if total > 0 else None

    # ------------------------------------------------------------------
    # Evidence: is the conditional mean linear?
    # ------------------------------------------------------------------

    def _probe(self, X: np.ndarray, y: np.ndarray, task: str) -> Optional[Dict]:
        """
        Compare a linear model against a nearest-neighbour model on
        mean-imputed data under a common CV split.

        This is a probe, not a benchmark.  Both arms see identical, crudely
        imputed data; the point is only the *ratio* between them, which is
        informative about curvature in the conditional mean even though the
        absolute scores are pessimistic.  A ratio near one says the linear
        families are not leaving anything obvious on the table.
        """
        if not self.probe_nonlinearity:
            return None
        try:
            from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge, LogisticRegression
            from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        except ImportError:                                  # pragma: no cover
            return None

        obs = ~np.isnan(y)
        Xp, yp = X[obs], y[obs]
        if Xp.shape[0] < 40:
            return None

        rng = np.random.default_rng(self.random_state)
        if Xp.shape[0] > self.probe_max_n:
            idx = rng.choice(Xp.shape[0], self.probe_max_n, replace=False)
            Xp, yp = Xp[idx], yp[idx]

        # Crude mean imputation, identical for both arms.
        mu = np.nanmean(Xp, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Xi = np.where(np.isnan(Xp), mu, Xp)
        keep = np.nanstd(Xi, axis=0) > 0
        if keep.sum() < 1:
            return None
        Xi = Xi[:, keep]

        k = min(5, max(2, Xi.shape[0] // 20))
        try:
            if task == 'classification':
                yp = yp.astype(int)
                if np.unique(yp).size < 2:
                    return None
                cv = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
                lin = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000))
                nn = make_pipeline(StandardScaler(),
                                   KNeighborsClassifier(n_neighbors=min(15, max(3, Xi.shape[0] // 20))))
                scoring = 'accuracy'
                floor = 1.0 / np.unique(yp).size
            else:
                cv = KFold(k, shuffle=True, random_state=self.random_state)
                lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                nn = make_pipeline(StandardScaler(),
                                   KNeighborsRegressor(n_neighbors=min(15, max(3, Xi.shape[0] // 20))))
                scoring = 'r2'
                floor = 0.0

            s_lin = float(np.mean(cross_val_score(lin, Xi, yp, cv=cv, scoring=scoring)))
            s_nn = float(np.mean(cross_val_score(nn, Xi, yp, cv=cv, scoring=scoring)))
        except Exception:
            return None

        # Compare skill above the trivial baseline, so a ratio is meaningful
        # even when both scores are small. Small is not the same as absent,
        # though: when neither arm clears the baseline the floors below make
        # the ratio 0, which reads as a decisive win for the linear probe. The
        # caller is told whether there was any skill to compare.
        lift_lin = max(1e-6, s_lin - floor)
        lift_nn = max(0.0, s_nn - floor)
        return {
            'linear_score': s_lin,
            'neighbour_score': s_nn,
            'metric': scoring,
            'ratio': lift_nn / lift_lin,
            'floor': floor,
            'has_skill': bool(max(s_lin, s_nn) - floor > _PROBE_SKILL_MARGIN),
        }

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y) -> 'MissRecommender':
        """Gather evidence and rank the candidate families."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        names = (list(self.feature_names) if self.feature_names is not None
                 else [f"X{j}" for j in range(p)])

        task = self.task if self.task != 'auto' else self._detect_task(y)
        self.task_ = task

        # -- basic missingness structure ------------------------------------
        R = np.isnan(X)
        col_rate = R.mean(axis=0)
        cell_rate = float(R.mean())
        row_rate = float(R.any(axis=1).mean())
        n_complete = int((~R.any(axis=1) & ~np.isnan(y)).sum())
        y_rate = float(np.isnan(y).mean())

        diag = MissDiagnostic(X, y=None, feature_names=names, alpha=self.alpha)
        little = diag.little_mcar_test()
        mar = diag.mar_plausibility()
        patterns = diag.pattern_summary()
        self.diagnostic_ = diag

        n_patterns = len(patterns)

        # -- structural facts ------------------------------------------------
        drop_cols = [names[j] for j in range(p) if col_rate[j] > self.drop_threshold]
        keep_idx = [j for j in range(p) if col_rate[j] <= self.drop_threshold]
        p_model = len(keep_idx)

        # Mechanism reading.  Little's test rejecting says "not MCAR".  What
        # matters next is whether the missingness of the substantially-missing
        # columns is predictable from the other observed columns: if it is,
        # MAR is tenable and FIML is consistent; if it is not, MNAR cannot be
        # ruled out from the data alone and sensitivity analysis is required.
        not_mcar = bool(little.get('significant', False))
        substantial = [c for j, c in enumerate(names)
                       if 0.02 < col_rate[j] <= self.drop_threshold]
        mar_supported = [c for c in substantial
                         if mar.get(c, {}).get('significant', False)]
        mar_unsupported = [c for c in substantial if c not in mar_supported]
        mnar_suspected = bool(not_mcar and mar_unsupported)

        # -- distributional checks -------------------------------------------
        chk = prefit_check(X, y, feature_names=names, raise_on_error=False,
                           emit_warnings=False)
        # Read notes as well as warnings. prefit_check files a kurtosis
        # finding as a note, not a warning, so searching only warnings made
        # heavy_tailed False on every dataset however heavy the tails, and
        # took the copula recommendation, the family score adjustments and
        # the summary line down with it. Checking both lists means a future
        # reclassification cannot break it the same way again.
        _advisories = list(chk.warnings) + list(chk.notes)
        heavy_tailed = any('kurtosis' in w.lower() for w in _advisories)
        scale_imbalance = any('scale' in w.lower() for w in _advisories)

        # -- grouping ---------------------------------------------------------
        icc = None
        n_groups = None
        if self.groups is not None:
            g = np.asarray(self.groups)
            n_groups = int(np.unique(g).size)
            if task == 'regression':
                icc = self._icc(y, g)

        probe = self._probe(X, y, task)

        self.evidence_ = {
            'n': n, 'p': p, 'p_after_drop': p_model,
            'task': task,
            'cell_missing_rate': cell_rate,
            'row_missing_rate': row_rate,
            'y_missing_rate': y_rate,
            'n_complete_cases': n_complete,
            'n_patterns': n_patterns,
            'column_missing_rate': dict(zip(names, col_rate.tolist())),
            'little_mcar': little,
            'mar_supported': mar_supported,
            'mar_unsupported': mar_unsupported,
            'not_mcar': not_mcar,
            'mnar_suspected': mnar_suspected,
            'heavy_tailed': heavy_tailed,
            'scale_imbalance': scale_imbalance,
            'n_groups': n_groups,
            'icc': icc,
            'nonlinearity_probe': probe,
        }

        self._score(n, p_model, n_complete, cell_rate, task, probe,
                    heavy_tailed, icc, n_groups)
        self._advise(names, drop_cols, col_rate, heavy_tailed,
                     scale_imbalance, mnar_suspected, y_rate,
                     n_patterns, n, p_model)
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, n, p, n_complete, cell_rate, task, probe,
               heavy_tailed, icc, n_groups) -> None:
        S = {k: _Score(k) for k in _FAMILY_ORDER}

        # -- vetoes ----------------------------------------------------------
        if self.groups is None:
            S['MissMixed'].veto(
                "no groups argument supplied, so there is no clustering "
                "variable for a random intercept")
        if n > self.gp_max_n:
            S['MissGaussian'].veto(
                f"exact GP inference is O(n^3) and n={n} exceeds the "
                f"gp_max_n={self.gp_max_n} threshold")

        # -- clustering ------------------------------------------------------
        if icc is not None and n_groups is not None:
            if icc >= 0.05:
                S['MissMixed'].add(
                    4.0, f"observations cluster within {n_groups} groups "
                         f"(ICC={icc:.2f}); a fixed-effects fit would treat "
                         f"correlated rows as independent")
                for k in ('MissLinear', 'MissRidge', 'MissLASSO'):
                    S[k].add(-1.5, f"ignores the group structure (ICC={icc:.2f})")
            else:
                S['MissMixed'].add(
                    -1.0, f"grouping present but negligible (ICC={icc:.2f}), "
                          f"so the random intercept buys little")

        # -- dimensionality --------------------------------------------------
        if p >= n_complete or p > 0.5 * n:
            S['MissRidge'].add(2.5, f"p={p} is large relative to n={n}; "
                                    f"shrinkage is needed for a stable fit")
            S['MissLASSO'].add(2.5, f"p={p} is large relative to n={n}; "
                                    f"an L1 penalty both shrinks and selects")
            S['MissLinear'].add(-2.5, "unpenalized fit is unstable when p "
                                      "approaches n")
            S['MissBayes'].add(-2.0, f"a full p x p covariance needs "
                                     f"{p * (p + 1) // 2} parameters, too many "
                                     f"here")
        elif p >= 15:
            S['MissRidge'].add(1.0, f"p={p} is moderately large; mild "
                                    f"shrinkage is usually worthwhile")
            S['MissLASSO'].add(0.5, f"p={p} may contain irrelevant predictors")

        # -- linearity evidence ----------------------------------------------
        if probe is not None:
            r = probe['ratio']
            lin, nn = probe['linear_score'], probe['neighbour_score']
            if not probe.get('has_skill', True):
                # Two failures ranked against each other are not evidence for
                # the better-placed one. On a target independent of X this
                # branch used to award +2 to MissLinear and report "a linear
                # conditional mean is adequate" from an r2 of -0.013 against
                # -0.064, neither of which beats predicting the mean.
                msg = (f"neither probe beats the trivial baseline "
                       f"({probe['metric']} {lin:.3f} linear against "
                       f"{nn:.3f} nearest-neighbour), so their ordering says "
                       f"nothing about the shape of the conditional mean")
                for k in ('MissLinear', 'MissRidge', 'MissLASSO',
                          'MissNeighbors', 'MissSupport', 'MissGaussian',
                          'MissBayes'):
                    if k in S:
                        S[k].add(0.0, msg)
            elif r > 1.15:
                msg = (f"a nearest-neighbour probe beats a linear one "
                       f"({probe['metric']} {nn:.3f} vs {lin:.3f}), so the "
                       f"conditional mean is not linear")
                for k, pts in (('MissNeighbors', 2.5), ('MissSupport', 2.5),
                               ('MissGaussian', 2.0), ('MissBayes', 0.5)):
                    S[k].add(pts, msg)
                for k in ('MissLinear', 'MissRidge', 'MissLASSO'):
                    S[k].add(-2.0, msg)
            elif r < 0.9:
                msg = (f"a linear probe beats a nearest-neighbour one "
                       f"({probe['metric']} {lin:.3f} vs {nn:.3f}), so a "
                       f"linear conditional mean is adequate")
                for k, pts in (('MissLinear', 2.0), ('MissRidge', 1.5),
                               ('MissLASSO', 1.0), ('MissBayes', 1.0)):
                    S[k].add(pts, msg)
                for k in ('MissNeighbors', 'MissSupport'):
                    S[k].add(-1.5, msg)
            else:
                for k in ('MissLinear', 'MissRidge', 'MissBayes'):
                    S[k].add(0.5, f"linear and nearest-neighbour probes are "
                                  f"within {abs(1 - r) * 100:.0f}% of each "
                                  f"other, so prefer the simpler model")

        # -- missingness load -------------------------------------------------
        if cell_rate >= self.high_missing:
            S['MissBayes'].add(
                2.0, f"{cell_rate:.0%} of cells are missing; a full-covariance "
                     f"generative model borrows the most strength across "
                     f"correlated predictors when marginalising")
            S['MissNeighbors'].add(
                -1.0, "distance-based methods degrade fastest when a large "
                      "fraction of coordinates must be marginalised")
        elif cell_rate >= 0.10:
            S['MissBayes'].add(
                1.0, f"{cell_rate:.0%} of cells are missing, enough that "
                     f"exploiting predictor correlation pays")

        # -- task-specific ------------------------------------------------------
        if task == 'classification':
            S['MissBayes'].add(
                1.0, "class-conditional marginalisation propagates predictor "
                     "uncertainty into the predicted probabilities, which "
                     "usually shows up as a better Brier score")

        # -- tails --------------------------------------------------------------
        if heavy_tailed:
            for k in ('MissNeighbors', 'MissSupport'):
                S[k].add(1.0, "heavy-tailed predictors detected; methods "
                              "without a global normality assumption are safer")
            S['MissBayes'].add(-1.0, "heavy-tailed predictors violate the "
                                     "Gaussian generative assumption")

        ranked = []
        for k in _FAMILY_ORDER:
            s = S[k]
            spec = _CANDIDATES[k]
            cls = spec['reg'] if self.task_ == 'regression' else spec['clf']
            ranked.append({
                'family': k,
                'estimator': cls,
                'label': spec['label'],
                'assumes': spec['assumes'],
                'score': s.points,
                'reasons': s.reasons,
                'vetoed': s.vetoed,
                'veto_reason': s.veto_reason,
            })

        live = [r for r in ranked if not r['vetoed']]
        live.sort(key=lambda r: (-r['score'], _FAMILY_ORDER.index(r['family'])))
        dead = [r for r in ranked if r['vetoed']]
        self.ranked_ = live + dead
        self.recommended_ = live[0]['estimator'] if live else 'MissRidge'
        self.recommended_family_ = live[0]['family'] if live else 'MissRidge'

    # ------------------------------------------------------------------
    # Preprocessing and follow-up advice
    # ------------------------------------------------------------------

    def _advise(self, names, drop_cols, col_rate, heavy_tailed,
                scale_imbalance, mnar_suspected, y_rate,
                n_patterns, n, p_model) -> None:
        rates = dict(zip(names, col_rate.tolist()))
        self.preprocessing_ = {
            'drop_columns': drop_cols,
            'drop_reason': {
                c: (f"{rates[c]:.0%} missing, above the {self.drop_threshold:.0%} "
                    f"threshold: its conditional moments would be estimated "
                    f"from a small and probably unrepresentative subset")
                for c in drop_cols
            },
            'copula': bool(heavy_tailed),
            'standardise': bool(scale_imbalance),
        }

        followups = [{
            'tool': 'MissExplainer',
            'required': False,
            'why': "attributes the prediction both to observed feature values "
                   "and to the act of observing each feature, which is what "
                   "tells you where measurement effort is worth spending",
        }]
        followups.append({
            'tool': 'MissSensitivity',
            'required': bool(mnar_suspected or y_rate > 0),
            'why': (
                "missingness is not MCAR and is not well predicted by the "
                "observed columns, so MNAR cannot be excluded from the data "
                "alone; a delta-adjustment sweep shows how far the conclusions "
                "can be pushed before they change"
                if mnar_suspected else
                "confirms the conclusions are stable under plausible MNAR "
                "departures"),
        })
        self.followups_ = followups

        notes: List[str] = []
        if n_patterns > max(30, n / 20):
            notes.append(
                f"{n_patterns} distinct missingness patterns were found. "
                f"Stage 1 costs O(G p^3) in the pattern count G, so fitting "
                f"will be slower than the row count alone suggests.")
        if self.evidence_['n_complete_cases'] < p_model + 2:
            notes.append(
                f"only {self.evidence_['n_complete_cases']} complete cases for "
                f"{p_model} modelled predictors. Complete-case analysis is not "
                f"an option here, and the optimiser will start from a "
                f"mean-imputed seed.")
        elif self.evidence_['row_missing_rate'] > 0.5:
            notes.append(
                f"{self.evidence_['row_missing_rate']:.0%} of rows have at "
                f"least one missing value, so dropping rows would discard most "
                f"of the data.")
        if y_rate > 0:
            notes.append(
                f"{y_rate:.0%} of the response values are missing. FIML uses "
                f"those rows to estimate the predictor moments even though "
                f"they contribute no conditional likelihood term.")
        self.notes_ = notes

    # ------------------------------------------------------------------
    # Estimator construction
    # ------------------------------------------------------------------

    def make_estimator(self, family: Optional[str] = None, **overrides):
        """
        Instantiate the recommended estimator, configured from the evidence.

        Parameters
        ----------
        family : str, optional
            Family key to build instead of the top-ranked one.
        **overrides
            Passed to the constructor, taking precedence over the
            recommender's own choices.
        """
        import MissLearn as ml

        key = family or self.recommended_family_
        if key not in _CANDIDATES:
            raise ValueError(f"Unknown family {key!r}. "
                             f"Choose one of {sorted(_CANDIDATES)}.")
        spec = _CANDIDATES[key]
        cls_name = spec['reg'] if self.task_ == 'regression' else spec['clf']
        cls = getattr(ml, cls_name)

        kwargs: Dict = {}
        if self.preprocessing_.get('copula'):
            kwargs['copula'] = True
        kwargs.update(overrides)

        # Only pass arguments the class actually accepts.
        import inspect
        accepted = set(inspect.signature(cls.__init__).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def summary(self, top: int = 4) -> None:
        """Print the evidence, the ranking, and the reasons behind it."""
        e = self.evidence_
        line = '=' * 74

        print(line)
        print("  MissLearn model recommendation")
        print(line)
        print(f"  Task                {e['task']}")
        print(f"  Shape               n={e['n']}, p={e['p']}"
              + (f" ({e['p_after_drop']} after recommended drops)"
                 if e['p_after_drop'] != e['p'] else ""))
        print(f"  Missing cells       {e['cell_missing_rate']:.1%}")
        print(f"  Rows with any hole  {e['row_missing_rate']:.1%}")
        print(f"  Complete cases      {e['n_complete_cases']}")
        print(f"  Missing patterns    {e['n_patterns']}")
        if e['y_missing_rate'] > 0:
            print(f"  Missing response    {e['y_missing_rate']:.1%}")
        if e['icc'] is not None:
            print(f"  Grouping            {e['n_groups']} groups, "
                  f"ICC={e['icc']:.3f}")

        lm = e['little_mcar']
        verdict = ("rejected, so the data are not MCAR"
                   if e['not_mcar'] else "not rejected")
        print(f"\n  Little's MCAR test  chi2={lm.get('statistic', float('nan')):.1f}, "
              f"df={lm.get('df', 0)}, p={lm.get('pvalue', float('nan')):.3g}"
              f"  [{verdict}]")
        if e['mar_supported']:
            print(f"  MAR-consistent      {', '.join(e['mar_supported'])}")
        if e['mar_unsupported']:
            print(f"  Not MAR-explained   {', '.join(e['mar_unsupported'])}")
        if e['nonlinearity_probe']:
            pr = e['nonlinearity_probe']
            print(f"  Linearity probe     linear {pr['metric']}="
                  f"{pr['linear_score']:.3f} vs neighbours="
                  f"{pr['neighbour_score']:.3f}")

        if self.preprocessing_['drop_columns']:
            print(f"\n{'-' * 74}\n  Before fitting: drop these columns\n{'-' * 74}")
            for c in self.preprocessing_['drop_columns']:
                print(f"  {c}")
                print(f"      {self.preprocessing_['drop_reason'][c]}")
        if self.preprocessing_['copula']:
            print("\n  Heavy tails detected: the recommendation is configured "
                  "with copula=True.")

        print(f"\n{'-' * 74}\n  Ranking\n{'-' * 74}")
        shown = [r for r in self.ranked_ if not r['vetoed']][:top]
        for i, r in enumerate(shown, 1):
            mark = '>>' if i == 1 else '  '
            print(f"{mark} {i}. {r['estimator']:<26} score {r['score']:+.1f}"
                  f"   ({r['label']})")
            for reason in r['reasons']:
                print(f"        {reason}")
            print()

        vetoed = [r for r in self.ranked_ if r['vetoed']]
        if vetoed:
            print("  Ruled out")
            for r in vetoed:
                print(f"    {r['estimator']:<26} {r['veto_reason']}")
            print()

        if self.notes_:
            print(f"{'-' * 74}\n  Notes\n{'-' * 74}")
            for nte in self.notes_:
                print(f"  * {nte}")
            print()

        print(f"{'-' * 74}\n  Recommended follow-up\n{'-' * 74}")
        for f in self.followups_:
            tag = 'required ' if f['required'] else 'suggested'
            print(f"  [{tag}] {f['tool']}")
            print(f"        {f['why']}")

        print(f"\n{line}")
        print("  This ranking is a heuristic triage from the evidence above,")
        print("  not a substitute for cross-validating the shortlist.")
        print(line)


def recommend_model(X, y, **kwargs) -> MissRecommender:
    """
    Fit a MissRecommender and return it.

    Convenience wrapper; see MissRecommender for the parameters.

        rec = recommend_model(X, y, feature_names=cols)
        model = rec.make_estimator().fit(X, y)
    """
    return MissRecommender(**kwargs).fit(X, y)
