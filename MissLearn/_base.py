"""
MissLearn._base
------------
Abstract base class providing shared state management, missingness
reporting, confidence intervals, feature importance, and summary
formatting for both MissLinear and MissLogistic.
"""


import numpy as np

from ._conformance import check_common_parameters
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def only_for(task_kind):
    """Hide a task-specific method on a dispatcher that resolved the other way.

    The dispatchers pick a regressor or a classifier from ``y`` at fit time,
    and carry the union of both interfaces. ``predict_proba`` and
    ``decision_function`` therefore existed on a dispatcher fitted for
    regression and raised AttributeError only when *called*, which is too
    late: ``hasattr`` reports True, so scikit-learn calls the method and gets
    an exception instead of skipping it. Four checks failed that way on each
    of six estimators.

    Used with scikit-learn's own ``available_if``, so the attribute lookup
    fails and ``hasattr`` is False, which is what every caller actually
    tests. Before a fit the task is unknown, and the method stays visible
    rather than being hidden on a guess.

    Parameters
    ----------
    task_kind : {'classification', 'regression'}
        The task this method belongs to.
    """
    def predicate(estimator):
        task = getattr(estimator, 'task_', None)
        return task is None or task == task_kind
    return predicate


class MissTags(object):
    """The scikit-learn tag declarations every estimator in this package needs.

    A mixin rather than a method on :class:`MissBase`, because the task
    dispatchers (MissRidge, MissBayes, MissSupport and the rest) subclass
    ``BaseEstimator`` directly and so inherited none of it. They therefore
    declared ``allow_nan=False`` while the concrete classes they delegate to
    declared ``True``: six estimators announcing that they reject the input
    they exist to accept. ``check_estimators_nan_inf`` failed all six for it,
    and any third-party utility consulting the tag would have refused data
    they handle correctly.

    Defining it in one place and mixing it into both is the same rule the
    rest of the package follows: shared behaviour lives at one site, because
    a copy in seven class bodies is how the divergence happened.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags

    def _more_tags(self):
        """The same declarations for scikit-learn < 1.6.

        The package supports scikit-learn from 1.1 and the tag mechanism
        changed in 1.6, so both are declared and the statement stays true
        across the whole supported range rather than only on the
        maintainer's machine.

        This has to sit in the same class body as ``__sklearn_tags__``.
        scikit-learn's ``check_estimator_tags_renamed`` walks the MRO and
        objects to any class whose ``vars()`` holds one without the other,
        on the reasoning that the old name alone means the estimator was
        never migrated. Splitting them across the mixin and MissBase failed
        that check on fourteen estimators at once.
        """
        return {'allow_nan': True, 'sparse': False}


class MissBase(MissTags, BaseEstimator):
    """
    Shared base for FIML-based models with missing-data support.

    Subclasses must set after fit:
        coef_      : ndarray, shape (p,)
        intercept_ : float
        se_        : ndarray, shape (p+1,)  -- [intercept, *coef]
        coef_std_  : ndarray, shape (p,), standardized coefficients
        loglik_    : float
        aic_, bic_ : float
    """

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #

    def _validate_and_convert(self, X, y=None):
        """
        Convert inputs to float64 numpy arrays while permitting NaN.
        Captures column names from pandas DataFrames if present.
        Does NOT call sklearn's check_array because that rejects NaN.
        """
        if hasattr(X, 'columns'):
            self._feature_names_from_data = list(X.columns)
        else:
            self._feature_names_from_data = None

        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = np.array(y, dtype=np.float64).ravel()
            return X, y
        return X

    def _canonical_fit_order_with_groups(self, X, y, groups):
        """Reorder X, y and groups together.

        The mixed-effects models were left out of the plain reordering
        because a third per-row array travels with the data, and permuting
        two of three would attach observations to the wrong subject. That
        exclusion was the only remaining gap: the conformance suite's
        determinism check found MissMixedRegressor giving predictions that
        moved by up to 6.3e-02 when the rows were reordered, while every
        other estimator was exact to the bit.

        groups joins the sort key rather than only being carried along, so
        that two rows identical in X and y but belonging to different
        subjects still have a defined order.
        """
        from ._utils import canonical_row_order

        Xa = np.asarray(X, dtype=float)
        ga = np.asarray(groups)
        try:
            ya = np.asarray(y, dtype=float).reshape(len(Xa), -1)
        except (TypeError, ValueError):
            ya = np.zeros((len(Xa), 0))
        # Group labels may be strings; rank them so they can join the key
        # without assuming they are numbers.
        _, gcode = np.unique(ga, return_inverse=True)
        key = np.column_stack([ya, gcode.reshape(-1, 1).astype(float), Xa])
        order = canonical_row_order(key)
        return Xa[order], np.asarray(y)[order], ga[order]

    def _canonical_fit_order(self, X, y=None):
        """Reorder rows so a fit cannot depend on the order they arrived in.

        Every accumulation in these models is a floating-point sum, and
        floating-point addition is not associative, so two orderings of the
        same rows give objective values differing in the last bits. On a flat
        or ill-conditioned surface the optimiser turns that into different
        stopping points: a fourteen-row design reached log-likelihoods
        differing in the sixth significant figure and predictions differing
        by 0.186. Sorting first removes the cause instead of widening a
        tolerance against the symptom.

        Nothing is unsorted afterwards. A fit produces parameters, not
        per-row output, and prediction is row-wise already.

        Returns the reordered arrays; y is passed through untouched when it
        cannot be read as numbers, which leaves rows that are identical in X
        but differ in y ordered as supplied. Those are the only rows whose
        relative order this does not pin down.
        """
        from ._utils import canonical_row_order

        Xa = np.asarray(X, dtype=float)
        if y is None:
            return Xa[canonical_row_order(Xa)], None
        try:
            ya = np.asarray(y, dtype=float)
            key = np.column_stack([ya.reshape(len(ya), -1), Xa])
        except (TypeError, ValueError):
            key = Xa
        order = canonical_row_order(key)
        return Xa[order], np.asarray(y)[order]

    def _store_fit_metadata(self, X, y):
        """Record shape, feature names, and missingness statistics at fit time."""
        # Checked here rather than in each estimator because this is the one
        # hook every fit path already runs through, and these parameters are
        # spread across eleven classes. Putting the guard where a parameter
        # is consumed rather than where it is declared is the rule this
        # package learned the hard way: a check written into the body of a
        # thin subclass that has no fit of its own never runs at all.
        check_common_parameters(self)
        n, p = X.shape
        self.n_samples_fit_ = n
        self.n_features_in_ = p

        if self._feature_names_from_data is not None:
            self.feature_names_in_ = np.array(self._feature_names_from_data)
        else:
            self.feature_names_in_ = np.array([f'X{i}' for i in range(p)])

        self.n_missing_X_ = int(np.isnan(X).sum())
        self.n_missing_y_ = int(np.isnan(y).sum())
        self.missing_rate_X_ = (
            self.n_missing_X_ / (n * p) if n * p > 0 else 0.0
        )

        complete_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        self.n_complete_ = int(complete_mask.sum())
        self.n_partial_ = n - self.n_complete_

    # ------------------------------------------------------------------ #
    # sklearn interface
    # ------------------------------------------------------------------ #

    def get_feature_names_out(self):
        """Return feature names (sklearn API)."""
        check_is_fitted(self)
        return self.feature_names_in_.copy()

    # ------------------------------------------------------------------ #
    # Missingness report
    # ------------------------------------------------------------------ #

    def missingness_report(self):
        """
        Print a structured summary of missing data in the training set.
        Shows overall counts and, if available, per-feature missing rates.
        """
        check_is_fitted(self)
        rate_pct = self.missing_rate_X_ * 100
        print()
        print('Missingness Report')
        print('=' * 52)
        print(f'  Observations       : {self.n_samples_fit_}')
        print(f'  Complete cases     : {self.n_complete_}')
        print(f'  Partial cases      : {self.n_partial_}')
        print(f'  Missing X entries  : {self.n_missing_X_} ({rate_pct:.1f}%)')
        if self.n_missing_y_ > 0:
            ry = self.n_missing_y_ / self.n_samples_fit_ * 100
            print(f'  Missing y entries  : {self.n_missing_y_} ({ry:.1f}%)')
        print('=' * 52)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def conf_int(self, alpha=0.05):
        """
        Asymptotic confidence intervals using the normal approximation.

        coef_i +/- z_{alpha/2} * se_i

        Parameters
        ----------
        alpha : float, significance level in (0, 1); default 0.05 for a 95%
            interval. Values outside that range were previously accepted and
            produced intervals that are not intervals: alpha=1.5 gave a mean
            width of -0.029, so the lower bound sat above the upper one, and
            alpha=0 gave infinite width.

        Returns
        -------
        ci : ndarray, shape (p+1, 2)
            Row 0 is the intercept; rows 1..p are feature coefficients.
            Columns are [lower, upper].
        """
        check_is_fitted(self)
        try:
            a = float(alpha)
        except (TypeError, ValueError):
            raise ValueError(
                "alpha must be a number in (0, 1), got %r." % (alpha,))
        if not (np.isfinite(a) and 0.0 < a < 1.0):
            raise ValueError(
                "alpha must be in (0, 1), got %r. It is the significance "
                "level, so 0.05 gives a 95%% interval; outside that range "
                "the bounds are not an interval." % (alpha,))
        z = norm.ppf(1.0 - alpha / 2.0)
        coefs = np.concatenate([[self.intercept_], self.coef_])
        lower = coefs - z * self.se_
        upper = coefs + z * self.se_
        return np.column_stack([lower, upper])

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #

    @property
    def feature_importances_(self):
        """
        Normalized absolute standardized coefficients, excluding the intercept.

        Standardized coefficients reflect the change in outcome per one
        standard deviation change in each predictor, making coefficients
        on different scales comparable.  Values sum to 1.0.

        Subclasses must set self.coef_std_ during fit.
        """
        check_is_fitted(self)
        abs_std = np.abs(self.coef_std_)
        total = abs_std.sum()
        return abs_std / total if total > 0 else abs_std

    # ------------------------------------------------------------------ #
    # Internal helpers for summary formatting
    # ------------------------------------------------------------------ #

    def _pvalues_from_zstat(self, z_stat):
        """Two-sided p-values from z-statistics (standard normal)."""
        return 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))

    @staticmethod
    def _stars(p):
        """Return significance star string for a p-value."""
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        if p < 0.1:
            return '.'
        return ''

    @staticmethod
    def _bar(fraction, width=24):
        """ASCII progress bar scaled to fraction in [0, 1]."""
        filled = int(round(fraction * width))
        return '[' + '#' * filled + ' ' * (width - filled) + ']'

    def _coef_table_lines(self, stat_label='z_stat', alpha=0.05,
                          extra_header='', extra_rows=None):
        """
        Build and return a list of formatted lines for the coefficient table.
        Used by both MissLinear.summary() and MissLogistic.summary().

        Parameters
        ----------
        stat_label  : str, label for the test statistic column
        alpha       : float, CI significance level
        extra_header: str, optional additional column header text
        extra_rows  : list of str or None, one extra string per coefficient row

        Returns
        -------
        list of str
        """
        check_is_fitted(self)
        ci = self.conf_int(alpha=alpha)
        coefs = np.concatenate([[self.intercept_], self.coef_])
        names = ['intercept'] + list(self.feature_names_in_)
        z_stats = np.where(
            self.se_ > 0, coefs / self.se_, np.full_like(coefs, np.nan)
        )
        pvals = self._pvalues_from_zstat(z_stats)

        # When no standard errors were computed at all, every column derived
        # from them is nan, and MissLASSO does this by default. A table
        # reading "nan nan nan nan" on every row looks like a fit that
        # failed rather than inference that was not requested, so print the
        # estimates on their own and say why the rest is absent.
        if not np.any(np.isfinite(self.se_)):
            header = (f"{'':>14}  {'coef':>10}"
                      + ('  ' + extra_header if extra_header else ''))
            sep = '-' * max(len(header), 26)
            lines = [header, sep]
            for i, name in enumerate(names):
                extra = ('  ' + extra_rows[i]) if extra_rows else ''
                lines.append(f"{name:>14}  {coefs[i]:>10.4f}" + extra)
            lines.append(sep)
            lines.append("Standard errors, p-values and confidence intervals "
                         "were not computed for this fit.")
            return lines

        header = (
            f"{'':>14}  {'coef':>10}  {'std_err':>10}  "
            f"{stat_label:>10}  {'p_value':>10}  "
            f"{'CI_lower':>10}  {'CI_upper':>10}  {'sig':>3}"
            + ('  ' + extra_header if extra_header else '')
        )
        sep = '-' * len(header)
        lines = [header, sep]

        for i, name in enumerate(names):
            extra = ('  ' + extra_rows[i]) if extra_rows else ''
            lines.append(
                f"{name:>14}  {coefs[i]:>10.4f}  {self.se_[i]:>10.4f}  "
                f"{z_stats[i]:>10.3f}  {pvals[i]:>10.3e}  "
                f"{ci[i, 0]:>10.4f}  {ci[i, 1]:>10.4f}  "
                f"{self._stars(pvals[i]):>3}"
                + extra
            )

        lines.append(sep)
        lines.append(
            "Significance: '***' p<0.001  '**' p<0.01  '*' p<0.05  '.' p<0.1"
        )
        return lines

    def _importance_lines(self):
        """Return formatted feature importance lines for summary output."""
        check_is_fitted(self)
        importances = self.feature_importances_
        lines = []
        order = np.argsort(importances)[::-1]
        for i in order:
            name = self.feature_names_in_[i]
            bar = self._bar(importances[i])
            lines.append(f"  {name:>14}: {bar}  {importances[i] * 100:5.1f}%")
        return lines

    # ------------------------------------------------------------------ #
    # sklearn metadata routing (sklearn >= 1.3)
    # ------------------------------------------------------------------ #

    def get_metadata_routing(self):
        """Return the sklearn metadata routing object (scikit-learn >= 1.3).

        Enables MissLearn models to participate in sklearn pipelines and
        GridSearchCV with metadata (``sample_weight``, ``groups``) forwarded
        correctly.

        This defers to ``BaseEstimator`` rather than building a request of its
        own. It used to return a freshly constructed, and therefore empty,
        ``MetadataRequest``, which broke the very routing it was written to
        enable: scikit-learn generates a ``set_fit_request`` for every
        estimator whose ``fit`` takes metadata, that call recorded the request
        correctly, and this method then handed the router an empty object that
        knew nothing about it. A ``MissMixedRegressor`` with
        ``set_fit_request(groups=True)`` reported ``{}`` where scikit-learn's
        own implementation reports ``{'fit': {'groups': True}}``, and a
        Pipeline given ``groups`` refused them as "not routed to any object".
        Since a random intercept fitted without its groups collapses to an
        ordinary regression, tau 2.71 against 0.00 on the same data, silently
        dropping them would have been worse still.

        The version check is kept: the package supports scikit-learn from 1.1,
        and routing arrived in 1.3, so on an older release this reports what
        is missing rather than raising ``AttributeError`` from ``super()``.
        """
        try:
            from sklearn.utils.metadata_routing import MetadataRequest  # noqa: F401
        except ImportError:
            raise NotImplementedError(
                "get_metadata_routing requires scikit-learn >= 1.3; "
                "found an older version.  Upgrade with: "
                "pip install 'scikit-learn>=1.3'"
            )
        return super().get_metadata_routing()

    # set_fit_request and set_predict_request are deliberately NOT defined
    # here. scikit-learn generates them from the fit and predict signatures,
    # and from 1.7 it skips an estimator that already has the attribute in its
    # MRO. Two no-op stubs used to sit at this point and were harmless on 1.6,
    # where the generated descriptor still won. On 1.7 and later they won
    # instead, so MissMixedRegressor().set_fit_request(groups=True) recorded
    # nothing, and a Pipeline or cross_validate given groups either refused the
    # fit or would have dropped them, which turns a random-intercept model into
    # an ordinary regression without saying so.
    #
    # With nothing defined here, an estimator whose fit takes metadata gets
    # scikit-learn's real method, and one that takes none has no such method,
    # which is what scikit-learn's own estimators do: LinearRegression has it
    # for sample_weight, PCA does not have it at all.

    # ------------------------------------------------------------------ #
    # scikit-learn estimator tags: see MissTags
    # ------------------------------------------------------------------ #
