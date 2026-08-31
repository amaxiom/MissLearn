"""
_explainer.py  --  SHAP-based explainability for MissLearn models.

MissShapley
    Low-level Shapley value engine that uses the FIML model as its own
    coalition value function.  Because FIML computes E[Y | X_obs] exactly
    via the joint MVN, setting non-coalition features to NaN and calling
    model.predict() is the theoretically exact coalition value -- no Monte
    Carlo background sampling needed.

    v(S | x_i) = model.predict(x_i with non-S features set to NaN)

    For p <= exact_threshold (default 15): exact Shapley via 2^p batch
    evaluation (one batched model.predict call per observation).
    For p > exact_threshold: KernelSHAP linear regression approximation.

MissExplainer
    High-level interface.  Provides:

    shap_values(X)       -- value SHAP:  how much does each observed feature
                            VALUE contribute to the prediction?

    miss_shap(X)         -- missingness SHAP:  how much does observing vs.
                            not observing each feature shift the prediction?

    plot_beeswarm(...)   -- viridis beeswarm (replaces SHAP's red-blue)
    plot_miss_importance -- bar chart of mean |miss_shap| per feature
    plot_waterfall(...)  -- per-observation breakdown
    plot_dependence(...) -- SHAP dependence with viridis feature coloring
    to_shap_explanation  -- wraps computed values in shap.Explanation for
                            compatibility with the SHAP library's own plots

Viridis colour convention
    Feature values       : viridis colormap (purple=low, yellow=high)
    Positive SHAP bars   : viridis yellow  (#FDE725)
    Negative SHAP bars   : viridis purple  (#440154)
    Missing (NaN) dots   : '#aaaaaa' (mid grey, SHAP convention)

    All plots accept show=False so users can customise further before
    calling plt.show() / fig.savefig().
"""

from __future__ import annotations

import copy
import warnings
from math import factorial
from typing import List, Optional, Tuple, Union

import numpy as np

# Viridis palette constants
_VIRIDIS_HI  = '#FDE725'   # yellow  -- positive / high value
_VIRIDIS_LO  = '#440154'   # purple  -- negative / low value
_VIRIDIS_MID = '#21908C'   # teal    -- neutral / zero
_NAN_GREY    = '#aaaaaa'   # missing values


# ---------------------------------------------------------------------------
# MissShapley  --  FIML coalition value function + Shapley computation
# ---------------------------------------------------------------------------

def _make_value_fn(model, output: str = 'auto', class_index=None):
    """
    Build the coalition value function v(x) used by the Shapley engine.

    This has to be a *continuous* quantity. Calling predict() on a classifier
    returns a hard class label, and Shapley values computed on hard labels are
    meaningless: on an imbalanced problem every coalition predicts the majority
    class, every difference is exactly zero, and the attributions silently
    vanish. On a balanced one they quantise to the class levels and describe
    label flips rather than any change in belief. Probabilities, or the
    log-odds of them, are the right value function for a classifier.

    Parameters
    ----------
    model : fitted estimator
    output : {'auto', 'proba', 'log-odds', 'raw'}, default 'auto'
        'auto' uses the predicted probability of the positive class when the
        model exposes predict_proba with two columns, and predict() otherwise,
        which is the correct choice for a regressor.
        'log-odds' is the additive scale SHAP conventionally uses for
        classifiers; it is better behaved when probabilities approach 0 or 1.
        'raw' forces predict(), which reproduces the pre-0.9.1 behaviour and is
        retained only for backwards comparison.

    Returns
    -------
    (fn, kind) where fn maps (n, p) -> (n,) and kind names the scale used.
    """
    if output not in ('auto', 'proba', 'log-odds', 'raw'):
        raise ValueError("output must be 'auto', 'proba', 'log-odds' or 'raw'; "
                         f"got {output!r}")

    has_proba = hasattr(model, 'predict_proba')

    if output == 'raw' or not has_proba:
        if output in ('proba', 'log-odds'):
            raise AttributeError(
                f"output={output!r} needs predict_proba, which "
                f"{type(model).__name__} does not provide. Use output='auto' "
                f"or 'raw' for a regressor.")
        return (lambda X: np.asarray(model.predict(X), dtype=float).ravel(),
                'prediction')

    def _proba(X):
        P = np.asarray(model.predict_proba(X), dtype=float)
        if P.ndim == 1:
            return P
        if class_index is not None:
            if not 0 <= class_index < P.shape[1]:
                raise ValueError(
                    f"class_index={class_index} is out of range for "
                    f"{P.shape[1]} classes.")
            return P[:, class_index]
        if P.shape[1] == 2:
            return P[:, 1]
        # Multi-class with no class named. Shapley values need a scalar value
        # function, and silently picking a column would attribute a class the
        # caller never asked about, so require the choice to be explicit.
        raise ValueError(
            f"predict_proba returned {P.shape[1]} classes, so there is no "
            f"single value function to attribute. Pass class_index=k to "
            f"explain the probability of class k, which is the meaningful "
            f"operation for a multi-class model, or output='raw' to attribute "
            f"the predicted label instead.")

    if output == 'log-odds':
        def _fn(X):
            q = np.clip(_proba(X), 1e-12, 1 - 1e-12)
            return np.log(q / (1.0 - q))
        return _fn, 'log-odds'

    return _proba, 'probability'


class MissShapley:
    """
    Shapley value engine using the FIML model as the exact coalition value
    function.

    Parameters
    ----------
    model : fitted MissLearn (or any) model with predict(X) -> ndarray
    exact_threshold : int, default 15
        Maximum p for exact 2^p Shapley.  Above this, KernelSHAP is used.
    n_kernel_samples : int, default 512
        Number of coalition samples for the KernelSHAP approximation.
    output : {'auto', 'proba', 'log-odds', 'raw'}, default 'auto'
        The scale on which coalitions are valued.  See _make_value_fn; the
        default uses predicted probability for classifiers and the prediction
        itself for regressors.  Attributing a classifier's hard label produces
        degenerate results and is available only as output='raw'.
    random_state : int or None, default None
    """

    def __init__(
        self,
        model,
        exact_threshold: int  = 15,
        n_kernel_samples: int = 512,
        output: str           = 'auto',
        class_index           = None,
        random_state          = None,
    ):
        self.model            = model
        self.exact_threshold  = exact_threshold
        self.n_kernel_samples = n_kernel_samples
        self.output           = output
        self.class_index      = class_index
        self.random_state     = random_state
        self._v, self.value_scale_ = _make_value_fn(model, output, class_index)

    # ------------------------------------------------------------------
    # Public: compute SHAP values
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: np.ndarray,
        baseline: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute Shapley values for each observation in X.

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        baseline : float, optional
            Expected prediction when all features are unknown (all-NaN row).
            Computed automatically if not supplied.

        Returns
        -------
        phi : ndarray of shape (n, p)
            phi[i, j] is the Shapley value of feature j for observation i.
            NaN-valued features in X receive phi = 0.

        Property:  sum_j phi[i, j]  =  model.predict(X[i]) - baseline
        """
        X = np.asarray(X, dtype=float)
        n, p = X.shape

        if baseline is None:
            baseline = self._baseline(p)

        if p <= self.exact_threshold:
            return self._exact_shap(X, baseline)
        else:
            return self._kernel_shap(X, baseline)

    # ------------------------------------------------------------------
    # Exact Shapley  (p <= exact_threshold)
    # ------------------------------------------------------------------

    def _baseline(self, p: int) -> float:
        """v(empty set) = model.predict(all-NaN row)."""
        return float(self._v(np.full((1, p), np.nan))[0])

    def _exact_shap(self, X: np.ndarray, baseline: float) -> np.ndarray:
        n, p = X.shape
        phi  = np.zeros((n, p))

        # Shapley weights: w(s, p) = s!(p-s-1)!/p!
        weights = np.zeros((p + 1, p + 1))
        fp = float(factorial(p))
        for s in range(p):
            weights[s, p] = factorial(s) * factorial(p - s - 1) / fp

        # Precompute bit-count lookup for speed
        popcount = np.array([bin(k).count('1') for k in range(2 ** p)])

        for i in range(n):
            x_i = X[i]
            obs  = ~np.isnan(x_i)

            # Build all 2^p masked rows in one vectorised operation
            S_range    = np.arange(2 ** p, dtype=np.int32)
            bit_matrix = ((S_range[:, None] >> np.arange(p, dtype=np.int32)[None, :]) & 1).astype(bool)
            # Where bit is set use x_i[j] (may be NaN for missing features), else NaN
            X_batch    = np.where(bit_matrix, x_i[None, :], np.nan)

            v = self._v(X_batch)              # shape (2^p,)

            # Accumulate Shapley contributions
            for j in range(p):
                if not obs[j]:      # feature j is NaN -- phi stays 0
                    continue
                bit_j = 1 << j
                for S_bits in range(2 ** p):
                    if (S_bits >> j) & 1:    # j already in S -- skip
                        continue
                    s = popcount[S_bits]
                    phi[i, j] += weights[s, p] * (v[S_bits | bit_j] - v[S_bits])

        return phi

    # ------------------------------------------------------------------
    # KernelSHAP approximation  (p > exact_threshold)
    # ------------------------------------------------------------------

    def _kernel_shap(self, X: np.ndarray, baseline: float) -> np.ndarray:
        """
        KernelSHAP: sample coalitions, evaluate FIML value function, fit
        a weighted linear regression to recover approximate Shapley values.

        Uses the SHAP kernel weight: w(S) = (p-1) / [C(p,|S|) * |S| * (p-|S|)]
        """
        from math import comb

        n, p = X.shape
        rng  = np.random.default_rng(self.random_state)
        phi  = np.zeros((n, p))

        # Sample M coalition masks (exclude all-zeros and all-ones)
        M   = self.n_kernel_samples
        masks = np.zeros((M, p), dtype=bool)
        wts   = np.zeros(M)

        idx = 0
        while idx < M:
            s = rng.integers(1, p)     # coalition size in {1, ..., p-1}
            chosen = rng.choice(p, size=s, replace=False)
            mask   = np.zeros(p, dtype=bool)
            mask[chosen] = True
            masks[idx] = mask
            # Sizes are sampled uniformly and subsets uniformly within size,
            # so q(S) ∝ 1/C(p,s).  The importance weight is π(S)/q(S)
            # ∝ (p-1)/(s(p-s)); including the kernel's 1/C(p,s) here again
            # would double-count it and crush mid-size coalitions.
            wts[idx] = (p - 1) / (s * (p - s)) if s * (p - s) > 0 else 1.0
            idx += 1

        for i in range(n):
            x_i  = X[i]
            obs  = ~np.isnan(x_i)

            # Build M masked rows
            X_batch = np.full((M, p), np.nan)
            for k in range(M):
                for j in range(p):
                    if masks[k, j] and obs[j]:
                        X_batch[k, j] = x_i[j]

            v = self._v(X_batch) - baseline              # shape (M,)

            # Weighted linear regression: mask @ phi = v (constrained)
            # Restrict to observed features only
            obs_idx  = np.where(obs)[0]
            if len(obs_idx) == 0:
                continue

            Z  = masks[:, obs_idx].astype(float)        # (M, |obs|)

            # WLS without forming the M×M diagonal weight matrix:
            # Z^T W Z = (Z * wts[:,None]).T @ Z
            Zwt   = Z * wts[:, None]   # (M, |obs|), row k scaled by wts[k]
            ZtWZ  = Zwt.T @ Z + 1e-8 * np.eye(len(obs_idx))
            ZtWv  = Zwt.T @ v

            try:
                phi_obs = np.linalg.solve(ZtWZ, ZtWv)
            except np.linalg.LinAlgError:
                phi_obs = np.linalg.lstsq(ZtWZ, ZtWv, rcond=None)[0]

            # Enforce efficiency: sum(phi) = f(x_i) - baseline.  Use the
            # constrained-WLS (Lagrangian) correction, shifting phi along
            # ZtWZ^{-1}·1; multiplicative rescaling explodes when the
            # unconstrained phis nearly cancel.
            f_xi    = float(self._v(x_i.reshape(1, -1))[0])
            target  = f_xi - baseline
            ones = np.ones(len(phi_obs))
            try:
                u = np.linalg.solve(ZtWZ, ones)
            except np.linalg.LinAlgError:
                u = ones
            denom = float(ones @ u)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                u, denom = ones, float(len(phi_obs))
            phi_obs = phi_obs + u * ((target - phi_obs.sum()) / denom)

            phi[i, obs_idx] = phi_obs

        return phi


# ---------------------------------------------------------------------------
# MissExplainer  --  high-level interface
# ---------------------------------------------------------------------------

class MissExplainer:
    """
    SHAP-based explainability for MissLearn models.

    Two complementary explanations:

    Value SHAP:  phi[i, j] = contribution of feature j's OBSERVED VALUE to
        observation i's prediction, relative to the all-missing baseline.
        Uses the FIML model as the exact coalition value function::

            v(S | x_i) = model.predict(x_i with non-S features as NaN)

        No Monte Carlo sampling required.

    Missingness SHAP:  phi_miss[i, j] = gain from OBSERVING feature j vs.
        leaving it unknown.  For each feature and observation::

            phi_miss[i,j] = model.predict(x_j known) - model.predict(x_j = NaN)

        where "known" uses the actual observed value if j is observed, or the
        conditional MVN mean E[X_j | X_i_obs] if j is already missing.
        Large ``|phi_miss[i,j]|`` means j carries a lot of predictive information
        for observation i.

    Parameters
    ----------
    model : fitted MissLearn (or sklearn) model
        Must implement predict(X: ndarray) -> ndarray.
        For classifiers, predict_proba is used instead where relevant.
    exact_threshold : int, default 15
        Use exact 2^p Shapley for p <= this value; KernelSHAP above.
    n_kernel_samples : int, default 512
    random_state : int or None

    output : {'auto', 'proba', 'log-odds', 'raw'}, default 'auto'
        The scale coalitions are valued on.  'auto' uses the predicted
        probability of the positive class for a binary classifier and the
        prediction itself for a regressor.  Attributing a classifier's hard
        label is degenerate (see _make_value_fn) and is available only as
        output='raw'.

    Attributes
    ----------
    expected_value_  : float -- baseline value (all features missing)
    value_scale_     : str -- 'probability', 'log-odds' or 'prediction'
    feature_names_   : list of str
    p_               : int -- number of features
    """

    def __init__(
        self,
        model,
        exact_threshold: int   = 15,
        n_kernel_samples: int  = 512,
        output: str            = 'auto',
        class_index            = None,
        random_state           = None,
    ):
        self.model            = model
        self.exact_threshold  = exact_threshold
        self.n_kernel_samples = n_kernel_samples
        self.output           = output
        self.class_index      = class_index
        self.random_state     = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'MissExplainer':
        """
        Prepare the explainer using the training data.

        Fits a joint MVN to X_train for the missingness SHAP conditional
        means, and computes the baseline prediction (all-NaN row).

        Parameters
        ----------
        X_train : ndarray of shape (n, p), may contain NaN
        feature_names : list of str, optional.  Defaults to ['X0', 'X1', ...].
        """
        from ._imputer import MissImputer

        X_train = np.asarray(X_train, dtype=float)
        n, p    = X_train.shape
        self.p_ = p

        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f'X{j}' for j in range(p)]
        )

        # The coalition value function. For a classifier this must be the
        # predicted probability, not the hard label: attributing labels makes
        # every coalition of an imbalanced problem identical and silently
        # zeroes every attribution.
        self._v, self.value_scale_ = _make_value_fn(
            self.model, self.output, self.class_index)

        # Baseline: value when all features are unknown
        self.expected_value_ = float(self._v(np.full((1, p), np.nan))[0])

        # Joint MVN for conditional means (missingness SHAP)
        self._mvn = MissImputer(m=1, random_state=self.random_state)
        self._mvn.fit(X_train)

        # SHAP engine, valued on the same scale
        self._shap_engine = MissShapley(
            self.model,
            exact_threshold  = self.exact_threshold,
            n_kernel_samples = self.n_kernel_samples,
            output           = self.output,
            class_index      = self.class_index,
            random_state     = self.random_state,
        )

        self._X_train = X_train
        return self

    # ------------------------------------------------------------------
    # Value SHAP
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: np.ndarray,
        method: str = 'auto',
    ) -> np.ndarray:
        """
        Compute value SHAP for each observation in X.

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN
        method : 'auto' | 'exact' | 'kernel'
            'auto' uses exact for p <= exact_threshold, else kernel.

        Returns
        -------
        phi : ndarray of shape (n, p)
            Sums to v(X[i]) - ``expected_value_`` for each row, where v is the
            coalition value function named by ``self.value_scale_``.
        """
        X  = np.asarray(X, dtype=float)
        p  = X.shape[1]

        if method == 'kernel':
            exact_threshold = 0
        elif method == 'exact':
            exact_threshold = p          # force the exact path at any p
        else:
            exact_threshold = self.exact_threshold

        # output must be forwarded. Omitting it silently valued every coalition
        # on the default scale regardless of what the explainer was configured
        # with, so the attributions did not match expected_value_.
        engine = MissShapley(
            self.model,
            exact_threshold  = exact_threshold,
            n_kernel_samples = self.n_kernel_samples,
            output           = self.output,
            class_index      = self.class_index,
            random_state     = self.random_state,
        )
        return engine.shap_values(X, baseline=self.expected_value_)

    # ------------------------------------------------------------------
    # Missingness SHAP
    # ------------------------------------------------------------------

    def miss_shap(self, X: np.ndarray) -> np.ndarray:
        """
        Compute missingness SHAP for each observation and feature.

        phi_miss[i, j] = gain from knowing X_j vs. leaving it missing.

        Positive: knowing X_j increases the prediction.
        Negative: knowing X_j decreases the prediction.
        Zero:     X_j carries no information for observation i.

        For observed X_j:  compares model.predict(actual x_j) vs. model.predict(X_j=NaN).
        For missing X_j:   compares model.predict(cond_mean) vs. model.predict(X_j=NaN).
        The conditional mean E[X_j | X_i_obs] comes from the joint MVN fitted
        during fit().

        Parameters
        ----------
        X : ndarray of shape (n, p), may contain NaN

        Returns
        -------
        phi_miss : ndarray of shape (n, p)
        """
        from ._utils import conditional_normal_params

        X        = np.asarray(X, dtype=float)
        n, p     = X.shape
        nan_mask = np.isnan(X)
        phi_miss = np.zeros((n, p))

        mu    = self._mvn.mu_
        Sigma = self._mvn.Sigma_

        for i in range(n):
            x_i     = X[i].copy()
            obs_idx = np.where(~nan_mask[i])[0]
            mis_idx = np.where( nan_mask[i])[0]

            # Build x_known: actual value for observed features;
            # conditional MVN mean for missing features.
            x_known = x_i.copy()
            for j in mis_idx:
                if len(obs_idx) == 0:
                    x_known[j] = mu[j]
                else:
                    mu_c, _ = conditional_normal_params(
                        mu, Sigma, obs_idx, np.array([j]), x_i[obs_idx]
                    )
                    x_known[j] = float(mu_c[0])

            # Build a (2*p, p) batch:
            #   row 2j   : x_i with feature j = x_known[j]  (j is "known")
            #   row 2j+1 : x_i with feature j = NaN          (j is "missing")
            # Use np.tile so each row starts as a copy of x_i.
            batch = np.tile(x_i, (2 * p, 1))
            for j in range(p):
                batch[2 * j,     j] = x_known[j]
                batch[2 * j + 1, j] = np.nan

            preds = self._v(batch)             # (2*p,) -- one call per obs
            for j in range(p):
                phi_miss[i, j] = preds[2 * j] - preds[2 * j + 1]

        return phi_miss

    # ------------------------------------------------------------------
    # SHAP library compatibility
    # ------------------------------------------------------------------

    def to_shap_explanation(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
    ):
        """
        Wrap computed SHAP values in a shap.Explanation object.

        Requires the 'shap' library to be installed.  The returned object
        can be passed to shap.plots.beeswarm(), shap.plots.waterfall(), etc.

        To apply viridis colours to shap library plots:
            import matplotlib.pyplot as plt
            shap.plots.beeswarm(explanation, color=plt.get_cmap('viridis'))

        Parameters
        ----------
        shap_values : ndarray (n, p)
        X : ndarray (n, p)

        Returns
        -------
        shap.Explanation
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "MissExplainer.to_shap_explanation requires the 'shap' library. "
                "Install it with: pip install shap"
            )
        return shap.Explanation(
            values          = shap_values,
            base_values     = np.full(X.shape[0], self.expected_value_),
            data            = X,
            feature_names   = self.feature_names_,
        )

    # ------------------------------------------------------------------
    # Plotting: beeswarm
    # ------------------------------------------------------------------

    def plot_beeswarm(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int     = 20,
        feature_names: Optional[List[str]] = None,
        title: str           = 'SHAP Value Beeswarm',
        figsize: Tuple       = (9, 6),
        show: bool           = True,
    ):
        """
        Beeswarm plot of SHAP values with viridis feature-value colouring.

        Each dot = one observation.  Position on x-axis = SHAP value.
        Colour = feature value (viridis: purple=low, yellow=high).
        Missing (NaN) feature values shown as grey dots.

        Parameters
        ----------
        shap_values : ndarray (n, p)
        X : ndarray (n, p)
        max_display : int -- maximum number of features to show (top by importance)
        feature_names : list of str, optional
        title : str
        figsize : tuple
        show : bool -- call plt.show() if True

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        X   = np.asarray(X, dtype=float)
        n, p = shap_values.shape
        names = feature_names or self.feature_names_

        # Sort features by mean |SHAP|
        importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1][:max_display]
        order = order[::-1]   # bottom-to-top for horizontal plot

        cmap = plt.colormaps['viridis']
        fig, ax = plt.subplots(figsize=figsize)

        for plot_row, feat_idx in enumerate(order):
            sv      = shap_values[:, feat_idx]
            fv      = X[:, feat_idx]
            nan_f   = np.isnan(fv)

            # Jitter dots along y-axis using binned density
            y_jitter = _beeswarm_jitter(sv, spread=0.35)
            y_base   = float(plot_row)

            # Colour by normalised feature value (viridis)
            if not nan_f.all():
                fv_obs  = fv[~nan_f]
                vmin, vmax = fv_obs.min(), fv_obs.max()
                norm    = mcolors.Normalize(
                    vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-10
                )
                colors = np.where(
                    nan_f,
                    np.nan,
                    norm(np.where(nan_f, vmin, fv))
                )
            else:
                colors = np.full(n, np.nan)

            # Plot NaN-feature dots in grey
            if nan_f.any():
                ax.scatter(
                    sv[nan_f], y_base + y_jitter[nan_f],
                    c=_NAN_GREY, s=16, alpha=0.7,
                    linewidths=0, zorder=3,
                )

            # Plot observed-feature dots in viridis
            if (~nan_f).any():
                obs_colors = cmap(colors[~nan_f])
                ax.scatter(
                    sv[~nan_f], y_base + y_jitter[~nan_f],
                    c=obs_colors, s=16, alpha=0.7,
                    linewidths=0, zorder=3,
                )

        ax.axvline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=10)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Colourbar for feature values
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
        cbar.set_label('Feature value', fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Plotting: missingness importance
    # ------------------------------------------------------------------

    def plot_miss_importance(
        self,
        miss_shap: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str   = 'Missingness SHAP: Information Cost of Not Observing Each Feature',
        figsize: Tuple = (8, 5),
        show: bool     = True,
    ):
        """
        Horizontal bar chart of mean ``|missingness SHAP|`` per feature.

        Bars are coloured by the viridis scale (most important = yellow,
        least = purple) to maintain consistent branding.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        names = feature_names or self.feature_names_
        p     = miss_shap.shape[1]
        imp   = np.abs(miss_shap).mean(axis=0)
        order = np.argsort(imp)   # ascending for horizontal bar

        cmap = plt.colormaps['viridis']
        norm = mcolors.Normalize(vmin=imp.min(), vmax=imp.max() + 1e-10)
        colors = [cmap(norm(imp[i])) for i in order]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(
            range(len(order)),
            imp[order],
            color=colors, edgecolor='white', linewidth=0.5,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=10)
        ax.set_xlabel('Mean |Missingness SHAP|  (avg. prediction shift from not observing)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Value labels on bars
        for bar, val in zip(bars, imp[order]):
            ax.text(
                val + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8.5, color='#333333',
            )

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Plotting: waterfall
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        i: int                     = 0,
        feature_names: Optional[List[str]] = None,
        max_display: int           = 12,
        title: str                 = 'SHAP Waterfall',
        figsize: Optional[Tuple]   = None,
        show: bool                 = True,
    ):
        """
        Waterfall plot for a single observation.

        Shows how each feature's SHAP value moves the prediction from
        the baseline (E[f(X)]) to the final prediction f(x_i).

        Positive bars: viridis yellow (#FDE725)
        Negative bars: viridis purple (#440154)

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        X      = np.asarray(X, dtype=float)
        names  = feature_names or self.feature_names_
        phi    = shap_values[i]        # shape (p,)
        x_i    = X[i]
        p      = len(phi)

        # Sort by |SHAP|, keep top max_display, group rest as "other"
        order      = np.argsort(np.abs(phi))[::-1]
        top_order  = order[:max_display]
        rest_idx   = order[max_display:]

        top_phi    = phi[top_order]
        rest_sum   = phi[rest_idx].sum() if len(rest_idx) > 0 else 0.0

        bar_values = list(top_phi) + ([rest_sum] if len(rest_idx) > 0 else [])
        bar_labels = [names[j] for j in top_order]
        if len(rest_idx) > 0:
            bar_labels.append(f'{len(rest_idx)} other features')
        bar_feature_vals = [x_i[j] for j in top_order] + \
                           ([np.nan] if len(rest_idx) > 0 else [])

        # Reverse for bottom-to-top display
        bar_values = bar_values[::-1]
        bar_labels = bar_labels[::-1]
        bar_feature_vals = bar_feature_vals[::-1]

        n_bars = len(bar_values)
        running = self.expected_value_
        starts  = []
        for v in bar_values:
            starts.append(running)
            running += v

        # Grow the canvas with the number of rows rather than squeezing them
        # into a fixed height, which is what made this plot hard to read.
        if figsize is None:
            figsize = (9.5, max(4.2, 0.52 * n_bars + 1.8))

        fig, ax = plt.subplots(figsize=figsize)

        for k in range(n_bars):
            v     = bar_values[k]
            s     = starts[k]
            color = _VIRIDIS_HI if v >= 0 else _VIRIDIS_LO
            ax.barh(k, v, left=s, color=color, alpha=0.88,
                    edgecolor='white', linewidth=0.5, height=0.6)

        # The contribution is annotated on the bar; the feature name and its
        # value go on the y axis. Printing the name in both places, as an
        # earlier version did, doubled the text on every row and was the main
        # source of crowding once p grew beyond a handful of features.
        span = max(abs(running - self.expected_value_),
                   max(abs(v) for v in bar_values) if n_bars else 1.0)
        pad  = 0.03 * span
        for k in range(n_bars):
            v, s = bar_values[k], starts[k]
            fv = bar_feature_vals[k]
            # A feature that is unobserved in this row gets a value attribution
            # of exactly zero under the exact coalition semantics, because the
            # model marginalises it in every coalition. Left unannotated it
            # renders as an empty row that reads as a plotting failure, so it
            # is marked explicitly and its information is attributed in
            # miss_shap() instead.
            if not np.isfinite(fv) and v == 0.0:
                ax.plot([s], [k], marker='|', markersize=11,
                        color=_NAN_GREY, markeredgewidth=1.6)
                ax.text(s + pad, k, 'not observed',
                        va='center', ha='left', fontsize=9.5,
                        color='#777777', style='italic')
                continue
            ax.text(s + v + pad * np.sign(v) if v != 0 else s + pad, k,
                    f'{v:+.3f}',
                    va='center', ha='left' if v >= 0 else 'right',
                    fontsize=10, color='#111111')

        # Baseline and prediction markers
        ax.axvline(self.expected_value_, color='#888888', linestyle=':', linewidth=1.2,
                   label=f'Baseline = {self.expected_value_:.3f}')
        ax.axvline(running, color=_VIRIDIS_MID, linestyle='--', linewidth=1.5,
                   label=f'Prediction = {running:.3f}')

        # Feature value on the tick label, which is the usual waterfall
        # convention and keeps the plotting area free of duplicated text.
        tick_labels = []
        for k in range(n_bars):
            fv = bar_feature_vals[k]
            if np.isfinite(fv):
                tick_labels.append(f'{bar_labels[k]} = {fv:.2f}')
            else:
                # Either the grouped remainder, or an unobserved feature whose
                # status is already annotated in the plot as "not observed".
                tick_labels.append(bar_labels[k])

        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(tick_labels, fontsize=10.5)
        ax.set_ylim(-0.7, n_bars - 0.3)
        ax.set_xlabel('Prediction', fontsize=12)
        ax.set_title(f'{title}  (observation {i})', fontsize=13)

        # Leave room for the annotations so they are not clipped at the edges.
        lo = min(min(starts), self.expected_value_, running)
        hi = max(max(s + v for s, v in zip(starts, bar_values)),
                 self.expected_value_, running)
        margin = 0.16 * max(hi - lo, 1e-12)
        ax.set_xlim(lo - margin, hi + margin)

        # Below the axes, so it can never sit on top of a bar. An in-axes
        # legend collided with the lowest-ranked features whenever their
        # contributions were small, which is the common case.
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                  ncol=2, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.25)
        ax.set_axisbelow(True)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Plotting: dependence
    # ------------------------------------------------------------------

    def plot_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_idx: int,
        interaction_idx: Optional[int]     = None,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple = (7, 5),
        show: bool     = True,
    ):
        """
        SHAP dependence plot for feature *feature_idx*.

        X-axis: feature value.
        Y-axis: SHAP value.
        Colour: interaction feature value (viridis), or viridis by the
        feature's own value if interaction_idx is None.

        Missing values in either axis are shown as grey dots.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        X       = np.asarray(X, dtype=float)
        names   = feature_names or self.feature_names_
        j       = feature_idx
        sv_j    = shap_values[:, j]
        fv_j    = X[:, j]

        color_feat = j if interaction_idx is None else interaction_idx
        fv_c       = X[:, color_feat]

        nan_x  = np.isnan(fv_j)
        nan_c  = np.isnan(fv_c)
        grey   = nan_x | nan_c

        cmap   = plt.colormaps['viridis']
        fv_obs = fv_c[~grey]
        if len(fv_obs) > 0:
            vmin, vmax = fv_obs.min(), fv_obs.max()
        else:
            vmin, vmax = 0.0, 1.0
        norm   = mcolors.Normalize(vmin=vmin, vmax=vmax + 1e-10)

        fig, ax = plt.subplots(figsize=figsize)

        if grey.any():
            ax.scatter(fv_j[grey], sv_j[grey], c=_NAN_GREY,
                       s=20, alpha=0.6, linewidths=0, label='NaN', zorder=2)
        if (~grey).any():
            sc = ax.scatter(
                fv_j[~grey], sv_j[~grey],
                c=cmap(norm(fv_c[~grey])),
                s=20, alpha=0.75, linewidths=0, zorder=3,
            )
            # Colourbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
            cbar.set_label(names[color_feat], fontsize=9)

        ax.axhline(0, color='#888888', linestyle='--', linewidth=0.8)
        ax.set_xlabel(names[j], fontsize=11)
        ax.set_ylabel(f'SHAP value for {names[j]}', fontsize=11)
        ax.set_title(f'SHAP Dependence: {names[j]}', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        shap_values: Optional[np.ndarray] = None,
        miss_shap: Optional[np.ndarray]   = None,
    ) -> None:
        W     = 70
        names = self.feature_names_
        print()
        print('=' * W)
        print(f"{'MissExplainer  --  FIML SHAP Explainability':^{W}}")
        print('=' * W)
        print(f"  Model         : {type(self.model).__name__}")
        print(f"  Features (p)  : {self.p_}")
        print(f"  Baseline (E[f]): {self.expected_value_:.4f}")
        print(f"  Exact SHAP if : p <= {self.exact_threshold}  "
              f"({'exact' if self.p_ <= self.exact_threshold else 'kernel'})")

        if shap_values is not None:
            importance = np.abs(shap_values).mean(axis=0)
            order      = np.argsort(importance)[::-1]
            print()
            print("  Value SHAP -- Feature Importance (mean |SHAP|):")
            bar_w = 24
            for rank, j in enumerate(order):
                v      = importance[j]
                filled = int(round(v / (importance.max() + 1e-10) * bar_w))
                bar    = '#' * filled + ' ' * (bar_w - filled)
                print(f"    {names[j]:<12} [{bar}] {v:.4f}")

        if miss_shap is not None:
            miss_imp = np.abs(miss_shap).mean(axis=0)
            order_m  = np.argsort(miss_imp)[::-1]
            print()
            print("  Missingness SHAP -- Cost of Not Observing Each Feature:")
            for j in order_m:
                v      = miss_imp[j]
                filled = int(round(v / (miss_imp.max() + 1e-10) * 24))
                bar    = '#' * filled + ' ' * (24 - filled)
                print(f"    {names[j]:<12} [{bar}] {v:.4f}")

        print('=' * W)
        print()

    def __repr__(self) -> str:
        return (f"MissExplainer(model={type(self.model).__name__}, "
                f"exact_threshold={self.exact_threshold})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beeswarm_jitter(values: np.ndarray, spread: float = 0.35) -> np.ndarray:
    """
    Compute y-axis jitter for a beeswarm plot.

    Uses simple binned counting: points in the same x-bin are spread
    symmetrically along y within [-spread, +spread].

    Parameters
    ----------
    values : ndarray (n,) -- SHAP values (x-axis positions)
    spread : float -- half-width of jitter band

    Returns
    -------
    jitter : ndarray (n,)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    jitter = np.zeros(n)
    n_bins = max(10, n // 5)
    bins   = np.linspace(values.min() - 1e-10, values.max() + 1e-10, n_bins + 1)
    for b in range(n_bins):
        mask = (values >= bins[b]) & (values < bins[b + 1])
        k    = mask.sum()
        if k == 0:
            continue
        if k == 1:
            continue  # jitter stays 0
        # Evenly spread across [-spread, +spread]
        positions = np.linspace(-spread, spread, k)
        idx       = np.where(mask)[0]
        # Sort by value within bin for consistent ordering
        local_order = np.argsort(values[idx])
        jitter[idx[local_order]] = positions

    return jitter
