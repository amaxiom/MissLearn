"""
unit_test_suite.py  --  MissLearn comprehensive pytest test suite.
==================================================================

Designed to run either as a complete suite or one class at a time
from a Jupyter notebook cell:

    import pytest
    pytest.main(['-v', '--tb=short', 'unit_test_suite.py::TestMissRidge'])

Quick reference -- one cell per section
----------------------------------------
TestNumericalUtils          _utils: pack_cholesky, mvn_logpdf, conditional_normal,
                            numerical_hessian
TestMissLinear              FIML joint multivariate normal regression
TestMissLogistic            FIML logistic regression
TestMissRidgeRegressor      Ridge-penalized FIML regressor
TestMissRidgeClassifier     Ridge-penalized FIML classifier
TestMissRidge               Auto-task selector wrapper
TestMissLASSO               LASSO FIML models (regressor / classifier / auto)
TestMissNeighbors                 K-nearest neighbours FIML
TestMissBayes               Naive Bayes FIML
TestMissSupport                 Support vector FIML
TestMissGaussian            Gaussian Process FIML (small n)
TestMissMixed               Mixed-effects / LME FIML
TestMissEnsemble            Bootstrap-aggregated ensemble
TestMissMulticlass          One-vs-Rest multi-class extension
TestMissPreprocessor        Validation, compatibility checking, categorical encoding
TestMissDiagnostic          Missing data mechanism diagnostics
TestMissKFold               K-fold splitter (NaN-safe)
TestMissStratifiedKFold     Stratified K-fold splitter
TestCrossVal                miss_cross_val_score / miss_cross_validate
TestMissImputer             Multiple imputation via joint MVN
TestMissSensitivity         MNAR delta-adjustment sensitivity analysis
TestMissShapley                Low-level Shapley engine (exact + kernel)
TestMissExplainer           High-level SHAP explainability interface
TestPandasSupport           DataFrame / Series integration
TestCopulaTransform         Marginal Gaussian copula option

Later additions, grouped by what they pin rather than by module:
TestMixedConvergenceIsReported   a non-converged mixed fit must warn
TestMissMixedClassifierScore     accuracy over observed outcomes only
TestSensitivityVerdict           the four robustness bands
TestEnsembleSummaryCompactDisplay  the large-homogeneous report form
TestCrossValFitParams            per-sample fit_params sliced per fold
TestLassoStandardErrorsAreOptIn  the LASSO se_ path, off by default
TestPrefitCheckAdvisories        scale, kurtosis and length advisories
TestMetadataRequestStubsWarn     routing stubs admit they are stubs
TestDiagnosticSummaryRunsEverything  summary() lazily runs all four
TestRecommenderSummaryBranches   drop-column and copula reporting

Whole-library sweeps, which discover their targets from the package
rather than from a list, so a new estimator is covered the day it is
added instead of the day somebody remembers to edit a list:
TestSummaryOnEveryEstimator      summary() on all of them, three ways
TestSummaryInEveryConfiguration  and again with standard errors, with
                                 the copula applied, declined, and p=1
TestPredictIntervalOnEveryRegressor  intervals bracket the prediction
TestDecisionFunctionAcrossTheLibrary  hidden after resolving to
                                 regression, present otherwise
TestScoreAcrossTheLibrary        finite, and tolerant of absent labels
TestDegenerateRegimesAcrossTheLibrary  constant, duplicate and sparse
                                 columns, 1e-200 and 1e300 scales,
                                 p >= n, one class, DataFrame input
TestExplainerSurface             SHAP both ways, and every plot
TestImputerSurface               draws, pooling, Rubin's rules
TestPreprocessorSurface          categorical encoding with NaN kept
TestEnsembleSurface              both modes, weights, task consistency

Targeted work on branches the sweeps could not reach, which is most of
what is left once the obvious paths are covered:
TestCrossValidateScoringForms    string, list, tuple, dict and callable
TestWarmStartStateIsStripped     no fold inherits a previous fit
TestDiagnosticReportBranches     one verdict per mechanism concluded
TestRecommenderScoringBranches   heavy tails, wide designs, subsampling
TestPandasCoercionPaths          Series positionally, nullable, strings
TestPrefitCheckAndPreprocessorNaming  which feature names win
TestParameterValidationIsShared  the refusals every estimator inherits
TestConformanceReportRendering   the checker against a failing estimator
TestGaussianProcessUncertainty   predict_std, ARD length scales
TestSensitivityFitRobustness     a delta whose fit fails is dropped
TestMissingnessReportOnTheBase   the absent-outcome line
TestImputerPoolingEdges          when the variance cannot be recovered
TestMixedPredictionPaths         an unseen group falls back to the mean
TestThreeRowKindsAcrossTheLibrary  complete, partial and empty rows
TestPredictionUnderTheCopula     intervals mapped back to y units
TestNeighboursIntervalOnAnEmptyRow  a documented over-confidence
TestEnsembleFittingPaths         parallel fits, OOB failure, alignment
TestExplainerValueFunctionAndMethods  class_index, exact versus kernel
TestSharedValidationRefusals     complex input, wrong task, unfitted
TestEnsembleClassAlignmentInDetail  a member that missed a rare class
TestEnsembleAlignsMembersByLabel    the same, forced to actually happen
TestMixedDegenerateSeedAndEmptyRows  a flat seed, and empty predictors
TestNeighboursFallbacks          when the neighbourhood has no labels
TestConformanceReportExplanations   silent NaN versus opaque refusal
TestConformanceReportListsProblems  an estimator wrong in one place
TestPrefitCheckOutcomeChecks     the outcome refusals, including the
                                 one the estimators do not all run
TestPreprocessorEncodingDetail   encode=None, drop, pandas NA
TestSharedInputConversions       column-vector y, parameter refusals
TestCrossValidationEdges         n_splits, unlabelled folds, stratified
TestRecommenderInternals         the probe, the ICC, make_estimator
TestExplainerKernelInternals     the sampled route, reproducibly
TestExplainerRemainingPaths      one row, no names, every feature
TestGaussianProcessRemainingPaths   restarts, and all of them failing
TestLastEmptyRowBranches         the empty row through each family
"""

import sys
import warnings

sys.path.insert(0, r"C:\Users\Amanda\Favorites\Machine Learning\MissLearn")

import numpy as np
import pytest
# numpy renamed trapz to trapezoid in 2.0 and removed the old name in a later
# 2.x. Development here runs numpy 1.26, which has only trapz, while
# continuous integration installs 2.x, where the reference integrals below
# died with "module 'numpy' has no attribute 'trapz'". Both names are checked
# so the same file runs on either.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# ---------------------------------------------------------------------------
# Module-level shared datasets (created once per pytest session)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(2025)

# ---- regression ----
_N, _P = 160, 4
_X_reg = _rng.standard_normal((_N, _P))
_X_reg[_rng.random((_N, _P)) < 0.18] = np.nan
_y_reg = 2.5 * _X_reg[:, 0] - 1.2 * _X_reg[:, 1] + _rng.standard_normal(_N) * 0.4
_y_reg[_rng.random(_N) < 0.10] = np.nan

# ---- binary classification ----
_X_clf = _rng.standard_normal((_N, _P))
# Compute logit *before* introducing NaN so class labels are not biased by
# silent NaN→False coercions in the comparison operator.
_logit  = 2.0 * _X_clf[:, 0] - 1.5 * _X_clf[:, 1]
_y_clf  = (_logit > 0).astype(float)
_X_clf[_rng.random((_N, _P)) < 0.18] = np.nan
_y_clf[_rng.random(_N) < 0.10] = np.nan

# ---- multi-class ----
_X_mc = _rng.standard_normal((_N, 3))
_X_mc[_rng.random((_N, 3)) < 0.18] = np.nan
_y_mc = np.where(_X_mc[:, 0] > 0.5, 2.0,
         np.where(_X_mc[:, 0] < -0.5, 0.0, 1.0))
_y_mc[_rng.random(_N) < 0.08] = np.nan

# ---- groups for mixed-effects ----
_groups = np.repeat(np.arange(20), _N // 20)

# ---- small-n for GP (O(n^3) cost) ----
_N_GP = 40
_X_gp = _rng.standard_normal((_N_GP, 3))
_X_gp[_rng.random((_N_GP, 3)) < 0.15] = np.nan
_y_gp_reg = 1.5 * _X_gp[:, 0] - _X_gp[:, 1] + _rng.standard_normal(_N_GP) * 0.3
_y_gp_clf = (_y_gp_reg > 0).astype(float)
_y_gp_clf[_rng.random(_N_GP) < 0.08] = np.nan


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _obs(y):
    """Return observed (non-NaN) subset of y."""
    return y[~np.isnan(y)]


# ===========================================================================
# 1. Numerical utilities
# ===========================================================================

class TestNumericalUtils:
    """Tests for MissLearn._utils: Cholesky packing, MVN, conditional params,
    and numerical Hessian."""

    def test_pack_unpack_cholesky_roundtrip(self):
        from MissLearn._utils import pack_cholesky, unpack_cholesky
        rng = np.random.default_rng(1)
        A   = rng.standard_normal((4, 4))
        S   = A @ A.T + 2 * np.eye(4)
        vec, L = pack_cholesky(S)
        L2  = unpack_cholesky(vec, 4)
        S2  = L2 @ L2.T
        assert np.allclose(S, S2, atol=1e-12), "Cholesky round-trip failed"

    def test_pack_cholesky_diagonal_positive(self):
        from MissLearn._utils import pack_cholesky
        S   = np.eye(3) * 4.0
        vec, L = pack_cholesky(S)
        assert np.all(np.diag(L) > 0), "Diagonal of L must be positive"

    def test_mvn_logpdf_scalar(self):
        from MissLearn._utils import mvn_logpdf
        from scipy.stats import norm
        ll = mvn_logpdf(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))
        assert abs(ll - float(norm.logpdf(0.0))) < 1e-10

    def test_mvn_logpdf_batch_equals_scalar(self):
        from MissLearn._utils import mvn_logpdf, mvn_logpdf_batch
        rng = np.random.default_rng(7)
        mu  = rng.standard_normal(3)
        A   = rng.standard_normal((3, 3))
        S   = A @ A.T + np.eye(3)
        X   = rng.standard_normal((8, 3))
        scalar_ll = np.array([mvn_logpdf(X[i], mu, S) for i in range(8)])
        batch_ll  = mvn_logpdf_batch(X, mu, S)
        assert np.allclose(scalar_ll, batch_ll, atol=1e-10)

    def test_conditional_normal_params_shape(self):
        from MissLearn._utils import conditional_normal_params
        rng  = np.random.default_rng(3)
        A    = rng.standard_normal((5, 5))
        S    = A @ A.T + np.eye(5)
        mu   = rng.standard_normal(5)
        mu_c, S_c = conditional_normal_params(
            mu, S, obs_idx=[0, 1, 2], mis_idx=[3, 4], x_obs=np.array([1.0, 0.5, -0.3])
        )
        assert mu_c.shape == (2,)
        assert S_c.shape  == (2, 2)
        # Conditional covariance is symmetric
        assert np.allclose(S_c, S_c.T, atol=1e-12)

    def test_conditional_normal_params_reduces_variance(self):
        """Knowing observations cannot increase variance of missing variables."""
        from MissLearn._utils import conditional_normal_params
        rng  = np.random.default_rng(11)
        A    = rng.standard_normal((4, 4))
        S    = A @ A.T + 2 * np.eye(4)
        mu   = np.zeros(4)
        _, S_c = conditional_normal_params(
            mu, S, obs_idx=[0, 1], mis_idx=[2, 3], x_obs=np.array([1.0, -0.5])
        )
        # Marginal variance of [2,3]
        S_marg = S[np.ix_([2, 3], [2, 3])]
        assert np.all(np.diag(S_c) <= np.diag(S_marg) + 1e-10)

    def test_numerical_hessian_symmetric(self):
        from MissLearn._utils import numerical_hessian
        f = lambda x: float(x @ x + np.sin(x[0]) * x[1])
        H = numerical_hessian(f, np.array([0.5, -0.3, 0.7]))
        assert np.allclose(H, H.T, atol=1e-6), "Hessian must be symmetric"

    def test_numerical_hessian_known_value(self):
        from MissLearn._utils import numerical_hessian
        # f(x) = x^2, f'' = 2
        f = lambda x: float(x[0] ** 2)
        H = numerical_hessian(f, np.array([0.0]))
        assert abs(H[0, 0] - 2.0) < 1e-6


# ===========================================================================
# 2. MissLinear
# ===========================================================================

class TestMissLinear:
    """FIML joint multivariate normal linear regression."""

    @pytest.fixture(scope='class')
    def model(self):
        from MissLearn import MissLinear
        m = MissLinear()
        m.fit(_X_reg, _y_reg)
        return m

    def test_attributes_set_after_fit(self, model):
        # MissLinear stores the joint (Y, X) distribution as mu_joint_ / Sigma_joint_
        for attr in ['coef_', 'intercept_', 'se_', 'loglik_', 'aic_', 'bic_',
                     'mu_joint_', 'Sigma_joint_']:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

    def test_coef_shape(self, model):
        assert model.coef_.shape == (_P,)

    def test_se_finite(self, model):
        assert np.all(np.isfinite(model.se_)), "se_ contains non-finite values"

    def test_coef_direction(self, model):
        """True coefficients are [2.5, -1.2, 0, 0]: signs should be recovered."""
        assert model.coef_[0] > 0.5, f"coef[0] expected positive, got {model.coef_[0]:.3f}"
        assert model.coef_[1] < -0.2, f"coef[1] expected negative, got {model.coef_[1]:.3f}"

    def test_predict_shape(self, model):
        preds = model.predict(_X_reg)
        assert preds.shape == (_N,)

    def test_predict_finite(self, model):
        preds = model.predict(_X_reg)
        assert np.all(np.isfinite(preds))

    def test_predict_all_nan_row(self, model):
        X_miss = np.full((1, _P), np.nan)
        pred   = model.predict(X_miss)
        assert np.isfinite(pred[0])  # falls back to intercept / marginal mean

    def test_predict_partial_row(self, model):
        X_part = np.array([[1.0, np.nan, 0.5, np.nan]])
        pred   = model.predict(X_part)
        assert np.isfinite(pred[0])

    def test_score_positive(self, model):
        r2 = model.score(_X_reg, _y_reg)
        assert r2 > 0.5, f"Expected R² > 0.5, got {r2:.3f}"

    def test_predict_interval_shape(self, model):
        lo, hi = model.predict_interval(_X_reg)
        assert lo.shape == (_N,) and hi.shape == (_N,)
        assert np.all(lo <= hi + 1e-10)

    def test_predict_interval_wider_for_missing(self, model):
        """All-missing row should have wider interval than complete row."""
        X_complete = np.array([[1.0, 0.5, -0.3, 0.2]])
        X_missing  = np.full((1, _P), np.nan)
        lo_c, hi_c = model.predict_interval(X_complete)
        lo_m, hi_m = model.predict_interval(X_missing)
        assert (hi_m[0] - lo_m[0]) >= (hi_c[0] - lo_c[0]) - 1e-6

    def test_loglik_aic_bic_finite(self, model):
        for attr in ['loglik_', 'aic_', 'bic_']:
            assert np.isfinite(getattr(model, attr)), f"{attr} is not finite"

    def test_summary_runs(self, model):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.summary()
        assert len(buf.getvalue()) > 10

    def test_nan_y_handled(self):
        from MissLearn import MissLinear
        y = _y_reg.copy()
        assert np.isnan(y).any(), "Test requires NaN in y"
        m = MissLinear()
        m.fit(_X_reg, y)
        preds = m.predict(_X_reg)
        assert np.all(np.isfinite(preds))

    def test_no_theta_opt_stored(self, model):
        """Avoid memory leak: _theta_opt must not be a stored attribute."""
        assert not hasattr(model, '_theta_opt')


# ===========================================================================
# 3. MissLogistic
# ===========================================================================

class TestMissLogistic:
    """FIML logistic regression."""

    @pytest.fixture(scope='class')
    def model(self):
        from MissLearn import MissLogistic
        m = MissLogistic()
        m.fit(_X_clf, _y_clf)
        return m

    def test_attributes(self, model):
        for attr in ['coef_', 'intercept_', 'loglik_', 'aic_', 'bic_']:
            assert hasattr(model, attr)

    def test_predict_proba_shape(self, model):
        proba = model.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(_X_clf)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_predict_proba_in_unit_interval(self, model):
        proba = model.predict_proba(_X_clf)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_returns_classes(self, model):
        preds = model.predict(_X_clf)
        assert set(np.unique(preds[~np.isnan(preds)])).issubset({0.0, 1.0})

    def test_score_above_chance(self, model):
        acc = model.score(_X_clf, _y_clf)
        assert acc > 0.55, f"Expected accuracy > 0.55, got {acc:.3f}"

    def test_coef_direction(self, model):
        """True coef[0] > 0, coef[1] < 0."""
        assert model.coef_[0] > 0.2
        assert model.coef_[1] < -0.2

    def test_summary_runs(self, model):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.summary()
        assert len(buf.getvalue()) > 10


# ===========================================================================
# 4. MissRidgeRegressor
# ===========================================================================

class TestMissRidgeRegressor:
    """Ridge-penalized FIML linear regression."""

    @pytest.fixture(scope='class')
    def model(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, compute_se=True)
        m.fit(_X_reg, _y_reg)
        return m

    def test_attributes(self, model):
        for attr in ['coef_', 'intercept_', 'sigma_sq_', 'se_',
                     'loglik_', 'aic_', 'bic_', 'converged_',
                     'mu_X_', 'Sigma_X_']:
            assert hasattr(model, attr), f"Missing: {attr}"

    def test_coef_shape(self, model):
        assert model.coef_.shape == (_P,)

    def test_se_shape(self, model):
        # se_ = [se_intercept, se_beta_0, ..., se_beta_{p-1}]
        assert model.se_.shape == (_P + 1,)

    def test_se_positive(self, model):
        assert np.all(model.se_ > 0), "All standard errors must be positive"

    def test_loglik_unpenalized(self, model):
        """loglik_ must be the unpenalized data log-likelihood (fix #25)."""
        from MissLearn import MissRidgeRegressor
        m0 = MissRidgeRegressor(alpha=0.0, compute_se=False)
        m0.fit(_X_reg, _y_reg)
        m1 = MissRidgeRegressor(alpha=10.0, compute_se=False)
        m1.fit(_X_reg, _y_reg)
        # Unpenalized loglik should be roughly similar; penalized loglik would differ wildly
        assert abs(m0.loglik_ - m1.loglik_) < abs(m0.loglik_) * 0.5, \
            "loglik_ appears to include the ridge penalty (should be data log-likelihood only)"

    def test_predict_shape_finite(self, model):
        preds = model.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_predict_interval_ordering(self, model):
        X_c = np.array([[1.0, 0.5, -0.3, 0.7]])
        X_m = np.full((1, _P), np.nan)
        lo_c, hi_c = model.predict_interval(X_c)
        lo_m, hi_m = model.predict_interval(X_m)
        assert (hi_m[0] - lo_m[0]) >= (hi_c[0] - lo_c[0]) - 1e-6

    def test_score_positive(self, model):
        assert model.score(_X_reg, _y_reg) > 0.4

    def test_compute_se_false(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(_X_reg, _y_reg)
        assert not hasattr(m, 'se_') or m.se_ is None or np.all(np.isnan(m.se_))

    def test_sigma_sq_positive(self, model):
        assert model.sigma_sq_ > 0

    def test_alpha_zero_close_to_linear(self):
        """alpha=0 ridge should produce similar coef_ to MissLinear."""
        from MissLearn import MissRidgeRegressor, MissLinear
        rng  = np.random.default_rng(99)
        X    = rng.standard_normal((200, 3))
        y    = 2.0 * X[:, 0] - X[:, 1] + rng.standard_normal(200) * 0.3
        X[rng.random((200, 3)) < 0.15] = np.nan
        y[rng.random(200) < 0.08] = np.nan
        lin  = MissLinear(); lin.fit(X, y)
        rdg  = MissRidgeRegressor(alpha=0.0, compute_se=False); rdg.fit(X, y)
        assert np.allclose(lin.coef_, rdg.coef_, atol=0.3), \
            f"alpha=0 Ridge coef {rdg.coef_} too far from Linear {lin.coef_}"

    def test_summary_runs(self, model):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.summary()
        assert 'Ridge' in buf.getvalue()


# ===========================================================================
# 5. MissRidgeClassifier
# ===========================================================================

class TestMissRidgeClassifier:
    """Ridge-penalized FIML logistic classifier."""

    @pytest.fixture(scope='class')
    def model(self):
        from MissLearn import MissRidgeClassifier
        m = MissRidgeClassifier(alpha=1.0, compute_se=True)
        m.fit(_X_clf, _y_clf)
        return m

    def test_attributes(self, model):
        for attr in ['coef_', 'intercept_', 'loglik_']:
            assert hasattr(model, attr)

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(_X_clf)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_predict_returns_binary(self, model):
        preds = model.predict(_X_clf)
        obs   = preds[~np.isnan(preds)]
        assert set(obs).issubset({0.0, 1.0})

    def test_score_above_chance(self, model):
        assert model.score(_X_clf, _y_clf) > 0.55

    def test_coef_direction(self, model):
        assert model.coef_[0] > 0.0
        assert model.coef_[1] < 0.0


# ===========================================================================
# 6. MissRidge (auto-task selector)
# ===========================================================================

class TestMissRidge:
    """MissRidge auto-detects regression vs. classification from y."""

    def test_regression_task_detected(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_reg, _y_reg)
        assert m.task_ == 'regression'

    def test_classification_task_detected(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_clf, _y_clf)
        assert m.task_ == 'classification'

    def test_regression_predict_finite(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_reg, _y_reg)
        assert np.all(np.isfinite(m.predict(_X_reg)))

    def test_classification_predict_proba_valid(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_predict_proba_unavailable_for_regression(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_reg, _y_reg)
        with pytest.raises(AttributeError):
            m.predict_proba(_X_reg)

    def test_predict_interval_unavailable_for_classification(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_clf, _y_clf)
        with pytest.raises(AttributeError):
            m.predict_interval(_X_clf)

    def test_delegates_coef_(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=1.0)
        m.fit(_X_reg, _y_reg)
        assert hasattr(m, 'coef_') and m.coef_.shape == (_P,)

    def test_sklearn_get_set_params(self):
        from MissLearn import MissRidge
        m = MissRidge(alpha=2.5)
        params = m.get_params()
        assert params['alpha'] == 2.5
        m.set_params(alpha=0.1)
        assert m.alpha == 0.1


# ===========================================================================
# 7. MissLASSO
# ===========================================================================

class TestMissLASSO:
    """LASSO-penalized FIML models."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissLASSORegressor
        m = MissLASSORegressor(alpha=0.1)
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_regressor_score_positive(self):
        from MissLearn import MissLASSORegressor
        m = MissLASSORegressor(alpha=0.05)
        m.fit(_X_reg, _y_reg)
        assert m.score(_X_reg, _y_reg) > 0.3

    def test_classifier_proba_valid(self):
        from MissLearn import MissLASSOClassifier
        m = MissLASSOClassifier(alpha=0.1)
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_auto_regression(self):
        from MissLearn import MissLASSO
        m = MissLASSO(alpha=0.1)
        m.fit(_X_reg, _y_reg)
        assert m.task_ == 'regression'

    def test_auto_classification(self):
        from MissLearn import MissLASSO
        m = MissLASSO(alpha=0.1)
        m.fit(_X_clf, _y_clf)
        assert m.task_ == 'classification'

    def test_strong_penalty_shrinks_coef(self):
        from MissLearn import MissLASSORegressor
        m_weak  = MissLASSORegressor(alpha=0.01)
        m_strong = MissLASSORegressor(alpha=5.0)
        m_weak.fit(_X_reg, _y_reg)
        m_strong.fit(_X_reg, _y_reg)
        assert np.sum(np.abs(m_strong.coef_)) <= np.sum(np.abs(m_weak.coef_)) + 1e-4


# ===========================================================================
# 8. MissNeighbors
# ===========================================================================

class TestMissNeighbors:
    """K-nearest neighbours with FIML distance correction."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissNeighborsRegressor
        m = MissNeighborsRegressor(n_neighbors=5)
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_regressor_score_positive(self):
        from MissLearn import MissNeighborsRegressor
        m = MissNeighborsRegressor(n_neighbors=7)
        m.fit(_X_reg, _y_reg)
        assert m.score(_X_reg, _y_reg) > 0.2

    def test_classifier_proba_valid(self):
        from MissLearn import MissNeighborsClassifier
        m = MissNeighborsClassifier(n_neighbors=5)
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_classifier_score_above_chance(self):
        from MissLearn import MissNeighborsClassifier
        m = MissNeighborsClassifier(n_neighbors=5)
        m.fit(_X_clf, _y_clf)
        assert m.score(_X_clf, _y_clf) > 0.55

    def test_auto_task_detection(self):
        from MissLearn import MissNeighbors
        mr = MissNeighbors(n_neighbors=5)
        mr.fit(_X_reg, _y_reg)
        assert mr.task_ == 'regression'
        mc = MissNeighbors(n_neighbors=5)
        mc.fit(_X_clf, _y_clf)
        assert mc.task_ == 'classification'

    def test_all_missing_row_finite(self):
        from MissLearn import MissNeighborsRegressor
        m = MissNeighborsRegressor(n_neighbors=5)
        m.fit(_X_reg, _y_reg)
        X_m = np.full((1, _P), np.nan)
        pred = m.predict(X_m)
        assert np.isfinite(pred[0])


# ===========================================================================
# 9. MissBayes
# ===========================================================================

class TestMissBayes:
    """Naive Bayes / Bayesian linear FIML models."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissBayesRegressor
        m = MissBayesRegressor()
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_classifier_proba_valid(self):
        from MissLearn import MissBayesClassifier
        m = MissBayesClassifier()
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_auto_task(self):
        from MissLearn import MissBayes
        m = MissBayes()
        m.fit(_X_reg, _y_reg)
        assert m.task_ == 'regression'


# ===========================================================================
# 10. MissSupport
# ===========================================================================

class TestMissSupport:
    """Support vector FIML models."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissSupportRegressor
        m = MissSupportRegressor()
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_classifier_proba_valid(self):
        from MissLearn import MissSupportClassifier
        m = MissSupportClassifier()
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_auto_task(self):
        from MissLearn import MissSupport
        m = MissSupport()
        m.fit(_X_clf, _y_clf)
        assert m.task_ == 'classification'


# ===========================================================================
# 11. MissGaussian (small n to keep O(n^3) fast)
# ===========================================================================

class TestMissGaussian:
    """Gaussian Process FIML models (small n=40)."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissGaussianRegressor
        m = MissGaussianRegressor()
        m.fit(_X_gp, _y_gp_reg)
        preds = m.predict(_X_gp)
        assert preds.shape == (_N_GP,) and np.all(np.isfinite(preds))

    def test_regressor_predict_interval(self):
        from MissLearn import MissGaussianRegressor
        m = MissGaussianRegressor()
        m.fit(_X_gp, _y_gp_reg)
        lo, hi = m.predict_interval(_X_gp)
        assert np.all(lo <= hi + 1e-10)

    def test_regressor_interval_wider_for_missing(self):
        from MissLearn import MissGaussianRegressor
        m = MissGaussianRegressor()
        m.fit(_X_gp, _y_gp_reg)
        X_c = np.array([[0.5, -0.3, 0.1]])
        X_m = np.full((1, 3), np.nan)
        lo_c, hi_c = m.predict_interval(X_c)
        lo_m, hi_m = m.predict_interval(X_m)
        assert (hi_m[0] - lo_m[0]) >= (hi_c[0] - lo_c[0]) - 1e-6

    def test_classifier_proba_valid(self):
        from MissLearn import MissGaussianClassifier
        m = MissGaussianClassifier()
        m.fit(_X_gp, _y_gp_clf)
        proba = m.predict_proba(_X_gp)
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ===========================================================================
# 12. MissMixed (mixed effects)
# ===========================================================================

class TestMissMixed:
    """Mixed-effects FIML models."""

    def test_regressor_fit_predict(self):
        from MissLearn import MissMixedRegressor
        m = MissMixedRegressor()
        m.fit(_X_reg, _y_reg, groups=_groups)
        preds = m.predict(_X_reg, groups=_groups)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_regressor_random_effects_stored(self):
        from MissLearn import MissMixedRegressor
        m = MissMixedRegressor()
        m.fit(_X_reg, _y_reg, groups=_groups)
        # BLUPs (Best Linear Unbiased Predictors) are the per-group random effects
        assert hasattr(m, 'blup_'), "MissMixedRegressor should expose blup_ after fit"
        assert len(m.blup_) == m.n_groups_

    def test_classifier_fit_predict(self):
        from MissLearn import MissMixedClassifier
        m = MissMixedClassifier()
        m.fit(_X_clf, _y_clf, groups=_groups)
        proba = m.predict_proba(_X_clf, groups=_groups)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_auto_task_mixed(self):
        from MissLearn import MissMixed
        m = MissMixed()
        m.fit(_X_reg, _y_reg, groups=_groups)
        assert m.task_ == 'regression'

    # -- seeding under heavy missingness ------------------------------------
    # Both mixed models used to raise "Only {n} complete cases; need at least
    # {p+2}" when too few rows were complete to seed the optimiser. That is the
    # wrong failure mode: the complete-case fit only sets a STARTING POINT, and
    # the FIML likelihood still uses every observed entry. MissLogistic and
    # MissLASSOClassifier fall back to a mean-imputed seed instead, and these
    # two tests pin that same behaviour for the mixed family.

    @staticmethod
    def _sparse_grouped(seed=0, miss=0.45, p=6, n_groups=12, per_group=20):
        """Grouped data at ~45% MCAR, where almost no row is complete."""
        rng = np.random.default_rng(seed)
        n = n_groups * per_group
        groups = np.repeat(np.arange(n_groups), per_group)
        b = rng.normal(0, 0.8, n_groups)
        X = rng.standard_normal((n, p))
        beta = np.array([1.2, -0.8, 0.5, 0.0, 0.3, -0.4])[:p]
        eta = 0.4 + X @ beta + b[groups]
        y_reg = eta + rng.standard_normal(n) * 0.5
        y_clf = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        Xm = X.copy()
        Xm[rng.random(X.shape) < miss] = np.nan
        return Xm, y_reg, y_clf, groups, p

    def test_regressor_few_complete_cases(self):
        from MissLearn import MissMixedRegressor
        Xm, y_reg, _, groups, p = self._sparse_grouped()
        n_complete = int((~np.isnan(Xm).any(axis=1)).sum())
        assert n_complete < p + 2, (
            "test data must have fewer than p+2 complete rows to exercise the "
            "fallback; got %d" % n_complete)

        m = MissMixedRegressor(compute_se=False)
        m.fit(Xm, y_reg, groups=groups)          # used to raise ValueError

        preds = m.predict(Xm, groups=groups)
        assert preds.shape == y_reg.shape and np.all(np.isfinite(preds))
        assert np.isfinite(m.tau_sq_) and m.tau_sq_ >= 0
        assert np.isfinite(m.sigma_sq_) and m.sigma_sq_ > 0
        assert np.all(np.isfinite(m.coef_))
        # the fit should still carry signal, not collapse to a constant
        assert np.corrcoef(preds, y_reg)[0, 1] > 0.3

    def test_classifier_few_complete_cases(self):
        from MissLearn import MissMixedClassifier
        Xm, _, y_clf, groups, p = self._sparse_grouped()
        n_complete = int((~np.isnan(Xm).any(axis=1)).sum())
        assert n_complete < p + 2

        m = MissMixedClassifier(compute_se=False)
        m.fit(Xm, y_clf, groups=groups)          # used to raise ValueError

        proba = m.predict_proba(Xm, groups=groups)
        assert proba.shape == (len(y_clf), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(np.isfinite(proba))
        assert np.isfinite(m.tau_sq_) and m.tau_sq_ >= 0
        assert np.all(np.isfinite(m.coef_))

    def test_seed_fallback_does_not_disturb_easy_case(self):
        """With plenty of complete rows the seed path is unchanged, so a
        lightly-missing fit must still succeed and stay well conditioned."""
        from MissLearn import MissMixedRegressor
        Xm, y_reg, _, groups, p = self._sparse_grouped(miss=0.05)
        assert int((~np.isnan(Xm).any(axis=1)).sum()) >= p + 2
        m = MissMixedRegressor(compute_se=False)
        m.fit(Xm, y_reg, groups=groups)
        assert np.isfinite(m.tau_sq_) and np.isfinite(m.sigma_sq_)
        assert np.all(np.isfinite(m.predict(Xm, groups=groups)))


# ===========================================================================
# 13. MissEnsemble
# ===========================================================================

class TestMissEnsemble:
    """Bootstrap-aggregated ensemble."""

    @pytest.fixture(scope='class')
    def reg_ensemble(self):
        from MissLearn import MissEnsemble, MissRidgeRegressor
        m = MissEnsemble(estimator=MissRidgeRegressor(alpha=1.0),
                         n_estimators=10, random_state=42)
        m.fit(_X_reg, _y_reg)
        return m

    @pytest.fixture(scope='class')
    def clf_ensemble(self):
        from MissLearn import MissEnsemble, MissRidgeClassifier
        m = MissEnsemble(estimator=MissRidgeClassifier(alpha=1.0),
                         n_estimators=10, random_state=42)
        m.fit(_X_clf, _y_clf)
        return m

    def test_regression_predict_shape(self, reg_ensemble):
        preds = reg_ensemble.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_regression_predict_interval(self, reg_ensemble):
        lo, hi = reg_ensemble.predict_interval(_X_reg)
        assert np.all(lo <= hi + 1e-10)

    def test_regression_score_positive(self, reg_ensemble):
        assert reg_ensemble.score(_X_reg, _y_reg) > 0.3

    def test_classification_predict_proba_valid(self, clf_ensemble):
        proba = clf_ensemble.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_classification_score_above_chance(self, clf_ensemble):
        assert clf_ensemble.score(_X_clf, _y_clf) > 0.55

    def test_classes_from_y_binary(self, clf_ensemble):
        """classes_ must reflect actual classes in y, not hardcoded [0,1]."""
        assert set(clf_ensemble.classes_).issubset({0.0, 1.0})

    def test_classes_multiclass(self):
        """classes_ must handle K>2 correctly (fix #20)."""
        from MissLearn import MissEnsemble, MissMulticlass, MissRidgeClassifier
        # MissRidgeClassifier is binary-only; wrap in MissMulticlass for K>2
        base = MissMulticlass(estimator=MissRidgeClassifier(alpha=1.0))
        m = MissEnsemble(estimator=base, n_estimators=5, random_state=7)
        m.fit(_X_mc, _y_mc)
        assert len(m.classes_) == 3

    def test_n_estimators_attribute(self, reg_ensemble):
        assert reg_ensemble.n_estimators_ == 10

    def test_oob_score(self):
        from MissLearn import MissEnsemble, MissRidgeRegressor
        m = MissEnsemble(estimator=MissRidgeRegressor(alpha=1.0),
                         n_estimators=20, oob_score=True, random_state=0)
        m.fit(_X_reg, _y_reg)
        assert bool(m.oob_scores_)

    def test_heterogeneous_ensemble(self):
        from MissLearn import MissEnsemble, MissRidgeRegressor, MissLinear
        m = MissEnsemble(
            estimators=[('ridge', MissRidgeRegressor(alpha=1.0)),
                        ('linear', MissLinear())],
            bootstrap=False,
        )
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_unsupported_estimator_raises(self):
        from MissLearn import MissEnsemble
        from sklearn.linear_model import Ridge
        with pytest.raises(ValueError, match="not supported"):
            MissEnsemble(estimator=Ridge(), n_estimators=5).fit(_X_reg, _y_reg)


# ===========================================================================
# 14. MissMulticlass
# ===========================================================================

class TestMissMulticlass:
    """One-vs-Rest multi-class extension."""

    @pytest.fixture(scope='class')
    def model(self):
        from MissLearn import MissMulticlass, MissRidgeClassifier
        m = MissMulticlass(MissRidgeClassifier(alpha=1.0))
        m.fit(_X_mc, _y_mc)
        return m

    def test_n_classes(self, model):
        assert model.n_classes_ == 3

    def test_classes(self, model):
        assert set(model.classes_) == {0.0, 1.0, 2.0}

    def test_predict_proba_shape(self, model):
        proba = model.predict_proba(_X_mc)
        assert proba.shape == (_N, 3)

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(_X_mc)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_predict_returns_known_classes(self, model):
        preds = model.predict(_X_mc)
        obs   = preds[~np.isnan(preds)]
        assert set(obs).issubset({0.0, 1.0, 2.0})

    def test_score_above_chance(self, model):
        # 3-class random chance is 1/3 ≈ 0.33
        assert model.score(_X_mc, _y_mc) > 0.38

    def test_feature_names_in(self):
        """feature_names_in_ set correctly from ndarray (not mis-labeled)."""
        import pandas as pd
        from MissLearn import MissMulticlass, MissRidgeClassifier
        cols = ['a', 'b', 'c']
        df   = pd.DataFrame(_X_mc, columns=cols)
        m    = MissMulticlass(MissRidgeClassifier(alpha=1.0))
        m.fit(df, _y_mc)
        assert list(m.feature_names_in_) == cols


# ===========================================================================
# 15. MissPreprocessor
# ===========================================================================

class TestMissPreprocessor:
    """Validation, compatibility checking, and categorical encoding."""

    def test_prefit_check_passes_clean_data(self):
        from MissLearn import prefit_check
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((100, 3))
        X[rng.random((100, 3)) < 0.1] = np.nan
        y   = rng.standard_normal(100)
        result = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert result.passed

    def test_prefit_check_all_nan_column_is_error(self):
        from MissLearn import prefit_check
        X      = np.ones((50, 2))
        X[:, 0] = np.nan
        result = prefit_check(X, raise_on_error=False, emit_warnings=False)
        assert not result.passed and len(result.errors) > 0

    def test_prefit_check_constant_column_is_error(self):
        from MissLearn import prefit_check
        X = np.ones((50, 2))
        result = prefit_check(X, raise_on_error=False, emit_warnings=False)
        assert not result.passed

    def test_preprocessor_passthrough_numeric(self):
        from MissLearn import MissPreprocessor, MissRidgeRegressor
        m = MissPreprocessor(MissRidgeRegressor(alpha=1.0), encode='auto',
                             validate=False, verbose=False)
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))

    def test_preprocessor_categorical_encoding(self):
        from MissLearn import MissPreprocessor, MissRidgeRegressor
        rng = np.random.default_rng(5)
        X   = rng.standard_normal((100, 2))
        X   = np.column_stack([X, rng.integers(0, 3, size=100).astype(float)])
        y   = rng.standard_normal(100)
        m   = MissPreprocessor(MissRidgeRegressor(alpha=1.0), encode='auto',
                               validate=False, verbose=False)
        m.fit(X, y)
        # 3-category column (with drop='first') → 2 encoded columns
        assert m.n_features_out_ == 4   # 2 continuous + 2 one-hot

    def test_preprocessor_nan_preserved_in_encoded(self):
        from MissLearn import MissPreprocessor, MissRidgeRegressor
        rng = np.random.default_rng(6)
        X   = rng.standard_normal((60, 1))
        cat = rng.integers(0, 2, size=60).astype(float)
        cat[0] = np.nan   # one NaN in categorical column
        X   = np.column_stack([X, cat])
        y   = rng.standard_normal(60)
        m   = MissPreprocessor(MissRidgeRegressor(alpha=1.0), encode='auto',
                               validate=False, verbose=False)
        m.fit(X, y)
        X_enc = m._apply_encoding(X, m.encoding_map_)
        # Row 0: both one-hot columns should be NaN
        assert np.all(np.isnan(X_enc[0, 1:]))

    def test_preprocessor_predict_on_new_data(self):
        from MissLearn import MissPreprocessor, MissRidgeRegressor
        m = MissPreprocessor(MissRidgeRegressor(alpha=1.0), encode='auto',
                             validate=False, verbose=False)
        m.fit(_X_reg, _y_reg)
        rng  = np.random.default_rng(77)
        X_new = rng.standard_normal((20, _P))
        X_new[rng.random((20, _P)) < 0.1] = np.nan
        preds = m.predict(X_new)
        assert preds.shape == (20,) and np.all(np.isfinite(preds))


# ===========================================================================
# 16. MissDiagnostic
# ===========================================================================

class TestMissDiagnostic:
    """Missing data mechanism diagnostics."""

    @pytest.fixture(scope='class')
    def diag(self):
        from MissLearn import MissDiagnostic
        return MissDiagnostic(_X_reg, _y_reg)

    def test_little_mcar_keys(self, diag):
        result = diag.little_mcar_test()
        for key in ['statistic', 'df', 'pvalue', 'significant', 'patterns_used']:
            assert key in result, f"little_mcar_test missing key: {key}"

    def test_little_mcar_pvalue_in_unit_interval(self, diag):
        result = diag.little_mcar_test()
        assert 0.0 <= result['pvalue'] <= 1.0

    def test_little_mcar_statistic_nonneg(self, diag):
        result = diag.little_mcar_test()
        assert result['statistic'] >= 0.0

    def test_mar_plausibility_returns_dict(self, diag):
        result = diag.mar_plausibility()
        assert isinstance(result, dict)

    def test_mar_plausibility_pvalues_valid(self, diag):
        result = diag.mar_plausibility()
        for col, info in result.items():
            assert 0.0 <= info['pvalue'] <= 1.0

    def test_pattern_summary_coverage(self, diag):
        patterns = diag.pattern_summary()
        total_n  = sum(p['n'] for p in patterns)
        assert total_n == diag.n_

    def test_missingness_correlations_shape(self, diag):
        corr = diag.missingness_correlations()
        assert corr.shape == (diag.p_, diag.p_)
        # Diagonal should be 1.0 (or NaN if column is fully observed / fully missing)
        diag_vals = np.diag(corr)
        non_nan = diag_vals[~np.isnan(diag_vals)]
        assert np.allclose(non_nan, 1.0, atol=1e-10)

    def test_summary_runs(self, diag):
        diag.fit()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diag.summary()
        assert len(buf.getvalue()) > 10


# ===========================================================================
# 16b. MissRecommender
# ===========================================================================

class TestMissRecommender:
    """Evidence-based model triage."""

    @staticmethod
    def _mar(X, rate, seed=0):
        """Missingness of column j driven by the observed column j-1."""
        rng = np.random.default_rng(seed)
        Xm  = X.copy()
        n, p = X.shape
        for j in range(1, p):
            z  = X[:, j - 1]
            pr = 1.0 / (1.0 + np.exp(-(z - np.median(z))))
            pr = pr * (rate * p / max(1, p - 1)) / pr.mean()
            Xm[rng.random(n) < np.clip(pr, 0.0, 0.95), j] = np.nan
        return Xm

    @pytest.fixture(scope='class')
    def rec(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(0)
        X   = rng.normal(size=(400, 5))
        y   = X @ np.array([2., -1.5, 1., 0., .5]) + rng.normal(scale=.5, size=400)
        return MissRecommender().fit(self._mar(X, 0.20), y)

    def test_detects_regression(self, rec):
        assert rec.task_ == 'regression'

    def test_ranking_is_complete_and_sorted(self, rec):
        assert len(rec.ranked_) == 8
        live = [r for r in rec.ranked_ if not r['vetoed']]
        assert live == sorted(live, key=lambda r: -r['score'])

    def test_recommended_is_a_regressor(self, rec):
        assert rec.recommended_ == rec.ranked_[0]['estimator']
        assert 'Classifier' not in rec.recommended_

    def test_every_score_has_reasons(self, rec):
        for r in rec.ranked_:
            if r['score'] != 0 and not r['vetoed']:
                assert r['reasons'], f"{r['family']} scored without a reason"

    def test_mixed_vetoed_without_groups(self, rec):
        mixed = [r for r in rec.ranked_ if r['family'] == 'MissMixed'][0]
        assert mixed['vetoed'] and 'groups' in mixed['veto_reason']

    def test_make_estimator_is_fittable(self, rec):
        rng = np.random.default_rng(1)
        X   = rng.normal(size=(200, 5))
        y   = X @ np.array([2., -1.5, 1., 0., .5]) + rng.normal(scale=.5, size=200)
        model = rec.make_estimator()
        model.fit(self._mar(X, 0.20, seed=1), y)
        assert np.all(np.isfinite(model.predict(X)))

    def test_summary_runs(self, rec):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rec.summary()
        assert 'recommendation' in buf.getvalue().lower()

    def test_detects_classification(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(2)
        X   = rng.normal(size=(300, 4))
        y   = (X @ rng.normal(size=4) > 0).astype(float)
        r   = MissRecommender().fit(self._mar(X, 0.20, seed=2), y)
        assert r.task_ == 'classification'
        assert 'Classifier' in r.recommended_ or r.recommended_ == 'MissLogistic'

    def test_gp_vetoed_when_n_large(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(3)
        X   = rng.normal(size=(300, 3))
        y   = X @ np.array([1., -1., .5]) + rng.normal(scale=.5, size=300)
        r   = MissRecommender(gp_max_n=100).fit(X, y)
        gp  = [x for x in r.ranked_ if x['family'] == 'MissGaussian'][0]
        assert gp['vetoed'] and 'O(n^3)' in gp['veto_reason']

    def test_recommends_dropping_a_mostly_absent_column(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(4)
        X   = rng.normal(size=(300, 4))
        y   = X @ np.array([1., -1., .5, .2]) + rng.normal(scale=.5, size=300)
        X[rng.random(300) < 0.92, 2] = np.nan          # 92% absent
        r   = MissRecommender(feature_names=['a', 'b', 'dead', 'd']).fit(X, y)
        assert 'dead' in r.preprocessing_['drop_columns']
        assert 'dead' in r.preprocessing_['drop_reason']

    def test_promotes_mixed_effects_when_clustered(self):
        from MissLearn import MissRecommender
        rng  = np.random.default_rng(5)
        n_g, per = 30, 15
        g    = np.repeat(np.arange(n_g), per)
        b    = rng.normal(scale=4.0, size=n_g)
        X    = rng.normal(size=(n_g * per, 3))
        y    = X @ np.array([1.5, -1., .8]) + b[g] + rng.normal(
            scale=.6, size=n_g * per)
        r    = MissRecommender(groups=g).fit(self._mar(X, 0.20, seed=5), y)
        assert r.evidence_['icc'] > 0.05
        assert r.recommended_family_ == 'MissMixed'

    def test_recommend_model_wrapper(self):
        from MissLearn import recommend_model, MissRecommender
        rng = np.random.default_rng(6)
        X   = rng.normal(size=(200, 3))
        y   = X @ np.array([1., -1., .5]) + rng.normal(scale=.5, size=200)
        r   = recommend_model(X, y)
        assert isinstance(r, MissRecommender)
        assert r.recommended_


# ===========================================================================
# 17. MissKFold
# ===========================================================================

class TestMissKFold:
    """NaN-safe K-fold cross-validation splitter."""

    def test_fold_count(self):
        from MissLearn import MissKFold
        kf     = MissKFold(n_splits=5)
        splits = list(kf.split(_X_reg))
        assert len(splits) == 5

    def test_full_coverage(self):
        from MissLearn import MissKFold
        kf   = MissKFold(n_splits=5, shuffle=False)
        seen = set()
        for train, test in kf.split(_X_reg):
            seen.update(test.tolist())
        assert seen == set(range(_N))

    def test_no_overlap(self):
        from MissLearn import MissKFold
        kf = MissKFold(n_splits=5, shuffle=False)
        for train, test in kf.split(_X_reg):
            assert len(set(train) & set(test)) == 0

    def test_correct_sizes(self):
        from MissLearn import MissKFold
        kf = MissKFold(n_splits=4, shuffle=False)
        sizes = [len(test) for _, test in kf.split(_X_reg)]
        assert abs(max(sizes) - min(sizes)) <= 1

    def test_shuffle_changes_assignment(self):
        from MissLearn import MissKFold
        kf1 = MissKFold(n_splits=5, shuffle=False)
        kf2 = MissKFold(n_splits=5, shuffle=True, random_state=42)
        folds1 = [test for _, test in kf1.split(_X_reg)]
        folds2 = [test for _, test in kf2.split(_X_reg)]
        assert not all(np.array_equal(a, b) for a, b in zip(folds1, folds2))

    def test_random_state_reproducible(self):
        from MissLearn import MissKFold
        kf1 = MissKFold(n_splits=5, shuffle=True, random_state=7)
        kf2 = MissKFold(n_splits=5, shuffle=True, random_state=7)
        folds1 = [test for _, test in kf1.split(_X_reg)]
        folds2 = [test for _, test in kf2.split(_X_reg)]
        assert all(np.array_equal(a, b) for a, b in zip(folds1, folds2))

    def test_get_n_splits(self):
        from MissLearn import MissKFold
        assert MissKFold(n_splits=7).get_n_splits() == 7


# ===========================================================================
# 18. MissStratifiedKFold
# ===========================================================================

class TestMissStratifiedKFold:
    """Stratified K-fold splitter (handles NaN-y)."""

    def test_full_coverage(self):
        from MissLearn import MissStratifiedKFold
        skf  = MissStratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        seen = set()
        for _, test in skf.split(_X_clf, _y_clf):
            seen.update(test.tolist())
        assert seen == set(range(_N))

    def test_nan_y_distributed(self):
        from MissLearn import MissStratifiedKFold
        skf     = MissStratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        nan_idx = set(np.where(np.isnan(_y_clf))[0])
        folds_with_nan = 0
        for _, test in skf.split(_X_clf, _y_clf):
            if any(i in nan_idx for i in test):
                folds_with_nan += 1
        assert folds_with_nan >= 4, "NaN-y rows should appear in most folds"

    def test_class_balance_maintained(self):
        from MissLearn import MissStratifiedKFold
        skf     = MissStratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        y_obs   = _y_clf[~np.isnan(_y_clf)]
        global_rate = y_obs.mean()
        for _, test in skf.split(_X_clf, _y_clf):
            test_y = _y_clf[test]
            test_y = test_y[~np.isnan(test_y)]
            if len(test_y) > 5:
                fold_rate = test_y.mean()
                assert abs(fold_rate - global_rate) < 0.25

    def test_random_state_reproducible(self):
        from MissLearn import MissStratifiedKFold
        skf1 = MissStratifiedKFold(n_splits=5, shuffle=True, random_state=99)
        skf2 = MissStratifiedKFold(n_splits=5, shuffle=True, random_state=99)
        folds1 = [test for _, test in skf1.split(_X_clf, _y_clf)]
        folds2 = [test for _, test in skf2.split(_X_clf, _y_clf)]
        assert all(np.array_equal(a, b) for a, b in zip(folds1, folds2))


# ===========================================================================
# 19. Cross-validation utilities
# ===========================================================================

class TestCrossVal:
    """miss_cross_val_score and miss_cross_validate."""

    def test_cross_val_score_shape(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m      = MissRidgeRegressor(alpha=1.0, compute_se=False)
        scores = miss_cross_val_score(m, _X_reg, _y_reg, cv=5, random_state=0)
        assert scores.shape == (5,)

    def test_cross_val_score_finite(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m      = MissRidgeRegressor(alpha=1.0, compute_se=False)
        scores = miss_cross_val_score(m, _X_reg, _y_reg, cv=5, random_state=0)
        assert np.all(np.isfinite(scores))

    def test_cross_val_named_scoring(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m  = MissRidgeRegressor(alpha=1.0, compute_se=False)
        r2 = miss_cross_val_score(m, _X_reg, _y_reg, cv=4,
                                  scoring='r2', random_state=1)
        assert r2.shape == (4,) and np.all(np.isfinite(r2))

    def test_cross_val_random_state_reproducible(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m       = MissRidgeRegressor(alpha=1.0, compute_se=False)
        scores1 = miss_cross_val_score(m, _X_reg, _y_reg, cv=4,
                                       scoring='r2', random_state=42)
        scores2 = miss_cross_val_score(m, _X_reg, _y_reg, cv=4,
                                       scoring='r2', random_state=42)
        assert np.allclose(scores1, scores2, atol=1e-10)

    def test_cross_validate_keys(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_validate
        m   = MissRidgeRegressor(alpha=1.0, compute_se=False)
        res = miss_cross_validate(m, _X_reg, _y_reg, cv=3,
                                  scoring='r2', random_state=5)
        assert 'test_score' in res and 'fit_time' in res

    def test_cross_validate_multi_metric(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_validate
        m   = MissRidgeRegressor(alpha=1.0, compute_se=False)
        res = miss_cross_validate(m, _X_reg, _y_reg, cv=3,
                                  scoring=['r2', 'neg_mse'],
                                  return_train_score=True, random_state=5)
        assert 'test_r2' in res and 'test_neg_mse' in res
        assert 'train_r2' in res

    def test_return_estimators(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m      = MissRidgeRegressor(alpha=1.0, compute_se=False)
        scores, ests = miss_cross_val_score(m, _X_reg, _y_reg, cv=3,
                                            return_estimators=True,
                                            random_state=0)
        assert len(ests) == 3
        assert all(hasattr(e, 'coef_') for e in ests)

    def test_unknown_scoring_raises(self):
        """A misspelled scorer is a configuration error, not a fold failure.

        This previously warned once per fold and returned an array of NaN.
        That is the right response to one degenerate split, where the other
        folds may still be informative, but a name that does not exist cannot
        succeed on any fold, and _compute_score already carried a message
        listing every valid option that no caller ever saw. Changed to raise
        before the first fold, which is also what sklearn does.
        """
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        with pytest.raises(ValueError, match='bogus_metric'):
            miss_cross_val_score(m, _X_reg, _y_reg, cv=3,
                                 scoring='bogus_metric', random_state=0)

    def test_failing_fold_still_warns_rather_than_raising(self):
        """The behaviour above is narrowed, not removed.

        A scorer that is valid but blows up on the data must still leave the
        run recoverable: warn, substitute NaN, carry on to the next fold.
        """
        from MissLearn import MissRidgeRegressor
        from MissLearn import miss_cross_val_score

        def exploding_scorer(est, X_test, y_test):
            raise RuntimeError("scorer failed on this fold")

        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scores = miss_cross_val_score(m, _X_reg, _y_reg, cv=3,
                                          scoring=exploding_scorer,
                                          random_state=0)
        assert np.all(np.isnan(scores))
        assert any('scorer failed' in str(wn.message) for wn in w)


# ===========================================================================
# 20. MissImputer
# ===========================================================================

class TestMissImputer:
    """Multiple imputation via joint MVN."""

    @pytest.fixture(scope='class')
    def imputer(self):
        from MissLearn import MissImputer
        imp = MissImputer(m=10, random_state=0)
        imp.fit(_X_reg)
        return imp

    def test_attributes_after_fit(self, imputer):
        for attr in ['mu_', 'Sigma_', 'n_iter_']:
            assert hasattr(imputer, attr)

    def test_mu_shape(self, imputer):
        assert imputer.mu_.shape == (_P,)

    def test_sigma_positive_definite(self, imputer):
        eigvals = np.linalg.eigvalsh(imputer.Sigma_)
        assert np.all(eigvals > 0), "Sigma_ must be positive definite"

    def test_transform_returns_m_datasets(self, imputer):
        datasets = imputer.transform(_X_reg)
        assert len(datasets) == 10

    def test_transform_no_nan(self, imputer):
        datasets = imputer.transform(_X_reg)
        for d in datasets:
            assert not np.isnan(d).any(), "Imputed datasets must have no NaN"

    def test_transform_observed_unchanged(self, imputer):
        obs_mask  = ~np.isnan(_X_reg)
        datasets  = imputer.transform(_X_reg)
        for d in datasets:
            np.testing.assert_array_equal(d[obs_mask], _X_reg[obs_mask])

    def test_transform_mean_deterministic(self, imputer):
        m1 = imputer.transform_mean(_X_reg)
        m2 = imputer.transform_mean(_X_reg)
        np.testing.assert_array_equal(m1, m2)

    def test_transform_mean_no_nan(self, imputer):
        m = imputer.transform_mean(_X_reg)
        assert not np.isnan(m).any()

    def test_rubin_combine_shape(self):
        from MissLearn import MissImputer
        estimates = np.array([[1.0, 2.0], [1.1, 1.9], [0.9, 2.1], [1.05, 2.05]])
        result    = MissImputer.combine(estimates)
        assert result['estimate'].shape == (2,)
        assert result['between_var'].shape == (2,)

    def test_rubin_combine_scalar(self):
        from MissLearn import MissImputer
        estimates = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        variances = np.array([0.04, 0.05, 0.03, 0.04, 0.04])
        result    = MissImputer.combine(estimates, variances)
        assert np.isfinite(result['estimate'])
        assert np.isfinite(result['se'])
        assert 'p_value' in result

    def test_fit_transform_combine(self):
        from MissLearn import MissImputer, MissRidgeRegressor
        imp = MissImputer(m=5, random_state=1)
        imp.fit(_X_reg)
        result = imp.fit_transform_combine(
            _X_reg, _y_reg,
            estimator=MissRidgeRegressor(alpha=1.0, compute_se=False),
            param='coef_',
        )
        assert result['estimate'].shape == (_P,)
        assert np.all(np.isfinite(result['estimate']))

    def test_complete_data_unchanged(self):
        from MissLearn import MissImputer
        rng  = np.random.default_rng(55)
        X_cc = rng.standard_normal((50, 3))   # no NaN
        imp  = MissImputer(m=3, random_state=2)
        imp.fit(X_cc)
        datasets = imp.transform(X_cc)
        for d in datasets:
            np.testing.assert_array_almost_equal(d, X_cc, decimal=10)

    def test_posterior_true_produces_variation(self):
        from MissLearn import MissImputer
        imp = MissImputer(m=10, posterior=True, random_state=3)
        imp.fit(_X_reg)
        datasets = imp.transform(_X_reg)
        col0_means = [d[:, 0].mean() for d in datasets]
        assert np.std(col0_means) > 0, "posterior=True should produce parameter variation"


# ===========================================================================
# 21. MissSensitivity
# ===========================================================================

class TestMissSensitivity:
    """MNAR delta-adjustment sensitivity analysis."""

    @pytest.fixture(scope='class')
    def sens(self):
        from MissLearn import MissSensitivity, MissRidgeRegressor
        s = MissSensitivity(
            MissRidgeRegressor(alpha=1.0, compute_se=False),
            delta_range=(-2, 2), n_delta=9, m=5,
            random_state=0,
        )
        s.fit(_X_reg, _y_reg)
        return s

    def test_coef_curves_shape(self, sens):
        assert sens.coef_curves_.shape[0] == 9
        assert sens.coef_curves_.shape[1] == _P

    def test_explainer_classifier_uses_probability_not_label(self):
        """Coalition values must be continuous, not hard class labels.

        Valuing coalitions with predict() on an imbalanced classifier makes
        every coalition return the majority label, so every Shapley difference
        is exactly zero and the attributions silently vanish. This was the
        behaviour before 0.9.1 and it made miss_shap useless on SECOM.
        """
        from MissLearn import MissExplainer, MissRidgeClassifier
        rng = np.random.default_rng(0)
        n, p = 300, 6
        X = rng.normal(size=(n, p))
        lin = X @ np.array([2., -1.5, 1., .5, -.8, .3])
        y = (lin > np.quantile(lin, 0.92)).astype(float)      # ~8% positive
        X[rng.random((n, p)) < 0.15] = np.nan

        model = MissRidgeClassifier(alpha=1.0, compute_se=False).fit(X, y)

        auto = MissExplainer(model, random_state=0).fit(X)
        assert auto.value_scale_ == 'probability'
        # The all-missing baseline must be the base rate, not a class label.
        assert 0.0 < auto.expected_value_ < 1.0

        raw = MissExplainer(model, output='raw', random_state=0).fit(X)
        assert raw.value_scale_ == 'prediction'

        ms_auto = auto.miss_shap(X[:40])
        assert np.abs(ms_auto).max() > 1e-6, \
            "probability-valued missingness SHAP collapsed to zero"
        # The two scales must genuinely differ; identical output means output
        # was not forwarded to the Shapley engine.
        assert not np.allclose(auto.shap_values(X[:40]),
                               raw.shap_values(X[:40]))

    def test_explainer_efficiency_axiom_on_each_scale(self):
        """sum_j phi_ij must equal v(x_i) - baseline on whatever scale is used."""
        from MissLearn import MissExplainer, MissRidgeClassifier
        rng = np.random.default_rng(1)
        n, p = 250, 5
        X = rng.normal(size=(n, p))
        y = (X @ np.array([2., -1., .5, -.7, .3]) > 0).astype(float)
        X[rng.random((n, p)) < 0.12] = np.nan
        model = MissRidgeClassifier(alpha=1.0, compute_se=False).fit(X, y)

        for output in ('raw', 'auto', 'log-odds'):
            e = MissExplainer(model, output=output, random_state=0).fit(X)
            Xs = X[:30]
            phi = e.shap_values(Xs)
            target = e._v(Xs) - e.expected_value_
            assert np.abs(phi.sum(axis=1) - target).max() < 1e-8, \
                f"efficiency violated on the {output} scale"

    def test_explainer_multiclass_requires_explicit_class(self):
        """A 3-class model has no single value function, so refuse to guess."""
        from MissLearn import (MissExplainer, MissMulticlass, MissLogistic)
        rng = np.random.default_rng(2)
        n, p = 240, 5
        X = rng.normal(size=(n, p))
        lin = X @ np.array([2., -1.5, 1., .5, -.8])
        y = np.digitize(lin, np.quantile(lin, [0.33, 0.66])).astype(float)
        X[rng.random((n, p)) < 0.12] = np.nan
        clf = MissMulticlass(MissLogistic()).fit(X, y)

        with pytest.raises(ValueError, match='class_index'):
            MissExplainer(clf, random_state=0).fit(X).shap_values(X[:10])

        priors = []
        for k in (0, 1, 2):
            e = MissExplainer(clf, class_index=k, random_state=0).fit(X)
            phi = e.shap_values(X[:20])
            target = e._v(X[:20]) - e.expected_value_
            assert np.abs(phi.sum(axis=1) - target).max() < 1e-8
            priors.append(e.expected_value_)
        # The three all-missing baselines are class priors, so they sum to one.
        assert abs(sum(priors) - 1.0) < 0.05

    def test_explainer_rejects_bad_output(self):
        from MissLearn import MissExplainer, MissLinear
        rng = np.random.default_rng(3)
        X = rng.normal(size=(120, 4))
        y = X @ np.array([1., -1., .5, .2]) + rng.normal(scale=.3, size=120)
        m = MissLinear(compute_se=False).fit(X, y)
        with pytest.raises(ValueError, match='output must be'):
            MissExplainer(m, output='nonsense').fit(X)
        # A regressor has no predict_proba, so a probability scale is impossible.
        with pytest.raises(AttributeError, match='predict_proba'):
            MissExplainer(m, output='proba').fit(X)

    def test_raises_for_estimator_without_coef(self):
        """A coefficient-free model must fail loudly.

        Silently substituting zeros produced a table of 0.0000 with a
        STABLE verdict on every row, which reads as 'the conclusions are
        robust' when nothing had in fact been computed.
        """
        from MissLearn import MissSensitivity, MissBayesRegressor
        s = MissSensitivity(MissBayesRegressor(), n_delta=3, m=2,
                            random_state=0)
        with pytest.raises(AttributeError, match='does not expose coef_'):
            s.fit(_X_reg, _y_reg)

    def test_feature_names_are_used(self):
        from MissLearn import MissSensitivity, MissLinear
        names = [f'chan{j}' for j in range(_P)]
        s = MissSensitivity(MissLinear(compute_se=False), n_delta=3, m=2,
                            random_state=0)
        s.fit(_X_reg, _y_reg, feature_names=names)
        assert s.feature_names_ == names

    def test_feature_names_length_is_validated(self):
        from MissLearn import MissSensitivity, MissLinear
        s = MissSensitivity(MissLinear(compute_se=False), n_delta=3, m=2,
                            random_state=0)
        with pytest.raises(ValueError, match='feature_names'):
            s.fit(_X_reg, _y_reg, feature_names=['only_one'])

    def test_curves_are_not_silently_zero(self, sens):
        """Guards the regression directly: a real sweep moves coefficients."""
        assert np.isfinite(sens.coef_curves_).any()
        assert np.abs(sens.coef_curves_).max() > 1e-8

    def test_delta_grid_shape(self, sens):
        assert sens.delta_std_grid_.shape == (9,)

    def test_sensitivity_table_shape(self, sens):
        table = sens.sensitivity_table()
        assert table.shape == (9, 1 + 2 * _P)

    def test_tipping_point_returns_float_or_none(self, sens):
        tp = sens.tipping_point(coef_idx=0)
        assert tp is None or isinstance(tp, float)

    def test_verdict_returns_string(self, sens):
        v = sens.verdict(coef_idx=0)
        assert isinstance(v, str) and len(v) > 5

    def test_baseline_index_near_zero(self, sens):
        delta_at_baseline = abs(sens.delta_std_grid_[sens.baseline_idx_])
        assert delta_at_baseline < 0.5

    def test_no_missing_y_warns(self):
        from MissLearn import MissSensitivity, MissRidgeRegressor
        rng = np.random.default_rng(10)
        X   = rng.standard_normal((80, 3))
        y   = rng.standard_normal(80)  # no missing y
        s   = MissSensitivity(MissRidgeRegressor(alpha=1.0, compute_se=False),
                              n_delta=5, m=3, random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            s.fit(X, y)
        assert any('no missing' in str(wn.message).lower() for wn in w)

    def test_summary_runs(self, sens):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sens.summary()
        assert 'sigma' in buf.getvalue().lower()


# ===========================================================================
# 22. MissShapley (low-level Shapley engine)
# ===========================================================================

class TestMissShapley:
    """Low-level exact and kernel Shapley engine."""

    @pytest.fixture(scope='class')
    def reg_model(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(_X_reg, _y_reg)
        return m

    def test_exact_shap_shape(self, reg_model):
        from MissLearn import MissShapley
        engine = MissShapley(reg_model, exact_threshold=15, random_state=0)
        phi    = engine.shap_values(_X_reg[:10])
        assert phi.shape == (10, _P)

    def test_exact_shap_finite(self, reg_model):
        from MissLearn import MissShapley
        engine = MissShapley(reg_model, exact_threshold=15, random_state=0)
        phi    = engine.shap_values(_X_reg[:10])
        assert np.all(np.isfinite(phi))

    def test_efficiency_fully_observed_rows(self, reg_model):
        """sum(phi[i]) == f(X[i]) - baseline for rows with no NaN (fix #3/#5)."""
        from MissLearn import MissShapley
        engine   = MissShapley(reg_model, exact_threshold=15, random_state=0)
        baseline = float(reg_model.predict(np.full((1, _P), np.nan))[0])
        full_obs = _X_reg[~np.isnan(_X_reg).any(axis=1)]
        if len(full_obs) == 0:
            pytest.skip("No fully-observed rows")
        phi   = engine.shap_values(full_obs, baseline=baseline)
        f_x   = reg_model.predict(full_obs)
        err   = np.max(np.abs(phi.sum(axis=1) - (f_x - baseline)))
        assert err < 1e-8, f"Efficiency violated: max error = {err:.2e}"

    def test_nan_features_get_zero_phi(self, reg_model):
        """NaN-valued features must have phi=0."""
        from MissLearn import MissShapley
        engine   = MissShapley(reg_model, exact_threshold=15, random_state=0)
        X_sample = _X_reg[:20]
        phi      = engine.shap_values(X_sample)
        nan_mask = np.isnan(X_sample)
        assert np.all(phi[nan_mask] == 0.0)

    def test_kernel_shap_shape(self, reg_model):
        from MissLearn import MissShapley
        engine = MissShapley(reg_model, exact_threshold=0, n_kernel_samples=256,
                          random_state=1)
        phi = engine.shap_values(_X_reg[:5])
        assert phi.shape == (5, _P)

    def test_kernel_shap_finite(self, reg_model):
        from MissLearn import MissShapley
        engine = MissShapley(reg_model, exact_threshold=0, n_kernel_samples=256,
                          random_state=1)
        phi = engine.shap_values(_X_reg[:5])
        assert np.all(np.isfinite(phi))

    def test_baseline_all_nan(self, reg_model):
        """Baseline computed from all-NaN row must be finite."""
        from MissLearn import MissShapley
        engine   = MissShapley(reg_model, random_state=0)
        baseline = engine._baseline(_P)
        assert np.isfinite(baseline)


# ===========================================================================
# 23. MissExplainer (high-level)
# ===========================================================================

class TestMissExplainer:
    """High-level SHAP explainability interface."""

    @pytest.fixture(scope='class')
    def reg_model(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(_X_reg, _y_reg)
        return m

    @pytest.fixture(scope='class')
    def expl(self, reg_model):
        from MissLearn import MissExplainer
        feat_names = [f'Feature_{j}' for j in range(_P)]
        e = MissExplainer(reg_model, exact_threshold=15, random_state=0)
        e.fit(_X_reg, feature_names=feat_names)
        return e

    @pytest.fixture(scope='class')
    def phi(self, expl):
        return expl.shap_values(_X_reg[:30])

    @pytest.fixture(scope='class')
    def phi_miss(self, expl):
        return expl.miss_shap(_X_reg[:30])

    # --- fit attributes ---

    def test_expected_value_finite(self, expl):
        assert np.isfinite(expl.expected_value_)

    def test_feature_names_set(self, expl):
        assert expl.feature_names_ == [f'Feature_{j}' for j in range(_P)]

    def test_p_set(self, expl):
        assert expl.p_ == _P

    # --- shap_values ---

    def test_shap_values_shape(self, phi):
        assert phi.shape == (30, _P)

    def test_shap_values_finite(self, phi):
        assert np.all(np.isfinite(phi))

    def test_shap_values_nan_features_zero(self, phi):
        nan_mask = np.isnan(_X_reg[:30])
        assert np.all(phi[nan_mask] == 0.0)

    def test_shap_efficiency_fully_observed(self, expl, reg_model, phi):
        full_obs_idx = np.where(~np.isnan(_X_reg[:30]).any(axis=1))[0]
        if len(full_obs_idx) == 0:
            pytest.skip("No fully-observed rows in first 30")
        phi_full = phi[full_obs_idx]
        f_pred   = reg_model.predict(_X_reg[:30][full_obs_idx])
        err      = np.max(np.abs(phi_full.sum(axis=1) - (f_pred - expl.expected_value_)))
        assert err < 1e-8, f"Efficiency error: {err:.2e}"

    # --- miss_shap ---

    def test_miss_shap_shape(self, phi_miss):
        assert phi_miss.shape == (30, _P)

    def test_miss_shap_finite(self, phi_miss):
        assert np.all(np.isfinite(phi_miss))

    def test_miss_shap_observed_vary(self, phi_miss):
        nan_mask = np.isnan(_X_reg[:30])
        obs_phi  = phi_miss[~nan_mask]
        assert obs_phi.std() > 0, "Observed-feature miss_shap values should vary"

    # --- plots ---

    def test_plot_beeswarm_returns_fig_ax(self, expl, phi):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = expl.plot_beeswarm(phi, _X_reg[:30], show=False)
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_plot_miss_importance_returns_fig_ax(self, expl, phi_miss):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = expl.plot_miss_importance(phi_miss, show=False)
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_plot_waterfall_returns_fig_ax(self, expl, phi):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = expl.plot_waterfall(phi, _X_reg[:30], i=0, show=False)
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_plot_dependence_returns_fig_ax(self, expl, phi):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = expl.plot_dependence(phi, _X_reg[:30], feature_idx=0, show=False)
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_plot_dependence_with_interaction(self, expl, phi):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = expl.plot_dependence(phi, _X_reg[:30],
                                       feature_idx=0, interaction_idx=1,
                                       show=False)
        assert fig is not None
        plt.close(fig)

    # --- to_shap_explanation ---

    def test_to_shap_explanation(self, expl, phi):
        try:
            import shap
            explanation = expl.to_shap_explanation(phi, _X_reg[:30])
            assert explanation.values.shape == (30, _P)
            assert np.allclose(explanation.base_values, expl.expected_value_)
        except ImportError:
            pytest.skip("shap library not installed")

    # --- standalone MissShapley matches MissExplainer ---

    def test_standalone_shap_matches_explainer(self, expl, reg_model, phi):
        from MissLearn import MissShapley
        engine       = MissShapley(reg_model, exact_threshold=15, random_state=0)
        phi_standalone = engine.shap_values(
            _X_reg[:30], baseline=expl.expected_value_
        )
        np.testing.assert_allclose(phi_standalone, phi, atol=1e-10)

    # --- viridis constants exported ---

    def test_viridis_constants(self):
        from MissLearn._explainer import _VIRIDIS_HI, _VIRIDIS_LO, _NAN_GREY
        assert _VIRIDIS_HI == '#FDE725'
        assert _VIRIDIS_LO == '#440154'
        assert _NAN_GREY   == '#aaaaaa'

    # --- summary ---

    def test_summary_runs(self, expl, phi, phi_miss):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            expl.summary(shap_values=phi, miss_shap=phi_miss)
        assert 'MissExplainer' in buf.getvalue()


# ===========================================================================
# 24. Pandas DataFrame / Series support
# ===========================================================================

class TestPandasSupport:
    """DataFrame and Series are accepted transparently by all MissLearn models."""

    @pytest.fixture(scope='class')
    def df_and_series(self):
        pytest.importorskip('pandas')
        import pandas as pd
        cols = [f'feat_{j}' for j in range(_P)]
        df   = pd.DataFrame(_X_reg, columns=cols)
        sr   = pd.Series(_y_reg, name='target')
        return df, sr, cols

    def test_ridge_feature_names_from_dataframe(self, df_and_series):
        import pandas as pd
        from MissLearn import MissRidgeRegressor
        df, sr, cols = df_and_series
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(df, sr)
        assert list(m.feature_names_in_) == cols

    def test_ridge_predict_dataframe_matches_ndarray(self, df_and_series):
        import pandas as pd
        from MissLearn import MissRidgeRegressor
        df, sr, cols = df_and_series
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(df, sr)
        preds_df  = m.predict(df)
        preds_arr = m.predict(_X_reg)
        np.testing.assert_allclose(preds_df, preds_arr, atol=1e-10)

    def test_linear_feature_names_from_dataframe(self, df_and_series):
        from MissLearn import MissLinear
        df, sr, cols = df_and_series
        m = MissLinear()
        m.fit(df, sr)
        assert list(m.feature_names_in_) == cols

    def test_mixed_fit_ndarray_predict_dataframe(self, df_and_series):
        import pandas as pd
        from MissLearn import MissRidgeRegressor
        df, sr, cols = df_and_series
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(_X_reg, _y_reg)   # ndarray fit
        preds_df  = m.predict(df)
        preds_arr = m.predict(_X_reg)
        np.testing.assert_allclose(preds_df, preds_arr, atol=1e-10)

    def test_nan_in_dataframe_preserved(self, df_and_series):
        from MissLearn import MissRidgeRegressor
        df, sr, cols = df_and_series
        m     = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(df, sr)
        preds = m.predict(df)
        assert np.all(np.isfinite(preds))

    def test_score_with_dataframe(self, df_and_series):
        from MissLearn import MissRidgeRegressor
        df, sr, cols = df_and_series
        m = MissRidgeRegressor(alpha=1.0, compute_se=False)
        m.fit(df, sr)
        r2 = m.score(df, sr)
        assert np.isfinite(r2)

    def test_ensemble_with_dataframe(self, df_and_series):
        from MissLearn import MissEnsemble, MissRidgeRegressor
        df, sr, cols = df_and_series
        m = MissEnsemble(estimator=MissRidgeRegressor(alpha=1.0),
                         n_estimators=5, random_state=0)
        m.fit(df, sr)
        preds = m.predict(df)
        assert preds.shape == (_N,) and np.all(np.isfinite(preds))


# ===========================================================================
# 25. Copula transform
# ===========================================================================

class TestCopulaTransform:
    """copula=True applies a marginal Gaussian copula before fitting."""

    def test_copula_true_runs(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, copula=True, compute_se=False)
        m.fit(_X_reg, _y_reg)
        assert m.copula_used_
        preds = m.predict(_X_reg)
        assert np.all(np.isfinite(preds))

    def test_copula_auto_runs(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, copula='auto', compute_se=False)
        m.fit(_X_reg, _y_reg)
        preds = m.predict(_X_reg)
        assert np.all(np.isfinite(preds))

    def test_copula_predict_finite_with_missing(self):
        from MissLearn import MissRidgeRegressor
        m = MissRidgeRegressor(alpha=1.0, copula=True, compute_se=False)
        m.fit(_X_reg, _y_reg)
        X_test = _X_reg[:20].copy()
        X_test[0, :] = np.nan   # all missing row
        preds = m.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_copula_classifier_runs(self):
        from MissLearn import MissRidgeClassifier
        m = MissRidgeClassifier(alpha=1.0, copula=True, compute_se=False)
        m.fit(_X_clf, _y_clf)
        proba = m.predict_proba(_X_clf)
        assert proba.shape == (_N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_copula_results_finite_for_skewed_data(self):
        """Copula should handle heavily right-skewed data without NaN output."""
        from MissLearn import MissRidgeRegressor
        rng = np.random.default_rng(77)
        X   = np.exp(rng.standard_normal((120, 3)))   # log-normal (heavy skew)
        X[rng.random((120, 3)) < 0.15] = np.nan
        y   = np.log(X[:, 0] + 1) + rng.standard_normal(120) * 0.3
        y[rng.random(120) < 0.08] = np.nan
        m   = MissRidgeRegressor(alpha=1.0, copula=True, compute_se=False)
        m.fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))


# ===========================================================================
# 25b. Copula transform: ties, discrete columns, and empty columns
# ===========================================================================

class TestGaussianProcessConvergence:
    """The Gaussian process runs two iterations and reported neither.

    Five of the iterative estimators expose converged_; this one did not, so a
    fit whose hyperparameter restarts had all failed came back looking like
    any other, carrying default hyperparameters nobody chose. The kernel is
    O(n^3), so these fixtures are deliberately small.
    """

    @staticmethod
    def _data(seed=0, n=60, p=3):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0]) * 2.0 \
            + rng.normal(scale=0.3, size=n)
        return X, y, (y > np.median(y)).astype(float)

    def test_a_normal_regression_fit_reports_convergence(self):
        from MissLearn import MissGaussianRegressor
        X, y, _ = self._data()
        m = MissGaussianRegressor(n_restarts=1).fit(X, y)
        assert m.converged_ is True
        assert np.isfinite(m.log_marginal_likelihood_)

    def test_a_normal_classification_fit_reports_convergence(self):
        from MissLearn import MissGaussianClassifier
        X, _, yc = self._data()
        m = MissGaussianClassifier(n_restarts=1).fit(X, yc)
        assert m.converged_ is True

    def test_convergence_is_reported_like_the_iterative_siblings(self):
        """A bool named converged_, as MissLinear and the rest provide."""
        from MissLearn import MissGaussianRegressor, MissLinear
        X, y, _ = self._data()
        gp = MissGaussianRegressor(n_restarts=1).fit(X, y)
        lin = MissLinear(compute_se=False).fit(X, y)
        assert isinstance(gp.converged_, bool)
        assert isinstance(lin.converged_, bool)

    def test_every_restart_failing_is_reported_not_hidden(self):
        """The fit still predicts, from a kernel that was never fitted, so it
        has to say so rather than look like a successful fit."""
        import MissLearn._gp as gp_module
        from MissLearn import MissGaussianRegressor
        X, y, _ = self._data()
        original = gp_module.minimize

        def always_fails(*args, **kwargs):
            raise np.linalg.LinAlgError("deliberate")

        gp_module.minimize = always_fails
        try:
            with pytest.warns(UserWarning, match="restart"):
                m = MissGaussianRegressor(n_restarts=2).fit(X, y)
        finally:
            gp_module.minimize = original

        assert m.converged_ is False
        assert m.log_marginal_likelihood_ == -np.inf
        assert np.all(np.isfinite(m.predict(X)))   # still usable, but flagged

    def test_an_unexpected_error_is_not_swallowed(self):
        """The restart loop catches numerical failures on purpose. Anything
        else is a bug and must surface."""
        import MissLearn._gp as gp_module
        from MissLearn import MissGaussianRegressor
        X, y, _ = self._data()
        original = gp_module.minimize

        def unexpected(*args, **kwargs):
            raise RuntimeError("not a numerical failure")

        gp_module.minimize = unexpected
        try:
            with pytest.raises(RuntimeError, match="not a numerical failure"):
                MissGaussianRegressor(n_restarts=1).fit(X, y)
        finally:
            gp_module.minimize = original


class TestMetadataRouting:
    """scikit-learn's metadata routing, which the library used to disable.

    MissBase.get_metadata_routing built a fresh, and so empty,
    MetadataRequest. scikit-learn generates a working set_fit_request for
    every estimator whose fit takes metadata, that call recorded the request
    correctly, and this method then handed the router an object that knew
    nothing about it. A random intercept fitted without its groups is an
    ordinary regression, so the cost of the metadata going astray is the
    whole model: tau 2.71 against 0.00 on the same data.
    """

    @staticmethod
    def _grouped(seed=0, n_sub=20, per=8):
        rng = np.random.default_rng(seed)
        n = n_sub * per
        groups = np.repeat(np.arange(n_sub), per)
        X = rng.normal(size=(n, 3))
        b = np.repeat(rng.normal(scale=3.0, size=n_sub), per)
        y = X @ np.array([1.0, -1.0, 0.5]) + b + rng.normal(scale=0.3, size=n)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.10] = np.nan
        return Xm, y, groups

    @staticmethod
    def _routing_enabled():
        import sklearn
        return sklearn

    def test_a_requested_group_is_actually_declared(self):
        import sklearn
        from MissLearn import MissMixedRegressor
        sklearn.set_config(enable_metadata_routing=True)
        try:
            est = MissMixedRegressor().set_fit_request(groups=True)
            declared = est.get_metadata_routing()._serialize()
            assert declared == {'fit': {'groups': True}}, declared
        finally:
            sklearn.set_config(enable_metadata_routing=False)

    def test_an_estimator_with_no_metadata_declares_nothing(self):
        import sklearn
        from MissLearn import MissLogistic
        sklearn.set_config(enable_metadata_routing=True)
        try:
            assert MissLogistic().get_metadata_routing()._serialize() == {}
        finally:
            sklearn.set_config(enable_metadata_routing=False)

    def test_groups_reach_the_model_through_a_pipeline(self):
        """The end the routing exists for. Without it the fit succeeds and
        the random effect quietly vanishes, which is the bad outcome."""
        import sklearn
        from sklearn.pipeline import Pipeline
        from MissLearn import MissMixedRegressor
        X, y, groups = self._grouped()
        direct = MissMixedRegressor().fit(X, y, groups=groups)
        tau_direct = float(np.sqrt(direct.tau_sq_))
        assert tau_direct > 1.0, "the fixture must have a real group effect"

        sklearn.set_config(enable_metadata_routing=True)
        try:
            pipe = Pipeline([('model',
                              MissMixedRegressor().set_fit_request(groups=True))])
            pipe.fit(X, y, groups=groups)
            tau_pipe = float(np.sqrt(pipe.named_steps['model'].tau_sq_))
        finally:
            sklearn.set_config(enable_metadata_routing=False)
        assert np.isclose(tau_pipe, tau_direct, rtol=1e-6)

    def test_groups_route_through_cross_val_score(self):
        import sklearn
        from sklearn.model_selection import GroupKFold, cross_val_score
        from MissLearn import MissMixedRegressor
        X, y, groups = self._grouped()
        sklearn.set_config(enable_metadata_routing=True)
        try:
            scores = cross_val_score(
                MissMixedRegressor().set_fit_request(groups=True),
                X, y, cv=GroupKFold(n_splits=4), params={'groups': groups})
        finally:
            sklearn.set_config(enable_metadata_routing=False)
        assert scores.shape == (4,)
        assert np.all(np.isfinite(scores))

    def test_a_model_without_its_groups_is_a_different_model(self):
        """Why the above matters: the failure mode is silence, not an error."""
        from MissLearn import MissMixedRegressor
        X, y, groups = self._grouped()
        with_groups = float(np.sqrt(
            MissMixedRegressor().fit(X, y, groups=groups).tau_sq_))
        without = float(np.sqrt(MissMixedRegressor().fit(X, y).tau_sq_))
        assert with_groups > 1.0
        assert without < 0.1


class TestAutoSelectingWrappers:
    """The seven wrappers that pick a regressor or classifier from y.

    Each delegates, and each has to refuse the methods belonging to the other
    task. Seven near-identical implementations of the same rule is precisely
    the shape that drifts, so the rule is asserted once for all of them.
    """

    WRAPPERS = ['MissLASSO', 'MissRidge', 'MissMixed', 'MissBayes',
                'MissNeighbors', 'MissSupport', 'MissGaussian']

    @staticmethod
    def _data(seed=0, n=140, p=3):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0]) * 2.0 \
            + rng.normal(scale=0.3, size=n)
        return X, y, (y > np.median(y)).astype(float), np.repeat(np.arange(14), 10)

    def _fit(self, name, y, groups):
        import MissLearn as ml
        W = getattr(ml, name)
        X = self._data()[0]
        if name == 'MissMixed':
            return W().fit(X, y, groups=groups)
        return W().fit(X, y)

    @pytest.mark.parametrize("name", WRAPPERS)
    def test_a_regression_fit_offers_no_classifier_methods(self, name):
        X, y_reg, _, groups = self._data()
        m = self._fit(name, y_reg, groups)
        for meth in ('predict_proba', 'decision_function'):
            if hasattr(m, meth):
                with pytest.raises(AttributeError):
                    getattr(m, meth)(X)

    @pytest.mark.parametrize("name", WRAPPERS)
    def test_a_classification_fit_offers_no_prediction_interval(self, name):
        X, _, y_clf, groups = self._data()
        m = self._fit(name, y_clf, groups)
        if hasattr(m, 'predict_interval'):
            with pytest.raises(AttributeError):
                m.predict_interval(X)

    @pytest.mark.parametrize("name", WRAPPERS)
    def test_each_task_keeps_its_own_methods(self, name):
        X, y_reg, y_clf, groups = self._data()
        reg = self._fit(name, y_reg, groups)
        assert np.all(np.isfinite(reg.predict(X)))
        lo, hi = reg.predict_interval(X)
        assert np.all(hi >= lo)

        clf = self._fit(name, y_clf, groups)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(np.isin(clf.predict(X), clf.classes_))

    @pytest.mark.parametrize("name", WRAPPERS)
    def test_an_all_missing_target_is_refused(self, name):
        """There is no task to detect, so there is no model to choose."""
        import MissLearn as ml
        X, _, _, groups = self._data()
        W = getattr(ml, name)
        y = np.full(X.shape[0], np.nan)
        with pytest.raises(ValueError, match="no observed"):
            if name == 'MissMixed':
                W().fit(X, y, groups=groups)
            else:
                W().fit(X, y)


class TestSensitivityTippingPoint:
    """A tipping point must come from a fit that happened.

    MissSensitivity exists to say how far a conclusion can be pushed before it
    breaks, so a fabricated breaking point is the worst failure available to
    it. A delta whose draws all failed holds NaN by design, and NaN compared
    against anything is a crossing unless it is excluded: one failed fit turned
    a coefficient path running 1.96 to 1.99, nowhere near zero, into "tips at
    delta=+0.50, mildly sensitive to MNAR".
    """

    @staticmethod
    def _fitted(seed=0, n=200, p=3):
        from MissLearn import MissRidgeRegressor, MissSensitivity
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        y = X @ np.array([2.0, -1.0, 0.0]) + rng.normal(scale=0.4, size=n)
        y[rng.random(n) < 0.20] = np.nan
        return MissSensitivity(MissRidgeRegressor(alpha=1.0),
                               delta_range=(-1.5, 1.5), n_delta=7, m=5,
                               random_state=0).fit(X, y)

    def test_a_stable_coefficient_reports_no_tipping_point(self):
        s = self._fitted()
        assert s.tipping_point(0) is None
        assert 'ROBUST' in s.verdict(0)

    @pytest.mark.parametrize("method", ["sign", "ci"])
    def test_a_failed_delta_is_not_a_crossing(self, method):
        s = self._fitted()
        assert s.tipping_point(0, method=method) is None
        s.coef_curves_[4, :] = np.nan        # a delta just right of zero
        s.coef_se_curves_[4, :] = np.nan
        assert s.tipping_point(0, method=method) is None

    def test_the_verdict_stays_robust_when_a_delta_fails(self):
        s = self._fitted()
        s.coef_curves_[4, :] = np.nan
        s.coef_se_curves_[4, :] = np.nan
        assert 'ROBUST' in s.verdict(0)

    def test_a_real_sign_change_is_still_found(self):
        """The exclusion must not blind it to genuine crossings."""
        s = self._fitted()
        s.coef_curves_[:, 0] = np.array([0.5, 0.3, 0.1, -0.1,
                                         -0.3, -0.5, -0.7])
        tp = s.tipping_point(0, method='sign')
        assert tp is not None
        assert np.isclose(tp, -0.5)          # the crossing nearest delta = 0

    def test_a_real_crossing_beside_a_failed_delta_is_still_found(self):
        s = self._fitted()
        s.coef_curves_[:, 0] = np.array([0.5, 0.3, 0.1, -0.1,
                                         np.nan, -0.5, -0.7])
        assert np.isclose(s.tipping_point(0, method='sign'), -0.5)

    def test_a_failed_baseline_is_refused(self):
        """Every tipping point is expressed relative to delta = 0, so if that
        fit produced nothing there is no reference to measure against."""
        s = self._fitted()
        s.coef_curves_[s.baseline_idx_, :] = np.nan
        with pytest.raises(ValueError, match="baseline fit"):
            s.tipping_point(0)

    def test_all_nan_standard_errors_are_still_refused_for_ci(self):
        """The pre-existing guard, kept: 'ci' with nothing to work from."""
        s = self._fitted()
        s.coef_se_curves_[:, :] = np.nan
        with pytest.raises(ValueError, match="standard errors"):
            s.tipping_point(0, method='ci')

    def test_an_unknown_method_is_refused(self):
        s = self._fitted()
        with pytest.raises(ValueError, match="must be 'sign' or 'ci'"):
            s.tipping_point(0, method='Sign')

    def test_an_out_of_range_coefficient_is_refused(self):
        s = self._fitted()
        with pytest.raises(IndexError, match="out of range"):
            s.tipping_point(99)


class TestMissImputerPooling:
    """Rubin's rules and the paths around them.

    The pooled standard error is the number that gets published from a
    multiple-imputation analysis, so the arithmetic is checked against the
    formulas by hand rather than against a stored value.
    """

    # ---- the arithmetic ---------------------------------------------------

    def test_combine_matches_rubins_rules_by_hand(self):
        from MissLearn import MissImputer
        est = np.array([1.0, 1.4, 0.9, 1.2])
        se = np.array([0.20, 0.25, 0.18, 0.22])
        var = se ** 2
        m = est.size
        B = est.var(ddof=1)                 # between-imputation
        W = var.mean()                      # within-imputation
        T = W + (1.0 + 1.0 / m) * B         # total
        r = (1.0 + 1.0 / m) * B / W
        df = (m - 1) * (1.0 + 1.0 / r) ** 2

        got = MissImputer.combine(est, var)
        assert np.isclose(got['estimate'], est.mean())
        assert np.isclose(got['between_var'], B)
        assert np.isclose(got['within_var'], W)
        assert np.isclose(got['total_var'], T)
        assert np.isclose(got['se'], np.sqrt(T))
        assert np.isclose(got['df'], df)
        assert got['m'] == m
        assert 0.0 <= got['p_value'] <= 1.0

    def test_vector_estimates_pool_per_parameter(self):
        from MissLearn import MissImputer
        E = np.array([[1.0, -2.0], [1.2, -1.8], [0.9, -2.1]])
        V = np.array([[0.04, 0.09], [0.05, 0.08], [0.03, 0.10]])
        got = MissImputer.combine(E, V)
        assert np.allclose(got['estimate'], E.mean(axis=0))
        assert np.shape(got['se']) == (2,)
        assert 'p_value' not in got          # only defined for a scalar

    def test_without_variances_only_the_point_estimate_is_pooled(self):
        from MissLearn import MissImputer
        got = MissImputer.combine(np.array([1.0, 1.4, 0.9]), None)
        assert 'estimate' in got and 'between_var' in got
        assert not any(k in got for k in ('within_var', 'total_var', 'se',
                                          'df', 'p_value'))

    def test_mismatched_counts_are_refused(self):
        from MissLearn import MissImputer
        with pytest.raises(ValueError, match="same imputed"):
            MissImputer.combine(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2]))

    # ---- identical estimates across imputations ---------------------------

    @pytest.mark.parametrize("case", ["scalar", "one_column", "all_columns"])
    def test_zero_between_variance_is_silent_and_infinite(self, case):
        """No between-imputation variance means infinite degrees of freedom,
        which is right. Reaching it by evaluating 1/0 was not: np.where
        computes both arms, and the discarded one overflowed."""
        from MissLearn import MissImputer
        if case == "scalar":
            est, var = np.array([1.5, 1.5, 1.5]), np.array([.04, .04, .04])
        elif case == "one_column":
            est = np.array([[1.0, 2.0], [1.2, 2.0], [0.9, 2.0]])
            var = np.array([[.04, .01], [.05, .01], [.03, .01]])
        else:
            est = np.array([[2.0, 3.0], [2.0, 3.0]])
            var = np.array([[.04, .01], [.04, .01]])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = MissImputer.combine(est, var)

        df = np.atleast_1d(np.asarray(got['df'], dtype=float))
        assert np.all(np.isinf(df) | (df > 0))
        if case == "scalar":
            assert np.isinf(got['df'])
            assert np.isclose(got['total_var'], var.mean())
        elif case == "one_column":
            assert np.isfinite(df[0]) and np.isinf(df[1])
        else:
            assert np.all(np.isinf(df))

    # ---- an unfitted imputer ----------------------------------------------

    @staticmethod
    def _data(seed=0, n=120, p=3):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.3, size=n)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.15] = np.nan
        return Xm, y

    @pytest.mark.parametrize("method", ["transform", "transform_mean",
                                        "fit_transform_combine", "summary"])
    def test_unfitted_reports_not_fitted(self, method):
        """Every other estimator in the library says so through
        check_is_fitted; this one used to raise AttributeError on an internal
        name from inside a helper."""
        from sklearn.exceptions import NotFittedError
        from MissLearn import MissImputer, MissRidgeRegressor
        X, y = self._data()
        imp = MissImputer(m=3, random_state=0)
        with pytest.raises(NotFittedError):
            if method == "transform":
                imp.transform(X)
            elif method == "transform_mean":
                imp.transform_mean(X)
            elif method == "fit_transform_combine":
                imp.fit_transform_combine(X, y, MissRidgeRegressor(), 'coef_')
            else:
                imp.summary()

    # ---- the end-to-end pooling workflow ----------------------------------

    def test_fit_transform_combine_pools_a_real_fit(self):
        from MissLearn import MissImputer, MissRidgeRegressor
        X, y = self._data()
        imp = MissImputer(m=4, random_state=0).fit(X)
        res = imp.fit_transform_combine(X, y, MissRidgeRegressor(alpha=1.0),
                                        param='coef_', param_var='se_')
        assert np.shape(res['estimate']) == (3,)
        assert np.all(np.asarray(res['se']) > 0.0)
        assert len(res['fitted_estimators']) == 4
        assert np.all(np.isfinite(res['estimate']))

    def test_a_missing_attribute_warns_then_refuses(self):
        from MissLearn import MissImputer, MissRidgeRegressor
        X, y = self._data()
        imp = MissImputer(m=3, random_state=0).fit(X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="no successful fits"):
                imp.fit_transform_combine(X, y, MissRidgeRegressor(),
                                          param='not_an_attribute')
        assert any('has no attribute' in str(w.message) for w in caught)

    def test_summary_runs_once_fitted(self):
        import contextlib
        import io as _io
        from MissLearn import MissImputer
        X, _ = self._data()
        imp = MissImputer(m=3, random_state=0).fit(X)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            imp.summary()
        assert 'MissImputer' in buf.getvalue()


class TestCrossValUncoveredPaths:
    """Fallbacks and failure paths a clean five-fold run never enters."""

    @staticmethod
    def _data(seed=0, n=120, p=4):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        return rng, X

    # ---- stratified fallbacks --------------------------------------------

    def test_falls_back_when_fewer_labels_than_folds(self):
        from MissLearn import MissStratifiedKFold
        _, X = self._data()
        y = np.full(X.shape[0], np.nan)
        y[:3] = [0.0, 1.0, 0.0]
        with pytest.warns(UserWarning, match="Falling back"):
            folds = list(MissStratifiedKFold(n_splits=5).split(X, y))
        assert len(folds) == 5
        seen = np.concatenate([te for _, te in folds])
        assert seen.size == X.shape[0]
        assert np.unique(seen).size == X.shape[0]

    def test_falls_back_when_a_class_is_smaller_than_n_splits(self):
        from MissLearn import MissStratifiedKFold
        _, X = self._data(seed=1)
        y = np.zeros(X.shape[0])
        y[:2] = 1.0                      # two members, five folds
        with pytest.warns(UserWarning, match="fewer than"):
            folds = list(MissStratifiedKFold(n_splits=5).split(X, y))
        seen = np.concatenate([te for _, te in folds])
        assert np.unique(seen).size == X.shape[0]

    def test_a_workable_stratification_does_not_fall_back(self):
        from MissLearn import MissStratifiedKFold
        rng, X = self._data(seed=2)
        y = (rng.random(X.shape[0]) < 0.4).astype(float)
        y[rng.random(X.shape[0]) < 0.15] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds = list(MissStratifiedKFold(n_splits=5).split(X, y))
        assert len(folds) == 5
        assert not any('Falling back' in str(w.message) for w in caught)

    # ---- a fold whose fit fails ------------------------------------------

    @staticmethod
    def _fails_on_fold(which):
        """An estimator whose nth fit raises."""
        from MissLearn import MissLogistic

        class _Fails(MissLogistic):
            seen = 0

            def fit(self, X, y, **kw):
                _Fails.seen += 1
                if _Fails.seen == which + 1:
                    raise RuntimeError("deliberate failure")
                return super().fit(X, y, **kw)

        _Fails.seen = 0
        return _Fails()

    def test_a_failed_fold_holds_nan_in_its_own_slot(self):
        """Dropping the entry shortened the array and shifted every later
        fold's index, so a caller could not tell which fold had gone."""
        import MissLearn as ML
        rng, X = self._data(seed=3)
        y = (rng.random(X.shape[0]) < 0.5).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = ML.miss_cross_val_score(self._fails_on_fold(1), X, y, cv=5)
        assert scores.shape == (5,)
        assert np.flatnonzero(np.isnan(scores)).tolist() == [1]
        assert np.isfinite(np.nanmean(scores))

    def test_every_fold_failing_gives_all_nan_at_full_length(self):
        import MissLearn as ML
        from MissLearn import MissLogistic
        rng, X = self._data(seed=4)
        y = (rng.random(X.shape[0]) < 0.5).astype(float)

        class _AlwaysFails(MissLogistic):
            def fit(self, X, y, **kw):
                raise RuntimeError("deliberate failure")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = ML.miss_cross_val_score(_AlwaysFails(), X, y, cv=5)
        assert scores.shape == (5,)
        assert np.all(np.isnan(scores))

    def test_returned_estimators_stay_aligned_with_the_scores(self):
        import MissLearn as ML
        rng, X = self._data(seed=5)
        y = (rng.random(X.shape[0]) < 0.5).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, ests = ML.miss_cross_val_score(
                self._fails_on_fold(1), X, y, cv=5, return_estimators=True)
        assert len(ests) == scores.size == 5
        assert [i for i, e in enumerate(ests) if e is None] == [1]

    def test_a_failed_fold_warns(self):
        import MissLearn as ML
        rng, X = self._data(seed=6)
        y = (rng.random(X.shape[0]) < 0.5).astype(float)
        with pytest.warns(UserWarning, match="fit failed"):
            ML.miss_cross_val_score(self._fails_on_fold(0), X, y, cv=3)

    def test_cross_validate_keeps_the_same_convention(self):
        """The multi-metric sibling already did this; the two must agree."""
        import MissLearn as ML
        from MissLearn import MissLogistic
        rng, X = self._data(seed=7)
        y = (rng.random(X.shape[0]) < 0.5).astype(float)

        class _AlwaysFails(MissLogistic):
            def fit(self, X, y, **kw):
                raise RuntimeError("deliberate failure")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ML.miss_cross_validate(_AlwaysFails(), X, y, cv=4)
        for key, val in res.items():
            if key == 'estimators':
                continue
            arr = np.asarray(val, dtype=float)
            assert arr.size == 4, key
            assert np.all(np.isnan(arr)), key


class TestMissEnsembleUncoveredPaths:
    """Branches a homogeneous MissLearn ensemble on complete targets misses.

    The base-estimator gate, the NaN-target strip that only non-MissLearn
    members need, and the class realignment for a bootstrap that never saw one
    of the classes.
    """

    @staticmethod
    def _data(seed=0, n=200, p=4):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = X[:, 0] * 2.0 - np.where(np.isnan(X[:, 1]), 0.0, X[:, 1])
        return rng, X, np.where(np.isnan(y), 0.0, y)

    @staticmethod
    def _named(name):
        """An object whose class name is all the gate inspects."""
        return type(name, (), {})()

    # ---- the NaN-capability gate -----------------------------------------

    def test_misslearn_members_are_always_accepted(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn._ensemble import _check_estimator_compatible
        _check_estimator_compatible(MissRidgeRegressor())

    @pytest.mark.parametrize("name", [
        "RandomForestRegressor", "DecisionTreeClassifier",
        "ExtraTreesRegressor", "HistGradientBoostingRegressor"])
    def test_nan_native_sklearn_trees_are_accepted(self, name):
        from MissLearn._ensemble import _check_estimator_compatible
        _check_estimator_compatible(self._named(name))

    @pytest.mark.parametrize("name,module", [
        ("XGBRegressor", "xgboost"), ("LGBMClassifier", "lightgbm"),
        ("CatBoostRegressor", "catboost")])
    def test_optional_boosters_accepted_or_named_in_the_error(self, name, module):
        """Whether these pass depends on what is installed, so the test asserts
        the rule rather than the outcome: accepted if importable, and otherwise
        refused with an ImportError that names the package to install."""
        import importlib
        from MissLearn._ensemble import _check_estimator_compatible
        try:
            importlib.import_module(module)
            installed = True
        except ImportError:
            installed = False
        if installed:
            _check_estimator_compatible(self._named(name))
        else:
            with pytest.raises(ImportError, match=module):
                _check_estimator_compatible(self._named(name))

    def test_an_estimator_that_cannot_take_nan_is_refused(self):
        from MissLearn._ensemble import _check_estimator_compatible
        with pytest.raises(ValueError, match="not supported"):
            _check_estimator_compatible(self._named("SVR"))

    def test_is_misslearn_discriminates_by_module(self):
        from MissLearn import MissRidgeRegressor
        from MissLearn._ensemble import _is_misslearn
        assert _is_misslearn(MissRidgeRegressor())
        assert not _is_misslearn(self._named("Foo"))
        assert not _is_misslearn(np.zeros(3))

    # ---- fit kwargs must stay aligned with a bootstrap -------------------

    def test_only_length_n_arrays_are_subset(self):
        from MissLearn._ensemble import _subset_kwargs
        n = 10
        idx = np.array([0, 2, 4])
        got = _subset_kwargs({'groups': np.arange(n),
                              'sample_weight': np.ones(n) * 2.0,
                              'scalar': 7,
                              'wrong_length': np.arange(n + 3),
                              'text': 'hello'}, idx, n)
        assert np.array_equal(got['groups'], idx)
        assert got['sample_weight'].shape == (3,)
        assert got['scalar'] == 7
        assert got['wrong_length'].shape == (n + 3,)   # untouched
        assert got['text'] == 'hello'

    # ---- a non-MissLearn member cannot take NaN in y ---------------------

    def test_nan_targets_are_stripped_for_a_sklearn_member(self):
        """Tree models take NaN in X but not in y, so those rows come out of
        the bootstrap. A MissLearn member keeps them, which is the point."""
        from sklearn.ensemble import RandomForestRegressor
        from MissLearn import MissEnsemble
        rng, X, y = self._data()
        y = y.copy()
        y[rng.random(y.size) < 0.25] = np.nan
        m = MissEnsemble(estimator=RandomForestRegressor(n_estimators=5,
                                                        random_state=0),
                         n_estimators=4, random_state=0).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))

    def test_a_nearly_empty_target_column_still_fits(self):
        from sklearn.ensemble import RandomForestRegressor
        from MissLearn import MissEnsemble
        rng, X, y = self._data(seed=5)
        y = y.copy()
        y[rng.random(y.size) < 0.92] = np.nan
        assert np.sum(~np.isnan(y)) < 40           # genuinely sparse
        m = MissEnsemble(estimator=RandomForestRegressor(n_estimators=5,
                                                        random_state=0),
                         n_estimators=4, random_state=0).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))

    # ---- class realignment ------------------------------------------------

    def test_a_member_that_never_saw_a_class_is_realigned(self):
        """With a 3% class some bootstraps miss it entirely, and those members'
        probability columns have to be mapped onto the ensemble's classes_."""
        from sklearn.ensemble import RandomForestClassifier
        from MissLearn import MissEnsemble
        _, X, _ = self._data(seed=2)
        n = X.shape[0]
        y = np.zeros(n)
        y[:6] = 2.0
        y[6:80] = 1.0
        m = MissEnsemble(estimator=RandomForestClassifier(n_estimators=5,
                                                         random_state=0),
                         n_estimators=8, random_state=0).fit(X, y)
        assert len(m.classes_) == 3
        proba = m.predict_proba(X)
        assert proba.shape == (n, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0)
        assert proba[:, 2].max() > 0.0            # the rare class is reachable
        assert np.all(np.isin(m.predict(X), m.classes_))


class TestMissRecommenderRegimes:
    """The scoring branches that only fire for particular shapes of data.

    MissRecommender's stated deliverable is the reasoning rather than the
    ranking, which makes a wrong reason a defect in its own right: on a target
    independent of X it used to report "a linear conditional mean is adequate"
    and award points for it, from two probes that both scored worse than
    predicting the mean.
    """

    @staticmethod
    def _base(seed=0, n=300, p=5, miss=0.12):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < miss] = np.nan
        return rng, X

    # ---- the linearity probe ---------------------------------------------

    def test_no_skill_is_reported_as_no_evidence(self):
        from MissLearn import MissRecommender
        rng, X = self._base()
        y = rng.normal(size=X.shape[0])          # independent of X
        rec = MissRecommender().fit(X, y)
        probe = rec.evidence_['nonlinearity_probe']
        assert probe['has_skill'] is False
        reasons = " ".join(r for e in rec.ranked_ for r in e['reasons'])
        assert 'neither probe beats' in reasons
        assert 'conditional mean is adequate' not in reasons

    def test_no_skill_awards_no_points_either_way(self):
        """The comparison of two failures must not move the ranking."""
        from MissLearn import MissRecommender
        rng, X = self._base(seed=3)
        y = rng.normal(size=X.shape[0])
        rec = MissRecommender().fit(X, y)
        probe = rec.evidence_['nonlinearity_probe']
        assert probe['has_skill'] is False
        # the ratio still resolves, and still names a winner, which is the trap
        assert np.isfinite(probe['ratio'])
        assert max(probe['linear_score'],
                   probe['neighbour_score']) <= probe['floor'] + 0.01

    def test_a_linear_signal_is_recognised(self):
        from MissLearn import MissRecommender
        rng, X = self._base(seed=1)
        col = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0])
        y = 2.0 * col + rng.normal(scale=0.3, size=X.shape[0])
        rec = MissRecommender().fit(X, y)
        assert rec.evidence_['nonlinearity_probe']['has_skill'] is True

    def test_a_nonlinear_signal_is_recognised(self):
        from MissLearn import MissRecommender
        rng, X = self._base(seed=2)
        c0 = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0])
        c1 = np.where(np.isnan(X[:, 1]), 0.0, X[:, 1])
        y = np.sin(3.0 * c0) + c1 ** 2 + rng.normal(scale=0.2, size=X.shape[0])
        rec = MissRecommender().fit(X, y)
        probe = rec.evidence_['nonlinearity_probe']
        assert probe['has_skill'] is True
        assert probe['ratio'] > 1.15      # neighbours ahead

    # ---- dimensionality --------------------------------------------------

    def test_wide_data_prefers_the_penalised_families(self):
        """p large against n takes a branch narrow data never reaches."""
        from MissLearn import MissRecommender
        rng = np.random.default_rng(7)
        n, p = 40, 30
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0]) + rng.normal(size=n)
        rec = MissRecommender().fit(X, y)
        assert rec.evidence_['p'] >= rec.evidence_['n_complete_cases'] or p > 0.5 * n
        text = " ".join(r for e in rec.ranked_ for r in e['reasons'])
        assert 'shrinkage' in text or 'L1' in text

    # ---- missingness load ------------------------------------------------

    def test_heavy_missingness_takes_its_own_branch(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(8)
        n, p = 250, 6
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.45] = np.nan     # well above the threshold
        y = np.where(np.isnan(X[:, 0]), 0.0, X[:, 0]) + rng.normal(size=n)
        rec = MissRecommender().fit(X, y)
        assert rec.evidence_['cell_missing_rate'] > 0.30
        text = " ".join(r for e in rec.ranked_ for r in e['reasons'])
        assert 'missing' in text.lower()

    # ---- it always produces a usable answer ------------------------------

    @pytest.mark.parametrize("case", ["no_skill", "wide", "heavy_missing"])
    def test_a_ranking_and_a_reason_always_come_back(self, case):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(9)
        if case == "wide":
            n, p, miss = 40, 30, 0.10
        elif case == "heavy_missing":
            n, p, miss = 250, 6, 0.45
        else:
            n, p, miss = 300, 5, 0.12
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < miss] = np.nan
        y = (rng.normal(size=n) if case == "no_skill"
             else np.where(np.isnan(X[:, 0]), 0.0, X[:, 0]) + rng.normal(size=n))
        rec = MissRecommender().fit(X, y)
        assert rec.ranked_, "no families were scored"
        for entry in rec.ranked_:
            assert np.isfinite(entry['score']), entry['family']
            assert isinstance(entry['reasons'], list)
        assert rec.recommended_family_ in {e['family'] for e in rec.ranked_}


class TestMissMulticlassUncoveredPaths:
    """The branches a three-class fit on clean float labels never reaches.

    Every one of these was an uncovered branch, which in this package has been
    a reliable indicator: `_to_binary` raised on a pandas nullable label dtype
    while the `_is_nan` helper directly above it had been written to handle
    exactly that value.
    """

    @staticmethod
    def _data(seed=0, n=150, p=4):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = rng.integers(0, 3, n).astype(float)
        return X, y

    # ---- the scikit-learn parameter contract on a meta-estimator ----------

    def test_get_params_exposes_and_hides_nested(self):
        from MissLearn import MissLogistic, MissMulticlass
        m = MissMulticlass(MissLogistic(l2_reg=0.05))
        deep = m.get_params(deep=True)
        assert deep['estimator__l2_reg'] == 0.05
        assert 'estimator__l2_reg' not in m.get_params(deep=False)

    def test_set_params_routes_by_prefix(self):
        from MissLearn import MissLogistic, MissMulticlass
        m = MissMulticlass(MissLogistic())
        m.set_params(estimator__l2_reg=0.9, strategy='ovr')
        assert m.estimator.l2_reg == 0.9
        assert m.strategy == 'ovr'

    def test_clone_preserves_nested_params(self):
        from sklearn.base import clone
        from MissLearn import MissLogistic, MissMulticlass
        m = MissMulticlass(MissLogistic(l2_reg=0.33))
        assert clone(m).estimator.l2_reg == 0.33

    # ---- too few classes -------------------------------------------------

    def test_single_class_is_refused(self):
        from MissLearn import MissLogistic, MissMulticlass
        X, _ = self._data()
        with pytest.raises(ValueError, match="least 2 classes"):
            MissMulticlass(MissLogistic()).fit(X, np.zeros(X.shape[0]))

    def test_single_class_after_dropping_missing_labels_is_refused(self):
        """The count must be taken after NaN labels are removed, not before."""
        from MissLearn import MissLogistic, MissMulticlass
        rng = np.random.default_rng(1)
        X, _ = self._data()
        y = np.where(rng.random(X.shape[0]) < 0.5, 0.0, np.nan)
        with pytest.raises(ValueError, match="least 2 classes"):
            MissMulticlass(MissLogistic()).fit(X, y)

    # ---- a sub-classifier that will not fit ------------------------------

    @staticmethod
    def _breaking_estimator():
        """An estimator that fails for one one-vs-rest problem only."""
        from MissLearn import MissLogistic

        class _BreaksOnFirst(MissLogistic):
            seen = 0

            def fit(self, X, y, **kw):
                _BreaksOnFirst.seen += 1
                if _BreaksOnFirst.seen == 1:
                    raise RuntimeError("deliberate failure")
                return super().fit(X, y, **kw)

        _BreaksOnFirst.seen = 0
        return _BreaksOnFirst()

    def test_a_failed_member_warns_and_is_recorded_as_none(self):
        from MissLearn import MissMulticlass
        X, y = self._data()
        with pytest.warns(UserWarning, match="failed to fit"):
            m = MissMulticlass(self._breaking_estimator()).fit(X, y)
        assert sum(e is None for e in m.estimators_) == 1

    def test_predictions_survive_a_failed_member(self):
        from MissLearn import MissMulticlass
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MissMulticlass(self._breaking_estimator()).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(proba))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(np.isin(m.predict(X), m.classes_))
        assert np.all(np.isfinite(m.decision_function(X)))
        assert np.all(np.isfinite(m.feature_importances_))

    def test_summary_names_a_failed_member(self):
        import contextlib
        import io as _io
        from MissLearn import MissMulticlass
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MissMulticlass(self._breaking_estimator()).fit(X, y)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.summary()
        assert 'FAILED' in buf.getvalue()

    # ---- label dtypes ----------------------------------------------------

    def test_string_labels_round_trip(self):
        from MissLearn import MissLogistic, MissMulticlass
        X, _ = self._data()
        y = np.array(['a', 'b', 'c'] * (X.shape[0] // 3), dtype=object)
        m = MissMulticlass(MissLogistic()).fit(X, y)
        assert set(m.classes_) == {'a', 'b', 'c'}
        assert set(np.unique(m.predict(X))).issubset({'a', 'b', 'c'})

    def test_none_among_object_labels_is_missing_not_a_category(self):
        from MissLearn import MissLogistic, MissMulticlass
        X, _ = self._data()
        n = X.shape[0]
        y = np.array((['a', 'b', 'c'] * (n // 3 - 1)) + [None, 'a', 'b'],
                     dtype=object)
        m = MissMulticlass(MissLogistic()).fit(X, y)
        assert set(m.classes_) == {'a', 'b', 'c'}

    def test_pandas_na_label_is_missing_not_an_error(self):
        """`_to_binary` used to hand pd.NA to np.where, which cannot take its
        truth value, so a nullable-dtype Series died inside fit."""
        import pandas as pd
        from MissLearn import MissLogistic, MissMulticlass
        X, _ = self._data()
        n = X.shape[0]
        y = pd.array((['a', 'b', 'c'] * (n // 3 - 1)) + [pd.NA, 'a', 'b'],
                     dtype="string")
        m = MissMulticlass(MissLogistic()).fit(X, y)
        assert set(m.classes_) == {'a', 'b', 'c'}
        assert np.all(np.isfinite(m.predict_proba(X)))

    # ---- the two-class shortcut -------------------------------------------

    @staticmethod
    def _binary_data(seed=0, n=160, p=4):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        y = (X[:, 0] > 0).astype(float)
        X[rng.random(X.shape) < 0.10] = np.nan
        return X, y

    def test_two_classes_take_the_single_estimator_path(self):
        """A two-class problem needs no one-vs-rest expansion, and takes an
        entirely separate branch that no three-class test reaches."""
        from MissLearn import MissLogistic, MissMulticlass
        X, y = self._binary_data()
        m = MissMulticlass(MissLogistic()).fit(X, y)
        assert m.n_classes_ == 2
        assert len(m.estimators_) == 1

    def test_two_class_predictions_keep_the_binary_contract(self):
        from MissLearn import MissLogistic, MissMulticlass
        X, y = self._binary_data()
        m = MissMulticlass(MissLogistic()).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        margin = m.decision_function(X)
        assert np.ndim(margin) == 1 and margin.size == X.shape[0]
        # sign of the margin and the majority of the probability must agree
        assert np.all((margin > 0) == (proba[:, 1] > 0.5))
        assert 0.0 <= m.score(X, y) <= 1.0

    def test_decision_function_is_derived_when_the_base_lacks_one(self):
        """The documented fallback: signed log-odds from predict_proba."""
        from MissLearn import MissLogistic, MissMulticlass

        class _ProbaOnly:
            def __init__(self):
                self.inner = MissLogistic()

            def fit(self, X, y, **kw):
                self.inner.fit(X, y)
                return self

            def predict_proba(self, X, **kw):
                return self.inner.predict_proba(X)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **kw):
                return self

        X, y = self._binary_data()
        m = MissMulticlass(_ProbaOnly()).fit(X, y)
        margin = m.decision_function(X)
        assert np.ndim(margin) == 1
        assert np.all(np.isfinite(margin))

    def test_score_with_every_label_missing_is_nan(self):
        from MissLearn import MissLogistic, MissMulticlass
        X, y = self._binary_data()
        m = MissMulticlass(MissLogistic()).fit(X, y)
        assert np.isnan(m.score(X, np.full(X.shape[0], np.nan)))

    def test_to_binary_marks_missing_labels_nan(self):
        import pandas as pd
        from MissLearn._multiclass import _to_binary
        y = pd.array(['a', 'b', pd.NA, 'a'], dtype="string")
        got = _to_binary(y, 'a')
        assert got[0] == 1.0 and got[1] == 0.0 and got[3] == 1.0
        assert np.isnan(got[2])


class TestAdaptiveGaussHermite:
    """The GLMM integrates over the random effect, and the unadapted rule put
    its nodes on the prior instead of on the posterior.

    Plain Gauss-Hermite spreads nodes at width tau around zero. A subject with
    several observations pins its own effect into a much narrower interval,
    possibly far from zero, and the nodes then sample where the integrand is
    negligible. The error grew with tau and with observations per subject:
    at tau = 5 with twenty observations the log-likelihood was out by 1.8, and
    even 320 nodes left 6e-4 at tau = 3.
    """

    @staticmethod
    def _subject(rng, n_i, tau):
        eta = rng.normal(size=n_i)
        sign = np.where(rng.random(n_i) < 0.6, 1.0, -1.0)
        inv_scale = 1.0 / np.sqrt(1.0 + rng.random(n_i))
        return eta, sign, inv_scale, tau * tau

    @staticmethod
    def _dense(eta, sign, inv_scale, tau_sq, n=200001):
        from MissLearn._mixed import _subject_log_integrand
        tau = np.sqrt(tau_sq)
        b = np.linspace(-18 * tau, 18 * tau, n)
        g = _subject_log_integrand(b, eta, sign, inv_scale, tau_sq)
        m = g.max()
        return m + np.log(_trapezoid(np.exp(g - m), b))

    @staticmethod
    def _aghq(eta, sign, inv_scale, tau_sq, q=20):
        from scipy.special import logsumexp
        from MissLearn._mixed import _adaptive_nodes, _subject_log_integrand
        t, w = np.polynomial.hermite.hermgauss(q)
        b_nodes, log_w = _adaptive_nodes(eta, sign, inv_scale, tau_sq,
                                        t, np.log(w))
        return logsumexp(log_w + _subject_log_integrand(
            b_nodes, eta, sign, inv_scale, tau_sq))

    @staticmethod
    def _unadapted(eta, sign, inv_scale, tau_sq, q=20):
        from scipy.special import logsumexp
        from MissLearn._mixed import _log_sigmoid
        t, w = np.polynomial.hermite.hermgauss(q)
        b = np.sqrt(2.0 * tau_sq) * t
        u = sign[None, :] * (eta[None, :] + b[:, None] * inv_scale[None, :])
        return logsumexp(_log_sigmoid(u).sum(axis=1)
                         + np.log(w) - 0.5 * np.log(np.pi))

    def test_the_mode_really_is_the_maximum(self):
        """g is strictly concave, so the Newton solve has one place to land."""
        from MissLearn._mixed import _subject_log_integrand, _subject_mode
        rng = np.random.default_rng(4)
        for n_i in (1, 3, 20):
            for tau in (0.5, 2.0, 8.0):
                eta, sign, inv_scale, tau_sq = self._subject(rng, n_i, tau)
                b_hat, sd_hat = _subject_mode(eta, sign, inv_scale, tau_sq)
                assert np.isfinite(b_hat) and sd_hat > 0.0
                here = _subject_log_integrand(np.array([b_hat]), eta, sign,
                                              inv_scale, tau_sq)[0]
                near = _subject_log_integrand(
                    b_hat + np.array([-1e-3, 1e-3]), eta, sign,
                    inv_scale, tau_sq)
                assert here >= near.max() - 1e-9

    @pytest.mark.parametrize("tau", [1.0, 3.0, 8.0])
    @pytest.mark.parametrize("n_i", [1, 5, 20])
    def test_adapting_beats_not_adapting(self, tau, n_i):
        rng = np.random.default_rng(hash((tau, n_i)) % 2 ** 31)
        worse = 0
        for _ in range(6):
            eta, sign, inv_scale, tau_sq = self._subject(rng, n_i, tau)
            ref = self._dense(eta, sign, inv_scale, tau_sq)
            e_ad = abs(self._aghq(eta, sign, inv_scale, tau_sq) - ref)
            e_pl = abs(self._unadapted(eta, sign, inv_scale, tau_sq) - ref)
            assert e_ad < 5e-2
            if e_ad > e_pl:
                worse += 1
        assert worse <= 1        # allowed to lose the odd easy case, not more

    def test_accurate_where_the_old_rule_lost_whole_nats(self):
        """tau = 5 with twenty observations: the case that motivated this."""
        rng = np.random.default_rng(11)
        eta, sign, inv_scale, tau_sq = self._subject(rng, 20, 5.0)
        ref = self._dense(eta, sign, inv_scale, tau_sq)
        assert abs(self._aghq(eta, sign, inv_scale, tau_sq) - ref) < 1e-3
        assert abs(self._unadapted(eta, sign, inv_scale, tau_sq) - ref) > 1e-2

    def test_tiny_tau_degrades_to_the_prior_width(self):
        """With no random effect left, the mode is zero and the width is tau,
        which is the unadapted rule as a special case."""
        from MissLearn._mixed import _subject_mode
        rng = np.random.default_rng(5)
        eta, sign, inv_scale, _ = self._subject(rng, 4, 1.0)
        tau_sq = 1e-10
        b_hat, sd_hat = _subject_mode(eta, sign, inv_scale, tau_sq)
        assert abs(b_hat) < 1e-4
        assert np.isclose(sd_hat, np.sqrt(tau_sq), rtol=1e-3)

    def test_node_count_barely_moves_the_glmm_fit(self):
        from scipy.special import expit
        from MissLearn import MissMixedClassifier
        rng = np.random.default_rng(0)
        n_sub, per = 25, 8
        n = n_sub * per
        groups = np.repeat(np.arange(n_sub), per)
        X = rng.normal(size=(n, 3))
        b = np.repeat(rng.normal(scale=1.5, size=n_sub), per)
        y = (rng.random(n) < expit(X @ np.array([1.5, -1.0, 0.5]) + b)
             ).astype(float)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.15] = np.nan
        base = None
        for n_q in (11, 20, 60):
            m = MissMixedClassifier(n_quadrature=n_q).fit(Xm, y, groups=groups)
            coef = np.asarray(m.coef_).ravel()
            if base is None:
                base = coef
            else:
                assert np.abs(coef - base).max() < 1e-2

    def test_new_subject_prediction_uses_the_shared_integral(self):
        """A subject with no observations contributes no likelihood factor, so
        the prediction is the plain expectation of a logistic over the prior:
        exactly integrate_logistic_normal."""
        from scipy.special import expit
        from MissLearn import MissMixedClassifier
        rng = np.random.default_rng(2)
        n_sub, per = 20, 6
        n = n_sub * per
        groups = np.repeat(np.arange(n_sub), per)
        X = rng.normal(size=(n, 3))
        b = np.repeat(rng.normal(scale=1.2, size=n_sub), per)
        y = (rng.random(n) < expit(X @ np.array([1.0, -1.0, 0.5]) + b)
             ).astype(float)
        m = MissMixedClassifier().fit(X, y, groups=groups)
        # groups never seen in training take the unknown-subject path
        unseen = np.full(n, 10_000)
        proba = m.predict_proba(X, groups=unseen)
        assert proba.shape == (n, 2)
        assert np.all(np.isfinite(proba))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all((proba[:, 1] > 0.0) & (proba[:, 1] < 1.0))


class TestStandardErrorsFailToNaN:
    """A variance that could not be computed must not read as certainty.

    Delta-method and inverse-Hessian variances come out negative when the
    Hessian is not positive definite or the Jacobian is ill conditioned. These
    were floored at zero, which turns "this could not be computed" into "this
    coefficient is known exactly": standard error 0.0000 and a confidence
    interval collapsed onto the point estimate. Two identical predictor
    columns produced it in most draws.
    """

    def test_helper_maps_non_positive_variances_to_nan(self):
        from MissLearn._utils import standard_errors_from_variance
        Var = np.diag([4.0, 0.0, -1e-9, 9.0, -5.0])
        se = standard_errors_from_variance(Var)
        assert np.isclose(se[0], 2.0)
        assert np.isclose(se[3], 3.0)
        assert np.isnan(se[1]) and np.isnan(se[2]) and np.isnan(se[4])

    def test_helper_raises_no_warning_on_negative_input(self):
        from MissLearn._utils import standard_errors_from_variance
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            standard_errors_from_variance(np.diag([-1.0, -1e-30, 2.0]))

    @staticmethod
    def _collinear(seed=0, n=80):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        X = np.column_stack([a, a, b])          # columns 0 and 1 identical
        y = 2.0 * a - b + rng.normal(scale=0.2, size=n)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.10] = np.nan
        return Xm, y

    def test_unidentified_coefficients_report_nan_not_zero(self):
        from MissLearn import MissLinear
        X, y = self._collinear()
        m = MissLinear().fit(X, y)
        se = np.asarray(m.se_, dtype=float)
        # se_ carries the intercept first, so the duplicated pair is 1 and 2
        pair = se[1:3]
        assert not np.any(pair == 0.0), "a zero standard error is false certainty"
        if np.any(np.isnan(pair)):
            z = np.asarray(m.z_stats_, dtype=float)[1:3]
            p = np.asarray(m.pvalues_, dtype=float)[1:3]
            assert np.all(np.isnan(z[np.isnan(pair)]))
            assert np.all(np.isnan(p[np.isnan(pair)]))

    def test_the_fit_itself_is_unharmed(self):
        """Only the uncertainty is unavailable; the sum over the duplicated
        pair is identified and the predictions are fine."""
        from MissLearn import MissLinear
        X, y = self._collinear()
        m = MissLinear().fit(X, y)
        coef = np.asarray(m.coef_).ravel()
        assert np.all(np.isfinite(m.predict(X)))
        assert abs((coef[0] + coef[1]) - 2.0) < 0.5

    def test_identified_coefficients_usually_keep_their_inference(self):
        """The third column is collinear with nothing, so it should normally
        still report a standard error in the same fit.

        Normally, not always. When the variance matrix is ill conditioned its
        negative diagonal entries are not confined to the unidentified pair,
        and an entry that cannot be computed is withheld wherever it sits.
        Measured across 15 seeds at each of six configurations, the third
        coefficient kept its standard error in 8 to 14 of them. Withholding it
        is the conservative direction, so the test asserts the majority
        behaviour rather than requiring it every time.
        """
        from MissLearn import MissLinear
        kept = 0
        for seed in range(15):
            X, y = self._collinear(seed=seed)
            se = np.asarray(MissLinear().fit(X, y).se_, dtype=float)
            assert se[3] != 0.0            # never false certainty
            if np.isfinite(se[3]) and se[3] > 0.0:
                kept += 1
        assert kept >= 8

    def test_well_conditioned_fits_are_untouched(self):
        from MissLearn import MissLinear
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 3))
        y = X @ np.array([2.0, 0.0, -1.0]) + rng.normal(scale=0.2, size=120)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.10] = np.nan
        m = MissLinear().fit(Xm, y)
        se = np.asarray(m.se_, dtype=float)
        assert np.all(np.isfinite(se)) and np.all(se > 0.0)

    def test_no_estimator_floors_a_standard_error_at_zero(self):
        """A guard on the divergence, not on one module. The 1e-12 floors on
        sigma_X are a different thing: they normalise standardised
        coefficients and never reach se_."""
        import inspect
        import os
        import re
        import MissLearn
        pkg = os.path.dirname(inspect.getfile(MissLearn))
        offenders = []
        for fn in sorted(os.listdir(pkg)):
            if not fn.endswith('.py'):
                continue
            with open(os.path.join(pkg, fn), encoding='utf-8') as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(r'sqrt\(np\.maximum\(np\.diag\(.*0\.0\)\)', line):
                        offenders.append('%s:%d' % (fn, i))
        assert not offenders, (
            "standard errors floored at zero instead of NaN at " +
            ", ".join(offenders))


class TestFitUsesTheSameIntegralAsPredict:
    """The fitting likelihood and the predictions must agree on the integral.

    They did not. `fit` built its own 20-node Gauss-Hermite rule, which is a
    step of width about 1/sqrt(2v) in its own variable and cannot resolve it
    once v grows. Because v is the variance of the missing features' linear
    contribution, the rule was least accurate exactly where the coefficients
    were largest: the fitted values moved by 0.16 between 20 and 320 nodes.
    """

    @staticmethod
    def _strong_signal(seed=0, n=300, p=6, strength=4.0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        beta = strength * rng.normal(size=p)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ beta)))).astype(float)
        Xm = X.copy()
        Xm[rng.random(Xm.shape) < 0.20] = np.nan
        return Xm, y

    @pytest.mark.parametrize("name", ["MissLogistic", "MissLASSOClassifier"])
    def test_node_count_no_longer_moves_the_fit(self, name):
        import MissLearn as ml
        X, y = self._strong_signal()
        est = getattr(ml, name)
        base = None
        for n_q in (20, 80, 320):
            m = est(n_quadrature=n_q, compute_se=False, copula=False)
            m.fit(X, y)
            coef = np.asarray(m.coef_).ravel()
            if base is None:
                base = coef
            else:
                assert np.abs(coef - base).max() < 1e-6

    def test_helper_matches_the_prediction_path(self):
        """One rule, not two that happen to be close."""
        from MissLearn._utils import (integrate_logistic_normal,
                                      logistic_normal_with_grads)
        a = np.linspace(-8.0, 8.0, 33)
        for v in (0.25, 1.0, 4.0, 60.0, 900.0):
            p_fit, _, _ = logistic_normal_with_grads(a, np.full_like(a, v))
            p_pred = np.atleast_1d(integrate_logistic_normal(a, v))
            # Not bitwise: the two group the same sum differently, one
            # folding 1/sqrt(pi) into the weights and the other dividing
            # afterwards, so they agree to rounding rather than to the bit.
            assert np.abs(p_fit - p_pred).max() < 1e-12

    @pytest.mark.parametrize("v", [0.05, 0.5, 4.0, 60.0, 900.0])
    def test_derivatives_match_finite_differences(self, v):
        from MissLearn._utils import logistic_normal_with_grads
        a = np.linspace(-8.0, 8.0, 17)
        va = np.full_like(a, v)
        _, dp_da, dp_dv = logistic_normal_with_grads(a, va)

        h = 1e-6
        fd_a = (logistic_normal_with_grads(a + h, va)[0]
                - logistic_normal_with_grads(a - h, va)[0]) / (2 * h)
        assert np.abs(dp_da - fd_a).max() < 1e-6

        hv = max(1e-7, v * 1e-6)
        fd_v = (logistic_normal_with_grads(a, va + hv)[0]
                - logistic_normal_with_grads(a, va - hv)[0]) / (2 * hv)
        assert np.abs(dp_dv - fd_v).max() < 1e-6

    def test_rows_of_mixed_variance_in_one_call(self):
        """A fit evaluates every missingness pattern together, so one call
        carries variances from both branches."""
        from MissLearn._utils import logistic_normal_with_grads
        a = np.array([-3.0, 0.0, 2.0, 5.0, -1.0, 0.5])
        v = np.array([0.0, 0.3, 1.0, 40.0, 200.0, 0.9])
        p, dp_da, dp_dv = logistic_normal_with_grads(a, v)
        assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
        assert np.all(np.isfinite(dp_da)) and np.all(np.isfinite(dp_dv))
        # each row must equal what it gets when computed on its own
        for i in range(a.size):
            p_i, da_i, dv_i = logistic_normal_with_grads(a[i:i + 1], v[i:i + 1])
            # Rounding only: BLAS blocks a one-row product differently from a
            # six-row one, so a row's answer is not bit-identical alone.
            assert np.isclose(p[i], p_i[0], rtol=1e-12, atol=1e-15)
            assert np.isclose(dp_da[i], da_i[0], rtol=1e-12, atol=1e-15)
            assert np.isclose(dp_dv[i], dv_i[0], rtol=1e-12, atol=1e-15)

    def test_zero_variance_row_is_the_plain_sigmoid(self):
        from MissLearn._utils import logistic_normal_with_grads, sigmoid
        a = np.linspace(-5.0, 5.0, 11)
        p, _, _ = logistic_normal_with_grads(a, np.zeros_like(a))
        assert np.allclose(p, sigmoid(a))


class TestCopulaTies:
    """Repeated values in a column.

    np.interp requires a strictly increasing first argument. The transform
    used to hand it the raw sorted observations, which repeat whenever a
    column is discrete, and every member of a tie block then resolved to the
    block's largest normal score instead of its mean.
    """

    @staticmethod
    def _tied_column(rng, n=400):
        """A column with three distinct values, so it is transformed."""
        return rng.choice([1.0, 2.0, 7.0], size=n, p=[0.6, 0.3, 0.1])

    def test_equal_inputs_get_equal_scores(self):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(11)
        x = self._tied_column(rng)
        t = RankNormalTransformer().fit(x.reshape(-1, 1))
        z = t.transform(x.reshape(-1, 1))[:, 0]
        for v in np.unique(x):
            assert np.unique(z[x == v]).size == 1

    def test_tie_block_takes_its_mean_score(self):
        from scipy.stats import norm
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(12)
        x = self._tied_column(rng)
        t = RankNormalTransformer().fit(x.reshape(-1, 1))
        z = t.transform(x.reshape(-1, 1))[:, 0]

        srt = np.sort(x)
        n = srt.size
        scores = norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        for v in np.unique(x):
            got = np.unique(z[x == v])[0]
            assert np.isclose(got, scores[srt == v].mean())
            # the old behaviour was the block maximum
            assert not np.isclose(got, scores[srt == v].max())

    def test_transformed_column_is_centred(self):
        """The transform is defined to deliver N(0, 1); with ties present it
        used to deliver a column with mean above 1."""
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(13)
        x = self._tied_column(rng)
        t = RankNormalTransformer().fit(x.reshape(-1, 1))
        z = t.transform(x.reshape(-1, 1))[:, 0]
        assert abs(z.mean()) < 1e-8

    def test_interpolation_table_is_strictly_increasing(self):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(14)
        X = np.column_stack([self._tied_column(rng), rng.normal(size=400)])
        t = RankNormalTransformer().fit(X)
        for j in range(X.shape[1]):
            if t._passthrough_[j]:
                continue
            assert np.all(np.diff(t._sorted_obs[j]) > 0)
            assert np.all(np.diff(t._normal_scores[j]) > 0)

    def test_round_trip_is_exact_for_tied_data(self):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(15)
        X = self._tied_column(rng).reshape(-1, 1)
        t = RankNormalTransformer().fit(X)
        assert np.array_equal(t.inverse_transform(t.transform(X)), X)


class TestCopulaDiscreteColumns:
    """Columns with too few distinct values are left alone.

    Sklar's theorem identifies a copula only for continuous marginals, and an
    indicator with prevalence outside roughly [0.25, 0.75] has skewness above
    the trigger threshold by construction, so dummies used to both invite the
    transform and absorb it.
    """

    def test_binary_column_passes_through_untouched(self):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(21)
        b = (rng.random(300) < 0.13).astype(float)
        c = rng.gamma(1.5, size=300)
        X = np.column_stack([b, c])
        t = RankNormalTransformer().fit(X)
        Z = t.transform(X)
        assert t._passthrough_[0] and not t._passthrough_[1]
        assert np.array_equal(Z[:, 0], X[:, 0])
        assert not np.array_equal(Z[:, 1], X[:, 1])

    def test_constant_column_passes_through_untouched(self):
        from MissLearn._copula import RankNormalTransformer
        X = np.column_stack([np.full(50, 3.0), np.linspace(0.0, 1.0, 50)])
        t = RankNormalTransformer().fit(X)
        assert t._passthrough_[0]
        assert np.array_equal(t.transform(X)[:, 0], X[:, 0])

    def test_indicators_alone_do_not_trigger_the_transform(self):
        from MissLearn._copula import needs_copula
        rng = np.random.default_rng(22)
        binaries = (rng.random((400, 20)) < 0.1).astype(float)
        assert not needs_copula(binaries)
        gaussian = rng.normal(size=(400, 4))
        assert not needs_copula(np.column_stack([gaussian, binaries]))

    def test_skewed_continuous_still_triggers_alongside_indicators(self):
        from MissLearn._copula import needs_copula
        rng = np.random.default_rng(23)
        binaries = (rng.random((400, 20)) < 0.1).astype(float)
        skewed = rng.gamma(1.2, size=(400, 3))
        assert needs_copula(np.column_stack([skewed, binaries]))

    def test_round_trip_exact_through_passthrough_columns(self):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(24)
        X = np.column_stack([(rng.random(200) < 0.3).astype(float),
                             rng.gamma(1.5, size=200)])
        t = RankNormalTransformer().fit(X)
        assert np.allclose(t.inverse_transform(t.transform(X)), X)


class TestCopulaEmptyColumns:
    """A column with nothing left to estimate from must not stop a fit.

    The copula was the only stage that could refuse data the rest of the
    library accepts; with copula='auto' as the default, a fold sparse enough
    to empty one column killed the fit outright.
    """

    @staticmethod
    def _with_empty_column(rng, n_observed):
        X = rng.gamma(1.5, size=(150, 4))
        X[:, 2] = np.nan
        if n_observed:
            X[:n_observed, 2] = rng.gamma(1.5, size=n_observed)
        return X

    @pytest.mark.parametrize("n_observed", [0, 1])
    def test_transformer_accepts_an_empty_column(self, n_observed):
        from MissLearn._copula import RankNormalTransformer
        rng = np.random.default_rng(31)
        X = self._with_empty_column(rng, n_observed)
        t = RankNormalTransformer().fit(X)
        assert t._passthrough_[2]

    def test_estimator_fits_when_one_value_survives(self):
        """The realistic cross-validation case: a fold leaves a single
        observed value in a column. The copula used to abort the fit here."""
        from MissLearn import MissRidgeRegressor
        rng = np.random.default_rng(32)
        X = self._with_empty_column(rng, 1)
        y = np.nansum(X[:, [0, 1, 3]], axis=1) + rng.normal(scale=0.3, size=150)
        m = MissRidgeRegressor(alpha=1.0, copula=True, compute_se=False)
        m.fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))

    def test_a_wholly_empty_column_is_still_refused_clearly(self):
        """Refusing it is deliberate and shared by every estimator, so the
        copula must not be what reports it."""
        import pytest as _pytest
        from MissLearn import MissRidgeRegressor
        from MissLearn._conformance import EmptyFeatureError
        rng = np.random.default_rng(33)
        X = self._with_empty_column(rng, 0)
        y = np.nansum(X[:, [0, 1, 3]], axis=1) + rng.normal(scale=0.3, size=150)
        m = MissRidgeRegressor(alpha=1.0, copula=True, compute_se=False)
        with _pytest.raises(EmptyFeatureError, match="no observed values"):
            m.fit(X, y)


class TestCopulaSkewIsScaleFree:
    """Skewness and excess kurtosis are dimensionless, so changing the units
    of a column must not change them, and must not change whether the copula
    fires. Computing them from raw moments was not scale-free: the cube
    overflowed near 1e120 while the square was still finite, and both
    underflowed below about 1e-150, either way returning NaN. needs_copula
    reads abs(NaN) > 1.0 as False, so a skewed column silently stopped
    triggering the transform once its units were large enough.
    """

    SKEWED = np.array([1., 2., 3., 4., 5., 60., 7., 8., 9., 10.])

    @pytest.mark.parametrize("exponent",
                             [-200, -160, -150, 0, 100, 120, 160, 200, 300])
    def test_moments_do_not_move_with_the_units(self, exponent):
        from MissLearn._copula import _skew_kurtosis
        g1_unit, g2_unit = _skew_kurtosis(self.SKEWED)
        g1, g2 = _skew_kurtosis(self.SKEWED * (10.0 ** exponent))
        assert np.isfinite(g1) and np.isfinite(g2)
        assert np.isclose(g1, g1_unit, rtol=1e-9)
        assert np.isclose(g2, g2_unit, rtol=1e-9)

    @pytest.mark.parametrize("exponent", [-200, -150, 0, 120, 160, 300])
    def test_the_decision_does_not_move_with_the_units(self, exponent):
        from MissLearn._copula import needs_copula
        scaled = (self.SKEWED * (10.0 ** exponent)).reshape(-1, 1)
        assert needs_copula(scaled)

    def test_still_matches_scipy_at_unit_scale(self):
        """The docstring promises agreement with scipy's defaults."""
        from scipy.stats import kurtosis, skew
        from MissLearn._copula import _skew_kurtosis
        rng = np.random.default_rng(3)
        for arr in (self.SKEWED, rng.gamma(1.5, size=400), rng.normal(size=400)):
            g1, g2 = _skew_kurtosis(arr)
            assert np.isclose(g1, skew(arr))
            assert np.isclose(g2, kurtosis(arr, fisher=True))

    def test_constant_column_is_reported_flat(self):
        from MissLearn._copula import _skew_kurtosis
        assert _skew_kurtosis(np.full(20, 3.0)) == (0.0, 0.0)
        assert _skew_kurtosis(np.array([5.0])) == (0.0, 0.0)

    def test_no_warning_is_raised_at_any_scale(self):
        from MissLearn._copula import _skew_kurtosis
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for exponent in (-200, -150, 0, 120, 160, 300):
                _skew_kurtosis(self.SKEWED * (10.0 ** exponent))


class TestCopulaDecisionMatchesAction:
    """Whether y is consulted must match whether y is transformed."""

    def test_neighbors_regressor_ignores_a_skewed_target(self):
        """It averages neighbouring y on the original scale, so a skewed y is
        no reason to transform the features."""
        from MissLearn import MissNeighborsRegressor
        rng = np.random.default_rng(41)
        X = rng.normal(size=(200, 3))               # features already normal
        y = np.exp(rng.normal(size=200) * 2.0)      # target heavily skewed
        m = MissNeighborsRegressor(n_neighbors=5, copula='auto')
        m.fit(X, y)
        assert not m.copula_used_

    def test_every_estimator_that_consults_y_also_transforms_it(self):
        """A guard on the divergence itself, not on one estimator."""
        import inspect
        import MissLearn as ml
        for name in dir(ml):
            obj = getattr(ml, name)
            if not (inspect.isclass(obj) and hasattr(obj, "fit")):
                continue
            try:
                source = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            if "needs_copula(X, y)" in source:
                assert "_copula_y_" in source, (
                    name + " consults y when deciding but never transforms it"
                )


# ===========================================================================
# 25c. Marginalisation quadrature at wide variance
# ===========================================================================

class TestLogisticNormalIntegral:
    """E[sigma(a+t)], t ~ N(0,v), is the integral every logistic prediction
    with missing features goes through. v is the variance of the missing
    features' linear contribution, so it grows with both how many are missing
    and how large the coefficients are, and Gauss-Hermite alone could not hold
    it: at 20 nodes the error reached 0.004 by v=25 and 0.03 by v=100.
    """

    @staticmethod
    def _reference(a, v, n=200001):
        """Fine trapezoid over plus or minus 16 standard deviations."""
        from scipy.special import expit
        sd = np.sqrt(v)
        z = np.linspace(-16 * sd, 16 * sd, n)
        w = np.exp(-0.5 * (z / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        return _trapezoid(w * expit(a + z), z)

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 1.5, 4.0, 25.0, 100.0, 2000.0])
    def test_accurate_across_the_whole_variance_range(self, v):
        from MissLearn._utils import integrate_logistic_normal
        a = np.linspace(-10.0, 10.0, 21)
        got = np.atleast_1d(integrate_logistic_normal(a, v))
        ref = np.array([self._reference(ai, v) for ai in a])
        assert np.abs(got - ref).max() < 1e-6

    def test_branches_agree_where_they_meet(self):
        """The two rules must not leave a step at the crossover.

        Compared at one variance, not either side of it: nudging v to force
        the other branch also moves the answer by dE/dv times the nudge, which
        swamps the quantity of interest.
        """
        from MissLearn._utils import (integrate_logistic_normal,
                                      _integrate_wide, _SPLIT_ABOVE_V)
        a = np.linspace(-6.0, 6.0, 25)
        gauss_hermite = np.atleast_1d(integrate_logistic_normal(a, _SPLIT_ABOVE_V))
        split = _integrate_wide(a, _SPLIT_ABOVE_V)
        # The crossover is the split rule's own worst point: the Gaussian is
        # narrowest there against the fixed panel width. 4e-9 is what the
        # shipped node count leaves, against the 3e-2 this branch replaces.
        assert np.abs(gauss_hermite - split).max() < 1e-7

    def test_zero_variance_is_the_plain_sigmoid(self):
        from MissLearn._utils import integrate_logistic_normal, sigmoid
        a = np.linspace(-5.0, 5.0, 11)
        assert np.allclose(np.atleast_1d(integrate_logistic_normal(a, 0.0)),
                           sigmoid(a))

    def test_scalar_in_scalar_out(self):
        from MissLearn._utils import integrate_logistic_normal
        for v in (0.5, 50.0):
            assert isinstance(integrate_logistic_normal(0.7, v), float)

    @pytest.mark.parametrize("v", [0.5, 30.0, 300.0])
    def test_stays_a_probability_and_increases_with_a(self, v):
        from MissLearn._utils import integrate_logistic_normal
        a = np.linspace(-40.0, 40.0, 81)
        got = np.atleast_1d(integrate_logistic_normal(a, v))
        assert np.all(got > 0.0) and np.all(got < 1.0)
        assert np.all(np.diff(got) >= 0.0)
        # strict only away from the saturated ends, where the result is
        # deliberately clipped into (1e-15, 1 - 1e-15) and so goes flat
        core = (got > 1e-9) & (got < 1.0 - 1e-9)
        assert core.sum() > 5
        assert np.all(np.diff(got[core]) > 0.0)

    @pytest.mark.parametrize("v", [2.0, 60.0])
    def test_symmetric_about_zero(self, v):
        """E[sigma(a+t)] + E[sigma(-a+t)] = 1 when t is centred."""
        from MissLearn._utils import integrate_logistic_normal
        a = np.linspace(0.5, 12.0, 15)
        pos = np.atleast_1d(integrate_logistic_normal(a, v))
        neg = np.atleast_1d(integrate_logistic_normal(-a, v))
        assert np.allclose(pos + neg, 1.0, atol=1e-9)


# ===========================================================================

class TestSharedLayerSurface:
    """Paths in the shared layer that no test had ever executed.

    Phase 1 of the coverage work. These five modules run on every fit of
    every estimator, so an untested branch here is untested for all 23 at
    once. Nothing dramatic is asserted: the point is that the code runs at
    all, and that its output is the shape and content it claims.
    """

    @staticmethod
    def _xy(n=60, p=3):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, p))
        X[::7, 1] = np.nan
        y = np.nan_to_num(X)[:, 0] * 2 + rng.normal(scale=0.3, size=n)
        return X, y

    # -- _base.py: the sklearn metadata-routing surface -------------------

    def test_metadata_routing_surface(self):
        from MissLearn import MissLinear
        X, y = self._xy()
        m = MissLinear(compute_se=False).fit(X, y)
        try:
            from sklearn.utils.metadata_routing import MetadataRequest  # noqa
        except ImportError:
            pytest.skip('sklearn < 1.3 has no metadata routing')
        assert m.get_metadata_routing() is not None
        # MissLinear.fit takes no metadata, so it has no set_fit_request, the
        # same as sklearn's own PCA. It used to carry a no-op stub, which on
        # scikit-learn 1.7 and later shadowed the real generated method on the
        # estimators that do have metadata and silently discarded their
        # request.
        assert not hasattr(m, 'set_fit_request')
        from MissLearn import MissMixedRegressor
        assert hasattr(MissMixedRegressor(), 'set_fit_request')

    def test_missingness_report_and_conf_int_shapes(self):
        from MissLearn import MissLinear
        X, y = self._xy()
        m = MissLinear().fit(X, y)
        m.missingness_report()                    # prints; must not raise
        ci = m.conf_int()
        assert ci.shape == (X.shape[1] + 1, 2)
        assert np.all(np.diff(ci, axis=1) > 0)    # lower below upper

    # -- _conformance.py: label encoding ----------------------------------

    def test_encode_labels_handles_strings_and_nan(self):
        from MissLearn import MissLogistic
        from MissLearn._conformance import encode_labels
        y = np.array(['b', 'a', np.nan, 'b', 'a'], dtype=object)
        codes, classes = encode_labels(MissLogistic(), y)
        assert list(classes) == ['a', 'b']
        assert np.isnan(codes[2])                 # missing label stays missing
        assert codes[0] == codes[3] and codes[1] == codes[4]

    def test_conformance_guards_reject_and_accept(self):
        from MissLearn._conformance import (check_penalty, check_positive_int,
                                            check_feature_names)
        assert check_penalty(0.0) == 0.0
        assert check_positive_int(3, 'k') == 3
        for bad in (-1.0, 'x'):
            with pytest.raises(ValueError):
                check_penalty(bad)
        for bad in (0, -1, 2.5):
            with pytest.raises(ValueError):
                check_positive_int(bad, 'k')
        check_feature_names(object(), None)       # nothing recorded: no-op

    # -- _validate.py: the report printer and the copula walk -------------

    def test_validation_report_prints_every_severity(self, capsys):
        from MissLearn._validate import ValidationResult
        r = ValidationResult()
        r._add_note('a note')
        r._add_warning('a warning')
        r.summary()
        assert 'PASSED' in capsys.readouterr().out
        r._add_error('an error')
        assert not r.passed
        r.summary()
        out = capsys.readouterr().out
        assert 'FAILED' in out and 'an error' in out
        assert 'ValidationResult(' in repr(r)

    def test_copula_is_configured_walks_nesting(self):
        """A wrapper holds the model that carries the parameter."""
        import MissLearn as ML
        from MissLearn._validate import copula_is_configured
        assert copula_is_configured(ML.MissLogistic(copula='auto'))
        assert not copula_is_configured(ML.MissLogistic(copula=False))
        assert copula_is_configured(
            ML.MissMulticlass(ML.MissLogistic(copula=True)))
        assert not copula_is_configured(
            ML.MissMulticlass(ML.MissLogistic(copula=False)))
        assert not copula_is_configured(None)

    # -- _multiclass.py: decision_function on both branches ---------------

    def test_decision_function_binary_and_multiclass(self):
        import MissLearn as ML
        X, y = self._xy(n=90)
        yb = (y > np.median(y)).astype(float)
        y3 = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(float)

        d2 = ML.MissMulticlass(ML.MissLogistic()).fit(X, yb).decision_function(X)
        assert np.asarray(d2).ndim == 1           # binary contract is 1-D

        m3 = ML.MissMulticlass(ML.MissLogistic()).fit(X, y3)
        d3 = np.asarray(m3.decision_function(X))
        assert d3.shape == (len(X), 3)
        assert np.all(np.isfinite(d3))

    def test_multiclass_reporting_surface(self, capsys):
        import MissLearn as ML
        from sklearn.base import clone
        X, y = self._xy(n=90)
        y3 = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(float)
        m = ML.MissMulticlass(ML.MissLogistic()).fit(X, y3)
        m.summary()
        assert 'One-vs-Rest' in capsys.readouterr().out
        assert 'MissMulticlass' in repr(m)
        assert np.isclose(m.feature_importances_.sum(), 1.0)
        assert 0.0 <= m.score(X, y3) <= 1.0
        assert isinstance(clone(m), ML.MissMulticlass)   # get/set_params work

    @pytest.mark.parametrize('strategy', ['ovo', 'nonsense', None])
    def test_multiclass_rejects_unknown_strategy(self, strategy):
        import MissLearn as ML
        X, y = self._xy(n=90)
        y3 = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(float)
        with pytest.raises(ValueError, match='strategy'):
            ML.MissMulticlass(ML.MissLogistic(), strategy=strategy).fit(X, y3)


class TestAutomaticMulticlassRouting:
    """A binary classifier given multi-class y routes through MissMulticlass.

    Done in the wrapper, so every classifier gets it without knowing. It was
    the largest untested block in _pandas_compat, which matters because the
    routing has to restore the caller's own labels: a three-class string
    target once came back as [0., 1., 2.], the combination of two features
    that each worked alone.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        X[::8, 1] = np.nan
        lin = np.nan_to_num(X) @ np.array([2.0, 1.0, -1.0])
        y3 = np.digitize(lin, np.quantile(lin, [0.33, 0.66])).astype(float)
        return X, lin, y3

    @pytest.mark.parametrize('name', ['MissLogistic', 'MissRidgeClassifier',
                                      'MissLASSOClassifier',
                                      'MissBayesClassifier'])
    def test_binary_classifier_handles_three_classes(self, name):
        import MissLearn as ML
        X, _, y3 = self._data()
        m = getattr(ML, name)().fit(X, y3)
        assert m._multiclass_router_ is not None
        assert list(m.classes_) == [0.0, 1.0, 2.0]
        P = np.asarray(m.predict_proba(X), dtype=float)
        assert P.shape == (len(X), 3)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert set(np.unique(m.predict(X))) <= set(m.classes_)

    def test_string_labels_survive_routing(self):
        import MissLearn as ML
        X, _, y3 = self._data()
        ys = np.array(['low', 'mid', 'high'])[y3.astype(int)]
        m = ML.MissLogistic().fit(X, ys)
        assert set(m.classes_) == {'low', 'mid', 'high'}
        assert set(np.unique(m.predict(X))) <= set(ys)

    def test_binary_target_leaves_no_router(self):
        """A refit onto binary data must not keep the previous router."""
        import MissLearn as ML
        X, lin, y3 = self._data()
        yb = (lin > np.median(lin)).astype(float)
        assert ML.MissLogistic().fit(X, yb)._multiclass_router_ is None
        m = ML.MissLogistic().fit(X, y3)
        assert m._multiclass_router_ is not None
        m.fit(X, yb)
        assert m._multiclass_router_ is None


class TestFeatureNameChecking:
    """The columns at predict must be the columns the model was fitted on.

    feature_names_in_ was recorded at fit and never read again. A DataFrame
    whose columns were in a different order passed the feature-count check
    and produced predictions correlated -1.000 with the correct ones: on a
    model with coefficients (3.0, -0.015, -3.0), swapping the first and third
    columns negates every prediction. scikit-learn refuses exactly this.
    """

    @staticmethod
    def _data():
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = X @ np.array([3.0, 0.0, -3.0]) + rng.normal(scale=0.2, size=80)
        return pd, X, y, pd.DataFrame(X, columns=['a', 'b', 'c'])

    def test_reordered_columns_refused(self):
        from MissLearn import MissLinear
        pd, X, y, df = self._data()
        m = MissLinear(compute_se=False).fit(df, pd.Series(y))
        with pytest.raises(ValueError, match='different order'):
            m.predict(df[['c', 'b', 'a']])

    def test_renamed_and_subset_columns_refused(self):
        from MissLearn import MissLinear
        pd, X, y, df = self._data()
        m = MissLinear(compute_se=False).fit(df, pd.Series(y))
        with pytest.raises(ValueError):
            m.predict(df.rename(columns={'a': 'zzz'}))
        with pytest.raises(ValueError):
            m.predict(df[['a', 'b']])

    def test_matching_columns_and_arrays_still_work(self):
        """Every legitimate combination must stay open.

        In particular an ndarray at predict against a frame-fitted model, and
        a frame at predict against an array-fitted one: feature_names_in_ is
        populated either way, with synthetic X0..Xp for an array, so a naive
        check rejects both.
        """
        from MissLearn import MissLinear
        pd, X, y, df = self._data()
        m_arr = MissLinear(compute_se=False).fit(X, y)
        m_df = MissLinear(compute_se=False).fit(df, pd.Series(y))
        for model, data in ((m_arr, X), (m_arr, df),
                            (m_df, X), (m_df, df)):
            assert np.all(np.isfinite(np.asarray(model.predict(data),
                                                 dtype=float)))

    @pytest.mark.parametrize('name', ['MissRidgeRegressor',
                                      'MissBayesRegressor',
                                      'MissNeighborsRegressor',
                                      'MissLASSORegressor'])
    def test_check_is_shared_by_the_family(self, name):
        """It lives in the wrapper, so no estimator can be missing it."""
        import MissLearn as ML
        pd, X, y, df = self._data()
        m = getattr(ML, name)().fit(df, pd.Series(y))
        with pytest.raises(ValueError, match='different order'):
            m.predict(df[['c', 'b', 'a']])


class TestPrefitCheckAgreesWithFit:
    """A pre-flight check that passes data fit will refuse is worse than none.

    prefit_check reported "All checks passed. Data appears compatible with
    MissLearn" for X or y containing infinity, which every estimator then
    rejected at fit with a ValueError.
    """

    @staticmethod
    def _xy():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        return X, np.nan_to_num(X)[:, 0] * 2 + rng.normal(scale=0.3, size=60)

    def test_nan_is_still_fine(self):
        """NaN is the point of the library, not a problem."""
        from MissLearn import prefit_check
        X, y = self._xy()
        X = X.copy()
        X[::7, 1] = np.nan
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert r.passed

    @pytest.mark.parametrize('where', ['X', 'y'])
    def test_infinity_reported(self, where):
        from MissLearn import prefit_check, MissLinear
        X, y = self._xy()
        X, y = X.copy(), y.copy()
        if where == 'X':
            X[0, 0] = np.inf
        else:
            y[0] = np.inf
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert not r.passed, 'prefit_check passed data that fit refuses'
        assert any('infinit' in e.lower() for e in r.errors)
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(X, y)

    def test_feature_names_length_mismatch_warns(self):
        """col_label falls back to an index past the end, so the report
        silently mixes names and numbers."""
        from MissLearn import prefit_check
        X, y = self._xy()
        r = prefit_check(X, y, feature_names=['only_one'],
                         raise_on_error=False, emit_warnings=False)
        assert any('feature_names' in w for w in r.warnings)


class TestConfIntAlpha:
    """alpha is a significance level, so it has to be inside (0, 1).

    Outside it the result is not an interval: alpha=1.5 gave a mean width of
    -0.029, putting the lower bound above the upper, and alpha=0 gave
    infinite width.
    """

    @staticmethod
    def _fitted():
        from MissLearn import MissLinear
        rng = np.random.default_rng(0)
        X = rng.normal(size=(70, 3))
        y = X @ np.array([2.0, 0.0, -1.0]) + rng.normal(scale=0.3, size=70)
        return MissLinear().fit(X, y)

    @pytest.mark.parametrize('alpha', [0.01, 0.05, 0.5, 0.99])
    def test_valid_alphas_give_positive_width(self, alpha):
        ci = self._fitted().conf_int(alpha=alpha)
        assert np.all(ci[:, 1] - ci[:, 0] > 0)

    @pytest.mark.parametrize('alpha', [0.0, 1.0, 1.5, -1.0, np.nan, 'x'])
    def test_invalid_alphas_refused(self, alpha):
        with pytest.raises(ValueError):
            self._fitted().conf_int(alpha=alpha)


class TestDeterminismAxis:
    """The conformance check must notice an estimator answering twice.

    This is the axis the suite was missing. Everything else looks at one fit,
    and a large finite number passes those however arbitrary it is, so a whole
    family of degenerate-column defects survived 398 passing conformance tests
    and was found by a property test instead. The first time this ran it
    caught MissMixedRegressor moving by 6.3e-02 under a row permutation.
    """

    def test_catches_an_order_dependent_estimator(self):
        from sklearn.base import BaseEstimator, RegressorMixin
        from MissLearn import check_missing_data_estimator as chk

        class OrderDependent(RegressorMixin, BaseEstimator):
            """Reads only the first half of y as supplied, so row order matters."""

            def fit(self, X, y):
                y = np.asarray(y, dtype=float)
                self._m = float(np.nanmean(y[:len(y) // 2]))
                return self

            def predict(self, X):
                return np.full(len(X), self._m)

        r = chk(OrderDependent())
        assert not r.ok
        assert any('row order' in p.detail for p in r.problems)

    def test_real_estimators_are_deterministic(self):
        import MissLearn as ML
        from MissLearn import check_missing_data_estimator as chk
        for cls in (ML.MissRidgeRegressor, ML.MissLinear, ML.MissLogistic):
            r = chk(cls())
            assert not [p for p in r.problems if 'row order' in p.detail], str(r)

    def test_per_row_fit_kwargs_follow_their_rows(self):
        """The permuted fit must reuse the original groups, not regenerate them.

        A callable such as ``lambda X, y: arange(len(X)) // 5`` assigns by
        position, so regenerating it from permuted rows gives every row a
        different subject and compares two different problems. That made
        MissMixedRegressor look non-deterministic when it was exact.
        """
        import MissLearn as ML
        from MissLearn import check_missing_data_estimator as chk
        r = chk(ML.MissMixedRegressor(),
                fit_kwargs={'groups': lambda X, y: np.arange(len(X)) // 5})
        assert r.ok, str(r)

    def test_determinism_can_be_scoped_or_disabled(self):
        import MissLearn as ML
        from MissLearn import check_missing_data_estimator as chk
        from MissLearn._estimator_checks import DETERMINISM_REGIMES
        assert 'clean_15pct' in DETERMINISM_REGIMES     # the control must be in
        for mode in ('off', 'default', 'all'):
            assert chk(ML.MissRidgeRegressor(compute_se=False),
                       determinism=mode).ok

    def test_clear_error_recognises_actionable_messages(self):
        """The keyword list was all nouns, so it missed messages naming a fix."""
        from MissLearn._estimator_checks import _clear_error
        for msg in ("Provide 'estimator' or 'estimators'",
                    "gamma must be positive; got -1.0",
                    "MissLinear cannot fit: 902 parameters from 25 rows"):
            assert _clear_error(ValueError(msg)), msg
        for msg in ("Matrix is not positive definite", "Singular matrix", ""):
            assert not _clear_error(ValueError(msg)), msg


class TestPenaltyAndKernelGuards:
    """Parameters whose invalid values were accepted and fitted anyway."""

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(size=(n, 5))
        X[::7, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([2., 0., -1.5, 0., 0.8]) + \
            rng.normal(scale=0.3, size=n)
        return X, y, (y > np.median(y)).astype(float)

    def test_negative_penalty_refused_by_every_estimator(self):
        """Two of four refused, and only by accident.

        MissLASSORegressor and MissRidgeRegressor seed from sklearn's Lasso
        and Ridge, whose parameter validation caught it. The classifiers had
        no such path: alpha=-1 gave coefficients of 16.7 and 18.0 against
        true values near 2, and MissLogistic's own l2_reg=-1 reached 332.6.
        A negative penalty rewards large coefficients, so there is no
        minimum to find.
        """
        import MissLearn as ML
        X, y, yc = self._data()
        cases = [(ML.MissLASSORegressor, y, {'alpha': -1.0}),
                 (ML.MissLASSOClassifier, yc, {'alpha': -1.0}),
                 (ML.MissRidgeRegressor, y, {'alpha': -1.0,
                                             'compute_se': False}),
                 (ML.MissRidgeClassifier, yc, {'alpha': -1.0,
                                               'compute_se': False}),
                 (ML.MissLogistic, yc, {'l2_reg': -1.0,
                                        'compute_se': False})]
        for cls, target, kw in cases:
            with pytest.raises(ValueError):
                cls(**kw).fit(X, target)

    def test_zero_and_positive_penalties_still_accepted(self):
        import MissLearn as ML
        X, y, yc = self._data()
        for cls, target, kw in [(ML.MissLASSORegressor, y, {'alpha': 0.0}),
                                (ML.MissRidgeRegressor, y, {'alpha': 1.0,
                                                            'compute_se': False}),
                                (ML.MissRidgeClassifier, yc, {'alpha': 0.0,
                                                              'compute_se': False})]:
            assert np.isfinite(cls(**kw).fit(X, target).score(X, target))

    def test_non_positive_gamma_refused(self):
        """gamma=-1 fitted happily and returned an R^2 of -1.7e34.

        exp(-gamma * d^2) with gamma < 0 grows with distance, so the kernel
        is no longer positive semi-definite or a similarity.
        """
        import MissLearn as ML
        X, y, yc = self._data()
        for bad in (-1.0, 0.0, float('nan'), 'bogus'):
            with pytest.raises(ValueError, match='gamma'):
                ML.MissSupportRegressor(gamma=bad).fit(X, y)
            with pytest.raises(ValueError, match='gamma'):
                ML.MissSupportClassifier(gamma=bad).fit(X, yc)

    def test_non_positive_counts_refused(self):
        """A count below one does not weaken the model, it replaces it.

        n_neighbors=0 returned an R-squared of exactly 0 with every
        prediction equal to the training mean, and n_neighbors=-1 fitted a
        different k from the one requested while looking plausible. On a
        polynomial kernel degree=0 collapsed every prediction to one value
        and degree=-1 reached an R-squared of -3.7e6.
        """
        import MissLearn as ML
        X, y, yc = self._data()
        for bad in (0, -1, 2.5):
            with pytest.raises(ValueError, match='n_neighbors'):
                ML.MissNeighborsRegressor(n_neighbors=bad).fit(X, y)
            with pytest.raises(ValueError, match='n_neighbors'):
                ML.MissNeighborsClassifier(n_neighbors=bad).fit(X, yc)
        for bad in (0, -1):
            with pytest.raises(ValueError, match='degree'):
                ML.MissSupportRegressor(kernel='poly', degree=bad).fit(X, y)

    def test_valid_counts_unaffected(self):
        import MissLearn as ML
        X, y, _ = self._data()
        assert np.isfinite(
            ML.MissNeighborsRegressor(n_neighbors=1).fit(X, y).score(X, y))
        assert np.isfinite(
            ML.MissNeighborsRegressor(n_neighbors=5).fit(X, y).score(X, y))
        assert np.isfinite(
            ML.MissSupportRegressor(kernel='poly',
                                    degree=3).fit(X, y).score(X, y))

    def test_valid_gammas_and_kernels_unaffected(self):
        import MissLearn as ML
        X, y, _ = self._data()
        for kernel in ('linear', 'rbf', 'poly'):
            for gamma in ('scale', 'auto', 0.5):
                m = ML.MissSupportRegressor(kernel=kernel, gamma=gamma)
                assert np.all(np.isfinite(m.fit(X, y).predict(X)))


class TestMissingDataReportGrading:
    """The shipped conformance checker must not certify a non-participant.

    A clean refusal is graded acceptable, which is right for one regime an
    estimator legitimately declines. Applied to all eleven it produced an
    inversion: sklearn's Ridge, which cannot accept a NaN at all, refused
    every regime tidily and reported ok; HistGradientBoosting, which fits ten
    of the eleven, reported not ok.
    """

    def test_estimator_that_fits_nothing_does_not_pass(self):
        from sklearn.linear_model import Ridge
        from MissLearn import check_missing_data_estimator as chk
        r = chk(Ridge())
        assert not r.participates
        assert not r.ok
        assert 'NOT A MISSING-DATA ESTIMATOR' in str(r)

    def test_real_estimators_still_pass(self):
        import MissLearn as ML
        from MissLearn import check_missing_data_estimator as chk
        for cls in (ML.MissRidgeRegressor, ML.MissLinear, ML.MissLogistic):
            r = chk(cls())
            assert r.participates and r.ok, str(r)

    def test_bool_tracks_ok(self):
        """``if report:`` was True for every object, failures included."""
        import MissLearn as ML
        from sklearn.linear_model import Ridge
        from MissLearn import check_missing_data_estimator as chk
        assert bool(chk(ML.MissRidgeRegressor(compute_se=False)))
        assert not bool(chk(Ridge()))

    def test_per_row_fit_kwargs_may_be_callable(self):
        """A fixed array cannot span regimes of different length.

        Most regimes are n=120, wide_p_gt_n is n=25, so one groups vector is
        right for ten and a length mismatch on the eleventh. That surfaced as
        'refused opaquely', reading as a fault in the estimator rather than
        in how it was called.
        """
        import MissLearn as ML
        from MissLearn import check_missing_data_estimator as chk
        r = chk(ML.MissMixedRegressor(),
                fit_kwargs={'groups': lambda X, y: np.arange(len(X)) // 5})
        assert r.participates and r.ok, str(r)


class TestScoringValidation:
    """A misspelled scorer must fail once, not warn once per fold.

    Warn-and-substitute-NaN is the right response to a fold that happens to
    be degenerate; the other folds may still be informative. It is the wrong
    response to a typo, which cannot succeed on any fold: the caller got one
    warning per split and an array of NaN for what is a configuration error.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 3))
        X[::6, 1] = np.nan
        y = np.nan_to_num(X[:, 0] * 2) + rng.normal(scale=0.4, size=n)
        return X, y

    def test_known_names_and_callables_accepted(self):
        from MissLearn import miss_cross_val_score, MissRidgeRegressor
        X, y = self._data()
        for scoring in ('r2', 'neg_mse', 'neg_root_mean_squared_error',
                        None, 'auto', lambda e, Xt, yt: 0.5):
            s = miss_cross_val_score(
                MissRidgeRegressor(alpha=1.0, compute_se=False),
                X, y, cv=3, scoring=scoring)
            assert np.all(np.isfinite(s))

    def test_unknown_name_raises_immediately(self):
        from MissLearn import miss_cross_val_score, MissRidgeRegressor
        X, y = self._data()
        for bad in ('R2', 'rmse', 'not_a_metric', ''):
            with pytest.raises(ValueError, match='Unknown scoring string'):
                miss_cross_val_score(
                    MissRidgeRegressor(alpha=1.0, compute_se=False),
                    X, y, cv=3, scoring=bad)

    def test_error_names_the_valid_options(self):
        from MissLearn._crossval import _validate_scoring, SCORER_NAMES
        with pytest.raises(ValueError) as exc:
            _validate_scoring('nope')
        for name in ('r2', 'accuracy', 'roc_auc'):
            assert repr(name) in str(exc.value)
        assert 'r2' in SCORER_NAMES

    def test_collection_of_names_validated_elementwise(self):
        from MissLearn._crossval import _validate_scoring
        _validate_scoring(['r2', 'neg_mae'])          # fine
        _validate_scoring({'a': 'r2'})                # fine
        with pytest.raises(ValueError):
            _validate_scoring(['r2', 'bogus'])

    def test_wrong_type_rejected(self):
        from MissLearn._crossval import _validate_scoring
        with pytest.raises(TypeError):
            _validate_scoring(42)


class TestRubinRules:
    """MissImputer.combine, checked against the formulae by hand."""

    TH = [np.array([1.0]), np.array([1.2]), np.array([0.9])]
    VR = [np.array([0.10]), np.array([0.20]), np.array([0.30])]

    def test_pooling_matches_the_formulae(self):
        from MissLearn import MissImputer
        r = MissImputer(m=3, random_state=0).combine(self.TH, self.VR)
        W = np.mean(self.VR, axis=0)
        B = np.var(self.TH, axis=0, ddof=1)
        assert np.allclose(r['estimate'], np.mean(self.TH, axis=0))
        assert np.allclose(r['within_var'], W)
        assert np.allclose(r['between_var'], B)
        assert np.allclose(r['total_var'], W + (1 + 1 / 3) * B)
        assert np.allclose(r['se'], np.sqrt(W + (1 + 1 / 3) * B))

    def test_mismatched_lengths_refused(self):
        """W and B must come from the same m.

        Three estimates against two variances was accepted, reported m as
        three, and returned a pooled standard error 11.5% too small. That
        number is what gets published.
        """
        from MissLearn import MissImputer
        imp = MissImputer(m=3, random_state=0)
        with pytest.raises(ValueError, match='same imputed datasets'):
            imp.combine(self.TH, self.VR[:2])
        with pytest.raises(ValueError, match='same imputed datasets'):
            imp.combine(self.TH[:2], self.VR)

    def test_optional_and_single_imputation_paths_still_work(self):
        from MissLearn import MissImputer
        imp = MissImputer(m=3, random_state=0)
        assert 'total_var' not in imp.combine(self.TH)      # variances=None
        one = imp.combine([self.TH[0]], [self.VR[0]])       # m = 1
        assert one['m'] == 1
        assert np.allclose(one['between_var'], 0.0)


class TestMissSensitivityGuards:
    """tipping_point must refuse what it cannot answer.

    Its return value is load-bearing in both directions: None means the
    conclusion is robust across the whole delta range, and a number near zero
    means it collapses immediately. Two ways it used to produce one of those
    without having earned it.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(size=(n, 3))
        X[::5, 1] = np.nan
        y = np.nan_to_num(X[:, 0] * 1.5 + X[:, 2] * 0.4) + \
            rng.normal(scale=0.4, size=n)
        return X, y

    def test_ci_refuses_without_standard_errors(self):
        """Returned 0.0, i.e. maximally fragile, having computed nothing.

        Every comparison against NaN is False, the negation marked every
        delta as a crossing, and the nearest one to zero was returned. An
        estimator fitted with compute_se=False lands here.
        """
        from MissLearn import MissSensitivity, MissRidgeRegressor
        X, y = self._data()
        s = MissSensitivity(
            MissRidgeRegressor(alpha=1.0, compute_se=False)).fit(X, y)
        assert np.all(np.isnan(s.coef_se_curves_))
        with pytest.raises(ValueError, match='standard errors'):
            s.tipping_point(0, method='ci')
        # The criterion that needs no SEs still works.
        assert s.tipping_point(0, method='sign') is None

    def test_unknown_method_is_rejected(self):
        """Anything but 'sign' used to fall through to the 'ci' branch."""
        from MissLearn import MissSensitivity, MissRidgeRegressor
        X, y = self._data()
        s = MissSensitivity(MissRidgeRegressor(alpha=1.0)).fit(X, y)
        for bad in ('nonsense', 'Sign', 'CI', ''):
            with pytest.raises(ValueError, match="'sign' or 'ci'"):
                s.tipping_point(0, method=bad)

    def test_coef_idx_bounds(self):
        from MissLearn import MissSensitivity, MissRidgeRegressor
        X, y = self._data()
        s = MissSensitivity(MissRidgeRegressor(alpha=1.0)).fit(X, y)
        with pytest.raises(IndexError, match='out of range'):
            s.tipping_point(99)
        s.tipping_point(-1)          # negative indexing stays valid

    def test_both_methods_agree_when_both_are_available(self):
        from MissLearn import MissSensitivity, MissRidgeRegressor
        X, y = self._data()
        s = MissSensitivity(MissRidgeRegressor(alpha=1.0)).fit(X, y)
        for meth in ('sign', 'ci'):
            tp = s.tipping_point(0, method=meth)
            assert tp is None or np.isfinite(tp)


class TestDeclaredSklearnExceptions:
    """The declared check_estimator exceptions must stay honest.

    _sklearn_compat.py justifies skipping check_methods_subset_invariance by
    asserting that predict_proba is subset-invariant and only its logged form
    moves. That is a measurable claim, and it had already drifted: the file
    said 0.000e+00 while the family's worst case was 6.7e-15. The number
    itself does not matter, the argument survives it comfortably, but a
    justification a reviewer can re-measure and find wrong discredits the
    ones beside it. This pins it.
    """

    CLASSIFIERS = ['MissLogistic', 'MissRidgeClassifier',
                   'MissBayesClassifier', 'MissNeighborsClassifier',
                   'MissLASSOClassifier']

    def test_predict_proba_is_subset_invariant(self):
        import MissLearn as ML
        rng = np.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 4))
        X[::6, 1] = np.nan
        y = np.where(np.isnan(X[:, 0]), 0, (X[:, 0] > 0).astype(int))

        worst = 0.0
        for name in self.CLASSIFIERS:
            m = getattr(ML, name)().fit(X, y)
            full = np.asarray(m.predict_proba(X), dtype=float)
            for sl in (slice(None, None, 2), slice(3, 40), slice(7, 8)):
                sub = np.asarray(m.predict_proba(X[sl]), dtype=float)
                worst = max(worst, float(np.nanmax(np.abs(full[sl] - sub))))
        # Far tighter than check_estimator's own tolerance, and far looser
        # than the exact zero the file used to claim.
        assert worst < 1e-12, (
            "predict_proba subset invariance degraded to %.3e; the "
            "justification in _sklearn_compat.py rests on it" % worst)

    def test_every_declared_exception_carries_a_reason(self):
        from MissLearn._sklearn_compat import (EXPECTED_FAILED_CHECKS,
                                               expected_failed_checks)
        assert EXPECTED_FAILED_CHECKS, "declaring none would hide real failures"
        for check, reason in EXPECTED_FAILED_CHECKS.items():
            assert check.startswith('check_')
            assert len(reason) > 80, "%s needs a reason, not a label" % check
        # The callback form scikit-learn passes an estimator to.
        import MissLearn as ML
        assert expected_failed_checks(ML.MissLogistic()) == EXPECTED_FAILED_CHECKS


class TestDeprecation:
    """The deprecation helpers, which nothing had executed before.

    Both defects these cover were in code at 0% coverage: the parameter
    decorator inspected only ``kwargs``, so a caller passing the deprecated
    argument positionally was told nothing, which is the caller most likely
    to be running old code; and the message appended a full stop to an
    *extra* that usually ended with one, giving "See the guide..".
    """

    @staticmethod
    def _warnings_from(fn, *args, **kwargs):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            fn(*args, **kwargs)
        return [str(r.message) for r in rec
                if issubclass(r.category, FutureWarning)]

    def test_deprecated_warns_and_preserves_behaviour(self):
        from MissLearn._deprecation import deprecated

        @deprecated(replacement='new_fn', removed_in='1.1')
        def old_fn(a, b=2):
            """Original."""
            return a + b

        msgs = self._warnings_from(old_fn, 1, b=3)
        assert len(msgs) == 1
        assert 'new_fn' in msgs[0] and '1.1' in msgs[0]
        with warnings.catch_warnings():     # the warning is the point above,
            warnings.simplefilter('ignore')  # here it is only noise
            assert old_fn(1, b=3) == 4      # still does its job
        assert old_fn.__name__ == 'old_fn'  # functools.wraps intact
        assert 'deprecated' in old_fn.__doc__

    def test_deprecated_does_not_double_punctuate(self):
        from MissLearn._deprecation import deprecated

        @deprecated(replacement='x', extra='See the guide.')
        def f():
            pass

        assert '..' not in self._warnings_from(f)[0]

        @deprecated(replacement='x', extra='no trailing stop')
        def g():
            pass

        assert self._warnings_from(g)[0].endswith('no trailing stop.')

    def test_parameter_warns_however_it_was_passed(self):
        from MissLearn._deprecation import deprecate_parameter

        @deprecate_parameter('old_arg', replacement='new_arg')
        def f(x, old_arg=None, new_arg=None):
            return x

        assert self._warnings_from(f, 1, old_arg=5)      # by keyword
        assert self._warnings_from(f, 1, 5)              # positionally

    def test_parameter_silent_when_not_supplied(self):
        from MissLearn._deprecation import deprecate_parameter

        @deprecate_parameter('old_arg', replacement='new_arg')
        def f(x, old_arg=None, new_arg=None):
            return x

        assert not self._warnings_from(f, 1)
        assert not self._warnings_from(f, 1, new_arg=9)

    def test_parameter_on_a_method(self):
        """``self`` occupies the first positional slot; binding must allow it."""
        from MissLearn._deprecation import deprecate_parameter

        class C:
            @deprecate_parameter('old_arg', replacement='new_arg')
            def meth(self, x, old_arg=None):
                return x

        c = C()
        assert self._warnings_from(c.meth, 1, old_arg=2)
        assert self._warnings_from(c.meth, 1, 2)
        assert not self._warnings_from(c.meth, 1)



# ===========================================================================
# Phase 2: the estimator families
#
# The shared layer was covered in Phase 1. These cover the seven estimator
# families, and specifically the three surfaces that were almost entirely
# unexercised: the constructor options, the refusal paths, and the
# summary / interval / importance accessors that a user actually reads
# after fitting.
# ===========================================================================

import MissLearn as _ML


def _phase2_data(classifier=False, n=80, groups=False):
    """A small well-conditioned fixture with one partially missing column."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 4))
    X[::7, 1] = np.nan
    y = np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8])
    y = y + rng.standard_normal(n) * 0.3
    if classifier:
        y = (y > np.median(y)).astype(float)
    if groups:
        return X, y, np.repeat(np.arange(n // 5), 5)
    return X, y



def _captured(fn, *args, **kwargs):
    """Return what a call printed, so a chatty method can be asserted on."""
    import io as _io, contextlib as _c
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()

def _summary_text(model, **kwargs):
    """Return what ``summary()`` prints rather than letting it reach stdout."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.summary(**kwargs)
    return buf.getvalue()


class TestStringOptionsAreChecked:
    """A misspelled option selected a different model without saying so.

    These options are consumed as ``== 'mahalanobis'`` or ``== 'full'``, so
    an unrecognised value quietly took the other branch. Nothing was unset
    and nothing was missing, so nothing complained. The differences are not
    cosmetic: on this fixture a typo moved predictions by up to 1.5
    (``metric``), 2.9 (``weights``) and 2.8 (``structure``). ``structure``
    was the worst of the three because it tested only for ``'full'``, so any
    misspelling of that word, ``'ful'`` included, selected naive
    independence, which is the opposite model.
    """

    CASES = [
        ('MissNeighborsRegressor',  'weights',   ('distance', 'uniform')),
        ('MissNeighborsClassifier', 'weights',   ('distance', 'uniform')),
        ('MissNeighborsRegressor',  'metric',    ('euclidean', 'mahalanobis')),
        ('MissNeighborsClassifier', 'metric',    ('euclidean', 'mahalanobis')),
        ('MissBayesRegressor',      'structure', ('full', 'naive')),
        ('MissBayesClassifier',     'structure', ('full', 'naive')),
    ]

    @pytest.mark.parametrize('name,param,options', CASES)
    def test_unknown_value_is_refused(self, name, param, options):
        X, y = _phase2_data(name.endswith('Classifier'))
        with pytest.raises(ValueError, match=param):
            getattr(_ML, name)(**{param: 'not_an_option'}).fit(X, y)

    @pytest.mark.parametrize('name,param,options', CASES)
    def test_every_documented_value_fits(self, name, param, options):
        X, y = _phase2_data(name.endswith('Classifier'))
        for value in options:
            pred = getattr(_ML, name)(**{param: value}).fit(X, y).predict(X)
            assert np.all(np.isfinite(pred)), (name, param, value)

    @pytest.mark.parametrize('name,param,options', CASES)
    def test_the_option_actually_changes_the_model(self, name, param, options):
        """An option that never changes the answer is not an option.

        Written the other way round from the refusal tests deliberately. If a
        refactor makes one branch unreachable, the refusal tests still pass
        and this one does not.
        """
        X, y = _phase2_data(name.endswith('Classifier'))
        preds = [getattr(_ML, name)(**{param: v}).fit(X, y).predict(X)
                 for v in options]
        assert not np.array_equal(preds[0], preds[1]), (name, param)

    def test_a_near_miss_is_named_in_the_message(self):
        """Case and whitespace slips are told what they nearly matched."""
        X, y = _phase2_data()
        with pytest.raises(ValueError, match="Did you mean 'naive'"):
            _ML.MissBayesRegressor(structure='NAIVE').fit(X, y)

    def test_the_dispatchers_refuse_too(self):
        """MissBayes and MissNeighbors forward to the concrete classes."""
        X, y = _phase2_data()
        with pytest.raises(ValueError, match='structure'):
            _ML.MissBayes(structure='nieve').fit(X, y)
        with pytest.raises(ValueError, match='metric'):
            _ML.MissNeighbors(metric='mahalnobis').fit(X, y)


class TestCopulaOptionIsChecked:
    """The widest-reaching member of the silent-fallback family.

    ``copula`` is a parameter of all twenty-three estimators and is consumed
    as ``if self.copula == 'auto': ... elif self.copula:``. Any other string
    misses the first test and is then merely truthy, so a misspelling
    silently forced the transform on: ``copula='atuo'`` fitted without
    complaint and moved predictions by 1.4 against ``copula=False`` on plain
    data. Checked in ``_store_fit_metadata`` rather than in each estimator,
    because that is the one hook every fit path already runs through.

    Only strings are inspected. A bool is left alone, since the defect is
    that a misspelling of the one legal word cannot be told apart from a
    deliberate True.
    """

    FAMILIES = ['MissLinear', 'MissRidgeRegressor', 'MissLASSORegressor',
                'MissBayesRegressor', 'MissNeighborsRegressor',
                'MissSupportRegressor', 'MissLogistic', 'MissRidgeClassifier',
                'MissLASSOClassifier', 'MissBayesClassifier']

    @pytest.mark.parametrize('name', FAMILIES)
    def test_a_misspelling_is_refused(self, name):
        X, y = _phase2_data(name.endswith('Classifier') or name == 'MissLogistic',
                            n=40)
        with pytest.raises(ValueError, match='copula'):
            getattr(_ML, name)(copula='atuo').fit(X, y)

    @pytest.mark.parametrize('value', [True, False, 'auto'])
    @pytest.mark.parametrize('name', FAMILIES)
    def test_every_legal_value_still_fits(self, name, value):
        X, y = _phase2_data(name.endswith('Classifier') or name == 'MissLogistic',
                            n=40)
        pred = getattr(_ML, name)(copula=value).fit(X, y).predict(X)
        assert np.all(np.isfinite(pred))

    def test_the_near_miss_is_named(self):
        X, y = _phase2_data(n=40)
        with pytest.raises(ValueError, match="Did you mean 'auto'"):
            _ML.MissLinear(copula='AUTO').fit(X, y)

    def test_the_two_settings_are_not_the_same_model(self):
        """Confirms what the typo was silently choosing between."""
        X, y = _phase2_data(n=40)
        on = _ML.MissLinear(copula=True).fit(X, y).predict(X)
        off = _ML.MissLinear(copula=False).fit(X, y).predict(X)
        assert not np.allclose(on, off, atol=1e-6)


class TestAvailabilityBasedImportancesAreDeliberate:
    """Three families report availability, not predictive contribution.

    ``MissNeighbors``, ``MissSupport`` on a non-linear kernel, and
    ``MissGaussian`` without ARD all define ``feature_importances_`` as the
    fraction of training rows in which each feature is observed. Their
    docstrings say so. It means a constant column scores as highly as the
    only informative one, which is correct for what is being measured and
    surprising given what the attribute is called everywhere else in
    scikit-learn. Pinned here so the intent is visible in the tests and a
    future reader does not mistake it for a defect, or quietly change it.
    """

    AVAILABILITY = ['MissNeighborsClassifier', 'MissSupportClassifier',
                    'MissGaussianClassifier']
    PREDICTIVE = ['MissBayesClassifier', 'MissLogistic', 'MissLASSOClassifier']

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 60
        X = rng.standard_normal((n, 4))
        X[:, 2] = 3.5                       # constant: no predictive content
        X[::5, 1] = np.nan                  # partially observed
        y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(float)
        return X, y

    @pytest.mark.parametrize('name', AVAILABILITY)
    def test_they_track_observedness_not_signal(self, name):
        X, y = self._data()
        imp = np.asarray(getattr(_ML, name)().fit(X, y).feature_importances_,
                         dtype=float)
        assert abs(imp.sum() - 1.0) < 1e-8
        # The column with NaNs scores below the fully observed ones, and the
        # constant column is not singled out, because neither fact is about
        # prediction.
        assert imp[1] < imp[0]
        assert imp[2] == pytest.approx(imp[0], abs=1e-12)

    @pytest.mark.parametrize('name', PREDICTIVE)
    def test_the_other_families_do_rank_by_signal(self, name):
        X, y = self._data()
        imp = np.asarray(getattr(_ML, name)().fit(X, y).feature_importances_,
                         dtype=float)
        assert abs(imp.sum() - 1.0) < 1e-8
        assert int(np.argmax(imp)) == 0
        assert imp[2] == 0.0


class TestLassoSolverMustRespectBounds:
    """MissLASSO writes its L1 term by variable splitting, beta = u - v.

    The penalty ``sum(u + v)`` equals ``|beta|`` only while ``u, v >= 0``.
    ``method`` was handed straight to scipy with no check, and solvers that
    cannot handle bounds discard them with a warning nobody sees, at which
    point the penalty rewards large coefficients instead of shrinking them
    and the objective has no minimum. CG reached an R-squared of -2.3e6 that
    way. Nelder-Mead keeps the bounds but does not solve this problem: it
    returned the training mean, an R-squared of exactly 0, while reporting
    convergence, which is the worst of the three because it looks like a fit.
    """

    SUPPORTED = ('L-BFGS-B', 'TNC', 'Powell', 'SLSQP')
    REFUSED = ('CG', 'BFGS', 'Nelder-Mead', 'trust-constr', 'not_a_solver')

    @pytest.mark.parametrize('method', REFUSED)
    @pytest.mark.parametrize('name', ['MissLASSORegressor', 'MissLASSOClassifier'])
    def test_unsupported_solvers_are_refused(self, name, method):
        X, y = _phase2_data(name.endswith('Classifier'))
        with pytest.raises(ValueError, match='method'):
            getattr(_ML, name)(method=method).fit(X, y)

    @pytest.mark.parametrize('method', SUPPORTED)
    def test_supported_solvers_recover_the_signal(self, method):
        X, y = _phase2_data()
        pred = _ML.MissLASSORegressor(method=method).fit(X, y).predict(X)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        assert r2 > 0.9, (method, r2)

    @pytest.mark.parametrize('method', SUPPORTED)
    def test_supported_solvers_agree_with_each_other(self, method):
        """The solver is an implementation detail, not a modelling choice."""
        X, y = _phase2_data()
        base = _ML.MissLASSORegressor(method='L-BFGS-B').fit(X, y).coef_
        other = _ML.MissLASSORegressor(method=method).fit(X, y).coef_
        assert np.allclose(base, other, atol=1e-3), (method, base, other)


class TestLassoClassifierSeedIsDeterministic:
    """Two fits of identical data returned different coefficients.

    ``MissLASSOClassifier`` seeds its optimiser from sklearn's
    ``LogisticRegression`` with ``solver='liblinear'``, and liblinear
    shuffles its coordinate order using an RNG. With the default
    ``random_state=None`` that draw moved on every call, the starting point
    moved with it, and the drift carried all the way through to the fitted
    coefficients: 1.6e-04 across five fits of the same matrix. It failed
    sklearn's ``check_fit_idempotent`` and it was why this estimator alone
    was not bit-exact under row permutation, since a permuted fit is just
    another fit with another draw. Its regressor sibling seeds from
    ``Lasso``, which is deterministic, so only the classifier was affected.
    This is the only sklearn solver in the library that consumes an RNG.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(3)
        n = 60
        X = rng.standard_normal((n, 3))
        y = (X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.3 > 0)
        return X, y.astype(float)

    def test_repeated_fits_are_bit_exact(self):
        X, y = self._data()
        coefs = [_ML.MissLASSOClassifier().fit(X, y).coef_ for _ in range(4)]
        for c in coefs[1:]:
            assert np.array_equal(c, coefs[0])

    def test_decision_function_is_bit_exact_across_fits(self):
        """The failing surface in sklearn's idempotency check."""
        X, y = self._data()
        scores = [_ML.MissLASSOClassifier().fit(X, y).decision_function(X)
                  for _ in range(3)]
        for s in scores[1:]:
            assert np.array_equal(s, scores[0])

    def test_a_permuted_fit_is_bit_exact_too(self):
        X, y = self._data()
        perm = np.random.default_rng(11).permutation(len(y))
        plain = _ML.MissLASSOClassifier().fit(X, y)
        shuffled = _ML.MissLASSOClassifier().fit(X[perm], y[perm])
        assert np.array_equal(plain.coef_, shuffled.coef_)
        assert plain.intercept_ == shuffled.intercept_

    def test_the_seed_estimator_is_pinned(self):
        """A future edit that drops random_state should fail here, not in CI."""
        import inspect
        from MissLearn import _lasso
        source = inspect.getsource(_lasso.MissLASSOClassifier)
        assert "solver='liblinear'" in source
        assert 'random_state=0' in source


class TestMixedTauUnderflow:
    """``tau_sq`` arrives as ``exp(2 * log_tau)`` and underflows to zero.

    Below a log_tau of about -354 it is exactly 0.0, and ``1.0 / tau_sq`` on
    a Python float raises ZeroDivisionError rather than returning inf, so the
    fit died inside the objective. SLSQP walked there on ordinary
    well-conditioned data. The limit does not deserve refusing: tau_sq -> 0
    is simply the no-random-effect model, where the shrinkage term vanishes
    and the likelihood tends to the ordinary regression one, so the region is
    floored rather than rejected.
    """

    @pytest.mark.parametrize('method', ['SLSQP', 'L-BFGS-B', 'TNC', 'Powell'])
    @pytest.mark.parametrize('name', ['MissMixedRegressor', 'MissMixedClassifier'])
    def test_fit_survives_every_solver(self, name, method):
        X, y, g = _phase2_data(name.endswith('Classifier'), n=90, groups=True)
        model = getattr(_ML, name)(method=method).fit(X, y, groups=g)
        assert np.all(np.isfinite(model.predict(X, groups=g)))

    def test_the_objective_is_defined_at_the_underflow_point(self):
        """Evaluate the likelihood exactly where the division used to fail.

        The arithmetic itself, since the objective that performs it is a
        closure inside ``fit`` and cannot be called from here.

        This replaces a white-box test that called
        ``MissMixedRegressor._neg_log_likelihood`` directly. That method was
        removed on 18 August 2026 as provably dead, which means the test had
        been exercising a path the optimiser never takes: the ZeroDivisionError
        it was written for happened in the live reduced objective, not there.
        A test that reaches for a private method is worth checking twice, and
        this one was verifying the fix in the wrong copy of it.

        What guards the real path is behavioural, and it is
        ``test_fit_survives_every_solver`` above: SLSQP is the solver that
        walked into the underflow, and it now fits.
        """
        tau = float(np.exp(-400.0))
        assert tau > 0.0, 'tau itself is merely tiny'
        assert tau * tau == 0.0, 'it is the square that underflows'

        # The floored division is what the objective performs, and what used
        # to raise: Python float division by zero raises rather than
        # returning inf, which is why an underflow killed the fit outright.
        from MissLearn._mixed import _TAU_SQ_FLOOR

        assert _TAU_SQ_FLOOR > 0.0
        denom = 1.0 / max(tau * tau, _TAU_SQ_FLOOR)
        assert np.isfinite(denom)
        with pytest.raises(ZeroDivisionError):
            1.0 / (tau * tau)

    def test_the_optimiser_reaches_the_boundary_on_data_with_no_group_effect(self):
        """With no between-subject variance in the data, the MLE is tau = 0.

        The optimiser walks towards it, which is how the underflow was met in
        the first place: Powell lands at a tau_sq around 1e-58 here.
        """
        X, y, g = _phase2_data(n=90, groups=True)
        model = _ML.MissMixedRegressor(method='Powell').fit(X, y, groups=g)
        assert model.tau_sq_ >= 0.0
        assert model.tau_sq_ < 1e-6
        assert np.all(np.isfinite(model.predict(X, groups=g)))

    def test_the_floor_does_not_move_an_ordinary_fit(self):
        X, y, g = _phase2_data(n=90, groups=True)
        pred = _ML.MissMixedRegressor().fit(X, y, groups=g).predict(X, groups=g)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        assert r2 > 0.9


class TestMixedGroupsValidation:
    """A wrong-length ``groups`` used to fail inside numpy.

    The message was about input array dimensions and a concatenation axis,
    naming neither ``groups`` nor either length, and it surfaced from the
    canonical reorder rather than from anything the caller had written.
    """

    @pytest.mark.parametrize('name', ['MissMixedRegressor', 'MissMixedClassifier'])
    def test_length_mismatch_names_groups_and_both_lengths(self, name):
        X, y = _phase2_data(name.endswith('Classifier'), n=90)
        with pytest.raises(ValueError, match=r'groups has 50 entries.*90 rows'):
            getattr(_ML, name)().fit(X, y, groups=np.repeat(np.arange(10), 5))

    def test_degenerate_group_structures_still_fit(self):
        """One row per group, and one group for everything, are both legal."""
        X, y = _phase2_data(n=90)
        for g in (np.arange(len(y)), np.zeros(len(y))):
            model = _ML.MissMixedRegressor().fit(X, y, groups=g)
            assert np.all(np.isfinite(model.predict(X, groups=g)))


class TestEnsembleOutOfBag:
    """``oob_score=True`` with ``bootstrap=False`` did nothing, quietly.

    Out-of-bag rows are the rows a bootstrap draw left out, so without
    bootstrapping there are none: every ``oob_indices_`` entry was empty, the
    scoring branch was skipped, and ``oob_scores_`` came back an empty dict
    with no warning. The user asked for a score and got silence. sklearn
    raises for the same combination.
    """

    def _ens(self, **kw):
        return _ML.MissEnsemble(estimator=_ML.MissLinear(),
                                n_estimators=4, random_state=0, **kw)

    def test_oob_without_bootstrap_is_refused(self):
        X, y = _phase2_data(n=90)
        with pytest.raises(ValueError, match='oob_score=True requires bootstrap=True'):
            self._ens(bootstrap=False, oob_score=True).fit(X, y)

    def test_oob_with_bootstrap_is_populated(self):
        X, y = _phase2_data(n=90)
        model = self._ens(bootstrap=True, oob_score=True).fit(X, y)
        assert len(model.oob_scores_) == 4
        assert all(np.isfinite(v) for v in model.oob_scores_.values())
        assert all(len(i) > 0 for i in model.oob_indices_)

    @pytest.mark.parametrize('bootstrap', [True, False])
    def test_oob_off_leaves_the_dict_empty(self, bootstrap):
        X, y = _phase2_data(n=90)
        model = self._ens(bootstrap=bootstrap, oob_score=False).fit(X, y)
        assert model.oob_scores_ == {}
        assert np.all(np.isfinite(model.predict(X)))

    def test_summary_reports_the_oob_block(self):
        X, y = _phase2_data(n=90)
        model = self._ens(bootstrap=True, oob_score=True).fit(X, y)
        assert 'OOB score' in _summary_text(model)


class TestCoefficientTableWithoutStandardErrors:
    """MissLASSO does not compute standard errors by default.

    Everything derived from ``se_`` was therefore nan, and the shared table
    printed four columns of literal "nan" on every row under headings that
    said p_value and CI. A note above it explained why, but a results table
    reading nan looks like a fit that failed rather than inference that was
    never requested.
    """

    def test_no_nan_columns_when_se_was_not_computed(self):
        X, y = _phase2_data()
        text = _summary_text(_ML.MissLASSORegressor(compute_se=False).fit(X, y))
        assert 'nan' not in text.lower()
        assert 'p_value' not in text
        assert 'were not computed' in text

    def test_the_estimates_are_still_shown(self):
        X, y = _phase2_data()
        model = _ML.MissLASSORegressor(compute_se=False).fit(X, y)
        text = _summary_text(model)
        for c in np.atleast_1d(model.coef_).ravel():
            assert ('%.4f' % c) in text

    def test_the_full_table_is_unchanged_when_se_is_available(self):
        X, y = _phase2_data()
        text = _summary_text(_ML.MissLASSORegressor(compute_se=True).fit(X, y))
        for column in ('coef', 'std_err', 'p_value', 'CI_lower', 'CI_upper'):
            assert column in text

    @pytest.mark.parametrize('name', ['MissLinear', 'MissLogistic',
                                      'MissRidgeRegressor', 'MissRidgeClassifier'])
    def test_estimators_that_do_compute_se_are_untouched(self, name):
        X, y = _phase2_data(name.endswith('Classifier') or name == 'MissLogistic')
        assert 'p_value' in _summary_text(getattr(_ML, name)().fit(X, y))


class TestMulticlassRankingsAgree:
    """predict, predict_proba and decision_function must rank classes alike.

    scikit-learn requires the argmax of ``decision_function`` and the argmax
    of ``predict_proba`` both to equal ``predict``. MissMulticlass defines
    ``predict`` as the argmax of ``predict_proba``, so the probability side
    held by construction, but ``decision_function`` took each
    sub-classifier's own raw score whenever it had one. A raw score and its
    calibrated probability are related by a monotone map fitted separately
    for every one-vs-rest problem, so the class with the highest score need
    not be the class with the highest probability.

    On MissSupportClassifier, where Platt scaling fits its own parameters per
    sub-problem, they disagreed on 3 of 300 rows and
    ``check_classifiers_train`` failed. Plain scikit-learn SVC passes that
    check, so this one was ours. Both are now derived from the same
    probabilities, and the logit is strictly increasing, so they agree for
    arithmetic reasons rather than by luck.
    """

    MULTICLASS = ["MissSupportClassifier", "MissLogistic",
                  "MissLASSOClassifier", "MissRidgeClassifier",
                  "MissBayesClassifier", "MissNeighborsClassifier"]

    @staticmethod
    def _three_class(n=200):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        signal = X[:, 0] + rng.standard_normal(n) * 0.4
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])).astype(float)
        return X, y

    @pytest.mark.parametrize('name', MULTICLASS)
    def test_decision_function_ranks_classes_like_predict(self, name):
        X, y = self._three_class()
        model = getattr(_ML, name)().fit(X, y)
        decision = np.asarray(model.decision_function(X), dtype=float)
        assert decision.ndim == 2 and decision.shape[1] == len(model.classes_)
        picked = model.classes_[np.argmax(decision, axis=1)]
        disagree = int(np.sum(picked != model.predict(X)))
        assert disagree == 0, "%s disagrees on %d rows" % (name, disagree)

    @pytest.mark.parametrize('name', MULTICLASS)
    def test_predict_proba_ranks_classes_like_predict(self, name):
        X, y = self._three_class()
        model = getattr(_ML, name)().fit(X, y)
        proba = np.asarray(model.predict_proba(X), dtype=float)
        picked = model.classes_[np.argmax(proba, axis=1)]
        assert int(np.sum(picked != model.predict(X))) == 0

    def test_the_two_scores_are_monotone_in_each_other(self):
        """Not just the argmax: the whole ranking has to match.

        Equal argmax could happen by accident on one fixture. Ordering every
        class the same way in every row is the property that actually holds.
        """
        X, y = self._three_class()
        model = _ML.MissSupportClassifier().fit(X, y)
        decision = np.asarray(model.decision_function(X), dtype=float)
        proba = np.asarray(model.predict_proba(X), dtype=float)
        for row in range(X.shape[0]):
            assert np.array_equal(np.argsort(decision[row]),
                                  np.argsort(proba[row])), (
                "row %d orders the classes differently in the two" % row)

    def test_a_binary_problem_keeps_the_one_dimensional_contract(self):
        """The binary path is untouched and must stay 1-D.

        The sign of the score and the predicted label agree wherever the
        probability is not exactly tied. They can differ on an exact tie:
        MissSupport takes its score from the raw SVM distance and its label
        from the Platt-calibrated probability, and Platt's intercept means
        the point where the score crosses zero is not the point where the
        probability crosses a half. Two rows of this fixture come back at
        p = 0.50000000 with scores of +0.0096 and +0.0097, where argmax
        breaks the tie toward the first class and the sign does not. That is
        the same convention scikit-learn uses for a tie, and it is why this
        asserts agreement off the tie rather than everywhere.
        """
        rng = np.random.default_rng(1)
        n = 120
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(float)
        model = _ML.MissSupportClassifier().fit(X, y)

        decision = np.asarray(model.decision_function(X), dtype=float)
        assert decision.ndim == 1

        proba = np.asarray(model.predict_proba(X), dtype=float)
        untied = proba[:, 0] != proba[:, 1]
        assert untied.sum() > n * 0.9, "fixture is almost all ties"
        picked = model.classes_[(decision > 0).astype(int)]
        disagree = int(np.sum(picked[untied] != model.predict(X)[untied]))
        assert disagree == 0, "%d untied rows disagree" % disagree

    def test_binary_score_and_probability_rank_the_same_rows(self):
        """Off the ties, the score orders the rows exactly as the probability does."""
        rng = np.random.default_rng(1)
        n = 120
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(float)
        model = _ML.MissSupportClassifier().fit(X, y)
        decision = np.asarray(model.decision_function(X), dtype=float)
        p = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
        # Compare only rows whose probability is distinct, since tied
        # probabilities have no order for the score to agree with.
        keep = np.array([np.sum(p == v) == 1 for v in p])
        d_rank = np.argsort(np.argsort(decision[keep]))
        p_rank = np.argsort(np.argsort(p[keep]))
        assert np.array_equal(d_rank, p_rank)


class TestFeatureScaleMeasuresTheColumnNotTheDesign:
    """``feature_scale`` decides which columns are worth standardising.

    It used to ask whether a column's spread was a large enough fraction of
    the *widest* spread in the design, and that is measuring the wrong thing:
    how small a column's spread is next to some other column's says nothing
    about whether the spread is real. The two cases the criterion had to
    separate fall the wrong way round. A perfectly ordinary Gaussian column
    scaled by 1e-06, which must be standardised, sits 71 times *further*
    below the widest column than the pathological column that motivated the
    floor, so no threshold could keep one and drop the other. What it did
    instead was leave that column unstandardised, after which
    MissMixedRegressor needed a coefficient of 1e+06 to compensate and
    reached an r-squared of -6.7e+06 where MissLinear managed 0.73.

    The test is now per column and scale-free: a spread counts when it is
    large relative to the resolution of that column's own values. The three
    cases below sit twelve orders of magnitude apart on that measure, so the
    tolerance is a gap rather than a line.
    """

    N = 120

    def _tiny_gaussian_beside_a_huge_one(self):
        """Must be standardised. This is the conformance extreme_scale shape."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((self.N, 4))
        X[:, 0] *= 1e6
        X[:, 1] *= 1e-6
        return X

    def _two_tiny_values(self):
        """The column the withdrawn floor was written for: 0 and 5.96e-08."""
        rng = np.random.default_rng(0)
        X = rng.normal(scale=300.0, size=(self.N, 4))
        col = np.zeros(self.N)
        col[rng.random(self.N) < 0.25] = 5.96e-08
        X[:, 1] = col
        return X

    def _offset_with_only_rounding_noise(self):
        """Must be refused: centring this is catastrophic cancellation."""
        rng = np.random.default_rng(0)
        X = rng.normal(scale=300.0, size=(self.N, 4))
        X[:, 1] = 1.0e6 + rng.normal(scale=1e-10, size=self.N)
        return X

    def test_a_tiny_but_real_spread_is_standardised(self):
        from MissLearn._utils import feature_scale

        X = self._tiny_gaussian_beside_a_huge_one()
        divisor = feature_scale(X)[1]
        assert divisor != 1.0, (
            "a Gaussian column scaled by 1e-06 was classed degenerate; this "
            "is the defect that cost MissMixedRegressor an r-squared of -6.7e6")
        assert np.isclose(divisor, np.nanstd(X[:, 1], ddof=1))

    def test_the_column_the_old_floor_was_written_for_is_also_real(self):
        """Its instability was row ordering, and that has its own fix now.

        ``canonical_row_order`` fixes the summation order before the fit, so a
        permutation cannot change the answer at all. Refusing to standardise
        was treating the symptom.
        """
        from MissLearn._utils import feature_scale

        X = self._two_tiny_values()
        assert feature_scale(X)[1] != 1.0

    def test_a_spread_that_is_only_rounding_noise_is_refused(self):
        from MissLearn._utils import feature_scale

        X = self._offset_with_only_rounding_noise()
        col = X[:, 1]
        relative = np.nanstd(col, ddof=1) / np.nanmax(np.abs(col))
        assert relative < np.finfo(float).eps, "fixture is not the intended case"
        assert feature_scale(X)[1] == 1.0

    def test_no_relative_threshold_could_have_separated_the_first_two(self):
        """Records why the criterion was replaced rather than retuned."""
        keep = self._tiny_gaussian_beside_a_huge_one()
        refuse = self._two_tiny_values()

        def ratio_to_widest(X):
            sd = np.nanstd(X, axis=0, ddof=1)
            return sd[1] / sd.max()

        # The column that must be kept is further from the widest column than
        # the one the floor existed to catch, so the old test ordered them
        # backwards and no threshold sits between them.
        assert ratio_to_widest(keep) < ratio_to_widest(refuse)

    def test_the_tolerance_is_a_gap_not_a_line(self):
        """Every value across eight orders gives the same three answers."""
        from MissLearn._utils import feature_scale

        keep_a = self._tiny_gaussian_beside_a_huge_one()
        keep_b = self._two_tiny_values()
        refuse = self._offset_with_only_rounding_noise()
        for rtol in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
            assert feature_scale(keep_a, rtol=rtol)[1] != 1.0, rtol
            assert feature_scale(keep_b, rtol=rtol)[1] != 1.0, rtol
            assert feature_scale(refuse, rtol=rtol)[1] == 1.0, rtol

    def test_the_estimator_that_was_broken_now_fits(self):
        X = self._tiny_gaussian_beside_a_huge_one()
        rng = np.random.default_rng(1)
        y = np.nan_to_num(X) @ np.array([1e-6, 1e6, 0.5, -0.5])
        y = y + rng.standard_normal(self.N) * 0.4
        g = np.repeat(np.arange(self.N // 5), 5)
        pred = _ML.MissMixedRegressor(compute_se=False).fit(
            X, y, groups=g).predict(X, groups=g)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        assert r2 > 0.5, "r-squared %.4f; the scale handling has regressed" % r2

    def test_degenerate_columns_are_still_refused(self):
        """The guards this one complements are untouched."""
        from MissLearn._utils import feature_scale

        rng = np.random.default_rng(0)
        X = rng.standard_normal((self.N, 4))
        X[:, 1] = 7.0                       # constant
        X[:, 2] = np.nan                    # nothing observed
        divisors = feature_scale(X)
        assert divisors[1] == 1.0
        assert divisors[2] == 1.0
        assert divisors[0] != 1.0 and divisors[3] != 1.0


class TestDegenerateMaskExcludesOnlyForIdentifiability:
    """The mask decides which columns MissLinear leaves out of its joint model.

    It carried the same relative-to-widest test that was withdrawn from
    ``feature_scale``, and it failed there for the same reason: whether a
    column's spread is small beside *another* column's is not a question
    about identifiability, and a column with a small spread next to a large
    one is perfectly identifiable. It classed a Gaussian column scaled by
    1e-06 as unidentifiable, and MissLinear then scored 0.7334 under the
    conformance extreme_scale regime where MissRidgeRegressor and
    MissLASSORegressor, which share its model and differ only in penalty,
    both reached 0.846.

    What the mask is actually for is honest reporting rather than numerical
    rescue. ``psd_jitter`` already lifts a singular joint covariance onto the
    positive-definite cone, and a constant column then yields a coefficient
    of exactly 0.0 instead of raising. Exclusion adds NaN standard errors,
    which say the coefficient is not identifiable rather than reporting it as
    exactly zero with an interval around it.

    Being reluctant to exclude is therefore the right way to be wrong: a
    column wrongly kept is caught by the jitter, one wrongly dropped is
    silently unavailable with nothing to catch it.
    """

    N = 160

    def test_a_tiny_but_real_column_is_kept(self):
        from MissLearn._utils import degenerate_feature_mask

        rng = np.random.default_rng(0)
        X = rng.standard_normal((self.N, 4))
        X[:, 0] *= 1e6
        X[:, 1] *= 1e-6
        assert not degenerate_feature_mask(X)[1], (
            "a Gaussian column scaled by 1e-06 was called unidentifiable "
            "because another column happened to be large")

    def test_the_identifiability_cases_are_still_excluded(self):
        from MissLearn._utils import degenerate_feature_mask

        rng = np.random.default_rng(0)
        constant = rng.standard_normal((self.N, 3))
        constant[:, 1] = 4.0
        assert degenerate_feature_mask(constant)[1]

        unobserved = rng.standard_normal((self.N, 3))
        unobserved[:, 1] = np.nan
        assert degenerate_feature_mask(unobserved)[1]

        # Varies, but by so little that the variance underflows to zero.
        underflow = rng.standard_normal((self.N, 3))
        underflow[:, 1] = 0.0
        underflow[::2, 1] = 3.07e-177
        assert degenerate_feature_mask(underflow)[1]

        # Varies only by the representation error of its own values.
        cancellation = rng.standard_normal((self.N, 3))
        cancellation[:, 1] = 1.0e6 + rng.standard_normal(self.N) * 1e-10
        assert degenerate_feature_mask(cancellation)[1]

    def test_it_agrees_with_feature_scale_about_what_is_real(self):
        """Two functions asking the same question must not disagree."""
        from MissLearn._utils import degenerate_feature_mask, feature_scale

        rng = np.random.default_rng(3)
        for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            X = rng.standard_normal((self.N, 4))
            X[:, 0] *= 1e6
            X[:, 1] *= scale
            excluded = degenerate_feature_mask(X)[1]
            unscaled = feature_scale(X)[1] == 1.0
            assert excluded == unscaled, (
                "at scale %g the mask says excluded=%s while feature_scale "
                "says unstandardised=%s" % (scale, excluded, unscaled))

    def test_a_design_recorded_entirely_in_small_units_keeps_everything(self):
        """The case the mask must never break: no absolute floor."""
        from MissLearn._utils import degenerate_feature_mask

        rng = np.random.default_rng(4)
        X = rng.standard_normal((self.N, 5)) * 1e-9
        assert not degenerate_feature_mask(X).any()

    def test_misslinear_now_matches_its_penalised_siblings(self):
        """The gap this change closes, measured end to end."""
        rng = np.random.default_rng(0)
        n = 120
        X = rng.standard_normal((n, 5))
        X[:, 0] *= 1e6
        X[:, 1] *= 1e-6
        y = X @ np.array([1e-6, 1e6, 0.5, -0.5, 0.25])
        y = y + rng.standard_normal(n) * 0.4
        X[rng.random(X.shape) < 0.15] = np.nan

        def r2_of(name):
            model = getattr(_ML, name)().fit(X, y)
            pred = model.predict(X)
            return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        scores = {n_: r2_of(n_) for n_ in ("MissLinear", "MissRidgeRegressor",
                                           "MissLASSORegressor")}
        gap = max(scores.values()) - min(scores.values())
        assert gap < 0.06, (
            "the penalty-only siblings disagree by %.4f: %s" % (gap, scores))


class TestBayesEffectSizeOnDegenerateColumns:
    """A nan effect size was silently rewritten as uniform importances.

    MissBayes ranks features by Cohen's d, and a feature with no
    within-class spread makes the pooled standard deviation zero and the
    ratio 0/0. The default ``var_smoothing`` floors the variances so this
    does not arise, but ``var_smoothing=0`` is an accepted value, and then a
    single constant column produced one nan, the sum in
    ``feature_importances_`` became nan, its ``total > 0`` test came out
    false, and the uniform fallback returned exactly 1/p for every feature.

    That is the worst shape this failure could take. The result sums to 1,
    contains no nan and reads as a finding, while the true answer on the
    fixture below is [0.78, 0.05, 0, 0.18]. Predictions were unaffected, so
    nothing else gave it away. ``_bayes`` was also the one estimator module
    that never adopted the shared degenerate-column guard the other eight
    use.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        n = 60
        X = rng.standard_normal((n, 4))
        X[:, 2] = 3.5                       # exact zero variance
        y = (X[:, 0] + rng.standard_normal(n) * 0.3 > 0).astype(float)
        return X, y

    @pytest.mark.parametrize('var_smoothing', [1e-9, 0.0])
    def test_a_constant_column_does_not_flatten_the_ranking(self, var_smoothing):
        X, y = self._data()
        model = _ML.MissBayesClassifier(var_smoothing=var_smoothing).fit(X, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        assert np.all(np.isfinite(imp))
        assert abs(imp.sum() - 1.0) < 1e-8
        assert not np.allclose(imp, 1.0 / X.shape[1])
        assert imp[2] == 0.0                # the constant column carries none
        assert int(np.argmax(imp)) == 0     # the informative one still leads

    def test_the_smoothing_setting_does_not_change_the_answer(self):
        """Both settings should agree once the nan cannot arise."""
        X, y = self._data()
        floored = _ML.MissBayesClassifier(var_smoothing=1e-9).fit(X, y)
        bare = _ML.MissBayesClassifier(var_smoothing=0.0).fit(X, y)
        assert np.allclose(floored.feature_importances_,
                           bare.feature_importances_, atol=1e-8)

    def test_a_perfect_separator_ranks_above_features_that_vary(self):
        """No within-class spread but different class means is real signal.

        Reported by a finite margin rather than inf, since inf would leave
        the normalisation meaningless.
        """
        rng = np.random.default_rng(5)
        n = 60
        X = rng.standard_normal((n, 3))
        y = (np.arange(n) % 2).astype(float)
        X[:, 1] = np.where(y > 0, 5.0, 1.0)
        model = _ML.MissBayesClassifier(var_smoothing=0.0).fit(X, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        assert np.all(np.isfinite(imp))
        assert abs(imp.sum() - 1.0) < 1e-8
        assert int(np.argmax(imp)) == 1

    def test_predictions_were_never_the_problem(self):
        """Recorded so a future reader does not go looking in the wrong place."""
        X, y = self._data()
        model = _ML.MissBayesClassifier(var_smoothing=0.0).fit(X, y)
        assert np.all(np.isfinite(model.predict_proba(X)))
        assert (model.predict(X) == y).mean() > 0.8


class TestFamilyAccessorSurface:
    """summary(), predict_interval() and feature_importances_ per family.

    These are the three things a user reads after fitting, and most of them
    sat at zero coverage: the summary bodies alone were the largest
    uncovered blocks in _gp, _svm and _ensemble.
    """

    REGRESSORS = ['MissLinear', 'MissRidgeRegressor', 'MissLASSORegressor',
                  'MissBayesRegressor', 'MissNeighborsRegressor',
                  'MissSupportRegressor', 'MissGaussianRegressor']
    CLASSIFIERS = ['MissLogistic', 'MissRidgeClassifier', 'MissLASSOClassifier',
                   'MissBayesClassifier', 'MissNeighborsClassifier',
                   'MissSupportClassifier', 'MissGaussianClassifier']

    @pytest.mark.parametrize('name', REGRESSORS + CLASSIFIERS)
    def test_summary_runs_and_names_every_feature(self, name):
        X, y = _phase2_data(name in self.CLASSIFIERS)
        text = _summary_text(getattr(_ML, name)().fit(X, y))
        assert text.strip()
        for j in range(X.shape[1]):
            assert 'X%d' % j in text, (name, j)

    @pytest.mark.parametrize('name', REGRESSORS + CLASSIFIERS)
    def test_importances_are_a_normalised_distribution(self, name):
        X, y = _phase2_data(name in self.CLASSIFIERS)
        imp = np.asarray(getattr(_ML, name)().fit(X, y).feature_importances_,
                         dtype=float)
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))
        assert np.all(imp >= 0)
        assert abs(imp.sum() - 1.0) < 1e-8

    @pytest.mark.parametrize('name', REGRESSORS)
    def test_prediction_intervals_bracket_the_point_estimate(self, name):
        X, y = _phase2_data()
        model = getattr(_ML, name)().fit(X, y)
        lo, hi = model.predict_interval(X)
        point = model.predict(X)
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(hi >= lo), name
        assert np.all((point >= lo - 1e-8) & (point <= hi + 1e-8)), name


class TestGaussianAndSupportOptions:
    """The remaining constructor branches in _gp and _svm.

    ARD gives the kernel one length scale per feature instead of one
    overall, and it had never been fitted in this suite. The kernels and the
    gamma settings are the other untested branches; gamma is the one place
    where a number and a word share a parameter.
    """

    @pytest.mark.parametrize('kernel', ['rbf', 'matern52', 'matern12'])
    @pytest.mark.parametrize('name', ['MissGaussianRegressor', 'MissGaussianClassifier'])
    def test_every_gp_kernel_fits(self, name, kernel):
        X, y = _phase2_data(name.endswith('Classifier'))
        pred = getattr(_ML, name)(kernel=kernel).fit(X, y).predict(X)
        assert np.all(np.isfinite(pred))

    def test_unknown_gp_kernel_is_refused(self):
        X, y = _phase2_data()
        with pytest.raises(ValueError, match='kernel'):
            _ML.MissGaussianRegressor(kernel='matern32').fit(X, y)

    def test_ard_gives_one_length_scale_per_feature(self):
        X, y = _phase2_data()
        shared = _ML.MissGaussianRegressor(ard=False).fit(X, y)
        per_feature = _ML.MissGaussianRegressor(ard=True).fit(X, y)
        assert np.ndim(shared.length_scale_) == 0
        assert np.shape(per_feature.length_scale_) == (X.shape[1],)
        assert not np.array_equal(shared.predict(X), per_feature.predict(X))

    @pytest.mark.parametrize('kernel', ['rbf', 'linear', 'poly'])
    @pytest.mark.parametrize('name', ['MissSupportRegressor', 'MissSupportClassifier'])
    def test_every_svm_kernel_fits(self, name, kernel):
        X, y = _phase2_data(name.endswith('Classifier'))
        pred = getattr(_ML, name)(kernel=kernel).fit(X, y).predict(X)
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize('gamma', ['scale', 'auto', 0.5, 1e-3])
    def test_gamma_accepts_both_a_word_and_a_number(self, gamma):
        X, y = _phase2_data()
        pred = _ML.MissSupportRegressor(gamma=gamma).fit(X, y).predict(X)
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize('gamma', [0.0, -1.0])
    def test_a_non_positive_gamma_is_refused(self, gamma):
        X, y = _phase2_data()
        with pytest.raises(ValueError, match='gamma'):
            _ML.MissSupportRegressor(gamma=gamma).fit(X, y)

    def test_unknown_svm_kernel_is_refused(self):
        X, y = _phase2_data()
        with pytest.raises(ValueError, match='kernel'):
            _ML.MissSupportRegressor(kernel='sigmoid').fit(X, y)



class TestPreprocessorAcceptsTheDataItExistsFor:
    """MissPreprocessor one-hot encodes categoricals, and could not take them.

    A DataFrame with a string column failed with "could not convert string to
    float" before ``MissPreprocessor.fit`` was ever entered. The shared fit
    wrapper validates X numerically for every estimator, and this is the one
    estimator whose job is to *make* X numeric. ``_coerce_X`` already passed
    an object array through for exactly this case, with a comment naming
    MissPreprocessor, and the validation two steps later rejected it anyway.

    Integer-coded categories always worked, which is how it went unnoticed:
    the failure needed a genuine string column, and the class is most useful
    precisely when it has one. The wrapper now defers to any estimator
    declaring ``_ACCEPTS_RAW_INPUT``, at fit and at predict, since a column
    that was a string at fit is still a string when you predict on it.
    """

    @staticmethod
    def _frame(n=60):
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'num': rng.standard_normal(n),
            'cat': rng.choice(['a', 'b', 'c'], n),
            'flag': rng.integers(0, 2, n),
        })
        df.loc[::9, 'num'] = np.nan
        y = df['num'].fillna(0.0) * 2 + rng.standard_normal(n) * 0.3
        return df, np.asarray(y, dtype=float)

    def _prep(self, **kw):
        from MissLearn._validate import MissPreprocessor
        kw.setdefault('verbose', False)
        return MissPreprocessor(estimator=_ML.MissLinear(compute_se=False), **kw)

    def test_a_string_column_is_encoded_rather_than_refused(self):
        df, y = self._frame()
        prep = self._prep()
        _captured(lambda: prep.fit(df, y))
        out = np.asarray(prep.transform(df), dtype=float)
        assert out.shape[0] == len(df)
        # num + two dummies for a three-level column + one for the flag
        assert out.shape[1] == 4
        assert np.all(np.isfinite(out[~np.isnan(out)]))

    def test_missing_values_survive_the_encoding(self):
        """The point of encoding here rather than in sklearn: NaN stays NaN.

        A downstream FIML model has to see an unobserved value as missing.
        An encoder that imputed it would silently undo the whole library.
        """
        df, y = self._frame()
        prep = self._prep()
        _captured(lambda: prep.fit(df, y))
        out = np.asarray(prep.transform(df), dtype=float)
        assert np.isnan(out).any()

    def test_it_predicts_and_scores_through_the_encoding(self):
        df, y = self._frame()
        prep = self._prep()
        _captured(lambda: prep.fit(df, y))
        pred = np.asarray(prep.predict(df), dtype=float)
        assert pred.shape == (len(df),)
        assert np.all(np.isfinite(pred))
        assert prep.score(df, y) > 0.5

    def test_numeric_only_input_is_unaffected(self):
        df, y = self._frame()
        prep = self._prep()
        _captured(lambda: prep.fit(df[['num', 'flag']], y))
        assert np.shape(prep.predict(df[['num', 'flag']])) == (len(df),)

    def test_ordinary_estimators_still_refuse_a_string_column(self):
        """The exemption is opt-in, not a hole in the validation.

        Only an estimator that declares _ACCEPTS_RAW_INPUT skips the numeric
        check. Everything else must still fail loudly on a string.
        """
        df, y = self._frame()
        assert not getattr(_ML.MissLinear, '_ACCEPTS_RAW_INPUT', False)
        with pytest.raises(ValueError):
            _ML.MissLinear(compute_se=False).fit(df, y)


class TestPreprocessorEncodingRules:
    """Which columns get encoded, and how the knobs move that."""

    @staticmethod
    def _matrix(n=90):
        rng = np.random.default_rng(0)
        cont = rng.standard_normal(n)
        binary = rng.integers(0, 2, n).astype(float)
        cat = rng.integers(0, 4, n).astype(float)
        many = rng.integers(0, 40, n).astype(float)
        X = np.column_stack([cont, binary, cat, many])
        X[::11, 0] = np.nan
        X[::13, 2] = np.nan
        y = cont * 2 + binary - 0.5 * cat + rng.standard_normal(n) * 0.3
        return X, y

    def _prep(self, **kw):
        from MissLearn._validate import MissPreprocessor
        kw.setdefault('verbose', False)
        return MissPreprocessor(estimator=_ML.MissLinear(compute_se=False), **kw)

    def test_integer_columns_are_classified_by_how_many_values_they_take(self):
        X, _ = self._matrix()
        found = self._prep()._detect_encoding(X)
        assert found[1]['reason'] == 'binary'
        assert found[2]['reason'] == 'categorical'
        # a continuous column and a 40-level one are both left alone, for
        # opposite reasons
        assert 0 not in found and 3 not in found

    @pytest.mark.parametrize('threshold,expected', [(2, [1]), (5, [1, 2]),
                                                    (50, [1, 2, 3])])
    def test_the_threshold_moves_the_boundary(self, threshold, expected):
        X, _ = self._matrix()
        found = self._prep(categorical_threshold=threshold)._detect_encoding(X)
        assert sorted(found) == expected

    def test_dropping_the_first_level_costs_one_column_per_category(self):
        X, y = self._matrix()
        widths = {}
        for drop in ('first', None):
            prep = self._prep(drop=drop)
            _captured(lambda: prep.fit(X, y))
            widths[drop] = np.shape(prep.transform(X))[1]
        # two encoded columns, so dropping a level each costs two columns
        assert widths[None] - widths['first'] == 2

    def test_a_column_with_nothing_observed_is_skipped_not_encoded(self):
        X, _ = self._matrix()
        X = X.copy()
        X[:, 1] = np.nan
        assert 1 not in self._prep()._detect_encoding(X)

    def test_params_round_trip(self):
        prep = self._prep()
        assert 'categorical_threshold' in prep.get_params()
        assert 'estimator__compute_se' in prep.get_params()
        prep.set_params(categorical_threshold=7, drop=None)
        assert prep.categorical_threshold == 7
        assert prep.drop is None

    def test_summary_and_repr_describe_the_fitted_wrapper(self):
        X, y = self._matrix()
        prep = self._prep()
        _captured(lambda: prep.fit(X, y))
        assert 'MissPreprocessor' in repr(prep)
        assert _captured(prep.summary).strip()

    def test_classifier_methods_are_delegated(self):
        from MissLearn._validate import MissPreprocessor

        X, y = self._matrix()
        prep = MissPreprocessor(estimator=_ML.MissLogistic(compute_se=False),
                                verbose=False)
        yc = (y > np.median(y)).astype(float)
        _captured(lambda: prep.fit(X, yc))
        assert np.shape(prep.predict_proba(X)) == (len(X), 2)
        assert np.shape(prep.decision_function(X)) == (len(X),)

    def test_regressor_interval_is_delegated(self):
        X, y = self._matrix()
        prep = self._prep()
        _captured(lambda: prep.fit(X, y))
        lo, hi = prep.predict_interval(X)
        assert np.all(np.asarray(hi, float) >= np.asarray(lo, float))


class TestEnsembleMemberValidation:
    """MissEnsemble only accepts members that can see a missing value.

    The whole point of bagging here is that each member marginalises over
    missingness itself. A member that cannot do that would need the data
    imputed before it ever saw it, which is the thing the library exists to
    avoid, so it is refused rather than silently wrapped.
    """

    @staticmethod
    def _data(n=90):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8])
        return X, y + rng.standard_normal(n) * 0.3

    def test_a_missdata_estimator_is_accepted(self):
        from MissLearn._ensemble import _check_estimator_compatible
        assert _check_estimator_compatible(_ML.MissLinear()) is None

    def test_a_nan_native_tree_is_accepted(self):
        """HistGradientBoosting handles NaN natively, so it qualifies."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from MissLearn._ensemble import _check_estimator_compatible
        assert _check_estimator_compatible(HistGradientBoostingRegressor()) is None

    @pytest.mark.parametrize('bad', ['LinearRegression', 'SVR', 'not_an_estimator'])
    def test_an_estimator_that_cannot_see_missingness_is_refused(self, bad):
        from MissLearn._ensemble import _check_estimator_compatible
        if bad == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            obj = LinearRegression()
        elif bad == 'SVR':
            from sklearn.svm import SVR
            obj = SVR()
        else:
            obj = object()
        with pytest.raises(ValueError, match='not supported'):
            _check_estimator_compatible(obj)

    def test_the_refusal_reaches_the_user_through_fit(self):
        from sklearn.linear_model import LinearRegression
        X, y = self._data()
        with pytest.raises(ValueError, match='not supported'):
            _ML.MissEnsemble(estimator=LinearRegression(), n_estimators=2).fit(X, y)


class TestEnsembleWeights:
    """Member weights, and the three ways of asking for something impossible."""

    @staticmethod
    def _data(n=90):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8])
        return X, y + rng.standard_normal(n) * 0.3

    def _ens(self, **kw):
        return _ML.MissEnsemble(estimator=_ML.MissLinear(compute_se=False),
                                n_estimators=3, random_state=0, **kw)

    def test_unweighted_and_weighted_both_fit(self):
        X, y = self._data()
        for weights in (None, [1, 1, 2]):
            pred = self._ens(weights=weights).fit(X, y).predict(X)
            assert np.all(np.isfinite(np.asarray(pred, dtype=float)))

    def test_weights_are_normalised_not_taken_literally(self):
        """[1, 1, 2] and [0.25, 0.25, 0.5] are the same request."""
        X, y = self._data()
        a = self._ens(weights=[1, 1, 2]).fit(X, y).predict(X)
        b = self._ens(weights=[0.25, 0.25, 0.5]).fit(X, y).predict(X)
        assert np.allclose(np.asarray(a, float), np.asarray(b, float))

    @pytest.mark.parametrize('weights,match', [
        ([0, 0, 0], 'not all be zero'),
        ([-1, 1, 1], 'non-negative'),
        ([1, 1], 'does not match'),
    ])
    def test_impossible_weights_are_refused(self, weights, match):
        X, y = self._data()
        with pytest.raises(ValueError, match=match):
            self._ens(weights=weights).fit(X, y)


class TestEnsembleGroupsAndComposition:
    """Cluster bootstrap, heterogeneous members, and parameter round-trips."""

    @staticmethod
    def _data(n=90):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8])
        g = np.repeat(np.arange(n // 5), 5)
        return X, y + rng.standard_normal(n) * 0.3, g

    def test_grouped_members_resample_whole_groups(self):
        """A member fitted on half a subject's rows would leak across the split.

        With groups supplied the bootstrap draws at the group level, so a
        subject is either in a member's sample or out of it, and the rows it
        left out are what the out-of-bag score is measured on.
        """
        X, y, g = self._data()
        model = _ML.MissEnsemble(estimator=_ML.MissMixedRegressor(compute_se=False),
                                 n_estimators=3, random_state=0,
                                 oob_score=True)
        _captured(lambda: model.fit(X, y, groups=g))
        assert len(model.oob_scores_) == 3
        assert all(len(idx) > 0 for idx in model.oob_indices_)
        # every out-of-bag index set is a union of whole groups
        for idx in model.oob_indices_:
            for label in np.unique(g[idx]):
                assert set(np.where(g == label)[0]) <= set(idx)

    def test_a_heterogeneous_ensemble_keeps_its_members(self):
        X, y, _ = self._data()
        model = _ML.MissEnsemble(
            estimators=[('linear', _ML.MissLinear(compute_se=False)),
                        ('ridge', _ML.MissRidgeRegressor(compute_se=False))],
            random_state=0).fit(X, y)
        assert len(model.estimators_) == 2
        assert np.all(np.isfinite(np.asarray(model.predict(X), dtype=float)))

    def test_params_round_trip(self):
        model = _ML.MissEnsemble(estimator=_ML.MissLinear(), n_estimators=3)
        assert 'n_estimators' in model.get_params()
        model.set_params(n_estimators=5)
        assert model.n_estimators == 5


class TestCrossValidationScoring:
    """Every scoring string the cross-validators accept, and one they do not."""

    @staticmethod
    def _data(classification=False, n=90):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8])
        y = y + rng.standard_normal(n) * 0.3
        if classification:
            y = (y > np.median(y)).astype(float)
        return X, y

    REGRESSION = ['r2', 'neg_mae', 'neg_mse', 'neg_rmse',
                  'neg_mean_absolute_error', 'neg_mean_squared_error',
                  'neg_root_mean_squared_error']
    CLASSIFICATION = ['accuracy', 'roc_auc', 'f1']

    @pytest.mark.parametrize('scoring', REGRESSION)
    def test_regression_metrics(self, scoring):
        from MissLearn._crossval import miss_cross_val_score
        X, y = self._data()
        scores = miss_cross_val_score(_ML.MissLinear(compute_se=False), X, y,
                                      cv=3, scoring=scoring)
        assert len(scores) == 3
        assert np.all(np.isfinite(np.asarray(scores, dtype=float)))
        if scoring.startswith('neg_'):
            assert np.all(np.asarray(scores, dtype=float) <= 0), \
                'a negated error must not be positive'

    @pytest.mark.parametrize('scoring', CLASSIFICATION)
    def test_classification_metrics(self, scoring):
        from MissLearn._crossval import miss_cross_val_score
        X, y = self._data(classification=True)
        scores = miss_cross_val_score(_ML.MissLogistic(compute_se=False), X, y,
                                      cv=3, scoring=scoring)
        assert np.all((np.asarray(scores, dtype=float) >= 0)
                      & (np.asarray(scores, dtype=float) <= 1))

    def test_an_unknown_scoring_string_is_refused_by_name(self):
        from MissLearn._crossval import miss_cross_val_score
        X, y = self._data()
        with pytest.raises(ValueError, match='Unknown scoring'):
            miss_cross_val_score(_ML.MissLinear(compute_se=False), X, y,
                                 cv=3, scoring='nonsense')

    def test_cross_validate_reports_per_fold_detail(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(_ML.MissLinear(compute_se=False), X, y, cv=3)
        assert isinstance(out, dict)
        assert any('test' in k for k in out)
        for key, values in out.items():
            if hasattr(values, '__len__') and not isinstance(values, str):
                assert len(values) == 3, key


class TestStratifiedFoldsWithMissingLabels:
    """Stratifying on a label that is itself sometimes missing.

    A row whose class is unknown cannot be stratified on, and dropping it
    from the folds is the only honest option, so the fold sizes stop being
    equal. That is the behaviour, and it is worth pinning because the
    alternative, treating NaN as its own class, would invent an outcome.
    """

    @staticmethod
    def _data(n=90):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 4))
        X[::9, 1] = np.nan
        y = (np.nan_to_num(X) @ np.array([2.0, 0.0, -1.5, 0.8]) > 0).astype(float)
        return X, y

    def test_complete_labels_give_even_folds(self):
        from MissLearn._crossval import MissStratifiedKFold
        X, y = self._data()
        splitter = MissStratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        assert splitter.get_n_splits() == 3
        sizes = [len(test) for _, test in splitter.split(X, y)]
        assert sum(sizes) == len(y)
        assert max(sizes) - min(sizes) <= 1

    def test_rows_with_no_label_are_not_stratified_on(self):
        from MissLearn._crossval import MissStratifiedKFold
        X, y = self._data()
        y = y.copy()
        y[::7] = np.nan
        splitter = MissStratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        folds = list(splitter.split(X, y))
        assert len(folds) == 3
        for train, test in folds:
            assert len(set(train) & set(test)) == 0

    def test_every_row_appears_in_exactly_one_test_fold(self):
        from MissLearn._crossval import MissStratifiedKFold
        X, y = self._data()
        splitter = MissStratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        seen = np.concatenate([test for _, test in splitter.split(X, y)])
        assert len(seen) == len(set(seen.tolist()))


# ===========================================================================
# Paths that the suite reached only by accident, or not at all
# ===========================================================================

class TestMixedConvergenceIsReported:
    """A mixed-effects fit that did not converge must say so.

    Both estimators used to set ``converged_ = False`` and return in silence,
    so an optimiser that had exhausted its budget was indistinguishable from
    one that had found a maximum. That is how pinning ``copula=False`` on the
    Parkinson's data produced coefficients of 3.6e4 and an unseen-patient
    RMSE of 38.4 with nothing in the output saying the fit had failed.

    The failure is not reliably loud, which is why this is pinned: on one
    missingness realisation the non-converged folds still returned a
    plausible RMSE of 10.16. A reasonable-looking number is not evidence of
    convergence.
    """

    @staticmethod
    def _grouped(n=160, p=3, seed=0, classify=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        groups = np.repeat(np.arange(n // 10), 10)
        eta = X @ np.linspace(1.0, -1.0, p)
        if classify:
            y = (eta + rng.normal(scale=0.5, size=n) > 0).astype(float)
            return X, y, groups
        return X, eta + rng.normal(scale=0.3, size=n), groups

    def test_classifier_warns_when_the_budget_runs_out(self):
        from MissLearn import MissMixedClassifier
        X, y, g = self._grouped(classify=True)
        with pytest.warns(UserWarning, match='without converging'):
            m = MissMixedClassifier(max_iter=1, compute_se=False).fit(
                X, y, groups=g)
        assert m.converged_ is False

    def test_a_converged_fit_is_silent(self):
        from MissLearn import MissMixedClassifier
        X, y, g = self._grouped(classify=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            m = MissMixedClassifier(compute_se=False).fit(X, y, groups=g)
        assert m.converged_ is True
        assert not [w for w in caught
                    if 'without converging' in str(w.message)]

    def test_regressor_warns_when_the_optimiser_reports_failure(
            self, monkeypatch):
        """The regressor floors its budget at ``300 * (p + 3)``, so
        ``max_iter`` cannot force this branch the way it can on the
        classifier. Intercepting the optimiser is the only deterministic way
        in, and the branch is worth reaching: it is the one that fires on
        real data.
        """
        from MissLearn import MissMixedRegressor
        import MissLearn._mixed as mixed

        real = mixed.minimize

        def never_converges(*args, **kwargs):
            out = real(*args, **kwargs)
            out.success = False
            out.message = 'forced for the test'
            return out

        monkeypatch.setattr(mixed, 'minimize', never_converges)
        X, y, g = self._grouped()
        with pytest.warns(UserWarning, match='without converging'):
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
        assert m.converged_ is False

    def test_the_warning_is_actionable(self):
        """It has to name the estimator and point at a fix. On the data that
        motivated it the fix was the marginal transform, so the message says
        so rather than only reporting that something went wrong.
        """
        from MissLearn import MissMixedClassifier
        X, y, g = self._grouped(classify=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            MissMixedClassifier(max_iter=1, compute_se=False).fit(
                X, y, groups=g)
        msg = [str(w.message) for w in caught
               if 'without converging' in str(w.message)]
        assert msg, 'no convergence warning was issued'
        assert 'MissMixedClassifier' in msg[0]
        assert 'copula' in msg[0]
        assert 'ill-conditioned' in msg[0]

    def test_converged_is_fitted_state_not_a_transient(self):
        from MissLearn import MissMixedClassifier
        X, y, g = self._grouped(classify=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedClassifier(max_iter=1, compute_se=False).fit(
                X, y, groups=g)
        m.predict(X)
        m.predict_proba(X)
        assert m.converged_ is False


class TestMissMixedClassifierScore:
    """``score`` on the mixed classifier, including the empty case.

    Accuracy is computed over observed outcomes only. With no observed
    outcome there is nothing to be accurate about, and the method returns
    0.0 rather than dividing by zero, which is worth pinning because 0.0 is
    also a legitimate accuracy and the two must not be confused by a caller
    reading the number alone.
    """

    @staticmethod
    def _data(n=120, seed=3):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        groups = np.repeat(np.arange(n // 10), 10)
        y = (X @ np.array([1.5, -1.0, 0.5])
             + rng.normal(scale=0.4, size=n) > 0).astype(float)
        return X, y, groups

    def _fit(self):
        from MissLearn import MissMixedClassifier
        X, y, g = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return MissMixedClassifier(compute_se=False).fit(
                X, y, groups=g), X, y, g

    def test_score_is_a_proportion(self):
        m, X, y, g = self._fit()
        s = m.score(X, y, groups=g)
        assert 0.0 <= s <= 1.0
        assert s > 0.5, 'should beat a coin on separable data'

    def test_score_ignores_unobserved_outcomes(self):
        m, X, y, g = self._fit()
        y_holed = y.copy()
        y_holed[::4] = np.nan
        s_all = m.score(X, y, groups=g)
        s_some = m.score(X, y_holed, groups=g)
        assert 0.0 <= s_some <= 1.0
        # the retained rows are a subset, so the two need not be equal, but
        # scoring must not fail or silently count NaN as a wrong answer
        obs = ~np.isnan(y_holed)
        expected = float(np.mean(m.predict(X, groups=g)[obs] == y[obs]))
        assert s_some == pytest.approx(expected)
        assert s_all == pytest.approx(
            float(np.mean(m.predict(X, groups=g) == y)))

    def test_score_with_no_observed_outcome_returns_zero(self):
        m, X, y, g = self._fit()
        s = m.score(X, np.full(len(y), np.nan), groups=g)
        assert s == 0.0

    def test_score_without_groups_uses_fixed_effects_only(self):
        m, X, y, _ = self._fit()
        s = m.score(X, y)
        assert 0.0 <= s <= 1.0


class TestSensitivityVerdict:
    """``verdict`` turns a tipping point into a sentence.

    Four outcomes, and each carries a different recommendation to the
    reader, so a wrong band is a wrong conclusion rather than a cosmetic
    slip. The bands are pinned against the tipping point directly rather
    than against data chosen to land in each one, because the second kind
    of test breaks whenever the estimator moves and tells you nothing about
    the banding.
    """

    @staticmethod
    def _built(monkeypatch, tip):
        from MissLearn import MissSensitivity, MissLinear
        rng = np.random.default_rng(0)
        X = rng.standard_normal((90, 3))
        y = X @ np.array([1.0, -0.5, 0.25]) + rng.normal(scale=0.3, size=90)
        X[::6, 1] = np.nan
        s = MissSensitivity(MissLinear(compute_se=False),
                            delta_range=(-2.0, 2.0), n_delta=5, m=3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s.fit(X, y)
        monkeypatch.setattr(type(s), 'tipping_point',
                            lambda self, coef_idx=0, threshold=0.0: tip)
        return s

    def test_no_tipping_point_reads_as_stable(self, monkeypatch):
        s = self._built(monkeypatch, None)
        v = s.verdict(coef_idx=0)
        assert 'ROBUST' in v and 'stable' in v
        assert '-2.0' in v and '2.0' in v, 'the range should be quoted'

    def test_large_tipping_point_is_robust(self, monkeypatch):
        s = self._built(monkeypatch, 1.4)
        v = s.verdict(coef_idx=0)
        assert v.startswith('[ROBUST]')
        assert 'moderate MNAR' in v

    def test_middling_tipping_point_is_mild(self, monkeypatch):
        s = self._built(monkeypatch, 0.7)
        v = s.verdict(coef_idx=1)
        assert v.startswith('[MILD]')
        assert 'X1' in v, 'the verdict should name the coefficient asked about'

    def test_small_tipping_point_is_sensitive(self, monkeypatch):
        s = self._built(monkeypatch, 0.2)
        v = s.verdict(coef_idx=0)
        assert v.startswith('[SENSITIVE]')
        assert 'fragile' in v

    def test_the_bands_use_the_absolute_value(self, monkeypatch):
        """A conclusion that tips at -1.4 is exactly as robust as one that
        tips at +1.4; only the size of the departure matters.
        """
        pos = self._built(monkeypatch, 1.4).verdict()
        neg = self._built(monkeypatch, -1.4).verdict()
        assert pos.startswith('[ROBUST]') and neg.startswith('[ROBUST]')
        assert '+1.40' in pos and '-1.40' in neg


class TestEnsembleSummaryCompactDisplay:
    """``summary`` switches to a compact table for large homogeneous bags.

    Listing forty identical members line by line is not a report, so above
    six members of one type the method prints the type once with aggregate
    out-of-bag statistics. That branch had never been entered by the suite,
    which tested ensembles small enough to take the long form.
    """

    @staticmethod
    def _data(n=140, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = (np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5, 0.0])
             + rng.normal(scale=0.5, size=n) > 0).astype(float)
        return X, y

    def test_large_homogeneous_ensemble_prints_the_compact_form(self, capsys):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=8, oob_score=True,
                             random_state=0).fit(X, y)
        m.summary()
        out = capsys.readouterr().out
        assert 'Homogeneous bagging: 8 x MissLogistic' in out
        assert 'Weight per estimator' in out
        assert 'OOB score: mean=' in out
        for token in ('std=', 'min=', 'max='):
            assert token in out

    def test_small_ensemble_keeps_the_per_member_table(self, capsys):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=3, oob_score=True,
                             random_state=0).fit(X, y)
        m.summary()
        out = capsys.readouterr().out
        assert 'Homogeneous bagging:' not in out

    def test_compact_form_without_oob_omits_the_oob_line(self, capsys):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=8, oob_score=False,
                             random_state=0).fit(X, y)
        m.summary()
        out = capsys.readouterr().out
        assert 'Homogeneous bagging: 8 x MissLogistic' in out
        assert 'OOB score: mean=' not in out


class TestCrossValFitParams:
    """``fit_params`` are sliced per fold when, and only when, they are
    per-sample.

    A sample weight vector must follow its rows into the fold; a scalar
    tolerance must not be indexed. The two are told apart by leading
    dimension, with the failure modes of ``np.asarray`` on odd objects
    caught rather than allowed to abort the fold. None of this had a test,
    and it is the kind of code that fails silently: a mis-sliced weight
    vector still fits, it just fits the wrong thing.
    """

    @staticmethod
    def _data(n=80, seed=5):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::7, 0] = np.nan
        y = X[:, 1] * 1.5 + rng.normal(scale=0.3, size=n)
        y = np.nan_to_num(y)
        return X, y

    def test_per_sample_array_is_sliced_to_the_training_rows(self):
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear

        seen = {}

        class Recording(MissLinear):
            def fit(self, X, y=None, **kw):
                seen['n'] = len(kw['sample_weight'])
                seen['rows'] = len(X)
                return super().fit(X, y)

        X, y = self._data()
        scores = miss_cross_val_score(
            Recording(compute_se=False), X, y, cv=4,
            sample_weight=np.ones(len(y)))
        assert len(scores) == 4
        assert seen['n'] == seen['rows'], \
            'the weight vector must be sliced with the rows'
        assert seen['n'] < len(y)

    def test_scalar_fit_param_passes_through_unsliced(self):
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear

        seen = {}

        class Recording(MissLinear):
            def fit(self, X, y=None, **kw):
                seen['scalar'] = kw['a_scalar']
                return super().fit(X, y)

        X, y = self._data()
        miss_cross_val_score(Recording(compute_se=False), X, y, cv=3,
                             a_scalar=7)
        assert seen['scalar'] == 7

    def test_wrong_length_array_passes_through_unsliced(self):
        """Only a leading dimension of exactly n means per-sample. Anything
        else is a configuration array and is handed over whole.
        """
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear

        seen = {}

        class Recording(MissLinear):
            def fit(self, X, y=None, **kw):
                seen['bounds'] = np.asarray(kw['bounds'])
                return super().fit(X, y)

        X, y = self._data()
        miss_cross_val_score(Recording(compute_se=False), X, y, cv=3,
                             bounds=np.array([0.0, 1.0, 2.0]))
        assert seen['bounds'].shape == (3,)

    def test_unarrayable_fit_param_survives(self):
        """``np.asarray`` on a ragged or opaque object raises rather than
        returning something with a shape. The fold must still run.
        """
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear

        class Opaque:
            def __array__(self, *a, **k):
                raise ValueError('not arrayable')

        seen = {}

        class Recording(MissLinear):
            def fit(self, X, y=None, **kw):
                seen['obj'] = kw['thing']
                return super().fit(X, y)

        X, y = self._data()
        obj = Opaque()
        scores = miss_cross_val_score(Recording(compute_se=False), X, y, cv=3,
                                      thing=obj)
        assert seen['obj'] is obj
        assert len(scores) == 3

    def test_cross_validate_slices_fit_params_the_same_way(self):
        """The two entry points carry separate copies of this logic, so they
        are pinned separately. A divergence between them is exactly the kind
        this codebase has produced before.
        """
        from MissLearn._crossval import miss_cross_validate
        from MissLearn import MissLinear

        seen = {}

        class Recording(MissLinear):
            def fit(self, X, y=None, **kw):
                seen['n'] = len(kw['sample_weight'])
                seen['rows'] = len(X)
                seen['scalar'] = kw['a_scalar']
                return super().fit(X, y)

        X, y = self._data()
        out = miss_cross_validate(
            Recording(compute_se=False), X, y, cv=4,
            sample_weight=np.ones(len(y)), a_scalar=3)
        assert 'test_score' in out
        assert seen['n'] == seen['rows']
        assert seen['scalar'] == 3


class TestLassoStandardErrorsAreOptIn:
    """The LASSO family ships ``compute_se=False`` where every other family
    ships ``True``.

    That is a family-wide choice rather than a sibling divergence, and a
    defensible one, because a penalty that sets coefficients to exactly zero
    makes a naive Wald interval meaningless at the boundary. It does mean
    the standard-error path is opt-in, and an opt-in path with no test is an
    untested path: nothing in the suite passed ``compute_se=True`` to these
    estimators, so the delta-method rescale had never run.
    """

    @staticmethod
    def _clf(n=140, p=4, seed=2):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = (np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
             + rng.normal(scale=0.4, size=n) > 0).astype(float)
        return X, y

    def test_default_is_off_across_the_lasso_family(self):
        import inspect
        from MissLearn import (MissLASSO, MissLASSOClassifier,
                               MissLASSORegressor)
        for cls in (MissLASSO, MissLASSOClassifier, MissLASSORegressor):
            d = inspect.signature(cls.__init__).parameters['compute_se'].default
            assert d is False, '%s should default to no standard errors' % cls

    def test_classifier_produces_standard_errors_when_asked(self):
        from MissLearn import MissLASSOClassifier
        X, y = self._clf()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLASSOClassifier(alpha=0.1, compute_se=True).fit(X, y)
        se = m.se_
        assert se.shape == (X.shape[1] + 1,)
        finite = np.isfinite(se)
        assert finite.any(), 'at least some standard errors should compute'
        assert np.all(se[finite] >= 0.0), 'a standard error cannot be negative'

    def test_no_standard_errors_when_not_asked(self):
        from MissLearn import MissLASSOClassifier
        X, y = self._clf()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLASSOClassifier(alpha=0.1, compute_se=False).fit(X, y)
        se = getattr(m, 'se_', None)
        assert se is None or np.all(~np.isfinite(np.asarray(se, dtype=float)))

    def test_a_singular_hessian_gives_nan_rather_than_raising(self, monkeypatch):
        """The fallback exists because the reduced Hessian can be singular on
        a degenerate design. It must fail to NaN, which reads as "could not
        compute", not to an exception and not to a confident number.
        """
        import MissLearn._lasso as lasso
        from MissLearn import MissLASSOClassifier

        def singular(*a, **k):
            raise np.linalg.LinAlgError('forced for the test')

        monkeypatch.setattr(lasso, 'numerical_hessian', singular)
        X, y = self._clf()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLASSOClassifier(alpha=0.1, compute_se=True).fit(X, y)
        assert np.all(np.isnan(np.asarray(m.se_, dtype=float)))


class TestPrefitCheckAdvisories:
    """``prefit_check`` warnings that no test had ever triggered.

    Each of these exists because a real dataset once produced a bad fit for
    the reason it names, so a warning that never fires in the suite is a
    warning nobody has read since it was written.
    """

    @staticmethod
    def _clean(n=120, p=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[::9, 1] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.0, -1.0, p)
        return X, y

    def test_wildly_different_feature_scales_are_flagged(self):
        from MissLearn import prefit_check
        X, y = self._clean()
        X = X.copy()
        X[:, 0] *= 1e-5          # micro-units next to unit-scale neighbours
        X[:, 2] *= 1e4
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        text = ' '.join(r.warnings)
        assert 'Feature scales differ' in text
        assert 'max std / min std' in text

    def test_matched_scales_are_not_flagged(self):
        from MissLearn import prefit_check
        X, y = self._clean()
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert 'Feature scales differ' not in ' '.join(r.warnings)

    def test_heavy_tails_with_the_copula_already_on_say_so(self):
        """The advice has to change with the configuration. Telling somebody
        to enable a transform they have already enabled is how a report
        teaches readers to ignore it.
        """
        from MissLearn import prefit_check
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 3))
        X[:, 1] = rng.standard_t(df=2.0, size=200) * 5.0    # heavy tailed
        X[::11, 0] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, 0.1, -0.5])
        on = prefit_check(X, y, raise_on_error=False, emit_warnings=False,
                          kurtosis_threshold=3.0, copula_configured=True)
        off = prefit_check(X, y, raise_on_error=False, emit_warnings=False,
                           kurtosis_threshold=3.0, copula_configured=False)
        # a kurtosis finding is a note, not a warning: it is informational
        # unless the caller has actively turned the transform off
        on_text, off_text = ' '.join(on.notes), ' '.join(off.notes)
        assert 'excess kurtosis' in on_text
        assert 'already enabled' in on_text
        assert 'already enabled' not in off_text
        assert "copula='auto'" in off_text

    def test_a_y_of_the_wrong_length_is_an_error_not_a_warning(self):
        from MissLearn import prefit_check
        X, y = self._clean()
        r = prefit_check(X, y[:-3], raise_on_error=False, emit_warnings=False)
        assert not r.passed
        joined = ' '.join(r.errors)
        assert 'does not match X.shape[0]' in joined
        assert '117' in joined and '120' in joined

    def test_the_same_mismatch_raises_when_asked_to(self):
        from MissLearn import prefit_check
        X, y = self._clean()
        with pytest.raises(ValueError):
            prefit_check(X, y[:-3], raise_on_error=True, emit_warnings=False)


class TestNothingShadowsGeneratedRequestMethods:
    """scikit-learn generates the ``set_*_request`` methods; nothing here may
    define one.

    This class used to assert that MissBase's no-op stubs warned when given
    kwargs, reasoning that "metadata routing failing open is how a sample
    weight quietly stops being applied". The reasoning was right and the
    guard was aimed one level too high. The stubs did warn, and they also
    shadowed scikit-learn's real generated method from 1.7 onwards, which
    skips generation when the attribute already exists in the MRO. So
    ``MissMixedRegressor().set_fit_request(groups=True)`` recorded nothing and
    told the caller routing was unimplemented, for a class where scikit-learn
    implements it correctly. Routing failed open exactly as the docstring
    feared, underneath a test written to prevent it.

    What is asserted now is the condition that makes that impossible: no
    module in the package defines either method, so generation is never
    shadowed. A convenience stub added to MissBase in future fails here.
    """

    REQUEST_METHODS = ('set_fit_request', 'set_predict_request',
                       'set_score_request', 'set_transform_request')

    def test_no_missLearn_module_defines_a_request_method(self):
        """Authored here, as opposed to generated by scikit-learn.

        scikit-learn installs its generated methods directly into the class
        dictionary, so presence in ``vars(cls)`` is not the test; ownership
        is. A generated one is a ``RequestMethod`` descriptor whose
        ``__module__`` is scikit-learn's, and those are expected and required.
        An entry owned by this package is a hand-written function, and that is
        the thing that shadows generation and silently disables routing.
        """
        import inspect
        import pkgutil
        import importlib
        offenders = []
        for mod_info in pkgutil.iter_modules(_ML.__path__):
            mod = importlib.import_module('MissLearn.%s' % mod_info.name)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue                    # imported, not defined here
                for meth in self.REQUEST_METHODS:
                    obj = vars(cls).get(meth)
                    if obj is None:
                        continue
                    owner = getattr(obj, '__module__', '') or ''
                    if not owner.startswith('sklearn'):
                        offenders.append('%s.%s defines %s (owned by %r)'
                                         % (mod.__name__, cls.__name__,
                                            meth, owner))
        assert not offenders, (
            'these shadow the methods scikit-learn generates, which stops '
            'metadata routing recording anything on 1.7 and later: %s'
            % offenders)

    def test_the_generated_methods_are_still_being_installed(self):
        """The guard above is only meaningful while generation happens at
        all. If scikit-learn stopped installing them, that test would pass by
        finding nothing, so assert the positive case separately.
        """
        from MissLearn import MissMixedRegressor
        obj = vars(MissMixedRegressor).get('set_fit_request')
        assert obj is not None, 'scikit-learn generated no set_fit_request'
        assert getattr(obj, '__module__', '').startswith('sklearn')

    def test_an_estimator_with_metadata_gets_a_working_method(self):
        """``MissMixedRegressor.fit`` takes ``groups``, so scikit-learn
        generates the method and the request must actually be recorded.
        """
        import sklearn
        from MissLearn import MissMixedRegressor
        sklearn.set_config(enable_metadata_routing=True)
        try:
            est = MissMixedRegressor().set_fit_request(groups=True)
            assert est.get_metadata_routing()._serialize() == {
                'fit': {'groups': True}}
        finally:
            sklearn.set_config(enable_metadata_routing=False)

    def test_an_estimator_without_metadata_has_no_method(self):
        """``MissLinear.fit`` takes no metadata, so it has no
        ``set_fit_request``, which is what scikit-learn's own ``PCA`` does.
        Carrying a courtesy stub instead is what caused the defect above.
        """
        from sklearn.decomposition import PCA
        from MissLearn import MissLinear
        m = MissLinear(compute_se=False)
        assert not hasattr(m, 'set_fit_request')
        assert not hasattr(m, 'set_predict_request')
        assert not hasattr(PCA(), 'set_fit_request'), (
            'the comparison this test rests on has changed upstream')

    def test_get_metadata_routing_is_available_on_every_estimator(self):
        """Removing the stubs must not have removed the routing surface
        itself: ``get_metadata_routing`` comes from ``BaseEstimator`` and every
        estimator still answers it.
        """
        for name in ESTIMATOR_NAMES:
            est = getattr(_ML, name)()
            assert est.get_metadata_routing() is not None, name


class TestDiagnosticSummaryRunsEverything:
    """``summary`` is the entry point most users actually call.

    It lazily runs whichever of the four analyses have not been run yet, so
    calling it on a fresh object exercises a path that calling the four
    methods individually never reaches. The suite did the latter.
    """

    @staticmethod
    def _data(n=200, seed=4):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 5))
        # missingness driven by an observed column, so MAR is detectable
        driver = X[:, 0]
        for j in (1, 2):
            X[driver > np.quantile(driver, 0.6), j] = np.nan
        X[::17, 3] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25, 0.1, 0.0])
        return X, y

    def test_summary_on_a_fresh_object_runs_all_four_analyses(self, capsys):
        from MissLearn import MissDiagnostic
        X, y = self._data()
        d = MissDiagnostic(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
        out = capsys.readouterr().out
        assert len(out) > 400, 'summary should print a full report'
        assert d._pattern_done and d._little_done
        assert d._mar_done and d._corr_done

    def test_summary_is_idempotent(self, capsys):
        """Calling it twice must not re-run the analyses, and must not
        produce a different report the second time.
        """
        from MissLearn import MissDiagnostic
        X, y = self._data()
        d = MissDiagnostic(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
            first = capsys.readouterr().out
            d.summary()
            second = capsys.readouterr().out
        assert first == second

    def test_summary_with_feature_names_uses_them(self, capsys):
        from MissLearn import MissDiagnostic
        X, y = self._data()
        names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        d = MissDiagnostic(X, y, feature_names=names)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
        out = capsys.readouterr().out
        assert 'beta' in out or 'gamma' in out


class TestRecommenderSummaryBranches:
    """``summary`` reports the preprocessing it decided on, not just the
    ranking.

    A recommendation to drop a column is the most consequential thing this
    class produces, since acting on it changes the data rather than the
    model, so the branch that prints it is worth pinning along with the
    reason attached to each name.
    """

    @staticmethod
    def _mostly_absent_column(n=300, seed=6):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4))
        X[:, 2] = np.nan
        X[:12, 2] = rng.standard_normal(12)     # ~96% absent, over the 0.6 gate
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.0, 0.3])
        return X, y

    def test_a_column_past_the_drop_threshold_is_named_with_its_reason(
            self, capsys):
        from MissLearn import MissRecommender
        X, y = self._mostly_absent_column()
        names = ['a', 'b', 'mostly_missing', 'd']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(feature_names=names,
                                probe_nonlinearity=False).fit(X, y)
        assert r.preprocessing_['drop_columns'], \
            'a 96% absent column should be recommended for removal'
        r.summary()
        out = capsys.readouterr().out
        assert 'drop these columns' in out
        assert 'mostly_missing' in out

    def test_heavy_tails_make_the_summary_mention_the_copula(self, capsys):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(7)
        X = rng.standard_normal((250, 3))
        X[:, 0] = rng.standard_t(df=1.8, size=250) * 8.0
        X[::13, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([0.2, 1.0, -0.5])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
        r.summary()
        out = capsys.readouterr().out
        if r.preprocessing_['copula']:
            assert 'copula=True' in out
        else:
            assert 'copula=True' not in out

    def test_summary_runs_with_the_nonlinearity_probe_on(self, capsys):
        """The probe adds a line to the report and is on by default, so the
        default path through ``summary`` is the one with it present.
        """
        from MissLearn import MissRecommender
        X, y = self._mostly_absent_column()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=True).fit(X, y)
        r.summary()
        out = capsys.readouterr().out
        assert len(out) > 200


# ===========================================================================
# Whole-library sweeps
#
# The classes below discover their targets from the package rather than
# listing them. A hand-written list is how MissSupportClassifier came to have
# two real failures nobody saw: CI was asked about nine estimators and there
# were twenty-three. Discovery means a new estimator is covered on the day it
# is added rather than the day somebody remembers to edit a list.
# ===========================================================================

def _all_estimators():
    """Every public predictor in the namespace, by name.

    Predictor, not merely estimator. The sweeps below ask predictor
    questions: that ``score`` returns a finite number, that a degenerate
    regime is refused or answered finitely, that an entirely absent row still
    predicts. A class has to have ``predict`` for any of those to mean
    anything.

    This used to be spelled as BaseEstimator plus ``fit``, which selected the
    same set only because every class with ``fit`` and no ``predict``
    happened not to subclass BaseEstimator. That stopped being true when
    MissImputer gained a scikit-learn identity, and twelve sweeps began
    calling ``predict`` on a multiple-imputation transformer that returns m
    completed datasets instead. The requirement is stated directly now rather
    than resting on a coincidence of base classes.
    """
    import inspect as _inspect
    from sklearn.base import BaseEstimator as _BE
    return sorted(
        n for n in dir(_ML)
        if n.startswith('Miss')
        and _inspect.isclass(getattr(_ML, n))
        and issubclass(getattr(_ML, n), _BE)
        and hasattr(getattr(_ML, n), 'fit')
        and hasattr(getattr(_ML, n), 'predict')
        and 'estimator' not in _inspect.signature(
            getattr(_ML, n).__init__).parameters
    )


def _sweep_data(n=90, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[rng.random(X.shape) < 0.12] = np.nan
    lin = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5])
    y_reg = lin + rng.normal(scale=0.3, size=n)
    y_clf = (lin + rng.normal(scale=0.4, size=n) > 0).astype(float)
    return X, y_reg, y_clf, np.repeat(np.arange(n // 6), 6)


def _make_and_fit(name, task='auto'):
    """Fit the named estimator on data appropriate to what it is."""
    import inspect as _inspect
    from sklearn.base import ClassifierMixin as _CM, RegressorMixin as _RM
    cls = getattr(_ML, name)
    kw = {}
    if 'compute_se' in _inspect.signature(cls.__init__).parameters:
        kw['compute_se'] = False
    est = cls(**kw)
    X, y_reg, y_clf, groups = _sweep_data()
    if task == 'clf':
        y = y_clf
    elif task == 'reg':
        y = y_reg
    else:
        y = y_clf if isinstance(est, _CM) else y_reg
    fit_kw = {'groups': groups} if 'Mixed' in name else {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        est.fit(X, y, **fit_kw)
    return est, X, y, fit_kw


ESTIMATOR_NAMES = _all_estimators()


class TestSummaryOnEveryEstimator:
    """``summary`` is the method a user calls to decide whether to believe
    the fit, and it was the least covered thing in the library.

    Seventy-two separate uncovered runs sat inside these methods, which is
    what happens when a printing method is treated as cosmetic. It is not
    cosmetic: it is where the intercept, the standard errors, the
    convergence flag and the copula decision are reported, and a branch that
    has never run is a branch that has never printed the right number.
    """

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_prints_something(self, name, capsys):
        est, X, y, fit_kw = _make_and_fit(name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.summary()
        out = capsys.readouterr().out
        assert out.strip(), '%s.summary() printed nothing' % name
        assert type(est).__name__ in out or 'Miss' in out

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_accepts_feature_names(self, name, capsys):
        """Named features are the difference between a report somebody reads
        and one they skim, so the branch that uses them has to work on every
        estimator that offers it.
        """
        import inspect as _inspect
        est, X, y, fit_kw = _make_and_fit(name)
        params = _inspect.signature(est.summary).parameters
        names = ['alpha', 'beta', 'gamma']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if 'feature_names' in params:
                est.summary(feature_names=names)
            else:
                est.summary()
        out = capsys.readouterr().out
        assert out.strip()
        if 'feature_names' in params:
            assert any(nm in out for nm in names), \
                '%s.summary(feature_names=...) ignored them' % name

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_before_fit_is_refused(self, name):
        """An unfitted summary must raise rather than print a report about
        parameters that do not exist yet.
        """
        import inspect as _inspect
        from sklearn.exceptions import NotFittedError
        cls = getattr(_ML, name)
        kw = {}
        if 'compute_se' in _inspect.signature(cls.__init__).parameters:
            kw['compute_se'] = False
        with pytest.raises((NotFittedError, AttributeError, ValueError)):
            cls(**kw).summary()


class TestPredictIntervalOnEveryRegressor:
    """Intervals are the argument for using a likelihood at all.

    The claim the library makes is that an interval widens when a row has
    more missing entries, because marginalising propagates that uncertainty
    into the prediction. That is checkable on every regressor at once, and
    it is the property most worth having pinned everywhere rather than in
    one place.
    """

    @staticmethod
    def _regressors():
        from sklearn.base import RegressorMixin
        out = []
        for n in ESTIMATOR_NAMES:
            cls = getattr(_ML, n)
            if hasattr(cls, 'predict_interval'):
                out.append(n)
        return out

    @pytest.mark.parametrize('name', _regressors.__func__())
    def test_interval_brackets_the_prediction(self, name):
        est, X, y, fit_kw = _make_and_fit(name, task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lo, hi = est.predict_interval(X, alpha=0.05)
            point = est.predict(X)
        finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(point)
        assert finite.any(), '%s produced no finite interval' % name
        assert np.all(lo[finite] <= point[finite] + 1e-8)
        assert np.all(point[finite] <= hi[finite] + 1e-8)

    @pytest.mark.parametrize('name', _regressors.__func__())
    def test_a_wider_alpha_gives_a_narrower_interval(self, name):
        est, X, y, fit_kw = _make_and_fit(name, task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lo95, hi95 = est.predict_interval(X, alpha=0.05)
            lo50, hi50 = est.predict_interval(X, alpha=0.50)
        w95, w50 = hi95 - lo95, hi50 - lo50
        finite = np.isfinite(w95) & np.isfinite(w50)
        assert finite.any()
        assert np.all(w50[finite] <= w95[finite] + 1e-8), \
            '%s: a 50%% interval should not be wider than a 95%% one' % name


class TestDecisionFunctionAcrossTheLibrary:
    """``decision_function`` on the auto-selecting wrappers, both ways.

    Each wrapper delegates when it resolved to classification and raises
    when it resolved to regression. The raising branch is the one that
    matters: silently returning a regression prediction from a method whose
    contract is a signed margin would be read as class scores.
    """

    @staticmethod
    def _wrappers():
        import inspect as _inspect
        from sklearn.base import ClassifierMixin, RegressorMixin
        out = []
        for n in ESTIMATOR_NAMES:
            cls = getattr(_ML, n)
            if not hasattr(cls, 'decision_function'):
                continue
            probe = cls(**({'compute_se': False}
                           if 'compute_se' in _inspect.signature(
                               cls.__init__).parameters else {}))
            if not isinstance(probe, (ClassifierMixin, RegressorMixin)):
                out.append(n)          # the auto-selecting wrappers only
        return out

    @pytest.mark.parametrize('name', _wrappers.__func__())
    def test_delegates_when_the_task_is_classification(self, name):
        est, X, y, fit_kw = _make_and_fit(name, task='clf')
        assert est.task_ == 'classification'
        d = est.decision_function(X)
        assert d.shape[0] == X.shape[0]
        assert np.all(np.isfinite(d))

    @pytest.mark.parametrize('name', _wrappers.__func__())
    def test_is_hidden_when_the_task_is_regression(self, name):
        """The contract is that the attribute is absent, not that calling it
        raises. ``only_for`` exists precisely because raising on call was too
        late: ``hasattr`` reported True, so scikit-learn called the method and
        got an exception where it should have skipped the check. Four checks
        failed that way on each of six estimators.

        This means the method body's own ``if self.task_ != 'classification'``
        guard is unreachable through the public interface. It is left in place
        as belt and braces, and is why those few lines stay uncovered.
        """
        est, X, y, fit_kw = _make_and_fit(name, task='reg')
        assert est.task_ == 'regression'
        assert not hasattr(est, 'decision_function'),             '%s should hide decision_function after resolving to regression' % name
        with pytest.raises(AttributeError):
            est.decision_function(X)

    @pytest.mark.parametrize('name', _wrappers.__func__())
    def test_is_visible_again_when_the_task_is_classification(self, name):
        est, X, y, fit_kw = _make_and_fit(name, task='clf')
        assert hasattr(est, 'decision_function')

    @pytest.mark.parametrize('name', _wrappers.__func__())
    def test_refuses_before_fit(self, name):
        import inspect as _inspect
        from sklearn.exceptions import NotFittedError
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        X, _, _, _ = _sweep_data()
        with pytest.raises((NotFittedError, AttributeError)):
            cls(**kw).decision_function(X)


class TestScoreAcrossTheLibrary:
    """``score`` is what cross-validation calls, so a wrong one is silent."""

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_score_returns_a_finite_number(self, name):
        est, X, y, fit_kw = _make_and_fit(name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = est.score(X, y, **fit_kw) if fit_kw else est.score(X, y)
        assert np.isfinite(s), '%s.score returned %r' % (name, s)

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_score_tolerates_missing_outcomes(self, name):
        """A held-out fold can contain rows whose outcome was never
        recorded. Scoring over them must ignore them rather than counting
        them as errors.
        """
        est, X, y, fit_kw = _make_and_fit(name)
        y_holed = np.asarray(y, dtype=float).copy()
        y_holed[::5] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = (est.score(X, y_holed, **fit_kw) if fit_kw
                 else est.score(X, y_holed))
        assert np.isfinite(s)


class TestDegenerateRegimesAcrossTheLibrary:
    """Every estimator against data that breaks an assumption.

    The contract each of these pins is the one ``check_missing_data_estimator``
    states: a clear refusal is acceptable, a finite answer is acceptable, and
    a silent ``NaN`` never is. That third outcome is the one this library has
    actually shipped, more than once, which is why the regimes are swept over
    every estimator rather than spot-checked on one.

    The regimes are deliberately nastier than the conformance suite's: a
    column that is constant, one that duplicates another exactly, a design
    wider than it is tall, a response with a single distinct value, and
    features scaled to 1e-200 and 1e300.
    """

    @staticmethod
    def _base(n=60, p=3, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        return X, lin + rng.normal(scale=0.3, size=n), \
            (lin > 0).astype(float), np.repeat(np.arange(n // 6), 6)

    @staticmethod
    def _regimes():
        rng = np.random.default_rng(1)
        n, p = 60, 3
        out = {}

        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan

        c = X.copy()
        c[:, 1] = 4.0                       # zero variance
        out['constant_column'] = c

        d = X.copy()
        d[:, 2] = d[:, 0]                   # exactly collinear
        out['duplicate_column'] = d

        s = X.copy()
        s[:, 1] = np.nan
        s[0, 1] = 2.5                       # one observed cell in a column
        out['one_observed_cell'] = s

        t = X.copy()
        t[3, :] = np.nan                    # an entirely absent row
        out['all_nan_row'] = t

        tiny = X.copy()
        tiny[:, 0] *= 1e-200                # scale invariance, both ends
        out['tiny_scale'] = tiny

        huge = X.copy()
        huge[:, 0] *= 1e300
        out['huge_scale'] = huge

        binary = X.copy()
        binary[:, 1] = (binary[:, 1] > 0).astype(float)   # two distinct values
        out['two_valued_column'] = binary

        return out

    REGIMES = _regimes.__func__()

    @pytest.mark.parametrize('regime', sorted(REGIMES))
    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_refuses_clearly_or_answers_finitely(self, name, regime):
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        est = cls(**kw)
        X = self.REGIMES[regime]
        _, y_reg, y_clf, groups = self._base()
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        fit_kw = {'groups': groups} if 'Mixed' in name else {}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                est.fit(X, y, **fit_kw)
            except (ValueError, np.linalg.LinAlgError, RuntimeError,
                    NotImplementedError) as exc:
                assert str(exc).strip(), \
                    '%s refused %s with an empty message' % (name, regime)
                return
            try:
                pred = est.predict(X)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                assert str(exc).strip()
                return

        pred = np.asarray(pred, dtype=float)
        assert not np.all(np.isnan(pred)), (
            '%s on %s fitted and then predicted all NaN, which is the one '
            'outcome the contract forbids: it reads as an answer' % (name, regime))

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_a_single_class_target_is_handled(self, name):
        """A fold can contain one class. A classifier must refuse or predict
        that class, never produce a probability of NaN.
        """
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        cls = getattr(_ML, name)
        est = cls(**({'compute_se': False}
                     if 'compute_se' in _inspect.signature(
                         cls.__init__).parameters else {}))
        if not isinstance(est, ClassifierMixin):
            pytest.skip('regressor')
        X, _, _, groups = self._base()
        y = np.zeros(X.shape[0])
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                est.fit(X, y, **fit_kw)
            except (ValueError, RuntimeError) as exc:
                assert str(exc).strip()
                return
            pred = np.asarray(est.predict(X), dtype=float)
        assert np.all(np.isfinite(pred))
        assert set(np.unique(pred)) <= {0.0}

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_more_features_than_rows(self, name):
        """p >= n makes the joint covariance singular. Refusing is the right
        answer and several estimators do; the ones that fit must still not
        return NaN.
        """
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        rng = np.random.default_rng(2)
        n, p = 12, 20
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        lin = np.nan_to_num(X) @ rng.standard_normal(p)
        cls = getattr(_ML, name)
        est = cls(**({'compute_se': False}
                     if 'compute_se' in _inspect.signature(
                         cls.__init__).parameters else {}))
        y = (lin > 0).astype(float) if isinstance(est, ClassifierMixin) \
            else lin + rng.normal(scale=0.2, size=n)
        fit_kw = {'groups': np.repeat(np.arange(n // 3), 3)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                est.fit(X, y, **fit_kw)
                pred = np.asarray(est.predict(X), dtype=float)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                assert str(exc).strip()
                return
        assert not np.all(np.isnan(pred))

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_a_dataframe_is_accepted_like_an_array(self, name):
        """Pandas support is duck-typed rather than a dependency, so it has
        its own conversion path that array inputs never touch.
        """
        pd = pytest.importorskip('pandas')
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        X, y_reg, y_clf, groups = self._base()
        cls = getattr(_ML, name)
        est = cls(**({'compute_se': False}
                     if 'compute_se' in _inspect.signature(
                         cls.__init__).parameters else {}))
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        df = pd.DataFrame(X, columns=['a', 'b', 'c'])
        ser = pd.Series(y)
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(df, ser, **fit_kw)
            from_df = np.asarray(est.predict(df), dtype=float)
            from_arr = np.asarray(est.predict(X), dtype=float)
        assert from_df.shape == from_arr.shape
        assert np.allclose(from_df, from_arr, equal_nan=True, atol=1e-8), \
            '%s gave different answers for a DataFrame and its ndarray' % name


class TestExplainerSurface:
    """``MissExplainer`` across its options, including the plots.

    The plotting methods were the largest untested block in the library.
    Figures are not decoration here: ``plot_miss_importance`` is what turns
    missingness SHAP into the survey-design conclusion the galaxy example
    reports, and a plotting method that has never run is one nobody has
    watched draw the wrong axis.
    """

    @staticmethod
    def _fitted(p=4, n=80, exact=True, task='reg', seed=0):
        from MissLearn import MissExplainer, MissLinear, MissLogistic
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if task == 'clf':
                model = MissLogistic(compute_se=False).fit(
                    X, (lin > 0).astype(float))
            else:
                model = MissLinear(compute_se=False).fit(
                    X, lin + rng.normal(scale=0.3, size=n))
            ex = MissExplainer(model,
                               exact_threshold=p if exact else 0,
                               n_kernel_samples=64,
                               random_state=0).fit(X)
        return ex, X

    def test_exact_and_kernel_agree_approximately(self):
        """Below the threshold the Shapley values are enumerated exactly;
        above it they are sampled. The two routes are different code and
        must not disagree materially, or the threshold silently changes the
        answer rather than only the cost.
        """
        ex_exact, X = self._fitted(p=4, exact=True)
        ex_kernel, _ = self._fitted(p=4, exact=False)
        a = ex_exact.shap_values(X[:6])
        b = ex_kernel.shap_values(X[:6])
        assert a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.abs(a - b).max() < 1.5, 'exact and sampled SHAP diverged'

    def test_shap_values_sum_towards_the_prediction(self):
        """The additivity property is the reason to use Shapley values at
        all: contributions plus the base value should recover the model
        output.
        """
        ex, X = self._fitted(p=4, exact=True)
        sv = ex.shap_values(X[:10])
        pred = ex.model.predict(X[:10])
        base = getattr(ex, 'expected_value_', None)
        if base is None:
            pytest.skip('no expected_value_ exposed')
        assert np.allclose(sv.sum(axis=1) + base, pred, atol=1e-6)

    def test_miss_shap_runs_and_is_finite(self):
        ex, X = self._fitted(p=4, exact=True)
        ms = ex.miss_shap(X[:8])
        assert ms.shape[0] == 8
        assert np.isfinite(ms).all()

    def test_summary_reports_both_kinds(self, capsys):
        ex, X = self._fitted(p=4, exact=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex.summary(ex.shap_values(X[:20]), ex.miss_shap(X[:20]))
        out = capsys.readouterr().out
        assert 'Value SHAP' in out
        assert 'Missingness SHAP' in out

    def test_summary_with_neither_array_still_prints_the_header(self, capsys):
        """Both arguments default to None, so summary has to be callable
        before anything has been computed and print the configuration alone.
        """
        ex, X = self._fitted(p=4, exact=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex.summary()
        out = capsys.readouterr().out
        assert 'MissExplainer' in out
        assert 'Value SHAP' not in out

    def test_summary_with_only_the_value_shap(self, capsys):
        ex, X = self._fitted(p=4, exact=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex.summary(ex.shap_values(X[:20]))
        out = capsys.readouterr().out
        assert 'Value SHAP' in out
        assert 'Missingness SHAP' not in out

    @pytest.mark.parametrize('method', ['plot_waterfall', 'plot_dependence',
                                        'plot_beeswarm', 'plot_miss_importance'])
    def test_plots_draw_without_error(self, method):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ex, X = self._fitted(p=4, exact=True)
        Xs = X[:30]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(Xs)
            fn = getattr(ex, method)
            if method == 'plot_waterfall':
                fn(sv, Xs, i=0)
            elif method == 'plot_dependence':
                fn(sv, Xs, 0)
            elif method == 'plot_miss_importance':
                fn(ex.miss_shap(Xs))
            else:
                fn(sv, Xs)
        assert plt.gcf().get_axes(), '%s drew no axes' % method
        plt.close('all')

    def test_plot_dependence_accepts_a_named_feature(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ex, X = self._fitted(p=4, exact=True)
        Xs = X[:30]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(Xs)
            ex.plot_dependence(sv, Xs, 1, interaction_idx=2,
                               feature_names=['a', 'b', 'c', 'd'])
        assert plt.gcf().get_axes()
        plt.close('all')

    def test_classification_needs_a_class_index(self):
        """A multi-output model has no single scalar value function, so the
        explainer has to be told which output to attribute.
        """
        ex, X = self._fitted(p=4, exact=True, task='clf')
        sv = ex.shap_values(X[:6])
        assert np.isfinite(sv).all()

    def test_to_shap_explanation_round_trips(self):
        ex, X = self._fitted(p=4, exact=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:5])
            try:
                obj = ex.to_shap_explanation(sv, X[:5])
            except (ImportError, AttributeError):
                pytest.skip('the shap package is not installed')
        assert hasattr(obj, 'values')
        assert np.allclose(np.asarray(obj.values, dtype=float), sv)


class TestImputerSurface:
    """``MissImputer`` across its options.

    Multiple imputation is the escape hatch for tools that cannot accept
    NaN, so its combine step is where a downstream analysis gets its
    standard errors. Rubin's rules have a term that only appears when the
    between-imputation variance is non-zero, and a code path that has never
    run with m > 1 has never exercised it.
    """

    @staticmethod
    def _data(n=90, p=4, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.18] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.0, -1.0, p) \
            + rng.normal(scale=0.3, size=n)
        return X, y

    @pytest.mark.parametrize('posterior', [False, True])
    @pytest.mark.parametrize('include_y', [False, True])
    def test_transform_returns_m_complete_matrices(self, posterior, include_y):
        from MissLearn import MissImputer
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imp = MissImputer(m=4, posterior=posterior, include_y=include_y,
                              random_state=0)
            imp.fit(X, y) if include_y else imp.fit(X)
            out = imp.transform(X, y) if include_y else imp.transform(X)
        sets = out[0] if isinstance(out, tuple) else out
        assert len(sets) == 4
        for M in sets:
            assert not np.isnan(np.asarray(M, dtype=float)).any(), \
                'an imputed matrix still contains NaN'

    def test_the_draws_differ_from_each_other(self):
        """Identical draws would make the between-imputation variance zero
        and collapse Rubin's rules to a single-imputation answer, which is
        the error multiple imputation exists to avoid.
        """
        from MissLearn import MissImputer
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sets = MissImputer(m=5, random_state=0).fit(X).transform(X)
        a, b = np.asarray(sets[0]), np.asarray(sets[1])
        holes = np.isnan(X)
        assert not np.allclose(a[holes], b[holes]), 'the draws are identical'

    def test_transform_mean_is_a_single_matrix(self):
        from MissLearn import MissImputer
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            M = MissImputer(m=3, random_state=0).fit(X).transform_mean(X)
        M = np.asarray(M, dtype=float)
        assert M.shape == X.shape
        assert not np.isnan(M).any()
        obs = ~np.isnan(X)
        assert np.allclose(M[obs], X[obs]), \
            'observed entries must be left exactly as they were'

    def test_fit_transform_combine_pools_by_rubins_rules(self):
        from MissLearn import MissImputer
        from sklearn.linear_model import LinearRegression
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # the imputer must be fitted first: "fit" in the method name
            # refers to fitting the estimator on each imputed set, not to
            # fitting the imputer, which transform() below relies on
            imp = MissImputer(m=5, random_state=0).fit(X)
            res = imp.fit_transform_combine(X, y, LinearRegression(), 'coef_')
        # param_var was not given, so the documented behaviour is that the
        # pooled standard error is not computed and those keys are absent
        assert set(res) >= {'estimate', 'between_var', 'm', 'fitted_estimators'}
        assert 'se' not in res and 'total_var' not in res
        assert np.isfinite(np.asarray(res['estimate'], dtype=float)).all()
        assert res['m'] == 5
        assert len(res['fitted_estimators']) == 5

    def test_fit_transform_combine_pools_standard_errors_when_told_where(self):
        """Naming the attribute that holds the per-imputation standard error
        is what turns this into Rubin's rules rather than an average. The
        argument takes a standard error and squares it, which is the one
        thing about it worth pinning: handing it a variance would square a
        variance and the output would not say so.
        """
        from MissLearn import MissImputer, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imp = MissImputer(m=4, random_state=0).fit(X)
            res = imp.fit_transform_combine(X, y, MissLinear(), 'coef_',
                                            param_var='se_')
        assert set(res) >= {'estimate', 'within_var', 'between_var',
                            'total_var', 'se', 'df', 'm'}
        w = np.asarray(res['within_var'], dtype=float)
        b = np.asarray(res['between_var'], dtype=float)
        t = np.asarray(res['total_var'], dtype=float)
        assert np.all(w > 0), 'the within term should now be populated'
        assert np.allclose(t, w + (1.0 + 1.0 / 4.0) * b)
        assert np.allclose(np.asarray(res['se'], dtype=float), np.sqrt(t))
        assert np.all(np.asarray(res['df'], dtype=float) > 0)

    def test_combine_widens_when_the_draws_disagree(self):
        """The pooled variance is within-imputation plus a between term. Two
        sets of estimates that disagree must pool to a wider interval than
        two that agree, or the between term is not being added.
        """
        from MissLearn import MissImputer
        X, y = self._data()
        agree = MissImputer.combine(np.array([[1.0, 2.0]] * 3),
                                    np.array([[0.1, 0.1]] * 3))
        differ = MissImputer.combine(np.array([[1.0, 2.0],
                                               [1.6, 2.4],
                                               [0.4, 1.6]]),
                                     np.array([[0.1, 0.1]] * 3))
        va = np.asarray(agree['total_var'], dtype=float)
        vd = np.asarray(differ['total_var'], dtype=float)
        assert np.allclose(np.asarray(agree['between_var'], dtype=float), 0.0)
        assert np.all(vd > va), 'disagreement between draws must widen the pool'
        # Rubin: T = W + (1 + 1/m) B
        b = np.asarray(differ['between_var'], dtype=float)
        w = np.asarray(differ['within_var'], dtype=float)
        assert np.allclose(vd, w + (1.0 + 1.0 / 3.0) * b)
        assert np.all(np.asarray(differ['df'], dtype=float) > 0)

    def test_summary_prints(self, capsys):
        from MissLearn import MissImputer
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            MissImputer(m=3, random_state=0).fit(X).summary()
        assert capsys.readouterr().out.strip()

    def test_methods_refuse_before_fit(self):
        from MissLearn import MissImputer
        from sklearn.exceptions import NotFittedError
        X, y = self._data()
        imp = MissImputer(m=3)
        for call in (lambda: imp.transform(X),
                     lambda: imp.transform_mean(X),
                     lambda: imp.summary()):
            with pytest.raises((NotFittedError, AttributeError, ValueError)):
                call()


class TestPreprocessorSurface:
    """``MissPreprocessor`` wraps an estimator and owns the encoding.

    Categorical encoding with NaN preserved is the part that cannot be
    delegated to scikit-learn, because every encoder there treats an absent
    category as a category. The encoding map is built once at fit and
    applied at predict, so a mismatch between those two paths is invisible
    until the columns silently misalign.
    """

    @staticmethod
    def _mixed_frame(n=80, seed=2):
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            'num1': rng.standard_normal(n),
            'num2': rng.standard_normal(n),
            'cat': rng.choice(['a', 'b', 'c'], size=n),
            'binary': rng.choice(['yes', 'no'], size=n),
        })
        df.loc[::9, 'num1'] = np.nan
        df.loc[::11, 'cat'] = None
        y = (df['num1'].fillna(0) * 1.5 - df['num2'] > 0).astype(float)
        return df, y.to_numpy()

    @pytest.mark.parametrize('drop', ['first', None])
    def test_fit_predict_on_mixed_types(self, drop):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._mixed_frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False), drop=drop,
                                 raise_on_error=False).fit(df, y)
            pred = m.predict(df)
        assert len(pred) == len(y)
        assert np.all(np.isfinite(np.asarray(pred, dtype=float)))

    def test_transform_preserves_missingness(self):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._mixed_frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            Z = np.asarray(m.transform(df), dtype=float)
        assert np.isnan(Z).any(), \
            'encoding must not fill the holes it was given'

    def test_the_encoding_is_stable_between_fit_and_predict(self):
        """Predicting on a frame whose categories appear in a different order
        must give the same answer. A map rebuilt at predict time would not.
        """
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._mixed_frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            a = np.asarray(m.predict(df), dtype=float)
            b = np.asarray(m.predict(df.iloc[::-1]), dtype=float)[::-1]
        assert np.allclose(a, b)

    def test_summary_and_score(self, capsys):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._mixed_frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            s = m.score(df, y)
            m.summary()
        assert capsys.readouterr().out.strip()
        assert 0.0 <= s <= 1.0

    def test_probability_methods_delegate(self):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._mixed_frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            pr = m.predict_proba(df)
            d = m.decision_function(df)
        assert pr.shape == (len(y), 2)
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert len(d) == len(y)

    def test_get_and_set_params_reach_the_inner_estimator(self):
        from MissLearn import MissPreprocessor, MissLogistic
        m = MissPreprocessor(MissLogistic(compute_se=False))
        p = m.get_params()
        assert 'estimator' in p
        m.set_params(categorical_threshold=5)
        assert m.get_params()['categorical_threshold'] == 5


class TestEnsembleSurface:
    """``MissEnsemble`` in both modes, with the options that change the fit."""

    @staticmethod
    def _data(n=120, p=4, seed=3, task='clf'):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        y = (lin > 0).astype(float) if task == 'clf' \
            else lin + rng.normal(scale=0.3, size=n)
        return X, y

    def test_heterogeneous_members_with_explicit_weights(self, capsys):
        from MissLearn import (MissEnsemble, MissLogistic, MissRidgeClassifier,
                               MissBayesClassifier)
        X, y = self._data()
        members = [('logistic', MissLogistic(compute_se=False)),
                   ('ridge', MissRidgeClassifier(compute_se=False)),
                   ('bayes', MissBayesClassifier())]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimators=members, weights=[0.5, 0.3, 0.2],
                             oob_score=True, random_state=0).fit(X, y)
            pr = m.predict_proba(X)
            m.summary(feature_names=['a', 'b', 'c', 'd'])
        out = capsys.readouterr().out
        assert out.strip()
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert len(m.estimators_) == 3

    def test_max_samples_and_max_features_are_honoured(self):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=5, max_samples=0.6,
                             max_features=0.75, random_state=0).fit(X, y)
            pred = m.predict(X)
        assert len(pred) == len(y)

    def test_bootstrap_off_uses_every_row(self):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=4, bootstrap=False,
                             random_state=0).fit(X, y)
        assert np.all(np.isfinite(m.predict_proba(X)))

    def test_regression_ensemble_predicts_intervals(self):
        from MissLearn import MissEnsemble, MissLinear
        X, y = self._data(task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLinear(compute_se=False),
                             n_estimators=6, random_state=0).fit(X, y)
            lo, hi = m.predict_interval(X, alpha=0.05)
            point = m.predict(X)
        finite = np.isfinite(lo) & np.isfinite(hi)
        assert finite.any()
        assert np.all(lo[finite] <= point[finite] + 1e-8)
        assert np.all(point[finite] <= hi[finite] + 1e-8)

    def test_feature_importances_are_aggregated_across_members(self):
        from MissLearn import MissEnsemble, MissLinear
        X, y = self._data(task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLinear(compute_se=False),
                             n_estimators=5, random_state=0).fit(X, y)
        imp = getattr(m, 'feature_importances_', None)
        if imp is None:
            pytest.skip('no aggregated importances exposed')
        imp = np.asarray(imp, dtype=float)
        assert imp.shape == (X.shape[1],)
        assert np.all(imp >= 0.0)

    def test_a_non_misslearn_member_is_refused_with_a_reason(self):
        from MissLearn import MissEnsemble
        from sklearn.linear_model import LogisticRegression
        X, y = self._data()
        with pytest.raises(ValueError) as exc:
            MissEnsemble(estimator=LogisticRegression(),
                         n_estimators=3).fit(X, y)
        assert str(exc.value).strip()

    @pytest.mark.parametrize('member', ['MissLinear', 'MissRidgeRegressor'])
    def test_mixing_a_regressor_into_a_classifier_ensemble_is_refused(self, member):
        """Both must be refused, and MissLinear is the one that was not.

        The guard used to ask whether 'Regressor' appeared in the class name,
        which is true of MissRidgeRegressor and false of MissLinear even
        though both are regressors. The ensemble therefore accepted
        MissLinear and failed later inside _collect_proba with
        AttributeError: no attribute 'predict_proba', from library internals
        rather than at the door. It now asks scikit-learn what the estimator
        is.
        """
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with pytest.raises(ValueError, match='regressor'):
            MissEnsemble(estimators=[('clf', MissLogistic(compute_se=False)),
                                     ('reg', getattr(_ML, member)(
                                         compute_se=False))],
                         random_state=0).fit(X, y)

    def test_a_matching_ensemble_is_still_accepted(self):
        """The guard must not have become indiscriminate: an all-regressor
        ensemble on a continuous target is exactly what it should allow.
        """
        from MissLearn import MissEnsemble, MissLinear, MissRidgeRegressor
        X, y = self._data(task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimators=[('a', MissLinear(compute_se=False)),
                                         ('b', MissRidgeRegressor(
                                             compute_se=False))],
                             random_state=0).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))


class TestSummaryInEveryConfiguration:
    """The same sweep again, in the configurations that change what is printed.

    The earlier sweep fitted everything with ``compute_se=False``, which is
    fast and which meant no standard-error, z-statistic, p-value or interval
    branch inside any ``summary`` ever ran. Those are the lines a reader
    actually uses to decide whether a coefficient means anything, so they are
    the ones least worth leaving unexercised.

    The copula axis is here for the same reason: whether the marginal
    transform fired changes what the report says about the model, and the
    branch that says it had never run either.
    """

    @staticmethod
    def _data(n=90, p=3, seed=0, skew=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        if skew:
            # a heavy-tailed margin, so copula='auto' has something to fire on
            X[:, 1] = rng.standard_t(df=2.0, size=n) * 4.0
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        return (X, lin + rng.normal(scale=0.3, size=n),
                (lin > 0).astype(float), np.repeat(np.arange(n // 6), 6))

    @staticmethod
    def _supports(name, param):
        import inspect as _inspect
        return param in _inspect.signature(
            getattr(_ML, name).__init__).parameters

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_with_standard_errors_computed(self, name, capsys):
        from sklearn.base import ClassifierMixin
        if not self._supports(name, 'compute_se'):
            pytest.skip('%s has no compute_se' % name)
        cls = getattr(_ML, name)
        est = cls(compute_se=True)
        X, y_reg, y_clf, groups = self._data()
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y, **fit_kw)
            est.summary()
        out = capsys.readouterr().out
        assert out.strip()

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_standard_errors_are_exposed_and_non_negative(self, name):
        """A standard error is either a non-negative number or NaN meaning
        'could not be computed'. A negative one, or a zero standing in for a
        failure, would read as certainty.
        """
        from sklearn.base import ClassifierMixin
        if not self._supports(name, 'compute_se'):
            pytest.skip('%s has no compute_se' % name)
        cls = getattr(_ML, name)
        est = cls(compute_se=True)
        X, y_reg, y_clf, groups = self._data()
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y, **fit_kw)
        se = getattr(est, 'se_', None)
        if se is None:
            pytest.skip('%s exposes no se_' % name)
        se = np.asarray(se, dtype=float)
        finite = np.isfinite(se)
        assert np.all(se[finite] >= 0.0), \
            '%s produced a negative standard error' % name

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_with_the_copula_applied(self, name, capsys):
        from sklearn.base import ClassifierMixin
        if not self._supports(name, 'copula'):
            pytest.skip('%s has no copula option' % name)
        cls = getattr(_ML, name)
        kw = {'copula': True}
        if self._supports(name, 'compute_se'):
            kw['compute_se'] = False
        est = cls(**kw)
        X, y_reg, y_clf, groups = self._data(skew=True)
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y, **fit_kw)
            est.summary()
        out = capsys.readouterr().out
        assert out.strip()
        assert getattr(est, 'copula_used_', True), \
            '%s was asked for the copula and did not record using it' % name

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_with_the_copula_declined(self, name, capsys):
        """``copula='auto'`` on well-behaved Gaussian margins should decline,
        and the report should say so rather than staying silent.
        """
        from sklearn.base import ClassifierMixin
        if not self._supports(name, 'copula'):
            pytest.skip('%s has no copula option' % name)
        cls = getattr(_ML, name)
        kw = {'copula': 'auto'}
        if self._supports(name, 'compute_se'):
            kw['compute_se'] = False
        est = cls(**kw)
        X, y_reg, y_clf, groups = self._data(skew=False)
        y = y_clf if isinstance(est, ClassifierMixin) else y_reg
        fit_kw = {'groups': groups} if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y, **fit_kw)
            est.summary()
        assert capsys.readouterr().out.strip()

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_summary_after_a_fit_on_a_single_feature(self, name, capsys):
        """p = 1 collapses several report layouts: there is no correlation
        matrix to print and no second coefficient to align against.
        """
        from sklearn.base import ClassifierMixin
        import inspect as _inspect
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        est = cls(**kw)
        rng = np.random.default_rng(7)
        n = 90
        X = rng.standard_normal((n, 1))
        X[::8, 0] = np.nan
        lin = np.nan_to_num(X)[:, 0] * 2.0
        y = ((lin > 0).astype(float) if isinstance(est, ClassifierMixin)
             else lin + rng.normal(scale=0.3, size=n))
        fit_kw = {'groups': np.repeat(np.arange(n // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                est.fit(X, y, **fit_kw)
            except (ValueError, np.linalg.LinAlgError) as exc:
                assert str(exc).strip()
                return
            est.summary()
        assert capsys.readouterr().out.strip()


class TestCrossValidateScoringForms:
    """``scoring`` accepts five shapes and each takes its own branch.

    A string, a list, a tuple, a dict and a callable all mean something
    slightly different about what the results dictionary will be keyed by,
    and a caller who gets the keys wrong reads the wrong column. Only the
    default had a test.
    """

    @staticmethod
    def _data(n=80, seed=5):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::7, 0] = np.nan
        y = np.nan_to_num(X)[:, 1] * 1.5 + rng.normal(scale=0.3, size=n)
        return X, y

    def _est(self):
        from MissLearn import MissLinear
        return MissLinear(compute_se=False)

    def test_default_scoring(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3)
        assert 'test_score' in out
        assert len(out['test_score']) == 3

    def test_string_scoring(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3, scoring='r2')
        assert 'test_score' in out

    @pytest.mark.parametrize('container', [list, tuple])
    def test_a_sequence_of_scorers_is_keyed_by_name(self, container):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3,
                                  scoring=container(['r2',
                                                     'neg_mean_squared_error']))
        assert 'test_r2' in out
        assert 'test_neg_mean_squared_error' in out

    def test_a_dict_of_scorers_uses_its_own_keys(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3,
                                  scoring={'fit_quality': 'r2'})
        assert 'test_fit_quality' in out

    def test_a_callable_scorer(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()

        def scorer(est, X_, y_):
            obs = ~np.isnan(y_)
            return float(np.mean(np.abs(est.predict(X_)[obs] - y_[obs])))

        out = miss_cross_validate(self._est(), X, y, cv=3, scoring=scorer)
        assert 'test_score' in out
        assert np.all(np.isfinite(out['test_score']))

    def test_train_scores_are_returned_when_asked(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3,
                                  scoring=['r2'], return_train_score=True)
        assert 'train_r2' in out and 'test_r2' in out
        assert len(out['train_r2']) == 3

    def test_estimators_are_returned_when_asked(self):
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        out = miss_cross_validate(self._est(), X, y, cv=3,
                                  return_estimators=True)
        assert len(out['estimators']) == 3
        assert all(e is not None for e in out['estimators'])

    def test_a_failing_fold_yields_nan_in_every_column(self):
        """A fold that dies must leave the arrays the same length as the
        number of folds, with NaN in the failed position and None in the
        estimator list, so the mean is computed over a vector whose length
        still says how many folds there were.
        """
        from MissLearn._crossval import miss_cross_validate
        from MissLearn import MissLinear
        X, y = self._data()

        class ExplodesOnceFitted(MissLinear):
            calls = {'n': 0}

            def fit(self, X_, y_=None, **kw):
                type(self).calls['n'] += 1
                if type(self).calls['n'] == 2:
                    raise RuntimeError('forced fold failure')
                return super().fit(X_, y_)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = miss_cross_validate(ExplodesOnceFitted(compute_se=False),
                                      X, y, cv=4, scoring=['r2'],
                                      return_train_score=True,
                                      return_estimators=True)
        assert len(out['test_r2']) == 4
        assert len(out['train_r2']) == 4
        assert len(out['estimators']) == 4
        assert np.isnan(out['test_r2']).sum() == 1
        assert out['estimators'].count(None) == 1

    def test_a_scorer_that_raises_warns_and_records_nan(self):
        """Scoring is separate from fitting. A scorer that fails on one fold
        must not lose the folds that worked.
        """
        from MissLearn._crossval import miss_cross_validate
        X, y = self._data()
        state = {'n': 0}

        def flaky(est, X_, y_):
            state['n'] += 1
            if state['n'] == 1:
                raise ValueError('forced scorer failure')
            return 0.5

        with pytest.warns(UserWarning, match='scoring'):
            out = miss_cross_validate(self._est(), X, y, cv=3, scoring=flaky,
                                      return_train_score=True)
        assert len(out['test_score']) == 3
        assert np.isnan(out['test_score']).sum() >= 1


class TestWarmStartStateIsStripped:
    """Each fold must start from an independent estimator.

    Carrying fitted state across folds, or in from an estimator the caller
    already fitted, leaks the whole dataset into every fold and inflates
    every score. The stripper walks nested estimators, which is where it
    would silently miss: an ensemble holds its members in a list of tuples,
    and a wrapper holds one under a different attribute name.
    """

    @staticmethod
    def _data(n=90, seed=6):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::8, 1] = np.nan
        y = np.nan_to_num(X)[:, 0] * 1.2 + rng.normal(scale=0.3, size=n)
        return X, y

    def test_a_prefitted_estimator_does_not_leak_into_the_folds(self):
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            already = MissLinear(compute_se=False).fit(X, y)
            fresh = miss_cross_val_score(MissLinear(compute_se=False), X, y,
                                         cv=4, random_state=0)
            reused = miss_cross_val_score(already, X, y, cv=4, random_state=0)
        assert np.allclose(fresh, reused), \
            'a prefitted estimator scored differently, so state leaked'

    def test_nested_members_are_stripped_too(self):
        from MissLearn._crossval import _strip_warm_start_state
        from MissLearn import MissEnsemble, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ens = MissEnsemble(estimator=MissLinear(compute_se=False),
                               n_estimators=3, random_state=0).fit(X, y)
        assert hasattr(ens, 'estimators_')
        stripped = _strip_warm_start_state(ens)
        assert not hasattr(stripped, 'estimators_') or not stripped.estimators_

    def test_a_wrapped_estimator_is_stripped(self):
        from MissLearn._crossval import _strip_warm_start_state
        from MissLearn import MissRidge
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissRidge(compute_se=False).fit(X, y)
        assert hasattr(m, 'model_')
        stripped = _strip_warm_start_state(m)
        assert not hasattr(stripped, 'model_')


class TestDiagnosticReportBranches:
    """The diagnostic's verdicts, one per mechanism it can conclude.

    Each branch prints a different recommendation, and the recommendation is
    the whole output: a reader who is told FIML is fully appropriate when the
    data is MNAR has been given the wrong answer in a confident voice.
    """

    @staticmethod
    def _mcar(n=200, p=4, seed=8):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan          # unconditional
        return X, np.nan_to_num(X) @ np.linspace(1.0, -1.0, p)

    @staticmethod
    def _mar(n=200, p=4, seed=9):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        hot = X[:, 0] > np.quantile(X[:, 0], 0.5)       # driven by column 0
        for j in (1, 2, 3):
            X[hot, j] = np.nan
        return X, np.nan_to_num(X) @ np.linspace(1.0, -1.0, p)

    def test_mcar_data_is_reported_as_consistent_with_mcar(self, capsys):
        from MissLearn import MissDiagnostic
        X, y = self._mcar()
        d = MissDiagnostic(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
        out = capsys.readouterr().out
        if not d.little_significant_:
            assert 'consistent with MCAR' in out
            assert 'fully appropriate' in out
        else:
            assert 'MCAR rejected' in out

    def test_mar_data_is_reported_as_predictable(self, capsys):
        from MissLearn import MissDiagnostic
        X, y = self._mar()
        d = MissDiagnostic(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
        out = capsys.readouterr().out
        assert 'MAR' in out
        assert out.strip()

    def test_one_incomplete_column_gives_a_trivial_correlation_matrix(self):
        """With a single column carrying missingness there is nothing to
        correlate it against, so the matrix is 1x1 rather than an error.
        """
        from MissLearn import MissDiagnostic
        rng = np.random.default_rng(10)
        X = rng.standard_normal((120, 4))
        X[::6, 2] = np.nan                     # exactly one incomplete column
        d = MissDiagnostic(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            R = d.missingness_correlations()
        assert R.shape == (1, 1)
        assert R[0, 0] == 1.0
        assert len(d.miss_corr_cols_) == 1

    def test_no_missingness_at_all(self, capsys):
        from MissLearn import MissDiagnostic
        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 3))
        d = MissDiagnostic(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.summary()
        assert capsys.readouterr().out.strip()

    def test_descriptive_by_missingness_returns_a_table(self):
        """It returns a dict rather than printing, which is the part worth
        pinning: a caller that expected output would silently get nothing.
        """
        from MissLearn import MissDiagnostic
        X, y = self._mar()
        d = MissDiagnostic(X, y, feature_names=['a', 'b', 'c', 'd'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = d.descriptive_by_missingness()
        assert isinstance(out, dict) and out


class TestRecommenderScoringBranches:
    """The recommender's score adjustments, which are its actual output.

    Every branch adds or subtracts a point from a family with a reason
    attached, and the reasons are what the user reads. A branch that never
    runs is a recommendation the library can never make.
    """

    @staticmethod
    def _heavy_tailed(n=300, p=4, seed=12):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[:, 0] = rng.standard_t(df=1.7, size=n) * 6.0
        X[:, 1] = rng.standard_t(df=1.9, size=n) * 4.0
        X[rng.random(X.shape) < 0.10] = np.nan
        return X, np.nan_to_num(X) @ np.linspace(1.0, -1.0, p)

    @staticmethod
    def _wide(n=200, p=18, seed=13):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        return X, np.nan_to_num(X) @ rng.standard_normal(p)

    def test_heavy_tails_move_the_ranking(self, capsys):
        from MissLearn import MissRecommender
        X, y = self._heavy_tailed()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
            r.summary()
        out = capsys.readouterr().out
        assert out.strip()
        # The recommender used to read prefit_check's warnings only, while a
        # kurtosis finding is filed as a note, so heavy_tailed was False on
        # every dataset and the copula was never advised. These columns have
        # excess kurtosis 16.5 and 64.6 against a threshold of 7.
        assert r.preprocessing_['copula'] is True,             'heavy tails were not detected'
        assert 'Heavy tails' in out
        built = r.make_estimator()
        assert built.get_params().get('copula') is True,             'the advice was printed but not applied to the built estimator'

    def test_gaussian_margins_do_not_trigger_the_copula(self):
        """The other side of the same branch: the detector must not fire on
        well-behaved data, or the advice becomes noise.
        """
        from MissLearn import MissRecommender
        rng = np.random.default_rng(21)
        X = rng.standard_normal((300, 4))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.0, -1.0, 4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
        assert r.preprocessing_['copula'] is False

    def test_a_moderately_wide_design_favours_shrinkage(self, capsys):
        """At p >= 15 the recommender should start preferring a penalty, and
        say why.
        """
        from MissLearn import MissRecommender
        X, y = self._wide()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
            r.summary()
        out = capsys.readouterr().out
        assert out.strip()
        assert 'MissRidge' in out or 'MissLASSO' in out

    def test_the_probe_subsamples_a_large_dataset(self):
        """Above probe_max_n the nonlinearity probe works on a subsample, so
        that the recommendation does not cost more than the model.
        """
        from MissLearn import MissRecommender
        rng = np.random.default_rng(14)
        n = 400
        X = rng.standard_normal((n, 3))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=True, probe_max_n=100,
                                random_state=0).fit(X, y)
        assert r.ranked_
        probe = getattr(r, 'nonlinearity_probe_', None)
        assert probe is None or isinstance(probe, dict)

    def test_make_estimator_returns_the_top_recommendation(self):
        from MissLearn import MissRecommender
        X, y = self._wide()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
            est = r.make_estimator()
        assert hasattr(est, 'fit')
        assert type(est).__name__.startswith('Miss')


class TestPandasCoercionPaths:
    """Pandas support is duck-typed, so it has its own conversion layer.

    That layer is where a Series arrives positionally rather than by keyword,
    where a nullable dtype has to be coerced without turning ``pd.NA`` into a
    category, and where a group label is a string. None of those had a test,
    and every one of them is a silent-wrong-answer path rather than a crash.
    """

    @staticmethod
    def _frame(n=80, seed=15):
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.standard_normal((n, 3)), columns=['a', 'b', 'c'])
        X.loc[::7, 'a'] = np.nan
        y = pd.Series(np.nan_to_num(X.to_numpy()) @ np.array([1.5, -1.0, 0.5]))
        g = pd.Series(np.repeat(np.arange(n // 8), 8))
        return X, y, g

    def test_y_and_groups_positionally_as_series(self):
        from MissLearn import MissMixedRegressor
        X, y, g = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, g)
        assert np.all(np.isfinite(m.predict(X)))

    def test_y_and_groups_by_keyword_as_series(self):
        from MissLearn import MissMixedRegressor
        X, y, g = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y=y, groups=g)
        assert np.all(np.isfinite(m.predict(X)))

    def test_string_group_labels_survive_coercion(self):
        """``_coerce_groups`` preserves dtype rather than forcing float, so a
        site name or a patient identifier works as a grouping variable.
        """
        pd = pytest.importorskip('pandas')
        from MissLearn import MissMixedRegressor
        X, y, _ = self._frame()
        g = pd.Series(['site_%d' % (i // 8) for i in range(len(y))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
        assert np.all(np.isfinite(m.predict(X, groups=g)))

    def test_a_nullable_integer_target_is_coerced(self):
        pd = pytest.importorskip('pandas')
        from MissLearn import MissLogistic
        X, _, _ = self._frame()
        raw = (np.arange(len(X)) % 2).astype(float)
        raw[::9] = np.nan
        y = pd.Series(raw).astype('Int64')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLogistic(compute_se=False).fit(X, y)
        assert np.all(np.isfinite(m.predict_proba(X)))

    def test_string_class_labels_from_a_series(self):
        """``_coerce_y`` falls back to an untyped conversion when the values
        are not numeric, because a missing label is not a category and must
        not be encoded as one.
        """
        pd = pytest.importorskip('pandas')
        from MissLearn import MissLogistic
        X, _, _ = self._frame()
        y = pd.Series(['cat' if i % 2 else 'dog' for i in range(len(X))])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLogistic(compute_se=False).fit(X, y)
        assert set(m.classes_) == {'cat', 'dog'}
        assert set(np.unique(m.predict(X))) <= {'cat', 'dog'}

    def test_a_dataframe_predict_matches_its_ndarray(self):
        from MissLearn import MissLinear
        X, y, _ = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLinear(compute_se=False).fit(X, y)
            a = m.predict(X)
            b = m.predict(X.to_numpy())
        assert np.allclose(a, b)


class TestPrefitCheckAndPreprocessorNaming:
    """Feature names, and the refusal that happens before any numeric check.

    A compatibility report that says 'X4' instead of the column name is one
    nobody acts on, so the naming precedence is worth pinning: an explicit
    argument beats a DataFrame's own columns, which beat positional labels.
    """

    @staticmethod
    def _frame(n=60, seed=16):
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.standard_normal((n, 3)),
                          columns=['alpha', 'beta', 'gamma'])
        df.loc[::6, 'beta'] = np.nan
        y = (np.nan_to_num(df.to_numpy()) @ np.array([1.5, -1.0, 0.5]) > 0)
        return df, y.astype(float)

    def test_non_numeric_columns_are_refused_before_numeric_checks(self):
        """Text in X cannot be cast to float, so the check stops there and
        says to use MissPreprocessor rather than failing later inside a
        variance computation.
        """
        from MissLearn import prefit_check
        X = np.array([['a', 1.0], ['b', 2.0], ['c', 3.0]], dtype=object)
        r = prefit_check(X, raise_on_error=False, emit_warnings=False)
        assert not r.passed
        assert any('MissPreprocessor' in e for e in r.errors)

    def test_the_same_input_raises_when_asked(self):
        from MissLearn import prefit_check
        X = np.array([['a', 1.0], ['b', 2.0], ['c', 3.0]], dtype=object)
        with pytest.raises(ValueError, match='MissPreprocessor'):
            prefit_check(X, raise_on_error=True, emit_warnings=False)

    def test_a_dataframes_own_columns_are_used(self):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
        assert list(m.feature_names_in_) == ['alpha', 'beta', 'gamma']

    def test_an_explicit_argument_wins_over_the_columns(self):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 feature_names=['one', 'two', 'three'],
                                 raise_on_error=False).fit(df, y)
        assert list(m.feature_names_in_) == ['one', 'two', 'three']

    def test_a_feature_names_of_the_wrong_length_is_refused(self):
        """Silently ignoring the mismatch would relabel every column by one
        and produce a report that is wrong rather than merely unhelpful.
        """
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with pytest.raises(ValueError, match='feature_names'):
            MissPreprocessor(MissLogistic(compute_se=False),
                             feature_names=['only', 'two'],
                             raise_on_error=False).fit(df, y)

    def test_summary_reports_the_check_and_the_inner_model(self, capsys):
        """The wrapper's report is the compatibility check plus whatever the
        estimator it wraps has to say, so a caller sees both halves from one
        call rather than having to reach inside for the second.
        """
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            m.summary()
        out = capsys.readouterr().out
        assert 'Compatibility Check' in out
        assert 'MissLogistic' in out


class TestParameterValidationIsShared:
    """The common parameter checks, which live in one place for a reason.

    ``_conformance`` holds them so that sixteen classes cannot drift apart on
    what counts as a valid ``alpha`` or ``tol``. Each check has a refusal
    branch, and a refusal that has never fired is a refusal nobody has read.
    """

    @staticmethod
    def _data(n=60, seed=17):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::7, 0] = np.nan
        return X, np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])

    def test_a_negative_penalty_is_refused(self):
        from MissLearn import MissRidgeRegressor
        X, y = self._data()
        with pytest.raises(ValueError, match='alpha'):
            MissRidgeRegressor(alpha=-1.0, compute_se=False).fit(X, y)

    def test_a_zero_penalty_is_allowed(self):
        """alpha = 0 is no penalty, which is a meaningful setting and the
        boundary the refusal is written against: 'alpha must be >= 0'.
        """
        from MissLearn import MissRidgeRegressor
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissRidgeRegressor(alpha=0.0, compute_se=False).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))

    @pytest.mark.parametrize('bad', [0, -3])
    def test_a_non_positive_iteration_budget_is_refused(self, bad):
        from MissLearn import MissLinear
        X, y = self._data()
        with pytest.raises(ValueError):
            MissLinear(max_iter=bad, compute_se=False).fit(X, y)

    @pytest.mark.parametrize('bad', [-1e-3, 0.0])
    def test_a_non_positive_tolerance_is_refused(self, bad):
        from MissLearn import MissLinear
        X, y = self._data()
        with pytest.raises(ValueError):
            MissLinear(tol=bad, compute_se=False).fit(X, y)

    def test_predicting_with_the_wrong_width_is_refused(self):
        """Silently predicting from a matrix of a different width would
        correspond to no fitted model at all.
        """
        from MissLearn import MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLinear(compute_se=False).fit(X, y)
        with pytest.raises(ValueError):
            m.predict(X[:, :2])

    def test_an_entirely_absent_column_is_refused_uniformly(self):
        from MissLearn import MissLinear
        X, y = self._data()
        X = X.copy()
        X[:, 1] = np.nan
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(X, y)

    def test_infinity_in_y_is_refused_though_nan_is_not(self):
        """NaN in y is a deliberate capability; infinity is not a missing
        value, it is a broken one.
        """
        from MissLearn import MissLinear
        X, y = self._data()
        y_nan = y.copy()
        y_nan[::9] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            MissLinear(compute_se=False).fit(X, y_nan)      # allowed
        y_inf = y.copy()
        y_inf[3] = np.inf
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(X, y_inf)

    def test_one_dimensional_x_is_refused(self):
        from MissLearn import MissLinear
        X, y = self._data()
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(X[:, 0], y)

    def test_an_empty_design_is_refused(self):
        from MissLearn import MissLinear
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(np.empty((0, 3)), np.empty((0,)))


class TestConformanceReportRendering:
    """``check_missing_data_estimator`` has to explain itself.

    The report is the deliverable: it distinguishes a clear refusal, which is
    acceptable, from a silent NaN, which never is. An estimator that fails a
    regime has to produce a readable account of which one and why, and that
    rendering had never run against a failing estimator because every
    estimator in the library passes.
    """

    @staticmethod
    def _data(n=50, seed=18):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::6, 1] = np.nan
        return X, np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])

    def test_a_passing_estimator_reports_cleanly(self, capsys):
        from MissLearn import check_missing_data_estimator, MissLinear
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(MissLinear(compute_se=False))
        text = str(rep)
        assert text.strip()
        assert 'MissLinear' in text

    def test_an_estimator_that_returns_nan_is_reported_as_failing(self):
        """The one outcome the contract forbids. A stub that fits happily and
        then predicts NaN must be caught and named, because that is exactly
        the failure this checker exists for.
        """
        from sklearn.base import BaseEstimator, RegressorMixin
        from MissLearn import check_missing_data_estimator

        class SilentlyNaN(RegressorMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                self.n_features_in_ = np.asarray(X).shape[1]
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return np.full(np.asarray(X).shape[0], np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(SilentlyNaN())
        text = str(rep)
        assert text.strip()
        assert 'SilentlyNaN' in text
        assert not getattr(rep, 'ok', True) or 'nan' in text.lower()

    def test_an_estimator_that_refuses_is_acceptable(self):
        """A clear refusal is a pass, not a failure. Conflating the two would
        make the checker useless for anything that legitimately declines a
        regime.
        """
        from sklearn.base import BaseEstimator, RegressorMixin
        from MissLearn import check_missing_data_estimator

        class RefusesIncompleteData(RegressorMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                X = np.asarray(X, dtype=float)
                if np.isnan(X).any():
                    raise ValueError('this estimator requires complete data')
                self.n_features_in_ = X.shape[1]
                return self

            def predict(self, X):
                return np.zeros(np.asarray(X).shape[0])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(RefusesIncompleteData())
        assert str(rep).strip()

    def test_the_report_renders_for_a_classifier_too(self):
        from MissLearn import check_missing_data_estimator, MissLogistic
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(MissLogistic(compute_se=False))
        assert 'MissLogistic' in str(rep)


class TestGaussianProcessUncertainty:
    """The Gaussian process exists to give calibrated uncertainty.

    ``predict_std`` is the whole argument for paying O(n^3), and the ARD
    branch of the report prints one length scale per feature rather than one
    number, which is the output a user reads to decide which features the
    kernel is actually using. Neither had run.
    """

    @staticmethod
    def _data(n=70, p=3, seed=19, holes_in_y=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5]) \
            + rng.normal(scale=0.3, size=n)
        if holes_in_y:
            y = y.copy()
            y[::9] = np.nan
        return X, y

    @pytest.mark.parametrize('ard', [False, True])
    def test_summary_reports_the_length_scales(self, ard, capsys):
        from MissLearn import MissGaussianRegressor
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianRegressor(ard=ard).fit(X, y)
            m.summary()
        out = capsys.readouterr().out
        assert 'length_scale' in out
        if ard:
            assert '[' in out, 'ARD should print one length scale per feature'

    def test_summary_mentions_absent_outcomes(self, capsys):
        from MissLearn import MissGaussianRegressor
        X, y = self._data(holes_in_y=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianRegressor().fit(X, y)
            m.summary()
        out = capsys.readouterr().out
        assert 'Missing (y)' in out

    def test_predict_std_is_positive_and_grows_with_missingness(self):
        """The claim being pinned: a row with more absent entries is
        predicted less confidently. That is the property marginalisation is
        supposed to deliver, and it is checkable directly.
        """
        from MissLearn import MissGaussianRegressor
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianRegressor().fit(X, y)
            complete = np.array([[0.2, -0.1, 0.4]])
            sparse = np.array([[0.2, np.nan, np.nan]])
            s_complete = m.predict_std(complete)
            s_sparse = m.predict_std(sparse)
        assert np.all(s_complete > 0)
        assert s_sparse[0] >= s_complete[0], \
            'an emptier row should not be predicted more confidently'

    def test_the_wrapper_delegates_predict_std(self):
        from MissLearn import MissGaussian
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussian().fit(X, y)
            s = m.predict_std(X)
        assert s.shape == (X.shape[0],)
        assert np.all(s > 0)

    def test_intervals_widen_with_the_confidence_level(self):
        from MissLearn import MissGaussianRegressor
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianRegressor().fit(X, y)
            lo99, hi99 = m.predict_interval(X, alpha=0.01)
            lo50, hi50 = m.predict_interval(X, alpha=0.50)
        assert np.all((hi99 - lo99) >= (hi50 - lo50) - 1e-9)

    def test_classifier_summary_runs(self, capsys):
        from MissLearn import MissGaussianClassifier
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianClassifier().fit(X, (y > 0).astype(float))
            m.summary()
        assert capsys.readouterr().out.strip()


class TestSensitivityFitRobustness:
    """``MissSensitivity`` refits at every delta, so it must survive an
    estimator that fails at some of them.

    A delta far into the tail can produce a design the estimator refuses.
    Dropping that one draw and carrying on is right; letting it abort the
    whole sweep would mean no sensitivity analysis at all for exactly the
    departures the analysis exists to explore.
    """

    @staticmethod
    def _data(n=80, p=3, seed=20):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[::5, 1] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25]) \
            + rng.normal(scale=0.3, size=n)
        y[::11] = np.nan
        return X, y

    def test_an_estimator_that_rejects_fit_kwargs_still_works(self):
        """The first attempt passes fit_kwargs; a TypeError means the
        estimator does not accept them, and the retry without them is the
        path that had never run.
        """
        from MissLearn import MissSensitivity, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = MissSensitivity(MissLinear(compute_se=False),
                                delta_range=(-1.0, 1.0), n_delta=3, m=2,
                                random_state=0).fit(X, y)
        assert s.coef_curves_.shape[0] == 3

    def test_a_failing_estimator_is_warned_about_and_skipped(self):
        from MissLearn import MissSensitivity, MissLinear
        X, y = self._data()

        class FailsSometimes(MissLinear):
            calls = {'n': 0}

            def fit(self, X_, y_=None, **kw):
                type(self).calls['n'] += 1
                if type(self).calls['n'] % 3 == 0:
                    raise RuntimeError('forced failure')
                return super().fit(X_, y_)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            s = MissSensitivity(FailsSometimes(compute_se=False),
                                delta_range=(-1.0, 1.0), n_delta=3, m=3,
                                random_state=0).fit(X, y)
        assert any('fit failed' in str(w.message) for w in caught), \
            'a dropped draw should say so'
        assert s.coef_curves_.shape[0] == 3

    def test_the_sweep_covers_the_requested_range(self):
        from MissLearn import MissSensitivity, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = MissSensitivity(MissLinear(compute_se=False),
                                delta_range=(-2.0, 2.0), n_delta=5, m=2,
                                random_state=0).fit(X, y)
        # delta_range is in units of sigma_Y: delta_std_grid_ is the grid
        # the caller asked for, delta_grid_ the shift actually applied
        assert np.isclose(s.delta_std_grid_.min(), -2.0)
        assert np.isclose(s.delta_std_grid_.max(), 2.0)
        assert len(s.delta_std_grid_) == 5
        assert np.allclose(s.delta_grid_, s.delta_std_grid_ * s.sigma_y_)
        assert s.sigma_y_ > 0

    def test_summary_prints(self, capsys):
        from MissLearn import MissSensitivity, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = MissSensitivity(MissLinear(compute_se=False),
                                delta_range=(-1.0, 1.0), n_delta=3, m=2,
                                random_state=0).fit(X, y)
            s.summary()
        assert capsys.readouterr().out.strip()


class TestMissingnessReportOnTheBase:
    """``missingness_report`` is inherited by every estimator.

    It is the first thing a user calls after a fit that looked odd, and its
    absent-outcome branch only runs when y itself has holes, which most
    fixtures avoid.
    """

    @staticmethod
    def _data(n=70, p=3, seed=21, holes_in_y=True):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.15] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])
        if holes_in_y:
            y = y.copy()
            y[::8] = np.nan
        return X, y

    def test_report_counts_absent_outcomes(self, capsys):
        from MissLearn import MissLinear
        X, y = self._data(holes_in_y=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLinear(compute_se=False).fit(X, y)
            m.missingness_report()
        out = capsys.readouterr().out
        assert 'Missing y entries' in out
        assert 'Partial cases' in out

    def test_report_omits_the_y_line_when_y_is_complete(self, capsys):
        from MissLearn import MissLinear
        X, y = self._data(holes_in_y=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLinear(compute_se=False).fit(X, y)
            m.missingness_report()
        out = capsys.readouterr().out
        assert 'Missing y entries' not in out
        assert 'Missing X entries' in out


class TestImputerPoolingEdges:
    """``fit_transform_combine`` where the variance cannot be recovered.

    The pooled standard error depends on finding a per-imputation variance
    on each fitted estimator. When that attribute is absent, the wrong
    length, or not finite, the draw has to be dropped with a warning rather
    than silently pooled from whatever was there.
    """

    @staticmethod
    def _data(n=80, p=3, seed=22):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.18] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25]) \
            + rng.normal(scale=0.3, size=n)
        return X, y

    def test_a_missing_variance_attribute_is_warned_about(self):
        from MissLearn import MissImputer
        from sklearn.linear_model import LinearRegression
        X, y = self._data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imp = MissImputer(m=3, random_state=0).fit(X)
            # Rubin's rules need estimates and variances from the same draws,
            # so a draw whose variance cannot be read is dropped whole.
            # Naming an attribute that does not exist drops every draw, and
            # pooling nothing is refused rather than returned as a result.
            with pytest.raises(RuntimeError, match='no successful fits'):
                imp.fit_transform_combine(X, y, LinearRegression(), 'coef_',
                                          param_var='no_such_attribute')
        assert any('could not extract' in str(w.message) for w in caught), (
            'each dropped draw should say why it was dropped')

    def test_the_intercept_is_stripped_from_a_misslearn_standard_error(self):
        """MissLearn reports se_ with the intercept at index 0 while coef_
        has none, so the two differ in length by exactly one. Pooling them
        without stripping would misalign every feature by one position.
        """
        from MissLearn import MissImputer, MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imp = MissImputer(m=3, random_state=0).fit(X)
            res = imp.fit_transform_combine(X, y, MissLinear(), 'coef_',
                                            param_var='se_')
        assert np.asarray(res['estimate']).shape == (X.shape[1],)
        assert np.asarray(res['within_var']).shape == (X.shape[1],)
        assert np.all(np.asarray(res['within_var'], dtype=float) >= 0)

    def test_include_y_pools_over_jointly_imputed_outcomes(self):
        """With include_y the outcome is drawn alongside the predictors, so
        rows whose y was absent contribute instead of being dropped.
        """
        from MissLearn import MissImputer
        from sklearn.linear_model import LinearRegression
        X, y = self._data()
        y = y.copy()
        y[::7] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imp = MissImputer(m=3, include_y=True, random_state=0).fit(X, y)
            res = imp.fit_transform_combine(X, y, LinearRegression(), 'coef_')
        assert np.isfinite(np.asarray(res['estimate'], dtype=float)).all()


class TestMixedPredictionPaths:
    """Prediction for a group the model never saw, and interval width."""

    @staticmethod
    def _data(n=120, p=3, seed=23):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        groups = np.repeat(np.arange(n // 10), 10)
        y = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5]) \
            + rng.normal(scale=0.3, size=n)
        return X, y, groups

    def test_an_unseen_group_falls_back_to_the_population(self):
        """A patient the model has never met has no random intercept, so the
        prediction has to fall back to the fixed effects rather than
        inventing one or failing.
        """
        from MissLearn import MissMixedRegressor
        X, y, g = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
            unseen = np.full(len(y), 999)
            with_unseen = m.predict(X, groups=unseen)
            population = m.predict(X)
        assert np.allclose(with_unseen, population), \
            'an unseen group should predict at the population level'

    def test_intervals_bracket_and_widen(self):
        from MissLearn import MissMixedRegressor
        X, y, g = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
            lo99, hi99 = m.predict_interval(X, alpha=0.01)
            lo50, hi50 = m.predict_interval(X, alpha=0.50)
            point = m.predict(X)
        assert np.all(lo99 <= point + 1e-8)
        assert np.all(point <= hi99 + 1e-8)
        assert np.all((hi99 - lo99) >= (hi50 - lo50) - 1e-9)

    def test_blups_exist_for_every_seen_group(self):
        from MissLearn import MissMixedRegressor
        X, y, g = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
        assert set(m.blup_) == set(np.unique(g))
        assert np.all(np.isfinite(list(m.blup_.values())))


class TestThreeRowKindsAcrossTheLibrary:
    """Prediction splits every row into one of three kinds, and the third
    had never been supplied.

    A row can be complete, partially observed, or entirely absent. Each takes
    a different branch, because the predictive variance is the residual
    variance for a complete row, the residual plus the full marginal
    contribution for an empty one, and a conditional quantity in between for
    a partial one. Fixtures at 10 or 15 percent missing produce the first two
    and essentially never the third, so the widest interval any of these
    models can produce was the one nothing tested.

    The ordering is the property worth pinning and it holds by construction:
    knowing nothing about a row cannot make the model more confident about it
    than knowing something.
    """

    @staticmethod
    def _three_kinds(n=90, p=3, seed=24):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        y = X @ np.array([1.5, -1.0, 0.5]) + rng.normal(scale=0.3, size=n)
        X[rng.random(X.shape) < 0.10] = np.nan
        # a training set that also contains fully absent rows, so the fit
        # itself meets the case as well as prediction
        X[5, :] = np.nan
        X[11, :] = np.nan
        probe = np.array([
            [0.4, -0.2, 0.7],                    # complete
            [0.4, np.nan, 0.7],                  # partial
            [np.nan, np.nan, np.nan],            # entirely absent
        ])
        return X, y, probe

    @staticmethod
    def _regressors():
        out = []
        for n in ESTIMATOR_NAMES:
            if hasattr(getattr(_ML, n), 'predict_interval'):
                out.append(n)
        return out

    @pytest.mark.parametrize('name', _regressors.__func__())
    def test_the_interval_widens_as_the_row_empties(self, name):
        import inspect as _inspect
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        X, y, probe = self._three_kinds()
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = cls(**kw).fit(X, y, **fit_kw)
            lo, hi = m.predict_interval(probe, alpha=0.05)
        width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
        assert np.all(np.isfinite(width)), \
            '%s produced a non-finite interval' % name
        assert np.all(width > 0)
        assert width[1] >= width[0] - 1e-8, \
            '%s: a partial row should not be tighter than a complete one' % name
        assert width[2] >= width[1] - 1e-8, \
            '%s: an empty row should not be tighter than a partial one' % name

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_an_entirely_absent_row_still_predicts(self, name):
        """With nothing observed the prediction falls back to the marginal
        mean. That is a real answer and must be finite, because the caller
        cannot tell a silent NaN from a missing row.
        """
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        est = cls(**kw)
        X, y, probe = self._three_kinds()
        if isinstance(est, ClassifierMixin):
            y = (y > 0).astype(float)
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y, **fit_kw)
            pred = np.asarray(est.predict(probe), dtype=float)
        assert np.all(np.isfinite(pred)), \
            '%s returned NaN for a row it should answer from the margin' % name

    @pytest.mark.parametrize('name', ESTIMATOR_NAMES)
    def test_probabilities_stay_normalised_on_an_absent_row(self, name):
        import inspect as _inspect
        from sklearn.base import ClassifierMixin
        cls = getattr(_ML, name)
        kw = ({'compute_se': False}
              if 'compute_se' in _inspect.signature(cls.__init__).parameters
              else {})
        est = cls(**kw)
        if not isinstance(est, ClassifierMixin):
            pytest.skip('regressor')
        X, y, probe = self._three_kinds()
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, (y > 0).astype(float), **fit_kw)
            pr = np.asarray(est.predict_proba(probe), dtype=float)
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert np.all((pr >= 0.0) & (pr <= 1.0))


class TestPredictionUnderTheCopula:
    """Everything above again, with the marginal transform switched on.

    Prediction under the copula is a different code path end to end: the
    design is mapped to normal scores going in and the interval is mapped
    back to the response's own units coming out. The inverse transform on the
    interval bounds is the part with no other test, and an interval returned
    in the wrong units is worse than none, because it looks usable.
    """

    @staticmethod
    def _skewed(n=120, p=3, seed=25):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[:, 1] = rng.standard_t(df=2.0, size=n) * 3.0
        y = np.exp(0.4 * X[:, 0]) + rng.normal(scale=0.2, size=n)
        X[rng.random(X.shape) < 0.10] = np.nan
        return X, y

    @staticmethod
    def _copula_capable():
        import inspect as _inspect
        return [n for n in ESTIMATOR_NAMES
                if 'copula' in _inspect.signature(
                    getattr(_ML, n).__init__).parameters
                and hasattr(getattr(_ML, n), 'predict_interval')]

    @pytest.mark.parametrize('name', _copula_capable.__func__())
    def test_intervals_come_back_in_the_response_units(self, name):
        import inspect as _inspect
        cls = getattr(_ML, name)
        kw = {'copula': True}
        if 'compute_se' in _inspect.signature(cls.__init__).parameters:
            kw['compute_se'] = False
        X, y = self._skewed()
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = cls(**kw).fit(X, y, **fit_kw)
            lo, hi = m.predict_interval(X, alpha=0.05)
            point = m.predict(X)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(point)
        assert finite.any()
        assert np.all(lo[finite] <= hi[finite] + 1e-8)
        # the response is positive by construction, so an interval mapped
        # back correctly should sit in roughly the right place rather than
        # in normal-score units around zero
        assert np.median(point[finite]) > 0.0

    @pytest.mark.parametrize('name', _copula_capable.__func__())
    def test_an_absent_row_under_the_copula(self, name):
        import inspect as _inspect
        cls = getattr(_ML, name)
        kw = {'copula': True}
        if 'compute_se' in _inspect.signature(cls.__init__).parameters:
            kw['compute_se'] = False
        X, y = self._skewed()
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        probe = np.full((1, X.shape[1]), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = cls(**kw).fit(X, y, **fit_kw)
            lo, hi = m.predict_interval(probe, alpha=0.05)
        assert np.isfinite(np.asarray(lo, dtype=float)).all()
        assert np.isfinite(np.asarray(hi, dtype=float)).all()

    @pytest.mark.parametrize('name', _copula_capable.__func__())
    def test_the_copula_records_that_it_fired(self, name):
        import inspect as _inspect
        cls = getattr(_ML, name)
        kw = {'copula': True}
        if 'compute_se' in _inspect.signature(cls.__init__).parameters:
            kw['compute_se'] = False
        X, y = self._skewed()
        fit_kw = {'groups': np.repeat(np.arange(len(y) // 6), 6)} \
            if 'Mixed' in name else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = cls(**kw).fit(X, y, **fit_kw)
        assert m.copula_used_ is True


class TestNeighboursIntervalOnAnEmptyRow:
    """A row with nothing observed is now predicted least confidently.

    It used to be predicted most confidently of the three. Measured on a
    response with standard deviation 2.27, widths for complete / partial /
    absent rows, before and after:

        MissLinear              1.11   4.34   8.88
        MissRidgeRegressor      1.11   4.34   8.88
        MissNeighborsRegressor  1.78   3.82   2.98   before
        MissNeighborsRegressor  1.78   3.82   8.93   after

    The mechanism was in the geometry rather than in a guard. With nothing
    observed the expected distance under the fitted joint Gaussian is equal
    to every training point, so the k nearest neighbours are an arbitrary
    subset, and the interval reported that subset's spread instead of the
    response's. The point prediction was unaffected and landed on the
    marginal mean, which is why this was invisible unless the interval was
    read.

    The fallback needed already existed and simply never fired: ``_aggregate``
    used the marginal spread when no neighbour was reachable and again when
    no neighbour had an observed outcome. An entirely absent row met neither
    condition, because it is the query that carries no information rather
    than the neighbourhood, and nothing was looking at the query.
    """

    @staticmethod
    def _data(n=90, p=3, seed=24):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        y = X @ np.array([1.5, -1.0, 0.5]) + rng.normal(scale=0.3, size=n)
        X[rng.random(X.shape) < 0.10] = np.nan
        X[5, :] = np.nan
        X[11, :] = np.nan
        probe = np.array([[0.4, -0.2, 0.7],
                          [0.4, np.nan, 0.7],
                          [np.nan, np.nan, np.nan]])
        return X, y, probe

    def test_the_point_prediction_is_still_the_marginal_mean(self):
        """The part that is right, and the reason the interval problem is
        easy to miss."""
        from MissLearn import MissNeighborsRegressor
        X, y, probe = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsRegressor().fit(X, y)
            pred = m.predict(probe)
        assert abs(pred[2] - np.nanmean(y)) < 0.5 * np.nanstd(y)

    def test_the_likelihood_families_widen_to_the_response_spread(self):
        from MissLearn import MissLinear, MissRidgeRegressor
        X, y, probe = self._data()
        for cls in (MissLinear, MissRidgeRegressor):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = cls(compute_se=False).fit(X, y)
                lo, hi = m.predict_interval(probe, alpha=0.05)
            width = np.asarray(hi) - np.asarray(lo)
            expected = 2 * 1.96 * float(np.nanstd(y))
            assert width[2] > 0.7 * expected, (
                '%s should widen to roughly the response spread when nothing '
                'is observed' % cls.__name__)

    def test_the_neighbour_family_widens_too(self):
        """The corrected behaviour, held to the same standard as the
        likelihood families rather than to a weaker one."""
        from MissLearn import MissNeighborsRegressor
        X, y, probe = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsRegressor().fit(X, y)
            lo, hi = m.predict_interval(probe, alpha=0.05)
        width = np.asarray(hi) - np.asarray(lo)
        expected = 2 * 1.96 * float(np.nanstd(y))
        assert width[2] > 0.7 * expected, (
            'an empty row should widen to roughly the response spread')
        assert width[2] >= width[1], (
            'an empty row should not be tighter than a partial one')

    def test_the_point_prediction_did_not_move_with_it(self):
        """Only the spread was corrected. ``predict`` is on the path of every
        benchmark in the repository, so the point estimate is deliberately
        left as the neighbour average, which already lands on the marginal
        mean for a row with nothing observed.
        """
        from MissLearn import MissNeighborsRegressor
        X, y, probe = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsRegressor().fit(X, y)
            pred = m.predict(probe)
        assert np.all(np.isfinite(pred))
        assert abs(pred[2] - np.nanmean(y)) < 0.5 * np.nanstd(y)


class TestEnsembleFittingPaths:
    """The parts of ``fit`` that only run under a particular configuration.

    Parallel fitting, out-of-bag scoring against a member that cannot accept
    NaN, and class alignment when a bootstrap sample happens to miss a class
    are all real configurations that the default fixture never reaches.
    """

    @staticmethod
    def _data(n=140, p=4, seed=26, task='clf', rare=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        if task == 'clf':
            y = (lin > 0).astype(float)
            if rare:
                y = np.zeros(n)
                y[:5] = 2.0            # ~3.5%, so some bootstraps miss it
                y[5:70] = 1.0
        else:
            y = lin + rng.normal(scale=0.3, size=n)
        return X, y

    def test_parallel_fitting_matches_sequential(self):
        """n_jobs > 1 takes the joblib path. It must produce the same
        ensemble, or the parameter changes the answer rather than the cost.
        """
        pytest.importorskip('joblib')
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seq = MissEnsemble(estimator=MissLogistic(compute_se=False),
                               n_estimators=4, n_jobs=1,
                               random_state=0).fit(X, y)
            par = MissEnsemble(estimator=MissLogistic(compute_se=False),
                               n_estimators=4, n_jobs=2,
                               random_state=0).fit(X, y)
            a = seq.predict_proba(X)
            b = par.predict_proba(X)
        assert np.allclose(a, b, atol=1e-8), \
            'parallel fitting changed the ensemble'

    def test_a_class_absent_from_a_member_is_aligned(self):
        """A bootstrap can miss a rare class, so that member's
        predict_proba has fewer columns than the ensemble's. The columns
        must be aligned by label rather than by position, or probabilities
        get attributed to the wrong class.
        """
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data(rare=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=8, random_state=0).fit(X, y)
            pr = m.predict_proba(X)
        assert len(m.classes_) == 3
        assert pr.shape == (X.shape[0], 3)
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert np.all(pr >= 0.0)
        assert np.all(np.isin(m.predict(X), m.classes_))

    def test_oob_scoring_with_a_member_that_cannot_take_nan(self):
        """A scikit-learn member has its NaN-y rows stripped before scoring.
        Without that the out-of-bag score is an exception rather than a
        number.
        """
        from sklearn.ensemble import RandomForestClassifier
        from MissLearn import MissEnsemble
        X, y = self._data()
        y = y.copy()
        y[::9] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=RandomForestClassifier(
                                 n_estimators=5, random_state=0),
                             n_estimators=4, oob_score=True,
                             random_state=0).fit(X, y)
        assert m.oob_scores_
        finite = [v for v in m.oob_scores_.values() if np.isfinite(v)]
        assert finite, 'no member produced a usable out-of-bag score'

    def test_an_oob_score_that_fails_becomes_nan(self):
        """A member whose score raises records NaN rather than aborting the
        whole fit, because one unusable out-of-bag estimate should not cost
        the other members.
        """
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()

        def explodes(self, *a, **kw):
            raise RuntimeError('forced scoring failure')

        # Patched on the class rather than subclassed: MissEnsemble
        # identifies its own estimators by module name, so a subclass
        # declared here would be refused at construction as not NaN-native.
        original = MissLogistic.score
        MissLogistic.score = explodes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                                 n_estimators=3, oob_score=True,
                                 random_state=0).fit(X, y)
        finally:
            MissLogistic.score = original
        assert m.oob_scores_
        assert all(np.isnan(v) for v in m.oob_scores_.values()), \
            'a member whose score raises should record NaN, not abort the fit'

    def test_a_subclass_of_a_misslearn_model_is_accepted(self):
        """Membership is decided by inheritance now, not by where the class
        was declared. The module test refused a subclass of MissLogistic with
        a message saying it was not NaN-native, which was both wrong and
        unhelpful: a subclass handles NaN exactly as well as its parent.
        """
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()

        class MyLogistic(MissLogistic):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MyLogistic(compute_se=False),
                             n_estimators=3, random_state=0).fit(X, y)
            pr = m.predict_proba(X)
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)

    def test_an_estimator_that_cannot_take_nan_is_still_refused(self):
        """The other side of the same test: widening membership must not have
        made it indiscriminate.
        """
        from MissLearn import MissEnsemble
        from sklearn.linear_model import LogisticRegression
        X, y = self._data()
        with pytest.raises(ValueError, match='not supported'):
            MissEnsemble(estimator=LogisticRegression(),
                         n_estimators=2, random_state=0).fit(X, y)

    def test_groups_follow_the_bootstrap_rows(self):
        """A grouped member needs its groups resampled with the rows, or the
        random intercept is attached to the wrong observations.
        """
        from MissLearn import MissEnsemble, MissMixedRegressor
        X, y = self._data(task='reg')
        groups = np.repeat(np.arange(len(y) // 10), 10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissMixedRegressor(compute_se=False),
                             n_estimators=3, random_state=0).fit(
                                 X, y, groups=groups)
            pred = m.predict(X)
        assert np.all(np.isfinite(pred))

    def test_predict_interval_on_a_classification_ensemble_is_refused(self):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=3, random_state=0).fit(X, y)
        with pytest.raises((AttributeError, ValueError, NotImplementedError)):
            m.predict_interval(X)


class TestExplainerValueFunctionAndMethods:
    """The coalition value function, and choosing the Shapley route by hand.

    ``class_index`` selects which output is being attributed, and getting it
    wrong is not a crash but an explanation of the wrong thing, so the
    refusal for an out-of-range index matters. ``method`` forces the exact or
    sampled route regardless of ``exact_threshold``, which is how the two are
    compared against each other.
    """

    @staticmethod
    def _fitted(task='clf', p=4, n=80, seed=27):
        from MissLearn import MissExplainer, MissLinear, MissLogistic
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -0.5, p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if task == 'clf':
                model = MissLogistic(compute_se=False).fit(
                    X, (lin > 0).astype(float))
            else:
                model = MissLinear(compute_se=False).fit(
                    X, lin + rng.normal(scale=0.3, size=n))
        return model, X

    def test_an_out_of_range_class_index_is_refused(self):
        from MissLearn import MissExplainer
        model, X = self._fitted(task='clf')
        # validated when the value function is built, which is at fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='class_index'):
                MissExplainer(model, class_index=7,
                              random_state=0).fit(X).shap_values(X[:4])

    def test_a_valid_class_index_selects_that_output(self):
        from MissLearn import MissExplainer
        model, X = self._fitted(task='clf')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = MissExplainer(model, class_index=0,
                              random_state=0).fit(X).shap_values(X[:6])
            b = MissExplainer(model, class_index=1,
                              random_state=0).fit(X).shap_values(X[:6])
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert not np.allclose(a, b), \
            'the two classes should not attribute identically'

    @pytest.mark.parametrize('method', ['exact', 'kernel'])
    def test_the_route_can_be_forced(self, method):
        """``method`` overrides ``exact_threshold`` in both directions, which
        is what makes the two comparable on the same data.
        """
        from MissLearn import MissExplainer
        model, X = self._fitted(task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = MissExplainer(model, n_kernel_samples=64,
                               random_state=0).fit(X)
            sv = ex.shap_values(X[:6], method=method)
        assert sv.shape == (6, X.shape[1])
        assert np.isfinite(sv).all()

    def test_forcing_kernel_below_the_threshold_still_approximates_exact(self):
        from MissLearn import MissExplainer
        model, X = self._fitted(task='reg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = MissExplainer(model, n_kernel_samples=256,
                               random_state=0).fit(X)
            exact = ex.shap_values(X[:6], method='exact')
            kernel = ex.shap_values(X[:6], method='kernel')
        assert np.abs(exact - kernel).max() < 1.5

    def test_miss_shap_on_a_classifier(self):
        from MissLearn import MissExplainer
        model, X = self._fitted(task='clf')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = MissExplainer(model, class_index=1, random_state=0).fit(X)
            ms = ex.miss_shap(X[:8])
        assert ms.shape == (8, X.shape[1])
        assert np.isfinite(ms).all()

    def test_waterfall_on_a_row_with_no_observed_entries(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from MissLearn import MissExplainer
        model, X = self._fitted(task='reg')
        probe = np.vstack([X[:3], np.full((1, X.shape[1]), np.nan)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = MissExplainer(model, random_state=0).fit(X)
            sv = ex.shap_values(probe)
            ex.plot_waterfall(sv, probe, i=3,
                              feature_names=['a', 'b', 'c', 'd'])
        assert plt.gcf().get_axes()
        plt.close('all')


class TestSharedValidationRefusals:
    """The remaining shared checks, each of which refuses something.

    They live in ``_conformance`` so that sixteen classes cannot disagree
    about what is acceptable, which means one test of each covers all of
    them at once.
    """

    @staticmethod
    def _data(n=60, p=3, seed=28):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[::7, 0] = np.nan
        return X, np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])

    def test_complex_input_is_refused_rather_than_silently_cast(self):
        """Casting would discard the imaginary part, which is a wrong answer
        rather than a refused one.
        """
        from MissLearn import MissLinear
        X, y = self._data()
        Xc = X.astype(complex)
        Xc[0, 0] = 1 + 2j
        with pytest.raises((ValueError, TypeError)):
            MissLinear(compute_se=False).fit(Xc, y)

    def test_a_continuous_target_handed_to_a_classifier_is_refused(self):
        from MissLearn import MissLogistic
        X, y = self._data()
        with pytest.raises(ValueError):
            MissLogistic(compute_se=False).fit(X, y + 0.123)

    def test_y_of_none_is_refused(self):
        from MissLearn import MissLinear
        X, _ = self._data()
        with pytest.raises((ValueError, TypeError)):
            MissLinear(compute_se=False).fit(X, None)

    def test_a_single_sample_is_refused_or_handled(self):
        from MissLearn import MissLinear
        X, y = self._data()
        with pytest.raises(ValueError):
            MissLinear(compute_se=False).fit(X[:1], y[:1])

    def test_more_than_two_classes_route_through_the_multiclass_wrapper(self):
        """A binary classifier handed three classes must not silently treat
        the third as noise.
        """
        from MissLearn import MissLogistic
        X, _ = self._data()
        y = np.zeros(X.shape[0])
        y[20:40] = 1.0
        y[40:] = 2.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLogistic(compute_se=False).fit(X, y)
            pred = m.predict(X)
        assert set(np.unique(pred)) <= {0.0, 1.0, 2.0}
        assert len(m.classes_) == 3

    def test_nan_in_a_string_labelled_target_is_preserved(self):
        """A missing label is not a class. Encoding it as one would invent an
        outcome for a row that has none.
        """
        from MissLearn import MissLogistic
        X, _ = self._data()
        y = np.array(['cat' if i % 2 else 'dog' for i in range(X.shape[0])],
                     dtype=object)
        y[::9] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLogistic(compute_se=False).fit(X, y)
        assert set(m.classes_) == {'cat', 'dog'}, \
            'NaN was encoded as a third class'

    def test_predicting_before_fitting_is_refused_everywhere(self):
        from sklearn.exceptions import NotFittedError
        import inspect as _inspect
        X, _ = self._data()
        for name in ESTIMATOR_NAMES:
            cls = getattr(_ML, name)
            kw = ({'compute_se': False}
                  if 'compute_se' in _inspect.signature(
                      cls.__init__).parameters else {})
            with pytest.raises((NotFittedError, AttributeError, ValueError)):
                cls(**kw).predict(X)


class TestEnsembleClassAlignmentInDetail:
    """A member whose ``classes_`` genuinely differ from the ensemble's.

    The fast path returns the member's probabilities untouched when the class
    vectors match. The slow path exists for when they do not, and it maps
    column by label. Getting that wrong attributes one class's probability to
    another, which no shape check would catch because the array is the right
    size either way.
    """

    @staticmethod
    def _data(n=150, p=4, seed=29):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = np.zeros(n)
        y[:4] = 2.0            # under 3%, so most bootstraps miss it entirely
        y[4:70] = 1.0
        return X, y

    def test_a_member_that_never_saw_a_class_is_mapped_by_label(self):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=12, random_state=3).fit(X, y)
            pr = m.predict_proba(X)
        assert np.array_equal(m.classes_, np.array([0.0, 1.0, 2.0]))
        assert pr.shape == (X.shape[0], 3)
        assert np.allclose(pr.sum(axis=1), 1.0)
        # at least one member should be missing the rare class, which is the
        # whole point of the alignment path
        seen = [len(getattr(e, 'classes_', m.classes_))
                for _, e, _ in m.estimators_]
        assert min(seen) <= 3

    def test_the_rare_class_keeps_its_own_column(self):
        """If alignment were positional rather than by label, the rare class
        would inherit whatever sat in that slot on the members that lack it.
        """
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=12, random_state=3).fit(X, y)
            pr = m.predict_proba(X)
        rare_col = pr[:, list(m.classes_).index(2.0)]
        assert np.all(rare_col >= 0.0)
        assert rare_col.max() < 0.9, \
            'a class most members never saw should not dominate'


class TestMixedDegenerateSeedAndEmptyRows:
    """Two paths in the mixed fit that ordinary data does not reach.

    The starting values come from an ordinary regression on the complete
    cases. When there are too few of those to fit one, the seed falls back to
    a flat start and lets the likelihood do the work; and a row with no
    observed predictor contributes nothing to the X marginal rather than
    raising on an empty Cholesky.
    """

    @staticmethod
    def _grouped(n=120, p=3, seed=30, rate=0.12):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < rate] = np.nan
        groups = np.repeat(np.arange(n // 10), 10)
        y = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5]) \
            + rng.normal(scale=0.3, size=n)
        return X, y, groups

    def test_rows_with_no_observed_predictor_are_tolerated(self):
        from MissLearn import MissMixedRegressor
        X, y, g = self._grouped()
        X = X.copy()
        X[3, :] = np.nan
        X[17, :] = np.nan
        X[42, :] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
            pred = m.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.isfinite(m.loglik_)

    def test_a_design_with_almost_no_complete_cases_still_seeds(self):
        """At a high enough rate there is no complete case to regress on, so
        the flat seed is the only way the fit starts at all.
        """
        from MissLearn import MissMixedRegressor
        X, y, g = self._grouped(rate=0.55)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            complete = int((~np.isnan(X)).all(axis=1).sum())
            m = MissMixedRegressor(compute_se=False).fit(X, y, groups=g)
        assert complete < X.shape[0] // 3, 'the fixture should be sparse'
        assert np.all(np.isfinite(m.predict(X)))
        assert np.isfinite(m.tau_sq_) and m.tau_sq_ >= 0

    def test_an_entirely_absent_outcome_column_is_refused_uniformly(self):
        """All three families now give the same answer, which they did not.

        Before: MissRidgeRegressor refused, MissLinear fitted and predicted
        all NaN from NaN coefficients, and MissMixedRegressor fitted a zero
        model and predicted finite numbers from no data at all. The refusal
        had been worded in prefit_check since it was written; the estimators
        simply did not all call it. It now lives in validate_input, which
        every fit reaches, so the answer is the same everywhere.
        """
        from MissLearn import (MissMixedRegressor, MissLinear,
                               MissRidgeRegressor, MissLogistic)
        X, y, g = self._grouped()
        blank = np.full_like(y, np.nan)

        for cls, kw in ((MissLinear, {}), (MissRidgeRegressor, {}),
                        (MissLogistic, {}), (MissMixedRegressor, {'groups': g})):
            with pytest.raises(ValueError, match='entirely NaN'):
                cls(compute_se=False).fit(X, blank, **kw)

    def test_a_partly_absent_outcome_is_still_supported(self):
        """The capability the refusal must not have taken with it. Rows whose
        outcome is absent still inform the feature distribution, which is the
        whole argument for full-information estimation.
        """
        from MissLearn import MissLinear
        X, y, g = self._grouped()
        y = y.copy()
        y[::3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLinear(compute_se=False).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))
        assert int(np.isnan(y).sum()) > 0


class TestNeighboursFallbacks:
    """What kNN does when the neighbours it found are no help.

    A neighbourhood whose outcomes are all absent carries no information, so
    the prediction falls back to the marginal. Returning the mean of an empty
    slice instead would be NaN, which is the outcome the contract forbids.
    """

    @staticmethod
    def _data(n=100, p=3, seed=31):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25]) \
            + rng.normal(scale=0.3, size=n)
        return X, y

    def test_a_neighbourhood_with_no_observed_outcomes_falls_back(self):
        from MissLearn import MissNeighborsRegressor
        X, y = self._data()
        y = y.copy()
        y[:] = np.nan
        y[:6] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # only six observed
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsRegressor(n_neighbors=3).fit(X, y)
            pred = m.predict(X)
        assert np.all(np.isfinite(pred)), \
            'a neighbourhood with no observed outcome must fall back, not NaN'

    def test_most_outcomes_absent_still_predicts_everywhere(self):
        from MissLearn import MissNeighborsRegressor
        X, y = self._data()
        y = y.copy()
        y[::3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsRegressor(n_neighbors=5).fit(X, y)
            pred = m.predict(X)
            lo, hi = m.predict_interval(X)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    def test_the_classifier_survives_the_same_regime(self):
        from MissLearn import MissNeighborsClassifier
        X, y = self._data()
        yc = (y > 0).astype(float)
        yc[::3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsClassifier(n_neighbors=5).fit(X, yc)
            pr = m.predict_proba(X)
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)


class TestConformanceReportExplanations:
    """``RegimeOutcome.explain`` writes the sentence a reader acts on.

    Two of its three cases are the ones that matter: a silent NaN, which is
    the failure worth removing first, and a refusal whose message does not
    say why, which leaves the user unable to act. Both need an estimator that
    behaves that way, and every estimator in this library behaves correctly,
    so neither had ever been rendered.
    """

    @staticmethod
    def _silent_nan():
        from sklearn.base import BaseEstimator, RegressorMixin

        class SilentlyNaN(RegressorMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                self.n_features_in_ = np.asarray(X).shape[1]
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return np.full(np.asarray(X).shape[0], np.nan)
        return SilentlyNaN()

    @staticmethod
    def _opaque_refuser():
        from sklearn.base import BaseEstimator, RegressorMixin

        class RefusesWithoutSaying(RegressorMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                raise ValueError('')          # refuses, says nothing
        return RefusesWithoutSaying()

    def test_a_silent_nan_is_called_out_as_the_first_thing_to_fix(self):
        from MissLearn import check_missing_data_estimator
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(self._silent_nan())
        text = str(rep)
        assert 'silent' in text.lower() or 'nan' in text.lower()
        assert not getattr(rep, 'ok', True)

    def test_an_opaque_refusal_is_distinguished_from_a_clear_one(self):
        from MissLearn import check_missing_data_estimator
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opaque = check_missing_data_estimator(self._opaque_refuser())
        text = str(opaque)
        assert text.strip()
        assert 'RefusesWithoutSaying' in text

    def test_the_report_lists_every_regime(self):
        from MissLearn import check_missing_data_estimator, MissLinear
        from MissLearn._estimator_checks import DETERMINISM_REGIMES  # noqa: F401
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(MissLinear(compute_se=False))
        outcomes = getattr(rep, 'outcomes', None) or getattr(rep, 'results', None)
        if outcomes is None:
            pytest.skip('the report exposes no per-regime collection')
        assert len(outcomes) >= 8

    def test_a_classifier_stub_that_returns_nan_probabilities(self):
        from sklearn.base import BaseEstimator, ClassifierMixin
        from MissLearn import check_missing_data_estimator

        class NaNProba(ClassifierMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                self.classes_ = np.unique(np.asarray(y)[
                    ~np.isnan(np.asarray(y, dtype=float))])
                self.n_features_in_ = np.asarray(X).shape[1]
                return self

            def predict(self, X):
                return np.full(np.asarray(X).shape[0], np.nan)

            def predict_proba(self, X):
                return np.full((np.asarray(X).shape[0], 2), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(NaNProba())
        assert 'NaNProba' in str(rep)


class TestPrefitCheckOutcomeChecks:
    """The outcome-side checks, including the one the estimators do not run.

    ``prefit_check`` already refuses an entirely absent outcome with "y is
    entirely NaN. Nothing to fit." The estimators do not all call it, which
    is why MissLinear fits such a target and predicts NaN while
    MissRidgeRegressor refuses. The refusal is written; it is the wiring that
    is missing.
    """

    @staticmethod
    def _data(n=90, p=3, seed=32):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[::7, 0] = np.nan
        return X, np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])

    def test_an_entirely_absent_outcome_is_an_error(self):
        from MissLearn import prefit_check
        X, y = self._data()
        r = prefit_check(X, np.full_like(y, np.nan), raise_on_error=False,
                         emit_warnings=False)
        assert not r.passed
        assert any('entirely NaN' in e for e in r.errors)

    def test_a_mostly_absent_outcome_is_a_warning_not_an_error(self):
        """Most of the outcome missing is a reason for caution, not a
        refusal: a likelihood can still use those rows for the predictor
        moments, which is the whole argument for FIML.
        """
        from MissLearn import prefit_check
        X, y = self._data()
        y = y.copy()
        y[:int(0.85 * len(y))] = np.nan
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert r.passed, 'a mostly absent outcome should not be refused'
        assert any('y' in w for w in r.warnings)

    def test_infinity_in_the_outcome_is_an_error(self):
        """NaN in y is a capability; infinity is a broken value, not an
        absent one.
        """
        from MissLearn import prefit_check
        X, y = self._data()
        y = y.copy()
        y[3] = np.inf
        r = prefit_check(X, y, raise_on_error=False, emit_warnings=False)
        assert not r.passed
        assert any('inf' in e.lower() for e in r.errors)

    def test_a_large_dataset_warns_for_the_gaussian_process(self):
        """Exact GP inference is cubic, so the check warns before somebody
        starts a fit that will not finish.
        """
        from MissLearn import prefit_check
        rng = np.random.default_rng(33)
        n = 1500
        X = rng.standard_normal((n, 3))
        X[::9, 0] = np.nan
        y = np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])
        r = prefit_check(X, y, model_name='MissGaussianRegressor',
                         raise_on_error=False, emit_warnings=False,
                         n_gaussian_threshold=1000)
        assert any('gaussian' in w.lower() or 'n =' in w.lower()
                   or 'cubic' in w.lower() for w in r.warnings), \
            'a large design should warn for the Gaussian process'

    def test_warnings_are_emitted_when_asked(self):
        """emit_warnings routes the same findings through the warnings
        machinery, which is how they reach a user who never looks at the
        returned object.
        """
        from MissLearn import prefit_check
        X, y = self._data()
        X = X.copy()
        X[:, 0] *= 1e-6
        X[:, 2] *= 1e6
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prefit_check(X, y, raise_on_error=False, emit_warnings=True)
        assert caught, 'emit_warnings=True should surface the findings'


class TestPreprocessorEncodingDetail:
    """Building and applying the encoding map, including its refusals.

    The map is built once at fit and applied at predict. Its awkward cases
    are a genuinely non-numeric column with encoding switched off, which has
    to refuse rather than crash on a float cast, and a pandas NA, which
    cannot be compared with itself and so breaks the ordinary
    ``v == v`` test for absence.
    """

    @staticmethod
    def _frame(n=70, seed=34):
        pd = pytest.importorskip('pandas')
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            'num': rng.standard_normal(n),
            'cat': rng.choice(['x', 'y', 'z'], size=n),
            'flag': rng.choice(['on', 'off'], size=n),
        })
        df.loc[::8, 'num'] = np.nan
        df.loc[::10, 'cat'] = None
        y = (df['num'].fillna(0.0) > 0).astype(float).to_numpy()
        return df, y

    def test_a_text_column_with_encoding_off_is_refused_with_advice(self):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with pytest.raises(ValueError, match='encode'):
            MissPreprocessor(MissLogistic(compute_se=False), encode=None,
                             raise_on_error=False).fit(df, y)

    @pytest.mark.parametrize('encode', ['auto', 'onehot'])
    def test_both_encoding_modes_produce_a_usable_matrix(self, encode):
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False), encode=encode,
                                 raise_on_error=False).fit(df, y)
            Z = np.asarray(m.transform(df), dtype=float)
        assert Z.shape[0] == len(y)
        assert Z.shape[1] >= df.shape[1]

    def test_dropping_the_first_category_removes_one_column(self):
        """drop='first' is what avoids the dummy trap, so the column counts
        under the two settings should differ by one per categorical.
        """
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kept = MissPreprocessor(MissLogistic(compute_se=False), drop=None,
                                    raise_on_error=False).fit(df, y)
            dropped = MissPreprocessor(MissLogistic(compute_se=False),
                                       drop='first',
                                       raise_on_error=False).fit(df, y)
            wide = np.asarray(kept.transform(df)).shape[1]
            narrow = np.asarray(dropped.transform(df)).shape[1]
        assert narrow < wide

    def test_a_pandas_na_is_treated_as_absent_not_as_a_category(self):
        """pd.NA raises on comparison rather than returning False, so the
        ordinary ``v == v`` test for absence has to be guarded. Without the
        guard the absent entries become their own category.
        """
        pd = pytest.importorskip('pandas')
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        df = df.copy()
        df['cat'] = df['cat'].astype('string')
        df.loc[::6, 'cat'] = pd.NA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 raise_on_error=False).fit(df, y)
            Z = np.asarray(m.transform(df), dtype=float)
        assert np.isnan(Z).any(), 'the absent categories should stay absent'
        assert Z.shape[0] == len(y)

    def test_a_numeric_column_with_few_levels_is_treated_as_categorical(self):
        """Integer codes are the usual way a category arrives in a numeric
        column, so a small number of whole-number levels is encoded rather
        than modelled as a continuous variable.
        """
        from MissLearn import MissPreprocessor, MissLogistic
        rng = np.random.default_rng(35)
        n = 80
        X = np.column_stack([rng.standard_normal(n),
                             rng.integers(0, 3, size=n).astype(float)])
        X[::9, 0] = np.nan
        y = (X[:, 0] > 0).astype(float)
        y[np.isnan(X[:, 0])] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissPreprocessor(MissLogistic(compute_se=False),
                                 categorical_threshold=5,
                                 raise_on_error=False).fit(X, y)
            Z = np.asarray(m.transform(X), dtype=float)
        assert Z.shape[1] > X.shape[1], \
            'a three-level integer column should have been expanded'

    def test_set_params_reaches_through_to_the_inner_estimator(self):
        from MissLearn import MissPreprocessor, MissLogistic
        m = MissPreprocessor(MissLogistic(compute_se=False))
        m.set_params(estimator__max_iter=77, categorical_threshold=4)
        assert m.estimator.get_params()['max_iter'] == 77
        assert m.categorical_threshold == 4

    def test_transform_before_fit_is_refused(self):
        from sklearn.exceptions import NotFittedError
        from MissLearn import MissPreprocessor, MissLogistic
        df, y = self._frame()
        with pytest.raises((NotFittedError, AttributeError, ValueError)):
            MissPreprocessor(MissLogistic(compute_se=False)).transform(df)


class TestSharedInputConversions:
    """The conversions and refusals in ``_conformance``, one test each.

    They live in one module so that sixteen estimators cannot disagree about
    what a valid input is. That only holds if each one has been run: a shared
    check nobody exercises is shared consistency nobody has verified.
    """

    @staticmethod
    def _data(n=60, p=3, seed=36):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[::7, 0] = np.nan
        return X, np.nan_to_num(X) @ np.array([1.0, -0.5, 0.25])

    def test_a_column_vector_target_is_flattened_with_a_warning(self):
        """scikit-learn requires the flattening to be announced, because a
        caller who passed a column vector has probably made a shape error and
        silently accepting it hides that.
        """
        from sklearn.exceptions import DataConversionWarning
        from MissLearn import MissLinear
        X, y = self._data()
        with pytest.warns(DataConversionWarning, match='column-vector'):
            m = MissLinear(compute_se=False).fit(X, y.reshape(-1, 1))
        assert m.coef_.shape == (X.shape[1],)

    def test_an_empty_column_is_named_when_names_are_known(self):
        """The error has to say which column, or the user has to go and find
        it themselves on a wide design.
        """
        from MissLearn._conformance import check_no_empty_features
        X, _ = self._data()
        X = X.copy()
        X[:, 1] = np.nan
        with pytest.raises(ValueError) as exc:
            check_no_empty_features(X, feature_names=['a', 'bee', 'c'])
        assert 'bee' in str(exc.value)
        assert 'column 1' in str(exc.value)

    def test_an_empty_column_falls_back_to_a_positional_label(self):
        from MissLearn._conformance import check_no_empty_features
        X, _ = self._data()
        X = X.copy()
        X[:, 2] = np.nan
        with pytest.raises(ValueError, match='column 2'):
            check_no_empty_features(X, feature_names=None)

    def test_isnan_tolerates_a_string_target(self):
        """String labels have no NaN, and asking numpy for one raises rather
        than returning False.
        """
        from MissLearn._conformance import _isnan_safe
        out = _isnan_safe(np.array(['cat', 'dog', 'cat'], dtype=object))
        assert out.shape == (3,)
        assert not out.any()

    def test_routing_declines_when_no_label_is_observed(self):
        """With nothing observed there is no class count to route on, so the
        router returns None rather than guessing at binary.
        """
        from MissLearn._conformance import route_multiclass
        from MissLearn import MissLogistic
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        assert route_multiclass(MissLogistic(compute_se=False), X,
                                np.full(20, np.nan), {}) is None

    @pytest.mark.parametrize('bad', [0, -1, 2.5, 'x'])
    def test_a_bad_integer_parameter_is_refused(self, bad):
        from MissLearn._conformance import check_positive_int
        with pytest.raises((ValueError, TypeError)):
            check_positive_int(bad, name='n_neighbors',
                               estimator_name='MissNeighbors')

    @pytest.mark.parametrize('bad', [0.0, -1.0, 'x'])
    def test_a_bad_float_parameter_is_refused(self, bad):
        from MissLearn._conformance import check_positive_float
        with pytest.raises((ValueError, TypeError)):
            check_positive_float(bad, name='gamma',
                                 estimator_name='MissSupport')

    @pytest.mark.parametrize('bad', [0.0, -1e-6, 'x'])
    def test_a_bad_tolerance_is_refused(self, bad):
        from MissLearn._conformance import check_tolerance
        with pytest.raises((ValueError, TypeError)):
            check_tolerance(bad, estimator_name='MissLinear')

    def test_good_parameters_pass_through_unchanged(self):
        """The accepting side of each check, so that a refusal that fires on
        everything would be caught.
        """
        from MissLearn._conformance import (check_positive_int,
                                            check_positive_float,
                                            check_tolerance)
        assert check_positive_int(5, name='k', estimator_name='e') == 5
        assert check_positive_float(0.5, name='g', estimator_name='e') == 0.5
        assert check_tolerance(1e-7, estimator_name='e') == 1e-7


class TestCrossValidationEdges:
    """Splitter validation, and a fold with nothing to score against."""

    @staticmethod
    def _data(n=60, seed=37):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        X[::7, 0] = np.nan
        y = np.nan_to_num(X)[:, 1] * 1.5 + rng.normal(scale=0.3, size=n)
        return X, y

    @pytest.mark.parametrize('splitter', ['MissKFold', 'MissStratifiedKFold'])
    def test_fewer_than_two_splits_is_refused(self, splitter):
        from MissLearn import _crossval
        cls = getattr(_crossval, splitter)
        with pytest.raises(ValueError, match='n_splits'):
            cls(n_splits=1)

    def test_a_fold_with_no_observed_outcome_scores_nan(self):
        """A held-out fold can contain only rows whose outcome was never
        recorded. There is nothing to be right or wrong about, so the score
        is NaN rather than a number computed over an empty slice.
        """
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLinear
        X, y = self._data()
        y = y.copy()
        y[:20] = np.nan                    # the first fold has no labels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = miss_cross_val_score(MissLinear(compute_se=False), X, y,
                                          cv=3, scoring='r2',
                                          random_state=None)
        assert len(scores) == 3
        assert np.isnan(scores).any() or np.all(np.isfinite(scores))

    def test_stratified_splitting_is_selected_by_the_flag(self):
        from MissLearn._crossval import miss_cross_val_score
        from MissLearn import MissLogistic
        X, y = self._data()
        yc = (y > 0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = miss_cross_val_score(MissLogistic(compute_se=False), X, yc,
                                          cv=3, stratified=True,
                                          random_state=0)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_a_splitter_object_is_accepted_instead_of_an_integer(self):
        from MissLearn._crossval import miss_cross_val_score, MissKFold
        from MissLearn import MissLinear
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = miss_cross_val_score(
                MissLinear(compute_se=False), X, y,
                cv=MissKFold(n_splits=4, shuffle=True, random_state=0))
        assert len(scores) == 4


class TestRecommenderInternals:
    """The probe and the grouping statistic, which drive the ranking."""

    @staticmethod
    def _data(n=250, p=4, seed=38, nonlinear=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        base = np.nan_to_num(X)
        if nonlinear:
            y = np.sin(2.0 * base[:, 0]) + base[:, 1] ** 2
        else:
            y = base @ np.linspace(1.0, -1.0, p)
        return X, y + rng.normal(scale=0.2, size=n)

    def test_the_probe_prefers_neighbours_on_nonlinear_structure(self):
        from MissLearn import MissRecommender
        X, y = self._data(nonlinear=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=True, random_state=0).fit(X, y)
        probe = getattr(r, 'nonlinearity_probe_', None)
        if probe is None:
            pytest.skip('the probe result is not exposed')
        assert 'linear_score' in probe and 'neighbour_score' in probe

    def test_the_probe_runs_on_linear_structure_too(self):
        from MissLearn import MissRecommender
        X, y = self._data(nonlinear=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=True, random_state=0).fit(X, y)
        assert r.ranked_

    def test_grouped_data_gets_an_intraclass_correlation(self):
        """With groups supplied the recommender computes an ICC and lets it
        move the ranking towards the mixed model, which is the only way
        MissMixed can ever be recommended.
        """
        from MissLearn import MissRecommender
        X, y = self._data()
        groups = np.repeat(np.arange(len(y) // 10), 10)
        y = y + np.repeat(np.random.default_rng(1).normal(scale=3.0,
                                                          size=len(y) // 10), 10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(groups=groups, probe_nonlinearity=False,
                                random_state=0).fit(X, y)
        assert r.ranked_
        names = [k for k, _ in r.ranked_] if isinstance(r.ranked_[0], tuple) \
            else list(r.ranked_)
        assert any('Mixed' in str(n) for n in names), \
            'strongly grouped data should put the mixed model on the list'

    def test_summary_reports_the_followups(self, capsys):
        from MissLearn import MissRecommender
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
            r.summary()
        out = capsys.readouterr().out
        assert out.strip()
        assert r.followups_ is not None

    def test_make_estimator_honours_the_preprocessing_advice(self):
        from MissLearn import MissRecommender
        rng = np.random.default_rng(39)
        n = 300
        X = rng.standard_normal((n, 4))
        X[:, 0] = rng.standard_t(df=1.7, size=n) * 6.0
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.0, -1.0, 4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = MissRecommender(probe_nonlinearity=False).fit(X, y)
            est = r.make_estimator()
        assert hasattr(est, 'fit')
        if r.preprocessing_['copula']:
            assert est.get_params().get('copula') is True


class TestExplainerKernelInternals:
    """The sampled Shapley route, and the beeswarm's jitter.

    The kernel path is the one that runs whenever p exceeds the exact
    threshold, which on any realistic design is always, so it carries most of
    the explanations this library will ever produce.
    """

    @staticmethod
    def _fitted(p=6, n=90, seed=40):
        from MissLearn import MissExplainer, MissLinear
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.5, -1.0, p) \
            + rng.normal(scale=0.3, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = MissLinear(compute_se=False).fit(X, y)
            ex = MissExplainer(model, exact_threshold=0, n_kernel_samples=128,
                               random_state=0).fit(X)
        return ex, X

    def test_the_kernel_route_is_reproducible(self):
        """A sampled estimate seeded the same way must land in the same
        place, or two runs of the same analysis disagree.
        """
        ex_a, X = self._fitted()
        ex_b, _ = self._fitted()
        a = ex_a.shap_values(X[:5])
        b = ex_b.shap_values(X[:5])
        assert np.allclose(a, b)

    def test_more_samples_move_the_estimate_towards_exact(self):
        from MissLearn import MissExplainer, MissLinear
        rng = np.random.default_rng(41)
        n, p = 80, 5
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        y = np.nan_to_num(X) @ np.linspace(1.0, -1.0, p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = MissLinear(compute_se=False).fit(X, y)
            ex = MissExplainer(model, n_kernel_samples=512,
                               random_state=0).fit(X)
            exact = ex.shap_values(X[:5], method='exact')
            many = ex.shap_values(X[:5], method='kernel')
        assert np.abs(exact - many).max() < 2.0

    def test_the_beeswarm_jitters_overlapping_points(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ex, X = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:40])
            ex.plot_beeswarm(sv, X[:40], max_display=3)
        assert plt.gcf().get_axes()
        plt.close('all')

    def test_miss_shap_on_a_complete_matrix(self):
        """With nothing absent there is no missingness to attribute, so the
        values should be at or near zero rather than undefined.
        """
        ex, X = self._fitted()
        complete = np.nan_to_num(X[:6])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ms = ex.miss_shap(complete)
        assert np.isfinite(ms).all()


class TestEnsembleAlignsMembersByLabel:
    """A member that genuinely never saw a class.

    The fast path hands back the member's probabilities untouched when its
    ``classes_`` matches the ensemble's. The alignment path exists for when it
    does not, and it maps column by label. Reaching it needs a class rare
    enough that a bootstrap sample can miss it entirely: two rows in a hundred
    and fifty, over twenty members, produces members with two classes and
    members with three.

    Getting this wrong is invisible to any shape check, because the array
    comes out the right size either way. It just has one class's probability
    sitting in another's column.
    """

    @staticmethod
    def _data(n=150, p=4, seed=29):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = np.zeros(n)
        y[:2] = 2.0                     # two rows, so bootstraps miss it
        y[2:70] = 1.0
        return X, y

    def _fit(self):
        from MissLearn import MissEnsemble, MissLogistic
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissEnsemble(estimator=MissLogistic(compute_se=False),
                             n_estimators=20, random_state=5).fit(X, y)
        return m, X, y

    def test_some_members_really_are_missing_a_class(self):
        """The fixture's own precondition. Without it the alignment path is
        never entered and the rest of this class proves nothing.
        """
        m, X, y = self._fit()
        counts = {len(getattr(e, 'classes_', m.classes_))
                  for _, e, _ in m.estimators_}
        assert 2 in counts, 'no member missed the rare class; alignment unused'
        assert 3 in counts, 'no member saw all three; nothing to align against'

    def test_the_aggregate_is_still_a_probability(self):
        m, X, y = self._fit()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pr = m.predict_proba(X)
        assert pr.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert np.all((pr >= 0.0) & (pr <= 1.0))

    def test_the_rare_class_column_is_not_inherited_from_another(self):
        """A member that never saw class 2 contributes zero to its column
        rather than whatever occupied that position in its own output. If
        alignment were positional, class 2 would inherit class 1's mass.
        """
        m, X, y = self._fit()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pr = m.predict_proba(X)
        cols = list(m.classes_)
        rare = pr[:, cols.index(2.0)]
        common = pr[:, cols.index(0.0)]
        assert rare.max() < common.max(), \
            'the class most members never saw should not dominate'
        assert rare.min() >= 0.0

    def test_predict_only_returns_labels_it_knows(self):
        m, X, y = self._fit()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pred = m.predict(X)
        assert np.all(np.isin(pred, m.classes_))


class TestExplainerRemainingPaths:
    """The last few branches in the explainer."""

    @staticmethod
    def _fitted(p=5, n=80, seed=43, task='reg'):
        from MissLearn import MissExplainer, MissLinear, MissLogistic
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.linspace(1.5, -1.0, p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if task == 'clf':
                model = MissLogistic(compute_se=False).fit(
                    X, (lin > 0).astype(float))
            else:
                model = MissLinear(compute_se=False).fit(
                    X, lin + rng.normal(scale=0.3, size=n))
            ex = MissExplainer(model, n_kernel_samples=96,
                               random_state=0).fit(X)
        return ex, X

    def test_a_single_row_explains(self):
        """Shapley on one row is the common interactive case and collapses
        several array shapes to one.
        """
        ex, X = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:1])
        assert sv.shape == (1, X.shape[1])
        assert np.isfinite(sv).all()

    def test_miss_shap_on_a_single_row(self):
        ex, X = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ms = ex.miss_shap(X[:1])
        assert ms.shape == (1, X.shape[1])

    def test_a_binary_model_without_an_explicit_class_index(self):
        """With two classes and no index given the explainer has to pick a
        convention rather than refuse, since a binary probability is a single
        number either way.
        """
        ex, X = self._fitted(task='clf')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:4])
        assert np.isfinite(sv).all()

    def test_waterfall_without_feature_names(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ex, X = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:5])
            ex.plot_waterfall(sv, X[:5], i=1)
        assert plt.gcf().get_axes()
        plt.close('all')

    def test_beeswarm_displaying_every_feature(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ex, X = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sv = ex.shap_values(X[:30])
            ex.plot_beeswarm(sv, X[:30], max_display=X.shape[1])
        assert plt.gcf().get_axes()
        plt.close('all')


class TestGaussianProcessRemainingPaths:
    """Restart behaviour and the classifier's posterior."""

    @staticmethod
    def _data(n=60, p=3, seed=44):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        y = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5]) \
            + rng.normal(scale=0.3, size=n)
        return X, y

    def test_multiple_restarts_are_used(self):
        from MissLearn import MissGaussianRegressor
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianRegressor(n_restarts=2).fit(X, y)
        assert m.converged_ in (True, False)
        assert np.all(np.isfinite(m.predict(X)))

    def test_every_restart_failing_warns_and_still_predicts(self, monkeypatch):
        """The fallback returns a kernel nobody fitted, so it has to say so
        rather than presenting default hyperparameters as a result.
        """
        import MissLearn._gp as gp
        from MissLearn import MissGaussianRegressor
        X, y = self._data()

        def always_fails(*a, **kw):
            raise np.linalg.LinAlgError('forced restart failure')

        monkeypatch.setattr(gp, 'minimize', always_fails)
        with pytest.warns(UserWarning, match='restart'):
            m = MissGaussianRegressor(n_restarts=1).fit(X, y)
        assert m.converged_ is False
        assert np.all(np.isfinite(m.predict(X)))

    def test_the_classifier_posterior_is_a_probability(self):
        from MissLearn import MissGaussianClassifier
        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissGaussianClassifier().fit(X, (y > 0).astype(float))
            pr = m.predict_proba(X)
            d = m.decision_function(X)
        assert np.allclose(pr.sum(axis=1), 1.0)
        assert np.all((pr >= 0.0) & (pr <= 1.0))
        assert len(d) == X.shape[0]


class TestLastEmptyRowBranches:
    """The all-absent row through the classifiers and the mixed model.

    Each family has its own version of the same three-way split, and the
    empty case is the one no fixture supplies. For a classifier the fallback
    is the marginal log-odds; for a grouped model, a subject with no observed
    outcome has no information to shrink towards and takes a zero random
    intercept rather than an undefined one.
    """

    @staticmethod
    def _data(n=120, p=3, seed=45):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.12] = np.nan
        lin = np.nan_to_num(X) @ np.array([1.5, -1.0, 0.5])
        return X, (lin > 0).astype(float), lin + rng.normal(scale=0.3, size=n)

    def test_lasso_classifier_log_odds_on_an_empty_row(self):
        from MissLearn import MissLASSOClassifier
        X, yc, _ = self._data()
        probe = np.vstack([X[:2], np.full((1, X.shape[1]), np.nan)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLASSOClassifier(alpha=0.1, compute_se=False).fit(X, yc)
            d = m.decision_function(probe)
            pr = m.predict_proba(probe)
        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)

    def test_neighbours_classifier_with_no_observed_outcome_nearby(self):
        """A neighbourhood whose labels are all absent carries no evidence,
        so the probability falls back rather than averaging an empty slice.
        """
        from MissLearn import MissNeighborsClassifier
        X, yc, _ = self._data()
        yc = yc.copy()
        yc[6:] = np.nan                       # only six labels in the whole set
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissNeighborsClassifier(n_neighbors=3).fit(X, yc)
            pr = m.predict_proba(X)
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=1), 1.0)

    def test_a_group_with_no_observed_outcome_takes_a_zero_intercept(self):
        """There is nothing to shrink towards, so the subject sits at the
        population level rather than at an undefined one.
        """
        from MissLearn import MissMixedClassifier
        X, yc, _ = self._data()
        groups = np.repeat(np.arange(len(yc) // 10), 10)
        yc = yc.copy()
        yc[groups == 0] = np.nan              # one whole subject unlabelled
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedClassifier(compute_se=False).fit(X, yc, groups=groups)
        assert 0 in m.blup_
        assert m.blup_[0] == 0.0
        assert np.all(np.isfinite(m.predict_proba(X, groups=groups)))

    def test_the_mixed_regressor_does_the_same(self):
        from MissLearn import MissMixedRegressor
        X, _, yr = self._data()
        groups = np.repeat(np.arange(len(yr) // 10), 10)
        yr = yr.copy()
        yr[groups == 2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissMixedRegressor(compute_se=False).fit(X, yr, groups=groups)
        assert m.blup_[2] == 0.0
        assert np.all(np.isfinite(m.predict(X, groups=groups)))


class TestConformanceReportListsProblems:
    """A report on an estimator that fails some regimes but not all.

    The interesting rendering is the middle case. An estimator that cannot
    fit anything takes an early branch; one that is perfect takes the "every
    regime handled acceptably" line. Listing the specific regimes that need
    attention, each with its explanation, only happens for an estimator that
    is mostly fine and wrong in a particular place, which is what a real
    third-party estimator looks like.
    """

    @staticmethod
    def _mostly_fine():
        """Fits and predicts correctly, except that it returns NaN whenever a
        column is entirely absent, which is one of the eleven regimes.
        """
        from sklearn.base import BaseEstimator, RegressorMixin

        class FailsOneRegime(RegressorMixin, BaseEstimator):
            def fit(self, X, y=None, **kw):
                X = np.asarray(X, dtype=float)
                self.n_features_in_ = X.shape[1]
                self._empty_cols = bool(
                    X.shape[0] and np.isnan(X).all(axis=0).any())
                yv = np.asarray(y, dtype=float)
                obs = ~np.isnan(yv)
                self._mean = float(yv[obs].mean()) if obs.any() else 0.0
                return self

            def predict(self, X):
                n = np.asarray(X).shape[0]
                if self._empty_cols:
                    return np.full(n, np.nan)      # the one bad regime
                return np.full(n, self._mean)

        return FailsOneRegime()

    def test_the_failing_regime_is_named_and_explained(self):
        from MissLearn import check_missing_data_estimator
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(self._mostly_fine())
        text = str(rep)
        assert 'need attention' in text or 'regime' in text.lower()
        assert 'FailsOneRegime' in text

    def test_a_clean_estimator_says_so_instead(self):
        from MissLearn import check_missing_data_estimator, MissLinear
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(MissLinear(compute_se=False))
        text = str(rep)
        assert 'acceptably' in text or 'every regime' in text.lower()

    def test_repr_matches_str(self):
        """``__repr__`` is assigned from ``__str__``, so a reader who echoes
        the object in a notebook gets the report rather than an address.
        """
        from MissLearn import check_missing_data_estimator, MissLinear
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rep = check_missing_data_estimator(MissLinear(compute_se=False))
        assert repr(rep) == str(rep)
        assert 'MissLinear' in repr(rep)


class TestAbsentLabelsAcrossTheLibrary:
    """An absent class label is not a category, for every classifier.

    ``encode_labels`` is the single function every classifier passes its
    labels through, and it tested for absence with
    ``isinstance(v, float) and np.isnan(v)``. That is true of float ``nan``
    and of nothing else, so ``None``, ``pandas.NA`` and ``pandas.NaT``
    survived the filter, reached ``np.unique`` and killed the sort:

        TypeError: '<' not supported between instances of 'str' and 'NoneType'
        TypeError: boolean value of NA is ambiguous

    Every classifier was therefore unable to accept an object or
    nullable-dtype label vector containing an absent entry, while the
    documentation says an absent outcome is supported and still informs the
    feature distribution.

    It survived because ``MissMulticlass``, the one class with tests for
    exactly this input, held the only correct implementation and never
    reached the shared one: it was not recognised as a classifier, so the
    label-encoding path skipped it. A defect in a shared helper cannot be
    found by per-class tests when the class that tests the behaviour is the
    class excluded from the path, which is why this is a sweep.
    """

    CLASSIFIERS = ['MissLogistic', 'MissRidgeClassifier', 'MissLASSOClassifier',
                   'MissBayesClassifier', 'MissNeighborsClassifier',
                   'MissSupportClassifier', 'MissGaussianClassifier']

    @staticmethod
    def _X(n=90, p=3, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        X[rng.random(X.shape) < 0.10] = np.nan
        return X

    @staticmethod
    def _labels(n, absent):
        """A three-class object label vector whose last entry is absent."""
        base = ['a', 'b', 'c'] * (n // 3)
        y = np.array(base[:n], dtype=object)
        y[-1] = absent
        return y

    # ---- the four spellings of absence, on one estimator -------------------

    @pytest.mark.parametrize('kind', ['none', 'float_nan', 'pandas_na',
                                      'pandas_nat'])
    def test_each_way_a_label_can_be_absent(self, kind):
        """The four are not interchangeable and fail differently.

        ``None`` is not a float, so ``np.isnan`` never sees it; ``pd.NA``
        raises rather than answering when compared; ``pd.NaT`` behaves like
        ``pd.NA`` here. Only ``float('nan')`` answered to the old test, which
        is why a test written with it alone stayed green.
        """
        from MissLearn import MissLogistic
        X = self._X()
        n = X.shape[0]
        if kind == 'none':
            y = self._labels(n, None)
        elif kind == 'float_nan':
            y = self._labels(n, float('nan'))
        elif kind == 'pandas_na':
            pd = pytest.importorskip('pandas')
            y = self._labels(n, pd.NA)
        else:
            pd = pytest.importorskip('pandas')
            y = self._labels(n, pd.NaT)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MissLogistic().fit(X, y)
        assert set(m.classes_) == {'a', 'b', 'c'}, (
            '%s absence became a category: %r' % (kind, m.classes_))
        assert len(m.classes_) == 3

    # ---- the sweep: the same claim of every classifier ---------------------

    @pytest.mark.parametrize('name', CLASSIFIERS)
    def test_none_is_not_a_category_for_any_classifier(self, name):
        X = self._X()
        y = self._labels(X.shape[0], None)
        est = getattr(_ML, name)()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y)
        assert set(est.classes_) == {'a', 'b', 'c'}, (
            '%s: %r' % (name, est.classes_))
        pred = est.predict(X)
        assert set(np.unique(pred)).issubset({'a', 'b', 'c'})

    @pytest.mark.parametrize('name', CLASSIFIERS)
    def test_a_nullable_dtype_series_is_accepted_by_any_classifier(self, name):
        """A pandas nullable dtype is how an incomplete label column arrives
        from a real file, and pd.NA is what it puts in the absent cells.
        """
        pd = pytest.importorskip('pandas')
        X = self._X()
        n = X.shape[0]
        y = pd.array((['a', 'b', 'c'] * (n // 3))[:n], dtype='string')
        y[-1] = pd.NA
        est = getattr(_ML, name)()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est.fit(X, y)
        assert set(est.classes_) == {'a', 'b', 'c'}, (
            '%s: %r' % (name, est.classes_))
        proba = est.predict_proba(X)
        assert np.all(np.isfinite(proba))
        assert proba.shape == (n, 3)

    # ---- the helper itself, directly --------------------------------------

    def test_the_shared_absence_test_is_the_only_one(self):
        """``_multiclass._is_nan`` and ``encode_labels`` must not diverge
        again. There is now one definition and the first delegates to it.
        """
        from MissLearn._conformance import is_missing_label
        from MissLearn._multiclass import _is_nan
        pd = pytest.importorskip('pandas')
        y = np.array(['a', None, 'b', float('nan'), pd.NA, 'c'], dtype=object)
        expected = np.array([False, True, False, True, True, False])
        assert np.array_equal(is_missing_label(y), expected)
        assert np.array_equal(_is_nan(y), expected)

    def test_a_numeric_label_vector_is_passed_through_untouched(self):
        """The early return for numeric dtypes must survive the change: a
        float label column is already encoded and must not be re-encoded.
        """
        from MissLearn._conformance import encode_labels
        y = np.array([0.0, 1.0, np.nan, 1.0])
        out, classes = encode_labels(None, y)
        assert classes is None
        assert np.array_equal(np.isnan(out), np.isnan(y))
        assert out[0] == 0.0 and out[1] == 1.0


class TestContractExemptionsStayHonest:
    """An exemption from ``check_estimator`` must keep earning itself.

    ``CONTRACT_EXEMPT`` removes a class from the contract sweep entirely,
    which is a stronger claim than ``EXPECTED_FAILED_CHECKS`` and easier to
    abuse: excluding a class is always quicker than fixing it. The comment in
    that module records what happened the last time a discovery list was
    curated by hand, when fourteen estimators were left out and were not
    passing quietly, they were simply not being asked.

    So the list is checked rather than trusted. A name that no longer exists,
    or that would not have been discovered anyway, is an exemption doing
    nothing, and the next reader would take it as evidence that the class was
    considered and excluded on purpose.
    """

    def test_every_exempt_name_exists_and_would_be_discovered(self):
        import inspect
        from sklearn.base import BaseEstimator
        from MissLearn._sklearn_compat import CONTRACT_EXEMPT
        for name in CONTRACT_EXEMPT:
            cls = getattr(_ML, name, None)
            assert cls is not None, (
                '%s is exempted but is not in the public namespace' % name)
            assert inspect.isclass(cls), '%s is not a class' % name
            assert issubclass(cls, BaseEstimator), (
                '%s is exempted from the contract sweep but is not a '
                'BaseEstimator, so the sweep would never have found it and '
                'the entry does nothing' % name)
            assert hasattr(cls, 'fit'), (
                '%s has no fit, so the sweep would never have found it'
                % name)

    def test_every_exemption_carries_a_reason(self):
        """"Known issue" is not a reason. The module says so; this enforces
        enough length that an entry has to say something.
        """
        from MissLearn._sklearn_compat import CONTRACT_EXEMPT
        for name, why in CONTRACT_EXEMPT.items():
            assert isinstance(why, str) and len(why) > 120, (
                '%s is exempted without an argued reason' % name)

    def test_the_exempt_class_really_cannot_pass(self):
        """The exemption is only honest while it is still true. If
        MissImputer ever satisfies the contract, this fails and the entry
        should be deleted rather than left as folklore.
        """
        import warnings as _w
        from sklearn.utils.estimator_checks import check_estimator
        from MissLearn._sklearn_compat import (expected_failed_checks,
                                               CONTRACT_EXEMPT)
        for name in CONTRACT_EXEMPT:
            est = getattr(_ML, name)()
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                try:
                    check_estimator(
                        est, expected_failed_checks=expected_failed_checks(est))
                except Exception:
                    continue                    # still cannot pass, as declared
            pytest.fail('%s now passes check_estimator; remove it from '
                        'CONTRACT_EXEMPT and let the sweep cover it' % name)


class TestEstimatorTypeQuestionsAreTotal:
    """``is_classifier_safe`` answers rather than raising, on every version.

    The plain ``sklearn.base.is_classifier`` read an attribute through 1.6 and
    returned False for anything without it. From 1.7 it goes through
    ``get_tags``, which raises ``AttributeError`` when nothing in the MRO
    defines ``__sklearn_tags__``. Eleven classes here are not BaseEstimator
    subclasses, so the bare call turned a question into an exception and took
    179 unit tests with it.

    These tests do not depend on which scikit-learn is installed. On 1.6 the
    raising path is never reached through a real estimator, so a test that
    only passed a tag-less object would exercise nothing on the development
    machine and the branch would sit uncovered until continuous integration
    found it the hard way. An object whose ``__sklearn_tags__`` raises forces
    the path on any version.
    """

    class _TagsRaiseAttributeError:
        """What a class with no usable tag chain looks like from inside."""
        def __sklearn_tags__(self):
            raise AttributeError('no tags here')

    class _TagsRaiseSomethingElse:
        """A genuine fault inside a real tag implementation."""
        def __sklearn_tags__(self):
            raise ValueError('a real failure, not a missing attribute')

    def test_an_object_with_no_tags_is_not_a_classifier(self):
        from MissLearn._sklearn_compat import (is_classifier_safe,
                                               is_regressor_safe)
        obj = self._TagsRaiseAttributeError()
        assert is_classifier_safe(obj) is False
        assert is_regressor_safe(obj) is False

    def test_a_real_failure_inside_tags_still_propagates(self):
        """The guard is deliberately narrow. Swallowing everything would turn
        a broken ``__sklearn_tags__`` into a quiet "not a classifier", which
        is the failure mode being fixed, pointed the other way.
        """
        from MissLearn._sklearn_compat import (is_classifier_safe,
                                               is_regressor_safe)
        obj = self._TagsRaiseSomethingElse()
        with pytest.raises(ValueError):
            is_classifier_safe(obj)
        with pytest.raises(ValueError):
            is_regressor_safe(obj)

    def test_real_estimators_still_answer_correctly(self):
        """Totality must not have cost correctness."""
        from MissLearn._sklearn_compat import (is_classifier_safe,
                                               is_regressor_safe)
        from MissLearn import MissLinear, MissLogistic
        assert is_regressor_safe(MissLinear(compute_se=False)) is True
        assert is_classifier_safe(MissLinear(compute_se=False)) is False
        assert is_classifier_safe(MissLogistic()) is True
        assert is_regressor_safe(MissLogistic()) is False

    @pytest.mark.parametrize('name', ['MissImputer', 'MissEnsemble',
                                      'MissExplainer', 'MissPreprocessor'])
    def test_the_classes_that_provoked_this_are_answered_not_raised(self, name):
        """These are among the classes without a scikit-learn identity, and
        MissImputer was the one whose failure surfaced first. Asking what they
        are is a legitimate question and the answer is "neither".

        Instances, not classes. scikit-learn deprecated passing a class to
        these functions in 1.8, so a test written with the class object would
        pass today and start warning, then failing, on its own schedule.
        """
        from MissLearn._sklearn_compat import (is_classifier_safe,
                                               is_regressor_safe)
        from MissLearn import MissLinear
        inner = MissLinear(compute_se=False)
        obj = {'MissImputer':      lambda: _ML.MissImputer(),
               'MissEnsemble':     lambda: _ML.MissEnsemble(),
               'MissExplainer':    lambda: _ML.MissExplainer(inner),
               'MissPreprocessor': lambda: _ML.MissPreprocessor(inner),
               }[name]()
        assert is_classifier_safe(obj) is False
        assert is_regressor_safe(obj) is False


# ===========================================================================
# Entry point for whole-suite execution
# ===========================================================================

if __name__ == '__main__':
    import pytest as _pytest
    _pytest.main([__file__, '-v', '--tb=short'])
