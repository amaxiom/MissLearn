"""
MissLearn
======
Full Information Maximum Likelihood (FIML) regression models with
native missing data support.

No imputation. No listwise deletion. No fake data.
Each observation contributes exactly the information it contains.

Models
------
MissLinear          : FIML linear regression   (joint multivariate normal)
MissLogistic        : FIML logistic regression  (logistic + multivariate normal X)
MissRidgeRegressor  : Ridge-penalized FIML linear regression (conditional model)
MissRidgeClassifier : Ridge-penalized FIML logistic regression
MissRidge           : Auto-selecting wrapper (regression vs. classification based on y)
MissGaussianRegressor     : Gaussian Process regression (marginalized kernel, exact Bayesian intervals)
MissGaussianClassifier    : Gaussian Process classification (Laplace approximation)
MissGaussian              : Auto-selecting GP wrapper
MissMixedRegressor        : FIML random-intercept LME for repeated-measures / longitudinal data
MissMixedClassifier       : FIML random-intercept GLMM (logistic link, GH quadrature)
MissMixed                 : Auto-selecting mixed-effects wrapper
MissEnsemble              : Bootstrap-aggregated ensemble, homogeneous or heterogeneous
MissMulticlass            : One-vs-Rest multi-class extension for binary MissLearn classifiers

Utilities
---------
MissPreprocessor          : Validates and encodes data before fitting any MissLearn model;
                            handles categorical columns (one-hot encoding with NaN preservation),
                            detects compatibility issues, and wraps the fitted model transparently

MissDiagnostic            : Missing data mechanism assessment (Little's MCAR test, MAR
                            plausibility via logistic regression, pattern summary, missingness
                            correlations, descriptive comparison by missingness)

prefit_check              : Standalone compatibility check function
MissRecommender           : Evidence-based triage that ranks the model families for a given
                            incomplete dataset, reports the reasoning, flags columns that should
                            be dropped rather than modelled, and names the required follow-up
                            analyses; recommend_model() is the one-call wrapper

MissImputer               : Multiple imputation by draws from the FIML-estimated joint MVN;
                            includes Rubin's rules combiner and fit_transform_combine convenience

MissSensitivity           : MNAR sensitivity analysis via delta-adjustment; sweeps a grid of
                            plausible MNAR departures and reports tipping-point deltas

MissExplainer             : SHAP-based explainability using the FIML model as the exact coalition
                            value function; computes value SHAP and missingness SHAP with viridis
                            plots; wraps results in shap.Explanation for SHAP library compatibility

MissShapley                  : Low-level Shapley engine (exact 2^p for p<=15, KernelSHAP above)
MissKFold                 : K-fold cross-validation splitter (NaN-safe, no check_array)
MissStratifiedKFold       : Stratified K-fold splitter (stratifies on observed y; distributes
                            NaN-y rows uniformly across folds)

miss_cross_val_score      : Cross-validated scoring with full NaN-in-y support
miss_cross_validate       : Multi-metric cross-validation returning a results dict

All models accept copula=True to apply a marginal Gaussian copula transform
before fitting.  This relaxes the joint normality assumption by mapping each
continuous feature (and the response, for regression models) to the normal
scale via its empirical CDF.  Predictions are automatically inverse-transformed
back to the original scale.  Use this option when predictors are skewed,
heavy-tailed, or otherwise clearly non-normal.

Pandas support
--------------
All models accept pandas DataFrames and Series directly.  Column names from
DataFrames are stored as feature_names_in_ on the fitted model.  No hard
dependency on pandas is required -- detection is duck-typed.

Usage
-----
    from MissLearn import MissLinear, MissLogistic, MissRidge
    import numpy as np

    X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
    y = np.array([0.5, 1.2, np.nan])

    model = MissLinear()
    model.fit(X, y)
    model.summary()

    # Auto-select regression or classification based on y
    model = MissRidge(alpha=1.0, copula='auto')
    model.fit(X, y)

    # Multi-class classification
    from MissLearn import MissMulticlass, MissRidgeClassifier
    clf = MissMulticlass(MissRidgeClassifier(alpha=1.0))
    clf.fit(X_train, y_multiclass)
    proba = clf.predict_proba(X_test)   # shape (n, K)

    # Cross-validation
    from MissLearn import miss_cross_val_score, MissStratifiedKFold
    scores = miss_cross_val_score(model, X, y, cv=5, scoring='r2')

All models follow the scikit-learn API (fit / predict / score /
get_params / set_params) and are compatible with sklearn pipelines
and cross-validation utilities that do not internally strip NaN values.

Memory
------
Gauss-Hermite quadrature nodes are cached at module level across all model
instances for the process lifetime.  Call ``MissLearn.clear_cache()`` to
release that memory if needed.

Reproducibility
---------------
Results are fully deterministic on a given machine: there is no randomness
in the optimization (L-BFGS-B) or the quadrature nodes.

Across different machines or NumPy builds the optimizer will converge to the
same solution, but the final parameter values may differ in the last few
significant digits.  The source of variation is floating-point non-associativity
in BLAS/LAPACK routines (Cholesky, triangular solves): Intel MKL, OpenBLAS,
Apple Accelerate, and AMD BLIS all implement these operations with different
internal summation orders, producing rounding differences that compound through
the gradient.  Typical cross-platform spread is on the order of 1e-6 to 1e-8
in coefficients and standard errors.  Information criteria and p-values are
unaffected at any scientifically meaningful precision.
"""

from ._linear import MissLinear
from ._logistic import MissLogistic
from ._ridge import MissRidgeRegressor, MissRidgeClassifier, MissRidge
from ._lasso import MissLASSORegressor, MissLASSOClassifier, MissLASSO
from ._knn import MissNeighborsRegressor, MissNeighborsClassifier, MissNeighbors
from ._bayes import MissBayesRegressor, MissBayesClassifier, MissBayes
from ._svm import MissSupportRegressor, MissSupportClassifier, MissSupport
from ._gp import MissGaussianRegressor, MissGaussianClassifier, MissGaussian
from ._mixed import MissMixedRegressor, MissMixedClassifier, MissMixed
from ._ensemble import MissEnsemble
from ._validate import MissPreprocessor, prefit_check
from ._diagnostic import MissDiagnostic
from ._recommend import MissRecommender, recommend_model
from ._multiclass import MissMulticlass
from ._crossval import MissKFold, MissStratifiedKFold, miss_cross_val_score, miss_cross_validate
from ._imputer import MissImputer
from ._sensitivity import MissSensitivity
from ._explainer import MissExplainer, MissShapley
from ._utils import clear_cache

# ---------------------------------------------------------------------------
# Pandas DataFrame / Series support
# Patch every exported model class at import time so that DataFrames and
# Series are accepted transparently in fit / predict / score / etc.
# ---------------------------------------------------------------------------
from ._estimator_checks import (check_missing_data_estimator,
                                MissingDataReport)
from ._pandas_compat import enable_dataframe_support as _enable_df

_MODEL_CLASSES = [
    MissLinear, MissLogistic,
    MissRidgeRegressor, MissRidgeClassifier, MissRidge,
    MissLASSORegressor, MissLASSOClassifier, MissLASSO,
    MissNeighborsRegressor, MissNeighborsClassifier, MissNeighbors,
    MissBayesRegressor, MissBayesClassifier, MissBayes,
    MissSupportRegressor, MissSupportClassifier, MissSupport,
    MissGaussianRegressor, MissGaussianClassifier, MissGaussian,
    MissMixedRegressor, MissMixedClassifier, MissMixed,
    MissEnsemble,
    MissPreprocessor,
    MissMulticlass,
    MissImputer,
    # MissSensitivity is a utility class that operates on a pre-fitted model
    # rather than accepting X directly via fit/predict, so it is excluded.
]
for _cls in _MODEL_CLASSES:
    _enable_df(_cls)
del _cls, _enable_df, _MODEL_CLASSES

__all__ = [
    'check_missing_data_estimator', 'MissingDataReport',
    'MissLinear', 'MissLogistic',
    'MissRidgeRegressor', 'MissRidgeClassifier', 'MissRidge',
    'MissLASSORegressor', 'MissLASSOClassifier', 'MissLASSO',
    'MissNeighborsRegressor', 'MissNeighborsClassifier', 'MissNeighbors',
    'MissBayesRegressor', 'MissBayesClassifier', 'MissBayes',
    'MissSupportRegressor', 'MissSupportClassifier', 'MissSupport',
    'MissGaussianRegressor', 'MissGaussianClassifier', 'MissGaussian',
    'MissMixedRegressor', 'MissMixedClassifier', 'MissMixed',
    'MissEnsemble',
    'MissPreprocessor', 'prefit_check',
    'MissDiagnostic',
    'MissRecommender', 'recommend_model',
    'MissMulticlass',
    'MissKFold', 'MissStratifiedKFold',
    'miss_cross_val_score', 'miss_cross_validate',
    'MissImputer',
    'MissSensitivity',
    'MissExplainer', 'MissShapley',
    'clear_cache',
]
__version__ = '0.9.2'
