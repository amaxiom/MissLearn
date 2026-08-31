# -*- coding: utf-8 -*-
"""One importable definition of every benchmarked model family.

Each entry pairs a MissLearn estimator with its natural scikit-learn
counterpart, so the benchmark varies only the missing-data strategy and never
the model class. The penalized families tune their regularisation strength by
inner cross-validation on BOTH arms, because a fixed alpha is not comparable
between two objectives that normalise the loss differently.

Used by the explorer notebooks and by the exported scripts, so all of them
describe the same experiment.
"""
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  Ridge, Lasso, BayesianRidge)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.gaussian_process import (GaussianProcessRegressor,
                                      GaussianProcessClassifier)
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              WhiteKernel)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import MissLearn as ML
from benchmark_core import tuned_sklearn, TunedMiss, ALPHA_GRID

SEED = 42


def _tuned_logistic(penalty, solver):
    """Logistic baseline with C chosen by inner CV, matching the tuning budget
    the penalized MissLearn arm receives."""
    def make():
        return GridSearchCV(
            LogisticRegression(penalty=penalty, solver=solver, max_iter=2000,
                               random_state=SEED),
            {"C": [1.0 / a for a in ALPHA_GRID]},
            cv=StratifiedKFold(3, shuffle=True, random_state=0),
            scoring="accuracy")
    return make


FAMILIES = {
    "MissLinear": dict(
        label="Linear / logistic regression",
        blurb="Exact FIML linear and logistic regression. The reference "
              "family: on linear ground truth it should sit at the efficiency "
              "bound, so parity with good imputation is the expected result.",
        reg_sklearn=lambda: LinearRegression(),
        reg_fiml=lambda: ML.MissLinear(compute_se=False),
        reg_name="MissLinear",
        clf_sklearn=lambda: LogisticRegression(max_iter=2000,
                                               random_state=SEED),
        clf_fiml=lambda: ML.MissLogistic(compute_se=False),
        clf_name="MissLogistic",
        has_sweep=True,
    ),
    "MissRidge": dict(
        label="Ridge (L2 penalized)",
        blurb="L2-penalized FIML. Both arms tune alpha by inner CV, so the "
              "comparison is about the missing-data treatment and not about "
              "who got the kinder penalty.",
        reg_sklearn=tuned_sklearn(lambda: Ridge(random_state=SEED),
                                  "regression"),
        reg_fiml=lambda: TunedMiss(lambda a: ML.MissRidgeRegressor(alpha=a),
                                   task="regression"),
        reg_name="MissRidgeRegressor",
        clf_sklearn=_tuned_logistic("l2", "lbfgs"),
        clf_fiml=lambda: TunedMiss(
            lambda a: ML.MissRidgeClassifier(alpha=a, compute_se=False),
            task="classification"),
        clf_name="MissRidgeClassifier",
        has_sweep=True,
    ),
    "MissLASSO": dict(
        label="LASSO (L1 penalized)",
        blurb="L1-penalized FIML with automatic variable selection. Both arms "
              "tune alpha by inner CV.",
        reg_sklearn=tuned_sklearn(lambda: Lasso(max_iter=5000,
                                                random_state=SEED),
                                  "regression"),
        reg_fiml=lambda: TunedMiss(lambda a: ML.MissLASSORegressor(alpha=a),
                                   task="regression"),
        reg_name="MissLASSORegressor",
        clf_sklearn=_tuned_logistic("l1", "liblinear"),
        clf_fiml=lambda: TunedMiss(
            lambda a: ML.MissLASSOClassifier(alpha=a, compute_se=False),
            task="classification"),
        clf_name="MissLASSOClassifier",
        has_sweep=True,
    ),
    "MissBayes": dict(
        label="Generative Gaussian",
        blurb="Full-covariance generative model: marginalises the "
              "class-conditional densities exactly. Its scikit-learn "
              "counterpart is scale-invariant, so any advantage here cannot "
              "be a preprocessing artefact. Usually the strongest family.",
        reg_sklearn=lambda: BayesianRidge(),
        reg_fiml=lambda: ML.MissBayesRegressor(),
        reg_name="MissBayesRegressor",
        clf_sklearn=lambda: GaussianNB(),
        clf_fiml=lambda: ML.MissBayesClassifier(),
        clf_name="MissBayesClassifier",
        has_sweep=True,
    ),
    "MissNeighbors": dict(
        label="k-nearest neighbours (expected distance)",
        blurb="Expected squared distances under the fitted joint Gaussian, "
              "rather than distances between imputed points.",
        reg_sklearn=lambda: KNeighborsRegressor(n_neighbors=5),
        reg_fiml=lambda: ML.MissNeighborsRegressor(n_neighbors=5),
        reg_name="MissNeighborsRegressor",
        clf_sklearn=lambda: KNeighborsClassifier(n_neighbors=5),
        clf_fiml=lambda: ML.MissNeighborsClassifier(n_neighbors=5),
        clf_name="MissNeighborsClassifier",
        has_sweep=True,
    ),
    "MissSupport": dict(
        label="Support vector (expected kernel)",
        blurb="Expected kernel evaluations, PSD-safe by an augmented-space "
              "embedding. Note both arms are standardised: without that the "
              "comparison measures preprocessing, not missingness.",
        reg_sklearn=lambda: SVR(C=1.0, kernel="rbf"),
        reg_fiml=lambda: ML.MissSupportRegressor(C=1.0, kernel="rbf"),
        reg_name="MissSupportRegressor",
        clf_sklearn=lambda: SVC(C=1.0, kernel="rbf", probability=True,
                                random_state=SEED),
        clf_fiml=lambda: ML.MissSupportClassifier(C=1.0, kernel="rbf"),
        clf_name="MissSupportClassifier",
        has_sweep=True,
    ),
    "MissGaussian": dict(
        label="Gaussian process (marginalised kernel)",
        blurb="Exact GP inference with a marginalised kernel and Bayesian "
              "predictive intervals. Capped at max_n rows because exact "
              "inference is O(n^3).",
        # MATCHED NOISE. MissGaussianRegressor fits a noise variance as a
        # hyperparameter (noise_var_, the log_sn parameter). The default
        # sklearn configuration uses alpha=1e-10 and a bare RBF, so it has no
        # noise floor and must interpolate noisy data exactly; on an incomplete
        # sample the imputed rows become near-duplicate inputs with different
        # targets, the kernel matrix goes near-singular, and the predictions
        # explode. Measured at n=300 that gave R2 = -3.26 for the baseline
        # against 0.63 for FIML, which reads as a large FIML win but is
        # entirely a misconfigured comparator: adding a WhiteKernel takes the
        # same baseline to parity with FIML. Both arms must be allowed to fit
        # noise or the comparison measures the noise floor rather than the
        # missing-data treatment.
        #
        # The kernel below is deliberately the sklearn defaults, matching the
        # published script data/rerun_bench.py exactly. Tightening the bounds
        # is tempting and was tried; it is actively harmful. Starting the noise
        # level low, at 1e-2 with a lower bound of 1e-8, sends the optimiser
        # into a degenerate all-noise optimum on three of five folds of
        # Regression-Small: the fitted noise settles at 0.999 against a unit
        # response variance and the model predicts the mean, giving R2 values
        # of -0.002, -0.004 and -0.008 and a fold mean of 0.2650 +/- 0.3311.
        # Starting from the default of 1.0 the same folds settle at a noise
        # level near 0.30 and give 0.6871 +/- 0.0430, reproducing the published
        # number 0.6871310 exactly. Do not narrow these bounds.
        reg_sklearn=lambda: GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(),
            normalize_y=True, n_restarts_optimizer=1, random_state=SEED),
        reg_fiml=lambda: ML.MissGaussianRegressor(n_restarts=1),
        reg_name="MissGaussianRegressor",
        # The classifier uses a Bernoulli likelihood rather than an additive
        # noise term, so it cannot diverge the same way, but it is given the
        # matching default kernel so the two GP arms are configured alike.
        # n_restarts_optimizer=1 on both GP arms matches MissGaussianRegressor's
        # n_restarts=1, so neither side gets more optimiser effort than the
        # other. It does not change the result: 0 and 1 restarts agree to four
        # decimal places here.
        clf_sklearn=lambda: GaussianProcessClassifier(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=1, random_state=SEED),
        clf_fiml=lambda: ML.MissGaussianClassifier(),
        clf_name="MissGaussianClassifier",
        has_sweep=True,
        # Exact inference is O(n^3), so rather than dropping to the single
        # smallest task this family now runs both tasks with the rows capped.
        # n=300 was thin enough that the fit was unstable; 900 is roughly a
        # 27-fold cost increase over the small task and still tractable.
        max_n=900,
    ),
}

SWEEP_FAMILIES = [k for k, v in FAMILIES.items() if v.get("has_sweep")]


def describe():
    """Print the registry, for the explorer notebooks."""
    print("%-15s %-38s %s" % ("KEY", "MODEL CLASS", "SWEEP?"))
    print("-" * 70)
    for k, v in FAMILIES.items():
        print("%-15s %-38s %s" % (k, v["label"],
                                  "yes" if v.get("has_sweep") else "no"))
