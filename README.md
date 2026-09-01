# MissLearn

[![CI](https://github.com/amaxiom/MissLearn/actions/workflows/ci.yml/badge.svg)](https://github.com/amaxiom/MissLearn/actions/workflows/ci.yml)
[![Docs](https://github.com/amaxiom/MissLearn/actions/workflows/docs.yml/badge.svg)](https://github.com/amaxiom/MissLearn/actions/workflows/docs.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/amaxiom/MissLearn#testing)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/amaxiom/MissLearn)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-orange)](https://scikit-learn.org)

**sklearn-style estimators with native missing-data support via Full Information Maximum Likelihood (FIML).**

> No imputation. No listwise deletion. No fake data.

> **Pre-release.** Version 0.9.2 is complete and tested (2,658 tests, 95%
> coverage) but has not yet been used by anyone outside its author. If you
> are reading this because you were asked to try it, that is what you are
> being asked to try: whether it works on your data, and whether the
> documentation tells you what you need. Please open an issue for anything
> that surprises you, including anything the guides fail to explain. There
> are templates under **Issues**, and `CONTRIBUTING.md` says what makes a
> report easy to act on.

MissLearn models accept `NaN` values directly in `fit` and `predict`. Every observation contributes exactly the information it contains: an observation with three of five predictors observed contributes a three-dimensional likelihood, and no value is ever invented, imputed, or assumed. The aim is simple; on incomplete data, match or beat the conventional strategies of dropping rows, dropping columns, or imputing.

All estimators follow the scikit-learn API (`fit` / `predict` / `score` / `get_params` / `set_params`) and work with sklearn pipelines and cross-validation utilities that do not internally strip NaN values. Pandas DataFrames and Series are accepted transparently.

## Features

### Model families

| Family | Estimators | Task |
|---|---|---|
| `MissLinear` | `MissLinear` | FIML linear regression (joint multivariate normal) |
| `MissLogistic` | `MissLogistic` | FIML logistic regression |
| `MissRidge` | `MissRidgeRegressor`, `MissRidgeClassifier`, `MissRidge` (auto) | Ridge-penalized FIML |
| `MissLASSO` | `MissLASSORegressor`, `MissLASSOClassifier`, `MissLASSO` (auto) | LASSO-penalized FIML |
| `MissNeighbors` | `MissNeighborsRegressor`, `MissNeighborsClassifier`, `MissNeighbors` (auto) | K-nearest neighbours FIML |
| `MissBayes` | `MissBayesRegressor`, `MissBayesClassifier`, `MissBayes` (auto) | Naive Bayes FIML |
| `MissSupport` | `MissSupportRegressor`, `MissSupportClassifier`, `MissSupport` (auto) | Support vector FIML |
| `MissGaussian` | `MissGaussianRegressor`, `MissGaussianClassifier`, `MissGaussian` (auto) | Gaussian Process (marginalized kernel, exact Bayesian intervals). O(n^3): see the cost note under [Benchmarks](#benchmarks) before using it on more than a few thousand rows |
| `MissMixed` | `MissMixedRegressor`, `MissMixedClassifier`, `MissMixed` (auto) | Random-intercept LME / GLMM for grouped and longitudinal data |
| `MissEnsemble` | `MissEnsemble` | Bootstrap-aggregated ensemble of MissLearn models, homogeneous or heterogeneous with weights and OOB scores |
| `MissMulticlass` | `MissMulticlass` | One-vs-Rest multi-class wrapper for any binary MissLearn classifier |
| `MissPreprocessor` | `MissPreprocessor`, `prefit_check` | Validation, FIML compatibility checking, categorical encoding with NaN preservation |

All models accept `copula=True` (or `'auto'`) to apply a marginal Gaussian copula transform for skewed or heavy-tailed features. The transform assumes continuous margins, so columns with fewer than three distinct observed values are passed through untouched rather than mapped, and `'auto'` decides from the margins alone. It is off by default: applying it to one arm of a comparison and not the others measures the transform rather than the missing-data treatment.

### Tools

| Tool | Purpose |
|---|---|
| `MissImputer` | Multiple imputation by draws from the FIML-estimated joint MVN; Rubin's rules combiner for downstream models that cannot accept NaN |
| `MissDiagnostic` | Missingness mechanism assessment: Little's MCAR test, MAR plausibility, pattern summary, missingness correlations |
| `MissRecommender` | Evidence-based model triage: ranks the families for a given incomplete dataset with the reasoning attached, flags columns to drop rather than impute, and names the required follow-up analyses |
| `MissExplainer` | SHAP explainability using the FIML model as the exact coalition value function; value SHAP and missingness SHAP (exact 2^p for p<=15, KernelSHAP above) |
| `MissSensitivity` | MNAR sensitivity analysis via delta-adjustment with tipping-point deltas |
| CV utilities | `MissKFold`, `MissStratifiedKFold`, `miss_cross_val_score`, `miss_cross_validate`: NaN-safe splitters and scorers with full NaN-in-y support |

## Installation

Requires Python 3.9 or later.

```bash
pip install git+https://github.com/amaxiom/MissLearn.git
```

To work from a clone, which is what you want if you plan to run the examples
or the test suite:

```bash
git clone https://github.com/amaxiom/MissLearn.git
cd MissLearn
pip install -e ".[all]"
```

`[all]` adds pandas, matplotlib and the gradient-boosted tree learners that
can be used as `MissEnsemble` members. The package itself needs only numpy,
scipy and scikit-learn; pandas support is duck-typed rather than a
dependency, so DataFrames work if pandas is present and nothing breaks if it
is not. Add `[test]` for pytest and hypothesis if you want to run the suites.

## Quick start

```python
import numpy as np
from MissLearn import MissLinear

X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [1.5, 2.5]])
y = np.array([0.5, 1.2, np.nan, 0.9])          # NaN in y is fine too

model = MissLinear().fit(X, y)                   # no imputation, no dropped rows
y_pred = model.predict(X)                        # NaN-in-X handled natively
lo, hi = model.predict_interval(X, alpha=0.05)   # 95% intervals; wider when more
model.summary()                                  # features are missing
```

## Benchmarks

Headline results (July 2026): synthetic data, 25% MAR missingness, 5-fold CV, against Drop Rows / Drop Columns / Mean imputation / KNN imputation / MICE baselines.

**Classification accuracy (medium dataset)**: MissLearn vs best baseline:

| Model | MissLearn | Best baseline |
|---|---|---|
| MissBayes | **0.939** | 0.889 |
| MissSupport | **0.940** | 0.927 |
| MissNeighbors | **0.933** | 0.921 |
| Linear family (Logistic/Ridge/LASSO) | **0.914** | 0.911 |

**Regression R²:** linear family at parity (~0.70); MissNeighbors 0.52 vs 0.50; MissSupport **0.64 vs 0.17**.

**Fast:** classifier fits in ~0.2 s and the LASSO regressor in ~1.3 s at n=600, p=8. The correctness suites are 2,658 tests (1,600 unit, 1,042 conformance, 16 property) at **95.0% coverage**, and run in about 25 minutes on an idle machine, considerably longer under load. Full per-family results, sweeps and plots are in [`benchmarks/`](benchmarks/), described in [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

**The exception is `MissGaussian`.** Exact Gaussian-process inference is
O(n^3) in the number of rows, and this family is the one place where that
cost is visible rather than theoretical. Measured at n = 90, p = 3 on an idle
machine, a single fit takes about **1.2 s** for `MissGaussianRegressor` and
about **11 s** for `MissGaussianClassifier`. The classifier is roughly nine
times the regressor because it runs a Laplace mode-finder of up to
`max_iter_newton` Newton steps inside *every* objective evaluation, and the
optimiser repeats that for each of `1 + n_restarts` restarts. Cubed in n and
multiplied by four nested loops, a few thousand rows will look like a hang.

Nothing in the estimator itself caps the row count. `MissRecommender`
will veto the family above `gp_max_n` (default 1,000) and say why, and the
benchmark harness applies its own `max_n = 900`, but neither of those is
consulted when you call `MissGaussian` directly, so on your own data the
ceiling is yours to set. If a
fit is taking longer than you expected, in rough order of how much they buy:
lower `n_restarts` (the default of 3 means four optimiser runs), subsample
the rows, lower `max_iter_newton` on the classifier, or use another family.
`MissSupport` and `MissNeighbors` cover much of the same ground with a
kernel or a distance instead of a full posterior.

## Documentation

| Document | Contents |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Practical guide to every model and tool |
| [`docs/COMPUTATIONAL_GUIDE.md`](docs/COMPUTATIONAL_GUIDE.md) | Complexity, scaling, and performance guidance |
| [`docs/INTERPRETATION_GUIDE.md`](docs/INTERPRETATION_GUIDE.md) | Interpreting coefficients, intervals, and SHAP output |
| [`docs/MissLearn_User_Guide.pdf`](docs/MissLearn_User_Guide.pdf) | Combined user guide (PDF) |
| [`docs/METHODS_GUIDE.md`](docs/METHODS_GUIDE.md) | Statistical methodology: missing-data theory, FIML derivations, model design, inference |
| [`examples/EXAMPLES.md`](examples/EXAMPLES.md) | Guide to the worked example notebooks |
| [`tests/TEST.md`](tests/TEST.md) | Guide to the unit, performance, and benchmark test suites |
| [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) | Guide to the synthetic benchmarks, and what makes each comparison fair |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup, style, and the deprecation policy |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Contributor Covenant 2.1, and how to report a concern |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Planned work |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `scikit-learn`
- Optional: `pandas` (DataFrame support is duck-typed; no hard dependency), `matplotlib` (plots)

## Repository layout

```
MissLearn/
├── MissLearn/        # the package (models, tools, utilities); single source of truth
├── benchmarks/       # per-family benchmark and sweep notebooks + generators
├── docs/             # methodology, user guides
├── examples/         # worked example notebooks (see examples/EXAMPLES.md)
├── tests/            # unit / conformance / performance suites (see tests/TEST.md)
├── pyproject.toml    # packaging, builds directly from MissLearn/
├── CHANGELOG.md
└── LICENSE
```

## Citation

If you use MissLearn in published work, please cite the software. There is no
paper yet; this section will be updated when there is one.

​```bibtex
@software{barnard2026misslearn,
  author  = {Barnard, Amanda S.},
  title   = {{MissLearn}: full-information machine learning from data with
             meaningful missingness},
  year    = {2026},
  version = {0.9.2},
  url     = {https://github.com/amaxiom/MissLearn},
  note    = {Python package, MIT licence}
}
​```

Or in text:

> Barnard, A. S. (2026). *MissLearn: full-information machine learning from
> data with meaningful missingness* (Version 0.9.2) [Computer software].
> https://github.com/amaxiom/MissLearn

## License

MIT. See [`LICENSE`](LICENSE).
