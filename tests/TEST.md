# MissLearn Test Suites

Five suites live in `tests/`, and CI runs the whole directory. Three of them have a matching notebook (`Unit_Test_Suite.ipynb`, `Performance_Test_Suite.ipynb`, `Benchmark_Test_Suite.ipynb`) that runs the same code one section per cell, with plots where applicable.

| Suite | File | What it answers | Duration |
|---|---|---|---|
| Unit | `unit_test_suite.py` | Is every component correct? | ~3 min |
| Conformance | `conformance_test_suite.py` | Do estimators meant to be interchangeable actually behave alike? | ~18 min |
| Property | `property_test_suite.py` | Do the mathematical invariants hold on data nobody chose by hand? | ~40 s |
| Performance | `performance_test_suite.py` | How does fit/predict time scale? | minutes-tens of minutes depending on benchmarks run |
| Benchmark | `benchmark_test_suite.py` | Which model family is best for which scenario, on real data? | minutes per tier |

The first three are the correctness suites and all three must be green: 2658 tests, 1600 unit, 1042 conformance and 16 property, of which 2600 pass, 57 skip and 1 is a declared xfail. Coverage is **95.0%** of statement-and-branch units. The skips are conditional on an optional package or on an estimator not offering the method under test; none is a disabled check. Timing is the least reliable number here: on a quiet machine the three suites take about 25 minutes under `--cov-branch`, and the same work has been measured at 87 and 104 minutes on runs where other Python processes were competing for the cores. Check what else is running before concluding anything about the code. That figure is worth stating carefully, because it collapses under load: the same suite took 74 and then 128 minutes on runs where other Python processes were competing for the cores, which is a four-fold spread on identical work. If a run feels slow, check what else is running before concluding anything about the code. The last two suites are measurement rather than pass/fail and CI does not gate on them.

Nearly all of that time is `MissGaussianClassifier`, which is O(n^3) and is fitted once per regime by each conformance section. Every one of the eight slowest tests in the suite is a Gaussian Process cell. If the wall time becomes a problem, the lever is that estimator's `n` in the registry rather than dropping checks.

**Run everything the way CI does:**

```bash
python -m pytest tests/ -q --cov=MissLearn --cov-branch --cov-report=term-missing
```

Branch coverage rather than line coverage, deliberately: every defect this package has had lived in a branch that was never taken, not in a line that was never reached.

## Unit tests: `unit_test_suite.py`

1600 pytest tests in 133 classes. The original set is one class per package component; the later ones are grouped by the behaviour they pin rather than by module, and several discover their targets from the package so that a new estimator is covered the day it is added rather than the day somebody remembers to edit a list: numerical utilities (`TestNumericalUtils`), each model family (`TestMissLinear`, `TestMissLogistic`, `TestMissRidgeRegressor` / `TestMissRidgeClassifier` / `TestMissRidge`, `TestMissLASSO`, `TestMissNeighbors`, `TestMissBayes`, `TestMissSupport`, `TestMissGaussian`, `TestMissMixed`, `TestMissEnsemble`, `TestMissMulticlass`), and every tool (`TestMissPreprocessor`, `TestMissDiagnostic`, `TestMissKFold`, `TestMissStratifiedKFold`, `TestCrossVal`, `TestMissImputer`, `TestMissSensitivity`, `TestMissExplainer`, `TestPandasSupport`, `TestCopulaTransform`).

A further group covers numerical behaviour that no estimator test would
catch, because the symptom is a slightly worse number rather than a
failure: `TestCopulaTies`, `TestCopulaDiscreteColumns`,
`TestCopulaEmptyColumns`, `TestCopulaSkewIsScaleFree` and
`TestCopulaDecisionMatchesAction` for the copula transform;
`TestLogisticNormalIntegral`, `TestFitUsesTheSameIntegralAsPredict` and
`TestAdaptiveGaussHermite` for the marginalisation quadrature; and
`TestStandardErrorsFailToNaN` for the direction a failed variance
computation reports in. Each of these was written against a defect that
the estimator suites had been green through.

Parametrisation makes the collected count higher than the number of functions. Shared regression, binary, multi-class, and grouped datasets are generated once per pytest session with a fixed seed.

Most classes added since July 2026 are named after the defect they pin rather than after a component, and their docstrings record what the defect was and how it was measured. `TestLassoClassifierSeedIsDeterministic` and `TestBayesEffectSizeOnDegenerateColumns` are typical: read the docstring before changing the assertion, because the specific number in it is usually the point.

**Run the whole suite (~3 min):**

```powershell
cd "C:\Users\Amanda\Favorites\Machine Learning\MissLearn\tests"
python -m pytest unit_test_suite.py -q
```

**Run a single class:**

```powershell
python -m pytest unit_test_suite.py::TestMissRidge -v --tb=short
```

`Unit_Test_Suite.ipynb` runs one class per cell via `pytest.main(['-v', '--tb=short', 'unit_test_suite.py::TestMissRidge'])`.

## Conformance tests: `conformance_test_suite.py`

The cross-estimator suite. It exists because of a specific reported defect: `MissLASSOClassifier` raised on high missingness while `MissLogistic` and `MissLASSORegressor` degraded gracefully, since a fallback present in both siblings had never been applied to that one class. Per-class discipline cannot catch that, because nothing compares the classes with each other. Every estimator is driven through every degenerate regime, and a regime one estimator survives is expected of all its siblings.

16 estimators x 14 regimes, plus the invariant axes below. Regimes are the conditions data actually arrives in, not adversarial constructions: an all-NaN column is a sensor that failed for a whole campaign, `wide_p_gt_n` is a pilot study, `imbalanced_5pct` is a rare outcome.

| Section | Asks |
|---|---|
| 1. Degenerate regimes | Does it fit, or refuse clearly? Silent NaN is never acceptable |
| 2. `predict_proba` | Finite and summing to 1, checked separately because four classifiers once returned finite labels over all-NaN probabilities |
| 3. sklearn contract | `get_params` / `clone` / `set_params` round-trip |
| 4. Determinism | The same data twice, and the same rows in a different order |
| 5. Feature names | Column names reach `feature_names_in_`, so reports name the measurement |
| 6. Sibling divergence | The guarantee itself: any divergence must be declared with a reason |
| 7. Invariant axes | The four techniques below |

### The invariant axes (section 7)

Sections 1 to 6 each grew in response to a defect that had already shipped. Section 7 is the four techniques that found the August 2026 defects, written down so they run on every commit instead of when somebody thinks to probe by hand. Each caught something real:

| Axis | Found |
|---|---|
| **Options** | Four options accepted a misspelling and silently selected the other branch, `copula` on all sixteen estimators among them. `structure` tested only for `'full'`, so any misspelling of it chose naive independence, the opposite model |
| **Accessors** | `MissBayes` returned exactly 1/p for every feature when one effect size came out nan, which sums to 1 and reads as a result. `MissLASSO` printed four columns of literal nan under headings saying p_value and CI |
| **Numerics** | An underflow crashed `MissMixed` and a nan reached the `MissLASSO` gradient, both at boundaries no test visited. Each announced itself as a `RuntimeWarning` first, which nothing was watching |
| **Accuracy** | Nothing anywhere graded whether a fit predicted better than the mean, so a divergence to r-squared -6.7e6 passed every other check, because a huge number is still finite and correctly shaped |

They share one fit per (estimator, regime) through `_fitted`, so adding an axis costs assertions rather than optimisations. The axes are scoped to five regimes for the same reason `ORDER_REGIMES` is scoped: the control, because an estimator that misbehaves on ordinary data misbehaves everywhere, plus the four regimes where the defects actually lived.

`test_every_string_option_is_in_the_matrix` is what makes the option axis systematic rather than a snapshot. It introspects each constructor and fails if a string or bool parameter is neither enumerated in `options_of` nor exempted in `UNCHECKED_OPTIONS` with a reason. It caught `warm_start` on the first run.

### Reading a failure

Three tables record known gaps, and every entry is a work item rather than a permanent exemption. When one is fixed, its test reports the entry is stale and the entry should be deleted.

| Table | Holds |
|---|---|
| `KNOWN_FAILURES` | Regimes an estimator legitimately cannot fit, with the reason |
| `KNOWN_NUMERICAL_WARNINGS` | Cells that warn, with the measured consequence |
| `KNOWN_ACCURACY_FAILURES` | Cells below the predictive floor, with the diagnosis |

A new failure usually means one of two things: a genuine divergence, where this estimator lacks a guard its siblings have, in which case fix the estimator rather than relaxing the test; or a regime legitimately impossible for the family, in which case it should raise a clear error and the expectation belongs in the relevant table with a reason.

```powershell
python -m pytest conformance_test_suite.py -q
python -m pytest conformance_test_suite.py -q -k "misspelled or accuracy"
```

The same checks ship as `MissLearn.check_missing_data_estimator`, so a user can grade their own estimator, or a scikit-learn one, on the same axes.

## Property tests: `property_test_suite.py`

Hypothesis generates the data instead of the author choosing it, which is how the degenerate-column family was found after it had survived 398 hand-written tests. Five property functions, each stating an invariant that must hold whatever the input, driven by an `incomplete_matrix()` strategy that builds missingness the way real data has it:

| Property | Invariant |
|---|---|
| `test_regressor_predicts_finite_or_refuses` | Either a usable prediction or a refusal, never silent nonsense |
| `test_classifier_probabilities_are_probabilities` | In [0, 1] and summing to 1 |
| `test_row_order_does_not_change_predictions` | Permuting the rows permutes the predictions and nothing more |
| `test_prediction_is_subset_invariant` | Predicting half the rows agrees with predicting all and slicing |
| `test_fit_is_reproducible` | The same data gives the same answer twice |

These are the checks worth writing when a bug class is suspected but no specific case is known. A failure comes with the minimal example Hypothesis shrank it to, which is usually the whole diagnosis.

```powershell
python -m pytest property_test_suite.py -q
```

## Performance tests: `performance_test_suite.py`

Scaling and timing benchmarks, designed to be imported and called one function per cell (see `Performance_Test_Suite.ipynb`). All functions accept keyword overrides, return a result dict/list, and accept `plot=True` for matplotlib figures. Two-variant models are reported with a consistent `(reg)` / `(clf)` suffix.

| Function | Measures |
|---|---|
| `bench_model_comparison()` | Fit / predict / score across all model families |
| `bench_n_scaling()` | Fit time vs sample size n |
| `bench_p_scaling()` | Fit time vs number of features p |
| `bench_missingness_impact()` | Fit time vs missing rate |
| `bench_ensemble_scaling()` | `MissEnsemble` vs `n_estimators` |
| `bench_shapley_scaling()` | `MissShapley` exact (2^p) → KernelSHAP transition |
| `bench_imputer_scaling()` | `MissImputer` vs number of imputations m |
| `bench_crossval_scaling()` | `miss_cross_val_score` fold overhead |
| `bench_gaussian_process()` | Dedicated `MissGaussian` O(n³) curve at small n |
| `bench_mixed_effects()` | `MissMixed` vs number of groups |
| `bench_sensitivity()` | `MissSensitivity` delta-grid scaling |

`MissGaussian` is O(n³): it is capped at `GP_PERF_CAP = 150` in n-scaling and omitted from the p and missingness sweeps (it has its own `bench_gaussian_process()`).

Typical single-model timings for context: a classifier fits in ~0.2 s and the LASSO regressor in ~1.3 s at n=600, p=8.

```powershell
cd "C:\Users\Amanda\Favorites\Machine Learning\MissLearn\tests"
python -c "import performance_test_suite as p; print(p.bench_model_comparison())"
```

## Benchmark tests: `benchmark_test_suite.py`

Model-family comparison on **real datasets**, organised in four tiers. This suite complements (does not replace) the synthetic strategy-comparison notebooks in `benchmarks/`: those ask "does FIML preserve performance vs Drop Rows / Drop Cols / Mean / KNN / MICE on controlled synthetic data, one algorithm at a time"; this suite asks "which MissLearn family is best for this scenario, and does any family struggle on non-synthetic conditions".

| Tier | Function(s) | Dataset | Scenario |
|---|---|---|---|
| 1; synthetic anchors | `bench_energy_efficiency()`, `bench_wisconsin()`, `bench_iris()` | Energy Efficiency (n=768, p=8, reg), Wisconsin (n=569, p=10, clf), Iris (n=150, 3-class via `MissMulticlass`) | Complete data + injected synthetic MAR; MICE anchor baseline only |
| 2; real missingness | `bench_auto_mpg()` | Auto MPG (n=392, p=7, reg) | Native missing values, no injection; all four baselines |
| 3; mechanism sweep | `bench_esol_sweep()` | ESOL (n=1128, p=6, reg) | MCAR / MAR / MNAR sweep; CC + MICE anchor baselines |
| 4; grouped data | `bench_mixed()` | Radon (n=919, p=2, 85 counties) | `MissMixedRegressor` / `MissMixedClassifier` on grouped/longitudinal data; all four baselines |

All families are evaluated simultaneously (Linear, Ridge, LASSO, Neighbors, Bayes, Support, Gaussian, Mixed); the Gaussian Process models run only when **n <= `GP_N_THRESHOLD` = 400** (so they are skipped on the larger datasets).

```python
from benchmark_test_suite import bench_energy_efficiency, bench_esol_sweep
results = bench_energy_efficiency(miss_rate=0.20, plot=True)
results = bench_esol_sweep(plot=True)
```

### Datasets: `prepare_datasets.py` and `benchmark_data/`

Download and cache the benchmark datasets once before running the benchmark suite:

```powershell
cd "C:\Users\Amanda\Favorites\Machine Learning\MissLearn\tests"
python prepare_datasets.py          # add --force to re-download
```

CSVs are saved to `tests/benchmark_data/` (`energy_efficiency.csv`, `wisconsin.csv`, `iris.csv`, `auto_mpg.csv`, `esol.csv`, `radon.csv`); existing files are skipped. All datasets are public domain or CC-BY; references are in the loader docstrings.
