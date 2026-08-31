# MissLearn Benchmarks

**Synthetic data only.** Every MissLearn model family is compared against the
standard missing-data strategies; **Drop rows**, **Drop cols**, **Mean**,
**kNN**, **MICE** and **FIML**, on controlled synthetic data, so the only thing
that varies is the missing-data strategy.

> Real-world data sets live in `../examples/`, which run the same strategy
> comparison on naturally incomplete data.

## What makes a comparison fair here

A difference between two arms is evidence about the missing-data strategy only
if nothing else differs. Three controls enforce that, all implemented in
`benchmark_core.py`:

1. **Model class held fixed.** Each MissLearn family is compared against its own
   scikit-learn counterpart (`MissSupport` against `SVC`/`SVR`, `MissNeighbors`
   against `KNeighbors*`, and so on), never against a different class.
   Comparing a linear model with a boosted tree would measure model capacity,
   not missing-data handling, so no tree appears anywhere in this directory.
2. **Matched preprocessing** (`FAIR_BASELINE_SCALING`). Several MissLearn
   estimators standardise internally, so every conventional arm is given the
   same standardisation after its deletion or imputation step. Without this the
   baselines lose on preprocessing rather than on their strategy, which
   previously inflated the apparent advantage of the kernel and distance
   families substantially.
3. **Matched regularisation** (`tuned_sklearn`, `TunedMiss`). For the penalized
   families the strength is chosen by inner cross-validation on *both* arms. A
   fixed `alpha` is not comparable across the two objectives, because
   scikit-learn divides the squared-error term by `n` and MissLearn does not.
4. **No marginal transform on one side only.** The Gaussian copula
   (`copula='auto'`) is a preprocessing step, not a missing-data strategy, so
   giving it to the FIML arm alone would measure the transform. It is left off
   throughout this directory. That costs nothing here, and the reason is worth
   recording: `needs_copula` returns `False` on all six synthetic datasets, so
   `'auto'` would decline to fire on every one of them even if it were asked.
   The controlled generators draw Gaussian features, which is precisely the
   regime the transform exists to leave alone. On the real data in
   `../examples/` the same question has real force and is handled there.

## Two ways to run

### 1. Explorer notebooks (start here)

Pick a model family from a dropdown and read the results inline:

| Notebook | What it does |
|----------|--------------|
| `Benchmark_Explorer.ipynb` | one family, six missing-data strategies, at a fixed missing rate |
| `Sweep_Explorer.ipynb` | one family, six strategies, swept from 10% to 50% missing |

Both display every table and figure inline and **write nothing to disk**.

### 2. Scripts

Self-contained command-line equivalents, for headless runs, for diffing in
version control, and for reading on GitHub without a notebook renderer. Nothing
is written unless you pass `--save`.

```bash
python scripts/run_benchmark.py --list                     # show the families
python scripts/run_benchmark.py --family MissBayes         # one family
python scripts/run_benchmark.py --all --save               # everything, saved
python scripts/run_sweep.py --family MissLASSO --rates 0.1 0.3 0.5
python scripts/run_mixed_benchmark.py                      # grouped data
```

| Script | Covers |
|--------|--------|
| `scripts/run_benchmark.py` | six-arm benchmark, any family, both tasks |
| `scripts/run_sweep.py` | missing-rate sweep, any family, both tasks |
| `scripts/run_mixed_benchmark.py` | `MissMixed`, which needs a `groups` variable and so carries its own grouped-data harness |

Both the explorers and the scripts read their model definitions from
`family_registry.py`, so they cannot describe different experiments. The scripts
are hand-maintained source, not generated from anything.

## Model families

| Key | Model class | Sweep |
|-----|-------------|-------|
| `MissLinear` | linear / logistic regression | yes |
| `MissRidge` | ridge, L2 penalized | yes |
| `MissLASSO` | LASSO, L1 penalized | yes |
| `MissBayes` | full-covariance generative Gaussian | yes |
| `MissNeighbors` | k-nearest neighbours, expected distance | yes |
| `MissSupport` | support vector, expected kernel | yes |
| `MissGaussian` | Gaussian process, marginalised kernel | yes (rows capped) |
| `MissMixed` | random-intercept mixed effects | benchmark only |

`MissGaussian` runs every task, but with the rows capped at `max_n` (900),
because exact Gaussian-process inference is O(n³). The scripts and explorers
apply the cap automatically, and the capped medium task is reported as
`Regression-Medium (n=900)` so the reduced sample size is visible in the
output rather than implied.

Both Gaussian-process arms are given a learnable noise term, using the
scikit-learn default kernel bounds. This matters more than it looks. A
comparator confined to near-exact interpolation diverges on incomplete data,
because the imputed rows form near-duplicate inputs carrying different
targets. Tightening the bounds to force a smaller noise floor does not help
either: it sends the optimiser into a degenerate all-noise solution on some
folds, where the fitted noise reaches the response variance and the model
predicts the mean. The defaults avoid both failures, and reproduce the
published numbers exactly.

## Interpreting the output

* Compare arms **within a row**. Rows are different datasets and are not
  comparable with one another.
* Column deletion is usually the worst arm by a wide margin, because it discards
  whole variables to avoid a few holes.
* On well-specified linear ground truth, **parity** between FIML and good
  imputation is the expected result, not a disappointment: a correct likelihood
  and good imputation both approach the efficiency bound. The argument for FIML
  there is one deterministic fit, honest standard errors, and intervals that
  widen with missingness.
* The clearest gains usually appear in the **Brier score** rather than in point
  accuracy, because marginalisation propagates feature uncertainty into the
  predicted probabilities.

## Files

| File | Role |
|------|------|
| `benchmark_core.py` | datasets, the six-arm CV engine, plots, tables, statistics |
| `family_registry.py` | one definition of each family and its matched counterpart |
| `Benchmark_Explorer.ipynb`, `Sweep_Explorer.ipynb` | interactive entry points |
| `explorer_io.py` | save/load helpers shared by the two explorers |
| `scripts/` | command-line equivalents |
| `NOTEBOOKS/` | generated per-family notebooks; regenerated by `_create_explorer_notebooks.py` and not published |
| `explorer_output/` | timestamped run archives, written only with `--save` |
| `_archive/` | withdrawn harness code, kept because this tree has no version control |
