# -*- coding: utf-8 -*-
"""Build the two interactive explorer notebooks.

    Benchmark_Explorer.ipynb   pick a family, run the six-arm benchmark
    Sweep_Explorer.ipynb       pick a family, run the missing-rate sweep

Both let the reader choose which family to run from a dropdown (or a plain
variable if ipywidgets is unavailable), show every table and figure inline, and
write nothing to disk.

Run from benchmarks/:

    python _create_explorer_notebooks.py
"""
import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))


def code(src, n):
    return {"cell_type": "code", "execution_count": None,
            "id": "ex%02d00-0000-0000-0000-000000000000" % n,
            "metadata": {}, "outputs": [],
            "source": src.rstrip("\n").splitlines(keepends=True)}


def md(src, n):
    return {"cell_type": "markdown",
            "id": "ex%02d00-0000-0000-0000-000000000000" % n,
            "metadata": {},
            "source": src.rstrip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python",
                                           "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------- shared cells

SETUP = textwrap.dedent('''\
    import sys, os, pathlib, warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown

    _BENCH = pathlib.Path.cwd()
    sys.path.insert(0, str(_BENCH.parent))   # the MissLearn package
    sys.path.insert(0, str(_BENCH))          # benchmark_core, family_registry

    import benchmark_core as bc
    import family_registry as fr

    # ---- Readability ------------------------------------------------------
    # Large enough to read on a projector or in a GitHub preview, with padding
    # so titles never sit on top of the axes.
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "figure.constrained_layout.use": True,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    })
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)

    # Nothing in this notebook writes to disk.
    SAVE = False

    print("Ready. Available model families:\\n")
    fr.describe()
''')

CHOOSER = textwrap.dedent('''\
    # ======================================================================
    #  CHOOSE WHAT TO RUN
    #  Pick from the dropdown, or set FAMILY directly if widgets are not
    #  available. Then run the remaining cells.
    # ======================================================================
    FAMILY   = "{default}"      # any key printed above
    N_FOLDS  = 5
    RNG_SEED = 42
    {extra}

    try:
        import ipywidgets as W
        _opts = [(f"{{k}}  --  {{v['label']}}", k)
                 for k, v in fr.FAMILIES.items() {filt}]
        _dd = W.Dropdown(options=_opts, value=FAMILY, description="Family:",
                         layout=W.Layout(width="640px"),
                         style={{"description_width": "90px"}})

        def _on_change(change):
            global FAMILY
            if change["name"] == "value":
                FAMILY = change["new"]
                print("FAMILY =", FAMILY, "->", fr.FAMILIES[FAMILY]["label"])

        _dd.observe(_on_change)
        display(_dd)
        print("Dropdown ready. Selecting an option updates FAMILY, "
              "then run the cells below.")
    except ImportError:
        print("ipywidgets not installed; edit FAMILY above instead.")

    print("\\nCurrent selection:", FAMILY)
''')

RESOLVE = textwrap.dedent('''\
    spec = fr.FAMILIES[FAMILY]
    display(Markdown(f"## {spec['label']}\\n\\n{spec['blurb']}"))

    # The Gaussian process is O(n^3), so it uses the small datasets only.
    _small_only = spec.get("small_only", False)
    _n_ds = 1 if _small_only else 2
    print(f"Model class held fixed; only the missing-data strategy varies.")
    print(f"Datasets: {'small only (O(n^3) model)' if _small_only else 'small + medium'}")
''')


# --------------------------------------------------------------- benchmark nb

def benchmark_notebook():
    c, i = [], 0
    c.append(md(textwrap.dedent('''\
        # Benchmark Explorer

        Choose a MissLearn model family and run the six-arm benchmark on
        controlled synthetic data. Every arm uses the **same model class** and
        differs only in how the missing values are handled:

        | Arm | Treatment |
        |-----|-----------|
        | Drop rows | listwise deletion |
        | Drop cols | discard every column containing a missing value |
        | Mean | per-column mean imputation |
        | kNN | `KNNImputer(n_neighbors=5)` |
        | MICE | `IterativeImputer`, the strongest conventional competitor |
        | FIML | the MissLearn estimator on the incomplete matrix |

        Because the model class is fixed and the conventional arms receive the
        same internal standardisation the MissLearn estimator applies, a
        difference between arms is attributable to the missing-data treatment
        and not to model capacity or preprocessing. For the penalized families
        the regularisation strength is tuned by inner cross-validation on both
        arms.

        Nothing here writes to disk. To save results, use the per-family
        scripts in `scripts/` with `--save`.
    '''), i := i + 1))
    c.append(md("---\n## 0. Setup", i := i + 1))
    c.append(code(SETUP, i := i + 1))
    c.append(md("---\n## 1. Choose a family", i := i + 1))
    c.append(code(CHOOSER.format(default="MissBayes", extra="MISS_RATE = 0.25",
                                 filt=""), i := i + 1))
    c.append(md("---\n## 2. Build the datasets", i := i + 1))
    c.append(code(RESOLVE, i := i + 1))
    c.append(code(textwrap.dedent('''\
        reg_datasets = bc.make_regression_datasets(mar_rate=MISS_RATE,
                                                   seed=RNG_SEED)[:_n_ds]
        clf_datasets = bc.make_classification_datasets(mar_rate=MISS_RATE,
                                                       seed=RNG_SEED)[:_n_ds]
        display(bc.dataset_summary_table(reg_datasets, clf_datasets))
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        _ = bc.plot_missing_profile(
            clf_datasets + reg_datasets,
            title=f"Per-feature missing rates, MAR injection at {MISS_RATE*100:.0f}%",
            col_labels=["Classification", "Regression"])
        plt.show()
    '''), i := i + 1))

    c.append(md(textwrap.dedent('''\
        ---
        ## 3. Regression

        Five-fold cross-validation; every arm sees the identical incomplete
        matrices and is scored against the same held-out targets.
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        reg_folds, reg_agg = {}, {}
        for ds in reg_datasets:
            print(f"  {ds['name']}  (n={ds['n']:,}, p={ds['p']}) ... ",
                  end="", flush=True)
            f = bc.run_cv(ds, spec["reg_sklearn"], spec["reg_fiml"],
                          task="regression", n_splits=N_FOLDS,
                          rng_seed=RNG_SEED)
            reg_folds[ds["name"]] = f
            reg_agg[ds["name"]] = bc.aggregate_folds(f)
            print("done")
        display(bc.make_results_table(reg_datasets, reg_agg, bc.REG_METRIC_META))
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_bars(reg_datasets, reg_agg, bc.REG_METRIC_META,
                                model_name=spec["reg_name"], task="regression",
                                n_folds=N_FOLDS):
            plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_strips(reg_datasets, reg_folds, bc.REG_METRIC_META,
                                  model_name=spec["reg_name"],
                                  task="regression"):
            plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        _ = bc.plot_gain_heatmap(reg_datasets, reg_agg, bc.REG_METRIC_META,
                                 model_name=spec["reg_name"],
                                 task="regression")
        plt.show()

        print("Paired t-tests, two-sided, alpha=0.05:\\n")
        display(bc.compute_ttests(reg_datasets, reg_folds,
                                  bc.REG_METRIC_META, N_FOLDS))
    '''), i := i + 1))

    c.append(md("---\n## 4. Classification", i := i + 1))
    c.append(code(textwrap.dedent('''\
        clf_folds, clf_agg = {}, {}
        for ds in clf_datasets:
            print(f"  {ds['name']}  (n={ds['n']:,}, p={ds['p']}, "
                  f"pos={ds['pos_rate']*100:.0f}%) ... ", end="", flush=True)
            f = bc.run_cv(ds, spec["clf_sklearn"], spec["clf_fiml"],
                          task="classification", n_splits=N_FOLDS,
                          rng_seed=RNG_SEED)
            clf_folds[ds["name"]] = f
            clf_agg[ds["name"]] = bc.aggregate_folds(f)
            print("done")
        display(bc.make_results_table(clf_datasets, clf_agg, bc.CLF_METRIC_META))
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_bars(clf_datasets, clf_agg, bc.CLF_METRIC_META,
                                model_name=spec["clf_name"],
                                task="classification", n_folds=N_FOLDS):
            plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_strips(clf_datasets, clf_folds, bc.CLF_METRIC_META,
                                  model_name=spec["clf_name"],
                                  task="classification"):
            plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        _ = bc.plot_gain_heatmap(clf_datasets, clf_agg, bc.CLF_METRIC_META,
                                 model_name=spec["clf_name"],
                                 task="classification")
        plt.show()

        print("Paired t-tests, two-sided, alpha=0.05:\\n")
        display(bc.compute_ttests(clf_datasets, clf_folds,
                                  bc.CLF_METRIC_META, N_FOLDS))
    '''), i := i + 1))

    c.append(md(textwrap.dedent('''\
        ---
        ## How to read this

        * Compare arms **within a row**. Rows are different datasets and are
          not comparable with one another.
        * Column deletion is usually the worst arm by a wide margin, because it
          discards whole variables to avoid a few holes.
        * On well-specified linear ground truth, parity between FIML and good
          imputation is the *expected* result, not a disappointment: a correct
          likelihood combined with good imputation both approach the efficiency
          bound. The argument for FIML there is one deterministic fit, honest
          standard errors, and intervals that widen with missingness.
        * The clearest gains usually appear in the Brier score rather than in
          point accuracy, because marginalisation propagates feature
          uncertainty into the predicted probabilities.
        * To run every family in one go, or to save output, use the scripts in
          `scripts/`.
    '''), i := i + 1))
    return notebook(c)


# ------------------------------------------------------------------ sweep nb

def sweep_notebook():
    c, i = [], 0
    c.append(md(textwrap.dedent('''\
        # Sweep Explorer

        Choose a MissLearn model family and sweep the missing rate from 10% to
        50%, re-injecting an independent MAR pattern at each rate and repeating
        the full six-arm, five-fold protocol.

        As in the benchmark explorer the model class is held fixed and only the
        missing-data treatment varies, so the curves compare strategies rather
        than learners. A complete-data run (0% missing) is included as the
        reference the degradation plots measure against.

        Nothing here writes to disk.
    '''), i := i + 1))
    c.append(md("---\n## 0. Setup", i := i + 1))
    c.append(code(SETUP, i := i + 1))
    c.append(md("---\n## 1. Choose a family and the rate grid", i := i + 1))
    c.append(code(CHOOSER.format(
        default="MissBayes",
        extra="MISS_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]",
        filt="if v.get('has_sweep')"), i := i + 1))
    c.append(md("---\n## 2. Run the sweep", i := i + 1))
    c.append(code(RESOLVE, i := i + 1))
    c.append(code(textwrap.dedent('''\
        reg_datasets = bc.make_regression_datasets(mar_rate=MISS_RATES[0],
                                                   seed=RNG_SEED)[:_n_ds]
        clf_datasets = bc.make_classification_datasets(mar_rate=MISS_RATES[0],
                                                       seed=RNG_SEED)[:_n_ds]
        print("Sweeping:")
        for d in reg_datasets + clf_datasets:
            print(f"  {d['name']:<24} n={d['n']:,}  p={d['p']}")


        def run_sweep(datasets, sk, fiml, task):
            """Sweep the rate grid, plus a complete-data reference at 0%."""
            agg, zero, raw = {}, {}, {}
            for ds in datasets:
                print(f"  {ds['name']} ... ", end="", flush=True)
                r = bc.run_cv_at_rate(ds, sk, fiml, task,
                                      miss_rates=MISS_RATES,
                                      n_splits=N_FOLDS, rng_seed=RNG_SEED)
                raw[ds["name"]] = r
                agg[ds["name"]] = {k: bc.aggregate_folds(v)
                                   for k, v in r.items()}
                z = bc.run_cv_at_rate(ds, sk, fiml, task, miss_rates=[0.0],
                                      n_splits=N_FOLDS, rng_seed=RNG_SEED)
                zero[ds["name"]] = bc.aggregate_folds(z[0.0])
                print("done")
            return agg, zero, raw
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        print(f"Regression sweep over {[f'{r*100:.0f}%' for r in MISS_RATES]}")
        reg_sweep, reg_zero, reg_raw = run_sweep(
            reg_datasets, spec["reg_sklearn"], spec["reg_fiml"], "regression")

        print(f"\\nClassification sweep over {[f'{r*100:.0f}%' for r in MISS_RATES]}")
        clf_sweep, clf_zero, clf_raw = run_sweep(
            clf_datasets, spec["clf_sklearn"], spec["clf_fiml"],
            "classification")
    '''), i := i + 1))

    c.append(md("---\n## 3. Metric against missing rate", i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_sweep_lines(reg_datasets, reg_sweep,
                                       bc.REG_METRIC_META,
                                       model_name=spec["reg_name"],
                                       task="regression",
                                       miss_rates=MISS_RATES,
                                       n_splits=N_FOLDS):
            plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_sweep_lines(clf_datasets, clf_sweep,
                                       bc.CLF_METRIC_META,
                                       model_name=spec["clf_name"],
                                       task="classification",
                                       miss_rates=MISS_RATES,
                                       n_splits=N_FOLDS):
            plt.show()
    '''), i := i + 1))

    c.append(md(textwrap.dedent('''\
        ---
        ## 4. Degradation from complete data

        How much each strategy loses relative to the same model fitted on the
        complete matrix.
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        for fig in bc.plot_sweep_degradation(reg_datasets, reg_sweep, reg_zero,
                                             bc.REG_METRIC_META,
                                             model_name=spec["reg_name"],
                                             task="regression",
                                             miss_rates=MISS_RATES):
            plt.show()
        for fig in bc.plot_sweep_degradation(clf_datasets, clf_sweep, clf_zero,
                                             bc.CLF_METRIC_META,
                                             model_name=spec["clf_name"],
                                             task="classification",
                                             miss_rates=MISS_RATES):
            plt.show()
    '''), i := i + 1))

    c.append(md("---\n## 5. Ranking and gain", i := i + 1))
    c.append(code(textwrap.dedent('''\
        _ = bc.plot_sweep_rank(reg_datasets, reg_sweep, "r2",
                               model_name=spec["reg_name"], task="regression",
                               miss_rates=MISS_RATES)
        plt.show()
        _ = bc.plot_sweep_rank(clf_datasets, clf_sweep, "auc",
                               model_name=spec["clf_name"],
                               task="classification", miss_rates=MISS_RATES)
        plt.show()
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        _ = bc.plot_sweep_gain_heatmap(reg_datasets, reg_sweep,
                                       bc.REG_METRIC_META,
                                       model_name=spec["reg_name"],
                                       task="regression",
                                       miss_rates=MISS_RATES)
        plt.show()
        _ = bc.plot_sweep_gain_heatmap(clf_datasets, clf_sweep,
                                       bc.CLF_METRIC_META,
                                       model_name=spec["clf_name"],
                                       task="classification",
                                       miss_rates=MISS_RATES)
        plt.show()
    '''), i := i + 1))

    c.append(md(textwrap.dedent('''\
        ---
        ## 6. Crossover

        The lowest rate at which the full-information arm first beats each
        conventional strategy on each metric.
    '''), i := i + 1))
    c.append(code(textwrap.dedent('''\
        print("Regression:")
        display(bc.compute_sweep_crossover(reg_datasets, reg_sweep,
                                           bc.REG_METRIC_META, MISS_RATES))
        print("Classification:")
        display(bc.compute_sweep_crossover(clf_datasets, clf_sweep,
                                           bc.CLF_METRIC_META, MISS_RATES))
    '''), i := i + 1))

    c.append(md(textwrap.dedent('''\
        ---
        ## How to read this

        * The interesting question is the **shape** of each curve, not its
          height: the panels use different scales and different datasets.
        * Listwise deletion falls away fastest, because the complete-case
          subset shrinks roughly geometrically as the rate rises.
        * Where FIML and MICE track each other closely, the honest reading is
          parity maintained across the range rather than an advantage. Look at
          the high-rate end for the real difference: imputation pipelines
          become unstable well before marginalisation does.
        * To run every family in one go, or to save output, use the scripts in
          `scripts/`.
    '''), i := i + 1))
    return notebook(c)


if __name__ == "__main__":
    for name, builder in (("Benchmark_Explorer.ipynb", benchmark_notebook),
                          ("Sweep_Explorer.ipynb", sweep_notebook)):
        p = os.path.join(HERE, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(builder(), fh, indent=1)
        print("wrote", name)
