# -*- coding: utf-8 -*-
"""Six-arm missing-data benchmark for any MissLearn model family.

This script is the source of truth for the benchmark: it is hand-maintained and
does not depend on the notebooks in any way. It reads its model definitions
from benchmarks/family_registry.py, the same registry the explorer notebooks
use, so the two can never describe different experiments.

Every arm uses the SAME model class and differs only in how missing values are
handled, so a difference between arms is attributable to the missing-data
treatment rather than to model capacity:

    Drop rows | Drop cols | Mean | kNN | MICE | FIML

The conventional arms are given the same internal standardisation the MissLearn
estimator applies, and for the penalized families the regularisation strength is
tuned by inner cross-validation on both arms.

Usage
-----
    python run_benchmark.py --family MissBayes
    python run_benchmark.py --all
    python run_benchmark.py --family MissSupport --rate 0.4 --folds 10 --save
    python run_benchmark.py --list
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# The results tables contain non-ASCII characters (arrows, plus-minus, R^2).
# A Windows console defaults to cp1252 and raises UnicodeEncodeError part way
# through printing them, after the compute has already finished.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_BENCH))    # MissLearn package
sys.path.insert(0, _BENCH)                     # benchmark_core, family_registry

import benchmark_core as bc                                    # noqa: E402
import family_registry as fr                                   # noqa: E402

# Readable output: no tiny fonts, and padding so titles clear the axes.
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})


def show(save_dir, stem):
    """Save when asked, then close. Never blocks a headless run."""
    if save_dir:
        for i, num in enumerate(plt.get_fignums()):
            suffix = "" if len(plt.get_fignums()) == 1 else "_%d" % (i + 1)
            plt.figure(num).savefig(
                os.path.join(save_dir, "%s%s.png" % (stem, suffix)), dpi=130)
    plt.close("all")


def print_table(df):
    import pandas as pd
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string())
        print()


def run_task(spec, task, datasets, folds, seed, save_dir):
    """One task (regression or classification) for one family."""
    sk = spec["reg_sklearn"] if task == "regression" else spec["clf_sklearn"]
    ml = spec["reg_fiml"] if task == "regression" else spec["clf_fiml"]
    name = spec["reg_name"] if task == "regression" else spec["clf_name"]
    meta = bc.REG_METRIC_META if task == "regression" else bc.CLF_METRIC_META
    if sk is None or ml is None:
        print("  (%s not defined for this family; skipping)\n" % task)
        return

    print("=" * 74)
    print("  %s  --  %s" % (name, task.upper()))
    print("=" * 74)

    fold_records, agg = {}, {}
    for ds in datasets:
        print("  %-24s n=%-6s p=%-3s ... " % (ds["name"], ds["n"], ds["p"]),
              end="", flush=True)
        f = bc.run_cv(ds, sk, ml, task=task, n_splits=folds, rng_seed=seed)
        fold_records[ds["name"]] = f
        agg[ds["name"]] = bc.aggregate_folds(f)
        print("done")
    print()

    print_table(bc.make_results_table(datasets, agg, meta))

    for _ in bc.plot_bars(datasets, agg, meta, model_name=name, task=task,
                          n_folds=folds):
        pass
    show(save_dir, "%s_bars" % name)

    for _ in bc.plot_strips(datasets, fold_records, meta, model_name=name,
                            task=task):
        pass
    show(save_dir, "%s_strips" % name)

    bc.plot_gain_heatmap(datasets, agg, meta, model_name=name, task=task)
    show(save_dir, "%s_gain_heatmap" % name)

    print("Paired t-tests (two-sided, alpha=0.05, %d folds):\n" % folds)
    print_table(bc.compute_ttests(datasets, fold_records, meta, folds))

    if save_dir:
        bc.save_results(save_dir, name, task, datasets, fold_records, agg, meta)


def _cap_rows(datasets, max_n):
    """Subsample each dataset to at most max_n rows, preserving everything else.

    Exact Gaussian-process inference is O(n^3), so the GP family cannot run the
    full-size tasks. The previous approach dropped it to the single smallest
    task, which left the regression fit at n=300; that was thin enough to be
    unstable. Capping the rows instead keeps both tasks in the comparison at a
    size the GP can actually manage.
    """
    if not max_n:
        return datasets
    out = []
    for ds in datasets:
        d = dict(ds)
        n = len(d["y"])
        if n > max_n:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, max_n, replace=False))
            d["X"] = d["X"][idx]
            d["y"] = d["y"][idx]
            if d.get("X_complete") is not None:
                d["X_complete"] = d["X_complete"][idx]
            d["n"] = max_n
            d["name"] = "%s (n=%d)" % (d["name"], max_n)
            d["description"] = ("%s; subsampled to n=%d for the O(n^3) GP"
                                % (d.get("description", ""), max_n))
        out.append(d)
    return out


def run_family(key, rate, folds, seed, save):
    spec = fr.FAMILIES[key]
    print("\n" + "#" * 74)
    print("#  %s  --  %s" % (key, spec["label"]))
    print("#" * 74)
    print(spec["blurb"] + "\n")

    # The Gaussian process is O(n^3). Rather than dropping it to the single
    # smallest task, both tasks are kept and the rows are capped.
    max_n = spec.get("max_n")
    reg = _cap_rows(bc.make_regression_datasets(mar_rate=rate, seed=seed)[:2],
                    max_n)
    clf = _cap_rows(bc.make_classification_datasets(mar_rate=rate, seed=seed)[:2],
                    max_n)

    save_dir = None
    if save:
        save_dir = os.path.join(_BENCH, "results", "%s_benchmark" % key)
        os.makedirs(save_dir, exist_ok=True)
        print("Writing to: %s\n" % save_dir)

    print_table(bc.dataset_summary_table(reg, clf))
    bc.plot_missing_profile(
        clf + reg,
        title="Per-feature missing rates, MAR injection at %.0f%%" % (rate * 100),
        col_labels=["Classification", "Regression"])
    show(save_dir, "missing_profile")

    run_task(spec, "regression", reg, folds, seed, save_dir)
    run_task(spec, "classification", clf, folds, seed, save_dir)


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(fr.FAMILIES),
                   help="model family to benchmark")
    p.add_argument("--all", action="store_true", help="run every family")
    p.add_argument("--list", action="store_true", help="list families and exit")
    p.add_argument("--rate", type=float, default=0.25,
                   help="MAR fraction injected (default %(default)s)")
    p.add_argument("--folds", type=int, default=5,
                   help="cross-validation folds (default %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default %(default)s)")
    p.add_argument("--save", action="store_true",
                   help="write figures and tables under benchmarks/results/")
    a = p.parse_args()

    if a.list or not (a.family or a.all):
        fr.describe()
        if not (a.family or a.all):
            print("\nChoose one with --family NAME, or run them all with --all.")
        return

    keys = sorted(fr.FAMILIES) if a.all else [a.family]
    for k in keys:
        run_family(k, a.rate, a.folds, a.seed, a.save)
    print("\nDone (%d famil%s)." % (len(keys), "y" if len(keys) == 1 else "ies"))


if __name__ == "__main__":
    main()
