# -*- coding: utf-8 -*-
"""Missing-rate sweep for any MissLearn model family.

This script is the source of truth for the sweep: it is hand-maintained and
does not depend on the notebooks in any way. It reads its model definitions
from benchmarks/family_registry.py, the same registry the explorer notebooks
use.

An independent MAR pattern is injected at each rate and the full six-arm,
k-fold protocol is repeated, with the model class held fixed throughout so the
curves compare missing-data strategies rather than learners. A complete-data
run at 0% is included as the reference the degradation plots measure against.

Usage
-----
    python run_sweep.py --family MissBayes
    python run_sweep.py --all
    python run_sweep.py --family MissLASSO --rates 0.1 0.3 0.5 --folds 3
    python run_sweep.py --list
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

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
sys.path.insert(0, os.path.dirname(_BENCH))
sys.path.insert(0, _BENCH)

import benchmark_core as bc                                    # noqa: E402
import family_registry as fr                                   # noqa: E402

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})

DEFAULT_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]


def show(save_dir, stem):
    if save_dir:
        nums = plt.get_fignums()
        for i, num in enumerate(nums):
            suffix = "" if len(nums) == 1 else "_%d" % (i + 1)
            plt.figure(num).savefig(
                os.path.join(save_dir, "%s%s.png" % (stem, suffix)), dpi=130)
    plt.close("all")


def print_table(df):
    import pandas as pd
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string())
        print()


def sweep_task(spec, task, datasets, rates, folds, seed, save_dir):
    sk = spec["reg_sklearn"] if task == "regression" else spec["clf_sklearn"]
    ml = spec["reg_fiml"] if task == "regression" else spec["clf_fiml"]
    name = spec["reg_name"] if task == "regression" else spec["clf_name"]
    meta = bc.REG_METRIC_META if task == "regression" else bc.CLF_METRIC_META
    rank_metric = "r2" if task == "regression" else "auc"
    if sk is None or ml is None:
        print("  (%s not defined for this family; skipping)\n" % task)
        return

    print("=" * 74)
    print("  %s  --  %s sweep over %s"
          % (name, task.upper(), ["%.0f%%" % (r * 100) for r in rates]))
    print("=" * 74)

    agg, zero = {}, {}
    for ds in datasets:
        print("  %-24s ... " % ds["name"], end="", flush=True)
        raw = bc.run_cv_at_rate(ds, sk, ml, task, miss_rates=rates,
                                n_splits=folds, rng_seed=seed)
        agg[ds["name"]] = {r: bc.aggregate_folds(f) for r, f in raw.items()}
        z = bc.run_cv_at_rate(ds, sk, ml, task, miss_rates=[0.0],
                              n_splits=folds, rng_seed=seed)
        zero[ds["name"]] = bc.aggregate_folds(z[0.0])
        print("done")
    print()

    for _ in bc.plot_sweep_lines(datasets, agg, meta, model_name=name,
                                 task=task, miss_rates=rates, n_splits=folds):
        pass
    show(save_dir, "%s_lines" % name)

    for _ in bc.plot_sweep_degradation(datasets, agg, zero, meta,
                                       model_name=name, task=task,
                                       miss_rates=rates):
        pass
    show(save_dir, "%s_degradation" % name)

    bc.plot_sweep_rank(datasets, agg, rank_metric, model_name=name, task=task,
                       miss_rates=rates)
    show(save_dir, "%s_rank" % name)

    bc.plot_sweep_gain_heatmap(datasets, agg, meta, model_name=name, task=task,
                               miss_rates=rates)
    show(save_dir, "%s_gain" % name)

    if save_dir:
        bc.save_sweep_results(save_dir, name, task, agg, zero)

    print("Crossover (lowest rate at which FIML beats each baseline):\n")
    print_table(bc.compute_sweep_crossover(datasets, agg, meta, rates))


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


def run_family(key, rates, folds, seed, save):
    spec = fr.FAMILIES[key]
    if not spec.get("has_sweep", True):
        print("%s has no sweep configuration; skipping." % key)
        return
    print("\n" + "#" * 74)
    print("#  %s  --  %s" % (key, spec["label"]))
    print("#" * 74)
    print(spec["blurb"] + "\n")

    # The Gaussian process is O(n^3). Rather than dropping it to the single
    # smallest task, both tasks are kept and the rows are capped.
    max_n = spec.get("max_n")
    reg = _cap_rows(bc.make_regression_datasets(mar_rate=rates[0], seed=seed)[:2],
                    max_n)
    clf = _cap_rows(bc.make_classification_datasets(mar_rate=rates[0], seed=seed)[:2],
                    max_n)

    save_dir = None
    if save:
        save_dir = os.path.join(_BENCH, "results", "%s_sweep" % key)
        os.makedirs(save_dir, exist_ok=True)
        print("Writing to: %s\n" % save_dir)

    sweep_task(spec, "regression", reg, rates, folds, seed, save_dir)
    sweep_task(spec, "classification", clf, rates, folds, seed, save_dir)


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(fr.SWEEP_FAMILIES),
                   help="model family to sweep")
    p.add_argument("--all", action="store_true", help="sweep every family")
    p.add_argument("--list", action="store_true", help="list families and exit")
    p.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES,
                   help="missing rates to sweep (default: %s)" % DEFAULT_RATES)
    p.add_argument("--folds", type=int, default=5,
                   help="cross-validation folds per rate (default %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default %(default)s)")
    p.add_argument("--save", action="store_true",
                   help="write figures under benchmarks/results/")
    a = p.parse_args()

    if a.list or not (a.family or a.all):
        fr.describe()
        if not (a.family or a.all):
            print("\nChoose one with --family NAME, or run them all with --all.")
        return

    keys = sorted(fr.SWEEP_FAMILIES) if a.all else [a.family]
    for k in keys:
        run_family(k, sorted(a.rates), a.folds, a.seed, a.save)
    print("\nDone (%d famil%s)." % (len(keys), "y" if len(keys) == 1 else "ies"))


if __name__ == "__main__":
    main()
