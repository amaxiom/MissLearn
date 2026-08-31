# -*- coding: utf-8 -*-
"""Grouped-data (mixed-effects) benchmark for MissMixed.

Source of truth for the MissMixed benchmark: hand-maintained, standalone, and
independent of any notebook.

MissMixed is kept out of run_benchmark.py because it needs a `groups` variable,
which benchmark_core's shared CV engine does not carry. This script therefore
ships its own grouped-data generator and CV harness. Everything else matches
the other benchmarks: the same model class is used throughout and only the
missing-data treatment varies.

Usage:
    python run_mixed_benchmark.py                  # nothing written to disk
    python run_mixed_benchmark.py --save           # write figures and tables
    python run_mixed_benchmark.py --rate 0.4 --folds 10 --seed 7
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# The results tables contain non-ASCII characters (arrows, plus-minus, R^2).
# A Windows console defaults to cp1252 and raises UnicodeEncodeError part way
# through printing them, which kills the run after the compute has finished.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")          # headless servers
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_BENCH))   # MissLearn package
sys.path.insert(0, _BENCH)                    # benchmark_core

_p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_p.add_argument("--rate", type=float, default=0.25,
                help="MAR fraction injected (default %(default)s)")
_p.add_argument("--folds", type=int, default=5,
                help="cross-validation folds (default %(default)s)")
_p.add_argument("--seed", type=int, default=42,
                help="random seed (default %(default)s)")
_p.add_argument("--save", action="store_true",
                help="write figures and tables under benchmarks/results/")
_args = _p.parse_args()

# Readable output: no tiny fonts, and room for titles so headers do not sit on
# top of the axes.
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})


def display(obj):
    """Stand-in for the notebook display(); prints tables in full."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            with pd.option_context("display.max_columns", None,
                                   "display.width", 200):
                print(obj.to_string())
                print()
                return
    except ImportError:
        pass
    print(obj)


_SHOW = plt.show


def _show(*a, **k):
    """Only block on a real display; headless runs just move on."""
    if _args.save or matplotlib.get_backend().lower() == "agg":
        plt.close("all")
    else:
        _SHOW(*a, **k)


plt.show = _show


# # MissLearn: MissMixed Benchmark
#
# **MissMixed** is a full-information random-intercept model: on top of the FIML fixed-effects fit it adds a per-group random intercept `b_g ~ N(0, τ²)`, the missing-data analogue of a linear/generalised linear mixed model. It is the right tool for **grouped / longitudinal** data where observations cluster within subjects, sites, or repeated measurements.
#
# This notebook benchmarks **MissMixedRegressor** and **MissMixedClassifier** on synthetic grouped data with a *known* intraclass correlation, against:
#
# | Method | Uses group structure? | Handles missing? |
# |--------|----------------------|------------------|
# | **Complete Case + linear** | no | drops incomplete rows |
# | **Mean + linear** | no | imputes |
# | **KNN + linear** | no | imputes |
# | **MICE + linear** | no | imputes |
# | **MissLinear / MissLogistic (FIML)** | no | no imputation |
# | **MissMixed (FIML + random intercept)** | **yes** | no imputation |
#
# The central question: **when between-group variance is real, does the random intercept buy accuracy that the flat models cannot?**
#
# ---
# **To run:** execute all cells in order.  Only Cell 1 (Configuration) ever needs editing.

# ## ⚙️ Configuration

# ================================================================
# Cell 1: Configuration
# ================================================================

N_GROUPS      = 40     # number of clusters / subjects
OBS_PER_GROUP = 20     # observations per group
P_FEATURES    = 5      # number of predictors
SIGMA_B       = 2.0    # between-group SD (random intercept)
SIGMA_E       = 1.0    # within-group residual SD
MISS_RATE = _args.rate
N_FOLDS = _args.folds
RNG_SEED = _args.seed
SAVE = _args.save

# Implied true intraclass correlation (ICC):
#   ICC = SIGMA_B**2 / (SIGMA_B**2 + SIGMA_E**2)

# ## Setup

import sys, os, warnings, datetime, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display

_BENCH_DIR      = pathlib.Path(_BENCH)
_MISSLEARN_ROOT = _BENCH_DIR.parent
sys.path.insert(0, str(_MISSLEARN_ROOT))

from MissLearn import (MissLinear, MissLogistic,
                       MissMixedRegressor, MissMixedClassifier)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, roc_auc_score,
                             f1_score, brier_score_loss)

_date       = datetime.date.today().strftime("%Y-%m-%d")
RESULTS_DIR = str(_BENCH_DIR / "results" / f"MissMixed_{_date}")
if SAVE:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to:\n  {RESULTS_DIR}")
else:
    print("SAVE=False; results will not be written.")

# Viridis colour per method (baselines grey, MissMixed the bright end).
_VIR = plt.cm.viridis
def method_color(name):
    if name.startswith("MissMixed"):
        return mcolors.to_hex(_VIR(0.95))
    if name.startswith("Miss"):
        return mcolors.to_hex(_VIR(0.55))
    return "#9e9e9e"
print("Setup complete.")

# ## Synthetic Grouped Data
#
# Each group *g* gets its own random intercept `b_g ~ N(0, SIGMA_B²)`. For regression `y = Xβ + b_g + ε`; for classification `y ~ Bernoulli(σ(Xβ + b_g))`. MAR missingness is then injected into `X` (60% of columns, scaled to `MISS_RATE` overall); `y` and the group labels stay fully observed.

def _inject_mar(X, rate, seed):
    """Mask ~rate of entries MAR: 60%% of columns are eligible, each
    masked where a *driver* column is above its median."""
    rng = np.random.default_rng(seed)
    Xm  = X.copy()
    n, p = X.shape
    n_cols = max(1, int(round(0.6 * p)))
    cols   = rng.choice(p, size=n_cols, replace=False)
    per_col = min(0.95, rate * p / n_cols)
    for j in cols:
        driver = (j + 1) % p
        above  = X[:, driver] > np.median(X[:, driver])
        thr    = rng.random(n) < (per_col * np.where(above, 1.4, 0.6))
        Xm[thr, j] = np.nan
    return Xm

def make_grouped(task, seed=RNG_SEED):
    rng    = np.random.default_rng(seed)
    n      = N_GROUPS * OBS_PER_GROUP
    groups = np.repeat(np.arange(N_GROUPS), OBS_PER_GROUP)
    X      = rng.normal(size=(n, P_FEATURES))
    beta   = rng.normal(size=P_FEATURES)
    b_g    = rng.normal(0.0, SIGMA_B, size=N_GROUPS)
    lin    = X @ beta + b_g[groups]
    if task == "regression":
        y = lin + rng.normal(0.0, SIGMA_E, size=n)
    else:
        p1 = 1.0 / (1.0 + np.exp(-lin))
        y  = (rng.random(n) < p1).astype(float)
    Xm = _inject_mar(X, MISS_RATE, seed + 1)
    return dict(X=Xm, X_complete=X, y=y, groups=groups,
                n=n, p=P_FEATURES, task=task)

reg_data = make_grouped("regression")
clf_data = make_grouped("classification")
print(f"Regression : n={reg_data['n']}, {N_GROUPS} groups, "
      f"missing={np.isnan(reg_data['X']).mean()*100:.1f}%")
print(f"Classification: n={clf_data['n']}, {N_GROUPS} groups, "
      f"pos_rate={clf_data['y'].mean()*100:.0f}%, "
      f"missing={np.isnan(clf_data['X']).mean()*100:.1f}%")

# ## Variance Components / ICC
#
# Before predicting, confirm MissMixed recovers the variance structure we built in. `τ²` is the estimated between-group variance, `σ²` the residual variance, and `ICC = τ²/(τ²+σ²)` the share of variance explained by group membership.

_mm = MissMixedRegressor(compute_se=False)
_mm.fit(reg_data['X'], reg_data['y'], reg_data['groups'])
true_icc = SIGMA_B**2 / (SIGMA_B**2 + SIGMA_E**2)
print(f"  true   τ²={SIGMA_B**2:.3f}  σ²={SIGMA_E**2:.3f}  "
      f"ICC={true_icc:.3f}")
print(f"  fitted τ²={_mm.tau_sq_:.3f}  σ²={_mm.sigma_sq_:.3f}  "
      f"ICC={_mm.icc_:.3f}")
print(f"\n  {N_GROUPS} random intercepts estimated (BLUPs); "
      f"the flat models below ignore them entirely.")

# ## Cross-Validation Harness
#
# All imputers are fit on the **training fold only** (no leakage). MissMixed receives the training group labels at fit time and the test group labels at predict time, so it can apply each group's estimated random intercept (BLUP); unseen test groups fall back to the population mean.

def _reg_metrics(y, yhat):
    return {"rmse": np.sqrt(mean_squared_error(y, yhat)),
            "mae":  mean_absolute_error(y, yhat),
            "r2":   r2_score(y, yhat)}

def _clf_metrics(y, yhat, proba):
    out = {"accuracy": accuracy_score(y, yhat),
           "f1": f1_score(y, yhat, average="weighted")}
    out["auc"]   = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    out["brier"] = brier_score_loss(y, proba)
    return out

def _impute_baseline(imputer, Xtr, ytr, Xte, task):
    imp   = imputer.fit(Xtr)
    Xtr_i = imp.transform(Xtr)
    Xte_i = imp.transform(Xte)
    if task == "regression":
        est = LinearRegression().fit(Xtr_i, ytr)
        return est.predict(Xte_i), None
    est = LogisticRegression(max_iter=1000).fit(Xtr_i, ytr)
    return est.predict(Xte_i), est.predict_proba(Xte_i)[:, 1]

def _complete_case(Xtr, ytr, Xte, task):
    ok = ~np.isnan(Xtr).any(axis=1)
    Xc, yc = Xtr[ok], ytr[ok]
    # Test rows still contain NaN → mean-fill test only for prediction.
    Xte_f = np.where(np.isnan(Xte), np.nanmean(Xc, axis=0), Xte)
    if task == "regression":
        est = LinearRegression().fit(Xc, yc)
        return est.predict(Xte_f), None
    est = LogisticRegression(max_iter=1000).fit(Xc, yc)
    return est.predict(Xte_f), est.predict_proba(Xte_f)[:, 1]

def run_cv(data):
    X, y, groups, task = (data['X'], data['y'], data['groups'], data['task'])
    if task == "regression":
        splits = list(KFold(N_FOLDS, shuffle=True,
                            random_state=RNG_SEED).split(X))
        meta = [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")]
    else:
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True,
                      random_state=RNG_SEED).split(X, y))
        meta = [("accuracy", "Accuracy"), ("auc", "ROC-AUC"),
                ("f1", "F1"), ("brier", "Brier")]
    records = {m: [] for m in [
        "Complete Case + linear", "Mean + linear", "KNN + linear",
        "MICE + linear",
        "MissLinear (FIML)" if task == "regression" else "MissLogistic (FIML)",
        "MissMixed (FIML + RE)"]}
    flat_name = "MissLinear (FIML)" if task == "regression" else "MissLogistic (FIML)"
    for tr, te in splits:
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        gtr, gte = groups[tr], groups[te]
        def _store(name, yhat, proba):
            records[name].append(_reg_metrics(yte, yhat) if task == "regression"
                                 else _clf_metrics(yte, yhat, proba))
        yh, pr = _complete_case(Xtr, ytr, Xte, task); _store("Complete Case + linear", yh, pr)
        yh, pr = _impute_baseline(SimpleImputer(strategy="mean"), Xtr, ytr, Xte, task); _store("Mean + linear", yh, pr)
        yh, pr = _impute_baseline(KNNImputer(n_neighbors=5), Xtr, ytr, Xte, task); _store("KNN + linear", yh, pr)
        yh, pr = _impute_baseline(IterativeImputer(random_state=RNG_SEED), Xtr, ytr, Xte, task); _store("MICE + linear", yh, pr)
        if task == "regression":
            fm = MissLinear(compute_se=False).fit(Xtr, ytr)
            _store(flat_name, fm.predict(Xte), None)
            mm = MissMixedRegressor(compute_se=False).fit(Xtr, ytr, gtr)
            _store("MissMixed (FIML + RE)", mm.predict(Xte, groups=gte), None)
        else:
            fm = MissLogistic(compute_se=False).fit(Xtr, ytr)
            _store(flat_name, fm.predict(Xte), fm.predict_proba(Xte)[:, 1])
            mm = MissMixedClassifier(compute_se=False).fit(Xtr, ytr, gtr)
            _store("MissMixed (FIML + RE)", mm.predict(Xte, groups=gte),
                   mm.predict_proba(Xte, groups=gte)[:, 1])
    return records, meta

def summarise(records, meta):
    rows = []
    for name, folds in records.items():
        row = {"Method": name}
        for key, lbl in meta:
            vals = np.array([f[key] for f in folds], dtype=float)
            row[lbl] = f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")

print("Harness defined.")

# ## Regression Benchmark: MissMixedRegressor

print(f"Running {N_FOLDS}-fold CV; regression ...")
reg_records, reg_meta = run_cv(reg_data)
reg_table = summarise(reg_records, reg_meta)
display(reg_table)

def plot_bars(records, meta, title, save):
    names = list(records.keys())
    fig, axes = plt.subplots(1, len(meta), figsize=(5.0*len(meta), 4.2),
                             constrained_layout=True)
    if len(meta) == 1: axes = [axes]
    colors = [method_color(n) for n in names]
    for ax, (key, lbl) in zip(axes, meta):
        mus = [np.nanmean([f[key] for f in records[n]]) for n in names]
        sds = [np.nanstd([f[key] for f in records[n]]) for n in names]
        ax.barh(range(len(names)), mus, xerr=sds, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(lbl, fontsize=12); ax.set_title(lbl, fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
    fig.suptitle(title, fontsize=14)
    if save: fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig

_save = os.path.join(RESULTS_DIR, "MissMixedRegressor_bars.png") if SAVE else None
plot_bars(reg_records, reg_meta,
          "MissMixedRegressor vs baselines (grouped data)", _save)
plt.show()

# ## Classification Benchmark: MissMixedClassifier

print(f"Running {N_FOLDS}-fold CV; classification ...")
clf_records, clf_meta = run_cv(clf_data)
clf_table = summarise(clf_records, clf_meta)
display(clf_table)

_save = os.path.join(RESULTS_DIR, "MissMixedClassifier_bars.png") if SAVE else None
plot_bars(clf_records, clf_meta,
          "MissMixedClassifier vs baselines (grouped data)", _save)
plt.show()

# ## Discussion
#
# With a substantial true ICC (default `SIGMA_B=2.0`, `SIGMA_E=1.0` → ICC ≈ 0.8), a large share of the variance is between groups. The flat models (imputation baselines and FIML-without-RE) cannot represent that structure, so they leave group-level signal in the residuals. **MissMixed** estimates each group's intercept and should lead on RMSE / R² (regression) and on AUC / Brier (classification).
#
# Sweep `SIGMA_B` down toward 0 in the Configuration cell to watch the advantage shrink: when there is no real between-group variance, the random intercept adds nothing and MissMixed converges to MissLinear / MissLogistic.
#
# For a real-data version (Radon, 85 Minnesota counties) see `tests/Benchmark_Test_Suite.ipynb` → `bench_mixed`.
