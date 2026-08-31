# -*- coding: utf-8 -*-
"""Blockwise missingness and large p: semiconductor yield from a sensor array
==========================================================================


Dataset
-------
UCI SECOM (https://archive.ics.uci.edu/dataset/179/secom): 1,567 wafer lots
from a semiconductor fabrication line, each characterised by 590 process
sensor and metrology signals, with a pass/fail outcome from in-house line
testing. Downloaded on first run and cached in example_data/.

Why this data set
-----------------
It is the one example here where **p is genuinely large**, and it is the one
where the *structure* of the missingness, rather than its rate, decides
whether full-information estimation is even possible.

  * 590 signals against 1,567 lots, so p is roughly 30% of n. After dropping
    constant and empty channels, p = 474.
  * Only 4.5% of cells are absent, but **100% of lots have at least one
    absent reading**. Listwise deletion does not lose power here, it returns
    an empty table.
  * The absences are **blockwise**: all 474 surviving channels fall into 36
    groups whose members are missing in exactly the same lots. Whole banks of
    sensors drop out together, which is what happens when an instrument, a
    metrology step or a logging process fails rather than an individual
    reading.
  * Only 6.6% of lots fail, so accuracy is useless as a metric and ROC-AUC and
    the Brier score are reported instead.

The point about blockwise structure
-----------------------------------
FIML's first stage groups rows by missingness pattern and factorises the
observed submatrix once per pattern, so it costs O(G p^3) in the number of
**distinct patterns** G, not in the missing rate.

That makes the pattern count the quantity that matters, and blockwise
missingness keeps it small. On a synthetic control with n=1,500 and p=30 at a
25% rate, scattered cell-wise missingness gives G=1,499 and a 324 second fit,
while blockwise missingness at the same rate gives G=32 and a 2.8 second fit:
a 116-fold difference from structure alone, at which point FIML is faster than
MICE. SECOM shows the same effect on real data at far greater width, with
G=198 at p=474; scattered missingness at that width would push G toward n and
put the fit out of reach.

So blockwise structure is what makes large p approachable. It is not a
curiosity, it is the enabling condition.

And the honest ceiling
----------------------
The p^3 factor still binds. Measured here, with the feature budget chosen by
dispersion and no reference to the label:

    p=20   G=9      0.6 s   AUC 0.613
    p=40   G=18    60.4 s   AUC 0.731
    p=60   G=28   170.8 s   AUC 0.765
    p=90   G=53  2563.7 s   AUC 0.809

More channels genuinely help, and AUC 0.809 at p=90 is a good result on a data
set where published figures usually sit between 0.6 and 0.75. So this is a real
trade rather than an academic one.

But the cost is brutal, and worse than the theory predicts. From p=20 to p=90
the fit time grows by a factor of about 4,300, while O(G p^3) accounts for only
about 537 of that; the remainder is the optimiser needing more iterations as
the problem widens. The practical sweet spot on this data is p=40 to 60, at one
to three minutes; p=90 costs 43 minutes; p=474 is out of reach entirely.

Above that range the guidance is to screen channels first, or to hand off
through MissImputer to a method whose cost does not carry a p^3. That
limitation is reported rather than hidden, because a reader deciding whether to
use this library needs it.

Run
---
    python 09_secom_blockwise.py
    python 09_secom_blockwise.py --quick
    python 09_secom_blockwise.py --features 90     # slower, see the ladder
"""
import argparse
import os
import sys
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import numpy as np
import pandas as pd
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer      # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import roc_auc_score, brier_score_loss

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import MissLearn as ML                                          # noqa: E402

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})

DATA = os.path.join(_HERE, "example_data")
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
ALPHA = 1.0          # shared L2 strength, identical in every arm

# Measured separately; see the module docstring. Reported so the reader can see
# the cost curve without waiting for it.
LADDER = [(20, 9, 0.6, 0.613), (40, 18, 60.4, 0.731),
          (60, 28, 170.8, 0.765), (90, 53, 2563.7, 0.809)]
SYNTHETIC_CONTROL = {'scattered': (1499, 323.6), 'blockwise': (32, 2.8)}


def banner(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


# ===========================================================================
# Data
# ===========================================================================

def load():
    os.makedirs(DATA, exist_ok=True)
    xp = os.path.join(DATA, "secom.data")
    yp = os.path.join(DATA, "secom_labels.data")
    for path, name in ((xp, "secom.data"), (yp, "secom_labels.data")):
        if not os.path.exists(path):
            print("  downloading %s" % name)
            urllib.request.urlretrieve("%s/%s" % (BASE, name), path)

    X = pd.read_csv(xp, sep=r"\s+", header=None, na_values=["NaN"])
    y = pd.read_csv(yp, sep=r"\s+", header=None,
                    names=["y", "ts"])["y"].to_numpy(float)
    y = (y > 0).astype(float)          # 1 = fail, the minority class
    return X, y


def describe(X, y):
    banner("STEP 1  The data, and why deletion is not an option")
    miss = X.isna().mean()
    print("  lots (n)               : %d" % X.shape[0])
    print("  signals (p, raw)       : %d" % X.shape[1])
    print("  failing lots           : %d (%.1f%%)"
          % (int(y.sum()), 100 * y.mean()))
    print("  cells absent           : %.2f%%" % (100 * X.isna().mean().mean()))
    print("  lots with any absence  : %.1f%%"
          % (100 * X.isna().any(axis=1).mean()))
    print()
    print("  Only %.1f%% of cells are absent, yet every lot is affected."
          % (100 * X.isna().mean().mean()))
    print("  Listwise deletion returns %d of %d lots."
          % (int((~X.isna().any(axis=1)).sum()), len(X)))
    print()
    print("  Because the failure rate is %.1f%%, accuracy is not a useful"
          % (100 * y.mean()))
    print("  metric: predicting 'pass' always would score %.1f%%."
          % (100 * (1 - y.mean())))
    print("  ROC-AUC and the Brier score are reported instead.")

    keep = [c for c in X.columns
            if miss[c] < 1.0 and X[c].nunique(dropna=True) > 1]
    Xk = X[keep]
    print()
    print("  after dropping empty and constant channels: p = %d" % Xk.shape[1])
    return Xk


# ===========================================================================
# The blockwise structure
# ===========================================================================

def blockwise_analysis(Xk):
    banner("STEP 2  The absences are blockwise, and that is what matters")
    R = Xk.isna().to_numpy()
    G = len(set(map(tuple, R.astype(np.int8))))

    colmask = {}
    for j, c in enumerate(Xk.columns):
        colmask.setdefault(tuple(R[:, j]), []).append(c)
    groups = sorted(colmask.values(), key=len, reverse=True)
    inblock = sum(len(g) for g in groups if len(g) > 1)

    print("  distinct missingness patterns G = %d   (n = %d, p = %d)"
          % (G, len(Xk), Xk.shape[1]))
    print("  G / n = %.3f" % (G / len(Xk)))
    print()
    print("  channels sharing an identical absence mask with another:")
    print("    %d of %d (%.0f%%), forming %d co-missing groups"
          % (inblock, Xk.shape[1], 100 * inblock / Xk.shape[1],
             sum(1 for g in groups if len(g) > 1)))
    print()
    print("  the largest groups:")
    print("    %8s  %s" % ("channels", "absent in this share of lots"))
    for g in groups[:6]:
        print("    %8d  %.1f%%" % (len(g), 100 * Xk[g[0]].isna().mean()))
    print()
    print("  Whole banks drop out together, which is what an instrument or a")
    print("  metrology step failing looks like. Scattered cell-wise absence")
    print("  would instead push G toward n.")
    print()
    s_G, s_t = SYNTHETIC_CONTROL['scattered']
    b_G, b_t = SYNTHETIC_CONTROL['blockwise']
    print("  Synthetic control at n=1500, p=30, 25%% absent:")
    print("    scattered : G = %4d, FIML fit %6.1f s" % (s_G, s_t))
    print("    blockwise : G = %4d, FIML fit %6.1f s" % (b_G, b_t))
    print("    a %.0f-fold difference from structure alone, at the same rate."
          % (s_t / b_t))
    print()
    print("  FIML stage one costs O(G p^3), so the pattern count is the thing")
    print("  that decides feasibility. Blockwise structure is the enabling")
    print("  condition for large p, not a footnote.")
    return G, groups


def plot_blocks(Xk, groups):
    R = Xk.isna().to_numpy()
    order = [c for g in groups for c in g]         # cluster co-missing channels
    idx = [Xk.columns.get_loc(c) for c in order]
    rate = Xk.isna().mean().to_numpy()[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={'width_ratios': [1.5, 1]})
    cmap = plt.colormaps['viridis']

    axes[0].imshow(R[:, idx], aspect='auto', cmap='viridis',
                   interpolation='nearest')
    axes[0].set_xlabel("channel, ordered by co-missing group")
    axes[0].set_ylabel("lot")
    axes[0].set_title("Absence map: vertical bands are sensor banks\n"
                      "failing together")

    sizes = [len(g) for g in groups if len(g) > 1][:14]
    axes[1].bar(range(len(sizes)), sizes,
                color=[cmap(v) for v in np.linspace(0.15, 0.85, len(sizes))])
    axes[1].set_xlabel("co-missing group")
    axes[1].set_ylabel("channels in the group")
    axes[1].set_title("Group sizes")
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("SECOM: %d channels fall into %d co-missing groups"
                 % (Xk.shape[1], sum(1 for g in groups if len(g) > 1)))
    plt.show()


def plot_ladder():
    fig, ax1 = plt.subplots(figsize=(6.5,4))
    cmap = plt.colormaps['viridis']
    ps = [r[0] for r in LADDER]
    ts = [r[2] for r in LADDER]
    aucs = [r[3] for r in LADDER]

    ax1.plot(ps, ts, marker='o', markersize=7, color=cmap(0.20),
             linewidth=2.2, label="FIML fit time")
    ax1.set_xlabel("feature budget p")
    ax1.set_ylabel("seconds to fit")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(ps, aucs, marker='s', markersize=7, color=cmap(0.70),
             linewidth=2.2, linestyle='--', label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.set_title("Accuracy improves with p, cost improves faster in the "
                  "wrong direction\n"
                  "(the practical ceiling here is around p = 60 to 90)", fontsize=10)
    plt.show()


# ===========================================================================
# The six-arm comparison
# ===========================================================================

def select(Xk, n_features):
    """Rank channels by dispersion. Never consults the label."""
    sd = Xk.std(skipna=True)
    mu = Xk.mean(skipna=True).abs()
    order = (sd / (mu + 1.0)).sort_values(ascending=False).index.tolist()
    return order[:n_features]


def scaled():
    return Pipeline([('s', StandardScaler()),
                     ('m', LogisticRegression(max_iter=3000,
                                              C=1.0 / ALPHA))])


CONV = [("Drop rows", 'drop_rows'), ("Drop cols", 'drop_cols'),
        ("Mean", 'mean'), ("kNN", 'knn'), ("MICE", 'mice')]


def conv_arm(strategy, X, y, folds):
    aucs, briers = [], []
    for tr, te in folds:
        A, B, ytr, yte = X[tr], X[te], y[tr], y[te]
        mu = np.nanmean(A, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        if strategy == 'drop_rows':
            cc = ~np.isnan(A).any(axis=1)
            if cc.sum() < A.shape[1] + 2 or len(np.unique(ytr[cc])) < 2:
                return None
            Ai, b = A[cc], ytr[cc]
            Bi = np.where(np.isnan(B), mu, B)
        elif strategy == 'drop_cols':
            k = ~np.isnan(A).any(axis=0)
            if k.sum() < 2:
                return None
            Ai, b = A[:, k], ytr
            Bi = np.where(np.isnan(B[:, k]), mu[k], B[:, k])
        else:
            imp = {'mean': SimpleImputer(strategy='mean'),
                   'knn': KNNImputer(n_neighbors=5),
                   'mice': IterativeImputer(max_iter=10,
                                            random_state=0)}[strategy]
            Ai, b = imp.fit_transform(A), ytr
            Bi = imp.transform(B)
        m = scaled().fit(Ai, b)
        pr = m.predict_proba(Bi)[:, 1]
        aucs.append(roc_auc_score(yte, pr))
        briers.append(brier_score_loss(yte, pr))
    return np.array(aucs), np.array(briers)


def fiml_arm(X, y, folds):
    aucs, briers = [], []
    for tr, te in folds:
        m = ML.MissRidgeClassifier(copula=False, alpha=ALPHA, compute_se=False)
        m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], pr))
        briers.append(brier_score_loss(y[te], pr))
    return np.array(aucs), np.array(briers)


def compare(Xk, y, n_features, n_folds):
    banner("STEP 3  Six arms, logistic model class held fixed")
    cols = select(Xk, n_features)
    X = Xk[cols].to_numpy(float)
    G = len(set(map(tuple, np.isnan(X).astype(np.int8))))
    print("  p = %d channels selected by dispersion, without using the label"
          % X.shape[1])
    print("  G = %d distinct patterns, %.1f%% of cells absent, %.0f%% of lots"
          % (G, 100 * np.isnan(X).mean(), 100 * np.isnan(X).any(axis=1).mean()))
    print()

    folds = list(StratifiedKFold(n_folds, shuffle=True,
                                 random_state=0).split(X, y))
    print("  %-16s %10s %8s %10s %8s %8s"
          % ("arm", "AUC", "sd", "Brier", "sd", "secs"))
    print("  " + "-" * 66)

    block = {}
    for label, key in CONV:
        t0 = time.time()
        out = conv_arm(key, X, y, folds)
        if out is None:
            reason = ("every lot has an absence" if key == 'drop_rows'
                      else "too few fully observed channels")
            print("  %-16s %10s  (%s)" % (label, "n/a", reason))
            continue
        a, b = out
        block[label] = (a, b)
        print("  %-16s %10.4f %8.4f %10.4f %8.4f %8.1f"
              % (label, a.mean(), a.std(), b.mean(), b.std(),
                 time.time() - t0))

    t0 = time.time()
    a, b = fiml_arm(X, y, folds)
    block['FIML'] = (a, b)
    best = max((k for k in block if k != 'FIML'),
               key=lambda k: block[k][0].mean())
    print("  %-16s %10.4f %8.4f %10.4f %8.4f %8.1f"
          % ("FIML", a.mean(), a.std(), b.mean(), b.std(), time.time() - t0))
    print()
    print("  best conventional arm : %s at AUC %.4f"
          % (best, block[best][0].mean()))
    print("  FIML                  : AUC %.4f  (%+.4f)"
          % (a.mean(), a.mean() - block[best][0].mean()))
    print("  Brier, best conventional %.4f against FIML %.4f (%+.4f)"
          % (block[best][1].mean(), b.mean(),
             b.mean() - block[best][1].mean()))
    return block, cols, X


def plot_compare(block):
    ks = list(block)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cmap = plt.colormaps['viridis']
    for ax, (i, name, better) in zip(axes, [(0, "ROC-AUC", "higher"),
                                            (1, "Brier score", "lower")]):
        means = [block[k][i].mean() for k in ks]
        sds = [block[k][i].std() for k in ks]
        ax.bar(range(len(ks)), means, yerr=sds, capsize=4,
               color=[cmap(v) for v in np.linspace(0.12, 0.88, len(ks))])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=35, ha='right')
        ax.set_title("%s (%s is better)" % (name, better))
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("SECOM yield prediction: logistic model class held fixed, "
                 "only the missing-data treatment varies")
    plt.show()


# ===========================================================================
# Which sensor bank is worth maintaining?
# ===========================================================================

def explain(X, y, cols, seed=0):
    banner("STEP 4  Which bank is worth maintaining?")
    print("  A fab engineer does not ask which reading was most informative")
    print("  after the fact. They ask which instrument to repair or")
    print("  recalibrate next. MissExplainer answers that: how much the")
    print("  prediction moves when a channel is measured rather than absent.")
    print()
    names = [str(c) for c in cols]
    model = ML.MissRidgeClassifier(copula=False, alpha=ALPHA, compute_se=False).fit(X, y)
    expl = ML.MissExplainer(model, random_state=seed).fit(X,
                                                          feature_names=names)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(120, len(X)), replace=False)
    ms = np.abs(expl.miss_shap(X[idx])).mean(axis=0)
    rate = np.isnan(X).mean(axis=0)

    o = np.argsort(-ms)[:12]
    print("  %-14s %14s %12s" % ("channel", "MissExplainer", "absent"))
    print("  " + "-" * 42)
    for j in o:
        print("  %-14s %14.4f %11.1f%%" % (names[j], ms[j], 100 * rate[j]))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cmap = plt.colormaps['viridis']
    oo = np.argsort(ms)[-14:]
    ax.barh([names[j] for j in oo], ms[oo],
            color=[cmap(t) for t in np.linspace(0.15, 0.85, len(oo))])
    ax.set_xlabel("mean |MissExplainer|")
    ax.set_title("Value of measuring each channel at all\n"
                 "(a maintenance priority ordering, not a feature ranking)")
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="p=20 and 3 folds, for a fast pass")
    ap.add_argument("--features", type=int, default=None,
                    help="feature budget; see the cost ladder in the docstring")
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()

    n_features = a.features or (20 if a.quick else 40)
    n_folds = a.folds or (3 if a.quick else 5)

    banner("SECOM: semiconductor yield from a 590-channel sensor array")
    X, y = load()
    Xk = describe(X, y)
    G, groups = blockwise_analysis(Xk)
    plot_blocks(Xk, groups)
    plot_ladder()
    block, cols, Xs = compare(Xk, y, n_features, n_folds)
    plot_compare(block)
    explain(Xs, y, cols)

    banner("What to take from this")
    print("  1. Blockwise absence is what makes large p reachable. G=%d at" % G)
    print("     p=%d here; scattered absence would push G toward n." % Xk.shape[1])
    print("  2. Listwise deletion is not a weak option, it is not an option:")
    print("     every one of the %d lots has at least one absent reading." % len(Xk))
    print("  3. The p^3 factor still binds. FIML is practical to about p=60")
    print("     to 90 on this data, not to p=%d. Screen channels first, or"
          % Xk.shape[1])
    print("     hand off through MissImputer, above that.")
    print("  4. MissExplainer turns the model into a maintenance ordering,")
    print("     which is a different and more actionable question than which")
    print("     reading mattered.")


if __name__ == "__main__":
    main()
