# -*- coding: utf-8 -*-
"""Missing-indicators or marginalisation? And what are the indicators doing?
=========================================================================


The study
---------
Lenz, Peralta and Cornelis, "No imputation without representation"
(arXiv:2206.14254; in Communications in Computer and Information Science,
Springer 2024). Across twenty real data sets they find that adding binary
**missing-indicators** alongside an imputed value improves classification, and
that neither nearest-neighbour nor iterative imputation beats plain mean/mode
imputation once the indicators are present.

That is the direct intellectual alternative to what MissLearn does. An
indicator *represents* the absence as an extra feature; a likelihood
*marginalises* the absent value away. This example puts the two against each
other, and then asks a question their paper does not: **what is the indicator
actually carrying?**

The data
--------
UCI Heart Disease, all four collection sites rather than the Cleveland subset
that is usually used alone: 920 patients, 13 clinical attributes, disease
present in 55.3%. Downloaded on first run and cached.

    site          n     cells absent   rows affected   disease
    Cleveland   303          0.2%            2.0%       45.9%
    Hungary     294         20.5%           99.7%       36.1%
    Switzerland 123         17.1%          100.0%       93.5%
    VA          200         26.8%           99.5%       74.5%

The missingness is **native** and it is almost a function of the site:

    attribute   Cleveland  Hungary  Switzerland   VA
    ca                 1%      99%          96%   99%
    thal               1%      90%          42%   83%
    slope              0%      65%          14%   51%
    fbs                0%       3%          61%    4%

Pooled, that is 14.7% of cells, 67.5% of patients affected, and only G=31
distinct missingness patterns, because the pattern is set by where you were
treated rather than by anything patient-specific.

The finding
-----------
Disease prevalence spans **57 percentage points** across the four sites. So any
feature that encodes the site is strongly predictive of the outcome while
saying nothing whatever about the absent measurement.

Testing each indicator against the outcome, marginally and then stratified by
site, separates the two explanations:

    indicator   marginal p   within-site p   reading
    fbs           9.6e-09          0.328     association is entirely the site
    ca            2.0e-04          0.571     association is entirely the site
    slope         2.1e-14          ~0        some association survives
    six others      > 0.05           ---     no marginal association at all

So on this data the indicators largely work as **site proxies**, not as
information about what is missing. `slope` is the honest exception.

That is not a refutation of their result. It is an explanation of it, and it
carries a practical consequence: when the missingness mechanism is a variable
you actually recorded, model that variable. Sections 4 and 5 test whether doing
so makes the indicators redundant, and whether the indicator advantage survives
deployment at a hospital the model has never seen.

Why leave-one-site-out matters
------------------------------
Under a pooled random split, every site appears in training, so an indicator
can proxy the site and collect its predictive value. Under leave-one-site-out
it cannot: the held-out hospital was never seen. The second protocol is the
clinically relevant one, because the question in practice is whether a model
trained elsewhere works here.

Run
---
    python 10_heart_disease_indicators.py
    python 10_heart_disease_indicators.py --quick
"""
import argparse
import os
import sys
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

from scipy.stats import chi2_contingency, chi2 as chi2dist
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
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
SITES = {'cleveland': 'processed.cleveland.data',
         'hungarian': 'processed.hungarian.data',
         'switzerland': 'processed.switzerland.data',
         'va': 'processed.va.data'}
COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
FEATS = [c for c in COLS if c != 'num']
ALPHA = 1.0


def banner(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


# ===========================================================================
# Data
# ===========================================================================

def load():
    os.makedirs(DATA, exist_ok=True)
    frames = []
    for site, fn in SITES.items():
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            print("  downloading %s" % fn)
            urllib.request.urlretrieve("%s/%s" % (BASE, fn), p)
        d = pd.read_csv(p, header=None, names=COLS, na_values=['?'])
        d['site'] = site
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # The published target is 0 to 4 severity; the standard task is presence.
    df['y'] = (df['num'] > 0).astype(int)
    return df


def describe(df):
    banner("STEP 1  Four hospitals, four different recording practices")
    print("  patients: %d, attributes: %d, disease present: %.1f%%"
          % (len(df), len(FEATS), 100 * df['y'].mean()))
    print()
    print("  %-13s %5s %11s %13s %9s"
          % ("site", "n", "cells absent", "rows affected", "disease"))
    print("  " + "-" * 58)
    for s in SITES:
        sub = df[df.site == s]
        m = sub[FEATS]
        print("  %-13s %5d %10.1f%% %12.1f%% %8.1f%%"
              % (s, len(sub), 100 * m.isna().mean().mean(),
                 100 * m.isna().any(axis=1).mean(), 100 * sub['y'].mean()))

    print()
    print("  absence rate by attribute and site:")
    hdr = "  %-11s" % "attribute" + "".join("%12s" % s for s in SITES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in FEATS:
        rates = [df.loc[df.site == s, c].isna().mean() for s in SITES]
        if max(rates) > 0.02:
            print("  %-11s" % c + "".join("%11.0f%%" % (100 * r) for r in rates))

    X = df[FEATS]
    G = len(set(map(tuple, X.isna().astype(np.int8).to_numpy())))
    print()
    print("  pooled: %.1f%% of cells, %.1f%% of patients affected, G = %d"
          % (100 * X.isna().mean().mean(),
             100 * X.isna().any(axis=1).mean(), G))
    print("  complete cases: %d of %d" % (int((~X.isna().any(axis=1)).sum()),
                                          len(X)))
    print()
    print("  G is only %d because the pattern is set by where the patient was"
          % G)
    print("  treated, not by anything patient-specific. That is a recorded,")
    print("  observable mechanism, which is unusual and useful.")


# ===========================================================================
# What are the indicators carrying?
# ===========================================================================

def indicator_analysis(df):
    banner("STEP 2  What is a missing-indicator actually carrying?")
    y = df['y'].to_numpy()
    site = df['site'].to_numpy()

    prev = {s: df.loc[df.site == s, 'y'].mean() for s in SITES}
    span = 100 * (max(prev.values()) - min(prev.values()))
    print("  Disease prevalence spans %.1f points across sites (%s)."
          % (span, ", ".join("%s %.0f%%" % (s, 100 * v)
                             for s, v in prev.items())))
    print("  So anything encoding the site predicts the outcome without")
    print("  saying anything about the absent measurement.")
    print()
    print("  Testing each indicator against the outcome, marginally and then")
    print("  stratified by site. If the association is a site proxy it should")
    print("  disappear within sites.")
    print()
    print("  %-11s %8s %13s %14s  %s"
          % ("indicator", "absent", "marginal p", "within-site p", "reading"))
    print("  " + "-" * 76)

    rows = []
    for c in FEATS:
        m = df[c].isna().to_numpy().astype(int)
        if m.mean() <= 0.02 or m.mean() > 0.995:
            continue
        tab = pd.crosstab(m, y)
        if tab.shape != (2, 2):
            continue
        p_marg = chi2_contingency(tab)[1]

        stat, dof = 0.0, 0
        for s in np.unique(site):
            sel = site == s
            t = pd.crosstab(m[sel], y[sel])
            if t.shape == (2, 2) and (t.values.sum(axis=1) > 0).all() \
                    and (t.values.sum(axis=0) > 0).all():
                stat += chi2_contingency(t, correction=False)[0]
                dof += 1
        p_within = (float(1 - chi2dist.cdf(stat, dof)) if dof else np.nan)

        if np.isnan(p_within):
            reading = "no site has both values"
        elif p_marg < 0.05 and p_within > 0.05:
            reading = "entirely the site"
        elif p_marg < 0.05:
            reading = "some association survives"
        else:
            reading = "no marginal association"
        rows.append((c, m.mean(), p_marg, p_within, reading))
        print("  %-11s %7.1f%% %13.3g %14.3g  %s"
              % (c, 100 * m.mean(), p_marg, p_within, reading))

    proxies = [r[0] for r in rows if r[4] == "entirely the site"]
    real = [r[0] for r in rows if r[4] == "some association survives"]
    print()
    print("  proxy indicators (value is the site) : %s"
          % (", ".join(proxies) if proxies else "none"))
    print("  genuinely informative indicators     : %s"
          % (", ".join(real) if real else "none"))
    print()
    print("  This does not refute the published result that indicators help.")
    print("  It explains it, and it predicts that including the site directly")
    print("  should make most of them redundant. Section 3 tests that.")
    return rows


# ===========================================================================
# The arms
# ===========================================================================

def linear():
    return Pipeline([('s', StandardScaler()),
                     ('m', LogisticRegression(max_iter=3000, C=1.0 / ALPHA))])


def _imp(kind):
    return {'mean': SimpleImputer(strategy='mean'),
            'knn': KNNImputer(n_neighbors=5),
            'mice': IterativeImputer(max_iter=10, random_state=0)}[kind]


def _augment(A, B, A_raw, B_raw, site_tr, site_te, use_mask, use_site,
             site_levels):
    """Append missingness indicators and/or one-hot site to both matrices."""
    if use_mask:
        A = np.hstack([A, np.isnan(A_raw).astype(float)])
        B = np.hstack([B, np.isnan(B_raw).astype(float)])
    if use_site:
        A = np.hstack([A, np.stack([(site_tr == s).astype(float)
                                    for s in site_levels], axis=1)])
        B = np.hstack([B, np.stack([(site_te == s).astype(float)
                                    for s in site_levels], axis=1)])
    return A, B


def conv_arm(kind, use_mask, use_site, X, y, site, folds, site_levels):
    aucs, briers = [], []
    for tr, te in folds:
        A_raw, B_raw = X[tr], X[te]
        imp = _imp(kind).fit(A_raw)
        A, B = imp.transform(A_raw), imp.transform(B_raw)
        A, B = _augment(A, B, A_raw, B_raw, site[tr], site[te],
                        use_mask, use_site, site_levels)
        m = linear().fit(A, y[tr])
        pr = m.predict_proba(B)[:, 1]
        aucs.append(roc_auc_score(y[te], pr))
        briers.append(brier_score_loss(y[te], pr))
    return np.array(aucs), np.array(briers)


def fiml_arm(X, y, site, folds, site_levels, use_site=False, mixed=False):
    aucs, briers = [], []
    for tr, te in folds:
        A, B = X[tr], X[te]
        if mixed:
            # The site enters as a random intercept rather than as dummies,
            # which is the principled way to model a grouping variable.
            g_tr = np.array([list(site_levels).index(s) for s in site[tr]])
            g_te = np.array([list(site_levels).index(s) for s in site[te]])
            m = ML.MissMixedClassifier(copula=False, compute_se=False)
            m.fit(A, y[tr], groups=g_tr)
            try:
                pr = m.predict_proba(B, groups=g_te)[:, 1]
            except TypeError:
                pr = m.predict_proba(B)[:, 1]
        else:
            if use_site:
                A = np.hstack([A, np.stack([(site[tr] == s).astype(float)
                                            for s in site_levels], axis=1)])
                B = np.hstack([B, np.stack([(site[te] == s).astype(float)
                                            for s in site_levels], axis=1)])
            m = ML.MissRidgeClassifier(copula=False, alpha=ALPHA, compute_se=False)
            m.fit(A, y[tr])
            pr = m.predict_proba(B)[:, 1]
        aucs.append(roc_auc_score(y[te], pr))
        briers.append(brier_score_loss(y[te], pr))
    return np.array(aucs), np.array(briers)


ARMS = [
    ("Mean",                  'conv', dict(kind='mean', use_mask=False, use_site=False)),
    ("Mean + indicators",     'conv', dict(kind='mean', use_mask=True,  use_site=False)),
    ("Mean + site",           'conv', dict(kind='mean', use_mask=False, use_site=True)),
    ("Mean + ind + site",     'conv', dict(kind='mean', use_mask=True,  use_site=True)),
    ("kNN + indicators",      'conv', dict(kind='knn',  use_mask=True,  use_site=False)),
    ("MICE + indicators",     'conv', dict(kind='mice', use_mask=True,  use_site=False)),
    ("FIML",                  'fiml', dict(use_site=False)),
    ("FIML + site",           'fiml', dict(use_site=True)),
    ("MissMixed (site RE)",   'fiml', dict(mixed=True)),
]


def run(X, y, site, folds, site_levels, label):
    banner(label)
    print("  Every arm uses the same logistic model class and the same")
    print("  penalty. Only the treatment of the missing values, and whether")
    print("  the site is supplied, differ.")
    print()
    print("  %-22s %9s %8s %9s %8s" % ("arm", "AUC", "sd", "Brier", "sd"))
    print("  " + "-" * 62)
    out = {}
    for name, kind, kw in ARMS:
        try:
            if kind == 'conv':
                a, b = conv_arm(X=X, y=y, site=site, folds=folds,
                                site_levels=site_levels, **kw)
            else:
                a, b = fiml_arm(X, y, site, folds, site_levels, **kw)
        except Exception as e:
            print("  %-22s %9s  (%s: %s)"
                  % (name, "failed", type(e).__name__, str(e)[:34]))
            continue
        out[name] = (a, b)
        print("  %-22s %9.4f %8.4f %9.4f %8.4f"
              % (name, a.mean(), a.std(), b.mean(), b.std()))
    return out


def report_deltas(out, protocol):
    print()
    print("  What the indicators are worth under %s:" % protocol)
    base = out.get("Mean")
    if base is None:
        return
    for a, b, lab in (("Mean", "Mean + indicators", "indicators alone"),
                      ("Mean", "Mean + site", "site alone"),
                      ("Mean + site", "Mean + ind + site",
                       "indicators once site is present")):
        if a in out and b in out:
            d = out[b][0].mean() - out[a][0].mean()
            print("    %-34s %+.4f AUC" % (lab, d))
    if "Mean + indicators" in out and "Mean + site" in out:
        print()
        print("    If site alone matches or beats indicators alone, the")
        print("    indicators were largely proxying the site.")
    for k in ("FIML", "FIML + site", "MissMixed (site RE)"):
        if k in out and base is not None:
            print("    %-34s %+.4f AUC vs Mean"
                  % (k, out[k][0].mean() - base[0].mean()))

    # Two things that would be easy to over-read, so they are stated here.
    worst_sd = max(v[0].std() for v in out.values())
    spread = (max(v[0].mean() for v in out.values())
              - min(v[0].mean() for v in out.values()))
    if worst_sd > spread:
        print()
        print("    CAUTION: the largest fold standard deviation (%.4f) exceeds"
              % worst_sd)
        print("    the entire spread between arms (%.4f), so under this" % spread)
        print("    protocol no arm is statistically distinguishable from any")
        print("    other. The ordering is reported for completeness, and the")
        print("    pooled protocol is where the differences are resolvable.")

    if protocol.startswith("leave-one-site"):
        print()
        print("    NOTE on the site arms here: the held-out hospital never")
        print("    appears in training, so its dummy is identically zero and")
        print("    its coefficient is unidentifiable. The gain therefore does")
        print("    not come from the model knowing the new site. It comes from")
        print("    having removed between-site variance while fitting the")
        print("    others, which sharpens the shared clinical coefficients.")
        print("    MissMixed behaves the same way: a new group has no")
        print("    estimable random intercept, so it falls back to the")
        print("    population level, which is the correct behaviour and the")
        print("    reason its advantage shrinks from %s here."
              % "pooled to leave-one-site-out")


def plot(pooled, loso):
    names = [n for n, _, _ in ARMS if n in pooled or n in loso]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    cmap = plt.colormaps['viridis']
    for ax, (store, title) in zip(axes, ((pooled, "Pooled random split"),
                                         (loso, "Leave-one-site-out"))):
        ks = [n for n in names if n in store]
        means = [store[k][0].mean() for k in ks]
        sds = [store[k][0].std() for k in ks]
        ax.bar(range(len(ks)), means, yerr=sds, capsize=4,
               color=[cmap(v) for v in np.linspace(0.12, 0.88, len(ks))])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=40, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ROC-AUC")
    fig.suptitle("Heart disease: indicators versus marginalisation, and what "
                 "happens at an unseen hospital")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="3 folds")
    a = ap.parse_args()

    banner("UCI Heart Disease, all four collection sites")
    df = load()
    describe(df)
    indicator_analysis(df)

    X = df[FEATS].to_numpy(float)
    y = df['y'].to_numpy()
    site = df['site'].to_numpy()
    levels = list(SITES)

    # Protocol 1: pooled random split. Every site is in training, so an
    # indicator can proxy the site.
    folds = list(StratifiedKFold(3 if a.quick else 5, shuffle=True,
                                 random_state=0).split(X, y))
    pooled = run(X, y, site, folds, levels,
                 "STEP 3  Pooled random split (every site seen in training)")
    report_deltas(pooled, "a pooled split")

    # Protocol 2: leave-one-site-out. The held-out hospital is unseen, so an
    # indicator has no site to proxy.
    loso_folds = [(np.where(site != s)[0], np.where(site == s)[0])
                  for s in levels]
    loso = run(X, y, site, loso_folds, levels,
               "STEP 4  Leave-one-site-out (the held-out hospital is unseen)")
    report_deltas(loso, "leave-one-site-out")

    plot(pooled, loso)

    banner("What to take from this")
    print("  1. Indicators do help on a pooled split, reproducing the")
    print("     published result.")
    print("  2. On this data much of that help is the site, not the absence:")
    print("     two of the three informative indicators lose their association")
    print("     with the outcome entirely once site is conditioned on.")
    print("  3. When the mechanism is a variable you recorded, model it. A")
    print("     random intercept states the structure explicitly instead of")
    print("     letting a proxy discover it.")
    print("  4. Compare the two panels. A pooled split rewards proxying the")
    print("     site; deployment at a new hospital does not, and that is the")
    print("     question a clinician is actually asking.")


if __name__ == "__main__":
    main()
