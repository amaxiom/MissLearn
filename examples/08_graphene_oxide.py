# -*- coding: utf-8 -*-
"""Two labels, one missingness pattern: which prediction is more at risk?
======================================================================


Dataset
-------
Graphene oxide structures from CSIRO (doi:10.25919/5e30b45f9852c), cached as
example_data/graphene_oxide.csv: 1,617 relaxed structures described by 465
composition, distortion and bond-topology descriptors, with two DFT labels,
Formation_energy and Fermi_energy.

The question
------------
Both labels are fully observed. 9.9% of the descriptor cells are not. Is that
missingness more critical to one label than to the other, and why?

What this example finds
-----------------------
1. The native missingness is **structural absence**, not measurement failure.
   A bond-motif descriptor is absent exactly when the motif does not occur in
   that structure, verified below across all 47 motif value/fraction pairs.
   Imputing such a column invents a bond length for a bond that is not there.

2. Because every structure lacks some motif, **100% of rows are incomplete**.
   Listwise deletion does not merely lose power here, it returns nothing.

3. Those columns carry **no incremental information** once composition is
   known, so the correct treatment is neither imputation nor marginalisation
   but removal, which is what MissRecommender advises at its 60% threshold.
   This is the example's main lesson: a mechanism can be formally MAR and
   still make imputation the wrong choice.

4. When missingness instead strikes the *informative* composition block,
   **Fermi_energy degrades about 2.7 times more than Formation_energy**.

5. The reason is **compositional closure**. C, H and O concentrations sum to
   exactly 1, so any one of them is recoverable from the other two, and FIML
   picks that up through the covariance structure. Formation_energy is a
   stoichiometric identity in composition (R2 = 0.99999548 from C/H/O alone)
   and is therefore nearly immune; Fermi_energy depends on the values
   themselves and is not protected.

Run
---
    python 08_graphene_oxide.py
    python 08_graphene_oxide.py --quick
"""
import argparse
import os
import sys
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

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.experimental import enable_iterative_imputer      # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import r2_score, mean_squared_error

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import MissLearn as ML                                          # noqa: E402

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})

CSV = os.path.join(_HERE, "example_data", "graphene_oxide.csv")

COMPOSITION = ['C_concentration', 'H_concentration', 'O_concentration']
CHEMISTRY = ['defects_concentration', 'agent_per_C', 'ether_concentration',
             'hydroxyl_concentration', 'carboxyl_concentration']
DISTORTION = ['max_oop', 'mae_oop', 'std_oop', 'rmse_oop', 'residual_oop']
INFORMATIVE = COMPOSITION + CHEMISTRY + DISTORTION
LABELS = ['Formation_energy', 'Fermi_energy']

# Shared L2 strength for the step 5 comparison. Compositional closure makes the
# design matrix singular, so both arms have to be penalised, and they have to be
# penalised identically or the comparison measures conditioning instead of the
# missing-data treatment.
ALPHA = 1.0


def banner(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


# ===========================================================================
# Step 1: what kind of missingness is this?
# ===========================================================================

def step1_mechanism(df):
    banner("STEP 1  What kind of missingness is this?")
    miss = df.isna().mean()
    print("  %d structures, %d descriptors" % df.shape)
    print("  descriptors with any hole : %d" % int((miss > 0).sum()))
    print("  cells missing             : %.2f%%" % (100 * df.isna().mean().mean()))
    print("  rows with at least one    : %.1f%%"
          % (100 * df.isna().any(axis=1).mean()))
    print("  labels missing            : %s"
          % ", ".join("%s %.0f%%" % (l, 100 * miss[l]) for l in LABELS))
    print()
    print("  Listwise deletion therefore returns %d of %d structures."
          % (int((~df.isna().any(axis=1)).sum()), len(df)))

    # The hypothesis: a motif's value is absent exactly when the motif is
    # absent, which the file records separately as a fraction of zero.
    pairs = [(c, c[:-len('_mean_value')] + '_total_frac')
             for c in df.columns if c.endswith('_mean_value')
             and c[:-len('_mean_value')] + '_total_frac' in df.columns]
    agree = [float((df[v].isna() == (df[f].fillna(0) == 0)).mean())
             for v, f in pairs]
    print()
    print("  Testing structural absence over %d motif pairs:" % len(pairs))
    print("    a descriptor is NaN exactly when its motif fraction is zero")
    print("    agreement: mean %.4f, min %.4f, %d of %d pairs exact"
          % (np.mean(agree), np.min(agree),
             sum(1 for a in agree if a == 1.0), len(pairs)))
    if np.min(agree) == 1.0:
        print()
        print("  Exact for every pair. The value is missing because the bond")
        print("  motif does not exist in that structure, so there is no bond")
        print("  length or angle to record. Mean-imputing such a column")
        print("  assigns a geometry to a bond that is not present.")

    top = miss[miss > 0].sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    cmap = plt.colormaps['viridis']
    k = min(18, len(top))
    axes[0].barh([c[:34] for c in top.index[:k]][::-1],
                 (100 * top.values[:k])[::-1],
                 color=[cmap(v) for v in np.linspace(0.15, 0.85, k)])
    axes[0].set_xlabel("missing (%)")
    axes[0].set_title("Most-absent descriptors")
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].hist(100 * miss[miss > 0].values, bins=24,
                 color=cmap(0.45), edgecolor='white')
    axes[1].set_xlabel("missing (%) per descriptor")
    axes[1].set_ylabel("number of descriptors")
    axes[1].set_title("Absence is bimodal: rare motifs are almost never seen")
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_axisbelow(True)
    fig.suptitle("Graphene oxide: missingness is structural absence")
    plt.show()
    return pairs


# ===========================================================================
# Step 2: is Formation_energy even a modelling problem?
# ===========================================================================

def step2_identity(df):
    banner("STEP 2  Are the two labels the same kind of problem?")
    Xc = df[COMPOSITION].to_numpy(float)
    print("  Regressing each label on C/H/O concentration alone:")
    print()
    print("  %-18s %14s %14s %12s" % ("label", "R2", "max |resid|", "sd(label)"))
    print("  " + "-" * 62)
    for lab in LABELS:
        yv = df[lab].to_numpy(float)
        lr = LinearRegression().fit(Xc, yv)
        res = yv - lr.predict(Xc)
        print("  %-18s %14.8f %14.3e %12.4f"
              % (lab, r2_score(yv, lr.predict(Xc)), np.abs(res).max(), yv.std()))
    print()
    print("  Formation_energy is an identity, not a prediction: composition")
    print("  determines it to eight significant figures. Fermi_energy leaves")
    print("  real residual structure to model.")

    # Compositional closure, which becomes the explanation in step 4.
    s = df[COMPOSITION].sum(axis=1)
    print()
    print("  Compositional closure: C+H+O = %.6f to %.6f (sd %.2e)"
          % (s.min(), s.max(), s.std()))
    if s.std() < 1e-6:
        err = max(np.abs((s.mean() - df[[c for c in COMPOSITION if c != t]]
                          .sum(axis=1)) - df[t]).max() for t in COMPOSITION)
        print("  The three sum to a constant, so any one is recoverable from")
        print("  the other two (max error %.1e). Remember this for step 4." % err)


# ===========================================================================
# Step 3: do the incomplete descriptors carry anything?
# ===========================================================================

def _cv_fiml(X, yv, folds):
    out = []
    for tr, te in folds:
        m = ML.MissLinear(copula=False, compute_se=False).fit(X[tr], yv[tr])
        out.append(r2_score(yv[te], m.predict(X[te])))
    return float(np.mean(out))


def step3_worth(df, folds):
    banner("STEP 3  Do the incomplete descriptors add anything?")
    miss = df.isna().mean()
    motif = sorted([c for c in df.columns if c.endswith('_mean_value')
                    and 0.02 < miss[c] < 0.85], key=lambda c: miss[c])
    print("  complete informative block : %d descriptors" % len(INFORMATIVE))
    print("  incomplete motif block     : %d descriptors, %.0f%% to %.0f%% absent"
          % (len(motif), 100 * miss[motif].min(), 100 * miss[motif].max()))
    print()
    X_comp = df[INFORMATIVE].to_numpy(float)
    X_both = df[INFORMATIVE + motif].to_numpy(float)
    print("  %-18s %16s %16s %10s"
          % ("label", "complete only", "plus motifs", "gain"))
    print("  " + "-" * 64)
    for lab in LABELS:
        yv = df[lab].to_numpy(float)
        a = _cv_fiml(X_comp, yv, folds)
        b = _cv_fiml(X_both, yv, folds)
        print("  %-18s %16.4f %16.4f %+10.4f" % (lab, a, b, b - a))
    print()
    print("  Adding them changes nothing. They are redundant given composition,")
    print("  which is why the right treatment is removal: imputing them would")
    print("  invent chemistry for no gain, and marginalising them would spend")
    print("  parameters on columns with no information.")
    return motif


# ===========================================================================
# Step 4: which label is more vulnerable?
# ===========================================================================

def _mar(X, cols, rate, driver, seed=42):
    """MAR: masking probability depends on an observed column not being masked."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    hi = X[:, driver] > np.median(X[:, driver])
    for j in cols:
        pr = np.where(hi, min(1.0, rate * 4 / 3), min(1.0, rate * 2 / 3))
        Xm[rng.random(len(X)) < pr, j] = np.nan
    return Xm


def step4_vulnerability(df, folds, rate=0.30):
    banner("STEP 4  Which label is more vulnerable to missingness?")
    print("  The native holes sit in redundant columns, so to answer the")
    print("  question we inject MAR missingness into the INFORMATIVE block")
    print("  instead, one sub-block at a time, at %.0f%% per column.\n" % (100 * rate))

    X0 = df[INFORMATIVE].to_numpy(float)
    driver = INFORMATIVE.index('mae_oop')
    blocks = [
        ("nothing masked", []),
        ("composition (C, H, O)", [INFORMATIVE.index(c) for c in COMPOSITION]),
        ("out-of-plane distortion",
         [INFORMATIVE.index(c) for c in DISTORTION if c != 'mae_oop']),
    ]

    print("  %-26s %18s %18s" % ("masked block", LABELS[0], LABELS[1]))
    print("  " + "-" * 66)
    base, results = {}, {}
    for name, cols in blocks:
        Xm = X0 if not cols else _mar(X0, cols, rate, driver)
        vals = {}
        for lab in LABELS:
            vals[lab] = _cv_fiml(Xm, df[lab].to_numpy(float), folds)
            if not cols:
                base[lab] = vals[lab]
        results[name] = vals
        print("  %-26s %18.4f %18.4f"
              % (name, vals[LABELS[0]], vals[LABELS[1]]))

    print()
    print("  Degradation from intact:")
    print("  %-26s %18s %18s" % ("masked block", LABELS[0], LABELS[1]))
    print("  " + "-" * 66)
    deg = {}
    for name, cols in blocks[1:]:
        d = {l: results[name][l] - base[l] for l in LABELS}
        deg[name] = d
        print("  %-26s %+18.4f %+18.4f" % (name, d[LABELS[0]], d[LABELS[1]]))

    comp = deg["composition (C, H, O)"]
    if abs(comp[LABELS[0]]) > 1e-9:
        ratio = abs(comp[LABELS[1]]) / abs(comp[LABELS[0]])
        print()
        print("  Losing composition costs Fermi_energy %.1fx what it costs"
              % ratio)
        print("  Formation_energy. The answer to the question is Fermi_energy,")
        print("  and the reason is the closure identity from step 2: with")
        print("  C+H+O fixed, a masked concentration is recoverable from the")
        print("  other two, and Formation_energy needs nothing else.")
        print("  Fermi_energy depends on the values themselves.")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cmap = plt.colormaps['viridis']
    names = list(deg)
    xs = np.arange(len(names))
    w = 0.36
    for i, lab in enumerate(LABELS):
        ax.bar(xs + (i - 0.5) * w, [abs(deg[n][lab]) for n in names], w,
               color=cmap(0.25 + 0.45 * i), label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"loss in $R^2$ from masking")
    ax.set_title("Same missingness, two labels: only one of them cares\n"
                 "(MAR at %.0f%% per masked column)" % (100 * rate))
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.show()
    return deg


# ===========================================================================
# Step 5: the six-arm comparison, per label
# ===========================================================================

def scaled(f):
    return lambda: Pipeline([('s', StandardScaler()), ('m', f())])


CONV = [("Drop rows", 'drop_rows'), ("Drop cols", 'drop_cols'),
        ("Mean imputation", 'mean'), ("kNN imputation", 'knn'),
        ("MICE", 'mice')]


def _conv(strategy, model_fn, X, yv, folds):
    r2s = []
    for tr, te in folds:
        Xtr, ytr, Xte, yte = X[tr], yv[tr], X[te], yv[te]
        mu = np.nanmean(Xtr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        if strategy == 'drop_rows':
            cc = ~np.isnan(Xtr).any(axis=1)
            if cc.sum() < Xtr.shape[1] + 2:
                return None
            A, b, B = Xtr[cc], ytr[cc], np.where(np.isnan(Xte), mu, Xte)
        elif strategy == 'drop_cols':
            keep = ~np.isnan(Xtr).any(axis=0)
            if keep.sum() == 0:
                return None
            A, b = Xtr[:, keep], ytr
            B = np.where(np.isnan(Xte[:, keep]), mu[keep], Xte[:, keep])
        else:
            imp = {'mean': SimpleImputer(strategy='mean'),
                   'knn': KNNImputer(n_neighbors=5),
                   'mice': IterativeImputer(max_iter=10,
                                            random_state=42)}[strategy]
            A, b, B = imp.fit_transform(Xtr), ytr, imp.transform(Xte)
        m = model_fn()
        m.fit(A, b)
        r2s.append(r2_score(yte, m.predict(B)))
    return np.array(r2s)


def step5_benchmark(df, folds, motif, rate=0.30):
    banner("STEP 5  Six-arm comparison, model class held fixed")
    print("  Each block varies only the missing-data treatment. Because the")
    print("  native holes are uninformative (step 3), the comparison is run on")
    print("  the informative block with composition masked at %.0f%%, which is"
          % (100 * rate))
    print("  the regime where the treatments can actually differ.\n")

    X0 = df[INFORMATIVE].to_numpy(float)
    driver = INFORMATIVE.index('mae_oop')
    X = _mar(X0, [INFORMATIVE.index(c) for c in COMPOSITION], rate, driver)

    # Compositional data are exactly collinear, so ordinary least squares is
    # inadmissible here whatever happens to the missing values, and a
    # comparison built on it would measure conditioning rather than strategy.
    print("  Why both arms are penalised:")
    print("    C + H + O = 1 exactly, so the descriptors carry an exact linear")
    print("    dependency and the intact design matrix is already singular")
    print("    (condition number %.1e). Unpenalised least squares diverges on"
          % np.linalg.cond(X0))
    print("    it, to R2 of order -1e8, and imputation compounds the problem by")
    print("    breaking the closure that FIML preserves through the covariance")
    print("    structure. Both arms therefore take the same L2 penalty")
    print("    (alpha = %.3g), so what follows reflects the missing-data" % ALPHA)
    print("    treatment and not the conditioning.")
    print()

    out = {}
    for lab in LABELS:
        yv = df[lab].to_numpy(float)
        print("  %s" % lab)
        print("    %-24s %10s %8s" % ("strategy", "R2", "sd"))
        print("    " + "-" * 46)
        block, best = {}, -np.inf
        for label, key in CONV:
            r = _conv(key, scaled(lambda: Ridge(alpha=ALPHA)), X, yv, folds)
            if r is None:
                print("    %-24s %10s" % (label, "n/a"))
                continue
            block[label] = r
            best = max(best, r.mean())
            print("    %-24s %10.4f %8.4f" % (label, r.mean(), r.std()))
        r2s = np.array([r2_score(yv[te],
                                 ML.MissRidgeRegressor(copula=False, alpha=ALPHA,
                                                       compute_se=False)
                                 .fit(X[tr], yv[tr]).predict(X[te]))
                        for tr, te in folds])
        block['FIML'] = r2s
        print("    %-24s %10.4f %8.4f   <-- %+.4f vs best conventional"
              % ("FIML (MissRidge)", r2s.mean(), r2s.std(),
                 r2s.mean() - best))
        print()
        out[lab] = block

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cmap = plt.colormaps['viridis']
    for ax, lab in zip(axes, LABELS):
        block = out[lab]
        ks = list(block)
        means = [block[k].mean() for k in ks]
        sds = [block[k].std() for k in ks]
        ax.bar(range(len(ks)), means, yerr=sds, capsize=4,
               color=[cmap(v) for v in np.linspace(0.12, 0.88, len(ks))])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=35, ha='right')
        ax.set_title(lab)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        lo = min(m - s for m, s in zip(means, sds))
        hi = max(m + s for m, s in zip(means, sds))
        pad = 0.08 * (hi - lo) if hi > lo else 0.01
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel(r"$R^2$")
    fig.suptitle("Missing-data strategy, linear model class held fixed, "
                 "composition masked at %.0f%%" % (100 * rate))
    plt.show()
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="3 folds instead of 5")
    ap.add_argument("--rate", type=float, default=0.30)
    a = ap.parse_args()

    if not os.path.exists(CSV):
        raise SystemExit(
            "graphene_oxide.csv not found in example_data/.\n"
            "Source: CSIRO Data Access Portal, doi:10.25919/5e30b45f9852c")

    df = pd.read_csv(CSV)
    banner("Graphene oxide: two DFT labels, one missingness pattern")
    print("  Formation_energy and Fermi_energy, from %d structures" % len(df))
    print("  Source: CSIRO, doi:10.25919/5e30b45f9852c")

    n_splits = 3 if a.quick else 5
    folds = list(KFold(n_splits, shuffle=True, random_state=0)
                 .split(np.zeros((len(df), 1))))

    step1_mechanism(df)
    step2_identity(df)
    motif = step3_worth(df, folds)
    step4_vulnerability(df, folds, rate=a.rate)
    step5_benchmark(df, folds, motif, rate=a.rate)

    banner("Done")


if __name__ == "__main__":
    main()
