# -*- coding: utf-8 -*-
"""Six-arm strategy comparison on sentinel-coded clinical data
===========================================================


Pima Indians Diabetes (UCI / NIDDK): 768 records, 8 clinical
measurements, predicting diabetes onset within five years.

The missingness is real and native but disguised as zeros. A recorded serum
insulin, triceps skinfold, blood pressure, BMI or glucose of exactly 0 is
physiologically impossible and encodes a measurement that was never taken.
Insulin is roughly 49% absent and skinfold roughly 30%.

Recognising that is itself part of the analysis. Treating those zeros as real
values biases every downstream estimate, and no amount of careful modelling
afterwards recovers from it.

Each model family is compared against its own matched counterpart across the six
standard arms, so a difference between arms is attributable to the missing-data
treatment rather than to model capacity.

Generated from 04_Strategies_Pima_Diabetes.ipynb so the two cannot drift apart. The notebook is the place
to read this interactively; the script is for headless runs, for diffing in
version control, and for reading without a notebook renderer.

Run
---
    python 04_pima_strategies.py
"""
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

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

# Resolve the repository root from this file rather than from the working
# directory, so the script runs from anywhere and on any machine.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    # --------------------------------------------------------------------
    # notebook cell 1
    # --------------------------------------------------------------------
    import os, sys, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                                 brier_score_loss)

    warnings.filterwarnings('ignore')


    from MissLearn import (MissLogistic, MissBayesClassifier,
                           MissSupportClassifier, MissNeighborsClassifier,
                           MissDiagnostic)

    DATA_DIR = os.path.join(_HERE, 'example_data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Setup complete.")

    # --------------------------------------------------------------------
    # notebook cell 2
    # --------------------------------------------------------------------
    COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    PATH = os.path.join(DATA_DIR, 'pima_diabetes.csv')

    if not os.path.exists(PATH):
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        df = pd.read_csv(url, header=None, names=COLS)
        df.to_csv(PATH, index=False)
        print(f"Downloaded: {df.shape[0]} rows")
    else:
        df = pd.read_csv(PATH)
        print(f"Loaded from cache: {df.shape[0]} rows")

    # A recorded zero is impossible for these five measurements: it encodes
    # "not measured".  Leaving them as zeros would silently corrupt every
    # model; recoding them as NaN is the honest representation.
    ZERO_IS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI']
    for c in ZERO_IS_MISSING:
        n_zero = int((df[c] == 0).sum())
        df.loc[df[c] == 0, c] = np.nan
        print(f"  {c:<16} {n_zero:>4} impossible zeros recoded as NaN")

    feature_names = COLS[:-1]
    y = df['Outcome'].values.astype(float)
    X = df[feature_names].values.astype(float)
    print(f"\nn={X.shape[0]}  p={X.shape[1]}  positive={y.mean():.1%}")
    print(f"cells missing={np.isnan(X).mean():.1%}   "
          f"rows with at least one hole={np.isnan(X).any(axis=1).mean():.1%}")

    # --------------------------------------------------------------------
    # notebook cell 3
    # --------------------------------------------------------------------
    miss = pd.Series(np.isnan(X).mean(axis=0) * 100, index=feature_names)
    miss = miss[miss > 0].sort_values()
    print(miss.round(1).to_frame('missing %').to_string())

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = plt.colormaps['viridis'](np.linspace(0.2, 0.8, len(miss)))
    ax.barh(range(len(miss)), miss.values, color=colors)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels(miss.index)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Pima diabetes: missingness after recoding impossible zeros')
    fig.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 4
    # --------------------------------------------------------------------
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    FOLDS = list(SKF.split(X, y))

    STRATEGIES = [('Drop rows', 'drop_rows'), ('Drop cols', 'drop_cols'),
                  ('Mean imputation', 'mean'), ('kNN imputation', 'knn'),
                  ('MICE', 'mice')]


    def scaled(make_est):
        """Give a conventional baseline the same standardisation the MissLearn
        counterpart applies internally, so the comparison is about missingness
        and not about preprocessing."""
        return lambda: Pipeline([('scale', StandardScaler()),
                                 ('clf', make_est())])


    def prepare(strategy, X_tr, y_tr, X_te):
        """Apply one conventional missing-data treatment. Returns None when the
        strategy cannot be used on this fold."""
        mu = np.nanmean(X_tr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        if strategy == 'drop_rows':
            cc = ~np.isnan(X_tr).any(axis=1)
            if cc.sum() < X_tr.shape[1] + 2 or len(np.unique(y_tr[cc])) < 2:
                return None
            return X_tr[cc], y_tr[cc], np.where(np.isnan(X_te), mu, X_te)
        if strategy == 'drop_cols':
            keep = ~np.isnan(X_tr).any(axis=0)
            if keep.sum() == 0:
                return None
            Xte = X_te[:, keep]
            return (X_tr[:, keep], y_tr,
                    np.where(np.isnan(Xte), mu[keep], Xte))
        imp = {'mean': SimpleImputer(strategy='mean'),
               'knn':  KNNImputer(n_neighbors=5),
               'mice': IterativeImputer(max_iter=10, random_state=42)}[strategy]
        return imp.fit_transform(X_tr), y_tr, imp.transform(X_te)


    def score_arm(factory, strategy=None):
        """Five-fold metrics for one arm. strategy=None means the FIML arm,
        which is fitted directly on the incomplete matrix."""
        acc, auc, f1s, brier = [], [], [], []
        for tr, te in FOLDS:
            X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
            if strategy is None:
                Xtr2, ytr2, Xte2 = X_tr, y_tr, X_te
            else:
                got = prepare(strategy, X_tr, y_tr, X_te)
                if got is None:
                    return None
                Xtr2, ytr2, Xte2 = got
            m = factory()
            m.fit(Xtr2, ytr2)
            proba = m.predict_proba(Xte2)[:, 1]
            pred = m.predict(Xte2)
            acc.append(accuracy_score(y_te, pred))
            auc.append(roc_auc_score(y_te, proba))
            f1s.append(f1_score(y_te, pred, average='weighted',
                                zero_division=0))
            brier.append(brier_score_loss(y_te, proba))
        return {'accuracy': np.array(acc), 'auc': np.array(auc),
                'f1': np.array(f1s), 'brier': np.array(brier)}


    # (label, conventional counterpart, MissLearn estimator)
    FAMILIES = [
        ('Logistic regression', scaled(lambda: LogisticRegression(max_iter=2000)),
         lambda: MissLogistic(copula=False, compute_se=False), 'MissLogistic'),
        ('Generative Gaussian', lambda: GaussianNB(),
         lambda: MissBayesClassifier(copula=False, ), 'MissBayes'),
        ('Support vector', scaled(lambda: SVC(probability=True, random_state=0)),
         lambda: MissSupportClassifier(copula=False, ), 'MissSupport'),
        ('k-nearest neighbours', scaled(lambda: KNeighborsClassifier()),
         lambda: MissNeighborsClassifier(copula=False, ), 'MissNeighbors'),
    ]
    print(f"{len(FAMILIES)} model classes x {len(STRATEGIES) + 1} strategies, "
          f"5 folds each.")

    # --------------------------------------------------------------------
    # notebook cell 5
    # --------------------------------------------------------------------
    results = {}          # results[class_label][arm_label] = metric dict

    for label, sk, ml, ml_name in FAMILIES:
        print("=" * 78)
        print(f"  {label}   [same estimator throughout; only the treatment varies]")
        print("=" * 78)
        print(f"    {'treatment':<20}{'accuracy':>10}{'AUC':>10}"
              f"{'F1':>10}{'Brier':>10}")
        arms = {}
        for arm_label, key in STRATEGIES:
            r = score_arm(sk, key)
            if r is None:
                print(f"    {arm_label:<20}{'unusable on this data':>40}")
                continue
            arms[arm_label] = r
            print(f"    {arm_label:<20}{r['accuracy'].mean():>10.4f}"
                  f"{r['auc'].mean():>10.4f}{r['f1'].mean():>10.4f}"
                  f"{r['brier'].mean():>10.4f}")
        r = score_arm(ml, None)
        arms[f'FIML ({ml_name})'] = r
        print(f"    {'FIML':<20}{r['accuracy'].mean():>10.4f}"
              f"{r['auc'].mean():>10.4f}{r['f1'].mean():>10.4f}"
              f"{r['brier'].mean():>10.4f}")

        best_auc = max(v['auc'].mean() for k, v in arms.items()
                       if not k.startswith('FIML'))
        best_br = min(v['brier'].mean() for k, v in arms.items()
                      if not k.startswith('FIML'))
        print(f"    --> FIML vs best conventional:  "
              f"AUC {r['auc'].mean() - best_auc:+.4f}   "
              f"Brier {best_br - r['brier'].mean():+.4f} "
              f"(positive favours FIML)\n")
        results[label] = arms

    # --------------------------------------------------------------------
    # notebook cell 6
    # --------------------------------------------------------------------
    # One panel per metric, one group per model class; colour encodes the
    # missing-data treatment (viridis).  The legend sits below the panels rather
    # than inside them, so it can never overlap a bar.
    metrics = [('auc', 'ROC-AUC', True), ('brier', 'Brier score', False)]
    arm_order = [lbl for lbl, _ in STRATEGIES] + ['FIML']
    cmap = plt.colormaps['viridis']
    colors = {a: cmap(v) for a, v in
              zip(arm_order, np.linspace(0.08, 0.88, len(arm_order)))}
    short = {'Mean imputation': 'Mean imp.', 'kNN imputation': 'kNN imp.'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    handles = None
    for ax, (key, nice, higher) in zip(axes, metrics):
        width = 0.78 / len(arm_order)
        for a_i, arm in enumerate(arm_order):
            xs, vals, errs = [], [], []
            for g_i, (cls, arms) in enumerate(results.items()):
                match = [k for k in arms if k.startswith(arm)]
                if not match:
                    continue
                r = arms[match[0]]
                xs.append(g_i - 0.39 + width * (a_i + 0.5))
                vals.append(r[key].mean())
                errs.append(r[key].std())
            ax.bar(xs, vals, width=width * 0.9, yerr=errs, capsize=1.5,
                   color=colors[arm], label=short.get(arm, arm),
                   error_kw=dict(lw=0.6))
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([c.replace(' ', '\n') for c in results],
                           fontsize=8)
        ax.set_ylabel(nice, fontsize=9)
        ax.set_title(f"{nice} " +
                     ("(higher is better)" if higher else "(lower is better)"),
                     fontsize=9.5)
        ax.tick_params(axis='y', labelsize=8)
        lo = min(min(r[key].mean() for r in arms.values())
                 for arms in results.values())
        hi = max(max(r[key].mean() for r in arms.values())
                 for arms in results.values())
        pad = (hi - lo) * 0.10
        ax.set_ylim(max(0, lo - pad), hi + pad)
        handles = ax.get_legend_handles_labels()

    fig.legend(*handles, loc='lower center', ncol=len(arm_order),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Missing-data treatment, model class held fixed within each '
                 'group', fontsize=10)
    # room for the suptitle above and the legend below
    fig.subplots_adjust(top=0.84, bottom=0.28, wspace=0.22)
    plt.show()
    print("Groups are not comparable with one another: the model class differs "
          "between them,\nso only the bars within a group answer the "
          "missing-data question.")

    # --------------------------------------------------------------------
    # notebook cell 7
    # --------------------------------------------------------------------
    # What kind of missingness is this?  MissDiagnostic assembles the evidence
    # that bears on the MAR assumption the FIML arm relies on.
    diag = MissDiagnostic(X, y, feature_names=feature_names)
    little = diag.little_mcar_test()
    print(f"Little's MCAR test: chi2 = {little['statistic']:.1f}  "
          f"df = {little['df']}  p = {little['pvalue']:.3g}")
    print("  -> " + ("MCAR rejected: missingness depends on observed values, "
                     "which is the MAR signature FIML handles."
                     if little['pvalue'] < 0.05 else
                     "MCAR not rejected."))

    mar = diag.mar_plausibility()
    n_sig = sum(1 for v in mar.values() if v['significant'])
    print(f"\nMAR plausibility: missingness is predictable from the observed "
          f"data for {n_sig} of {len(mar)} incomplete features.")
    print("(Predictable missingness is what full-information estimation "
          "exploits; it is also what biases listwise deletion.)")


if __name__ == "__main__":
    main()
