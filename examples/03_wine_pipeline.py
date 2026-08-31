# -*- coding: utf-8 -*-
"""The full workflow: diagnose, validate, fit, explain
===================================================


Wine Quality, white wines (UCI, Cortez et al. 2009): 4,898 wines, 11
physicochemical measurements collapsed to three quality classes, with 15% MAR
missingness injected.

This is the template to adapt to a new data set rather than a performance claim.
It runs MissDiagnostic, then prefit_check and MissPreprocessor, then a
One-vs-Rest FIML multiclass classifier, then MissExplainer.

Three details are worth noting. The compatibility report is given
feature_names, so its warnings name the measurement rather than 'X4'. Its
advice is acted on rather than printed and ignored: the scale warning is
explained, because MissLogistic standardises internally, and the kurtosis notes
drive an actual copula comparison whose result is reported whichever way it
falls. And the report appears twice on purpose: once standalone, then again
from inside MissPreprocessor.fit, where it can see the estimator it is
protecting and stops advising a setting that estimator already has.

MissExplainer is given class_index, because a three-class model has no single
scalar value function. Attributing the predicted label instead would describe
changes in the argmax rather than changes in belief.

Generated from 03_MissLearn_Pipeline_Wine.ipynb so the two cannot drift apart. The notebook is the place
to read this interactively; the script is for headless runs, for diffing in
version control, and for reading without a notebook renderer.

Run
---
    python 03_wine_pipeline.py
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
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        confusion_matrix, ConfusionMatrixDisplay,
    )

    warnings.filterwarnings('ignore')


    from MissLearn import (
        MissLogistic, MissMulticlass, MissPreprocessor,
        MissDiagnostic, MissExplainer, prefit_check,
    )

    DATA_DIR = os.path.join(_HERE, 'example_data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Setup complete.")

    # --------------------------------------------------------------------
    # notebook cell 2
    # --------------------------------------------------------------------
    DATA_PATH = os.path.join(DATA_DIR, 'winequality-white.csv')

    if not os.path.exists(DATA_PATH):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "wine-quality/winequality-white.csv")
        df_raw = pd.read_csv(url, sep=';')
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"Downloaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"Loaded from cache: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    print("\nFeatures:", list(df_raw.columns[:-1]))
    print("Quality distribution:")
    print(df_raw['quality'].value_counts().sort_index().to_string())

    # --------------------------------------------------------------------
    # notebook cell 3
    # --------------------------------------------------------------------
    FEATURE_NAMES = list(df_raw.columns[:-1])
    quality = df_raw['quality'].values

    # Ordinal bucketing: 0=low (<=5), 1=medium (=6), 2=high (>=7)
    y = np.where(quality <= 5, 0.0, np.where(quality == 6, 1.0, 2.0))
    X = df_raw[FEATURE_NAMES].values.astype(float)

    class_counts = pd.Series(y).value_counts().sort_index()
    class_labels = {0: 'low (<=5)', 1: 'medium (=6)', 2: 'high (>=7)'}
    print("3-class target distribution:")
    for k, cnt in class_counts.items():
        print(f"  Class {int(k)} [{class_labels[k]}] : {cnt:4d}  ({cnt/len(y):.1%})")

    n, p = X.shape
    print(f"\nn={n}  p={p}")

    # --------------------------------------------------------------------
    # notebook cell 4
    # --------------------------------------------------------------------
    def inject_mar(X, rate=0.15, seed=42):
        rng = np.random.default_rng(seed)
        Xm = X.copy()
        n, p = X.shape
        for j in range(p):
            ref = X[:, (j + 1) % p]
            med = np.nanmedian(ref)
            p_high = min(rate * 4 / 3, 1.0)
            p_low  = min(rate * 2 / 3, 1.0)
            prob   = np.where(ref > med, p_high, p_low)
            Xm[rng.random(n) < prob, j] = np.nan
        actual = np.isnan(Xm).mean()
        return Xm, actual

    X_miss, actual_rate = inject_mar(X, rate=0.15, seed=42)
    print(f"Target miss rate: 15.0%   Actual: {actual_rate:.1%}")
    print("\nMissing % per feature:")
    miss_s = pd.Series(np.isnan(X_miss).mean(axis=0) * 100,
                       index=FEATURE_NAMES).round(1)
    print(miss_s.to_string())

    # Train / test split (stratified, 80/20)
    rng_split = np.random.default_rng(0)
    idx = rng_split.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_miss[tr_idx], y[tr_idx]
    X_te, y_te = X_miss[te_idx], y[te_idx]
    print(f"\nTrain: n={len(tr_idx)}  Test: n={len(te_idx)}")

    # --------------------------------------------------------------------
    # notebook cell 5
    # --------------------------------------------------------------------
    diag = MissDiagnostic(X_tr, y_tr, feature_names=FEATURE_NAMES, alpha=0.05)

    # Little's MCAR test
    little = diag.little_mcar_test()
    print("Little's MCAR test:")
    print(f"  chi2 = {little['statistic']:.2f}  df = {little['df']}  "
          f"p = {little['pvalue']:.4g}  --> {'REJECT MCAR' if little['significant'] else 'Cannot reject MCAR'}")

    # --------------------------------------------------------------------
    # notebook cell 6
    # --------------------------------------------------------------------
    # MAR plausibility: logistic regression of each missingness indicator
    mar = diag.mar_plausibility()
    print("MAR plausibility (significant = missingness predicted from observed data):")
    print(f"{'Feature':<28} {'LR stat':>9} {'df':>4} {'p-value':>10} {'Significant':>12}")
    print("-" * 68)
    for feat, stats in mar.items():
        sig = '*' if stats['significant'] else ''
        print(f"  {feat:<26} {stats['lr_statistic']:>9.2f} {stats['df']:>4d} "
              f"{stats['pvalue']:>10.4g} {sig:>12}")

    # --------------------------------------------------------------------
    # notebook cell 7
    # --------------------------------------------------------------------
    # Full summary report (plain-language verdict)
    diag.summary()

    # --------------------------------------------------------------------
    # notebook cell 8
    # --------------------------------------------------------------------
    # Missingness pattern summary
    patterns = diag.pattern_summary()
    print(f"\nUnique missing-data patterns: {len(patterns)}")
    print("Top 8 patterns by frequency:")
    top_pat = sorted(patterns, key=lambda r: -r['n'])[:8]
    for row in top_pat:
        print(f"  count={row['n']:4d}  miss_cols={', '.join(row['missing_cols'])}")

    # --------------------------------------------------------------------
    # notebook cell 9
    # --------------------------------------------------------------------
    # Missingness correlation heatmap (phi-coefficients)
    phi_mat = diag.missingness_correlations()
    feat_labels = diag.miss_corr_cols_   # columns with any missingness

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(phi_mat, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_labels)))
    ax.set_yticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feat_labels, fontsize=8)
    for ri in range(len(feat_labels)):
        for ci in range(len(feat_labels)):
            ax.text(ci, ri, f'{phi_mat[ri, ci]:.2f}',
                    ha='center', va='center', fontsize=6.5,
                    color='white' if abs(phi_mat[ri, ci]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Phi coefficient')
    ax.set_title('Missingness correlations (phi-coefficients)\n'
                 'Positive = features tend to be missing together')
    fig.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 10
    # --------------------------------------------------------------------
    # Standalone compatibility check. Run on its own, before any estimator
    # exists, so its advice can only be generic.
    print("Compatibility check, standalone: nothing known about the model yet.")
    result = prefit_check(X_tr, y_tr, feature_names=FEATURE_NAMES)
    result.summary()

    # --------------------------------------------------------------------
    # notebook cell 11
    # --------------------------------------------------------------------
    # MissPreprocessor wraps the model transparently: it runs the check at fit()
    # and raises on errors or warns on issues.
    #
    # Pass feature_names so the report names the measurement rather than 'X4'.
    # A DataFrame's own columns are picked up automatically; with a plain ndarray
    # the names have to be supplied.
    base_clf = MissMulticlass(MissLogistic())
    prep_clf = MissPreprocessor(base_clf, feature_names=FEATURE_NAMES)

    # The report below is the second in this log, and it is not a repeat. The
    # same check runs again inside fit(), this time knowing which estimator it
    # is protecting, so compare its kurtosis notes with the ones above: the
    # standalone run suggests copula='auto', while this one can see that the
    # wrapped MissLogistic already has it and says so instead. The look-through
    # goes one level down, since the copula here sits on the model inside
    # MissMulticlass rather than on the wrapper.
    #
    # Pass verbose=False to MissPreprocessor to keep only one of the two.
    print("Compatibility check, again inside fit(): now estimator-aware.")
    prep_clf.fit(X_tr, y_tr)
    print("MissPreprocessor: fit complete.")
    print("Underlying model class:", type(prep_clf.estimator_).__name__)

    # --------------------------------------------------------------------
    # notebook cell 12
    # --------------------------------------------------------------------
    # Does the copula transform actually help on this data? Same model class,
    # same folds; the only difference is the marginal transform.
    from MissLearn import MissRidgeClassifier   # noqa: F401  (kept for parity)

    skf_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    def cv_auc(make_model):
        aucs = []
        for tr, te in skf_c.split(X_miss, y):
            m = make_model()
            m.fit(X_miss[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X_miss[te]),
                                      multi_class='ovr', average='macro'))
        return np.array(aucs)


    # copula=False is stated rather than left to the default. The default
    # is now 'auto', so a bare MissLogistic() would be the same model as
    # the arm below and the comparison would quietly report a difference
    # of zero.
    plain  = cv_auc(lambda: MissMulticlass(MissLogistic(copula=False)))
    copula = cv_auc(lambda: MissMulticlass(MissLogistic(copula='auto')))

    print("macro AUC, 5-fold, same splits")
    print("  joint normal as-is   : %.4f +/- %.4f" % (plain.mean(), plain.std()))
    print("  copula='auto'        : %.4f +/- %.4f" % (copula.mean(), copula.std()))
    print("  difference           : %+.4f" % (copula.mean() - plain.mean()))
    print()
    if copula.mean() - plain.mean() > 0.002:
        print("The copula transform helps, so the kurtosis note was worth acting on.")
    elif copula.mean() - plain.mean() < -0.002:
        print("The copula transform hurts here. Heavy margins do not always mean")
        print("the joint-normal working model is the binding constraint, and a")
        print("diagnostic note is a prompt to test rather than an instruction.")
    else:
        print("The two are within noise. The kurtosis is real but it is not what")
        print("limits this model, which is worth knowing before adding machinery.")

    # Act on the result. A diagnostic that is printed and then ignored is
    # decoration; the point of running the comparison is to let it choose.
    USE_COPULA = copula.mean() - plain.mean() > 0.002


    def make_clf():
        """The model this example settled on, copula included only if it earned it."""
        return MissMulticlass(MissLogistic(copula='auto' if USE_COPULA else False))


    print()
    print("Downstream models use copula='auto': %s" % USE_COPULA)

    # --------------------------------------------------------------------
    # notebook cell 13
    # --------------------------------------------------------------------
    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for fold_tr, fold_te in skf.split(X_miss, y):
        clf = make_clf()
        clf.fit(X_miss[fold_tr], y[fold_tr])
        ypred  = clf.predict(X_miss[fold_te])
        yproba = clf.predict_proba(X_miss[fold_te])
        accs.append(accuracy_score(y[fold_te], ypred))
        f1s.append(f1_score(y[fold_te], ypred, average='weighted', zero_division=0))
        aucs.append(roc_auc_score(y[fold_te], yproba, multi_class='ovr',
                                  average='macro'))

    MODEL_LABEL = ("MissMulticlass(MissLogistic(copula=auto))" if USE_COPULA
                   else "MissMulticlass(MissLogistic)")
    print("5-fold CV results -- %s:" % MODEL_LABEL)
    print(f"  Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Macro AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Wt'd F1  : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # --------------------------------------------------------------------
    # notebook cell 14
    # --------------------------------------------------------------------
    # Final fit on training data + confusion matrix on held-out test set
    final_clf = make_clf()
    final_clf.fit(X_tr, y_tr)
    y_pred_te = final_clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred_te)
    disp = ConfusionMatrixDisplay(cm,
           display_labels=['Low', 'Medium', 'High'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap='viridis', colorbar=False)
    ax.set_title('Confusion matrix -- held-out test set\n'
                 + MODEL_LABEL)
    fig.tight_layout()
    plt.show()

    print(f"Test accuracy : {accuracy_score(y_te, y_pred_te):.4f}")
    print(f"Test macro AUC: {roc_auc_score(y_te, final_clf.predict_proba(X_te), multi_class='ovr', average='macro'):.4f}")

    # --------------------------------------------------------------------
    # notebook cell 15
    # --------------------------------------------------------------------
    # This is a three-class model, so there is no single quantity to attribute:
    # Shapley values need a scalar value function. class_index names the class whose
    # PROBABILITY is explained. Class 2 is "high quality", which is the interesting
    # one for a winemaker.
    #
    # Note what this is not doing. Attributing the predicted *label* would ask which
    # features flip the argmax, which is a different and much coarser question, and
    # on an imbalanced problem it degenerates: every coalition returns the majority
    # class and every attribution is exactly zero. MissLearn refuses to guess a
    # class for that reason, so class_index is required here.
    HIGH_QUALITY = 2

    exp = MissExplainer(
        final_clf,
        exact_threshold=15,   # p=11 < 15, so exact 2^11 Shapley is used
        class_index=HIGH_QUALITY,
        random_state=42,
    )
    exp.fit(X_tr, feature_names=FEATURE_NAMES)

    print(f"MissExplainer fitted on the {exp.value_scale_} scale.")
    print(f"Explaining P(class {HIGH_QUALITY} = high quality).")
    print(f"Baseline, all features unknown: {exp.expected_value_:.4f}")
    print(f"(that is the prior share of high-quality wines, since with nothing")
    print(f" observed the model can only return the marginal probability)")

    # --------------------------------------------------------------------
    # notebook cell 16
    # --------------------------------------------------------------------
    # Compute SHAP values on a sample of 200 test observations
    N_EXPLAIN = 200
    rng_exp = np.random.default_rng(7)
    sample_idx = rng_exp.choice(len(X_te), size=N_EXPLAIN, replace=False)
    X_sample = X_te[sample_idx]

    phi      = exp.shap_values(X_sample)    # value SHAP   (200, 11)
    phi_miss = exp.miss_shap(X_sample)       # missingness SHAP (200, 11)

    print(f"Value SHAP matrix shape : {phi.shape}")
    print(f"Missingness SHAP shape  : {phi_miss.shape}")

    # Global feature importance (mean |SHAP|)
    importance = pd.Series(np.abs(phi).mean(axis=0), index=FEATURE_NAMES)
    print("\nMean |SHAP| (value contribution):")
    print(importance.sort_values(ascending=False).round(4).to_string())

    # --------------------------------------------------------------------
    # notebook cell 17
    # --------------------------------------------------------------------
    # Value SHAP beeswarm -- which features drive quality predictions?
    fig, ax = exp.plot_beeswarm(
        phi, X_sample,
        title='Value SHAP: Feature contributions to predicted wine quality class',
        figsize=(8, 5),
        show=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black') # Optional: ensure it's black
        spine.set_linewidth(1)   # Optional: adjust thickness

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 18
    # --------------------------------------------------------------------
    # Missingness SHAP importance -- which features carry the most information?
    fig, ax = exp.plot_miss_importance(
        phi_miss,
        title='Missingness SHAP: Information cost of not observing each feature',
        figsize=(8, 5),
        show=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black') # Optional: ensure it's black
        spine.set_linewidth(1)   # Optional: adjust thickness

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 19
    # --------------------------------------------------------------------
    # Dependence plot: alcohol SHAP vs alcohol value, coloured by sulphates
    ALCOHOL_IDX   = FEATURE_NAMES.index('alcohol')
    SULPHATES_IDX = FEATURE_NAMES.index('sulphates')

    fig, ax = exp.plot_dependence(
        phi, X_sample,
        feature_idx=ALCOHOL_IDX,
        interaction_idx=SULPHATES_IDX,
        figsize=(7, 5),
        show=False,
    )
    ax.set_title('SHAP dependence: alcohol content vs. quality prediction\n'
                 '(colour = sulphates concentration)')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black') # Optional: ensure it's black
        spine.set_linewidth(1)   # Optional: adjust thickness

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 20
    # --------------------------------------------------------------------
    # Waterfall plot for one specific wine
    wine_idx = 0   # first sample in the explanation set
    fig, ax = exp.plot_waterfall(
        phi, X_sample,
        i=wine_idx,
        title='SHAP Waterfall: one-wine breakdown',
        figsize=(8, 6),
        show=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black') # Optional: ensure it's black
        spine.set_linewidth(1)   # Optional: adjust thickness
    plt.tight_layout()
    plt.show()

    # Show the wine's actual values for reference
    wine_df = pd.DataFrame(X_sample[[wine_idx]], columns=FEATURE_NAMES)
    wine_df['predicted_class'] = final_clf.predict(X_sample[[wine_idx]])[0]
    wine_df['actual_class']    = y_te[sample_idx[wine_idx]]
    print("Wine under analysis:")
    print(wine_df.T.to_string(header=False))

    # --------------------------------------------------------------------
    # notebook cell 21
    # --------------------------------------------------------------------
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


    def ovr_logistic():
        """The conventional counterpart of MissMulticlass(MissLogistic): the same
        One-vs-Rest structure, the same logistic model class, and the same
        internal standardisation."""
        return OneVsRestClassifier(
            Pipeline([('scale', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=2000))]))


    STRATEGIES = [("Drop rows", 'drop_rows'), ("Drop cols", 'drop_cols'),
                   ("Mean imputation", 'mean'), ("kNN imputation", 'knn'),
                   ("MICE", 'mice')]

    # The same folds section 5 used, so the arms are paired fold by fold.
    BENCH_FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=42).split(X_miss, y))


    def score_arm(yte, ypred, yproba):
        return dict(
            acc=accuracy_score(yte, ypred),
            auc=roc_auc_score(yte, yproba, multi_class='ovr', average='macro'),
            f1=f1_score(yte, ypred, average='weighted', zero_division=0),
        )


    def run_conventional(strategy):
        """One conventional arm across the shared folds. Returns None when the
        strategy cannot be applied to a fold at all."""
        out = []
        for tr, te in BENCH_FOLDS:
            Xtr, ytr, Xte, yte = X_miss[tr], y[tr], X_miss[te], y[te]
            mu = np.nanmean(Xtr, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)

            if strategy == 'drop_rows':
                cc = ~np.isnan(Xtr).any(axis=1)
                # Listwise deletion needs enough surviving rows, and all three
                # classes must still be present or the fit is undefined.
                if cc.sum() < Xtr.shape[1] + 2 or len(np.unique(ytr[cc])) < 3:
                    return None
                A, b = Xtr[cc], ytr[cc]
                B = np.where(np.isnan(Xte), mu, Xte)
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
                A, b = imp.fit_transform(Xtr), ytr
                B = imp.transform(Xte)

            m = ovr_logistic()
            m.fit(A, b)
            out.append(score_arm(yte, m.predict(B), m.predict_proba(B)))
        return out


    def run_fiml():
        # Deliberately the plain model, not make_clf(). This block compares
        # missing-data treatments with the model class held fixed, so both sides
        # must get the same preprocessing; handing the copula transform to the
        # FIML arm alone would measure the transform rather than the treatment.
        out = []
        for tr, te in BENCH_FOLDS:
            clf = MissMulticlass(MissLogistic())
            clf.fit(X_miss[tr], y[tr])
            out.append(score_arm(y[te], clf.predict(X_miss[te]),
                                 clf.predict_proba(X_miss[te])))
        return out


    def agg(folds, key):
        v = np.array([f[key] for f in folds])
        return v.mean(), v.std()


    print("Running six arms over 5 shared folds ...")
    bench = {}
    for label, key in STRATEGIES:
        r = run_conventional(key)
        if r is None:
            print("  %-20s not applicable" % label)
            continue
        bench[label] = r
        print("  %-20s done" % label)
    bench['FIML (MissMulticlass)'] = run_fiml()
    print("  %-20s done" % 'FIML')

    # --------------------------------------------------------------------
    # notebook cell 22
    # --------------------------------------------------------------------
    print("%-26s %16s %16s %16s" % ("arm", "accuracy", "macro AUC", "weighted F1"))
    print("-" * 78)
    for label, folds in bench.items():
        a, asd = agg(folds, 'acc')
        u, usd = agg(folds, 'auc')
        f, fsd = agg(folds, 'f1')
        print("%-26s %8.4f+/-%.4f %8.4f+/-%.4f %8.4f+/-%.4f"
              % (label, a, asd, u, usd, f, fsd))

    conv = {k: v for k, v in bench.items() if not k.startswith('FIML')}
    best_conv = max(conv, key=lambda k: agg(conv[k], 'auc')[0])
    fiml_auc = agg(bench['FIML (MissMulticlass)'], 'auc')[0]
    gap = fiml_auc - agg(conv[best_conv], 'auc')[0]

    print()
    print("Best conventional arm : %s (macro AUC %.4f)"
          % (best_conv, agg(conv[best_conv], 'auc')[0]))
    print("FIML                  : macro AUC %.4f" % fiml_auc)
    print("Gap                   : %+.4f  ->  %s"
          % (gap, "FIML ahead" if gap > 0.002 else
                  ("FIML behind" if gap < -0.002 else "parity")))

    # --------------------------------------------------------------------
    # notebook cell 23
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.colormaps['viridis']
    labels = list(bench)
    cols = [cmap(v) for v in np.linspace(0.12, 0.88, len(labels))]

    for ax, (key, title) in zip(axes, [('acc', 'Accuracy'),
                                        ('auc', 'Macro AUC'),
                                        ('f1', 'Weighted F1')]):
        means = [agg(bench[k], key)[0] for k in labels]
        sds = [agg(bench[k], key)[1] for k in labels]
        ax.bar(range(len(labels)), means, yerr=sds, capsize=4, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        lo = min(m - s for m, s in zip(means, sds))
        hi = max(m + s for m, s in zip(means, sds))
        pad = 0.08 * (hi - lo) if hi > lo else 0.01
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))

    fig.suptitle("Wine quality: missing-data strategy, One-vs-Rest logistic "
                 "held fixed across every arm")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
