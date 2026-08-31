# -*- coding: utf-8 -*-
"""Homogeneous and heterogeneous FIML ensembles on native MNAR data
================================================================


Thyroid Disease "sick" task (UCI / Garavan Institute, 1987): 4,664
clinical records, 16 binary clinical flags and 6 continuous lab assays.

The assays carry real, native, plausibly MNAR missingness: a test is ordered
only when a clinician suspects a specific dysfunction, so the pattern of absence
is itself informative and naive imputation is biased.

Every comparison holds the model class fixed at logistic regression. The
heterogeneous ensemble is built from four MissLearn members that differ in
regularisation and in whether the response is modelled discriminatively or
generatively, so its gain comes from decorrelated errors rather than from extra
capacity. No tree model appears anywhere, because comparing a likelihood against
a boosted tree would measure capacity rather than NaN handling.

The honest result is that FIML loses the strategy comparison here. The task is
dominated by binary indicator flags, the regime where a joint-Gaussian working
model is weakest, and that is why the example is included.

Generated from 02_MissEnsemble_Thyroid.ipynb so the two cannot drift apart. The notebook is the place
to read this interactively; the script is for headless runs, for diffing in
version control, and for reading without a notebook renderer.

Run
---
    python 02_thyroid_ensemble.py
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
    import os, sys, warnings, copy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, f1_score,
                                 brier_score_loss, RocCurveDisplay)

    warnings.filterwarnings('ignore')


    from MissLearn import (
        MissLogistic, MissRidgeClassifier, MissLASSOClassifier,
        MissBayesClassifier, MissEnsemble,
    )

    DATA_DIR = os.path.join(_HERE, 'example_data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Setup complete.")

    # --------------------------------------------------------------------
    # notebook cell 2
    # --------------------------------------------------------------------
    # Column names for the UCI Thyroid Sick dataset
    COLUMNS = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'illness_flag', 'pregnant',
        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
        'hypopituitary', 'psych',
        'TSH_measured', 'TSH', 'T3_measured', 'T3',
        'TT4_measured', 'TT4', 'T4U_measured', 'T4U',
        'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
        'referral_source', 'target',
    ]
    BINARY_COLS = [
        'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'illness_flag', 'pregnant',
        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
        'hypopituitary', 'psych',
        'TSH_measured', 'T3_measured', 'TT4_measured',
        'T4U_measured', 'FTI_measured', 'TBG_measured',
    ]

    def load_thyroid(data_dir):
        frames = []
        for fname in ['sick.data', 'sick.test']:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                       f"thyroid-disease/{fname}")
                df = pd.read_csv(url, header=None, names=COLUMNS, na_values='?')
                df.to_csv(fpath, index=False)
                print(f"Downloaded {fname}: {len(df)} rows")
            else:
                df = pd.read_csv(fpath)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        # Clean target
        # Split off the '|id' suffix FIRST, then strip the trailing dot
        # (raw labels look like 'sick.|3733').
        df['target'] = (df['target'].str.strip()
                                    .str.split('|').str[0]
                                    .str.strip()
                                    .str.rstrip('.'))
        y = (df['target'] == 'sick').astype(float).values

        # Encode binary string columns
        bool_map = {'t': 1.0, 'f': 0.0, 'M': 1.0, 'F': 0.0}
        for col in BINARY_COLS:
            df[col] = df[col].map(bool_map)

        # Drop columns we do not use as features
        drop = ['target', 'referral_source',
                'TSH_measured', 'T3_measured', 'TT4_measured',
                'T4U_measured', 'FTI_measured', 'TBG_measured',
                'TBG']  # TBG itself is 100% missing in this dataset
        df = df.drop(columns=drop)

        X = df.values.astype(float)
        return X, y, list(df.columns)

    X, y, feature_names = load_thyroid(DATA_DIR)
    print(f"\nFinal dataset: n={len(X)}  p={len(feature_names)}")
    print(f"Class balance: {y.mean():.1%} sick  ({int(y.sum())} of {len(y)})")

    # --------------------------------------------------------------------
    # notebook cell 3
    # --------------------------------------------------------------------
    miss_pct = pd.Series(np.isnan(X).mean(axis=0) * 100, index=feature_names)
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)

    print("Features with native missing values:")
    print(miss_pct.to_frame('missing %').to_string())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = plt.colormaps['viridis'](np.linspace(0.15, 0.85, len(miss_pct)))[::-1]
    ax.barh(range(len(miss_pct)), miss_pct.values, color=colors,
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(miss_pct)))
    ax.set_yticklabels(miss_pct.index, fontsize=10)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Native missingness: Thyroid lab assays\n'
                 '(Tests ordered only when clinically indicated -- MNAR pattern)', fontsize=12)
    fig.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 4
    # --------------------------------------------------------------------
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    FOLDS = list(SKF.split(X, y))

    def eval_folds(model_fn, X, y, folds, needs_complete_y=False):
        """
        Run model_fn() fresh on each fold.
        model_fn() must return a fitted model.
        Returns dict of metric arrays (len = n_folds).
        """
        aucs, f1s, briers = [], [], []
        for tr, te in folds:
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]
            if needs_complete_y:
                valid = ~np.isnan(ytr)
                Xtr, ytr = Xtr[valid], ytr[valid]
            m = model_fn()
            m.fit(Xtr, ytr)
            proba = m.predict_proba(Xte)[:, 1]
            pred  = m.predict(Xte)
            valid_te = ~np.isnan(yte)
            aucs.append(roc_auc_score(yte[valid_te], proba[valid_te]))
            f1s.append(f1_score(yte[valid_te], pred[valid_te], zero_division=0))
            briers.append(brier_score_loss(yte[valid_te], proba[valid_te]))
        return {'auc': np.array(aucs), 'f1': np.array(f1s), 'brier': np.array(briers)}

    def summarise(name, res):
        print(f"  {name:<40}  "
              f"AUC={res['auc'].mean():.4f}+/-{res['auc'].std():.4f}  "
              f"F1={res['f1'].mean():.4f}  "
              f"Brier={res['brier'].mean():.4f}")

    print("CV protocol: 5-fold stratified, metrics: AUC / F1 / Brier score")

    # --------------------------------------------------------------------
    # notebook cell 5
    # --------------------------------------------------------------------
    def scaled_logistic():
        """The conventional counterpart of MissLogistic, with the same internal
        standardisation so the comparison is not decided by preprocessing."""
        return Pipeline([('scale', StandardScaler()),
                         ('clf',   LogisticRegression(max_iter=2000))])


    def eval_strategy(strategy, X, y, folds, model_fn=scaled_logistic):
        """Evaluate one conventional missing-data strategy with the model class
        held fixed. `strategy` is one of drop_rows / drop_cols / mean / knn / mice.
        Returns None when the strategy is unusable on a fold."""
        aucs, f1s, briers = [], [], []
        for tr, te in folds:
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            mu = np.nanmean(Xtr, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            if strategy == 'drop_rows':
                cc = ~np.isnan(Xtr).any(axis=1)
                if cc.sum() < Xtr.shape[1] + 2 or len(np.unique(ytr[cc])) < 2:
                    return None
                Xtr2, ytr2 = Xtr[cc], ytr[cc]
                Xte2 = np.where(np.isnan(Xte), mu, Xte)
            elif strategy == 'drop_cols':
                keep = ~np.isnan(Xtr).any(axis=0)
                if keep.sum() == 0:
                    return None
                Xtr2, ytr2 = Xtr[:, keep], ytr
                Xte2 = np.where(np.isnan(Xte[:, keep]), mu[keep], Xte[:, keep])
            else:
                imp = {'mean': SimpleImputer(strategy='mean'),
                       'knn':  KNNImputer(n_neighbors=5),
                       'mice': IterativeImputer(max_iter=10, random_state=42)}[strategy]
                Xtr2, ytr2 = imp.fit_transform(Xtr), ytr
                Xte2 = imp.transform(Xte)
            m = model_fn()
            m.fit(Xtr2, ytr2)
            proba, pred = m.predict_proba(Xte2)[:, 1], m.predict(Xte2)
            v = ~np.isnan(yte)
            aucs.append(roc_auc_score(yte[v], proba[v]))
            f1s.append(f1_score(yte[v], pred[v], zero_division=0))
            briers.append(brier_score_loss(yte[v], proba[v]))
        return {'auc': np.array(aucs), 'f1': np.array(f1s),
                'brier': np.array(briers)}


    print("=" * 80)
    print("  3a. MISSING-DATA STRATEGY  (logistic regression held fixed)")
    print("      These arms ARE directly comparable to each other.")
    print("=" * 80)
    strategy_results = {}
    for label, key in [('Drop rows', 'drop_rows'), ('Drop cols', 'drop_cols'),
                       ('Mean imputation', 'mean'), ('kNN imputation', 'knn'),
                       ('MICE', 'mice')]:
        r = eval_strategy(key, X, y, FOLDS)
        if r is not None:
            strategy_results[label] = r
            summarise(label + ' + logistic', r)

    # The full-information arm: same model class, no imputation at all
    res_ml = eval_folds(lambda: MissLogistic(copula=False, ), X, y, FOLDS)
    strategy_results['FIML (MissLogistic)'] = res_ml
    summarise('FIML (MissLogistic)', res_ml)

    best_conv = max(r['auc'].mean() for k, r in strategy_results.items()
                    if not k.startswith('FIML'))
    gap = res_ml['auc'].mean() - best_conv
    print(f"\n  --> FIML AUC {res_ml['auc'].mean():.4f} vs best conventional "
          f"{best_conv:.4f}  (gap {gap:+.4f})")
    print("      On this indicator-dominated task the full-information arm trails;"
          "\n      see section 9.5 of the user guide for when to expect this.")

    # --------------------------------------------------------------------
    # notebook cell 6
    # --------------------------------------------------------------------
    # Four FIML families on identical folds. Only the model class varies, so
    # this block answers "which model", not "which missing-data strategy".
    from MissLearn import MissSupportClassifier, MissNeighborsClassifier

    family_results = {}
    for label, make in [
        ("MissLogistic (linear)",      lambda: MissLogistic(copula=False, compute_se=False)),
        ("MissBayes (generative)",     lambda: MissBayesClassifier(copula=False, )),
        ("MissNeighbors (distance)",   lambda: MissNeighborsClassifier(copula=False, n_neighbors=15)),
        ("MissSupport (RBF kernel)",   lambda: MissSupportClassifier(copula=False, kernel='rbf', C=4.0)),
    ]:
        res = eval_folds(make, X, y, FOLDS)
        family_results[label] = res
        summarise(label, res)

    # --------------------------------------------------------------------
    # notebook cell 7
    # --------------------------------------------------------------------
    # Same model class throughout: RBF SVM. Only the NaN treatment varies.
    # C is tuned by inner CV on the training rows of each outer fold, identically
    # on both sides, so neither arm gets a regularisation advantage.
    from sklearn.svm import SVC

    C_GRID = [4.0, 8.0, 16.0, 32.0]


    def _tune_C(fit_predict, Xtr, ytr, n_inner=3):
        """Choose C on the training rows only, never on the outer test fold."""
        inner = list(StratifiedKFold(n_splits=n_inner, shuffle=True,
                                     random_state=0).split(Xtr, ytr))
        best, best_score = C_GRID[0], -np.inf
        for C in C_GRID:
            scores = []
            for itr, ite in inner:
                try:
                    pp = fit_predict(C, Xtr[itr], ytr[itr], Xtr[ite])
                    scores.append(roc_auc_score(ytr[ite], pp))
                except Exception:
                    scores.append(np.nan)
            m = np.nanmean(scores) if np.any(~np.isnan(scores)) else -np.inf
            if m > best_score:
                best_score, best = m, C
        return best


    def fp_fiml(C, Xtr, ytr, Xte):
        """Full information: the NaNs are marginalised, nothing is imputed."""
        m = MissSupportClassifier(copula=False, kernel='rbf', C=C).fit(Xtr, ytr)
        pp = m.predict_proba(Xte)
        return pp[:, 1] if pp.ndim == 2 else pp


    def fp_mice_svc(C, Xtr, ytr, Xte):
        """The matched counterpart: same kernel, same standardisation, MICE."""
        imp = IterativeImputer(max_iter=10, random_state=42).fit(Xtr)
        pipe = Pipeline([('scale', StandardScaler()),
                         ('clf', SVC(C=C, kernel='rbf', probability=True,
                                     random_state=0))])
        return pipe.fit(imp.transform(Xtr), ytr).predict_proba(
            imp.transform(Xte))[:, 1]


    def eval_tuned(fit_predict, label):
        aucs, briers, f1s, chosen = [], [], [], []
        for tr, te in FOLDS:
            C = _tune_C(fit_predict, X[tr], y[tr])
            chosen.append(C)
            pp = fit_predict(C, X[tr], y[tr], X[te])
            aucs.append(roc_auc_score(y[te], pp))
            briers.append(brier_score_loss(y[te], pp))
            f1s.append(f1_score(y[te], (pp > 0.5).astype(int), zero_division=0))
        res = {'auc': np.array(aucs), 'brier': np.array(briers),
               'f1': np.array(f1s)}
        print(f"  {label:<40}  AUC={res['auc'].mean():.4f}"
              f"+/-{res['auc'].std():.4f}  "
              f"Brier={res['brier'].mean():.4f}   C per fold={chosen}")
        return res


    # This is the slow cell in the notebook: four C values, three inner folds,
    # five outer folds, on both arms.
    res_fiml_tuned = eval_tuned(fp_fiml,     'FIML (MissSupport, tuned)')
    res_mice_tuned = eval_tuned(fp_mice_svc, 'MICE + SVC(rbf), tuned')
    auc_fiml, auc_mice = res_fiml_tuned['auc'], res_mice_tuned['auc']

    # The arms share folds, so the fold-to-fold variation is common to both and a
    # paired comparison removes it. Comparing the two standard deviations above
    # would be the wrong test.
    d = auc_fiml - auc_mice
    print()
    print("  paired difference per fold : %s" % np.round(d, 4).tolist())
    print("  mean difference            : %+.4f" % d.mean())
    print("  folds won by FIML          : %d of %d" % (int((d > 0).sum()), len(d)))
    try:
        from scipy import stats
        t, p = stats.ttest_rel(auc_fiml, auc_mice)
        print("  paired t-test              : t=%.3f  p=%.3f" % (t, p))
    except ImportError:
        pass

    # --------------------------------------------------------------------
    # notebook cell 8
    # --------------------------------------------------------------------
    print("=" * 80)
    print("  TIER 1: HOMOGENEOUS FIML ENSEMBLE")
    print("=" * 80)

    from sklearn.ensemble import BaggingClassifier

    BASE_C = 16.0        # the modal choice of the tuned comparison above
    N_MEMBERS = 10

    res_ens_homo = eval_folds(
        lambda: MissEnsemble(
            estimator=MissSupportClassifier(copula=False, kernel='rbf', C=BASE_C),
            n_estimators=N_MEMBERS,
            oob_score=True,
            random_state=42,
        ),
        X, y, FOLDS,
    )
    summarise("MissEnsemble(MissSupport x%d)" % N_MEMBERS, res_ens_homo)


    def bagged_svc_mice(Xtr, ytr, Xte):
        """Matched counterpart: same kernel, same C, same members, same bootstrap.
        The only difference is that the NaNs are imputed rather than marginalised."""
        imp = IterativeImputer(max_iter=10, random_state=42).fit(Xtr)
        base = Pipeline([('scale', StandardScaler()),
                         ('clf', SVC(C=BASE_C, kernel='rbf', probability=True,
                                     random_state=0))])
        bag = BaggingClassifier(base, n_estimators=N_MEMBERS, bootstrap=True,
                                random_state=42)
        return bag.fit(imp.transform(Xtr), ytr).predict_proba(
            imp.transform(Xte))[:, 1]


    auc_conv = np.array([roc_auc_score(y[te], bagged_svc_mice(X[tr], y[tr], X[te]))
                         for tr, te in FOLDS])
    print("  %-40s  AUC=%.4f+/-%.4f"
          % ('MICE + Bagging(SVC rbf x%d)' % N_MEMBERS,
             auc_conv.mean(), auc_conv.std()))

    d_bag = res_ens_homo['auc'] - auc_conv
    print("  paired difference        : %+.4f   folds won by FIML: %d of %d"
          % (d_bag.mean(), int((d_bag > 0).sum()), len(d_bag)))

    # --------------------------------------------------------------------
    # notebook cell 9
    # --------------------------------------------------------------------
    print("=" * 80)
    print("  TIER 2: HETEROGENEOUS FIML ENSEMBLE")
    print("=" * 80)


    def build_hetero():
        """Four FIML members spanning two model classes and two regularisations."""
        return MissEnsemble(
            estimators=[
                ('fiml_svm_tight', MissSupportClassifier(copula=False, kernel='rbf', C=BASE_C)),
                ('fiml_svm_wide',  MissSupportClassifier(copula=False, kernel='rbf',
                                                         C=BASE_C / 4)),
                ('fiml_logistic',  MissLogistic(copula=False, )),
                ('fiml_ridge',     MissRidgeClassifier(copula=False, )),
            ],
            weights=[3, 2, 1, 1],
            bootstrap=True,
            oob_score=True,
            random_state=42,
        )


    res_ens_hetero = eval_folds(build_hetero, X, y, FOLDS)
    summarise("MissEnsemble(4 mixed members)", res_ens_hetero)

    # --------------------------------------------------------------------
    # notebook cell 10
    # --------------------------------------------------------------------
    # Fit on one fold for inspection.
    tr0, te0 = FOLDS[0]
    ens_demo = build_hetero()
    ens_demo.fit(X[tr0], y[tr0])

    # Pass the names, or the importance table labels its rows X17, X18, ... which
    # cannot be read against the data dictionary. The member OOB scores below are
    # the interesting part: the two kernel members sit around 0.963 and the two
    # linear members around 0.952, which is the gap that pulls the weighted
    # average below the homogeneous ensemble.
    ens_demo.summary(feature_names=feature_names)

    # --------------------------------------------------------------------
    # notebook cell 11
    # --------------------------------------------------------------------
    # The arms below no longer all share one model class, so they are printed in
    # two blocks. Within a block the comparison is about the missing-data
    # treatment; across blocks it is about the model.
    print(f"\n{'Arm':<44} {'AUC':>8} {'F1':>8} {'Brier':>8}")
    print("-" * 72)
    print("  logistic model class")
    for name, res in strategy_results.items():
        print(f"    {name:<42} {res['auc'].mean():>8.4f} "
              f"{res['f1'].mean():>8.4f} {res['brier'].mean():>8.4f}")

    kernel_block = {
        'FIML (MissSupport, tuned)':        res_fiml_tuned,
        'MissEnsemble (homogeneous)':       res_ens_homo,
        'MissEnsemble (heterogeneous)':     res_ens_hetero,
    }
    print("  kernel model class")
    for name, res in kernel_block.items():
        print(f"    {name:<42} {res['auc'].mean():>8.4f} "
              f"{res['f1'].mean():>8.4f} {res['brier'].mean():>8.4f}")

    best_name = max(kernel_block, key=lambda k: kernel_block[k]['auc'].mean())
    print(f"\nBest arm by AUC: {best_name} "
          f"({kernel_block[best_name]['auc'].mean():.4f})")

    # --------------------------------------------------------------------
    # notebook cell 12
    # --------------------------------------------------------------------
    # ROC curves on fold 0. Every arm here is in the kernel model class, so the
    # curves differ only in how the missing values are handled.
    tr0, te0 = FOLDS[0]
    valid_te = ~np.isnan(y[te0])


    def svc_rbf_pipe():
        return Pipeline([('scale', StandardScaler()),
                         ('clf', SVC(C=BASE_C, kernel='rbf', probability=True,
                                     random_state=0))])


    roc_arms = {
        'Mean imputation':            ('strategy', 'mean'),
        'kNN imputation':             ('strategy', 'knn'),
        'MICE':                       ('strategy', 'mice'),
        'FIML (MissSupport)':         ('fiml',     None),
        'MissEnsemble (homogeneous)': ('homo',     None),
    }

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.colormaps['viridis']
    colors = [cmap(v) for v in np.linspace(0.05, 0.92, len(roc_arms))]

    for (name, (kind, key)), color in zip(roc_arms.items(), colors):
        if kind == 'strategy':
            imp = {'mean': SimpleImputer(strategy='mean'),
                   'knn':  KNNImputer(n_neighbors=5),
                   'mice': IterativeImputer(max_iter=10, random_state=42)}[key]
            Xtr = imp.fit_transform(X[tr0])
            Xte = imp.transform(X[te0])
            m = svc_rbf_pipe()
            m.fit(Xtr, y[tr0])
            proba = m.predict_proba(Xte)[:, 1]
        else:
            if kind == 'fiml':
                m = MissSupportClassifier(copula=False, kernel='rbf', C=BASE_C)
            else:
                m = MissEnsemble(
                    estimator=MissSupportClassifier(copula=False, kernel='rbf', C=BASE_C),
                    n_estimators=N_MEMBERS, random_state=42)
            m.fit(X[tr0], y[tr0])
            proba = m.predict_proba(X[te0])[:, 1]

        RocCurveDisplay.from_predictions(
            y[te0][valid_te], proba[valid_te], ax=ax, name=name, color=color)

    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.8)
    ax.set_title('ROC curves, Thyroid Sick (fold 0)\n'
                 'RBF kernel model class held fixed across every arm')
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
