# -*- coding: utf-8 -*-
"""Mixed-effects FIML and multiple imputation on repeated measures
===============================================================


Parkinson's Telemonitoring (UCI, Little et al. 2009): 5,875 voice
recordings from 42 patients over six months, predicting total UPDRS. 20% MAR
missingness is injected into the sixteen voice features.

Two things this example is careful about, and they matter more than the model
does.

Cross-validation is grouped by patient. Each patient contributes about 140
sessions, so a shuffled split places every held-out patient in the training
folds as well; measured on this data, 100% of them. A number obtained that way
describes tracking a patient already under monitoring, not predicting a new one,
and the difference is large: 2.60 RMSE against 10.43.

Under patient-grouped folds no arm beats the training mean. The best reaches an
R2 of 0.012 and the full-information arm is slightly negative, against a
response standard deviation of 10.70. The voice features carry ample information
for following a known patient and essentially none that transfers across people.
The mixed-effects gain is real but it belongs to the monitoring regime
specifically.

Generated from 01_MissMixed_MissImputer_Parkinsons.ipynb so the two cannot drift apart. The notebook is the place
to read this interactively; the script is for headless runs, for diffing in
version control, and for reading without a notebook renderer.

Run
---
    python 01_parkinsons_mixed.py
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
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    warnings.filterwarnings('ignore')


    from MissLearn import (
        MissLinear, MissMixedRegressor, MissImputer,
        MissKFold, miss_cross_val_score,
    )

    DATA_DIR = os.path.join(_HERE, 'example_data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print("MissLearn imported.  Data directory:", DATA_DIR)

    # --------------------------------------------------------------------
    # notebook cell 2
    # --------------------------------------------------------------------
    DATA_PATH = os.path.join(DATA_DIR, 'parkinsons_updrs.data')

    if not os.path.exists(DATA_PATH):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "parkinsons/telemonitoring/parkinsons_updrs.data")
        df_raw = pd.read_csv(url)
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"Downloaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"Loaded from cache: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    print("\nFirst 3 rows:")
    print(df_raw.head(3).to_string())

    # --------------------------------------------------------------------
    # notebook cell 3
    # --------------------------------------------------------------------
    # Extract components
    groups = df_raw['subject#'].values.astype(int)   # 42 patients
    y      = df_raw['total_UPDRS'].values.astype(float)

    FEATURE_COLS = [c for c in df_raw.columns
                    if c not in ('subject#', 'motor_UPDRS', 'total_UPDRS')]
    X = df_raw[FEATURE_COLS].values.astype(float)
    feature_names = FEATURE_COLS

    # Voice feature indices (columns 3 onward: Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE)
    VOICE_IDX = list(range(3, len(FEATURE_COLS)))  # skip age, sex, test_time

    n, p = X.shape
    n_groups = len(np.unique(groups))
    print(f"n={n}  p={p}  groups={n_groups} patients")
    print(f"total_UPDRS: mean={y.mean():.2f}  std={y.std():.2f}  "
          f"range=[{y.min():.1f}, {y.max():.1f}]")
    print(f"\nRecordings per patient (min/median/max): "
          f"{pd.Series(groups).value_counts().agg(['min','median','max']).values}")

    # --------------------------------------------------------------------
    # notebook cell 4
    # --------------------------------------------------------------------
    def inject_mar(X, voice_idx, rate=0.20, seed=42):
        """MAR into voice_idx columns only.  Rate is per-feature average."""
        rng = np.random.default_rng(seed)
        Xm = X.copy()
        n_voice = len(voice_idx)
        for k, j in enumerate(voice_idx):
            ref_j   = (k + 1) % n_voice
            ref_col = X[:, voice_idx[ref_j]]
            med     = np.nanmedian(ref_col)
            p_high  = min(rate * 4 / 3, 1.0)
            p_low   = min(rate * 2 / 3, 1.0)
            prob    = np.where(ref_col > med, p_high, p_low)
            Xm[rng.random(len(X)) < prob, j] = np.nan
        actual = np.isnan(Xm[:, voice_idx]).mean()
        return Xm, actual

    X_miss, actual_rate = inject_mar(X, VOICE_IDX, rate=0.20, seed=42)
    print(f"Target miss rate: 20.0%   Actual: {actual_rate:.1%}")
    print("\nMissing values per feature:")
    miss_pct = pd.Series(np.isnan(X_miss).mean(axis=0) * 100,
                         index=feature_names).round(1)
    print(miss_pct[miss_pct > 0].to_string())

    # --------------------------------------------------------------------
    # notebook cell 5
    # --------------------------------------------------------------------
    def cv_rmse_linear(X, y, n_splits=5, seed=42):
        """Per-fold RMSE for the pooled MissLinear model (SEs off for speed)."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        out = []
        for tr, te in kf.split(X):
            m = MissLinear(copula=False, compute_se=False).fit(X[tr], y[tr])
            v = ~np.isnan(y[te])
            out.append(np.sqrt(mean_squared_error(y[te][v], m.predict(X[te])[v])))
        return np.array(out)

    def cv_rmse_mixed(X, y, groups, n_splits=5, seed=42):
        """Per-fold RMSE for MissMixedRegressor under two prediction scenarios,
        fitting once per fold: population-level (unseen patient, no random
        intercept) and BLUP-adjusted (monitored patient, its intercept supplied).
        Standard errors are turned off here as they are not needed for RMSE."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pop, blup = [], []
        for tr, te in kf.split(X):
            m = MissMixedRegressor(copula=False, compute_se=False).fit(
                X[tr], y[tr], groups=groups[tr])
            v = ~np.isnan(y[te])
            pop.append(np.sqrt(mean_squared_error(
                y[te][v], m.predict(X[te])[v])))
            blup.append(np.sqrt(mean_squared_error(
                y[te][v], m.predict(X[te], groups=groups[te])[v])))
        return np.array(pop), np.array(blup)

    print("Fitting MissLinear (pooled, ignores patients)...")
    rmse_linear = cv_rmse_linear(X_miss, y)
    print(f"  RMSE: {rmse_linear.mean():.3f} +/- {rmse_linear.std():.3f}")

    print("\nFitting MissMixedRegressor (one fit per fold, two prediction modes)...")
    rmse_mixed_pop, rmse_mixed = cv_rmse_mixed(X_miss, y, groups)
    print(f"  unseen patient   (no random intercept) : "
          f"{rmse_mixed_pop.mean():.3f} +/- {rmse_mixed_pop.std():.3f}")
    print(f"  monitored patient (BLUP supplied)      : "
          f"{rmse_mixed.mean():.3f} +/- {rmse_mixed.std():.3f}")

    # --------------------------------------------------------------------
    # notebook cell 6
    # --------------------------------------------------------------------
    # Visualise per-fold RMSE for the three prediction scenarios
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = np.arange(1, 6)
    ax.plot(folds, rmse_linear,    'o-', color='#440154',
            label='MissLinear (pooled)')
    ax.plot(folds, rmse_mixed_pop, '^-', color='#31688E',
            label='MissMixed (unseen patient)')
    ax.plot(folds, rmse_mixed,     's-', color='#35B779',
            label='MissMixed (monitored patient, BLUP)')
    ax.set_xlabel('CV Fold')
    ax.set_ylabel('RMSE (total UPDRS)')
    ax.set_title("MissLinear vs MissMixedRegressor: per-fold RMSE\n"
                 "Parkinson's Telemonitoring, 20% MAR, 5-fold CV")
    ax.legend()
    fig.tight_layout()
    plt.show()

    improve = rmse_linear.mean() - rmse_mixed.mean()
    pct     = improve / rmse_linear.mean() * 100
    print(f"For a monitored patient, MissMixed cuts RMSE from "
          f"{rmse_linear.mean():.2f} to {rmse_mixed.mean():.2f} "
          f"({pct:.0f}% lower) by using the patient's random intercept.")
    print(f"For an unseen patient (no random intercept), MissMixed matches the "
          f"pooled model at {rmse_mixed_pop.mean():.2f}.")

    # --------------------------------------------------------------------
    # notebook cell 7
    # --------------------------------------------------------------------
    import copy
    mixed_full = MissMixedRegressor(copula=False, )
    mixed_full.fit(X_miss, y, groups=groups)
    mixed_full.summary()

    # --------------------------------------------------------------------
    # notebook cell 8
    # --------------------------------------------------------------------
    # Use a single held-out fold for the imputation demonstration
    kf       = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, te_idx = next(iter(kf.split(X_miss)))
    X_tr, y_tr = X_miss[tr_idx], y[tr_idx]
    X_te, y_te = X_miss[te_idx], y[te_idx]

    # Fit imputer on training data only
    imp = MissImputer(m=20, random_state=42)
    imp.fit(X_tr)
    print(f"MissImputer fitted.  Estimated joint MVN: mu shape={imp.mu_.shape}, "
          f"Sigma shape={imp.Sigma_.shape}")

    datasets_tr = imp.transform(X_tr)  # list of 20 complete training sets
    datasets_te = imp.transform(X_te)  # imputed using training-fit parameters

    print(f"Generated {len(datasets_tr)} complete training datasets, "
          f"each shape {datasets_tr[0].shape}")
    print(f"NaN remaining in first imputed dataset: {np.isnan(datasets_tr[0]).sum()}")

    # --------------------------------------------------------------------
    # notebook cell 9
    # --------------------------------------------------------------------
    # Fit Ridge on each imputed training set, predict on each imputed test set,
    # then average predictions (prediction combination under MI)
    valid_tr = ~np.isnan(y_tr)
    valid_te = ~np.isnan(y_te)
    preds_mi = []

    for X_imp_tr, X_imp_te in zip(datasets_tr, datasets_te):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_imp_tr[valid_tr], y_tr[valid_tr])
        preds_mi.append(ridge.predict(X_imp_te))

    y_pred_mi = np.mean(preds_mi, axis=0)
    rmse_mi   = np.sqrt(mean_squared_error(y_te[valid_te], y_pred_mi[valid_te]))

    # For comparison on the same fold: MissLinear (pooled)
    ml_fold = MissLinear(copula=False, compute_se=False)
    ml_fold.fit(X_tr, y_tr)
    rmse_ml = np.sqrt(mean_squared_error(
        y_te[valid_te], ml_fold.predict(X_te)[valid_te]))

    # MissMixed on the same fold, predicting monitored patients (BLUP supplied)
    mm_fold = MissMixedRegressor(copula=False, compute_se=False)
    mm_fold.fit(X_tr, y_tr, groups=groups[tr_idx])
    rmse_mm = np.sqrt(mean_squared_error(
        y_te[valid_te], mm_fold.predict(X_te, groups=groups[te_idx])[valid_te]))

    print("Single held-out fold RMSE comparison:")
    print(f"  MissLinear (FIML, pooled)               : {rmse_ml:.3f}")
    print(f"  MissMixed  (FIML, patient intercept)    : {rmse_mm:.3f}")
    print(f"  MissImputer (m=20) + Ridge              : {rmse_mi:.3f}")

    # --------------------------------------------------------------------
    # notebook cell 10
    # --------------------------------------------------------------------
    # Visualise prediction spread across 20 imputations for 20 test patients
    fig, ax = plt.subplots(figsize=(10, 4))
    sample_idx = np.where(valid_te)[0][:20]
    preds_arr  = np.array(preds_mi)[:, sample_idx]  # (20 imputations, 20 samples)

    ax.boxplot(preds_arr.T, positions=range(20),
               medianprops=dict(color='#FDE725', linewidth=2),
               boxprops=dict(color='#440154'),
               whiskerprops=dict(color='#440154'),
               capprops=dict(color='#440154'),
               flierprops=dict(marker='.', color='#aaaaaa', markersize=3))
    ax.scatter(range(20), y_te[valid_te][:20], color='#21908C', zorder=5,
               label='Observed total_UPDRS', s=30)
    ax.set_xlabel('Test observation (first 20)')
    ax.set_ylabel('total_UPDRS')
    ax.set_title('MissImputer: prediction spread across m=20 imputations\n'
                 '(box = model uncertainty from imputation; dot = observed value)')
    ax.legend()
    fig.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # notebook cell 11
    # --------------------------------------------------------------------
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.metrics import r2_score

    # Grouped folds: a held-out patient is genuinely unseen.
    GFOLDS = list(GroupKFold(n_splits=5).split(X_miss, y, groups=groups))
    print("grouped folds (train rows, test rows, test patients):")
    for tr, te in GFOLDS:
        print("   %5d %5d %4d" % (len(tr), len(te), len(np.unique(groups[te]))))


    def scaled_linear():
        """The conventional counterpart of MissLinear: same model class, same
        internal standardisation."""
        return Pipeline([('scale', StandardScaler()), ('m', LinearRegression())])


    STRATEGIES = [("Drop rows", 'drop_rows'), ("Drop cols", 'drop_cols'),
                   ("Mean imputation", 'mean'), ("kNN imputation", 'knn'),
                   ("MICE", 'mice')]


    def conv_arm(strategy):
        """One conventional arm over the grouped folds."""
        rmses, r2s = [], []
        for tr, te in GFOLDS:
            Xtr, ytr, Xte, yte = X_miss[tr], y[tr], X_miss[te], y[te]
            v = ~np.isnan(yte)
            Xte, yte = Xte[v], yte[v]
            mu = np.nanmean(Xtr, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)

            if strategy == 'drop_rows':
                cc = ~np.isnan(Xtr).any(axis=1)
                if cc.sum() < Xtr.shape[1] + 2:
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

            m = scaled_linear()
            m.fit(A, b)
            pred = m.predict(B)
            rmses.append(np.sqrt(mean_squared_error(yte, pred)))
            r2s.append(r2_score(yte, pred))
        return np.array(rmses), np.array(r2s)


    def fiml_flat_arm():
        """MissLinear: same flat model class, no imputation at all."""
        rmses, r2s = [], []
        for tr, te in GFOLDS:
            m = MissLinear(copula=False, compute_se=False).fit(X_miss[tr], y[tr])
            v = ~np.isnan(y[te])
            pred = m.predict(X_miss[te])[v]
            rmses.append(np.sqrt(mean_squared_error(y[te][v], pred)))
            r2s.append(r2_score(y[te][v], pred))
        return np.array(rmses), np.array(r2s)


    print("\nBlock A: running six arms over the grouped folds ...")
    blockA = {}
    for label, key in STRATEGIES:
        r = conv_arm(key)
        if r is None:
            print("   %-20s not applicable" % label)
            continue
        blockA[label] = r
        print("   %-20s done" % label)
    blockA['FIML (MissLinear)'] = fiml_flat_arm()
    print("   %-20s done" % 'FIML')

    # --------------------------------------------------------------------
    # notebook cell 12
    # --------------------------------------------------------------------
    print("BLOCK A   missing-data strategy, flat linear model class held fixed")
    print("          patient-grouped folds, so no patient is in both sides")
    print()
    print("  %-24s %16s %16s" % ("arm", "RMSE", "R2"))
    print("  " + "-" * 58)
    for label, (rm, r2) in blockA.items():
        print("  %-24s %8.3f+/-%.3f %8.3f+/-%.3f"
              % (label, rm.mean(), rm.std(), r2.mean(), r2.std()))

    convA = {k: v for k, v in blockA.items() if not k.startswith('FIML')}
    best = min(convA, key=lambda k: convA[k][0].mean())
    gap = blockA['FIML (MissLinear)'][0].mean() - convA[best][0].mean()
    print()
    print("  best conventional : %s (RMSE %.3f)" % (best, convA[best][0].mean()))
    print("  FIML              : RMSE %.3f" % blockA['FIML (MissLinear)'][0].mean())
    print("  difference        : %+.3f RMSE  ->  %s"
          % (gap, "FIML ahead" if gap < -0.05 else
                  ("FIML behind" if gap > 0.05 else "parity")))

    # --------------------------------------------------------------------
    # notebook cell 13
    # --------------------------------------------------------------------
    # BLOCK B: what the random intercept adds, and what the split choice is worth.
    #
    # Under grouped folds a held-out patient is genuinely new, so no random
    # intercept can exist for it and only the population-level prediction is
    # defined. That is the honest "new patient" scenario.
    rmse_pop_grouped = []
    for tr, te in GFOLDS:
        m = MissMixedRegressor(copula=False, compute_se=False).fit(
            X_miss[tr], y[tr], groups=groups[tr])
        v = ~np.isnan(y[te])
        rmse_pop_grouped.append(np.sqrt(mean_squared_error(
            y[te][v], m.predict(X_miss[te])[v])))
    rmse_pop_grouped = np.array(rmse_pop_grouped)

    print("BLOCK B   what the per-patient random intercept adds")
    print("          NOT comparable with block A: the model differs by more")
    print("          than its NaN handling")
    print()
    print("  %-46s %8s" % ("scenario", "RMSE"))
    print("  " + "-" * 58)
    print("  %-46s %8.3f" % ("new patient, grouped folds, no intercept",
                              rmse_pop_grouped.mean()))
    print("  %-46s %8.3f" % ("monitored patient, shuffled folds, BLUP",
                              rmse_mixed.mean()))
    print("  %-46s %8.3f" % ("flat FIML reference (block A)",
                              blockA['FIML (MissLinear)'][0].mean()))
    print()
    print("  Cost of pretending a monitored patient is new: %+.3f RMSE"
          % (rmse_pop_grouped.mean() - rmse_mixed.mean()))
    print("  This is the single most important number in this notebook: it is how")
    print("  much of the apparent accuracy comes from having seen the patient")
    print("  before, rather than from the voice features.")

    # --------------------------------------------------------------------
    # notebook cell 14
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={'width_ratios': [1.55, 1]})
    cmap = plt.colormaps['viridis']

    labels = list(blockA)
    means = [blockA[k][0].mean() for k in labels]
    sds = [blockA[k][0].std() for k in labels]
    axes[0].bar(range(len(labels)), means, yerr=sds, capsize=4,
                color=[cmap(v) for v in np.linspace(0.12, 0.88, len(labels))])
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha='right')
    axes[0].set_ylabel("RMSE (patient-grouped CV)")
    axes[0].set_title("Block A: missing-data strategy\n"
                      "flat linear model class held fixed")
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_axisbelow(True)

    bl = ["new patient\n(grouped)", "monitored patient\n(BLUP)",
          "flat FIML\n(block A)"]
    bv = [rmse_pop_grouped.mean(), rmse_mixed.mean(),
          blockA['FIML (MissLinear)'][0].mean()]
    axes[1].bar(range(3), bv,
                color=[cmap(v) for v in (0.20, 0.55, 0.85)])
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(bl)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Block B: what the random intercept adds\n"
                      "not comparable with block A")
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Parkinsons telemonitoring: strategy comparison, then the "
                 "modelling gain, kept apart")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
