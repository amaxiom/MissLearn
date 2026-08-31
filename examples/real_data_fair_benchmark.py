# -*- coding: utf-8 -*-
"""Fair real-data benchmarks: strategy comparison, model class held fixed.

Putting a linear FIML model in the same table as a boosted-tree ensemble
confounds two separate questions.  A GBM may beat a logistic model simply
because it has more capacity, which says nothing about whether FIML is a
good way to handle missing values.  This harness therefore varies exactly
one thing:

  For each model class, the same estimator is trained after listwise
  deletion, column deletion, mean / kNN / MICE imputation, and by native
  FIML.  This is the question the library exists to answer, and it is the
  only question a difference between these arms can answer.

No tree ensemble appears here, because no comparison in this file is allowed
to differ in model capacity.

Run with the Anaconda interpreter:
    python benchmarks/real_data_fair_benchmark.py            # all datasets
    python benchmarks/real_data_fair_benchmark.py pima       # one dataset
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import MissLearn as ML
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

DATA_DIR = os.path.join(HERE, "example_data")
def out_csv(keys):
    """Per-dataset file names, so concurrent runs never clash."""
    stem = "real_data_fair_results"
    if len(keys) == 1:
        return os.path.join(HERE, f"{stem}_{keys[0]}.csv")
    return os.path.join(HERE, f"{stem}.csv")



# ------------------------------------------------------------------ loaders
def load_pima():
    path = os.path.join(DATA_DIR, "pima_diabetes.csv")
    df = pd.read_csv(path)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df.loc[df[c] == 0, c] = np.nan
    y = df["Outcome"].values.astype(float)
    X = df.drop(columns=["Outcome"]).values.astype(float)
    return X, y, "Pima diabetes (8 continuous features, native missingness)"


def load_thyroid():
    COLUMNS = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
               'on_antithyroid_medication', 'illness_flag', 'pregnant',
               'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
               'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
               'hypopituitary', 'psych', 'TSH_measured', 'TSH',
               'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured',
               'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
               'referral_source', 'target']
    BINARY = ['sex', 'on_thyroxine', 'query_on_thyroxine',
              'on_antithyroid_medication', 'illness_flag', 'pregnant',
              'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
              'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
              'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
              'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
    frames = []
    for fname in ["sick.data", "sick.test"]:
        frames.append(pd.read_csv(os.path.join(DATA_DIR, fname)))
    df = pd.concat(frames, ignore_index=True)
    df['target'] = (df['target'].astype(str).str.strip()
                    .str.split('|').str[0].str.strip().str.rstrip('.'))
    y = (df['target'] == 'sick').astype(float).values
    for col in BINARY:
        df[col] = df[col].map({'t': 1.0, 'f': 0.0, 'M': 1.0, 'F': 0.0})
    df = df.drop(columns=['target', 'referral_source', 'TSH_measured',
                          'T3_measured', 'TT4_measured', 'T4U_measured',
                          'FTI_measured', 'TBG_measured', 'TBG'])
    X = df.values.astype(float)
    return X, y, "Thyroid sick (indicator-dominated features, native missingness)"


def load_credit():
    path = os.path.join(DATA_DIR, "crx.data")
    df = pd.read_csv(path, header=None, na_values="?")
    df.columns = [f"A{i+1}" for i in range(16)]
    y = (df["A16"] == "+").astype(float).values
    df = df.drop(columns=["A16"])
    CONT = ["A2", "A3", "A8", "A11", "A14", "A15"]
    BIN = {"A1": ("a", "b"), "A9": ("f", "t"), "A10": ("f", "t"),
           "A12": ("f", "t")}
    MULTI = ["A4", "A5", "A6", "A7", "A13"]
    blocks = []
    for c in CONT:
        blocks.append(pd.to_numeric(df[c], errors="coerce").values[:, None])
    for c, (lo, hi) in BIN.items():
        blocks.append(df[c].map({lo: 0.0, hi: 1.0}).values[:, None])
    for c in MULTI:
        for cat in sorted(df[c].dropna().unique())[:-1]:
            blocks.append(np.where(df[c].isna(), np.nan,
                                   (df[c] == cat).astype(float))[:, None])
    X = np.hstack(blocks).astype(float)
    return X, y, "Credit approval (mixed encoded features, native missingness)"


DATASETS = {"pima": load_pima, "thyroid": load_thyroid, "credit": load_credit}


# -------------------------------------------------------------- strategies
def _prep(strategy, X_tr, y_tr, X_te, seed=42):
    """Return (X_tr', y_tr', X_te') for a conventional missing-data strategy,
    or None if the strategy is not usable on this fold."""
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    fill = lambda A: np.where(np.isnan(A), col_means, A)

    if strategy == "native":
        return X_tr, y_tr, X_te
    if strategy == "drop_rows":
        cc = ~np.isnan(X_tr).any(axis=1)
        if cc.sum() < X_tr.shape[1] + 2 or len(np.unique(y_tr[cc])) < 2:
            return None
        return X_tr[cc], y_tr[cc], fill(X_te)
    if strategy == "drop_cols":
        keep = ~np.isnan(X_tr).any(axis=0)
        if keep.sum() == 0:
            return None
        mu = col_means[keep]
        Xte = X_te[:, keep]
        return (X_tr[:, keep], y_tr,
                np.where(np.isnan(Xte), mu, Xte))
    imp = {"mean": SimpleImputer(strategy="mean"),
           "knn": KNNImputer(n_neighbors=5),
           "mice": IterativeImputer(max_iter=10, random_state=seed)}[strategy]
    return imp.fit_transform(X_tr), y_tr, imp.transform(X_te)


def eval_arm(factory, X, y, folds, strategy, is_fiml=False):
    aucs, f1s, briers = [], [], []
    for tr, te in folds:
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        if is_fiml:
            Xtr2, ytr2, Xte2 = X_tr, y_tr, X_te
        else:
            got = _prep(strategy, X_tr, y_tr, X_te)
            if got is None:
                return None
            Xtr2, ytr2, Xte2 = got
        try:
            m = factory()
            m.fit(Xtr2, ytr2)
            proba = m.predict_proba(Xte2)[:, 1]
            pred = m.predict(Xte2)
        except Exception:
            return None
        aucs.append(roc_auc_score(y_te, proba))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        briers.append(brier_score_loss(y_te, proba))
    return dict(auc=np.mean(aucs), auc_sd=np.std(aucs),
                f1=np.mean(f1s), brier=np.mean(briers))


CONV = [("Drop rows", "drop_rows"), ("Drop cols", "drop_cols"),
        ("Mean imp.", "mean"), ("kNN imp.", "knn"), ("MICE", "mice")]

# Part A: (class label, sklearn factory, MissLearn factory)
#
# Fairness in the OTHER direction matters too.  MissSupport, MissNeighbors
# and the penalized models standardise their features internally, so a bare
# scale-sensitive scikit-learn baseline would lose on preprocessing rather
# than on its missing-data strategy, and MissLearn would look good for the
# wrong reason.  Every scale-sensitive baseline is therefore wrapped in a
# StandardScaler (fitted on the already-completed training matrix), which is
# what a competent practitioner would do.  GaussianNB is genuinely
# scale-invariant and tree ensembles are invariant to monotone feature
# transforms, so neither is wrapped.
def scaled(make_est):
    return lambda: Pipeline([("scale", StandardScaler()), ("clf", make_est())])


FAMILIES = [
    ("Logistic regression", scaled(lambda: LogisticRegression(max_iter=2000)),
     lambda: ML.MissLogistic(compute_se=False), "MissLogistic"),
    ("Generative Gaussian", lambda: GaussianNB(),
     lambda: ML.MissBayesClassifier(), "MissBayes"),
    ("Support vector", scaled(lambda: SVC(probability=True, random_state=0)),
     lambda: ML.MissSupportClassifier(), "MissSupport"),
    ("k-nearest neighbours", scaled(lambda: KNeighborsClassifier()),
     lambda: ML.MissNeighborsClassifier(), "MissNeighbors"),
]



def run(key):
    X, y, desc = DATASETS[key]()
    folds = list(StratifiedKFold(5, shuffle=True,
                                 random_state=42).split(X, y))
    print("\n" + "#" * 78)
    print("# " + desc)
    print("# n=%d  p=%d  positive=%.1f%%  cells missing=%.2f%%  rows w/ hole=%.1f%%"
          % (X.shape[0], X.shape[1], 100 * y.mean(),
             100 * np.isnan(X).mean(), 100 * np.isnan(X).any(axis=1).mean()))
    print("#" * 78)
    rows = []

    print("\nMissing-data strategy, model class held fixed")
    print("-" * 78)
    for cls, sk, ml, mlname in FAMILIES:
        res = {}
        for lbl, strat in CONV:
            res[lbl] = eval_arm(sk, X, y, folds, strat)
        res["FIML (%s)" % mlname] = eval_arm(ml, X, y, folds, None,
                                             is_fiml=True)
        print("\n  %s   [same estimator; only the missing-data strategy differs]"
              % cls)
        print("    %-22s %8s %8s %8s" % ("strategy", "AUC", "F1", "Brier"))
        best_conv = max(v["auc"] for k, v in res.items()
                        if v and not k.startswith("FIML"))
        for lbl, v in res.items():
            if v is None:
                print("    %-22s %8s" % (lbl, "n/a"))
                continue
            star = ""
            if lbl.startswith("FIML"):
                star = "  <-- FIML best" if v["auc"] >= best_conv - 1e-12 \
                    else "  (best conventional ahead by %.4f)" % (best_conv - v["auc"])
            print("    %-22s %8.4f %8.4f %8.4f%s"
                  % (lbl, v["auc"], v["f1"], v["brier"], star))
            rows.append(dict(dataset=key, model_class=cls,
                             arm=lbl, **v))

    return rows


if __name__ == "__main__":
    keys = sys.argv[1:] or list(DATASETS)
    path = out_csv(keys)
    allrows = []
    for k in keys:
        allrows += run(k)
        pd.DataFrame(allrows).to_csv(path, index=False)
    print("\nsaved", path)
