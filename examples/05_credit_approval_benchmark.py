# -*- coding: utf-8 -*-
"""Finance benchmark: UCI Credit Approval (native missing values)
==============================================================


Real incomplete financial data: 690 credit-card applications, 15 mixed
categorical and continuous features, '?' entries are genuinely missing
(no injection needed). Two conditions are benchmarked with 5-fold
stratified CV: the native missingness (about 1% of cells, 5.4% of rows)
and an amplified condition with an additional 20% MAR injected into every
continuous attribute on top of the native holes. Arms: drop rows, mean
imputation, MICE, FIML (MissLogistic) and generative FIML (MissBayes).

Every arm is scored on the same folds, and the first four hold the logistic
model class fixed so that a difference between them is attributable to the
missing-data treatment rather than to model capacity. MissBayes is reported
alongside them as a change of model class, not as a like-for-like rival.

Run with the Anaconda interpreter:
    python examples/05_credit_approval_benchmark.py
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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

DATA_DIR = os.path.join(HERE, "example_data")
PATH = os.path.join(DATA_DIR, "crx.data")

if not os.path.exists(PATH):
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "credit-screening/crx.data")
    df = pd.read_csv(url, header=None, na_values="?")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PATH, index=False, header=False)
    print("downloaded crx.data:", df.shape)
else:
    df = pd.read_csv(PATH, header=None, na_values="?")
    print("loaded cached crx.data:", df.shape)

df.columns = [f"A{i+1}" for i in range(16)]
y = (df["A16"] == "+").astype(float).values
df = df.drop(columns=["A16"])

CONT = ["A2", "A3", "A8", "A11", "A14", "A15"]
BIN = {"A1": ("a", "b"), "A9": ("f", "t"), "A10": ("f", "t"),
       "A12": ("f", "t")}
MULTI = ["A4", "A5", "A6", "A7", "A13"]

blocks, names = [], []
for c in CONT:
    blocks.append(pd.to_numeric(df[c], errors="coerce").values[:, None])
    names.append(c)
for c, (lo, hi) in BIN.items():
    v = df[c].map({lo: 0.0, hi: 1.0}).values[:, None]
    blocks.append(v)
    names.append(c)
for c in MULTI:
    cats = sorted(df[c].dropna().unique())[:-1]     # drop one level
    for cat in cats:
        v = np.where(df[c].isna(), np.nan,
                     (df[c] == cat).astype(float))[:, None]
        blocks.append(v)
        names.append(f"{c}={cat}")
X = np.hstack(blocks).astype(float)
n, p = X.shape
print(f"n={n} p={p} native cell missingness "
      f"{np.isnan(X).mean():.3%}, rows with any hole "
      f"{np.isnan(X).any(axis=1).mean():.1%}")


def amplify(X, rate, seed=42):
    """Additional MAR into the continuous attributes."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    cont_idx = [names.index(c) for c in CONT]
    for k, j in enumerate(cont_idx):
        ref = cont_idx[(k + 1) % len(cont_idx)]
        med = np.nanmedian(X[:, ref])
        prob = np.where(np.nan_to_num(X[:, ref], nan=med) > med,
                        rate * 4 / 3, rate * 2 / 3)
        Xm[rng.random(len(X)) < prob, j] = np.nan
    return Xm


def run(Xc, label):
    res = []
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    arms = ["Drop rows", "Mean", "MICE", "FIML (MissLogistic)",
            "FIML (MissBayes)"]
    scores = {a: {"acc": [], "auc": [], "brier": []} for a in arms}
    for tr, te in skf.split(Xc, y):
        Xtr, Xte, ytr, yte = Xc[tr], Xc[te], y[tr], y[te]

        def score(name, pp):
            scores[name]["acc"].append(accuracy_score(yte, pp > 0.5))
            scores[name]["auc"].append(roc_auc_score(yte, pp))
            scores[name]["brier"].append(brier_score_loss(yte, pp))

        keep = ~np.isnan(Xtr).any(axis=1)
        mu = np.nanmean(Xtr, axis=0)
        Xte_f = np.where(np.isnan(Xte), mu, Xte)
        if keep.sum() > 30 and len(np.unique(ytr[keep])) == 2:
            lr = LogisticRegression(max_iter=2000).fit(Xtr[keep], ytr[keep])
            score("Drop rows", lr.predict_proba(Xte_f)[:, 1])
        im = SimpleImputer()
        lr = LogisticRegression(max_iter=2000).fit(im.fit_transform(Xtr), ytr)
        score("Mean", lr.predict_proba(im.transform(Xte))[:, 1])
        im = IterativeImputer(max_iter=10, random_state=0)
        lr = LogisticRegression(max_iter=2000).fit(im.fit_transform(Xtr), ytr)
        score("MICE", lr.predict_proba(im.transform(Xte))[:, 1])
        m = ML.MissLogistic(copula=False, compute_se=False).fit(Xtr, ytr)
        score("FIML (MissLogistic)", m.predict_proba(Xte)[:, 1])
        m = ML.MissBayesClassifier(copula=False, ).fit(Xtr, ytr)
        score("FIML (MissBayes)", m.predict_proba(Xte)[:, 1])

    for a in arms:
        if not scores[a]["acc"]:
            continue
        res.append(dict(
            condition=label, arm=a,
            acc=np.mean(scores[a]["acc"]), acc_sd=np.std(scores[a]["acc"]),
            auc=np.mean(scores[a]["auc"]), auc_sd=np.std(scores[a]["auc"]),
            brier=np.mean(scores[a]["brier"]),
            brier_sd=np.std(scores[a]["brier"])))
        print(label, a,
              "acc=%.3f auc=%.3f brier=%.3f" % (
                  res[-1]["acc"], res[-1]["auc"], res[-1]["brier"]),
              flush=True)
    return res


all_rows = run(X, "native")
Xa = amplify(X, 0.20)
print("amplified cell missingness %.3f%%" % (np.isnan(Xa).mean() * 100))
all_rows += run(Xa, "amplified")
pd.DataFrame(all_rows).to_csv(
    os.path.join(HERE, "credit_approval_results.csv"), index=False)
print("CREDIT BENCHMARK DONE", flush=True)
