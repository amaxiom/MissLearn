"""
benchmark_test_suite.py
=======================
MissLearn model family comparison and scenario coverage benchmarks.

Division of labour
------------------
This test suite complements the benchmark notebooks in benchmarks/.
Each covers different questions and should NOT be run as a substitute
for the other.

  benchmarks/MissLinear_Benchmark.ipynb  (and future algorithm notebooks)
  ↳ Question: "Does FIML preserve performance compared to every missingness
               handling strategy on controlled synthetic data?"
  ↳ Covers:  Drop Rows / Drop Cols / Mean / KNN / MICE / FIML
             Rich visuaisations (bars, strips, gain heatmaps, t-tests)
             One algorithm at a time, three synthetic datasets

  THIS FILE
  ↳ Question: "Which MissLearn model family is best for this scenario,
               and does any family struggle on non-synthetic conditions?"
  ↳ Covers:  All MissLearn families simultaneously (Linear, Ridge, LASSO,
             Neighbors, Bayes, Support, Gaussian, Mixed)
             Real missing data (Tier 2)
             MCAR / MAR / MNAR mechanism sweep (Tier 3)
             Grouped/longitudinal data with MissMixed (Tier 4)
             Multi-class classification (Iris, MissMulticlass)

Baseline policy
---------------
  Tier 1 (synthetic MAR):  MICE + LinearModel only.
    MICE is the strongest conventional imputation competitor; the benchmark
    notebooks establish the full CC / Mean / KNN / MICE comparison exhaustively.
    A single anchor keeps Tier 1 focused on inter-model comparison.

  Tier 2 (real missing data):  all four baselines retained.
    Real missingness is not covered by the benchmark notebooks; the full
    baseline set provides necessary context here.

  Tier 3 (mechanism sweep):  CC + MICE (anchor extremes).
    Already the minimal set needed to show mechanism sensitivity.

  Tier 4 (grouped data):  all four baselines retained.
    MissMixed is unique to this suite; full context is warranted.

  IMPORTANT, on reading these tables
  ----------------------------------
  The MICE anchor in every tier uses a LINEAR model (LinearRegression or
  LogisticRegression).  Rows for MissSupport (kernel SVM), MissGaussian
  (Gaussian process) and MissNeighbors are therefore NOT strategy comparisons
  against it: those families differ from the anchor in model capacity as well
  as in missing-data treatment, so a gap between them and the anchor is not
  evidence that FIML beats MICE.  This file answers "which family suits this
  scenario", and cross-family rows are to be read that way only.

  For a controlled missing-data comparison, where the model class is held fixed
  and each MissLearn family is matched against its own scikit-learn counterpart
  under every strategy, use benchmarks/*_Benchmark.ipynb (synthetic) or
  benchmarks/real_data_fair_benchmark.py (real data).  Those harnesses also give
  the conventional arms the same internal standardisation the MissLearn
  estimators apply, and tune the regularisation strength on both arms, both of
  which are needed before a difference can be attributed to the missing-data
  treatment.

Structure
---------
Tier 1 -- complete datasets, synthetic MAR at a fixed rate:
  bench_energy_efficiency()   Energy Efficiency  n=768  p=8   regression
  bench_wisconsin()           Wisconsin          n=569  p=10  binary clf
  bench_iris()                Iris               n=150  p=4   3-class clf

Tier 2 -- real missing data (no injection):
  bench_auto_mpg()            Auto MPG           n=392  p=7   regression

Tier 3 -- missingness mechanism sweep over MCAR / MAR / MNAR:
  bench_esol_sweep()          ESOL               n=1128 p=6   regression

Tier 4 -- grouped / longitudinal data:
  bench_mixed()               Radon              n=919  p=2   reg + binary clf

MissLearn models evaluated
---------------------------
  Regression:   MissLinear, MissRidgeRegressor, MissLASSORegressor,
                MissNeighborsRegressor, MissBayesRegressor, MissSupportRegressor,
                MissGaussianRegressor (only when n <= GP_N_THRESHOLD=400)
  Binary clf:   MissLogistic, MissRidgeClassifier, MissLASSOClassifier,
                MissNeighborsClassifier, MissBayesClassifier, MissSupportClassifier,
                MissGaussianClassifier (only when n <= GP_N_THRESHOLD)
  Multiclass:   above binary classifiers wrapped in MissMulticlass (Iris only)
  Mixed:        MissMixedRegressor, MissMixedClassifier (Tier 4 only)

Usage
-----
  from benchmark_test_suite import (
      bench_energy_efficiency, bench_wisconsin, bench_iris,
      bench_auto_mpg, bench_esol_sweep, bench_mixed,
  )
  results = bench_energy_efficiency(miss_rate=0.20, plot=True)
  results = bench_esol_sweep(plot=True)
"""

from __future__ import annotations

import gc
import os
import sys
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Ensure tests/ is on the path so prepare_datasets can be imported directly
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
from prepare_datasets import (
    load_energy_efficiency,
    load_wisconsin,
    load_iris,
    load_auto_mpg,
    load_esol,
    load_radon,
    prepare_all as _prepare_all,
    DATA_DIR,
)

# ---------------------------------------------------------------------------
# MissLearn imports
# ---------------------------------------------------------------------------
_ML_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "MissLearn")
if _ML_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_TESTS_DIR))

from MissLearn import (
    MissLinear,
    MissLogistic,
    MissRidgeRegressor,  MissRidgeClassifier,
    MissLASSORegressor,  MissLASSOClassifier,
    MissNeighborsRegressor, MissNeighborsClassifier,
    MissBayesRegressor,  MissBayesClassifier,
    MissSupportRegressor, MissSupportClassifier,
    MissGaussianRegressor, MissGaussianClassifier,
    MissMixedRegressor,  MissMixedClassifier,
    MissMulticlass,
)

# ---------------------------------------------------------------------------
# Sklearn imports
# ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GP_N_THRESHOLD = 400   # skip MissGaussian for datasets with n > this
RNG_SEED       = 42
N_SPLITS       = 5

# Viridis palette
_C_GREY = "#aaaaaa"    # grey         (baselines)

# Canonical model order: used in all tables and plots for cross-figure consistency
_MODEL_ORDER_REG = [
    "MissLinear",
    "MissRidgeRegressor",
    "MissLASSORegressor",
    "MissNeighborsRegressor",
    "MissBayesRegressor",
    "MissSupportRegressor",
    "MissGaussianRegressor",
    "MissMixedRegressor",
]
_MODEL_ORDER_CLF = [
    "MissLogistic",
    "MissRidgeClassifier",
    "MissLASSOClassifier",
    "MissNeighborsClassifier",
    "MissBayesClassifier",
    "MissSupportClassifier",
    "MissGaussianClassifier",
    "MissMixedClassifier",
]
# For multiclass (MissMulticlass-wrapped), prefix MC()
_MODEL_ORDER_MC = [f"MC({m})" for m in _MODEL_ORDER_CLF]
# Baseline methods always listed last, in this fixed order
_BASELINE_ORDER_REG = ["CC + LinearReg", "Mean + LinearReg", "KNN + LinearReg", "MICE + LinearReg"]
_BASELINE_ORDER_CLF = ["CC + LogReg",    "Mean + LogReg",    "KNN + LogReg",    "MICE + LogReg"]
# Slim variants (Tier 1 only)
_BASELINE_ORDER_REG_SLIM = ["MICE + LinearReg"]
_BASELINE_ORDER_CLF_SLIM = ["MICE + LogReg"]

# Persistent viridis colour per MissLearn model family (same across all plots).
# We use explicit hex strings (not RGBA tuples) so matplotlib receives a
# consistent type regardless of NumPy version.  The range 0.10 to 0.92 avoids
# the near-black end of viridis so every family colour is clearly visible.
import matplotlib.pyplot as _plt
import matplotlib.colors as _mcolors
import numpy as _np
_FIML_CMAP   = _plt.cm.viridis
_FAMILIES    = ["Linear", "Logistic", "Ridge", "LASSO",
                "Neighbors", "Bayes", "Support", "Gaussian", "Mixed"]
_N_FAMILIES  = len(_FAMILIES)
_FAMILY_COLORS = {
    name: _mcolors.to_hex(_FIML_CMAP(v))
    for name, v in zip(
        _FAMILIES,
        _np.linspace(0.20, 0.95, _N_FAMILIES),
    )
}

def _model_color(method_name):
    """Return the consistent viridis colour for a MissLearn model, grey for baselines.

    Baselines are checked first so that names like 'MICE + LinearReg' are never
    matched against the 'Linear' family key.
    """
    if not (method_name.startswith("Miss") or method_name.startswith("MC(")):
        return _C_GREY
    for family, color in _FAMILY_COLORS.items():
        if family in method_name:
            return color
    return _C_GREY


def _sort_agg(agg, task="regression", n_classes=2, slim=False):
    """
    Return a new OrderedDict with methods in canonical order:
    baselines first (in fixed order), then MissLearn models (in fixed order).
    Unknown methods are appended at the end.
    """
    from collections import OrderedDict
    if task == "regression":
        baseline_order = _BASELINE_ORDER_REG_SLIM if slim else _BASELINE_ORDER_REG
        model_order    = _MODEL_ORDER_REG
    elif n_classes > 2:
        baseline_order = _BASELINE_ORDER_CLF_SLIM if slim else _BASELINE_ORDER_CLF
        model_order    = _MODEL_ORDER_MC
    else:
        baseline_order = _BASELINE_ORDER_CLF_SLIM if slim else _BASELINE_ORDER_CLF
        model_order    = _MODEL_ORDER_CLF

    ordered = OrderedDict()
    for k in baseline_order:
        if k in agg:
            ordered[k] = agg[k]
    for k in model_order:
        if k in agg:
            ordered[k] = agg[k]
    # Any remaining unknown methods
    for k in agg:
        if k not in ordered:
            ordered[k] = agg[k]
    return ordered

# ---------------------------------------------------------------------------
# Missingness injectors
# ---------------------------------------------------------------------------

def inject_mcar(X: np.ndarray, rate: float, seed: int = 42):
    """
    Missing Completely At Random: each cell independently masked with
    probability *rate*.

    Returns (Xm, actual_rate).
    """
    rng  = np.random.default_rng(seed)
    Xm   = X.astype(np.float64).copy()
    mask = rng.uniform(0.0, 1.0, Xm.shape) < rate
    Xm[mask] = np.nan
    return Xm, float(np.isnan(Xm).mean())


def inject_mar(X: np.ndarray, rate: float, seed: int = 42):
    """
    Missing At Random: for column j, missingness probability depends on
    whether column (j+1) % p lies above or below its median.  Rows above
    the median are masked at 2x the rate of rows below, so the overall
    rate matches the target.

    MAR is satisfied: missingness depends only on observed data.

    Returns (Xm, actual_rate).
    """
    rng  = np.random.default_rng(seed)
    Xm   = X.astype(np.float64).copy()
    n, p = Xm.shape
    for j in range(p):
        cond  = Xm[:, (j + 1) % p]
        med   = np.nanmedian(cond)
        base  = rate * (2.0 / 3.0)
        probs = np.where(cond > med, 2.0 * base, base)
        probs = np.clip(probs, 0.0, 1.0)
        Xm[rng.uniform(0.0, 1.0, n) < probs, j] = np.nan
    return Xm, float(np.isnan(Xm).mean())


def inject_mnar(X: np.ndarray, rate: float, seed: int = 42):
    """
    Missing Not At Random: for column j, missingness depends on the value
    in column j itself (high values more likely to be missing).  Not
    identifiable from observed data alone; creates systematic bias for
    imputation-based baselines.

    Returns (Xm, actual_rate).
    """
    rng  = np.random.default_rng(seed)
    Xm   = X.astype(np.float64).copy()
    n, p = Xm.shape
    for j in range(p):
        col   = Xm[:, j]
        med   = np.nanmedian(col)
        base  = rate * (2.0 / 3.0)
        probs = np.where(col > med, 2.0 * base, base)
        probs = np.clip(probs, 0.0, 1.0)
        Xm[rng.uniform(0.0, 1.0, n) < probs, j] = np.nan
    return Xm, float(np.isnan(Xm).mean())


_INJECTORS = {"MCAR": inject_mcar, "MAR": inject_mar, "MNAR": inject_mnar}

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_reg_models(n: int) -> dict:
    """Return {name: unfitted_model} for all regression MissLearn families."""
    models = {
        "MissLinear":           MissLinear(compute_se=False, max_iter=2000, tol=1e-8),
        "MissRidgeRegressor":   MissRidgeRegressor(alpha=1.0),
        "MissLASSORegressor":   MissLASSORegressor(alpha=0.1),
        "MissNeighborsRegressor": MissNeighborsRegressor(n_neighbors=5),
        "MissBayesRegressor":   MissBayesRegressor(),
        "MissSupportRegressor": MissSupportRegressor(C=1.0),
    }
    if n <= GP_N_THRESHOLD:
        models["MissGaussianRegressor"] = MissGaussianRegressor()
    return models


def _make_clf_models(n: int, n_classes: int = 2, gp_threshold: int = GP_N_THRESHOLD) -> dict:
    """Return {name: unfitted_model} for all classification MissLearn families."""
    binary_ctors = {
        "MissLogistic":           lambda: MissLogistic(compute_se=False),
        "MissRidgeClassifier":    lambda: MissRidgeClassifier(alpha=1.0),
        "MissLASSOClassifier":    lambda: MissLASSOClassifier(alpha=0.1),
        "MissNeighborsClassifier": lambda: MissNeighborsClassifier(n_neighbors=5),
        "MissBayesClassifier":    lambda: MissBayesClassifier(),
        "MissSupportClassifier":  lambda: MissSupportClassifier(C=1.0),
    }
    if n <= gp_threshold:
        binary_ctors["MissGaussianClassifier"] = lambda: MissGaussianClassifier()

    if n_classes == 2:
        return {name: ctor() for name, ctor in binary_ctors.items()}
    else:
        # Wrap each binary classifier in MissMulticlass for K > 2
        return {
            f"MC({name})": MissMulticlass(ctor())
            for name, ctor in binary_ctors.items()
        }


def _make_reg_baselines() -> dict:
    """Return {name: (imputer_or_None, model)} for regression baselines."""
    return {
        "CC + LinearReg":   (None,                          LinearRegression()),
        "Mean + LinearReg": (SimpleImputer(strategy="mean"), LinearRegression()),
        "KNN + LinearReg":  (KNNImputer(n_neighbors=5),     LinearRegression()),
        "MICE + LinearReg": (IterativeImputer(max_iter=10, random_state=RNG_SEED),
                             LinearRegression()),
    }


def _make_clf_baselines(n_classes: int = 2) -> dict:
    """Return {name: (imputer_or_None, model)} for classification baselines.

    Full four-method set; used for Tier 2 (real data) and Tier 4 (grouped).
    For Tier 1 (synthetic) use _make_clf_baselines_slim(); the benchmark
    notebooks cover the full baseline comparison exhaustively.
    """
    lr_kw = dict(max_iter=1000, random_state=RNG_SEED)
    if n_classes > 2:
        lr_kw["multi_class"] = "ovr"
    return {
        "CC + LogReg":   (None,                          LogisticRegression(**lr_kw)),
        "Mean + LogReg": (SimpleImputer(strategy="mean"), LogisticRegression(**lr_kw)),
        "KNN + LogReg":  (KNNImputer(n_neighbors=5),     LogisticRegression(**lr_kw)),
        "MICE + LogReg": (IterativeImputer(max_iter=10, random_state=RNG_SEED),
                          LogisticRegression(**lr_kw)),
    }


def _make_reg_baselines_slim() -> dict:
    """Single MICE anchor for Tier 1 model-family comparison benchmarks.

    MICE is the strongest conventional imputation competitor; if MissLearn
    beats MICE it beats CC / Mean / KNN too (established by the benchmark
    notebooks).  A single baseline keeps Tier 1 focused on the inter-model
    comparison rather than replicating the baseline story.
    """
    return {
        "MICE + LinearReg": (
            IterativeImputer(max_iter=10, random_state=RNG_SEED),
            LinearRegression(),
        ),
    }


def _make_clf_baselines_slim(n_classes: int = 2) -> dict:
    """Single MICE anchor for Tier 1 model-family comparison benchmarks."""
    lr_kw = dict(max_iter=1000, random_state=RNG_SEED)
    if n_classes > 2:
        lr_kw["multi_class"] = "ovr"
    return {
        "MICE + LogReg": (
            IterativeImputer(max_iter=10, random_state=RNG_SEED),
            LogisticRegression(**lr_kw),
        ),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    finite = np.isfinite(y_pred)
    yt, yp = y_true[finite], y_pred[finite]
    if len(yt) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae":  float(mean_absolute_error(yt, yp)),
        "r2":   float(r2_score(yt, yp)),
    }


def _clf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 2,
) -> dict:
    if len(np.unique(y_true)) < 2:
        return dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if n_classes == 2:
        auc   = float(roc_auc_score(y_true, y_proba))
        brier = float(brier_score_loss(y_true, y_proba))
    else:
        try:
            auc = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
        except Exception:
            auc = np.nan
        # Multiclass Brier: mean squared distance from one-hot encoding
        n = len(y_true)
        K = y_proba.shape[1] if y_proba.ndim == 2 else n_classes
        one_hot = np.zeros((n, K))
        one_hot[np.arange(n), y_true.astype(int)] = 1.0
        brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))
    return dict(accuracy=acc, auc=auc, f1=f1, brier=brier)


def _fill_with_means(X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    Xf = X.copy()
    for j in range(Xf.shape[1]):
        mask = np.isnan(Xf[:, j])
        if mask.any():
            Xf[mask, j] = col_means[j]
    return Xf


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def _eval_baseline_reg(name, imputer, model, X_tr, y_tr, X_te, y_te):
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    if name.startswith("CC"):
        cc_mask = ~np.isnan(X_tr).any(axis=1)
        if cc_mask.sum() < X_tr.shape[1] + 2:
            return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
        model.fit(X_tr[cc_mask], y_tr[cc_mask])
        return _reg_metrics(y_te, model.predict(_fill_with_means(X_te, col_means)))
    Xtr_imp = imputer.fit_transform(X_tr)
    Xte_imp = imputer.transform(X_te)
    model.fit(Xtr_imp, y_tr)
    return _reg_metrics(y_te, model.predict(Xte_imp))


def _eval_baseline_clf(name, imputer, model, X_tr, y_tr, X_te, y_te, n_classes):
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_row   = dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)
    if len(np.unique(y_tr)) < 2:
        return nan_row
    if name.startswith("CC"):
        cc_mask = ~np.isnan(X_tr).any(axis=1)
        if cc_mask.sum() < X_tr.shape[1] + 2 or len(np.unique(y_tr[cc_mask])) < 2:
            return nan_row
        model.fit(X_tr[cc_mask], y_tr[cc_mask])
        Xte_f    = _fill_with_means(X_te, col_means)
        yp       = model.predict(Xte_f)
        ypr_all  = model.predict_proba(Xte_f)
        ypr      = ypr_all if n_classes > 2 else ypr_all[:, 1]
        try:
            return _clf_metrics(y_te, yp, ypr, n_classes)
        except ValueError:
            return nan_row
    Xtr_imp = imputer.fit_transform(X_tr)
    Xte_imp = imputer.transform(X_te)
    model.fit(Xtr_imp, y_tr)
    yp      = model.predict(Xte_imp)
    ypr_all = model.predict_proba(Xte_imp)
    ypr     = ypr_all if n_classes > 2 else ypr_all[:, 1]
    return _clf_metrics(y_te, yp, ypr, n_classes)


def _eval_fiml_reg(name, model, X_tr, y_tr, X_te, y_te):
    try:
        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        return _reg_metrics(y_te, m.predict(X_te))
    except Exception as exc:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "_error": str(exc)}


def _eval_fiml_clf(name, model, X_tr, y_tr, X_te, y_te, n_classes):
    nan_row = dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)
    try:
        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        yp      = m.predict(X_te)
        ypr_all = m.predict_proba(X_te)
        ypr     = ypr_all if n_classes > 2 else ypr_all[:, 1]
        return _clf_metrics(y_te, yp, ypr, n_classes)
    except Exception as exc:
        print(f"  [WARN] {name} fold error: {type(exc).__name__}: {exc}")
        return {**nan_row, "_error": str(exc)}


def _run_cv_reg(X, y, models, baselines, n_splits=N_SPLITS, seed=RNG_SEED):
    """Run regression CV; return list of per-fold dicts."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records = []
    for fold_i, (tr, te) in enumerate(kf.split(X)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        out = {}
        # Baselines
        for bname, (imp, mdl) in baselines.items():
            import copy
            out[bname] = _eval_baseline_reg(
                bname, copy.deepcopy(imp) if imp else None,
                copy.deepcopy(mdl), X_tr, y_tr, X_te, y_te
            )
        # MissLearn models
        for mname, mdl in models.items():
            out[mname] = _eval_fiml_reg(mname, mdl, X_tr, y_tr, X_te, y_te)
        gc.collect()
        fold_records.append(out)
    return fold_records


def _run_cv_clf(X, y, models, baselines, n_classes=2,
                n_splits=N_SPLITS, seed=RNG_SEED):
    """Run classification CV; return list of per-fold dicts."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records = []
    for fold_i, (tr, te) in enumerate(splitter.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        out = {}
        for bname, (imp, mdl) in baselines.items():
            import copy
            out[bname] = _eval_baseline_clf(
                bname, copy.deepcopy(imp) if imp else None,
                copy.deepcopy(mdl), X_tr, y_tr, X_te, y_te, n_classes
            )
        for mname, mdl in models.items():
            out[mname] = _eval_fiml_clf(
                mname, mdl, X_tr, y_tr, X_te, y_te, n_classes
            )
        gc.collect()
        fold_records.append(out)
    return fold_records


def aggregate_folds(fold_records: list) -> dict:
    """
    Aggregate per-fold metrics to (mean, std) pairs.

    Returns nested dict: {method: {metric: (mean, std)}}.
    """
    methods = list(fold_records[0].keys())
    metrics = [k for k in fold_records[0][methods[0]] if not k.startswith("_")]
    agg = {}
    for m in methods:
        agg[m] = {}
        for met in metrics:
            vals = np.array([f[m].get(met, np.nan) for f in fold_records], dtype=float)
            n_ok = int(np.sum(~np.isnan(vals)))
            agg[m][met] = (
                float(np.nanmean(vals)),
                float(np.nanstd(vals, ddof=1)) if n_ok > 1 else np.nan,
            )
    return agg


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _banner(title: str, info: str = "") -> None:
    w = 74
    print()
    print("=" * w)
    print(f"  {title}")
    if info:
        print(f"  {info}")
    print("=" * w)


def _print_table_reg(agg: dict, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    hdr = f"  {'Method':<28}  {'RMSE':>9}  {'MAE':>9}  {'R²':>9}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    n_base = sum(1 for k in agg if "Miss" not in k and "MC(" not in k)
    for i, (method, metrics) in enumerate(agg.items()):
        if i == n_base:
            print("  " + "·" * (len(hdr) - 2))
        r, rs = metrics.get("rmse", (np.nan, np.nan))
        a, as_ = metrics.get("mae",  (np.nan, np.nan))
        q, qs = metrics.get("r2",   (np.nan, np.nan))
        def _fmt(v, s):
            if np.isnan(v):
                return "      nan"
            if np.isnan(s):
                return f" {v:8.4f}"
            return f" {v:.4f}±{s:.4f}"
        print(f"  {method:<28}{_fmt(r,rs):>10}{_fmt(a,as_):>10}{_fmt(q,qs):>10}")
    print()


def _print_table_clf(agg: dict, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    hdr = f"  {'Method':<30}  {'Accuracy':>10}  {'AUC':>9}  {'F1':>9}  {'Brier':>9}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    n_base = sum(1 for k in agg if "Miss" not in k and "MC(" not in k)
    for i, (method, metrics) in enumerate(agg.items()):
        if i == n_base:
            print("  " + "·" * (len(hdr) - 2))
        acc, accs = metrics.get("accuracy", (np.nan, np.nan))
        auc, aucs = metrics.get("auc",      (np.nan, np.nan))
        f1,  f1s  = metrics.get("f1",       (np.nan, np.nan))
        br,  brs  = metrics.get("brier",    (np.nan, np.nan))
        def _fmt(v, s):
            if np.isnan(v):
                return "       nan"
            if np.isnan(s):
                return f"  {v:.4f}   "
            return f"  {v:.4f}±{s:.4f}"
        print(f"  {method:<30}{_fmt(acc,accs):>12}{_fmt(auc,aucs):>11}"
              f"{_fmt(f1,f1s):>11}{_fmt(br,brs):>11}")
    print()


def _method_colors(methods: list) -> list:
    """Consistent viridis colour per MissLearn family; grey for baselines."""
    return [_model_color(m) for m in methods]


def _plot_reg(agg: dict, title: str = "", metric: str = "rmse") -> None:
    import matplotlib.pyplot as plt

    methods = list(agg.keys())
    vals    = np.array([agg[m].get(metric, (np.nan, np.nan))[0] for m in methods])
    errs    = np.array([agg[m].get(metric, (np.nan, np.nan))[1] for m in methods])
    errs    = np.where(np.isnan(errs), 0.0, errs)
    colors  = _method_colors(methods)

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.42)))
    y_pos   = np.arange(len(methods))
    ax.barh(y_pos, vals, xerr=errs, color=colors,
            edgecolor="white", linewidth=0.5, capsize=3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel(metric.upper(), fontsize=10)
    ax.set_title(title or f"Benchmark; {metric.upper()}", fontsize=11, pad=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def _plot_clf(agg: dict, title: str = "", metric: str = "auc") -> None:
    import matplotlib.pyplot as plt

    methods = list(agg.keys())
    vals    = np.array([agg[m].get(metric, (np.nan, np.nan))[0] for m in methods])
    errs    = np.array([agg[m].get(metric, (np.nan, np.nan))[1] for m in methods])
    errs    = np.where(np.isnan(errs), 0.0, errs)
    colors  = _method_colors(methods)

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.42)))
    y_pos   = np.arange(len(methods))
    ax.barh(y_pos, vals, xerr=errs, color=colors,
            edgecolor="white", linewidth=0.5, capsize=3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel(metric.upper(), fontsize=10)
    ax.set_title(title or f"Benchmark; {metric.upper()}", fontsize=11, pad=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tier 1: Energy Efficiency
# ---------------------------------------------------------------------------

def bench_energy_efficiency(
    miss_rate: float = 0.20,
    n_splits:  int   = N_SPLITS,
    seed:      int   = RNG_SEED,
    plot:      bool  = False,
) -> dict:
    """
    Compare all MissLearn regression model families on UCI Energy Efficiency
    (n=768, p=8) with synthetic MAR missingness injected at *miss_rate*.

    Baseline: MICE + LinearRegression (single strongest imputation anchor).
    For the full Drop Rows / Drop Cols / Mean / KNN / MICE / FIML comparison
    on synthetic data, see benchmarks/MissLinear_Benchmark.ipynb.

    Returns dict with keys ``agg``, ``fold_records``, ``dataset``.
    """
    ds = load_energy_efficiency()
    X_base, y = ds["X"], ds["y"]
    Xm, actual_rate = inject_mar(X_base, rate=miss_rate, seed=seed)

    _banner(
        "Energy Efficiency  (Tier 1; model family comparison, synthetic MAR)",
        f"n={ds['n']}  p={ds['p']}  miss={actual_rate:.1%}  "
        f"cv={n_splits}  seed={seed}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Task     : regression  |  target: {ds['target_label']}")
    print(f"  Baseline : MICE + LinearRegression  (see benchmarks/ for full comparison)")
    print(f"  GP skipped (n={ds['n']} > {GP_N_THRESHOLD})\n")

    models    = _make_reg_models(ds["n"])
    baselines = _make_reg_baselines_slim()

    print(f"  Running {n_splits}-fold CV: "
          f"{len(baselines)} baseline + {len(models)} MissLearn models ...")
    fold_records = _run_cv_reg(Xm, y, models, baselines, n_splits, seed)
    agg = aggregate_folds(fold_records)
    agg = _sort_agg(agg, task="regression", slim=True)

    _print_table_reg(agg, title="5-fold CV results")
    if plot:
        _plot_reg(agg, title="Energy Efficiency; RMSE (lower is better)")
        _plot_reg(agg, title="Energy Efficiency; R²  (higher is better)",
                  metric="r2")
    return dict(agg=agg, fold_records=fold_records, dataset=ds)


# ---------------------------------------------------------------------------
# Tier 1: Wisconsin
# ---------------------------------------------------------------------------

def bench_wisconsin(
    miss_rate: float = 0.25,
    n_splits:  int   = N_SPLITS,
    seed:      int   = RNG_SEED,
    plot:      bool  = False,
) -> dict:
    """
    Compare all MissLearn binary classification model families on Breast Cancer
    Wisconsin (n=569, p=10) with synthetic MAR missingness at *miss_rate*.

    Baseline: MICE + LogisticRegression (single strongest imputation anchor).
    For the full Drop Rows / Drop Cols / Mean / KNN / MICE / FIML comparison
    on synthetic data, see benchmarks/MissLinear_Benchmark.ipynb.

    Returns dict with keys ``agg``, ``fold_records``, ``dataset``.
    """
    ds = load_wisconsin()
    X_base, y = ds["X"], ds["y"]
    Xm, actual_rate = inject_mar(X_base, rate=miss_rate, seed=seed)

    _banner(
        "Breast Cancer Wisconsin  (Tier 1; model family comparison, synthetic MAR)",
        f"n={ds['n']}  p={ds['p']}  miss={actual_rate:.1%}  "
        f"positive={ds['pos_rate']:.1%}  cv={n_splits}  seed={seed}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Task     : binary classification  |  target: {ds['target_label']}")
    print(f"  Baseline : MICE + LogisticRegression  (see benchmarks/ for full comparison)")
    print(f"  GP skipped (n={ds['n']} > {GP_N_THRESHOLD})\n")

    models    = _make_clf_models(ds["n"], n_classes=2)
    baselines = _make_clf_baselines_slim(n_classes=2)

    print(f"  Running {n_splits}-fold CV: "
          f"{len(baselines)} baseline + {len(models)} MissLearn models ...")
    fold_records = _run_cv_clf(Xm, y, models, baselines, n_classes=2,
                               n_splits=n_splits, seed=seed)
    agg = aggregate_folds(fold_records)
    agg = _sort_agg(agg, task="classification", n_classes=2, slim=True)

    _print_table_clf(agg, title="5-fold CV results")
    if plot:
        _plot_clf(agg, title="Wisconsin; AUC  (higher is better)",
                  metric="auc")
        _plot_clf(agg, title="Wisconsin; Brier  (lower is better)",
                  metric="brier")
    return dict(agg=agg, fold_records=fold_records, dataset=ds)


# ---------------------------------------------------------------------------
# Tier 1: Iris
# ---------------------------------------------------------------------------

def bench_iris(
    miss_rate: float = 0.20,
    n_splits:  int   = N_SPLITS,
    seed:      int   = RNG_SEED,
    plot:      bool  = False,
) -> dict:
    """
    Compare all MissLearn classifiers (wrapped in MissMulticlass for 3-class
    support) on the Iris dataset (n=150, p=4) with synthetic MAR missingness.

    This benchmark is unique in the suite: multi-class via MissMulticlass OvR,
    and MissGaussianClassifier is included (n=150 <= GP_N_THRESHOLD=400).

    Baseline: MICE + LogisticRegression (single strongest imputation anchor).
    For the full Drop Rows / Drop Cols / Mean / KNN / MICE / FIML comparison,
    see benchmarks/MissLinear_Benchmark.ipynb.

    Returns dict with keys ``agg``, ``fold_records``, ``dataset``.
    """
    ds = load_iris()
    X_base, y = ds["X"], ds["y"].astype(np.int64)
    Xm, actual_rate = inject_mar(X_base, rate=miss_rate, seed=seed)

    _banner(
        "Iris  (Tier 1; model family comparison, 3-class, MissMulticlass OvR)",
        f"n={ds['n']}  p={ds['p']}  miss={actual_rate:.1%}  "
        f"cv={n_splits}  seed={seed}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Task     : 3-class classification  |  target: {ds['target_label']}")
    print(f"  Baseline : MICE + LogisticRegression  (see benchmarks/ for full comparison)")
    print(f"  MissLearn models wrapped in MissMulticlass (OvR)")
    print(f"  MissGaussianClassifier included (n={ds['n']} <= {GP_N_THRESHOLD})\n")

    models    = _make_clf_models(ds["n"], n_classes=3)
    baselines = _make_clf_baselines_slim(n_classes=3)

    print(f"  Running {n_splits}-fold CV: "
          f"{len(baselines)} baseline + {len(models)} MissLearn models ...")
    fold_records = _run_cv_clf(Xm, y, models, baselines, n_classes=3,
                               n_splits=n_splits, seed=seed)
    agg = aggregate_folds(fold_records)
    agg = _sort_agg(agg, task="classification", n_classes=3, slim=True)

    _print_table_clf(agg, title="5-fold CV results  (AUC = macro OvR)")
    if plot:
        _plot_clf(agg, title="Iris; AUC macro OvR  (higher is better)",
                  metric="auc")
        _plot_clf(agg, title="Iris; Accuracy  (higher is better)",
                  metric="accuracy")
    return dict(agg=agg, fold_records=fold_records, dataset=ds)


# ---------------------------------------------------------------------------
# Tier 2: Auto MPG
# ---------------------------------------------------------------------------

def bench_auto_mpg(
    miss_rate: float = 0.20,
    n_splits:  int   = N_SPLITS,
    seed:      int   = RNG_SEED,
    plot:      bool  = False,
) -> dict:
    """
    Benchmark all MissLearn regression models on Auto MPG (n=392, p=7).

    The raw dataset has only ~0.2% real missing data (6 rows with missing
    horsepower), which is insufficient for meaningful comparison.  Synthetic
    MAR missingness is injected on top of the real missingness at *miss_rate*.

    MissGaussianRegressor is included (n=392 <= 400 threshold).

    Returns dict with keys ``agg``, ``fold_records``, ``dataset``.
    """
    ds = load_auto_mpg()
    X_base, y = ds["X"], ds["y"]
    Xm, actual_rate = inject_mar(X_base, rate=miss_rate, seed=seed)

    _banner(
        "Auto MPG Benchmark  (Tier 2; real missing data + synthetic MAR)",
        f"n={ds['n']}  p={ds['p']}  real_miss={ds['missing_rate']:.1%}  "
        f"injected_mar={miss_rate:.0%}  actual_miss={actual_rate:.1%}  "
        f"cv={n_splits}  seed={seed}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Task     : regression  |  target: {ds['target_label']}")
    print(f"  Missingness: real (horsepower) + synthetic MAR at {miss_rate:.0%}\n")

    models    = _make_reg_models(ds["n"])
    baselines = _make_reg_baselines()

    print(f"  Running {n_splits}-fold CV for "
          f"{len(baselines)} baselines + {len(models)} MissLearn models ...")
    fold_records = _run_cv_reg(Xm, y, models, baselines, n_splits, seed)
    agg = aggregate_folds(fold_records)
    agg = _sort_agg(agg, task="regression", slim=False)

    _print_table_reg(agg, title="5-fold CV results")
    if plot:
        _plot_reg(agg, title="Auto MPG; RMSE  (lower is better)")
        _plot_reg(agg, title="Auto MPG; R²  (higher is better)", metric="r2")
    return dict(agg=agg, fold_records=fold_records, dataset=ds)


# ---------------------------------------------------------------------------
# Tier 3: ESOL mechanism sweep
# ---------------------------------------------------------------------------

def bench_esol_sweep(
    mechanisms: list | None = None,
    miss_rates: list | None = None,
    n_splits:   int         = N_SPLITS,
    seed:       int         = RNG_SEED,
    plot:       bool        = False,
) -> dict:
    """
    Sweep MCAR / MAR / MNAR missingness mechanisms over a range of rates on
    the ESOL (Delaney aqueous solubility) dataset (n=1128, p=6).

    The flagship MissLearn regression models are compared against the best
    imputation baseline (MICE + LinearRegression) and the naive complete-case
    baseline at each (mechanism, rate) combination.

    Parameters
    ----------
    mechanisms : list of str, default ["MCAR", "MAR", "MNAR"]
    miss_rates : list of float, default [0.10, 0.20, 0.30, 0.40, 0.50]
    n_splits   : int, cross-validation folds
    seed       : int, random seed
    plot       : bool, show mechanism comparison line plots

    Returns
    -------
    dict with keys ``sweep_results``, ``dataset``.
    sweep_results[mechanism][miss_rate] = agg dict.
    """
    if mechanisms is None:
        mechanisms = ["MCAR", "MAR", "MNAR"]
    if miss_rates is None:
        miss_rates = [0.10, 0.20, 0.30, 0.40, 0.50]

    ds = load_esol()
    X_base, y = ds["X"], ds["y"]

    _banner(
        "ESOL Aqueous Solubility; Mechanism Sweep  (Tier 3)",
        f"n={ds['n']}  p={ds['p']}  mechanisms={mechanisms}  "
        f"rates={miss_rates}  cv={n_splits}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Task     : regression  |  target: {ds['target_label']}")
    print(f"  GP skipped (n={ds['n']} > {GP_N_THRESHOLD})")
    print(f"  Flagship models: MissLinear, MissRidgeRegressor")
    print(f"  Baselines: CC + LinearReg, MICE + LinearReg\n")

    # Use only the two flagship models and two anchor baselines for the sweep
    # (running all models × all rates × all mechanisms would be very slow)
    sweep_models = {
        "MissLinear":         MissLinear(compute_se=False, max_iter=2000, tol=1e-8),
        "MissRidgeRegressor": MissRidgeRegressor(alpha=1.0),
    }
    sweep_baselines = {
        "CC + LinearReg":   (None,
                             LinearRegression()),
        "MICE + LinearReg": (IterativeImputer(max_iter=10, random_state=RNG_SEED),
                             LinearRegression()),
    }

    sweep_results: dict = {mech: {} for mech in mechanisms}
    total = len(mechanisms) * len(miss_rates)
    done  = 0

    # Deterministic per-mechanism offset (Python's built-in hash() is salted
    # per-process, so it cannot be used for reproducible seeding across runs).
    _MECH_OFFSET = {"MCAR": 0, "MAR": 1, "MNAR": 2}

    for mech in mechanisms:
        injector = _INJECTORS[mech]
        for rate in miss_rates:
            done += 1
            inj_seed  = seed + int(rate * 1000) + _MECH_OFFSET.get(mech, 0)
            Xm, actual = injector(X_base, rate=rate, seed=inj_seed)
            print(f"  [{done:2d}/{total}] {mech:<5}  rate={rate:.0%}  "
                  f"actual={actual:.1%}  ...", end=" ", flush=True)
            folds = _run_cv_reg(
                Xm, y, sweep_models, sweep_baselines, n_splits, seed
            )
            agg = aggregate_folds(folds)
            agg = _sort_agg(agg, task="regression", slim=False)
            sweep_results[mech][rate] = agg
            # Quick inline summary
            fiml_rmse = agg["MissLinear"]["rmse"][0]
            mice_rmse = agg["MICE + LinearReg"]["rmse"][0]
            print(f"RMSE MissLinear={fiml_rmse:.4f}  MICE={mice_rmse:.4f}")
        gc.collect()

    # Print per-mechanism tables
    for mech in mechanisms:
        print(f"\n{'─'*70}")
        print(f"  {mech} sweep results (RMSE mean ± std across {n_splits} folds)")
        print(f"{'─'*70}")
        methods_list = list(sweep_results[mech][miss_rates[0]].keys())
        rate_hdr = "   ".join(f"{r:.0%}" for r in miss_rates)
        print(f"  {'Method':<24}  {rate_hdr}")
        print(f"  {'-'*22}  " + "-"*40)
        for method in methods_list:
            row = f"  {method:<24}"
            for rate in miss_rates:
                mu, sd = sweep_results[mech][rate][method]["rmse"]
                if np.isnan(mu):
                    row += "    nan  "
                else:
                    row += f"  {mu:.3f}"
            print(row)
    print()

    if plot:
        import matplotlib.pyplot as plt
        fig = _plot_esol_sweep(sweep_results, miss_rates, mechanisms)
        plt.show()

    return dict(sweep_results=sweep_results, dataset=ds)


def _plot_esol_sweep(
    sweep_results: dict,
    miss_rates:    list,
    mechanisms:    list,
) -> "matplotlib.figure.Figure":
    """
    Single combined figure: 3 rows (mechanisms) x 2 columns (RMSE | R²).
    Returns the Figure (does not call plt.show()).
    """
    import matplotlib.pyplot as plt

    methods = list(next(iter(sweep_results.values()))[miss_rates[0]].keys())

    # Build method colour map
    grey_colors = ["#888888", "#bbbbbb"]
    fiml_idx, base_idx = 0, 0
    method_colors = {}
    for m in methods:
        if m.startswith("Miss") or m.startswith("MC("):
            method_colors[m] = _model_color(m)
            fiml_idx += 1
        else:
            method_colors[m] = grey_colors[base_idx % len(grey_colors)]
            base_idx += 1

    nrows = len(mechanisms)
    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(11, 4.0 * nrows),
        constrained_layout=True,
    )
    # Ensure axes is always 2-D
    if nrows == 1:
        axes = [axes]

    x_vals = [r * 100 for r in miss_rates]

    for row_i, mech in enumerate(mechanisms):
        for col_i, (metric, ylabel) in enumerate([("rmse", "RMSE"), ("r2", "R²")]):
            ax = axes[row_i][col_i]
            for method in methods:
                ys = [sweep_results[mech][r][method][metric][0] for r in miss_rates]
                is_fiml = method.startswith("Miss") or method.startswith("MC(")
                ls = "-" if is_fiml else "--"
                lw = 2.0 if is_fiml else 1.4
                ax.plot(
                    x_vals, ys,
                    marker="o", markersize=5,
                    color=method_colors[method],
                    linestyle=ls, linewidth=lw,
                    label=method,
                )
            # y-axis label on every subplot in each column
            ax.set_ylabel(ylabel, fontsize=11)
            # x-axis label on bottom row only
            if row_i == nrows - 1:
                ax.set_xlabel("Missing rate (%)", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)

        # Row label (mechanism name) as ylabel on left column
        axes[row_i][0].set_ylabel(f"{mech}\n\nRMSE", fontsize=11)

    # Column titles on top row
    axes[0][0].set_title("RMSE (lower is better)", fontsize=11)
    axes[0][1].set_title("R² (higher is better)", fontsize=11)

    # Shared legend below figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(methods),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "ESOL Aqueous Solubility; RMSE and R² across missingness mechanisms",
        fontsize=11,
    )
    return fig


# ---------------------------------------------------------------------------
# Tier 4: MissMixed: Radon grouped benchmark
# ---------------------------------------------------------------------------

def _eval_mixed_reg(name, model, X_tr, y_tr, groups_tr, X_te, y_te):
    """Fit a MissMixed regression model with the groups array, then predict."""
    try:
        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr, groups=groups_tr)
        return _reg_metrics(y_te, m.predict(X_te))
    except Exception as exc:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "_error": str(exc)}


def _eval_mixed_clf(name, model, X_tr, y_tr, groups_tr, X_te, y_te):
    """Fit a MissMixed classification model with the groups array, then predict."""
    nan_row = dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)
    try:
        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr, groups=groups_tr)
        yp      = m.predict(X_te)
        ypr_all = m.predict_proba(X_te)
        ypr     = ypr_all[:, 1]
        return _clf_metrics(y_te, yp, ypr, n_classes=2)
    except Exception as exc:
        return {**nan_row, "_error": str(exc)}


def _run_cv_mixed_reg(X, y, groups, mixed_models, regular_models, baselines,
                      n_splits=N_SPLITS, seed=RNG_SEED):
    """
    Regression CV that handles both:
    - ``mixed_models``: fitted with groups (MissMixedRegressor)
    - ``regular_models``: fitted without groups (MissLinear, etc.)
    - ``baselines``: imputer + sklearn model
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records = []
    for tr, te in kf.split(X):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te, y_te       = X[te], y[te]
        out = {}
        import copy
        for bname, (imp, mdl) in baselines.items():
            out[bname] = _eval_baseline_reg(
                bname, copy.deepcopy(imp) if imp else None,
                copy.deepcopy(mdl), X_tr, y_tr, X_te, y_te
            )
        for mname, mdl in regular_models.items():
            out[mname] = _eval_fiml_reg(mname, mdl, X_tr, y_tr, X_te, y_te)
        for mname, mdl in mixed_models.items():
            out[mname] = _eval_mixed_reg(mname, mdl, X_tr, y_tr, g_tr, X_te, y_te)
        gc.collect()
        fold_records.append(out)
    return fold_records


def _run_cv_mixed_clf(X, y, groups, mixed_models, regular_models, baselines,
                      n_splits=N_SPLITS, seed=RNG_SEED):
    """
    Classification CV that handles MissMixed (needs groups) and regular models.
    Uses StratifiedKFold on y.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records = []
    for tr, te in skf.split(X, y):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te, y_te       = X[te], y[te]
        out = {}
        import copy
        for bname, (imp, mdl) in baselines.items():
            out[bname] = _eval_baseline_clf(
                bname, copy.deepcopy(imp) if imp else None,
                copy.deepcopy(mdl), X_tr, y_tr, X_te, y_te, n_classes=2
            )
        for mname, mdl in regular_models.items():
            out[mname] = _eval_fiml_clf(mname, mdl, X_tr, y_tr, X_te, y_te,
                                        n_classes=2)
        for mname, mdl in mixed_models.items():
            out[mname] = _eval_mixed_clf(mname, mdl, X_tr, y_tr, g_tr, X_te, y_te)
        gc.collect()
        fold_records.append(out)
    return fold_records


def bench_mixed(
    miss_rate: float = 0.20,
    n_splits:  int   = N_SPLITS,
    seed:      int   = RNG_SEED,
    plot:      bool  = False,
) -> dict:
    """
    Benchmark MissMixedRegressor and MissMixedClassifier on the Radon dataset
    (Gelman & Hill 2007), the canonical multilevel modeling benchmark.

    Dataset: 919 Minnesota radon measurements in 85 counties.
    Features: floor (basement=0/first floor=1), log_uranium (county-level).
    Groups:   county (85 groups, median 11 obs/group).
    Missingness: synthetic MAR at *miss_rate* injected before CV.

    The comparison is structured to isolate the value of random effects:

      Baselines (no group structure):
        CC + LinearReg, Mean + LinearReg, KNN + LinearReg, MICE + LinearReg
        CC + LogReg,    Mean + LogReg,    KNN + LogReg,    MICE + LogReg

      FIML without random effects:
        MissLinear      (regression)
        MissLogistic    (classification: log_radon > median = 1)

      FIML with random intercepts per county:
        MissMixedRegressor
        MissMixedClassifier

    The benchmark runs twice: once as regression (target = log_radon) and once
    as binary classification (target = above-median log_radon).

    Returns dict with keys ``agg_reg``, ``agg_clf``, ``fold_records_reg``,
    ``fold_records_clf``, ``dataset``.
    """
    ds = load_radon()
    X_base  = ds["X"]
    y_reg   = ds["y"]
    y_clf   = ds["y_bin"]
    groups  = ds["groups"]
    n, p    = X_base.shape

    Xm, actual_rate = inject_mar(X_base, rate=miss_rate, seed=seed)

    _banner(
        "Radon Benchmark; MissMixed  (Tier 4; grouped/longitudinal)",
        f"n={n}  p={p}  groups={ds['n_groups']} counties  "
        f"miss={actual_rate:.1%}  cv={n_splits}  seed={seed}",
    )
    print(f"  Dataset  : {ds['name']}  |  {ds['reference']}")
    print(f"  Features : floor (0=basement, 1=first floor), log_uranium")
    print(f"  Groups   : county (85 Minnesota counties)")
    print(f"  Regression target   : log radon level")
    print(f"  Classification target: above-median log radon (median="
          f"{ds['y_median']:.3f})\n")

    # Models
    mixed_reg  = {"MissMixedRegressor":   MissMixedRegressor()}
    mixed_clf  = {"MissMixedClassifier":  MissMixedClassifier()}
    regular_reg = {
        "MissLinear":         MissLinear(compute_se=False, max_iter=2000, tol=1e-8),
        "MissRidgeRegressor": MissRidgeRegressor(alpha=1.0),
    }
    regular_clf = {
        "MissLogistic":         MissLogistic(compute_se=False),
        "MissRidgeClassifier":  MissRidgeClassifier(alpha=1.0),
    }
    baselines_reg = _make_reg_baselines()
    baselines_clf = _make_clf_baselines(n_classes=2)

    # ---- Regression ----
    print(f"  [Regression]  Running {n_splits}-fold CV ...")
    fold_reg = _run_cv_mixed_reg(
        Xm, y_reg, groups, mixed_reg, regular_reg, baselines_reg,
        n_splits=n_splits, seed=seed,
    )
    agg_reg = aggregate_folds(fold_reg)
    agg_reg = _sort_agg(agg_reg, task="regression", slim=False)
    _print_table_reg(agg_reg, title="Regression results (log radon)")
    if plot:
        _plot_reg(agg_reg,
                  title="Radon; RMSE  (lower is better)\n"
                         "MissMixed adds random county intercepts")
        _plot_reg(agg_reg,
                  title="Radon; R²  (higher is better)",
                  metric="r2")

    # ---- Classification ----
    print(f"  [Classification]  Running {n_splits}-fold CV ...")
    fold_clf = _run_cv_mixed_clf(
        Xm, y_clf, groups, mixed_clf, regular_clf, baselines_clf,
        n_splits=n_splits, seed=seed,
    )
    agg_clf = aggregate_folds(fold_clf)
    agg_clf = _sort_agg(agg_clf, task="classification", slim=False)
    _print_table_clf(agg_clf, title="Classification results (high radon vs low)")
    if plot:
        _plot_clf(agg_clf,
                  title="Radon; AUC  (higher is better)\n"
                         "MissMixed adds random county intercepts",
                  metric="auc")
        _plot_clf(agg_clf,
                  title="Radon; Accuracy  (higher is better)",
                  metric="accuracy")

    return dict(
        agg_reg=agg_reg, agg_clf=agg_clf,
        fold_records_reg=fold_reg, fold_records_clf=fold_clf,
        dataset=ds,
    )
