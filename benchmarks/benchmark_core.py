# ============================================================
# benchmark_core.py
# MissLearn Benchmark Core: reusable across all benchmark notebooks.
#
# Provides:
#   - MAR injection
#   - Synthetic dataset generation (regression + classification)
#   - 6-method CV engine (Drop Rows, Drop Cols, Mean, KNN, MICE, FIML)
#   - Aggregation, plotting, statistical tests, and save utilities
#
# The FIML model is never imported here; notebooks pass it as a factory
# callable so this module stays independent of any MissLearn algorithm.
# ============================================================

import gc
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import ttest_rel

from sklearn.datasets import make_regression, make_classification
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
)

sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.05)


# ============================================================
# Baseline fairness
# ============================================================
# Several MissLearn estimators standardise internally: the expected-kernel and
# expected-distance families operate on standardised features, the penalized
# models follow the glmnet convention, and MissSupportRegressor also
# standardises the response.  A bare scale-sensitive scikit-learn baseline
# would therefore lose on preprocessing rather than on its missing-data
# strategy, and MissLearn would appear to win for the wrong reason.  Every
# conventional arm is consequently given the same standardisation, applied
# after its imputation/deletion step, which is what a competent practitioner
# would do.  Standardisation is exactly neutral for the unpenalized linear
# models and for GaussianNB, and matched to the internal convention for the
# rest, so a single uniform rule keeps the comparison about missingness.
FAIR_BASELINE_SCALING = True


def _fair_sklearn_factory(sklearn_factory, task):
    """Wrap a baseline factory so it standardises features (and, for
    regression, the response) exactly as the MissLearn counterparts do."""
    if not FAIR_BASELINE_SCALING:
        return sklearn_factory
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if task == "regression":
        from sklearn.compose import TransformedTargetRegressor

        def make():
            return TransformedTargetRegressor(
                regressor=Pipeline([("scale", StandardScaler()),
                                    ("est", sklearn_factory())]),
                transformer=StandardScaler(),
            )
    else:
        def make():
            return Pipeline([("scale", StandardScaler()),
                             ("est", sklearn_factory())])
    return make


# ------------------------------------------------------------------
# Matched regularisation for the penalized families
# ------------------------------------------------------------------
# A fixed penalty is NOT comparable between the two arms: scikit-learn divides
# the squared-error term by n while the MissLearn objective does not, so the
# same nominal alpha is a different effective penalty.  Comparing, say,
# Ridge(alpha=1.0) with MissRidgeRegressor(alpha=0.1) confounds the penalty
# strength with the missing-data strategy.  The only clean protocol is to give
# both arms the same tuning budget and let each choose its own strength by
# inner cross-validation on the training fold.
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0]


def tuned_sklearn(make_est, task, param="alpha", grid=None):
    """Baseline factory whose regularisation strength is chosen by inner CV."""
    from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
    grid = ALPHA_GRID if grid is None else grid

    def make():
        cv = (StratifiedKFold(3, shuffle=True, random_state=0)
              if task == "classification"
              else KFold(3, shuffle=True, random_state=0))
        return GridSearchCV(make_est(), {param: grid}, cv=cv,
                            scoring=("accuracy" if task == "classification"
                                     else "r2"))
    return make


class TunedMiss:
    """Choose alpha for a MissLearn estimator by inner CV on NaN-bearing data.

    `make_est(alpha)` must return a fresh MissLearn estimator.  Mirrors the
    tuning budget given to the conventional arm by `tuned_sklearn`.
    """

    def __init__(self, make_est, task="regression", grid=None, n_splits=3):
        self.make_est = make_est
        self.task = task
        self.grid = ALPHA_GRID if grid is None else grid
        self.n_splits = n_splits

    def fit(self, X, y):
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.metrics import r2_score, accuracy_score
        splitter = (StratifiedKFold(self.n_splits, shuffle=True, random_state=0)
                    if self.task == "classification"
                    else KFold(self.n_splits, shuffle=True, random_state=0))
        strat = y if self.task == "classification" else None
        best_a, best_s = self.grid[0], -np.inf
        for a in self.grid:
            scores = []
            for tr, va in splitter.split(X, strat):
                try:
                    m = self.make_est(a)
                    m.fit(X[tr], y[tr])
                    p = m.predict(X[va])
                    ok = ~np.isnan(y[va])
                    scores.append(r2_score(y[va][ok], p[ok])
                                  if self.task == "regression"
                                  else accuracy_score(y[va][ok], p[ok]))
                except Exception:
                    scores.append(-np.inf)
            s = float(np.mean(scores))
            if s > best_s:
                best_s, best_a = s, a
        self.alpha_ = best_a
        self.model_ = self.make_est(best_a)
        self.model_.fit(X, y)
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ============================================================
# Palette & method constants
# ============================================================

METHODS = [
    "Drop Rows",
    "Drop Cols",
    "Mean Imputation",
    "KNN Imputation",
    "MICE (Iterative)",
    "MissLearn (FIML)",
]

SHORT_LABELS = [
    "Drop\nRows", "Drop\nCols",
    "Mean\nImp.", "KNN\nImp.", "MICE", "FIML",
]

_CMAP = plt.cm.viridis
# Sample viridis over [0.12, 0.95] rather than the full [0, 1] range: the very
# bottom of viridis is a near-black purple that is hard to read on white, and
# this keeps FIML (last) at the bright, clearly distinct end.
COLORS = {
    m: _CMAP(x)
    for m, x in zip(METHODS, np.linspace(0.12, 0.95, len(METHODS)))
}

REG_METRIC_META = [
    # (key, display_label, higher_is_better, direction_string)
    ("rmse", "RMSE", False, "lower is better"),
    ("mae",  "MAE",  False, "lower is better"),
    ("r2",   "R²",   True,  "higher is better"),
]

CLF_METRIC_META = [
    ("accuracy", "Accuracy",      True,  "higher is better"),
    ("auc",      "ROC-AUC",       True,  "higher is better"),
    ("f1",       "F1 (weighted)", True,  "higher is better"),
    ("brier",    "Brier Score",   False, "lower is better"),
]


# ============================================================
# MAR injection
# ============================================================

def inject_mar(X, rate, seed=42, col_frac=0.60):
    """Inject Missing At Random into a complete feature matrix.

    For each affected column j, missingness probability depends on whether
    the value in column (j+1) % p is above or below the column median.
    Rows above the median are masked at 2× the rate of rows below, satisfying
    MAR: missingness depends only on observed data, never on the missing value.

    Only a random subset of columns (``col_frac`` fraction) are affected by
    injection.  The remaining columns are left completely clean.  This serves
    two purposes:

    1. **Drop Cols baseline**: always has at least some clean columns to train
       on, making it a meaningful comparison rather than returning NaN.
    2. **Drop Cols**: with 40% of features always clean, Drop Cols always
       has columns to train on and provides a meaningful comparison.

    The per-column injection rate is scaled up so that the overall missing
    fraction across all columns remains equal to ``rate``.

    Parameters
    ----------
    X        : ndarray (n, p), complete (no NaN)
    rate     : float, target overall missing fraction across all columns
    seed     : int
    col_frac : float in (0, 1], fraction of columns to inject NaN into.
               Default 0.60; 60 % of columns are affected, 40 % stay clean.

    Returns
    -------
    Xm          : ndarray (n, p) with NaN inserted
    actual_rate : float, realised overall missing fraction
    """
    rng  = np.random.default_rng(seed)
    Xm   = X.astype(np.float64).copy()
    n, p = Xm.shape

    # Select which columns to affect.
    # For p > 1: leave at least one clean column (cap at p-1).
    # For p == 1: we must affect the only column; no clean column is possible.
    if p > 1:
        n_affected = min(p - 1, max(1, round(p * col_frac)))
    else:
        n_affected = 1
    affected    = rng.choice(p, size=n_affected, replace=False)

    # Scale per-column rate so the overall rate across all p columns is preserved
    per_col_rate = min(rate * p / n_affected, 0.85)

    for j in affected:
        cond  = Xm[:, (j + 1) % p]
        med   = np.nanmedian(cond)
        base  = per_col_rate * (2.0 / 3.0)
        probs = np.where(cond > med, 2.0 * base, base)
        probs = np.clip(probs, 0.0, 1.0)
        Xm[rng.uniform(0.0, 1.0, n) < probs, j] = np.nan

    return Xm, float(np.isnan(Xm).mean())


# ============================================================
# Synthetic dataset generation
# ============================================================

def make_regression_datasets(mar_rate=0.25, seed=42):
    """Three synthetic regression datasets of increasing size and complexity.

    Each dataset dict contains:
        X          : (n, p) ndarray with injected MAR missingness
        X_complete : (n, p) ndarray, no NaN; retained for reference; not used in CV
        y          : (n,) ndarray of targets
        name, feature_names, n, p, missing_rate, description
    """
    configs = [
        dict(
            n=300, p=5, noise=0.3, n_informative=4,
            name="Regression-Small",
            description="n=300, p=5, low noise, 4 informative features, ~60% cols affected by MAR",
        ),
        dict(
            n=1500, p=10, noise=1.0, n_informative=7,
            name="Regression-Medium",
            description="n=1,500, p=10, moderate noise, correlated features, ~60% cols affected by MAR",
        ),
        dict(
            n=6000, p=20, noise=2.0, n_informative=12,
            name="Regression-Large",
            description="n=6,000, p=20, higher noise, redundant features, ~60% cols affected by MAR",
        ),
    ]
    datasets = []
    for i, cfg in enumerate(configs):
        Xc, y = make_regression(
            n_samples=cfg["n"],
            n_features=cfg["p"],
            n_informative=cfg["n_informative"],
            noise=cfg["noise"],
            random_state=seed + i,
        )
        Xc = Xc.astype(np.float64)
        y  = y.astype(np.float64)
        Xm, actual_rate = inject_mar(Xc, rate=mar_rate, seed=seed + i)
        datasets.append(dict(
            X=Xm,
            X_complete=Xc,
            y=y,
            name=cfg["name"],
            feature_names=[f"x{j+1}" for j in range(cfg["p"])],
            n=cfg["n"],
            p=cfg["p"],
            missing_rate=actual_rate,
            description=cfg["description"],
        ))
    return datasets


def make_classification_datasets(mar_rate=0.25, seed=42):
    """Three synthetic binary classification datasets of increasing size and complexity.

    n_clusters_per_class=1 gives approximately Gaussian class-conditional
    distributions, which is consistent with the MVN predictor assumption
    used by MissLogistic FIML.

    Each dataset dict contains:
        X, X_complete, y, name, feature_names, n, p,
        missing_rate, pos_rate, description
    """
    configs = [
        dict(
            n=400, p=5, n_informative=4, n_redundant=0,
            weights=[0.5, 0.5], flip_y=0.02,
            name="Classification-Small",
            description="n=400, p=5, balanced classes, clear separation, ~60% cols affected by MAR",
        ),
        dict(
            n=1500, p=10, n_informative=7, n_redundant=2,
            weights=[0.7, 0.3], flip_y=0.03,
            name="Classification-Medium",
            description="n=1,500, p=10, imbalanced (30% positive), correlated features, ~60% cols affected by MAR",
        ),
        dict(
            n=5000, p=15, n_informative=9, n_redundant=4,
            weights=[0.6, 0.4], flip_y=0.05,
            name="Classification-Large",
            description="n=5,000, p=15, moderate imbalance, redundant features, ~60% cols affected by MAR",
        ),
    ]
    datasets = []
    for i, cfg in enumerate(configs):
        Xc, y = make_classification(
            n_samples=cfg["n"],
            n_features=cfg["p"],
            n_informative=cfg["n_informative"],
            n_redundant=cfg["n_redundant"],
            n_clusters_per_class=1,
            weights=cfg["weights"],
            flip_y=cfg["flip_y"],
            random_state=seed + i,
        )
        Xc = Xc.astype(np.float64)
        y  = y.astype(np.float64)
        Xm, actual_rate = inject_mar(Xc, rate=mar_rate, seed=seed + i)
        datasets.append(dict(
            X=Xm,
            X_complete=Xc,
            y=y,
            name=cfg["name"],
            feature_names=[f"x{j+1}" for j in range(cfg["p"])],
            n=cfg["n"],
            p=cfg["p"],
            missing_rate=actual_rate,
            pos_rate=float(y.mean()),
            description=cfg["description"],
        ))
    return datasets


# ============================================================
# Metrics
# ============================================================

def _reg_metrics(y_true, y_pred):
    finite = np.isfinite(y_pred)
    yt, yp = y_true[finite], y_pred[finite]
    if len(yt) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae":  float(mean_absolute_error(yt, yp)),
        "r2":   float(r2_score(yt, yp)),
    }


def _clf_metrics(y_true, y_pred, y_proba):
    if len(np.unique(y_true)) < 2:
        return dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=float(roc_auc_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        brier=float(brier_score_loss(y_true, y_proba)),
    )


# ============================================================
# CV fold internals
# ============================================================

def _fill_means(X, col_means):
    Xf = X.copy()
    for j in range(Xf.shape[1]):
        nans = np.isnan(Xf[:, j])
        if nans.any():
            Xf[nans, j] = col_means[j]
    return Xf


def _drop_cols_then_impute(X_tr, X_te):
    """Drop columns that have any missing values in the training fold.

    Residual NaN in test (possible when a column is clean in train but not
    in test) are handled by a mean imputer fitted on the kept training columns.

    Returns (X_tr_out, X_te_out), or (None, None) if all columns are dropped.
    """
    missing_cols = set(int(j) for j in np.where(np.isnan(X_tr).any(axis=0))[0])
    keep = [j for j in range(X_tr.shape[1]) if j not in missing_cols]
    if not keep:
        return None, None
    X_tr_dc = X_tr[:, keep]
    X_te_dc = X_te[:, keep]
    imp = SimpleImputer(strategy="mean")
    X_tr_dc = imp.fit_transform(X_tr_dc)
    X_te_dc = imp.transform(X_te_dc)
    return X_tr_dc, X_te_dc


# ============================================================
# Fold evaluation: regression
# ============================================================

def evaluate_fold_regression(X_tr, y_tr, X_te, y_te,
                              sklearn_factory, misslearn_factory,
                              rng_seed=42):
    """Evaluate all 6 missing-data methods on one regression CV fold.

    Every method receives the same X_tr and X_te (with NaN at the injected
    positions) and is evaluated against the same y_te.  No method has access
    to the true un-masked feature values; this is a fair comparison of
    missing-data strategies, not a comparison against a perfect-data ceiling.

    Parameters
    ----------
    X_tr, y_tr       : training features (with NaN) and targets
    X_te, y_te       : test features (with NaN) and targets
    sklearn_factory  : callable() → new sklearn regressor
    misslearn_factory: callable() → new MissLearn regressor
    rng_seed         : int, passed to IterativeImputer for reproducibility
    """
    nan_r     = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    out       = {}
    # Baselines are standardised to match the MissLearn internals; see
    # _fair_sklearn_factory.
    sklearn_factory = _fair_sklearn_factory(sklearn_factory, "regression")

    # 1. Drop Rows; listwise deletion on train; mean-fill NaN in test
    X_te_f  = _fill_means(X_te, col_means)
    cc_mask = ~np.isnan(X_tr).any(axis=1)
    if cc_mask.sum() >= X_tr.shape[1] + 2:
        m = sklearn_factory()
        m.fit(X_tr[cc_mask], y_tr[cc_mask])
        out["Drop Rows"] = _reg_metrics(y_te, m.predict(X_te_f))
    else:
        out["Drop Rows"] = nan_r.copy()

    # 2. Drop Cols; drop any column with missing from train and test
    X_tr_dc, X_te_dc = _drop_cols_then_impute(X_tr, X_te)
    if X_tr_dc is not None:
        m = sklearn_factory()
        m.fit(X_tr_dc, y_tr)
        out["Drop Cols"] = _reg_metrics(y_te, m.predict(X_te_dc))
    else:
        out["Drop Cols"] = nan_r.copy()

    # 3. Mean Imputation
    imp = SimpleImputer(strategy="mean")
    m   = sklearn_factory()
    m.fit(imp.fit_transform(X_tr), y_tr)
    out["Mean Imputation"] = _reg_metrics(y_te, m.predict(imp.transform(X_te)))

    # 4. KNN Imputation
    imp = KNNImputer(n_neighbors=5)
    m   = sklearn_factory()
    m.fit(imp.fit_transform(X_tr), y_tr)
    out["KNN Imputation"] = _reg_metrics(y_te, m.predict(imp.transform(X_te)))

    # 5. MICE (Iterative Imputation)
    imp = IterativeImputer(max_iter=10, random_state=rng_seed)
    m   = sklearn_factory()
    m.fit(imp.fit_transform(X_tr), y_tr)
    out["MICE (Iterative)"] = _reg_metrics(y_te, m.predict(imp.transform(X_te)))

    # 6. MissLearn FIML; no imputation; native missing-data handling
    try:
        ml = misslearn_factory()
        ml.fit(X_tr, y_tr)
        out["MissLearn (FIML)"] = _reg_metrics(y_te, ml.predict(X_te))
    except Exception as exc:
        out["MissLearn (FIML)"] = {**nan_r, "_error": str(exc)}

    gc.collect()
    return out


# ============================================================
# Fold evaluation: classification
# ============================================================

def evaluate_fold_classification(X_tr, y_tr, X_te, y_te,
                                   sklearn_factory, misslearn_factory,
                                   rng_seed=42):
    """Evaluate all 6 missing-data methods on one classification CV fold.

    Every method receives the same X_tr and X_te (with NaN) and is evaluated
    against the same y_te.  No method has access to the true un-masked values.
    """
    nan_c = dict(accuracy=np.nan, auc=np.nan, f1=np.nan, brier=np.nan)

    if len(np.unique(y_tr)) < 2:
        return {m: nan_c.copy() for m in METHODS}

    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    out       = {}
    # Baselines are standardised to match the MissLearn internals; see
    # _fair_sklearn_factory.
    sklearn_factory = _fair_sklearn_factory(sklearn_factory, "classification")

    def _fit_predict(X_tr_, y_tr_, X_te_):
        m   = sklearn_factory()
        m.fit(X_tr_, y_tr_)
        yp  = m.predict(X_te_)
        ypr = m.predict_proba(X_te_)[:, 1]
        return _clf_metrics(y_te, yp, ypr)

    # 1. Drop Rows
    X_te_f  = _fill_means(X_te, col_means)
    cc_mask = ~np.isnan(X_tr).any(axis=1)
    if cc_mask.sum() >= X_tr.shape[1] + 2 and len(np.unique(y_tr[cc_mask])) == 2:
        out["Drop Rows"] = _fit_predict(X_tr[cc_mask], y_tr[cc_mask], X_te_f)
    else:
        out["Drop Rows"] = nan_c.copy()

    # 2. Drop Cols
    X_tr_dc, X_te_dc = _drop_cols_then_impute(X_tr, X_te)
    if X_tr_dc is not None:
        try:
            out["Drop Cols"] = _fit_predict(X_tr_dc, y_tr, X_te_dc)
        except Exception as exc:
            out["Drop Cols"] = {**nan_c, "_error": str(exc)}
    else:
        out["Drop Cols"] = nan_c.copy()

    # 3. Mean Imputation
    imp = SimpleImputer(strategy="mean")
    out["Mean Imputation"] = _fit_predict(
        imp.fit_transform(X_tr), y_tr, imp.transform(X_te))

    # 4. KNN Imputation
    imp = KNNImputer(n_neighbors=5)
    out["KNN Imputation"] = _fit_predict(
        imp.fit_transform(X_tr), y_tr, imp.transform(X_te))

    # 5. MICE
    imp = IterativeImputer(max_iter=10, random_state=rng_seed)
    out["MICE (Iterative)"] = _fit_predict(
        imp.fit_transform(X_tr), y_tr, imp.transform(X_te))

    # 6. MissLearn FIML
    try:
        ml      = misslearn_factory()
        ml.fit(X_tr, y_tr)
        yp_fiml  = ml.predict(X_te)
        ypr_fiml = ml.predict_proba(X_te)[:, 1]
        out["MissLearn (FIML)"] = _clf_metrics(y_te, yp_fiml, ypr_fiml)
    except Exception as exc:
        out["MissLearn (FIML)"] = {**nan_c, "_error": str(exc)}

    gc.collect()
    return out


# ============================================================
# Cross-validation runner
# ============================================================

def run_cv(dataset, sklearn_factory, misslearn_factory,
           task="regression", n_splits=5, rng_seed=42):
    """K-fold CV over all 6 missing-data methods for one dataset.

    Every method sees the same missing data; a fair comparison of strategies
    for handling missingness, not a comparison against a complete-data ceiling.

    Parameters
    ----------
    dataset           : dict from make_regression_datasets / make_classification_datasets
    sklearn_factory   : callable() → new sklearn estimator
    misslearn_factory : callable() → new MissLearn estimator
    task              : "regression" or "classification"
    n_splits          : number of CV folds
    rng_seed          : random state for fold splitting and MICE

    Returns
    -------
    List of per-fold result dicts (one per fold).
    """
    X = dataset["X"]
    y = dataset["y"]

    fold_fn = (evaluate_fold_regression if task == "regression"
               else evaluate_fold_classification)

    if task == "regression":
        kf     = KFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
        splits = list(kf.split(X))
    else:
        kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
        splits = list(kf.split(X, y))

    return [
        fold_fn(
            X[tr], y[tr], X[te], y[te],
            sklearn_factory, misslearn_factory, rng_seed,
        )
        for tr, te in splits
    ]


def aggregate_folds(fold_records):
    """Collapse per-fold result dicts into {method: {metric: (mean, std)}}.

    std uses ddof=1.  NaN folds are excluded per metric.
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


# ============================================================
# Results table
# ============================================================

def make_results_table(datasets, all_agg, metric_meta):
    """Build a mean ± std summary DataFrame."""
    rows = []
    for ds in datasets:
        name = ds["name"]
        for method in METHODS:
            row = {"Dataset": name, "Method": method}
            for metric, label, higher, _ in metric_meta:
                mu, sd = all_agg[name][method][metric]
                arrow  = "↑" if higher else "↓"
                row[f"{label} {arrow}"] = (
                    f"{mu:.4f} ± {sd:.4f}"
                    if np.isfinite(mu) and np.isfinite(sd)
                    else "-"
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["Dataset", "Method"])


def dataset_summary_table(reg_datasets, clf_datasets):
    """Overview table of all benchmark datasets."""
    rows = []
    for ds in reg_datasets:
        rows.append({
            "Task":         "Regression",
            "Name":         ds["name"],
            "n":            ds["n"],
            "p":            ds["p"],
            "Missing rate": f"{ds['missing_rate']*100:.1f}%",
            "Description":  ds["description"],
        })
    for ds in clf_datasets:
        rows.append({
            "Task":         "Classification",
            "Name":         ds["name"],
            "n":            ds["n"],
            "p":            ds["p"],
            "Missing rate": f"{ds['missing_rate']*100:.1f}%",
            "Description":  ds["description"],
        })
    return pd.DataFrame(rows).set_index(["Task", "Name"])


# ============================================================
# Plotting
# ============================================================

def plot_missing_profile(datasets, title="Missing data profile",
                          col_labels=None, save_path=None):
    """Horizontal bar charts of per-feature missing rates.

    Always shown; illustrates the MAR pattern injected into each dataset.

    Layout: 2 columns, rows = ceil(n / 2).  Axes are filled column-by-column
    (top to bottom in col 0, then top to bottom in col 1), so callers control
    which task appears on which side simply by the order of ``datasets``.

    Parameters
    ----------
    datasets    : list of dataset dicts (pass classification first for left column)
    title       : figure suptitle
    col_labels  : optional list of 2 strings shown as column headers above
                  the top row, e.g. ["Classification", "Regression"]
    save_path   : optional file path to save the figure
    """
    n_ds   = len(datasets)
    n_cols = 2
    n_rows = (n_ds + 1) // 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8.5 * n_cols, 5.0 * n_rows),
        constrained_layout=True,
    )
    # Column-major order: fill col 0 top-to-bottom, then col 1 top-to-bottom.
    axes_flat = np.array(axes).ravel(order="F")

    for i, (ax, ds) in enumerate(zip(axes_flat, datasets)):
        miss_pct   = np.isnan(ds["X"]).mean(axis=0) * 100.0
        overall    = ds["missing_rate"] * 100.0
        feat_names = ds["feature_names"]
        order      = np.argsort(miss_pct)[::-1]
        y_pos      = np.arange(len(feat_names))
        ordered    = miss_pct[order]
        xmax       = max(miss_pct.max() * 1.40, 35)

        # Faint full-width track behind every row so that complete (0%-missing)
        # columns read as present-but-empty rather than as blank gaps in the
        # chart.  Injected columns are teal; complete columns are grey.
        ax.barh(y_pos, np.full(len(y_pos), xmax), color="#000000",
                alpha=0.05, height=0.6, zorder=0)
        bar_colors = ["#21908c" if p > 0 else "#c7c7c7" for p in ordered]
        bars = ax.barh(y_pos, ordered, color=bar_colors,
                       alpha=0.85, height=0.6, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[j] for j in order], fontsize=14)
        ax.set_xlabel("Missing (%)", fontsize=14)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlim(0, xmax)
        ax.axvline(x=overall, color="#440154", linestyle="--",
                   linewidth=1.5, label=f"overall {overall:.1f}%")
        ax.legend(fontsize=13, loc="lower right")

        # Column header prepended to the title of the top-row panel only
        row_idx = i % n_rows
        col_idx = i // n_rows
        col_hdr = ""
        if col_labels and row_idx == 0 and col_idx < len(col_labels):
            col_hdr = f"{col_labels[col_idx]}\n"

        ax.set_title(
            f"{col_hdr}{ds['name']}  (n={ds['n']:,}, p={ds['p']})",
            fontsize=15, pad=10,
        )
        for bar, pct in zip(bars, ordered):
            label = f"{pct:.1f}%" if pct > 0 else "complete"
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=13,
                    color="#333333" if pct > 0 else "#888888")

    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bars(datasets, all_agg, metric_meta,
              model_name, task, n_folds, save_path=None):
    """Bar chart: mean ± 1 std across folds, one figure per dataset.

    Each figure shows all metrics for one dataset as a single row of panels
    (or 2 × 2 for four metrics), keeping each panel large enough to read.
    Method names are on the x-axis so no legend is needed.

    Returns
    -------
    list of matplotlib Figure objects, one per dataset.
    """
    n_met  = len(metric_meta)
    # Layout: ≤3 metrics → 1 row; 4 metrics → 2 × 2
    if n_met <= 3:
        n_rows, n_cols = 1, n_met
    else:
        n_rows, n_cols = 2, 2

    figures = []
    for ds in datasets:
        name = ds["name"]
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(8 * n_cols, 7 * n_rows),
            constrained_layout=True,
        )
        axes_flat = np.array(axes).ravel() if (n_rows * n_cols) > 1 else [axes]

        for i, (metric, label, higher, direction) in enumerate(metric_meta):
            ax     = axes_flat[i]
            means  = [all_agg[name][m][metric][0] for m in METHODS]
            stds   = [all_agg[name][m][metric][1] for m in METHODS]
            colors = [COLORS[m] for m in METHODS]

            bars = ax.bar(
                range(len(METHODS)), means, yerr=stds,
                color=colors, alpha=0.85, capsize=6, width=0.62,
                error_kw=dict(elinewidth=1.8, ecolor="black", capthick=1.8),
            )
            for bar, mu, sd in zip(bars, means, stds):
                if np.isfinite(mu):
                    clearance = (sd if np.isfinite(sd) else 0.0) + abs(mu) * 0.01
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + clearance,
                        f"{mu:.3g}",
                        ha="center", va="bottom", fontsize=14,
                    )

            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(SHORT_LABELS, fontsize=12.5)
            ax.set_ylabel(label, fontsize=13)
            ax.set_title(f"{label}  ({direction})", fontsize=13, pad=9)
            ax.tick_params(axis="y", labelsize=11)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

            finite_means = [v for v in means if np.isfinite(v)]
            if finite_means:
                y_hi = max(
                    m + (s if np.isfinite(s) else 0.0)
                    for m, s in zip(means, stds) if np.isfinite(m)
                )
                y_lo  = min(finite_means)
                span  = max(y_hi - y_lo, abs(y_hi) * 0.05 + 0.01)
                bottom = (max(0.0, y_lo - span * 0.15) if not higher
                          else y_lo - span * 0.15)
                ax.set_ylim(bottom=bottom, top=y_hi + span * 0.60)

        for ax in axes_flat[len(metric_meta):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{model_name}; {name}  ({task.capitalize()}, "
            f"{n_folds}-fold CV, error bars = ±1 std)",
            fontsize=15,
        )

        if save_path:
            base, ext = os.path.splitext(save_path)
            safe = name.replace(" ", "_").replace("-", "_")
            fig.savefig(f"{base}_{safe}{ext}", dpi=150, bbox_inches="tight")

        figures.append(fig)

    return figures


def plot_strips(datasets, all_fold_records, metric_meta,
                model_name, task, save_path=None):
    """Strip / dot plot: individual fold values with fold-mean line overlaid.

    One figure per dataset, metrics as columns (or 2 × 2 for four metrics).
    Method names on the x-axis; no legend needed.

    Returns
    -------
    list of matplotlib Figure objects, one per dataset.
    """
    n_met  = len(metric_meta)
    if n_met <= 3:
        n_rows, n_cols = 1, n_met
    else:
        n_rows, n_cols = 2, 2

    JITTER  = 0.15
    figures = []

    for ds in datasets:
        name  = ds["name"]
        folds = all_fold_records[name]

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(8 * n_cols, 7 * n_rows),
            constrained_layout=True,
        )
        axes_flat = np.array(axes).ravel() if (n_rows * n_cols) > 1 else [axes]

        for i, (metric, label, higher, direction) in enumerate(metric_meta):
            ax = axes_flat[i]
            for mi, method in enumerate(METHODS):
                vals = np.array(
                    [f[method].get(metric, np.nan) for f in folds], dtype=float
                )
                xs = mi + np.linspace(-JITTER, JITTER, len(vals))
                ax.scatter(xs, vals, color=COLORS[method], s=60, zorder=3,
                           edgecolors="white", linewidths=0.6, alpha=0.90)
                mu = float(np.nanmean(vals))
                ax.plot([mi - 0.30, mi + 0.30], [mu, mu],
                        color=COLORS[method], linewidth=3.0, zorder=4,
                        solid_capstyle="round")

            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(SHORT_LABELS, fontsize=12.5)
            ax.set_ylabel(label, fontsize=13)
            ax.set_title(f"{label}  ({direction})", fontsize=13, pad=9)
            ax.tick_params(axis="y", labelsize=11)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

        for ax in axes_flat[len(metric_meta):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{model_name}; {name}  ({task.capitalize()}): "
            "per-fold values (dots) · fold means (bars)",
            fontsize=15,
        )

        if save_path:
            base, ext = os.path.splitext(save_path)
            safe = name.replace(" ", "_").replace("-", "_")
            fig.savefig(f"{base}_{safe}{ext}", dpi=150, bbox_inches="tight")

        figures.append(fig)

    return figures


def plot_gain_heatmap(datasets, all_agg, metric_meta,
                      model_name, task, save_path=None):
    """Heatmap of FIML % gain over each baseline, per dataset and metric.

    Yellow = FIML wins.  Purple = baseline wins.
    """
    FIML_KEY  = "MissLearn (FIML)"
    BASELINES = [m for m in METHODS if m != FIML_KEY]

    def pct_gain(fiml_mu, base_mu, higher):
        # Percentage gain relative to the baseline.  The denominator is floored
        # so a baseline value near zero (common for R², which can sit at ~0)
        # cannot produce a near-infinite gain that swamps the colour scale.
        if not (np.isfinite(fiml_mu) and np.isfinite(base_mu)):
            return np.nan
        denom = max(abs(base_mu), 1e-2)
        diff  = (fiml_mu - base_mu) if higher else (base_mu - fiml_mu)
        return float(diff / denom * 100.0)

    gain_rows = []
    for ds in datasets:
        name = ds["name"]
        for baseline in BASELINES:
            row = {"Dataset": name, "vs.": baseline}
            for metric, label, higher, _ in metric_meta:
                row[label] = pct_gain(
                    all_agg[name][FIML_KEY][metric][0],
                    all_agg[name][baseline][metric][0],
                    higher,
                )
            gain_rows.append(row)

    gain_df = (
        pd.DataFrame(gain_rows)
        .set_index(["Dataset", "vs."])
        .astype(float)
    )
    # Robust colour scale: use the 90th percentile of |gain| (not the max) so a
    # single extreme cell; e.g. FIML vs a baseline whose R² ≈ 0; saturates
    # rather than washing every other cell out to the neutral colour.
    _abs = gain_df.abs().to_numpy().ravel()
    _abs = _abs[np.isfinite(_abs)]
    vmax = float(np.nanpercentile(_abs, 90)) if _abs.size else 1.0
    vmax = max(vmax, 5.0)
    # Honest but compact annotations: drop decimals once the magnitude is large.
    annot = gain_df.applymap(
        lambda v: "" if not np.isfinite(v)
        else (f"{v:+.0f}" if abs(v) >= 100 else f"{v:+.1f}")
    )

    # Size the figure so each cell is at least ~0.9 inches tall for readability
    cell_h = max(0.9, 7.0 / max(len(gain_df), 1))
    fig, ax = plt.subplots(
        figsize=(10, cell_h * len(gain_df) + 2.5),
        constrained_layout=True,
    )
    sns.heatmap(
        gain_df, ax=ax,
        annot=annot, fmt="",
        annot_kws={"size": 12, "fontweight": "normal"},
        center=0, vmin=-vmax, vmax=vmax,
        cmap="viridis", linewidths=0.8, linecolor="white",
        cbar_kws={"label": "% gain  (+ = FIML better)"},
    )
    ax.set_title(
        f"{model_name} ({task.capitalize()}): FIML % gain over each baseline\n"
        "Yellow = FIML wins   |   Purple = baseline wins",
        fontsize=13, pad=10,
    )
    ax.tick_params(axis="x", labelsize=12, rotation=20)
    ax.tick_params(axis="y", labelsize=11)
    try:
        ax.figure.axes[-1].tick_params(labelsize=11)   # colour-bar tick labels
    except Exception:
        pass
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# Statistical tests
# ============================================================

def compute_ttests(datasets, all_fold_records, metric_meta,
                   n_splits, alpha=0.05):
    """Paired t-tests: FIML vs each baseline, per dataset per metric.

    Low power at n_folds=5; treat p-values as indicative, not definitive.

    Returns a styled DataFrame.
    """
    FIML_KEY  = "MissLearn (FIML)"
    BASELINES = [m for m in METHODS if m != FIML_KEY]

    rows = []
    for ds in datasets:
        name  = ds["name"]
        folds = all_fold_records[name]
        for metric, label, higher, direction in metric_meta:
            fiml_vals = np.array(
                [f[FIML_KEY].get(metric, np.nan) for f in folds]
            )
            for baseline in BASELINES:
                base_vals = np.array(
                    [f[baseline].get(metric, np.nan) for f in folds]
                )
                ok = ~(np.isnan(fiml_vals) | np.isnan(base_vals))
                t_stat, p_val = (
                    ttest_rel(fiml_vals[ok], base_vals[ok])
                    if ok.sum() >= 3
                    else (np.nan, np.nan)
                )
                diff_mu   = float(np.nanmean(fiml_vals) - np.nanmean(base_vals))
                fiml_wins = (diff_mu > 0) if higher else (diff_mu < 0)
                rows.append({
                    "Dataset":      name,
                    "Metric":       label,
                    "vs. Baseline": baseline,
                    "FIML mean":    round(float(np.nanmean(fiml_vals)), 4),
                    "Base mean":    round(float(np.nanmean(base_vals)), 4),
                    "Diff":         round(diff_mu, 4),
                    "t-stat":       (round(float(t_stat), 3)
                                     if np.isfinite(t_stat) else np.nan),
                    "p-value":      (round(float(p_val), 4)
                                     if np.isfinite(p_val) else np.nan),
                    "sig.":         "*" if (np.isfinite(p_val) and p_val < alpha) else "",
                    "FIML wins":    "Yes" if fiml_wins else "No",
                })

    return (
        pd.DataFrame(rows)
        .set_index(["Dataset", "Metric", "vs. Baseline"])
    )


# ============================================================
# Save utilities
# ============================================================

def save_results(results_dir, model_name, task,
                 datasets, all_fold_records, all_agg, metric_meta):
    """Write raw fold scores and summary table to CSV in results_dir."""
    os.makedirs(results_dir, exist_ok=True)

    # Raw per-fold scores
    raw_rows = []
    for ds in datasets:
        name  = ds["name"]
        folds = all_fold_records[name]
        for fold_i, fold in enumerate(folds):
            for method, scores in fold.items():
                if method.startswith("_"):
                    continue
                row = {"dataset": name, "method": method, "fold": fold_i}
                row.update({
                    k: v for k, v in scores.items()
                    if not k.startswith("_") and isinstance(v, float)
                })
                raw_rows.append(row)
    raw_df   = pd.DataFrame(raw_rows)
    raw_path = os.path.join(results_dir, f"{model_name}_{task}_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"  Raw scores  → {raw_path}")

    # Summary (mean ± std)
    summary_df   = make_results_table(datasets, all_agg, metric_meta)
    summary_path = os.path.join(results_dir, f"{model_name}_{task}_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"  Summary     → {summary_path}")


def save_sweep_results(results_dir, model_name, task, agg, zero=None):
    """Write the per-rate sweep aggregates to CSV in results_dir.

    The sweep previously saved figures only, so a run that took hours left
    nothing you could tabulate or diff: the curves were on disk but the numbers
    behind them were not. This writes them in long form, one row per
    (dataset, rate, method, metric), which is the shape the paper's own tables
    are built from.

    agg is {dataset: {rate: {method: {metric: (mean, std)}}}} as returned by
    aggregate_folds, and zero, if given, is the complete-data reference in the
    same shape minus the rate level. The reference is written as rate 0.0 so a
    single file carries the whole curve including its left endpoint.
    """
    os.makedirs(results_dir, exist_ok=True)

    rows = []

    def _emit(ds_name, rate, per_method):
        for method, metrics in per_method.items():
            if method.startswith("_"):
                continue
            for metric, val in metrics.items():
                if metric.startswith("_"):
                    continue
                mean, std = val if isinstance(val, tuple) else (val, float("nan"))
                rows.append({"dataset": ds_name, "missing_rate": float(rate),
                             "method": method, "metric": metric,
                             "mean": mean, "std": std})

    for ds_name in agg:
        if zero and ds_name in zero:
            _emit(ds_name, 0.0, zero[ds_name])
        for rate in sorted(agg[ds_name]):
            _emit(ds_name, rate, agg[ds_name][rate])

    df   = pd.DataFrame(rows).sort_values(
        ["dataset", "metric", "method", "missing_rate"], kind="stable")
    path = os.path.join(results_dir, f"{model_name}_{task}_sweep.csv")
    df.to_csv(path, index=False)
    print(f"  Sweep table → {path}")
    return path


# ── Sweep engine ──────────────────────────────────────────────────────────

# Visual style constants shared by all sweep plots.
_SWEEP_LINESTYLES = {
    "Drop Rows":          (0, (3, 1)),   # densely dashed
    "Drop Cols":          (0, (5, 2)),   # dashed
    "Mean Imputation":    (0, (1, 1)),   # dotted
    "KNN Imputation":     (0, (3, 1, 1, 1)),  # dash-dot
    "MICE (Iterative)":   (0, (5, 2, 1, 2)),  # dash-dot-dot
    "MissLearn (FIML)":   "solid",
}
_SWEEP_MARKERS = {
    "Drop Rows":          "o",
    "Drop Cols":          "s",
    "Mean Imputation":    "^",
    "KNN Imputation":     "D",
    "MICE (Iterative)":   "v",
    "MissLearn (FIML)":   "*",
}


def run_cv_at_rate(dataset, sklearn_factory, misslearn_factory,
                   task, miss_rates, n_splits=5, rng_seed=42):
    """Sweep CV: re-inject MAR at each rate in *miss_rates* and run K-fold CV.

    For each rate a fresh injection seed is derived from the rate and
    *rng_seed* so each rate gets an independently sampled missing pattern
    while remaining fully reproducible across runs.

    Uses ``StratifiedKFold`` when ``task == "classification"``, ``KFold``
    otherwise.

    Parameters
    ----------
    dataset           : dict from make_regression_datasets /
                        make_classification_datasets; the **complete**
                        ``X_complete`` array is used as the base; if absent
                        the ``X`` array is used as-is (pattern already in it).
    sklearn_factory   : callable() → new sklearn estimator
    misslearn_factory : callable() → new MissLearn estimator
    task              : ``"regression"`` or ``"classification"``
    miss_rates        : sequence of floats, e.g. ``[0.05, 0.10, ..., 0.50]``
    n_splits          : number of CV folds
    rng_seed          : base random state

    Returns
    -------
    dict  {rate: list_of_fold_dicts}
        Inner fold dicts have the same shape as those returned by
        ``run_cv``: keyed by method name, values are metric dicts.
    """
    # Prefer the complete base matrix so we re-inject cleanly at every rate.
    base_X = dataset.get("X_complete", dataset["X"])
    y      = dataset["y"]

    fold_fn = (evaluate_fold_regression if task == "regression"
               else evaluate_fold_classification)

    results = {}
    for rate in miss_rates:
        inject_seed = rng_seed + int(rate * 1000)
        X, _        = inject_mar(base_X, rate=rate, seed=inject_seed)

        if task == "regression":
            kf     = KFold(n_splits=n_splits, shuffle=True,
                           random_state=rng_seed)
            splits = list(kf.split(X))
        else:
            kf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=rng_seed)
            splits = list(kf.split(X, y))

        results[rate] = [
            fold_fn(X[tr], y[tr], X[te], y[te],
                    sklearn_factory, misslearn_factory, rng_seed)
            for tr, te in splits
        ]
    return results


def compute_sweep_crossover(datasets, sweep_results, metric_meta, miss_rates):
    """Find the lowest missing rate at which FIML first outperforms each baseline.

    Parameters
    ----------
    datasets      : list of dataset dicts
    sweep_results : {ds_name: {rate: list_of_fold_dicts}}; raw fold records
                    *or* {ds_name: {rate: {method: {metric: (mu, sd)}}}}; already
                    aggregated.  The function handles both by checking the type of
                    the innermost value.
    metric_meta   : list of (key, label, higher_is_better, direction) tuples
    miss_rates    : ordered sequence of float rates (ascending)

    Returns
    -------
    pd.DataFrame indexed by [Dataset, Metric, vs. Baseline] with columns
    ``Crossover rate`` and ``FIML wins all``.
    """
    FIML_KEY  = "MissLearn (FIML)"
    BASELINES = [m for m in METHODS if m != FIML_KEY]

    def _get_mu(rate_data, rate, method, metric):
        """Extract mean value whether rate_data holds fold lists or agg dicts."""
        entry = rate_data[rate]
        # Aggregated: {method: {metric: (mu, sd)}}
        if isinstance(entry, dict) and isinstance(
                next(iter(entry.values())), dict):
            inner = entry[method][metric]
            return inner[0] if isinstance(inner, tuple) else float(inner)
        # Raw fold list
        vals = np.array([f[method].get(metric, np.nan) for f in entry],
                        dtype=float)
        return float(np.nanmean(vals))

    rows = []
    for ds in datasets:
        name      = ds["name"]
        rate_data = sweep_results[name]
        for metric, met_label, higher, _ in metric_meta:
            for baseline in BASELINES:
                crossover_rate = None
                for rate in miss_rates:
                    fiml_mu = _get_mu(rate_data, rate, FIML_KEY,  metric)
                    base_mu = _get_mu(rate_data, rate, baseline,  metric)
                    if np.isnan(fiml_mu) or np.isnan(base_mu):
                        continue
                    fiml_wins = (fiml_mu > base_mu) if higher else (fiml_mu < base_mu)
                    if fiml_wins:
                        crossover_rate = rate
                        break
                rows.append({
                    "Dataset":        name,
                    "Metric":         met_label,
                    "vs. Baseline":   baseline,
                    "Crossover rate": (
                        f"{crossover_rate * 100:.0f}%"
                        if crossover_rate is not None else "Never"
                    ),
                    "FIML wins all":  (
                        "Yes"     if crossover_rate == miss_rates[0]
                        else "No" if crossover_rate is None
                        else "Partial"
                    ),
                })

    return pd.DataFrame(rows).set_index(["Dataset", "Metric", "vs. Baseline"])


# ── Sweep visualisation ───────────────────────────────────────────────────

def plot_sweep_lines(datasets, sweep_results, metric_meta,
                     model_name, task, miss_rates,
                     n_splits=5, save_path=None):
    """Performance vs. missingness rate line plots.

    One panel per dataset × metric.  Each line is one method with a shaded
    ±1 std band.  FIML is always drawn last (on top) with a solid line.

    Works for both regression and classification; pass the appropriate
    ``metric_meta``.

    Parameters
    ----------
    datasets      : list of dataset dicts
    sweep_results : {ds_name: {rate: {method: {metric: (mu, sd)}}}}
                    Output of ``{aggregate_folds(run_cv_at_rate(...))}`` keyed
                    first by dataset name then by rate.
    metric_meta   : list of (key, label, higher_is_better, direction) tuples
    model_name    : string used in the suptitle
    task          : ``"regression"`` or ``"classification"``
    miss_rates    : ordered sequence of float rates
    n_splits      : used in the suptitle
    save_path     : optional base path; ``_{ds_name}.png`` is appended per figure

    Returns
    -------
    list of matplotlib Figure objects, one per dataset.
    """
    FIML_KEY  = "MissLearn (FIML)"
    rates_pct = [r * 100 for r in miss_rates]
    n_met     = len(metric_meta)
    if n_met <= 3:
        n_rows, n_cols = 1, n_met
    else:
        n_rows, n_cols = 2, 2

    figures = []
    for ds in datasets:
        name = ds["name"]
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4.5 * n_rows),
            constrained_layout=True,
        )
        axes_flat = np.array(axes).ravel() if (n_rows * n_cols) > 1 else [axes]

        for i, (metric, label, higher, direction) in enumerate(metric_meta):
            ax         = axes_flat[i]
            draw_order = [m for m in METHODS if m != FIML_KEY] + [FIML_KEY]

            for method in draw_order:
                means = np.array([
                    sweep_results[name][r][method][metric][0]
                    for r in miss_rates
                ], dtype=float)
                stds  = np.array([
                    sweep_results[name][r][method][metric][1]
                    for r in miss_rates
                ], dtype=float)
                stds_safe = np.where(np.isfinite(stds), stds, 0.0)
                is_fiml   = method == FIML_KEY

                ax.plot(
                    rates_pct, means,
                    color=COLORS[method],
                    linestyle=_SWEEP_LINESTYLES[method],
                    marker=_SWEEP_MARKERS[method],
                    linewidth=2.5 if is_fiml else 1.5,
                    markersize=5,
                    zorder=5 if is_fiml else 3,
                    label=method,
                )
                ax.fill_between(
                    rates_pct,
                    means - stds_safe,
                    means + stds_safe,
                    color=COLORS[method],
                    alpha=0.10,
                    zorder=4 if is_fiml else 2,
                )

            ax.set_xlabel("Missing rate (%)", fontsize=12.5)
            ax.set_ylabel(label, fontsize=12.5)
            ax.set_title(f"{label}  ({direction})", fontsize=13, pad=9)
            ax.set_xticks(rates_pct)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
            ax.tick_params(axis="x", labelsize=11, rotation=30)
            ax.tick_params(axis="y", labelsize=11)

            # Tighten y-axis around finite data
            finite = [
                sweep_results[name][r][m][metric][0]
                for r in miss_rates for m in METHODS
                if np.isfinite(sweep_results[name][r][m][metric][0])
            ]
            if finite:
                lo, hi = min(finite), max(finite)
                span   = max(hi - lo, abs(hi) * 0.02, 0.01)
                if higher:
                    ax.set_ylim(bottom=max(0.0, lo - span * 0.3),
                                top=min(1.0, hi + span * 0.3)
                                    if max(finite) <= 1.0
                                    else hi + span * 0.3)
                else:
                    ax.set_ylim(bottom=max(0.0, lo - span * 0.15),
                                top=hi + span * 0.5)

        for ax in axes_flat[n_met:]:
            ax.set_visible(False)

        handles, labels_leg = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg,
                   loc="outside lower center", ncol=3, fontsize=12.5, frameon=True)

        fig.suptitle(
            f"{model_name}; {name}  ({task.capitalize()}): "
            f"performance vs. missingness rate  ({n_splits}-fold CV)\n"
            "Shaded band = ±1 std across folds",
            fontsize=14,
        )

        if save_path:
            base, ext = os.path.splitext(save_path)
            safe = name.replace(" ", "_").replace("-", "_")
            fig.savefig(f"{base}_{safe}{ext}", dpi=150, bbox_inches="tight")

        figures.append(fig)

    return figures


def plot_sweep_gain_heatmap(datasets, sweep_results, metric_meta,
                             model_name, task, miss_rates, save_path=None):
    """Per-rate gain heatmap: FIML vs each baseline across all missing rates.

    One figure per dataset × metric.  Rows = missing rates, cols = baselines.
    Yellow = FIML wins, purple = baseline wins.

    Parameters
    ----------
    datasets      : list of dataset dicts
    sweep_results : {ds_name: {rate: {method: {metric: (mu, sd)}}}}
    metric_meta   : list of (key, label, higher_is_better, direction) tuples
    model_name    : string for figure titles
    task          : ``"regression"`` or ``"classification"``
    miss_rates    : ordered sequence of float rates
    save_path     : optional base path; ``_{ds_name}_{metric}.png`` appended

    Returns
    -------
    list of matplotlib Figure objects (one per dataset × metric combination).
    """
    FIML_KEY  = "MissLearn (FIML)"
    BASELINES = [m for m in METHODS if m != FIML_KEY]

    def _pct_gain(fiml_mu, base_mu, higher):
        # Denominator floored so a near-zero baseline (e.g. R² ≈ 0) cannot
        # blow the gain up and dominate the colour scale.
        if not (np.isfinite(fiml_mu) and np.isfinite(base_mu)):
            return np.nan
        denom = max(abs(base_mu), 1e-2)
        diff  = (fiml_mu - base_mu) if higher else (base_mu - fiml_mu)
        return float(diff / denom * 100.0)

    figures = []
    for ds in datasets:
        name = ds["name"]
        for metric, met_label, higher, direction in metric_meta:
            gain_data = {}
            for baseline in BASELINES:
                col = [
                    _pct_gain(
                        sweep_results[name][r][FIML_KEY][metric][0],
                        sweep_results[name][r][baseline][metric][0],
                        higher,
                    )
                    for r in miss_rates
                ]
                gain_data[baseline] = col

            gain_df = pd.DataFrame(
                gain_data,
                index=[f"{r * 100:.0f}%" for r in miss_rates],
            ).astype(float)
            gain_df.index.name = "Missing rate"
            # Robust colour scale (90th pct of |gain|) so one extreme cell
            # saturates instead of flattening the whole map.
            _abs = gain_df.abs().to_numpy().ravel()
            _abs = _abs[np.isfinite(_abs)]
            vmax = float(np.nanpercentile(_abs, 90)) if _abs.size else 1.0
            vmax = max(vmax, 5.0)
            annot = gain_df.applymap(
                lambda v: "" if not np.isfinite(v)
                else (f"{v:+.0f}" if abs(v) >= 100 else f"{v:+.1f}")
            )

            fig, ax = plt.subplots(
                figsize=(9, 0.55 * len(miss_rates) + 2.2),
                constrained_layout=True,
            )
            sns.heatmap(
                gain_df, ax=ax,
                annot=annot, fmt="",
                annot_kws={"size": 11, "fontweight": "normal"},
                center=0, vmin=-vmax, vmax=vmax,
                cmap="viridis", linewidths=0.6, linecolor="white",
                cbar_kws={"label": "% gain  (+ = FIML better)"},
            )
            ax.set_title(
                f"{model_name} ({task.capitalize()}); {name}\n"
                f"FIML % gain over baselines  |  {met_label}  ({direction})\n"
                "Yellow = FIML wins   |   Purple = baseline wins",
                fontsize=12.5, pad=10,
            )
            ax.tick_params(axis="x", labelsize=11, rotation=15)
            ax.tick_params(axis="y", labelsize=11)
            try:
                ax.figure.axes[-1].tick_params(labelsize=10)
            except Exception:
                pass

            if save_path:
                base, ext = os.path.splitext(save_path)
                safe = name.replace(" ", "_").replace("-", "_")
                fig.savefig(f"{base}_{safe}_{metric}{ext}",
                            dpi=150, bbox_inches="tight")

            figures.append(fig)

    return figures


def plot_cc_retention(datasets, save_path=None):
    """Grouped bar chart of rows retained vs. discarded by listwise deletion.

    For each dataset shows the number of complete-case rows (used by Drop
    Rows) alongside the rows that would be discarded.  Also prints a summary
    table to stdout.

    Parameters
    ----------
    datasets  : list of dataset dicts (with ``X`` and ``missing_rate`` keys)
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    cc_rows = []
    for ds in datasets:
        X          = ds["X"]
        n_total    = X.shape[0]
        n_complete = int((~np.isnan(X).any(axis=1)).sum())
        pct        = n_complete / n_total * 100.0
        theoretical = (1.0 - ds["missing_rate"]) ** ds["p"] * 100.0
        cc_rows.append({
            "Dataset":              ds["name"],
            "n total":              n_total,
            "n complete cases":     n_complete,
            "Retention (%)":        round(pct, 1),
            "MCAR theoretical (%)": round(theoretical, 1),
        })

    ds_names  = [r["Dataset"]          for r in cc_rows]
    retained  = [r["n complete cases"] for r in cc_rows]
    discarded = [r["n total"] - r["n complete cases"] for r in cc_rows]
    x = np.arange(len(ds_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.0), constrained_layout=True)
    ax.bar(x - w / 2, retained,  w,
           label="Complete cases (Drop Rows trains on these)",
           color="#440154", alpha=0.85)
    ax.bar(x + w / 2, discarded, w,
           label="Discarded by Drop Rows (all other methods use these)",
           color="#fde725", alpha=0.85)

    y_max = max(r + d for r, d in zip(retained, discarded))
    for xi, (r, d) in enumerate(zip(retained, discarded)):
        ax.text(xi - w / 2, r + y_max * 0.01, str(r),
                ha="center", va="bottom", fontsize=12.5)
        ax.text(xi + w / 2, d + y_max * 0.01, str(d),
                ha="center", va="bottom", fontsize=12.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=12.5)
    ax.set_ylabel("Number of rows", fontsize=12.5)
    ax.set_title("Rows retained vs. discarded by listwise deletion (Drop Rows)",
                 fontsize=12.5, pad=10)
    ax.legend(fontsize=12.5)
    ax.tick_params(axis="y", labelsize=10)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_balance(datasets, save_path=None):
    """Grouped bar chart of positive / negative class frequencies.

    Intended for classification benchmark datasets.  Each dataset dict must
    contain a ``pos_rate`` key (float in [0, 1]).

    Parameters
    ----------
    datasets  : list of classification dataset dicts
    save_path : optional file path

    Returns
    -------
    matplotlib Figure
    """
    ds_names  = [ds["name"]              for ds in datasets]
    pos_rates = [ds["pos_rate"] * 100.0  for ds in datasets]
    neg_rates = [100.0 - r               for r in pos_rates]
    x = np.arange(len(ds_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    bars_pos = ax.bar(x - w / 2, pos_rates, w, label="Positive class",
                      color="#fde725", alpha=0.85)
    bars_neg = ax.bar(x + w / 2, neg_rates, w, label="Negative class",
                      color="#440154", alpha=0.85)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="50 % balance")

    for bar, val in zip(bars_pos, pos_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, fontsize=12.5)
    ax.set_ylabel("Class frequency (%)", fontsize=12.5)
    ax.set_ylim(0, 110)
    ax.set_title("Class balance across benchmark datasets",
                 fontsize=12.5, pad=10)
    ax.legend(fontsize=12.5)
    ax.tick_params(axis="y", labelsize=10)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Additional sweep visualisation ────────────────────────────────────────────

def plot_sweep_degradation(datasets, sweep_results, zero_rate_agg,
                           metric_meta, model_name, task,
                           miss_rates, save_path=None):
    """Degradation curves: how much does each method's performance change
    relative to its own complete-data (0 % missingness) baseline?

    Sign convention (same for all metrics):
        positive  → method *held up* compared to complete data
        negative  → method *degraded* compared to complete data

    For higher-is-better metrics (R², AUC, …):
        degradation = metric(rate) − metric(0%)
    For lower-is-better metrics (RMSE, MAE, Brier):
        degradation = metric(0%) − metric(rate)

    Parameters
    ----------
    datasets       : list of dataset dicts
    sweep_results  : {ds_name: {rate: {method: {metric: (mean, std)}}}}
    zero_rate_agg  : {ds_name: {method: {metric: (mean, std)}}}
                     result of ``aggregate_folds`` run on complete data
    metric_meta    : list of (key, label, higher_is_better, direction_str)
    model_name     : str; used in the figure title
    task           : "regression" | "classification"
    miss_rates     : list of float rates used in the sweep
    save_path      : optional path prefix; one file per dataset

    Returns
    -------
    list of matplotlib Figure objects (one per dataset)
    """
    rates_pct = [r * 100 for r in miss_rates]
    deg_metrics = [(k, lbl, hi) for k, lbl, hi, _ in metric_meta]
    n_met = len(deg_metrics)
    figs = []

    for ds in datasets:
        name = ds["name"]
        fig, axes = plt.subplots(
            1, n_met,
            figsize=(6.0 * n_met, 4.5),
            constrained_layout=True,
        )
        if n_met == 1:
            axes = [axes]

        for col, (metric, met_label, higher) in enumerate(deg_metrics):
            ax = axes[col]
            draw_order = ([m for m in METHODS if m != "MissLearn (FIML)"]
                          + ["MissLearn (FIML)"])
            for method in draw_order:
                base_mu = zero_rate_agg.get(name, {}).get(
                    method, {}).get(metric, (np.nan,))[0]
                degs = []
                for rate in miss_rates:
                    mu = (sweep_results.get(name, {})
                          .get(rate, {})
                          .get(method, {})
                          .get(metric, (np.nan,))[0])
                    if np.isnan(mu) or np.isnan(base_mu):
                        degs.append(np.nan)
                    elif higher:
                        degs.append(mu - base_mu)
                    else:
                        degs.append(base_mu - mu)

                lw     = 2.5 if method == "MissLearn (FIML)" else 1.5
                zorder = 5  if method == "MissLearn (FIML)" else 3
                ax.plot(rates_pct, degs,
                        color=COLORS[method],
                        linestyle=_SWEEP_LINESTYLES[method],
                        marker=_SWEEP_MARKERS[method],
                        linewidth=lw, markersize=5, zorder=zorder,
                        label=method)

            ax.axhline(y=0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.set_xlabel("Missing rate (%)", fontsize=12.5)
            ax.set_ylabel("Change from 0% baseline", fontsize=12.5)
            ax.set_title(f"{name}; {met_label}", fontsize=12.5, pad=9)
            ax.set_xticks(rates_pct)
            ax.xaxis.set_major_formatter(
                mticker.FormatStrFormatter("%g%%"))
            ax.tick_params(axis="x", labelsize=10, rotation=30)
            ax.tick_params(axis="y", labelsize=10)

        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="outside lower center",
                   ncol=min(len(METHODS), 6), fontsize=12.5, frameon=True)
        fig.suptitle(
            f"Degradation relative to complete-data baseline; {name}\n"
            "Closer to zero = method held up better as missingness increased",
            fontsize=12.5,
        )

        if save_path:
            fig.savefig(f"{save_path}_{name}_degradation.png",
                        dpi=150, bbox_inches="tight")
        figs.append(fig)

    return figs


def plot_sweep_rank(datasets, sweep_results, rank_metric,
                    model_name, task, miss_rates, save_path=None):
    """Method rank plot: at each missingness level, rank all methods by
    the chosen metric.  Rank 1 = best.

    Parameters
    ----------
    datasets     : list of dataset dicts
    sweep_results: {ds_name: {rate: {method: {metric: (mean, std)}}}}
    rank_metric  : str; metric key to rank on, e.g. ``"r2"`` or ``"auc"``
    model_name   : str; used in the figure title
    task         : "regression" | "classification"
    miss_rates   : list of float rates
    save_path    : optional path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    rates_pct = [r * 100 for r in miss_rates]
    n_ds = len(datasets)

    fig, axes = plt.subplots(
        1, n_ds,
        figsize=(6.5 * n_ds, 4.5),
        constrained_layout=True,
    )
    if n_ds == 1:
        axes = [axes]

    higher_map = {"r2": True, "auc": True, "accuracy": True,
                  "f1": True, "rmse": False, "mae": False, "brier": False}
    higher = higher_map.get(rank_metric, True)

    for ax, ds in zip(axes, datasets):
        name = ds["name"]
        n_methods = len(METHODS)
        rank_matrix = np.zeros((len(miss_rates), n_methods))

        for ri, rate in enumerate(miss_rates):
            vals = np.array([
                sweep_results.get(name, {}).get(rate, {})
                .get(m, {}).get(rank_metric, (np.nan,))[0]
                for m in METHODS
            ], dtype=float)
            nan_mask = np.isnan(vals)
            vals[nan_mask] = -np.inf if higher else np.inf
            order = (np.argsort(vals)[::-1] if higher
                     else np.argsort(vals))
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, n_methods + 1)
            ranks[nan_mask] = n_methods
            rank_matrix[ri] = ranks

        for mi, method in enumerate(METHODS):
            lw     = 2.5 if method == "MissLearn (FIML)" else 1.5
            zorder = 5  if method == "MissLearn (FIML)" else 3
            ax.plot(rates_pct, rank_matrix[:, mi],
                    color=COLORS[method],
                    linestyle=_SWEEP_LINESTYLES[method],
                    marker=_SWEEP_MARKERS[method],
                    linewidth=lw, markersize=6, zorder=zorder,
                    label=method)

        ax.invert_yaxis()
        ax.set_yticks(range(1, n_methods + 1))
        ax.set_yticklabels([f"Rank {i}" for i in range(1, n_methods + 1)],
                           fontsize=12.5)
        ax.set_xlabel("Missing rate (%)", fontsize=12.5)
        ax.set_title(
            f"{name}\nMethod rank by {rank_metric.upper()} (1 = best)",
            fontsize=12.5, pad=9,
        )
        ax.set_xticks(rates_pct)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.tick_params(axis="x", labelsize=10, rotation=30)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(METHODS), 6), fontsize=12.5, frameon=True)
    fig.suptitle(
        f"Method ranking by {rank_metric.upper()} at each missingness level\n"
        "(rank 1 = best; higher on plot = better)",
        fontsize=12.5,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
