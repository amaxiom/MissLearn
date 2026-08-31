"""
performance_test_suite.py  --  MissLearn scaling and performance benchmarks
============================================================================

Import this module in the Performance_Test_Suite notebook, then call one
benchmark function per cell.

Naming convention
-----------------
All two-variant models use a consistent "(reg)" / "(clf)" suffix:
  MissLinear, MissLogistic          -- single-task, no suffix needed
  MissRidge (reg) / MissRidge (clf)
  MissLASSO (reg) / MissLASSO (clf)
  MissNeighbors (reg) / (clf)
  MissBayes (reg) / (clf)
  MissSupport (reg) / (clf)
  MissGaussian (reg) / (clf)        -- O(n³); capped at GP_PERF_CAP=150 in
                                       n-scaling; omitted from p / miss benchmarks
  MissEnsemble (reg) / (clf)

Quick reference
---------------
bench_model_comparison()      Fit / predict / score across all model families
bench_n_scaling()             Fit time vs sample size n  (all models, reg+clf)
bench_p_scaling()             Fit time vs features p    (all models, reg+clf)
bench_missingness_impact()    Fit time vs missing rate  (all models, reg+clf)
bench_ensemble_scaling()      MissEnsemble n_estimators (all bases, reg+clf)
bench_shapley_scaling()       MissShapley exact (2^p) vs KernelSHAP transition
bench_imputer_scaling()       MissImputer imputations (m) scaling
bench_crossval_scaling()      miss_cross_val_score cv-fold overhead
bench_gaussian_process()      MissGaussian O(n³) curve (small n)
bench_mixed_effects()         MissMixed n_groups scaling
bench_sensitivity()           MissSensitivity delta-grid scaling

All functions accept keyword overrides and return a result dict / list.
Pass plot=True to any function for matplotlib figures.
"""

import sys
import time
import warnings
import numpy as np

sys.path.insert(0, r"C:\Users\Amanda\Favorites\Machine Learning\MissLearn")
warnings.filterwarnings("ignore")

# MissGaussian is O(n³): skip in benchmarks that sweep large n or p.
# It has its own dedicated bench_gaussian_process() function.
GP_PERF_CAP = 150   # maximum n allowed for GP timing in bench_n_scaling


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_reg(n, p, miss=0.20, seed=0):
    """Regression dataset: clean signal, NaN mask applied to X (and 5% of y)."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, p))
    coef = np.zeros(p); coef[0] = 2.0; coef[min(p - 1, 1)] = -1.5
    y   = X @ coef + rng.standard_normal(n) * 0.5
    X[rng.random((n, p)) < miss] = np.nan
    y[rng.random(n)      < 0.05] = np.nan
    return X, y


def _make_clf(n, p, miss=0.20, seed=0):
    """Binary classification dataset: clean signal, NaN mask."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, p))
    coef = np.zeros(p); coef[0] = 2.0; coef[min(p - 1, 1)] = -1.5
    y   = (X @ coef > 0).astype(float)
    X[rng.random((n, p)) < miss] = np.nan
    y[rng.random(n)      < 0.05] = np.nan
    return X, y


def _timeit(fn, repeats=3):
    """Warm up once, then time `repeats` calls. Returns (mean_s, std_s)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fn()
        except Exception:
            return float("nan"), float("nan")
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                return float("nan"), float("nan")
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _fmt(s):
    if np.isnan(s):  return "  ERR  "
    if s < 0.001:    return f"{s * 1e3:.2f} ms"
    if s < 1.0:      return f"{s * 1e3:.1f} ms"
    return               f"{s:.2f} s"


def _fmt_score(v):
    return "  ERR  " if np.isnan(v) else f"{v:.4f}"


def _banner(title, params=""):
    w = 74
    print()
    print("=" * w)
    print(f"  {title}")
    if params:
        print(f"  {params}")
    print("=" * w)


def _table(headers, rows):
    widths = []
    for i, h in enumerate(headers):
        col_vals = [str(r[i]) for r in rows]
        widths.append(max(len(str(h)), max((len(v) for v in col_vals), default=0)))
    def fmt_row(vals):
        return "  " + "   ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    print(fmt_row(headers))
    print("  " + "   ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _score(model, X, y):
    try:
        return model.score(X, y)
    except Exception:
        return float("nan")


def _viridis(n):
    """Return n evenly-spaced viridis colours as a list."""
    import matplotlib.pyplot as plt
    cmap = plt.colormaps["viridis"]
    return [cmap(v) for v in np.linspace(0.05, 0.92, max(n, 1))]


# ---------------------------------------------------------------------------
# Canonical model specs: (label, constructor_lambda, task)
# ---------------------------------------------------------------------------

def _all_reg_specs():
    """All regression models as (label, lambda) pairs. GP listed last."""
    from MissLearn import (
        MissLinear,
        MissRidgeRegressor, MissLASSORegressor,
        MissNeighborsRegressor, MissBayesRegressor, MissSupportRegressor,
        MissGaussianRegressor,
    )
    return [
        ("MissLinear",          lambda: MissLinear()),
        ("MissRidge (reg)",     lambda: MissRidgeRegressor()),
        ("MissLASSO (reg)",     lambda: MissLASSORegressor()),
        ("MissNeighbors (reg)", lambda: MissNeighborsRegressor()),
        ("MissBayes (reg)",     lambda: MissBayesRegressor()),
        ("MissSupport (reg)",   lambda: MissSupportRegressor()),
        ("MissGaussian (reg)",  lambda: MissGaussianRegressor()),
    ]


def _all_clf_specs():
    """All classification models as (label, lambda) pairs. GP listed last."""
    from MissLearn import (
        MissLogistic,
        MissRidgeClassifier, MissLASSOClassifier,
        MissNeighborsClassifier, MissBayesClassifier, MissSupportClassifier,
        MissGaussianClassifier,
    )
    return [
        ("MissLogistic",        lambda: MissLogistic()),
        ("MissRidge (clf)",     lambda: MissRidgeClassifier()),
        ("MissLASSO (clf)",     lambda: MissLASSOClassifier()),
        ("MissNeighbors (clf)", lambda: MissNeighborsClassifier()),
        ("MissBayes (clf)",     lambda: MissBayesClassifier()),
        ("MissSupport (clf)",   lambda: MissSupportClassifier()),
        ("MissGaussian (clf)",  lambda: MissGaussianClassifier()),
    ]


_GP_LABELS = {"MissGaussian (reg)", "MissGaussian (clf)"}


# ---------------------------------------------------------------------------
# 1. Model family comparison
# ---------------------------------------------------------------------------

def bench_model_comparison(n=500, p=5, miss=0.20, repeats=3, plot=False):
    """
    Single-point fit / predict / score for every model family.

    Parameters
    ----------
    n        : training sample size
    p        : number of features
    miss     : fraction of X cells set to NaN
    repeats  : timing repeats (after one warmup)
    plot     : show horizontal bar chart of fit times

    Returns list of dicts  (model, task, fit_mean, fit_std, pred_mean, score).
    """
    from MissLearn import (
        MissLinear, MissLogistic,
        MissRidgeRegressor, MissRidgeClassifier,
        MissLASSORegressor, MissLASSOClassifier,
        MissNeighborsRegressor, MissNeighborsClassifier,
        MissBayesRegressor, MissBayesClassifier,
        MissSupportRegressor, MissSupportClassifier,
        MissGaussianRegressor, MissGaussianClassifier,
        MissEnsemble,
    )

    X_r, y_r    = _make_reg(n, p, miss)
    X_c, y_c    = _make_clf(n, p, miss)
    n_gp        = min(n, 80)
    X_gp_r, y_gp_r = _make_reg(n_gp, p, miss)
    X_gp_c, y_gp_c = _make_clf(n_gp, p, miss)

    specs = [
        # (label, model, X_tr, y_tr, task)
        ("MissLinear",          MissLinear(),                                       X_r,    y_r,    "reg"),
        ("MissLogistic",        MissLogistic(),                                     X_c,    y_c,    "clf"),
        ("MissRidge (reg)",     MissRidgeRegressor(),                               X_r,    y_r,    "reg"),
        ("MissRidge (clf)",     MissRidgeClassifier(),                              X_c,    y_c,    "clf"),
        ("MissLASSO (reg)",     MissLASSORegressor(),                               X_r,    y_r,    "reg"),
        ("MissLASSO (clf)",     MissLASSOClassifier(),                              X_c,    y_c,    "clf"),
        ("MissNeighbors (reg)", MissNeighborsRegressor(),                           X_r,    y_r,    "reg"),
        ("MissNeighbors (clf)", MissNeighborsClassifier(),                          X_c,    y_c,    "clf"),
        ("MissBayes (reg)",     MissBayesRegressor(),                               X_r,    y_r,    "reg"),
        ("MissBayes (clf)",     MissBayesClassifier(),                              X_c,    y_c,    "clf"),
        ("MissSupport (reg)",   MissSupportRegressor(),                             X_r,    y_r,    "reg"),
        ("MissSupport (clf)",   MissSupportClassifier(),                            X_c,    y_c,    "clf"),
        ("MissGaussian (reg)",  MissGaussianRegressor(),                            X_gp_r, y_gp_r, "reg"),
        ("MissGaussian (clf)",  MissGaussianClassifier(),                           X_gp_c, y_gp_c, "clf"),
        ("MissEnsemble (reg)",  MissEnsemble(estimator=MissRidgeRegressor(),
                                             n_estimators=10, random_state=0),      X_r,    y_r,    "reg"),
        ("MissEnsemble (clf)",  MissEnsemble(estimator=MissRidgeClassifier(),
                                             n_estimators=10, random_state=0),      X_c,    y_c,    "clf"),
    ]

    _banner("Model Family Comparison",
            f"n={n}  p={p}  miss={miss:.0%}  repeats={repeats}  "
            f"(GP capped at n={n_gp})")

    rows, results = [], []
    for label, model, Xtr, ytr, task in specs:
        fit_m, fit_s = _timeit(lambda m=model, X=Xtr, y=ytr: m.fit(X, y), repeats)
        pred_m, _    = _timeit(lambda m=model, X=Xtr: m.predict(X), repeats)
        sc           = _score(model, Xtr, ytr)
        rows.append((label, task,
                     _fmt(fit_m) + f" ±{_fmt(fit_s)}",
                     _fmt(pred_m), _fmt_score(sc)))
        results.append(dict(model=label, task=task,
                            fit_mean=fit_m, fit_std=fit_s,
                            pred_mean=pred_m, score=sc))

    _table(["Model", "Task", "Fit time", "Predict", "Score"], rows)

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        names  = [r["model"]    for r in results]
        fit_t  = [r["fit_mean"] for r in results]
        colors = ["#440154" if r["task"] == "reg" else "#21908C" for r in results]

        # Height scales with number of models so all bars are visible
        fig_h = max(6, len(results) * 0.48)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(range(len(names)), fit_t, color=colors, height=0.65)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Fit time (s)")
        ax.set_title(f"Model Family Comparison  (n={n}, p={p}, {miss:.0%} missing)")
        ax.legend(handles=[Patch(color="#440154", label="regression"),
                            Patch(color="#21908C", label="classification")],
                  fontsize=9)
        plt.tight_layout()
        plt.show()

    return results


# ---------------------------------------------------------------------------
# 2. n-scaling  (all models, reg + clf)
# ---------------------------------------------------------------------------

def bench_n_scaling(n_values=None, p=5, miss=0.20, repeats=2, plot=False):
    """
    Fit time vs sample size for ALL regression and classification models.

    MissGaussian is included but marked n/a for n > GP_PERF_CAP (={cap}).
    See bench_gaussian_process() for the dedicated O(n³) curve.

    Parameters
    ----------
    n_values : list of int  -- default [100, 250, 500, 1000, 2000]
    p        : int
    miss     : float
    repeats  : int
    plot     : bool  -- separate log-log plots for reg and clf

    Returns dict with keys 'reg' and 'clf', each a list of row dicts.
    """.format(cap=GP_PERF_CAP)
    if n_values is None:
        n_values = [100, 250, 500, 1000, 2000]

    reg_specs = _all_reg_specs()
    clf_specs = _all_clf_specs()

    all_results = {}
    for title, model_specs, task in [
        ("n-Scaling: Regression Models (all families)",    reg_specs, "reg"),
        ("n-Scaling: Classification Models (all families)", clf_specs, "clf"),
    ]:
        _banner(title,
                f"p={p}  miss={miss:.0%}  repeats={repeats}  "
                f"(GP capped at n={GP_PERF_CAP})")

        model_names = [ms[0] for ms in model_specs]
        rows, results = [], []

        for n in n_values:
            make = _make_reg if task == "reg" else _make_clf
            X, y = make(n, p, miss)
            row  = [str(n)]
            row_r = {"n": n}
            for mname, MCls in model_specs:
                if mname in _GP_LABELS and n > GP_PERF_CAP:
                    row.append("  n/a  ")
                    row_r[mname] = float("nan")
                else:
                    m = MCls()
                    t_m, _ = _timeit(lambda m=m, X=X, y=y: m.fit(X, y), repeats)
                    row.append(_fmt(t_m))
                    row_r[mname] = t_m
            rows.append(row)
            results.append(row_r)

        _table(["n"] + model_names, rows)
        all_results[task] = results

        if plot:
            import matplotlib.pyplot as plt
            palette = _viridis(len(model_specs))
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for i, (mname, _) in enumerate(model_specs):
                pts = [(r["n"], r[mname]) for r in results
                       if not np.isnan(r.get(mname, float("nan")))]
                if pts:
                    ns_v, ts_v = zip(*pts)
                    ls = "--" if mname in _GP_LABELS else "-"
                    ax.plot(ns_v, ts_v, marker="o", markersize=5,
                            label=mname, color=palette[i], linestyle=ls)
            ax.set_xlabel("n  (sample size)", fontsize=10)
            ax.set_ylabel("Fit time (s)", fontsize=10)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(title + f"\n(p={p}, {miss:.0%} missing)", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
            plt.tight_layout(); plt.show()

    return all_results


# ---------------------------------------------------------------------------
# 3. p-scaling  (all models, reg + clf)
# ---------------------------------------------------------------------------

def bench_p_scaling(p_values=None, n=500, miss=0.20, repeats=2, plot=False):
    """
    Fit time vs number of features p for ALL regression and classification models.

    MissGaussian is omitted here (n=500 makes it impractically slow; use
    bench_gaussian_process() for GP-specific scaling).

    Parameters
    ----------
    p_values : list of int  -- default [2, 4, 6, 8, 10, 15, 20]
    n        : int
    miss     : float
    repeats  : int
    plot     : bool

    Returns dict with keys 'reg' and 'clf'.
    """
    if p_values is None:
        p_values = [2, 4, 6, 8, 10, 15, 20]

    # GP excluded: at n=500 each fit takes many minutes
    reg_specs = [(nm, fn) for nm, fn in _all_reg_specs() if nm not in _GP_LABELS]
    clf_specs = [(nm, fn) for nm, fn in _all_clf_specs() if nm not in _GP_LABELS]

    all_results = {}
    for title, model_specs, task in [
        ("p-Scaling: Regression Models (all families, GP omitted)",    reg_specs, "reg"),
        ("p-Scaling: Classification Models (all families, GP omitted)", clf_specs, "clf"),
    ]:
        _banner(title, f"n={n}  miss={miss:.0%}  repeats={repeats}")

        model_names = [ms[0] for ms in model_specs]
        rows, results = [], []

        for p in p_values:
            make = _make_reg if task == "reg" else _make_clf
            X, y = make(n, p, miss)
            row  = [str(p)]
            row_r = {"p": p}
            for mname, MCls in model_specs:
                m = MCls()
                t_m, _ = _timeit(lambda m=m, X=X, y=y: m.fit(X, y), repeats)
                row.append(_fmt(t_m))
                row_r[mname] = t_m
            rows.append(row)
            results.append(row_r)

        _table(["p"] + model_names, rows)
        all_results[task] = results

        if plot:
            import matplotlib.pyplot as plt
            palette = _viridis(len(model_specs))
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for i, (mname, _) in enumerate(model_specs):
                ps = [r["p"]    for r in results]
                ts = [r[mname]  for r in results]
                ax.plot(ps, ts, marker="o", markersize=5,
                        label=mname, color=palette[i])
            ax.set_xlabel("p  (features)", fontsize=10)
            ax.set_ylabel("Fit time (s)", fontsize=10)
            ax.set_title(title + f"\n(n={n}, {miss:.0%} missing)", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
            plt.tight_layout(); plt.show()

    return all_results


# ---------------------------------------------------------------------------
# 4. Missingness impact  (all models, reg + clf)
# ---------------------------------------------------------------------------

def bench_missingness_impact(miss_rates=None, n=500, p=5, repeats=3, plot=False):
    """
    Fit time vs fraction of X cells that are NaN for ALL model families.

    MissGaussian is omitted (n=500 too slow).

    Parameters
    ----------
    miss_rates : list of float  -- default [0, 5%, 10%, 20%, 30%, 50%, 70%]
    n          : int
    p          : int
    repeats    : int
    plot       : bool

    Returns dict with keys 'reg' and 'clf'.
    """
    if miss_rates is None:
        miss_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

    reg_specs = [(nm, fn) for nm, fn in _all_reg_specs() if nm not in _GP_LABELS]
    clf_specs = [(nm, fn) for nm, fn in _all_clf_specs() if nm not in _GP_LABELS]

    all_results = {}
    for title, model_specs, task in [
        ("Missingness Impact: Regression Models (all families)",    reg_specs, "reg"),
        ("Missingness Impact: Classification Models (all families)", clf_specs, "clf"),
    ]:
        _banner(title, f"n={n}  p={p}  repeats={repeats}")

        model_names = [ms[0] for ms in model_specs]
        rows, results = [], []

        for mr in miss_rates:
            make = _make_reg if task == "reg" else _make_clf
            X, y = make(n, p, mr)
            row  = [f"{mr:.0%}"]
            row_r = {"miss_rate": mr}
            for mname, MCls in model_specs:
                m = MCls()
                t_m, _ = _timeit(lambda m=m, X=X, y=y: m.fit(X, y), repeats)
                row.append(_fmt(t_m))
                row_r[mname] = t_m
            rows.append(row)
            results.append(row_r)

        _table(["miss%"] + model_names, rows)
        all_results[task] = results

        if plot:
            import matplotlib.pyplot as plt
            palette = _viridis(len(model_specs))
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for i, (mname, _) in enumerate(model_specs):
                xs = [r["miss_rate"] * 100 for r in results]
                ts = [r[mname]             for r in results]
                ax.plot(xs, ts, marker="o", markersize=5,
                        label=mname, color=palette[i])
            ax.set_xlabel("Missing rate (%)", fontsize=10)
            ax.set_ylabel("Fit time (s)", fontsize=10)
            ax.set_title(title + f"\n(n={n}, p={p})", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
            plt.tight_layout(); plt.show()

    return all_results


# ---------------------------------------------------------------------------
# 5. Ensemble scaling  (all base estimators, reg + clf)
# ---------------------------------------------------------------------------

def bench_ensemble_scaling(n_estimators_values=None, n=400, p=5,
                            miss=0.20, repeats=2, plot=False):
    """
    MissEnsemble fit time vs n_estimators for ALL supported base estimators,
    separately for regression and classification.

    Expected: near-linear growth; slopes reveal per-estimator cost.

    Parameters
    ----------
    n_estimators_values : list of int  -- default [5, 10, 20, 50, 100]
    n                   : int
    p                   : int
    miss                : float
    repeats             : int
    plot                : bool

    Returns dict with keys 'reg' and 'clf'.
    """
    if n_estimators_values is None:
        n_estimators_values = [5, 10, 20, 50, 100]

    from MissLearn import (
        MissEnsemble,
        MissLinear, MissLogistic,
        MissRidgeRegressor, MissRidgeClassifier,
        MissLASSORegressor, MissLASSOClassifier,
        MissNeighborsRegressor, MissNeighborsClassifier,
        MissBayesRegressor, MissBayesClassifier,
        MissSupportRegressor, MissSupportClassifier,
    )

    X_r, y_r = _make_reg(n, p, miss)
    X_c, y_c = _make_clf(n, p, miss)

    base_reg = [
        ("MissLinear",          MissLinear()),
        ("MissRidge (reg)",     MissRidgeRegressor()),
        ("MissLASSO (reg)",     MissLASSORegressor()),
        ("MissNeighbors (reg)", MissNeighborsRegressor()),
        ("MissBayes (reg)",     MissBayesRegressor()),
        ("MissSupport (reg)",   MissSupportRegressor()),
    ]
    base_clf = [
        ("MissLogistic",        MissLogistic()),
        ("MissRidge (clf)",     MissRidgeClassifier()),
        ("MissLASSO (clf)",     MissLASSOClassifier()),
        ("MissNeighbors (clf)", MissNeighborsClassifier()),
        ("MissBayes (clf)",     MissBayesClassifier()),
        ("MissSupport (clf)",   MissSupportClassifier()),
    ]

    all_results = {}
    for title, base_specs, X, y, task in [
        ("MissEnsemble Scaling; Regression base estimators",     base_reg, X_r, y_r, "reg"),
        ("MissEnsemble Scaling; Classification base estimators", base_clf, X_c, y_c, "clf"),
    ]:
        _banner(title, f"n={n}  p={p}  miss={miss:.0%}  repeats={repeats}")

        base_names = [b[0] for b in base_specs]
        rows, results = [], []

        for k in n_estimators_values:
            row  = [str(k)]
            row_r = {"n_estimators": k}
            for bname, base_est in base_specs:
                ens = MissEnsemble(estimator=base_est, n_estimators=k, random_state=0)
                t_m, _ = _timeit(lambda e=ens, X=X, y=y: e.fit(X, y), repeats)
                row.append(_fmt(t_m))
                row_r[bname] = t_m
            rows.append(row)
            results.append(row_r)

        _table(["n_estimators"] + base_names, rows)
        all_results[task] = results

        if plot:
            import matplotlib.pyplot as plt
            palette = _viridis(len(base_specs))
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for i, (bname, _) in enumerate(base_specs):
                ks = [r["n_estimators"] for r in results]
                ts = [r[bname]          for r in results]
                ax.plot(ks, ts, marker="o", markersize=5,
                        label=bname, color=palette[i])
            ax.set_xlabel("n_estimators", fontsize=10)
            ax.set_ylabel("Fit time (s)", fontsize=10)
            ax.set_title(title + f"\n(n={n}, p={p})", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="upper left")
            plt.tight_layout(); plt.show()

    return all_results


# ---------------------------------------------------------------------------
# 6. MissShapley scaling
# ---------------------------------------------------------------------------

def bench_shapley_scaling(p_values=None, n_obs=100, n_train=300,
                           exact_threshold=15, repeats=2, plot=False):
    """
    MissShapley computation time vs p.

    For p <= exact_threshold the engine enumerates all 2^p coalitions;
    above the threshold it switches to KernelSHAP.  The exponential-to-linear
    transition should be clearly visible.

    Uses MissLinear as the underlying model (no complete-case requirement,
    so it works reliably across all p values with any miss rate).

    Parameters
    ----------
    p_values        : list of int -- default [2, 4, 6, 8, 10, 12, 14, 15, 16, 20, 25]
    n_obs           : int         -- observations to explain
    n_train         : int         -- training set size
    exact_threshold : int         -- matches MissShapley default (15)
    repeats         : int
    plot            : bool
    """
    if p_values is None:
        p_values = [2, 4, 6, 8, 10, 12, 14, 15, 16, 20, 25]

    from MissLearn import MissLinear, MissShapley

    _banner("MissShapley Scaling: Compute Time vs p",
            f"n_train={n_train}  n_obs={n_obs}  "
            f"exact_threshold={exact_threshold}  repeats={repeats}")

    headers = ["p", "mode", "shap_values() time", "time / obs"]
    rows, results = [], []

    for p in p_values:
        # Use miss=0.10 to keep enough signal while avoiding complete-case issues
        X_tr, y_tr = _make_reg(n_train, p, miss=0.10)
        X_ex       = _make_reg(n_obs,   p, miss=0.10)[0]
        mode       = "exact (2^p)" if p <= exact_threshold else "kernel"

        # MissLinear: pure FIML, no complete-case requirement
        model = MissLinear(compute_se=False, max_iter=2000, tol=1e-8)
        model.fit(X_tr, y_tr)
        baseline = float(model.predict(np.full((1, p), np.nan))[0])

        engine = MissShapley(model,
                             exact_threshold=exact_threshold,
                             random_state=0)

        t_m, _ = _timeit(
            lambda e=engine, X=X_ex, b=baseline: e.shap_values(X, baseline=b),
            repeats,
        )

        rows.append((str(p), mode, _fmt(t_m), _fmt(t_m / n_obs) + "/obs"))
        results.append(dict(p=p, mode=mode,
                            total_s=t_m, per_obs_s=t_m / n_obs))

    _table(headers, rows)

    if plot:
        import matplotlib.pyplot as plt
        exact_pts  = [(r["p"], r["total_s"]) for r in results
                      if r["mode"] == "exact (2^p)" and not np.isnan(r["total_s"])]
        kernel_pts = [(r["p"], r["total_s"]) for r in results
                      if r["mode"] == "kernel" and not np.isnan(r["total_s"])]
        fig, ax = plt.subplots(figsize=(8, 4))
        if exact_pts:
            ax.plot(*zip(*exact_pts),  marker="o", label="exact (2^p)",
                    color="#440154")
        if kernel_pts:
            ax.plot(*zip(*kernel_pts), marker="s", label="KernelSHAP",
                    color="#21908C", linestyle="--")
        ax.axvline(exact_threshold, color="#fde725", linestyle=":",
                   label=f"threshold p={exact_threshold}")
        ax.set_xlabel("p  (features)", fontsize=10)
        ax.set_ylabel(f"shap_values() time (s)  [{n_obs} obs]", fontsize=10)
        ax.set_title("MissShapley Scaling: Exact vs KernelSHAP")
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.show()

    return results


# ---------------------------------------------------------------------------
# 7. MissImputer scaling
# ---------------------------------------------------------------------------

def bench_imputer_scaling(m_values=None, n=400, p=5,
                           miss=0.20, repeats=2, plot=False):
    """
    MissImputer fit and transform time vs number of imputations m.

    Fit time is dominated by joint MVN estimation (constant in m).
    Transform time grows linearly with m.

    Parameters
    ----------
    m_values : list of int  -- default [5, 10, 20, 50]
    n        : int
    p        : int
    miss     : float
    repeats  : int
    plot     : bool
    """
    if m_values is None:
        m_values = [5, 10, 20, 50]

    from MissLearn import MissImputer

    X, _ = _make_reg(n, p, miss)

    _banner("MissImputer Scaling: Fit+Transform Time vs m (imputations)",
            f"n={n}  p={p}  miss={miss:.0%}  repeats={repeats}")

    headers = ["m", "fit time", "transform(1)", "per imputation"]
    rows, results = [], []

    for m in m_values:
        imp = MissImputer(m=m, random_state=0)
        t_fit, _ = _timeit(lambda i=imp, X=X: i.fit(X), repeats)
        imp.fit(X)
        t_tr, _  = _timeit(lambda i=imp, X=X: i.transform(X), repeats)
        rows.append((str(m), _fmt(t_fit), _fmt(t_tr),
                     _fmt(t_tr / m) + "/imp"))
        results.append(dict(m=m, fit_s=t_fit, transform_s=t_tr,
                            per_imp_s=t_tr / m))

    _table(headers, rows)

    if plot:
        import matplotlib.pyplot as plt
        ms   = [r["m"]           for r in results]
        fits = [r["fit_s"]       for r in results]
        trs  = [r["transform_s"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ms, fits, marker="o", label="fit",       color="#440154")
        ax.plot(ms, trs,  marker="s", label="transform", color="#21908C")
        ax.set_xlabel("m  (imputations)", fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=10)
        ax.set_title(f"MissImputer Scaling  (n={n}, p={p})")
        ax.legend()
        plt.tight_layout(); plt.show()

    return results


# ---------------------------------------------------------------------------
# 8. Cross-validation overhead
# ---------------------------------------------------------------------------

def bench_crossval_scaling(cv_values=None, n=400, p=5,
                            miss=0.20, repeats=2, plot=False):
    """
    miss_cross_val_score wall time vs number of folds cv.

    Shows near-linear overhead relative to a single fit.  A dotted line
    shows the naive cv × fit_time baseline for reference.

    Parameters
    ----------
    cv_values : list of int  -- default [3, 5, 10]
    n         : int
    p         : int
    miss      : float
    repeats   : int
    plot      : bool
    """
    if cv_values is None:
        cv_values = [3, 5, 10]

    from MissLearn import (
        MissRidgeRegressor, MissRidgeClassifier,
        miss_cross_val_score,
    )

    X_r, y_r = _make_reg(n, p, miss)
    X_c, y_c = _make_clf(n, p, miss)

    specs = [
        ("MissRidge (reg)", MissRidgeRegressor,  X_r, y_r, "r2"),
        ("MissRidge (clf)", MissRidgeClassifier, X_c, y_c, "accuracy"),
    ]

    _banner("Cross-Validation Overhead: Time vs cv",
            f"n={n}  p={p}  miss={miss:.0%}  repeats={repeats}")

    all_results = {}
    for label, ModelCls, X, y, scoring in specs:
        t_fit, _ = _timeit(lambda C=ModelCls, X=X, y=y: C().fit(X, y), repeats)
        print(f"\n  {label}  (single fit = {_fmt(t_fit)})")
        headers = ["cv", "total time", "vs 1 fit"]
        rows, results = [], []
        for cv in cv_values:
            t_cv, _ = _timeit(
                lambda C=ModelCls, X=X, y=y, cv=cv, s=scoring:
                    miss_cross_val_score(C(), X, y, cv=cv, scoring=s,
                                        random_state=0),
                repeats,
            )
            ratio = t_cv / t_fit if t_fit > 0 else float("nan")
            rows.append((str(cv), _fmt(t_cv), f"{ratio:.1f}x"))
            results.append(dict(model=label, cv=cv, total_s=t_cv,
                                fit_s=t_fit, ratio=ratio))
        _table(headers, rows)
        all_results[label] = results

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        palette = ["#440154", "#21908C"]
        for i, (label, res_list) in enumerate(all_results.items()):
            cvs    = [r["cv"]      for r in res_list]
            totals = [r["total_s"] for r in res_list]
            fit_t  = res_list[0]["fit_s"]
            ax.plot(cvs, totals, marker="o", label=label,
                    color=palette[i])
            ax.plot(cvs, [fit_t * cv for cv in cvs],
                    linestyle=":", color=palette[i], alpha=0.5)
        ax.set_xlabel("cv  (folds)", fontsize=10)
        ax.set_ylabel("Total time (s)", fontsize=10)
        ax.set_title(f"Cross-Validation Overhead  (n={n}, p={p})\n"
                     "dotted = naive cv × fit_time baseline")
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.show()

    return all_results


# ---------------------------------------------------------------------------
# 9. Gaussian Process  (O(n³))
# ---------------------------------------------------------------------------

def bench_gaussian_process(n_values=None, p=3, miss=0.15,
                            repeats=2, plot=False):
    """
    MissGaussian fit time vs n -- demonstrates O(n³) Cholesky scaling.

    Keep n <= 150 to stay tractable (default).

    Parameters
    ----------
    n_values : list of int  -- default [10, 20, 30, 50, 75, 100, 125, 150]
    p        : int
    miss     : float
    repeats  : int
    plot     : bool
    """
    if n_values is None:
        n_values = [10, 20, 30, 50, 75, 100, 125, 150]

    from MissLearn import MissGaussianRegressor, MissGaussianClassifier

    _banner("MissGaussian Scaling: Fit Time vs n  (O(n³))",
            f"p={p}  miss={miss:.0%}  repeats={repeats}")

    headers = ["n", "MissGaussian (reg)", "MissGaussian (clf)"]
    rows, results = [], []

    for n in n_values:
        X_r, y_r = _make_reg(n, p, miss)
        X_c, y_c = _make_clf(n, p, miss)
        t_r, _ = _timeit(lambda X=X_r, y=y_r: MissGaussianRegressor().fit(X, y),  repeats)
        t_c, _ = _timeit(lambda X=X_c, y=y_c: MissGaussianClassifier().fit(X, y), repeats)
        rows.append((str(n), _fmt(t_r), _fmt(t_c)))
        results.append(dict(n=n, reg_s=t_r, clf_s=t_c))

    _table(headers, rows)

    if plot:
        import matplotlib.pyplot as plt
        ns   = [r["n"]     for r in results]
        regs = [r["reg_s"] for r in results]
        clfs = [r["clf_s"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ns, regs, marker="o", label="MissGaussian (reg)", color="#440154")
        ax.plot(ns, clfs, marker="s", label="MissGaussian (clf)", color="#21908C")
        ax.set_xlabel("n  (samples)", fontsize=10)
        ax.set_ylabel("Fit time (s)", fontsize=10)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"MissGaussian O(n³) Scaling  (p={p})")
        ax.legend()
        plt.tight_layout(); plt.show()

    return results


# ---------------------------------------------------------------------------
# 10. Mixed effects scaling
# ---------------------------------------------------------------------------

def bench_mixed_effects(n_groups_values=None, obs_per_group=8, p=4,
                         miss=0.20, repeats=1, plot=False):
    """
    MissMixed fit time vs number of random-effect groups.

    Total sample size = n_groups × obs_per_group.

    Parameters
    ----------
    n_groups_values : list of int  -- default [5, 10, 20, 40]
    obs_per_group   : int
    p               : int
    miss            : float
    repeats         : int  -- default 1 (slow model)
    plot            : bool
    """
    if n_groups_values is None:
        n_groups_values = [5, 10, 20, 40]

    from MissLearn import MissMixedRegressor, MissMixedClassifier

    _banner("MissMixed Scaling: Fit Time vs n_groups",
            f"obs_per_group={obs_per_group}  p={p}  "
            f"miss={miss:.0%}  repeats={repeats}")

    headers = ["n_groups", "n_total",
               "MissMixed (reg)", "MissMixed (clf)"]
    rows, results = [], []

    for g in n_groups_values:
        n      = g * obs_per_group
        X_r, y_r = _make_reg(n, p, miss)
        X_c, y_c = _make_clf(n, p, miss)
        groups   = np.repeat(np.arange(g), obs_per_group)

        t_r, _ = _timeit(
            lambda X=X_r, y=y_r, gr=groups: MissMixedRegressor().fit(X, y, groups=gr),
            repeats,
        )
        t_c, _ = _timeit(
            lambda X=X_c, y=y_c, gr=groups: MissMixedClassifier().fit(X, y, groups=gr),
            repeats,
        )
        rows.append((str(g), str(n), _fmt(t_r), _fmt(t_c)))
        results.append(dict(n_groups=g, n_total=n, reg_s=t_r, clf_s=t_c))

    _table(headers, rows)

    if plot:
        import matplotlib.pyplot as plt
        gs   = [r["n_groups"] for r in results]
        regs = [r["reg_s"]    for r in results]
        clfs = [r["clf_s"]    for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gs, regs, marker="o", label="MissMixed (reg)", color="#440154")
        ax.plot(gs, clfs, marker="s", label="MissMixed (clf)", color="#21908C")
        ax.set_xlabel("n_groups", fontsize=10)
        ax.set_ylabel("Fit time (s)", fontsize=10)
        ax.set_title(f"MissMixed Scaling  (obs/group={obs_per_group}, p={p})")
        ax.legend()
        plt.tight_layout(); plt.show()

    return results


# ---------------------------------------------------------------------------
# 11. Sensitivity analysis scaling
# ---------------------------------------------------------------------------

def bench_sensitivity(n_delta_values=None, n=300, p=4,
                       miss=0.20, repeats=2, plot=False):
    """
    MissSensitivity fit time vs delta-grid size (n_delta).

    Each delta value re-fits the full FIML model, so wall time grows
    linearly with n_delta.

    Parameters
    ----------
    n_delta_values : list of int  -- default [5, 10, 20, 40, 80]
    n              : int
    p              : int
    miss           : float
    repeats        : int
    plot           : bool
    """
    if n_delta_values is None:
        n_delta_values = [5, 10, 20, 40, 80]

    from MissLearn import MissSensitivity, MissLinear

    X, y = _make_reg(n, p, miss)
    model = MissLinear()
    model.fit(X, y)

    _banner("MissSensitivity Scaling: Fit Time vs n_delta",
            f"n={n}  p={p}  miss={miss:.0%}  repeats={repeats}")

    headers = ["n_delta", "fit time", "time / delta"]
    rows, results = [], []

    for nd in n_delta_values:
        sens = MissSensitivity(model, n_delta=nd)
        t_m, _ = _timeit(lambda s=sens, X=X, y=y: s.fit(X, y), repeats)
        rows.append((str(nd), _fmt(t_m), _fmt(t_m / nd) + "/delta"))
        results.append(dict(n_delta=nd, total_s=t_m, per_delta_s=t_m / nd))

    _table(headers, rows)

    if plot:
        import matplotlib.pyplot as plt
        nds = [r["n_delta"] for r in results]
        ts  = [r["total_s"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(nds, ts, marker="o", color="#440154", label="total fit time")
        ax.set_xlabel("n_delta", fontsize=10)
        ax.set_ylabel("Fit time (s)", fontsize=10)
        ax.set_title(f"MissSensitivity Scaling  (n={n}, p={p})")
        ax.legend()
        plt.tight_layout(); plt.show()

    return results
