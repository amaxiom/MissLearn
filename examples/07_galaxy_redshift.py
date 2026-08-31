# -*- coding: utf-8 -*-
"""Photometric redshift from an incomplete multiwavelength catalogue
=================================================================


A head-to-head against a published missing-data study on its own data and
its own protocol, plus the case that study explicitly defers.

The study
---------
Luken, Padhy and Wang, "Missing Data Imputation for Galaxy Redshift
Estimation" (arXiv:2111.13806). They benchmark mean, median, minimum,
maximum, kNN, MICE and GAIN imputation, then estimate redshift with kNN
regression, and report MICE as the best imputer.

Their protocol, reproduced here
-------------------------------
* 1,311 radio sources from ATLAS DR3, cross-matched to DES optical and
  Spitzer/SWIRE infrared photometry, with spectroscopic redshifts.
* Nine predictors: 1.4 GHz integrated radio flux, g/r/i/z magnitudes, and
  3.6/4.5/5.8/8.0 micron infrared fluxes.
* All predictors standardised to N(0, 1) on the training set.
* 70/30 train/test split.
* Missingness injected into the **test set only**, at 2, 5, 10, 15, 20, 25
  and 30 percent, and repeated over many seeds.

That last point is worth dwelling on, because it is unusual and it decides
what the comparison actually measures. The model is trained on complete data
and must predict from incomplete inputs. An imputer therefore has to guess a
missing band from the test row alone, whereas a marginalising model can
integrate over the joint distribution it estimated during training.

Why the comparison here is fair
-------------------------------
Every arm uses the **same model class**, k-nearest-neighbours regression with
the same k. The conventional arms impute and then run kNN; the MissLearn arm
is `MissNeighborsRegressor`, which is kNN on the expected distance under a
fitted joint Gaussian. Only the missing-data treatment differs, so a gap
between arms is attributable to that and not to model capacity.

Two honest deviations from the published setup, both of which matter when
reading the results: they used a Mahalanobis distance metric and tuned k by
inner cross-validation, while this script uses Euclidean distance on
standardised predictors with a fixed k; and their MICE and GAIN arms came
from specific third-party implementations, whereas this uses scikit-learn's
IterativeImputer and omits GAIN entirely. The ordering among the conventional
arms should therefore be read as a property of this reproduction rather than
as a restatement of their result.

The case the study defers
-------------------------
Their paper states plainly that it "only looks at the case where data is
missing at random", and that "future work will look at the case where
astronomical sources are too faint to be seen at particular wavelengths".
That deferred case is MNAR, and it is the physically realistic one: a band is
absent precisely because the source is faint in it, and faintness correlates
with distance. This script implements it as a second mechanism, because it is
the regime in which the choice of missing-data method should matter most.

Run
---
    python 07_galaxy_redshift.py
    python 07_galaxy_redshift.py --quick
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer      # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import MissLearn as ML                                          # noqa: E402

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.titlesize": 15, "figure.constrained_layout.use": True,
    "axes.titlepad": 10, "savefig.bbox": "tight", "figure.dpi": 110,
})

CSV = os.path.join(_HERE, "example_data", "atlas_redshift_clean.csv")
RAW = os.path.join(_HERE, "example_data", "ATLAS_complete_DR2.fits")
FITS_URL = ("https://raw.githubusercontent.com/kluken/Redshift-kNN-2021/"
            "main/ATLAS_complete_DR2.fits")

RADIO = ['S']
OPTICAL = ['MAG_APER_4_G', 'MAG_APER_4_R', 'MAG_APER_4_I', 'MAG_APER_4_Z']
INFRARED = ['flux_ap2_36', 'flux_ap2_45', 'flux_ap2_58', 'flux_ap2_80']
FEATURES = RADIO + OPTICAL + INFRARED
PRETTY = {'S': 'radio 1.4 GHz',
          'MAG_APER_4_G': 'g', 'MAG_APER_4_R': 'r',
          'MAG_APER_4_I': 'i', 'MAG_APER_4_Z': 'z-band',
          'flux_ap2_36': '3.6 um', 'flux_ap2_45': '4.5 um',
          'flux_ap2_58': '5.8 um', 'flux_ap2_80': '8.0 um'}

RATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ARMS = ['Mean', 'Median', 'Min', 'kNN', 'MICE', 'FIML']
K = 11
OUTLIER_THRESHOLD = 0.15     # the standard photometric-redshift definition


def banner(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


# ===========================================================================
# Data
# ===========================================================================

def load():
    """Read the cached catalogue, building it from the FITS file if needed."""
    if os.path.exists(CSV):
        return pd.read_csv(CSV)

    if not os.path.exists(RAW):
        import urllib.request
        print("Downloading the catalogue from the authors' repository ...")
        urllib.request.urlretrieve(FITS_URL, RAW)

    try:
        from astropy.table import Table
    except ImportError:
        raise SystemExit("astropy is needed once, to convert the FITS "
                         "catalogue to CSV: pip install astropy")
    df = Table.read(RAW).to_pandas()
    sub = df[FEATURES + ['z']].replace(-999, np.nan).dropna()
    sub = sub.reset_index(drop=True)
    sub.to_csv(CSV, index=False)
    print("Cached %d sources at %s" % (len(sub), CSV))
    return sub


# ===========================================================================
# Metrics and mechanisms
# ===========================================================================

def metrics(z_true, z_pred):
    """RMSE, and the outlier rate convention used in photometric redshift work.

    A source counts as an outlier when |dz| / (1 + z) exceeds 0.15. The
    (1 + z) denominator matters: a fixed absolute tolerance would be far
    stricter for nearby galaxies than for distant ones.
    """
    rmse = float(np.sqrt(np.mean((z_pred - z_true) ** 2)))
    dz = np.abs(z_pred - z_true) / (1.0 + z_true)
    return rmse, float((dz > OUTLIER_THRESHOLD).mean())


def inject(Xs, rate, rng, mechanism):
    """Return a boolean mask of entries to blank.

    'mcar' reproduces the published injection: each cell is dropped
    independently with probability `rate`.

    'mnar' models a non-detection: a band is dropped preferentially when the
    source is faint in it. Magnitudes run backwards (larger is fainter) while
    fluxes do not, so faintness is established by rank within each column and
    the drop probability rises linearly with it, holding the mean at `rate`.
    """
    if mechanism == 'mcar':
        return rng.random(Xs.shape) < rate

    M = np.zeros(Xs.shape, dtype=bool)
    for j in range(Xs.shape[1]):
        col = Xs[:, j]
        faint = col if FEATURES[j].startswith('MAG') else -col
        r = np.argsort(np.argsort(faint)) / max(1, len(col) - 1)
        M[:, j] = rng.random(len(col)) < np.clip(2.0 * rate * r, 0.0, 1.0)
    return M


def run_seed(X_all, z_all, rate, seed, mechanism):
    """One split at one rate: every arm, same folds, same standardisation."""
    Xtr, Xte, ztr, zte = train_test_split(X_all, z_all, test_size=0.30,
                                          random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    rng = np.random.default_rng(seed)
    Xte_m = Xte_s.copy()
    Xte_m[inject(Xte_s, rate, rng, mechanism)] = np.nan

    out = {}
    knn = KNeighborsRegressor(n_neighbors=K).fit(Xtr_s, ztr)
    for name in ['Mean', 'Median', 'Min', 'kNN', 'MICE']:
        if name == 'Min':
            # Imputing the training minimum is the astronomer's instinct for a
            # non-detection: treat it as an upper limit on the brightness.
            Xi = np.where(np.isnan(Xte_m), np.nanmin(Xtr_s, axis=0), Xte_m)
        else:
            imp = {'Mean': SimpleImputer(strategy='mean'),
                   'Median': SimpleImputer(strategy='median'),
                   'kNN': KNNImputer(n_neighbors=5),
                   'MICE': IterativeImputer(max_iter=10,
                                            random_state=0)}[name].fit(Xtr_s)
            Xi = imp.transform(Xte_m)
        out[name] = metrics(zte, knn.predict(Xi))

    out['FIML'] = metrics(zte, ML.MissNeighborsRegressor(copula=False, n_neighbors=K)
                          .fit(Xtr_s, ztr).predict(Xte_m))
    return out


# ===========================================================================
# The sweep
# ===========================================================================

def sweep(X_all, z_all, mechanism, n_seeds):
    store = {}
    for rate in RATES:
        acc = {a: [] for a in ARMS}
        for s in range(n_seeds):
            for arm, vals in run_seed(X_all, z_all, rate, s, mechanism).items():
                acc[arm].append(vals)
        store[rate] = {a: np.array(v) for a, v in acc.items()}
        print("    %.0f%% done" % (100 * rate), end="", flush=True)
    print()
    return store


def report(store, mechanism, n_seeds):
    label = ("MCAR, as the published study injects it" if mechanism == 'mcar'
             else "MNAR non-detection, the case the study defers")
    banner("Mechanism: %s" % label)
    for k, title, fmt in ((0, "RMSE", "%11.4f"),
                          (1, "Outlier rate, |dz|/(1+z) > 0.15", "%11.3f")):
        print("  %s, mean over %d seeds" % (title, n_seeds))
        print("  %-6s" % "rate" + "".join("%11s" % a for a in ARMS))
        print("  " + "-" * 74)
        for rate in RATES:
            print("  %-6s" % ("%.0f%%" % (100 * rate))
                  + "".join(fmt % store[rate][a][:, k].mean() for a in ARMS))
        print()

    best_conv = min((a for a in ARMS if a != 'FIML'),
                    key=lambda a: store[RATES[-1]][a][:, 0].mean())
    f = store[RATES[-1]]['FIML'][:, 0].mean()
    b = store[RATES[-1]][best_conv][:, 0].mean()
    print("  At %.0f%%: FIML RMSE %.4f, best conventional %s %.4f (%+.4f)"
          % (100 * RATES[-1], f, best_conv, b, f - b))
    mice = store[RATES[-1]]['MICE'][:, 0].mean()
    print("  MICE, the method the study reports as best, is %.4f here (%+.4f "
          "against FIML)." % (mice, mice - f))


def plot(stores, n_seeds):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)
    cmap = plt.colormaps['viridis']
    cols = {a: cmap(v) for a, v in
            zip(ARMS, np.linspace(0.05, 0.92, len(ARMS)))}
    xs = [100 * r for r in RATES]

    for row, (mech, store) in enumerate(stores.items()):
        for col, (k, ylab) in enumerate(((0, "RMSE"),
                                         (1, r"outlier rate"))):
            ax = axes[row][col]
            for a in ARMS:
                m = [store[r][a][:, k].mean() for r in RATES]
                sd = [store[r][a][:, k].std() for r in RATES]
                ax.plot(xs, m, marker='o', markersize=4.5, color=cols[a],
                        linewidth=2.2 if a == 'FIML' else 1.4,
                        linestyle='-' if a == 'FIML' else '--', label=a)
                ax.fill_between(xs, np.array(m) - np.array(sd),
                                np.array(m) + np.array(sd),
                                color=cols[a], alpha=0.10, linewidth=0)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_title("%s, %s" % (mech.upper(), ylab))
    for ax in axes[1]:
        ax.set_xlabel("missing rate injected into the test set (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(ARMS),
               frameon=False)
    fig.suptitle("Photometric redshift from incomplete photometry: "
                 "same kNN model class in every arm\n"
                 "(%d seeds per rate; solid line is FIML)" % n_seeds)
    plt.show()


# ===========================================================================
# Which band is worth observing?
# ===========================================================================

def explain(X_all, z_all, seed=0):
    banner("Which band is worth the telescope time?")
    print("  Missingness SHAP asks how much the prediction moves when a band")
    print("  is observed rather than left unknown. That is a survey-design")
    print("  question, and it is not answerable from ordinary feature")
    print("  attribution, which only ranks the values already in hand.\n")

    sc = StandardScaler().fit(X_all)
    Xs = sc.transform(X_all)
    model = ML.MissLinear(copula=False, compute_se=False).fit(Xs, z_all)
    expl = ML.MissExplainer(model, random_state=seed).fit(
        Xs, feature_names=[PRETTY[f] for f in FEATURES])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(250, len(Xs)), replace=False)
    ms = np.abs(expl.miss_shap(Xs[idx])).mean(axis=0)
    vs = np.abs(expl.shap_values(Xs[idx])).mean(axis=0)

    print("  %-16s %14s %16s" % ("band", "value SHAP", "missingness SHAP"))
    print("  " + "-" * 48)
    for j in np.argsort(-ms):
        print("  %-16s %14.4f %16.4f" % (PRETTY[FEATURES[j]], vs[j], ms[j]))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    cmap = plt.colormaps['viridis']
    o = np.argsort(ms)
    ax.barh([PRETTY[FEATURES[j]] for j in o], ms[o],
            color=[cmap(t) for t in np.linspace(0.15, 0.85, len(o))])
    ax.set_xlabel("mean |missingness SHAP|")
    ax.set_title("Value of observing each band at all")
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="5 seeds, not 20")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    n_seeds = a.seeds if a.seeds else (5 if a.quick else 20)

    df = load()
    X_all = df[FEATURES].to_numpy(float)
    z_all = df['z'].to_numpy(float)

    banner("ATLAS DR3 radio sources with DES and Spitzer photometry")
    print("  sources           : %d" % len(df))
    print("  predictors        : %d (%s)"
          % (len(FEATURES), ", ".join(PRETTY[f] for f in FEATURES)))
    print("  spectroscopic z   : %.4f to %.4f, median %.4f"
          % (z_all.min(), z_all.max(), np.median(z_all)))
    print("  sources at z > 1  : %d (%.1f%%)"
          % (int((z_all > 1).sum()), 100 * (z_all > 1).mean()))
    print()
    print("  All nine predictors are complete in this catalogue, which is why")
    print("  the study injects missingness rather than finding it.")

    stores = {}
    for mech in ('mcar', 'mnar'):
        print("\n  sweeping %s:" % mech.upper(), end=" ", flush=True)
        stores[mech] = sweep(X_all, z_all, mech, n_seeds)
        report(stores[mech], mech, n_seeds)

    plot(stores, n_seeds)
    explain(X_all, z_all)
    banner("Done")


if __name__ == "__main__":
    main()
