"""
prepare_datasets.py
===================
Download and cache the seven benchmark datasets used by benchmark_test_suite.py.

Run once before benchmarking:

    python prepare_datasets.py

Each dataset is saved to tests/benchmark_data/ as a CSV file.  Subsequent
calls skip files that already exist.  Pass --force to re-download everything.

Datasets
--------
Tier 1 -- complete datasets (missingness injected synthetically)
  energy_efficiency.csv   UCI ENB2012  n=768  p=8   regression
  wisconsin.csv           UCI WDBC     n=569  p=10  binary classification
  iris.csv                UCI Iris     n=150  p=4   3-class classification

Tier 2 -- real missing data
  auto_mpg.csv            UCI          n=392  p=7   regression

Tier 3 -- mechanism sweep
  esol.csv                Delaney      n=1128 p=6   regression

Tier 4 -- grouped / longitudinal (MissMixed benchmark)
  radon.csv               Gelman       n=919  p=2   regression  85 counties

All data are in the public domain or freely redistributable under CC-BY.
References are in the individual loader docstrings.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _csv_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch_text(url: str, timeout: int = 60) -> io.StringIO:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return io.StringIO(resp.read().decode("utf-8", errors="replace"))


def _fetch_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Tier 1: Energy Efficiency
# ---------------------------------------------------------------------------

def prepare_energy_efficiency(force: bool = False) -> str:
    """
    Energy Efficiency -- UCI ML Repository  (complete, regression).

    Reference
    ---------
    Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy
    performance of residential buildings using statistical machine learning tools.
    Energy and Buildings, 49, 560-567.
    https://doi.org/10.1016/j.enbuild.2012.03.003

    Target   : Heating Load (Y1, kWh/m2). Cooling Load (Y2) dropped.
    Features : 8 building attributes (relative compactness ... glazing area
               distribution).
    n        : 768  (no missing values in original).
    """
    dest = _csv_path("energy_efficiency.csv")
    if os.path.exists(dest) and not force:
        print(f"  energy_efficiency.csv  already exists; skipping")
        return dest

    # Ensure xlrd or openpyxl is available for pd.read_excel
    import importlib
    for pkg in ("openpyxl", "xlrd"):
        if importlib.util.find_spec(pkg) is not None:
            break
    else:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "openpyxl"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        importlib.invalidate_caches()

    URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00242/ENB2012_data.xlsx"
    )
    feat_names = [
        "relative_compactness", "surface_area", "wall_area", "roof_area",
        "overall_height", "orientation", "glazing_area",
        "glazing_area_distribution",
    ]
    print("  Downloading Energy Efficiency (XLSX) ...", end=" ", flush=True)
    raw = _fetch_bytes(URL)
    df  = pd.read_excel(io.BytesIO(raw)).dropna()
    y   = df.iloc[:, 8].to_numpy(dtype=np.float64)   # Y1 = Heating Load
    Xc  = df.iloc[:, :8].to_numpy(dtype=np.float64)
    out = pd.DataFrame(Xc, columns=feat_names)
    out["heating_load"] = y
    out.to_csv(dest, index=False)
    print(f"saved ({len(out)} rows)")
    return dest


# ---------------------------------------------------------------------------
# Tier 1: Breast Cancer Wisconsin
# ---------------------------------------------------------------------------

def prepare_wisconsin(force: bool = False) -> str:
    """
    Breast Cancer Wisconsin (Diagnostic) -- UCI / sklearn.

    Reference
    ---------
    Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature
    extraction for breast tumor diagnosis. IS&T/SPIE Symposium on Electronic
    Imaging, 1905, 861-870.
    Wolberg, W., Street, N., & Mangasarian, O. (1993). Breast Cancer Wisconsin
    (Diagnostic) [Dataset]. UCI ML Repository.
    https://doi.org/10.24432/C5DW2B

    Target   : diagnosis (0=benign, 1=malignant).
    Features : 10 mean nuclear features (radius, texture, perimeter, area,
               smoothness, compactness, concavity, concave_points, symmetry,
               fractal_dimension).
    n        : 569  (no missing values in original).
    """
    dest = _csv_path("wisconsin.csv")
    if os.path.exists(dest) and not force:
        print(f"  wisconsin.csv           already exists; skipping")
        return dest

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    feat_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dim_mean",
    ]
    # sklearn returns 30 features; we take the first 10 (mean group only)
    df = pd.DataFrame(data.data[:, :10], columns=feat_names)
    df["diagnosis"] = (data.target == 0).astype(int)  # sklearn: 0=malignant,1=benign
    # Recode: 1=malignant (positive), 0=benign (negative)
    df["diagnosis"] = 1 - df["diagnosis"]
    df.to_csv(dest, index=False)
    print(f"  wisconsin.csv           saved ({len(df)} rows)")
    return dest


# ---------------------------------------------------------------------------
# Tier 1: Iris
# ---------------------------------------------------------------------------

def prepare_iris(force: bool = False) -> str:
    """
    Iris -- Fisher (1936) / UCI.

    Reference
    ---------
    Fisher, R.A. (1936). The use of multiple measurements in taxonomic
    problems. Annals of Eugenics, 7(2), 179-188.
    https://doi.org/10.1111/j.1469-1809.1936.tb02137.x
    UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/iris

    Target   : species (0=setosa, 1=versicolor, 2=virginica).
    Features : sepal_length, sepal_width, petal_length, petal_width  (p=4).
    n        : 150  (no missing values).
    """
    dest = _csv_path("iris.csv")
    if os.path.exists(dest) and not force:
        print(f"  iris.csv                already exists; skipping")
        return dest

    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(
        data.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    df["species"] = data.target.astype(int)
    df.to_csv(dest, index=False)
    print(f"  iris.csv                saved ({len(df)} rows)")
    return dest


# ---------------------------------------------------------------------------
# Tier 2: Horse Colic
# ---------------------------------------------------------------------------

def prepare_horse_colic(force: bool = False) -> str:
    """
    Horse Colic -- UCI ML Repository  (real missing data, classification).

    Reference
    ---------
    McConaghy, M.T. (1989). Horse colic database.
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/ml/datasets/Horse+Colic

    Target   : outcome binarized: 0=lived, 1=died or euthanized.
    Features : 20 clinical measurements (surgery indicator, age, rectal
               temperature, pulse, respiratory rate, temperatures, mucous
               membranes, capillary refill time, pain level, peristalsis,
               abdominal distension, nasogastric tube/reflux/pH, rectal
               exam, abdomen, packed cell volume, total protein,
               abdominocentesis appearance/protein).
               Hospital number (ID) and lesion-type columns are dropped.
    n        : up to 368 rows with observed outcome.
    Missing  : real clinical missingness (~24% overall).
    """
    dest = _csv_path("horse_colic.csv")
    if os.path.exists(dest) and not force:
        print(f"  horse_colic.csv         already exists; skipping")
        return dest

    URLS = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "horse-colic/horse-colic.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "horse-colic/horse-colic.test",
    ]
    feat_names = [
        "surgery", "age", "rectal_temp", "pulse", "respiratory_rate",
        "temp_extremities", "peripheral_pulse", "mucous_membranes",
        "cap_refill_time", "pain", "peristalsis", "abdom_distension",
        "nasogastric_tube", "nasogastric_reflux", "nasogastric_reflux_ph",
        "rectal_exam_feces", "abdomen", "packed_cell_volume",
        "total_protein", "abdominocentesis_appearance", "abdomcent_protein",
    ]

    print("  Downloading Horse Colic (train+test) ...", end=" ", flush=True)
    frames = []
    for url in URLS:
        try:
            raw = _fetch_text(url)
            df_part = pd.read_csv(raw, sep=r"\s+", header=None, na_values=["?"])
            frames.append(df_part)
        except Exception as exc:
            print(f"\n    WARNING: {url} failed: {exc}")

    if not frames:
        raise RuntimeError("Horse Colic: all download URLs failed.")

    df = pd.concat(frames, ignore_index=True)
    # Column layout (0-indexed):
    #  0=surgery, 1=age, 2=hospital_number(drop), 3=rectal_temp, ..., 22=outcome
    #  23=surgical_lesion, 24-26=lesion type/subtype/code, 27=cp_data
    # Build feature columns: indices 0,1 then 3-21 (skip hospital_number at 2)
    feat_cols = list(range(0, 2)) + list(range(3, 22))  # 20 features
    target_col = 22  # outcome: 1=lived, 2=died, 3=euthanized

    Xdf = df.iloc[:, feat_cols].copy()
    Xdf.columns = feat_names
    y_raw = df.iloc[:, target_col]

    # Drop rows with missing outcome
    valid = y_raw.notna() & y_raw.isin([1.0, 2.0, 3.0])
    Xdf  = Xdf.loc[valid].reset_index(drop=True)
    y    = y_raw.loc[valid].astype(float).reset_index(drop=True)
    # Binarize: 1=lived -> 0, 2=died or 3=euthanized -> 1
    y_bin = (y != 1.0).astype(int)

    Xdf["outcome"] = y_bin
    Xdf.to_csv(dest, index=False)
    print(f"saved ({len(Xdf)} rows, {Xdf.isna().mean().mean():.1%} missing)")
    return dest


# ---------------------------------------------------------------------------
# Tier 2: Auto MPG
# ---------------------------------------------------------------------------

def prepare_auto_mpg(force: bool = False) -> str:
    """
    Auto MPG -- UCI ML Repository  (real missing data, regression).

    Reference
    ---------
    Quinlan, J.R. (1993). Combining Instance-Based and Model-Based Learning.
    Proceedings of the 10th International Conference on Machine Learning,
    236-243.
    UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/auto+mpg

    Target   : fuel consumption in miles per gallon (mpg, continuous).
    Features : cylinders, displacement, horsepower (6 missing values),
               weight, acceleration, model_year, origin  (p=7).
    n        : 392 rows with complete target (6 rows with missing horsepower
               retained; car_name string column dropped).
    Missing  : 1.5% natural missing (horsepower column only).
    """
    dest = _csv_path("auto_mpg.csv")
    if os.path.exists(dest) and not force:
        print(f"  auto_mpg.csv            already exists; skipping")
        return dest

    URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "auto-mpg/auto-mpg.data"
    )
    feat_names = [
        "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin",
    ]
    col_names = ["mpg"] + feat_names + ["car_name"]
    print("  Downloading Auto MPG ...", end=" ", flush=True)
    df = pd.read_csv(
        _fetch_text(URL),
        sep=r"\s+", header=None, names=col_names,
        na_values=["?"],
    )
    # Drop rows with missing target (mpg)
    df = df[df["mpg"].notna()].copy()
    # Drop car_name (string)
    df = df.drop(columns=["car_name"])
    # Ensure numeric
    for col in feat_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reset_index(drop=True)
    df.to_csv(dest, index=False)
    miss_pct = df[feat_names].isna().mean().mean()
    print(f"saved ({len(df)} rows, {miss_pct:.1%} missing in X)")
    return dest


# ---------------------------------------------------------------------------
# Tier 3: ESOL (Delaney aqueous solubility)
# ---------------------------------------------------------------------------

def prepare_esol(force: bool = False) -> str:
    """
    ESOL (Delaney aqueous solubility) -- DeepChem / MoleculeNet.

    Reference
    ---------
    Delaney, J.S. (2004). ESOL: Estimating Aqueous Solubility Directly from
    Molecular Structure. Journal of Chemical Information and Computer Sciences,
    44(3), 1000-1005.
    https://doi.org/10.1021/ci034243x
    DeepChem processed version: https://github.com/deepchem/deepchem

    Target   : measured log solubility in mols/L (continuous, regression).
    Features : 6 molecular descriptors -- minimum degree, molecular weight,
               number of H-bond donors, number of rings, number of rotatable
               bonds, polar surface area.
               (ESOL-predicted column and Compound ID are dropped.)
    n        : 1128  (complete; no missing values in original).
    """
    dest = _csv_path("esol.csv")
    if os.path.exists(dest) and not force:
        print(f"  esol.csv                already exists; skipping")
        return dest

    # Primary: MoleculeNet / DeepChem S3 bucket (stable, maintained).
    # Fallbacks: known GitHub mirrors.  Column name varies across versions -
    # the target column is matched by prefix ("measured log solubility") and
    # then normalised to the canonical name used throughout this codebase.
    URLS = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "https://raw.githubusercontent.com/deepchem/deepchem/master/"
        "deepchem/feat/tests/assets/delaney-processed.csv",
    ]
    feat_names = [
        "min_degree", "mol_weight", "n_hbond_donors",
        "n_rings", "n_rotatable_bonds", "polar_surface_area",
    ]
    # Canonical target name written to the cached CSV.
    target_col = "measured log solubility in mols/L"
    # Prefix used to locate the target regardless of exact column name variant
    # (e.g. "mols/L" vs "mols per litre" across DeepChem versions).
    target_prefix = "measured log solubility"

    src_feat_cols = [
        "Minimum Degree", "Molecular Weight", "Number of H-Bond Donors",
        "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area",
    ]

    print("  Downloading ESOL (Delaney) ...", end=" ", flush=True)
    df = None
    found_target_col = None
    last_exc = None
    last_col_mismatch = None
    for url in URLS:
        try:
            raw = _fetch_text(url, timeout=30)
            candidate = pd.read_csv(raw)
            # Accept any column whose name starts with the canonical prefix
            matches = [c for c in candidate.columns
                       if c.lower().startswith(target_prefix.lower())]
            if matches and all(c in candidate.columns for c in src_feat_cols):
                df = candidate
                found_target_col = matches[0]
                break
            else:
                last_col_mismatch = (
                    f"target prefix '{target_prefix}' not found or feature columns "
                    f"missing in {url}; got columns: {list(candidate.columns)}"
                )
        except Exception as exc:
            last_exc = exc

    if df is None:
        detail = last_exc if last_exc is not None else last_col_mismatch
        raise RuntimeError(
            f"ESOL: could not obtain a valid dataset from any URL.  "
            f"Last issue: {detail}\n"
            "Please download delaney-processed.csv manually from\n"
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv\n"
            f"and place it in {DATA_DIR}"
        )

    # Keep only the 6 molecular descriptor features and the target;
    # normalise the target column name to the canonical form.
    missing_cols = [c for c in src_feat_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"ESOL CSV missing expected feature columns: {missing_cols}\n"
            f"Found: {list(df.columns)}"
        )

    Xdf = df[src_feat_cols].copy()
    Xdf.columns = feat_names
    Xdf[target_col] = df[found_target_col].to_numpy(dtype=np.float64)
    Xdf = Xdf.dropna().reset_index(drop=True)
    Xdf.to_csv(dest, index=False)
    print(f"saved ({len(Xdf)} rows)")
    return dest


# ---------------------------------------------------------------------------
# Tier 4: Radon (Gelman & Hill multilevel benchmark)
# ---------------------------------------------------------------------------

def prepare_radon(force: bool = False) -> str:
    """
    Radon -- Gelman & Hill multilevel modeling benchmark  (grouped, regression).

    Reference
    ---------
    Gelman, A. & Hill, J. (2007). Data Analysis Using Regression and
    Multilevel/Hierarchical Models. Cambridge University Press.
    Original data from the US EPA National Residential Radon Survey (1988-1992).
    Teaching dataset: https://github.com/avehtari/ROS-Examples

    Target   : log radon level = log(activity + 0.1), continuous regression.
    Features : floor (0=basement, 1=first floor), log_uranium (county-level
               uranium concentration, log-transformed).  p=2.
    Groups   : county (85 Minnesota counties), stored as integer codes in
               column ``county_id``.  Original county names in ``county_name``.
    n        : 919 observations (complete; no missing values in original).
               Missingness is injected synthetically by the benchmark.

    Note: this is the canonical dataset for random-intercept mixed-effects
    models.  The county random intercept captures unmeasured geological
    variation.  The floor predictor is individual-level; log_uranium is
    group-level (constant within county).
    """
    dest = _csv_path("radon.csv")
    if os.path.exists(dest) and not force:
        print(f"  radon.csv               already exists; skipping")
        return dest

    URLS = [
        # PyMC examples repo; full 25-column dataset, verified working
        "https://raw.githubusercontent.com/pymc-devs/pymc-examples/"
        "main/examples/data/radon.csv",
        # PyMC4 repo; same data, alternative mirror
        "https://raw.githubusercontent.com/pymc-devs/pymc4/"
        "master/notebooks/data/radon.csv",
        # Minimal 4-column version (floor, county, log_radon, log_uranium)
        "https://raw.githubusercontent.com/roualdes/data/master/radon.csv",
    ]
    print("  Downloading Radon (Gelman multilevel) ...", end=" ", flush=True)
    df = None
    last_exc = None
    for url in URLS:
        try:
            raw = _fetch_text(url, timeout=30)
            df = pd.read_csv(raw)
            # Minimal column check
            if "floor" in df.columns or "activity" in df.columns:
                break
            df = None
        except Exception as exc:
            last_exc = exc
            df = None

    if df is None:
        raise RuntimeError(
            f"Radon: all download URLs failed.  Last error: {last_exc}\n"
            "Please download radon.csv from https://github.com/avehtari/ROS-Examples"
            f" and place it in {DATA_DIR}"
        )

    # Normalise column names (different mirrors use different capitalisation)
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify columns; handle both raw (activity/uppm) and pre-logged variants
    activity_col    = next((c for c in df.columns if c in ("activity", "radon")), None)
    log_radon_col   = next((c for c in df.columns if c == "log_radon"), None)
    floor_col       = next((c for c in df.columns if "floor" in c), None)
    uranium_col     = next((c for c in df.columns if "uppm" in c
                            or (c == "uranium")), None)
    log_uranium_col = next((c for c in df.columns if c == "log_uranium"), None)
    county_col      = next((c for c in df.columns if "county" in c
                            and "fips" not in c and c != "county_id"), None)

    # Require either raw or logged radon; same for uranium
    has_radon   = activity_col is not None or log_radon_col is not None
    has_uranium = uranium_col  is not None or log_uranium_col is not None

    missing = []
    if not has_radon:   missing.append("activity/radon/log_radon")
    if floor_col is None: missing.append("floor")
    if not has_uranium: missing.append("uppm/uranium/log_uranium")
    if county_col is None: missing.append("county")
    if missing:
        raise ValueError(
            f"Radon CSV missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Build clean output
    out = pd.DataFrame()
    out["floor"] = pd.to_numeric(df[floor_col], errors="coerce")
    if log_uranium_col is not None:
        out["log_uranium"] = pd.to_numeric(df[log_uranium_col], errors="coerce")
    else:
        raw_uranium        = pd.to_numeric(df[uranium_col], errors="coerce")
        out["log_uranium"] = np.log(raw_uranium.clip(lower=1e-6))
    if log_radon_col is not None:
        out["log_radon"] = pd.to_numeric(df[log_radon_col], errors="coerce")
    else:
        raw_activity     = pd.to_numeric(df[activity_col], errors="coerce")
        out["log_radon"] = np.log(raw_activity.clip(lower=0.0) + 0.1)
    # Encode county as integer id; save name for reference
    county_names        = df[county_col].astype(str).str.strip()
    county_cats         = county_names.astype("category")
    out["county_id"]    = county_cats.cat.codes.astype(int)
    out["county_name"]  = county_names

    out = out.dropna(subset=["floor", "log_uranium", "log_radon"]).reset_index(drop=True)
    out.to_csv(dest, index=False)
    n_groups = out["county_id"].nunique()
    print(f"saved ({len(out)} rows, {n_groups} counties)")
    return dest


# ---------------------------------------------------------------------------
# Prepare all
# ---------------------------------------------------------------------------

def prepare_all(force: bool = False) -> None:
    """Download and cache all seven benchmark datasets."""
    _ensure_data_dir()
    print(f"Saving datasets to: {DATA_DIR}\n")
    results = {}
    for name, fn in [
        ("Energy Efficiency", prepare_energy_efficiency),
        ("Wisconsin",         prepare_wisconsin),
        ("Iris",              prepare_iris),
        ("Auto MPG",          prepare_auto_mpg),
        ("ESOL",              prepare_esol),
        ("Radon",             prepare_radon),
    ]:
        try:
            fn(force=force)
            results[name] = "OK"
        except Exception as exc:
            results[name] = f"FAILED: {exc}"
            print(f"  ERROR: {name}: {exc}")

    print("\nSummary:")
    for name, status in results.items():
        print(f"  {name:<22} {status}")


# ---------------------------------------------------------------------------
# Loaders: read from cached CSV and return a dataset dict
# ---------------------------------------------------------------------------

def load_energy_efficiency() -> dict:
    """
    Load Energy Efficiency from cache.  Returns dict with keys:
      X, y, name, feature_names, n, p, task, target_label, reference, doi.
    X is complete (no NaN); missingness is injected by the benchmark.
    """
    path = _csv_path("energy_efficiency.csv")
    if not os.path.exists(path):
        prepare_energy_efficiency()
    df = pd.read_csv(path)
    feat_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:,  -1].to_numpy(dtype=np.float64)
    return dict(
        X=X, y=y,
        name="Energy Efficiency",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="regression",
        has_natural_missing=False,
        target_label="heating load (kWh/m²)",
        reference="Tsanas & Xifara (2012)",
        doi="https://doi.org/10.1016/j.enbuild.2012.03.003",
    )


def load_wisconsin() -> dict:
    """
    Load Breast Cancer Wisconsin from cache.  X is complete; missingness
    is injected by the benchmark.
    """
    path = _csv_path("wisconsin.csv")
    if not os.path.exists(path):
        prepare_wisconsin()
    df = pd.read_csv(path)
    feat_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:,  -1].to_numpy(dtype=np.float64)
    return dict(
        X=X, y=y,
        name="Wisconsin",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="binary_classification",
        has_natural_missing=False,
        pos_rate=float(y.mean()),
        target_label="malignant (0/1)",
        reference="Street et al. (1993)",
        doi="https://doi.org/10.24432/C5DW2B",
    )


def load_iris() -> dict:
    """Load Iris from cache.  X is complete; missingness injected by benchmark."""
    path = _csv_path("iris.csv")
    if not os.path.exists(path):
        prepare_iris()
    df = pd.read_csv(path)
    feat_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:,  -1].to_numpy(dtype=np.int64)
    return dict(
        X=X, y=y,
        name="Iris",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="multiclass_classification",
        n_classes=3,
        class_names=["setosa", "versicolor", "virginica"],
        has_natural_missing=False,
        target_label="species (0/1/2)",
        reference="Fisher (1936)",
        doi="https://doi.org/10.1111/j.1469-1809.1936.tb02137.x",
    )


def load_horse_colic() -> dict:
    """
    Load Horse Colic from cache.  Contains real clinical missingness (~24%).
    No additional injection needed for Tier 2 benchmarking.
    """
    path = _csv_path("horse_colic.csv")
    if not os.path.exists(path):
        prepare_horse_colic()
    df = pd.read_csv(path)
    feat_names = list(df.columns[:-1])
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:,  -1].to_numpy(dtype=np.float64)
    miss_rate = float(np.isnan(X).mean())
    return dict(
        X=X, y=y,
        name="Horse Colic",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="binary_classification",
        has_natural_missing=True,
        missing_rate=miss_rate,
        pos_rate=float(y.mean()),
        target_label="died or euthanized (0/1)",
        reference="McConaghy (1989), UCI ML Repository",
        doi="https://archive.ics.uci.edu/ml/datasets/Horse+Colic",
    )


def load_auto_mpg() -> dict:
    """
    Load Auto MPG from cache.  Contains real missing data in horsepower (~1.5%).
    No additional injection needed for Tier 2 benchmarking.
    """
    path = _csv_path("auto_mpg.csv")
    if not os.path.exists(path):
        prepare_auto_mpg()
    df = pd.read_csv(path)
    # Columns: mpg (target) + 7 features
    y_col = "mpg"
    feat_names = [c for c in df.columns if c != y_col]
    X = df[feat_names].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    miss_rate = float(np.isnan(X).mean())
    return dict(
        X=X, y=y,
        name="Auto MPG",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="regression",
        has_natural_missing=True,
        missing_rate=miss_rate,
        target_label="fuel efficiency (mpg)",
        reference="Quinlan (1993), UCI ML Repository",
        doi="https://archive.ics.uci.edu/ml/datasets/auto+mpg",
    )


def load_esol() -> dict:
    """
    Load ESOL (Delaney aqueous solubility) from cache.  X is complete;
    missingness is injected by the Tier 3 mechanism sweep.
    """
    path = _csv_path("esol.csv")
    if not os.path.exists(path):
        prepare_esol()
    df = pd.read_csv(path)
    target_col = "measured log solubility in mols/L"
    feat_names = [c for c in df.columns if c != target_col]
    X = df[feat_names].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    return dict(
        X=X, y=y,
        name="ESOL",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="regression",
        has_natural_missing=False,
        target_label="log solubility (mol/L)",
        reference="Delaney (2004)",
        doi="https://doi.org/10.1021/ci034243x",
    )


def load_radon() -> dict:
    """
    Load Radon (Gelman & Hill multilevel benchmark) from cache.

    Returns a dataset dict with an extra key ``groups`` (integer county codes,
    shape (n,)) in addition to the standard X, y keys.  Pass ``groups`` to
    MissMixedRegressor.fit() and MissMixedClassifier.fit().

    X is complete (no NaN); missingness is injected by the benchmark.
    The ``county_name`` column is available separately for display only.

    Additional keys
    ---------------
    groups       : ndarray (n,)  integer county codes 0..n_groups-1
    group_names  : list of str   county names aligned with groups values
    n_groups     : int           number of unique counties (85)
    """
    path = _csv_path("radon.csv")
    if not os.path.exists(path):
        prepare_radon()
    df = pd.read_csv(path)
    feat_names = ["floor", "log_uranium"]
    X      = df[feat_names].to_numpy(dtype=np.float64)
    y      = df["log_radon"].to_numpy(dtype=np.float64)
    groups = df["county_id"].to_numpy(dtype=np.int64)
    # Build a name lookup: group code -> county name
    code_name = df.drop_duplicates("county_id").set_index("county_id")["county_name"]
    group_names = [
        str(code_name.get(i, f"county_{i}"))
        for i in range(groups.max() + 1)
    ]
    # Binarized target for the classification variant: above-median = high radon
    y_median = float(np.median(y))
    y_bin = (y > y_median).astype(np.float64)
    n_groups = int(df["county_id"].nunique())
    return dict(
        X=X, y=y,
        y_bin=y_bin, y_median=y_median,
        groups=groups, group_names=group_names, n_groups=n_groups,
        name="Radon",
        feature_names=feat_names,
        n=len(y), p=X.shape[1],
        task="regression",
        has_natural_missing=False,
        target_label="log radon level",
        reference="Gelman & Hill (2007)",
        doi="https://github.com/avehtari/ROS-Examples",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare MissLearn benchmark datasets")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if CSV files already exist")
    args = parser.parse_args()
    prepare_all(force=args.force)
