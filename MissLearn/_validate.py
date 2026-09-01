"""
_validate.py  --  Data validation, compatibility checking, and preprocessing
                  for MissLearn.

prefit_check(X, y, ...)
    Standalone function that inspects a dataset for known compatibility issues
    with MissLearn's joint-normal FIML approach.  Returns a structured report
    and optionally raises / warns.  Called internally by MissPreprocessor and
    can be called directly by users.

MissPreprocessor
    Wrapper that runs prefit_check and handles categorical encoding before
    delegating to any MissLearn or NaN-native model.  Preserves NaN values
    through the encoding pipeline so FIML models receive them intact.

    Categorical handling policy
    ---------------------------
    object/string columns  → always one-hot encoded (error if encode=None)
    integer columns with low cardinality (≤ categorical_threshold unique
      values, excluding NaN)  → warn; encode if encode='onehot' or 'auto'

    binary 0/1 columns     → note (MVN approximation); pass through by default
    all other columns      → pass through as-is

    NaN preservation in encoded columns
    ------------------------------------
    If a categorical column has NaN in row i, all corresponding one-hot
    columns for row i are also NaN, so downstream FIML models treat that
    observation as missing (consistent with the no-imputation philosophy).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class ValidationResult:
    """
    Structured output of prefit_check.

    Attributes
    ----------
    errors   : list of str  -- fatal issues; fit should not proceed
    warnings : list of str  -- non-fatal issues worth fixing
    notes    : list of str  -- informational, no action required
    passed   : bool         -- True if no errors
    """

    def __init__(self):
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self.notes:    List[str] = []
        self.passed:   bool      = True

    def _add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def _add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def _add_note(self, msg: str) -> None:
        self.notes.append(msg)

    def summary(self, verbose: bool = True) -> None:
        status = "PASSED" if self.passed else "FAILED"
        print(f"\n{'=' * 66}")
        print(f"  MissLearn Compatibility Check  [{status}]")
        print(f"{'=' * 66}")
        if self.errors:
            print(f"  ERRORS ({len(self.errors)})  -- fit cannot proceed:")
            for e in self.errors:
                print(f"    [E] {e}")
        if self.warnings:
            print(f"  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    [W] {w}")
        if self.notes and verbose:
            print(f"  NOTES ({len(self.notes)}):")
            for n in self.notes:
                print(f"    [i] {n}")
        if self.passed and not self.warnings:
            print("  All checks passed.  Data appears compatible with MissLearn.")
        print(f"{'=' * 66}\n")

    def __repr__(self) -> str:
        return (f"ValidationResult(passed={self.passed}, "
                f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
                f"notes={len(self.notes)})")


# ---------------------------------------------------------------------------
# prefit_check
# ---------------------------------------------------------------------------

def copula_is_configured(estimator) -> bool:
    """True if this estimator, or a template nested inside it, sets copula.

    A wrapper such as MissMulticlass, MissEnsemble or MissPreprocessor holds
    the model that actually carries the copula parameter, so a shallow check
    on the outer object reports False for a configuration that is in fact
    already in force. This walks the same nesting attributes the rest of the
    library uses.

    Returns True for copula=True and for copula='auto', since in both cases
    the decision has been handed to the model and advising the user to set it
    is noise.
    """
    seen: set = set()
    stack = [estimator]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        cop = getattr(obj, 'copula', None)
        if cop is True or (isinstance(cop, str) and cop.lower() == 'auto'):
            return True
        for attr in ('estimator', 'estimator_', '_est', 'model_'):
            stack.append(getattr(obj, attr, None))
        for item in (getattr(obj, 'estimators', None) or []):
            stack.append(item[1] if isinstance(item, tuple) else item)
    return False


def prefit_check(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    model_name: str = '',
    feature_names: Optional[List[str]] = None,
    categorical_threshold: int = 10,
    missingness_threshold: float = 0.70,
    scale_ratio_threshold: float = 1000.0,
    kurtosis_threshold: float = 7.0,
    n_gaussian_threshold: int = 1000,
    raise_on_error: bool = True,
    emit_warnings: bool = True,
    copula_configured: bool = False,
) -> ValidationResult:
    """
    Check whether (X, y) is compatible with MissLearn's FIML approach.

    Parameters
    ----------
    X : array-like of shape (n, p)
    y : array-like of shape (n,), optional
    model_name : str
        Name of the model being fitted (used for model-specific checks).
    feature_names : list of str, optional
        Column names for clearer messages.
    categorical_threshold : int, default 10
        Integer columns with at most this many unique values are flagged as
        potentially categorical.
    missingness_threshold : float, default 0.70
        Columns with more than this fraction of NaN are flagged.
    scale_ratio_threshold : float, default 1000.0
        If max(col_std) / min(col_std) exceeds this, flag scale imbalance.
    kurtosis_threshold : float, default 7.0
        Columns with excess kurtosis above this are flagged (suggests copula).
    n_gaussian_threshold : int, default 1000
        Warn if n > this and model_name starts with 'MissGaussian'.
    raise_on_error : bool, default True
        If True, raise ValueError for fatal errors.
    emit_warnings : bool, default True
        If True, emit Python warnings for non-fatal issues.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    X = np.asarray(X)
    n, p = X.shape

    # A short feature_names list is not fatal, but col_label falls back to
    # "col 7" past the end, so the report silently mixes named and numbered
    # columns and the reader cannot tell which names were believed. Warn
    # rather than error: unlike MissSensitivity, where names map onto
    # coefficients, here they only label the output.
    if feature_names is not None and len(feature_names) != p:
        result._add_warning(
            f"feature_names has {len(feature_names)} entries but X has {p} "
            f"columns.  Columns beyond the list are reported by index, so "
            f"the report below mixes names and numbers."
        )

    # Helper for column label
    def col_label(j: int) -> str:
        if feature_names and j < len(feature_names):
            return f"'{feature_names[j]}' (col {j})"
        return f"col {j}"

    # ------------------------------------------------------------------
    # 1. Non-numeric dtype
    # ------------------------------------------------------------------
    if not np.issubdtype(X.dtype, np.number):
        result._add_error(
            f"X has non-numeric dtype ({X.dtype}).  MissLearn requires "
            f"numeric inputs.  Encode categorical columns before fitting, or "
            f"use MissPreprocessor which handles encoding automatically."
        )
        if raise_on_error:
            raise ValueError(result.errors[-1])
        return result   # cannot proceed with further numeric checks

    X_float = X.astype(float)

    # ------------------------------------------------------------------
    # 1b. Infinity
    #
    # NaN is the whole point of this library and is not a problem. Infinity
    # is a different thing: no conditional distribution makes an infinite
    # observation meaningful, and every estimator here refuses it at fit.
    # This check existed nowhere, so prefit_check printed "All checks passed.
    # Data appears compatible with MissLearn" for data that fit then rejected
    # with a ValueError, which is the one outcome a pre-flight check must
    # never produce.
    # ------------------------------------------------------------------
    n_inf = int(np.isinf(X_float).sum())
    if n_inf:
        cols = [j for j in range(p) if np.isinf(X_float[:, j]).any()]
        result._add_error(
            f"X contains {n_inf} infinite value(s), in "
            f"{', '.join(col_label(j) for j in cols[:4])}"
            f"{' and others' if len(cols) > 4 else ''}.  NaN marks an "
            f"unobserved entry and is handled; infinity is not an "
            f"observation any distribution can place, and fitting will "
            f"refuse it.  Replace these with NaN if the value is unknown."
        )

    # ------------------------------------------------------------------
    # 2. All-NaN columns
    # ------------------------------------------------------------------
    for j in range(p):
        if np.all(np.isnan(X_float[:, j])):
            result._add_error(
                f"{col_label(j)} is entirely NaN.  Drop this column before "
                f"fitting."
            )

    # ------------------------------------------------------------------
    # 3. All-NaN rows
    # ------------------------------------------------------------------
    all_nan_rows = np.where(np.all(np.isnan(X_float), axis=1))[0]
    if len(all_nan_rows) > 0:
        result._add_warning(
            f"{len(all_nan_rows)} row(s) have all features missing "
            f"(rows {all_nan_rows[:5].tolist()}{'...' if len(all_nan_rows)>5 else ''}).  "
            f"These rows contribute no information to the likelihood and will "
            f"receive population-mean predictions."
        )

    # ------------------------------------------------------------------
    # 4. Constant columns (after removing NaN)
    # ------------------------------------------------------------------
    for j in range(p):
        col = X_float[:, j]
        obs = col[~np.isnan(col)]
        if len(obs) > 0 and np.nanstd(col) == 0.0:
            result._add_error(
                f"{col_label(j)} is constant (value={obs[0]:.4g}).  "
                f"Constant columns cause a singular covariance matrix.  "
                f"Drop this column before fitting."
            )

    # Bail on errors before continuing with numeric checks
    if not result.passed:
        if raise_on_error and result.errors:
            raise ValueError(
                "prefit_check found fatal issues:\n  "
                + "\n  ".join(result.errors)
            )
        return result

    # ------------------------------------------------------------------
    # 5. Categorical / binary columns
    # ------------------------------------------------------------------
    for j in range(p):
        col = X_float[:, j]
        obs = col[~np.isnan(col)]
        if len(obs) == 0:
            continue
        unique_vals = np.unique(obs)
        n_unique = len(unique_vals)

        # Binary 0/1
        if n_unique <= 2 and set(unique_vals.tolist()).issubset({0.0, 1.0}):
            result._add_note(
                f"{col_label(j)} appears binary (0/1).  The multivariate "
                f"normal assumption is approximate for binary predictors.  "
                f"Results are typically still useful."
                + ("" if copula_configured else
                   "  The default copula='auto' improves the approximation "
                   "where a margin warrants it.")
            )
        # Integer-valued with low cardinality -- likely ordinal / nominal
        elif (np.all(obs == np.floor(obs)) and
              n_unique <= categorical_threshold):
            result._add_warning(
                f"{col_label(j)} appears categorical or ordinal "
                f"({n_unique} unique integer values: "
                f"{unique_vals[:6].astype(int).tolist()}"
                f"{'...' if n_unique > 6 else ''}).  "
                f"The multivariate normal assumption is violated for "
                f"categorical predictors.  Use MissPreprocessor with "
                f"encode='auto' to one-hot encode, or accept the "
                f"approximation for ordinal scales."
            )

    # ------------------------------------------------------------------
    # 6. High missingness per column
    # ------------------------------------------------------------------
    for j in range(p):
        frac = np.mean(np.isnan(X_float[:, j]))
        if frac > missingness_threshold:
            result._add_warning(
                f"{col_label(j)} is {frac*100:.1f}% missing.  "
                f"Conditional distributions for columns with very high "
                f"missingness can be unreliable.  Consider dropping if "
                f">80% missing."
            )

    # ------------------------------------------------------------------
    # 7. n < p  (ill-conditioned Sigma_X)
    # ------------------------------------------------------------------
    n_eff = int(np.sum(~np.all(np.isnan(X_float), axis=1)))
    if n_eff < p:
        result._add_warning(
            f"Effective sample size ({n_eff} rows with at least one observed "
            f"feature) is less than p={p}.  The sample covariance of X will "
            f"be rank-deficient, causing numerical issues.  Reduce p or "
            f"collect more data."
        )
    elif n_eff < 3 * p:
        result._add_warning(
            f"Effective sample size ({n_eff}) is small relative to p={p} "
            f"(rule of thumb: n >= 3p).  Parameter estimates may be "
            f"unreliable.  Consider ridge regularisation (MissRidge)."
        )

    # ------------------------------------------------------------------
    # 8. Scale imbalance
    # ------------------------------------------------------------------
    col_stds = np.nanstd(X_float, axis=0)   # vectorised; single pass over data
    col_stds_pos = col_stds[col_stds > 0]
    if len(col_stds_pos) >= 2:
        ratio = col_stds_pos.max() / col_stds_pos.min()
        if ratio > scale_ratio_threshold:
            result._add_warning(
                f"Feature scales differ by a factor of {ratio:.0f} "
                f"(max std / min std).  Large scale differences can cause "
                f"numerical instability in the Cholesky decomposition of "
                f"Sigma_X.  Consider standardising features first."
            )

    # ------------------------------------------------------------------
    # 9. Near-zero variance (not quite constant -- caught above)
    # ------------------------------------------------------------------
    for j in range(p):
        std_j = col_stds[j]
        if 0 < std_j < 1e-6:
            result._add_warning(
                f"{col_label(j)} has near-zero variance (std={std_j:.2e}).  "
                f"This may cause near-singular Sigma_X.  Consider dropping "
                f"or standardising."
            )

    # ------------------------------------------------------------------
    # 10. High kurtosis (suggests non-normality; copula recommended)
    # ------------------------------------------------------------------
    try:
        # Local moment arithmetic rather than scipy.stats, matching
        # _copula. This runs on every MissPreprocessor fit, and scipy's
        # version routes through its array-API shim, which is the one
        # place this environment has shown itself to be fragile.
        from ._copula import _skew_kurtosis
        for j in range(p):
            col = X_float[:, j]
            obs = col[~np.isnan(col)]
            if len(obs) >= 8:
                k = float(_skew_kurtosis(obs)[1])
                if abs(k) > kurtosis_threshold:
                    advice = (
                        "The copula transform is already enabled, which is "
                        "the intended response."
                        if copula_configured else
                        "Estimators default to copula='auto', which applies "
                        "the transform when a margin warrants it, so this is "
                        "informational unless copula=False was set."
                    )
                    result._add_note(
                        f"{col_label(j)} has excess kurtosis {k:.1f} "
                        f"(|k|>{kurtosis_threshold}).  The joint normal "
                        f"assumption may be violated.  {advice}"
                    )
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 11. Model-specific: MissGaussian with large n
    # ------------------------------------------------------------------
    is_gaussian = 'gaussian' in model_name.lower() or 'gp' in model_name.lower()
    if is_gaussian and n > n_gaussian_threshold:
        result._add_warning(
            f"MissGaussian models have O(n^3) complexity.  With n={n} "
            f"(threshold={n_gaussian_threshold}) fitting may be very slow "
            f"or run out of memory.  Consider MissRidge or MissMixed for "
            f"large datasets."
        )

    # ------------------------------------------------------------------
    # 12. y checks
    # ------------------------------------------------------------------
    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) != n:
            result._add_error(
                f"len(y)={len(y_arr)} does not match X.shape[0]={n}."
            )
        else:
            frac_y_missing = np.mean(np.isnan(y_arr))
            if frac_y_missing > missingness_threshold:
                result._add_warning(
                    f"y is {frac_y_missing*100:.1f}% missing.  With very "
                    f"few observed outcomes the model may not converge "
                    f"reliably."
                )
            if frac_y_missing == 1.0:
                result._add_error("y is entirely NaN.  Nothing to fit.")
            n_inf_y = int(np.isinf(y_arr).sum())
            if n_inf_y:
                result._add_error(
                    f"y contains {n_inf_y} infinite value(s).  A missing "
                    f"outcome is written as NaN and still contributes "
                    f"through the joint distribution; an infinite one "
                    f"cannot, and fitting will refuse it."
                )

    # ------------------------------------------------------------------
    # Emit Python warnings for warnings (not errors)
    # ------------------------------------------------------------------
    if emit_warnings:
        for w in result.warnings:
            warnings.warn(f"MissLearn [{model_name or 'prefit_check'}]: {w}",
                          UserWarning, stacklevel=3)

    if raise_on_error and result.errors:
        raise ValueError(
            "prefit_check found fatal compatibility issues:\n  "
            + "\n  ".join(result.errors)
        )

    return result


# ---------------------------------------------------------------------------
# MissPreprocessor
# ---------------------------------------------------------------------------

class MissPreprocessor:
    """
    Wrapper that validates and encodes a dataset before delegating to any
    MissLearn or NaN-native tree model.

    Notes
    -----
    1. Run prefit_check on raw X and report compatibility issues.
    2. One-hot encode categorical columns while preserving NaN (so downstream
       FIML models see missing categorical values as missing, not imputed).

    3. Delegate fit / predict / predict_proba / predict_interval / score /
       summary and all other method calls to the underlying estimator.

    4. Apply the same encoding transformation at predict time using the
       categories learned at fit time.

    Parameters
    ----------
    estimator : unfitted MissLearn or NaN-native tree model
    encode : {'auto', 'onehot', None}, default 'auto'
        'auto'   -- always encode object/string columns; encode integer
                    columns with <= categorical_threshold unique values.

        'onehot' -- encode any column detected as categorical (object or
                    low-cardinality integer).

        None     -- no encoding; raise an error for non-numeric columns.
    categorical_threshold : int, default 10
        Integer columns with at most this many unique non-NaN values are
        considered categorical.
    drop : {'first', None}, default 'first'
        Whether to drop one category per encoded column to avoid
        multicollinearity.  'first' drops the lexicographically first
        category.
    validate : bool, default True
        Whether to run prefit_check before fitting.
    feature_names : list of str, optional
        Column labels, so that the compatibility report names the features the
        user recognises rather than X0, X1, ...  A DataFrame's own columns are
        picked up automatically and this argument overrides them.
    raise_on_error : bool, default True
        Whether to raise ValueError on fatal compatibility issues.
    verbose : bool, default True
        Print the validation summary.

    Attributes
    ----------
    estimator_           : fitted underlying model
    encoding_map_        : dict {col_idx: {'type', 'categories', 'encoded_names'}}
    feature_names_in_    : list of input feature names
    feature_names_out_   : list of output feature names after encoding
    n_features_in_       : int
    n_features_out_      : int
    validation_report_   : ValidationResult
    """

    #: This class takes data that is not yet numeric and makes it numeric, so
    #: the shared fit wrapper must not apply its numeric validation first.
    #: See _pandas_compat._wrap_fit.
    _ACCEPTS_RAW_INPUT = True


    def __init__(
        self,
        estimator,
        encode: str = 'auto',
        categorical_threshold: int = 10,
        drop: Optional[str] = 'first',
        validate: bool = True,
        raise_on_error: bool = True,
        verbose: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        self.estimator            = estimator
        self.encode               = encode
        self.categorical_threshold = categorical_threshold
        self.drop                 = drop
        self.validate             = validate
        self.raise_on_error       = raise_on_error
        self.verbose              = verbose
        self.feature_names        = feature_names

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'estimator':             self.estimator,
            'encode':                self.encode,
            'categorical_threshold': self.categorical_threshold,
            'drop':                  self.drop,
            'validate':              self.validate,
            'raise_on_error':        self.raise_on_error,
            'verbose':               self.verbose,
            'feature_names':         self.feature_names,
        }
        if deep and hasattr(self.estimator, 'get_params'):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f'estimator__{k}'] = v
        return params

    def set_params(self, **params):
        est_params = {}
        for k, v in params.items():
            if k.startswith('estimator__'):
                est_params[k[len('estimator__'):]] = v
            else:
                setattr(self, k, v)
        if est_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**est_params)
        return self

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _detect_encoding(self, X: np.ndarray) -> Dict[int, dict]:
        """
        Return {col_idx: info} for columns that need encoding.
        info keys: 'reason' ('object'|'categorical'|'binary'), 'categories'
        """
        enc: Dict[int, dict] = {}
        n, p = X.shape
        for j in range(p):
            col = X[:, j]

            # Object/string handled separately (X has been forced to float)
            # so we cannot reach here with string dtype -- handled pre-conversion.

            obs = col[~np.isnan(col.astype(float))]
            if len(obs) == 0:
                continue

            unique_vals = np.unique(obs)
            n_unique = len(unique_vals)

            if (np.all(obs == np.floor(obs)) and n_unique <= self.categorical_threshold):
                reason = 'binary' if n_unique <= 2 else 'categorical'
                enc[j] = {'reason': reason, 'categories': unique_vals}

        return enc

    def _build_encoding_map(
        self,
        X_raw,                      # may be object dtype before conversion
        feature_names: List[str],
    ) -> Tuple[Dict[int, dict], List[str]]:
        """
        Scan raw X (before float conversion) to build the encoding map.
        Also handles object/string columns.
        Returns (encoding_map, output_feature_names).
        """
        n, p = X_raw.shape
        encoding_map: Dict[int, dict] = {}
        out_names: List[str] = []

        for j in range(p):
            fname = feature_names[j]
            col = X_raw[:, j]

            # Determine if this is a string/object column
            col_is_object = False
            try:
                col_f = col.astype(float)
            except (ValueError, TypeError):
                col_is_object = True

            if col_is_object:
                if self.encode is None:
                    raise ValueError(
                        f"MissPreprocessor: column '{fname}' (col {j}) has "
                        f"non-numeric dtype and encode=None.  Set "
                        f"encode='auto' or encode='onehot' to handle "
                        f"categorical encoding, or convert manually."
                    )
                # Compute categories from non-null values
                # Safe against float NaN and pandas NA (pd.NA == pd.NA raises TypeError)
                _omask = np.zeros(len(col), dtype=bool)
                for _vi, _v in enumerate(col):
                    try:
                        _omask[_vi] = _v is not None and bool(_v == _v)
                    except (TypeError, ValueError):
                        _omask[_vi] = False  # pd.NA or similar
                obs_mask = _omask
                obs_vals = col[obs_mask]
                cats = np.unique(obs_vals.astype(str))
                cats_to_use = cats[1:] if self.drop == 'first' and len(cats) > 1 else cats
                enc_names = [f"{fname}_{c}" for c in cats_to_use]
                encoding_map[j] = {
                    'type': 'object',
                    'categories': cats,
                    'encoded_names': enc_names,
                }
                out_names.extend(enc_names)
                continue

            # Numeric column
            col_f_full = col_f
            obs = col_f_full[~np.isnan(col_f_full)]
            if len(obs) == 0:
                out_names.append(fname)
                continue

            unique_vals = np.unique(obs)
            n_unique = len(unique_vals)
            is_int_valued = np.all(obs == np.floor(obs))

            should_encode = False
            if is_int_valued and n_unique <= self.categorical_threshold:
                if self.encode in ('auto', 'onehot'):
                    should_encode = True

            if should_encode:
                cats = unique_vals
                cats_to_use = cats[1:] if self.drop == 'first' and len(cats) > 1 else cats
                enc_names = [f"{fname}_{int(c)}" for c in cats_to_use]
                encoding_map[j] = {
                    'type': 'categorical',
                    'categories': cats,
                    'encoded_names': enc_names,
                }
                out_names.extend(enc_names)
            else:
                out_names.append(fname)

        return encoding_map, out_names

    def _apply_encoding(self, X_raw, encoding_map: Dict[int, dict]) -> np.ndarray:
        """
        Apply one-hot encoding according to encoding_map.
        NaN in categorical columns → NaN in all corresponding output columns.
        Returns float64 ndarray.
        """
        n, p = X_raw.shape
        cols_out: List[np.ndarray] = []

        for j in range(p):
            if j not in encoding_map:
                # Pass through as float
                try:
                    cols_out.append(X_raw[:, j].astype(float).reshape(-1, 1))
                except (ValueError, TypeError):
                    # String column not in encoding map -- shouldn't happen
                    raise ValueError(
                        f"Column {j} could not be converted to float and is "
                        f"not in the encoding map.  Set encode='auto'."
                    )
                continue

            info = encoding_map[j]
            cats_all = info['categories']           # full category array
            n_encoded = len(info['encoded_names'])  # number of output columns

            # Determine which categories were kept (after drop='first')
            if self.drop == 'first' and len(cats_all) > 1:
                cats_kept = cats_all[1:]
            else:
                cats_kept = cats_all

            block = np.full((n, n_encoded), np.nan, dtype=float)
            col_raw = X_raw[:, j]

            # Identify NaN rows
            try:
                col_float = col_raw.astype(float)
                nan_mask = np.isnan(col_float)
            except (ValueError, TypeError):
                # Same guard as _build_encoding_map: treat None, float NaN,
                # and pd.NA (whose == raises TypeError) as missing.
                nan_mask = np.empty(len(col_raw), dtype=bool)
                for _vi, _v in enumerate(col_raw):
                    try:
                        nan_mask[_vi] = _v is None or not bool(_v == _v)
                    except (TypeError, ValueError):
                        nan_mask[_vi] = True  # pd.NA or similar

            observed_mask = ~nan_mask
            if observed_mask.any():
                col_obs = col_raw[observed_mask]
                # Branch on the encoding type recorded at fit time rather
                # than re-guessing from the transform data's dtype, so both
                # phases compare on the same scale.
                if info['type'] == 'object':
                    # Categories were stored via str() at fit time
                    col_obs_s = np.array([str(v) for v in col_obs])
                    for k, cat in enumerate(cats_kept):
                        block[observed_mask, k] = (
                            col_obs_s == str(cat)
                        ).astype(float)
                else:
                    # Numeric categorical: vectorised float comparison
                    col_obs_f = col_obs.astype(float)
                    cats_f = np.asarray(cats_kept, dtype=float)
                    block[observed_mask, :] = (
                        col_obs_f[:, None] == cats_f[None, :]
                    ).astype(float)

            cols_out.append(block)

        return np.concatenate(cols_out, axis=1).astype(float)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, **kwargs):
        """
        Validate, encode, and fit the underlying estimator.

        Parameters
        ----------
        X : array-like of shape (n, p)
        y : array-like of shape (n,)
        **kwargs : passed through to the underlying estimator (e.g. groups)
        """
        # Capture column labels before the conversion to ndarray discards them.
        df_names = None
        if hasattr(X, 'columns'):
            df_names = [str(c) for c in X.columns]

        X_raw = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)
        n, p  = X_raw.shape

        # Feature names, in order of preference. Hardcoding X0..Xp here used to
        # discard both an explicit feature_names argument and a DataFrame's own
        # columns, which made every compatibility warning refer to 'X4' rather
        # than to the column the user would recognise, and so unusable.
        if self.feature_names is not None:
            if len(self.feature_names) != p:
                raise ValueError(
                    f"feature_names has {len(self.feature_names)} entries but "
                    f"X has {p} columns.")
            self.feature_names_in_ = [str(c) for c in self.feature_names]
        elif df_names is not None and len(df_names) == p:
            self.feature_names_in_ = df_names
        elif (getattr(self, 'feature_names_in_', None) is not None
              and len(self.feature_names_in_) == p):
            # Set by the pandas compatibility layer; do not overwrite it.
            self.feature_names_in_ = [str(c) for c in self.feature_names_in_]
        else:
            self.feature_names_in_ = [f"X{j}" for j in range(p)]

        # Build encoding map from raw X (handles object dtypes)
        self.encoding_map_, self.feature_names_out_ = self._build_encoding_map(
            X_raw, self.feature_names_in_
        )

        # Apply encoding
        X_enc = self._apply_encoding(X_raw, self.encoding_map_)

        # Validate encoded X
        if self.validate:
            model_name = type(self.estimator).__name__
            self.validation_report_ = prefit_check(
                X_enc, y_arr,
                model_name=model_name,
                feature_names=self.feature_names_out_,
                raise_on_error=self.raise_on_error,
                emit_warnings=True,
                # The wrapped model is what carries copula, and it may sit one
                # level down inside MissMulticlass or MissEnsemble. Without
                # this the report advises copula='auto' at a model that
                # already has it, which is exactly the advice a user has
                # already taken.
                copula_configured=copula_is_configured(self.estimator),
            )
            if self.verbose:
                self.validation_report_.summary()
        else:
            self.validation_report_ = ValidationResult()

        # Fit underlying estimator
        import copy
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(X_enc, y_arr, **kwargs)

        self.n_features_in_  = p
        self.n_features_out_ = X_enc.shape[1]
        return self

    # ------------------------------------------------------------------
    # Transform (encode without fitting)
    # ------------------------------------------------------------------

    def transform(self, X) -> np.ndarray:
        """Apply fitted encoding to X."""
        X_raw = np.asarray(X)
        if X_raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_raw.shape[1]} features but MissPreprocessor was "
                f"fitted on {self.n_features_in_} features."
            )
        return self._apply_encoding(X_raw, self.encoding_map_)

    # ------------------------------------------------------------------
    # Prediction delegation
    # ------------------------------------------------------------------

    def _enc(self, X) -> np.ndarray:
        return self.transform(np.asarray(X))

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.estimator_.predict(self._enc(X), **kwargs)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        return self.estimator_.predict_proba(self._enc(X), **kwargs)

    def predict_interval(self, X, **kwargs):
        return self.estimator_.predict_interval(self._enc(X), **kwargs)

    def decision_function(self, X, **kwargs) -> np.ndarray:
        return self.estimator_.decision_function(self._enc(X), **kwargs)

    def score(self, X, y, **kwargs) -> float:
        return self.estimator_.score(self._enc(X), np.asarray(y, dtype=float),
                                     **kwargs)

    def summary(self) -> None:
        if hasattr(self, 'validation_report_'):
            self.validation_report_.summary()
        if hasattr(self.estimator_, 'summary'):
            self.estimator_.summary()

    # ------------------------------------------------------------------
    # Attribute delegation to underlying estimator
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails
        if name.startswith('_') or name in ('estimator', 'estimator_'):
            raise AttributeError(name)
        try:
            return getattr(self.__dict__['estimator_'], name)
        except KeyError:
            raise AttributeError(
                f"MissPreprocessor has no attribute '{name}' and the "
                f"underlying estimator has not been fitted yet."
            )

    def __repr__(self) -> str:
        return (f"MissPreprocessor(estimator={self.estimator!r}, "
                f"encode='{self.encode}')")
