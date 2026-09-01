# MissLearn Interpretation Guide

How to read and interpret every output and visualisation MissLearn produces; model
summaries, coefficient tables, prediction intervals, diagnostics, SHAP explanations,
sensitivity analyses, multiple-imputation pooling, ensembles, and the benchmark figures.

This guide assumes you have already fitted a model. For *how* to fit one, see the
User Guide (`docs/MissLearn_User_Guide.pdf`); for the estimation theory, see
`docs/METHODS_GUIDE.md`.

## Table of contents

1. [Model `summary()` blocks](#1-model-summary-blocks)
   - [Regressor example (MissLinear)](#regressor-example-misslinear)
   - [Classifier example (MissLogistic)](#classifier-example-misslogistic)
   - [Caveats on standard errors](#caveats-on-standard-errors)
2. [Coefficients & inference](#2-coefficients--inference)
3. [Predictions & uncertainty](#3-predictions--uncertainty)
4. [Diagnostics; MissDiagnostic, and turning them into a model choice with MissRecommender](#4-diagnostics-missdiagnostic)
5. [Explanations; MissExplainer](#5-explanations-missexplainer)
6. [Sensitivity analysis; MissSensitivity](#6-sensitivity-analysis-misssensitivity)
7. [Multiple-imputation outputs; MissImputer](#7-multiple-imputation-outputs-missimputer)
8. [Ensemble outputs; MissEnsemble](#8-ensemble-outputs-missensemble)
9. [Benchmark figures](#9-benchmark-figures)

---

## 1. Model `summary()` blocks

Every MissLearn estimator prints a structured report from `model.summary()`. The layout
is shared across the family (`MissLinear`, `MissLogistic`, `MissRidge*`, `MissLASSO*`,
`MissBayes*`, `MissNeighbors*`, `MissSupport*`, `MissGaussian*`, `MissMixed*`), so once
you can read one, you can read them all. The two walk-throughs below cover the common
regression and classification layouts.

### Regressor example (MissLinear)

```text
==========================================================================
                  MissLinear  --  FIML Linear Regression
==========================================================================
  Observations    : 500  (complete: 312,  partial: 188)          (1)
  Features        : 4
  Missing (X)     : 264 (13.2%)                                  (2)
  Missing (y)     : 21                                           (3)
  Converged       : True                                         (4)
  Copula          : auto (applied)                               (5)
  Log-likelihood  : -2841.7093                                   (6)
  AIC             : 5721.4186   BIC: 5801.6650                   (7)
  Residual std    : 0.8412  (var: 0.7076)                        (8)
--------------------------------------------------------------------------
  Coefficients:
                      coef     std_err      z_stat     p_value    CI_lower    CI_upper  sig
    ----------------------------------------------------------------------------------------
     intercept      0.1873      0.0451       4.153   3.283e-05      0.0989      0.2757  ***
           age      0.4210      0.0522       8.065   7.327e-16      0.3187      0.5233  ***   (9)
           bmi      0.1104      0.0489       2.258   2.394e-02      0.0146      0.2062    *
      pressure     -0.0312      0.0510      -0.612   5.406e-01     -0.1312      0.0688         (10)
       glucose      0.2988      0.0603       4.955   7.222e-07      0.1806      0.4170  ***
    ----------------------------------------------------------------------------------------
    Significance: '***' p<0.001  '**' p<0.01  '*' p<0.05  '.' p<0.1

  Feature Importances (normalized standardized |coef|):
             age: [############            ]   48.9%              (11)
         glucose: [########                ]   34.7%
             bmi: [###                     ]   12.8%
        pressure: [#                       ]    3.6%
==========================================================================
```

**(1) Observations / complete / partial.** `n_samples_fit_` total rows.
*Complete* rows have no NaN in X or y; *partial* rows have at least one NaN. Every
partial row still contributes its observed entries to the likelihood; nothing is
deleted. A classical (listwise-deletion) analysis of this dataset would have used
only the 312 complete rows and discarded 38% of the sample. If `partial` is 0,
FIML reduces exactly to ordinary maximum likelihood and there is nothing special
to interpret.

**(2) Missing (X).** Count and percentage of missing *cells* out of `n × p`
(`missing_rate_X_`). This is cell-level, not row-level: 13.2% missing cells can
easily mean 40%+ of rows are partial. Use `model.missingness_report()` for the
row-level breakdown.

**(3) Missing (y).** Only printed when y has NaN. Rows with missing y still inform
the joint distribution of X (and, through the correlations, the coefficients); they
are semi-supervised information, not waste.

**(4) Converged.** `converged_`, the optimizer's success flag. If `False`,
coefficients are usable as point estimates but standard errors may be unreliable -
refit with a larger `max_iter`, looser `tol`, or standardised inputs before trusting
the inference.

**(5) Copula.** Present only when `copula=True` or `copula='auto'`. Three variants:
`auto (applied)`, `auto (not applied)`, or `yes  (coefficients on normal-transformed
scale)`. **This line changes how you read the coefficients**: when the copula is
applied, coefficients are on the rank-normalised scale of each variable, not the raw
units. Signs, z-stats, p-values, and importances are still directly interpretable;
"a one-unit change in X" is not. Predictions and prediction intervals are
automatically transformed back to the original y scale.

**(6) Log-likelihood.** The maximised FIML log-likelihood (`loglik_`). Meaningless in
isolation; useful only for comparing models; see [Section 2](#2-coefficients--inference).

**(7) AIC / BIC.** `2k − 2ℓ` and `k·ln(n) − 2ℓ` where `k` is the number of free joint
MVN parameters. Lower is better. BIC penalises complexity harder, so it prefers
smaller models than AIC on the same data.

**(8) Residual std.** `sqrt(sigma_sq_)`: the conditional standard deviation of Y
given a *complete* X row. This is the floor of predictive uncertainty: no observation
can get a prediction interval narrower than ±1.96 × this (at 95%). Compare it to
`std(y)`: if they are close, the model explains little variance.

**(9) Coefficient rows.** For each term:

| Column | Meaning | Rule of thumb |
|---|---|---|
| `coef` | Expected change in y per one-unit increase in the feature, holding others fixed (raw units; or normal-transformed units when the copula line says applied) | Judge magnitude against the scale of y, not other coefficients |
| `std_err` | Standard error of the coefficient | Larger with more missingness in that feature; this is honest, not a bug |
| `z_stat` | `coef / std_err` | \|z\| > 2 ≈ significant at 5% |
| `p_value` | Two-sided p-value under the standard normal | Small = evidence the true coefficient is not 0 |
| `CI_lower / CI_upper` | 95% CI (or `1 − alpha` if you pass a different `alpha` to `summary`) | Excludes 0 ⇔ starred |
| `sig` | Stars: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.1 | A visual index, nothing more |

**(10) A non-significant row.** `pressure` has \|z\| < 1 and a CI straddling zero:
the data are compatible with no effect. Note the SE (0.0510) is comparable to the
others despite this feature having missing values; FIML recovered most of the
information from correlated observed features.

**(11) Feature importances.** `feature_importances_` = normalised absolute
*standardised* coefficients (see `coef_std_` in Section 2), summing to 100%. They
compare relative influence across features regardless of units. They do **not**
carry uncertainty; a 3.6% bar next to a non-significant coefficient means "small
*and* uncertain".

### Classifier example (MissLogistic)

```text
================================================================================
                   MissLogistic  --  FIML Logistic Regression
================================================================================
  Observations    : 800  (complete: 505,  partial: 295)
  Features        : 3
  Classes         : [0 1]                                        (1)
  Missing (X)     : 396 (16.5%)
  Converged       : True
  Log-likelihood  : -3120.5518
  AIC             : 6267.1036   BIC: 6328.0091
--------------------------------------------------------------------------------
  Coefficients (log-odds scale) with Odds Ratios:               (2)
                      coef     std_err      z_stat     p_value    CI_lower    CI_upper  sig    odds_ratio
    ------------------------------------------------------------------------------------------------------
     intercept     -0.9241      0.1103      -8.378   0.000e+00     -1.1403     -0.7079  ***        0.3969
           age      0.6832      0.0987       6.922   4.457e-12      0.4898      0.8766  ***        1.9802  (3)
       smoking      1.2415      0.1250       9.932   0.000e+00      0.9965      1.4865  ***        3.4608
      exercise     -0.4108      0.1042      -3.942   8.073e-05     -0.6150     -0.2066  ***        0.6631  (4)
    ------------------------------------------------------------------------------------------------------
    Significance: '***' p<0.001  '**' p<0.01  '*' p<0.05  '.' p<0.1

  Feature Importances (normalized standardized |coef|):
         smoking: [############            ]   52.1%
             age: [#######                 ]   30.4%
        exercise: [####                    ]   17.5%
================================================================================
```

**(1) Classes.** `classes_`: always exactly two for `MissLogistic`. The model
predicts `P(Y = classes_[1])`, i.e. the *second* listed class is the "positive" one.
Any pair of labels works (`{0,1}`, `{1,2}`, `{-1,1}`); everything is internally
binarised against `classes_[1]`. For multi-class problems, `MissMulticlass` wraps
one-vs-rest copies and its summary lists all classes.

**(2) Log-odds scale.** `coef` is the change in log-odds of the positive class per
unit increase in the feature. The extra `odds_ratio` column is `exp(coef)`
(`odds_ratios_`, which includes the intercept).

**(3) Odds ratio > 1.** Each additional unit of `age` multiplies the odds of class 1
by 1.98; roughly doubles them. OR of the intercept (0.397) is the baseline odds when
all features are 0. Rules of thumb: OR ≈ 1.0 no effect; OR 1.5 to 3 moderate; OR > 3
strong. For a CI on the OR, exponentiate the coefficient CI: here
`exp(0.4898)-exp(0.8766)` = 1.63 to 2.40.

**(4) Odds ratio < 1.** Protective: each unit of `exercise` multiplies the odds by
0.66, i.e. cuts them by 34%. `1/OR` converts to the equivalent "risk factor" framing
(1/0.663 ≈ 1.51).

**Class priors.** `MissBayesClassifier.summary()` additionally prints
`Class priors : P(Y=0)=0.700  P(Y=1)=0.300`: the estimated marginal class
frequencies (`class_prior_`). If the priors are very unbalanced, accuracy is a weak
metric (predicting the majority class scores 70% here); judge the model on AUC and
Brier score instead. That summary also tabulates per-class feature means/stds, a
Cohen's-d `effect_d` column (class separation per feature), and an `avail` column
showing what fraction of each feature was observed at training time.

### Caveats on standard errors

- **Penalised SEs are approximate.** `MissLogistic` applies a small L2 penalty by
  default (`l2_reg=0.01`) to guard against separation, and `MissRidge`/`MissLASSO`
  penalise by construction. The reported SEs come from the curvature of the
  *penalised* objective, so they are slightly optimistic (too narrow) for strongly
  shrunk coefficients. For publication-grade inference from `MissLogistic`, set
  `l2_reg=0` (exact unpenalised FIML). For ridge/LASSO, treat stars as heuristic and
  prefer the multiple-imputation route ([Section 7](#7-multiple-imputation-outputs-missimputer))
  when p-values matter.
- **SEs widen honestly with missingness.** As the missing rate on a feature rises,
  its SE grows because there is genuinely less information about it. This is a
  feature: single imputation followed by OLS produces *falsely narrow* SEs (it
  pretends imputed values were observed), and listwise deletion produces wide *and*
  biased ones under MAR. If a FIML CI looks wider than your imputation-pipeline CI,
  the FIML one is usually the honest one.
- **Numerical source.** SEs come from a numerical Hessian of the reduced
  conditional likelihood over the regression block (all models), with the fitted
  X-moments held fixed.
- **`NaN` means the standard error could not be computed, and is a result in its
  own right.** It appears when the Hessian was singular, when the fit did not
  converge, and when the variance for that coefficient came out non-positive,
  which happens when the Hessian is not positive definite or the Jacobian is ill
  conditioned. Collinear predictors are the common cause of the last one: given
  two columns carrying the same information, neither coefficient is identified
  on its own even though their sum is, and there is no standard error to report
  for either. Read `NaN` as "this coefficient is not identified from these
  data", not as a glitch. The point estimate, the predictions and `loglik_` are
  unaffected, and other coefficients in the same fit usually keep their standard
  errors; when a whole row of the variance matrix is unusable they are withheld
  too, which is the conservative direction.

  These entries used to print as `0.0000`, with a confidence interval collapsed
  onto the point estimate, which reads as a coefficient known exactly. If you
  have output from before this change, a standard error of exactly zero is that
  bug and not a real result.
- **A collinear design is where the penalised models earn their place.**
  `MissRidgeRegressor` and `MissLASSORegressor` shrink explicitly, so the split
  between two collinear predictors is determined by the penalty rather than
  left free, and their standard errors are reported normally. If you need
  per-coefficient inference on correlated predictors, prefer them to
  `MissLinear`, or reduce the correlation before fitting.
- `compute_se=False` skips SEs entirely (they print as `nan`); use it for speed when
  you only need predictions.

---

## 2. Coefficients & inference

Fitted attributes shared by the linear-family models:

### `coef_` vs `coef_std_`

- **`coef_`** (shape `(p,)`); raw-scale coefficients. Use these for **prediction
  arithmetic and substantive statements** ("each extra year of age adds 0.42 units
  of y"). Their magnitudes are *not* comparable across features measured in
  different units.
- **`coef_std_`** (shape `(p,)`); standardised coefficients:
  `coef_j · σ_Xj / σ_Y` for regressors ("SD of y per SD of X_j"), `coef_j · σ_Xj`
  for `MissLogistic` ("log-odds per SD of X_j"). Use these to **rank features** and
  compare effect sizes across features or across datasets. `feature_importances_`
  is just `|coef_std_|` normalised to sum to 1.

Rule of thumb: |standardised coef| < 0.1 small, 0.1 to 0.3 moderate, > 0.3 large
(regression case, Cohen-style).

### Standard errors and tests

- **`se_`**: shape `(p+1,)`, **intercept first**, then the p features. This
  ordering matters when you index: `se_[0]` is the intercept, `se_[1+j]` pairs with
  `coef_[j]`.
- **`z_stats_`**: `coef / se`, same `(p+1,)` layout. NaN where `se_` is 0 or NaN.
- **`pvalues_`**: two-sided p-values from the standard normal. Asymptotic: with
  very small n (< ~50) they are approximate.
- **`conf_int(alpha=0.05)`**: returns an `(p+1, 2)` array `[lower, upper]`, row 0 =
  intercept. `alpha=0.05` gives 95% intervals; pass `alpha=0.10` for 90%.

### `odds_ratios_` (classifiers)

Shape `(p+1,)` including the intercept, equal to `exp([intercept_, *coef_])`.
Interpretation is multiplicative on the odds: OR = 1.98 means +98% odds per unit.
For a k-unit change, the OR is `odds_ratios_[j] ** k`. Confidence intervals for the
OR: `np.exp(model.conf_int())`.

### `loglik_`, `aic_`, `bic_`

- `loglik_` is the **data log-likelihood only; any L2 penalty is excluded** (the
  penalty is added back after optimisation, a convention shared by `MissLogistic`
  and `MissRidgeRegressor`). So `aic_`/`bic_` measure fit-to-data, and models with
  different `l2_reg` values remain comparable.
- AIC/BIC are **comparable across MissLearn models fitted to the same data**: same
  rows, same columns, same transformation. Lower wins. Typical uses: compare a model
  with and without a candidate feature set, or `copula=False` vs `copula=True` fits
  only if the likelihood is on the same scale (a copula fit changes the scale of the
  modelled variables, so **do not compare AIC across copula settings**).
- Do **not** compare against a plain sklearn model's log-likelihood: MissLearn's is
  a *joint* likelihood of (Y, X) (or Y|X plus the X-marginal for `MissLogistic`),
  whereas conventional regression likelihoods are conditional-only. The absolute
  numbers live on different scales.
- Differences that matter: ΔAIC < 2; essentially tied; 4 to 7; noticeably better;
  \> 10; decisively better (Burnham-Anderson).

### `converged_`

The optimizer's success flag. `False` does not automatically invalidate the fit
(L-BFGS-B sometimes reports failure after reaching a perfectly good optimum on a
flat objective), but treat SEs and p-values with suspicion, and re-fit with scaled
inputs or higher `max_iter` before publishing numbers.

---

## 3. Predictions & uncertainty

### The four prediction methods

| Method | Available on | Returns | Missing-X handling |
|---|---|---|---|
| `predict(X)` | all models | point prediction / class label | conditional-normal expectation `E[Y \| X_obs]`; no imputation |
| `predict_proba(X)` | classifiers | `(n, 2)` probabilities, column 1 = `P(Y = classes_[1])` | exact 1-D Gauss-Hermite integration over the *distribution* of the missing features |
| `decision_function(X)` | classifiers | log-odds (linear predictor) | missing features replaced by their conditional means (plug-in) |
| `predict_interval(X, alpha=0.05)` | every regressor: `MissLinear`, the ridge and LASSO regressors, `MissBayes`, `MissGaussian`, `MissNeighbors`, `MissSupport`, `MissMixed` (which also takes `groups`), `MissEnsemble`, and through `MissPreprocessor` | `(lower, upper)` arrays | interval width grows with the row's missingness |

Note the subtle difference for partial rows:
`sigmoid(decision_function(x)) ≠ predict_proba(x)[1]`. `predict_proba` integrates the
sigmoid over the conditional distribution of the missing features (which pulls
probabilities toward 0.5 as uncertainty grows); `decision_function` plugs in the
conditional mean (no such moderation). Use `predict_proba` whenever the probability
itself matters; use `decision_function` only when you need a monotone score
(e.g. for ROC curves; both give identical AUC on complete rows).

### Intervals widen with row missingness: the key property

For `MissLinear`, the 95% interval half-width is `1.96 · sqrt(Var[Y | X_obs])`:

```text
row                     prediction        95% interval          half-width
------------------------------------------------------------------------------
all 4 features observed    5.21        [ 3.56,  6.86]      1.65 = 1.96·√σ²         (a)
2 of 4 observed            5.05        [ 2.83,  7.27]      2.22                     (b)
all 4 missing              4.90        [ 0.98,  8.82]      3.92 = 1.96·√Σ_YY        (c)
```

- **(a) Complete row**: the narrowest possible interval. Its width is set by the
  residual std (`sqrt(sigma_sq_)` from the summary): irreducible outcome noise given
  full information.
- **(b) Half-missing row**: the model integrates over what the two unobserved
  features *could be* given the two observed ones. The interval widens by exactly
  the extra conditional variance; how much depends on how predictable the missing
  features are from the observed ones (strong correlations → barely wider).
- **(c) All-missing row**: the prediction collapses to the marginal mean of y
  (`mu_joint_[0]`) and the interval to the *marginal* ±1.96 SD of y. The model is
  telling you: "with no inputs I know nothing beyond the training distribution of y."

If your intervals do **not** vary across rows, you are looking at a complete-data
matrix. If a downstream consumer needs one number per row for "how sure is this
prediction", use the interval half-width; it is the honest per-row uncertainty.

`MissBayes` and `MissGaussian` regressors return posterior *credible* intervals with
the same qualitative behaviour (all-observed rows tightest, all-missing rows widest).
`MissEnsemble.predict_interval` is different in kind; see
[Section 8](#8-ensemble-outputs-missensemble).

### Calibration and the Brier score

The **Brier score** = mean squared error of predicted probabilities against 0/1
outcomes; 0 is perfect, 0.25 is what constant-0.5 predictions score. It rewards
*calibration*: saying 70% when the event happens 70% of the time.

This is the metric where FIML most consistently beats imputation baselines in the
benchmarks. The reason is structural: for a partial row, imputation pipelines
produce a single filled-in x and an overconfident sigmoid output, while FIML's
Gauss-Hermite integration produces a probability correctly shrunk toward the base
rate in proportion to what is actually unknown. Accuracy and AUC only see the
ranking, so they often tie; Brier sees the overconfidence. **If you report one
classification metric for a missing-data problem, report Brier.**

---

## 4. Diagnostics: MissDiagnostic

`MissDiagnostic(X, y).fit().summary()` answers one question: *is FIML's MAR
assumption plausible for this dataset?* The report has five parts plus a verdict.

```text
======================================================================
          MissDiagnostic  --  Missing Data Mechanism Report
======================================================================
  n = 500  |  p = 5  |  significance level alpha = 0.05
  Overall missingness: 312/2500 cells (12.5%)
    age          [###                 ]  14.8%
    bmi          [####                ]  18.4%
    ...
```

### Little's MCAR test

```text
  Little's MCAR Test
  --------------------------------------------------------------------
  chi2(14) = 41.203  p = 0.0002  (patterns used: 6)
  Result: MCAR rejected (p < 0.05).  Data is NOT missing completely at random.
  This does not invalidate FIML -- see MAR analysis below.
```

- **Significant (p < α) → reject MCAR.** Rows with missing values differ
  systematically from complete rows, therefore **complete-case (listwise) deletion
  is biased**: the retained rows are an unrepresentative sample. This is the single
  most actionable line in the report: it disqualifies "just drop the NaNs".
- **Not significant → cannot reject MCAR.** The data are *consistent with* MCAR
  (not proof; the test has limited power with few patterns or small n). Under
  MCAR both deletion and FIML are unbiased, but FIML is still more efficient.
- Rejecting MCAR does **not** hurt FIML: FIML only needs MAR, which is weaker.
- `patterns_used`: patterns with ≥ 2 rows that entered the statistic; if it is very
  small the test is weak.

### MAR-plausibility table

```text
  MAR Plausibility  (logistic regression of missingness on observed variables)
  --------------------------------------------------------------------
  Variable        %Missing    LR stat    df    p-value  MAR evidence
  --------------------------------------------------------------------
  age                14.8%     28.412     4     0.0000  YES *
  bmi                18.4%     19.877     4     0.0005  YES *
  glucose             9.0%      3.104     4     0.5406  no
```

Each row: a logistic regression predicting "is this variable missing?" from all
*other* variables (likelihood-ratio test). **Significant = missingness is
predictable from observed data; the observable signature of MAR**, exactly the
structure FIML exploits and corrects for. So counter-intuitively, `YES *` rows are
*good news* for FIML. A `no` row means missingness in that variable is either random
(MCAR-like, also fine) **or** depends on the unobserved value itself (MNAR, not
fine); the data cannot distinguish these two.

### Pattern summary

```text
  Missingness Patterns
  --------------------------------------------------------------------
    1. n= 312 ( 62.4%)  missing: (none)
    2. n=  95 ( 19.0%)  missing: bmi
    3. n=  51 ( 10.2%)  missing: age, bmi
  Total unique patterns: 6
```

What to look for: a small number of dominant patterns suggests a *structural*
mechanism (an instrument that fails, a survey section skipped); good, because
FIML's pattern-grouped likelihood handles it efficiently and you can often name the
cause. Hundreds of scattered single-cell patterns look more like random dropout.
Watch for rows missing *everything*; they contribute nothing and inflate n.

### Missingness correlations

Phi coefficients between missingness indicators; pairs with |r| > 0.3 are printed
with a plain-language direction ("often missing together" / "rarely missing
together"). High positive values identify variables sharing one mechanism (e.g. the
same lab panel); expect them to appear together in the pattern table. This section
is descriptive; it feeds your understanding of the mechanism, not a decision rule.

### Descriptive comparison by missingness

For each variable j with NaNs, Mann-Whitney U tests compare every other variable's
distribution between rows where j is missing vs observed:

```text
  When 'bmi' is missing vs observed:
    age           mean(obs)=  52.310  mean(miss)=  61.804  diff=+9.494  p=0.0004
```

Read: rows lacking `bmi` are markedly older. This is the concrete story behind a
`YES` in the MAR table; it tells you *which* observed variable drives the
missingness and in which direction, which is what you would report in a paper's
missing-data paragraph.

### The verdict, and a decision flowchart in words

The report ends with `[OK]` / `[WARN]` plus a reminder that MAR vs MNAR is
formally untestable. The logic it applies; and the way you should reason:

1. **Little's test not significant?** → Consistent with MCAR. FIML fully valid
   (deletion also unbiased, merely wasteful). Done.
2. **MCAR rejected, and most variables show MAR evidence?** → Missingness is
   predictable from observed data. FIML remains valid under MAR; deletion is biased.
   Proceed with FIML.
3. **MCAR rejected, only some MAR evidence?** → FIML likely still appropriate, but
   ask the domain question: *could the probability of a value being missing depend on
   that value itself* (sicker patients skip visits; high earners hide income)? If
   plausible, run a sensitivity analysis ([Section 6](#6-sensitivity-analysis-misssensitivity)).
4. **MCAR rejected and no MAR evidence anywhere?** → Missingness is not explained by
   anything you observed. MNAR cannot be ruled out; FIML estimates may be biased.
   Sensitivity analysis is mandatory, and domain knowledge should carry the final
   judgement.

**Turning the diagnosis into a model choice.** `MissRecommender` (and the
`recommend_model()` wrapper) reads this same evidence and ranks the model
families *with the reasoning attached*, rather than returning a bare name. It
also reports which columns should be dropped rather than modelled, sets
`copula=True` when the tails demand it, separates outright vetoes (a Gaussian
process above `gp_max_n`, mixed effects without `groups`) from merely low
scores, and marks `MissSensitivity` as required when MNAR cannot be excluded,
which is case 4 above. Read its `evidence_` before its `recommended_`: the
evidence is the part you can check against what you know about how the data
were collected. Full treatment in the user guide, section 6.3, and worked end
to end in `examples/06_Guided_Workflow_Air_Quality.ipynb`.

---

## 5. Explanations: MissExplainer

`MissExplainer` produces two distinct explanations. Keep their semantics separate:

### Value SHAP vs missingness SHAP

**Value SHAP**: `explainer.shap_values(X)`, shape `(n, p)`.
`phi[i, j]` = how much feature j's *observed value* moved observation i's
prediction, relative to the all-features-unknown baseline `expected_value_`.
Because a FIML model computes `E[Y | X_obs]` exactly for any subset of observed
features, the model itself is the exact coalition value function; dropping a
feature from a coalition literally means setting it to NaN and evaluating the
model there. No background dataset, no Monte-Carlo sampling of replacement values.

The coalition is valued on a *continuous* scale, which for a classifier is not
`predict`. A classifier's `predict` returns a hard label, so on an imbalanced
problem every coalition returns the majority class, every Shapley difference is
exactly zero, and the whole attribution collapses to zeros that look like a
genuine finding. The value function is therefore chosen by `output`: `'auto'`
(the default) uses the predicted probability of the positive class for a binary
classifier and the prediction itself for a regressor, `'proba'` and `'log-odds'`
name those scales explicitly, and `'raw'` is the old `predict` behaviour, kept
only for comparison. `value_scale_` reports which was used. A multi-class model
has no single scalar value function, so it requires `class_index` and raises
without it.

- **`phi = 0` for unobserved features.** A NaN feature gets exactly zero credit.
  Whatever the model inferred *about* that feature from its correlated observed
  neighbours is credited to those observed neighbours; they are what you actually
  measured. So in a beeswarm, grey (NaN) dots sit on the zero line by construction.
- **Efficiency guarantee (exact, not approximate):** for every row,
  `phi[i, :].sum() == v(X[i]) − expected_value_`, where `v` is the value
  function named by `value_scale_`, not necessarily `predict`. You can (and should)
  assert this in pipelines. For `p ≤ exact_threshold` (default 15) the values come
  from full 2^p enumeration and are *exact* Shapley values; above that, a
  KernelSHAP-style weighted regression approximates them but the efficiency
  constraint is still enforced exactly.

**Missingness SHAP**: `explainer.miss_shap(X)`, shape `(n, p)`.
`phi_miss[i, j]` = `v(x with j known) − v(x with j = NaN)`, on the same value
scale as above: **the value of observing feature j** for observation i. For features already missing in row i,
"known" means the conditional MVN mean, so you still get an estimate of what
observing it would have been worth. Signs: positive = knowing j raises the
prediction; magnitude = how much predictive information j carries for that row.

**Use `mean |miss_shap|` to prioritise data collection.** It directly answers "which
measurement is worth paying for next?"; a question value SHAP cannot answer,
because value SHAP only scores features you already observed. If `glucose` tops the
miss-SHAP ranking, reducing its missing rate buys the most predictive accuracy per
measurement.

### Reading each plot

All plots use the viridis convention: **yellow = high/positive, purple =
low/negative, teal = neutral, grey = missing (NaN)**.

**Beeswarm** (`plot_beeswarm(shap_values, X)`); global ranking + direction.
- One row per feature, sorted top-down by mean |SHAP|; top row = most influential
  overall.
- Each dot is one observation; horizontal position = its SHAP value; colour = the
  *feature's value* (purple low → yellow high); grey dots = the feature was NaN in
  that row (and therefore sit at SHAP = 0).
- Read direction from the colour gradient: yellow dots on the right = "high values
  of this feature push predictions up" (positive relationship); yellow on the left =
  negative relationship; colours mixed on both sides = interaction or
  non-monotonicity; follow up with a dependence plot.
- Horizontal spread = heterogeneity: a wide row matters a lot for some rows and not
  others.

**Waterfall** (`plot_waterfall(shap_values, X, i=k)`); single-prediction
decomposition. Starts at the dotted grey line (`Baseline = expected_value_`, the
all-unknown prediction) and stacks each feature's bar until the dashed teal line
(`Prediction`). Yellow bars push up, purple bars push down. The feature name and
its value sit together on the axis tick (`glucose = 148.00`) and the signed
contribution sits on the bar itself, which keeps the two readable when several
features have similar magnitudes; a feature that was not observed is marked as
unobserved rather than shown with a value, and its bar has zero length. The
canvas grows with the number of bars and the legend sits below the axes, so
neither the outermost label nor the legend overlaps the data. Beyond
`max_display` features, the remainder are pooled into one
"`k other features`" bar. Because of the efficiency property the bars *exactly*
bridge baseline → prediction, so this plot is a complete audit of one prediction -
use it for case-level explanation ("why did this patient score 0.83?").

**Miss-importance bar chart** (`plot_miss_importance(miss_shap)`); mean
|missingness SHAP| per feature, i.e. the average prediction shift you forfeit by not
observing each feature. Brightest (yellow) bar = most valuable measurement. This is
the data-collection priority list; the numeric labels are in units of the model
output (y units for regressors, probability for classifiers).

**Dependence plot** (`plot_dependence(shap_values, X, feature_idx=j)`); feature
value (x-axis) vs its SHAP value (y-axis), one dot per row. A straight line = linear
effect (expected for linear-family models); colour encodes a second feature
(`interaction_idx`); a vertical colour gradient at fixed x indicates an interaction.
Grey dots have NaN in either the plotted or the colouring feature.

**`explainer.summary(shap_values, miss_shap)`** prints the model type, baseline
`E[f]`, whether the exact or kernel path applies at this p, and ASCII bar rankings of
both importance types; a quick text-only digest of the two bar plots.

### One combined reading

A feature can rank *high* on value SHAP but *low* on miss-SHAP (it is informative
but almost redundant given its correlated neighbours; cheap to leave unmeasured) or
vice versa (individually modest but irreplaceable). Comparing the two rankings is
the most decision-relevant output of the explainer.

---

## 6. Sensitivity analysis: MissSensitivity

FIML assumes MAR. MAR cannot be verified from data
([Section 4](#4-diagnostics-missdiagnostic)), so `MissSensitivity` quantifies the
next best thing: **how wrong would MAR have to be before your conclusion changes?**

### The delta axis

The method is delta-adjustment: missing y values are imputed under MAR, then shifted
by `delta`, and the model is refitted (m imputations per delta, pooled by Rubin's
rules). With the default `standardise=True`, **delta is in units of σ_Y, the standard
deviation of observed y**:

- `delta = 0`: MAR holds; reproduces the original FIML fit.
- `delta = +1`: the unseen y values are on average 1 SD *higher* than MAR predicts
  (e.g. dropouts were doing better than their observed covariates suggested).
- `delta = −1`: 1 SD lower (the classic "sicker patients drop out" scenario).

For context, |delta| = 0.5 is already a fairly strong MNAR mechanism and |delta| = 2
is extreme; the default grid (−3, +3) deliberately over-covers.

### Coefficient curves and the summary table

`coef_curves_` (`(n_delta, p)`) traces each pooled coefficient across the grid;
`sensitivity_table()` returns `[delta_std, coef_0, se_0, coef_1, se_1, ...]` for
custom plotting. `summary()` condenses it:

```text
  Coef    Baseline   at delta=-1   at delta=+1   Tipping pt     Verdict
  ------  --------  ------------  ------------  ------------  ----------
  X0        0.4210        0.3722        0.4698        stable      STABLE
  X1        0.1104        0.0381        0.1827         -1.52      ROBUST
  X2       -0.0312       -0.1039        0.0415         +0.36   SENSITIVE
```

- **Baseline**: pooled estimate at delta ≈ 0 (should match the FIML fit up to
  Monte-Carlo noise from the m imputations).
- **at delta = ∓1**: the estimate under a ±1 σ_Y MNAR departure. Flat across the
  row = insensitive; a steep gradient = the coefficient depends on rows with missing
  y.
- **Tipping point**: the smallest |delta| at which the conclusion changes: by
  default where the coefficient crosses 0 (`method='sign'`); with `method='ci'`,
  where the 95% CI first includes (or excludes) 0; the significance-based version,
  usually the one reviewers care about.

### ROBUST vs SENSITIVE verdicts

`verdict(coef_idx)` applies conventional thresholds to |tipping point|:

| Verdict | Threshold | Meaning |
|---|---|---|
| `ROBUST` | never tips in the grid, or tips at ≥ 1.0 σ_Y | the conclusion holds across every delta examined, or only an implausibly large MNAR shift overturns it |
| `MILD` | 0.5 to 1.0 σ_Y | a strong but conceivable MNAR mechanism could overturn it |
| `SENSITIVE` | < 0.5 σ_Y | fragile; a modest MNAR departure flips the sign/significance |

There are three labels, not four. A coefficient that never tips is reported as
`ROBUST` with the delta range quoted, rather than under a separate heading.

A `SENSITIVE` verdict does not mean the finding is wrong; it means the finding
*rests on the MAR assumption* and you must argue for MAR on substantive grounds (or
soften the claim).

`MissSensitivity` requires a coefficient-bearing estimator. Given one that
exposes no `coef_` after fitting, such as `MissBayesRegressor`, `MissSupport*`,
`MissNeighbors*` or `MissGaussian*`, `fit()` raises `AttributeError` naming the
alternatives. It previously substituted a zero vector, so every row read
`0.0000` and looked maximally robust, which is the most dangerous way for this
particular tool to fail. Pass `feature_names` to `fit()` to have the summary
table label rows with the real variable names.

### How to report it in a paper

> "Because the missing-at-random assumption is untestable, we performed a
> delta-adjustment sensitivity analysis (Carpenter & Kenward, 2013): imputed missing
> outcomes were shifted by δ ∈ [−3, 3] SD of Y and the model refitted with m = 10
> imputations per δ, pooled by Rubin's rules. The effect of X1 remained positive
> until δ = −1.52 SD, i.e. only if unobserved outcomes averaged more than 1.5 SD
> below their MAR-predicted values would the conclusion reverse. We judge such a
> departure implausible given [domain reason]; the finding is robust to moderate
> MNAR."

Report: the delta range and m, the tipping point per key coefficient (or "stable"),
and a substantive argument about which delta values are plausible. Never report only
the verdict label.

---

## 7. Multiple-imputation outputs: MissImputer

`MissImputer` draws m completed datasets from the FIML-estimated joint MVN.
`combine()` (or the one-call `fit_transform_combine(X, y, estimator, param='coef_',
param_var='se_')`) pools per-dataset estimates by **Rubin's rules** and returns a
dict:

| Key | Formula | Interpretation |
|---|---|---|
| `estimate` | mean of the m estimates | the pooled point estimate; report this |
| `within_var` (W) | mean of the m variances | average sampling uncertainty *if the imputed values were real* |
| `between_var` (B) | variance across the m estimates | extra uncertainty *because* values were missing; how much the imputations disagree |
| `total_var` (T) | `W + (1 + 1/m)·B` | the honest total variance |
| `se` | `sqrt(T)` | pooled standard error; always ≥ the naive single-imputation SE |
| `df` | `(m−1)(1 + 1/r)²`, `r = (1+1/m)B/W` | Rubin/Barnard degrees of freedom for the t reference |
| `p_value` | 2·t-tail at `df` (scalar estimates only) | pooled two-sided test vs 0 |
| `m` |; | number of imputations actually pooled |

Confidence interval: `estimate ± t_{df, 1−α/2} · se` (use `scipy.stats.t.ppf` with the
returned `df`: for small df the t quantile is meaningfully larger than 1.96).

### Fraction of missing information (FMI) intuition

The ratio `λ ≈ (1 + 1/m)·B / T` estimates the **fraction of missing information**:
how much of your total uncertainty about this parameter is due to missingness rather
than finite sampling.

- `λ ≈ 0.05`: missingness barely matters for this parameter; m = 5 to 10 suffices.
- `λ ≈ 0.3`: substantial; use m ≥ 20 and expect visibly wider CIs than a
  complete-data analysis.
- `λ ≈ 0.7`: the parameter is mostly determined by what you *didn't* observe;
  conclusions are dominated by the imputation model; pair with a sensitivity
  analysis.

Symptoms of large λ: `df` collapses toward `m − 1` (the t interval widens), and
re-running with a different `random_state` visibly moves the estimate. The fix is
more imputations (larger `m`), and `posterior=True` for fully "proper" imputation
that also propagates parameter uncertainty.

`fit_transform_combine` additionally returns `'fitted_estimators'`: the m fitted
models, useful for pooling secondary quantities yourself.

### `MissImputer.summary()`

Prints m, the variable count (flagging whether y was included), EM iterations to
convergence, the `posterior` flag, then the fitted marginal means/SDs per variable
and the full correlation matrix of the joint MVN. Sanity checks to run by eye: the
means/SDs should match your descriptive statistics on observed data, and the
correlation matrix is the machinery generating imputations; a near-zero row means
that variable's imputations are essentially marginal draws (uninformed by the rest).

---

## 8. Ensemble outputs: MissEnsemble

`MissEnsemble.summary()` reports the task, member count, bootstrap settings, and:

- **Member weights** (`weights_`); normalised to sum to 1; predictions are the
  weighted mean (regression) or weighted-mean probability then argmax
  (classification). In homogeneous bagging all weights are equal and the summary
  collapses to one line (`100 x MissRidgeRegressor, weight 0.0100 each`). In
  heterogeneous ensembles the per-member table shows name, type, weight; a member
  with weight 0.5 contributes half of every prediction, so check the weights reflect
  your intent (they are *your inputs*, not learned).
- **OOB scores** (`oob_scores_`, with `oob_score=True`); each member scored on the
  rows its bootstrap sample never drew: R² for regression, accuracy for
  classification. This is an honest, no-extra-holdout generalisation estimate. For
  large homogeneous ensembles the summary shows mean/std/min/max across members -
  the mean estimates generalisation, the spread shows member instability. In
  heterogeneous ensembles, compare members' OOB columns to decide weights for the
  next fit. `nan` = scoring failed for that member (e.g. degenerate OOB sample).
- **Feature importances**: weighted average across members that expose
  `feature_importances_`, with a `+/-` std column: a large std means members
  disagree about that feature (often a sign of correlated features being used
  interchangeably).

### `predict_interval` from member spread: epistemic, not aleatoric

The ensemble interval is the empirical `[α/2, 1−α/2]` quantile band of the B member
predictions at each row. **It measures epistemic (model) uncertainty only**: disagreement among refits/models about the *mean* prediction; and contains **no
aleatoric noise** (the irreducible scatter of individual outcomes around that mean).

Practically:

- Use it to answer *"how stable is this prediction under resampling/model choice?"*
  A wide band = the prediction is data-fragile there (extrapolation region, small
  local sample). A narrow band means the models agree; **not** that a future
  observation will land inside it.
- Do **not** quote it as a prediction interval for new outcomes: it will
  systematically under-cover (often drastically; with enough data, ensemble members
  converge and the band shrinks toward zero width while outcome noise does not).
  For outcome-level intervals use a FIML regressor's own
  `predict_interval` ([Section 3](#3-predictions--uncertainty)), which includes the
  residual variance.
- For homogeneous bagging the band approximates a bootstrap confidence interval for
  the conditional mean; for heterogeneous ensembles it is model-disagreement, a
  looser notion. The summary footer prints which interpretation applies to your fit.

---

## 9. Benchmark figures

The benchmark harness compares six strategies on the *same* NaN-bearing data
(no method ever sees the unmasked values): **Drop Rows** (listwise deletion),
**Drop Cols**, **Mean Imputation**, **KNN Imputation**, **MICE (Iterative)**, and
**MissLearn (FIML)**. Run it from `benchmarks/scripts/run_benchmark.py` and
`run_sweep.py`, or interactively through `Benchmark_Explorer.ipynb` and
`Sweep_Explorer.ipynb` in `benchmarks/`. All figures share the viridis palette
with FIML at the bright-yellow end and drawn last, on top.

Each run writes to its own directory under `benchmarks/results/`. The two
harnesses do not save the same things, which is worth knowing before you go
looking for a file that was never written:

* `run_benchmark.py --save` writes both figures and numbers: raw per-fold
  scores to `{model}_{task}_raw.csv` and mean±std tables to
  `{model}_{task}_summary.csv`, alongside the PNGs.
* `run_sweep.py --save` writes **figures only**. It prints a crossover table
  (the lowest rate at which FIML beats each baseline) to standard output, but
  the per-rate aggregates behind the curves stay local to the run and are not
  exported. To get sweep numbers in a form you can tabulate, capture stdout, or
  call `benchmark_core.run_cv_at_rate` and `aggregate_folds` yourself, which is
  exactly what the script does internally.

Metrics and directions: regression RMSE ↓, MAE ↓, R² ↑; classification Accuracy ↑,
ROC-AUC ↑, F1 ↑, Brier ↓. Every panel title repeats "(lower is better)" or
"(higher is better)"; always check it before comparing bar heights.

### Bar charts (`plot_bars`)

One figure per dataset; one panel per metric. Bar height = **mean over the K CV
folds** (typically 5); black error bars = **±1 standard deviation across folds**
(not a standard error, not a CI). Numeric labels sit above each bar.

Reading rule: a method is convincingly better only if the gap between means is large
relative to the error bars. Overlapping ±1 SD whiskers at K = 5 folds means the
ordering could plausibly flip on a re-split; corroborate with the strip plot and
t-test table.

### Strip plots (`plot_strips`)

Same layout, but showing **every fold as an individual dot** (jittered) with a thick
horizontal bar at the fold mean. This is the honesty check on the bar chart:

- Tight dot clusters, clearly separated between methods → the bar-chart ordering is
  real.
- Interleaved dot clouds → the mean difference is noise-level.
- One stray dot dragging a mean → a single pathological fold (e.g. Drop Rows left
  too few complete cases); the bar chart alone would have hidden it.

The accompanying paired t-test table (`compute_ttests`) tests FIML vs each baseline
per metric; but at 5 folds power is low; treat p-values as indicative, and the
`FIML wins` column (sign of the mean difference) as the headline.

### Gain heatmaps (`plot_gain_heatmap`, `plot_sweep_gain_heatmap`)

Cells show **FIML's percentage gain over each baseline**, sign-corrected so
**positive = FIML better regardless of metric direction**:

- higher-is-better metrics: `(FIML − baseline) / |baseline| × 100`
- lower-is-better metrics: `(baseline − FIML) / |baseline| × 100`

Colour: **yellow = FIML wins, purple = baseline wins**, white/teal ≈ tied
(diverging scale centred at 0). In the benchmark version, rows are dataset ×
baseline and columns are metrics; in the sweep version, rows are missing rates and
columns baselines (one figure per dataset × metric), so you can watch the yellow
deepen as missingness rises.

Two honest-rendering details to keep in mind:

1. The denominator is floored at 0.01, so a gain against a near-zero baseline (a
   collapsed R², say) is *understated*, not exaggerated. A cell reading `+900` next
   to a baseline R² of ~0 really means "the baseline broke".
2. The colour scale saturates at the 90th percentile of |gain|; extreme cells clip
   to full yellow/purple rather than washing out the rest. Read the printed numbers,
   not just the hue, for the extremes.

### Sweep line plots (`plot_sweep_lines`)

Performance vs missing rate: x-axis = injected MAR rate (e.g. 5%-50%), y-axis = the
metric, one line per method (FIML solid with star markers, drawn on top; baselines
dashed/dotted), **shaded band = ±1 SD across folds**. Data is re-injected fresh at
each rate from the complete matrix, so the rates are directly comparable.

What to look for:

- **Slopes.** Every method degrades as missingness rises; the question is how fast.
  A flat FIML line over steeply falling baselines is the core FIML story.
- **Divergence point.** Lines typically bundle at low rates (little information is
  missing, all strategies cope) and fan out beyond ~20 to 30%.
- **Band overlap.** As with the bars, separation matters only relative to the bands.

Two companion figures: **degradation plots** (`plot_sweep_degradation`) re-express
each line as *change from that method's own 0%-missing baseline*, sign-flipped so
that for every metric "closer to zero = held up better"; this removes level
differences between methods and isolates robustness to missingness. **Rank plots**
(`plot_sweep_rank`) show each method's rank (1 = best, higher on the plot = better;
the y-axis is inverted) at each rate; a scale-free summary; watch for FIML's line
climbing to rank 1 and staying there as the rate grows.

### Crossover tables (`compute_sweep_crossover`)

Indexed by dataset × metric × baseline, the `Crossover rate` column is **the lowest
missing rate at which FIML first beats that baseline** on fold-mean performance:

- `5%` (the lowest swept rate) with `FIML wins all = Yes`: FIML dominates
  everywhere in the sweep.
- `20%` with `Partial`: the baseline is competitive at low missingness; FIML pulls
  ahead from 20% on. This is the number to quote when advising "at what missingness
  level should I switch to FIML?"
- `Never` (`No`); the baseline held its lead across the whole sweep for that
  metric/dataset (most often seen for cheap metrics like accuracy at low rates).

Caveat: it reports the *first* crossing on fold means, without a significance
requirement and without checking monotonicity; always sanity-check the
corresponding sweep line plot before quoting a crossover.

### Missing-profile plots (`plot_missing_profile`) and companions

Per-dataset horizontal bars of each feature's missing percentage, sorted descending.
**Teal bars = columns with injected missingness; grey "complete" bars = the ~40% of
columns deliberately left clean** (so the Drop Cols baseline always has something to
train on); the **dashed purple vertical line = the overall missing rate** the sweep
labels refer to. Use it to understand what "25% missing" actually means: the
injection concentrates missingness (at 2× the rate above the driver column's median
- an explicitly MAR mechanism), so affected columns individually run well above the
overall rate.

Two supporting figures round out the data profile: **complete-case retention**
(`plot_cc_retention`); purple bars = rows Drop Rows keeps, yellow bars = rows it
discards (the visual indictment of listwise deletion; compare against the printed
MCAR-theoretical retention `(1 − rate)^p`), and **class balance**
(`plot_class_balance`); positive/negative frequencies per classification dataset,
with the 50% line marked; the more imbalanced the dataset, the more you should weight
AUC/Brier over accuracy when reading the other figures.

---

*Guide generated for the MissLearn package; covers `MissLearn/` model outputs and
`benchmarks/benchmark_core.py` figures as of July 2026.*
