# MissLearn: Full Information Maximum Likelihood Models with Native Missing Data Support

**Version:** 0.9.2  
**Package:** `MissLearn`  
**Import:** `from MissLearn import MissLinear, MissLogistic, MissRidge, ...`

---

## Table of Contents

1. [Motivation and Philosophy](#1-motivation-and-philosophy)
2. [Missing Data Theory](#2-missing-data-theory)
3. [Full Information Maximum Likelihood](#3-full-information-maximum-likelihood)
4. [MissLinear: Design and Method](#4-misslinear-design-and-method)
5. [MissLogistic: Design and Method](#5-misslogistic-design-and-method)
6. [Shared Infrastructure](#6-shared-infrastructure)
7. [Parameter Estimation and Optimization](#7-parameter-estimation-and-optimization)
8. [Inference and Interpretability](#8-inference-and-interpretability)
9. [Prediction with Missing Features at Inference Time](#9-prediction-with-missing-features-at-inference-time)
10. [Extended Model Families](#10-extended-model-families)
11. [Software Architecture and Performance](#11-software-architecture-and-performance)
12. [API Reference](#12-api-reference)
13. [Usage Examples](#13-usage-examples)
14. [Comparison with Alternative Approaches](#14-comparison-with-alternative-approaches)
15. [Limitations and Assumptions](#15-limitations-and-assumptions)
16. [References](#16-references)

---

## 1. Motivation and Philosophy

Real-world datasets, particularly in the biological and clinical sciences, are rarely complete. Instruments fail, assays are cost-prohibitive, patients miss follow-up appointments, and sensors malfunction. The result is data that is structurally incomplete -- not by accident, but as an intrinsic property of how the data was collected.

The conventional response to missing data is one of three approaches: listwise deletion (discard any observation with a missing value), mean imputation (replace each missing value with the column mean), or model-based imputation (predict missing values from a secondary model and fill them in before fitting the primary model). All three approaches share a common flaw: they treat missingness as a defect to be corrected rather than a structural property of the data to be respected.

MissLearn takes a different position. Every observation contributes exactly the information it contains -- no more, no less. An observation with three of five predictors observed contributes a three-dimensional likelihood. An observation with only the outcome observed contributes a one-dimensional likelihood. An observation that is entirely missing contributes nothing. At no point is a value invented, imputed, or assumed.

This approach is not an approximation. It is, under the Missing at Random assumption, the statistically optimal procedure for extracting information from incomplete data.

---

## 2. Missing Data Theory

### 2.1 Missing Data Mechanisms

The validity of any missing data method depends critically on why data are missing. Rubin (1976) established the canonical taxonomy of missing data mechanisms.

**Missing Completely at Random (MCAR).** The probability that a value is missing is independent of both the observed and unobserved data. Listwise deletion is unbiased under MCAR but inefficient, as it discards information from partially observed rows. MCAR is a strong and often unrealistic assumption.

**Missing at Random (MAR).** The probability that a value is missing may depend on the observed data, but not on the value that is missing. For example, a patient's blood pressure measurement may be more likely to be missing if they are older (observed), but not because of the value of the blood pressure itself (unobserved). MAR is the assumption required by FIML and by most principled missing data methods. It is substantially more realistic than MCAR and applies to the majority of biomedical missing data scenarios.

**Missing Not at Random (MNAR).** The probability that a value is missing depends on the value itself. For example, high-income individuals may be less likely to report their income. MNAR requires explicit modeling of the missingness mechanism and is outside the scope of standard FIML. MissLearn's `MissSensitivity` class provides delta-adjustment sensitivity analysis for exploring plausible MNAR departures.

### 2.2 Why Conventional Approaches Fail Under MAR

Under MAR, listwise deletion produces biased estimates unless the data happen to be MCAR. This is because the complete cases are not a random sample of the original population -- they are a self-selected subset whose selection criterion may be correlated with the outcome.

Imputation-based methods introduce a secondary source of uncertainty and, when the imputation model is misspecified, introduce systematic bias into the primary analysis. Multiple imputation mitigates but does not eliminate this problem and requires careful attention to the compatibility of the imputation and analysis models.

FIML avoids both problems by working directly with the observed data likelihood, integrating over unobserved values using the joint model. Under MAR, FIML produces maximum likelihood estimates that are asymptotically unbiased and efficient.

---

## 3. Full Information Maximum Likelihood

### 3.1 Conceptual Foundation

Full Information Maximum Likelihood, introduced into the structural equation modeling literature by Arbuckle (1996), is a method for estimating model parameters directly from incomplete data without imputation. The term "full information" refers to the fact that the method uses all available information in the data -- every observed value in every row -- rather than restricting analysis to complete cases.

The key insight is that for an observation with some values missing, the likelihood contribution of that observation is not zero. It is the probability of observing exactly what was observed, marginalized over all possible values of what was not observed. Specifically, for observation i with observed values z_obs and missing values z_mis:

```
p(z_obs_i ; theta) = integral p(z_obs_i, z_mis_i ; theta) dz_mis_i
```

When the joint distribution is multivariate normal, this marginal distribution is itself a multivariate normal -- with mean and covariance given by the corresponding submatrix of the joint parameters. This means the integral has a closed-form solution and no numerical integration is required.

### 3.2 The FIML Objective

Given n observations and model parameters theta, the FIML log-likelihood is:

```
L(theta) = sum_{i=1}^{n} log p(z_obs_i ; theta)
```

where o(i) denotes the set of indices that are observed for observation i, and the density is evaluated only on those dimensions. Parameters are chosen to maximize this quantity, or equivalently to minimize the negative log-likelihood.

### 3.3 Information Recovery

A key property of FIML is that it recovers information from partially observed rows that listwise deletion would discard entirely. Consider a dataset where 30% of predictor values are missing uniformly at random, producing roughly 17% complete cases if there are five predictors (0.7^5 ≈ 0.17 complete). FIML retains contributions from all rows. Each partial row contributes its observed subvector to the marginal likelihood, which informs estimation of the joint covariance structure and, through it, the regression parameters.

---

## 4. MissLinear: Design and Method

### 4.1 Model Specification

MissLinear models the full vector Z = (Y, X_1, ..., X_p) as a joint multivariate normal distribution:

```
Z ~ N(mu, Sigma),   Z in R^{p+1}
```

where mu is the (p+1)-dimensional joint mean vector and Sigma is the (p+1) x (p+1) joint covariance matrix. The first element of Z is the response Y; the remaining p elements are the predictors.

This is the correct generative model for linear regression when predictors are also allowed to be random variables -- which they must be treated as when they are subject to missing values.

### 4.2 Likelihood Contribution per Observation

For observation i, let o(i) be the index set of non-NaN entries in z_i. The likelihood contribution is:

```
log p(z_obs_i) = log N(z_obs_i ; mu_{o(i)}, Sigma_{o(i), o(i)})
```

where mu_{o(i)} and Sigma_{o(i), o(i)} are the subvector and submatrix of mu and Sigma indexed by o(i). The total log-likelihood is the sum over all observations:

```
L(mu, Sigma) = sum_{i=1}^{n} log N(z_obs_i ; mu_{o(i)}, Sigma_{o(i), o(i)})
```

This is a closed-form expression -- no numerical integration is required regardless of how many values are missing or the dimensionality of the problem.

### 4.3 Missingness Patterns and Their Contributions

| Observed variables | Contribution to likelihood |
|---|---|
| Y and all X | Full (p+1)-dimensional normal log-PDF |
| Y and some X | Marginal normal over the observed subset |
| Y only | Univariate normal log-PDF on Y alone |
| Some X, no Y | Informs the X block of mu and Sigma; constrains beta indirectly |
| All missing | Excluded; contributes zero information |

No observation is discarded. No observation contributes information it does not possess.

### 4.4 Regression Coefficient Recovery

Regression coefficients are not directly optimized. Once mu and Sigma have been estimated by FIML, the linear regression coefficients are recovered analytically via the standard partitioned normal formulae. Partition Sigma as:

```
Sigma = [[Sigma_YY,  Sigma_YX],
         [Sigma_XY,  Sigma_XX]]
```

The regression coefficients, intercept, and residual variance are then:

```
beta      = Sigma_XX^{-1} Sigma_YX
intercept = mu_Y - beta' mu_X
sigma^2   = Sigma_YY - Sigma_YX' Sigma_XX^{-1} Sigma_XY
```

This is the same formula as the population OLS formula applied to the FIML-estimated parameters. No separate optimization of beta is required, and the estimates inherit the statistical properties of the FIML estimator.

The system Sigma_XX beta = Sigma_YX is solved via `numpy.linalg.solve` rather than explicit matrix inversion, which is numerically more stable and avoids amplifying rounding errors in near-singular covariance matrices.

### 4.5 Interpretation of sigma^2

The scalar sigma^2 = Sigma_YY - Sigma_YX' Sigma_XX^{-1} Sigma_XY is the conditional variance of Y given all X. It is equivalent to the OLS residual variance in the complete-data case and represents the irreducible variance in Y after accounting for all predictors. It determines prediction interval width for complete-feature observations at inference time.

---

## 5. MissLogistic: Design and Method

### 5.1 Model Specification

MissLogistic models the joint distribution of Y and X as a semi-parametric combination:

```
Y | X ~ Bernoulli(sigma(beta_0 + beta' X))
X     ~ N(mu_X, Sigma_X)
```

where sigma is the logistic sigmoid function. The binary outcome Y is linked to the linear predictor via the logistic function; the predictors X follow a multivariate normal distribution. This joint model is standard in the FIML logistic regression literature (Ibrahim, 1990; Ibrahim et al., 2005).

Unlike MissLinear, the joint distribution of (Y, X) is not Gaussian, so regression coefficients beta must be included in the parameter vector and optimized directly alongside the predictor distribution parameters (mu_X, Sigma_X).

### 5.2 Likelihood Contribution per Observation

For observation i with observed predictor subset x_obs_i and outcome y_i:

```
log p(y_i, x_obs_i) = log P(y_i | x_obs_i) + log p(x_obs_i)
```

The second term, log p(x_obs_i), is the marginal normal density on the observed predictors -- identical to the corresponding term in MissLinear, closed form and exact.

The first term, log P(y_i | x_obs_i), requires marginalization over the missing predictors:

```
P(y=1 | x_obs) = integral sigma(beta_0 + beta' x) p(x_mis | x_obs) dx_mis
```

where p(x_mis | x_obs) is the conditional distribution of missing given observed predictors -- also a multivariate normal, with parameters derived from the partitioned covariance (see Section 6.1).

For observations where y_i is missing, only the predictor density term is evaluated, allowing predictor-only rows to inform the X distribution.

### 5.3 Reduction to a One-Dimensional Integral

The marginalization integral above has dimension equal to the number of missing predictors. Naively, this would require multi-dimensional numerical integration, which becomes computationally prohibitive for even moderate numbers of missing features.

The critical simplification is that sigma(beta' x) depends on the missing features only through the scalar linear combination:

```
s = beta_mis' x_mis
```

Because x_mis | x_obs is multivariate normal, and s is a linear transformation of x_mis, s is itself normally distributed:

```
s | x_obs ~ N(beta_mis' mu_c, beta_mis' Sigma_c beta_mis)
```

where mu_c and Sigma_c are the conditional mean and covariance of x_mis given x_obs (derived in Section 6.1). The multi-dimensional integral therefore reduces to a one-dimensional expectation regardless of how many features are missing:

```
P(y=1 | x_obs) = E_t[sigma(a + t)],   t ~ N(0, v)
```

where:

```
a = beta_0 + beta_obs' x_obs + beta_mis' mu_c
v = beta_mis' Sigma_c beta_mis
```

This one-dimensional logistic-normal expectation is evaluated by Gauss-Hermite quadrature (Section 6.2). The reduction is exact -- it is not an approximation.

### 5.4 Odds Ratios and Interpretation

Because the logistic regression coefficients beta are directly parameterized in the model, they have their standard interpretation as log-odds. The odds ratio for predictor j is:

```
OR_j = exp(beta_j)
```

representing the multiplicative change in the odds of Y=1 for a one-unit increase in X_j, holding all other predictors constant. This interpretation is unchanged by the presence of missing data; FIML simply ensures the estimates are derived from all available information.

### 5.5 L2 Regularization on beta

Maximum likelihood estimation of logistic regression coefficients is undefined when the classes are perfectly or quasi-perfectly linearly separable. In such cases, the MLE for beta diverges to infinity as the optimizer attempts to achieve zero training loss. This pathology is unrelated to missing data and affects complete-data logistic regression identically.

MissLearn addresses this by adding an optional L2 penalty on the slope coefficients:

```
L_penalized(theta) = L_FIML(theta) - (lambda/2) * ||beta_{1:p}||^2
```

The intercept is excluded from the penalty. The penalty parameter lambda is controlled by the `l2_reg` argument (default 0.01). Setting `l2_reg=0` recovers exact FIML for well-conditioned problems. The X distribution parameters mu_X and Sigma_X are never penalized.

---

## 6. Shared Infrastructure

### 6.1 Conditional Normal Distribution

Both models use the partitioned normal formulae to compute the conditional distribution of unobserved variables given observed ones. For Z ~ N(mu, Sigma), partitioned as Z = (Z_obs, Z_mis), the conditional distribution is:

```
Z_mis | Z_obs = z_obs ~ N(mu_c, Sigma_c)

mu_c    = mu_mis + Sigma_{mis,obs} Sigma_{obs,obs}^{-1} (z_obs - mu_obs)
Sigma_c = Sigma_{mis,mis} - Sigma_{mis,obs} Sigma_{obs,obs}^{-1} Sigma_{obs,mis}
```

The implementation solves the linear system Sigma_{obs,obs} K' = Sigma_{mis,obs}' for K rather than computing the matrix inverse explicitly. The resulting Sigma_c is symmetrized by averaging with its transpose to eliminate floating-point asymmetry introduced by the solve operation.

### 6.2 Gauss-Hermite Quadrature for the Logistic-Normal Integral

The one-dimensional expectation E_t[sigma(a + t)] with t ~ N(0, v) is evaluated by Gauss-Hermite quadrature. The physicist's Gauss-Hermite formula states:

```
integral f(x) exp(-x^2) dx = sum_{i=1}^{Q} w_i f(x_i)
```

Substituting t = sqrt(2v) * s (so that t ~ N(0, v) when s has the Hermite distribution) gives:

```
E_t[sigma(a + t)] = (1/sqrt(pi)) * sum_{i=1}^{Q} w_i * sigma(a + sqrt(2v) * s_i)
```

where s_i are the Gauss-Hermite nodes and w_i are the corresponding weights. For nearly deterministic cases (v < 1e-12), the integral degenerates to sigma(a) and is evaluated directly.

This rule is used only while v is small, where **Q = 20** is accurate to about 1e-11. It cannot be pushed further: in the quadrature variable the integrand is a step of width about 1/sqrt(2v), so its error grows with v whatever Q is, reaching 0.004 in a probability at v = 25 and 0.03 at v = 100, and raising Q from 20 to 640 still leaves more than 1e-4 there. Since v is the variance the missing features contribute, it grows with both their number and the size of the coefficients. Wide variances are therefore taken by a step-plus-remainder split, accurate to about 1e-13 for any v, described in the Computational Guide. Fitting and prediction share the rule, so `n_quadrature` sizes the small-variance branch and no longer changes the answer.

**Vectorised batch evaluation.** The integral is evaluated simultaneously for all observations sharing the same missingness pattern in a single matrix operation:

```
# a_vals: (n_group,) array of per-row linear predictor offsets
# t:      (Q,)      array of scaled quadrature nodes
vals   = sigmoid(a_vals[:, None] + t[None, :])   # (n_group, Q)
result = vals @ weights / sqrt(pi)                # (n_group,)
```

This eliminates the per-row Python loop that was present in earlier versions, giving a 30 to 50% speedup on classification with partial observations.

### 6.3 Cholesky Parameterization of the Covariance Matrix

Covariance matrices must be symmetric positive definite throughout optimization. Rather than constraining Sigma directly, both models parameterize Sigma through its Cholesky factor L, where Sigma = L L'. L is lower triangular; its diagonal entries must be strictly positive.

The off-diagonal entries of L are stored directly in the parameter vector (unconstrained). The diagonal entries are stored as their natural logarithms, which maps the positive real line to the entire real line, eliminating the need for bound constraints during optimization. At each evaluation, the diagonal entries are recovered by exponentiation:

```
L_{ii} = exp(theta_{ii})
```

This parameterization guarantees that Sigma = L L' is symmetric positive definite for any finite parameter vector, allowing the use of gradient-based unconstrained optimizers without constraint handling.

### 6.4 Multivariate Normal Log-PDF

Log-densities are computed via Cholesky decomposition rather than direct matrix inversion, which is numerically stabler and computationally efficient:

```
log N(x; mu, Sigma) = -k/2 log(2pi) - sum_i log(L_ii) - (1/2) ||L^{-1}(x - mu)||^2
```

where L is the Cholesky factor of Sigma. The triangular solve L v = (x - mu) is solved for v; the Mahalanobis distance is then ||v||^2 and the log-determinant is 2 sum_i log(L_ii). For grouped observations (all rows sharing the same missingness pattern), a single Cholesky decomposition is shared via `mvn_logpdf_batch`, replacing per-row Cholesky with a single batched triangular solve.

---

## 7. Parameter Estimation and Optimization

### 7.1 Parameter Vectors

**MissLinear:**

```
theta = [mu (p+1), L_vec (p+1)(p+2)/2]
```

Total parameters: (p+1) + (p+1)(p+2)/2

**MissLogistic:**

```
theta_r = [intercept, beta (p)]
```

Total parameters optimised: p+1

MissLogistic does **not** optimise jointly over the X-moments. Those are obtained first by EM (section 7.2) and then held fixed, so the quasi-Newton stage searches only the (p+1)-dimensional regression vector. The objective closes over the per-pattern conditional moments computed from the EM solution, which is why the search space is so much smaller than the parameter count of the underlying joint model suggests.

In both cases, all covariance parameters are stored as log-diagonal Cholesky vectors.

### 7.2 Initialization

Neither model initializes from complete cases. Complete-case moments are consistent only under MCAR, and on the data these models are built for the complete cases can be a small and unrepresentative subset, so the starting point would be both biased and unstable.

Instead both models begin with EM on the joint matrix, through the internal `_JointMVNFitter` (`max_iter=500`, `tol=1e-10`, `reg=1e-8`). EM uses every observed entry, so the starting point is the full-information solution rather than an approximation to it.

For MissLinear this fitter is run on the joint matrix Z = [y, X] and returns mu_0 and Sigma_0 directly; the covariance is symmetrised and given a 1e-10 ridge for positive definiteness. For MissLogistic the same EM stage supplies the X-moments, which are then treated as known while the regression vector is optimised.

### 7.3 Optimization Algorithm

The likelihood is not maximised by quasi-Newton search alone. The FIML MLE of a joint MVN is available far more cheaply by EM, so EM does essentially all of the work and L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bound constraints, from `scipy.optimize.minimize`) only polishes the result. In practice the polish terminates after one or two iterations, because it starts at the optimum. L-BFGS-B is the right tool for that role: it needs only function and gradient evaluations, no Hessian, and is reliable on smooth log-likelihoods.

This has a consequence for inference that is easy to miss, and is dealt with in section 7.4: because the polish barely moves, the optimizer's inverse-Hessian approximation never accumulates any curvature information and is close to the identity, so it cannot be used for standard errors.

Convergence is controlled by two separate criteria, and they are not the same number. The scipy options are `ftol = tol` on the relative function change and `gtol = tol * 1e-2` on the gradient norm, with `tol` defaulting to **1e-7**, so the gradient criterion is a hundred times tighter than the function criterion. Earlier versions used 1e-9 throughout; the looser setting converges 10 to 30% faster without any statistically meaningful loss of accuracy. The maximum number of iterations is controlled by `max_iter` (default 2000).

**Warm-starting between successive fits.** When `warm_start=True`, the parameter vector from the previous `fit()` call is reused as the optimizer starting point instead of re-initializing. The vector is stored in `_theta_opt_` after each fit, and the saving is real when the same estimator is refitted on related data, for example along a regularisation path or when refitting after adding rows.

It does **not** apply across cross-validation folds, and deliberately so. Successive training sets overlap by (k-1)/k, so a fold started from the previous fold's parameters would be initialised from an optimum fitted partly on its own held-out rows. `miss_cross_val_score` and `miss_cross_validate` therefore call `_strip_warm_start_state` on each per-fold estimator copy (`_crossval.py`), which deletes `_theta_opt_` from the estimator and from any template nested inside `MissMulticlass`, `MissEnsemble` or `MissPreprocessor`. Every fold starts from its own initialisation. The cost is optimizer iterations; the alternative is a fold that has seen its own test rows, which is not a trade worth making.

### 7.4 Numerical Stability Safeguards

Several safeguards prevent numerical failures from corrupting gradient estimates or producing non-finite log-likelihoods:

**Cholesky parameterization.** Ensures Sigma remains positive definite at every optimizer step.

**Adaptive jitter.** When initializing covariance matrices from complete-case data, the minimum eigenvalue is checked via `eigvalsh`. If the matrix is near-singular, a jitter of `max(1e-6, -λ_min + 1e-4)` is added to the diagonal rather than a fixed 1e-6. This prevents Cholesky failures on pathological datasets with near-collinear features.

**Eigenvalue-based Cholesky for GP models.** `MissGaussianRegressor` and `MissGaussianClassifier` use an eigenvalue-clipping strategy as the primary Cholesky approach: negative eigenvalues are clamped to zero before attempting factorization, avoiding the multi-step escalating-jitter loop. A fast path (direct Cholesky with base jitter) is tried first; the eigenvalue path activates only when the fast path fails.

**Log-probability clamping.** Log-probabilities in MissLogistic are computed as `log(max(p, 1e-300))` to prevent log(0) errors near the boundary of the parameter space.

**Sigmoid stability.** The sigmoid is computed via `scipy.special.expit`, which is numerically stable for all finite inputs including large positive and negative values.

---

## 8. Inference and Interpretability

### 8.1 Standard Errors

Standard errors are derived from the curvature of the log-likelihood at the optimum, using the observed Fisher information matrix (negative Hessian of the log-likelihood):

```
Var(theta_hat) = I(theta_hat)^{-1} = [-H(theta_hat)]^{-1}
```

Standard errors are computed from a numerical Hessian of the reduced conditional likelihood over the regression block only (the fitted X-moments are held fixed), which is small enough to make the exact computation cheap. The optimizer's own low-rank inverse-Hessian approximation is never used for inference; when the optimiser starts at or near the optimum (as the two-stage fits do) that approximation is essentially the identity matrix. The numerical Hessian uses the symmetric central-difference formula:

```
H_{ij} = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4 eps^2)
```

with eps = 1e-4. The Hessian is regularized with 1e-8 * I before inversion.

**MissLogistic:** Beta appears directly in the first p+1 entries of theta, so the standard errors for beta are read directly from the diagonal of the inverted Hessian.

**MissLinear:** Beta is not part of theta; it is derived from (mu, Sigma) post-optimization, so its standard errors cannot be read off the joint parameterisation.

They are also not taken from the optimizer. As section 7.3 explains, the L-BFGS-B stage only polishes an EM solution and usually stops after one or two iterations, so its inverse-Hessian approximation is essentially the identity and carries no curvature information. Using it would produce standard errors that look plausible and mean nothing.

Instead the standard errors come from the curvature of the *conditional* likelihood of y given the observed part of X, evaluated at the fitted moments: a small (p+2)-dimensional numerical Hessian of a fully vectorised NLL in (intercept, beta, log sigma). This is the same convention the penalized and mixed families use, so standard errors are comparable across the library.

A delta-method routine, `_compute_se`, used to sit unused in `_linear.py` alongside this. It was removed on 21 August 2026 and is kept under `_archive/linear_compute_se_removed_2026-08-21/`, since the repository is not under version control. The conditional-Hessian computation above is the behaviour of the model, and now the only standard-error code in the module. The other families each keep their own live `_compute_se` with a different signature; those are unaffected.

### 8.2 Test Statistics and P-values

Under the usual regularity conditions, the MLE satisfies:

```
(beta_j - beta_j^0) / se(beta_j) ->_d N(0, 1)
```

Two-sided z-statistics and p-values are computed as:

```
z_j  = beta_j / se(beta_j)
p_j  = 2 * (1 - Phi(|z_j|))
```

where Phi is the standard normal CDF. This is the same procedure used by statsmodels, R's glm, and sklearn's LogisticRegressionCV, making results directly comparable.

### 8.3 Confidence Intervals

Asymptotic confidence intervals at level (1 - alpha) are:

```
CI_j = [beta_j - z_{alpha/2} * se_j,  beta_j + z_{alpha/2} * se_j]
```

Available via `conf_int(alpha=0.05)`. The returned array has shape (p+1, 2), with row 0 for the intercept and rows 1..p for the feature coefficients.

### 8.4 Standardized Coefficients and Feature Importance

Raw coefficients are not directly comparable when predictors are on different scales. Standardized coefficients rescale each coefficient by the predictor's marginal standard deviation (as estimated from the fitted FIML covariance), making effect sizes comparable across features.

**MissLinear standardized coefficient:**

```
beta_std_j = beta_j * sigma_Xj / sigma_Y
```

**MissLogistic standardized coefficient:**

```
beta_std_j = beta_j * sigma_Xj
```

Feature importances are the normalized absolute standardized coefficients:

```
importance_j = |beta_std_j| / sum_k |beta_std_k|
```

They sum to 1.0 and represent the relative contribution of each predictor to the model's fitted values, accounting for scale differences.

### 8.5 Information Criteria

AIC and BIC are computed on the full parameter count n_params (the length of the optimization parameter vector theta):

```
AIC = 2 * n_params - 2 * L(theta_hat)
BIC = n_params * log(n) - 2 * L(theta_hat)
```

---

## 9. Prediction with Missing Features at Inference Time

A distinctive capability of both models is the ability to make predictions on new observations that themselves have missing features. This is handled consistently with the training-time philosophy: no value is imputed; the conditional distribution of the outcome given only the observed features is used directly.

### 9.1 MissLinear Prediction

For a new observation x_new with observed feature subset x_obs:

```
E[Y | X_obs = x_obs] = mu_Y + Sigma_{Y, X_obs} Sigma_{X_obs, X_obs}^{-1} (x_obs - mu_{X_obs})
```

This is the conditional expectation of Y under the fitted joint normal model. For complete observations, it reduces exactly to the standard linear prediction intercept + beta' x_new. For observations with all X missing, it returns the marginal mean mu_Y.

Prediction is pattern-grouped: all rows sharing the same missingness pattern share a single Cholesky solve, so cost scales with the number of distinct patterns rather than n.

### 9.2 MissLinear Prediction Intervals

The conditional variance of Y given X_obs determines the width of the prediction interval:

```
Var[Y | X_obs] = Sigma_YY - Sigma_{Y, X_obs} Sigma_{X_obs, X_obs}^{-1} Sigma_{X_obs, Y}
```

Observations with more missing features have larger conditional variance and therefore wider prediction intervals. This correctly reflects that predictions made from incomplete information are less certain, and the width varies continuously with the number and identity of observed features.

### 9.3 MissLogistic Prediction

Probability estimation for observations with missing X uses the vectorised Gauss-Hermite procedure: observations are grouped by pattern, Sigma_c and v are computed once per pattern, and the logistic-normal integral is evaluated across all observations in the group in a single matrix operation. This produces calibrated probability estimates that correctly account for predictor uncertainty.

---

## 10. Extended Model Families

MissLearn provides FIML variants of the most common supervised learning algorithms. All share the same core philosophy: no imputation, native NaN support, and the same sklearn fit/predict API.

### 10.1 Penalized Linear Models

**MissRidgeRegressor / MissRidgeClassifier** add an L2 penalty (alpha parameter) on top of the FIML objective, shrinking coefficients toward zero. Useful when p is large relative to n or when features are correlated.

**MissLASSORegressor / MissLASSOClassifier** add an L1 penalty, producing sparse solutions. The L1 term is handled via variable splitting (beta = u - v, u,v >= 0) to maintain differentiability for L-BFGS-B.

Both families use the same missingness pattern grouping and Cholesky parameterization as the base FIML models. Neither exposes `warm_start`: that parameter is specific to `MissLinear` and `MissLogistic`.

### 10.2 Nearest Neighbours

**MissNeighborsRegressor / MissNeighborsClassifier** compute available-case distances scaled by sqrt(p / n_shared) under the MAR assumption; observations sharing more features are compared on a commensurate scale. The fitted correlation matrix uses adaptive jitter to guard against near-singular configurations.

### 10.3 Naive Bayes

**MissBayesRegressor / MissBayesClassifier** use closed-form Gaussian posteriors. For regression, the posterior Y | X_obs is a normal distribution; for classification, the class posterior is computed from the product of per-feature conditionals over observed features only. No numerical optimization is required.

### 10.4 Support Vector Machines

**MissSupportRegressor / MissSupportClassifier** use available-case kernel matrices scaled by p / n_shared. Uncertainty from missing features is propagated into prediction intervals (regression) and probability calibration (classification via Platt scaling).

### 10.5 Gaussian Processes

**MissGaussianRegressor / MissGaussianClassifier** use a marginalized kernel that integrates out missing feature values under the empirical marginal distribution. This produces a valid positive-semidefinite kernel by construction (see module docstring for mathematical details). Hyperparameters are optimized via log-marginal-likelihood. The Cholesky factorization of the kernel matrix uses eigenvalue clipping as the primary stability strategy (see Section 7.4). Practical for n ≤ ~400 due to O(n³) cost; for larger n consider MissRidge or MissNeighbors.

### 10.6 Random-Intercept Mixed Effects

**MissMixedRegressor / MissMixedClassifier** add a random intercept per group on top of FIML, making them full-information mixed-effects models. Appropriate for longitudinal data, repeated measures, or any data where observations are nested within groups. Requires a `groups` argument at fit time. The random intercept is integrated out via Gauss-Hermite quadrature (Q = 20 nodes, same default as MissLogistic).

### 10.7 Ensemble

**MissEnsemble** wraps any MissLearn model (or NaN-native tree) in a bootstrap-aggregated ensemble. Reduces variance at the cost of interpretability.

### 10.8 Multi-class

**MissMulticlass** wraps any MissLearn binary classifier in a one-vs-rest scheme for K > 2 classes.

### 10.9 Multiple Imputation

**MissImputer** fits a joint MVN via EM and draws m complete datasets from conditional distributions (proper multiple imputation). Transform uses pattern-grouped conditional covariance computation: Sigma_c is computed once per unique missingness pattern rather than once per row, giving a 20 to 40% speedup when n is large. Combines m downstream model estimates via Rubin's rules.

### 10.10 Sensitivity Analysis

**MissSensitivity** provides delta-adjustment sensitivity analysis for MNAR departures, sweeping a grid of plausible MNAR shifts and reporting tipping-point deltas.

### 10.11 Explainability

**MissExplainer** computes SHAP values using the FIML model as the exact coalition value function. Produces both value SHAP (feature effect) and missingness SHAP (contribution of the missing indicator) with viridis visualizations.

---

## 11. Software Architecture and Performance

### 11.1 Module Structure

```
MissLearn/
    __init__.py         -- Public exports: all model classes and utilities
    _base.py            -- MissBase: input validation, inference, display,
                           sklearn metadata routing (sklearn >= 1.3)
    _linear.py          -- MissLinear
    _logistic.py        -- MissLogistic
    _ridge.py           -- MissRidgeRegressor, MissRidgeClassifier, MissRidge
    _lasso.py           -- MissLASSORegressor, MissLASSOClassifier, MissLASSO
    _knn.py             -- MissNeighborsRegressor, MissNeighborsClassifier, MissNeighbors
    _bayes.py           -- MissBayesRegressor, MissBayesClassifier, MissBayes
    _svm.py             -- MissSupportRegressor, MissSupportClassifier, MissSupport
    _gp.py              -- MissGaussianRegressor, MissGaussianClassifier, MissGaussian
    _mixed.py           -- MissMixedRegressor, MissMixedClassifier, MissMixed
    _ensemble.py        -- MissEnsemble
    _multiclass.py      -- MissMulticlass
    _imputer.py         -- MissImputer
    _sensitivity.py     -- MissSensitivity
    _explainer.py       -- MissExplainer
    _diagnostic.py      -- MissDiagnostic
    _recommend.py       -- MissRecommender, recommend_model
    _crossval.py        -- MissKFold, MissStratifiedKFold, miss_cross_val_score,
                           miss_cross_validate  (strips warm-start state)
    _validate.py        -- MissPreprocessor, prefit_check
    _copula.py          -- RankNormalTransformer, needs_copula
    _utils.py           -- Numerical primitives: sigmoid, Cholesky packing,
                           vectorised GH quadrature, conditional normal, Hessian
    _pandas_compat.py   -- Transparent DataFrame/Series support
```

### 11.2 Class Hierarchy

```
sklearn.base.BaseEstimator
    MissBase  (+ metadata routing: sklearn >= 1.3)
        MissLinear          (RegressorMixin)
        MissLogistic        (ClassifierMixin)
        MissRidgeRegressor  (RegressorMixin)
        MissRidgeClassifier (ClassifierMixin)
        MissLASSORegressor  (RegressorMixin)
        MissLASSOClassifier (ClassifierMixin)
        MissNeighborsRegressor  (RegressorMixin)
        MissNeighborsClassifier (ClassifierMixin)
        MissBayesRegressor  (RegressorMixin)
        MissBayesClassifier (ClassifierMixin)
        MissSupportRegressor  (RegressorMixin)
        MissSupportClassifier (ClassifierMixin)
        MissGaussianRegressor  (RegressorMixin)
        MissGaussianClassifier (ClassifierMixin)
        MissMixedRegressor  (RegressorMixin)
        MissMixedClassifier (ClassifierMixin)
    MissEnsemble
    MissMulticlass
```

MissBase provides: `_validate_and_convert`, `_store_fit_metadata`, `get_feature_names_out`, `missingness_report`, `conf_int`, `feature_importances_`, `_pvalues_from_zstat`, `_coef_table_lines`, `_importance_lines`, `get_metadata_routing`.

`set_fit_request` and `set_predict_request` are deliberately absent. scikit-learn generates them from the `fit` and `predict` signatures, and from 1.7 it skips any estimator that already carries the attribute. MissBase used to define both as no-op stubs, which was harmless on 1.6, where the generated method still won, and on 1.7 and later shadowed it: `MissMixedRegressor().set_fit_request(groups=True)` recorded nothing and the groups were then refused or dropped, which turns a random-intercept model into an ordinary regression.

### 11.3 Performance Architecture

Several implementation choices give MissLearn practical performance on large datasets:

**Pattern grouping.** All FIML models group observations by their unique missingness pattern before optimization. The Cholesky decomposition of Sigma_obs and the conditional covariance Sigma_c are computed once per pattern, not once per observation. For typical datasets with a moderate number of distinct patterns, this reduces the dominant O(n * p³) cost to O(n_patterns * max_group * p³).

**Vectorised GH quadrature.** The logistic-normal integral `E[sigma(a + t)]` is evaluated across all observations in a pattern group simultaneously using a single (n_group × Q) sigmoid evaluation, replacing a Python loop over rows. This gives a 30 to 50% speedup on the NLL gradient and `predict_proba`.

**Warm starting.** Setting `warm_start=True` on MissLinear or MissLogistic reuses the previous fit's parameter vector when the same estimator is refitted, which saves optimizer iterations on a sequence of related fits. Cross-validation is the one place it does not apply: `miss_cross_val_score` and `miss_cross_validate` strip the stored vector from every per-fold copy so that no fold begins from parameters fitted on its own held-out rows.

**Pattern-grouped imputation.** `MissImputer.transform()` pre-computes all pattern groups once outside the m-imputation loop. Sigma_c (shared across all rows in a pattern) is computed once per pattern per draw, not once per row. Effect: 20 to 40% speedup for large n with many rows per pattern.

**Adaptive jitter.** Covariance matrices are regularized with data-adaptive jitter (`max(1e-6, -λ_min + 1e-4)`) rather than a fixed small constant, preventing Cholesky failures on near-singular configurations without over-regularizing well-conditioned problems.

**Eigenvalue-based Cholesky.** MissGaussian replaces the previous six-step escalating-jitter loop with a two-step approach: fast-path direct Cholesky, then immediate eigenvalue-clip recovery. This is 3 to 5× faster on ill-conditioned kernel matrices and eliminates try/except branching overhead.

### 11.4 sklearn Compatibility

All models implement the sklearn estimator API (fit / predict / score / get_params / set_params / clone). For sklearn ≥ 1.3, `get_metadata_routing`, `set_fit_request`, and `set_predict_request` are implemented on `MissBase`, enabling correct metadata (e.g., `groups` for MissMixed) forwarding through `Pipeline` and `GridSearchCV`.

Pandas DataFrames and Series are accepted as inputs to all models. Column names are stored as `feature_names_in_` after fit and used in summary output and explainability plots.

### 11.5 Dependencies

| Package | Purpose | Minimum Version |
|---|---|---|
| numpy | Array operations, linear algebra | 1.22 |
| scipy | Optimization, quadrature, statistics | 1.8 |
| scikit-learn | Base classes, initialization, CV | 1.1 |
| pandas | Optional DataFrame support | 1.4 |
| matplotlib | Plotting (summary, importances, SHAP) | 3.5 |

These are the minimums declared in `pyproject.toml`; the first three are hard
requirements and the last two are optional extras.

---

## 12. API Reference

### MissLinear

```python
MissLinear(
    max_iter   = 2000,      # Maximum optimizer iterations
    tol        = 1e-7,      # Convergence tolerance (function value and gradient norm)
    method     = 'L-BFGS-B',
    compute_se = True,      # Standard errors from the conditional Hessian
    copula     = False,     # Gaussian copula transform ('auto', True, False)
    warm_start = False,     # Reuse previous theta as optimizer starting point
)
```

| Method / Attribute | Description |
|---|---|
| `fit(X, y)` | Fit the FIML model. X and y may contain NaN. Returns self. |
| `predict(X)` | Predict E[Y \| X_obs] via conditional normal mean. NaN allowed in X. |
| `predict_interval(X, alpha=0.05)` | Return (lower, upper) prediction interval arrays. |
| `score(X, y)` | R² on rows where y is not NaN. |
| `summary(alpha=0.05)` | Print full formatted model summary to stdout. |
| `conf_int(alpha=0.05)` | ndarray (p+1, 2) of confidence interval bounds. |
| `missingness_report()` | Print missingness statistics from training data. |
| `feature_importances_` | ndarray (p,) of normalized importance scores, sum to 1. |
| `coef_` | ndarray (p,) regression coefficients. |
| `intercept_` | float, regression intercept. |
| `se_` | ndarray (p+1,) standard errors [intercept, coef_0, ...]. |
| `pvalues_` | ndarray (p+1,) two-sided p-values. |
| `z_stats_` | ndarray (p+1,) z-statistics. |
| `coef_std_` | ndarray (p,) standardized coefficients. |
| `mu_joint_` | ndarray (p+1,) fitted joint mean vector [mu_Y, mu_X]. |
| `Sigma_joint_` | ndarray (p+1, p+1) fitted joint covariance matrix. |
| `sigma_sq_` | float, conditional variance of Y given X. |
| `loglik_`, `aic_`, `bic_` | float, model fit statistics. |
| `converged_` | bool, optimizer convergence status. |
| `_theta_opt_` | ndarray, fitted parameter vector (available when warm_start=True). |

### MissLogistic

```python
MissLogistic(
    max_iter     = 2000,
    tol          = 1e-7,
    method       = 'L-BFGS-B',
    n_quadrature = 20,       # GH nodes for the small-variance branch only
    compute_se   = True,
    l2_reg       = 0.01,     # L2 penalty on slopes; 0 for exact FIML
    copula       = False,
    warm_start   = False,    # Reuse previous theta as optimizer starting point
)
```

| Method / Attribute | Description |
|---|---|
| `fit(X, y)` | Fit the FIML model. X may contain NaN. y must be binary {0,1}; NaN allowed. |
| `predict(X)` | Predict class labels. NaN allowed in X. |
| `predict_proba(X)` | Return (n, 2) array of class probabilities. NaN allowed in X. |
| `decision_function(X)` | Return log-odds (linear predictor). |
| `score(X, y)` | Classification accuracy on rows where y is not NaN. |
| `summary(alpha=0.05)` | Print full formatted model summary. |
| `conf_int(alpha=0.05)` | ndarray (p+1, 2) confidence interval bounds. |
| `feature_importances_` | ndarray (p,) normalized importance scores. |
| `coef_` | ndarray (p,) log-odds coefficients. |
| `intercept_` | float, log-odds intercept. |
| `odds_ratios_` | ndarray (p+1,) of exp(coef) values. |
| `se_`, `pvalues_`, `z_stats_` | Inference arrays, shape (p+1,). |
| `coef_std_` | ndarray (p,) standardized coefficients. |
| `mu_X_`, `Sigma_X_` | Fitted predictor distribution parameters. |
| `loglik_`, `aic_`, `bic_`, `converged_` | Model fit statistics. |
| `_theta_opt_` | ndarray, fitted parameter vector (available when warm_start=True). |

### MissImputer

```python
MissImputer(
    m            = 20,      # Number of imputed datasets
    include_y    = False,
    max_iter     = 200,     # EM iterations for joint MVN fit
    tol          = 1e-6,
    reg          = 1e-6,    # Diagonal regularisation for PD guarantee
    posterior    = False,   # Propagate parameter uncertainty via bootstrap
    random_state = None,
)
```

`transform(X)` returns a list of m complete ndarrays (pattern-grouped for efficiency).  
`transform_mean(X)` returns a single deterministic conditional-mean imputation.  
`fit_transform_combine(X, y, estimator, param)` fits m models and pools via Rubin's rules.

### Cross-validation

```python
from MissLearn import miss_cross_val_score, miss_cross_validate, MissKFold, MissStratifiedKFold

# A warm-started estimator can be passed directly, but the warm start is
# not used across folds; each fold is fitted from its own initialisation.
model = MissLinear(warm_start=True)
scores = miss_cross_val_score(model, X, y, cv=5, scoring='r2')
```

`miss_cross_val_score` and `miss_cross_validate` delete `_theta_opt_` from every per-fold estimator copy, so setting `warm_start=True` neither speeds up nor changes a cross-validation score. Fold independence is the reason: a fold started from the previous fold's optimum would be initialised from parameters fitted partly on its own test rows.

---

## 13. Usage Examples

### Basic usage

```python
from MissLearn import MissLinear, MissLogistic
import numpy as np

# Data with missing values encoded as NaN
X = np.array([[1.0, 2.0, np.nan],
              [np.nan, 3.5, 1.2],
              [2.1, np.nan, 0.8],
              [1.5, 2.8, 1.1]])
y_cont   = np.array([3.2, np.nan, 2.1, 3.8])  # MissLinear; NaN in y is allowed
y_binary = np.array([1, 0, 1, 1])             # MissLogistic; must be {0, 1}

# Linear regression
lm = MissLinear()
lm.fit(X, y_cont)
lm.summary()
predictions = lm.predict(X)
lower, upper = lm.predict_interval(X, alpha=0.05)

# Logistic regression
lg = MissLogistic()
lg.fit(X, y_binary)
lg.summary()
labels = lg.predict(X)
probabilities = lg.predict_proba(X)
```

### Warm starting a sequence of fits

```python
from MissLearn import MissLinear

# The saving comes from refitting the same estimator on related data, for
# example along a regularisation path. The second fit starts from the first
# fit's parameter vector rather than from a fresh initialisation.
lm = MissLinear(compute_se=False, warm_start=True)
lm.fit(X, y)
lm.fit(X_extended, y_extended)   # starts from the previous optimum

# Cross-validation is the exception: miss_cross_val_score strips the stored
# vector from each fold's copy, so folds stay independent of one another.
```

### Extended model families

```python
from MissLearn import (
    MissRidgeRegressor, MissLASSOClassifier,
    MissNeighborsRegressor, MissGaussianClassifier,
    MissMixedRegressor, MissEnsemble,
)

# Ridge regression with missing data
ridge = MissRidgeRegressor(alpha=1.0)
ridge.fit(X, y)

# LASSO classification
lasso_clf = MissLASSOClassifier(alpha=0.1)
lasso_clf.fit(X, y_binary)

# KNN regression (available-case distances)
knn = MissNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)

# GP classification (small n only: O(n³))
gp_clf = MissGaussianClassifier()
gp_clf.fit(X_small, y_small)

# Mixed-effects regression (grouped data)
mixed = MissMixedRegressor()
mixed.fit(X, y, groups=group_ids)   # group_ids: array of group labels

# Ensemble (bootstrap-aggregated)
ens = MissEnsemble(MissRidgeRegressor(alpha=1.0), n_estimators=50)
ens.fit(X, y)
```

### Multiple imputation

```python
from MissLearn import MissImputer

imp = MissImputer(m=20, random_state=0)
imp.fit(X)

# Get m complete datasets
datasets = imp.transform(X)   # list of 20 ndarrays

# Deterministic mean imputation (for exploration only)
X_mean = imp.transform_mean(X)
```

### Working with pandas DataFrames

```python
import pandas as pd
from MissLearn import MissLinear

df = pd.DataFrame({
    'age':     [45, np.nan, 62, 38],
    'bmi':     [24.1, 29.3, np.nan, 22.8],
    'crp':     [1.2, 3.4, 2.1, np.nan],
    'outcome': [12.4, 15.1, 18.3, 10.2],
})

lm = MissLinear()
lm.fit(df[['age', 'bmi', 'crp']], df['outcome'])
lm.summary()   # column names from DataFrame appear in output
```

### sklearn pipeline integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from MissLearn import MissLinear

# MissLearn models are fully pipeline-compatible (sklearn >= 1.3 metadata routing)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  MissLinear(warm_start=True)),
])
pipe.fit(X, y)
```

---

## 14. Comparison with Alternative Approaches

### 14.1 Listwise Deletion (Complete Case Analysis)

Listwise deletion removes any observation with at least one missing value before fitting. Under MCAR it is unbiased but inefficient; under MAR it is biased. The bias arises because complete cases are a non-random subset of the sample, and their distribution may differ systematically from the full population.

The efficiency loss is substantial even at moderate missing rates. With 5 predictors and 20% missingness per predictor independently, only 0.8^5 = 33% of observations are complete. FIML retains information from the remaining 67%.

### 14.2 Mean Imputation

Mean imputation replaces each missing value with the column mean, then fits the model on the augmented complete dataset. This artificially reduces variance in the imputed columns, distorts the covariance structure between imputed and non-imputed variables, and underestimates standard errors by treating imputed values as if they were observed.

### 14.3 Multiple Imputation

Multiple imputation (Rubin, 1987) generates m complete datasets by sampling from the predictive distribution of the missing values, fits the analysis model to each, and pools the estimates. Under MAR and with a correctly specified imputation model, multiple imputation is approximately valid and is the closest conventional alternative to FIML.

FIML and multiple imputation are asymptotically equivalent under the joint normal model (Schafer and Graham, 2002). FIML is preferred when the joint model is correctly specified because it avoids the overhead of specifying and fitting a separate imputation model, avoids the pooling step, and produces a single coherent fitted model object. Multiple imputation may be preferred when the imputation model can incorporate auxiliary variables not in the analysis model. MissLearn's `MissImputer` class provides proper multiple imputation when it is preferred.

### 14.4 Summary Comparison

| Method | Bias under MAR | Efficiency | Missing y | Inference validity |
|---|---|---|---|---|
| Listwise deletion | Biased | Low | Discards rows | Requires MCAR |
| Mean imputation | Biased | Moderate | Distorts y | Invalid |
| Multiple imputation | Unbiased | High | Supported | Valid (m → ∞) |
| FIML (MissLearn) | Unbiased | Optimal | Supported | Valid (asymptotic) |

---

## 15. Limitations and Assumptions

**Multivariate normality of predictors.** All FIML models assume X ~ N(mu_X, Sigma_X). For MissLinear this extends to the joint normality of (Y, X). Categorical or highly skewed predictors violate this assumption. The Gaussian copula option (`copula=True` or `copula='auto'`) relaxes it for the skewed continuous case by mapping each such marginal to a normal scale before fitting. It does not help with categorical predictors and does not try to: columns with fewer than three distinct values pass through untouched, because a copula is identified only for continuous marginals and a rank-normal map of two categories is just a relabelling.

**Missing at Random.** FIML is valid under MAR: missingness may depend on observed values but not on the missing values themselves. If data are MNAR, use `MissSensitivity` to assess the impact of plausible MNAR departures.

**Sample size requirements.** FIML estimates the full (p+1) x (p+1) joint covariance matrix, which has (p+1)(p+2)/2 free parameters. Initialization requires at least (p+1)+1 complete cases. As a practical guideline, at least 10*(p+1) total observations are recommended for stable optimization.

**Computational scaling.** The per-observation likelihood evaluation is O(p³) due to Cholesky decompositions and triangular solves on submatrices. For large p (above ~50) or large n (above ~10,000), computation time may become substantial. Pattern grouping reduces the effective per-iteration cost, but the fundamental O(p³) factor per pattern cannot be avoided. Use `compute_se=False` to reduce overall fitting time, and `warm_start=True` when refitting the same estimator repeatedly. Warm starting buys nothing inside cross-validation, where the stored parameter vector is stripped per fold.

**Separation in MissLogistic.** When classes are linearly separable, the MLE for beta diverges. The `l2_reg` parameter stabilizes estimation in such cases. For well-conditioned problems, `l2_reg=0` (exact FIML) is appropriate.

**Binary outcomes only for MissLogistic.** MissLogistic supports binary classification. For K > 2 classes, use `MissMulticlass` to wrap any MissLearn binary classifier in a one-vs-rest scheme.

**GP cost.** `MissGaussianRegressor` and `MissGaussianClassifier` scale as O(n³) in training due to the full kernel matrix Cholesky. They are practical for n ≤ ~400; for larger datasets consider MissRidge or MissNeighbors.

---

## 16. References

Arbuckle, J. L. (1996). Full information estimation in the presence of incomplete data. In G. A. Marcoulides and R. E. Schumacker (Eds.), *Advanced structural equation modeling: Issues and techniques* (pp. 243-277). Lawrence Erlbaum Associates.

Ibrahim, J. G. (1990). Incomplete data in generalized linear models. *Journal of the American Statistical Association*, 85(411), 765-769. https://doi.org/10.2307/2290013

Ibrahim, J. G., Chen, M. H., Lipsitz, S. R., and Herring, A. H. (2005). Missing-data methods for generalized linear models: A comparative review. *Journal of the American Statistical Association*, 100(469), 332-346. https://doi.org/10.1198/016214504000001844

Little, R. J. A., and Rubin, D. B. (2002). *Statistical analysis with missing data* (2nd ed.). Wiley. https://doi.org/10.1002/9781119013563

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592. https://doi.org/10.1093/biomet/63.3.581

Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*. Wiley. https://doi.org/10.1002/9780470316696

Schafer, J. L., and Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods*, 7(2), 147-177. https://doi.org/10.1037/1082-989X.7.2.147
