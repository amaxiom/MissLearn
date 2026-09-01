# MissLearn Computational Guide

**Statistical methods and their implementation**

This document is the definitive background reference for the statistical machinery inside MissLearn and how each piece is realized in code. It reflects the current implementation (July 2026) of the modules in `MissLearn/`: `_linear.py`, `_logistic.py`, `_ridge.py`, `_lasso.py`, `_knn.py`, `_bayes.py`, `_svm.py`, `_gp.py`, `_mixed.py`, `_ensemble.py`, `_imputer.py`, `_recommend.py`, `_utils.py`, and `_copula.py`. It is a companion to `METHODS_GUIDE.md`, which covers philosophy, API, and usage; this guide covers the mathematics and the numerical engineering.

---

## Table of Contents

1. [Missing-Data Theory](#1-missing-data-theory)
   - [1.1 Mechanisms: MCAR, MAR, MNAR](#11-mechanisms-mcar-mar-mnar)
   - [1.2 Likelihood factorization under ignorability](#12-likelihood-factorization-under-ignorability)
   - [1.3 Why FIML is consistent and efficient under MAR](#13-why-fiml-is-consistent-and-efficient-under-mar)
   - [1.4 Relation to EM](#14-relation-to-em)
   - [1.5 Relation to multiple imputation](#15-relation-to-multiple-imputation)
2. [The Joint-MVN Machinery](#2-the-joint-mvn-machinery)
   - [2.1 Marginal density of an observed subvector](#21-marginal-density-of-an-observed-subvector)
   - [2.2 Partitioned-normal conditioning](#22-partitioned-normal-conditioning)
   - [2.3 Pattern grouping; the central implementation trick](#23-pattern-grouping-the-central-implementation-trick)
3. [Two-Stage FIML for the Linear Family](#3-two-stage-fiml-for-the-linear-family)
   - [3.1 Stage 1: EM for the MVN nuisance moments](#31-stage-1-em-for-the-mvn-nuisance-moments)
   - [3.2 Stage 2: reduced conditional likelihood](#32-stage-2-reduced-conditional-likelihood)
   - [3.3 Vectorization of the Stage-2 objective](#33-vectorization-of-the-stage-2-objective)
   - [3.4 Exact analytic gradients](#34-exact-analytic-gradients)
   - [3.5 The MissLinear special case](#35-the-misslinear-special-case)
   - [3.6 Assembling the full-information log-likelihood](#36-assembling-the-full-information-log-likelihood)
4. [Penalized Models: Ridge and LASSO](#4-penalized-models-ridge-and-lasso)
   - [4.1 Variable splitting for the L1 penalty](#41-variable-splitting-for-the-l1-penalty)
   - [4.2 Internal standardization and why raw-scale penalties collapse](#42-internal-standardization-and-why-raw-scale-penalties-collapse)
   - [4.3 Affine-equivariant back-conversion](#43-affine-equivariant-back-conversion)
   - [4.4 Penalty exclusion from loglik_/AIC/BIC](#44-penalty-exclusion-from-loglik_aicbic)
   - [4.5 Standard errors under penalties](#45-standard-errors-under-penalties)
5. [Multiple Imputation: MissImputer](#5-multiple-imputation-missimputer)
6. [Expected-Distance Geometry: KNN and SVM](#6-expected-distance-geometry-knn-and-svm)
   - [6.1 Conditional moments (F, s)](#61-conditional-moments-f-s)
   - [6.2 Expected squared distances (Eirola et al. 2013)](#62-expected-squared-distances-eirola-et-al-2013)
   - [6.3 The augmented-space embedding and PSD kernels](#63-the-augmented-space-embedding-and-psd-kernels)
   - [6.4 Expected inner products and the Gram diagonal](#64-expected-inner-products-and-the-gram-diagonal)
   - [6.5 gamma='scale' on the embedding variance](#65-gammascale-on-the-embedding-variance)
   - [6.6 KNN specifics](#66-knn-specifics)
   - [6.7 SVM specifics](#67-svm-specifics)
7. [Full-Covariance Generative Bayes](#7-full-covariance-generative-bayes)
   - [7.1 Regressor: linear-Gaussian evidence model](#71-regressor-linear-gaussian-evidence-model)
   - [7.2 Estimating the residual covariance T](#72-estimating-the-residual-covariance-t)
   - [7.3 Classifier: per-class shrunk covariances](#73-classifier-per-class-shrunk-covariances)
   - [7.4 structure='naive' as the diagonal special case](#74-structurenaive-as-the-diagonal-special-case)
8. [Gaussian Processes with Marginalized Kernels](#8-gaussian-processes-with-marginalized-kernels)
   - [8.1 The marginalized product kernel](#81-the-marginalized-product-kernel)
   - [8.2 Log marginal likelihood and analytic gradients (regression)](#82-log-marginal-likelihood-and-analytic-gradients-regression)
   - [8.3 Laplace approximation for classification](#83-laplace-approximation-for-classification)
   - [8.4 y-scaled hyperparameter bounds and restarts](#84-y-scaled-hyperparameter-bounds-and-restarts)
9. [Mixed-Effects Models](#9-mixed-effects-models)
   - [9.1 Random-intercept LME via Woodbury](#91-random-intercept-lme-via-woodbury)
   - [9.2 GLMM via adaptive Gauss-Hermite quadrature](#92-glmm-via-adaptive-gauss-hermite-quadrature)
   - [9.3 BLUPs](#93-blups)
   - [9.4 FIML handling of missing X within groups](#94-fiml-handling-of-missing-x-within-groups)
10. [The Copula Transform](#10-the-copula-transform)
11. [Evidence-Based Model Triage: MissRecommender](#11-evidence-based-model-triage-missrecommender)
    - [11.1 What the evidence consists of](#111-what-the-evidence-consists-of)
    - [11.2 The linearity probe](#112-the-linearity-probe)
    - [11.3 The clustering signal](#113-the-clustering-signal)
    - [11.4 Vetoes, scores and preprocessing](#114-vetoes-scores-and-preprocessing)
12. [Numerical Reliability and Complexity](#12-numerical-reliability-and-complexity)
    - [12.1 Cholesky parameterization](#121-cholesky-parameterization)
    - [12.2 PSD repair and jitter policies](#122-psd-repair-and-jitter-policies)
    - [12.3 Determinism](#123-determinism)
    - [12.4 Computational complexity per model](#124-computational-complexity-per-model)
13. [References](#13-references)

---

## 1. Missing-Data Theory

### 1.1 Mechanisms: MCAR, MAR, MNAR

Let $Z = (Z_{\text{obs}}, Z_{\text{mis}})$ be the complete data and $R$ the missingness indicator matrix ($R_{ij}=1$ if $Z_{ij}$ is observed). Rubin (1976) classified missingness mechanisms by how the distribution of $R$ depends on the data:

- **MCAR (Missing Completely at Random):** $P(R \mid Z) = P(R)$. Missingness is independent of both observed and unobserved values. Complete cases are a simple random subsample, so listwise deletion is unbiased but inefficient. MCAR is the strongest and least realistic assumption.
- **MAR (Missing at Random):** $P(R \mid Z) = P(R \mid Z_{\text{obs}})$. Missingness may depend on observed values but not, conditionally, on the values that are missing. Example: blood pressure more often missing for older patients (age observed), but not because of the pressure value itself. MAR is the operating assumption of every likelihood-based model in MissLearn.
- **MNAR (Missing Not at Random):** $P(R \mid Z)$ depends on $Z_{\text{mis}}$ even after conditioning on $Z_{\text{obs}}$. Example: high incomes selectively unreported. MNAR is not identifiable from the observed data alone; MissLearn's `MissSensitivity` class provides delta-adjustment sensitivity analysis for exploring plausible MNAR departures.

### 1.2 Likelihood factorization under ignorability

The likelihood of the observed data $(Z_{\text{obs}}, R)$ under a data model with parameters $\theta$ and a missingness model with parameters $\phi$ is

$$
L(\theta, \phi \mid Z_{\text{obs}}, R) \;=\; \int p(Z_{\text{obs}}, z_{\text{mis}} \mid \theta)\; p(R \mid Z_{\text{obs}}, z_{\text{mis}}, \phi)\; dz_{\text{mis}} .
$$

Under MAR, $p(R \mid Z_{\text{obs}}, z_{\text{mis}}, \phi) = p(R \mid Z_{\text{obs}}, \phi)$ does not depend on $z_{\text{mis}}$ and factors out of the integral:

$$
L(\theta, \phi \mid Z_{\text{obs}}, R) \;=\; p(R \mid Z_{\text{obs}}, \phi) \int p(Z_{\text{obs}}, z_{\text{mis}} \mid \theta)\, dz_{\text{mis}}
\;=\; p(R \mid Z_{\text{obs}}, \phi)\; p(Z_{\text{obs}} \mid \theta).
$$

If additionally $\theta$ and $\phi$ are distinct (separable parameter spaces), the missingness mechanism is **ignorable** for likelihood inference: maximizing $p(Z_{\text{obs}} \mid \theta)$ alone yields valid inference about $\theta$ (Rubin 1976; Little & Rubin 2002, ch. 6). This is precisely what FIML does:

$$
\ell(\theta) \;=\; \sum_{i=1}^{n} \log p\!\left(z_{o(i)} ; \theta\right),
$$

where $o(i)$ indexes the observed entries of row $i$ and $p(z_{o(i)};\theta)$ is the marginal density of that subvector under the joint model; nothing is imputed, and every row contributes exactly the dimensions it possesses.

### 1.3 Why FIML is consistent and efficient under MAR

FIML (Arbuckle 1996) is *maximum likelihood on the observed-data likelihood*. Under MAR with an ignorable mechanism and a correctly specified joint model, the observed-data likelihood is a genuine likelihood for $\theta$, so the standard ML asymptotics apply:

- **Consistency:** $\hat\theta_{\text{FIML}} \xrightarrow{p} \theta_0$; the score of the observed-data log-likelihood has expectation zero at the truth, because marginalizing over $Z_{\text{mis}}$ is done under the true conditional law.
- **Asymptotic efficiency:** $\sqrt{n}(\hat\theta - \theta_0) \xrightarrow{d} N(0, I_{\text{obs}}(\theta_0)^{-1})$, where $I_{\text{obs}}$ is the observed-data Fisher information. No estimator based on the same observed data can have smaller asymptotic variance. Listwise deletion discards rows (strictly less information); single imputation understates variability; FIML attains the information bound.
- **Valid standard errors** come from the curvature (observed information / Hessian) of the observed-data log-likelihood at the optimum; which is how every MissLearn model computes them.

The contrast with deletion: under MAR-but-not-MCAR, complete cases are a *selected* subsample and complete-case estimates are generally biased; FIML remains unbiased because the selection operates only through variables retained in the conditioning set.

### 1.4 Relation to EM

The EM algorithm (Dempster, Laird & Rubin 1977) maximizes the same observed-data likelihood by iterating

- **E-step:** compute $Q(\theta \mid \theta^{(t)}) = E\!\left[\log p(Z_{\text{obs}}, Z_{\text{mis}} \mid \theta) \,\middle|\, Z_{\text{obs}}, \theta^{(t)}\right]$,
- **M-step:** $\theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)})$,

and each iteration increases the observed-data likelihood monotonically. For the multivariate normal, both steps are closed-form: the E-step fills sufficient statistics with conditional means and adds conditional covariances; the M-step is sample mean/covariance of the completed statistics. **EM and FIML therefore converge to the same MLE for the joint MVN**: EM is simply a different (and for this model much faster) route to the same stationary point, because each EM iteration is $O(\text{patterns} \times p^3)$ closed-form arithmetic, while a quasi-Newton iteration over a Cholesky parameterization needs many likelihood evaluations for finite-difference gradients. MissLearn exploits this equivalence heavily (§3).

### 1.5 Relation to multiple imputation

Multiple imputation (Rubin 1987) draws $m$ completions of $Z_{\text{mis}}$ from (an approximation to) the posterior predictive distribution $p(Z_{\text{mis}} \mid Z_{\text{obs}})$, fits the analysis model on each completed dataset, and pools with Rubin's rules (§5). Under a joint normal model, FIML and MI are asymptotically equivalent as $m \to \infty$ (Schafer & Graham 2002). FIML avoids the imputation-model/analysis-model compatibility issue and the pooling step, and yields a single coherent fitted object; MI can incorporate auxiliary variables outside the analysis model. MissLearn provides both: FIML natively in every model, and proper MI via `MissImputer`.

---

## 2. The Joint-MVN Machinery

Every likelihood-based model in MissLearn rests on two facts about the multivariate normal, plus one implementation trick.

### 2.1 Marginal density of an observed subvector

If $Z \sim N(\mu, \Sigma)$ in $\mathbb{R}^q$ and $o \subseteq \{1,\dots,q\}$ indexes the observed coordinates of a row, then the marginal law of the subvector is again normal with the corresponding sub-parameters:

$$
Z_o \sim N\!\left(\mu_o,\; \Sigma_{oo}\right).
$$

Consequently the FIML integral $\int p(z_o, z_m;\theta)\,dz_m$ has the **closed form** $N(z_o; \mu_o, \Sigma_{oo})$; no numerical integration, regardless of dimension or number of missing entries. The log-density is evaluated via Cholesky (`mvn_logpdf` / `mvn_logpdf_batch` in `_utils.py`):

$$
\log N(x;\mu,\Sigma) = -\tfrac{k}{2}\log 2\pi \;-\; \sum_i \log L_{ii} \;-\; \tfrac12 \left\lVert L^{-1}(x-\mu)\right\rVert^2, \qquad \Sigma = LL^\top,
$$

with one triangular solve per batch and $\log|\Sigma| = 2\sum_i \log L_{ii}$. A non-PD submatrix returns $-\infty$, which the optimizers translate into an objective value of $+\infty$ (rejected step).

### 2.2 Partitioned-normal conditioning

Partition $Z = (Z_o, Z_m)$ with

$$
\mu = \begin{pmatrix}\mu_o \\ \mu_m\end{pmatrix}, \qquad
\Sigma = \begin{pmatrix}\Sigma_{oo} & \Sigma_{om} \\ \Sigma_{mo} & \Sigma_{mm}\end{pmatrix}.
$$

Then the conditional distribution of the missing block given the observed values is normal:

$$
Z_m \mid Z_o = z_o \;\sim\; N\!\left(\mu_c,\; \Sigma_c\right),
$$

$$
\boxed{\;\mu_c \;=\; \mu_m + \Sigma_{mo}\,\Sigma_{oo}^{-1}\,(z_o - \mu_o), \qquad
\Sigma_c \;=\; \Sigma_{mm} - \Sigma_{mo}\,\Sigma_{oo}^{-1}\,\Sigma_{om}.\;}
$$

Implementation (`conditional_normal_params`, and inlined everywhere for batching): the regression matrix $K = \Sigma_{mo}\Sigma_{oo}^{-1}$ is obtained by solving the linear system $\Sigma_{oo} K^\top = \Sigma_{om}$ (`np.linalg.solve`), never by explicit inversion; $\Sigma_c$ is then symmetrized as $\tfrac12(\Sigma_c + \Sigma_c^\top)$ to cancel floating-point asymmetry. Where $\Sigma_{oo}$ is singular, the code falls back to the pseudoinverse.

Two properties drive the entire library:

1. $\mu_c$ is **affine in $z_o$**: $\mu_c = \mu_m + K(z_o - \mu_o)$; so it can be computed for a whole batch of rows with one matrix product.
2. $\Sigma_c$ (and hence any quadratic form $\beta_m^\top \Sigma_c \beta_m$) **does not depend on $z_o$ at all**: only on the missingness *pattern*.

### 2.3 Pattern grouping: the central implementation trick

Rows are grouped by their missingness pattern (the tuple of observed column indices). Because of property 2 above, for each of the $G$ distinct patterns:

- one Cholesky factorization of $\Sigma_{oo}$ serves **all** rows in the group (batched log-density via `mvn_logpdf_batch`);
- one solve produces $K$ and $\Sigma_c$ for the group; per-row work reduces to the affine map $\mu_c = \mu_m + (x_o - \mu_o)K^\top$, a single BLAS matrix multiply.

This turns the naive per-row cost $O(n\,p^3)$ into $O(G\,p^3 + n\,p^2)$ (often $O(G\,p^3 + n\,p)$ in the reduced Stage-2 objectives of §3). Pattern grouping appears in every module: the FIML NLLs (`_linear`, `_logistic`, `_ridge`, `_lasso`, `_mixed`), the EM fitter and imputer (`_imputer`), prediction and prediction intervals in every regression model, distance/kernel computation (`_knn`, `_svm`), and the generative-Bayes posteriors (`_bayes`). In real datasets $G \ll n$ (missingness patterns repeat), so the saving is typically one to two orders of magnitude.

---

## 3. Two-Stage FIML for the Linear Family

The "linear family"; `MissLinear`, `MissLogistic`, `MissRidgeRegressor/Classifier`, `MissLASSORegressor/Classifier`: shares a two-stage estimation architecture introduced to eliminate the dominant cost of earlier single-stage fits: quasi-Newton optimization over the $p + p(p+1)/2$ MVN nuisance parameters with finite-difference gradients.

**Stage 1** estimates the predictor-distribution nuisance moments $(\mu_X, \Sigma_X)$ once, by EM, at full information. **Stage 2** maximizes the *reduced conditional likelihood* over only the small regression block ($p+1$ or $p+2$ parameters), fully vectorized and with exact analytic gradients.

### 3.1 Stage 1: EM for the MVN nuisance moments

`_JointMVNFitter` (`_imputer.py`) computes the FIML MLE of a joint MVN on a NaN-bearing matrix:

**Initialization; pairwise available-case moments.** $\mu^{(0)} = $ column-wise `nanmean`; $\Sigma^{(0)}_{jk}$ is the sample covariance over rows where both columns $j$ and $k$ are observed (variance on the diagonal; pairs with fewer than 2 joint observations set to 0); a ridge `reg * I` is added for positive definiteness. Only the upper triangle is computed and mirrored.

**Pattern-grouped E-step with the conditional-covariance correction.** Rows are grouped by missingness pattern once, before iterating. For each pattern $(rows, o, m)$:

$$
\hat{X}_{rows,\,m} \;=\; \mu_m + (X_{rows,\,o} - \mu_o)\,K^\top, \qquad K = \Sigma_{mo}\Sigma_{oo}^{-1},
$$

and the **`S_extra` correction** accumulates the conditional covariance of the filled block,

$$
S_{\text{extra}}[m,m] \;\mathrel{+}=\; |rows| \cdot \Sigma_c, \qquad \Sigma_c = \Sigma_{mm} - K\Sigma_{om},
$$

(for fully-missing rows, $\hat{X} = \mu_m$ and $S_{\text{extra}} \mathrel{+}= |rows| \cdot \Sigma_{mm}$). This term is what distinguishes EM from naive conditional-mean imputation: the M-step covariance must account for $E[z_m z_m^\top \mid z_o] = \mu_c\mu_c^\top + \Sigma_c$, not just the mean outer products. Omitting `S_extra` systematically shrinks $\Sigma$.

**M-step.**

$$
\mu \leftarrow \frac1n \sum_i \hat z_i, \qquad
\Sigma \leftarrow \frac{1}{n}\left( \hat Z_c^\top \hat Z_c + S_{\text{extra}} \right) + \text{reg}\cdot I,
$$

with $\hat Z_c$ the demeaned filled matrix, followed by symmetrization.

**Relative log-likelihood convergence.** After each iteration the true observed-data FIML log-likelihood $\ell = $ `_mvn_loglik(X, mu, Sigma)` (itself pattern-grouped, one Cholesky per pattern) is evaluated, and EM stops when

$$
|\ell^{(t)} - \ell^{(t-1)}| \;<\; \text{tol} \cdot \left(1 + |\ell^{(t)}|\right).
$$

The *relative* criterion matters: $\ell$ has magnitude $\sim n\,p$, so an absolute tolerance of $10^{-6}$ would force every fit to run to `max_iter` for nothing.

Stage-1 configurations by caller: `MissLinear` uses `_JointMVNFitter(max_iter=500, tol=1e-10, reg=1e-8)` on the full joint $Z=[y \mid X]$ (EM does essentially all the work; see §3.5). The conditional models use `(max_iter=100, tol=1e-5, reg=1e-6)`: `MissLogistic`/`MissLASSOClassifier` fit the MVN on $X$ alone; `MissRidgeRegressor`/`MissLASSORegressor` fit it on $[X \mid y]$ jointly; so rows with observed $y$ but missing $X$ still sharpen the $X$ moments; and then take the $X$ block $(\hat\mu_X, \hat\Sigma_X) = (\hat\mu_{[:p]}, \hat\Sigma_{[:p,:p]})$.

### 3.2 Stage 2: reduced conditional likelihood

With $(\hat\mu_X, \hat\Sigma_X)$ fixed, the remaining likelihood over rows with observed $y$ is the conditional part $\sum_i \log p(y_i \mid x_{o(i)})$, which depends only on the regression block:

- regression: $\theta_r = (\beta_0, \beta, \log\sigma)$, dimension $p+2$;
- logistic: $\theta_r = (\beta_0, \beta)$, dimension $p+1$ (plus split copies $u,v$ for LASSO, §4.1).

`prep_conditional_terms(X_rows, mu_X, Sigma_X)` (`_utils.py`) precomputes, once per fit, the *pattern-constant* pieces:

- $F$; the design matrix with each missing entry replaced by its conditional mean $E[X_m \mid X_o]$ (all-missing rows receive $\mu_X$); and
- `groups`: a list of `(row_indices, mis_idx, Sigma_c)` triples, one per pattern.

The reduced conditional model for a row in pattern $g$ is then exactly

**Regression** (marginalizing $x_m \sim N(\mu_c, \Sigma_c^{(g)})$ out of $y\mid x \sim N(\beta_0 + \beta^\top x, \sigma^2)$):

$$
y_i \mid x_{o(i)} \;\sim\; N\!\left(\beta_0 + F_i^\top\beta,\;\; v_g\right), \qquad
v_g = \sigma^2 + \beta_m^\top \Sigma_c^{(g)} \beta_m .
$$

**Logistic** (the multi-dimensional marginalization collapses to 1-D because $\sigma(\beta^\top x)$ depends on $x_m$ only through the scalar $s = \beta_m^\top x_m$, which is Gaussian):

$$
P(y_i = 1 \mid x_{o(i)}) \;=\; E_{t\sim N(0, v_g)}\!\left[\sigma(a_i + t)\right], \qquad
a_i = \beta_0 + F_i^\top\beta, \quad v_g = \beta_m^\top \Sigma_c^{(g)} \beta_m,
$$

evaluated by Gauss-Hermite quadrature (physicists' convention, $t = \sqrt{2v}\,\tau$):

$$
p_{1,i} \;=\; \frac{1}{\sqrt{\pi}} \sum_{k=1}^{Q} w_k\, \sigma\!\left(a_i + \sqrt{2 v_g}\, t_k\right), \qquad Q = 20 \text{ by default}.
$$

**This rule is accurate only while $v_g$ is small.** In the quadrature variable
the integrand is $\sigma(a + \sqrt{2v}\,\tau)$, a step of width about
$1/\sqrt{2v}$, so the node count needed grows with $v$ faster than it is
practical to supply: at $Q = 20$ the worst error over $a \in [-6, 6]$ is
$5 \times 10^{-6}$ at $v = 4$ but $4 \times 10^{-3}$ at $v = 25$ and
$3 \times 10^{-2}$ at $v = 100$, and raising $Q$ to 640 still leaves more than
$10^{-4}$ at $v = 100$. Since $v_g = \beta_m^\top \Sigma_c^{(g)} \beta_m$, it
grows with both the number of missing features and the size of the
coefficients; a fitted model with $|\beta|$ up to 3.6 had a median $v_g$ of 23
across its patterns.

`integrate_logistic_normal` therefore switches rules above $v = 1$. Writing
$\sigma(s) = H(s) + r(s)$ for the unit step $H$ at the origin and substituting
$s = a + t$,

$$
E\left[\sigma(a+t)\right] \;=\; \Phi\!\left(\frac{a}{\sqrt{v}}\right)
\;+\; \int r(s)\, N(s; a, v)\, ds .
$$

The first term is exact and carries the whole $v$ dependence. The second has a
support that does not widen with $v$, because $|r(s)| < e^{-|s|}$, so
Gauss-Legendre on $[-40, 40]$ resolves it; $r$ jumps by $-1$ at the origin, so
it is integrated as two panels meeting there, which restores spectral
convergence. The result is accurate to about $10^{-16}$ for every $v$ from 0.1
to 2000. Gauss-Hermite is kept below $v = 1$, where it is exact to $10^{-11}$
and the split rule is the weaker of the two.

The Stage-2 likelihood optimised during `fit` uses the same rule, through
`logistic_normal_with_grads`, which returns the value together with
$\partial p / \partial a$ and $\partial p / \partial v$. Differentiating the
split form under the integral sign gives both in closed form,

$$
\frac{\partial p}{\partial a} = \frac{\phi(a/\sqrt{v})}{\sqrt{v}}
  + \int r(s)\, N(s; a, v)\, \frac{s-a}{v}\, ds ,
$$

$$
\frac{\partial p}{\partial v} = -\frac{a\,\phi(a/\sqrt{v})}{2 v^{3/2}}
  + \int r(s)\, N(s; a, v)
    \left[\frac{(s-a)^2}{2v^2} - \frac{1}{2v}\right] ds ,
$$

both of which reuse the density already formed for the value. The chain rule
from $(a, v_g)$ out to $(\beta_0, \beta)$ is unchanged and stays in each
estimator, since it differs between them; the integral and its two derivatives
do not, so they live in `_utils`.

Before this, fitting and prediction disagreed, and the fit was biased where the
signal was strongest. On a model with $|\beta|$ up to 5.3 the coefficients
moved by 0.16 between $Q = 20$ and $Q = 320$; they now move by exactly zero,
and the $Q = 20$ fit lands on what $Q = 320$ was previously needed to reach.
`n_quadrature` still selects the node count for the small-variance branch,
where 20 is accurate to $10^{-11}$, but it no longer changes the answer.

**`MissMixedClassifier` needed a different remedy.** Its quadrature is over the
random effect rather than over the missing features, and its integrand is a
*product* over a subject's observations rather than a single logistic, so there
is no step to peel off and the split above does not apply. It is adapted
instead; see [Section 9.2](#92-glmm-via-adaptive-gauss-hermite-quadrature).

Rows with observed $y$ but *no* observed $X$ are handled through the all-missing pattern ($F_i = \mu_X$, $\Sigma_c = \Sigma_X$), so they still inform $\beta$; notably the intercept; via the marginal $P(y)$.

With $F$ and the per-pattern $\Sigma_c$ fixed, **each Stage-2 NLL evaluation is $O(np)$** (plus $O(Gp^2)$ for the pattern-level quadratic forms): the optimizer never touches the MVN nuisance parameters at all.

### 3.3 Vectorization of the Stage-2 objective

The Stage-2 objective and gradient are pure array expressions; no Python loop over patterns or rows. The group structures built once at fit time are:

- `group_id`: an `(n,)` integer array assigning each y-observed row to its pattern $g$ (used for gather/scatter indexing);
- `M`: a stacked **$(G, p, p)$ conditional-covariance tensor**: `M[g]` is $\Sigma_c^{(g)}$ scattered into full-$p$ coordinates (zeros outside the missing block), so $\beta_m^\top \Sigma_c^{(g)}\beta_m = \beta^\top M[g]\, \beta$ without index bookkeeping;
- `n_g`: pattern sizes (regression only).

A single NLL+gradient evaluation is then:

```
Mb   = M @ beta                          # (G, p)   one batched matmul
v_g  = sigma_sq + Mb @ beta              # (G,)     regression variance per pattern
                                         #  (logistic: v_g = max(Mb @ beta, 0))
a    = intercept + F @ beta              # (n,)     linear predictor
```

**Regression:** residuals are reduced per pattern with `np.bincount`:

```
resid = y - a
rss_g = np.bincount(group_id, weights=resid**2, minlength=G)     # (G,)
nll   = 0.5 * n_g @ (log 2π + log v_g)  +  0.5 * sum(rss_g / v_g)
```

**Logistic (batched Gauss-Hermite):** the GH sum is evaluated for *all* rows simultaneously as an `(n, Q)` sigmoid:

```
c_g   = sqrt(2 v_g);  c_row = c_g[group_id]                       # (n,)
Zmat  = a[:, None] + c_row[:, None] * t_k[None, :]                # (n, Q)
S     = sigmoid(Zmat)
p1    = clip(S @ (w_k / sqrt(pi)), 1e-12, 1 - 1e-12)              # (n,)
nll   = -( sum(log p1[y==1]) + sum(log1p(-p1[y==0])) )
```

Complete rows have $v_g = 0$, so their GH nodes collapse to $\sigma(a)$ *exactly*; no special-casing needed; the same array expression is correct for complete and partial rows.

### 3.4 Exact analytic gradients

Both objectives supply exact gradients to L-BFGS-B (`jac=True`), eliminating finite differences entirely.

**Regression** (`_ridge.py`, `_lasso.py`). Parameterize $\sigma^2 = e^{2\theta_\sigma}$. With per-pattern weight

$$
\boxed{\;w_v^{(g)} \;=\; \frac{\partial\,\text{NLL}}{\partial v_g} \;=\; \frac{n_g}{2 v_g} \;-\; \frac{\text{rss}_g}{2 v_g^2}\;}
$$

and per-row weight $r_i / v_{g(i)}$ (with $r_i = y_i - \beta_0 - F_i^\top\beta$), the gradient components are

$$
\frac{\partial\,\text{NLL}}{\partial \beta_0} = -\sum_i \frac{r_i}{v_{g(i)}}, \qquad
\frac{\partial\,\text{NLL}}{\partial \beta} = -F^\top\!\left(\frac{r}{v_{g(\cdot)}}\right) \;+\; 2\sum_{g} w_v^{(g)}\, M_g\,\beta, \qquad
\frac{\partial\,\text{NLL}}{\partial \theta_\sigma} = 2\sigma^2 \sum_g w_v^{(g)} .
$$

The first $\beta$ term is the usual weighted-least-squares score through the mean; the second flows through the variance, since $\partial v_g / \partial \beta = 2 M_g \beta$ (in code: `2.0 * (w_v[:, None] * Mb).sum(axis=0)`). The $\theta_\sigma$ term uses $\partial v_g/\partial\theta_\sigma = 2\sigma^2$ for every pattern.

**Logistic** (`_logistic.py`, `_lasso.py`). Write $c_g = \sqrt{2 v_g}$ and $p_{1,i} = \sum_k \tilde w_k\, \sigma(a_i + c_{g(i)} t_k)$ with $\tilde w_k = w_k/\sqrt\pi$. Chain rule through $p_1$:

$$
d_i = \frac{\partial\,\text{NLL}}{\partial p_{1,i}} =
\begin{cases} -1/p_{1,i} & y_i = 1 \\ \;\;\,1/(1 - p_{1,i}) & y_i = 0 \end{cases}
$$

With $S_{ik} = \sigma(a_i + c_{g(i)} t_k)$ and $\sigma' = S(1-S)$:

$$
g_{a,i} = d_i \sum_k \tilde w_k\, S_{ik}(1{-}S_{ik}), \qquad
g_{c,i} = d_i \sum_k \tilde w_k\, t_k\, S_{ik}(1{-}S_{ik}),
$$

then scatter to patterns and convert from $c$ to $v$ using $dc/dv = 1/c$:

$$
g_v^{(g)} = \frac{1}{c_g} \sum_{i \in g} g_{c,i} \quad (\text{via } \texttt{np.bincount};\; 0 \text{ when } c_g \le 10^{-12}),
$$

$$
\frac{\partial\,\text{NLL}}{\partial \beta_0} = \sum_i g_{a,i}, \qquad
\frac{\partial\,\text{NLL}}{\partial \beta} = F^\top g_a \;+\; 2\sum_g g_v^{(g)}\, M_g\, \beta .
$$

Penalty gradients are appended per §4 ($+\lambda\beta$ for L2; $\pm\alpha$ on the split variables for L1). Since gradients are exact, L-BFGS-B converges in far fewer function evaluations, and each evaluation is a handful of BLAS calls.

### 3.5 The MissLinear special case

`MissLinear` models the *fully joint* MVN $Z = (Y, X) \sim N(\mu, \Sigma)$, so the entire parameter is "nuisance moments" and Stage 1 alone reaches the FIML MLE. The fit is therefore:

1. **EM to the joint MLE:** `_JointMVNFitter(max_iter=500, tol=1e-10, reg=1e-8)` on $Z = [y \mid X]$, with tight tolerance; EM iterations are far cheaper than quasi-Newton steps, so EM does almost all the work.
2. **Short quasi-Newton polish:** the EM solution is packed into the Cholesky parameterization (§12.1) and L-BFGS-B runs on the exact pattern-grouped joint NLL for at most `min(max_iter, 50)` iterations (`ftol=tol`, `gtol=tol*1e-2`), refining the final digits. Because the polish starts at the (EM-converged) optimum it typically terminates after one or two iterations, so its inverse-Hessian approximation is essentially the identity and is *never* used for inference.
3. **Analytic coefficient recovery** from the fitted joint moments (population-OLS identities, solved not inverted):

$$
\beta = \Sigma_{XX}^{-1}\Sigma_{XY}, \qquad
\beta_0 = \mu_Y - \beta^\top\mu_X, \qquad
\sigma^2 = \Sigma_{YY} - \Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}.
$$

4. **Conditional-likelihood standard errors:** SEs for $(\beta_0, \beta)$ come from the curvature of the *conditional* likelihood of $y$ given $X_{\text{obs}}$ at the fitted moments; the same reduced, fully vectorized NLL used by the penalized models (conditionally completed design matrix $F$, per-pattern marginalization variances); via a small $(p{+}2)$-dimensional numerical Hessian (`numerical_hessian`, symmetric central differences with $\varepsilon = 10^{-4}$, regularized by $10^{-8}I$). This treats the X-moments as fixed, the same convention as the ridge, LASSO and mixed models, and costs milliseconds.

Prediction with missing features uses the same conditioning: $E[Y \mid X_o] = \mu_Y + \Sigma_{Y,o}\Sigma_{oo}^{-1}(x_o - \mu_o)$ and $\text{Var}[Y \mid X_o] = \Sigma_{YY} - \Sigma_{Y,o}\Sigma_{oo}^{-1}\Sigma_{o,Y}$, pattern-grouped so cost scales with the number of distinct patterns, and prediction intervals widen continuously with missingness.

### 3.6 Assembling the full-information log-likelihood

For the two-stage models, `loglik_` reported to users is the full-information quantity: the maximized Stage-2 conditional part **plus** the Stage-1 X-marginal part,

$$
\hat\ell \;=\; \underbrace{-\big(\text{result.fun} - \text{penalty}\big)}_{\text{conditional } \sum_i \log p(y_i\mid x_{o(i)})} \;+\; \underbrace{\texttt{\_mvn\_loglik}(X;\hat\mu_X,\hat\Sigma_X)}_{\sum_i \log p(x_{o(i)})} \;\; (+\;\text{Jacobian, §4.3}),
$$

and the parameter count for AIC/BIC includes the MVN moments: $n_{\text{params}} = \dim(\theta_r) + p + p(p+1)/2$. AIC $= 2 n_{\text{params}} - 2\hat\ell$; BIC $= n_{\text{params}}\log n - 2\hat\ell$.

---

## 4. Penalized Models: Ridge and LASSO

`MissRidgeRegressor` adds $\tfrac{\alpha}{2}\lVert\beta\rVert_2^2$; `MissLogistic` adds $\tfrac{\lambda}{2}\lVert\beta\rVert_2^2$ via `l2_reg` (default 0.01, guarding against separation; `MissRidgeClassifier` is a thin subclass aliasing `alpha` to `l2_reg`); `MissLASSORegressor/Classifier` add $\alpha\lVert\beta\rVert_1$. Intercepts are never penalized, and the MVN nuisance moments are never penalized.

### 4.1 Variable splitting for the L1 penalty

The L1 norm is non-differentiable at zero, which breaks quasi-Newton methods. MissLearn uses the classical **variable-splitting** reformulation: write

$$
\beta = u - v, \qquad u \ge 0,\; v \ge 0,
$$

so that on the feasible set $\lVert\beta\rVert_1 \le \mathbf 1^\top(u + v)$, with equality when the complementarity condition $u_j v_j = 0$ holds. The penalized objective becomes *smooth and linear in the penalty*,

$$
\text{NLL}(\beta_0, u - v, \dots) + \alpha\,\mathbf 1^\top(u + v),
$$

subject to simple bound constraints $u, v \ge 0$; which **L-BFGS-B handles natively** (no smoothing approximation, no subgradients). At any minimizer, complementarity holds automatically (if both $u_j, v_j > 0$, decreasing both by $\min(u_j,v_j)$ strictly reduces the penalty without changing $\beta$), so $u_j + v_j = |\beta_j|$ and the penalty equals the true L1 norm. Gradients map through $\partial\beta_j/\partial u_j = 1$, $\partial\beta_j/\partial v_j = -1$: the code emits `[g_int, g_beta + alpha, -g_beta + alpha, ...]`. Sparsity is reported as `n_nonzero_` with threshold $|\beta_j| > 10^{-4}$ *on the standardized scale* (unit-free).

### 4.2 Internal standardization and why raw-scale penalties collapse

Both penalized families standardize internally before fitting (the **glmnet convention**):

$$
\tilde X_{ij} = \frac{X_{ij} - m_{x_j}}{s_{x_j}}, \qquad \tilde y_i = \frac{y_i - m_y}{s_y},
$$

with `nanmean` / `nanstd(ddof=1)` statistics, guards $s \ge 10^{-8}$ (else 1), and NaN preserved.

**Why this is not optional.** The penalized FIML objective trades off the data NLL against $\text{pen}(\beta)$. On raw scales, $\lVert\beta\rVert$ is an artifact of units: measuring a predictor in grams instead of kilograms multiplies its coefficient by 1000 and its penalty contribution by $10^6$, while the likelihood is unchanged under the compensating reparameterization. Worse, in these models $\sigma$ is a *free* parameter: the profile likelihood lets the optimizer escape the penalty entirely by setting $\beta \approx 0$ and absorbing all response variance into $\sigma^2 \approx \text{Var}(y)$; the fit "collapses" to the null model whenever raw units make $\lVert\beta\rVert$ large relative to the log-likelihood scale. Standardizing puts every coefficient on the same unit-free footing, so a single $\alpha$ has a consistent meaning and the penalty-vs-likelihood trade-off is well posed.

### 4.3 Affine-equivariant back-conversion

The Gaussian FIML model is closed under affine maps, so the standardized-scale optimum converts exactly to raw-scale public parameters:

$$
\hat\beta_{\text{raw}} = \hat\beta \cdot \frac{s_y}{s_x}, \qquad
\hat\beta_{0,\text{raw}} = m_y + s_y\,\hat\beta_0 - \hat\beta_{\text{raw}}^\top m_x, \qquad
\hat\sigma^2_{\text{raw}} = s_y^2\, \hat\sigma^2,
$$

$$
\hat\mu_{X,\text{raw}} = m_x + s_x \odot \hat\mu_X, \qquad
\hat\Sigma_{X,\text{raw}} = \hat\Sigma_X \odot (s_x s_x^\top).
$$

`predict()` and `predict_interval()` then work unchanged on raw-scale parameters. The reported `loglik_` receives the **change-of-variables Jacobian** so it refers to the raw-scale data density:

$$
\hat\ell_{\text{raw}} = \hat\ell_{\text{std}} \;-\; \sum_j n^{\text{obs}}_j \log s_{x_j} \;-\; n_y \log s_y,
$$

where $n^{\text{obs}}_j$ counts observed entries in column $j$ and $n_y$ the observed responses. Slope SEs rescale exactly ($se_{\beta_j} \cdot s_y/s_{x_j}$); the intercept SE uses a diagonal delta approximation, $se_{\beta_0,\text{raw}} = s_y\sqrt{se_{\beta_0}^2 + \sum_j (m_{x_j}/s_{x_j})^2 se_{\beta_j}^2}$ (slope-intercept covariances ignored, consistently across the penalized models).

### 4.4 Penalty exclusion from loglik_/AIC/BIC

`scipy`'s `result.fun` is the *penalized* NLL. All penalized models add the penalty back before reporting:

$$
\texttt{loglik\_} = -\big(\text{result.fun} - \text{pen}(\hat\beta)\big) + \text{marginal terms},
$$

so `loglik_`, `aic_`, and `bic_` always describe the **data likelihood only**, comparable across `alpha` values and across models. (The penalty is a device for estimation, not part of the probability model.)

### 4.5 Standard errors under penalties

Ridge models report Hessian-based SEs directly (the regression block occupies the leading coordinates of $\theta_r$, no delta method needed). LASSO models default to `compute_se=False`: the L1 objective is non-differentiable at $\beta_j = 0$, so asymptotic SEs are theoretically invalid at zeroed coefficients. With `compute_se=True`, SEs are mapped from the $(u,v)$ Hessian through the splitting Jacobian and should be interpreted only for clearly non-zero coefficients.

---

## 5. Multiple Imputation: MissImputer

`MissImputer` (`_imputer.py`) is proper multiple imputation built directly on §2-§3.1:

1. **Fit:** `_JointMVNFitter` estimates $(\hat\mu, \hat\Sigma)$ of $X$ (optionally augmented with $y$ when `include_y=True`) by full-information EM.
2. **Transform (draws):** rows are grouped by pattern once; for each pattern, $K$ and $\Sigma_c$ are computed a single time, $\mu_c$ per row by the affine map, and missing blocks are drawn $X_{i,m} \sim N(\mu_{c,i}, \Sigma_c + \text{reg}\,I)$. Repeating $m$ times (default 20) yields $m$ completed datasets. With `posterior=True`, each draw first refits $(\mu,\Sigma)$ on a nonparametric bootstrap resample; propagating parameter uncertainty, i.e., "proper" MI in Rubin's sense.
3. **`transform_mean`:** deterministic conditional-mean imputation (baseline/EDA only; understates variance by construction).
4. **Rubin's rules (`combine`):** given estimates $\hat\theta_1,\dots,\hat\theta_m$ with within-imputation variances $V_k$,

$$
\bar\theta = \frac1m\sum_k \hat\theta_k, \quad
W = \frac1m\sum_k V_k, \quad
B = \frac{1}{m-1}\sum_k (\hat\theta_k - \bar\theta)^2, \quad
T = W + \left(1 + \tfrac1m\right)B,
$$

with Barnard-Rubin degrees of freedom $\;\nu = (m-1)\left(1 + \frac{W}{(1+1/m)B}\right)^2$ and $t$-based p-values. `fit_transform_combine` automates fit-per-imputation pooling for any sklearn-compatible estimator, dropping (with a warning) any imputation whose variance attribute cannot be extracted so that estimates and variances always come from the same imputations.

---

## 6. Expected-Distance Geometry: KNN and SVM

`MissNeighbors*` and `MissSupport*` avoid imputation by replacing distances and kernels with their **exact expectations under the fitted joint Gaussian**.

### 6.1 Conditional moments (F, s)

`_ConditionalMoments` (`_imputer.py`) fits $N(\mu,\Sigma)$ to the standardized feature matrix by the Stage-1 EM (§3.1), then `transform(X)` returns, pattern-grouped (one solve per pattern):

- $F$; $X$ with each missing entry replaced by its conditional mean $E[X_m \mid X_o]$ (all-missing rows get $\mu$);
- $s$; the per-row **summed conditional variance** of the missing entries, $s_i = \operatorname{tr}\!\big(\operatorname{diag}(\Sigma_c^{(g_i)})\big) = \sum_{j \in m(i)} \text{Var}(X_{ij} \mid X_{i,o})$ (clipped at 0; all-missing rows get $\operatorname{tr}\Sigma$).

### 6.2 Expected squared distances (Eirola et al. 2013)

For two independent rows $q \ne t$ with $x_q \mid \text{obs}_q \sim N(F_q, C_q)$ and $x_t \mid \text{obs}_t \sim N(F_t, C_t)$:

$$
\boxed{\;E\big\lVert x_q - x_t\big\rVert^2 \;=\; \lVert F_q - F_t\rVert^2 \;+\; s_q \;+\; s_t\;}
$$

since $E\lVert x_q - x_t\rVert^2 = \lVert E x_q - E x_t\rVert^2 + \operatorname{tr}(C_q) + \operatorname{tr}(C_t)$ and only the diagonal traces $s = \operatorname{tr}(C)$ enter. This is the estimator of Eirola et al. (2013). It is computed for all pairs at once via the BLAS identity $\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b$ plus the rank-one additions `s_q[:, None] + s_t[None, :]`. Every pair is comparable (no "unreachable" rows, unlike available-case intersections), and rows with more missingness are correctly farther/more uncertain.

### 6.3 The augmented-space embedding and PSD kernels

Why do kernels built on these expected distances stay positive semidefinite? Embed row $i$ as the augmented vector

$$
\phi(i) = \big(F_i,\; \sqrt{s_i}\, e_i\big) \in \mathbb{R}^{p + n},
$$

where $e_i$ is the $i$-th standard basis vector; each row gets its **own orthogonal uncertainty axis**. Then for $i \ne j$,

$$
\lVert\phi(i) - \phi(j)\rVert^2 = \lVert F_i - F_j\rVert^2 + s_i + s_j = E\lVert x_i - x_j\rVert^2,
$$

i.e., the expected distances *are exact Euclidean distances in the augmented space*. Any kernel that is PSD as a function of Euclidean geometry (RBF, linear, polynomial with the matching Gram diagonal) therefore remains PSD when evaluated on the expected quantities; unlike rescaled available-case kernels, which have no such embedding and can produce indefinite Gram matrices. No imputed dataset is ever created: the distance itself is the model expectation.

### 6.4 Expected inner products and the Gram diagonal

For the linear/polynomial kernels the relevant expectations are

$$
E\langle x_a, x_b\rangle = \langle F_a, F_b\rangle \;\; (a \ne b), \qquad
E\lVert x_a\rVert^2 = \lVert F_a\rVert^2 + s_a \;\; (\text{diagonal}),
$$

matching $\langle\phi(a),\phi(b)\rangle$ in the augmented space. In `_svm.py::_compute_kernel`, when `Z_a is Z_b` (the training Gram matrix) the code treats identical objects as the *same realisation*: RBF self-distances are forced to exactly 0 (`np.fill_diagonal(sq_d, 0.0)`) and the linear/poly Gram diagonal is set to $\lVert F_a\rVert^2 + s_a$. Cross-kernels between different matrices never touch the diagonal special case.

### 6.5 gamma='scale' on the embedding variance

For RBF/poly kernels, `gamma='scale'` resolves to

$$
\gamma = \frac{1}{\text{total\_var}}, \qquad
\text{total\_var} = \sum_j \operatorname{Var}(F_{\cdot j}) + \overline{s},
$$

- the total variance of the **embedding** (variance of the conditionally-filled features plus the mean per-row conditional variance), i.e., the scale of the expected squared distances the kernel actually sees. This mirrors sklearn's `1/(p·Var)` convention but on the correct (uncertainty-inflated) geometry. `'auto'` gives $1/p$; a float is used directly.

### 6.6 KNN specifics

`MissNeighborsRegressor/Classifier` (metric `'euclidean'`, the default): standardize features (NaN-aware mean/std, constant-feature guard $s \ge 10^{-8}$), fit `_ConditionalMoments` on the standardized training matrix, cache $(F_{\text{train}}, s_{\text{train}})$, and compute the full expected-distance matrix at query time as $\sqrt{\lVert F_q - F_t\rVert^2 + s_q + s_t}$. Predictions are distance-weighted ($w = 1/d$; exact matches $d=0$ take over exclusively) or uniform means/votes over the $k$ nearest; the regressor's `predict_interval` uses the weighted std of neighbour targets.

The legacy `metric='mahalanobis'` path keeps the older **available-case** scheme: for each (query-pattern, train-pattern) pair, distances are computed on the shared observed features, optionally whitened by the Cholesky of the complete-case correlation submatrix, and scaled by $\sqrt{p/m}$ ($m$ = shared features); under MAR, $E[d^2_{\text{full}}] = (p/m)\,E[d^2_{\text{shared}}]$, so the rescaled partial distance is unbiased for the full one. Pairs sharing no features remain at $\infty$.

Determinism: ties at the $k$-th distance are resolved by gathering all candidates within the $k$-th smallest value and stable-sorting, so equal distances break by ascending training index regardless of BLAS floating-point variation.

### 6.7 SVM specifics

`MissSupportRegressor/Classifier` compute the expected kernel of §6.3 to 6.4 and hand it to LIBSVM via `kernel='precomputed'`: the solver is unchanged; only the kernel definition knows about missingness. The joint Gaussian is fitted on **all** rows (missing-$y$ rows still inform the feature distribution); rows with missing $y$ are then dropped from the SVM fit itself. The training Gram matrix is symmetrized ($K \leftarrow (K + K^\top)/2$) to cancel floating-point asymmetry. SVR standardizes $y$ internally (LIBSVM's $C$ and $\varepsilon$ are only meaningful on a unit response scale) and unscales at the API boundary. SVC uses `probability=True` (Platt scaling, internal 5-fold CV seeded `random_state=0`) for calibrated `predict_proba`; `predict` is the argmax of the calibrated probabilities for consistency with the other classifiers.

SVR prediction intervals are empirical: half-width $= z_{\alpha/2}\, \hat\sigma_{\text{train}} \sqrt{p / \max(m_i, 1)}$ where $\hat\sigma_{\text{train}}$ is the training-residual std and $m_i$ counts observed features in query row $i$ (missingness counted on the original, pre-transform matrix); widest for fully missing rows, $\pm z\hat\sigma$ for complete rows.

---

## 7. Full-Covariance Generative Bayes

`_bayes.py` provides closed-form Gaussian generative models; no optimizer anywhere; results are exactly reproducible across platforms.

### 7.1 Regressor: linear-Gaussian evidence model

Generative model:

$$
Y \sim N(\mu_Y, \sigma_Y^2), \qquad
X \mid Y \;\sim\; N\!\big(a + b\,Y,\; T\big),
$$

with slope vector $b$, intercept vector $a$, and **full** residual covariance $T$ of $X$ given $Y$ (equivalently, $(X,Y)$ jointly MVN). Because the observed subvector $X_o \mid Y$ is again Gaussian, missing features drop out of the evidence exactly, and the posterior for a row with observed set $o$ is Gaussian with closed-form precision and mean:

$$
\boxed{\;\frac{1}{\sigma^2_{\text{post}}} \;=\; \frac{1}{\sigma_Y^2} \;+\; b_o^\top T_{oo}^{-1} b_o, \qquad
\mu_{\text{post}} \;=\; \sigma^2_{\text{post}} \left( \frac{\mu_Y}{\sigma_Y^2} \;+\; b_o^\top T_{oo}^{-1}\,(x_o - a_o) \right).\;}
$$

Implementation: rows grouped by pattern; per pattern one solve $w = T_{oo}^{-1} b_o$ (with a diagonal fallback if $T_{oo}$ is singular), then $\sigma^2_{\text{post}}$ is pattern-constant and $\mu_{\text{post}}$ vectorizes over the group as $\sigma^2_{\text{post}}(\mu_Y/\sigma_Y^2 + (X_o - a_o) w)$. All-missing rows stay at the prior $(\mu_Y, \sigma_Y)$; the widest interval. `predict_interval` is the posterior credible interval $\mu_{\text{post}} \pm z\,\sigma_{\text{post}}$: precision (interval tightness) grows monotonically with the number of informative observed features.

Per-feature parameters $(b_j, a_j, \tau_j^2)$ come from available-case OLS of $X_j$ on $Y$ (pairs with $\ge 3$ joint observations), with sklearn-style variance smoothing `var_smoothing * max_j Var(X_j)` added to the diagonal.

### 7.2 Estimating the residual covariance T

With `structure='full'` (default), the off-diagonal of $T$ is estimated from **available-case residual pairs**: residuals $r_{ij} = x_{ij} - (a_j + b_j y_i)$ (NaN wherever $x_{ij}$ or $y_i$ is missing), pairwise covariances over rows where both residual columns are observed (`_pairwise_cov`, minimum 3 joint pairs, diagonal forced to $\tau_j^2$). Pairwise-complete covariance matrices need not be PSD, so two repairs follow (`_shrink_psd`):

1. **Shrinkage toward the diagonal:** $T \leftarrow (1-\lambda)T + \lambda\,\operatorname{diag}(T)$, with `shrinkage='auto'` setting $\lambda = \min\!\big(0.9, \max(0.05,\; p/(p + n_{\text{eff}}))\big)$ where $n_{\text{eff}}$ is the smallest jointly-observed pair count; more features / fewer joint observations ⇒ stronger shrinkage.
2. **PSD repair by eigenvalue clipping:** eigenvalues below a floor ($\max(10^{-12}, 10^{-8}\lambda_{\max})$) are clipped, the matrix is reassembled and symmetrized, and the (well-estimated) per-feature variances are restored by rescaling with $\sqrt{d_j / T_{jj}}$ so the repair perturbs only correlations, not variances.

The point of full $T$: with correlated features, the naive (diagonal) posterior **double-counts evidence**: ten highly correlated biomarkers are treated as ten independent measurements, and the posterior becomes badly overconfident. $T_{oo}^{-1}$ downweights redundant evidence correctly.

### 7.3 Classifier: per-class shrunk covariances

`MissBayesClassifier` is the QDA-style analogue:

$$
P(Y=k \mid X_o) \;\propto\; \pi_k \cdot N\!\big(x_o;\; \mu_k[o],\; \Sigma_k[o,o]\big),
$$

with per-class means/variances from available cases in that class (falling back to global feature statistics when a feature is nearly unobserved within a class), per-class full covariances $\Sigma_k$ from available-case pairs centred on the class means, and the same shrink-then-PSD-repair pipeline as §7.2. **Marginalization over missing dimensions is exact**: the observed subvector of an MVN is MVN, so each pattern costs one batched `mvn_logpdf_batch` per class; a non-PD submatrix (signalled by $-\infty$) triggers a per-pattern diagonal fallback. Posteriors are normalized with a log-sum-exp softmax. Feature importances are normalized Cohen's $d$ effect sizes $|\mu_{j1} - \mu_{j0}| / \sqrt{(\sigma^2_{j0} + \sigma^2_{j1})/2}$.

### 7.4 structure='naive' as the diagonal special case

Setting `structure='naive'` restricts $T$ (regressor) or $\Sigma_k$ (classifier) to its diagonal. The regressor posterior then reduces exactly to the classic naive-Bayes update

$$
\frac{1}{\sigma^2_{\text{post}}} = \frac{1}{\sigma_Y^2} + \sum_{j \in o} \frac{b_j^2}{\tau_j^2}, \qquad
\mu_{\text{post}} = \sigma^2_{\text{post}}\left(\frac{\mu_Y}{\sigma_Y^2} + \sum_{j\in o} \frac{b_j (x_j - a_j)}{\tau_j^2}\right),
$$

and the classifier to the per-feature product Gaussian NB. The naive variants are retained for reference and for very small samples where off-diagonal estimation is hopeless; `'full'` is the default because it weights correlated evidence correctly.

---

## 8. Gaussian Processes with Marginalized Kernels

### 8.1 The marginalized product kernel

Features are standardized (NaN-aware), so a missing value is modeled as a draw from the standard normal $N(0,1)$ in the standardized space. The kernel is a **product over features** of one-dimensional kernels, with missing coordinates *integrated out*:

$$
K_{\text{marg}}(x, x') \;=\; \sigma_f^2 \prod_{j=1}^{p} K_j(x_j, x'_j),
$$

$$
K_j =
\begin{cases}
k_{1d}\!\big(|z_j - z'_j| / \ell_j\big) & \text{both observed} \\[2pt]
E_{z \sim N(0,1)}\big[k_{1d}(|z_j - z| / \ell_j)\big] & \text{one missing} \\[2pt]
E_{z, z' \sim N(0,1)}\big[k_{1d}(|z - z'| / \ell_j)\big] & \text{both missing}
\end{cases}
$$

The 1-D expectations are evaluated by Gauss-Hermite quadrature with $Q=20$ nodes (module-level cache), uniformly for all supported kernels (`rbf`, `matern52`, `matern12`): one-missing uses nodes $\sqrt2\,t_m$ (since $z \sim N(0,1)$), both-missing uses $2t_m$ (since $z - z' \sim N(0,2)$). For the RBF kernel these expectations also have closed forms; $K_j^{\text{om}} = \frac{\ell}{\sqrt{\ell^2+1}}\exp\!\big(-\tfrac12 z_{\text{obs}}^2/(\ell^2+1)\big)$ and $K_j^{\text{mm}} = \frac{\ell}{\sqrt{\ell^2+2}}$; which the GH values reproduce to machine precision; the implementation uses GH so all kernels share one code path.

**PSD by construction:** each $K_j$ is an expectation of PSD kernels, expectations preserve PSD-ness, and products of PSD kernels are PSD (Schur). This solves the indefiniteness problem of raw available-case kernels. Predictive behaviour follows automatically: missing test features shrink $K(x_*, X)$ toward the small marginal values, so posterior variance rises smoothly toward the prior $\sigma_f^2$ for a fully missing test point; no heuristic uncertainty multipliers.

With `ard=True`, each feature has its own length scale $\ell_j$; normalized $1/\ell_j$ serves as a principled feature-importance ranking.

### 8.2 Log marginal likelihood and analytic gradients (regression)

`MissGaussianRegressor` maximizes the exact log marginal likelihood over log-hyperparameters $\theta = (\log\ell_1,\dots,\log\ell_p, \log\sigma_f, \log\sigma_n)$:

$$
\log p(y \mid X, \theta) = -\tfrac12 y^\top K_y^{-1} y - \tfrac12 \log|K_y| - \tfrac{n}{2}\log 2\pi, \qquad K_y = K_f + \sigma_n^2 I,
$$

via Cholesky ($\alpha = K_y^{-1}y$ by two triangular solves; $\log|K_y| = 2\sum\log L_{ii}$). The gradient uses the standard identity $\partial\,\text{LML}/\partial\theta_j = \tfrac12 \operatorname{tr}\!\big((\alpha\alpha^\top - K_y^{-1})\, \partial K_y/\partial\theta_j\big)$. With $W_{\text{eff}} = \alpha\alpha^\top - K_y^{-1}$ and the per-feature log-derivative fields $\texttt{logd}_j = \partial \log K_j / \partial \log\ell_j$ produced alongside the kernel (both for the observed and GH-marginalized branches, using $\partial k_{1d}/\partial\log\ell = -r\,k'(r)$):

$$
\frac{\partial\,\text{LML}}{\partial \log\ell_j} = \tfrac12 \sum_{ik} W_{\text{eff},ik}\, K_{f,ik}\, \texttt{logd}_{j,ik}, \qquad
\frac{\partial\,\text{LML}}{\partial \log\sigma_f} = \operatorname{tr}(W_{\text{eff}} K_f), \qquad
\frac{\partial\,\text{LML}}{\partial \log\sigma_n} = \sigma_n^2 \operatorname{tr}(W_{\text{eff}}),
$$

(the isotropic case sums the $\texttt{logd}_j$ before contracting).

The response is centred at fit time and the training mean is added back in `predict` and `predict_interval`. The reason is that the GP prior has mean zero, so an uncentred $y$ would make the posterior revert towards 0 rather than towards $\bar y$ away from the training data, and would force $\sigma_f$ to absorb the mean, which it cannot, because its bounds track the response *scale* and not its location (§8.4).

Predictions cache $L$ and $\alpha$: $\mu_* = K_{*X}\alpha + \bar y$, $\text{Var}_* = \sigma_f^2 - \lVert L^{-1}K_{*X}^\top\rVert^2_{\text{col}}$, predictive intervals add $\sigma_n^2$. The variance is unaffected by the centring.

### 8.3 Laplace approximation for classification

`MissGaussianClassifier` uses the Laplace approximation of Rasmussen & Williams (2006, ch. 3 & 5): the posterior over latent values is $p(f\mid y) \approx N(\hat f, (W + K^{-1})^{-1})$, with $\hat f$ the mode found by the numerically stable Newton iteration of **Algorithm 3.1** ($W = \operatorname{diag}(\pi(1-\pi))$, $B = I + W^{1/2} K W^{1/2}$, iterating $a = b - W^{1/2}B^{-1}W^{1/2}Kb$, $f = Ka$). The approximate LML is

$$
\log q(y\mid X,\theta) = y^\top\hat f - \sum_i \log\!\big(1 + e^{\hat f_i}\big) - \tfrac12 \hat f^\top K^{-1}\hat f - \tfrac12\log|B|.
$$

Its gradient has two parts (**Algorithm 5.1**):

- **Explicit part:** same trace formula as regression with $W_{\text{eff}} = a a^\top - R$, where $R = W^{1/2}B^{-1}W^{1/2} = (K + W^{-1})^{-1}$, and no noise term ($\sigma_n \equiv 0$; the classifier's parameter vector is $[\log\ell_\cdot, \log\sigma_f]$ and the code pads a dummy $\log\sigma_n = 0$ internally).
- **Implicit part**: the term most implementations omit: the mode $\hat f$ itself moves with $\theta$ through the $-\tfrac12\log|B|$ term. With third-derivative of the Bernoulli log-likelihood $\partial^3 \log p(y_i\mid f_i)/\partial f_i^3 = -\pi_i(1-\pi_i)(1-2\pi_i)$, the correction is

$$
s_{2,i} = -\tfrac12 \big[\operatorname{diag}(K - K R K)\big]_i \cdot \frac{\partial^3 \log p(y_i \mid f_i)}{\partial f_i^3}, \qquad
\text{corr}_j = s_2^\top (I - KR)\, \frac{\partial K}{\partial \theta_j}\,(y - \hat\pi),
$$

added per hyperparameter (with $\partial K/\partial\log\sigma_f = 2K_f$ and $\partial K/\partial\log\ell_j = K_f \odot \texttt{logd}_j$). Prediction uses the cached Laplace posterior: $\bar f_* = K_{*X} a$, $\text{Var}(f_*) = \sigma_f^2 - \lVert L_B^{-1} W^{1/2} K_{*X}^\top \rVert^2_{\text{col}}$, and the **probit approximation** for the class probability, $p(y_*{=}1) \approx \sigma\!\big(\bar f_* / \sqrt{1 + \pi \text{Var}(f_*)/8}\big)$.

### 8.4 y-scaled hyperparameter bounds and restarts

`_optimise` runs L-BFGS-B with `1 + n_restarts` starts from a fixed-seed generator (`np.random.default_rng(0)`: reproducible regardless of global NumPy state). Length-scale bounds are fixed at $\log\ell \in [-5, 3]$ (standardized inputs), but the amplitude and noise bounds are **shifted by $\log(\text{std}(y))$**:

$$
\log\sigma_f \in [-3, 3] + \log s_y, \qquad \log\sigma_n \in [-8, 1] + \log s_y,
$$

because $y$ is centred but *not* rescaled internally; the bounds must track the response scale or the optimizer would be boxed away from the optimum for large- or small-scale targets. The first start is the neutral point ($\log\ell = 0$, $\log\sigma_f = \log s_y$, $\log\sigma_n = \log s_y - 2$); failed restarts (LinAlgError etc.) are skipped; the best LML wins.

---

## 9. Mixed-Effects Models

### 9.1 Random-intercept LME via Woodbury

`MissMixedRegressor` fits the FIML random-intercept model

$$
y_{ij} = \beta_0 + x_{ij}^\top\beta + b_i + \varepsilon_{ij}, \qquad
b_i \sim N(0, \tau^2), \quad \varepsilon_{ij} \sim N(0, \sigma^2),
$$

jointly with $X \sim N(\mu_X, \Sigma_X)$. For subject $i$, marginalizing $b_i$ gives the observed outcome subvector

$$
y_i^{\text{obs}} \sim N\!\big(\mu_i,\; V_i\big), \qquad
V_i = \tau^2 J J^\top + D_i, \quad D_i = \operatorname{diag}\!\big(\sigma^2 + \delta_j\big),
$$

where $J = \mathbf 1$ over the subject's observed rows, $\mu_{ij} = \beta_0 + E[x_{ij}^\top\beta \mid x_{ij,\text{obs}}]$, and $\delta_j = \beta_m^\top \Sigma_{m|o}\beta_m$ absorbs missing-covariate uncertainty on the diagonal (§9.4). Because $V_i$ is diagonal-plus-rank-one, its inverse and determinant are $O(n_i)$ by the **Woodbury identity and matrix-determinant lemma**:

$$
V_i^{-1} r = D^{-1}r \;-\; D^{-1}\mathbf 1 \cdot \frac{\mathbf 1^\top D^{-1} r}{\,1/\tau^2 + \mathbf 1^\top D^{-1}\mathbf 1\,},
\qquad
\log|V_i| = \sum_j \log d_j + \log\!\big(1 + \tau^2\, \mathbf 1^\top D^{-1}\mathbf 1\big),
$$

implemented with `log1p` for the determinant correction. No $n_i \times n_i$ matrix is ever formed. Variance components are parameterized as $\log\tau$, $\log\sigma$ (unconstrained). This model uses finite-difference gradients, so `gtol` is floored at $\max(\text{tol}, 10^{-5})$; tighter values sit below the numerical-gradient noise floor and merely exhaust `maxiter`.

### 9.2 GLMM via adaptive Gauss-Hermite quadrature

`MissMixedClassifier` fits the random-intercept logistic GLMM

$$
y_{ij} \mid b_i \sim \text{Bernoulli}\!\big(\sigma(\beta_0 + x_{ij}^\top\beta + b_i)\big), \qquad b_i \sim N(0,\tau^2).
$$

The subject likelihood requires integrating over $b_i$. Writing the whole log-integrand as

$$
g_i(b) \;=\; \sum_j \log\sigma\!\big(s_{ij}\,\eta_{ij}(b)\big) \;+\; \log N(b; 0, \tau^2), \qquad s_{ij} = 2y_{ij} - 1,
$$

the integral is $\int e^{g_i(b)}\,db$, and it is taken by **adaptive** Gauss-Hermite (Liu and Pierce 1994) rather than the plain rule:

$$
\hat b_i = \arg\max_b g_i(b), \qquad \hat\sigma_i = \big(-g_i''(\hat b_i)\big)^{-1/2}, \qquad b_m = \hat b_i + \sqrt2\,\hat\sigma_i t_m ,
$$

$$
\log P(y_i^{\text{obs}} \mid X_i) \;=\; \operatorname{logsumexp}_m \left[ \log w_m + t_m^2 + \log\big(\sqrt2\,\hat\sigma_i\big) + g_i(b_m) \right] .
$$

The $e^{t_m^2}$ factor undoes the Gaussian weight that Gauss-Hermite carries, since the prior is now inside $g_i$ rather than being the quadrature weight. With $Q=1$ this is the Laplace approximation; the plain rule is the special case $\hat b_i = 0$, $\hat\sigma_i = \tau$.

**Why adapting is necessary.** The plain rule spreads its nodes over the *prior*, at width $\tau$ about zero. A subject carrying several observations has its own effect pinned into a much narrower interval, which may sit far from zero, and the nodes then sample where the integrand is negligible. The error grows with $\tau$ *and* with observations per subject, which is the opposite of the intuition that more data makes an integral easier: measured against a dense reference, the plain rule at $Q=20$ was out by $4\times10^{-10}$ at $\tau=1$ with one observation, but by $1.8$ at $\tau=5$ with twenty, and by $17$ at $\tau=12$. Raising $Q$ barely helps, since the problem is where the nodes are rather than how many: at $\tau=3$ with twenty observations even $Q=320$ left $6\times10^{-4}$. Adapted at $Q=20$ the same cases come out at $1.1\times10^{-14}$, $7.7\times10^{-6}$ and $3.3\times10^{-8}$, so $Q=20$ adapted beats $Q=80$ plain by several orders of magnitude everywhere.

$g_i$ is strictly concave, since

$$
g_i''(b) \;=\; -\sum_j \frac{1}{\text{scale}_j^2}\,\sigma(u_{ij})\,\sigma(-u_{ij}) \;-\; \frac{1}{\tau^2} ,
$$

every term of which is negative, so the mode is unique and Newton from zero reaches it in a handful of steps with no safeguarding. The same adapted nodes are reused for the BLUPs, $E[b_i \mid \text{data}_i]$, which is a ratio of two such integrals.

**Predicting for an unseen subject** carries no likelihood factor at all, so it reduces to $E[\sigma(a + t)]$ with $t \sim N(0, (\tau/\text{scale})^2)$: the same integral as [Section 3.2](#32-stage-2-reduced-conditional-likelihood), taken by `integrate_logistic_normal` and accurate for any variance.

evaluated with a stable log-sum-exp and vectorized over all nodes at once ($\eta$ built as a $(Q, n_i)$ array). Missing covariates enter through the **probit approximation**: for row $j$ with conditional linear-predictor variance $v_j = \delta_j$,

$$
P(y_{ij}=1 \mid x_{\text{obs}}, b_i) \;\approx\; \sigma\!\left( \frac{\eta^{\text{base}}_{ij} + b_i}{\sqrt{1 + \tfrac{\pi}{8}\, v_j}} \right),
$$

i.e., marginalizing a Gaussian through a logistic link is approximated by rescaling the argument by $\sqrt{1 + \pi v/8}$ (the classic logit-probit matching). This keeps the row-level integral one-dimensional (over $b_i$ only) instead of two-dimensional.

### 9.3 BLUPs

Fitted models expose per-subject random-effect predictions in `blup_`:

- **LME**: the Best Linear Unbiased Predictor, computed with the same Woodbury pieces:

$$
\hat b_i = \tau^2\, \mathbf 1^\top V_i^{-1}\big(y_i^{\text{obs}} - \mu_i\big).
$$

- **GLMM**: the exact posterior mean under the GH grid:

$$
\hat b_i = E[b_i \mid \text{data}_i] \;\approx\; \frac{\sum_m b_m\, \tilde w_m}{\sum_m \tilde w_m}, \qquad \tilde w_m = e^{\log p_m - \max_m \log p_m}.
$$

Prediction for known subjects adds $\hat b_i$ to the linear predictor (GLMM: a BLUP-shifted sigmoid); unknown subjects get the population-level prediction (GLMM: GH-integrated marginal probability). LME prediction intervals include $\tau^2$ for unknown subjects and drop it for known ones (the BLUP absorbs it): $\text{Var}_i = \sigma^2 + \delta_i\ (+\ \tau^2)$.

### 9.4 FIML handling of missing X within groups

Missing covariates are marginalized *analytically* using the shared conditional-normal machinery (`_row_contribs_batched`), which computes the contributions for every row of a missingness pattern at once:

$$
\mu^{\text{adj}}_{ij} = E[x_{ij}^\top\beta \mid x_{ij,\text{obs}}] = x_o^\top\beta_o + \mu_c^\top\beta_m, \qquad
\delta_{ij} = \operatorname{Var}[x_{ij,m}^\top\beta_m \mid x_{ij,\text{obs}}] = \beta_m^\top \Sigma_c\, \beta_m,
$$

with $\mu^{\text{adj}} = \mu_X^\top\beta$, $\delta = \beta^\top\Sigma_X\beta$ for all-missing rows. $\mu^{\text{adj}}$ shifts the mean; $\delta$ enters the LME's $V_i$ diagonal (extra outcome variance from covariate uncertainty) and the GLMM's probit scale. The predictor marginal $\sum \log p(x_{o(ij)})$ is **not** part of this objective. It is accounted for in stage one, where $\mu_X$ and $\Sigma_X$ are estimated by EM over the observed entries, so the objective above is conditional on those moments and carries only the outcome terms. That split removes $p + p(p+1)/2$ dimensions from the optimiser and from the standard-error Hessian.

The estimate is still full information, and it is worth being explicit about why, because an objective with no marginal term in it reads like a loss: missing $y$ rows inform the $X$ moments through stage one, missing $X$ rows inform $\beta$ through their conditional expectations in stage two, and the random-intercept correlation structure is preserved throughout. Earlier releases fitted this in one stage with $\mu_X$ and $\Sigma_X$ inside the parameter vector, and added the marginal through a helper called `_predictor_nll_batched`; that helper and the per-class `_unpack_params` routines that went with it were removed in August 2026, and are kept under `_archive/`.

---

## 10. The Copula Transform

`RankNormalTransformer` (`_copula.py`) relaxes the Gaussian marginal assumption while leaving all FIML/quadrature machinery untouched. By Sklar's theorem, a joint law decomposes into marginals plus a copula; mapping each marginal to $N(0,1)$ makes the Gaussian-copula model applicable to arbitrary continuous marginals (Liu et al. 2012).

**Forward transform.** Per column, the observed values are sorted and assigned **Blom (1958) plotting positions**

$$
u_{(r)} = \frac{r - 0.375}{n_{\text{obs}} + 0.25}, \qquad z_{(r)} = \Phi^{-1}(u_{(r)}),
$$

which keep the extreme ranks strictly inside $(0,1)$ (avoiding $\Phi^{-1}(0) = -\infty$). Repeated values are then collapsed: every member of a block of tied observations takes the **mean** of the normal scores the block spans, the mid-rank or van der Waerden convention, so the transform is a well-defined function of $x$ and the interpolation table is strictly increasing. Arbitrary values map through **linear interpolation** of the (distinct-value → mean-normal-score) table (`np.interp`); values outside the training range clip to the boundary normal score. **NaN maps to NaN** in both directions; missingness passes through the transform untouched, which is what allows every model to compose `copula` with FIML freely.

The collapse is required, not cosmetic. `np.interp` needs a strictly increasing first argument, and handing it the raw sorted values leaves duplicates whenever a column is discrete; each tied value then resolves to its block's *largest* score. On a binary column that mapped $0$ to $+1.18$ and $1$ to $+3.12$, a column of mean $1.41$ where the transform is defined to deliver $N(0,1)$, and the score a block landed on moved with the block boundaries, so fold-to-fold variance inflated too.

**Discrete columns pass through untouched.** A column with fewer than `MIN_DISTINCT_FOR_COPULA` (3) distinct observed values is returned on its own scale in both directions. Sklar's theorem identifies a copula uniquely only for continuous marginals; for a discrete marginal the empirical CDF is a step function and a rank-normal map is a relabelling of the categories rather than a route to normality. A column left with zero or one observed value takes the same route, so the copula is never what refuses data the rest of the library accepts (a wholly empty column is still rejected, by the shared `EmptyFeatureError` check).

**Inverse transform** interpolates the reverse table (normal score → sorted value), used to return predictions and interval endpoints to the original scale (`inverse_transform_1d` for single columns, e.g. $y$). Regression models transform both $X$ and $y$ (with separate transformers so $y$ predictions can be inverted); classifiers transform $X$ only ($y$ is discrete).

**`needs_copula` trigger rule** (used when `copula='auto'`): the transform is applied iff **any** eligible column with at least 8 observed values shows

$$
|\text{skewness}| > 1 \quad\text{or}\quad |\text{excess kurtosis}| > 2
$$

on its observed values; moderate asymmetry (an exponential has skewness ≈ 2) or substantial tail departure ($\chi^2_4$ has excess kurtosis 3). Columns with fewer than 8 observations are skipped, and so are discrete columns under the same rule the transform uses: an indicator with prevalence outside roughly $[0.25, 0.75]$ has skewness above 1 by construction, so without this one-hot dummies both invited the transform and were damaged by it.

Which columns count as eligible depends on what the estimator transforms. Regression models pass $y$ as well as $X$, because they transform $y$ and invert it on output. Classifiers pass $X$ only, since $y$ is a class label. `MissNeighborsRegressor` also passes $X$ only: it predicts by averaging neighbouring $y$ on the original scale and uses the copula to calibrate feature distances, so a skewed target is no reason to transform the features. The resolved decision is stored in `copula_used_` and reported in `summary()`.

---

## 11. Evidence-Based Model Triage: MissRecommender

`MissRecommender` (`_recommend.py`) returns no fitted predictive model. It gathers evidence about an incomplete data set, scores the model families against that evidence, and returns the ranking together with the reasoning that produced it. The reasoning is the deliverable: a bare recommendation cannot be checked against what you know about how the data were collected, and a ranked list with its evidence can.

### 11.1 What the evidence consists of

Four things are measured before any family is scored.

**Mechanism.** Little's MCAR test gives a global rejection or not. A rejection alone is weak guidance, so the recommender then asks the more useful question: is the missingness of the substantially-missing columns predictable from the *observed* values of the other columns? If it is, MAR is a defensible working assumption. If MCAR is rejected and no such association is found anywhere, MNAR cannot be excluded, and `MissSensitivity` is marked *required* rather than suggested.

**Shape.** The number of rows against the number of columns that survive the drop rule, which sets how much regularisation the problem needs.

**Tails.** Per-column excess kurtosis, which decides whether `copula=True` is set in the returned preprocessing.

**Clustering.** An intraclass correlation, when `groups` is supplied. See 11.3.

Complete cases are counted as rows with no missing feature *and* an observed y, since a row without a response contributes nothing to a complete-case fit.

### 11.2 The linearity probe

The question the probe answers is whether the conditional mean looks linear, because that is what separates the linear and penalized families from the kernel and neighbour families.

Two cheap cross-validated models are fitted on the same mean-imputed matrix, with identical preprocessing on both arms: a linear model (`Ridge` for regression, `LogisticRegression` for classification) and a nearest-neighbour model. Comparing raw scores would be misleading when both are small, so the comparison is made on skill *above the trivial baseline*:

```
lift_lin = max(1e-6, s_lin - floor)
lift_nn  = max(0,    s_nn  - floor)
ratio    = lift_nn / lift_lin
```

where `floor` is 0 for R² and 1/n_classes for accuracy. A ratio above 1.15 promotes the neighbour, kernel and Gaussian-process families and penalises the linear and penalized ones; below 0.9 it does the reverse. Between the two the probes are treated as tied, and the tie is broken towards the simpler model: `MissLinear`, `MissRidge` and `MissBayes` each gain a small amount, the neighbour and kernel families gain nothing. Indistinguishable evidence is a reason to prefer the model that is easier to interpret and cheaper to fit, not a reason to abstain.

The probe needs at least 40 observed rows and subsamples to `probe_max_n` (default 2000) above that, so its cost stays bounded. It is the one part of the recommender that fits models, and it is disabled by `probe_nonlinearity=False`.

### 11.3 The clustering signal

With `groups` supplied, a one-way random-effects ICC is computed on the observed response:

```
MS_b = SS_between / (G - 1)
MS_w = SS_within  / (n - G)
k    = n / G                          (balanced-design approximation)
ICC  = max(0, (MS_b - MS_w) / k) / (max(0, (MS_b - MS_w) / k) + MS_w)
```

It returns nothing when there are fewer than two groups or no residual degrees of freedom. An ICC of at least 0.05 promotes `MissMixed`; a negligible ICC penalises it, on the grounds that a random intercept costs parameters and should not be paid for when there is no between-group variance to absorb. `k` is the balanced-design approximation to the group size, so a badly unbalanced design makes the ICC approximate; it is evidence, not an estimate to report.

### 11.4 Vetoes, scores and preprocessing

Two conditions are vetoes rather than penalties, because they make a family inapplicable rather than merely unattractive: `MissMixed` without a `groups` argument, and `MissGaussian` when n exceeds `gp_max_n` (default 1000), since exact GP inference is O(n³). Vetoes are reported with their reason and kept separate from low scores, so a family that is merely unsuited is never confused with one that cannot run.

The returned `preprocessing_` also names columns to drop rather than model: any column whose missing rate exceeds `drop_threshold` (default 0.60). Those columns are excluded from the p used in the shape evidence, so the recommendation is made for the problem you would actually fit.

The output is a ranking, not a verdict. It is built from cheap probes and cross-validated only inside the linearity probe, so it should be read as a prior over families to try, and confirmed by the cross-validated comparison you were going to run anyway.

---

## 12. Numerical Reliability and Complexity

### 12.1 Cholesky parameterization

All directly-optimized covariance matrices are parameterized through their Cholesky factor: $\Sigma = LL^\top$ with off-diagonal entries of $L$ stored raw and diagonal entries stored as $\log L_{ii}$ (`pack_cholesky` / `unpack_cholesky`, row-major `tril_indices` layout; the diagonal of row $i$ sits at flat index $i(i+3)/2$). Exponentiating on unpack guarantees $L_{ii} > 0$, hence $\Sigma \succ 0$ **for every finite parameter vector**: unconstrained optimizers need no cone constraints, and every optimizer step is a valid covariance. Scalar variances use the same idea ($\sigma = e^\theta$, $\tau = e^\theta$).

### 12.2 PSD repair and jitter policies

Distinct policies by context, tuned to the failure mode:

- **Adaptive jitter for empirical covariances** (initialization, KNN correlation): check $\lambda_{\min}$ via `eigvalsh`; add $\max(10^{-6},\, -\lambda_{\min} + 10^{-4})$ when near-singular, else the base $10^{-6}$; enough to fix pathological collinearity without over-regularizing good data.
- **EM regularization:** `_JointMVNFitter` adds `reg * I` ($10^{-6}$-$10^{-8}$ by caller) at initialization and after every M-step.
- **Eigenvalue-clipping Cholesky for GP kernels** (`_safe_cholesky`): symmetrize, try a fast direct Cholesky with base jitter $10^{-8}$; on failure, one `eigh`, clip eigenvalues at 0, reassemble, add $\max(\text{base}, -\lambda_{\min} + 10^{-6})$, factorize. Two deterministic steps, no escalating-jitter loop; 3 to 5× faster on ill-conditioned kernels.
- **Shrink-then-repair for pairwise covariances** (`_bayes._shrink_psd`): diagonal shrinkage with auto intensity, eigenvalue floor at $\max(10^{-12}, 10^{-8}\lambda_{\max})$, then per-feature variance restoration so only correlations are perturbed (§7.2).
- **Graceful degradation:** `mvn_logpdf(_batch)` return $-\infty$ on Cholesky failure and NLLs convert that to $+\infty$ (rejected optimizer step); conditional solves fall back to `pinv`; Bayes classification falls back per-pattern to the diagonal model.
- **Probability clamps:** GH outputs clipped to $[10^{-12}, 1-10^{-12}]$ (or $10^{-15}$ at API boundaries); $\log$ arguments floored at $10^{-300}$; sigmoid via `scipy.special.expit`; log-sum-exp for all quadrature mixtures; `log1p`/`logaddexp` where relevant.

### 12.3 Determinism

Every model is fully deterministic given its inputs:

- Gauss-Hermite nodes/weights are computed once and cached (module dict in `_utils`, module constants in `_gp`, per-fit instance attributes in `_mixed`).
- GP hyperparameter restarts use `np.random.default_rng(0)`; SVC Platt scaling uses `random_state=0`; L-BFGS-B and Newton iterations are deterministic.
- KNN tie-breaking gathers all candidates at the $k$-th distance and stable-sorts, so ties break by ascending training index independent of BLAS reduction order.
- Randomness exists only where it is the point; `MissImputer` draws (seeded by `random_state`) and `MissEnsemble` bootstraps (per-estimator seeds derived from `random_state`; group-level cluster bootstrap with duplicate-subject relabelling when `groups` is supplied).

### 12.4 Computational complexity per model

Let $n$ = rows, $p$ = features, $G$ = distinct missingness patterns, $Q$ = GH nodes (20), $S$ = subjects.

| Model | Fit cost (dominant terms) | Notes |
|---|---|---|
| `_JointMVNFitter` (EM) | init $O(np^2)$ pairwise; per iteration $O(Gp^3 + np^2)$ | one solve per pattern per E-step; loglik check $O(Gp^3 + np^2)$ |
| `MissLinear` | EM above (≤500 iters) + polish ≤50 L-BFGS-B iters, each $O(Gp^3 + np)$ NLL with finite-diff over $O(p^2)$ params | SEs: $(p{+}2)$-dim numerical Hessian of the conditional NLL |
| `MissLogistic` / `MissRidgeClassifier` / `MissLASSOClassifier` | Stage 1 EM + Stage 2: $O(npQ + Gp^2)$ per NLL+grad, exact gradient, $p{+}1$ params | setup $O(Gp^3)$ once for $F$, $M$ tensor |
| `MissRidgeRegressor` / `MissLASSORegressor` | Stage 1 EM (on $[X\mid y]$) + Stage 2: $O(np + Gp^2)$ per NLL+grad | `bincount` reductions; $p{+}2$ params |
| `MissBayes*` | $O(np^2)$ fit (pairwise moments) + $O(p^3)$ repair; predict $O(Gp^3 + np)$ (regr.) or per class (clf.) | closed form, no optimizer |
| `MissNeighbors*` | fit: EM + $O(np^2)$ moments; predict: $O(n_q n_t p)$ dense distances + $O(n_q n_t)$ selection | expected-distance path is pattern-free (dense BLAS) |
| `MissSupport*` | kernel $O(n^2 p)$ (RBF, per feature folded into $F$ products) + LIBSVM $O(n^2)$-$O(n^3)$ | Platt adds internal 5-fold CV |
| `MissGaussian*` | per LML eval $O(n^2 p Q)$ kernel+grad $+\,O(n^3)$ Cholesky; ×(iters × restarts) | practical to $n \lesssim 1000$, the default `gp_max_n` above which `MissRecommender` vetoes the family; predict $O(n_q n p Q + n_q n^2)$ |
| `MissMixedRegressor` | per NLL $O(Gp^3 + np + \sum_i n_i)$; finite-diff gradient ⇒ ×$O(p^2)$ evals per L-BFGS-B iter | Woodbury keeps subjects $O(n_i)$ |
| `MissMixedClassifier` | as above with GH: $O(nQ)$ outcome term per NLL | probit approx keeps integral 1-D |
| `MissImputer.transform` | $O(Gp^3)$ per draw + $O(np^2)$ fills, ×$m$ (×refit under `posterior=True`) | patterns precomputed once outside the $m$-loop |

The universal scaling story: everything likelihood-based costs *one $p^3$ factorization per pattern, not per row*, plus BLAS-level per-row arithmetic; and the two-stage architecture confines iterative optimization to parameter blocks of size $O(p)$ with exact gradients.

---

## 13. References

- Arbuckle, J. L. (1996). Full information estimation in the presence of incomplete data. In G. A. Marcoulides & R. E. Schumacker (Eds.), *Advanced structural equation modeling: Issues and techniques* (pp. 243 to 277). Lawrence Erlbaum.
- Blom, G. (1958). *Statistical Estimates and Transformed Beta-Variables.* Wiley.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1 to 38.
- Eirola, E., Doquire, G., Verleysen, M., & Lendasse, A. (2013). Distance estimation in numerical data sets with missing values. *Information Sciences*, 240, 115 to 128.
- Ibrahim, J. G. (1990). Incomplete data in generalized linear models. *Journal of the American Statistical Association*, 85(411), 765 to 769.
- Ibrahim, J. G., Chen, M.-H., Lipsitz, S. R., & Herring, A. H. (2005). Missing-data methods for generalized linear models: A comparative review. *Journal of the American Statistical Association*, 100(469), 332 to 346.
- Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.
- Liu, H., Han, F., Yuan, M., Lafferty, J., & Wasserman, L. (2012). High-dimensional semiparametric Gaussian copula graphical models. *Annals of Statistics*, 40(4), 2293 to 2326.
- Liu, Q., & Pierce, D. A. (1994). A note on Gauss-Hermite quadrature. *Biometrika*, 81(3), 624 to 629.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. (Algorithms 3.1 and 5.1.)
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581 to 592.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys.* Wiley.
- Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods*, 7(2), 147 to 177.
