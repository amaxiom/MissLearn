# Changelog

All notable changes to MissLearn are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.9.2]: 2026-08

### Removed
- **`MissLinear._compute_se`**, a 47-line delta-method standard-error routine
  that was never called. `fit` computes standard errors inline instead, from a
  numerical Hessian of the reduced conditional likelihood, and `METHODS_GUIDE`
  had already recorded that this method was off the fitted path. Its siblings
  each keep a live `_compute_se` with a different signature and are unaffected.
  Archived under `_archive/linear_compute_se_removed_2026-08-21/`, since the
  repository is not under version control, with a note on the two things worth
  knowing before reviving it.
- Dead state in `_mixed.py`: `_INV_SQRTPI`, and the `_gh_t_` and `_gh_w_`
  attributes, whose only consumer was the hand-rolled prediction quadrature now
  replaced by `integrate_logistic_normal`.
- **Six single-stage leftovers**, 85 lines, in `_mixed.py`, `_logistic.py` and
  `_ridge.py`: `_predictor_nll_batched`, four per-class `_unpack_params`
  routines and one `_group_patterns`. These models were once fitted in one
  stage, with the MVN nuisance moments carried inside the optimiser's
  parameter vector, so each class needed a routine to read its own layout back
  out and the joint objective needed the predictor marginal added to it. They
  are now two-stage: the X-moments are estimated by EM first, which is still
  full information, and the objective that follows is conditional. The helpers
  lost their callers at that point and were left behind. `MissLinear` keeps a
  live `_unpack_params`, being the one class that still both defines and calls
  one. Verified dead before removal: no call sites in the package, its tests,
  its examples or its benchmarks; every defining class inherits from
  `MissBase` rather than from `MissLinear`; no dynamic dispatch reaches either
  name. Archived under
  `_archive/mixed_ridge_single_stage_helpers_removed_2026-08-26/`, with a note
  warning that the layouts encoded there are the old ones and no longer match
  what the current fit puts in `theta`. This is the counterpart of the LASSO
  removal archived on 21 August.

### Added
- **`check_missing_data_estimator`**, a public conformance checker for any
  NaN-tolerant estimator following the scikit-learn API. `check_estimator`
  verifies the scikit-learn contract but says nothing about incomplete data,
  because scikit-learn estimators mostly refuse it. This drives an estimator
  through eleven degenerate regimes and reports what it does in each,
  distinguishing a clear refusal (acceptable) from a silent `NaN` (never).
  Nothing in it depends on MissLearn.
- **Classifiers accept string class labels.** `fit(X, ['cat', 'dog'])` works,
  `classes_` reports the original labels and `predict` returns them. NaN in `y`
  is preserved rather than encoded as a category, since a missing label is not
  a class.
- **`validate_input`**, applying scikit-learn's own `check_array` with NaN
  permitted. MissLearn accepted one-dimensional `X`, empty arrays, single
  samples, sparse matrices, infinities, a continuous target handed to a
  classifier and `y=None`, all of which the contract requires an estimator to
  reject. scikit-learn's validator is reused rather than reimplemented because
  several checks assert on the message text.
- **Sphinx documentation site** (`docs/`) with a generated API reference
  covering all 39 exported names, the existing guides rendered through
  `myst-parser`, and a `sphinx-gallery` example gallery.
- **`CONTRIBUTING.md`**, issue and pull-request templates, and a documented
  deprecation policy.
- **`CODE_OF_CONDUCT.md`**, Contributor Covenant 2.1, linked from
  `CONTRIBUTING.md` and rendered on the documentation site. It ships with its
  reporting address unset and marked `CONTACT_ADDRESS_TO_BE_SET`, with a note
  listing the three reasonable choices. A code of conduct whose contact
  reaches nobody invites a report and then drops it, so the placeholder is a
  deliberate blocker rather than an oversight.
- **`.github/workflows/docs.yml`**, which publishes the documentation to
  GitHub Pages from `main`. The existing `docs` job in `ci.yml` built the site
  and uploaded it as a downloadable artefact, which proves it compiles but puts
  it nowhere a reader can reach. The build is repeated in the new workflow
  rather than shared, because a deploy that consumes another workflow run's
  artefact can publish a build made from a different commit. Requires one
  manual step before its first run: Settings, Pages, Source set to "GitHub
  Actions".
- **`docs/Makefile` and `docs/make.bat`**, both carrying a `strict` target that
  reproduces the `-W --keep-going` flags CI gates on, so a warning is found
  before the push rather than in the failed run.
- **870 tests**, taking the unit suite from 727 to 1,600 and coverage from
  89.7% to **95.0%** of statement-and-branch units. Two things are worth recording about how, because the second
  half cost a fraction of what the first half's rate predicted.

  Broad sweeps saturate. Discovering every estimator from the package and
  driving each through `summary`, `predict_interval`, `decision_function`
  and `score` was 692 tests for 247 units, because most of what they touched
  was already covered. Targeted work, reading each uncovered branch and
  constructing the state that enters it, ran at 2 to 3 units per test, seven
  times better, and took the last 4.8 points in 260 tests rather than the
  thousand the sweep rate implied.

  The single most productive shape was the **entirely absent row**. Every
  family splits prediction into complete, partial and empty, and fixtures at
  10 to 15 percent missingness produce the first two and essentially never
  the third, so the widest interval any of these models can produce was the
  one nothing tested. Supplying it reached `predict_interval` across seven
  regressors, `decision_function` on the LASSO classifier, the neighbour
  fallback when a neighbourhood has no labels, and the zero random intercept
  for an unlabelled subject. It also surfaced two of the defects below.

  Where a fixture has to reach a rare branch, the tests assert their own
  precondition first. The ensemble class-alignment test originally used a 3
  percent class over twelve members, every member saw all three classes, the
  branch never ran and the test passed anyway.

- **39 of those tests**, in the first pass, took the unit suite from 727 to
  766 and coverage from 89.7% to 90.4%. They were chosen by ranking
  modules on absolute uncovered units and then on the length of each
  contiguous uncovered run, because twenty consecutive missed lines are
  usually one untested method reachable by one test whereas twenty scattered
  singles are twenty separate branches. What they found was not obscure:
  the LASSO family's `se_` path had never run, because that family ships
  `compute_se=False` where every other family ships `True`, so nothing in the
  suite had ever asked it for a standard error. `miss_cross_val_score` and
  `miss_cross_validate` carry separate copies of the per-fold `fit_params`
  slicing logic and neither was tested, which matters because a mis-sliced
  weight vector still fits, it just fits the wrong thing. Three `prefit_check`
  advisories, all four `MissSensitivity.verdict` bands, `MissDiagnostic.summary`
  running its four analyses lazily, `MissEnsemble.summary`'s compact form for
  bags of more than six identical members, and `MissMixedClassifier.score`
  with no observed outcome were all unreached.
- **Coverage measurement** in CI, branch coverage rather than line, because
  every defect this project has shipped lived in a branch ordinary data never
  reaches.
- `n_iter_` on the estimators that expose `max_iter`.

### Fixed
- **`MissMixedClassifier` put its quadrature nodes on the prior instead of the
  posterior.** The subject likelihood integrates over the random intercept, and
  plain Gauss-Hermite spreads its nodes at width `tau` about zero. A subject
  carrying several observations has its own effect pinned into a much narrower
  interval, which may sit far from zero, and the nodes then sample where the
  integrand is negligible. The error grew with `tau` *and* with observations per
  subject, which is the opposite of the usual intuition that more data makes an
  integral easier: against a dense reference the rule was out by 4e-10 at
  `tau = 1` with one observation, by 1.8 at `tau = 5` with twenty, and by 17 at
  `tau = 12`. Raising the node count barely helped, because the problem is where
  the nodes sit rather than how many there are; at `tau = 3` with twenty
  observations even 320 nodes left 6e-4.

  The integral is now taken by adaptive Gauss-Hermite (Liu and Pierce 1994),
  recentring each subject's nodes on its own posterior mode and scaling them to
  the curvature there. The three cases above come out at 1.1e-14, 7.7e-6 and
  3.3e-8, so 20 adapted nodes beat 80 unadapted ones by several orders of
  magnitude throughout. The log-integrand is strictly concave, since every term
  of its second derivative is negative, so the mode is unique and Newton from
  zero reaches it in a handful of steps without safeguarding. With one node the
  rule is the Laplace approximation, and the unadapted rule is the special case
  of a mode at zero and a width of `tau`.

  The same adapted nodes are reused for the BLUPs. Predicting for a subject not
  seen in training carries no likelihood factor at all, so it reduces to the
  plain expectation of a logistic over the random-effect prior, and now goes
  through `integrate_logistic_normal` rather than repeating a quadrature by
  hand. No gradient work was needed: this fit has always used numerical
  differencing, so the change is confined to the objective's value.

  Random effects of the size ordinarily seen, `tau` below about 1.5, were never
  materially affected and are unchanged. `n_quadrature` now sizes an adaptive
  rule, and the fitted coefficients no longer move with it.

  The mode is found by Newton once per subject per objective evaluation, which
  costs about 1.8 times the previous fit time for this estimator, measured
  across 15 to 60 subjects and 8 to 20 observations each. No other family is
  affected.
- **A standard error that could not be computed reported certainty instead.**
  Delta-method and inverse-Hessian variances come out negative when the
  Hessian is not positive definite or the Jacobian is ill conditioned, which
  is a computation reporting its own failure. Eight sites across `_linear`,
  `_logistic`, `_ridge`, `_lasso` and `_mixed` took
  `np.sqrt(np.maximum(np.diag(Var), 0.0))`, so that failure was floored to
  zero and printed as a standard error of `0.0000` with a confidence interval
  of zero width: the most confident statement the table can make, arrived at
  because the arithmetic broke.

  Two identical predictor columns produce it, their individual coefficients
  being unidentified while their sum is not. `MissLinear` reported a standard
  error below 0.5 on that pair in 7 to 16 draws out of 20, across n from 40 to
  300 and missingness from 0 to 25 per cent, once printing coefficients of
  -223.20 and +225.04 with intervals of zero width around each.
  `MissRidgeRegressor`, whose penalty genuinely determines the split, was
  never wrong in any of those twelve configurations.

  All eight now route through `standard_errors_from_variance`, which returns
  NaN for any diagonal entry that is not positive. That is the reading the
  library already uses for a coefficient it cannot identify, and
  `z_stats_` and `pvalues_` already derived from
  `np.where(se > 0, se, np.nan)`, so they were NaN in these cases while the
  standard error printed beside them was not; the three now agree. The rate is
  0 of 20 in all twelve configurations. Exact zeros take the same route as
  negatives, deliberately: a standard error of exactly zero asserts a
  coefficient known without error, which no finite sample supports.

  Coefficients, predictions and `loglik_` are untouched, and a well
  conditioned fit reports exactly what it did before. Where a variance matrix
  is ill conditioned its negative entries are not always confined to the
  unidentified coefficients, so an identified coefficient in the same fit can
  also have its standard error withheld; measured across 15 seeds at each of
  six configurations it kept it in 8 to 14 of them. Withholding is the
  conservative direction, and the alternative is to report a number the
  computation did not produce.
- **Fitting and prediction now use the same marginalisation rule.** The entry
  below fixed `integrate_logistic_normal`, which prediction goes through, but
  the likelihood optimised during `fit` built its own 20-node Gauss-Hermite
  rule alongside an analytic gradient, so the two disagreed and the fit was
  biased where the signal was strongest: on a model with `|beta|` up to 5.3 the
  coefficients moved by 0.16 between 20 and 320 nodes. `MissLogistic` and
  `MissLASSOClassifier` now call `logistic_normal_with_grads`, which returns
  the value together with `dp/da` and `dp/dv`, both obtained in closed form by
  differentiating the split under the integral sign and both reusing the
  density already formed for the value. The chain rule out to the coefficients
  is unchanged and stays in each estimator, since it differs between them.
  Coefficients now move by exactly zero as the node count rises, and the
  default 20-node fit lands on what 320 nodes were previously needed to reach.
  `n_quadrature` still sizes the small-variance branch, where 20 nodes are
  accurate to 1e-11, but it no longer changes the answer.

  Cost is unchanged where the old rule was already right (rows with a small
  marginalised variance take the same Gauss-Hermite path, and skipping the
  wide branch entirely makes them marginally faster), and two to six times per
  objective evaluation where it was wrong.

  `MissMixedClassifier` is deliberately not included. Its quadrature is over
  the random effect rather than the missing features, and its integrand is a
  product over a subject's observations rather than a single logistic, so
  there is no step to peel off and the split does not apply. Its relative error
  is 4e-10 at tau = 1 and 1.7e-3 at tau = 2, so ordinary random effects are
  unaffected; a tau large enough to matter signals near-separation within
  subjects. The remedy there is adaptive Gauss-Hermite, which recentres each
  subject's nodes on the mode of its own integrand, and is separate work.
- **The half-width of the correction integral was too wide to be accurate.**
  The remainder it integrates is bounded by `exp(-|s|)`, so a half-width of 40
  spread the nodes over ground the integrand had already vacated. Narrowing it
  to 25 improves the worst error from 3.5e-9 to 1.9e-13 at no cost at all, the
  same node count doing better work, and makes the crossover between the two
  rules agree to 1.8e-11, limited by Gauss-Hermite rather than by the split.
- **Skewness and excess kurtosis were not scale-free, though both are
  dimensionless.** `_skew_kurtosis` built them from raw central moments, so the
  arithmetic depended on the units even though the answer does not: near 1e120
  the cube overflows while the square is still finite, giving `inf/inf`, and
  below about 1e-150 both underflow, giving `0/0`. Either way the result was
  `NaN`, and `needs_copula` reads `abs(NaN) > 1.0` as `False`, so a visibly
  skewed column silently stopped asking for the copula once its units were
  large enough; above 1e160 the square overflowed too and the constant-column
  guard returned the same silent answer by another route. A column with
  skewness 2.52 was reported as skewness 0 at a scale of 1e160. The moments are
  now made dimensionless before the powers are taken, and are exactly invariant
  from 1e-200 to 1e300 while still matching `scipy.stats.skew` and
  `scipy.stats.kurtosis` at their defaults. This matters because `needs_copula`
  runs on raw `X`, before the internal standardisation.
- **`odds_ratios_` warned on overflow in two of the three classifiers that
  report it.** A coefficient past about 709 means the predictor separates the
  classes, where the odds ratio is genuinely unbounded and `inf` is the reading
  rather than a fault. `MissMixedClassifier` already silenced the expected
  warning; `MissLogistic` and `MissLASSOClassifier` did not, and emitted a
  `RuntimeWarning` that could mask a real one. The value is unchanged, since
  clipping it would report a finite odds ratio for a variable that has none.
- **The marginalisation integral lost accuracy exactly where the signal is
  strongest.** `integrate_logistic_normal` computes `E[sigma(a+t)]` with
  `t ~ N(0, v)`, the integral every logistic prediction with missing features
  goes through, and did it by Gauss-Hermite quadrature on 20 nodes. That fails
  structurally rather than for want of nodes: in the quadrature variable the
  integrand is a step of width about `1/sqrt(2v)`, and raising the count from
  20 to 640 still leaves an error above `1e-4` at `v = 100`. The worst error
  over `a` in `[-6, 6]` reached 0.004 by `v = 25` and 0.03 by `v = 100`, in a
  returned probability. `v` is the variance of the missing features' linear
  contribution, so it grows with both how many features are missing and how
  large the coefficients are: across the missingness patterns of a fitted model
  with `|beta|` up to 3.6 the *median* `v` was 23, and at `|beta|` up to 9.9 it
  was 121. Wide variances are now handled by splitting the logistic into a unit
  step plus a remainder; the step integrates exactly to `Phi(a/sqrt(v))` and the
  remainder is bounded by `exp(-|s|)`, so its support does not widen with `v`
  and Gauss-Legendre on two panels meeting at the jump resolves it. Accurate to
  about `1e-16` for every `v` from 0.1 to 2000, against 0.09 before at the top
  of that range. Gauss-Hermite is retained below `v = 1`, where it is already
  exact to `1e-11` and the split rule is the weaker of the two.

  This corrects prediction. The likelihood optimised during `fit` builds its own
  Gauss-Hermite nodes alongside an analytic gradient, in `_logistic.py`,
  `_lasso.py` and `_mixed.py`, and still uses the 20-node rule; moving it to the
  split scheme means rederiving those gradients and has not been done here.
- **The copula transform mishandled every repeated value.** `np.interp` requires
  a strictly increasing first argument, and `RankNormalTransformer` was handing
  it the raw sorted observations, which repeat whenever a column is discrete.
  Each tied value then resolved to its block's *largest* normal score instead of
  the block mean. On a binary column that mapped `0` to `+1.18` and `1` to
  `+3.12`, producing a column of mean 1.41 and standard deviation 0.62 where the
  transform is defined to deliver `N(0, 1)`. Which score a block landed on moved
  with the block boundaries, so the damage showed up as inflated fold-to-fold
  variance rather than as an error: on the air quality data every model family
  lost roughly half its R2 stability (standard deviation 0.16 against 0.08) with
  the transform engaged. Tied values now take the mean of the scores their block
  spans, the mid-rank convention, which also makes the interpolation table
  strictly increasing in both directions and the round trip exact.
- **The copula fired on indicator columns, which it cannot help.** An indicator
  with prevalence outside roughly `[0.25, 0.75]` has skewness above 1 by
  construction, so `needs_copula` triggered on one-hot dummies, and the
  transform then damaged them. Sklar's theorem identifies a copula uniquely only
  for continuous marginals. Columns with fewer than `MIN_DISTINCT_FOR_COPULA`
  (3) distinct observed values are now skipped when deciding and passed through
  untouched when transforming.
- **A sparse column no longer aborts a fit through the copula.** A column with
  fewer than two observed values raised `ValueError`, making the copula the only
  stage that could refuse data the rest of the library accepts: the same matrix
  fitted without complaint under `copula=False`. With `copula='auto'` the
  default, any cross-validation fold sparse enough to empty a column killed the
  fit, in a library whose subject is missing data. Such columns now pass
  through. A wholly empty column is still refused, by the shared
  `EmptyFeatureError` check that reports it properly.
- **`MissNeighborsRegressor` consulted `y` when deciding but never transformed
  it.** It predicts by averaging neighbouring `y` on the original scale and uses
  the copula to calibrate feature distances, so a skewed target was triggering a
  feature transform that had nothing to do with it. It now passes `X` alone,
  matching what it does. A test guards the divergence itself: every estimator
  whose source calls `needs_copula(X, y)` must also hold a `_copula_y_`.
- **`psd_jitter`**: `MissLinear` used a fixed `1e-10` conditioning constant and
  was the only regressor that failed at 90% missingness, where the EM
  covariance is marginally indefinite and no constant that small can lift it.
  `MissLASSORegressor` already applied a spectrum-aware rule; that rule is now
  shared. Also promotes the 0-d array `np.cov` returns for a single column,
  which is what made `MissLASSORegressor` fail at p = 1 where its own
  classifier coped.
- **`MissLinear` refuses p >= n** instead of returning a meaningless fit. It
  estimates the full joint MVN: 902 parameters from 25 rows at p = 40. The
  conditioning fix above was tried here and rejected, because it removed the
  error only by substituting a slow optimisation over an unidentified
  likelihood. It now names `MissRidgeRegressor` and `MissLASSORegressor`.
- pytest discovery: the suites are named `*_test_suite.py`, so the default
  pattern matched nothing and `pytest tests/` exited with "no tests ran".

### Added
- **`MissLearn/_conformance.py`**: behaviour shared by every estimator, defined
  once and applied at a single wrapping point. Created because a user found
  `MissLASSOClassifier` raising on high missingness where `MissLogistic` and
  `MissLASSORegressor` degraded gracefully: a fallback present in both siblings
  had never been applied to that one class. Sixteen independent
  implementations of the same behaviour is the defect; per-class discipline
  cannot detect divergence because nothing compares the classes.
- **`tests/conformance_test_suite.py`**: every estimator driven through every
  degenerate regime, with `test_no_undeclared_sibling_divergence` asserting
  that a regime some estimators survive is expected of all of them. Found 13
  divergences on its first run, four of which a `predict`-only sweep had
  reported healthy.
- **CI** (`.github/workflows/ci.yml`): both suites on Python 3.9 and 3.12
  across Linux and Windows, plus a build job that installs the wheel into a
  clean environment and exercises it there.

### Changed
- **Binary classifiers now accept more than two classes**, routing through
  `MissMulticlass` internally rather than raising. Users are no longer required
  to know which wrapper to reach for, and `predict_proba` returns a correctly
  shaped `(n, K)` matrix. Binary behaviour is unchanged.
- **A fully missing feature column is now refused uniformly**, by every
  estimator, with `EmptyFeatureError` naming the column. Previously this single
  regime produced four different behaviours: four regressors fitted and
  returned `NaN` predictions; four classifiers returned finite class labels
  whose `predict_proba` was entirely `NaN`, so the label path concealed the
  failure; two raised without naming the column; three coped. A column with no
  observed values has no conditional distribution to marginalise over.
- Estimators now declare `allow_nan = True` and `sparse = False`. The first was
  wrong in a way that mattered: scikit-learn uses the tag to decide what to
  expect, so with it unset `check_estimator` required these estimators to
  *raise* on NaN input, and any third-party utility consulting it would have
  refused input they handle correctly.
- scikit-learn mixins now precede the base class in all 15 estimator
  declarations, as the estimator contract requires. Verified inert: all 16
  estimator scores are identical to eight decimal places before and after.

### Fixed
- `fit` no longer presents as `fit(X, *args, **kwargs)` to introspection. The
  pandas compatibility wrapper had replaced the signature, so scikit-learn
  rejected every estimator in the library with "Expected y or Y as second
  argument".
- `predict` now raises on a feature-count mismatch instead of silently
  producing predictions that correspond to no fitted model.
- Complex input is refused rather than silently cast to float, which had been
  discarding the imaginary part.
- **`MissMulticlass` crashed on a pandas nullable target.** `_to_binary`
  compared the whole column at once, and `y == positive_class` on a nullable
  dtype yields `pd.NA` in the absent positions rather than `False`, which then
  raises when cast to float. The comparison is now made on the observed entries
  only, and absent labels stay absent instead of becoming a class.
- **A failed cross-validation fold was skipped rather than recorded.**
  `miss_cross_val_score` used `continue`, so a fold that raised silently
  shortened the score vector: five folds requested, four returned, and the mean
  computed over whichever ones happened to work. It now appends `nan` (and
  `None` to the estimator list), so the failure is visible in the output and
  the vector length still matches the number of folds.
- **`MissImputer` used four public methods before checking it was fitted.**
  `transform`, `transform_mean`, `fit_transform_combine` and `summary` reached
  straight for fitted attributes, so calling them early raised `AttributeError`
  from deep inside the module instead of the `NotFittedError` the scikit-learn
  contract requires. Every estimator in the library already did this; this
  module was the exception.
- **`MissSensitivity.tipping_point` searched over non-finite coefficients.** A
  delta whose fit failed contributes `nan`, and `nan` comparisons are always
  `False`, so a tipping point could be reported from a sweep in which the
  relevant fits had not converged. Non-finite deltas are now excluded from both
  the sign-change and the interval branches, and a non-finite baseline is
  refused outright rather than compared against.
- **`get_metadata_routing` did not call `super()`.** The override returned its
  own router and stopped, discarding whatever the base class contributed, so
  routing worked by accident wherever the base had nothing to add and silently
  lost information wherever it did.
- **The Gaussian-process estimators reported no convergence status.** Both now
  expose `converged_`, built from `_optimiser_converged_` and, for the
  classifier, `_laplace_converged_`, and warn when every restart of the kernel
  optimiser fails rather than returning the initial kernel as though it had
  been fitted.
- **`MissNeighbors` predicted a row it knew nothing about more confidently
  than one it knew something about.** Measured against a response with
  standard deviation 2.27, interval widths for complete, partial and
  entirely absent rows were 1.11, 4.34 and 8.88 for `MissLinear`, and 1.78,
  3.82 and **2.98** for `MissNeighborsRegressor`: the likelihood families
  widened to roughly `2 * 1.96 * sigma_Y`, which is what total ignorance
  about a row costs, and the neighbour family narrowed instead. The
  mechanism was geometric rather than a missing guard. With no observed
  feature the expected distance under the fitted joint Gaussian is identical
  to every training point, so the k nearest neighbours are an arbitrary
  subset and the interval reported that subset's spread rather than the
  response's. The fallback needed already existed and simply never fired:
  `_aggregate` uses the marginal spread when no neighbour is reachable and
  again when no neighbour has an observed outcome, and an entirely absent
  row meets neither condition, because it is the query that carries no
  information rather than the neighbourhood. The absent row now widens to
  8.93. The point prediction is deliberately unchanged: it already lands on
  the marginal mean, and `predict` is on the path of every benchmark in the
  repository.
- **An entirely absent outcome column got three different answers.**
  `MissRidgeRegressor` refused with a clear message, `MissLinear` fitted and
  then predicted all `NaN` from `NaN` coefficients, and `MissMixedRegressor`
  fitted a zero model and predicted finite numbers from no data at all. The
  library's own conformance contract names the second as the failure worth
  removing first, because an exception stops a pipeline while a silent `NaN`
  propagates into whatever consumes the prediction; the third is worse
  again, since a finite number from no data reads as an answer. The refusal
  had been worded in `prefit_check` since it was written ("y is entirely
  NaN. Nothing to fit."), but the estimators did not all call it. It now
  lives in `validate_input`, which every fit reaches through
  `_pandas_compat`, so all sixteen estimators give the same answer. A
  partly absent outcome is unaffected, which is the capability the refusal
  had to avoid taking with it.
- **`MissEnsemble` refused a subclass of its own estimators.** `_is_misslearn`
  decided membership from `type(est).__module__`, so a user subclass of
  `MissLogistic` declared anywhere else was turned away with a message
  saying it was not NaN-native, which was both wrong and unhelpful: a
  subclass handles `NaN` exactly as well as its parent. It now asks
  `isinstance(est, MissBase)`, keeping the module test as a fallback for
  anything inside the package that does not inherit from it. A plain
  `LogisticRegression` is still refused.
- **`MissRecommender` never detected heavy tails, so it never advised the
  copula.** `prefit_check` files a kurtosis finding with `_add_note`;
  `_recommend` searched `chk.warnings`. The two lists are disjoint, so
  `heavy_tailed` was `False` on every dataset however heavy the tails, and
  everything downstream of that flag was unreachable: `preprocessing_['copula']`
  could never be `True`, `make_estimator()` never set `copula=True` on what it
  built, the score adjustments favouring `MissNeighbors` and `MissSupport` and
  penalising `MissBayes` on heavy-tailed predictors never fired, and `summary()`
  never printed the line saying so. Measured on t-distributed columns with
  excess kurtosis 16.5 and 64.6 against the default threshold of 7.0:
  `prefit_check` flagged both and the recommender reported `copula False`. It
  now reads notes as well as warnings, which also means a future
  reclassification of either cannot break it the same way. The neighbouring
  `scale_imbalance` check was unaffected, because a scale finding really is a
  warning; that asymmetry is what made this survive.
- **`MissEnsemble` could not see two of its own estimators.**
  `_validate_task_consistency` decided whether a member was a regressor or a
  classifier by looking for those words in the class name. That is true of
  `MissRidgeRegressor` and false of `MissLinear`, which is equally a
  regressor and simply does not carry the suffix; `MissLogistic` is the same
  case on the classifier side. The two estimators a user is most likely to
  reach for were the two the guard could not see. The result was not a
  missing warning but a deferred crash: `fit` accepted a `MissLinear` inside
  a classification ensemble, and `predict_proba` then died inside
  `_collect_proba` with `AttributeError: 'MissLinear' object has no attribute
  'predict_proba'`, raised from library internals rather than at the door,
  while the same mistake made with `MissRidgeRegressor` was refused
  immediately with a clear message. The check now asks scikit-learn's
  `is_regressor` and `is_classifier`, which read the estimator type tag that
  every class in the library sets correctly through its mixin, so the answer
  no longer depends on what the class is called.
- **A mixed model that failed to converge returned in silence.** Both
  estimators in `_mixed.py` set `converged_ = False` and carried on, so a fit
  that had exhausted its iteration budget was indistinguishable from a good one
  unless the caller inspected the attribute or called `summary()`. They now
  warn, as the Gaussian-process estimators already did. This was found by
  pinning `copula=False` on the Parkinson's data: the optimiser then converged
  on only two cross-validation folds of five, and the coefficients it stopped
  at reached 3.6e4 against 0.6 with the copula applied. The failure is not
  loud. On one missingness realisation the non-converged folds still produced a
  plausible RMSE of 10.16; on the example's own realisation the same setting
  produced 38.36 with a fold standard deviation of 56.60. A number that looks
  reasonable is not evidence that the fit converged, which is exactly why the
  warning belongs at the point of failure.

### Documentation
- **`MissMixedRegressor.max_iter` was documented as a ceiling and behaves as a
  floor.** The budget actually used is `max(max_iter, 300 * (p + 3))`, which
  from `p = 4` upward already exceeds the documented default of 2000, so
  lowering the parameter had no effect and the stated default never bound. The
  scaling is deliberate, because the reduced objective uses a numerical
  gradient costing `2(p + 3)` evaluations per step; only the docstring was
  wrong. Its sibling `MissMixedClassifier` passes `max_iter` through
  unmodified, so the two estimators still interpret the same parameter
  differently.
- **`MissMixedClassifier`'s class docstring still described plain Gauss-Hermite
  quadrature** after the switch to the adaptive rule. The parameter
  documentation had been updated; the class summary had not.
- **A broken cross-reference kept the documentation site unbuildable.**
  `COMPUTATIONAL_GUIDE.md` linked to `#3-fiml-for-logistic-regression`, a
  section under no such name; the integral it means is derived in 3.2. Because
  the docs build with `-W`, this single dangling anchor failed the whole build,
  so the site could not have been published until it was fixed.
- **`COMPUTATIONAL_GUIDE.md` described the mixed-effects objective as it used
  to be.** It said the predictor marginal is added inside the objective by
  `_predictor_nll_batched`. That was true of the single-stage fit; the model
  has been two-stage for some time, and the helper named in the sentence has
  now been removed, so the guide would have pointed at a function that does
  not exist. The corrected passage says where the marginal is actually
  accounted for, and says explicitly that the estimate is still full
  information, because an objective with no marginal term in it reads like a
  loss if the two-stage split is not stated alongside it.
- **A guard in `docs/conf.py` had never done anything.** When `sphinx-gallery`
  is unavailable the file computes `exclude_patterns_extra` to keep the
  `auto_examples` toctree entry from dangling under `-W`, then assigns
  `exclude_patterns` afterwards without consulting it, discarding the result.
  On a machine without the extension the build failed exactly as it would have
  with no guard at all. The two are now merged.

### Changed
- **The declared scikit-learn floor is now 1.6**, raised from 1.1. It was
  wrong rather than merely conservative: `MissTags.__sklearn_tags__` calls
  `super().__sklearn_tags__()`, and `_sklearn_compat` feeds a `check_estimator`
  keyword, and both arrived in 1.6. Installing against 1.1 through 1.5 gave a
  package that imported and then misreported its tags, which is worse than one
  that refuses to install. No upper bound is set. The continuous integration
  matrix resolves scikit-learn 1.6.1 on Python 3.9 and 1.9.0 on Python 3.12,
  so both ends of the supported range are now exercised on every run. That
  coverage is what was missing when the incompatibility below went unnoticed.

### Fixed
- **Compatibility with scikit-learn 1.7 through 1.9.** `_estimator_type` was
  removed in scikit-learn 1.9. Six places in the package read it through
  `getattr(est, "_estimator_type", None)`, so its removal raised nothing: the
  expression became `None`, every comparison against it became false, and the
  guards stopped guarding while continuing to report success. Multiclass
  routing no longer fired, so a binary classifier given three classes raised
  "requires exactly 2 classes" instead of decomposing into one-vs-rest; the
  continuous-target rejection stopped rejecting; and the pandas label handling
  was skipped. All six now ask scikit-learn through `is_classifier` and
  `is_regressor`, which read the tag on new versions and the attribute on old
  ones. This is the same shape as the `MissEnsemble` membership defect fixed
  above: a capability question answered by inspecting an attribute rather than
  by asking, and failing silently when the attribute moved.
- **`MissMulticlass` was not a scikit-learn estimator.** It was a plain class
  with no `BaseEstimator` in its ancestry and no `__sklearn_tags__`, which
  scikit-learn 1.6 tolerated by duck typing and 1.9 does not. It now inherits
  `ClassifierMixin`, `MissTags` and `BaseEstimator`. Sitting outside the reach
  of `check_estimator` had hidden four further defects, all fixed here:
  `predict`, `predict_proba` and `decision_function` never called
  `check_is_fitted`, so an unfitted call raised `AttributeError` from inside
  the class rather than `NotFittedError`, which is the same defect corrected in
  `MissImputer` earlier in this release; `__init__` validated `strategy`, which
  breaks `clone` and rules the parameter out of a grid search, so that check
  moved to `fit`; `estimator` had no default, so the class could not be
  constructed bare, and it now falls back to `MissLogistic`; and
  `n_features_in_` and `feature_names_in_` were never set. Making it a
  classifier also meant `route_multiclass` would route a router into another
  router without end, so routing now declines for anything marked
  `_is_multiclass_router`. The lesson is the one this suite keeps teaching: the
  class the contract could not see is the class that drifted from it.
- All 21 estimators pass `check_estimator` on scikit-learn 1.6.1 and 1.9.0.
- **Absent class labels were rejected by every classifier**, and had been
  before this release. `encode_labels` is the one function every classifier
  passes its labels through, and it tested for an absent entry with
  `isinstance(v, float) and np.isnan(v)`, which is true of float `nan` and of
  nothing else. `None`, `pandas.NA` and `pandas.NaT` all survived the filter,
  reached `np.unique` and killed the sort: `TypeError: '<' not supported
  between instances of 'str' and 'NoneType'`, or `boolean value of NA is
  ambiguous` for the pandas forms. So `MissLogistic().fit(X, y)` raised for an
  object or nullable-dtype label column with an absent cell, which is how an
  incomplete label column arrives from a real file, while the documentation
  said an absent outcome is supported and still informs the feature
  distribution.

  It survived because `MissMulticlass` held the only correct implementation,
  in its own `_is_nan`, and never reached the shared one: it was not
  recognised as a classifier, so the label-encoding path skipped it. The class
  whose tests covered exactly this input was the class excluded from the code
  under test. Making it a classifier, above, routed it through the shared
  helper and the defect surfaced at once.

  There is now one definition, `is_missing_label` in `_conformance`, and
  `_multiclass._is_nan` delegates to it rather than keeping a second copy.
  `TestAbsentLabelsAcrossTheLibrary` asserts the claim of all seven
  classifiers rather than one, with all four spellings of absence, because a
  per-class test cannot find a defect in the helper the classes share.
- **`MissMixedRegressor` silently discarded its `groups` on scikit-learn 1.7
  and later.** `MissBase` defined `set_fit_request` and `set_predict_request`
  as no-op stubs that warned and returned `self`, on the stated reasoning that
  scikit-learn's generated method takes precedence. That was true on 1.6 and
  false from 1.7, which skips generation when the attribute already exists in
  the MRO, so the stub won: `set_fit_request(groups=True)` recorded nothing,
  and a `Pipeline` or `cross_validate` given `groups` then raised
  `UnsetMetadataPassedError`. `get_metadata_routing`'s own docstring records
  what was at stake, tau 2.71 against 0.00 on the same data, since a random
  intercept fitted without its groups collapses to an ordinary regression.
  Both stubs are removed, so scikit-learn generates the real method for the
  estimators that have metadata and the ones that have none simply do not
  carry it, which is what `LinearRegression` and `PCA` respectively do.
- **`is_classifier` and `is_regressor` are no longer called bare.** Replacing
  the `_estimator_type` reads with them, above, fixed the silent-false problem
  and introduced a raising one: from 1.7 those functions go through `get_tags`,
  which raises `AttributeError` when nothing in the object's MRO defines
  `__sklearn_tags__`. Eleven public classes are not `BaseEstimator`
  subclasses, so each turned a question into an exception, and the unit suite
  went from green on 1.6.1 to 179 failures on 1.9. `is_classifier_safe` and
  `is_regressor_safe` in `_sklearn_compat` restore the totality 1.7 removed
  and are used at all six sites. They catch `AttributeError` only, so a
  genuine failure inside a real `__sklearn_tags__` still propagates.
- **`MissImputer` was a plain class**, like `MissMulticlass` before it, and no
  compatibility shim could cover it: scikit-learn's own `check_is_fitted`
  calls `get_tags` from 1.7, and `MissImputer.transform` calls
  `check_is_fitted(self)`. It now inherits `MissTags` and `BaseEstimator`, and
  deliberately not `TransformerMixin`, since `transform` returns m completed
  datasets rather than one array and should not claim a contract it does not
  meet.
- `np.trapz`, removed in numpy 2.x, was used twice in the test suite for
  reference integrals. Both go through a shim that prefers `np.trapezoid`, so
  the file runs on numpy 1.26 and on 2.x alike. A scan of every `np.` attribute
  used anywhere in the package, tests, examples and benchmarks against numpy
  2.4 found no others.
- The library-wide sweeps selected estimators by `BaseEstimator` plus `fit`,
  which had meant "has predict" only by accident, since the classes without
  predict were not `BaseEstimator` subclasses. Giving `MissImputer` its
  identity changed the membership as a side effect and twelve sweeps called
  `predict` or `score` on a multiple-imputation transformer that has neither.
  The predicate now requires `predict`, which restores the previous membership
  and states the requirement instead of resting on a coincidence.

- **`MissImputer.fit` never set `n_features_in_` or `feature_names_in_`**, so
  nothing recorded the shape or the column names it was fitted on. Both are
  set now, from the features the caller passed rather than from the joint
  matrix, which with `include_y=True` also carries the response.
- `MissImputer` is declared in a new `CONTRACT_EXEMPT` in `_sklearn_compat`,
  with the reason. Giving it a scikit-learn identity brought it into the
  continuous-integration contract sweep, where scikit-learn sees `transform`
  and requires the transformer contract: one array of shape
  `(n_samples, n_features_out)`. It returns a list of m completed datasets,
  which is what multiple imputation is, and collapsing them to one array would
  discard the between-imputation variance that makes the estimate honest.
  Declaring `transformer_tags` to get past discovery would state a contract
  the class does not meet. The workflow still enumerates everything and then
  subtracts a named list, so an exemption is a visible decision rather than a
  class quietly missing, and `TestContractExemptionsStayHonest` fails if an
  exempt name stops existing, would not have been discovered anyway, carries
  no argued reason, or starts passing the contract after all.
- `test_prediction_is_subset_invariant` compared predictions with
  `np.allclose` and no `equal_nan`, while the helper eleven lines above it in
  the same file passes `equal_nan=True`. Any generated design that made a
  prediction `NaN` failed the assertion even though both arrays were
  elementwise identical, so the test was flaky by construction, passing or
  failing on which examples hypothesis happened to draw. The claim it makes is
  invariance; finiteness is a separate claim made by the conformance suite.
- `test_fit_emits_no_numerical_warning` failed the `MissLinear` collinear cell
  with "declared but no longer warns; delete the entry". The entry is not
  stale. The warning is raised inside scipy's finite-difference gradient,
  which the declaration itself says, and it appears on scipy 1.17 and not on
  1.13, which is what Python 3.9 resolves. Deleting it would make every
  newer-scipy environment fail the undeclared-warning assertion instead. That
  branch is now a skip naming the numpy and scipy versions. The check that
  catches new problems, that an undeclared cell must not warn, is unchanged.
- `TestMetadataRequestStubsWarn` asserted that the no-op stubs warned, on the
  reasoning that "metadata routing failing open is how a sample weight quietly
  stops being applied". The reasoning was right and the guard was aimed one
  level too high: the stubs did warn, and they also shadowed the real
  generated method, so routing failed open underneath a test written to
  prevent it. It is replaced by
  `TestNothingShadowsGeneratedRequestMethods`, which asserts that no class in
  the package defines a `set_*_request` of its own, by ownership rather than
  presence, since scikit-learn installs its generated ones into the class
  dictionary too.

### Documentation
- **The user guide warned about a defect that had already been fixed.** It
  said the `allow_nan` tag "currently reports `False`, even though the
  estimators do accept `NaN`", and that a third-party utility consulting it
  "may refuse input it could have handled". It reports `True`, and has since
  `MissTags` was introduced. Corrected, along with the stated MRO, which had
  `BaseEstimator` in the wrong position, and the claim that the estimators
  report `_estimator_type`, an attribute scikit-learn removed in 1.9.
- The methods guide listed `set_fit_request` and `set_predict_request` among
  what `MissBase` provides. It no longer provides either, and the entry now
  says why.
- The declared scikit-learn floor is stated as 1.6 in the user guide's
  requirements, matching `pyproject.toml`.

## [0.9.1]: 2026-07


### Added
- New example `09_secom_blockwise.py`: UCI SECOM semiconductor yield, 1,567
  lots by 590 channels. It is the large-p case, and it demonstrates that
  **blockwise missingness is what makes large p reachable**. All 474 surviving
  channels fall into 36 co-missing groups, giving G=198 distinct patterns at
  p=474; on a synthetic control at matched rate, scattered missingness gives
  G=1,499 and a 324 second fit while blockwise gives G=32 and 2.8 seconds, a
  116-fold difference from structure alone. The measured cost ladder
  (p=20 0.6s AUC 0.613; p=40 60s 0.731; p=60 171s 0.765; p=90 2564s 0.809) is
  reported with its ceiling stated: the empirical scaling is about 8 times worse
  than O(G p^3) predicts, and p=474 is out of reach.
- New example `07_galaxy_redshift.py`: a head-to-head against a published
  imputation study (arXiv:2111.13806) on its own catalogue and protocol, where
  FIML beats the method that study reports as best at every missing rate from
  2% to 25%, and where the MNAR non-detection case that study defers to future
  work degrades MICE to the level of mean imputation.
- New example `08_graphene_oxide.py`: structural absence, where a descriptor is
  missing because the bond motif does not exist, so imputation invents
  chemistry. Also shows that compositional closure makes the design matrix
  singular before any missingness is involved.
- New example `10_heart_disease_indicators.py`: a head-to-head against "No
  imputation without representation" (arXiv:2206.14254), which argues for adding
  binary missing-indicators alongside imputation. On UCI Heart Disease across
  all four collection sites the indicators are close to functions of the site,
  and site prevalence spans 57 percentage points, so most of their apparent
  value is confounding: `fbs` and `ca` lose their association with the outcome
  entirely once site is conditioned on, while `slope` genuinely survives.
  Indicators are worth +0.0052 AUC, site alone +0.0163, and indicators given
  site -0.0039. Modelling the mechanism wins: `FIML + site` 0.9027 and
  `MissMixed` with a site random intercept 0.9021.
- **`MissRecommender`** and the `recommend_model()` wrapper: evidence-based
  model triage. Gathers the missingness mechanism (Little's test, MAR
  plausibility), the shape of the problem, tail behaviour, grouping structure
  and a cheap linearity probe, then ranks the eight model families **with the
  reasoning attached**. Also reports columns that should be dropped rather
  than modelled, sets `copula=True` when tails demand it, separates vetoes
  (a Gaussian process above `gp_max_n`, mixed effects without `groups`) from
  merely low scores, and marks `MissSensitivity` as *required* when MNAR
  cannot be excluded. Documented in user guide 6.3; worked end to end in
  `examples/06_Guided_Workflow_Air_Quality.ipynb`.
- `MissSensitivity.fit()` now accepts `feature_names`, so the summary table
  is labelled with real variable names.
- `MissPreprocessor` accepts `feature_names` as well, and otherwise resolves
  names in priority order: an explicit argument, then the DataFrame columns,
  then an existing `feature_names_in_`, and only then generic `X0..Xp`. Its
  warnings previously always named `X4` rather than the measurement.
- New examples: `06_Guided_Workflow_Air_Quality.ipynb` (native missingness
  from three distinct mechanisms in one file, plus the guided workflow).
- The example suite is now ten examples, and each ships both a notebook and
  a command-line script, so every one can be read interactively, run
  headless, or diffed.
- Every example now doubles as a mini benchmark against dropping and
  imputing, with the model class held fixed.

### Changed
- No comparison anywhere in the repository crosses model classes. The
  tree-ensemble arms have been removed from `examples/02`, from
  `examples/real_data_fair_benchmark.py` and from the results tables, since
  a boosted tree beating a logistic model measures capacity rather than
  missing-data handling. `MissEnsemble` still accepts NaN-native third-party
  learners as *members*, which is a library capability and not a comparison.
- **A fourth fairness control in the benchmark suite: a matched noise floor.**
  `MissGaussianRegressor` fits a noise variance as a hyperparameter, but the
  conventional counterpart was a bare RBF with `alpha=1e-10`, so it had no
  noise floor and had to interpolate noisy data exactly. On an incomplete
  sample the imputed rows form near-duplicate inputs with different targets,
  the kernel matrix goes near-singular, and the predictions explode: R2 of
  -3.26 at n=300, falling to -49.44 at n=1000. That reads as a large FIML win
  and is entirely a misconfigured comparator. Both arms are now given a
  learnable noise term. This joins the three existing controls: model class
  held fixed, matched preprocessing, and matched regularisation.
- The Gaussian-process kernel uses the scikit-learn default bounds, matching
  the published script. Tightening them was tried and is harmful: starting
  the noise level at 1e-2 with a lower bound of 1e-8 sends the optimiser into
  a degenerate all-noise optimum on three of five folds, where the fitted
  noise reaches the response variance and the model predicts the mean.
- The benchmark registry's `small_only` flag is replaced by a numeric `max_n`
  row cap, so a family too expensive for the full task runs every task with
  its rows capped rather than dropping to the smallest one alone.

### Fixed
- **`MissExplainer` and `MissShapley` attributed a classifier's hard class
  label instead of its predicted probability.** The coalition value function
  called `predict()`, which returns a label for a classifier. On an imbalanced
  problem every coalition then returns the majority class, every Shapley
  difference is exactly zero, and both `shap_values` and `miss_shap` silently
  return all-zero attributions. This was found on UCI SECOM, where the failure
  rate is 6.6% and every attribution collapsed to 0.0000. On a balanced problem
  the damage is subtler: attributions quantise to the class levels and describe
  label flips rather than any change in belief.

  The coalition value is now the predicted probability of the positive class for
  a binary classifier and the prediction itself for a regressor, selected
  automatically. A new `output` parameter accepts `'auto'` (default),
  `'proba'`, `'log-odds'` (the additive scale SHAP conventionally uses, better
  behaved near 0 and 1) and `'raw'` (the previous behaviour, retained only for
  comparison). `value_scale_` reports which scale was used, and the efficiency
  axiom now holds to machine precision on all of them.

  Regressors are unaffected, which is why this went unnoticed: every published
  figure and every regression example used `predict()` correctly.
- **`MissExplainer.shap_values` did not forward `output` to the Shapley
  engine**, so attributions were computed on the default scale regardless of how
  the explainer was configured, and did not reconcile with `expected_value_`.
- Multi-class models previously had no meaningful attribution at all, since
  there is no scalar value function. A new `class_index` parameter explains the
  probability of one named class; omitting it on a multi-class model now raises
  with instructions rather than silently attributing a label. `examples/03` has
  been updated to explain the high-quality class explicitly.
- `MissEnsemble.summary()` labelled its feature-importance table `X0..Xp` with
  no way to supply real names, so an importance ranking could not be read
  against a data dictionary. It now accepts `feature_names` and otherwise falls
  back to a `feature_names_in_` recorded at fit time, matching the resolution
  order `MissPreprocessor` and `MissSensitivity` already use. The label column
  is sized to the names.
- `plot_waterfall` was rewritten because the labels collided on any realistic
  number of features. The feature name and its value now sit on the axis tick
  and the contribution sits on the bar, the canvas height grows with the bar
  count, the x-axis carries a margin so the outermost label cannot be clipped,
  the legend sits below the axes rather than over the bars, and a feature that
  was not observed is marked as such instead of being shown with a value.
- **`MissSensitivity` silently returned an all-zero coefficient table** for
  any estimator without a `coef_` attribute, such as `MissBayesRegressor`,
  `MissSupport*`, `MissNeighbors*` and `MissGaussian*`. Every row read
  `0.0000` with a `STABLE` verdict, which is indistinguishable from a
  genuinely robust result. It now raises `AttributeError` naming the
  coefficient-bearing alternatives. Regression tests added.
- `MissSensitivity.summary()` sized its label column to six characters, so
  real feature names pushed the numeric columns out of alignment with their
  headers. The column is now sized to the data.

### Removed
- The NHIS income example has been withdrawn and archived under
  `DRAFTS/withdrawn_nhis_example/`. It was a survey rather than a scientific
  measurement, its target was an eleven-level ordinal income bracket with a
  fifth of the mass censored in the top category yet regressed as continuous,
  and its absolute scores could not be compared with the published study
  because that study's four-table feature pool cannot be reconstructed from the
  public files. Its three transferable findings are retained: that
  marginalisation and a missingness mask are complementary under MNAR, that
  tree-native methods such as MIA have no linear analogue and so cannot be
  compared across model classes, and the measured O(G p^3) cost scaling, now
  demonstrated on SECOM instead.

## [0.8.0]: 2026-07

First public beta. Every estimator handles `NaN` natively via
full-information maximum likelihood. Across the benchmarks each family
matches or beats listwise deletion, column deletion and mean imputation;
against well-tuned kNN and MICE the usual result is parity rather than a
win, which is what theory predicts when the working model is correct, and
on indicator-dominated data (see the thyroid example) FIML can trail.

### Models
- **Linear family**: `MissLinear`, `MissLogistic`, `MissRidge`/`MissRidgeRegressor`/`MissRidgeClassifier`, `MissLASSO`/`MissLASSORegressor`/`MissLASSOClassifier`: FIML (optionally penalized) regression and classification with standard errors, p-values, odds ratios, AIC/BIC, and prediction intervals that widen with missingness.
- **Distance / kernel**: `MissNeighbors`, `MissSupport`: k-NN and SVMs on the *expected* distance / kernel under a fitted joint Gaussian (PSD-safe, no imputation).
- **Generative**: `MissBayes`: closed-form full-covariance Gaussian models (QDA-style classifier, linear-Gaussian regressor); `structure='naive'` recovers classic Gaussian Naive Bayes.
- **Nonparametric**: `MissGaussian`: Gaussian-process regression and Laplace-approximation classification with a marginalised kernel and exact Bayesian intervals.
- **Grouped data**: `MissMixed`: random-intercept LME and logistic GLMM with BLUPs and intraclass correlation.
- **Meta-estimators**: `MissEnsemble` (bagging; accepts NaN-native tree learners such as HistGradientBoosting / XGBoost / LightGBM / CatBoost as members), `MissMulticlass` (one-vs-rest).

### Tools
- `MissImputer`: multiple imputation from the FIML joint MVN, with Rubin's-rules pooling and a one-call `fit_transform_combine`.
- `MissDiagnostic`: Little's MCAR test, MAR-plausibility checks, pattern summaries, missingness correlations, descriptive-by-missingness comparisons.
- `MissExplainer` / `MissShapley`: exact SHAP using the FIML model as the coalition value function, including *missingness SHAP* (the value of observing each feature).
- `MissSensitivity`: MNAR delta-adjustment stress-testing with tipping-point deltas.
- `MissPreprocessor` / `prefit_check`: validation and NaN-preserving categorical encoding.
- `MissKFold`, `MissStratifiedKFold`, `miss_cross_val_score`, `miss_cross_validate`: NaN-safe cross-validation.

### Engineering
- **Two-stage FIML** for the linear and mixed families: the joint-MVN nuisance moments are estimated once by a pattern-grouped EM, then a small regression block is optimised against a fully vectorised reduced likelihood with **exact analytic gradients**. Fits are 20 to 150× faster than a naive implementation (e.g. logistic-type fits ~0.2 s, LASSO regression ~1.3 s, mixed models ~2 s at n=600, p=8) with identical estimates.
- Penalized models standardise internally (glmnet convention) and convert all reported quantities back to the original scale, so `alpha` is unit-free.
- Pandas DataFrames/Series are accepted transparently (duck-typed; pandas is optional). String class labels and NaN-preserving categorical encoding are supported.
- Deterministic and reproducible: no randomness in the likelihood fits or quadrature.

### Requirements
- Python ≥ 3.9; NumPy ≥ 1.22, SciPy ≥ 1.8, scikit-learn ≥ 1.1.
- Optional: pandas (`[pandas]`), matplotlib (`[plots]`), XGBoost/LightGBM (`[trees]`).
