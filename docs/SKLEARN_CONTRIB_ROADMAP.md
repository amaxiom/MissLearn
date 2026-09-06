# scikit-learn-contrib readiness

A working note, not a plan of record. It states what was measured, what is
missing, and in what order the gaps are worth closing.

The reason for targeting a listing is credibility rather than distribution.
`imbalanced-learn` is trusted partly because it sits under
scikit-learn-contrib and is therefore known to meet a published standard. A
self-published package makes the same claims with nothing behind them.

> Confirm the current submission process before acting on the ordering below.
> Requirements change, and this note reflects what the existing contrib
> projects visibly satisfy rather than a checklist read on the day.

---

## What prompted this

A user hit a crash within twenty minutes of first contact.
`MissLASSOClassifier` raised on high missingness while `MissLogistic` and
`MissLASSORegressor` degraded gracefully. The complete-case seeding fallback
existed in both siblings and had simply not been applied to that class.

The useful diagnostic is not "there was a bug". It is **behaviour diverged
between classes that are meant to be interchangeable**, and nothing in the
project could have detected that, because every test targeted one class at a
time. scikit-learn and imbalanced-learn do not have this failure mode, because
they run one set of checks over every estimator. The guarantee here was
per-class discipline, which does not scale and did not hold.

---

## Measured state

Against scikit-learn 1.6.1 on 2026-08-01.

### `check_estimator`: passing

**9 of 9 estimators pass**, with two declared exceptions recorded in
`MissLearn/_sklearn_compat.py` and enforced as a hard CI gate.

The two exceptions cannot be removed without making the library worse, which
is why they are declared rather than worked around:

`check_supervised_y_no_nan`
    Requires raising on NaN in `y`. MissLearn accepts it deliberately, because
    a row with an unobserved response still informs the feature distribution.
    Infinity in `y` *is* rejected. Passing this check would mean deleting a
    central capability.

`check_methods_subset_invariance`
    `decision_function` is `log(p / (1 - p))`. Where a model is certain, `p`
    sits within 1e-8 of 0 or 1 and the logarithm turns a difference far below
    any meaningful tolerance into a large one in log-odds. `predict_proba`
    itself is exactly subset-invariant, verified at 0.000e+00. Meeting the
    check would require clipping so coarsely that genuine confidence became
    unrepresentable.

Counted individually rather than by estimator, the starting position was 100
failures of 471 checks across 15 distinct checks. Every one was fixed in
`_conformance.py` at a single application point; none was family-specific.

### The first pass, for the record

Six conformance defects found and fixed, none of which was family-specific.
Every one lived in behaviour that should have been identical across estimators
and was instead absent everywhere, which is the same structural fault as the
reported bug rather than a different one.

| # | Defect | Fix |
|---|---|---|
| 1 | Binary classifiers rejected K > 2 | route internally through `MissMulticlass` |
| 2 | `fit` signature erased by the pandas wrapper | `functools.wraps`, so `signature(fit)` reads `(self, X, y)` |
| 3 | `predict` accepted a mismatched feature count | `check_n_features`, scikit-learn's own wording |
| 4 | `allow_nan` tag reported `False` | declared `True` for both tag APIs |
| 5 | Sparse support implied but absent | `sparse = False` declared |
| 6 | Complex input silently cast to float | `check_no_complex_data`, refused with the expected phrase |

All six went into `MissLearn/_conformance.py` and are applied at a single
wrapping point, not copied into sixteen classes.

**At that point `check_estimator` still did not pass.** It was a long tail, and
each fix revealed the next requirement. It passes now, as the measured state
above records; this section is kept because the shape of the tail is the useful
part, and because the estimate it supported turned out to be right. Read every
verb in this section as past tense.

The original two blockers were:

**1. The `fit` signature is erased.**

```
MissLinear: Expected y or Y as second argument for method fit of MissLinear.
            Got arguments: ['X', 'args', 'kwargs'].
```

The pandas compatibility layer wraps `fit` and replaces its signature with
`(X, *args, **kwargs)`. scikit-learn introspects signatures for metadata
routing and for its own checks, so the wrapper makes every estimator look
malformed. The fix is to preserve the wrapped signature with
`functools.wraps` together with an explicit `__signature__`, which is a change
to one layer rather than to sixteen classes.

**2. Multiclass on binary-only classifiers.**

```
MissLogistic: requires exactly 2 classes; found [0 1 2]
```

`check_estimator` exercises multiclass on every classifier. The MissLearn
binary classifiers are binary by design and multiclass is provided separately
through `MissMulticlass`.

There are two honest resolutions and they are not equivalent:

- Declare the `binary_only` estimator tag. Cheap, truthful, and leaves the
  user to reach for `MissMulticlass` themselves.
- Route multiclass internally, so `MissLogistic().fit(X, y_three_class)`
  works. More work, and it makes the estimators behave the way a scikit-learn
  user expects without reading anything.

This is an API decision rather than a testing one and should be taken
deliberately.

### Estimator tags

`allow_nan` currently reports `False` on estimators whose entire purpose is
accepting `NaN`. Nothing in the common paths breaks, because `Pipeline`,
`GridSearchCV` and `cross_val_score` all work, but any third-party utility
that consults the tag will refuse input the estimator could have handled. This
is a one-line correctness fix and should not wait.

### Infrastructure

Absent rather than deficient, as of the start of this work:

| Item | State |
|---|---|
| `.github/workflows` | added in this pass |
| `tests/conformance_test_suite.py` | added in this pass |
| `pyproject.toml` at the root | present, builds from `MissLearn/` directly |
| Duplicated package tree | removed in this pass, archived |
| `tox.ini` or `noxfile.py` | still missing |
| `conftest.py`, `pytest.ini` | still missing |
| `.pre-commit-config.yaml` | still missing |
| Published documentation site | still missing; guides are markdown in `docs/` |

---

## Cross-estimator conformance: what the matrix found

All 16 fittable estimators driven through 14 degenerate regimes.
**13 divergences, in six classes.**

| Regime | Diverging estimator | Nature |
|---|---|---|
| `missing_90pct` | `MissLinear` | `LinAlgError: not positive definite`; seven sibling regressors cope |
| `wide_p_gt_n` | `MissLinear` | `LinAlgError` at p > n; every sibling copes |
| `single_feature` | `MissLASSORegressor` | `LinAlgError` at p = 1, while `MissLASSOClassifier` copes |
| `all_nan_column` | `MissLinear`, `MissRidgeRegressor`, `MissLASSORegressor`, `MissMixedRegressor` | fit and predict succeed, predictions are `NaN` |
| `all_nan_column` | `MissSupportRegressor`, `MissSupportClassifier` | `ValueError: Input X contains NaN` where siblings cope |
| `all_nan_column` | `MissLogistic`, `MissRidgeClassifier`, `MissLASSOClassifier`, `MissMixedClassifier` | `predict` returns finite labels, `predict_proba` is entirely `NaN` |

Three observations matter more than the count.

`MissLASSORegressor` fails on a single feature where its own classifier
copes. That is the mirror image of the reported bug, in the same family, and
it was found by machine in one run.

Four regressors **fit, predict, and return `NaN`**. That is worse than
raising: an exception stops a pipeline, whereas silent `NaN` propagates into
whatever consumes the prediction.

**The last row was missed on the first pass, and the reason is instructive.**
The initial sweep exercised `predict` only, and all four of those classifiers
return perfectly finite labels, so they were recorded as healthy. Their
`predict_proba` is entirely `NaN`. A caller using `predict` sees nothing
wrong at all, which makes this the most dangerous variant found: the label
path masks a probability path that has no defined answer. Three sibling
classifiers, `MissBayes`, `MissNeighbors` and `MissGaussian`, produce valid
probabilities under the same regime.

The general lesson is that a conformance check is only as wide as the API
surface it touches. The suite now exercises `predict_proba` in the guarantee
test as well as in its own test, because checking one method and inferring the
health of another is precisely the reasoning that let the original bug through.

Each divergence is recorded in `KNOWN_FAILURES` in the suite with a reason, so
it reports as an expected failure rather than blocking CI. Every entry is a
work item. When one is fixed the test reports `XPASS` and the entry should be
deleted.

---

## Ordering

The sequence matters, because some of these make the others cheaper.

**First, correctness of the contract.** Fix the `fit` signature wrapper and
the `allow_nan` tag. Both are small, both are unambiguous, and the signature
fix is a precondition for `check_estimator` telling you anything useful about
the rest.

**Second, decide the multiclass question.** It changes what conformance means
for eight classes, so making it late means redoing work.

**Third, close the thirteen declared divergences.** They cluster hard, so this
is far less work than thirteen separate fixes. Ten of the thirteen are the
single `all_nan_column` regime, and eight of those ten are one shared path
through the likelihood models surfacing twice, as `NaN` predictions in the
regressors and as `NaN` probabilities in the classifiers. Two more are
`MissLinear` needing the conditioning guard its siblings already apply.

A column with no observed values has no conditional distribution to
marginalise over, so the honest fix is probably to detect it at fit time and
either drop it with a warning, as `MissRecommender.preprocessing_` already
recommends for high-missingness columns, or raise a clear error naming the
column. What must not continue is fitting successfully and returning `NaN`.

**Fourth, the remaining infrastructure.** `tox` or `nox` for the interpreter
matrix locally, `pre-commit` for style, and a published documentation build.

**Fifth, submit.** With `check_estimator` passing, a conformance suite in CI
across two interpreters and two platforms, and documentation building, the
listing request rests on evidence.

---

## What this does not fix

The conformance suite defends against *divergence between siblings*. It does
not test statistical correctness: an estimator can be perfectly conformant and
still compute the wrong likelihood. The unit suite covers that, and the
benchmark harness covers whether the numbers are competitive.

Nor does conformance testing substitute for the numerical work. The nine
declared failures are real defects in degenerate regimes, and closing them is
engineering on the estimators themselves, not on the tests.
