# Cross-estimator conformance

Why this library tests every estimator against the same checks rather than
each one on its own, and what that does and does not guarantee.

---

## What prompted it

A user hit a crash within twenty minutes of first contact.
`MissLASSOClassifier` raised on high missingness while `MissLogistic` and
`MissLASSORegressor` degraded gracefully. The complete-case seeding fallback
existed in both siblings and had simply never been applied to that one class.

The useful diagnostic is not "there was a bug". It is that **behaviour
diverged between classes meant to be interchangeable**, and nothing in the
project could have detected it, because every test targeted one class at a
time. Per-class discipline does not scale, and it did not hold.

The same shape has recurred since, which is why it is written down rather than
fixed and forgotten:

- A guard placed in a subclass body that has no `fit` of its own never runs.
  A check belongs where the parameter is *consumed*, not where it is declared.
- Two of eight estimators refused an invalid penalty, so the family looked
  protected. They refused only because they seed from a scikit-learn estimator
  whose own validation caught it. The other six accepted it.
- A shared label encoder recognised only one of the four ways a value can be
  absent, so every classifier rejected input the documentation says is
  supported. It survived because the one class whose tests covered that input
  was the one class excluded from the shared path.

Each was invisible to per-class testing and obvious to a sweep.

---

## What is checked

**The scikit-learn contract.** `check_estimator` runs over every estimator,
discovered from the package rather than from a list, so a class is covered the
day it is added. This matters: an earlier hand-maintained list named nine of
twenty-three, and the fourteen omitted were not passing quietly, they were
simply not being asked.

The declared exceptions live in `MissLearn/_sklearn_compat.py`, each with a
reason a reviewer can weigh. Two apply to the whole library and cannot be
resolved without making it worse. `check_supervised_y_no_nan` requires an
estimator to reject `NaN` in `y`; this library accepts it deliberately,
because a row with an unobserved response still informs the feature
distribution, so passing that check would mean deleting a central capability.
One further exception is an upstream limitation that a plain scikit-learn
`SVC(probability=True)` fails identically.

`CONTRACT_EXEMPT` in the same module is a stronger statement, used once: it
removes a class from the sweep entirely rather than excusing individual
checks. `MissImputer.transform` returns m completed datasets rather than one
array, which is what multiple imputation is, so it cannot meet the transformer
contract and should not claim to.

**Degenerate regimes.** The conformance suite drives every estimator through
regimes the scikit-learn contract says nothing about: no complete cases, an
entirely absent column, a single observed cell, an absent target, blockwise
missingness, a design wider than it is tall, one feature, extreme scales. Any
undeclared divergence between siblings fails the build.

**Determinism.** Row order must not change predictions. This axis was added
after the suite was found to grade whether a fit produced something
defensible, never whether it produced the same thing twice. It found a real
defect on its first run.

Comparisons on that axis use the continuous surface, `decision_function` or
`predict_proba`, never the labels. A label moves only when a score crosses a
boundary, so comparing labels hides drift, and it did: one classifier drifted
through its solver's random number generator for weeks while a label-based
determinism test stayed green.

---

## What it does not cover

The suite defends against divergence between siblings. It does **not** test
statistical correctness: an estimator can be perfectly conformant and still
compute the wrong likelihood. The unit suite covers that, and the benchmark
harness covers whether the numbers are competitive.

Nor does it substitute for numerical work. A declared divergence in a
degenerate regime is a real defect in that estimator, and closing it is
engineering on the estimator, not on the tests.

Finally, conformance says nothing about whether a model is appropriate for
your data. That is what the diagnostics are for.
