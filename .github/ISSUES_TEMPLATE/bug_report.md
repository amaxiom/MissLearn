---
name: Bug report
about: Something behaves incorrectly
labels: bug
---

## What happened

## What you expected

## Minimal reproduction

```python
# please make this runnable as-is
```

## Do the sibling estimators agree?

The most useful single piece of information in a MissLearn bug report.
If `MissLASSOClassifier` misbehaves, try `MissLogistic` and
`MissLASSORegressor` on the same input.

This is not busywork: the bug that prompted the whole conformance effort was
found exactly this way. A guard existed in two siblings and had never been
applied to the third, and "these classes disagree" pointed straight at the
fix in a way that "this crashed" would not have.

- [ ] I tried at least one sibling estimator
- Sibling tried, and what it did:

## Versions

```
MissLearn:
scikit-learn:
numpy:
Python:
OS:
```
