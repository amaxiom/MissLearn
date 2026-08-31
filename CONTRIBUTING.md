# Contributing to MissLearn

Participation in this project is governed by the
[Code of Conduct](https://github.com/amaxiom/MissLearn/blob/main/CODE_OF_CONDUCT.md).

Thank you for considering it. This document says what the project expects, so
that a contribution is not rejected for a reason nobody wrote down.

---

## The one rule that is specific to this project

**Behaviour shared by several estimators is defined once, in
`MissLearn/_conformance.py`, and applied at a single wrapping point.**

This is not a style preference. MissLearn has sixteen estimators that are meant
to be interchangeable, and it once had sixteen independent implementations of
behaviour that should have been identical. A user found the consequence within
twenty minutes of first contact: `MissLASSOClassifier` raised on high
missingness while `MissLogistic` and `MissLASSORegressor` degraded gracefully,
because a fallback present in both siblings had never been applied to that one
class.

A cross-estimator conformance suite now holds the guarantee, but the suite only
catches drift after it happens. Not writing the sixteenth copy is what prevents
it.

So: if you find yourself adding a guard to one estimator, stop and ask whether
its siblings need it too. If they do, it belongs in `_conformance.py`.

---

## Getting set up

```bash
git clone https://github.com/amaxiom/MissLearn
cd MissLearn
pip install -e ".[all,test]"
```

Run the suites:

```bash
pytest tests/unit_test_suite.py -q
pytest tests/conformance_test_suite.py -q -ra
```

The unit suite takes about a minute. The conformance suite drives sixteen
estimators through fourteen degenerate regimes and takes considerably longer;
`-ra` prints the declared divergences, which are the interesting part.

---

## Before opening a pull request

**Both suites must pass.** The conformance suite reports declared divergences
as `xfail`. If one of yours reports `XPASS`, you have fixed it: delete its
entry from `KNOWN_FAILURES` in the same pull request, so the guarantee tightens
rather than drifting.

**New estimators must join the conformance suite.** Add the class to
`REGRESSORS` or `CLASSIFIERS` in `tests/conformance_test_suite.py`. An
estimator outside the suite is outside the guarantee.

**A new degenerate regime is welcome.** If you find data that breaks something,
add it to `REGIMES` rather than only fixing the symptom. That is how the regime
list grew to fourteen.

**Do not relax a test to make it pass.** If an estimator genuinely cannot
handle a regime, declare it in `KNOWN_FAILURES` with a reason explaining why
its siblings can and it cannot. A reason a reader can evaluate is the point;
"known issue" is not one.

---

## Documentation

Numbers in documentation must come from a run, not from memory. Several errors
in this project's history were transcribed figures that drifted from the code
that produced them: a test count, a set of dependency minimums, an estimator's
sample-size ceiling.

If you quote a result, say where it came from. If you change behaviour that a
guide describes, update the guide in the same pull request.

**Build the docs before you push.** CI builds them with `-W`, which turns
every warning into an error, so a broken cross-reference or a malformed
docstring fails the run rather than reaching a reader. The same flags are
available locally:

```bash
cd docs
make strict          # or make.bat strict on Windows
```

`make html` tolerates warnings that CI will reject, so `strict` is the one
that tells you what CI will say. `.github/workflows/docs.yml` publishes the
built site to GitHub Pages from `main`.

House style, for consistency with what is already written:

- No em dashes or en dashes. Use a comma, a colon, a semicolon or a full stop.
- Explain why a choice was made, not only what it does.
- Report negative results plainly. Several examples in this repository exist
  precisely because MissLearn loses on that data, and they say so.

---

## Reporting a bug

The most useful report gives a minimal reproduction and, where relevant,
**says what a sibling estimator does with the same input**. That comparison is
what turns "this crashed" into "these two classes disagree", which is a much
stronger signal and usually points straight at the fix.

Please include:

- MissLearn, scikit-learn, numpy and Python versions
- A short script that reproduces it
- What you expected, and what happened

---

## Releases

The package builds from `MissLearn/` directly via the root `pyproject.toml`,
with the version read from `MissLearn.__version__`.

```bash
python -m build
python -m twine check dist/*
```

There is deliberately **no second copy of the package** for packaging. One
existed and drifted: three library fixes made in a single session reached only
the source tree and would have shipped missing. If you find yourself creating a
staging copy, that is the mistake being repeated.

Before publishing, install the built wheel into a clean environment and
exercise it there. Building is not evidence that the artefact works; CI does
this on every run and a release should not skip it.

---

## Deprecation policy

Public API is what `MissLearn.__all__` exports. Changing it follows
scikit-learn's convention: a `FutureWarning` naming the replacement, kept for
two minor releases before removal. Anything prefixed with an underscore is
internal and may change without notice.
