# Roadmap: reaching and exceeding imbalanced-learn

What separates MissLearn from a library people adopt without asking questions.

The comparison is deliberate. `imbalanced-learn` is the closest analogue: a
single-purpose scikit-learn companion, in scikit-learn-contrib, widely trusted,
with a JMLR paper behind it. It is a reachable target rather than an
aspiration, and most of the distance is engineering rather than research.

---

## Where MissLearn already leads

Worth stating, because the gap list below is long and one-sided.

- **The science is further ahead.** Marginalisation over a joint working model
  across eight families, exact Shapley with the model as its own coalition
  value function, missingness SHAP, MNAR sensitivity analysis and
  evidence-based model triage have no counterpart in imbalanced-learn, which
  is a focused resampling toolkit.
- **The benchmarking discipline is stronger.** Four fairness controls, held
  model classes, matched preprocessing, matched regularisation and a matched
  noise floor, with negative results shipped rather than hidden.
- **The cross-estimator conformance suite is arguably better than what
  imbalanced-learn runs**, because it tests degenerate regimes rather than
  only the scikit-learn contract.

The deficit is not capability. It is the surrounding apparatus that lets a
stranger trust the capability without reading the source.

---

## Tier 1: prerequisites for a scikit-learn-contrib listing

Nothing below is optional for the stated goal.

**1.1 Make `check_estimator` pass. DONE.**
All nine estimators pass against scikit-learn 1.6.1, with two declared exceptions
recorded in `MissLearn/_sklearn_compat.py` and enforced as a hard CI gate. Both
are deliberate: MissLearn accepts NaN in `y`, which `check_supervised_y_no_nan`
forbids, and passing that check would mean deleting a central capability.
`SKLEARN_CONTRIB_ROADMAP.md` carries the detail.

**1.2 Hosted, versioned documentation. DONE.**
`docs/conf.py` builds a Sphinx site with `numpydoc`, the guides rendered
through `myst-parser`, and a generated API reference over all 39 exported
names. `.github/workflows/docs.yml` publishes it to GitHub Pages from `main`.
One manual step remains, once: set Settings, Pages, Source to "GitHub
Actions", without which the deploy returns 404. A version switcher is still
outstanding and is the remaining difference from imbalanced-learn's site.

**1.3 A rendered example gallery. DONE.**
`sphinx-gallery` is wired into `conf.py` and picks up the ten worked examples,
thumbnails only: `filename_pattern` is set so that no example executes during a
docs build, which matters because one of them takes hours. It publishes with
1.2.

**1.4 Coverage measurement. DONE.**
`pytest-cov` runs in CI on branch coverage rather than line, which is the
right choice here because every defect this project has shipped lived in a
branch that ordinary data does not reach. Currently **95%** over 2,658 tests
(1,600 unit, 1,042 conformance, 16 property). The paragraph this replaces asked
what "216 tests pass" was worth; the answer turned out to be eleven real
defects found in the branches those tests never entered.

**1.5 Contributor infrastructure. DONE.**
`CONTRIBUTING.md` exists, with a release process and a deprecation policy, as
do issue and pull-request templates, a CI workflow and a Contributor Covenant
2.1 code of conduct. The code of conduct ships with its reporting address
unset, marked `CONTACT_ADDRESS_TO_BE_SET`: a channel that reaches nobody is
worse than none, so it is left as a deliberate blocker rather than a guess.

---

## Tier 2: the quality signals that create trust

**2.1 Close the remaining conformance debt.**
One declared divergence remains, and it is a design decision rather than a
defect. Keep it at one. The suite fails CI on any undeclared divergence, so the
guarantee holds as long as CI runs.

**2.2 A deprecation policy. DONE.**
`CONTRIBUTING.md` documents it: a `FutureWarning` for two minor releases
before removal, following scikit-learn's convention.

**2.3 Type hints and `py.typed`. PARTLY DONE.**
`MissLearn/py.typed` ships, so the marker is in place. The hints behind it are
still partial, which is the weaker half of this item: a marker on incomplete
annotations promises more than it delivers.

**2.4 Performance regression tracking.**
A benchmark harness exists but nothing tracks timings over commits. The GP
kernel drift, where a misconfigured comparator went unnoticed for two days,
would have been caught by a tracked number.

**2.5 Property-based testing. DONE, and worth extending.**
`tests/property_test_suite.py` runs 16 `hypothesis` properties over data
shape, missingness rate and pattern structure. Sixteen is a start rather than
a finish: the properties that pay are the scale-invariance and
degenerate-column ones, and both families have more members than are currently
written.

---

## Tier 3: what would make MissLearn better than imbalanced-learn

**3.1 Publish the paper and get a DOI.**
imbalanced-learn's JMLR paper is much of why it is cited rather than merely
used. The arXiv paper exists; finishing it and obtaining a citable DOI converts
the research lead into a credibility lead.

**3.2 Ship the conformance suite as a reusable tool. DONE.**
`check_missing_data_estimator` is public, drives any NaN-tolerant estimator
through eleven degenerate regimes, and depends on nothing in MissLearn. It
distinguishes a clear refusal, which is acceptable, from a silent `NaN`, which
never is.

**3.3 Close the recommender's blind spot. STILL OPEN.**
`MissRecommender` compares a linear model against a nearest-neighbour probe,
which cannot see structure an RBF kernel finds; it recommends `MissRidge` on
the thyroid data where `MissSupport` wins by 0.037 AUC. A cheap kernel probe
is still the fix. What has changed is only that the recommender now says when
it cannot tell: neither probe beating the trivial baseline is reported as
such, instead of being resolved into a confident recommendation. That narrows
the damage without closing the gap.

**3.4 Interoperate rather than compete.**
`MissEnsemble` already accepts NaN-native learners as members. A documented
recipe for combining MissLearn with imbalanced-learn samplers, for the common
case of data that is both incomplete and imbalanced, serves users neither
library serves alone.

**3.5 Scale beyond the current ceiling.**
Cost is O(G p^3) in the number of distinct missingness patterns. Blockwise data
keeps G small; scattered data does not. A documented approximation for large G,
with its error characterised, would lift the practical limit that currently
sends users elsewhere.

---

## Suggested ordering

Tier 1 in order: 1.1, then 1.2 and 1.3 together, then 1.4 and 1.5. The
conformance work gates a listing; documentation gates adoption; coverage and
contributor infrastructure gate contributions.

Tier 2 can proceed in parallel and is mostly independent.

Tier 3 should wait. Each item is more valuable once the library is trusted, and
3.1 in particular is worth more after 1.2 exists to point at.

---

## What this list is not

It is not a research programme. Every Tier 1 and Tier 2 item is engineering
with a known shape and a verifiable endpoint. The two-to-three-month estimate
covers Tier 1 and most of Tier 2 at a sustained pace.

Nor does any of it improve the statistics. The library's estimates are already
sound; what is missing is the apparatus that lets someone else believe that
without reading `_linear.py`.
