# Project status and known limitations

What is in place, and what is not. This is a factual note for anyone deciding
whether the library suits their work, not a schedule.

---

## In place

**Continuous integration.** Unit, cross-estimator conformance and
property-based suites run on Python 3.9 and 3.12 across Linux and Windows,
alongside a build job that installs the wheel into a clean environment and
exercises it there.

**The scikit-learn estimator contract.** Every estimator is checked with
`check_estimator`, discovered from the package rather than from a list, so a
new estimator is covered the day it is added. The few declared exceptions are
recorded with their reasons in `MissLearn/_sklearn_compat.py`, and
`CONFORMANCE.md` explains what the sweep covers.

**Documentation.** A Sphinx site with `numpydoc`, the guides rendered through
`myst-parser`, a generated API reference, and a thumbnail gallery of the ten
worked examples. Examples are not executed during a docs build, deliberately:
one of them takes hours.

**Coverage.** Measured on branch coverage rather than line coverage, currently
around 95 per cent over more than 2,600 tests. Branch coverage is the right
measure here because the defects this project has shipped lived in branches
that ordinary data does not reach.

**Contributor infrastructure.** `CONTRIBUTING.md` with a release process and a
deprecation policy following scikit-learn's convention, issue and pull-request
templates, and a Contributor Covenant 2.1 code of conduct.

**A reusable conformance checker.** `check_missing_data_estimator` is public,
drives any NaN-tolerant estimator through eleven degenerate regimes, and
depends on nothing in MissLearn. It distinguishes a clear refusal, which is
acceptable, from a silent `NaN`, which is not.

---

## Known limitations

These are stated because a user should know them before relying on the
library, not because a fix is scheduled.

**Type hints are partial.** `MissLearn/py.typed` ships, so type checkers will
trust the annotations they find. The annotations behind that marker are
incomplete, which promises more than it currently delivers.

**No performance regression tracking.** A benchmark harness exists, but
nothing tracks timings across commits, so a performance regression is found by
noticing rather than by measurement.

**The model recommender has a blind spot.** `MissRecommender` compares a
linear model against a nearest-neighbour probe, and neither can see structure
that an RBF kernel would find, so it can recommend a linear model where a
kernel method does better. It now reports when neither probe beats the trivial
baseline, rather than resolving that into a confident recommendation, which
narrows the consequence without removing the gap.

**Cost grows with the number of distinct missingness patterns.** The work is
O(G p³) in the number of distinct patterns G. Blockwise missingness keeps G
small; missingness scattered across many columns does not, and large G is
where the practical ceiling sits.

**The code of conduct has no reporting address.** It ships with the contact
marked `CONTACT_ADDRESS_TO_BE_SET`. A reporting channel that reaches nobody is
worse than an absent one, so it is left visibly unset rather than filled with
a guess.

---

## Where the property-based suite could go furthest

`tests/property_test_suite.py` runs a set of `hypothesis` properties over data
shape, missingness rate and pattern structure. The two families that have paid
best are scale invariance and degenerate columns, and both have more members
than are currently written. Anyone looking for a useful contribution will find
more defects there than anywhere else in the suite.
