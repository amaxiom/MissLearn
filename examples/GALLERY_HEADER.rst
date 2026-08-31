.. _examples_gallery:

Worked examples
===============

Ten end-to-end analyses on **real data sets**, each downloaded and cached on
first run so subsequent runs are offline.

Two of them (07 and 10) are head-to-head comparisons against published
missing-data studies, on those studies' own data and protocols. Example 02 is a
case where the obvious MissLearn model loses, and it is included for exactly
that reason: it then diagnoses why, and the answer turns out to be the model
class rather than the missing-data handling.

Every example is also a benchmark
---------------------------------

Each one compares its model family against the standard alternatives,
**dropping** and **imputing**, across the same six arms: drop rows, drop
columns, mean, kNN, MICE, and full information.

Two rules make those comparisons mean anything, and they are enforced
throughout:

1. **The model class is held fixed within a comparison.** A difference between
   arms is evidence about the missing-data strategy only if nothing else
   differs. Comparing a likelihood model against a boosted tree would measure
   capacity instead, so no tree model appears in any comparison here.

2. **Preprocessing is matched.** The MissLearn estimators standardise
   internally, so every conventional arm is given the same standardisation
   after its deletion or imputation step. Without that the baselines lose on
   preprocessing rather than on strategy.

Where an example varies something other than the missing-data treatment, the
comparisons are reported as separate blocks and never merged into one ranking.

Runtimes vary from seconds to about half an hour. Several scripts accept
``--quick`` for a fast pass.
