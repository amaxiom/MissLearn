# MissLearn Example Notebooks

Ten worked, end-to-end examples on **real datasets**. Each notebook downloads its dataset on first run and caches it in [`example_data/`](example_data/); subsequent runs are fully offline.

Two of them (07 and 10) are **head-to-head comparisons against published missing-data studies**, on those studies' own data and protocols. Example 02 is a case where the obvious MissLearn model loses, and it is included for exactly that reason: it then diagnoses *why*, and the answer turns out to be the model class rather than the missing-data handling.

> Synthetic data lives in [`../benchmarks/`](../benchmarks/), where the ground truth is known and the missingness is injected under a controlled mechanism. This directory is the real-data counterpart.

## Every example is also a mini benchmark

Each notebook compares its model family against the standard alternatives, **dropping** and **imputing**, across the same six arms:

**Drop rows | Drop cols | Mean | kNN | MICE | FIML**

Two rules make those comparisons mean something, and they are enforced everywhere in this directory:

1. **The model class is held fixed within a comparison.** A difference between arms is evidence about the missing-data strategy only if nothing else differs. Comparing a likelihood model against a boosted tree would measure model capacity instead, so no tree model appears in any comparison here.
2. **Preprocessing is matched.** The MissLearn estimators standardise internally, so every conventional arm is given the same standardisation after its deletion or imputation step. Without this the baselines lose on preprocessing rather than on strategy.

Where a notebook does vary something other than the missing-data treatment (example 01 adds a random intercept), the two comparisons are reported as separate blocks and are never merged into one ranking.

## How to run

With the Anaconda environment at `C:\ProgramData\anaconda3`:

```powershell
C:\ProgramData\anaconda3\Scripts\jupyter.exe notebook
```

then open a notebook and run all cells, or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute 01_MissMixed_MissImputer_Parkinsons.ipynb
```

The notebooks locate the repository root by walking up the directory tree, so no package installation is required.

Every example ships **both** a notebook and a script. For 01 to 04 the script is
generated from the notebook's code cells, so the two cannot drift apart; for 05
to 10 the script is hand-maintained and independent, and carries its own
`--quick` flag. Either form runs headless:

```bash
python 09_secom_blockwise.py --quick
```

---

## 01_MissMixed_MissImputer_Parkinsons.ipynb: Mixed-effects FIML and multiple imputation

- **Dataset:** Parkinson's Telemonitoring (UCI, Little et al. 2009); 5,875 voice recordings from 42 patients over six months; cached as `example_data/parkinsons_updrs.data`.
- **Task:** Regression; predict the total UPDRS clinical severity score.
- **Missingness:** injected, 20% MAR into the 16 voice features (missingness of feature *j* depends on the observed adjacent feature), simulating failed or rejected remote recording sessions.
- **What it demonstrates:** Repeated-measures data where observations within a patient are correlated. `MissMixedRegressor` adds a patient-level random intercept b_i ~ N(0, tau^2) that absorbs between-patient UPDRS variation. Missing voice features are marginalised analytically; no records dropped, no fake values. `MissImputer` then draws *m* complete datasets from the FIML-estimated joint MVN (conditional variance plus, with `posterior=True`, parameter uncertainty) so NaN-intolerant downstream learners can be used, with predictions combined under Rubin's rules. Cross-validation is grouped by patient, because a random split would leak patient identity through the repeated measures.
- **The number the notebook exists to produce:** the fitted intraclass correlation is **0.935** (tau 9.74 against a residual sigma of 2.57). Almost all of the variation in this data is *between* patients rather than within them, and that single fact governs everything else below. Predicting for a **monitored** patient, whose random intercept is known, gives RMSE **2.602**. Predicting for a **new** patient gives **10.433**, against a response standard deviation of 10.70. Reporting the first number for a task that is really the second overstates accuracy by **7.83 RMSE**, and a shuffled split does exactly that: every one of the 42 patients contributes 101 to 168 recordings, so a random split puts each held-out patient in the training folds too.
- **Block A, missing-data strategy with the flat linear class held fixed** (patient-grouped folds, 20% MAR injected, 19.8% realised): drop columns 10.390, mean imputation 10.581, kNN 10.690, FIML 10.761, MICE 10.791. FIML sits 0.371 RMSE behind the best conventional arm, so **it does not win here**, and every arm has an R² of roughly zero or below: under grouped folds nothing beats the training mean. The voice features carry ample information for tracking a patient already under monitoring and essentially none that transfers across people.
- **Listwise deletion diverges outright**, to RMSE 970.2 and R² of about -10,300. With 20% MAR spread over sixteen features, complete cases are rare enough that the surviving design is near-singular and the fit extrapolates wildly. It is reported rather than quietly dropped, because "drop the incomplete rows" is still the most common default in practice and this is what it can do.
- **APIs exercised:** `MissMixedRegressor` (`groups`, BLUPs, `summary()`), `MissLinear`, `MissImputer` (`fit_transform_combine`), cross-validation utilities.
- **Runtime:** roughly 10 to 20 minutes; the mixed-effects CV dominates.

## 02_MissEnsemble_Thyroid.ipynb: Homogeneous and heterogeneous FIML ensembles

- **Dataset:** Thyroid Disease "sick" task (UCI / Garavan Institute, 1987); 3,772 clinical records, 21 features: 15 binary indicators and 6 continuous measurements, patient age plus the five assays TSH, T3, TT4, T4U and FTI. A sixth assay, TBG, is dropped because it is absent for every record. Cached as `example_data/sick.data` / `sick.test`.
- **Task:** Binary classification; sick vs. euthyroid.
- **Missingness:** **real and native**, and plausibly MNAR: a test is only ordered when a clinician suspects a specific dysfunction, so the pattern itself is informative and naive imputation is biased.
- **What it demonstrates:** how to read a negative result rather than report it and stop. The notebook runs in three stages: the six-arm strategy comparison with the logistic class held fixed, then a model-class investigation that explains why that result came out as it did, then `MissEnsemble` built on whichever family the investigation selected. The ordering is the point. Building the ensembles first, on the family the notebook started with, would have meant bagging a model this data rejects.
- **Stage 1, the negative result:** with the logistic class held fixed, FIML **loses** (0.9344 against 0.9522 for MICE). The obvious reading is that the task is dominated by binary indicator flags, the regime where a joint-Gaussian working model is weakest.
- **Stage 2, the diagnosis:** that reading is incomplete. `MissLogistic` also assumes the log-odds are *linear*, which has nothing to do with missing data, and here that is the binding constraint. Across four families on identical folds: kernel 0.9713, linear 0.9344, nearest-neighbour 0.9330, generative 0.8394. Only the kernel moves. Tuning `C` by inner cross-validation **on both arms** then puts the single kernel model at **0.9751 against 0.9731** for MICE + SVC, winning all five folds (paired t-test p = 0.048; paired difference sd 0.0016 against an unpaired spread of 0.005). The ranking only flips once both sides are tuned, because a marginalising likelihood and an imputation pipeline have different effective sample sizes and do not want the same regularisation.
- **Stage 3, the best model:** `MissEnsemble` of ten `MissSupportClassifier(kernel='rbf')` members, at **0.9796 AUC and 0.0250 Brier**, beating the single tuned model on every fold. The heterogeneous ensemble **loses** (0.9692), and the member out-of-bag scores show why: the two kernel members sit at 0.9664 and 0.9628 and the two linear members at 0.9529 and 0.9522, so averaging drags the strong members down. Diversity pays only when the members are individually competent.
- **Where it does not win:** against a *matched* bagged baseline (MICE + `BaggingClassifier(SVC)`, same kernel, same C, same member count, same bootstrap) the FIML ensemble is nominally ahead at 0.9796 against 0.9782 but wins only two folds of five. The single-model advantage does not survive bagging: at roughly 3% missing cells, bagging and marginalising are largely buying the same variance reduction. Brier scores stay level or slightly favour the imputed arm throughout, so the advantage on this data is in ranking rather than calibration.
- Column deletion collapses to 0.6445 against roughly 0.95 for every other logistic arm, because the columns that go missing are the assays carrying the signal. The example also shows `MissRecommender` getting this data wrong: its linearity probe compares a linear model against a nearest-neighbour model, those land within 2% here, so it recommends `MissRidgeClassifier`. Nearest neighbours is a poor proxy for what a kernel finds.
- **APIs exercised:** `MissEnsemble` (both modes, `weights`, OOB scores, `summary(feature_names=...)`), `MissSupportClassifier`, `MissNeighborsClassifier`, `MissLogistic`, `MissRidgeClassifier`, `MissLASSOClassifier`, `MissBayesClassifier`. Also shows `MissEnsemble` validating members at construction: a plain `LogisticRegression` raises a descriptive `ValueError`.
- **Runtime:** roughly 20 to 30 minutes; the tuned comparison and the kernel ensembles dominate, since every kernel fit is far more expensive than a logistic one.

## 03_MissLearn_Pipeline_Wine.ipynb: Full pipeline, diagnose to explain

- **Dataset:** Wine Quality, white wines (UCI, Cortez et al. 2009); 4,898 wines, 11 physicochemical features; cached as `example_data/winequality-white.csv`.
- **Task:** 3-class classification; low / medium / high quality, collapsed from the 0 to 10 sommelier rating.
- **Missingness:** injected, 15% MAR.
- **What it demonstrates:** The complete practitioner workflow. **Diagnose** with `MissDiagnostic` (Little's MCAR test, MAR plausibility regressions, pattern summary, missingness correlation heatmap). **Validate** with `prefit_check` / `MissPreprocessor` (constant features, infinities, near-multicollinearity, categorical encoding). **Fit** `MissMulticlass(MissLogistic)`, a One-vs-Rest FIML multiclass classifier with row-normalised probabilities. **Explain** with `MissExplainer`, which uses the FIML model as its own exact coalition value function (unselected features set to NaN, no background dataset); with p=11, exact 2^p Shapley enumeration is used. Produces both **value SHAP** and **missingness SHAP**.
- **Diagnostics that are acted on, not printed and ignored:** `MissPreprocessor` is given `feature_names`, so its report names the measurement rather than `X4`. Its kurtosis notes (chlorides has excess kurtosis 41.8) drive an actual copula comparison on shared folds, and **whichever way that comparison falls decides the model used for the rest of the notebook**. On this data the two are within noise, so the simpler model is kept and the notebook says so; heavy margins do not always mean the joint-normal working model is what limits the fit. The six-arm strategy benchmark deliberately stays on the plain model, because giving only the FIML arm an extra marginal transform would measure the transform rather than the missing-data treatment.
- **Explaining a multi-class model:** `MissExplainer` is given `class_index`, because a three-class model has no single scalar value function. Attributing the predicted label instead would describe changes in the argmax rather than changes in belief.
- **APIs exercised:** `MissDiagnostic`, `prefit_check`, `MissPreprocessor` (`feature_names`), `MissMulticlass`, `MissLogistic` (`copula='auto'`), `MissExplainer` (`class_index`).
- **Runtime:** roughly 15 to 25 minutes; multiclass FIML fits plus exact Shapley enumeration.

## 04_Strategies_Pima_Diabetes.ipynb: Strategy comparison on clinical data

- **Dataset:** Pima Indians Diabetes (UCI / NIDDK); 768 records, 8 clinical measurements; cached as `example_data/pima_diabetes.csv`.
- **Task:** Binary classification; diabetes onset within five years.
- **Missingness:** **real and native**, disguised as zeros. A recorded serum insulin, triceps skinfold, blood pressure, BMI or glucose of exactly 0 is physiologically impossible and encodes a measurement that was never taken. Insulin is roughly 49% absent and skinfold roughly 30%.
- **What it demonstrates:** The six-arm strategy comparison across several model families, each against its own matched counterpart. A good illustration that recognising sentinel-coded missingness is itself part of the analysis: treating those zeros as real values biases every downstream estimate.
- **Runtime:** a few minutes.

## 05_Strategies_Credit_Approval.ipynb: Strategy comparison with mixed types

- **Dataset:** Credit Approval (UCI, Quinlan); 690 applications, 15 anonymised attributes, a mixture of continuous and categorical; cached as `example_data/crx.data`.
- **Task:** Binary classification; application approved or not.
- **Missingness:** **real and native**, marked `?` in the source file, spread across both continuous and categorical attributes.
- **What it demonstrates:** The six-arm comparison on genuinely mixed-type data, exercising `MissPreprocessor`'s one-hot encoding with NaN preservation. Shows what changes when the missingness is spread thinly across many columns rather than concentrated in a few.
- **Runtime:** a few minutes.

## 06_Guided_Workflow_Air_Quality.ipynb: Diagnose, recommend, fit, explain, stress-test

- **Dataset:** Air Quality (UCI, De Vito et al. 2008); 9,357 hourly records from a five-sensor metal-oxide device deployed at road level for one year, alongside a co-located reference analyser bank; cached as `example_data/air_quality.csv`.
- **Task:** Regression; reference-grade NO2 from the low-cost sensor array plus meteorology. This is the field-calibration problem in low-cost air-quality monitoring: metal-oxide sensors are cross-sensitive and drift, so the mapping to a reference analyser has to be learned.
- **Missingness:** **real and native**, with **three distinct mechanisms in one file**. The `-200` sentinel resolves to 3.9% on every channel at once (whole-station outages), roughly 17.5% on the reference channels (analyser servicing), and 90.2% on `NMHC(GT)`, whose analyser was decommissioned early in the campaign so that the column is missing by *time* rather than at random.
- **What it demonstrates:** The full guided workflow, driven by the data rather than by the analyst's taste. **`MissDiagnostic`** establishes the mechanism. **`MissRecommender`** then ranks the model families with reasons attached, flags `NMHC(GT)` for removal rather than imputation, and names the required follow-up analyses. Acting on that recommendation takes the complete-case count from 827 to 6,941. The six-arm benchmark then decides among the shortlist under **blocked temporal cross-validation**, because the records are hourly and a shuffled split would leak autocorrelation. **`MissExplainer`** separates the value of a reading from the value of having measured the channel at all. **`MissSensitivity`** sweeps MNAR departures and reports tipping points.
- **Two channels are deliberately excluded** from the predictors and the notebook says why: `NOx(GT)` leaks the target because NO2 is a component of NOx by definition, and `C6H6(GT)` correlates 0.982 with `PT08.S2(NMHC)` because benzene in this release was derived from that sensor.
- **Honest result:** column deletion is **undefined** here (every channel has at least one hole). FIML reaches **parity** with the best imputation for the linear and ridge families, beating listwise deletion by about 0.028 R^2; only the generative Gaussian family shows a real advantage, about +0.023 R^2 over its own matched counterpart.
- **APIs exercised:** `MissDiagnostic`, `MissRecommender`, `MissLinear`, `MissBayesRegressor`, `MissRidgeRegressor`, `MissExplainer`, `MissSensitivity`.
- **Runtime:** roughly 5 to 10 minutes.
- **Script equivalent:** [`06_air_quality.py`](06_air_quality.py), independent of the notebook, with `--quick` for a fast pass.

---

## 07_Galaxy_Redshift_Head_To_Head.ipynb: Beating a published imputation study on its own data

- **Dataset:** 1,311 radio sources from ATLAS DR3, cross-matched to DES optical and Spitzer/SWIRE infrared photometry, with spectroscopic redshifts; cached as `example_data/atlas_redshift_clean.csv`. Downloaded from the authors' own GPL-3.0 repository, so this is a genuine head-to-head rather than a comparison of numbers across different setups.
- **Task:** Regression; spectroscopic redshift from nine photometric predictors (1.4 GHz radio flux, g/r/i/z magnitudes, 3.6/4.5/5.8/8.0 micron infrared fluxes).
- **The study:** Luken, Padhy and Wang, *"Missing Data Imputation for Galaxy Redshift Estimation"* (arXiv:2111.13806). They benchmark mean, median, minimum, maximum, kNN, MICE and GAIN imputation, then estimate redshift with kNN regression, and report **MICE as their best imputer**.
- **Missingness:** injected, following their protocol exactly: 70/30 split, missingness into the **test set only**, at 2, 5, 10, 15, 20, 25 and 30 percent, over 20 seeds. That design is unusual and it decides what the comparison measures, because the model trains on complete data and must predict from incomplete inputs.
- **Fairness:** every arm is the same model class, k-nearest-neighbours with the same k. `MissNeighborsRegressor` is kNN on the *expected* distance under a fitted joint Gaussian, so only the missing-data treatment varies. Two deviations are stated in the notebook: they used Mahalanobis distance and a tuned k, and a different MICE implementation.
- **Result:** FIML leads at every rate from 2% to 25% and ties kNN imputation at 30%, on both RMSE and the photometric outlier rate, **despite** their preprocessing choice (standardised but unlogged fluxes) working against a Gaussian working model.
- **The case they defer:** their paper states it treats only data "missing at random" and leaves non-detections to future work. That deferred case is MNAR, and it is the physically real one. Implemented here as faintness-driven absence, **MICE degrades to the level of mean imputation**, because conditional imputation cannot extrapolate to values that are missing precisely *because* they were extreme.
- **Also demonstrates:** missingness SHAP inverts the value-SHAP ranking. The optical bands contribute most per reading but are the *least* worth observing, since g/r/i/z reconstruct each other; the 8.0 and 5.8 micron bands are the most valuable to acquire, because they probe warm dust that no optical filter substitutes for. That is a survey-design conclusion invisible to ordinary attribution.
- **Runtime:** roughly 5 minutes.

## 08_Two_Labels_Graphene_Oxide.ipynb: One missingness pattern, two labels, and a conservation law

- **Dataset:** 1,617 relaxed graphene oxide structures with 465 composition, distortion and bond-topology descriptors, from CSIRO (doi:10.25919/5e30b45f9852c); supplied as `example_data/graphene_oxide.csv`.
- **Task:** Two regressions from identical descriptors, `Formation_energy` and `Fermi_energy`. Both labels are fully observed; 9.9% of descriptor cells are not.
- **Missingness:** **real, native, and structural**. A bond-motif descriptor is `NaN` exactly when its motif fraction is zero, verified at agreement 1.000 across all 47 motif pairs. The value is absent because the bond does not exist in that structure, so **imputing it invents a bond length for a bond that is not there**. Because every structure lacks some motif, **100% of rows are incomplete** and listwise deletion returns zero structures.
- **The main lesson:** this mechanism is formally **MAR**, being fully determined by an observed column, and imputation is still the wrong choice. MAR is usually taught as the condition under which imputation is safe; it is not sufficient. The incomplete columns turn out to carry **no incremental information** (adding them moves R² by −0.0000 and −0.0002), so the correct treatment is neither imputation nor marginalisation but **removal**, which is what `MissRecommender` advises at its 60% threshold.
- **The label question, answered:** with 30% MAR injected into the informative composition block, `Fermi_energy` degrades about **2.7 times more** than `Formation_energy`. The reason is **compositional closure**: C+H+O sums to exactly 1 (sd 2.4e-08), so any masked concentration is recoverable from the other two, and `Formation_energy` is a stoichiometric identity in composition (R² = 0.99999548 from C/H/O alone). A conservation law is free information, and FIML exploits it through the covariance structure.
- **A trap worth knowing:** the same closure makes the design matrix **singular before any missingness** (condition number 5.4e17), so ordinary least squares diverges to R² of order −1e8 on this data. Both arms of the benchmark are therefore penalised identically; had they not been, the notebook would have reported a spectacular FIML win that was purely a numerical artifact.
- **Result:** FIML ahead of MICE on both labels, narrowly on formation energy and more clearly on Fermi energy (0.9974 vs 0.9967; 0.9643 vs 0.9626), both clear of mean and kNN imputation. MICE does unusually well here for a good reason: the closure constraint is exactly linear, which is what its chained regressions can recover.
- **Runtime:** roughly 5 to 10 minutes.

## 09_Blockwise_Large_p_SECOM.ipynb: Blockwise missingness, and large p

- **Dataset:** UCI SECOM semiconductor fabrication line; 1,567 wafer lots by 590 process and metrology signals, 474 after dropping constant and empty channels; cached as `example_data/secom.data`.
- **Task:** Binary classification; wafer lot pass or fail. Only **6.6% fail**, so accuracy is useless (always predicting "pass" scores 93.4%) and ROC-AUC and the Brier score are used instead.
- **Missingness:** **real and native**, and **blockwise**. Only 4.5% of cells are absent, yet **100% of lots have at least one absent reading**, so listwise deletion returns an empty table. All 474 surviving channels fall into **36 co-missing groups**: whole banks of sensors drop out together when an instrument or a logging step fails.
- **The central point:** FIML stage one costs **O(G p³)** in the number of *distinct patterns* G, not in the missing rate, so **blockwise structure is what makes large p reachable**. SECOM gives **G = 198 at p = 474**. On a synthetic control at matched rate, scattered cell-wise missingness gives G = 1,499 and a 324 second fit while blockwise gives G = 32 and 2.8 seconds, a **116-fold difference from structure alone**, at which point FIML is faster than MICE.
- **The honest ceiling, reported not hidden:** p=20 G=9 0.6s AUC 0.613; p=40 G=18 60s 0.731; p=60 G=28 171s 0.765; p=90 G=53 **2,564s** 0.809. More channels genuinely help and 0.809 is good for SECOM, but cost grew about **4,300-fold** from p=20 to p=90 where O(G p³) accounts for only 537 of it, so the **empirical scaling is roughly 8 times worse than theory**. The practical range is p = 40 to 60; above it, screen channels or hand off through `MissImputer`.
- **Also demonstrates:** repair priority is not absence frequency. The channel absent in 65% of lots ranks among the *least* worth fixing, because the channels co-measured with it already carry its information. This dataset is also what exposed the classifier SHAP bug fixed in 0.9.1, since attributing hard labels at a 6.6% base rate zeroed every attribution.
- **Runtime:** roughly 5 minutes at the default p = 40.

## 10_Heart_Disease_Indicators.ipynb: Do missing-indicators inform, or proxy the site?

- **Dataset:** UCI Heart Disease, **all four collection sites** rather than the Cleveland subset usually used alone; 920 patients, 13 clinical attributes; cached as `example_data/processed.*.data`.
- **Task:** Binary classification; presence of heart disease, prevalence 55.3%.
- **The study:** Lenz, Peralta and Cornelis, *"No imputation without representation"* (arXiv:2206.14254; Springer CCIS 2024). Across twenty datasets they find that adding binary **missing-indicators** alongside imputation improves classification, and that kNN and iterative imputation do not beat plain mean imputation once indicators are present. That is the direct alternative to marginalisation: an indicator *represents* the absence, a likelihood *marginalises* it.
- **Missingness:** **real and native**, and almost a function of the collection site. `ca` is 1% absent at Cleveland and 99% at Hungary; `thal` 1% and 90%; `slope` 0% and 65%; `fbs` 0% at Cleveland and 61% at Switzerland. Pooled: 14.7% of cells, 67.5% of patients, but only **G = 31** patterns, because the pattern is set by *where the patient was treated*. That makes MAR-given-site a **verifiable fact** here rather than an assumption.
- **The finding:** disease prevalence spans **57 percentage points** across sites (36.1% Hungary to 93.5% Switzerland), so anything encoding the site predicts the outcome while saying nothing about the absent measurement. Testing each indicator marginally and then within sites: `fbs` (p = 9.6e-09 marginally, **0.328** within-site) and `ca` (2.0e-04, **0.571**) lose their association **entirely**; `slope` is the honest exception, retaining real within-site information.
- **Result:** indicators help (**+0.0076** AUC), reproducing the published finding. But **site alone helps about twice as much** (+0.0145), and **once site is present the indicators add nothing at all** (−0.0008): a proxy for a variable you already have is not worth its variance. The strongest arms model the mechanism: `FIML + site` 0.9024 and `MissMixed` with a site random intercept 0.9033, against 0.8859 for plain mean imputation and 0.8891 for FIML without the site.
- **Also demonstrates:** the split can matter more than the estimator. Under **leave-one-site-out**, every arm loses about 0.10 AUC, the fold standard deviations (0.09 to 0.11) exceed the entire spread between arms, and a site coefficient for a hospital never seen in training is not even identifiable. All three caveats are printed by the notebook rather than left for the reader to notice.
- **Runtime:** roughly 5 minutes.

---

## Other files

| File | Role |
|------|------|
| `01_parkinsons_mixed.py` | command-line equivalent of notebook 01 |
| `02_thyroid_ensemble.py` | command-line equivalent of notebook 02 |
| `03_wine_pipeline.py` | command-line equivalent of notebook 03 |
| `04_pima_strategies.py` | command-line equivalent of notebook 04 |
| `05_credit_approval_benchmark.py` | command-line equivalent of notebook 05 |
| `06_air_quality.py` | command-line equivalent of notebook 06, hand-maintained and independent |
| `07_galaxy_redshift.py` | command-line equivalent of notebook 07 |
| `08_graphene_oxide.py` | command-line equivalent of notebook 08 |
| `09_secom_blockwise.py` | command-line equivalent of notebook 09 |
| `10_heart_disease_indicators.py` | command-line equivalent of notebook 10 |
| `real_data_fair_benchmark.py` | strategy comparison across several families on Pima, Credit and Thyroid, model class held fixed |
| `real_data_fair_results.csv` | results written by the above |
| `example_data/` | cached datasets, populated on first run |
