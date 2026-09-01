API reference
=============

.. currentmodule:: MissLearn

Estimators
----------

Each family provides a regressor, a classifier, and a dispatcher that selects
between them from ``y``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissLinear
   MissLogistic
   MissRidgeRegressor
   MissRidgeClassifier
   MissLASSORegressor
   MissLASSOClassifier
   MissBayesRegressor
   MissBayesClassifier
   MissNeighborsRegressor
   MissNeighborsClassifier
   MissSupportRegressor
   MissSupportClassifier
   MissGaussianRegressor
   MissGaussianClassifier
   MissMixedRegressor
   MissMixedClassifier

Dispatchers
-----------

Each selects the regressor or the classifier of its family by inspecting ``y``.
Convenient when the task is decided at runtime; prefer the explicit name when
you know it, since it reads more clearly.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissRidge
   MissLASSO
   MissBayes
   MissNeighbors
   MissSupport
   MissGaussian
   MissMixed

Meta-estimators
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissEnsemble
   MissMulticlass
   MissPreprocessor

Tools
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissImputer
   MissDiagnostic
   MissRecommender
   MissExplainer
   MissSensitivity

Model selection
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissKFold
   MissStratifiedKFold

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   prefit_check
   recommend_model
   miss_cross_val_score
   miss_cross_validate
   check_missing_data_estimator
   clear_cache

Conformance reporting
---------------------

``check_missing_data_estimator`` returns a report object rather than
raising, so it can be printed, asserted on, or iterated in a
parametrised test.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissingDataReport
