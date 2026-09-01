MissLearn
=========

Full-information maximum likelihood estimators with native missing-data
support, following the scikit-learn estimator API.

MissLearn fits models directly on matrices containing ``NaN``. Nothing is
deleted, nothing is filled in. Every estimator maximises the likelihood of the
data you actually observed, marginalising analytically over the entries you did
not.

.. code-block:: python

   import numpy as np
   from MissLearn import MissLinear

   X = np.random.default_rng(0).normal(size=(500, 6))
   y = X @ np.random.default_rng(1).normal(size=6)
   X[np.random.default_rng(2).random(X.shape) < 0.2] = np.nan

   model = MissLinear().fit(X, y)      # no imputation, no deletion
   model.summary()

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   USER_GUIDE
   COMPUTATIONAL_GUIDE
   INTERPRETATION_GUIDE
   METHODS_GUIDE
   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   ROADMAP
   SKLEARN_CONTRIB_ROADMAP
   code_of_conduct
   contributing
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
