Test Suite
==========

The project uses ``pytest`` for testing.

Running tests
-------------

.. code-block:: bash

   cd backend
   pytest -v

Test Coverage
-------------

The test suite includes:

- Unit tests for data loading and preprocessing (``fivedreg.data``).
- Unit tests for the MLP model (training, prediction shapes, loss decreasing).
- Integration tests that exercise FastAPI endpoints (``/health``, ``/upload``,
  ``/train``, ``/predict``).
]
