Usage Guide
===========

Installation
------------

.. code-block:: bash

   git clone <your-gitlab-url>
   cd backend
   pip install -e ".[dev]"

Running the backend locally
---------------------------

.. code-block:: bash

   uvicorn fivedreg.main:app --reload --port 8000

Running the frontend locally
----------------------------

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Example: training via API
-------------------------

.. code-block:: bash

   curl -X POST "http://localhost:8000/train" \
        -H "Content-Type: application/json" \
        -d '{"dataset_name": "default", "epochs": 100}'

Example: prediction via API
---------------------------

.. code-block:: bash

   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"inputs": [0.1, 0.2, -0.3, 0.5, 1.0]}'

