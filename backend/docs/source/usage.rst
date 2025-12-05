Usage Guide
===========

Installation
------------

**Clone the repository:**

.. code-block:: bash

   # SSH (if you have SSH keys set up)
   git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/c1_coursework/bk489.git
   cd bk489

   # OR HTTPS (works with username/password)
   git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/c1_coursework/bk489.git
   cd bk489

**Install backend package:**

.. code-block:: bash

   cd backend
   pip install -e ".[dev]"

Running the backend locally
---------------------------

**Without Docker:**

.. code-block:: bash

   cd backend
   source .venv/bin/activate
   uvicorn fivedreg.main:app --reload --port 8000

Backend will be available at ``http://localhost:8000``

**With Docker:**

.. code-block:: bash

   ./launch.sh

Backend will be available at ``http://localhost:8001``

Running the frontend locally
----------------------------

**Without Docker:**

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Frontend will be available at ``http://localhost:3000``

**With Docker:**

Use ``./launch.sh`` (frontend available at ``http://localhost:3001``)

Example: training via API
-------------------------

.. code-block:: bash

   # Docker deployment (port 8001)
   curl -X POST "http://localhost:8001/train?temp_path=/tmp/tmpXXXX.npz" \
        -H "Content-Type: application/json" \
        -d '{"hidden": [64,32], "lr": 0.001, "max_epochs": 100, "batch_size": 256, "patience": 15, "scale_y": false}'

   # Local development (port 8000)
   curl -X POST "http://localhost:8000/train?temp_path=/tmp/tmpXXXX.npz" \
        -H "Content-Type: application/json" \
        -d '{"hidden": [64,32], "lr": 0.001, "max_epochs": 100, "batch_size": 256, "patience": 15, "scale_y": false}'

Example: prediction via API
---------------------------

.. code-block:: bash

   # Docker deployment (port 8001)
   curl -X POST "http://localhost:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{"X": [[0.1, 0.2, -0.3, 0.5, 1.0]]}'

   # Local development (port 8000)
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"X": [[0.1, 0.2, -0.3, 0.5, 1.0]]}'

