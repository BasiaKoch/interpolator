Deployment and Local Stack
==========================

Docker-based deployment
-----------------------

The repository includes a ``docker-compose.yml`` file that defines services for:

- The FastAPI backend (``backend``)
- The Next.js frontend (``frontend``)
- Optionally an Nginx reverse proxy (for production)

Local launch script
-------------------

To start the full stack locally:

.. code-block:: bash

   ./launch.sh

This will:

- Build Docker images for backend and frontend.
- Start all services defined in ``docker-compose.yml``.

Accessing the application
-------------------------

**Docker deployment (default):**

- Frontend UI: ``http://localhost:3001``
- Backend API: ``http://localhost:8001``
- API Docs: ``http://localhost:8001/docs``

**Local development (without Docker):**

- Frontend UI: ``http://localhost:3000``
- Backend API: ``http://localhost:8000``

Uploading the dataset
---------------------

From the frontend:

1. Open ``http://localhost:3001/upload`` (Docker) or ``http://localhost:3000/upload`` (local dev).
2. Select the provided ``.pkl`` dataset file.
3. Click *Upload*. The backend stores the dataset and makes it available for training.

From the API directly:

.. code-block:: bash

   # Docker deployment
   curl -X POST "http://localhost:8001/upload" \
        -F "file=@sample_dataset.pkl"

   # Local development
   curl -X POST "http://localhost:8000/upload" \
        -F "file=@sample_dataset.pkl"

