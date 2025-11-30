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

   ./run_local.sh

This will:

- Build Docker images for backend and frontend.
- Start all services defined in ``docker-compose.yml``.

Accessing the application
-------------------------

Once the stack is running:

- Frontend UI: ``http://localhost:3000``
- Backend API: ``http://localhost:8000`` (or via Nginx on port 80/443 in production)

Uploading the dataset
---------------------

From the frontend:

1. Open ``http://localhost:3000/upload``.
2. Select the provided ``.npz`` dataset file.
3. Click *Upload*. The backend stores the dataset and makes it available for training.

From the API directly:

.. code-block:: bash

   curl -X POST "http://localhost:8000/upload" \
        -F "file=@dataset.npz"

