# 5D Dataset Interpolator

Welcome! This is a full-stack machine learning application for training neural network models on 5-dimensional datasets and making predictions through an intuitive web interface. This repo has been designed for the coursework assigment of the Research Comuting class at University of Cambridge.

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Installation & Setup](#installation--setup)
  - [Docker Setup (Recommended)](#docker-setup-recommended)
  - [Local Development Setup](#local-development-setup)
- [Usage](#usage)
  - [Upload Dataset](#1-upload-dataset)
  - [Train Model](#2-train-model)
  - [Make Predictions](#3-make-predictions)
  - [Using the Package Programmatically](#4-using-the-package-programmatically)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

This application provides an end-to-end machine learning workflow:

1. **Upload**: Upload 5-dimensional datasets in `.pkl` format
2. **Train**: Configure and train Multi-Layer Perceptron (MLP) models with customizable hyperparameters
3. **Predict**: Make real-time predictions using trained models

### Key Features

-  **FastAPI Backend**: High-performance REST API with automatic documentation
-  **Next.js Frontend**: Modern React-based UI with TypeScript
-  **PyTorch Models**: Flexible neural network architecture with early stopping
-  **Docker Support**: Containerized deployment with Docker Compose
-  **Comprehensive Testing**: 60+ tests covering all components
-  **Health Checks**: Built-in monitoring for production deployments

### Technology Stack

**Backend:**
- Python 3.12
- FastAPI
- PyTorch
- scikit-learn
- NumPy

**Frontend:**
- Next.js 16
- React 19
- TypeScript
- Tailwind CSS

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                            │
│                  (Next.js + React + TS)                     │
│                    Port: 3000/3001                          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/JSON
                         │
┌────────────────────────▼────────────────────────────────────┐
│                         Backend                             │
│                   (FastAPI + PyTorch)                       │
│                    Port: 8000/8001                          │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   /upload   │  │    /train    │  │   /predict   │        │
│  └─────────────┘  └──────────────┘  └──────────────┘        │
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │           MLP Model (PyTorch)                   │        │
│  │  Input(5) → Hidden Layers → Output(1)          │         │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   Artifacts   │
                 │ (Saved Models)│
                 └───────────────┘
```

---

## Prerequisites

### For Docker Deployment (Recommended)
- [Docker](https://docs.docker.com/get-docker/) >= 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.0

### For Local Development
- [Python](https://www.python.org/downloads/) >= 3.11
- [Node.js](https://nodejs.org/) >= 20.0
- [npm](https://www.npmjs.com/) >= 10.0

---

## Environment Variables

### Backend Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `UVICORN_PORT` | Port for FastAPI server | `8000` | No |
| `PYTHONDONTWRITEBYTECODE` | Disable .pyc file generation | `1` | No |
| `PYTHONUNBUFFERED` | Force unbuffered stdout/stderr | `1` | No |

**Example `.env` for backend:**
```env
UVICORN_PORT=8000
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

### Frontend Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | `http://localhost:8000` | Yes |
| `PORT` | Port for Next.js server | `3000` | No |
| `NODE_ENV` | Environment mode | `production` | No |
| `NEXT_TELEMETRY_DISABLED` | Disable Next.js telemetry | `1` | No |

**Example `.env.local` for frontend:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
PORT=3000
NODE_ENV=development
```

**Docker Compose Environment:**
```yaml
# Frontend connects to backend via Docker network
NEXT_PUBLIC_API_URL=http://backend:8000
```

---

## Installation & Setup

### Docker Setup (Recommended)

The easiest way to get started is using the automated launch script:

#### Quick Start

```bash
# Clone the repository (SSH)
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/c1_coursework/bk489.git
cd bk489/C1_coursework/interpolator

# OR clone with HTTPS
# git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/c1_coursework/bk489.git
# cd bk489/C1_coursework/interpolator

# Launch the application
./launch.sh
```

**That's it!** The script will:
- ✓ Check Docker and Docker Compose are installed
- ✓ Build both frontend and backend images
- ✓ Start all services with health checks
- ✓ Wait for services to be ready
- ✓ Display access URLs and helpful information
- ✓ Stream logs from both containers

**Access the application:**
- Frontend: `http://localhost:3001`
- Backend API: `http://localhost:8001`
- API Docs: `http://localhost:8001/docs`

**To stop:** Press `Ctrl+C` (containers keep running) or run `docker-compose down`

---

#### Alternative: Manual Docker Compose

If you prefer manual control:

```bash
# Build and start
docker-compose up --build

# Or run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services (preserves volumes)
docker-compose down

# Stop and remove everything (including trained models!)
docker-compose down -v
```

---

#### Verify Services

```bash
# Check backend health
curl http://localhost:8001/health

# Check frontend
curl http://localhost:3001

# View running containers
docker ps
```

---

### Local Development Setup

#### Backend Setup

1. **Navigate to backend directory:**
```bash
# From the interpolator directory:
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Install core dependencies (FastAPI, PyTorch, etc.)
pip install -e .

# Install development dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"

# Install documentation dependencies (Sphinx)
pip install -e ".[docs]"

# Install profiling dependencies (memory-profiler, psutil)
pip install -e ".[bench]"

# Or install everything at once:
pip install -e ".[dev,docs,bench]"
```

4. **Start backend server:**
```bash
uvicorn fivedreg.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Create environment file:**
```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

4. **Start development server:**
```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

---

## Usage

### 1. Upload Dataset

Navigate to `http://localhost:3001/upload` (or `http://localhost:3000` for local dev)

**Dataset Requirements:**
- File format: `.pkl` (Python pickle)
- Data structure: `(X, y)` tuple or `{'X': ..., 'y': ...}` dict
- X shape: `(n_samples, 5)` - exactly 5 features
- y shape: `(n_samples,)` - target values
- Maximum file size: 50MB

**Example Dataset Creation:**
```python
import numpy as np
import pickle

# Generate sample data
X = np.random.randn(1000, 5).astype(np.float32)
y = (2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(1000) * 0.1).astype(np.float32)

# Save as pickle
with open('dataset.pkl', 'wb') as f:
    pickle.dump((X, y), f)
```

**Upload Process:**
1. Click "Select .pkl file"
2. Choose your dataset file
3. Click "Upload Dataset"
4. Copy the displayed `temp_path` (auto-filled in Train page)

---

### 2. Train Model

Navigate to `http://localhost:3001/train`

**Hyperparameters:**

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Hidden Layers** | Layer sizes (comma-separated) | `128,64,32` | e.g., `64,32,16` |
| **Learning Rate** | Optimizer learning rate | `0.001` | `0.0001` - `0.1` |
| **Max Epochs** | Maximum training iterations | `150` | `1` - `1000` |
| **Batch Size** | Training batch size | `256` | `8` - `1024` |
| **Patience** | Early stopping patience | `15` | `1` - `100` |
| **Scale Y** | Standardize target values | `false` | `true/false` |

**Training Process:**
1. Temp path is auto-filled from upload
2. Configure hyperparameters
3. Click "Start Training"
4. View validation metrics (MSE and R²)

**What Happens During Training:**
- Data is split: 70% train, 15% validation, 15% test
- Features (X) are standardized (mean=0, std=1)
- Model trains with early stopping on validation loss
- Best model is saved to `/app/artifacts/latest.joblib`

---

### 3. Make Predictions

Navigate to `http://localhost:3001/predict`

1. Enter values for all 5 features
2. Click "Predict"
3. View predicted value

**Example:**
```
Feature 1: 1.5
Feature 2: -0.8
Feature 3: 2.3
Feature 4: 0.0
Feature 5: -1.2

Result: ŷ = 3.456789
```

---

### 4. Using the Package Programmatically

You can also use the `fivedreg` package directly in Python scripts or Jupyter notebooks.

#### Installation

```bash
cd backend
source .venv/bin/activate
pip install -e .
```

#### Basic Usage

```python
import numpy as np
from fivedreg.interpolator import train_model, interpolate, synthetic_5d

# Generate synthetic data
X, y = synthetic_5d(n=1000, seed=42)

# Train model
model, stats, (val_mse, test_mse) = train_model(
    X, y,
    batch_size=256,
    max_epochs=50,
    patience=10
)

print(f"Validation MSE: {val_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")

# Make predictions
X_new = np.array([[0.5, -0.3, 0.8, -0.2, 0.1]])
predictions = interpolate(model, stats, X_new)
print(f"Prediction: {predictions[0]:.6f}")
```

#### Save and Load Models

```python
from fivedreg.interpolator import save_model, load_model

# Save model
save_model(model, stats, "my_model.pkl")

# Load model
loaded_model, loaded_stats = load_model("my_model.pkl")

# Use loaded model
predictions = interpolate(loaded_model, loaded_stats, X_new)
```

#### Available Functions

Import from `fivedreg.interpolator`:

```python
from fivedreg.interpolator import (
    train_model,      # Train a model
    interpolate,      # Make predictions
    save_model,       # Save model to file
    load_model,       # Load model from file
    synthetic_5d,     # Generate synthetic data
    MLP,              # Model class
    NormStats,        # Normalization statistics class
)
```

#### Data Loading

```python
from fivedreg.data.loader import load_dataset_pkl

# Load dataset from pickle file
X, y = load_dataset_pkl("dataset.pkl")
```

#### Using with Jupyter

Install Jupyter support:
```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name c1_cw --display-name "c1_cw"
```

Start Jupyter Lab:
```bash
jupyter lab
```

Select the "c1_cw" kernel and you'll have access to all `fivedreg` functions.

---

## API Documentation

### Base URL
- Docker: `http://localhost:8001`
- Local: `http://localhost:8000`

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

#### 2. Upload Dataset
```http
POST /upload
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@dataset.pkl"
```

**Response:**
```json
{
  "temp_path": "/tmp/tmpXXXXXX.npz",
  "n_samples": 1000
}
```

---

#### 3. Train Model
```http
POST /train?temp_path={path}
Content-Type: application/json
```

**Request:**
```bash
curl -X POST "http://localhost:8000/train?temp_path=/tmp/tmpXXXXXX.npz" \
  -H "Content-Type: application/json" \
  -d '{
    "hidden": [128, 64, 32],
    "lr": 0.001,
    "max_epochs": 150,
    "batch_size": 256,
    "patience": 15,
    "scale_y": false
  }'
```

**Response:**
```json
{
  "metrics": {
    "val_mse": 0.123456,
    "val_r2": 0.987654
  }
}
```

---

#### 4. Make Predictions
```http
POST /predict
Content-Type: application/json
```

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1.0, 2.0, 3.0, 4.0, 5.0]]
  }'
```

**Response:**
```json
{
  "y": [3.456789]
}
```

---

## Testing

### Backend Tests

**Run all tests:**
```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_data_loader.py -v
pytest tests/test_model.py -v
pytest tests/test_utils.py -v
pytest tests/test_api.py -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=fivedreg --cov-report=html
```

**Test Summary:**
- **60 tests total** (58 passing, 2 skipped)
- **test_data_loader.py**: 19 tests - Data loading, validation, preprocessing
- **test_model.py**: 13 tests - MLP model architecture, training, prediction
- **test_utils.py**: 10 tests - Training utilities, metrics
- **test_api.py**: 18 tests - API endpoints, error handling

---

## Documentation

### Building Sphinx Documentation

The project includes comprehensive Sphinx documentation covering API reference, usage examples, testing, and deployment.

**Install documentation dependencies:**
```bash
cd backend
source .venv/bin/activate
pip install -e ".[docs]"
```

**Build the documentation:**
```bash
cd docs
./build_docs.sh

# Or manually:
make html
```

**View the documentation:**
```bash
# Option 1: Open directly in browser (macOS)
open build/html/index.html

# Option 2: Start a local web server
cd build/html
python -m http.server 8080
# Then visit: http://localhost:8080

# Option 3: File path
# Navigate to: backend/docs/build/html/index.html
```

**Documentation includes:**
- **API Reference**: Complete API endpoint documentation
- **Usage Guide**: How to use the package programmatically
- **Testing**: Test suite documentation and how to run tests
- **Deployment**: Production deployment guide
- **Performance**: Profiling and optimization details

---

## Project Structure

```
bk489/
└── C1_coursework/
    └── interpolator/         # Main project directory
        ├── backend/          # Python FastAPI backend
        │   ├── fivedreg/     # Main package
        │   │   ├── api/      # API state management
        │   │   ├── data/     # Data loading utilities
        │   │   └── main.py   # FastAPI application
        │   ├── docs/         # Sphinx documentation
        │   │   ├── source/   # Documentation source files
        │   │   └── build/    # Built HTML documentation
        │   ├── tests/        # Backend tests (42 tests)
        │   ├── Dockerfile    # Backend container definition
        │   └── pyproject.toml # Python dependencies & project config
        │
        ├── frontend/         # Next.js frontend
        │   ├── src/
        │   │   ├── app/      # Next.js 16 app directory
        │   │   │   ├── upload/   # Upload page
        │   │   │   ├── train/    # Training page
        │   │   │   ├── predict/  # Prediction page
        │   │   │   ├── layout.tsx # Root layout
        │   │   │   └── page.tsx   # Home page
        │   │   └── lib/
        │   │       └── api.ts     # API client
        │   ├── Dockerfile    # Frontend container definition
        │   └── package.json  # Node dependencies
        │
        ├── scripts/          # Utility scripts
        │   ├── build-docs.sh        # Build documentation
        │   ├── run-profiling.sh     # Performance profiling
        │   └── test-pipeline.sh     # Test automation
        │
        ├── docker-compose.yml       # Multi-container orchestration
        ├── launch.sh               # Automated startup script
        ├── README.md               # This file
        └── sample_dataset.pkl      # Example dataset for testing
```

---

## Development Workflow

### Making Changes

#### Backend Changes

1. **Edit code** in `backend/fivedreg/`
2. **Run tests** to verify changes:
   ```bash
   pytest tests/ -v
   ```
3. **Test locally** with hot-reload:
   ```bash
   uvicorn fivedreg.main:app --reload
   ```
4. **Rebuild Docker image** if needed:
   ```bash
   docker-compose build backend
   ```

#### Frontend Changes

1. **Edit code** in `frontend/src/`
2. **Test locally** with hot-reload:
   ```bash
   npm run dev
   ```
3. **Build for production** to check for errors:
   ```bash
   npm run build
   ```
4. **Rebuild Docker image** if needed:
   ```bash
   docker-compose build frontend
   ```

### Code Quality

**Backend:**
```bash
# Linting
ruff check backend/

# Type checking
mypy backend/
```

**Frontend:**
```bash
# Linting
npm run lint
```

---

## Deployment

### Production Deployment with Docker

1. **Set production environment variables:**

Create `.env.production`:
```env
# Backend
UVICORN_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://backend:8000
NODE_ENV=production
```

2. **Build production images:**
```bash
docker-compose -f docker-compose.yml build
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Check health:**
```bash
# Backend health
curl http://localhost:8001/health

# Frontend health
curl http://localhost:3001
```

5. **Monitor logs:**
```bash
docker-compose logs -f
```

### Scaling

**Scale backend instances:**
```bash
docker-compose up -d --scale backend=3
```

**Note**: For production scaling, use a reverse proxy (nginx/traefik) for load balancing.

### Production Checklist

- [ ] Set `NODE_ENV=production`
- [ ] Configure CORS origins in `backend/fivedreg/main.py`
- [ ] Set up HTTPS/SSL certificates
- [ ] Configure volume backups for `backend-artifacts`
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation
- [ ] Set resource limits in docker-compose.yml
- [ ] Enable Docker health checks

---

## Troubleshooting

### Backend won't start

**Check logs:**
```bash
docker-compose logs backend
```

**Common issues:**
- Port 8001 already in use: Change port in `docker-compose.yml` (e.g., `"8002:8000"`)
- Memory issues: Ensure Docker has at least 4GB RAM allocated

### Frontend can't connect to backend

**Verify backend is running:**
```bash
curl http://localhost:8001/health
```

**If using local development** (not Docker):
- Check `.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000`
- Backend must be running on port 8000

**If using Docker:**
- Frontend should use `NEXT_PUBLIC_API_URL=http://backend:8000` (set in docker-compose.yml)
- Both containers must be on the same network

### Upload fails

**File format:**
- Must be `.pkl` file
- Must contain `(X, y)` tuple or `{'X': ..., 'y': ...}` dict
- X must have exactly 5 features
- Maximum size: 50MB

### Training fails

**Common causes:**
- Invalid temp_path from upload step
- Insufficient memory (increase Docker memory limit)
- Invalid hyperparameters (check ranges in Usage section)

### Docker build is slow

**First build** takes 5-10 minutes (downloading dependencies)

**Subsequent builds** should be fast (~10-30 seconds) due to layer caching

**To speed up:**
```bash
# Remove old images
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### View detailed logs

```bash
# All logs
./scripts/docker-logs.sh

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

---

## License

MIT

---

## Authors

-- Barbara Koch bk489@cam.ac.uk

---

**Last Updated:** 2025-11-30
