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
│                       Port: 8000                            │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   /upload   │  │    /train    │  │   /predict   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │           MLP Model (PyTorch)                   │       │
│  │  Input(5) → Hidden Layers → Output(1)          │       │
│  └─────────────────────────────────────────────────┘       │
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

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd C1_coursework
```

#### 2. Build and Start Services
```bash
docker-compose up --build
```

This will:
- Build both frontend and backend Docker images
- Start the backend on `http://localhost:8001`
- Start the frontend on `http://localhost:3001`
- Create a persistent volume for trained models
- Set up health checks for both services

#### 3. Verify Services
```bash
# Check backend health
curl http://localhost:8001/health

# Check frontend
curl http://localhost:3001

# View logs
docker-compose logs -f
```

#### 4. Stop Services
```bash
# Stop containers (preserves volumes)
docker-compose down

# Stop and remove volumes (deletes trained models)
docker-compose down -v
```

---

### Local Development Setup

#### Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -e .
```

4. **Install development dependencies (for testing):**
```bash
pip install -e ".[dev]"
```

5. **Start backend server:**
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

## Project Structure

```
C1_coursework/
├── backend/                  # Python FastAPI backend
│   ├── fivedreg/            # Main package
│   │   ├── api/             # API state management
│   │   ├── data/            # Data loading utilities
│   │   ├── models/          # PyTorch MLP model
│   │   ├── utils/           # Training utilities
│   │   └── main.py          # FastAPI application
│   ├── tests/               # Backend tests (60 tests)
│   ├── Dockerfile           # Backend container definition
│   └── pyproject.toml       # Python dependencies
│
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # Next.js 16 app directory
│   │   │   ├── upload/      # Upload page
│   │   │   ├── train/       # Training page
│   │   │   ├── predict/     # Prediction page
│   │   │   ├── layout.tsx   # Root layout
│   │   │   ├── page.tsx     # Home page
│   │   │   └── globals.css  # Global styles
│   │   └── lib/
│   │       └── api.ts       # API client
│   ├── Dockerfile           # Frontend container definition
│   ├── package.json         # Node dependencies
│   └── tsconfig.json        # TypeScript config
│
├── docker-compose.yml        # Multi-container orchestration
├── .gitignore               # Git ignore rules
└── README.md                # This file
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

4. **API documentation:**
   Visit `http://localhost:8001/docs` for interactive API testing

---

## License

MIT

---

## Authors

-- Barbara Koch bk489@cam.ac.uk

---

**Last Updated:** 2025-11-30
