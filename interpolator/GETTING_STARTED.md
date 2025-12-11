# Getting Started

This guide will help you launch the complete application stack locally.

## Quick Start (3 Steps)

### 1. Build Documentation

```bash
cd backend/docs
./build_docs.sh
```

This will:
- Build the Sphinx HTML documentation
- Output to `backend/docs/build/html/`
- Open with: `open backend/docs/build/html/index.html`

### 2. Launch the Application Stack

```bash
./launch.sh
```

This script will:
- ✅ Check Docker and Docker Compose are installed
- ✅ Build Docker images for backend and frontend
- ✅ Start all services (backend + frontend)
- ✅ Wait for health checks to pass
- ✅ Display access URLs

**Services will be available at:**
- Frontend: http://localhost:3001
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

### 3. Upload Sample Dataset

1. Open the frontend: http://localhost:3001
2. Navigate to **Upload** page
3. Upload the file: `sample_dataset.pkl` (in project root)
4. Go to **Train** page and configure hyperparameters
5. Click "Start Training"
6. Go to **Predict** page and make predictions

---

## Detailed Documentation

### Sphinx HTML Documentation

After running `./backend/docs/build_docs.sh`, open the docs:

```bash
# macOS
open backend/docs/build/html/index.html

# Linux
xdg-open backend/docs/build/html/index.html

# Windows
start backend/docs/build/html/index.html
```

Or manually navigate to:
```
file:///Users/basiakoch/bk489/C1_coursework/backend/docs/build/html/index.html
```

The documentation includes:
- **Installation**: Setup instructions
- **Usage**: How to use the application
- **API Reference**: Complete API documentation
- **Performance**: Benchmark results
- **Tests**: Test suite description
- **Deployment**: Production deployment guide

---

## Creating Custom Datasets

To create your own test dataset:

```python
import numpy as np
import pickle

# Generate data
X = np.random.randn(1000, 5).astype(np.float32)  # 1000 samples, 5 features
y = np.random.randn(1000).astype(np.float32)     # 1000 target values

# Save as pickle
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump((X, y), f)
```

**Requirements:**
- X must have exactly 5 features (columns)
- X and y must have the same number of samples (rows)
- Supported formats:
  - Tuple: `(X, y)`
  - Dict: `{'X': X, 'y': y}` or `{'data': X, 'target': y}`

---

## File Locations

### Documentation
```
backend/docs/
├── build_docs.sh          # Script to build docs
├── source/                # Source .rst files
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main page
│   ├── api.rst           # API reference
│   ├── usage.rst         # Usage guide
│   ├── performance.rst   # Benchmark results
│   ├── tests.rst         # Test documentation
│   └── deployment.rst    # Deployment guide
└── build/html/           # Built HTML docs (after build)
    └── index.html        # Main HTML page
```

### Sample Dataset
```
sample_dataset.pkl         # 1000 samples, 5D features
```

### Launch Scripts
```
launch.sh                  # Complete stack launcher
create_sample_dataset.py   # Dataset generator
```

---

## Stopping the Stack

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes trained models)
docker-compose down -v
```

---

## Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend
```

---

## Rebuilding After Code Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# Or use launch script
./launch.sh
```

---

## Troubleshooting

### Documentation Build Fails

**Problem:** `sphinx-build: command not found`

**Solution:**
```bash
cd backend
source .venv/bin/activate
pip install -e ".[docs]"
cd docs
./build_docs.sh
```

### Launch Script Fails

**Problem:** Docker or Docker Compose not found

**Solution:** Install Docker Desktop from https://www.docker.com/products/docker-desktop

### Services Won't Start

**Problem:** Ports 3001 or 8001 already in use

**Solution:**
```bash
# Find and kill processes using the ports
lsof -ti:3001 | xargs kill -9
lsof -ti:8001 | xargs kill -9

# Or change ports in docker-compose.yml
```

### Frontend Can't Connect to Backend

**Problem:** CORS errors or connection refused

**Solution:**
```bash
# Check backend is running
curl http://localhost:8001/health

# Restart services
docker-compose restart
```

---

## Complete Workflow Example

```bash
# 1. Build documentation
cd backend/docs
./build_docs.sh
cd ../..

# 2. Launch stack
./launch.sh
# Wait for "✅ Application Stack Running!"
# Press Ctrl+C to stop following logs (services keep running)

# 3. Open frontend in browser
open http://localhost:3001

# 4. Upload sample dataset
# - Click "Upload" → Select sample_dataset.pkl → Upload

# 5. Train model
# - Click "Train" → Configure params → Start Training

# 6. Make predictions
# - Click "Predict" → Enter values → Predict

# 7. View documentation
open backend/docs/build/html/index.html

# 8. When done, stop services
docker-compose down
```

---

## Additional Resources

- **Main README**: `README.md`
- **Quick Start**: `QUICK_START.md`
- **Deployment Guide**: `DEPLOYMENT.md`
- **API Documentation**: http://localhost:8001/docs (when running)
- **HTML Documentation**: `backend/docs/build/html/index.html` (after build)

---

## Summary of Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `launch.sh` | Start complete stack | Project root |
| `build_docs.sh` | Build Sphinx docs | `backend/docs/` |
| `create_sample_dataset.py` | Generate test data | Project root |

All scripts are ready to use - just run them!
