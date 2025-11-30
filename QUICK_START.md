# Quick Start Guide


### 1. Start the Application
```bash
docker-compose up --build
```

### 2. Access the Application
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

### 3. Use the Application
1. Go to Upload page → Upload `.pkl` file
2. Go to Train page → Configure & train model
3. Go to Predict page → Make predictions

---

## Common Commands

### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up --build

# Clean restart (removes volumes)
docker-compose down -v && docker-compose up --build
```

### Backend Commands

```bash
cd backend

# Activate virtual environment
source .venv/bin/activate

# Run server
uvicorn fivedreg.main:app --reload

# Run tests
pytest tests/ -v

# Run specific tests
pytest tests/test_api.py -v

# Test coverage
pytest tests/ --cov=fivedreg
```

### Frontend Commands

```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build
npm start

# Linting
npm run lint
```

---

## Environment Variables

### Backend (.env)
```env
UVICORN_PORT=8000
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
PORT=3000
```

---

## 📖 Full Documentation

See [README.md](README.md) for comprehensive documentation.
