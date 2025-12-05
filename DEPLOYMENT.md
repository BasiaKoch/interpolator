# Deployment Guide

### Pre-Deployment

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Frontend builds successfully (`npm run build`)
- [ ] Docker images build successfully
- [ ] Environment variables configured
- [ ] SSL/TLS certificates ready
- [ ] Domain names configured
- [ ] Backup strategy in place

---

## Docker Compose Production

### 1. Production Configuration

Create `docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: backend-prod
    ports:
      - "8000:8000"
    environment:
      - UVICORN_PORT=8000
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    volumes:
      - backend-artifacts:/app/artifacts
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    networks:
      - prod-network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: frontend-prod
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
      - NODE_ENV=production
    depends_on:
      backend:
        condition: service_healthy
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    networks:
      - prod-network

  nginx:
    image: nginx:alpine
    container_name: nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - backend
    restart: always
    networks:
      - prod-network

volumes:
  backend-artifacts:
    driver: local

networks:
  prod-network:
    driver: bridge
```

### 2. Deploy

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

---

## Cloud Deployment

### AWS EC2

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone <repository-url>
cd C1_coursework

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```
---

## Monitoring

### Health Checks

```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Check all services
docker-compose ps
```

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

---


## CI/CD Pipeline (GitHub Actions Example)

`.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: |
          cd backend
          pip install -e ".[dev]"
          pytest tests/

      - name: Build Docker images
        run: docker-compose build

      - name: Deploy to server
        run: |
          # Add deployment script here
```


---

## Rollback

```bash
# Stop current deployment
docker-compose down

# Pull previous version
git checkout <previous-tag>

# Redeploy
docker-compose up --build -d
```

---

## Support

For issues, check:
1. Service logs: `docker-compose logs -f`
2. Container status: `docker-compose ps`
3. Resource usage: `docker stats`
