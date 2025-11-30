# Deployment Guide

## Production Deployment Checklist

### Pre-Deployment

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Frontend builds successfully (`npm run build`)
- [ ] Docker images build successfully (`docker-compose build`)
- [ ] Environment variables configured (if needed)
- [ ] Domain name configured (optional)
- [ ] SSL/TLS certificates ready (if using HTTPS)
- [ ] Backup strategy in place

---

## Simple Production Deployment

**Good news**: Your existing `docker-compose.yml` is already production-ready! It includes:
- ✅ Health checks
- ✅ Restart policies
- ✅ Volume persistence
- ✅ Non-root users
- ✅ Proper networking

### Basic Deployment

```bash
# 1. Clone repository on production server
git clone <repository-url>
cd C1_coursework

# 2. Build images
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Verify deployment
docker-compose ps
curl http://localhost:8001/health
curl http://localhost:3001
```

That's it! Your application is now running in production.

---

## Customizing for Production (Optional)

If you need different settings for production, you can customize using environment variables.

### Option 1: Using .env File

Create a `.env` file in the root directory:

```env
# Custom ports
BACKEND_PORT=8000
FRONTEND_PORT=80

# Restart policy
RESTART_POLICY=always

# Resource limits (if needed)
BACKEND_CPU_LIMIT=2
BACKEND_MEMORY_LIMIT=4g
```

Then update `docker-compose.yml` to use these:

```yaml
services:
  backend:
    ports:
      - "${BACKEND_PORT:-8001}:8000"
    restart: ${RESTART_POLICY:-unless-stopped}
```

### Option 2: Override Specific Values

Use docker-compose override syntax:

```bash
# Override ports for production
docker-compose -f docker-compose.yml up -d \
  -p backend=8000:8000 \
  -p frontend=80:3000
```

### Option 3: Environment-Specific Files

For multiple environments (dev, staging, prod), you can use:

```bash
# Development
docker-compose up

# Production (with overrides)
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

Create `docker-compose.override.yml`:
```yaml
version: "3.9"

services:
  backend:
    ports:
      - "8000:8000"
    restart: always

  frontend:
    ports:
      - "80:3000"
    restart: always
```

---

## Cloud Deployment

### AWS EC2

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Clone repository
git clone <repository-url>
cd C1_coursework

# 4. Deploy
docker-compose up -d
```

### Security Group Rules (AWS)

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| HTTP | 3001 | 0.0.0.0/0 | Frontend access |
| HTTP | 8001 | 0.0.0.0/0 | Backend API |
| SSH | 22 | Your IP | Server access |

**For production with nginx (see below):**

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| HTTP | 80 | 0.0.0.0/0 | Public access |
| HTTPS | 443 | 0.0.0.0/0 | Secure access |
| SSH | 22 | Your IP | Server access |

### Digital Ocean / GCP / Azure

Same process as AWS:
1. Create VM with Docker installed
2. Clone repository
3. Run `docker-compose up -d`

---

## Advanced: Nginx Reverse Proxy (Optional)

For production deployments, you may want to add nginx for:
- SSL/TLS termination
- Load balancing
- Custom domain routing
- Security headers

### Add Nginx to docker-compose.yml

```yaml
services:
  # ... existing services ...

  nginx:
    image: nginx:alpine
    container_name: nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro  # Your SSL certificates
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
    networks:
      - interpolator-network
```

### nginx.conf

Create `nginx.conf` in root directory:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # HTTP server (redirect to HTTPS)
    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

        # Backend API
        location /api/ {
            proxy_pass http://backend/;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks
        location /health {
            proxy_pass http://backend/health;
        }
    }
}
```

### Get SSL Certificate (Free with Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Certificates will be at:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

Then update nginx.conf to use these certificates.

---

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8001/health

# Frontend health
curl http://localhost:3001

# Check all services
docker-compose ps

# Detailed status
docker-compose ps -a
```

### Logs

```bash
# All services (follow mode)
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend

# Since specific time
docker-compose logs --since=2h backend
```

### Resource Usage

```bash
# Real-time stats
docker stats

# Disk usage
docker system df

# Volume usage
docker volume ls
```

---

## Backup & Recovery

### Backup Model Artifacts

The trained models are stored in a Docker volume. Back them up regularly:

```bash
# Create backup
docker run --rm \
  -v backend-artifacts:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/artifacts-backup-$(date +%Y%m%d-%H%M%S).tar.gz /data

# List backups
ls -lh artifacts-backup-*.tar.gz
```

### Restore from Backup

```bash
# Stop services
docker-compose down

# Restore volume
docker run --rm \
  -v backend-artifacts:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/artifacts-backup-20251130-140000.tar.gz -C /

# Restart services
docker-compose up -d
```

### Automated Backups (Cron)

```bash
# Add to crontab
crontab -e

# Backup daily at 2 AM
0 2 * * * cd /path/to/C1_coursework && docker run --rm -v backend-artifacts:/data -v $(pwd):/backup alpine tar czf /backup/artifacts-backup-$(date +\%Y\%m\%d).tar.gz /data

# Keep only last 7 days
0 3 * * * find /path/to/C1_coursework/artifacts-backup-* -mtime +7 -delete
```

---

## Scaling

### Horizontal Scaling (Multiple Backend Instances)

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# Note: You'll need a load balancer (nginx/traefik) to distribute requests
```

### Vertical Scaling (More Resources)

Add resource limits to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  frontend:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

Then restart:
```bash
docker-compose up -d
```

---

## Updates & Rollbacks

### Update to Latest Version

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

### Rollback to Previous Version

```bash
# Stop current deployment
docker-compose down

# Checkout previous version
git checkout <previous-commit-or-tag>

# Redeploy
docker-compose up -d --build
```

### Zero-Downtime Updates (Advanced)

```bash
# Build new images
docker-compose build

# Update one service at a time
docker-compose up -d --no-deps --build backend
docker-compose up -d --no-deps --build frontend
```

---

## CI/CD Pipeline (Optional)

Example GitHub Actions workflow:

`.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run backend tests
        run: |
          cd backend
          pip install -e ".[dev]"
          pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /path/to/C1_coursework
            git pull origin main
            docker-compose up -d --build
```

---

## Security Best Practices

1. **Use HTTPS** - Set up SSL/TLS with nginx (see above)
2. **Firewall** - Only expose necessary ports
3. **Updates** - Regularly update dependencies
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```
4. **Secrets** - Never commit sensitive data
   - Use environment variables
   - Use `.env` files (add to `.gitignore`)
5. **Non-root User** - Already configured in Dockerfiles ✅
6. **Network Isolation** - Already configured ✅
7. **Health Checks** - Already configured ✅
8. **Backups** - Regular automated backups (see above)

---

## Troubleshooting Production Issues

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Check disk space
df -h

# Check if ports are in use
sudo lsof -i :8001
sudo lsof -i :3001
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Restart specific service
docker-compose restart backend

# Check for memory leaks in logs
docker-compose logs backend | grep -i "memory\|oom"
```

### Database/Volume Issues

```bash
# Check volumes
docker volume ls
docker volume inspect backend-artifacts

# Remove and recreate volume (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

### Can't Connect to Backend

```bash
# Check if backend is healthy
docker-compose ps
curl http://localhost:8001/health

# Check network
docker network ls
docker network inspect c1_coursework_interpolator-network

# Check logs
docker-compose logs backend
```

---

## Performance Optimization

### Docker Image Optimization

Your Dockerfiles already use multi-stage builds ✅

### Application Performance

```bash
# Monitor in real-time
docker stats

# Check startup time
docker-compose up --build --detach
docker-compose logs --timestamps
```

### Database/Volume Performance

```bash
# Use local driver (already configured)
volumes:
  backend-artifacts:
    driver: local
```

---

## Support

**For deployment issues:**

1. **Check logs**: `docker-compose logs -f`
2. **Check status**: `docker-compose ps`
3. **Check resources**: `docker stats`
4. **Check health**: `curl http://localhost:8001/health`
5. **Restart services**: `docker-compose restart`

**For clean restart:**
```bash
docker-compose down
docker-compose up -d --build
```

**For complete reset** (WARNING: deletes all data):
```bash
docker-compose down -v
docker system prune -a
docker-compose up -d --build
```

---

## Summary

**Minimal Production Deployment:**
```bash
git clone <repository-url>
cd C1_coursework
docker-compose up -d
```

**With Custom Configuration:**
```bash
# Create .env with your settings
docker-compose up -d
```

**With Nginx + SSL:**
```bash
# Add nginx service to docker-compose.yml
# Configure nginx.conf
# Get SSL certificate
docker-compose up -d
```

Your existing setup is production-ready. Additional complexity (nginx, monitoring, CI/CD) can be added as needed.
