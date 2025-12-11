#!/usr/bin/env bash
set -e

echo "========================================="
echo "5D Dataset Interpolator - Local Launch"
echo "========================================="
echo ""

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo " Checking prerequisites..."
echo ""

# Check Docker and Docker Compose
if command_exists docker; then
    echo " Docker found: $(docker --version)"
else
    echo " Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi

if command_exists docker-compose; then
    echo " Docker Compose found: $(docker-compose --version)"
else
    echo "  Docker Compose not found. Please install Docker Compose:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo ""
echo "========================================="
echo " Starting Application Stack"
echo "========================================="
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Stop any existing containers
echo " Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start services
echo ""
echo " Building Docker images (this may take a few minutes)..."
docker-compose build

echo ""
echo "▶  Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo " Waiting for services to be ready..."
sleep 5

# Check backend health
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        echo "Backend is healthy"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "   Waiting for backend... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo " Backend failed to start. Check logs with: docker-compose logs backend"
    exit 1
fi

# Check frontend
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:3001 >/dev/null 2>&1; then
        echo " Frontend is healthy"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "   Waiting for frontend... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo " Frontend failed to start. Check logs with: docker-compose logs frontend"
    exit 1
fi

echo ""
echo "========================================="
echo " Application Stack Running!"
echo "========================================="
echo ""
echo " Access points:"
echo "   Frontend:     http://localhost:3001"
echo "   Backend API:  http://localhost:8001"
echo "   API Docs:     http://localhost:8001/docs"
echo ""
echo " Sample Dataset:"
echo "   Location:     $PROJECT_ROOT/sample_dataset.pkl"
echo "   Use this file to test the upload functionality"
echo ""
echo "  Documentation:"
echo "   HTML Docs:    file://$PROJECT_ROOT/backend/docs/build/html/index.html"
echo "   README:       $PROJECT_ROOT/README.md"
echo ""
echo "  Useful commands:"
echo "   View logs:    docker-compose logs -f"
echo "   Stop stack:   docker-compose down"
echo "   Restart:      docker-compose restart"
echo ""
echo "========================================="
echo "Press Ctrl+C to stop following logs, or close terminal to keep running"
echo "========================================="
echo ""

# Follow logs
docker-compose logs -f
