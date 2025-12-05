#!/usr/bin/env bash
# Clean up Docker containers and volumes

set -e

echo "========================================="
echo "Docker Cleanup"
echo "========================================="
echo ""

echo "⚠️  This will:"
echo "  - Stop all running containers"
echo "  - Remove containers"
echo "  - Remove volumes (data will be lost)"
echo ""

read -p "Continue? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "🛑 Stopping containers..."
docker-compose down

echo ""
echo "🗑️  Removing volumes..."
docker-compose down -v

echo ""
echo "🧹 Cleaning up unused Docker resources..."
docker system prune -f

echo ""
echo "========================================="
echo "✅ Cleanup complete!"
echo "========================================="
