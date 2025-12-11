#!/usr/bin/env bash
# View Docker container logs

set -e

echo "========================================="
echo "Docker Container Logs"
echo "========================================="
echo ""

# Check if docker-compose is running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ No containers are running."
    echo "   Start them with: ./launch.sh"
    exit 1
fi

echo "Available options:"
echo "  1. View all logs"
echo "  2. View backend logs only"
echo "  3. View frontend logs only"
echo "  4. Follow logs (live tail)"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        docker-compose logs
        ;;
    2)
        docker-compose logs backend
        ;;
    3)
        docker-compose logs frontend
        ;;
    4)
        echo "Following logs (Ctrl+C to stop)..."
        docker-compose logs -f
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
