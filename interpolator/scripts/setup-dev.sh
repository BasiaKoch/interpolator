#!/usr/bin/env bash
# Setup development environment

set -e

echo "========================================="
echo "Development Environment Setup"
echo "========================================="
echo ""

# Backend setup
echo "1️⃣  Setting up backend..."
cd backend

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "📦 Activating virtual environment..."
source .venv/bin/activate

echo "📥 Installing backend dependencies..."
pip install -e ".[dev,docs,bench]"

echo "🔧 Installing ipykernel for Jupyter..."
pip install ipykernel jupyterlab
python -m ipykernel install --user --name c1_cw --display-name "c1_cw"

cd ..

# Frontend setup
echo ""
echo "2️⃣  Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    npm install
else
    echo "✅ Frontend dependencies already installed"
fi

cd ..

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  Backend:  ./backend/start_server.sh"
echo "  Frontend: cd frontend && npm run dev"
echo "  Docker:   ./launch.sh"
echo "  Tests:    ./scripts/test-pipeline.sh"
echo "  Docs:     ./scripts/build-docs.sh"
echo ""
