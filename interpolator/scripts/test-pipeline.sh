#!/usr/bin/env bash
# Run complete test pipeline

set -e

echo "========================================="
echo "Test Pipeline"
echo "========================================="
echo ""

cd backend

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found"
    exit 1
fi

echo ""
echo "1️⃣  Running unit tests..."
echo "========================================="
pytest tests/ -v --tb=short

echo ""
echo "2️⃣  Running tests with coverage..."
echo "========================================="
pytest tests/ --cov=fivedreg --cov-report=term-missing --cov-report=html

echo ""
echo "3️⃣  Checking code quality (if ruff installed)..."
echo "========================================="
if command -v ruff &> /dev/null; then
    ruff check fivedreg/
else
    echo "⚠️  Ruff not installed, skipping linting"
fi

echo ""
echo "========================================="
echo "✅ All tests passed!"
echo "========================================="
echo ""
echo "📊 Coverage report: backend/htmlcov/index.html"
echo "🌐 Open with: open htmlcov/index.html"
echo ""
