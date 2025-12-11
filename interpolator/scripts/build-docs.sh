#!/usr/bin/env bash
# Build Sphinx documentation

set -e

echo "========================================="
echo "Building Documentation"
echo "========================================="
echo ""

cd backend/docs

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "❌ Virtual environment not found at backend/.venv"
    exit 1
fi

# Check if sphinx is installed
if ! command -v sphinx-build &> /dev/null; then
    echo "❌ Sphinx not installed. Installing..."
    pip install sphinx
fi

echo "🔨 Building HTML documentation..."
make html

echo ""
echo "========================================="
echo "✅ Documentation built successfully!"
echo "========================================="
echo ""
echo "📂 Output: backend/docs/build/html/index.html"
echo "🌐 Open with: open backend/docs/build/html/index.html"
echo ""
