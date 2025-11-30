#!/usr/bin/env bash
set -e

echo "========================================="
echo "Building Sphinx Documentation"
echo "========================================="

# Go to docs directory (this script lives in backend/docs)
DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$DOCS_DIR")"

cd "$DOCS_DIR"

# Activate virtual environment
if [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$BACKEND_DIR/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at $BACKEND_DIR/.venv"
    echo "Please run: cd backend && python -m venv .venv && pip install -e '.[docs]'"
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/html

# Build documentation
echo "Building HTML documentation..."
sphinx-build -b html source build/html

echo ""
echo "========================================="
echo "✅ Documentation built successfully!"
echo "========================================="
echo "Location: $DOCS_DIR/build/html/index.html"
echo ""
echo "To view, open in browser:"
echo "  file://$DOCS_DIR/build/html/index.html"
echo ""
echo "Or run: open build/html/index.html"
echo "========================================="

