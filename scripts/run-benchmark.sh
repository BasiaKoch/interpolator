#!/usr/bin/env bash
# Run performance benchmark

set -e

echo "========================================="
echo "Performance Benchmark"
echo "========================================="
echo ""

cd backend

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found"
    exit 1
fi

echo "🚀 Running benchmark (this may take a few minutes)..."
echo ""

python -m fivedreg.benchmark

echo ""
echo "========================================="
echo "✅ Benchmark complete!"
echo "========================================="
echo ""
echo "📊 Results saved to: backend/fivedreg/benchmark_results.json"
echo ""

if [ -f "fivedreg/benchmark_results.json" ]; then
    echo "Results preview:"
    cat fivedreg/benchmark_results.json | python -m json.tool
fi
