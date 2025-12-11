#!/usr/bin/env bash
# Detailed profiling with memory_profiler and cProfile

set -e

echo "========================================="
echo "Detailed Performance Profiling"
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

# Check if profiling tools are installed
if ! python -c "import memory_profiler" 2>/dev/null; then
    echo "⚠️  memory_profiler not installed. Installing..."
    pip install memory_profiler
fi

echo "📊 Running detailed profiling..."
echo ""

# Run with memory profiler
echo "1️⃣  Memory profiling (line-by-line)..."
python -m memory_profiler -m fivedreg.benchmark

echo ""
echo "2️⃣  CPU profiling (cProfile)..."
python -m cProfile -o profile_output.prof -m fivedreg.benchmark

echo ""
echo "3️⃣  Analyzing CPU profile..."
python << 'EOF'
import pstats
from pstats import SortKey

p = pstats.Stats('profile_output.prof')
print("\nTop 20 functions by cumulative time:")
print("=" * 80)
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

print("\nTop 20 functions by time:")
print("=" * 80)
p.sort_stats(SortKey.TIME).print_stats(20)
EOF

echo ""
echo "========================================="
echo "✅ Profiling complete!"
echo "========================================="
echo ""
echo "📄 CPU profile saved to: backend/profile_output.prof"
echo "📊 View with: python -m pstats profile_output.prof"
echo ""
