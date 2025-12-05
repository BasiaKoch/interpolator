#!/usr/bin/env bash
# Run PyTorch profiling with visualization

set -e

echo "========================================="
echo "PyTorch Training Profiler"
echo "========================================="
echo ""

# Get project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/backend"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Check if matplotlib is installed
if ! python -c "import matplotlib" 2>/dev/null; then
    echo "⚠️  matplotlib not installed. Installing benchmark dependencies..."
    pip install -e ".[bench]"
fi

echo "📊 Step 1: Running PyTorch Profiler..."
echo ""
python -m fivedreg.profile_training

echo ""
echo "📈 Step 2: Generating visualization plots..."
echo ""
python -m fivedreg.plot_profiling

echo ""
echo "========================================="
echo "✅ Profiling complete!"
echo "========================================="
echo ""
echo "📄 Results saved to:"
echo "   • backend/fivedreg/profiling_results.json"
echo ""
echo "📊 Plots saved to:"
echo "   • backend/fivedreg/profile_operation_times.png"
echo "   • backend/fivedreg/profile_time_breakdown.png"
echo "   • backend/fivedreg/profile_flops.png"
echo "   • backend/fivedreg/profile_bottlenecks.png"
echo ""
echo "📊 TensorBoard trace:"
echo "   View with: tensorboard --logdir=backend/fivedreg/profiler_trace"
echo "   Then open http://localhost:6006"
echo ""
