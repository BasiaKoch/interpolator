#!/usr/bin/env python3
"""
Performance Benchmark for C1 Coursework.
Measures:
- Training time (1K, 5K, 10K samples)
- Memory usage during training & prediction
- Accuracy metrics (MSE, R²)
Outputs results as JSON for documentation.
"""

import json
import time
import psutil
import os
import gc
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from fivedreg.interpolator import train_model, interpolate


# ---------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------

def mem_mb():
    """Return current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6


def make_dataset(n, seed=42):
    """Generate synthetic 5D regression dataset."""
    np.random.seed(seed)
    X = np.random.randn(n, 5).astype(np.float32)
    y = (
        2 * X[:, 0] + 0.5 * X[:, 1] - 1.5 * X[:, 2] +
        0.8 * X[:, 3] - 0.3 * X[:, 4] +
        0.1 * np.random.randn(n)
    ).astype(np.float32)
    return X, y


# ---------------------------------------------------------
# Core Benchmark
# ---------------------------------------------------------

def run_single(n_samples):
    """Run performance benchmark on dataset of size n_samples."""
    print(f"\n=== Benchmark: {n_samples:,} samples ===")

    # Generate data
    X, y = make_dataset(n_samples)

    # --- Train ---
    gc.collect()  # Force garbage collection for clean memory measurement
    mem_before = mem_mb()
    t0 = time.time()
    model, norm_stats, (val_mse, test_mse) = train_model(
        X, y,
        batch_size=256,
        max_epochs=200,
        lr=1e-3,
        weight_decay=1e-6,
        patience=15,
    )
    train_time = time.time() - t0
    gc.collect()  # Force garbage collection before measuring final memory
    mem_after = mem_mb()

    mem_train = max(0, mem_after - mem_before)  # Ensure non-negative

    # --- Predict on a sample of test data ---
    # Generate fresh test data
    X_test, y_test = make_dataset(int(n_samples * 0.15), seed=99)

    gc.collect()  # Clean memory before prediction measurement
    mem_before_pred = mem_mb()
    y_test_pred = interpolate(model, norm_stats, X_test)
    gc.collect()
    mem_after_pred = mem_mb()

    mem_pred = max(0, mem_after_pred - mem_before_pred)  # Ensure non-negative

    # --- Metrics ---
    test_mse_external = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "samples": n_samples,
        "train_time_sec": train_time,
        "memory_train_mb": mem_train,
        "memory_pred_mb": mem_pred,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mse_external": test_mse_external,
        "test_r2": test_r2
    }


# ---------------------------------------------------------
# Run All Benchmarks
# ---------------------------------------------------------

def main():
    print("\n========================================")
    print("PERFORMANCE BENCHMARK")
    print("========================================")

    dataset_sizes = [1000, 5000, 10000]
    results = []

    for n in dataset_sizes:
        r = run_single(n)
        results.append(r)

    # Write results to JSON for Sphinx
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:", out_path)
    print("\nDone.\n")


if __name__ == "__main__":
    main()

