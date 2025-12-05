#!/usr/bin/env python3
"""
Advanced PyTorch Profiling for Training Loop
Tracks: time per operation, FLOPs, memory usage, bottlenecks
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, List, Any
import torch.profiler as profiler

from fivedreg.interpolator import MLP, compute_norm_stats, apply_x_norm, apply_y_norm


def make_synthetic_dataset(n: int = 5000, seed: int = 42):
    """Generate synthetic 5D dataset for profiling."""
    np.random.seed(seed)
    X = np.random.randn(n, 5).astype(np.float32)
    y = (
        2 * X[:, 0] + 0.5 * X[:, 1] - 1.5 * X[:, 2] +
        0.8 * X[:, 3] - 0.3 * X[:, 4] +
        0.1 * np.random.randn(n)
    ).astype(np.float32)
    return X, y


def profile_training_loop(n_samples: int = 5000, num_epochs: int = 10):
    """
    Profile the training loop with PyTorch Profiler.

    Args:
        n_samples: Dataset size
        num_epochs: Number of epochs to profile

    Returns:
        dict: Profiling statistics
    """
    print(f"\n{'='*60}")
    print(f"PROFILING TRAINING LOOP ({n_samples:,} samples, {num_epochs} epochs)")
    print(f"{'='*60}\n")

    # Generate data
    X, y = make_synthetic_dataset(n_samples)

    # Normalize
    stats = compute_norm_stats(X, y)
    Xn = apply_x_norm(X, stats).astype(np.float32)
    yn = apply_y_norm(y, stats).astype(np.float32)

    Xn_t = torch.from_numpy(Xn)
    yn_t = torch.from_numpy(yn).unsqueeze(1)

    ds = TensorDataset(Xn_t, yn_t)
    n_total = len(ds)
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(123)
    )

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # Directory for profiling outputs
    output_dir = os.path.dirname(__file__)
    trace_path = os.path.join(output_dir, "profiler_trace")

    print(f"\nStarting profiling for {num_epochs} epochs...")
    print(f"Trace will be saved to: {trace_path}\n")

    # PyTorch Profiler configuration
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
        ],
        schedule=profiler.schedule(
            wait=0,      # Skip first step
            warmup=1,    # Warmup for 1 step
            active=3,    # Profile next 3 steps
            repeat=1     # Repeat once
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,  # Record FLOPs
    ) as prof:

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0

            for step, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)

                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                train_loss += loss.item() * xb.size(0)

                # Step profiler
                prof.step()

                # Only profile first few batches
                if step >= 5:
                    break

            # Validation (not profiled in detail)
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss += nn.functional.mse_loss(pred, yb, reduction="sum").item()
                    n += xb.size(0)
            val_loss /= n

            print(f"Epoch {epoch:2d}/{num_epochs} | Train Loss: {train_loss/n_train:.6f} | Val Loss: {val_loss:.6f}")

    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}\n")

    # Extract profiling statistics
    prof_stats = extract_profiler_stats(prof)

    return prof_stats, trace_path


def extract_profiler_stats(prof: profiler.profile) -> Dict[str, Any]:
    """Extract key statistics from profiler."""

    print("\n" + "="*60)
    print("KEY PERFORMANCE METRICS")
    print("="*60)

    # Get profiler key averages
    key_averages = prof.key_averages()

    # Extract key operations
    stats = {
        "operations": [],
        "total_cpu_time_ms": 0,
        "total_cuda_time_ms": 0,
        "total_flops": 0,
        "memory_stats": {}
    }

    # Top operations by CPU time
    print("\n📊 Top 10 Operations by CPU Time:")
    print(f"{'Operation':<40} {'CPU Time (ms)':<15} {'CUDA Time (ms)':<15} {'FLOPs':<15}")
    print("-" * 85)

    sorted_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]

    for op in sorted_ops:
        cpu_time = op.cpu_time_total / 1000  # Convert to ms
        cuda_time = op.cuda_time_total / 1000 if hasattr(op, 'cuda_time_total') else 0
        flops = op.flops if hasattr(op, 'flops') else 0

        stats["operations"].append({
            "name": op.key,
            "cpu_time_ms": cpu_time,
            "cuda_time_ms": cuda_time,
            "flops": flops,
            "calls": op.count
        })

        stats["total_cpu_time_ms"] += cpu_time
        stats["total_cuda_time_ms"] += cuda_time
        stats["total_flops"] += flops

        # Format FLOPs for display
        flops_str = f"{flops:,.0f}" if flops > 0 else "N/A"

        print(f"{op.key[:40]:<40} {cpu_time:<15.2f} {cuda_time:<15.2f} {flops_str:<15}")

    print("\n" + "="*60)
    print(f"Total CPU Time: {stats['total_cpu_time_ms']:.2f} ms")
    print(f"Total CUDA Time: {stats['total_cuda_time_ms']:.2f} ms")
    print(f"Total FLOPs: {stats['total_flops']:,.0f}")
    print("="*60)

    # Memory statistics (if available)
    try:
        memory_profile = prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=5
        )
        print(f"\n📊 Memory Usage (Top 5):\n{memory_profile}")
    except:
        print("\n⚠️  Memory profiling not available")

    return stats


def analyze_bottlenecks(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze profiling data to identify bottlenecks."""

    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60 + "\n")

    bottlenecks = {
        "slow_operations": [],
        "memory_intensive": [],
        "recommendations": []
    }

    # Find slow operations (>10% of total time)
    total_time = stats["total_cpu_time_ms"]

    for op in stats["operations"]:
        percentage = (op["cpu_time_ms"] / total_time) * 100 if total_time > 0 else 0
        if percentage > 10:
            bottlenecks["slow_operations"].append({
                "name": op["name"],
                "time_ms": op["cpu_time_ms"],
                "percentage": percentage
            })

    # Provide recommendations
    if bottlenecks["slow_operations"]:
        print("⚠️  Slow Operations (>10% of total time):")
        for op in bottlenecks["slow_operations"]:
            print(f"   • {op['name']}: {op['time_ms']:.2f}ms ({op['percentage']:.1f}%)")
        bottlenecks["recommendations"].append(
            "Consider optimizing slow operations or using GPU acceleration"
        )
    else:
        print("✓ No major bottlenecks detected (all operations <10% of total time)")
        bottlenecks["recommendations"].append(
            "Training is well-balanced across operations"
        )

    # Check dataloader efficiency
    dataloader_ops = [op for op in stats["operations"] if "DataLoader" in op["name"]]
    if dataloader_ops and dataloader_ops[0]["cpu_time_ms"] / total_time > 0.2:
        print("\n⚠️  DataLoader appears slow (>20% of time)")
        bottlenecks["recommendations"].append(
            "Consider increasing num_workers in DataLoader or using pin_memory=True"
        )

    # Check GPU utilization
    if stats["total_cuda_time_ms"] > 0:
        gpu_util = stats["total_cuda_time_ms"] / total_time
        print(f"\n📊 GPU Utilization: {gpu_util*100:.1f}%")
        if gpu_util < 0.3:
            print("   ⚠️  Low GPU utilization - GPU may be idle")
            bottlenecks["recommendations"].append(
                "Increase batch size or model complexity to improve GPU utilization"
            )

    if bottlenecks["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in bottlenecks["recommendations"]:
            print(f"   • {rec}")

    print("\n" + "="*60)

    return bottlenecks


def main():
    """Run profiling and save results."""

    # Profile training
    stats, trace_path = profile_training_loop(n_samples=5000, num_epochs=5)

    # Analyze bottlenecks
    bottlenecks = analyze_bottlenecks(stats)

    # Combine results
    results = {
        "profiling_stats": stats,
        "bottleneck_analysis": bottlenecks,
        "trace_location": trace_path
    }

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "profiling_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Profiling results saved to: {output_path}")
    print(f"📊 TensorBoard trace saved to: {trace_path}")
    print("\nTo view the trace in TensorBoard:")
    print(f"   tensorboard --logdir={trace_path}")
    print("   Then open http://localhost:6006 in your browser\n")


if __name__ == "__main__":
    main()
