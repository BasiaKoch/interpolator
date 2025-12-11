#!/usr/bin/env python3
"""
Generate visualization plots from profiling results.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_profiling_results():
    """Load profiling results from JSON."""
    path = os.path.join(os.path.dirname(__file__), "profiling_results.json")
    if not os.path.exists(path):
        print(f"Profiling results not found at {path}")
        print("   Run 'python -m fivedreg.profile_training' first!")
        return None

    with open(path) as f:
        return json.load(f)


def plot_operation_times(stats, output_path):
    """Plot CPU and CUDA time for top operations."""
    operations = stats["operations"][:8]  # Top 8 operations

    op_names = [op["name"][:30] for op in operations]  # Truncate long names
    cpu_times = [op["cpu_time_ms"] for op in operations]
    cuda_times = [op["cuda_time_ms"] for op in operations]

    x = np.arange(len(op_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.barh(x - width/2, cpu_times, width, label='CPU Time', color='steelblue')
    bars2 = ax.barh(x + width/2, cuda_times, width, label='CUDA Time', color='orange')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)
    ax.set_title('Profiling: Time per Operation', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(op_names, fontsize=9)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved operation times plot: {output_path}")
    plt.close()


def plot_time_breakdown(stats, output_path):
    """Plot pie chart of time breakdown."""
    operations = stats["operations"][:6]  # Top 6

    labels = [op["name"][:25] for op in operations]
    times = [op["cpu_time_ms"] for op in operations]

    # Add "Other" category for remaining time
    total_top = sum(times)
    total_all = stats["total_cpu_time_ms"]
    other_time = max(0, total_all - total_top)

    if other_time > 0:
        labels.append("Other")
        times.append(other_time)

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        times,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('CPU Time Breakdown by Operation', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved time breakdown plot: {output_path}")
    plt.close()


def plot_flops_comparison(stats, output_path):
    """Plot FLOPs for operations."""
    operations = [op for op in stats["operations"] if op["flops"] > 0][:8]

    if not operations:
        print(" No FLOPs data available, skipping FLOPs plot")
        return

    op_names = [op["name"][:30] for op in operations]
    flops = [op["flops"] / 1e6 for op in operations]  # Convert to MFLOPs

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(op_names, flops, color='green', alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, flops):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}M',
                ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('FLOPs (Millions)', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)
    ax.set_title('Floating Point Operations (FLOPs) by Operation', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved FLOPs plot: {output_path}")
    plt.close()


def plot_bottleneck_summary(stats, bottlenecks, output_path):
    """Create summary visualization of bottlenecks."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top operations by time
    ops = stats["operations"][:5]
    names = [op["name"][:20] for op in ops]
    times = [op["cpu_time_ms"] for op in ops]

    ax1.barh(names, times, color='steelblue')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Top 5 Operations by Time', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 2. CPU vs CUDA time
    total_cpu = stats["total_cpu_time_ms"]
    total_cuda = stats["total_cuda_time_ms"]

    if total_cuda > 0:
        ax2.bar(['CPU', 'CUDA'], [total_cpu, total_cuda], color=['steelblue', 'orange'])
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('CPU vs CUDA Total Time', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'CUDA not available\n(CPU-only execution)',
                ha='center', va='center', fontsize=12)
        ax2.set_title('CPU vs CUDA Total Time', fontweight='bold')
        ax2.axis('off')

    # 3. Bottleneck categories
    slow_ops = bottlenecks.get("slow_operations", [])
    if slow_ops:
        bottleneck_names = [op["name"][:20] for op in slow_ops]
        bottleneck_pcts = [op["percentage"] for op in slow_ops]

        ax3.barh(bottleneck_names, bottleneck_pcts, color='red', alpha=0.7)
        ax3.set_xlabel('% of Total Time')
        ax3.set_title('Identified Bottlenecks (>10% time)', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10% threshold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '✓ No bottlenecks detected\n(all ops <10% of time)',
                ha='center', va='center', fontsize=12, color='green', fontweight='bold')
        ax3.set_title('Identified Bottlenecks', fontweight='bold')
        ax3.axis('off')

    # 4. Recommendations summary
    recommendations = bottlenecks.get("recommendations", [])
    ax4.axis('off')
    ax4.set_title('Optimization Recommendations', fontweight='bold', loc='left')

    if recommendations:
        rec_text = "\n\n".join([f"• {rec}" for rec in recommendations])
        ax4.text(0.05, 0.9, rec_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, '✓ No optimizations needed\nTraining is efficient',
                ha='center', va='center', fontsize=12, color='green',
                fontweight='bold', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved bottleneck summary: {output_path}")
    plt.close()


def main():
    """Generate all profiling plots."""
    print("\n" + "="*60)
    print("GENERATING PROFILING VISUALIZATIONS")
    print("="*60 + "\n")

    results = load_profiling_results()
    if not results:
        return

    stats = results["profiling_stats"]
    bottlenecks = results["bottleneck_analysis"]

    output_dir = os.path.dirname(__file__)

    # Generate plots
    plot_operation_times(
        stats,
        os.path.join(output_dir, "profile_operation_times.png")
    )

    plot_time_breakdown(
        stats,
        os.path.join(output_dir, "profile_time_breakdown.png")
    )

    plot_flops_comparison(
        stats,
        os.path.join(output_dir, "profile_flops.png")
    )

    plot_bottleneck_summary(
        stats,
        bottlenecks,
        os.path.join(output_dir, "profile_bottlenecks.png")
    )

    print("\n" + "="*60)
    print(" ALL PLOTS GENERATED")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}/")
    print("Files:")
    print("  • profile_operation_times.png")
    print("  • profile_time_breakdown.png")
    print("  • profile_flops.png")
    print("  • profile_bottlenecks.png")
    print("\n")


if __name__ == "__main__":
    main()
