#!/usr/bin/env python3
"""
Script to inspect the saved model pickle file.
Usage: python inspect_model.py
"""
import pickle
import sys
from pathlib import Path

model_path = Path(__file__).parent / "artifacts" / "latest.pkl"

if not model_path.exists():
    print(f"❌ Model file not found: {model_path}")
    print("   Run training first to generate latest.pkl")
    sys.exit(1)

with open(model_path, "rb") as f:
    payload = pickle.load(f)

print("=" * 60)
print("📦 Model Pickle Contents")
print("=" * 60)
print()

print("Keys in pickle file:")
for key in payload.keys():
    print(f"  - {key}")

print("\n" + "=" * 60)
print("🔧 Model Metadata")
print("=" * 60)
if "meta" in payload:
    for k, v in payload["meta"].items():
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("📊 Normalization Stats")
print("=" * 60)
if "norm_stats" in payload:
    stats = payload["norm_stats"]
    print(f"  x_mean: {stats['x_mean']}")
    print(f"  x_std: {stats['x_std']}")
    print(f"  y_mean: {stats['y_mean']}")
    print(f"  y_std: {stats['y_std']}")

print("\n" + "=" * 60)
print("ℹ️  Model Info")
print("=" * 60)
if "model_info" in payload:
    for k, v in payload["model_info"].items():
        print(f"  {k}: {v}")

print()
