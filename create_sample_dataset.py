#!/usr/bin/env python3
"""
Create a sample 5D dataset for testing the interpolator.

This script generates a synthetic dataset with:
- 1000 samples
- 5 input features (X)
- 1 target value (y)
- Linear relationship: y = 2*x0 + 3*x1 - 1.5*x2 + 0.8*x3 - 0.3*x4 + noise
"""

import numpy as np
import pickle

def create_sample_dataset(n_samples=1000, seed=42):
    """Generate synthetic 5D regression dataset."""
    np.random.seed(seed)

    # Generate 5 random features
    X = np.random.randn(n_samples, 5).astype(np.float32)

    # Generate target with known relationship plus small noise
    y = (
        2.0 * X[:, 0] +      # Strong positive effect
        3.0 * X[:, 1] +      # Strongest positive effect
        -1.5 * X[:, 2] +     # Strong negative effect
        0.8 * X[:, 3] +      # Moderate positive effect
        -0.3 * X[:, 4] +     # Small negative effect
        0.1 * np.random.randn(n_samples)  # Small noise
    ).astype(np.float32)

    return X, y

if __name__ == "__main__":
    print("Creating sample dataset...")

    # Create dataset
    X, y = create_sample_dataset(n_samples=1000)

    # Save as pickle file (tuple format)
    output_file = "sample_dataset.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump((X, y), f)

    print(f"   Sample dataset created: {output_file}")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}]")
    print("")
    print("You can now upload this file in the web interface!")
