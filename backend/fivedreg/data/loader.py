from __future__ import annotations
import pickle
import numpy as np
from typing import Tuple

def load_dataset_pkl(path_or_bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from .pkl file. Handles multiple formats:
    - Tuple: (X, y)
    - Dictionary: {'X': ..., 'y': ...} or {'data': ..., 'target': ...}
    - Tuple with extra data: (X, y, ...) - uses first two elements
    """
    # Load the pickle file
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = pickle.loads(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            data = pickle.load(f)

    # Try to extract X and y from various formats
    X, y = None, None

    # Format 1: Dictionary with 'X' and 'y' keys
    if isinstance(data, dict):
        if 'X' in data and 'y' in data:
            X, y = data['X'], data['y']
        elif 'data' in data and 'target' in data:
            X, y = data['data'], data['target']
        else:
            raise ValueError(
                f"Dictionary must contain keys ('X', 'y') or ('data', 'target'). "
                f"Found keys: {list(data.keys())}"
            )

    # Format 2: Tuple or list (X, y) or (X, y, ...)
    elif isinstance(data, (tuple, list)):
        if len(data) < 2:
            raise ValueError(f"Tuple/list must contain at least 2 elements (X, y), got {len(data)}")
        X, y = data[0], data[1]

    # Format 3: Other formats
    else:
        raise ValueError(
            f"Unsupported pickle format. Expected dict or tuple, got {type(data).__name__}. "
            f"Pickle file should contain either (X, y) tuple or {{'X': ..., 'y': ...}} dict."
        )

    return _validate(X, y)

def _validate(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"Expected X with shape (n,5); got {X.shape}")
    if len(y) != len(X):
        raise ValueError("X and y must have same length")
    # Basic missing-value handling: drop rows with NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if mask.sum() < len(X):
        X, y = X[mask], y[mask]
    return X, y

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to load a 5D dataset from .pkl format.

    Reads X (5 features) and y (target) arrays, validates input dimensions,
    and handles missing values appropriately.

    """
    return load_dataset_pkl(filepath)

