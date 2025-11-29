from __future__ import annotations
import pickle, io
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class DatasetSplits:
    X_train: np.ndarray; y_train: np.ndarray
    X_val:   np.ndarray; y_val:   np.ndarray
    X_test:  np.ndarray; y_test:  np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler | None = None  # keep y unscaled if desired

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

def load_dataset_npz(path_or_bytes) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = np.load(io.BytesIO(path_or_bytes))
    else:
        data = np.load(path_or_bytes)
    X, y = data["X"], data["y"]
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

def split_and_standardize(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42,
    scale_y: bool = False
) -> DatasetSplits:
    """
    Split data into train/validation/test sets and standardize features.

    """
    # First split: separate training data from temp (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        random_state=random_state,
        shuffle=True
    )

    # Second split: separate validation from test
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - rel_val,
        random_state=random_state,
        shuffle=True
    )

    # Standardize features (fit on training data only)
    x_scaler = StandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    # Optionally standardize targets
    y_scaler = None
    if scale_y:
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler)

