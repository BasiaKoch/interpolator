"""
fivedreg: 5D Dataset Interpolator

A Python package for training neural network models on 5-dimensional datasets
and making predictions.
"""

__version__ = "0.1.0"
__author__ = "Barbara Koch"
__email__ = "bk489@cam.ac.uk"

# Core functionality
from .interpolator import (
    train_model,
    interpolate,
    save_model,
    load_model,
    synthetic_5d,
    MLP,
    NormStats,
)

# Data loading
from .data.loader import load_dataset_pkl, load_dataset

__all__ = [
    "train_model",
    "interpolate",
    "save_model",
    "load_model",
    "synthetic_5d",
    "MLP",
    "NormStats",
    "load_dataset_pkl",
    "load_dataset",
]
