import pytest
import numpy as np
import torch
import tempfile
import os
from fivedreg.interpolator import (
    MLP,
    NormStats,
    compute_norm_stats,
    apply_x_norm,
    apply_y_norm,
    invert_y_norm,
    train_model,
    interpolate,
    save_model,
    load_model,
    synthetic_5d,
)


class TestNormStats:
    """Test normalization statistics"""

    def test_compute_norm_stats(self):
        """Test computing normalization statistics"""
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)

        stats = compute_norm_stats(X, y)

        assert stats.x_mean.shape == (5,)
        assert stats.x_std.shape == (5,)
        assert isinstance(stats.y_mean, float)
        assert isinstance(stats.y_std, float)

    def test_apply_x_norm(self):
        """Test applying X normalization"""
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        stats = NormStats(
            x_mean=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            x_std=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            y_mean=0.0,
            y_std=1.0,
        )

        X_norm = apply_x_norm(X, stats)
        assert np.allclose(X_norm, np.zeros((1, 5)))

    def test_y_normalization_inversion(self):
        """Test y normalization and inversion"""
        y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        stats = NormStats(
            x_mean=np.zeros(5, dtype=np.float32),
            x_std=np.ones(5, dtype=np.float32),
            y_mean=20.0,
            y_std=10.0,
        )

        y_norm = apply_y_norm(y, stats)
        y_recovered = invert_y_norm(y_norm, stats)

        assert np.allclose(y_recovered, y, atol=1e-5)


class TestMLP:
    """Test MLP model"""

    def test_initialization(self):
        """Test MLP initialization with default parameters"""
        model = MLP()
        assert isinstance(model, torch.nn.Module)

    def test_custom_architecture(self):
        """Test MLP with custom architecture"""
        model = MLP(in_dim=5, hidden=32, depth=2, out_dim=1)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass"""
        model = MLP()
        X = torch.randn(10, 5)
        y = model(X)
        assert y.shape == (10, 1)


class TestTrainModel:
    """Test train_model function"""

    def test_train_model_small(self):
        """Test training on small dataset"""
        X, y = synthetic_5d(n=200, seed=42)

        model, stats, (val_mse, val_r2, test_mse, test_r2) = train_model(
            X, y, batch_size=32, max_epochs=10, patience=5
        )

        assert isinstance(model, MLP)
        assert isinstance(stats, NormStats)
        assert val_mse > 0
        assert test_mse > 0
        assert -1.0 <= val_r2 <= 1.0  # R² can be negative for poor models
        assert -1.0 <= test_r2 <= 1.0

    def test_train_model_convergence(self):
        """Test that model can fit simple data"""
        # Create very simple linear data
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        model, stats, (val_mse, val_r2, test_mse, test_r2) = train_model(
            X, y, batch_size=64, max_epochs=50, patience=10
        )

        # Should achieve reasonable MSE on this simple problem
        assert val_mse < 1.0
        assert test_mse < 1.0
        # R² should be positive for a model that can fit linear data
        assert val_r2 > 0.5
        assert test_r2 > 0.5


class TestInterpolate:
    """Test interpolation function"""

    def test_interpolate_shape(self):
        """Test interpolation output shape"""
        X_train, y_train = synthetic_5d(n=200, seed=42)
        model, stats, _ = train_model(X_train, y_train, batch_size=32, max_epochs=10)

        X_test = np.random.randn(50, 5).astype(np.float32)
        y_pred = interpolate(model, stats, X_test)

        assert y_pred.shape == (50,)

    def test_interpolate_no_nans(self):
        """Test that interpolation doesn't produce NaN values"""
        X_train, y_train = synthetic_5d(n=200, seed=42)
        model, stats, _ = train_model(X_train, y_train, batch_size=32, max_epochs=10)

        X_test = np.random.randn(50, 5).astype(np.float32)
        y_pred = interpolate(model, stats, X_test)

        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))


class TestSaveLoad:
    """Test model save/load functionality"""

    def test_save_and_load(self):
        """Test saving and loading model"""
        X, y = synthetic_5d(n=200, seed=42)
        model, stats, _ = train_model(X, y, batch_size=32, max_epochs=10)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_model(model, stats, tmp_path)

            # Load and verify
            loaded_model, loaded_stats = load_model(tmp_path)

            # Test predictions match
            X_test = np.random.randn(10, 5).astype(np.float32)
            y_pred_original = interpolate(model, stats, X_test)
            y_pred_loaded = interpolate(loaded_model, loaded_stats, X_test)

            assert np.allclose(y_pred_original, y_pred_loaded, atol=1e-5)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestSyntheticData:
    """Test synthetic data generation"""

    def test_synthetic_5d_shape(self):
        """Test synthetic data generation shape"""
        X, y = synthetic_5d(n=1000, seed=42)

        assert X.shape == (1000, 5)
        assert y.shape == (1000,)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_synthetic_5d_reproducible(self):
        """Test that synthetic data is reproducible with same seed"""
        X1, y1 = synthetic_5d(n=100, seed=123)
        X2, y2 = synthetic_5d(n=100, seed=123)

        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)
