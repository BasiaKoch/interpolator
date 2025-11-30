import pytest
import numpy as np
from fivedreg.utils.train import train_from_arrays
from fivedreg.models.mlp import MLPConfig


class TestTrainFromArrays:
    """Test train_from_arrays utility function"""

    def test_basic_training(self):
        """Test basic training flow"""
        X = np.random.randn(500, 5).astype(np.float32)
        y = np.random.randn(500).astype(np.float32)

        cfg = MLPConfig(hidden=(32, 16), max_epochs=5)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        # Check that model is fitted
        assert model._fitted is True

        # Check that dataset splits exist
        assert ds.X_train.shape[1] == 5
        assert ds.X_val.shape[1] == 5
        assert ds.X_test.shape[1] == 5

        # Check that metrics are returned
        assert 'val_mse' in metrics
        assert 'val_r2' in metrics
        assert isinstance(metrics['val_mse'], float)
        assert isinstance(metrics['val_r2'], float)

    def test_returns_correct_types(self):
        """Test that function returns correct types"""
        X = np.random.randn(300, 5).astype(np.float32)
        y = np.random.randn(300).astype(np.float32)

        cfg = MLPConfig(max_epochs=3)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        # Check model type
        from fivedreg.models.mlp import MLPRegressor
        assert isinstance(model, MLPRegressor)

        # Check dataset type
        from fivedreg.data.loader import DatasetSplits
        assert isinstance(ds, DatasetSplits)

        # Check metrics type
        assert isinstance(metrics, dict)

    def test_with_y_scaling(self):
        """Test training with y scaling enabled"""
        X = np.random.randn(400, 5).astype(np.float32)
        y = np.random.randn(400).astype(np.float32) * 100 + 50  # Large scale

        cfg = MLPConfig(hidden=(32,), max_epochs=5)
        model, ds, metrics = train_from_arrays(X, y, cfg, scale_y=True)

        # Check that y_scaler exists
        assert ds.y_scaler is not None

        # Check that metrics are computed
        assert 'val_mse' in metrics
        assert 'val_r2' in metrics

    def test_without_y_scaling(self):
        """Test training without y scaling"""
        X = np.random.randn(400, 5).astype(np.float32)
        y = np.random.randn(400).astype(np.float32)

        cfg = MLPConfig(hidden=(32,), max_epochs=5)
        model, ds, metrics = train_from_arrays(X, y, cfg, scale_y=False)

        # Check that y_scaler is None
        assert ds.y_scaler is None

    def test_metrics_validity(self):
        """Test that metrics have valid values"""
        # Generate simple linear data
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (2 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(500)).astype(np.float32)

        cfg = MLPConfig(hidden=(64, 32), max_epochs=30, lr=0.01)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        # MSE should be positive
        assert metrics['val_mse'] >= 0

        # R2 should be between -inf and 1 (typically between 0 and 1 for good models)
        assert metrics['val_r2'] <= 1.0

    def test_model_can_predict(self):
        """Test that trained model can make predictions"""
        X = np.random.randn(300, 5).astype(np.float32)
        y = np.random.randn(300).astype(np.float32)

        cfg = MLPConfig(hidden=(32,), max_epochs=5)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        # Model should be able to predict on validation data
        predictions = model.predict(ds.X_val)
        assert predictions.shape == ds.y_val.shape

    def test_different_configs(self):
        """Test training with different configurations"""
        X = np.random.randn(400, 5).astype(np.float32)
        y = np.random.randn(400).astype(np.float32)

        configs = [
            MLPConfig(hidden=(64,), max_epochs=3),
            MLPConfig(hidden=(32, 16), max_epochs=5, lr=0.01),
            MLPConfig(hidden=(128, 64, 32), max_epochs=5, batch_size=128),
        ]

        for cfg in configs:
            model, ds, metrics = train_from_arrays(X, y, cfg)
            assert model._fitted is True
            assert 'val_mse' in metrics
            assert 'val_r2' in metrics

    def test_small_dataset(self):
        """Test training with small dataset"""
        # Very small dataset (but enough for train/val/test split)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        cfg = MLPConfig(hidden=(16,), max_epochs=3, batch_size=8)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        assert model._fitted is True
        assert ds.X_train.shape[0] > 0
        assert ds.X_val.shape[0] > 0
        assert ds.X_test.shape[0] > 0

    def test_dataset_splits_sum_to_total(self):
        """Test that train/val/test splits sum to original size"""
        X = np.random.randn(1000, 5).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)

        cfg = MLPConfig(max_epochs=3)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        total = ds.X_train.shape[0] + ds.X_val.shape[0] + ds.X_test.shape[0]
        assert total == 1000

    def test_x_scaler_exists(self):
        """Test that x_scaler is created"""
        X = np.random.randn(300, 5).astype(np.float32)
        y = np.random.randn(300).astype(np.float32)

        cfg = MLPConfig(max_epochs=3)
        model, ds, metrics = train_from_arrays(X, y, cfg)

        assert ds.x_scaler is not None
        from sklearn.preprocessing import StandardScaler
        assert isinstance(ds.x_scaler, StandardScaler)
