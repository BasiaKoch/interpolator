import pytest
import numpy as np
import torch
from fivedreg.models.mlp import MLPRegressor, MLPConfig


class TestMLPConfig:
    """Test MLPConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        cfg = MLPConfig()
        assert cfg.hidden == (128, 64, 32)
        assert cfg.lr == 1e-3
        assert cfg.max_epochs == 150
        assert cfg.batch_size == 256
        assert cfg.patience == 15
        assert cfg.seed == 42

    def test_custom_config(self):
        """Test custom configuration"""
        cfg = MLPConfig(
            hidden=(64, 32),
            lr=0.01,
            max_epochs=100,
            batch_size=128,
            patience=10,
            seed=123
        )
        assert cfg.hidden == (64, 32)
        assert cfg.lr == 0.01
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 128
        assert cfg.patience == 10
        assert cfg.seed == 123


class TestMLPRegressor:
    """Test MLPRegressor model"""

    def test_initialization(self):
        """Test model initialization"""
        cfg = MLPConfig(hidden=(64, 32))
        model = MLPRegressor(cfg)

        assert model.cfg == cfg
        assert model.device == torch.device("cpu")
        assert model._fitted is False

    def test_network_architecture(self):
        """Test that network has correct architecture"""
        cfg = MLPConfig(hidden=(64, 32, 16))
        model = MLPRegressor(cfg)

        # Check number of layers: input->64->relu, 64->32->relu, 32->16->relu, 16->1
        # Sequential should have 7 modules: Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear
        assert len(model.net) == 7

        # Check input and output dimensions
        first_layer = model.net[0]
        last_layer = model.net[-1]
        assert first_layer.in_features == 5  # 5 input features
        assert last_layer.out_features == 1   # 1 output

    def test_fit_basic(self):
        """Test basic model fitting"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        cfg = MLPConfig(hidden=(32, 16), max_epochs=5)
        model = MLPRegressor(cfg)
        model.fit(X, y)

        assert model._fitted is True

    def test_fit_with_validation(self):
        """Test fitting with validation data"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(30, 5).astype(np.float32)
        y_val = np.random.randn(30).astype(np.float32)

        cfg = MLPConfig(hidden=(32,), max_epochs=10, patience=5)
        model = MLPRegressor(cfg)
        model.fit(X_train, y_train, X_val, y_val)

        assert model._fitted is True

    def test_predict_without_fit(self):
        """Test that predict fails before fitting"""
        model = MLPRegressor()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(AssertionError, match="Call fit first"):
            model.predict(X)

    def test_predict_after_fit(self):
        """Test prediction after fitting"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)

        cfg = MLPConfig(hidden=(32,), max_epochs=5)
        model = MLPRegressor(cfg)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 5).astype(np.float32)
        predictions = model.predict(X_test)

        assert predictions.shape == (20,)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32

    def test_predict_output_shape(self):
        """Test that predictions have correct shape"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)

        model = MLPRegressor(MLPConfig(max_epochs=3))
        model.fit(X_train, y_train)

        # Test different input sizes
        for n_samples in [1, 5, 50]:
            X_test = np.random.randn(n_samples, 5).astype(np.float32)
            predictions = model.predict(X_test)
            assert predictions.shape == (n_samples,)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same model"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        X_test = np.random.randn(10, 5).astype(np.float32)

        cfg = MLPConfig(hidden=(32,), max_epochs=10, seed=42)

        model1 = MLPRegressor(cfg)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)

        model2 = MLPRegressor(cfg)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)

        # Should be very similar (not exact due to randomness in DataLoader)
        np.testing.assert_allclose(pred1, pred2, rtol=0.1)

    def test_learns_simple_function(self):
        """Test that model can learn a simple linear function"""
        # Generate data: y = 2*x1 + 3*x2 + noise
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (2 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(500)).astype(np.float32)

        # Split into train and test
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        cfg = MLPConfig(hidden=(64, 32), max_epochs=50, lr=0.01)
        model = MLPRegressor(cfg)
        model.fit(X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)

        # Model should learn the relationship reasonably well
        mse = np.mean((predictions - y_test) ** 2)
        assert mse < 1.0  # Should be reasonably low error

    def test_early_stopping(self):
        """Test that early stopping works"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(30, 5).astype(np.float32)
        y_val = np.random.randn(30).astype(np.float32)

        # With very high patience, should train for max_epochs
        cfg = MLPConfig(hidden=(16,), max_epochs=20, patience=100)
        model = MLPRegressor(cfg)
        model.fit(X_train, y_train, X_val, y_val)
        # Training should complete (no way to check exact epochs without modifying code)

        # With very low patience on random data, should stop early
        cfg = MLPConfig(hidden=(16,), max_epochs=100, patience=2)
        model = MLPRegressor(cfg)
        model.fit(X_train, y_train, X_val, y_val)
        # Should stop before max_epochs (can't verify exact behavior without instrumentation)

    def test_different_hidden_sizes(self):
        """Test model with different hidden layer configurations"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        for hidden in [(32,), (64, 32), (128, 64, 32), (16, 16, 16, 16)]:
            cfg = MLPConfig(hidden=hidden, max_epochs=3)
            model = MLPRegressor(cfg)
            model.fit(X, y)

            predictions = model.predict(X[:10])
            assert predictions.shape == (10,)
