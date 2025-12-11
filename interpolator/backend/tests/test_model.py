"""
Tests for the scikit-learn compatible model wrapper.

Tests that the NeuralNetworkRegressor meets coursework requirements:
1. Configurable layer sizes
2. Standard fit/predict interface
3. Performance requirements
"""

import pytest
import numpy as np
from fivedreg.interpolator import NeuralNetworkRegressor, ConfigurableMLP, synthetic_5d
import torch


class TestConfigurableMLP:
    """Test the configurable MLP architecture"""

    def test_variable_layer_sizes(self):
        """Test that layer sizes can vary (e.g., [64, 32, 16])"""
        model = ConfigurableMLP(
            in_dim=5,
            hidden_layers=[64, 32, 16],
            out_dim=1
        )

        # Check architecture
        assert isinstance(model, torch.nn.Module)

        # Test forward pass
        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 1)

    def test_single_hidden_layer(self):
        """Test with single hidden layer"""
        model = ConfigurableMLP(
            in_dim=5,
            hidden_layers=[32],
            out_dim=1
        )

        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 1)

    def test_multiple_hidden_layers(self):
        """Test with many hidden layers"""
        model = ConfigurableMLP(
            in_dim=5,
            hidden_layers=[128, 64, 32, 16, 8],
            out_dim=1
        )

        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 1)


class TestNeuralNetworkRegressor:
    """Test the sklearn-compatible interface"""

    def test_fit_predict_interface(self):
        """Test standard fit/predict interface"""
        X, y = synthetic_5d(n=200, seed=42)

        model = NeuralNetworkRegressor(
            hidden_layers=[32, 16],
            max_epochs=10,
            verbose=False
        )

        # Test fit returns self
        result = model.fit(X, y)
        assert result is model

        # Test predict works
        X_test = np.random.randn(50, 5).astype(np.float32)
        y_pred = model.predict(X_test)

        assert y_pred.shape == (50,)
        assert not np.any(np.isnan(y_pred))

    def test_configurable_layers(self):
        """Test that layer sizes are configurable"""
        X, y = synthetic_5d(n=200, seed=42)

        # Test with [64, 32, 16]
        model1 = NeuralNetworkRegressor(
            hidden_layers=[64, 32, 16],
            max_epochs=5,
            verbose=False
        )
        model1.fit(X, y)

        # Test with [128, 64]
        model2 = NeuralNetworkRegressor(
            hidden_layers=[128, 64],
            max_epochs=5,
            verbose=False
        )
        model2.fit(X, y)

        # Both should work
        assert model1.hidden_layers == [64, 32, 16]
        assert model2.hidden_layers == [128, 64]

    def test_configurable_hyperparameters(self):
        """Test that all hyperparameters are configurable"""
        model = NeuralNetworkRegressor(
            hidden_layers=[64, 32, 16],
            learning_rate=0.001,
            max_epochs=100,
            batch_size=128,
            patience=15
        )

        assert model.hidden_layers == [64, 32, 16]
        assert model.learning_rate == 0.001
        assert model.max_epochs == 100
        assert model.batch_size == 128
        assert model.patience == 15

    def test_score_method(self):
        """Test R² scoring"""
        X, y = synthetic_5d(n=500, seed=42)

        # Split data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = NeuralNetworkRegressor(
            hidden_layers=[32, 16],
            max_epochs=20,
            verbose=False
        )
        model.fit(X_train, y_train)

        # Test scoring
        r2 = model.score(X_test, y_test)

        assert isinstance(r2, float)
        assert -1.0 <= r2 <= 1.0  # Valid R² range

    def test_error_before_fit(self):
        """Test that predict raises error before fitting"""
        model = NeuralNetworkRegressor()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_wrong_input_dimensions(self):
        """Test error handling for wrong input dimensions"""
        X_train, y_train = synthetic_5d(n=100, seed=42)

        model = NeuralNetworkRegressor(max_epochs=5, verbose=False)
        model.fit(X_train, y_train)

        # Try to predict with wrong number of features
        X_wrong = np.random.randn(10, 3)  # 3 features instead of 5

        with pytest.raises(ValueError, match="Expected 5 features"):
            model.predict(X_wrong)

    def test_convergence_on_simple_data(self):
        """Test that model converges on simple linear data"""
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        model = NeuralNetworkRegressor(
            hidden_layers=[32, 16],
            max_epochs=50,
            verbose=False
        )
        model.fit(X, y)

        # Should achieve reasonable R²
        r2 = model.score(X[:100], y[:100])
        assert r2 > 0.5, f"Model should fit simple linear data (got R²={r2})"

    def test_performance_10k_samples(self):
        """Test performance requirement: < 60s for 10,000 samples"""
        import time

        X, y = synthetic_5d(n=10000, seed=42)

        model = NeuralNetworkRegressor(
            hidden_layers=[64, 32, 16],
            max_epochs=150,
            verbose=False
        )

        start = time.time()
        model.fit(X, y)
        duration = time.time() - start

        # Coursework requirement: < 1 minute
        assert duration < 60, f"Training took {duration:.1f}s (should be < 60s)"

    def test_works_with_5d_dataset(self):
        """Test that model works with 5D datasets"""
        X, y = synthetic_5d(n=1000, seed=42)

        assert X.shape[1] == 5, "Dataset should have 5 features"

        model = NeuralNetworkRegressor(
            hidden_layers=[64, 32, 16],
            max_epochs=20,
            verbose=False
        )

        model.fit(X, y)
        y_pred = model.predict(X[:100])

        assert y_pred.shape == (100,)
        assert not np.any(np.isnan(y_pred))
