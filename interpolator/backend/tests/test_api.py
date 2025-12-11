import pytest
import numpy as np
import pickle
import tempfile
import os
from fastapi.testclient import TestClient
from fivedreg.main import app
from fivedreg.api.state import STATE


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def reset_state():
    """Reset global state before each test"""
    STATE.model = None
    STATE.x_scaler = None
    STATE.y_scaler = None
    STATE.last_metrics = None
    yield
    # Clean up after test
    STATE.model = None
    STATE.x_scaler = None
    STATE.y_scaler = None
    STATE.last_metrics = None


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    return X, y


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_redirects(self, client):
        """Test that root redirects to docs"""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/docs" in response.headers["location"]


class TestUploadEndpoint:
    """Test /upload endpoint"""

    def test_upload_valid_pkl_tuple(self, client, sample_dataset):
        """Test uploading valid .pkl file with tuple format"""
        X, y = sample_dataset
        data = (X, y)
        pkl_bytes = pickle.dumps(data)

        files = {"file": ("test.pkl", pkl_bytes, "application/octet-stream")}
        response = client.post("/upload", files=files)

        assert response.status_code == 200
        result = response.json()
        assert "temp_path" in result
        assert "n_samples" in result
        assert result["n_samples"] == 100
        assert os.path.exists(result["temp_path"])

        # Clean up
        os.unlink(result["temp_path"])

    def test_upload_valid_pkl_dict(self, client, sample_dataset):
        """Test uploading valid .pkl file with dict format"""
        X, y = sample_dataset
        data = {"X": X, "y": y}
        pkl_bytes = pickle.dumps(data)

        files = {"file": ("test.pkl", pkl_bytes, "application/octet-stream")}
        response = client.post("/upload", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result["n_samples"] == 100

        # Clean up
        os.unlink(result["temp_path"])

    def test_upload_wrong_extension(self, client):
        """Test error on non-.pkl file"""
        files = {"file": ("test.txt", b"some data", "text/plain")}
        response = client.post("/upload", files=files)

        assert response.status_code == 400
        assert "Expect .pkl file" in response.json()["detail"]

    def test_upload_invalid_data(self, client):
        """Test error on invalid pickle data"""
        # Invalid data: wrong shape
        X = np.random.randn(100, 3)  # Wrong: 3 features instead of 5
        y = np.random.randn(100)
        data = (X, y)
        pkl_bytes = pickle.dumps(data)

        files = {"file": ("test.pkl", pkl_bytes, "application/octet-stream")}
        response = client.post("/upload", files=files)

        assert response.status_code == 400
        assert "Failed to load dataset" in response.json()["detail"]


class TestTrainEndpoint:
    """Test /train endpoint"""

    def test_train_basic(self, client, reset_state, sample_dataset):
        """Test basic training"""
        X, y = sample_dataset

        # First upload the dataset
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        try:
            # Train
            payload = {
                "batch_size": 32,
                "max_epochs": 3,
                "lr": 0.01,
                "weight_decay": 1e-6,
                "patience": 5
            }
            response = client.post(f"/train?temp_path={temp_path}", json=payload)

            assert response.status_code == 200
            result = response.json()
            assert "metrics" in result
            assert "val_mse" in result["metrics"]
            assert "val_r2" in result["metrics"]
            assert "test_mse" in result["metrics"]
            assert "test_r2" in result["metrics"]

            # Check state was updated
            assert STATE.model is not None
            assert STATE.norm_stats is not None
            assert STATE.last_metrics is not None
        finally:
            # Temp file should be deleted by the endpoint
            assert not os.path.exists(temp_path)

    def test_train_with_y_scaling(self, client, reset_state, sample_dataset):
        """Test training succeeds (note: new interpolator always normalizes y internally)"""
        X, y = sample_dataset

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        try:
            payload = {
                "batch_size": 32,
                "max_epochs": 3,
                "lr": 0.01,
                "weight_decay": 1e-6,
                "patience": 5
            }
            response = client.post(f"/train?temp_path={temp_path}", json=payload)

            assert response.status_code == 200
            # Check that norm_stats is set (includes y normalization)
            assert STATE.norm_stats is not None
            assert STATE.norm_stats.y_mean is not None
            assert STATE.norm_stats.y_std is not None
        finally:
            pass  # File already deleted

    def test_train_missing_temp_path(self, client, reset_state):
        """Test error when temp_path is missing"""
        payload = {
            "hidden": [32, 16],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        response = client.post("/train", json=payload)

        assert response.status_code == 422  # Validation error

    def test_train_invalid_temp_path(self, client, reset_state):
        """Test error when temp_path doesn't exist"""
        payload = {
            "hidden": [32],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        response = client.post("/train?temp_path=/nonexistent/file.npz", json=payload)

        assert response.status_code == 400
        assert "Invalid or missing temp_path" in response.json()["detail"]

    def test_train_temp_path_outside_tmpdir(self, client, reset_state):
        """Test error when temp_path is outside temp directory"""
        # Try to use a path outside temp directory
        payload = {
            "hidden": [32],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        response = client.post("/train?temp_path=/etc/passwd", json=payload)

        assert response.status_code == 400
        assert "must be in temp directory" in response.json()["detail"]


class TestPredictEndpoint:
    """Test /predict endpoint"""

    def test_predict_without_model(self, client, reset_state):
        """Test error when no model is trained"""
        payload = {"X": [[1.0, 2.0, 3.0, 4.0, 5.0]]}
        response = client.post("/predict", json=payload)

        assert response.status_code == 400
        assert "No model trained" in response.json()["detail"]

    def test_predict_after_training(self, client, reset_state, sample_dataset):
        """Test prediction after training"""
        X, y = sample_dataset

        # Upload and train
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        train_payload = {
            "hidden": [16],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        client.post(f"/train?temp_path={temp_path}", json=train_payload)

        # Now predict
        predict_payload = {"X": [[1.0, 2.0, 3.0, 4.0, 5.0]]}
        response = client.post("/predict", json=predict_payload)

        assert response.status_code == 200
        result = response.json()
        assert "y" in result
        assert len(result["y"]) == 1
        assert isinstance(result["y"][0], float)

    def test_predict_multiple_samples(self, client, reset_state, sample_dataset):
        """Test prediction with multiple samples"""
        X, y = sample_dataset

        # Upload and train
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        train_payload = {
            "hidden": [16],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        client.post(f"/train?temp_path={temp_path}", json=train_payload)

        # Predict multiple
        predict_payload = {
            "X": [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.5, 1.5, 2.5, 3.5, 4.5],
                [2.0, 3.0, 4.0, 5.0, 6.0]
            ]
        }
        response = client.post("/predict", json=predict_payload)

        assert response.status_code == 200
        result = response.json()
        assert len(result["y"]) == 3

    def test_predict_wrong_feature_count(self, client, reset_state, sample_dataset):
        """Test error with wrong number of features"""
        X, y = sample_dataset

        # Upload and train
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        train_payload = {
            "hidden": [16],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        client.post(f"/train?temp_path={temp_path}", json=train_payload)

        # Predict with wrong number of features
        predict_payload = {"X": [[1.0, 2.0, 3.0]]}  # Only 3 features
        response = client.post("/predict", json=predict_payload)

        assert response.status_code == 422  # Validation error (pydantic)

    @pytest.mark.skip(reason="NaN/Inf cannot be sent via JSON - tested separately in backend validation")
    def test_predict_with_nan(self, client, reset_state, sample_dataset):
        """Test error with NaN values (skipped - JSON doesn't support NaN)"""
        pass

    @pytest.mark.skip(reason="NaN/Inf cannot be sent via JSON - tested separately in backend validation")
    def test_predict_with_inf(self, client, reset_state, sample_dataset):
        """Test error with Inf values (skipped - JSON doesn't support Inf)"""
        pass

    def test_predict_empty_input(self, client, reset_state, sample_dataset):
        """Test error with empty input"""
        X, y = sample_dataset

        # Upload and train
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        train_payload = {
            "hidden": [16],
            "lr": 0.01,
            "max_epochs": 3,
            "batch_size": 32,
            "patience": 5,
            "scale_y": False
        }
        client.post(f"/train?temp_path={temp_path}", json=train_payload)

        # Predict with empty input
        predict_payload = {"X": []}
        response = client.post("/predict", json=predict_payload)

        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]
