import pytest
import numpy as np
import pickle
import tempfile
import os
from fivedreg.data.loader import (
    load_dataset_pkl,
    load_dataset_npz,
    split_and_standardize,
    _validate
)


class TestLoadDatasetPkl:
    """Test load_dataset_pkl with various input formats"""

    def test_load_tuple_format(self):
        """Test loading (X, y) tuple format"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = (X, y)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            temp_path = f.name

        try:
            X_loaded, y_loaded = load_dataset_pkl(temp_path)
            assert X_loaded.shape == (100, 5)
            assert y_loaded.shape == (100,)
            np.testing.assert_array_almost_equal(X_loaded, X)
            np.testing.assert_array_almost_equal(y_loaded, y)
        finally:
            os.unlink(temp_path)

    def test_load_dict_format_xy(self):
        """Test loading {'X': ..., 'y': ...} dict format"""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        data = {'X': X, 'y': y}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            temp_path = f.name

        try:
            X_loaded, y_loaded = load_dataset_pkl(temp_path)
            assert X_loaded.shape == (50, 5)
            assert y_loaded.shape == (50,)
        finally:
            os.unlink(temp_path)

    def test_load_dict_format_data_target(self):
        """Test loading {'data': ..., 'target': ...} dict format"""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        data = {'data': X, 'target': y}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            temp_path = f.name

        try:
            X_loaded, y_loaded = load_dataset_pkl(temp_path)
            assert X_loaded.shape == (50, 5)
            assert y_loaded.shape == (50,)
        finally:
            os.unlink(temp_path)

    def test_load_bytes(self):
        """Test loading from bytes"""
        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        data = (X, y)
        data_bytes = pickle.dumps(data)

        X_loaded, y_loaded = load_dataset_pkl(data_bytes)
        assert X_loaded.shape == (30, 5)
        assert y_loaded.shape == (30,)

    def test_load_tuple_with_extra_elements(self):
        """Test loading (X, y, extra) tuple - should use first two"""
        X = np.random.randn(40, 5).astype(np.float32)
        y = np.random.randn(40).astype(np.float32)
        data = (X, y, "extra_data")

        data_bytes = pickle.dumps(data)
        X_loaded, y_loaded = load_dataset_pkl(data_bytes)
        assert X_loaded.shape == (40, 5)
        assert y_loaded.shape == (40,)

    def test_invalid_dict_keys(self):
        """Test error on dict with wrong keys"""
        data = {'wrong': np.random.randn(10, 5), 'keys': np.random.randn(10)}
        data_bytes = pickle.dumps(data)

        with pytest.raises(ValueError, match="Dictionary must contain keys"):
            load_dataset_pkl(data_bytes)

    def test_invalid_tuple_length(self):
        """Test error on tuple with only 1 element"""
        data = (np.random.randn(10, 5),)
        data_bytes = pickle.dumps(data)

        with pytest.raises(ValueError, match="must contain at least 2 elements"):
            load_dataset_pkl(data_bytes)

    def test_invalid_format(self):
        """Test error on unsupported format"""
        data = "invalid string data"
        data_bytes = pickle.dumps(data)

        with pytest.raises(ValueError, match="Unsupported pickle format"):
            load_dataset_pkl(data_bytes)


class TestValidate:
    """Test the _validate function"""

    def test_valid_data(self):
        """Test validation with valid data"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        X_val, y_val = _validate(X, y)
        assert X_val.shape == (100, 5)
        assert y_val.shape == (100,)
        assert X_val.dtype == np.float32
        assert y_val.dtype == np.float32

    def test_wrong_x_dimensions(self):
        """Test error on wrong X dimensions"""
        X = np.random.randn(100, 3)  # Wrong: 3 features instead of 5
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="Expected X with shape"):
            _validate(X, y)

    def test_mismatched_lengths(self):
        """Test error on mismatched X and y lengths"""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)  # Wrong: different length

        with pytest.raises(ValueError, match="must have same length"):
            _validate(X, y)

    def test_removes_nan_rows(self):
        """Test that rows with NaN are removed"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        X[10, 2] = np.nan  # Add NaN to row 10
        y[20] = np.nan     # Add NaN to row 20

        X_val, y_val = _validate(X, y)
        assert X_val.shape == (98, 5)  # 2 rows removed
        assert y_val.shape == (98,)
        assert not np.any(np.isnan(X_val))
        assert not np.any(np.isnan(y_val))

    def test_removes_inf_rows(self):
        """Test that rows with Inf are removed"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        X[15, 3] = np.inf

        X_val, y_val = _validate(X, y)
        assert X_val.shape == (99, 5)  # 1 row removed
        assert not np.any(np.isinf(X_val))


class TestSplitAndStandardize:
    """Test split_and_standardize function"""

    def test_basic_split(self):
        """Test basic data splitting"""
        X = np.random.randn(1000, 5).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)

        ds = split_and_standardize(X, y, test_size=0.15, val_size=0.15, random_state=42)

        # Check split sizes (approximately)
        assert ds.X_train.shape[0] == 700  # 70%
        assert ds.X_val.shape[0] == 150    # 15%
        assert ds.X_test.shape[0] == 150   # 15%

        # Check all have 5 features
        assert ds.X_train.shape[1] == 5
        assert ds.X_val.shape[1] == 5
        assert ds.X_test.shape[1] == 5

        # Check y shapes
        assert ds.y_train.shape == (700,)
        assert ds.y_val.shape == (150,)
        assert ds.y_test.shape == (150,)

    def test_standardization(self):
        """Test that X is standardized"""
        X = np.random.randn(1000, 5).astype(np.float32) * 10 + 5  # Mean ~5, std ~10
        y = np.random.randn(1000).astype(np.float32)

        ds = split_and_standardize(X, y)

        # Training data should have mean ~0 and std ~1
        assert np.abs(ds.X_train.mean()) < 0.1
        assert np.abs(ds.X_train.std() - 1.0) < 0.1

    def test_y_scaling_disabled(self):
        """Test that y is not scaled when scale_y=False"""
        X = np.random.randn(1000, 5).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32) * 100 + 50

        ds = split_and_standardize(X, y, scale_y=False)

        assert ds.y_scaler is None
        # y should not be standardized
        assert np.abs(ds.y_train.mean() - 50) < 10  # Still has original scale

    def test_y_scaling_enabled(self):
        """Test that y is scaled when scale_y=True"""
        X = np.random.randn(1000, 5).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32) * 100 + 50

        ds = split_and_standardize(X, y, scale_y=True)

        assert ds.y_scaler is not None
        # y should be standardized
        assert np.abs(ds.y_train.mean()) < 0.1
        assert np.abs(ds.y_train.std() - 1.0) < 0.1

    def test_reproducibility(self):
        """Test that same random_state gives same split"""
        X = np.random.randn(1000, 5).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)

        ds1 = split_and_standardize(X, y, random_state=42)
        ds2 = split_and_standardize(X, y, random_state=42)

        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)


class TestLoadDatasetNpz:
    """Test load_dataset_npz function"""

    def test_load_npz_file(self):
        """Test loading from .npz file"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f, X=X, y=y)
            temp_path = f.name

        try:
            X_loaded, y_loaded = load_dataset_npz(temp_path)
            assert X_loaded.shape == (100, 5)
            assert y_loaded.shape == (100,)
            np.testing.assert_array_almost_equal(X_loaded, X)
            np.testing.assert_array_almost_equal(y_loaded, y)
        finally:
            os.unlink(temp_path)
