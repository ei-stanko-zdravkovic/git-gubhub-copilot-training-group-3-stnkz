"""
Unit tests for train.py module.

Validates:
- train_model function
- Data loading from database
- Model saving
- Error handling and validation
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.train import train_model, main


@pytest.fixture
def synthetic_training_data():
    """Create synthetic training dataset."""
    np.random.seed(42)
    n = 100
    
    X = pd.DataFrame({
        "distance_km": np.random.uniform(1000, 6000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(100, 2000, n),
        "route_id": np.random.choice(["R1", "R2", "R3"], n),
        "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
        "destination": np.random.choice(["LON", "TYO", "PAR"], n),
    })
    
    # Simple linear target
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    
    return X, y


@pytest.fixture
def mock_database():
    """Mock database connection for training tests."""
    db_mock = Mock()
    db_mock.fetch_data.return_value = []
    db_mock.close.return_value = None
    return db_mock


def test_train_model_basic(synthetic_training_data):
    """Test train_model function with valid data."""
    X, y = synthetic_training_data
    data = {"features": X, "labels": y}
    
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    trained = train_model(model, data)
    
    # Model should be trained and able to predict
    predictions = trained.predict(X)
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X)
    assert predictions.name == "discount_value"


def test_train_model_returns_fitted_model(synthetic_training_data):
    """Test train_model returns a fitted model instance."""
    X, y = synthetic_training_data
    data = {"features": X, "labels": y}
    
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    trained = train_model(model, data)
    
    # Should not raise RuntimeError about not being fitted
    try:
        trained.predict(X)
        fitted = True
    except RuntimeError:
        fitted = False
    
    assert fitted, "Model should be fitted after train_model"


def test_train_model_validates_data_structure():
    """Test train_model validates data dictionary structure."""
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    # Missing 'features' key
    with pytest.raises(KeyError):
        train_model(model, {"labels": pd.Series([1, 2, 3])})
    
    # Missing 'labels' key
    with pytest.raises(KeyError):
        train_model(model, {"features": pd.DataFrame()})


def test_train_model_empty_features_raises():
    """Test train_model with empty features raises error."""
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    data = {
        "features": pd.DataFrame(),
        "labels": pd.Series([1])
    }
    
    with pytest.raises(ValueError, match="empty"):
        train_model(model, data)


@patch("src.training.train.get_connection")
def test_main_loads_data_from_database(mock_get_conn, mock_database, tmp_path):
    """Test main function loads data from database."""
    # Setup mock database with test data
    np.random.seed(42)
    n = 50
    rows = [
        {
            "discount_value": 10.0 + i,
            "distance_km": 2000.0 + i * 10,
            "history_trips": 5 + i % 10,
            "avg_spend": 500.0 + i * 5,
            "route_id": f"R{i % 3}",
            "origin": ["NYC", "LAX", "SFO"][i % 3],
            "destination": ["LON", "TYO", "PAR"][i % 3],
        }
        for i in range(n)
    ]
    
    mock_db = Mock()
    mock_db.fetch_data.return_value = rows
    mock_db.close.return_value = None
    mock_get_conn.return_value = mock_db
    
    # Use real temporary path instead of mocking Path
    model_path = tmp_path / "test_model.pkl"
    
    with patch("src.training.train.Path") as mock_path_class:
        # Return real path when Path() is called
        mock_path_class.return_value = model_path
        
        # Should not raise
        result = main()
        
        # Verify database was queried
        assert mock_db.fetch_data.called
        assert mock_db.close.called
        
        # Verify model was returned and saved
        assert result is not None
        assert model_path.exists()


@patch("src.training.train.get_connection")
def test_main_raises_on_no_data(mock_get_conn):
    """Test main raises error when no training data is available."""
    mock_db = Mock()
    mock_db.fetch_data.return_value = []
    mock_db.close.return_value = None
    mock_get_conn.return_value = mock_db
    
    with pytest.raises(ValueError, match="No training data found"):
        main()


@patch("src.training.train.get_connection")
def test_main_raises_on_none_data(mock_get_conn):
    """Test main raises error when database returns None."""
    mock_db = Mock()
    mock_db.fetch_data.return_value = None
    mock_db.close.return_value = None
    mock_get_conn.return_value = mock_db
    
    with pytest.raises(ValueError, match="No training data found"):
        main()


def test_train_model_preserves_feature_order(synthetic_training_data):
    """Test that train_model maintains feature column order."""
    X, y = synthetic_training_data
    data = {"features": X, "labels": y}
    
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    original_columns = list(X.columns)
    trained = train_model(model, data)
    
    # Predict to ensure model works with same column order
    predictions = trained.predict(X)
    assert isinstance(predictions, pd.Series)
    
    # X columns should not have changed
    assert list(X.columns) == original_columns


def test_train_model_with_custom_index(synthetic_training_data):
    """Test train_model works with custom DataFrame index."""
    X, y = synthetic_training_data
    
    # Set custom index
    custom_index = [f"sample_{i}" for i in range(len(X))]
    X.index = custom_index
    y.index = custom_index
    
    data = {"features": X, "labels": y}
    
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    
    trained = train_model(model, data)
    predictions = trained.predict(X)
    
    # Predictions should preserve index
    assert list(predictions.index) == custom_index


def test_train_model_deterministic(synthetic_training_data):
    """Test train_model produces deterministic results with same seed."""
    X, y = synthetic_training_data
    data = {"features": X, "labels": y}
    
    from src.models.discount_predictor import DiscountPredictor
    
    # Train first model
    np.random.seed(42)
    model1 = DiscountPredictor()
    trained1 = train_model(model1, data)
    preds1 = trained1.predict(X)
    
    # Train second model with same seed
    np.random.seed(42)
    model2 = DiscountPredictor()
    trained2 = train_model(model2, data)
    preds2 = trained2.predict(X)
    
    # Predictions should be identical
    pd.testing.assert_series_equal(preds1, preds2)
