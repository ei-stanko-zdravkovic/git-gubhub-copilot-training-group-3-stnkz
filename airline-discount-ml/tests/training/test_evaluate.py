"""
Unit tests for evaluate.py module.

Validates:
- evaluate_model function
- Metric calculations (MAE, MSE, RMSE, R²)
- Data loading from database
- Error handling and validation
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.evaluate import evaluate_model, calculate_accuracy, main


@pytest.fixture
def synthetic_test_data():
    """Create synthetic test dataset."""
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
    
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    
    return X, y


@pytest.fixture
def trained_mock_model(synthetic_test_data):
    """Create a mock trained model."""
    X, y = synthetic_test_data
    
    from src.models.discount_predictor import DiscountPredictor
    model = DiscountPredictor()
    model.fit(X, y)
    
    return model


def test_evaluate_model_returns_metrics(trained_mock_model, synthetic_test_data):
    """Test evaluate_model returns all expected metrics."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    assert "predictions" in results
    assert "mae" in results
    assert "mse" in results
    assert "rmse" in results
    assert "r2_score" in results


def test_evaluate_model_predictions_shape(trained_mock_model, synthetic_test_data):
    """Test predictions have correct shape and type."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    predictions = results["predictions"]
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
    assert predictions.name == "discount_value"


def test_evaluate_model_metric_values(trained_mock_model, synthetic_test_data):
    """Test metric values are reasonable and match manual calculation."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    # Manual calculation
    predictions = trained_mock_model.predict(X)
    expected_mae = mean_absolute_error(y, predictions)
    expected_mse = mean_squared_error(y, predictions)
    expected_rmse = np.sqrt(expected_mse)
    expected_r2 = r2_score(y, predictions)
    
    # Compare
    assert abs(results["mae"] - expected_mae) < 1e-6
    assert abs(results["mse"] - expected_mse) < 1e-6
    assert abs(results["rmse"] - expected_rmse) < 1e-6
    assert abs(results["r2_score"] - expected_r2) < 1e-6


def test_evaluate_model_metrics_non_negative(trained_mock_model, synthetic_test_data):
    """Test MAE, MSE, RMSE are non-negative."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    assert results["mae"] >= 0
    assert results["mse"] >= 0
    assert results["rmse"] >= 0


def test_evaluate_model_r2_bounded(trained_mock_model, synthetic_test_data):
    """Test R² score is within reasonable bounds."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    # R² can be negative for very bad models, but should be < 1
    assert results["r2_score"] <= 1.0


def test_evaluate_model_rmse_sqrt_mse(trained_mock_model, synthetic_test_data):
    """Test RMSE is sqrt of MSE."""
    X, y = synthetic_test_data
    test_data = {"features": X, "labels": y}
    
    results = evaluate_model(trained_mock_model, test_data)
    
    expected_rmse = np.sqrt(results["mse"])
    assert abs(results["rmse"] - expected_rmse) < 1e-6


def test_evaluate_model_validates_data_structure():
    """Test evaluate_model validates data dictionary structure."""
    mock_model = Mock()
    
    # Missing 'features' key
    with pytest.raises(KeyError):
        evaluate_model(mock_model, {"labels": pd.Series([1, 2, 3])})
    
    # Missing 'labels' key
    with pytest.raises(KeyError):
        evaluate_model(mock_model, {"features": pd.DataFrame()})


def test_evaluate_model_with_perfect_predictions():
    """Test evaluate_model with perfect predictions (MAE=0, R²=1)."""
    np.random.seed(42)
    n = 50
    
    X = pd.DataFrame({
        "distance_km": np.random.uniform(1000, 6000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(100, 2000, n),
        "route_id": np.random.choice(["R1", "R2", "R3"], n),
        "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
        "destination": np.random.choice(["LON", "TYO", "PAR"], n),
    })
    y = pd.Series(np.random.uniform(5, 30, n), name="discount_value")
    
    # Mock model that predicts perfectly
    mock_model = Mock()
    mock_model.predict.return_value = y.copy()
    
    test_data = {"features": X, "labels": y}
    results = evaluate_model(mock_model, test_data)
    
    assert results["mae"] < 1e-10  # Nearly zero
    assert results["mse"] < 1e-10
    assert results["rmse"] < 1e-10
    assert abs(results["r2_score"] - 1.0) < 1e-10


def test_calculate_accuracy_basic():
    """Test legacy calculate_accuracy function."""
    predictions = [1, 2, 3, 4, 5]
    labels = [1, 2, 0, 4, 0]
    
    accuracy = calculate_accuracy(predictions, labels)
    
    # 3 out of 5 match
    assert accuracy == 0.6


def test_calculate_accuracy_all_correct():
    """Test calculate_accuracy with all correct predictions."""
    predictions = [1, 2, 3]
    labels = [1, 2, 3]
    
    accuracy = calculate_accuracy(predictions, labels)
    assert accuracy == 1.0


def test_calculate_accuracy_none_correct():
    """Test calculate_accuracy with no correct predictions."""
    predictions = [1, 2, 3]
    labels = [4, 5, 6]
    
    accuracy = calculate_accuracy(predictions, labels)
    assert accuracy == 0.0


def test_calculate_accuracy_empty_lists():
    """Test calculate_accuracy with empty inputs."""
    predictions = []
    labels = []
    
    accuracy = calculate_accuracy(predictions, labels)
    assert accuracy == 0.0


@patch("src.training.evaluate.get_connection")
@patch("src.training.evaluate.DiscountPredictor")
def test_main_loads_model(mock_predictor_class, mock_get_conn, tmp_path):
    """Test main function loads model from disk."""
    # Setup mock model
    mock_model = Mock()
    mock_model.predict.return_value = pd.Series([10.0, 15.0, 20.0], name="discount_value")
    mock_predictor_class.load.return_value = mock_model
    
    # Setup mock database
    rows = [
        {
            "discount_value": 10.0,
            "distance_km": 2000.0,
            "history_trips": 5,
            "avg_spend": 500.0,
            "route_id": "R1",
            "origin": "NYC",
            "destination": "LON",
        },
        {
            "discount_value": 15.0,
            "distance_km": 3000.0,
            "history_trips": 10,
            "avg_spend": 800.0,
            "route_id": "R2",
            "origin": "LAX",
            "destination": "TYO",
        },
        {
            "discount_value": 20.0,
            "distance_km": 4000.0,
            "history_trips": 15,
            "avg_spend": 1200.0,
            "route_id": "R3",
            "origin": "SFO",
            "destination": "PAR",
        },
    ]
    
    mock_db = Mock()
    mock_db.fetch_data.return_value = rows
    mock_db.close.return_value = None
    mock_get_conn.return_value = mock_db
    
    # Mock model path
    with patch("src.training.evaluate.Path") as mock_path_class:
        mock_model_path = Mock()
        mock_model_path.exists.return_value = True
        mock_path_class.return_value = mock_model_path
        
        result = main()
        
        # Verify model was loaded
        assert mock_predictor_class.load.called
        
        # Verify database was queried
        assert mock_db.fetch_data.called
        assert mock_db.close.called
        
        # Verify results returned
        assert result is not None
        assert "predictions" in result


@patch("src.training.evaluate.Path")
def test_main_raises_if_model_not_found(mock_path_class):
    """Test main raises error if model file doesn't exist."""
    mock_model_path = Mock()
    mock_model_path.exists.return_value = False
    mock_path_class.return_value = mock_model_path
    
    with pytest.raises(FileNotFoundError, match="Model not found"):
        main()


@patch("src.training.evaluate.get_connection")
@patch("src.training.evaluate.DiscountPredictor")
@patch("src.training.evaluate.Path")
def test_main_raises_on_no_test_data(mock_path_class, mock_predictor_class, mock_get_conn):
    """Test main raises error when no test data is available."""
    # Mock model exists
    mock_model_path = Mock()
    mock_model_path.exists.return_value = True
    mock_path_class.return_value = mock_model_path
    
    # Mock model load
    mock_model = Mock()
    mock_predictor_class.load.return_value = mock_model
    
    # Mock database returns empty
    mock_db = Mock()
    mock_db.fetch_data.return_value = []
    mock_db.close.return_value = None
    mock_get_conn.return_value = mock_db
    
    with pytest.raises(ValueError, match="No test data found"):
        main()


def test_evaluate_model_index_preservation(trained_mock_model):
    """Test evaluate_model preserves DataFrame index."""
    np.random.seed(42)
    n = 50
    
    X = pd.DataFrame({
        "distance_km": np.random.uniform(1000, 6000, n),
        "history_trips": np.random.randint(1, 50, n),
        "avg_spend": np.random.uniform(100, 2000, n),
        "route_id": np.random.choice(["R1", "R2", "R3"], n),
        "origin": np.random.choice(["NYC", "LAX", "SFO"], n),
        "destination": np.random.choice(["LON", "TYO", "PAR"], n),
    }, index=[f"sample_{i}" for i in range(n)])
    
    y = pd.Series(
        np.random.uniform(5, 30, n),
        index=X.index,
        name="discount_value"
    )
    
    test_data = {"features": X, "labels": y}
    results = evaluate_model(trained_mock_model, test_data)
    
    # Predictions should have same index as input
    assert list(results["predictions"].index) == list(X.index)
