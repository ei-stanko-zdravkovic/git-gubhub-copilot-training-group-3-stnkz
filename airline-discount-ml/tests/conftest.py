"""
Shared pytest fixtures for all test modules.

This conftest.py provides common fixtures that can be used across all test files
to avoid code duplication. Fixtures defined here are automatically available to
all tests in the tests/ directory and subdirectories.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_fixture():
    """Legacy fixture - example of session-scoped fixture."""
    return "Hello, World!"


@pytest.fixture
def sample_features():
    """Create minimal feature DataFrame for testing.
    
    Provides a small 3-row DataFrame with all required columns for
    DiscountPredictor model testing. Uses deterministic data (no randomness).
    
    Returns:
        pd.DataFrame: Features with columns [distance_km, history_trips, 
                     avg_spend, route_id, origin, destination]
    """
    return pd.DataFrame({
        "distance_km": [2000.0, 3500.0, 5000.0],
        "history_trips": [5, 15, 30],
        "avg_spend": [500.0, 1200.0, 2000.0],
        "route_id": ["R1", "R2", "R3"],
        "origin": ["NYC", "LAX", "SFO"],
        "destination": ["LON", "TYO", "PAR"],
    })


@pytest.fixture
def synthetic_data():
    """Create synthetic training dataset with 100 samples.
    
    Uses fixed random seed (42) for reproducibility. Returns features
    and target variable suitable for model training and evaluation.
    
    Returns:
        tuple: (X: pd.DataFrame, y: pd.Series) with 100 rows
    """
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
    
    # Simple linear target: discount increases with distance and history
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    
    return X, y


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database file path.
    
    Creates a temporary directory and returns a Path object pointing to
    a test.db file within it. The directory is automatically cleaned up
    after the test completes.
    
    Args:
        tmp_path: pytest's built-in tmp_path fixture
        
    Returns:
        Path: Path to temporary database file
    """
    return tmp_path / "test.db"


@pytest.fixture
def mock_database_connection():
    """Create a mock database connection for testing.
    
    Returns a Mock object configured with common database methods
    (fetch_data, execute, close). Useful for testing code that depends
    on database connections without requiring a real database.
    
    Returns:
        Mock: Configured mock database connection
    """
    from unittest.mock import Mock
    
    mock_db = Mock()
    mock_db.fetch_data.return_value = []
    mock_db.execute.return_value = None
    mock_db.close.return_value = None
    
    return mock_db
