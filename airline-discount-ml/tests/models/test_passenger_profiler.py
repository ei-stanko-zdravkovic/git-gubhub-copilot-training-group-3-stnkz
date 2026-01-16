"""
Unit tests for passenger_profiler module.

Validates:
- build_features produces required columns
- Distance conversion from miles to km
- Error handling for empty input
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models import build_features


def test_build_features_basic():
    """Test build_features produces required columns."""
    df = pd.DataFrame({
        "distance_km": [3459.0, 5478.0],
        "history_trips": [10, 25],
        "avg_spend": [500.0, 1200.0],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    })
    
    result = build_features(df)
    
    expected_cols = [
        "distance_km", "history_trips", "avg_spend",
        "route_id", "origin", "destination"
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == 2
    assert result.index.equals(df.index)


def test_build_features_converts_miles():
    """Test build_features converts distance in miles to km."""
    df = pd.DataFrame({
        "distance": [2150.0],  # miles
        "history_trips": [5],
        "avg_spend": [800.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert result["distance_km"].iloc[0] == pytest.approx(2150.0 * 1.60934, rel=1e-3)


def test_build_features_empty_raises():
    """Test build_features raises on empty DataFrame."""
    with pytest.raises(ValueError, match="non-empty"):
        build_features(pd.DataFrame())


def test_build_features_rejects_list():
    """Test build_features raises ValueError for list input."""
    with pytest.raises(ValueError, match="must be a non-empty pandas DataFrame"):
        build_features([{"distance_km": 1000}])


def test_build_features_rejects_none():
    """Test build_features raises ValueError for None input."""
    with pytest.raises(ValueError, match="must be a non-empty pandas DataFrame"):
        build_features(None)


def test_build_features_uses_trips_count_fallback():
    """Test build_features uses trips_count when history_trips missing."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "trips_count": [15],  # Should be used as history_trips
        "avg_spend": [800.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert result["history_trips"].iloc[0] == 15


def test_build_features_calculates_avg_spend():
    """Test build_features calculates avg_spend from total_spend and history_trips."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "history_trips": [10],
        "total_spend": [5000.0],  # avg_spend should be 500
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert result["avg_spend"].iloc[0] == pytest.approx(500.0)


def test_build_features_handles_division_by_zero():
    """Test build_features handles division by zero (returns inf)."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "history_trips": [0],  # Division by zero
        "total_spend": [5000.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    # Division by zero produces inf, not NA
    import numpy as np
    assert np.isinf(result["avg_spend"].iloc[0])


def test_build_features_missing_distance_returns_na():
    """Test build_features returns NA when distance columns missing."""
    df = pd.DataFrame({
        "history_trips": [10],
        "avg_spend": [500.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert pd.isna(result["distance_km"].iloc[0])


def test_build_features_missing_history_returns_na():
    """Test build_features returns NA when history columns missing."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "avg_spend": [500.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert pd.isna(result["history_trips"].iloc[0])


def test_build_features_missing_avg_spend_returns_na():
    """Test build_features returns NA when avg_spend cannot be derived."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "history_trips": [10],
        # No avg_spend or total_spend
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    assert pd.isna(result["avg_spend"].iloc[0])


def test_build_features_missing_optional_columns():
    """Test build_features returns NA for missing optional columns."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "history_trips": [10],
        "avg_spend": [500.0],
        # Missing route_id, origin, destination
    })
    
    result = build_features(df)
    assert pd.isna(result["route_id"].iloc[0])
    assert pd.isna(result["origin"].iloc[0])
    assert pd.isna(result["destination"].iloc[0])


def test_build_features_preserves_index():
    """Test build_features preserves custom DataFrame index."""
    df = pd.DataFrame({
        "distance_km": [3000.0, 4000.0],
        "history_trips": [10, 15],
        "avg_spend": [500.0, 800.0],
        "route_id": ["R1", "R2"],
        "origin": ["NYC", "LAX"],
        "destination": ["LON", "TYO"],
    }, index=["passenger_1", "passenger_2"])
    
    result = build_features(df)
    assert list(result.index) == ["passenger_1", "passenger_2"]


def test_build_features_returns_only_required_columns():
    """Test build_features returns only required columns, not extra ones."""
    df = pd.DataFrame({
        "distance_km": [3000.0],
        "history_trips": [10],
        "avg_spend": [500.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
        "extra_column": ["should_not_appear"],
        "another_extra": [123],
    })
    
    result = build_features(df)
    
    expected_cols = [
        "distance_km", "history_trips", "avg_spend",
        "route_id", "origin", "destination"
    ]
    assert list(result.columns) == expected_cols
    assert "extra_column" not in result.columns
    assert "another_extra" not in result.columns


def test_build_features_column_order():
    """Test build_features returns columns in specified order."""
    df = pd.DataFrame({
        "destination": ["LON"],  # Intentionally out of order
        "origin": ["NYC"],
        "avg_spend": [500.0],
        "route_id": ["R1"],
        "history_trips": [10],
        "distance_km": [3000.0],
    })
    
    result = build_features(df)
    
    expected_order = [
        "distance_km", "history_trips", "avg_spend",
        "route_id", "origin", "destination"
    ]
    assert list(result.columns) == expected_order


def test_build_features_all_columns_present_even_if_na():
    """Test build_features always returns all required columns even if NA."""
    df = pd.DataFrame({
        # Only provide one column
        "distance_km": [3000.0],
    })
    
    result = build_features(df)
    
    expected_cols = [
        "distance_km", "history_trips", "avg_spend",
        "route_id", "origin", "destination"
    ]
    assert list(result.columns) == expected_cols
    assert len(result.columns) == 6


def test_build_features_with_trips_count_and_total_spend():
    """Test build_features derives both history_trips and avg_spend."""
    df = pd.DataFrame({
        "distance": [2000.0],  # miles
        "trips_count": [20],
        "total_spend": [10000.0],
        "route_id": ["R1"],
        "origin": ["NYC"],
        "destination": ["LON"],
    })
    
    result = build_features(df)
    
    # distance converted to km
    assert result["distance_km"].iloc[0] == pytest.approx(2000.0 * 1.60934, rel=1e-3)
    # trips_count used as history_trips
    assert result["history_trips"].iloc[0] == 20
    # avg_spend calculated from total_spend / trips_count
    assert result["avg_spend"].iloc[0] == pytest.approx(500.0)
