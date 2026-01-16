---
name: generate-test-fixtures
description: Standard procedure for creating reusable pytest fixtures in conftest.py files for the airline-discount-ml project.
---

# Generating Test Fixtures

When the user needs to create test fixtures, follow these standardized patterns for adding them to conftest.py files.

## Fixture Location Strategy

### Project-Wide Fixtures
**Location:** `tests/conftest.py`

Use for fixtures needed across multiple test modules:
- `synthetic_data` - Training datasets
- `sample_features` - Small feature DataFrames
- `temp_db_path` - Temporary database paths
- `mock_database_connection` - Mock DB connections

### Module-Specific Fixtures
**Location:** `tests/<module>/conftest.py`

Use for fixtures specific to one module:
- `tests/models/conftest.py` - Model-specific fixtures
- `tests/data/conftest.py` - Database-specific fixtures
- `tests/agents/conftest.py` - Agent-specific fixtures

## Standard Fixture Patterns

### 1. Synthetic Data Fixture (ML Models)
```python
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
    
    # Simple linear target
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    
    return X, y
```

### 2. Sample Features Fixture (Quick Tests)
```python
@pytest.fixture
def sample_features():
    """Create minimal feature DataFrame for testing.
    
    Provides a small 3-row DataFrame with all required columns for
    DiscountPredictor model testing. Uses deterministic data.
    
    Returns:
        pd.DataFrame: Features with required columns
    """
    return pd.DataFrame({
        "distance_km": [2000.0, 3500.0, 5000.0],
        "history_trips": [5, 15, 30],
        "avg_spend": [500.0, 1200.0, 2000.0],
        "route_id": ["R1", "R2", "R3"],
        "origin": ["NYC", "LAX", "SFO"],
        "destination": ["LON", "TYO", "PAR"],
    })
```

### 3. Temporary Database Fixture
```python
@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database for testing.
    
    Creates an in-memory or file-based temporary database that is
    automatically cleaned up after the test completes.
    
    Args:
        tmp_path: pytest's built-in tmp_path fixture
        
    Yields:
        Database: Temporary database connection
    """
    from src.data.database import Database
    
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    # Initialize schema
    db.init_database()
    
    yield db
    
    # Cleanup
    db.close()
```

### 4. Mock Database Connection Fixture
```python
@pytest.fixture
def mock_database_connection():
    """Create a mock database connection for testing.
    
    Returns a Mock object configured with common database methods.
    Useful for testing code that depends on database connections
    without requiring a real database.
    
    Returns:
        Mock: Configured mock database connection
    """
    from unittest.mock import Mock
    
    mock_db = Mock()
    mock_db.fetch_data.return_value = []
    mock_db.execute.return_value = None
    mock_db.close.return_value = None
    
    return mock_db
```

### 5. Trained Model Fixture
```python
@pytest.fixture
def trained_model(synthetic_data):
    """Provide a pre-trained model for evaluation tests.
    
    Trains a DiscountPredictor on synthetic data to avoid
    repeated training in multiple tests.
    
    Args:
        synthetic_data: Fixture providing X, y
        
    Returns:
        DiscountPredictor: Trained model instance
    """
    from src.models.discount_predictor import DiscountPredictor
    
    X, y = synthetic_data
    model = DiscountPredictor()
    model.fit(X, y)
    
    return model
```

### 6. Temporary File Fixture
```python
@pytest.fixture
def temp_model_file(tmp_path):
    """Provide temporary file path for model save/load tests.
    
    Args:
        tmp_path: pytest's built-in tmp_path fixture
        
    Returns:
        Path: Temporary file path that will be cleaned up
    """
    return tmp_path / "test_model.pkl"
```

## Fixture Scopes

### Function Scope (Default)
Recreated for each test function:
```python
@pytest.fixture
def function_scoped_data():
    return {"value": 42}
```

### Class Scope
Shared across all tests in a class:
```python
@pytest.fixture(scope="class")
def class_scoped_model():
    model = train_expensive_model()
    return model
```

### Module Scope
Shared across all tests in a module:
```python
@pytest.fixture(scope="module")
def module_scoped_db():
    db = setup_test_database()
    yield db
    db.teardown()
```

### Session Scope
Created once for entire test session:
```python
@pytest.fixture(scope="session")
def session_config():
    return load_test_config()
```

## Fixture Dependencies

### Using Other Fixtures
```python
@pytest.fixture
def complete_dataset(sample_features, synthetic_data):
    """Combine multiple fixtures."""
    X_sample, _ = sample_features
    X_synthetic, y_synthetic = synthetic_data
    
    # Combine or use both
    return {
        "sample": X_sample,
        "synthetic": (X_synthetic, y_synthetic)
    }
```

### Parameterized Fixtures
Test same code with different inputs:
```python
@pytest.fixture(params=[10, 100, 1000])
def dataset_size(request):
    """Provide different dataset sizes."""
    return request.param

def test_model_scales(dataset_size, sample_features):
    X = sample_features.iloc[:dataset_size]
    # Test scales with different sizes
```

## Adding Fixtures to conftest.py

### Step 1: Determine Scope
```python
# Is it used by multiple modules? → tests/conftest.py
# Is it specific to one module? → tests/<module>/conftest.py
```

### Step 2: Add Import Statements
```python
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
```

### Step 3: Add Fixture with Documentation
```python
@pytest.fixture
def your_fixture_name():
    """Clear description of what this fixture provides.
    
    Include details about:
    - What data/objects it creates
    - Any setup/teardown actions
    - When to use it vs. other fixtures
    
    Returns:
        type: Description of return value
    """
    # Setup
    data = create_test_data()
    
    # Optionally yield for cleanup
    yield data
    
    # Cleanup (if needed)
    cleanup(data)
```

### Step 4: Document in Test File
```python
def test_example(your_fixture_name):
    """Test using the new fixture.
    
    Args:
        your_fixture_name: Provides test data from conftest.py
    """
    result = function_under_test(your_fixture_name)
    assert result is not None
```

## Best Practices

### DO:
✅ Use descriptive fixture names (`synthetic_training_data` not `data`)  
✅ Add comprehensive docstrings  
✅ Set random seeds for reproducibility (`np.random.seed(42)`)  
✅ Use appropriate scope (function by default)  
✅ Clean up resources (yield pattern)  
✅ Keep fixtures focused and simple  

### DON'T:
❌ Create global state or side effects  
❌ Make fixtures too complex (split into multiple fixtures)  
❌ Use session scope unless truly needed  
❌ Hardcode file paths (use tmp_path)  
❌ Access real databases or network  

## Testing Fixtures

Verify fixtures work correctly:
```python
def test_synthetic_data_fixture(synthetic_data):
    """Verify synthetic_data fixture provides correct format."""
    X, y = synthetic_data
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) == 100  # Expected size
    assert y.name == "discount_value"
```

## Quick Reference

| Fixture Type | Location | Scope | Use Case |
|--------------|----------|-------|----------|
| Synthetic data | tests/conftest.py | function | ML training/testing |
| Sample features | tests/conftest.py | function | Quick model tests |
| Temp DB | tests/data/conftest.py | function | Database tests |
| Mock connection | tests/conftest.py | function | Integration tests |
| Trained model | tests/models/conftest.py | module | Evaluation tests |
| Config | tests/conftest.py | session | Shared config |

## Example conftest.py Structure

```python
"""
Shared pytest fixtures for all test modules.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features():
    """Small DataFrame for quick tests."""
    # ... implementation

@pytest.fixture
def synthetic_data():
    """100-row dataset for training tests."""
    # ... implementation

@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database file path."""
    return tmp_path / "test.db"

@pytest.fixture
def mock_database_connection():
    """Mock DB connection."""
    # ... implementation
```
