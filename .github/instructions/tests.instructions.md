---
applyTo: "airline-discount-ml/tests/**/*.py,tests/**/*.py"
---

# Copilot Instructions for tests/

Purpose
- Ensure Copilot produces accurate, deterministic, fast, and maintainable test code.
- Guide test generation for the airline discount ML project using pytest best practices.

Scope (this folder)
- tests/models/: Unit tests for ML models (discount_predictor.py, passenger_profiler.py).
- tests/data/: Tests for database operations and data loading.
- tests/agents/: Tests for business logic agents.
- All test files follow pytest conventions (test_*.py or *_test.py).

Project context Copilot must assume
- Test framework: pytest with fixtures and parametrization.
- Test structure mirrors src/: tests/models/ → src/models/, tests/data/ → src/data/, etc.
- Python >=3.8, all tests must be deterministic and isolated.
- No external I/O: use mocks, fixtures, or in-memory databases for database tests.
- Synthetic data: prefer np.random with fixed seeds (seed=42) for reproducibility.
- Type hints and docstrings required for all test functions and fixtures.

Hard constraints
- No network calls, file I/O to real paths, or time-based tests without mocking.
- All tests must be independent (no shared state between tests).
- Use pytest fixtures for test data—avoid repeated setup in each test.
- Tests must be fast (<100ms per test preferred, <1s max).
- No production data or PII in tests—only synthetic test data.
- Models must never import from src.data.database (test in isolation with DataFrames).

Coding standards
- Test functions: descriptive names starting with test_, clear docstrings.
- Fixtures: use @pytest.fixture with function or session scope as appropriate.
- Assertions: use specific assertions (assert, pytest.raises, pd.testing.assert_*).
- Parametrization: use @pytest.mark.parametrize for multiple similar test cases.
- Error messages: clear, specific messages in assertions and pytest.raises(match="...").
- Deterministic data: always set random seeds (np.random.seed(42)) in fixtures.
- Index preservation: for pandas operations, verify returned index matches input.

Test patterns by module

Models (tests/models/)
- Test public APIs: fit(), predict(), save(), load(), build_features().
- Validate input checking: empty DataFrames, missing columns, wrong types.
- Verify state management: predict() before fit() raises RuntimeError.
- Baseline comparison: model must beat DummyRegressor on MAE and R².
- Save/load roundtrip: predictions identical before and after serialization.
- Index preservation: predict() output index must match input X index.
- Feature engineering: build_features() produces required columns in correct order.
- Unit conversions: test miles to km conversion, avg_spend derivation.

Data (tests/data/)
- Use in-memory SQLite (:memory:) or tempfile databases—never real data/airline_discount.db.
- Test schema creation: tables exist with correct columns and constraints.
- Test data loading: load_synthetic_data functions insert correct record counts.
- Test foreign key constraints: invalid references raise IntegrityError.
- Test query functions: get_connection(), fetch methods return expected DataFrames.
- Clean up: use fixtures with yield to close connections and clean temp files.

Agents (tests/agents/)
- Mock database and model dependencies—test business logic in isolation.
- Test edge cases: empty inputs, invalid parameters, boundary conditions.
- Verify return types and structure (DataFrames, dicts, lists).
- Test error handling: appropriate exceptions for invalid inputs.

Fixtures to create or reuse

@pytest.fixture
def synthetic_data():
    """Create minimal synthetic dataset for model testing.
    
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
    y = (
        0.002 * X["distance_km"] 
        + 0.3 * X["history_trips"] 
        + 0.005 * X["avg_spend"]
        + np.random.normal(0, 2, n)
    )
    y = pd.Series(y, name="discount_value")
    return X, y


@pytest.fixture
def temp_db():
    """Create temporary SQLite database for testing.
    
    Yields:
        Database: Temporary database connection
    """
    import tempfile
    from pathlib import Path
    from src.data.database import Database
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        yield db
        db.close()


Ready-to-copy prompts for Copilot

Model tests:
- "/tests #file:discount_predictor.py #sym:fit Write pytest tests for the fit method. Cover: empty X, missing columns, wrong types, successful fit. Use synthetic_data fixture."
- "/tests #file:discount_predictor.py #sym:predict Write pytest tests for predict. Cover: predict before fit (raises), index preservation, correct output type."
- "/tests #file:discount_predictor.py #sym:save Write pytest tests for save/load roundtrip. Use tempfile, verify predictions identical."
- "/tests #file:passenger_profiler.py #sym:build_features Write pytest tests covering: miles to km conversion, avg_spend derivation, empty input raises, required columns present."

Data tests:
- "/tests #file:database.py Write pytest tests using in-memory SQLite. Test: init_database creates tables, get_connection returns valid connection, load functions insert data."
- "/tests #file:load_synthetic_data.py Test load_all function with mock JSON data. Verify record counts, foreign key constraints, clear_existing flag behavior."

Validation tests:
- "/tests #file:discount_predictor.py #sym:_validate_X Write pytest tests for input validation. Cover: valid DataFrame (no exception), not a DataFrame (raises), empty DataFrame (raises). Use pytest.raises."
- "/tests #file:discount_predictor.py #sym:_validate_y Write pytest tests for target validation. Cover: valid Series, not a Series, empty Series. Use pytest.raises(ValueError, match='...')."

Gap finding:
- "#codebase Which functions in src/models/discount_predictor.py lack test coverage? List untested branches and error paths."
- "#file:discount_predictor.py #file:test_discount_predictor.py What test cases are missing? Compare implementation to existing tests."

Do
- Set random seeds in fixtures: np.random.seed(42), random.seed(42).
- Use pytest.raises(ExceptionType, match="expected message") for exceptions.
- Use pd.testing.assert_frame_equal() and pd.testing.assert_series_equal() for pandas.
- Use tempfile.TemporaryDirectory() for file-based tests (save/load).
- Test both success and failure paths (happy path + edge cases).
- Keep test functions small and focused (one concept per test).
- Use descriptive test names: test_<function>_<scenario>_<expected_result>.
- Add docstrings to complex tests explaining what is being validated.

Don't
- Read/write real database file (data/airline_discount.db) in tests.
- Use time.sleep() or datetime.now() without mocking.
- Make network calls (HTTP, external APIs).
- Use random data without seeding (tests must be reproducible).
- Test implementation details (private methods)—prefer public API.
- Create fixtures with class scope unless truly needed (prefer function scope).
- Skip tests without a clear reason and a linked issue.

Testing checklist Copilot should satisfy

Models:
- [ ] fit() accepts valid DataFrame/Series, rejects invalid inputs.
- [ ] predict() raises if called before fit().
- [ ] predict() preserves input DataFrame index.
- [ ] save() creates file, load() restores model with identical predictions.
- [ ] Model beats DummyRegressor baseline on MAE and R².
- [ ] build_features() returns required columns in correct order.
- [ ] build_features() converts miles to km when distance_km missing.
- [ ] build_features() derives avg_spend from total_spend/trips if needed.

Data:
- [ ] Database initialization creates schema (passengers, routes, discounts).
- [ ] Foreign key constraints enforced (discounts references passengers/routes).
- [ ] load_all() loads correct counts from JSON file.
- [ ] load_all() with clear_existing=True empties tables first.
- [ ] get_connection() returns valid connection.

Agents:
- [ ] calculate_discount() returns numeric discount within valid range.
- [ ] analyze_route() handles missing route_id gracefully.
- [ ] Functions raise clear exceptions for invalid inputs.

Baseline comparison pattern:
```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def test_model_outperforms_baseline(synthetic_data):
    """Test model beats DummyRegressor baseline on MAE and R²."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_r2 = r2_score(y_test, baseline_preds)
    
    # Model
    model = DiscountPredictor()
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, model_preds)
    model_r2 = r2_score(y_test, model_preds)
    
    # Assertions
    assert model_mae < baseline_mae, f"Model MAE {model_mae:.3f} should be < baseline {baseline_mae:.3f}"
    assert model_r2 > baseline_r2, f"Model R² {model_r2:.3f} should be > baseline {baseline_r2:.3f}"
```

Commands (for trainees)
- Run all tests: `pytest tests/ -v` or `make test`
- Run specific test file: `pytest tests/models/test_discount_predictor.py -v`
- Run specific test: `pytest tests/models/test_discount_predictor.py::test_fit_validates_empty_X -v`
- Run with coverage: `pytest tests/ --cov=src --cov-report=html` or `make test-cov`
- Run only failed tests: `pytest --lf`
- Run in parallel: `pytest -n auto` (requires pytest-xdist)

Configuration (pytest.ini at airline-discount-ml/pytest.ini)
```ini
[pytest]
testpaths = tests
pythonpath = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

VS Code Test Explorer setup
- Configured via .vscode/settings.json
- Test icons appear in sidebar and editor gutter
- Run/debug individual tests with one click
- View test output inline
- Use `/setupTests` slash command to auto-configure

Notes for Copilot
- Prefer pytest fixtures over setup/teardown methods.
- Use @pytest.mark.parametrize for multiple input variations.
- For pandas: always check index, column names, dtypes, and values.
- For models: test error messages explicitly with pytest.raises(match="...").
- For database: use :memory: or tempfile, never shared test database.
- Tests should complete in <5 seconds total (aim for <100ms per test).
- If a test needs >1s, mark with @pytest.mark.slow.

Example test structure:
```python
"""
Unit tests for <module_name>.

Validates:
- <key functionality 1>
- <key functionality 2>
- Error handling and validation
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import DiscountPredictor


@pytest.fixture
def synthetic_data():
    """Create minimal synthetic dataset for testing."""
    np.random.seed(42)
    # ... fixture implementation
    return X, y


def test_function_name_success_case(synthetic_data):
    """Test function succeeds with valid input."""
    # Arrange
    X, y = synthetic_data
    model = DiscountPredictor()
    
    # Act
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Assert
    assert isinstance(predictions, pd.Series)
    assert predictions.index.equals(X.index)
    assert len(predictions) == len(X)


def test_function_name_error_case():
    """Test function raises ValueError on invalid input."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        model.fit([], [])
```

Acceptance criteria for Copilot-generated tests
- [ ] Test file mirrors src/ structure (tests/models/ for src/models/).
- [ ] All tests have descriptive names and docstrings.
- [ ] Fixtures used for shared test data with np.random.seed(42).
- [ ] Tests are independent (no shared state).
- [ ] Tests complete in <5 seconds total.
- [ ] No external I/O (network, filesystem except tempfile).
- [ ] Error cases use pytest.raises with match parameter.
- [ ] Pandas operations verified with pd.testing.assert_*.
- [ ] Baseline comparison tests pass (model beats DummyRegressor).
- [ ] All tests pass when run with `pytest tests/ -v`.
