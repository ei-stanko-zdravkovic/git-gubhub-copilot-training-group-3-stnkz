---
name: generate-test-suite
description: Standard procedure for generating comprehensive pytest test suites following airline-discount-ml project patterns and test instructions.
---

# Generating Test Suites

When the user requests test generation, follow this systematic 6-step workflow to create comprehensive, project-compliant test suites.

## Prerequisites

1. **Test instructions must be available**:
   - Read `.github/instructions/tests.instructions.md` first
   - Understand project test patterns and standards

2. **Source code must be available**:
   - The module/function to test must exist
   - Understand the code's purpose and behavior

3. **conftest.py must be checked**:
   - Review existing fixtures before creating new ones
   - Reuse shared fixtures when possible

## 6-Step Test Generation Workflow

### Step 1: List Test Cases
**ALWAYS list test cases BEFORE generating any code.**

Create a comprehensive list covering:
- ✅ Valid inputs (happy path)
- ✅ Edge cases (empty, zero, boundary values)
- ✅ Invalid inputs (wrong types, None, negative)
- ✅ Error conditions (exceptions, validation failures)
- ✅ Integration (interactions with dependencies)

Example:
```markdown
## Test Cases for DiscountPredictor.fit()

1. Valid training (happy path)
   - Fit with valid X (DataFrame) and y (Series)
   - Verify model is fitted (has `model_` attribute)

2. Input validation
   - Empty X raises ValueError
   - Empty y raises ValueError
   - X/y length mismatch raises ValueError
   - Missing required columns raises ValueError

3. Determinism
   - Same inputs produce same outputs
   - Verify random_state works correctly

4. Edge cases
   - Single row training data
   - All categorical columns identical
```

### Step 2: Check conftest.py for Existing Fixtures
**ALWAYS check before creating new fixtures.**

```bash
# Check project-wide fixtures
cat tests/conftest.py

# Check module-specific fixtures
cat tests/models/conftest.py
cat tests/data/conftest.py
```

Common available fixtures:
- `sample_features` - 3-row DataFrame
- `synthetic_data` - 100-row training set
- `temp_db_path` - Temporary database path
- `mock_database_connection` - Mock DB connection
- `tmp_path` - pytest built-in temporary directory

### Step 3: Add New Fixtures if Needed
**Only create fixtures if they don't exist in conftest.py.**

Add to appropriate location:
- `tests/conftest.py` - Used by multiple modules
- `tests/<module>/conftest.py` - Module-specific

Follow fixture patterns from `generate-test-fixtures` skill.

### Step 4: Generate Test File
**Location:** `tests/<module>/test_<file_under_test>.py`

Follow naming convention:
- Source: `src/models/discount_predictor.py`
- Tests: `tests/models/test_discount_predictor.py`

Template structure:
```python
"""
Tests for src.models.discount_predictor module.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.discount_predictor import DiscountPredictor


class TestDiscountPredictorFit:
    """Tests for DiscountPredictor.fit() method."""
    
    def test_fit_with_valid_data(self, synthetic_data):
        """Test fitting with valid DataFrame and Series."""
        X, y = synthetic_data
        model = DiscountPredictor()
        
        model.fit(X, y)
        
        assert hasattr(model, "model_")
        assert model.model_ is not None
    
    def test_fit_validates_empty_X(self):
        """Test that fit raises ValueError for empty X."""
        X = pd.DataFrame()
        y = pd.Series([1, 2, 3])
        model = DiscountPredictor()
        
        with pytest.raises(ValueError, match="empty"):
            model.fit(X, y)
```

### Step 5: Implement All Test Cases
**One test function per test case from Step 1.**

Use standard patterns:

#### Valid Input Tests
```python
def test_function_with_valid_input(fixture_name):
    """Test function with valid inputs."""
    # Arrange
    input_data = prepare_input(fixture_name)
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert isinstance(result, expected_type)
```

#### Exception Tests
```python
def test_function_raises_on_invalid_input():
    """Test function raises exception for invalid input."""
    invalid_input = None
    
    with pytest.raises(ValueError, match="cannot be None"):
        function_under_test(invalid_input)
```

#### Edge Case Tests
```python
def test_function_handles_edge_case():
    """Test function handles edge case correctly."""
    edge_case_input = create_edge_case()
    
    result = function_under_test(edge_case_input)
    
    assert result == expected_edge_case_result
```

#### Determinism Tests
```python
def test_function_is_deterministic(synthetic_data):
    """Test function produces consistent results."""
    X, y = synthetic_data
    
    result1 = function_under_test(X, y, seed=42)
    result2 = function_under_test(X, y, seed=42)
    
    pd.testing.assert_series_equal(result1, result2)
```

### Step 6: Verify Tests Pass
**Run tests and check coverage.**

```bash
# Run new test file
cd airline-discount-ml
pytest tests/models/test_discount_predictor.py -v

# Check coverage
pytest --cov=src.models.discount_predictor tests/models/test_discount_predictor.py --cov-report=term-missing

# Run all tests
pytest tests/ -v
```

Expected output:
```
tests/models/test_discount_predictor.py::test_fit_with_valid_data PASSED
tests/models/test_discount_predictor.py::test_fit_validates_empty_X PASSED
...
========================= 10 passed in 2.43s =========================

---------- coverage: platform linux, python 3.11.9 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/models/discount_predictor.py           85      3    96%   45, 62, 78
```

## Test Organization Patterns

### Group Tests by Method (Classes)
```python
class TestDiscountPredictorFit:
    """Tests for fit() method."""
    
    def test_fit_valid_data(self, synthetic_data):
        ...
    
    def test_fit_empty_X(self):
        ...


class TestDiscountPredictorPredict:
    """Tests for predict() method."""
    
    def test_predict_after_fit(self, synthetic_data):
        ...
    
    def test_predict_before_fit(self):
        ...
```

### Use Parametrization for Similar Tests
```python
@pytest.mark.parametrize("invalid_input,error_msg", [
    (None, "cannot be None"),
    ([], "cannot be empty"),
    ({}, "must be DataFrame"),
])
def test_fit_validates_input(invalid_input, error_msg):
    """Test fit validates various invalid inputs."""
    model = DiscountPredictor()
    
    with pytest.raises(ValueError, match=error_msg):
        model.fit(invalid_input, pd.Series([1, 2, 3]))
```

## Test Quality Checklist

After generating tests, verify:

- [ ] **Location:** All tests in `tests/` folder, mirroring `src/` structure
- [ ] **Coverage:** >90% line coverage, >85% branch coverage
- [ ] **Documentation:** Docstrings explain what each test verifies
- [ ] **Independence:** Tests can run in any order
- [ ] **Speed:** Individual tests complete in <100ms
- [ ] **Determinism:** Tests produce consistent results (use seed=42)
- [ ] **Fixtures:** Reuse from conftest.py, create new only if needed
- [ ] **Assertions:** Clear, specific assertions with helpful error messages
- [ ] **Naming:** test_<method>_<scenario> pattern
- [ ] **Cleanup:** No side effects, temp files cleaned up

## Common Test Patterns

### Model Tests (src/models/*)
```python
def test_model_fit_predict_integration(synthetic_data):
    """Test complete fit → predict workflow."""
    X, y = synthetic_data
    model = ModelClass()
    
    # Fit
    model.fit(X, y)
    
    # Predict
    predictions = model.predict(X)
    
    # Verify
    assert len(predictions) == len(X)
    assert predictions.index.equals(X.index)
```

### Data Tests (src/data/*)
```python
def test_database_query(mock_database_connection):
    """Test database query returns expected format."""
    from src.data.database import fetch_passengers
    
    # Mock return value
    mock_database_connection.execute.return_value = [
        (1, "John", 25),
        (2, "Jane", 30)
    ]
    
    result = fetch_passengers(mock_database_connection)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
```

### Training Tests (src/training/*)
```python
def test_train_saves_model(tmp_path, synthetic_data):
    """Test training saves model to file."""
    from src.training.train import train_model
    
    X, y = synthetic_data
    model_path = tmp_path / "model.pkl"
    
    train_model(X, y, output_path=model_path)
    
    assert model_path.exists()
```

## Troubleshooting

### Tests Failing After Generation
**Solution:** Check imports and fixture names:
```python
# Wrong
from models.discount_predictor import DiscountPredictor  # ModuleNotFoundError

# Right
from src.models.discount_predictor import DiscountPredictor
```

### Fixture Not Found
**Solution:** Check conftest.py exists and is in correct location:
```bash
# Ensure __init__.py exists
ls tests/__init__.py
ls tests/models/__init__.py

# Check fixture is in scope
cat tests/conftest.py | grep "def fixture_name"
```

### Coverage Lower Than Expected
**Solution:** Add tests for missing branches:
```bash
# Identify missing lines
pytest --cov=src.models.discount_predictor tests/models/test_discount_predictor.py --cov-report=term-missing

# Add tests for reported missing lines
```

## Quick Reference

| Step | Action | Tool/Command |
|------|--------|--------------|
| 1 | List test cases | Document in PR or issue |
| 2 | Check fixtures | `cat tests/conftest.py` |
| 3 | Add fixtures | Edit conftest.py |
| 4 | Generate test file | Create `tests/<module>/test_*.py` |
| 5 | Implement tests | Follow AAA pattern |
| 6 | Verify | `pytest tests/ -v --cov` |

## Example Workflow

```bash
# Step 1: Read source code
cat src/models/discount_predictor.py

# Step 2: Check conftest
cat tests/conftest.py

# Step 3: Create test file
touch tests/models/test_discount_predictor.py

# Step 4: Edit test file (add imports, test functions)
# ... implement tests following patterns above

# Step 5: Run tests
pytest tests/models/test_discount_predictor.py -v

# Step 6: Check coverage
pytest --cov=src.models.discount_predictor tests/models/test_discount_predictor.py --cov-report=term-missing
```
