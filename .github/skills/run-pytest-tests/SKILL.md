---
name: run-pytest-tests
description: Standard procedure for running pytest tests in the airline-discount-ml project with proper Python environment configuration.
---

# Running Pytest Tests

When the user wants to run tests, **always** use the configured Python environment and follow these standardized procedures.

## Prerequisites

1. **Python environment must be configured**:
   ```bash
   cd airline-discount-ml
   # Ensure .venv or venv is activated
   ```

2. **Dependencies must be installed**:
   ```bash
   pip install -e ".[dev]"
   # or
   pip install -r requirements.txt
   ```

## Standard Test Execution Procedures

### Run All Tests
```bash
cd airline-discount-ml
pytest tests/ -v
# or
make test
```

### Run Specific Test File
```bash
cd airline-discount-ml
pytest tests/models/test_discount_predictor.py -v
```

### Run Specific Test Function
```bash
cd airline-discount-ml
pytest tests/models/test_discount_predictor.py::test_fit_validates_empty_X -v
```

### Run Tests Matching Pattern
Use `-k` flag to run tests matching a pattern:
```bash
cd airline-discount-ml
pytest -k "validate" -v         # All tests with "validate" in name
pytest -k "discount_predictor" -v   # All tests in discount_predictor
pytest -k "not slow" -v         # Exclude tests marked as slow
```

### Run Only Failed Tests
```bash
cd airline-discount-ml
pytest --lf                      # Last failed
pytest --ff                      # Failed first, then rest
```

### Quiet Mode (Summary Only)
```bash
cd airline-discount-ml
pytest tests/ -q                 # Minimal output
pytest tests/models/ --tb=no -q  # No traceback, quiet
```

## Test Output Formats

### Verbose Mode (`-v`)
Shows individual test names and PASS/FAIL status:
```
tests/models/test_discount_predictor.py::test_fit_validates_empty_X PASSED
tests/models/test_discount_predictor.py::test_predict_before_fit_raises PASSED
```

### Very Verbose Mode (`-vv`)
Shows test names with full assertion details:
```bash
pytest tests/ -vv
```

### Show Print Statements (`-s`)
Displays print/stdout from tests:
```bash
pytest tests/models/test_discount_predictor.py -s
```

## Test Selection by Markers

### Run Only Integration Tests
```bash
pytest -m integration -v
```

### Exclude Slow Tests
```bash
pytest -m "not slow" -v
```

## Expected Output

### Successful Run
```
======================== test session starts =========================
platform linux -- Python 3.11.9, pytest-9.0.2, pluggy-1.6.0
rootdir: /path/to/airline-discount-ml
configfile: pytest.ini
collected 42 items

tests/models/test_discount_predictor.py ............    [ 28%]
tests/models/test_passenger_profiler.py ............ [ 57%]
tests/training/test_train.py ..........              [ 81%]
tests/training/test_evaluate.py ........             [100%]

========================= 42 passed in 6.32s =========================
```

### Failed Test
```
FAILED tests/models/test_discount_predictor.py::test_example - AssertionError: ...
```

## Troubleshooting

### ModuleNotFoundError: No module named 'src'
**Solution**: Ensure you're in the `airline-discount-ml` directory and Python can find the src package:
```bash
cd airline-discount-ml
export PYTHONPATH=.
pytest tests/ -v
```

Or install in development mode:
```bash
pip install -e .
```

### Tests Hang or Timeout
**Solution**: Use timeout or run in isolated mode:
```bash
pytest tests/ --timeout=10     # 10 second timeout per test
pytest tests/ -n auto          # Parallel execution (requires pytest-xdist)
```

### Import Errors in Test Files
**Solution**: Check that:
1. `__init__.py` exists in tests/ subdirectories
2. Imports use absolute paths: `from src.models import DiscountPredictor`
3. Tests are in correct location (tests/ folder structure mirrors src/)

### Pytest Not Found
**Solution**: Install pytest in the correct environment:
```bash
cd airline-discount-ml
source venv/bin/activate  # or .venv/bin/activate
pip install pytest pytest-cov
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest tests/ -v` | Run all tests verbosely |
| `pytest tests/models/ -v` | Run all model tests |
| `pytest -k "validate"` | Run tests matching pattern |
| `pytest --lf` | Re-run last failed tests |
| `pytest tests/ -q` | Quick quiet run |
| `pytest --collect-only` | List tests without running |
| `make test` | Project standard test command |

## Configuration

Tests are configured via `pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
```
