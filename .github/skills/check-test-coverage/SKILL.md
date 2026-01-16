---
name: check-test-coverage
description: Standard procedure for checking test coverage using pytest-cov in the airline-discount-ml project.
---

# Checking Test Coverage

When the user wants to check test coverage, use pytest with the coverage plugin to generate detailed reports.

## Prerequisites

1. **pytest-cov must be installed**:
   ```bash
   cd airline-discount-ml
   pip install pytest-cov
   ```

2. **Tests must exist** in the tests/ directory

## Standard Coverage Procedures

### Coverage for All Tests
```bash
cd airline-discount-ml
pytest --cov=src tests/ --cov-report=term-missing
# or
make test-cov
```

### Coverage for Specific Module
```bash
cd airline-discount-ml
pytest --cov=src.models tests/models/ --cov-report=term-missing
pytest --cov=src.data tests/data/ --cov-report=term-missing
pytest --cov=src.training tests/training/ --cov-report=term-missing
```

### Coverage for Specific File
```bash
cd airline-discount-ml
pytest --cov=src.models.discount_predictor tests/models/test_discount_predictor.py --cov-report=term-missing
```

## Coverage Report Formats

### Terminal Report (Default)
Shows coverage percentage per file:
```bash
pytest --cov=src tests/ --cov-report=term
```

Output:
```
---------- coverage: platform linux, python 3.11.9 -----------
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
src/__init__.py                             2      0   100%
src/models/__init__.py                      3      0   100%
src/models/discount_predictor.py           85      5    94%
src/models/passenger_profiler.py           42      2    95%
-----------------------------------------------------------
TOTAL                                     132      7    95%
```

### Terminal with Missing Lines
Shows which lines are not covered:
```bash
pytest --cov=src tests/ --cov-report=term-missing
```

Output:
```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/models/discount_predictor.py           85      5    94%   45-47, 62, 78
src/models/passenger_profiler.py           42      2    95%   23, 67
```

### HTML Report (Interactive)
Generates detailed HTML coverage report:
```bash
pytest --cov=src tests/ --cov-report=html
```

View report:
```bash
# Open in browser
open htmlcov/index.html          # macOS
xdg-open htmlcov/index.html      # Linux
start htmlcov/index.html         # Windows
```

### XML Report (CI/CD)
For continuous integration tools:
```bash
pytest --cov=src tests/ --cov-report=xml
```

Generates `coverage.xml` for tools like CodeCov, Coveralls.

### Multiple Report Formats
Combine multiple report types:
```bash
pytest --cov=src tests/ --cov-report=term-missing --cov-report=html --cov-report=xml
```

## Coverage Thresholds

### Fail Build if Coverage Too Low
```bash
pytest --cov=src tests/ --cov-fail-under=80
```

Returns non-zero exit code if coverage < 80%.

### Check Minimum Coverage Per File
```bash
pytest --cov=src tests/ --cov-report=term-missing --cov-fail-under=90
```

## Advanced Coverage Options

### Branch Coverage
Check both line and branch coverage:
```bash
pytest --cov=src tests/ --cov-branch --cov-report=term-missing
```

Shows coverage of conditional branches (if/else, try/except):
```
Name                              Stmts   Miss Branch BrPart  Cover
-------------------------------------------------------------------
src/models/discount_predictor.py     85      5     24      3    92%
```

### Focus on New Code
Only show coverage for modified files:
```bash
pytest --cov=src --cov-report=term-missing $(git diff --name-only HEAD | grep '\.py$' | sed 's/^/tests\//; s/src/test/; s/\.py$/_test.py/')
```

### Exclude Patterns
Exclude specific files or directories:
```bash
pytest --cov=src --cov-report=term-missing --cov-config=.coveragerc tests/
```

Create `.coveragerc`:
```ini
[run]
omit = 
    */tests/*
    */migrations/*
    */__init__.py
```

## Expected Output

### High Coverage (Good)
```
---------- coverage: platform linux, python 3.11.9 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/models/discount_predictor.py           85      2    98%   45, 62
src/models/passenger_profiler.py           42      0   100%
src/training/train.py                      63      4    94%   23-26
-------------------------------------------------------------------
TOTAL                                     190      6    97%
```

### Low Coverage (Needs Attention)
```
---------- coverage: platform linux, python 3.11.9 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/models/discount_predictor.py           85     35    59%   12-45, 62-78, 92-105
-------------------------------------------------------------------
TOTAL                                      85     35    59%
```

## Coverage Targets

### Project Standards
- **Models:** >95% line coverage, >85% branch coverage
- **Data:** >90% line coverage (use mocks for DB)
- **Training:** >85% line coverage
- **Agents:** >80% line coverage
- **Overall:** >90% line coverage

### Untestable Code
Mark code that shouldn't be covered:
```python
def debug_only_function():  # pragma: no cover
    """Only used for manual debugging."""
    import pdb; pdb.set_trace()
```

## Troubleshooting

### No Coverage Data Collected
**Solution**: Ensure pytest-cov is installed and tests are actually running:
```bash
pip install pytest-cov
pytest --cov=src tests/ -v  # Verbose to see test execution
```

### Coverage Shows 0% for All Files
**Solution**: Check that module paths are correct:
```bash
# Wrong (from project root)
pytest --cov=airline-discount-ml/src tests/

# Right
cd airline-discount-ml
pytest --cov=src tests/
```

### HTML Report Not Generating
**Solution**: Ensure output directory is writable:
```bash
rm -rf htmlcov/
pytest --cov=src tests/ --cov-report=html
ls -la htmlcov/  # Verify files created
```

## Integration with Make

Add to `Makefile`:
```makefile
test-cov:
	pytest --cov=src tests/ --cov-report=term-missing --cov-report=html

test-cov-fail:
	pytest --cov=src tests/ --cov-fail-under=90
```

Run:
```bash
make test-cov
make test-cov-fail
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest --cov=src tests/` | Basic coverage report |
| `pytest --cov=src tests/ --cov-report=term-missing` | Show missing lines |
| `pytest --cov=src tests/ --cov-report=html` | HTML report |
| `pytest --cov=src tests/ --cov-branch` | Include branch coverage |
| `pytest --cov=src tests/ --cov-fail-under=90` | Fail if coverage < 90% |
| `make test-cov` | Project standard coverage command |
