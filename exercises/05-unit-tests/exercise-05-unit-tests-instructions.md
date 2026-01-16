# Exercise 05 — Unit Tests with Skills

## Learning Objectives
- Use the unit-testing skill to guide test creation
- Write deterministic, fast unit tests using pytest
- Practice test-driven development with Copilot
- Create tests for models, data access, and agents

---

## What You'll Learn

In this exercise, you'll use a **skill-based approach** to unit testing. Instead of memorizing pytest commands and patterns, you'll:

1. Reference the `unit-testing` skill for guidance
2. Use Copilot Chat to generate tests following the skill's best practices
3. Run and debug tests with proper fixtures and mocks

---

## Prerequisites

- Completed Exercise 01 (setup and instructions)
- Python environment activated
- VS Code with Copilot Chat extensions

---

## Task 1: Examine the Unit Testing Skill

**Goal:** Understand the skill that guides test creation for this project.

### Instructions

**Step 1:** Open and examine the skill file:

```bash
cat .github/skills/unit-testing/SKILL.md
```

Notice:
- **When to use** - Creating, running, or debugging tests
- **Test structure** - Mirrors the `src/` directory
- **Test template** - Standard Arrange-Act-Assert pattern
- **Running tests** - Commands for different scenarios
- **Best practices** - Module-specific guidelines (models, data, agents)
- **Troubleshooting** - Common issues and solutions

**Step 2:** Understand the project test structure:

```bash
cd airline-discount-ml
tree tests/
```

You should see:
```
tests/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── test_discount_predictor.py
│   └── test_passenger_profiler.py
├── data/
│   ├── __init__.py
│   └── test_database.py
└── agents/
    ├── __init__.py
    └── test_discount_agent.py
```

### Acceptance Criteria

- [ ] You understand the skill's test template structure
- [ ] You know the difference between testing models, data, and agents
- [ ] You can identify best practices for each module type

---

## Task 2: Configure pytest (if not done)

**Goal:** Set up pytest configuration using the `/setupTests` workspace agent.

### Instructions

**Step 1:** Check if pytest is already configured:

```bash
cd airline-discount-ml
cat pytest.ini
```

If `pytest.ini` exists with proper settings, skip to Task 3.

**Step 2:** Use Copilot Chat to configure pytest:

```
@workspace /setupTests for pytest
Tests live in airline-discount-ml/tests/ directory with subdirectories: airline-discount-ml/tests/models/, airline-discount-ml/tests/data/, airline-discount-ml/tests/agents/.
Use imports from src.* (CWD=airline-discount-ml).
Create pytest.ini at airline-discount-ml/pytest.ini with testpaths=tests.
Update .vscode/settings.json to set python.testing.cwd=airline-discount-ml and extraPaths.
Show the diff before applying
```

**Step 3:** Verify test discovery works:

```bash
cd airline-discount-ml
pytest --collect-only
```

You should see all test files discovered.

### Acceptance Criteria

- [ ] `pytest.ini` exists with correct `testpaths` and `pythonpath`
- [ ] `.vscode/settings.json` configured for Python testing
- [ ] `pytest --collect-only` discovers all tests

---

## Task 3: Run Existing Tests

**Goal:** Run the existing test suite to establish a baseline.

### Instructions

**Step 1:** Run all tests:

```bash
cd airline-discount-ml
pytest tests/ -v
```

**Step 2:** Run tests for a specific module:

```bash
pytest tests/models/ -v
```

**Step 3:** Run a specific test file:

```bash
pytest tests/models/test_discount_predictor.py -v
```

**Step 4:** Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Acceptance Criteria

- [ ] All existing tests pass
- [ ] You understand the test output format
- [ ] Coverage report shows which lines are tested

---

## Task 4: Create Tests Using the Skill

**Goal:** Use Copilot Chat with the unit-testing skill to generate new tests.

### Instructions

**Step 1:** Choose a function to test

Open `airline-discount-ml/src/models/passenger_profiler.py` and find a function that needs more test coverage.

**Step 2:** Ask Copilot to generate tests using the skill

In Copilot Chat, reference the skill:

```
Using the unit-testing skill at .github/skills/unit-testing/, create tests for the build_features function in #file:passenger_profiler.py

Include tests for:
- Happy path with valid passenger data
- Edge case with missing optional columns
- Error case with missing required columns
- Validation that miles are converted to kilometers correctly

Follow the test template from the skill.
```

**Step 3:** Review generated tests

Check that the generated tests:
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names (`test_<function>_<scenario>`)
- Include docstrings
- Test both success and error paths
- Are deterministic (no random values)

**Step 4:** Run the new tests:

```bash
pytest tests/models/test_passenger_profiler.py -v
```

### Acceptance Criteria

- [ ] New tests follow the skill's template structure
- [ ] Tests cover happy path, edge cases, and errors
- [ ] All tests pass
- [ ] Test names are descriptive and follow conventions

---

## Task 5: Test Models with Mocks

**Goal:** Create tests for the agent layer using mocks, following the skill's guidance.

### Instructions

**Step 1:** Review the skill's agent testing section

Look at the "For Agents" best practices in `.github/skills/unit-testing/SKILL.md`.

**Step 2:** Ask Copilot to create agent tests with mocks:

```
Using the unit-testing skill, create tests for #file:discount_agent.py

Follow the skill's guidance for agent testing:
- Mock database and model dependencies
- Test business logic and orchestration
- Verify interactions using unittest.mock

Create tests for:
- Successful discount calculation
- Handling missing passenger data
- Model prediction errors
```

**Step 3:** Review the generated mocks:

Ensure the tests:
- Use `unittest.mock.Mock` or `pytest-mock`
- Mock database calls (no real DB access)
- Mock model predictions
- Verify method calls with `assert_called_once()`

**Step 4:** Run agent tests:

```bash
pytest tests/agents/test_discount_agent.py -v
```

### Acceptance Criteria

- [ ] Tests use mocks for dependencies
- [ ] No real database or model calls
- [ ] Tests verify business logic correctly
- [ ] All mocked interactions are verified

---

## Task 6: Add Tests for Edge Cases

**Goal:** Use Copilot to identify and test edge cases.

### Instructions

**Step 1:** Ask Copilot to find untested paths:

```
Analyze #file:discount_predictor.py and identify edge cases or error paths that lack test coverage.

Reference the unit-testing skill's "Common test scenarios" section.
```

**Step 2:** Generate tests for identified gaps:

```
Using the unit-testing skill, create tests for the edge cases you identified in DiscountPredictor.

Include:
- Empty DataFrame input
- Single-row DataFrame
- Missing required columns
- Invalid data types
```

**Step 3:** Run the new edge case tests:

```bash
pytest tests/models/test_discount_predictor.py::TestDiscountPredictor::test_empty_dataframe -v
```

### Acceptance Criteria

- [ ] Edge cases are identified and documented
- [ ] Tests exist for empty inputs, boundary values, and missing data
- [ ] Error messages are validated in tests
- [ ] All edge case tests pass

---

## Task 7: Use VS Code Test Explorer

**Goal:** Run and debug tests using VS Code's integrated testing UI.

### Instructions

**Step 1:** Open the Testing view:

- Click the beaker icon in the Activity Bar (left sidebar)
- Or press `Cmd+Shift+T` (macOS) / `Ctrl+Shift+T` (Windows)

**Step 2:** Run tests from the UI:

- Click "Run All Tests" to execute the entire suite
- Click the play button next to a specific test to run it
- Use the refresh button to rediscover tests

**Step 3:** Debug a failing test:

1. Set a breakpoint in a test function
2. Right-click the test in Test Explorer
3. Select "Debug Test"
4. Step through the code using the debugger

**Step 4:** View test output:

- Click on a test to see its output
- Use the "Show Output" button for detailed logs

### Acceptance Criteria

- [ ] Test Explorer shows all tests
- [ ] You can run individual tests from the UI
- [ ] Debugging works with breakpoints
- [ ] Test results are clearly visible

---

## Summary

| Concept | Description |
|---------|-------------|
| **Skill** | `.github/skills/unit-testing/SKILL.md` provides testing guidance |
| **Structure** | Tests mirror `src/` directory structure |
| **Template** | Arrange-Act-Assert pattern with descriptive names |
| **Mocks** | Used for agents to isolate business logic |
| **Coverage** | Track with `pytest --cov=src` |

## Best Practices Recap

1. **Target code precisely** - Highlight specific functions to test
2. **Reference the skill** - Use `.github/skills/unit-testing/` for guidance
3. **Be deterministic** - Seed random generators, avoid I/O
4. **Test all paths** - Happy path, edge cases, errors
5. **Use mocks** - Isolate units from dependencies
6. **Run often** - Use Test Explorer for quick feedback

---

## Bonus Challenge

Create a new skill for **integration testing** that:
- Lives at `.github/skills/integration-testing/SKILL.md`
- Describes how to test database + model interactions
- Includes fixture templates for test databases
- Guides on using Docker for isolated test environments
- References the real database schema from `data/schema.sql`
