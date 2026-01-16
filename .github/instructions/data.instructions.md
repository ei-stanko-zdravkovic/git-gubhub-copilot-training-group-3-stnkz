---
applyTo: "airline-discount-ml/src/data/**/*.py,src/data/**/*.py"
description: Instructions for SQLite data access layer and preprocessing
---

# Copilot Instructions for src/data

Purpose
- Define safe, consistent patterns for SQLite operations and data preprocessing used by models and agents.

Scope (this folder)
- `database.py`: connection handling, schema init, query helpers.
- `preprocessor.py`: DataFrame transformations prior to modeling.
- `load_synthetic_data.py`: loading JSON-based synthetic data into SQLite.

Hard rules
- Use `Database` or `get_connection()` from `src/data/database.py`; do not call `sqlite3` directly from outside `src/data/`.
- Initialize schema via `init_database()`; keep schema in `data/schema.sql` and sample data in `data/sample_data.sql`.
- Prefer context managers for connections: `with get_connection() as conn:` then commit/rollback appropriately.
- Return `pandas.DataFrame` or simple types from helpers; avoid leaking raw cursors to other layers.
- No PII exposure beyond schema-defined fields; keep transformations schema-driven.

Patterns
- Transactions: group related inserts/updates in a single transaction; rollback on exceptions.
- Queries: parameterize SQL (no string concatenation) and convert results to DataFrames with explicit dtypes.
- Preprocessing: keep functions pure where possible; avoid global state. Handle missing values with clear, documented defaults.
- Indexing: preserve DataFrame indexes when preparing features used by models.

Testing & dev
- Use in-memory SQLite (`:memory:`) or temporary files for tests; never write to `data/airline_discount.db` in tests.
- Seed randomness (`random.seed(42)`, `np.random.seed(42)`) when generating synthetic data.
- Validate foreign key constraints and record counts when loading synthetic data.

Error handling
- Raise clear exceptions for schema violations (e.g., `sqlite3.IntegrityError` propagated) and invalid inputs.
- Wrap DB operations with try/except to ensure connections close and transactions rollback.

Examples
- Schema & sample data: `data/schema.sql`, `data/sample_data.sql`.
- Entry points: `src/data/database.py:init_database`, `src/data/load_synthetic_data.py`.

Notes
- Keep data access isolated: other layers (models/agents) import high-level helpers, not `sqlite3` or raw SQL.
