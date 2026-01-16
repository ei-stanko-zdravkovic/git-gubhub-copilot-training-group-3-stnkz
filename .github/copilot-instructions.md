## GitHub Copilot Instructions — Repository

Purpose: Provide concise, actionable guidance so AI coding agents can be immediately productive here while respecting project boundaries and workflows.

Principles
- Use ATTACHED context only; if context is missing or ambiguous, ask before guessing.
- Prefer minimal diffs; avoid multi-file edits unless those files are provided.
- For >2 files or >50 lines, propose a brief plan and wait for confirmation.
- If a command/tool is required, show the exact command and ask to run it.

Architecture Overview (airline-discount-ml)
- Layers under `airline-discount-ml/src/`: `data/` (SQLite access + preprocessing), `models/` (deterministic ML, no I/O), `agents/` (business orchestration), `training/` (scripts), `mcp/` (tools), `utils/`.
- Data flow: `src/data/database.py` → DataFrames → `src/models/*` → predictions returned as `pandas.Series` preserving input index.
- Boundaries: models never import `src.data.database`; DB access happens only via `Database`/`get_connection()` in `src/data`.

Developer Workflows
- Setup (recommended): `cd airline-discount-ml && ./setup.sh` (macOS/Linux) or `setup.bat` (Windows). Manual: `pip install -e " .[dev]"` then `python -c "from src.data.database import init_database; init_database()"`.
- Tests: `pytest tests/ -v` (or `pytest -q`), coverage: `make test-cov`.
- Format/Lint: `make format` (black), `make lint` (flake8).
- Train/Evaluate: `make train`, `make evaluate` or run the Python scripts in `src/training/`.

Project-Specific Patterns
- Determinism in models: set seeds (`random.seed(42)`, `np.random.seed(42)`) and pass `random_state=42`.
- Feature schema (discount model): required columns `distance_km, history_trips, avg_spend, route_id, origin, destination`; validate early and raise `ValueError` on missing/empty inputs.
- `predict(X)` returns a Series named `discount_value` with the same index as `X`; raise `RuntimeError` if predict is called before fit.
- Use `ColumnTransformer` + `Pipeline` for preprocessing (numeric: `SimpleImputer(median)`, `StandardScaler`; categorical: `SimpleImputer(most_frequent)`, `OneHotEncoder`).

Integration & Tools
- MCP tools in `src/mcp/tools.py`: `query_db`, `train_model`, `predict`, `evaluate`. Prefer these for scripted interactions.
- Path-scoped instructions live in `.github/instructions/` and apply via `applyTo` globs (e.g., models: `"airline-discount-ml/src/models/**/*.py,src/models/**/*.py"`; data: `"airline-discount-ml/src/data/**/*.py,src/data/**/*.py"`).

Clarifications
- If required columns or DB tables are unspecified in a prompt, ask one clarifying question rather than inventing names or schemas.

Key References
- Data access: `airline-discount-ml/src/data/database.py`, `airline-discount-ml/data/schema.sql`
- Models: `airline-discount-ml/src/models/discount_predictor.py`, `airline-discount-ml/src/models/passenger_profiler.py`
- Agents: `airline-discount-ml/src/agents/discount_agent.py`, `airline-discount-ml/src/agents/route_analyzer.py`
- Training: `airline-discount-ml/src/training/` and `airline-discount-ml/Makefile`
