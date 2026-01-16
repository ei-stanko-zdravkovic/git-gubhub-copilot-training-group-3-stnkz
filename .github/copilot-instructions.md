## GitHub Copilot Instructions — Airline Discount ML

Purpose: Make AI agents productive quickly in this repo. Focus on the ML subproject in airline-discount-ml/.

Big Picture
- Layers in airline-discount-ml/src/: data (SQLite), models (pure ML, no I/O), agents (business logic), training (scripts), mcp (custom tools), utils.
- Data flow: src/data/database.py → pandas DataFrames → src/models → predictions joined back by preserved index.

Architecture Essentials
- Database (src/data/): SQLite at data/airline_discount.db; tables passengers, routes, discounts. Initialize via python -c "from src.data.database import init_database; init_database()". Use Database or get_connection(); no direct sqlite3 in other layers.
- Models (src/models/): Never import src.data.database. Deterministic ML: set random.seed(42); np.random.seed(42); estimators use random_state=42.
  - discount_predictor.py: sklearn Pipeline with ColumnTransformer → LinearRegression. API: fit(X,y), predict(X), save(path), load(path). Required columns: [distance_km, history_trips, avg_spend, route_id, origin, destination]. Numeric: SimpleImputer(median) → StandardScaler. Categorical: SimpleImputer(most_frequent) → OneHotEncoder. predict() returns Series preserving input index; raise RuntimeError if not fitted.
  - passenger_profiler.py: build_features(df) → DataFrame. Convert miles→km (distance*1.60934), derive avg_spend if missing, return only required columns in order, exclude PII (no passenger_id).
- Agents (src/agents/): Classes coordinate DB + models; kept simple for exercises.

Developer Workflows
- From repo root: cd airline-discount-ml
- Setup (macOS/Linux): ./setup.sh; Windows: setup.bat; Manual: pip install -e ".[dev]" then init DB as above.
- Tests: pytest tests/ -v; coverage: make test-cov. Key suites: tests/models/test_discount_predictor.py, tests/models/test_passenger_profiler.py, tests/agents/test_discount_agent.py, tests/data/test_database.py.
- Notebooks: jupyter lab or make run-notebook; kernel name: "Python (airline-discount-ml)". See notebooks/exploratory_analysis.ipynb for DB pattern.
- Common commands: make format (black), make lint (flake8), make train/evaluate or python src/training/train.py|evaluate.py.

Conventions & Gotchas
- Editable install required for imports (setup.py extras include dev tools).
- Validation-first in models: clear ValueError for empty/missing columns; RuntimeError if predict before fit.
- Strict feature schema and index preservation are test-enforced; don’t add PII.
- Keep PR-sized changes: <2 files and <50 lines proceed; otherwise propose a brief plan first. Ask before changing public APIs.

MCP Integration (src/mcp/)
- Tools: query_db, train_model, predict, evaluate — used to interact with DB and models via MCP. Prefer these when integrating agents with external tooling.

Troubleshooting
- No module 'src' in notebooks: prepend sys.path with repo parent as shown in notebooks.
- SQLite table missing: make db-init or call init_database().
- Notebook imports failing: ensure kernel is "Python (airline-discount-ml)".

Reference Files
- Data access: src/data/database.py; Models: src/models/discount_predictor.py, src/models/passenger_profiler.py; Agents: src/agents/; Training: src/training/; Tests: tests/.

Goal: Fast, reproducible iterations with clean boundaries: DB ↔ pure models ↔ agents. If required columns are unclear, pause and ask rather than guessing.
