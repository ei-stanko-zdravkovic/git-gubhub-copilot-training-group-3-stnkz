---
description: Instructions for Jupyter notebooks in the airline-discount-ml project
applyTo: airline-discount-ml/notebooks/**/*.ipynb,notebooks/**/*.ipynb
---

## Notebook Standards for Airline Discount ML

### Setup Requirements

- **First executable cell** must add project root to sys.path:
  ```python
  import sys
  from pathlib import Path
  
  # Add project root to path
  project_root = Path().resolve().parent
  sys.path.insert(0, str(project_root))
  ```

- **Import from src modules** using absolute imports:
  ```python
  from src.data.database import get_connection
  from src.models.discount_predictor import DiscountPredictor
  from src.models.passenger_profiler import build_features
  ```

### Database Access

- **Use get_connection()** for database access, never hard-code paths:
  ```python
  from src.data.database import get_connection
  
  db = get_connection()
  database_connection = db.connection
  ```

- **SQL queries** should use pandas for data loading:
  ```python
  data = pd.read_sql_query('SELECT * FROM passengers', con=database_connection)
  ```

- **JOIN queries** should be explicit about table relationships:
  ```python
  query = """
  SELECT 
      p.id as passenger_id,
      r.origin,
      r.destination,
      d.discount_value
  FROM discounts d
  JOIN passengers p ON d.passenger_id = p.id
  JOIN routes r ON d.route_id = r.id
  """
  ```

### Visualization Standards

- **Set seaborn style** at the beginning:
  ```python
  import seaborn as sns
  sns.set(style='whitegrid')
  ```

- **Figure sizing** should be consistent (10, 6 for standard plots):
  ```python
  plt.figure(figsize=(10, 6))
  ```

- **Always include grids** for readability:
  ```python
  plt.grid(True, alpha=0.3)
  ```

- **Use descriptive titles** and axis labels:
  ```python
  plt.title('Route Distance vs Discount Value')
  plt.xlabel('Distance (miles)')
  plt.ylabel('Discount Value (%)')
  ```

### Documentation

- **Start with markdown cell** explaining the notebook's purpose
- **Use markdown cells** to section analysis steps (e.g., "## Data Loading", "## Visualization")
- **End with markdown cell** summarizing conclusions
- **Print confirmations** for successful operations:
  ```python
  print("✓ Connected to database successfully")
  print(f"Loaded {len(data)} records")
  ```

### Data Exploration

- **Show dataset shape** and structure:
  ```python
  print("Dataset shape:", data.shape)
  print("\nColumn names:", data.columns.tolist())
  print("\nData types:")
  print(data.dtypes)
  ```

- **Use .head()** to preview data after loading
- **Group statistics** should use .describe() or .groupby():
  ```python
  data.groupby(['origin', 'destination'])['discount_value'].describe()
  ```

### Best Practices

- **Clear outputs before committing** to keep notebooks clean
- **Run cells in order** - notebooks should execute top-to-bottom without errors
- **No hard-coded paths** - use Path().resolve() or config
- **Consistent variable naming**:
  - `data` for main DataFrame
  - `discount_data` for discount-specific data
  - `database_connection` for DB connections
  - `db` for Database instance

### Testing Models in Notebooks

When testing models interactively:

```python
from src.models.discount_predictor import DiscountPredictor
from src.models.passenger_profiler import build_features

# Build features from raw data
features = build_features(data)

# Train model
model = DiscountPredictor()
model.fit(features, y)

# Make predictions
predictions = model.predict(features)
```

### Avoid

- ❌ Direct imports like `from database import get_connection`
- ❌ Relative imports like `from ..src.data import database`
- ❌ Hard-coded file paths like `/home/user/project/data.db`
- ❌ Uncommitted outputs (clear before commit)
- ❌ Cells that depend on out-of-order execution
- ❌ Prints without context (use descriptive messages)
