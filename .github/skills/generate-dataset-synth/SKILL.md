---
name: generate-dataset-synth
description: Instructions for generating synthetic airline data with Synth CLI and loading it into SQLite.
---

# Synth Data Generation

When the user wants to generate synthetic data, **do not** run raw synth commands. Use the included Python script which handles generation, JSON processing, and database loading automatically.

## Prerequisites

1. **Synth CLI must be installed**: Run `synth version` to verify.
   - If not installed, use `@install-synth` agent or run:
     ```bash
     python3 .github/skills/install-synth/install_synth.py
     ```

2. **Database must be initialized**: Run `make db-init` if tables don't exist.

## Standard Generation Procedure

Run the following commands in order:

1. **Navigate to Project Root**:
   ```bash
   cd airline-discount-ml
   ```

2. **Run the Generation Script**:
   This script generates data using Synth, processes the JSON, and loads it into SQLite.

   ```bash
   python3 ../.github/skills/generate-dataset-synth/run_synth.py
   ```

   **With custom record count** (default is 100 per table):
   ```bash
   python3 ../.github/skills/generate-dataset-synth/run_synth.py --count 500
   ```

   **Generate only (no database load)**:
   ```bash
   python3 ../.github/skills/generate-dataset-synth/run_synth.py --no-load
   ```

3. **Verify Data**:
   ```bash
   sqlite3 data/airline_discount.db "SELECT COUNT(*) FROM passengers;"
   sqlite3 data/airline_discount.db "SELECT COUNT(*) FROM routes;"
   sqlite3 data/airline_discount.db "SELECT COUNT(*) FROM discounts;"
   ```

## Synth Models Location

The Synth schema files are located in:
```
airline-discount-ml/synth_models/airline_data/
├── passengers.json    # Passenger profiles with travel history
├── routes.json        # Flight routes with origin/destination/distance
└── discounts.json     # Discount records linking passengers to routes
```

## Generated Data Schema

### passengers
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Faker-generated name |
| travel_history | TEXT | JSON with trips count and total_spend |

### routes
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| origin | TEXT | 3-letter airport code |
| destination | TEXT | 3-letter airport code |
| distance | REAL | Distance in miles |

### discounts
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| passenger_id | INTEGER | FK to passengers |
| route_id | INTEGER | FK to routes |
| discount_value | REAL | Discount percentage |

## Troubleshooting

### synth: command not found
- Run `@install-synth` or install manually per SKILL.md in install-synth.

### "no such table" error
- Run `make db-init` to create tables first.

### JSON parse errors
- Ensure Synth version is 0.6.9 (`synth version`).
- Check that synth_models/*.json files are valid JSON.
