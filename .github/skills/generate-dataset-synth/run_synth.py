#!/usr/bin/env python3
"""
Synth data generation and database loading script.

Generates synthetic airline data using Synth CLI and loads it into SQLite.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find the airline-discount-ml project root."""
    # Try to find relative to this script
    script_dir = Path(__file__).parent
    # Go up from .github/skills/run-synth to repo root, then into airline-discount-ml
    repo_root = script_dir.parent.parent.parent
    project_root = repo_root / "airline-discount-ml"
    if project_root.exists():
        return project_root
    # Fallback: current directory
    cwd = Path.cwd()
    if (cwd / "synth_models").exists():
        return cwd
    if (cwd.parent / "synth_models").exists():
        return cwd.parent
    raise FileNotFoundError("Could not find airline-discount-ml project root")


def check_synth_installed() -> bool:
    """Check if Synth CLI is installed and accessible."""
    if shutil.which("synth"):
        return True
    # Check common install location
    synth_home = Path.home() / ".synth" / "bin" / "synth"
    if synth_home.exists():
        return True
    return False


def get_synth_path() -> str:
    """Get the path to Synth executable."""
    if shutil.which("synth"):
        return "synth"
    synth_home = Path.home() / ".synth" / "bin" / "synth"
    if synth_home.exists():
        return str(synth_home)
    raise FileNotFoundError("Synth CLI not found. Run @install-synth first.")


def generate_all_data(project_root: Path, count: int) -> dict:
    """
    Generate synthetic data for all collections using Synth CLI.
    
    Args:
        project_root: Path to airline-discount-ml
        count: Number of records to generate per collection
        
    Returns:
        Dict with passengers, routes, discounts lists
    """
    synth_path = get_synth_path()
    synth_models = project_root / "synth_models" / "airline_data"
    
    print(f"📊 Generating {count} records per collection...")
    
    try:
        result = subprocess.run(
            [synth_path, "generate", str(synth_models), "--size", str(count)],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        print(f"✓ Generated: {len(data.get('passengers', []))} passengers, "
              f"{len(data.get('routes', []))} routes, "
              f"{len(data.get('discounts', []))} discounts")
        return data
    except subprocess.CalledProcessError as e:
        print(f"✗ Synth generation failed: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse Synth output: {e}")
        raise


def load_passengers(db, records: list) -> None:
    """Load passenger records into database."""
    cursor = db.connection.cursor()
    for rec in records:
        travel_history = json.dumps(rec.get("travel_history", {}))
        cursor.execute(
            "INSERT INTO passengers (name, travel_history) VALUES (?, ?)",
            (rec["name"], travel_history)
        )
    db.connection.commit()
    print(f"✓ Loaded {len(records)} passengers into database")


def load_routes(db, records: list) -> None:
    """Load route records into database."""
    cursor = db.connection.cursor()
    for rec in records:
        cursor.execute(
            "INSERT INTO routes (origin, destination, distance) VALUES (?, ?, ?)",
            (rec["origin"], rec["destination"], rec["distance"])
        )
    db.connection.commit()
    print(f"✓ Loaded {len(records)} routes into database")


def load_discounts(db, passenger_count: int, route_count: int, records: list) -> None:
    """Load discount records into database, linking to existing passengers/routes."""
    import random
    random.seed(42)
    
    cursor = db.connection.cursor()
    for i, rec in enumerate(records):
        # Assign to random existing passenger and route
        passenger_id = (i % passenger_count) + 1
        route_id = (i % route_count) + 1
        cursor.execute(
            "INSERT INTO discounts (passenger_id, route_id, discount_value) VALUES (?, ?, ?)",
            (passenger_id, route_id, rec["discount_value"])
        )
    db.connection.commit()
    print(f"✓ Loaded {len(records)} discounts into database")


def clear_tables(db) -> None:
    """Clear existing data from tables."""
    cursor = db.connection.cursor()
    cursor.execute("DELETE FROM discounts")
    cursor.execute("DELETE FROM routes")
    cursor.execute("DELETE FROM passengers")
    db.connection.commit()
    print("✓ Cleared existing data from tables")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and load synthetic airline data")
    parser.add_argument("--count", type=int, default=100, help="Number of records per table (default: 100)")
    parser.add_argument("--no-load", action="store_true", help="Generate only, don't load into database")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing data before loading")
    args = parser.parse_args()

    # Check Synth is installed
    if not check_synth_installed():
        print("✗ Synth CLI not found.")
        print("  Install with: python3 ../.github/skills/install-synth/install_synth.py")
        print("  Or use: @install-synth agent")
        return 1

    try:
        project_root = find_project_root()
        print(f"📁 Project root: {project_root}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1

    # Generate data
    try:
        data = generate_all_data(project_root, args.count)
        passengers = data.get("passengers", [])
        routes = data.get("routes", [])
        discounts = data.get("discounts", [])
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return 1

    if args.no_load:
        print("\n✅ Data generated (not loaded into database)")
        print(f"   Passengers: {len(passengers)}")
        print(f"   Routes: {len(routes)}")
        print(f"   Discounts: {len(discounts)}")
        return 0

    # Load into database
    print("\n💾 Loading data into SQLite...")
    
    # Add project src to path for imports
    sys.path.insert(0, str(project_root))
    
    try:
        from src.data.database import Database
    except ImportError as e:
        print(f"✗ Failed to import database module: {e}")
        print("  Make sure you're in the airline-discount-ml directory")
        print("  and have run: pip install -e .")
        return 1

    db = Database(project_root / "data" / "airline_discount.db")
    db.connect()

    try:
        if not args.no_clear:
            clear_tables(db)
        
        load_passengers(db, passengers)
        load_routes(db, routes)
        load_discounts(db, len(passengers), len(routes), discounts)
        
        print("\n✅ Synthetic data generation complete!")
        print(f"   Passengers: {len(passengers)}")
        print(f"   Routes: {len(routes)}")
        print(f"   Discounts: {len(discounts)}")
        return 0
    except Exception as e:
        print(f"✗ Database load failed: {e}")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
