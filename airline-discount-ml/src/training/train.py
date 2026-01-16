from pathlib import Path
import pandas as pd

from models.discount_predictor import DiscountPredictor
from data.database import get_connection


def train_model(model, data):
    """
    Train the machine learning model with the provided data.

    Parameters:
    model: The machine learning model to be trained.
    data: The training data used for training the model.

    Returns:
    The trained model.
    """
    X = data['features']
    y = data['labels']
    
    model.fit(X, y)
    return model


def main():
    """Load data from SQLite, train model, and save to disk."""
    print("📊 Loading training data from SQLite...")
    
    # Load training data from database
    db = get_connection()
    query = """
    SELECT 
        d.discount_value,
        r.distance * 1.60934 as distance_km,
        CAST(json_extract(p.travel_history, '$.trips') AS INTEGER) as history_trips,
        CAST(json_extract(p.travel_history, '$.total_spend') AS REAL) / 
            NULLIF(CAST(json_extract(p.travel_history, '$.trips') AS INTEGER), 0) as avg_spend,
        d.route_id,
        r.origin,
        r.destination
    FROM discounts d
    JOIN passengers p ON d.passenger_id = p.id
    JOIN routes r ON d.route_id = r.id
    WHERE CAST(json_extract(p.travel_history, '$.trips') AS INTEGER) > 0
    """
    rows = db.fetch_data(query)
    
    if not rows or len(rows) == 0:
        db.close()
        print(f"Debug: rows = {rows}, type = {type(rows)}")
        raise ValueError("No training data found. Run 'make db-init' to load sample data.")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['discount_value', 'distance_km', 'history_trips', 
                                      'avg_spend', 'route_id', 'origin', 'destination'])
    db.close()
    
    print(f"✅ Loaded {len(df)} training samples")
    
    # Prepare features and labels
    X = df.drop('discount_value', axis=1)
    y = df['discount_value']
    
    # Create and train model
    model = DiscountPredictor()
    data = {'features': X, 'labels': y}
    trained_model = train_model(model, data)
    
    # Save model to disk
    model_path = Path("models/discount_predictor.pkl")
    trained_model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    return trained_model


if __name__ == "__main__":
    main()