from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.discount_predictor import DiscountPredictor
from data.database import get_connection


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data.
    
    Parameters:
    model: Trained DiscountPredictor model
    test_data: Dictionary with 'features' (X) and 'labels' (y)
    
    Returns:
    Dictionary with predictions and metrics
    """
    X_test = test_data['features']
    y_test = test_data['labels']
    
    predictions = model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {
        'predictions': predictions,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2
    }


def calculate_accuracy(predictions, labels):
    """Legacy function - not suitable for regression tasks."""
    correct_predictions = sum(p == l for p, l in zip(predictions, labels))
    return correct_predictions / len(labels) if labels else 0.0


def main():
    """Load trained model, fetch test data, and evaluate performance."""
    print("📊 Evaluating DiscountPredictor model...")
    
    # Load trained model
    model_path = Path("models/discount_predictor.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python src/training/train.py' first."
        )
    
    model = DiscountPredictor.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Load test data from database
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
        raise ValueError("No test data found. Run 'make db-init' to load sample data.")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['discount_value', 'distance_km', 'history_trips', 
                                      'avg_spend', 'route_id', 'origin', 'destination'])
    db.close()
    
    print(f"✅ Loaded {len(df)} test samples")
    
    # Prepare features and labels
    X_test = df.drop('discount_value', axis=1)
    y_test = df['discount_value']
    
    # Evaluate model
    test_data = {'features': X_test, 'labels': y_test}
    results = evaluate_model(model, test_data)
    
    # Display results
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Absolute Error (MAE):  {results['mae']:.2f}")
    print(f"Root Mean Squared Error:    {results['rmse']:.2f}")
    print(f"R² Score:                   {results['r2_score']:.4f}")
    print("="*50)
    
    # Show sample predictions
    print("\n📋 Sample Predictions (first 5):")
    print("-" * 50)
    for i in range(min(5, len(y_test))):
        print(f"  Actual: {y_test.iloc[i]:6.2f} | Predicted: {results['predictions'].iloc[i]:6.2f} | "
              f"Error: {abs(y_test.iloc[i] - results['predictions'].iloc[i]):6.2f}")
    print("-" * 50)
    
    return results


if __name__ == "__main__":
    main()