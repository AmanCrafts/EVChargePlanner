"""
model.py
--------
Simple functions for training and evaluating ML model.
Uses Random Forest - handles non-linear patterns well and is easy to understand.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    
    Random Forest is like having many decision trees "vote" on the answer.
    It works well for non-linear patterns like peak hours affecting energy usage.
    
    Parameters:
        X_train: Training features
        y_train: Training target values
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 50)
    print("TRAINING MODEL...")
    print("=" * 50)
    
    # Create Random Forest model
    # n_estimators = number of trees (100 is a good default)
    # max_depth = how deep each tree can grow (limits complexity)
    # random_state = for reproducible results
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train the model on training data
    model.fit(X_train, y_train)
    
    print("Model: Random Forest Regressor")
    print("Number of trees: 100")
    print("Training completed successfully!")
    
    # Print feature importance (which features matter most)
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    print("\nFeature Importance (how much each feature affects predictions):")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.2%}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Actual test values
    
    Returns:
        predictions: Predicted values
    """
    print("\n" + "=" * 50)
    print("EVALUATING MODEL...")
    print("=" * 50)
    
    # Make predictions on test data
    predictions = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    # MAE tells us the average difference between predicted and actual values
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculate Root Mean Squared Error (RMSE)
    # RMSE penalizes larger errors more than MAE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Print results
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kWh")
    print("-" * 30)
    
    # Explain what these numbers mean
    print("\nWhat this means:")
    print(f"On average, our predictions are off by about {mae:.2f} kWh")
    
    return predictions


def get_model_summary(model, X_test, y_test, predictions):
    """
    Get a summary of model performance in simple terms.
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Actual test values
        predictions: Model predictions
    
    Returns:
        Dictionary with summary information
    """
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Calculate R-squared (how well model explains variance)
    r_squared = model.score(X_test, y_test)
    
    # Determine performance level
    if r_squared >= 0.7:
        performance = "Good"
    elif r_squared >= 0.4:
        performance = "Moderate"
    else:
        performance = "Needs Improvement"
    
    summary = {
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'performance': performance
    }
    
    return summary
