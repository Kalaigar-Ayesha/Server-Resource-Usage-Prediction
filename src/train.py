import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json

def train_model(data_path='data/raw/server_metrics.csv', model_dir='models', metrics_dir='metrics'):
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Feature engineering (e.g., hour of day)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Define features and target
    features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_in', 
                'network_out', 'request_count', 'response_time', 'hour', 'day_of_week']
    target = 'future_cpu_usage'
    
    X = df[features]
    y = df[target]
    
    print("Splitting data into train and test sets (time-based split)...")
    # For time series, it's better not to shuffle, though standard Random Forest doesn't inherently model sequence
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': mae,
        'r2': r2
    }
    
    print(f"Metrics: {metrics}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_model()
