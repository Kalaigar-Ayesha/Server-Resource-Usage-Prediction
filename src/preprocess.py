import pandas as pd
import numpy as np
import os
import joblib
import logging
import yaml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, target_col: str, val_size: float, test_size: float):
    features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_in', 
                'network_out', 'request_count', 'response_time', 'hour', 'day_of_week']
    X, y = df[features], df[target_col]
    
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    input_path = params['data_generation']['output_path']
    output_dir = 'data/processed'
    model_dir = 'models'
    
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract features and sort strictly chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, 'future_cpu_usage', 
        params['preprocessing']['val_size'], 
        params['preprocessing']['test_size']
    )
    
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    X_val_scaled.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
