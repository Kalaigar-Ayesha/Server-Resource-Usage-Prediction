import pandas as pd
import numpy as np
import os
import joblib
import logging
import argparse
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found at {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting time-based features")
    # Sort chronologically to prevent leakage
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df

def split_data(df: pd.DataFrame, target_col: str):
    logger.info("Splitting data chronologically into train, validation, and test sets")
    # Features and target
    features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_in', 
                'network_out', 'request_count', 'response_time', 'hour', 'day_of_week']
    
    X = df[features]
    y = df[target_col]
    
    # 70% train, 15% validation, 15% test chronologically (no shuffling)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, model_dir):
    logger.info("Scaling features using StandardScaler")
    scaler = StandardScaler()
    
    # Fit ONLY on training data to prevent data leakage from future data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def main(input_path, output_dir, model_dir):
    try:
        df = load_data(input_path)
        df = extract_features(df)
        
        target_col = 'future_cpu_usage'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
            
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)
        X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test, model_dir)
        
        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_dir}")
        X_train_scaled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        X_val_scaled.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        logger.info("Preprocessing complete!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw server metrics data")
    parser.add_argument('--input', type=str, default='data/raw/server_metrics.csv', help='Path to raw data')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Directory to save processed data')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save the scaler')
    args = parser.parse_args()
    
    main(args.input, args.output_dir, args.model_dir)
