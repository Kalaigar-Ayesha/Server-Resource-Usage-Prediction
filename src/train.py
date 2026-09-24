import pandas as pd
import os
import joblib
import logging
import argparse
from sklearn.ensemble import RandomForestRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(data_dir: str):
    logger.info(f"Loading processed training data from {data_dir}")
    X_train_path = os.path.join(data_dir, 'X_train.csv')
    y_train_path = os.path.join(data_dir, 'y_train.csv')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(f"Processed training data not found in {data_dir}. Run preprocess.py first.")
        
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns") # ensure it's a Series
    return X_train, y_train

def train_model(X_train, y_train):
    logger.info("Initializing RandomForestRegressor")
    # Using a reasonably constrained RF to prevent massive file sizes and overfitting
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    logger.info("Training model... this might take a moment")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def main(data_dir, model_dir):
    try:
        X_train, y_train = load_processed_data(data_dir)
        model = train_model(X_train, y_train)
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'server_cpu_model.pkl')
        
        logger.info(f"Saving trained model to {model_path}")
        joblib.dump(model, model_path)
        logger.info("Training pipeline complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model on preprocessed data")
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory containing processed data')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save the trained model')
    args = parser.parse_args()
    
    main(args.data_dir, args.model_dir)
