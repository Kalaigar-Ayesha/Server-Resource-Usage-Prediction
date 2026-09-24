import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data_and_model(data_dir: str, model_path: str):
    logger.info(f"Loading test data from {data_dir} and model from {model_path}")
    X_test_path = os.path.join(data_dir, 'X_test.csv')
    y_test_path = os.path.join(data_dir, 'y_test.csv')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test data not found in {data_dir}. Run preprocess.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
        
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    model = joblib.load(model_path)
    
    return X_test, y_test, model

def evaluate_model(y_true, y_pred, metrics_dir):
    logger.info("Calculating evaluation metrics")
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2)
    }
    
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'metrics.json')
    
    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics: MAE={mae:.4f}, RMSE={np.sqrt(mse):.4f}, R2={r2:.4f}")
    return metrics

def plot_predictions(y_true, y_pred, metrics_dir):
    logger.info("Generating prediction vs actual visualization")
    plt.figure(figsize=(12, 6))
    
    # Plot a subset to make it readable (e.g., first 500 points)
    subset_size = min(500, len(y_true))
    plt.plot(y_true.values[:subset_size], label='Actual CPU Usage', alpha=0.7)
    plt.plot(y_pred[:subset_size], label='Predicted CPU Usage', alpha=0.7)
    
    plt.title('Future CPU Usage: Actual vs Predicted (Test Set Subset)')
    plt.xlabel('Time Step (Chronological)')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(metrics_dir, 'prediction_vs_actual.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

def main(data_dir, model_path, metrics_dir):
    try:
        X_test, y_test, model = load_test_data_and_model(data_dir, model_path)
        
        logger.info("Making predictions on test data")
        predictions = model.predict(X_test)
        
        evaluate_model(y_test, predictions, metrics_dir)
        plot_predictions(y_test, predictions, metrics_dir)
        
        logger.info("Evaluation pipeline complete!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML model performance")
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory containing processed test data')
    parser.add_argument('--model-path', type=str, default='models/server_cpu_model.pkl', help='Path to the trained model')
    parser.add_argument('--metrics-dir', type=str, default='metrics', help='Directory to save metrics and plots')
    args = parser.parse_args()
    
    main(args.data_dir, args.model_path, args.metrics_dir)
