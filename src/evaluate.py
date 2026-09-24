import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    data_dir = 'data/processed'
    model_path = 'models/server_cpu_model.pkl'
    metrics_dir = 'metrics'
    
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze("columns")
    model = joblib.load(model_path)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2)
    }
    
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics: MAE={mae:.4f}, RMSE={np.sqrt(mse):.4f}, R2={r2:.4f}")
    
    # Model Validation step
    if r2 < params['evaluation']['r2_threshold']:
        logger.error(f"Validation FAILED: R2 {r2:.4f} is below threshold {params['evaluation']['r2_threshold']}")
        raise ValueError("Model failed R2 validation")
    if mae > params['evaluation']['mae_threshold']:
        logger.error(f"Validation FAILED: MAE {mae:.4f} is above threshold {params['evaluation']['mae_threshold']}")
        raise ValueError("Model failed MAE validation")
        
    logger.info("Model PASSED validation!")
    
    # Plot
    plt.figure(figsize=(12, 6))
    subset_size = min(500, len(y_test))
    plt.plot(y_test.values[:subset_size], label='Actual CPU Usage', alpha=0.7)
    plt.plot(predictions[:subset_size], label='Predicted CPU Usage', alpha=0.7)
    plt.title('Future CPU Usage: Actual vs Predicted (Test Set Subset)')
    plt.xlabel('Time Step')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'prediction_vs_actual.png'))
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
