import os
import yaml
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, "r") as f:
            self.params = yaml.safe_load(f)
        self.config = self.params.get('training', {})
        self.random_state = self.params.get('base', {}).get('random_state', 42)
    
    def initiate_model_training(self):
        data_dir = 'data/processed'
        model_dir = 'models'
        
        logger.info("Loading training data")
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze("columns")
        
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', 10)
        
        logger.info(f"Training RandomForestRegressor (n_estimators={n_estimators}, max_depth={max_depth})")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'server_cpu_model.pkl'))
        logger.info("Model saved successfully.")
