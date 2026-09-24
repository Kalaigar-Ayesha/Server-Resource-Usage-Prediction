import pandas as pd
import os
import joblib
import logging
import yaml
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    data_dir = 'data/processed'
    model_dir = 'models'
    
    logger.info("Loading training data")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze("columns")
    
    logger.info(f"Training RandomForestRegressor (n_estimators={params['training']['n_estimators']}, max_depth={params['training']['max_depth']})")
    model = RandomForestRegressor(
        n_estimators=params['training']['n_estimators'],
        max_depth=params['training']['max_depth'],
        random_state=params['base']['random_state'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'server_cpu_model.pkl'))
    logger.info("Model saved.")

if __name__ == "__main__":
    main()
