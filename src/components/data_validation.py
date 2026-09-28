import os
import logging
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class DataValidation:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, "r") as f:
            self.params = yaml.safe_load(f)
            
    def initiate_data_validation(self, input_path: str) -> bool:
        logger.info(f"Starting data validation on {input_path}")
        if not os.path.exists(input_path):
            logger.error(f"File {input_path} does not exist.")
            return False
            
        try:
            df = pd.read_csv(input_path)
            required_columns = ['timestamp', 'cpu_usage', 'memory_usage', 'disk_usage', 
                                'network_in', 'network_out', 'request_count', 'response_time', 'future_cpu_usage']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in dataset: {missing_cols}")
                return False
                
            logger.info("Data validation passed. All required columns are present.")
            return True
        except Exception as e:
            logger.error(f"Exception occurred during data validation: {str(e)}")
            return False
