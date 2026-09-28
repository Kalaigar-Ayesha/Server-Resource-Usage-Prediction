import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_training = ModelTraining()
        self.model_evaluation = ModelEvaluation()
        
    def run_pipeline(self):
        logger.info(">>> Training Pipeline Started <<<")
        
        # 1. Data Ingestion
        data_path = self.data_ingestion.initiate_data_ingestion()
        
        # 2. Data Validation
        is_valid = self.data_validation.initiate_data_validation(data_path)
        if not is_valid:
            raise Exception("Data validation failed. Pipeline terminated.")
            
        # 3. Data Transformation
        self.data_transformation.initiate_data_transformation(data_path)
        
        # 4. Model Training
        self.model_training.initiate_model_training()
        
        # 5. Model Evaluation
        self.model_evaluation.initiate_model_evaluation()
        
        logger.info(">>> Training Pipeline Completed Successfully <<<")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
