# app/ml/model_updater.py
import schedule
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional, Dict
from app.core.logging_config import logger
from app.core.config import settings
from app.ml.data_processing import AdvancedDataProcessor
from app.ml.model_selection import ModelSelector
from app.ml.model_training import ModelTrainer

class ModelUpdater:
    def __init__(self):
        self.last_update = None
        self.update_interval = settings.MODEL_UPDATE_INTERVAL

    def start(self):
        """Start automated model updating"""
        schedule.every(self.update_interval).seconds.do(self._update_model)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def _update_model(self):
        """Update model with new data"""
        try:
            logger.info("Starting model update process")

            # Get new data
            new_data = self._get_new_data()
            if new_data is None or new_data.empty:
                logger.info("No new data available for update")
                return

            # Process new data
            processor = AdvancedDataProcessor()
            processed_data = processor.process_data(new_data)

            # Select and train new model
            selector = ModelSelector()
            model, problem_type = selector.select_model(processed_data)

            trainer = ModelTrainer(model, problem_type)
            new_model, metrics = trainer.train(
                processed_data.drop('target', axis=1),
                processed_data['target']
            )

            # Evaluate if new model should replace current
            if self._should_replace_model(metrics):
                self._deploy_new_model(new_model)
                logger.info("Model updated successfully", extra={"metrics": metrics})
            else:
                logger.info("Current model retained - new model did not show significant improvement")

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error in model update: {str(e)}")

    def _get_new_data(self) -> Optional[pd.DataFrame]:
        """Get new data for model update"""
        try:
            # This is a placeholder. Replace with actual data collection logic
            new_data_path = Path(settings.NEW_DATA_PATH)
            if new_data_path.exists():
                return pd.read_csv(new_data_path)
            return None
        except Exception as e:
            logger.error(f"Error getting new data: {str(e)}")
            return None

    def _should_replace_model(self, new_metrics: Dict[str, float]) -> bool:
        """Decide if new model should replace current model"""
        try:
            # This is a placeholder. Replace with actual comparison logic
            current_metrics_path = Path(settings.CURRENT_MODEL_METRICS_PATH)
            if not current_metrics_path.exists():
                return True
            
            current_metrics = pd.read_json(current_metrics_path)
            primary_metric = settings.PRIMARY_METRIC
            
            return new_metrics[primary_metric] > current_metrics[primary_metric]
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return False

    def _deploy_new_model(self, new_model) -> None:
        """Deploy new model to production"""
        try:
            # This is a placeholder. Replace with actual deployment logic
            model_path = Path(settings.MODEL_PATH)
            new_model.save(model_path)
            logger.info(f"New model deployed to {model_path}")
        except Exception as e:
            logger.error(f"Error deploying new model: {str(e)}")