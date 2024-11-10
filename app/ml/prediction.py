# app/ml/prediction.py
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from app.core.logging_config import logger
from app.ml.monitoring import ModelMonitor
from app.core.config import settings

class PredictionService:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.monitor = None
        self.feature_names = None

    def load_model(self, model_path: str) -> None:
        """Load model and initialize monitoring"""
        try:
            self.model = joblib.load(model_path)
            self.model_id = model_path.split('/')[-1].split('.')[0]
            self.monitor = ModelMonitor(self.model_id)

            # Load feature names if available
            feature_names_path = Path(model_path).parent / f"{self.model_id}_features.json"
            if feature_names_path.exists():
                with open(feature_names_path) as f:
                    self.feature_names = json.load(f)

            logger.info(f"Model loaded successfully: {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, input_data: Dict[str, Any], actual: Optional[Any] = None) -> Any:
        """Make prediction with monitoring"""
        try:
            # Validate input
            validated_input = self._validate_input(input_data)

            # Transform input to DataFrame
            input_df = pd.DataFrame([validated_input])

            # Make prediction
            prediction = self.model.predict(input_df)[0]

            # Log prediction for monitoring
            self.monitor.log_prediction(input_data, prediction, actual)

            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and preprocess input data"""
        if self.feature_names is None:
            return input_data

        validated_data = {}
        for feature in self.feature_names:
            if feature not in input_data:
                raise ValueError(f"Missing required feature: {feature}")
            validated_data[feature] = input_data[feature]

        return validated_data