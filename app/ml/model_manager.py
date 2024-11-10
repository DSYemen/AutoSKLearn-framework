# app/ml/model_manager.py
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import joblib
from datetime import datetime
from app.core.logging_config import logger
from app.core.config import settings

class ModelManager:
    def __init__(self):
        self.models_path = settings.MODEL_REGISTRY_PATH
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.models_metadata = self._load_metadata()

    def save_model(self, model: Any, metadata: Dict[str, Any]) -> str:
        """
        Save model and its metadata
        """
        try:
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.models_path / f"{model_id}.joblib"

            # Save model
            joblib.dump(model, model_path)

            # Save metadata
            metadata.update({
                "model_id": model_id,
                "created_at": datetime.now().isoformat(),
                "model_type": type(model).__name__
            })

            self.models_metadata[model_id] = metadata
            self._save_metadata()

            logger.info(f"Model saved successfully: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_id: str) -> tuple[Any, Dict[str, Any]]:
        """
        Load model and its metadata
        """
        try:
            model_path = self.models_path / f"{model_id}.joblib"
            if not model_path.exists():
                raise ValueError(f"Model not found: {model_id}")

            model = joblib.load(model_path)
            metadata = self.models_metadata.get(model_id, {})

            return model, metadata

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models and their metadata
        """
        return [
            {
                "model_id": model_id,
                **metadata
            }
            for model_id, metadata in self.models_metadata.items()
        ]

    def delete_model(self, model_id: str) -> None:
        """
        Delete model and its metadata
        """
        try:
            model_path = self.models_path / f"{model_id}.joblib"
            if model_path.exists():
                model_path.unlink()

            self.models_metadata.pop(model_id, None)
            self._save_metadata()

            logger.info(f"Model deleted successfully: {model_id}")

        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            raise

    def _load_metadata(self) -> Dict[str, Any]:
        """Load models metadata from file"""
        metadata_path = self.models_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save models metadata to file"""
        metadata_path = self.models_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[str]:
        """
        Get the best performing model based on specified metric
        """
        if not self.models_metadata:
            return None
    
        return max(
            self.models_metadata.items(),
            key=lambda x: x[1].get('metrics', {}).get(metric, 0)
        )[0]