# app/utils/documentation.py
from typing import Dict, Any
import yaml
from pathlib import Path
from app.core.config import settings

class ModelDocumentation:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.docs_path = settings.MODELS_DIR / model_id / "documentation"
        self.docs_path.mkdir(parents=True, exist_ok=True)

    def generate_documentation(self, model_info: Dict[str, Any]) -> str:
        """Generate comprehensive model documentation"""
        doc = {
            "model_overview": {
                "id": self.model_id,
                "type": model_info['model_type'],
                "created_at": model_info['created_at'],
                "purpose": model_info.get('purpose', 'Not specified')
            },
            "performance_metrics": model_info.get('metrics', {}),
            "feature_importance": model_info.get('feature_importance', {}),
            "training_details": {
                "data_size": model_info.get('training_data_size'),
                "training_duration": model_info.get('training_duration'),
                "parameters": model_info.get('parameters', {})
            },
            "usage_guidelines": {
                "input_format": model_info.get('input_format', {}),
                "output_format": model_info.get('output_format', {}),
                "limitations": model_info.get('limitations', [])
            },
            "maintenance": {
                "last_updated": model_info.get('last_updated'),
                "update_frequency": model_info.get('update_frequency'),
                "monitoring_metrics": model_info.get('monitoring_metrics', {})
            }
        }

        # Save documentation
        doc_path = self.docs_path / "model_documentation.yaml"
        with open(doc_path, 'w') as f:
            yaml.dump(doc, f, default_flow_style=False)

        return str(doc_path)

    def generate_api_documentation(self) -> str:
        """Generate API usage documentation"""
        api_doc = {
            "endpoints": {
                "prediction": {
                    "url": f"/api/v1/predict/{self.model_id}",
                    "method": "POST",
                    "input_example": self._get_input_example(),
                    "output_example": self._get_output_example()
                },
                "batch_prediction": {
                    "url": f"/api/v1/predict/{self.model_id}/batch",
                    "method": "POST",
                    "input_example": self._get_batch_input_example()
                }
            },
            "authentication": {
                "type": "API Key",
                "location": "Header",
                "name": "X-API-Key"
            },
            "rate_limits": {
                "requests_per_minute": 60,
                "batch_size_limit": 1000
            }
        }

        # Save API documentation
        api_doc_path = self.docs_path / "api_documentation.yaml"
        with open(api_doc_path, 'w') as f:
            yaml.dump(api_doc, f, default_flow_style=False)

        return str(api_doc_path)

    def _get_input_example(self) -> Dict[str, Any]:
        """Get example input format"""
        return prediction_service.get_input_example(self.model_id)

    def _get_output_example(self) -> Dict[str, Any]:
        """Get example output format"""
        return prediction_service.get_output_example(self.model_id)

    def _get_batch_input_example(self) -> Dict[str, Any]:
        """Get example batch input format"""
        return {
            "instances": [
                self._get_input_example()
                for _ in range(2)
            ]
        }