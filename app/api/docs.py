# app/api/docs.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.core.config import settings

docs_router = APIRouter()

@docs_router.get("/docs/api", response_class=HTMLResponse)
async def get_api_docs():
    """
    Generate interactive API documentation
    """
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>ML Framework API Documentation</title>
            <link href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script>
                window.onload = () => {
                    SwaggerUIBundle({
                        url: "/openapi.json",
                        dom_id: '#swagger-ui',
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIBundle.SwaggerUIStandalonePreset
                        ],
                        layout: "BaseLayout"
                    });
                };
            </script>
        </body>
    </html>
    """

@docs_router.get("/docs/models")
async def get_models_documentation():
    """
    Get documentation about available models and their capabilities
    """
    return {
        "available_models": {
            "classification": [
                {
                    "name": "RandomForest",
                    "description": "Ensemble learning method for classification",
                    "use_cases": ["Binary classification", "Multi-class classification"],
                    "advantages": [
                        "Handles non-linear relationships",
                        "Robust to outliers",
                        "Provides feature importance"
                    ]
                },
                # Add other models...
            ],
            "regression": [
                {
                    "name": "XGBoost",
                    "description": "Gradient boosting framework",
                    "use_cases": ["Numeric prediction", "Time series forecasting"],
                    "advantages": [
                        "High performance",
                        "Handles missing values",
                        "Built-in regularization"
                    ]
                },
                # Add other models...
            ]
        },
        "metrics_explanation": {
            "classification": {
                "accuracy": "Ratio of correct predictions to total predictions",
                "precision": "Ratio of true positives to predicted positives",
                "recall": "Ratio of true positives to actual positives",
                "f1": "Harmonic mean of precision and recall"
            },
            "regression": {
                "mse": "Mean squared error",
                "rmse": "Root mean squared error",
                "mae": "Mean absolute error",
                "r2": "Coefficient of determination"
            }
        },
        "preprocessing_steps": {
            "data_cleaning": [
                "Missing value imputation",
                "Outlier detection and handling",
                "Duplicate removal"
            ],
            "feature_engineering": [
                "Automatic feature creation",
                "Polynomial features",
                "Date-time features"
            ],
            "encoding": [
                "One-hot encoding",
                "Label encoding",
                "Target encoding"
            ]
        }
    }