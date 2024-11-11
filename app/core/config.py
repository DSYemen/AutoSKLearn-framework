# app/core/config.py
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import yaml

class Settings(BaseModel):
    # App Settings
    APP_NAME: str = "ML Framework"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    REDIS_HOST: str = "https://127.0.0.1"
    REDIS_PORT: int = 3278

    PREDICTION_BATCH_SIZE: int = 20
    # MONITORING_PATH: str =""
    MONITORING_ENABLED: bool = False
    AUTO_UPDATE_ENABLED: bool = True
    MODEL_UPDATE_CHECK_INTERVAL: int = 360

    # MODEL_UPDATE Settings
    NEW_DATA_PATH: str = 'data/new_data.csv'
    CURRENT_MODEL_METRICS_PATH: str = 'models/current_model_metrics.json'
    MODEL_PATH: str = 'models/current_model.pkl'
    PRIMARY_METRIC: str = 'accuracy'  # or 'f1', 'mse', etc. depending on your use case

    # Paths
    # BASE_DIR: Path = Path(__file__).parent.parent.parent
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    MODELS_DIR: Path = BASE_DIR / "static/models"
    REPORTS_DIR: Path = BASE_DIR / "static/reports"
    MONITORING_PATH: Path = BASE_DIR / "static/Monitoring"

    # ML Settings
    MAX_FEATURES: int = 100
    MAX_POLY_FEATURES: int = 5
    MODEL_UPDATE_INTERVAL: int = 86400  # 24 hours in seconds
    FEATURE_SELECTION_THRESHOLD: float = 0.01

    # File Settings
    ALLOWED_EXTENSIONS: List[str] = ['.csv', '.xlsx', '.parquet']
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    # Monitoring Settings
    MONITORING_ENABLED: bool = True
    ALERT_THRESHOLDS: Dict[str, float] = {
        "accuracy": 0.8,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.75,
        "mse": 0.1,
        "mae": 0.1
    }

    # Email Settings
    EMAIL_CONFIG: Dict[str, Any] = {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": "ml-framework@example.com",
        "recipients": ["admin@example.com"],
        "username": "",
        "password": ""
    }

    class Config:
        env_file = ".env"

    @classmethod
    def load_yaml(cls, yaml_path: str = "config.yaml"):
        with open(yaml_path) as f:
            yaml_settings = yaml.safe_load(f)
        return cls(**yaml_settings)

settings = Settings.load_yaml()