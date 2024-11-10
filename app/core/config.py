# # app/core/config.py
# from pydantic import BaseModel
# from typing import List, Dict
# import yaml
# from pathlib import Path

# class Settings(BaseModel):
#     # App Settings
#     APP_NAME: str = "ML Framework"
#     VERSION: str = "1.0.0"
#     DEBUG: bool = False
#     API_PREFIX: str = "/api/v1"

#     # ML Settings
#     MAX_FEATURES: int = 100
#     MAX_POLY_FEATURES: int = 5
#     MODEL_UPDATE_INTERVAL: int = 86400  # 24 hours

#     # File Settings
#     ALLOWED_EXTENSIONS: List[str] = ['.csv', '.xlsx', '.parquet']
#     MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

#     # Model Settings
#     MODEL_REGISTRY_PATH: Path = Path("models")
#     MONITORING_PATH: Path = Path("monitoring")

#     @classmethod
#     def load_yaml(cls, yaml_path: str = "config.yaml"):
#         with open(yaml_path) as f:
#             yaml_settings = yaml.safe_load(f)
#         return cls(**yaml_settings)

# settings = Settings.load_yaml()

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

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    MODELS_DIR: Path = BASE_DIR / "static/models"
    REPORTS_DIR: Path = BASE_DIR / "static/reports"

    # ML Settings
    MAX_FEATURES: int = 100
    MAX_POLY_FEATURES: int = 5
    MODEL_UPDATE_INTERVAL: int = 86400
    FEATURE_SELECTION_THRESHOLD: float = 0.01

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
