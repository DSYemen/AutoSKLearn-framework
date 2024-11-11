# app/core/config.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

class AppConfig(BaseModel):
    name: str
    version: str
    debug: bool

class ModelConfig(BaseModel):
    time_limit: int
    cv_folds: int
    test_size: float
    random_state: int

class OptimizationConfig(BaseModel):
    n_trials: int
    timeout: int

class ImputationConfig(BaseModel):
    numeric_strategy: str
    categorical_strategy: str

class ScalingConfig(BaseModel):
    method: str

class FeatureSelectionConfig(BaseModel):
    max_features: int

class PreprocessingConfig(BaseModel):
    imputation: ImputationConfig
    scaling: ScalingConfig
    feature_selection: FeatureSelectionConfig

class MonitoringConfig(BaseModel):
    update_interval: int
    performance_threshold: float
    monitoring_interval: int

class Settings(BaseSettings):
    """إعدادات التطبيق"""
    # إعدادات أساسية
    APP_NAME: str = "ML Framework"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    
    # إعدادات Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # المسارات
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    MODEL_REGISTRY_PATH: Path = MODELS_DIR / "registry"
    MONITORING_PATH: Path = BASE_DIR / "monitoring"
    
    # إعدادات النماذج
    MAX_FEATURES: int = 100
    OPTIMIZATION_TRIALS: int = 20
    OPTIMIZATION_TIMEOUT: int = 600
    
    # إعدادات المراقبة
    MONITORING_ENABLED: bool = True
    AUTO_UPDATE_ENABLED: bool = True
    PREDICTION_BATCH_SIZE: int = 1000
    PREDICTION_CACHE_TTL: int = 3600
    MAX_PREDICTION_HISTORY: int = 1000
    MONITORING_INTERVAL: int = 60
    
    # إعدادات التنبيهات
    ALERT_THRESHOLDS: Dict[str, float] = {
        "accuracy": 0.8,
        "latency": 1.0,
        "drift": 0.1
    }
    
    # إعدادات البريد الإلكتروني
    EMAIL_CONFIG: Dict[str, Any] = {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": "alerts@mlframework.com",
        "recipients": ["admin@mlframework.com"],
        "username": "",
        "password": ""
    }

    # إعدادات إضافية من الملف
    app_settings: Optional[AppConfig] = None
    model_settings: Optional[ModelConfig] = None
    optimization_settings: Optional[OptimizationConfig] = None
    preprocessing_settings: Optional[PreprocessingConfig] = None
    monitoring_settings: Optional[MonitoringConfig] = None

    # إعدادات تحديث النموذج
    NEW_DATA_PATH: Path = BASE_DIR / "data/new"
    MODEL_UPDATE_CHECK_INTERVAL: int = 3600  # بالثواني
    MODEL_UPDATE_THRESHOLD: float = 0.1  # نسبة التغير في الأداء التي تتطلب التحديث
    
    # إنشاء المجلدات المطلوبة
    def create_required_directories(self):
        self.NEW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # إضافة إلى الإعدادات الحالية
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB

    class Config:
        env_file = ".env"

    @classmethod
    def load_yaml(cls, yaml_path: str = "config.yaml") -> "Settings":
        """تحميل الإعدادات من ملف YAML"""
        with open(yaml_path) as f:
            yaml_settings = yaml.safe_load(f)
            
        settings = cls()
        
        # تحميل الإعدادات من YAML
        if 'app' in yaml_settings:
            settings.app_settings = AppConfig(**yaml_settings['app'])
            settings.APP_NAME = settings.app_settings.name
            settings.VERSION = settings.app_settings.version
            settings.DEBUG = settings.app_settings.debug
            
        if 'model' in yaml_settings:
            settings.model_settings = ModelConfig(**yaml_settings['model'])
            
        if 'optimization' in yaml_settings:
            settings.optimization_settings = OptimizationConfig(**yaml_settings['optimization'])
            settings.OPTIMIZATION_TRIALS = settings.optimization_settings.n_trials
            settings.OPTIMIZATION_TIMEOUT = settings.optimization_settings.timeout
            
        if 'preprocessing' in yaml_settings:
            settings.preprocessing_settings = PreprocessingConfig(**yaml_settings['preprocessing'])
            
        if 'monitoring' in yaml_settings:
            settings.monitoring_settings = MonitoringConfig(**yaml_settings['monitoring'])
            settings.MONITORING_INTERVAL = settings.monitoring_settings.monitoring_interval
            
        return settings

settings = Settings.load_yaml()