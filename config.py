from pydantic import BaseSettings

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = True
    WORKERS: int = 1


    # Database settings (if needed in the future)
    # DATABASE_URL: str = "sqlite:///./test.db"

    # ML Model settings
    MODEL_PATH: str = "static/trained_model.joblib"
    DATA_PROFILE_PATH: str = "static/profile_report.html"

    # Auto-sklearn settings
    AUTOSKLEARN_TIME_LIMIT: int = 3600  # 1 hour
    AUTOSKLEARN_MEMORY_LIMIT: int = 3072  # 3GB

    # Optuna settings
    OPTUNA_N_TRIALS: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
