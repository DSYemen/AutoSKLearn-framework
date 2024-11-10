# app/core/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from pathlib import Path

class CustomJSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        return json.dumps(log_record)

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logger = logging.getLogger("ml_framework")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomJSONFormatter())
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        log_dir / "ml_framework.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(CustomJSONFormatter())
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()