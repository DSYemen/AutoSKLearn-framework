# app/utils/exceptions.py
from fastapi import HTTPException
from typing import Dict, Any

class MLFrameworkException(Exception):
    """Base exception for ML Framework"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ModelNotFoundError(MLFrameworkException):
    """Raised when model is not found"""
    pass

class DataValidationError(MLFrameworkException):
    """Raised when data validation fails"""
    pass

class TrainingError(MLFrameworkException):
    """Raised when model training fails"""
    pass

class PredictionError(MLFrameworkException):
    """Raised when prediction fails"""
    pass

def handle_ml_exception(exc: MLFrameworkException):
    """Convert ML exceptions to HTTP exceptions"""
    status_code = {
        ModelNotFoundError: 404,
        DataValidationError: 400,
        TrainingError: 500,
        PredictionError: 500
    }.get(type(exc), 500)

    return HTTPException(
        status_code=status_code,
        detail={
            "message": str(exc),
            "details": exc.details
        }
    )

class ModelUpdateError(MLFrameworkException):
    """Raised when model update fails"""
    pass