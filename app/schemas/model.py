# app/schemas/model.py
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class ModelBase(BaseModel):
    model_type: str
    parameters: Dict[str, Any]

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    model_id: str
    created_at: datetime
    metrics: Optional[Dict[str, float]]
    feature_importance: Optional[Dict[str, float]]

    class Config:
        orm_mode = True

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_id: str
    prediction: float
    confidence: Optional[float]
    timestamp: datetime