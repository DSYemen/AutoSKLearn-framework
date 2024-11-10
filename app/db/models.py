# app/db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    model_type = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    parameters = Column(JSON)
    metrics = Column(JSON)
    feature_importance = Column(JSON)

class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    input_data = Column(JSON)
    prediction = Column(Float)
    actual = Column(Float, nullable=True)

class MonitoringLog(Base):
    __tablename__ = "monitoring"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    metrics = Column(JSON)
    alerts = Column(JSON)