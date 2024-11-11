# app/db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    model_type = Column(String)
    status = Column(String, default="active")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    parameters = Column(JSON)
    metrics = Column(JSON)
    feature_importance = Column(JSON)
    problem_type = Column(String)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'status' not in kwargs:
            self.status = "active"

class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    input_data = Column(JSON)
    prediction = Column(Float)
    actual = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

class MonitoringLog(Base):
    __tablename__ = "monitoring"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    metrics = Column(JSON)
    alerts = Column(JSON)

class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"

    id = Column(Integer, primary_key=True, index=True)
    dataset_hash = Column(String, unique=True, index=True)
    profile_path = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    stats = Column(JSON)

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, index=True)
    status = Column(String)  # pending, running, completed, failed
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    config = Column(JSON)