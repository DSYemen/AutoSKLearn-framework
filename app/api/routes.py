# app/api/routes.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from app.db.database import get_db
from app.ml.data_processing import AdvancedDataProcessor
from app.ml.model_selection import ModelSelector
from app.ml.model_training import ModelTrainer
from app.ml.prediction import PredictionService
from app.ml.model_evaluation import ModelEvaluator
from app.visualization.dashboard import DashboardGenerator
from app.utils.alerts import AlertSystem
from app.schemas.model import ModelResponse, PredictionRequest, PredictionResponse
from app.core.logging_config import logger
from app.db.models import DatasetProfile, TrainingJob, ModelRecord, PredictionLog

router = APIRouter()
prediction_service = PredictionService()
alert_system = AlertSystem()

@router.post("/train", response_model=ModelResponse)
async def train_model(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    تدريب نموذج جديد باستخدام البيانات المرفوعة
    """
    try:
        # إنشاء مهمة تدريب جديدة
        training_job = TrainingJob(
            status="processing",
            config={"filename": file.filename}
        )
        db.add(training_job)
        db.commit()

        # معالجة البيانات
        processor = AdvancedDataProcessor()
        processed_data = await processor.process_data(file)

        # إنشاء ملف تعريف البيانات
        profile = DatasetProfile(
            dataset_hash=processed_data.dataset_hash,
            stats=processed_data.stats
        )
        db.add(profile)
        db.commit()

        # اختيار النموذج الأفضل
        selector = ModelSelector()
        model, problem_type = selector.select_model(
            processed_data.processed_df,
            processed_data.target
        )

        # تدريب النموذج
        trainer = ModelTrainer(model, problem_type)
        trained_model, metrics = trainer.train(
            processed_data.processed_df.drop(processed_data.target, axis=1),
            processed_data.processed_df[processed_data.target]
        )

        # تقييم النموذج
        evaluator = ModelEvaluator(trained_model, problem_type)
        evaluation_results = evaluator.evaluate(
            processed_data.X_test,
            processed_data.y_test
        )

        # حفظ النموذج وبياناته
        model_record = ModelRecord(
            model_type=type(trained_model).__name__,
            parameters=trained_model.get_params(),
            metrics=evaluation_results,
            feature_importance=processed_data.feature_importance
        )
        db.add(model_record)
        db.commit()

        # تحديث حالة مهمة التدريب
        training_job.status = "completed"
        training_job.completed_at = datetime.utcnow()
        db.commit()

        # بدء المراقبة في الخلفية
        if background_tasks:
            background_tasks.add_task(
                start_model_monitoring,
                model_record.model_id,
                processed_data.processed_df
            )

        return ModelResponse(
            model_id=model_record.model_id,
            model_type=model_record.model_type,
            metrics=evaluation_results,
            feature_importance=processed_data.feature_importance
        )

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        if training_job:
            training_job.status = "failed"
            training_job.error_message = str(e)
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/{model_id}")
async def predict(
    model_id: str,
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    استخدام النموذج للتنبؤ
    """
    try:
        # التحقق من وجود النموذج
        model_record = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id
        ).first()
        if not model_record:
            raise HTTPException(status_code=404, detail="Model not found")

        # تنفيذ التنبؤ
        prediction = prediction_service.predict(
            model_id,
            request.features
        )

        # تسجيل التنبؤ
        prediction_log = PredictionLog(
            model_id=model_id,
            input_data=request.features,
            prediction=prediction
        )
        db.add(prediction_log)
        db.commit()

        return PredictionResponse(
            model_id=model_id,
            prediction=prediction,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/performance")
async def get_model_performance(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    الحصول على بيانات أداء النموذج
    """
    try:
        model_record = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id
        ).first()
        if not model_record:
            raise HTTPException(status_code=404, detail="Model not found")

        # جمع بيانات الأداء
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).all()

        performance_data = {
            "metrics": model_record.metrics,
            "feature_importance": model_record.feature_importance,
            "predictions_history": [
                {
                    "timestamp": p.timestamp,
                    "prediction": p.prediction,
                    "actual": p.actual
                }
                for p in predictions
            ]
        }

        return performance_data

    except Exception as e:
        logger.error(f"Error fetching performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/update")
async def update_model(
    model_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    تحديث النموذج ببيانات جديدة
    """
    try:
        # التحقق من وجود النموذج
        model_record = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id
        ).first()
        if not model_record:
            raise HTTPException(status_code=404, detail="Model not found")

        # معالجة البيانات الجديدة
        processor = AdvancedDataProcessor()
        processed_data = await processor.process_data(file)

        # تحديث النموذج
        trainer = ModelTrainer(
            prediction_service.get_model(model_id),
            model_record.model_type
        )
        updated_model, new_metrics = trainer.update(
            processed_data.processed_df.drop(processed_data.target, axis=1),
            processed_data.processed_df[processed_data.target]
        )

        # تحديث سجل النموذج
        model_record.metrics = new_metrics
        model_record.updated_at = datetime.utcnow()
        db.commit()

        return {
            "message": "Model updated successfully",
            "new_metrics": new_metrics
        }

    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility functions
def start_model_monitoring(model_id: str, data: pd.DataFrame):
    """Start monitoring for a model"""
    try:
        monitor = ModelMonitor(model_id)
        monitor.initialize_monitoring(data)
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")

def log_prediction(db: Session, prediction: float, input_data: Dict[str, Any]):
    """Log prediction to database"""
    try:
        log = PredictionLog(
            model_id=prediction_service.model_id,
            input_data=input_data,
            prediction=prediction
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")