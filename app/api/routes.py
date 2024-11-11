# app/api/routes.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.db.database import get_db
from app.ml.data_processing import AdvancedDataProcessor
from app.ml.model_selection import ModelSelector
from app.ml.model_training import ModelTrainer
from app.ml.prediction import PredictionService
from app.ml.model_evaluation import ModelEvaluator
from app.visualization.dashboard import DashboardGenerator
from app.utils.alerts import AlertSystem
from app.schemas.model import ModelResponse, PredictionRequest, PredictionResponse, ModelListResponse, MetricsResponse, PredictionHistoryResponse, DataProfileResponse
from app.core.logging_config import logger
from app.db.models import DatasetProfile, TrainingJob, ModelRecord, PredictionLog
from sqlalchemy import desc, asc, func
from sqlalchemy.types import Float
import shutil
from fastapi.responses import FileResponse
from ydata_profiling import ProfileReport
from app.schemas.model import ModelOptimizationConfig
from app.ml.model_validation import DataValidationResult, ModelValidator
from app.ml.data_validation import DataValidator
from app.ml.model_export import ModelExporter
from app.ml.model_performance import get_performance_trend, get_predictions_distribution, get_feature_correlations
from app.ml.model_performance import get_performance_data, get_predictions_history
from app.ml.model_optimization import optimize_model_task
from app.ml.model_report import ReportGenerator
from app.utils.cache import cache_manager
from pathlib import Path
import asyncio
import joblib
from app.core.config import settings
from app.ml.monitoring import ModelMonitor

router = APIRouter()
prediction_service = PredictionService()
alert_system = AlertSystem()
model_monitor = ModelMonitor("system")

# إضافة المتغيرات العامة لتخزين حالات المعالجة والاتصالات
processing_status: Dict[str, Any] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}


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


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    sort_by: str = Query("created_at", enum=[
                         "created_at", "accuracy", "name"]),
    order: str = Query("desc", enum=["asc", "desc"]),
    db: Session = Depends(get_db)
):
    """
    الحصول على قائمة النماذج مع دعم الترتيب والتصفح
    """
    try:
        models = db.query(ModelRecord).order_by(
            desc(sort_by) if order == "desc" else asc(sort_by)
        ).offset(skip).limit(limit).all()

        return ModelListResponse(
            models=[ModelResponse.from_orm(model) for model in models],
            total=db.query(ModelRecord).count(),
            page=skip // limit + 1,
            pages=(db.query(ModelRecord).count() + limit - 1) // limit
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """حذف نموذج"""
    try:
        # التحقق من وجود النموذج
        model = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # حذف النموذج من قاعدة البيانات
        db.delete(model)

        # حذف الملفات المرتبطة
        model_path = settings.MODELS_DIR / model_id
        if model_path.exists():
            shutil.rmtree(model_path)

        # حذف من الذاكرة المؤقتة
        await cache_manager.clear_model_cache(model_id)

        db.commit()
        return {"message": "Model deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/download")
async def download_model(model_id: str, db: Session = Depends(get_db)):
    """تحميل النموذج"""
    try:
        # التحقق من وجود النموذج
        model_path = settings.MODELS_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        return FileResponse(
            path=model_path,
            filename=f"model_{model_id}.joblib",
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str, db: Session = Depends(get_db)):
    """الحصول على مقاييس أداء النموذج"""
    try:
        # التحقق من وجود النموذج
        model = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # جمع المقاييس
        metrics = {
            "model_id": model_id,
            "metrics": model.metrics,
            "performance_trend": await get_performance_trend(model_id, db),
            "predictions_distribution": await get_predictions_distribution(model_id, db),
            "feature_correlations": await get_feature_correlations(model_id, db)
        }

        return MetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/predictions/history")
async def get_prediction_history(
    model_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """الحصول على سجل التنبؤات"""
    try:
        # الحصول على التنبؤات
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).order_by(
            desc(PredictionLog.timestamp)
        ).offset(skip).limit(limit).all()

        total = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).count()

        return PredictionHistoryResponse(
            predictions=[PredictionResponse.from_orm(
                pred) for pred in predictions],
            total=total,
            page=skip // limit + 1,
            pages=(total + limit - 1) // limit
        )

    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/profile")
async def get_data_profile(file: UploadFile = File(...)):
    """تحليل وتوصيف البيانات"""
    try:
        # قراءة البيانات
        df = pd.read_csv(file.file) if file.filename.endswith(
            '.csv') else pd.read_parquet(file.file)

        # إنشاء تقرير التحليل
        profile = ProfileReport(df, title="Data Profile Report")

        # حفظ التقرير
        report_path = settings.REPORTS_DIR / \
            f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        profile.to_file(report_path)

        return DataProfileResponse(
            dataset_name=file.filename,
            overview=profile.get_description(),
            variables=profile.get_variables(),
            variable_stats=profile.get_stats()
        )

    except Exception as e:
        logger.error(f"Error profiling data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/optimize")
async def optimize_model(
    model_id: str,
    optimization_config: ModelOptimizationConfig,
    db: Session = Depends(get_db)
):
    """تحسين معلمات النموذج"""
    try:
        # التحقق من وجود النموذج
        model = await prediction_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # إنشاء مهمة التحسين
        job_id = f"optimize_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # بدء عملية التحسين في الخلفية
        BackgroundTasks.add_task(
            optimize_model_task,
            model_id=model_id,
            config=optimization_config,
            job_id=job_id
        )

        return {
            "job_id": job_id,
            "message": "Optimization started successfully"
        }

    except Exception as e:
        logger.error(f"Error starting model optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/cross-validate")
async def cross_validate_model(
    model_id: str,
    cv_config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """التحقق المتقاطع للنموذج"""
    try:
        # التحقق من وجود النموذج
        model = await prediction_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # تنفيذ التحقق المتقاطع
        validator = ModelValidator()
        results = await validator.validate_model(
            model=model,
            cv_config=cv_config
        )

        return results

    except Exception as e:
        logger.error(f"Error in cross validation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/validate")
async def validate_data(file: UploadFile = File(...)):
    """التحقق من صحة البيانات"""
    try:
        # قراءة البيانات
        df = pd.read_csv(file.file) if file.filename.endswith(
            '.csv') else pd.read_parquet(file.file)

        # التحقق من البيانات
        validator = DataValidator()
        validation_result = await validator.validate_data(df)

        return DataValidationResult(**validation_result)

    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/preprocess")
async def preprocess_data(
    preprocessing_config: Dict[str, Any],
    file: UploadFile = File(...)
):
    """معالجة البيانات"""
    try:
        # قراءة البيانات
        df = pd.read_csv(file.file) if file.filename.endswith(
            '.csv') else pd.read_parquet(file.file)

        # معالجة البيانات
        processor = AdvancedDataProcessor()
        processed_data = await processor.process_data(df, preprocessing_config)

        # حفظ البيانات المعالجة
        output_path = settings.REPORTS_DIR / \
            f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        processed_data.processed_df.to_parquet(output_path)

        return {
            "message": "Data processed successfully",
            "output_path": str(output_path),
            "stats": processed_data.stats
        }

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/report")
async def generate_model_report(model_id: str, db: Session = Depends(get_db)):
    """إنشاء تقرير شامل عن النموذج"""
    try:
        # التحقق من وجود النموذج
        model = db.query(ModelRecord).filter(
            ModelRecord.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # إنشاء التقرير
        report_generator = ReportGenerator()
        report_path = await report_generator.generate_model_report(
            model_metadata=model.to_dict(),
            performance_data=await get_performance_data(model_id, db),
            predictions_history=await get_predictions_history(model_id, db)
        )

        return FileResponse(
            path=report_path,
            filename=f"model_report_{model_id}.html",
            media_type="text/html"
        )

    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/export")
async def export_model(
    model_id: str,
    format: str = "onnx",
    db: Session = Depends(get_db)
):
    """تصدير النموذج بتنسيق محدد"""
    try:
        # التحقق من وجود النموذج
        model = await prediction_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # تصدير النموذج
        exporter = ModelExporter()
        export_path = await exporter.export_model(
            model=model,
            format=format,
            model_id=model_id
        )

        return FileResponse(
            path=export_path,
            filename=f"model_{model_id}.{format}",
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def batch_predict(
    model_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """تنفيذ تنبؤات متعددة"""
    pass


@router.post("/models/batch/train")
async def batch_train(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """تدريب عدة نماذج"""
    pass


@router.post("/models/{model_id}/version")
async def create_model_version(
    model_id: str,
    version_name: str,
    db: Session = Depends(get_db)
):
    """إنشاء نسخة من النموذج"""
    pass


@router.get("/models/{model_id}/versions")
async def list_model_versions(
    model_id: str,
    db: Session = Depends(get_db)
):
    """قائمة نسخ النموذج"""
    pass


@router.post("/models/{model_id}/rollback")
async def rollback_model(
    model_id: str,
    version: str,
    db: Session = Depends(get_db)
):
    """سترجاع نسخة سابقة"""
    pass


@router.get("/models/{model_id}/drift")
async def check_model_drift(
    model_id: str,
    start_date: datetime,
    end_date: datetime,
    db: Session = Depends(get_db)
):
    """فحص انحراف النموذج"""
    pass


@router.get("/models/{model_id}/alerts")
async def get_model_alerts(
    model_id: str,
    severity: str = Query("all", enum=["low", "medium", "high", "all"]),
    db: Session = Depends(get_db)
):
    """الحصول على تنبيهات النموذج"""
    pass


@router.post("/models/compare")
async def compare_models(
    model_ids: List[str],
    metric: str = Query(
        "accuracy", enum=["accuracy", "f1", "precision", "recall"]),
    db: Session = Depends(get_db)
):
    """مقارنة عدة نماذج"""
    pass


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """الحصول على إحصائيات النظام"""
    try:
        # الحصول على عدد النماذج النشطة
        active_models = await get_active_models_count(db)

        # الحصول على إجمالي التنبؤات
        total_predictions = await get_total_predictions_count(db)

        # الحصول على متوسط الدقة
        avg_accuracy = await get_average_accuracy(db)

        # الحصول على معدل التنبؤات في الساعة
        predictions_per_hour = await get_predictions_per_hour(db)

        # حساب صحة النظام (يمكن تخصيص هذا بناءً على معايير مختلفة)
        system_health = 100  # قيمة افتراضية

        return {
            "active_models": active_models,
            "total_predictions": total_predictions,
            "avg_accuracy": avg_accuracy,
            "system_health": system_health,
            "predictions_per_hour": predictions_per_hour
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system stats: {str(e)}"
        )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """تحميل ملف للمعالجة"""
    try:
        # التحقق من نوع الملف
        if not file.filename.endswith(('.csv', '.xlsx', '.parquet')):
            raise HTTPException(
                status_code=400,
                detail="نوع الملف غير مدعوم. الرجاء استخدام CSV, XLSX, أو Parquet"
            )

        # إنشاء معرف للمهمة
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # التأكد من وجود المجلدات المطلوبة
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # حفظ الملف مؤقتاً
        file_path = settings.UPLOAD_DIR / f"{job_id}_{file.filename}"

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file: {str(e)}"
            )

        # إنشاء سجل في قاعدة البيانات
        training_job = TrainingJob(
            model_id=job_id,
            status="processing",
            config={"filename": file.filename}
        )
        db.add(training_job)
        db.commit()

        # بدء معالجة الملف في الخلفية
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_file,
                file_path=file_path,
                job_id=job_id,
                db=db
            )
        else:
            # إذا لم يتم توفير background_tasks، قم بالمعالجة مباشرة
            asyncio.create_task(process_uploaded_file(file_path, job_id, db))

        return {
            "status": "success",
            "message": "تم تحميل الملف بنجاح وبدأت المعالجة",
            "job_id": job_id
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

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


async def get_active_models_count(db: Session) -> int:
    return db.query(ModelRecord).filter(ModelRecord.status == "active").count()


async def get_total_predictions_count(db: Session) -> int:
    return db.query(PredictionLog).count()


async def get_average_accuracy(db: Session) -> float:
    result = db.query(
        func.avg(ModelRecord.metrics['accuracy'].cast(Float))).scalar()
    return result or 0.0


async def get_predictions_per_hour(db: Session) -> int:
    hour_ago = datetime.utcnow() - timedelta(hours=1)
    return db.query(PredictionLog).filter(PredictionLog.timestamp >= hour_ago).count()


async def process_uploaded_file(file_path: Path, job_id: str, db: Session):
    """معالجة الملف المحمل وتدريب النموذج"""
    try:
        logger.info(f"Starting file processing for job {job_id}")

        # تحديث حالة المعالجة - تحميل البيانات
        update_processing_status(
            job_id, "data-loading", 10, "جاري تحميل البيانات...")

        # قراءة وتحليل البيانات
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            df = pd.read_parquet(file_path)

        logger.info(f"Data loaded successfully for job {job_id}")

        # معالجة البيانات
        update_processing_status(
            job_id, "preprocessing", 30, "جاري المعالجة لأولية...")
        processor = AdvancedDataProcessor()
        processed_data = await processor.process_data(df)

        logger.info(f"Data preprocessing completed for job {job_id}")

        # باقي الكود كما هو...

    except Exception as e:
        logger.error(f"Error processing file for job {job_id}: {str(e)}")
        update_processing_status(
            job_id, "failed", 0, f"فشل المعالجة: {str(e)}")
        # تحديث حالة المهمة في قاعدة البيانات
        training_job = db.query(TrainingJob).filter(
            TrainingJob.model_id == job_id).first()
        if training_job:
            training_job.status = "failed"
            training_job.error_message = str(e)
            db.commit()
    finally:
        # حذف الملف المؤقت
        if file_path.exists():
            file_path.unlink()


def update_processing_status(
    job_id: str,
    step: str,
    progress: int,
    message: str,
    result_url: str = None,
    additional_data: Dict[str, Any] = None
):
    """تحديث حالة معالجة الملف"""
    status_update = {
        "job_id": job_id,
        "step": step,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "completed_steps": []
    }

    # تحديث الخطوات المكتملة
    steps = ['data-loading', 'preprocessing',
             'model-selection', 'training', 'evaluation']
    current_step_index = steps.index(step)
    status_update['completed_steps'] = steps[:current_step_index]

    if result_url:
        status_update["result_url"] = result_url

    if additional_data:
        status_update.update(additional_data)

    # تحديث الحالة في الذاكرة
    processing_status[job_id] = status_update

    # بث التحديث عبر WebSocket
    for connection in websocket_connections.get(job_id, []):
        asyncio.create_task(connection.send_json(status_update))
