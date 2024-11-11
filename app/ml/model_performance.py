from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.core.logging_config import logger
from app.db.models import PredictionLog, ModelRecord
from app.utils.cache import cache_manager

async def get_performance_trend(model_id: str, db: Session, days: int = 30) -> Dict[str, List[float]]:
    """
    الحصول على اتجاه أداء النموذج
    """
    try:
        # تحديد نطاق التاريخ
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # استرجاع سجلات التنبؤ
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.timestamp >= start_date,
            PredictionLog.timestamp <= end_date
        ).order_by(PredictionLog.timestamp.asc()).all()
        
        # تنظيم البيانات
        timestamps = []
        accuracies = []
        
        for pred in predictions:
            if pred.actual is not None:  # فقط التنبؤات التي لها قيم فعلية
                timestamps.append(pred.timestamp)
                accuracies.append(1 if pred.prediction == pred.actual else 0)
        
        # حساب المتوسط المتحرك
        window_size = min(7, len(accuracies))  # نافذة 7 أيام أو أقل
        if accuracies:
            moving_avg = pd.Series(accuracies).rolling(window=window_size).mean().tolist()
        else:
            moving_avg = []
            
        return {
            "timestamps": [t.isoformat() for t in timestamps],
            "accuracy": moving_avg
        }
        
    except Exception as e:
        logger.error(f"Error getting performance trend: {str(e)}")
        return {"timestamps": [], "accuracy": []}

async def get_predictions_distribution(model_id: str, db: Session) -> Dict[str, int]:
    """
    الحصول على توزيع التنبؤات
    """
    try:
        # استرجاع التنبؤات الأخيرة
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).order_by(desc(PredictionLog.timestamp)).limit(1000).all()
        
        # حساب التوزيع
        distribution = {}
        for pred in predictions:
            prediction = str(pred.prediction)
            distribution[prediction] = distribution.get(prediction, 0) + 1
            
        return distribution
        
    except Exception as e:
        logger.error(f"Error getting predictions distribution: {str(e)}")
        return {}

async def get_feature_correlations(model_id: str, db: Session) -> Dict[str, float]:
    """
    الحصول على ارتباطات المميزات مع الهدف
    """
    try:
        # استرجاع البيانات من الذاكرة المؤقتة
        cached_correlations = await cache_manager.get_model_metadata(f"correlations:{model_id}")
        if cached_correlations:
            return cached_correlations
        
        # استرجاع سجلات التنبؤ مع القيم الفعلية
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.actual.isnot(None)
        ).order_by(desc(PredictionLog.timestamp)).limit(1000).all()
        
        if not predictions:
            return {}
            
        # تحويل البيانات إلى DataFrame
        data = []
        for pred in predictions:
            row = pred.input_data.copy()
            row['actual'] = pred.actual
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # حساب الارتباطات
        correlations = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            if column != 'actual':
                corr = df[column].corr(df['actual'])
                correlations[column] = corr
                
        # تخزين في الذاكرة المؤقتة
        await cache_manager.cache_model_metadata(
            f"correlations:{model_id}",
            correlations,
            ttl=3600  # تخزين لمدة ساعة
        )
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error getting feature correlations: {str(e)}")
        return {}

async def get_performance_data(model_id: str, db: Session) -> Dict[str, Any]:
    """
    الحصول على بيانات الأداء الشاملة
    """
    try:
        # استرجاع معلومات النموذج
        model = db.query(ModelRecord).filter(ModelRecord.model_id == model_id).first()
        if not model:
            raise ValueError("Model not found")
            
        # جمع البيانات
        performance_data = {
            "trend": await get_performance_trend(model_id, db),
            "distribution": await get_predictions_distribution(model_id, db),
            "correlations": await get_feature_correlations(model_id, db),
            "metrics": {
                "accuracy": model.metrics.get("accuracy", 0),
                "f1": model.metrics.get("f1", 0),
                "precision": model.metrics.get("precision", 0),
                "recall": model.metrics.get("recall", 0)
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        return {}

async def get_predictions_history(model_id: str, db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    """
    الحصول على سجل التنبؤات
    """
    try:
        # استرجاع التنبؤات الأخيرة
        predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).order_by(desc(PredictionLog.timestamp)).limit(limit).all()
        
        # تنسيق البيانات
        history = []
        for pred in predictions:
            history.append({
                "timestamp": pred.timestamp.isoformat(),
                "input": pred.input_data,
                "prediction": pred.prediction,
                "actual": pred.actual,
                "confidence": pred.confidence
            })
            
        return history
        
    except Exception as e:
        logger.error(f"Error getting predictions history: {str(e)}")
        return []

async def analyze_model_drift(model_id: str, db: Session) -> Dict[str, Any]:
    """
    تحليل انحراف النموذج
    """
    try:
        # استرجاع التنبؤات القديمة والحديثة
        recent_predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.actual.isnot(None)
        ).order_by(desc(PredictionLog.timestamp)).limit(500).all()
        
        if len(recent_predictions) < 100:  # نحتاج لعدد كافٍ من البيانات
            return {
                "drift_detected": False,
                "message": "Not enough data for drift analysis"
            }
            
        # تقسيم البيانات إلى مجموعتين
        mid_point = len(recent_predictions) // 2
        old_accuracy = sum(1 for p in recent_predictions[mid_point:] if p.prediction == p.actual) / mid_point
        new_accuracy = sum(1 for p in recent_predictions[:mid_point] if p.prediction == p.actual) / mid_point
        
        # تحديد الانحراف
        drift_threshold = 0.1  # 10% تغيير في الدقة
        drift_detected = abs(new_accuracy - old_accuracy) > drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "drift_magnitude": abs(new_accuracy - old_accuracy),
            "message": "Model drift detected" if drift_detected else "Model performance is stable"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing model drift: {str(e)}")
        return {
            "drift_detected": False,
            "message": f"Error in drift analysis: {str(e)}"
        } 