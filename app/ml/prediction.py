# app/ml/prediction.py
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import json
from fastapi import HTTPException
import asyncio

from app.core.config import settings
from app.core.logging_config import logger
from app.utils.cache import cache_manager
from app.utils.exceptions import PredictionError
from app.ml.data_processing import AdvancedDataProcessor

class PredictionService:
    """خدمة التنبؤ المتقدمة مع دعم للذاكرة المؤقتة والمراقبة"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, AdvancedDataProcessor] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        self.batch_size = settings.PREDICTION_BATCH_SIZE

    async def predict(self, model_id: str, features: Dict[str, Any]) -> Union[float, List[float]]:
        """تنفيذ التنبؤ مع المعالجة المسبقة والتحقق"""
        try:
            # التحقق من الذاكرة المؤقتة
            cache_key = self._generate_cache_key(model_id, features)
            cached_prediction = await cache_manager.get_cached_prediction(cache_key)
            if cached_prediction is not None:
                logger.info(f"تم استرجاع التنبؤ من الذاكرة المؤقتة: {model_id}")
                return cached_prediction

            # تحميل النموذج إذا لم يكن محملاً
            if model_id not in self.models:
                await self._load_model(model_id)

            # معالجة البيانات
            processed_features = await self._preprocess_features(model_id, features)

            # التنبؤ
            prediction = await self._make_prediction(model_id, processed_features)

            # تخزين في الذاكرة المؤقتة
            await cache_manager.cache_prediction(
                cache_key,
                prediction,
                ttl=settings.PREDICTION_CACHE_TTL
            )

            # تسجيل التنبؤ
            self._log_prediction(model_id, features, prediction)

            return prediction

        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {str(e)}")
            raise PredictionError(f"فشل التنبؤ: {str(e)}")

    async def batch_predict(self, model_id: str, features_list: List[Dict[str, Any]]) -> List[float]:
        """تنفيذ تنبؤات متعددة بكفاءة"""
        try:
            predictions = []
            for i in range(0, len(features_list), self.batch_size):
                batch = features_list[i:i + self.batch_size]
                batch_predictions = await asyncio.gather(*[
                    self.predict(model_id, features)
                    for features in batch
                ])
                predictions.extend(batch_predictions)
            return predictions

        except Exception as e:
            logger.error(f"خطأ في التنبؤ المجمع: {str(e)}")
            raise PredictionError(f"فشل التنبؤ المجمع: {str(e)}")

    async def _load_model(self, model_id: str) -> None:
        """تحميل النموذج والمعالج المسبق"""
        try:
            model_dir = Path(settings.MODELS_DIR) / model_id
            
            # تحميل النموذج
            model_path = model_dir / 'model.joblib'
            if not model_path.exists():
                raise FileNotFoundError(f"النموذج غير موجود: {model_id}")
            
            self.models[model_id] = joblib.load(model_path)
            
            # تحميل المعالج المسبق
            preprocessor_path = model_dir / 'preprocessor.joblib'
            if preprocessor_path.exists():
                self.preprocessors[model_id] = joblib.load(preprocessor_path)
            
            # تحميل البيانات الوصفية
            metadata_path = model_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                await cache_manager.cache_model_metadata(model_id, metadata)
            
            logger.info(f"تم تحميل النموذج: {model_id}")

        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج: {str(e)}")
            raise

    async def _preprocess_features(self, model_id: str, features: Dict[str, Any]) -> np.ndarray:
        """معالجة المدخلات قبل التنبؤ"""
        try:
            # التحقق من صحة المدخلات
            self._validate_features(model_id, features)
            
            # تحويل إلى DataFrame
            features_df = pd.DataFrame([features])
            
            # تطبيق المعالجة المسبقة إذا كانت متوفرة
            if model_id in self.preprocessors:
                features_df = self.preprocessors[model_id].transform(features_df)
            
            return features_df.values

        except Exception as e:
            logger.error(f"خطأ في معالجة المدخلات: {str(e)}")
            raise

    async def _make_prediction(self, model_id: str, processed_features: np.ndarray) -> Union[float, List[float]]:
        """تنفيذ التنبؤ مع التحقق من الصحة"""
        try:
            model = self.models[model_id]
            
            # التنبؤ
            prediction = model.predict(processed_features)
            
            # التحقق من صحة النتيجة
            if not self._validate_prediction(prediction):
                raise ValueError("نتيجة التنبؤ غير صالحة")
            
            # إضافة درجة الثقة إذا كانت متوفرة
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_features)
                confidence = np.max(probabilities, axis=1)
                return {
                    'prediction': prediction.tolist(),
                    'confidence': confidence.tolist()
                }
            
            return prediction.tolist()

        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {str(e)}")
            raise

    def _validate_features(self, model_id: str, features: Dict[str, Any]) -> None:
        """التحقق من صحة المدخلات"""
        metadata = cache_manager.get_model_metadata(model_id)
        if not metadata:
            raise ValueError("البيانات الوصفية للنموذج غير متوفرة")

        required_features = metadata.get('features', [])
        missing_features = set(required_features) - set(features.keys())
        if missing_features:
            raise ValueError(f"المميزات المفقودة: {missing_features}")

    def _validate_prediction(self, prediction: np.ndarray) -> bool:
        """التحقق من صحة نتيجة التنبؤ"""
        return (
            prediction is not None and
            not np.any(np.isnan(prediction)) and
            not np.any(np.isinf(prediction))
        )

    def _generate_cache_key(self, model_id: str, features: Dict[str, Any]) -> str:
        """إنشاء مفتاح للذاكرة المؤقتة"""
        features_str = json.dumps(features, sort_keys=True)
        return f"pred:{model_id}:{hash(features_str)}"

    def _log_prediction(self, model_id: str, features: Dict[str, Any], prediction: Any) -> None:
        """تسجيل التنبؤ للمراقبة"""
        log_entry = {
            'model_id': model_id,
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.prediction_history.append(log_entry)
        
        # حفظ آخر N تنبؤ فقط
        if len(self.prediction_history) > settings.MAX_PREDICTION_HISTORY:
            self.prediction_history = self.prediction_history[-settings.MAX_PREDICTION_HISTORY:]

    def get_prediction_statistics(self, model_id: str) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبؤات"""
        model_predictions = [
            log for log in self.prediction_history
            if log['model_id'] == model_id
        ]
        
        if not model_predictions:
            return {}
        
        return {
            'total_predictions': len(model_predictions),
            'last_prediction': model_predictions[-1]['timestamp'],
            'average_confidence': np.mean([
                p.get('confidence', 1.0)
                for p in model_predictions
                if isinstance(p.get('prediction'), dict)
            ]),
            'prediction_distribution': pd.Series([
                p['prediction'] if isinstance(p['prediction'], (int, float))
                else p['prediction']['prediction']
                for p in model_predictions
            ]).value_counts().to_dict()
        }

    async def update_model(self, model_id: str, new_model_path: str) -> None:
        """تحديث النموذج مع الحفاظ على التوافق"""
        try:
            # تحميل النموذج الجديد للتحقق
            new_model = joblib.load(new_model_path)
            
            # التحقق من التوافق
            if not self._check_model_compatibility(model_id, new_model):
                raise ValueError("النموذج الجديد غير متوافق مع النموذج الحالي")
            
            # تحديث النموذج
            self.models[model_id] = new_model
            
            # تحديث البيانات الوصفية
            metadata = {
                'updated_at': datetime.utcnow().isoformat(),
                'model_type': type(new_model).__name__,
                'parameters': new_model.get_params()
            }
            await cache_manager.cache_model_metadata(model_id, metadata)
            
            logger.info(f"تم تحديث النموذج: {model_id}")

        except Exception as e:
            logger.error(f"خطأ في تحديث النموذج: {str(e)}")
            raise

    def _check_model_compatibility(self, model_id: str, new_model: Any) -> bool:
        """التحقق من توافق النموذج الجديد"""
        if model_id not in self.models:
            return True
            
        current_model = self.models[model_id]
        return (
            type(new_model) == type(current_model) and
            hasattr(new_model, 'predict') and
            hasattr(new_model, 'get_params')
        )