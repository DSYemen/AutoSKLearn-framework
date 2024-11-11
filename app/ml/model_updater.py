# app/ml/model_updater.py
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Optional, Dict, Any

from app.core.config import settings
from app.core.logging_config import logger
from app.ml.data_processing import AdvancedDataProcessor
from app.ml.model_selection import ModelSelector
from app.ml.model_training import ModelTrainer
from app.ml.model_evaluation import ModelEvaluator
from app.utils.alerts import AlertSystem
from app.utils.exceptions import ModelUpdateError

class ModelUpdater:
    """محدث النماذج التلقائي مع مراقبة الأداء"""
    
    def __init__(self):
        self.alert_system = AlertSystem()
        self.last_update: Optional[datetime] = None
        self.update_history: Dict[str, list] = {
            'timestamps': [],
            'performance_changes': [],
            'model_changes': []
        }
        self.current_performance: Dict[str, float] = {}

    async def start(self):
        """بدء عملية التحديث التلقائي"""
        logger.info("بدء خدمة تحديث النماذج التلقائي")
        while True:
            try:
                await self._check_and_update_models()
                await asyncio.sleep(settings.MODEL_UPDATE_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"خطأ في دورة تحديث النماذج: {str(e)}")
                await asyncio.sleep(60)  # انتظار قبل المحاولة مرة أخرى

    async def _check_and_update_models(self):
        """التحقق من الحاجة لتحديث النماذج وتنفيذ التحديث"""
        try:
            # التحقق من وجود بيانات جديدة
            new_data = await self._get_new_data()
            if new_data is None:
                return

            # تحليل البيانات الجديدة
            data_quality = self._analyze_data_quality(new_data)
            if not data_quality['is_valid']:
                self.alert_system.send_alert(
                    "تحذير: مشاكل في جودة البيانات الجديدة",
                    data_quality['issues']
                )
                return

            # تحديث النماذج
            for model_id in self._get_active_models():
                await self._update_single_model(model_id, new_data)

            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"خطأ في عملية التحديث: {str(e)}")
            self.alert_system.send_alert(
                "خطأ في تحديث النماذج",
                str(e)
            )

    async def _get_new_data(self) -> Optional[pd.DataFrame]:
        """الحصول على البيانات الجديدة"""
        try:
            data_path = Path(settings.NEW_DATA_PATH)
            if not data_path.exists():
                return None

            # التحقق من تاريخ آخر تعديل
            last_modified = datetime.fromtimestamp(data_path.stat().st_mtime)
            if self.last_update and last_modified <= self.last_update:
                return None

            # قراءة البيانات
            if data_path.suffix == '.csv':
                return pd.read_csv(data_path)
            elif data_path.suffix == '.parquet':
                return pd.read_parquet(data_path)
            else:
                raise ValueError(f"نوع الملف غير مدعوم: {data_path.suffix}")

        except Exception as e:
            logger.error(f"خطأ في قراءة البيانات الجديدة: {str(e)}")
            return None

    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل جودة البيانات الجديدة"""
        issues = []
        
        # التحقق من القيم المفقودة
        missing_pct = df.isnull().mean()
        if (missing_pct > settings.MAX_MISSING_THRESHOLD).any():
            issues.append(f"نسبة عالية من القيم المفقودة: {missing_pct.max():.2%}")

        # التحقق من القيم الشاذة
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            if (z_scores > settings.OUTLIER_THRESHOLD).sum() / len(df) > 0.1:
                issues.append(f"قيم شاذة كثيرة في العمود {col}")

        # التحقق من توزيع البيانات
        for col in df.select_dtypes(include=[np.number]).columns:
            skewness = df[col].skew()
            if abs(skewness) > settings.MAX_SKEWNESS:
                issues.append(f"انحراف كبير في توزيع العمود {col}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'stats': {
                'row_count': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
        }

    def _get_active_models(self) -> list:
        """الحصول على قائمة النماذج النشطة"""
        # يمكن تحسين هذه الدالة للحصول على النماذج من قاعدة البيانات
        return settings.ACTIVE_MODELS

    async def _update_single_model(self, model_id: str, new_data: pd.DataFrame):
        """تحديث نموذج واحد"""
        try:
            # معالجة البيانات
            processor = AdvancedDataProcessor()
            processed_data = await processor.process_data(new_data)

            # الحصول على النموذج الحالي
            current_model = self._load_model(model_id)
            current_metrics = self.current_performance.get(model_id, {})

            # تدريب نموذج جديد
            trainer = ModelTrainer(current_model, processed_data.problem_type)
            updated_model, new_metrics = trainer.update(
                processed_data.X_train,
                processed_data.y_train
            )

            # تقييم التحسن
            if self._is_improvement(current_metrics, new_metrics):
                # حفظ النموذج المحدث
                self._save_model(model_id, updated_model, new_metrics)
                
                # تحديث السجلات
                self._update_history(model_id, current_metrics, new_metrics)
                
                # إرسال تنبيه بالتحسن
                self.alert_system.send_alert(
                    f"تم تحديث النموذج {model_id}",
                    f"تحسن الأداء من {current_metrics.get('score', 0):.4f} إلى {new_metrics.get('score', 0):.4f}"
                )
            
            else:
                logger.info(f"لم يتم تحديث النموذج {model_id} - لم يتحسن الأداء")

        except Exception as e:
            logger.error(f"خطأ في تحديث النموذج {model_id}: {str(e)}")
            raise ModelUpdateError(f"فشل تحديث النموذج {model_id}: {str(e)}")

    def _load_model(self, model_id: str) -> Any:
        """تحميل النموذج من الملف"""
        model_path = Path(settings.MODELS_DIR) / f"{model_id}.joblib"
        if not model_path.exists():
            raise ModelUpdateError(f"النموذج غير موجود: {model_id}")
        return joblib.load(model_path)

    def _save_model(self, model_id: str, model: Any, metrics: Dict[str, float]):
        """حفظ النموذج المحدث"""
        model_path = Path(settings.MODELS_DIR) / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
        # حفظ البيانات الوصفية
        metadata = {
            'model_id': model_id,
            'updated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'parameters': model.get_params()
        }
        
        metadata_path = Path(settings.MODELS_DIR) / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _is_improvement(self, current_metrics: Dict[str, float], 
                       new_metrics: Dict[str, float]) -> bool:
        """التحقق من تحسن الأداء"""
        if not current_metrics:
            return True

        primary_metric = settings.PRIMARY_METRIC
        improvement_threshold = settings.IMPROVEMENT_THRESHOLD

        current_score = current_metrics.get(primary_metric, 0)
        new_score = new_metrics.get(primary_metric, 0)

        return (new_score - current_score) > improvement_threshold

    def _update_history(self, model_id: str, old_metrics: Dict[str, float], 
                       new_metrics: Dict[str, float]):
        """تحديث سجل التحديثات"""
        self.update_history['timestamps'].append(datetime.now().isoformat())
        self.update_history['performance_changes'].append({
            'model_id': model_id,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvement': {
                metric: new_metrics.get(metric, 0) - old_metrics.get(metric, 0)
                for metric in new_metrics
            }
        })
        
        self.current_performance[model_id] = new_metrics

    def get_update_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص التحديثات"""
        return {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_count': len(self.update_history['timestamps']),
            'performance_trends': self._calculate_performance_trends(),
            'current_performance': self.current_performance
        }

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """حساب اتجاهات الأداء"""
        if not self.update_history['performance_changes']:
            return {}

        trends = {}
        for model_id in self.current_performance.keys():
            model_changes = [
                change for change in self.update_history['performance_changes']
                if change['model_id'] == model_id
            ]
            
            if model_changes:
                trends[model_id] = {
                    'improvement_rate': sum(1 for change in model_changes 
                                         if any(v > 0 for v in change['improvement'].values())) / len(model_changes),
                    'average_improvement': {
                        metric: np.mean([change['improvement'].get(metric, 0) for change in model_changes])
                        for metric in self.current_performance[model_id].keys()
                    }
                }

        return trends