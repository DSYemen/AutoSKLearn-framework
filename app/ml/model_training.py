# app/ml/model_training.py
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import joblib
from datetime import datetime
from pathlib import Path
import json

from app.core.logging_config import logger
from app.core.config import settings
from app.utils.alerts import AlertSystem
from app.utils.exceptions import TrainingError

class ModelTrainer:
    """مدرب النموذج المتقدم مع دعم للمراقبة والتتبع"""
    
    def __init__(self, model: BaseEstimator, problem_type: str):
        self.model = model
        self.problem_type = problem_type
        self.metrics: Dict[str, float] = {}
        self.training_history: Dict[str, list] = {
            'train_scores': [],
            'val_scores': [],
            'learning_rate': [],
            'batch_size': []
        }
        self.alert_system = AlertSystem()
        self.model_path: Optional[Path] = None

    def train(self, X, y, validation_data: Optional[Tuple] = None) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """تدريب النموذج مع دعم للتحقق والمراقبة"""
        try:
            logger.info(f"بدء تدريب النموذج: {type(self.model).__name__}")
            start_time = datetime.now()

            # تقسيم البيانات إذا لم يتم توفير بيانات التحقق
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y if self.problem_type == 'classification' else None
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = validation_data

            # بدء جلسة MLflow
            with mlflow.start_run() as run:
                # تسجيل المعلمات
                mlflow.log_params(self.model.get_params())
                
                # تدريب النموذج مع التحقق
                self._train_with_validation(X_train, y_train, X_val, y_val)
                
                # حساب المقاييس
                self.metrics = self._calculate_metrics(X_train, X_val, y_train, y_val)
                
                # تسجيل المقاييس
                mlflow.log_metrics(self.metrics)
                
                # حفظ النموذج
                self._save_model(run.info.run_id)
                
                # تسجيل الأشكال البيانية
                self._log_training_plots()

            # حساب وقت التدريب
            training_time = (datetime.now() - start_time).total_seconds()
            self.metrics['training_time'] = training_time

            # التحقق من أداء النموذج وإرسال تنبيهات إذا لزم الأمر
            self._check_model_performance()

            logger.info("اكتمل تدريب النموذج", extra={"metrics": self.metrics})
            return self.model, self.metrics

        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {str(e)}")
            raise TrainingError(f"فشل تدريب النموذج: {str(e)}")

    def _train_with_validation(self, X_train, y_train, X_val, y_val):
        """تدريب النموذج مع مراقبة الأداء على مجموعة التحقق"""
        # التحقق من دعم التدريب التدريجي
        if hasattr(self.model, 'partial_fit'):
            self._incremental_training(X_train, y_train, X_val, y_val)
        else:
            # التدريب العادي
            self.model.fit(X_train, y_train)
            
            # حساب النتائج على مجموعة التدريب والتحقق
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            self.training_history['train_scores'].append(train_score)
            self.training_history['val_scores'].append(val_score)

    def _incremental_training(self, X_train, y_train, X_val, y_val):
        """التدريب التدريجي للنماذج التي تدعمه"""
        batch_size = 1000
        n_samples = len(X_train)
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            # تدريب على الدفعة
            self.model.partial_fit(
                X_batch, y_batch,
                classes=np.unique(y_train) if self.problem_type == 'classification' else None
            )
            
            # حساب النتائج كل عدة دفعات
            if (i + batch_size) % (5 * batch_size) == 0:
                train_score = self.model.score(X_train, y_train)
                val_score = self.model.score(X_val, y_val)
                
                self.training_history['train_scores'].append(train_score)
                self.training_history['val_scores'].append(val_score)
                self.training_history['batch_size'].append(batch_size)

    def _calculate_metrics(self, X_train, X_val, y_train, y_val) -> Dict[str, float]:
        """حساب مقاييس الأداء المختلفة"""
        metrics = {}
        
        if self.problem_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            metrics.update({
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'val_accuracy': accuracy_score(y_val, y_pred_val),
                'precision': precision_score(y_val, y_pred_val, average='weighted'),
                'recall': recall_score(y_val, y_pred_val, average='weighted'),
                'f1': f1_score(y_val, y_pred_val, average='weighted')
            })
            
            # إضافة ROC AUC إذا كان النموذج يدعم predict_proba
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score
                y_prob_val = self.model.predict_proba(X_val)
                metrics['roc_auc'] = roc_auc_score(y_val, y_prob_val[:, 1])
                
        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            metrics.update({
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'val_mse': mean_squared_error(y_val, y_pred_val),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
                'mae': mean_absolute_error(y_val, y_pred_val)
            })

        # إضافة نتائج التحقق المتقاطع
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5,
            scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        )
        metrics['cv_score_mean'] = np.mean(cv_scores)
        metrics['cv_score_std'] = np.std(cv_scores)

        return metrics

    def _save_model(self, run_id: str):
        """حفظ النموذج والبيانات الوصفية"""
        # إنشاء مجلد للنموذج
        model_dir = Path(settings.MODELS_DIR) / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # حفظ النموذج
        model_path = model_dir / 'model.joblib'
        joblib.dump(self.model, model_path)
        self.model_path = model_path
        
        # حفظ البيانات الوصفية
        metadata = {
            'model_type': type(self.model).__name__,
            'problem_type': self.problem_type,
            'metrics': self.metrics,
            'parameters': self.model.get_params(),
            'training_history': self.training_history,
            'created_at': datetime.now().isoformat()
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _log_training_plots(self):
        """تسجيل الأشكال البيانية للتدريب"""
        import matplotlib.pyplot as plt
        
        # منحنى التعلم
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['train_scores'], label='Training Score')
        plt.plot(self.training_history['val_scores'], label='Validation Score')
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        mlflow.log_figure(plt.gcf(), "learning_curve.png")
        plt.close()

    def _check_model_performance(self):
        """التحقق من أداء النموذج وإرسال تنبيهات إذا لزم الأمر"""
        # التحقق من التلاؤم الزائد
        train_score = self.metrics.get('train_accuracy', self.metrics.get('train_r2'))
        val_score = self.metrics.get('val_accuracy', self.metrics.get('val_r2'))
        
        if train_score - val_score > settings.OVERFITTING_THRESHOLD:
            self.alert_system.send_alert(
                "تحذير: تم اكتشاف تلاؤم زائد",
                f"الفرق بين نتيجة التدريب والتحقق: {train_score - val_score:.4f}"
            )
        
        # التحقق من الأداء العام
        if val_score < settings.MINIMUM_PERFORMANCE_THRESHOLD:
            self.alert_system.send_alert(
                "تحذير: أداء النموذج منخفض",
                f"نتيجة التحقق: {val_score:.4f}"
            )

    def update(self, X_new, y_new) -> Tuple[BaseEstimator, Dict[str, float]]:
        """تحديث النموذج ببيانات جديدة"""
        try:
            if not hasattr(self.model, 'partial_fit'):
                raise TrainingError("النموذج لا يدعم التحديث التدريجي")
            
            # تحديث النموذج
            self.model.partial_fit(X_new, y_new)
            
            # حساب المقاييس الجديدة
            new_metrics = self._calculate_metrics(X_new, X_new, y_new, y_new)
            
            return self.model, new_metrics
            
        except Exception as e:
            logger.error(f"خطأ في تحديث النموذج: {str(e)}")
            raise TrainingError(f"فشل تحديث النموذج: {str(e)}")