from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from app.core.logging_config import logger
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydantic import BaseModel, Field

class DataValidationResult(BaseModel):
    """نتيجة التحقق من البيانات"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    stats: Dict[str, Any]
    recommendations: List[str]
    validation_time: datetime = Field(default_factory=datetime.utcnow)

class ModelValidator:
    """فئة للتحقق من صحة النماذج والبيانات"""

    def __init__(self):
        self.validation_rules = {
            'missing_values': self._check_missing_values,
            'data_types': self._check_data_types,
            'duplicates': self._check_duplicates,
            'outliers': self._check_outliers,
            'value_ranges': self._check_value_ranges
        }

    async def validate_model(
        self,
        model: Any,
        cv_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """التحقق المتقاطع للنموذج"""
        try:
            X = cv_config.get('X')
            y = cv_config.get('y')
            cv = cv_config.get('cv', 5)
            scoring = cv_config.get('scoring', 'accuracy')

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            return {
                'scores': scores.tolist(),
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'cv_folds': cv,
                'scoring': scoring
            }

        except Exception as e:
            logger.error(f"Error in model validation: {str(e)}")
            raise

    async def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """التحقق من صحة البيانات"""
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {},
                'recommendations': []
            }

            # تنفيذ جميع فحوصات التحقق
            for rule_name, rule_func in self.validation_rules.items():
                result = rule_func(df)
                self._update_validation_results(validation_results, result)

            # إضافة إحصائيات عامة
            validation_results['stats'].update({
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            })

            return DataValidationResult(**validation_results)

        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            raise

    def validate_predictions(
        self,
        predictions: np.ndarray,
        actual: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """التحقق من صحة التنبؤات"""
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }

            # التحقق من القيم غير الصالحة
            if np.isnan(predictions).any():
                validation_results['errors'].append({
                    'type': 'nan_values',
                    'message': "Predictions contain NaN values"
                })
                validation_results['is_valid'] = False

            if np.isinf(predictions).any():
                validation_results['errors'].append({
                    'type': 'inf_values',
                    'message': "Predictions contain infinite values"
                })
                validation_results['is_valid'] = False

            # إحصائيات التنبؤات
            validation_results['stats'] = {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions))
            }

            # التحقق من الدقة إذا كانت القيم الفعلية متوفرة
            if actual is not None:
                accuracy = np.mean(predictions == actual)
                validation_results['stats']['accuracy'] = float(accuracy)

            return validation_results

        except Exception as e:
            logger.error(f"Error in predictions validation: {str(e)}")
            raise

    def validate_model_performance(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """التحقق من أداء النموذج"""
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'metrics_status': {}
            }

            for metric, value in metrics.items():
                if metric in thresholds:
                    threshold = thresholds[metric]
                    if value < threshold:
                        validation_results['warnings'].append({
                            'type': 'low_performance',
                            'message': f"{metric} ({value:.4f}) is below threshold ({threshold})",
                            'details': {'metric': metric, 'value': value, 'threshold': threshold}
                        })
                        validation_results['metrics_status'][metric] = 'warning'
                    else:
                        validation_results['metrics_status'][metric] = 'ok'

            return validation_results

        except Exception as e:
            logger.error(f"Error in performance validation: {str(e)}")
            raise

    async def validate_feature_importance(
        self,
        feature_importance: Dict[str, float],
        min_importance: float = 0.01
    ) -> Dict[str, Any]:
        """التحقق من أهمية المميزات"""
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'recommendations': []
            }

            # تحديد المميزات ذات الأهمية المنخفضة
            low_importance_features = [
                feature for feature, importance in feature_importance.items()
                if importance < min_importance
            ]

            if low_importance_features:
                validation_results['warnings'].append(
                    f"Features with low importance: {', '.join(low_importance_features)}"
                )
                validation_results['recommendations'].append(
                    "Consider removing or combining low importance features"
                )

            return validation_results

        except Exception as e:
            logger.error(f"Error in feature importance validation: {str(e)}")
            raise

    async def validate_data_drift(
        self,
        training_data: pd.DataFrame,
        new_data: pd.DataFrame,
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """التحقق من انحراف البيانات"""
        try:
            validation_results = {
                'drift_detected': False,
                'drifted_features': [],
                'drift_scores': {},
                'recommendations': []
            }

            for column in training_data.columns:
                if training_data[column].dtype in ['int64', 'float64']:
                    # حساب الانحراف للمتغيرات العددية
                    training_mean = training_data[column].mean()
                    training_std = training_data[column].std()
                    new_mean = new_data[column].mean()
                    new_std = new_data[column].std()

                    mean_drift = abs(training_mean - new_mean) / training_mean
                    std_drift = abs(training_std - new_std) / training_std

                    drift_score = max(mean_drift, std_drift)
                    validation_results['drift_scores'][column] = drift_score

                    if drift_score > drift_threshold:
                        validation_results['drift_detected'] = True
                        validation_results['drifted_features'].append(column)

            if validation_results['drift_detected']:
                validation_results['recommendations'].append(
                    "Consider retraining the model with recent data"
                )

            return validation_results

        except Exception as e:
            logger.error(f"Error in data drift validation: {str(e)}")
            raise

    async def validate_model_stability(
        self,
        performance_history: List[Dict[str, float]],
        stability_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """التحقق من استقرار النموذج"""
        try:
            validation_results = {
                'is_stable': True,
                'warnings': [],
                'metrics_stability': {},
                'recommendations': []
            }

            for metric in performance_history[0].keys():
                values = [record[metric] for record in performance_history]
                std = np.std(values)
                mean = np.mean(values)
                cv = std / mean if mean != 0 else float('inf')

                validation_results['metrics_stability'][metric] = {
                    'mean': float(mean),
                    'std': float(std),
                    'cv': float(cv)
                }

                if cv > stability_threshold:
                    validation_results['is_stable'] = False
                    validation_results['warnings'].append(
                        f"High variability in {metric}: CV = {cv:.4f}"
                    )

            if not validation_results['is_stable']:
                validation_results['recommendations'].append(
                    "Consider implementing ensemble methods or regularization"
                )

            return validation_results

        except Exception as e:
            logger.error(f"Error in stability validation: {str(e)}")
            raise 