# app/ml/model_validator.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from app.core.logging_config import logger
from app.schemas.model import DataValidationResult
from sklearn.model_selection import cross_val_score
import pandas as pd

class ModelValidator:
    """فئة للتحقق من صحة النماذج"""

    async def validate_model(
        self,
        model: Any,
        cv_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        التحقق المتقاطع للنموذج
        """
        try:
            # استخراج البيانات من التكوين
            X = cv_config.get('X')
            y = cv_config.get('y')
            cv = cv_config.get('cv', 5)
            scoring = cv_config.get('scoring', 'accuracy')

            # التحقق المتقاطع
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

    async def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        التحقق من صحة البيانات
        """
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {},
                'recommendations': []
            }

            # التحقق من القيم المفقودة
            missing_stats = df.isnull().sum()
            if missing_stats.any():
                validation_results['warnings'].append(
                    "Dataset contains missing values"
                )
                validation_results['recommendations'].append(
                    "Consider imputing missing values"
                )

            # التحقق من القيم الشاذة
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    validation_results['warnings'].append(
                        f"Column {col} contains {outliers} outliers"
                    )

            # التحقق من التوازن في المتغيرات الفئوية
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                if (value_counts / len(df)).max() > 0.95:
                    validation_results['warnings'].append(
                        f"Column {col} is highly imbalanced"
                    )

            # إحصائيات عامة
            validation_results['stats'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': missing_stats.to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }

            return validation_results

        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            raise

    def validate_predictions(
        self,
        predictions: np.ndarray,
        actual: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        التحقق من صحة التنبؤات
        """
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }

            # التحقق من القيم غير الصالحة
            if np.isnan(predictions).any():
                validation_results['errors'].append("Predictions contain NaN values")
                validation_results['is_valid'] = False

            if np.isinf(predictions).any():
                validation_results['errors'].append("Predictions contain infinite values")
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
        """
        التحقق من أداء النموذج
        """
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
                        validation_results['warnings'].append(
                            f"{metric} ({value:.4f}) is below threshold ({threshold})"
                        )
                        validation_results['metrics_status'][metric] = 'warning'
                    else:
                        validation_results['metrics_status'][metric] = 'ok'

            return validation_results

        except Exception as e:
            logger.error(f"Error in performance validation: {str(e)}")
            raise