from typing import Dict, Any, List
import pandas as pd
import numpy as np
from app.core.logging_config import logger
from app.schemas.model import DataValidationResult

class DataValidator:
    """فئة للتحقق من صحة البيانات"""

    def __init__(self):
        self.validation_rules = {
            'missing_values': self._check_missing_values,
            'data_types': self._check_data_types,
            'duplicates': self._check_duplicates,
            'outliers': self._check_outliers,
            'value_ranges': self._check_value_ranges
        }

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

            return validation_results

        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            raise

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من القيم المفقودة"""
        result = {
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # حساب نسبة القيم المفقودة لكل عمود
        missing_stats = df.isnull().mean()
        
        for column, missing_ratio in missing_stats.items():
            if missing_ratio > 0.5:
                result['errors'].append(f"Column '{column}' has {missing_ratio*100:.1f}% missing values")
            elif missing_ratio > 0.1:
                result['warnings'].append(f"Column '{column}' has {missing_ratio*100:.1f}% missing values")

        result['stats']['missing_values'] = missing_stats.to_dict()
        return result

    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من أنواع البيانات"""
        result = {
            'errors': [],
            'warnings': [],
            'stats': {'dtypes': df.dtypes.astype(str).to_dict()}
        }

        # التحقق من تناسق أنواع البيانات
        for column in df.columns:
            if df[column].dtype == 'object':
                unique_count = df[column].nunique()
                if unique_count < 10 and len(df) > 1000:
                    result['warnings'].append(
                        f"Column '{column}' might be categorical but is stored as object"
                    )

        return result

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من القيم المكررة"""
        result = {
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # التحقق من الصفوف المكررة
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            percentage = (duplicates / len(df)) * 100
            if percentage > 10:
                result['errors'].append(f"High number of duplicate rows: {percentage:.1f}%")
            else:
                result['warnings'].append(f"Found {duplicates} duplicate rows ({percentage:.1f}%)")

        result['stats']['duplicate_rows'] = duplicates
        return result

    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من القيم الشاذة"""
        result = {
            'errors': [],
            'warnings': [],
            'stats': {'outliers': {}}
        }

        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > 0:
                percentage = (outliers / len(df)) * 100
                if percentage > 10:
                    result['warnings'].append(
                        f"Column '{column}' has {percentage:.1f}% outliers"
                    )
                result['stats']['outliers'][column] = outliers

        return result

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من نطاقات القيم"""
        result = {
            'errors': [],
            'warnings': [],
            'stats': {'ranges': {}}
        }

        for column in df.select_dtypes(include=[np.number]).columns:
            col_min = df[column].min()
            col_max = df[column].max()
            result['stats']['ranges'][column] = {'min': col_min, 'max': col_max}

            # التحقق من القيم السالبة في الأعمدة التي يفترض أن تكون موجبة
            if column.lower().contains(('count', 'amount', 'price', 'quantity')):
                if col_min < 0:
                    result['errors'].append(
                        f"Column '{column}' contains negative values"
                    )

        return result

    def _update_validation_results(self, validation_results: Dict[str, Any], 
                                 rule_results: Dict[str, Any]) -> None:
        """تحديث نتائج التحقق"""
        validation_results['errors'].extend(rule_results.get('errors', []))
        validation_results['warnings'].extend(rule_results.get('warnings', []))
        validation_results['stats'].update(rule_results.get('stats', {}))
        
        # تحديث حالة الصلاحية
        if rule_results.get('errors'):
            validation_results['is_valid'] = False

        # إضافة توصيات بناءً على النتائج
        self._add_recommendations(validation_results)

    def _add_recommendations(self, validation_results: Dict[str, Any]) -> None:
        """إضافة توصيات بناءً على نتائج التحقق"""
        if validation_results.get('stats', {}).get('missing_values'):
            validation_results['recommendations'].append(
                "Consider imputing missing values using appropriate methods"
            )

        if validation_results.get('stats', {}).get('outliers'):
            validation_results['recommendations'].append(
                "Review and handle outliers before model training"
            )

        if validation_results.get('stats', {}).get('duplicate_rows', 0) > 0:
            validation_results['recommendations'].append(
                "Remove or handle duplicate rows based on business rules"
            ) 