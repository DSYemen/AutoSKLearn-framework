# app/ml/data_processing.py
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from category_encoders import TargetEncoder
from dataclasses import dataclass
import hashlib
from datetime import datetime
from app.core.config import settings
from app.core.logging_config import logger

@dataclass
class ProcessedData:
    """حاوية لنتائج معالجة البيانات"""
    processed_df: pd.DataFrame
    feature_importance: Dict[str, float]
    dataset_hash: str
    stats: Dict[str, Any]
    target: str
    problem_type: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

class AdvancedDataProcessor:
    """معالج البيانات المتقدم مع ميزات إضافية"""
    
    def __init__(self):
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.datetime_features: List[str] = []
        self.target: Optional[str] = None
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
    async def process_data(self, file) -> ProcessedData:
        """المعالجة الرئيسية للبيانات"""
        try:
            # قراءة البيانات
            df = await self._read_data(file)
            
            # إنشاء هاش للبيانات
            dataset_hash = self._create_dataset_hash(df)
            
            # تحليل البيانات الأولي
            stats = self._analyze_data(df)
            
            # تحديد نوع المشكلة والهدف
            self.target = self._identify_target(df)
            problem_type = self._determine_problem_type(df[self.target])
            
            # تحديد أنواع الأعمدة
            self._identify_feature_types(df)
            
            # معالجة القيم المفقودة
            df = self._handle_missing_values(df)
            
            # معالجة القيم الشاذة
            df = self._handle_outliers(df)
            
            # ترميز المتغيرات الفئوية
            df = self._encode_categorical_features(df)
            
            # تطبيع المتغيرات العددية
            df = self._scale_numerical_features(df)
            
            # إنشاء متغيرات جديدة
            df = self._create_features(df)
            
            # اختيار المتغيرات
            df, feature_importance = self._select_features(df)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = self._split_data(df)
            
            return ProcessedData(
                processed_df=df,
                feature_importance=feature_importance,
                dataset_hash=dataset_hash,
                stats=stats,
                target=self.target,
                problem_type=problem_type,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
            
        except Exception as e:
            logger.error(f"خطأ في معالجة البيانات: {str(e)}")
            raise
            
    async def _read_data(self, file) -> pd.DataFrame:
        """قراءة البيانات من الملف"""
        try:
            file_content = await file.read()
            if file.filename.endswith('.csv'):
                return pd.read_csv(file_content)
            elif file.filename.endswith('.xlsx'):
                return pd.read_excel(file_content)
            elif file.filename.endswith('.parquet'):
                return pd.read_parquet(file_content)
            else:
                raise ValueError("نوع الملف غير مدعوم")
        except Exception as e:
            logger.error(f"خطأ في قراءة الملف: {str(e)}")
            raise

    def _create_dataset_hash(self, df: pd.DataFrame) -> str:
        """إنشاء هاش فريد للبيانات"""
        data_string = df.to_json()
        return hashlib.md5(data_string.encode()).hexdigest()

    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل البيانات وإنشاء إحصائيات"""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # إحصائيات للمتغيرات العددية
        numeric_stats = df.describe().to_dict()
        stats["numeric_stats"] = numeric_stats
        
        # إحصائيات للمتغيرات الفئوية
        categorical_stats = {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include=['object']).columns
        }
        stats["categorical_stats"] = categorical_stats
        
        return stats

    def _identify_target(self, df: pd.DataFrame) -> str:
        """تحديد عمود الهدف"""
        # يمكن تحسين هذه المنطق حسب احتياجاتك
        return df.columns[-1]

    def _determine_problem_type(self, target_series: pd.Series) -> str:
        """تحديد نوع المشكلة (تصنيف أو انحدار)"""
        if target_series.dtype == 'object' or len(target_series.unique()) < 10:
            return 'classification'
        return 'regression'

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """تحديد أنواع المتغيرات"""
        for column in df.columns:
            if column == self.target:
                continue
                
            if pd.api.types.is_numeric_dtype(df[column]):
                self.numerical_features.append(column)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                self.datetime_features.append(column)
            else:
                self.categorical_features.append(column)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المفقودة"""
        # للمتغيرات العددية
        if self.numerical_features:
            num_imputer = KNNImputer(n_neighbors=5)
            df[self.numerical_features] = num_imputer.fit_transform(df[self.numerical_features])
            self.imputers['numerical'] = num_imputer

        # للمتغيرات الفئوية
        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[self.categorical_features] = cat_imputer.fit_transform(df[self.categorical_features])
            self.imputers['categorical'] = cat_imputer

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم الشاذة"""
        for feature in self.numerical_features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[feature] = df[feature].clip(lower_bound, upper_bound)
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ترميز المتغيرات الفئوية"""
        if not self.categorical_features:
            return df

        # استخدام Target Encoding للمتغيرات الفئوية
        target_encoder = TargetEncoder()
        df[self.categorical_features] = target_encoder.fit_transform(
            df[self.categorical_features],
            df[self.target]
        )
        self.encoders['target'] = target_encoder

        return df

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """تطبيع المتغيرات العددية"""
        if not self.numerical_features:
            return df

        scaler = RobustScaler()
        df[self.numerical_features] = scaler.fit_transform(df[self.numerical_features])
        self.scalers['numerical'] = scaler

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء متغيرات جديدة"""
        # معالجة المتغيرات الزمنية
        for feature in self.datetime_features:
            df[f"{feature}_year"] = df[feature].dt.year
            df[f"{feature}_month"] = df[feature].dt.month
            df[f"{feature}_day"] = df[feature].dt.day
            df[f"{feature}_dayofweek"] = df[feature].dt.dayofweek
            
            # التشفير الدائري للشهر
            df[f"{feature}_month_sin"] = np.sin(2 * np.pi * df[feature].dt.month / 12)
            df[f"{feature}_month_cos"] = np.cos(2 * np.pi * df[feature].dt.month / 12)

        # إنشاء تفاعلات بين المتغيرات العددية
        if len(self.numerical_features) > 1:
            for i, feat1 in enumerate(self.numerical_features[:-1]):
                for feat2 in self.numerical_features[i+1:]:
                    df[f"{feat1}_{feat2}_interaction"] = df[feat1] * df[feat2]

        return df

    def _select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """اختيار المتغيرات الأكثر أهمية"""
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # اختيار عدد المتغيرات
        k = min(settings.MAX_FEATURES, len(X.columns))

        # اختيار المتغيرات بناءً على نوع المشكلة
        if self._determine_problem_type(y) == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=k)

        # تطبيق الاختيار
        X_selected = selector.fit_transform(X, y)
        
        # الحصول على أهمية المتغيرات
        feature_importance = dict(zip(
            X.columns[selector.get_support()],
            selector.scores_[selector.get_support()]
        ))

        # إنشاء DataFrame جديد مع المتغيرات المختارة
        selected_features = X.columns[selector.get_support()].tolist()
        df_selected = pd.concat([
            df[selected_features],
            df[self.target]
        ], axis=1)

        return df_selected, feature_importance

    def _split_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """تقسيم البيانات إلى تدريب واختبار"""
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[self.target])
        y = df[self.target]

        return train_test_split(
            X.values,
            y.values,
            test_size=0.2,
            random_state=42,
            stratify=y if self._determine_problem_type(y) == 'classification' else None
        )
