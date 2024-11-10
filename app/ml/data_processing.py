# app/ml/data_processing.py
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder, SumEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from ydata_profiling import ProfileReport
from dataclasses import dataclass
from app.core.config import Settings
from app.core.logging_config import logger


@dataclass
class ProcessingResults:
    """Data class to store processing results and metadata"""
    processed_df: pd.DataFrame
    profile_path: str
    feature_importance: dict
    processing_stats: dict
    warnings: List[str]


class FeatureEngineer:
    """Handles feature engineering operations"""

    def __init__(self):
        self.encoder = SumEncoder()
        self.pca = None
        self.feature_combinations = []

    def create_polynomial_features(
            self, df: pd.DataFrame,
            numeric_features: List[str]) -> pd.DataFrame:
        """Creates polynomial features from numeric columns"""
        selected_features = numeric_features[:Settings.MAX_POLY_FEATURES]
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                df[f'{feat1}_{feat2}_prod'] = df[feat1] * df[feat2]
                df[f'{feat1}_{feat2}_sum'] = df[feat1] + df[feat2]
        return df

    def reduce_dimensions(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Applies dimensionality reduction if needed"""
        if df.shape[1] > Settings.MAX_FEATURES:
            self.pca = PCA(n_components=Settings.MAX_FEATURES)
            pca_features = self.pca.fit_transform(df.drop(target, axis=1))
            return pd.concat([
                pd.DataFrame(
                    pca_features,
                    columns=[f'PCA_{i}' for i in range(Settings.MAX_FEATURES)],
                    index=df.index), df[target]
            ],
                             axis=1)
        return df


class AdvancedDataProcessor:
    """Advanced data processing pipeline with error handling and logging"""

    def __init__(self):
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.datetime_features: List[str] = []
        self.target: Optional[str] = None
        self.feature_engineer = FeatureEngineer()
        self.processing_stats = {}
        self.warnings = []

    def process_data(self, df: pd.DataFrame) -> ProcessingResults:
        """Main processing pipeline with comprehensive error handling"""
        try:
            logger.info("Starting data processing pipeline")
            self._identify_feature_types(df)

            # Track initial data stats
            self.processing_stats['initial_shape'] = df.shape
            self.processing_stats['missing_values'] = df.isnull().sum(
            ).to_dict()

            # Process pipeline
            df = self._handle_missing_values(df)
            df = self._handle_outliers(df)
            df = self._encode_categorical_features(df)
            df = self._scale_numeric_features(df)
            df = self._create_datetime_features(df)
            df = self._feature_selection(df)
            df = self.feature_engineer.create_polynomial_features(
                df, self.numeric_features)
            df = self.feature_engineer.reduce_dimensions(df, self.target)

            # Generate profile report
            profile_path = self._generate_profile_report(df)

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(df)

            logger.info("Data processing completed successfully")
            return ProcessingResults(processed_df=df,
                                     profile_path=profile_path,
                                     feature_importance=feature_importance,
                                     processing_stats=self.processing_stats,
                                     warnings=self.warnings)

        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """Identifies and validates feature types"""
        self.numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        self.datetime_features = df.select_dtypes(
            include=['datetime64']).columns.tolist()

        if not self.numeric_features and not self.categorical_features:
            raise ValueError("No valid features found in the dataset")

        # Identify target variable
        self.target = df.columns[-1]
        if self.target in self.numeric_features:
            self.numeric_features.remove(self.target)
        elif self.target in self.categorical_features:
            self.categorical_features.remove(self.target)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling with multiple strategies"""
        try:
            initial_missing = df.isnull().sum().sum()

            # Numeric features
            if self.numeric_features:
                if df[self.numeric_features].isnull().sum().sum() / len(
                        df) < 0.1:
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = KNNImputer(n_neighbors=5)
                df[self.numeric_features] = imputer.fit_transform(
                    df[self.numeric_features])

            # Categorical features
            if self.categorical_features:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[self.categorical_features] = cat_imputer.fit_transform(
                    df[self.categorical_features])

            final_missing = df.isnull().sum().sum()
            self.processing_stats[
                'missing_values_handled'] = initial_missing - final_missing

            return df
        except Exception as e:
            logger.error(f"Error in missing value handling: {str(e)}")
            raise

    # ... (remaining methods with similar improvements)

    def _generate_profile_report(self, df: pd.DataFrame) -> str:
        """Generates an enhanced profile report"""
        try:
            profile = ProfileReport(df,
                                    title="Data Analysis Report",
                                    explorative=True,
                                    minimal=False)
            profile_path = "static/profile_report.html"
            profile.to_file(profile_path)
            return profile_path
        except Exception as e:
            logger.warning(f"Could not generate profile report: {str(e)}")
            self.warnings.append("Profile report generation failed")
            return ""


# async def process_data(file) -> Tuple[pd.DataFrame, str]:
#     """Async wrapper for data processing"""
#     try:
#         df = pd.read_csv(file.file)
#         processor = AdvancedDataProcessor()
#         results = processor.process_data(df)

#         return results.processed_df, results.profile_path
#     except Exception as e:
#         logger.error(f"Error processing file: {str(e)}")
#         raise

# app/ml/data_processing.py (continued)

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced outlier detection and handling"""
        try:
            outlier_stats = {}
            for feature in self.numeric_features:
                # Calculate initial statistics
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = len(df[(df[feature] < (Q1 - 1.5 * IQR)) |
                                       (df[feature] > (Q3 + 1.5 * IQR))])

                # Apply Winsorization
                winsorizer = Winsorizer(capping_method='iqr',
                                        tail='both',
                                        fold=1.5,
                                        variables=[feature])
                df[feature] = winsorizer.fit_transform(df[[feature]])

                # Store statistics
                outlier_stats[feature] = outlier_count

            self.processing_stats['outliers_handled'] = outlier_stats
            return df

        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced categorical encoding with multiple strategies"""
        try:
            if not self.categorical_features:
                return df

            encoding_stats = {}

            # Handle rare categories first
            rare_encoder = RareLabelEncoder(
                tol=0.05, n_categories=1, variables=self.categorical_features)
            df = rare_encoder.fit_transform(df)
            encoding_stats['rare_categories'] = rare_encoder.encoder_dict_

            # Choose encoding strategy based on target type
            if df[self.target].dtype == 'object':  # Classification
                target_encoder = TargetEncoder(cols=self.categorical_features,
                                               smoothing=10)
                df[self.categorical_features] = target_encoder.fit_transform(
                    df[self.categorical_features], df[self.target])
                encoding_stats['encoding_method'] = 'target_encoding'
            else:  # Regression
                onehot_encoder = OneHotEncoder(sparse=False,
                                               handle_unknown='ignore',
                                               drop='if_binary')
                encoded_features = onehot_encoder.fit_transform(
                    df[self.categorical_features])
                encoded_feature_names = onehot_encoder.get_feature_names(
                    self.categorical_features)

                # Create new dataframe with encoded features
                df = pd.concat([
                    df.drop(self.categorical_features, axis=1),
                    pd.DataFrame(encoded_features,
                                 columns=encoded_feature_names,
                                 index=df.index)
                ],
                               axis=1)
                encoding_stats['encoding_method'] = 'onehot_encoding'
                encoding_stats['new_features'] = len(encoded_feature_names)

            self.processing_stats['categorical_encoding'] = encoding_stats
            return df

        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def _scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent feature scaling with multiple options"""
        try:
            if not self.numeric_features:
                return df

            scaling_stats = {}

            # Choose scaler based on data characteristics
            for feature in self.numeric_features:
                skewness = df[feature].skew()
                has_outliers = abs(skewness) > 1

                if has_outliers:
                    scaler = RobustScaler()
                    scaling_method = 'robust'
                else:
                    scaler = StandardScaler()
                    scaling_method = 'standard'

                df[feature] = scaler.fit_transform(df[[feature]])
                
                scaling_stats[feature] = {
                    'method': scaling_method,
                    'skewness_before': skewness,
                    'skewness_after': df[feature].skew()
                }

            self.processing_stats['scaling'] = scaling_stats
            return df

        except Exception as e:
            logger.error(f"Error scaling numeric features: {str(e)}")
            raise

    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced datetime feature extraction"""
        try:
            if not self.datetime_features:
                return df

            datetime_stats = {}

            for feature in self.datetime_features:
                new_features = []

                # Basic datetime components
                df[f'{feature}_year'] = df[feature].dt.year
                df[f'{feature}_month'] = df[feature].dt.month
                df[f'{feature}_day'] = df[feature].dt.day
                df[f'{feature}_dayofweek'] = df[feature].dt.dayofweek
                new_features.extend(['year', 'month', 'day', 'dayofweek'])

                # Advanced features
                df[f'{feature}_is_weekend'] = df[feature].dt.dayofweek.isin(
                    [5, 6])
                df[f'{feature}_quarter'] = df[feature].dt.quarter
                df[f'{feature}_is_month_end'] = df[feature].dt.is_month_end
                new_features.extend(['is_weekend', 'quarter', 'is_month_end'])

                # Cyclical encoding for periodic features
                df[f'{feature}_month_sin'] = np.sin(2 * np.pi *
                                                    df[feature].dt.month / 12)
                df[f'{feature}_month_cos'] = np.cos(2 * np.pi *
                                                    df[feature].dt.month / 12)
                new_features.extend(['month_sin', 'month_cos'])

                datetime_stats[feature] = new_features

                # Drop original datetime column
                df = df.drop(feature, axis=1)

            self.processing_stats['datetime_features'] = datetime_stats
            return df

        except Exception as e:
            logger.error(f"Error creating datetime features: {str(e)}")
            raise

    def _calculate_feature_importance(self, df: pd.DataFrame) -> dict:
        """Calculate feature importance using multiple methods"""
        try:
            feature_importance = {}
            X = df.drop(self.target, axis=1)
            y = df[self.target]

            # Correlation based importance
            if not df[self.target].dtype == 'object':
                correlations = abs(X.corrwith(y))
                feature_importance['correlation'] = correlations.to_dict()

            # Mutual information based importance
            if df[self.target].dtype == 'object':
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(X, y)
                feature_importance['mutual_info'] = dict(
                    zip(X.columns, mi_scores))
            else:
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X, y)
                feature_importance['mutual_info'] = dict(
                    zip(X.columns, mi_scores))

            return feature_importance

        except Exception as e:
            logger.warning(f"Error calculating feature importance: {str(e)}")
            return {}

    # app/ml/data_processing.py (continued)

    def _feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature selection with multiple strategies"""
        try:
            selection_stats = {}
            X = df.drop(self.target, axis=1)
            y = df[self.target]

            # 1. Remove constant and quasi-constant features
            variance_selector = VarianceThreshold(threshold=0.01)
            X_var = variance_selector.fit_transform(X)
            selected_features = X.columns[
                variance_selector.get_support()].tolist()

            selection_stats['variance_threshold'] = {
                'removed_features': len(X.columns) - len(selected_features),
                'remaining_features': len(selected_features)
            }

            # 2. Correlation analysis
            if len(selected_features) > 1:
                correlation_matrix = X[selected_features].corr().abs()
                upper = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape),
                            k=1).astype(bool))
                to_drop = [
                    column for column in upper.columns
                    if any(upper[column] > 0.95)
                ]
                selected_features = [
                    f for f in selected_features if f not in to_drop
                ]

                selection_stats['correlation_analysis'] = {
                    'removed_features':
                    len(to_drop),
                    'high_correlation_pairs':
                    [(col, upper[col][upper[col] > 0.95].index[0])
                     for col in to_drop]
                }

            # 3. Statistical feature selection
            if df[self.target].dtype == 'object':  # Classification
                selector = SelectKBest(score_func=f_classif,
                                       k=min(50, len(selected_features)))
            else:  # Regression
                selector = SelectKBest(score_func=f_regression,
                                       k=min(50, len(selected_features)))

            X_selected = selector.fit_transform(X[selected_features], y)
            final_features = np.array(selected_features)[
                selector.get_support()].tolist()

            selection_stats['statistical_selection'] = {
                'initial_features': len(selected_features),
                'final_features': len(final_features),
                'feature_scores': dict(zip(selected_features,
                                           selector.scores_))
            }

            # 4. Add target back to the dataset
            final_df = pd.concat([X[final_features], df[self.target]], axis=1)

            self.processing_stats['feature_selection'] = selection_stats
            return final_df

        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate dataset quality and raise warnings"""
        warnings = []

        # Check sample size
        if len(df) < 100:
            warnings.append("Small dataset size may affect model performance")

        # Check class balance for classification
        if df[self.target].dtype == 'object':
            class_counts = df[self.target].value_counts()
            if (class_counts / len(df)).min() < 0.1:
                warnings.append("Significant class imbalance detected")

        # Check feature cardinality
        for feature in self.categorical_features:
            if df[feature].nunique() > 100:
                warnings.append(f"High cardinality in feature {feature}")

        # Check correlation with target
        for feature in self.numeric_features:
            if df[self.target].dtype != 'object':
                correlation = df[feature].corr(df[self.target])
                if abs(correlation) < 0.01:
                    warnings.append(
                        f"Feature {feature} shows very low correlation with target"
                    )

        self.warnings.extend(warnings)

    def get_feature_metadata(self) -> dict:
        """Return metadata about features for documentation"""
        return {
            'numeric_features': {
                'count': len(self.numeric_features),
                'names': self.numeric_features,
                'scaling_methods': self.processing_stats.get('scaling', {})
            },
            'categorical_features': {
                'count':
                len(self.categorical_features),
                'names':
                self.categorical_features,
                'encoding_info':
                self.processing_stats.get('categorical_encoding', {})
            },
            'datetime_features': {
                'count':
                len(self.datetime_features),
                'names':
                self.datetime_features,
                'generated_features':
                self.processing_stats.get('datetime_features', {})
            },
            'target': {
                'name': self.target,
                'type':
                'categorical' if self.target_is_categorical else 'numeric'
            }
        }


class DatasetValidator:
    """Separate class for comprehensive dataset validation"""

    @staticmethod
    def validate_file_format(file) -> bool:
        """Validate file format and basic structure"""
        allowed_extensions = ['.csv', '.xlsx', '.parquet']
        return any(file.filename.endswith(ext) for ext in allowed_extensions)

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> List[str]:
        """Comprehensive data quality checks"""
        issues = []

        # Basic checks
        if df.empty:
            issues.append("Dataset is empty")
        if len(df.columns) < 2:
            issues.append("Dataset should have at least two columns")

        # Data type checks
        if not any(dt.kind in 'biufc' for dt in df.dtypes):
            issues.append("No numeric columns found")

        # Missing value checks
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percentages[missing_percentages > 50]
        if not high_missing.empty:
            issues.append(
                f"Columns with >50% missing values: {list(high_missing.index)}"
            )

        # Duplicate checks
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate rows")

        return issues


async def process_data(file) -> Tuple[pd.DataFrame, str]:
    """Enhanced async data processing with validation"""

    try:
        # Validate file format
        validator = DatasetValidator()
        if not validator.validate_file_format(file):
            raise ValueError("Unsupported file format")

        # Read file based on format
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(file.file)

        # Validate data quality
        issues = validator.validate_data_quality(df)
        if issues:
            logger.warning("Data quality issues found: %s", issues)

        # Process data
        processor = AdvancedDataProcessor()
        results = processor.process_data(df)

        # Log processing summary
        logger.info(
            "Data processing completed: %s", {
                'initial_shape': processor.processing_stats['initial_shape'],
                'final_shape': results.processed_df.shape,
                'warnings': len(results.warnings)
            })

        return results.processed_df, results.profile_path

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise
