import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder
from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures


class AdvancedDataProcessor:

    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.target = None

    def process_data(self, df):
        self._identify_feature_types(df)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        df = self._encode_categorical_features(df)
        df = self._scale_numeric_features(df)
        df = self._create_datetime_features(df)
        df = self._feature_selection(df)
        df = self._feature_engineering(df)
        return df

    def _identify_feature_types(self, df):
        self.numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        self.datetime_features = df.select_dtypes(
            include=['datetime64']).columns.tolist()
        self.target = df.columns[-1]  # Assuming the last column is the target

    def _handle_missing_values(self, df):
        # For numeric features
        numeric_imputer = KNNImputer(n_neighbors=5)
        df[self.numeric_features] = numeric_imputer.fit_transform(
            df[self.numeric_features])

        # For categorical features
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[self.categorical_features] = categorical_imputer.fit_transform(
            df[self.categorical_features])

        return df

    def _handle_outliers(self, df):
        for feature in self.numeric_features:
            winsorizer = Winsorizer(capping_method='iqr',
                                    tail='both',
                                    fold=1.5)
            df[feature] = winsorizer.fit_transform(df[[feature]])
        return df

    def _encode_categorical_features(self, df):
        rare_encoder = RareLabelEncoder(tol=0.05, n_categories=1)
        df[self.categorical_features] = rare_encoder.fit_transform(
            df[self.categorical_features])

        if df[self.target].dtype == 'object':  # Classification task
            target_encoder = TargetEncoder()
            df[self.categorical_features] = target_encoder.fit_transform(
                df[self.categorical_features], df[self.target])
        else:  # Regression task
            onehot_encoder = OneHotEncoder(sparse=False,
                                           handle_unknown='ignore')
            encoded_features = onehot_encoder.fit_transform(
                df[self.categorical_features])
            encoded_feature_names = onehot_encoder.get_feature_names(
                self.categorical_features)
            df = pd.concat([
                df.drop(self.categorical_features, axis=1),
                pd.DataFrame(encoded_features,
                             columns=encoded_feature_names,
                             index=df.index)
            ],
                           axis=1)

        return df

    def _scale_numeric_features(self, df):
        scaler = RobustScaler()
        df[self.numeric_features] = scaler.fit_transform(
            df[self.numeric_features])
        return df

    def _create_datetime_features(self, df):
        if self.datetime_features:
            datetime_transformer = DatetimeFeatures(
                variables=self.datetime_features)
            df = datetime_transformer.fit_transform(df)
        return df

    def _feature_selection(self, df):
        # Remove low variance features
        var_threshold = VarianceThreshold(threshold=0.1)
        df_high_variance = var_threshold.fit_transform(
            df.drop(self.target, axis=1))
        high_variance_features = df.drop(
            self.target, axis=1).columns[var_threshold.get_support()]

        # Select K best features
        if df[self.target].dtype == 'object':  # Classification task
            selector = SelectKBest(f_classif,
                                   k=min(50, len(high_variance_features)))
        else:  # Regression task
            selector = SelectKBest(f_regression,
                                   k=min(50, len(high_variance_features)))

        X_selected = selector.fit_transform(df[high_variance_features],
                                            df[self.target])
        selected_features = high_variance_features[selector.get_support()]

        return df[selected_features.tolist() + [self.target]]

    def _feature_engineering(self, df):
        # Polynomial features for numeric columns
        combiner = CombineWithReferenceFeature(
            variables_to_combine=self.
            numeric_features[:
                             5],  # Limit to first 5 numeric features to avoid explosion
            reference_variables=self.numeric_features[:5],
            operations=['sum', 'prod'])
        df = combiner.fit_transform(df)

        # Dimensionality reduction if we have many features
        if df.shape[1] > 100:
            pca = PCA(n_components=100)
            pca_features = pca.fit_transform(df.drop(self.target, axis=1))
            df = pd.concat([
                pd.DataFrame(pca_features,
                             columns=[f'PCA_{i}' for i in range(100)],
                             index=df.index), df[self.target]
            ],
                           axis=1)

        return df


async def process_data(file):
    df = pd.read_csv(file.file)
    processor = AdvancedDataProcessor()
    processed_df = processor.process_data(df)

    # Generate a data profile report
    profile = ProfileReport(processed_df,
                            title="Pandas Profiling Report",
                            explorative=True)
    profile.to_file("static/profile_report.html")

    return processed_df, "static/profile_report.html"
