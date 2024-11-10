# app/ml/model_training.py
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator
import mlflow
from app.core.logging_config import logger
from app.core.config import settings

class ModelTrainer:
    def __init__(self, model: BaseEstimator, problem_type: str):
        self.model = model
        self.problem_type = problem_type
        self.metrics = {}

    def train(self, X, y) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Train model with comprehensive logging and tracking"""
        try:
            logger.info(f"Starting model training: {type(self.model).__name__}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if self.problem_type == 'classification' else None
            )

            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.model.get_params())

                # Train model
                self.model.fit(X_train, y_train)

                # Calculate metrics
                self.metrics = self._calculate_metrics(X_train, X_test, y_train, y_test)

                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Save model
                mlflow.sklearn.log_model(self.model, "model")

            logger.info("Model training completed", extra={"metrics": self.metrics})
            return self.model, self.metrics

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _calculate_metrics(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        metrics = {}

        # Training metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        if self.problem_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics.update({
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred, average='weighted'),
                'recall': recall_score(y_test, test_pred, average='weighted'),
                'f1': f1_score(y_test, test_pred, average='weighted')
            })
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics.update({
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred)
            })

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5,
            scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        )
        metrics['cv_score_mean'] = np.mean(cv_scores)
        metrics['cv_score_std'] = np.std(cv_scores)

        return metrics