# app/ml/model_validator.py
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import cross_val_score
from app.core.logging_config import logger

class ModelValidator:
    def __init__(self):
        self.validation_results = {}

    def validate_model(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      problem_type: str) -> Dict[str, Any]:
        """
        Comprehensive model validation
        """
        try:
            validation_results = {
                "cross_validation": self._perform_cross_validation(model, X, y, problem_type),
                "stability": self._check_model_stability(model, X, y),
                "assumptions": self._check_assumptions(model, X, y, problem_type),
                "robustness": self._check_robustness(model, X, y)
            }

            self.validation_results = validation_results
            return validation_results

        except Exception as e:
            logger.error(f"Error in model validation: {str(e)}")
            raise

    def _perform_cross_validation(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                problem_type: str) -> Dict[str, float]:
        """
        Perform cross-validation with multiple metrics
        """
        metrics = ['accuracy'] if problem_type == 'classification' else ['neg_mean_squared_error']
        cv_results = {}

        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=5, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

        return cv_results

    def _check_model_stability(self,
                             model: Any,
                             X: np.ndarray,
                             y: np.ndarray) -> Dict[str, float]:
        """
        Check model stability across different subsets
        """
        stability_scores = []
        n_splits = 5

        for _ in range(n_splits):
            # Random subsample
            indices = np.random.choice(len(X), size=int(len(X)*0.8), replace=False)
            X_subset = X[indices]
            y_subset = y[indices]

            # Train and score
            model.fit(X_subset, y_subset)
            score = model.score(X[~indices], y[~indices])
            stability_scores.append(score)

        return {
            'mean_score': np.mean(stability_scores),
            'score_std': np.std(stability_scores),
            'stability_index': 1 - np.std(stability_scores) / np.mean(stability_scores)
        }

    def _check_assumptions(self,
                         model: Any,
                         X: np.ndarray,
                         y: np.ndarray,
                         problem_type: str) -> Dict[str, Any]:
        """
        Check model assumptions based on problem type
        """
        assumptions = {}

        if problem_type == 'regression':
            # Check linearity
            predictions = model.predict(X)
            residuals = y - predictions

            assumptions['normality'] = self._check_normality(residuals)
            assumptions['homoscedasticity'] = self._check_homoscedasticity(predictions, residuals)

        return assumptions

    def _check_robustness(self,
                         model: Any,
                         X: np.ndarray,
                         y: np.ndarray) -> Dict[str, float]:
        """
        Check model robustness to noise and perturbations
        """
        # Add small noise to features
        X_noisy = X + np.random.normal(0, 0.1, X.shape)

        # Compare performance
        original_score = model.score(X, y)
        noisy_score = model.score(X_noisy, y)

        return {
            'original_score': original_score,
            'noisy_score': noisy_score,
            'robustness_score': noisy_score / original_score
        }