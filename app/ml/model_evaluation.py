# from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from app.logging_config import logger
# import pandas as pd

# def evaluate_model(model, data, problem_type):
#     X_test, y_test = data

#     y_pred = model.predict(X_test)

#     if problem_type == 'classification':
#         score = accuracy_score(y_test, y_pred)
#         metric = 'Accuracy'
#         report = classification_report(y_test, y_pred, output_dict=True)

#         plt.figure(figsize=(10, 8))
#         sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
#         plt.title('Classification Report')
#         plt.tight_layout()
#         plt.savefig('static/classification_report.png')
#         plt.close()
#     else:
#         score = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         metric = 'Mean Squared Error'

#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_test, y_pred, alpha=0.5)
#         plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#         plt.xlabel('Actual')
#         plt.ylabel('Predicted')
#         plt.title('Actual vs Predicted')
#         plt.tight_layout()
#         plt.savefig('static/regression_plot.png')
#         plt.close()

#         logger.info(f"{metric}: {score}")
#         if problem_type == 'regression':
#             logger.info(f"R2 Score: {r2}")

#         # SHAP values for model interpretability
#         explainer = shap.Explainer(model)
#         shap_values = explainer(X_test)

#         shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
#         plt.tight_layout()
#         plt.savefig('static/shap_summary.png')
#         plt.close()

#         # Feature importance
#         if hasattr(model, 'feature_importances_'):
#             feature_importance = pd.DataFrame({
#                 'feature': X_test.columns,
#                 'importance': model.feature_importances_
#             }).sort_values('importance', ascending=False)

#             plt.figure(figsize=(10, 8))
#             sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
#             plt.title('Top 20 Feature Importances')
#             plt.tight_layout()
#             plt.savefig('static/feature_importance.png')
#             plt.close()
#         else:
#             feature_importance = None

#         # Learning curve
#         from sklearn.model_selection import learning_curve
#         train_sizes, train_scores, test_scores = learning_curve(
#             model, X_test, y_test, cv=5, 
#             scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error',
#             n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

#         plt.figure(figsize=(10, 8))
#         plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
#         plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
#         plt.title('Learning Curve')
#         plt.xlabel('Training examples')
#         plt.ylabel('Score')
#         plt.legend(loc="best")
#         plt.tight_layout()
#         plt.savefig('static/learning_curve.png')
#         plt.close()

#         return {
#             'metric': metric,
#             'score': score,
#             'r2_score': r2 if problem_type == 'regression' else None,
#             'classification_report': 'static/classification_report.png' if problem_type == 'classification' else None,
#             'regression_plot': 'static/regression_plot.png' if problem_type == 'regression' else None,
#             'shap_plot': 'static/shap_summary.png',
#             'feature_importance': 'static/feature_importance.png' if feature_importance is not None else None,
#             'learning_curve': 'static/learning_curve.png'
#         }





# app/ml/model_evaluation.py
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import learning_curve
import shap
import plotly.graph_objects as go
import plotly.express as px
from app.core.logging_config import logger
from app.utils.exceptions import EvaluationError

class ModelEvaluator:
    def __init__(self, model, problem_type: str):
        self.model = model
        self.problem_type = problem_type
        self.evaluation_results = {}
        self.shap_values = None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)

            # Calculate basic metrics
            self.evaluation_results['basic_metrics'] = self._calculate_basic_metrics(y_test, y_pred)

            # Calculate advanced metrics
            self.evaluation_results['advanced_metrics'] = self._calculate_advanced_metrics(
                X_test, y_test, y_pred
            )

            # Generate learning curves
            self.evaluation_results['learning_curves'] = self._generate_learning_curves(
                X_test, y_test
            )

            # Calculate feature importance
            self.evaluation_results['feature_importance'] = self._calculate_feature_importance(X_test)

            # Generate SHAP values
            self.evaluation_results['shap_values'] = self._calculate_shap_values(X_test)

            logger.info("Model evaluation completed successfully")
            return self.evaluation_results

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise EvaluationError(f"Model evaluation failed: {str(e)}")

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = {}

        if self.problem_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            })

            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            })

        return metrics

    def _calculate_advanced_metrics(self, X_test: np.ndarray, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        metrics = {}

        if self.problem_type == 'classification':
            # ROC curve and AUC
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }

                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }

            # Classification report
            metrics['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True
            )

        else:  # regression
            # Residual analysis
            residuals = y_true - y_pred
            metrics['residuals'] = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skew': pd.Series(residuals).skew(),
                'kurtosis': pd.Series(residuals).kurtosis()
            }

            # Error distribution
            metrics['error_distribution'] = {
                'percentiles': np.percentile(
                    np.abs(residuals),
                    [25, 50, 75, 90, 95, 99]
                ).tolist()
            }

        return metrics

    def _generate_learning_curves(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate learning curves"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
            'test_scores_std': np.std(test_scores, axis=1).tolist()
        }

    def _calculate_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance"""
        feature_importance = {}

        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                np.abs(self.model.coef_)
            ))

        return feature_importance

    def _calculate_shap_values(self, X: np.ndarray) -> Optional[Dict[str, Any]]:
        """Calculate SHAP values for model interpretability"""
        try:
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X)

            return {
                'values': shap_values.values.tolist(),
                'base_values': shap_values.base_values.tolist(),
                'feature_names': [f"feature_{i}" for i in range(X.shape[1])]
            }
        except Exception as e:
            logger.warning(f"Could not calculate SHAP values: {str(e)}")
            return None

    def generate_evaluation_plots(self) -> Dict[str, Any]:
        """Generate evaluation plots"""
        plots = {}

        try:
            if self.problem_type == 'classification':
                plots['confusion_matrix'] = self._plot_confusion_matrix()
                plots['roc_curve'] = self._plot_roc_curve()
                plots['precision_recall_curve'] = self._plot_precision_recall_curve()
            else:
                plots['residuals'] = self._plot_residuals()
                plots['error_distribution'] = self._plot_error_distribution()
    
            plots['learning_curve'] = self._plot_learning_curve()
            plots['feature_importance'] = self._plot_feature_importance()
            plots['shap_summary'] = self._plot_shap_summary()
    
            return plots
    
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}")
            return {}
    
    def _plot_confusion_matrix(self) -> go.Figure:
        """Plot confusion matrix"""
        cm = self.evaluation_results['basic_metrics']['confusion_matrix']
    
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
    
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=600,
            height=600
        )
    
        return fig
    
    def _plot_roc_curve(self) -> go.Figure:
        """Plot ROC curve"""
        roc_data = self.evaluation_results['advanced_metrics']['roc_curve']
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name=f'ROC (AUC = {self.evaluation_results["advanced_metrics"]["roc_auc"]:.3f})'
        ))
    
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random'
        ))
    
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=600
        )
    
        return fig