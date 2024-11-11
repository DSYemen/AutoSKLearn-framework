from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    log_loss, explained_variance_score
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.logging_config import logger
from app.utils.visualization import create_plot

class ModelEvaluator:
    """مقيم النموذج المتقدم مع تحليلات شاملة"""
    
    def __init__(self, model, problem_type: str):
        self.model = model
        self.problem_type = problem_type
        self.evaluation_results: Dict[str, Any] = {}
        self.plots_dir: Optional[Path] = None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """تقييم شامل للنموذج"""
        try:
            logger.info(f"بدء تقييم النموذج: {type(self.model).__name__}")
            
            # إنشاء مجلد للمخططات
            self.plots_dir = Path(settings.REPORTS_DIR) / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            
            # حساب التنبؤات
            y_pred = self.model.predict(X_test)
            y_prob = self._get_prediction_probabilities(X_test)
            
            # حساب المقاييس الأساسية
            self.evaluation_results['basic_metrics'] = self._calculate_basic_metrics(y_test, y_pred)
            
            # حساب المقاييس المتقدمة
            self.evaluation_results['advanced_metrics'] = self._calculate_advanced_metrics(
                y_test, y_pred, y_prob
            )
            
            # تحليل الأخطاء
            self.evaluation_results['error_analysis'] = self._analyze_errors(y_test, y_pred)
            
            # تحليل SHAP
            self.evaluation_results['feature_importance'] = self._analyze_feature_importance(X_test)
            
            # إنشاء المخططات
            self._create_evaluation_plots(X_test, y_test, y_pred, y_prob)
            
            logger.info("اكتمل تقييم النموذج", extra={"metrics": self.evaluation_results['basic_metrics']})
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"خطأ في تقييم النموذج: {str(e)}")
            raise

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """حساب المقاييس الأساسية"""
        metrics = {}
        
        if self.problem_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            })
        else:
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred)
            })
            
        return metrics

    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
        """حساب المقاييس المتقدمة"""
        metrics = {}
        
        if self.problem_type == 'classification':
            # مصفوفة الارتباك
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            if y_prob is not None:
                # ROC AUC
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                
                # منحنى الدقة-الاستدعاء
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
                
                # Log Loss
                metrics['log_loss'] = log_loss(y_true, y_prob)
        else:
            # تحليل البواقي
            residuals = y_true - y_pred
            metrics['residuals'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(pd.Series(residuals).skew()),
                'kurtosis': float(pd.Series(residuals).kurtosis())
            }
            
        return metrics

    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """تحليل الأخطاء"""
        error_analysis = {}
        
        if self.problem_type == 'classification':
            # تحليل الأخطاء حسب الفئة
            error_indices = y_true != y_pred
            error_analysis['error_rate'] = float(np.mean(error_indices))
            error_analysis['error_distribution'] = pd.Series(y_true[error_indices]).value_counts().to_dict()
            
        else:
            # تحليل البواقي
            residuals = y_true - y_pred
            error_analysis['residuals_stats'] = {
                'percentiles': np.percentile(residuals, [25, 50, 75]).tolist(),
                'outliers': len(residuals[np.abs(residuals) > 2 * np.std(residuals)])
            }
            
        return error_analysis

    def _analyze_feature_importance(self, X_test: np.ndarray) -> Dict[str, float]:
        """تحليل أهمية المميزات"""
        feature_importance = {}
        
        try:
            # SHAP Values
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_test)
            
            # حساب متوسط القيم المطلقة لـ SHAP
            feature_importance['shap'] = np.abs(shap_values.values).mean(0).tolist()
            
            # أهمية المميزات المدمجة في النموذج إذا كانت متوفرة
            if hasattr(self.model, 'feature_importances_'):
                feature_importance['model'] = self.model.feature_importances_.tolist()
                
        except Exception as e:
            logger.warning(f"لم يتم حساب SHAP values: {str(e)}")
            
        return feature_importance

    def _create_evaluation_plots(self, X_test: np.ndarray, y_true: np.ndarray, 
                               y_pred: np.ndarray, y_prob: Optional[np.ndarray]):
        """إنشاء المخططات التقييمية"""
        try:
            if self.problem_type == 'classification':
                # مصفوفة الارتباك
                plt.figure(figsize=(10, 8))
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
                plt.title('Confusion Matrix')
                self._save_plot('confusion_matrix.png')
                
                if y_prob is not None:
                    # منحنى ROC
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                    plt.figure(figsize=(8, 8))
                    plt.plot(fpr, tpr)
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.title(f'ROC Curve (AUC = {auc(fpr, tpr):.3f})')
                    self._save_plot('roc_curve.png')
            else:
                # مخطط البواقي
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_pred, y=y_true - y_pred)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title('Residuals Plot')
                self._save_plot('residuals.png')
                
            # SHAP Summary Plot
            shap.summary_plot(
                shap.Explainer(self.model)(X_test),
                plot_type="bar",
                show=False
            )
            self._save_plot('shap_summary.png')
            
        except Exception as e:
            logger.warning(f"خطأ في إنشاء المخططات: {str(e)}")

    def _get_prediction_probabilities(self, X: np.ndarray) -> Optional[np.ndarray]:
        """الحصول على احتمالات التنبؤ"""
        if self.problem_type == 'classification' and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def _save_plot(self, filename: str):
        """حفظ المخطط"""
        plt.savefig(self.plots_dir / filename)
        plt.close()

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص التقييم"""
        return {
            'model_type': type(self.model).__name__,
            'problem_type': self.problem_type,
            'basic_metrics': self.evaluation_results.get('basic_metrics', {}),
            'error_analysis': self.evaluation_results.get('error_analysis', {}),
            'plots_directory': str(self.plots_dir) if self.plots_dir else None,
            'evaluation_timestamp': datetime.now().isoformat()
        }