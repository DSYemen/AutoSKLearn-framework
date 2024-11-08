from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from app.logging_config import logger
import pandas as pd

def evaluate_model(model, data, problem_type):
    X_test, y_test = data

    y_pred = model.predict(X_test)

    if problem_type == 'classification':
        score = accuracy_score(y_test, y_pred)
        metric = 'Accuracy'
        report = classification_report(y_test, y_pred, output_dict=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
        plt.title('Classification Report')
        plt.tight_layout()
        plt.savefig('static/classification_report.png')
        plt.close()
        else:
            score = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metric = 'Mean Squared Error'

            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.tight_layout()
            plt.savefig('static/regression_plot.png')
            plt.close()

        logger.info(f"{metric}: {score}")
        if problem_type == 'regression':
            logger.info(f"R2 Score: {r2}")

        # SHAP values for model interpretability
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('static/shap_summary.png')
        plt.close()

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('static/feature_importance.png')
            plt.close()
        else:
            feature_importance = None

        # Learning curve
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_test, y_test, cv=5, 
            scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

        plt.figure(figsize=(10, 8))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig('static/learning_curve.png')
        plt.close()

        return {
            'metric': metric,
            'score': score,
            'r2_score': r2 if problem_type == 'regression' else None,
            'classification_report': 'static/classification_report.png' if problem_type == 'classification' else None,
            'regression_plot': 'static/regression_plot.png' if problem_type == 'regression' else None,
            'shap_plot': 'static/shap_summary.png',
            'feature_importance': 'static/feature_importance.png' if feature_importance is not None else None,
            'learning_curve': 'static/learning_curve.png'
        }