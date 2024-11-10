# app/visualization/dashboard.py
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from app.core.config import settings


class DashboardGenerator:

    def __init__(self):
        self.figures = {}
        self.layout_template = "plotly_white"

    def generate_dashboard(self, model_data: Dict[str, Any],
                           performance_data: Dict[str, Any],
                           predictions_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard with multiple visualizations
        """
        self.figures = {
            "model_performance":
            self._create_performance_plot(performance_data),
            "feature_importance":
            self._create_feature_importance_plot(model_data),
            "predictions_analysis":
            self._create_predictions_analysis(predictions_data),
            "confusion_matrix":
            self._create_confusion_matrix(performance_data),
            "learning_curve":
            self._create_learning_curve(performance_data),
            "residuals_plot":
            self._create_residuals_plot(predictions_data)
        }

        return self._generate_html_dashboard()

    def _create_performance_plot(
            self, performance_data: Dict[str, Any]) -> go.Figure:
        """
        Create performance metrics trend plot
        """
        df = pd.DataFrame(performance_data['metrics_history'])

        fig = go.Figure()
        for metric in df.columns:
            if metric != 'timestamp':
                fig.add_trace(
                    go.Scatter(x=df['timestamp'],
                               y=df[metric],
                               name=metric,
                               mode='lines+markers'))

        fig.update_layout(title="Model Performance Trends",
                          xaxis_title="Time",
                          yaxis_title="Metric Value",
                          template=self.layout_template,
                          hovermode='x unified')

        return fig

    def _create_feature_importance_plot(
            self, model_data: Dict[str, Any]) -> go.Figure:
        """
        Create feature importance visualization
        """
        feature_importance = pd.DataFrame(
            model_data['feature_importance']).sort_values('importance',
                                                          ascending=True)

        fig = go.Figure(
            go.Bar(x=feature_importance['importance'],
                   y=feature_importance['feature'],
                   orientation='h'))

        fig.update_layout(title="Feature Importance",
                          xaxis_title="Importance Score",
                          yaxis_title="Feature",
                          template=self.layout_template)

        return fig

    def _create_predictions_analysis(
            self, predictions_data: pd.DataFrame) -> go.Figure:
        """
        Create predictions distribution and analysis plot
        """
        fig = go.Figure()

        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=predictions_data['actual'],
                       y=predictions_data['predicted'],
                       mode='markers',
                       name='Predictions',
                       marker=dict(size=8,
                                   color=predictions_data['error'],
                                   colorscale='RdYlBu',
                                   showscale=True)))

        # Perfect prediction line
        max_val = max(predictions_data['actual'].max(),
                      predictions_data['predicted'].max())
        min_val = min(predictions_data['actual'].min(),
                      predictions_data['predicted'].min())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val],
                       y=[min_val, max_val],
                       mode='lines',
                       name='Perfect Prediction',
                       line=dict(dash='dash', color='gray')))

        fig.update_layout(title="Actual vs Predicted Values",
                          xaxis_title="Actual Values",
                          yaxis_title="Predicted Values",
                          template=self.layout_template)

        return fig

    def _create_confusion_matrix(
            self, performance_data: Dict[str, Any]) -> go.Figure:
        """
        Create confusion matrix heatmap
        """
        if 'confusion_matrix' not in performance_data:
            return None

        cm = performance_data['confusion_matrix']
        labels = performance_data.get('labels', list(range(len(cm))))

        fig = go.Figure(data=go.Heatmap(z=cm,
                                        x=labels,
                                        y=labels,
                                        colorscale='RdBu',
                                        text=cm,
                                        texttemplate="%{text}",
                                        textfont={"size": 16},
                                        hoverongaps=False))

        fig.update_layout(title="Confusion Matrix",
                          xaxis_title="Predicted",
                          yaxis_title="Actual",
                          template=self.layout_template)

        return fig

    def _create_learning_curve(self, performance_data: Dict[str,
                                                            Any]) -> go.Figure:
        """
        Create learning curve visualization
        """
        if 'learning_curve' not in performance_data:
            return None

        lc = performance_data['learning_curve']

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=lc['train_sizes'],
                       y=lc['train_scores_mean'],
                       mode='lines+markers',
                       name='Training Score',
                       error_y=dict(type='data',
                                    array=lc['train_scores_std'],
                                    visible=True)))

        fig.add_trace(
            go.Scatter(x=lc['train_sizes'],
                       y=lc['test_scores_mean'],
                       mode='lines+markers',
                       name='Validation Score',
                       error_y=dict(type='data',
                                    array=lc['test_scores_std'],
                                    visible=True)))

        fig.update_layout(title="Learning Curve",
                          xaxis_title="Training Examples",
                          yaxis_title="Score",
                          template=self.layout_template)

        return fig

    def _create_residuals_plot(self,
                               predictions_data: pd.DataFrame) -> go.Figure:
        """
        Create residuals analysis plot
        """
        residuals = predictions_data['actual'] - predictions_data['predicted']

        fig = go.Figure()

        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=predictions_data['predicted'],
                       y=residuals,
                       mode='markers',
                       name='Residuals',
                       marker=dict(size=8,
                                   color=abs(residuals),
                                   colorscale='Viridis',
                                   showscale=True)))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(title="Residuals Analysis",
                          xaxis_title="Predicted Values",
                          yaxis_title="Residuals",
                          template=self.layout_template)

        return fig

    def _generate_html_dashboard(self) -> str:
        """
        Generate HTML dashboard combining all visualizations
        """
        dashboard_html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>ML Model Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link href="https://cdn.tailwindcss.com" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto px-4 py-8">
                    <h1 class="text-3xl font-bold mb-8">ML Model Dashboard</h1>
        
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {% for name, figure in figures.items() %}
                            {% if figure %}
                            <div class="bg-white rounded-lg shadow-lg p-4">
                                <h2 class="text-xl font-semibold mb-4">{{ name | title }}</h2>
                                <div id="{{ name }}_plot"></div>
                                <script>
                                    var plotData = {{ figure | safe }};
                                    Plotly.newPlot('{{ name }}_plot', plotData.data, plotData.layout);
                                </script>
                            </div>
                            {% endif %}
                        {% endfor %}
                    </div>
        
                    <div class="mt-8 bg-white rounded-lg shadow-lg p-6">
                        <h2 class="text-xl font-semibold mb-4">Model Summary</h2>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {% for metric, value in metrics.items() %}
                            <div class="bg-gray-50 p-4 rounded-lg">
                                <h3 class="text-sm text-gray-600">{{ metric | title }}</h3>
                                <p class="text-2xl font-bold">{{ value }}</p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """

        return dashboard_html
