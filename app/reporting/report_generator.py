# app/reporting/report_generator.py
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from app.core.logging_config import logger
from app.core.config import settings

class ReportGenerator:
    def __init__(self):
        self.reports_path = Path("reports")
        self.reports_path.mkdir(exist_ok=True)

    def generate_model_report(self, 
                            model_metadata: Dict[str, Any],
                            performance_data: Dict[str, Any],
                            predictions_history: List[Dict[str, Any]]) -> str:
        """
        Generate comprehensive model performance report
        """
        try:
            report_data = {
                "model_info": self._generate_model_info(model_metadata),
                "performance_metrics": self._generate_performance_metrics(performance_data),
                "predictions_analysis": self._analyze_predictions(predictions_history),
                "visualizations": self._generate_visualizations(performance_data, predictions_history)
            }

            report_path = self._save_report(report_data)
            return report_path

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _generate_model_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model information section"""
        return {
            "model_type": metadata.get("model_type"),
            "creation_date": metadata.get("created_at"),
            "features": metadata.get("features", []),
            "parameters": metadata.get("parameters", {}),
            "training_summary": {
                "data_size": metadata.get("training_data_size"),
                "training_duration": metadata.get("training_duration"),
                "convergence_info": metadata.get("convergence_info", {})
            }
        }

    def _generate_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics section"""
        return {
            "current_metrics": performance_data.get("current_metrics", {}),
            "metrics_history": performance_data.get("metrics_history", []),
            "performance_trend": self._calculate_performance_trend(
                performance_data.get("metrics_history", [])
            )
        }

    def _generate_visualizations(self, 
                               performance_data: Dict[str, Any],
                               predictions_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate visualization plots"""
        visualizations = {}

        # Performance trend plot
        fig = self._create_performance_trend_plot(performance_data)
        visualizations["performance_trend"] = self._save_plot(fig, "performance_trend")

        # Predictions distribution plot
        fig = self._create_predictions_distribution_plot(predictions_history)
        visualizations["predictions_distribution"] = self._save_plot(fig, "predictions_dist")

        # Feature importance plot
        if "feature_importance" in performance_data:
            fig = self._create_feature_importance_plot(performance_data["feature_importance"])
            visualizations["feature_importance"] = self._save_plot(fig, "feature_importance")

        return visualizations

    def _create_performance_trend_plot(self, performance_data: Dict[str, Any]) -> go.Figure:
        """Create performance trend visualization"""
        metrics_history = pd.DataFrame(performance_data.get("metrics_history", []))

        fig = go.Figure()
        for metric in metrics_history.columns:
            if metric != "timestamp":
                fig.add_trace(go.Scatter(
                    x=metrics_history["timestamp"],
                    y=metrics_history[metric],
                    name=metric,
                    mode='lines+markers'
                ))

        fig.update_layout(
            title="Model Performance Trend",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            hovermode='x unified'
        )
        return fig

    def _save_plot(self, fig: go.Figure, name: str) -> str:
        """Save plot to HTML file"""
        plot_path = self.reports_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(plot_path))
        return str(plot_path)