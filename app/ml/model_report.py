from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from app.core.config import settings
from app.core.logging_config import logger
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    """فئة لإنشاء تقارير النماذج"""

    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.template_env = Environment(
            loader=FileSystemLoader('templates/reports')
        )

    async def generate_model_report(
        self,
        model_metadata: Dict[str, Any],
        performance_data: Dict[str, Any],
        predictions_history: List[Dict[str, Any]]
    ) -> Path:
        """
        إنشاء تقرير شامل عن النموذج
        """
        try:
            # إنشاء مسار التقرير
            report_path = self.reports_dir / f"model_report_{model_metadata['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            # إنشاء المخططات
            plots = self._create_report_plots(performance_data)

            # تحضير بيانات التقرير
            report_data = {
                "model_metadata": model_metadata,
                "performance_data": performance_data,
                "predictions_history": predictions_history,
                "plots": plots,
                "generated_at": datetime.now().isoformat()
            }

            # توليد التقرير
            template = self.template_env.get_template('model_report.html')
            report_html = template.render(**report_data)

            # حفظ التقرير
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            logger.info(f"تم إنشاء تقرير النموذج: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"خطأ في إنشاء تقرير النموذج: {str(e)}")
            raise

    def _create_report_plots(self, performance_data: Dict[str, Any]) -> Dict[str, str]:
        """إنشاء مخططات التقرير"""
        plots = {}
        
        try:
            # منحنى الأداء
            performance_fig = go.Figure()
            performance_fig.add_trace(go.Scatter(
                x=performance_data['timestamps'],
                y=performance_data['accuracy'],
                mode='lines+markers',
                name='الدقة'
            ))
            plots['performance'] = performance_fig.to_html(full_html=False)

            # توزيع التنبؤات
            distribution_fig = px.histogram(
                x=performance_data['predictions'],
                title='توزيع التنبؤات'
            )
            plots['distribution'] = distribution_fig.to_html(full_html=False)

            # أهمية المميزات
            importance_fig = px.bar(
                x=list(performance_data['feature_importance'].values()),
                y=list(performance_data['feature_importance'].keys()),
                orientation='h',
                title='أهمية المميزات'
            )
            plots['importance'] = importance_fig.to_html(full_html=False)

            # مصفوفة الارتباك
            confusion_fig = px.imshow(
                performance_data['confusion_matrix'],
                labels=dict(x="التنبؤ", y="القيمة الفعلية"),
                title='مصفوفة الارتباك'
            )
            plots['confusion'] = confusion_fig.to_html(full_html=False)

            return plots

        except Exception as e:
            logger.error(f"خطأ في إنشاء مخططات التقرير: {str(e)}")
            return {}

    async def generate_comparison_report(
        self,
        models_data: List[Dict[str, Any]],
        comparison_metrics: Dict[str, Any]
    ) -> Path:
        """
        إنشاء تقرير مقارنة بين النماذج
        """
        try:
            # إنشاء مسار التقرير
            report_path = self.reports_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            # إنشاء مخططات المقارنة
            comparison_plots = self._create_comparison_plots(models_data, comparison_metrics)

            # تحضير بيانات التقرير
            report_data = {
                "models_data": models_data,
                "comparison_metrics": comparison_metrics,
                "plots": comparison_plots,
                "generated_at": datetime.now().isoformat()
            }

            # توليد التقرير
            template = self.template_env.get_template('comparison_report.html')
            report_html = template.render(**report_data)

            # حفظ التقرير
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            logger.info(f"تم إنشاء تقرير المقارنة: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"خطأ في إنشاء تقرير المقارنة: {str(e)}")
            raise

    def _create_comparison_plots(
        self,
        models_data: List[Dict[str, Any]],
        comparison_metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """إنشاء مخططات المقارنة"""
        plots = {}
        
        try:
            # مقارنة المقاييس
            metrics_fig = go.Figure()
            for metric, values in comparison_metrics.items():
                metrics_fig.add_trace(go.Bar(
                    x=list(values.keys()),
                    y=list(values.values()),
                    name=metric
                ))
            plots['metrics_comparison'] = metrics_fig.to_html(full_html=False)

            # مقارنة أهمية المميزات
            importance_fig = go.Figure()
            for model in models_data:
                importance_fig.add_trace(go.Bar(
                    x=list(model['feature_importance'].values()),
                    y=list(model['feature_importance'].keys()),
                    name=model['name'],
                    orientation='h'
                ))
            plots['importance_comparison'] = importance_fig.to_html(full_html=False)

            return plots

        except Exception as e:
            logger.error(f"خطأ في إنشاء مخططات المقارنة: {str(e)}")
            return {}

    async def generate_drift_report(
        self,
        model_id: str,
        drift_analysis: Dict[str, Any]
    ) -> Path:
        """
        إنشاء تقرير انحراف النموذج
        """
        try:
            # إنشاء مسار التقرير
            report_path = self.reports_dir / f"drift_report_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            # إنشاء مخططات الانحراف
            drift_plots = self._create_drift_plots(drift_analysis)

            # تحضير بيانات التقرير
            report_data = {
                "model_id": model_id,
                "drift_analysis": drift_analysis,
                "plots": drift_plots,
                "generated_at": datetime.now().isoformat()
            }

            # توليد التقرير
            template = self.template_env.get_template('drift_report.html')
            report_html = template.render(**report_data)

            # حفظ التقرير
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            logger.info(f"تم إنشاء تقرير الانحراف: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"خطأ في إنشاء تقرير الانحراف: {str(e)}")
            raise

    def _create_drift_plots(self, drift_analysis: Dict[str, Any]) -> Dict[str, str]:
        """إنشاء مخططات الانحراف"""
        plots = {}
        
        try:
            # انحراف المميزات
            drift_fig = px.bar(
                x=list(drift_analysis['feature_drifts'].keys()),
                y=list(drift_analysis['feature_drifts'].values()),
                title='انحراف المميزات'
            )
            plots['feature_drift'] = drift_fig.to_html(full_html=False)

            # تغير الأداء
            performance_fig = go.Figure()
            performance_fig.add_trace(go.Scatter(
                x=drift_analysis['timestamps'],
                y=drift_analysis['performance_values'],
                mode='lines+markers',
                name='الأداء'
            ))
            plots['performance_drift'] = performance_fig.to_html(full_html=False)

            return plots

        except Exception as e:
            logger.error(f"خطأ في إنشاء مخططات الانحراف: {str(e)}")
            return {} 