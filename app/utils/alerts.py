# app/utils/alerts.py
from typing import Dict, Any, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.core.logging_config import logger

class AlertSystem:
    def __init__(self):
        self.email_config = settings.EMAIL_CONFIG
        self.alert_thresholds = settings.ALERT_THRESHOLDS

    def check_and_alert(self, 
                       metrics: Dict[str, float],
                       model_id: str) -> None:
        """
        Check metrics against thresholds and send alerts if needed
        """
        alerts = self._check_thresholds(metrics)
        if alerts:
            self._send_alerts(alerts, model_id)

    def _check_thresholds(self, metrics: Dict[str, float]) -> List[str]:
        """
        Check if any metrics exceed their thresholds
        """
        alerts = []
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                threshold = self.alert_thresholds[metric]
                if value < threshold:
                    alerts.append(
                        f"Alert: {metric} is below threshold "
                        f"(current: {value:.4f}, threshold: {threshold})"
                    )
        return alerts

    def _send_alerts(self, alerts: List[str], model_id: str) -> None:
        """
        Send email alerts to configured recipients
        """
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'ML Framework Alert - Model {model_id}'
            msg['From'] = self.email_config['sender']
            msg['To'] = ', '.join(self.email_config['recipients'])

            body = "\n".join(alerts)
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)

            logger.info(f"Alerts sent successfully for model {model_id}")

        except Exception as e:
            logger.error(f"Error sending alerts: {str(e)}")