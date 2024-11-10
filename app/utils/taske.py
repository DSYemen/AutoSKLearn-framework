# app/utils/tasks.py
from celery import Celery
from app.core.config import settings
from app.ml.model_training import ModelTrainer
from app.ml.monitoring import ModelMonitor
from app.utils.alerts import AlertSystem

celery_app = Celery(
    'ml_framework',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@celery_app.task
def retrain_model(model_id: str, new_data_path: str):
    """Background task for model retraining"""
    try:
        # Load and process new data
        processor = AdvancedDataProcessor()
        processed_data = processor.process_data_from_path(new_data_path)

        # Get current model
        current_model = prediction_service.load_model(model_id)

        # Retrain model
        trainer = ModelTrainer(current_model, processed_data.problem_type)
        new_model, metrics = trainer.train(
            processed_data.processed_df.drop('target', axis=1),
            processed_data.processed_df['target']
        )

        # Update model if performance improved
        if metrics['test_score'] > current_model.metrics['test_score']:
            prediction_service.update_model(model_id, new_model, metrics)

        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return {"status": "error", "message": str(e)}

@celery_app.task
def update_monitoring_metrics(model_id: str):
    """Background task for updating monitoring metrics"""
    try:
        monitor = ModelMonitor(model_id)
        metrics = monitor.update_metrics()

        # Check for alerts
        alert_system = AlertSystem()
        alert_system.check_and_alert(metrics, model_id)

        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Monitoring update error: {str(e)}")
        return {"status": "error", "message": str(e)}