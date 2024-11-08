import schedule
import time
from app.data_processing import process_data
from app.model_selection import select_model
from app.model_training import train_model
from app.model_evaluation import evaluate_model
from app.logging_config import logger
import joblib

def update_model():
    try:
        # Здесь должна быть логика для получения новых данных
        # Для примера, предположим, что у нас есть функция get_new_data()
        new_data = get_new_data()

        data, _ = process_data(new_data)
        model, problem_type = select_model(data)
        trained_model, X_test, y_test, _ = train_model(model, data, problem_type)
        evaluation_results = evaluate_model(trained_model, (X_test, y_test), problem_type)

        joblib.dump(trained_model, 'static/trained_model.joblib')

        logger.info("Model updated successfully")
        logger.info(f"New model performance: {evaluation_results}")
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")

def start_model_updater():
    schedule.every().day.at("02:00").do(update_model)

    while True:
        schedule.run_pending()
        time.sleep(1)