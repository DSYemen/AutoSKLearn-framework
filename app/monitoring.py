import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from app.logging_config import logger

def monitor_model_performance(model, X, y, problem_type):
    predictions = model.predict(X)

    if problem_type == 'classification':
        score = accuracy_score(y, predictions)
        metric = 'Accuracy'
    else:
        score = mean_squared_error(y, predictions)
        metric = 'Mean Squared Error'

    logger.info(f"Model Performance - {metric}: {score}")

    return score