from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from app.logging_config import logger

def train_model(model, data, problem_type):
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Starting model training...")
    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    # Perform cross-validation
    if problem_type == 'classification':
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {np.mean(cv_scores)}")

    return model, X_test, y_test, cv_scores