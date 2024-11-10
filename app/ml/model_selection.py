# app/ml/model_selection.py
from typing import Tuple, Any
import optuna
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    XGBClassifier, XGBRegressor,
    LightGBMClassifier, LightGBMRegressor
)
from sklearn.metrics import make_scorer
import yaml

class ModelSelector:
    def __init__(self):
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)

        self.classifiers = {
            'RandomForest': (RandomForestClassifier, self._get_rf_params),
            'GradientBoosting': (GradientBoostingClassifier, self._get_gb_params),
            'XGBoost': (XGBClassifier, self._get_xgb_params),
            'LightGBM': (LightGBMClassifier, self._get_lgb_params)
        }

        self.regressors = {
            'RandomForest': (RandomForestRegressor, self._get_rf_params),
            'GradientBoosting': (GradientBoostingRegressor, self._get_gb_params),
            'XGBoost': (XGBRegressor, self._get_xgb_params),
            'LightGBM': (LightGBMRegressor, self._get_lgb_params)
        }

    def select_model(self, X, y) -> Tuple[Any, str]:
        problem_type = self._determine_problem_type(y)
        models = self.classifiers if problem_type == 'classification' else self.regressors

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, X, y, models, problem_type),
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization']['timeout']
        )

        return self._create_best_model(study.best_params, models), problem_type

    def _objective(self, trial, X, y, models, problem_type):
        model_name = trial.suggest_categorical('model', list(models.keys()))
        model_class, param_func = models[model_name]
        params = param_func(trial)
        model = model_class(**params)

        scorer = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
        scores = cross_val_score(model, X, y, 
                               cv=self.config['model']['cv_folds'],
                               scoring=scorer)
        return np.mean(scores)

    @staticmethod
    def _get_rf_params(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

    # Add similar methods for other algorithms...