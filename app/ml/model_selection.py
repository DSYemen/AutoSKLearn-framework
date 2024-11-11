# app/ml/model_selection.py
from typing import Tuple, Any, Dict, List
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from app.core.config import settings
from app.core.logging_config import logger

class ModelSelector:
    """محدد النموذج المتقدم مع تحسين المعلمات"""
    
    def __init__(self):
        self.classifiers = {
            'RandomForest': (RandomForestClassifier, self._get_rf_params),
            'GradientBoosting': (GradientBoostingClassifier, self._get_gb_params),
            'XGBoost': (XGBClassifier, self._get_xgb_params),
            'LightGBM': (LGBMClassifier, self._get_lgb_params),
            'CatBoost': (CatBoostClassifier, self._get_catboost_params),
            'ExtraTrees': (ExtraTreesClassifier, self._get_et_params)
        }
        
        self.regressors = {
            'RandomForest': (RandomForestRegressor, self._get_rf_params),
            'GradientBoosting': (GradientBoostingRegressor, self._get_gb_params),
            'XGBoost': (XGBRegressor, self._get_xgb_params),
            'LightGBM': (LGBMRegressor, self._get_lgb_params),
            'CatBoost': (CatBoostRegressor, self._get_catboost_params),
            'ExtraTrees': (ExtraTreesRegressor, self._get_et_params)
        }
        
        self.best_models: Dict[str, List[Dict[str, Any]]] = {
            'classification': [],
            'regression': []
        }

    def select_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, str]:
        """اختيار أفضل نموذج للبيانات"""
        try:
            problem_type = self._determine_problem_type(y)
            models = self.classifiers if problem_type == 'classification' else self.regressors
            
            logger.info(f"بدء اختيار النموذج لمشكلة {problem_type}")
            
            # إنشاء دراسة Optuna
            study = optuna.create_study(
                direction='maximize',
                study_name=f'{problem_type}_model_selection'
            )
            
            # تحسين النماذج
            study.optimize(
                lambda trial: self._objective(trial, X, y, models, problem_type),
                n_trials=settings.OPTIMIZATION_TRIALS,
                timeout=settings.OPTIMIZATION_TIMEOUT
            )
            
            # إنشاء النموذج الأفضل
            best_model = self._create_best_model(study.best_params, models)
            
            # حفظ معلومات النموذج الأفضل
            self.best_models[problem_type].append({
                'model_type': study.best_params['model'],
                'parameters': study.best_params,
                'score': study.best_value
            })
            
            logger.info(f"تم اختيار النموذج الأفضل: {study.best_params['model']}")
            return best_model, problem_type
            
        except Exception as e:
            logger.error(f"خطأ في اختيار النموذج: {str(e)}")
            raise

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                  models: Dict[str, Tuple], problem_type: str) -> float:
        """دالة الهدف لتحسين النموذج"""
        # اختيار نوع النموذج
        model_name = trial.suggest_categorical('model', list(models.keys()))
        model_class, param_func = models[model_name]
        
        # الحصول على المعلمات
        params = param_func(trial)
        
        try:
            # إنشاء النموذج
            model = model_class(**params)
            
            # تقييم النموذج
            scorer = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
            scores = cross_val_score(
                model, X, y,
                cv=settings.CV_FOLDS,
                scoring=scorer,
                n_jobs=-1
            )
            
            # حساب متوسط النتيجة
            mean_score = np.mean(scores)
            
            # تسجيل المعلومات
            trial.set_user_attr('model_type', model_name)
            trial.set_user_attr('cv_scores', scores.tolist())
            
            return mean_score
            
        except Exception as e:
            logger.error(f"خطأ في تقييم النموذج {model_name}: {str(e)}")
            return float('-inf')

    @staticmethod
    def _determine_problem_type(y: pd.Series) -> str:
        """تحديد نوع المشكلة"""
        if y.dtype == 'object' or len(np.unique(y)) < 10:
            return 'classification'
        return 'regression'

    def _create_best_model(self, best_params: Dict[str, Any], models: Dict[str, Tuple]) -> Any:
        """إنشاء النموذج الأفضل"""
        model_name = best_params.pop('model')
        model_class, _ = models[model_name]
        return model_class(**best_params)

    # دوال الحصول على معلمات النماذج
    @staticmethod
    def _get_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }

    @staticmethod
    def _get_gb_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات Gradient Boosting"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
        }

    @staticmethod
    def _get_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات XGBoost"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0)
        }

    @staticmethod
    def _get_lgb_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات LightGBM"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        }

    @staticmethod
    def _get_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات CatBoost"""
        return {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 100.0),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
            'random_strength': trial.suggest_uniform('random_strength', 1e-8, 10.0)
        }

    @staticmethod
    def _get_et_params(trial: optuna.Trial) -> Dict[str, Any]:
        """معلمات Extra Trees"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }

    def get_best_models_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """الحصول على ملخص لأفضل النماذج"""
        return self.best_models