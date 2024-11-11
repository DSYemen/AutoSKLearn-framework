from typing import Dict, Any, Optional, List
import optuna
from optuna.trial import Trial
import numpy as np
from datetime import datetime
from app.core.logging_config import logger
from app.core.config import settings
from app.schemas.model import ModelOptimizationConfig
from app.utils.cache import cache_manager
from app.ml.model_training import ModelTrainer
from app.ml.model_evaluation import ModelEvaluator

async def optimize_model_task(
    model_id: str,
    config: ModelOptimizationConfig,
    job_id: str
) -> Dict[str, Any]:
    """
    مهمة تحسين معلمات النموذج
    """
    try:
        # الحصول على بيانات النموذج
        model_data = await cache_manager.get_model_metadata(model_id)
        if not model_data:
            raise ValueError("Model not found")

        # إنشاء دالة الهدف
        def objective(trial: Trial) -> float:
            # تحديد نطاقات المعلمات
            params = {}
            for param_name, param_config in config.optimization_parameters.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # تدريب النموذج بالمعلمات المقترحة
            trainer = ModelTrainer(model_data['type'])
            model = trainer.train(
                model_data['X_train'],
                model_data['y_train'],
                params
            )

            # تقييم النموذج
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(
                model,
                model_data['X_val'],
                model_data['y_val']
            )

            return metrics[config.optimization_metric]

        # إنشاء دراسة التحسين
        study = optuna.create_study(
            direction="maximize",
            study_name=f"optimize_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # تنفيذ التحسين
        study.optimize(
            objective,
            n_trials=config.max_trials,
            timeout=config.timeout,
            callbacks=[
                lambda study, trial: update_optimization_progress(
                    job_id,
                    study.trials,
                    config.max_trials
                )
            ]
        )

        # الحصول على أفضل المعلمات
        best_params = study.best_params
        best_value = study.best_value
        optimization_history = [
            {
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]

        # تحديث النموذج بأفضل المعلمات
        trainer = ModelTrainer(model_data['type'])
        updated_model = trainer.train(
            model_data['X_train'],
            model_data['y_train'],
            best_params
        )

        # تقييم النموذج المحسن
        evaluator = ModelEvaluator()
        final_metrics = evaluator.evaluate(
            updated_model,
            model_data['X_test'],
            model_data['y_test']
        )

        # تحديث البيانات الوصفية للنموذج
        model_data['parameters'] = best_params
        model_data['metrics'] = final_metrics
        model_data['optimization_history'] = optimization_history
        await cache_manager.cache_model_metadata(model_id, model_data)

        return {
            'status': 'success',
            'best_params': best_params,
            'best_value': best_value,
            'optimization_history': optimization_history,
            'final_metrics': final_metrics
        }

    except Exception as e:
        logger.error(f"Error in model optimization: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def update_optimization_progress(job_id: str, trials: List[Trial], max_trials: int):
    """تحديث تقدم التحسين"""
    try:
        progress = min(len(trials) / max_trials * 100, 100)
        current_best = max([t.value for t in trials]) if trials else 0

        from app.main import update_processing_status
        update_processing_status(
            job_id=job_id,
            status='optimizing',
            progress=progress,
            message=f'Current best: {current_best:.4f}'
        )
    except Exception as e:
        logger.error(f"Error updating optimization progress: {str(e)}")

class ModelOptimizer:
    """فئة تحسين النموذج"""

    def __init__(self):
        self.evaluator = ModelEvaluator()

    async def optimize_model(
        self,
        model_id: str,
        config: ModelOptimizationConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        تحسين معلمات النموذج
        """
        try:
            # إنشاء دالة الهدف
            def objective(trial: Trial) -> float:
                params = self._suggest_parameters(trial, config.optimization_parameters)
                trainer = ModelTrainer(config.model_type)
                model = trainer.train(X_train, y_train, params)
                metrics = self.evaluator.evaluate(model, X_val, y_val)
                return metrics[config.optimization_metric]

            # إنشاء دراسة التحسين
            study = optuna.create_study(
                direction="maximize",
                study_name=f"optimize_{model_id}"
            )

            # تنفيذ التحسين
            study.optimize(
                objective,
                n_trials=config.max_trials,
                timeout=config.timeout
            )

            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'optimization_history': self._get_optimization_history(study)
            }

        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            raise

    def _suggest_parameters(
        self,
        trial: Trial,
        parameter_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """اقتراح معلمات للتجربة"""
        params = {}
        for param_name, config in parameter_config.items():
            if config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    config['low'],
                    config['high']
                )
            elif config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    config['low'],
                    config['high'],
                    log=config.get('log', False)
                )
            elif config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config['choices']
                )
        return params

    def _get_optimization_history(self, study: optuna.Study) -> List[Dict[str, Any]]:
        """الحصول على سجل التحسين"""
        return [
            {
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ] 