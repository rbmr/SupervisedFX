import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple

from stable_baselines3.common.base_class import BaseAlgorithm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.models.utils import (load_model_with_metadata, save_model_with_metadata)
from common.scripts import set_seed
from common.data.data import ForexCandleData
from common.data.feature_engineer import FeatureEngineer
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer

from common.models.train_eval import train_model, evaluate_models, analyse_results, evaluate_model, analyze_result, combine_finals


from dataclasses import dataclass

@dataclass
class ExperimentBlueprint:
    name: str
    data_func: Callable[[], ForexCandleData]
    feature_engineers_func: Callable[[], Tuple[FeatureEngineer, StepwiseFeatureEngineer]]
    envs_func: Callable[[List[float], ForexCandleData, FeatureEngineer, StepwiseFeatureEngineer], List[ForexEnv]]  # Returns train, validate, eval envs
    model_func_with_seed: Callable[[ForexEnv, int], BaseAlgorithm]  # Takes an environment, returns a model
    train_episodes: int


def run_experiments(group_name: str, 
                    base_folder_path: Path,
                    seeds: List[int],
                    experiment_blueprints: List[ExperimentBlueprint],
                    num_workers: int = 1) -> None:
    for seed in seeds:
        for blueprint in experiment_blueprints:
            logging.info(f"Running experiment: {blueprint.name} with seed {seed}")
            run_experiment(group_name=group_name,
                           base_folder_path=base_folder_path,
                           seed=seed,
                           experiment_blueprint = blueprint,
                           num_workers=num_workers)
            logging.info(f"Completed experiment: {blueprint.name} with seed {seed}")
    
    logging.info("Combining final results...")
    combine_finals(
        experiment_group=base_folder_path / "experiments" / group_name
    )
    logging.info("Final results combined successfully.")

def run_experiment(group_name: str,
                    base_folder_path: Path,
                    seed: int,
                    experiment_blueprint: ExperimentBlueprint,
                    num_workers: int = 1
                    ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    """

    set_seed(seed)


    # -------------------------- #
    # ---- EXPERIMENT SETUP ---- #
    # -------------------------- #
    # Validate input
    if not base_folder_path.is_dir():
        raise ValueError(f"{base_folder_path} is not a directory")
    if not base_folder_path.exists():
        raise ValueError(f"{base_folder_path} does not exist")
    if not isinstance(experiment_blueprint, ExperimentBlueprint):
        raise ValueError(f"experiment_blueprint must be an instance of ExperimentBlueprint, got {type(experiment_blueprint)}")
    if not isinstance(group_name, str) or not group_name:
        raise ValueError(f"group_name must be a non-empty string, got {group_name}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError(f"num_workers must be a positive integer, got {num_workers}")
    if not experiment_blueprint.train_episodes > 0:
        raise ValueError(f"train_episodes must be greater than 0, got {experiment_blueprint.train_episodes}")
    if not isinstance(experiment_blueprint.model_func_with_seed, Callable):
        raise ValueError(f"model_func must be a callable, got {type(experiment_blueprint.model_func_with_seed)}")
    if not isinstance(experiment_blueprint.data_func, Callable):
        raise ValueError(f"data_func must be a callable, got {type(experiment_blueprint.data_func)}")
    if not isinstance(experiment_blueprint.envs_func, Callable):
        raise ValueError(f"envs_func must be a callable, got {type(experiment_blueprint.envs_func)}")
    if not isinstance(experiment_blueprint.feature_engineers_func, Callable):
        raise ValueError(f"feature_engineers_func must be a callable, got {type(experiment_blueprint.feature_engineers_func)}")
    
    # Extract blueprint details
    experiment_name = experiment_blueprint.name

    forex_data = experiment_blueprint.data_func()
    if not isinstance(forex_data, ForexCandleData):
        raise ValueError(f"Data function must return an instance of ForexCandleData, got {type(forex_data)}")
    
    feature_engineer, stepwise_feature_engineer = experiment_blueprint.feature_engineers_func()
    if not isinstance(feature_engineer, FeatureEngineer) or not isinstance(stepwise_feature_engineer, StepwiseFeatureEngineer):
        raise ValueError(f"Feature engineers must be instances of FeatureEngineer and StepwiseFeatureEngineer, got {type(feature_engineer)} and {type(stepwise_feature_engineer)}")
    
    # Create environments
    envs = experiment_blueprint.envs_func(
        forex_data=forex_data,
        feature_engineer=feature_engineer,
        stepwise_feature_engineer=stepwise_feature_engineer
    )
    if len(envs) != 3:
        raise ValueError(f"envs_func must return a list of 3 environments, got {len(envs)}")
    train_env, validate_env, eval_env = envs

    if not isinstance(train_env, ForexEnv) or not isinstance(validate_env, ForexEnv) or (eval_env is not None and not isinstance(eval_env, ForexEnv)):
        raise ValueError(f"All environments must be instances of ForexEnv, got {type(train_env)}, {type(validate_env)}, {type(eval_env)}")
    
    # Create model
    model = experiment_blueprint.model_func_with_seed(train_env, seed)
    if not isinstance(model, BaseAlgorithm):
        raise ValueError(f"Model function must return an instance of BaseAlgorithm, got {type(model)}")
    
    # ----------------------- #
    # ---- FOLDER  SETUP ---- #
    # ----------------------- #
    # Set up folders
    experiment_path = base_folder_path / "experiments" / group_name / experiment_name / f"seed_{seed}"
    results_path = experiment_path / "results"
    models_path = experiment_path / "models"

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # set tensorboard logging if enabled
    model.tensorboard_log = str(experiment_path / "tensorboard_logs")

    # --------------------- #
    # ---- TRAIN MODEL ---- #
    # --------------------- #

    starting_episode = 0
    train_episodes = experiment_blueprint.train_episodes
    # in the models dir, find the latest model and load it if it exists (latest model is the newest by modification time)
    latest_model = max(models_path.glob("*.zip"), key=lambda p: p.stat().st_mtime, default=None)
    if latest_model and latest_model.is_file():
        logging.info(f"Loading latest model from {latest_model}")
        model = load_model_with_metadata(latest_model)
        starting_episode = int(latest_model.stem.split('_')[1])
        train_episodes -= starting_episode
        train_episodes = max(train_episodes, 0)  # Ensure non-negative episodes


    callbacks = []
    save_on_episode_callback = SaveOnEpisodeEndCallback(models_dir=models_path, episode_num=starting_episode)
    callbacks.append(save_on_episode_callback)

    train_model(model, train_env=train_env, train_episodes=train_episodes, callback=callbacks)

    save_path = models_path / f"model_{train_episodes}_episodes.zip"
    save_model_with_metadata(model, save_path)

    # ------------------------- #
    # ---- VALDIATE MODELS ---- #
    # ------------------------- #

    evaluate_models(models_dir=models_path,
                    results_dir=results_path,
                    eval_envs={"train": train_env,
                               "validate": validate_env},
                    eval_episodes=1,
                    num_workers=num_workers)

    # ------------------------ #
    # ---- ANALYZE MODELS ---- #
    # ------------------------ #

    analyse_results(
        results_dir=results_path,
        model_name_suffix=f"[{group_name}::{experiment_name}]",
        num_workers=num_workers
    )

    # --------------------------------- #
    # ---- EVALUATE SELECTED MODEL ---- #
    # --------------------------------- #

    if eval_env:
        logging.info("Selecting best model based on custom score (Val_Perf - |Val_Perf - Train_Perf|)...")
        
        # 1. Collect performance data for all models from their info.json files
        model_performance = {}
        selection_metric = 'sharpe_ratio' # The metric used for performance (e.g., sharpe_ratio)

        for info_path in results_path.rglob('info.json'):
            env_name = info_path.parent.name
            if env_name in ['train', 'validate']:
                model_name = info_path.parent.parent.name
                model_performance.setdefault(model_name, {})
                try:
                    with open(info_path, 'r') as f:
                        data = json.load(f)
                    metric_value = data.get(selection_metric)
                    if metric_value is not None:
                        model_performance[model_name][env_name] = metric_value
                except (IOError, json.JSONDecodeError) as e:
                    logging.warning(f"Could not read or parse {info_path}: {e}")

        # 2. Calculate the custom score for each model and find the best one
        scored_models = []
        for model_name, perfs in model_performance.items():
            if 'train' in perfs and 'validate' in perfs:
                train_perf = perfs['train']
                val_perf = perfs['validate']
                score = val_perf - abs(val_perf - train_perf)
                
                model_zip_path = models_path / f"{model_name}.zip"
                if model_zip_path.is_file():
                    scored_models.append((score, model_zip_path, val_perf, train_perf))
                else:
                    logging.warning(f"Could not find model zip for scored model {model_name}, skipping.")

        # 3. If a best model is found, evaluate it on the final evaluation environment
        if not scored_models:
            logging.error("No models with both train and validate results found. Cannot perform final evaluation.")
        else:
            scored_models.sort(key=lambda x: x[0], reverse=True)
            best_score, best_model_path, best_val_perf, best_train_perf = scored_models[0]
            
            logging.info(f"Best model selected: {best_model_path.name}")
            logging.info(f" -> Score: {best_score:.4f} (Validation Sharpe: {best_val_perf:.4f}, Training Sharpe: {best_train_perf:.4f})")
            logging.info("Evaluating best model on the final evaluation environment...")

            # Evaluate the best model on the eval_env
            evaluate_model(
                model_zip=best_model_path,
                results_dir=results_path,
                eval_envs={"evaluation": eval_env}, # Saves results in a dedicated 'evaluation' folder
                eval_episodes=1,
                progress_bar=True
            )

            # 4. Analyze just the new evaluation result to create its specific info.json and plots
            logging.info("Analyzing final evaluation result...")
            eval_data_path = results_path / best_model_path.stem / "evaluation" / "data.csv"
            if eval_data_path.is_file():
                analyze_result(
                    data_csv=eval_data_path,
                    model_name_suffix=f"[{group_name}::{experiment_name}]"
                )
            else:
                logging.error(f"Final evaluation did not produce a data file at {eval_data_path}")