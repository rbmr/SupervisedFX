import json
import logging
from functools import partial
from multiprocessing import Lock, Manager, Process
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.models.analysis import analyse_finals, analyse_individual_run
from common.models.dummy_models import DUMMY_MODELS
from common.models.utils import (load_model_with_metadata, save_model_with_metadata)
from common.scripts import parallel_run

from common.models.dummy_models import DummyModel


def run_experiment(train_env: ForexEnv,
                       eval_env: ForexEnv,
                       model: BaseAlgorithm,
                       base_folder_path: Path,
                       experiment_group_name: str,
                       experiment_name: str,
                       train_episodes: int = 10,
                       eval_episodes: int = 1,
                       checkpoints: bool = False,
                       tensorboard_logging: bool = False
                       ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    """
    # Set up folders
    experiment_path = base_folder_path / "experiments" / experiment_group_name / experiment_name
    results_path = experiment_path / "results"
    models_path = experiment_path / "models"

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # model class
    if not isinstance(model, BaseAlgorithm):
        raise ValueError(f"Model must be an instance of BaseAlgorithm, got {type(model)}")

    # TRAINING THE MODEL

    # set tensorboard logging if enabled
    if tensorboard_logging:
        model.tensorboard_log = str(experiment_path / "tensorboard_logs")
        

    callbacks = []
    if checkpoints:
        callbacks.append(SaveOnEpisodeEndCallback(models_dir=models_path))

    train_model(model, train_env=train_env, train_episodes=train_episodes, callback=callbacks)

    if not checkpoints:
        save_path = models_path / f"model_{train_episodes}_episodes.zip"
        save_model_with_metadata(model, save_path)

    # EVALUATING THE MODEL

    evaluate_models(models_dir=models_path,
                    results_dir=results_path,
                    eval_envs={"train": train_env,
                               "eval": eval_env},
                    eval_episodes=eval_episodes)

    # ANALYZING RESULTS

    analyse_results(
        results_dir=results_path,
        model_name_suffix=f"[{experiment_group_name}::{experiment_name}]"
    )

def train_model(model: BaseAlgorithm,
                train_env: ForexEnv,
                train_episodes: int | float = 1,
                callback: list[BaseCallback] | None = None
                ) -> None:
    """
    Trains a model on a ForexEnv for a given number of episodes.
    """

    logging.info(f"Training {model.__class__.__name__} model for {train_episodes} episodes...")

    train_dummy_env = DummyVecEnv([lambda: train_env])
    model.set_env(train_dummy_env)
    total_timesteps = int(train_env.episode_len * train_episodes)

    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True)
    else:
        logging.warning("Total timesteps is 0, skipping training.")

    logging.info("Training complete.")

def evaluate_dummy(dummy_model: DummyModel, name: str, results_dir: Path, eval_env: ForexEnv, eval_env_name: str) -> None:
    """
    Evaluates a dummy model on a ForexEnv and saves the results.
    """
    model_results_dir = results_dir / name
    env_results_dir = model_results_dir / eval_env_name
    env_results_file = env_results_dir / "data.csv"
    eval_episode_length = eval_env.episode_len

    logging.info(f"Running dummy model ({name}) on environment ({eval_env_name}) for 1 episode...")

    run_model(model=dummy_model,
              env=eval_env,
              data_path=env_results_file,
              total_steps=eval_episode_length,
              deterministic=True,
              progress_bar=True)

def evaluate_dummies(results_dir: Path, eval_envs: dict[str, ForexEnv]):

    for model_fn in DUMMY_MODELS:
        for eval_env_name, eval_env in eval_envs.items():

            model = model_fn(eval_env)
            model_name = model_fn.__name__

            logging.info(f"Running model ({model_name}) on environment ({eval_env_name}) for 1 episode...")

            evaluate_dummy(dummy_model=model,
                           name=model_name,
                           results_dir=results_dir,
                           eval_env=eval_env,
                           eval_env_name=eval_env_name)

def evaluate_model(model_zip: Path,
                   results_dir: Path,
                   eval_envs: dict[str, ForexEnv],
                   eval_episodes: int = 1,
                   force_eval: bool = False,
                   progress_bar: bool = False) -> None:
    """
    Evaluates a model on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    if not model_zip.is_file():
        raise ValueError("model_zip doesnt exist.")
    if not model_zip.suffix == ".zip":
        raise ValueError("model_zip is not a zip file.")

    model_name = model_zip.stem
    model = load_model_with_metadata(model_zip)
    model_results_dir = results_dir / model_name
    if not force_eval and model_results_dir.exists():
        logging.info(f"{model_name} has already been evaluated, skipping...")
        return

    for eval_env_name, eval_env in eval_envs.items():
        env_results_dir = model_results_dir / eval_env_name
        env_results_file = env_results_dir / "data.csv"
        eval_episode_length = eval_env.episode_len

        logging.info(f"Running model ({model_name}) on environment ({eval_env_name}) for {eval_episodes} episodes...")

        run_model(model=model,
                  env=eval_env,
                  data_path=env_results_file,
                  total_steps=eval_episodes * eval_episode_length,
                  deterministic=True,
                  progress_bar=progress_bar)

class ModelQueue:
    def __init__(self, models_dir: Path, seen, lock: Lock):
        if not models_dir.is_dir():
            raise ValueError(f"{models_dir} is not a directory")
        self.models_dir = models_dir
        self.seen = seen # Shared list
        self.lock = lock # Shared lock

    def get(self) -> Path | None:
        with self.lock:
            models = sorted(self.models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            for model_zip in models:
                if not model_zip.is_file():
                    continue
                if model_zip in self.seen:
                    continue
                self.seen.append(model_zip)
                return model_zip
        return None

def evaluate_worker(queue: ModelQueue, func: Callable):
    while True:
        model_path = queue.get()
        if model_path is None:
            break
        func(model_path)

def evaluate_models(models_dir: Path,
                    results_dir: Path,
                    eval_envs: dict[str, ForexEnv],
                    eval_episodes: int = 1,
                    force_eval: bool = False,
                    num_workers: int = 4,
                    eval_dummies: bool = False) -> None:
    """
    Evaluates each model in a directory on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    logging.info("Starting evaluation...")

    if eval_dummies:
        evaluate_dummies(results_dir, eval_envs)

    progress_bar = num_workers == 1
    func = partial(
        evaluate_model,
        results_dir=results_dir,
        eval_envs=eval_envs,
        eval_episodes=eval_episodes,
        force_eval=force_eval,
        progress_bar=progress_bar,
    )
    manager = Manager()
    shared_lock = manager.Lock()
    shared_seen = manager.list()
    queue = ModelQueue(models_dir, shared_seen, shared_lock)

    workers = []
    for i in range(num_workers):
        p = Process(target=evaluate_worker, args=(queue, func), name=f"Worker-{i+1}" )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    logging.info("Finished evaluation.")

def analyze_result(data_csv: Path, model_name_suffix: str = ""):
    """
    Extracts the environment name from a data.csv file.
    """
    analyse_individual_run(data_csv, f"{data_csv.parent.parent.name}{model_name_suffix}")

def analyse_results(results_dir: Path, model_name_suffix: str = "") -> None:
    """
    Searches a directory for data.csv files, and performs analysis.
    Expected directory structure: /{model_name}/{env_name}/data.csv
    """
    logging.info("Analysing results...")

    # Validate input
    if not results_dir.exists():
        raise ValueError(f"Directory {results_dir} does not exist.")
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir} is not a directory.")

    # Evaluate runs
    func = partial(analyze_result, model_name_suffix=model_name_suffix)
    result_files = list(results_dir.rglob("data.csv"))
    result_files.sort(key=lambda x: x.stat().st_mtime) # Old to new
    parallel_run(func, result_files, num_workers=6)

    # Aggregate environment results
    logging.info("Aggregating analysis results...")
    metrics = {}
    for result_file in result_files:
        env_dir = result_file.parent
        env_name = env_dir.name
        if env_name not in metrics:
            metrics[env_name] = []
        info_file = env_dir / "info.json"
        with open(info_file, "r") as f:
            info = json.load(f)
        metrics[env_name].append(info)

    logging.info("Analyzing aggregates...")
    for env_name, metrics in metrics.items():
        analyse_finals(metrics, output_dir=results_dir / f"final_{env_name}", env_name=env_name)

    logging.info("Analysis complete.")

def run_model(model: BaseAlgorithm,
              env: ForexEnv,
              data_path: Path | None,
              total_steps: int,
              deterministic: bool,
              progress_bar: bool = True
              ) -> pd.DataFrame:
    """
    Run a model on a ForexEnv for a number of episodes.
    Results are saved to data_path.
    """
    # Validate input
    if data_path is not None:
        if data_path.suffix != ".csv":
            raise ValueError(f"{data_path} is not a CSV file")
        if total_steps <= 0:
            raise ValueError("Total steps must be greater than 0.")
        data_path.parent.mkdir(parents=True, exist_ok=True)

    # Function setup
    env = DummyVecEnv([lambda: env])
    steps = iter(tqdm(range(total_steps)) if progress_bar else range(total_steps))
    logs_df: None | pd.DataFrame = None

    # Start first episode
    obs = env.reset()
    episode_log: list[dict[str, Any]] = [{
        "step": 0,
        "action": None,
        "reward": None,
        "done": None,
    }]
    next(steps) # skip one value

    # Start the loop
    for step in steps:

        # Take action
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        episode_log.append({
            "step": step,
            "action": action[0].tolist(),
            "reward": rewards[0],
            "done": dones[0],
        })

        # Check for end of episode
        if any(dones):

            # Save episode info
            info = infos[0] if infos else {}
            market_data_df: pd.DataFrame = info['market_data']
            market_features_df = info['market_features']
            agent_data_df = info['agent_data']

            market_data_df.columns = [f"info.market_data.{col}" for col in market_data_df.columns]
            market_features_df.columns = [f"info.market_features.{col}" for col in market_features_df.columns]
            agent_data_df.columns = [f"info.agent_data.{col}" for col in agent_data_df.columns]

            assert len(episode_log) == len(market_data_df)
            assert len(episode_log) == len(market_features_df)
            assert len(episode_log) == len(agent_data_df)

            temp_df = pd.DataFrame(episode_log)
            temp_df = pd.concat([temp_df, agent_data_df, market_data_df, market_features_df], axis=1)
            logs_df = temp_df if logs_df is None else pd.concat([logs_df, temp_df], ignore_index=True, axis=0)

            # Start new episode
            obs = env.reset()
            episode_log = [{
                "step": 0,
                "action": None,
                "reward": None,
                "done": None,
            }]

    # Save collected logs to JSON file
    if data_path is not None:
        logs_df.to_csv(data_path, index=False)
    return logs_df
