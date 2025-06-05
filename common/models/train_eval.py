import logging
from functools import partial
from multiprocessing import Pool, current_process, Manager, Process, Lock
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.scripts import set_seed
from common.models.utils import load_models, save_model_with_metadata, load_model_with_metadata
from common.analysis import analyse_individual_run, analyse_finals


def train_test_analyse(train_env: ForexEnv,
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
    # Set seeds
    set_seed(42)

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
        callbacks.append(SaveOnEpisodeEndCallback(save_path=models_path))

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

    analyse_evaluation_results(
        models_dir=models_path,
        results_dir=results_path,
        eval_envs_names= ["train", "eval"],
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
    total_timesteps = int(train_env.total_steps * train_episodes)

    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True)

    logging.info("Training complete.")

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
        eval_episode_length = eval_env.total_steps

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
                    num_workers: int = 1) -> None:
    """
    Evaluates each model in a directory on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    logging.info("Starting evaluation...")

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

def analyse_results(results_dir: Path) -> None:
    """
    Recursively searches a directory for data.csv files, and performs analysis.

    Directory structure assumption:
    - Each immediate subdirectory of 'results_dir' corresponds to a different model.
    - Each model directory contains one or more subdirectories, each representing a different environment.
    - Each environment directory contains exactly one 'data.csv' file.
    """
    logging.info("Analysing results...")

    # Validate input
    if not results_dir.exists():
        raise ValueError(f"Directory {results_dir} does not exist.")
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir} is not a directory.")

    # Setup
    eval_envs_metrics = {}
    result_files = list(results_dir.rglob("data.csv"))
    result_files.sort(key=lambda x: x.stat().st_mtime) # Old to new
    for results_file in result_files:

        # Extract names (shady business)
        env_dir = results_file.parent
        env_name = env_dir.name
        if eval_envs_metrics.get(env_name, None) is None:
            eval_envs_metrics[env_name] = []
        model_dir = env_dir.parent
        model_name = model_dir.name
        assert model_dir.parent == results_dir, f"{results_file} is not a great-grandchild of {results_dir}"

        # Load results
        df = pd.read_csv(results_file, low_memory=False)
        if df.empty:
            logging.warning(f"Results file {results_file} is empty, skipping.")
            continue

        # Analyze results
        logging.info(f"Analyzing results of {model_name} on {env_name} from {results_file}...")
        metrics = analyse_individual_run(df, env_dir, name=model_name)
        eval_envs_metrics[env_name].append(metrics) # save results

    # Aggregate environment results
    for eval_env_name, metrics in eval_envs_metrics.items():
        analyse_finals(metrics, results_dir / eval_env_name, name=eval_env_name)

    logging.info("Analysis complete.")

def analyse_evaluation_results(models_dir: Path,
                               results_dir: Path,
                               eval_envs_names: list[str],
                               model_name_suffix: str = "",
                               ) -> None:
    """
    Analyze evaluation results from multiple models and environments.
    This function reads the results from the specified models and environments,
    and generates a summary of the performance metrics.
    """

    logging.info("Analyzing results...")
    logging.warning("This method (analyze_evaluation_results) is a little wonky, please use analyse_results instead.")

    eval_envs_model_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in eval_envs_names}

    for model_name, model in load_models(models_dir):
        model_results_dir = results_dir / model_name

        for eval_env_name in eval_envs_names:
            env_results_dir = model_results_dir / eval_env_name
            env_results_file = env_results_dir / "data.csv"

            if not env_results_file.exists():
                logging.warning(f"Results file {env_results_file} does not exist, skipping.")
                continue

            # Load results
            df = pd.read_csv(env_results_file)
            if df.empty:
                logging.warning(f"Results file {env_results_file} is empty, skipping.")
                continue
            logging.info(f"Analyzing results for model: {model_name} on environment: {eval_env_name}")
            metrics = analyse_individual_run(df, env_results_dir, name=model_name + model_name_suffix + f"[{eval_env_name}]")
            eval_envs_model_metrics[eval_env_name].append(metrics)
    
    for eval_env_name, metrics in eval_envs_model_metrics.items():
        analyse_finals(metrics, results_dir / eval_env_name, name="episodic_results" + model_name_suffix + f"[{eval_env_name}]")

    logging.info("Analysis complete.")


def run_model(model: BaseAlgorithm,
              env: ForexEnv,
              data_path: Path,
              total_steps: int,
              deterministic: bool,
              progress_bar: bool = True
              ) -> None:
    """
    Run a model on a ForexEnv for a number of episodes.
    Results are saved to data_path.
    """
    # Validate input
    if data_path.suffix != ".csv":
        raise ValueError(f"{data_path} is not a CSV file")
    if total_steps <= 0:
        raise ValueError("Total steps must be greater than 0.")

    # Function setup
    data_path.parent.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: env])
    steps = iter(tqdm(range(total_steps)) if progress_bar else range(total_steps))
    logs_df = None

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
    for step in steps: # skip the first step

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
    logs_df.to_csv(data_path, index=False)
