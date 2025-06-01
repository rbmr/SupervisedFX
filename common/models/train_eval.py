import logging
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.scripts import set_seed
from common.models.utils import load_models
from common.analysis import analyse_individual_run, analyse_finals


def train_test_analyse(train_env: ForexEnv,
                       eval_env: ForexEnv,
                       model: BaseAlgorithm,
                       base_folder_path: Path,
                       experiment_group_name: str,
                       experiment_name: str,
                       train_episodes: int = 10,
                       eval_episodes: int = 1,
                       checkpoints: bool = False
                       ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    
    Parameters
    ----------
    
    """
    # Set seeds
    set_seed(42)

    # Set up folders
    experiment_path = base_folder_path / "experiments" / experiment_group_name / experiment_name
    results_path = experiment_path / "results"
    logs_path = experiment_path / "logs"
    models_path = experiment_path / "models"

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # model class
    if not isinstance(model, BaseAlgorithm):
        raise ValueError(f"Model must be an instance of BaseAlgorithm, got {type(model)}")

    # -- TRAINING THE MODEL
    logging.info(f"Training model for {train_episodes} episodes...")
    callbacks = []
    if checkpoints:
        checkpoint_callback = SaveOnEpisodeEndCallback(save_path=models_path)
        callbacks.append(checkpoint_callback)

    train_model(model, train_env=train_env, train_episodes=train_episodes, callback=callbacks)

    # save the final model if we are not saving checkpoints
    if not checkpoints:
        save_path = models_path / f"model_{train_episodes}_episodes.zip"
        model.save(save_path)
        logging.info(f"Model(s) saved to '{models_path}'.")
    
    logging.info("Training complete.")
    # -- END TRAINING

    # -- EVALUATING THE MODEL(S)
    logging.info("Starting evaluation...")

    evaluate_models(models_dir=models_path,
                    results_dir=results_path,
                    eval_envs={
                        "train": train_env,
                        "eval": eval_env
                        },
                    eval_episodes=eval_episodes)    
    logging.info("Evaluation complete.")
    # -- END EVALUATION

    # -- ANALYZING RESULTS
    logging.info("Analyzing results...")
    analyse_evaluation_results(
        models_dir=models_path,
        results_dir=results_path,
        eval_envs_names= ["train", "eval"],
        model_name_suffix=f"[{experiment_group_name}::{experiment_name}]"
    )
    logging.info("Analysis complete.")
    # -- END ANALYSIS

    
    

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

def evaluate_models(models_dir: Path,
                    results_dir: Path,
                    eval_envs: dict[str, ForexEnv],
                    eval_episodes: int = 1,
                    ) -> None:
    """
    Evaluates each model in a directory on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    logging.info("Starting evaluation...")

    for model_name, model in load_models(models_dir):

        model_results_dir = results_dir / model_name

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
                      progress_bar=True
                      )

    logging.info("Finished evaluation.")

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
            metrics = analyse_individual_run(df, env_results_dir, name=model_name + model_name_suffix)
            eval_envs_model_metrics[eval_env_name].append(metrics)
    
    for eval_env_name, metrics in eval_envs_model_metrics.items():
        analyse_finals(metrics, results_dir / eval_env_name, name="eval_results_" + eval_env_name)

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
