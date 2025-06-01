import logging
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from common.envs.forex_env import ForexEnv
from common.models.utils import load_models


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
