from pathlib import Path
from typing import Generator

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

from tqdm import tqdm

from common.envs.forex_env import ForexEnv
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG

import zipfile
import json

ALGO_DICT = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}

def detect_algo(model_zip: Path) -> str:
    """
    Reads a model .zip to determine its algorithm.
    """
    if not model_zip.exists():
        raise ValueError(f"{model_zip} does not exist")
    if not model_zip.suffix == ".zip":
        raise ValueError(f"{model_zip} is not a zip file")
    with zipfile.ZipFile(model_zip, 'r') as archive:
        with archive.open('data/model_data.json') as f:
            model_data = json.load(f)
            return model_data.get('algo')

def get_model_class(algo: str) -> BaseAlgorithm:
    """
    Returns the class corresponding to an algorithm.
    """
    if algo in ALGO_DICT:
        return ALGO_DICT[algo]
    raise ValueError(f"Unknown algorithm {algo}")

def load_model(model_zip: Path) -> BaseAlgorithm:
    """
    Loads any model .zip
    """
    model_algo = detect_algo(model_zip)
    model_class = get_model_class(model_algo)
    return model_class.load(model_zip)

def load_models(models_dir: Path) -> Generator[tuple[str, BaseAlgorithm], None, None]:
    """
    Generator that yields all the models in a directory.
    """
    if not models_dir.is_dir():
        raise ValueError(f"{models_dir} is not a directory")

    model_zips = list(f for f in models_dir.glob("*.zip") if f.is_file())
    model_zips.sort(key=lambda x: x.stat().st_mtime) # sort on last modified
    logging.info(f"Found {len(model_zips)} model zips in '{models_dir}'.")

    for model_zip in model_zips:

        logging.info(f"Loading model from {model_zip}...")
        model_name = model_zip.stem
        model = load_model(model_zip)
        logging.info(f"Model loaded from {model_zip}.")

        yield model_name, model

def train_model(model: BaseAlgorithm,
                train_env: ForexEnv,
                train_episodes: int = 1,
                callback: list[BaseCallback] | None = None
                ) -> None:
    """
    Trains a model on a ForexEnv for a given number of episodes.
    """

    logging.info(f"Training {model.__class__.__name__} for {train_episodes} episodes...")

    train_dummy_env = DummyVecEnv([lambda: train_env])
    model.set_env(train_dummy_env)
    total_timesteps = train_env.total_steps * train_episodes

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
    Run a trained model on a ForexEnv for a number of episodes.
    """

    if total_steps <= 0:
        raise ValueError("Total steps must be greater than 0.")

    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: env])  # Ensure env is wrapped in DummyVecEnv

    pbar = None
    if progress_bar:
        pbar = tqdm(total=total_steps, desc="Total Steps")

    step_count = 1
    logs_df = None
    episode_logs: List[Dict[str, Any]] = []
    obs = env.reset()
    episode_logs.append({
        "step": 0,
        "action": None,
        "reward": None,
        "done": None,
    })
    while step_count < total_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, rewards, dones, infos = env.step(action)

        log_entry: Dict[str, Any] = {
            "step": step_count,
            "action": action[0].tolist(),
            "reward": rewards[0],
            "done": dones[0],
        }
        episode_logs.append(log_entry)

        if pbar:
            pbar.update(1)

        # If done, break out of the while loop
        if any(dones):
            if pbar:
                pbar.set_description("Episode completed")
            # Reset the environment
            obs = env.reset()

            episode_info = infos[0] if infos else {}
            market_data_df = episode_info.get('market_data', pd.DataFrame())
            market_features_df = episode_info.get('market_features', pd.DataFrame())
            agent_data_df = episode_info.get('agent_data', pd.DataFrame())
            # prepend the dataframes columns with their respective names
            market_data_df.columns = [f"info.market_data.{col}" for col in market_data_df.columns]
            market_features_df.columns = [f"info.market_features.{col}" for col in market_features_df.columns]
            agent_data_df.columns = [f"info.agent_data.{col}" for col in agent_data_df.columns]

            # check lengths of dataframes and episode_logs match
            if len(episode_logs) != len(market_data_df) or len(episode_logs) != len(market_features_df) or len(
                    episode_logs) != len(agent_data_df):
                raise ValueError(
                    "Length of episode logs does not match length of market data, market features, or agent data.")

            temp_df = pd.DataFrame(episode_logs)
            temp_df = pd.concat([temp_df, agent_data_df, market_data_df, market_features_df], axis=1)

            if logs_df is None:
                logs_df = temp_df
            else:
                logs_df = pd.concat([logs_df, temp_df], ignore_index=True)

            episode_logs = []
            episode_logs.append({
                "step": 0,
                "action": None,
                "reward": None,
                "done": None,
            })
        else:
            obs = next_obs

        step_count += 1

    if pbar:
        pbar.close()

    # Save collected logs to JSON file
    # flatten each dictionary in collected_log_entries to be only one level deep.
    # flat_log_entries = [flatten_dict(entry) for entry in collected_log_entries]
    # log_df = pd.DataFrame(flat_log_entries)
    logs_df.to_csv(data_path, index=False)
