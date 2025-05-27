"""
This file contains some simpel scripts that can be useful anywhere during the project.
"""

from datetime import datetime, timedelta
import pandas as pd
from stockstats import StockDataFrame
from common.constants import *
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, List
import json
from tqdm.auto import tqdm

import random
import os
import numpy as np
import tensorflow as tf
import torch

def combine_df(bid_df: pd.DataFrame, ask_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines bid and ask DataFrames into a single `DataFrame`, renaming columns
    by adding a `_bid`, or `_ask` postfix, and calculating the average volume.
    """
    bid_columns = set(bid_df.columns)
    ask_columns = set(ask_df.columns)
    expected_columns = set(DATA_COLUMNS)
    if bid_columns != expected_columns or ask_columns != expected_columns:
        raise ValueError(f"{bid_columns} and {ask_columns} must be equal to {expected_columns}")

    bid_rename = {
        Col.VOL : "volume_bid",
        Col.HIGH : "high_bid",
        Col.LOW : "low_bid",
        Col.OPEN : "open_bid",
        Col.CLOSE : "close_bid"
    }
    ask_rename = {
        Col.VOL : "volume_ask",
        Col.HIGH : "high_ask",
        Col.LOW : "low_ask",
        Col.OPEN : "open_ask",
        Col.CLOSE : "close_ask"
    }
    bid_df.rename(columns=bid_rename, inplace=True)
    ask_df.rename(columns=ask_rename, inplace=True)

    df = pd.merge(bid_df, ask_df, on=Col.TIME, how="inner")
    df[Col.VOL] = (df["volume_bid"] + df["volume_ask"] ) / 2

    return df

def round_datetime(date_time: datetime, interval: int) -> datetime:
    """
    Rounds a datetime object to the nearest multiple of `interval` in seconds.
    """
    start_of_day = date_time.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_start = (date_time - start_of_day).total_seconds()
    rounded_seconds = round(seconds_since_start / interval) * interval
    return start_of_day + timedelta(seconds=rounded_seconds)

def exact_divide(a: int, b: int) -> int:
    """
    Performs an exact division of `a` by `b`, returning an integer.
    Raises a `ValueError`, if `a` is not divisible by `b`.
    """
    if a % b == 0:
        return a // b
    raise ValueError(f"{a} is not divisible by {b}")

def split_df(df: pd.DataFrame, ratio: float):
    """
    Splits a dataframe into two parts based on a given ratio.
    """
    if ratio < 0 or ratio > 1:
        raise ValueError(f"{ratio} is not a valid ratio")
    split_index = int(len(df) * ratio)
    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]
    return df1, df2

def find_first_row_without_nan(df: pd.DataFrame) -> int:
    """
    Removes leading rows with NaN values in any of the columns
    """
    found_nan = True
    index = 0

    while found_nan:
        found_nan = False
        for col in df.columns:
            if pd.isna(df.iloc[index][col]):
                found_nan = True
                break
        if found_nan:
            index += 1

    return index

def run_model_on_vec_env(
        model: Any,
        env: 'VecEnv', # Use quotes for forward reference if VecEnv is not yet defined
        log_path: str | Path,
        n_episodes: int = 1,
        progress_bar: bool = True, # New parameter for progress bar
    ):
    """
    Run a trained RL model on a vectorized environment for a number of episodes,
    log each step, and write all logs at the end. Optionally displays a progress bar.

    Args:
        model: The trained RL model with a .predict() method.
        env: The vectorized environment (e.g., Stable Baselines3 VecEnv).
        log_path: Path to save the JSON log file.
        n_episodes: Number of episodes to run per environment.
        progress_bar: If True, display a progress bar for episode completion (requires tqdm).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    collected_log_entries: List[str] = []
    episode_counts: List[int] = [0] * env.num_envs
    timesteps: List[int] = [0] * env.num_envs
    
    # Ensure obs is initialized before the loop, especially if n_episodes might be 0
    if n_episodes == 0:
        with open(log_path, "w") as f: # Create an empty file if no episodes
            pass
        return

    obs = env.reset()

    pbar = None
    if progress_bar:
        if tqdm is None:
            print("Warning: tqdm is not installed. Progress bar will not be shown. "
                  "Install with: pip install tqdm")
        else:
            total_episodes_to_complete = env.num_envs * n_episodes
            if total_episodes_to_complete > 0 : # Only initialize if there's work to do
                pbar = tqdm(total=total_episodes_to_complete, desc="Running Episodes", unit="ep")

    try:
        while min(episode_counts) < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, rewards, dones, infos = env.step(action)

            for i in range(env.num_envs):
                # Log data only if the current environment hasn't completed its n_episodes
                if episode_counts[i] < n_episodes:
                    log_entry: Dict[str, Any] = {
                        "env_index": i,
                        "episode": episode_counts[i],
                        "timestep": timesteps[i],
                        "action": action[i].tolist() if hasattr(action[i], 'tolist') else action[i],
                        "obs": obs[i].tolist() if hasattr(obs[i], 'tolist') else obs[i],
                        "reward": float(rewards[i]),
                        "done": bool(dones[i]),
                        "info": infos[i] # infos[i] can be an empty dict or contain data
                    }
                    collected_log_entries.append(log_entry)
                    timesteps[i] += 1

                    if dones[i]:
                        episode_counts[i] += 1
                        timesteps[i] = 0
                        if pbar:
                            pbar.update(1)
            
            obs = next_obs
    finally:
        if pbar:
            pbar.close()

    # add .csv to the log_path if it doesn't exist
    if not log_path.suffix:
        log_path = log_path.with_suffix(".csv")
    with open(log_path, "w") as f:
        print(f"Writing {len(collected_log_entries)} log entries to {log_path}")
        df = pd.DataFrame(collected_log_entries)
        df.to_csv(f, index=False)
        print(f"Log entries written to {log_path}")

def set_seed(seed_value: int):
    """
    Sets the random seed for Python, NumPy, TensorFlow, and PyTorch.

    Args:
        seed_value (int): The integer value to use as the seed.
    """
    # 1. Set Python's built-in random module seed
    random.seed(seed_value)

    # 2. Set NumPy's seed
    np.random.seed(seed_value)

    # 3. Set TensorFlow's seed
    # For TensorFlow 2.x
    tf.random.set_seed(seed_value)

    # 4. Set PyTorch's seed
    torch.manual_seed(seed_value)
    # If you are using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # The following two lines are often recommended for deterministic behavior on CUDA,
        # but they can impact performance. Use them if reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seeds set to {seed_value} for Python, NumPy, TensorFlow, and PyTorch.")