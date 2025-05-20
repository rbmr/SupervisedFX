import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stockstats import StockDataFrame
import gymnasium as gym
from typing import Any
from pathlib import Path
import json

RSI_PERIOD = 14 # Technical Indicators Periods (stockstats uses these by appending to indicator name e.g. rsi_14)

def prepare_data(df: pd.DataFrame, lookback_window_size: int, features: list[str]):
    """
    Prepares the data by calculating necessary features and normalizing them.
    Uses stockstats for RSI and MACD.
    """

    # Calculate mid prices for indicators
    df['mid_close'] = (df['close_bid'] + df['close_ask']) / 2
    df['mid_open'] = (df['open_bid'] + df['open_ask']) / 2
    df['mid_high'] = (df['high_bid'] + df['high_ask']) / 2
    df['mid_low'] = (df['low_bid'] + df['low_ask']) / 2

    # 1. Log returns
    df['log_ret_bid'] = np.log(df['close_bid'] / df['close_bid'].shift(1)).fillna(0)
    df['log_ret_ask'] = np.log(df['close_ask'] / df['close_ask'].shift(1)).fillna(0)

    # 2. Normalized Spread
    df['spread'] = df['close_ask'] - df['close_bid']
    df['norm_spread'] = df['spread'] / df['mid_close']

    # 3. Normalized Volume
    # Ensure volume exists and handle potential division by zero or all-zero window
    if 'volume' in df.columns:
        vol_min = df['volume'].rolling(window=lookback_window_size, min_periods=1).min()
        vol_max = df['volume'].rolling(window=lookback_window_size, min_periods=1).max()
        denominator = vol_max - vol_min
        df['norm_volume'] = np.where(denominator == 0, 0.5, (df['volume'] - vol_min) / denominator)
        df['norm_volume'] = df['norm_volume'].fillna(0.5)
    else:
        df['norm_volume'] = 0.5  # Default if no volume data

    # 4. Technical Indicators using stockstats
    # stockstats requires lowercase column names: open, high, low, close, volume
    stock_df_input = pd.DataFrame(index=df.index)
    stock_df_input['open'] = df['mid_open']
    stock_df_input['high'] = df['mid_high']
    stock_df_input['low'] = df['mid_low']
    stock_df_input['close'] = df['mid_close']
    if 'volume' in df.columns:
        stock_df_input['volume'] = df['volume']
    else:  # stockstats might need a volume column, even if it's just zeros or ones
        stock_df_input['volume'] = np.zeros(len(df))

    stock_sdataframe = StockDataFrame.retype(stock_df_input)

    # RSI
    rsi_col_name = f'rsi_{RSI_PERIOD}'
    df['rsi'] = stock_sdataframe[rsi_col_name]

    # MACD
    # stockstats calculates 'macd' (MACD line), 'macds' (signal line), 'macdh' (histogram = macd - macds)
    # We need MACD difference (histogram)
    df['macd_diff'] = stock_sdataframe['macdh']  # macdh is macd - macds

    # Normalize indicators (MinMaxScaler like approach over the entire dataset for simplicity here)
    # For a proper setup, fit scalers ONLY on training data and transform train/test.
    for col in ['rsi', 'macd_diff']:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[f'norm_{col}'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f'norm_{col}'] = 0.5
        df[f'norm_{col}'] = df[f'norm_{col}'].fillna(0.5)  # Fill NaNs from indicator calculation period

    df = df[features + ['close_bid', 'close_ask']]
    df = df.fillna(0)

    if len(df) < lookback_window_size:
        raise ValueError("DataFrame length is less than lookback_window_size after processing.")

    return df

def filter_df(df: pd.DataFrame):
    """
    Removes rows containing NaNs or Infinities.
    """
    return df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

def run_model_on_vec_env(
        model: Any,
        env: VecEnv,
        log_path: str | Path,
        n_episodes: int = 10,
        max_timesteps: int = int(1e9)
    ):
    """
    Run a trained RL model on a vectorized environment for a number of episodes and log each step.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        episode_counts = [0] * env.num_envs
        timesteps = [0] * env.num_envs
        obs = env.reset()

        while min(episode_counts) < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, rewards, dones, infos = env.step(action)

            for i in range(env.num_envs):
                if episode_counts[i] < n_episodes:
                    log_entry = {
                        "env_index": i,
                        "episode": episode_counts[i],
                        "timestep": timesteps[i],
                        "action": action[i].tolist() if hasattr(action[i], "tolist") else action[i],
                        "obs": obs[i].tolist() if hasattr(obs[i], "tolist") else obs[i],
                        "reward": float(rewards[i]),
                        "done": bool(dones[i]),
                        #"info": infos[i]
                    }
                    f.write(json.dumps(log_entry) + "\n")
                    timesteps[i] += 1

                    if dones[i]:
                        episode_counts[i] += 1
                        timesteps[i] = 0

            obs = next_obs
