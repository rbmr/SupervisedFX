import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from ForexEnv_RQ5 import ForexEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from common.constants import DEVICE, SEED
from common.data import ForexData, combine_df

def make_env(df, seed=None):
    def _init():
        env = ForexEnv(df.copy(), log_level=1, seed=seed, debug_mode=False)
        return env
    return _init

if __name__ == "__main__":
    # Load data
    bid_path = 'D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\data\\forex\\EURUSD\\15M\\BID\\10.05.2022T00.00-10.05.2025T23.45.csv'
    ask_path = 'D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\data\\forex\\EURUSD\\15M\\ASK\\10.05.2022T00.00-10.05.2025T23.45.csv'

    ask_df = ForexData(ask_path).df
    bid_df = ForexData(bid_path).df
    forex_data = combine_df(bid_df, ask_df)

    train_split = 0.8
    split_idx = int(len(forex_data) * train_split)
    train_df = forex_data.iloc[:split_idx]
    test_df = forex_data.iloc[split_idx:]

    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))

    # Env setup
    NUM_ENVS = 8
    train_env = SubprocVecEnv([make_env(train_df, seed=i) for i in range(NUM_ENVS)])

    # Model setup
    policy_kwargs = dict(net_arch=[128, 128])
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )

    # Train
    TOTAL_TIMESTEPS = 100_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100_000, progress_bar=True)
    model.save("D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\models\\dqn_forex_model.zip")
