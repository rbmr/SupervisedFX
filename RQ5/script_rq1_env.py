import random

import numpy as np
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from ForexEnv_RQ1 import ForexEnv
from common.data import ForexData
from common.constants import *
from common.scripts import *

# set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Get ask and bid data, and combine
ask_path = FOREX_DIR / "EURUSD" / "15M" / "ASK" / "10.05.2022T00.00-10.05.2025T23.45.csv"
bid_path = FOREX_DIR / "EURUSD" / "15M" / "BID" / "10.05.2022T00.00-10.05.2025T23.45.csv"
ask_df = ForexData(ask_path).df
bid_df = ForexData(ask_path).df
forex_data = combine_df(bid_df, ask_df)
forex_data = filter_df(forex_data)
train_df, eval_df = split_df(forex_data, 0.7)

# --- Configuration Parameters ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.0 # Example: 0.1% commission per trade
LOOKBACK_WINDOW_SIZE = 30 # Number of past timesteps to include in the state

print("Creating training environment...")
train_env = DummyVecEnv([lambda: ForexEnv(train_df,
                                            initial_capital=INITIAL_CAPITAL,
                                            transaction_cost_pct=TRANSACTION_COST_PCT,
                                            lookback_window_size=LOOKBACK_WINDOW_SIZE)])
print("Training environment created.")

policy_kwargs = dict(net_arch=[128, 128])
model = A2C(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.001,
    gamma=0.99,
    n_steps=10,                 # Slightly higher n_steps for more stable estimates
    ent_coef=0.02,              # Increase entropy coefficient to encourage more exploration
    gae_lambda=0.95,            # Lower lambda for more bias, but faster learning
    vf_coef=0.5,                # Value function loss coefficient (default)
    max_grad_norm=0.5,          # Gradient clipping (default)
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=DEVICE
)

print("Training the DQN agent...")
model.learn(total_timesteps=10_000)
print("Training finished.")

print("Saving the DQN model...")
model_path = "D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\models\\forex_model_rq1.zip"
model.save(model_path)
print(f"Model saved to {model_path}.")


print("\\nEvaluating the agent on the eval_df...")
eval_env = DummyVecEnv([lambda: ForexEnv(eval_df,
                                         initial_capital=INITIAL_CAPITAL,
                                         transaction_cost_pct=TRANSACTION_COST_PCT,
                                         lookback_window_size=LOOKBACK_WINDOW_SIZE)])

n_eval_episodes = 10
max_timesteps_per_episode = 1e9
model_name = "forex_model_rq1"
log_path = "D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\logs\\" + model_name
run_model_on_vec_env(model, eval_env, log_path)