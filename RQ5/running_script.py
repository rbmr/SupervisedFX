import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from ForexEnv_RQ5 import ForexEnv  # or your env filename
from common.data import ForexData, combine_df

# --- Load your saved model ---
model_path = "D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\models\\dqn_forex_model.zip"
model = DQN.load(model_path)

# --- Load 1min EUR/USD data ---
bid_path = 'D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\data\\forex\\EURUSD\\15M\\BID\\10.05.2022T00.00-10.05.2025T23.45.csv'
ask_path = 'D:\\Facultate\\Y3\\Q4\\TUD-CSE-RP-RLinFinance\\RQ5\\data\\forex\\EURUSD\\15M\\ASK\\10.05.2022T00.00-10.05.2025T23.45.csv'

ask_df = ForexData(ask_path).df
bid_df = ForexData(bid_path).df
forex_data = combine_df(bid_df, ask_df)

# --- Split into train/test ---
train_split = 0.8
split_idx = int(len(forex_data) * train_split)
train_df = forex_data.iloc[:split_idx]
test_df = forex_data.iloc[split_idx:]

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

# --- Create test environment ---
env = DummyVecEnv([lambda: ForexEnv(test_df.copy(), log_level=0, debug_mode=True)])

obs = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0
equity_history = []
action_history = []

print("\\n--- EVALUATION DEBUG LOG ---\\n")
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)

    print(action)
    # Uncomment to test action flip (LONG ↔ SHORT)
    # action = np.where(action == 1, 2, np.where(action == 2, 1, action))

    obs, reward, dones, infos = env.step(action)
    terminated = dones[0]
    truncated = infos[0].get("TimeLimit.truncated", False) or infos[0].get("truncated", False)
    current_equity = infos[0].get("equity", None)
    equity_history.append(current_equity)
    action_history.append(action[0])
    total_reward += reward[0]

    print(f"[Step {step:>4}] Action: {action[0]} | Equity: {current_equity:.2f} | Reward: {reward[0]:+.5f}")
    step += 1

print("\\n--- FINAL STATS ---")
print(f"Total steps: {step}")
print(f"Final equity: {equity_history[-1]:.2f}")
print(f"Total cumulative reward: {total_reward:.2f}")

from collections import Counter
action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CASH"}
counts = Counter(action_history)
print("\\nAction distribution:")
for k, v in counts.items():
    print(f"  {action_map.get(k, '?')} ({k}): {v} times")
