import logging
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.scripts import combine_df
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame

from common.data import ForexData
from common.constants import *
from common.scripts import *

if __name__ != '__main__':
    raise ImportError("Do not import this module.")

# --- Configuration Parameters ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.0 # Example: 0.1% commission per trade
LOOKBACK_WINDOW_SIZE = 30 # Number of past time steps to include in the state
RSI_PERIOD = 14 # Technical Indicators Periods (stockstats uses these by appending to indicator name e.g. rsi_14)


class ForexEnv(gym.Env):
    """
    A custom trading environment for Forex EURUSD with bid/ask spread and transaction costs.
    Actions: 0: CASH, 1: LONG, 2: SHORT
    Uses stockstats for technical indicators.
    """

    def __init__(self, df,
                 initial_capital=INITIAL_CAPITAL,
                 transaction_cost_pct=TRANSACTION_COST_PCT,
                 lookback_window_size=LOOKBACK_WINDOW_SIZE):
        super(ForexEnv, self).__init__()

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window_size = lookback_window_size

        # Define action space: 0 (CASH), 1 (LONG), 2 (SHORT)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Market features: log_ret_bid, log_ret_ask, norm_spread, norm_vol, norm_rsi, norm_macd_diff (6 features)
        self.num_market_features = 6
        self.observation_space_shape = (self.lookback_window_size * self.num_market_features) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_shape,), dtype=np.float32)

        # Internal state variables
        self.current_step = 0
        self.cash = 0
        self.shares = 0 # Positive for LONG, negative for SHORT
        self.position = 0 # 0: CASH, 1: LONG, -1: SHORT
        self.entry_price_long = 0.0
        self.entry_price_short = 0.0
        self.equity = self.initial_capital
        self.total_steps = len(self.df) - self.lookback_window_size -1

        self.equity_history = []
        self.action_history = []
        self.trade_history = []

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the data by calculating necessary features and normalizing them.
        Uses stockstats for RSI and MACD.
        """
        df = self.df.copy()

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
            vol_min = df['volume'].rolling(window=self.lookback_window_size, min_periods=1).min()
            vol_max = df['volume'].rolling(window=self.lookback_window_size, min_periods=1).max()
            denominator = vol_max - vol_min
            df['norm_volume'] = np.where(denominator == 0, 0.5, (df['volume'] - vol_min) / denominator)
            df['norm_volume'] = df['norm_volume'].fillna(0.5)
        else:
            df['norm_volume'] = 0.5 # Default if no volume data


        # 4. Technical Indicators using stockstats
        # stockstats requires lowercase column names: open, high, low, close, volume
        stock_df_input = pd.DataFrame(index=df.index)
        stock_df_input['open'] = df['mid_open']
        stock_df_input['high'] = df['mid_high']
        stock_df_input['low'] = df['mid_low']
        stock_df_input['close'] = df['mid_close']
        if 'volume' in df.columns:
            stock_df_input['volume'] = df['volume']
        else: # stockstats might need a volume column, even if it's just zeros or ones
            stock_df_input['volume'] = np.zeros(len(df))


        stock_sdataframe = StockDataFrame.retype(stock_df_input)

        # RSI
        rsi_col_name = f'rsi_{RSI_PERIOD}'
        df['rsi'] = stock_sdataframe[rsi_col_name]

        # MACD
        # stockstats calculates 'macd' (MACD line), 'macds' (signal line), 'macdh' (histogram = macd - macds)
        # We need MACD difference (histogram)
        df['macd_diff'] = stock_sdataframe['macdh'] # macdh is macd - macds

        # Normalize indicators (MinMaxScaler like approach over the entire dataset for simplicity here)
        # For a proper setup, fit scalers ONLY on training data and transform train/test.
        for col in ['rsi', 'macd_diff']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'norm_{col}'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'norm_{col}'] = 0.5
            df[f'norm_{col}'] = df[f'norm_{col}'].fillna(0.5) # Fill NaNs from indicator calculation period

        self.processed_df = df[[
            'log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff',
            'close_bid', 'close_ask' # Keep original prices for trading logic
        ]].copy()
        self.processed_df = self.processed_df.fillna(0)

        if len(self.processed_df) < self.lookback_window_size:
            raise ValueError("DataFrame length is less than lookback_window_size after processing.")

    def _get_current_prices(self):
        idx = self.current_step + self.lookback_window_size
        if idx >= len(self.processed_df):
            idx = len(self.processed_df) -1
        current_bid = self.processed_df.iloc[idx]['close_bid']
        current_ask = self.processed_df.iloc[idx]['close_ask']
        return current_bid, current_ask

    def _calculate_equity(self, current_bid, current_ask):
        if self.position == 1: # LONG
            return self.cash + self.shares * current_bid
        elif self.position == -1: # SHORT
            return self.cash - abs(self.shares) * current_ask
        else: # CASH
            return self.cash

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.position = 0
        self.entry_price_long = 0.0
        self.entry_price_short = 0.0
        self.equity = self.initial_capital
        self.equity_history = [self.initial_capital]
        self.action_history = []
        self.trade_history = []
        return self._get_observation(), {}

    def _get_observation(self):
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window_size
        market_data_window = self.processed_df[[
            'log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff'
        ]].iloc[start_idx:end_idx].values.flatten()

        current_bid, current_ask = self._get_current_prices()
        current_equity = self._calculate_equity(current_bid, current_ask)
        if current_equity == 0 :
            cash_fraction = 0
        else:
            cash_fraction = np.clip(self.cash / current_equity, -1e6, 1e6)

        pos_long_indicator = 1.0 if self.position == 1 else 0.0
        pos_short_indicator = 1.0 if self.position == -1 else 0.0
        portfolio_data = np.array([cash_fraction, pos_long_indicator, pos_short_indicator], dtype=np.float32)

        observation = np.concatenate((market_data_window, portfolio_data))
        return observation.astype(np.float32)


    def step(self, action):
        self.current_step += 1
        terminated = False # Gymnasium uses "terminated"
        truncated = False # Gymnasium uses "truncated"

        prev_equity = self.equity
        current_bid, current_ask = self._get_current_prices()

        self._execute_trade(action, current_bid, current_ask)
        self.action_history.append(action)

        self.equity = self._calculate_equity(current_bid, current_ask)
        self.equity_history.append(self.equity)

        reward = self.equity - prev_equity

        if self.equity <= 0:
            terminated = True
            logging.info(f"Step {self.current_step}: Agent ruined. Equity: {self.equity:.2f}")

        if self.current_step >= self.total_steps -1 :
            truncated = True
            logging.info(f"Step {self.current_step}: End of data reached.")

        observation = self._get_observation()

        info_dict = {} # Initialize info dictionary
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            # Make a copy of the list to avoid issues if it's modified elsewhere later
            info_dict['episode_action_history'] = list(self.action_history)
            info_dict['episode_equity_history'] = list(self.equity_history) # Example of other final info
            # Add any other summary stats for the episode you might want

        logging.info(f"Step: {self.current_step}, Action: {action}, Bid: {current_bid:.5f}, Ask: {current_ask:.5f}, "
                     f"Shares: {self.shares:.2f}, Cash: {self.cash:.2f}, Equity: {self.equity:.2f}, Reward: {reward:.2f}")

        return observation, reward, terminated, truncated, {}


    def _execute_trade(self, action, bid_price, ask_price):
        logging.info(f"Executing trade action: {action} at step {self.current_step}.")

        # Action 0: CASH, 1: LONG, 2: SHORT
        if action == 0: # Go to CASH
            if self.position == 1: # Was LONG
                sell_value = self.shares * bid_price
                commission = sell_value * self.transaction_cost_pct
                self.cash += sell_value - commission
                logging.warning(f"Step {self.current_step}: Closing LONG. Sold {self.shares:.2f} at {bid_price:.5f}. Commission: {commission:.2f}. Cash: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "CLOSE_LONG", bid_price, self.shares, commission))
                self.shares = 0
                self.position = 0
            elif self.position == -1: # Was SHORT
                cost_to_cover = abs(self.shares) * ask_price
                commission = cost_to_cover * self.transaction_cost_pct
                self.cash -= (cost_to_cover + commission)
                logging.warning(f"Step {self.current_step}: Closing SHORT. Covered {abs(self.shares):.2f} at {ask_price:.5f}. Commission: {commission:.2f}. Cash: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "CLOSE_SHORT", ask_price, abs(self.shares), commission))
                self.shares = 0
                self.position = 0

        elif action == 1: # Go LONG
            if self.position == -1: # Close SHORT first
                cost_to_cover = abs(self.shares) * ask_price
                commission = cost_to_cover * self.transaction_cost_pct
                self.cash -= (cost_to_cover + commission)
                logging.warning(f"Step {self.current_step}: Closing SHORT before LONG. Covered {abs(self.shares):.2f} at {ask_price:.5f}. Commission: {commission:.2f}. Cash: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "CLOSE_SHORT_FOR_LONG", ask_price, abs(self.shares), commission))
                self.shares = 0
                self.position = 0

            if self.position == 0 and self.cash > 0:
                if ask_price <= 0 or (ask_price * (1 + self.transaction_cost_pct)) <= 0 :
                    logging.warning(f"Step {self.current_step}: Cannot go LONG, invalid ask price or total cost. Ask: {ask_price}")
                    return
                num_shares_to_buy = self.cash / (ask_price * (1 + self.transaction_cost_pct))
                cost_of_shares = num_shares_to_buy * ask_price
                commission = cost_of_shares * self.transaction_cost_pct
                self.shares = num_shares_to_buy
                self.cash -= (cost_of_shares + commission)
                self.position = 1
                self.entry_price_long = ask_price
                logging.warning(f"Step {self.current_step}: Opened LONG. Bought {self.shares:.2f} at {ask_price:.5f}. Commission: {commission:.2f}. Cash left: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "OPEN_LONG", ask_price, self.shares, commission))
            elif self.cash <= 0:
                logging.warning(f"Step {self.current_step}: Cannot go LONG, insufficient cash: {self.cash:.2f}")

        elif action == 2: # Go SHORT
            if self.position == 1: # Close LONG first
                sell_value = self.shares * bid_price
                commission = sell_value * self.transaction_cost_pct
                self.cash += sell_value - commission
                logging.warning(f"Step {self.current_step}: Closing LONG before SHORT. Sold {self.shares:.2f} at {bid_price:.5f}. Commission: {commission:.2f}. Cash: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "CLOSE_LONG_FOR_SHORT", bid_price, self.shares, commission))
                self.shares = 0
                self.position = 0

            if self.position == 0 and self.cash > 0:
                if bid_price <= 0 :
                    logging.warning(f"Step {self.current_step}: Cannot go SHORT, invalid bid price: {bid_price}")
                    return
                notional_value_to_short = self.cash
                num_shares_to_short = notional_value_to_short / bid_price
                proceeds_from_sale = num_shares_to_short * bid_price
                commission = proceeds_from_sale * self.transaction_cost_pct
                self.cash += proceeds_from_sale - commission
                self.shares = -num_shares_to_short
                self.position = -1
                self.entry_price_short = bid_price
                logging.warning(f"Step {self.current_step}: Opened SHORT. Shorted {abs(self.shares):.2f} at {bid_price:.5f}. Notional: {notional_value_to_short:.2f}. Commission: {commission:.2f}. Cash: {self.cash:.2f}")
                self.trade_history.append((self.current_step, "OPEN_SHORT", bid_price, abs(self.shares), commission))
            elif self.cash <=0:
                logging.warning(f"Step {self.current_step}: Cannot go SHORT, insufficient cash for notional value: {self.cash:.2f}")

    def render(self, mode='human'):
        if mode == 'human':
            logging.info(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Shares: {self.shares:.2f}, Cash: {self.cash:.2f}, Position: {self.position}")

    def close(self):
        pass

# set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get ask and bid data, and combine
ask_path = "C:\\Users\\rober\\TUD-CSE-RP-RLinFinance\\data\\forex\\EURUSD\\15M\\ASK\\10.05.2022T00.00-10.05.2025T23.45.csv"
bid_path = "C:\\Users\\rober\\TUD-CSE-RP-RLinFinance\\data\\forex\\EURUSD\\15M\\BID\\10.05.2022T00.00-10.05.2025T23.45.csv"
ask_df = ForexData(ask_path).df
bid_df = ForexData(ask_path).df
forex_data = combine_df(bid_df, ask_df)

# remove NaNs, and Infinities.
forex_data = forex_data[~forex_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

# Split up data
train_df, trade_df = split_df(forex_data, 0.7)

if len(train_df) <= LOOKBACK_WINDOW_SIZE + 50 or len(trade_df) <= LOOKBACK_WINDOW_SIZE + 50 :
    logging.error(f"Not enough data for reliable training/testing.")

print("Creating training environment...")
train_env = DummyVecEnv([lambda: ForexEnv(train_df,
                                            initial_capital=INITIAL_CAPITAL,
                                            transaction_cost_pct=TRANSACTION_COST_PCT,
                                            lookback_window_size=LOOKBACK_WINDOW_SIZE,
                                            seed=SEED)])
print("Training environment created.")

policy_kwargs = dict(net_arch=[128, 128])

model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=DEVICE
)

print("Training the DQN agent...")
TOTAL_TIMESTEPS = 50000 # Reduce for quicker testing if needed
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100) # Increased log_interval
print("Training finished.")

print("Saving the DQN model...")
models_dir = Path(__file__).resolve().parent / "models"
model_path = models_dir / "dqn_forex_eurusd_stockstats_model"
model.save(model_path)
print("Model saved to {model_path}.")

print("\nEvaluating the agent on the trade_df...")
eval_env = DummyVecEnv([lambda: ForexEnv(trade_df,
                                         initial_capital=INITIAL_CAPITAL,
                                         transaction_cost_pct=TRANSACTION_COST_PCT,
                                         lookback_window_size=LOOKBACK_WINDOW_SIZE)])

obs = eval_env.reset()
terminated = False
truncated = False
total_rewards_eval = 0
num_eval_steps = 0

episode_action_history_from_info = None # To store history if episode completes
 
for i in range(len(trade_df) - LOOKBACK_WINDOW_SIZE - 2):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones_arr, infos_arr = eval_env.step(action)

    done_from_vec_env = dones_arr[0]
    current_info_dict = infos_arr[0] # Get the info dict for our single environment

    total_rewards_eval += rewards[0]
    num_eval_steps +=1

    if done_from_vec_env:
        print(f"Evaluation episode ended at step {i+1} due to done=True.")
        if 'episode_action_history' in current_info_dict:
            episode_action_history_from_info = current_info_dict['episode_action_history']
            # print(f"  Retrieved action history (len: {len(episode_action_history_from_info)}) via info dict.") # For debug
        else:
            # print("  Done, but 'episode_action_history' not found in info dict this step.") # For debug
            pass # Will try direct access later if this was not populated

        # Optional: Print termination reason from info
        if current_info_dict.get("terminated", False):
            print("  Reason: Environment self-terminated (e.g., agent ruined).")
        elif current_info_dict.get("TimeLimit.truncated", False) or current_info_dict.get("truncated", False):
            print("  Reason: Episode truncated (e.g., time limit or end of data).")
        break # Exit loop as episode is done

# After the loop, determine which action history to use
action_history_to_print = []
if episode_action_history_from_info is not None:
    print("[Main] Using action history captured from 'info' dict upon episode completion.")
    action_history_to_print = episode_action_history_from_info
else:
    # Fallback: If episode didn't 'done' or info wasn't populated, try direct access.
    # This will show history if loop ended by range, or an empty list if env was reset and info not caught.
    underlying_eval_env = eval_env.envs[0]
    print("[Main] Using action history via direct access from underlying_eval_env (may be from reset state or incomplete episode if loop ended by range).")
    if hasattr(underlying_eval_env, 'action_history'):
        action_history_to_print = underlying_eval_env.action_history
    else:
        print("  [Main Fallback] underlying_eval_env does not have action_history attribute.")


# Now print action_history_to_print
print("\nAction History (showing first 50 actions from evaluation):")
if action_history_to_print: # Check if the list is not None and not empty
    action_map = {0: "CASH", 1: "LONG", 2: "SHORT"}
    for i, action_code in enumerate(action_history_to_print[:50]):
        action_name = action_map.get(action_code, f"Unknown action code: {action_code}")
        print(f"Eval Step {i + 1}: Action Code = {action_code} ({action_name})")
    
    if len(action_history_to_print) > 50:
        print(f"... and {len(action_history_to_print) - 50} more actions not shown.")
    
    from collections import Counter
    action_counts = Counter(action_history_to_print)
    print("\nAction Counts in Evaluation Episode:")
    for action_code, count in action_counts.items():
        action_name = action_map.get(action_code, f"Unknown action code: {action_code}")
        print(f"  {action_name}: {count} times")
else:
    print("No action history available or history is empty after evaluation.")
