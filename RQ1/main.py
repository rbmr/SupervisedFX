import logging
import random

from RQ1.constants import LOGS_DIR, MODELS_DIR
from RQ1.scripts import prepare_data, filter_df, run_model_on_vec_env
import gymnasium as gym
import numpy as np
from common.scripts import combine_df
from gymnasium import spaces
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from typing import NamedTuple

from common.data import ForexCandleData
from common.constants import *
from common.scripts import *

if __name__ != '__main__':
    raise ImportError("Do not import this module.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Parameters ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.0 # Example: 0.1% commission per trade
LOOKBACK_WINDOW_SIZE = 30 # Number of past timesteps to include in the state

class State(NamedTuple):
    step: int               # step=0 is on reset, increments at start of every .step call.
    action: float | None    # action staken at step
    bid_price: float        # bid_price at step
    ask_price: float        # ask_price at step
    position: float         # position at step
    shares: float           # shares at step
    cash: float             # cash at step
    equity: float           # equity at step
    reward: float | None    # reward received at step

class ForexEnv(gym.Env):

    def __init__(self, df,
                 initial_capital=INITIAL_CAPITAL,
                 transaction_cost_pct=TRANSACTION_COST_PCT,
                 lookback_window_size=LOOKBACK_WINDOW_SIZE):
        super(ForexEnv, self).__init__()

        # Validate input
        if transaction_cost_pct < 0.0 or transaction_cost_pct > 1.0: 
            raise ValueError(f"transaction_cost_pct must be between 0.0 and 1.0, was {transaction_cost_pct}")
        if lookback_window_size < 1:
            raise ValueError("lookback_window_size must be larger than 0, was {lookback_window_size}")
        if len(df) < lookback_window_size:
            raise ValueError(f"DataFrame length (was {len(df)}) must be more than lookback_window_size ({lookback_window_size})")

        # Environment settings
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window_size = lookback_window_size
        self.market_features = ['log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff']
        self.df = prepare_data(df, lookback_window_size, self.market_features)
        self.total_steps = len(self.df) - self.lookback_window_size - 1

        # Define action space: -1 (Fully short), 0 (Cash), 1 (Fully Long)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space
        self.num_market_features = len(self.market_features)
        self.observation_space_shape = (self.lookback_window_size * self.num_market_features) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_shape,), dtype=np.float32)

        # Internal state variables
        self.current_step = 0                       # Determines which prices are current.
        self.cash = self.initial_capital            # Available cash not invested in any position
        self.shares = 0.0                           # Number of shares held (positive for long, negative for short)
        self.equity = self.initial_capital          # (Cache) Total portfolio value (cash + position value)
        self.position = 0.0                         # (Cache) Position value / equity (in -1 to 1, short to long)

        bid_price, ask_price = self._get_current_prices()
        self.state_trace = [State(self.current_step, None, bid_price, ask_price, self.position, self.shares, self.cash, self.equity, None),]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Internal state variables
        self.current_step = 0                       # Determines which prices are current.
        self.cash = self.initial_capital            # Available cash not invested in any position
        self.shares = 0.0                           # Number of shares held (positive for long, negative for short)
        self.equity = self.initial_capital          # (Cache) Total portfolio value (cash + position value)
        self.position = 0.0                         # (Cache) Position value / equity (in -1 to 1, short to long)

        bid_price, ask_price = self._get_current_prices()
        self.state_trace = [State(self.current_step, None, bid_price, ask_price, self.position, self.shares, self.cash, self.equity, None),]

        return self._get_observation(), {}

    def step(self, action):
        
        # Perform step
        self.current_step += 1
        prev_equity = self.equity

        current_bid, current_ask = self._get_current_prices()
        target_position = float(action[0])
        self._execute_trade(target_position, current_bid, current_ask)

        self.equity = self._calculate_equity(current_bid, current_ask)
        reward = self.equity - prev_equity

        # Determine done
        terminated = False
        truncated = False
        if self.equity <= 0:
            terminated = True
            logging.info(f"Step {self.current_step}: Agent ruined. Equity: {self.equity:.2f}")
        if self.current_step >= self.total_steps - 1:
            truncated = True
            logging.info(f"Step {self.current_step}: End of data reached.") 

        # Save state
        self.state_trace.append(State(
            step=self.current_step, action=action, bid_price=current_bid, ask_price=current_ask, 
            position=self.position, shares=self.shares, cash=self.cash, 
            equity=self.equity, reward=reward
        ))
        
        logging.debug(f"Step: {self.current_step}, Action: {action}, Bid: {current_bid:.5f}, Ask: {current_ask:.5f}, "
                     f"Position: {self.position:.4f}, Shares: {self.shares:.2f}, Cash: {self.cash:.2f}, "
                     f"Equity: {self.equity:.2f}, Reward: {reward:.2f}")

        # Determine info dict 
        info = {}
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            # Make a copy of the list to avoid issues if it's modified elsewhere later
            info['episode_state_trace'] = list(self.state_trace)
            # Add any other summary stats for the episode you might want
        
        return self._get_observation(), reward, terminated, truncated, info

    def _get_current_prices(self) -> tuple[float, float]:
        """
        Retrieves the current bid and ask prices from the dataframe.
        """
        idx = self.current_step + self.lookback_window_size
        if idx >= len(self.df):
            raise ValueError("Reached end of data, no price available.")
        current_bid: float = self.df['close_bid'].iloc[idx]
        current_ask: float = self.df['close_ask'].iloc[idx]
        return current_bid, current_ask

    def _calculate_equity(self, bid_price: float, ask_price: float) -> float:
        """
        Calculates the equity based on current cash, shares and prices.
        """
        return self.cash + self.shares * (bid_price if self.shares > 0 else ask_price)

    def _get_observation(self):

        # Get lookback
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window_size
        market_data_window = self.df[self.market_features].iloc[start_idx:end_idx].values.flatten()

        # Get portfolio information
        current_bid, current_ask = self._get_current_prices()
        current_equity = self._calculate_equity(current_bid, current_ask)
        cash_fraction = np.clip(self.cash / current_equity, -1e6, 1e6) if current_equity != 0 else 0
        portfolio_data = np.array([cash_fraction, self.position], dtype=np.float32)

        # Combine
        observation = np.concatenate((market_data_window, portfolio_data))
        return observation.astype(np.float32)

    def _buy_shares(self, shares_to_buy: float, ask_price: float) -> tuple[float, float, float]:
        """
        Attempts to buy a specific number of shares given an ask_price.
        If not enough cash, buys the maximum number of shares possible.
        Updates internal state variables accordingly.
        Returns the number of shares bought, the cost, and the commission.
        """
        if shares_to_buy < 0:
            raise ValueError(f"shares must be positive, was {shares_to_buy}")
        if ask_price < 0:
            raise ValueError(f"ask_price must be positive, was {ask_price}")
        if self.cash < 0:
            raise ValueError(f"cash must be positive, was {self.cash}")
        
        # Buy as much as possible
        max_affordable = self.cash / (ask_price * (1 + self.transaction_cost_pct))
        shares_bought = min(shares_to_buy, max_affordable)
        cost = shares_bought * ask_price
        commission = cost * self.transaction_cost_pct
        
        # Update internal state
        self.cash -= (cost + commission)
        self.shares += shares_bought
        return shares_bought, cost, commission

    def _sell_shares(self, shares_to_sell: float, bid_price: float) -> tuple[float, float, float]:
        """
        Sells a specific number of shares given an bid_price.
        Updates internal state variables accordingly.
        Returns the number of shares sold, the value they were sold for, and the commission.
        """
        if shares_to_sell < 0:
            raise ValueError(f"shares must be positive, was {shares_to_sell}")
        if bid_price < 0:
            raise ValueError(f"bid_price must be positive, was {bid_price}")
        
        # Sell
        sell_value = shares_to_sell * bid_price
        commission = sell_value * self.transaction_cost_pct
        
        # Update internal state
        self.cash += sell_value - commission
        self.shares -= shares_to_sell
        return shares_to_sell, sell_value, commission

    def _execute_trade(self, target_position: float, bid_price: float, ask_price: float):
        """
        Attempts to transfer the portfolio to the target position ratio, where -1.0 is full short, 0.0 is cash, 1.0 is full long.
        """
        if target_position < -1.0 or target_position > 1.0:
            raise ValueError(f"Target position must be in [-1.0, 1.0], was {target_position}")

        # Calculate position change needed
        position_change = target_position - self.position
        if abs(position_change) < 1e-6:
            return

        # Step 1: Close opposing positions if necessary
        if self.shares < 0 and position_change > 0:  # Have short position but want more long exposure
            
            # Calculate shares to cover
            denom = 1 + self.position
            if abs(denom) < 1e-6:
                shares_to_cover = -self.shares
            else:
                shares_to_cover = min(-self.shares, -self.shares * (position_change / (1 + self.position)))
            self._buy_shares(shares_to_cover, ask_price)
        
        elif self.shares > 0 and position_change < 0:  # Have long position but want more short exposure
            
            # Calculate shares to sell
            if self.position == 0:
                shares_to_sell = self.shares
            else: 
                shares_to_sell = min(self.shares, self.shares * (abs(position_change) / self.position))
            self._sell_shares(shares_to_sell, bid_price)
        
        # Step 2: Establish new position in the target direction
        if target_position > 0 and self.cash > 0:  # Want long position
            
            # Calculate shares to buy
            current_equity = self._calculate_equity(bid_price, ask_price)
            target_shares = (current_equity * target_position) / ask_price
            shares_to_buy = max(0, target_shares - self.shares)  # How many more shares we need
            self._buy_shares(shares_to_buy, ask_price)

        elif target_position < 0 and self.cash > 0:  # Want short position
            
            # Target short shares based on desired equity exposure
            current_equity = self._calculate_equity(bid_price, ask_price)
            target_short_shares = (current_equity * abs(target_position)) / bid_price
            shares_to_short = max(0, target_short_shares - abs(min(0, self.shares)))  # How many more to short
            self._sell_shares(shares_to_short, bid_price)
          
        # Step 3: Update the position indicator based on the new state
        current_equity = self._calculate_equity(bid_price, ask_price)
        position_value = self.shares * (bid_price if self.shares > 0 else ask_price)
        self.position = np.clip(position_value / current_equity, -1.0, 1.0) if current_equity != 0 else 0

# set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Get ask and bid data, and combine
ask_path = FOREX_DIR / "EURUSD" / "15M" / "ASK" / "10.05.2022T00.00-10.05.2025T23.45.csv"
bid_path = FOREX_DIR / "EURUSD" / "15M" / "BID" / "10.05.2022T00.00-10.05.2025T23.45.csv"
ask_df = ForexCandleData(ask_path).df
bid_df = ForexCandleData(ask_path).df
forex_data = combine_df(bid_df, ask_df)
forex_data = filter_df(forex_data)
train_df, eval_df = split_df(forex_data, 0.7)

logging.info("Creating training environment...")
train_env = DummyVecEnv([lambda: ForexEnv(train_df,
                                            initial_capital=INITIAL_CAPITAL,
                                            transaction_cost_pct=TRANSACTION_COST_PCT,
                                            lookback_window_size=LOOKBACK_WINDOW_SIZE)])
logging.info("Training environment created.")

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

logging.info("Training the DQN agent...")
model.learn(total_timesteps=200_000)
logging.info("Training finished.")

logging.info("Saving the DQN model...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{timestamp}_A2C_model"
model_path = MODELS_DIR / model_name
model.save(model_path)
logging.info(f"Model saved to {model_path}.")

logging.info("\nEvaluating the agent on the eval_df...")
eval_env = DummyVecEnv([lambda: ForexEnv(eval_df,
                                         initial_capital=INITIAL_CAPITAL,
                                         transaction_cost_pct=TRANSACTION_COST_PCT,
                                         lookback_window_size=LOOKBACK_WINDOW_SIZE)])

n_eval_episodes = 10
max_timesteps_per_episode = 1e9
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{timestamp}_A2C_model"
log_path = LOGS_DIR / model_name
run_model_on_vec_env(model, eval_env, log_path)

