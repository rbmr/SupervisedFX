import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from common.constants import *
from common.data.data import ForexCandleData
from common.data.feature_engineer import FeatureEngineer
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer
from common.scripts import find_first_row_with_nan, find_first_row_without_nan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForexEnv(gym.Env):

    def __init__(self,
                 market_data_df: pd.DataFrame,
                 market_feature_df: pd.DataFrame,
                 agent_feature_engineer: StepwiseFeatureEngineer,
                 initial_capital: float = 10000.0,
                 transaction_cost_pct: float = 0.0,
                 n_actions: int = 1,
                 custom_reward_function: Optional[Callable[['ForexEnv'], float]] = None
                 ):
        super(ForexEnv, self).__init__()

        # Validate input
        if not isinstance(market_data_df, pd.DataFrame):
            raise ValueError(f"market_data_df must be a pandas DataFrame, was {type(market_data_df)}.")
        if not isinstance(market_feature_df, pd.DataFrame):
            raise ValueError(f"market_feature_df must be a pandas DataFrame, was {type(market_feature_df)}.")
        actual_columns = set(market_data_df.columns)
        expected_columns = set(MarketDataCol.all_names())
        if not expected_columns.issubset(actual_columns):
            missing_columns = expected_columns - actual_columns
            raise ValueError(f"market_data_df is missing columns {missing_columns}.")
        if initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, was {initial_capital}.")
        if transaction_cost_pct < 0.0 or transaction_cost_pct > 1.0:
            raise ValueError(f"transaction_cost_pct must be between 0.0 and 1.0, was {transaction_cost_pct}. For 0.1%, use 0.001.")
        if n_actions < 0:
            raise ValueError(f"n_actions must be >= 0. If n_actions == 0, then the action space is continuous between -1 and 1.")

        # Environment parameters
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.agent_feature_engineer = agent_feature_engineer
        self.custom_reward_function = custom_reward_function

        # Market data and Market features
        market_data_df = market_data_df.copy(deep=True)[MarketDataCol.all_names()]
        market_feature_df = market_feature_df.copy(deep=True)
        start_index = find_first_row_without_nan(market_data_df)
        start_index = max(start_index, find_first_row_without_nan(market_feature_df))
        market_data_df = market_data_df.iloc[start_index:]
        market_feature_df = market_feature_df.iloc[start_index:]
        market_data_df.reset_index(drop=True, inplace=True)
        market_feature_df.reset_index(drop=True, inplace=True)

        # Processed data validation
        if len(market_data_df) != len(market_feature_df):
            raise ValueError(f"market_data, and market_features must be the same length, was {len(market_data_df)}, and {len(market_feature_df)}.")
        if market_data_df.isna().any().any():
            first_nan_index = find_first_row_with_nan(market_data_df)
            raise ValueError(f"market_data_df contains NaN values. First NaN index: {first_nan_index}. Row: {market_data_df.iloc[first_nan_index]}.")
        if market_feature_df.isna().any().any():
            first_nan_index = find_first_row_with_nan(market_feature_df)
            raise ValueError(f"market_feature_df contains NaN values. First NaN index: {first_nan_index}. Row: {market_feature_df.iloc[first_nan_index]}.")

        # Step counter
        self.current_step = 0
        self.total_steps = len(market_data_df)

        # Use numpy arrays for speed
        self.market_data = market_data_df.to_numpy(dtype=np.float32)
        self.market_features = market_feature_df.to_numpy(dtype=np.float32)
        self.market_feature_names = market_feature_df.columns.tolist()
        self.agent_data = np.zeros(shape = (self.market_data.shape[0], len(AgentDataCol.all_names())), dtype=np.float32)
        self.agent_data[0, :] = (
            self.initial_capital, # cash
            0.0,                  # shares
            self.initial_capital, # equity_open
            self.initial_capital, # equity_high
            self.initial_capital, # equity_low
            self.initial_capital, # equity_close
        )
        assert self.market_data.shape == (self.total_steps, len(MarketDataCol))
        assert self.market_features.shape == (self.total_steps, len(self.market_feature_names))
        assert self.agent_data.shape == (self.total_steps, len(AgentDataCol))

        # Action space
        self.prev_action = None
        self.n_actions = n_actions
        if self.n_actions == 0:
            logging.info(f"n_actions is zero, using continuous action space")
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            logging.info(f"n_actions is larger than zero, using discrete action space with {n_actions} actions for buys, {n_actions} for sells, and 1 action for no_participation.")
            self.action_space = spaces.Discrete(2 * n_actions + 1)

        # Define observation space
        num_market_features = len(market_feature_df.columns)
        num_state_features = self.agent_feature_engineer.num_of_features()
        observation_space_shape = num_market_features + num_state_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_shape,), dtype=np.float32)   
    
    @staticmethod
    def create_train_eval_envs(
        split_ratio: float,
        forex_candle_data: ForexCandleData,
        market_feature_engineer: FeatureEngineer,
        agent_feature_engineer: StepwiseFeatureEngineer,
        initial_capital: float = 10000.0,
        transaction_cost_pct: float = 0.0,
        n_actions: int = 1,
        custom_reward_function: Optional[Callable[['ForexEnv'], float]] = None,
    ) -> tuple['ForexEnv', 'ForexEnv']:
        """
        Creates training and evaluation environments from ForexCandleData and FeatureEngineers.
        Splits the data into 70% training and 30% evaluation.
        """
        if split_ratio <= 0.0 or split_ratio >= 1.0:
            raise ValueError(f"split_ratio must be between 0.0 and 1.0, was {split_ratio}. For 70% training, use 0.7.")
        if not isinstance(forex_candle_data, ForexCandleData):
            raise ValueError(f"forex_candle_data must be an instance of ForexCandleData, was {type(forex_candle_data)}.")
        if not isinstance(market_feature_engineer, FeatureEngineer):
            raise ValueError(f"market_feature_engineer must be an instance of FeatureEngineer, was {type(market_feature_engineer)}.")
        

        market_data_df = forex_candle_data.df.copy(deep=True)
        market_features_df = market_feature_engineer.run(market_data_df, remove_original_columns=True)

        # Split data
        split_index = int(len(market_data_df) * split_ratio)
        train_market_data = market_data_df.iloc[:split_index]
        eval_market_data = market_data_df.iloc[split_index:]
        train_market_features = market_features_df.iloc[:split_index]
        eval_market_features = market_features_df.iloc[split_index:]

        # Create training environment
        train_env = ForexEnv(
            market_data_df=train_market_data,
            market_feature_df=train_market_features,
            agent_feature_engineer=agent_feature_engineer,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            n_actions=n_actions,
            custom_reward_function=custom_reward_function
        )

        # Create evaluation environment
        eval_env = ForexEnv(
            market_data_df=eval_market_data,
            market_feature_df=eval_market_features,
            agent_feature_engineer=agent_feature_engineer,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            n_actions=n_actions,
            custom_reward_function=custom_reward_function
        )

        return train_env, eval_env

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environment state
        self.current_step = 0
        self.prev_action = None

        return self._get_observation(), {}

    def step(self, action):

        # Standardize action
        action = self._get_continuous_action(action)

        # Perform step
        self.current_step += 1

        # Perform action
        current_data = self.market_data[self.current_step, :]
        current_cash = self.agent_data[self.current_step - 1, AgentDataCol.cash]
        current_shares = self.agent_data[self.current_step - 1, AgentDataCol.shares]
        new_cash, new_shares = self._execute_action(action, self.prev_action, current_data, current_cash, current_shares)
        equity_open, equity_high, equity_low, equity_close = self._calculate_equity(current_data, new_cash, new_shares)
        agent_step_data = (new_cash, new_shares, equity_open, equity_high, equity_low, equity_close)
        agent_step_data = tuple(float(x) for x in agent_step_data) # convert from (1,) shape arrays to floats
        self.agent_data[self.current_step, :] = agent_step_data

        # calculate reward
        reward = self._get_reward()

        # Determine done
        terminated = False
        truncated = False
        if equity_close <= 0:
            terminated = True
            logging.info(f"Step {self.current_step}: Agent ruined. Equity: {equity_close}.")
        if self.current_step >= self.total_steps - 1:
            truncated = True
            logging.info(f"Step {self.current_step}: End of data reached.") 

        # Determine info dict
        info = {}
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            number_of_steps = self.current_step + 1
            market_data = self.market_data[:number_of_steps, :]
            market_features = self.market_features[:number_of_steps, :]
            agent_data = self.agent_data[:number_of_steps, :]

            market_data_df = pd.DataFrame(market_data, columns=MarketDataCol.all_names())
            market_features_df = pd.DataFrame(market_features, columns=self.market_feature_names)
            agent_data_df = pd.DataFrame(agent_data, columns=AgentDataCol.all_names())

            info['market_data'] = market_data_df
            info['market_features'] = market_features_df
            info['agent_data'] = agent_data_df

        self.prev_action = action
        
        return self._get_observation(), reward, terminated, truncated, info

    def _get_reward(self):
        """
        Calculates the reward based on the current equity.
        Uses a custom reward function if provided, otherwise defaults to equity change.
        """
        if self.custom_reward_function is not None:
            return self.custom_reward_function(self)
        current_equity = self.agent_data[self.current_step, AgentDataCol.equity_close]
        prev_equity = self.agent_data[self.current_step - 1, AgentDataCol.equity_close]
        return current_equity - prev_equity

    def _calculate_equity(self, prices: np.ndarray, cash: float, shares: float):
        """
        Calculates the equity based on current cash, shares and prices.
        """
        equity_open = self._calculate_equity_from_prices(prices[MarketDataCol.open_bid], prices[MarketDataCol.open_ask], cash, shares)
        equity_high = self._calculate_equity_from_prices(prices[MarketDataCol.high_bid], prices[MarketDataCol.high_ask], cash, shares)
        equity_low = self._calculate_equity_from_prices(prices[MarketDataCol.low_bid], prices[MarketDataCol.low_ask], cash, shares)
        equity_close = self._calculate_equity_from_prices(prices[MarketDataCol.close_bid], prices[MarketDataCol.close_ask], cash, shares)
        return equity_open, equity_high, equity_low, equity_close
    
    def _calculate_equity_from_prices(self, bid_price, ask_price, cash: float, shares: float) -> float:
        """
        Calculates the equity based on current cash, shares and prices.
        """
        current_price = bid_price if shares >= 0 else ask_price
        return cash + (shares * bid_price)

    def _get_observation(self): 
        """
        Returns the current observation of the environment.
        The observation is a combination of market features and state features.
        """
        market_features: np.ndarray = self.market_features[self.current_step]
        state_features = self.agent_feature_engineer.run(self.agent_data, self.current_step)
        observation = np.concatenate((market_features, state_features), axis=0)
        
        return observation

    def _get_continuous_action(self, action) -> float:
        """
        Converts discrete actions to continuous actions.
        Action remains the same if it is already continuous.
        """
        if self.n_actions > 0:
            action = (action - self.n_actions) / self.n_actions
        return action

    def _execute_action(self, action, prev_action, current_data, current_cash, current_shares) -> tuple[float, float]:
        """
        Determines the target position based on the agent's action (percentage of equity)
        and executes trades by calling buy or sell instrument methods.
        """

        # Jitter Mitigation: action hasn't changed significantly, do nothing.
        if prev_action is not None and abs(action - prev_action) < 1e-5:
            logging.debug(f"Step {self.current_step}: Raw action {action} close to previous {prev_action}. No trade due to jitter mitigation.")
            return current_cash, current_shares

        # Calculate current equity (mark-to-market)
        # Equity calculation uses bid for valuing long positions, ask for valuing short positions (cost to close)
        current_equity = self._calculate_equity_from_prices(current_data[MarketDataCol.open_bid], current_data[MarketDataCol.open_ask], current_cash, current_shares)
        
        # Ensure equity is not negative for target calculation, as it might lead to reversed logic if agent is bankrupt
        # Although termination on equity <= 0 should prevent this during active trading.
        if current_equity <= 0:
            raise ValueError(f"current_equity should be greater than zero, was {current_equity:.2f}")

        # Determine target position value in currency
        target_position_value = action * current_equity
        
        # Determine target number of shares based on target value
        if abs(action) < 1e-6: # Target is to be flat
            target_num_shares = 0.0
        elif action > 0: # Target is LONG
            current_open_ask = current_data[MarketDataCol.open_ask]
            if current_open_ask <= 1e-6:
                raise ValueError(f"Step {self.current_step}: Invalid ask price ({current_open_ask}) for calculating long target shares.")
            target_num_shares = target_position_value / current_open_ask
        else: # Target is SHORT (target_percentage < 0)
            current_open_bid = current_data[MarketDataCol.open_bid]
            if current_open_bid <= 1e-6:
                raise ValueError(f"Step {self.current_step}: Invalid bid price ({current_open_bid}) for calculating short target shares.")
            target_num_shares = target_position_value / current_open_bid

        # Determine the change in shares needed
        shares_to_trade = target_num_shares - current_shares

        if abs(shares_to_trade) < 1e-6:
            logging.debug(f"Step {self.current_step}: Target shares effectively same as current. No trade execution needed.")
            return current_cash, current_shares

        if shares_to_trade > 0: # Need to buy
            new_cash, new_shares = self._buy_instrument(
                shares_to_buy_abs=shares_to_trade, 
                ask_price=current_data[MarketDataCol.open_ask],
                current_cash=current_cash, 
                current_shares=current_shares
            )
        else: # Need to sell
            new_cash, new_shares = self._sell_instrument(
                shares_to_sell_abs=abs(shares_to_trade),
                bid_price=current_data[MarketDataCol.open_bid],
                current_cash=current_cash, 
                current_shares=current_shares
            )
            
        return new_cash, new_shares
    
    def _buy_instrument(self, shares_to_buy_abs: float, ask_price: float, current_cash: float, current_shares: float) -> tuple[float, float]:
        """
        Executes a buy order for a specified absolute number of shares.
        """
        if shares_to_buy_abs < 1e-6:  # Effectively zero shares, no action needed
            return current_cash, current_shares
        if ask_price <= 1e-6:  # Invalid price
            logging.warning(f"Step {self.current_step}: Attempted to buy with invalid price: {ask_price}")
            return current_cash, current_shares # No transaction

        # Calculate cost including commission
        cost_per_share_incl_commission = ask_price * (1 + self.transaction_cost_pct)

        affordable_shares = max(0.0, current_cash / cost_per_share_incl_commission)
        
        actual_shares_bought = min(shares_to_buy_abs, affordable_shares)

        cost_before_commission = actual_shares_bought * ask_price
        total_commission = cost_before_commission * self.transaction_cost_pct

        updated_cash = current_cash - (cost_before_commission + total_commission)
        updated_shares = current_shares + actual_shares_bought

        return updated_cash, updated_shares

    def _sell_instrument(self, shares_to_sell_abs: float, bid_price: float, current_cash: float, current_shares: float) -> tuple[float, float]:
        """
        Executes a sell order, potentially closing a long position and/or initiating/increasing a short position.
        Shorting is limited by a margin rule: total absolute short position value * 2 <= available cash.
        """
        if shares_to_sell_abs < 1e-6:  # Effectively zero shares
            return current_cash, current_shares
        if bid_price <= 1e-6:  # Invalid price
            logging.warning(f"Step {self.current_step}: Attempted to sell with invalid price: {bid_price}")
            return current_cash, current_shares

        # Part 1: Handle closing of any existing long position
        if current_shares > 0: # If currently long
            shares_sold_from_long = min(shares_to_sell_abs, current_shares)
            proceeds = shares_sold_from_long * bid_price
            commission = proceeds * self.transaction_cost_pct
            current_cash += (proceeds - commission)
            current_shares -= shares_sold_from_long
            shares_to_sell_abs -= shares_sold_from_long

        # Part 2: Handle shares intended for shorting (initiating or increasing short)
        # These are the shares remaining from the original sell order after closing any long position.

        # Apply margin rule: max_total_abs_short_shares = available_cash / (2 * price)
        # available_cash for margin is updated_cash (cash after closing any long positions).
        max_total_abs_short_position_allowed = max(0.0, current_cash / (2 * bid_price))

        # current_abs_short_position is abs(updated_shares) because updated_shares is the state *after* closing longs.
        # At this point, updated_shares <= 0 if we are considering shorting.
        current_abs_short_portion = abs(current_shares) # Should be 0 if we just closed a long, or abs of existing short.

        # How many *more* shares can we short without violating the total limit?
        allowable_additional_short_shares = max(0.0, max_total_abs_short_position_allowed - current_abs_short_portion)

        actual_shares_shorted = min(shares_to_sell_abs, allowable_additional_short_shares)

        proceeds = actual_shares_shorted * bid_price
        commission = proceeds * self.transaction_cost_pct
        current_cash += (proceeds - commission) # Cash increases from short sale
        current_shares -= actual_shares_shorted   # Makes share count more negative

        return current_cash, current_shares
