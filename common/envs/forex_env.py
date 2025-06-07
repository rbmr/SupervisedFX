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
from common.data.utils import shuffle
from common.envs.trade import execute_trade, calculate_ohlc_equity, calculate_equity
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
                 allow_short: bool = True,
                 allow_long: bool = True,
                 custom_reward_function: Callable[['ForexEnv'], float] | None = None,
                 shuffled: bool = False,
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
            0.0                   # action (no action at start)
        )
        assert self.market_data.shape == (self.total_steps, len(MarketDataCol))
        assert self.market_features.shape == (self.total_steps, len(self.market_feature_names))
        assert self.agent_data.shape == (self.total_steps, len(AgentDataCol))

        # Shuffle data if required
        if shuffled:
            self.market_data, self.market_features = shuffle(self.market_data, self.market_features)

        # Action space
        self.n_actions = n_actions
        self.allow_short = allow_short
        self.allow_long = allow_long
        if self.n_actions == 0:
            logging.info(f"n_actions is zero, using continuous action space")

            low = -1.0 if allow_short else 0.0
            high = 1.0 if allow_long else 0.0

            self.action_space = spaces.Box(low=low, high=high, shape=(), dtype=np.float32)
        else:
            logging.info(f"n_actions is larger than zero, using discrete action space with {n_actions} actions for buys, {n_actions} for sells, and 1 action for no_participation.")

            if not allow_short and not allow_long:
                raise ValueError("If n_actions > 0, at least one of allow_short or allow_long must be True.")
            if allow_short and allow_long:
                self.action_space = spaces.Discrete(2 * self.n_actions + 1)
            else:
                self.action_space = spaces.Discrete(self.n_actions + 1)

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
        allow_short: bool = True,
        allow_long: bool = True,
        custom_reward_function: Optional[Callable[['ForexEnv'], float]] = None,
        shuffled = False,
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

        # Retrieve market data and features.
        market_data_df = forex_candle_data.df.copy(deep=True)
        market_features_df = market_feature_engineer.run(market_data_df, remove_original_columns=True)

        # log the amount of features, and the specific column names
        logging.info(f"Market data contains {len(market_data_df)} rows and {len(market_data_df.columns)} columns.")
        logging.info(f"Market features contain {len(market_features_df)} rows and {len(market_features_df.columns)} columns: {market_features_df.columns.tolist()}.")
        logging.info(f"Agent feature engineer contains {agent_feature_engineer.num_of_features()} features: {agent_feature_engineer.get_features()}.")

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
            allow_short=allow_short,
            allow_long=allow_long,
            custom_reward_function=custom_reward_function,
            shuffled=shuffled,
        )

        # Create evaluation environment
        eval_env = ForexEnv(
            market_data_df=eval_market_data,
            market_feature_df=eval_market_features,
            agent_feature_engineer=agent_feature_engineer,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            n_actions=n_actions,
            allow_short=allow_short,
            allow_long=allow_long,
            custom_reward_function=custom_reward_function,
            shuffled=shuffled,
        )

        return train_env, eval_env

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environment state
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):

        # Standardize action
        target_exposure = self._get_target_exposure(action)

        # Perform step
        self.current_step += 1

        # Perform action
        current_data = self.market_data[self.current_step, :]
        current_cash = self.agent_data[self.current_step - 1, AgentDataCol.cash]
        current_shares = self.agent_data[self.current_step - 1, AgentDataCol.shares]
        new_cash, new_shares = execute_trade(target_exposure, current_data, current_cash, current_shares, self.transaction_cost_pct) # type: ignore
        equity_open, equity_high, equity_low, equity_close = calculate_ohlc_equity(current_data, new_cash, new_shares)
        self.agent_data[self.current_step, :] = (new_cash, new_shares, equity_open, equity_high, equity_low, equity_close, target_exposure)

        # calculate reward
        reward = self._get_reward()

        # Determine done
        terminated = False
        truncated = False
        if equity_close <= 0:
            terminated = True
            logging.warning(f"Step {self.current_step}: Agent ruined. Equity: {equity_close}.")
        if self.current_step >= self.total_steps - 1:
            truncated = True

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

    def _get_observation(self):
        """
        Returns the current observation of the environment.
        The observation is a combination of market features and state features.
        """
        market_features = self.market_features[self.current_step]
        state_features = self.agent_feature_engineer.run(self.agent_data, self.current_step)
        observation = np.concatenate((market_features, state_features), axis=0)
        
        return observation

    def _get_target_exposure(self, action: np.ndarray | np.generic) -> float:
        """
        Standardizes actions by converting them to the target exposure.
        """
        action = action.item()
        if self.n_actions > 0:
            if self.allow_long and self.allow_short:
                action = (action - self.n_actions) / self.n_actions
            elif self.allow_long:
                action = action / self.n_actions
            else: # self.allow_short
                action = action / self.n_actions
                action -= 1.0  # Shift to [-1, 0] for shorting
                
        return action