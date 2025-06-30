import logging
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import tensorflow as tf

from src.constants import *
from src.data.feature_engineer import FeatureEngineer
from src.data.stepwise_feature_engineer import StepwiseFeatureEngineer
from src.envs.trade import execute_trade, calculate_equity
from src.scripts import contains_nan_or_inf


@dataclass(frozen=True)
class ActionConfig:
    """
    Class responsible for preparing and validating action info for ForexEnv.
    """
    n: int = 3
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self):
        assert self.n >= 0, f"n must be >= 0. If n == 0, then the action space is continuous."
        assert self.low <= self.high, f"low ({self.low}) must be less than or equal to high ({self.high})."
        assert self.low >= -1.0 and self.high <= 1.0, f"actions must be within [-1, 1], was [{self.low}, {self.high}]."
        if self.n_actions == 0:
            logging.info(f"n_actions is zero, using continuous action space, over the range [{self.low}, {self.high}]")
        else:
            logging.info(f"n_actions is larger than zero, using discrete actions {self.actions}")

    @property
    def n_actions(self):
        return self.n

    @property
    def actions(self):
        return np.linspace(self.low, self.high, self.n)

    @property
    def action_space(self):
        if self.n == 0:
            return spaces.Box(low=self.low, high=self.high, shape=(1,), dtype=np.float32)
        return spaces.Discrete(self.n)

@dataclass(frozen=True)
class EnvConfig:
    """
    Class responsible for validating and storing the general parameters for ForexEnv.
    """

    initial_capital: float = 10_000.0
    transaction_cost_pct: float = 0.0
    reward_function: Optional[Callable[[gym.Env], float]] = None

    def __post_init__(self):
        if self.initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, was {self.initial_capital}.")
        if self.transaction_cost_pct < 0.0 or self.transaction_cost_pct > 1.0:
            raise ValueError(f"transaction_cost_pct must in [0.0, 1.0], was {self.transaction_cost_pct}.")

@dataclass(frozen=True)
class EnvObs:

    features_data: np.ndarray
    feature_names: list[str]
    sfe: StepwiseFeatureEngineer
    name: str
    window: int

    def get_observation_space(self) -> spaces.Space:
        """Get the observation space for this observation."""
        num_features = self.features_data.shape[1]
        if self.sfe is not None:
            num_features += self.sfe.num_of_features()
        if self.window == 1:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1, num_features, self.window), dtype=np.float32)

    def get_observation(self, n_steps: int, agent_data: np.ndarray) -> np.ndarray:
        """Get observation for the current step."""

        # Get static features
        static_features = None
        if len(self.feature_names) > 0:
            if self.window == 1:
                static_features = self.features_data[n_steps]
            else:
                start_idx = max(0, n_steps - self.window + 1)
                end_idx = n_steps + 1
                static_features = self.features_data[start_idx:end_idx]
                if len(static_features) < self.window:
                    pad_size = self.window - len(static_features)
                    padding = np.zeros((pad_size, static_features.shape[1]), dtype=np.float32)
                    static_features = np.vstack((padding, static_features))
                static_features = static_features.transpose()[np.newaxis, ...]

        # Get stepwise features
        stepwise_features = None
        if self.sfe is not None:
            stepwise_features = self.sfe.run(agent_data, n_steps)

        # Combine features
        if static_features is None and stepwise_features is None:
            return np.empty((1,), dtype=np.float32)
        elif static_features is None:
            return stepwise_features
        elif stepwise_features is None:
            return static_features
        return np.concatenate((static_features, stepwise_features))

@dataclass(frozen=True)
class DataConfig:
    """
    Class responsible for preparing and validating the data for ForexEnv.
    """
    market_data: np.ndarray
    observations: list[EnvObs]

    def __post_init__(self):
        data_len = self.market_data.shape[0]
        assert not contains_nan_or_inf(self.market_data)
        for obs in self.observations:
            assert obs.features_data.shape[0] == data_len
            assert not contains_nan_or_inf(obs.features_data)

class ForexEnv(gym.Env):

    def __init__(self, action_config: ActionConfig, env_config: EnvConfig, data_config: DataConfig):
        super(ForexEnv, self).__init__()

        # Environment parameters
        self.initial_capital = env_config.initial_capital
        self.transaction_cost_pct = env_config.transaction_cost_pct
        self.custom_reward_fn = env_config.reward_function
        self.observations = data_config.observations

        # Market data
        self.market_data = data_config.market_data

        # Step counter
        self.current_step = 0 # the current step index
        self.episode_len = len(self.market_data) - 1 # the max #steps in an episode
        self.data_len = len(self.market_data)

        # Agent data
        self.agent_data = np.zeros(shape = (self.data_len, len(AgentDataCol)), dtype=np.float32)
        self.agent_data[0, :] = (
            self.initial_capital, # cash
            0.0,                  # shares
            self.initial_capital, # eot_equity
        )

        assert self.market_data.shape == (self.data_len, len(MarketDataCol))
        assert self.agent_data.shape == (self.data_len, len(AgentDataCol))

        # Define action space
        self.n_actions = action_config.n_actions
        self.actions = action_config.actions
        self.action_space = action_config.action_space

        # Define observation space
        if len(self.observations) == 0:
            self.observation_space = self._create_default_obs_space()
            self._get_observation = self._get_default_obs
        elif len(self.observations) == 1:
            self.observation_space = self._create_single_obs_space()
            self._get_observation = self._get_single_obs
        else:
            self.observation_space = self._create_multi_obs_space()
            self._get_observation = self._get_multi_obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environment state
        # market data and features are static,
        # starting agent data is the same for each episode,
        # therefore, only the index needs to be reset.
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):

        # Increment step
        self.current_step += 1

        # Execute trade at open of current timestep t
        target_exposure = tf.constant([[self._get_target_exposure(action)]], tf.float32)
        prev_cash = tf.constant([[self.agent_data[self.current_step - 1, AgentDataCol.cash]]], tf.float32)
        prev_shares = tf.constant([[self.agent_data[self.current_step - 1, AgentDataCol.shares]]], tf.float32)
        open_ask = tf.constant([[self.market_data[self.current_step, MarketDataCol.open_ask]]], tf.float32)
        open_bid = tf.constant([[self.market_data[self.current_step, MarketDataCol.open_bid]]], tf.float32)
        cash, shares = execute_trade(target_exposure, open_bid, open_ask, prev_cash, prev_shares, self.transaction_cost_pct)

        # Compute equity at the open of the next step before the trade is executed
        next_open_ask = tf.constant([[self.market_data[self.current_step + 1, MarketDataCol.open_ask]]], tf.float32)
        next_open_bid = tf.constant([[self.market_data[self.current_step + 1, MarketDataCol.open_bid]]], tf.float32)
        eot_equity = calculate_equity(next_open_bid, next_open_ask, cash, shares)

        # Store results
        self.agent_data[self.current_step, :] = (cash.numpy().item(), shares.numpy().item(), eot_equity.numpy().item())

        # Determine done
        eot_equity = eot_equity.numpy().item()
        terminated = eot_equity <= 0
        truncated = self.current_step >= self.episode_len - 1

        # Determine info
        info = {}
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            number_of_steps = self.current_step + 1
            market_data = self.market_data[:number_of_steps]
            agent_data = self.agent_data[:number_of_steps]

            market_data_df = pd.DataFrame(market_data, columns=MarketDataCol.all_names())
            agent_data_df = pd.DataFrame(agent_data, columns=AgentDataCol.all_names())

            info['market_data'] = market_data_df
            info['agent_data'] = agent_data_df

            for obs in self.observations:
                info[obs.name] = pd.DataFrame(obs.features_data[:number_of_steps], columns=obs.feature_names)

            logging.info(f"Finished with equity {eot_equity}")

        return self._get_observation(), self._get_reward(), terminated, truncated, info

    def _get_reward(self):
        """
        Calculates the reward based on the current equity.
        Uses a custom reward function if provided, otherwise defaults to equity change.
        """
        if self.custom_reward_fn is not None:
            return self.custom_reward_fn(self)
        prev_equity = self.agent_data[self.current_step - 1, AgentDataCol.eot_equity]
        curr_equity = self.agent_data[self.current_step, AgentDataCol.eot_equity]
        return curr_equity - prev_equity

    def _get_target_exposure(self, action: np.ndarray | np.generic) -> float:
        """
        Standardizes actions by converting them to the target exposure.
        """
        action = np.asarray(action).item() # ensure native python scalars
        if self.n_actions == 0: # Actions are already continuous
            return action
        return self.actions[int(action)] # type: ignore

    def _get_default_obs(self):
        market_features = self.market_data[self.current_step]
        agent_features = self.agent_data[self.current_step]
        return np.concatenate((market_features, agent_features))

    @staticmethod
    def _create_default_obs_space():
        # Default case: concatenate market data and agent data (raw, no processing)
        num_market_features = len(MarketDataCol)
        num_agent_features = len(AgentDataCol)
        total_features = num_market_features + num_agent_features
        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)

    def _get_single_obs(self):
        return self.observations[0].get_observation(self.current_step, self.agent_data)

    def _create_single_obs_space(self):
        # Single observation: return Box space
        return self.observations[0].get_observation_space()

    def _get_multi_obs(self):
        return {
            obs.name: obs.get_observation(self.current_step, self.agent_data)
            for obs in self.observations
        }

    def _create_multi_obs_space(self):
        # Multiple observations: return Dict space
        return spaces.Dict({
            obs.name: obs.get_observation_space()
            for obs in self.observations
        })