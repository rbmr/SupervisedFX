import logging
from dataclasses import dataclass
from typing import Callable, Optional, Self

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from common.constants import *
from common.data.data import ForexCandleData
from common.data.feature_engineer import FeatureEngineer
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer
from common.envs.trade import calculate_ohlc_equity, execute_trade, calculate_equity
from common.scripts import find_first_row_with_nan, find_first_row_without_nan

class ActionConfig:
    """
    Class responsible for preparing and validating action info for ForexEnv.
    """

    __slots__ = ("n_actions", "actions", "action_space")

    def __init__(self, n: int = 3, low: float = -1.0, high: float = 1.0):

        assert n >= 0, f"n must be >= 0. If n == 0, then the action space is continuous."
        assert low <= high, f"low ({low}) must be less than or equal to high ({high})."
        assert low >= -1.0 and high <= 1.0, f"actions must be within [-1, 1], was [{low}, {high}]."

        self.n_actions = n
        self.actions = np.linspace(low, high, n)
        if self.n_actions == 0:
            logging.info(f"n_actions is zero, using continuous action space, over the range [{low}, {high}]")
            self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        else:
            logging.info(f"n_actions is larger than zero, using discrete actions {self.actions}")
            self.action_space = spaces.Discrete(self.n_actions)

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
class ObsConfig:
    """
    Immutable configuration for a single observation component.
    """
    name: str
    fe: Optional[FeatureEngineer] = None
    sfe: Optional[StepwiseFeatureEngineer] = None
    window: int = 1

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"window_size must be > 0, was {self.window}")
        if self.fe is None and self.sfe is None:
            raise ValueError("At least one of feature_engineer or stepwise_feature_engineer must be provided")
        if self.window > 1 and self.sfe is not None:
            raise ValueError("stepwise_feature_engineer cannot be used with window_size > 1")

@dataclass(frozen=True)
class EnvObs:

    features_data: np.ndarray
    feature_names: list[str]
    sfe: StepwiseFeatureEngineer
    name: str
    window: int

    @classmethod
    def from_config(cls, config: ObsConfig, market_data: pd.DataFrame):
        """Create EnvObs from ObsConfig and market data."""
        if config.fe is not None:
            features_df = config.fe.run(market_data.copy(deep=True), remove_original_columns=True)
            features_data = features_df.to_numpy(dtype=np.float32)
            feature_names = features_df.columns.tolist()
        else:
            # No feature engineer, create empty features
            features_data = np.empty((len(market_data), 0), dtype=np.float32)
            feature_names = []

        if config.sfe is not None:
            sfe_n_features = config.sfe.num_of_features()
            sfe_features = config.sfe.get_features()
        else:
            sfe_n_features = 0
            sfe_features = []

        logging.info(f"'{config.name}' features ({len(features_data)}, {len(feature_names)}): {feature_names})")
        logging.info(f"'{config.name}' stepwise features ({sfe_n_features}): {sfe_features})")

        return cls(features_data, feature_names, config.sfe, config.name, config.window)

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

class DataConfig:
    """
    Class responsible for preparing and validating the data for ForexEnv.
    """

    __slots__ = ("market_data", "observations")

    @classmethod
    def from_splits(cls,
                    forex_candle_data: ForexCandleData,
                    split_pcts: list[float],
                    obs_configs: list[ObsConfig]):
        # --- VALIDATION ---
        assert abs(sum(split_pcts) - 1.0) < 1e-9, f"split_ratios must sum to 1.0, but sum is {sum(split_pcts)}."
        assert all(r > 0 for r in split_pcts), "All split_ratios must be positive."

        # Retrieve market data.
        market_data = forex_candle_data.df.copy(deep=True)
        logging.info(f"Market data ({len(market_data)}, {len(market_data.columns)}): {market_data.columns.tolist()}.")
        # Retrieve observations
        observations = [EnvObs.from_config(config, market_data) for config in obs_configs]

        # --- DATA SPLITTING LOGIC ---
        data_configs = []
        total_len = len(market_data)
        
        # Calculate the absolute indices for splitting the data
        split_indices = [0] + [int(total_len * sum(split_pcts[:i+1])) for i in range(len(split_pcts))]
        
        # Ensure the last index goes to the very end of the dataframe
        split_indices[-1] = total_len
        
        # Create a DataConfig for each data slice
        for i in range(len(split_pcts)):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Slice market data
            split_market_data = market_data.iloc[start_idx:end_idx]

            # Slice observation data
            split_env_obs = []
            for env_ob in observations:
                split_env_obs.append(EnvObs(
                    features_data=env_ob.features_data[start_idx:end_idx],
                    feature_names=env_ob.feature_names,
                    sfe=env_ob.sfe,
                    name=env_ob.name,
                    window=env_ob.window
                ))
            
            # Create and store the DataConfig for this split
            data_configs.append(cls(
                market_data=split_market_data,
                observations=split_env_obs
            ))
            
        return data_configs


    def __init__(self, market_data: pd.DataFrame, observations: list[EnvObs]):

        # Validate input
        expected_columns = set(MarketDataCol.all_names())
        actual_columns = set(market_data.columns)
        assert expected_columns.issubset(actual_columns), f"market_data is missing columns: {expected_columns - actual_columns}."

        # Find initial NaNs in market data.
        market_data = market_data.copy(deep=True)[MarketDataCol.all_names()]
        start_index = find_first_row_without_nan(market_data)

        # Find initial NaNs in features.
        for env_ob in observations:
            if len(env_ob.feature_names) > 0:
                temp_df = pd.DataFrame(env_ob.features_data, columns=env_ob.feature_names)
                start_index = max(start_index, find_first_row_without_nan(temp_df))

        # Clean and market data.
        market_data = market_data.iloc[start_index:]
        market_data.reset_index(drop=True, inplace=True)
        assert not market_data.isna().any().any(), f"market_data contains NaN values at index {find_first_row_with_nan(market_data)}"

        # Clean and validate features.
        final_env_obs = []
        for env_ob in observations:
            adjusted_features_data = env_ob.features_data[start_index:]
            temp_df = pd.DataFrame(adjusted_features_data, columns=env_ob.feature_names)
            assert not temp_df.isna().any().any(), f"market_data contains NaN values at index {find_first_row_with_nan(market_data)}"
            assert len(market_data) == len(adjusted_features_data), f"len market_data ({len(market_data)}) != len features for '{env_ob.name}' ({len(adjusted_features_data)})"
            final_env_obs.append(EnvObs(adjusted_features_data, env_ob.feature_names, env_ob.sfe, env_ob.name, env_ob.window))

        # Set parameters
        self.market_data = market_data
        self.observations = final_env_obs

class ForexEnv(gym.Env):

    @classmethod
    def create_split_envs(cls,
                               split_pcts: list[float],
                               forex_candle_data: ForexCandleData,
                               market_feature_engineer: FeatureEngineer,
                               agent_feature_engineer: StepwiseFeatureEngineer,
                               initial_capital: float = 10000.0,
                               transaction_cost_pct: float = 0.0,
                               n_actions: int = 3,
                               action_low: float = -1.0,
                               action_high: float = 1.0,
                               custom_reward_function: Optional[Callable[[Self], float]] = None
                            ) -> tuple[Self, Self]:
        obs_configs = [ObsConfig(
            name = 'market_features',
            fe = market_feature_engineer,
            sfe = agent_feature_engineer,
            window = 1
        )]
        env_config = EnvConfig(
            initial_capital = initial_capital,
            transaction_cost_pct = transaction_cost_pct,
            reward_function = custom_reward_function,
        )
        action_config = ActionConfig(
            n = n_actions,
            low = action_low,
            high = action_high,
        )
        data_configs = DataConfig.from_splits(
            forex_candle_data=forex_candle_data,
            split_pcts=split_pcts,
            obs_configs=obs_configs,
        )

        return [cls(action_config, env_config, data_config) for data_config in data_configs]


    def __init__(self, action_config: ActionConfig, env_config: EnvConfig, data_config: DataConfig):
        super(ForexEnv, self).__init__()

        # Environment parameters
        self.initial_capital = env_config.initial_capital
        self.transaction_cost_pct = env_config.transaction_cost_pct
        self.custom_reward_fn = env_config.reward_function
        self.observations = data_config.observations

        # Market data
        self.market_data = data_config.market_data.to_numpy(dtype=np.float32)

        # Step counter
        self.n_steps = 0 # the current step index
        self.episode_len = len(self.market_data) - 1 # the total #steps in an episode
        self.data_len = len(self.market_data)

        # Agent data
        self.agent_data = np.zeros(shape = (self.data_len, len(AgentDataCol.all_names())), dtype=np.float32)
        self.agent_data[0, :] = (
            self.initial_capital, # cash
            0.0,                  # shares
            self.initial_capital, # equity_open
            self.initial_capital, # equity_high
            self.initial_capital, # equity_low
            self.initial_capital, # equity_close
            0.0,                  # action (no action at start)
            self.initial_capital, # pre_action_equity
        )
        self.agent_data[1, AgentDataCol.pre_action_equity] = self.initial_capital

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
        self.n_steps = 0

        return self._get_observation(), {}

    def step(self, action):

        # Standardize action
        target_exposure = self._get_target_exposure(action)

        # Increment step
        self.n_steps += 1

        # Perform action
        prev_cash = self.agent_data[self.n_steps - 1, AgentDataCol.cash]
        prev_shares = self.agent_data[self.n_steps - 1, AgentDataCol.shares]
        open_ask = self.market_data[self.n_steps, MarketDataCol.open_ask]
        open_bid = self.market_data[self.n_steps, MarketDataCol.open_bid]
        cash, shares = execute_trade(target_exposure, open_bid, open_ask, prev_cash, prev_shares, self.transaction_cost_pct) # type: ignore

        # Update agent data
        equity_ohlc = calculate_ohlc_equity(self.market_data[self.n_steps], cash, shares)
        self.agent_data[self.n_steps, :-1] = (cash, shares, *equity_ohlc, target_exposure)

        # Compute pre_action_equity of next timeframe
        next_open_ask = self.market_data[self.n_steps + 1, MarketDataCol.open_ask]
        next_open_bid = self.market_data[self.n_steps + 1, MarketDataCol.open_bid]
        next_equity_open = calculate_equity(next_open_bid, next_open_ask, cash, shares)
        self.agent_data[self.n_steps + 1, AgentDataCol.pre_action_equity] = next_equity_open

        # Determine done
        terminated = False
        truncated = False
        equity_close = equity_ohlc[3]
        if equity_close <= 0:
            terminated = True
            logging.warning(f"Step {self.n_steps}: Agent ruined. Equity: {equity_close}.")
        if self.n_steps >= self.episode_len - 1: # We are at or after the last step.
            truncated = True

        # Determine info dict
        info = {}
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            number_of_steps = self.n_steps + 1
            market_data = self.market_data[:number_of_steps]
            agent_data = self.agent_data[:number_of_steps]

            market_data_df = pd.DataFrame(market_data, columns=MarketDataCol.all_names())
            agent_data_df = pd.DataFrame(agent_data, columns=AgentDataCol.all_names())

            info['market_data'] = market_data_df
            info['agent_data'] = agent_data_df

            for obs in self.observations:
                info[obs.name] = pd.DataFrame(obs.features_data[:number_of_steps], columns=obs.feature_names)

        return self._get_observation(), self._get_reward(), terminated, truncated, info

    def _get_reward(self):
        """
        Calculates the reward based on the current equity.
        Uses a custom reward function if provided, otherwise defaults to equity change.
        """
        if self.custom_reward_fn is not None:
            return self.custom_reward_fn(self)
        prev_equity = self.agent_data[self.n_steps, AgentDataCol.pre_action_equity]
        next_equity = self.agent_data[self.n_steps + 1, AgentDataCol.pre_action_equity]
        return next_equity - prev_equity

    def _get_target_exposure(self, action: np.ndarray | np.generic) -> float:
        """
        Standardizes actions by converting them to the target exposure.
        """
        action = np.asarray(action).item() # ensure native python scalars
        if self.n_actions == 0: # Actions are already continuous
            return action
        return self.actions[int(action)] # type: ignore

    def _get_default_obs(self):
        market_features = self.market_data[self.n_steps]
        agent_features = self.agent_data[self.n_steps]
        return np.concatenate((market_features, agent_features))

    @staticmethod
    def _create_default_obs_space():
        # Default case: concatenate market data and agent data (raw, no processing)
        num_market_features = len(MarketDataCol)
        num_agent_features = len(AgentDataCol)
        total_features = num_market_features + num_agent_features
        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)

    def _get_single_obs(self):
        return self.observations[0].get_observation(self.n_steps, self.agent_data)

    def _create_single_obs_space(self):
        # Single observation: return Box space
        return self.observations[0].get_observation_space()

    def _get_multi_obs(self):
        return {
            obs.name: obs.get_observation(self.n_steps, self.agent_data)
            for obs in self.observations
        }

    def _create_multi_obs_space(self):
        # Multiple observations: return Dict space
        return spaces.Dict({
            obs.name: obs.get_observation_space()
            for obs in self.observations
        })