import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.constants import Price, Account, COMMISSION_PCT
from src.trade.trade import trade

logger = logging.getLogger(__name__)

class TradeEnv(gym.Env):

    def __init__(self,
                 prices: np.ndarray,
                 features: np.ndarray,
                 feature_names: list[str],
                 commission_pct: float = COMMISSION_PCT,
                 initial_capital: float = 1.0,
                 n_actions: int = 0,
                 t_start: int = 0
                 ):
        super(TradeEnv, self).__init__()

        # Environment parameters
        assert initial_capital > 0, "initial capital must be positive"
        assert 0 <= commission_pct <= 1, "commission_pct must be between 0 and 1"

        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

        # Static state
        assert prices.ndim == features.ndim == 2, "prices and features must be two dimensional"
        assert prices.shape[0] == features.shape[0], "prices and features must have same number of rows"
        assert prices.shape[1] == len(Price), "prices has an unexpected number of columns"
        assert features.shape[1] == len(feature_names), f"number of feature columns and feature names do not match"

        self.prices = prices
        self.features = features
        self.feature_names = feature_names

        # Step counter
        assert t_start >= 0, "t_start must be non-negative"
        self.t_start = t_start
        self.t = self.t_start # the current step index
        self.episode_len = len(self.prices) # the #steps in the episode

        # Dynamic state
        self.account = np.zeros(shape = (self.episode_len, len(Account)), dtype=np.float64)
        self.account[self.t, Account.CASH] = self.initial_capital
        self.account[self.t, Account.SHARES] = 0.0
        self.account[self.t, Account.CLOSE_EQUITY] = self.initial_capital
        self.account[self.t, Account.CLOSE_EXPOSURE] = 0.0
        self.account[self.t, Account.CLOSE_PVAL] = 0.0
        self.account[self.t, Account.CLOSE_LEQUITY] = 0.0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.features.shape[1] + 1,), dtype=np.float64)

        # Action space
        assert n_actions >= 0, "n_actions must be non-negative"
        if n_actions > 0:
            self.action_space = spaces.Discrete(n_actions)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.actions = np.linspace(-1.0, 1.0, n_actions)
        self.n_actions = n_actions

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.t_start
        return self._get_obs(), { }

    def _get_obs(self):
        current_features = self.features[self.t, :]
        current_exposure = self.account[self.t, Account.CLOSE_EXPOSURE]
        return np.concat([current_features, [current_exposure,]])

    def step(self, action):
        self.t += 1

        # Retrieve relevant state and price information
        prev_cash = self.account[self.t-1, Account.CASH]
        prev_shares = self.account[self.t-1, Account.SHARES]
        prev_exposure = self.account[self.t-1, Account.CLOSE_EXPOSURE]
        decision_bid = self.prices[self.t-1, Price.CLOSE_BID]
        decision_ask = self.prices[self.t-1, Price.CLOSE_ASK]
        exec_bid = self.prices[self.t, Price.EXEC_BID]
        exec_ask = self.prices[self.t, Price.EXEC_ASK]
        close_bid = self.prices[self.t, Price.CLOSE_BID]
        close_ask = self.prices[self.t, Price.CLOSE_ASK]

        # Trade
        cash, shares, pval, equity, exposure, log_equity = trade(
            prev_exposure, action, prev_cash, prev_shares,
            decision_bid, decision_ask, exec_bid, exec_ask,
            close_bid, close_ask, self.commission_pct
        )

        # Store updated account state
        self.account[self.t, Account.CASH] = cash
        self.account[self.t, Account.SHARES] = shares
        self.account[self.t, Account.CLOSE_EQUITY] = equity
        self.account[self.t, Account.CLOSE_EXPOSURE] = exposure
        self.account[self.t, Account.CLOSE_PVAL] = pval
        self.account[self.t, Account.CLOSE_LEQUITY] = log_equity

        # Determine reward
        reward = log_equity - self.account[self.t-1, Account.CLOSE_LEQUITY]

        # Determine done
        terminated = equity.item() <= 0
        truncated = self.t >= self.episode_len - 1 # If the current step is the last step.

        # Determine info
        info = {}
        if terminated or truncated:
            # Episode is ending, put relevant final info here
            end = self.t + 1
            prices = self.prices[self.t_start:end]
            account = self.account[self.t_start:end]
            features = self.features[self.t_start:end]

            prices_df = pd.DataFrame(prices, columns=Price.names)
            account_df = pd.DataFrame(account, columns=Account.names)
            features_df = pd.DataFrame(features, columns=self.feature_names)

            info['prices'] = prices_df
            info['account'] = account_df
            info['features'] = features_df

            logger.info(f"Finished with equity {equity}")

        return self._get_obs(), reward.item(), terminated, truncated, info