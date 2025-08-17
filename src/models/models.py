from abc import ABC, abstractmethod

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from src.constants import Account
from src.trade.dp import DPTable
from src.trade.env import TradeEnv
from src.trade.trade import interpolate

class CustomModel(ABC):

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass

class SB3ModelWrapper(CustomModel):

    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        actions, _ = self.model.predict(observation)
        return actions

class PerfectModel(CustomModel):

    def __init__(self, env: TradeEnv):
        self.env = env
        dp_table = get_dp_table_from_env(env)
        self.policy = dp_table.policy_table
        self.exposures = np.linspace(-1.0, 1.0, dp_table.n_exposures)

    def predict(self, _: np.ndarray) -> np.ndarray:
        t = self.env.t
        prev_exposure = self.env.account[t, Account.CLOSE_EXPOSURE]
        return interpolate(prev_exposure, self.exposures, self.policy[t+1, :])

class RandomModel(CustomModel):
    def __init__(self, env: TradeEnv):
        self.env = env

    def predict(self, _: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()

def get_dp_table_from_env(env: TradeEnv):
    return DPTable.get(env.prices, env.commission_pct, env.n_actions or 15, 15)



