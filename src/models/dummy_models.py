import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

from src.constants import AgentDataCol
from src.envs.dp import get_dp_table_from_env, get_optimal_action
from src.envs.forex_env import ForexEnv

class DummyModel(BaseAlgorithm, ABC):
    """
    Algorithm that is only capable of providing predictions.
    """

    def __init__(self):
        # Just passing standard parameters, these are not actually used.
        self.policy_aliases["NoPolicy"] = BasePolicy
        super().__init__(policy="NoPolicy", env=None, learning_rate=lambda _: 0)

    @abstractmethod
    def predict(self, obs: Union[np.ndarray, dict[str, np.ndarray]], *args, **kwargs) -> tuple[np.ndarray, None]:
        pass

    def _setup_model(self, *args, **kwargs) -> None:
        pass

    def learn(self, *args, **kwargs):
        print("DummyModel does not support learning. It only provides predictions.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("DummyModel does not support saving.")

    def load(self, *args, **kwargs):
        raise NotImplementedError("DummyModel does not support loading.")

    def get_env(self):
        raise NotImplementedError("DummyModel has no environments.")

    def set_env(self, *args, **kwargs):
        print("DummyModel does not support setting environments.")

    def get_parameters(self):
        raise NotImplementedError("DummyModel has no parameters.")

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError("DummyModel has no parameters.")

class ConstantModel(DummyModel):
    """
    Model that provides the same prediction everytime.
    """
    def __init__(self, v: float):
        super().__init__()
        self.v = v

    def predict(self, obs: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, None]:
        return np.float64(self.v), None

def short_model(env: ForexEnv) -> ConstantModel:
    if isinstance(env.action_space, spaces.Discrete):
        return ConstantModel(0)
    if isinstance(env.action_space, spaces.Box):
        return ConstantModel(env.action_space.low)
    raise TypeError("Invalid action space.")

def long_model(env: ForexEnv) -> ConstantModel:
    if isinstance(env.action_space, spaces.Discrete):
        return ConstantModel(env.action_space.n - 1)
    if isinstance(env.action_space, spaces.Box):
        return ConstantModel(env.action_space.high)
    raise TypeError("Invalid action space.")

def cash_model(env: ForexEnv) -> ConstantModel:
    if isinstance(env.action_space, spaces.Discrete):
        for x in env.actions:
            if math.isclose(x, 0.0, abs_tol=1e-8):
                return ConstantModel(0.0)
        raise ValueError("cash model needs cash in the action space.")
    if isinstance(env.action_space, spaces.Box):
        if env.action_space.low > 0.0 or env.action_space.high < 0.0:
            raise ValueError("cash model needs cash in the action space.")
        return ConstantModel(0.0)
    raise TypeError("Invalid action space.")

class PerfectModel(DummyModel):

    def __init__(self, env: ForexEnv):
        super().__init__()
        self.env = env
        self.dp_table = get_dp_table_from_env(env)
        assert self.dp_table.n_timesteps == self.env.data_len
        assert self.env.n_actions == 0 or self.dp_table.n_actions == self.env.n_actions

    def predict(self, obs: Union[np.ndarray, dict[str, np.ndarray]], *args, **kwargs) -> tuple[np.ndarray, None]:
        current_cash = self.env.agent_data[self.env.current_step, AgentDataCol.cash]
        current_equity = self.env.agent_data[self.env.current_step + 1, AgentDataCol.pre_action_equity]
        current_exposure = (current_equity - current_cash) / current_equity
        return np.float64(get_optimal_action(self.dp_table, self.env.current_step + 1, current_exposure)), None