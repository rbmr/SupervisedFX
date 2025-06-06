from typing import Any, Callable, Union

import numpy as np
import torch as th
from gymnasium import spaces, Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

def constant_fn(x: Any) -> Callable[[Any], Any]:
    """Creates a single argument function that returns a constant value."""
    return lambda _: x

class DummyModel(BaseAlgorithm):
    """
    Algorithm that is only capable of providing predictions.
    Predictions are made by passing the observation to a function.
    """

    def __init__(self, pred_fn: Callable[[np.ndarray], Any]):
        # Just passing standard parameters, these are not actually used.
        self.policy_aliases["NoPolicy"] = BasePolicy
        super().__init__(policy="NoPolicy", env=None, learning_rate=lambda _: 0)
        # Set function used to generate predictions
        self._pred_fn = pred_fn

    # Overrides
    def predict(self, obs: Union[np.ndarray, dict[str, np.ndarray]], *args, **kwargs) -> tuple[np.ndarray, None]:
        return np.array([self._pred_fn(obs)]), None

    def _setup_model(self, *args, **kwargs) -> None:
        pass

    def learn(self, *args, **kwargs):
        raise NotImplementedError("DummyModel does not support learning.")

    def save(self, *args, **kwargs):
        raise NotImplementedError("DummyModel does not support saving.")

    def load(self, *args, **kwargs):
        raise NotImplementedError("DummyModel does not support loading.")

    def get_env(self):
        raise NotImplementedError("DummyModel has no environments.")

    def set_env(self, *args, **kwargs):
        raise NotImplementedError("DummyModel has no environments.")

    def get_parameters(self):
        raise NotImplementedError("DummyModel has no parameters.")

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError("DummyModel has no parameters.")

def short_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(0))
    if isinstance(action_space, spaces.Box):
        return DummyModel(constant_fn(action_space.low))
    raise TypeError("Invalid action space.")

def long_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(action_space.n - 1))
    if isinstance(action_space, spaces.Box):
        return DummyModel(constant_fn(action_space.high))
    raise TypeError("Invalid action space.")

def hold_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(action_space.n // 2))
    if isinstance(action_space, spaces.Box):
        middle = (action_space.high - action_space.low) / 2
        return DummyModel(constant_fn(middle))
    raise TypeError("Invalid action space.")

def random_model(action_space: Space) -> DummyModel:
    return DummyModel(lambda _: action_space.sample())

DUMMY_MODELS: list[Callable[[Space], DummyModel]] = [short_model, long_model, hold_model, random_model]