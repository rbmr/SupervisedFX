from typing import Any, Callable

import numpy as np
import torch as th
from gymnasium import spaces, Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

class DummyModel(BaseAlgorithm):
    """
    Algorithm that is only capable of providing predictions.
    Predictions are made by returning a constant value, or calling a function without arguments.
    """

    def __init__(self, prediction: Any | Callable[[], Any]):
        # Just passing standard parameters, these are not actually used.
        super().__init__(policy="MlpPolicy", env=None, learning_rate=lambda _: 0)
        # Set function used to generate predictions
        self._pred = prediction if callable(prediction) else lambda: prediction

    # Overrides
    def predict(self, *args, **kwargs) -> tuple[np.ndarray, None]:
        return np.array([self._pred()]), None

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

def get_short_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(0)
    if isinstance(action_space, spaces.Box):
        return DummyModel(action_space.low)
    raise TypeError("Invalid action space.")

def get_long_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(action_space.n - 1)
    if isinstance(action_space, spaces.Box):
        return DummyModel(action_space.high)
    raise TypeError("Invalid action space.")

def get_hold_model(action_space: Space) -> DummyModel:
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(action_space.n // 2)
    if isinstance(action_space, spaces.Box):
        middle = (action_space.high - action_space.low) / 2
        return DummyModel(middle)
    raise TypeError("Invalid action space.")

def get_random_model(action_space: Space) -> DummyModel:
    return DummyModel(lambda: action_space.sample())

