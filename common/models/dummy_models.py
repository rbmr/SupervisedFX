from typing import Any, Callable, Union

import numpy as np
from gymnasium import Space, spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from common.envs.dp import DPTable, get_optimal_action, get_exposure_idx, get_dp_table_from_env
from common.envs.forex_env import ForexEnv, AgentDataCol
from common.constants import PROJECT_DIR

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
        return np.array([self._pred_fn(obs[0])]), None

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

def short_model(env: ForexEnv) -> DummyModel:
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(0))
    if isinstance(action_space, spaces.Box):
        return DummyModel(constant_fn(action_space.low))
    raise TypeError("Invalid action space.")

def long_model(env: ForexEnv) -> DummyModel:
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(action_space.n - 1))
    if isinstance(action_space, spaces.Box):
        return DummyModel(constant_fn(action_space.high))
    raise TypeError("Invalid action space.")

def hold_model(env: ForexEnv) -> DummyModel:
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return DummyModel(constant_fn(action_space.n // 2))
    if isinstance(action_space, spaces.Box):
        middle = (action_space.high - action_space.low) / 2
        return DummyModel(constant_fn(middle))
    raise TypeError("Invalid action space.")

def random_model(env: ForexEnv) -> DummyModel:
    action_space = env.action_space
    return DummyModel(lambda _: action_space.sample())

def custom_comparison_model(env: ForexEnv) -> DummyModel:
    """
    Creates a DummyModel that predicts True if for every adjacent pair of values
    in the observation, the left is strictly greater than the right (e.g., obs[0] > obs[1],
    obs[2] > obs[3], etc.). The last element is ignored if the array has an odd length.
    """
    action_space = env.action_space

    def prediction_logic(obs: np.ndarray) -> bool:
        # Iterate through the observation array with a step of 2 to get adjacent pairs
        for i in range(0, len(obs) - 1, 2):
            # Check if the left element is strictly greater than the right one
            if obs[i] <= obs[i+1]:
                return False
        # If the loop completes, all pairs satisfy the condition
        return True
    
    def prediction_value(obs: np.ndarray) -> Any:
        b = prediction_logic(obs)
        if b:
            if isinstance(action_space, spaces.Discrete):
                return action_space.n - 1
            if isinstance(action_space, spaces.Box):
                return action_space.high
        else:
            if isinstance(action_space, spaces.Discrete):
                return 0
            if isinstance(action_space, spaces.Box):
                return action_space.low
        raise TypeError("Invalid action space.")
            

    return DummyModel(pred_fn=prediction_value)

def dp_perfect_model(env: ForexEnv) -> DummyModel:
    # cache dir is a Path from
    cache_dir = PROJECT_DIR / "temp_cache"
    dp_table = get_dp_table_from_env(env, cache_dir, None)

    def prediction_logic(obs: np.ndarray) -> Any:

        cash = env.agent_data[env.n_steps-1, AgentDataCol.cash]
        pre_action_equity = env.agent_data[env.n_steps, AgentDataCol.pre_action_equity]
        exposure = (pre_action_equity - cash) / pre_action_equity
        optimal_exposure = get_optimal_action(dp_table, env.n_steps, exposure)
        action = get_exposure_idx(optimal_exposure, dp_table.n_actions)
        return action
    
    return DummyModel(pred_fn=prediction_logic)


DUMMY_MODELS: list[Callable[[Space], DummyModel]] = [short_model, long_model, hold_model, random_model]
