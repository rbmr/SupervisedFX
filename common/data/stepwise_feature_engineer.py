from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from common.constants import AgentDataCol


class StepwiseFeatureEngineer:

    def __init__(self):
        self._steps: List[tuple[list[str], Callable[[np.ndarray, int], np.ndarray]]] = []
        self._features: list[str] = []

    def get_features(self) -> list[str]:
        return self._features

    def num_of_features(self) -> int:
        return len(self._features)

    def add(self, features: list[str], func: Callable[[np.ndarray, int], np.ndarray]):
        """
        Add a step to the pipeline.
        """
        self._steps.append((features, func))
        self._features += features
        return self

    def run(self, arr: np.ndarray, index: int) -> NDArray[np.float32]:
        """
        Run the pipeline on the given DataFrame.
        """

        res = np.empty(len(self._features), dtype=np.float32)
        offset = 0

        for features, func in self._steps:
            # Get step output
            func_res = func(arr, index)

            # Validate output
            if not isinstance(func_res, np.ndarray):
                raise ValueError(f"Function {func.__name__} must return a ndarray.")
            if func_res.ndim != 1:
                raise ValueError(f"Function {func.__name__} must return a 1d array.")
            if func_res.shape[0] != len(features):
                raise ValueError("Function didnt return expected number of features.")
            if np.isinf(func_res).any():
                inf_indices = np.where(np.isinf(func_res))[0]
                inf_features = ", ".join(features[i] for i in inf_indices)
                raise ValueError(f"Feature(s) {inf_features} returned by {func.__name__} are infinity.")
            if np.isnan(func_res).any():
                nan_indices = np.where(np.isnan(func_res))[0]
                nan_features = ", ".join(features[i] for i in nan_indices)
                raise ValueError(f"Feature(s) {nan_features} returned by {func.__name__} are NaN.")

            # Store output
            n = func_res.shape[0]
            res[offset:offset+n] = func_res
            offset += n

        return res

#function to calculate historic lookback of another function
def calculate_historic_lookback(data: np.ndarray, index: int, lookback: int, func: Callable[[np.ndarray, int], np.ndarray]) -> NDArray[np.float32]:
    """
    Calculate the historic lookback of a feature.
    """
    if index < lookback:
        return np.zeros(lookback * func(data, 0).shape[0], dtype=np.float32)
    
    # Calculate the feature for the current index
    current_feature = func(data, index)
    
    # Calculate the feature for the previous indices
    historic_features = np.array([func(data, i) for i in range(index - lookback, index)])
    
    # Combine current and historic features
    return np.concatenate((historic_features.flatten(), current_feature))

def calculate_current_exposure(data: np.ndarray, index: int) -> NDArray[np.float32]:
    """Calculates the current exposure as a value between -1 and 1."""
    equity = data[index, AgentDataCol.equity_close]
    cash = data[index, AgentDataCol.cash]
    if equity <= 0:
        exposure = 0.0
    else:
        exposure = (equity - cash) / equity
    return np.array([exposure,], dtype=np.float32)

def get_current_exposure(data: np.ndarray, index) -> NDArray[np.float32]:
    """
    Calculate the cash to shares ratio.
    """
    exposure = data[index, AgentDataCol.target_exposure]
    return np.array([exposure,], dtype=np.float32)

def duration_of_current_trade(data: np.ndarray, index, scaling_factor = 24) -> NDArray[np.float32]:
    curr = data[index, AgentDataCol.target_exposure]
    length = 1
    index -= 1

    while index >= 0 and data[index, AgentDataCol.target_exposure] == curr:
        length += 1
        index -= 1

    length = min(length / scaling_factor, 1.0)

    return np.array([length,], dtype=np.float32)
    
        


        