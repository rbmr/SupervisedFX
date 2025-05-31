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
        

def calculate_cash_percentage(data: np.ndarray, index) -> NDArray[np.float32]:
    """
    Calculate the cash to shares ratio.
    """
    current_cash = data[index, AgentDataCol.cash]
    current_equity = data[index, AgentDataCol.equity_close]
    percentage = current_cash / current_equity
    return np.array([percentage,], dtype=np.float32)


        