import pandas as pd
import numpy as np
from typing import Callable, List, Any

from common.envs.forex_env import AgentDataCol

class StepwiseFeatureEngineer:

    def __init__(self, columns: List[str]):
        self._pipeline_steps: List[Callable[[np.ndarray, int], dict[str, Any]]] = []
        self._columns = columns
    
    def get_columns(self) -> List[str]:
        """
        Get the columns of the dataframe.
        """
        return self._columns
    
    def set_columns(self, columns: List[str]) -> 'StepwiseFeatureEngineer':
        """
        Set the columns of the dataframe.
        """
        self._columns = columns
        return self
        
    def add(self, func: Callable[[np.ndarray, int], dict[str, Any]]) -> 'StepwiseFeatureEngineer':
        """
        Add a step to the pipeline.
        """
        self._pipeline_steps.append(func)
        return self

    def run(self, arr: np.ndarray, index: int) -> dict[str, Any]:
        """
        Run the pipeline on the given DataFrame.
        """

        dictionary = {}

        for func in self._pipeline_steps:
            # run func
            temp = func(arr, index)
            if not isinstance(temp, dict):
                raise ValueError("Function must return a dictionary.")
            for key, value in temp.items():
                if key not in self._columns:
                    raise ValueError(f"Key {key} returned by {func.__name__} not in columns {self._columns}.")
                if pd.isna(value):
                    raise ValueError(f"Value {value} for key {key} returned by {func.__name__} is NaN.")
                if np.isinf(dictionary[key]):
                    raise ValueError(f"Value {value} for key {key} returned by {func.__name__} is infinity.")
            dictionary.update(temp)

        return dictionary
        

def calculate_cash_percentage(data: np.ndarray, index) -> dict[str, Any]:
    """
    Calculate the cash to shares ratio.
    """
    current_cash = data[index, AgentDataCol.cash]
    current_equity = data[index, AgentDataCol.equity_close]
    percentage = current_cash / current_equity
    return {'cash_percentage': percentage}


        