import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any
from common.optimization import DataFrameToNumPyAccessor

class StepwiseFeatureEngineer:
    def __init__(self, columns: List[str]):
        self._pipeline_steps: List[Dict[str, Any]] = []
        self._columns = columns
    
    def get_columns(self) -> List[str]:
        """
        Get the columns of the dataframe.
        
        Returns
        -------
        List[str]
            The columns returned by the pipeline.
        """
        return self._columns
    
    def set_columns(self, columns: List[str]) -> 'StepwiseFeatureEngineer':
        """
        Set the columns of the dataframe.
        
        Parameters
        ----------
        columns : List[str]
            The columns returned by the pipeline.
        """
        self._columns = columns
        return self
        
    def add(self, func: Callable[[DataFrameToNumPyAccessor, int], dict]) -> 'StepwiseFeatureEngineer':
        """
        Add a step to the pipeline.
        
        Parameters
        ----------
        func : Callable
            The function to be applied to the dataframe and index. Changes the dataframe in place.
            THIS FUNCTION SHOULD NEVER RETURN NaN OR INFINITY.
        """
        self._pipeline_steps.append(func)

        return self
    
    #return array of float

    def run(self, df: DataFrameToNumPyAccessor, index: int) -> dict:
        """
        Run the pipeline on the given DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be processed.
        
        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """

        dictionary = {}

        for func in self._pipeline_steps:
            # run func
            temp = func(df, index)
            if isinstance(temp, dict):
                dictionary.update(temp)
            else:
                raise ValueError("Function must return a dictionary.")
            
        # for each key in dictionary, check if it is in columns
        for key in dictionary.keys():
            if key not in self._columns:
                # remove key from dictionary
                del dictionary[key]

        # check all columns in dictionary are in columns
        for key in dictionary.keys():
            if key not in self._columns:
                raise ValueError(f"Key {key} not in columns of the stepwise feature engineer. Fix the function {func} to return a key in the columns.")
            
        # check none of the values in dictionary are NaN or infinity
        for key in dictionary.keys():
            if pd.isna(dictionary[key]):
                raise ValueError(f"Value {dictionary[key]} for key {key} is NaN. Fix the function {func} to return a value that is not NaN.")
            if np.isinf(dictionary[key]):
                raise ValueError(f"Value {dictionary[key]} for key {key} is infinity. Fix the function {func} to return a value that is not infinity.")


        return dictionary
        

def calculate_cash_percentage(data_accessor, index):
        """
        Calculate the cash to shares ratio.
        """
        current_cash = data_accessor[index, 'cash']
        current_equity = data_accessor[index, 'equity_close']
        percentage = current_cash / current_equity
        return {'cash_percentage': percentage}


        