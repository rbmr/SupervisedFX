import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any

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
            The columns of the dataframe.
        """
        return self._columns
    
    def set_columns(self, columns: List[str]) -> 'StepwiseFeatureEngineer':
        """
        Set the columns of the dataframe.
        
        Parameters
        ----------
        columns : List[str]
            The columns of the dataframe.
        """
        self._columns = columns
        return self
        
    def add(self, func: Callable[[pd.DataFrame, int], dict]) -> 'StepwiseFeatureEngineer':
        """
        Add a step to the pipeline.
        
        Parameters
        ----------
        func : Callable
            The function to be applied to the dataframe and index. Changes the dataframe in place.
        """
        self._pipeline_steps.append(func)

        return self
    
    #return array of float

    def run(self, df: pd.DataFrame, index: int) -> dict:
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


        return dictionary
        




        