import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any

class FeatureEngineer:
    def __init__(self):
        self._pipeline_steps: List[Dict[str, Any]] = []
        
    def add(self, func: Callable[[pd.DataFrame], None]) -> 'FeatureEngineer':
        """
        Add a step to the pipeline.
        
        Parameters
        ----------
        func : Callable
            The function to be applied to the dataframe.
        """
        self._pipeline_steps.append(func)

        return self
    
    def run(self, df: pd.DataFrame, remove_original_columns=True) -> pd.DataFrame:
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

        df = df.copy()  # Avoid modifying the original DataFrame

        original_columns = df.columns.tolist()

        for func in self._pipeline_steps:
            # run func
            func(df)

        # if remove original columns
        if remove_original_columns:
            # remove original columns
            for col in original_columns:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

        
        return df