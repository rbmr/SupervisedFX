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
    
def remove_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove OHLCV columns from the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be processed.
    
    Returns
    -------
    pd.DataFrame
        The DataFrame with OHLCV columns removed.
    """
    ohlcv_columns = ['volume', 'date_gmt',
                     'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid',
                     'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask']
    df.drop(columns=ohlcv_columns, inplace=True, errors='ignore')
    
    return df

def history_lookback(df: pd.DataFrame, lookback_window_size: int, columns: List[str] = []) -> pd.DataFrame:
    """
    Create a history lookback window for the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be processed.
    lookback_window_size : int
        The size of the lookback window.
    columns : List[str]
        The columns to be included in the lookback window.
    
    Returns
    -------
    pd.DataFrame
        The DataFrame with the lookback window added.
    """
    if not columns:
        columns = df.columns.tolist()

    # for each column make a new column shifted by 1, 2, ..., lookback_window_size
    for col in columns:
        for i in range(1, lookback_window_size + 1):
            df[f'{col}_shift_{i}'] = df[col].shift(i)
    
    return df

def rsi(df, window=14):
        """
        Calculate the Relative Strength Index (RSI) column.
        """
        delta = df['close_bid'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        
        # replace rs with 1 if NaN, 100 if rs is np.inf, or 0 if rs is -np.inf
        rs = rs.fillna(1)
        rs = rs.replace(np.inf, 100)
        rs = rs.replace(-np.inf, 0)


        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi
        return df

