"""
This file contains some simpel scripts that can be useful anywhere during the project.
"""

from datetime import datetime, timedelta
import pandas as pd
from stockstats import StockDataFrame
from common.constants import *

def combine_df(bid_df: pd.DataFrame, ask_df: pd.DataFrame):
    """
    Combines bid and ask DataFrames into a single DataFrame, renaming columns
    and calculating the average volume.
    """

    bid_df.rename(columns={col : col+"_bid" for col in bid_df.columns if col != Col.TIME}, inplace=True)
    ask_df.rename(columns={col : col+"_ask" for col in bid_df.columns if col != Col.TIME}, inplace=True)

    df = pd.merge(bid_df, ask_df, on=Col.TIME, how="inner")
    df[Col.VOL] = (df[Col.VOL+"_bid"] + df[Col.VOL+"_ask"] ) / 2

    return df

def round_datetime(date_time: datetime, interval: int) -> datetime:
    """
    Rounds a datetime object to the nearest multiple of `interval` in seconds.
    """
    start_of_day = date_time.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_start = (date_time - start_of_day).total_seconds()
    rounded_seconds = round(seconds_since_start / interval) * interval
    return start_of_day + timedelta(seconds=rounded_seconds)

def exact_divide(a: int, b: int) -> int:
    """
    Performs an exact division of `a` by `b`, returning an integer.
    Raises a `ValueError`, if `a` is not divisible by `b`.
    """
    if a % b == 0:
        return a // b
    raise ValueError(f"{a} is not divisible by {b}")

def split_df(df: pd.DataFrame, ratio: float):
    """
    Splits a dataframe into two parts based on a given ratio.
    """
    split_index = int(len(df) * ratio)
    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]
    return df1, df2
