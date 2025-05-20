"""
This file contains some simpel scripts that can be useful anywhere during the project.
"""

from datetime import datetime, timedelta
import pandas as pd
from stockstats import StockDataFrame
from common.constants import *

def combine_df(bid_df: pd.DataFrame, ask_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines bid and ask DataFrames into a single `DataFrame`, renaming columns
    by adding a `_bid`, or `_ask` postfix, and calculating the average volume.
    """
    bid_columns = set(bid_df.columns)
    ask_columns = set(ask_df.columns)
    expected_columns = set(DATA_COLUMNS)
    if bid_columns != expected_columns or ask_columns != expected_columns:
        raise ValueError(f"{bid_columns} and {ask_columns} must be equal to {expected_columns}")

    bid_rename = {
        Col.VOL : "volume_bid",
        Col.HIGH : "high_bid",
        Col.LOW : "low_bid",
        Col.OPEN : "open_bid",
        Col.CLOSE : "close_bid"
    }
    ask_rename = {
        Col.VOL : "volume_ask",
        Col.HIGH : "high_ask",
        Col.LOW : "low_ask",
        Col.OPEN : "open_ask",
        Col.CLOSE : "close_ask"
    }
    bid_df.rename(columns=bid_rename, inplace=True)
    ask_df.rename(columns=ask_rename, inplace=True)

    df = pd.merge(bid_df, ask_df, on=Col.TIME, how="inner")
    df[Col.VOL] = (df["volume_bid"] + df["volume_ask"] ) / 2

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
    if ratio < 0 or ratio > 1:
        raise ValueError(f"{ratio} is not a valid ratio")
    split_index = int(len(df) * ratio)
    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]
    return df1, df2
