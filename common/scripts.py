"""
This file contains some simpel scripts that can be useful anywhere during the project.
"""

import random
from typing import Any, Dict

import numpy as np
import pandas as pd

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
        RawDataCol.VOL : "volume_bid",
        RawDataCol.HIGH : "high_bid",
        RawDataCol.LOW : "low_bid",
        RawDataCol.OPEN : "open_bid",
        RawDataCol.CLOSE : "close_bid"
    }
    ask_rename = {
        RawDataCol.VOL : "volume_ask",
        RawDataCol.HIGH : "high_ask",
        RawDataCol.LOW : "low_ask",
        RawDataCol.OPEN : "open_ask",
        RawDataCol.CLOSE : "close_ask"
    }
    bid_df.rename(columns=bid_rename, inplace=True)
    ask_df.rename(columns=ask_rename, inplace=True)

    df = pd.merge(bid_df, ask_df, on=RawDataCol.TIME, how="inner")
    df[RawDataCol.VOL] = (df["volume_bid"] + df["volume_ask"]) / 2

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
    df1 = df.iloc[:split_index].reset_index(drop=True)
    df2 = df.iloc[split_index:].reset_index(drop=True)
    return df1, df2

def find_first_row_without_nan(df: pd.DataFrame) -> int:
    """
    Returns the index of the first row that contains no NaN values.
    Returns -1 if no such row exists.
    """
    for i in range(len(df)):
        if not df.iloc[i].isnull().any():
            return i
    return -1

def find_first_row_with_nan(df: pd.DataFrame) -> int:
    """
    Returns the index of the first row that contains a NaN value.
    Returns -1 if no such row exists.
    """
    for i in range(len(df)):
        if df.iloc[i].isnull().any():
            return i
    return -1

def set_seed(seed_value: int):
    """
    Sets the random seed for Python, NumPy, TensorFlow, and PyTorch.
    """
    import tensorflow as tf
    import torch

    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    torch.manual_seed(seed_value)

    # If you are using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # The following two lines are often recommended for deterministic behavior on CUDA,
        # but they can impact performance. Use them if reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seeds set to {seed_value} for Python, NumPy, TensorFlow, and PyTorch.")

def flatten_dict(d: Dict[str, Any], sep=".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary to a single level.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = flatten_dict(value)
            for sub_key, sub_value in value.items():
                flat_dict[f"{key}{sep}{sub_key}"] = sub_value
        else:
            flat_dict[key] = value
    return flat_dict