"""
This file contains some simple scripts that can be useful anywhere during the project.
"""
import random
import signal
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import cpu_count, Pool, get_context
from pathlib import Path
from typing import Any, Dict, Generator, Callable, TypeVar
import time
import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from common.constants import MarketDataCol

K = TypeVar("K")
V = TypeVar("V")

def index_wrapper(func: Callable[[K], V], pair: tuple[int, K]) -> tuple[int, V]:
    i, x = pair
    return i, func(x)

def parallel_run(func: Callable[[K], V], inputs: list[K], num_workers: int) -> list[V]:
    """
    Applies a function to a list in parallel, returning results in proper order.
    """
    num_workers = max(min(cpu_count()-1, num_workers), 1)
    results: list[V | None] = [None] * len(inputs)
    ctx = get_context("spawn")
    indexed_inps = enumerate(inputs)
    indexed_fn = partial(index_wrapper, func)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with ctx.Pool(processes=num_workers) as pool:
        signal.signal(signal.SIGINT, original_sigint_handler)
        try:
            for i, res in pool.imap_unordered(indexed_fn, indexed_inps):
                results[i] = res
        except KeyboardInterrupt as e:
            pool.terminate()
            pool.join()
            raise e
        else:
            pool.close()
            pool.join()
    return results

def fetch(session, url, retries: int = 16, raise_on_fail: bool = True) -> bytes | None:
    delay = 1
    for _ in range(retries):
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            time.sleep(delay)
            delay *= 2
    if not raise_on_fail:
        print(f"Failed to fetch {url} after {retries} retries")
        return None
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")

def fetch_all(urls: list[str], raise_on_fail: bool = True) -> list:
    results = []
    with requests.Session() as session:
        for url in urls:
            result = fetch(session, url, raise_on_fail)
            results.append(result)
    return results

def raise_value_error(msg):
    raise ValueError(msg)

def map_input(input_str: str, fns: list[Callable]):
    while True:
        try:
            inp = input(input_str)
            for fn in fns:
                inp = fn(inp)
            return inp
        except BaseException as e:
            print(f"Invalid input: {e}")

def date_range(start: datetime, end: datetime, step: timedelta) -> Generator[datetime, None, None]:
    current = start
    while current < end:
        yield current
        current += step

def has_nonempty_subdir(path: Path, subdir_name: str) -> bool:
    return has_subdir(path, subdir_name) and not is_empty(path / subdir_name)

def has_subdir(path: Path, subdir_name: str) -> bool:
    """Checks if a path has a subdirectory with some name"""
    return path.is_dir() and (path / subdir_name).is_dir()

def lookup(options: list[tuple[str, Any]], key: str):
    for k, v in options:
        if k == key:
            return v
    return None

def safe_int(i: str) -> int | None:
    try:
        return int(i)
    except ValueError:
        return None

def is_empty(p: Path) -> bool:
    return n_children(p) == 0

def n_children(p: Path) -> int:
    return sum(1 for _ in p.iterdir())

def picker(options: list[tuple[str, Any]], default: int | None = 0) -> Any:
    """
    Prompts user to pick one of a list of options.
    Supports default values.
    """
    default_str = "" if default is None else f" (default: {default})"
    print(f"Pick one of the following options{default_str}:")
    for i, (name, _) in reversed(list(enumerate(options))):
        print(f"[{i}] {name}")

    while True:

        inp = input("> ").strip()
        if inp == "" and default is not None:
            return options[0][1]
        i = safe_int(inp)
        if i is not None:
            if 0 <= i < len(options):
                return options[i][1]
            print("Index out of range. Try again.")
            continue
        val = lookup(options, inp)
        if val is not None:
            return val
        print("Name not found. Try again.")

def most_recent_modified(dir_path: Path):
    """finds the most recently modified file or folder in a directory"""
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")
    entries = list(dir_path.iterdir())
    if len(entries) == 0:
        return None
    return max(entries, key=lambda p: p.stat().st_mtime)

def calculate_equity(bid_price: float, ask_price: float, cash: float, shares: float) -> float:
    """
    Calculates the equity based on current cash, shares and prices.
    """
    return cash + shares * (bid_price if shares >= 0 else ask_price)

def calculate_ohlc_equity(current_prices: NDArray[np.float32], cash: float, shares: float) -> tuple[float, float, float, float]:
    """
    Calculates the equity based on current cash, shares and prices.
    """
    assert current_prices.ndim == 1
    assert current_prices.shape[0] == len(MarketDataCol)
    equity_open = calculate_equity(current_prices[MarketDataCol.open_bid], current_prices[MarketDataCol.open_ask], cash, shares)
    equity_high = calculate_equity(current_prices[MarketDataCol.high_bid], current_prices[MarketDataCol.high_ask], cash, shares)
    equity_low = calculate_equity(current_prices[MarketDataCol.low_bid], current_prices[MarketDataCol.low_ask], cash, shares)
    equity_close = calculate_equity(current_prices[MarketDataCol.close_bid], current_prices[MarketDataCol.close_ask], cash, shares)
    return equity_open, equity_high, equity_low, equity_close

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

def render_horz_bar(height: float) -> str:
    """Renders a horizontal bar using fractional Unicode block characters"""
    full_blocks = int(height)
    remainder = height - full_blocks
    partial_block = " ▏▎▍▌▋▊▉"[int(remainder * 8)] # 0/8 - 7/8
    return "█" * full_blocks + partial_block

def circ_slice(arr, i, j):
    """Perform circular (wraparound) slicing on a NumPy array."""
    n = len(arr)
    i %= n
    j %= n
    if i > j:
        return np.concatenate((arr[i:], arr[:j]))
    if j > i:
        return arr[i:j]
    return arr[:]

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