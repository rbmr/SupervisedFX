"""
This file contains some simple scripts that can be useful anywhere during the project.
"""
import ast
import functools
import json
import os
import random
import signal
import tempfile
import time
from asyncio import FIRST_EXCEPTION
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Generator, TypeVar, Optional

import numpy as np
import pandas as pd
import requests

K = TypeVar("K")
V = TypeVar("V")

def contains_nan_or_inf(arr: np.ndarray) -> bool:
    """
    Returns true if any element of the array is NaN or Inf, else returns false.
    arr can be any shape.
    """
    return np.isnan(arr).any() or np.isinf(arr).any()

def find_first_valid_row(arr: np.ndarray) -> int:
    """
    Returns the index of the first row without a NaN or an Inf.
    """
    assert arr.ndim == 2
    for i in range(arr.shape[0]):
        if not contains_nan_or_inf(arr[i]):
            return i
    return -1

def find_first_invalid_row(arr: np.ndarray) -> int:
    """
    Returns the index of the first row that contains a NaN or an Inf.
    """
    assert arr.ndim == 2
    for i in range(arr.shape[0]):
        if contains_nan_or_inf(arr[i]):
            return i
    return -1

def safe_literal_eval(s: Optional[str]):
    if s is None:
        return None
    try:
        return ast.literal_eval(s)
    except ValueError or SyntaxError:
        return None

def rolling_apply(arr: np.ndarray, window: int, fn: Callable[[np.ndarray], np.float64]):
    assert window > 0, f"window must be positive, was {window}"
    result = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for end in range(window, arr.shape[0]+1):
        result[end-1] = fn(arr[end-window:end])
    return result

def shift(arr: np.ndarray, num: int) -> np.array:
    """Helper function to shift an array similar to Pandas shift"""
    shifted_arr = np.full_like(arr, np.nan)
    if num > 0:
        shifted_arr[num:] = arr[:-num]
    elif num < 0:
        shifted_arr[:num] = arr[-num:]
    else:
        shifted_arr[:] = arr
    return shifted_arr

def df_to_np_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Converts a Pandas DataFrame to a dictionary of NumPy arrays.
    """
    numpy_dict = {}
    for col_name in df.columns:
        col_data = df[col_name]
        assert col_data.dtype != 'object'
        if col_data.dtype == 'object':
            raise TypeError(f"Column '{col_name}' has 'object' dtype, which is not allowed. "
                            "Please ensure all columns are numeric, datetime, or boolean.")
        if pd.api.types.is_datetime64_any_dtype(col_data):
            numpy_dict[col_name] = col_data.to_numpy(dtype='datetime64[ns]')
        else:
            numpy_dict[col_name] = col_data.to_numpy()
    return numpy_dict

def compute_sliding_window(arr: np.ndarray, window: int, fns: list[Callable[[np.ndarray], float]]):
    """
    Applies functions to a sliding window, caching the result.
    """
    T = arr.shape[0]
    results = [np.zeros(T,dtype=np.float32) for _ in range(len(fns))]
    for t in range(T):
        start = max(0, t - window + 1)
        window_slice = arr[start:t + 1]
        for res, fn in zip(results, fns):
            res[t] = fn(window_slice)
    return results

class lazy_singleton:
    """
    Decorator that makes a function behave like a lazy singleton:
    1. Value is computed at most once (cached after first call)
    2. Value is computed only if function is called (lazy evaluation)
    """
    _NOT_COMPUTED = object()

    def __init__(self, func: Callable[[], Any]):
        self.func = func
        self.cache = self._NOT_COMPUTED
        functools.update_wrapper(self, func)

    def __call__(self):
        if self.cache is self._NOT_COMPUTED:
            self.cache = self.func()
        return self.cache

def parallel_apply(func: Callable[[K], V], inputs: list[K], num_workers: int) -> None:
    """
    Applies a function to a list of inputs in parallel.
    Raises an exception when an exception occurs in any process, and cancels all other processes.
    """
    assert num_workers > 0, "number of workers must be greater than 0"
    if num_workers == 1:
        for inp in inputs:
            func(inp)
        return None
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = [executor.submit(func, inp) for inp in inputs]
    try:
        done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            exc = f.exception()
            if exc is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                raise exc
    finally:
        # Ensure executor is cleaned up (no-op if already shut down)
        executor.shutdown(wait=False, cancel_futures=True)

def index_wrapper(func: Callable[[K], V], pair: tuple[int, K]) -> tuple[int, V]:
    i, x = pair
    return i, func(x)

def parallel_map(func: Callable[[K], V], inputs: list[K], num_workers: int) -> list[V]:
    """
    Maps a function to a list in parallel, returning results in proper order.
    Raises an exception when an exception occurs in any process, and cancels all other processes.
    """
    assert num_workers > 0, "number of workers must be greater than 0"
    if num_workers == 1:
        return [func(inp) for inp in inputs]
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
        except Exception as e:
            pool.terminate()
            pool.join()
            raise e
        else:
            pool.close()
            pool.join()
    return results

def clean_numpy(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = clean_numpy(v)
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = clean_numpy(v)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

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
        except KeyboardInterrupt as e:
            raise e
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

def shuffle(arr1: np.ndarray, arr2: np.ndarray, axis=0):
    """
    Shuffles two arrays along axis1 preserving row matching.
    """
    assert arr1.shape[axis] == arr2.shape[axis]

    indices = np.random.permutation(arr1.shape[axis])
    arr1_shuffled = arr1[indices]
    arr2_shuffled = arr2[indices]
    return arr1_shuffled, arr2_shuffled

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
    Raises a `ValueError` if `a` is not divisible by `b`.
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

def parse_args(args_string: str | None) -> tuple:
    """
    Parses function arguments.
    None -> ()
    "" -> ()
    "1" -> (1,)
    "(1,2,3)" -> ((1,2,3),)
    "1,2,3" -> (1,2,3)
    """
    if args_string is None:
        return ()
    args_string = args_string.strip()
    if args_string == "":
        return ()
    return ast.literal_eval(f"({args_string},)")

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

def write_atomic_json(data, path: Path):
    """
    Writes a JSON file guaranteeing atomicity.
    """
    dir_name = path.parent
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
        json.dump(data, tf)
        tf.flush() # write python buffer to os buffer
        os.fsync(tf.fileno()) # write os buffer to disk
        temp_path = tf.name
    os.replace(temp_path, path)

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