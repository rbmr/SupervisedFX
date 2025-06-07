"""
Dynamic Programming Table for Forex Trading Optimization

This module computes the optimal trading strategy using dynamic programming
by working backwards from the final timestep to determine the best action
at each state (timestep, exposure_level).
"""
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import trange

from common.constants import MarketDataCol, AgentDataCol
from common.envs.forex_env import ForexEnv
from common.envs.trade import execute_trade, calculate_equity, reverse_equity
from common.scripts import compute_sliding_window

DATA_HASH_LENGTH = 16

@dataclass
class DPTable:
    """Results from DP computation"""
    value_table: np.ndarray         # Shape: (timesteps, exposure_levels)
    policy_table: np.ndarray        # Shape: (timesteps, exposure_levels)
    n_actions: int                  # The value for n_actions used.
    transaction_cost_pct: float     # The value for transaction_cost_pct used.
    data_hash: str                  # Hash of the market_data used.

def get_exposure_levels(n: int) -> np.ndarray:
    """
    Divides the range [-1, 1] into (2n + 1) equally spaced steps including endpoints.
    """
    return np.linspace(-1, 1, n * 2 + 1, dtype=np.float32)

def get_exposure_idx(x: float, n: int) -> int:
    """
    Given a value x in the range [-1, 1], returns the index of the closest discrete step
    if the range were to be divided into (2n + 1) equally spaced steps including endpoints.
    """
    return int((x + 1) * n + 0.5)

def get_low_high_exposure_idx(x: float, n: int) -> tuple[int, int]:
    """
    Given a value x in the range [-1, 1], returns the index of the closest two discrete steps
    if the range were to be divided into (2n + 1) equally spaced steps including endpoints.
    """
    exact_idx = (x + 1) * n
    low_idx = math.floor(exact_idx)
    high_idx = math.ceil(exact_idx)
    max_idx = 2 * n
    low_idx = np.clip(low_idx, 0, max_idx)
    high_idx = np.clip(high_idx, 0, max_idx)
    return low_idx, high_idx

def get_exposure_val(i: int, n: int) -> float:
    """
    Given an index in the range [0, 2n], returns the corresponding exposure value
    in the range [-1, 1], assuming the range is divided into (2n + 1) equally spaced steps.
    """
    return (i / n) - 1

def get_optimal_action(dp_result: DPTable, timestep: int, current_exposure: float) -> float:
    """
    Get optimal action for a given state timestep dp_result.
    """
    if timestep >= dp_result.policy_table.shape[0]:
        return current_exposure  # No action at terminal state
    exposure_idx = get_exposure_idx(current_exposure, dp_result.n_actions)
    action_idx = dp_result.policy_table[timestep, exposure_idx]
    return get_exposure_val(action_idx, dp_result.n_actions) # type: ignore

def get_dp_table_from_env(env: ForexEnv, cache_dir, n_actions: int | None = None) -> DPTable:
    if n_actions is None:
        n_actions = env.n_actions if env.n_actions > 0 else 7
    return get_dp_table(
        market_data = env.market_data,
        transaction_cost_pct = env.transaction_cost_pct,
        n_actions = n_actions,
        cache_dir = cache_dir
    )

def get_dp_table(market_data: np.ndarray,
                 transaction_cost_pct: float,
                 n_actions: int,
                 cache_dir: Path) -> DPTable:
    """
    Gets the dp table from cache if it exists, otherwise computes it.
    """
    data_hash = get_data_hash(market_data)
    table_name = get_db_table_name(data_hash, transaction_cost_pct, n_actions)
    table_dir = cache_dir / table_name
    table = load_dp_table(table_dir, safe=True)
    if table is not None:
        return table
    table = compute_dp_table(market_data, transaction_cost_pct, n_actions)
    save_dp_table(table, cache_dir)
    return table

def calculate_dp_reward(prev_exposure, curr_exposure, prev_data, current_data, transaction_cost_pct: float) -> float:
    """
    Calculate equity return for transitioning from prev_exposure to curr_exposure.
    Attempts to exactly mimic the .step logic from ForexEnv.
    """
    # Retrieve previous state
    prev_bid = prev_data[MarketDataCol.close_bid]
    prev_ask = prev_data[MarketDataCol.close_ask]
    prev_equity = 10_000.0 # Can be any value, just determines the scale of the output.
    prev_cash, prev_shares = reverse_equity(prev_bid, prev_ask, prev_equity, prev_exposure)

    # Get next state
    curr_bid = current_data[MarketDataCol.close_bid]
    curr_ask = current_data[MarketDataCol.close_ask]
    curr_cash, curr_shares = execute_trade(curr_exposure, current_data, prev_cash, prev_shares, transaction_cost_pct)  # type: ignore
    current_equity = calculate_equity(curr_bid, curr_ask, curr_cash, curr_shares)

    # Calculate equity return
    return current_equity - prev_equity

def compute_dp_table(market_data: np.ndarray,
                     transaction_cost_pct: float = 0.0,
                     n_actions: int = 1,
                     data_hash: str | None = None) -> DPTable:
    """
    Compute the optimal value and policy tables using backward induction
    """
    if n_actions < 1:
        raise ValueError(f"n_actions must be at least one, was {n_actions}")
    if transaction_cost_pct < 0 or transaction_cost_pct > 1:
        raise ValueError(f"transaction_cost_pct must be between 0 and 1, was {transaction_cost_pct}")

    actions = get_exposure_levels(n_actions)
    timesteps = market_data.shape[0]
    n_exposures = len(actions)
    data_hash = data_hash if data_hash is not None else get_data_hash(market_data)

    logging.info(f"Generating DP table with data hash {data_hash}, n_actions {n_actions}, and transaction_cost_pct {transaction_cost_pct}.")

    # Initialize tables
    value_table = np.zeros((timesteps, n_exposures), dtype=np.float32)
    policy_table = np.zeros((timesteps, n_exposures), dtype=int)

    # Backward induction
    for t in trange(timesteps - 2, -1, -1):  # Skip last timestep (terminal)
        for i, current_exposure in enumerate(actions):

            best_val = -np.inf
            best_j = 0

            # Try all possible actions
            for j, target_exposure in enumerate(actions):
                # Calculate immediate reward + future value
                dval = calculate_dp_reward(current_exposure, target_exposure,
                                           market_data[t], market_data[t + 1],
                                           transaction_cost_pct)
                val = dval + value_table[t+1, j]

                if val > best_val:
                    best_val = val
                    best_j = j

            value_table[t, i] = best_val
            policy_table[t, i] = best_j

    return DPTable(
        value_table=value_table,
        policy_table=policy_table,
        n_actions=n_actions,
        transaction_cost_pct=transaction_cost_pct,
        data_hash=data_hash
    )

def get_data_hash(market_data: np.ndarray):
    return hashlib.sha512(market_data.tobytes()).hexdigest()[:DATA_HASH_LENGTH]

def get_db_table_name(data_hash: str,
                      transaction_cost_pct: float,
                      n_actions: int):
    return f"dp_table_n{n_actions}_tc{transaction_cost_pct:.4f}_{data_hash}"

def save_dp_table(table: DPTable, cache_dir: Path) -> Path:
    """
    Save DP table to CSV files for caching.
    """
    # Validate input
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError(f"{cache_dir} is not a directory.")

    # Create paths
    table_dir = cache_dir / get_db_table_name(table.data_hash, table.transaction_cost_pct, table.n_actions)
    table_dir.mkdir(parents=True, exist_ok=True)
    value_path = table_dir / f"values.csv"
    policy_path = table_dir / f"policy.csv"
    metadata_path = table_dir / f"metadata.json"

    logging.info(f"Saving DP table to {table_dir}.")

    # Save data
    pd.DataFrame(table.value_table).to_csv(value_path, index=False)
    pd.DataFrame(table.policy_table).to_csv(policy_path, index=False)
    metadata = {
        "transaction_cost_pct": table.transaction_cost_pct,
        "n_actions": table.n_actions,
        "data_hash": table.data_hash,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f) # Type: ignore

    # Exit
    logging.info(f"DP table saved to {table_dir}")
    return table_dir

def load_dp_table(table_dir: Path, safe: bool = False) -> DPTable | None:
    """
    Load DP table from files.
    """
    logging.info(f"Getting DP table from {table_dir}.")

    # Create paths
    value_path = table_dir / f"values.csv"
    policy_path = table_dir / f"policy.csv"
    metadata_path = table_dir / f"metadata.json"

    # Validate input
    if not all(p.is_file() for p in [value_path, policy_path, metadata_path]):
        if safe:
            logging.info(f"DP table doesnt exist {table_dir}.")
            return None
        raise FileNotFoundError(f"DP table files not found for {table_dir}")

    # Load data
    value_table = pd.read_csv(value_path).to_numpy(dtype=np.float32)
    policy_table = pd.read_csv(policy_path).to_numpy(dtype=np.int16)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Exit
    return DPTable(
        value_table=value_table,
        policy_table=policy_table,
        n_actions=metadata["n_actions"],
        transaction_cost_pct=metadata["transaction_cost_pct"],
        data_hash=metadata["data_hash"],
    )

RewardNormFactory = Callable[[np.ndarray, int | None], Callable[[float, int], float]]

def create_norm_global_mmx(V: np.ndarray, _):
    vmin = V.min()
    vmax = V.max()
    span = (vmax - vmin) or 1.0
    return lambda raw, t: (raw - vmin) / span

def create_norm_global_zscore(V: np.ndarray, _):
    mu = V.mean()
    sigma = V.std() or 1.0
    return lambda raw, t: (raw - mu) / sigma

def create_norm_sliding_mmx(V: np.ndarray, window: int | None):
    if window is None or window < 1:
        raise ValueError("Sliding minmax normalization requires a valid window >= 1")
    mins, maxs = compute_sliding_window(V, window, [np.min, np.max])
    spans = maxs - mins
    spans[spans == 0] = 1.0
    return lambda raw, t: (raw - mins[t]) / spans[t]

def create_norm_sliding_zscore(V: np.ndarray, window: int | None):
    if window is None or window < 1:
        raise ValueError("Sliding z-score normalization requires a valid window >= 1")
    means, stds = compute_sliding_window(V, window, [np.mean, np.std])
    stds[stds == 0] = 1.0
    return lambda raw, t: (raw - means[t]) / stds[t]

def create_dp_reward_function(table: DPTable,
                              rnf: RewardNormFactory,
                              window: int | None = None):
    """
    Returns a dp_reward_function(env) that
    fetches table.value_table[t,idx] and
    applies the chosen normalization.

    Sliding modes require `window` >= 1.
    """
    V = table.value_table
    T, M = V.shape
    normalize = rnf(V, window)
    n_actions = table.n_actions
    actions = get_exposure_levels(table.n_actions)

    def dp_reward_function(env) -> float:
        t = env.current_step
        if t >= T:
            return 0.0

        # extract exposure as before
        cash = env.agent_data[t, AgentDataCol.cash]
        equity = env.agent_data[t, AgentDataCol.equity_close]
        exposure = (equity - cash) / equity

        # bi-linear interpolation of V[t, exposure]
        low_idx, high_idx = get_low_high_exposure_idx(exposure, n_actions)
        low, high = actions[low_idx], actions[high_idx]
        alpha = 0 if high == low else (exposure - low) / (high - low)
        raw = (1 - alpha) * V[t, low_idx] + alpha * V[t, high_idx]

        # Normalize and return
        return normalize(raw, t) # type: ignore

    return dp_reward_function