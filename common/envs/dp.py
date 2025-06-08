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

import numpy as np
import pandas as pd
from tqdm import trange

from common.constants import AgentDataCol, MarketDataCol
from common.envs.forex_env import ForexEnv
from common.envs.trade import calculate_equity, execute_trade, reverse_equity

DATA_HASH_LENGTH = 16

@dataclass
class DPTable:
    """Results from DP computation"""
    value_table: np.ndarray         # Shape: (timesteps, 2 * n_actions + 1)
    policy_table: np.ndarray        # Shape: (timesteps, 2 * n_actions + 1)
    q_min_table: np.ndarray         # Shape: (timesteps, 2 * n_actions + 1)
    n_actions: int                  # The value for n_actions used.
    transaction_cost_pct: float     # The value for transaction_cost_pct used.
    data_hash: str                  # Hash of the market_data used.

def get_exposure_levels(n: int) -> np.ndarray:
    """
    Divides the range [-1, 1] into (2n + 1) equally spaced steps including endpoints.
    """
    return np.linspace(-1, 1, n * 2 + 1, dtype=np.float64)

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

def get_optimal_action(table: DPTable, t: int, current_exposure: float) -> float:
    """
    Get optimal action from a table given a timestep t and current exposure value.
    """
    if t >= table.policy_table.shape[0]:
        return current_exposure  # No action at terminal state
    exposure_idx = get_exposure_idx(current_exposure, table.n_actions)
    action_idx = table.policy_table[t, exposure_idx]
    return get_exposure_val(action_idx, table.n_actions) # type: ignore

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

def compute_dp_table(market_data: np.ndarray,
                     transaction_cost_pct: float = 0.0,
                     n_actions: int = 1,
                     data_hash: str | None = None) -> DPTable:
    """
    Compute the optimal value and policy tables using backward induction.
    - value_table[t, e]: the maximal expected cumulative log-equity from (t,e).
    - policy_table[t, e]: contains the best target exposure (action) for timestep t and exposure e.
    - q_min_table[t, e] the worst one-step continuation log-equity among all actions at (t,e).
    Where the equity ratio is defined as the ratio between the current equity and the equity and the end of the time domain.
    """
    if n_actions < 1:
        raise ValueError(f"n_actions must be at least one, was {n_actions}")
    if n_actions > 127:
        raise ValueError(f"n_actions must be at most 127, was {n_actions}")
    if transaction_cost_pct < 0 or transaction_cost_pct > 1:
        raise ValueError(f"transaction_cost_pct must be between 0 and 1, was {transaction_cost_pct}")

    actions = get_exposure_levels(n_actions)
    timesteps = market_data.shape[0]
    data_hash = data_hash if data_hash is not None else get_data_hash(market_data)

    logging.info(f"Generating DP table with data hash {data_hash}, n_actions {n_actions}, and transaction_cost_pct {transaction_cost_pct}.")

    # Initialize tables
    value_table = np.zeros((timesteps, len(actions)), dtype=np.float64)
    policy_table = np.full((timesteps, len(actions)), 255, dtype=np.uint8) # 255 indicates unfilled value.
    q_min_table = np.full((timesteps, len(actions)), np.inf, dtype=np.float64)

    # Backward induction
    for t in trange(timesteps - 2, -1, -1):  # Skip last timestep (terminal)
        for i, curr_exposure in enumerate(actions):

            best_val = -np.inf
            worst_val = np.inf
            best_j = 0

            # Try all possible actions
            for j, target_exposure in enumerate(actions):

                # Get cash, and shares just BEFORE executing the trade.
                curr_bid = market_data[t, MarketDataCol.open_bid]
                curr_ask = market_data[t, MarketDataCol.open_ask]
                curr_cash, curr_shares = reverse_equity(curr_bid, curr_ask, 1, curr_exposure) # type: ignore

                # Get cash, and shares just AFTER executing the trade using target exposure.
                next_cash, next_shares = execute_trade(target_exposure, curr_bid, curr_ask, # type: ignore
                                                       curr_cash, curr_shares, transaction_cost_pct)

                # Get equity just BEFORE executing the next trade.
                next_bid = market_data[t + 1, MarketDataCol.open_bid]
                next_ask = market_data[t + 1, MarketDataCol.open_ask]
                next_equity = calculate_equity(next_bid, next_ask, next_cash, next_shares) # type: ignore
                next_exposure = (next_equity - next_cash) / next_equity

                # next_equity is equity ratio, since curr_equity was 1.
                val = np.log(next_equity) + interp(value_table[t+1], next_exposure, n_actions)

                if val > best_val:
                    best_val = val
                    best_j = j
                if val < worst_val:
                    worst_val = val

            value_table[t, i] = best_val
            policy_table[t, i] = best_j
            q_min_table[t, i] = worst_val

    return DPTable(
        value_table=value_table,
        policy_table=policy_table,
        q_min_table=q_min_table,
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
    value_path = table_dir / f"value.csv"
    policy_path = table_dir / f"policy.csv"
    q_min_path = table_dir / f"q_min.csv"
    metadata_path = table_dir / f"metadata.json"

    logging.info(f"Saving DP table to {table_dir}.")

    # Save data
    pd.DataFrame(table.value_table).to_csv(value_path, index=False)
    pd.DataFrame(table.policy_table).to_csv(policy_path, index=False)
    pd.DataFrame(table.q_min_table).to_csv(q_min_path, index=False)
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
    value_path = table_dir / f"value.csv"
    policy_path = table_dir / f"policy.csv"
    q_min_path = table_dir / f"q_min.csv"
    metadata_path = table_dir / f"metadata.json"

    # Validate input
    if not all(p.is_file() for p in [value_path, policy_path, q_min_path, metadata_path]):
        if safe:
            logging.info(f"DP table doesnt exist {table_dir}.")
            return None
        raise FileNotFoundError(f"DP table files not found for {table_dir}")

    # Load data
    value_table = pd.read_csv(value_path).to_numpy(dtype=np.float64)
    policy_table = pd.read_csv(policy_path).to_numpy(dtype=np.uint8)
    q_min_table = pd.read_csv(q_min_path).to_numpy(dtype=np.float64)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Exit
    return DPTable(
        value_table=value_table,
        policy_table=policy_table,
        q_min_table=q_min_table,
        n_actions=metadata["n_actions"],
        transaction_cost_pct=metadata["transaction_cost_pct"],
        data_hash=metadata["data_hash"],
    )

def interp(v_row: np.ndarray, exposure: float, n_actions: int):
    """
    Calculates a bi-linear interpolation of a row from the value table using the exposure.
    """
    i0, i1 = get_low_high_exposure_idx(exposure, n_actions)
    x0 = get_exposure_val(i0, n_actions)
    x1 = get_exposure_val(i1, n_actions)
    if x0 == x1:  # shortcut and prevent div by zero
        return v_row[i0]
    a = (exposure - x0) / (x1 - x0)
    return (1 - a) * v_row[i0] + a * v_row[i1]

class DPRewardFunction:

    def __init__(self, table: DPTable, lam: int = 0.1, alpha_mu: float = 0.001, alpha_std: float = 0.001, clip: float = 5.0):
        # Reward computation
        self.v = table.value_table
        self.pi = table.policy_table
        self.q_min = table.q_min_table
        self.n_actions = table.n_actions
        self.actions = get_exposure_levels(table.n_actions)
        self.T = self.v.shape[0]
        self.c = table.transaction_cost_pct
        self.lam = lam # penalty weight

        # Z-score tracking (normalization)
        self.mu = 0.0
        self.var = 1.0  # Track variance, not std directly (for numerical stability)
        self.alpha_mu = alpha_mu
        self.alpha_std = alpha_std
        self.clip = clip

    def __call__(self, env) -> float:
        t = env.n_steps
        if t >= self.T - 1:
            return 0.0

        # Unwrap env.agent_data for exposures and equities
        curr_cash = env.agent_data[t-1, AgentDataCol.cash]
        curr_equity = env.agent_data[t, AgentDataCol.pre_action_equity]
        curr_exposure = (curr_equity - curr_cash) / curr_equity

        next_cash = env.agent_data[t, AgentDataCol.cash]
        next_equity = env.agent_data[t+1, AgentDataCol.pre_action_equity]
        next_exposure = (next_equity - next_cash) / next_equity

        # True one-step log-return
        true_log_return = np.log(next_equity / curr_equity)

        # Interpolated DP expected cumulative log-equity from current/next state
        exp_next = interp(self.v[t+1], next_exposure, self.n_actions)
        exp_curr = interp(self.v[t], curr_exposure, self.n_actions)

        # Baseline shaped reward (potential-based shaping)
        r = true_log_return + exp_next - exp_curr

        # Downside penalty: Difference to the worst DP action.
        i_curr = get_exposure_idx(curr_exposure, self.n_actions)
        q_dp = true_log_return + exp_next
        q_min_curr = self.q_min[t, i_curr]
        r += self.lam * (q_dp - q_min_curr)

        # Update running mean and variance
        delta = r - self.mu
        self.mu += self.alpha_mu * delta
        self.var = (1 - self.alpha_std) * self.var + self.alpha_std * delta**2
        std = np.sqrt(self.var + 1e-8)

        # Z-score scaling and clipping
        r = (r - self.mu) / std
        return np.clip(r, -self.clip, self.clip)
