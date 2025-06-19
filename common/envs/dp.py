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
from typing import Callable, Self

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from common.constants import MarketDataCol, DP_CACHE_DIR, AgentDataCol
from common.envs.forex_env import ForexEnv
from common.envs.trade import calculate_equity, execute_trade, reverse_equity

DATA_HASH_LENGTH = 16

@dataclass(frozen=True)
class DPTable:

    value_table: NDArray[np.float64] # shape: (n_timesteps, n_exposures)
    policy_table: NDArray[np.uint8] # shape: (n_timesteps, n_exposures)
    q_min_table: NDArray[np.float64] # shape: (n_timesteps, n_exposures)
    n_actions: int
    n_exposures: int
    n_timesteps: int
    transaction_cost_pct: float
    data_hash: str

    def save(self, path: Path):
        """
        Saves this DPTable to a .npz file.
        """
        # Validate input
        if path.suffix != '.npz':
            raise ValueError("File must be a .npz file.")

        # Setup metadata
        metadata = json.dumps({
            "transaction_cost_pct": self.transaction_cost_pct,
            "n_actions": self.n_actions,
            "n_exposures": self.n_exposures,
            "n_timesteps": self.n_timesteps,
            "data_hash": self.data_hash,
        }).encode("utf-8")

        # Save DPTable
        np.savez(path,
                 value=self.value_table,
                 policy=self.policy_table,
                 q_min=self.q_min_table,
                 metadata=metadata)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Loads this DPTable from a .npz file.
        """
        # Validate input
        if path.suffix != '.npz':
            raise ValueError("File must be a .npz file.")

        # Load data
        data = np.load(path)
        metadata = json.loads(data["metadata"].item())

        # Return DPTable
        return cls(
            value_table=data["value"],
            policy_table=data["policy"],
            q_min_table=data["q_min"],
            **metadata
        )

def get_bins(n: int) -> np.ndarray:
    """
    n -> {-1, -1 + 2/n, ..., 1}
    """
    return np.linspace(-1.0, 1.0, n, dtype=np.float64)

def get_bin_idx(x: float, n: int) -> int:
    """
    x in [-1, 1] -> i in [0, n-1]
    """
    return int((x + 1) * 0.5 * (n - 1) + 0.5)

def get_bin_val(i: int, n: int) -> float:
    """
    i in [0, n-1] -> x in [-1, 1]
    """
    return i / (n - 1.0) * 2.0 - 1.0

def interp(vs: np.ndarray, x: float, n: int):
    """
    Calculates a bi-linear interpolation of a row from the value table using an exposure x.
    """
    x = np.clip(x, -1.0, 1.0) # Could exit -1.0, to 1.0 because of floating point inaccuracies.
    pos = (x + 1) * 0.5 * (n - 1)
    i0 = math.floor(pos)
    i1 = math.ceil(pos)
    if i0 == i1: # shortcut and prevent div by zero
        return vs[i0]
    x0 = get_bin_val(i0, n)
    x1 = get_bin_val(i1, n)
    a = (x - x0) / (x1 - x0)
    return (1 - a) * vs[i0] + a * vs[i1]

def compute_dp_table(market_data: np.ndarray,
                     transaction_cost_pct: float = 0.0,
                     n_actions: int = 15,
                     n_exposures: int = 15,
                     data_hash: str | None = None) -> DPTable:
    """
    Compute the optimal value and policy tables using backward induction.
    - value_table[t, e]: the maximal expected cumulative log-equity from (t,e).
    - policy_table[t, e]: contains the best target exposure (action) for timestep t and exposure e.
    - q_min_table[t, e] the worst one-step continuation log-equity among all actions at (t,e).
    Where the equity ratio is defined as the ratio between the current equity and the equity at the end of the timestep.
    """
    # Validate input
    if not 1 <= n_actions <= 256:
        raise ValueError(f"n_actions must be in [1, 256], was {n_actions}")
    if not 1 <= n_exposures <= 256:
        raise ValueError(f"n_exposures must be in [1, 256], was {n_actions}")
    if not 0 <= transaction_cost_pct <= 1:
        raise ValueError(f"transaction_cost_pct must be in [0, 1], was {transaction_cost_pct}")

    # Setup variables
    exposures = get_bins(n_exposures)
    actions = get_bins(n_actions)
    n_timesteps = market_data.shape[0]
    data_hash = data_hash if data_hash is not None else get_data_hash(market_data)

    logging.info(f"Generating {get_dp_table_name(data_hash, transaction_cost_pct, n_actions, n_exposures)}")

    # Initialize tables
    value_table = np.empty((n_timesteps, n_exposures), dtype=np.float64)
    policy_table = np.empty((n_timesteps, n_exposures), dtype=np.uint8)
    q_min_table = np.empty((n_timesteps, n_exposures), dtype=np.float64)

    # Set terminals
    value_table[-1, :] = 0
    policy_table[-1, :] = np.arange(n_exposures, dtype=np.uint8)
    q_min_table[-1, :] = 0

    # Perform backward in to fill the tables
    for t in trange(n_timesteps - 2, -1, -1):  # Skip terminal

        # Get prices
        curr_bid = market_data[t, MarketDataCol.open_bid]
        curr_ask = market_data[t, MarketDataCol.open_ask]
        next_bid = market_data[t + 1, MarketDataCol.open_bid]
        next_ask = market_data[t + 1, MarketDataCol.open_ask]

        for i, curr_exposure in enumerate(exposures):

            # Get cash, and shares just BEFORE executing the trade.
            curr_cash, curr_shares = reverse_equity(curr_bid, curr_ask, 1, curr_exposure)  # type: ignore

            # Set defaults
            best_val = -np.inf
            worst_val = np.inf
            best_j = 0

            # Try all possible actions
            for j, target_exposure in enumerate(actions):

                # Get cash, and shares just AFTER executing the trade using target exposure.
                next_cash, next_shares = execute_trade(target_exposure, curr_bid, curr_ask, # type: ignore
                                                       curr_cash, curr_shares, transaction_cost_pct)

                # Get equity just BEFORE executing the next trade.
                next_equity = calculate_equity(next_bid, next_ask, next_cash, next_shares) # type: ignore
                next_exposure = (next_equity - next_cash) / next_equity

                # next_equity is equity ratio, since curr_equity was 1.
                val = np.log(next_equity) + interp(value_table[t+1], next_exposure, n_exposures)

                # Compare actions
                if val > best_val:
                    best_val = val
                    best_j = j
                if val < worst_val:
                    worst_val = val

            # Store results
            value_table[t, i] = best_val
            policy_table[t, i] = best_j
            q_min_table[t, i] = worst_val

    return DPTable(
        value_table=value_table,
        policy_table=policy_table,
        q_min_table=q_min_table,
        n_actions=n_actions,
        n_exposures=n_exposures,
        n_timesteps=n_timesteps,
        transaction_cost_pct=transaction_cost_pct,
        data_hash=data_hash
    )

def get_optimal_action(table: DPTable, t: int, current_exposure: float) -> float:
    """
    Given the current exposure and timestep, returns the optimal target exposure according to the DPTable.
    """
    if t >= table.n_timesteps:
        raise ValueError("t is out of bounds.")
    exposure_idx = get_bin_idx(current_exposure, table.n_exposures)
    action_idx = table.policy_table[t, exposure_idx]
    return get_bin_val(action_idx, table.n_actions) # type: ignore

def get_optimal_action_fn(table: DPTable, env: ForexEnv) -> Callable[[...], float]:
    """
    Given a DPTable and a ForexEnv, creates a function that retrieves the optimal action
    by retrieving the state from the env and reading the table.
    """
    assert table.n_timesteps == env.data_len
    assert env.n_actions == 0 or table.n_actions == env.n_actions
    def predict(*_) -> float:
        current_cash = env.agent_data[env.n_steps, AgentDataCol.cash]
        current_equity = env.agent_data[env.n_steps + 1, AgentDataCol.pre_action_equity]
        current_exposure = (current_equity - current_cash) / current_equity
        return get_optimal_action(table, env.n_steps + 1, current_exposure)
    return predict

def get_dp_table_from_env(env: ForexEnv, cache_dir: Path | None = None, n_exposures: int = 15) -> DPTable:
    return get_dp_table(
        market_data = env.market_data,
        transaction_cost_pct = env.transaction_cost_pct,
        n_actions = env.n_actions if env.n_actions != 0 else 15,
        n_exposures = n_exposures,
        cache_dir = cache_dir
    )

def get_dp_table(market_data: np.ndarray,
                 transaction_cost_pct: float,
                 n_actions: int,
                 n_exposures: int = 15,
                 cache_dir: Path | None = None) -> DPTable:
        """
        Gets the dp table from cache if it exists, otherwise computes it.
        """
        if cache_dir is None:
            cache_dir = DP_CACHE_DIR
        data_hash = get_data_hash(market_data)
        table_name = get_dp_table_name(data_hash, transaction_cost_pct, n_actions, n_exposures)
        table_file = cache_dir / f"{table_name}.npz"
        if table_file.exists():
            return DPTable.load(table_file)
        table = compute_dp_table(market_data, transaction_cost_pct, n_actions, n_exposures, data_hash)
        table.save(table_file)
        return table

def get_data_hash(market_data: np.ndarray):
    return hashlib.sha512(market_data.tobytes()).hexdigest()[:DATA_HASH_LENGTH]

def get_dp_table_name(data_hash: str,
                      transaction_cost_pct: float,
                      n_actions: int,
                      n_exposures: int):
    tc_string = f"tc{transaction_cost_pct:.4f}".replace(".", "p")
    return f"dp_table_a{n_actions}_e{n_exposures}_{tc_string}_data{data_hash}"




