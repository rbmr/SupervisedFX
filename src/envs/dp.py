"""
Dynamic Programming Table for Forex Trading Optimization

This module computes the optimal trading strategy using dynamic programming
by working backwards from the final timestep to determine the best action
at each state (timestep, exposure_level).
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from tqdm import trange

from src.constants import DP_CACHE_DIR
from src.envs.forex_env import ForexEnv
from src.envs.trade import execute_trade_1equity

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
        assert path.suffix == ".npz", "File must be a .npz file."
        metadata = json.dumps({
            "transaction_cost_pct": self.transaction_cost_pct,
            "n_actions": self.n_actions,
            "n_exposures": self.n_exposures,
            "n_timesteps": self.n_timesteps,
            "data_hash": self.data_hash,
        }).encode("utf-8")
        np.savez(path, value=self.value_table, policy=self.policy_table,
                 q_min=self.q_min_table, metadata=metadata)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Loads this DPTable from a .npz file.
        """
        assert path.exists(), f"File {path} does not exist."
        assert path.suffix == ".npz", "File must be a .npz file."
        data = np.load(path)
        metadata = json.loads(data["metadata"].item())
        return cls(value_table=data["value"], policy_table=data["policy"],
                   q_min_table=data["q_min"], **metadata)

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

@tf.function
def interp(vs: tf.Tensor, x: tf.Tensor, n: int) -> tf.Tensor:
    """
    Calculates a bi-linear interpolation of rows from a value table using a batch of exposures.
    """
    x = tf.clip_by_value(x, -1.0, 1.0)
    pos = (x + 1.0) * 0.5 * (tf.cast(n, tf.float32) - 1.0)
    i0_f = tf.floor(pos)
    i1_f = tf.math.ceil(pos)
    i0 = tf.cast(i0_f, tf.int32)
    i1 = tf.cast(i1_f, tf.int32)
    x0 = tf.cast(i0, tf.float32) / (tf.cast(n, tf.float32) - 1.0) * 2.0 - 1.0
    x1 = tf.cast(i1, tf.float32) / (tf.cast(n, tf.float32) - 1.0) * 2.0 - 1.0
    a = tf.math.divide_no_nan((x - x0), (x1 - x0))
    v0 = tf.gather(vs, i0, batch_dims=1)
    v1 = tf.gather(vs, i1, batch_dims=1)
    return (1.0 - a) * v0 + a * v1


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
    # Setup
    assert 1 <= n_actions <= 256, f"n_actions must be in [1, 256], was {n_actions}"
    assert 1 <= n_exposures <= 256, f"n_exposures must be in [1, 256], was {n_actions}"
    assert 0 <= transaction_cost_pct <= 1, f"transaction_cost_pct must be in [0, 1], was {transaction_cost_pct}"
    exposures = get_bins(n_exposures)
    actions = get_bins(n_actions)
    n_timesteps = market_data.shape[0]
    data_hash = data_hash if data_hash is not None else get_data_hash(market_data)
    logging.info(f"Generating {get_dp_table_name(data_hash, transaction_cost_pct, n_actions, n_exposures)}")

    # Setup tensors
    market_data = tf.constant(market_data, dtype=tf.float32)
    exposures = tf.constant(exposures.reshape(1, -1), dtype=tf.float32) # (1, n_exposures)
    actions = tf.constant(actions.reshape(-1, 1), dtype=tf.float32) # (n_actions, 1)

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

        # We want to compute the Q-value for every possible (action, exposure) pair.
        # Shape: (n_actions, n_exposures)
        curr_exposure_grid = tf.tile(exposures, [n_actions, 1])
        target_exposure_grid = tf.tile(actions, [1, n_exposures])

        # Reshape for batch operations
        # Shape: (n_actions * n_exposures, 1)
        current_exposure = tf.reshape(curr_exposure_grid, [-1, 1])
        target_exposure = tf.reshape(target_exposure_grid, [-1, 1])
        indices_batch = tf.ones_like(current_exposure, dtype=tf.int32) * t

        # Simulate one step for all pairs
        next_equity, next_exposure = execute_trade_1equity(indices_batch, current_exposure, target_exposure, market_data, transaction_cost_pct)

        # Interpolate V(t+1) for all resulting exposures
        v_next_slice = tf.constant(value_table[t+1].reshape(1, -1), dtype=tf.float32)
        v_next_slice_batch = tf.tile(v_next_slice, [tf.shape(next_exposure)[0], 1])
        v_next = interp(v_next_slice_batch, next_exposure, n_exposures)

        # Calculate Q-values for all (action, exposure) pairs.
        reward = tf.math.log(tf.maximum(next_equity, 1e-9))
        q_values = reward + v_next
        q_values_grid = tf.reshape(q_values, [n_actions, n_exposures]) # Shape: (n_actions, n_exposures)

        # Find optimal values and policies for this timestep
        value_table[t, :] = tf.reduce_max(q_values_grid, axis=0).numpy()
        policy_table[t, :] = tf.argmax(q_values_grid, axis=0, output_type=tf.int32).numpy()
        q_min_table[t, :] = tf.reduce_min(q_values_grid, axis=0).numpy()

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
    policy_row = tf.constant(table.policy_table[t].reshape(1, -1), dtype=tf.float32)
    exposure = tf.constant([[current_exposure]], dtype=tf.float32)
    optimal_action_idx = round(interp(policy_row, exposure, table.n_exposures).numpy()[0,0])
    return get_bin_val(optimal_action_idx, table.n_actions)

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




