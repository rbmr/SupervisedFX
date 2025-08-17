"""
Some scripts used for trading logic.
"""
import numpy as np
from numpy._typing import NDArray

def interpolate(x: NDArray, xp: NDArray, fp: NDArray) -> NDArray:
    """Performs 1D linear interpolation.

    Wraps numpy.interp to handle multidimensional inputs for `x` by flattening and reshaping.
    """
    return np.interp(x.flatten(), xp, fp).reshape(x.shape)

def calculate_shares_to_trade(prev_exposure: np.array, target_exposure: np.array, prev_cash: np.array, prev_shares: np.array, decision_bid: np.array, decision_ask: np.array, commission_pct: np.array):
    val_price = np.where(target_exposure >= 0, decision_bid, decision_ask)
    cost = np.where(target_exposure >= prev_exposure, (1 + commission_pct) * decision_ask, (1 - commission_pct) * decision_bid)
    tmp = val_price * (1 - target_exposure)
    return (target_exposure * prev_cash - prev_shares * tmp) / (tmp + target_exposure * cost)

def execute_trade(prev_cash: np.array, prev_shares: np.array, shares_to_trade: np.array, exec_bid: np.array, exec_ask: np.array, commission_pct: float):
    cost = np.where(shares_to_trade >= 0, (1 + commission_pct) * exec_ask, (1 - commission_pct) * exec_bid)
    return prev_cash - shares_to_trade * cost, prev_shares + shares_to_trade