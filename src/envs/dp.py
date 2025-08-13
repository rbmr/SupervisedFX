import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np
from numpy._typing import NDArray
from tqdm import trange

DATA_HASH_LENGTH = 8

class Price(IntEnum):
    OPEN_BID = 0
    OPEN_ASK = 1
    HIGH_BID = 2
    HIGH_ASK = 3
    LOW_BID = 4
    LOW_ASK = 5
    CLOSE_BID = 6
    CLOSE_ASK = 7
    REF_ASK = 8
    REF_BID = 9
    EXEC_ASK = 10
    EXEC_BID = 11

@dataclass(frozen=True)
class DPTable:

    value_table: NDArray[np.float64] # shape: (n_timesteps+1, n_exposures)
    policy_table: NDArray[np.uint8] # shape: (n_timesteps+1, n_exposures)
    n_actions: int
    n_exposures: int
    n_timesteps: int
    commission_pct: float
    data_hash: str

    def save(self, path: Path):
        """
        Saves this DPTable to a .npz file.
        """
        assert path.suffix == ".npz", "File must be a .npz file."
        metadata = json.dumps({
            "commission_pct": self.commission_pct,
            "n_actions": self.n_actions,
            "n_exposures": self.n_exposures,
            "n_timesteps": self.n_timesteps,
            "data_hash": self.data_hash,
        }).encode("utf-8")
        np.savez(path,
                 value=self.value_table,
                 policy=self.policy_table,
                 metadata=metadata)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Loads this DPTable from a .npz file.
        """
        assert path.exists(), f"File {path} does not exist."
        assert path.suffix == ".npz", "File must be a .npz file."
        data = np.load(path)
        metadata = json.loads(data["metadata"].item())
        return cls(value_table=data["value"],
                   policy_table=data["policy"],
                   **metadata)

    @classmethod
    def compute(cls,
                prices: np.ndarray,
                commission_pct: float = 0.0,
                n_actions: int = 15,
                n_exposures: int = 15
                ) -> Self:
        assert prices.ndim == 2, "prices must be a 2D array."
        assert prices.shape[1] == len(Price), f"prices must have {len(Price)} columns."
        n_timesteps = prices.shape[0]
        assert 0.0 <= commission_pct <= 1.0, f"commission_pct must be in [0.0, 1.0], was {commission_pct}."
        assert 1 <= n_actions <= 256, f"n_actions must be in [1, 256], was {n_actions}"
        assert 1 <= n_exposures <= 256, f"n_exposures must be in [1, 256], was {n_actions}"
        assert np.all(prices >= 0), "prices must be non-negative."

        data_hash = hashlib.sha512(prices.tobytes()).hexdigest()[:DATA_HASH_LENGTH]

        exposures = np.linspace(-1.0, 1.0, n_exposures)
        prev_exposure = exposures[:, np.newaxis] # \epsilon_{t-1,c}' AKA \epsilon
        target_exposure = exposures[np.newaxis, :] # \tilde{\epsilon}_t' AKA a_t
        # e_{t-1,c}' = 1 (equity is normalized to 1)
        prev_cash = 1 - prev_exposure # C_{t-1}
        prev_pval = prev_exposure # p_{t-1,c}'

        v_table = np.zeros((n_timesteps+1, n_exposures), dtype=np.float64)
        pi_table = np.zeros((n_timesteps, n_exposures), dtype=np.int8)

        for t in trange(n_timesteps-1, -1, -1):

            # Retrieve relevant prices
            prev_close_bid = prices[t-1, Price.CLOSE_BID] if t > 0 else prices[0, Price.CLOSE_BID]
            prev_close_ask = prices[t-1, Price.CLOSE_ASK] if t > 0 else prices[0, Price.CLOSE_ASK]
            ref_bid = prices[t, Price.REF_BID] # \tilde{P}_t^b
            ref_ask = prices[t, Price.REF_ASK] # \tilde{P}_t^a
            exec_bid = prices[t, Price.EXEC_BID] # P_t^b
            exec_ask = prices[t, Price.EXEC_ASK] # P_t^a
            close_bid = prices[t, Price.CLOSE_BID] # P_{t,c}^b
            close_ask = prices[t, Price.CLOSE_ASK] # P_{t,c}^a

            # Compute shares (cash is already known)
            prev_val_price = np.where(prev_exposure >= 0, prev_close_bid, prev_close_ask) # P_{t-1,c}'^*
            prev_shares = prev_pval / prev_val_price # S_{t-1}

            # Derive shares to trade
            var18 = calculate_shares_to_trade(prev_exposure, target_exposure, prev_cash, prev_shares, ref_bid, ref_ask, commission_pct) # \Delta S_t

            # Execute trade
            cash, shares = execute_trade(prev_cash, prev_shares, var18, exec_bid, exec_ask, commission_pct)

            # Determine next equity and exposure
            val_price = np.where(shares >= 0, close_bid, close_ask) # P_{t,c}'^*
            pval = shares * val_price # p_{t,c}'
            equity = pval + cash # e_{t,c}' AKA e'
            is_solvent = equity > 0 # determine non-bankrupt states
            exposure = np.zeros_like(equity)
            exposure[is_solvent] = pval[is_solvent] / equity[is_solvent] # \epsilon'_{t,c}
            reward = np.full_like(equity, -np.inf)
            reward[is_solvent] = np.log(equity[is_solvent]) # r_t

            # Determine reward, interpolate next state value
            v_next_interp = interpolate(exposure, exposures, v_table[t + 1]) # V^\text{DP}_{t+1}(\epsilon')
            v_next_interp[~is_solvent] = -np.inf
            q_values = reward + v_next_interp # Q_t^\text{DP}
            q_values = np.nan_to_num(q_values, nan=-np.inf, neginf=-np.inf)
            v_table[t] = np.max(q_values, axis=1) # V_t^\text{DP}
            pi_table[t] = np.argmax(q_values, axis=1).astype(np.uint8) # \pi_t^\text{DP}

        return cls(
            value_table=v_table,
            policy_table=pi_table,
            n_actions=n_actions,
            n_exposures=n_exposures,
            n_timesteps=n_timesteps,
            commission_pct=commission_pct,
            data_hash=data_hash,
        )

def interpolate(x: NDArray, xp: NDArray, fp: NDArray) -> NDArray:
    """Performs 1D linear interpolation.

    Wraps numpy.interp to handle multidimensional inputs for `x` by flattening and reshaping.
    """
    return np.interp(x.flatten(), xp, fp).reshape(x.shape)

def calculate_shares_to_trade(prev_exposure, target_exposure, prev_cash, prev_shares, ref_bid, ref_ask, commission_pct):
    """"""
    val_price = np.where(target_exposure >= 0, ref_bid, ref_ask)  # \tilde{P}_t'^*
    cost = np.where(target_exposure >= prev_exposure, (1 + commission_pct) * ref_ask, (1 - commission_pct) * ref_bid)  # \tilde{\varphi}_t
    tmp = val_price * (1 - target_exposure)  # \tilde{P}_t'^*(1-a_t)
    return (target_exposure * prev_cash - prev_shares * tmp) / (tmp + target_exposure * cost) # \Delta S_t

def execute_trade(prev_cash, prev_shares, shares_to_trade, exec_bid, exec_ask, commission_pct):
    cost = np.where(shares_to_trade >= 0, (1 + commission_pct) * exec_ask, (1 - commission_pct) * exec_bid) # \varphi_t
    return prev_cash - shares_to_trade * cost, prev_shares + shares_to_trade # C_t, S_t