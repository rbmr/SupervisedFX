import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy._typing import NDArray
from tqdm import trange

from src.constants import DP_CACHE_DIR, Price, COMMISSION_PCT
from src.trade.trade import trade, norm_linspace_interp

DATA_HASH_LENGTH = 8

@dataclass(frozen=True)
class DPTable:

    value_table: NDArray[np.float64] # shape: (n_timesteps+1, n_exposures)
    policy_table: NDArray[np.uint8] # shape: (n_timesteps, n_exposures)
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

    @staticmethod
    def _calculate_data_hash(prices: NDArray[np.float64]) -> str:
        """Helper method to compute the data hash."""
        return hashlib.sha512(prices.tobytes()).hexdigest()[:DATA_HASH_LENGTH]

    @classmethod
    def _get_cache_path(cls, n_actions: int, n_exposures: int, commission_pct: float, data_hash: str) -> Path:
        """Helper method to generate a consistent filename."""
        commission_str = str(commission_pct * 100).replace(".", "p")
        filename = f"dp_table_a{n_actions}_e{n_exposures}_c{commission_str}_{data_hash}.npz"
        return DP_CACHE_DIR / filename

    @classmethod
    def get(cls,
            prices: np.ndarray,
            commission_pct: float = COMMISSION_PCT,
            n_actions: int = 15,
            n_exposures: int = 15
            ) -> Self:
        """
        Gets a DPTable, loading from cache if available, otherwise computing and saving it.
        """
        data_hash = cls._calculate_data_hash(prices)
        cache_path = cls._get_cache_path(n_actions, n_exposures, commission_pct, data_hash)
        if cache_path.exists():
            print(f"✅ Found cached DPTable, loading from {cache_path}")
            return cls.load(cache_path)
        print(f"Cache not found. Computing DPTable for hash {data_hash}...")
        dp_table = cls.compute(prices, commission_pct, n_actions, n_exposures, data_hash)
        print(f"💾 Saving newly computed DPTable to {cache_path}")
        dp_table.save(cache_path)
        return dp_table

    @classmethod
    def compute(cls,
                prices: np.ndarray,
                commission_pct: float = COMMISSION_PCT,
                n_actions: int = 15,
                n_exposures: int = 15,
                _data_hash: str = None
                ) -> Self:
        assert prices.shape[1] == len(Price)
        data_hash = _data_hash if _data_hash is not None else cls._calculate_data_hash(prices)
        n_timesteps = prices.shape[0]
        assert 0.0 <= commission_pct <= 1.0, f"commission_pct must be in [0.0, 1.0], was {commission_pct}."
        assert 1 <= n_actions <= 256, f"n_actions must be in [1, 256], was {n_actions}"
        assert 1 <= n_exposures <= 256, f"n_exposures must be in [1, 256], was {n_exposures}"
        assert np.all(prices >= 0), "prices must be non-negative."

        prev_exposure = np.linspace(-1.0, 1.0, n_exposures)[:, np.newaxis] # \epsilon_{t-1,c} AKA \epsilon
        target_exposure = np.linspace(-1.0, 1.0, n_actions)[np.newaxis, :] # \epsilon_{t-1,c}' AKA a
        # e_{t-1,c} AKA e = 1
        prev_cash = 1 - prev_exposure # C_{t-1}
        prev_pval = prev_exposure # p_{t-1,c}

        v_table = np.zeros((n_timesteps+1, n_exposures), dtype=np.float64)
        pi_table = np.zeros((n_timesteps, n_exposures), dtype=np.int8)

        for t in trange(n_timesteps-1, -1, -1):

            # Retrieve relevant prices
            decision_bid = prices[t-1, Price.CLOSE_BID] if t > 0 else prices[0, Price.OPEN_BID] # P_{t-1,c}^b
            decision_ask = prices[t-1, Price.CLOSE_ASK] if t > 0 else prices[0, Price.OPEN_ASK] # P_{t-1,c}^a
            exec_bid = prices[t, Price.EXEC_BID] # P_t^b
            exec_ask = prices[t, Price.EXEC_ASK] # P_t^a
            close_bid = prices[t, Price.CLOSE_BID] # P_{t,c}^b
            close_ask = prices[t, Price.CLOSE_ASK] # P_{t,c}^a

            # Compute shares (cash is already known)
            prev_val_price = np.where(prev_exposure >= 0, decision_bid, decision_ask) # P_{t-1,c}^*
            prev_shares = prev_pval / prev_val_price # S_{t-1}

            # Execute order
            cash, shares, _, _, exposure, log_equity = trade(
                prev_exposure, target_exposure, prev_cash, prev_shares,
                decision_bid, decision_ask, exec_bid, exec_ask, close_bid, close_ask,
                commission_pct
            )

            # Determine reward, interpolate next state value
            v_next_interp = norm_linspace_interp(exposure, v_table[t + 1]) # V^\text{DP}_{t+1}(\epsilon')
            q_values = log_equity + v_next_interp # Q_t^\text{DP}
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