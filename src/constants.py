from enum import IntEnum
from pathlib import Path

from torch.cuda import is_available as is_cuda_available

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DP_CACHE_DIR = DATA_DIR / "dp_cache"
MODEL_DIR = DATA_DIR / "models"

SEED = 42

class ColumnCollection(IntEnum):

    @classmethod
    def all_names(cls) -> list[str]:
        """Returns all the names, in order of value"""
        return [x.name for x in sorted(cls, key=lambda x: x.value)]

class MarketDataCol(ColumnCollection):
    """
    Market Data Columns
    """
    open_bid = 0
    high_bid = 1
    low_bid = 2
    close_bid = 3
    open_ask = 4
    high_ask = 5
    low_ask = 6
    close_ask = 7
    volume = 8

class AgentDataCol(ColumnCollection):
    """
    Agent Data Columns
    """
    cash = 0
    shares = 1
    equity_open = 2
    equity_high = 3
    equity_low = 4
    equity_close = 5
    target_exposure = 6
    pre_action_equity = 7

DEVICE = "cuda" if is_cuda_available() else "cpu"