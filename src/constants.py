from enum import IntEnum
from pathlib import Path

from torch.cuda import is_available as is_cuda_available

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DP_CACHE_DIR = DATA_DIR / "dp_cache"
MODELS_DIR = DATA_DIR / "models"

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
    open_ask = 1
    time_ns = 2

class AgentDataCol(ColumnCollection):
    """
    Agent Data Columns
    """
    cash = 0
    shares = 1
    eot_equity = 2 # equity at the end of the timeframe, right before the next action is taken.


DEVICE = "cuda" if is_cuda_available() else "cpu"