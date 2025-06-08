from enum import IntEnum
from pathlib import Path

from torch.cuda import is_available as is_cuda_available

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

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
    date_gmt = 0
    open_bid = 1
    high_bid = 2
    low_bid = 3
    close_bid = 4
    open_ask = 5
    high_ask = 6
    low_ask = 7
    close_ask = 8
    volume = 9

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

class RawDataCol:
    """
    Class with static attributes to group column names in one place,
    helps prevent typos, and makes refactoring easier.
    """

    TIME = "date_gmt"
    VOL = "volume"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"

DEVICE = "cuda" if is_cuda_available() else "cpu"