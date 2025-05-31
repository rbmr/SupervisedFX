from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path

import pytz
from torch.cuda import is_available

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FOREX_DIR = DATA_DIR / "forex"

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
    open_ask = 2
    high_bid = 3
    high_ask = 4
    low_bid = 5
    low_ask = 6
    close_bid = 7
    close_ask = 8
    volume_bid = 9
    date_gmt = 10

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

NUMERIC_DATA_COLUMNS = [RawDataCol.OPEN, RawDataCol.HIGH, RawDataCol.LOW, RawDataCol.CLOSE, RawDataCol.VOL]
DATA_COLUMNS = [RawDataCol.TIME, RawDataCol.OPEN, RawDataCol.HIGH, RawDataCol.LOW, RawDataCol.CLOSE, RawDataCol.VOL]

DT_TIMEZONE = timezone.utc # datetime timezone
PD_TIMEZONE = pytz.timezone("GMT") # pandas timezone

DATE_FORMAT = "%d.%m.%YT%H.%M"
CSV_TIME_FORMAT = "%d.%m.%Y %H:%M:%S.%f"

DEVICE = "cuda" if is_available() else "cpu"