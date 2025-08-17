from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import Optional

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_REPORTS_DIR = DATA_DIR / "reports"
DP_CACHE_DIR = DATA_DIR / "dp_cache"
MODELS_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"

SEED = 42

class Columns(IntEnum):
    """Enum superclass for all column name Enums"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.names = tuple(col.name.lower() for col in cls)

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """Causes auto() to count from 0."""
        return count

class Price(Columns):
    """Enum defining all CandleData columns and their order."""

    OPEN_BID = auto()
    OPEN_ASK = auto()
    HIGH_BID = auto()
    HIGH_ASK = auto()
    LOW_BID = auto()
    LOW_ASK = auto()
    CLOSE_BID = auto()
    CLOSE_ASK = auto()
    EXEC_BID = auto()
    EXEC_ASK = auto()
    VOLUME = auto()

class Account(Columns):

    CASH = auto()
    SHARES = auto()
    CLOSE_EQUITY = auto()
    CLOSE_EXPOSURE = auto()
    CLOSE_PVAL = auto()

class Timeframe(Enum):
    """Enum for different trading timeframes."""
    TICK = (None, None, "TICK")
    M1 = ("1Min", 1, "1M")
    M5 = ("5Min", 5, "5M")
    M15 = ("15Min", 15, "15M")
    M30 = ("30Min", 30, "30M")
    H1 = ("H", 60, "1H")
    H4 = ("4H", 240, "4H")
    D1 = ("D", 1440, "1D")

    def __init__(self, pandas_freq: Optional[str], minutes: Optional[int], pathname: str):
        self.pandas_freq = pandas_freq
        self.minutes = minutes
        self.pathname = pathname