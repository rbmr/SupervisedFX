from pathlib import Path
from datetime import datetime, timedelta, timezone
import pytz
import torch 

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FOREX_DIR = DATA_DIR / "forex"

SEED = 42

class Col:
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

NUMERIC_DATA_COLUMNS = [Col.OPEN, Col.HIGH, Col.LOW, Col.CLOSE, Col.VOL]
DATA_COLUMNS = [Col.TIME, Col.OPEN, Col.HIGH, Col.LOW, Col.CLOSE, Col.VOL]

DT_TIMEZONE = timezone.utc # datetime timezone
PD_TIMEZONE = pytz.timezone("GMT") # pandas timezone

DATE_FORMAT = "%d.%m.%YT%H.%M"
CSV_TIME_FORMAT = "%d.%m.%Y %H:%M:%S.%f"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"