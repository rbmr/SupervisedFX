from pathlib import Path
from enum import Enum
import pandas as pd 
from ta import add_all_ta_features
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
import pandas as pd
from pandas.api.types import is_numeric_dtype
from itertools import product
from typing import NamedTuple
import os
from functools import total_ordering

DATA_DIR = Path(__file__).resolve().parent
COLUMNS = ["Gmt time", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DATE_FORMAT = "%d.%m.%YT%H.%M"

# Enum values are strings representing their component of the path

class Currency(Enum):
    EUR = "EUR" 
    AUD = "AUD"
    GBP = "GBP"
    NZD = "NZD"
    USD = "USD"
    CAD = "CAD"
    CHF = "CHF"
    JPY = "JPY"

class Granularity(Enum):
    M1 = "1_M" 
    M5 = "5_M"
    M15 = "15_M"
    H1 = "1_H"
    D1 = "1_D"

    def get_interval(self):
        """
        Gets the number of seconds corresponding to an granularity interval.
        """
        value, unit = self.value.split("_")
        value = int(value)
        if unit == "M":
            return value * 60
        if unit == "H":
            return value * (60 * 60)
        if unit == "D":
            return value * (24 * 60 * 60)
        raise ValueError("Unsupported granularity unit")

class OfferSide(Enum):
    ASK = "ASK"
    BID = "BID"

class DataConfig(NamedTuple):
    c1: Currency
    c2: Currency
    gran: Granularity
    off: OfferSide
    start: datetime
    end: datetime

    def to_path(self) -> Path:
        date_range = f"{self.start.strftime(DATE_FORMAT)}-{self.end.strftime(DATE_FORMAT)}"
        currency_pair = f"{self.c1.value}{self.c2.value}"
        return DATA_DIR / currency_pair / self.gran.value / self.off.value / f"{date_range}.csv"
    
    @staticmethod
    def from_path(path: Path):
        path = path.resolve()
        assert path.is_file(), "Path must point to a file."
        assert path.is_relative_to(DATA_DIR), f"File must be inside {DATA_DIR}."
        path = path.relative_to(DATA_DIR, walk_up=False)
        pair, gran, off, fname = path.parts
        assert len(pair) == 6, "Currency pairs must be 6 characters."
        c1 = Currency(pair[:3])
        c2 = Currency(pair[3:])
        gran = Granularity(gran)
        off = OfferSide(off)
        name, ext = os.path.splitext(fname)
        assert ext == ".csv"
        start, end = name.split('-')
        start = datetime.strptime(start, DATE_FORMAT)
        end = datetime.strptime(end, DATE_FORMAT)
        return DataConfig(c1, c2, gran, off, start, end)

def round_datetime(dt: datetime, interval: int):
    """
    Rounds a datetime object to the nearest interval in seconds.
    """
    start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_start = (dt - start_of_day).total_seconds()
    rounded_seconds = round(seconds_since_start / interval) * interval
    return start_of_day + timedelta(seconds=rounded_seconds)

def validate_df(df: pd.DataFrame):
    missing_columns = [col for col in COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}. The DataFrame must contain {", ".join(COLUMNS)} columns.")
    
    for col in NUMERIC_COLUMNS:
        if not is_numeric_dtype(df[col]):
            raise ValueError(f"Columns {col} must be numeric.")

def invert_pair(df: pd.DataFrame):

    validate_df(df)

    # Invert the exchange rates (1 / rate)
    df['Open'] = 1 / df['Open']
    df['High'] = 1 / df['Low']
    df['Low'] = 1 / df['High']
    df['Close'] = 1 / df['Close']
    
    return df

def gran_ratio(target_gran: Granularity, source_gran: Granularity):
    """
    Returns the ratio of the target granularity to the source granularity,
    assuming the target interval is an exact multiple of the source interval.
    """
    return exact_divide(target_gran.get_interval(), source_gran.get_interval())


def downscale_df(df: pd.DataFrame, n_rows: int):
    """
    Decrease granularity of a pandas dataframe by combining each n_rows into a single row.
    Leaves out the remaining rows if the number of rows is not divisible by n_rows.
    """
    arr = df.to_numpy()
    rows, cols = arr.shape

    print(arr[:5])
    print(rows, cols)

    df.shape
    
def exact_divide(a: int, b: int) -> int:
    """
    Divides a by b, if a is divisible by b.
    Otherwise it raises a ValueError.
    """
    if a % b == 0:
        return a // b
    raise ValueError(f"{a} is not divisible by {b}")
    
def get_data(config: DataConfig) -> pd.DataFrame:

    if config.start > config.end:
        raise ValueError(f"Start datetime ({config.start}) is after end datetime ({config.end}).")

    start = round_datetime(config.start, config.gran.get_interval())
    end = round_datetime(config.end, config.gran.get_interval())

    path = config.to_path()
    if not path.is_file():
        raise ValueError("Data not found.")
    df = pd.read_csv(path)
    return df

def preprocess(df) -> pd.DataFrame:
    """
    Add technical indicators and clean data.
    """
    validate_df(df)

    df = add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )
    
    # Fill missing values if any remain
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

if __name__ == "__main__":
    config = DataConfig(
        Currency.EUR,
        Currency.USD,
        Granularity.M1,
        OfferSide.BID,
        datetime(2022, 5, 10),
        datetime(2025, 5, 10)
    )
    df = get_data(config)

    downscale_df(df, 4)

