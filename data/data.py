from pathlib import Path
from enum import Enum
import pandas as pd 
from ta import add_all_ta_features
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import NamedTuple
import os

DATA_DIR = Path(__file__).resolve().parent
FOREX_DIR = DATA_DIR / "forex"
TIME_COL = "Gmt time"
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
COLUMNS = [TIME_COL,] + NUMERIC_COLUMNS
DT_TIMEZONE = timezone.utc # same as gmt
PD_TIMEZONE = "GMT"

DATE_FORMAT = "%d.%m.%YT%H.%M"
CSV_TIME_FORMAT = "%d.%m.%Y %H:%M:%S.%f"

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
    M1 = "1M" 
    M5 = "5M"
    M15 = "15M"
    H1 = "1H"
    D1 = "1D"

    def get_interval(self):
        """
        Gets the number of seconds corresponding to an granularity interval.
        """
        unit = self.value[-1]
        value = int(self.value[:-1])
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

class ForexRef(NamedTuple):
    """
    Class used for parsing, manipulating, and generating forex data file paths.
    """

    c1: Currency
    c2: Currency
    gran: Granularity
    off: OfferSide
    start: datetime
    end: datetime

    def get_path(self) -> Path:
        date_range = f"{self.start.strftime(DATE_FORMAT)}-{self.end.strftime(DATE_FORMAT)}"
        currency_pair = f"{self.c1.value}{self.c2.value}"
        return FOREX_DIR / currency_pair / self.gran.value / self.off.value / f"{date_range}.csv"
    
    @staticmethod
    def from_path(path: Path):
        path = path.resolve()
        assert path.is_file(), "Path must point to a file."
        assert path.is_relative_to(FOREX_DIR), f"File must be inside {FOREX_DIR}."
        path = path.relative_to(FOREX_DIR, walk_up=False)
        pair, gran, off, fname = path.parts
        assert len(pair) == 6, "Currency pairs must be 6 characters, ex: EURUSD, GBPUSD, etc"
        c1 = Currency(pair[:3])
        c2 = Currency(pair[3:])
        gran = Granularity(gran)
        off = OfferSide(off)
        name, ext = os.path.splitext(fname)
        assert ext == ".csv"
        start, end = name.split('-')
        start = datetime.strptime(start, DATE_FORMAT)
        start = start.replace(tzinfo = DT_TIMEZONE)
        end = datetime.strptime(end, DATE_FORMAT)
        end = end.replace(tzinfo = DT_TIMEZONE)
        return ForexRef(c1, c2, gran, off, start, end)

    def load(self):
        return ForexData(self)

    def invert_pair(self):
        return ForexRef(
            c1 = self.c2,
            c2 = self.c1, 
            gran = self.gran,
            off = self.off,
            start = self.start,
            end = self.end
        )

class ForexData:
    """
    Class used for loading, validating, processing, and saving forex data files.
    """

    def __init__(self, ref: ForexRef | Path | str, df: pd.DataFrame = None):
        
        # Convert ref to ForexRef if possible
        if isinstance(ref, str):
            ref = Path(ref)
        if isinstance(ref, Path):
            ref = ForexRef.from_path(ref)
        if not isinstance(ref, ForexRef):
            raise TypeError("Reference type is not supported")

        # Load dataframe if not loaded yet
        if df is None:
            path = ref.get_path()
            if not path.is_file():
                raise ValueError("Forex data not found.")
            df = pd.read_csv(path)
        
        # Validate columns
        actual_columns = set(df.columns)
        expected_columns = set(COLUMNS)
        if actual_columns != expected_columns:
            raise ValueError(f"Dataframe has columns {actual_columns}, expected: {expected_columns}")

        # Ensure correct order
        df = df[COLUMNS]

        # Check if numeric columns are numeric
        for col in NUMERIC_COLUMNS:
            if not is_numeric_dtype(df[col]):
                raise ValueError(f"Columns {col} must be numeric.")

        # Check if time column is set correctly and convert to datetime
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=CSV_TIME_FORMAT)
        df[TIME_COL] = df[TIME_COL].dt.tz_localize(PD_TIMEZONE)

        # Check if start and end is correct
        assert df[TIME_COL].iloc[0] == ref.start 
        assert df[TIME_COL].iloc[-1] == ref.end

        # Check if granularity is correct
        time_delta = pd.Timedelta(seconds = ref.gran.get_interval())
        deltas = df[TIME_COL].diff().dropna()
        mode_delta = deltas.mode().iloc[0]
        assert (mode_delta == time_delta)

        # set attributes
        self.df = df
        self.ref = ref

    def invert_pair(self):
        """
        Invert the exchange rates (1 / rate). For example:
        EUR/USD -> USD/EUR, GBP/USD -> USD/GBP, etc
        """
        self.df['Open'] = 1 / self.df['Open']
        self.df['High'] = 1 / self.df['Low']
        self.df['Low'] = 1 / self.df['High']
        self.df['Close'] = 1 / self.df['Close']
        self.ref = self.ref.invert_pair()
        return self
        
    def set_gran(self, target_gran: Granularity):
        """
        Sets the granularity of the ForexData to the target_gran.
        Leaves out remaining rows.
        """
        raise NotImplementedError("Setting granularity is not implemented.")

        # source_gran = self.ref.gran
        # if target_gran.get_interval() < source_gran.get_interval():
        #     raise ValueError("Granularity increasing is not supported.")
        # n_rows = exact_divide(target_gran.get_interval(), source_gran.get_interval())
        
        # arr = self.df.to_numpy()
        # rows, cols = arr.shape

        # print(arr[:5])
        # print(rows, cols)

        # return self

    def set_period(self, start: datetime = None, end: datetime = None):
        if start is None:
            start = self.ref.start 
        if end is None:
            end = self.ref.end
        pd_start = pd.to_datetime(start) 
        pd_end = pd.to_datetime(end)
        self.df = self.df[self.df[TIME_COL] >= pd_start]
        self.df = self.df[self.df[TIME_COL] <= pd_end]
        self.ref = ForexRef(self.ref.c1, self.ref.c2, self.ref.gran, self.ref.off, start, end)
        return self
    
    def save(self):
        if self.ref is None:
            raise ValueError("Forex data is processed, can not be saved.")
        path = self.ref.get_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return self

    def preprocess(self):

        self.df = add_all_ta_features(
            self.df,
            open="Open", high="High", low="Low", close="Close", volume="Volume",
            fillna=True
        )
    
        # Fill missing values if any remain
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        # Mark this object as processed, and therefore not allowed to be saved
        # You can still save it yourself by extracting the dataframe.
        self.ref = None


def round_datetime(date_time: datetime, interval: int) -> datetime:
    """
    Rounds a datetime object to the nearest multiple of `interval` in seconds.
    """
    start_of_day = date_time.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_start = (date_time - start_of_day).total_seconds()
    rounded_seconds = round(seconds_since_start / interval) * interval
    return start_of_day + timedelta(seconds=rounded_seconds)

def exact_divide(a: int, b: int) -> int:
    """
    Performs an exact division of `a` by `b`, returning an integer.
    Raises a ValueError, if `a` is not divisible by `b`.
    """
    if a % b == 0:
        return a // b
    raise ValueError(f"{a} is not divisible by {b}")


if __name__ == "__main__":
    
    fd = ForexData("C:\\Users\\rober\\TUD-CSE-RP-RLinFinance\\data\\forex\\EURUSD\\15M\\BID\\10.05.2022T00.00-09.05.2025T20.45.csv")
    fd.set_period(end = datetime(2023, 5, 10, tzinfo=DT_TIMEZONE)).save()



