import sys
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
import numpy as np
from common.scripts import *

DATA_DIR = Path(__file__).resolve().parent
FOREX_DIR = DATA_DIR / "forex"
TIME_COL = "date_gmt"
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
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

class ForexData:
    """
    Class used for loading, validating, converting, and saving forex data files.
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

        # Check if time column is formatted correctly, and convert it to datetime
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=CSV_TIME_FORMAT)
        df[TIME_COL] = df[TIME_COL].dt.tz_localize(PD_TIMEZONE)
        
        # sort rows based on increasing time
        df.sort_values(by=TIME_COL, inplace=True)

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
        self.ref = ForexRef(
            c1 = self.ref.c2,
            c2 = self.ref.c1, 
            gran = self.ref.gran,
            off = self.ref.off,
            start = self.ref.start,
            end = self.ref.end
        )
        return self

    def set_gran(self, target_gran: Granularity):
        """
        Sets the granularity of the ForexData to the target_gran.
        Leaves out remaining rows.
        """

        source_gran = self.ref.gran
        if target_gran.get_interval() < source_gran.get_interval():
            raise ValueError("Granularity increasing is not supported.")
        n_rows = exact_divide(target_gran.get_interval(), source_gran.get_interval())
        
        arr = self.df.to_numpy()
        rows, cols = arr.shape

        batches = rows // n_rows
        arr = arr[:(batches*n_rows), :]
        arr = arr.reshape( (batches, n_rows, cols) )
        
        funcs = [
            lambda x: x[0], # gmt_time
            lambda x: x[0], # open
            np.max, # high
            np.min, # low
            lambda x: x[-1], # close
            np.sum # volume
        ]

        arr = np.array([
            [funcs[col](arr[batch, :, col]) for col in range(cols)]
            for batch in range(batches)
        ])

        self.arr = arr
        self.ref = ForexRef(
            c1 = self.ref.c1,
            c2 = self.ref.c2, 
            gran = target_gran,
            off = self.ref.off,
            start = self.ref.start,
            end = self.ref.end
        )

        return self

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
        path = self.ref.get_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return self

if __name__ == "__main__":

    # Example on how to retrieve data.
    fd = ForexData("C:\\Users\\rober\\TUD-CSE-RP-RLinFinance\\data\\forex\\EURUSD\\1M\\BID\\01.05.2022T00.00-01.05.2025T23.59.csv")
    fd.set_gran(Granularity.H1)
    df = fd.df
    print(df.head())



