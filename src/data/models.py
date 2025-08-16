from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Self, Iterable, TypeVar, Type

import pandas as pd

from src.constants import DATA_DIR

T = TypeVar('T', bound='PriceData')

class Timeframe(Enum):
    """Enum for different trading timeframes."""
    TICK = (None, None, "TICK")
    M1 = ("1Min", 1.0, "1M")
    M5 = ("5Min", 5.0, "5M")
    M15 = ("15Min", 15.0, "15M")
    M30 = ("30Min", 30.0, "30M")
    H1 = ("H", 60.0, "1H")
    H4 = ("4H", 240.0, "4H")
    D1 = ("D", 1440.0, "1D")

    def __init__(self, pandas_freq: Optional[str], minutes: Optional[float], pathname: str):
        self.pandas_freq = pandas_freq
        self.minutes = minutes
        self.pathname = pathname

@dataclass(frozen=True)
class PriceData(ABC):
    """ABC for standardized and validated price data."""
    source: str
    instrument: str
    timeframe: Timeframe
    df: pd.DataFrame

    def __post_init__(self):
        """Standardizes the DataFrame after initialization."""
        df = self.df.copy()
        if 'time' not in df.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        required_cols = self._get_required_columns()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{type(self).__name__} DataFrame is missing required columns: {missing_cols}")
        for col in self._get_numeric_columns():
            df[col] = pd.to_numeric(df[col])
        df = df[list(required_cols)]
        object.__setattr__(self, 'df', df)

    @abstractmethod
    def _get_required_columns(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def _get_numeric_columns(self) -> tuple[str, ...]:
        pass

    @staticmethod
    def get_path(source: str, instrument: str, timeframe: Timeframe, d: date) -> Path:
        date_str = d.strftime("%Y%m%d")
        return DATA_DIR / source.upper() / instrument.upper() / timeframe.pathname / f"{date_str}.parquet"

    def save(self, d: date) -> None:
        """Saves the DataFrame for a single specified day to a parquet file."""
        if not (self.df.index.date == d).all():
            raise ValueError(f"Cannot save, DataFrame contains data from dates other than {d}")
        path = self.get_path(self.source, self.instrument, self.timeframe, d)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path)
        print(f"Saved {self.instrument} {self.timeframe.name} data to {path}")

    @classmethod
    def load(cls: Type[T], source: str, instrument: str, timeframe: Timeframe, d: date) -> Optional[T]:
        """Loads the DataFrame for a single specified day from a parquet file."""
        path = cls.get_path(source, instrument, timeframe, d)
        if not path.exists():
            return None
        df = pd.read_parquet(path).reset_index()
        return cls(source=source, instrument=instrument, timeframe=timeframe, df=df)

    @classmethod
    def load_range(cls: Type[T], source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date) -> T:
        """Loads and concatenates all the available data into a single dataframe."""
        days = pd.date_range(start_date, end_date, freq='D')
        daily_dfs = []
        for d in days:
            daily_data = cls.load(source, instrument, timeframe, d.date())
            if daily_data is not None and not daily_data.df.empty:
                daily_dfs.append(daily_data.df)
        if not daily_dfs:
            empty_df = pd.DataFrame(columns=['time'] + list(cls._get_required_columns()))
            return cls(source=source, instrument=instrument, timeframe=timeframe, df=empty_df)
        combined_df = pd.concat(daily_dfs).sort_index().reset_index()
        return cls(source=source, instrument=instrument, timeframe=timeframe, df=combined_df)

    @abstractmethod
    def downsample(self, timeframe: Timeframe) -> Self:
        pass

class CandleData(PriceData):
    """Standardized and validated DataFrame wrapper for candle data."""

    def _get_required_columns(self) -> tuple[str, ...]:
        return 'open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume'

    def _get_numeric_columns(self) -> tuple[str, ...]:
        return self._get_required_columns()

    def __post_init__(self):
        if self.timeframe == Timeframe.TICK:
            raise ValueError("CandleData cannot have a TICK timeframe.")
        super().__post_init__()

    def downsample(self, timeframe: Timeframe) -> Self:
        if timeframe.minutes is None:
            raise ValueError("timeframe.minutes cannot be None")
        if timeframe.minutes < self.timeframe.minutes:
            raise ValueError("Upsampling is not supported")
        if timeframe == self.timeframe:
            return self
        agg_rules = {
            'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last',
            'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last',
            'volume': 'sum'
        }
        resampled_df = self.df.resample(timeframe.pandas_freq).agg(agg_rules)
        resampled_df.dropna(how='all', inplace=True)
        return CandleData(self.source, self.instrument, timeframe, resampled_df.reset_index())

class TickData(PriceData):
    """Standardized and validated DataFrame wrapper for tick data."""

    def _get_required_columns(self) -> tuple[str, ...]:
        return 'bid', 'ask', 'bid_vol', 'ask_vol'

    def _get_numeric_columns(self) -> tuple[str, ...]:
        return self._get_required_columns()

    def __post_init__(self):
        if self.timeframe != Timeframe.TICK:
            raise ValueError("TickData must have a TICK timeframe.")
        super().__post_init__()

    def downsample(self, timeframe: Timeframe) -> CandleData:
        pandas_freq = timeframe.pandas_freq
        if pandas_freq is None:
            raise TypeError(f"invalid timeframe {timeframe}")

        # Resample for OHLCV
        bid_ohlc = self.df['bid'].resample(timeframe.pandas_freq).ohlc()
        ask_ohlc = self.df['ask'].resample(timeframe.pandas_freq).ohlc()
        volume = self.df['bid_vol'].resample(timeframe.pandas_freq).sum() + \
                 self.df['ask_vol'].resample(timeframe.pandas_freq).sum()

        # Rename columns
        bid_ohlc.columns = [f'{col}_bid' for col in bid_ohlc.columns]
        ask_ohlc.columns = [f'{col}_ask' for col in ask_ohlc.columns]
        volume.name = 'volume'

        # Combine, reset index to make 'time' a column, and filter out empty candles
        combined_df = pd.concat([bid_ohlc, ask_ohlc, volume], axis=1)
        combined_df.dropna(subset=['open_bid'], inplace=True) # Each candle must have at least one tick
        combined_df.reset_index(inplace=True)

        return CandleData(self.source, self.instrument, timeframe, combined_df)