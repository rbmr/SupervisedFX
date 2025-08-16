from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional, Self, TypeVar, Type

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
        # Ensure 'time' index
        if self.df.index.name != 'time':
            if 'time' not in self.df.columns:
                raise ValueError("DataFrame must contain a 'time' column or a 'time' index.")
            self.df.set_index('time', inplace=True)

        # Ensure index is UTC datetime.
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index, utc=True)
        elif self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('utc')
        elif self.df.index.tz.zone != 'UTC':
            self.df.index = self.df.index.tz_convert('utc')

        # Verify and standardize columns
        required_cols = set(self._get_required_columns())
        actual_cols = set(self.df.columns)
        missing_cols = required_cols - actual_cols
        if missing_cols:
            raise ValueError(f"{type(self).__name__} DataFrame is missing required columns: {missing_cols}")
        for col in self._get_numeric_columns():
            self.df[col] = pd.to_numeric(self.df[col])
        extra_cols = actual_cols - required_cols
        if extra_cols:
            self.df.drop(extra_cols, axis=1, inplace=True)

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
        df = pd.read_parquet(path)
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
            empty_index = pd.DatetimeIndex([], name='time', tz='utc')
            empty_df = pd.DataFrame(index=empty_index, columns=list(cls._get_required_columns()))
            return cls(source=source, instrument=instrument, timeframe=timeframe, df=empty_df)
        combined_df = pd.concat(daily_dfs).sort_index()
        return cls(source=source, instrument=instrument, timeframe=timeframe, df=combined_df)

    @abstractmethod
    def downsample(self, timeframe: Timeframe) -> Self:
        pass

class CandleData(PriceData):
    """Standardized and validated DataFrame wrapper for candle data."""

    def _get_required_columns(self) -> tuple[str, ...]:
        return 'open_bid', 'high_bid', 'low_bid', 'close_bid', 'exec_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'exec_ask', 'volume'

    def _get_numeric_columns(self) -> tuple[str, ...]:
        return self._get_required_columns()

    def __post_init__(self):
        if self.timeframe == Timeframe.TICK:
            raise ValueError("CandleData cannot have a TICK timeframe.")
        super().__post_init__()

    def downsample(self, timeframe: Timeframe) -> Self:
        if timeframe.minutes is None:
            raise ValueError("timeframe.minutes cannot be None")
        if timeframe.pandas_freq is None:
            raise ValueError("timeframe.pandas_freq cannot be None")
        if timeframe.minutes < self.timeframe.minutes:
            raise ValueError("Upsampling is not supported")
        if timeframe == self.timeframe:
            return self
        agg_rules = {
            'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last', 'exec_bid': 'first',
            'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last', 'exec_ask': 'first',
            'volume': 'sum'
        }
        resampled_df = self.df.resample(timeframe.pandas_freq).agg(agg_rules)
        resampled_df.dropna(how='all', inplace=True)
        return CandleData(self.source, self.instrument, timeframe, resampled_df)

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

    def downsample(self, timeframe: Timeframe, delay: pd.Timedelta = pd.Timedelta("0s")) -> CandleData:
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

        # Combine and filter out empty candles
        combined_df = pd.concat([bid_ohlc, ask_ohlc, volume], axis=1)
        combined_df.dropna(subset=['open_bid'], inplace=True) # Each candle must have at least one tick

        # Calculate execution prices based on delay.
        candle_starts = pd.DataFrame(index=combined_df.index)
        candle_starts['exec_time'] = candle_starts.index + delay
        exec_ticks = pd.merge_asof(
            left=candle_starts,
            right=self.df[['bid', 'ask']],
            left_on='exec_time',
            right_index=True,
            direction='backward' # ticks represent changes in price, we get the most recent change to find the price.
        )

        # Rename columns and join back to the main OHLCV timeframe.
        exec_ticks.set_index(combined_df.index, inplace=True)
        exec_ticks.rename(columns={'bid': 'exec_bid', 'ask': 'exec_ask'}, inplace=True)
        combined_df = combined_df.join(exec_ticks[['exec_bid', 'exec_ask']])
        combined_df.fillna({"exec_bid": combined_df['close_bid']}, inplace=True)
        combined_df.fillna({"exec_ask": combined_df['close_ask']}, inplace=True)

        return CandleData(self.source, self.instrument, timeframe, combined_df)