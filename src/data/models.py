from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Self, TypeVar, Type

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DATA_DIR, Timeframe, Price

T = TypeVar('T', bound='PriceData')

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PriceData(ABC):
    """ABC for standardized and validated price data."""

    source: str
    instrument: str
    timeframe: Timeframe
    df: pd.DataFrame
    start_date: date
    end_date: date

    def __post_init__(self):
        """Standardize and validates the DataFrame after initialization."""

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

        # Sort based on index
        self.df.sort_index(inplace=True)

        # Verify and standardize columns
        required_cols = set(self._get_required_columns())
        actual_cols = set(self.df.columns)
        missing_cols = required_cols - actual_cols
        if missing_cols:
            raise ValueError(f"{type(self).__name__} DataFrame is missing required columns: {missing_cols}")
        extra_cols = actual_cols - required_cols
        if extra_cols:
            self.df.drop(extra_cols, axis=1, inplace=True)
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col])

        # Verify timeframe duration
        if self.timeframe.minutes is not None and not self.df.empty:
            time_diffs: pd.Series = self.df.index.to_series().diff(periods=1)
            time_diffs.dropna(inplace=True)
            time_diffs_s = time_diffs.dt.total_seconds().round(decimals=0)
            timeframe_s: float = self.timeframe.minutes * 60.0
            remainders = time_diffs_s % timeframe_s
            if not np.all(np.isclose(remainders, 0)):
                raise ValueError(f"Wrong timeframe specified for {type(self).__name__}.")

        # Verify start and end date
        if not self.df.empty and self.df.index[0].date() < self.start_date:
            raise ValueError(f"DataFrame contains data before start_date {self.start_date}")
        if not self.df.empty and self.df.index[-1].date() > self.end_date:
            raise ValueError(f"DataFrame contains data after end_date {self.start_date}")

    @abstractmethod
    def _get_required_columns(self) -> tuple[str, ...]:
        pass

    @staticmethod
    def get_path(source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date) -> Path:
        if start_date == end_date:
            date_str = start_date.strftime("%Y%m%d")
        else:
            start_date_str = start_date.strftime("%Y%m%d")
            end_date_str = end_date.strftime("%Y%m%d")
            date_str = f"{start_date_str}-{end_date_str}"
        return DATA_DIR / source.upper() / instrument.upper() / timeframe.pathname / f"{date_str}.parquet"

    def save(self):
        path = self.get_path(self.source, self.instrument, self.timeframe, self.start_date, self.end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path)
        logging.info(f"Saved {self.instrument} {self.timeframe.name} data to {path}")

    @classmethod
    def _load(cls: Type[T], source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date) -> Optional[T]:
        """Directly loads a DataFrame from a parquet file if it exists, otherwise returns None."""
        path = cls.get_path(source, instrument, timeframe, start_date, end_date)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return cls(source, instrument, timeframe, df, start_date, end_date)

    @classmethod
    def load(cls: Type[T], source: str, instrument: str, timeframe: Timeframe, start_date: date, end_date: date, use_cache: bool = True) -> T:
        """Loads and concatenates all the available data into a single dataframe."""

        # Use cache for the date range if available and intended.
        if use_cache:
            price_data = cls._load(source, instrument, timeframe, start_date, end_date)
            if price_data is not None:
                return price_data

        # Load data day-by-day.
        days = pd.date_range(start_date, end_date, freq='D')
        daily_dfs = []
        for d in tqdm(days, desc=f"Loading {instrument} {timeframe.name} daily files"):
            daily_data = cls._load(source, instrument, timeframe, d.date(), d.date())
            if daily_data is None:
                raise ValueError(f"Missing data for day {instrument} {timeframe.name} {d.date()}")
            if not daily_data.df.empty:
                daily_dfs.append(daily_data.df)

        # Combine day-by-day dataframes
        if not daily_dfs:
            empty_index = pd.DatetimeIndex([], name='time', tz='utc')
            combined_df = pd.DataFrame(index=empty_index, columns=list(cls._get_required_columns()))
        else:
            combined_df = pd.concat(daily_dfs)

        # Create, cache and return price_data.
        price_data = cls(source, instrument, timeframe, combined_df, start_date, end_date)
        if use_cache:
            price_data.save()
        return price_data

    def to_array(self) -> np.ndarray:
        """Gets a numpy array from the internal dataframe.

        Follows the order specified by ._get_required_columns()
        """
        columns = list(self._get_required_columns())
        return self.df[columns].to_numpy(dtype=np.float64, copy=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Gets a copy of the internal dataframe resetting the index."""

        columns = list(self._get_required_columns())
        df: pd.DataFrame = self.df[columns].copy()
        df.reset_index(inplace=True)
        return df

    @abstractmethod
    def downsample(self, timeframe: Timeframe) -> Self:
        pass

class CandleData(PriceData):
    """Standardized and validated DataFrame wrapper for candle data."""

    def _get_required_columns(self) -> tuple[str, ...]:
        return Price.names

    def __post_init__(self):
        if self.timeframe == Timeframe.TICK:
            raise ValueError(f"{self.__class__.__name__} cannot have a TICK timeframe.")
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
        return CandleData(self.source, self.instrument, timeframe, resampled_df, self.start_date, self.end_date)

class TickData(PriceData):
    """Standardized and validated DataFrame wrapper for tick data."""

    def _get_required_columns(self) -> tuple[str, ...]:
        return 'bid', 'ask', 'bid_vol', 'ask_vol'

    def __post_init__(self):
        if self.timeframe != Timeframe.TICK:
            raise ValueError(f"{self.__class__.__name__} must have a TICK timeframe.")
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

        return CandleData(self.source, self.instrument, timeframe, combined_df, self.start_date, self.end_date)