import io
import lzma
import struct
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import DATA_DIR
from common.scripts import (date_range, fetch_all, map_input, parallel_run,
                            raise_value_error)

# Mock DATA_DIR for standalone execution
DATA_DIR.mkdir(parents=True, exist_ok=True)

class DukascopyDataDownloader:

    @staticmethod
    def _decompress_lzma_bytes(data_bytes):
        """
        Decompresses LZMA-compressed bytes and returns the raw decompressed bytes.
        """
        with lzma.open(io.BytesIO(data_bytes)) as f:
            return f.read()

    @staticmethod
    def _bi5_to_df_from_bytes(raw_bytes, fmt):
        """
        Converts bi5 formatted byte data into a Pandas DataFrame.
        """
        chunk_size = struct.calcsize(fmt)
        data = []

        for i in range(0, len(raw_bytes), chunk_size):
            chunk = raw_bytes[i:i+chunk_size]
            if len(chunk) != chunk_size:
                raise ValueError(f"Incomplete chunk found in bi5 data: {raw_bytes}")
            data.append(struct.unpack(fmt, chunk))

        return pd.DataFrame(data)

    @staticmethod
    def _get_url(symbol: str, dt: datetime):

        symbol = symbol.upper()
        year_str = str(dt.year).zfill(4)
        month_str = str(dt.month - 1).zfill(2) # duka months are 0-indexed
        day_str = str(dt.day).zfill(2) # duka days are 1-indexed
        hour_str = str(dt.hour).zfill(2) # duka hours are 1-indexed
        return f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year_str}/{month_str}/{day_str}/{hour_str}h_ticks.bi5"

    @staticmethod
    def _process_tick_df(df: pd.DataFrame, symbol: str, dt: datetime):
        """
        Converts tick data to the right format.
        """
        # Standardize input
        assert df is not None
        assert not df.empty
        assert len(df.columns) == 5
        df = df.copy()
        df.columns = ['ms_offset', 'ask', 'bid', 'ask_vol', 'bid_vol']

        # Convert timestamp (milliseconds offset from the hour) to a proper datetime
        base_datetime = pd.Timestamp(dt)
        df['time'] = base_datetime + pd.to_timedelta(df['ms_offset'], unit='ms')
        df.drop('ms_offset', axis=1, inplace=True)

        # Convert prices (Dukascopy stores prices as integers, e.g., EURUSD 1.23456 is 123456)
        # The divisor depends on the symbol's pip definition.
        # For EURUSD, it's usually 10^5. For JPY pairs, it might be 10^3.
        # This needs to be adjusted based on the symbol.
        divisor = 100000.0 if 'JPY' not in symbol else 1000.0 # Simplified logic
        df['ask'] = df['ask'] / divisor
        df['bid'] = df['bid'] / divisor

        # Volumes are typically floats representing millions
        df['ask_vol'] = df['ask_vol'] / 1000000.0
        df['bid_vol'] = df['bid_vol'] / 1000000.0

        # Reorder columns
        df = df[['time', 'ask', 'bid', 'ask_vol', 'bid_vol']]
        return df

    @classmethod
    def _fetch_tick_day(cls, d: datetime, symbol, tick_format='>IIIff'):
        """Fetches day tick data, either from cache or from dukascopy."""
        date_str = d.strftime("%Y%m%d")
        file_path = DATA_DIR / "TICK" / "DUKASCOPY" / symbol / f"{date_str}.csv"
        if file_path.exists():
            return file_path

        print(f"[{date_str}] Fetching data...")
        hours_of_the_day = [d.replace(hour=h) for h in range(24)]
        urls = [cls._get_url(symbol, dt) for dt in hours_of_the_day]
        results = fetch_all(urls)

        print(f"[{date_str}] Processing data... ")
        hour_dfs = []
        for res, dt in zip(results, hours_of_the_day):
            if res == None:
                raise ValueError("One hour of data couldn't be fetched.")
            if res == b'':
                continue
            raw_bytes = cls._decompress_lzma_bytes(res)
            hour_df = cls._bi5_to_df_from_bytes(raw_bytes, tick_format)
            hour_df = cls._process_tick_df(hour_df, symbol, dt)
            hour_dfs.append(hour_df)

        if not hour_dfs:
            # For free days, just create an empty csv.
            dtypes = (('time', 'datetime64[ns]'), ('ask', 'float64'), ('bid', 'float64'), ('ask_vol', 'float64'),
                      ('bid_vol', 'float64'))
            empty_df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes})
            hour_dfs.append(empty_df)

        day_df = pd.concat(hour_dfs, ignore_index=True)
        assert pd.api.types.is_datetime64_any_dtype(day_df['time'])
        day_df.sort_values(by="time", ascending=True, inplace=True)

        print(f"[{date_str}] Saving data... ")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        day_df.to_csv(file_path, index=False)

        print(f"[{date_str}] Finished.")
        return file_path

    @classmethod
    def request_tick_range(cls, symbol, start: datetime, end: datetime, tick_format='>IIIff'):

        # Standardize input
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0)
        symbol = symbol.upper()

        # Go day by day, and cache results.

        days = list(date_range(start, end, timedelta(days=1)))
        fetch_tick_day = partial(cls._fetch_tick_day, symbol=symbol, tick_format=tick_format)
        data_files = parallel_run(fetch_tick_day, days, num_workers=4)

        # Load and combine all day files
        day_dfs = parallel_run(pd.read_csv, data_files, num_workers=2)
        final_df = pd.concat(day_dfs, ignore_index=True)

        # Return ForexTickData Object.
        forex_tick_data = ForexTickData(
            source='DUKASCOPY',
            instrument=symbol,
            ask_column='ask',
            bid_column='bid',
            time_column='time',
            volume_column=None,
            df=final_df
        )

        return forex_tick_data


class Timeframe(Enum):
    M1 = "1Min"
    M5 = "5Min"
    M15 = "15Min"
    M30 = "30Min"
    H1 = "H"
    H4 = "4H"
    D1 = "D"
    W1 = "W" # Weekly
    MN1 = "M" # Monthly

    def to_pandas_freq(self) -> str:
        """Returns the pandas frequency string for this timeframe."""
        return self.value

    def as_minutes(self) -> int:
        """Returns the number of minutes for this timeframe."""
        if self == Timeframe.M1:
            return 1
        elif self == Timeframe.M5:
            return 5
        elif self == Timeframe.M15:
            return 15
        elif self == Timeframe.M30:
            return 30
        elif self == Timeframe.H1:
            return 60
        elif self == Timeframe.H4:
            return 240
        elif self == Timeframe.D1:
            return 1440
        elif self == Timeframe.W1:
            return 10080
        elif self == Timeframe.MN1: # Added missing return for MN1
            # Assuming 30 days for a month for simplicity in minutes calculation
            # This might need adjustment based on actual month lengths if critical
            return 30 * 1440
        else:
            raise ValueError(f"Unhandled timeframe: {self}")


class ForexTickData:
    """
    Class to handle Forex tick data, ensuring the DataFrame has the required structure.
    """

    def __init__(self, source: str, instrument: str, df: pd.DataFrame,
                 time_column: str = 'date_gmt',
                 bid_column: str = 'bid',
                 ask_column: str = 'ask',
                 volume_column: Optional[str] = None):
        # upper
        self.source = source.upper()
        self.instrument = instrument.upper()
        self.df = df.copy() # Use a copy to avoid modifying the original DataFrame outside the class

        self.time_column = time_column
        self.bid_column = bid_column
        self.ask_column = ask_column
        self.volume_column = volume_column

        self.validate_dataframe()
        # Set the time column as index for resampling
        if not pd.api.types.is_datetime64_any_dtype(self.df.index) or self.df.index.name != self.time_column:
             if self.time_column in self.df.columns:
                self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
                self.df = self.df.set_index(self.time_column)
             else:
                raise ValueError(f"Time column '{self.time_column}' not found in DataFrame for setting index.")


    def validate_dataframe(self):
        """
        Validates that the DataFrame contains all required columns.
        """
        required_columns = [self.bid_column, self.ask_column] # Time column will be index or checked separately
        if self.time_column not in self.df.columns and self.df.index.name != self.time_column :
             raise ValueError(f"Time column '{self.time_column}' not found in DataFrame columns or as index name.")

        # Check if time column is already the index
        if self.df.index.name == self.time_column:
            if not pd.api.types.is_datetime64_any_dtype(self.df.index):
                # This case should ideally not happen if constructor sets index correctly
                raise ValueError(f"Index '{self.time_column}' must be of datetime type.")
        elif self.time_column in self.df.columns:
             if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_column]):
                try:
                    self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
                except Exception as e:
                    raise ValueError(f"Could not convert time column '{self.time_column}' to datetime: {e}")
        else: # Should be caught by the initial check
            pass


        if self.volume_column:
            required_columns.append(self.volume_column)

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

        # Ensure bid and ask columns are numeric
        if not pd.api.types.is_numeric_dtype(self.df[self.bid_column]):
            raise ValueError(f"Column {self.bid_column} must be numeric.")
        if not pd.api.types.is_numeric_dtype(self.df[self.ask_column]):
            raise ValueError(f"Column {self.ask_column} must be numeric.")
        if self.volume_column and not pd.api.types.is_numeric_dtype(self.df[self.volume_column]):
            raise ValueError(f"Column {self.volume_column} must be numeric.")

    def to_candles(self, granularity: Timeframe) -> 'ForexCandleData':
        """
        Converts the tick data to OHLCV format for both bid and ask prices.
        """

        pandas_freq = granularity.to_pandas_freq()

        print(f"Converting tick data to {pandas_freq} OHLCV candles...")

        # Ensure the index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
             if self.time_column in self.df.columns:
                self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
                self.df = self.df.set_index(self.time_column)
             elif self.df.index.name == self.time_column and not pd.api.types.is_datetime64_any_dtype(self.df.index):
                 self.df.index = pd.to_datetime(self.df.index)
             else:
                raise ValueError("DataFrame index is not a DatetimeIndex and time column cannot be set as index.")

        # Ensure the index is datetime for resampling
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex for resampling.")

        # Resample for Bid OHLC
        bid_ohlc = self.df[self.bid_column].resample(pandas_freq).ohlc()
        bid_ohlc.columns = [f'{col}_bid' for col in bid_ohlc.columns]

        # Resample for Ask OHLC
        ask_ohlc = self.df[self.ask_column].resample(pandas_freq).ohlc()
        ask_ohlc.columns = [f'{col}_ask' for col in ask_ohlc.columns]

        # Calculate Volume based on available columns
        if self.volume_column:
            volume_series = self.df[self.volume_column].resample(pandas_freq).sum()
            volume_series.name = 'volume'
        else:
            volume_series = self.df[self.bid_column].resample(pandas_freq).count() # count of ticks
            volume_series.name = 'volume'

        # Combine all into a single DataFrame
        ohlcv_df = pd.concat([bid_ohlc, ask_ohlc, volume_series], axis=1)

        ohlcv_df.reset_index(inplace=True)
        # The time column from reset_index will have the name of the original index or 'index'
        # We want to rename it to 'date_gmt' consistently.
        if self.df.index.name and self.df.index.name in ohlcv_df.columns:
            ohlcv_df.rename(columns={self.df.index.name: 'date_gmt'}, inplace=True)
        elif 'index' in ohlcv_df.columns and pd.api.types.is_datetime64_any_dtype(ohlcv_df['index']):
            ohlcv_df.rename(columns={'index': 'date_gmt'}, inplace=True)
        elif self.time_column in ohlcv_df.columns: # If original time_column name was somehow preserved
             ohlcv_df.rename(columns={self.time_column: 'date_gmt'}, inplace=True)
        else: # Fallback if the datetime column name is unexpected after reset_index
            # Try to find the datetime column
            dt_col = next((col for col in ohlcv_df.columns if pd.api.types.is_datetime64_any_dtype(ohlcv_df[col])), None)
            if dt_col:
                ohlcv_df.rename(columns={dt_col: 'date_gmt'}, inplace=True)
            else:
                print("Warning: Could not automatically identify and rename the time column to 'date_gmt'.")


        return ForexCandleData(
            source=self.source,
            instrument=self.instrument,
            granularity= granularity,
            df=ohlcv_df
        )


class ForexCandleData:
    def __init__(self, source: str, instrument: str, granularity: Timeframe, df: pd.DataFrame):
        self.source = source.upper()
        self.instrument = instrument.upper()
        self.granularity = granularity
        self.df = df.copy()
        self.validate_dataframe()

        self.df = self.df.sort_values(by='date_gmt')
        self.df = self.df[self.df['volume'] > 0]

        # dropna
        self.df.dropna(inplace=True)
        # reset index
        self.df.reset_index(drop=True, inplace=True)

        if self.df.empty:
            print("DataFrame is empty after cleaning (volume > 0 and dropna). No data to save or analyze.")

    @staticmethod
    def load(source: str, instrument: str, granularity: Timeframe, start_time: datetime, end_time: datetime) -> 'ForexCandleData':
        file_path = DATA_DIR / source.upper() / instrument.upper() / granularity.to_pandas_freq() / f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}" / "data.csv" if start_time and end_time else None

        print(f"Loading Forex data from {file_path}...")

        if file_path and file_path.is_file():
            df = pd.read_csv(file_path)
            df['date_gmt'] = pd.to_datetime(df['date_gmt'])
            return ForexCandleData(source=source, instrument=instrument, granularity=granularity, df=df)
        else:
            raise FileNotFoundError(f"No data file found for {source}, {instrument}, {granularity} in the specified date range.")


    def validate_dataframe(self):
        """
        Validates that the DataFrame contains all required columns.
        Raises
        -------
        ValueError
            If the DataFrame does not contain the required columns.
        """
        required_columns = ['date_gmt', 'open_bid', 'high_bid', 'low_bid', 'close_bid',
                            'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume']
        if not all(col in self.df.columns for col in required_columns):
            expected_columns = set(required_columns)
            actual_columns = set(self.df.columns)
            missing_columns = expected_columns - actual_columns
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

        # Ensure date_gmt is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['date_gmt']):
            try:
                self.df['date_gmt'] = pd.to_datetime(self.df['date_gmt'])
            except Exception as e:
                raise ValueError(f"Could not convert 'date_gmt' to datetime: {e}")

    def combine(self, other: 'ForexCandleData') -> 'ForexCandleData':
        """
        Combines two ForexData objects into one.

        Parameters
        ----------
        other : ForexData
            Another ForexData object to combine with this one.

        Returns
        -------
        ForexData
            A new ForexData object containing the combined data.
        """
        if self.source != other.source or self.instrument != other.instrument or self.granularity != other.granularity:
            raise ValueError("Cannot combine ForexData from different sources, instruments, or granularities.")

        # if date ranges overlap, fail
        if not (self.df['date_gmt'].max() < other.df['date_gmt'].min() or
                other.df['date_gmt'].max() < self.df['date_gmt'].min()):
            raise ValueError("Cannot combine ForexData with overlapping date ranges.")

        combined_df = pd.concat([self.df, other.df]).drop_duplicates(subset='date_gmt').reset_index(drop=True)
        return ForexCandleData(source=self.source, instrument=self.instrument, granularity=self.granularity, df=combined_df)

    def set_period(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> 'ForexCandleData':
        """
        Sets the period for the ForexCandleData object.

        Parameters
        ----------
        start : pd.Timestamp, optional
            The start date for the period. If None, uses the earliest date in the DataFrame.
        end : pd.Timestamp, optional
            The end date for the period (exclusive). If None, uses the latest date in the DataFrame. 

        Returns
        -------
        ForexCandleData
            A new ForexCandleData object with the specified period.
        """
        if start is None:
            start = self.df['date_gmt'].min()
        if end is None:
            end = self.df['date_gmt'].max()
            end += pd.Timedelta(days=100)  # Make end exclusive

        filtered_df = self.df[(self.df['date_gmt'] >= start) & (self.df['date_gmt'] < end)].copy()
        return ForexCandleData(source=self.source, instrument=self.instrument, granularity=self.granularity, df=filtered_df)

    def analyse_save(self):
        """
        Analyzes the Forex data and saves it to a CSV file.
        """

        if self.df.empty:
            print("DataFrame is empty. Skipping analysis and save.")
            return

        start_time = self.df['date_gmt'].min()
        end_time = self.df['date_gmt'].max()

        folder_path = DATA_DIR / self.source / self.instrument / self.granularity.to_pandas_freq() / f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}"
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = "data.csv"
        file_path = folder_path / file_name

        print(f"Saving Forex data to {file_path}...")
        self.df.to_csv(file_path, index=False)
        print(f"Saved Forex data to {file_path}")

        # ANALYSIS
        print(f"Analyzing Forex data for {self.instrument} at {self.granularity.to_pandas_freq()} granularity...")

        granularity_in_minutes = self.granularity.as_minutes()

        self.df['time_diff'] = self.df['date_gmt'].diff().dt.total_seconds() / 60
        self.df.fillna({ 'time_diff' : 0 }, inplace=True)

        tolerance = 0.1 * granularity_in_minutes # Allow 10% deviation for the base granularity
        weekend_base_duration_minutes = 2 * 24 * 60 # Approx 48 hours

        is_standard_gap = np.isclose(self.df['time_diff'], granularity_in_minutes, atol=tolerance)
        is_weekend_gap = np.isclose(self.df['time_diff'], weekend_base_duration_minutes + granularity_in_minutes, atol=granularity_in_minutes * 2) # Wider tolerance for weekends

        # Rows that are NOT preceded by a standard gap AND NOT preceded by a weekend gap
        # (and ignoring the first row which has time_diff = 0)
        missing_rows_mask = (self.df['time_diff'] > 0) & ~is_standard_gap & ~is_weekend_gap
        missing_rows = self.df[missing_rows_mask]


        missing_rows_file_path = folder_path / f"missing_rows_analysis.txt"
        with open(missing_rows_file_path, 'w') as f:
            f.write(f"Analysis of time differences for {self.instrument} at {granularity_in_minutes} min granularity.\n")
            f.write(f"Expected regular time diff: {granularity_in_minutes} minutes.\n")
            f.write(f"Approximate weekend gap expectation (time diff for first candle after weekend): {weekend_base_duration_minutes + granularity_in_minutes} minutes.\n")
            f.write("Rows listed below are those where the time difference from the previous candle was neither the standard granularity nor a typical weekend gap.\n")
            f.write("This suggests potential missing data or unusual market closure.\n\n")

            total_estimated_missing_candles = 0
            if not missing_rows.empty:
                for index, row in missing_rows.iterrows():
                    # Exclude the row itself, we're talking about the gap *before* this row.
                    # If time_diff is X, and granularity is G, it means (X/G - 1) candles are missing.
                    num_missing_candles = (row['time_diff'] / granularity_in_minutes) - 1
                    if num_missing_candles > 0.1: # Only report if it's a substantial part of a candle
                        f.write(f"Unusual gap before {row['date_gmt']} (local time in data): {row['time_diff']:.2f} minutes. Estimated ~{num_missing_candles:.2f} missing {granularity_in_minutes}-min candles.\n")
                        total_estimated_missing_candles += num_missing_candles
                f.write(f"\nTotal estimated number of missing candles from these unusual gaps: {total_estimated_missing_candles:.2f}\n")
            else:
                f.write("No significant unexpected gaps found.\n")


        # plot close prices with missing rows highlighted
        plt.figure(figsize=(15, 7))
        plt.plot(self.df['date_gmt'], self.df['close_bid'], label='Bid Close', color='blue', linewidth=0.8)
        plt.plot(self.df['date_gmt'], self.df['close_ask'], label='Ask Close', color='orange', linewidth=0.8, alpha=0.7)

        # Highlight the points *after* an unusual gap
        if not missing_rows.empty:
            plt.scatter(missing_rows['date_gmt'], missing_rows['close_bid'], color='red', label='After Unusual Gap', marker='x', s=50, zorder=5)

        plt.title(f"{self.instrument} Close Prices ({granularity_in_minutes} Min) with Unusual Gaps Highlighted")
        plt.xlabel('Date GMT')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, which='major', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # save to file
        plt_file_path = folder_path / f"close_prices_with_missing_data.png"
        plt.savefig(plt_file_path)
        plt.close()

        # save a histogram of the volume amounts (90th percentile)
        if 'volume' in self.df.columns and not self.df['volume'].empty:
            volume_99th_percentile = self.df['volume'].quantile(0.99) # Use 99th to see more of the tail
            temp_df_volume = self.df[self.df['volume'] <= volume_99th_percentile]
            if not temp_df_volume.empty:
                plt.figure(figsize=(12, 6))
                plt.hist(temp_df_volume['volume'], bins=100, color='teal', alpha=0.7)
                plt.title(f"{self.instrument} Volume Distribution (up to 99th percentile: {volume_99th_percentile:.2f}) - {granularity_in_minutes} Min")
                plt.xlabel('Volume (Ticks or Summed Volume)')
                plt.ylabel('Frequency')
                plt.grid(True, linestyle=':', alpha=0.7)
                # save to file
                volume_hist_file_path = folder_path / f"volume_distribution.png"
                plt.savefig(volume_hist_file_path)
                plt.close()
            else:
                print("Not enough data points for volume histogram after percentile filtering.")
        else:
            print("Volume column not available or empty for histogram.")


def combine_and_analyze_multiple_files():
    # ask user for all source, instrument, and granularity
    print("Forex Data Combination and Analysis")
    print("=" * 30)

    source_name = input("Enter the data source name (e.g., Dukascopy, FXCM): ").strip()
    instrument_name = input("Enter the instrument name (e.g., EURUSD, GBPJPY): ").strip()

    # Get granularity
    print("\nAvailable granularities:")
    for tf in Timeframe:
        print(f"- {tf.name} ({tf.value})")

    while True:
        try:
            granularity_input = input("Choose a granularity (e.g., M15, H1, D1): ").strip().upper()
            selected_granularity = Timeframe[granularity_input]
            break
        except KeyError:
            print("Invalid granularity. Please choose from the list (e.g., M1, M5, H1).")

    # Get file paths
    file_paths = []
    print("\nEnter the full paths to your OHLCV CSV files (one per line). Type 'done' when finished:")
    while True:
        file_path_str = input("File path: ").strip()
        if file_path_str.lower() == 'done':
            break
        file_path = Path(file_path_str)
        if file_path.is_file() and file_path.suffix.lower() == '.csv':
            file_paths.append(file_path)
        else:
            print("Invalid file path or not a CSV file. Please try again.")
    if not file_paths:
        print("No valid file paths provided. Exiting.")
        return
    print(f"Found {len(file_paths)} files to process.")
    combined_data = None

    for file_path in file_paths:
        try:
            print(f"\nLoading data from {file_path}...")
            df = pd.read_csv(file_path, parse_dates=['date_gmt'])
            if combined_data is None:
                combined_data = ForexCandleData(
                    source=source_name,
                    instrument=instrument_name,
                    granularity=selected_granularity,
                    df=df
                )
            else:
                new_data = ForexCandleData(
                    source=source_name,
                    instrument=instrument_name,
                    granularity=selected_granularity,
                    df=df
                )
                combined_data = combined_data.combine(new_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    if combined_data is not None:
        print("\nCombining and analyzing data...")
        combined_data.analyse_save()
        print(f"\nCombined data has {len(combined_data.df)} rows.")
        print(f"Output files are in subdirectories of: {DATA_DIR}")
    else:
        print("No valid data was combined. Please check your input files and try again.")


def download_convert_save_dukascopy_data():
    print("Dukascopy Data Downloader and Converter")
    print("=" * 30)

    # Get user inputs for download
    default_pair = lambda x: "EURUSD" if x == "" else x
    normalize = lambda x: x.strip().upper()
    len_filter = lambda x: x if len(x) == 6 else raise_value_error(f"Expected pair length is 6, was {len(x)}")
    pair = map_input("Enter the currency pair (default: EURUSD): ", [normalize, default_pair, len_filter])

    convert_dtime = lambda x: datetime.strptime(x.strip(), "%Y%m%d")
    start = map_input("Enter the start time to download (format: yyyymmdd): ", [convert_dtime])
    end = map_input("Enter the end time (excl) to download (format: yyyymmdd): ", [convert_dtime])

    print("\nAvailable granularities:")
    for tf in Timeframe:
        print(f"- {tf.name} ({tf.value})")

    get_gran = lambda x: Timeframe[x.strip().upper()]
    selected_granularity: Timeframe = map_input("Choose a granularity (e.g., M15, H1, D1): ", [get_gran])

    # Download data
    try:
        print(f"\nDownloading {pair} data from {start} to {end} at {selected_granularity.name} granularity...")
        forex_tick_data = DukascopyDataDownloader.request_tick_range(pair, start, end)
        print("Download complete.")
        if forex_tick_data is not None:
            print(f"\nConverting tick data to {selected_granularity.name} granularity...")
            forex_candle_data = forex_tick_data.to_candles(granularity=selected_granularity)
            print("Conversion complete.")
            if not forex_candle_data.df.empty:
                print("\nAnalyzing and saving data...")
                forex_candle_data.analyse_save()
                print(f"Processed OHLCV data has {len(forex_candle_data.df)} rows.")
                print(f"Output files are in subdirectories of: {DATA_DIR}")
            else:
                print("No OHLCV data was generated (e.g., due to empty input or filtering).")
        else:
            print("No data was downloaded. Please check the pair and year.")
    except Exception as e:
        print(f"An error occurred during download or conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    while True:
        print("\n" + "=" * 30)
        print("Welcome to the Forex Data Processor!")
        print("Please choose an option:")
        print("0. Download Dukascopy Tick Data, convert, analyze, and save")
        print("1. Load Tick Data and convert it to OHLCV")
        print("2. Load multiple OHLCV files and combine them, then analyze and save")
        print("3. Exit")
        choice = input("Enter your choice (0-3): ").strip()
        if choice == '0':
            download_convert_save_dukascopy_data()
        elif choice == '1':
            print("Your choice sucks, just pick 0.")
        elif choice == '2':
            combine_and_analyze_multiple_files()
        elif choice == '3':
            print("Exiting the Forex Data Processor. Goodbye!")
            break
        else:
            print("Invalid choice. Please run the program again and choose a valid option.")
