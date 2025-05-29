import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt # Duplicate import, but keeping as in original
from common.constants import DATA_DIR
import lzma
import struct
import pandas as pd
import numpy as np
import os
import sys
import io
import requests
import calendar
from datetime import datetime

# Mock DATA_DIR for standalone execution
DATA_DIR.mkdir(parents=True, exist_ok=True)

class DukascopyDataDownloader:
    def __init__(self):
        pass

    
    def _bi5_to_df_from_bytes(self, data_bytes, fmt):
        """
        Converts bi5 formatted byte data (after LZMA decompression) into a Pandas DataFrame.

        Args:
            data_bytes (bytes): The byte content of the bi5 data (must be LZMA compressed).
            fmt (str): The struct format string for unpacking the binary data.

        Returns:
            pd.DataFrame: A DataFrame containing the tick data, or None if an error occurs.
        """
        chunk_size = struct.calcsize(fmt)
        data = []
        try:
            # Wrap the byte data in an io.BytesIO object to make it file-like
            # Then open it with lzma to decompress
            with lzma.open(io.BytesIO(data_bytes)) as f:
                while True:
                    chunk = f.read(chunk_size)
                    if chunk:
                        if len(chunk) == chunk_size: # Ensure the chunk is complete
                            data.append(struct.unpack(fmt, chunk))
                        else:
                            # Handle potential incomplete chunk at the end of the stream if necessary
                            # For tick data, usually, an incomplete chunk means corrupted data or end of a non-aligned stream
                            print(f"Warning: Incomplete chunk of size {len(chunk)} read, expected {chunk_size}. Skipping.")
                            break
                    else:
                        break
            df = pd.DataFrame(data)
            return df
        except lzma.LZMAError as e:
            print(f"Error decompressing data: {e}")
            return None
        except Exception as e:
            print(f"Error processing bi5 data: {e}")
            return None

    def _request_tick_data(self, symbol, year, month, day, hour, tick_format='>IIIff'): # Added tick_format
        """
        Downloads tick data for a given symbol and time, then converts it to a DataFrame.

        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            year (int): The year.
            month (int): The month (1-12).
            day (int): The day.
            hour (int): The hour (0-23).
            tick_format (str): The struct format for unpacking tick data.
                            Default is '>iIIii' for Dukascopy (time, ask, bid, ask_vol, bid_vol).

        Returns:
            pd.DataFrame: A DataFrame containing the tick data, or None if an error occurs.
        """
        # Dukascopy month is 0-indexed for the URL, but we accept 1-12 for convenience
        req_month = month - 1

        symbol_upper = symbol.upper()

        # year as string of 4 digits, month and day as string of 2 digits for URL
        year_str = str(year).zfill(4)
        month_str = str(req_month).zfill(2) # Use req_month for URL
        day_str = str(day).zfill(2)
        hour_str = str(hour).zfill(2)

        # Construct the URL (example was hardcoded to EURUSD and 2025, making it dynamic)
        # url = f"https://datafeed.dukascopy.com/datafeed/EURUSD/2025/{month_str}/{day_str}/{hour_str}h_ticks.bi5" # Original
        url = f"https://datafeed.dukascopy.com/datafeed/{symbol_upper}/{year_str}/{month_str}/{day_str}/{hour_str}h_ticks.bi5"
        print(f"Requesting data from {url}")

        try:
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            file_content = response.content

            # --- Optional: Save the file if you still need it ---
            # filename = f"{symbol_upper}_{year_str}{str(month).zfill(2)}{day_str}_{hour_str}h_ticks.bi5" # Use original month for filename
            # folder = "data"
            # filepath = f"{folder}/{filename}"
            # import os
            # if not os.path.exists(folder):
            #     os.makedirs(folder)
            # with open(filepath, 'wb') as f:
            #     f.write(file_content)
            # print(f"File saved to {filepath}")
            # --- End Optional Save ---

            # Directly process the downloaded content
            df = self._bi5_to_df_from_bytes(file_content, tick_format)

            if df is not None and not df.empty:
                # Assuming the standard Dukascopy format: timestamp, ask, bid, ask_volume, bid_volume
                df.columns = ['ms_offset', 'ask', 'bid', 'ask_vol', 'bid_vol']

                # Convert timestamp (milliseconds offset from the hour) to a proper datetime
                base_datetime = pd.Timestamp(f'{year}-{month}-{day} {hour}:00:00')
                df['time'] = base_datetime + pd.to_timedelta(df['ms_offset'], unit='ms')

                # Convert prices (Dukascopy stores prices as integers, e.g., EURUSD 1.23456 is 123456)
                # The divisor depends on the symbol's pip definition.
                # For EURUSD, it's usually 10^5. For JPY pairs, it might be 10^3.
                # This needs to be adjusted based on the symbol.
                divisor = 100000.0 if 'JPY' not in symbol_upper else 1000.0 # Simplified logic
                df['ask'] = df['ask'] / divisor
                df['bid'] = df['bid'] / divisor
                # Volumes are typically floats representing millions
                df['ask_vol'] = df['ask_vol'] / 1000000.0
                df['bid_vol'] = df['bid_vol'] / 1000000.0

                df = df[['time', 'ask', 'bid', 'ask_vol', 'bid_vol']] # Reorder and select columns

            return df

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error downloading file: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in request_tick_data: {e}")
            return None
        
    def _request_ticks_for_day(self, symbol, year, month, day, tick_format='>IIIff'):
        """
        Requests tick data for a specific day and symbol for each hour of that day.

        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            year (int): The year.
            month (int): The month (1-12).
            day (int): The day.
            tick_format (str): The struct format for unpacking tick data.

        Returns:
            pd.DataFrame: A DataFrame containing the tick data for the entire day,
                        or None if no data is successfully retrieved.
        """
        # Create a list to hold DataFrames for each hour
        hourly_data = []

        # Get the number of hours in the day (24 for most days, 23 for DST changes)
        num_hours = 24

        # Loop through each hour of the day
        for hour in range(num_hours):
            print(f"Requesting data for {symbol} on {year}-{month:02d}-{day:02d} at hour {hour:02d}")
            df = self._request_tick_data(symbol, year, month, day, hour, tick_format)
            if df is not None and not df.empty:
                hourly_data.append(df)

        # Concatenate all hourly DataFrames into one
        if hourly_data:
            full_day_df = pd.concat(hourly_data, ignore_index=True)
            return full_day_df
        else:
            print(f"No data retrieved for {symbol} on {year}-{month:02d}-{day:02d}")
            return None
        
    def _request_ticks_for_month(self, symbol, year, month, tick_format='>IIIff'):
        """
        Requests tick data for a specific month and symbol.

        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            year (int): The year.
            month (int): The month (1-12).
            tick_format (str): The struct format for unpacking tick data.

        Returns:
            pd.DataFrame: A DataFrame containing the tick data for the entire month.
        """
        # Get the number of days in the month
        num_days = calendar.monthrange(year, month)[1]

        # Initialize an empty list to store DataFrames
        dfs = []

        # Loop through each day of the month
        for day in range(1, num_days + 1):
            df = self._request_ticks_for_day(symbol, year, month, day, tick_format)
            if df is not None and not df.empty:
                dfs.append(df)

        # Concatenate all DataFrames into one
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            print("No data available for the specified month.")
        return None
    
    def request_ticks_for_year(self, symbol, year, tick_format='>IIIff') -> 'ForexTickData':
        """
        Downloads tick data for a given symbol and year, then converts it to a DataFrame.

        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            year (int): The year.
            tick_format (str): The struct format for unpacking tick data.

        Returns:
            pd.DataFrame: A DataFrame containing the tick data for the entire year, or None if an error occurs.
        """

        symbol = symbol.upper()

        all_data = []
        for month in range(1, 13):
            for day in range(1, 32):
                for hour in range(0, 24):
                    df = self._request_tick_data(symbol, year, month, day, hour, tick_format)
                    if df is not None and not df.empty:
                        all_data.append(df)

        final_df = pd.concat(all_data, ignore_index=True) if all_data else None

        if final_df is None or final_df.empty:
            print(f"No data retrieved for {symbol} in {year}.")
            return None
        
        file_path = DATA_DIR / "TICK" / "DUKASCOPY" / "EURUSD" / f"{year}.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(file_path, index=False)
        
        forex_tick_data = ForexTickData(
            source='DUKASCOPY',
            instrument=symbol,
            ask_column='ask',
            bid_column='bid',
            time_column='time',
            volume_column=None,
            df=final_df)
        
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

    Attributes
    ----------
    source : str
        The source of the Forex data.
    instrument : str
        The instrument for which the data is being handled.
    df : pd.DataFrame
        The DataFrame containing Forex tick data.

    Methods
    -------
    validate_dataframe()
        Validates that the DataFrame contains all required columns.
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
        Raises
        -------
        ValueError
            If the DataFrame does not contain the required columns.
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

        Parameters
        ----------
        pandas_freq : str
            The pandas frequency string for resampling (e.g., '1Min', '5Min', 'H').

        Returns
        -------
        pd.DataFrame
            A DataFrame with the converted OHLCV data, including bid OHLC, ask OHLC,
            and volume. Volume is calculated as the sum of the specified volume_column
            if present, otherwise as the count of ticks.
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
            The end date for the period. If None, uses the latest date in the DataFrame.

        Returns
        -------
        ForexCandleData
            A new ForexCandleData object with the specified period.
        """
        if start is None:
            start = self.df['date_gmt'].min()
        if end is None:
            end = self.df['date_gmt'].max()

        filtered_df = self.df[(self.df['date_gmt'] >= start) & (self.df['date_gmt'] <= end)].copy()
        return ForexCandleData(source=self.source, instrument=self.instrument, granularity=self.granularity, df=filtered_df)
    
    def analyse_save(self):
        """
        Analyzes the Forex data and saves it to a CSV file.

        Parameters
        ----------
        folder_path : Path
            The path where the CSV file will be saved.
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
        self.df['time_diff'].fillna(0, inplace=True)

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







def load_tick_data():
    """
    Main function to get user inputs, load data, and process Forex tick data.
    """
    print("Forex Tick Data Processor")
    print("=" * 30)

    # Get data source and instrument
    source_name = input("Enter the data source name (e.g., Dukascopy, FXCM): ").strip()
    instrument_name = input("Enter the instrument name (e.g., EURUSD, GBPJPY): ").strip()

    # Get file path
    while True:
        file_path_str = input("Enter the full path to your tick data CSV file: ").strip()
        tick_data_file = Path(file_path_str)
        if tick_data_file.is_file() and tick_data_file.suffix.lower() == '.csv':
            break
        else:
            print("Invalid file path or not a CSV file. Please try again.")

    # Get column names
    print("\nPlease specify the column names in your CSV file:")
    time_col = input("Name of the time/datetime column (e.g., 'Timestamp', 'date_gmt'): ").strip()
    bid_col = input("Name of the bid price column (e.g., 'Bid', 'bid_price'): ").strip()
    ask_col = input("Name of the ask price column (e.g., 'Ask', 'ask_price'): ").strip()

    volume_col_input = input("Name of the volume column (optional, press Enter if none): ").strip()
    volume_col = volume_col_input if volume_col_input else None

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

    # Load data
    try:
        print(f"\nLoading data from {tick_data_file}...")
        # Determine if header is present by peeking at the first few lines
        # A simple heuristic: if first line contains typical header names (like the ones user provided)
        # This is not foolproof but better than always assuming header=0
        try:
            preview_df = pd.read_csv(tick_data_file, nrows=5)
            has_header = any(col_name in preview_df.columns for col_name in [time_col, bid_col, ask_col])
            header_option = 0 if has_header else None
            if has_header:
                print("Detected header row in CSV.")
            else:
                print("Assuming no header row in CSV, or specified columns not in the first line as column names.")
        except Exception:
            print("Could not preview CSV for header detection, assuming header row is present.")
            header_option = 0


        raw_df = pd.read_csv(tick_data_file, header=header_option)

        # If no header was read, assign column names based on user input if counts match
        if header_option is None:
            expected_cols = [time_col, bid_col, ask_col]
            if volume_col:
                expected_cols.append(volume_col)

            if len(raw_df.columns) == len(expected_cols):
                 # Check if user provided names are just indices
                if all(c.isdigit() for c in expected_cols):
                    print("Warning: Column names provided are all digits. This might lead to issues if they are meant to be actual names.")
                    # If user provided indices as names, it's tricky.
                    # Let's assume for now if no header, and names are provided, they are the intended names.
                    # This part requires careful handling of user input.
                    # For now, if header=None, pandas assigns 0,1,2... We need to map.
                    # Let's assume user provided actual names, and we need to ensure they are used.
                    # The ForexTickData class expects these names to exist.

                    # Simplest for now: if no header, and user provides names,
                    # we assume the CSV was truly headerless and columns are in the order of input.
                    # This is a big assumption. A better way is to ask user for column *positions* if no header.
                    print(f"CSV loaded without headers. Attempting to use provided column names for the first {len(expected_cols)} columns.")
                    # This is risky, user must be sure about the column order.
                    # A safer approach would be to rename based on position *after* loading,
                    # but then we need to know which position corresponds to which name.

                    # For now, let's rename based on a naive assumption if header_option is None
                    # and user provided names. ForexTickData will validate.
                    # The read_csv with header=None results in numbered columns.
                    # We need to rename these.
                    rename_map = {}
                    current_cols = list(raw_df.columns) # These will be 0, 1, 2...
                    user_cols_map = {
                        time_col: None,
                        bid_col: None,
                        ask_col: None,
                    }
                    if volume_col:
                        user_cols_map[volume_col] = None

                    print(f"Please map your provided column names to the loaded column indices (0 to {len(current_cols)-1}):")
                    for i, u_col_name in enumerate([time_col, bid_col, ask_col] + ([volume_col] if volume_col else [])):
                        while True:
                            try:
                                col_idx = int(input(f"Which column index (0-{len(current_cols)-1}) corresponds to '{u_col_name}'? "))
                                if 0 <= col_idx < len(current_cols):
                                    rename_map[current_cols[col_idx]] = u_col_name
                                    break
                                else:
                                    print("Index out of bounds.")
                            except ValueError:
                                print("Invalid input. Please enter a number.")
                    raw_df.rename(columns=rename_map, inplace=True)
                    print("Columns renamed based on your mapping.")

            elif not has_header: # No header detected and column count mismatch
                 print(f"Warning: CSV loaded without headers, but number of columns in CSV ({len(raw_df.columns)}) "
                       f"does not match number of critical columns provided ({len(expected_cols)}). "
                       "Please ensure your column name inputs are correct.")


        print(f"Successfully loaded {len(raw_df)} rows.")

    except FileNotFoundError:
        print(f"Error: File not found at {tick_data_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {tick_data_file} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        return

    # Instantiate ForexTickData and process
    try:
        print("\nInitializing ForexTickData object...")
        forex_tick_data = ForexTickData(
            source=source_name,
            instrument=instrument_name,
            df=raw_df,
            time_column=time_col,
            bid_column=bid_col,
            ask_column=ask_col,
            volume_column=volume_col
        )
        print("ForexTickData object initialized successfully.")

        print(f"\nConverting data to {selected_granularity.name} granularity...")
        forex_data = forex_tick_data.to_candles(granularity=selected_granularity)
        forex_data.analyse_save()

        if not forex_data.df.empty:
            print(f"\nProcessing complete. Output files are in subdirectories of: {DATA_DIR}")
            print(f"Processed OHLCV data has {len(forex_data.df)} rows.")
        else:
            print("\nProcessing complete, but no OHLCV data was generated (e.g., due to empty input or filtering).")

    except ValueError as e:
        print(f"ValueError during processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

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
    downloader = DukascopyDataDownloader()
    # Get user inputs for download
    pair = input("Enter the currency pair (e.g., EURUSD): ").strip().upper()
    year = input("Enter the year to download (e.g., 2023): ").strip()

    # get user input for granularity
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

    # Download data
    try:
        print(f"\nDownloading {pair} data for {year} at {selected_granularity.name} granularity...")
        forex_tick_data = downloader.request_ticks_for_year(pair, year)
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
        print("0. Donwload Dukascopy Tick Data, convert, analyze, and save")
        print("1. Load Tick Data and convert it to OHLCV")
        print("2. Load multiple OHLCV files and combine them, then analyze and save")
        print("3. Exit")
        choice = input("Enter your choice (0-3): ").strip()
        if choice == '0':
            download_convert_save_dukascopy_data()
        elif choice == '1':
            load_tick_data()
        elif choice == '2':
            combine_and_analyze_multiple_files()
        elif choice == '3':
            print("Exiting the Forex Data Processor. Goodbye!")
            break
        else:
            print("Invalid choice. Please run the program again and choose a valid option.")
