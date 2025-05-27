import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt # Duplicate import, but keeping as in original

# Mock DATA_DIR for standalone execution
DATA_DIR = Path("./forex_data_output") # You can change this to your preferred output directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

    def convert_analyze_save(self, granularity: Timeframe) -> pd.DataFrame:
        """
        Converts the tick data to OHLCV format and saves it to a CSV file.

        Parameters
        ----------
        granularity : Timeframe
            The desired time granularity for the data.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the converted OHLCV data.
        """
        if self.df.empty:
            print("DataFrame is empty. Skipping conversion and analysis.")
            return pd.DataFrame()

        # Ensure the index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
             if self.time_column in self.df.columns:
                self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
                self.df = self.df.set_index(self.time_column)
             elif self.df.index.name == self.time_column and not pd.api.types.is_datetime64_any_dtype(self.df.index):
                 self.df.index = pd.to_datetime(self.df.index)
             else:
                raise ValueError("DataFrame index is not a DatetimeIndex and time column cannot be set as index.")


        start_time = self.df.index.min()
        end_time = self.df.index.max()

        # Create the file path based on source, instrument, and granularity, and start and end time
        folder_path = DATA_DIR / self.source / self.instrument / granularity.to_pandas_freq() / f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}"

        ohlcv_df = self._convert(granularity.to_pandas_freq())

        if ohlcv_df.empty:
            print(f"No data to process for {self.instrument} at {granularity.to_pandas_freq()} granularity.")
            return ohlcv_df

        self._clean_analyze_save(ohlcv_df, folder_path, granularity_in_minutes=granularity.as_minutes())
        return ohlcv_df


    def _convert(self, pandas_freq: str) -> pd.DataFrame:
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
        print(f"Converting tick data to {pandas_freq} OHLCV candles...")

        # Ensure the index is datetime for resampling
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex for resampling.")

        # Resample for Bid OHLC
        bid_ohlc = self.df[self.bid_column].resample(pandas_freq).ohlc()
        bid_ohlc.columns = [f'bid_{col}' for col in bid_ohlc.columns]

        # Resample for Ask OHLC
        ask_ohlc = self.df[self.ask_column].resample(pandas_freq).ohlc()
        ask_ohlc.columns = [f'ask_{col}' for col in ask_ohlc.columns]

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


        return ohlcv_df

    def _clean_analyze_save(self, df: pd.DataFrame, folder_path: Path, granularity_in_minutes: int):
        """
        Cleans, analyzes, and saves the OHLCV DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing OHLCV data.
        folder_path : Path
            The path where the CSV file will be saved.
        granularity_in_minutes : int
            The granularity of the data in minutes.
        """
        if 'date_gmt' not in df.columns:
            print("Error: 'date_gmt' column missing in OHLCV data. Skipping clean, analyze, save.")
            return

        print(f"Cleaning and analyzing OHLCV data for {self.instrument} at {granularity_in_minutes} minutes...")

        df['date_gmt'] = pd.to_datetime(df['date_gmt'])
        df = df.sort_values(by='date_gmt')

        # filter where volume is greater than 0
        df = df[df['volume'] > 0]

        # dropna
        df.dropna(inplace=True)
        # reset index
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            print("DataFrame is empty after cleaning (volume > 0 and dropna). No data to save or analyze.")
            folder_path.mkdir(parents=True, exist_ok=True) # Create folder even if no data
            no_data_file = folder_path / "no_data_after_cleaning.txt"
            with open(no_data_file, 'w') as f:
                f.write(f"No data remained for {self.instrument} at {granularity_in_minutes} min granularity after filtering for volume > 0 and dropping NA values.\n")
            return


        df['time_diff'] = df['date_gmt'].diff().dt.total_seconds() / 60
        df['time_diff'].fillna(0, inplace=True) # Fill NaN for the first row

        # Define expected difference, allowing for weekend gaps.
        # Forex market closes on Friday evening and reopens Sunday evening.
        # A typical weekend gap might be around 48 hours (2 days) + time until next candle.
        # For simplicity, let's define a generous weekend gap threshold.
        # Consider a gap from Friday 21:00 GMT to Sunday 21:00 GMT = 48 hours.
        # If granularity is H1, the next candle after Friday 21:00 is Sunday 22:00.
        # The diff would be 48 hours + 1 hour = 49 hours for H1.
        # For M15, diff would be 48h + 15min.
        # A more robust way is to check if the time diff is significantly larger than granularity
        # and also corresponds to a weekend.
        # Max normal gap = granularity_in_minutes
        # Min weekend gap (approx) = 2 * 24 * 60 (can be less, e.g. Fri 22:00 to Sun 22:00)
        # Max weekend gap (approx) = 2 * 24 * 60 + some hours + granularity_in_minutes

        # Simplified logic for identifying non-standard gaps (not exactly weekend or exact granularity)
        # A gap is "missing" if it's NOT the expected granularity AND NOT a typical weekend gap.
        # Typical weekend gap: Fri PM to Sun PM. For daily, it's just 2 days.
        # Let's flag anything that isn't `granularity_in_minutes` or a multiple of it due to market close.
        # A simple check: if time_diff > granularity_in_minutes and time_diff is not a known large gap (like weekend)
        # The original logic: missing_rows = df[(df['time_diff'] != granularity_in_minutes) | (df['time_diff'] != weekend_in_minutes)]
        # This OR condition means it flags rows if EITHER it's not granularity_in_minutes OR it's not weekend_in_minutes.
        # This will flag almost everything if weekend_in_minutes is a single value.
        # It should be: rows where time_diff is not granularity_in_minutes AND time_diff is not a weekend gap.

        # Corrected logic for missing rows:
        # A row indicates a gap *before* it. The 'time_diff' is the duration since the *previous* candle.
        # We are looking for gaps where the time difference to the *previous* candle is not the standard granularity,
        # and also not a standard weekend gap.
        standard_gap = granularity_in_minutes
        # Approximate weekend gap (e.g. Friday 21:00 to Sunday 21:00 = 48 hours = 2880 minutes)
        # This needs to be more flexible. Forex closes roughly Friday 21:00/22:00 GMT to Sunday 21:00/22:00 GMT.
        # Let's allow a range for weekend gaps. min_weekend_gap ~40 hours, max_weekend_gap ~55 hours.
        # This depends on the exact closing/opening times which vary by broker/source.
        # For simplicity, the original code used a fixed `weekend_in_minutes`.
        # Let's use a threshold: if a gap is > granularity_in_minutes * factor_for_small_gaps AND not clearly a weekend.
        # For now, stick to a simplified version of the original intent: highlight unexpected gaps.
        # We consider a row as "following a gap" if 'time_diff' is unusual.
        # The time_diff of the first valid row after a gap will be large.

        # Filter for rows where the preceding gap was NOT the expected granularity
        # and also NOT a typical weekend gap.
        # Weekend gaps can be tricky because their exact duration varies slightly.
        # Let's define a weekend gap as roughly 2 days.
        # A common pattern: Friday close (e.g., 21:00 UTC) to Sunday open (e.g., 21:00 UTC).
        # This is a ~48-hour gap. The `time_diff` for the first candle after the weekend
        # would be `granularity_in_minutes + ~48 hours`.

        # Example: H1 data (60 min). First candle after weekend. Previous was Friday 20:00-21:00. Next is Sunday 21:00-22:00.
        # Time diff could be (Sunday 21:00 - Friday 21:00) = 48 hours = 2880 minutes.
        # This is the time from the *start* of the previous candle interval to the *start* of the current.
        # So time_diff should be around 2 days if it's a weekend.

        # Let's consider a gap "missing" if it's greater than the granularity but not a plausible weekend gap.
        # A plausible weekend gap is roughly 2 days (2880 minutes) up to 2 days + a few hours.
        # For M1 data, weekend_in_minutes = 2880 + 1 = 2881
        # For H1 data, weekend_in_minutes = 2880 + 60 = 2940

        # The original logic:
        # weekend_in_minutes = 60 * 24 * 2 + granularity_in_minutes
        # This seems like a reasonable approximation for the time difference for the first candle after a weekend.
        # missing_rows = df[(df['time_diff'] != granularity_in_minutes) | (df['time_diff'] != weekend_in_minutes)]
        # This should be AND, or rather, a check for unexpected gaps.
        # A gap is unexpected if: (time_diff > granularity_in_minutes) AND (time_diff is NOT a weekend_gap)
        # Let's refine:
        # A time_diff is "regular" if time_diff is approximately granularity_in_minutes.
        # A time_diff is a "weekend" if time_diff is approximately weekend_duration_base + granularity_in_minutes.
        # weekend_duration_base is approx 2 days.
        # We use a small tolerance for floating point comparisons.
        tolerance = 0.1 * granularity_in_minutes # Allow 10% deviation for the base granularity
        weekend_base_duration_minutes = 2 * 24 * 60 # Approx 48 hours

        is_standard_gap = np.isclose(df['time_diff'], granularity_in_minutes, atol=tolerance)
        is_weekend_gap = np.isclose(df['time_diff'], weekend_base_duration_minutes + granularity_in_minutes, atol=granularity_in_minutes * 2) # Wider tolerance for weekends

        # Rows that are NOT preceded by a standard gap AND NOT preceded by a weekend gap
        # (and ignoring the first row which has time_diff = 0)
        missing_rows_mask = (df['time_diff'] > 0) & ~is_standard_gap & ~is_weekend_gap
        missing_rows = df[missing_rows_mask]


        folder_path.mkdir(parents=True, exist_ok=True) # Ensure folder exists before writing files

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
        plt.plot(df['date_gmt'], df['bid_close'], label='Bid Close', color='blue', linewidth=0.8)
        plt.plot(df['date_gmt'], df['ask_close'], label='Ask Close', color='orange', linewidth=0.8, alpha=0.7)

        # Highlight the points *after* an unusual gap
        if not missing_rows.empty:
            plt.scatter(missing_rows['date_gmt'], missing_rows['bid_close'], color='red', label='After Unusual Gap', marker='x', s=50, zorder=5)

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
        if 'volume' in df.columns and not df['volume'].empty:
            volume_99th_percentile = df['volume'].quantile(0.99) # Use 99th to see more of the tail
            temp_df_volume = df[df['volume'] <= volume_99th_percentile]
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

        print(f"Data cleaned and analyzed. Saving to {folder_path}...")

        # SAVING
        # Folder creation is now at the start of the function
        file_name = f"data_ohlcv.csv"
        file_path = folder_path / file_name
        df.to_csv(file_path, index=False)
        print(f"Saved OHLCV data to {file_path}")


def main():
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
        forex_data = ForexTickData(
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
        ohlcv_df = forex_data.convert_analyze_save(granularity=selected_granularity)

        if not ohlcv_df.empty:
            print(f"\nProcessing complete. Output files are in subdirectories of: {DATA_DIR}")
            print(f"Processed OHLCV data has {len(ohlcv_df)} rows.")
        else:
            print("\nProcessing complete, but no OHLCV data was generated (e.g., due to empty input or filtering).")

    except ValueError as e:
        print(f"ValueError during processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Create a dummy CSV for testing if one doesn't exist
    dummy_csv_path = Path("dummy_tick_data.csv")
    if not dummy_csv_path.exists():
        print(f"Creating a dummy CSV file for testing: {dummy_csv_path}")
        num_rows = 10000
        start_time = pd.Timestamp('2023-01-01 00:00:00')
        time_deltas = np.random.randint(1, 60, num_rows).cumsum() # irregular tick arrivals
        timestamps = [start_time + pd.Timedelta(seconds=int(s)) for s in time_deltas]
        bids = 1.10000 + np.random.randn(num_rows) * 0.0001
        asks = bids + np.random.uniform(0.00005, 0.00020, num_rows)
        volumes = np.random.randint(1, 100, num_rows)
        dummy_data = pd.DataFrame({
            'Timestamp': timestamps,
            'BidPrice': bids,
            'AskPrice': asks,
            'Volume': volumes
        })
        dummy_data.to_csv(dummy_csv_path, index=False)
        print(f"Dummy CSV created. When prompted, you can use:\n"
              f"File path: {dummy_csv_path.resolve()}\n"
              f"Time column: Timestamp\n"
              f"Bid column: BidPrice\n"
              f"Ask column: AskPrice\n"
              f"Volume column: Volume\n")

    main()