import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Dict, Any
from typing import Optional
from pathlib import Path
from common.constants import DATA_DIR
from enum import Enum
import matplotlib.pyplot as plt

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
        self.df = df

        self.time_column = time_column
        self.bid_column = bid_column
        self.ask_column = ask_column
        self.volume_column = volume_column

        self.validate_dataframe()
        
    def validate_dataframe(self):
        """        Validates that the DataFrame contains all required columns.
        Raises  -------
        ValueError
            If the DataFrame does not contain the required columns.
        """
        required_columns = [self.time_column, self.bid_column, self.ask_column]
        if self.volume_column:
            required_columns.append(self.volume_column)

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

        # Ensure time column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_column]):
            self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
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
        granularity : str, optional
            The desired time granularity for the data (e.g., '1Min', '5Min', '15Min', 'H').
            Defaults to '15Min'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the converted OHLCV data.
        """

        start_time = self.df[self.time_column].min()
        end_time = self.df[self.time_column].max()

        # Create the file path based on source, instrument, and granularity, and start and end time
        folder_path = DATA_DIR / self.source / self.instrument / granularity.to_pandas_freq() / f"{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}" 

        ohlcv_df = self._convert(folder_path, granularity.to_pandas_freq())

        self.clean_analyze_save(ohlcv_df, folder_path, granularity_in_minutes=granularity.as_minutes())



        
    def _convert(self, granularity: str) -> pd.DataFrame:
        """
        Converts the tick data to OHLCV format for both bid and ask prices.

        Parameters
        ----------
        granularity : str, optional
            The desired time granularity for the data (e.g., '1Min', '5Min', '15Min', 'H').
            Defaults to '1Min'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the converted OHLCV data, including bid OHLC, ask OHLC,
            and volume. Volume is calculated as the sum of the specified volume_column
            if present, otherwise as the count of ticks.
        """
        print(f"Converting tick data to {granularity} OHLCV candles...")

        # Resample for Bid OHLC
        bid_ohlc = self.df[self.bid_column].resample(granularity).ohlc()
        bid_ohlc.columns = [f'bid_{col}' for col in bid_ohlc.columns]

        # Resample for Ask OHLC
        ask_ohlc = self.df[self.ask_column].resample(granularity).ohlc()
        ask_ohlc.columns = [f'ask_{col}' for col in ask_ohlc.columns]

        # Calculate Volume based on available columns
        if self.volume_column:
            # Use the sum of the specified volume_column if it exists
            # This would be for sources like Capital.com where 'last_trade_volume' might be available
            volume_series = self.df[self.volume_column].resample(granularity).sum()
            volume_series.name = 'volume' # Renaming for clarity
        else:
            # Fallback to tick count if no specific volume_column is provided
            # This is standard for Dukascopy tick data which lacks 'last_traded_volume'
            volume_series = self.df[self.bid_column].resample(granularity).size()
            volume_series.name = 'volume' # Renaming for clarity

        # Combine all into a single DataFrame
        ohlcv_df = pd.concat([bid_ohlc, ask_ohlc, volume_series], axis=1)

        ohlcv_df.reset_index(inplace=True)
        ohlcv_df.rename(columns={self.time_column: 'date_gmt'}, inplace=True)

        return ohlcv_df

    def _clean_analyze_save(self, df: pd.DataFrame, folder_path: Path, granularity_in_minutes: int):
        """
        Cleans, analyzes, and saves the OHLCV DataFrame to a CSV file.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            The DataFrame containing OHLCV data.
        folder_path : Path
            The path where the CSV file will be saved.
        granularity_in_minutes : int
            The granularity of the data in minutes.
        """
        print(f"Cleaning and analyzing OHLCV data for {self.instrument} at {granularity_in_minutes} minutes...")

        df['date_gmt'] = pd.to_datetime(df['date_gmt'])
        df = df.sort_values(by='date_gmt')

        # filter where volume is greater than 0
        df = df[df['volume'] > 0]

        # dropna
        df.dropna(inplace=True)
        # reset index
        df.reset_index(drop=True, inplace=True)

        df['time_diff'] = df['date_gmt'].diff().dt.total_seconds() / 60
        df['time_diff'] = df['time_diff'].fillna(0)

        weekend_in_minutes = 60 * 24 * 2 + granularity_in_minutes # 2 days in minutes
        missing_rows = df[(df['time_diff'] != granularity_in_minutes) | (df['time_diff'] != weekend_in_minutes)]

        missing_rows_file_path = folder_path / f"missing_rows.txt"
        with open(missing_rows_file_path, 'w') as f:
            # write a line for each missing row, explaining that right before this date x amount of rows were missing
            for index, row in missing_rows.iterrows():
                f.write(f"Missing rows before {row['date_gmt']}: {row['time_diff']/granularity_in_minutes}\n")

            #write the total number of missing rows
            f.write(f"Total number of missing rows: {missing_rows['time_diff'].sum() / granularity_in_minutes}\n")

        # plot close prices with missing rows highlighted
        plt.figure(figsize=(12, 6))
        plt.plot(df['date_gmt'], df['bid_close'], label='Bid Close', color='blue')
        plt.plot(df['date_gmt'], df['ask_close'], label='Ask Close', color='orange')
        plt.scatter(missing_rows['date_gmt'], missing_rows['bid_close'], color='red', label='Missing Rows', marker='x')
        plt.title(f"{self.instrument} OHLCV Data at {granularity_in_minutes} Minutes")
        plt.xlabel('Date GMT')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # save to file
        plt_file_path = folder_path / f"close_prices_with_missing_data.png"
        plt.savefig(plt_file_path)
        plt.close()

        # save a histogram of the volume amopunts (90th percentile)
        volume_90th_percentile = df['volume'].quantile(0.9)
        temp = df[df['volume'] <= volume_90th_percentile]
        plt.figure(figsize=(12, 6))
        plt.hist(temp['volume'], bins=100, color='blue', alpha=0.7)
        plt.title(f"{self.instrument} Volume Distribution at {granularity_in_minutes} Minutes")
        plt.xlabel('Volume')
        plt.ylabel('Frequency')
        plt.grid()
        # save to file
        volume_hist_file_path = folder_path / f"volume_distribution.png"
        plt.savefig(volume_hist_file_path)
        plt.close()
        print(f"Data cleaned and analyzed. Saving to {folder_path}...")

        # SAVING
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = f"data.csv"
        file_path = folder_path / file_name
        df.to_csv(file_path, index=False)
        print(f"Saved OHLCV data to {file_path}")


## main method that asks the user for the tick data file path, column names, and granularity