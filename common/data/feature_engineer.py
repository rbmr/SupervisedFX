import warnings
from functools import partial
from typing import Callable, List, Self, Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

# Suppress only the specific PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

class FeatureEngineer:

    _UNSET = object()

    def __init__(self, remove_original_columns = _UNSET):
        self._pipeline_steps: List[Callable[[pd.DataFrame], None]] = []
        self.remove_original_columns = remove_original_columns

    def add(self, func: Callable[..., None], **kwargs) -> Self:
        """
        Add a step to the pipeline. Function should modify dataframe in place.
        """
        self._pipeline_steps.append(partial(func, **kwargs))
        return self
    
    def run(self, df: pd.DataFrame, remove_original_columns=True) -> pd.DataFrame:
        """
        Run the pipeline on the given DataFrame.
        Applies each of the steps in the pipeline to the dataframe in place.
        """
        if self.remove_original_columns is not self._UNSET:
            remove_original_columns = self.remove_original_columns

        df = df.copy(deep=True)  # Avoid modifying the original DataFrame

        original_columns = df.columns.tolist()

        for func in self._pipeline_steps:
            func(df)

        if remove_original_columns:
            df.drop(columns=original_columns, inplace=True, errors='ignore')
        
        return df
    

# #######################################
# # Time Analysis                     #
# #######################################

def _norm_time_of_day(dt_series: pd.Series):
    """Converts a time column to a [0,1] range of time of day."""
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        raise ValueError("series must be of datetime type.")
    return (
        dt_series.dt.hour * (60 * 60) +
        dt_series.dt.minute * 60 +
        dt_series.dt.second +
        dt_series.dt.microsecond / 1_000_000
    ) / (60 * 60 * 24)

def _norm_time_of_week(dt_series: pd.Series):
    """Converts a time column to a [0,1] range of time of week."""
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        raise ValueError("series must be of datetime type.")
    return (
        dt_series.dt.weekday * (60 * 60 * 24) +  # days since Monday
        dt_series.dt.hour * (60 * 60) +
        dt_series.dt.minute * 60 +
        dt_series.dt.second +
        dt_series.dt.microsecond / 1_000_000
    ) / (7 * 60 * 60 * 24)

def lin_24h(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    df['lin_24h'] = _norm_time_of_day(df['date_gmt'])

def lin_7d(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    df['lin_7d'] = _norm_time_of_week(df['date_gmt'])

def sin_7d(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntow = _norm_time_of_week(df['date_gmt'])
    df['sin_7d'] = np.sin(ntow * np.pi * 2)

def sin_24h(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntod = _norm_time_of_day(df['date_gmt'])
    df['sin_24h'] = np.sin(ntod * np.pi * 2)

def cos_7d(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntow = _norm_time_of_week(df['date_gmt'])
    df['cos_7d'] = np.cos(ntow * np.pi * 2)

def cos_24h(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntod = _norm_time_of_day(df['date_gmt'])
    df['cos_24h'] = np.cos(ntod * np.pi * 2)

def complex_7d(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntow = _norm_time_of_week(df['date_gmt'])
    df['sin_7d'] = np.sin(ntow * np.pi * 2)
    df['cos_7d'] = np.cos(ntow * np.pi * 2)

def complex_24h(df: pd.DataFrame):
    if 'date_gmt' not in df.columns:
        raise ValueError("DataFrame must contain 'date_gmt' column with datetime values.")
    ntod = _norm_time_of_day(df['date_gmt'])
    df['sin_24h'] = np.sin(ntod * np.pi * 2)
    df['cos_24h'] = np.cos(ntod * np.pi * 2)


# ####################################### #
# # End Time Analysis                   # #
# ####################################### #


# ####################################### #
# # Technical Analysis                  # #
# ####################################### #

# ---------------------- #
# -- TREND Indicators -- #
# ---------------------- #

def sma(df: pd.DataFrame, window: int, column: str = 'close_bid'):
    """
    Calculate the Simple Moving Average (SMA) for a given column.
    """
    df[f'sma_{window}_{column}'] = df[column].rolling(window=window).mean()

def ema(df: pd.DataFrame, window: int, column: str = 'close_bid'):
    """
    Calculate the Exponential Moving Average (EMA) for a given column.
    """
    df[f'ema_{window}_{column}'] = df[column].ewm(span=window, adjust=False).mean()

def kama(df: pd.DataFrame, window: int = 10, fast: int = 2, slow: int = 30, column: str = 'close_bid'):
    """
    Calculate the Kaufman's Adaptive Moving Average (KAMA) for a given column.
    KAMA adjusts its sensitivity based on the volatility of the price.
    """
    # Calculate Directional Movement
    change = abs(df[column] - df[column].shift(window))

    # Calculate Volatility
    volatility = (abs(df[column] - df[column].shift(1))).rolling(window=window).sum()

    # Calculate Efficiency Ratio
    er = change / volatility

    # Calculate Smoothing Constant
    fastest = 2 / (fast + 1)
    slowest = 2 / (slow + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2

    # Calculate KAMA
    kama = pd.Series(index=df.index, dtype=float)
    kama.iloc[window-1] = df[column].iloc[window-1] # Start with the price at period n

    for i in range(window, len(df)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df[column].iloc[i] - kama.iloc[i-1])

    df[f'kama_{window}_{column}'] = kama

def vwap(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Volume Weighted Average Price (VWAP) for a given column.
    VWAP is the average price a security has traded at throughout the day, based on both volume and price.
    """
    cumulative_volume = df['volume'].cumsum()
    cumulative_vwap = (df['close_bid'] * df['volume']).cumsum() / cumulative_volume
    df[f'vwap_{window}'] = cumulative_vwap.rolling(window=window).mean()
    df[f'vwap_{window}'] = df[f'vwap_{window}'].fillna(0)  # Fill NaN values with 0

def adx(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Average Directional Index (ADX) for trend strength.
    """
    high = df['high_bid']
    low = df['low_bid']
    close = df['close_bid']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    # Directional Movement
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

    plus_di = 100 * (plus_dm.rolling(window=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).sum() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()

    df['adx'] = adx

def parabolic_sar(df: pd.DataFrame, acceleration_factor: float = 0.02, max_acceleration: float = 0.2):
    """
    Calculate the Parabolic SAR (Stop and Reverse) indicator.
    """
    # Use bid prices for calculation
    high = df['high_bid']
    low = df['low_bid']
    close = df['close_bid']

    # Initial values
    initial_af = acceleration_factor
    max_af = max_acceleration
    sar = [close[0]]
    ep = [high[0]]
    af = [initial_af]
    trend = [1]  # 1 for uptrend, -1 for downtrend

    for i in range(1, len(df)):
        # Determine current trend
        if trend[-1] == 1:  # Uptrend
            sar_i = sar[-1] + af[-1] * (ep[-1] - sar[-1])
            if low[i] < sar_i:
                # Switch to downtrend
                trend.append(-1)
                sar_i = ep[-1]
                ep_i = low[i]
                af_i = initial_af
            else:
                trend.append(1)
                ep_i = max(ep[-1], high[i])
                if ep_i > ep[-1]:
                    af_i = min(max_af, af[-1] + initial_af)
                else:
                    af_i = af[-1]
        else:  # Downtrend
            sar_i = sar[-1] - af[-1] * (sar[-1] - ep[-1])
            if high[i] > sar_i:
                # Switch to uptrend
                trend.append(1)
                sar_i = ep[-1]
                ep_i = high[i]
                af_i = initial_af
            else:
                trend.append(-1)
                ep_i = min(ep[-1], low[i])
                if ep_i < ep[-1]:
                    af_i = min(max_af, af[-1] + initial_af)
                else:
                    af_i = af[-1]

        sar.append(sar_i)
        ep.append(ep_i)
        af.append(af_i)

    df['sar'] = sar
    return df

# -------------------------- #
# -- END TREND Indicators -- #
# -------------------------- #

#---------------------------------------#

# ------------------------- #
# -- MOMENTUM Indicators -- #
# ------------------------- #


def macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) and Signal Line.
    """
    short_ema = df['close_bid'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close_bid'].ewm(span=long_window, adjust=False).mean()
    
    df['macd'] = short_ema - long_ema
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

def rsi(df, window=14):
    """
    Calculate the Relative Strength Index (RSI) column.
    """
    delta = df['close_bid'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss

    # replace rs with 1 if NaN, 100 if rs is np.inf, or 0 if rs is -np.inf
    rs = rs.fillna(1)
    rs = rs.replace(np.inf, 100)
    rs = rs.replace(-np.inf, 0)

    rsi = 100 - (100 / (1 + rs))
    df[f'rsi_{window}'] = rsi

def stochastic_oscillator(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Stochastic Oscillator %K and %D.
    %K is the current close relative to the range of the last 'window' periods.
    %D is a smoothed version of %K.
    """
    low_min = df['low_bid'].rolling(window=window).min()
    high_max = df['high_bid'].rolling(window=window).max()
    
    df['stoch_k'] = 100 * (df['close_bid'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()  # 3-period smoothing for %D

def historic_pct_change(df: pd.DataFrame, window: int = 14):
    """
    Calculate the historical percentage change over a given window.
    This is a momentum indicator that shows how much the price has changed over the window.
    """
    df[f'historic_pct_change_{window}'] = df['close_bid'].pct_change(periods=window) * 100
    df[f'historic_pct_change_{window}'] = df[f'historic_pct_change_{window}'].fillna(0)  # Fill NaN values with 0

def cci(df: pd.DataFrame, window: int = 20):
    """
    Calculate the Commodity Channel Index (CCI) for a given column.
    CCI measures the deviation of the price from its average.
    """
    typical_price = (df['high_bid'] + df['low_bid'] + df['close_bid']) / 3
    sma_typical_price = typical_price.rolling(window=window).mean()
    mean_deviation = (typical_price - sma_typical_price).abs().rolling(window=window).mean()
    
    df[f'cci_{window}'] = (typical_price - sma_typical_price) / (0.015 * mean_deviation)

def williams_r(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Williams %R indicator.
    It measures the current closing price relative to the high-low range over a specified period.
    """
    high_max = df['high_bid'].rolling(window=window).max()
    low_min = df['low_bid'].rolling(window=window).min()
    
    df[f'williams_r_{window}'] = -100 * (high_max - df['close_bid']) / (high_max - low_min)
    df[f'williams_r_{window}'] = df[f'williams_r_{window}'].fillna(0)  # Fill NaN values with 0

def mfi(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Money Flow Index (MFI) for a given column.
    MFI is a momentum indicator that measures the flow of money into and out of a security.
    """
    typical_price = (df['high_bid'] + df['low_bid'] + df['close_bid']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(df['close_bid'].diff() > 0, 0).rolling(window=window).sum()
    negative_flow = money_flow.where(df['close_bid'].diff() < 0, 0).rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df[f'mfi_{window}'] = mfi.fillna(0)  # Fill NaN values with 0

def cmf(df: pd.DataFrame, window: int = 20):
    """
    Calculate the Chaikin Money Flow (CMF) for a given column.
    CMF measures the buying and selling pressure for a security over a specified period.
    """
    money_flow_multiplier = ((df['close_bid'] - df['low_bid']) - (df['high_bid'] - df['close_bid'])) / (df['high_bid'] - df['low_bid'])
    money_flow_volume = money_flow_multiplier * df['volume']
    
    df[f'cmf_{window}'] = money_flow_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    df[f'cmf_{window}'] = df[f'cmf_{window}'].fillna(0)  # Fill NaN values with 0

def obv(df: pd.DataFrame):
    """
    Calculate the On-Balance Volume (OBV) indicator.
    OBV is a cumulative volume indicator that adds volume on up days and subtracts volume on down days.
    """
    df['obv'] = 0
    for i in range(1, len(df)):
        if df['close_bid'].iloc[i] > df['close_bid'].iloc[i - 1]:
            df.at[i, 'obv'] = df['obv'].iloc[i - 1] + df['volume'].iloc[i]
        elif df['close_bid'].iloc[i] < df['close_bid'].iloc[i - 1]:
            df.at[i, 'obv'] = df['obv'].iloc[i - 1] - df['volume'].iloc[i]
        else:
            df.at[i, 'obv'] = df['obv'].iloc[i - 1]

def ad_line(df: pd.DataFrame):
    """
    Calculate the Accumulation/Distribution Line (AD Line) for a given column.
    The AD Line is a cumulative indicator that shows the flow of money into and out of a security.
    """
    ad = ((df['close_bid'] - df['low_bid']) - (df['high_bid'] - df['close_bid'])) / (df['high_bid'] - df['low_bid']) * df['volume']
    df['ad_line'] = ad.cumsum()


# ----------------------------- #
# -- END MOMENTUM Indicators -- #
# ----------------------------- #

#---------------------------------------#

# --------------------------- #
# -- VOLATILITY Indicators -- #
# --------------------------- #

def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: float = 2.0):
    """
    Calculate Bollinger Bands for a given column.
    """
    sma_col = f'sma_{window}_close_bid'
    df[sma_col] = df['close_bid'].rolling(window=window).mean()
    rolling_std = df['close_bid'].rolling(window=window).std()
    
    df[f'bb_upper_{window}'] = df[sma_col] + (num_std_dev * rolling_std)
    df[f'bb_lower_{window}'] = df[sma_col] - (num_std_dev * rolling_std)

def atr(df: pd.DataFrame, window: int = 14, column_high: str = "high_bid", column_low: str = "low_bid", column_close: str = "close_bid"):
    """
    Adds an 'atr_{window}' column to df containing the rolling ATR.
    ATR = rolling mean of True Range over `window` bars.
    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    """
    high = df[column_high]
    low = df[column_low]
    close_shifted = df[column_close].shift(1).bfill()

    # Compute True Range using np.maximum, then convert to Series
    true_range_array = np.maximum.reduce([
        high - low,
        (high - close_shifted).abs(),
        (low - close_shifted).abs()
    ])

    true_range = pd.Series(true_range_array, index=df.index)

    df[f"atr_{window}"] = true_range.rolling(window=window, min_periods=1).mean().fillna(0)

def chaikin_volatility(df: pd.DataFrame, ema_window: int = 10, roc_period: int = 10):
    """
    Calculate the Chaikin Volatility indicator.
    It measures the rate of change of an EMA of the High-Low range.
    """
    # Step 1: Calculate the EMA of the High-Low difference
    high_low_range = df['high_bid'] - df['low_bid']
    ema_high_low = high_low_range.ewm(span=ema_window, adjust=False).mean()
    
    # Step 2: Calculate the rate of change of that EMA
    roc_ema = ema_high_low.pct_change(periods=roc_period)
    
    df[f'chaikin_vol_{ema_window}_{roc_period}'] = roc_ema
    df[f'chaikin_vol_{ema_window}_{roc_period}'] = df[f'chaikin_vol_{ema_window}_{roc_period}'].fillna(0)

def ease_of_movement(df: pd.DataFrame, window: int = 14):
    """
    Calculate the Ease of Movement (EOM) indicator.
    EOM highlights the relationship between price change and volume, showing
    how easily prices move for a given amount of volume.
    """
    # Calculate the distance moved from the previous period's midpoint
    high = df['high_bid']
    low = df['low_bid']
    volume = df['volume']

    distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)

    # Calculate the price range for the current period
    price_range = high - low

    # Replace zero price_range with NaN to avoid division by zero errors.
    # We will handle the resulting NaNs later.
    price_range = price_range.replace(0, np.nan)

    # Calculate the Box Ratio, which relates volume to the price range
    # A large volume scale is used to keep the resulting EOM values in a manageable range.
    box_ratio = (volume / 100_000_000) / price_range

    # Calculate the 1-period Ease of Movement
    eom_1_period = distance_moved / box_ratio

    # The EOM indicator is typically a Simple Moving Average of the 1-period EOM
    eom = eom_1_period.rolling(window=window).mean()

    # Add the new feature to the DataFrame and fill any NaNs with 0
    df[f'eom_{window}'] = eom
    df[f'eom_{window}'] = df[f'eom_{window}'].fillna(0)

# ------------------------------- #
# -- END VOLATILITY Indicators -- #
# ------------------------------- #

# ####################################### #
# # End Technical Analysis              # #
# ####################################### #


# ############################################### #
# # NORMALIZATION TRANSFORMATION SCALIUNG       # #
# ############################################### #

def as_robust_norm(df: pd.DataFrame, column: str, window: int = 500):
    """
    Normalizes the specified column in-place using rolling robust normalization.
    This method uses the rolling median and IQR to reduce the influence of outliers.
    """
    log_column = np.log1p(df[column])
    rolling_median = log_column.rolling(window=window, min_periods=1, center=False).median()
    q75 = log_column.rolling(window=window, min_periods=1, center=False).quantile(0.75)
    q25 = log_column.rolling(window=window, min_periods=1, center=False).quantile(0.25)
    iqr = q75 - q25
    df[column] = (log_column - rolling_median) / (iqr + 1e-6)

def as_pct_change(df: pd.DataFrame, column: str, periods: int = 1):
    """
    Normalize a column as percentage change.
    """
    df[f'{column}'] = df[column].pct_change(periods=periods)
    df[f'{column}'] = df[f'{column}'].fillna(0)  # Fill NaN values with 0

def as_ratio_of_other_column(df: pd.DataFrame, column: str, other_column: str):
    """
    Normalize a column as a ratio of another column.
    """
    df[f'{column}'] = df[column] / df[other_column]

    # minus 1 to center around 0
    df[f'{column}'] = df[f'{column}'] - 1

    df[f'{column}'] = df[f'{column}'].fillna(0)  # Fill NaN values with 0
    df[f'{column}'] = df[f'{column}'].replace(np.inf, 0)
      # Replace inf with 0

def as_z_score(df: pd.DataFrame, column: str, window: int = 50):
    """
    Normalize a column as z-score with a window.
    To reduce NaNs, we use 0:index window for the rows where index<window.
    """
    assert window >= 0
    if window == 0:
        df[column] = (df[column] - df[column].mean()) / df.std()
    else:
        df[column] = (df[column] - df[column].rolling(window=window, min_periods=1).mean()) / df[column].rolling(window=window, min_periods=1).std()

    df[column] = df[column].fillna(0)  # Fill NaN values with 0
    df[column] = df[column].replace(np.inf, 0)  # Replace inf with 0

def as_min_max_window(df: pd.DataFrame, column: str, window: int = 50):
    """
    Normalize a column as min-max scaling with a rolling window.
    """
    df[f'{column}'] = (df[column] - df[column].rolling(window=window, min_periods=1).min()) / (df[column].rolling(window=window, min_periods=1).max() - df[column].rolling(window=window, min_periods=1).min())
    
    # center around 0
    df[f'{column}'] = 2 * df[f'{column}'] - 1
    
    df[f'{column}'] = df[f'{column}'].fillna(0)  # Fill NaN values with 0
    df[f'{column}'] = df[f'{column}'].replace(np.inf, 0)  # Replace inf with 0

def as_min_max_fixed(df: pd.DataFrame, column: str, min: int = 0, max: int = 100):
    """
    Normalize a column as min-max scaling with a lookahead bias.
    This is not recommended for training, but can be used for testing.
    """
    df[f'{column}'] = (df[column] - min) / (max - min)

    # center around 0
    df[f'{column}'] = 2 * df[f'{column}'] - 1

    df[f'{column}'] = df[f'{column}'].fillna(0)  # Fill NaN values with 0
    df[f'{column}'] = df[f'{column}'].replace(np.inf, 0)  # Replace inf with 0

def as_below_above_column(df: pd.DataFrame, column: str, other_column: str):
    """
    Normalize a column as below/above another column.
    This will create a new column with 1 if the value is above the other column, -1 if below, and 0 if equal.
    """
    df[f'{column}'] = np.where(df[column] > df[other_column], 1, 
                                                           np.where(df[column] < df[other_column], -1, 0))
    df[f'{column}'] = df[f'{column}'].fillna(0)  # Fill NaN values with 0


## other
def apply_column(df: pd.DataFrame, fn: Callable[[Any], Any], column: str):
    """
    Applies a function all the values of a column in place.
    """
    df[column] = df[column].apply(fn)

def remove_columns(df: pd.DataFrame, columns: List[str]):
    """
    Remove specified columns from the DataFrame.
    """
    df.drop(columns=columns, inplace=True, errors='ignore')

def remove_ohlcv(df: pd.DataFrame):
    """
    Remove OHLCV columns from the DataFrame.
    """
    ohlcv_columns = ['volume', 'date_gmt',
                     'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid',
                     'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask']
    remove_columns(df, ohlcv_columns)

def history_lookback(df: pd.DataFrame, lookback_window_size: int, columns: List[str] = None, not_columns: List[str] = None, step: int = 1):
    """
    Create a history lookback window for the DataFrame.
    """
    if lookback_window_size == 0:
        return
    if columns is None:
        columns = df.columns.tolist()
    if not_columns is None:
        not_columns = ()

    # for each column make a new column shifted by 1, 2, ..., lookback_window_size
    for col in columns:
        if col in not_columns:
            continue
        for i in range(1, lookback_window_size + 1, step):
            df[f'{col}_shift_{i}'] = df[col].shift(i)

def copy_columns(df: pd.DataFrame, source_columns: List[str], target_columns: List[str]):
    if len(source_columns) != len(target_columns):
        raise ValueError("len columns and target_columns are not the same.")
    for source_column, target_column in zip(source_columns, target_columns):
        copy_column(df, source_column, target_column)

def copy_column(df: pd.DataFrame, source_column: str, target_column: str):
    """
    Copy a column from source to target.
    """
    df[target_column] = df[source_column].copy()
    df[target_column] = df[target_column].fillna(0)  # Fill NaN values with 0
    df[target_column] = df[target_column].replace(np.inf, 0)  # Replace inf with 0

# ############################################### #
# # END NORMALIZATION TRANSFORMATION SCALIUNG   # #
# ############################################### #

