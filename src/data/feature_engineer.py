import re
from copy import deepcopy
from typing import Callable, List

import numpy as np
import pandas as pd

from src.scripts import df_to_np_dict, shift, parse_args


class FeatureEngineer:
    """
    Class to efficiently extract features from a dataframe-like object.
    - feature methods must only ADD or READ columns, no columns are modified.
    - feature methods must return numpy arrays.
    - feature methods must not access the future (each index i of the resulting array may only be determined using indices 0...i).
    - feature methods must have a proper reason for padding and filling of values.
    - get_all must return a numpy array where the first dimension is the same length as the input data.
    """

    def __init__(self, data: dict[str, np.ndarray] | pd.DataFrame):
        if isinstance(data, pd.DataFrame):
            self._data = df_to_np_dict(data)
            self._len = len(data)
        elif isinstance(data, dict):
            self._data = deepcopy(data)
            self._len = None
            for key, arr in self._data.items():
                assert isinstance(key, str), f"keys must be of type string, was {type(key)}"
                assert isinstance(arr, np.ndarray), f"values must be of type ndarray, was {type(arr)}"
                assert arr.ndim == 1, f"arrays must be 1D, shape was {arr.shape}"
                if self._len is None:
                    self._len = len(arr)
                assert self._len == len(arr), "all arrays must have equal length"
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def get(self, column: str) -> np.ndarray:
        """
        Gets (or computes) a single column from the internal data.
        """
        if column in self._data:
            return self._data[column]
        match = re.match(r"([a-zA-Z_][a-zA-Z_0-9]*)(?:\((.*)\))?$", column.strip())
        assert match, f"'{column}' could not be parsed to a function."
        func_name, args_str = match.groups()
        assert func_name not in ['get', 'get_all', '__init__'], f"{func_name} is protected."
        assert hasattr(self, func_name), f"{func_name} isn't recognized."
        res = getattr(self, func_name)(*parse_args(args_str))
        assert isinstance(res, np.ndarray), f"Invalid result type: expected {np.ndarray}, was {type(res)}"
        assert res.ndim == 1, f"Invalid result ndim: expected 1, was {res.ndim}"
        self._data[column] = res
        return res

    def get_all(self, columns: list[str]) -> np.ndarray:
        """
        Gets (or computes) a list of columns from the internal data.
        Returns a single numpy array of shape (len(data), len(columns)).
        Columns are returned in proper order.
        """
        if not columns:
            return np.empty((self._len, 0), dtype=np.float64)
        result_columns = [self.get(c) for c in columns]
        return np.vstack(result_columns).transpose().astype(np.float64)

    # ####################################### #
    # # Time Indicators                     # #
    # ####################################### #

    def time_ns(self) -> np.ndarray:
        date_gmt = self.get('date_gmt')
        assert date_gmt.dtype == 'datetime64[ns]', f"date_gmt must be of type datetime64[ns], but was {date_gmt.dtype}"
        return date_gmt.astype(np.int64)

    def lin_24h(self) -> np.ndarray:
        """Computes the time of the day as a float in [0, 1)"""
        date_gmt = self.get('date_gmt')
        assert date_gmt.dtype == 'datetime64[ns]', f"date_gmt must be of type datetime64[ns], but was {date_gmt.dtype}"
        ns_since_midnight = (date_gmt - date_gmt.astype('datetime64[D]')).astype(np.int64)
        return ns_since_midnight / (24 * 3600 * 1_000_000_000)

    def lin_7d(self) -> np.ndarray:
        """Computes the time of the week as a float in [0, 1)"""
        date_gmt = self.get('date_gmt')
        assert date_gmt.dtype == 'datetime64[ns]', f"date_gmt must be of type datetime64[ns], but was {date_gmt.dtype}"
        day_of_the_week = pd.Series(date_gmt).dt.dayofweek.to_numpy()
        precise_dotw = day_of_the_week + self.get("lin_24h")
        return precise_dotw / 7

    def sin_24h(self) -> np.ndarray:
        """Cyclical feature for time of day (sine component)."""
        return np.sin(self.get("lin_24h") * np.pi * 2)

    def sin_7d(self) -> np.ndarray:
        """Cyclical feature for day of week (sine component)."""
        return np.sin(self.get("lin_7d") * np.pi * 2)

    def cos_24h(self) -> np.ndarray:
        """Cyclical feature for time of day (cosine component)."""
        return np.cos(self.get("lin_24h") * np.pi * 2)

    def cos_7d(self) -> np.ndarray:
        """Cyclical feature for day of week (cosine component)."""
        return np.cos(self.get("lin_7d") * np.pi * 2)

    # ####################################### #
    # # Trend Indicators                    # #
    # ####################################### #

    def sma(self, window: int, column: str = 'close_bid') -> np.ndarray:
        """
        Calculate the Simple Moving Average (SMA) for a given column.
        """
        data = self.get(column)
        result = np.full(self._len, np.nan, dtype=np.float64)
        cumsum = np.cumsum(np.insert(data, 0, 0))
        result[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
        return result

    def ema(self, window: int, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Exponential Moving Average (EMA)."""
        return pd.Series(self.get(column)).ewm(span=window, adjust=False).mean().to_numpy(dtype=np.float64)

    def kama(self, window: int = 10, fast: int = 2, slow: int = 30, column: str = 'close_bid') -> np.ndarray:
        """Calculate Kaufman's Adaptive Moving Average (KAMA)."""
        series = pd.Series(self.get(column))
        change = abs(series - series.shift(window))
        volatility = (abs(series - series.shift(1))).rolling(window=window).sum()
        er = change / volatility
        fastest = 2 / (fast + 1)
        slowest = 2 / (slow + 1)
        sc = (er * (fastest - slowest) + slowest) ** 2

        kama_series = pd.Series(index=series.index, dtype=np.float64)
        if 0 < window <= len(series):
            kama_series.iloc[window - 1] = series.iloc[window - 1]
            for i in range(window, len(series)):
                # Ensure previous KAMA is not NaN before calculating next
                if pd.notna(kama_series.iloc[i - 1]) and pd.notna(sc.iloc[i]):
                    kama_series.iloc[i] = kama_series.iloc[i - 1] + sc.iloc[i] * (
                                series.iloc[i] - kama_series.iloc[i - 1])
                # If previous KAMA is NaN, but we have a price, start over
                elif pd.isna(kama_series.iloc[i - 1]) and pd.notna(series.iloc[i]):
                    kama_series.iloc[i] = series.iloc[i]

        return kama_series.to_numpy(dtype=np.float64)

    def vwap(self, window: int = 14) -> np.ndarray:
        """Calculate the Volume Weighted Average Price (VWAP) over a rolling window."""
        volume = pd.Series(self.get('volume'))
        close = pd.Series(self.get('close_bid'))
        price_volume = (close * volume).rolling(window=window).sum()
        total_volume = volume.rolling(window=window).sum()
        vwap_series = price_volume / total_volume.replace(0, np.nan)
        return vwap_series.to_numpy(dtype=np.float64)

    def adx(self, window: int = 14) -> np.ndarray:
        """Calculate the Average Directional Index (ADX)."""
        high = pd.Series(self.get('high_bid'))
        low = pd.Series(self.get('low_bid'))
        close = pd.Series(self.get('close_bid'))
        close_shifted = close.shift(1)

        tr1 = high - low
        tr2 = (high - close_shifted).abs()
        tr3 = (low - close_shifted).abs()
        tr = pd.Series(np.maximum.reduce([tr1, tr2, tr3]))

        atr = tr.ewm(alpha=1 / window, adjust=False).mean()

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)

        plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)

        dx_denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * ((plus_di - minus_di).abs() / dx_denom)
        adx = dx.ewm(alpha=1 / window, adjust=False).mean()

        return adx.to_numpy(dtype=np.float64)

    def parabolic_sar(self, acceleration_factor: float = 0.02, max_acceleration: float = 0.2) -> np.ndarray:
        """Calculate the Parabolic SAR (Stop and Reverse) indicator."""
        high = self.get('high_bid')
        low = self.get('low_bid')
        length = len(high)
        if length == 0:
            return np.array([], dtype=np.float64)
        sar_arr = np.full(length, np.nan, dtype=np.float64)
        uptrend = True  # 1 for uptrend, -1 for downtrend
        af = acceleration_factor
        # Initial values
        sar_arr[0] = low[0]
        ep = high[0]

        for i in range(1, length):
            prev_sar = sar_arr[i - 1]
            # Store previous extreme point to check for updates
            prev_ep = ep
            if uptrend:  # Uptrend
                sar_i = prev_sar + af * (ep - prev_sar)
                # Ensure SAR does not move into the previous period's low
                sar_i = min(sar_i, low[i - 1], low[i])
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_acceleration, af + acceleration_factor)
                if low[i] < sar_i: # Check for reversal
                    uptrend = False
                    sar_i = prev_ep  # Switch to previous EP on reversal
                    ep = low[i]
                    af = acceleration_factor
            else:  # Downtrend
                sar_i = prev_sar + af * (ep - prev_sar)
                # Ensure SAR does not move into the previous period's high
                sar_i = max(sar_i, high[i - 1], high[i])
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_acceleration, af + acceleration_factor)
                if high[i] > sar_i: # Check for reversal
                    uptrend = True
                    sar_i = prev_ep  # Switch to previous EP on reversal
                    ep = high[i]
                    af = acceleration_factor
            sar_arr[i] = sar_i

        return sar_arr

    # ####################################### #
    # # Momentum Indicators                 # #
    # ####################################### #

    def macd(self, short_window: int = 12, long_window: int = 26, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Moving Average Convergence Divergence (MACD) line."""
        short_ema = self.ema(short_window, column)
        long_ema = self.ema(long_window, column)
        return short_ema - long_ema

    def macd_signal(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9,
                    column: str = 'close_bid') -> np.ndarray:
        """Calculate the MACD Signal Line."""
        macd_line_name = f"macd({short_window}, {long_window}, '{column}')"
        macd_line = self.get(macd_line_name)
        return pd.Series(macd_line).ewm(span=signal_window, adjust=False).mean().to_numpy(dtype=np.float64)

    def macd_hist(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9,
                  column: str = 'close_bid') -> np.ndarray:
        """Calculate the MACD Histogram."""
        macd_line_name = f"macd({short_window}, {long_window}, '{column}')"
        signal_line_name = f"macd_signal({short_window}, {long_window}, {signal_window}, '{column}')"
        macd_line = self.get(macd_line_name)
        signal_line = self.get(signal_line_name)
        return macd_line - signal_line

    def rsi(self, window: int = 14, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Relative Strength Index (RSI)."""
        series = pd.Series(self.get(column))
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1 / window, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series[gain == loss] = 50.0
        return rsi_series.to_numpy(dtype=np.float64)

    def stoch_k(self, window: int = 14, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Stochastic Oscillator %K."""
        low_s = pd.Series(self.get('low_bid'))
        high_s = pd.Series(self.get('high_bid'))
        close_s = pd.Series(self.get(column))
        low_min = low_s.rolling(window=window).min()
        high_max = high_s.rolling(window=window).max()
        stoch_k_series = 100 * (close_s - low_min) / (high_max - low_min).replace(0, np.nan)
        return stoch_k_series.to_numpy(dtype=np.float64)

    def stoch_d(self, k_window: int = 14, d_window: int = 3, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Stochastic Oscillator %D (smoothed %K)."""
        stoch_k_name = f"stoch_k({k_window}, '{column}')"
        stoch_k_line = self.get(stoch_k_name)
        stoch_d_series = pd.Series(stoch_k_line).rolling(window=d_window).mean()
        return stoch_d_series.to_numpy(dtype=np.float64)

    def historic_pct_change(self, window: int = 14, column: str = 'close_bid') -> np.ndarray:
        """Calculate the historical percentage change over a given window."""
        series = pd.Series(self.get(column))
        return series.pct_change(periods=window).to_numpy(dtype=np.float64) * 100

    def cci(self, window: int = 20) -> np.ndarray:
        """Calculate the Commodity Channel Index (CCI)."""
        high = pd.Series(self.get('high_bid'))
        low = pd.Series(self.get('low_bid'))
        close = pd.Series(self.get('close_bid'))
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_dev = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci_series = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
        return cci_series.to_numpy(dtype=np.float64)

    def williams_r(self, window: int = 14) -> np.ndarray:
        """Calculate the Williams %R indicator."""
        high_s = pd.Series(self.get('high_bid'))
        low_s = pd.Series(self.get('low_bid'))
        close_s = pd.Series(self.get('close_bid'))
        high_max = high_s.rolling(window=window).max()
        low_min = low_s.rolling(window=window).min()
        williams_r_series = -100 * (high_max - close_s) / (high_max - low_min).replace(0, np.nan)
        return williams_r_series.to_numpy(dtype=np.float64)

    def mfi(self, window: int = 14) -> np.ndarray:
        """Calculate the Money Flow Index (MFI)."""
        high = pd.Series(self.get('high_bid'))
        low = pd.Series(self.get('low_bid'))
        close = pd.Series(self.get('close_bid'))
        volume = pd.Series(self.get('volume'))
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(window=window).sum()
        negative_flow = money_flow.where(typical_price.diff() < 0, 0).rolling(window=window).sum()

        money_ratio = positive_flow / negative_flow.replace(0, np.nan)
        mfi_series = 100 - (100 / (1 + money_ratio))
        return mfi_series.to_numpy(dtype=np.float64)

    def cmf(self, window: int = 20) -> np.ndarray:
        """Calculate the Chaikin Money Flow (CMF)."""
        high = self.get('high_bid')
        low = self.get('low_bid')
        close = self.get('close_bid')
        volume = self.get('volume')

        hl_range = high - low
        mfm = np.divide(((close - low) - (high - close)), hl_range, out=np.zeros_like(hl_range, dtype=float), where=hl_range != 0)
        mfv = mfm * volume

        # Use pandas for the rolling sum, which is convenient and handles NaNs well
        mfv_series = pd.Series(mfv)
        volume_series = pd.Series(volume)

        cmf_series = mfv_series.rolling(window=window).sum() / volume_series.rolling(window=window).sum()
        return cmf_series.to_numpy(dtype=np.float64)

    def obv(self, column: str = 'close_bid') -> np.ndarray:
        """Calculate the On-Balance Volume (OBV)."""
        close = self.get(column)
        volume = self.get('volume')
        obv_arr = np.zeros_like(close, dtype=np.float64)
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv_arr[i] = obv_arr[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv_arr[i] = obv_arr[i - 1] - volume[i]
            else:
                obv_arr[i] = obv_arr[i - 1]
        return obv_arr

    def ad_line(self) -> np.ndarray:
        """Calculate the Accumulation/Distribution Line (AD Line)."""
        high = self.get('high_bid')
        low = self.get('low_bid')
        close = self.get('close_bid')
        volume = self.get('volume')

        hl_range = high - low
        # If high equals low, the multiplier is 0.
        mfm = np.divide(((close - low) - (high - close)), hl_range, out=np.zeros_like(hl_range), where=hl_range != 0)
        ad = mfm * volume
        return np.cumsum(ad)

    # ####################################### #
    # # Volatility Indicators               # #
    # ####################################### #

    def bb_middle(self, window: int = 20, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Middle Bollinger Band (which is just an SMA)."""
        return self.sma(window, column)

    def bb_upper(self, window: int = 20, num_std_dev: float = 2.0, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Upper Bollinger Band."""
        middle_band = self.bb_middle(window, column)
        rolling_std = pd.Series(self.get(column)).rolling(window=window).std().to_numpy(dtype=np.float64)
        return middle_band + (num_std_dev * rolling_std)

    def bb_lower(self, window: int = 20, num_std_dev: float = 2.0, column: str = 'close_bid') -> np.ndarray:
        """Calculate the Lower Bollinger Band."""
        middle_band = self.bb_middle(window, column)
        rolling_std = pd.Series(self.get(column)).rolling(window=window).std().to_numpy(dtype=np.float64)
        return middle_band - (num_std_dev * rolling_std)

    def atr(self, window: int = 14) -> np.ndarray:
        """Calculate the Average True Range (ATR)."""
        high = pd.Series(self.get("high_bid"))
        low = pd.Series(self.get("low_bid"))
        close = pd.Series(self.get("close_bid"))

        # Calculate True Range using standard pandas operations for robustness with NaNs
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/N)
        atr_series = true_range.ewm(alpha=1 / window, adjust=False).mean()
        return atr_series.to_numpy(dtype=np.float64)

    def chaikin_volatility(self, ema_window: int = 10, roc_period: int = 10) -> np.ndarray:
        """Calculate the Chaikin Volatility indicator."""
        high_low_range = pd.Series(self.get('high_bid') - self.get('low_bid'))
        ema_high_low = high_low_range.ewm(span=ema_window, adjust=False).mean()
        roc_ema = ema_high_low.pct_change(periods=roc_period) * 100
        return roc_ema.to_numpy(dtype=np.float64)

    def ease_of_movement(self, window: int = 14) -> np.ndarray:
        """Calculate the Ease of Movement (EOM) indicator."""
        high = pd.Series(self.get('high_bid'))
        low = pd.Series(self.get('low_bid'))
        volume = pd.Series(self.get('volume')).replace(0, np.nan)

        distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 100_000_000) / (high - low).replace(0, np.nan)
        eom_1_period = distance_moved / box_ratio
        eom = eom_1_period.rolling(window=window).mean()
        return eom.to_numpy(dtype=np.float64)

    def spread_ratio(self):
        """Calculate the Spread Ratio indicator."""
        return (self.get("close_ask") - self.get("close_bid")) / self.get("close_bid")

    # ############################################### #
    # # NORMALIZATION, TRANSFORMATION, SCALING      # #
    # ############################################### #

    def as_robust_norm(self, column: str, window: int = 500) -> np.ndarray:
        """Normalizes a column using rolling robust normalization (median and IQR)."""
        log_column = pd.Series(np.log1p(self.get(column)))
        rolling_median = log_column.rolling(window=window, min_periods=1).median()
        q75 = log_column.rolling(window=window, min_periods=1).quantile(0.75)
        q25 = log_column.rolling(window=window, min_periods=1).quantile(0.25)
        iqr = q75 - q25
        return ((log_column - rolling_median) / (iqr + 1e-6)).to_numpy()

    def as_pct_change(self, column: str, periods: int = 1) -> np.ndarray:
        """Transforms a column to its percentage change."""
        return pd.Series(self.get(column)).pct_change(periods=periods).to_numpy()

    def as_ratio_of_other_column(self, column: str, other_column: str) -> np.ndarray:
        """Transforms a column to be a ratio of another column, centered around 0."""
        col_data = self.get(column)
        other_col_data = self.get(other_column)
        ratio = np.divide(col_data, other_col_data, out=np.full_like(col_data, np.nan), where=other_col_data != 0)
        return ratio - 1

    def as_z_score(self, column: str, window: int = 50) -> np.ndarray:
        """Normalizes a column as a rolling z-score."""
        series = pd.Series(self.get(column))
        if window == 0:  # Full dataset z-score
            mean = series.mean()
            std = series.std()
        else:  # Rolling z-score
            mean = series.rolling(window=window, min_periods=1).mean()
            std = series.rolling(window=window, min_periods=1).std()

        return ((series - mean) / (std + 1e-6)).to_numpy()

    def as_min_max_window(self, column: str, window: int = 50) -> np.ndarray:
        """Normalizes a column using rolling min-max scaling to [-1, 1]."""
        series = pd.Series(self.get(column))
        roll_min = series.rolling(window=window, min_periods=1).min()
        roll_max = series.rolling(window=window, min_periods=1).max()
        scaled = (series - roll_min) / ((roll_max - roll_min) + 1e-6)
        return (2 * scaled - 1).to_numpy()

    def as_min_max_fixed(self, column: str, min_val: float, max_val: float) -> np.ndarray:
        """Normalizes a column using fixed min-max scaling to [-1, 1]."""
        series = self.get(column)
        scaled = (series - min_val) / ((max_val - min_val) + 1e-6)
        return 2 * scaled - 1

    def as_below_above_column(self, column: str, other_column: str) -> np.ndarray:
        """Returns 1 if `column` > `other_column`, -1 if less, 0 if equal."""
        col_data = self.get(column)
        other_col_data = self.get(other_column)
        return np.sign(col_data - other_col_data)

    # ############################################### #
    # # Data Structure Manipulation Methods         # #
    # ############################################### #

    def apply_to_column(self, fn: Callable[[np.ndarray], np.ndarray], column: str,
                        new_column_name: str | None = None) -> 'FeatureEngineer':
        """
        Applies a function to a column's data.
        Note: This method modifies the internal data and is not called via `get`.

        Args:
            fn (Callable): A function that takes a numpy array and returns one.
            column (str): The name of the column to apply the function to.
            new_column_name (str, optional): If provided, stores the result in a new column.
                                             Otherwise, overwrites the existing column.

        Returns:
            FeatureEngineer: self for method chaining.
        """
        target_name = new_column_name if new_column_name else column
        self._data[target_name] = fn(self.get(column))
        return self

    def remove_columns(self, columns: List[str]) -> 'FeatureEngineer':
        """
        Removes specified columns from the internal data.
        Note: This method modifies the internal data and is not called via `get`.
        """
        for col in columns:
            self._data.pop(col, None)
        return self

    def remove_ohlcv(self) -> 'FeatureEngineer':
        """Removes standard OHLCV and date columns."""
        ohlcv_columns = ['open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid',
                         'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask',
                         'volume', 'date_gmt']
        self.remove_columns(ohlcv_columns)
        return self

    def history_lookback(self, lookback_window_size: int, columns: List[str] = None, not_columns: List[str] = None,
                         step: int = 1) -> 'FeatureEngineer':
        """
        Creates shifted (lagged) versions of columns.
        Note: This method modifies the internal data and is not called via `get`.
        """
        if lookback_window_size == 0:
            return self
        if columns is None:
            columns = list(self._data.keys())
        if not_columns is None:
            not_columns = []

        for col in columns:
            if col in not_columns:
                continue
            original_data = self.get(col)
            for i in range(1, lookback_window_size + 1, step):
                self._data[f'{col}_shift_{i}'] = shift(original_data, i)
        return self

    def copy_column(self, source_column: str, target_column: str) -> 'FeatureEngineer':
        """Copies a column."""
        if source_column not in self._data:
            raise KeyError(f"Source column '{source_column}' not found.")
        self._data[target_column] = self._data[source_column].copy()
        return self
