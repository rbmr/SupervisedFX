import numpy as np
import pandas as pd

def get_feature_engineer():
    from common.data.feature_engineer import FeatureEngineer, as_pct_change, ema, rsi, copy_column, as_ratio_of_other_column, as_min_max_fixed

    feature_engineer = FeatureEngineer()

    # Suggestion: Use fewer features and no history_lookback at first.
    # Let the agent's recurrent policy (if you use one) or the network itself find temporal patterns.

    # 1. Price Change (Momentum)
    def feature_1(df):
        copy_column(df, "close_bid", "close_pct_change_1")
        as_pct_change(df, "close_pct_change_1", periods=1)
        copy_column(df, "close_bid", "close_pct_change_5")
        as_pct_change(df, "close_pct_change_5", periods=5)  # Look at change over 5 periods

    feature_engineer.add(feature_1)

    # 2. Trend (EMA)
    def feature_2(df):
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")  # How far is the price from the EMA?
        ema(df, window=50)
        as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")

    feature_engineer.add(feature_2)

    # 3. Oscillator (RSI)
    def feature_3(df):
        rsi(df, window=14)
        as_min_max_fixed(df, "rsi_14", 0, 100)  # Normalize between 0 and 1

    feature_engineer.add(feature_3)

    return feature_engineer


def get_feature_engineer_chatgpt():
    from common.data.feature_engineer import (
        FeatureEngineer,
        as_pct_change,
        ema,
        rsi,
        copy_column,
        as_ratio_of_other_column,
        as_min_max_window,
        bollinger_bands,
        macd,
        adx,
        cci,
        stochastic_oscillator,
        historic_pct_change,
        as_z_score,
        as_min_max_fixed
    )

    feature_engineer = FeatureEngineer()

    #
    # Helper: ATR (Average True Range)
    #
    import pandas as pd
    import numpy as np

    def atr(df: pd.DataFrame, window: int = 14, column_high: str = "high_bid", column_low: str = "low_bid", column_close: str = "close_bid"):
        """
        Adds an 'atr_{window}' column to df containing the rolling ATR.
        ATR = rolling mean of True Range over `window` bars.
        True Range = max(high - low, |high - prev_close|, |low - prev_close|).
        """
        high = df[column_high]
        low = df[column_low]
        close_shifted = df[column_close].shift(1).fillna(method="bfill")

        # Compute True Range using np.maximum, then convert to Series
        true_range_array = np.maximum.reduce([
            high - low,
            (high - close_shifted).abs(),
            (low - close_shifted).abs()
        ])

        true_range = pd.Series(true_range_array, index=df.index)

        df[f"atr_{window}"] = true_range.rolling(window=window, min_periods=1).mean().fillna(0)

    #
    # 1) Price Momentum (Pct Change)
    #
    def feature_pct_changes(df):
        # 1-bar pct change
        copy_column(df, "close_bid", "close_pct_change_1")
        as_pct_change(df, "close_pct_change_1", periods=1)
        # 5-bar pct change
        copy_column(df, "close_bid", "close_pct_change_5")
        as_pct_change(df, "close_pct_change_5", periods=5)

        # 14-bar historic pct change (just as an alternative momentum)
        historic_pct_change(df, window=14)  # already multiplies by 100

        # Normalize those pct changes via rolling min-max over last 500 bars
        for col in ["close_pct_change_1", "close_pct_change_5", "historic_pct_change_14"]:
            as_min_max_window(df, column=col, window=500)

    feature_engineer.add(feature_pct_changes)

    #
    # 2) Trend / Moving Averages
    #
    def feature_trend(df):
        # EMA 20 & EMA 50
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")  # price / EMA20
        ema(df, window=50)
        as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")  # price / EMA50

        # Bollinger Band Width (normalized)
        bollinger_bands(df, window=20, num_std_dev=2.0)
        # width = (upper - lower) / middle
        df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20_close_bid"]
        df["bb_width_20"] = df["bb_width_20"].fillna(0)
        as_z_score(df, "bb_width_20", window=500)  # z-score normalize

        # MACD histogram (macd_hist) as-is, then normalize
        macd(df, short_window=12, long_window=26, signal_window=9)
        as_z_score(df, "macd_hist", window=500)

        # ADX (trend strength)
        adx(df, window=14)
        as_z_score(df, "adx", window=500)

    feature_engineer.add(feature_trend)

    #
    # 3) Momentum / Oscillators
    #
    def feature_oscillators(df):
        # RSI 14, normalize [0,1]
        rsi(df, window=14)
        df["rsi_14"] = df["rsi_14"].fillna(50)  # midpoint if NaN
        as_min_max_fixed(df, "rsi_14", 0, 100)

        # Stochastic %K and %D (14)
        stochastic_oscillator(df, window=14)
        df["stoch_k"] = df["stoch_k"].fillna(50)
        df["stoch_d"] = df["stoch_d"].fillna(50)
        as_min_max_fixed(df, "stoch_k", 0, 100)
        as_min_max_fixed(df, "stoch_d", 0, 100)

        # CCI 20, typically ranges roughly [−200, +200]—z-score normalize
        cci(df, window=20)
        as_z_score(df, "cci_20", window=500)

    feature_engineer.add(feature_oscillators)

    #
    # 4) Volatility (ATR / Price)
    #
    def feature_volatility(df):
        atr(df, window=14)  # adds atr_14
        # Normalize ATR by current close price → fractional volatility
        df["atr_ratio_14"] = df["atr_14"] / df["close_bid"]
        df["atr_ratio_14"] = df["atr_ratio_14"].fillna(0)
        as_z_score(df, "atr_ratio_14", window=500)

    feature_engineer.add(feature_volatility)

    #
    # 5) (Optional) Remove raw OHLCV once all engineered features exist
    #
    def drop_raw_ohlcv(df):
        # You can uncomment this if you want to drop all raw price/volume columns.
        raw_cols = [
            "open_bid", "high_bid", "low_bid", "close_bid", "volume",
            "open_ask", "high_ask", "low_ask", "close_ask",  # if present
            "volume_bid", "volume_ask"
        ]
        for col in raw_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

    feature_engineer.add(drop_raw_ohlcv)

    return feature_engineer
