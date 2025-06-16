import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from stable_baselines3.common.base_class import BaseAlgorithm
from torch import nn

from RQ1.constants import TENSORBOARD_DIR, EXPERIMENT_NAME_FORMAT
from common.envs.callbacks import SaveCallback, ActionHistogramCallback
from common.envs.forex_env import ForexEnv
from common.models.train_eval import train_model, evaluate_models, analyse_results


def train_eval_analyze(experiment_dir: Path, model: BaseAlgorithm, train_env: ForexEnv, eval_env: ForexEnv):

    # Setup directories
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"

    # Train model
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

        callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
                    ActionHistogramCallback(train_env, log_freq=train_env.episode_len)]

        train_model(model, train_env, train_episodes=50, callback=callback)

    # Evaluate resulting models

    train_env.reset()
    eval_env.reset()

    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1, num_workers=3)

    # Analyze results

    analyse_results(results_dir)

# def _feat_pct_change(df):
#     # 1‐bar pct change
#     copy_column(df, "close_bid", "close_pct_change_1")
#     as_pct_change(df, "close_pct_change_1", periods=1)
#
#     # 5‐bar pct change
#     copy_column(df, "close_bid", "close_pct_change_5")
#     as_pct_change(df, "close_pct_change_5", periods=5)
#
#     # 14‐bar historic pct change (multiplied by 100 inside function)
#     historic_pct_change(df, window=14)
#
#     # Normalize via rolling min‐max over last 500 bars
#     for col in ["close_pct_change_1", "close_pct_change_5", "historic_pct_change_14"]:
#         as_min_max_window(df, column=col, window=500)
#
# def _feat_trend(df):
#     # EMA 20 & EMA 50, then price/EMA ratios
#     ema(df, window=20)
#     as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
#     ema(df, window=50)
#     as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")
#
#     # Bollinger Bands width (normalized)
#     bollinger_bands(df, window=20, num_std_dev=2.0)
#     # width = (upper − lower) / middle
#     df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20_close_bid"]
#     df["bb_width_20"] = df["bb_width_20"].fillna(0.0)
#     as_z_score(df, "bb_width_20", window=500)
#
#     # MACD histogram, then z‐score normalize
#     macd(df, short_window=12, long_window=26, signal_window=9)
#     as_z_score(df, "macd_hist", window=500)
#
#     # ADX (trend strength), z‐score normalize
#     adx(df, window=14)
#     as_z_score(df, "adx", window=500)
#
# def _feat_oscillators(df):
#     # RSI(14) normalized to [0,1]
#     rsi(df, window=14)
#     df["rsi_14"] = df["rsi_14"].fillna(50.0)
#     as_min_max_fixed(df, "rsi_14", 0, 100)
#
#     # Stochastic %K and %D (14), normalize [0,100]
#     stochastic_oscillator(df, window=14)
#     df["stoch_k"] = df["stoch_k"].fillna(50.0)
#     df["stoch_d"] = df["stoch_d"].fillna(50.0)
#     as_min_max_fixed(df, "stoch_k", 0, 100)
#     as_min_max_fixed(df, "stoch_d", 0, 100)
#
#     # CCI(20), z‐score normalize
#     cci(df, window=20)
#     as_z_score(df, "cci_20", window=500)
#
# def _feat_volatility(df):
#     atr(df, window=14)  # adds “atr_14”
#     df["atr_ratio_14"] = df["atr_14"] / df["close_bid"]
#     df["atr_ratio_14"] = df["atr_ratio_14"].fillna(0.0)
#     as_z_score(df, "atr_ratio_14", window=500)
#
#
# def get_feature_engineers(lookback: int = 3):
#     """
#     Returns a FeatureEngineer that constructs exactly the four groups of features used
#     in Zhang et al. (2019):
#
#       1) Price Momentum (pct‐changes)
#       2) Trend / Moving Averages (EMA ratios, BB width, MACD, ADX)
#       3) Momentum/Oscillators (RSI, Stochastic‐K/D, CCI)
#       4) Volatility (ATR‐ratio)
#
#     And adds time-based features.
#     """
#     fe = FeatureEngineer()
#
#     # 1) Price Momentum (Pct Change)
#
#     fe.add(_feat_pct_change)
#
#     # 2) Trend / Moving Averages
#
#     fe.add(_feat_trend)
#
#     # 3) Momentum / Oscillators
#
#     fe.add(_feat_oscillators)
#
#     # 4) Volatility (ATR / Price)
#
#     fe.add(_feat_volatility)
#
#     # 5) Add small lookback
#     ohlcv_columns = ['volume', 'date_gmt',
#                      'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid',
#                      'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask']
#     fe.add(partial(history_lookback, lookback_window_size=lookback, not_columns=ohlcv_columns))
#
#     # 6) Time of day/week
#     fe.add(complex_7d)
#     fe.add(complex_24h)
#
#     # Setup stepwise feature engineer
#     sfe = StepwiseFeatureEngineer()
#     sfe.add(["current_exposure"], calculate_current_exposure)
#
#     return fe, sfe

def cleanup_tensorboard():

    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    experiments = TENSORBOARD_DIR.iterdir()
    experiments = list(filter(lambda f: not f.name.startswith("_") and f.is_dir(), experiments))
    n_experiments = len(experiments)
    max_experiments = 8

    logging.info(f"Found {n_experiments} experiments in the tensorboard directory (max {max_experiments})")

    experiments.sort(key= lambda x: x.stat().st_mtime)
    if n_experiments > max_experiments:
        n_rem_experiments = n_experiments - max_experiments
        logging.info(f"Cleanup: deleting {n_rem_experiments} oldest experiments.")
        old_experiments = experiments[:n_rem_experiments]
        for experiment in old_experiments:
            shutil.rmtree(experiment)
        logging.info(f"Removed {n_rem_experiments} old experiments")

