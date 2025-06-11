import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from torch import nn

from RQ1.constants import TENSORBOARD_DIR
from common.constants import DP_CACHE_DIR
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import (FeatureEngineer, adx,
                                          as_min_max_fixed, as_min_max_window,
                                          as_pct_change,
                                          as_ratio_of_other_column, as_z_score,
                                          atr, bollinger_bands, cci,
                                          copy_column, ema,
                                          historic_pct_change, macd, rsi,
                                          stochastic_oscillator, complex_7d, complex_24h, history_lookback,
                                          remove_ohlcv)
from common.data.stepwise_feature_engineer import (StepwiseFeatureEngineer,
                                                   calculate_current_exposure, duration_of_current_trade)
from common.envs.dp import get_dp_table_from_env
from common.envs.forex_env import ForexEnv
from common.envs.rewards import DPRewardFunction


def get_train_model(env: ForexEnv, tb_log: Path | None = None):

    logging.info("Creating model...")

    # def linear_lr(start: float, end: float):
    #     diff = start - end
    #     def func(progress_remaining):
    #         return diff * progress_remaining + end
    #     return func

    # a2c_hyperparams = dict(
    #     policy="MlpPolicy",
    #     env=env,
    #     learning_rate=linear_lr(1e-3, 1e-5), # Rate of policy updates
    #     n_steps=128,
    #     gamma=1.0,
    #     gae_lambda=0.95,
    #     ent_coef=0.05,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     rms_prop_eps=1e-5,
    #     normalize_advantage=True,
    #     policy_kwargs=dict(
    #         activation_fn=nn.ReLU,
    #         net_arch=dict(pi=[64, 64], vf=[64, 64]),
    #     ),
    #     verbose=0,
    #     tensorboard_log=tb_log,
    #     device="cpu",
    # )

    sac_hyperparams = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3.e-4,
        buffer_size=200_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        gradient_steps=1,
        train_freq=32,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[64, 64], qf=[64, 64]),
        ),
        verbose=0,
        tensorboard_log=str(tb_log) if tb_log is not None else None,
        device="cpu",
    )
    if tb_log is not None:
        logging.info(f"Logging to tensorboard at {tb_log}")

    model = SAC(**sac_hyperparams)
    model.hyperparams = sac_hyperparams

    logging.info("Model created.")

    return model

def get_data():
    logging.info("Loading market data...")
    return ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M30,
        start_time=datetime(2020, 1, 1, 22, 0, 0, 0),
        end_time=datetime(2023, 12, 29, 21, 30, 0, 0),
    )

def get_train_env():

    forex_candle_data = get_data()

    logging.info("Setting up feature engineer...")
    market_feature_engineer, agent_feature_engineer = get_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=0.7,
        forex_candle_data=forex_candle_data,
        market_feature_engineer=market_feature_engineer,
        agent_feature_engineer=agent_feature_engineer,
        initial_capital=10_000.0,
        transaction_cost_pct=0.005,
        n_actions=0,
        custom_reward_function=None, # None for now, set later
        shuffled=True,
    )

    logging.info("Setting reward function...")

    # Get db table.
    table = get_dp_table_from_env(train_env, DP_CACHE_DIR, 7)
    train_env.custom_reward_function = DPRewardFunction(table)

    logging.info("Environments created.")

    return train_env

def get_eval_envs():

    forex_candle_data = get_data()

    logging.info("Setting up feature engineers...")
    market_feature_engineer, agent_feature_engineer = get_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=0.7,
        forex_candle_data=forex_candle_data,
        market_feature_engineer=market_feature_engineer,
        agent_feature_engineer=agent_feature_engineer,
        initial_capital=10_000.0,
        transaction_cost_pct=0.0,
        n_actions=0,
        custom_reward_function=None, # None for now, set later
        shuffled=False,
    )

    return {
        "train": train_env,
        "eval": eval_env,
    }

def get_feature_engineers():
    """
    Returns a FeatureEngineer that constructs exactly the four groups of features used
    in Zhang et al. (2019):

      1) Price Momentum (pct‐changes)
      2) Trend / Moving Averages (EMA ratios, BB width, MACD, ADX)
      3) Momentum/Oscillators (RSI, Stochastic‐K/D, CCI)
      4) Volatility (ATR‐ratio)

    And adds time-based features.
    """
    fe = FeatureEngineer()

    # 1) Price Momentum (Pct Change)
    def _feat_pct_change(df):
        # 1‐bar pct change
        copy_column(df, "close_bid", "close_pct_change_1")
        as_pct_change(df, "close_pct_change_1", periods=1)

        # 5‐bar pct change
        copy_column(df, "close_bid", "close_pct_change_5")
        as_pct_change(df, "close_pct_change_5", periods=5)

        # 14‐bar historic pct change (multiplied by 100 inside function)
        historic_pct_change(df, window=14)

        # Normalize via rolling min‐max over last 500 bars
        for col in ["close_pct_change_1", "close_pct_change_5", "historic_pct_change_14"]:
            as_min_max_window(df, column=col, window=500)

    fe.add(_feat_pct_change)

    # 2) Trend / Moving Averages
    def _feat_trend(df):
        # EMA 20 & EMA 50, then price/EMA ratios
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
        ema(df, window=50)
        as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")

        # Bollinger Bands width (normalized)
        bollinger_bands(df, window=20, num_std_dev=2.0)
        # width = (upper − lower) / middle
        df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20_close_bid"]
        df["bb_width_20"] = df["bb_width_20"].fillna(0.0)
        as_z_score(df, "bb_width_20", window=500)

        # MACD histogram, then z‐score normalize
        macd(df, short_window=12, long_window=26, signal_window=9)
        as_z_score(df, "macd_hist", window=500)

        # ADX (trend strength), z‐score normalize
        adx(df, window=14)
        as_z_score(df, "adx", window=500)

    fe.add(_feat_trend)

    # 3) Momentum / Oscillators
    def _feat_oscillators(df):
        # RSI(14) normalized to [0,1]
        rsi(df, window=14)
        df["rsi_14"] = df["rsi_14"].fillna(50.0)
        as_min_max_fixed(df, "rsi_14", 0, 100)

        # Stochastic %K and %D (14), normalize [0,100]
        stochastic_oscillator(df, window=14)
        df["stoch_k"] = df["stoch_k"].fillna(50.0)
        df["stoch_d"] = df["stoch_d"].fillna(50.0)
        as_min_max_fixed(df, "stoch_k", 0, 100)
        as_min_max_fixed(df, "stoch_d", 0, 100)

        # CCI(20), z‐score normalize
        cci(df, window=20)
        as_z_score(df, "cci_20", window=500)

    fe.add(_feat_oscillators)

    # 4) Volatility (ATR / Price)
    def _feat_volatility(df):
        atr(df, window=14)  # adds “atr_14”
        df["atr_ratio_14"] = df["atr_14"] / df["close_bid"]
        df["atr_ratio_14"] = df["atr_ratio_14"].fillna(0.0)
        as_z_score(df, "atr_ratio_14", window=500)

    fe.add(_feat_volatility)

    # 5) Time of day/week
    fe.add(complex_7d)
    fe.add(complex_24h)

    # 6) Add small lookback
    fe.add(remove_ohlcv)
    fe.add(lambda df: history_lookback(df, 4))

    # Setup stepwise feature engineer
    sfe = StepwiseFeatureEngineer()
    sfe.add(["current_exposure"], calculate_current_exposure)
    sfe.add(['current_trade_length'], duration_of_current_trade) # 1

    return fe, sfe

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

