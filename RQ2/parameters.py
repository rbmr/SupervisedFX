import logging
from typing import Callable, List

import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import LeakyReLU

from RQ2.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure, duration_of_current_trade 
from common.envs.forex_env import ForexEnv
from common.models.train_eval import run_experiment
from common.envs.rewards import percentage_return
from common.scripts import *


SEED = 42
set_seed(SEED)

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 5/100_000

TRAIN_EPISODES = 100

def get_baseline_feature_engineers() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    """
    Create and return the feature engineer and stepwise feature engineer objects.
    """
    # Create a feature engineer object
    feature_engineer = FeatureEngineer()

    looky_backy = 4

    # ------------------------- #
    # ---- TIME Indicators ---- #
    # ------------------------- #
    feature_engineer.add(complex_24h) # 2


    # ----------------------------- #
    # ---- TA-Trend Indicators ---- #
    # ----------------------------- #
    def feat_sar(df):
        parabolic_sar(df)
        as_ratio_of_other_column(df, 'sar', 'close_bid')
        history_lookback(df, looky_backy, ["sar"])
    feature_engineer.add(feat_sar) # 1 * looky_backy
    
    def feat_vwap(df):
        vwap(df)
        as_ratio_of_other_column(df, 'vwap_14', 'close_bid')
        history_lookback(df, looky_backy, ["vwap_14"])
    feature_engineer.add(feat_vwap) # 1 * looky_backy

    # -------------------------------- #
    # ---- TA-Momentum Indicators ---- #
    # -------------------------------- #

    def feat_macd(df):
        macd(df, short_window=12, long_window=26, signal_window=9)
        remove_columns(df, ["macd_signal", "macd"])
        as_z_score(df, 'macd_hist', window=50)
        history_lookback(df, looky_backy, ["macd_hist"])
    feature_engineer.add(feat_macd) # 1 * looky_backy

    def feat_mfi(df):
        mfi(df)
        as_min_max_fixed(df, 'mfi_14', 0, 100)
        history_lookback(df, looky_backy, ["mfi_14"])
    feature_engineer.add(feat_mfi) # 1 * looky_backy

    # ---------------------------------- #
    # ---- TA-Volatility Indicators ---- #
    # ---------------------------------- #

    def feat_boll_bands(df):
        bollinger_bands(df, window=20, num_std_dev=2)
        as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
        as_ratio_of_other_column(df, "bb_lower_20", "close_bid")
        history_lookback(df, looky_backy, ["bb_upper_20"])
        history_lookback(df, looky_backy, ["bb_lower_20"])
    feature_engineer.add(feat_boll_bands) # 2 * looky_backy

    def feat_ch_vol(df):
        chaikin_volatility(df)
        history_lookback(df, looky_backy, ['chaikin_vol_10_10'])
    feature_engineer.add(feat_ch_vol) # 1 * looky_backy

    # -------------------------- #
    # ---- Agent Indicators ---- #
    # -------------------------- #

    # Create a stepwise feature engineer object
    stepwise_feature_engineer = StepwiseFeatureEngineer()
    stepwise_feature_engineer.add(['cash_percentage'], get_current_exposure) # 1
    stepwise_feature_engineer.add(['current_trade_length'], duration_of_current_trade) # 1

    return feature_engineer, stepwise_feature_engineer

def base_dqn_kwargs(temp_env: DummyVecEnv) -> Dict[str, Any]:
    policy_kwargs = dict(
        net_arch=[32, 16], 
        optimizer_class=optim.Adam, 
        activation_fn=LeakyReLU
    )

    kwargs = dict(
        policy="MlpPolicy",
        env=temp_env,
        learning_starts=1000,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED
    )
    return kwargs