import logging

import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import LeakyReLU

from RQ2.constants import *
from RQ2.parameters import INITIAL_CAPITAL, TRANSACTION_COST_PCT
from common.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure
from common.envs.forex_env import ForexEnv
from common.envs.rewards import percentage_return
from common.models.train_eval import run_experiment
from common.scripts import *


def get_feature_engineer() -> FeatureEngineer:
    """
    Create a feature engineer object with the required features.
    """
    feature_engineer = FeatureEngineer()

    feature_engineer.add(complex_7d)
    feature_engineer.add(complex_24h)
    
    # Add basic features
    # FEATURE 0 - CLOSE_PCT_CHANGE - 12 features
    def feature_0(df):
        copy_column(df, "close_bid", "close_pct_change")
        as_pct_change(df, "close_pct_change")
        history_lookback(df, 11, ["close_pct_change"])
    feature_engineer.add(feature_0)

    # -- TREND FEATURES --

    # FEATURE 1 - EMA_20 - 12 features 
    def feature_1(df):
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
        history_lookback(df, 11, ["ema_20_close_bid"])
    feature_engineer.add(feature_1)

    # FEATURE 2 - BOLLINGER_BANDS - 24 features
    def feature_2(df):
        bollinger_bands(df, window=20, num_std_dev=2)
        as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
        as_ratio_of_other_column(df, "bb_lower_20", "close_bid")
        history_lookback(df, 11, ["bb_upper_20"])
        history_lookback(df, 11, ["bb_lower_20"])
    feature_engineer.add(feature_2)

    # FEATURE 3 - MACD - 12 features
    def feature_3(df):
        macd(df, short_window=12, long_window=26, signal_window=9)
        remove_columns(df, ["macd_signal", "macd"])
        history_lookback(df, 11, ["macd_hist"])
    feature_engineer.add(feature_3)

    # -- TREND FEATURES END --

    # -- MOMENTUM FEATURES --
    # FEATURE 4 - RSI - 12 features
    def feature_4(df):
        rsi(df, window=14)
        as_min_max_fixed(df, "rsi_14", 0, 100)
        history_lookback(df, 11, ["rsi_14"])
    feature_engineer.add(feature_4)

    # FEATURE 5 - STOCHASTIC_OSCILLATOR - 12 features
    def feature_5(df):
        stochastic_oscillator(df, window=3)
        as_min_max_fixed(df, "stoch_k", 0, 100)
        as_min_max_fixed(df, "stoch_d", 0, 100)
        history_lookback(df, 11, ["stoch_k"])
        history_lookback(df, 11, ["stoch_d"])
    feature_engineer.add(feature_5)

    # FEATURE 6 - CCI - 12 features
    def feature_6(df):
        cci(df, window=20)
        as_min_max_fixed(df, "cci_20", -100, 100)
        history_lookback(df, 11, ["cci_20"])
    feature_engineer.add(feature_6)

    # -- MOMENTUM FEATURES END --=

    # -- VOLUME FEATURES --
    # FEATURE 7 - MFI - 12 features
    def feature_7(df):
        mfi(df, window=14)
        as_min_max_fixed(df, "mfi_14", 0, 100)
        history_lookback(df, 11, ["mfi_14"])
    feature_engineer.add(feature_7)

    def feature_8(df):
        # FEATURE 8 - OBV - 12 features
        obv(df)
        as_ratio_of_other_column(df, "obv", "volume")
        history_lookback(df, 11, ["obv"])
    feature_engineer.add(feature_8)
    
    def feature_9(df):
        # FEATURE 9 - CMF - 12 features
        cmf(df, window=20)
        as_min_max_fixed(df, "cmf_20", -1, 1)
        history_lookback(df, 11, ["cmf_20"])
    feature_engineer.add(feature_9)

    # -- VOLUME FEATURES END --

    # -- EXTRA TEST FEATURES --
    def feature_10(df):
        # FEATURE 10 - KAMA
        kama(df, window=10)
        as_ratio_of_other_column(df, "kama_10_close_bid", "close_bid")
        history_lookback(df, 11, ["kama_10_close_bid"])
    feature_engineer.add(feature_10)

    return feature_engineer


def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # forex_data = ForexCandleData.load(source="dukascopy",
    #                                   instrument="EURUSD",
    #                                   granularity=Timeframe.M15,
    #                                   start_time=RQ2_HYPERPARAMETERS_START_DATE,
    #                                   end_time= RQ2_HYPERPARAMETERS_END_DATE,
    #                                 )
    
    

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.H1,
                                      start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
                                      end_time=datetime(2023, 6, 30, 20, 0, 0, 0),
                                    )
    
    # --- Feature Engineering ---
    # Create a feature engineer object
    feature_engineer = get_feature_engineer()

    # Add stepwise feature engineering
    stepwise_feature_engineer = StepwiseFeatureEngineer()
    stepwise_feature_engineer.add(['cash_percentage'], get_current_exposure)

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=RQ2_DATA_SPLIT_RATIO,
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=3,
        custom_reward_function=percentage_return)
    logging.info("Environments created.")

    policy_kwargs = dict(net_arch=[128,64,32], optimizer_class=optim.Adam, activation_fn=LeakyReLU)
    temp_env = DummyVecEnv([lambda: train_env])
    model = DQN(
        policy="MlpPolicy",
        env=temp_env,
        learning_rate=0.0003,
        buffer_size=10000,
        learning_starts=5000,
        batch_size=32,
        tau=0.05,
        gamma=0.95,
        train_freq=4,
        target_update_interval=16,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42,
        device=DEVICE
    )

    logging.info("Model created.")
    logging.info("Model architecture:" + str(model.policy))

    logging.info("Running train test analyze...")
    run_experiment(
        train_env=train_env,
        eval_env=eval_env,
        model=model,
        base_folder_path=RQ2_DIR,
        experiment_group_name="hyperparameters",
        experiment_name="overfit_for_life", # formerly 1h_test
        train_episodes=0,
        eval_episodes=1,
        checkpoints=True,
        tensorboard_logging=True
    )
    

if __name__ == '__main__':
    main()