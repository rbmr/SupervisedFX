import logging
from typing import Callable, List

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ2.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure, duration_of_current_trade 
from common.envs.forex_env import ForexEnv
from common.models.train_eval import run_experiment
from common.scripts import *


def get_feature_engineers() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    """
    Create and return the feature engineer and stepwise feature engineer objects.
    """
    # Create a feature engineer object
    feature_engineer = FeatureEngineer()
    feature_engineer.add(complex_7d)
    feature_engineer.add(complex_24h)
    
    # Add basic features
    # FEATURE 0 - CLOSE_PCT_CHANGE - 12 features
    def feature_0(df):
        copy_column(df, "close_bid", "close_pct_change")
        as_pct_change(df, "close_pct_change")
        history_lookback(df, 3, ["close_pct_change"])
    feature_engineer.add(feature_0)

    # -- TREND FEATURES --

    # FEATURE 1 - EMA_20 - 12 features 
    def feature_1(df):
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
        history_lookback(df, 3, ["ema_20_close_bid"])
    feature_engineer.add(feature_1)

    # FEATURE 2 - BOLLINGER_BANDS - 24 features
    def feature_2(df):
        bollinger_bands(df, window=20, num_std_dev=2)
        as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
        as_ratio_of_other_column(df, "bb_lower_20", "close_bid")
        history_lookback(df, 3, ["bb_upper_20"])
        history_lookback(df, 3, ["bb_lower_20"])
    feature_engineer.add(feature_2)

    # FEATURE 3 - MACD - 12 features
    def feature_3(df):
        macd(df, short_window=12, long_window=26, signal_window=9)
        remove_columns(df, ["macd_signal", "macd"])
        history_lookback(df, 3, ["macd_hist"])
    feature_engineer.add(feature_3)

    # -- TREND FEATURES END --

    # -- MOMENTUM FEATURES --
    # FEATURE 4 - RSI - 12 features
    def feature_4(df):
        rsi(df, window=14)
        as_min_max_fixed(df, "rsi_14", 0, 100)
        history_lookback(df, 3, ["rsi_14"])
    feature_engineer.add(feature_4)

    # FEATURE 5 - STOCHASTIC_OSCILLATOR - 12 features
    def feature_5(df):
        stochastic_oscillator(df, window=3)
        as_min_max_fixed(df, "stoch_k", 0, 100)
        as_min_max_fixed(df, "stoch_d", 0, 100)
        history_lookback(df, 3, ["stoch_k"])
        history_lookback(df, 3, ["stoch_d"])
    feature_engineer.add(feature_5)

    # FEATURE 6 - CCI - 12 features
    def feature_6(df):
        cci(df, window=20)
        as_min_max_fixed(df, "cci_20", -100, 100)
        history_lookback(df, 3, ["cci_20"])
    feature_engineer.add(feature_6)

    # -- MOMENTUM FEATURES END --=

    # -- VOLUME FEATURES --
    # FEATURE 7 - MFI - 12 features
    def feature_7(df):
        mfi(df, window=14)
        as_min_max_fixed(df, "mfi_14", 0, 100)
        history_lookback(df, 3, ["mfi_14"])
    feature_engineer.add(feature_7)

    def feature_8(df):
        # FEATURE 8 - OBV - 12 features
        obv(df)
        as_ratio_of_other_column(df, "obv", "volume")
        history_lookback(df, 3, ["obv"])
    feature_engineer.add(feature_8)
    
    def feature_9(df):
        # FEATURE 9 - CMF - 12 features
        cmf(df, window=20)
        as_min_max_fixed(df, "cmf_20", -1, 1)
        history_lookback(df, 3, ["cmf_20"])
    feature_engineer.add(feature_9)

    # -- VOLUME FEATURES END --

    # -- EXTRA TEST FEATURES --
    def feature_10(df):
        # FEATURE 10 - KAMA
        kama(df, window=10)
        as_ratio_of_other_column(df, "kama_10_close_bid", "close_bid")
        history_lookback(df, 3, ["kama_10_close_bid"])
    feature_engineer.add(feature_10)

    # Create a stepwise feature engineer object
    stepwise_feature_engineer = StepwiseFeatureEngineer()
    stepwise_feature_engineer.add(['cash_percentage'], get_current_exposure)
    stepwise_feature_engineer.add(['current_trade_length'], duration_of_current_trade)

    return feature_engineer, stepwise_feature_engineer



def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE,
                                    )
    
    # --- Feature Engineering ---
    
    feature_engineer, stepwise_feature_engineer = get_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=RQ2_DATA_SPLIT_RATIO,
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=1)
    logging.info("Environments created.")

    temp_env = DummyVecEnv([lambda: train_env])

    experiment_funcs: List[Callable[[DummyVecEnv], DQN]] = [
        exprmt_fast,
        exprmt_medium,
        exprmt_slow,
    ]

    for experiment_func in experiment_funcs:
        logging.info(f"Running experiment: {experiment_func.__name__}")
        dqn_model = experiment_func(temp_env)
        logging.info("Running train test analyze...")
        run_experiment(
            train_env=train_env,
            eval_env=eval_env,
            model=dqn_model,
            base_folder_path=RQ2_DIR,
            experiment_group_name="hyperparameters_1h",
            experiment_name=experiment_func.__name__,
            train_episodes=200,
            eval_episodes=1,
            checkpoints=True
        )
        logging.info(f"Experiment {experiment_func.__name__} completed.")
    

# EXPERIMENT FUNCTIONS

def base_experiment_func(temp_env: DummyVecEnv) -> DQN:
    policy_kwargs = dict(net_arch=[20, 10])
    return DQN(
        policy="MlpPolicy",
        env=temp_env,
        learning_rate=0.001,
        buffer_size=1000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=64,
        gradient_steps=64,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42
    )

def exprmt_slow(temp_env: DummyVecEnv) -> DQN:
    dqn = base_experiment_func(temp_env)
    dqn.learning_rate = 0.001
    dqn.buffer_size = 1000

    return dqn

def exprmt_medium(temp_env: DummyVecEnv) -> DQN:
    dqn = base_experiment_func(temp_env)
    dqn.learning_rate = 0.0005
    dqn.buffer_size = 5000

    return dqn

def exprmt_fast(temp_env: DummyVecEnv) -> DQN:
    dqn = base_experiment_func(temp_env)
    dqn.learning_rate = 0.0009
    dqn.buffer_size = 10_000

    return dqn



if __name__ == '__main__':
    main()