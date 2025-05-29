import logging
import random

from common.scripts import set_seed
from common.scripts import combine_df
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from common.envs.forex_env import GeneralForexEnv
from common.feature.feature_engineer import FeatureEngineer, rsi, history_lookback, remove_ohlcv
from common.feature.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
from RQ2.constants import RQ2_DIR

from common.data import ForexCandleData, Timeframe
from common.trainertester import train_test_analyze
from common.constants import *
from common.scripts import *

if __name__ == '__main__':
    
    set_seed(42)
   

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration Parameters ---
    INITIAL_CAPITAL = 10000.0
    TRANSACTION_COST_PCT = 0.0

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
                                      end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
                                    )
    forex_data_df = forex_data.df
    train_df, eval_df = split_df(forex_data_df, 0.7)

    # --- Feature Engineering ---
    # Create a feature engineer object
    feature_engineer = FeatureEngineer()
    feature_engineer.add(rsi)
    feature_engineer.add(remove_ohlcv)
    feature_engineer.add(lambda df: history_lookback(df, 20))

    # Add stepwise feature engineering
    stepwise_feature_engineer = StepwiseFeatureEngineer(columns=['cash_percentage'])
    stepwise_feature_engineer.add(calculate_cash_percentage)

    logging.info("Creating environments...")
    train_env = GeneralForexEnv(
        market_data_df=train_df,
        data_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    eval_env = GeneralForexEnv(
        market_data_df=eval_df,
        data_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    logging.info("Environments created.")

    policy_kwargs = dict(net_arch=[20,10])
    temp_env = DummyVecEnv([lambda: train_env])
    model = DQN(
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
        verbose=1,
        seed=42,
    )

    logging.info("Running train test analyze...")
    train_test_analyze(
        train_env=train_env,
        eval_env=eval_env,
        model=model,
        base_folder_path=RQ2_DIR,
        experiment_group_name="testing123",
        experiment_name="capitalcom",
        train_episodes=1,
        eval_episodes=1,
        checkpoints=False,
        deterministic=True
    )
    
