import logging
from datetime import datetime

from RQ2.main import get_feature_engineer
from common.data.data import Timeframe, ForexCandleData
from common.data.stepwise_feature_engineer import calculate_cash_percentage, StepwiseFeatureEngineer
from common.envs.forex_env import ForexEnv

def get_environments():

    logging.info("Loading market data...")
    forex_candle_data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M15,
        start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
        end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
    )

    logging.info("Generating market features...")
    market_feature_engineer = get_feature_engineer()

    logging.info("Setting up stepwise feature engineer...")
    agent_feature_engineer = StepwiseFeatureEngineer()
    agent_feature_engineer.add(["cash_percentage"], calculate_cash_percentage)

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=0.7,
        forex_candle_data=forex_candle_data,
        market_feature_engineer=market_feature_engineer,
        agent_feature_engineer=agent_feature_engineer,
        initial_capital=10_000.0,
        transaction_cost_pct=0.0,
        n_actions=0
    )
    logging.info("Environments created.")

    return train_env, eval_env
