import logging
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os

# Add project root (2 levels up from RQ5/secondary/general/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from common.scripts import set_seed, combine_df, split_df
from common.data import ForexData
from common.constants import FOREX_DIR
from common.feature.feature_engineer import FeatureEngineer, rsi, history_lookback, remove_ohlcv
from common.feature.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
from common.envs.forex_env import GeneralForexEnv
from common.trainertester import train_test_analyze

from pathlib import Path

def load_forex_data(symbol="EURUSD", timeframe="15M"):
    ask_path = FOREX_DIR / symbol / timeframe / "ASK" / "10.05.2022T00.00-10.05.2025T23.45.csv"
    bid_path = FOREX_DIR / symbol / timeframe / "BID" / "10.05.2022T00.00-10.05.2025T23.45.csv"
    ask_df = ForexData(ask_path).df
    bid_df = ForexData(bid_path).df
    return combine_df(bid_df, ask_df)

def create_feature_engineers():
    feature_engineer = FeatureEngineer()
    feature_engineer.add(rsi)
    feature_engineer.add(remove_ohlcv)
    feature_engineer.add(lambda df: history_lookback(df, 20))

    stepwise_engineer = StepwiseFeatureEngineer(columns=["cash_percentage"])
    stepwise_engineer.add(calculate_cash_percentage)

    return feature_engineer, stepwise_engineer

def create_envs(forex_df, feature_engineer, stepwise_engineer, capital, transaction_cost):
    train_df, eval_df = split_df(forex_df, 0.7)
    train_env = GeneralForexEnv(train_df, feature_engineer, stepwise_engineer, capital, transaction_cost)
    eval_env = GeneralForexEnv(eval_df, feature_engineer, stepwise_engineer, capital, transaction_cost)
    return train_env, eval_env

def run_experiment(strategy_name: str, exploration_config: dict):
    logging.info(f"Starting experiment for strategy: {strategy_name}")
    set_seed(42)

    INITIAL_CAPITAL = 10000.0
    TRANSACTION_COST_PCT = 0.001

    forex_df = load_forex_data()
    feature_engineer, stepwise_engineer = create_feature_engineers()
    train_env, eval_env = create_envs(forex_df, feature_engineer, stepwise_engineer, INITIAL_CAPITAL, TRANSACTION_COST_PCT)

    temp_env = DummyVecEnv([lambda: train_env])

    model = DQN(
        policy="MlpPolicy",
        env=temp_env,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        verbose=1,
        seed=42,
        **exploration_config
    )

    base_output_dir = Path("RQ5")  # or wherever you want to store results
    train_test_analyze(
        train_env=train_env,
        eval_env=eval_env,
        model=model,
        base_folder_path=base_output_dir,
        experiment_group_name=f"exploration_{strategy_name}",
        experiment_name=f"{strategy_name}_test",
        train_episodes=1,
        eval_episodes=1,
        checkpoints=False,
        deterministic=True
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: epsilon-greedy
    run_experiment("epsilon_greedy", {
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    })

    # Example: Boltzmann (requires custom policy or algorithm – to be added later)
    # run_experiment("boltzmann", {...})

    # Example: UCB (requires custom RL implementation)
    # run_experiment("ucb", {...})
