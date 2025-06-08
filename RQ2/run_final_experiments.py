import logging
from typing import Callable, Tuple

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ2.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import FeatureEngineer, rsi, history_lookback, remove_ohlcv
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
from common.envs.forex_env import ForexEnv
from common.models.train_eval import run_experiment
from common.scripts import *


def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_EXPERIMENTS_START_DATE,
                                      end_time= RQ2_EXPERIMENTS_END_DATE,
                                    )
    
    experiment_funcs: list[Callable[[], Tuple[FeatureEngineer, StepwiseFeatureEngineer]]] = [
        lambda: experiment_1(),
    ]

    for func in experiment_funcs:
        logging.info(f"Running experiment: {func.__name__}")

        feature_engineer, stepwise_feature_engineer = func()

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
            verbose=0,
            seed=42,
        )

        logging.info("Running train test analyze...")
        run_experiment(
            train_env=train_env,
            eval_env=eval_env,
            model=model,
            base_folder_path=RQ2_DIR,
            experiment_group_name="hyperparameters",
            experiment_name= func.__name__,
            train_episodes=1,
            eval_episodes=1,
            checkpoints=True
        )

        logging.info(f"Experiment {func.__name__} completed.")


def base_experiment_func() -> Tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    """
    Base experiment function that returns the feature engineers.
    """
    feature_engineer = FeatureEngineer()
    feature_engineer.add(rsi)
    feature_engineer.add(remove_ohlcv)
    feature_engineer.add(lambda df: history_lookback(df, 20))

    stepwise_feature_engineer = StepwiseFeatureEngineer()
    stepwise_feature_engineer.add(['cash_percentage'], calculate_cash_percentage)

    return feature_engineer, stepwise_feature_engineer

def experiment_1() -> Tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    """
    Example experiment function that returns the feature engineers for experiment 1.
    """
    feature_engineer, stepwise_feature_engineer = base_experiment_func()
    
    # You can modify the feature engineers here if needed for this specific experiment
    # For example, adding more features or changing parameters

    return feature_engineer, stepwise_feature_engineer


if __name__ == '__main__':
    main()