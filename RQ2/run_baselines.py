import logging

import torch.optim as optim
from torch.nn import LeakyReLU

from common.scripts import set_seed
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from common.envs.forex_env import ForexEnv
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
from RQ2.constants import *

from common.data.data import ForexCandleData, Timeframe
from common.models.train_eval import run_experiment, evaluate_models, analyse_results
from common.models.dummy_models import DummyModel, long_model, short_model, custom_comparison_model
from common.constants import *
from common.scripts import *
from common.envs.rewards import risk_adjusted_return

from typing import Callable, Dict, Any, List, Tuple

def get_baselines() -> List[Tuple[str, DummyModel, FeatureEngineer, StepwiseFeatureEngineer]]:

    baselines = []

    # Baseline 1: Long Only Model
    baselines.append((
        "LONG_Only",
        long_model,
        FeatureEngineer(),
        StepwiseFeatureEngineer()
    ))

    # Baseline 2: Short Only Model
    baselines.append((
        "SHORT_Only",
        short_model,
        FeatureEngineer(),
        StepwiseFeatureEngineer()
    ))

    # Baseline 3: Custom Comparison Model :: KAMA comparison
    baselines.append((
        "KAMA_Comparison",
        custom_comparison_model,
        FeatureEngineer()
        .add(kama, window=10)
        .add(kama, window=25),
        StepwiseFeatureEngineer()
    ))

    return baselines


def main():
    
    set_seed(SEED)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_EXPERIMENTS_START_DATE,
                                      end_time= RQ2_EXPERIMENTS_END_DATE,
                                    )
    
    for (name, model, feature_engineer, stepwise_feature_engineer) in get_baselines():
        logging.info(f"Running baseline model: {model.__name__}")

        # Create environments
        logging.info("Creating environments...")
        train_env, eval_env = ForexEnv.create_train_eval_envs(
            split_ratio=RQ2_DATA_SPLIT_RATIO,
            forex_candle_data=forex_data,
            market_feature_engineer=feature_engineer,
            agent_feature_engineer=stepwise_feature_engineer,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            n_actions=1,
            allow_short=False,
            custom_reward_function=risk_adjusted_return)
        logging.info("Environments created.")

        # Run experiment
        run_experiment(
            train_env=train_env,
            eval_env=eval_env,
            model=model,
            base_folder_path=RQ2_DIR,
            experiment_group_name="baselines",
            experiment_name=name,
            train_episodes=0,
            eval_episodes=1,
            checkpoints=True,
            tensorboard_logging=True
        )

    
if __name__ == '__main__':
    main()