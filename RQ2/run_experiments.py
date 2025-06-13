import logging

import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import LeakyReLU

from RQ2.constants import *
from common.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure
from common.envs.forex_env import ForexEnv
from common.envs.rewards import percentage_return
from common.models.train_eval import run_experiment
from common.scripts import *
from RQ2.parameters import *
from RQ2.experiments import *


def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE_15M,
                                    )
    
    experiments = [
        HP_P2_cautious_baseline,
        HP_P2_cautious_minimalist,
        HP_P2_cautious_reduced,
        HP_P2_cautious_increased,
        HP_P2_cautious_medium_symmetric,
        HP_P2_cautious_symmetric,
        HP_P2_cautious_high,

        HP_P2_balanced_baseline,
        HP_P2_balanced_minimalist,
        HP_P2_balanced_reduced,
        HP_P2_balanced_increased,
        HP_P2_balanced_medium_symmetric,
        HP_P2_balanced_symmetric,
        HP_P2_balanced_high,
    ]

    for experiment in experiments:
        logging.info(f"Running experiment: {experiment.__name__}")

        temp_env = DummyVecEnv([])

        logging.info("Fetching Experiment Model and Feature Engineers...")
        model, feature_engineer, stepwise_feature_engineer = experiment(temp_env)
        logging.info("Experiment Model and Feature Engineers fetched.")

        logging.info("Creating environments...")
        train_env, eval_env = ForexEnv.create_train_eval_envs(
            split_ratio=RQ2_DATA_SPLIT_RATIO,
            forex_candle_data=forex_data,
            market_feature_engineer=feature_engineer,
            agent_feature_engineer=stepwise_feature_engineer,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            n_actions=3,
            custom_reward_function=percentage_return
        )
        logging.info("Environments created.")

        group_name = "[hyperparameters-P2]-15m_data"

        logging.info("Model created.")
        logging.info("Model architecture:" + str(model.policy))
        logging.info("Running Experiment parts...")
        run_experiment(
            train_env=train_env,
            eval_env=eval_env,
            model=model,
            base_folder_path=RQ2_DIR,
            experiment_group_name=group_name,
            experiment_name=experiment.__name__,
            train_episodes=125,
            eval_episodes=1,
            checkpoints=True,
            tensorboard_logging=True
        )
        logging.info(f"Experiment {experiment.__name__} completed.\n")


if __name__ == '__main__':
    main()