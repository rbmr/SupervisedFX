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
from common.models.train_eval import run_experiment, combine_finals
from common.scripts import *
from RQ2.parameters import *
from RQ2.hyperparameter_experiments import *


def main():
    
    set_seed(SEED)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.H1,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE_1H,
                                    )
    
    experiments: List[Callable[[DummyVecEnv], DQN]] = [
        HP_P3_hybrid_baseline,
        HP_P3_hybrid_reduced,
        HP_P3_hybrid_increased,
        HP_P3_hybrid_medium_symmetric,
        HP_P3_hybrid_symmetric,
        HP_P3_hybdrid_minimalist,
        HP_P3_hybrid_high
    ]

    feature_engineer, stepwise_feature_engineer = get_baseline_feature_engineers() # Get the feature engineers from the first experiment
    train_env, eval_env = ForexEnv.create_train_eval_envs(
            split_pcts=[RQ2_DATA_SPLIT_RATIO, 1-RQ2_DATA_SPLIT_RATIO],
            forex_candle_data=forex_data,
            market_feature_engineer=feature_engineer,
            agent_feature_engineer=stepwise_feature_engineer,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            n_actions=3,
            custom_reward_function=percentage_return
        )


    temp_dummy_env = DummyVecEnv([lambda: train_env])
    
    group_name = "[hyperparameters-P3]-1h_data"
    for experiment in experiments:
        logging.info(f"Running experiment: {experiment.__name__}")

        logging.info("Fetching Experiment Model...")
        model = experiment(temp_dummy_env)
        logging.info("Experiment Model fetched.")

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
            train_episodes=TRAIN_EPISODES,
            eval_episodes=1,
            checkpoints=True,
            tensorboard_logging=True,
            seed=SEED
        )

        logging.info(f"Experiment {experiment.__name__} completed.\n")

    combine_finals(
        experiment_group= RQ2_DIR / "experiments" / group_name,
    )


if __name__ == '__main__':
    main()