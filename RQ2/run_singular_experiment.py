import logging

import torch.optim as optim
from stable_baselines3 import DQN, SAC
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
from RQ2.hyperparameter_experiments import apply_cautious_parameters, apply_balanced_parameters


def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M30,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE_30M,
                                    )
    
    # --- Feature Engineering ---
    feature_engineer, stepwise_feature_engineer = get_baseline_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_split_envs(
        split_pcts=[RQ2_DATA_SPLIT_RATIO, 1-RQ2_DATA_SPLIT_RATIO],
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=3,
        custom_reward_function=percentage_return )
    logging.info("Environments created.")

    temp_env = DummyVecEnv([lambda: train_env])


    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    model = DQN(**dqn_kwargs)


    logging.info("Model created.")
    logging.info("Model architecture:" + str(model.policy))

    logging.info("Running train test analyze...")
    run_experiment(
        train_env=train_env,
        validate_env=eval_env,
        model=model,
        base_folder_path=RQ2_DIR,
        experiment_group_name="dqn",
        experiment_name="scaled_rewards",
        train_episodes=40,
        eval_episodes=1,
        checkpoints=True,
        tensorboard_logging=True,
        seed=SEED,
        num_workers=3,
    )
    

if __name__ == '__main__':
    main()