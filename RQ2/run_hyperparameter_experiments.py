import logging
from typing import Callable, List

import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import LeakyReLU

from RQ2.constants import *
from RQ2.parameters import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure, duration_of_current_trade 
from common.envs.forex_env import ForexEnv
from common.models.train_eval import run_experiment
from common.envs.rewards import percentage_return
from common.scripts import *

def main():
    
    set_seed(SEED)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE_15M,
                                    )
    
    # --- Feature Engineering ---
    feature_engineer, stepwise_feature_engineer = get_baseline_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=RQ2_DATA_SPLIT_RATIO,
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=1,
        custom_reward_function=percentage_return
        )
    logging.info("Environments created.")

    temp_env = DummyVecEnv([lambda: train_env])

    experiment_funcs: List[Callable[[DummyVecEnv], DQN]] = [
        exprmt_aggresive,
        exprmt_balanced,
        exprmt_cautious,
        exprmt_patient
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
            experiment_group_name="[hyperparameters-P1]-15m_data",
            experiment_name=experiment_func.__name__,
            train_episodes=TRAIN_EPISODES,
            eval_episodes=1,
            checkpoints=True,
            tensorboard_logging=True,
        )
        logging.info(f"Experiment {experiment_func.__name__} completed.")
    

# EXPERIMENT FUNCTIONS
def exprmt_patient(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'patient' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)

    dqn_kwargs['learning_rate'] = 0.00001
    dqn_kwargs['buffer_size'] = 100_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.001
    dqn_kwargs['train_freq'] = 512
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 5_000
    dqn_kwargs['exploration_fraction'] = 0.5
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.01

    return DQN(**dqn_kwargs)

def exprmt_cautious(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'cautious' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)

    dqn_kwargs['learning_rate'] = 0.00005
    dqn_kwargs['buffer_size'] = 60_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.0025
    dqn_kwargs['train_freq'] = 64
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 2500
    dqn_kwargs['exploration_fraction'] = 0.4
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.02

    return DQN(**dqn_kwargs)

def exprmt_balanced(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'balanced' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)

    dqn_kwargs['learning_rate'] = 0.0001
    dqn_kwargs['buffer_size'] = 30_000
    dqn_kwargs['batch_size'] = 256
    dqn_kwargs['tau'] = 0.005
    dqn_kwargs['train_freq'] = 16
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 1_000
    dqn_kwargs['exploration_fraction'] = 0.33
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.05

    return DQN(**dqn_kwargs)

def exprmt_aggresive(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes an 'aggressive' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)

    dqn_kwargs['learning_rate'] = 0.001
    dqn_kwargs['buffer_size'] = 5_000
    dqn_kwargs['batch_size'] = 64
    dqn_kwargs['tau'] = 0.01
    dqn_kwargs['train_freq'] = 4
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 500
    dqn_kwargs['exploration_fraction'] = 0.25
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.1

    return DQN(**dqn_kwargs)



if __name__ == '__main__':
    main()