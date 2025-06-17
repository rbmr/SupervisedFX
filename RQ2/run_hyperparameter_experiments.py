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
from common.models.train_eval import run_experiment_deprecated, combine_finals
from common.scripts import *
from RQ2.parameters import *
from RQ2.hyperparameter_experiments import *

from common.models.experiment import ExperimentBlueprint, run_experiments
from RQ2.feature_engineer_factory import RQ2FeatureEngineerFactory

def get_data():
    return ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.H1,
        start_time=RQ2_HYPERPARAMETERS_START_DATE,
        end_time= RQ2_HYPERPARAMETERS_END_DATE_1H,
    )

def get_envs(forex_data: ForexCandleData, feature_engineer: FeatureEngineer, stepwise_feature_engineer: StepwiseFeatureEngineer):
    train_env, eval_env = ForexEnv.create_split_envs(
        split_pcts=[RQ2_DATA_SPLIT_RATIO, 1-RQ2_DATA_SPLIT_RATIO, 0.0],
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=3,
        custom_reward_function=percentage_return
    )
    return [train_env, eval_env, None]

def main():
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    group_name = "[hyperparameters-P3]-1h_data"
    experiments: List[Callable[[DummyVecEnv], DQN]] = [
        HP_P1_aggressive
    ]

    blueprints: List[ExperimentBlueprint] = []
    for experiment in experiments:
        blueprints.append(ExperimentBlueprint(
            name=experiment.__name__,
            model_func_with_seed= lambda env, seed: HP_P1_aggressive(env, seed),
            data_func=get_data,
            feature_engineers_func= lambda: RQ2FeatureEngineerFactory.create_core_factory().give_me_them_engineers,
            envs_func=get_envs,
            train_episodes=TRAIN_EPISODES,
        ))

    run_experiments(
        group_name=group_name,
        experiment_blueprints=blueprints,
        base_folder_path=RQ2_DIR,
        num_workers=1)


if __name__ == '__main__':
    main()