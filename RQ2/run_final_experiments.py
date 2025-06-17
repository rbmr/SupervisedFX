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
from common.models.train_eval import combine_finals
from common.scripts import *
from RQ2.parameters import *
from RQ2.hyperparameter_experiments import *

from common.models.experiment import ExperimentBlueprint, run_experiments

from RQ2.final_experiments import *

def get_data():
    return ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.H1,
        start_time=RQ2_HYPERPARAMETERS_START_DATE,
        end_time= RQ2_HYPERPARAMETERS_END_DATE_1H,
    )

def get_model(forex_env: ForexEnv, seed: int) -> DQN:
    dqn_kwargs = base_dqn_kwargs(forex_env, seed)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_increased_capacity_network(dqn_kwargs)
    model = DQN(**dqn_kwargs)
    return model

def get_envs(forex_data: ForexCandleData, feature_engineer: FeatureEngineer, stepwise_feature_engineer: StepwiseFeatureEngineer):
    train_env, validate_env, eval_env = ForexEnv.create_split_envs(
        split_pcts=RQ2_FINAL_EXPERIMENTS_DATA_SPLITS,
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=3,
        custom_reward_function=percentage_return
    )
    return [train_env, validate_env, eval_env]

def get_s1_experiment_functions() -> Dict[str, List[Callable]]:
    return {
        "Time": [
            S1_TM_NONE,
            S1_TM_L24,
            S1_TM_S24,
            S1_TM_SC24,
            S1_TM_L24L7,
            S1_TM_SC24SC7,
            S1_TM_COMBO,
            S1_TM_ALL
        ],
        "Trend": [
            S1_TR_NONE,
            S1_TR_NV,
            S1_TR_V,
            S1_TR_COMBO,
            S1_TR_ALL
        ],
        "Momentum": [
            S1_MO_NONE,
            S1_MO_NV,
            S1_MO_V,
            S1_MO_COMBO,
            S1_MO_ALL
        ],
        "Volatility": [
            S1_VO_NONE,
            S1_VO_NV,
            S1_VO_V,
            S1_VO_COMBO,
            S1_VO_ALL
        ],
        "Agent": [
            S1_AG_NONE,
            S1_AG_CE,
            S1_AG_DT,
            S1_AG_ALL
        ],
        "Combinatory": [
            S1_COMBO_COMBO,
            S1_COMBO_ALL
        ]
    }


def main(experiment_type: str = "Combinatory"):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    group_name = "FE_S1_" + experiment_type
    experiments: List[Callable[[], tuple[FeatureEngineer, StepwiseFeatureEngineer]]] = get_s1_experiment_functions()[experiment_type]

    blueprints = []
    for experiment in experiments:
        blueprint = ExperimentBlueprint(
            name=experiment.__name__,
            data_func=get_data,
            feature_engineers_func=experiment,
            envs_func=get_envs,
            model_func_with_seed=get_model,
            train_episodes=TRAIN_EPISODES,
        )
        blueprints.append(blueprint)
    
    run_experiments(base_folder_path=RQ2_DIR,
                    group_name=group_name,
                    experiment_blueprints=blueprints,
                    seeds=SEEDS,
                    num_workers=1)


if __name__ == '__main__':
    # let the user choose the experiment type with input
    options = get_s1_experiment_functions().keys()
    print("Available experiment types:")
    for i, option in enumerate(options):
        print(f"{i + 1}: {option}")
    choice = input("Choose an experiment type (number): ")
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(options):
            selected_experiment_type = list(options)[choice_index]
            main(selected_experiment_type)
        else:
            print("Invalid choice. Please select a valid option.")
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the options.")
