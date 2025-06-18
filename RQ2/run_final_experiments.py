import logging
import operator

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
        start_time=RQ2_EXPERIMENTS_START_DATE,
        end_time=RQ2_EXPERIMENTS_END_DATE,
    )

def get_model(forex_env: ForexEnv, seed: int) -> DQN:
    dqn_kwargs = base_dqn_kwargs(forex_env, seed)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
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

def get_s2_experiment_functions() -> Dict[str, List[Callable]]:
    return {
        "SMALL": [
            S2_SMALL_0,
            S2_SMALL_1,
            S2_SMALL_2,
            S2_SMALL_4,
            S2_SMALL_8,
            S2_SMALL_16,
            S2_SMALL_32,
        ],
        "MEDIUM": [
            S2_MEDIUM_0,
            S2_MEDIUM_1,
            S2_MEDIUM_2,
            S2_MEDIUM_4,
            S2_MEDIUM_8,
            S2_MEDIUM_16,
            S2_MEDIUM_32,
        ],
        "LARGE": [
            S2_LARGE_0,
            S2_LARGE_1,
            S2_LARGE_2,
            S2_LARGE_4,
            S2_LARGE_8,
            S2_LARGE_16,
            S2_LARGE_32,
        ],
    }

def get_all_experiments_functions() -> Dict[str, List[Callable]]:
    return {
        "S1": get_s1_experiment_functions(),
        "S2": get_s2_experiment_functions()
    }

def main(experiment_group: str, experiment_type: str):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    group_name = f"[FE_{experiment_group}_{experiment_type}]"
    experiments: List[Callable[[], tuple[FeatureEngineer, StepwiseFeatureEngineer]]] = get_all_experiments_functions()[experiment_group][experiment_type]

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




def create_workload_partitions(num_parts: int) -> list[list[tuple[str, str]]]:
    """
    Analyzes all experiments and divides them into balanced partitions.
    
    Returns:
        A list of partitions, where each partition is a list of (experiment_group, experiment_type) tuples.
    """
    all_experiments = get_all_experiments_functions()
    
    # 1. Create a flat list of all work items with their size (number of experiments)
    work_items = []
    for group, types in all_experiments.items():
        for type_name, experiments in types.items():
            work_items.append({
                "group": group,
                "type": type_name,
                "workload": len(experiments)
            })

    # 2. Sort work items by workload, descending. This helps balance the partitions.
    work_items.sort(key=operator.itemgetter("workload"), reverse=True)
    
    # 3. Distribute work items into partitions using a greedy algorithm
    partitions = [[] for _ in range(num_parts)]
    partition_loads = [0] * num_parts
    
    for item in work_items:
        # Find the partition with the minimum current load
        min_load_idx = partition_loads.index(min(partition_loads))
        
        # Add the work item to that partition
        partitions[min_load_idx].append((item["group"], item["type"]))
        
        # Update the load of that partition
        partition_loads[min_load_idx] += item["workload"]
        
    return partitions, partition_loads


if __name__ == '__main__':
    NUM_WORKLOADS = 4
    partitions, loads = create_workload_partitions(NUM_WORKLOADS)

    print("The total workload has been divided into the following parts:")
    print("-" * 50)
    for i, (part, load) in enumerate(zip(partitions, loads)):
        print(f"Part {i + 1} (Total Experiments: {load}):")
        for group, type_name in part:
            print(f"  - Group: '{group}', Type: '{type_name}'")
        print()
    print("-" * 50)

    # 4. Let the user choose which partition to run
    choice = 0
    while choice not in range(1, NUM_WORKLOADS + 1):
        try:
            raw_choice = input(f"Choose a workload part to run (1-{NUM_WORKLOADS}): ")
            choice = int(raw_choice)
            if choice not in range(1, NUM_WORKLOADS + 1):
                print("Invalid input. Please enter a number shown above.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_partition_idx = choice - 1
    partition_to_run = partitions[selected_partition_idx]
    
    print(f"\nExecuting workload Part {choice}...")
    print("-" * 50)
    
    # 5. Run main() for each experiment type in the chosen partition
    for experiment_group, experiment_type in partition_to_run:
        main(experiment_group, experiment_type)
        
    print(f"\nAll experiments in Part {choice} have been completed.")
