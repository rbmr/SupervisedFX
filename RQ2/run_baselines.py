import logging
from typing import List, Tuple

from RQ2.constants import *
from common.constants import *
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer
from common.envs.forex_env import ForexEnv
from common.envs.rewards import risk_adjusted_return
from common.models.dummy_models import DummyModel, long_model, short_model, custom_comparison_model, dp_perfect_model
from common.models.train_eval import run_experiment, evaluate_dummy, analyse_results
from common.scripts import *


def get_baselines() -> List[Tuple[str, Callable[[ForexEnv], DummyModel], FeatureEngineer, StepwiseFeatureEngineer]]:

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
        .add(lambda df: kama(df, window=10))
        .add(lambda df: kama(df, window=25)),
        StepwiseFeatureEngineer()
    ))

    

    baselines.append((
        "DP_Perfect_Policy",
        dp_perfect_model,
        FeatureEngineer(),
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
            n_actions=0,
            custom_reward_function=risk_adjusted_return)
        logging.info("Environments created.")

        eval_envs = {
            "train": train_env,
            "eval": eval_env
        }

        results_dir = RQ2_DIR / "experiments" / "baselines"
        results_dir.mkdir(parents=True, exist_ok=True)

        for env_name, env in eval_envs.items():
            logging.info(f"Running evaluation on {env_name} environment...")
            evaluate_dummy(
                dummy_model=model(env),
                name=name,
                results_dir=results_dir,
                eval_env=env,
                eval_env_name=env_name
            )
            logging.info(f"Evaluation on {env_name} environment completed.")
    
    analyse_results(
        results_dir=results_dir,
        model_name_suffix= "[Baseline Models]",
    )

        


    
if __name__ == '__main__':
    main()