import logging
from datetime import datetime

from stable_baselines3 import A2C

from RQ1.constants import EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT
from common.constants import SEED
from common.data.data import Timeframe, ForexCandleData
from common.data.feature_engineer import (FeatureEngineer, history_lookback,
                                          remove_ohlcv, rsi)
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.models.train_eval import train_model, evaluate_models, analyse_evaluation_results
from common.models.utils import save_model_with_metadata
from common.scripts import most_recent_modified

if __name__ != '__main__':
    raise ImportError("Do not import this module.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Finished imports")

# ENVIRONMENT

logging.info("Loading market data...")
forex_candle_data = ForexCandleData.load(
    source="dukascopy",
    instrument="EURUSD",
    granularity=Timeframe.M15,
    start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
    end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
)

logging.info("Generating market features...")
market_feature_engineer = FeatureEngineer()
market_feature_engineer.add(rsi)
market_feature_engineer.add(remove_ohlcv)
market_feature_engineer.add(lambda df: history_lookback(df, 20))

logging.info("Setting up stepwise feature engineer...")
agent_feature_engineer = StepwiseFeatureEngineer()
agent_feature_engineer.add(["cash_percentage"], calculate_cash_percentage)

logging.info("Creating environments...")
train_env, eval_env = ForexEnv.create_train_eval_envs(
    split_ratio=0.7,
    forex_candle_data=forex_candle_data,
    market_feature_engineer=market_feature_engineer,
    agent_feature_engineer=agent_feature_engineer,
    initial_capital=10_000.0,
    transaction_cost_pct=0.0,
    n_actions=0
)
logging.info("Environments created.")

# TRAINING

logging.info("Creating model...")

model = A2C(
    policy="MlpPolicy",
    learning_rate=0.001,
    env=train_env,
    gamma=0.99,
    n_steps=10,
    ent_coef=0.02,
    gae_lambda=0.95,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=0,
    seed=SEED,
    device="cpu"
)

logging.info("Model created.")

experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
experiment_dir = EXPERIMENTS_DIR / experiment_name
models_dir = experiment_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

callback = [SaveOnEpisodeEndCallback(models_dir)]
train_model(model, train_env, train_episodes=1, callback=callback)
save_model_with_metadata(model, models_dir / "model_final.zip")

# EVALUATION

experiment_dir = most_recent_modified(EXPERIMENTS_DIR)
models_dir = experiment_dir / "models"
results_dir = experiment_dir / "results"
eval_envs = {
    "train": train_env,
    "eval": eval_env,
}
evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=2)

# ANALYSIS

analyse_evaluation_results(
    models_dir=models_dir,
    results_dir=results_dir,
    eval_envs_names=list(eval_envs.keys()),
)

logging.info("Done!")