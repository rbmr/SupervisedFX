import logging
from datetime import datetime, timezone

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from RQ1.constants import RQ1_DIR
from RQ1.scripts import train_model, evaluate_models
from common.analysis import analyse_individual_run, analyse_finals
from common.constants import SEED
from common.data.data import Timeframe, ForexCandleData
from common.data.feature_engineer import (FeatureEngineer, history_lookback,
                                          remove_ohlcv, rsi)
from common.data.stepwise_feature_engineer import (
    StepwiseFeatureEngineer, calculate_cash_percentage)
from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.trainertester import run_model

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

# MODEL

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

# FOLDERS

EXPERIMENTS_DIR = RQ1_DIR / "experiments"
experiment_group = "testing"
experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_dir = EXPERIMENTS_DIR / experiment_group / experiment_name
logs_dir = experiment_dir / "logs"
models_dir = experiment_dir / "models"
results_dir = experiment_dir / "results"

logs_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# TRAINING

callback = [SaveOnEpisodeEndCallback(models_dir)]

train_model(model, train_env, train_episodes=5, callback=callback)

# EVALUATION

eval_envs = {
    "train": train_env,
    "eval": eval_env,
}

evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1)

# ANALYSIS

logging.info("Analyzing results...")

model_train_metrics = []
model_eval_metrics = []

for model_file in model_files:

    model_name = model_file.stem
    model_results_dir = results_dir / model_name
    train_results_dir = model_results_dir / "train"
    eval_results_dir = model_results_dir / "eval"
    train_results_file = train_results_dir / "data.csv"
    eval_results_file = eval_results_dir / "data.csv"

    logging.info(f"Analyzing results for model: {model_name}")

    # Load train and eval results
    results_df = pd.read_csv(train_results_file)
    metrics = analyse_individual_run(results_df, train_results_dir, name=model_name)
    model_train_metrics.append(metrics)

    results_df = pd.read_csv(eval_results_file)
    metrics = analyse_individual_run(results_df, eval_results_dir, name=model_name)
    model_eval_metrics.append(metrics)

analyse_finals(model_train_metrics, results_dir, name="train_results")
analyse_finals(model_eval_metrics, results_dir, name="eval_results")

logging.info("Analysis complete.")

logging.info("Done!")