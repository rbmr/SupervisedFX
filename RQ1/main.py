import logging
from datetime import datetime, timezone

import pandas as pd
import pytz
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ1.constants import RQ1_DIR
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

initial_capital = 10_000.0
transaction_cost_pct = 0.0

logging.info("Creating environments...")
train_env, eval_env = ForexEnv.create_train_eval_envs(
    split_ratio=0.7,
    forex_candle_data=forex_candle_data,
    market_feature_engineer=market_feature_engineer,
    agent_feature_engineer=agent_feature_engineer,
    initial_capital=initial_capital,
    transaction_cost_pct=transaction_cost_pct,
    n_actions=0
)
logging.info("Environments created.")

# MODEL

logging.info("Creating model...")
policy_kwargs = dict(net_arch=[128, 128])

DEVICE = "cpu"

model = A2C(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.001,
    gamma=0.99,
    n_steps=10,
    ent_coef=0.02,
    gae_lambda=0.95,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=0,
    seed=SEED,
    device=DEVICE
)
logging.info("Model created.")

# FOLDERS

EXPERIMENTS_DIR = RQ1_DIR / "experiments"
experiment_group = "testing"
experiment_name = datetime.now(pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
experiment_dir = EXPERIMENTS_DIR / experiment_group / experiment_name
logs_dir = experiment_dir / "logs"
models_dir = experiment_dir / "models"
results_dir = experiment_dir / "results"

logs_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# TRAINING

train_episodes = 1
train_dummy_env = DummyVecEnv([lambda: train_env])
model.set_env(train_dummy_env)
total_timesteps = train_env.total_steps * train_episodes

logging.info(f"Training model for {train_episodes} episodes...")

callback = [SaveOnEpisodeEndCallback(save_path=models_dir)]
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True)

logging.info("Training complete.")

# Saving the final model

model.save(models_dir / f"model_{total_timesteps}_steps.zip")
logging.info(f"Model(s) saved to '{models_dir}'.")

# EVALUATION

logging.info("Starting evaluation...")

eval_episodes = 1
model_class = type(model)

model_files = list(models_dir.glob("*.zip"))
model_files.sort(key=lambda x: x.stat().st_mtime)

logging.info(f"Found {len(model_files)} model files in '{models_dir}'.")

for model_file in model_files:

    logging.info(f"Loading model from {model_file}...")

    model = model_class.load(model_file, env=train_dummy_env, device=DEVICE)

    logging.info(f"Model loaded from {model_file}.")

    model_name = model_file.stem
    model_results_dir = results_dir / model_name
    train_results_dir = model_results_dir / "train"
    eval_results_dir = model_results_dir / "eval"
    train_results_file = train_results_dir / "data"
    eval_results_file = eval_results_dir / "data"

    train_episode_length = train_env.total_steps
    eval_episode_length = eval_env.total_steps

    logging.info("Running model on train environment.")

    run_model(model,
              train_env,
              train_results_file,
              total_steps=eval_episodes * train_episode_length,
              deterministic=True,
              progress_bar=True
              )

    logging.info("Running model on eval environment.")

    run_model(model,
              eval_env,
              eval_results_file,
              total_steps=eval_episodes * eval_episode_length,
              deterministic=True,
              progress_bar=True
              )

logging.info("Finished evaluation.")

# ANALYSIS

logging.info("Analyzing results...")

model_train_metrics = []
model_eval_metrics = []

for model_file in model_files:

    model_name = model_file.stem
    model_results_dir = results_dir / model_name
    train_results_dir = model_results_dir / "train"
    eval_results_dir = model_results_dir / "eval"
    train_results_file = train_results_dir / "data"
    eval_results_file = eval_results_dir / "data"

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