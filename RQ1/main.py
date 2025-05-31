import logging
from datetime import datetime

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ1.constants import RQ1_DIR
from common.analysis import analyse_individual_run, analyse_finals
from common.constants import DEVICE, SEED
from common.data.data import Timeframe, ForexCandleData
from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.data.feature_engineer import (FeatureEngineer, history_lookback,
                                          remove_ohlcv, rsi)
from common.data.stepwise_feature_engineer import (
    StepwiseFeatureEngineer, calculate_cash_percentage)
from common.scripts import split_df
from common.trainertester import run_model

if __name__ != '__main__':
    raise ImportError("Do not import this module.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Finished imports")

# Load, preprocess and split market data

logging.info("Loading market data...")
market_data = ForexCandleData.load(
    source="dukascopy",
    instrument="EURUSD",
    granularity=Timeframe.M15,
    start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
    end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
).df

logging.info("Generating market features...")
fe = FeatureEngineer()
fe.add(rsi)
fe.add(remove_ohlcv)
fe.add(lambda df: history_lookback(df, 20))

market_features = fe.run(market_data)

logging.info("Splitting data...")
train_data_df, eval_data_df = split_df(market_data, 0.7)
train_feature_df, eval_feature_df = split_df(market_features, 0.7)

# Setup stepwise feature engineering

logging.info("Setting up stepwise feature engineer...")
stepwise_feature_engineer = StepwiseFeatureEngineer()
stepwise_feature_engineer.add(["cash_percentage"], calculate_cash_percentage)

# Creating environments

initial_capital = 10_000.0
transaction_cost_pct = 0.0  # Example: 0.1% commission per trade

logging.info("Creating training environment...")
train_env = ForexEnv(
    market_data_df=train_data_df,
    market_feature_df=train_feature_df,
    agent_feature_engineer=stepwise_feature_engineer,
    initial_capital=initial_capital,
    transaction_cost_pct=transaction_cost_pct,
    n_actions=0
)

logging.info("Creating evaluation environment...")
eval_env = ForexEnv(
    market_data_df=eval_data_df,
    market_feature_df=eval_feature_df,
    agent_feature_engineer=stepwise_feature_engineer,
    initial_capital=initial_capital,
    transaction_cost_pct=transaction_cost_pct,
    n_actions=0
)
logging.info("Environments created.")

# Creating model

logging.info("Creating model...")
policy_kwargs = dict(net_arch=[128, 128])

model = A2C(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.001,
    gamma=0.99,
    n_steps=10,                 # Slightly higher n_steps for more stable estimates
    ent_coef=0.02,              # Increase entropy coefficient to encourage more exploration
    gae_lambda=0.95,            # Lower lambda for more bias, but faster learning
    vf_coef=0.5,                # Value function loss coefficient (default)
    max_grad_norm=0.5,          # Gradient clipping (default)
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=DEVICE
)
logging.info("Model created.")

# Set up folders

EXPERIMENT_DIR = RQ1_DIR / "experiments" / "testing123" / "test2"
LOGS_DIR = EXPERIMENT_DIR / "logs"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TRAINING

train_episodes = 1
train_dummy_env = DummyVecEnv([lambda: train_env])
model.set_env(train_dummy_env)
total_timesteps = train_env.max_episode_timesteps() * train_episodes

logging.info(f"Training model for {train_episodes} epochs...")

callback = [SaveOnEpisodeEndCallback(save_path=MODELS_DIR)]
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True)

logging.info("Training complete.")

# Saving the final model

model.save(MODELS_DIR / f"model_{total_timesteps}_steps.zip")
logging.info(f"Model(s) saved to '{MODELS_DIR}'.")

# EVALUATION

logging.info("Starting evaluation...")

eval_episodes = 1
model_class = type(model)

model_files = list(MODELS_DIR.glob("*.zip"))
model_files.sort(key=lambda x: x.stat().st_mtime)

logging.info(f"Found {len(model_files)} model files in '{MODELS_DIR}'.")

for model_file in model_files:

    logging.info(f"Loading model from {model_file}...")

    model = model_class.load(model_file, env=train_dummy_env)

    logging.info(f"Model loaded from {model_file}.")

    model_name = model_file.stem
    model_results_dir = RESULTS_DIR / model_name
    train_results_dir = model_results_dir / "train"
    eval_results_dir = model_results_dir / "eval"
    train_results_file = train_results_dir / "data"
    eval_results_file = eval_results_dir / "data"

    train_episode_length = train_env.max_episode_timesteps()
    eval_episode_length = eval_env.max_episode_timesteps()

    run_model(model,
              train_env,
              train_results_file,
              total_steps=eval_episodes * train_episode_length,
              deterministic=True,
              progress_bar=True
              )

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
    model_results_dir = RESULTS_DIR / model_name
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

analyse_finals(model_train_metrics, RESULTS_DIR, name="train_results")
analyse_finals(model_eval_metrics, RESULTS_DIR, name="eval_results")

logging.info("Analysis complete.")

logging.info("Done!")