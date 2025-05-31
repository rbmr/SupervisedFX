import logging

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ1.constants import RQ1_DIR
from common.constants import DEVICE, FOREX_DIR, SEED
from common.data.data import ForexData
from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.data.feature_engineer import (FeatureEngineer, history_lookback,
                                          remove_ohlcv, rsi)
from common.data.stepwise_feature_engineer import (
    StepwiseFeatureEngineer, calculate_cash_percentage)
from common.scripts import combine_df, split_df
from common.trainertester import train_test_analyze

if __name__ != '__main__':
    raise ImportError("Do not import this module.")
# --- Configuration Parameters ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Finished imports")

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST_PCT = 0.0  # Example: 0.1% commission per trade

# Get ask and bid data, and combine

ask_path = FOREX_DIR / "EURUSD" / "15M" / "ASK" / "10.05.2022T00.00-10.05.2025T23.45.csv"
bid_path = FOREX_DIR / "EURUSD" / "15M" / "BID" / "10.05.2022T00.00-10.05.2025T23.45.csv"
ask_df = ForexData(ask_path).df
bid_df = ForexData(ask_path).df

combined_data = combine_df(bid_df, ask_df)

# Compute features
fe = FeatureEngineer()
fe.add(rsi)
fe.add(remove_ohlcv)
fe.add(lambda df: history_lookback(df, 20))

combined_features = fe.run(combined_data)

# Split data and features

train_data_df, eval_data_df = split_df(combined_data, 0.7)
train_feature_df, eval_feature_df = split_df(combined_features, 0.7)

# Add stepwise feature engineering
stepwise_feature_engineer = StepwiseFeatureEngineer()
stepwise_feature_engineer.add(["cash_percentage"], calculate_cash_percentage)

logging.info("Creating training environment...")
train_env = ForexEnv(
    market_data_df=train_data_df,
    market_feature_df=train_feature_df,
    agent_feature_engineer=stepwise_feature_engineer,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
    n_actions=0
)
logging.info("Creating evaluation environment...")
eval_env = ForexEnv(
    market_data_df=eval_data_df,
    market_feature_df=eval_feature_df,
    agent_feature_engineer=stepwise_feature_engineer,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
    n_actions=0
)
logging.info("Environments created.")


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

logging.info("Running train test analyze...")

# Set up parameters
base_folder_path = RQ1_DIR
experiment_group = "testing123"
experiment_name = "test2"
train_episodes = 1
eval_episodes = 1

# Set up folders
experiment_path = base_folder_path / "experiments" / experiment_group / experiment_name
results_path = experiment_path / "results"
logs_path = experiment_path / "logs"
models_path = experiment_path / "models"

models_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)
logs_path.mkdir(parents=True, exist_ok=True)

# model class
model_class = type(model)

# set env
train_dummy_env = DummyVecEnv([lambda: train_env])
model.set_env(train_dummy_env)

# total timesteps
total_timesteps = train_env.max_episode_timesteps() * train_episodes

# train the model (saving it every epoch)
logging.info(f"Training model for {train_episodes} epochs...")
callbacks = [
    SaveOnEpisodeEndCallback(save_path=str(models_path)),
]
model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1, progress_bar=True)
logging.info("Training complete.")

# save the final model
save_path = models_path / f"model_{total_timesteps}_steps.zip"
model.save(save_path)
logging.info(f"Model(s) saved to '{models_path}'.")

# TESTING THE MODELS
# 1. find all model zips in the models_path
model_files = list(models_path.glob("*.zip"))
logging.info(f"Found {len(model_files)} model files in '{models_path}'.")

# sort model files by modification time (oldest first)
model_files.sort(key=lambda x: x.stat().st_mtime)

# 2. Run Each Model on the train_env and eval_env
for model_file in model_files:
    logging.info(f"Loading model from {model_file}...")
    model = model_class.load(model_file, env=train_dummy_env)
    logging.info(f"Model loaded from {model_file}.")

    this_model_path = results_path / model_file.stem
    train_data_path = this_model_path / "train"
    eval_data_path = this_model_path / "eval"
    train_results_full_file = train_data_path / "data"
    eval_results_full_file = eval_data_path / "data"
    train_data_path.mkdir(parents=True, exist_ok=True)

    train_episode_length = train_env.max_episode_timesteps()
    eval_episode_length = eval_env.market_data_df)
    train_total_timesteps =

    run_model_on_vec_env(model, train_env, train_results_full_file,
                         total_steps=eval_episodes * train_episode_length, deterministic=deterministic,
                         progress_bar=True)

    run_model_on_vec_env(model, eval_env, eval_results_full_file, total_steps=eval_episodes * eval_episode_length,
                         deterministic=deterministic, progress_bar=True)

# ANALYSIS
logging.info("Analyzing results...")
model_train_metrics = []
model_eval_metrics = []
for model_file in model_files:
    model_name = model_file.stem
    this_model_path = results_path / model_name
    train_data_path = this_model_path / "train"
    eval_data_path = this_model_path / "eval"
    train_results_full_file = train_data_path / "data.csv"
    eval_results_full_file = eval_data_path / "data.csv"
    if not train_results_full_file.exists() or not eval_results_full_file.exists():
        logging.warning(f"Skipping analysis for {model_name} as one or both result files do not exist.")
        continue

    logging.info(f"Analyzing results for model: {model_name}")

    # Load train and eval results
    results_df = pd.read_csv(train_results_full_file)
    metrics = analyse_individual_run(results_df, train_data_path, name=model_name)
    model_train_metrics.append(metrics)

    results_df = pd.read_csv(eval_results_full_file)
    metrics = analyse_individual_run(results_df, eval_data_path, name=model_name)
    model_eval_metrics.append(metrics)

analyse_finals(model_train_metrics, results_path, name="train_results")
analyse_finals(model_eval_metrics, results_path, name="eval_results")

logging.info("Analysis complete.")

logging.info("Done!")

logging.info("Finished")


