import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_environments():
    from common.data.data import ForexCandleData, Timeframe
    from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_cash_percentage
    from common.envs.forex_env import ForexEnv
    from RQ2.main import get_feature_engineer

    logging.info("Loading market data...")
    forex_candle_data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M15,
        start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
        end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
    )

    logging.info("Setting up feature engineer...")
    market_feature_engineer = get_feature_engineer()

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

    return train_env, eval_env

def train():
    from stable_baselines3 import A2C
    from RQ1.constants import EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT
    from common.constants import SEED
    from common.envs.callbacks import SaveOnEpisodeEndCallback, ActionHistogramCallback, CoolStatsCallback
    from common.models.train_eval import train_model
    from common.models.utils import save_model_with_metadata

    train_env, _ = get_environments()

    logging.info("Creating model...")

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=5_000,
        gamma=0.995,
        gae_lambda=0.9,
        ent_coef=0.005,
        vf_coef=0.4,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=SEED,
        verbose=1,
        device="cpu"
    )

    logging.info("Model created.")

    experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveOnEpisodeEndCallback(models_dir),
                ActionHistogramCallback(train_env, log_freq=model.n_steps),
                CoolStatsCallback(train_env, log_freq=model.n_steps)]
    train_model(model, train_env, train_episodes=50, callback=callback)
    save_model_with_metadata(model, models_dir / "model_final.zip")

def evaluate(experiments_dir, limit = 10):
    from common.models.train_eval import evaluate_models
    from common.scripts import picker, has_nonempty_subdir, n_children

    experiment_dirs: list[Path] = list(experiments_dir.iterdir())
    experiment_dirs = list(f for f in experiment_dirs if has_nonempty_subdir(f, "models"))
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    experiment_dirs = experiment_dirs[:limit] if limit is not None else experiment_dirs
    named_dirs = list((f"{f.name} ({n_children(f/"models")}", f) for f in experiment_dirs)

    experiment_dir = picker(named_dirs)
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = get_environments()
    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1)

def analyze(experiments_dir, limit = 10):
    from common.scripts import picker, has_nonempty_subdir, n_children
    from common.models.train_eval import analyse_results

    experiment_dirs: list[Path] = list(experiments_dir.iterdir())
    experiment_dirs = list(f for f in experiment_dirs if has_nonempty_subdir(f, "results"))
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    experiment_dirs = experiment_dirs[:limit] if limit is not None else experiment_dirs
    named_dirs = list((f"{f.name} ({n_children(f/"results")})", f) for f in experiment_dirs)

    experiment_dir = picker(named_dirs)
    results_dir = experiment_dir / "results"

    analyse_results(results_dir)

if __name__ == "__main__":
    from RQ1.constants import EXPERIMENTS_DIR
    from common.scripts import picker

    options = [
        ("train", train),
        ("eval", lambda: evaluate(EXPERIMENTS_DIR, 10)),
        ("analyze", lambda: analyze(EXPERIMENTS_DIR, 10)),
    ]
    picker(options, default=None)()

