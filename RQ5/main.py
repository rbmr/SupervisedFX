import logging
from datetime import datetime
import sys
import os

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.data.data import ForexCandleData, Timeframe
from common.envs.forex_env import ForexEnv
from common.envs.rewards import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_current_exposure
from common.data.feature_engineer import FeatureEngineer, as_pct_change, ema, rsi, copy_column, as_ratio_of_other_column, as_min_max_fixed
from common.constants import SEED
from RQ5.constants import EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT
from common.envs.callbacks import *
from common.models.train_eval import train_model
from common.models.utils import save_model_with_metadata
from common.scripts import picker, has_nonempty_subdir, n_children

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_feature_engineer():
    fe = FeatureEngineer()

    def price_change(df):
        copy_column(df, "close_bid", "close_pct_change_1")
        as_pct_change(df, "close_pct_change_1", periods=1)
        copy_column(df, "close_bid", "close_pct_change_5")
        as_pct_change(df, "close_pct_change_5", periods=5)

    def trend(df):
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
        ema(df, window=50)
        as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")

    def oscillator(df):
        rsi(df, window=14)
        as_min_max_fixed(df, "rsi_14", 0, 100)

    fe.add(price_change)
    fe.add(trend)
    fe.add(oscillator)
    return fe

def get_environments():
    logging.info("Loading market data...")
    data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M15,
        start_time=datetime(2022, 1, 2, 22),
        end_time=datetime(2025, 5, 16, 20, 45),
    )

    logging.info("Creating feature pipelines...")
    market_fe = get_feature_engineer()
    agent_fe = StepwiseFeatureEngineer()
    agent_fe.add(["current_exposure"], calculate_current_exposure)

    logging.info("Building environments...")
    return ForexEnv.create_train_eval_envs(
        split_ratio=0.8,
        forex_candle_data=data,
        market_feature_engineer=market_fe,
        agent_feature_engineer=agent_fe,
        initial_capital=10000.0,
        transaction_cost_pct=0.0,
        n_actions=1,
        custom_reward_function=log_equity_change
    )

def train(exploration_strategy: str):
    train_env, _ = get_environments()
    logging.info("Instantiating DQN model...")

    dqn_args = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=SEED,
        verbose=1,
        device="cpu"
    )

    if exploration_strategy == "epsilon_greedy":
        from stable_baselines3 import DQN
        dqn_args.update({
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.8
        })
        model = DQN(**dqn_args)

    elif exploration_strategy == "boltzmann":
        from RQ5.boltzmann_dqn import BoltzmannDQN 
        model = BoltzmannDQN(**dqn_args, temperature=1.0)
    elif exploration_strategy == "max_boltzmann":
        from RQ5.boltzmann_dqn import MaxBoltzmannDQN
        model = MaxBoltzmannDQN(**dqn_args, epsilon=0.1, temperature=1.0)

    else:
        raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")

    exp_name = f"{datetime.now().strftime(EXPERIMENT_NAME_FORMAT)}_{exploration_strategy}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
            ActionHistogramCallback(train_env, log_freq=train_env.episode_len),
            SneakyLogger(verbose=1)]

    logging.info("Training...")
    train_model(model, train_env, train_episodes=20, callback=callback)
    save_model_with_metadata(model, models_dir / "model_final.zip")

def evaluate(experiments_dir, limit=10):
    from common.models.train_eval import evaluate_models

    dirs = sorted([
        f for f in experiments_dir.iterdir() if has_nonempty_subdir(f, "models")
    ], key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    exp_dir = picker([(f"{f.name} ({n_children(f / 'models')})", f) for f in dirs])
    models_dir = exp_dir / "models"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = get_environments()
    evaluate_models(models_dir, results_dir, {"train": train_env, "eval": eval_env}, eval_episodes=1)

def analyze(experiments_dir, limit=10):
    from common.models.train_eval import analyse_results

    dirs = sorted([
        f for f in experiments_dir.iterdir() if has_nonempty_subdir(f, "results")
    ], key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    exp_dir = picker([(f"{f.name} ({n_children(f / 'results')})", f) for f in dirs])
    analyse_results(exp_dir / "results")

if __name__ == "__main__":
    import sys
    import os

    # Add the root directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    strategies = ["epsilon_greedy", "boltzmann", "max_boltzmann"]
    options = [(f"train_{s}", lambda s=s: train(s)) for s in strategies]
    options += [
        ("eval", lambda: evaluate(EXPERIMENTS_DIR)),
        ("analyze", lambda: analyze(EXPERIMENTS_DIR)),
    ]
    picker(options, default=None)()