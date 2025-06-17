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
from common.models.train_eval import evaluate_and_analyze_model, train_model, analyse_results, evaluate_models
from common.models.utils import save_model_with_metadata
from common.scripts import picker, has_nonempty_subdir, n_children
from common.envs.dp import get_dp_table_from_env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PARAM_PRESETS = {
    "M30_0.7": {
        "granularity": Timeframe.M30,
        "start_time": datetime(2020, 1, 1, 22),
        "end_time": datetime(2024, 12, 31, 21, 30),
        "split_ratio": 0.7
    },
    "M15_0.8": {
        "granularity": Timeframe.M15,
        "start_time": datetime(2017, 1, 1, 22),
        "end_time": datetime(2024, 12, 31, 21, 45),
        "split_ratio": 0.8
    },
    "M15_0.7": {
        "granularity": Timeframe.M15,
        "start_time": datetime(2017, 1, 1, 22),
        "end_time": datetime(2024, 12, 31, 21, 45),
        "split_ratio": 0.7
    },
    "M30_0.8": {
        "granularity": Timeframe.M30,
        "start_time": datetime(2020, 1, 1, 22),
        "end_time": datetime(2024, 12, 31, 21, 30),
        "split_ratio": 0.8
    }
}

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

def get_environments(data_config, use_optimal_reward=False):
    logging.info("Loading market data...")
    data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=data_config["granularity"],
        start_time=data_config["start_time"],
        end_time=data_config["end_time"],
    )

    logging.info("Creating feature pipelines...")
    market_fe = get_feature_engineer()
    agent_fe = StepwiseFeatureEngineer()
    agent_fe.add(["current_exposure"], calculate_current_exposure)

    split_ratio = data_config.get("split_ratio", 0.8)

    logging.info("Building environments...")
    if use_optimal_reward:
        train_env, eval_env = ForexEnv.create_train_eval_envs(
            split_ratio=split_ratio,
            forex_candle_data=data,
            market_feature_engineer=market_fe,
            agent_feature_engineer=agent_fe,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            custom_reward_function=None
        )
        table = get_dp_table_from_env(train_env)
        dp_reward = DPRewardFunction(table)

        train_env.custom_reward_fn = dp_reward
        return train_env, eval_env
    else:
        return ForexEnv.create_train_eval_envs(
            split_ratio=split_ratio,
            forex_candle_data=data,
            market_feature_engineer=market_fe,
            agent_feature_engineer=agent_fe,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            custom_reward_function=log_equity_change
        )
    
def run_experiment(exploration_strategy: str, use_optimal_reward=False):
    note = input("Enter a short note for this experiment: ").strip().replace(' ', '_')

    print("Available data configs:", ', '.join(PARAM_PRESETS.keys()))
    config_options = list(PARAM_PRESETS.keys())
    config_key = picker([(key, key) for key in config_options], default="M15_0.8").strip()
    config_key = config_key if config_key in PARAM_PRESETS else "M15_0.8"
    data_config = PARAM_PRESETS[config_key]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{exploration_strategy}_{note}_{config_key}_{timestamp}"
    exp_dir = EXPERIMENTS_DIR / exp_name

    train_episodes = input("Enter number of training episodes (default 25): ")
    train_episodes = int(train_episodes) if train_episodes.isdigit() else 25

    # Train
    train_env, eval_env = get_environments(data_config, use_optimal_reward=use_optimal_reward)
    logging.info(f"Instantiating {exploration_strategy.upper()} model...")

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
        eps_init = input("Epsilon initial (default 1.0): ")
        eps_final = input("Epsilon final (default 0.05): ")
        exploration_fraction = input("Exploration fraction (default 0.8): ")

        dqn_args.update({
            "exploration_initial_eps": float(eps_init) if eps_init else 1.0,
            "exploration_final_eps": float(eps_final) if eps_final else 0.05,
            "exploration_fraction": float(exploration_fraction) if exploration_fraction else 0.8
        })
        from stable_baselines3 import DQN
        model = DQN(**dqn_args)

    elif exploration_strategy == "boltzmann":
        temperature = input("Temperature (default 1.0): ")
        from RQ5.boltzmann_dqn import BoltzmannDQN
        model = BoltzmannDQN(**dqn_args, temperature=float(temperature) if temperature else 1.0)

    elif exploration_strategy == "max_boltzmann":
        temperature = input("Temperature (default 1.0): ")
        epsilon = input("Epsilon (default 0.1): ")
        from RQ5.boltzmann_dqn import MaxBoltzmannDQN
        model = MaxBoltzmannDQN(
            **dqn_args,
            temperature=float(temperature) if temperature else 1.0,
            epsilon=float(epsilon) if epsilon else 0.1
        )
    elif exploration_strategy == "curiosity":
        from common.envs.curiosity import CuriosityModule
        from stable_baselines3 import DQN
        from common.models.train_eval import train_model_with_curiosity

        curiosity_beta = input("Curiosity beta (intrinsic reward scaling, default 0.2): ")
        curiosity_beta = float(curiosity_beta) if curiosity_beta else 0.2

        dqn_args.update({
            "exploration_initial_eps": 0.1,
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.8
        })

        model = DQN(**dqn_args)

        curiosity_module = CuriosityModule(
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.n,
            hidden_dim=128,
            lr=1e-4
        )

        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        callback = [
            SaveCallback(models_dir, save_freq=train_env.episode_len),
            ActionHistogramCallback(train_env, log_freq=train_env.episode_len),
            SneakyLogger(verbose=0)  # reduce logging to avoid slowdowns
        ]

        logging.info("Starting curiosity-driven training...")
        train_model_with_curiosity(
            model=model,
            curiosity_module=curiosity_module,
            train_env=train_env,
            train_episodes=train_episodes,
            beta=curiosity_beta,
            model_save_path=models_dir,
            callback=callback
        )

        save_model_with_metadata(model, models_dir / "model_final.zip")

    else:
        raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")

        
    # Only train again if strategy is not curiosity (already trained above)
    if exploration_strategy != "curiosity":
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        callback = [
            SaveCallback(models_dir, save_freq=train_env.episode_len),
            ActionHistogramCallback(train_env, log_freq=train_env.episode_len),
            SneakyLogger(verbose=1)
        ]

        logging.info("Starting training...")
        train_model(model, train_env, train_episodes=train_episodes, callback=callback)
        save_model_with_metadata(model, models_dir / "model_final.zip")

    # Evaluate and analyze model (always done regardless of strategy)
    logging.info("Evaluating and Analyzing results...")
    evaluate_and_analyze_model(exp_dir, train_env, eval_env)

if __name__ == "__main__":
    strategies = ["epsilon_greedy", "boltzmann", "max_boltzmann", "curiosity"]
    options = [(f"run_{s}", lambda s=s: run_experiment(s)) for s in strategies]
    options += [("run_optimal", lambda: run_experiment("optimal_reward", use_optimal_reward=True))]

    picker(options, default=None)()