import logging
from datetime import datetime
from pathlib import Path

import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from common.data.data import ForexCandleData, Timeframe
from common.data.stepwise_feature_engineer import (StepwiseFeatureEngineer,
                                                   calculate_current_exposure)
from common.envs.callbacks import (ActionHistogramCallback, CoolStatsCallback,
                                   SaveOnEpisodeEndCallback)
from common.envs.forex_env import ForexEnv, log_equity_diff
from common.models.train_eval import (analyse_results, evaluate_models,
                                      train_model)
from common.models.utils import save_model_with_metadata
from common.scripts import has_nonempty_subdir, n_children, picker
from RQ1.constants import EXPERIMENT_NAME_FORMAT, EXPERIMENTS_DIR
from RQ1.some_feature_engineers import get_feature_engineer_chatgpt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_environments(shuffled = False):

    logging.info("Loading market data...")
    forex_candle_data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M15,
        start_time=datetime(2022, 1, 2, 22, 0, 0, 0),
        end_time=datetime(2025, 5, 16, 20, 45, 0, 0),
    )

    logging.info("Setting up feature engineer...")
    market_feature_engineer = get_feature_engineer_chatgpt()

    logging.info("Setting up stepwise feature engineer...")
    agent_feature_engineer = StepwiseFeatureEngineer()
    agent_feature_engineer.add(["current_exposure"], calculate_current_exposure)

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=0.7,
        forex_candle_data=forex_candle_data,
        market_feature_engineer=market_feature_engineer,
        agent_feature_engineer=agent_feature_engineer,
        initial_capital=10_000.0,
        transaction_cost_pct=0.0,
        n_actions=1,
        custom_reward_function=log_equity_diff,
        shuffled=shuffled,
    )
    logging.info("Environments created.")

    return train_env, eval_env

class CustomLSTMActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomLSTMActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

def get_model(env: ForexEnv):

    learning_rate_actor = 0.0001
    learning_rate_critic = 0.001
    gamma = 0.3
    batch_size = 128

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 32], vf=[64, 32]),  # Actor (pi) and Critic (vf) networks
        activation_fn=nn.LeakyReLU,
        optimizer_class=th.optim.Adam,
    )

    model = A2C(
        policy=CustomLSTMActorCriticPolicy,
        env=env,  # Use the actual train_env you have
        learning_rate=learning_rate_critic,
        n_steps=batch_size,
        gamma=gamma,
        verbose=1,
        policy_kwargs=policy_kwargs,
        use_rms_prop=False,
        device="cpu"
    )

    return model

def train():

    train_env, _ = get_environments(shuffled=True)

    logging.info("Creating model...")

    model = get_model(train_env)

    logging.info("Model created.")

    experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveOnEpisodeEndCallback(models_dir),
                ActionHistogramCallback(train_env, log_freq=train_env.total_steps),
                CoolStatsCallback(train_env, log_freq=train_env.total_steps)]
    train_model(model, train_env, train_episodes=200, callback=callback)
    save_model_with_metadata(model, models_dir / "model_final.zip")

def evaluate(experiments_dir, limit = 10):

    experiment_dirs: list[Path] = list(experiments_dir.iterdir())
    experiment_dirs = list(f for f in experiment_dirs if has_nonempty_subdir(f, "models"))
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    experiment_dirs = experiment_dirs[:limit] if limit is not None else experiment_dirs
    named_dirs = list((f"{f.name} ({n_children(f/"models")})", f) for f in experiment_dirs)

    experiment_dir = picker(named_dirs)
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = get_environments(shuffled=False)
    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1)

def analyze(experiments_dir, limit = 10):

    experiment_dirs: list[Path] = list(experiments_dir.iterdir())
    experiment_dirs = list(f for f in experiment_dirs if has_nonempty_subdir(f, "results"))
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    experiment_dirs = experiment_dirs[:limit] if limit is not None else experiment_dirs
    named_dirs = list((f"{f.name} ({n_children(f/"results")})", f) for f in experiment_dirs)

    experiment_dir = picker(named_dirs)
    results_dir = experiment_dir / "results"

    analyse_results(results_dir)

if __name__ == "__main__":

    options = [
        ("train", train),
        ("eval", lambda: evaluate(EXPERIMENTS_DIR, 10)),
        ("analyze", lambda: analyze(EXPERIMENTS_DIR, 10)),
    ]
    picker(options, default=None)()

