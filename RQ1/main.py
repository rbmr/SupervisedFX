import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
from datetime import datetime
from pathlib import Path

import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
        n_actions=0,
        custom_reward_function=log_equity_diff,
        shuffled=shuffled,
    )
    logging.info("Environments created.")

    return train_env, eval_env

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Each observation is (obs_dim,). We treat each call as a “sequence of length=1,”
    let SB3’s recurrent wrapper handle hidden states across time, and project
    the final LSTM output (64) down to 32 with LeakyReLU.
    """
    def __init__(self, observation_space, features_dim: int = 32):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        lstm_hidden = 64
        num_layers = 2

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.post_lstm = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.LeakyReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, obs_dim)
        x = observations.unsqueeze(1)         # → (batch_size, 1, obs_dim)
        lstm_out, _ = self.lstm(x)            # → (batch_size, 1, 64)
        last_step = lstm_out[:, -1, :]        # → (batch_size, 64)
        return self.post_lstm(last_step)      # → (batch_size, 32)

class CustomLSTMActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        """
        We expect alpha_actor and alpha_critic in kwargs.pop("alpha_actor"/"alpha_critic"),
        then pass them into the policy so that _setup_optimizer can see them.
        """
        # Extract and store α_actor / α_critic; pop them so parent __init__ doesn't choke
        self.alpha_actor = kwargs.pop("alpha_actor")
        self.alpha_critic = kwargs.pop("alpha_critic")

        # Force SB3 to use our LSTMFeatureExtractor and two‐layer MLP heads (64 → 32).
        kwargs["features_extractor_class"] = LSTMFeatureExtractor
        kwargs["net_arch"] = dict(pi=[64, 32], vf=[64, 32])
        kwargs["activation_fn"] = nn.LeakyReLU

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _setup_optimizer(self) -> None:
        """
        Build a single Adam optimizer with TWO param‐groups:
        SB3 will call this once after networks are built.
        """
        # 1) Collect actor vs. critic parameters
        actor_params = list(self.mlp_extractor.policy_net.parameters())
        actor_params += list(self.action_net.parameters())
        critic_params = list(self.mlp_extractor.value_net.parameters())

        # 2) Create Adam with two learning rates
        OptimClass = self.optimizer_class  # should be th.optim.Adam
        self.optimizer = OptimClass(
            [
                {"params": actor_params,  "lr": self.alpha_actor},
                {"params": critic_params, "lr": self.alpha_critic},
            ]
        )

        # SB3 expects these attributes to exist, even if empty:
        self.optimizer_config = {}
        self.lr_schedule_kwargs = {}

def get_model(env: ForexEnv):

    alpha_critic = 0.001
    alpha_actor  = 0.0001
    gamma        = 0.3
    batch_size   = 128
    ent_coef     = 0.02
    bp_rate      = 0.0020 # should be set in the environment

    # — policy_kwargs must include alpha_actor & alpha_critic —
    policy_kwargs = dict(
        optimizer_class=th.optim.Adam,
        alpha_actor=alpha_actor,
        alpha_critic=alpha_critic,
    )

    model = A2C(
        policy=CustomLSTMActorCriticPolicy,
        env=env,
        learning_rate=alpha_critic,  # pass a float so SB3 type-checks OK
        n_steps=batch_size,
        gamma=gamma,
        verbose=1,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        use_rms_prop=False,  # ensure Adam is used
        device="cpu",
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

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1, num_workers=4)

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

