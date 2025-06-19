import os

from torch.optim import Optimizer

from common.data.data import ForexCandleData, Timeframe

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s, %(name)s, PID:%(process)d] %(message)s' )
import argparse
import itertools
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Callable, Type
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from torch import nn

from common.data.feature_engineer import (FeatureEngineer, as_min_max_fixed,
                                          as_ratio_of_other_column, as_z_score,
                                          bollinger_bands, complex_7d, complex_24h,
                                          history_lookback, macd, mfi,
                                          parabolic_sar, remove_columns, vwap, as_robust_norm)
from common.data.stepwise_feature_engineer import (StepwiseFeatureEngineer,
                                                   calculate_current_exposure)
from common.envs.callbacks import ActionHistogramCallback, SaveCallback
from common.envs.dp import get_dp_table_from_env
from common.envs.forex_env import (ActionConfig, DataConfig, EnvConfig,
                                   ForexEnv, ObsConfig)
from common.envs.rewards import DPRewardFunction
from common.models.train_eval import (analyse_results, combine_finals,
                                      evaluate_dummy, evaluate_models,
                                      train_model)
from common.scripts import parallel_apply, lazy_singleton
from RQ1.constants import CUD_COLORS, DUMMY_MODELS, RQ1_EXPERIMENTS_DIR, SEED, SPLIT_RATIO, MARKERS
from RQ1.utils import get_n_params, get_widths, get_shapes
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class CnnOnlyExtractor(BaseFeaturesExtractor):
    """
    Custom CNN extractor for windowed time-series data. The CNN is designed to output a specific number of features.
    :param observation_space: (gym.Space) The observation space for the CNN input.
    :param features_dim: (int) Number of features to be extracted.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 27):
        super().__init__(observation_space, features_dim)
        # Expected shapes (channels=1, features, time_steps)
        assert len(observation_space.shape) == 3, f"unexpected observation_space shape {observation_space.shape}"
        assert observation_space.shape[0] == 1, f"first dim (channel) should be of size 1, got {observation_space.shape}"

        n_input_channels, n_features, seq_len = observation_space.shape

        # Define CNN: first conv collapses feature dimension, then temporal conv
        self.cnn = nn.Sequential(
            # Conv across all features and small temporal window
            nn.Conv2d(n_input_channels, 32, kernel_size=(n_features, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            # Further temporal convolution
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(),
            # Optional pooling to reduce temporal dimension
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Flatten(),
        )

        # Compute output dimension of CNN
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, n_features, seq_len)
            cnn_output = self.cnn(sample)
        cnn_out_dim = cnn_output.shape[1]

        # Final linear layer to get desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch_size, 1, features, time_steps)
        return self.linear(self.cnn(observations.float()))

class CnnCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that combines the output of a CNN for image-like data
    and a simple FlattenExtractor for vector data.
    """
    cnn_output_dim = 27

    def __init__(self, observation_space: gym.spaces.Dict):

        extractors = {
            "cnn_input": CnnOnlyExtractor(observation_space.spaces["cnn_input"], features_dim=self.cnn_output_dim), # type: ignore
            "vector_input": FlattenExtractor(observation_space.spaces["vector_input"]),
        }

        # Calculate the total feature dimension from all sub-extractors
        vector_input_dim = gym.spaces.flatdim(observation_space.spaces["vector_input"])
        total_features_dim = self.cnn_output_dim + vector_input_dim

        super().__init__(observation_space, features_dim=total_features_dim)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Processes dictionary observations by passing each item to its corresponding extractor
        and concatenating the results.
        """
        encoded_tensor_list = [
            extractor(observations[key]) for key, extractor in self.extractors.items()
        ]
        return torch.cat(encoded_tensor_list, dim=1)

def add_technical_analysis(df):
    """Default technical analysis features"""

    # Trend (6 features)

    parabolic_sar(df)
    as_ratio_of_other_column(df, 'sar', 'close_bid')

    vwap(df, window=12)
    vwap(df, window=48)
    as_ratio_of_other_column(df, 'vwap_12', 'close_bid')
    as_ratio_of_other_column(df, 'vwap_48', 'close_bid')

    history_lookback(df, 1, ["sar", "vwap_12", "vwap_48"])

    # Momentum (4 features)

    macd(df, short_window=12, long_window=26, signal_window=9)
    remove_columns(df, ["macd_signal", "macd"])
    as_z_score(df, 'macd_hist', window=50)

    mfi(df, window=14)
    as_min_max_fixed(df, "mfi_14", min=0, max=100)

    history_lookback(df, 1, ["macd_hist", "mfi_14"])

    # Technical Analysis (4 features)

    bollinger_bands(df, window=20, num_std_dev=2)
    as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
    as_ratio_of_other_column(df, "bb_lower_20", "close_bid")

    history_lookback(df, 1, ["bb_upper_20", "bb_lower_20"])

def add_cnn_features(df):
    """Feature engineering for CNN input"""

    ohlc_cols = ['open_bid', 'high_bid', 'low_bid', 'close_bid',
                 'open_ask', 'high_ask', 'low_ask', 'close_ask']

    # Remove date column
    remove_columns(df, columns=["date_gmt"])

    # Convert to log returns
    for col in ohlc_cols:
        ratio = np.array(df[col] / df[col].shift(1))
        return_pct = ratio - 1 # within [-0.01, 0.01] for almost all data.
        df[f'{col}_return'] = return_pct * 100 # scaled return
        df[f'{col}_return'] = df[f'{col}_return'].fillna(0)

    # Add volume normalization
    as_robust_norm(df, column="volume", window=100)

    # Add some technical indicators as raw features
    df['hl_ratio'] = (df['high_bid'] - df['low_bid']) / df['close_bid']
    df['oc_ratio'] = (df['open_bid'] - df['close_bid']) / df['close_bid']
    df['spread'] = (df['close_ask'] - df['close_bid']) / df['close_bid']

    # Remove original OHLC columns
    remove_columns(df, columns=ohlc_cols)

    return df


@lazy_singleton
def get_default_forex_data() -> ForexCandleData:
    return ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.H1,
        start_time=datetime(2020, 1, 1, 22, 0, 0, 0),
        end_time=datetime(2024, 12, 31, 21, 00, 0, 0),
    )

@lazy_singleton
def get_default_env_config() -> EnvConfig:
    return EnvConfig(
        initial_capital=10_000,
        transaction_cost_pct=5 / 100_000,
        reward_function=None
    )

@lazy_singleton
def get_default_action_config() -> ActionConfig:
    return ActionConfig(
        n=0, # continuous actions
        low=-1.0,
        high=1.0
    )

@lazy_singleton
def get_default_data_configs() -> tuple[DataConfig, DataConfig]:
    fe = FeatureEngineer()
    fe.add(complex_24h)  # 2 features
    fe.add(complex_7d)  # 2 features
    fe.add(add_technical_analysis)

    sfe = StepwiseFeatureEngineer()
    sfe.add(["current_exposure"], calculate_current_exposure)  # 1 feature

    obs_configs = [ObsConfig(name='market_features', fe=fe, sfe=sfe, window=1)]
    train_data_config, eval_data_config = DataConfig.from_splits(
        forex_candle_data=get_default_forex_data(),
        split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
        obs_configs=obs_configs
    )
    return train_data_config, eval_data_config

DEFAULT_INP = 32 # default input size
DEFAULT_OUT = 1 # default output sizes

@dataclass
class ExperimentConfig:

    # General
    name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Matplotlib settings
    line_color: str = "black"
    line_style: str = "-"
    line_marker: Optional[str] = None

    # Environment settings
    env_config: EnvConfig = field(default_factory=get_default_env_config)
    action_config: ActionConfig = field(default_factory=get_default_action_config)
    train_data_config: Optional[DataConfig] = None
    eval_data_config: Optional[DataConfig] = None

    # Model settings
    policy: str = "MlpPolicy"
    device: str = "cpu"
    activation_fn: Callable = nn.ReLU
    net_shape: Optional[list[int]] = None # Shortcut for setting actor and critic shape.
    actor_shape: list[int] = field(default_factory=lambda: [64, 64])
    critic_shape: list[int] = field(default_factory=lambda: [64, 64])
    features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None
    features_extractor_kwargs: Optional[dict] = None
    optimizer_kwargs: Optional[dict] = None
    optimizer_class: Optional[Type[Optimizer]] = None

    def __post_init__(self):
        if self.net_shape is not None:
            self.actor_shape = self.net_shape
            self.critic_shape = self.net_shape
        if self.train_data_config is None or self.eval_data_config is None:
            train_data_config, eval_data_config = get_default_data_configs()
            self.train_data_config = self.train_data_config or train_data_config
            self.eval_data_config = self.eval_data_config or eval_data_config

    def get_policy_kwargs(self):
        policy_kwargs = dict(
            activation_fn=self.activation_fn,
            net_arch=dict(pi=self.actor_shape, qf=self.critic_shape),
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )
        return {k: v for k, v in policy_kwargs.items() if v is not None}

    def get_style(self):
        return dict(
            color = self.line_color,
            linestyle = self.line_style,
            marker = self.line_marker,
        )

    def log_info(self, experiment_dir: Path):
        """Logs some information about this experiment to a JSON file inside experiment_dir."""
        info = dict(
            name = self.name,
            actor_shape = self.actor_shape,
            critic_shape = self.critic_shape,
            activation_fn = self.activation_fn.__name__,
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "experiment_config.json", "w") as f:
            json.dump(info, f, indent=4) # type: ignore

def _run_experiment(experiment_group: str, config: ExperimentConfig, seed: int = SEED):
    """
    Runs a single experiment: trains the model, evaluates it, and analyzes the results.
    """
    logging.info(f"Running experiment {experiment_group} / {config.name} / seed_{seed}")

    # Set seeds, only important for training, evaluation is deterministic.
    # The ForexEnvs are entirely deterministic, taking the same sequence of actions,
    # always leads to the same result.
    # Model seed is set upon model creation.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Get environments

    train_env = ForexEnv(config.action_config, config.env_config, config.train_data_config)
    train_env.custom_reward_fn = DPRewardFunction(get_dp_table_from_env(train_env))

    eval_env = ForexEnv(config.action_config, config.env_config, config.eval_data_config)
    eval_env.custom_reward_fn = DPRewardFunction(get_dp_table_from_env(eval_env))

    # Setup directories

    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_group / config.name / f"seed_{seed}"
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"
    config.log_info(experiment_dir)  # log some info about this experiment

    # Train model
    if not models_dir.exists():
        model = SAC(
            policy=config.policy,
            env=train_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=1.0,
            ent_coef='auto',
            gradient_steps=2,
            train_freq=48,
            policy_kwargs=config.get_policy_kwargs(),
            verbose=0,
            device=config.device,
            seed=seed
        )
        # SaveCallback creates models_dir upon first model save.
        callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
                    ActionHistogramCallback(train_env, log_freq=train_env.episode_len)]
        train_model(model, train_env, train_episodes=80, callback=callback)

    # Evaluate models

    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_models(models_dir, results_dir, eval_envs, num_workers=3)

    # Analyze results

    analyse_results(results_dir)

def _run_experiment_wrapper(config_seed: tuple[ExperimentConfig, int], experiment_group: str):
    """
    Helper to unpack arguments for parallel processing.
    """
    config, seed = config_seed
    _run_experiment(experiment_group, config, seed)

def _run_experiments(experiment_group: str, experiments: List[ExperimentConfig], n_seeds=1, num_workers=3, add_timestamp: bool=True):
    """
    Runs each of the experiments for a number of seeds.
    """
    logging.info(f"Running {experiment_group}, with {len(experiments)} experiments, for {n_seeds} seeds.")
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_group = f"{timestamp}_{experiment_group}"

    func = partial(_run_experiment_wrapper, experiment_group=experiment_group)
    inputs = list(itertools.product(experiments, range(SEED, SEED + n_seeds)))
    parallel_apply(func, inputs, num_workers=num_workers)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")

def run_baselines():
    """
    Runs the baseline models on the train and eval environments. Used as reference.
    """
    train_data_config, eval_data_config = get_default_data_configs()
    action_config = get_default_action_config()
    env_config = get_default_env_config()
    train_env = ForexEnv(action_config, env_config, train_data_config)
    eval_env = ForexEnv(action_config, env_config, eval_data_config)

    experiment_group = "baselines"
    experiment_group_dir = RQ1_EXPERIMENTS_DIR / experiment_group

    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }
    results_dir = experiment_group_dir
    for dummy_name, dummy_factory in DUMMY_MODELS.items():
        for eval_env_name, eval_env in eval_envs.items():
            dummy_model = dummy_factory(eval_env)
            evaluate_dummy(dummy_model = dummy_model, name=dummy_name, results_dir=results_dir, eval_env=eval_env, eval_env_name=eval_env_name)

    analyse_results(results_dir)

def run_shape_experiments():
    """
    Determine the impact of network shapes on model performance.
    Number of parameters remains roughly equal.
    """
    shapes = get_shapes(DEFAULT_INP, DEFAULT_OUT, level_low=16, level_high=64)

    experiments = [
        ExperimentConfig(name="shape_flat", net_shape=shapes["flat"], line_color=CUD_COLORS[0], line_marker="s"),
        ExperimentConfig(name="shape_diamond", net_shape=shapes["diamond"], line_color=CUD_COLORS[1], line_marker="D"),
        ExperimentConfig(name="shape_funnel", net_shape=shapes["funnel"], line_color=CUD_COLORS[2], line_marker=">"),
        ExperimentConfig(name="shape_inv_funnel", net_shape=shapes["inv_funnel"], line_color=CUD_COLORS[3], line_marker="<"),
        ExperimentConfig(name="shape_shallow", net_shape=shapes["shallow"], line_color=CUD_COLORS[4], line_marker="*"),
    ]

    _run_experiments(experiment_group="network_shapes", experiments=experiments, n_seeds=5)

def run_division_experiments():
    """
    Determine the impact of dividing parameters over the actor and the critic networks.
    Number of parameters remains roughly equal.
    """
    # No bias
    base_net = [32, 32]
    experiments = [ExperimentConfig(name="no_bias", net_shape=base_net, line_color=CUD_COLORS[0], line_marker="o"),]

    # Slight bias
    n_layers = 2
    total_params = get_n_params(DEFAULT_INP, *base_net, DEFAULT_OUT) * 2
    w1, w2 = get_widths(DEFAULT_INP, DEFAULT_OUT, n_layers, total_params, 0.60)
    experiments.append(ExperimentConfig(name="moderate_actor_bias",actor_shape=[w1]*n_layers,critic_shape=[w2]*n_layers, line_color=CUD_COLORS[1], line_marker="s"))
    experiments.append(ExperimentConfig(name="moderate_critic_bias",actor_shape=[w2]*n_layers,critic_shape=[w1]*n_layers, line_color=CUD_COLORS[2], line_marker="D"))

    # Large bias
    w1, w2 = get_widths(DEFAULT_INP, DEFAULT_OUT, n_layers, total_params, 0.75)
    experiments.append(ExperimentConfig(name="large_actor_bias",actor_shape=[w1]*n_layers,critic_shape=[w2]*n_layers, line_color=CUD_COLORS[3], line_marker=">"))
    experiments.append(ExperimentConfig(name="large_critic_bias",actor_shape=[w2]*n_layers,critic_shape=[w1]*n_layers, line_color=CUD_COLORS[4], line_marker="<"))

    _run_experiments(experiment_group="network_division", experiments=experiments, n_seeds=5)

def run_size_experiments():
    """
    Determine the impact of network size on model performance.
    We modify two aspects: Model depth (number of layers), and Model width (number of neurons per layer).
    """
    for depth_name, depth in zip(["shallow", "moderate", "deep", "very_deep"], [1, 2, 3, 4]):
        experiments = [
            ExperimentConfig(name=f"{depth_name}-{width_name}", net_shape=[width] * depth, line_marker=marker, line_color=color)
            for width_name, width, color, marker in zip(["narrow", "moderate", "wide", "very_wide"], [8, 16, 32, 64], CUD_COLORS[:4], MARKERS[:4])
        ]
        _run_experiments(experiment_group=f"{depth_name}_networks", experiments=experiments)

def run_activation_experiments():
    """
    Determine the impact of network activation functions on model performance.
    All networks are the same size and shape.
    """
    logging.info("Running activation function experiments...")

    experiments = []

    for activation_fn, color, marker in zip([nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.SiLU, nn.Tanh, nn.ELU], CUD_COLORS[:6], MARKERS[:6]):
        experiments.append(ExperimentConfig(
            name = activation_fn.__name__,
            line_marker = marker,
            line_color = color,
            activation_fn=activation_fn,
        ))

    _run_experiments(experiment_group=f"activation_functions", experiments=experiments, n_seeds=3)

def run_cnn_experiments():
    """
    Compares a model using technical analysis features against one using a CNN
    to process normalized candlestick data.
    """
    logging.info("Running CNN experiments...")

    # Setup config for CNN
    vector_fe = FeatureEngineer()
    vector_fe.add(complex_24h)
    vector_fe.add(complex_7d)
    vector_sfe = StepwiseFeatureEngineer()
    vector_sfe.add(["current_exposure"], calculate_current_exposure)
    vector_obs_config = ObsConfig(name='vector_input', fe=vector_fe, sfe=vector_sfe, window=1)

    cnn_fe = FeatureEngineer(remove_original_columns=False)
    cnn_fe.add(add_cnn_features)
    cnn_obs_config = ObsConfig(name='cnn_input', fe=cnn_fe, sfe=None, window=48)

    train_data_config, eval_data_config = DataConfig.from_splits(
        forex_candle_data=get_default_forex_data(),
        split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
        obs_configs=[vector_obs_config, cnn_obs_config]
    )

    # Run experiments
    experiments = [
        ExperimentConfig(
            name="cnn_features",
            net_shape=[64, 64],
            line_color=CUD_COLORS[1],
            line_marker="X",
            train_data_config=train_data_config,
            eval_data_config=eval_data_config,
            device="auto", # use cuda if available
            policy="MultiInputPolicy",
            features_extractor_class=CnnCombinedExtractor,
        ),
        ExperimentConfig(
            name="technical_analysis",
            net_shape=[64, 64],
            line_color=CUD_COLORS[0],
            line_marker="o",
            device="cpu",
        )
    ]
    _run_experiments(
        experiment_group="cnn_vs_ta",
        experiments=experiments,
        n_seeds=3,
    )

def run_decay_experiments():
    """

    """
    logging.info("Running decay experiments...")

    experiments = [
        ExperimentConfig(
            name=f"decay_{weight_decay}",
            net_shape=[32, 32],
            line_color=color,
            line_marker=marker,
            device="cpu",
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(weight_decay=weight_decay),
        )
        for weight_decay, marker, color in zip([1e-6, 2e-6, 4e-6, 8e-6, 16e-6], MARKERS, CUD_COLORS)
    ]

    _run_experiments(
        experiment_group="weight_decay",
        experiments=experiments,
        n_seeds=1,
    )

def run():
    """
    Parses command-line arguments and executes the chosen experiment(s).
    """

    experiments = [
        run_baselines,
        run_decay_experiments,
        run_size_experiments,
        run_shape_experiments,
        run_activation_experiments,
        run_division_experiments,
        run_cnn_experiments,
    ]

    def run_all():
        for f in experiments:
            f()

    experiments_to_run = {i+1: f for i, f in enumerate(experiments)}
    experiments_to_run[0] = run_all

    help_str = "\n".join(f"{exp_id}: {fn.__name__}" for exp_id, fn in experiments_to_run.items())

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--exp_id", default=0, type=int, help=help_str)
    exp_id = parser.parse_args().exp_id

    if exp_id in experiments_to_run:
        experiment_fn = experiments_to_run[exp_id]
        experiment_fn()
    else:
        logging.error(f"Error: Unknown experiment ID {exp_id}")
        parser.print_help()
        exit(1)

if __name__ == "__main__":

    run()