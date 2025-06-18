import os

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
from typing import List, Optional
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from torch import nn

from common.data.feature_engineer import (FeatureEngineer, as_min_max_fixed,
                                          as_ratio_of_other_column, as_z_score,
                                          bollinger_bands, chaikin_volatility,
                                          complex_7d, complex_24h,
                                          history_lookback, macd, mfi,
                                          parabolic_sar, remove_columns, vwap, copy_columns, as_pct_change,
                                          apply_column, as_robust_norm)
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
from common.scripts import parallel_apply
from RQ1.constants import (ACTION_HIGH, ACTION_LOW, CUD_COLORS, DUMMY_MODELS,
                           INITIAL_CAPITAL, MARKERS, N_ACTIONS,
                           RQ1_EXPERIMENTS_DIR, SAC_HYPERPARAMS, SEED,
                           SPLIT_RATIO, TRANSACTION_COST_PCT)
from RQ1.scripts import get_n_flops, get_n_neurons, get_n_params, get_widths
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class CnnOnlyExtractor(BaseFeaturesExtractor):
    """
    Custom CNN extractor for windowed time-series data. The CNN is designed to output a specific number of features.
    :param observation_space: (gym.Space) The observation space for the CNN input.
    :param features_dim: (int) Number of features to be extracted.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 59):
        super().__init__(observation_space, features_dim)
        # Input to Conv2d should be: (batch_size, 1, num_features, lookback_window)
        # We add the channel dimension '1' before passing data to the network.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
        )

        # Compute the shape of the output of the CNN layers by doing one forward pass
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs.unsqueeze(1)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Add the channel dimension for the Conv2d layers
        return self.linear(self.cnn(observations.unsqueeze(1)))

class CnnCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that combines the output of a CNN for image-like data
    and a simple FlattenExtractor for vector data.
    """
    cnn_output_dim = 59

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

def get_shapes(inp: int, out: int):
    """
    Given some network shapes, and constraints on these shapes, determines the division
    of neurons across these layers such that the difference in number of parameters is minimized.
    Works in polynomial time.
    """

    logging.info("Setting up valid configurations")

    # Hard constraints:
    # (1) No layer has an absurd size: level_low <= x, a, (a*b), c, (c*d), (c*d*d), e, (e*f), (e*f*f) <= level_high
    # (2) The difference in shape is significant: 1.5 <= b, d, f <= 2.0

    fac_range = np.linspace(1.5, 2.0, 100)
    level_low = 32
    level_high = 80

    shape_layers_fn = {
        "flat": lambda x: (int(x), int(x), int(x)),
        "diamond": lambda a, b: (int(a), int(a*b), int(a)),
        "inv_funnel": lambda c, d: (int(c), int(c*d), int(c*d*d)),
        "funnel": lambda e, f: (int(e*f*f), int(e*f), int(e)),
        "shallow": lambda g: (int(g), int(g)),
    }

    shape_configs = {
        "flat": [(x,) for x in range(level_low, level_high + 1)],
        "diamond": [(a, float(b)) for b in fac_range for a in range(level_low, int(level_high / b) + 1)],
        "inv_funnel": [(c, float(d)) for d in fac_range for c in range(level_low, int(level_high / (d * d)) + 1)],
        "funnel": [(e, float(f)) for f in fac_range for e in range(level_low, int(level_high / (f * f)) + 1)],
        "shallow" : [(g,) for g in range(level_low, level_high + 1)]
    }

    logging.info("Sorting and caching shape info")

    shapes = set(shape_configs.keys()) & set(shape_layers_fn.keys())

    shape_arch_params_pairs = {}
    for s in shapes:
        layers_fn = shape_layers_fn[s]
        configs = shape_configs[s]
        archs = [layers_fn(*config) for config in configs]
        arch_params_pairs = [(arch, get_n_params(inp, *arch, out)) for arch in archs]
        shape_arch_params_pairs[s] = arch_params_pairs

    for config_params in shape_arch_params_pairs.values():
        config_params.sort(key=lambda x: x[1])

    logging.info("Finding optimal solution")

    # Loop over all configurations to find the minimal combination.
    # We loop in order of minimal n_params, repeatedly taking the difference between the min and max.
    # Every iteration we increment the index for the shape corresponding to min n_params.
    # This either gives:
    # (1) An equal or better min max diff (since the new value is in [old min, old max] and new min >= old min)
    # (2) A different max (the new value is > old max), which could have a worse, better, or equal min max diff.
    # We can exit when any value reaches the last element because that means the index that was incremented
    # corresponded to the minimum, and incrementing any other will therefore cause the min max diff to increase.
    shape_i = {s: 0 for s in shapes}
    min_diff = float("inf")
    solution = None

    while all(shape_i[s] < len(shape_arch_params_pairs[s]) for s in shapes):

        params = tuple(shape_arch_params_pairs[s][shape_i[s]][1] for s in shapes)
        max_params = max(params)
        min_params = min(params)
        diff = max_params - min_params

        if diff < min_diff:
            min_diff = diff
            solution = shape_i.copy()

        for s in shapes:
            i = shape_i[s]
            arch_params_pairs = shape_arch_params_pairs[s]
            arch, n_params = arch_params_pairs[i]
            if n_params == min_params:
                shape_i[s] = i + 1

    assert solution is not None, "No solution could be found."

    logging.info("Found optimal solution.")

    result = {}
    for shape, arch_params_pairs in shape_arch_params_pairs.items():
        arch, n_params = arch_params_pairs[solution[shape]]
        n_neurons = get_n_neurons(inp, *arch, out)
        n_flops = get_n_flops(inp, *arch, out)
        logging.info(f"{shape} {arch}: {n_params} params, {n_neurons} neurons, {n_flops} flops")
        result[shape] = arch

    return result

def add_technical_analysis(df):
    """Default technical analysis features"""

    lookback = 3

    # Trend (4 * lookback)

    parabolic_sar(df)
    as_ratio_of_other_column(df, 'sar', 'close_bid')

    vwap(df, window=4)
    vwap(df, window=12)
    vwap(df, window=48)
    as_ratio_of_other_column(df, 'vwap_4', 'close_bid')
    as_ratio_of_other_column(df, 'vwap_12', 'close_bid')
    as_ratio_of_other_column(df, 'vwap_48', 'close_bid')

    history_lookback(df, lookback, ["sar", "vwap_4", "vwap_12", "vwap_48"])

    # Momentum (2 * lookback)

    macd(df, short_window=12, long_window=26, signal_window=9)
    remove_columns(df, ["macd_signal", "macd"])
    as_z_score(df, 'macd_hist', window=50)

    mfi(df, window=14)
    as_min_max_fixed(df, "mfi_14", min=0, max=100)

    history_lookback(df, lookback, ["macd_hist", "mfi_14"])

    # Technical Analysis (3 * lookback)

    bollinger_bands(df, window=20, num_std_dev=2)
    as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
    as_ratio_of_other_column(df, "bb_lower_20", "close_bid")

    chaikin_volatility(df, ema_window=10, roc_period=10) # adds chaikin_vol_{ema_window}_{roc_period}
    as_z_score(df, "chaikin_vol_10_10", window=50)

    history_lookback(df, lookback, ["bb_upper_20", "bb_lower_20", "chaikin_vol_10_10"])


@dataclass
class ExperimentConfig:

    # General
    name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Matplotlib settings
    line_color: str = "black"
    line_style: str = "-"
    line_marker: str = None

    # Environment settings
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        reward_function=None
    ))
    action_config: Optional[ActionConfig] = field(default_factory=lambda: ActionConfig(
        n=N_ACTIONS,
        low=ACTION_LOW,
        high=ACTION_HIGH
    ))
    train_data_config: Optional[DataConfig] = None
    eval_data_config: Optional[DataConfig] = None

    inp: int = 32
    out: int = 1

    # Model settings
    policy = "MlpPolicy"
    device = "cpu"
    activation_fn = nn.ReLU
    net_shape: Optional[list[int]] = None
    actor_shape: list[int] = [64, 64],
    critic_shape: list[int] = [64, 64],
    features_extractor_class = FlattenExtractor
    features_extractor_kwargs = {}

    forex_candle_data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.H1,
        start_time=datetime(2020, 1, 1, 22, 0, 0, 0),
        end_time=datetime(2024, 12, 31, 21, 00, 0, 0),
    )

    def __post_init__(self):
        if self.net_shape is not None:
            self.actor_shape = self.net_shape
            self.critic_shape = self.net_shape
        if self.train_data_config is None or self.eval_data_config is None:


            fe = FeatureEngineer()
            fe.add(complex_24h)  # 2 features
            fe.add(complex_7d)  # 2 features
            fe.add(add_technical_analysis)

            sfe = StepwiseFeatureEngineer()
            sfe.add(["current_exposure"], calculate_current_exposure)  # 1 feature

            obs_configs = [ObsConfig(name='market_features', fe=fe, sfe=sfe, window=1)]
            train_data_config, eval_data_config = DataConfig.from_splits(
                forex_candle_data=self.forex_candle_data,
                split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
                obs_configs=obs_configs
            )
            self.train_data_config = self.train_data_config or train_data_config
            self.eval_data_config = self.eval_data_config or eval_data_config

    def get_policy_kwargs(self):
        return dict(
            activation_fn=self.activation_fn,
            net_arch=dict(pi=self.actor_shape, qf=self.critic_shape),
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )

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
        with open(experiment_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4) # type: ignore

def _run_experiment(experiment_group: str, config: ExperimentConfig, seed: int = SEED):
    """
    Runs a single experiment: trains the model, evaluates it, and analyzes the results.
    """

    # Set seeds, only important for training, evaluation is deterministic.
    # The environments are entirely deterministic, no seeds need to be set.
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
            **SAC_HYPERPARAMS,
            policy_kwargs=config.get_policy_kwargs(),
            verbose=0,
            device=config.device,
            seed=seed
        )
        # SaveCallback creates models_dir upon first model save.
        callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
                    ActionHistogramCallback(train_env, log_freq=train_env.episode_len)]
        train_model(model, train_env, train_episodes=50, callback=callback)

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
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_group = f"{timestamp}_{experiment_group}"

    func = partial(_run_experiment_wrapper, experiment_group=experiment_group)
    inputs = list(itertools.product(experiments, range(SEED, SEED + n_seeds)))
    parallel_apply(func, inputs, num_workers=num_workers)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")

def run_baselines(self):
    """
    Runs the baseline models on the train and eval environments. Used as reference.
    """
    train_env = ForexEnv(self.action_config, self.env_config, self.train_config)
    eval_env = ForexEnv(self.action_config, self.env_config, self.eval_config)

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
    shapes = get_shapes(ExperimentConfig.inp, ExperimentConfig.out)

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
    base_net = [64, 64]
    experiments = [ExperimentConfig(name="no_bias", net_shape=base_net)]

    # Slight bias
    n_layers = 2
    total_params = get_n_params(ExperimentConfig.inp, *base_net, ExperimentConfig.out) * 2
    w1, w2 = get_widths(ExperimentConfig.inp, ExperimentConfig.out, n_layers, total_params, 0.60)
    experiments.append(ExperimentConfig(name="moderate_actor_bias",actor_shape=[w1]*n_layers,critic_shape=[w2]*n_layers))
    experiments.append(ExperimentConfig(name="moderate_critic_bias",actor_shape=[w2]*n_layers,critic_shape=[w1]*n_layers))

    # Large bias
    w1, w2 = get_widths(ExperimentConfig.inp, ExperimentConfig.out, n_layers, total_params, 0.75)
    experiments.append(ExperimentConfig(name="large_actor_bias",actor_shape=[w1]*n_layers,critic_shape=[w2]*n_layers))
    experiments.append(ExperimentConfig(name="large_critic_bias",actor_shape=[w2]*n_layers,critic_shape=[w1]*n_layers))

    _run_experiments(experiment_group="network_division", experiments=experiments, n_seeds=5)

def run_size_experiments():
    """
    Determine the impact of network size on model performance.
    We modify two aspects: Model depth (number of layers), and Model width (number of neurons per layer).
    """
    for depth_name, depth in zip(["shallow", "moderate", "deep", "very_deep"], [1, 2, 3, 4]):
        experiments = [
            ExperimentConfig(name=f"{depth_name}-{width_name}", net_shape=[width] * depth, line_marker=marker, line_color=color)
            for width_name, width, color, marker in zip(["narrow", "moderate", "wide", "very_wide"], [16, 32, 64, 128], CUD_COLORS[:4], MARKERS[:4])
        ]
        _run_experiments(experiment_group=f"{depth_name}_networks", experiments=experiments)

def run_activation_experiments():
    """
    Determine the impact of network activation functions on model performance.
    All networks are the same size and shape.
    """
    experiments = []

    for activation_fn, color, marker in zip([nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.SiLU, nn.Tanh, nn.ELU], CUD_COLORS[:6], MARKERS[:6]):
        experiments.append(ExperimentConfig(
            name = activation_fn.__name__,
            line_marker = marker,
            line_color = color,
            activation_fn=activation_fn,
        ))

    _run_experiments(experiment_group=f"activation_functions", experiments=experiments, n_seeds=5)

def run_cnn_experiments():
    """
    Compares a model using technical analysis features against one using a CNN
    to process raw candlestick data.
    """

    # Setup config for CNN
    vector_fe = FeatureEngineer()
    vector_fe.add(complex_24h)
    vector_fe.add(complex_7d)
    vector_sfe = StepwiseFeatureEngineer()
    vector_sfe.add(["current_exposure"], calculate_current_exposure)
    vector_obs_config = ObsConfig(name='vector_input', fe=vector_fe, sfe=vector_sfe, window=1)

    cnn_fe = FeatureEngineer(remove_original_columns=False)
    ohlc_cols = ['open_bid', 'high_bid', 'low_bid', 'close_bid', "open_ask", "high_ask", "low_ask", "close_ask"]
    cnn_fe.add(remove_columns, columns=["date_gmt"])
    for col in ohlc_cols: # convert to returns, and multiply by 100
        cnn_fe.add(as_pct_change, column=col, periods=1)
        cnn_fe.add(apply_column, fn=lambda x: x * 100, column=col)
    cnn_fe.add(as_robust_norm, column="volume", window=500)
    cnn_obs_config = ObsConfig(name='cnn_input', fe=cnn_fe, sfe=None, window=48)

    train_data_config, eval_data_config = DataConfig.from_splits(
        forex_candle_data=ExperimentConfig.forex_candle_data,
        split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
        obs_configs=[vector_obs_config, cnn_obs_config]
    )

    # Run experiments
    experiments = [
        ExperimentConfig(
            name="technical_analysis",
            net_shape=[64, 64],
            line_color=CUD_COLORS[0],
            line_marker="o",
            train_data_config=train_data_config,
            eval_data_config=eval_data_config,
        ),
        ExperimentConfig(
            name="cnn_features",
            net_shape=[64, 64],
            line_color=CUD_COLORS[1],
            line_marker="X",
            train_data_config=train_data_config,
            eval_data_config=eval_data_config,
        )
    ]
    _run_experiments(
        experiment_group="cnn_vs_ta",
        experiments=experiments,
        n_seeds=3,
        num_workers=3,
        add_timestamp=False,
    )

def run():
    """
    Parses command-line arguments and executes the chosen experiment(s).
    """
    experiments_to_run = {
        1: run_shape_experiments,
        2: run_size_experiments,
        3: run_activation_experiments,
        4: run_baselines,
        5: run_division_experiments,
    }

    def run_all():
        for fn in experiments_to_run.values():
            fn()

    experiments_to_run[0] = run_all
    help_str = "\n".join(f"{exp_id}: {fn.__name__}" for exp_id, fn in experiments_to_run.items())

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("exp_id", default=0, type=int, help=help_str)
    exp_id = parser.parse_args().exp_id

    if exp_id in experiments_to_run:
        experiment_fn = experiments_to_run[exp_id]
        experiment_fn()
    else:
        print(f"Error: Unknown experiment ID {exp_id}")
        parser.print_help()
        exit(1)

if __name__ == "__main__":

    run()