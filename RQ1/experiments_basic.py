import argparse
import itertools
import json
import logging
from asyncio import FIRST_EXCEPTION
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial

from common.envs.dp import DPTable, get_dp_table_from_env
from common.envs.rewards import DPRewardFunction
from common.scripts import parallel_apply

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s, %(levelname)s] %(message)s' )

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
from stable_baselines3 import SAC
from torch import nn
import torch

from RQ1.constants import RQ1_EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT, SAC_HYPERPARAMS, SPLIT_RATIO, FOREX_CANDLE_DATA, \
    ACTION_HIGH, ACTION_LOW, N_ACTIONS, TRANSACTION_COST_PCT, INITIAL_CAPITAL, CUD_COLORS, MARKERS, DUMMY_MODELS, SEED
from RQ1.scripts import get_n_params, get_n_neurons, get_n_flops, get_widths
from common.data.feature_engineer import FeatureEngineer, complex_24h, complex_7d, parabolic_sar, \
    as_ratio_of_other_column, history_lookback, as_z_score, remove_columns, macd, bollinger_bands, vwap, mfi, \
    as_min_max_fixed, chaikin_volatility
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_current_exposure
from common.envs.callbacks import SaveCallback, ActionHistogramCallback
from common.envs.forex_env import ForexEnv, DataConfig, ActionConfig, EnvConfig, ObsConfig
from common.models.train_eval import combine_finals, evaluate_models, analyse_results, train_model, evaluate_dummy

def get_shapes():
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

    inp = 32
    out = 1

    shape_arch_params_pairs = {}
    for s in shapes:
        layers_fn = shape_layers_fn[s]
        configs = shape_configs[s]
        archs = [layers_fn(*config) for config in configs]
        arch_params_pairs = [(arch, get_n_params(inp, *arch, out)) for arch in archs]
        shape_arch_params_pairs[s] = arch_params_pairs

    del shape_layers_fn, shape_configs

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

@dataclass
class ExperimentConfig:

    name: str = field(default_factory=lambda: datetime.now().strftime(EXPERIMENT_NAME_FORMAT))
    net_shape: list[int] = field(default_factory=lambda: [64, 64])
    actor_shape: Optional[list[int]] = None
    critic_shape: Optional[list[int]] = None
    activation_fn: Callable = nn.ReLU
    lookback: int = 3

    line_color: str = "black"
    line_style: str = "-"
    line_marker: Optional[str] = None

    def __post_init__(self):
        if self.actor_shape is None:
            self.actor_shape = self.net_shape
        if self.critic_shape is None:
            self.critic_shape = self.net_shape

    def get_style(self):
        return {
            "color" : self.line_color,
            "linestyle" : self.line_style,
            "marker": self.line_marker,
        }

    def log_info(self, experiment_dir: Path):
        """Logs some information about this experiment to a JSON file inside experiment_dir."""
        info = dict(
            name = self.name,
            actor_shape = self.actor_shape,
            critic_shape = self.critic_shape,
            activation_fn = self.activation_fn.__name__,
            lookback = self.lookback,
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4) # type: ignore

def add_technical_analysis(df, lookback: int):

    # Trend

    parabolic_sar(df)
    as_ratio_of_other_column(df, 'sar', 'close_bid')

    vwap(df, window=4)
    vwap(df, window=12)
    vwap(df, window=48)
    as_ratio_of_other_column(df, 'vwap_4', 'close_bid')
    as_ratio_of_other_column(df, 'vwap_12', 'close_bid')
    as_ratio_of_other_column(df, 'vwap_48', 'close_bid')

    history_lookback(df, lookback, ["sar", "vwap_4", "vwap_12", "vwap_48"])

    # Momentum

    macd(df, short_window=12, long_window=26, signal_window=9)
    remove_columns(df, ["macd_signal", "macd"])
    as_z_score(df, 'macd_hist', window=50)

    mfi(df, window=14)
    as_min_max_fixed(df, "mfi_14", min=0, max=100)

    history_lookback(df, lookback, ["macd_hist", "mfi_14"])

    # Technical Analysis

    bollinger_bands(df, window=20, num_std_dev=2)
    as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
    as_ratio_of_other_column(df, "bb_lower_20", "close_bid")

    chaikin_volatility(df, ema_window=10, roc_period=10) # adds chaikin_vol_{ema_window}_{roc_period}
    as_z_score(df, "chaikin_vol_10_10", window=50)

    history_lookback(df, lookback, ["bb_upper_20", "bb_lower_20", "chaikin_vol_10_10"])

def get_envs(config: Optional[ExperimentConfig] = None):
    """
    Sets up the environment. The environment is the same for all models.
    Lookback parameter is not used currently. Always remains equal to 3.
    """

    if config is None:
        config = ExperimentConfig()

    # Setup feature engineers
    fe = FeatureEngineer()
    fe.add(complex_24h) # 2 features
    fe.add(complex_7d) # 2 features
    fe.add(partial(add_technical_analysis, lookback=config.lookback)) # 4 * lookback

    sfe = StepwiseFeatureEngineer()
    sfe.add(["current_exposure"], calculate_current_exposure) # 1 feature

    # Setup environments

    obs_configs = [
        ObsConfig(
            name='market_features',
            fe=fe,
            sfe=sfe,
            window=1
        ),
    ]
    env_config = EnvConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        reward_function=None,
    )
    action_config = ActionConfig(
        n=N_ACTIONS,
        low=ACTION_LOW,
        high=ACTION_HIGH,
    )
    train_config, eval_config = DataConfig.from_splits(
        forex_candle_data=FOREX_CANDLE_DATA,
        split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
        obs_configs=obs_configs,
    )
    train_env = ForexEnv(action_config, env_config, train_config)
    train_env.custom_reward_fn = DPRewardFunction(get_dp_table_from_env(train_env))

    eval_env = ForexEnv(action_config, env_config, eval_config)
    eval_env.custom_reward_fn = DPRewardFunction(get_dp_table_from_env(eval_env))

    return train_env, eval_env

def run_experiment(experiment_group: str, config: ExperimentConfig, seed: int = SEED):
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

    train_env, eval_env = get_envs(config)

    # Setup directories

    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_group / config.name / f"seed_{seed}"
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"
    config.log_info(experiment_dir) # log some info about this experiment

    # Train model

    if not models_dir.exists():

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            **SAC_HYPERPARAMS,
            policy_kwargs=dict(
                activation_fn=config.activation_fn,
                net_arch=dict(pi=config.actor_shape, qf=config.critic_shape)
            ),
            verbose=0,
            device="cpu", # cpu is faster than cuda for SAC.
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

def run_experiment_wrapper(config_seed: tuple[ExperimentConfig, int], experiment_group: str):
    config, seed = config_seed
    run_experiment(experiment_group, config, seed)

def run_experiments(experiment_group: str, experiments: List[ExperimentConfig], n_seeds=1, n_workers=3, add_timestamp: bool=True):
    """
    Runs each of the experiments for a number of seeds.
    """
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_group = f"{timestamp}_{experiment_group}"

    func = partial(run_experiment_wrapper, experiment_group=experiment_group)
    inputs = list(itertools.product(experiments, range(SEED, SEED + n_seeds)))
    parallel_apply(func, inputs, num_workers=n_workers)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")

def run_baselines():
    """
    Runs the baseline models on the train and eval environments. Used as reference.
    """
    train_env, eval_env = get_envs()

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
    Number of parameters is roughly equal across networks.
    """
    shapes = get_shapes()

    experiments = [
        ExperimentConfig(name="shape_flat", net_shape=shapes["flat"], line_color=CUD_COLORS[0], line_marker="s"),
        ExperimentConfig(name="shape_diamond", net_shape=shapes["diamond"], line_color=CUD_COLORS[1], line_marker="D"),
        ExperimentConfig(name="shape_funnel", net_shape=shapes["funnel"], line_color=CUD_COLORS[2], line_marker=">"),
        ExperimentConfig(name="shape_inv_funnel", net_shape=shapes["inv_funnel"], line_color=CUD_COLORS[3], line_marker="<"),
        ExperimentConfig(name="shape_shallow", net_shape=shapes["shallow"], line_color=CUD_COLORS[4], line_marker="*"),
    ]

    run_experiments(experiment_group="network_shapes", experiments=experiments, n_seeds=5)

def run_division_experiments():
    """
    Determine the impact of dividing parameters over the actor and the critic networks.
    Number of parameters remains equal.
    """

    # Setup
    inp = 32
    out = 1
    n_layers = 2
    base_net = [64, 64]
    total_params = get_n_params(inp, *base_net, out) * 2

    # Adding experiments
    experiments = [ExperimentConfig(
        name="no_bias",
        net_shape=base_net,
    )]

    # Slight bias
    w1, w2 = get_widths(inp, out, n_layers, total_params, 0.60)
    experiments.append(ExperimentConfig(
        name = "moderate_actor_bias",
        actor_shape = [w1] * n_layers,
        critic_shape = [w2] * n_layers,
    ))
    experiments.append(ExperimentConfig(
        name = "moderate_critic_bias",
        actor_shape = [w2] * n_layers,
        critic_shape = [w1] * n_layers,
    ))

    # Large bias
    w1, w2 = get_widths(inp, out, n_layers, total_params, 0.75)
    experiments.append(ExperimentConfig(
        name = "large_actor_bias",
        actor_shape = [w1] * n_layers,
        critic_shape = [w2] * n_layers,
    ))
    experiments.append(ExperimentConfig(
        name = "large_critic_bias",
        actor_shape = [w2] * n_layers,
        critic_shape = [w1] * n_layers,
    ))

    run_experiments(experiment_group="network_division", experiments=experiments, n_seeds=5)

def run_size_experiments():
    """
    Determine the impact of network size on model performance.
    We modify two aspects: Model depth (number of layers), and Model width (number of neurons per layer).
    """
    for depth_name, depth in zip(["shallow", "moderate", "deep", "very_deep"], [1, 2, 3, 4]):

        experiments = []

        for width_name, width, color, marker in zip(["narrow", "moderate", "wide", "very_wide"], [16, 32, 64, 128], CUD_COLORS[:4], MARKERS[:4]):
            experiments.append(ExperimentConfig(
                name = f"{depth_name}-{width_name}",
                net_shape = [width] * depth,
                line_marker = marker,
                line_color = color,
            ))

        run_experiments(experiment_group=f"{depth_name}_networks", experiments=experiments)

def run_activation_experiments():
    """
    Determine the impact of network activation functions on model performance.
    All networks are the same size and shape.
    """
    experiments = []

    for activation_fn, color, marker in zip([nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.SiLU, nn.Tanh, nn.ELU], CUD_COLORS[:6], MARKERS[:6]):
        experiments.append(ExperimentConfig(
            name = activation_fn.__name__,
            activation_fn = activation_fn,
            line_marker = marker,
            line_color = color,
        ))

    run_experiments(experiment_group=f"activation_functions", experiments=experiments, n_seeds=5)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", default=0, type=int, help="all (0), shapes (1), sizes (2), activations (3), baselines (4), division (5)")
    args = parser.parse_args()
    experiment_id = args.experiment

    if experiment_id == 1 or experiment_id == 0:
        run_shape_experiments()
    if experiment_id == 2 or experiment_id == 0:
        run_size_experiments()
    if experiment_id == 3 or experiment_id == 0:
        run_activation_experiments()
    if experiment_id == 4 or experiment_id == 0:
        run_baselines()
    if experiment_id == 5 or experiment_id == 0:
        run_division_experiments()