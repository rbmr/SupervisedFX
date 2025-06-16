import argparse
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, List

import numpy as np
from stable_baselines3 import SAC
from torch import nn

from RQ1.constants import RQ1_EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT, SAC_HYPERPARAMS, SPLIT_RATIO, FOREX_CANDLE_DATA, \
    ACTION_HIGH, ACTION_LOW, N_ACTIONS, TRANSACTION_COST_PCT, INITIAL_CAPITAL, CUD_COLORS, MARKERS, TENSORBOARD_DIR, \
    DUMMY_MODELS, SEED, SEEDS
from common.data.feature_engineer import FeatureEngineer, complex_24h, complex_7d, parabolic_sar, \
    as_ratio_of_other_column, history_lookback, as_z_score, remove_columns, macd, bollinger_bands, vwap, mfi, \
    as_min_max_fixed, chaikin_volatility
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_current_exposure
from common.envs.callbacks import SaveCallback, ActionHistogramCallback
from common.envs.forex_env import ForexEnv, DataConfig, ActionConfig, EnvConfig, ObsConfig
from common.models.train_eval import combine_finals, evaluate_models, analyse_results, train_model, evaluate_dummy


def get_shapes():

    inp = 32
    out = 1

    def n_params(p: int, q: int, r: int):
        """
        Computes the number of parameters in the neural network,
        given global input size and output size and 3 fully connected layers.
        """
        return (inp * p) + (p * q) + (q * r) + (r * out) + p + q + r + out

    # Constraints
    # 64 <= x <= 128
    # 32 <= a, (a*b), c, (c*d), (c*d*d), e, (e*f), (e*f*f) <= 128
    # 1.25 <= b, d, f <= 2.0

    # objective function: minimize (max n_params - min n_params) for these shapes:
    # "flat" : [x, x, x],
    # "diamond" : [a, a*b, a],
    # "inv": [c, c*d, c*d*d],
    # "funnel": [e*f*f, e*f, e]

    fac_range = np.linspace(1.5, 2.0, 100)
    x_low = 16
    x_high = 64
    level_low = 16
    level_high = 64

    print("Generating configurations")

    flat = [(x, n_params(x, x, x)) for x in range(x_low, x_high + 1)]
    diamond = [(a, b, n_params(a, int(a * b), a)) for b in fac_range for a in range(level_low, int(level_high / b) + 1)]
    inv = [(c, d, n_params(c, int(c * d), int(c * d * d))) for d in fac_range for c in
           range(level_low, int(level_high / (d * d)) + 1)]
    funnel = [(e, f, n_params(int(e * f * f), int(e * f), e)) for f in fac_range for e in
              range(level_low, int(level_high / (f * f)) + 1)]

    print("Sorting configurations")

    flat.sort(key=lambda x: x[1])
    diamond.sort(key=lambda x: x[2])
    inv.sort(key=lambda x: x[2])
    funnel.sort(key=lambda x: x[2])

    print("Finding optimal combination")

    i, j, k, l = 0, 0, 0, 0

    min_diff = float("inf")
    solution = None

    while (i < len(flat) and
           j < len(diamond) and
           k < len(inv) and
           l < len(funnel)):

        params = (flat[i][1], diamond[j][2], inv[k][2], funnel[l][2])
        current_max = max(params)
        current_min = min(params)
        diff = current_max - current_min

        if diff < min_diff:
            min_diff = diff
            solution = (i, j, k, l)

        if current_min == flat[i][1]:
            i += 1
        elif current_min == diamond[j][2]:
            j += 1
        elif current_min == inv[k][2]:
            k += 1
        else:
            l += 1

    assert solution is not None

    i, j, k, l = solution
    x, flat_params = flat[i]
    a, b, diamond_params = diamond[j]
    c, d, inv_params = inv[k]
    e, f, funnel_params = funnel[l]

    shapes = {
        "flat" : [x, x, x],
        "diamond": [a, int(a * b), a],
        "inv_funnel": [c, int(c * d), int(c * d * d)],
        "funnel":  [int(e * f * f), int(e * f), e],
    }

    print(shapes)
    print("flat", flat_params)
    print("diamond", diamond_params)
    print("inv_funnel", inv_params)
    print("funnel", funnel_params)

    return shapes

@dataclass(frozen=True)
class ExperimentConfig:

    name: str = field(default_factory=lambda: datetime.now().strftime(EXPERIMENT_NAME_FORMAT))
    net_shape: list = field(default_factory=lambda: [64, 64])
    activation_fn: Callable = nn.ReLU
    lookback: int = 3

    line_color: str = "black"
    line_style: str = "-"
    line_marker: Optional[str] = None

    def get_style(self):
        return {
            "color" : self.line_color,
            "linestyle" : self.line_style,
            "marker": self.line_marker,
        }

def get_envs(config: Optional[ExperimentConfig] = None):

    if config is None:
        config = ExperimentConfig()

    # Setup feature engineers
    fe = FeatureEngineer()
    fe.add(complex_24h) # 2 features
    fe.add(complex_7d) # 2 features

    def _trend(df):

        parabolic_sar(df)
        as_ratio_of_other_column(df, 'sar', 'close_bid')

        vwap(df, window=4)
        vwap(df, window=12)
        vwap(df, window=48)
        as_ratio_of_other_column(df, 'vwap_12', 'close_bid')

        history_lookback(df, config.lookback, ["sar", "vwap_4", "vwap_12", "vwap_48"])

    fe.add(_trend) # 4 * lookback

    def _momentum(df):

        macd(df, short_window=12, long_window=26, signal_window=9)
        remove_columns(df, ["macd_signal", "macd"])
        as_z_score(df, 'macd_hist', window=50)

        mfi(df, window=14)
        as_min_max_fixed(df, "mfi_14", min=0, max=100)

        history_lookback(df, config.lookback, ["macd_hist", "mfi_14"])

    fe.add(_momentum) # 2 * lookback

    def _volatility(df):

        bollinger_bands(df, window=20, num_std_dev=2)
        as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
        as_ratio_of_other_column(df, "bb_lower_20", "close_bid")

        chaikin_volatility(df, ema_window=10, roc_period=10)
        #chaikin_vol_{ema_window}_{roc_period}
        as_z_score(df, "chaikin_vol_10_10", window=50)

        history_lookback(df, config.lookback, ["bb_upper_20", "bb_lower_20", "chaikin_vol_10_10"])

    fe.add(_volatility) # 3 * lookback

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
    train_config, eval_config = DataConfig.get_configs(
        forex_candle_data=FOREX_CANDLE_DATA,
        split_ratio=SPLIT_RATIO,
        obs_configs=obs_configs,
    )
    train_env = ForexEnv(action_config, env_config, train_config)
    eval_env = ForexEnv(action_config, env_config, eval_config)
    return train_env, eval_env

def run_experiment(experiment_group: str, config: ExperimentConfig, seed: int = SEED):

    # Set seeds
    np.random.seed(SEED)
    random.seed(SEED)
    # The environments are entirely deterministic, no seeds need to be set.

    # Get environments

    train_env, eval_env = get_envs(config)

    # Setup directories

    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_group / config.name / f"seed_{seed}"
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"

    # Train model

    if not models_dir.exists():

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            **SAC_HYPERPARAMS,
            policy_kwargs=dict(
                activation_fn=config.activation_fn,
                net_arch=dict(pi=config.net_shape, qf=config.net_shape)
            ),
            verbose=0,
            tensorboard_log=TENSORBOARD_DIR / config.name,
            device="cpu",
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

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1, num_workers=3)

    # Analyze results

    analyse_results(results_dir)

def run_experiments(experiment_group: str, experiments: List[ExperimentConfig]):

    for experiment, seed in experiments, SEEDS:
        logging.info(f"Running experiment: {experiment}")
        run_experiment(experiment_group=experiment_group, config=experiment, seed=seed)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")

def run_baselines():

    train_env, eval_env = get_envs()

    experiment_group = "baselines"
    experiment_group_dir = RQ1_EXPERIMENTS_DIR / experiment_group

    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }
    for dummy_factory in DUMMY_MODELS:
        name = dummy_factory.__name__
        for eval_env_name, eval_env in eval_envs.items():
            dummy_model = dummy_factory(eval_env)
            results_dir = experiment_group_dir / name
            evaluate_dummy(dummy_model = dummy_model, name=name, results_dir=results_dir, eval_env=eval_env, eval_env_name=eval_env_name)

def run_shape_experiments():

    shapes = get_shapes()

    experiments = [
        ExperimentConfig(name="shape_flat", net_shape=shapes["flat"], line_color=CUD_COLORS[0], line_marker="s"),
        ExperimentConfig(name="shape_diamond", net_shape=shapes["diamond"], line_color=CUD_COLORS[1], line_marker="D"),
        ExperimentConfig(name="shape_funnel", net_shape=shapes["funnel"], line_color=CUD_COLORS[2], line_marker=">"),
        ExperimentConfig(name="shape_inv_funnel", net_shape=shapes["inv_funnel"], line_color=CUD_COLORS[3], line_marker="<"),
    ]

    run_experiments(experiment_group="network_shapes", experiments=experiments)

def run_size_experiments():

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

    experiments = []

    for activation_fn, color, marker in zip([nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.SiLU, nn.Tanh, nn.ELU], CUD_COLORS[:6], MARKERS[:6]):
        experiments.append(ExperimentConfig(
            name = activation_fn.__name__,
            line_marker = marker,
            line_color = color,
        ))

    run_experiments(experiment_group=f"activation_functions", experiments=experiments)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int, help="all (0), shapes (1), sizes (2), activations (3), baselines (4)")
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
