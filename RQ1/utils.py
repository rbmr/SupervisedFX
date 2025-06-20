import logging
import os
import shutil
from pathlib import Path

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from RQ1.constants import TENSORBOARD_DIR
from common.envs.callbacks import SaveCallback, ActionHistogramCallback
from common.envs.forex_env import ForexEnv
from common.models.train_eval import train_model, evaluate_models, analyse_results

def get_shapes(inp: int, out: int, level_low=32, level_high=80):
    """
    Given some network shapes, and constraints on these shapes, determines the division
    of neurons across these layers such that the difference in number of parameters is minimized.
    Works in polynomial time, really fast (less than 2 seconds).
    """

    logging.info("Setting up valid configurations")

    # Hard constraints:
    # (1) No layer has an absurd size: level_low <= x, a, (a*b), c, (c*d), (c*d*d), e, (e*f), (e*f*f) <= level_high
    # (2) The difference in shape is significant: 1.5 <= b, d, f <= 2.0

    fac_range = np.linspace(1.5, 2.0, 100)

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

def get_width(inp: int, out: int, n_layers: int, n_params: float):
    """
    Gets the positive solution for w:
    get_n_params(inp, [w] * n_layers, out) = n_params
    inp * w + (n_layers - 1) * w * w + w * out + n_layers * w + 1 = n_params
    (n_layers - 1) * w^2 + (inp + out + n_layers) * w + (1-n_params) = 0
    """
    assert inp >= 0
    assert out >= 0
    assert n_layers >= 1
    assert n_params >= 1.0
    a = (n_layers - 1)
    b = inp + out + n_layers
    c = (1 - n_params)
    D = b * b - 4 * a * c
    if D < 0:
        raise ValueError('No solution')
    return int( (-b + np.sqrt(D)) / (2 * a) )

def get_widths(inp: int, out: int, n_layers: int, n_params: int, division: float):
    """
    Compute the number of neurons per layer for two fully connected networks.
    """
    w1 = get_width(inp, out, n_layers, n_params * division)
    w2 = get_width(inp, out, n_layers, n_params * (1-division))
    return w1, w2

def get_n_params(*layers: int) -> int:
    """
    Computes the number of parameters in the neural network.
    Layers = [inp, L1, ... Ln, out]
    """
    assert len(layers) >= 2, "params cannot be calculated without at least input and output layers."
    network = layers[1:]
    return np.dot(layers[:-1], network) + np.sum(network)

def get_n_neurons(*layers: int) -> int:
    """
    Computes the number of neurons in the neural network.
    Layers = [inp, L1, ... Ln, out]
    """
    assert len(layers) >= 2, "neurons cannot be calculated without at least input and output layers."
    return sum(layers[1:])

def get_n_flops(*layers: int) -> int:
    """
    Computes the number of flops in the neural network.
    Layers = [inp, L1, ... Ln, out]
    """
    assert len(layers) >= 2, "flops cannot be calculated without at least input and output layers."
    layers = np.array(layers)
    return np.sum(2 * layers[:-1] * layers[1:] + layers[1:])

def train_eval_analyze(experiment_dir: Path, model: BaseAlgorithm, train_env: ForexEnv, eval_env: ForexEnv):

    # Setup directories
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"

    # Train model
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

        callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
                    ActionHistogramCallback(train_env, log_freq=train_env.episode_len)]

        train_model(model, train_env, train_episodes=50, callback=callback)

    # Evaluate resulting models

    train_env.reset()
    eval_env.reset()

    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1, num_workers=3)

    # Analyze results

    analyse_results(results_dir)

def cleanup_tensorboard():

    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    experiments = TENSORBOARD_DIR.iterdir()
    experiments = list(filter(lambda f: not f.name.startswith("_") and f.is_dir(), experiments))
    n_experiments = len(experiments)
    max_experiments = 8

    logging.info(f"Found {n_experiments} experiments in the tensorboard directory (max {max_experiments})")

    experiments.sort(key= lambda x: x.stat().st_mtime)
    if n_experiments > max_experiments:
        n_rem_experiments = n_experiments - max_experiments
        logging.info(f"Cleanup: deleting {n_rem_experiments} oldest experiments.")
        old_experiments = experiments[:n_rem_experiments]
        for experiment in old_experiments:
            shutil.rmtree(experiment)
        logging.info(f"Removed {n_rem_experiments} old experiments")

