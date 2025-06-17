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
    return int(-b + np.sqrt(D) / (2 * a))

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
    assert len(layers) >= 2, "params cannot be calculated without at least input and output layers."
    return sum(layers[1:])

def get_n_flops(*layers: int) -> int:
    """
    Computes the number of flops in the neural network.
    Layers = [inp, L1, ... Ln, out]
    """
    assert len(layers) >= 2, "params cannot be calculated without at least input and output layers."
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

