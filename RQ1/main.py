import os
os.environ['TPU_NAME'] = ''
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Avoid GPU detection if unneeded

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("Loading imports...")
from datetime import datetime
from pathlib import Path

from common.envs.callbacks import ActionHistogramCallback, SaveCallback, A2CRolloutLogger, SneakyLogger, \
    SACMetricsLogger, BasicCallback
from common.models.train_eval import (analyse_results, evaluate_models, train_model)
from common.models.utils import save_model_with_metadata
from common.scripts import has_nonempty_subdir, n_children, picker
from RQ1.constants import EXPERIMENT_NAME_FORMAT, RQ1_EXPERIMENTS_DIR, TENSORBOARD_DIR
from RQ1.parameters import get_train_env, get_eval_envs, get_train_model, cleanup_tensorboard

logging.info("Done.")

def train():

    experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
    tensorboard_log = TENSORBOARD_DIR / experiment_name

    train_env = get_train_env()

    model = get_train_model(train_env, tb_log=tensorboard_log)

    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_name
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveCallback(models_dir, save_freq=train_env.episode_len),
                ActionHistogramCallback(train_env, log_freq=train_env.episode_len),
                SACMetricsLogger(verbose=1, log_freq=500),
                BasicCallback(verbose=1, log_freq=500)]
    train_model(model, train_env, train_episodes=50, callback=callback)
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

    eval_envs = get_eval_envs()

    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1, num_workers=3)

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
    cleanup_tensorboard()
    options = [
        ("train", train),
        ("eval", lambda: evaluate(RQ1_EXPERIMENTS_DIR, 10)),
        ("analyze", lambda: analyze(RQ1_EXPERIMENTS_DIR, 10)),
    ]
    picker(options, default=None)()

