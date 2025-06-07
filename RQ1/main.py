import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Loading imports...")
from datetime import datetime
from pathlib import Path

from common.envs.callbacks import ActionHistogramCallback, SaveCallback
from common.models.train_eval import (analyse_results, evaluate_models,
                                      train_model)
from common.models.utils import save_model_with_metadata
from common.scripts import has_nonempty_subdir, n_children, picker
from RQ1.constants import EXPERIMENT_NAME_FORMAT, RQ1_EXPERIMENTS_DIR
from RQ1.parameters import get_environments, get_model

logging.info("Done.")

def train():

    train_env, _ = get_environments(shuffled=True)
    save_freq = 20_000

    model = get_model(train_env)

    experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_name
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveCallback(models_dir, save_freq=save_freq),
                ActionHistogramCallback(train_env, log_freq=save_freq)]
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
        ("eval", lambda: evaluate(RQ1_EXPERIMENTS_DIR, 10)),
        ("analyze", lambda: analyze(RQ1_EXPERIMENTS_DIR, 10)),
    ]
    picker(options, default=None)()

