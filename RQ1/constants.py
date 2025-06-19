from pathlib import Path

import torch

from common.models.dummy_models import long_model, cash_model, short_model, dp_perfect_model, random_model

RQ1_DIR = Path(__file__).resolve().parent 
RQ1_EXPERIMENTS_DIR = RQ1_DIR / "experiments" / "report"
EXPERIMENT_NAME_FORMAT = "%Y%m%d-%H%M%S"
TENSORBOARD_DIR = RQ1_DIR / "tensorboard"

SPLIT_RATIO = 0.7
DUMMY_MODELS = {"long_model": long_model, "cash_model": cash_model, "short_model": short_model, "perfect_model": dp_perfect_model, "random_model": random_model}

CUD_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
MARKERS = ["o", "v", "s", "*", "D", "P", "X"]

SEED = 42