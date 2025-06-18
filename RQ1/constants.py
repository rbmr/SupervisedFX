import itertools
from pathlib import Path

from common.models.dummy_models import long_model, cash_model, short_model, dp_perfect_model, random_model

RQ1_DIR = Path(__file__).resolve().parent 
RQ1_EXPERIMENTS_DIR = RQ1_DIR / "experiments" / "report"
EXPERIMENT_NAME_FORMAT = "%Y%m%d-%H%M%S"
TENSORBOARD_DIR = RQ1_DIR / "tensorboard"

SPLIT_RATIO = 0.7
TRANSACTION_COST_PCT = 5 / 100_000
INITIAL_CAPITAL = 10_000
N_ACTIONS = 0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DUMMY_MODELS = {"long_model": long_model, "cash_model": cash_model, "short_model": short_model, "perfect_model": dp_perfect_model, "random_model": random_model}

SAC_HYPERPARAMS = dict(
    learning_rate=3.e-4,
    buffer_size=200_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=1.0,
    ent_coef='auto',
    gradient_steps=2,
    train_freq=48,
)

CUD_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
MARKERS = ["o", "v", "s", "*", "D", "P", "X"]

SEED = 42
SEEDS = itertools.count(SEED)