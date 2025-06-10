from pathlib import Path

RQ1_DIR = Path(__file__).resolve().parent 
RQ1_EXPERIMENTS_DIR = RQ1_DIR / "experiments" / "testing"
EXPERIMENT_NAME_FORMAT = "%Y%m%d-%H%M%S"
RQ1_DP_CACHE_DIR = RQ1_DIR / "dp_cache"
TENSORBOARD_DIR = RQ1_DIR / "tensorboard"