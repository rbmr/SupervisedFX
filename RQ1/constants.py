from pathlib import Path

RQ1_DIR = Path(__file__).resolve().parent 
RQ1_EXPERIMENTS_DIR = RQ1_DIR / "experiments" / "report"
EXPERIMENT_NAME_FORMAT = "%Y%m%d-%H%M%S"
TENSORBOARD_DIR = RQ1_DIR / "tensorboard"