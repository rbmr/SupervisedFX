from pathlib import Path
from datetime import datetime

# -- DIRECTORIES -- #

RQ2_DIR = Path(__file__).resolve().parent 

# -- END DIRECTORIES -- #

# -- PARAMETERS -- #

# - Data
RQ2_HYPERPARAMETERS_START_DATE = datetime(2022, 1, 2, 22, 0, 0, 0)
RQ2_HYPERPARAMETERS_END_DATE = datetime(2023, 6, 30, 20, 45, 0, 0)

RQ2_EXPERIMENTS_START_DATE = datetime(2023, 7, 2, 21, 0, 0)
RQ2_EXPERIMENTS_END_DATE = datetime(2024, 12, 31, 21, 45, 0, 0)

RQ2_DATA_SPLIT_RATIO = 0.7

# - Other
SEED = 42
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.0

# -- END PARAMETERS -- #