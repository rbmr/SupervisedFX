from pathlib import Path
from datetime import datetime

# -- DIRECTORIES -- #

RQ2_DIR = Path(__file__).resolve().parent 

# -- END DIRECTORIES -- #

# -- PARAMETERS -- #

# - Data
RQ2_HYPERPARAMETERS_START_DATE = datetime(2022, 1, 2, 22, 0, 0, 0)
RQ2_HYPERPARAMETERS_END_DATE_15M = datetime(2023, 6, 30, 20, 45, 0, 0)
RQ2_HYPERPARAMETERS_END_DATE_30M = datetime(2023, 6, 30, 20, 30, 0, 0)
RQ2_HYPERPARAMETERS_END_DATE_1H = datetime(2023, 6, 30, 20, 0, 0, 0)

RQ2_EXPERIMENTS_START_DATE = datetime(2023, 7, 2, 21, 0, 0)
RQ2_EXPERIMENTS_END_DATE = datetime(2024, 12, 31, 21, 45, 0, 0)

RQ2_DATA_SPLIT_RATIO = 0.7
RQ2_VALTEST_SPLIT_RATIO = 0.5  # 50% of the validation set will be used for testing

# -- END PARAMETERS -- #