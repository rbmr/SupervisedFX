from datetime import datetime
from pathlib import Path

from common.data.data import ForexCandleData, Timeframe

RQ1_DIR = Path(__file__).resolve().parent 
RQ1_EXPERIMENTS_DIR = RQ1_DIR / "experiments" / "report"
EXPERIMENT_NAME_FORMAT = "%Y%m%d-%H%M%S"
TENSORBOARD_DIR = RQ1_DIR / "tensorboard"

SPLIT_RATIO = 0.7
TRANSACTION_COST_PCT = 10 / 100_000
INITIAL_CAPITAL = 10_000
N_ACTIONS = 0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
FOREX_CANDLE_DATA = ForexCandleData.load(
    source="dukascopy",
    instrument="EURUSD",
    granularity=Timeframe.M30,
    start_time=datetime(2020, 1, 1, 22, 0, 0, 0),
    end_time=datetime(2024, 12, 31, 21, 30, 0, 0),
)