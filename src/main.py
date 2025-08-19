from src.data.main import get_data

from datetime import date

from src.constants import Timeframe
from src.data.models import CandleData
from src.features.feature_engineer import FeatureEngineer
from src.models.eval import run_and_analyze
from src.models.models import PerfectModel, RandomModel
from src.models.sl import DPSLModel
from src.scripts import find_first_valid_row
from src.trade.env import TradeEnv
from src.debug.log_config import setup_logging

def get_features() -> list[str]:
    time_features = ["sin_24h", "cos_24h", "cos_7d", "sin_7d"]
    technical_analysis = [
        "as_ratio_of_other_column('parabolic_sar(0.02, 0.2)', 'close_bid')",
        "as_ratio_of_other_column('ema(24)', 'close_bid')",
        "as_ratio_of_other_column('ema(72)', 'close_bid')",
        "as_min_max_fixed('adx(14)', 0, 100)",
        "as_min_max_fixed('rsi(14)', 0, 100)",
        "as_z_score('macd_hist(12, 26, 9)', 50)",
        "as_min_max_fixed('stoch_k(14)', 0, 100)",
        "as_ratio_of_other_column('atr(14)', 'close_bid')",
        "as_ratio_of_other_column('bb_upper(20, 2.0)', 'close_bid')",
        "as_ratio_of_other_column('bb_lower(20, 2.0)', 'close_bid')",
        "as_z_score(\"as_ratio_of_other_column('close_ask', 'close_bid')\", 50)"
    ]
    return [*time_features, *technical_analysis]

if __name__ == "__main__":

    setup_logging()

    candle_data = get_data(
        "DUKASCOPY",
        "EURUSD",
        Timeframe.H1,
        date(2020, 1, 1),
        date(2025, 1, 1)
    )
    prices = candle_data.to_array()

    feature_names = get_features()
    fe = FeatureEngineer(candle_data.to_dataframe())
    features = fe.get_all(feature_names)

    crop_idx = max(
        find_first_valid_row(features),
        find_first_valid_row(prices)
    )
    features = features[crop_idx:]
    prices = prices[crop_idx:]

    lookback = 16_384

    env = TradeEnv(
        prices = prices,
        features = features,
        feature_names = feature_names,
        t_start = lookback
    )

    run_and_analyze(DPSLModel(env), env)
    run_and_analyze(PerfectModel(env), env)
    run_and_analyze(RandomModel(env), env)
