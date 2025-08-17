from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import Timeframe, RUNS_DIR
from src.data.models import CandleData
from src.models.models import CustomModel, PerfectModel, RandomModel
from src.trade.env import TradeEnv

def run(model: CustomModel, env: TradeEnv, path: Path | None = None):
    if path is not None:
        if path.suffix != ".csv":
            raise ValueError(f"{path} is not a CSV file")
        path.parent.mkdir(parents=True, exist_ok=True)

    done = False
    obs, _ = env.reset()

    episode_log: list[dict[str, Any]] = [{
        "step": 0,
        "action": 0.0,
        "reward": 0.0,
        "done": False,
    }]

    while not done:
        action = model.predict(obs)

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_log.append({
            "step": env.t,
            "action": action.item(),
            "reward": reward,
            "done": done,
        })

    log_df = pd.DataFrame(episode_log)
    all_dfs = [log_df] + [df for df in info.values() if isinstance(df, pd.DataFrame)]
    final_df = pd.concat(all_dfs, axis=1)

    if path is not None:
        final_df.to_csv(path, index=False)
    return final_df

if __name__ == "__main__":

    candle_data = CandleData.load(
        "DUKASCOPY",
        "EURUSD",
        Timeframe.M30,
        date(2020,1,1),
        date(2025,1,1)
    )
    commission_pct = 0.0
    prices = candle_data.to_array()
    env = TradeEnv(
        prices = prices,
        features = np.empty((prices.shape[0], 0)),
        feature_names = [],
        commission_pct = commission_pct,
        initial_capital = 1.0,
        n_actions = 0
    )
    perfect_model = PerfectModel(env)
    run(perfect_model, env, RUNS_DIR / "perfect.csv")
    random_model = RandomModel(env)
    run(random_model, env, RUNS_DIR / "random.csv")
