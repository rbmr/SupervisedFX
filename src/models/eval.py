from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.constants import RUNS_DIR
from src.models.analysis import analyze_individual_run
from src.models.models import CustomModel
from src.trade.env import TradeEnv


def run_and_analyze(model: CustomModel, env: TradeEnv):

    model_name = model.__class__.__name__
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{dt_str} {model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "log.parquet"
    run(model, env, run_log)
    analyze_individual_run(run_log, model_name)

def run(model: CustomModel, env: TradeEnv, path: Path | None = None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    done = False
    obs, _ = env.reset()

    episode_log: list[dict[str, Any]] = [{
        "step": 0,
        "action": 0.0,
        "reward": 0.0,
        "done": False,
    }]

    total_steps = env.episode_len - env.t_start - 1
    with tqdm(total=total_steps, desc="Running model", leave=True) as pbar:
        while not done:
            action = model.predict(obs)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_log.append({
                "step": env.t,
                "action": action.item(),
                "reward": reward,
                "done": done,
            })

            pbar.update(1)
    log_df = pd.DataFrame(episode_log)
    all_dfs = [log_df] + [df for df in info.values() if isinstance(df, pd.DataFrame)]
    final_df = pd.concat(all_dfs, axis=1)

    if path is not None:
        final_df.to_parquet(path, index=False)
    return final_df