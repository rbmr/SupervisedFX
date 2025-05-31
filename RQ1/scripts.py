import logging
from pathlib import Path
from typing import List, Dict, Any

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from common.envs.forex_env import ForexEnv


def run_model_on_vec_env(model: BaseAlgorithm, env: ForexEnv, data_path: Path, total_steps: int, deterministic: bool,
                         progress_bar: bool = True) -> None:
    """
    Run a trained RL model on a vectorized environment for a number of episodes,
    log each step, and write all logs at the end. Optionally displays a progress bar.
    """

    if total_steps <= 0:
        logging.warning("Total steps must be greater than 0. No steps will be executed.")
        return

    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    collected_log_entries: List[Dict[str, Any]] = []

    env = DummyVecEnv([lambda: env])  # Ensure env is wrapped in DummyVecEnv

    obs = env.reset()

    pbar = None
    if progress_bar:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_steps, desc="Total Steps")
        except ImportError:
            print("Warning: tqdm is not installed. Progress bar will not be shown. "
                  "Install with: pip install tqdm")

    step_count = 1
    while step_count < total_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, rewards, dones, infos = env.step(action)

        for i in range(env.num_envs):
            log_entry: Dict[str, Any] = {
                "env_index": i,
                "step": step_count,
                "action": action[i].tolist(),
                "reward": rewards[i],
                "done": dones[i],
                "info": infos[i],
            }
            collected_log_entries.append(log_entry)

        if pbar:
            pbar.update(1)

        # If done, break out of the while loop
        if any(dones):
            if pbar:
                pbar.set_description("Episode completed")
            # Reset the environment
            obs = env.reset()

        obs = next_obs
        step_count += 1

    if pbar:
        pbar.close()

    # Save collected logs to JSON file
    # flatten each dictionary in collected_log_entries to be only one level deep.
    flat_log_entries = [flatten_dict(entry) for entry in collected_log_entries]
    log_df = pd.DataFrame(flat_log_entries)
    log_df.to_csv(data_path.with_suffix('.csv'), index=False)