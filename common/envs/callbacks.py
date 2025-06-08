import logging
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from common.constants import AgentDataCol
from common.envs.forex_env import ForexEnv
from common.models.utils import save_model_with_metadata
from common.scripts import circ_slice, render_horz_bar


class SaveCallback(BaseCallback):
    def __init__(self, models_dir: Path, save_freq: int, verbose=0):
        super().__init__(verbose)
        if models_dir.exists() and not models_dir.is_dir():
            raise ValueError(f"{models_dir} is not a valid directory.")
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq != 0:
            return True
        filename = self.models_dir / f"model_{self.num_timesteps}_steps.zip"
        save_model_with_metadata(self.model, filename)
        if self.verbose > 0:
            logging.info(f"Saved model at timestep {self.num_timesteps} to {filename}")
        return True

class SaveOnEpisodeEndCallback(BaseCallback):
    def __init__(self, models_dir: Path, verbose=0):
        super().__init__(verbose)
        if models_dir.exists() and not models_dir.is_dir():
            raise ValueError(f"{models_dir} is not a valid directory.")
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.episode_num = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        done_array = self.locals.get("dones")
        if done_array is None:
            return True
        if not any(done_array):
            return True
        self.episode_num += 1
        filename = self.models_dir / f"model_{self.episode_num}_episodes.zip"
        save_model_with_metadata(self.model, filename)
        if self.verbose > 0:
            logging.info(f"Saved model at episode {self.episode_num} to {filename}")
        return True

class ActionHistogramCallback(BaseCallback):
    """
    Logs a histogram of actions taken during training, every `log_freq` steps.
    """
    def __init__(self, env: ForexEnv, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self.bins = 11 if env.n_actions == 0 or env.n_actions > 5 else env.n_actions * 2 + 1
        self.max_height = 40

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        start = self.num_timesteps - self.log_freq
        end = self.num_timesteps
        actions = circ_slice(self.env.agent_data[:, AgentDataCol.action], start, end)
        hist, bin_edges = np.histogram(actions, bins=self.bins)
        max_count = hist.max()
        if max_count == 0:
            return True
        bar_heights = hist / max_count * self.max_height
        logging.info(f"Histogram of actions taken in the past {self.log_freq} steps.")
        for height, count, bin_start, bin_end in zip(bar_heights, hist, bin_edges[:-1], bin_edges[1:]):
            label = f"({bin_start:>5.2f})–({bin_end:>5.2f})"
            bar = f"{render_horz_bar(height)} ({count})"
            logging.info(f"{label}: {bar}")
        return True
