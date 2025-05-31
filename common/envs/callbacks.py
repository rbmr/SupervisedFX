from pathlib import Path

import logging
from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveOnEpisodeEndCallback(BaseCallback):
    def __init__(self, save_path: Path, verbose=0):
        super().__init__(verbose)
        if save_path.exists() and not save_path.is_dir():
            raise ValueError(f"{save_path} is not a valid directory.")
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.episode_num = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        done_array = self.locals.get("dones")
        if done_array is not None and any(done_array):
            self.episode_num += 1
            filename = self.save_path / f"model_{self.episode_num}_episodes.zip"
            self.model.save(filename)
            if self.verbose > 0:
                logging.info(f"Saved model at episode {self.episode_num} to {filename}")
        return True
