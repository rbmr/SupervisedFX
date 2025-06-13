import logging
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC
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
        self.bins = 11 if env.n_actions == 0 or env.n_actions > 11 else env.n_actions
        self.max_height = 40

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        start = self.num_timesteps - self.log_freq
        end = self.num_timesteps
        actions = circ_slice(self.env.agent_data[:, AgentDataCol.target_exposure], start, end)
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

class SneakyLogger(BaseCallback):
    """
    Adds a buffer to the training environment of which all values are logged at the end of each rollout.
    """

    def _on_rollout_start(self):
        raw_env = self.training_env.envs[0] #type: ignore
        raw_env.sneaky_buffer = {}

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        raw_env = self.training_env.envs[0] #type: ignore
        for key, values in raw_env.sneaky_buffer.items():
            if len(values) > 1:
                self.logger.record(f"{key}_mean", np.mean(values))
                self.logger.record(f"{key}_std", np.std(values))
        raw_env.sneaky_buffer.clear()

class A2CRolloutLogger(BaseCallback):
    """
    Logs all the rollout values that are collected by default.
    """
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        # Access the rollout buffer
        buffer = self.model.rollout_buffer

        self.logger.record("rollout/adv_mean", np.mean(buffer.advantages))
        self.logger.record("rollout/adv_std", np.std(buffer.advantages))

        self.logger.record("rollout/return_mean", np.mean(buffer.returns))
        self.logger.record("rollout/return_std", np.std(buffer.returns))

        self.logger.record("rollout/value_mean", np.mean(buffer.values))
        self.logger.record("rollout/value_std", np.std(buffer.values))

        self.logger.record("rollout/log_prob_mean", np.mean(buffer.log_probs))
        self.logger.record("rollout/log_prob_std", np.std(buffer.log_probs))

class BasicCallback(BaseCallback):

    def __init__(self, verbose: int = 0, log_freq: int = 5_000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._reset()

    def _on_step(self) -> bool:
        self._collect()
        if self.num_timesteps % self.log_freq == 0:
            self._log()
            self._reset()
        return True

    def _collect(self):
        self.rewards.extend(self.locals["rewards"])
        self.actions.extend(self.locals['actions'])

    def _log(self):
        self.logger.record("custom/reward_mean", np.mean(self.rewards))
        self.logger.record("custom/reward_std", np.std(self.rewards))
        self.logger.record("custom/action_mean", np.mean(self.actions))
        self.logger.record("custom/action_std", np.std(self.actions))
        self.logger.dump()

    def _reset(self) -> None:
        self.rewards = []
        self.actions = []

class SACMetricsLogger(BaseCallback):
    """
    Custom callback to log metrics for the SAC algorithm.
    """
    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_log_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_timestep >= self.log_freq:
            self._last_log_timestep = self.num_timesteps
            self._log_sac_metrics()
        return True

    def _log_sac_metrics(self):
        """
        Computes and logs metrics relevant to SAC from the replay buffer.
        """
        # Ensure the model is an instance of SAC
        if not isinstance(self.model, SAC):
            if self.verbose > 0:
                logging.warning("SACMetricsLogger is designed for SAC models only.")
            return

        # Ensure the replay buffer has enough samples to form a batch
        buffer = self.model.replay_buffer
        if buffer.size() < self.model.batch_size:
            if self.verbose > 0:
                logging.warning("Replay buffer too small to sample for logging. Skipping metrics logging.")
            return

        # Sample a batch from the replay buffer
        data = buffer.sample(self.model.batch_size)

        # Move sampled data to the same device as the model (CPU or GPU)
        observations = data.observations.to(self.model.device)
        actions = data.actions.to(self.model.device)

        # Use torch.no_grad() as we are only evaluating and not performing gradient updates
        with th.no_grad():

            # SAC uses two critic (Q-value) networks to mitigate overestimation bias.
            # We evaluate both and take the minimum, as done in SAC's training.
            qf1_values, qf2_values = self.model.critic(observations, actions)
            q_values = th.min(qf1_values, qf2_values)
            self.logger.record("sac_metrics/q_value_mean", q_values.mean().item())
            self.logger.record("sac_metrics/q_value_std", q_values.std().item())

            # Get the actions and their log probabilities from the actor (policy) for the sampled observations
            _, log_prob_pi = self.model.actor.action_log_prob(observations)
            self.logger.record("sac_metrics/log_prob_mean", log_prob_pi.mean().item())
            self.logger.record("sac_metrics/log_prob_std", log_prob_pi.std().item())

            # SAC maximizes entropy, so logging it provides insight into exploration.
            # Entropy is often defined as the negative mean of log probabilities.
            policy_entropy = -log_prob_pi.mean()
            self.logger.record("sac_metrics/policy_entropy", policy_entropy.item())

            # Log the current size of the replay buffer to monitor its growth
            self.logger.record("sac_metrics/replay_buffer_size", buffer.size())
