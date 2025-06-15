import torch
import numpy as np
from stable_baselines3.dqn import DQN
from stable_baselines3.common.type_aliases import GymStepReturn
from typing import Optional
from typing import Optional
from stable_baselines3.common.noise import ActionNoise

class BoltzmannDQN(DQN):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature


    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts:
            actions = np.array([self.action_space.sample() for _ in range(n_envs)])
            return actions, actions

        with torch.no_grad():
            q_values = self.q_net(torch.as_tensor(self._last_obs).to(self.device))
            q_values = q_values.cpu().numpy()

        # Boltzmann distribution
        logits = q_values / self.temperature
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for stability
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        actions = np.array([np.random.choice(len(p), p=p) for p in probs])
        return actions, actions
    
class MaxBoltzmannDQN(DQN):
    def __init__(self, *args, epsilon: float = 0.1, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.temperature = temperature

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts:
            actions = np.array([self.action_space.sample() for _ in range(n_envs)])
            return actions, actions

        with torch.no_grad():
            q_values = self.q_net(torch.as_tensor(self._last_obs).to(self.device))
            q_values = q_values.cpu().numpy()

        actions = []
        for q in q_values:
            if np.random.rand() < self.epsilon:
                # Softmax sampling (exploration)
                logits = q / self.temperature
                probs = np.exp(logits - np.max(logits))  # numerical stability
                probs = probs / np.sum(probs)
                a = np.random.choice(len(probs), p=probs)
            else:
                # Greedy (exploitation)
                a = np.argmax(q)
            actions.append(a)

        actions = np.array(actions)
        return actions, actions


