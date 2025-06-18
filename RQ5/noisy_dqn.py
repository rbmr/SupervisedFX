import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NoisyMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = NoisyLinear(input_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.output = NoisyLinear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.output.reset_noise()

class NoisyDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.q_net.q_net[0].in_features
        output_dim = self.action_space.n
        self.q_net.q_net = NoisyMLP(input_dim, output_dim)
        self.q_net_target.q_net = NoisyMLP(input_dim, output_dim)

    def reset_noise(self):
        self.q_net.q_net.reset_noise()
        self.q_net_target.q_net.reset_noise()

class NoisyDQN(DQN):
    def __init__(self, *args, **kwargs):
        kwargs["policy"] = NoisyDQNPolicy
        kwargs["exploration_fraction"] = 0.0  # No epsilon needed
        kwargs["exploration_initial_eps"] = 0.0
        kwargs["exploration_final_eps"] = 0.0
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        self.policy.reset_noise()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        self.policy.reset_noise()
        return super().predict(observation, state, episode_start, deterministic)
