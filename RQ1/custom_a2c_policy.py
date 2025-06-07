# custom_a2c_policy.py

from typing import Callable, List, Type

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TwoLayerLSTM(BaseFeaturesExtractor):
    """
    A FeaturesExtractor that takes (batch_size, input_dim) as input,
    unsqueezes to (batch_size, seq_len=1, input_dim), then applies two
    stacked LSTM layers (64 -> 32 units), and returns the final LSTM output
    at that single time‐step as a 32‐dim feature vector.

    This matches Zhang et al. (2019), Section 4.3:
      - Two‐layer LSTM (hidden sizes 64, then 32)
      - LeakyReLU activations after each LSTM layer
    """

    def __init__(
        self,
        observation_space,              # passed by SB3 (we ignore it here)
        input_size: int,
        lstm_hidden: List[int] = [64, 32],
    ):
        # We tell BaseFeaturesExtractor that the final feature dimension is lstm_hidden[-1] (=32).
        super(TwoLayerLSTM, self).__init__(observation_space=observation_space, features_dim=lstm_hidden[-1])

        self.lstm_hidden = lstm_hidden
        hidden_1, hidden_2 = lstm_hidden

        # First LSTM layer: input_size -> hidden_1
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            num_layers=1,
            batch_first=True,
        )
        self.activation1 = nn.LeakyReLU()

        # Second LSTM layer: hidden_1 -> hidden_2
        self.lstm2 = nn.LSTM(
            input_size=hidden_1,
            hidden_size=hidden_2,
            num_layers=1,
            batch_first=True,
        )
        self.activation2 = nn.LeakyReLU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        observations: either
          - shape [batch_size, input_dim], or
          - shape [batch_size, seq_len, input_dim].

        If 2D, we insert a dummy time‐dimension of length 1, so that LSTM sees
        shape = [batch_size, 1, input_dim].
        """
        if observations.dim() == 2:
            # Add a fake "time" dimension → [batch_size, 1, input_dim]
            obs_seq = observations.unsqueeze(1)
        else:
            obs_seq = observations  # Already 3D.

        # Pass through first LSTM: out1 has shape [batch, seq_len, hidden_1]
        out1, _ = self.lstm1(obs_seq)
        out1 = self.activation1(out1)

        # Pass through second LSTM: out2 has shape [batch, seq_len, hidden_2]
        out2, _ = self.lstm2(out1)
        out2 = self.activation2(out2)

        # We only need the final time‐step. If seq_len=1, that’s out2[:, 0, :].
        last_output = out2[:, -1, :]  # shape = [batch, hidden_2]
        return last_output


class A2C_LSTM_Policy(ActorCriticPolicy):
    """
    Custom A2C policy that uses TwoLayerLSTM as the features extractor,
    then applies separate actor/value heads *with no extra hidden layers*.
    By passing net_arch=[] and specifying features_extractor_class=TwoLayerLSTM,
    SB3 will build both the policy and value nets to accept a 32‐dim input.

    This fixes the “mat1 and mat2 shapes cannot be multiplied” error because
    SB3 now knows the LSTM produces 32 features and creates a Linear(32 → action_dim)
    and Linear(32 → 1) directly.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Callable[[float], float],
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        lstm_hidden: List[int] = [64, 32],
        **kwargs
    ):
        # First, compute input_dim = (n_market_features + n_state_features)
        # We can get that from observation_space.shape[-1]:
        input_dim = observation_space.shape[-1]

        # Now call the parent constructor, passing:
        #   - features_extractor_class = TwoLayerLSTM
        #   - features_extractor_kwargs = {"input_size": input_dim, "lstm_hidden": lstm_hidden}
        #   - net_arch = [] (so SB3 builds no additional hidden layers; the heads take 32 dims directly)
        super(A2C_LSTM_Policy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[],  # no extra hidden layers
            activation_fn=activation_fn,
            features_extractor_class=TwoLayerLSTM,
            features_extractor_kwargs={
                "input_size": input_dim,
                "lstm_hidden": lstm_hidden,
            },
            **kwargs,
        )

        # At this point, SB3 has already done:
        #   self.features_extractor = TwoLayerLSTM(observation_space, input_size=input_dim, lstm_hidden=lstm_hidden)
        # and built policy/value nets that accept a 32‐dim input.

