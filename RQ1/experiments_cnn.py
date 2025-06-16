from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Optional

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn

from RQ1.constants import N_ACTIONS, ACTION_LOW, ACTION_HIGH, INITIAL_CAPITAL, TRANSACTION_COST_PCT, FOREX_CANDLE_DATA, \
    SPLIT_RATIO, RQ1_EXPERIMENTS_DIR, EXPERIMENT_NAME_FORMAT, SAC_HYPERPARAMS, CUD_COLORS
from RQ1.scripts import train_eval_analyze
from common.data.feature_engineer import FeatureEngineer, copy_columns, complex_24h, complex_7d
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_current_exposure
from common.envs.forex_env import ActionConfig, DataConfig, EnvConfig, ObsConfig, ForexEnv
from common.models.train_eval import combine_finals


class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels, height, width)
        # Here, height is the number of features (e.g., OHLCV) and width is the lookback window
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # First convolutional layer
            # Input: (batch_size, 1, num_features, lookback_window)
            # Output: (batch_size, 32, num_features, lookback_window)
            nn.Conv2d(n_input_channels, 32, kernel_size=(1, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce the time dimension

            # Second convolutional layer
            # Input: (batch_size, 32, num_features, lookback_window/2)
            # Output: (batch_size, 64, num_features, lookback_window/2)
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce the time dimension again

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines the outputs of a CNN and a FlattenExtractor.
    It replicates the behavior of CombinedExtractor but allows for custom sub-extractors.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        # Define the output dimension of the CNN extractor
        cnn_output_dim = 64  # Must match the features_dim of your CustomCnnExtractor

        # Build the sub-extractors
        extractors = {
            "cnn_input": CustomCnnExtractor(observation_space.spaces["cnn_input"], features_dim=cnn_output_dim),
            "vector_input": FlattenExtractor(observation_space.spaces["vector_input"]),
        }

        # Compute the total feature dimension
        total_features_dim = cnn_output_dim + gym.spaces.flatdim(observation_space.spaces["vector_input"])


        # It's important to call the super constructor with the correct total feature dimension
        super().__init__(observation_space, features_dim=total_features_dim)

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Takes the dictionary of observations and passes each component to the
        correct extractor, then concatenates the results.
        """
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)

@dataclass(frozen=True)
class ExperimentConfig:

    name: str = field(default_factory=lambda: datetime.now().strftime(EXPERIMENT_NAME_FORMAT))
    line_color: str = "black"
    line_style: str = "-"
    line_marker: Optional[str] = None

    def get_style(self):
        return {
            "color" : self.line_color,
            "linestyle" : self.line_style,
            "marker": self.line_marker,
        }

def run_experiment(experiment_group: str, config: ExperimentConfig):

    # Setup feature engineers

    ohlcv_columns = ['volume', 'date_gmt', 'open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask']

    cnn_fe = FeatureEngineer()
    cnn_fe.add(partial(copy_columns, source_columns=ohlcv_columns, target_columns=[f"_{c}" for c in ohlcv_columns]))

    vector_fe = FeatureEngineer()
    vector_fe.add(complex_24h)
    vector_fe.add(complex_7d)

    vector_sfe = StepwiseFeatureEngineer()
    vector_sfe.add(["current_exposure"], calculate_current_exposure)

    # Setup environments

    obs_configs = [
        # Config for the CNN input
        ObsConfig(
            name='cnn_input',
            fe=cnn_fe,
            sfe=None, # Stepwise features shouldn't be windowed
            window=48
        ),
        # Config for the engineered vector input
        ObsConfig(
            name='vector_input',
            fe=vector_fe,
            sfe=vector_sfe,
            window=1
        )
    ]
    env_config = EnvConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        reward_function=None,
    )
    action_config = ActionConfig(
        n=N_ACTIONS,
        low=ACTION_LOW,
        high=ACTION_HIGH,
    )
    train_config, eval_config = DataConfig.from_splits(
        forex_candle_data=FOREX_CANDLE_DATA,
        split_pcts=[SPLIT_RATIO, 1 - SPLIT_RATIO],
        obs_configs=obs_configs,
    )
    train_env = ForexEnv(action_config, env_config, train_config)
    eval_env = ForexEnv(action_config, env_config, eval_config)

    # Setup model parameters

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        **SAC_HYPERPARAMS,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[64, 64], qf=[64, 64]),
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={}
        ),
        verbose=0,
        device="cuda"
    )

    # Run

    experiment_dir = RQ1_EXPERIMENTS_DIR / experiment_group / config.name
    train_eval_analyze(experiment_dir, model, train_env, eval_env)


if __name__ == '__main__':

    experiment_group = "network_cnn"
    experiments = [ExperimentConfig(
        name="test",
        line_color=CUD_COLORS[0],
    )]

    for experiment in experiments:
        run_experiment(experiment_group, experiment)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name: exp.get_style() for exp in experiments}, ext=".svg")

