from RQ1.constants import N_ACTIONS, ACTION_LOW, ACTION_HIGH
from RQ1.parameters import get_data
from common.envs.forex_env import ActionConfig, DataConfig


def get_envs():
    obs_configs = [
        # Config for the CNN input
        ObsConfig(
            name='cnn_input',
            fe=cnn_feature_engineer,
            sfe=None, # Stepwise features shouldn't be windowed
            window=cnn_window_size
        ),
        # Config for the engineered vector input
        ObsConfig(
            name='vector_input',
            fe=vector_feature_engineer,
            sfe=agent_feature_engineer,
            window=1
        )
    ]
    env_config = EnvConfig(
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        shuffled=shuffled,
        reward_function=custom_reward_function,
    )
    action_config = ActionConfig(
        n=N_ACTIONS,
        low=ACTION_LOW,
        high=ACTION_HIGH,
    )
    train_config, eval_config = DataConfig.get_configs(
        forex_candle_data=get_data(),
        split_ratio=0.7,
        obs_configs=obs_configs,
    )
    train_env = ForexEnv(action_config, env_config, train_config)
    eval_env = ForexEnv(action_config, env_config, eval_config)
    return train_env, eval_env
