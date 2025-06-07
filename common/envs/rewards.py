import numpy as np

from common.constants import MarketDataCol
from common.envs.forex_env import ForexEnv, AgentDataCol

def equity_change(env: ForexEnv) -> float:
    """
    Calculate the change in equity from the start to the end of the episode.
    """
    current_time_step = env.n_steps
    current_close_equity = env.agent_data[current_time_step, AgentDataCol.equity_close]
    previous_close_equity = env.agent_data[current_time_step - 1, AgentDataCol.equity_close]
    return current_close_equity - previous_close_equity

def log_equity_change(env: ForexEnv) -> float:
    """
    Calculate the log change in equity from the start to the end of the episode.
    """
    current_time_step = env.n_steps
    current_close_equity = env.agent_data[current_time_step, AgentDataCol.equity_close]
    previous_close_equity = env.agent_data[current_time_step - 1, AgentDataCol.equity_close]

    if previous_close_equity <= 0:
        return 0.0  # Avoid log(0) or negative values

    return (current_close_equity / previous_close_equity) - 1.0


def risk_adjusted_return(env: ForexEnv) -> float:
    """
    Calculate the risk-adjusted return based on the Sharpe ratio.
    """
    current_time_step = env.n_steps

    current_equity_change = equity_change(env)
    volatility = env.agent_data[current_time_step, AgentDataCol.equity_high] - env.agent_data[
        current_time_step, AgentDataCol.equity_low]
    epsilon = 1e-5  # Small value to avoid division by zero

    return current_equity_change / (volatility + epsilon)