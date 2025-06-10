import numpy as np

from common.constants import AgentDataCol
from common.envs.forex_env import ForexEnv


def equity_change(env: ForexEnv) -> float:
    """
    Calculate the difference between the equity just BEFORE making the current trade,
    and the equity just BEFORE making the next trade.
    """
    prev_equity = env.agent_data[env.n_steps, AgentDataCol.pre_action_equity]
    next_equity = env.agent_data[env.n_steps+1, AgentDataCol.pre_action_equity]
    return next_equity - prev_equity

def percentage_return(env: ForexEnv) -> float:
    """
    Measures the percentage change in equity.
    """
    prev_equity = env.agent_data[env.n_steps, AgentDataCol.pre_action_equity]
    if prev_equity == 0:
        return 0.0 # Avoid div by zero.
    next_equity = env.agent_data[env.n_steps+1, AgentDataCol.pre_action_equity]
    return (next_equity / prev_equity - 1.0) * 100.0

def log_equity_change(env: ForexEnv) -> float:
    """
    Calculate the log change in equity from the start to the end of the time period.
    """
    prev_equity = env.agent_data[env.n_steps, AgentDataCol.pre_action_equity]
    next_equity = env.agent_data[env.n_steps+1, AgentDataCol.pre_action_equity]
    if prev_equity <= 0:
        return 0.0  # Avoid log(x) with x <= 0 (undefined)
    return np.log(next_equity) - np.log(prev_equity)

def risk_adjusted_return(env: ForexEnv) -> float:
    """
    Calculate the risk-adjusted return based on the Sharpe ratio.
    """
    current_time_step = env.n_steps

    current_equity_change = equity_change(env)
    volatility = env.agent_data[current_time_step, AgentDataCol.equity_high] - env.agent_data[
        current_time_step, AgentDataCol.equity_low]
    epsilon = 1e-10  # Small value to avoid division by zero

    return current_equity_change / (volatility + epsilon)