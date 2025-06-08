import numpy as np

from common.constants import MarketDataCol
from common.envs.forex_env import AgentDataCol, ForexEnv
from common.envs.trade import calculate_equity

def get_curr_equity_open(env: ForexEnv):
    """Get the equity immediately after making the current trade."""
    return env.agent_data[env.n_steps, AgentDataCol.equity_open]

def get_next_equity_open(env: ForexEnv):
    """Get the equity just before making the next trade."""
    cash = env.agent_data[env.n_steps, AgentDataCol.cash]
    shares = env.agent_data[env.n_steps, AgentDataCol.shares]
    next_bid_price = env.market_data[env.n_steps + 1, MarketDataCol.open_bid]
    next_ask_price = env.market_data[env.n_steps + 1, MarketDataCol.open_ask]
    next_equity_open = calculate_equity(next_bid_price, next_ask_price, cash, shares)  # type: ignore
    return next_equity_open

def equity_change(env: ForexEnv) -> float:
    """
    Calculate the difference between the equity immediately AFTER making the current trade,
    and the equity just BEFORE making the next trade.
    """
    return get_next_equity_open(env) - get_curr_equity_open(env)

def percentage_return(env: ForexEnv) -> float:
    """
    Measures the percentage change in equity.
    """
    curr_equity = get_curr_equity_open(env)
    if curr_equity == 0:
        return 0.0 # Avoid div by zero.
    
    reward = get_next_equity_open(env) / curr_equity - 1.0
    # scale to be a full percentage
    reward *= 100.0
    return reward

def log_equity_change(env: ForexEnv) -> float:
    """
    Calculate the log change in equity from the start to the end of the time period.
    """
    curr_equity = get_curr_equity_open(env)
    if curr_equity <= 0:
        return 0.0  # Avoid log(x) with x <= 0 (undefined)
    return np.log(get_next_equity_open(env) / curr_equity)

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