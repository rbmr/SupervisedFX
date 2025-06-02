import numpy as np

from common.constants import AgentDataCol
from common.envs.forex_env import ForexEnv

def log_equity_diff(env: ForexEnv) -> float:
    current_equity = env.agent_data[env.current_step, AgentDataCol.equity_close]
    prev_equity = env.agent_data[env.current_step - 1, AgentDataCol.equity_close]
    # we assume prev_equity is always > 0, since episodes ends if it goes below zero.
    # however, current_equity may be below zero because .step() does not immediately exit.
    if current_equity <= 0:
        return 0.0
    return np.log(current_equity) - np.log(prev_equity)

def equity_diff(env: ForexEnv) -> float:
    current_equity = env.agent_data[env.current_step, AgentDataCol.equity_close]
    prev_equity = env.agent_data[env.current_step - 1, AgentDataCol.equity_close]
    return current_equity - prev_equity