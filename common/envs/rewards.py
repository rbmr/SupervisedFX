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

def volatility_scaled_reward(env,
                             vol_target: float = 0.10,
                             bp: float = 0.0020,
                             window: int = 60,
                             lam: float = 0.94) -> float:
    """
    Custom reward function that replicates Equation (4) in Zhang et al. (2019).

    Reward formula (for t >= window and t >= 2):
      R_t = (σ_tgt / σ_{t-1}) * [ A_{t-1} * (p_t - p_{t-1})  -  bp * p_{t-1} * |A_{t-1} - A_{t-2}| ]

    Parameters:
    - env: instance of ForexEnv, already stepped so that current_step = t.
    - vol_target: target volatility σ_tgt (e.g. 0.10 for 10%).
    - bp: transaction cost rate (e.g. 0.0020 for 0.20%).
    - window: lookback window (in bars) for ex‐ante EWMA volatility (e.g. 60).
    - lam: EWMA decay factor (e.g. 0.94).

    Returns:
    - A single float reward for time t. If t < max(window, 2), returns 0.0.
    """

    t = env.current_step

    # Need at least window bars to estimate σ_{t-1}, and at least 2 steps to have A_{t-2}.
    if t < window or t < 2:
        return 0.0

    # --- 1) Extract A_{t-1} and A_{t-2} from agent_data ---
    # Note: agent_data was just written at index t, so
    #   agent_data[t-1, AgentDataCol.action] is A_{t-1],
    #   agent_data[t-2, AgentDataCol.action] is A_{t-2].
    A_tm1 = float(env.agent_data[t - 1, AgentDataCol.action])
    A_tm2 = float(env.agent_data[t - 2, AgentDataCol.action])

    # --- 2) Extract p_t and p_{t-1} from market_data (use close_bid) ---
    close_bid_t = float(env.market_data[t, MarketDataCol.close_bid])
    close_bid_tm1 = float(env.market_data[t - 1, MarketDataCol.close_bid])
    r_t = close_bid_t - close_bid_tm1

    # --- 3) Compute ex‐ante EWMA volatility σ_{t-1} using last `window` close_bid bars ---
    # We want returns over [t-window, …, t-1], so prices[i+1] - prices[i] for i = t-window .. t-2
    start_idx = t - window
    end_idx = t  # exclusive, so market_data[start_idx : end_idx] has length = window
    recent_prices = env.market_data[start_idx:end_idx, MarketDataCol.close_bid]  # shape = (window,)

    # Compute simple arithmetic returns of length (window-1)
    recent_returns = recent_prices[1:] - recent_prices[:-1]  # shape = (window-1,)

    # Build EWMA weights of length (window-1): w_i ∝ (1-lam) * lam^(i), with i = 0 for most recent.
    # We want weights such that the most recent return (i = window-2) gets the largest lam^0.
    # So build it reversed: index 0 corresponds to t-2 (most recent), index window-2 corresponds to t-window.
    L = len(recent_returns)  # = window - 1
    raw_weights = np.array([(1 - lam) * (lam ** i) for i in range(L)])
    # raw_weights[0] = (1-lam)*lam^0 corresponds to oldest return, so reverse:
    ewma_weights = raw_weights[::-1]
    ewma_weights = ewma_weights / ewma_weights.sum()

    # σ_{t-1} is the sqrt of weighted average of squared returns:
    sigma_tm1 = float(np.sqrt(np.dot(ewma_weights, recent_returns ** 2)))
    if sigma_tm1 < 1e-12:
        # If volatility is essentially zero (flat market), we give zero reward to avoid blow‐ups.
        return 0.0

    # --- 4) Compute volatility‐scaling factor ---
    vol_scaling = vol_target / sigma_tm1

    # --- 5) Compute transaction cost term: bp * p_{t-1} * |A_{t-1} - A_{t-2}| ---
    trans_cost = bp * close_bid_tm1 * abs(A_tm1 - A_tm2)

    # --- 6) Compute final reward ---
    # R_t = vol_scaling * [ A_{t-1} * r_t  -  trans_cost ]
    reward = vol_scaling * (A_tm1 * r_t - trans_cost)
    return float(reward)