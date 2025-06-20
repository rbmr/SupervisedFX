import logging

import numpy as np

from common.constants import AgentDataCol
from common.envs.dp import interp, get_bins, DPTable
from common.envs.forex_env import ForexEnv
from common.models.dummy_models import cash_model, DummyModelFactory
from common.models.train_eval import run_model
from common.scripts import to_percentiles


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

def empirical_rewards(env: ForexEnv, models: list[DummyModelFactory] | None = None) -> np.ndarray:
    """
    Runs random model on the forex_env for a single episode (or some specified duration) to collect rewards.
    """

    logging.info("Starting reward evaluation.")

    if models is None:
        models = [cash_model]

    all_rewards = np.zeros((0,), dtype=np.int32)

    for dummy_model in models:

        logging.info(f"Evaluating model: {dummy_model.__name__}")
        model = dummy_model(env)

        logs_df = run_model(
            model=model,
            env=env,
            data_path=None,
            total_steps=env.episode_len,
            deterministic=False,
            progress_bar=True,
        )
        env.reset()  # Make sure environment is not impacted

        rewards = logs_df["reward"][1:]
        all_rewards = np.concatenate((all_rewards, rewards))

    logging.info(f"Finished reward evaluation. Collected {len(all_rewards)} rewards.")

    return all_rewards

class DPRewardFunctionC:
    def __init__(self, table: DPTable):
        self.v = table.value_table
        self.q_min = table.q_min_table
        self.n_exposures = table.n_exposures
        self.T = self.v.shape[0]
        importance = self.v - self.q_min
        importance = importance[importance > 1e-9]
        self.importance = to_percentiles(importance)

    def __call__(self, env) -> float:
        t = env.n_steps
        if t >= self.T - 1:
            return 0.0

        curr_cash = env.agent_data[t-1, AgentDataCol.cash]
        curr_equity = env.agent_data[t, AgentDataCol.pre_action_equity]
        curr_exposure = (curr_equity - curr_cash) / curr_equity

        v_current = interp(self.v[t], curr_exposure, self.n_exposures)
        q_min_current = interp(self.q_min[t], curr_exposure, self.n_exposures)

        raw_importance = v_current - q_min_current
        if raw_importance < 1e-9:
            return 0.0

        next_cash = env.agent_data[t, AgentDataCol.cash]
        next_equity = env.agent_data[t+1, AgentDataCol.pre_action_equity]
        next_exposure = (next_equity - next_cash) / next_equity

        true_log_return = np.log(next_equity / curr_equity)
        v_next = interp(self.v[t+1], next_exposure, self.n_exposures)
        q_taken = true_log_return + v_next

        goodness = ((q_taken - q_min_current) / raw_importance) * 2 - 1
        importance = interp(self.importance[t], curr_exposure, self.n_exposures)

        r = importance * goodness

        return np.clip(r, -1.0, 1.0)

class DPRewardFunctionB:
    def __init__(self, table: DPTable):
        self.v = table.value_table
        self.q_min = table.q_min_table
        self.n_exposures = table.n_exposures
        self.T = self.v.shape[0]
        importance = self.v - self.q_min
        importance = importance[importance > 1e-9]
        self.norm = 1 / np.percentile(importance, 99) if len(importance) > 0 else 1.0

    def __call__(self, env) -> float:
        t = env.n_steps
        if t >= self.T - 1:
            return 0.0

        curr_cash = env.agent_data[t-1, AgentDataCol.cash]
        curr_equity = env.agent_data[t, AgentDataCol.pre_action_equity]
        curr_exposure = (curr_equity - curr_cash) / curr_equity

        next_cash = env.agent_data[t, AgentDataCol.cash]
        next_equity = env.agent_data[t+1, AgentDataCol.pre_action_equity]
        next_exposure = (next_equity - next_cash) / next_equity

        true_log_return = np.log(next_equity / curr_equity)

        v_current = interp(self.v[t], curr_exposure, self.n_exposures)
        v_next = interp(self.v[t+1], next_exposure, self.n_exposures)

        r = (true_log_return + v_next - v_current) * self.norm

        return np.clip(r+1, 0.0, 1.0)

class DPRewardFunction:

    def __init__(self, table: DPTable):
        # Reward computation
        self.v = table.value_table
        self.q_min = table.q_min_table
        self.n_exposures = table.n_exposures
        self.T = self.v.shape[0]
        importance = self.v - self.q_min
        importance = importance[importance > 1e-9]
        self.norm = np.percentile(importance, 99) if len(importance) > 0 else 1.0

    def __call__(self, env) -> float:
        t = env.n_steps
        if t >= self.T - 1:
            return 0.0

        curr_cash = env.agent_data[t-1, AgentDataCol.cash]
        curr_equity = env.agent_data[t, AgentDataCol.pre_action_equity]
        curr_exposure = (curr_equity - curr_cash) / curr_equity

        next_cash = env.agent_data[t, AgentDataCol.cash]
        next_equity = env.agent_data[t+1, AgentDataCol.pre_action_equity]
        next_exposure = (next_equity - next_cash) / next_equity

        true_log_return = np.log(next_equity / curr_equity)

        v_next = interp(self.v[t+1], next_exposure, self.n_exposures)
        q_min_current = interp(self.q_min[t], curr_exposure, self.n_exposures)

        r = (true_log_return + v_next - q_min_current) / self.norm

        return np.clip(r, 0.0, 1.0)
