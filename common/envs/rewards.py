import logging
import math
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tdigest import TDigest

from common.constants import AgentDataCol
from common.envs.dp import interp, get_exposure_levels, DPTable
from common.envs.forex_env import ForexEnv
from common.models.dummy_models import hold_model, DummyModelFactory
from common.models.train_eval import run_model

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
        models = [hold_model]

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

class DPRewardFunction:

    def __init__(self, table: DPTable):
        # Reward computation
        self.v = table.value_table
        self.pi = table.policy_table
        self.q_min = table.q_min_table
        self.n_actions = table.n_actions
        self.actions = get_exposure_levels(table.n_actions)
        self.T = self.v.shape[0]
        self.c = table.transaction_cost_pct
        step_diffs = (self.v - self.q_min)[:-1].flatten() # Exclude terminal state, and flatten
        self.max_diff = np.percentile(step_diffs, 98)
        if self.max_diff <= 1e-9:
            raise ValueError("Unexpected table values, step importance is almost non-existent.")

    def __call__(self, env) -> float:
        t = env.n_steps
        if t >= self.T - 1:
            return 0.0

        # Unwrap env.agent_data for exposures and equities
        curr_cash = env.agent_data[t-1, AgentDataCol.cash]
        curr_equity = env.agent_data[t, AgentDataCol.pre_action_equity]
        curr_exposure = (curr_equity - curr_cash) / curr_equity

        next_cash = env.agent_data[t, AgentDataCol.cash]
        next_equity = env.agent_data[t+1, AgentDataCol.pre_action_equity]
        next_exposure = (next_equity - next_cash) / next_equity

        # True one-step log-return
        true_log_return = np.log(next_equity / curr_equity)

        # Interpolated DP expected cumulative log-equity from current/next state
        exp_log_next_optimal = interp(self.v[t+1], next_exposure, self.n_actions)
        exp_log_worst = interp(self.q_min[t], curr_exposure, self.n_actions)

        # How much was the action better than the worst action you could have taken?
        # and how significant was this difference? (Normalize)
        r = (true_log_return + exp_log_next_optimal - exp_log_worst) / self.max_diff

        # Clip and adjust
        return np.clip(r, 0.0, 1.0)
