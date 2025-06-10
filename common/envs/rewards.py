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

def get_tdigest_delta(data: np.ndarray) -> float:
    """
    Tries some delta values with default k to determine the optimal delta using an example dataset.
    """
    logging.info("Starting TDigest delta evaluation")

    target_percentiles = [0.1, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]

    logging.info("Determining ground truth")
    sorted_data = np.sort(data)
    ground_truth = {p: np.percentile(sorted_data, p) for p in target_percentiles}

    logging.info("Determining estimates")
    delta_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_delta = None
    best_avg_rel_error = float("inf")

    for delta in delta_values:

        digest = TDigest(delta=delta)
        digest.batch_update(data)

        current_errors = []
        for p in target_percentiles:
            estimated_value = digest.percentile(p)
            exact_value = ground_truth[p]
            abs_error = abs(estimated_value - exact_value)
            rel_error = abs_error / abs(exact_value) if exact_value != 0 else abs_error
            current_errors.append(rel_error)
        avg_rel_error = np.mean(current_errors)
        num_centroids = len(digest.centroids_to_list())

        logging.info(f"  Delta: {delta:.4f}, Avg Rel Error: {avg_rel_error:.6f}, Centroids: {num_centroids}")

        if avg_rel_error < best_avg_rel_error:
            best_avg_rel_error = avg_rel_error
            best_delta = delta

    return best_delta

def best_empirical_tdigest_delta(env: ForexEnv) -> float:
    rewards = empirical_rewards(env)
    delta = get_tdigest_delta(rewards)
    logging.info(f"Best delta: {delta}")
    return delta

class Normalizer(ABC):

    @abstractmethod
    def normalize(self, x: float) -> float: ...

class TDigestNormalizer(Normalizer):

    def __init__(self, digest: TDigest):
        self.digest = digest

    def normalize(self, x: float) -> float:
        return self.digest.cdf(x) * 2 - 1

class ArctanNormalizer(Normalizer):
    """
    Maps values by applying arctan and multiplying by a factor.
    """

    def __init__(self, f: float) -> None:
        self.factor = f

    def normalize(self, x: float) -> float:
        return self.factor * np.arctan(x)

class SubDivNormalizer(Normalizer):

    def __init__(self, sub: float, div: float):
        self.sub = sub
        self.div = div

    def normalize(self, x: float) -> float:
        return (x - self.sub) / self.div

class RunningZScoreNormalizer(Normalizer):

    def __init__(self):
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.count: int = 0

    def normalize(self, x: float) -> float:
        """
        Uses Welford's algorithm to normalize the value using the running mean and variance.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        return (x - self.mean) / self.std()

    def std(self):
        var = self.M2 / (self.count - 1) if self.count > 1 else 1.0
        return math.sqrt(var)

class DPRewardFunction:

    def __init__(self, table: DPTable, normalizer: Normalizer | None = None):
        # Reward computation
        self.v = table.value_table
        self.pi = table.policy_table
        self.q_min = table.q_min_table
        self.n_actions = table.n_actions
        self.actions = get_exposure_levels(table.n_actions)
        self.T = self.v.shape[0]
        self.c = table.transaction_cost_pct
        self.normalizer = normalizer
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
