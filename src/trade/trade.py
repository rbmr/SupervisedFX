"""
Some scripts used for trading logic.
"""
import numpy as np
import tensorflow as tf

def action_to_order(prev_exposure: np.ndarray, target_exposure: np.ndarray, prev_cash: np.ndarray, prev_shares: np.ndarray, decision_bid: np.ndarray, decision_ask: np.ndarray, commission_pct: np.ndarray):
    """Performs the action to order derivation."""
    val_price = np.where(target_exposure >= 0, decision_bid, decision_ask)
    cost_per_share = np.where(target_exposure >= prev_exposure, (1 + commission_pct) * decision_ask, (1 - commission_pct) * decision_bid)
    tmp = val_price * (1 - target_exposure)
    return (target_exposure * prev_cash - prev_shares * tmp) / (tmp + target_exposure * cost_per_share)

@tf.function(jit_compile=True)
def tf_action_to_order(prev_exposure: tf.Tensor, target_exposure: tf.Tensor, prev_cash: tf.Tensor, prev_shares: tf.Tensor, decision_bid: tf.Tensor, decision_ask: tf.Tensor, commission_pct: tf.Tensor):
    """TensorFlow implementation for action to order derivation."""
    val_price = tf.where(target_exposure >= 0, decision_bid, decision_ask)
    cost_per_share = tf.where(target_exposure >= prev_exposure, (1 + commission_pct) * decision_ask, (1 - commission_pct) * decision_bid)
    tmp = val_price * (1 - target_exposure)
    return (target_exposure * prev_cash - prev_shares * tmp) / (tmp + target_exposure * cost_per_share)

def execute_trade(prev_cash: np.ndarray, prev_shares: np.ndarray, shares_to_trade: np.ndarray, exec_bid: np.ndarray, exec_ask: np.ndarray, commission_pct: np.ndarray):
    """Updates account state given a previous state, order and corresponding prices."""
    cost_per_share = np.where(shares_to_trade >= 0, (1 + commission_pct) * exec_ask, (1 - commission_pct) * exec_bid)
    return prev_cash - shares_to_trade * cost_per_share, prev_shares + shares_to_trade

@tf.function(jit_compile=True)
def tf_execute_trade(prev_cash: tf.Tensor, prev_shares: tf.Tensor, shares_to_trade: tf.Tensor, exec_bid: tf.Tensor, exec_ask: tf.Tensor, commission_pct: tf.Tensor):
    """TensorFlow implementation for executing a trade."""
    cost_per_share = tf.where(shares_to_trade >= 0, (1 + commission_pct) * exec_ask, (1 - commission_pct) * exec_bid)
    return prev_cash - shares_to_trade * cost_per_share, prev_shares + shares_to_trade

def state_price(cash: np.ndarray, shares: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the pval, equity, exposure and log-equity safely."""
    val_price = np.where(shares >= 0, bid, ask)
    pval = shares * val_price
    equity = pval + cash
    is_solvent = equity > 0  # non-bankrupt states
    exposure = np.zeros_like(equity)
    exposure[is_solvent] = pval[is_solvent] / equity[is_solvent]
    log_equity = np.full_like(equity, -np.inf)
    log_equity[is_solvent] = np.log(equity[is_solvent])
    return pval, equity, exposure, log_equity

@tf.function(jit_compile=True)
def tf_state_price(cash: tf.Tensor, shares: tf.Tensor, bid: tf.Tensor, ask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """TensorFlow implementation for calculating state-price derivatives."""
    val_price = tf.where(shares >= 0, bid, ask)
    pval = shares * val_price
    equity = pval + cash
    is_solvent = equity > 0

    safe_equity = tf.where(is_solvent, equity, 1.0)
    exposure = tf.where(is_solvent, pval / safe_equity, tf.zeros_like(equity))
    log_equity = tf.where(is_solvent, tf.math.log(safe_equity), tf.fill(tf.shape(equity), -np.inf))

    return pval, equity, exposure, log_equity

def norm_linspace_interp(x: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """O(1) version of np.interp(x, np.linspace(-1.0, 1.0, len(yp)), yp)"""
    n = len(yp)
    if n == 1:
        return np.zeros_like(x) + yp[0]

    # Convert x to float index [0, n-1]
    x_norm = np.clip(x, -1.0, 1.0) * 0.5 + 0.5
    idx = x_norm * (n - 1)

    # Find neighbours and interpolate
    idx0 = idx.astype(np.intp)
    idx1 = np.minimum(idx0 + 1, n - 1)
    a = idx - idx0
    y0 = yp[idx0]
    y1 = yp[idx1]
    return y0 + a * (y1 - y0)

@tf.function(jit_compile=True)
def tf_norm_linspace_interp(x: tf.Tensor, yp: tf.Tensor) -> tf.Tensor:
    """TensorFlow O(1) version of interpolation on a linspace."""
    yp = tf.cast(yp, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    n = tf.shape(yp)[-1]

    # Convert x to float index [0, n-1]
    x_norm = tf.clip_by_value(x, -1.0, 1.0) * 0.5 + 0.5
    idx = x_norm * tf.cast(n - 1, dtype=tf.float32)

    # Find neighbours and interpolate
    idx0 = tf.floor(idx)
    idx1 = tf.minimum(idx0 + 1, tf.cast(n - 1, dtype=tf.float32))
    a = idx - idx0

    idx0_int = tf.cast(idx0, dtype=tf.int32)
    idx1_int = tf.cast(idx1, dtype=tf.int32)

    batch_dims0 = idx0_int.shape.rank
    batch_dims1 = idx1_int.shape.rank
    y0 = tf.gather(yp, idx0_int, axis=-1, batch_dims=batch_dims0)
    y1 = tf.gather(yp, idx1_int, axis=-1, batch_dims=batch_dims1)

    return y0 + a * (y1 - y0)

def trade(
        prev_exposure: np.ndarray, target_exposure: np.ndarray, prev_cash: np.ndarray, prev_shares: np.ndarray,
        decision_bid: np.ndarray, decision_ask: np.ndarray, exec_bid: np.ndarray, exec_ask: np.ndarray,
        close_bid: np.ndarray, close_ask: np.ndarray, commission_pct: np.ndarray
    ):

    order_size = action_to_order(prev_exposure, target_exposure, prev_cash, prev_shares, decision_bid, decision_ask, commission_pct)
    cash, shares = execute_trade(prev_cash, prev_shares, order_size, exec_bid, exec_ask, commission_pct)
    pval, equity, exposure, log_equity = state_price(cash, shares, close_bid, close_ask)

    return cash, shares, pval, equity, exposure, log_equity
