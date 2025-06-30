"""
Some scripts used for trading logic.
"""
import tensorflow as tf

from src.constants import MarketDataCol


@tf.function
def reverse_equity(bid_price: tf.Tensor, ask_price: tf.Tensor, equity: tf.Tensor, exposure: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Given equity, bid/ask prices, and exposure in [-1, 1],
    Returns (cash, shares) such that:
    - price == bid_price if shares >= 0 else ask_price
    - equity == cash + shares * price
    - exposure == shares * price / equity == (equity - cash) / equity
    """
    tf.Assert(tf.reduce_all((exposure >= -1.0) & (exposure <= 1.0)), ["Exposure must be in [-1, 1]"])
    price = tf.where(exposure >= 0, bid_price, ask_price)
    cash = equity * (1.0 - exposure)
    shares = tf.math.divide_no_nan(equity * exposure, price)
    return cash, shares

@tf.function
def calculate_equity(bid_price: tf.Tensor, ask_price: tf.Tensor, cash: tf.Tensor, shares: tf.Tensor) -> tf.Tensor:
    """
    Calculates equity (mark-to-market) based on current cash, shares and prices.
    """
    price = tf.where(shares >= 0, bid_price, ask_price)
    return cash + shares * price

@tf.function
def execute_trade(target_exposure: tf.Tensor, bid_price: tf.Tensor, ask_price: tf.Tensor, current_cash: tf.Tensor, current_shares: tf.Tensor, transaction_cost_pct: float) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Execute trade to achieve target exposure.
    Returns (new_cash, new_shares).
    """
    # Calculate current equity
    current_equity = calculate_equity(bid_price, ask_price, current_cash, current_shares)
    tf.Assert(tf.reduce_all(current_equity > 0), ["current_equity should be greater than zero."])

    # Determine target position value in currency
    target_position_value = target_exposure * current_equity

    # Determine target number of shares based on target value
    price_for_target = tf.where(target_exposure >= 0, ask_price, bid_price)
    target_num_shares = tf.math.divide_no_nan(target_position_value, price_for_target)

    # Determine the change in shares needed
    shares_to_trade = target_num_shares - current_shares

    # Vectorized Buy
    cost_per_share = ask_price * (1.0 + transaction_cost_pct)
    shares_bought = tf.maximum(0.0, shares_to_trade)
    cash_after_buy = current_cash - shares_bought * cost_per_share
    shares_after_buy = current_shares + shares_bought

    # Vectorized Sell
    proceeds_per_share = bid_price * (1.0 - transaction_cost_pct)
    shares_sold = tf.maximum(0.0, -shares_to_trade) # shares_to_sell is positive
    cash_after_sell = cash_after_buy + shares_sold * proceeds_per_share
    shares_after_sell = shares_after_buy - shares_sold

    return cash_after_sell, shares_after_sell

@tf.function
def execute_trade_1equity(
        indices: tf.Tensor,
        current_exposures: tf.Tensor,
        target_exposures: tf.Tensor,
        market_data: tf.Tensor,
        transaction_cost_pct: float
    ) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Execute trades for a batch of current_exposures and target_exposures.
    """
    batch_size = tf.shape(indices)[0]
    indices_flat = tf.reshape(indices, [-1])

    curr_prices = tf.gather(market_data, indices_flat)
    next_prices = tf.gather(market_data, indices_flat + 1)

    curr_bid = tf.reshape(curr_prices[:, MarketDataCol.open_bid], [batch_size, 1])
    curr_ask = tf.reshape(curr_prices[:, MarketDataCol.open_ask], [batch_size, 1])
    next_bid = tf.reshape(next_prices[:, MarketDataCol.open_bid], [batch_size, 1])
    next_ask = tf.reshape(next_prices[:, MarketDataCol.open_ask], [batch_size, 1])

    curr_cash, curr_shares = reverse_equity(curr_bid, curr_ask, tf.ones_like(current_exposures), current_exposures)
    next_cash, next_shares = execute_trade(target_exposures, curr_bid, curr_ask, curr_cash, curr_shares, transaction_cost_pct)
    next_equity = calculate_equity(next_bid, next_ask, next_cash, next_shares)
    next_exposure = tf.math.divide_no_nan(next_equity - next_cash, next_equity)

    return next_equity, next_exposure