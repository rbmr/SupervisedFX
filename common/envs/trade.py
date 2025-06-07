"""
Some scripts used for trading logic.
"""
import numpy as np
from numpy.typing import NDArray

from common.constants import MarketDataCol


def reverse_equity(bid_price: float, ask_price: float, equity: float, exposure: float) -> tuple[float, float]:
    """
    Given equity, bid/ask prices, and exposure in [-1, 1],
    Returns (cash, shares) such that:
    - price == bid_price if shares >= 0 else ask_price
    - equity == cash + shares * price
    - exposure == shares * price / equity == (equity - cash) / equity
    """
    assert -1.0 <= exposure <= 1.0, "Exposure must be between -1 and 1"
    price = bid_price if exposure >= 0 else ask_price
    cash   = equity * (1 - exposure)
    shares = (equity * exposure) / price
    return cash, shares

def calculate_equity(bid_price: float, ask_price: float, cash: float, shares: float) -> float:
    """
    Calculates equity (mark-to-market) based on current cash, shares and prices.
    """
    return cash + shares * (bid_price if shares >= 0 else ask_price)

def calculate_ohlc_equity(current_prices: NDArray[np.float32], cash: float, shares: float) -> tuple[float, float, float, float]:
    """
    Calculates the equity based on current cash, shares and prices.
    """
    assert current_prices.ndim == 1
    assert current_prices.shape[0] == len(MarketDataCol)
    equity_open = calculate_equity(current_prices[MarketDataCol.open_bid], current_prices[MarketDataCol.open_ask], cash, shares) # type: ignore
    equity_high = calculate_equity(current_prices[MarketDataCol.high_bid], current_prices[MarketDataCol.high_ask], cash, shares) # type: ignore
    equity_low = calculate_equity(current_prices[MarketDataCol.low_bid], current_prices[MarketDataCol.low_ask], cash, shares) # type: ignore
    equity_close = calculate_equity(current_prices[MarketDataCol.close_bid], current_prices[MarketDataCol.close_ask], cash, shares) # type: ignore
    return equity_open, equity_high, equity_low, equity_close

def execute_trade(target_exposure: float, current_data, current_cash: float, current_shares: float, transaction_cost_pct: float) -> tuple[float, float]:
    """
    Execute trade to achieve target exposure.
    Returns (new_cash, new_shares).
    """
    # Calculate current equity
    open_bid = current_data[MarketDataCol.open_bid]
    open_ask = current_data[MarketDataCol.open_ask]
    current_equity = calculate_equity(open_bid, open_ask, current_cash, current_shares)
    assert current_equity > 0, f"current_equity should be greater than zero, was {current_equity:.2f}"

    # Jitter Mitigation, skip action if it has no significant effect.
    current_exposure = (current_equity - current_cash) / current_equity
    if abs(target_exposure - current_exposure) < 1e-5:
        return current_cash, current_shares

    # Determine target position value in currency
    target_position_value = target_exposure * current_equity

    # Determine target number of shares based on target value
    if target_exposure > 0:  # Target is LONG
        target_num_shares = target_position_value / open_ask
    else:  # Target is SHORT
        target_num_shares = target_position_value / open_bid

    # Determine the change in shares needed
    shares_to_trade = target_num_shares - current_shares

    if shares_to_trade > 0:  # Need to buy
        new_cash, new_shares = buy_shares(
            shares_to_buy=shares_to_trade,
            ask_price=open_ask,
            current_cash=current_cash,
            current_shares=current_shares,
            transaction_cost_pct=transaction_cost_pct
        )
    else:  # Need to sell
        new_cash, new_shares = sell_shares(
            shares_to_sell=-shares_to_trade,
            bid_price=open_bid,
            ask_price=open_ask,  # needed to determine max shares to sell
            current_cash=current_cash,
            current_shares=current_shares,
            transaction_cost_pct=transaction_cost_pct
        )

    return new_cash, new_shares

def buy_shares(shares_to_buy: float,
               ask_price: float,
               current_cash: float, current_shares: float,
               transaction_cost_pct: float) -> tuple[float, float]:
    """
    Executes a buy order for a specified absolute number of shares.
    """
    assert ask_price > 1e-6, f"Attempted to buy with invalid ask_price: {ask_price}"

    # Determine shares to sell such that we don't go above 100% leverage.
    cost_per_share = ask_price * (1 + transaction_cost_pct)
    max_shares_to_buy = current_cash / cost_per_share
    shares_bought = min(shares_to_buy, max_shares_to_buy)

    # Compute updated portfolio state
    updated_cash = current_cash - shares_bought * cost_per_share
    updated_shares = current_shares + shares_bought
    return updated_cash, updated_shares

def sell_shares(shares_to_sell: float,
                bid_price: float, ask_price: float,
                current_cash: float, current_shares: float,
                transaction_cost_pct: float) -> tuple[float, float]:
    """
    Executes a sell order for a specified absolute number of shares.
    """
    assert ask_price > 1e-6, f"Attempted to sell with invalid ask_price: {ask_price}"
    assert bid_price > 1e-6, f"Attempted to sell with invalid bid_price: {bid_price}"

    # Determine shares to sell such that we don't go below -100% leverage.
    proceeds_per_share = bid_price * (1 - transaction_cost_pct)
    max_shares_to_sell = (current_cash + 2 * current_shares * ask_price) / (2 * ask_price - proceeds_per_share)
    shares_sold = min(shares_to_sell, max_shares_to_sell)

    # Compute updated portfolio state
    updated_cash = current_cash + shares_sold * proceeds_per_share
    updated_shares = current_shares - shares_sold
    return updated_cash, updated_shares